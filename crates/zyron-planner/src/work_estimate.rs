//! Estimated work in seconds for a physical plan.
//!
//! The optimizer's own cost is in abstract units, which is right for comparing
//! two plans for the same query and useless for anything else: units do not
//! compare across queries and they cannot be measured against a latency
//! objective.
//!
//! This is the other number. Each operator's row estimate is multiplied by what
//! a row of that operator actually costs on this machine, measured from the
//! real work the node has been doing, and the tree is summed. The result is in
//! seconds, which is the same unit the objectives are written in, so admission
//! can compare them without a conversion constant.
//!
//! Being an estimate is fine and being wrong is recoverable: every completion
//! records what the query actually cost against what it was predicted to cost,
//! and the ratio is published as the calibration error. A cost model that lies
//! says so.

use zyron_catalog::TableId;
use zyron_common::checksum::hash_combine;
use zyron_pressure::capability::{OperatorCoefficients, OperatorKind};

use crate::physical::PhysicalPlan;

/// Rows a node is assumed to handle when the optimizer produced no estimate.
///
/// Deliberately not zero. A plan with no statistics behind it is unknown, not
/// free, and treating it as free would let an unbounded query bypass admission
/// as though it were a point lookup.
const UNKNOWN_ROWS: f64 = 1_000.0;

/// Where the shape hash starts, so an empty walk is still a distinct value
/// rather than zero.
const SHAPE_SEED: u64 = 0x5A79_726F_6E50_4C4E;

/// The work unit each plan node is priced as.
///
/// Nodes that do no measurable work of their own return None and contribute
/// only what their children cost. A Gather is a channel, a Broadcast is a
/// send: the work is in the operators feeding them.
pub fn operator_kind_of(plan: &PhysicalPlan) -> Option<OperatorKind> {
    match plan {
        PhysicalPlan::SeqScan { .. }
        | PhysicalPlan::HybridScan { .. }
        | PhysicalPlan::ParallelSeqScan { .. }
        | PhysicalPlan::ColumnarMetadataAggregate { .. } => Some(OperatorKind::SeqScan),
        PhysicalPlan::LakeScan { .. }
        | PhysicalPlan::LakeDelete { .. }
        | PhysicalPlan::LakeUpdate { .. }
        | PhysicalPlan::LakeMetadataAggregate { .. } => Some(OperatorKind::LakeScan),
        PhysicalPlan::IndexScan { .. }
        | PhysicalPlan::FulltextScan { .. }
        | PhysicalPlan::VectorScan { .. }
        | PhysicalPlan::SpatialScan { .. } => Some(OperatorKind::IndexScan),
        // A foreign scan's time is a peer's, not this node's, but it still
        // occupies the query for that long, so it is priced as a scan here
        PhysicalPlan::ForeignScan { .. } => Some(OperatorKind::SeqScan),
        PhysicalPlan::Filter { .. } | PhysicalPlan::LockRows { .. } => Some(OperatorKind::Filter),
        // A view trigger write shapes source rows into per-row parameters;
        // the trigger bodies it fires are priced as their own statements
        PhysicalPlan::Project { .. }
        | PhysicalPlan::Values { .. }
        | PhysicalPlan::ViewTriggerWrite { .. } => Some(OperatorKind::Project),
        PhysicalPlan::HashJoin { .. } | PhysicalPlan::ParallelHashJoin { .. } => {
            Some(OperatorKind::HashJoinProbe)
        }
        PhysicalPlan::MergeJoin { .. }
        | PhysicalPlan::NestedLoopJoin { .. }
        | PhysicalPlan::LateralJoin { .. } => Some(OperatorKind::HashJoinBuild),
        PhysicalPlan::HashAggregate { .. }
        | PhysicalPlan::SortAggregate { .. }
        | PhysicalPlan::GapFill { .. }
        | PhysicalPlan::AnalyticsTableFunction { .. }
        | PhysicalPlan::GraphAlgorithm { .. } => Some(OperatorKind::Aggregate),
        PhysicalPlan::Sort { .. } | PhysicalPlan::HashDistinct { .. } => Some(OperatorKind::Sort),
        PhysicalPlan::Window { .. } => Some(OperatorKind::Window),
        PhysicalPlan::SetOp { .. } => Some(OperatorKind::SetOp),
        // Writes pay for the durable record they produce, not for a row of
        // compute, so they are priced per page of WAL rather than per row
        PhysicalPlan::Insert { .. } | PhysicalPlan::Update { .. } | PhysicalPlan::Delete { .. } => {
            Some(OperatorKind::WalFsync)
        }
        // Plumbing: moves rows, does not transform them
        PhysicalPlan::Limit { .. }
        | PhysicalPlan::Gather { .. }
        | PhysicalPlan::Repartition { .. }
        | PhysicalPlan::Broadcast { .. } => None,
    }
}

/// The relation a node reads or writes, where it names one.
///
/// Part of the shape hash because two plans of identical structure over
/// different tables are different work: a survivor prefetching for the wrong
/// table has warmed the wrong pages.
fn relation_of(plan: &PhysicalPlan) -> Option<TableId> {
    match plan {
        PhysicalPlan::SeqScan { table_id, .. }
        | PhysicalPlan::HybridScan { table_id, .. }
        | PhysicalPlan::ParallelSeqScan { table_id, .. }
        | PhysicalPlan::ColumnarMetadataAggregate { table_id, .. }
        | PhysicalPlan::LakeScan { table_id, .. }
        | PhysicalPlan::LakeDelete { table_id, .. }
        | PhysicalPlan::LakeUpdate { table_id, .. }
        | PhysicalPlan::LakeMetadataAggregate { table_id, .. }
        | PhysicalPlan::ForeignScan { table_id, .. }
        | PhysicalPlan::IndexScan { table_id, .. }
        | PhysicalPlan::FulltextScan { table_id, .. }
        | PhysicalPlan::VectorScan { table_id, .. }
        | PhysicalPlan::SpatialScan { table_id, .. }
        | PhysicalPlan::LockRows { table_id, .. }
        | PhysicalPlan::Insert { table_id, .. }
        | PhysicalPlan::Update { table_id, .. }
        | PhysicalPlan::Delete { table_id, .. } => Some(*table_id),
        _ => None,
    }
}

/// What one node contributes to the shape hash.
///
/// The work unit, the relation, and the arity. Not the predicate values, not
/// the row estimate: the point of a shape is that two runs of the same
/// statement with different parameters are one shape, so a survivor sees a
/// query template rather than a list of individual executions.
fn shape_of(plan: &PhysicalPlan) -> u64 {
    let kind = match operator_kind_of(plan) {
        Some(kind) => kind.index() as u64 + 1,
        // Plumbing still contributes its arity, because a Gather is what
        // distinguishes a parallel plan from the serial one beside it
        None => 0,
    };
    let relation = relation_of(plan).map(|t| t.0 as u64 + 1).unwrap_or(0);
    let arity = plan.children().len() as u64;
    hash_combine(hash_combine(kind, relation), arity)
}

/// Rows the node is estimated to handle.
fn units_for(plan: &PhysicalPlan) -> f64 {
    let rows = plan.cost().row_count;
    if !rows.is_finite() || rows <= 0.0 {
        return UNKNOWN_ROWS;
    }
    rows
}

/// What a plan costs and what shape it is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlanPrice {
    /// Estimated work in seconds, which is what admission compares against
    /// the class objective
    pub work_seconds: f64,
    /// Structural hash of the plan, stable across parameter values, which is
    /// what identifies a query template in the working-set manifest
    pub fingerprint: u64,
}

/// Sums the plan tree into seconds of work and hashes its shape.
///
/// Both in one walk, because the admission path takes it on every query and
/// walking the tree twice to answer two questions about the same nodes is a
/// cost paid per query for no reason.
///
/// Iterative rather than recursive: a deeply nested plan must not decide how
/// much stack the admission path needs.
pub fn price_plan(plan: &PhysicalPlan, coefficients: &OperatorCoefficients) -> PlanPrice {
    let mut total = 0.0f64;
    let mut fingerprint = SHAPE_SEED;
    let mut stack = vec![plan];
    while let Some(node) = stack.pop() {
        if let Some(kind) = operator_kind_of(node) {
            total += coefficients.estimate_seconds(kind, units_for(node));
        }
        fingerprint = hash_combine(fingerprint, shape_of(node));
        stack.extend(node.children());
    }
    PlanPrice {
        work_seconds: total,
        // Zero is the reserved value for a shape nothing recorded, so a plan
        // that happened to hash to it is moved off rather than dropped
        fingerprint: if fingerprint == 0 {
            SHAPE_SEED
        } else {
            fingerprint
        },
    }
}

/// Sums the plan tree into seconds of work.
pub fn estimate_work_seconds(plan: &PhysicalPlan, coefficients: &OperatorCoefficients) -> f64 {
    price_plan(plan, coefficients).work_seconds
}

/// What the plan costs and what shape it is, against the calibration this node
/// has learned, which is what the serving path uses.
pub fn price_plan_live(plan: &PhysicalPlan) -> PlanPrice {
    // The merged view, so a node that has just joined a mesh prices its first
    // query against what its siblings measured rather than against a constant
    let coefficients = zyron_pressure::pressure_control::PressureController::global()
        .coefficients()
        .effective();
    price_plan(plan, &coefficients)
}

/// The same sum against the calibration this node has learned.
pub fn estimate_work_seconds_live(plan: &PhysicalPlan) -> f64 {
    price_plan_live(plan).work_seconds
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost::PlanCost;

    fn cost(rows: f64) -> PlanCost {
        PlanCost {
            io_cost: 0.0,
            cpu_cost: 0.0,
            row_count: rows,
        }
    }

    fn scan(rows: f64) -> PhysicalPlan {
        PhysicalPlan::SeqScan {
            table_id: zyron_catalog::TableId(1),
            columns: vec![],
            predicate: None,
            cost: cost(rows),
            as_of: None,
        }
    }

    fn coefficients() -> OperatorCoefficients {
        let mut c = OperatorCoefficients::cold_start();
        // A round number per kind, so the arithmetic in each assertion is
        // checkable by hand
        c.ns_per_unit[OperatorKind::SeqScan.index()] = 100.0;
        c.ns_per_unit[OperatorKind::Sort.index()] = 200.0;
        c.ns_per_unit[OperatorKind::Filter.index()] = 10.0;
        c
    }

    #[test]
    fn a_scan_costs_its_rows_times_the_measured_rate() {
        // A million rows at 100ns each is a tenth of a second
        let seconds = estimate_work_seconds(&scan(1_000_000.0), &coefficients());
        assert!((seconds - 0.100).abs() < 1e-9, "got {seconds}");
    }

    #[test]
    fn the_estimate_is_the_sum_over_the_tree() {
        let sorted = PhysicalPlan::Sort {
            order_by: vec![],
            child: Box::new(scan(1_000_000.0)),
            limit: None,
            cost: cost(1_000_000.0),
        };
        // 100ns of scan plus 200ns of sort, per row, over a million rows
        let seconds = estimate_work_seconds(&sorted, &coefficients());
        assert!((seconds - 0.300).abs() < 1e-9, "got {seconds}");
    }

    #[test]
    fn plumbing_nodes_add_nothing_of_their_own() {
        let gathered = PhysicalPlan::Gather {
            child: Box::new(scan(1_000_000.0)),
            num_workers: 4,
            cost: cost(1_000_000.0),
        };
        let bare = estimate_work_seconds(&scan(1_000_000.0), &coefficients());
        let wrapped = estimate_work_seconds(&gathered, &coefficients());
        assert!(
            (bare - wrapped).abs() < 1e-12,
            "a channel charged {} of work",
            wrapped - bare
        );
    }

    /// A plan with no statistics behind it must not price as free, or an
    /// unbounded query would walk through admission as a point lookup.
    #[test]
    fn a_plan_with_no_estimate_is_unknown_rather_than_free() {
        let seconds = estimate_work_seconds(&scan(0.0), &coefficients());
        assert!(seconds > 0.0, "an unestimated scan priced at zero");
        assert!((seconds - UNKNOWN_ROWS * 100.0 / 1e9).abs() < 1e-12);

        // And an impossible estimate is treated the same way rather than
        // producing a NaN that would compare false against every objective
        let nonsense = estimate_work_seconds(&scan(f64::NAN), &coefficients());
        assert!(nonsense.is_finite() && nonsense > 0.0);
        let negative = estimate_work_seconds(&scan(-5.0), &coefficients());
        assert!(negative.is_finite() && negative > 0.0);
    }

    #[test]
    fn a_point_lookup_lands_under_the_admission_bypass() {
        let mut c = OperatorCoefficients::cold_start();
        c.ns_per_unit[OperatorKind::IndexScan.index()] = 200.0;
        c.ns_per_unit[OperatorKind::SeqScan.index()] = 200.0;
        let seconds = estimate_work_seconds(&scan(1.0), &c);
        assert!(
            seconds < zyron_pressure::WorkloadClass::BYPASS_WORK_SECONDS,
            "a one row index lookup estimated at {seconds}s would not bypass"
        );
    }

    #[test]
    fn a_large_scan_classifies_as_bulk() {
        // Ten billion rows at a hundred nanoseconds is well past the
        // interactive objective, and must be classified accordingly
        let seconds = estimate_work_seconds(&scan(1e10), &coefficients());
        assert_eq!(
            zyron_pressure::WorkloadClass::classify(seconds, false),
            zyron_pressure::WorkloadClass::Bulk
        );
    }

    #[test]
    fn every_variant_is_priced_or_deliberately_not() {
        // The match in operator_kind_of is exhaustive, so a new plan variant
        // fails to compile rather than silently costing nothing. This pins the
        // handful that are intentionally free
        assert!(operator_kind_of(&scan(1.0)).is_some());
        let limit = PhysicalPlan::Limit {
            limit: Some(1),
            offset: None,
            child: Box::new(scan(1.0)),
            cost: cost(1.0),
        };
        assert!(operator_kind_of(&limit).is_none());
    }
}

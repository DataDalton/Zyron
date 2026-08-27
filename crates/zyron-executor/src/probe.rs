//! Active probing, for the operator kinds real traffic has stopped exercising.
//!
//! Passive measurement is the whole design: the cost model is built from the
//! work the node was going to do anyway, so nothing is spent on calibration
//! and nothing waits for it at startup. It has one gap. A coefficient can only
//! be refreshed by traffic, and traffic that has stopped refreshes nothing, so
//! a kind that was well measured last week and has not run since is being
//! trusted on evidence that predates whatever changed. A migrated volume, a
//! busier SAN, and a host that started sharing cores all look identical from
//! here: the number stays where it was and the plans priced against it get
//! quietly worse.
//!
//! So the gap is closed by running the operator, and only where that is the
//! honest thing to do. Three rules keep this from becoming the thing passive
//! measurement was chosen to avoid:
//!
//! - Only a kind with too few recent samples is a candidate. A kind real
//!   traffic is exercising is never probed, because the traffic is a better
//!   measurement than the probe.
//! - Never at startup. A node serves immediately on what it inherited, and a
//!   probe that ran first would put a cold start back.
//! - Bounded. One probe gets a fixed slice of one thread and stops at it,
//!   whether or not it finished, so the cost is known before it is spent.
//!
//! Kinds whose cost is a device rather than a loop are not probed at all.
//! Fabricating a page read means either reading a page the node did not need,
//! which pollutes the buffer pool, or reading a synthetic one, which measures
//! a file the workload never touches. Fabricating a WAL fsync means writing
//! to the log. Both are measured passively at high volume anyway, which is why
//! neither needs a probe.

use std::hint::black_box;
use std::time::{Duration, Instant};

use zyron_pressure::capability::OperatorKind;
use zyron_pressure::pressure_control::PressureController;

use crate::column::{Column, ColumnData};
use crate::compute::{self, ArithOp, CmpOp};

/// Wall time one probe may spend.
///
/// A tenth of a second is long enough to time a loop over thousands of rows
/// past any clock granularity, and short enough that a probe landing next to
/// a latency-sensitive query costs that query less than one scheduler
/// timeslice.
pub const PROBE_BUDGET: Duration = Duration::from_millis(100);

/// Rows in a probe batch.
///
/// The executor's own batch size, so what the probe measures is a batch of the
/// shape the operators actually process. Measuring a different shape would
/// produce a per-row cost that no real batch reproduces.
pub const PROBE_ROWS: usize = 8_192;

/// What one probe measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProbeOutcome {
    pub kind: OperatorKind,
    /// Rows the probe put through the operator
    pub units: u64,
    pub elapsed: Duration,
}

impl ProbeOutcome {
    pub fn nanos_per_unit(&self) -> f64 {
        if self.units == 0 {
            return 0.0;
        }
        self.elapsed.as_nanos() as f64 / self.units as f64
    }
}

/// What a probing pass did.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ProbeSummary {
    pub measured: Vec<ProbeOutcome>,
    /// Candidates whose cost is a device rather than a loop, listed so the
    /// view can say why they were left alone rather than leaving an operator
    /// to wonder whether the pass silently failed
    pub not_probeable: Vec<OperatorKind>,
    pub total_elapsed: Duration,
}

/// Whether a kind can be measured without side effects or fabricated storage.
pub const fn probeable(kind: OperatorKind) -> bool {
    match kind {
        OperatorKind::SeqScan
        | OperatorKind::Filter
        | OperatorKind::Project
        | OperatorKind::HashJoinBuild
        | OperatorKind::HashJoinProbe
        | OperatorKind::Sort
        | OperatorKind::Aggregate
        | OperatorKind::Window
        | OperatorKind::SetOp => true,
        // An index scan's cost is the index it walks, a lake scan's is the
        // files it opens, and neither exists to be walked synthetically. Both
        // are measured whenever a plan uses them
        OperatorKind::IndexScan | OperatorKind::LakeScan => false,
        // A device, and fabricating one would either pollute the buffer pool
        // or write to the log
        OperatorKind::PageRead | OperatorKind::WalFsync => false,
    }
}

/// Runs one probe and records what it measured.
///
/// Returns None for a kind that cannot be probed. Records through the same
/// path the operators use, so a probed coefficient and a measured one are
/// indistinguishable afterwards, which is correct: both are this machine
/// running this operator.
pub fn probe_kind(kind: OperatorKind, budget: Duration) -> Option<ProbeOutcome> {
    if !probeable(kind) {
        return None;
    }
    let started = Instant::now();
    let units = match kind {
        OperatorKind::SeqScan => run_scan(budget),
        OperatorKind::Filter => run_filter(budget),
        OperatorKind::Project => run_project(budget),
        OperatorKind::HashJoinBuild => run_join_build(budget),
        OperatorKind::HashJoinProbe => run_join_probe(budget),
        OperatorKind::Sort => run_sort(budget),
        OperatorKind::Aggregate => run_aggregate(budget),
        OperatorKind::Window => run_window(budget),
        OperatorKind::SetOp => run_setop(budget),
        _ => 0,
    };
    let elapsed = started.elapsed();
    if units == 0 {
        return None;
    }
    PressureController::global().record_operator(kind, units, elapsed);
    Some(ProbeOutcome {
        kind,
        units,
        elapsed,
    })
}

/// Probes every kind the node has too little recent evidence for.
///
/// The candidate list comes from the accumulator's recent window, so a node
/// under real traffic probes nothing: the traffic already answered the
/// question the probe would ask.
pub fn probe_stale_kinds(budget_per_probe: Duration) -> ProbeSummary {
    let candidates = PressureController::global()
        .coefficients()
        .probe_candidates();
    probe_these(&candidates, budget_per_probe)
}

/// Probes a named set, which is what a test and an operator-triggered refresh
/// both need.
pub fn probe_these(kinds: &[OperatorKind], budget_per_probe: Duration) -> ProbeSummary {
    let started = Instant::now();
    let mut summary = ProbeSummary::default();
    for kind in kinds {
        if !probeable(*kind) {
            summary.not_probeable.push(*kind);
            continue;
        }
        if let Some(outcome) = probe_kind(*kind, budget_per_probe) {
            summary.measured.push(outcome);
        }
    }
    summary.total_elapsed = started.elapsed();
    summary
}

// ---------------------------------------------------------------------------
// The synthetic batch
// ---------------------------------------------------------------------------

/// Values a probe runs over.
///
/// Not sorted and not uniform: a sorted input measures the branch predictor
/// rather than the sort, and constant keys collapse a hash join into one
/// bucket. The sequence is generated rather than random so two probes on one
/// machine are comparable.
fn probe_values(count: usize, offset: u64) -> Vec<i64> {
    let mut out = Vec::with_capacity(count);
    let mut state = 0x9E37_79B9_7F4A_7C15u64 ^ offset;
    for _ in 0..count {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        out.push((state >> 16) as i64);
    }
    out
}

fn probe_column(count: usize, offset: u64) -> Column {
    Column::new(
        ColumnData::Int64(probe_values(count, offset)),
        zyron_common::TypeId::Int64,
    )
}

fn scalar_column(value: i64) -> Column {
    Column::new(ColumnData::Int64(vec![value]), zyron_common::TypeId::Int64)
}

/// Repeats `body` until the budget runs out, returning rows processed.
///
/// The deadline is checked once per batch rather than once per row, so the
/// check is spread over thousands of rows and the loop measures the operator
/// rather than the clock.
fn until_budget(budget: Duration, mut body: impl FnMut() -> u64) -> u64 {
    let deadline = Instant::now() + budget;
    let mut units = 0u64;
    loop {
        units = units.saturating_add(body());
        if Instant::now() >= deadline {
            return units;
        }
    }
}

// ---------------------------------------------------------------------------
// Per kind
// ---------------------------------------------------------------------------

/// A scan's per-row cost is reading a value out of column storage and
/// checking its null bit, which is the loop every scan operator runs.
fn run_scan(budget: Duration) -> u64 {
    let column = probe_column(PROBE_ROWS, 1);
    until_budget(budget, || {
        let ColumnData::Int64(values) = &column.data else {
            return 0;
        };
        let mut total = 0i64;
        for (i, v) in values.iter().enumerate() {
            if column.nulls.is_valid(i) {
                total = total.wrapping_add(*v);
            }
        }
        black_box(total);
        values.len() as u64
    })
}

/// A filter evaluates a predicate over the batch and turns the answer into a
/// selection, which is both halves of what a Filter node does.
fn run_filter(budget: Duration) -> u64 {
    let column = probe_column(PROBE_ROWS, 2);
    let threshold = scalar_column(1 << 40);
    until_budget(budget, || {
        let Ok(mask) = compute::compare_scalar(&column, &threshold, CmpOp::Gt, false) else {
            return 0;
        };
        let selected = compute::column_to_mask(&mask);
        black_box(selected.iter().filter(|s| **s).count());
        column.len() as u64
    })
}

/// A projection evaluates an expression per output column. Arithmetic over
/// two columns is the shape that costs something, so that is what is timed.
fn run_project(budget: Duration) -> u64 {
    let left = probe_column(PROBE_ROWS, 3);
    let right = probe_column(PROBE_ROWS, 4);
    until_budget(budget, || {
        let Ok(sum) = compute::arithmetic(&left, &right, ArithOp::Add) else {
            return 0;
        };
        black_box(sum.len());
        left.len() as u64
    })
}

/// Building a hash side is hashing the keys and inserting them.
fn run_join_build(budget: Duration) -> u64 {
    let keys = probe_column(PROBE_ROWS, 5);
    until_budget(budget, || {
        let hashes = compute::hash_column_batch(&[&keys], keys.len());
        let mut table: std::collections::HashMap<u64, u32> =
            std::collections::HashMap::with_capacity(hashes.len());
        for (row, hash) in hashes.iter().enumerate() {
            table.insert(*hash, row as u32);
        }
        black_box(table.len());
        keys.len() as u64
    })
}

/// Probing is hashing the other side and looking each key up, which is a
/// different cost from building: the table is already warm and the access
/// pattern is a scattered read rather than an insert.
fn run_join_probe(budget: Duration) -> u64 {
    let build_keys = probe_column(PROBE_ROWS, 5);
    let probe_keys = probe_column(PROBE_ROWS, 6);
    let build_hashes = compute::hash_column_batch(&[&build_keys], build_keys.len());
    let mut table: std::collections::HashMap<u64, u32> =
        std::collections::HashMap::with_capacity(build_hashes.len());
    for (row, hash) in build_hashes.iter().enumerate() {
        table.insert(*hash, row as u32);
    }
    until_budget(budget, || {
        let hashes = compute::hash_column_batch(&[&probe_keys], probe_keys.len());
        let mut matched = 0u32;
        for hash in &hashes {
            if table.get(hash).is_some() {
                matched += 1;
            }
        }
        black_box(matched);
        probe_keys.len() as u64
    })
}

/// A sort's cost is the sort itself, over a fresh copy each pass: sorting an
/// already sorted vector measures nothing a real sort ever encounters.
fn run_sort(budget: Duration) -> u64 {
    let source = probe_values(PROBE_ROWS, 7);
    until_budget(budget, || {
        let mut data = ColumnData::Int64(source.clone());
        compute::sort_column_inplace(&mut data, true);
        black_box(data.len());
        source.len() as u64
    })
}

/// An aggregate hashes the grouping keys and accumulates into the group.
/// The key set is narrower than the batch, so groups are revisited, which is
/// what an aggregate actually does.
fn run_aggregate(budget: Duration) -> u64 {
    let values = probe_values(PROBE_ROWS, 8);
    let groups: Vec<i64> = values.iter().map(|v| v % 512).collect();
    let group_column = Column::new(ColumnData::Int64(groups), zyron_common::TypeId::Int64);
    until_budget(budget, || {
        let hashes = compute::hash_column_batch(&[&group_column], group_column.len());
        let mut accumulator: std::collections::HashMap<u64, i64> =
            std::collections::HashMap::with_capacity(512);
        for (row, hash) in hashes.iter().enumerate() {
            let slot = accumulator.entry(*hash).or_insert(0);
            *slot = slot.wrapping_add(values[row]);
        }
        black_box(accumulator.len());
        values.len() as u64
    })
}

/// A window function walks a frame per row. A running sum over a sliding
/// frame is the cheapest shape that still touches every row more than once.
fn run_window(budget: Duration) -> u64 {
    let values = probe_values(PROBE_ROWS, 9);
    const FRAME: usize = 16;
    until_budget(budget, || {
        let mut out = Vec::with_capacity(values.len());
        let mut running = 0i64;
        for (i, v) in values.iter().enumerate() {
            running = running.wrapping_add(*v);
            if i >= FRAME {
                running = running.wrapping_sub(values[i - FRAME]);
            }
            out.push(running);
        }
        black_box(out.len());
        values.len() as u64
    })
}

/// A set operation deduplicates by hash, which is the cost that separates a
/// UNION from a UNION ALL.
fn run_setop(budget: Duration) -> u64 {
    let column = probe_column(PROBE_ROWS, 10);
    until_budget(budget, || {
        let hashes = compute::hash_column_batch(&[&column], column.len());
        let seen: std::collections::HashSet<u64> = hashes.into_iter().collect();
        black_box(seen.len());
        column.len() as u64
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A probe stays inside the slice it was given, or the bound is not a
    /// bound and a probe on a slow machine becomes a stall.
    #[test]
    fn a_probe_stops_at_its_budget() {
        let budget = Duration::from_millis(40);
        for kind in OperatorKind::ALL {
            if !probeable(kind) {
                continue;
            }
            let outcome = probe_kind(kind, budget).expect("probeable kind measured nothing");
            assert!(
                outcome.elapsed < budget * 4,
                "{kind:?} ran {:?} against a {budget:?} budget",
                outcome.elapsed
            );
            assert!(outcome.units > 0);
            assert!(outcome.nanos_per_unit() > 0.0);
        }
    }

    /// The kinds whose cost is a device are refused rather than faked.
    #[test]
    fn device_bound_kinds_are_not_probed() {
        for kind in [
            OperatorKind::PageRead,
            OperatorKind::WalFsync,
            OperatorKind::IndexScan,
            OperatorKind::LakeScan,
        ] {
            assert!(!probeable(kind));
            assert_eq!(probe_kind(kind, PROBE_BUDGET), None);
        }
        let summary = probe_these(&[OperatorKind::PageRead], Duration::from_millis(1));
        assert_eq!(summary.not_probeable, vec![OperatorKind::PageRead]);
        assert!(summary.measured.is_empty());
    }

    /// A probe has to move the coefficient, or it spent the time for nothing.
    #[test]
    fn a_probe_feeds_the_cost_model() {
        let controller = PressureController::global();
        let before = controller.coefficients().samples(OperatorKind::Window);
        probe_kind(OperatorKind::Window, Duration::from_millis(20)).expect("measured");
        controller.drain_calibration();
        assert!(
            controller.coefficients().samples(OperatorKind::Window) > before,
            "the probe did not reach the accumulator"
        );
        assert!(controller.coefficients().get(OperatorKind::Window) > 0.0);
    }

    /// A pass over the whole set costs at most the budget times the kinds it
    /// can actually probe, which is what makes the cost knowable in advance.
    #[test]
    fn a_full_pass_is_bounded_by_the_kinds_it_can_probe() {
        let budget = Duration::from_millis(10);
        let summary = probe_these(&OperatorKind::ALL, budget);
        let probeable_count = OperatorKind::ALL.iter().filter(|k| probeable(**k)).count();
        assert_eq!(summary.measured.len(), probeable_count);
        assert_eq!(
            summary.not_probeable.len(),
            OperatorKind::COUNT - probeable_count
        );
        assert!(
            summary.total_elapsed < budget * (probeable_count as u32) * 4,
            "a full pass took {:?}",
            summary.total_elapsed
        );
    }
}

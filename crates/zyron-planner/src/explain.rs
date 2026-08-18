//! EXPLAIN query plan output.
//!
//! Builds a tree representation of a physical plan with estimated costs,
//! and optionally merges actual execution metrics from EXPLAIN ANALYZE.
//! Supports text, JSON, and YAML output formats.

use crate::cost::PlanCost;
use crate::physical::PhysicalPlan;
use std::fmt::Write;

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

/// Controls what information EXPLAIN includes in its output.
#[derive(Debug, Clone)]
pub struct ExplainOptions {
    /// Execute the query and collect runtime metrics.
    pub analyze: bool,
    /// Show cost estimates (default: true).
    pub costs: bool,
    /// Show buffer hit/miss counts (requires ANALYZE).
    pub buffers: bool,
    /// Show per-operator timing (requires ANALYZE).
    pub timing: bool,
    /// Output format.
    pub format: ExplainFormat,
}

impl Default for ExplainOptions {
    fn default() -> Self {
        Self {
            analyze: false,
            costs: true,
            buffers: false,
            timing: true,
            format: ExplainFormat::Text,
        }
    }
}

/// Output format for EXPLAIN results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExplainFormat {
    Text,
    Json,
    Yaml,
}

impl ExplainFormat {
    /// Parses a format name string (case-insensitive).
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "json" => ExplainFormat::Json,
            "yaml" => ExplainFormat::Yaml,
            _ => ExplainFormat::Text,
        }
    }
}

// ---------------------------------------------------------------------------
// Actual metrics
// ---------------------------------------------------------------------------

/// Auxiliary counter slots an operator may fill. Fixed width so the
/// executor's per-operator metrics stay allocation free on the hot path;
/// what each slot means depends on the operator and is resolved at render
/// time by `aux_labels`
pub const ACTUAL_AUX_SLOTS: usize = 6;

/// Elapsed nanoseconds split into whole milliseconds and thousandths, the two
/// integers the renderers print as `{}.{:03}`.
///
/// The digits match `{:.3}` applied to the same count as an f64 millisecond
/// value, exactly, including where the float's representation error decides the
/// last place. `{:.3}` rounds the true value of the float to three places with
/// ties to even, and that value is `m * 2^e` for the integers sitting in the
/// float's bits, so the rounding is a shift and a tie test rather than a decimal
/// conversion. The conversion is where the float formatter spends its time.
///
/// The divide stays, because which side of a tie the quotient lands on is a
/// property of the division. Scaling the integer nanosecond count instead would
/// change printed digits at an exact half microsecond
#[inline]
pub fn millis_parts(ns: u64) -> (u64, u64) {
    let thousandths = millis_thousandths(ns);
    (thousandths / 1_000, thousandths % 1_000)
}

/// Thousandths of a millisecond, rounded the way `{:.3}` rounds.
#[inline]
fn millis_thousandths(ns: u64) -> u64 {
    let value = ns as f64 / 1_000_000.0;
    let bits = value.to_bits();
    let biased_exponent = ((bits >> 52) & 0x7FF) as i32;
    let fraction = bits & ((1u64 << 52) - 1);

    // A subnormal carries no implicit leading one and sits at a fixed exponent
    let (mantissa, exponent) = if biased_exponent == 0 {
        (fraction, -1074i32)
    } else {
        (fraction | (1u64 << 52), biased_exponent - 1075)
    };
    if mantissa == 0 {
        return 0;
    }

    // Under 2^63, so scaling by a thousand still needs only a u128 to stay exact
    let scaled = mantissa as u128 * 1_000u128;

    if exponent >= 0 {
        // A u64 nanosecond count reaches about 1.8e13 milliseconds, far under
        // the 2^52 needed to land here. Kept so the function is total
        return u64::try_from(scaled << (exponent as u32).min(127)).unwrap_or(u64::MAX);
    }

    let shift = exponent.unsigned_abs();
    if shift >= 128 {
        return 0;
    }
    let quotient = scaled >> shift;
    let remainder = scaled & ((1u128 << shift) - 1);
    let half = 1u128 << (shift - 1);
    let rounded = if remainder > half || (remainder == half && quotient & 1 == 1) {
        quotient + 1
    } else {
        quotient
    };
    u64::try_from(rounded).unwrap_or(u64::MAX)
}

/// Runtime metrics collected during EXPLAIN ANALYZE execution.
#[derive(Debug, Clone, Default)]
pub struct ActualMetrics {
    pub rows: u64,
    pub elapsed_ns: u64,
    pub batches: u64,
    /// Operator-specific counters, zero when the operator fills none
    pub aux: [u64; ACTUAL_AUX_SLOTS],
}

/// What one executed operator measured, as a tree mirroring the plan.
///
/// A tree rather than a flat list on purpose. Matching measurements to plan
/// nodes by pre-order position assumes the two trees have identical shape,
/// and when they do not the result is not missing numbers but a join's
/// timing reported against a scan. Merging structurally, by name at each
/// level, attaches what matches and leaves the rest empty.
#[derive(Debug, Clone, Default)]
pub struct NodeMetrics {
    pub name: String,
    pub rows: u64,
    pub elapsed_ns: u64,
    pub batches: u64,
    pub aux: [u64; ACTUAL_AUX_SLOTS],
    pub children: Vec<NodeMetrics>,
}

/// Labels for an operator's auxiliary counters, empty for a slot the
/// operator does not fill. Presentation lives here so the executor's
/// counters stay a fixed-width array of atomics
pub fn aux_labels(operator_name: &str) -> [&'static str; ACTUAL_AUX_SLOTS] {
    match operator_name {
        "LakeScan" | "LakeDelete" | "LakeUpdate" => [
            "files_considered",
            "files_pruned",
            "bytes_considered",
            "bytes_pruned",
            // Zero index files read means the scan answered the predicate
            // from the data files, either because no index applied or
            // because probing one would have read more than it saved
            "index_files_read",
            "index_rows_addressed",
        ],
        // A foreign scan's cost is the peer's answer, so what it reports is
        // how much came back and how long the round trip took
        "ForeignScan" => ["rows_fetched", "remote_ms", "", "", "", ""],
        _ => ["", "", "", "", "", ""],
    }
}

// ---------------------------------------------------------------------------
// Explain node
// ---------------------------------------------------------------------------

/// Tree node representing one operator in the EXPLAIN output.
#[derive(Debug, Clone)]
pub struct ExplainNode {
    /// Operator name (e.g., "SeqScan", "HashJoin").
    pub operator_name: String,
    /// Key-value detail pairs (e.g., ("table", "orders"), ("predicate", "id > 5")).
    pub details: Vec<(String, String)>,
    /// Estimated cost from the planner.
    pub estimated_cost: Option<PlanCost>,
    /// Actual runtime metrics (populated by EXPLAIN ANALYZE).
    pub actual_metrics: Option<ActualMetrics>,
    /// Child operator nodes.
    pub children: Vec<ExplainNode>,
}

/// Records the time-travel qualifier a scan reads at, so a plan says which
/// version answered rather than only which access path ran.
fn push_as_of(details: &mut Vec<(String, String)>, as_of: &Option<crate::logical::AsOfTarget>) {
    match as_of {
        Some(crate::logical::AsOfTarget::Version(v)) => {
            details.push(("as_of_version".to_string(), v.to_string()))
        }
        Some(crate::logical::AsOfTarget::Timestamp(us)) => {
            details.push(("as_of_timestamp_us".to_string(), us.to_string()))
        }
        Some(crate::logical::AsOfTarget::Branch(name)) => {
            details.push(("as_of_branch".to_string(), name.clone()))
        }
        None => {}
    }
}

impl ExplainNode {
    /// Builds an ExplainNode tree from a PhysicalPlan.
    pub fn from_physical_plan(plan: &PhysicalPlan) -> Self {
        match plan {
            PhysicalPlan::SeqScan {
                table_id,
                columns,
                predicate,
                cost,
                ..
            } => {
                let mut details = vec![
                    ("table_id".to_string(), format!("{}", table_id.0)),
                    ("columns".to_string(), format!("{}", columns.len())),
                ];
                if predicate.is_some() {
                    details.push(("filter".to_string(), "yes".to_string()));
                }
                Self {
                    operator_name: "SeqScan".to_string(),
                    details,
                    estimated_cost: Some(*cost),
                    actual_metrics: None,
                    children: Vec::new(),
                }
            }
            PhysicalPlan::HybridScan {
                table_id,
                columns,
                predicate,
                cost,
                ..
            } => {
                let mut details = vec![
                    ("table_id".to_string(), format!("{}", table_id.0)),
                    ("columns".to_string(), format!("{}", columns.len())),
                    ("stores".to_string(), "columnar+heap".to_string()),
                ];
                if predicate.is_some() {
                    details.push(("filter".to_string(), "yes".to_string()));
                }
                Self {
                    operator_name: "HybridScan".to_string(),
                    details,
                    estimated_cost: Some(*cost),
                    actual_metrics: None,
                    children: Vec::new(),
                }
            }
            PhysicalPlan::LakeScan {
                table_id,
                columns,
                predicate,
                lowered,
                as_of,
                cost,
            } => {
                let mut details = vec![
                    ("table_id".to_string(), format!("{}", table_id.0)),
                    ("columns".to_string(), format!("{}", columns.len())),
                    ("store".to_string(), "lake".to_string()),
                ];
                if predicate.is_some() {
                    details.push(("filter".to_string(), "yes".to_string()));
                    // Whether the filter skips whole data files, which is
                    // the difference between reading one file and all of them
                    details.push((
                        "file_pruning".to_string(),
                        if lowered.is_some() {
                            "exact".to_string()
                        } else {
                            "none".to_string()
                        },
                    ));
                }
                match as_of {
                    Some(crate::logical::AsOfTarget::Version(v)) => {
                        details.push(("as_of_version".to_string(), v.to_string()))
                    }
                    Some(crate::logical::AsOfTarget::Timestamp(us)) => {
                        details.push(("as_of_timestamp_us".to_string(), us.to_string()))
                    }
                    Some(crate::logical::AsOfTarget::Branch(name)) => {
                        details.push(("as_of_branch".to_string(), name.clone()))
                    }
                    None => {}
                }
                Self {
                    operator_name: "LakeScan".to_string(),
                    details,
                    estimated_cost: Some(*cost),
                    actual_metrics: None,
                    children: Vec::new(),
                }
            }
            PhysicalPlan::ForeignScan {
                table_id,
                columns,
                residual,
                request,
                cost,
            } => {
                let mut details = vec![
                    ("table_id".to_string(), format!("{}", table_id.0)),
                    ("columns".to_string(), format!("{}", columns.len())),
                    ("peer".to_string(), request.peer.clone()),
                    ("remote_table".to_string(), request.table.clone()),
                ];
                // Where each half of the filter runs. A predicate the peer
                // applies costs one round trip of narrower rows; one left
                // here costs the full result crossing the wire first, so
                // the split is the number to look at
                match &request.predicate {
                    Some(sql) => details.push(("pushed_filter".to_string(), sql.clone())),
                    None => details.push(("pushed_filter".to_string(), "none".to_string())),
                }
                if residual.is_some() {
                    details.push(("local_filter".to_string(), "yes".to_string()));
                }
                if let Some(limit) = request.limit {
                    details.push(("pushed_limit".to_string(), limit.to_string()));
                }
                Self {
                    operator_name: "ForeignScan".to_string(),
                    details,
                    estimated_cost: Some(*cost),
                    actual_metrics: None,
                    children: Vec::new(),
                }
            }
            PhysicalPlan::LakeDelete {
                table_id,
                predicate,
                sql,
                cost,
                ..
            } => Self {
                operator_name: "LakeDelete".to_string(),
                details: vec![
                    ("table_id".to_string(), format!("{}", table_id.0)),
                    (
                        "predicate".to_string(),
                        if predicate.is_some() {
                            sql.clone()
                        } else {
                            "all rows".to_string()
                        },
                    ),
                ],
                estimated_cost: Some(*cost),
                actual_metrics: None,
                children: Vec::new(),
            },
            PhysicalPlan::LakeUpdate {
                table_id,
                assignments,
                predicate,
                sql,
                child,
                cost,
                ..
            } => Self {
                operator_name: "LakeUpdate".to_string(),
                details: vec![
                    ("table_id".to_string(), format!("{}", table_id.0)),
                    ("assignments".to_string(), format!("{}", assignments.len())),
                    (
                        "predicate".to_string(),
                        if predicate.is_some() {
                            sql.clone()
                        } else {
                            "all rows".to_string()
                        },
                    ),
                ],
                estimated_cost: Some(*cost),
                actual_metrics: None,
                children: vec![Self::from_physical_plan(child)],
            },
            PhysicalPlan::ColumnarMetadataAggregate {
                table_id,
                specs,
                cost,
                ..
            } => {
                let aggs = specs
                    .iter()
                    .map(|s| match s.kind {
                        crate::physical::MetaAggKind::CountStar => "count(*)".to_string(),
                        crate::physical::MetaAggKind::CountCol => "count(col)".to_string(),
                        crate::physical::MetaAggKind::Min => "min(col)".to_string(),
                        crate::physical::MetaAggKind::Max => "max(col)".to_string(),
                    })
                    .collect::<Vec<_>>()
                    .join(",");
                Self {
                    operator_name: "ColumnarMetadataAggregate".to_string(),
                    details: vec![
                        ("table_id".to_string(), format!("{}", table_id.0)),
                        ("aggs".to_string(), aggs),
                        ("source".to_string(), "segment-headers+heap".to_string()),
                    ],
                    estimated_cost: Some(*cost),
                    actual_metrics: None,
                    children: Vec::new(),
                }
            }
            PhysicalPlan::IndexScan {
                table_id,
                index_id,
                cost,
                ..
            } => Self {
                operator_name: "IndexScan".to_string(),
                details: vec![
                    ("table_id".to_string(), format!("{}", table_id.0)),
                    ("index_id".to_string(), format!("{}", index_id.0)),
                ],
                estimated_cost: Some(*cost),
                actual_metrics: None,
                children: Vec::new(),
            },
            PhysicalPlan::Filter {
                predicate: _,
                child,
                cost,
            } => Self {
                operator_name: "Filter".to_string(),
                details: Vec::new(),
                estimated_cost: Some(*cost),
                actual_metrics: None,
                children: vec![Self::from_physical_plan(child)],
            },
            PhysicalPlan::Project {
                expressions,
                child,
                cost,
                ..
            } => Self {
                operator_name: "Project".to_string(),
                details: vec![("columns".to_string(), format!("{}", expressions.len()))],
                estimated_cost: Some(*cost),
                actual_metrics: None,
                children: vec![Self::from_physical_plan(child)],
            },
            PhysicalPlan::NestedLoopJoin {
                left,
                right,
                join_type,
                cost,
                ..
            } => Self {
                operator_name: "NestedLoopJoin".to_string(),
                details: vec![("join_type".to_string(), format!("{:?}", join_type))],
                estimated_cost: Some(*cost),
                actual_metrics: None,
                children: vec![
                    Self::from_physical_plan(left),
                    Self::from_physical_plan(right),
                ],
            },
            PhysicalPlan::LateralJoin {
                left,
                join_type,
                cost,
                ..
            } => Self {
                operator_name: "LateralJoin".to_string(),
                details: vec![("join_type".to_string(), format!("{:?}", join_type))],
                estimated_cost: Some(*cost),
                actual_metrics: None,
                // The lateral subquery is executed per left row, not a static
                // child plan, so only the left input shows as a child.
                children: vec![Self::from_physical_plan(left)],
            },
            PhysicalPlan::HashJoin {
                left,
                right,
                join_type,
                cost,
                ..
            } => Self {
                operator_name: "HashJoin".to_string(),
                details: vec![("join_type".to_string(), format!("{:?}", join_type))],
                estimated_cost: Some(*cost),
                actual_metrics: None,
                children: vec![
                    Self::from_physical_plan(left),
                    Self::from_physical_plan(right),
                ],
            },
            PhysicalPlan::MergeJoin {
                left,
                right,
                join_type,
                cost,
                ..
            } => Self {
                operator_name: "MergeJoin".to_string(),
                details: vec![("join_type".to_string(), format!("{:?}", join_type))],
                estimated_cost: Some(*cost),
                actual_metrics: None,
                children: vec![
                    Self::from_physical_plan(left),
                    Self::from_physical_plan(right),
                ],
            },
            PhysicalPlan::HashAggregate {
                group_by,
                aggregates,
                child,
                cost,
            } => Self {
                operator_name: "HashAggregate".to_string(),
                details: vec![
                    ("groups".to_string(), format!("{}", group_by.len())),
                    ("aggregates".to_string(), format!("{}", aggregates.len())),
                ],
                estimated_cost: Some(*cost),
                actual_metrics: None,
                children: vec![Self::from_physical_plan(child)],
            },
            PhysicalPlan::SortAggregate {
                group_by,
                aggregates,
                child,
                cost,
            } => Self {
                operator_name: "SortAggregate".to_string(),
                details: vec![
                    ("groups".to_string(), format!("{}", group_by.len())),
                    ("aggregates".to_string(), format!("{}", aggregates.len())),
                ],
                estimated_cost: Some(*cost),
                actual_metrics: None,
                children: vec![Self::from_physical_plan(child)],
            },
            PhysicalPlan::GapFill {
                bucket_col,
                width,
                child,
                cost,
            } => Self {
                operator_name: "GapFill".to_string(),
                details: vec![
                    ("bucket_col".to_string(), format!("{bucket_col}")),
                    ("width".to_string(), format!("{width}")),
                ],
                estimated_cost: Some(*cost),
                actual_metrics: None,
                children: vec![Self::from_physical_plan(child)],
            },
            PhysicalPlan::Sort {
                child, limit, cost, ..
            } => {
                let mut details = Vec::new();
                if let Some(l) = limit {
                    details.push(("top_n".to_string(), format!("{}", l)));
                }
                Self {
                    operator_name: "Sort".to_string(),
                    details,
                    estimated_cost: Some(*cost),
                    actual_metrics: None,
                    children: vec![Self::from_physical_plan(child)],
                }
            }
            PhysicalPlan::Limit {
                limit,
                offset,
                child,
                cost,
            } => {
                let mut details = Vec::new();
                if let Some(l) = limit {
                    details.push(("limit".to_string(), format!("{}", l)));
                }
                if let Some(o) = offset {
                    details.push(("offset".to_string(), format!("{}", o)));
                }
                Self {
                    operator_name: "Limit".to_string(),
                    details,
                    estimated_cost: Some(*cost),
                    actual_metrics: None,
                    children: vec![Self::from_physical_plan(child)],
                }
            }
            PhysicalPlan::HashDistinct { child, cost } => Self {
                operator_name: "HashDistinct".to_string(),
                details: Vec::new(),
                estimated_cost: Some(*cost),
                actual_metrics: None,
                children: vec![Self::from_physical_plan(child)],
            },
            PhysicalPlan::LockRows {
                table_id,
                mode,
                wait,
                cap,
                child,
                cost,
            } => {
                let mut details = vec![
                    ("table_id".to_string(), format!("{}", table_id.0)),
                    ("mode".to_string(), format!("{:?}", mode)),
                    ("wait".to_string(), format!("{:?}", wait)),
                ];
                if let Some(c) = cap {
                    details.push(("cap".to_string(), format!("{c}")));
                }
                Self {
                    operator_name: "LockRows".to_string(),
                    details,
                    estimated_cost: Some(*cost),
                    actual_metrics: None,
                    children: vec![Self::from_physical_plan(child)],
                }
            }
            PhysicalPlan::SetOp {
                op,
                all,
                left,
                right,
                cost,
            } => Self {
                operator_name: format!("{:?}", op),
                details: vec![("all".to_string(), format!("{}", all))],
                estimated_cost: Some(*cost),
                actual_metrics: None,
                children: vec![
                    Self::from_physical_plan(left),
                    Self::from_physical_plan(right),
                ],
            },
            PhysicalPlan::Insert {
                table_id,
                source,
                cost,
                ..
            } => Self {
                operator_name: "Insert".to_string(),
                details: vec![("table_id".to_string(), format!("{}", table_id.0))],
                estimated_cost: Some(*cost),
                actual_metrics: None,
                children: vec![Self::from_physical_plan(source)],
            },
            PhysicalPlan::Values { rows, cost, .. } => Self {
                operator_name: "Values".to_string(),
                details: vec![("rows".to_string(), format!("{}", rows.len()))],
                estimated_cost: Some(*cost),
                actual_metrics: None,
                children: Vec::new(),
            },
            PhysicalPlan::Update {
                table_id,
                child,
                cost,
                ..
            } => Self {
                operator_name: "Update".to_string(),
                details: vec![("table_id".to_string(), format!("{}", table_id.0))],
                estimated_cost: Some(*cost),
                actual_metrics: None,
                children: vec![Self::from_physical_plan(child)],
            },
            PhysicalPlan::Delete {
                table_id,
                child,
                cost,
            } => Self {
                operator_name: "Delete".to_string(),
                details: vec![("table_id".to_string(), format!("{}", table_id.0))],
                estimated_cost: Some(*cost),
                actual_metrics: None,
                children: vec![Self::from_physical_plan(child)],
            },
            // Parallel plan variants
            PhysicalPlan::ParallelSeqScan {
                table_id,
                columns,
                num_workers,
                cost,
                ..
            } => Self {
                operator_name: "ParallelSeqScan".to_string(),
                details: vec![
                    ("table_id".to_string(), format!("{}", table_id.0)),
                    ("columns".to_string(), format!("{}", columns.len())),
                    ("workers".to_string(), format!("{}", num_workers)),
                ],
                estimated_cost: Some(*cost),
                actual_metrics: None,
                children: Vec::new(),
            },
            PhysicalPlan::ParallelHashJoin {
                left,
                right,
                join_type,
                num_workers,
                cost,
                ..
            } => Self {
                operator_name: "ParallelHashJoin".to_string(),
                details: vec![
                    ("join_type".to_string(), format!("{:?}", join_type)),
                    ("workers".to_string(), format!("{}", num_workers)),
                ],
                estimated_cost: Some(*cost),
                actual_metrics: None,
                children: vec![
                    Self::from_physical_plan(left),
                    Self::from_physical_plan(right),
                ],
            },
            PhysicalPlan::Gather {
                child,
                num_workers,
                cost,
            } => Self {
                operator_name: "Gather".to_string(),
                details: vec![("workers".to_string(), format!("{}", num_workers))],
                estimated_cost: Some(*cost),
                actual_metrics: None,
                children: vec![Self::from_physical_plan(child)],
            },
            PhysicalPlan::Repartition {
                child,
                num_partitions,
                cost,
                ..
            } => Self {
                operator_name: "Repartition".to_string(),
                details: vec![("partitions".to_string(), format!("{}", num_partitions))],
                estimated_cost: Some(*cost),
                actual_metrics: None,
                children: vec![Self::from_physical_plan(child)],
            },
            PhysicalPlan::Broadcast {
                child,
                num_workers,
                cost,
            } => Self {
                operator_name: "Broadcast".to_string(),
                details: vec![("workers".to_string(), format!("{}", num_workers))],
                estimated_cost: Some(*cost),
                actual_metrics: None,
                children: vec![Self::from_physical_plan(child)],
            },

            PhysicalPlan::FulltextScan {
                table_id,
                index_id,
                columns,
                as_of,
                cost,
                ..
            } => {
                let mut details = vec![
                    ("table_id".to_string(), format!("{}", table_id.0)),
                    ("index_id".to_string(), format!("{}", index_id.0)),
                    ("columns".to_string(), format!("{}", columns.len())),
                ];
                push_as_of(&mut details, as_of);
                Self {
                    operator_name: "FulltextScan".to_string(),
                    details,
                    estimated_cost: Some(*cost),
                    actual_metrics: None,
                    children: Vec::new(),
                }
            }
            PhysicalPlan::VectorScan {
                table_id,
                index_id,
                columns,
                as_of,
                cost,
                k,
                ..
            } => {
                let mut details = vec![
                    ("table_id".to_string(), format!("{}", table_id.0)),
                    ("index_id".to_string(), format!("{}", index_id.0)),
                    ("columns".to_string(), format!("{}", columns.len())),
                    ("k".to_string(), format!("{}", k)),
                ];
                push_as_of(&mut details, as_of);
                Self {
                    operator_name: "VectorScan".to_string(),
                    details,
                    estimated_cost: Some(*cost),
                    actual_metrics: None,
                    children: Vec::new(),
                }
            }
            PhysicalPlan::SpatialScan {
                table_id,
                index_id,
                columns,
                kind,
                as_of,
                cost,
                ..
            } => {
                let kind_str = match kind {
                    super::physical::SpatialScanKind::Knn { k, .. } => format!("knn(k={})", k),
                    super::physical::SpatialScanKind::DWithin { radius_meters, .. } => {
                        format!("dwithin(radius={:.1}m)", radius_meters)
                    }
                    super::physical::SpatialScanKind::Range { .. } => "range".to_string(),
                };
                let mut details = vec![
                    ("table_id".to_string(), format!("{}", table_id.0)),
                    ("index_id".to_string(), format!("{}", index_id.0)),
                    ("columns".to_string(), format!("{}", columns.len())),
                    ("kind".to_string(), kind_str),
                ];
                push_as_of(&mut details, as_of);
                Self {
                    operator_name: "SpatialScan".to_string(),
                    details,
                    estimated_cost: Some(*cost),
                    actual_metrics: None,
                    children: Vec::new(),
                }
            }
            PhysicalPlan::GraphAlgorithm {
                schema_name,
                algorithm,
                cost,
                ..
            } => Self {
                operator_name: "GraphAlgorithm".to_string(),
                details: vec![
                    ("schema".to_string(), schema_name.clone()),
                    ("algorithm".to_string(), format!("{:?}", algorithm)),
                ],
                estimated_cost: Some(*cost),
                actual_metrics: None,
                children: Vec::new(),
            },
            PhysicalPlan::AnalyticsTableFunction {
                function_name,
                named_args,
                positional_args,
                cost,
                ..
            } => Self {
                operator_name: "AnalyticsTableFunction".to_string(),
                details: vec![
                    ("function".to_string(), function_name.clone()),
                    ("named_args".to_string(), format!("{}", named_args.len())),
                    (
                        "positional_args".to_string(),
                        format!("{}", positional_args.len()),
                    ),
                ],
                estimated_cost: Some(*cost),
                actual_metrics: None,
                children: Vec::new(),
            },
            PhysicalPlan::Window {
                window_exprs,
                child,
                cost,
                ..
            } => Self {
                operator_name: "Window".to_string(),
                details: vec![("functions".to_string(), format!("{}", window_exprs.len()))],
                estimated_cost: Some(*cost),
                actual_metrics: None,
                children: vec![Self::from_physical_plan(child)],
            },
        }
    }

    /// Merges actual execution metrics into this node and its children.
    /// Metrics are matched by tree position (pre-order traversal).
    /// Attaches measured metrics to the plan nodes they belong to.
    ///
    /// Walks both trees together and attaches only where the operator names
    /// agree. A subtree the executor shaped differently from the plan, which
    /// happens whenever an operator expands into more than one or collapses
    /// into none, is left without actuals rather than given someone else's.
    /// Returns how many nodes were attached, so a caller can tell a plan
    /// that measured nothing from one that measured everything.
    pub fn merge_metrics(&mut self, metrics: &NodeMetrics) -> usize {
        if self.operator_name != metrics.name {
            return 0;
        }
        self.actual_metrics = Some(ActualMetrics {
            rows: metrics.rows,
            elapsed_ns: metrics.elapsed_ns,
            batches: metrics.batches,
            aux: metrics.aux,
        });
        let mut attached = 1;
        for (child, child_metrics) in self.children.iter_mut().zip(metrics.children.iter()) {
            attached += child.merge_metrics(child_metrics);
        }
        attached
    }

    /// Renders the explain output in the specified format.
    pub fn render(&self, options: &ExplainOptions) -> String {
        match options.format {
            ExplainFormat::Text => self.to_text(options),
            ExplainFormat::Json => self.to_json(options),
            ExplainFormat::Yaml => self.to_yaml(options),
        }
    }

    // -----------------------------------------------------------------------
    // Text format
    // -----------------------------------------------------------------------

    fn to_text(&self, options: &ExplainOptions) -> String {
        let mut output = String::with_capacity(self.render_capacity(96));
        self.write_text_node(&mut output, options, 0);
        output
    }

    /// Nodes in this subtree, so a render reserves its buffer once instead of
    /// growing it by doubling. The doubling also made ANALYZE look more
    /// expensive than it is, since its longer output crossed one more growth
    /// step than the same plan rendered without it
    fn node_count(&self) -> usize {
        1 + self
            .children
            .iter()
            .map(|child| child.node_count())
            .sum::<usize>()
    }

    /// Buffer size to reserve for a render, at `per_node` bytes a node.
    fn render_capacity(&self, per_node: usize) -> usize {
        self.node_count() * per_node
    }

    fn write_text_node(&self, output: &mut String, options: &ExplainOptions, depth: usize) {
        let indent = if depth == 0 {
            String::new()
        } else {
            format!("{}-> ", "  ".repeat(depth))
        };

        let _ = write!(output, "{}{}", indent, self.operator_name);

        // Details
        for (key, value) in &self.details {
            let _ = write!(output, " {}={}", key, value);
        }

        // Estimated cost
        if options.costs {
            if let Some(cost) = &self.estimated_cost {
                let _ = write!(
                    output,
                    " (cost={:.2} rows={:.0})",
                    cost.total(),
                    cost.row_count
                );
            }
        }

        // Actual metrics (ANALYZE) folded into the trailing newline so the
        // analyze path is a single writeln! instead of two write!s plus a
        // writeln!, on tiny plans the per-call fmt::Arguments setup cost
        // otherwise shows up as a multi-percent rendering overhead
        match (options.analyze, &self.actual_metrics, options.timing) {
            (true, Some(actual), true) => {
                let (ms, frac) = millis_parts(actual.elapsed_ns);
                let _ = writeln!(
                    output,
                    " (actual rows={} time={}.{:03}ms)",
                    actual.rows, ms, frac
                );
            }
            (true, Some(actual), false) => {
                let _ = writeln!(output, " (actual rows={})", actual.rows);
            }
            _ => {
                let _ = writeln!(output);
            }
        }

        // Operator-specific counters on their own line, so the pruning a
        // scan actually did is visible rather than inferable from the row
        // count it produced
        if options.analyze {
            if let Some(actual) = &self.actual_metrics {
                let labels = aux_labels(&self.operator_name);
                if labels.iter().any(|l| !l.is_empty()) {
                    let mut line = String::new();
                    for (label, value) in labels.iter().zip(actual.aux.iter()) {
                        if label.is_empty() {
                            continue;
                        }
                        if !line.is_empty() {
                            line.push(' ');
                        }
                        let _ = write!(line, "{}={}", label, value);
                    }
                    if !line.is_empty() {
                        let _ = writeln!(output, "{}  {}", indent, line);
                    }
                }
            }
        }

        for child in &self.children {
            child.write_text_node(output, options, depth + 1);
        }
    }

    // -----------------------------------------------------------------------
    // JSON format
    // -----------------------------------------------------------------------

    fn to_json(&self, options: &ExplainOptions) -> String {
        let mut output = String::with_capacity(self.render_capacity(256));
        self.write_json_node(&mut output, options, 0);
        let _ = writeln!(output);
        output
    }

    fn write_json_node(&self, output: &mut String, options: &ExplainOptions, depth: usize) {
        let pad = "  ".repeat(depth);
        let _ = writeln!(output, "{}{{", pad);
        let _ = writeln!(output, "{}  \"operator\": \"{}\",", pad, self.operator_name);

        // Details
        if !self.details.is_empty() {
            let _ = write!(output, "{}  \"details\": {{", pad);
            for (i, (key, value)) in self.details.iter().enumerate() {
                if i > 0 {
                    let _ = write!(output, ", ");
                }
                let _ = write!(output, "\"{}\": \"{}\"", key, value);
            }
            let _ = writeln!(output, "}},");
        }

        // Estimated cost
        if options.costs {
            if let Some(cost) = &self.estimated_cost {
                let _ = writeln!(output, "{}  \"estimated_cost\": {:.2},", pad, cost.total());
                let _ = writeln!(
                    output,
                    "{}  \"estimated_rows\": {:.0},",
                    pad, cost.row_count
                );
            }
        }

        // Actual metrics
        if options.analyze {
            if let Some(actual) = &self.actual_metrics {
                let _ = writeln!(output, "{}  \"actual_rows\": {},", pad, actual.rows);
                if options.timing {
                    let (ms, frac) = millis_parts(actual.elapsed_ns);
                    let _ = writeln!(output, "{}  \"actual_time_ms\": {}.{:03},", pad, ms, frac);
                }
            }
        }

        // Children
        if self.children.is_empty() {
            let _ = writeln!(output, "{}  \"children\": []", pad);
        } else {
            let _ = writeln!(output, "{}  \"children\": [", pad);
            for (i, child) in self.children.iter().enumerate() {
                child.write_json_node(output, options, depth + 2);
                if i < self.children.len() - 1 {
                    let _ = write!(output, ",");
                }
                let _ = writeln!(output);
            }
            let _ = writeln!(output, "{}  ]", pad);
        }

        let _ = write!(output, "{}}}", pad);
    }

    // -----------------------------------------------------------------------
    // YAML format
    // -----------------------------------------------------------------------

    fn to_yaml(&self, options: &ExplainOptions) -> String {
        let mut output = String::with_capacity(self.render_capacity(192));
        self.write_yaml_node(&mut output, options, 0);
        output
    }

    fn write_yaml_node(&self, output: &mut String, options: &ExplainOptions, depth: usize) {
        let pad = "  ".repeat(depth);
        let _ = writeln!(output, "{}operator: {}", pad, self.operator_name);

        for (key, value) in &self.details {
            let _ = writeln!(output, "{}{}: {}", pad, key, value);
        }

        if options.costs {
            if let Some(cost) = &self.estimated_cost {
                let _ = writeln!(output, "{}estimated_cost: {:.2}", pad, cost.total());
                let _ = writeln!(output, "{}estimated_rows: {:.0}", pad, cost.row_count);
            }
        }

        if options.analyze {
            if let Some(actual) = &self.actual_metrics {
                let _ = writeln!(output, "{}actual_rows: {}", pad, actual.rows);
                if options.timing {
                    let (ms, frac) = millis_parts(actual.elapsed_ns);
                    let _ = writeln!(output, "{}actual_time_ms: {}.{:03}", pad, ms, frac);
                }
                for (label, value) in aux_labels(&self.operator_name)
                    .iter()
                    .zip(actual.aux.iter())
                {
                    if !label.is_empty() {
                        let _ = writeln!(output, "{}{}: {}", pad, label, value);
                    }
                }
            }
        }

        if !self.children.is_empty() {
            let _ = writeln!(output, "{}children:", pad);
            for child in &self.children {
                let _ = write!(output, "{}  - operator: {}\n", pad, child.operator_name);
                for (key, value) in &child.details {
                    let _ = writeln!(output, "{}    {}: {}", pad, key, value);
                }
                if options.costs {
                    if let Some(cost) = &child.estimated_cost {
                        let _ = writeln!(output, "{}    estimated_cost: {:.2}", pad, cost.total());
                        let _ =
                            writeln!(output, "{}    estimated_rows: {:.0}", pad, cost.row_count);
                    }
                }
                if options.analyze {
                    if let Some(actual) = &child.actual_metrics {
                        let _ = writeln!(output, "{}    actual_rows: {}", pad, actual.rows);
                        if options.timing {
                            let (ms, frac) = millis_parts(actual.elapsed_ns);
                            let _ =
                                writeln!(output, "{}    actual_time_ms: {}.{:03}", pad, ms, frac);
                        }
                    }
                }
                if !child.children.is_empty() {
                    let _ = writeln!(output, "{}    children:", pad);
                    for grandchild in &child.children {
                        grandchild.write_yaml_node(output, options, depth + 4);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Every digit the renderers print has to match what `{:.3}` printed for
    /// the same count as an f64 millisecond value. The cases that matter are
    /// exact half microseconds, where the answer is decided by which side of
    /// the tie the division landed on rather than by any rounding rule
    #[test]
    fn millis_parts_matches_float_formatting_exactly() {
        let check = |ns: u64| {
            let (ms, frac) = millis_parts(ns);
            assert_eq!(
                format!("{}.{:03}", ms, frac),
                format!("{:.3}", ns as f64 / 1_000_000.0),
                "ns={ns}"
            );
        };

        // Every nanosecond across the first two milliseconds, covering every
        // tie and every carry in that range
        for ns in 0u64..2_000_000 {
            check(ns);
        }
        // Exact half microseconds much further out, where float spacing is
        // coarser and ties resolve the other way
        for k in 0u64..100_000 {
            check(k * 1_000 + 500);
            check(k * 1_000_000 + 500);
        }
        // Magnitudes up to the width of the field itself
        for shift in 0..64 {
            let base = 1u64 << shift;
            for delta in [0u64, 1, 499, 500, 501, 999] {
                check(base.saturating_add(delta));
                check(base.saturating_sub(delta));
            }
        }
        check(u64::MAX);
    }

    fn make_simple_plan() -> ExplainNode {
        ExplainNode {
            operator_name: "HashJoin".to_string(),
            details: vec![("join_type".to_string(), "Inner".to_string())],
            estimated_cost: Some(PlanCost {
                io_cost: 10.0,
                cpu_cost: 50.0,
                row_count: 1000.0,
            }),
            actual_metrics: None,
            children: vec![
                ExplainNode {
                    operator_name: "SeqScan".to_string(),
                    details: vec![("table_id".to_string(), "1".to_string())],
                    estimated_cost: Some(PlanCost {
                        io_cost: 5.0,
                        cpu_cost: 20.0,
                        row_count: 5000.0,
                    }),
                    actual_metrics: None,
                    children: Vec::new(),
                },
                ExplainNode {
                    operator_name: "IndexScan".to_string(),
                    details: vec![("table_id".to_string(), "2".to_string())],
                    estimated_cost: Some(PlanCost {
                        io_cost: 2.0,
                        cpu_cost: 10.0,
                        row_count: 200.0,
                    }),
                    actual_metrics: None,
                    children: Vec::new(),
                },
            ],
        }
    }

    #[test]
    fn test_text_output() {
        let node = make_simple_plan();
        let options = ExplainOptions::default();
        let text = node.render(&options);
        assert!(text.contains("HashJoin"));
        assert!(text.contains("SeqScan"));
        assert!(text.contains("IndexScan"));
        assert!(text.contains("cost="));
        assert!(text.contains("rows="));
    }

    #[test]
    fn test_text_no_costs() {
        let node = make_simple_plan();
        let options = ExplainOptions {
            costs: false,
            ..Default::default()
        };
        let text = node.render(&options);
        assert!(text.contains("HashJoin"));
        assert!(!text.contains("cost="));
    }

    #[test]
    fn test_text_with_analyze() {
        let mut node = make_simple_plan();
        node.actual_metrics = Some(ActualMetrics {
            rows: 982,
            elapsed_ns: 3_200_000,
            batches: 5,
            ..Default::default()
        });
        let options = ExplainOptions {
            analyze: true,
            ..Default::default()
        };
        let text = node.render(&options);
        assert!(text.contains("actual rows=982"));
        assert!(text.contains("time=3.200ms"));
    }

    #[test]
    fn test_json_output() {
        let node = make_simple_plan();
        let options = ExplainOptions {
            format: ExplainFormat::Json,
            ..Default::default()
        };
        let json = node.render(&options);
        assert!(json.contains("\"operator\": \"HashJoin\""));
        assert!(json.contains("\"children\""));
    }

    #[test]
    fn test_yaml_output() {
        let node = make_simple_plan();
        let options = ExplainOptions {
            format: ExplainFormat::Yaml,
            ..Default::default()
        };
        let yaml = node.render(&options);
        assert!(yaml.contains("operator: HashJoin"));
        assert!(yaml.contains("children:"));
    }

    fn measured(name: &str, rows: u64, children: Vec<NodeMetrics>) -> NodeMetrics {
        NodeMetrics {
            name: name.to_string(),
            rows,
            elapsed_ns: 1_000_000,
            batches: 2,
            aux: [0; ACTUAL_AUX_SLOTS],
            children,
        }
    }

    #[test]
    fn test_merge_metrics_attaches_every_matching_node() {
        let mut node = make_simple_plan();
        let names: Vec<String> = node
            .children
            .iter()
            .map(|c| c.operator_name.clone())
            .collect();
        let metrics = measured(
            &node.operator_name,
            1000,
            vec![
                measured(&names[0], 5000, vec![]),
                measured(&names[1], 200, vec![]),
            ],
        );
        assert_eq!(node.merge_metrics(&metrics), 3);
        assert_eq!(node.actual_metrics.as_ref().map(|m| m.rows), Some(1000));
        assert_eq!(
            node.children[0].actual_metrics.as_ref().map(|m| m.rows),
            Some(5000)
        );
        assert_eq!(
            node.children[1].actual_metrics.as_ref().map(|m| m.rows),
            Some(200)
        );
    }

    /// The reason the merge is structural: a shape the executor built
    /// differently must leave a node without actuals rather than give it
    /// another operator's numbers
    #[test]
    fn test_a_shape_mismatch_attaches_nothing_rather_than_the_wrong_numbers() {
        let mut node = make_simple_plan();
        let metrics = measured("SomeOtherOperator", 1000, vec![]);
        assert_eq!(node.merge_metrics(&metrics), 0);
        assert!(node.actual_metrics.is_none());

        // A root that matches with children that do not: the root gets its
        // own numbers and the children get none
        let mut node = make_simple_plan();
        let metrics = measured(
            &node.operator_name,
            1000,
            vec![measured("Mismatch", 5000, vec![])],
        );
        assert_eq!(node.merge_metrics(&metrics), 1);
        assert_eq!(node.actual_metrics.as_ref().map(|m| m.rows), Some(1000));
        assert!(node.children[0].actual_metrics.is_none());
        assert!(node.children[1].actual_metrics.is_none());
    }

    /// Auxiliary counters render under the label their operator gives them
    #[test]
    fn test_lake_scan_reports_its_pruning_counters() {
        let mut node = ExplainNode {
            operator_name: "LakeScan".to_string(),
            details: Vec::new(),
            estimated_cost: None,
            actual_metrics: None,
            children: Vec::new(),
        };
        let mut metrics = measured("LakeScan", 12, vec![]);
        metrics.aux = [40, 37, 4_096_000, 3_788_800, 1, 6];
        assert_eq!(node.merge_metrics(&metrics), 1);
        let text = node.render(&ExplainOptions {
            analyze: true,
            ..Default::default()
        });
        assert!(text.contains("files_considered=40"), "{text}");
        assert!(text.contains("files_pruned=37"), "{text}");
        assert!(text.contains("bytes_pruned=3788800"), "{text}");
        // Which access path answered, so a plan distinguishes a scan whose
        // statistics were enough from one an index addressed
        assert!(text.contains("index_files_read=1"), "{text}");
        assert!(text.contains("index_rows_addressed=6"), "{text}");

        // An operator that fills no aux slot prints none of them
        let mut plain = make_simple_plan();
        let plain_metrics = measured(&plain.operator_name, 5, vec![]);
        plain.merge_metrics(&plain_metrics);
        let text = plain.render(&ExplainOptions {
            analyze: true,
            ..Default::default()
        });
        assert!(!text.contains("files_considered"));
    }

    #[test]
    fn test_explain_format_from_str() {
        assert_eq!(ExplainFormat::from_str("json"), ExplainFormat::Json);
        assert_eq!(ExplainFormat::from_str("JSON"), ExplainFormat::Json);
        assert_eq!(ExplainFormat::from_str("yaml"), ExplainFormat::Yaml);
        assert_eq!(ExplainFormat::from_str("text"), ExplainFormat::Text);
        assert_eq!(ExplainFormat::from_str("anything"), ExplainFormat::Text);
    }
}

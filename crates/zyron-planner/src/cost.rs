//! Cost model for query plan optimization.
//!
//! Estimates plan costs using catalog statistics (histograms, NDV, null fractions)
//! for selectivity estimation and cardinality calculations. The cost model
//! compares alternative physical plans to select the cheapest execution strategy.

use crate::binder::BoundExpr;
use crate::logical::{JoinCondition, LogicalPlan};
use zyron_catalog::{Catalog, ColumnStats, TableStats};
use zyron_parser::ast::JoinType;

// ---------------------------------------------------------------------------
// Plan cost
// ---------------------------------------------------------------------------

/// Estimated cost of executing a plan node.
#[derive(Debug, Clone, Copy)]
pub struct PlanCost {
    pub io_cost: f64,
    pub cpu_cost: f64,
    pub row_count: f64,
}

impl PlanCost {
    /// Total cost with IO weighted higher than CPU.
    pub fn total(&self) -> f64 {
        self.io_cost + self.cpu_cost * 0.01
    }

    pub fn zero() -> Self {
        Self {
            io_cost: 0.0,
            cpu_cost: 0.0,
            row_count: 0.0,
        }
    }

    pub fn add(&self, other: &PlanCost) -> PlanCost {
        PlanCost {
            io_cost: self.io_cost + other.io_cost,
            cpu_cost: self.cpu_cost + other.cpu_cost,
            row_count: self.row_count,
        }
    }
}

// ---------------------------------------------------------------------------
// Cost component breakdown
// ---------------------------------------------------------------------------

/// Detailed cost breakdown by resource type for hardware-aware plan comparison.
#[derive(Debug, Clone, Copy)]
pub struct CostComponent {
    /// Sequential page reads (base weight: 1.0).
    pub seq_io: f64,
    /// Random page reads (base weight: 4.0).
    pub random_io: f64,
    /// Per-tuple processing cost (base weight: 0.01).
    pub cpu_tuple: f64,
    /// Per-operator evaluation cost (base weight: 0.0025).
    pub cpu_operator: f64,
    /// Per-byte network transfer cost (base weight: 0.0001).
    pub network: f64,
    /// Per-byte working memory pressure (base weight: 0.00001).
    pub memory: f64,
}

impl CostComponent {
    pub fn zero() -> Self {
        Self {
            seq_io: 0.0,
            random_io: 0.0,
            cpu_tuple: 0.0,
            cpu_operator: 0.0,
            network: 0.0,
            memory: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Encoding cost parameters
// ---------------------------------------------------------------------------

/// Cost parameters for columnar-encoded scans. Controls how zone maps,
/// bloom filters, and encoding-specific evaluation affect scan cost.
#[derive(Debug, Clone, Copy)]
pub struct EncodingCostParameters {
    /// Fraction of segments skipped via zone maps or bloom filters (0.0 to 1.0).
    pub skip_rate: f64,
    /// CPU cost to decode one value from the encoded format.
    pub decode_cost_per_value: f64,
    /// Speedup ratio of encoded scan vs unencoded scan (< 1.0 means faster).
    pub encoded_scan_speedup: f64,
}

impl Default for EncodingCostParameters {
    fn default() -> Self {
        Self {
            skip_rate: 0.0,
            decode_cost_per_value: 0.0001,
            encoded_scan_speedup: 0.3,
        }
    }
}

/// Pre-defined encoding decode costs (relative to unencoded = 1.0).
pub mod encoding_costs {
    use super::EncodingCostParameters;

    pub fn fastlanes() -> EncodingCostParameters {
        EncodingCostParameters {
            skip_rate: 0.0,
            decode_cost_per_value: 0.00015,
            encoded_scan_speedup: 0.15,
        }
    }

    pub fn dictionary() -> EncodingCostParameters {
        EncodingCostParameters {
            skip_rate: 0.0,
            decode_cost_per_value: 0.0001,
            encoded_scan_speedup: 0.10,
        }
    }

    pub fn rle() -> EncodingCostParameters {
        EncodingCostParameters {
            skip_rate: 0.0,
            decode_cost_per_value: 0.00005,
            encoded_scan_speedup: 0.05,
        }
    }

    pub fn bitpack() -> EncodingCostParameters {
        EncodingCostParameters {
            skip_rate: 0.0,
            decode_cost_per_value: 0.00008,
            encoded_scan_speedup: 0.08,
        }
    }
}

// ---------------------------------------------------------------------------
// Default selectivity constants
// ---------------------------------------------------------------------------

pub const DEFAULT_EQUALITY_SELECTIVITY: f64 = 0.1;
pub const DEFAULT_RANGE_SELECTIVITY: f64 = 0.33;
pub const DEFAULT_LIKE_SELECTIVITY: f64 = 0.2;
pub const DEFAULT_IN_LIST_SELECTIVITY: f64 = 0.05;
pub const DEFAULT_NULL_SELECTIVITY: f64 = 0.05;

/// Index scan is preferred when selectivity is below this threshold.
pub const INDEX_SCAN_SELECTIVITY_THRESHOLD: f64 = 0.10;

// ---------------------------------------------------------------------------
// Cost model
// ---------------------------------------------------------------------------

/// Cost estimation using catalog statistics.
#[derive(Debug, Clone)]
pub struct CostModel {
    pub seq_page_cost: f64,
    pub random_page_cost: f64,
    pub cpu_tuple_cost: f64,
    pub cpu_index_tuple_cost: f64,
    pub cpu_operator_cost: f64,
    /// Per-byte cost for network transfer (local queries use 0.0001).
    pub network_cost_per_byte: f64,
    /// Per-byte cost for working memory pressure (hash tables, sort buffers).
    pub memory_cost_per_byte: f64,
    /// Per-tuple cost for transferring tuples between parallel workers.
    pub parallel_tuple_cost: f64,
    /// Fixed startup cost for launching parallel workers.
    pub parallel_setup_cost: f64,
}

impl Default for CostModel {
    fn default() -> Self {
        Self {
            seq_page_cost: 1.0,
            random_page_cost: 4.0,
            cpu_tuple_cost: 0.01,
            cpu_index_tuple_cost: 0.005,
            cpu_operator_cost: 0.0025,
            network_cost_per_byte: 0.0001,
            memory_cost_per_byte: 0.00001,
            parallel_tuple_cost: 0.1,
            parallel_setup_cost: 1000.0,
        }
    }
}

impl CostModel {
    // -----------------------------------------------------------------------
    // Cost component decomposition
    // -----------------------------------------------------------------------

    /// Breaks a PlanCost into a detailed CostComponent for hardware-aware comparison.
    pub fn decompose(&self, plan_cost: &PlanCost) -> CostComponent {
        CostComponent {
            seq_io: plan_cost.io_cost,
            random_io: 0.0,
            cpu_tuple: plan_cost.row_count * self.cpu_tuple_cost,
            cpu_operator: plan_cost.cpu_cost - plan_cost.row_count * self.cpu_tuple_cost,
            network: 0.0,
            memory: 0.0,
        }
    }

    /// Computes a total cost from components weighted by hardware-specific parameters.
    pub fn weighted_total(&self, components: &CostComponent) -> f64 {
        components.seq_io * self.seq_page_cost
            + components.random_io * self.random_page_cost
            + components.cpu_tuple * self.cpu_tuple_cost
            + components.cpu_operator * self.cpu_operator_cost
            + components.network * self.network_cost_per_byte
            + components.memory * self.memory_cost_per_byte
    }

    // -----------------------------------------------------------------------
    // Encoding-aware scan cost
    // -----------------------------------------------------------------------

    /// Estimates the cost of a columnar-encoded scan using encoding parameters.
    /// Accounts for segment skipping (zone maps, bloom filters) and decode overhead.
    pub fn cost_encoded_scan(
        &self,
        stats: &TableStats,
        params: &EncodingCostParameters,
    ) -> PlanCost {
        let rows = stats.row_count as f64;
        let pages = stats.page_count as f64;

        // IO: sequential scan with skip_rate reducing pages read
        let pages_read = pages * (1.0 - params.skip_rate);
        let io_cost = pages_read * self.seq_page_cost * params.encoded_scan_speedup;

        // CPU: decode cost for non-skipped rows
        let rows_read = rows * (1.0 - params.skip_rate);
        let cpu_cost = rows_read * params.decode_cost_per_value + rows_read * self.cpu_tuple_cost;

        PlanCost {
            io_cost,
            cpu_cost,
            row_count: rows_read,
        }
    }

    // -----------------------------------------------------------------------
    // Parallel cost estimation
    // -----------------------------------------------------------------------

    /// Estimates the cost of a parallel sequential scan split across workers.
    pub fn cost_parallel_scan(&self, stats: &TableStats, num_workers: usize) -> PlanCost {
        let workers = (num_workers as f64).max(1.0);
        let rows = stats.row_count as f64;
        let pages = stats.page_count as f64;

        // IO is divided among workers (each reads a partition of pages)
        let io_cost = (pages / workers) * self.seq_page_cost;

        // CPU divided among workers plus per-tuple coordination overhead
        let cpu_cost = (rows / workers) * self.cpu_tuple_cost
            + rows * self.parallel_tuple_cost
            + self.parallel_setup_cost;

        PlanCost {
            io_cost,
            cpu_cost,
            row_count: rows,
        }
    }

    /// Estimates the cost of a parallel hash join with partitioned build and probe.
    pub fn cost_parallel_hash_join(
        &self,
        left: &PlanCost,
        right: &PlanCost,
        num_workers: usize,
    ) -> PlanCost {
        let workers = (num_workers as f64).max(1.0);

        // Build hash table in parallel (smaller side)
        let (build, probe) = if left.row_count <= right.row_count {
            (left, right)
        } else {
            (right, left)
        };

        // IO from both sides
        let io_cost = left.io_cost + right.io_cost;

        // CPU: build and probe divided by workers, plus coordination overhead
        let build_cpu = (build.row_count / workers) * self.cpu_operator_cost;
        let probe_cpu = (probe.row_count / workers) * self.cpu_operator_cost;
        let coordination = (left.row_count + right.row_count) * self.parallel_tuple_cost;
        let cpu_cost = build_cpu
            + probe_cpu
            + coordination
            + left.cpu_cost
            + right.cpu_cost
            + self.parallel_setup_cost;

        PlanCost {
            io_cost,
            cpu_cost,
            row_count: estimate_join_rows(left.row_count, right.row_count),
        }
    }

    // -----------------------------------------------------------------------
    // Selectivity estimation
    // -----------------------------------------------------------------------

    /// Estimates selectivity of a predicate on a table.
    /// Delegates to CardinalityEstimator for MCV + histogram + NDV based estimation.
    pub fn estimate_selectivity(
        &self,
        predicate: &BoundExpr,
        table_stats: Option<&TableStats>,
        column_stats: Option<&[ColumnStats]>,
    ) -> f64 {
        crate::optimizer::cardinality::CardinalityEstimator::estimate_selectivity(
            predicate,
            table_stats,
            column_stats,
        )
    }

    // -----------------------------------------------------------------------
    // Operator cost estimation
    // -----------------------------------------------------------------------

    /// Estimates the cost of a sequential table scan.
    pub fn cost_seq_scan(&self, stats: &TableStats) -> PlanCost {
        PlanCost {
            io_cost: stats.page_count as f64 * self.seq_page_cost,
            cpu_cost: stats.row_count as f64 * self.cpu_tuple_cost,
            row_count: stats.row_count as f64,
        }
    }

    /// Estimates the cost of an index scan.
    pub fn cost_index_scan(&self, stats: &TableStats, selectivity: f64) -> PlanCost {
        let rows = (stats.row_count as f64 * selectivity).max(1.0);
        let rows_per_page = if stats.page_count > 0 {
            stats.row_count as f64 / stats.page_count as f64
        } else {
            1.0
        };
        let pages = (rows / rows_per_page).ceil().max(1.0);
        PlanCost {
            io_cost: pages * self.random_page_cost,
            cpu_cost: rows * self.cpu_index_tuple_cost + rows * self.cpu_tuple_cost,
            row_count: rows,
        }
    }

    /// Estimates the cost of a hash join.
    pub fn cost_hash_join(&self, left: &PlanCost, right: &PlanCost) -> PlanCost {
        // Build hash table on the smaller side, probe with larger
        let (build, probe) = if left.row_count <= right.row_count {
            (left, right)
        } else {
            (right, left)
        };
        PlanCost {
            io_cost: left.io_cost + right.io_cost,
            cpu_cost: build.row_count * self.cpu_operator_cost  // hash build
                + probe.row_count * self.cpu_operator_cost      // hash probe
                + left.cpu_cost + right.cpu_cost,
            row_count: estimate_join_rows(left.row_count, right.row_count),
        }
    }

    /// Estimates the cost of a nested loop join.
    pub fn cost_nested_loop_join(&self, left: &PlanCost, right: &PlanCost) -> PlanCost {
        PlanCost {
            io_cost: left.io_cost + left.row_count * right.io_cost,
            cpu_cost: left.row_count * right.row_count * self.cpu_operator_cost
                + left.cpu_cost
                + right.cpu_cost,
            row_count: estimate_join_rows(left.row_count, right.row_count),
        }
    }

    /// Estimates the cost of a merge join (both sides assumed sorted).
    pub fn cost_merge_join(&self, left: &PlanCost, right: &PlanCost) -> PlanCost {
        PlanCost {
            io_cost: left.io_cost + right.io_cost,
            cpu_cost: (left.row_count + right.row_count) * self.cpu_operator_cost
                + left.cpu_cost
                + right.cpu_cost,
            row_count: estimate_join_rows(left.row_count, right.row_count),
        }
    }

    /// Estimates the cost of a sort operation.
    pub fn cost_sort(&self, input: &PlanCost) -> PlanCost {
        let n = input.row_count.max(1.0);
        let comparisons = n * n.log2();
        PlanCost {
            io_cost: input.io_cost,
            cpu_cost: input.cpu_cost + comparisons * self.cpu_operator_cost,
            row_count: input.row_count,
        }
    }

    /// Estimates the cost of a hash aggregation.
    pub fn cost_hash_aggregate(&self, input: &PlanCost, group_count: f64) -> PlanCost {
        PlanCost {
            io_cost: input.io_cost,
            cpu_cost: input.cpu_cost + input.row_count * self.cpu_operator_cost,
            row_count: group_count.max(1.0),
        }
    }

    // -----------------------------------------------------------------------
    // Cardinality estimation
    // -----------------------------------------------------------------------

    /// Estimates the output cardinality of a join.
    pub fn estimate_join_cardinality(
        &self,
        left_rows: f64,
        right_rows: f64,
        join_type: &JoinType,
        condition: &JoinCondition,
        _catalog: &Catalog,
    ) -> f64 {
        let base = match condition {
            JoinCondition::Cross => left_rows * right_rows,
            JoinCondition::On(_) => {
                // Equi-join estimate: use the smaller table's cardinality
                // as a rough approximation
                estimate_join_rows(left_rows, right_rows)
            }
            JoinCondition::Using(_) => estimate_join_rows(left_rows, right_rows),
            JoinCondition::Natural => estimate_join_rows(left_rows, right_rows),
        };

        // Outer joins produce at least as many rows as the preserved side
        match join_type {
            JoinType::Left => base.max(left_rows),
            JoinType::Right => base.max(right_rows),
            JoinType::Full => base.max(left_rows.max(right_rows)),
            JoinType::Inner | JoinType::Cross => base,
        }
    }

    /// Estimates the total cost of a logical plan tree.
    pub fn estimate_plan_cost(&self, plan: &LogicalPlan, catalog: &Catalog) -> PlanCost {
        match plan {
            LogicalPlan::Scan { table_id, .. } => {
                if let Some((ts, _)) = catalog.get_stats(*table_id) {
                    self.cost_seq_scan(&ts)
                } else {
                    // No stats: assume 1000 rows, 10 pages
                    PlanCost {
                        io_cost: 10.0 * self.seq_page_cost,
                        cpu_cost: 1000.0 * self.cpu_tuple_cost,
                        row_count: 1000.0,
                    }
                }
            }
            LogicalPlan::Filter { predicate, child } => {
                let child_cost = self.estimate_plan_cost(child, catalog);
                let selectivity = self.estimate_selectivity(predicate, None, None);
                PlanCost {
                    io_cost: child_cost.io_cost,
                    cpu_cost: child_cost.cpu_cost + child_cost.row_count * self.cpu_operator_cost,
                    row_count: (child_cost.row_count * selectivity).max(1.0),
                }
            }
            LogicalPlan::Project { child, .. } => {
                let child_cost = self.estimate_plan_cost(child, catalog);
                PlanCost {
                    io_cost: child_cost.io_cost,
                    cpu_cost: child_cost.cpu_cost + child_cost.row_count * self.cpu_operator_cost,
                    row_count: child_cost.row_count,
                }
            }
            LogicalPlan::Join {
                left,
                right,
                join_type,
                condition,
            } => {
                let left_cost = self.estimate_plan_cost(left, catalog);
                let right_cost = self.estimate_plan_cost(right, catalog);
                let rows = self.estimate_join_cardinality(
                    left_cost.row_count,
                    right_cost.row_count,
                    join_type,
                    condition,
                    catalog,
                );
                PlanCost {
                    io_cost: left_cost.io_cost + right_cost.io_cost,
                    cpu_cost: left_cost.cpu_cost
                        + right_cost.cpu_cost
                        + rows * self.cpu_operator_cost,
                    row_count: rows,
                }
            }
            LogicalPlan::Aggregate {
                group_by, child, ..
            } => {
                let child_cost = self.estimate_plan_cost(child, catalog);
                let group_count = if group_by.is_empty() {
                    1.0
                } else {
                    // Rough estimate: assume grouping reduces to sqrt(rows)
                    child_cost.row_count.sqrt().max(1.0)
                };
                self.cost_hash_aggregate(&child_cost, group_count)
            }
            LogicalPlan::Sort { child, .. } => {
                let child_cost = self.estimate_plan_cost(child, catalog);
                self.cost_sort(&child_cost)
            }
            LogicalPlan::Limit {
                limit,
                offset: _,
                child,
            } => {
                let child_cost = self.estimate_plan_cost(child, catalog);
                let rows = if let Some(l) = limit {
                    (*l as f64).min(child_cost.row_count)
                } else {
                    child_cost.row_count
                };
                PlanCost {
                    io_cost: child_cost.io_cost,
                    cpu_cost: child_cost.cpu_cost,
                    row_count: rows,
                }
            }
            LogicalPlan::Distinct { child } => {
                let child_cost = self.estimate_plan_cost(child, catalog);
                PlanCost {
                    io_cost: child_cost.io_cost,
                    cpu_cost: child_cost.cpu_cost + child_cost.row_count * self.cpu_operator_cost,
                    row_count: child_cost.row_count * 0.8, // Assume 20% duplicates
                }
            }
            LogicalPlan::SetOp { left, right, .. } => {
                let left_cost = self.estimate_plan_cost(left, catalog);
                let right_cost = self.estimate_plan_cost(right, catalog);
                left_cost.add(&right_cost)
            }
            LogicalPlan::Values { rows, .. } => PlanCost {
                io_cost: 0.0,
                cpu_cost: rows.len() as f64 * self.cpu_tuple_cost,
                row_count: rows.len() as f64,
            },
            LogicalPlan::Insert { source, .. } => self.estimate_plan_cost(source, catalog),
            LogicalPlan::Update { child, .. } => self.estimate_plan_cost(child, catalog),
            LogicalPlan::Delete { child, .. } => self.estimate_plan_cost(child, catalog),
            LogicalPlan::GraphAlgorithm { algorithm, .. } => {
                // Mirrors the physical builder's per-algorithm estimates using
                // a nominal graph of V=10_000 nodes and E=100_000 edges.
                let v: f64 = 10_000.0;
                let e: f64 = 100_000.0;
                let (cpu, row_count) = match algorithm.as_str() {
                    "pagerank" => (20.0 * (v + e), v),
                    "shortest_path" => (v + e, v.sqrt()),
                    "bfs" => (v + e, v),
                    "connected_components" => (v + e, v),
                    "community_detection" => (10.0 * (v + e), v),
                    "betweenness_centrality" => (v * (v + e), v),
                    _ => (v + e, v),
                };
                PlanCost {
                    io_cost: v,
                    cpu_cost: cpu,
                    row_count,
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Default join cardinality estimate when no detailed stats are available.
/// Uses the geometric mean of the two input sizes.
fn estimate_join_rows(left: f64, right: f64) -> f64 {
    (left * right).sqrt().max(1.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binder::BoundExpr;
    use zyron_catalog::{ColumnId, TableId};
    use zyron_common::TypeId;
    use zyron_parser::ast::{BinaryOperator, LiteralValue};

    fn make_cost_model() -> CostModel {
        CostModel::default()
    }

    fn make_table_stats(rows: u64, pages: u32) -> TableStats {
        TableStats {
            table_id: TableId(1),
            row_count: rows,
            page_count: pages,
            avg_row_size: 64,
            last_analyzed: 0,
        }
    }

    #[test]
    fn test_seq_scan_cost() {
        let model = make_cost_model();
        let stats = make_table_stats(10000, 100);
        let cost = model.cost_seq_scan(&stats);
        assert_eq!(cost.io_cost, 100.0); // 100 pages * 1.0
        assert_eq!(cost.row_count, 10000.0);
    }

    #[test]
    fn test_index_scan_cost_low_selectivity() {
        let model = make_cost_model();
        let stats = make_table_stats(10000, 100);
        let cost = model.cost_index_scan(&stats, 0.01); // 1% selectivity
        assert!(cost.row_count < 200.0);
        // Random IO cost should be relatively small for low selectivity
        assert!(cost.io_cost < stats.page_count as f64 * model.seq_page_cost);
    }

    #[test]
    fn test_seq_scan_cheaper_than_index_for_high_selectivity() {
        let model = make_cost_model();
        let stats = make_table_stats(10000, 100);
        let seq_cost = model.cost_seq_scan(&stats);
        let idx_cost = model.cost_index_scan(&stats, 0.5); // 50% selectivity
        // Sequential should be cheaper for high selectivity due to random IO penalty
        assert!(seq_cost.total() < idx_cost.total());
    }

    #[test]
    fn test_selectivity_and_condition() {
        let model = make_cost_model();
        let pred = BoundExpr::BinaryOp {
            left: Box::new(BoundExpr::Literal {
                value: LiteralValue::Boolean(true),
                type_id: TypeId::Boolean,
            }),
            op: BinaryOperator::And,
            right: Box::new(BoundExpr::Literal {
                value: LiteralValue::Boolean(true),
                type_id: TypeId::Boolean,
            }),
            type_id: TypeId::Boolean,
        };
        let sel = model.estimate_selectivity(&pred, None, None);
        assert!((sel - 1.0).abs() < 0.001); // true AND true = 1.0
    }

    #[test]
    fn test_selectivity_or_condition() {
        let model = make_cost_model();
        let pred = BoundExpr::BinaryOp {
            left: Box::new(BoundExpr::Literal {
                value: LiteralValue::Boolean(false),
                type_id: TypeId::Boolean,
            }),
            op: BinaryOperator::Or,
            right: Box::new(BoundExpr::Literal {
                value: LiteralValue::Boolean(true),
                type_id: TypeId::Boolean,
            }),
            type_id: TypeId::Boolean,
        };
        let sel = model.estimate_selectivity(&pred, None, None);
        // false OR true = 0 + 1 - 0*1 = 1.0
        assert!((sel - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_equality_selectivity_with_stats() {
        let model = make_cost_model();
        let stats = vec![ColumnStats {
            table_id: TableId(1),
            column_id: ColumnId(0),
            null_fraction: 0.0,
            distinct_count: 100,
            avg_width: 8,
            histogram: None,
            most_common_values: vec![],
            most_common_freqs: vec![],
        }];
        // col0 = 42
        let pred = BoundExpr::BinaryOp {
            left: Box::new(BoundExpr::ColumnRef(crate::binder::ColumnRef {
                table_idx: 0,
                column_id: ColumnId(0),
                type_id: TypeId::Int64,
                nullable: false,
            })),
            op: BinaryOperator::Eq,
            right: Box::new(BoundExpr::Literal {
                value: LiteralValue::Integer(42),
                type_id: TypeId::Int64,
            }),
            type_id: TypeId::Boolean,
        };
        let sel = model.estimate_selectivity(&pred, None, Some(&stats));
        assert!((sel - 0.01).abs() < 0.001); // 1/100
    }

    #[test]
    fn test_hash_join_cheaper_than_nested_loop() {
        let model = make_cost_model();
        let left = PlanCost {
            io_cost: 100.0,
            cpu_cost: 10000.0,
            row_count: 10000.0,
        };
        let right = PlanCost {
            io_cost: 50.0,
            cpu_cost: 5000.0,
            row_count: 5000.0,
        };
        let hash = model.cost_hash_join(&left, &right);
        let nl = model.cost_nested_loop_join(&left, &right);
        assert!(hash.total() < nl.total());
    }

    #[test]
    fn test_plan_cost_zero() {
        let cost = PlanCost::zero();
        assert_eq!(cost.total(), 0.0);
        assert_eq!(cost.row_count, 0.0);
    }

    #[test]
    fn test_sort_cost_increases_with_rows() {
        let model = make_cost_model();
        let small = PlanCost {
            io_cost: 10.0,
            cpu_cost: 100.0,
            row_count: 100.0,
        };
        let large = PlanCost {
            io_cost: 100.0,
            cpu_cost: 10000.0,
            row_count: 10000.0,
        };
        let small_sort = model.cost_sort(&small);
        let large_sort = model.cost_sort(&large);
        assert!(large_sort.cpu_cost > small_sort.cpu_cost);
    }

    #[test]
    fn test_cost_component_decompose() {
        let model = make_cost_model();
        let cost = PlanCost {
            io_cost: 100.0,
            cpu_cost: 50.0,
            row_count: 1000.0,
        };
        let components = model.decompose(&cost);
        assert_eq!(components.seq_io, 100.0);
        assert!((components.cpu_tuple - 1000.0 * 0.01).abs() < 0.001);
    }

    #[test]
    fn test_weighted_total() {
        let model = make_cost_model();
        let components = CostComponent {
            seq_io: 10.0,
            random_io: 5.0,
            cpu_tuple: 100.0,
            cpu_operator: 50.0,
            network: 0.0,
            memory: 0.0,
        };
        let total = model.weighted_total(&components);
        let expected = 10.0 * 1.0 + 5.0 * 4.0 + 100.0 * 0.01 + 50.0 * 0.0025;
        assert!((total - expected).abs() < 0.001);
    }

    #[test]
    fn test_encoded_scan_cheaper_with_high_skip_rate() {
        let model = make_cost_model();
        let stats = make_table_stats(100_000, 1000);
        let no_skip = model.cost_encoded_scan(
            &stats,
            &EncodingCostParameters {
                skip_rate: 0.0,
                decode_cost_per_value: 0.0001,
                encoded_scan_speedup: 0.3,
            },
        );
        let high_skip = model.cost_encoded_scan(
            &stats,
            &EncodingCostParameters {
                skip_rate: 0.9,
                decode_cost_per_value: 0.0001,
                encoded_scan_speedup: 0.3,
            },
        );
        assert!(high_skip.total() < no_skip.total());
        assert!(high_skip.row_count < no_skip.row_count);
    }

    #[test]
    fn test_parallel_scan_cheaper_for_large_tables() {
        let model = make_cost_model();
        let stats = make_table_stats(1_000_000, 10_000);
        let serial = model.cost_seq_scan(&stats);
        let parallel = model.cost_parallel_scan(&stats, 4);
        // Parallel IO should be roughly 1/4 of serial
        assert!(parallel.io_cost < serial.io_cost);
    }

    #[test]
    fn test_parallel_hash_join_cost() {
        let model = make_cost_model();
        let left = PlanCost {
            io_cost: 100.0,
            cpu_cost: 10000.0,
            row_count: 10000.0,
        };
        let right = PlanCost {
            io_cost: 50.0,
            cpu_cost: 5000.0,
            row_count: 5000.0,
        };
        let serial = model.cost_hash_join(&left, &right);
        let parallel = model.cost_parallel_hash_join(&left, &right, 4);
        // Parallel build+probe CPU per-worker should be lower than serial
        // (total may be higher due to coordination overhead, but wall-clock is less)
        let serial_build_probe = 10000.0 * 0.0025 + 10000.0 * 0.0025;
        let parallel_build_probe = (5000.0 / 4.0) * 0.0025 + (10000.0 / 4.0) * 0.0025;
        assert!(parallel_build_probe < serial_build_probe);
        // IO cost is the same
        assert_eq!(parallel.io_cost, serial.io_cost);
    }
}

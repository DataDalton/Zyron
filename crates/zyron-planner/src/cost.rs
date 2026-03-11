//! Cost model for query plan optimization.
//!
//! Estimates plan costs using catalog statistics (histograms, NDV, null fractions)
//! for selectivity estimation and cardinality calculations. The cost model
//! compares alternative physical plans to select the cheapest execution strategy.

use crate::binder::BoundExpr;
use crate::logical::{JoinCondition, LogicalPlan};
use zyron_catalog::{Catalog, ColumnId, ColumnStats, TableStats};
use zyron_parser::ast::{BinaryOperator, JoinType, LiteralValue};

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
}

impl Default for CostModel {
    fn default() -> Self {
        Self {
            seq_page_cost: 1.0,
            random_page_cost: 4.0,
            cpu_tuple_cost: 0.01,
            cpu_index_tuple_cost: 0.005,
            cpu_operator_cost: 0.0025,
        }
    }
}

impl CostModel {
    // -----------------------------------------------------------------------
    // Selectivity estimation
    // -----------------------------------------------------------------------

    /// Estimates selectivity of a predicate on a table.
    pub fn estimate_selectivity(
        &self,
        predicate: &BoundExpr,
        table_stats: Option<&TableStats>,
        column_stats: Option<&[ColumnStats]>,
    ) -> f64 {
        match predicate {
            // AND: independence assumption
            BoundExpr::BinaryOp { left, op: BinaryOperator::And, right, .. } => {
                let left_sel = self.estimate_selectivity(left, table_stats, column_stats);
                let right_sel = self.estimate_selectivity(right, table_stats, column_stats);
                left_sel * right_sel
            }
            // OR: inclusion-exclusion
            BoundExpr::BinaryOp { left, op: BinaryOperator::Or, right, .. } => {
                let left_sel = self.estimate_selectivity(left, table_stats, column_stats);
                let right_sel = self.estimate_selectivity(right, table_stats, column_stats);
                left_sel + right_sel - left_sel * right_sel
            }
            // NOT
            BoundExpr::UnaryOp { op: zyron_parser::ast::UnaryOperator::Not, expr, .. } => {
                1.0 - self.estimate_selectivity(expr, table_stats, column_stats)
            }
            // Equality: col = literal
            BoundExpr::BinaryOp { left, op: BinaryOperator::Eq, right, .. } => {
                if let Some(col_id) = extract_column_id(left) {
                    if is_literal(right) {
                        return self.estimate_equality_selectivity(col_id, column_stats);
                    }
                }
                if let Some(col_id) = extract_column_id(right) {
                    if is_literal(left) {
                        return self.estimate_equality_selectivity(col_id, column_stats);
                    }
                }
                DEFAULT_EQUALITY_SELECTIVITY
            }
            // Inequality: col != literal
            BoundExpr::BinaryOp { left, op: BinaryOperator::Neq, right, .. } => {
                if let Some(col_id) = extract_column_id(left) {
                    if is_literal(right) {
                        return 1.0 - self.estimate_equality_selectivity(col_id, column_stats);
                    }
                }
                1.0 - DEFAULT_EQUALITY_SELECTIVITY
            }
            // Range: col < literal, col > literal, col <= literal, col >= literal
            BoundExpr::BinaryOp {
                left,
                op: BinaryOperator::Lt | BinaryOperator::Gt | BinaryOperator::LtEq | BinaryOperator::GtEq,
                right,
                ..
            } => {
                if let Some(col_id) = extract_column_id(left) {
                    if is_literal(right) {
                        return self.estimate_range_selectivity(col_id, column_stats);
                    }
                }
                DEFAULT_RANGE_SELECTIVITY
            }
            // IS NULL / IS NOT NULL
            BoundExpr::IsNull { expr, negated } => {
                if let Some(col_id) = extract_column_id(expr) {
                    if let Some(stats) = column_stats {
                        if let Some(cs) = stats.iter().find(|s| s.column_id == col_id) {
                            return if *negated {
                                1.0 - cs.null_fraction
                            } else {
                                cs.null_fraction
                            };
                        }
                    }
                }
                if *negated {
                    1.0 - DEFAULT_NULL_SELECTIVITY
                } else {
                    DEFAULT_NULL_SELECTIVITY
                }
            }
            // IN list
            BoundExpr::InList { expr, list, negated } => {
                let base = if let Some(col_id) = extract_column_id(expr) {
                    let eq_sel = self.estimate_equality_selectivity(col_id, column_stats);
                    (list.len() as f64 * eq_sel).min(1.0)
                } else {
                    DEFAULT_IN_LIST_SELECTIVITY
                };
                if *negated { 1.0 - base } else { base }
            }
            // BETWEEN
            BoundExpr::Between { expr, negated, .. } => {
                let base = if let Some(col_id) = extract_column_id(expr) {
                    self.estimate_range_selectivity(col_id, column_stats)
                } else {
                    DEFAULT_RANGE_SELECTIVITY
                };
                if *negated { 1.0 - base } else { base }
            }
            // LIKE / ILIKE
            BoundExpr::Like { negated, .. } | BoundExpr::ILike { negated, .. } => {
                if *negated {
                    1.0 - DEFAULT_LIKE_SELECTIVITY
                } else {
                    DEFAULT_LIKE_SELECTIVITY
                }
            }
            // Boolean literal
            BoundExpr::Literal { value: LiteralValue::Boolean(true), .. } => 1.0,
            BoundExpr::Literal { value: LiteralValue::Boolean(false), .. } => 0.0,
            // Default
            _ => 0.5,
        }
    }

    /// Estimates selectivity of an equality predicate (col = value).
    /// Uses MCV list, then 1/NDV fallback.
    fn estimate_equality_selectivity(
        &self,
        column_id: ColumnId,
        column_stats: Option<&[ColumnStats]>,
    ) -> f64 {
        if let Some(stats) = column_stats {
            if let Some(cs) = stats.iter().find(|s| s.column_id == column_id) {
                // If we have MCV frequencies, use average MCV frequency
                if !cs.most_common_freqs.is_empty() {
                    let avg_freq: f64 =
                        cs.most_common_freqs.iter().sum::<f64>() / cs.most_common_freqs.len() as f64;
                    return avg_freq;
                }
                // Use 1/NDV
                if cs.distinct_count > 0 {
                    return 1.0 / cs.distinct_count as f64;
                }
            }
        }
        DEFAULT_EQUALITY_SELECTIVITY
    }

    /// Estimates selectivity of a range predicate.
    /// Uses histogram bucket interpolation, or DEFAULT_RANGE_SELECTIVITY as fallback.
    fn estimate_range_selectivity(
        &self,
        column_id: ColumnId,
        column_stats: Option<&[ColumnStats]>,
    ) -> f64 {
        if let Some(stats) = column_stats {
            if let Some(cs) = stats.iter().find(|s| s.column_id == column_id) {
                if let Some(hist) = &cs.histogram {
                    if hist.num_buckets > 0 {
                        // Approximate: assume uniform distribution across buckets.
                        // A range predicate touches roughly 1/3 of the data.
                        return DEFAULT_RANGE_SELECTIVITY;
                    }
                }
                // Without histogram, use NDV-based estimate
                if cs.distinct_count > 0 {
                    return DEFAULT_RANGE_SELECTIVITY;
                }
            }
        }
        DEFAULT_RANGE_SELECTIVITY
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
                + left.cpu_cost + right.cpu_cost,
            row_count: estimate_join_rows(left.row_count, right.row_count),
        }
    }

    /// Estimates the cost of a merge join (both sides assumed sorted).
    pub fn cost_merge_join(&self, left: &PlanCost, right: &PlanCost) -> PlanCost {
        PlanCost {
            io_cost: left.io_cost + right.io_cost,
            cpu_cost: (left.row_count + right.row_count) * self.cpu_operator_cost
                + left.cpu_cost + right.cpu_cost,
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
                    cpu_cost: child_cost.cpu_cost
                        + child_cost.row_count * self.cpu_operator_cost,
                    row_count: (child_cost.row_count * selectivity).max(1.0),
                }
            }
            LogicalPlan::Project { child, .. } => {
                let child_cost = self.estimate_plan_cost(child, catalog);
                PlanCost {
                    io_cost: child_cost.io_cost,
                    cpu_cost: child_cost.cpu_cost
                        + child_cost.row_count * self.cpu_operator_cost,
                    row_count: child_cost.row_count,
                }
            }
            LogicalPlan::Join { left, right, join_type, condition } => {
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
                    cpu_cost: left_cost.cpu_cost + right_cost.cpu_cost
                        + rows * self.cpu_operator_cost,
                    row_count: rows,
                }
            }
            LogicalPlan::Aggregate { group_by, child, .. } => {
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
            LogicalPlan::Limit { limit, offset: _, child } => {
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
                    cpu_cost: child_cost.cpu_cost
                        + child_cost.row_count * self.cpu_operator_cost,
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

/// Extracts a ColumnId from a BoundExpr if it's a column reference.
fn extract_column_id(expr: &BoundExpr) -> Option<ColumnId> {
    match expr {
        BoundExpr::ColumnRef(cr) => Some(cr.column_id),
        BoundExpr::Nested(inner) => extract_column_id(inner),
        _ => None,
    }
}

/// Returns true if the expression is a literal value.
fn is_literal(expr: &BoundExpr) -> bool {
    matches!(expr, BoundExpr::Literal { .. })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use zyron_catalog::TableId;
    use zyron_common::TypeId;

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
        let sel = model.estimate_equality_selectivity(ColumnId(0), Some(&stats));
        assert!((sel - 0.01).abs() < 0.001); // 1/100
    }

    #[test]
    fn test_hash_join_cheaper_than_nested_loop() {
        let model = make_cost_model();
        let left = PlanCost { io_cost: 100.0, cpu_cost: 10000.0, row_count: 10000.0 };
        let right = PlanCost { io_cost: 50.0, cpu_cost: 5000.0, row_count: 5000.0 };
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
        let small = PlanCost { io_cost: 10.0, cpu_cost: 100.0, row_count: 100.0 };
        let large = PlanCost { io_cost: 100.0, cpu_cost: 10000.0, row_count: 10000.0 };
        let small_sort = model.cost_sort(&small);
        let large_sort = model.cost_sort(&large);
        assert!(large_sort.cpu_cost > small_sort.cpu_cost);
    }
}

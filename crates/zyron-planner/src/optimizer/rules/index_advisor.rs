//! Index advisor: passive optimization rule that tracks column access patterns
//! and recommends indexes based on workload analysis.
//!
//! This rule never transforms the plan tree. It walks the plan to record
//! which columns appear in filter predicates, then provides recommendations
//! for indexes that would improve query performance.

use crate::binder::BoundExpr;
use crate::logical::LogicalPlan;
use crate::optimizer::OptimizationRule;
use std::collections::HashMap;
use std::sync::Mutex;
use zyron_catalog::{Catalog, ColumnId, TableId};

// ---------------------------------------------------------------------------
// Index advisor
// ---------------------------------------------------------------------------

/// Index advisor: records column scan patterns and generates recommendations.
/// Uses a Mutex-protected HashMap since tracking happens during planning
/// (not on the query execution hot path).
pub struct IndexAdvisor {
    /// (table_id raw, column_id raw) -> scan count
    scan_tracker: Mutex<HashMap<(u32, u16), u64>>,
    /// Minimum table size (rows) to consider index recommendations.
    min_table_rows: u64,
    /// Selectivity threshold: recommend index when selectivity < this value.
    selectivity_threshold: f64,
}

/// A recommended index based on workload analysis.
#[derive(Debug, Clone)]
pub struct IndexRecommendation {
    pub table_id: TableId,
    pub columns: Vec<ColumnId>,
    pub scan_count: u64,
    pub estimated_selectivity: f64,
    pub reason: String,
}

impl IndexAdvisor {
    pub fn new() -> Self {
        Self {
            scan_tracker: Mutex::new(HashMap::new()),
            min_table_rows: 10_000,
            selectivity_threshold: 0.10,
        }
    }

    /// Records a column scan from a filter predicate.
    pub fn record_scan(&self, table_id: TableId, column_id: ColumnId) {
        let key = (table_id.0, column_id.0);
        if let Ok(mut tracker) = self.scan_tracker.lock() {
            *tracker.entry(key).or_insert(0) += 1;
        }
    }

    /// Walks a plan tree and records all filtered column references.
    fn walk_and_record(&self, plan: &LogicalPlan) {
        match plan {
            LogicalPlan::Filter { predicate, child } => {
                if let LogicalPlan::Scan { table_id, .. } = child.as_ref() {
                    self.extract_filter_columns(predicate, *table_id);
                }
                self.walk_and_record(child);
            }
            LogicalPlan::Scan { .. }
            | LogicalPlan::Values { .. }
            | LogicalPlan::GraphAlgorithm { .. } => {}
            LogicalPlan::Project { child, .. }
            | LogicalPlan::Sort { child, .. }
            | LogicalPlan::Limit { child, .. }
            | LogicalPlan::Distinct { child, .. }
            | LogicalPlan::Aggregate { child, .. }
            | LogicalPlan::Insert { source: child, .. }
            | LogicalPlan::Update { child, .. }
            | LogicalPlan::Delete { child, .. } => {
                self.walk_and_record(child);
            }
            LogicalPlan::Join { left, right, .. } | LogicalPlan::SetOp { left, right, .. } => {
                self.walk_and_record(left);
                self.walk_and_record(right);
            }
        }
    }

    /// Extracts column references from a predicate expression and records them.
    fn extract_filter_columns(&self, expr: &BoundExpr, table_id: TableId) {
        match expr {
            BoundExpr::ColumnRef(cr) => {
                self.record_scan(table_id, cr.column_id);
            }
            BoundExpr::BinaryOp { left, right, .. } => {
                self.extract_filter_columns(left, table_id);
                self.extract_filter_columns(right, table_id);
            }
            BoundExpr::UnaryOp { expr, .. } => {
                self.extract_filter_columns(expr, table_id);
            }
            BoundExpr::IsNull { expr, .. } => {
                self.extract_filter_columns(expr, table_id);
            }
            BoundExpr::InList { expr, list, .. } => {
                self.extract_filter_columns(expr, table_id);
                for item in list {
                    self.extract_filter_columns(item, table_id);
                }
            }
            BoundExpr::Between {
                expr, low, high, ..
            } => {
                self.extract_filter_columns(expr, table_id);
                self.extract_filter_columns(low, table_id);
                self.extract_filter_columns(high, table_id);
            }
            BoundExpr::Like { expr, .. } | BoundExpr::ILike { expr, .. } => {
                self.extract_filter_columns(expr, table_id);
            }
            BoundExpr::Nested(inner) => {
                self.extract_filter_columns(inner, table_id);
            }
            _ => {}
        }
    }

    /// Returns current index recommendations based on accumulated scan patterns.
    /// Only recommends indexes for tables with > min_table_rows rows
    /// and columns scanned with selectivity < selectivity_threshold.
    pub fn recommendations(&self, catalog: &Catalog) -> Vec<IndexRecommendation> {
        let mut recs = Vec::new();

        let tracker = match self.scan_tracker.lock() {
            Ok(t) => t,
            Err(e) => e.into_inner(),
        };

        for (&(table_raw, col_raw), &count) in tracker.iter() {
            let table_id = TableId(table_raw);
            let column_id = ColumnId(col_raw);

            // Check table size
            if let Some((ts, cs)) = catalog.get_stats(table_id) {
                if ts.row_count < self.min_table_rows {
                    continue;
                }

                // Check if column already has an index
                let indexes = catalog.get_indexes_for_table(table_id);
                let already_indexed = indexes
                    .iter()
                    .any(|idx| idx.columns.first().map(|c| c.column_id) == Some(column_id));
                if already_indexed {
                    continue;
                }

                // Estimate selectivity from column stats
                let selectivity =
                    if let Some(col_stat) = cs.iter().find(|s| s.column_id == column_id) {
                        if col_stat.distinct_count > 0 {
                            1.0 / col_stat.distinct_count as f64
                        } else {
                            0.1
                        }
                    } else {
                        0.1
                    };

                if selectivity < self.selectivity_threshold {
                    recs.push(IndexRecommendation {
                        table_id,
                        columns: vec![column_id],
                        scan_count: count,
                        estimated_selectivity: selectivity,
                        reason: format!(
                            "Column scanned {} times with estimated selectivity {:.4}, table has {} rows",
                            count, selectivity, ts.row_count,
                        ),
                    });
                }
            }
        }

        // Sort by scan count descending (most impactful first)
        recs.sort_by(|a, b| b.scan_count.cmp(&a.scan_count));
        recs
    }
}

impl Default for IndexAdvisor {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationRule for IndexAdvisor {
    fn name(&self) -> &str {
        "index_advisor"
    }

    /// Passive rule: walks the plan to record scan patterns but never transforms it.
    fn apply(&self, plan: &LogicalPlan, _catalog: &Catalog) -> Option<LogicalPlan> {
        self.walk_and_record(plan);
        None
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_advisor_new() {
        let advisor = IndexAdvisor::new();
        assert_eq!(advisor.min_table_rows, 10_000);
        assert!((advisor.selectivity_threshold - 0.10).abs() < 0.001);
    }

    #[test]
    fn test_record_scan() {
        let advisor = IndexAdvisor::new();
        advisor.record_scan(TableId(1), ColumnId(0));
        advisor.record_scan(TableId(1), ColumnId(0));
        advisor.record_scan(TableId(1), ColumnId(1));

        let tracker = advisor.scan_tracker.lock().unwrap();
        assert_eq!(tracker.get(&(1, 0)), Some(&2));
        assert_eq!(tracker.get(&(1, 1)), Some(&1));
    }

    #[test]
    fn test_walk_records_columns() {
        let advisor = IndexAdvisor::new();
        let scan = LogicalPlan::Scan {
            table_id: TableId(1),
            table_idx: 0,
            columns: vec![],
            alias: "t".to_string(),
            encoding_hints: None,
            as_of: None,
        };
        let filter = LogicalPlan::Filter {
            predicate: BoundExpr::ColumnRef(crate::binder::ColumnRef {
                table_idx: 0,
                column_id: ColumnId(5),
                type_id: zyron_common::TypeId::Int64,
                nullable: false,
            }),
            child: Box::new(scan),
        };
        advisor.walk_and_record(&filter);

        let tracker = advisor.scan_tracker.lock().unwrap();
        assert_eq!(tracker.get(&(1, 5)), Some(&1));
    }
}

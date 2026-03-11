//! Converts optimized logical plans into physical execution plans.
//!
//! Makes cost-based decisions for operator selection:
//! - SeqScan vs IndexScan (index scan preferred when selectivity < 10%)
//! - HashJoin vs MergeJoin vs NestedLoopJoin
//! - HashAggregate vs SortAggregate

use crate::binder::BoundExpr;
use crate::cost::{CostModel, PlanCost, INDEX_SCAN_SELECTIVITY_THRESHOLD};
use crate::logical::{JoinCondition, LogicalPlan};
use crate::physical::*;
use zyron_catalog::{Catalog, IndexEntry};
use zyron_common::{Result, TypeId};
use zyron_parser::ast::{BinaryOperator, JoinType};
use std::sync::Arc;

/// Converts an optimized logical plan into a physical plan using cost-based decisions.
pub fn build_physical_plan(logical: LogicalPlan, catalog: &Catalog) -> Result<PhysicalPlan> {
    let cost_model = CostModel::default();
    PhysicalPlanner::new(catalog, cost_model).plan(logical)
}

struct PhysicalPlanner<'a> {
    catalog: &'a Catalog,
    cost_model: CostModel,
}

impl<'a> PhysicalPlanner<'a> {
    fn new(catalog: &'a Catalog, cost_model: CostModel) -> Self {
        Self { catalog, cost_model }
    }

    fn plan(&self, logical: LogicalPlan) -> Result<PhysicalPlan> {
        match logical {
            LogicalPlan::Scan { table_id, table_idx: _, columns, alias: _ } => {
                self.plan_scan(table_id, columns, None)
            }
            LogicalPlan::Filter { predicate, child } => {
                // Try to push the filter into a scan (index scan opportunity)
                if let LogicalPlan::Scan { table_id, columns, .. } = *child {
                    return self.plan_scan(table_id, columns, Some(predicate));
                }

                let child_plan = self.plan(*child)?;
                let child_cost = *child_plan.cost();
                let selectivity = self.cost_model.estimate_selectivity(&predicate, None, None);
                let cost = PlanCost {
                    io_cost: 0.0,
                    cpu_cost: child_cost.row_count * self.cost_model.cpu_operator_cost,
                    row_count: (child_cost.row_count * selectivity).max(1.0),
                };
                Ok(PhysicalPlan::Filter {
                    predicate,
                    child: Box::new(child_plan),
                    cost,
                })
            }
            LogicalPlan::Project { expressions, aliases, child } => {
                let child_plan = self.plan(*child)?;
                let child_cost = *child_plan.cost();
                let cost = PlanCost {
                    io_cost: 0.0,
                    cpu_cost: child_cost.row_count * self.cost_model.cpu_operator_cost,
                    row_count: child_cost.row_count,
                };
                Ok(PhysicalPlan::Project {
                    expressions,
                    aliases,
                    child: Box::new(child_plan),
                    cost,
                })
            }
            LogicalPlan::Join { left, right, join_type, condition } => {
                self.plan_join(*left, *right, join_type, condition)
            }
            LogicalPlan::Aggregate { group_by, aggregates, child } => {
                self.plan_aggregate(group_by, aggregates, *child)
            }
            LogicalPlan::Sort { order_by, child } => {
                let child_plan = self.plan(*child)?;
                let child_cost = *child_plan.cost();
                let sort_cost = self.cost_model.cost_sort(&child_cost);
                Ok(PhysicalPlan::Sort {
                    order_by,
                    child: Box::new(child_plan),
                    limit: None,
                    cost: PlanCost {
                        io_cost: 0.0,
                        cpu_cost: sort_cost.cpu_cost - child_cost.cpu_cost,
                        row_count: child_cost.row_count,
                    },
                })
            }
            LogicalPlan::Limit { limit, offset, child } => {
                // Check if there's a Sort below for top-N optimization
                let child_plan = self.plan(*child)?;
                let rows = limit
                    .map(|l| (l as f64).min(child_plan.cost().row_count))
                    .unwrap_or(child_plan.cost().row_count);
                let cost = PlanCost {
                    io_cost: 0.0,
                    cpu_cost: rows * self.cost_model.cpu_tuple_cost,
                    row_count: rows,
                };
                Ok(PhysicalPlan::Limit {
                    limit,
                    offset,
                    child: Box::new(child_plan),
                    cost,
                })
            }
            LogicalPlan::Distinct { child } => {
                let child_plan = self.plan(*child)?;
                let child_cost = *child_plan.cost();
                let cost = PlanCost {
                    io_cost: 0.0,
                    cpu_cost: child_cost.row_count * self.cost_model.cpu_operator_cost,
                    row_count: child_cost.row_count * 0.8,
                };
                Ok(PhysicalPlan::HashDistinct {
                    child: Box::new(child_plan),
                    cost,
                })
            }
            LogicalPlan::SetOp { op, all, left, right } => {
                let left_plan = self.plan(*left)?;
                let right_plan = self.plan(*right)?;
                let cost = PlanCost {
                    io_cost: 0.0,
                    cpu_cost: (left_plan.cost().row_count + right_plan.cost().row_count)
                        * self.cost_model.cpu_tuple_cost,
                    row_count: left_plan.cost().row_count + right_plan.cost().row_count,
                };
                Ok(PhysicalPlan::SetOp {
                    op,
                    all,
                    left: Box::new(left_plan),
                    right: Box::new(right_plan),
                    cost,
                })
            }
            LogicalPlan::Insert { table_id, target_columns, source } => {
                let source_plan = self.plan(*source)?;
                let cost = *source_plan.cost();
                Ok(PhysicalPlan::Insert {
                    table_id,
                    target_columns,
                    source: Box::new(source_plan),
                    cost,
                })
            }
            LogicalPlan::Values { rows, schema } => {
                let cost = PlanCost {
                    io_cost: 0.0,
                    cpu_cost: rows.len() as f64 * self.cost_model.cpu_tuple_cost,
                    row_count: rows.len() as f64,
                };
                Ok(PhysicalPlan::Values { rows, schema, cost })
            }
            LogicalPlan::Update { table_id, assignments, child } => {
                let child_plan = self.plan(*child)?;
                let cost = *child_plan.cost();
                Ok(PhysicalPlan::Update {
                    table_id,
                    assignments,
                    child: Box::new(child_plan),
                    cost,
                })
            }
            LogicalPlan::Delete { table_id, child } => {
                let child_plan = self.plan(*child)?;
                let cost = *child_plan.cost();
                Ok(PhysicalPlan::Delete {
                    table_id,
                    child: Box::new(child_plan),
                    cost,
                })
            }
        }
    }

    // -----------------------------------------------------------------------
    // Scan planning: SeqScan vs IndexScan
    // -----------------------------------------------------------------------

    fn plan_scan(
        &self,
        table_id: zyron_catalog::TableId,
        columns: Vec<crate::logical::LogicalColumn>,
        predicate: Option<BoundExpr>,
    ) -> Result<PhysicalPlan> {
        // Get table stats
        let table_stats = self.catalog.get_stats(table_id);

        // Get available indexes
        let indexes = self.catalog.get_indexes_for_table(table_id);

        // Try to find an index scan opportunity
        if let Some(pred) = &predicate {
            if let Some((ts, cs)) = &table_stats {
                for index in &indexes {
                    if let Some((index_pred, remaining)) = match_index(pred, index) {
                        let selectivity = self.cost_model.estimate_selectivity(
                            &index_pred,
                            Some(ts),
                            Some(cs),
                        );

                        if selectivity < INDEX_SCAN_SELECTIVITY_THRESHOLD {
                            let cost = self.cost_model.cost_index_scan(ts, selectivity);
                            return Ok(PhysicalPlan::IndexScan {
                                table_id,
                                index_id: index.id,
                                index: Arc::clone(index),
                                columns,
                                predicate: index_pred,
                                remaining_predicate: remaining,
                                scan_direction: ScanDirection::Forward,
                                cost,
                            });
                        }
                    }
                }
            }
        }

        // Default to sequential scan
        let cost = if let Some((ts, _)) = &table_stats {
            let mut scan_cost = self.cost_model.cost_seq_scan(ts);
            if let Some(pred) = &predicate {
                let selectivity = self.cost_model.estimate_selectivity(pred, table_stats.as_ref().map(|(ts, _)| ts), table_stats.as_ref().map(|(_, cs)| cs.as_slice()));
                scan_cost.row_count = (scan_cost.row_count * selectivity).max(1.0);
            }
            scan_cost
        } else {
            PlanCost {
                io_cost: 10.0,
                cpu_cost: 1000.0 * self.cost_model.cpu_tuple_cost,
                row_count: if predicate.is_some() { 100.0 } else { 1000.0 },
            }
        };

        Ok(PhysicalPlan::SeqScan {
            table_id,
            columns,
            predicate,
            cost,
        })
    }

    // -----------------------------------------------------------------------
    // Join planning: Hash vs Merge vs Nested Loop
    // -----------------------------------------------------------------------

    fn plan_join(
        &self,
        left: LogicalPlan,
        right: LogicalPlan,
        join_type: JoinType,
        condition: JoinCondition,
    ) -> Result<PhysicalPlan> {
        let left_plan = self.plan(left)?;
        let right_plan = self.plan(right)?;
        let left_cost = *left_plan.cost();
        let right_cost = *right_plan.cost();

        match &condition {
            JoinCondition::On(expr) => {
                // Try to extract equi-join keys
                if let Some((left_keys, right_keys, remaining)) = extract_equi_keys(expr) {
                    // Cost all three strategies
                    let hash_cost = self.cost_model.cost_hash_join(&left_cost, &right_cost);
                    let merge_cost_base = self.cost_model.cost_merge_join(&left_cost, &right_cost);
                    let nl_cost = self.cost_model.cost_nested_loop_join(&left_cost, &right_cost);

                    // Add sort cost to merge join if needed
                    let left_sort_cost = self.cost_model.cost_sort(&left_cost);
                    let right_sort_cost = self.cost_model.cost_sort(&right_cost);
                    let merge_total = merge_cost_base.total()
                        + left_sort_cost.total()
                        + right_sort_cost.total();

                    // Pick cheapest
                    if nl_cost.total() < hash_cost.total() && nl_cost.total() < merge_total
                        && right_cost.row_count < 100.0
                    {
                        // Nested loop for small right side
                        Ok(PhysicalPlan::NestedLoopJoin {
                            left: Box::new(left_plan),
                            right: Box::new(right_plan),
                            join_type,
                            condition: Some(expr.clone()),
                            cost: nl_cost,
                        })
                    } else if merge_total < hash_cost.total() {
                        // Merge join
                        Ok(PhysicalPlan::MergeJoin {
                            left: Box::new(left_plan),
                            right: Box::new(right_plan),
                            join_type,
                            left_keys,
                            right_keys,
                            cost: merge_cost_base,
                        })
                    } else {
                        // Hash join (default for equi-joins)
                        Ok(PhysicalPlan::HashJoin {
                            left: Box::new(left_plan),
                            right: Box::new(right_plan),
                            join_type,
                            left_keys,
                            right_keys,
                            remaining_condition: remaining,
                            cost: hash_cost,
                        })
                    }
                } else {
                    // Non-equi join: must use nested loop
                    let cost = self.cost_model.cost_nested_loop_join(&left_cost, &right_cost);
                    Ok(PhysicalPlan::NestedLoopJoin {
                        left: Box::new(left_plan),
                        right: Box::new(right_plan),
                        join_type,
                        condition: Some(expr.clone()),
                        cost,
                    })
                }
            }
            JoinCondition::Using(_) | JoinCondition::Natural => {
                // Treat as hash join with the condition being equality on shared columns
                let cost = self.cost_model.cost_hash_join(&left_cost, &right_cost);
                Ok(PhysicalPlan::HashJoin {
                    left: Box::new(left_plan),
                    right: Box::new(right_plan),
                    join_type,
                    left_keys: vec![],
                    right_keys: vec![],
                    remaining_condition: None,
                    cost,
                })
            }
            JoinCondition::Cross => {
                let cost = self.cost_model.cost_nested_loop_join(&left_cost, &right_cost);
                Ok(PhysicalPlan::NestedLoopJoin {
                    left: Box::new(left_plan),
                    right: Box::new(right_plan),
                    join_type,
                    condition: None,
                    cost,
                })
            }
        }
    }

    // -----------------------------------------------------------------------
    // Aggregate planning: Hash vs Sort
    // -----------------------------------------------------------------------

    fn plan_aggregate(
        &self,
        group_by: Vec<BoundExpr>,
        aggregates: Vec<crate::logical::AggregateExpr>,
        child: LogicalPlan,
    ) -> Result<PhysicalPlan> {
        let child_plan = self.plan(child)?;
        let child_cost = *child_plan.cost();

        let group_count = if group_by.is_empty() {
            1.0
        } else {
            child_cost.row_count.sqrt().max(1.0)
        };

        let cost = self.cost_model.cost_hash_aggregate(&child_cost, group_count);

        // Use HashAggregate by default (better for random group distributions)
        Ok(PhysicalPlan::HashAggregate {
            group_by,
            aggregates,
            child: Box::new(child_plan),
            cost: PlanCost {
                io_cost: 0.0,
                cpu_cost: cost.cpu_cost - child_cost.cpu_cost,
                row_count: group_count,
            },
        })
    }
}

// ---------------------------------------------------------------------------
// Index matching
// ---------------------------------------------------------------------------

/// Checks if a predicate matches an index's leading column(s).
/// Returns (index_predicate, remaining_predicate) if a match is found.
fn match_index(
    predicate: &BoundExpr,
    index: &IndexEntry,
) -> Option<(BoundExpr, Option<BoundExpr>)> {
    if index.columns.is_empty() {
        return None;
    }

    let leading_col = index.columns[0].column_id;

    // Check if the predicate references the leading index column
    match predicate {
        BoundExpr::BinaryOp {
            left,
            op: BinaryOperator::Eq | BinaryOperator::Lt | BinaryOperator::Gt
                | BinaryOperator::LtEq | BinaryOperator::GtEq,
            right,
            ..
        } => {
            let left_col = extract_column_id_from_expr(left);
            let right_col = extract_column_id_from_expr(right);

            if left_col == Some(leading_col) || right_col == Some(leading_col) {
                return Some((predicate.clone(), None));
            }
            None
        }
        // AND: check if any conjunct matches the index
        BoundExpr::BinaryOp { left, op: BinaryOperator::And, right, .. } => {
            let left_match = match_index(left, index);
            let right_match = match_index(right, index);

            match (left_match, right_match) {
                (Some((l_pred, _)), Some((r_pred, _))) => {
                    // Both sides match: combine as the index predicate
                    Some((
                        BoundExpr::BinaryOp {
                            left: Box::new(l_pred),
                            op: BinaryOperator::And,
                            right: Box::new(r_pred),
                            type_id: TypeId::Boolean,
                        },
                        None,
                    ))
                }
                (Some((idx_pred, _)), None) => {
                    Some((idx_pred, Some(right.as_ref().clone())))
                }
                (None, Some((idx_pred, _))) => {
                    Some((idx_pred, Some(left.as_ref().clone())))
                }
                (None, None) => None,
            }
        }
        BoundExpr::Between { expr, negated: false, .. } => {
            if extract_column_id_from_expr(expr) == Some(leading_col) {
                Some((predicate.clone(), None))
            } else {
                None
            }
        }
        BoundExpr::InList { expr, negated: false, .. } => {
            if extract_column_id_from_expr(expr) == Some(leading_col) {
                Some((predicate.clone(), None))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn extract_column_id_from_expr(expr: &BoundExpr) -> Option<zyron_catalog::ColumnId> {
    match expr {
        BoundExpr::ColumnRef(cr) => Some(cr.column_id),
        BoundExpr::Nested(inner) => extract_column_id_from_expr(inner),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Equi-join key extraction
// ---------------------------------------------------------------------------

/// Extracts equi-join keys from a conjunction.
/// Given `a.x = b.y AND a.z = b.w AND a.q > 5`,
/// returns (vec![a.x, a.z], vec![b.y, b.w], Some(a.q > 5)).
fn extract_equi_keys(
    expr: &BoundExpr,
) -> Option<(Vec<BoundExpr>, Vec<BoundExpr>, Option<BoundExpr>)> {
    let conjuncts = split_conjuncts(expr);
    let mut left_keys = Vec::new();
    let mut right_keys = Vec::new();
    let mut remaining = Vec::new();

    for conj in conjuncts {
        if let BoundExpr::BinaryOp { left, op: BinaryOperator::Eq, right, .. } = &conj {
            if is_column_ref(left) && is_column_ref(right) {
                left_keys.push(left.as_ref().clone());
                right_keys.push(right.as_ref().clone());
                continue;
            }
        }
        remaining.push(conj);
    }

    if left_keys.is_empty() {
        return None;
    }

    let remaining_expr = if remaining.is_empty() {
        None
    } else {
        Some(combine_conjuncts(remaining))
    };

    Some((left_keys, right_keys, remaining_expr))
}

fn split_conjuncts(expr: &BoundExpr) -> Vec<BoundExpr> {
    match expr {
        BoundExpr::BinaryOp { left, op: BinaryOperator::And, right, .. } => {
            let mut result = split_conjuncts(left);
            result.extend(split_conjuncts(right));
            result
        }
        other => vec![other.clone()],
    }
}

fn combine_conjuncts(mut conjuncts: Vec<BoundExpr>) -> BoundExpr {
    if conjuncts.len() == 1 {
        return conjuncts.remove(0);
    }
    let mut result = conjuncts.remove(0);
    for conj in conjuncts {
        result = BoundExpr::BinaryOp {
            left: Box::new(result),
            op: BinaryOperator::And,
            right: Box::new(conj),
            type_id: TypeId::Boolean,
        };
    }
    result
}

fn is_column_ref(expr: &BoundExpr) -> bool {
    matches!(expr, BoundExpr::ColumnRef(_))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binder::ColumnRef;
    use zyron_catalog::ColumnId;

    #[test]
    fn test_extract_equi_keys() {
        let left_col = BoundExpr::ColumnRef(ColumnRef {
            table_idx: 0,
            column_id: ColumnId(0),
            type_id: TypeId::Int64,
            nullable: false,
        });
        let right_col = BoundExpr::ColumnRef(ColumnRef {
            table_idx: 1,
            column_id: ColumnId(0),
            type_id: TypeId::Int64,
            nullable: false,
        });
        let eq = BoundExpr::BinaryOp {
            left: Box::new(left_col.clone()),
            op: BinaryOperator::Eq,
            right: Box::new(right_col.clone()),
            type_id: TypeId::Boolean,
        };

        let result = extract_equi_keys(&eq);
        assert!(result.is_some());
        let (lk, rk, rem) = result.unwrap();
        assert_eq!(lk.len(), 1);
        assert_eq!(rk.len(), 1);
        assert!(rem.is_none());
    }

    #[test]
    fn test_extract_equi_keys_with_remaining() {
        let left_col = BoundExpr::ColumnRef(ColumnRef {
            table_idx: 0,
            column_id: ColumnId(0),
            type_id: TypeId::Int64,
            nullable: false,
        });
        let right_col = BoundExpr::ColumnRef(ColumnRef {
            table_idx: 1,
            column_id: ColumnId(0),
            type_id: TypeId::Int64,
            nullable: false,
        });
        let eq = BoundExpr::BinaryOp {
            left: Box::new(left_col.clone()),
            op: BinaryOperator::Eq,
            right: Box::new(right_col.clone()),
            type_id: TypeId::Boolean,
        };
        let extra = BoundExpr::BinaryOp {
            left: Box::new(left_col.clone()),
            op: BinaryOperator::Gt,
            right: Box::new(BoundExpr::Literal {
                value: zyron_parser::ast::LiteralValue::Integer(5),
                type_id: TypeId::Int64,
            }),
            type_id: TypeId::Boolean,
        };
        let combined = BoundExpr::BinaryOp {
            left: Box::new(eq),
            op: BinaryOperator::And,
            right: Box::new(extra),
            type_id: TypeId::Boolean,
        };

        let result = extract_equi_keys(&combined);
        assert!(result.is_some());
        let (lk, rk, rem) = result.unwrap();
        assert_eq!(lk.len(), 1);
        assert_eq!(rk.len(), 1);
        assert!(rem.is_some());
    }

    #[test]
    fn test_no_equi_keys() {
        let expr = BoundExpr::BinaryOp {
            left: Box::new(BoundExpr::ColumnRef(ColumnRef {
                table_idx: 0,
                column_id: ColumnId(0),
                type_id: TypeId::Int64,
                nullable: false,
            })),
            op: BinaryOperator::Gt,
            right: Box::new(BoundExpr::Literal {
                value: zyron_parser::ast::LiteralValue::Integer(5),
                type_id: TypeId::Int64,
            }),
            type_id: TypeId::Boolean,
        };
        assert!(extract_equi_keys(&expr).is_none());
    }
}

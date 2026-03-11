//! Predicate pushdown optimization rule.
//!
//! Pushes filter predicates closer to table scans to reduce the number
//! of rows processed by upstream operators. Splits conjuncts across
//! join sides when possible.
//! Uses a changed-flag pattern to avoid cloning unchanged plan trees.

use crate::binder::{BoundExpr, ColumnRef};
use crate::logical::LogicalPlan;
use crate::optimizer::OptimizationRule;
use zyron_catalog::Catalog;
use zyron_common::TypeId;
use zyron_parser::ast::BinaryOperator;

pub struct PredicatePushdown;

impl OptimizationRule for PredicatePushdown {
    fn name(&self) -> &str {
        "predicate_pushdown"
    }

    fn apply(&self, plan: &LogicalPlan, _catalog: &Catalog) -> Option<LogicalPlan> {
        // Quick check: no Filter nodes means nothing to push.
        if !has_filter(plan) {
            return None;
        }
        let (pushed, changed) = push_predicates(plan);
        if changed { Some(pushed) } else { None }
    }
}

/// Returns true if the plan tree contains any Filter node.
fn has_filter(plan: &LogicalPlan) -> bool {
    match plan {
        LogicalPlan::Filter { .. } => true,
        other => other.children().iter().any(|c| has_filter(c)),
    }
}

/// Returns (pushed_plan, changed).
fn push_predicates(plan: &LogicalPlan) -> (LogicalPlan, bool) {
    match plan {
        // Filter above Join: try to push predicates into join sides
        LogicalPlan::Filter {
            predicate,
            child,
        } => {
            let (child_plan, child_changed) = push_predicates(child);
            match &child_plan {
                LogicalPlan::Join { left, right, join_type, condition } => {
                    let conjuncts = split_conjuncts(predicate);
                    let left_tables = collect_table_indices(left);
                    let right_tables = collect_table_indices(right);

                    let mut left_preds = Vec::new();
                    let mut right_preds = Vec::new();
                    let mut remaining = Vec::new();

                    for conj in conjuncts {
                        let refs = collect_column_refs(&conj);
                        let touches_left = refs.iter().any(|r| left_tables.contains(&r.table_idx));
                        let touches_right = refs.iter().any(|r| right_tables.contains(&r.table_idx));

                        if touches_left && !touches_right {
                            left_preds.push(conj);
                        } else if touches_right && !touches_left {
                            right_preds.push(conj);
                        } else {
                            remaining.push(conj);
                        }
                    }

                    // If nothing was pushed down and child didn't change, skip clone.
                    if left_preds.is_empty() && right_preds.is_empty() && !child_changed {
                        return (plan.clone(), false);
                    }

                    let new_left = if left_preds.is_empty() {
                        left.as_ref().clone()
                    } else {
                        LogicalPlan::Filter {
                            predicate: combine_conjuncts(left_preds),
                            child: Box::new(left.as_ref().clone()),
                        }
                    };

                    let new_right = if right_preds.is_empty() {
                        right.as_ref().clone()
                    } else {
                        LogicalPlan::Filter {
                            predicate: combine_conjuncts(right_preds),
                            child: Box::new(right.as_ref().clone()),
                        }
                    };

                    let (pushed_left, _) = push_predicates(&new_left);
                    let (pushed_right, _) = push_predicates(&new_right);

                    let join = LogicalPlan::Join {
                        left: Box::new(pushed_left),
                        right: Box::new(pushed_right),
                        join_type: *join_type,
                        condition: condition.clone(),
                    };

                    if remaining.is_empty() {
                        (join, true)
                    } else {
                        (LogicalPlan::Filter {
                            predicate: combine_conjuncts(remaining),
                            child: Box::new(join),
                        }, true)
                    }
                }
                // Filter above Project: keep filter above for now
                LogicalPlan::Project { expressions, aliases, child: proj_child } => {
                    let (pushed_proj_child, _) = push_predicates(proj_child);
                    (LogicalPlan::Filter {
                        predicate: predicate.clone(),
                        child: Box::new(LogicalPlan::Project {
                            expressions: expressions.clone(),
                            aliases: aliases.clone(),
                            child: Box::new(pushed_proj_child),
                        }),
                    }, child_changed)
                }
                _ => {
                    if child_changed {
                        (LogicalPlan::Filter {
                            predicate: predicate.clone(),
                            child: Box::new(child_plan),
                        }, true)
                    } else {
                        (plan.clone(), false)
                    }
                }
            }
        }
        // Recursively apply to all other node types
        LogicalPlan::Project { expressions, aliases, child } => {
            let (fc, changed) = push_predicates(child);
            if changed {
                (LogicalPlan::Project {
                    expressions: expressions.clone(),
                    aliases: aliases.clone(),
                    child: Box::new(fc),
                }, true)
            } else {
                (plan.clone(), false)
            }
        }
        LogicalPlan::Join { left, right, join_type, condition } => {
            let (fl, lc) = push_predicates(left);
            let (fr, rc) = push_predicates(right);
            if lc || rc {
                (LogicalPlan::Join {
                    left: Box::new(fl),
                    right: Box::new(fr),
                    join_type: *join_type,
                    condition: condition.clone(),
                }, true)
            } else {
                (plan.clone(), false)
            }
        }
        LogicalPlan::Aggregate { group_by, aggregates, child } => {
            let (fc, changed) = push_predicates(child);
            if changed {
                (LogicalPlan::Aggregate {
                    group_by: group_by.clone(),
                    aggregates: aggregates.clone(),
                    child: Box::new(fc),
                }, true)
            } else {
                (plan.clone(), false)
            }
        }
        LogicalPlan::Sort { order_by, child } => {
            let (fc, changed) = push_predicates(child);
            if changed {
                (LogicalPlan::Sort {
                    order_by: order_by.clone(),
                    child: Box::new(fc),
                }, true)
            } else {
                (plan.clone(), false)
            }
        }
        LogicalPlan::Limit { limit, offset, child } => {
            let (fc, changed) = push_predicates(child);
            if changed {
                (LogicalPlan::Limit { limit: *limit, offset: *offset, child: Box::new(fc) }, true)
            } else {
                (plan.clone(), false)
            }
        }
        LogicalPlan::Distinct { child } => {
            let (fc, changed) = push_predicates(child);
            if changed {
                (LogicalPlan::Distinct { child: Box::new(fc) }, true)
            } else {
                (plan.clone(), false)
            }
        }
        LogicalPlan::SetOp { op, all, left, right } => {
            let (fl, lc) = push_predicates(left);
            let (fr, rc) = push_predicates(right);
            if lc || rc {
                (LogicalPlan::SetOp { op: *op, all: *all, left: Box::new(fl), right: Box::new(fr) }, true)
            } else {
                (plan.clone(), false)
            }
        }
        LogicalPlan::Insert { table_id, target_columns, source } => {
            let (fs, changed) = push_predicates(source);
            if changed {
                (LogicalPlan::Insert { table_id: *table_id, target_columns: target_columns.clone(), source: Box::new(fs) }, true)
            } else {
                (plan.clone(), false)
            }
        }
        LogicalPlan::Update { table_id, assignments, child } => {
            let (fc, changed) = push_predicates(child);
            if changed {
                (LogicalPlan::Update { table_id: *table_id, assignments: assignments.clone(), child: Box::new(fc) }, true)
            } else {
                (plan.clone(), false)
            }
        }
        LogicalPlan::Delete { table_id, child } => {
            let (fc, changed) = push_predicates(child);
            if changed {
                (LogicalPlan::Delete { table_id: *table_id, child: Box::new(fc) }, true)
            } else {
                (plan.clone(), false)
            }
        }
        other => (other.clone(), false),
    }
}

/// Splits an AND expression into its conjuncts.
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

/// Combines conjuncts into an AND expression.
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

/// Collects all table indices referenced in a plan's scan nodes.
fn collect_table_indices(plan: &LogicalPlan) -> Vec<usize> {
    let mut indices = Vec::new();
    collect_table_indices_recursive(plan, &mut indices);
    indices
}

fn collect_table_indices_recursive(plan: &LogicalPlan, out: &mut Vec<usize>) {
    match plan {
        LogicalPlan::Scan { table_idx, .. } => out.push(*table_idx),
        other => {
            for child in other.children() {
                collect_table_indices_recursive(child, out);
            }
        }
    }
}

/// Collects all column references in an expression.
fn collect_column_refs(expr: &BoundExpr) -> Vec<ColumnRef> {
    let mut refs = Vec::new();
    collect_column_refs_recursive(expr, &mut refs);
    refs
}

fn collect_column_refs_recursive(expr: &BoundExpr, out: &mut Vec<ColumnRef>) {
    match expr {
        BoundExpr::ColumnRef(cr) => out.push(*cr),
        BoundExpr::BinaryOp { left, right, .. } => {
            collect_column_refs_recursive(left, out);
            collect_column_refs_recursive(right, out);
        }
        BoundExpr::UnaryOp { expr, .. } => collect_column_refs_recursive(expr, out),
        BoundExpr::IsNull { expr, .. } => collect_column_refs_recursive(expr, out),
        BoundExpr::InList { expr, list, .. } => {
            collect_column_refs_recursive(expr, out);
            for item in list {
                collect_column_refs_recursive(item, out);
            }
        }
        BoundExpr::Between { expr, low, high, .. } => {
            collect_column_refs_recursive(expr, out);
            collect_column_refs_recursive(low, out);
            collect_column_refs_recursive(high, out);
        }
        BoundExpr::Like { expr, pattern, .. } | BoundExpr::ILike { expr, pattern, .. } => {
            collect_column_refs_recursive(expr, out);
            collect_column_refs_recursive(pattern, out);
        }
        BoundExpr::Function { args, .. } => {
            for arg in args {
                collect_column_refs_recursive(arg, out);
            }
        }
        BoundExpr::AggregateFunction { args, .. } => {
            for arg in args {
                collect_column_refs_recursive(arg, out);
            }
        }
        BoundExpr::Cast { expr, .. } => collect_column_refs_recursive(expr, out),
        BoundExpr::Nested(inner) => collect_column_refs_recursive(inner, out),
        BoundExpr::Case { operand, conditions, else_result, .. } => {
            if let Some(op) = operand {
                collect_column_refs_recursive(op, out);
            }
            for wc in conditions {
                collect_column_refs_recursive(&wc.condition, out);
                collect_column_refs_recursive(&wc.result, out);
            }
            if let Some(e) = else_result {
                collect_column_refs_recursive(e, out);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binder::ColumnRef;
    use zyron_catalog::ColumnId;
    use zyron_common::TypeId;
    use zyron_parser::ast::LiteralValue;

    fn make_col_ref(table_idx: usize, col: u16) -> BoundExpr {
        BoundExpr::ColumnRef(ColumnRef {
            table_idx,
            column_id: ColumnId(col),
            type_id: TypeId::Int64,
            nullable: false,
        })
    }

    fn make_lit_int(val: i64) -> BoundExpr {
        BoundExpr::Literal {
            value: LiteralValue::Integer(val),
            type_id: TypeId::Int64,
        }
    }

    #[test]
    fn test_split_conjuncts() {
        let pred = BoundExpr::BinaryOp {
            left: Box::new(make_col_ref(0, 0)),
            op: BinaryOperator::And,
            right: Box::new(make_col_ref(1, 0)),
            type_id: TypeId::Boolean,
        };
        let parts = split_conjuncts(&pred);
        assert_eq!(parts.len(), 2);
    }

    #[test]
    fn test_combine_conjuncts() {
        let a = make_col_ref(0, 0);
        let b = make_col_ref(1, 0);
        let combined = combine_conjuncts(vec![a, b]);
        assert!(matches!(combined, BoundExpr::BinaryOp { op: BinaryOperator::And, .. }));
    }

    #[test]
    fn test_collect_column_refs() {
        let expr = BoundExpr::BinaryOp {
            left: Box::new(make_col_ref(0, 0)),
            op: BinaryOperator::Eq,
            right: Box::new(make_lit_int(5)),
            type_id: TypeId::Boolean,
        };
        let refs = collect_column_refs(&expr);
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].table_idx, 0);
    }
}

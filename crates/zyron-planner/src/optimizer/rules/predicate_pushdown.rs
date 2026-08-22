//! Predicate pushdown optimization rule.
//!
//! Pushes filter predicates closer to table scans to reduce the number
//! of rows processed by upstream operators. Splits conjuncts across
//! join sides when possible.
//! Uses a changed-flag pattern to avoid cloning unchanged plan trees.

use crate::binder::{BoundExpr, ColumnRef};
use crate::logical::LogicalPlan;
use crate::optimizer::OptimizationRule;
use std::sync::Arc;
use zyron_catalog::Catalog;
use zyron_common::TypeId;
use zyron_parser::ast::{BinaryOperator, JoinType};

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
        LogicalPlan::Filter { predicate, child } => {
            let (child_plan, child_changed) = push_predicates(child);
            match &child_plan {
                LogicalPlan::Join {
                    left,
                    right,
                    join_type,
                    condition,
                } => {
                    let conjuncts = split_conjuncts(predicate);
                    let left_tables = collect_table_indices(left);
                    let right_tables = collect_table_indices(right);

                    // A predicate may only be pushed into a side that the join
                    // never fills with NULLs. Pushing a predicate into the
                    // null-supplying side of an outer join changes the result:
                    // it filters base rows before the join manufactures the
                    // unmatched NULL row, so the common `LEFT JOIN ... WHERE
                    // right.col IS NULL` anti-join would lose its semantics.
                    let (can_push_left, can_push_right) = match join_type {
                        JoinType::Inner | JoinType::Cross => (true, true),
                        JoinType::Left => (true, false),
                        JoinType::Right => (false, true),
                        JoinType::Full => (false, false),
                    };

                    let mut left_preds = Vec::new();
                    let mut right_preds = Vec::new();
                    let mut remaining = Vec::new();

                    for conj in conjuncts {
                        // A conjunct containing a subquery stays above the join.
                        // Its correlated columns may reference either side and are
                        // not seen by the column-ref scan, so pushing it into one
                        // side could strip a column the subquery needs.
                        if crate::binder::expr_contains_subquery(&conj) {
                            remaining.push(conj);
                            continue;
                        }
                        let refs = collect_column_refs(&conj);
                        let touches_left = refs.iter().any(|r| left_tables.contains(&r.table_idx));
                        let touches_right =
                            refs.iter().any(|r| right_tables.contains(&r.table_idx));

                        if touches_left && !touches_right && can_push_left {
                            left_preds.push(conj);
                        } else if touches_right && !touches_left && can_push_right {
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
                            child: Arc::new(left.as_ref().clone()),
                        }
                    };

                    let new_right = if right_preds.is_empty() {
                        right.as_ref().clone()
                    } else {
                        LogicalPlan::Filter {
                            predicate: combine_conjuncts(right_preds),
                            child: Arc::new(right.as_ref().clone()),
                        }
                    };

                    let (pushed_left, _) = push_predicates(&new_left);
                    let (pushed_right, _) = push_predicates(&new_right);

                    let join = LogicalPlan::Join {
                        left: Arc::new(pushed_left),
                        right: Arc::new(pushed_right),
                        join_type: *join_type,
                        condition: condition.clone(),
                    };

                    if remaining.is_empty() {
                        (join, true)
                    } else {
                        (
                            LogicalPlan::Filter {
                                predicate: combine_conjuncts(remaining),
                                child: Arc::new(join),
                            },
                            true,
                        )
                    }
                }
                // Filter above Project: push down conjuncts that reference only
                // columns the projection passes through verbatim. ColumnRef is a
                // stable (table_idx, column_id) identity, so a conjunct over
                // passthrough columns evaluates identically below the projection.
                // A conjunct touching a computed/aliased output stays above,
                // since that column does not exist beneath the projection.
                LogicalPlan::Project {
                    expressions,
                    aliases,
                    child: proj_child,
                    output_table_idx,
                } => {
                    let passthrough: Vec<(usize, u16)> = expressions
                        .iter()
                        .filter_map(|e| match e {
                            BoundExpr::ColumnRef(cr) => Some((cr.table_idx, cr.column_id.0)),
                            _ => None,
                        })
                        .collect();

                    let mut pushable = Vec::new();
                    let mut keep_above = Vec::new();
                    for conjunct in split_conjuncts(predicate) {
                        let refs = collect_column_refs(&conjunct);
                        let pushes = !refs.is_empty()
                            && refs
                                .iter()
                                .all(|r| passthrough.contains(&(r.table_idx, r.column_id.0)));
                        if pushes {
                            pushable.push(conjunct);
                        } else {
                            keep_above.push(conjunct);
                        }
                    }

                    if pushable.is_empty() {
                        let (pushed_proj_child, _) = push_predicates(proj_child);
                        (
                            LogicalPlan::Filter {
                                predicate: predicate.clone(),
                                child: Arc::new(LogicalPlan::Project {
                                    expressions: expressions.clone(),
                                    aliases: aliases.clone(),
                                    child: Arc::new(pushed_proj_child),
                                    output_table_idx: *output_table_idx,
                                }),
                            },
                            child_changed,
                        )
                    } else {
                        let filtered_child = LogicalPlan::Filter {
                            predicate: combine_conjuncts(pushable),
                            child: Arc::new(proj_child.as_ref().clone()),
                        };
                        let (pushed_child, _) = push_predicates(&filtered_child);
                        let project = LogicalPlan::Project {
                            expressions: expressions.clone(),
                            aliases: aliases.clone(),
                            child: Arc::new(pushed_child),
                            output_table_idx: *output_table_idx,
                        };
                        if keep_above.is_empty() {
                            (project, true)
                        } else {
                            (
                                LogicalPlan::Filter {
                                    predicate: combine_conjuncts(keep_above),
                                    child: Arc::new(project),
                                },
                                true,
                            )
                        }
                    }
                }
                _ => {
                    if child_changed {
                        (
                            LogicalPlan::Filter {
                                predicate: predicate.clone(),
                                child: Arc::new(child_plan),
                            },
                            true,
                        )
                    } else {
                        (plan.clone(), false)
                    }
                }
            }
        }
        // Recursively apply to all other node types
        LogicalPlan::Project {
            expressions,
            aliases,
            child,
            output_table_idx,
        } => {
            let (fc, changed) = push_predicates(child);
            if changed {
                (
                    LogicalPlan::Project {
                        expressions: expressions.clone(),
                        aliases: aliases.clone(),
                        child: Arc::new(fc),
                        output_table_idx: *output_table_idx,
                    },
                    true,
                )
            } else {
                (plan.clone(), false)
            }
        }
        LogicalPlan::Join {
            left,
            right,
            join_type,
            condition,
        } => {
            let (fl, lc) = push_predicates(left);
            let (fr, rc) = push_predicates(right);
            if lc || rc {
                (
                    LogicalPlan::Join {
                        left: Arc::new(fl),
                        right: Arc::new(fr),
                        join_type: *join_type,
                        condition: condition.clone(),
                    },
                    true,
                )
            } else {
                (plan.clone(), false)
            }
        }
        LogicalPlan::Aggregate {
            group_by,
            aggregates,
            child,
        } => {
            let (fc, changed) = push_predicates(child);
            if changed {
                (
                    LogicalPlan::Aggregate {
                        group_by: group_by.clone(),
                        aggregates: aggregates.clone(),
                        child: Arc::new(fc),
                    },
                    true,
                )
            } else {
                (plan.clone(), false)
            }
        }
        LogicalPlan::Sort { order_by, child } => {
            let (fc, changed) = push_predicates(child);
            if changed {
                (
                    LogicalPlan::Sort {
                        order_by: order_by.clone(),
                        child: Arc::new(fc),
                    },
                    true,
                )
            } else {
                (plan.clone(), false)
            }
        }
        LogicalPlan::Limit {
            limit,
            offset,
            child,
        } => {
            let (fc, changed) = push_predicates(child);
            if changed {
                (
                    LogicalPlan::Limit {
                        limit: *limit,
                        offset: *offset,
                        child: Arc::new(fc),
                    },
                    true,
                )
            } else {
                (plan.clone(), false)
            }
        }
        LogicalPlan::Distinct { child } => {
            let (fc, changed) = push_predicates(child);
            if changed {
                (
                    LogicalPlan::Distinct {
                        child: Arc::new(fc),
                    },
                    true,
                )
            } else {
                (plan.clone(), false)
            }
        }
        LogicalPlan::SetOp {
            op,
            all,
            left,
            right,
        } => {
            let (fl, lc) = push_predicates(left);
            let (fr, rc) = push_predicates(right);
            if lc || rc {
                (
                    LogicalPlan::SetOp {
                        op: *op,
                        all: *all,
                        left: Arc::new(fl),
                        right: Arc::new(fr),
                    },
                    true,
                )
            } else {
                (plan.clone(), false)
            }
        }
        LogicalPlan::Insert {
            table_id,
            target_columns,
            column_defaults,
            check_constraints,
            expectations,
            source,
        } => {
            let (fs, changed) = push_predicates(source);
            if changed {
                (
                    LogicalPlan::Insert {
                        table_id: *table_id,
                        target_columns: target_columns.clone(),
                        column_defaults: column_defaults.clone(),
                        check_constraints: check_constraints.clone(),
                        expectations: expectations.clone(),
                        source: Arc::new(fs),
                    },
                    true,
                )
            } else {
                (plan.clone(), false)
            }
        }
        LogicalPlan::Update {
            table_id,
            assignments,
            check_constraints,
            child,
        } => {
            let (fc, changed) = push_predicates(child);
            if changed {
                (
                    LogicalPlan::Update {
                        table_id: *table_id,
                        assignments: assignments.clone(),
                        check_constraints: check_constraints.clone(),
                        child: Arc::new(fc),
                    },
                    true,
                )
            } else {
                (plan.clone(), false)
            }
        }
        LogicalPlan::Delete { table_id, child } => {
            let (fc, changed) = push_predicates(child);
            if changed {
                (
                    LogicalPlan::Delete {
                        table_id: *table_id,
                        child: Arc::new(fc),
                    },
                    true,
                )
            } else {
                (plan.clone(), false)
            }
        }
        other => (other.clone(), false),
    }
}

/// Splits an AND expression into its conjuncts.
pub(crate) fn split_conjuncts(expr: &BoundExpr) -> Vec<BoundExpr> {
    match expr {
        BoundExpr::BinaryOp {
            left,
            op: BinaryOperator::And,
            right,
            ..
        } => {
            let mut result = split_conjuncts(left);
            result.extend(split_conjuncts(right));
            result
        }
        other => vec![other.clone()],
    }
}

/// Combines conjuncts into an AND expression.
pub(crate) fn combine_conjuncts(mut conjuncts: Vec<BoundExpr>) -> BoundExpr {
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

/// Collects the table indices a plan's output can be addressed by.
///
/// A derived table is addressed by the index its enclosing query gave it,
/// which its projection carries as `output_table_idx`, and the indices of
/// the scans inside it are not visible above it. Descending past that
/// projection would report the inner indices instead, and a caller deciding
/// which side of a join a predicate belongs to would then conclude the
/// predicate touches neither side and push it into the wrong one.
pub(crate) fn collect_table_indices(plan: &LogicalPlan) -> Vec<usize> {
    let mut indices = Vec::new();
    collect_table_indices_recursive(plan, &mut indices);
    indices
}

fn collect_table_indices_recursive(plan: &LogicalPlan, out: &mut Vec<usize>) {
    match plan {
        LogicalPlan::Scan { table_idx, .. } => out.push(*table_idx),
        // A relabeled projection is the boundary of a derived table: above
        // it only this index exists
        LogicalPlan::Project {
            output_table_idx: Some(idx),
            ..
        } => out.push(*idx),
        other => {
            for child in other.children() {
                collect_table_indices_recursive(child, out);
            }
        }
    }
}

/// Collects all column references in an expression.
pub(crate) fn collect_column_refs(expr: &BoundExpr) -> Vec<ColumnRef> {
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
        BoundExpr::Between {
            expr, low, high, ..
        } => {
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
        BoundExpr::Case {
            operand,
            conditions,
            else_result,
            ..
        } => {
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
        BoundExpr::InSubquery { expr, .. } => collect_column_refs_recursive(expr, out),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binder::ColumnRef;
    use crate::logical::JoinCondition;
    use zyron_catalog::ColumnId;
    use zyron_common::TypeId;
    use zyron_parser::ast::LiteralValue;

    fn make_col_ref(table_idx: usize, col: u16) -> BoundExpr {
        BoundExpr::ColumnRef(ColumnRef {
            table_idx,
            column_id: ColumnId(col),
            type_id: TypeId::Int64,
            nullable: false,
            fractional_digits: None,
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
        assert!(matches!(
            combined,
            BoundExpr::BinaryOp {
                op: BinaryOperator::And,
                ..
            }
        ));
    }

    fn make_is_null(table_idx: usize, col: u16) -> BoundExpr {
        BoundExpr::IsNull {
            expr: Box::new(make_col_ref(table_idx, col)),
            negated: false,
        }
    }

    fn scan(table_idx: usize) -> LogicalPlan {
        LogicalPlan::Scan {
            table_id: zyron_catalog::TableId(table_idx as u32),
            table_idx,
            columns: vec![],
            alias: String::new(),
            encoding_hints: None,
            as_of: None,
        }
    }

    // A right-side predicate over a LEFT join must stay above the join. Pushing
    // it into the null-supplying side would break the anti-join idiom.
    #[test]
    fn test_no_pushdown_into_null_supplying_side_of_left_join() {
        let join = LogicalPlan::Join {
            left: Arc::new(scan(0)),
            right: Arc::new(scan(1)),
            join_type: JoinType::Left,
            condition: JoinCondition::On(BoundExpr::BinaryOp {
                left: Box::new(make_col_ref(0, 0)),
                op: BinaryOperator::Eq,
                right: Box::new(make_col_ref(1, 0)),
                type_id: TypeId::Boolean,
            }),
        };
        let plan = LogicalPlan::Filter {
            predicate: make_is_null(1, 0),
            child: Arc::new(join),
        };
        let (pushed, changed) = push_predicates(&plan);
        assert!(
            !changed,
            "right-side predicate must not move below a LEFT join"
        );
        assert!(
            matches!(pushed, LogicalPlan::Filter { child, .. } if matches!(*child, LogicalPlan::Join { .. }))
        );
    }

    // The preserved side of a LEFT join still accepts pushed predicates.
    #[test]
    fn test_pushdown_into_preserved_side_of_left_join() {
        let join = LogicalPlan::Join {
            left: Arc::new(scan(0)),
            right: Arc::new(scan(1)),
            join_type: JoinType::Left,
            condition: JoinCondition::On(BoundExpr::BinaryOp {
                left: Box::new(make_col_ref(0, 0)),
                op: BinaryOperator::Eq,
                right: Box::new(make_col_ref(1, 0)),
                type_id: TypeId::Boolean,
            }),
        };
        let plan = LogicalPlan::Filter {
            predicate: BoundExpr::BinaryOp {
                left: Box::new(make_col_ref(0, 0)),
                op: BinaryOperator::Eq,
                right: Box::new(make_lit_int(5)),
                type_id: TypeId::Boolean,
            },
            child: Arc::new(join),
        };
        let (pushed, changed) = push_predicates(&plan);
        assert!(
            changed,
            "left-side predicate should push into the preserved side"
        );
        assert!(
            matches!(pushed, LogicalPlan::Join { .. }),
            "filter dissolves into the join"
        );
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

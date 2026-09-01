//! Predicate pushdown optimization rule.
//!
//! Pushes filter predicates closer to table scans to reduce the number
//! of rows processed by upstream operators. Splits conjuncts across
//! join sides when possible.
//! push_predicates returns None for an unchanged subtree, so untouched
//! nodes allocate nothing and unchanged plan children are reused by Arc.

use super::rebuilt_child;
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
        push_predicates(plan)
    }
}

/// Returns true if the plan tree contains any Filter node.
fn has_filter(plan: &LogicalPlan) -> bool {
    match plan {
        LogicalPlan::Filter { .. } => true,
        other => other.children().iter().any(|c| has_filter(c)),
    }
}

/// Returns the rewritten plan, or None when no predicate moved and no
/// child changed.
fn push_predicates(plan: &LogicalPlan) -> Option<LogicalPlan> {
    match plan {
        // Filter above Join: try to push predicates into join sides
        LogicalPlan::Filter { predicate, child } => {
            let pushed_child = push_predicates(child);
            let effective_child = pushed_child.as_ref().unwrap_or(child);
            match effective_child {
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

                    // Nothing pushed and the child kept its shape: unchanged
                    if left_preds.is_empty() && right_preds.is_empty() && pushed_child.is_none() {
                        return None;
                    }

                    // A side that receives predicates gets a Filter over the
                    // existing subtree by Arc and pushes again so the new
                    // filter cascades toward the scans. A side receiving
                    // nothing is already fully pushed and is reused as is
                    let new_left = if left_preds.is_empty() {
                        Arc::clone(left)
                    } else {
                        let filtered = LogicalPlan::Filter {
                            predicate: combine_conjuncts(left_preds),
                            child: Arc::clone(left),
                        };
                        Arc::new(match push_predicates(&filtered) {
                            Some(p) => p,
                            None => filtered,
                        })
                    };

                    let new_right = if right_preds.is_empty() {
                        Arc::clone(right)
                    } else {
                        let filtered = LogicalPlan::Filter {
                            predicate: combine_conjuncts(right_preds),
                            child: Arc::clone(right),
                        };
                        Arc::new(match push_predicates(&filtered) {
                            Some(p) => p,
                            None => filtered,
                        })
                    };

                    let join = LogicalPlan::Join {
                        left: new_left,
                        right: new_right,
                        join_type: *join_type,
                        condition: condition.clone(),
                    };

                    Some(if remaining.is_empty() {
                        join
                    } else {
                        LogicalPlan::Filter {
                            predicate: combine_conjuncts(remaining),
                            child: Arc::new(join),
                        }
                    })
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
                        // No conjunct crosses the projection, the filter only
                        // moves when the subtree underneath changed
                        let changed_child = pushed_child?;
                        return Some(LogicalPlan::Filter {
                            predicate: predicate.clone(),
                            child: Arc::new(changed_child),
                        });
                    }

                    let filtered_child = LogicalPlan::Filter {
                        predicate: combine_conjuncts(pushable),
                        child: Arc::clone(proj_child),
                    };
                    let pushed_inner = match push_predicates(&filtered_child) {
                        Some(p) => p,
                        None => filtered_child,
                    };
                    let project = LogicalPlan::Project {
                        expressions: expressions.clone(),
                        aliases: aliases.clone(),
                        child: Arc::new(pushed_inner),
                        output_table_idx: *output_table_idx,
                    };
                    Some(if keep_above.is_empty() {
                        project
                    } else {
                        LogicalPlan::Filter {
                            predicate: combine_conjuncts(keep_above),
                            child: Arc::new(project),
                        }
                    })
                }
                _ => pushed_child.map(|changed_child| LogicalPlan::Filter {
                    predicate: predicate.clone(),
                    child: Arc::new(changed_child),
                }),
            }
        }
        // Recursively apply to all other node types
        LogicalPlan::Project {
            expressions,
            aliases,
            child,
            output_table_idx,
        } => {
            let pushed = push_predicates(child)?;
            Some(LogicalPlan::Project {
                expressions: expressions.clone(),
                aliases: aliases.clone(),
                child: Arc::new(pushed),
                output_table_idx: *output_table_idx,
            })
        }
        LogicalPlan::Join {
            left,
            right,
            join_type,
            condition,
        } => {
            let pushed_left = push_predicates(left);
            let pushed_right = push_predicates(right);
            if pushed_left.is_none() && pushed_right.is_none() {
                return None;
            }
            Some(LogicalPlan::Join {
                left: rebuilt_child(left, pushed_left),
                right: rebuilt_child(right, pushed_right),
                join_type: *join_type,
                condition: condition.clone(),
            })
        }
        LogicalPlan::Aggregate {
            group_by,
            aggregates,
            child,
        } => {
            let pushed = push_predicates(child)?;
            Some(LogicalPlan::Aggregate {
                group_by: group_by.clone(),
                aggregates: aggregates.clone(),
                child: Arc::new(pushed),
            })
        }
        LogicalPlan::Sort { order_by, child } => {
            let pushed = push_predicates(child)?;
            Some(LogicalPlan::Sort {
                order_by: order_by.clone(),
                child: Arc::new(pushed),
            })
        }
        LogicalPlan::Limit {
            limit,
            offset,
            child,
        } => {
            let pushed = push_predicates(child)?;
            Some(LogicalPlan::Limit {
                limit: *limit,
                offset: *offset,
                child: Arc::new(pushed),
            })
        }
        LogicalPlan::Distinct { child } => {
            let pushed = push_predicates(child)?;
            Some(LogicalPlan::Distinct {
                child: Arc::new(pushed),
            })
        }
        LogicalPlan::SetOp {
            op,
            all,
            left,
            right,
        } => {
            let pushed_left = push_predicates(left);
            let pushed_right = push_predicates(right);
            if pushed_left.is_none() && pushed_right.is_none() {
                return None;
            }
            Some(LogicalPlan::SetOp {
                op: *op,
                all: *all,
                left: rebuilt_child(left, pushed_left),
                right: rebuilt_child(right, pushed_right),
            })
        }
        LogicalPlan::Insert {
            table_id,
            target_columns,
            column_defaults,
            check_constraints,
            expectations,
            generated_columns,
            source,
        } => {
            let pushed = push_predicates(source)?;
            Some(LogicalPlan::Insert {
                table_id: *table_id,
                target_columns: target_columns.clone(),
                column_defaults: column_defaults.clone(),
                check_constraints: check_constraints.clone(),
                expectations: expectations.clone(),
                generated_columns: generated_columns.clone(),
                source: Arc::new(pushed),
            })
        }
        LogicalPlan::Update {
            table_id,
            assignments,
            check_constraints,
            generated_columns,
            child,
        } => {
            let pushed = push_predicates(child)?;
            Some(LogicalPlan::Update {
                table_id: *table_id,
                assignments: assignments.clone(),
                check_constraints: check_constraints.clone(),
                generated_columns: generated_columns.clone(),
                child: Arc::new(pushed),
            })
        }
        LogicalPlan::Delete { table_id, child } => {
            let pushed = push_predicates(child)?;
            Some(LogicalPlan::Delete {
                table_id: *table_id,
                child: Arc::new(pushed),
            })
        }
        _other => None,
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
        assert!(
            push_predicates(&plan).is_none(),
            "right-side predicate must not move below a LEFT join"
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
        let pushed = push_predicates(&plan)
            .expect("left-side predicate should push into the preserved side");
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

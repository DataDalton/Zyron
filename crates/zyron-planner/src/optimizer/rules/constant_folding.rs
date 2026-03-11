//! Constant folding optimization rule.
//!
//! Evaluates constant sub-expressions at plan time and simplifies
//! boolean logic to reduce runtime computation.
//! Uses a changed-flag pattern to avoid cloning unchanged plan trees.

use crate::binder::BoundExpr;
use crate::logical::LogicalPlan;
use crate::optimizer::OptimizationRule;
use zyron_catalog::Catalog;
use zyron_common::TypeId;
use zyron_parser::ast::{BinaryOperator, LiteralValue};

pub struct ConstantFolding;

impl OptimizationRule for ConstantFolding {
    fn name(&self) -> &str {
        "constant_folding"
    }

    fn apply(&self, plan: &LogicalPlan, _catalog: &Catalog) -> Option<LogicalPlan> {
        let (folded, changed) = fold_plan(plan);
        if changed { Some(folded) } else { None }
    }
}

/// Returns (folded_plan, changed). Only clones nodes that actually change.
fn fold_plan(plan: &LogicalPlan) -> (LogicalPlan, bool) {
    match plan {
        LogicalPlan::Filter { predicate, child } => {
            let (folded_child, child_changed) = fold_plan(child);
            let (folded_pred, pred_changed) = fold_expr(predicate);

            // Filter with TRUE predicate: remove the filter
            if is_true_literal(&folded_pred) {
                return (folded_child, true);
            }

            // Filter with FALSE predicate: replace with empty Values
            if is_false_literal(&folded_pred) {
                return (LogicalPlan::Values {
                    rows: vec![],
                    schema: folded_child.output_schema(),
                }, true);
            }

            if child_changed || pred_changed {
                (LogicalPlan::Filter {
                    predicate: folded_pred,
                    child: Box::new(folded_child),
                }, true)
            } else {
                (plan.clone(), false)
            }
        }
        LogicalPlan::Project { expressions, aliases, child } => {
            let (folded_child, child_changed) = fold_plan(child);
            let mut any_expr_changed = false;
            let folded_exprs: Vec<BoundExpr> = expressions.iter().map(|e| {
                let (fe, changed) = fold_expr(e);
                if changed { any_expr_changed = true; }
                fe
            }).collect();

            if child_changed || any_expr_changed {
                (LogicalPlan::Project {
                    expressions: folded_exprs,
                    aliases: aliases.clone(),
                    child: Box::new(folded_child),
                }, true)
            } else {
                (plan.clone(), false)
            }
        }
        LogicalPlan::Join { left, right, join_type, condition } => {
            let (folded_left, left_changed) = fold_plan(left);
            let (folded_right, right_changed) = fold_plan(right);
            let (folded_condition, cond_changed) = match condition {
                crate::logical::JoinCondition::On(expr) => {
                    let (fe, changed) = fold_expr(expr);
                    (crate::logical::JoinCondition::On(fe), changed)
                }
                other => (other.clone(), false),
            };
            if left_changed || right_changed || cond_changed {
                (LogicalPlan::Join {
                    left: Box::new(folded_left),
                    right: Box::new(folded_right),
                    join_type: *join_type,
                    condition: folded_condition,
                }, true)
            } else {
                (plan.clone(), false)
            }
        }
        LogicalPlan::Aggregate { group_by, aggregates, child } => {
            let (folded_child, child_changed) = fold_plan(child);
            let mut any_changed = false;
            let folded_group_by: Vec<BoundExpr> = group_by.iter().map(|e| {
                let (fe, changed) = fold_expr(e);
                if changed { any_changed = true; }
                fe
            }).collect();
            if child_changed || any_changed {
                (LogicalPlan::Aggregate {
                    group_by: folded_group_by,
                    aggregates: aggregates.clone(),
                    child: Box::new(folded_child),
                }, true)
            } else {
                (plan.clone(), false)
            }
        }
        LogicalPlan::Sort { order_by, child } => {
            let (folded_child, changed) = fold_plan(child);
            if changed {
                (LogicalPlan::Sort {
                    order_by: order_by.clone(),
                    child: Box::new(folded_child),
                }, true)
            } else {
                (plan.clone(), false)
            }
        }
        LogicalPlan::Limit { limit, offset, child } => {
            let (folded_child, changed) = fold_plan(child);
            if changed {
                (LogicalPlan::Limit {
                    limit: *limit,
                    offset: *offset,
                    child: Box::new(folded_child),
                }, true)
            } else {
                (plan.clone(), false)
            }
        }
        LogicalPlan::Distinct { child } => {
            let (folded_child, changed) = fold_plan(child);
            if changed {
                (LogicalPlan::Distinct { child: Box::new(folded_child) }, true)
            } else {
                (plan.clone(), false)
            }
        }
        LogicalPlan::SetOp { op, all, left, right } => {
            let (fl, lc) = fold_plan(left);
            let (fr, rc) = fold_plan(right);
            if lc || rc {
                (LogicalPlan::SetOp {
                    op: *op,
                    all: *all,
                    left: Box::new(fl),
                    right: Box::new(fr),
                }, true)
            } else {
                (plan.clone(), false)
            }
        }
        LogicalPlan::Insert { table_id, target_columns, source } => {
            let (fs, changed) = fold_plan(source);
            if changed {
                (LogicalPlan::Insert {
                    table_id: *table_id,
                    target_columns: target_columns.clone(),
                    source: Box::new(fs),
                }, true)
            } else {
                (plan.clone(), false)
            }
        }
        LogicalPlan::Update { table_id, assignments, child } => {
            let (fc, changed) = fold_plan(child);
            if changed {
                (LogicalPlan::Update {
                    table_id: *table_id,
                    assignments: assignments.clone(),
                    child: Box::new(fc),
                }, true)
            } else {
                (plan.clone(), false)
            }
        }
        LogicalPlan::Delete { table_id, child } => {
            let (fc, changed) = fold_plan(child);
            if changed {
                (LogicalPlan::Delete {
                    table_id: *table_id,
                    child: Box::new(fc),
                }, true)
            } else {
                (plan.clone(), false)
            }
        }
        // Leaf nodes: no folding
        _other => (plan.clone(), false),
    }
}

/// Returns (folded_expr, changed).
fn fold_expr(expr: &BoundExpr) -> (BoundExpr, bool) {
    match expr {
        BoundExpr::BinaryOp { left, op, right, type_id } => {
            let (folded_left, lc) = fold_expr(left);
            let (folded_right, rc) = fold_expr(right);

            // Arithmetic on two integer literals
            if let (
                BoundExpr::Literal { value: LiteralValue::Integer(l), .. },
                BoundExpr::Literal { value: LiteralValue::Integer(r), .. },
            ) = (&folded_left, &folded_right)
            {
                if let Some(result) = fold_integer_op(*l, *op, *r) {
                    return (BoundExpr::Literal {
                        value: result,
                        type_id: *type_id,
                    }, true);
                }
            }

            // Arithmetic on two float literals
            if let (
                BoundExpr::Literal { value: LiteralValue::Float(l), .. },
                BoundExpr::Literal { value: LiteralValue::Float(r), .. },
            ) = (&folded_left, &folded_right)
            {
                if let Some(result) = fold_float_op(*l, *op, *r) {
                    return (BoundExpr::Literal {
                        value: result,
                        type_id: *type_id,
                    }, true);
                }
            }

            // Boolean simplification: x AND true -> x
            if *op == BinaryOperator::And {
                if is_true_literal(&folded_right) {
                    return (folded_left, true);
                }
                if is_true_literal(&folded_left) {
                    return (folded_right, true);
                }
                if is_false_literal(&folded_left) || is_false_literal(&folded_right) {
                    return (BoundExpr::Literal {
                        value: LiteralValue::Boolean(false),
                        type_id: TypeId::Boolean,
                    }, true);
                }
            }

            // Boolean simplification: x OR true -> true
            if *op == BinaryOperator::Or {
                if is_true_literal(&folded_left) || is_true_literal(&folded_right) {
                    return (BoundExpr::Literal {
                        value: LiteralValue::Boolean(true),
                        type_id: TypeId::Boolean,
                    }, true);
                }
                if is_false_literal(&folded_right) {
                    return (folded_left, true);
                }
                if is_false_literal(&folded_left) {
                    return (folded_right, true);
                }
            }

            if lc || rc {
                (BoundExpr::BinaryOp {
                    left: Box::new(folded_left),
                    op: *op,
                    right: Box::new(folded_right),
                    type_id: *type_id,
                }, true)
            } else {
                (expr.clone(), false)
            }
        }
        BoundExpr::UnaryOp { op: zyron_parser::ast::UnaryOperator::Not, expr: inner, type_id } => {
            let (folded, changed) = fold_expr(inner);
            if let BoundExpr::Literal { value: LiteralValue::Boolean(b), .. } = &folded {
                return (BoundExpr::Literal {
                    value: LiteralValue::Boolean(!b),
                    type_id: TypeId::Boolean,
                }, true);
            }
            if changed {
                (BoundExpr::UnaryOp {
                    op: zyron_parser::ast::UnaryOperator::Not,
                    expr: Box::new(folded),
                    type_id: *type_id,
                }, true)
            } else {
                (expr.clone(), false)
            }
        }
        BoundExpr::UnaryOp { op: zyron_parser::ast::UnaryOperator::Minus, expr: inner, type_id } => {
            let (folded, changed) = fold_expr(inner);
            if let BoundExpr::Literal { value: LiteralValue::Integer(n), .. } = &folded {
                return (BoundExpr::Literal {
                    value: LiteralValue::Integer(-n),
                    type_id: *type_id,
                }, true);
            }
            if let BoundExpr::Literal { value: LiteralValue::Float(n), .. } = &folded {
                return (BoundExpr::Literal {
                    value: LiteralValue::Float(-n),
                    type_id: *type_id,
                }, true);
            }
            if changed {
                (BoundExpr::UnaryOp {
                    op: zyron_parser::ast::UnaryOperator::Minus,
                    expr: Box::new(folded),
                    type_id: *type_id,
                }, true)
            } else {
                (expr.clone(), false)
            }
        }
        BoundExpr::IsNull { expr: inner, negated } => {
            let (folded, changed) = fold_expr(inner);
            if let BoundExpr::Literal { value: LiteralValue::Null, .. } = &folded {
                return (BoundExpr::Literal {
                    value: LiteralValue::Boolean(!negated),
                    type_id: TypeId::Boolean,
                }, true);
            }
            if matches!(&folded, BoundExpr::Literal { value, .. } if !matches!(value, LiteralValue::Null))
            {
                return (BoundExpr::Literal {
                    value: LiteralValue::Boolean(*negated),
                    type_id: TypeId::Boolean,
                }, true);
            }
            if changed {
                (BoundExpr::IsNull {
                    expr: Box::new(folded),
                    negated: *negated,
                }, true)
            } else {
                (expr.clone(), false)
            }
        }
        BoundExpr::Nested(inner) => fold_expr(inner),
        BoundExpr::Cast { expr: inner, target_type } => {
            let (folded, changed) = fold_expr(inner);
            if changed {
                (BoundExpr::Cast {
                    expr: Box::new(folded),
                    target_type: *target_type,
                }, true)
            } else {
                (expr.clone(), false)
            }
        }
        // No folding for other expression types
        _other => (expr.clone(), false),
    }
}

fn fold_integer_op(left: i64, op: BinaryOperator, right: i64) -> Option<LiteralValue> {
    match op {
        BinaryOperator::Plus => left.checked_add(right).map(LiteralValue::Integer),
        BinaryOperator::Minus => left.checked_sub(right).map(LiteralValue::Integer),
        BinaryOperator::Multiply => left.checked_mul(right).map(LiteralValue::Integer),
        BinaryOperator::Divide => {
            if right != 0 {
                left.checked_div(right).map(LiteralValue::Integer)
            } else {
                None
            }
        }
        BinaryOperator::Modulo => {
            if right != 0 {
                left.checked_rem(right).map(LiteralValue::Integer)
            } else {
                None
            }
        }
        BinaryOperator::Eq => Some(LiteralValue::Boolean(left == right)),
        BinaryOperator::Neq => Some(LiteralValue::Boolean(left != right)),
        BinaryOperator::Lt => Some(LiteralValue::Boolean(left < right)),
        BinaryOperator::Gt => Some(LiteralValue::Boolean(left > right)),
        BinaryOperator::LtEq => Some(LiteralValue::Boolean(left <= right)),
        BinaryOperator::GtEq => Some(LiteralValue::Boolean(left >= right)),
        _ => None,
    }
}

fn fold_float_op(left: f64, op: BinaryOperator, right: f64) -> Option<LiteralValue> {
    match op {
        BinaryOperator::Plus => Some(LiteralValue::Float(left + right)),
        BinaryOperator::Minus => Some(LiteralValue::Float(left - right)),
        BinaryOperator::Multiply => Some(LiteralValue::Float(left * right)),
        BinaryOperator::Divide => {
            if right != 0.0 {
                Some(LiteralValue::Float(left / right))
            } else {
                None
            }
        }
        BinaryOperator::Eq => Some(LiteralValue::Boolean(left == right)),
        BinaryOperator::Neq => Some(LiteralValue::Boolean(left != right)),
        BinaryOperator::Lt => Some(LiteralValue::Boolean(left < right)),
        BinaryOperator::Gt => Some(LiteralValue::Boolean(left > right)),
        BinaryOperator::LtEq => Some(LiteralValue::Boolean(left <= right)),
        BinaryOperator::GtEq => Some(LiteralValue::Boolean(left >= right)),
        _ => None,
    }
}

fn is_true_literal(expr: &BoundExpr) -> bool {
    matches!(expr, BoundExpr::Literal { value: LiteralValue::Boolean(true), .. })
}

fn is_false_literal(expr: &BoundExpr) -> bool {
    matches!(expr, BoundExpr::Literal { value: LiteralValue::Boolean(false), .. })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fold_integer_addition() {
        let expr = BoundExpr::BinaryOp {
            left: Box::new(BoundExpr::Literal {
                value: LiteralValue::Integer(3),
                type_id: TypeId::Int64,
            }),
            op: BinaryOperator::Plus,
            right: Box::new(BoundExpr::Literal {
                value: LiteralValue::Integer(4),
                type_id: TypeId::Int64,
            }),
            type_id: TypeId::Int64,
        };
        let (folded, changed) = fold_expr(&expr);
        assert!(changed);
        assert!(matches!(
            folded,
            BoundExpr::Literal { value: LiteralValue::Integer(7), .. }
        ));
    }

    #[test]
    fn test_fold_and_true() {
        let expr = BoundExpr::BinaryOp {
            left: Box::new(BoundExpr::ColumnRef(crate::binder::ColumnRef {
                table_idx: 0,
                column_id: zyron_catalog::ColumnId(0),
                type_id: TypeId::Boolean,
                nullable: false,
            })),
            op: BinaryOperator::And,
            right: Box::new(BoundExpr::Literal {
                value: LiteralValue::Boolean(true),
                type_id: TypeId::Boolean,
            }),
            type_id: TypeId::Boolean,
        };
        let (folded, changed) = fold_expr(&expr);
        assert!(changed);
        assert!(matches!(folded, BoundExpr::ColumnRef(_)));
    }

    #[test]
    fn test_fold_and_false() {
        let expr = BoundExpr::BinaryOp {
            left: Box::new(BoundExpr::ColumnRef(crate::binder::ColumnRef {
                table_idx: 0,
                column_id: zyron_catalog::ColumnId(0),
                type_id: TypeId::Boolean,
                nullable: false,
            })),
            op: BinaryOperator::And,
            right: Box::new(BoundExpr::Literal {
                value: LiteralValue::Boolean(false),
                type_id: TypeId::Boolean,
            }),
            type_id: TypeId::Boolean,
        };
        let (folded, changed) = fold_expr(&expr);
        assert!(changed);
        assert!(matches!(
            folded,
            BoundExpr::Literal { value: LiteralValue::Boolean(false), .. }
        ));
    }

    #[test]
    fn test_fold_is_null_on_null_literal() {
        let expr = BoundExpr::IsNull {
            expr: Box::new(BoundExpr::Literal {
                value: LiteralValue::Null,
                type_id: TypeId::Null,
            }),
            negated: false,
        };
        let (folded, changed) = fold_expr(&expr);
        assert!(changed);
        assert!(matches!(
            folded,
            BoundExpr::Literal { value: LiteralValue::Boolean(true), .. }
        ));
    }

    #[test]
    fn test_fold_not_true() {
        let expr = BoundExpr::UnaryOp {
            op: zyron_parser::ast::UnaryOperator::Not,
            expr: Box::new(BoundExpr::Literal {
                value: LiteralValue::Boolean(true),
                type_id: TypeId::Boolean,
            }),
            type_id: TypeId::Boolean,
        };
        let (folded, changed) = fold_expr(&expr);
        assert!(changed);
        assert!(matches!(
            folded,
            BoundExpr::Literal { value: LiteralValue::Boolean(false), .. }
        ));
    }

    #[test]
    fn test_fold_division_by_zero_returns_none() {
        assert!(fold_integer_op(10, BinaryOperator::Divide, 0).is_none());
    }

    #[test]
    fn test_fold_comparison() {
        assert_eq!(
            fold_integer_op(5, BinaryOperator::Lt, 10),
            Some(LiteralValue::Boolean(true))
        );
        assert_eq!(
            fold_integer_op(10, BinaryOperator::Lt, 5),
            Some(LiteralValue::Boolean(false))
        );
    }
}

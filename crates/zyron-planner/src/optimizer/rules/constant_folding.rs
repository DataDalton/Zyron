//! Constant folding optimization rule.
//!
//! Evaluates constant sub-expressions at plan time and simplifies
//! boolean logic to reduce runtime computation.
//! The fold functions return None for an unchanged subtree, so untouched
//! nodes allocate nothing and unchanged plan children are reused by Arc.

use crate::binder::BoundExpr;
use crate::logical::LogicalPlan;
use crate::optimizer::OptimizationRule;
use std::sync::Arc;
use zyron_catalog::Catalog;
use zyron_common::TypeId;
use zyron_parser::ast::{BinaryOperator, LiteralValue};

pub struct ConstantFolding;

impl OptimizationRule for ConstantFolding {
    fn name(&self) -> &str {
        "constant_folding"
    }

    fn apply(&self, plan: &LogicalPlan, _catalog: &Catalog) -> Option<LogicalPlan> {
        fold_plan(plan)
    }
}

use super::rebuilt_child;

/// Rebuilds a folded child expression, cloning the original only when the
/// parent is being rebuilt around an unchanged child
fn child_box(original: &BoundExpr, folded: Option<BoundExpr>) -> Box<BoundExpr> {
    match folded {
        Some(e) => Box::new(e),
        None => Box::new(original.clone()),
    }
}

/// Folds a list of expressions. None when every element is unchanged,
/// otherwise the fully rebuilt list
fn fold_expr_list(exprs: &[BoundExpr]) -> Option<Vec<BoundExpr>> {
    let folded: Vec<Option<BoundExpr>> = exprs.iter().map(fold_expr).collect();
    if folded.iter().all(Option::is_none) {
        return None;
    }
    Some(
        folded
            .into_iter()
            .zip(exprs.iter())
            .map(|(f, orig)| f.unwrap_or_else(|| orig.clone()))
            .collect(),
    )
}

/// Returns the folded plan, or None when nothing under this node changed.
fn fold_plan(plan: &LogicalPlan) -> Option<LogicalPlan> {
    match plan {
        LogicalPlan::Filter { predicate, child } => {
            let folded_child = fold_plan(child);
            let folded_pred = fold_expr(predicate);
            let effective_pred = folded_pred.as_ref().unwrap_or(predicate);

            // Filter with TRUE predicate: remove the filter
            if is_true_literal(effective_pred) {
                return Some(match folded_child {
                    Some(p) => p,
                    None => (**child).clone(),
                });
            }

            // Filter with FALSE predicate: replace with empty Values
            if is_false_literal(effective_pred) {
                let schema = match &folded_child {
                    Some(p) => p.output_schema(),
                    None => child.output_schema(),
                };
                return Some(LogicalPlan::Values {
                    rows: vec![],
                    schema,
                });
            }

            if folded_child.is_none() && folded_pred.is_none() {
                return None;
            }
            Some(LogicalPlan::Filter {
                predicate: folded_pred.unwrap_or_else(|| predicate.clone()),
                child: rebuilt_child(child, folded_child),
            })
        }
        LogicalPlan::Project {
            expressions,
            aliases,
            child,
            output_table_idx,
        } => {
            let folded_child = fold_plan(child);
            let folded_exprs = fold_expr_list(expressions);
            if folded_child.is_none() && folded_exprs.is_none() {
                return None;
            }
            Some(LogicalPlan::Project {
                expressions: folded_exprs.unwrap_or_else(|| expressions.clone()),
                aliases: aliases.clone(),
                child: rebuilt_child(child, folded_child),
                output_table_idx: *output_table_idx,
            })
        }
        LogicalPlan::Join {
            left,
            right,
            join_type,
            condition,
        } => {
            let folded_left = fold_plan(left);
            let folded_right = fold_plan(right);
            let folded_condition = match condition {
                crate::logical::JoinCondition::On(expr) => {
                    fold_expr(expr).map(crate::logical::JoinCondition::On)
                }
                _ => None,
            };
            if folded_left.is_none() && folded_right.is_none() && folded_condition.is_none() {
                return None;
            }
            Some(LogicalPlan::Join {
                left: rebuilt_child(left, folded_left),
                right: rebuilt_child(right, folded_right),
                join_type: *join_type,
                condition: folded_condition.unwrap_or_else(|| condition.clone()),
            })
        }
        LogicalPlan::Aggregate {
            group_by,
            aggregates,
            child,
        } => {
            let folded_child = fold_plan(child);
            let folded_group_by = fold_expr_list(group_by);
            if folded_child.is_none() && folded_group_by.is_none() {
                return None;
            }
            Some(LogicalPlan::Aggregate {
                group_by: folded_group_by.unwrap_or_else(|| group_by.clone()),
                aggregates: aggregates.clone(),
                child: rebuilt_child(child, folded_child),
            })
        }
        LogicalPlan::Sort { order_by, child } => {
            let folded_child = fold_plan(child)?;
            Some(LogicalPlan::Sort {
                order_by: order_by.clone(),
                child: Arc::new(folded_child),
            })
        }
        LogicalPlan::Limit {
            limit,
            offset,
            child,
        } => {
            let folded_child = fold_plan(child)?;
            Some(LogicalPlan::Limit {
                limit: *limit,
                offset: *offset,
                child: Arc::new(folded_child),
            })
        }
        LogicalPlan::Distinct { child } => {
            let folded_child = fold_plan(child)?;
            Some(LogicalPlan::Distinct {
                child: Arc::new(folded_child),
            })
        }
        LogicalPlan::SetOp {
            op,
            all,
            left,
            right,
        } => {
            let folded_left = fold_plan(left);
            let folded_right = fold_plan(right);
            if folded_left.is_none() && folded_right.is_none() {
                return None;
            }
            Some(LogicalPlan::SetOp {
                op: *op,
                all: *all,
                left: rebuilt_child(left, folded_left),
                right: rebuilt_child(right, folded_right),
            })
        }
        LogicalPlan::Insert {
            table_id,
            target_columns,
            column_defaults,
            check_constraints,
            expectations,
            source,
        } => {
            let folded_source = fold_plan(source)?;
            Some(LogicalPlan::Insert {
                table_id: *table_id,
                target_columns: target_columns.clone(),
                column_defaults: column_defaults.clone(),
                check_constraints: check_constraints.clone(),
                expectations: expectations.clone(),
                source: Arc::new(folded_source),
            })
        }
        LogicalPlan::Update {
            table_id,
            assignments,
            check_constraints,
            child,
        } => {
            let folded_child = fold_plan(child)?;
            Some(LogicalPlan::Update {
                table_id: *table_id,
                assignments: assignments.clone(),
                check_constraints: check_constraints.clone(),
                child: Arc::new(folded_child),
            })
        }
        LogicalPlan::Delete { table_id, child } => {
            let folded_child = fold_plan(child)?;
            Some(LogicalPlan::Delete {
                table_id: *table_id,
                child: Arc::new(folded_child),
            })
        }
        // Leaf nodes: no folding
        _other => None,
    }
}

/// Returns the folded expression, or None when nothing under it changed.
fn fold_expr(expr: &BoundExpr) -> Option<BoundExpr> {
    match expr {
        BoundExpr::BinaryOp {
            left,
            op,
            right,
            type_id,
        } => {
            let folded_left = fold_expr(left);
            let folded_right = fold_expr(right);
            let effective_left = folded_left.as_ref().unwrap_or(left);
            let effective_right = folded_right.as_ref().unwrap_or(right);

            // Arithmetic on two integer literals
            if let (
                BoundExpr::Literal {
                    value: LiteralValue::Integer(l),
                    ..
                },
                BoundExpr::Literal {
                    value: LiteralValue::Integer(r),
                    ..
                },
            ) = (effective_left, effective_right)
            {
                if let Some(result) = fold_integer_op(*l, *op, *r) {
                    return Some(BoundExpr::Literal {
                        value: result,
                        type_id: *type_id,
                    });
                }
            }

            // Arithmetic on two float literals
            if let (
                BoundExpr::Literal {
                    value: LiteralValue::Float(l),
                    ..
                },
                BoundExpr::Literal {
                    value: LiteralValue::Float(r),
                    ..
                },
            ) = (effective_left, effective_right)
            {
                if let Some(result) = fold_float_op(*l, *op, *r) {
                    return Some(BoundExpr::Literal {
                        value: result,
                        type_id: *type_id,
                    });
                }
            }

            // Boolean simplification: x AND true -> x
            if *op == BinaryOperator::And {
                if is_true_literal(effective_right) {
                    return Some(match folded_left {
                        Some(e) => e,
                        None => (**left).clone(),
                    });
                }
                if is_true_literal(effective_left) {
                    return Some(match folded_right {
                        Some(e) => e,
                        None => (**right).clone(),
                    });
                }
                if is_false_literal(effective_left) || is_false_literal(effective_right) {
                    return Some(BoundExpr::Literal {
                        value: LiteralValue::Boolean(false),
                        type_id: TypeId::Boolean,
                    });
                }
            }

            // Boolean simplification: x OR true -> true
            if *op == BinaryOperator::Or {
                if is_true_literal(effective_left) || is_true_literal(effective_right) {
                    return Some(BoundExpr::Literal {
                        value: LiteralValue::Boolean(true),
                        type_id: TypeId::Boolean,
                    });
                }
                if is_false_literal(effective_right) {
                    return Some(match folded_left {
                        Some(e) => e,
                        None => (**left).clone(),
                    });
                }
                if is_false_literal(effective_left) {
                    return Some(match folded_right {
                        Some(e) => e,
                        None => (**right).clone(),
                    });
                }
            }

            if folded_left.is_none() && folded_right.is_none() {
                return None;
            }
            Some(BoundExpr::BinaryOp {
                left: child_box(left, folded_left),
                op: *op,
                right: child_box(right, folded_right),
                type_id: *type_id,
            })
        }
        BoundExpr::UnaryOp {
            op: zyron_parser::ast::UnaryOperator::Not,
            expr: inner,
            type_id,
        } => {
            let folded = fold_expr(inner);
            if let BoundExpr::Literal {
                value: LiteralValue::Boolean(b),
                ..
            } = folded.as_ref().unwrap_or(inner)
            {
                return Some(BoundExpr::Literal {
                    value: LiteralValue::Boolean(!b),
                    type_id: TypeId::Boolean,
                });
            }
            let folded_inner = folded?;
            Some(BoundExpr::UnaryOp {
                op: zyron_parser::ast::UnaryOperator::Not,
                expr: Box::new(folded_inner),
                type_id: *type_id,
            })
        }
        BoundExpr::UnaryOp {
            op: zyron_parser::ast::UnaryOperator::Minus,
            expr: inner,
            type_id,
        } => {
            let folded = fold_expr(inner);
            if let BoundExpr::Literal {
                value: LiteralValue::Integer(n),
                ..
            } = folded.as_ref().unwrap_or(inner)
            {
                return Some(BoundExpr::Literal {
                    value: LiteralValue::Integer(-n),
                    type_id: *type_id,
                });
            }
            if let BoundExpr::Literal {
                value: LiteralValue::Float(n),
                ..
            } = folded.as_ref().unwrap_or(inner)
            {
                return Some(BoundExpr::Literal {
                    value: LiteralValue::Float(-n),
                    type_id: *type_id,
                });
            }
            let folded_inner = folded?;
            Some(BoundExpr::UnaryOp {
                op: zyron_parser::ast::UnaryOperator::Minus,
                expr: Box::new(folded_inner),
                type_id: *type_id,
            })
        }
        BoundExpr::IsNull {
            expr: inner,
            negated,
        } => {
            let folded = fold_expr(inner);
            let effective = folded.as_ref().unwrap_or(inner);
            if matches!(
                effective,
                BoundExpr::Literal {
                    value: LiteralValue::Null,
                    ..
                }
            ) {
                return Some(BoundExpr::Literal {
                    value: LiteralValue::Boolean(!negated),
                    type_id: TypeId::Boolean,
                });
            }
            if matches!(effective, BoundExpr::Literal { value, .. } if !matches!(value, LiteralValue::Null))
            {
                return Some(BoundExpr::Literal {
                    value: LiteralValue::Boolean(*negated),
                    type_id: TypeId::Boolean,
                });
            }
            let folded_inner = folded?;
            Some(BoundExpr::IsNull {
                expr: Box::new(folded_inner),
                negated: *negated,
            })
        }
        BoundExpr::Nested(inner) => fold_expr(inner),
        BoundExpr::Cast {
            expr: inner,
            target_type,
            fractional_digits,
        } => {
            let folded_inner = fold_expr(inner)?;
            Some(BoundExpr::Cast {
                fractional_digits: *fractional_digits,
                expr: Box::new(folded_inner),
                target_type: *target_type,
            })
        }
        // No folding for other expression types
        _other => None,
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
    matches!(
        expr,
        BoundExpr::Literal {
            value: LiteralValue::Boolean(true),
            ..
        }
    )
}

fn is_false_literal(expr: &BoundExpr) -> bool {
    matches!(
        expr,
        BoundExpr::Literal {
            value: LiteralValue::Boolean(false),
            ..
        }
    )
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
        let folded = fold_expr(&expr).expect("constant arithmetic folds");
        assert!(matches!(
            folded,
            BoundExpr::Literal {
                value: LiteralValue::Integer(7),
                ..
            }
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
                fractional_digits: None,
            })),
            op: BinaryOperator::And,
            right: Box::new(BoundExpr::Literal {
                value: LiteralValue::Boolean(true),
                type_id: TypeId::Boolean,
            }),
            type_id: TypeId::Boolean,
        };
        let folded = fold_expr(&expr).expect("AND true simplifies");
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
                fractional_digits: None,
            })),
            op: BinaryOperator::And,
            right: Box::new(BoundExpr::Literal {
                value: LiteralValue::Boolean(false),
                type_id: TypeId::Boolean,
            }),
            type_id: TypeId::Boolean,
        };
        let folded = fold_expr(&expr).expect("AND false simplifies");
        assert!(matches!(
            folded,
            BoundExpr::Literal {
                value: LiteralValue::Boolean(false),
                ..
            }
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
        let folded = fold_expr(&expr).expect("IS NULL on literal folds");
        assert!(matches!(
            folded,
            BoundExpr::Literal {
                value: LiteralValue::Boolean(true),
                ..
            }
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
        let folded = fold_expr(&expr).expect("NOT literal folds");
        assert!(matches!(
            folded,
            BoundExpr::Literal {
                value: LiteralValue::Boolean(false),
                ..
            }
        ));
    }

    #[test]
    fn test_unchanged_expression_returns_none() {
        let expr = BoundExpr::ColumnRef(crate::binder::ColumnRef {
            table_idx: 0,
            column_id: zyron_catalog::ColumnId(0),
            type_id: TypeId::Int64,
            nullable: false,
            fractional_digits: None,
        });
        assert!(fold_expr(&expr).is_none());
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

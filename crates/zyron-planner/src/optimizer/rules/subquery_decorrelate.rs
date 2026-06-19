//! Subquery decorrelation optimization rule.
//!
//! Converts correlated subqueries into joins where possible.
//! Handles EXISTS -> SemiJoin and IN subquery -> SemiJoin patterns.

use crate::binder::BoundExpr;
use crate::logical::{JoinCondition, LogicalPlan};
use crate::optimizer::OptimizationRule;
use std::sync::Arc;
use zyron_catalog::Catalog;
use zyron_common::TypeId;
use zyron_parser::ast::{BinaryOperator, JoinType};

pub struct SubqueryDecorrelate;

impl OptimizationRule for SubqueryDecorrelate {
    fn name(&self) -> &str {
        "subquery_decorrelate"
    }

    fn apply(&self, plan: &LogicalPlan, _catalog: &Catalog) -> Option<LogicalPlan> {
        // Quick check: only apply if the plan has a Filter with subquery expressions.
        if !has_subquery_filter(plan) {
            return None;
        }
        let decorrelated = decorrelate_plan(plan);
        if decorrelated != *plan {
            Some(decorrelated)
        } else {
            None
        }
    }
}

/// Returns true if a plan has a Filter whose predicate contains Exists or InSubquery.
fn has_subquery_filter(plan: &LogicalPlan) -> bool {
    match plan {
        LogicalPlan::Filter { predicate, child } => {
            has_subquery_expr(predicate) || has_subquery_filter(child)
        }
        other => other.children().iter().any(|c| has_subquery_filter(c)),
    }
}

fn has_subquery_expr(expr: &BoundExpr) -> bool {
    match expr {
        BoundExpr::Exists { .. } | BoundExpr::InSubquery { .. } | BoundExpr::Subquery { .. } => {
            true
        }
        BoundExpr::BinaryOp { left, right, .. } => {
            has_subquery_expr(left) || has_subquery_expr(right)
        }
        BoundExpr::UnaryOp { expr, .. } => has_subquery_expr(expr),
        BoundExpr::Nested(inner) => has_subquery_expr(inner),
        _ => false,
    }
}

fn decorrelate_plan(plan: &LogicalPlan) -> LogicalPlan {
    match plan {
        // Filter with EXISTS subquery -> SemiJoin
        LogicalPlan::Filter { predicate, child } => {
            let child = decorrelate_plan(child);

            if let Some(semi_join) = try_decorrelate_exists(predicate, &child) {
                return semi_join;
            }

            if let Some(semi_join) = try_decorrelate_in_subquery(predicate, &child) {
                return semi_join;
            }

            LogicalPlan::Filter {
                predicate: predicate.clone(),
                child: Arc::new(child),
            }
        }
        // Recursively process children
        LogicalPlan::Project {
            expressions,
            aliases,
            child,
            output_table_idx,
        } => LogicalPlan::Project {
            expressions: expressions.clone(),
            aliases: aliases.clone(),
            child: Arc::new(decorrelate_plan(child)),
            output_table_idx: *output_table_idx,
        },
        LogicalPlan::Join {
            left,
            right,
            join_type,
            condition,
        } => LogicalPlan::Join {
            left: Arc::new(decorrelate_plan(left)),
            right: Arc::new(decorrelate_plan(right)),
            join_type: *join_type,
            condition: condition.clone(),
        },
        LogicalPlan::Aggregate {
            group_by,
            aggregates,
            child,
        } => LogicalPlan::Aggregate {
            group_by: group_by.clone(),
            aggregates: aggregates.clone(),
            child: Arc::new(decorrelate_plan(child)),
        },
        LogicalPlan::Sort { order_by, child } => LogicalPlan::Sort {
            order_by: order_by.clone(),
            child: Arc::new(decorrelate_plan(child)),
        },
        LogicalPlan::Limit {
            limit,
            offset,
            child,
        } => LogicalPlan::Limit {
            limit: *limit,
            offset: *offset,
            child: Arc::new(decorrelate_plan(child)),
        },
        LogicalPlan::Distinct { child } => LogicalPlan::Distinct {
            child: Arc::new(decorrelate_plan(child)),
        },
        other => other.clone(),
    }
}

/// Tries to convert EXISTS(subquery) into a semi-join.
fn try_decorrelate_exists(predicate: &BoundExpr, child: &LogicalPlan) -> Option<LogicalPlan> {
    match predicate {
        BoundExpr::Exists {
            plan: subquery,
            negated: false,
        } => {
            // A correlated EXISTS references the outer row and is evaluated per
            // row by the executor's correlated filter; only an uncorrelated one
            // is rewritten to a join here.
            if crate::binder::subquery_is_correlated(subquery) {
                return None;
            }
            // Convert the subquery's FROM into a join with the outer plan.
            // This is a simplified decorrelation: wrap the subquery as a Distinct + Join.
            let subquery_plan = crate::logical::builder::build_logical_plan(
                &crate::binder::BoundStatement::Select(*subquery.clone()),
            )
            .ok()?;

            Some(LogicalPlan::Join {
                left: Arc::new(child.clone()),
                right: Arc::new(LogicalPlan::Distinct {
                    child: Arc::new(subquery_plan),
                }),
                join_type: JoinType::Inner,
                condition: JoinCondition::Cross,
            })
        }
        _ => None,
    }
}

/// Tries to convert IN(subquery) into a semi-join with equality condition.
fn try_decorrelate_in_subquery(predicate: &BoundExpr, child: &LogicalPlan) -> Option<LogicalPlan> {
    match predicate {
        BoundExpr::InSubquery {
            expr,
            plan: subquery,
            negated: false,
        } => {
            // A correlated IN is evaluated per outer row by the executor; only
            // an uncorrelated one is rewritten to a semi-join here.
            if crate::binder::subquery_is_correlated(subquery) {
                return None;
            }
            let subquery_plan = crate::logical::builder::build_logical_plan(
                &crate::binder::BoundStatement::Select(*subquery.clone()),
            )
            .ok()?;

            // Build equality condition between the outer expr and the first output column of the subquery
            let subquery_schema = subquery_plan.output_schema();
            if subquery_schema.is_empty() {
                return None;
            }

            let join_condition = BoundExpr::BinaryOp {
                left: Box::new(expr.as_ref().clone()),
                op: BinaryOperator::Eq,
                right: Box::new(BoundExpr::ColumnRef(crate::binder::ColumnRef {
                    table_idx: subquery_schema[0].table_idx.unwrap_or(0),
                    column_id: subquery_schema[0].column_id,
                    type_id: subquery_schema[0].type_id,
                    nullable: subquery_schema[0].nullable,
                    ts_precision: subquery_schema[0].ts_precision,
                })),
                type_id: TypeId::Boolean,
            };

            Some(LogicalPlan::Join {
                left: Arc::new(child.clone()),
                right: Arc::new(LogicalPlan::Distinct {
                    child: Arc::new(subquery_plan),
                }),
                join_type: JoinType::Inner,
                condition: JoinCondition::On(join_condition),
            })
        }
        _ => None,
    }
}

//! Subquery decorrelation optimization rule.
//!
//! Converts correlated subqueries into joins where possible.
//! Handles EXISTS -> SemiJoin and IN subquery -> SemiJoin patterns.

use crate::binder::BoundExpr;
use crate::logical::{JoinCondition, LogicalPlan};
use crate::optimizer::OptimizationRule;
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
        BoundExpr::Exists { .. } | BoundExpr::InSubquery { .. } | BoundExpr::Subquery { .. } => true,
        BoundExpr::BinaryOp { left, right, .. } => has_subquery_expr(left) || has_subquery_expr(right),
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
                child: Box::new(child),
            }
        }
        // Recursively process children
        LogicalPlan::Project { expressions, aliases, child } => LogicalPlan::Project {
            expressions: expressions.clone(),
            aliases: aliases.clone(),
            child: Box::new(decorrelate_plan(child)),
        },
        LogicalPlan::Join { left, right, join_type, condition } => LogicalPlan::Join {
            left: Box::new(decorrelate_plan(left)),
            right: Box::new(decorrelate_plan(right)),
            join_type: *join_type,
            condition: condition.clone(),
        },
        LogicalPlan::Aggregate { group_by, aggregates, child } => LogicalPlan::Aggregate {
            group_by: group_by.clone(),
            aggregates: aggregates.clone(),
            child: Box::new(decorrelate_plan(child)),
        },
        LogicalPlan::Sort { order_by, child } => LogicalPlan::Sort {
            order_by: order_by.clone(),
            child: Box::new(decorrelate_plan(child)),
        },
        LogicalPlan::Limit { limit, offset, child } => LogicalPlan::Limit {
            limit: *limit,
            offset: *offset,
            child: Box::new(decorrelate_plan(child)),
        },
        LogicalPlan::Distinct { child } => LogicalPlan::Distinct {
            child: Box::new(decorrelate_plan(child)),
        },
        other => other.clone(),
    }
}

/// Tries to convert EXISTS(subquery) into a semi-join.
fn try_decorrelate_exists(predicate: &BoundExpr, child: &LogicalPlan) -> Option<LogicalPlan> {
    match predicate {
        BoundExpr::Exists { plan: subquery, negated: false } => {
            // Convert the subquery's FROM into a join with the outer plan.
            // This is a simplified decorrelation: wrap the subquery as a Distinct + Join.
            let subquery_plan = crate::logical::builder::build_logical_plan(
                &crate::binder::BoundStatement::Select(*subquery.clone()),
            )
            .ok()?;

            Some(LogicalPlan::Join {
                left: Box::new(child.clone()),
                right: Box::new(LogicalPlan::Distinct {
                    child: Box::new(subquery_plan),
                }),
                join_type: JoinType::Inner,
                condition: JoinCondition::Cross,
            })
        }
        _ => None,
    }
}

/// Tries to convert IN(subquery) into a semi-join with equality condition.
fn try_decorrelate_in_subquery(
    predicate: &BoundExpr,
    child: &LogicalPlan,
) -> Option<LogicalPlan> {
    match predicate {
        BoundExpr::InSubquery { expr, plan: subquery, negated: false } => {
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
                })),
                type_id: TypeId::Boolean,
            };

            Some(LogicalPlan::Join {
                left: Box::new(child.clone()),
                right: Box::new(LogicalPlan::Distinct {
                    child: Box::new(subquery_plan),
                }),
                join_type: JoinType::Inner,
                condition: JoinCondition::On(join_condition),
            })
        }
        _ => None,
    }
}

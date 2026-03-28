//! Projection pushdown optimization rule.
//!
//! Narrows scan columns to only those needed by upstream operators,
//! reducing IO and memory usage.

use crate::binder::BoundExpr;
use crate::logical::{LogicalColumn, LogicalPlan};
use crate::optimizer::OptimizationRule;
use std::collections::HashSet;
use zyron_catalog::{Catalog, ColumnId};

pub struct ProjectionPushdown;

impl OptimizationRule for ProjectionPushdown {
    fn name(&self) -> &str {
        "projection_pushdown"
    }

    fn apply(&self, plan: &LogicalPlan, _catalog: &Catalog) -> Option<LogicalPlan> {
        // Quick check: no Project nodes means nothing to push.
        if !has_project(plan) {
            return None;
        }
        let pushed = push_projections(plan, None);
        if pushed != *plan { Some(pushed) } else { None }
    }
}

/// Returns true if the plan tree contains any Project node.
fn has_project(plan: &LogicalPlan) -> bool {
    match plan {
        LogicalPlan::Project { .. } => true,
        other => other.children().iter().any(|c| has_project(c)),
    }
}

/// Pushes projections down into scans by tracking which columns are needed.
fn push_projections(
    plan: &LogicalPlan,
    needed: Option<&HashSet<(usize, ColumnId)>>,
) -> LogicalPlan {
    match plan {
        LogicalPlan::Scan {
            table_id,
            table_idx,
            columns,
            alias,
            as_of,
            ..
        } => {
            if let Some(needed_cols) = needed {
                let pruned: Vec<LogicalColumn> = columns
                    .iter()
                    .filter(|c| {
                        c.table_idx
                            .map(|ti| needed_cols.contains(&(ti, c.column_id)))
                            .unwrap_or(true)
                    })
                    .cloned()
                    .collect();
                // Only prune if we actually removed columns
                if pruned.len() < columns.len() && !pruned.is_empty() {
                    return LogicalPlan::Scan {
                        table_id: *table_id,
                        table_idx: *table_idx,
                        columns: pruned,
                        alias: alias.clone(),
                        encoding_hints: None,
                        as_of: as_of.clone(),
                    };
                }
            }
            plan.clone()
        }
        LogicalPlan::Project {
            expressions,
            aliases,
            child,
        } => {
            // Collect columns needed by the projection expressions
            let mut child_needed = HashSet::new();
            for expr in expressions {
                collect_needed_columns(expr, &mut child_needed);
            }
            LogicalPlan::Project {
                expressions: expressions.clone(),
                aliases: aliases.clone(),
                child: Box::new(push_projections(child, Some(&child_needed))),
            }
        }
        LogicalPlan::Filter { predicate, child } => {
            let mut child_needed = needed.cloned().unwrap_or_default();
            collect_needed_columns(predicate, &mut child_needed);
            LogicalPlan::Filter {
                predicate: predicate.clone(),
                child: Box::new(push_projections(child, Some(&child_needed))),
            }
        }
        LogicalPlan::Join {
            left,
            right,
            join_type,
            condition,
        } => {
            let mut left_needed = HashSet::new();
            let mut right_needed = HashSet::new();

            if let Some(n) = needed {
                let left_tables = collect_table_set(left);
                let right_tables = collect_table_set(right);
                for &(ti, ci) in n {
                    if left_tables.contains(&ti) {
                        left_needed.insert((ti, ci));
                    }
                    if right_tables.contains(&ti) {
                        right_needed.insert((ti, ci));
                    }
                }
            }

            // Also add columns needed by the join condition
            if let crate::logical::JoinCondition::On(expr) = condition {
                collect_needed_columns(expr, &mut left_needed);
                collect_needed_columns(expr, &mut right_needed);
            }

            LogicalPlan::Join {
                left: Box::new(push_projections(
                    left,
                    if left_needed.is_empty() {
                        None
                    } else {
                        Some(&left_needed)
                    },
                )),
                right: Box::new(push_projections(
                    right,
                    if right_needed.is_empty() {
                        None
                    } else {
                        Some(&right_needed)
                    },
                )),
                join_type: *join_type,
                condition: condition.clone(),
            }
        }
        LogicalPlan::Aggregate {
            group_by,
            aggregates,
            child,
        } => {
            let mut child_needed = HashSet::new();
            for expr in group_by {
                collect_needed_columns(expr, &mut child_needed);
            }
            for agg in aggregates {
                for arg in &agg.args {
                    collect_needed_columns(arg, &mut child_needed);
                }
            }
            LogicalPlan::Aggregate {
                group_by: group_by.clone(),
                aggregates: aggregates.clone(),
                child: Box::new(push_projections(child, Some(&child_needed))),
            }
        }
        LogicalPlan::Sort { order_by, child } => {
            let mut child_needed = needed.cloned().unwrap_or_default();
            for ob in order_by {
                collect_needed_columns(&ob.expr, &mut child_needed);
            }
            LogicalPlan::Sort {
                order_by: order_by.clone(),
                child: Box::new(push_projections(child, Some(&child_needed))),
            }
        }
        // Pass through for other node types
        LogicalPlan::Limit {
            limit,
            offset,
            child,
        } => LogicalPlan::Limit {
            limit: *limit,
            offset: *offset,
            child: Box::new(push_projections(child, needed)),
        },
        LogicalPlan::Distinct { child } => LogicalPlan::Distinct {
            child: Box::new(push_projections(child, needed)),
        },
        LogicalPlan::SetOp {
            op,
            all,
            left,
            right,
        } => LogicalPlan::SetOp {
            op: *op,
            all: *all,
            left: Box::new(push_projections(left, needed)),
            right: Box::new(push_projections(right, needed)),
        },
        other => other.clone(),
    }
}

/// Collects (table_idx, column_id) pairs from an expression.
fn collect_needed_columns(expr: &BoundExpr, out: &mut HashSet<(usize, ColumnId)>) {
    match expr {
        BoundExpr::ColumnRef(cr) => {
            out.insert((cr.table_idx, cr.column_id));
        }
        BoundExpr::BinaryOp { left, right, .. } => {
            collect_needed_columns(left, out);
            collect_needed_columns(right, out);
        }
        BoundExpr::UnaryOp { expr, .. } => collect_needed_columns(expr, out),
        BoundExpr::IsNull { expr, .. } => collect_needed_columns(expr, out),
        BoundExpr::InList { expr, list, .. } => {
            collect_needed_columns(expr, out);
            for item in list {
                collect_needed_columns(item, out);
            }
        }
        BoundExpr::Between {
            expr, low, high, ..
        } => {
            collect_needed_columns(expr, out);
            collect_needed_columns(low, out);
            collect_needed_columns(high, out);
        }
        BoundExpr::Like { expr, pattern, .. } | BoundExpr::ILike { expr, pattern, .. } => {
            collect_needed_columns(expr, out);
            collect_needed_columns(pattern, out);
        }
        BoundExpr::Function { args, .. } | BoundExpr::AggregateFunction { args, .. } => {
            for arg in args {
                collect_needed_columns(arg, out);
            }
        }
        BoundExpr::Cast { expr, .. } => collect_needed_columns(expr, out),
        BoundExpr::Nested(inner) => collect_needed_columns(inner, out),
        BoundExpr::Case {
            operand,
            conditions,
            else_result,
            ..
        } => {
            if let Some(op) = operand {
                collect_needed_columns(op, out);
            }
            for wc in conditions {
                collect_needed_columns(&wc.condition, out);
                collect_needed_columns(&wc.result, out);
            }
            if let Some(e) = else_result {
                collect_needed_columns(e, out);
            }
        }
        BoundExpr::WindowFunction {
            function,
            partition_by,
            order_by,
            ..
        } => {
            collect_needed_columns(function, out);
            for pb in partition_by {
                collect_needed_columns(pb, out);
            }
            for ob in order_by {
                collect_needed_columns(&ob.expr, out);
            }
        }
        _ => {}
    }
}

/// Collects all table indices from scan nodes in a plan.
fn collect_table_set(plan: &LogicalPlan) -> HashSet<usize> {
    let mut set = HashSet::new();
    collect_table_set_recursive(plan, &mut set);
    set
}

fn collect_table_set_recursive(plan: &LogicalPlan, out: &mut HashSet<usize>) {
    match plan {
        LogicalPlan::Scan { table_idx, .. } => {
            out.insert(*table_idx);
        }
        other => {
            for child in other.children() {
                collect_table_set_recursive(child, out);
            }
        }
    }
}

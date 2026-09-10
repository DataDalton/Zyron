//! Projection pushdown optimization rule.
//!
//! Narrows scan columns to only those needed by upstream operators,
//! reducing IO and memory usage.

use crate::binder::{BoundExpr, BoundSelect};
use crate::logical::{LogicalColumn, LogicalPlan};
use crate::optimizer::OptimizationRule;
use std::collections::HashSet;
use std::sync::Arc;
use zyron_catalog::{Catalog, ColumnId};

pub struct ProjectionPushdown;

impl OptimizationRule for ProjectionPushdown {
    fn name(&self) -> &str {
        "projection_pushdown"
    }

    fn apply(&self, plan: &LogicalPlan, catalog: &Catalog) -> Option<LogicalPlan> {
        // Quick check: no Project nodes means nothing to push.
        if !has_project(plan) {
            return None;
        }
        let pushed = push_projections(plan, None, catalog);
        if pushed != *plan { Some(pushed) } else { None }
    }
}

/// The conjuncts of a filter the scan below it does not answer, which are
/// the only ones anything above the scan still reads columns for.
///
/// Only a lake scan answers any of them, and only the ones that lower onto
/// stored bytes with nothing left over. A time-travel read is excluded: it
/// reads a manifest whose schema predates the catalog's, so what the
/// current columns prove about the lowering is not what that file will be
/// asked. Everything else returns the whole predicate, which projects
/// exactly what it did before
fn filter_residual(
    child: &LogicalPlan,
    predicate: &BoundExpr,
    catalog: &Catalog,
) -> Vec<BoundExpr> {
    let whole = || vec![predicate.clone()];
    let LogicalPlan::Scan {
        table_id, as_of, ..
    } = child
    else {
        return whole();
    };
    if as_of.is_some() {
        return whole();
    }
    let Ok(te) = catalog.get_table_by_id(*table_id) else {
        return whole();
    };
    crate::lake_predicate::split_scan_answered(predicate, &te).1
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
    catalog: &Catalog,
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
                // Prune whenever we removed columns, including down to zero:
                // a scan that needs no data columns (bare COUNT(*)) becomes a
                // count-only scan that walks visibility without decoding any
                // tuple bytes.
                if pruned.len() < columns.len() {
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
            output_table_idx,
        } => {
            // Collect columns needed by the projection expressions
            let mut child_needed = HashSet::new();
            for expr in expressions {
                collect_needed_columns(expr, &mut child_needed);
            }
            LogicalPlan::Project {
                expressions: expressions.clone(),
                aliases: aliases.clone(),
                child: Arc::new(push_projections(child, Some(&child_needed), catalog)),
                output_table_idx: *output_table_idx,
            }
        }
        LogicalPlan::ExpandRows {
            child,
            spec,
            carry,
            output_columns,
            outer_input,
        } => {
            // The expansion carries input columns through to its output. One
            // nothing above reads is dropped here rather than gathered once
            // per produced row, which for a ten element array is ten copies
            // of a value with no reader
            let carried = carry.len();
            let (kept_carry, kept_columns): (Vec<crate::binder::ColumnRef>, Vec<LogicalColumn>) =
                match needed {
                    None => (carry.clone(), output_columns.clone()),
                    Some(needed_cols) => {
                        let mut sources = Vec::with_capacity(carried);
                        let mut columns = Vec::with_capacity(output_columns.len());
                        for (i, column) in output_columns.iter().take(carried).enumerate() {
                            let wanted = column
                                .table_idx
                                .map(|ti| needed_cols.contains(&(ti, column.column_id)))
                                .unwrap_or(true);
                            if wanted {
                                sources.push(carry[i].clone());
                                columns.push(column.clone());
                            }
                        }
                        columns.extend(output_columns.iter().skip(carried).cloned());
                        (sources, columns)
                    }
                };
            // What the expansion itself reads is needed below it, whatever
            // the query above asked for, and so is every column it still
            // carries
            let mut child_needed = HashSet::new();
            crate::binder::for_each_ref_in_expand_spec(spec, &mut |r| {
                child_needed.insert((r.table_idx, r.column_id));
            });
            for reference in &kept_carry {
                child_needed.insert((reference.table_idx, reference.column_id));
            }
            LogicalPlan::ExpandRows {
                child: Arc::new(push_projections(child, Some(&child_needed), catalog)),
                spec: spec.clone(),
                carry: kept_carry,
                output_columns: kept_columns,
                outer_input: *outer_input,
            }
        }
        LogicalPlan::Filter { predicate, child } => {
            let mut child_needed = needed.cloned().unwrap_or_default();
            // A conjunct the scan answers itself reads its columns off the
            // scan's own encoded bytes, so projecting them would decode a
            // column whose only reader is a term with nothing left to do
            for conjunct in filter_residual(child, predicate, catalog) {
                collect_needed_columns(&conjunct, &mut child_needed);
            }
            LogicalPlan::Filter {
                predicate: predicate.clone(),
                child: Arc::new(push_projections(child, Some(&child_needed), catalog)),
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
                left: Arc::new(push_projections(
                    left,
                    if left_needed.is_empty() {
                        None
                    } else {
                        Some(&left_needed)
                    },
                    catalog,
                )),
                right: Arc::new(push_projections(
                    right,
                    if right_needed.is_empty() {
                        None
                    } else {
                        Some(&right_needed)
                    },
                    catalog,
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
                child: Arc::new(push_projections(child, Some(&child_needed), catalog)),
            }
        }
        LogicalPlan::Sort { order_by, child } => {
            let mut child_needed = needed.cloned().unwrap_or_default();
            for ob in order_by {
                collect_needed_columns(&ob.expr, &mut child_needed);
            }
            LogicalPlan::Sort {
                order_by: order_by.clone(),
                child: Arc::new(push_projections(child, Some(&child_needed), catalog)),
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
            child: Arc::new(push_projections(child, needed, catalog)),
        },
        LogicalPlan::Distinct { child } => LogicalPlan::Distinct {
            child: Arc::new(push_projections(child, needed, catalog)),
        },
        LogicalPlan::SetOp {
            op,
            all,
            left,
            right,
        } => LogicalPlan::SetOp {
            op: *op,
            all: *all,
            left: Arc::new(push_projections(left, needed, catalog)),
            right: Arc::new(push_projections(right, needed, catalog)),
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
        // A subquery's correlated references point at outer columns the scan
        // must still produce; the subquery's own columns are owned by its plan.
        BoundExpr::Subquery { plan, .. } | BoundExpr::Exists { plan, .. } => {
            collect_correlated_outer_columns(plan, out);
        }
        BoundExpr::InSubquery { expr, plan, .. } => {
            collect_needed_columns(expr, out);
            collect_correlated_outer_columns(plan, out);
        }
        _ => {}
    }
}

/// Adds the outer (correlated) column references a subquery plan uses to `out`.
/// These are columns the enclosing scan must still produce so the correlated
/// subquery reads the current outer row's values. Columns the subquery produces
/// itself are owned by its own plan and excluded.
fn collect_correlated_outer_columns(plan: &BoundSelect, out: &mut HashSet<(usize, ColumnId)>) {
    let mut owned = HashSet::new();
    crate::binder::subquery_owned_indices(plan, &mut owned);
    crate::binder::for_each_subquery_ref(plan, &mut |cr| {
        if !owned.contains(&cr.table_idx) {
            out.insert((cr.table_idx, cr.column_id));
        }
    });
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

//! Which JSON paths a plan reads out of its variant columns.
//!
//! Dotted access binds to `variant_extract(column, 'path')`, and a columnar
//! segment can hold a promoted path as a column of its own. A scan that knows
//! the paths the statement asks for reads exactly those columns and leaves
//! the rest of the promoted set on disk, so a query naming no path pays
//! nothing for a table with a dozen of them.
//!
//! A path this misses costs the walk over the documents that would have
//! happened anyway, never a different answer

use super::PhysicalPlan;
use crate::binder::BoundExpr;
use zyron_parser::ast::LiteralValue;

/// One path a statement reads out of one variant column
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VariantPath {
    /// The table instance the column belongs to, as a `ColumnRef` spells it
    pub table_idx: usize,
    /// The variant column the path is read out of
    pub column_id: u16,
    /// Dotted path, as `variant_extract` spells it
    pub path: String,
}

/// Every variant path the plan reads, deduplicated
pub fn variant_paths(plan: &PhysicalPlan) -> Vec<VariantPath> {
    let mut out = Vec::new();
    collect_plan(plan, &mut out);
    out
}

fn push(out: &mut Vec<VariantPath>, found: VariantPath) {
    if !out.contains(&found) {
        out.push(found);
    }
}

/// Walks one expression tree, recording every `variant_extract` reading a
/// literal path out of a column
fn collect_expr(expr: &BoundExpr, out: &mut Vec<VariantPath>) {
    match expr {
        BoundExpr::Function { name, args, .. } => {
            if name == "variant_extract"
                && args.len() == 2
                && let BoundExpr::ColumnRef(cr) = &args[0]
                && let BoundExpr::Literal {
                    value: LiteralValue::String(path),
                    ..
                } = &args[1]
            {
                push(
                    out,
                    VariantPath {
                        table_idx: cr.table_idx,
                        column_id: cr.column_id.0,
                        path: path.clone(),
                    },
                );
            }
            for a in args {
                collect_expr(a, out);
            }
        }
        BoundExpr::AggregateFunction { args, .. } => {
            for a in args {
                collect_expr(a, out);
            }
        }
        BoundExpr::BinaryOp { left, right, .. } => {
            collect_expr(left, out);
            collect_expr(right, out);
        }
        BoundExpr::UnaryOp { expr, .. }
        | BoundExpr::IsNull { expr, .. }
        | BoundExpr::Cast { expr, .. }
        | BoundExpr::Nested(expr)
        | BoundExpr::TemporalRef { inner: expr, .. } => collect_expr(expr, out),
        BoundExpr::Between {
            expr, low, high, ..
        } => {
            collect_expr(expr, out);
            collect_expr(low, out);
            collect_expr(high, out);
        }
        BoundExpr::InList { expr, list, .. } => {
            collect_expr(expr, out);
            for item in list {
                collect_expr(item, out);
            }
        }
        BoundExpr::Like { expr, pattern, .. } | BoundExpr::ILike { expr, pattern, .. } => {
            collect_expr(expr, out);
            collect_expr(pattern, out);
        }
        BoundExpr::Case {
            operand,
            conditions,
            else_result,
            ..
        } => {
            if let Some(o) = operand {
                collect_expr(o, out);
            }
            for w in conditions {
                collect_expr(&w.condition, out);
                collect_expr(&w.result, out);
            }
            if let Some(e) = else_result {
                collect_expr(e, out);
            }
        }
        BoundExpr::WindowFunction {
            function,
            partition_by,
            order_by,
            ..
        } => {
            collect_expr(function, out);
            for e in partition_by {
                collect_expr(e, out);
            }
            for o in order_by {
                collect_expr(&o.expr, out);
            }
        }
        BoundExpr::Subquery { .. } | BoundExpr::Exists { .. } | BoundExpr::InSubquery { .. } => {}
        BoundExpr::ColumnRef(_) | BoundExpr::Literal { .. } | BoundExpr::Parameter { .. } => {}
    }
}

/// Walks one node's own expressions, then its children.
///
/// A variant carrying no expression contributes none, so the last arm is a
/// catch-all. A node shape nobody thought of costs the document walk rather
/// than a wrong answer
fn collect_plan(plan: &PhysicalPlan, out: &mut Vec<VariantPath>) {
    let one = |e: &Option<BoundExpr>, out: &mut Vec<VariantPath>| {
        if let Some(e) = e {
            collect_expr(e, out);
        }
    };
    match plan {
        PhysicalPlan::SeqScan { predicate, .. }
        | PhysicalPlan::HybridScan { predicate, .. }
        | PhysicalPlan::LakeScan { predicate, .. }
        | PhysicalPlan::ParallelSeqScan { predicate, .. } => one(predicate, out),
        PhysicalPlan::ForeignScan { residual, .. } => one(residual, out),
        PhysicalPlan::LakeDelete {
            bound_predicate, ..
        } => one(bound_predicate, out),
        PhysicalPlan::LakeUpdate {
            assignments,
            check_constraints,
            ..
        } => {
            for a in assignments {
                collect_expr(&a.value, out);
            }
            for c in check_constraints {
                collect_expr(c, out);
            }
        }
        PhysicalPlan::IndexScan {
            predicate,
            remaining_predicate,
            ordered_by,
            ..
        } => {
            collect_expr(predicate, out);
            one(remaining_predicate, out);
            if let Some(order) = ordered_by {
                for o in order {
                    collect_expr(&o.expr, out);
                }
            }
        }
        PhysicalPlan::Filter { predicate, .. } => collect_expr(predicate, out),
        PhysicalPlan::Project { expressions, .. }
        | PhysicalPlan::Window {
            window_exprs: expressions,
            ..
        } => {
            for e in expressions {
                collect_expr(e, out);
            }
        }
        PhysicalPlan::NestedLoopJoin { condition, .. }
        | PhysicalPlan::LateralJoin { condition, .. } => one(condition, out),
        PhysicalPlan::HashJoin {
            left_keys,
            right_keys,
            remaining_condition,
            ..
        }
        | PhysicalPlan::ParallelHashJoin {
            left_keys,
            right_keys,
            remaining_condition,
            ..
        } => {
            for e in left_keys.iter().chain(right_keys) {
                collect_expr(e, out);
            }
            one(remaining_condition, out);
        }
        PhysicalPlan::MergeJoin {
            left_keys,
            right_keys,
            ..
        } => {
            for e in left_keys.iter().chain(right_keys) {
                collect_expr(e, out);
            }
        }
        PhysicalPlan::HashAggregate {
            group_by,
            aggregates,
            ..
        }
        | PhysicalPlan::SortAggregate {
            group_by,
            aggregates,
            ..
        } => {
            for e in group_by {
                collect_expr(e, out);
            }
            for a in aggregates {
                for e in &a.args {
                    collect_expr(e, out);
                }
            }
        }
        PhysicalPlan::Sort { order_by, .. } => {
            for o in order_by {
                collect_expr(&o.expr, out);
            }
        }
        PhysicalPlan::Insert {
            column_defaults,
            check_constraints,
            ..
        } => {
            for (_, e) in column_defaults {
                collect_expr(e, out);
            }
            for c in check_constraints {
                collect_expr(c, out);
            }
        }
        PhysicalPlan::Values { rows, .. } => {
            for row in rows {
                for e in row {
                    collect_expr(e, out);
                }
            }
        }
        PhysicalPlan::Update {
            assignments,
            check_constraints,
            ..
        } => {
            for a in assignments {
                collect_expr(&a.value, out);
            }
            for c in check_constraints {
                collect_expr(c, out);
            }
        }
        PhysicalPlan::Repartition { partition_keys, .. } => {
            for e in partition_keys {
                collect_expr(e, out);
            }
        }
        PhysicalPlan::FulltextScan {
            match_expr,
            remaining_predicate,
            ..
        } => {
            collect_expr(match_expr, out);
            one(remaining_predicate, out);
        }
        PhysicalPlan::VectorScan {
            remaining_predicate,
            ..
        }
        | PhysicalPlan::SpatialScan {
            remaining_predicate,
            ..
        } => one(remaining_predicate, out),
        PhysicalPlan::GraphAlgorithm { params, .. } => {
            for (_, e) in params {
                collect_expr(e, out);
            }
        }
        PhysicalPlan::AnalyticsTableFunction {
            named_args,
            positional_args,
            ..
        } => {
            for (_, e) in named_args {
                collect_expr(e, out);
            }
            for e in positional_args {
                collect_expr(e, out);
            }
        }
        _ => {}
    }
    plan.for_each_child(&mut |child| collect_plan(child, out));
}

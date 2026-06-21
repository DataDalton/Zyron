//! Uncorrelated subquery materialization.
//!
//! The expression evaluator is synchronous and has no execution context, so it
//! cannot run a subquery plan. This pass runs before operator construction:
//! it walks a bound expression, executes each uncorrelated scalar, IN, or
//! EXISTS subquery against the current transaction snapshot, and rewrites the
//! node to a constant the evaluator understands (a literal, an IN list, or a
//! boolean). A correlated subquery references an outer column its own plan does
//! not produce, so executing it raises an explicit "column not found" error
//! rather than a wrong result; correlated decorrelation is a separate concern.

use std::sync::Arc;

use zyron_common::{Result, TypeId, ZyronError};
use zyron_parser::ast::LiteralValue;
use zyron_planner::binder::{BoundExpr, BoundOrderBy, BoundSelect, BoundStatement, BoundWhen};
use zyron_planner::optimizer::Optimizer;

use crate::column::ScalarValue;
use crate::context::ExecutionContext;

/// Returns true when a bound expression tree contains any subquery node, so the
/// rewrite pass can skip expressions that need no work.
pub fn contains_subquery(expr: &BoundExpr) -> bool {
    match expr {
        BoundExpr::Subquery { .. } | BoundExpr::Exists { .. } | BoundExpr::InSubquery { .. } => {
            true
        }
        BoundExpr::BinaryOp { left, right, .. } => {
            contains_subquery(left) || contains_subquery(right)
        }
        BoundExpr::UnaryOp { expr, .. } => contains_subquery(expr),
        BoundExpr::Nested(inner) => contains_subquery(inner),
        BoundExpr::IsNull { expr, .. } => contains_subquery(expr),
        BoundExpr::Between {
            expr, low, high, ..
        } => contains_subquery(expr) || contains_subquery(low) || contains_subquery(high),
        BoundExpr::InList { expr, list, .. } => {
            contains_subquery(expr) || list.iter().any(contains_subquery)
        }
        BoundExpr::Like { expr, pattern, .. } | BoundExpr::ILike { expr, pattern, .. } => {
            contains_subquery(expr) || contains_subquery(pattern)
        }
        BoundExpr::Cast { expr, .. } => contains_subquery(expr),
        BoundExpr::Case {
            operand,
            conditions,
            else_result,
            ..
        } => {
            operand.as_deref().is_some_and(contains_subquery)
                || conditions
                    .iter()
                    .any(|w| contains_subquery(&w.condition) || contains_subquery(&w.result))
                || else_result.as_deref().is_some_and(contains_subquery)
        }
        BoundExpr::Function { args, .. } => args.iter().any(contains_subquery),
        BoundExpr::AggregateFunction { args, .. } => args.iter().any(contains_subquery),
        BoundExpr::WindowFunction {
            function,
            partition_by,
            order_by,
            ..
        } => {
            contains_subquery(function)
                || partition_by.iter().any(contains_subquery)
                || order_by.iter().any(|o| contains_subquery(&o.expr))
        }
        _ => false,
    }
}

/// Materializes uncorrelated subqueries in a single expression, returning it
/// unchanged when it holds none. Used by operator builders that evaluate a
/// scalar expression (sort key, aggregate argument) the synchronous evaluator
/// cannot run a subquery for.
pub async fn materialize_one(expr: BoundExpr, ctx: &Arc<ExecutionContext>) -> Result<BoundExpr> {
    if contains_subquery(&expr) {
        materialize_expr(expr, ctx).await
    } else {
        Ok(expr)
    }
}

/// Materializes uncorrelated subqueries in an optional predicate, leaving None
/// and subquery-free predicates untouched.
pub async fn materialize_opt(
    expr: Option<BoundExpr>,
    ctx: &Arc<ExecutionContext>,
) -> Result<Option<BoundExpr>> {
    match expr {
        Some(e) => Ok(Some(materialize_one(e, ctx).await?)),
        None => Ok(None),
    }
}

/// Materializes uncorrelated subqueries across a list of expressions in order.
pub async fn materialize_vec(
    exprs: Vec<BoundExpr>,
    ctx: &Arc<ExecutionContext>,
) -> Result<Vec<BoundExpr>> {
    let mut out = Vec::with_capacity(exprs.len());
    for e in exprs {
        out.push(materialize_one(e, ctx).await?);
    }
    Ok(out)
}

/// Rewrites a bound expression, materializing every uncorrelated subquery it
/// contains. Recurses children first so nested subqueries are handled bottom up.
pub fn materialize_expr<'a>(
    expr: BoundExpr,
    ctx: &'a Arc<ExecutionContext>,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<BoundExpr>> + Send + 'a>> {
    Box::pin(async move {
        match expr {
            BoundExpr::Subquery { plan, type_id } => materialize_scalar(*plan, type_id, ctx).await,
            BoundExpr::Exists { plan, negated } => materialize_exists(*plan, negated, ctx).await,
            BoundExpr::InSubquery {
                expr,
                plan,
                negated,
            } => materialize_in(*expr, *plan, negated, ctx).await,
            BoundExpr::BinaryOp {
                left,
                op,
                right,
                type_id,
            } => Ok(BoundExpr::BinaryOp {
                left: Box::new(materialize_expr(*left, ctx).await?),
                op,
                right: Box::new(materialize_expr(*right, ctx).await?),
                type_id,
            }),
            BoundExpr::UnaryOp { op, expr, type_id } => Ok(BoundExpr::UnaryOp {
                op,
                expr: Box::new(materialize_expr(*expr, ctx).await?),
                type_id,
            }),
            BoundExpr::Nested(inner) => Ok(BoundExpr::Nested(Box::new(
                materialize_expr(*inner, ctx).await?,
            ))),
            BoundExpr::IsNull { expr, negated } => Ok(BoundExpr::IsNull {
                expr: Box::new(materialize_expr(*expr, ctx).await?),
                negated,
            }),
            BoundExpr::Between {
                expr,
                low,
                high,
                negated,
            } => Ok(BoundExpr::Between {
                expr: Box::new(materialize_expr(*expr, ctx).await?),
                low: Box::new(materialize_expr(*low, ctx).await?),
                high: Box::new(materialize_expr(*high, ctx).await?),
                negated,
            }),
            BoundExpr::InList {
                expr,
                list,
                negated,
            } => {
                let mut new_list = Vec::with_capacity(list.len());
                for item in list {
                    new_list.push(materialize_expr(item, ctx).await?);
                }
                Ok(BoundExpr::InList {
                    expr: Box::new(materialize_expr(*expr, ctx).await?),
                    list: new_list,
                    negated,
                })
            }
            BoundExpr::Like {
                expr,
                pattern,
                negated,
            } => Ok(BoundExpr::Like {
                expr: Box::new(materialize_expr(*expr, ctx).await?),
                pattern: Box::new(materialize_expr(*pattern, ctx).await?),
                negated,
            }),
            BoundExpr::ILike {
                expr,
                pattern,
                negated,
            } => Ok(BoundExpr::ILike {
                expr: Box::new(materialize_expr(*expr, ctx).await?),
                pattern: Box::new(materialize_expr(*pattern, ctx).await?),
                negated,
            }),
            BoundExpr::Cast { expr, target_type } => Ok(BoundExpr::Cast {
                expr: Box::new(materialize_expr(*expr, ctx).await?),
                target_type,
            }),
            BoundExpr::Case {
                operand,
                conditions,
                else_result,
                type_id,
            } => {
                let operand = match operand {
                    Some(o) => Some(Box::new(materialize_expr(*o, ctx).await?)),
                    None => None,
                };
                let mut new_conditions = Vec::with_capacity(conditions.len());
                for w in conditions {
                    new_conditions.push(BoundWhen {
                        condition: materialize_expr(w.condition, ctx).await?,
                        result: materialize_expr(w.result, ctx).await?,
                    });
                }
                let else_result = match else_result {
                    Some(e) => Some(Box::new(materialize_expr(*e, ctx).await?)),
                    None => None,
                };
                Ok(BoundExpr::Case {
                    operand,
                    conditions: new_conditions,
                    else_result,
                    type_id,
                })
            }
            BoundExpr::Function {
                name,
                args,
                return_type,
                distinct,
            } => {
                let mut new_args = Vec::with_capacity(args.len());
                for a in args {
                    new_args.push(materialize_expr(a, ctx).await?);
                }
                Ok(BoundExpr::Function {
                    name,
                    args: new_args,
                    return_type,
                    distinct,
                })
            }
            BoundExpr::AggregateFunction {
                name,
                args,
                distinct,
                return_type,
                uda,
            } => {
                let mut new_args = Vec::with_capacity(args.len());
                for a in args {
                    new_args.push(materialize_expr(a, ctx).await?);
                }
                Ok(BoundExpr::AggregateFunction {
                    name,
                    args: new_args,
                    distinct,
                    return_type,
                    uda,
                })
            }
            BoundExpr::WindowFunction {
                function,
                partition_by,
                order_by,
                frame,
                type_id,
            } => {
                let function = Box::new(materialize_expr(*function, ctx).await?);
                let mut new_partition = Vec::with_capacity(partition_by.len());
                for p in partition_by {
                    new_partition.push(materialize_expr(p, ctx).await?);
                }
                let mut new_order = Vec::with_capacity(order_by.len());
                for o in order_by {
                    new_order.push(BoundOrderBy {
                        expr: materialize_expr(o.expr, ctx).await?,
                        asc: o.asc,
                        nulls_first: o.nulls_first,
                    });
                }
                Ok(BoundExpr::WindowFunction {
                    function,
                    partition_by: new_partition,
                    order_by: new_order,
                    frame,
                    type_id,
                })
            }
            other => Ok(other),
        }
    })
}

/// Builds, optimizes, and executes a subquery's plan against the current
/// transaction, returning the flattened scalar values of its first output
/// column in row order.
async fn run_first_column(
    plan: BoundSelect,
    ctx: &Arc<ExecutionContext>,
) -> Result<Vec<ScalarValue>> {
    let logical =
        zyron_planner::logical::builder::build_logical_plan(&BoundStatement::Select(plan))?;
    let optimizer = Optimizer::new(&ctx.catalog);
    let optimized = optimizer.optimize(logical)?;
    let physical = zyron_planner::physical::builder::build_physical_plan(optimized, &ctx.catalog)?;
    let batches = crate::executor::execute(physical, ctx).await?;

    let mut values = Vec::new();
    for batch in &batches {
        let Some(col) = batch.columns.first() else {
            continue;
        };
        for row in 0..batch.num_rows {
            values.push(col.get_scalar(row));
        }
    }
    Ok(values)
}

/// Counts the rows a subquery produces, short-circuiting on the first batch
/// that has any. Used for EXISTS.
async fn subquery_has_rows(plan: BoundSelect, ctx: &Arc<ExecutionContext>) -> Result<bool> {
    let logical =
        zyron_planner::logical::builder::build_logical_plan(&BoundStatement::Select(plan))?;
    let optimizer = Optimizer::new(&ctx.catalog);
    let optimized = optimizer.optimize(logical)?;
    let physical = zyron_planner::physical::builder::build_physical_plan(optimized, &ctx.catalog)?;
    let batches = crate::executor::execute(physical, ctx).await?;
    Ok(batches.iter().any(|b| b.num_rows > 0))
}

async fn materialize_scalar(
    plan: BoundSelect,
    type_id: TypeId,
    ctx: &Arc<ExecutionContext>,
) -> Result<BoundExpr> {
    let values = run_first_column(plan, ctx).await?;
    match values.len() {
        0 => Ok(BoundExpr::Literal {
            value: LiteralValue::Null,
            type_id,
        }),
        1 => Ok(BoundExpr::Literal {
            value: scalar_to_literal(&values[0])?,
            type_id,
        }),
        n => Err(ZyronError::ExecutionError(format!(
            "scalar subquery returned {n} rows, expected at most one"
        ))),
    }
}

async fn materialize_exists(
    plan: BoundSelect,
    negated: bool,
    ctx: &Arc<ExecutionContext>,
) -> Result<BoundExpr> {
    let has = subquery_has_rows(plan, ctx).await?;
    Ok(BoundExpr::Literal {
        value: LiteralValue::Boolean(has != negated),
        type_id: TypeId::Boolean,
    })
}

async fn materialize_in(
    expr: BoundExpr,
    plan: BoundSelect,
    negated: bool,
    ctx: &Arc<ExecutionContext>,
) -> Result<BoundExpr> {
    let probe = materialize_expr(expr, ctx).await?;
    // Type the list literals as the probe so the IN comparison matches the
    // probe's column type instead of a widened literal type.
    let probe_type = probe.type_id();
    let values = run_first_column(plan, ctx).await?;
    let mut list = Vec::with_capacity(values.len());
    for v in &values {
        let lit = scalar_to_literal(v)?;
        let type_id = if matches!(lit, LiteralValue::Null) {
            TypeId::Null
        } else {
            probe_type
        };
        list.push(BoundExpr::Literal {
            value: lit,
            type_id,
        });
    }
    Ok(BoundExpr::InList {
        expr: Box::new(probe),
        list,
        negated,
    })
}

/// Maps a scalar value to a parser literal so it can be spliced into the bound
/// expression tree. Types with no literal representation raise an explicit
/// error rather than substituting a wrong value.
fn scalar_to_literal(s: &ScalarValue) -> Result<LiteralValue> {
    Ok(match s {
        ScalarValue::Null => LiteralValue::Null,
        ScalarValue::Boolean(b) => LiteralValue::Boolean(*b),
        ScalarValue::Int8(v) => LiteralValue::Integer(*v as i64),
        ScalarValue::Int16(v) => LiteralValue::Integer(*v as i64),
        ScalarValue::Int32(v) => LiteralValue::Integer(*v as i64),
        ScalarValue::Int64(v) => LiteralValue::Integer(*v),
        ScalarValue::UInt8(v) => LiteralValue::Integer(*v as i64),
        ScalarValue::UInt16(v) => LiteralValue::Integer(*v as i64),
        ScalarValue::UInt32(v) => LiteralValue::Integer(*v as i64),
        ScalarValue::Float32(v) => LiteralValue::Float(*v as f64),
        ScalarValue::Float64(v) => LiteralValue::Float(*v),
        ScalarValue::Utf8(v) => LiteralValue::String(v.clone()),
        ScalarValue::Interval(i) => LiteralValue::Interval(*i),
        ScalarValue::Int128(_)
        | ScalarValue::UInt64(_)
        | ScalarValue::Binary(_)
        | ScalarValue::FixedBinary16(_) => {
            return Err(ZyronError::ExecutionError(
                "subquery result type is not representable as a literal for substitution".into(),
            ));
        }
    })
}

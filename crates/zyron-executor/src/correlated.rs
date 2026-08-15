//! Correlated subquery execution.
//!
//! A correlated subquery references columns from an enclosing query, so its
//! result depends on the current outer row and cannot be folded to a constant
//! once like an uncorrelated subquery. This module runs the subquery once per
//! outer row: each outer column the subquery references is turned into a query
//! parameter, the subquery's physical plan is built a single time, and for each
//! outer row the referenced values are bound as parameters and the plan is
//! executed against a child context that shares the enclosing transaction and
//! snapshot.
//!
//! The CorrelatedFilterOperator and CorrelatedProjectOperator drive this for
//! WHERE predicates and projection lists, the two places a correlated subquery
//! commonly appears. Scalar, EXISTS, and IN subqueries are supported. An
//! uncorrelated subquery left inside the rewritten expression is folded by the
//! existing materialize pass before per-row evaluation begins.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use zyron_common::{Result, TypeId, ZyronError};
use zyron_parser::ast::JoinType;
use zyron_planner::binder::{
    BoundExpr, BoundFromItem, BoundJoinCondition, BoundSelect, BoundSelectItem, BoundStatement,
    ColumnRef, for_each_subquery_ref, subquery_is_correlated, subquery_owned_indices,
};
use zyron_planner::logical::LogicalColumn;
use zyron_planner::optimizer::Optimizer;
use zyron_planner::physical::PhysicalPlan;

use crate::batch::{ColumnBuilder, DataBatch};
use crate::column::{Column, ScalarValue};
use crate::compute::column_to_mask;
use crate::context::ExecutionContext;
use crate::expr::evaluate;
use crate::operator::{ExecutionBatch, Operator, OperatorResult};

// ---------------------------------------------------------------------------
// Correlation detection
// ---------------------------------------------------------------------------

/// Returns true when an expression contains a subquery that correlates to the
/// enclosing query. The executor routes such expressions through the per-row
/// correlated operators instead of the one-shot materialize pass.
pub fn expr_has_correlated_subquery(expr: &BoundExpr) -> bool {
    let mut found = false;
    for_each_top_subquery(expr, &mut |plan| {
        if subquery_is_correlated(plan) {
            found = true;
        }
    });
    found
}

// ---------------------------------------------------------------------------
// Expression and plan walks
// ---------------------------------------------------------------------------

/// Calls `f` on each top-level subquery plan within `expr` without descending
/// into the subquery plans themselves (subquery_is_correlated inspects internals).
fn for_each_top_subquery(expr: &BoundExpr, f: &mut dyn FnMut(&BoundSelect)) {
    match expr {
        BoundExpr::Subquery { plan, .. } | BoundExpr::Exists { plan, .. } => f(plan),
        BoundExpr::InSubquery { expr, plan, .. } => {
            f(plan);
            for_each_top_subquery(expr, f);
        }
        BoundExpr::BinaryOp { left, right, .. } => {
            for_each_top_subquery(left, f);
            for_each_top_subquery(right, f);
        }
        BoundExpr::UnaryOp { expr, .. }
        | BoundExpr::IsNull { expr, .. }
        | BoundExpr::Cast { expr, .. }
        | BoundExpr::Nested(expr)
        | BoundExpr::TemporalRef { inner: expr, .. } => for_each_top_subquery(expr, f),
        BoundExpr::Between {
            expr, low, high, ..
        } => {
            for_each_top_subquery(expr, f);
            for_each_top_subquery(low, f);
            for_each_top_subquery(high, f);
        }
        BoundExpr::InList { expr, list, .. } => {
            for_each_top_subquery(expr, f);
            for item in list {
                for_each_top_subquery(item, f);
            }
        }
        BoundExpr::Like { expr, pattern, .. } | BoundExpr::ILike { expr, pattern, .. } => {
            for_each_top_subquery(expr, f);
            for_each_top_subquery(pattern, f);
        }
        BoundExpr::Function { args, .. } | BoundExpr::AggregateFunction { args, .. } => {
            for arg in args {
                for_each_top_subquery(arg, f);
            }
        }
        BoundExpr::Case {
            operand,
            conditions,
            else_result,
            ..
        } => {
            if let Some(o) = operand {
                for_each_top_subquery(o, f);
            }
            for w in conditions {
                for_each_top_subquery(&w.condition, f);
                for_each_top_subquery(&w.result, f);
            }
            if let Some(e) = else_result {
                for_each_top_subquery(e, f);
            }
        }
        BoundExpr::WindowFunction {
            function,
            partition_by,
            order_by,
            ..
        } => {
            for_each_top_subquery(function, f);
            for e in partition_by {
                for_each_top_subquery(e, f);
            }
            for o in order_by {
                for_each_top_subquery(&o.expr, f);
            }
        }
        BoundExpr::ColumnRef(_) | BoundExpr::Literal { .. } | BoundExpr::Parameter { .. } => {}
    }
}

/// Replaces every column reference for which `f` returns Some with the returned
/// expression, descending into nested subquery plans. Used to turn outer
/// references into query parameters in a correlated subquery's plan.
fn map_refs_in_expr(expr: &mut BoundExpr, f: &dyn Fn(&ColumnRef) -> Option<BoundExpr>) {
    match expr {
        BoundExpr::ColumnRef(cr) => {
            if let Some(replacement) = f(cr) {
                *expr = replacement;
            }
        }
        BoundExpr::Literal { .. } | BoundExpr::Parameter { .. } => {}
        BoundExpr::BinaryOp { left, right, .. } => {
            map_refs_in_expr(left, f);
            map_refs_in_expr(right, f);
        }
        BoundExpr::UnaryOp { expr, .. }
        | BoundExpr::IsNull { expr, .. }
        | BoundExpr::Cast { expr, .. }
        | BoundExpr::Nested(expr)
        | BoundExpr::TemporalRef { inner: expr, .. } => map_refs_in_expr(expr, f),
        BoundExpr::Between {
            expr, low, high, ..
        } => {
            map_refs_in_expr(expr, f);
            map_refs_in_expr(low, f);
            map_refs_in_expr(high, f);
        }
        BoundExpr::InList { expr, list, .. } => {
            map_refs_in_expr(expr, f);
            for item in list {
                map_refs_in_expr(item, f);
            }
        }
        BoundExpr::Like { expr, pattern, .. } | BoundExpr::ILike { expr, pattern, .. } => {
            map_refs_in_expr(expr, f);
            map_refs_in_expr(pattern, f);
        }
        BoundExpr::Function { args, .. } | BoundExpr::AggregateFunction { args, .. } => {
            for arg in args {
                map_refs_in_expr(arg, f);
            }
        }
        BoundExpr::Case {
            operand,
            conditions,
            else_result,
            ..
        } => {
            if let Some(o) = operand {
                map_refs_in_expr(o, f);
            }
            for w in conditions {
                map_refs_in_expr(&mut w.condition, f);
                map_refs_in_expr(&mut w.result, f);
            }
            if let Some(e) = else_result {
                map_refs_in_expr(e, f);
            }
        }
        BoundExpr::WindowFunction {
            function,
            partition_by,
            order_by,
            ..
        } => {
            map_refs_in_expr(function, f);
            for e in partition_by {
                map_refs_in_expr(e, f);
            }
            for o in order_by {
                map_refs_in_expr(&mut o.expr, f);
            }
        }
        BoundExpr::Subquery { plan, .. } | BoundExpr::Exists { plan, .. } => {
            map_refs_in_select(plan, f)
        }
        BoundExpr::InSubquery { expr, plan, .. } => {
            map_refs_in_expr(expr, f);
            map_refs_in_select(plan, f);
        }
    }
}

fn map_refs_in_select(plan: &mut BoundSelect, f: &dyn Fn(&ColumnRef) -> Option<BoundExpr>) {
    for item in &mut plan.projections {
        if let BoundSelectItem::Expr(e, _) = item {
            map_refs_in_expr(e, f);
        }
    }
    if let Some(w) = &mut plan.where_clause {
        map_refs_in_expr(w, f);
    }
    for e in &mut plan.group_by {
        map_refs_in_expr(e, f);
    }
    if let Some(h) = &mut plan.having {
        map_refs_in_expr(h, f);
    }
    for o in &mut plan.order_by {
        map_refs_in_expr(&mut o.expr, f);
    }
    if let Some(l) = &mut plan.limit {
        map_refs_in_expr(l, f);
    }
    if let Some(o) = &mut plan.offset {
        map_refs_in_expr(o, f);
    }
    for item in &mut plan.from {
        map_refs_in_from(item, f);
    }
    for sop in &mut plan.set_ops {
        map_refs_in_select(&mut sop.right, f);
    }
    for cte in &mut plan.ctes {
        map_refs_in_select(&mut cte.query, f);
    }
}

fn map_refs_in_from(item: &mut BoundFromItem, f: &dyn Fn(&ColumnRef) -> Option<BoundExpr>) {
    match item {
        BoundFromItem::BaseTable { .. } => {}
        BoundFromItem::Subquery { query, .. } => map_refs_in_select(query, f),
        BoundFromItem::Join {
            left,
            right,
            condition,
            ..
        } => {
            map_refs_in_from(left, f);
            map_refs_in_from(right, f);
            if let BoundJoinCondition::On(e) = condition {
                map_refs_in_expr(e, f);
            }
        }
        BoundFromItem::GraphQuery { params, .. } => {
            for (_, e) in params {
                map_refs_in_expr(e, f);
            }
        }
        BoundFromItem::AnalyticsFunction {
            params, positional, ..
        } => {
            for (_, e) in params {
                map_refs_in_expr(e, f);
            }
            for e in positional {
                map_refs_in_expr(e, f);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Prepared correlated subqueries
// ---------------------------------------------------------------------------

/// The kind of correlated subquery and the data needed to reduce its per-row
/// result to a single scalar that fills the rewritten expression's slot.
enum SubKind {
    /// A scalar subquery. Reduces to its single output value, NULL for no rows,
    /// or an error for more than one row.
    Scalar,
    /// An EXISTS subquery. Reduces to a boolean: whether any row was produced,
    /// negated for NOT EXISTS.
    Exists { negated: bool },
    /// An IN subquery. The probe is evaluated against the outer row, then tested
    /// for membership in the subquery's first output column with SQL three
    /// valued semantics. `negated` flips the result for NOT IN.
    In { probe: BoundExpr, negated: bool },
}

/// A correlated subquery extracted from an expression. The expression node it
/// replaced is now a parameter at `slot`; per outer row the operator evaluates
/// `outer_refs` against the row, binds them as the subquery's parameters,
/// executes `template`, and reduces the result into that slot.
struct CorrelatedSub {
    kind: SubKind,
    /// Physical plan with outer references replaced by parameters, built once.
    template: PhysicalPlan,
    /// Outer column references in plan order; the j-th maps to the subquery
    /// parameter at index base_len + j + 1.
    outer_refs: Vec<ColumnRef>,
}

/// State threaded through expression rewriting: the enclosing query's table
/// indices, the count of statement parameters, the next slot to assign, and the
/// subqueries extracted so far.
struct Prep<'a> {
    ctx: &'a Arc<ExecutionContext>,
    input_set: &'a HashSet<usize>,
    base_len: usize,
    subs: Vec<CorrelatedSub>,
}

/// Rewrites an expression, replacing each correlated subquery with a parameter
/// reference and recording how to evaluate it per row. Uncorrelated subqueries
/// are left in place for the materialize pass to fold.
fn rewrite_expr(expr: BoundExpr, prep: &mut Prep) -> Result<BoundExpr> {
    match expr {
        BoundExpr::Subquery { plan, type_id } => {
            if subquery_is_correlated(&plan) {
                extract(SubKind::Scalar, *plan, type_id, prep)
            } else {
                Ok(BoundExpr::Subquery { plan, type_id })
            }
        }
        BoundExpr::Exists { plan, negated } => {
            if subquery_is_correlated(&plan) {
                extract(SubKind::Exists { negated }, *plan, TypeId::Boolean, prep)
            } else {
                Ok(BoundExpr::Exists { plan, negated })
            }
        }
        BoundExpr::InSubquery {
            expr,
            plan,
            negated,
        } => {
            if subquery_is_correlated(&plan) {
                let probe = rewrite_expr(*expr, prep)?;
                extract(SubKind::In { probe, negated }, *plan, TypeId::Boolean, prep)
            } else {
                let expr = Box::new(rewrite_expr(*expr, prep)?);
                Ok(BoundExpr::InSubquery {
                    expr,
                    plan,
                    negated,
                })
            }
        }
        BoundExpr::BinaryOp {
            left,
            op,
            right,
            type_id,
        } => Ok(BoundExpr::BinaryOp {
            left: Box::new(rewrite_expr(*left, prep)?),
            op,
            right: Box::new(rewrite_expr(*right, prep)?),
            type_id,
        }),
        BoundExpr::UnaryOp { op, expr, type_id } => Ok(BoundExpr::UnaryOp {
            op,
            expr: Box::new(rewrite_expr(*expr, prep)?),
            type_id,
        }),
        BoundExpr::IsNull { expr, negated } => Ok(BoundExpr::IsNull {
            expr: Box::new(rewrite_expr(*expr, prep)?),
            negated,
        }),
        BoundExpr::Nested(inner) => Ok(BoundExpr::Nested(Box::new(rewrite_expr(*inner, prep)?))),
        BoundExpr::Between {
            expr,
            low,
            high,
            negated,
        } => Ok(BoundExpr::Between {
            expr: Box::new(rewrite_expr(*expr, prep)?),
            low: Box::new(rewrite_expr(*low, prep)?),
            high: Box::new(rewrite_expr(*high, prep)?),
            negated,
        }),
        BoundExpr::InList {
            expr,
            list,
            negated,
        } => {
            let mut new_list = Vec::with_capacity(list.len());
            for item in list {
                new_list.push(rewrite_expr(item, prep)?);
            }
            Ok(BoundExpr::InList {
                expr: Box::new(rewrite_expr(*expr, prep)?),
                list: new_list,
                negated,
            })
        }
        BoundExpr::Like {
            expr,
            pattern,
            negated,
        } => Ok(BoundExpr::Like {
            expr: Box::new(rewrite_expr(*expr, prep)?),
            pattern: Box::new(rewrite_expr(*pattern, prep)?),
            negated,
        }),
        BoundExpr::ILike {
            expr,
            pattern,
            negated,
        } => Ok(BoundExpr::ILike {
            expr: Box::new(rewrite_expr(*expr, prep)?),
            pattern: Box::new(rewrite_expr(*pattern, prep)?),
            negated,
        }),
        BoundExpr::Cast {
            expr,
            target_type,
            fractional_digits,
        } => Ok(BoundExpr::Cast {
            fractional_digits,
            expr: Box::new(rewrite_expr(*expr, prep)?),
            target_type,
        }),
        BoundExpr::Case {
            operand,
            conditions,
            else_result,
            type_id,
        } => {
            let operand = match operand {
                Some(o) => Some(Box::new(rewrite_expr(*o, prep)?)),
                None => None,
            };
            let mut new_conditions = Vec::with_capacity(conditions.len());
            for w in conditions {
                new_conditions.push(zyron_planner::binder::BoundWhen {
                    condition: rewrite_expr(w.condition, prep)?,
                    result: rewrite_expr(w.result, prep)?,
                });
            }
            let else_result = match else_result {
                Some(e) => Some(Box::new(rewrite_expr(*e, prep)?)),
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
                new_args.push(rewrite_expr(a, prep)?);
            }
            Ok(BoundExpr::Function {
                name,
                args: new_args,
                return_type,
                distinct,
            })
        }
        other => Ok(other),
    }
}

/// Builds a parameterized physical plan for a correlated subquery and records
/// it, returning the parameter reference that replaces the subquery node.
fn extract(
    kind: SubKind,
    plan: BoundSelect,
    slot_type: TypeId,
    prep: &mut Prep,
) -> Result<BoundExpr> {
    let (template, outer_refs) =
        parameterize_subquery(plan, prep.input_set, prep.base_len, prep.ctx)?;

    let slot = prep.base_len + prep.subs.len() + 1;
    prep.subs.push(CorrelatedSub {
        kind,
        template,
        outer_refs,
    });
    Ok(BoundExpr::Parameter {
        index: slot,
        type_id: slot_type,
    })
}

/// Turns a subquery's references to outer columns into query parameters and
/// builds its physical plan once. `outer_set` is the set of table indices the
/// enclosing operator can supply (a correlated subquery's WHERE columns or a
/// LATERAL inner plan's references to the left input). Returns the parameterized
/// physical template and the outer column references in parameter order: the
/// j-th maps to parameter index base_len + j + 1, which the operator fills per
/// row before executing the template.
pub fn parameterize_subquery(
    plan: BoundSelect,
    outer_set: &HashSet<usize>,
    base_len: usize,
    ctx: &Arc<ExecutionContext>,
) -> Result<(PhysicalPlan, Vec<ColumnRef>)> {
    let mut owned = HashSet::new();
    subquery_owned_indices(&plan, &mut owned);

    let mut order: Vec<ColumnRef> = Vec::new();
    let mut seen: HashSet<(usize, u16)> = HashSet::new();
    for_each_subquery_ref(&plan, &mut |cr| {
        if !owned.contains(&cr.table_idx) && outer_set.contains(&cr.table_idx) {
            let key = (cr.table_idx, cr.column_id.0);
            if seen.insert(key) {
                order.push(cr.clone());
            }
        }
    });

    let mut pos: HashMap<(usize, u16), usize> = HashMap::with_capacity(order.len());
    for (i, cr) in order.iter().enumerate() {
        pos.insert((cr.table_idx, cr.column_id.0), i);
    }

    let owned_for_map = owned.clone();
    let outer_for_map = outer_set.clone();
    let mut parameterized = plan;
    map_refs_in_select(&mut parameterized, &|cr| {
        if !owned_for_map.contains(&cr.table_idx) && outer_for_map.contains(&cr.table_idx) {
            let idx = pos[&(cr.table_idx, cr.column_id.0)];
            Some(BoundExpr::Parameter {
                index: base_len + idx + 1,
                type_id: cr.type_id,
            })
        } else {
            None
        }
    });

    let logical = zyron_planner::logical::builder::build_logical_plan(&BoundStatement::Select(
        parameterized,
    ))?;
    let optimized = Optimizer::new(&ctx.catalog).optimize(logical)?;
    // Costing a foreign scan in a re-planned subquery needs the same
    // peer facts the outer plan used. The guard spans the build and
    // nothing else, because a lock held across an await would pin the
    // registry for the length of the query
    let template = {
        let peerGuard = ctx.peers.as_ref().map(|p| p.read());
        zyron_planner::physical::builder::build_physical_plan(
            optimized,
            &ctx.catalog,
            peerGuard.as_deref().map(|p| &**p),
        )?
    };
    Ok((template, order))
}

// ---------------------------------------------------------------------------
// Per-row evaluation
// ---------------------------------------------------------------------------

/// Evaluates a list of expressions over a batch, running each correlated
/// subquery once per row. Returns one output column per expression.
async fn eval_rows(
    ctx: &Arc<ExecutionContext>,
    exprs: &[BoundExpr],
    subs: &[CorrelatedSub],
    batch: &DataBatch,
    input_schema: &[LogicalColumn],
    base_params: &[ScalarValue],
) -> Result<Vec<Column>> {
    let n = batch.num_rows;

    // Precompute, over the whole batch, the outer-reference columns each
    // subquery needs and the probe column for IN subqueries.
    let mut sub_outer_cols: Vec<Vec<Column>> = Vec::with_capacity(subs.len());
    let mut sub_probe_cols: Vec<Option<Column>> = Vec::with_capacity(subs.len());
    for s in subs {
        let mut cols = Vec::with_capacity(s.outer_refs.len());
        for cr in &s.outer_refs {
            cols.push(evaluate(
                &BoundExpr::ColumnRef(cr.clone()),
                batch,
                input_schema,
                base_params,
            )?);
        }
        sub_outer_cols.push(cols);
        let probe = match &s.kind {
            SubKind::In { probe, .. } => Some(evaluate(probe, batch, input_schema, base_params)?),
            _ => None,
        };
        sub_probe_cols.push(probe);
    }

    let mut builders: Vec<ColumnBuilder> = exprs
        .iter()
        .map(|e| ColumnBuilder::new(e.type_id(), n))
        .collect();

    for row in 0..n {
        // Run each correlated subquery for this row and place its scalar result
        // in the slot region of the parameter set.
        let mut full_params = base_params.to_vec();
        for (i, s) in subs.iter().enumerate() {
            let mut child_params = base_params.to_vec();
            for col in &sub_outer_cols[i] {
                child_params.push(col.get_scalar(row));
            }
            let child = Arc::new(ctx.child_with_params(child_params));
            let probe_val = sub_probe_cols[i].as_ref().map(|c| c.get_scalar(row));
            let value = run_sub(s, probe_val, &child).await?;
            full_params.push(value);
        }

        let row_batch = batch.slice(row, 1);
        for (e, b) in exprs.iter().zip(builders.iter_mut()) {
            let col = evaluate(e, &row_batch, input_schema, &full_params)?;
            b.push(&col.get_scalar(0));
        }
    }

    Ok(builders.into_iter().map(|b| b.finish()).collect())
}

/// Executes a correlated subquery's prepared plan against the child context and
/// reduces the result to the scalar that fills its slot.
async fn run_sub(
    sub: &CorrelatedSub,
    probe_val: Option<ScalarValue>,
    child: &Arc<ExecutionContext>,
) -> Result<ScalarValue> {
    let batches = crate::executor::execute(sub.template.clone(), child).await?;
    match &sub.kind {
        SubKind::Exists { negated } => {
            let has = batches.iter().any(|b| b.num_rows > 0);
            Ok(ScalarValue::Boolean(has != *negated))
        }
        SubKind::Scalar => {
            let mut values = first_column_scalars(&batches);
            match values.len() {
                0 => Ok(ScalarValue::Null),
                1 => Ok(values.pop().unwrap()),
                k => Err(ZyronError::ExecutionError(format!(
                    "scalar subquery returned {k} rows, expected at most one"
                ))),
            }
        }
        SubKind::In { negated, .. } => {
            let probe = probe_val.unwrap_or(ScalarValue::Null);
            let values = first_column_scalars(&batches);
            Ok(in_membership(&probe, &values, *negated))
        }
    }
}

/// Flattens the first output column of every batch into a scalar vector.
fn first_column_scalars(batches: &[DataBatch]) -> Vec<ScalarValue> {
    let mut out = Vec::new();
    for b in batches {
        if let Some(col) = b.columns.first() {
            for row in 0..b.num_rows {
                out.push(col.get_scalar(row));
            }
        }
    }
    out
}

/// Applies SQL three-valued IN semantics. An empty list is false for IN and
/// true for NOT IN. A NULL probe against a non-empty list is unknown (NULL). A
/// match yields true; no match with a NULL present is unknown; otherwise false.
/// `negated` flips a definite result and leaves unknown as NULL.
fn in_membership(probe: &ScalarValue, values: &[ScalarValue], negated: bool) -> ScalarValue {
    if values.is_empty() {
        return ScalarValue::Boolean(negated);
    }
    if matches!(probe, ScalarValue::Null) {
        return ScalarValue::Null;
    }
    let mut saw_null = false;
    for v in values {
        match scalar_eq(probe, v) {
            Some(true) => return ScalarValue::Boolean(!negated),
            Some(false) => {}
            None => saw_null = true,
        }
    }
    if saw_null {
        ScalarValue::Null
    } else {
        ScalarValue::Boolean(negated)
    }
}

/// Compares two scalars for equality, widening across the integer and floating
/// families so an Int32 probe matches an Int64 subquery value. Returns None when
/// either side is NULL. Falls back to per-variant comparison for non-numeric
/// types; mismatched non-numeric variants compare unequal.
fn scalar_eq(a: &ScalarValue, b: &ScalarValue) -> Option<bool> {
    use ScalarValue::*;
    if matches!(a, Null) || matches!(b, Null) {
        return None;
    }
    if let (Some(x), Some(y)) = (as_i128(a), as_i128(b)) {
        return Some(x == y);
    }
    if let (Some(x), Some(y)) = (as_f64(a), as_f64(b)) {
        return Some(x == y);
    }
    Some(match (a, b) {
        (Boolean(x), Boolean(y)) => x == y,
        (Utf8(x), Utf8(y)) => x == y,
        (Binary(x), Binary(y)) => x == y,
        (FixedBinary16(x), FixedBinary16(y)) => x == y,
        (Interval(x), Interval(y)) => x == y,
        _ => false,
    })
}

/// Returns the integer value of a scalar in the signed integer or unsigned
/// integer family, widened to i128. None for non-integer scalars.
fn as_i128(s: &ScalarValue) -> Option<i128> {
    use ScalarValue::*;
    match s {
        Int8(v) => Some(*v as i128),
        Int16(v) => Some(*v as i128),
        Int32(v) => Some(*v as i128),
        Int64(v) => Some(*v as i128),
        Int128(v) => Some(*v),
        UInt8(v) => Some(*v as i128),
        UInt16(v) => Some(*v as i128),
        UInt32(v) => Some(*v as i128),
        UInt64(v) => Some(*v as i128),
        _ => None,
    }
}

/// Returns the floating value of any numeric scalar (integer or float). None for
/// non-numeric scalars.
fn as_f64(s: &ScalarValue) -> Option<f64> {
    use ScalarValue::*;
    match s {
        Float32(v) => Some(*v as f64),
        Float64(v) => Some(*v),
        other => as_i128(other).map(|i| i as f64),
    }
}

// ---------------------------------------------------------------------------
// Operators
// ---------------------------------------------------------------------------

/// Filters rows using a predicate that contains a correlated subquery. The
/// subquery runs once per input row.
pub struct CorrelatedFilterOperator {
    child: Box<dyn Operator>,
    predicate: BoundExpr,
    subs: Vec<CorrelatedSub>,
    input_schema: Vec<LogicalColumn>,
    base_params: Vec<ScalarValue>,
    ctx: Arc<ExecutionContext>,
}

impl Operator for CorrelatedFilterOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            loop {
                let Some(exec_batch) = self.child.next().await? else {
                    return Ok(None);
                };
                let cols = eval_rows(
                    &self.ctx,
                    std::slice::from_ref(&self.predicate),
                    &self.subs,
                    &exec_batch.batch,
                    &self.input_schema,
                    &self.base_params,
                )
                .await?;
                let mask = column_to_mask(&cols[0]);
                let filtered = exec_batch.batch.filter(&mask);
                if filtered.num_rows == 0 {
                    continue;
                }
                let filtered_locs = exec_batch.locators.map(|locs| {
                    mask.iter()
                        .enumerate()
                        .filter_map(|(i, &keep)| if keep { Some(locs[i]) } else { None })
                        .collect::<Vec<_>>()
                });
                return Ok(Some(ExecutionBatch {
                    batch: filtered,
                    locators: filtered_locs,
                }));
            }
        })
    }
}

/// Projects expressions where one or more contain a correlated subquery. The
/// subqueries run once per input row.
pub struct CorrelatedProjectOperator {
    child: Box<dyn Operator>,
    expressions: Vec<BoundExpr>,
    subs: Vec<CorrelatedSub>,
    input_schema: Vec<LogicalColumn>,
    base_params: Vec<ScalarValue>,
    ctx: Arc<ExecutionContext>,
}

impl Operator for CorrelatedProjectOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            let Some(exec_batch) = self.child.next().await? else {
                return Ok(None);
            };
            let cols = eval_rows(
                &self.ctx,
                &self.expressions,
                &self.subs,
                &exec_batch.batch,
                &self.input_schema,
                &self.base_params,
            )
            .await?;
            Ok(Some(ExecutionBatch::new(DataBatch::new(cols))))
        })
    }
}

// ---------------------------------------------------------------------------
// Build entry points
// ---------------------------------------------------------------------------

/// Builds a filter operator for a predicate containing a correlated subquery.
/// Extracts the correlated subqueries, folds any remaining uncorrelated ones,
/// and returns an operator that evaluates the predicate per row.
pub async fn build_correlated_filter(
    child: Box<dyn Operator>,
    predicate: BoundExpr,
    input_schema: Vec<LogicalColumn>,
    base_params: Vec<ScalarValue>,
    ctx: &Arc<ExecutionContext>,
) -> Result<CorrelatedFilterOperator> {
    let input_set: HashSet<usize> = input_schema.iter().filter_map(|c| c.table_idx).collect();
    let mut prep = Prep {
        ctx,
        input_set: &input_set,
        base_len: base_params.len(),
        subs: Vec::new(),
    };
    let rewritten = rewrite_expr(predicate, &mut prep)?;
    let predicate = fold_uncorrelated(rewritten, ctx).await?;
    Ok(CorrelatedFilterOperator {
        child,
        predicate,
        subs: prep.subs,
        input_schema,
        base_params,
        ctx: Arc::clone(ctx),
    })
}

/// Builds a project operator for a projection list where at least one
/// expression contains a correlated subquery.
pub async fn build_correlated_project(
    child: Box<dyn Operator>,
    expressions: Vec<BoundExpr>,
    input_schema: Vec<LogicalColumn>,
    base_params: Vec<ScalarValue>,
    ctx: &Arc<ExecutionContext>,
) -> Result<CorrelatedProjectOperator> {
    let input_set: HashSet<usize> = input_schema.iter().filter_map(|c| c.table_idx).collect();
    let mut prep = Prep {
        ctx,
        input_set: &input_set,
        base_len: base_params.len(),
        subs: Vec::new(),
    };
    let mut rewritten = Vec::with_capacity(expressions.len());
    for e in expressions {
        let r = rewrite_expr(e, &mut prep)?;
        rewritten.push(fold_uncorrelated(r, ctx).await?);
    }
    Ok(CorrelatedProjectOperator {
        child,
        expressions: rewritten,
        subs: prep.subs,
        input_schema,
        base_params,
        ctx: Arc::clone(ctx),
    })
}

/// Value expressions containing correlated subqueries, prepared once and
/// evaluated per batch. UPDATE uses this for SET values whose subqueries
/// reference the updated table, the shape a desugared MERGE produces.
pub struct CorrelatedValues {
    expressions: Vec<BoundExpr>,
    subs: Vec<CorrelatedSub>,
    input_schema: Vec<LogicalColumn>,
    base_params: Vec<ScalarValue>,
}

/// Prepares value expressions for repeated per batch evaluation. Correlated
/// subqueries become parameterized templates built once, uncorrelated ones
/// fold to constants.
pub async fn prepare_correlated_values(
    expressions: Vec<BoundExpr>,
    input_schema: Vec<LogicalColumn>,
    base_params: Vec<ScalarValue>,
    ctx: &Arc<ExecutionContext>,
) -> Result<CorrelatedValues> {
    let input_set: HashSet<usize> = input_schema.iter().filter_map(|c| c.table_idx).collect();
    let mut prep = Prep {
        ctx,
        input_set: &input_set,
        base_len: base_params.len(),
        subs: Vec::new(),
    };
    let mut rewritten = Vec::with_capacity(expressions.len());
    for e in expressions {
        let r = rewrite_expr(e, &mut prep)?;
        rewritten.push(fold_uncorrelated(r, ctx).await?);
    }
    Ok(CorrelatedValues {
        expressions: rewritten,
        subs: prep.subs,
        input_schema,
        base_params,
    })
}

impl CorrelatedValues {
    /// One result column per prepared expression, aligned with the batch rows
    pub async fn eval(
        &self,
        ctx: &Arc<ExecutionContext>,
        batch: &DataBatch,
    ) -> Result<Vec<Column>> {
        eval_rows(
            ctx,
            &self.expressions,
            &self.subs,
            batch,
            &self.input_schema,
            &self.base_params,
        )
        .await
    }
}

/// Folds any uncorrelated subquery left in a rewritten expression to a constant.
async fn fold_uncorrelated(expr: BoundExpr, ctx: &Arc<ExecutionContext>) -> Result<BoundExpr> {
    if crate::subquery::contains_subquery(&expr) {
        crate::subquery::materialize_expr(expr, ctx).await
    } else {
        Ok(expr)
    }
}

/// A boolean predicate containing one or more subqueries, prepared once for
/// repeated per-batch evaluation. Correlated subqueries are parameterized into
/// physical templates built a single time; uncorrelated ones are folded to
/// constants. Used by the join operator to evaluate a subquery-bearing ON
/// condition over each joined row.
pub struct CorrelatedPredicate {
    predicate: BoundExpr,
    subs: Vec<CorrelatedSub>,
    input_schema: Vec<LogicalColumn>,
    base_params: Vec<ScalarValue>,
}

impl CorrelatedPredicate {
    /// Prepares a predicate evaluated against `input_schema`. Correlated outer
    /// references are the columns in `input_schema`.
    pub async fn prepare(
        predicate: BoundExpr,
        input_schema: Vec<LogicalColumn>,
        base_params: Vec<ScalarValue>,
        ctx: &Arc<ExecutionContext>,
    ) -> Result<Self> {
        let input_set: HashSet<usize> = input_schema.iter().filter_map(|c| c.table_idx).collect();
        let mut prep = Prep {
            ctx,
            input_set: &input_set,
            base_len: base_params.len(),
            subs: Vec::new(),
        };
        let rewritten = rewrite_expr(predicate, &mut prep)?;
        let predicate = fold_uncorrelated(rewritten, ctx).await?;
        Ok(Self {
            predicate,
            subs: prep.subs,
            input_schema,
            base_params,
        })
    }

    /// Evaluates the predicate over `batch`, returning a keep flag per row.
    pub async fn eval_mask(
        &self,
        ctx: &Arc<ExecutionContext>,
        batch: &DataBatch,
    ) -> Result<Vec<bool>> {
        let cols = eval_rows(
            ctx,
            std::slice::from_ref(&self.predicate),
            &self.subs,
            batch,
            &self.input_schema,
            &self.base_params,
        )
        .await?;
        Ok(column_to_mask(&cols[0]))
    }
}

// ---------------------------------------------------------------------------
// LATERAL join
// ---------------------------------------------------------------------------

/// Executes a LATERAL join: the right side is a subquery that references the
/// left input, run once per left row with those columns bound as parameters.
/// For each left row it emits one output row per right row (the left values
/// repeated), filtered by the optional ON condition. A LEFT or FULL join emits
/// a NULL-extended row when the subquery yields no surviving rows for a left
/// row; an inner or cross join drops that left row.
pub struct LateralJoinOperator {
    left: Box<dyn Operator>,
    /// Parameterized subquery plan, built once; executed per left row.
    template: PhysicalPlan,
    /// Left columns the subquery references, in parameter order.
    outer_refs: Vec<ColumnRef>,
    /// Number of left output columns.
    left_len: usize,
    /// Combined left++right schema, used to type output builders and to resolve
    /// the ON condition over a joined row.
    joined_schema: Vec<LogicalColumn>,
    condition: Option<BoundExpr>,
    null_extend: bool,
    base_params: Vec<ScalarValue>,
    ctx: Arc<ExecutionContext>,
}

impl Operator for LateralJoinOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            loop {
                let Some(left_eb) = self.left.next().await? else {
                    return Ok(None);
                };
                let lb = &left_eb.batch;
                let n = lb.num_rows;
                if n == 0 {
                    continue;
                }
                let right_len = self.joined_schema.len() - self.left_len;

                // Evaluate the left columns the subquery needs once over the
                // whole left batch, then index per row.
                let mut outer_cols: Vec<Column> = Vec::with_capacity(self.outer_refs.len());
                for cr in &self.outer_refs {
                    outer_cols.push(evaluate(
                        &BoundExpr::ColumnRef(cr.clone()),
                        lb,
                        &self.joined_schema[..self.left_len],
                        &self.base_params,
                    )?);
                }

                let mut builders: Vec<ColumnBuilder> = self
                    .joined_schema
                    .iter()
                    .map(|c| ColumnBuilder::new(c.type_id, n))
                    .collect();

                for row in 0..n {
                    let mut child_params = self.base_params.clone();
                    for col in &outer_cols {
                        child_params.push(col.get_scalar(row));
                    }
                    let child = Arc::new(self.ctx.child_with_params(child_params));
                    let right_batches =
                        crate::executor::execute(self.template.clone(), &child).await?;

                    let mut matched = 0usize;
                    for rb in &right_batches {
                        for rr in 0..rb.num_rows {
                            if let Some(cond) = &self.condition {
                                let mut cols = lb.slice(row, 1).columns;
                                cols.extend(rb.slice(rr, 1).columns);
                                let joined = DataBatch::new(cols);
                                let mask_col = evaluate(
                                    cond,
                                    &joined,
                                    &self.joined_schema,
                                    &self.base_params,
                                )?;
                                if !column_to_mask(&mask_col).first().copied().unwrap_or(false) {
                                    continue;
                                }
                            }
                            for c in 0..self.left_len {
                                builders[c].push(&lb.columns[c].get_scalar(row));
                            }
                            for c in 0..right_len {
                                builders[self.left_len + c].push(&rb.columns[c].get_scalar(rr));
                            }
                            matched += 1;
                        }
                    }

                    if matched == 0 && self.null_extend {
                        for c in 0..self.left_len {
                            builders[c].push(&lb.columns[c].get_scalar(row));
                        }
                        for c in 0..right_len {
                            builders[self.left_len + c].push(&ScalarValue::Null);
                        }
                    }
                }

                let cols: Vec<Column> = builders.into_iter().map(|b| b.finish()).collect();
                let batch = DataBatch::new(cols);
                if batch.num_rows == 0 {
                    // An inner or cross lateral join produced no matches for any
                    // row in this left batch; pull the next left batch.
                    continue;
                }
                return Ok(Some(ExecutionBatch::new(batch)));
            }
        })
    }
}

/// Builds a LATERAL join operator. The subquery's references to left columns are
/// parameterized once into a physical template; per left row the operator binds
/// those values and executes it.
#[allow(clippy::too_many_arguments)]
pub fn build_lateral_join(
    left: Box<dyn Operator>,
    subquery: BoundSelect,
    join_type: JoinType,
    condition: Option<BoundExpr>,
    left_schema: Vec<LogicalColumn>,
    right_schema: Vec<LogicalColumn>,
    base_params: Vec<ScalarValue>,
    ctx: &Arc<ExecutionContext>,
) -> Result<LateralJoinOperator> {
    let outer_set: HashSet<usize> = left_schema.iter().filter_map(|c| c.table_idx).collect();
    let (template, outer_refs) =
        parameterize_subquery(subquery, &outer_set, base_params.len(), ctx)?;
    let left_len = left_schema.len();
    let mut joined_schema = left_schema;
    joined_schema.extend(right_schema);
    let null_extend = matches!(join_type, JoinType::Left | JoinType::Full);
    Ok(LateralJoinOperator {
        left,
        template,
        outer_refs,
        left_len,
        joined_schema,
        condition,
        null_extend,
        base_params,
        ctx: Arc::clone(ctx),
    })
}

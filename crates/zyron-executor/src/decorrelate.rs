//! A correlated EXISTS answered as a hash semi join.
//!
//! `WHERE EXISTS (SELECT 1 FROM inner WHERE inner.k = outer.k)` describes a
//! semi join, and running it as written costs one execution of the inner
//! plan per outer row. Lifting the correlation out turns it into what it
//! describes: the inner query runs once with its correlating equalities
//! removed and its projection replaced by their inner sides, and every
//! outer row probes the keys that run produced.
//!
//! The lift is refused rather than approximated. A subquery whose row
//! count depends on anything but its own WHERE, or that still mentions an
//! outer column once the equalities are taken out, keeps the per row path
//! in [`crate::correlated`] that already answers it correctly.
//!
//! Equality is the whole basis for the rewrite. A hash set answers `=` and
//! nothing else, and `=` is also what makes NULL behave: a NULL key equals
//! nothing, so it joins nothing, which is the same answer the per row form
//! reaches by evaluating a comparison that is never true.

use std::collections::HashSet;
use std::sync::Arc;

use zyron_common::{Result, TypeId};
use zyron_parser::ast::BinaryOperator;
use zyron_planner::binder::{
    BoundColumnDef, BoundExpr, BoundSelect, BoundSelectItem, BoundStatement, ColumnRef,
    for_each_ref_in_bound_expr, for_each_subquery_ref, subquery_owned_indices,
};
use zyron_planner::logical::LogicalColumn;
use zyron_planner::optimizer::Optimizer;
use zyron_planner::physical::PhysicalPlan;

use crate::batch::DataBatch;
use crate::column::{Column, ScalarValue};
use crate::compute::column_to_mask;
use crate::context::ExecutionContext;
use crate::expr::evaluate;
use crate::operator::{ExecutionBatch, Operator, OperatorResult};

/// Splits a predicate into its top level AND terms.
///
/// Only these are separable. A term under an OR cannot be lifted on its
/// own, because the rows the other side of the OR admits do not depend on
/// it
fn conjuncts(expr: &BoundExpr, out: &mut Vec<BoundExpr>) {
    match expr {
        BoundExpr::BinaryOp {
            left,
            op: BinaryOperator::And,
            right,
            ..
        } => {
            conjuncts(left, out);
            conjuncts(right, out);
        }
        BoundExpr::Nested(inner) => conjuncts(inner, out),
        other => out.push(other.clone()),
    }
}

/// Rebuilds a predicate from terms, None when there are none left
fn all_of(mut terms: Vec<BoundExpr>) -> Option<BoundExpr> {
    let mut combined = terms.pop()?;
    while let Some(next) = terms.pop() {
        combined = BoundExpr::BinaryOp {
            left: Box::new(next),
            op: BinaryOperator::And,
            right: Box::new(combined),
            type_id: TypeId::Boolean,
        };
    }
    Some(combined)
}

/// Which side of the correlation an expression sits on, or None when it
/// straddles both and belongs to neither
enum Side {
    Inner,
    Outer,
}

fn side_of(expr: &BoundExpr, owned: &HashSet<usize>) -> Option<Side> {
    let mut refs: Vec<ColumnRef> = Vec::new();
    for_each_ref_in_bound_expr(expr, &mut |cr| refs.push(cr.clone()));
    let inner = refs.iter().any(|cr| owned.contains(&cr.table_idx));
    let outer = refs.iter().any(|cr| !owned.contains(&cr.table_idx));
    match (inner, outer) {
        (true, false) => Some(Side::Inner),
        // A term of only literals is a constant on either side, and is
        // left in the subquery where it already filters
        (false, true) => Some(Side::Outer),
        _ => None,
    }
}

/// Whether the subquery's row count depends only on its own WHERE.
///
/// An aggregate produces a row whether or not anything matched, a LIMIT
/// caps what a lifted run would see, and a set operation or a CTE brings
/// in a shape this rewrite does not inspect. The projection is whitelisted
/// rather than searched: it is replaced wholesale below, and a literal or
/// a column reference is the only thing an EXISTS subquery's projection
/// needs to be for that replacement to preserve its meaning
fn shape_is_liftable(select: &BoundSelect) -> bool {
    select.group_by.is_empty()
        && select.having.is_none()
        && select.limit.is_none()
        && select.offset.is_none()
        && select.set_ops.is_empty()
        && select.ctes.is_empty()
        && select.order_by.is_empty()
        && select.row_lock.is_none()
        && !select.distinct
        && select.projections.iter().all(|p| match p {
            BoundSelectItem::AllColumns(_) | BoundSelectItem::Wildcard => true,
            BoundSelectItem::Expr(e, _) => matches!(
                e,
                BoundExpr::Literal { .. } | BoundExpr::ColumnRef(_) | BoundExpr::Parameter { .. }
            ),
        })
}

/// One EXISTS subquery with its correlation lifted into a join key
struct Lifted {
    /// The subquery with its correlating equalities removed and its
    /// projection replaced by their inner sides, so one run of it yields
    /// exactly the keys to probe
    select: Box<BoundSelect>,
    /// The outer expression each projected key compares against, in the
    /// same order
    outer_keys: Vec<BoundExpr>,
}

/// A subquery's WHERE split into the equalities that correlate it and
/// everything else
struct Correlation {
    inner_keys: Vec<BoundExpr>,
    outer_keys: Vec<BoundExpr>,
    residual: Vec<BoundExpr>,
    owned: HashSet<usize>,
}

/// Separates the correlating equalities from the rest of a subquery's
/// WHERE, or None when there are none to separate.
///
/// Only a top level `inner = outer` counts. One side has to name the
/// subquery's own tables and the other only the enclosing query's, so a
/// term straddling both, or one comparing two outer columns, stays where
/// it is
fn split_correlation(select: &BoundSelect) -> Option<Correlation> {
    let mut owned: HashSet<usize> = HashSet::new();
    subquery_owned_indices(select, &mut owned);

    let mut terms = Vec::new();
    conjuncts(select.where_clause.as_ref()?, &mut terms);

    let mut inner_keys: Vec<BoundExpr> = Vec::new();
    let mut outer_keys: Vec<BoundExpr> = Vec::new();
    let mut residual: Vec<BoundExpr> = Vec::new();
    for term in terms {
        let pair = match &term {
            BoundExpr::BinaryOp {
                left,
                op: BinaryOperator::Eq,
                right,
                ..
            } => match (side_of(left, &owned), side_of(right, &owned)) {
                (Some(Side::Inner), Some(Side::Outer)) => {
                    Some((left.as_ref().clone(), right.as_ref().clone()))
                }
                (Some(Side::Outer), Some(Side::Inner)) => {
                    Some((right.as_ref().clone(), left.as_ref().clone()))
                }
                _ => None,
            },
            _ => None,
        };
        match pair {
            Some((inner, outer)) => {
                inner_keys.push(inner);
                outer_keys.push(outer);
            }
            None => residual.push(term),
        }
    }
    if inner_keys.is_empty() {
        return None;
    }
    Some(Correlation {
        inner_keys,
        outer_keys,
        residual,
        owned,
    })
}

/// One column definition per projected expression, named by position
fn schema_of(exprs: &[BoundExpr]) -> Vec<BoundColumnDef> {
    exprs
        .iter()
        .enumerate()
        .map(|(i, e)| BoundColumnDef {
            column_id: zyron_catalog::ColumnId(i as u16),
            name: format!("k{i}"),
            type_id: e.type_id(),
            nullable: true,
            ordinal: i as u16,
            fractional_digits: None,
        })
        .collect()
}

/// Whether any outer reference survived the lift.
///
/// Every one of them had to be accounted for by an equality. One left in
/// the residual WHERE, a join condition or a nested subquery means a
/// single run cannot stand in for the per row runs it replaces
fn outer_ref_escapes(
    select: &BoundSelect,
    owned: &HashSet<usize>,
    outer_tables: &HashSet<usize>,
) -> bool {
    let mut escapes = false;
    for_each_subquery_ref(select, &mut |cr| {
        if !owned.contains(&cr.table_idx) && outer_tables.contains(&cr.table_idx) {
            escapes = true;
        }
    });
    escapes
}

/// Lifts the correlation out of one EXISTS subquery, or None when it
/// cannot be lifted exactly
fn lift(select: &BoundSelect, outer_tables: &HashSet<usize>) -> Option<Lifted> {
    if !shape_is_liftable(select) {
        return None;
    }
    let correlation = split_correlation(select)?;

    let mut lifted = select.clone();
    lifted.where_clause = all_of(correlation.residual);
    lifted.output_schema = schema_of(&correlation.inner_keys);
    lifted.projections = correlation
        .inner_keys
        .into_iter()
        .map(|e| BoundSelectItem::Expr(e, None))
        .collect();

    if outer_ref_escapes(&lifted, &correlation.owned, outer_tables) {
        return None;
    }
    Some(Lifted {
        select: Box::new(lifted),
        outer_keys: correlation.outer_keys,
    })
}

/// A correlated scalar aggregate rewritten as one grouped run.
///
/// `(SELECT MAX(x) FROM inner WHERE inner.k = outer.k)` asks for one group
/// of `SELECT k, MAX(x) FROM inner GROUP BY k`, and asks for a different
/// group per outer row. Running the grouped form once answers every outer
/// row at the cost of one
pub struct LiftedAggregate {
    pub select: Box<BoundSelect>,
    /// The outer expression each key column compares against
    pub outer_keys: Vec<BoundExpr>,
    /// What a key the grouped run did not produce means for this
    /// aggregate. Over no rows an aggregate is NULL, except a count, which
    /// is zero, and that difference is the whole reason this is carried
    /// rather than assumed
    pub empty: ScalarValue,
}

/// The value an aggregate takes over no rows at all, or None for one whose
/// empty answer this cannot state
fn empty_value(name: &str, uda: bool) -> Option<ScalarValue> {
    if uda {
        return None;
    }
    match name.to_lowercase().as_str() {
        "count" => Some(ScalarValue::Int64(0)),
        "min" | "max" | "sum" | "avg" => Some(ScalarValue::Null),
        _ => None,
    }
}

/// Rewrites a correlated scalar aggregate into a grouped run, or None when
/// it cannot be
pub fn lift_scalar_aggregate(
    select: &BoundSelect,
    outer_tables: &HashSet<usize>,
) -> Option<LiftedAggregate> {
    if !select.group_by.is_empty()
        || select.having.is_some()
        || select.limit.is_some()
        || select.offset.is_some()
        || !select.set_ops.is_empty()
        || !select.ctes.is_empty()
        || !select.order_by.is_empty()
        || select.row_lock.is_some()
        || select.distinct
        || select.projections.len() != 1
    {
        return None;
    }
    let (aggregate, empty) = match select.projections.first()? {
        BoundSelectItem::Expr(e @ BoundExpr::AggregateFunction { name, uda, .. }, _) => {
            (e.clone(), empty_value(name, uda.is_some())?)
        }
        _ => return None,
    };
    // A decimal folds through its text form on the per row path so the
    // parameter channel can carry its scale, and a value taken straight
    // out of a map would skip that
    if aggregate.type_id() == TypeId::Decimal {
        return None;
    }
    let correlation = split_correlation(select)?;

    let mut lifted = select.clone();
    lifted.where_clause = all_of(correlation.residual);
    lifted.group_by = correlation.inner_keys.clone();
    let mut projected = correlation.inner_keys.clone();
    projected.push(aggregate);
    lifted.output_schema = schema_of(&projected);
    lifted.projections = projected
        .into_iter()
        .map(|e| BoundSelectItem::Expr(e, None))
        .collect();

    if outer_ref_escapes(&lifted, &correlation.owned, outer_tables) {
        return None;
    }
    Some(LiftedAggregate {
        select: Box::new(lifted),
        outer_keys: correlation.outer_keys,
        empty,
    })
}

/// Plans a lifted subquery against the same catalog and peer facts the
/// enclosing plan was built with
pub fn plan_lifted(select: Box<BoundSelect>, ctx: &Arc<ExecutionContext>) -> Result<PhysicalPlan> {
    let logical =
        zyron_planner::logical::builder::build_logical_plan(&BoundStatement::Select(*select))?;
    let optimized = Optimizer::new(&ctx.catalog).optimize(logical)?;
    // The guard spans the build alone, because a lock held across an await
    // pins the registry for the length of the query
    let peerGuard = ctx.peers.as_ref().map(|p| p.read());
    zyron_planner::physical::builder::build_physical_plan(
        optimized,
        &ctx.catalog,
        peerGuard.as_deref().map(|p| &**p),
    )
}

/// One lifted EXISTS, holding the keys its subquery produced
struct Probe {
    template: PhysicalPlan,
    outer_keys: Vec<BoundExpr>,
    negated: bool,
    keys: HashSet<Vec<ScalarValue>>,
}

/// Answers a WHERE whose EXISTS terms were lifted into hash probes.
///
/// Each subquery runs once, before the first outer batch, and its keys are
/// held for the rest of the scan. What is left of the predicate is
/// evaluated over the batch exactly as an ordinary filter would
pub struct SemiJoinFilterOperator {
    child: Box<dyn Operator>,
    residual: Option<BoundExpr>,
    probes: Vec<Probe>,
    input_schema: Vec<LogicalColumn>,
    base_params: Vec<ScalarValue>,
    ctx: Arc<ExecutionContext>,
    built: bool,
}

impl SemiJoinFilterOperator {
    /// Runs every lifted subquery once and keeps the keys it produced.
    ///
    /// A row holding NULL in any key position is dropped rather than
    /// stored, because an equality against it is never true and so it can
    /// match no outer row
    async fn build_keys(&mut self) -> Result<()> {
        for probe in &mut self.probes {
            let batches = crate::executor::execute(probe.template.clone(), &self.ctx).await?;
            for batch in &batches {
                for row in 0..batch.num_rows {
                    if batch.columns.iter().any(|c| c.is_null(row)) {
                        continue;
                    }
                    probe
                        .keys
                        .insert(batch.columns.iter().map(|c| c.get_scalar(row)).collect());
                }
            }
        }
        Ok(())
    }

    /// The rows of one batch that survive every probe and the residual.
    ///
    /// An outer row with NULL in a key compares equal to nothing, so it
    /// finds no match: EXISTS is false for it and NOT EXISTS is true, which
    /// is what a miss already means
    fn surviving(&self, batch: &DataBatch) -> Result<Vec<bool>> {
        let mut mask = match &self.residual {
            Some(predicate) => {
                let col = evaluate(predicate, batch, &self.input_schema, &self.base_params)?;
                column_to_mask(&col)
            }
            None => vec![true; batch.num_rows],
        };
        for probe in &self.probes {
            let key_cols: Vec<Column> = probe
                .outer_keys
                .iter()
                .map(|e| evaluate(e, batch, &self.input_schema, &self.base_params))
                .collect::<Result<_>>()?;
            for (row, keep) in mask.iter_mut().enumerate() {
                if !*keep {
                    continue;
                }
                let found = if key_cols.iter().any(|c| c.is_null(row)) {
                    false
                } else {
                    let key: Vec<ScalarValue> =
                        key_cols.iter().map(|c| c.get_scalar(row)).collect();
                    probe.keys.contains(&key)
                };
                *keep = found != probe.negated;
            }
        }
        Ok(mask)
    }
}

impl Operator for SemiJoinFilterOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            if !self.built {
                self.build_keys().await?;
                self.built = true;
            }
            loop {
                let Some(exec_batch) = self.child.next().await? else {
                    return Ok(None);
                };
                let mask = self.surviving(&exec_batch.batch)?;
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

/// A predicate rewritten into probes and what is left of it, decided
/// before any child operator is built so a refusal costs nothing
pub struct SemiJoinPlan {
    residual: Option<BoundExpr>,
    probes: Vec<Probe>,
}

impl SemiJoinPlan {
    /// Puts the plan over a child, which is built only once the rewrite is
    /// known to apply
    pub fn into_operator(
        self,
        child: Box<dyn Operator>,
        input_schema: &[LogicalColumn],
        base_params: &[ScalarValue],
        ctx: &Arc<ExecutionContext>,
    ) -> Box<dyn Operator> {
        Box::new(SemiJoinFilterOperator {
            child,
            residual: self.residual,
            probes: self.probes,
            input_schema: input_schema.to_vec(),
            base_params: base_params.to_vec(),
            ctx: ctx.clone(),
            built: false,
        })
    }
}

/// Rewrites a WHERE whose EXISTS terms all lift into hash probes, or None
/// when any of them does not.
///
/// All or nothing per predicate: a WHERE holding one liftable EXISTS and
/// one that is not still needs the per row operator for the second, and
/// that operator evaluates the whole predicate, so splitting them would
/// run the first subquery twice
pub async fn plan_semi_join(
    predicate: &BoundExpr,
    input_schema: &[LogicalColumn],
    ctx: &Arc<ExecutionContext>,
) -> Result<Option<SemiJoinPlan>> {
    let outer_tables: HashSet<usize> = input_schema.iter().filter_map(|c| c.table_idx).collect();
    let mut terms = Vec::new();
    conjuncts(predicate, &mut terms);

    let mut lifted: Vec<(Lifted, bool)> = Vec::new();
    let mut residual: Vec<BoundExpr> = Vec::new();
    for term in terms {
        match &term {
            BoundExpr::Exists { plan, negated } => match lift(plan, &outer_tables) {
                Some(l) => lifted.push((l, *negated)),
                None => return Ok(None),
            },
            // Any other term holding a correlated subquery keeps the whole
            // predicate on the per row path
            other if crate::correlated::expr_has_correlated_subquery(other) => return Ok(None),
            other => residual.push(other.clone()),
        }
    }
    if lifted.is_empty() {
        return Ok(None);
    }

    let mut probes = Vec::with_capacity(lifted.len());
    for (l, negated) in lifted {
        probes.push(Probe {
            template: plan_lifted(l.select, ctx)?,
            outer_keys: l.outer_keys,
            negated,
            keys: HashSet::new(),
        });
    }

    // The residual is evaluated by the synchronous expression evaluator,
    // which cannot run a plan. An uncorrelated subquery beside the lifted
    // ones is folded to a constant here, the same fold the per row path
    // applies before it starts evaluating rows
    let residual = match all_of(residual) {
        Some(expr) if crate::subquery::contains_subquery(&expr) => {
            Some(crate::subquery::materialize_expr(expr, ctx).await?)
        }
        other => other,
    };

    Ok(Some(SemiJoinPlan { residual, probes }))
}

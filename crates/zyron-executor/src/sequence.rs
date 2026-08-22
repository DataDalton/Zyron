//! Sequence function resolution for query execution.
//!
//! The synchronous expression evaluator cannot reach the async catalog, so
//! nextval/currval/setval/lastval are resolved in an async pre-pass before a
//! batch is evaluated. Each sequence call is materialized into an Int64 column
//! appended to the batch, and the call node is rewritten to a column reference
//! to that column. nextval produces a distinct value per row; currval, lastval,
//! and setval broadcast one value across the batch.

use zyron_common::{Result, TypeId, ZyronError};
use zyron_parser::ast::LiteralValue;
use zyron_planner::binder::{BoundExpr, ColumnRef};
use zyron_planner::logical::LogicalColumn;

use crate::batch::DataBatch;
use crate::column::{Column, ColumnData, ScalarValue};
use crate::context::ExecutionContext;

/// Reserved table index for columns synthesized to carry sequence values.
/// Chosen above any real table index so synthesized columns never collide with
/// bound column references.
const SEQUENCE_TABLE_IDX: usize = usize::MAX;

/// Per-session sequence state. Tracks the most recent value handed out by
/// nextval for each sequence (currval) and the most recent value of any
/// sequence in the session (lastval).
pub struct SessionSeqState {
    by_sequence: scc::HashMap<u32, i64>,
    last: parking_lot::Mutex<Option<i64>>,
}

impl SessionSeqState {
    pub fn new() -> Self {
        Self {
            by_sequence: scc::HashMap::new(),
            last: parking_lot::Mutex::new(None),
        }
    }

    fn record(&self, sequence_id: u32, value: i64) {
        self.by_sequence.upsert_sync(sequence_id, value);
        *self.last.lock() = Some(value);
    }

    fn currval(&self, sequence_id: u32) -> Option<i64> {
        self.by_sequence.read_sync(&sequence_id, |_, v| *v)
    }

    fn lastval(&self) -> Option<i64> {
        *self.last.lock()
    }
}

impl Default for SessionSeqState {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, PartialEq)]
enum SeqKind {
    NextVal,
    CurrVal,
    LastVal,
    SetVal,
}

fn classify(name: &str) -> Option<SeqKind> {
    match name.to_lowercase().as_str() {
        "nextval" => Some(SeqKind::NextVal),
        "currval" => Some(SeqKind::CurrVal),
        "lastval" => Some(SeqKind::LastVal),
        "setval" => Some(SeqKind::SetVal),
        _ => None,
    }
}

/// True when the expression tree contains a sequence function call.
pub fn contains_sequence(expr: &BoundExpr) -> bool {
    match expr {
        BoundExpr::Function { name, args, .. } => {
            classify(name).is_some() || args.iter().any(contains_sequence)
        }
        BoundExpr::BinaryOp { left, right, .. } => {
            contains_sequence(left) || contains_sequence(right)
        }
        BoundExpr::UnaryOp { expr, .. }
        | BoundExpr::IsNull { expr, .. }
        | BoundExpr::Cast { expr, .. }
        | BoundExpr::Nested(expr) => contains_sequence(expr),
        BoundExpr::InList { expr, list, .. } => {
            contains_sequence(expr) || list.iter().any(contains_sequence)
        }
        BoundExpr::Between {
            expr, low, high, ..
        } => contains_sequence(expr) || contains_sequence(low) || contains_sequence(high),
        BoundExpr::Like { expr, pattern, .. } | BoundExpr::ILike { expr, pattern, .. } => {
            contains_sequence(expr) || contains_sequence(pattern)
        }
        BoundExpr::Case {
            operand,
            conditions,
            else_result,
            ..
        } => {
            operand.as_deref().is_some_and(contains_sequence)
                || conditions
                    .iter()
                    .any(|w| contains_sequence(&w.condition) || contains_sequence(&w.result))
                || else_result.as_deref().is_some_and(contains_sequence)
        }
        _ => false,
    }
}

/// A sequence call captured during the rewrite pass, to be resolved against
/// the catalog. `synth_id` is the synthesized column id its value lands in.
struct PendingSeq {
    kind: SeqKind,
    seq_name: String,
    value_expr: Option<BoundExpr>,
    is_called_expr: Option<BoundExpr>,
    synth_id: u16,
}

/// Extracts the sequence name from a sequence call's first argument. The name
/// is a string literal: nextval('seq'). lastval takes no argument.
fn sequence_name_from_args(kind: SeqKind, args: &[BoundExpr]) -> Result<String> {
    if kind == SeqKind::LastVal {
        return Ok(String::new());
    }
    match args.first() {
        Some(BoundExpr::Literal {
            value: LiteralValue::String(s),
            ..
        }) => Ok(s.clone()),
        _ => Err(ZyronError::ExecutionError(
            "sequence function requires a string literal sequence name".to_string(),
        )),
    }
}

/// Walks the expression, replacing each sequence call with a reference to a
/// synthesized column and recording the call in `pending`.
fn rewrite(expr: &mut BoundExpr, pending: &mut Vec<PendingSeq>, counter: &mut u16) -> Result<()> {
    if let BoundExpr::Function { name, args, .. } = expr {
        if let Some(kind) = classify(name) {
            let seq_name = sequence_name_from_args(kind, args)?;
            let value_expr = args.get(1).cloned();
            let is_called_expr = args.get(2).cloned();
            let synth_id = *counter;
            *counter += 1;
            pending.push(PendingSeq {
                kind,
                seq_name,
                value_expr,
                is_called_expr,
                synth_id,
            });
            *expr = BoundExpr::ColumnRef(ColumnRef {
                table_idx: SEQUENCE_TABLE_IDX,
                column_id: zyron_catalog::ColumnId(synth_id),
                type_id: TypeId::Int64,
                nullable: false,
                fractional_digits: None,
            });
            return Ok(());
        }
    }

    match expr {
        BoundExpr::BinaryOp { left, right, .. } => {
            rewrite(left, pending, counter)?;
            rewrite(right, pending, counter)?;
        }
        BoundExpr::UnaryOp { expr, .. }
        | BoundExpr::IsNull { expr, .. }
        | BoundExpr::Cast { expr, .. }
        | BoundExpr::Nested(expr) => rewrite(expr, pending, counter)?,
        BoundExpr::InList { expr, list, .. } => {
            rewrite(expr, pending, counter)?;
            for e in list {
                rewrite(e, pending, counter)?;
            }
        }
        BoundExpr::Between {
            expr, low, high, ..
        } => {
            rewrite(expr, pending, counter)?;
            rewrite(low, pending, counter)?;
            rewrite(high, pending, counter)?;
        }
        BoundExpr::Like { expr, pattern, .. } | BoundExpr::ILike { expr, pattern, .. } => {
            rewrite(expr, pending, counter)?;
            rewrite(pattern, pending, counter)?;
        }
        BoundExpr::Case {
            operand,
            conditions,
            else_result,
            ..
        } => {
            if let Some(o) = operand.as_deref_mut() {
                rewrite(o, pending, counter)?;
            }
            for w in conditions.iter_mut() {
                rewrite(&mut w.condition, pending, counter)?;
                rewrite(&mut w.result, pending, counter)?;
            }
            if let Some(e) = else_result.as_deref_mut() {
                rewrite(e, pending, counter)?;
            }
        }
        BoundExpr::Function { args, .. } => {
            for a in args {
                rewrite(a, pending, counter)?;
            }
        }
        _ => {}
    }
    Ok(())
}

/// Resolves sequence calls in `exprs` against the catalog and session state,
/// appending one Int64 column per call to `batch` and `schema`. After this
/// returns, the synchronous evaluator handles `exprs` with the synthesized
/// columns supplying the sequence values.
pub async fn materialize_sequences(
    exprs: &mut [BoundExpr],
    batch: &mut DataBatch,
    schema: &mut Vec<LogicalColumn>,
    ctx: &ExecutionContext,
) -> Result<()> {
    let mut pending: Vec<PendingSeq> = Vec::new();
    let mut counter: u16 = 0;
    for expr in exprs.iter_mut() {
        rewrite(expr, &mut pending, &mut counter)?;
    }
    if pending.is_empty() {
        return Ok(());
    }

    let num_rows = batch.num_rows.max(1);
    for p in pending {
        let values = resolve_one(&p, batch, schema, ctx, num_rows).await?;
        let synth = LogicalColumn {
            table_idx: Some(SEQUENCE_TABLE_IDX),
            column_id: zyron_catalog::ColumnId(p.synth_id),
            name: format!("__seq_{}", p.synth_id),
            type_id: TypeId::Int64,
            nullable: false,
            fractional_digits: None,
        };
        batch
            .columns
            .push(Column::new(ColumnData::Int64(values), TypeId::Int64));
        schema.push(synth);
    }
    // The batch row count is unchanged; appended columns all carry num_rows
    // entries. Guard the synthetic single-row case used by bare SELECT.
    if batch.num_rows == 0 {
        batch.num_rows = num_rows;
    }
    Ok(())
}

/// Produces the per-row Int64 values for one sequence call.
async fn resolve_one(
    p: &PendingSeq,
    batch: &DataBatch,
    schema: &[LogicalColumn],
    ctx: &ExecutionContext,
    num_rows: usize,
) -> Result<Vec<i64>> {
    match p.kind {
        SeqKind::NextVal => {
            let live = ctx.catalog.find_sequence_by_name(&p.seq_name)?;
            let mut out = Vec::with_capacity(num_rows);
            let mut last = 0i64;
            for _ in 0..num_rows {
                last = ctx.catalog.nextval_on(&live).await?;
                out.push(last);
            }
            if let Some(state) = &ctx.session_sequences {
                state.record(live.id, last);
            }
            Ok(out)
        }
        SeqKind::CurrVal => {
            let live = ctx.catalog.find_sequence_by_name(&p.seq_name)?;
            let state = ctx.session_sequences.as_ref().ok_or_else(|| {
                ZyronError::ExecutionError("currval is not available outside a session".to_string())
            })?;
            let v = state.currval(live.id).ok_or_else(|| {
                ZyronError::ExecutionError(format!(
                    "currval of sequence '{}' is not defined in this session",
                    p.seq_name
                ))
            })?;
            Ok(vec![v; num_rows])
        }
        SeqKind::LastVal => {
            let state = ctx.session_sequences.as_ref().ok_or_else(|| {
                ZyronError::ExecutionError("lastval is not available outside a session".to_string())
            })?;
            let v = state.lastval().ok_or_else(|| {
                ZyronError::ExecutionError("lastval is not yet defined in this session".to_string())
            })?;
            Ok(vec![v; num_rows])
        }
        SeqKind::SetVal => {
            let live = ctx.catalog.find_sequence_by_name(&p.seq_name)?;
            let value = scalar_i64(p.value_expr.as_ref(), batch, schema, &ctx.params)?.ok_or_else(
                || ZyronError::ExecutionError("setval requires a value argument".to_string()),
            )?;
            let is_called =
                match scalar_bool(p.is_called_expr.as_ref(), batch, schema, &ctx.params)? {
                    Some(b) => b,
                    None => true,
                };
            let v = ctx.catalog.setval_on(&live, value, is_called).await?;
            if let Some(state) = &ctx.session_sequences {
                state.record(live.id, value);
            }
            Ok(vec![v; num_rows])
        }
    }
}

/// Evaluates an argument expression to a single i64 at row 0.
fn scalar_i64(
    expr: Option<&BoundExpr>,
    batch: &DataBatch,
    schema: &[LogicalColumn],
    params: &[ScalarValue],
) -> Result<Option<i64>> {
    let Some(expr) = expr else {
        return Ok(None);
    };
    let col = crate::expr::evaluate(expr, batch, schema, params)?;
    Ok(col.data.get_scalar(0).to_i128().map(|v| v as i64))
}

/// Evaluates an argument expression to a single bool at row 0.
fn scalar_bool(
    expr: Option<&BoundExpr>,
    batch: &DataBatch,
    schema: &[LogicalColumn],
    params: &[ScalarValue],
) -> Result<Option<bool>> {
    let Some(expr) = expr else {
        return Ok(None);
    };
    let col = crate::expr::evaluate(expr, batch, schema, params)?;
    match col.data.get_scalar(0) {
        ScalarValue::Boolean(b) => Ok(Some(b)),
        other => Ok(other.to_i128().map(|v| v != 0)),
    }
}

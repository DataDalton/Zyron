//! Project operator for column projection and expression evaluation.
//!
//! Pulls batches from a child operator, evaluates projection expressions,
//! and outputs a new batch with the projected columns. Projections that
//! are bare column references move their column out of the owned input
//! batch instead of deep-copying it.

use std::sync::Arc;

use zyron_common::{TypeId, ZyronError};
use zyron_planner::binder::BoundExpr;
use zyron_planner::logical::LogicalColumn;

use crate::batch::DataBatch;
use crate::column::{Column, ScalarValue};
use crate::context::ExecutionContext;
use crate::expr::{evaluate, resolve_column_index};
use crate::operator::{ExecutionBatch, Operator, OperatorResult};

/// Evaluates projection expressions and outputs a new batch with only
/// the projected columns.
pub struct ProjectOperator {
    child: Box<dyn Operator>,
    expressions: Vec<BoundExpr>,
    input_schema: Vec<LogicalColumn>,
    /// Bound parameter values from the extended query protocol. Passed to
    /// each evaluator call so projection expressions referencing `$1`, `$2`,
    /// ... resolve against the values supplied at Bind time.
    params: Vec<ScalarValue>,
    /// Execution context for resolving sequence functions (nextval/currval/
    /// setval) in the projection list against the catalog. None disables
    /// sequence resolution for projections that never reference a sequence.
    ctx: Option<Arc<ExecutionContext>>,
    /// True when any projection expression references a sequence function,
    /// computed once at construction so the common no-sequence path skips the
    /// per-batch clone and async pre-pass.
    has_sequence: bool,
    /// For each projection, Some(input column index) when the expression is
    /// a bare column reference. Those skip the evaluator, taking the input
    /// column directly
    bare_refs: Vec<Option<usize>>,
    /// True at the last projection of each referenced input column, where
    /// the column moves out of the batch. Earlier duplicate references clone
    bare_last_use: Vec<bool>,
}

/// Identifies bare column-reference projections and marks the last use of
/// each referenced input column. Unresolvable references stay None and
/// fall back to the evaluator, which reports the error.
fn bare_ref_plan(
    expressions: &[BoundExpr],
    input_schema: &[LogicalColumn],
) -> (Vec<Option<usize>>, Vec<bool>) {
    let bare_refs: Vec<Option<usize>> = expressions
        .iter()
        .map(|e| match e {
            BoundExpr::ColumnRef(cr) => {
                resolve_column_index(cr.table_idx, cr.column_id, input_schema).ok()
            }
            _ => None,
        })
        .collect();
    let mut bare_last_use = vec![false; bare_refs.len()];
    let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for (i, bare) in bare_refs.iter().enumerate().rev() {
        if let Some(idx) = bare {
            if seen.insert(*idx) {
                bare_last_use[i] = true;
            }
        }
    }
    (bare_refs, bare_last_use)
}

impl ProjectOperator {
    pub fn new(
        child: Box<dyn Operator>,
        expressions: Vec<BoundExpr>,
        input_schema: Vec<LogicalColumn>,
    ) -> Self {
        Self::with_params(child, expressions, input_schema, Vec::new())
    }

    pub fn with_params(
        child: Box<dyn Operator>,
        expressions: Vec<BoundExpr>,
        input_schema: Vec<LogicalColumn>,
        params: Vec<ScalarValue>,
    ) -> Self {
        let has_sequence = expressions.iter().any(crate::sequence::contains_sequence);
        let (bare_refs, bare_last_use) = bare_ref_plan(&expressions, &input_schema);
        Self {
            child,
            expressions,
            input_schema,
            params,
            ctx: None,
            has_sequence,
            bare_refs,
            bare_last_use,
        }
    }

    /// Attaches the execution context so sequence functions in the projection
    /// resolve against the catalog.
    pub fn with_context(mut self, ctx: Arc<ExecutionContext>) -> Self {
        self.ctx = Some(ctx);
        self
    }
}

impl Operator for ProjectOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            let input = self.child.next().await?;
            let Some(exec_batch) = input else {
                return Ok(None);
            };

            if self.has_sequence {
                if let Some(ctx) = &self.ctx {
                    // Resolve sequence calls per batch: each call appends an
                    // Int64 column to a working batch and rewrites the call to
                    // a reference to it, then the evaluator runs as usual.
                    let mut exprs = self.expressions.clone();
                    let mut schema = self.input_schema.clone();
                    let mut batch = exec_batch.batch;
                    crate::sequence::materialize_sequences(
                        &mut exprs,
                        &mut batch,
                        &mut schema,
                        ctx,
                    )
                    .await?;
                    let mut columns = Vec::with_capacity(exprs.len());
                    for expr in &exprs {
                        columns.push(evaluate(expr, &batch, &schema, &self.params)?);
                    }
                    return Ok(Some(ExecutionBatch::new(DataBatch::new(columns))));
                }
            }

            // Computed projections evaluate first against the intact batch,
            // then bare references take their columns, moving each out of
            // the batch at its last use instead of deep-copying.
            let mut batch = exec_batch.batch;
            let mut slots: Vec<Option<Column>> = Vec::with_capacity(self.expressions.len());
            slots.resize_with(self.expressions.len(), || None);
            for (i, expr) in self.expressions.iter().enumerate() {
                if self.bare_refs[i].is_none() {
                    slots[i] = Some(evaluate(expr, &batch, &self.input_schema, &self.params)?);
                }
            }
            for (i, bare) in self.bare_refs.iter().enumerate() {
                if let Some(idx) = bare {
                    let col = if self.bare_last_use[i] {
                        std::mem::replace(
                            &mut batch.columns[*idx],
                            Column::null_column(TypeId::Null, 0),
                        )
                    } else {
                        batch.columns[*idx].clone()
                    };
                    slots[i] = Some(col);
                }
            }
            let mut columns = Vec::with_capacity(slots.len());
            for slot in slots {
                match slot {
                    Some(col) => columns.push(col),
                    None => {
                        return Err(ZyronError::ExecutionError(
                            "projection produced no column for one of its expressions"
                                .to_string(),
                        ));
                    }
                }
            }

            let batch = DataBatch::new(columns);
            Ok(Some(ExecutionBatch::new(batch)))
        })
    }
}

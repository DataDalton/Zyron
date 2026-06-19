//! Project operator for column projection and expression evaluation.
//!
//! Pulls batches from a child operator, evaluates projection expressions,
//! and outputs a new batch with the projected columns.

use std::sync::Arc;

use zyron_planner::binder::BoundExpr;
use zyron_planner::logical::LogicalColumn;

use crate::batch::DataBatch;
use crate::column::ScalarValue;
use crate::context::ExecutionContext;
use crate::expr::evaluate;
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
}

impl ProjectOperator {
    pub fn new(
        child: Box<dyn Operator>,
        expressions: Vec<BoundExpr>,
        input_schema: Vec<LogicalColumn>,
    ) -> Self {
        let has_sequence = expressions.iter().any(crate::sequence::contains_sequence);
        Self {
            child,
            expressions,
            input_schema,
            params: Vec::new(),
            ctx: None,
            has_sequence,
        }
    }

    pub fn with_params(
        child: Box<dyn Operator>,
        expressions: Vec<BoundExpr>,
        input_schema: Vec<LogicalColumn>,
        params: Vec<ScalarValue>,
    ) -> Self {
        let has_sequence = expressions.iter().any(crate::sequence::contains_sequence);
        Self {
            child,
            expressions,
            input_schema,
            params,
            ctx: None,
            has_sequence,
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

            let mut columns = Vec::with_capacity(self.expressions.len());
            for expr in &self.expressions {
                let col = evaluate(expr, &exec_batch.batch, &self.input_schema, &self.params)?;
                columns.push(col);
            }

            let batch = DataBatch::new(columns);
            Ok(Some(ExecutionBatch::new(batch)))
        })
    }
}

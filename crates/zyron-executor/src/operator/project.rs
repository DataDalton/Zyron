//! Project operator for column projection and expression evaluation.
//!
//! Pulls batches from a child operator, evaluates projection expressions,
//! and outputs a new batch with the projected columns.

use zyron_planner::binder::BoundExpr;
use zyron_planner::logical::LogicalColumn;

use crate::batch::DataBatch;
use crate::expr::evaluate;
use crate::operator::{ExecutionBatch, Operator, OperatorResult};

/// Evaluates projection expressions and outputs a new batch with only
/// the projected columns.
pub struct ProjectOperator {
    child: Box<dyn Operator>,
    expressions: Vec<BoundExpr>,
    input_schema: Vec<LogicalColumn>,
}

impl ProjectOperator {
    pub fn new(
        child: Box<dyn Operator>,
        expressions: Vec<BoundExpr>,
        input_schema: Vec<LogicalColumn>,
    ) -> Self {
        Self {
            child,
            expressions,
            input_schema,
        }
    }
}

impl Operator for ProjectOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            let input = self.child.next().await?;
            let Some(exec_batch) = input else {
                return Ok(None);
            };

            let mut columns = Vec::with_capacity(self.expressions.len());
            for expr in &self.expressions {
                let col = evaluate(expr, &exec_batch.batch, &self.input_schema)?;
                columns.push(col);
            }

            let batch = DataBatch::new(columns);
            Ok(Some(ExecutionBatch::new(batch)))
        })
    }
}

//! Filter operator for predicate evaluation.
//!
//! Pulls batches from a child operator, evaluates a boolean predicate
//! expression, and emits only the rows where the predicate is true.

use zyron_planner::binder::BoundExpr;
use zyron_planner::logical::LogicalColumn;

use crate::compute::column_to_mask;
use crate::expr::evaluate;
use crate::operator::{ExecutionBatch, Operator, OperatorResult};

/// Filters rows from a child operator using a boolean predicate expression.
/// Skips batches that produce zero matching rows and continues pulling
/// from the child until a non-empty result or exhaustion.
pub struct FilterOperator {
    child: Box<dyn Operator>,
    predicate: BoundExpr,
    input_schema: Vec<LogicalColumn>,
}

impl FilterOperator {
    pub fn new(
        child: Box<dyn Operator>,
        predicate: BoundExpr,
        input_schema: Vec<LogicalColumn>,
    ) -> Self {
        Self {
            child,
            predicate,
            input_schema,
        }
    }
}

impl Operator for FilterOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            loop {
                let input = self.child.next().await?;
                let Some(exec_batch) = input else {
                    return Ok(None);
                };

                let mask_col = evaluate(&self.predicate, &exec_batch.batch, &self.input_schema)?;
                let mask = column_to_mask(&mask_col);

                let filtered = exec_batch.batch.filter(&mask);

                if filtered.num_rows == 0 {
                    continue;
                }

                // Filter tuple_ids in parallel with the boolean mask.
                let filtered_ids = exec_batch.tuple_ids.map(|ids| {
                    mask.iter()
                        .enumerate()
                        .filter_map(|(i, &keep)| if keep { Some(ids[i]) } else { None })
                        .collect::<Vec<_>>()
                });

                return match filtered_ids {
                    Some(ids) => Ok(Some(ExecutionBatch::with_tuple_ids(filtered, ids))),
                    None => Ok(Some(ExecutionBatch::new(filtered))),
                };
            }
        })
    }
}

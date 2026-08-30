//! View write operator for INSTEAD OF triggers.
//!
//! Executes DML aimed at a view that carries an INSTEAD OF trigger. The
//! source operator produces one row per affected view row; for each row the
//! view's trigger body runs with the row values bound as positional
//! parameters. The view's underlying tables are never written here, the
//! trigger body decides what to write. Emits the affected row count the way
//! the other DML operators do.

use std::sync::Arc;

use zyron_common::Result;

use crate::context::ExecutionContext;
use crate::operator::modify::count_batch;
use crate::operator::{ExecutionBatch, Operator, OperatorResult};

/// Drains the source, fires the view's INSTEAD OF trigger per row, and
/// returns a single count batch.
pub struct ViewTriggerWriteOperator {
    source: Box<dyn Operator>,
    ctx: Arc<ExecutionContext>,
    view_id: u32,
    event: u8,
    param_map: Vec<Option<usize>>,
    done: bool,
}

impl ViewTriggerWriteOperator {
    pub fn new(
        source: Box<dyn Operator>,
        ctx: Arc<ExecutionContext>,
        view_id: u32,
        event: u8,
        param_map: Vec<Option<usize>>,
    ) -> Self {
        Self {
            source,
            ctx,
            view_id,
            event,
            param_map,
            done: false,
        }
    }
}

impl Operator for ViewTriggerWriteOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            if self.done {
                return Ok(None);
            }
            self.done = true;

            let mut total_rows: i64 = 0;
            loop {
                self.ctx.check_cancelled()?;
                let input = self.source.next().await?;
                let Some(exec_batch) = input else {
                    break;
                };
                if exec_batch.batch.num_rows == 0 {
                    continue;
                }
                crate::trigger::fire_instead_of_triggers(
                    &self.ctx,
                    self.view_id,
                    self.event,
                    &exec_batch.batch,
                    &self.param_map,
                )
                .await?;
                total_rows += exec_batch.batch.num_rows as i64;
            }
            Ok(Some(ExecutionBatch::new(count_batch(total_rows))))
        })
    }
}

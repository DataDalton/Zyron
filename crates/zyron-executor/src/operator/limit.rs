//! Limit and offset operator.
//!
//! Handles LIMIT by capping the number of emitted rows, and OFFSET
//! by skipping initial rows. Uses batch slicing for zero-copy sub-batches.

use crate::operator::{ExecutionBatch, Operator, OperatorResult};

/// Passes through up to `limit` rows after skipping `offset` rows.
pub struct LimitOperator {
    child: Box<dyn Operator>,
    limit: Option<u64>,
    offset: u64,
    rows_skipped: u64,
    rows_emitted: u64,
}

impl LimitOperator {
    pub fn new(child: Box<dyn Operator>, limit: Option<u64>, offset: Option<u64>) -> Self {
        Self {
            child,
            limit,
            offset: offset.unwrap_or(0),
            rows_skipped: 0,
            rows_emitted: 0,
        }
    }
}

impl Operator for LimitOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            // Check if limit already reached.
            if let Some(limit) = self.limit {
                if self.rows_emitted >= limit {
                    return Ok(None);
                }
            }

            loop {
                let input = self.child.next().await?;
                let Some(exec_batch) = input else {
                    return Ok(None);
                };

                let batch_rows = exec_batch.batch.num_rows as u64;

                // Handle OFFSET: skip rows.
                if self.rows_skipped < self.offset {
                    let remaining_to_skip = self.offset - self.rows_skipped;
                    if batch_rows <= remaining_to_skip {
                        self.rows_skipped += batch_rows;
                        continue;
                    }
                    // Partial skip: slice off the beginning.
                    let skip = remaining_to_skip as usize;
                    self.rows_skipped = self.offset;
                    let sliced_batch = exec_batch.batch.slice(skip, batch_rows as usize - skip);
                    let sliced_locs = exec_batch.locators.map(|locs| locs[skip..].to_vec());

                    let mut result_batch = sliced_batch;
                    let mut result_locs = sliced_locs;

                    // Apply limit to the sliced batch.
                    if let Some(limit) = self.limit {
                        let remaining_limit = limit.saturating_sub(self.rows_emitted) as usize;
                        if result_batch.num_rows > remaining_limit {
                            result_batch = result_batch.slice(0, remaining_limit);
                            result_locs = result_locs.map(|locs| locs[..remaining_limit].to_vec());
                        }
                    }

                    self.rows_emitted += result_batch.num_rows as u64;
                    return Ok(Some(ExecutionBatch {
                        batch: result_batch,
                        locators: result_locs,
                    }));
                }

                // Apply limit.
                let mut result_batch = exec_batch.batch;
                let mut result_locs = exec_batch.locators;

                if let Some(limit) = self.limit {
                    let remaining_limit = limit.saturating_sub(self.rows_emitted) as usize;
                    if result_batch.num_rows > remaining_limit {
                        result_batch = result_batch.slice(0, remaining_limit);
                        result_locs = result_locs.map(|locs| locs[..remaining_limit].to_vec());
                    }
                }

                self.rows_emitted += result_batch.num_rows as u64;
                return Ok(Some(ExecutionBatch {
                    batch: result_batch,
                    locators: result_locs,
                }));
            }
        })
    }
}

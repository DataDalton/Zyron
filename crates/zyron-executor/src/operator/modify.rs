//! DML operators for INSERT, UPDATE, and DELETE.
//!
//! Each operator pulls rows from a child, performs the storage mutation
//! (with WAL logging), and returns a single-row batch containing the
//! affected row count.

use std::sync::Arc;

use zyron_common::{TypeId, ZyronError};
use zyron_planner::binder::{BoundAssignment, BoundExpr};
use zyron_planner::logical::LogicalColumn;
use zyron_wal::Lsn;

use zyron_storage::TupleId;

use crate::batch::{DataBatch, batch_to_tuples};
use crate::column::{Column, ColumnData, NullBitmap, ScalarValue};
use crate::context::ExecutionContext;
use crate::expr::evaluate;
use crate::operator::{ExecutionBatch, Operator, OperatorResult};

/// Serializes a TupleId into bytes for WAL payload.
fn tuple_id_payload(tid: &TupleId) -> Vec<u8> {
    let mut buf = Vec::with_capacity(14);
    buf.extend_from_slice(&tid.page_id.file_id.to_le_bytes());
    buf.extend_from_slice(&tid.page_id.page_num.to_le_bytes());
    buf.extend_from_slice(&tid.slot_id.to_le_bytes());
    buf
}

// ---------------------------------------------------------------------------
// Helper: build a single-row batch with the affected row count
// ---------------------------------------------------------------------------

fn count_batch(count: i64) -> DataBatch {
    let data = ColumnData::Int64(vec![count]);
    let nulls = NullBitmap::none(1);
    let col = Column::with_nulls(data, nulls, TypeId::Int64);
    DataBatch::new(vec![col])
}

// ---------------------------------------------------------------------------
// ValuesOperator
// ---------------------------------------------------------------------------

/// Produces rows from literal VALUES expressions.
/// Evaluates each row of expressions into a columnar batch.
pub struct ValuesOperator {
    rows: Vec<Vec<BoundExpr>>,
    schema: Vec<LogicalColumn>,
    emitted: bool,
}

impl ValuesOperator {
    pub fn new(rows: Vec<Vec<BoundExpr>>, schema: Vec<LogicalColumn>) -> Self {
        Self {
            rows,
            schema,
            emitted: false,
        }
    }
}

impl Operator for ValuesOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            if self.emitted || self.rows.is_empty() {
                return Ok(None);
            }
            self.emitted = true;

            let num_cols = self.schema.len();
            let num_rows = self.rows.len();

            let mut col_data: Vec<ColumnData> = self
                .schema
                .iter()
                .map(|c| ColumnData::with_capacity(c.type_id, num_rows))
                .collect();
            let mut col_nulls: Vec<NullBitmap> =
                (0..num_cols).map(|_| NullBitmap::empty()).collect();

            // Create a dummy empty batch for evaluating literal expressions.
            let dummy = DataBatch::empty();

            for row_exprs in &self.rows {
                for (c, expr) in row_exprs.iter().enumerate() {
                    let col = evaluate(expr, &dummy, &self.schema)?;
                    let scalar = if col.len() > 0 {
                        col.get_scalar(0)
                    } else {
                        ScalarValue::Null
                    };
                    col_nulls[c].push(scalar.is_null());
                    col_data[c].push_scalar(&scalar);
                }
            }

            let columns: Vec<Column> = col_data
                .into_iter()
                .zip(col_nulls)
                .zip(self.schema.iter())
                .map(|((data, nulls), lc)| Column::with_nulls(data, nulls, lc.type_id))
                .collect();

            Ok(Some(ExecutionBatch::new(DataBatch::new(columns))))
        })
    }
}

// ---------------------------------------------------------------------------
// InsertOperator
// ---------------------------------------------------------------------------

/// Pulls rows from a source operator, encodes them as tuples,
/// logs to WAL, inserts into the heap file, and returns the row count.
pub struct InsertOperator {
    source: Box<dyn Operator>,
    ctx: Arc<ExecutionContext>,
    table_id: zyron_catalog::TableId,
    finished: bool,
}

impl InsertOperator {
    pub fn new(
        source: Box<dyn Operator>,
        ctx: Arc<ExecutionContext>,
        table_id: zyron_catalog::TableId,
    ) -> Self {
        Self {
            source,
            ctx,
            table_id,
            finished: false,
        }
    }
}

impl Operator for InsertOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            if self.finished {
                return Ok(None);
            }
            self.finished = true;

            let table_entry = self.ctx.get_table_entry(self.table_id)?;
            let heap_file = self.ctx.get_heap_file(self.table_id)?;
            let mut total_inserted: i64 = 0;
            let mut prev_lsn = Lsn(0);

            loop {
                let input = self.source.next().await?;
                let Some(exec_batch) = input else {
                    break;
                };

                let tuples =
                    batch_to_tuples(&exec_batch.batch, &table_entry.columns, self.ctx.txn_id);

                // WAL log each insert.
                for tuple in &tuples {
                    prev_lsn = self
                        .ctx
                        .wal
                        .log_insert(self.ctx.txn_id, prev_lsn, tuple.data())?;
                }

                heap_file.insert_batch(&tuples).await?;
                total_inserted += tuples.len() as i64;
            }

            Ok(Some(ExecutionBatch::new(count_batch(total_inserted))))
        })
    }
}

// ---------------------------------------------------------------------------
// DeleteOperator
// ---------------------------------------------------------------------------

/// Pulls rows with tuple IDs from a child scan, logs deletions to WAL,
/// deletes from the heap, and returns the row count.
pub struct DeleteOperator {
    child: Box<dyn Operator>,
    ctx: Arc<ExecutionContext>,
    table_id: zyron_catalog::TableId,
    finished: bool,
}

impl DeleteOperator {
    pub fn new(
        child: Box<dyn Operator>,
        ctx: Arc<ExecutionContext>,
        table_id: zyron_catalog::TableId,
    ) -> Self {
        Self {
            child,
            ctx,
            table_id,
            finished: false,
        }
    }
}

impl Operator for DeleteOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            if self.finished {
                return Ok(None);
            }
            self.finished = true;

            let heap_file = self.ctx.get_heap_file(self.table_id)?;
            let mut total_deleted: i64 = 0;
            let mut prev_lsn = Lsn(0);

            loop {
                let input = self.child.next().await?;
                let Some(exec_batch) = input else {
                    break;
                };

                let tuple_ids = exec_batch.tuple_ids.ok_or_else(|| {
                    ZyronError::Internal("DeleteOperator requires tuple IDs from scan".into())
                })?;

                // WAL log each deletion.
                for tid in &tuple_ids {
                    let payload = tuple_id_payload(tid);
                    prev_lsn = self
                        .ctx
                        .wal
                        .log_delete(self.ctx.txn_id, prev_lsn, &payload)?;
                }

                let deleted = heap_file.delete_batch(&tuple_ids).await?;
                total_deleted += deleted as i64;
            }

            Ok(Some(ExecutionBatch::new(count_batch(total_deleted))))
        })
    }
}

// ---------------------------------------------------------------------------
// UpdateOperator
// ---------------------------------------------------------------------------

/// Pulls rows with tuple IDs from a child scan, evaluates assignment
/// expressions to produce updated column values, deletes old tuples,
/// inserts new tuples, and returns the row count.
pub struct UpdateOperator {
    child: Box<dyn Operator>,
    ctx: Arc<ExecutionContext>,
    table_id: zyron_catalog::TableId,
    assignments: Vec<BoundAssignment>,
    input_schema: Vec<LogicalColumn>,
    finished: bool,
}

impl UpdateOperator {
    pub fn new(
        child: Box<dyn Operator>,
        ctx: Arc<ExecutionContext>,
        table_id: zyron_catalog::TableId,
        assignments: Vec<BoundAssignment>,
        input_schema: Vec<LogicalColumn>,
    ) -> Self {
        Self {
            child,
            ctx,
            table_id,
            assignments,
            input_schema,
            finished: false,
        }
    }
}

impl Operator for UpdateOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            if self.finished {
                return Ok(None);
            }
            self.finished = true;

            let table_entry = self.ctx.get_table_entry(self.table_id)?;
            let heap_file = self.ctx.get_heap_file(self.table_id)?;
            let mut total_updated: i64 = 0;
            let mut prev_lsn = Lsn(0);

            loop {
                let input = self.child.next().await?;
                let Some(exec_batch) = input else {
                    break;
                };

                let tuple_ids = exec_batch.tuple_ids.ok_or_else(|| {
                    ZyronError::Internal("UpdateOperator requires tuple IDs from scan".into())
                })?;

                // Build the updated batch by cloning original columns
                // and replacing assigned columns with new values.
                let mut updated_columns = exec_batch.batch.columns.clone();

                for assignment in &self.assignments {
                    let new_col =
                        evaluate(&assignment.value, &exec_batch.batch, &self.input_schema)?;

                    // Find the column index matching this assignment's column_id.
                    let col_idx = self
                        .input_schema
                        .iter()
                        .position(|lc| lc.column_id == assignment.column_id)
                        .ok_or_else(|| {
                            ZyronError::Internal(format!(
                                "Assignment column {:?} not found in schema",
                                assignment.column_id
                            ))
                        })?;

                    updated_columns[col_idx] = new_col;
                }

                let updated_batch = DataBatch::new(updated_columns);
                let new_tuples =
                    batch_to_tuples(&updated_batch, &table_entry.columns, self.ctx.txn_id);

                // Delete old tuples (MVCC: mark with xmax).
                for tid in &tuple_ids {
                    let payload = tuple_id_payload(tid);
                    prev_lsn = self
                        .ctx
                        .wal
                        .log_delete(self.ctx.txn_id, prev_lsn, &payload)?;
                }
                heap_file.delete_batch(&tuple_ids).await?;

                // Insert new tuples.
                for tuple in &new_tuples {
                    prev_lsn = self
                        .ctx
                        .wal
                        .log_insert(self.ctx.txn_id, prev_lsn, tuple.data())?;
                }
                heap_file.insert_batch(&new_tuples).await?;

                total_updated += tuple_ids.len() as i64;
            }

            Ok(Some(ExecutionBatch::new(count_batch(total_updated))))
        })
    }
}

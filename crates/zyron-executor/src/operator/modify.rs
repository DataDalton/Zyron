//! DML operators for INSERT, UPDATE, and DELETE.
//!
//! Each operator pulls rows from a child, performs the storage mutation
//! (with WAL logging), and returns a single-row batch containing the
//! affected row count.

use std::sync::Arc;

use zyron_common::{TypeId, ZyronError};
use zyron_planner::binder::{BoundAssignment, BoundExpr};
use zyron_planner::logical::LogicalColumn;
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
            let txn_id = self.ctx.txn_id;

            loop {
                self.ctx.check_cancelled()?;
                let input = self.source.next().await?;
                let Some(exec_batch) = input else {
                    break;
                };

                let tuples = batch_to_tuples(&exec_batch.batch, &table_entry.columns, txn_id);

                // Batch WAL log: one CAS + commit for all inserts in this batch.
                let batch_records: Vec<(u32, &[u8])> =
                    tuples.iter().map(|t| (txn_id, t.data())).collect();
                let lsns = self.ctx.wal.log_insert_batch(&batch_records)?;
                let last_lsn = lsns.last().copied().unwrap_or(zyron_wal::Lsn::INVALID);

                let tuple_ids = heap_file.insert_batch(&tuples).await?;

                // Stamp dirty pages with WAL LSN for checkpoint ordering.
                // Duplicate page_ids are harmless: set_dirty_lsn uses CAS from 0,
                // so only the first call per page succeeds.
                for tid in &tuple_ids {
                    self.ctx
                        .buffer_pool
                        .mark_dirty_with_lsn(tid.page_id, last_lsn.0);
                }

                total_inserted += tuples.len() as i64;

                // Notify CDC hook if present.
                if let Some(ref hook) = self.ctx.cdc_hook {
                    let tuple_refs: Vec<&[u8]> = tuples.iter().map(|t| t.data()).collect();
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_micros() as i64;
                    if let Err(e) =
                        hook.on_insert(self.table_id.0, &tuple_refs, last_lsn.0, now, txn_id, true)
                    {
                        eprintln!("CDC insert hook failed: {e}");
                    }
                }
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
            let txn_id = self.ctx.txn_id;

            loop {
                self.ctx.check_cancelled()?;
                let input = self.child.next().await?;
                let Some(exec_batch) = input else {
                    break;
                };

                let tuple_ids = exec_batch.tuple_ids.ok_or_else(|| {
                    ZyronError::Internal("DeleteOperator requires tuple IDs from scan".into())
                })?;

                // Capture old tuples for CDC hook (batch data is from the scan).
                let old_tuples_for_cdc = if self.ctx.cdc_hook.is_some() {
                    let table_entry = self.ctx.get_table_entry(self.table_id)?;
                    Some(batch_to_tuples(
                        &exec_batch.batch,
                        &table_entry.columns,
                        txn_id,
                    ))
                } else {
                    None
                };

                // Batch WAL log: one CAS + commit for all deletes in this batch.
                let payloads: Vec<Vec<u8>> = tuple_ids.iter().map(tuple_id_payload).collect();
                let batch_records: Vec<(u32, &[u8])> =
                    payloads.iter().map(|p| (txn_id, p.as_slice())).collect();
                let lsns = self.ctx.wal.log_delete_batch(&batch_records)?;
                let last_lsn = lsns.last().copied().unwrap_or(zyron_wal::Lsn::INVALID);

                let deleted = heap_file.delete_batch(&tuple_ids).await?;

                // Stamp dirty pages with WAL LSN for checkpoint ordering.
                // Duplicate page_ids are harmless: set_dirty_lsn uses CAS from 0.
                for tid in &tuple_ids {
                    self.ctx
                        .buffer_pool
                        .mark_dirty_with_lsn(tid.page_id, last_lsn.0);
                }

                total_deleted += deleted as i64;

                // Notify CDC hook if present.
                if let Some(ref hook) = self.ctx.cdc_hook {
                    if let Some(ref old_tuples) = old_tuples_for_cdc {
                        let refs: Vec<&[u8]> = old_tuples.iter().map(|t| t.data()).collect();
                        let now = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_micros() as i64;
                        if let Err(e) =
                            hook.on_delete(self.table_id.0, &refs, last_lsn.0, now, txn_id, true)
                        {
                            eprintln!("CDC delete hook failed: {e}");
                        }
                    }
                }
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
            let txn_id = self.ctx.txn_id;

            loop {
                self.ctx.check_cancelled()?;
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
                let new_tuples = batch_to_tuples(&updated_batch, &table_entry.columns, txn_id);

                // Batch WAL log deletes: one CAS + commit for all.
                let delete_payloads: Vec<Vec<u8>> =
                    tuple_ids.iter().map(tuple_id_payload).collect();
                let delete_records: Vec<(u32, &[u8])> = delete_payloads
                    .iter()
                    .map(|p| (txn_id, p.as_slice()))
                    .collect();
                let del_lsns = self.ctx.wal.log_delete_batch(&delete_records)?;
                let del_last_lsn = del_lsns.last().copied().unwrap_or(zyron_wal::Lsn::INVALID);
                heap_file.delete_batch(&tuple_ids).await?;

                // Stamp deleted pages with WAL LSN for checkpoint ordering.
                // Duplicate page_ids are harmless: set_dirty_lsn uses CAS from 0.
                for tid in &tuple_ids {
                    self.ctx
                        .buffer_pool
                        .mark_dirty_with_lsn(tid.page_id, del_last_lsn.0);
                }

                // Batch WAL log inserts: one CAS + commit for all.
                let insert_records: Vec<(u32, &[u8])> =
                    new_tuples.iter().map(|t| (txn_id, t.data())).collect();
                let ins_lsns = self.ctx.wal.log_insert_batch(&insert_records)?;
                let ins_last_lsn = ins_lsns.last().copied().unwrap_or(zyron_wal::Lsn::INVALID);
                let new_tuple_ids = heap_file.insert_batch(&new_tuples).await?;

                // Stamp inserted pages with WAL LSN for checkpoint ordering.
                for tid in &new_tuple_ids {
                    self.ctx
                        .buffer_pool
                        .mark_dirty_with_lsn(tid.page_id, ins_last_lsn.0);
                }

                total_updated += tuple_ids.len() as i64;

                // Notify CDC hook if present.
                if let Some(ref hook) = self.ctx.cdc_hook {
                    let old_tuples =
                        batch_to_tuples(&exec_batch.batch, &table_entry.columns, txn_id);
                    let old_slices: Vec<&[u8]> = old_tuples.iter().map(|t| t.data()).collect();
                    let new_refs_data: Vec<&[u8]> = new_tuples.iter().map(|t| t.data()).collect();
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_micros() as i64;
                    if let Err(e) = hook.on_update(
                        self.table_id.0,
                        &old_slices,
                        &new_refs_data,
                        ins_last_lsn.0,
                        now,
                        txn_id,
                        true,
                    ) {
                        eprintln!("CDC update hook failed: {e}");
                    }
                }
            }

            Ok(Some(ExecutionBatch::new(count_batch(total_updated))))
        })
    }
}

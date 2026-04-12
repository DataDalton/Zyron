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

/// Extracts text content from a DataBatch row for FTS indexing into a reusable buffer.
/// Concatenates all text-type columns (Varchar, Text, Char) for the given row
/// into the buffer, separated by spaces. The caller should call buf.clear() between rows.
fn extract_fts_text_into(
    batch: &DataBatch,
    row_idx: usize,
    columns: &[zyron_catalog::ColumnEntry],
    buf: &mut String,
) {
    for (col_idx, col_entry) in columns.iter().enumerate() {
        match col_entry.type_id {
            TypeId::Varchar | TypeId::Text | TypeId::Char => {}
            _ => continue,
        }
        if col_idx >= batch.columns.len() {
            continue;
        }
        let col = &batch.columns[col_idx];
        if row_idx >= col.data.len() {
            continue;
        }
        if col.nulls.is_null(row_idx) {
            continue;
        }
        if let ColumnData::Utf8(ref strings) = col.data {
            if row_idx < strings.len() {
                if !buf.is_empty() {
                    buf.push(' ');
                }
                buf.push_str(&strings[row_idx]);
            }
        }
    }
}

/// Extracts the raw vector bytes for a specific column from a DataBatch row.
/// The target column is identified by its catalog ColumnId rather than by
/// position, so tables with multiple vector columns correctly route each
/// index's maintenance to its own column. Returns None if the column is not
/// present in the batch, is null, or is not a Vector column.
fn extract_vector_bytes<'a>(
    batch: &'a DataBatch,
    row_idx: usize,
    columns: &[zyron_catalog::ColumnEntry],
    target_column_id: u16,
) -> Option<&'a [u8]> {
    let col_idx = columns
        .iter()
        .position(|c| c.id.0 == target_column_id && c.type_id == TypeId::Vector)?;
    if col_idx >= batch.columns.len() {
        return None;
    }
    let col = &batch.columns[col_idx];
    if row_idx >= col.data.len() || col.nulls.is_null(row_idx) {
        return None;
    }
    match &col.data {
        ColumnData::Binary(blobs) if row_idx < blobs.len() => Some(&blobs[row_idx]),
        _ => None,
    }
}

/// Reinterprets a byte slice as a slice of f32 values. Each 4 bytes in
/// little-endian order represent one f32. Returns an empty slice if the
/// input length is not a multiple of 4.
fn bytes_to_f32_slice(bytes: &[u8]) -> &[f32] {
    if bytes.len() % 4 != 0 || bytes.is_empty() {
        return &[];
    }
    // The vector column stores raw f32 bytes in native endianness (LE on x86).
    // Alignment is guaranteed by Vec<u8> backing store on all supported platforms.
    let (prefix, floats, suffix) = unsafe { bytes.align_to::<f32>() };
    if !prefix.is_empty() || !suffix.is_empty() {
        // Fallback: not aligned, should not happen in practice.
        return &[];
    }
    floats
}

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
                    let col = evaluate(expr, &dummy, &self.schema, &[])?;
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

                // Fire BEFORE INSERT triggers if present.
                if let Some(ref hook) = self.ctx.dml_hook {
                    let tuple_refs: Vec<&[u8]> = tuples.iter().map(|t| t.data()).collect();
                    if !hook.before_insert(self.table_id.0, &tuple_refs, txn_id)? {
                        continue; // Trigger cancelled the insert
                    }
                }

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

                // Maintain FTS indexes: add each inserted document.
                let fts_indexes = self.ctx.fts_indexes_for_table(self.table_id.0);
                if !fts_indexes.is_empty() {
                    let analyzer = zyron_search::SimpleAnalyzer;
                    let mut fts_buf = zyron_search::AnalysisBuffer::new();
                    let mut text_buf = String::with_capacity(256);
                    for (row_idx, tid) in tuple_ids.iter().enumerate() {
                        let doc_id =
                            zyron_search::encode_doc_id(tid.page_id.page_num, tid.slot_id)?;
                        text_buf.clear();
                        extract_fts_text_into(
                            &exec_batch.batch,
                            row_idx,
                            &table_entry.columns,
                            &mut text_buf,
                        );
                        for (idx_id, fts_idx) in &fts_indexes {
                            if let Err(e) = fts_idx.add_document_with_buf(
                                doc_id,
                                &text_buf,
                                &analyzer,
                                &mut fts_buf,
                            ) {
                                eprintln!("FTS index {} insert failed: {e}", idx_id.0);
                            }
                        }
                    }
                }

                // Maintain vector indexes: insert each new vector into every
                // vector index on the table, sourced from that index's column.
                let vec_index_ids = self.ctx.vector_indexes_for_table(self.table_id.0);
                if !vec_index_ids.is_empty() {
                    for (row_idx, tid) in tuple_ids.iter().enumerate() {
                        let vec_id =
                            zyron_search::encode_doc_id(tid.page_id.page_num, tid.slot_id)?;
                        for &idx_id in &vec_index_ids {
                            let Some(vec_idx) = self.ctx.get_vector_index(idx_id) else {
                                continue;
                            };
                            let col_id = vec_idx.column_id();
                            if let Some(vec_bytes) = extract_vector_bytes(
                                &exec_batch.batch,
                                row_idx,
                                &table_entry.columns,
                                col_id,
                            ) {
                                let vec_data = bytes_to_f32_slice(vec_bytes);
                                if let Err(e) = zyron_search::vector::VectorSearch::insert(
                                    vec_idx.as_ref(),
                                    vec_id,
                                    vec_data,
                                ) {
                                    eprintln!("vector index {} insert failed: {e}", idx_id);
                                }
                            }
                        }
                    }
                }

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

                // Fire BEFORE DELETE triggers if present.
                if let Some(ref hook) = self.ctx.dml_hook {
                    let table_entry = self.ctx.get_table_entry(self.table_id)?;
                    let old_tuples =
                        batch_to_tuples(&exec_batch.batch, &table_entry.columns, txn_id);
                    let refs: Vec<&[u8]> = old_tuples.iter().map(|t| t.data()).collect();
                    if !hook.before_delete(self.table_id.0, &refs, txn_id)? {
                        continue; // Trigger cancelled the delete
                    }
                }

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

                // Maintain FTS indexes: remove deleted documents.
                let fts_indexes = self.ctx.fts_indexes_for_table(self.table_id.0);
                if !fts_indexes.is_empty() {
                    for tid in &tuple_ids {
                        if let Ok(doc_id) =
                            zyron_search::encode_doc_id(tid.page_id.page_num, tid.slot_id)
                        {
                            for (idx_id, fts_idx) in &fts_indexes {
                                if let Err(e) = fts_idx.delete_document(doc_id) {
                                    eprintln!("FTS index {} delete failed: {e}", idx_id.0);
                                }
                            }
                        }
                    }
                }

                // Maintain vector indexes: delete vectors for removed rows.
                let vec_index_ids = self.ctx.vector_indexes_for_table(self.table_id.0);
                if !vec_index_ids.is_empty() {
                    for tid in &tuple_ids {
                        if let Ok(vec_id) =
                            zyron_search::encode_doc_id(tid.page_id.page_num, tid.slot_id)
                        {
                            for &idx_id in &vec_index_ids {
                                if let Some(vec_idx) = self.ctx.get_vector_index(idx_id) {
                                    if let Err(e) = zyron_search::vector::VectorSearch::delete(
                                        vec_idx.as_ref(),
                                        vec_id,
                                    ) {
                                        eprintln!("vector index {} delete failed: {e}", idx_id);
                                    }
                                }
                            }
                        }
                    }
                }

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
                    let new_col = evaluate(
                        &assignment.value,
                        &exec_batch.batch,
                        &self.input_schema,
                        &[],
                    )?;

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

                // Fire BEFORE UPDATE triggers if present.
                if let Some(ref hook) = self.ctx.dml_hook {
                    let old_tuples =
                        batch_to_tuples(&exec_batch.batch, &table_entry.columns, txn_id);
                    let old_refs: Vec<&[u8]> = old_tuples.iter().map(|t| t.data()).collect();
                    let new_refs: Vec<&[u8]> = new_tuples.iter().map(|t| t.data()).collect();
                    if !hook.before_update(self.table_id.0, &old_refs, &new_refs, txn_id)? {
                        continue; // Trigger cancelled the update
                    }
                }

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

                // Maintain FTS indexes: delete old docs, add new docs.
                let fts_indexes = self.ctx.fts_indexes_for_table(self.table_id.0);
                if !fts_indexes.is_empty() {
                    let analyzer = zyron_search::SimpleAnalyzer;
                    let mut fts_buf = zyron_search::AnalysisBuffer::new();
                    // Delete old documents
                    for tid in &tuple_ids {
                        if let Ok(doc_id) =
                            zyron_search::encode_doc_id(tid.page_id.page_num, tid.slot_id)
                        {
                            for (idx_id, fts_idx) in &fts_indexes {
                                if let Err(e) = fts_idx.delete_document(doc_id) {
                                    eprintln!("FTS index {} update-delete failed: {e}", idx_id.0);
                                }
                            }
                        }
                    }
                    // Add new documents
                    let mut text_buf = String::with_capacity(256);
                    for (row_idx, tid) in new_tuple_ids.iter().enumerate() {
                        let doc_id =
                            zyron_search::encode_doc_id(tid.page_id.page_num, tid.slot_id)?;
                        text_buf.clear();
                        extract_fts_text_into(
                            &updated_batch,
                            row_idx,
                            &table_entry.columns,
                            &mut text_buf,
                        );
                        for (idx_id, fts_idx) in &fts_indexes {
                            if let Err(e) = fts_idx.add_document_with_buf(
                                doc_id,
                                &text_buf,
                                &analyzer,
                                &mut fts_buf,
                            ) {
                                eprintln!("FTS index {} update-insert failed: {e}", idx_id.0);
                            }
                        }
                    }
                }

                // Maintain vector indexes: delete old vectors, insert new vectors.
                let vec_index_ids = self.ctx.vector_indexes_for_table(self.table_id.0);
                if !vec_index_ids.is_empty() {
                    // Delete old vectors
                    for tid in &tuple_ids {
                        if let Ok(vec_id) =
                            zyron_search::encode_doc_id(tid.page_id.page_num, tid.slot_id)
                        {
                            for &idx_id in &vec_index_ids {
                                if let Some(vec_idx) = self.ctx.get_vector_index(idx_id) {
                                    if let Err(e) = zyron_search::vector::VectorSearch::delete(
                                        vec_idx.as_ref(),
                                        vec_id,
                                    ) {
                                        eprintln!(
                                            "vector index {} update-delete failed: {e}",
                                            idx_id
                                        );
                                    }
                                }
                            }
                        }
                    }
                    // Insert new vectors, routing each index to its own column.
                    for (row_idx, tid) in new_tuple_ids.iter().enumerate() {
                        let vec_id =
                            zyron_search::encode_doc_id(tid.page_id.page_num, tid.slot_id)?;
                        for &idx_id in &vec_index_ids {
                            let Some(vec_idx) = self.ctx.get_vector_index(idx_id) else {
                                continue;
                            };
                            let col_id = vec_idx.column_id();
                            if let Some(vec_bytes) = extract_vector_bytes(
                                &updated_batch,
                                row_idx,
                                &table_entry.columns,
                                col_id,
                            ) {
                                let vec_data = bytes_to_f32_slice(vec_bytes);
                                if let Err(e) = zyron_search::vector::VectorSearch::insert(
                                    vec_idx.as_ref(),
                                    vec_id,
                                    vec_data,
                                ) {
                                    eprintln!("vector index {} update-insert failed: {e}", idx_id);
                                }
                            }
                        }
                    }
                }

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

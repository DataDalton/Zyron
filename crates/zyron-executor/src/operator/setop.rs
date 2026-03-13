//! Set operation operators for UNION, INTERSECT, and EXCEPT.
//!
//! UNION ALL concatenates left then right. UNION deduplicates via typed hashing.
//! INTERSECT and EXCEPT use a count-based HashMap with typed hashing and
//! columnar storage for collision resolution.

use std::collections::HashMap;

use crate::compute::PreHashMap;
use zyron_common::Result;
use zyron_parser::ast::SetOpType;

use crate::batch::DataBatch;
use crate::column::{Column, ColumnData};
use crate::compute;
use crate::operator::{ExecutionBatch, Operator, OperatorResult};

/// Executes set operations (UNION, INTERSECT, EXCEPT) on two child operators.
pub struct SetOpOperator {
    left: Box<dyn Operator>,
    right: Box<dyn Operator>,
    op: SetOpType,
    all: bool,
    state: SetOpState,
}

enum SetOpState {
    /// Draining left side.
    Left,
    /// Draining right side (used by UNION ALL).
    Right,
    /// Materialized result for INTERSECT/EXCEPT or UNION (distinct).
    Materialized {
        result: Option<DataBatch>,
        cursor: usize,
    },
    Done,
}

/// Columnar row store for set operations. Stores rows in column builders
/// and uses typed hashing for deduplication/counting.
struct RowStore {
    columns: Vec<Column>,
    hash_map: PreHashMap<u64, Vec<usize>>,
    counts: Vec<usize>,
    num_rows: usize,
}

impl RowStore {
    fn new() -> Self {
        Self {
            columns: Vec::new(),
            hash_map: PreHashMap::default(),
            counts: Vec::new(),
            num_rows: 0,
        }
    }

    fn ensure_columns(&mut self, batch: &DataBatch) {
        if self.columns.is_empty() {
            self.columns = batch
                .columns
                .iter()
                .map(|c| Column::new(ColumnData::with_capacity(c.type_id, 64), c.type_id))
                .collect();
        }
    }

    /// Finds or inserts a row, returning (store_index, was_new).
    fn find_or_insert(&mut self, batch: &DataBatch, row: usize, hash: u64) -> (usize, bool) {
        // First pass: check if row already exists (immutable borrow).
        if let Some(candidates) = self.hash_map.get(&hash) {
            for &idx in candidates {
                if self.row_equals(batch, row, idx) {
                    return (idx, false);
                }
            }
        }
        // Not found, insert.
        let idx = self.num_rows;
        self.hash_map.entry(hash).or_default().push(idx);
        for (ci, src) in batch.columns.iter().enumerate() {
            self.columns[ci].push_row_from(src, row);
        }
        self.counts.push(0);
        self.num_rows += 1;
        (idx, true)
    }

    /// Compares a batch row against a stored row using typed equality.
    fn row_equals(&self, batch: &DataBatch, batch_row: usize, store_row: usize) -> bool {
        for (ci, src) in batch.columns.iter().enumerate() {
            let a_null = src.is_null(batch_row);
            let b_null = self.columns[ci].is_null(store_row);
            if a_null != b_null {
                return false;
            }
            if a_null {
                continue;
            }
            if !column_values_equal_cross(&src.data, batch_row, &self.columns[ci].data, store_row) {
                return false;
            }
        }
        true
    }

    /// Builds a DataBatch from stored rows at the given indices.
    fn extract_rows(&self, indices: &[usize]) -> DataBatch {
        if indices.is_empty() || self.columns.is_empty() {
            return DataBatch::empty();
        }
        let num_cols = self.columns.len();
        let mut out_cols: Vec<Column> = self
            .columns
            .iter()
            .map(|c| {
                Column::new(
                    ColumnData::with_capacity(c.type_id, indices.len()),
                    c.type_id,
                )
            })
            .collect();

        for &idx in indices {
            for ci in 0..num_cols {
                out_cols[ci].push_row_from(&self.columns[ci], idx);
            }
        }

        DataBatch::new(out_cols)
    }
}

impl SetOpOperator {
    pub fn new(
        left: Box<dyn Operator>,
        right: Box<dyn Operator>,
        op: SetOpType,
        all: bool,
    ) -> Self {
        let state = match (&op, all) {
            (SetOpType::Union, true) => SetOpState::Left,
            _ => SetOpState::Materialized {
                result: None,
                cursor: 0,
            },
        };

        Self {
            left,
            right,
            op,
            all,
            state,
        }
    }

    /// Materializes the set operation result for non-streaming variants.
    async fn materialize(&mut self) -> Result<Option<DataBatch>> {
        match self.op {
            SetOpType::Union => self.materialize_union_distinct().await,
            SetOpType::Intersect => self.materialize_intersect().await,
            SetOpType::Except => self.materialize_except().await,
        }
    }

    async fn materialize_union_distinct(&mut self) -> Result<Option<DataBatch>> {
        let mut store = RowStore::new();

        // Drain left.
        loop {
            match self.left.next().await? {
                Some(eb) => {
                    let batch = &eb.batch;
                    store.ensure_columns(batch);
                    let col_refs: Vec<&Column> = batch.columns.iter().collect();
                    let hashes = compute::hash_column_batch(&col_refs, batch.num_rows);
                    for row in 0..batch.num_rows {
                        store.find_or_insert(batch, row, hashes[row]);
                    }
                }
                None => break,
            }
        }

        // Drain right.
        loop {
            match self.right.next().await? {
                Some(eb) => {
                    let batch = &eb.batch;
                    store.ensure_columns(batch);
                    let col_refs: Vec<&Column> = batch.columns.iter().collect();
                    let hashes = compute::hash_column_batch(&col_refs, batch.num_rows);
                    for row in 0..batch.num_rows {
                        store.find_or_insert(batch, row, hashes[row]);
                    }
                }
                None => break,
            }
        }

        if store.num_rows == 0 {
            return Ok(None);
        }

        let indices: Vec<usize> = (0..store.num_rows).collect();
        Ok(Some(store.extract_rows(&indices)))
    }

    async fn materialize_intersect(&mut self) -> Result<Option<DataBatch>> {
        let mut store = RowStore::new();

        // Build counts from left side.
        loop {
            match self.left.next().await? {
                Some(eb) => {
                    let batch = &eb.batch;
                    store.ensure_columns(batch);
                    let col_refs: Vec<&Column> = batch.columns.iter().collect();
                    let hashes = compute::hash_column_batch(&col_refs, batch.num_rows);
                    for row in 0..batch.num_rows {
                        let (idx, _) = store.find_or_insert(batch, row, hashes[row]);
                        store.counts[idx] += 1;
                    }
                }
                None => break,
            }
        }

        // Probe with right side, emit rows present in both.
        let mut result_indices: Vec<usize> = Vec::new();

        loop {
            match self.right.next().await? {
                Some(eb) => {
                    let batch = &eb.batch;
                    let col_refs: Vec<&Column> = batch.columns.iter().collect();
                    let hashes = compute::hash_column_batch(&col_refs, batch.num_rows);
                    for row in 0..batch.num_rows {
                        // Look up in store without inserting.
                        let hash = hashes[row];
                        if let Some(candidates) = store.hash_map.get(&hash) {
                            for &idx in candidates {
                                if store.row_equals(batch, row, idx) {
                                    if store.counts[idx] > 0 {
                                        result_indices.push(idx);
                                        if self.all {
                                            store.counts[idx] -= 1;
                                        } else {
                                            store.counts[idx] = 0;
                                        }
                                    }
                                    break;
                                }
                            }
                        }
                    }
                }
                None => break,
            }
        }

        if result_indices.is_empty() {
            return Ok(None);
        }

        Ok(Some(store.extract_rows(&result_indices)))
    }

    async fn materialize_except(&mut self) -> Result<Option<DataBatch>> {
        let mut store = RowStore::new();
        let mut left_order: Vec<usize> = Vec::new();

        // Build counts from left side, preserving order.
        loop {
            match self.left.next().await? {
                Some(eb) => {
                    let batch = &eb.batch;
                    store.ensure_columns(batch);
                    let col_refs: Vec<&Column> = batch.columns.iter().collect();
                    let hashes = compute::hash_column_batch(&col_refs, batch.num_rows);
                    for row in 0..batch.num_rows {
                        let (idx, _) = store.find_or_insert(batch, row, hashes[row]);
                        store.counts[idx] += 1;
                        left_order.push(idx);
                    }
                }
                None => break,
            }
        }

        // Remove right side rows from left counts.
        loop {
            match self.right.next().await? {
                Some(eb) => {
                    let batch = &eb.batch;
                    let col_refs: Vec<&Column> = batch.columns.iter().collect();
                    let hashes = compute::hash_column_batch(&col_refs, batch.num_rows);
                    for row in 0..batch.num_rows {
                        let hash = hashes[row];
                        if let Some(candidates) = store.hash_map.get(&hash) {
                            for &idx in candidates {
                                if store.row_equals(batch, row, idx) {
                                    if self.all {
                                        store.counts[idx] = store.counts[idx].saturating_sub(1);
                                    } else {
                                        store.counts[idx] = 0;
                                    }
                                    break;
                                }
                            }
                        }
                    }
                }
                None => break,
            }
        }

        // Emit remaining left rows in original order.
        let mut result_indices: Vec<usize> = Vec::new();
        let mut emit_counts: HashMap<usize, usize> = HashMap::new();

        for &idx in &left_order {
            let allowed = store.counts[idx];
            let emitted = emit_counts.entry(idx).or_insert(0);
            if *emitted < allowed {
                result_indices.push(idx);
                *emitted += 1;
            }
        }

        if result_indices.is_empty() {
            return Ok(None);
        }

        Ok(Some(store.extract_rows(&result_indices)))
    }
}

impl Operator for SetOpOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            // Check if we need to materialize first (separate borrow scope).
            let needs_materialize = matches!(
                &self.state,
                SetOpState::Materialized { result, cursor } if result.is_none() && *cursor == 0
            );

            if needs_materialize {
                let materialized = self.materialize().await?;
                self.state = SetOpState::Materialized {
                    result: materialized,
                    cursor: 0,
                };
            }

            match &mut self.state {
                SetOpState::Left => {
                    // UNION ALL: drain left first, then right.
                    match self.left.next().await? {
                        Some(eb) => Ok(Some(eb)),
                        None => {
                            self.state = SetOpState::Right;
                            self.right.next().await
                        }
                    }
                }
                SetOpState::Right => self.right.next().await,
                SetOpState::Materialized { result, cursor } => {
                    let Some(batch) = result else {
                        self.state = SetOpState::Done;
                        return Ok(None);
                    };

                    if *cursor >= batch.num_rows {
                        self.state = SetOpState::Done;
                        return Ok(None);
                    }

                    let remaining = batch.num_rows - *cursor;
                    let chunk = remaining.min(crate::batch::BATCH_SIZE);
                    let out = batch.slice(*cursor, chunk);
                    *cursor += chunk;

                    Ok(Some(ExecutionBatch::new(out)))
                }
                SetOpState::Done => Ok(None),
            }
        })
    }
}

/// Compares values at different indices across two ColumnData instances of the same type.
#[inline]
fn column_values_equal_cross(a: &ColumnData, a_idx: usize, b: &ColumnData, b_idx: usize) -> bool {
    match (a, b) {
        (ColumnData::Boolean(va), ColumnData::Boolean(vb)) => va[a_idx] == vb[b_idx],
        (ColumnData::Int8(va), ColumnData::Int8(vb)) => va[a_idx] == vb[b_idx],
        (ColumnData::Int16(va), ColumnData::Int16(vb)) => va[a_idx] == vb[b_idx],
        (ColumnData::Int32(va), ColumnData::Int32(vb)) => va[a_idx] == vb[b_idx],
        (ColumnData::Int64(va), ColumnData::Int64(vb)) => va[a_idx] == vb[b_idx],
        (ColumnData::Int128(va), ColumnData::Int128(vb)) => va[a_idx] == vb[b_idx],
        (ColumnData::UInt8(va), ColumnData::UInt8(vb)) => va[a_idx] == vb[b_idx],
        (ColumnData::UInt16(va), ColumnData::UInt16(vb)) => va[a_idx] == vb[b_idx],
        (ColumnData::UInt32(va), ColumnData::UInt32(vb)) => va[a_idx] == vb[b_idx],
        (ColumnData::UInt64(va), ColumnData::UInt64(vb)) => va[a_idx] == vb[b_idx],
        (ColumnData::Float32(va), ColumnData::Float32(vb)) => {
            va[a_idx].to_bits() == vb[b_idx].to_bits()
        }
        (ColumnData::Float64(va), ColumnData::Float64(vb)) => {
            va[a_idx].to_bits() == vb[b_idx].to_bits()
        }
        (ColumnData::Utf8(va), ColumnData::Utf8(vb)) => va[a_idx] == vb[b_idx],
        (ColumnData::Binary(va), ColumnData::Binary(vb)) => va[a_idx] == vb[b_idx],
        (ColumnData::FixedBinary16(va), ColumnData::FixedBinary16(vb)) => va[a_idx] == vb[b_idx],
        _ => false,
    }
}

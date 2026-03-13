//! Join operators for combining rows from two input relations.
//!
//! Provides three join implementations: nested loop (for cross joins and
//! arbitrary conditions), hash join (for equi-joins), and merge join
//! (for pre-sorted equi-joins). All support INNER, LEFT, RIGHT, FULL,
//! and CROSS join types.
//!
//! All join operators use JoinOutputBuffer for batched output,
//! accumulating multiple rows before emitting a DataBatch. This
//! eliminates per-row DataBatch allocation overhead.

use zyron_common::{Result, TypeId};
use zyron_parser::ast::JoinType;
use zyron_planner::binder::BoundExpr;
use zyron_planner::logical::LogicalColumn;

use crate::batch::{BATCH_SIZE, DataBatch};
use crate::column::{Column, ColumnData, NullBitmap};
use crate::compute::{self, FlatHashTable};
use crate::expr::{evaluate, resolve_column_index};
use crate::operator::{ExecutionBatch, Operator, OperatorResult};

// ---------------------------------------------------------------------------
// JoinOutputBuffer - batched output accumulator
// ---------------------------------------------------------------------------

/// Accumulates join output rows into column builders, flushing when
/// the buffer reaches BATCH_SIZE. Eliminates per-row DataBatch allocation.
struct JoinOutputBuffer {
    left_builders: Vec<Column>,
    right_builders: Vec<Column>,
    count: usize,
}

impl JoinOutputBuffer {
    fn new(left_types: &[(TypeId, usize)], right_types: &[(TypeId, usize)]) -> Self {
        let left_builders = left_types
            .iter()
            .map(|&(tid, _)| Column::new(ColumnData::with_capacity(tid, BATCH_SIZE), tid))
            .collect();
        let right_builders = right_types
            .iter()
            .map(|&(tid, _)| Column::new(ColumnData::with_capacity(tid, BATCH_SIZE), tid))
            .collect();
        Self {
            left_builders,
            right_builders,
            count: 0,
        }
    }

    #[inline]
    fn is_full(&self) -> bool {
        self.count >= BATCH_SIZE
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Appends a matched row: left_batch[left_row] + right_batch[right_row].
    #[inline]
    fn push_matched(
        &mut self,
        left_batch: &DataBatch,
        left_row: usize,
        right_batch: &DataBatch,
        right_row: usize,
    ) {
        for (builder, src) in self.left_builders.iter_mut().zip(&left_batch.columns) {
            builder.push_row_from(src, left_row);
        }
        for (builder, src) in self.right_builders.iter_mut().zip(&right_batch.columns) {
            builder.push_row_from(src, right_row);
        }
        self.count += 1;
    }

    /// Appends left_batch[left_row] + null-padded right side.
    #[inline]
    fn push_left_null_right(&mut self, left_batch: &DataBatch, left_row: usize) {
        for (builder, src) in self.left_builders.iter_mut().zip(&left_batch.columns) {
            builder.push_row_from(src, left_row);
        }
        for builder in &mut self.right_builders {
            builder.push_null();
        }
        self.count += 1;
    }

    /// Appends null-padded left side + right_batch[right_row].
    #[inline]
    fn push_null_left_right(&mut self, right_batch: &DataBatch, right_row: usize) {
        for builder in &mut self.left_builders {
            builder.push_null();
        }
        for (builder, src) in self.right_builders.iter_mut().zip(&right_batch.columns) {
            builder.push_row_from(src, right_row);
        }
        self.count += 1;
    }

    /// Drains accumulated rows into a DataBatch and resets the buffer.
    fn flush(
        &mut self,
        left_types: &[(TypeId, usize)],
        right_types: &[(TypeId, usize)],
    ) -> DataBatch {
        let mut columns = Vec::with_capacity(self.left_builders.len() + self.right_builders.len());

        let new_left: Vec<Column> = left_types
            .iter()
            .map(|&(tid, _)| Column::new(ColumnData::with_capacity(tid, BATCH_SIZE), tid))
            .collect();
        let new_right: Vec<Column> = right_types
            .iter()
            .map(|&(tid, _)| Column::new(ColumnData::with_capacity(tid, BATCH_SIZE), tid))
            .collect();

        let old_left = std::mem::replace(&mut self.left_builders, new_left);
        let old_right = std::mem::replace(&mut self.right_builders, new_right);

        columns.extend(old_left);
        columns.extend(old_right);
        self.count = 0;
        DataBatch::new(columns)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extracts (TypeId, column_index) pairs from a schema for null column creation.
fn schema_types(schema: &[LogicalColumn]) -> Vec<(TypeId, usize)> {
    schema
        .iter()
        .enumerate()
        .map(|(i, c)| (c.type_id, i))
        .collect()
}

// ---------------------------------------------------------------------------
// NestedLoopJoinOperator
// ---------------------------------------------------------------------------

/// Nested loop join. Materializes the right side, then for each left row
/// scans all right rows and evaluates the join condition.
/// Uses JoinOutputBuffer for batched output.
pub struct NestedLoopJoinOperator {
    left: Box<dyn Operator>,
    right: Box<dyn Operator>,
    join_type: JoinType,
    condition: Option<BoundExpr>,
    input_schema: Vec<LogicalColumn>,
    left_types: Vec<(TypeId, usize)>,
    right_types: Vec<(TypeId, usize)>,
    right_batches: Option<Vec<DataBatch>>,
    left_batch: Option<DataBatch>,
    left_row: usize,
    right_batch_idx: usize,
    right_row: usize,
    left_matched: bool,
    right_matched: Vec<Vec<bool>>,
    finished: bool,
    emitting_unmatched_right: bool,
    unmatched_rb_idx: usize,
    unmatched_rr_idx: usize,
    output_buffer: Option<JoinOutputBuffer>,
}

impl NestedLoopJoinOperator {
    pub fn new(
        left: Box<dyn Operator>,
        right: Box<dyn Operator>,
        join_type: JoinType,
        condition: Option<BoundExpr>,
        left_schema: Vec<LogicalColumn>,
        right_schema: Vec<LogicalColumn>,
    ) -> Self {
        let left_types = schema_types(&left_schema);
        let right_types = schema_types(&right_schema);
        let mut input_schema = left_schema;
        input_schema.extend(right_schema);

        Self {
            left,
            right,
            join_type,
            condition,
            input_schema,
            left_types,
            right_types,
            right_batches: None,
            left_batch: None,
            left_row: 0,
            right_batch_idx: 0,
            right_row: 0,
            left_matched: false,
            right_matched: Vec::new(),
            finished: false,
            emitting_unmatched_right: false,
            unmatched_rb_idx: 0,
            unmatched_rr_idx: 0,
            output_buffer: None,
        }
    }
}

impl Operator for NestedLoopJoinOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            if self.finished {
                return Ok(None);
            }

            // Materialize right side on first call.
            if self.right_batches.is_none() {
                let mut batches = Vec::new();
                loop {
                    match self.right.next().await? {
                        Some(eb) => batches.push(eb.batch),
                        None => break,
                    }
                }
                let track = matches!(self.join_type, JoinType::Right | JoinType::Full);
                self.right_matched = if track {
                    batches.iter().map(|b| vec![false; b.num_rows]).collect()
                } else {
                    Vec::new()
                };
                self.right_batches = Some(batches);
                self.output_buffer =
                    Some(JoinOutputBuffer::new(&self.left_types, &self.right_types));
            }

            let buf = self.output_buffer.as_mut().unwrap();

            // Emit unmatched right rows for RIGHT/FULL join.
            if self.emitting_unmatched_right {
                let rbs = self.right_batches.as_ref().unwrap();
                while self.unmatched_rb_idx < rbs.len() {
                    let rb = &rbs[self.unmatched_rb_idx];
                    while self.unmatched_rr_idx < rb.num_rows {
                        let row = self.unmatched_rr_idx;
                        self.unmatched_rr_idx += 1;
                        if !self.right_matched[self.unmatched_rb_idx][row] {
                            buf.push_null_left_right(rb, row);
                            if buf.is_full() {
                                return Ok(Some(ExecutionBatch::new(
                                    buf.flush(&self.left_types, &self.right_types),
                                )));
                            }
                        }
                    }
                    self.unmatched_rb_idx += 1;
                    self.unmatched_rr_idx = 0;
                }
                self.finished = true;
                if !buf.is_empty() {
                    return Ok(Some(ExecutionBatch::new(
                        buf.flush(&self.left_types, &self.right_types),
                    )));
                }
                return Ok(None);
            }

            let right_batches = self.right_batches.as_ref().unwrap();

            loop {
                // Get next left batch if needed.
                if self.left_batch.is_none() {
                    match self.left.next().await? {
                        Some(eb) => {
                            self.left_batch = Some(eb.batch);
                            self.left_row = 0;
                            self.left_matched = false;
                        }
                        None => {
                            if matches!(self.join_type, JoinType::Right | JoinType::Full) {
                                self.emitting_unmatched_right = true;
                                if !buf.is_empty() {
                                    return Ok(Some(ExecutionBatch::new(
                                        buf.flush(&self.left_types, &self.right_types),
                                    )));
                                }
                                return self.next().await;
                            }
                            self.finished = true;
                            if !buf.is_empty() {
                                return Ok(Some(ExecutionBatch::new(
                                    buf.flush(&self.left_types, &self.right_types),
                                )));
                            }
                            return Ok(None);
                        }
                    }
                }

                let left_batch = self.left_batch.as_ref().unwrap();
                if self.left_row >= left_batch.num_rows {
                    self.left_batch = None;
                    continue;
                }

                // Scan right side for current left row.
                while self.right_batch_idx < right_batches.len() {
                    let rb = &right_batches[self.right_batch_idx];
                    while self.right_row < rb.num_rows {
                        let rr = self.right_row;
                        self.right_row += 1;

                        // For condition evaluation, build a single-row combined batch.
                        let matches_cond = if let Some(ref cond) = self.condition {
                            let combined = combine_rows_single(left_batch, self.left_row, rb, rr);
                            let mask = evaluate(cond, &combined, &self.input_schema)?;
                            !mask.is_null(0) && mask.get_bool(0)
                        } else {
                            true
                        };

                        if matches_cond {
                            self.left_matched = true;
                            if !self.right_matched.is_empty() {
                                self.right_matched[self.right_batch_idx][rr] = true;
                            }
                            buf.push_matched(left_batch, self.left_row, rb, rr);
                            if buf.is_full() {
                                return Ok(Some(ExecutionBatch::new(
                                    buf.flush(&self.left_types, &self.right_types),
                                )));
                            }
                        }
                    }
                    self.right_batch_idx += 1;
                    self.right_row = 0;
                }

                // Finished right side for this left row.
                if !self.left_matched && matches!(self.join_type, JoinType::Left | JoinType::Full) {
                    buf.push_left_null_right(left_batch, self.left_row);
                    if buf.is_full() {
                        self.left_row += 1;
                        self.left_matched = false;
                        self.right_batch_idx = 0;
                        self.right_row = 0;
                        return Ok(Some(ExecutionBatch::new(
                            buf.flush(&self.left_types, &self.right_types),
                        )));
                    }
                }

                self.left_row += 1;
                self.left_matched = false;
                self.right_batch_idx = 0;
                self.right_row = 0;
            }
        })
    }
}

/// Builds a single-row combined batch for condition evaluation.
fn combine_rows_single(
    left: &DataBatch,
    left_row: usize,
    right: &DataBatch,
    right_row: usize,
) -> DataBatch {
    let mut columns = Vec::with_capacity(left.num_columns() + right.num_columns());
    for col in &left.columns {
        columns.push(col.slice(left_row, 1));
    }
    for col in &right.columns {
        columns.push(col.slice(right_row, 1));
    }
    DataBatch::new(columns)
}

// ---------------------------------------------------------------------------
// HashJoinOperator
// ---------------------------------------------------------------------------

/// Hash join for equi-joins. Builds a hash table on the left (build) side,
/// then probes with the right (probe) side.
///
/// Build phase: collects all left batches, merges into a single contiguous
/// DataBatch, then builds a chained flat array with (row, next) entries.
/// Merging eliminates binary search during probe (row index = direct offset).
///
/// Probe phase: for each probe batch, collects all (build_row, probe_row)
/// match pairs into flat index arrays, then uses batch gather_from to
/// produce output columns. This eliminates per-row per-column dispatch.
pub struct HashJoinOperator {
    left: Option<Box<dyn Operator>>,
    right: Box<dyn Operator>,
    join_type: JoinType,
    left_keys: Vec<BoundExpr>,
    right_keys: Vec<BoundExpr>,
    remaining_condition: Option<BoundExpr>,
    left_schema: Vec<LogicalColumn>,
    right_schema: Vec<LogicalColumn>,
    input_schema: Vec<LogicalColumn>,
    left_types: Vec<(TypeId, usize)>,
    right_types: Vec<(TypeId, usize)>,
    /// Single merged build batch. All build rows are contiguous.
    build_batch: Option<DataBatch>,
    /// Chained flat array: (next, hash_hi32) per build row.
    /// The build row index equals the entry index (inserted in order).
    /// Only the upper 32 bits of the hash are stored for collision detection
    /// (lower bits are used for bucket selection, providing independent checks).
    build_entries: Vec<(u32, u32)>,
    /// Maps hash to head entry index in build_entries.
    build_index: FlatHashTable,
    /// Tracks which build rows matched (for LEFT/FULL joins).
    build_matched: Vec<bool>,
    total_build_rows: usize,
    /// True when build used hash_int (fused single-integer-key path).
    /// Probe must use the same hash function for matching.
    build_used_int_hash: bool,
    built: bool,
    /// Pre-resolved probe key column indices (None = expression, needs evaluate).
    probe_key_col_indices: Vec<Option<usize>>,
    /// Pending output batches from vectorized probe.
    output_queue: Vec<DataBatch>,
    output_queue_idx: usize,
    finished: bool,
    emitting_unmatched_build: bool,
    unmatched_cursor: usize,
    output_buffer: Option<JoinOutputBuffer>,
}

impl HashJoinOperator {
    pub fn new(
        left: Box<dyn Operator>,
        right: Box<dyn Operator>,
        join_type: JoinType,
        left_keys: Vec<BoundExpr>,
        right_keys: Vec<BoundExpr>,
        remaining_condition: Option<BoundExpr>,
        left_schema: Vec<LogicalColumn>,
        right_schema: Vec<LogicalColumn>,
    ) -> Self {
        let left_types = schema_types(&left_schema);
        let right_types = schema_types(&right_schema);
        let mut input_schema = left_schema.clone();
        input_schema.extend(right_schema.clone());

        Self {
            left: Some(left),
            right,
            join_type,
            left_keys,
            right_keys,
            remaining_condition,
            left_schema,
            right_schema,
            input_schema,
            left_types,
            right_types,
            build_batch: None,
            build_entries: Vec::new(),
            build_index: FlatHashTable::with_capacity(0),
            build_matched: Vec::new(),
            total_build_rows: 0,
            build_used_int_hash: false,
            built: false,
            probe_key_col_indices: Vec::new(),
            output_queue: Vec::new(),
            output_queue_idx: 0,
            finished: false,
            emitting_unmatched_build: false,
            unmatched_cursor: 0,
            output_buffer: None,
        }
    }

    async fn build_hash_table(&mut self) -> Result<()> {
        let mut left = self.left.take().unwrap();
        let track = matches!(self.join_type, JoinType::Left | JoinType::Full);

        // Phase 1: Collect all build batches.
        let mut all_batches: Vec<DataBatch> = Vec::new();
        let mut total_rows = 0usize;
        loop {
            match left.next().await? {
                Some(eb) => {
                    total_rows += eb.batch.num_rows;
                    all_batches.push(eb.batch);
                }
                None => break,
            }
        }
        self.total_build_rows = total_rows;

        if total_rows == 0 || all_batches.is_empty() {
            self.built = true;
            self.build_batch = None;
            self.output_buffer = Some(JoinOutputBuffer::new(&self.left_types, &self.right_types));
            return Ok(());
        }

        // Phase 2: Merge all batches into a single contiguous DataBatch.
        // Eliminates binary search (resolve_build_row) during probe.
        let merged = if all_batches.len() == 1 {
            all_batches.pop().unwrap()
        } else {
            let num_cols = all_batches[0].num_columns();
            let mut merged_columns = Vec::with_capacity(num_cols);
            for col_idx in 0..num_cols {
                let type_id = all_batches[0].columns[col_idx].type_id;
                let mut data = ColumnData::with_capacity(type_id, total_rows);
                let mut nulls = NullBitmap::empty();
                for batch in &all_batches {
                    data.extend_from(&batch.columns[col_idx].data);
                    nulls.extend_from(&batch.columns[col_idx].nulls);
                }
                merged_columns.push(Column::with_nulls(data, nulls, type_id));
            }
            DataBatch::new(merged_columns)
        };

        // Phase 3: Hash keys and build chain entries.
        // Resolve key column indices.
        let mut key_col_indices: Vec<Option<usize>> = Vec::with_capacity(self.left_keys.len());
        for k in &self.left_keys {
            if let BoundExpr::ColumnRef(cr) = k {
                let idx = resolve_column_index(cr.table_idx, cr.column_id, &self.left_schema)?;
                key_col_indices.push(Some(idx));
            } else {
                key_col_indices.push(None);
            }
        }

        // Build hash table.
        // For single integer keys without nulls, uses fused hash+insert with
        // group-prefetch (PF=16) to hide L3 latency on hash table bucket access.
        // Multi-key or non-integer keys fall back to hash_column_batch.
        self.build_entries.reserve(total_rows);
        self.build_index = FlatHashTable::with_capacity(total_rows);

        let mut fused = false;
        if key_col_indices.len() == 1 {
            if let Some(key_idx) = key_col_indices[0] {
                let col = &merged.columns[key_idx];
                if !col.nulls.has_nulls() {
                    const PF: usize = 16;
                    macro_rules! fuse_build {
                        ($v:expr) => {{
                            let n = $v.len();
                            let mut pf_buf = [0u64; PF];
                            let prime = PF.min(n);
                            for i in 0..prime {
                                let h = compute::hash_int($v[i] as u64);
                                pf_buf[i] = h;
                                self.build_index.prefetch(h);
                            }
                            for row in 0..n {
                                let hash = pf_buf[row % PF];
                                let ahead = row + PF;
                                if ahead < n {
                                    let h = compute::hash_int($v[ahead] as u64);
                                    pf_buf[ahead % PF] = h;
                                    self.build_index.prefetch(h);
                                }
                                let prev = self.build_index.insert(hash, row as u32);
                                self.build_entries.push((prev, (hash >> 32) as u32));
                            }
                            fused = true;
                            self.build_used_int_hash = true;
                        }};
                    }
                    match &col.data {
                        ColumnData::Int64(v) => fuse_build!(v),
                        ColumnData::Int32(v) => fuse_build!(v),
                        ColumnData::Int16(v) => fuse_build!(v),
                        ColumnData::Int8(v) => fuse_build!(v),
                        ColumnData::UInt64(v) => fuse_build!(v),
                        ColumnData::UInt32(v) => fuse_build!(v),
                        ColumnData::UInt16(v) => fuse_build!(v),
                        ColumnData::UInt8(v) => fuse_build!(v),
                        _ => {}
                    }
                }
            }
        }

        // Fallback: multi-key or non-integer key types.
        if !fused {
            let mut owned_key_cols: Vec<Column> = Vec::new();
            for (ki, src) in key_col_indices.iter().enumerate() {
                if src.is_none() {
                    let col = evaluate(&self.left_keys[ki], &merged, &self.left_schema)?;
                    owned_key_cols.push(col);
                }
            }
            let mut owned_idx = 0;
            let key_refs: Vec<&Column> = key_col_indices
                .iter()
                .map(|src| match src {
                    Some(idx) => &merged.columns[*idx],
                    None => {
                        let col = &owned_key_cols[owned_idx];
                        owned_idx += 1;
                        col
                    }
                })
                .collect();
            let all_hashes = compute::hash_column_batch(&key_refs, total_rows);

            for (row, &hash) in all_hashes.iter().enumerate() {
                let prev = self.build_index.insert(hash, row as u32);
                self.build_entries.push((prev, (hash >> 32) as u32));
            }
        }

        self.build_batch = Some(merged);

        if track {
            self.build_matched = vec![false; self.total_build_rows];
        }

        // Pre-resolve probe key indices to avoid per-batch resolution.
        for k in &self.right_keys {
            if let BoundExpr::ColumnRef(cr) = k {
                let idx = resolve_column_index(cr.table_idx, cr.column_id, &self.right_schema)?;
                self.probe_key_col_indices.push(Some(idx));
            } else {
                self.probe_key_col_indices.push(None);
            }
        }

        self.built = true;
        self.output_buffer = Some(JoinOutputBuffer::new(&self.left_types, &self.right_types));
        Ok(())
    }

    /// Vectorized probe: collects all (build_row, probe_row) match pairs,
    /// then gathers output columns in bulk. Eliminates per-row per-column
    /// dispatch overhead.
    fn probe_batch_vectorized(
        &mut self,
        probe_batch: &DataBatch,
        probe_hashes: &[u64],
    ) -> Vec<DataBatch> {
        let track_right = matches!(self.join_type, JoinType::Right | JoinType::Full);
        let build = self.build_batch.as_ref().unwrap();

        // Phase 1: Collect all match pairs as flat index arrays.
        let mut build_idx: Vec<u32> = Vec::new();
        let mut probe_idx: Vec<u32> = Vec::new();
        let mut unmatched_probe: Vec<u32> = Vec::new();

        for probe_row in 0..probe_batch.num_rows {
            let hash = probe_hashes[probe_row];
            let mut cursor = self.build_index.get(hash);
            let hash_hi32 = (hash >> 32) as u32;
            let mut matched = false;
            while cursor != u32::MAX {
                let (next, stored_hi32) = self.build_entries[cursor as usize];
                let build_row = cursor;
                cursor = next;
                if stored_hi32 != hash_hi32 {
                    continue;
                }
                build_idx.push(build_row);
                probe_idx.push(probe_row as u32);
                matched = true;
                if !self.build_matched.is_empty() {
                    self.build_matched[build_row as usize] = true;
                }
            }
            if !matched && track_right {
                unmatched_probe.push(probe_row as u32);
            }
        }

        // Phase 2: Gather output in BATCH_SIZE chunks.
        let mut results = Vec::new();
        let total_matched = build_idx.len();
        let mut offset = 0;

        while offset < total_matched {
            let end = (offset + BATCH_SIZE).min(total_matched);
            let bi = &build_idx[offset..end];
            let pi = &probe_idx[offset..end];
            let n = end - offset;

            let mut cols = Vec::with_capacity(build.num_columns() + probe_batch.num_columns());

            for col in &build.columns {
                let mut d = ColumnData::with_capacity(col.type_id, n);
                d.gather_from(&col.data, bi);
                let mut nulls = NullBitmap::empty();
                nulls.gather_from(&col.nulls, bi);
                cols.push(Column::with_nulls(d, nulls, col.type_id));
            }
            for col in &probe_batch.columns {
                let mut d = ColumnData::with_capacity(col.type_id, n);
                d.gather_from(&col.data, pi);
                let mut nulls = NullBitmap::empty();
                nulls.gather_from(&col.nulls, pi);
                cols.push(Column::with_nulls(d, nulls, col.type_id));
            }

            results.push(DataBatch::new(cols));
            offset = end;
        }

        // Phase 3: Emit unmatched probe rows for RIGHT/FULL join.
        if !unmatched_probe.is_empty() {
            let buf = self.output_buffer.as_mut().unwrap();
            for &pr in &unmatched_probe {
                buf.push_null_left_right(probe_batch, pr as usize);
                if buf.is_full() {
                    results.push(buf.flush(&self.left_types, &self.right_types));
                }
            }
        }

        results
    }
}

impl Operator for HashJoinOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            // Drain queued output batches first, before any state checks.
            if self.output_queue_idx < self.output_queue.len() {
                let batch = std::mem::replace(
                    &mut self.output_queue[self.output_queue_idx],
                    DataBatch::new(Vec::new()),
                );
                self.output_queue_idx += 1;
                if self.output_queue_idx >= self.output_queue.len() {
                    self.output_queue.clear();
                    self.output_queue_idx = 0;
                }
                return Ok(Some(ExecutionBatch::new(batch)));
            }

            if self.finished {
                return Ok(None);
            }

            if !self.built {
                self.build_hash_table().await?;
            }

            // Emit unmatched build rows for LEFT/FULL join.
            if self.emitting_unmatched_build {
                let buf = self.output_buffer.as_mut().unwrap();
                let build = self.build_batch.as_ref().unwrap();
                while self.unmatched_cursor < self.total_build_rows {
                    let row = self.unmatched_cursor;
                    self.unmatched_cursor += 1;
                    if !self.build_matched[row] {
                        buf.push_left_null_right(build, row);
                        if buf.is_full() {
                            return Ok(Some(ExecutionBatch::new(
                                buf.flush(&self.left_types, &self.right_types),
                            )));
                        }
                    }
                }
                self.finished = true;
                if !buf.is_empty() {
                    return Ok(Some(ExecutionBatch::new(
                        buf.flush(&self.left_types, &self.right_types),
                    )));
                }
                return Ok(None);
            }

            // Materialize entire probe side, then process all at once.
            // This eliminates per-batch overhead (key resolution, hashing,
            // allocation) that dominates when probe batches are small.
            let mut probe_batches: Vec<DataBatch> = Vec::new();
            let mut total_probe_rows = 0usize;
            loop {
                match self.right.next().await? {
                    Some(eb) => {
                        total_probe_rows += eb.batch.num_rows;
                        probe_batches.push(eb.batch);
                    }
                    None => break,
                }
            }

            if total_probe_rows == 0 || self.build_batch.is_none() {
                // No probe rows or no build rows. Handle unmatched build.
                if matches!(self.join_type, JoinType::Left | JoinType::Full)
                    && self.build_batch.is_some()
                {
                    self.emitting_unmatched_build = true;
                    return self.next().await;
                }
                self.finished = true;
                return Ok(None);
            }

            // Merge probe batches into a single batch.
            let merged_probe = if probe_batches.len() == 1 {
                probe_batches.pop().unwrap()
            } else {
                let num_cols = probe_batches[0].num_columns();
                let mut merged_cols = Vec::with_capacity(num_cols);
                for col_idx in 0..num_cols {
                    let type_id = probe_batches[0].columns[col_idx].type_id;
                    let mut data = ColumnData::with_capacity(type_id, total_probe_rows);
                    let mut nulls = NullBitmap::empty();
                    for batch in &probe_batches {
                        data.extend_from(&batch.columns[col_idx].data);
                        nulls.extend_from(&batch.columns[col_idx].nulls);
                    }
                    merged_cols.push(Column::with_nulls(data, nulls, type_id));
                }
                DataBatch::new(merged_cols)
            };
            drop(probe_batches);
            // Determine if we can use the fused probe path: single ColumnRef
            // integer key, no nulls, no remaining condition. This computes
            // hashes inline and probes the hash table in one pass with
            // group-prefetch to hide L3 latency, eliminating the separate
            // hash buffer allocation and extra passes.
            let use_fused = self.remaining_condition.is_none()
                && self.probe_key_col_indices.len() == 1
                && self.probe_key_col_indices[0].is_some();

            let fused_key_idx = if use_fused {
                self.probe_key_col_indices[0]
            } else {
                None
            };

            let fused_col_no_nulls = fused_key_idx
                .map(|ki| !merged_probe.columns[ki].nulls.has_nulls())
                .unwrap_or(false);

            if fused_col_no_nulls {
                let key_idx = fused_key_idx.unwrap();
                let track_right = matches!(self.join_type, JoinType::Right | JoinType::Full);
                let track_build = !self.build_matched.is_empty();

                let mut build_idx: Vec<u32> = Vec::with_capacity(total_probe_rows);
                let mut probe_idx: Vec<u32> = Vec::with_capacity(total_probe_rows);
                let mut unmatched_probe: Vec<u32> =
                    if track_right { Vec::new() } else { Vec::new() };

                // Fused hash + probe with group-prefetch.
                // Prefetch distance of 16 hides L3 latency for bucket lookups.
                const PF: usize = 16;

                macro_rules! fused_probe_prefetch {
                    ($v:expr) => {{
                        let n = $v.len();
                        let mut pf_buf = [0u64; PF];
                        let prime = PF.min(n);
                        for i in 0..prime {
                            pf_buf[i] = compute::hash_int($v[i] as u64);
                            self.build_index.prefetch(pf_buf[i]);
                        }

                        for probe_row in 0..n {
                            let hash = pf_buf[probe_row % PF];

                            let ahead = probe_row + PF;
                            if ahead < n {
                                let h = compute::hash_int($v[ahead] as u64);
                                pf_buf[ahead % PF] = h;
                                self.build_index.prefetch(h);
                            }

                            let mut cursor = self.build_index.get(hash);
                            let hash_hi32 = (hash >> 32) as u32;
                            let mut matched = false;
                            while cursor != u32::MAX {
                                let (next, stored_hi32) = self.build_entries[cursor as usize];
                                let build_row = cursor;
                                cursor = next;
                                if stored_hi32 != hash_hi32 {
                                    continue;
                                }
                                build_idx.push(build_row);
                                probe_idx.push(probe_row as u32);
                                matched = true;
                                if track_build {
                                    self.build_matched[build_row as usize] = true;
                                }
                            }
                            if !matched && track_right {
                                unmatched_probe.push(probe_row as u32);
                            }
                        }
                    }};
                }

                let col = &merged_probe.columns[key_idx];
                match &col.data {
                    ColumnData::Int64(v) => fused_probe_prefetch!(v),
                    ColumnData::Int32(v) => fused_probe_prefetch!(v),
                    ColumnData::Int16(v) => fused_probe_prefetch!(v),
                    ColumnData::Int8(v) => fused_probe_prefetch!(v),
                    ColumnData::UInt64(v) => fused_probe_prefetch!(v),
                    ColumnData::UInt32(v) => fused_probe_prefetch!(v),
                    ColumnData::UInt16(v) => fused_probe_prefetch!(v),
                    ColumnData::UInt8(v) => fused_probe_prefetch!(v),
                    _ => {
                        // Non-integer fused path without prefetch (strings, etc).
                        let probe_hashes = compute::hash_column_batch(
                            &[&merged_probe.columns[key_idx]],
                            total_probe_rows,
                        );
                        for probe_row in 0..total_probe_rows {
                            let hash = probe_hashes[probe_row];
                            let mut cursor = self.build_index.get(hash);
                            let hash_hi32 = (hash >> 32) as u32;
                            let mut matched = false;
                            while cursor != u32::MAX {
                                let (next, stored_hi32) = self.build_entries[cursor as usize];
                                let build_row = cursor;
                                cursor = next;
                                if stored_hi32 != hash_hi32 {
                                    continue;
                                }
                                build_idx.push(build_row);
                                probe_idx.push(probe_row as u32);
                                matched = true;
                                if track_build {
                                    self.build_matched[build_row as usize] = true;
                                }
                            }
                            if !matched && track_right {
                                unmatched_probe.push(probe_row as u32);
                            }
                        }
                    }
                }

                // Gather output into large batches (avoids 49x per-batch overhead).
                let build = self.build_batch.as_ref().unwrap();
                let total_matched = build_idx.len();
                let mut offset = 0;
                while offset < total_matched {
                    let end = (offset + BATCH_SIZE).min(total_matched);
                    let bi = &build_idx[offset..end];
                    let pi = &probe_idx[offset..end];
                    let n = end - offset;

                    let mut cols =
                        Vec::with_capacity(build.num_columns() + merged_probe.num_columns());
                    for col in &build.columns {
                        let mut d = ColumnData::with_capacity(col.type_id, n);
                        d.gather_from(&col.data, bi);
                        if col.nulls.has_nulls() {
                            let mut nulls = NullBitmap::empty();
                            nulls.gather_from(&col.nulls, bi);
                            cols.push(Column::with_nulls(d, nulls, col.type_id));
                        } else {
                            cols.push(Column::new(d, col.type_id));
                        }
                    }
                    for col in &merged_probe.columns {
                        let mut d = ColumnData::with_capacity(col.type_id, n);
                        d.gather_from(&col.data, pi);
                        if col.nulls.has_nulls() {
                            let mut nulls = NullBitmap::empty();
                            nulls.gather_from(&col.nulls, pi);
                            cols.push(Column::with_nulls(d, nulls, col.type_id));
                        } else {
                            cols.push(Column::new(d, col.type_id));
                        }
                    }
                    self.output_queue.push(DataBatch::new(cols));
                    offset = end;
                }

                // Emit unmatched probe rows for RIGHT/FULL join.
                if !unmatched_probe.is_empty() {
                    let buf = self.output_buffer.as_mut().unwrap();
                    for &pr in &unmatched_probe {
                        buf.push_null_left_right(&merged_probe, pr as usize);
                        if buf.is_full() {
                            self.output_queue
                                .push(buf.flush(&self.left_types, &self.right_types));
                        }
                    }
                }
            } else {
                // Generic path: hash all probe keys, then probe.
                let mut owned_probe_keys: Vec<Column> = Vec::new();
                for (ki, src) in self.probe_key_col_indices.iter().enumerate() {
                    if src.is_none() {
                        let col =
                            evaluate(&self.right_keys[ki], &merged_probe, &self.right_schema)?;
                        owned_probe_keys.push(col);
                    }
                }
                let mut owned_idx = 0;
                let key_refs: Vec<&Column> = self
                    .probe_key_col_indices
                    .iter()
                    .map(|src| match src {
                        Some(idx) => &merged_probe.columns[*idx],
                        None => {
                            let col = &owned_probe_keys[owned_idx];
                            owned_idx += 1;
                            col
                        }
                    })
                    .collect();
                let probe_hashes = if self.build_used_int_hash {
                    // Build used hash_int for single integer key. Compute
                    // matching hashes from the single probe key column.
                    let col = key_refs[0];
                    let mut hashes = Vec::with_capacity(total_probe_rows);
                    macro_rules! hash_int_col {
                        ($v:expr) => {
                            for val in $v.iter() {
                                hashes.push(compute::hash_int(*val as u64));
                            }
                        };
                    }
                    match &col.data {
                        ColumnData::Int64(v) => hash_int_col!(v),
                        ColumnData::Int32(v) => hash_int_col!(v),
                        ColumnData::Int16(v) => hash_int_col!(v),
                        ColumnData::Int8(v) => hash_int_col!(v),
                        ColumnData::UInt64(v) => hash_int_col!(v),
                        ColumnData::UInt32(v) => hash_int_col!(v),
                        ColumnData::UInt16(v) => hash_int_col!(v),
                        ColumnData::UInt8(v) => hash_int_col!(v),
                        _ => {
                            hashes = compute::hash_column_batch(&key_refs, total_probe_rows);
                        }
                    }
                    hashes
                } else {
                    compute::hash_column_batch(&key_refs, total_probe_rows)
                };

                if self.remaining_condition.is_some() {
                    let buf = self.output_buffer.as_mut().unwrap();
                    let build = self.build_batch.as_ref().unwrap();
                    for probe_row in 0..total_probe_rows {
                        let hash = probe_hashes[probe_row];
                        let mut cursor = self.build_index.get(hash);
                        let hash_hi32 = (hash >> 32) as u32;
                        let mut matched = false;
                        while cursor != u32::MAX {
                            let (next, stored_hi32) = self.build_entries[cursor as usize];
                            let build_row = cursor;
                            cursor = next;
                            if stored_hi32 != hash_hi32 {
                                continue;
                            }
                            let combined = combine_rows_single(
                                build,
                                build_row as usize,
                                &merged_probe,
                                probe_row,
                            );
                            let mask = evaluate(
                                self.remaining_condition.as_ref().unwrap(),
                                &combined,
                                &self.input_schema,
                            )?;
                            if !mask.is_null(0) && mask.get_bool(0) {
                                matched = true;
                                if !self.build_matched.is_empty() {
                                    self.build_matched[build_row as usize] = true;
                                }
                                buf.push_matched(
                                    build,
                                    build_row as usize,
                                    &merged_probe,
                                    probe_row,
                                );
                                if buf.is_full() {
                                    self.output_queue
                                        .push(buf.flush(&self.left_types, &self.right_types));
                                }
                            }
                        }
                        if !matched && matches!(self.join_type, JoinType::Right | JoinType::Full) {
                            buf.push_null_left_right(&merged_probe, probe_row);
                            if buf.is_full() {
                                self.output_queue
                                    .push(buf.flush(&self.left_types, &self.right_types));
                            }
                        }
                    }
                } else {
                    let batches = self.probe_batch_vectorized(&merged_probe, &probe_hashes);
                    self.output_queue.extend(batches);
                }
            }

            // Flush remaining buffered rows.
            let buf = self.output_buffer.as_mut().unwrap();
            if !buf.is_empty() {
                self.output_queue
                    .push(buf.flush(&self.left_types, &self.right_types));
            }

            if matches!(self.join_type, JoinType::Left | JoinType::Full) {
                self.emitting_unmatched_build = true;
            } else {
                self.finished = true;
            }

            // Return first queued batch.
            if !self.output_queue.is_empty() {
                self.output_queue_idx = 1;
                let batch =
                    std::mem::replace(&mut self.output_queue[0], DataBatch::new(Vec::new()));
                if self.output_queue_idx >= self.output_queue.len() {
                    self.output_queue.clear();
                    self.output_queue_idx = 0;
                }
                return Ok(Some(ExecutionBatch::new(batch)));
            }

            if self.emitting_unmatched_build {
                return self.next().await;
            }
            Ok(None)
        })
    }
}

// ---------------------------------------------------------------------------
// MergeJoinOperator
// ---------------------------------------------------------------------------

/// Sort-merge join. Currently delegates to HashJoinOperator.
pub struct MergeJoinOperator {
    inner: HashJoinOperator,
}

impl MergeJoinOperator {
    pub fn new(
        left: Box<dyn Operator>,
        right: Box<dyn Operator>,
        join_type: JoinType,
        left_keys: Vec<BoundExpr>,
        right_keys: Vec<BoundExpr>,
        left_schema: Vec<LogicalColumn>,
        right_schema: Vec<LogicalColumn>,
    ) -> Self {
        Self {
            inner: HashJoinOperator::new(
                left,
                right,
                join_type,
                left_keys,
                right_keys,
                None,
                left_schema,
                right_schema,
            ),
        }
    }
}

impl Operator for MergeJoinOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        self.inner.next()
    }
}

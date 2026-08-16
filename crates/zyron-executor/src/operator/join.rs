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

use std::borrow::Cow;
use std::sync::Arc;

use zyron_common::{Result, TypeId};
use zyron_parser::ast::JoinType;
use zyron_planner::binder::BoundExpr;
use zyron_planner::logical::LogicalColumn;

use crate::batch::{BATCH_SIZE, DataBatch};
use crate::column::{Column, ColumnData, NullBitmap};
use crate::compute::{self, FlatHashTable};
use crate::context::ExecutionContext;
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
    /// Set when the ON condition contains a subquery. The condition is then
    /// evaluated per joined row through this prepared predicate (which runs any
    /// correlated subquery against that row) instead of the synchronous
    /// evaluator. `condition` is None in this mode.
    correlated: Option<(
        Arc<ExecutionContext>,
        crate::correlated::CorrelatedPredicate,
    )>,
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
            correlated: None,
        }
    }

    /// Returns the join's input schema (left columns followed by right),
    /// used to prepare a correlated ON condition before attaching it.
    pub fn input_schema(&self) -> &[LogicalColumn] {
        &self.input_schema
    }

    /// Evaluates the ON condition through a prepared correlated predicate so a
    /// subquery in the condition runs per joined row. Clears the synchronous
    /// condition; the outer-join row-matching and NULL-extension are unchanged.
    pub fn with_correlated_condition(
        mut self,
        ctx: Arc<ExecutionContext>,
        predicate: crate::correlated::CorrelatedPredicate,
    ) -> Self {
        self.condition = None;
        self.correlated = Some((ctx, predicate));
        self
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
                        let matches_cond = if let Some((cctx, pred)) = &self.correlated {
                            let combined = combine_rows_single(left_batch, self.left_row, rb, rr);
                            let mask = pred.eval_mask(cctx, &combined).await?;
                            mask.first().copied().unwrap_or(false)
                        } else if let Some(ref cond) = self.condition {
                            let combined = combine_rows_single(left_batch, self.left_row, rb, rr);
                            let mask = evaluate(cond, &combined, &self.input_schema, &[])?;
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

/// Merges batches into one contiguous DataBatch, None when no rows exist.
/// A single batch moves through without copying.
fn merge_batches(mut batches: Vec<DataBatch>, total_rows: usize) -> Option<DataBatch> {
    if total_rows == 0 || batches.is_empty() {
        return None;
    }
    if batches.len() == 1 {
        return batches.pop();
    }
    let num_cols = batches[0].num_columns();
    let mut merged_columns = Vec::with_capacity(num_cols);
    for col_idx in 0..num_cols {
        let template = &batches[0].columns[col_idx];
        let type_id = template.type_id;
        let fractional_digits = template.fractional_digits;
        let mut data = ColumnData::with_capacity(type_id, total_rows);
        let mut nulls = NullBitmap::empty();
        for batch in &batches {
            data.extend_from(&batch.columns[col_idx].data);
            nulls.extend_from(&batch.columns[col_idx].nulls);
        }
        merged_columns.push(Column::with_nulls_ts(data, nulls, type_id, fractional_digits));
    }
    Some(DataBatch::new(merged_columns))
}

/// Gathers the given rows of each column into fresh columns appended to
/// out. Columns without nulls skip the bitmap gather entirely.
fn gather_columns_into(cols: &[Column], idx: &[u32], n: usize, out: &mut Vec<Column>) {
    for col in cols {
        let mut d = ColumnData::with_capacity(col.type_id, n);
        d.gather_from(&col.data, idx);
        if col.nulls.has_nulls() {
            let mut nulls = NullBitmap::empty();
            nulls.gather_from(&col.nulls, idx);
            out.push(Column::with_nulls_ts(
                d,
                nulls,
                col.type_id,
                col.fractional_digits,
            ));
        } else {
            out.push(Column::new_ts(d, col.type_id, col.fractional_digits));
        }
    }
}

// ---------------------------------------------------------------------------
// HashJoinOperator
// ---------------------------------------------------------------------------

/// Hash join for equi-joins. Materializes both inputs, builds a hash
/// table on whichever side produced fewer rows, and probes with the other.
///
/// Build phase: collects all batches from both inputs, chooses the build
/// side by actual row count, merges it into a single contiguous DataBatch,
/// then builds a chained flat array with (row, next) entries. Merging
/// eliminates binary search during probe (row index = direct offset).
/// When the right input is smaller the sides swap internally, mirroring
/// join direction for the tracking logic while output columns are still
/// emitted left-then-right, so the swap is invisible outside.
///
/// Probe phase: for each probe batch, collects all (build_row, probe_row)
/// match pairs into flat index arrays, then uses batch gather_from to
/// produce output columns. This eliminates per-row per-column dispatch.
/// Compares a build row's key values against a probe row's key values.
/// Returns true only when every key column is equal and non-NULL on both
/// sides so a 64-bit hash collision cannot false-join and NULL never matches.
#[inline]
fn keys_match_columns(
    build_keys: &[Column],
    probe_keys: &[Cow<'_, Column>],
    build_row: usize,
    probe_row: usize,
) -> bool {
    for (bk, pk) in build_keys.iter().zip(probe_keys.iter()) {
        if !compute::cross_column_value_equal(bk, build_row, pk.as_ref(), probe_row) {
            return false;
        }
    }
    true
}

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
    /// Materialized build key columns, one per join key, indexed by build row.
    /// Used to compare actual key values after the hi32 hash match so a 64-bit
    /// hash collision cannot false-join.
    build_key_columns: Vec<Column>,
    /// Tracks which build rows matched (for LEFT/FULL joins).
    build_matched: Vec<bool>,
    total_build_rows: usize,
    /// True when build used hash_int (fused single-integer-key path).
    /// Probe must use the same hash function for matching.
    build_used_int_hash: bool,
    /// Per-key decimal alignment target. When either side of a key is a
    /// decimal, both sides materialize as Int128 on this scale before
    /// hashing and comparing, so equal numbers share bytes. None for keys
    /// with no decimal side
    key_align_scales: Vec<Option<u8>>,
    /// True when the original right input is the build side because it
    /// materialized fewer rows. Tracking logic runs on internal_join and
    /// output assembly restores left-then-right column order
    swapped: bool,
    /// join_type with LEFT and RIGHT mirrored when swapped, so build-side
    /// and probe-side outer tracking stays direction-agnostic
    internal_join: JoinType,
    /// Probe side batches drained during the build phase, probed one at a
    /// time. They are deliberately not merged, so a join never has to hold
    /// the whole probe side contiguously and never copies it
    probe_batches_pending: Vec<DataBatch>,
    /// Output columns accumulated across probe batches in external
    /// left-then-right order, flushed once they reach BATCH_SIZE rows so
    /// per-batch probing does not fragment the output
    pending_out: Vec<Column>,
    pending_out_rows: usize,
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
            build_key_columns: Vec::new(),
            build_matched: Vec::new(),
            total_build_rows: 0,
            build_used_int_hash: false,
            key_align_scales: Vec::new(),
            swapped: false,
            internal_join: join_type,
            probe_batches_pending: Vec::new(),
            pending_out: Vec::new(),
            pending_out_rows: 0,
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

        // Phase 1: Drain both inputs. Both sides always materialize in
        // full before probing, so the build side is chosen by actual row
        // count instead of plan-time estimates, because hashing the smaller
        // side shrinks the table every probe row walks.
        let mut left_batches: Vec<DataBatch> = Vec::new();
        let mut left_rows = 0usize;
        loop {
            match left.next().await? {
                Some(eb) => {
                    left_rows += eb.batch.num_rows;
                    left_batches.push(eb.batch);
                }
                None => break,
            }
        }
        let mut right_batches: Vec<DataBatch> = Vec::new();
        let mut right_rows = 0usize;
        loop {
            match self.right.next().await? {
                Some(eb) => {
                    right_rows += eb.batch.num_rows;
                    right_batches.push(eb.batch);
                }
                None => break,
            }
        }

        self.swapped = right_rows < left_rows;
        self.internal_join = if self.swapped {
            match self.join_type {
                JoinType::Left => JoinType::Right,
                JoinType::Right => JoinType::Left,
                other => other,
            }
        } else {
            self.join_type
        };
        let (build_batches, build_rows, probe_batches, probe_rows) = if self.swapped {
            (right_batches, right_rows, left_batches, left_rows)
        } else {
            (left_batches, left_rows, right_batches, right_rows)
        };
        self.total_build_rows = build_rows;
        let _ = probe_rows;
        self.probe_batches_pending = probe_batches;
        self.output_buffer = Some(JoinOutputBuffer::new(&self.left_types, &self.right_types));

        let track = matches!(self.internal_join, JoinType::Left | JoinType::Full);

        if build_rows == 0 {
            self.built = true;
            self.build_batch = None;
            return Ok(());
        }

        // Phase 2: Merge all build batches into a single contiguous
        // DataBatch. Eliminates binary search (resolve_build_row) during
        // probe. build_rows > 0 guarantees a merged batch exists.
        let merged = match merge_batches(build_batches, build_rows) {
            Some(b) => b,
            None => {
                return Err(zyron_common::ZyronError::ExecutionError(
                    "hash join build side lost its batches before merging".to_string(),
                ));
            }
        };

        // Key expressions and schema of whichever input is the build side.
        let build_key_exprs: &[BoundExpr] = if self.swapped {
            &self.right_keys
        } else {
            &self.left_keys
        };
        let build_schema: &[LogicalColumn] = if self.swapped {
            &self.right_schema
        } else {
            &self.left_schema
        };
        let probe_key_exprs: &[BoundExpr] = if self.swapped {
            &self.left_keys
        } else {
            &self.right_keys
        };

        // Phase 3: Hash keys and build chain entries.
        // Resolve key column indices.
        let mut key_col_indices: Vec<Option<usize>> = Vec::with_capacity(build_key_exprs.len());
        for k in build_key_exprs {
            if let BoundExpr::ColumnRef(cr) = k {
                let idx = resolve_column_index(cr.table_idx, cr.column_id, build_schema)?;
                key_col_indices.push(Some(idx));
            } else {
                key_col_indices.push(None);
            }
        }

        // Materialize the build key columns once so probe-time can compare
        // actual key values after the hi32 hash match, and so NULL keys can be
        // excluded from the build side.
        let mut build_key_columns: Vec<Column> = Vec::with_capacity(key_col_indices.len());
        for (ki, src) in key_col_indices.iter().enumerate() {
            match src {
                Some(idx) => build_key_columns.push(merged.columns[*idx].clone()),
                None => {
                    let col = evaluate(&build_key_exprs[ki], &merged, build_schema, &[])?;
                    build_key_columns.push(col);
                }
            }
        }

        // When either side of a key is a decimal, both sides hash and
        // compare as Int128 on the wider scale so equal numbers share
        // bytes. The build side's scale comes from the column just
        // materialized, the probe side's from its bound expression
        self.key_align_scales = build_key_columns
            .iter()
            .zip(probe_key_exprs.iter())
            .map(|(bc, rk)| {
                let build_dec = bc.type_id == TypeId::Decimal;
                let probe_dec = rk.type_id() == TypeId::Decimal;
                if !build_dec && !probe_dec {
                    return None;
                }
                let bs = if build_dec {
                    bc.fractional_digits.unwrap_or(0)
                } else {
                    0
                };
                let ps = if probe_dec {
                    rk.fractional_digits().unwrap_or(0)
                } else {
                    0
                };
                Some(bs.max(ps))
            })
            .collect();
        for (col, target) in build_key_columns.iter_mut().zip(&self.key_align_scales) {
            if let Some(scale) = target {
                *col = compute::cast_column_to_decimal(col, *scale)?;
            }
        }

        // A build row is excluded when any key column is NULL so NULL never
        // joins NULL. Rows with no NULL key are insertable.
        let any_key_has_nulls = build_key_columns.iter().any(|c| c.nulls.has_nulls());

        // Build hash table.
        // For single integer keys without nulls, uses fused hash+insert with
        // group-prefetch (PF=16) to hide L3 latency on hash table bucket access.
        // Multi-key or non-integer keys fall back to hash_column_batch.
        self.build_entries.reserve(build_rows);
        self.build_index = FlatHashTable::with_capacity(build_rows);

        let mut fused = false;
        let key_needs_align = self.key_align_scales.iter().any(|s| s.is_some());
        if key_col_indices.len() == 1 && !any_key_has_nulls && !key_needs_align {
            if let Some(key_idx) = key_col_indices[0] {
                let col = &merged.columns[key_idx];
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

        // Fallback: multi-key, non-integer key types, or NULL-bearing keys.
        if !fused {
            let key_refs: Vec<&Column> = build_key_columns.iter().collect();
            let all_hashes = compute::hash_column_batch(&key_refs, build_rows);

            for row in 0..build_rows {
                // Skip rows with any NULL key so NULL never matches NULL.
                if any_key_has_nulls && build_key_columns.iter().any(|c| c.is_null(row)) {
                    // Push a placeholder entry so build_entries stays row-indexed,
                    // but do not link it into any hash chain.
                    self.build_entries.push((u32::MAX, 0));
                    continue;
                }
                let hash = all_hashes[row];
                let prev = self.build_index.insert(hash, row as u32);
                self.build_entries.push((prev, (hash >> 32) as u32));
            }
        }

        self.build_key_columns = build_key_columns;
        self.build_batch = Some(merged);

        if track {
            self.build_matched = vec![false; self.total_build_rows];
        }

        // Pre-resolve probe key indices to avoid per-batch resolution.
        let probe_schema: &[LogicalColumn] = if self.swapped {
            &self.left_schema
        } else {
            &self.right_schema
        };
        let mut probe_key_col_indices = Vec::with_capacity(probe_key_exprs.len());
        for k in probe_key_exprs {
            if let BoundExpr::ColumnRef(cr) = k {
                let idx = resolve_column_index(cr.table_idx, cr.column_id, probe_schema)?;
                probe_key_col_indices.push(Some(idx));
            } else {
                probe_key_col_indices.push(None);
            }
        }
        self.probe_key_col_indices = probe_key_col_indices;

        self.built = true;
        Ok(())
    }

    /// Materializes the probe key columns for a probe batch in the same order
    /// as build_key_columns so values can be compared after the hi32 match.
    /// A plain column-reference key borrows the batch's column outright. Only
    /// an expression key or a decimal alignment produces an owned column, so
    /// the common case copies nothing however wide the probe side is.
    /// Keys with a decimal side convert to the same Int128 scale the build
    /// side hashed on
    fn materialize_probe_keys<'a>(
        &self,
        probe_batch: &'a DataBatch,
    ) -> Result<Vec<Cow<'a, Column>>> {
        let (probe_key_exprs, probe_schema) = if self.swapped {
            (&self.left_keys, &self.left_schema)
        } else {
            (&self.right_keys, &self.right_schema)
        };
        let mut cols = Vec::with_capacity(self.probe_key_col_indices.len());
        for (ki, src) in self.probe_key_col_indices.iter().enumerate() {
            let col: Cow<'a, Column> = match src {
                Some(idx) => Cow::Borrowed(&probe_batch.columns[*idx]),
                None => Cow::Owned(evaluate(
                    &probe_key_exprs[ki],
                    probe_batch,
                    probe_schema,
                    &[],
                )?),
            };
            let col = match self.key_align_scales.get(ki).copied().flatten() {
                Some(scale) => Cow::Owned(compute::cast_column_to_decimal(col.as_ref(), scale)?),
                None => col,
            };
            cols.push(col);
        }
        Ok(cols)
    }

    /// Appends matched rows to the pending output, emitting a batch every
    /// time BATCH_SIZE rows accumulate. Matches are gathered in bulk per
    /// column, so probing one batch at a time costs no more per row than
    /// probing a merged probe side did, and the emitted batches stay the
    /// same size regardless of how the probe input was chunked.
    fn append_matches(&mut self, build_rows: &[u32], probe_batch: &DataBatch, probe_rows: &[u32]) {
        debug_assert_eq!(build_rows.len(), probe_rows.len());
        if self.pending_out.is_empty() {
            let build = self
                .build_batch
                .as_ref()
                .expect("append_matches requires a build batch");
            let (first, second) = if self.swapped {
                (&probe_batch.columns, &build.columns)
            } else {
                (&build.columns, &probe_batch.columns)
            };
            self.pending_out = first
                .iter()
                .chain(second.iter())
                .map(|c| {
                    Column::with_nulls_ts(
                        ColumnData::with_capacity(c.type_id, BATCH_SIZE),
                        NullBitmap::empty(),
                        c.type_id,
                        c.fractional_digits,
                    )
                })
                .collect();
        }

        let mut offset = 0usize;
        while offset < build_rows.len() {
            let take = (BATCH_SIZE - self.pending_out_rows).min(build_rows.len() - offset);
            let bi = &build_rows[offset..offset + take];
            let pi = &probe_rows[offset..offset + take];
            {
                let build = self
                    .build_batch
                    .as_ref()
                    .expect("append_matches requires a build batch");
                let (first_src, first_idx, second_src, second_idx) = if self.swapped {
                    (&probe_batch.columns, pi, &build.columns, bi)
                } else {
                    (&build.columns, bi, &probe_batch.columns, pi)
                };
                let split = first_src.len();
                for (out, src) in self.pending_out[..split].iter_mut().zip(first_src.iter()) {
                    out.data.gather_from(&src.data, first_idx);
                    out.nulls.gather_from(&src.nulls, first_idx);
                }
                for (out, src) in self.pending_out[split..].iter_mut().zip(second_src.iter()) {
                    out.data.gather_from(&src.data, second_idx);
                    out.nulls.gather_from(&src.nulls, second_idx);
                }
            }
            self.pending_out_rows += take;
            offset += take;
            if self.pending_out_rows >= BATCH_SIZE {
                self.flush_pending_out();
            }
        }
    }

    /// Emits whatever matched rows have accumulated, if any.
    fn flush_pending_out(&mut self) {
        if self.pending_out_rows == 0 {
            return;
        }
        let fresh: Vec<Column> = self
            .pending_out
            .iter()
            .map(|c| {
                Column::with_nulls_ts(
                    ColumnData::with_capacity(c.type_id, BATCH_SIZE),
                    NullBitmap::empty(),
                    c.type_id,
                    c.fractional_digits,
                )
            })
            .collect();
        let done = std::mem::replace(&mut self.pending_out, fresh);
        self.pending_out_rows = 0;
        self.output_queue.push(DataBatch::new(done));
    }

    /// Compares a build row's key values against a probe row's key values.
    /// Returns true only when every key column is equal and non-NULL on both
    /// sides, so a 64-bit hash collision cannot false-join.
    #[inline]
    fn keys_match(
        &self,
        probe_keys: &[Cow<'_, Column>],
        build_row: usize,
        probe_row: usize,
    ) -> bool {
        keys_match_columns(&self.build_key_columns, probe_keys, build_row, probe_row)
    }

    /// Vectorized probe: collects all (build_row, probe_row) match pairs,
    /// then gathers output columns in bulk. Eliminates per-row per-column
    /// dispatch overhead.
    fn probe_batch_vectorized(
        &mut self,
        probe_batch: &DataBatch,
        probe_hashes: &[u64],
        probe_keys: &[Cow<'_, Column>],
    ) -> Vec<DataBatch> {
        let track_right = matches!(self.internal_join, JoinType::Right | JoinType::Full);
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
                if !self.keys_match(probe_keys, build_row as usize, probe_row) {
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

        // Phase 2: accumulate matches into full-size output batches,
        // external left columns before right columns regardless of which
        // side built.
        let _ = build;
        self.append_matches(&build_idx, probe_batch, &probe_idx);

        // Phase 3: Emit unmatched probe rows for probe-outer joins,
        // null-padding the build side in external column order.
        let mut results = Vec::new();
        if !unmatched_probe.is_empty() {
            self.flush_pending_out();
            let swapped = self.swapped;
            let buf = self.output_buffer.as_mut().unwrap();
            for &pr in &unmatched_probe {
                if swapped {
                    buf.push_left_null_right(probe_batch, pr as usize);
                } else {
                    buf.push_null_left_right(probe_batch, pr as usize);
                }
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

            // Emit unmatched build rows for a build-outer join, padded on
            // the probe side in external column order.
            if self.emitting_unmatched_build {
                let swapped = self.swapped;
                let buf = self.output_buffer.as_mut().unwrap();
                let build = self.build_batch.as_ref().unwrap();
                while self.unmatched_cursor < self.total_build_rows {
                    let row = self.unmatched_cursor;
                    self.unmatched_cursor += 1;
                    if !self.build_matched[row] {
                        if swapped {
                            buf.push_null_left_right(build, row);
                        } else {
                            buf.push_left_null_right(build, row);
                        }
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

            // Probe batches were drained during the build phase, when the
            // build side was chosen by actual row count. They stay separate,
            // so the probe side is never copied into one contiguous block.
            let probe_batches = std::mem::take(&mut self.probe_batches_pending);
            if probe_batches.is_empty() {
                // Probe side produced no rows. A build-outer join still
                // owes every build row, null-padded on the probe side.
                if matches!(self.internal_join, JoinType::Left | JoinType::Full)
                    && self.build_batch.is_some()
                {
                    self.emitting_unmatched_build = true;
                    return self.next().await;
                }
                self.finished = true;
                return Ok(None);
            }

            if self.build_batch.is_none() {
                // An empty build side can match nothing. A probe-outer join
                // still owes every probe row, null-padded on the build side.
                if matches!(self.internal_join, JoinType::Right | JoinType::Full) {
                    let swapped = self.swapped;
                    for probe_batch in &probe_batches {
                        let buf = self.output_buffer.as_mut().unwrap();
                        for probe_row in 0..probe_batch.num_rows {
                            if swapped {
                                buf.push_left_null_right(probe_batch, probe_row);
                            } else {
                                buf.push_null_left_right(probe_batch, probe_row);
                            }
                            if buf.is_full() {
                                let b = buf.flush(&self.left_types, &self.right_types);
                                self.output_queue.push(b);
                            }
                        }
                    }
                    let buf = self.output_buffer.as_mut().unwrap();
                    if !buf.is_empty() {
                        let b = buf.flush(&self.left_types, &self.right_types);
                        self.output_queue.push(b);
                    }
                }
                self.finished = true;
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
                return Ok(None);
            }

            for merged_probe in &probe_batches {
            let total_probe_rows = merged_probe.num_rows;
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

            // Materialize probe key columns once for value comparison after the
            // hi32 hash match across every probe path.
            let probe_keys = self.materialize_probe_keys(merged_probe)?;

            if fused_col_no_nulls {
                let key_idx = fused_key_idx.unwrap();
                let track_right = matches!(self.internal_join, JoinType::Right | JoinType::Full);
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
                                if !self.keys_match(&probe_keys, build_row as usize, probe_row) {
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

                // The integer-hash probe is only valid when the build side
                // filled its buckets with the same integer hash. A build key
                // with NULLs or an expression key hashed generically, and a
                // probe hashing the same values differently would look up
                // buckets that were never filled, dropping every match
                let col = &merged_probe.columns[key_idx];
                match (&col.data, self.build_used_int_hash) {
                    (ColumnData::Int64(v), true) => fused_probe_prefetch!(v),
                    (ColumnData::Int32(v), true) => fused_probe_prefetch!(v),
                    (ColumnData::Int16(v), true) => fused_probe_prefetch!(v),
                    (ColumnData::Int8(v), true) => fused_probe_prefetch!(v),
                    (ColumnData::UInt64(v), true) => fused_probe_prefetch!(v),
                    (ColumnData::UInt32(v), true) => fused_probe_prefetch!(v),
                    (ColumnData::UInt16(v), true) => fused_probe_prefetch!(v),
                    (ColumnData::UInt8(v), true) => fused_probe_prefetch!(v),
                    _ => {
                        // Non-integer fused path without prefetch (strings,
                        // decimals, and any probe whose build side hashed
                        // generically). Hashes the materialized key column,
                        // which carries any decimal alignment
                        let probe_hashes =
                            compute::hash_column_batch(&[probe_keys[0].as_ref()], total_probe_rows);
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
                                if !self.keys_match(&probe_keys, build_row as usize, probe_row) {
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

                // Accumulate matches into full-size output batches.
                self.append_matches(&build_idx, merged_probe, &probe_idx);

                // Emit unmatched probe rows for probe-outer joins. Pending
                // matches flush first so a row's position never depends on
                // which probe batch it came from.
                if !unmatched_probe.is_empty() {
                    self.flush_pending_out();
                    let swapped = self.swapped;
                    let buf = self.output_buffer.as_mut().unwrap();
                    for &pr in &unmatched_probe {
                        if swapped {
                            buf.push_left_null_right(merged_probe, pr as usize);
                        } else {
                            buf.push_null_left_right(merged_probe, pr as usize);
                        }
                        if buf.is_full() {
                            self.output_queue
                                .push(buf.flush(&self.left_types, &self.right_types));
                        }
                    }
                }
            } else {
                // Generic path: hash all probe keys, then probe. The
                // materialized probe key columns carry any decimal
                // alignment, so hashing them matches the build side
                let key_refs: Vec<&Column> = probe_keys.iter().map(|c| c.as_ref()).collect();
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
                    let swapped = self.swapped;
                    let track_right = matches!(self.internal_join, JoinType::Right | JoinType::Full);
                    let build_key_columns = &self.build_key_columns;
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
                            if !keys_match_columns(
                                build_key_columns,
                                &probe_keys,
                                build_row as usize,
                                probe_row,
                            ) {
                                continue;
                            }
                            // The condition schema is external left then
                            // right, so the combined row follows that order
                            let combined = if swapped {
                                combine_rows_single(
                                    merged_probe,
                                    probe_row,
                                    build,
                                    build_row as usize,
                                )
                            } else {
                                combine_rows_single(
                                    build,
                                    build_row as usize,
                                    merged_probe,
                                    probe_row,
                                )
                            };
                            let mask = evaluate(
                                self.remaining_condition.as_ref().unwrap(),
                                &combined,
                                &self.input_schema,
                                &[],
                            )?;
                            if !mask.is_null(0) && mask.get_bool(0) {
                                matched = true;
                                if !self.build_matched.is_empty() {
                                    self.build_matched[build_row as usize] = true;
                                }
                                if swapped {
                                    buf.push_matched(
                                        merged_probe,
                                        probe_row,
                                        build,
                                        build_row as usize,
                                    );
                                } else {
                                    buf.push_matched(
                                        build,
                                        build_row as usize,
                                        merged_probe,
                                        probe_row,
                                    );
                                }
                                if buf.is_full() {
                                    self.output_queue
                                        .push(buf.flush(&self.left_types, &self.right_types));
                                }
                            }
                        }
                        if !matched && track_right {
                            if swapped {
                                buf.push_left_null_right(merged_probe, probe_row);
                            } else {
                                buf.push_null_left_right(merged_probe, probe_row);
                            }
                            if buf.is_full() {
                                self.output_queue
                                    .push(buf.flush(&self.left_types, &self.right_types));
                            }
                        }
                    }
                } else {
                    let batches =
                        self.probe_batch_vectorized(merged_probe, &probe_hashes, &probe_keys);
                    self.output_queue.extend(batches);
                }
            }
            }

            // Emit whatever matched rows are still accumulating.
            self.flush_pending_out();

            // Flush remaining buffered rows.
            let buf = self.output_buffer.as_mut().unwrap();
            if !buf.is_empty() {
                self.output_queue
                    .push(buf.flush(&self.left_types, &self.right_types));
            }

            if matches!(self.internal_join, JoinType::Left | JoinType::Full) {
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

// ---------------------------------------------------------------------------
// ParallelHashJoinOperator
// ---------------------------------------------------------------------------

/// Yields one pre-materialized batch then stops. Feeds a partition's rows into
/// a per-partition serial hash join. A None batch yields nothing, so an empty
/// partition side needs no zero-row batch allocation.
struct SingleBatchSource {
    batch: Option<DataBatch>,
}

impl Operator for SingleBatchSource {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move { Ok(self.batch.take().map(ExecutionBatch::new)) })
    }
}

/// Total input rows below which partition + task-spawn overhead outweighs the
/// gain, so a single serial hash join runs instead.
const PARALLEL_JOIN_MIN_ROWS: usize = 8192;

/// Hash-partitioned parallel hash join. Drains both inputs, partitions their
/// rows by join-key hash into P buckets so equal keys co-locate, then runs P
/// independent serial hash joins concurrently and concatenates their output.
/// Outer-join semantics hold because a build row and every probe row that could
/// match it hash to the same partition, so unmatched detection stays correct
/// within each partition.
pub struct ParallelHashJoinOperator {
    left: Option<Box<dyn Operator>>,
    right: Option<Box<dyn Operator>>,
    join_type: JoinType,
    left_keys: Vec<BoundExpr>,
    right_keys: Vec<BoundExpr>,
    remaining_condition: Option<BoundExpr>,
    left_schema: Vec<LogicalColumn>,
    right_schema: Vec<LogicalColumn>,
    output: Vec<DataBatch>,
    output_idx: usize,
    started: bool,
}

impl ParallelHashJoinOperator {
    #[allow(clippy::too_many_arguments)]
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
        Self {
            left: Some(left),
            right: Some(right),
            join_type,
            left_keys,
            right_keys,
            remaining_condition,
            left_schema,
            right_schema,
            output: Vec::new(),
            output_idx: 0,
            started: false,
        }
    }

    /// Builds a serial hash join over two pre-materialized partition batches.
    fn build_partition_join(
        &self,
        build: Option<DataBatch>,
        probe: Option<DataBatch>,
    ) -> Box<dyn Operator> {
        let left_src = Box::new(SingleBatchSource { batch: build });
        let right_src = Box::new(SingleBatchSource { batch: probe });
        Box::new(HashJoinOperator::new(
            left_src,
            right_src,
            self.join_type,
            self.left_keys.clone(),
            self.right_keys.clone(),
            self.remaining_condition.clone(),
            self.left_schema.clone(),
            self.right_schema.clone(),
        ))
    }

    async fn run(&mut self) -> Result<()> {
        let left = self.left.take().expect("left taken once");
        let right = self.right.take().expect("right taken once");
        let build = merge_drained(left).await?;
        let probe = merge_drained(right).await?;

        let build_rows = build.as_ref().map(|b| b.num_rows).unwrap_or(0);
        let probe_rows = probe.as_ref().map(|b| b.num_rows).unwrap_or(0);

        let workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .clamp(1, 16);

        // Small inputs, no build rows, or a single worker: a single serial join
        // is cheaper than partition + spawn overhead.
        if workers <= 1 || build_rows == 0 || build_rows + probe_rows < PARALLEL_JOIN_MIN_ROWS {
            let op = self.build_partition_join(build, probe);
            self.output = drain_operator(op).await?;
            return Ok(());
        }

        let part_count = workers;
        let build = build.unwrap();
        let build_parts = self.partition_rows(
            &build,
            &self.left_keys,
            &self.left_schema,
            &probe,
            part_count,
        )?;
        let probe_parts = match &probe {
            Some(pb) => {
                self.partition_rows_for(pb, &self.right_keys, &self.right_schema, part_count)?
            }
            None => vec![Vec::new(); part_count],
        };

        // Spawn one serial join per non-empty partition. Each is self-contained
        // (no shared state), so they run concurrently across worker threads.
        let mut handles = Vec::with_capacity(part_count);
        for p in 0..part_count {
            let build_idx = &build_parts[p];
            let probe_idx = &probe_parts[p];
            if build_idx.is_empty() && probe_idx.is_empty() {
                continue;
            }
            let build_part = if build_idx.is_empty() {
                None
            } else {
                Some(take_rows(&build, build_idx))
            };
            let probe_part = match (&probe, probe_idx.is_empty()) {
                (Some(pb), false) => Some(take_rows(pb, probe_idx)),
                _ => None,
            };
            let op = self.build_partition_join(build_part, probe_part);
            handles.push(tokio::spawn(async move { drain_operator(op).await }));
        }

        let mut output = Vec::new();
        for handle in handles {
            match handle.await {
                Ok(Ok(batches)) => output.extend(batches),
                Ok(Err(e)) => return Err(e),
                Err(e) => {
                    return Err(zyron_common::ZyronError::ExecutionError(format!(
                        "parallel hash join partition task failed: {e}"
                    )));
                }
            }
        }
        self.output = output;
        Ok(())
    }

    /// Per-key decimal alignment target derived from the bound key
    /// expressions alone, so both sides of the partitioner compute the same
    /// scale and equal numbers land in the same partition. The serial join
    /// inside each partition aligns again for its own hashing
    fn static_key_align_scales(&self) -> Vec<Option<u8>> {
        self.left_keys
            .iter()
            .zip(self.right_keys.iter())
            .map(|(l, r)| {
                let ld = l.type_id() == TypeId::Decimal;
                let rd = r.type_id() == TypeId::Decimal;
                if !ld && !rd {
                    return None;
                }
                let ls = if ld { l.fractional_digits().unwrap_or(0) } else { 0 };
                let rs = if rd { r.fractional_digits().unwrap_or(0) } else { 0 };
                Some(ls.max(rs))
            })
            .collect()
    }

    /// Partition index per build row, casting each key to the common type it
    /// shares with the matching probe key so equal values hash identically.
    fn partition_rows(
        &self,
        build: &DataBatch,
        keys: &[BoundExpr],
        schema: &[LogicalColumn],
        probe: &Option<DataBatch>,
        part_count: usize,
    ) -> Result<Vec<Vec<u32>>> {
        let common_types = self.key_common_types(build, keys, schema, probe)?;
        let align = self.static_key_align_scales();
        let hashes = hash_keys(build, keys, schema, &common_types, &align)?;
        Ok(bucket_indices(&hashes, part_count))
    }

    /// Partitions probe rows using the same per-key common types.
    fn partition_rows_for(
        &self,
        probe: &DataBatch,
        keys: &[BoundExpr],
        schema: &[LogicalColumn],
        part_count: usize,
    ) -> Result<Vec<Vec<u32>>> {
        let common_types = self.probe_common_types(probe, keys, schema)?;
        let align = self.static_key_align_scales();
        let hashes = hash_keys(probe, keys, schema, &common_types, &align)?;
        Ok(bucket_indices(&hashes, part_count))
    }

    fn key_common_types(
        &self,
        build: &DataBatch,
        keys: &[BoundExpr],
        schema: &[LogicalColumn],
        probe: &Option<DataBatch>,
    ) -> Result<Vec<TypeId>> {
        let mut out = Vec::with_capacity(keys.len());
        for (i, k) in keys.iter().enumerate() {
            let lt = evaluate(k, build, schema, &[])?.type_id;
            let rt = match probe {
                Some(pb) if pb.num_rows > 0 => {
                    evaluate(&self.right_keys[i], pb, &self.right_schema, &[])?.type_id
                }
                _ => lt,
            };
            out.push(join_key_common_type(lt, rt));
        }
        Ok(out)
    }

    fn probe_common_types(
        &self,
        probe: &DataBatch,
        keys: &[BoundExpr],
        schema: &[LogicalColumn],
    ) -> Result<Vec<TypeId>> {
        let mut out = Vec::with_capacity(keys.len());
        for (i, k) in keys.iter().enumerate() {
            let rt = evaluate(k, probe, schema, &[])?.type_id;
            // The left key declared type pairs with the probe key so both sides
            // agree on the common type without re-materializing the build batch.
            let lt = self.left_keys[i].type_id();
            out.push(join_key_common_type(lt, rt));
        }
        Ok(out)
    }
}

impl Operator for ParallelHashJoinOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            if !self.started {
                self.started = true;
                self.run().await?;
            }
            if self.output_idx < self.output.len() {
                let batch = std::mem::replace(
                    &mut self.output[self.output_idx],
                    DataBatch::new(Vec::new()),
                );
                self.output_idx += 1;
                return Ok(Some(ExecutionBatch::new(batch)));
            }
            Ok(None)
        })
    }
}

/// Drives an operator to completion, collecting all output batches.
async fn drain_operator(mut op: Box<dyn Operator>) -> Result<Vec<DataBatch>> {
    let mut out = Vec::new();
    while let Some(eb) = op.next().await? {
        out.push(eb.batch);
    }
    Ok(out)
}

/// Drains an operator and merges its batches into one contiguous batch, or None
/// when it produced no rows.
async fn merge_drained(mut op: Box<dyn Operator>) -> Result<Option<DataBatch>> {
    let mut batches: Vec<DataBatch> = Vec::new();
    let mut total = 0usize;
    while let Some(eb) = op.next().await? {
        total += eb.batch.num_rows;
        batches.push(eb.batch);
    }
    if total == 0 || batches.is_empty() {
        return Ok(None);
    }
    if batches.len() == 1 {
        return Ok(Some(batches.pop().unwrap()));
    }
    let num_cols = batches[0].num_columns();
    let mut cols = Vec::with_capacity(num_cols);
    for c in 0..num_cols {
        let template = &batches[0].columns[c];
        let type_id = template.type_id;
        let fractional_digits = template.fractional_digits;
        let mut data = ColumnData::with_capacity(type_id, total);
        let mut nulls = NullBitmap::empty();
        for b in &batches {
            data.extend_from(&b.columns[c].data);
            nulls.extend_from(&b.columns[c].nulls);
        }
        cols.push(Column::with_nulls_ts(data, nulls, type_id, fractional_digits));
    }
    Ok(Some(DataBatch::new(cols)))
}

/// Builds a row-subset batch by taking the given row indices from every column.
fn take_rows(batch: &DataBatch, indices: &[u32]) -> DataBatch {
    let cols = batch.columns.iter().map(|c| c.take(indices)).collect();
    DataBatch::new(cols)
}

/// Hashes each row's join-key columns after casting them to common_types so
/// equal values hash identically regardless of declared key width.
fn hash_keys(
    batch: &DataBatch,
    keys: &[BoundExpr],
    schema: &[LogicalColumn],
    common_types: &[TypeId],
    align_scales: &[Option<u8>],
) -> Result<Vec<u64>> {
    let mut key_cols: Vec<Column> = Vec::with_capacity(keys.len());
    for (i, k) in keys.iter().enumerate() {
        let col = evaluate(k, batch, schema, &[])?;
        // A key with a decimal side hashes as Int128 on the aligned scale,
        // the same layout both sides convert to, so equal numbers share a
        // partition regardless of declared scale or integer width
        let col = if let Some(scale) = align_scales.get(i).copied().flatten() {
            compute::cast_column_to_decimal(&col, scale)?
        } else if col.type_id != common_types[i] {
            compute::cast_column(&col, common_types[i])?
        } else {
            col
        };
        key_cols.push(col);
    }
    let refs: Vec<&Column> = key_cols.iter().collect();
    Ok(compute::hash_column_batch(&refs, batch.num_rows))
}

/// Assigns each hashed row to a partition bucket.
fn bucket_indices(hashes: &[u64], part_count: usize) -> Vec<Vec<u32>> {
    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); part_count];
    for (row, &h) in hashes.iter().enumerate() {
        let p = (h % part_count as u64) as usize;
        buckets[p].push(row as u32);
    }
    buckets
}

/// Common type two join-key columns must share so equal values hash identically.
/// Equal types pass through; mixed integers widen to Int64; any float pairing
/// widens to Float64; otherwise the left type is used.
fn join_key_common_type(a: TypeId, b: TypeId) -> TypeId {
    if a == b {
        return a;
    }
    let a_num = is_integer_type(a) || is_float_type(a);
    let b_num = is_integer_type(b) || is_float_type(b);
    if a_num && b_num {
        if is_float_type(a) || is_float_type(b) {
            TypeId::Float64
        } else {
            TypeId::Int64
        }
    } else {
        a
    }
}

fn is_integer_type(t: TypeId) -> bool {
    matches!(
        t,
        TypeId::Int8
            | TypeId::Int16
            | TypeId::Int32
            | TypeId::Int64
            | TypeId::Int128
            | TypeId::UInt8
            | TypeId::UInt16
            | TypeId::UInt32
            | TypeId::UInt64
            | TypeId::UInt128
    )
}

fn is_float_type(t: TypeId) -> bool {
    matches!(t, TypeId::Float32 | TypeId::Float64)
}

#[cfg(test)]
mod parallel_join_tests {
    use super::*;
    use crate::column::ColumnData;
    use zyron_catalog::ColumnId;
    use zyron_planner::binder::ColumnRef;

    fn int_col(vals: Vec<i64>) -> Column {
        Column::new(ColumnData::Int64(vals), TypeId::Int64)
    }

    fn lcol(table_idx: usize, col: u16, name: &str) -> LogicalColumn {
        LogicalColumn {
            table_idx: Some(table_idx),
            column_id: ColumnId(col),
            name: name.to_string(),
            type_id: TypeId::Int64,
            nullable: false,
            fractional_digits: None,
        }
    }

    fn key_ref(table_idx: usize, col: u16) -> BoundExpr {
        BoundExpr::ColumnRef(ColumnRef {
            table_idx,
            column_id: ColumnId(col),
            type_id: TypeId::Int64,
            nullable: false,
            fractional_digits: None,
        })
    }

    fn src(batch: DataBatch) -> Box<dyn Operator> {
        Box::new(SingleBatchSource { batch: Some(batch) })
    }

    async fn collect_key0(mut op: Box<dyn Operator>) -> Vec<i64> {
        let mut out = Vec::new();
        while let Some(eb) = op.next().await.unwrap() {
            if let ColumnData::Int64(v) = &eb.batch.columns[0].data {
                out.extend_from_slice(v);
            }
        }
        out.sort_unstable();
        out
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn parallel_inner_matches_serial() {
        let n: i64 = 20_000; // above PARALLEL_JOIN_MIN_ROWS to force partitioning
        let build = DataBatch::new(vec![
            int_col((0..n).collect()),
            int_col((0..n).map(|i| i * 10).collect()),
        ]);
        // Probe keys repeat the lower half so each match is exercised many times.
        let probe = DataBatch::new(vec![int_col((0..n).map(|i| i % (n / 2)).collect())]);

        let lschema = vec![lcol(0, 0, "k"), lcol(0, 1, "payload")];
        let rschema = vec![lcol(1, 0, "rk")];

        let serial = Box::new(HashJoinOperator::new(
            src(build.clone()),
            src(probe.clone()),
            JoinType::Inner,
            vec![key_ref(0, 0)],
            vec![key_ref(1, 0)],
            None,
            lschema.clone(),
            rschema.clone(),
        ));
        let parallel = Box::new(ParallelHashJoinOperator::new(
            src(build),
            src(probe),
            JoinType::Inner,
            vec![key_ref(0, 0)],
            vec![key_ref(1, 0)],
            None,
            lschema,
            rschema,
        ));

        let s = collect_key0(serial).await;
        let p = collect_key0(parallel).await;
        assert_eq!(
            s.len(),
            p.len(),
            "row counts match (serial={}, parallel={})",
            s.len(),
            p.len()
        );
        assert_eq!(s, p, "joined left-key multisets match");
        assert!(!s.is_empty(), "join produced rows");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn parallel_left_outer_keeps_unmatched() {
        let n: i64 = 20_000;
        // Build keys 0..n; probe keys only n..2n so nothing matches.
        let build = DataBatch::new(vec![int_col((0..n).collect())]);
        let probe = DataBatch::new(vec![int_col((n..2 * n).collect())]);
        let lschema = vec![lcol(0, 0, "k")];
        let rschema = vec![lcol(1, 0, "rk")];

        let parallel = Box::new(ParallelHashJoinOperator::new(
            src(build),
            src(probe),
            JoinType::Left,
            vec![key_ref(0, 0)],
            vec![key_ref(1, 0)],
            None,
            lschema,
            rschema,
        ));
        let p = collect_key0(parallel).await;
        // LEFT join with no matches still emits every build row once.
        assert_eq!(p.len(), n as usize, "all unmatched build rows preserved");
    }
}

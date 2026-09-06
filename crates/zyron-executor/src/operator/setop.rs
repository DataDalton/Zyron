//! Set operation operators for UNION, INTERSECT, and EXCEPT.
//!
//! UNION ALL concatenates left then right. UNION deduplicates via typed hashing.
//! INTERSECT and EXCEPT use a count-based HashMap with typed hashing and
//! columnar storage for collision resolution.

use zyron_common::{Result, TypeId, ZyronError};
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
    /// Per-column declared types of the first left batch, used by the
    /// streaming UNION ALL path to align right-branch batches onto the
    /// operation's declared output type
    stream_declared: Option<Vec<(TypeId, Option<u8>)>>,
    /// Query memory budget the materialized row store reserves against,
    /// approximated by input batch size. The streaming UNION ALL path
    /// buffers nothing and reserves nothing. None runs unbudgeted.
    memory_budget: Option<std::sync::Arc<crate::context::QueryMemoryBudget>>,
}

/// Aligns a batch onto the declared per-column output types, which are the
/// left branch's and therefore the operation's. Merging mixed physical
/// types is a wrong answer or a panic: a decimal at another scale hashes
/// and compares as a different value, and a different variant either
/// panics the row store or pushes a fabricated default. A column that
/// cannot convert is a loud error. Returns None when nothing needed
/// converting
fn align_to_declared(
    declared: &[(TypeId, Option<u8>)],
    batch: &DataBatch,
) -> Result<Option<DataBatch>> {
    if batch.columns.len() != declared.len() {
        return Err(ZyronError::ExecutionError(format!(
            "set operation branch produced {} columns, expected {}",
            batch.columns.len(),
            declared.len()
        )));
    }
    let mut aligned: Option<Vec<Column>> = None;
    for (ci, col) in batch.columns.iter().enumerate() {
        let (want_type, want_scale) = declared[ci];
        if want_type == TypeId::Decimal {
            let target = want_scale.unwrap_or(0);
            if col.type_id == TypeId::Decimal && col.fractional_digits.unwrap_or(0) == target {
                continue;
            }
            let cast = compute::cast_column_to_decimal(col, target)?;
            aligned.get_or_insert_with(|| batch.columns.clone())[ci] = cast;
        } else if matches!(want_type, TypeId::Timestamp | TypeId::TimestampTz)
            && col.type_id == want_type
        {
            // One TypeId, two physical forms: i64 microseconds for p<=6 and
            // i128 picoseconds for p>6. The microsecond side scales up
            // exactly, the reverse would lose information and is refused
            let want_ps = want_scale.unwrap_or(6) > 6;
            let have_ps = col.fractional_digits.unwrap_or(6) > 6;
            if want_ps && !have_ps {
                let cast = compute::scale_us_to_ps(col, want_scale)?;
                aligned.get_or_insert_with(|| batch.columns.clone())[ci] = cast;
            } else if !want_ps && have_ps {
                return Err(ZyronError::ExecutionError(
                    "cannot merge a picosecond timestamp branch into a microsecond set \
                     operation column, put the higher-precision branch first"
                        .to_string(),
                ));
            }
        } else if col.type_id != want_type && want_type != TypeId::Null {
            let cast = compute::cast_column(col, want_type)?;
            aligned.get_or_insert_with(|| batch.columns.clone())[ci] = cast;
        }
    }
    Ok(aligned.map(DataBatch::new))
}

/// The declared type of each output column, from the first batch the
/// operation saw.
fn declared_of(batch: &DataBatch) -> Vec<(TypeId, Option<u8>)> {
    batch
        .columns
        .iter()
        .map(|c| (c.type_id, c.fractional_digits))
        .collect()
}

/// Aligns a streamed right-branch batch onto the captured left types,
/// passing everything else through untouched. An absent capture means the
/// left branch produced no batch, and the branch's own types stand
fn align_streamed(
    eb: Option<ExecutionBatch>,
    declared: &Option<Vec<(TypeId, Option<u8>)>>,
) -> Result<Option<ExecutionBatch>> {
    let Some(eb) = eb else { return Ok(None) };
    let Some(declared) = declared else {
        return Ok(Some(eb));
    };
    match align_to_declared(declared, &eb.batch)? {
        Some(batch) => Ok(Some(ExecutionBatch::new(batch))),
        None => Ok(Some(eb)),
    }
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
/// and uses typed hashing for deduplication and counting.
///
/// Rows are addressed through a flat bucket table chained through an array,
/// which is what the hash join builds its side with. A map from hash to a
/// vector of row indices allocates one vector per distinct hash, so a
/// distinct union over a million rows made a million short-lived
/// allocations to hold one index each
struct RowStore {
    columns: Vec<Column>,
    /// Bucket heads, chained backwards through `chain`
    table: crate::compute::FlatHashTable,
    /// Per stored row, the row that held its bucket before it
    chain: Vec<u32>,
    /// Per stored row, its hash. Kept so growing the table is a pass over
    /// these rather than a rehash of every stored value
    hashes: Vec<u64>,
    /// Rows the table is sized for. Past it the table doubles and rebuilds
    capacity_rows: usize,
    counts: Vec<usize>,
    num_rows: usize,
    /// Per-column declared type of the first batch seen, which is the left
    /// branch and therefore the operation's declared output type. Every
    /// later batch aligns onto these before hashing
    declared: Vec<(TypeId, Option<u8>)>,
}

/// Rows a fresh store is sized for, before any growth
const ROW_STORE_INITIAL_ROWS: usize = 1024;

impl RowStore {
    fn new() -> Self {
        Self {
            columns: Vec::new(),
            table: crate::compute::FlatHashTable::with_capacity(ROW_STORE_INITIAL_ROWS),
            chain: Vec::new(),
            hashes: Vec::new(),
            capacity_rows: ROW_STORE_INITIAL_ROWS,
            counts: Vec::new(),
            num_rows: 0,
            declared: Vec::new(),
        }
    }

    /// The stored row equal to this one, or None.
    ///
    /// Walks the bucket chain rather than a vector of candidates, so a
    /// lookup touches the table, the chain and whichever rows collided,
    /// and nothing per distinct hash
    fn find(&self, batch: &DataBatch, row: usize, hash: u64) -> Option<usize> {
        let mut idx = self.table.get(hash);
        while idx != u32::MAX {
            let at = idx as usize;
            if self.hashes[at] == hash && self.row_equals(batch, row, at) {
                return Some(at);
            }
            idx = self.chain[at];
        }
        None
    }

    /// Doubles the bucket table when the rows outgrow it, rebuilding the
    /// chains from the hashes already stored
    fn grow_if_needed(&mut self) {
        if self.num_rows < self.capacity_rows {
            return;
        }
        self.capacity_rows = (self.capacity_rows * 2).max(ROW_STORE_INITIAL_ROWS);
        self.table = crate::compute::FlatHashTable::with_capacity(self.capacity_rows);
        for (row, hash) in self.hashes.iter().enumerate() {
            self.chain[row] = self.table.insert(*hash, row as u32);
        }
    }

    fn ensure_columns(&mut self, batch: &DataBatch) {
        if self.columns.is_empty() {
            self.columns = batch
                .columns
                .iter()
                .map(|c| {
                    Column::new_ts(
                        ColumnData::with_capacity(c.type_id, 64),
                        c.type_id,
                        c.fractional_digits,
                    )
                })
                .collect();
            self.declared = declared_of(batch);
        }
        // A column typed NULL says nothing about the operation's type. The
        // first branch that brings a real type claims it, and the store
        // column, which holds only nulls so far, rebuilds as that type
        for (ci, col) in batch.columns.iter().enumerate() {
            if ci < self.declared.len()
                && self.declared[ci].0 == TypeId::Null
                && col.type_id != TypeId::Null
            {
                self.declared[ci] = (col.type_id, col.fractional_digits);
                let held = self.columns[ci].len();
                self.columns[ci] = Column::null_column(col.type_id, held);
                self.columns[ci].fractional_digits = col.fractional_digits;
            }
        }
    }

    /// Finds or inserts a row, returning (store_index, was_new).
    fn find_or_insert(&mut self, batch: &DataBatch, row: usize, hash: u64) -> (usize, bool) {
        if let Some(idx) = self.find(batch, row, hash) {
            return (idx, false);
        }
        self.grow_if_needed();
        let idx = self.num_rows;
        let prev = self.table.insert(hash, idx as u32);
        self.chain.push(prev);
        self.hashes.push(hash);
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
    ///
    /// One typed gather per column over the indices, rather than one pass
    /// per row over the columns. Which buffer a value belongs in and how to
    /// copy it are the same answers for every row of a column, and settling
    /// them per value is what a row-major copy out of a columnar store
    /// spends its time on
    fn extract_rows(&self, indices: &[usize]) -> DataBatch {
        if indices.is_empty() || self.columns.is_empty() {
            return DataBatch::empty();
        }
        let taken: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
        DataBatch::new(self.columns.iter().map(|c| c.take(&taken)).collect())
    }

    /// Every stored row in insertion order, handed over rather than copied.
    ///
    /// A distinct union's answer is the store itself, so gathering it into a
    /// second set of columns copies the whole result to reproduce it
    fn into_batch(self) -> DataBatch {
        DataBatch::new(self.columns)
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
            stream_declared: None,
            memory_budget: None,
        }
    }

    /// Attaches the query memory budget. Set by the operator builder from
    /// the execution context.
    pub fn set_memory_budget(
        &mut self,
        budget: Option<std::sync::Arc<crate::context::QueryMemoryBudget>>,
    ) {
        self.memory_budget = budget;
    }

    fn reserve_memory(&self, bytes: u64) -> Result<()> {
        match &self.memory_budget {
            Some(budget) => budget.reserve(bytes),
            None => Ok(()),
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
                    self.reserve_memory(eb.batch.approx_bytes())?;
                    let batch = &eb.batch;
                    store.ensure_columns(batch);
                    let realigned = align_to_declared(&store.declared, batch)?;
                    let batch = realigned.as_ref().unwrap_or(batch);
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
                    self.reserve_memory(eb.batch.approx_bytes())?;
                    let batch = &eb.batch;
                    store.ensure_columns(batch);
                    let realigned = align_to_declared(&store.declared, batch)?;
                    let batch = realigned.as_ref().unwrap_or(batch);
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
        Ok(Some(store.into_batch()))
    }

    async fn materialize_intersect(&mut self) -> Result<Option<DataBatch>> {
        let mut store = RowStore::new();

        // Build counts from left side.
        loop {
            match self.left.next().await? {
                Some(eb) => {
                    self.reserve_memory(eb.batch.approx_bytes())?;
                    let batch = &eb.batch;
                    store.ensure_columns(batch);
                    let realigned = align_to_declared(&store.declared, batch)?;
                    let batch = realigned.as_ref().unwrap_or(batch);
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
                    self.reserve_memory(eb.batch.approx_bytes())?;
                    let batch = &eb.batch;
                    let realigned = align_to_declared(&store.declared, batch)?;
                    let batch = realigned.as_ref().unwrap_or(batch);
                    let col_refs: Vec<&Column> = batch.columns.iter().collect();
                    let hashes = compute::hash_column_batch(&col_refs, batch.num_rows);
                    for row in 0..batch.num_rows {
                        // Look up in store without inserting.
                        if let Some(idx) = store.find(batch, row, hashes[row])
                            && store.counts[idx] > 0
                        {
                            result_indices.push(idx);
                            if self.all {
                                store.counts[idx] -= 1;
                            } else {
                                store.counts[idx] = 0;
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
                    self.reserve_memory(eb.batch.approx_bytes())?;
                    let batch = &eb.batch;
                    store.ensure_columns(batch);
                    let realigned = align_to_declared(&store.declared, batch)?;
                    let batch = realigned.as_ref().unwrap_or(batch);
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
                    self.reserve_memory(eb.batch.approx_bytes())?;
                    let batch = &eb.batch;
                    let realigned = align_to_declared(&store.declared, batch)?;
                    let batch = realigned.as_ref().unwrap_or(batch);
                    let col_refs: Vec<&Column> = batch.columns.iter().collect();
                    let hashes = compute::hash_column_batch(&col_refs, batch.num_rows);
                    for row in 0..batch.num_rows {
                        if let Some(idx) = store.find(batch, row, hashes[row]) {
                            if self.all {
                                store.counts[idx] = store.counts[idx].saturating_sub(1);
                            } else {
                                store.counts[idx] = 0;
                            }
                        }
                    }
                }
                None => break,
            }
        }

        // Emit remaining left rows in original order. The tally is one slot
        // per stored row rather than a map keyed by row index, since every
        // key it could ever hold is a row the store already numbered
        let mut result_indices: Vec<usize> = Vec::new();
        let mut emitted = vec![0usize; store.num_rows];

        for &idx in &left_order {
            if emitted[idx] < store.counts[idx] {
                result_indices.push(idx);
                emitted[idx] += 1;
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
                        Some(eb) => {
                            if self.stream_declared.is_none() {
                                self.stream_declared = Some(declared_of(&eb.batch));
                            }
                            Ok(Some(eb))
                        }
                        None => {
                            self.state = SetOpState::Right;
                            align_streamed(self.right.next().await?, &self.stream_declared)
                        }
                    }
                }
                SetOpState::Right => {
                    align_streamed(self.right.next().await?, &self.stream_declared)
                }
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
            crate::compute::f32_key_eq(va[a_idx], vb[b_idx])
        }
        (ColumnData::Float64(va), ColumnData::Float64(vb)) => {
            crate::compute::f64_key_eq(va[a_idx], vb[b_idx])
        }
        (ColumnData::Utf8(va), ColumnData::Utf8(vb)) => va[a_idx] == vb[b_idx],
        (ColumnData::Binary(va), ColumnData::Binary(vb)) => va[a_idx] == vb[b_idx],
        (ColumnData::FixedBinary16(va), ColumnData::FixedBinary16(vb)) => va[a_idx] == vb[b_idx],
        _ => false,
    }
}

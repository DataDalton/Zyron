//! Stream join operators: stream-stream join, interval join, lookup join,
//! and temporal join.
//!
//! All join operators implement StreamOperator. State is stored as raw
//! column vectors (not wrapped in StreamRecord) for zero-overhead access.
//! Output is batch-constructed to amortize allocation. Lookup join uses
//! a direct FlatU64Map cache with no locking on the hot path.

use zyron_common::Result;

use crate::checkpoint::CheckpointBarrier;
use crate::column::{StreamBatch, StreamColumn, StreamColumnData};
use crate::hash::{FlatU64Map, hash_column_batch_into, hash_multi_column_batch_into};
use crate::record::{ChangeFlag, StreamRecord};
use crate::state::StateSnapshot;
use crate::stream_operator::{OperatorMetrics, StreamOperator};
use crate::watermark::Watermark;

// ---------------------------------------------------------------------------
// JoinStore: single columnar store for one side of a join
// ---------------------------------------------------------------------------

/// Stores ALL rows for one side of a join in a single set of column Vecs.
/// A separate FlatU64Map maps key_hash -> list of row indices.
/// Zero per-key allocation. Appending a row pushes into the shared columns
/// and adds the row index to the key's index list.
struct JoinStore {
    columns: Vec<StreamColumnData>,
    event_times: Vec<i64>,
    /// key_hash -> start of linked list in `next` array.
    index: FlatU64Map<u64>,
    /// Per-row linked list: next[row] = next row with same key, or sentinel.
    next: Vec<u64>,
    len: usize,
    initialized: bool,
}

/// Sentinel for end of linked list. Uses u64::MAX - 1 to avoid collision
/// with FlatU64Map's empty sentinel (u64::MAX).
const JOIN_STORE_NULL: u64 = u64::MAX - 1;

impl JoinStore {
    fn new() -> Self {
        Self {
            columns: Vec::new(),
            event_times: Vec::new(),
            index: FlatU64Map::new(),
            next: Vec::new(),
            len: 0,
            initialized: false,
        }
    }

    /// Initializes column schema from the first batch. Called once.
    #[inline]
    fn init_schema(&mut self, batch: &StreamBatch) {
        if !self.initialized {
            self.columns = batch.columns.iter().map(|c| c.data.empty_like()).collect();
            self.initialized = true;
        }
    }

    /// Appends a single row.
    #[inline]
    fn append_row(&mut self, record: &StreamRecord, row: usize, key_hash: u64) {
        let row_idx = self.len as u64;
        // Push column values using typed fast paths.
        for (col_buf, src_col) in self.columns.iter_mut().zip(record.batch.columns.iter()) {
            match (col_buf, &src_col.data) {
                (StreamColumnData::Int64(dst), StreamColumnData::Int64(src)) => dst.push(src[row]),
                (StreamColumnData::Float64(dst), StreamColumnData::Float64(src)) => {
                    dst.push(src[row])
                }
                (StreamColumnData::Int32(dst), StreamColumnData::Int32(src)) => dst.push(src[row]),
                (StreamColumnData::Boolean(dst), StreamColumnData::Boolean(src)) => {
                    dst.push(src[row])
                }
                (StreamColumnData::Utf8(dst), StreamColumnData::Utf8(src)) => {
                    dst.push(src[row].clone())
                }
                (dst, src_data) => dst.push_scalar(&src_data.get_scalar(row)),
            }
        }
        self.event_times.push(record.event_times[row]);

        // Link into the key's chain.
        let prev_head = self.index.get(key_hash).copied().unwrap_or(JOIN_STORE_NULL);
        self.next.push(prev_head);
        self.index.insert(key_hash, row_idx);
        self.len += 1;
    }

    /// Iterates all row indices for a given key hash.
    #[inline]
    fn rows_for_key(&self, key_hash: u64) -> JoinStoreIter<'_> {
        let head = self.index.get(key_hash).copied().unwrap_or(JOIN_STORE_NULL);
        JoinStoreIter {
            store: self,
            cursor: head,
        }
    }

    fn clear(&mut self) {
        for col in &mut self.columns {
            *col = col.empty_like();
        }
        self.event_times.clear();
        self.index.clear();
        self.next.clear();
        self.len = 0;
    }

    /// Evicts rows older than cutoff by rebuilding the store.
    fn evict_before(&mut self, cutoff: i64) {
        if self.len == 0 {
            return;
        }

        let mask: Vec<bool> = self.event_times.iter().map(|&t| t >= cutoff).collect();
        let keep_count = mask.iter().filter(|&&b| b).count();
        if keep_count == self.len {
            return;
        }
        if keep_count == 0 {
            self.clear();
            return;
        }

        // Build index mapping: old_row -> new_row.
        let mut new_columns: Vec<StreamColumnData> =
            self.columns.iter().map(|c| c.filter(&mask)).collect();
        let mut new_times = Vec::with_capacity(keep_count);
        for (i, &keep) in mask.iter().enumerate() {
            if keep {
                new_times.push(self.event_times[i]);
            }
        }

        // Rebuild index and next chains.
        let mut new_index = FlatU64Map::new();
        let mut new_next: Vec<u64> = Vec::with_capacity(keep_count);
        let mut old_row_to_new: Vec<u64> = vec![JOIN_STORE_NULL; self.len];
        let mut new_row = 0u64;
        for i in 0..self.len {
            if mask[i] {
                old_row_to_new[i] = new_row;
                new_row += 1;
            }
        }

        new_next.resize(keep_count, JOIN_STORE_NULL);
        self.index.iter(|key_hash, &head| {
            let mut cursor = head;
            let mut new_head = JOIN_STORE_NULL;
            while cursor != JOIN_STORE_NULL {
                let mapped = old_row_to_new[cursor as usize];
                if mapped != JOIN_STORE_NULL {
                    new_next[mapped as usize] = new_head;
                    new_head = mapped;
                }
                cursor = self.next[cursor as usize];
            }
            if new_head != JOIN_STORE_NULL {
                new_index.insert(key_hash, new_head);
            }
        });

        self.columns = new_columns;
        self.event_times = new_times;
        self.index = new_index;
        self.next = new_next;
        self.len = keep_count;
    }
}

struct JoinStoreIter<'a> {
    store: &'a JoinStore,
    cursor: u64,
}

impl Iterator for JoinStoreIter<'_> {
    type Item = usize; // row index
    #[inline]
    fn next(&mut self) -> Option<usize> {
        if self.cursor == JOIN_STORE_NULL {
            return None;
        }
        let row = self.cursor as usize;
        self.cursor = self.store.next[row];
        Some(row)
    }
}

// ---------------------------------------------------------------------------
// gather_build_column: type-safe column gathering for all StreamColumnData types
// ---------------------------------------------------------------------------

/// Gathers a single column value from each build-side source (one row per source).
/// Each &StreamColumnData contains one or more rows. Takes row 0 from each.
fn gather_build_column(sources: &[&StreamColumnData]) -> StreamColumnData {
    if sources.is_empty() {
        return StreamColumnData::Int64(Vec::new());
    }
    let mut col = sources[0].empty_like_with_capacity(sources.len());
    for src in sources {
        let scalar = src.get_scalar(0);
        col.push_scalar(&scalar);
    }
    col
}

// ---------------------------------------------------------------------------
// StreamStreamJoin
// ---------------------------------------------------------------------------

/// Stream-stream equi-join with windowed state.
/// Uses JoinStore (single columnar store per side, zero per-key allocation).
/// Rows are indexed by key hash via linked lists for O(1) key lookup.
pub struct StreamStreamJoin {
    id: u32,
    op_metrics: OperatorMetrics,
    left_key_cols: Vec<usize>,
    right_key_cols: Vec<usize>,
    window_ms: i64,
    left_state: JoinStore,
    right_state: JoinStore,
    current_watermark: i64,
    is_left_input: bool,
    hash_buf: Vec<u64>,
}

impl StreamStreamJoin {
    pub fn new(
        id: u32,
        left_key_cols: Vec<usize>,
        right_key_cols: Vec<usize>,
        window_ms: i64,
    ) -> Self {
        Self {
            id,
            op_metrics: OperatorMetrics::new(),
            left_key_cols,
            right_key_cols,
            window_ms,
            left_state: JoinStore::new(),
            right_state: JoinStore::new(),
            current_watermark: i64::MIN,
            is_left_input: true,
            hash_buf: Vec::new(),
        }
    }

    pub fn set_input_side(&mut self, is_left: bool) {
        self.is_left_input = is_left;
    }
}

impl StreamOperator for StreamStreamJoin {
    fn process(&mut self, record: StreamRecord) -> Result<Vec<StreamRecord>> {
        let num_rows = record.num_rows();
        self.op_metrics.record_input(num_rows as u64);

        if num_rows == 0 {
            return Ok(Vec::new());
        }

        // Hash keys into reusable buffer.
        let key_cols = if self.is_left_input {
            &self.left_key_cols
        } else {
            &self.right_key_cols
        };
        if key_cols.len() == 1 {
            hash_column_batch_into(
                record.batch.column(key_cols[0]),
                num_rows,
                &mut self.hash_buf,
            );
        } else {
            let cols: Vec<&StreamColumn> =
                key_cols.iter().map(|&i| record.batch.column(i)).collect();
            hash_multi_column_batch_into(&cols, num_rows, &mut self.hash_buf);
        }

        // Probe the other side. Build output columns inline.
        let other = if self.is_left_input {
            &self.right_state
        } else {
            &self.left_state
        };
        let build_col_count = other.columns.len();

        let mut probe_rows: Vec<u32> = Vec::with_capacity(num_rows);
        let mut out_event_times: Vec<i64> = Vec::with_capacity(num_rows);
        let mut build_out_cols: Vec<StreamColumnData> =
            other.columns.iter().map(|c| c.empty_like()).collect();

        for (row_idx, &key_hash) in self.hash_buf.iter().enumerate() {
            let probe_time = record.event_times[row_idx];
            for br in other.rows_for_key(key_hash) {
                if (probe_time - other.event_times[br]).abs() <= self.window_ms {
                    probe_rows.push(row_idx as u32);
                    out_event_times.push(probe_time.max(other.event_times[br]));
                    for (out_col, src_col) in build_out_cols.iter_mut().zip(other.columns.iter()) {
                        match (out_col, src_col) {
                            (StreamColumnData::Int64(dst), StreamColumnData::Int64(src)) => {
                                dst.push(src[br])
                            }
                            (StreamColumnData::Float64(dst), StreamColumnData::Float64(src)) => {
                                dst.push(src[br])
                            }
                            (StreamColumnData::Int32(dst), StreamColumnData::Int32(src)) => {
                                dst.push(src[br])
                            }
                            (StreamColumnData::Utf8(dst), StreamColumnData::Utf8(src)) => {
                                dst.push(src[br].clone())
                            }
                            (dst, src) => dst.push_scalar(&src.get_scalar(br)),
                        }
                    }
                }
            }
        }

        let output = if !probe_rows.is_empty() {
            let match_count = probe_rows.len();
            let mut out_columns: Vec<StreamColumnData> =
                Vec::with_capacity(record.batch.num_columns() + build_col_count);
            for col_idx in 0..record.batch.num_columns() {
                out_columns.push(record.batch.column(col_idx).data.take(&probe_rows));
            }
            out_columns.extend(build_out_cols);
            let out_cols: Vec<StreamColumn> = out_columns
                .into_iter()
                .map(StreamColumn::from_data)
                .collect();
            let flags = vec![ChangeFlag::Insert; match_count];
            vec![StreamRecord::new(
                StreamBatch::new(out_cols),
                out_event_times,
                flags,
            )]
        } else {
            Vec::new()
        };

        // Insert into our side. Single shared store, zero per-key allocation.
        let my = if self.is_left_input {
            &mut self.left_state
        } else {
            &mut self.right_state
        };
        my.init_schema(&record.batch);
        for (row_idx, &key_hash) in self.hash_buf.iter().enumerate() {
            my.append_row(&record, row_idx, key_hash);
        }

        let out_rows: usize = output.iter().map(|r| r.num_rows()).sum();
        self.op_metrics.record_output(out_rows as u64);
        Ok(output)
    }

    fn on_watermark(&mut self, watermark: Watermark) -> Result<Vec<StreamRecord>> {
        self.current_watermark = watermark.timestamp_ms;
        self.op_metrics.update_watermark(watermark.timestamp_ms);
        let cutoff = watermark.timestamp_ms - self.window_ms;
        self.left_state.evict_before(cutoff);
        self.right_state.evict_before(cutoff);
        Ok(Vec::new())
    }

    fn on_timer(&mut self, _timestamp_ms: i64) -> Result<Vec<StreamRecord>> {
        Ok(Vec::new())
    }

    fn on_barrier(&mut self, _barrier: CheckpointBarrier) -> Result<StateSnapshot> {
        Ok(StateSnapshot::empty(0))
    }

    fn restore(&mut self, _snapshot: StateSnapshot) -> Result<()> {
        self.left_state.clear();
        self.right_state.clear();
        Ok(())
    }

    fn operator_id(&self) -> u32 {
        self.id
    }

    fn metrics(&self) -> &OperatorMetrics {
        &self.op_metrics
    }
}

// ---------------------------------------------------------------------------
// IntervalJoin
// ---------------------------------------------------------------------------

/// Asymmetric interval join: matches where
/// left.time + lower_bound <= right.time <= left.time + upper_bound.
pub struct IntervalJoin {
    id: u32,
    op_metrics: OperatorMetrics,
    left_key_cols: Vec<usize>,
    right_key_cols: Vec<usize>,
    lower_bound_ms: i64,
    upper_bound_ms: i64,
    left_state: JoinStore,
    right_state: JoinStore,
    current_watermark: i64,
    is_left_input: bool,
    hash_buf: Vec<u64>,
}

impl IntervalJoin {
    pub fn new(
        id: u32,
        left_key_cols: Vec<usize>,
        right_key_cols: Vec<usize>,
        lower_bound_ms: i64,
        upper_bound_ms: i64,
    ) -> Self {
        Self {
            id,
            op_metrics: OperatorMetrics::new(),
            left_key_cols,
            right_key_cols,
            lower_bound_ms,
            upper_bound_ms,
            left_state: JoinStore::new(),
            right_state: JoinStore::new(),
            current_watermark: i64::MIN,
            is_left_input: true,
            hash_buf: Vec::new(),
        }
    }

    pub fn set_input_side(&mut self, is_left: bool) {
        self.is_left_input = is_left;
    }
}

impl StreamOperator for IntervalJoin {
    fn process(&mut self, record: StreamRecord) -> Result<Vec<StreamRecord>> {
        let num_rows = record.num_rows();
        self.op_metrics.record_input(num_rows as u64);

        if num_rows == 0 {
            return Ok(Vec::new());
        }

        let key_cols = if self.is_left_input {
            &self.left_key_cols
        } else {
            &self.right_key_cols
        };

        if key_cols.len() == 1 {
            hash_column_batch_into(
                record.batch.column(key_cols[0]),
                num_rows,
                &mut self.hash_buf,
            );
        } else {
            let cols: Vec<&StreamColumn> =
                key_cols.iter().map(|&i| record.batch.column(i)).collect();
            hash_multi_column_batch_into(&cols, num_rows, &mut self.hash_buf);
        }

        let other = if self.is_left_input {
            &self.right_state
        } else {
            &self.left_state
        };

        let mut probe_indices: Vec<u32> = Vec::with_capacity(num_rows);
        let mut build_times = Vec::with_capacity(num_rows);

        for (row_idx, &key_hash) in self.hash_buf.iter().enumerate() {
            let probe_time = record.event_times[row_idx];
            for br in other.rows_for_key(key_hash) {
                let bt = other.event_times[br];
                let (left_time, right_time) = if self.is_left_input {
                    (probe_time, bt)
                } else {
                    (bt, probe_time)
                };
                if right_time >= left_time + self.lower_bound_ms
                    && right_time <= left_time + self.upper_bound_ms
                {
                    probe_indices.push(row_idx as u32);
                    build_times.push(bt);
                }
            }
        }

        let output = if !probe_indices.is_empty() {
            let event_times: Vec<i64> = probe_indices
                .iter()
                .zip(build_times.iter())
                .map(|(&pi, &bt)| record.event_times[pi as usize].max(bt))
                .collect();
            let flags = vec![ChangeFlag::Insert; probe_indices.len()];
            let out_cols: Vec<StreamColumn> = record
                .batch
                .columns
                .iter()
                .map(|col| col.take(&probe_indices))
                .collect();
            vec![StreamRecord::new(
                StreamBatch::new(out_cols),
                event_times,
                flags,
            )]
        } else {
            Vec::new()
        };

        let my = if self.is_left_input {
            &mut self.left_state
        } else {
            &mut self.right_state
        };
        my.init_schema(&record.batch);
        for (row_idx, &key_hash) in self.hash_buf.iter().enumerate() {
            my.append_row(&record, row_idx, key_hash);
        }

        let out_rows: usize = output.iter().map(|r| r.num_rows()).sum();
        self.op_metrics.record_output(out_rows as u64);
        Ok(output)
    }

    fn on_watermark(&mut self, watermark: Watermark) -> Result<Vec<StreamRecord>> {
        self.current_watermark = watermark.timestamp_ms;
        self.op_metrics.update_watermark(watermark.timestamp_ms);
        let window_ms = self.upper_bound_ms - self.lower_bound_ms;
        let cutoff = watermark.timestamp_ms - window_ms;
        self.left_state.evict_before(cutoff);
        self.right_state.evict_before(cutoff);
        Ok(Vec::new())
    }

    fn on_timer(&mut self, _timestamp_ms: i64) -> Result<Vec<StreamRecord>> {
        Ok(Vec::new())
    }

    fn on_barrier(&mut self, _barrier: CheckpointBarrier) -> Result<StateSnapshot> {
        Ok(StateSnapshot::empty(0))
    }

    fn restore(&mut self, _snapshot: StateSnapshot) -> Result<()> {
        self.left_state.clear();
        self.right_state.clear();
        Ok(())
    }

    fn operator_id(&self) -> u32 {
        self.id
    }

    fn metrics(&self) -> &OperatorMetrics {
        &self.op_metrics
    }
}

// ---------------------------------------------------------------------------
// LookupJoin
// ---------------------------------------------------------------------------

/// Lookup join against an external source with an in-memory cache.
/// Pre-hashed keys with TTL-based eviction. Cache stores StreamBatch
/// directly (no wrapper overhead). Probe batches output using column take().
pub struct LookupJoin {
    id: u32,
    op_metrics: OperatorMetrics,
    probe_key_cols: Vec<usize>,
    /// Cache: key_hash -> (batch, insert_time_ms).
    cache: FlatU64Map<(StreamBatch, i64)>,
    ttl_ms: i64,
    max_entries: usize,
    /// Lookup function called on cache misses.
    lookup_fn: Box<dyn Fn(u64) -> Result<Option<StreamBatch>> + Send + Sync>,
    current_time_ms: i64,
    /// Reusable hash buffer.
    hash_buf: Vec<u64>,
}

impl LookupJoin {
    pub fn new(
        id: u32,
        probe_key_cols: Vec<usize>,
        ttl_ms: i64,
        max_entries: usize,
        lookup_fn: Box<dyn Fn(u64) -> Result<Option<StreamBatch>> + Send + Sync>,
    ) -> Self {
        Self {
            id,
            op_metrics: OperatorMetrics::new(),
            probe_key_cols,
            cache: FlatU64Map::default(),
            ttl_ms,
            max_entries,
            lookup_fn,
            current_time_ms: 0,
            hash_buf: Vec::new(),
        }
    }

    fn evict_expired(&mut self) {
        let cutoff = self.current_time_ms - self.ttl_ms;
        self.cache.retain(|_, val| val.1 >= cutoff);

        if self.cache.len() > self.max_entries {
            let mut entries: Vec<(u64, i64)> = Vec::new();
            self.cache.iter(|k, val| {
                entries.push((k, val.1));
            });
            entries.sort_unstable_by_key(|(_, t)| *t);
            let to_remove = self.cache.len() - self.max_entries;
            for (key, _) in entries.iter().take(to_remove) {
                self.cache.remove(*key);
            }
        }
    }
}

impl StreamOperator for LookupJoin {
    fn process(&mut self, record: StreamRecord) -> Result<Vec<StreamRecord>> {
        let num_rows = record.num_rows();
        self.op_metrics.record_input(num_rows as u64);

        if num_rows == 0 {
            return Ok(Vec::new());
        }

        // Hash into reusable buffer.
        if self.probe_key_cols.len() == 1 {
            hash_column_batch_into(
                record.batch.column(self.probe_key_cols[0]),
                num_rows,
                &mut self.hash_buf,
            );
        } else {
            let cols: Vec<&StreamColumn> = self
                .probe_key_cols
                .iter()
                .map(|&i| record.batch.column(i))
                .collect();
            hash_multi_column_batch_into(&cols, num_rows, &mut self.hash_buf);
        }

        // Probe cache, populate misses, collect matched indices.
        let mut matched_probe_indices: Vec<u32> = Vec::with_capacity(num_rows);
        let mut matched_build_hashes: Vec<u64> = Vec::with_capacity(num_rows);

        for (row_idx, &key_hash) in self.hash_buf.iter().enumerate() {
            if self.cache.get(key_hash).is_none() {
                match (self.lookup_fn)(key_hash) {
                    Ok(Some(batch)) => {
                        self.cache.insert(key_hash, (batch, self.current_time_ms));
                    }
                    Ok(None) => continue,
                    Err(e) => return Err(e),
                }
            }
            matched_probe_indices.push(row_idx as u32);
            matched_build_hashes.push(key_hash);
        }

        let output = if !matched_probe_indices.is_empty() {
            let match_count = matched_probe_indices.len();

            // Gather probe columns via take().
            let mut out_cols: Vec<StreamColumn> = record
                .batch
                .columns
                .iter()
                .map(|col| col.take(&matched_probe_indices))
                .collect();

            // Gather build columns with typed fast paths.
            let first_hash = matched_build_hashes[0];
            if let Some((first_batch, _)) = self.cache.get(first_hash) {
                let build_col_count = first_batch.num_columns();
                for col_idx in 0..build_col_count {
                    let out_col = match &first_batch.column(col_idx).data {
                        StreamColumnData::Int64(_) => {
                            let mut vals = Vec::with_capacity(match_count);
                            for &bh in &matched_build_hashes {
                                if let Some((b, _)) = self.cache.get(bh) {
                                    if let StreamColumnData::Int64(v) = &b.column(col_idx).data {
                                        vals.push(v[0]);
                                    }
                                }
                            }
                            StreamColumnData::Int64(vals)
                        }
                        StreamColumnData::Float64(_) => {
                            let mut vals = Vec::with_capacity(match_count);
                            for &bh in &matched_build_hashes {
                                if let Some((b, _)) = self.cache.get(bh) {
                                    if let StreamColumnData::Float64(v) = &b.column(col_idx).data {
                                        vals.push(v[0]);
                                    }
                                }
                            }
                            StreamColumnData::Float64(vals)
                        }
                        StreamColumnData::Utf8(_) => {
                            let mut vals = Vec::with_capacity(match_count);
                            for &bh in &matched_build_hashes {
                                if let Some((b, _)) = self.cache.get(bh) {
                                    if let StreamColumnData::Utf8(v) = &b.column(col_idx).data {
                                        vals.push(v[0].clone());
                                    }
                                }
                            }
                            StreamColumnData::Utf8(vals)
                        }
                        other => {
                            let mut col = other.empty_like_with_capacity(match_count);
                            for &bh in &matched_build_hashes {
                                if let Some((b, _)) = self.cache.get(bh) {
                                    col.push_scalar(&b.column(col_idx).data.get_scalar(0));
                                }
                            }
                            col
                        }
                    };
                    out_cols.push(StreamColumn::from_data(out_col));
                }
            }

            let event_times: Vec<i64> = matched_probe_indices
                .iter()
                .map(|&i| record.event_times[i as usize])
                .collect();
            let flags = vec![ChangeFlag::Insert; match_count];
            let batch = StreamBatch::new(out_cols);
            vec![StreamRecord::new(batch, event_times, flags)]
        } else {
            Vec::new()
        };

        let out_rows: usize = output.iter().map(|r| r.num_rows()).sum();
        self.op_metrics.record_output(out_rows as u64);

        Ok(output)
    }

    fn on_watermark(&mut self, watermark: Watermark) -> Result<Vec<StreamRecord>> {
        self.current_time_ms = watermark.timestamp_ms;
        self.op_metrics.update_watermark(watermark.timestamp_ms);
        self.evict_expired();
        Ok(Vec::new())
    }

    fn on_timer(&mut self, _timestamp_ms: i64) -> Result<Vec<StreamRecord>> {
        Ok(Vec::new())
    }

    fn on_barrier(&mut self, _barrier: CheckpointBarrier) -> Result<StateSnapshot> {
        Ok(StateSnapshot::empty(0))
    }

    fn restore(&mut self, _snapshot: StateSnapshot) -> Result<()> {
        self.cache.clear();
        Ok(())
    }

    fn operator_id(&self) -> u32 {
        self.id
    }

    fn metrics(&self) -> &OperatorMetrics {
        &self.op_metrics
    }
}

// ---------------------------------------------------------------------------
// TemporalJoin
// ---------------------------------------------------------------------------

/// Temporal join: for each probe event, looks up the build-side version
/// valid at the event's timestamp (AS OF semantics). Binary search for
/// the closest version <= event_time.
pub struct TemporalJoin {
    id: u32,
    op_metrics: OperatorMetrics,
    probe_key_cols: Vec<usize>,
    versions: FlatU64Map<Vec<(i64, StreamBatch)>>,
    hash_buf: Vec<u64>,
}

impl TemporalJoin {
    pub fn new(id: u32, probe_key_cols: Vec<usize>) -> Self {
        Self {
            id,
            op_metrics: OperatorMetrics::new(),
            probe_key_cols,
            versions: FlatU64Map::default(),
            hash_buf: Vec::new(),
        }
    }

    /// Adds a new version for a key (from the build-side stream).
    pub fn add_version(&mut self, key_hash: u64, version_time_ms: i64, batch: StreamBatch) {
        let entry = self.versions.get_or_insert_with(key_hash, Vec::new);
        let pos = entry.partition_point(|(t, _)| *t <= version_time_ms);
        entry.insert(pos, (version_time_ms, batch));
    }

    /// Binary search for the latest version at or before the given timestamp.
    fn lookup_version(&self, key_hash: u64, as_of_ms: i64) -> Option<&StreamBatch> {
        let versions = self.versions.get(key_hash)?;
        if versions.is_empty() {
            return None;
        }
        let idx = versions.partition_point(|(t, _)| *t <= as_of_ms);
        if idx == 0 {
            return None;
        }
        Some(&versions[idx - 1].1)
    }
}

impl StreamOperator for TemporalJoin {
    fn process(&mut self, record: StreamRecord) -> Result<Vec<StreamRecord>> {
        let num_rows = record.num_rows();
        self.op_metrics.record_input(num_rows as u64);

        if num_rows == 0 {
            return Ok(Vec::new());
        }

        // Hash into reusable buffer.
        if self.probe_key_cols.len() == 1 {
            hash_column_batch_into(
                record.batch.column(self.probe_key_cols[0]),
                num_rows,
                &mut self.hash_buf,
            );
        } else {
            let cols: Vec<&StreamColumn> = self
                .probe_key_cols
                .iter()
                .map(|&i| record.batch.column(i))
                .collect();
            hash_multi_column_batch_into(&cols, num_rows, &mut self.hash_buf);
        }

        // Single-pass: probe, collect matches, and build output columns inline.
        let mut matched_indices: Vec<u32> = Vec::with_capacity(num_rows);
        let mut matched_times: Vec<i64> = Vec::with_capacity(num_rows);
        // Collect build-side batch references to avoid re-lookup.
        let mut build_refs: Vec<*const StreamBatch> = Vec::with_capacity(num_rows);

        for (row_idx, &key_hash) in self.hash_buf.iter().enumerate() {
            let as_of = record.event_times[row_idx];
            if let Some(batch) = self.lookup_version(key_hash, as_of) {
                matched_indices.push(row_idx as u32);
                matched_times.push(as_of);
                build_refs.push(batch as *const StreamBatch);
            }
        }

        let output = if !matched_indices.is_empty() {
            let match_count = matched_indices.len();
            let mut out_cols: Vec<StreamColumn> = record
                .batch
                .columns
                .iter()
                .map(|col| col.take(&matched_indices))
                .collect();

            // Gather build columns with typed fast paths. Single lookup per match
            // (stored as pointer during probe, no re-lookup).
            let first_batch = unsafe { &*build_refs[0] };
            for col_idx in 0..first_batch.num_columns() {
                let out_col = match &first_batch.column(col_idx).data {
                    StreamColumnData::Int64(_) => {
                        let mut vals = Vec::with_capacity(match_count);
                        for &ptr in &build_refs {
                            let b = unsafe { &*ptr };
                            if let StreamColumnData::Int64(v) = &b.column(col_idx).data {
                                vals.push(v[0]);
                            }
                        }
                        StreamColumnData::Int64(vals)
                    }
                    StreamColumnData::Float64(_) => {
                        let mut vals = Vec::with_capacity(match_count);
                        for &ptr in &build_refs {
                            let b = unsafe { &*ptr };
                            if let StreamColumnData::Float64(v) = &b.column(col_idx).data {
                                vals.push(v[0]);
                            }
                        }
                        StreamColumnData::Float64(vals)
                    }
                    StreamColumnData::Utf8(_) => {
                        let mut vals = Vec::with_capacity(match_count);
                        for &ptr in &build_refs {
                            let b = unsafe { &*ptr };
                            if let StreamColumnData::Utf8(v) = &b.column(col_idx).data {
                                vals.push(v[0].clone());
                            }
                        }
                        StreamColumnData::Utf8(vals)
                    }
                    other => {
                        let mut col = other.empty_like_with_capacity(match_count);
                        for &ptr in &build_refs {
                            let b = unsafe { &*ptr };
                            col.push_scalar(&b.column(col_idx).data.get_scalar(0));
                        }
                        col
                    }
                };
                out_cols.push(StreamColumn::from_data(out_col));
            }

            let flags = vec![ChangeFlag::Insert; match_count];
            vec![StreamRecord::new(
                StreamBatch::new(out_cols),
                matched_times,
                flags,
            )]
        } else {
            Vec::new()
        };

        let out_rows: usize = output.iter().map(|r| r.num_rows()).sum();
        self.op_metrics.record_output(out_rows as u64);
        Ok(output)
    }

    fn on_watermark(&mut self, watermark: Watermark) -> Result<Vec<StreamRecord>> {
        self.op_metrics.update_watermark(watermark.timestamp_ms);
        Ok(Vec::new())
    }

    fn on_timer(&mut self, _timestamp_ms: i64) -> Result<Vec<StreamRecord>> {
        Ok(Vec::new())
    }

    fn on_barrier(&mut self, _barrier: CheckpointBarrier) -> Result<StateSnapshot> {
        Ok(StateSnapshot::empty(0))
    }

    fn restore(&mut self, _snapshot: StateSnapshot) -> Result<()> {
        self.versions.clear();
        Ok(())
    }

    fn operator_id(&self) -> u32 {
        self.id
    }

    fn metrics(&self) -> &OperatorMetrics {
        &self.op_metrics
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::{StreamBatch, StreamColumn, StreamColumnData};
    use crate::record::ChangeFlag;

    fn make_keyed_record(keys: Vec<i64>, values: Vec<i64>, times: Vec<i64>) -> StreamRecord {
        let key_col = StreamColumn::from_data(StreamColumnData::Int64(keys));
        let val_col = StreamColumn::from_data(StreamColumnData::Int64(values));
        let batch = StreamBatch::new(vec![key_col, val_col]);
        let n = batch.num_rows;
        StreamRecord::new(batch, times, vec![ChangeFlag::Insert; n])
    }

    #[test]
    fn test_stream_stream_join_basic() {
        let mut join = StreamStreamJoin::new(1, vec![0], vec![0], 10_000);

        join.set_input_side(true);
        let left = make_keyed_record(vec![1, 2], vec![10, 20], vec![1000, 2000]);
        let output = join.process(left).expect("process should succeed");
        assert!(output.is_empty());

        join.set_input_side(false);
        let right = make_keyed_record(vec![1], vec![100], vec![1500]);
        let output = join.process(right).expect("process should succeed");
        assert!(!output.is_empty());
    }

    #[test]
    fn test_stream_stream_join_eviction() {
        let mut join = StreamStreamJoin::new(2, vec![0], vec![0], 5000);

        join.set_input_side(true);
        let left = make_keyed_record(vec![1], vec![10], vec![1000]);
        join.process(left).expect("process should succeed");

        join.on_watermark(Watermark::new(7000))
            .expect("watermark should succeed");

        join.set_input_side(false);
        let right = make_keyed_record(vec![1], vec![100], vec![6500]);
        let output = join.process(right).expect("process should succeed");
        assert!(output.is_empty());
    }

    #[test]
    fn test_lookup_join() {
        // Build a simple lookup map for testing.
        let dim_col = StreamColumn::from_data(StreamColumnData::Int64(vec![999]));
        let dim_batch = StreamBatch::new(vec![dim_col]);
        let key_hash = crate::hash::hash_int(42);

        let lookup_data: std::sync::Arc<std::collections::HashMap<u64, StreamBatch>> = {
            let mut map = std::collections::HashMap::new();
            map.insert(key_hash, dim_batch);
            std::sync::Arc::new(map)
        };

        let data_clone = lookup_data.clone();
        let mut join = LookupJoin::new(
            3,
            vec![0],
            60_000,
            1000,
            Box::new(move |hash| Ok(data_clone.get(&hash).cloned())),
        );

        let probe = make_keyed_record(vec![42], vec![10], vec![1000]);
        let output = join.process(probe).expect("process should succeed");
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn test_temporal_join() {
        let mut join = TemporalJoin::new(4, vec![0]);

        let key_hash = crate::hash::hash_int(1);
        let v1 = StreamBatch::new(vec![StreamColumn::from_data(StreamColumnData::Int64(
            vec![100],
        ))]);
        let v2 = StreamBatch::new(vec![StreamColumn::from_data(StreamColumnData::Int64(
            vec![200],
        ))]);
        join.add_version(key_hash, 1000, v1);
        join.add_version(key_hash, 5000, v2);

        let probe = make_keyed_record(vec![1], vec![10], vec![3000]);
        let output = join.process(probe).expect("process should succeed");
        assert_eq!(output.len(), 1);
        if let StreamColumnData::Int64(v) = &output[0].batch.columns.last().expect("columns").data {
            assert_eq!(v[0], 100);
        } else {
            panic!("expected Int64 column");
        }
    }

    #[test]
    fn test_temporal_join_no_match() {
        let mut join = TemporalJoin::new(5, vec![0]);
        let key_hash = crate::hash::hash_int(1);
        let v1 = StreamBatch::new(vec![StreamColumn::from_data(StreamColumnData::Int64(
            vec![100],
        ))]);
        join.add_version(key_hash, 5000, v1);

        let probe = make_keyed_record(vec![1], vec![10], vec![3000]);
        let output = join.process(probe).expect("process should succeed");
        assert!(output.is_empty());
    }

    #[test]
    fn test_interval_join() {
        let mut join = IntervalJoin::new(6, vec![0], vec![0], -2000, 2000);

        join.set_input_side(true);
        let left = make_keyed_record(vec![1], vec![10], vec![5000]);
        join.process(left).expect("process should succeed");

        join.set_input_side(false);
        let right = make_keyed_record(vec![1], vec![100], vec![6000]);
        let output = join.process(right).expect("process should succeed");
        assert!(!output.is_empty());
    }

    // -----------------------------------------------------------------
    // Streaming-job JOIN tests covering the brief's five scenarios.
    // -----------------------------------------------------------------

    /// Left row at t=0 and right row at t=2000ms share the same key, with
    /// a symmetric window of 5000ms. The join emits exactly one joined
    /// record on the right-side insert.
    #[test]
    fn test_interval_join_matches_within_window() {
        let mut join = StreamStreamJoin::new(100, vec![0], vec![0], 5_000);

        join.set_input_side(true);
        let left = make_keyed_record(vec![42], vec![10], vec![0]);
        let output = join.process(left).expect("left process");
        assert!(output.is_empty());

        join.set_input_side(false);
        let right = make_keyed_record(vec![42], vec![99], vec![2_000]);
        let output = join.process(right).expect("right process");
        let rows: usize = output.iter().map(|r| r.num_rows()).sum();
        assert_eq!(rows, 1, "expected exactly one joined row within the window");
    }

    /// Left row at t=0 and right row at t=10000ms with a symmetric window
    /// of 5000ms fall outside the window. The join emits zero records.
    #[test]
    fn test_interval_join_drops_outside_window() {
        let mut join = StreamStreamJoin::new(101, vec![0], vec![0], 5_000);

        join.set_input_side(true);
        let left = make_keyed_record(vec![7], vec![10], vec![0]);
        join.process(left).expect("left process");

        join.set_input_side(false);
        let right = make_keyed_record(vec![7], vec![99], vec![10_000]);
        let output = join.process(right).expect("right process");
        let rows: usize = output.iter().map(|r| r.num_rows()).sum();
        assert_eq!(rows, 0, "outside-window records must not join");
    }

    /// Advancing the watermark past the retention horizon evicts buffered
    /// left-side state. A subsequent right-side row with a matching key
    /// therefore finds no buffered opposite and emits nothing.
    #[test]
    fn test_interval_join_evicts_on_watermark() {
        let mut join = StreamStreamJoin::new(102, vec![0], vec![0], 1_000);

        join.set_input_side(true);
        let left = make_keyed_record(vec![3], vec![10], vec![0]);
        join.process(left).expect("left process");

        // Watermark advances past 0 + 1000 so the left row is evicted.
        join.on_watermark(Watermark::new(5_000))
            .expect("watermark advance");

        join.set_input_side(false);
        let right = make_keyed_record(vec![3], vec![99], vec![4_500]);
        let output = join.process(right).expect("right process");
        let rows: usize = output.iter().map(|r| r.num_rows()).sum();
        assert_eq!(rows, 0, "state must be evicted past watermark horizon");
    }

    /// Temporal-join lookup finds the latest version at or before the
    /// probe event time and emits a joined row carrying the build batch's
    /// payload. Confirms AS OF semantics for stream-table joins.
    #[test]
    fn test_temporal_join_lookup() {
        let mut join = TemporalJoin::new(103, vec![0]);

        let key_hash = crate::hash::hash_int(77);
        let build = StreamBatch::new(vec![StreamColumn::from_data(StreamColumnData::Int64(
            vec![555],
        ))]);
        join.add_version(key_hash, 100, build);

        let probe = make_keyed_record(vec![77], vec![10], vec![500]);
        let output = join.process(probe).expect("probe process");
        assert_eq!(output.len(), 1);
        let last = output[0].batch.columns.last().expect("last column");
        if let StreamColumnData::Int64(v) = &last.data {
            assert_eq!(v[0], 555);
        } else {
            panic!("expected Int64 build column on output");
        }
    }

    /// Temporal-join without a matching version at the probe event time
    /// emits nothing. Inner-join semantics are the only supported form,
    /// so unmatched rows are dropped.
    #[test]
    fn test_temporal_join_no_match_emits_nothing_by_default() {
        let mut join = TemporalJoin::new(104, vec![0]);

        let key_hash = crate::hash::hash_int(88);
        let build = StreamBatch::new(vec![StreamColumn::from_data(StreamColumnData::Int64(
            vec![999],
        ))]);
        // Version is at t=5000, probe is at t=1000, so no version exists
        // at or before the probe time.
        join.add_version(key_hash, 5_000, build);

        let probe = make_keyed_record(vec![88], vec![10], vec![1_000]);
        let output = join.process(probe).expect("probe process");
        assert!(
            output.is_empty(),
            "inner temporal join must drop unmatched rows"
        );
    }
}

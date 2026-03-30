//! Streaming operator trait and built-in operator implementations.
//!
//! StreamOperator defines the core interface for push-based processing.
//! Built-in operators: WindowAggregateOperator (fires windows on watermark),
//! StreamFilterOperator (batch-optimized predicate evaluation),
//! StreamProjectOperator (zero-copy column selection),
//! StreamKeyByOperator (pre-hashes key columns),
//! and OperatorChain (DAG of operators connected by SPSC channels).

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

use zyron_common::Result;

use crate::accumulator::StreamAccumulator;
use crate::checkpoint::CheckpointBarrier;
use crate::column::{ScalarValue, StreamBatch, StreamColumn, StreamColumnData};
use crate::hash::{
    FlatU64Map, hash_column_batch, hash_column_batch_into, hash_multi_column_batch,
    hash_multi_column_batch_into,
};
use crate::record::{ChangeFlag, StreamRecord};
use crate::state::StateSnapshot;
use crate::watermark::Watermark;
use crate::window::WindowRange;

// ---------------------------------------------------------------------------
// OperatorMetrics
// ---------------------------------------------------------------------------

/// Lock-free operator metrics. Read by metrics views without blocking
/// the operator thread.
pub struct OperatorMetrics {
    pub records_in: AtomicU64,
    pub records_out: AtomicU64,
    pub processing_time_ns: AtomicU64,
    pub watermark_ms: AtomicI64,
}

impl OperatorMetrics {
    pub fn new() -> Self {
        Self {
            records_in: AtomicU64::new(0),
            records_out: AtomicU64::new(0),
            processing_time_ns: AtomicU64::new(0),
            watermark_ms: AtomicI64::new(i64::MIN),
        }
    }

    #[inline]
    pub fn record_input(&self, count: u64) {
        self.records_in.fetch_add(count, Ordering::Relaxed);
    }

    #[inline]
    pub fn record_output(&self, count: u64) {
        self.records_out.fetch_add(count, Ordering::Relaxed);
    }

    #[inline]
    pub fn add_processing_time(&self, ns: u64) {
        self.processing_time_ns.fetch_add(ns, Ordering::Relaxed);
    }

    #[inline]
    pub fn update_watermark(&self, wm: i64) {
        self.watermark_ms.fetch_max(wm, Ordering::Relaxed);
    }
}

impl Default for OperatorMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// StreamOperator trait
// ---------------------------------------------------------------------------

/// Core trait for streaming operators in the DAG.
/// Each operator is single-threaded. All internal state uses thread-local
/// data structures (hashbrown::HashMap, FlatU64Map). Cross-thread
/// communication is via SPSC ring buffers only.
pub trait StreamOperator: Send {
    /// Process a micro-batch of records.
    fn process(&mut self, record: StreamRecord) -> Result<Vec<StreamRecord>>;

    /// Called when a watermark advances. May trigger window emission.
    fn on_watermark(&mut self, watermark: Watermark) -> Result<Vec<StreamRecord>>;

    /// Called when a timer fires (event-time or processing-time).
    fn on_timer(&mut self, timestamp_ms: i64) -> Result<Vec<StreamRecord>>;

    /// Called when a checkpoint barrier arrives. Returns the operator's
    /// state snapshot for checkpointing.
    fn on_barrier(&mut self, barrier: CheckpointBarrier) -> Result<StateSnapshot>;

    /// Restores operator state from a checkpoint snapshot.
    fn restore(&mut self, snapshot: StateSnapshot) -> Result<()>;

    /// Returns the operator's unique ID.
    fn operator_id(&self) -> u32;

    /// Returns reference to the operator's metrics.
    fn metrics(&self) -> &OperatorMetrics;
}

// ---------------------------------------------------------------------------
// WindowAggregateOperator
// ---------------------------------------------------------------------------

/// Window aggregate operator that accumulates values per (group_key, window)
/// and fires results when the watermark passes window end.
/// Uses FlatU64Map for O(1) lookup with pre-computed hash keys.
pub struct WindowAggregateOperator {
    id: u32,
    op_metrics: OperatorMetrics,
    /// Key column indices to group by.
    key_columns: Vec<usize>,
    /// Column index to aggregate.
    agg_column: usize,
    /// Factory function to create accumulators.
    accumulator_factory: Box<dyn Fn() -> Box<dyn StreamAccumulator> + Send>,
    /// Per-group-key accumulator state: key_hash -> [(window, accumulator)].
    state: FlatU64Map<Vec<(WindowRange, Box<dyn StreamAccumulator>)>>,
    /// Window assigner (determines which windows an event belongs to).
    window_assigner: Box<dyn crate::window::WindowAssigner>,
    /// Current watermark.
    current_watermark: i64,
    /// Reusable buffer for window assignment (avoids per-row allocation).
    window_buf: Vec<WindowRange>,
    /// Reusable buffer for key hashes (avoids per-call allocation).
    hash_buf: Vec<u64>,
}

impl WindowAggregateOperator {
    pub fn new(
        id: u32,
        key_columns: Vec<usize>,
        agg_column: usize,
        accumulator_factory: Box<dyn Fn() -> Box<dyn StreamAccumulator> + Send>,
        window_assigner: Box<dyn crate::window::WindowAssigner>,
    ) -> Self {
        Self {
            id,
            op_metrics: OperatorMetrics::new(),
            key_columns,
            agg_column,
            accumulator_factory,
            state: FlatU64Map::default(),
            window_assigner,
            current_watermark: i64::MIN,
            window_buf: Vec::new(),
            hash_buf: Vec::new(),
        }
    }

    /// Fires all windows whose end time is at or below the watermark.
    fn fire_windows(&mut self, watermark_ms: i64) -> Vec<StreamRecord> {
        let mut output_keys = Vec::new();
        let mut output_windows = Vec::new();
        let mut output_values = Vec::new();

        let mut keys_to_clean = Vec::new();

        self.state.iter_mut(|key_hash, windows| {
            let mut i = 0;
            while i < windows.len() {
                if windows[i].0.end_ms <= watermark_ms {
                    let (window, acc) = windows.swap_remove(i);
                    output_keys.push(key_hash);
                    output_windows.push(window);
                    output_values.push(acc.finalize());
                } else {
                    i += 1;
                }
            }
            if windows.is_empty() {
                keys_to_clean.push(key_hash);
            }
        });

        for key in keys_to_clean {
            self.state.remove(key);
        }

        if output_keys.is_empty() {
            return Vec::new();
        }

        // Build output batch with columns: key_hash, window_start, window_end, agg_value.
        let n = output_keys.len();
        let key_col = StreamColumn::from_data(StreamColumnData::UInt64(
            output_keys.iter().map(|k| *k).collect(),
        ));
        let start_col = StreamColumn::from_data(StreamColumnData::Int64(
            output_windows.iter().map(|w| w.start_ms).collect(),
        ));
        let end_col = StreamColumn::from_data(StreamColumnData::Int64(
            output_windows.iter().map(|w| w.end_ms).collect(),
        ));

        // Convert aggregated values to a column.
        let agg_values: Vec<f64> = output_values
            .iter()
            .map(|v| match v {
                ScalarValue::Float64(f) => *f,
                ScalarValue::Int64(i) => *i as f64,
                _ => 0.0,
            })
            .collect();
        let agg_col = StreamColumn::from_data(StreamColumnData::Float64(agg_values));

        let batch = StreamBatch::new(vec![key_col, start_col, end_col, agg_col]);
        let event_times = output_windows.iter().map(|w| w.end_ms).collect();
        let flags = vec![ChangeFlag::Insert; n];

        self.op_metrics.record_output(n as u64);

        vec![StreamRecord::new(batch, event_times, flags)]
    }
}

impl StreamOperator for WindowAggregateOperator {
    fn process(&mut self, record: StreamRecord) -> Result<Vec<StreamRecord>> {
        let start = std::time::Instant::now();
        self.op_metrics.record_input(record.num_rows() as u64);

        let num_rows = record.num_rows();
        if num_rows == 0 {
            return Ok(Vec::new());
        }

        // Compute key hashes into reusable buffer.
        if self.key_columns.is_empty() {
            self.hash_buf.clear();
            self.hash_buf.resize(num_rows, 0u64);
        } else if self.key_columns.len() == 1 {
            hash_column_batch_into(
                record.batch.column(self.key_columns[0]),
                num_rows,
                &mut self.hash_buf,
            );
        } else {
            let cols: Vec<&StreamColumn> = self
                .key_columns
                .iter()
                .map(|&i| record.batch.column(i))
                .collect();
            hash_multi_column_batch_into(&cols, num_rows, &mut self.hash_buf);
        };

        let agg_col = record.batch.column(self.agg_column);

        for row in 0..num_rows {
            let event_time = record.event_times[row];
            let key_hash = self.hash_buf[row];
            self.window_assigner
                .assign_windows_into(event_time, &mut self.window_buf);

            let entry = self.state.get_or_insert_with(key_hash, Vec::new);

            for window in &self.window_buf {
                // Find or create accumulator for this (key, window).
                // Search from end. Most recently added window is most likely to match.
                let mut found_idx = None;
                for j in (0..entry.len()).rev() {
                    if entry[j].0 == *window {
                        found_idx = Some(j);
                        break;
                    }
                }
                match found_idx {
                    Some(j) => {
                        entry[j].1.update_typed(agg_col, row);
                    }
                    None => {
                        let acc = (self.accumulator_factory)();
                        entry.push((*window, acc));
                        // Update after insertion to avoid mut binding.
                        let last = entry.last_mut().expect("just pushed");
                        last.1.update_typed(agg_col, row);
                    }
                }
            }
        }

        let elapsed = start.elapsed().as_nanos() as u64;
        self.op_metrics.add_processing_time(elapsed);
        Ok(Vec::new())
    }

    fn on_watermark(&mut self, watermark: Watermark) -> Result<Vec<StreamRecord>> {
        self.current_watermark = watermark.timestamp_ms;
        self.op_metrics.update_watermark(watermark.timestamp_ms);
        let output = self.fire_windows(watermark.timestamp_ms);
        Ok(output)
    }

    fn on_timer(&mut self, _timestamp_ms: i64) -> Result<Vec<StreamRecord>> {
        Ok(Vec::new())
    }

    fn on_barrier(&mut self, _barrier: CheckpointBarrier) -> Result<StateSnapshot> {
        // Serialize accumulator state for checkpointing.
        let mut data = Vec::new();
        self.state.iter(|key_hash, windows| {
            for (window, acc) in windows {
                let mut key_bytes = Vec::with_capacity(24);
                key_bytes.extend_from_slice(&key_hash.to_le_bytes());
                key_bytes.extend_from_slice(&window.start_ms.to_le_bytes());
                key_bytes.extend_from_slice(&window.end_ms.to_le_bytes());
                let val_bytes = acc.serialize();
                data.push((b"window_agg".to_vec(), key_bytes, val_bytes));
            }
        });
        Ok(StateSnapshot {
            data,
            snapshot_id: 0,
        })
    }

    fn restore(&mut self, snapshot: StateSnapshot) -> Result<()> {
        self.state.clear();
        for (_, key_bytes, val_bytes) in &snapshot.data {
            if key_bytes.len() < 24 {
                continue;
            }
            let key_hash = u64::from_le_bytes([
                key_bytes[0],
                key_bytes[1],
                key_bytes[2],
                key_bytes[3],
                key_bytes[4],
                key_bytes[5],
                key_bytes[6],
                key_bytes[7],
            ]);
            let start = i64::from_le_bytes([
                key_bytes[8],
                key_bytes[9],
                key_bytes[10],
                key_bytes[11],
                key_bytes[12],
                key_bytes[13],
                key_bytes[14],
                key_bytes[15],
            ]);
            let end = i64::from_le_bytes([
                key_bytes[16],
                key_bytes[17],
                key_bytes[18],
                key_bytes[19],
                key_bytes[20],
                key_bytes[21],
                key_bytes[22],
                key_bytes[23],
            ]);
            let window = WindowRange::new(start, end);
            let mut acc = (self.accumulator_factory)();
            // Restore accumulator state from the serialized checkpoint bytes.
            acc.deserialize(val_bytes);
            let entry = self.state.get_or_insert_with(key_hash, Vec::new);
            entry.push((window, acc));
        }
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
// StreamFilterOperator
// ---------------------------------------------------------------------------

/// Filters rows using a batch-optimized predicate.
/// Builds a boolean mask over the entire batch, then calls batch.filter(mask).
pub struct StreamFilterOperator {
    id: u32,
    op_metrics: OperatorMetrics,
    /// Predicate applied to each row. Returns true to keep the row.
    predicate: Box<dyn Fn(&StreamBatch, usize) -> bool + Send>,
}

impl StreamFilterOperator {
    pub fn new(id: u32, predicate: Box<dyn Fn(&StreamBatch, usize) -> bool + Send>) -> Self {
        Self {
            id,
            op_metrics: OperatorMetrics::new(),
            predicate,
        }
    }
}

impl StreamOperator for StreamFilterOperator {
    fn process(&mut self, record: StreamRecord) -> Result<Vec<StreamRecord>> {
        let start = std::time::Instant::now();
        let num_rows = record.num_rows();
        self.op_metrics.record_input(num_rows as u64);

        if num_rows == 0 {
            return Ok(Vec::new());
        }

        let mut mask = vec![false; num_rows];
        for i in 0..num_rows {
            mask[i] = (self.predicate)(&record.batch, i);
        }

        let filtered = record.filter(&mask);
        let out_rows = filtered.num_rows() as u64;
        self.op_metrics.record_output(out_rows);

        let elapsed = start.elapsed().as_nanos() as u64;
        self.op_metrics.add_processing_time(elapsed);

        if filtered.is_empty() {
            Ok(Vec::new())
        } else {
            Ok(vec![filtered])
        }
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
// StreamProjectOperator
// ---------------------------------------------------------------------------

/// Projects (selects) specific columns from a batch.
/// Zero-copy column selection via index list.
pub struct StreamProjectOperator {
    id: u32,
    op_metrics: OperatorMetrics,
    /// Indices of columns to project.
    column_indices: Vec<usize>,
}

impl StreamProjectOperator {
    pub fn new(id: u32, column_indices: Vec<usize>) -> Self {
        Self {
            id,
            op_metrics: OperatorMetrics::new(),
            column_indices,
        }
    }
}

impl StreamOperator for StreamProjectOperator {
    fn process(&mut self, record: StreamRecord) -> Result<Vec<StreamRecord>> {
        let start = std::time::Instant::now();
        let num_rows = record.num_rows();
        self.op_metrics.record_input(num_rows as u64);

        let projected_columns: Vec<StreamColumn> = self
            .column_indices
            .iter()
            .map(|&i| record.batch.column(i).clone())
            .collect();

        let batch = StreamBatch::new(projected_columns);
        let result = StreamRecord {
            batch,
            event_times: record.event_times,
            keys: record.keys,
            change_flags: record.change_flags,
        };

        self.op_metrics.record_output(num_rows as u64);
        let elapsed = start.elapsed().as_nanos() as u64;
        self.op_metrics.add_processing_time(elapsed);

        Ok(vec![result])
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
// StreamKeyByOperator
// ---------------------------------------------------------------------------

/// Computes key hashes from specified columns and stores them in
/// StreamRecord.keys for downstream operators (joins, aggregates).
pub struct StreamKeyByOperator {
    id: u32,
    op_metrics: OperatorMetrics,
    /// Key column indices.
    key_columns: Vec<usize>,
}

impl StreamKeyByOperator {
    pub fn new(id: u32, key_columns: Vec<usize>) -> Self {
        Self {
            id,
            op_metrics: OperatorMetrics::new(),
            key_columns,
        }
    }
}

impl StreamOperator for StreamKeyByOperator {
    fn process(&mut self, mut record: StreamRecord) -> Result<Vec<StreamRecord>> {
        let start = std::time::Instant::now();
        let num_rows = record.num_rows();
        self.op_metrics.record_input(num_rows as u64);

        if num_rows > 0 {
            let hashes = if self.key_columns.len() == 1 {
                hash_column_batch(record.batch.column(self.key_columns[0]), num_rows)
            } else {
                let cols: Vec<&StreamColumn> = self
                    .key_columns
                    .iter()
                    .map(|&i| record.batch.column(i))
                    .collect();
                hash_multi_column_batch(&cols, num_rows)
            };
            record.keys = Some(hashes);
        }

        self.op_metrics.record_output(num_rows as u64);
        let elapsed = start.elapsed().as_nanos() as u64;
        self.op_metrics.add_processing_time(elapsed);

        Ok(vec![record])
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
// OperatorChain
// ---------------------------------------------------------------------------

/// A chain of operators connected by SPSC channels.
/// Each operator runs on its own thread. Records flow through the chain
/// in order. Shutdown is signaled by sending an empty (poison pill) record.
pub struct OperatorChain {
    /// Operators in topological order.
    operators: Vec<Box<dyn StreamOperator>>,
}

impl OperatorChain {
    pub fn new() -> Self {
        Self {
            operators: Vec::new(),
        }
    }

    /// Adds an operator to the end of the chain.
    pub fn add_operator(&mut self, operator: Box<dyn StreamOperator>) {
        self.operators.push(operator);
    }

    /// Number of operators in the chain.
    pub fn len(&self) -> usize {
        self.operators.len()
    }

    /// Returns true if the chain has no operators.
    pub fn is_empty(&self) -> bool {
        self.operators.is_empty()
    }

    /// Processes a record through all operators in sequence.
    /// This is the synchronous single-threaded execution path.
    pub fn process(&mut self, record: StreamRecord) -> Result<Vec<StreamRecord>> {
        let mut current = vec![record];

        for operator in &mut self.operators {
            let mut next = Vec::new();
            for rec in current {
                let results = operator.process(rec)?;
                next.extend(results);
            }
            current = next;
        }

        Ok(current)
    }

    /// Propagates a watermark through all operators.
    pub fn on_watermark(&mut self, watermark: Watermark) -> Result<Vec<StreamRecord>> {
        let mut output = Vec::new();
        for operator in &mut self.operators {
            let results = operator.on_watermark(watermark)?;
            output.extend(results);
        }
        Ok(output)
    }

    /// Propagates a checkpoint barrier through all operators.
    /// Returns state snapshots from each operator.
    pub fn on_barrier(&mut self, barrier: CheckpointBarrier) -> Result<Vec<(u32, StateSnapshot)>> {
        let mut snapshots = Vec::new();
        for operator in &mut self.operators {
            let snap = operator.on_barrier(barrier)?;
            snapshots.push((operator.operator_id(), snap));
        }
        Ok(snapshots)
    }

    /// Returns metrics for all operators in the chain.
    pub fn operator_ids(&self) -> Vec<u32> {
        self.operators.iter().map(|op| op.operator_id()).collect()
    }
}

impl Default for OperatorChain {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accumulator::SumAccumulator;
    use crate::column::{StreamBatch, StreamColumn, StreamColumnData};
    use crate::record::ChangeFlag;
    use crate::window::TumblingWindowAssigner;

    fn make_record(values: Vec<i64>, times: Vec<i64>) -> StreamRecord {
        let col = StreamColumn::from_data(StreamColumnData::Int64(values));
        let batch = StreamBatch::new(vec![col]);
        let n = batch.num_rows;
        StreamRecord::new(batch, times, vec![ChangeFlag::Insert; n])
    }

    #[test]
    fn test_filter_operator() {
        let mut op = StreamFilterOperator::new(
            1,
            Box::new(|batch, row| {
                if let StreamColumnData::Int64(v) = &batch.column(0).data {
                    v[row] > 5
                } else {
                    false
                }
            }),
        );

        let record = make_record(vec![1, 10, 3, 20], vec![0, 1000, 2000, 3000]);
        let results = op.process(record).expect("process should succeed");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].num_rows(), 2);
    }

    #[test]
    fn test_project_operator() {
        let mut op = StreamProjectOperator::new(2, vec![0]);
        let col0 = StreamColumn::from_data(StreamColumnData::Int64(vec![1, 2, 3]));
        let col1 = StreamColumn::from_data(StreamColumnData::Float64(vec![1.0, 2.0, 3.0]));
        let batch = StreamBatch::new(vec![col0, col1]);
        let record = StreamRecord::new_insert(batch, vec![0, 1000, 2000]);

        let results = op.process(record).expect("process should succeed");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].batch.num_columns(), 1);
        assert_eq!(results[0].num_rows(), 3);
    }

    #[test]
    fn test_key_by_operator() {
        let mut op = StreamKeyByOperator::new(3, vec![0]);
        let record = make_record(vec![10, 20, 30], vec![0, 1000, 2000]);
        assert!(record.keys.is_none());

        let results = op.process(record).expect("process should succeed");
        assert_eq!(results.len(), 1);
        assert!(results[0].keys.is_some());
        let keys = results[0].keys.as_ref().expect("keys should be set");
        assert_eq!(keys.len(), 3);
        // Different values should produce different hashes.
        assert_ne!(keys[0], keys[1]);
    }

    #[test]
    fn test_window_aggregate_operator() {
        let mut op = WindowAggregateOperator::new(
            4,
            vec![], // No group-by key, single global group.
            0,      // Aggregate column 0.
            Box::new(|| Box::new(SumAccumulator::new())),
            Box::new(TumblingWindowAssigner::new(10_000)),
        );

        // Add events in window [0, 10000).
        let record = make_record(vec![10, 20, 30], vec![1000, 5000, 9000]);
        let results = op.process(record).expect("process should succeed");
        assert!(results.is_empty()); // No output until watermark fires.

        // Advance watermark past window end.
        let results = op
            .on_watermark(Watermark::new(10_000))
            .expect("watermark should succeed");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].num_rows(), 1);
        // Sum should be 60.0 (10 + 20 + 30).
        if let StreamColumnData::Float64(v) = &results[0].batch.column(3).data {
            assert!((v[0] - 60.0).abs() < 0.01);
        } else {
            panic!("expected Float64 column");
        }
    }

    #[test]
    fn test_operator_chain() {
        let filter = StreamFilterOperator::new(
            1,
            Box::new(|batch, row| {
                if let StreamColumnData::Int64(v) = &batch.column(0).data {
                    v[row] > 5
                } else {
                    false
                }
            }),
        );

        let mut chain = OperatorChain::new();
        chain.add_operator(Box::new(filter));
        assert_eq!(chain.len(), 1);

        let record = make_record(vec![1, 10, 3, 20, 2], vec![0, 1000, 2000, 3000, 4000]);
        let results = chain.process(record).expect("chain process should succeed");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].num_rows(), 2);
    }

    #[test]
    fn test_operator_metrics() {
        let metrics = OperatorMetrics::new();
        metrics.record_input(10);
        metrics.record_output(5);
        metrics.add_processing_time(1000);
        metrics.update_watermark(5000);

        assert_eq!(metrics.records_in.load(Ordering::Relaxed), 10);
        assert_eq!(metrics.records_out.load(Ordering::Relaxed), 5);
        assert_eq!(metrics.processing_time_ns.load(Ordering::Relaxed), 1000);
        assert_eq!(metrics.watermark_ms.load(Ordering::Relaxed), 5000);
    }
}

// -----------------------------------------------------------------------------
// Dead-letter queue for Zyron-to-Zyron sink failures
// -----------------------------------------------------------------------------
//
// When a sink write fails after retry_max_attempts, the failing row is routed
// to a local DLQ table. The DLQ holds a bounded buffer of FailedRow records,
// exposes replay and eviction operations, and tracks basic counters. The
// queue itself is an in-memory FIFO backed by a parking_lot Mutex, suitable
// for process-local failure capture. The LocalSink trait lets the caller
// attach whatever persistent store is appropriate, including the existing
// ZyronRowSink path that lands rows into a local heap file.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use parking_lot::Mutex;
use zyron_common::{Result, ZyronError};

// -----------------------------------------------------------------------------
// FailedRow
// -----------------------------------------------------------------------------

/// Record of a row that could not be written to the remote sink.
#[derive(Debug, Clone)]
pub struct FailedRow {
    pub received_at: i64,
    pub failed_at: i64,
    pub error_class: String,
    pub error_message: String,
    pub source_table_id: u32,
    pub source_commit_version: u64,
    pub source_row_bytes: Vec<u8>,
    pub attempt_count: u32,
}

impl FailedRow {
    /// Creates a new FailedRow, filling received_at and failed_at with the
    /// current system time when the caller does not care about the exact
    /// wall clock.
    pub fn new(
        error_class: &str,
        error_message: String,
        source_table_id: u32,
        source_commit_version: u64,
        source_row_bytes: Vec<u8>,
        attempt_count: u32,
    ) -> Self {
        let now = now_millis();
        Self {
            received_at: now,
            failed_at: now,
            error_class: error_class.to_string(),
            error_message,
            source_table_id,
            source_commit_version,
            source_row_bytes,
            attempt_count,
        }
    }
}

fn now_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

// -----------------------------------------------------------------------------
// LocalSink trait
// -----------------------------------------------------------------------------

/// Pluggable persistent backing for a DLQ. The buffer ships drained rows to
/// this implementation. For production the caller wires this to the local
/// heap insert path via ZyronRowSink. In tests a VecLocalSink captures rows
/// for assertions.
pub trait LocalSink: Send + Sync {
    fn write_rows(&self, rows: &[FailedRow]) -> Result<()>;
}

/// Simple in-memory LocalSink used by tests and as a fall-through when the
/// caller has no persistent store yet. Rows are appended to an internal Vec.
pub struct VecLocalSink {
    inner: Mutex<Vec<FailedRow>>,
}

impl VecLocalSink {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Vec::new()),
        }
    }

    pub fn snapshot(&self) -> Vec<FailedRow> {
        self.inner.lock().clone()
    }

    pub fn len(&self) -> usize {
        self.inner.lock().len()
    }
}

impl Default for VecLocalSink {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalSink for VecLocalSink {
    fn write_rows(&self, rows: &[FailedRow]) -> Result<()> {
        let mut g = self.inner.lock();
        g.extend_from_slice(rows);
        Ok(())
    }
}

// -----------------------------------------------------------------------------
// DlqMetrics
// -----------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct DlqMetrics {
    pub rows_received: AtomicU64,
    pub rows_evicted: AtomicU64,
    pub rows_replayed: AtomicU64,
    pub write_errors: AtomicU64,
}

// -----------------------------------------------------------------------------
// Replay target trait
// -----------------------------------------------------------------------------

/// Pluggable target for replay. A real replay uses a ZyronSinkClient.
pub trait ReplayTarget: Send + Sync {
    fn replay_row(&self, row: &FailedRow) -> Result<()>;
}

// -----------------------------------------------------------------------------
// DeadLetterQueue
// -----------------------------------------------------------------------------

/// FIFO buffer of FailedRow records. Uses a parking_lot Mutex to serialize
/// writers. When the buffer reaches `max_rows`, subsequent writes evict the
/// oldest entry. Rows are forwarded to the configured LocalSink so the
/// caller may stream them to persistent storage.
pub struct DeadLetterQueue {
    table_name: String,
    max_rows: u64,
    rows: Mutex<std::collections::VecDeque<FailedRow>>,
    local_sink: Arc<dyn LocalSink>,
    metrics: Arc<DlqMetrics>,
}

impl DeadLetterQueue {
    pub fn new(table_name: String, max_rows: u64, local_sink: Arc<dyn LocalSink>) -> Self {
        Self {
            table_name,
            max_rows: max_rows.max(1),
            rows: Mutex::new(std::collections::VecDeque::new()),
            local_sink,
            metrics: Arc::new(DlqMetrics::default()),
        }
    }

    pub fn table_name(&self) -> &str {
        &self.table_name
    }

    pub fn metrics(&self) -> Arc<DlqMetrics> {
        Arc::clone(&self.metrics)
    }

    /// Writes a failed row into the queue. Evicts the oldest entry first if
    /// the buffer is full.
    pub fn write(&self, failed_row: FailedRow) -> Result<()> {
        let to_persist = failed_row.clone();
        {
            let mut g = self.rows.lock();
            while g.len() as u64 >= self.max_rows {
                if g.pop_front().is_some() {
                    self.metrics.rows_evicted.fetch_add(1, Ordering::Relaxed);
                }
            }
            g.push_back(failed_row);
        }
        self.metrics.rows_received.fetch_add(1, Ordering::Relaxed);
        match self.local_sink.write_rows(&[to_persist]) {
            Ok(()) => Ok(()),
            Err(e) => {
                self.metrics.write_errors.fetch_add(1, Ordering::Relaxed);
                Err(e)
            }
        }
    }

    /// Returns the number of rows currently held in the in-memory buffer.
    pub fn count(&self) -> u64 {
        self.rows.lock().len() as u64
    }

    /// Drops the oldest entry. No-op on an empty queue.
    pub fn evict_oldest(&self) -> Result<()> {
        let mut g = self.rows.lock();
        if g.pop_front().is_some() {
            self.metrics.rows_evicted.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

    /// Iterates over the buffer and forwards every row to the target. Rows
    /// older than the supplied cutoff are skipped. On the first replay error
    /// the remaining rows stay in the queue and the error is returned.
    pub fn replay(&self, target: &dyn ReplayTarget, since_epoch_ms: i64) -> Result<u64> {
        // Drain into a local snapshot first so the lock is held only briefly.
        let snapshot: Vec<FailedRow> = {
            let g = self.rows.lock();
            g.iter()
                .filter(|r| r.received_at >= since_epoch_ms)
                .cloned()
                .collect()
        };
        let mut count: u64 = 0;
        for row in &snapshot {
            target.replay_row(row)?;
            count += 1;
        }
        self.metrics
            .rows_replayed
            .fetch_add(count, Ordering::Relaxed);
        Ok(count)
    }

    /// Takes the full buffer, returning the rows and leaving the queue empty.
    pub fn drain_all(&self) -> Vec<FailedRow> {
        let mut g = self.rows.lock();
        g.drain(..).collect()
    }
}

// -----------------------------------------------------------------------------
// Convenience constructors
// -----------------------------------------------------------------------------

impl DeadLetterQueue {
    /// Convenience constructor that wires the DLQ to a VecLocalSink. Returns
    /// the Arc'd VecLocalSink so tests can inspect captured rows.
    pub fn with_vec_sink(table_name: &str, max_rows: u64) -> (Self, Arc<VecLocalSink>) {
        let sink = Arc::new(VecLocalSink::new());
        let dlq = DeadLetterQueue::new(table_name.to_string(), max_rows, sink.clone());
        (dlq, sink)
    }
}

// -----------------------------------------------------------------------------
// Failure wrapper helpers
// -----------------------------------------------------------------------------

/// Builds a FailedRow from a source row payload and the outcome of the most
/// recent attempt. The helper avoids the caller having to import SystemTime.
pub fn make_failed_row(
    error_class: &str,
    error_message: impl Into<String>,
    source_table_id: u32,
    source_commit_version: u64,
    source_row_bytes: Vec<u8>,
    attempt_count: u32,
) -> FailedRow {
    FailedRow::new(
        error_class,
        error_message.into(),
        source_table_id,
        source_commit_version,
        source_row_bytes,
        attempt_count,
    )
}

/// ReplayTarget shim around a closure. Lets callers plug a custom replay
/// path without implementing the trait on a named type.
pub struct ClosureReplay<F: Fn(&FailedRow) -> Result<()> + Send + Sync> {
    func: F,
}

impl<F: Fn(&FailedRow) -> Result<()> + Send + Sync> ClosureReplay<F> {
    pub fn new(f: F) -> Self {
        Self { func: f }
    }
}

impl<F: Fn(&FailedRow) -> Result<()> + Send + Sync> ReplayTarget for ClosureReplay<F> {
    fn replay_row(&self, row: &FailedRow) -> Result<()> {
        (self.func)(row)
    }
}

/// Returns a descriptive error when the DLQ cannot accept a row and the
/// caller wants to surface that as a streaming failure.
pub fn dlq_overflow(reason: &str) -> ZyronError {
    ZyronError::StreamingError(format!("DLQ overflow: {reason}"))
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(tag: u64) -> FailedRow {
        make_failed_row(
            "transient",
            format!("err-{tag}"),
            42,
            tag,
            vec![tag as u8; 4],
            1,
        )
    }

    #[test]
    fn write_increments_count_and_persists_locally() {
        let (dlq, sink) = DeadLetterQueue::with_vec_sink("dlq", 4);
        dlq.write(sample(1)).unwrap();
        dlq.write(sample(2)).unwrap();
        assert_eq!(dlq.count(), 2);
        assert_eq!(sink.len(), 2);
    }

    #[test]
    fn write_evicts_oldest_past_cap() {
        let (dlq, _sink) = DeadLetterQueue::with_vec_sink("dlq", 3);
        for i in 0..5 {
            dlq.write(sample(i)).unwrap();
        }
        assert_eq!(dlq.count(), 3);
        let drained = dlq.drain_all();
        let ids: Vec<u64> = drained.iter().map(|r| r.source_commit_version).collect();
        assert_eq!(ids, vec![2, 3, 4]);
    }

    #[test]
    fn evict_oldest_explicit() {
        let (dlq, _sink) = DeadLetterQueue::with_vec_sink("dlq", 10);
        dlq.write(sample(1)).unwrap();
        dlq.write(sample(2)).unwrap();
        dlq.evict_oldest().unwrap();
        assert_eq!(dlq.count(), 1);
        let remaining = dlq.drain_all();
        assert_eq!(remaining[0].source_commit_version, 2);
    }

    #[test]
    fn replay_visits_every_row() {
        let (dlq, _sink) = DeadLetterQueue::with_vec_sink("dlq", 10);
        dlq.write(sample(1)).unwrap();
        dlq.write(sample(2)).unwrap();
        dlq.write(sample(3)).unwrap();
        let observed = Arc::new(Mutex::new(Vec::<u64>::new()));
        let observed_c = Arc::clone(&observed);
        let target = ClosureReplay::new(move |row: &FailedRow| {
            observed_c.lock().push(row.source_commit_version);
            Ok(())
        });
        let n = dlq.replay(&target, 0).unwrap();
        assert_eq!(n, 3);
        assert_eq!(*observed.lock(), vec![1, 2, 3]);
    }

    #[test]
    fn replay_respects_since_cutoff() {
        let (dlq, _sink) = DeadLetterQueue::with_vec_sink("dlq", 10);
        let mut row = sample(1);
        row.received_at = 100;
        dlq.write(row).unwrap();
        let mut row2 = sample(2);
        row2.received_at = 200;
        dlq.write(row2).unwrap();
        let observed = Arc::new(Mutex::new(Vec::<u64>::new()));
        let observed_c = Arc::clone(&observed);
        let target = ClosureReplay::new(move |row: &FailedRow| {
            observed_c.lock().push(row.source_commit_version);
            Ok(())
        });
        let n = dlq.replay(&target, 150).unwrap();
        assert_eq!(n, 1);
        assert_eq!(*observed.lock(), vec![2]);
    }

    #[test]
    fn metrics_track_write_and_eviction() {
        let (dlq, _sink) = DeadLetterQueue::with_vec_sink("dlq", 2);
        dlq.write(sample(1)).unwrap();
        dlq.write(sample(2)).unwrap();
        dlq.write(sample(3)).unwrap();
        let m = dlq.metrics();
        assert_eq!(m.rows_received.load(Ordering::Relaxed), 3);
        assert_eq!(m.rows_evicted.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn failed_row_constructor_fills_timestamps() {
        let r = sample(1);
        assert!(r.received_at > 0);
        assert!(r.failed_at > 0);
        assert_eq!(r.error_class, "transient");
    }
}

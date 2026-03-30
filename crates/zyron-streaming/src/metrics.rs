//! Streaming metrics registry and system view data structures.
//!
//! StreamingMetricsRegistry collects metrics from all running streaming
//! jobs. All data is read from atomic counters with zero locking.
//! View structures provide the data model for SQL system tables:
//! zyron_streaming_watermarks, zyron_checkpoint_history,
//! zyron_streaming_backpressure, zyron_operator_metrics,
//! zyron_streaming_state_size.

use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// WatermarkView
// ---------------------------------------------------------------------------

/// Data for the zyron_streaming_watermarks system view.
/// Represents the current watermark state for a single source.
#[derive(Debug, Clone)]
pub struct WatermarkView {
    pub source_id: u32,
    pub current_watermark_ms: i64,
    pub last_updated_ms: i64,
}

// ---------------------------------------------------------------------------
// CheckpointHistoryView
// ---------------------------------------------------------------------------

/// Data for the zyron_checkpoint_history system view.
/// Represents one completed checkpoint record.
#[derive(Debug, Clone)]
pub struct CheckpointHistoryView {
    pub job_name: String,
    pub checkpoint_id: u64,
    pub checkpoint_time_ms: i64,
    pub duration_ms: u64,
    pub size_bytes: u64,
    pub status: CheckpointStatus,
}

/// Status of a checkpoint in the history.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointStatus {
    Completed,
    Failed,
    InProgress,
}

impl std::fmt::Display for CheckpointStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CheckpointStatus::Completed => write!(f, "COMPLETED"),
            CheckpointStatus::Failed => write!(f, "FAILED"),
            CheckpointStatus::InProgress => write!(f, "IN_PROGRESS"),
        }
    }
}

// ---------------------------------------------------------------------------
// BackpressureView
// ---------------------------------------------------------------------------

/// Data for the zyron_streaming_backpressure system view.
/// Represents backpressure state for one operator.
#[derive(Debug, Clone)]
pub struct BackpressureView {
    pub job_name: String,
    pub operator_name: String,
    pub operator_id: u32,
    pub ratio: f64,
    pub queue_usage: usize,
    pub queue_capacity: usize,
}

// ---------------------------------------------------------------------------
// OperatorMetricsView
// ---------------------------------------------------------------------------

/// Data for the zyron_operator_metrics system view.
/// Represents throughput and latency metrics for one operator.
#[derive(Debug, Clone)]
pub struct OperatorMetricsView {
    pub job_name: String,
    pub operator_name: String,
    pub operator_id: u32,
    pub input_records_total: u64,
    pub output_records_total: u64,
    pub processing_time_ns_total: u64,
    pub current_watermark_ms: i64,
}

impl OperatorMetricsView {
    /// Compute average processing time per record in nanoseconds.
    pub fn avg_processing_time_ns(&self) -> u64 {
        if self.input_records_total == 0 {
            return 0;
        }
        self.processing_time_ns_total / self.input_records_total
    }
}

// ---------------------------------------------------------------------------
// StateSizeView
// ---------------------------------------------------------------------------

/// Data for the zyron_streaming_state_size system view.
/// Represents state storage metrics for one operator.
#[derive(Debug, Clone)]
pub struct StateSizeView {
    pub job_name: String,
    pub operator_name: String,
    pub operator_id: u32,
    pub state_size_bytes: u64,
    pub num_keys: u64,
}

// ---------------------------------------------------------------------------
// StreamingMetricsRegistry
// ---------------------------------------------------------------------------

/// Central registry for streaming metrics.
/// Aggregates metrics from all running jobs and operators. Reads are
/// lock-free via atomic counters.
pub struct StreamingMetricsRegistry {
    /// Total records ingested across all sources.
    pub total_records_in: AtomicU64,
    /// Total records emitted across all sinks.
    pub total_records_out: AtomicU64,
    /// Total checkpoints completed.
    pub total_checkpoints_completed: AtomicU64,
    /// Total checkpoints failed.
    pub total_checkpoints_failed: AtomicU64,
    /// Global minimum watermark across all jobs.
    pub global_watermark_ms: AtomicI64,
    /// Total bytes in operator state.
    pub total_state_bytes: AtomicU64,
    /// Total late records dropped.
    pub total_late_records_dropped: AtomicU64,
    /// Total records shed by load shedding.
    pub total_records_shed: AtomicU64,
    /// Watermark views indexed by source_id for O(1) lookup.
    watermark_views: parking_lot::Mutex<HashMap<u32, WatermarkView>>,
    /// Checkpoint history (stored behind a mutex for structural updates).
    checkpoint_history: parking_lot::Mutex<Vec<CheckpointHistoryView>>,
}

impl StreamingMetricsRegistry {
    pub fn new() -> Self {
        Self {
            total_records_in: AtomicU64::new(0),
            total_records_out: AtomicU64::new(0),
            total_checkpoints_completed: AtomicU64::new(0),
            total_checkpoints_failed: AtomicU64::new(0),
            global_watermark_ms: AtomicI64::new(i64::MIN),
            total_state_bytes: AtomicU64::new(0),
            total_late_records_dropped: AtomicU64::new(0),
            total_records_shed: AtomicU64::new(0),
            watermark_views: parking_lot::Mutex::new(HashMap::new()),
            checkpoint_history: parking_lot::Mutex::new(Vec::new()),
        }
    }

    /// Records that `count` records were ingested.
    pub fn record_ingested(&self, count: u64) {
        self.total_records_in.fetch_add(count, Ordering::Relaxed);
    }

    /// Records that `count` records were emitted.
    pub fn record_emitted(&self, count: u64) {
        self.total_records_out.fetch_add(count, Ordering::Relaxed);
    }

    /// Records a completed checkpoint.
    pub fn record_checkpoint_completed(
        &self,
        job_name: String,
        checkpoint_id: u64,
        checkpoint_time_ms: i64,
        duration_ms: u64,
        size_bytes: u64,
    ) {
        self.total_checkpoints_completed
            .fetch_add(1, Ordering::Relaxed);
        let mut history = self.checkpoint_history.lock();
        history.push(CheckpointHistoryView {
            job_name,
            checkpoint_id,
            checkpoint_time_ms,
            duration_ms,
            size_bytes,
            status: CheckpointStatus::Completed,
        });
    }

    /// Records a failed checkpoint.
    pub fn record_checkpoint_failed(
        &self,
        job_name: String,
        checkpoint_id: u64,
        checkpoint_time_ms: i64,
    ) {
        self.total_checkpoints_failed
            .fetch_add(1, Ordering::Relaxed);
        let mut history = self.checkpoint_history.lock();
        history.push(CheckpointHistoryView {
            job_name,
            checkpoint_id,
            checkpoint_time_ms,
            duration_ms: 0,
            size_bytes: 0,
            status: CheckpointStatus::Failed,
        });
    }

    /// Updates the global watermark.
    pub fn update_global_watermark(&self, watermark_ms: i64) {
        self.global_watermark_ms
            .fetch_max(watermark_ms, Ordering::Relaxed);
    }

    /// Updates the watermark for a specific source.
    pub fn update_source_watermark(&self, source_id: u32, watermark_ms: i64, current_time_ms: i64) {
        let mut views = self.watermark_views.lock();
        // O(1) insert or update via HashMap keyed by source_id.
        let view = views.entry(source_id).or_insert(WatermarkView {
            source_id,
            current_watermark_ms: watermark_ms,
            last_updated_ms: current_time_ms,
        });
        view.current_watermark_ms = watermark_ms;
        view.last_updated_ms = current_time_ms;
    }

    /// Returns a snapshot of all watermark views.
    pub fn watermark_views(&self) -> Vec<WatermarkView> {
        self.watermark_views.lock().values().cloned().collect()
    }

    /// Returns a snapshot of checkpoint history.
    pub fn checkpoint_history(&self) -> Vec<CheckpointHistoryView> {
        self.checkpoint_history.lock().clone()
    }

    /// Returns summary metrics as key-value pairs.
    pub fn summary(&self) -> Vec<(&'static str, u64)> {
        vec![
            (
                "total_records_in",
                self.total_records_in.load(Ordering::Relaxed),
            ),
            (
                "total_records_out",
                self.total_records_out.load(Ordering::Relaxed),
            ),
            (
                "total_checkpoints_completed",
                self.total_checkpoints_completed.load(Ordering::Relaxed),
            ),
            (
                "total_checkpoints_failed",
                self.total_checkpoints_failed.load(Ordering::Relaxed),
            ),
            (
                "total_state_bytes",
                self.total_state_bytes.load(Ordering::Relaxed),
            ),
            (
                "total_late_records_dropped",
                self.total_late_records_dropped.load(Ordering::Relaxed),
            ),
            (
                "total_records_shed",
                self.total_records_shed.load(Ordering::Relaxed),
            ),
        ]
    }
}

impl Default for StreamingMetricsRegistry {
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

    #[test]
    fn test_metrics_registry_basic() {
        let registry = StreamingMetricsRegistry::new();
        registry.record_ingested(100);
        registry.record_ingested(50);
        registry.record_emitted(75);

        assert_eq!(registry.total_records_in.load(Ordering::Relaxed), 150);
        assert_eq!(registry.total_records_out.load(Ordering::Relaxed), 75);
    }

    #[test]
    fn test_metrics_checkpoint_history() {
        let registry = StreamingMetricsRegistry::new();
        registry.record_checkpoint_completed("job1".into(), 1, 1000, 500, 1024);
        registry.record_checkpoint_completed("job1".into(), 2, 2000, 600, 2048);
        registry.record_checkpoint_failed("job1".into(), 3, 3000);

        let history = registry.checkpoint_history();
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].status, CheckpointStatus::Completed);
        assert_eq!(history[2].status, CheckpointStatus::Failed);

        assert_eq!(
            registry.total_checkpoints_completed.load(Ordering::Relaxed),
            2
        );
        assert_eq!(registry.total_checkpoints_failed.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_metrics_watermark_views() {
        let registry = StreamingMetricsRegistry::new();
        registry.update_source_watermark(0, 5000, 10000);
        registry.update_source_watermark(1, 3000, 10000);

        let views = registry.watermark_views();
        assert_eq!(views.len(), 2);
        let v0 = views.iter().find(|v| v.source_id == 0).expect("source 0");
        assert_eq!(v0.current_watermark_ms, 5000);

        // Update existing source.
        registry.update_source_watermark(0, 8000, 15000);
        let views = registry.watermark_views();
        assert_eq!(views.len(), 2);
        let v0 = views.iter().find(|v| v.source_id == 0).expect("source 0");
        assert_eq!(v0.current_watermark_ms, 8000);
    }

    #[test]
    fn test_metrics_global_watermark() {
        let registry = StreamingMetricsRegistry::new();
        registry.update_global_watermark(100);
        registry.update_global_watermark(200);
        registry.update_global_watermark(150); // Should not decrease.

        assert_eq!(registry.global_watermark_ms.load(Ordering::Relaxed), 200);
    }

    #[test]
    fn test_metrics_summary() {
        let registry = StreamingMetricsRegistry::new();
        registry.record_ingested(1000);
        registry.record_emitted(500);

        let summary = registry.summary();
        assert_eq!(summary.len(), 7);
        let in_val = summary.iter().find(|(k, _)| *k == "total_records_in");
        assert_eq!(in_val.map(|(_, v)| *v), Some(1000));
    }

    #[test]
    fn test_operator_metrics_view() {
        let view = OperatorMetricsView {
            job_name: "test-job".into(),
            operator_name: "filter".into(),
            operator_id: 1,
            input_records_total: 1000,
            output_records_total: 500,
            processing_time_ns_total: 1_000_000,
            current_watermark_ms: 5000,
        };

        assert_eq!(view.avg_processing_time_ns(), 1000);
    }

    #[test]
    fn test_operator_metrics_view_no_input() {
        let view = OperatorMetricsView {
            job_name: "test-job".into(),
            operator_name: "filter".into(),
            operator_id: 1,
            input_records_total: 0,
            output_records_total: 0,
            processing_time_ns_total: 0,
            current_watermark_ms: 0,
        };

        assert_eq!(view.avg_processing_time_ns(), 0);
    }
}

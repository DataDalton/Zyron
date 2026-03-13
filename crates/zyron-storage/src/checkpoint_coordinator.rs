//! Checkpoint coordinator for non-blocking checkpoint orchestration.
//!
//! Captures a WAL LSN boundary, directs the background writer to flush all
//! dirty pages below that boundary, logs CheckpointBegin/End to WAL, then
//! cleans up old WAL segments. Writers never block during this process.

use crate::checkpoint::CheckpointTracker;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use zyron_buffer::BackgroundWriter;
use zyron_buffer::BufferPool;
use zyron_common::{Result, ZyronError};
use zyron_wal::{Lsn, WalWriter};

/// Configuration for the checkpoint coordinator and scheduler.
#[derive(Debug, Clone)]
pub struct CheckpointCoordinatorConfig {
    /// Maximum time to wait for background writer to flush all pages (seconds).
    pub checkpoint_timeout_secs: u64,
    /// Seconds between automatic checkpoints (default 300). Time trigger.
    pub checkpoint_interval_secs: u64,
    /// Maximum WAL segments before forcing a checkpoint (default 8). WAL trigger.
    pub max_wal_segments: u32,
}

impl Default for CheckpointCoordinatorConfig {
    fn default() -> Self {
        Self {
            checkpoint_timeout_secs: 60,
            checkpoint_interval_secs: 300,
            max_wal_segments: 8,
        }
    }
}

/// Result of a checkpoint operation.
#[derive(Debug)]
pub struct CheckpointResult {
    /// The LSN boundary for this checkpoint.
    pub checkpoint_lsn: Lsn,
    /// Number of WAL segments deleted.
    pub segments_deleted: usize,
    /// Time spent waiting for the background writer.
    pub wait_duration: Duration,
}

/// Orchestrates non-blocking checkpoints.
///
/// Checkpoint procedure:
/// 1. Capture current WAL LSN as the checkpoint boundary
/// 2. Log CheckpointBegin to WAL
/// 3. Direct background writer to flush all dirty pages below the boundary
/// 4. Wait for background writer to complete (with timeout)
/// 5. Log CheckpointEnd with payload to WAL
/// 6. Advance per-table checkpoint LSNs
/// 7. Clean up old WAL segments
pub struct CheckpointCoordinator {
    pool: Arc<BufferPool>,
    wal: Arc<WalWriter>,
    background_writer: Arc<BackgroundWriter>,
    tracker: Arc<CheckpointTracker>,
    config: CheckpointCoordinatorConfig,
}

impl CheckpointCoordinator {
    /// Creates a new checkpoint coordinator.
    pub fn new(
        pool: Arc<BufferPool>,
        wal: Arc<WalWriter>,
        background_writer: Arc<BackgroundWriter>,
        tracker: Arc<CheckpointTracker>,
        config: CheckpointCoordinatorConfig,
    ) -> Self {
        Self {
            pool,
            wal,
            background_writer,
            tracker,
            config,
        }
    }

    /// Runs a full checkpoint. Writers are never blocked during this process.
    pub fn run_checkpoint(&self) -> Result<CheckpointResult> {
        // 1. Capture checkpoint boundary (current WAL position)
        let checkpoint_lsn = self.wal.next_lsn();

        // 2. Log CheckpointBegin marker
        self.wal.log_checkpoint_begin()?;

        // 3. Direct background writer to prioritize pages below checkpoint_lsn
        self.background_writer.set_target_lsn(checkpoint_lsn.0);

        // 4. Wait for background writer to flush all dirty pages below boundary
        let wait_start = Instant::now();
        self.wait_for_flush_completion(checkpoint_lsn)?;
        let wait_duration = wait_start.elapsed();

        // 5. Serialize and log CheckpointEnd with payload
        let payload = self.serialize_checkpoint_payload(checkpoint_lsn);
        self.wal.log_checkpoint_end(&payload)?;

        // 6. Advance all per-table checkpoint LSNs
        self.tracker.advance_all_tables(checkpoint_lsn.0);

        // 7. Clean up old WAL segments below the global minimum
        let min_lsn = self.tracker.global_min_checkpoint_lsn();
        let segments_deleted = self.wal.cleanup_old_segments(Lsn(min_lsn))?;

        // 8. Return background writer to trickle mode
        self.background_writer.set_target_lsn(0);

        Ok(CheckpointResult {
            checkpoint_lsn,
            segments_deleted,
            wait_duration,
        })
    }

    /// Waits for the background writer to flush all dirty pages below checkpoint_lsn.
    /// Uses exponential backoff from 100us to 10ms with a configurable timeout.
    fn wait_for_flush_completion(&self, checkpoint_lsn: Lsn) -> Result<()> {
        let timeout = Duration::from_secs(self.config.checkpoint_timeout_secs);
        let start = Instant::now();
        let mut backoff_us: u64 = 100;

        loop {
            // Check the pool directly for remaining dirty pages below the boundary.
            // Cannot rely on min_dirty_lsn alone because its initial value (u64::MAX)
            // means "unknown", not "all clean".
            if !self.pool.has_dirty_pages_below(checkpoint_lsn.0) {
                return Ok(());
            }

            if start.elapsed() > timeout {
                return Err(ZyronError::Internal(format!(
                    "Checkpoint timed out after {}s waiting for background writer. Dirty pages remain below LSN {}",
                    self.config.checkpoint_timeout_secs, checkpoint_lsn
                )));
            }

            // Exponential backoff: 100us -> 200us -> 400us -> ... -> 10ms cap
            std::thread::sleep(Duration::from_micros(backoff_us));
            backoff_us = (backoff_us * 2).min(10_000);

            // Wake the background writer in case it's parked
            self.background_writer.wake();
        }
    }

    /// Serializes the checkpoint payload for the CheckpointEnd WAL record.
    ///
    /// Format:
    /// - checkpoint_lsn: u64 (8 bytes)
    /// - num_tables: u32 (4 bytes)
    /// - for each table:
    ///   - table_id: u32 (4 bytes)
    ///   - last_checkpoint_lsn: u64 (8 bytes)
    fn serialize_checkpoint_payload(&self, checkpoint_lsn: Lsn) -> Vec<u8> {
        let table_lsns = self.tracker.snapshot_table_lsns();
        let num_tables = table_lsns.len() as u32;

        // 8 (checkpoint_lsn) + 4 (num_tables) + num_tables * 12
        let mut payload = Vec::with_capacity(12 + (num_tables as usize) * 12);
        payload.extend_from_slice(&checkpoint_lsn.0.to_le_bytes());
        payload.extend_from_slice(&num_tables.to_le_bytes());

        for (table_id, lsn) in &table_lsns {
            payload.extend_from_slice(&table_id.to_le_bytes());
            payload.extend_from_slice(&lsn.to_le_bytes());
        }

        payload
    }

    /// Deserializes a checkpoint payload from a CheckpointEnd WAL record.
    /// Returns the checkpoint LSN.
    pub fn deserialize_checkpoint_payload(payload: &[u8]) -> Option<u64> {
        if payload.len() < 8 {
            return None;
        }
        let checkpoint_lsn = u64::from_le_bytes(payload[..8].try_into().ok()?);
        Some(checkpoint_lsn)
    }
}

/// Cumulative statistics from the checkpoint scheduler.
pub struct CheckpointStats {
    /// Total checkpoints completed since scheduler start.
    pub checkpoints_completed: AtomicU64,
    /// Total WAL segments deleted across all checkpoints.
    pub total_segments_deleted: AtomicU64,
    /// LSN of the most recent completed checkpoint.
    pub last_checkpoint_lsn: AtomicU64,
}

impl CheckpointStats {
    fn new() -> Self {
        Self {
            checkpoints_completed: AtomicU64::new(0),
            total_segments_deleted: AtomicU64::new(0),
            last_checkpoint_lsn: AtomicU64::new(0),
        }
    }
}

/// Automatic checkpoint scheduler with dual triggers (time + WAL segment count).
///
/// Polls every second and fires a checkpoint when either trigger condition is met:
/// 1. Time since last checkpoint exceeds `checkpoint_interval_secs`
/// 2. WAL segments written since last checkpoint exceeds `max_wal_segments`
///
/// Between checkpoints, the background writer continuously trickle-flushes dirty pages,
/// so when a checkpoint fires, most pages are already on disk. This eliminates the I/O
/// spikes that PostgreSQL's checkpointer creates.
pub struct CheckpointScheduler {
    shutdown: Arc<AtomicBool>,
    waker: Arc<OnceLock<thread::Thread>>,
    thread: Option<JoinHandle<()>>,
    stats: Arc<CheckpointStats>,
}

impl CheckpointScheduler {
    /// Starts the scheduler thread. It polls every second and fires checkpoints
    /// when either the time or WAL segment trigger is hit.
    pub fn start(
        coordinator: Arc<CheckpointCoordinator>,
        wal: Arc<WalWriter>,
        config: CheckpointCoordinatorConfig,
    ) -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));
        let waker = Arc::new(OnceLock::new());
        let stats = Arc::new(CheckpointStats::new());

        let thread_shutdown = Arc::clone(&shutdown);
        let thread_waker = Arc::clone(&waker);
        let thread_stats = Arc::clone(&stats);

        let handle = thread::Builder::new()
            .name("zyron-ckpt-sched".into())
            .spawn(move || {
                let _ = thread_waker.set(thread::current());

                Self::scheduler_loop(
                    &coordinator,
                    &wal,
                    &config,
                    &thread_shutdown,
                    &thread_stats,
                );
            })
            .expect("failed to spawn checkpoint scheduler thread");

        Self {
            shutdown,
            waker,
            thread: Some(handle),
            stats,
        }
    }

    /// Main scheduler loop. Checks both trigger conditions each wake cycle.
    fn scheduler_loop(
        coordinator: &CheckpointCoordinator,
        wal: &WalWriter,
        config: &CheckpointCoordinatorConfig,
        shutdown: &AtomicBool,
        stats: &CheckpointStats,
    ) {
        let mut last_checkpoint_time = Instant::now();
        let mut last_checkpoint_segment_id = wal.segment_id();

        loop {
            if shutdown.load(Ordering::Acquire) {
                return;
            }

            let time_elapsed = last_checkpoint_time.elapsed().as_secs();
            let current_segment = wal.segment_id();
            let segments_since = current_segment.saturating_sub(last_checkpoint_segment_id);

            let time_triggered = time_elapsed >= config.checkpoint_interval_secs;
            let wal_triggered = segments_since >= config.max_wal_segments as u32;

            if time_triggered || wal_triggered {
                match coordinator.run_checkpoint() {
                    Ok(result) => {
                        stats.checkpoints_completed.fetch_add(1, Ordering::Relaxed);
                        stats
                            .total_segments_deleted
                            .fetch_add(result.segments_deleted as u64, Ordering::Relaxed);
                        stats
                            .last_checkpoint_lsn
                            .store(result.checkpoint_lsn.0, Ordering::Release);

                        last_checkpoint_time = Instant::now();
                        last_checkpoint_segment_id = wal.segment_id();
                    }
                    Err(_) => {
                        // Checkpoint failed. Continue polling, will retry next cycle.
                    }
                }
            }

            thread::park_timeout(Duration::from_secs(1));
        }
    }

    /// Returns a reference to the cumulative checkpoint stats.
    pub fn stats(&self) -> &Arc<CheckpointStats> {
        &self.stats
    }

    /// Wakes the scheduler thread early (e.g. for testing).
    pub fn wake(&self) {
        if let Some(t) = self.waker.get() {
            t.unpark();
        }
    }

    /// Gracefully shuts down the scheduler thread.
    pub fn shutdown(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        if let Some(t) = self.waker.get() {
            t.unpark();
        }
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for CheckpointScheduler {
    fn drop(&mut self) {
        if self.thread.is_some() {
            self.shutdown();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_payload_round_trip() {
        let lsn = Lsn(123456789);
        let coordinator_config = CheckpointCoordinatorConfig::default();

        // Manually create the payload like serialize_checkpoint_payload does
        let mut payload = Vec::new();
        payload.extend_from_slice(&lsn.0.to_le_bytes());
        payload.extend_from_slice(&0u32.to_le_bytes()); // 0 tables

        let decoded = CheckpointCoordinator::deserialize_checkpoint_payload(&payload);
        assert_eq!(decoded, Some(123456789));

        // Confirm config defaults
        assert_eq!(coordinator_config.checkpoint_timeout_secs, 60);
        assert_eq!(coordinator_config.checkpoint_interval_secs, 300);
        assert_eq!(coordinator_config.max_wal_segments, 8);
    }

    #[test]
    fn test_deserialize_empty_payload() {
        assert_eq!(
            CheckpointCoordinator::deserialize_checkpoint_payload(&[]),
            None
        );
        assert_eq!(
            CheckpointCoordinator::deserialize_checkpoint_payload(&[1, 2, 3]),
            None
        );
    }

    #[test]
    fn test_checkpoint_stats_initial() {
        let stats = CheckpointStats::new();
        assert_eq!(stats.checkpoints_completed.load(Ordering::Relaxed), 0);
        assert_eq!(stats.total_segments_deleted.load(Ordering::Relaxed), 0);
        assert_eq!(stats.last_checkpoint_lsn.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_scheduler_shutdown() {
        // Verify the scheduler can start and shut down cleanly without panicking.
        // Uses a mock coordinator setup with real WAL + buffer pool.
        let tmp_dir = tempfile::tempdir().unwrap();

        let wal = Arc::new(
            WalWriter::new(zyron_wal::WalWriterConfig {
                wal_dir: tmp_dir.path().to_path_buf(),
                segment_size: 1024 * 1024,
                fsync_enabled: false,
                ring_buffer_capacity: 256 * 1024,
            })
            .unwrap(),
        );

        let pool = Arc::new(zyron_buffer::BufferPool::new(
            zyron_buffer::BufferPoolConfig { num_frames: 100 },
        ));

        let write_fn: zyron_buffer::WriteFn = Arc::new(|_pid, _data| Ok(()));
        let bg_writer = Arc::new(zyron_buffer::BackgroundWriter::new(
            Arc::clone(&pool),
            write_fn,
            zyron_buffer::BackgroundWriterConfig::default(),
        ));

        let tracker = Arc::new(CheckpointTracker::new());

        let coordinator = Arc::new(CheckpointCoordinator::new(
            Arc::clone(&pool),
            Arc::clone(&wal),
            bg_writer,
            tracker,
            CheckpointCoordinatorConfig::default(),
        ));

        let mut scheduler = CheckpointScheduler::start(
            coordinator,
            Arc::clone(&wal),
            CheckpointCoordinatorConfig {
                checkpoint_interval_secs: 3600, // long interval, won't fire
                max_wal_segments: 1000,
                ..Default::default()
            },
        );

        // Should start and shut down without panic
        scheduler.shutdown();
        assert_eq!(
            scheduler.stats().checkpoints_completed.load(Ordering::Relaxed),
            0
        );
    }

    #[test]
    fn test_scheduler_time_trigger() {
        // Set a 1-second interval and verify checkpoint fires within 3 seconds.
        let tmp_dir = tempfile::tempdir().unwrap();

        let wal = Arc::new(
            WalWriter::new(zyron_wal::WalWriterConfig {
                wal_dir: tmp_dir.path().to_path_buf(),
                segment_size: 1024 * 1024,
                fsync_enabled: false,
                ring_buffer_capacity: 256 * 1024,
            })
            .unwrap(),
        );

        let pool = Arc::new(zyron_buffer::BufferPool::new(
            zyron_buffer::BufferPoolConfig { num_frames: 100 },
        ));

        let write_fn: zyron_buffer::WriteFn = Arc::new(|_pid, _data| Ok(()));
        let bg_writer = Arc::new(zyron_buffer::BackgroundWriter::new(
            Arc::clone(&pool),
            write_fn,
            zyron_buffer::BackgroundWriterConfig::default(),
        ));

        let tracker = Arc::new(CheckpointTracker::new());
        tracker.register_table(1, &[200, 201]);

        let coordinator = Arc::new(CheckpointCoordinator::new(
            Arc::clone(&pool),
            Arc::clone(&wal),
            bg_writer,
            tracker,
            CheckpointCoordinatorConfig::default(),
        ));

        let mut scheduler = CheckpointScheduler::start(
            coordinator,
            Arc::clone(&wal),
            CheckpointCoordinatorConfig {
                checkpoint_interval_secs: 1,
                max_wal_segments: 1000, // won't fire from WAL trigger
                ..Default::default()
            },
        );

        // Wait for the time trigger to fire
        std::thread::sleep(Duration::from_secs(3));

        let completed = scheduler.stats().checkpoints_completed.load(Ordering::Relaxed);
        assert!(
            completed >= 1,
            "expected at least 1 checkpoint from time trigger, got {}",
            completed
        );

        scheduler.shutdown();
    }
}

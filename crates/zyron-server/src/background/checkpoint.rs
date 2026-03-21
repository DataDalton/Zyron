//! Checkpoint worker with WAL-bytes-based adaptive triggering.
//!
//! Wraps the existing CheckpointCoordinator and adds WAL byte threshold
//! as the primary trigger. Fires a checkpoint when accumulated WAL bytes
//! since the last checkpoint exceed the threshold, with a fallback timer
//! for idle systems and a minimum gap to prevent thrashing.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use tracing::{error, info};

use zyron_storage::CheckpointCoordinator;
use zyron_wal::WalWriter;

/// Configuration for the checkpoint worker.
#[derive(Debug, Clone)]
pub struct CheckpointWorkerConfig {
    /// WAL bytes threshold before triggering a checkpoint (default 64 MB).
    pub wal_bytes_threshold: u64,
    /// Maximum seconds between checkpoints for idle systems (default 600).
    pub max_interval_secs: u64,
    /// Minimum seconds between checkpoints to prevent thrashing (default 5).
    pub min_interval_secs: u64,
}

impl Default for CheckpointWorkerConfig {
    fn default() -> Self {
        Self {
            wal_bytes_threshold: 64 * 1024 * 1024,
            max_interval_secs: 600,
            min_interval_secs: 5,
        }
    }
}

/// Cumulative checkpoint statistics.
pub struct CheckpointWorkerStats {
    pub checkpoints_completed: AtomicU64,
    pub total_segments_deleted: AtomicU64,
    pub last_checkpoint_lsn: AtomicU64,
}

impl CheckpointWorkerStats {
    fn new() -> Self {
        Self {
            checkpoints_completed: AtomicU64::new(0),
            total_segments_deleted: AtomicU64::new(0),
            last_checkpoint_lsn: AtomicU64::new(0),
        }
    }
}

/// Background checkpoint worker that monitors WAL accumulation and fires
/// checkpoints based on byte threshold, time, or both.
pub struct CheckpointWorker {
    shutdown: Arc<AtomicBool>,
    waker: Arc<OnceLock<thread::Thread>>,
    thread: Option<JoinHandle<()>>,
    stats: Arc<CheckpointWorkerStats>,
    coordinator: Arc<CheckpointCoordinator>,
}

impl CheckpointWorker {
    /// Starts the checkpoint worker thread.
    pub fn start(
        coordinator: Arc<CheckpointCoordinator>,
        wal: Arc<WalWriter>,
        config: CheckpointWorkerConfig,
    ) -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));
        let waker = Arc::new(OnceLock::new());
        let stats = Arc::new(CheckpointWorkerStats::new());

        let thread_shutdown = Arc::clone(&shutdown);
        let thread_waker = Arc::clone(&waker);
        let thread_stats = Arc::clone(&stats);
        let thread_coordinator = Arc::clone(&coordinator);

        let handle = thread::Builder::new()
            .name("zyron-ckpt-worker".into())
            .spawn(move || {
                let _ = thread_waker.set(thread::current());
                Self::worker_loop(
                    &thread_coordinator,
                    &wal,
                    &config,
                    &thread_shutdown,
                    &thread_stats,
                );
            })
            .expect("failed to spawn checkpoint worker thread");

        Self {
            shutdown,
            waker,
            thread: Some(handle),
            stats,
            coordinator,
        }
    }

    /// Main worker loop. Checks WAL bytes, time elapsed, and min gap each cycle.
    fn worker_loop(
        coordinator: &CheckpointCoordinator,
        wal: &WalWriter,
        config: &CheckpointWorkerConfig,
        shutdown: &AtomicBool,
        stats: &CheckpointWorkerStats,
    ) {
        let mut last_checkpoint_time = Instant::now();
        let mut last_checkpoint_lsn = wal.next_lsn().0;

        loop {
            if shutdown.load(Ordering::Acquire) {
                return;
            }

            let elapsed = last_checkpoint_time.elapsed();

            // Enforce minimum gap between checkpoints
            if elapsed < Duration::from_secs(config.min_interval_secs) {
                thread::park_timeout(Duration::from_secs(1));
                continue;
            }

            let current_lsn = wal.next_lsn().0;
            let wal_bytes_since = current_lsn.saturating_sub(last_checkpoint_lsn);
            let time_triggered = elapsed.as_secs() >= config.max_interval_secs;
            let bytes_triggered = wal_bytes_since >= config.wal_bytes_threshold;

            if time_triggered || bytes_triggered {
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
                        last_checkpoint_lsn = result.checkpoint_lsn.0;

                        info!(
                            "Checkpoint complete: lsn={}, segments_deleted={}, wait={}ms",
                            result.checkpoint_lsn,
                            result.segments_deleted,
                            result.wait_duration.as_millis()
                        );
                    }
                    Err(e) => {
                        error!("Checkpoint failed: {}. Will retry next cycle.", e);
                    }
                }
            }

            thread::park_timeout(Duration::from_secs(1));
        }
    }

    /// Returns a reference to checkpoint statistics.
    pub fn stats(&self) -> &Arc<CheckpointWorkerStats> {
        &self.stats
    }

    /// Runs a final checkpoint for clean shutdown (zero-replay restart).
    /// Returns Ok on success, Err if the checkpoint failed.
    pub fn final_checkpoint(&self) -> zyron_common::Result<()> {
        info!("Running final checkpoint for clean shutdown");
        let result = self.coordinator.run_checkpoint()?;
        info!(
            "Final checkpoint complete: lsn={}, segments_deleted={}",
            result.checkpoint_lsn, result.segments_deleted
        );
        Ok(())
    }

    /// Wakes the worker thread early.
    pub fn wake(&self) {
        if let Some(t) = self.waker.get() {
            t.unpark();
        }
    }

    /// Gracefully shuts down the worker thread.
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

impl Drop for CheckpointWorker {
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
    fn test_config_defaults() {
        let config = CheckpointWorkerConfig::default();
        assert_eq!(config.wal_bytes_threshold, 64 * 1024 * 1024);
        assert_eq!(config.max_interval_secs, 600);
        assert_eq!(config.min_interval_secs, 5);
    }

    #[test]
    fn test_stats_initial() {
        let stats = CheckpointWorkerStats::new();
        assert_eq!(stats.checkpoints_completed.load(Ordering::Relaxed), 0);
        assert_eq!(stats.total_segments_deleted.load(Ordering::Relaxed), 0);
        assert_eq!(stats.last_checkpoint_lsn.load(Ordering::Relaxed), 0);
    }
}

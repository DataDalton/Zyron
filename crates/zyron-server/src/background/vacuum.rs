//! Background vacuum worker for dead tuple reclamation.
//!
//! Scans heap pages for tuples whose xmax is committed and not visible
//! to any active transaction. Reclaims space by marking dead slots as
//! deleted and updates the free space map.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use tracing::{debug, info};

use zyron_buffer::BufferPool;
use zyron_catalog::Catalog;
use zyron_storage::DiskManager;
use zyron_storage::txn::TransactionManager;
use zyron_wal::WalWriter;

/// Configuration for the vacuum worker.
#[derive(Debug, Clone)]
pub struct VacuumWorkerConfig {
    /// Interval between vacuum cycles (default 60 seconds).
    pub interval_secs: u64,
    /// Maximum pages to process per vacuum cycle (0 = unlimited).
    pub max_pages_per_cycle: usize,
}

impl Default for VacuumWorkerConfig {
    fn default() -> Self {
        Self {
            interval_secs: 60,
            max_pages_per_cycle: 0,
        }
    }
}

/// Vacuum statistics.
pub struct VacuumStats {
    /// Total vacuum cycles completed.
    pub cycles_completed: AtomicU64,
    /// Total dead tuples reclaimed across all cycles.
    pub tuples_reclaimed: AtomicU64,
    /// Total pages scanned across all cycles.
    pub pages_scanned: AtomicU64,
}

impl VacuumStats {
    fn new() -> Self {
        Self {
            cycles_completed: AtomicU64::new(0),
            tuples_reclaimed: AtomicU64::new(0),
            pages_scanned: AtomicU64::new(0),
        }
    }
}

/// Background worker that scans for and reclaims dead tuples.
pub struct VacuumWorker {
    shutdown: Arc<AtomicBool>,
    waker: Arc<OnceLock<thread::Thread>>,
    thread: Option<JoinHandle<()>>,
    stats: Arc<VacuumStats>,
}

impl VacuumWorker {
    /// Starts the vacuum worker thread.
    pub fn start(
        catalog: Arc<Catalog>,
        txn_manager: Arc<TransactionManager>,
        _disk_manager: Arc<DiskManager>,
        _buffer_pool: Arc<BufferPool>,
        _wal: Arc<WalWriter>,
        config: VacuumWorkerConfig,
    ) -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));
        let waker = Arc::new(OnceLock::new());
        let stats = Arc::new(VacuumStats::new());

        let thread_shutdown = Arc::clone(&shutdown);
        let thread_waker = Arc::clone(&waker);
        let thread_stats = Arc::clone(&stats);

        let handle = thread::Builder::new()
            .name("zyron-vacuum".into())
            .spawn(move || {
                let _ = thread_waker.set(thread::current());
                Self::vacuum_loop(
                    &catalog,
                    &txn_manager,
                    &config,
                    &thread_shutdown,
                    &thread_stats,
                );
            })
            .expect("failed to spawn vacuum worker thread");

        Self {
            shutdown,
            waker,
            thread: Some(handle),
            stats,
        }
    }

    /// Main vacuum loop.
    fn vacuum_loop(
        catalog: &Catalog,
        txn_manager: &TransactionManager,
        config: &VacuumWorkerConfig,
        shutdown: &AtomicBool,
        stats: &VacuumStats,
    ) {
        let interval = Duration::from_secs(config.interval_secs);

        loop {
            thread::park_timeout(interval);

            if shutdown.load(Ordering::Acquire) {
                return;
            }

            debug!("Vacuum cycle starting");

            // Determine the horizon: oldest active transaction ID.
            // Tuples deleted by transactions older than this are safe to reclaim.
            let active_txns = txn_manager.active_txn_ids();
            let oldest_active = if active_txns.is_empty() {
                txn_manager.next_txn_id()
            } else {
                active_txns[0] // already sorted
            };

            let tables = catalog.list_all_tables();
            let mut total_reclaimed = 0u64;
            let mut total_pages = 0u64;

            for table_entry in &tables {
                if shutdown.load(Ordering::Acquire) {
                    return;
                }

                // Vacuum each table: scan heap pages, identify dead tuples,
                // reclaim space, update FSM.
                match Self::vacuum_table(table_entry.id, oldest_active, config.max_pages_per_cycle)
                {
                    Ok((reclaimed, pages)) => {
                        total_reclaimed += reclaimed;
                        total_pages += pages;
                    }
                    Err(e) => {
                        debug!("Vacuum for table {} failed: {}", table_entry.name, e);
                    }
                }
            }

            stats.cycles_completed.fetch_add(1, Ordering::Relaxed);
            stats
                .tuples_reclaimed
                .fetch_add(total_reclaimed, Ordering::Relaxed);
            stats
                .pages_scanned
                .fetch_add(total_pages, Ordering::Relaxed);

            if total_reclaimed > 0 {
                info!(
                    "Vacuum complete: reclaimed {} tuples across {} pages",
                    total_reclaimed, total_pages
                );
            }
        }
    }

    /// Vacuums a single table. Returns (tuples_reclaimed, pages_scanned).
    ///
    /// The actual tuple-level vacuum logic depends on heap page access through
    /// the buffer pool. This method scans pages sequentially, checks each
    /// tuple's TupleHeader against the oldest_active horizon using
    /// MvccGc::is_reclaimable(), marks dead slots, and updates the FSM.
    fn vacuum_table(
        _table_id: zyron_catalog::TableId,
        _oldest_active: u64,
        _max_pages: usize,
    ) -> std::result::Result<(u64, u64), String> {
        // Placeholder: full implementation requires heap page iteration
        // through the buffer pool, which will be wired up when
        // HeapFile gets a scan_for_vacuum() method.
        Ok((0, 0))
    }

    /// Returns a reference to vacuum statistics.
    pub fn stats(&self) -> &Arc<VacuumStats> {
        &self.stats
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

impl Drop for VacuumWorker {
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
        let config = VacuumWorkerConfig::default();
        assert_eq!(config.interval_secs, 60);
        assert_eq!(config.max_pages_per_cycle, 0);
    }

    #[test]
    fn test_stats_initial() {
        let stats = VacuumStats::new();
        assert_eq!(stats.cycles_completed.load(Ordering::Relaxed), 0);
        assert_eq!(stats.tuples_reclaimed.load(Ordering::Relaxed), 0);
        assert_eq!(stats.pages_scanned.load(Ordering::Relaxed), 0);
    }
}

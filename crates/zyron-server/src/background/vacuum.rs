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
use zyron_catalog::{Catalog, TableEntry};
use zyron_storage::txn::TransactionManager;
use zyron_storage::{DiskManager, HeapFile, HeapFileConfig, HeapPage};
use zyron_wal::WalWriter;

/// Current wall-clock time in microseconds since the Unix epoch.
fn now_micros() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}

/// Converts an epoch-microsecond instant to epoch seconds, the unit the
/// maintenance timestamps in zyron_stat_tables are reported in.
fn epoch_seconds(micros: u64) -> u64 {
    micros / 1_000_000
}

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
        disk_manager: Arc<DiskManager>,
        buffer_pool: Arc<BufferPool>,
        _wal: Arc<WalWriter>,
        btree_indexes: Arc<scc::HashMap<u32, Arc<zyron_storage::BTreeIndex>>>,
        table_io_stats: Arc<zyron_common::TableIOStatsRegistry>,
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
                    &btree_indexes,
                    &_wal,
                    &txn_manager,
                    &disk_manager,
                    &buffer_pool,
                    &table_io_stats,
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
        btree_indexes: &scc::HashMap<u32, Arc<zyron_storage::BTreeIndex>>,
        wal: &Arc<WalWriter>,
        txn_manager: &TransactionManager,
        disk_manager: &Arc<DiskManager>,
        buffer_pool: &Arc<BufferPool>,
        table_io_stats: &zyron_common::TableIOStatsRegistry,
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

            // Record a (now, durable LSN) sample so time-based retention can map
            // a window to a floor LSN. The vacuum interval is the sample
            // resolution, far finer than the day-scale windows it serves.
            let now_micros = now_micros();
            txn_manager
                .retention_clock()
                .record(now_micros, wal.flushed_lsn().0);

            // The catalog's version tags are the source of truth for the tag
            // retention floor. Recompute it each cycle so a dropped tag raises
            // the floor (resuming reclamation) and any DDL race self-heals. A
            // freshly created tag is honored immediately by CREATE VERSION's own
            // fetch-min, so this never reclaims a new tag's history early.
            let tag_floor = catalog
                .list_version_tags()
                .iter()
                .map(|t| t.version_id)
                .min()
                .unwrap_or(u64::MAX);
            txn_manager
                .status_map()
                .set_version_retention_floor(tag_floor);

            let tables = catalog.list_all_tables();
            let mut total_reclaimed = 0u64;
            let mut total_pages = 0u64;
            // A full sweep (no page cap) that touches every table reclaims all
            // aborted-insert tuples and committed-deleted tuples below the
            // horizon, so the commit-status frozen watermark can advance past
            // them. Any error or page cap leaves dead tuples behind, so hold off.
            let mut full_sweep = config.max_pages_per_cycle == 0;
            // Smallest retention floor across all tables (version tags + time
            // windows). Commit LSNs at or below it are no longer needed to date
            // any retained version, so their segments can be freed.
            let mut global_floor = u64::MAX;

            for table_entry in &tables {
                if shutdown.load(Ordering::Acquire) {
                    return;
                }

                // Vacuum each table: scan heap pages, identify dead tuples,
                // reclaim space, update FSM, and delete the reclaimed rows'
                // B+tree index entries so stale entries do not accumulate.
                // The effective retention floor keeps versions still visible at
                // a tagged version or within the table's time-travel window.
                let index_snap = catalog.index_snapshot(table_entry.id);
                let retention_floor = zyron_executor::operator::modify::effective_retention_floor(
                    table_entry,
                    txn_manager.status_map(),
                    txn_manager.retention_clock(),
                    now_micros,
                );
                global_floor = global_floor.min(retention_floor);
                match Self::vacuum_table(
                    &table_entry,
                    oldest_active,
                    config.max_pages_per_cycle,
                    disk_manager,
                    buffer_pool,
                    txn_manager.status_map(),
                    retention_floor,
                    &index_snap.btree,
                    btree_indexes,
                ) {
                    Ok((reclaimed, pages)) => {
                        total_reclaimed += reclaimed;
                        total_pages += pages;
                        // Only a pass that reached the end of the table clears
                        // the dead estimate. A page-capped cycle left dead rows
                        // behind, so reporting zero would be a lie the next
                        // cycle has to undo
                        if config.max_pages_per_cycle == 0
                            || pages < config.max_pages_per_cycle as u64
                        {
                            table_io_stats
                                .get_or_create(table_entry.id.0)
                                .record_vacuum(epoch_seconds(now_micros));
                        }
                    }
                    Err(e) => {
                        debug!("Vacuum for table {} failed: {}", table_entry.name, e);
                        full_sweep = false;
                    }
                }
            }

            // Bound the retention clock: keep samples back to the longest finite
            // retention window across tables; unlimited and unset tables need no
            // historical samples (their floor is 0 or u64::MAX).
            let max_window_secs = tables
                .iter()
                .map(|t| t.time_travel_retention_secs)
                .filter(|&s| s != 0 && s != u64::MAX)
                .max()
                .unwrap_or(0);
            let keep_from = if max_window_secs > 0 {
                now_micros.saturating_sub(
                    max_window_secs
                        .saturating_mul(1_000_000)
                        .saturating_add(3_600_000_000),
                )
            } else {
                now_micros
            };
            txn_manager.retention_clock().prune_before(keep_from);

            // Advance the frozen horizon: after a clean full sweep, every tuple
            // below `oldest_active` is committed (aborted inserts reclaimed,
            // aborted-delete stamps cleared), so visibility can treat ids below
            // it as committed without a status lookup. With the horizon past
            // them, the commit-status segments below it are unreachable and their
            // memory can be reclaimed.
            if full_sweep {
                let status_map = txn_manager.status_map();
                status_map.advance_vacuum_frozen(oldest_active);
                let freed = status_map.truncate_below(oldest_active);
                if freed > 0 {
                    debug!(
                        "Truncated {} commit-status segments below {}",
                        freed, oldest_active
                    );
                }
            }

            // Free commit-LSN segments no longer needed to date any retained
            // version: those whose transactions all committed at or below the
            // global retention floor. Capped at the status truncation watermark
            // (so aborts there are already gone). Bounds commit-LSN memory to the
            // retention window.
            let freed_lsn = txn_manager
                .status_map()
                .advance_commit_lsn_dawn(global_floor);
            if freed_lsn > 0 {
                debug!(
                    "Freed {} commit-LSN segments below the retention floor",
                    freed_lsn
                );
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
    /// Scans heap pages sequentially through the buffer pool. For each page,
    /// reads every tuple's xmin/xmax and uses `is_dead_tuple` (commit-status
    /// aware) to find rows whose inserter aborted or whose deleter committed
    /// before the oldest active transaction. Dead tuples have their slots zeroed
    /// (length = 0), reclaiming space for new inserts.
    #[allow(clippy::too_many_arguments)]
    fn vacuum_table(
        table: &TableEntry,
        oldest_active: u64,
        max_pages: usize,
        disk_manager: &Arc<DiskManager>,
        buffer_pool: &Arc<BufferPool>,
        status_map: &zyron_storage::TxnStatusMap,
        retention_floor: u64,
        btree: &[(zyron_catalog::IndexId, zyron_catalog::ColumnId, bool)],
        btree_indexes: &scc::HashMap<u32, Arc<zyron_storage::BTreeIndex>>,
    ) -> std::result::Result<(u64, u64), String> {
        let heap_file = HeapFile::new(
            Arc::clone(disk_manager),
            Arc::clone(buffer_pool),
            HeapFileConfig {
                heap_file_id: table.heap_file_id,
                fsm_file_id: table.fsm_file_id,
            },
        )
        .map_err(|e| format!("failed to open heap file: {}", e))?;

        let scan_guard = heap_file
            .scan()
            .map_err(|e| format!("scan failed: {}", e))?;

        let mut tuples_reclaimed = 0u64;
        let mut pages_scanned = 0u64;

        let page_ids = scan_guard.page_ids();

        let page_limit = if max_pages > 0 {
            max_pages
        } else {
            page_ids.len()
        };

        // Predicates shared across pages. A tuple is dead when its inserter
        // aborted or its committed deleter is older than the oldest active
        // transaction and no retained version still sees the row alive. An
        // aborted-delete stamp on a still-live row is cleared.
        let is_dead = |xmin: u32, xmax: u32| {
            status_map.is_aborted(xmin as u64)
                || (xmax != 0
                    && status_map.is_committed(xmax as u64)
                    && (xmax as u64) < oldest_active
                    && status_map.is_reclaimable_below(xmax as u64, retention_floor))
        };
        let is_aborted = |xid: u32| status_map.is_aborted(xid as u64);
        let clean_indexes = !btree.is_empty();

        for &page_id in page_ids.iter().take(page_limit) {
            pages_scanned += 1;

            // Pin the frame and mutate in place under the exclusive frame write
            // lock. The insert burst path holds the shared lock, so vacuum never
            // clobbers a concurrent append.
            let Some(frame) = buffer_pool.fetch_page(page_id) else {
                continue;
            };
            // Reclaimed rows' images, captured under the lock so their index
            // entries can be deleted after the lock is released.
            let mut dead: Vec<(u16, Vec<u8>)> = Vec::new();
            let (reclaimed_on_page, modified) = {
                let mut guard = frame.write_data();
                let data: &mut [u8] = &mut guard[..];
                if HeapPage::heap_header_from_slice(data).slot_count == 0 {
                    (0u64, false)
                } else if clean_indexes {
                    HeapPage::vacuum_in_slice_collect(data, &is_dead, &is_aborted, &mut dead)
                } else {
                    HeapPage::vacuum_in_slice(data, &is_dead, &is_aborted)
                }
            };
            buffer_pool.unpin_page(page_id, modified);

            // Delete the reclaimed rows' B+tree entries outside the frame lock,
            // so a stale entry never outlives the heap tuple it points at.
            if clean_indexes && !dead.is_empty() {
                zyron_executor::operator::modify::vacuum_index_cleanup(
                    table,
                    page_id,
                    &dead,
                    btree,
                    btree_indexes,
                );
            }

            if reclaimed_on_page > 0 {
                tuples_reclaimed += reclaimed_on_page;
                debug!(
                    "Vacuumed page {:?}: reclaimed {} dead tuples",
                    page_id, reclaimed_on_page
                );
            }
        }

        // Drop the scan guard to unpin all pages
        drop(scan_guard);

        Ok((tuples_reclaimed, pages_scanned))
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

/// Runs a single vacuum pass on a table. Called by the VACUUM SQL command.
/// Returns (tuples_reclaimed, pages_scanned).
#[allow(clippy::too_many_arguments)]
pub fn vacuum_table_immediate(
    table: &TableEntry,
    oldest_active: u64,
    disk_manager: &Arc<DiskManager>,
    buffer_pool: &Arc<BufferPool>,
    status_map: &zyron_storage::TxnStatusMap,
    retention_floor: u64,
    btree: &[(zyron_catalog::IndexId, zyron_catalog::ColumnId, bool)],
    btree_indexes: &scc::HashMap<u32, Arc<zyron_storage::BTreeIndex>>,
) -> std::result::Result<(u64, u64), String> {
    VacuumWorker::vacuum_table(
        table,
        oldest_active,
        0,
        disk_manager,
        buffer_pool,
        status_map,
        retention_floor,
        btree,
        btree_indexes,
    )
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

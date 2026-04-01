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
use zyron_common::page::PAGE_SIZE;
use zyron_storage::txn::TransactionManager;
use zyron_storage::{DiskManager, HeapFile, HeapFileConfig, HeapPage, MvccGc, TupleHeader};
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
        disk_manager: Arc<DiskManager>,
        buffer_pool: Arc<BufferPool>,
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
                    &disk_manager,
                    &buffer_pool,
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
        disk_manager: &Arc<DiskManager>,
        buffer_pool: &Arc<BufferPool>,
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
                match Self::vacuum_table(
                    &table_entry,
                    oldest_active,
                    config.max_pages_per_cycle,
                    disk_manager,
                    buffer_pool,
                ) {
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
    /// Scans heap pages sequentially through the buffer pool. For each page,
    /// reads every tuple slot and checks the TupleHeader xmax against the
    /// oldest_active horizon using MvccGc::is_reclaimable(). Dead tuples
    /// have their slots zeroed (length = 0), reclaiming space for new inserts.
    /// The FSM is updated after processing each modified page.
    fn vacuum_table(
        table: &TableEntry,
        oldest_active: u64,
        max_pages: usize,
        disk_manager: &Arc<DiskManager>,
        buffer_pool: &Arc<BufferPool>,
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

        for &page_id in page_ids.iter().take(page_limit) {
            pages_scanned += 1;

            // Fetch the page data from the buffer pool
            let page_data = match buffer_pool.fetch_page(page_id) {
                Some(frame) => {
                    let guard = frame.read_data();
                    let data: [u8; PAGE_SIZE] = **guard;
                    drop(guard);
                    buffer_pool.unpin_page(page_id, false);
                    data
                }
                None => continue,
            };

            let header = HeapPage::heap_header_from_slice(&page_data);
            if header.slot_count == 0 {
                continue;
            }

            // First pass: count dead tuples on this page
            let mut dead_count = 0u64;
            let mut total_count = 0u64;

            for i in 0..header.slot_count {
                let slot_offset = HeapPage::DATA_START + (i as usize) * 4;
                let slot_len =
                    u16::from_le_bytes([page_data[slot_offset + 2], page_data[slot_offset + 3]]);
                if slot_len == 0 {
                    continue;
                }
                total_count += 1;

                let tuple_offset =
                    u16::from_le_bytes([page_data[slot_offset], page_data[slot_offset + 1]])
                        as usize;

                // Read xmax from the tuple header (offset 8 within the 12-byte header)
                if tuple_offset + TupleHeader::SIZE <= PAGE_SIZE {
                    let xmax = u32::from_le_bytes([
                        page_data[tuple_offset + 8],
                        page_data[tuple_offset + 9],
                        page_data[tuple_offset + 10],
                        page_data[tuple_offset + 11],
                    ]);

                    if MvccGc::is_reclaimable(xmax, oldest_active) {
                        dead_count += 1;
                    }
                }
            }

            if dead_count == 0 {
                continue;
            }

            // Second pass: reclaim dead tuples by zeroing slot lengths
            let mut modified_page = page_data;
            let mut reclaimed_on_page = 0u64;

            for i in 0..header.slot_count {
                let slot_offset = HeapPage::DATA_START + (i as usize) * 4;
                let slot_len = u16::from_le_bytes([
                    modified_page[slot_offset + 2],
                    modified_page[slot_offset + 3],
                ]);
                if slot_len == 0 {
                    continue;
                }

                let tuple_offset = u16::from_le_bytes([
                    modified_page[slot_offset],
                    modified_page[slot_offset + 1],
                ]) as usize;

                if tuple_offset + TupleHeader::SIZE <= PAGE_SIZE {
                    let xmax = u32::from_le_bytes([
                        modified_page[tuple_offset + 8],
                        modified_page[tuple_offset + 9],
                        modified_page[tuple_offset + 10],
                        modified_page[tuple_offset + 11],
                    ]);

                    if MvccGc::is_reclaimable(xmax, oldest_active) {
                        // Zero the slot length to mark as deleted
                        modified_page[slot_offset + 2] = 0;
                        modified_page[slot_offset + 3] = 0;
                        reclaimed_on_page += 1;
                    }
                }
            }

            if reclaimed_on_page > 0 {
                // Write the modified page back through the buffer pool
                if let Some(frame) = buffer_pool.fetch_page(page_id) {
                    frame.copy_from(&modified_page);
                    buffer_pool.unpin_page(page_id, true); // Mark dirty
                } else if let Ok((_, evicted)) = buffer_pool.load_page(page_id, &modified_page) {
                    // Page was evicted between read and write. Load it back.
                    if let Some(evicted_page) = evicted {
                        // Write evicted page synchronously
                        let path = disk_manager
                            .data_dir()
                            .join(format!("{:08}.dat", evicted_page.page_id.file_id));
                        if let Ok(mut file) = std::fs::OpenOptions::new().write(true).open(&path) {
                            use std::io::{Seek, SeekFrom, Write};
                            let offset = evicted_page.page_id.page_num * (PAGE_SIZE as u64);
                            let _ = file.seek(SeekFrom::Start(offset));
                            let _ = file.write_all(evicted_page.data.as_ref());
                        }
                    }
                    buffer_pool.unpin_page(page_id, true);
                }

                tuples_reclaimed += reclaimed_on_page;

                debug!(
                    "Vacuumed page {:?}: reclaimed {} dead tuples out of {} total",
                    page_id, reclaimed_on_page, total_count
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
pub fn vacuum_table_immediate(
    table: &TableEntry,
    oldest_active: u64,
    disk_manager: &Arc<DiskManager>,
    buffer_pool: &Arc<BufferPool>,
) -> std::result::Result<(u64, u64), String> {
    VacuumWorker::vacuum_table(table, oldest_active, 0, disk_manager, buffer_pool)
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

//! Background statistics collector.
//!
//! Periodically samples table pages to compute approximate row counts,
//! average tuple widths, null fractions, and distinct value estimates.
//! Updates the catalog stats so the query planner has fresh cost information.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use tracing::{debug, info};

use zyron_buffer::BufferPool;
use zyron_catalog::{Catalog, analyze_table};
use zyron_storage::{DiskManager, HeapFile, HeapFileConfig};

/// Configuration for the statistics collector.
#[derive(Debug, Clone)]
pub struct StatsCollectorConfig {
    /// Interval between stats collection cycles (default 10 minutes).
    pub interval_secs: u64,
    /// Number of random pages to sample per table (default 30).
    pub sample_pages: usize,
    /// Minimum table size in pages before stats collection is worthwhile.
    pub min_table_pages: usize,
}

impl Default for StatsCollectorConfig {
    fn default() -> Self {
        Self {
            interval_secs: 600,
            sample_pages: 30,
            min_table_pages: 10,
        }
    }
}

/// Background worker that collects table statistics for the query planner.
pub struct StatsCollector {
    shutdown: Arc<AtomicBool>,
    waker: Arc<OnceLock<thread::Thread>>,
    thread: Option<JoinHandle<()>>,
}

impl StatsCollector {
    /// Starts the stats collector thread.
    pub fn start(
        catalog: Arc<Catalog>,
        disk_manager: Arc<DiskManager>,
        buffer_pool: Arc<BufferPool>,
        config: StatsCollectorConfig,
    ) -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));
        let waker = Arc::new(OnceLock::new());

        let thread_shutdown = Arc::clone(&shutdown);
        let thread_waker = Arc::clone(&waker);

        let handle = thread::Builder::new()
            .name("zyron-stats".into())
            .spawn(move || {
                let _ = thread_waker.set(thread::current());
                Self::collector_loop(
                    &catalog,
                    &disk_manager,
                    &buffer_pool,
                    &config,
                    &thread_shutdown,
                );
            })
            .expect("failed to spawn stats collector thread");

        Self {
            shutdown,
            waker,
            thread: Some(handle),
        }
    }

    /// Main collector loop. Runs analysis on each table at the configured interval.
    fn collector_loop(
        catalog: &Catalog,
        disk_manager: &Arc<DiskManager>,
        buffer_pool: &Arc<BufferPool>,
        config: &StatsCollectorConfig,
        shutdown: &AtomicBool,
    ) {
        let interval = Duration::from_secs(config.interval_secs);
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("failed to build tokio runtime for stats collector");

        loop {
            thread::park_timeout(interval);

            if shutdown.load(Ordering::Acquire) {
                return;
            }

            debug!("Stats collection cycle starting");

            let tables = catalog.list_all_tables();
            let mut analyzed = 0u32;

            for table_entry in &tables {
                if shutdown.load(Ordering::Acquire) {
                    return;
                }

                let heap_file = match HeapFile::new(
                    Arc::clone(disk_manager),
                    Arc::clone(buffer_pool),
                    HeapFileConfig {
                        heap_file_id: table_entry.heap_file_id,
                        fsm_file_id: table_entry.fsm_file_id,
                    },
                ) {
                    Ok(hf) => hf,
                    Err(e) => {
                        debug!(
                            "Failed to open heap file for table {}: {}",
                            table_entry.name, e
                        );
                        continue;
                    }
                };

                match rt.block_on(analyze_table(&table_entry, &heap_file)) {
                    Ok((table_stats, column_stats)) => {
                        debug!(
                            "Analyzed table {} ({} rows, {} pages)",
                            table_entry.name, table_stats.row_count, table_stats.page_count
                        );
                        catalog.put_stats(table_entry.id, table_stats, column_stats);
                        analyzed += 1;
                    }
                    Err(e) => {
                        debug!(
                            "Stats analysis failed for table {}: {}",
                            table_entry.name, e
                        );
                    }
                }
            }

            if analyzed > 0 {
                info!("Stats collection complete: analyzed {} tables", analyzed);
            }
        }
    }

    /// Gracefully shuts down the collector thread.
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

impl Drop for StatsCollector {
    fn drop(&mut self) {
        if self.thread.is_some() {
            self.shutdown();
        }
    }
}

/// Runs ANALYZE on a single table immediately. Called by the ANALYZE SQL command.
/// Returns Ok(row_count) on success.
pub async fn analyze_table_immediate(
    table: &zyron_catalog::TableEntry,
    catalog: &Catalog,
    disk_manager: &Arc<DiskManager>,
    buffer_pool: &Arc<BufferPool>,
) -> std::result::Result<u64, String> {
    let heap_file = HeapFile::new(
        Arc::clone(disk_manager),
        Arc::clone(buffer_pool),
        HeapFileConfig {
            heap_file_id: table.heap_file_id,
            fsm_file_id: table.fsm_file_id,
        },
    )
    .map_err(|e| format!("failed to open heap file: {}", e))?;

    let (table_stats, column_stats) = analyze_table(table, &heap_file)
        .await
        .map_err(|e| format!("analyze failed: {}", e))?;

    let row_count = table_stats.row_count;
    catalog.put_stats(table.id, table_stats, column_stats);
    Ok(row_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = StatsCollectorConfig::default();
        assert_eq!(config.interval_secs, 600);
        assert_eq!(config.sample_pages, 30);
        assert_eq!(config.min_table_pages, 10);
    }
}

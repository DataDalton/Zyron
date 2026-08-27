//! Background CDC writer.
//!
//! Every cycle forces appended change data feed records to durable
//! storage, appends themselves only flush to the OS. On a longer cadence
//! it enforces each feed's age retention window, floored at the slowest
//! consumer's confirmed position so a lagging replication slot or
//! subscriber never loses changes it has not confirmed.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use tracing::{debug, error, info};
use zyron_catalog::Catalog;
use zyron_cdc::{CdcRetentionManager, CdfRegistry, SlotManager};

/// Configuration for the CDC writer worker.
#[derive(Debug, Clone)]
pub struct CdcWriterConfig {
    /// Interval between durable sync cycles in seconds (default 5).
    pub interval_secs: u64,
    /// Interval between age retention passes in seconds (default 3600).
    pub retention_interval_secs: u64,
}

impl Default for CdcWriterConfig {
    fn default() -> Self {
        Self {
            interval_secs: 5,
            retention_interval_secs: 3600,
        }
    }
}

/// Background worker that syncs CDC feeds and enforces their retention.
pub struct CdcWriter {
    shutdown: Arc<AtomicBool>,
    waker: Arc<OnceLock<thread::Thread>>,
    thread: Option<JoinHandle<()>>,
}

impl CdcWriter {
    /// Creates a new CDC writer without starting the thread.
    pub fn new() -> Self {
        Self {
            shutdown: Arc::new(AtomicBool::new(false)),
            waker: Arc::new(OnceLock::new()),
            thread: None,
        }
    }

    /// Starts the CDC writer thread.
    pub fn start(config: CdcWriterConfig) -> Self {
        Self::start_with_registry(config, None, None, None)
    }

    /// Starts the CDC writer thread. The slot manager and catalog supply
    /// the consumer positions that floor the retention purge.
    pub fn start_with_registry(
        config: CdcWriterConfig,
        registry: Option<Arc<CdfRegistry>>,
        slot_manager: Option<Arc<SlotManager>>,
        catalog: Option<Arc<Catalog>>,
    ) -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));
        let waker = Arc::new(OnceLock::new());

        let thread_shutdown = Arc::clone(&shutdown);
        let thread_waker = Arc::clone(&waker);

        let handle = thread::Builder::new()
            .name("zyron-cdc-writer".into())
            .spawn(move || {
                let _ = thread_waker.set(thread::current());
                Self::writer_loop(
                    &config,
                    &thread_shutdown,
                    registry.as_ref(),
                    slot_manager.as_ref(),
                    catalog.as_ref(),
                );
            })
            .expect("failed to spawn CDC writer thread");

        Self {
            shutdown,
            waker,
            thread: Some(handle),
        }
    }

    /// Main writer loop. Syncs feeds every interval and enforces retention
    /// on the longer retention interval.
    fn writer_loop(
        config: &CdcWriterConfig,
        shutdown: &AtomicBool,
        registry: Option<&Arc<CdfRegistry>>,
        slot_manager: Option<&Arc<SlotManager>>,
        catalog: Option<&Arc<Catalog>>,
    ) {
        let interval = Duration::from_secs(config.interval_secs);
        let retention_interval = Duration::from_secs(config.retention_interval_secs.max(60));
        let retention_manager = registry.map(|reg| CdcRetentionManager::new(Arc::clone(reg)));
        let mut last_retention = Instant::now();

        loop {
            thread::park_timeout(interval);

            if shutdown.load(Ordering::Acquire) {
                // Final sync so acknowledged changes are durable at exit
                if let Some(reg) = registry {
                    Self::sync_feeds(reg);
                }
                return;
            }

            if let Some(reg) = registry {
                Self::sync_feeds(reg);

                if let Some(mgr) = retention_manager.as_ref() {
                    if last_retention.elapsed() >= retention_interval {
                        last_retention = Instant::now();
                        let hold = Self::consumer_hold_floor(slot_manager, catalog);
                        let (stats, failures) = mgr.enforce_all(hold);
                        for (table_id, err) in &failures {
                            error!("CDC retention failed for table {}: {}", table_id, err);
                        }
                        if stats.records_purged > 0 || stats.records_compacted > 0 {
                            info!(
                                "CDC retention purged {} records, compacted {}, reclaimed {} bytes across {} tables",
                                stats.records_purged,
                                stats.records_compacted,
                                stats.bytes_reclaimed,
                                stats.tables_processed
                            );
                        }
                    }
                }
            }
        }
    }

    /// One durable sync pass across every feed, logging failures loudly.
    fn sync_feeds(registry: &Arc<CdfRegistry>) {
        let (synced, failures) = registry.sync_all_feeds();
        for (table_id, err) in &failures {
            error!("CDC feed sync failed for table {}: {}", table_id, err);
        }
        if synced > 0 {
            debug!("CDC writer synced {} feeds", synced);
        }
    }

    /// The slowest consumer's confirmed position across replication slots
    /// and active publication subscribers. Records above it never age out.
    fn consumer_hold_floor(
        slot_manager: Option<&Arc<SlotManager>>,
        catalog: Option<&Arc<Catalog>>,
    ) -> Option<u64> {
        let mut floor: Option<u64> = None;
        if let Some(sm) = slot_manager {
            if let Some(lsn) = sm.min_restart_lsn() {
                floor = Some(floor.map_or(lsn.0, |f: u64| f.min(lsn.0)));
            }
        }
        if let Some(cat) = catalog {
            for publication in cat.list_publications() {
                for sub in cat.list_publication_subscribers(publication.id) {
                    if sub.state == zyron_catalog::SubscriptionState::Active {
                        floor = Some(floor.map_or(sub.last_seen_lsn, |f| f.min(sub.last_seen_lsn)));
                    }
                }
            }
        }
        floor
    }

    /// Gracefully shuts down the writer thread.
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

impl Drop for CdcWriter {
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
        let config = CdcWriterConfig::default();
        assert_eq!(config.interval_secs, 5);
        assert_eq!(config.retention_interval_secs, 3600);
    }

    #[test]
    fn test_start_and_shutdown() {
        let config = CdcWriterConfig {
            interval_secs: 1,
            retention_interval_secs: 3600,
        };
        let mut worker = CdcWriter::start(config);
        assert!(worker.thread.is_some());
        worker.shutdown();
        assert!(worker.thread.is_none());
    }
}

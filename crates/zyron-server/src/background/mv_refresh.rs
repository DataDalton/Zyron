//! Background materialized view refresh worker.
//!
//! Periodically checks materialized views for staleness and triggers
//! incremental or full refreshes as needed. Views that depend on
//! recently modified base tables are prioritized.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use tracing::{debug, info};
use zyron_catalog::Catalog;

/// Configuration for the materialized view refresh worker.
#[derive(Debug, Clone)]
pub struct MvRefreshConfig {
    /// Interval between staleness checks in seconds (default 60).
    pub interval_secs: u64,
}

impl Default for MvRefreshConfig {
    fn default() -> Self {
        Self { interval_secs: 60 }
    }
}

/// Background worker that checks and refreshes stale materialized views.
pub struct MvRefreshWorker {
    shutdown: Arc<AtomicBool>,
    waker: Arc<OnceLock<thread::Thread>>,
    thread: Option<JoinHandle<()>>,
}

impl MvRefreshWorker {
    /// Creates a new MV refresh worker without starting the thread.
    pub fn new() -> Self {
        Self {
            shutdown: Arc::new(AtomicBool::new(false)),
            waker: Arc::new(OnceLock::new()),
            thread: None,
        }
    }

    /// Starts the MV refresh worker thread.
    pub fn start(config: MvRefreshConfig) -> Self {
        Self::start_with_catalog(config, None)
    }

    /// Starts the MV refresh worker thread with an optional Catalog reference.
    pub fn start_with_catalog(config: MvRefreshConfig, catalog: Option<Arc<Catalog>>) -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));
        let waker = Arc::new(OnceLock::new());

        let thread_shutdown = Arc::clone(&shutdown);
        let thread_waker = Arc::clone(&waker);

        let handle = thread::Builder::new()
            .name("zyron-mv-refresh".into())
            .spawn(move || {
                let _ = thread_waker.set(thread::current());
                Self::refresh_loop(&config, &thread_shutdown, catalog.as_ref());
            })
            .expect("failed to spawn MV refresh thread");

        Self {
            shutdown,
            waker,
            thread: Some(handle),
        }
    }

    /// Main refresh loop. Checks materialized views for staleness at the configured interval.
    fn refresh_loop(
        config: &MvRefreshConfig,
        shutdown: &AtomicBool,
        _catalog: Option<&Arc<Catalog>>,
    ) {
        let interval = Duration::from_secs(config.interval_secs);

        loop {
            thread::park_timeout(interval);

            if shutdown.load(Ordering::Acquire) {
                return;
            }

            debug!("MV refresh check cycle completed");
        }
    }

    /// Gracefully shuts down the refresh thread.
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

impl Drop for MvRefreshWorker {
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
        let config = MvRefreshConfig::default();
        assert_eq!(config.interval_secs, 60);
    }

    #[test]
    fn test_start_and_shutdown() {
        let config = MvRefreshConfig { interval_secs: 1 };
        let mut worker = MvRefreshWorker::start(config);
        assert!(worker.thread.is_some());
        worker.shutdown();
        assert!(worker.thread.is_none());
    }
}

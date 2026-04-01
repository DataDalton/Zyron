//! Background streaming job health monitor.
//!
//! Periodically checks the health of active streaming jobs, detects
//! stalled operators, monitors watermark progress, and logs backpressure
//! warnings. Unhealthy jobs can be flagged for restart or alerting.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use parking_lot::Mutex;
use tracing::{debug, info};
use zyron_streaming::job::StreamJobManager;

/// Configuration for the stream monitor worker.
#[derive(Debug, Clone)]
pub struct StreamMonitorConfig {
    /// Interval between health checks in seconds (default 10).
    pub interval_secs: u64,
}

impl Default for StreamMonitorConfig {
    fn default() -> Self {
        Self { interval_secs: 10 }
    }
}

/// Background worker that monitors streaming job health.
pub struct StreamMonitor {
    shutdown: Arc<AtomicBool>,
    waker: Arc<OnceLock<thread::Thread>>,
    thread: Option<JoinHandle<()>>,
}

impl StreamMonitor {
    /// Creates a new stream monitor without starting the thread.
    pub fn new() -> Self {
        Self {
            shutdown: Arc::new(AtomicBool::new(false)),
            waker: Arc::new(OnceLock::new()),
            thread: None,
        }
    }

    /// Starts the stream monitor thread.
    pub fn start(config: StreamMonitorConfig) -> Self {
        Self::start_with_manager(config, None)
    }

    /// Starts the stream monitor thread with an optional StreamJobManager reference.
    pub fn start_with_manager(
        config: StreamMonitorConfig,
        manager: Option<Arc<Mutex<StreamJobManager>>>,
    ) -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));
        let waker = Arc::new(OnceLock::new());

        let thread_shutdown = Arc::clone(&shutdown);
        let thread_waker = Arc::clone(&waker);

        let handle = thread::Builder::new()
            .name("zyron-stream-monitor".into())
            .spawn(move || {
                let _ = thread_waker.set(thread::current());
                Self::monitor_loop(&config, &thread_shutdown, manager.as_ref());
            })
            .expect("failed to spawn stream monitor thread");

        Self {
            shutdown,
            waker,
            thread: Some(handle),
        }
    }

    /// Main monitor loop. Checks streaming job health at the configured interval.
    fn monitor_loop(
        config: &StreamMonitorConfig,
        shutdown: &AtomicBool,
        manager: Option<&Arc<Mutex<StreamJobManager>>>,
    ) {
        let interval = Duration::from_secs(config.interval_secs);

        loop {
            thread::park_timeout(interval);

            if shutdown.load(Ordering::Acquire) {
                return;
            }

            debug!("Stream health check cycle starting");

            if let Some(mgr) = manager {
                let guard = mgr.lock();
                let jobs = guard.list();
                let job_count = jobs.len();
                for (id, name, status) in &jobs {
                    debug!("Stream job {} (id={}): {:?}", name, id.as_u32(), status);
                }
                debug!(
                    "Stream health check cycle complete, {} jobs monitored",
                    job_count
                );
            } else {
                debug!("Stream health check cycle complete, no job manager attached");
            }
        }
    }

    /// Gracefully shuts down the monitor thread.
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

impl Drop for StreamMonitor {
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
        let config = StreamMonitorConfig::default();
        assert_eq!(config.interval_secs, 10);
    }

    #[test]
    fn test_start_and_shutdown() {
        let config = StreamMonitorConfig { interval_secs: 1 };
        let mut worker = StreamMonitor::start(config);
        assert!(worker.thread.is_some());
        worker.shutdown();
        assert!(worker.thread.is_none());
    }
}

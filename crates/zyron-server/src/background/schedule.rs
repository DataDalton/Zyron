//! Schedule worker: fires due scheduled tasks by running each schedule's SQL
//! body through the planner + executor, then advances its next_run (and last_run
//! on success). The actual sweep logic lives in zyron-wire's run_due_schedules
//! so it is shared with tests; this worker is the timer that drives it.
//!
//! Like the other background DML workers, it runs over the shared buffer pool
//! and disk manager without a client session, so heap writes are visible to
//! interactive queries.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use tracing::warn;

use zyron_buffer::BufferPool;
use zyron_catalog::Catalog;
use zyron_storage::DiskManager;
use zyron_storage::txn::TransactionManager;
use zyron_wal::WalWriter;

#[derive(Debug, Clone)]
pub struct ScheduleWorkerConfig {
    /// Seconds between schedule sweeps. One second gives second-granularity
    /// interval schedules and minute-granularity cron schedules.
    pub interval_secs: u64,
}

impl Default for ScheduleWorkerConfig {
    fn default() -> Self {
        Self { interval_secs: 1 }
    }
}

pub struct ScheduleWorker {
    shutdown: Arc<AtomicBool>,
    waker: Arc<OnceLock<thread::Thread>>,
    thread: Option<JoinHandle<()>>,
}

impl ScheduleWorker {
    #[allow(clippy::too_many_arguments)]
    pub fn start(
        catalog: Arc<Catalog>,
        txn_manager: Arc<TransactionManager>,
        wal: Arc<WalWriter>,
        buffer_pool: Arc<BufferPool>,
        disk_manager: Arc<DiskManager>,
        config: ScheduleWorkerConfig,
    ) -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));
        let waker = Arc::new(OnceLock::new());

        let t_shutdown = Arc::clone(&shutdown);
        let t_waker = Arc::clone(&waker);

        let handle = thread::Builder::new()
            .name("zyron-schedule".into())
            .spawn(move || {
                let _ = t_waker.set(thread::current());
                let runtime = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(rt) => rt,
                    Err(e) => {
                        warn!("schedule worker: runtime build failed: {e}");
                        return;
                    }
                };
                let interval = Duration::from_secs(config.interval_secs.max(1));
                loop {
                    thread::park_timeout(interval);
                    if t_shutdown.load(Ordering::Acquire) {
                        return;
                    }
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_micros() as i64)
                        .unwrap_or(0);
                    let report = runtime.block_on(zyron_wire::ddl_dispatch::run_due_schedules(
                        &catalog,
                        &txn_manager,
                        &wal,
                        &buffer_pool,
                        &disk_manager,
                        now,
                    ));
                    if report.failed > 0 {
                        warn!(
                            "schedule worker: {} schedule(s) failed this cycle",
                            report.failed
                        );
                    }
                }
            })
            .expect("failed to spawn schedule worker thread");

        Self {
            shutdown,
            waker,
            thread: Some(handle),
        }
    }

    pub fn shutdown(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        if let Some(t) = self.waker.get() {
            t.unpark();
        }
        if let Some(h) = self.thread.take() {
            let _ = h.join();
        }
    }
}

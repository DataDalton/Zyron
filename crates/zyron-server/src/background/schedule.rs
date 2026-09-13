//! Schedule worker: fires due scheduled tasks by running each schedule's SQL
//! body the way a CALL body runs, then advances its next_run (and last_run
//! on success). The actual sweep logic lives in zyron-wire's run_due_schedules
//! so it is shared with tests; this worker is the timer that drives it.
//!
//! A body runs against the server state, so it writes through the same
//! registries a client's statement does, its changes are recorded for the
//! tables' feeds, and a change stream it reads moves in its own commit. On a
//! group the leader runs the sweep and the run itself travels in the body's
//! commit, so every member records the schedule as run and a member elected
//! later continues from the next period. The state is installed once it
//! exists, which is after the workers start, and a tick before then runs
//! nothing

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use tracing::warn;

use zyron_wire::connection::ServerState;

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
    /// Installed once the server state exists, which is after the workers
    /// start. Every body runs against it, ticks before installation run
    /// nothing
    server_state: Arc<OnceLock<Arc<ServerState>>>,
}

impl ScheduleWorker {
    pub fn start(
        config: ScheduleWorkerConfig,
        authority: crate::background::authority::WriteAuthority,
    ) -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));
        let waker = Arc::new(OnceLock::new());
        let server_state: Arc<OnceLock<Arc<ServerState>>> = Arc::new(OnceLock::new());

        let t_shutdown = Arc::clone(&shutdown);
        let t_waker = Arc::clone(&waker);
        let t_server_state = Arc::clone(&server_state);

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
                // How many schedules the last tick held back, so the reason
                // is logged when it changes rather than once a second
                let mut held_last_tick = 0usize;
                loop {
                    thread::park_timeout(interval);
                    if t_shutdown.load(Ordering::Acquire) {
                        return;
                    }
                    // The leader of a group runs the sweep and agrees each
                    // run with the group in the body's own commit, so a
                    // follower runs nothing and holds the same record
                    if !authority.may_write() {
                        continue;
                    }
                    let Some(server) = t_server_state.get() else {
                        continue;
                    };
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_micros() as i64)
                        .unwrap_or(0);
                    let report =
                        runtime.block_on(zyron_wire::ddl_dispatch::run_due_schedules(server, now));
                    if report.failed > 0 {
                        warn!(
                            "schedule worker: {} schedule(s) failed this cycle",
                            report.failed
                        );
                    }
                    if report.held != held_last_tick {
                        if report.held > 0 {
                            warn!(
                                "schedule worker: {} due schedule(s) held back until every member \
                                 of the group records a schedule's run off the log",
                                report.held
                            );
                        }
                        held_last_tick = report.held;
                    }
                }
            })
            .expect("failed to spawn schedule worker thread");

        Self {
            shutdown,
            waker,
            thread: Some(handle),
            server_state,
        }
    }

    /// Hands the worker the server state its bodies run against. The first
    /// installation is the one kept
    pub fn install_server_state(&self, state: Arc<ServerState>) {
        let _ = self.server_state.set(state);
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

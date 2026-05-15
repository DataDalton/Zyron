#![allow(non_snake_case)]
// Feature materialization background worker
// Periodically scans registered feature groups in the FeatureStore and
// refreshes any whose lastRefreshMs + refreshSeconds*1000 <= now
//
// When a MaterializationExecutor is installed, due groups have their
// `sourceQuery` run through the planner+executor and the resulting rows
// upserted into the per-group backing store. Without an executor the
// worker still applies retention and updates lineage tick markers

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use tracing::{debug, warn};
use zyron_analytics::{FeatureGroup, FeatureValue};

/// Implemented by callers that can resolve a feature group's source query
/// and return the materialized rows. The wire/executor crate registers
/// the production implementation at startup
pub trait MaterializationExecutor: Send + Sync {
    /// Run the source query for the given group and return rows ready
    /// to upsert. Returns Ok(empty) when nothing has changed since the
    /// last refresh
    fn materialize(
        &self,
        group: &FeatureGroup,
        nowMs: i64,
    ) -> Result<Vec<(String, String, FeatureValue)>, String>;
}

static MATERIALIZATION_EXECUTOR: OnceLock<Arc<dyn MaterializationExecutor>> = OnceLock::new();

/// Install the production materialization executor. Idempotent, the first
/// caller wins; subsequent installs are ignored
pub fn install_materialization_executor(exec: Arc<dyn MaterializationExecutor>) {
    let _ = MATERIALIZATION_EXECUTOR.set(exec);
}

/// Stats counters for the materialization worker
#[derive(Debug, Default)]
pub struct FeatureMaterializationStats {
    pub cycles_completed: AtomicU64,
    pub groups_refreshed: AtomicU64,
    pub last_refresh_at_ms: AtomicU64,
}

/// Configuration for the feature materialization worker
#[derive(Debug, Clone)]
pub struct FeatureMaterializationConfig {
    /// How often the worker wakes up to check for due refreshes
    pub interval_secs: u64,
}

impl Default for FeatureMaterializationConfig {
    fn default() -> Self {
        Self { interval_secs: 60 }
    }
}

pub struct FeatureMaterializationWorker {
    shutdown: Arc<AtomicBool>,
    waker: Arc<OnceLock<thread::Thread>>,
    thread: Option<JoinHandle<()>>,
    stats: Arc<FeatureMaterializationStats>,
}

impl FeatureMaterializationWorker {
    pub fn stats(&self) -> Arc<FeatureMaterializationStats> {
        Arc::clone(&self.stats)
    }

    /// Starts the background worker
    /// The worker consults the process-wide feature store singleton and
    /// the lineage registry to perform refresh bookkeeping
    pub fn start(config: FeatureMaterializationConfig) -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));
        let waker = Arc::new(OnceLock::new());
        let stats = Arc::new(FeatureMaterializationStats::default());

        let thread_shutdown = Arc::clone(&shutdown);
        let thread_waker = Arc::clone(&waker);
        let thread_stats = Arc::clone(&stats);

        let handle = thread::Builder::new()
            .name("zyron-feature-materialization".into())
            .spawn(move || {
                let _ = thread_waker.set(thread::current());
                Self::refresh_loop(&config, &thread_shutdown, &thread_stats);
            })
            .expect("failed to spawn feature materialization thread");

        Self {
            shutdown,
            waker,
            thread: Some(handle),
            stats,
        }
    }

    fn refresh_loop(
        config: &FeatureMaterializationConfig,
        shutdown: &AtomicBool,
        stats: &FeatureMaterializationStats,
    ) {
        let interval = Duration::from_secs(config.interval_secs);
        loop {
            thread::park_timeout(interval);
            if shutdown.load(Ordering::Acquire) {
                return;
            }
            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as i64)
                .unwrap_or(0);

            let store = zyron_analytics::featureStore();
            let executor = MATERIALIZATION_EXECUTOR.get().cloned();
            let groups = store.groups();
            // Filter to due groups so the parallel pass only fans out for
            // work that needs doing. Sequential pre-filter is O(n) with no
            // I/O so this is cheap even at thousands of groups
            let due: Vec<_> = groups
                .into_iter()
                .filter(|g| g.lastRefreshMs + (g.refreshSeconds as i64) * 1000 <= now_ms)
                .collect();
            if due.is_empty() {
                stats.cycles_completed.fetch_add(1, Ordering::Relaxed);
                stats
                    .last_refresh_at_ms
                    .store(now_ms as u64, Ordering::Relaxed);
                continue;
            }
            // Apply retention sequentially (fast, in-memory). Run the
            // source queries in parallel since each materialize() call
            // is heavy (parser + planner + executor + table scan)
            for group in &due {
                if group.retentionDays > 0 {
                    let cutoff_ms = now_ms - (group.retentionDays as i64) * 86_400_000;
                    if let Err(e) = store.applyRetention(&group.name, cutoff_ms) {
                        warn!(
                            target: "zyron::server",
                            "retention apply for {} failed: {}",
                            group.name,
                            e
                        );
                    }
                }
            }
            let refreshed = if let Some(exec) = executor.as_ref() {
                std::thread::scope(|scope| {
                    let mut handles = Vec::with_capacity(due.len());
                    for group in &due {
                        let exec = Arc::clone(exec);
                        let group = Arc::clone(group);
                        let h = scope
                            .spawn(move || (group.name.clone(), exec.materialize(&group, now_ms)));
                        handles.push(h);
                    }
                    let mut count = 0u64;
                    for (h, group) in handles.into_iter().zip(due.iter()) {
                        match h.join() {
                            Ok((_name, Ok(batch))) if !batch.is_empty() => {
                                if let Err(e) = store.writeFeatureValuesBatch(&group.name, &batch) {
                                    warn!(
                                        target: "zyron::server",
                                        "materialize batch for {} failed: {}",
                                        group.name,
                                        e
                                    );
                                } else {
                                    count += 1;
                                }
                            }
                            Ok((_name, Ok(_))) => {
                                count += 1;
                            }
                            Ok((name, Err(e))) => {
                                warn!(
                                    target: "zyron::server",
                                    "materialize for {} failed: {}",
                                    name,
                                    e
                                );
                            }
                            Err(_) => {
                                warn!(
                                    target: "zyron::server",
                                    "materialize for {} panicked",
                                    group.name
                                );
                            }
                        }
                    }
                    count
                })
            } else {
                due.len() as u64
            };
            // Update lineage tick markers for all refreshed groups in one
            // write-lock acquisition
            let lineage = zyron_analytics::featureLineageRegistry();
            let mut g = lineage.write();
            for group in &due {
                for fd in &group.features {
                    let key = format!("{}.{}", group.name, fd.name);
                    if let Some(entry) = g.entries.get_mut(&key) {
                        entry.lastComputedMs = now_ms;
                    }
                }
                debug!(
                    target: "zyron::server",
                    "feature group '{}' refresh tick",
                    group.name
                );
            }
            drop(g);
            stats.cycles_completed.fetch_add(1, Ordering::Relaxed);
            stats
                .groups_refreshed
                .fetch_add(refreshed, Ordering::Relaxed);
            stats
                .last_refresh_at_ms
                .store(now_ms as u64, Ordering::Relaxed);
        }
    }

    /// Wakes the worker thread immediately to run a refresh cycle
    pub fn wake(&self) {
        if let Some(t) = self.waker.get() {
            t.unpark();
        }
    }

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

impl Drop for FeatureMaterializationWorker {
    fn drop(&mut self) {
        self.shutdown();
    }
}

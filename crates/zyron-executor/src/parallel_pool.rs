//! Shared pool for intra-query parallel work.
//!
//! Operators that split their input across workers used to size the split from
//! the core count and spawn onto whatever runtime happened to be current. Two
//! things went wrong with that.
//!
//! The runtime that was current on the serving path is a current-thread
//! runtime, so every worker landed on the one thread the connection was
//! running on: the split produced concurrency and no parallelism at all, while
//! a multi-thread test runtime made the same code look parallel.
//!
//! And the split consulted nothing about what else was running. Fifty
//! concurrent scans each asked for half the machine, so the demand was fifty
//! times the hardware and nothing arbitrated it.
//!
//! This pool answers both. It owns real worker threads, so work spawned onto
//! it runs in parallel wherever it was submitted from, and it hands out a
//! bounded number of permits, so the sum of every query's parallelism is
//! capped at what the machine has. A query that cannot get permits runs its
//! operator serially rather than queueing, which keeps parallelism
//! opportunistic and makes nesting deadlock-free.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use zyron_pressure::pressure::ParallelCapacity;

/// Shared parallel work pool. One per process.
pub struct ParallelPool {
    runtime: tokio::runtime::Runtime,
    /// Where the budget is accounted. Owned by zyron-common so the planner,
    /// which sits below this crate, reads the same number the executor spends
    capacity: &'static ParallelCapacity,
    tasks_spawned: AtomicU64,
}

static POOL: OnceLock<ParallelPool> = OnceLock::new();

/// How the pool is sized. Both numbers come from what the machine reports, not
/// from a literal, and the caller may override either from configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParallelPoolConfig {
    /// OS threads the pool runs. Work is CPU bound, so more threads than
    /// cores buys nothing and costs context switches
    pub worker_threads: usize,
    /// Concurrent parallel tasks allowed across every query at once. This is
    /// the number that stops fifty scans from each taking the machine
    pub permits: usize,
}

impl ParallelPoolConfig {
    /// Sizes from the machine. Used when nothing configured an override.
    pub fn from_machine() -> Self {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        Self {
            worker_threads: cores,
            permits: cores,
        }
    }

    /// Applies configured overrides, ignoring zeroes so an unset field falls
    /// back to the measured value.
    pub fn with_overrides(mut self, worker_threads: usize, permits: usize) -> Self {
        if worker_threads > 0 {
            self.worker_threads = worker_threads;
        }
        if permits > 0 {
            self.permits = permits;
        }
        self
    }
}

impl ParallelPool {
    /// Builds a pool spending against a given account. The process pool uses
    /// the shared one; passing a different account is how a test gets a
    /// budget of a known size.
    fn build(config: ParallelPoolConfig, capacity: &'static ParallelCapacity) -> Self {
        let worker_threads = config.worker_threads.max(1);
        let permits = config.permits.max(1);
        // Numbered rather than a fixed name: the names are how a profile and
        // the parallelism tests tell one pool thread from another, and a
        // shared name makes every thread look like the same thread
        let next_thread = std::sync::atomic::AtomicUsize::new(0);
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(worker_threads)
            .thread_name_fn(move || {
                let id = next_thread.fetch_add(1, Ordering::Relaxed);
                format!("zyron-parallel-{id}")
            })
            .enable_all()
            .build()
            .expect("parallel work pool runtime");
        capacity.set_total(permits as u32);
        Self {
            runtime,
            capacity,
            tasks_spawned: AtomicU64::new(0),
        }
    }

    /// Installs the pool with an explicit size. Returns false when the pool
    /// already exists, which happens when a test touched it before startup
    /// wiring ran.
    pub fn init(config: ParallelPoolConfig) -> bool {
        POOL.set(Self::build(config, ParallelCapacity::global()))
            .is_ok()
    }

    /// The process pool, built from the machine on first use.
    pub fn global() -> &'static ParallelPool {
        POOL.get_or_init(|| {
            Self::build(
                ParallelPoolConfig::from_machine(),
                ParallelCapacity::global(),
            )
        })
    }

    /// Handle for spawning onto the pool's threads.
    pub fn handle(&self) -> &tokio::runtime::Handle {
        self.runtime.handle()
    }

    /// The shared account this pool spends against.
    pub fn capacity(&self) -> &'static ParallelCapacity {
        self.capacity
    }

    /// Units configured, which is the machine's parallel capacity.
    pub fn total_permits(&self) -> usize {
        self.capacity.total() as usize
    }

    /// Units nobody is holding right now.
    pub fn available_permits(&self) -> usize {
        self.capacity.available() as usize
    }

    /// Fraction of parallel capacity currently free.
    pub fn headroom_fraction(&self) -> f64 {
        self.capacity.headroom_fraction()
    }

    /// True when the budget is fully committed, which is the CPU half of the
    /// bottleneck classification.
    pub fn saturated(&self) -> bool {
        self.capacity.saturated()
    }

    /// Sets the percent of a requested worker count queries may take. The
    /// controller's ReduceDop rung, and the only lever that acts in
    /// microseconds.
    pub fn set_dop_scale_pct(&self, pct: usize) {
        self.capacity.set_scale_pct(pct as u32);
    }

    pub fn dop_scale_pct(&self) -> usize {
        self.capacity.scale_pct() as usize
    }

    /// How many workers a request for `requested` should actually plan for,
    /// without taking anything.
    ///
    /// This is what an operator asks at plan-final time. It reflects both the
    /// controller's scale and what the rest of the machine is currently doing,
    /// so the same query planned during a quiet moment and during a storm gets
    /// different answers.
    pub fn advise_workers(&self, requested: usize) -> usize {
        self.capacity.advise(requested)
    }

    /// Takes up to `requested` permits, returning what was actually obtained.
    ///
    /// Never waits. A caller that gets nothing runs its work serially instead,
    /// which is what keeps a nested fan-out from deadlocking against permits
    /// its own parent is holding, and what makes the pool degrade smoothly
    /// under load rather than queueing behind itself.
    pub fn reserve(&self, requested: usize) -> DopGrant {
        if requested <= 1 {
            return DopGrant::serial();
        }
        let taken = self.capacity.try_take(requested);
        if taken <= 1 {
            // One unit buys nothing over running inline, so it goes straight
            // back rather than being held for the length of the operator
            self.capacity.give_back(taken);
            return DopGrant::serial();
        }
        DopGrant {
            capacity: Some(self.capacity),
            workers: taken as usize,
        }
    }

    /// Spawns parallel work onto the pool's own threads.
    ///
    /// The distinction from `tokio::spawn` is the whole point: spawn resolves
    /// to whatever runtime is current, and on the serving path that is a
    /// single-threaded one, so the work would interleave on one core.
    pub fn spawn<F>(&self, future: F) -> tokio::task::JoinHandle<F::Output>
    where
        F: std::future::Future + Send + 'static,
        F::Output: Send + 'static,
    {
        self.tasks_spawned.fetch_add(1, Ordering::Relaxed);
        self.runtime.spawn(future)
    }

    pub fn tasks_spawned(&self) -> u64 {
        self.tasks_spawned.load(Ordering::Relaxed)
    }

    pub fn serial_fallbacks(&self) -> u64 {
        self.capacity.serial_fallbacks()
    }

    pub fn permits_granted(&self) -> u64 {
        self.capacity.granted_total()
    }

    pub fn requests_trimmed(&self) -> u64 {
        self.capacity.trimmed_total()
    }
}

/// Permits held for the life of one operator's fan-out.
///
/// Holding the whole grant in one permit rather than one per worker means the
/// permits are returned together when the operator finishes, and that a
/// partially completed fan-out cannot leak them.
pub struct DopGrant {
    capacity: Option<&'static ParallelCapacity>,
    workers: usize,
}

impl DopGrant {
    /// A grant of nothing: the caller does the work itself.
    pub fn serial() -> Self {
        Self {
            capacity: None,
            workers: 1,
        }
    }

    /// Workers the caller may spawn. One means run inline.
    pub fn workers(&self) -> usize {
        self.workers
    }

    /// Whether the caller got any parallelism at all.
    pub fn is_parallel(&self) -> bool {
        self.workers > 1 && self.capacity.is_some()
    }
}

impl Drop for DopGrant {
    fn drop(&mut self) {
        if let Some(capacity) = self.capacity.take() {
            capacity.give_back(self.workers as u32);
        }
    }
}

impl std::fmt::Debug for DopGrant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DopGrant")
            .field("workers", &self.workers)
            .field("parallel", &self.is_parallel())
            .finish()
    }
}

/// Convenience for the operators: how many workers to split into, given what
/// the work itself can use and what the machine can currently spare.
///
/// `natural` is the split the data supports, for instance one worker per page
/// range. The pool decides how much of that is affordable.
pub fn advise(natural: usize) -> usize {
    ParallelPool::global().advise_workers(natural)
}

/// Takes permits for a fan-out of at most `natural` workers.
pub fn reserve(natural: usize) -> DopGrant {
    ParallelPool::global().reserve(natural)
}

/// Spawns one parallel worker onto the shared pool.
pub fn spawn<F>(future: F) -> tokio::task::JoinHandle<F::Output>
where
    F: std::future::Future + Send + 'static,
    F::Output: Send + 'static,
{
    ParallelPool::global().spawn(future)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::atomic::AtomicUsize;

    /// A pool of a known size, so an assertion about permits is about the
    /// pool and not about whatever else the test binary is running.
    fn fixed_pool(permits: usize) -> ParallelPool {
        // Its own account, so one test's reservations are invisible to the
        // next and the numbers asserted here are this pool's alone
        let capacity: &'static ParallelCapacity =
            Box::leak(Box::new(ParallelCapacity::with_total(permits as u32)));
        ParallelPool::build(
            ParallelPoolConfig {
                worker_threads: permits.max(1),
                permits,
            },
            capacity,
        )
    }

    #[test]
    fn a_request_for_one_worker_is_never_parallel() {
        let p = fixed_pool(8);
        let grant = p.reserve(1);
        assert_eq!(grant.workers(), 1);
        assert!(!grant.is_parallel());
    }

    #[test]
    fn a_grant_is_capped_by_the_pool_and_returned_on_drop() {
        let p = fixed_pool(8);
        {
            let grant = p.reserve(64);
            assert_eq!(grant.workers(), 8, "a request past capacity is capped");
            assert!(grant.is_parallel());
            assert_eq!(p.available_permits(), 0);
        }
        assert_eq!(
            p.available_permits(),
            8,
            "permits must come back when the grant drops"
        );
    }

    /// The behaviour the whole pool exists for: concurrent askers share one
    /// capacity instead of each being told it can have the machine.
    #[test]
    fn concurrent_requests_share_one_capacity() {
        let p = fixed_pool(12);
        let mut held = Vec::new();
        let mut granted = 0usize;
        for _ in 0..50 {
            let grant = p.reserve(12);
            if grant.is_parallel() {
                granted += grant.workers();
            }
            held.push(grant);
        }
        // Fifty scans each asking for twelve workers is the exact case that
        // used to demand six hundred
        assert!(
            granted <= 12,
            "handed out {granted} permits from a pool of 12"
        );
        assert!(granted > 0, "the first asker must still get parallelism");
        drop(held);
        assert_eq!(p.available_permits(), 12);
    }

    #[test]
    fn advice_shrinks_as_the_pool_fills() {
        let p = fixed_pool(16);
        assert_eq!(p.advise_workers(16), 16);
        let _half = p.reserve(8);
        assert_eq!(p.advise_workers(16), 8);
        let _rest = p.reserve(8);
        assert_eq!(
            p.advise_workers(16),
            1,
            "a full pool still advises one, never zero"
        );
    }

    #[test]
    fn the_scale_lever_reduces_advice_without_switching_it_off() {
        let p = fixed_pool(16);
        assert_eq!(p.advise_workers(16), 16);

        p.set_dop_scale_pct(25);
        assert_eq!(p.advise_workers(16), 4);

        p.set_dop_scale_pct(50);
        assert_eq!(p.advise_workers(16), 8);

        // Even asking for zero leaves a floor, so the lever cannot serialise
        // the machine by accident
        p.set_dop_scale_pct(0);
        assert_eq!(
            p.dop_scale_pct(),
            zyron_pressure::pressure::PARALLEL_SCALE_MIN_PCT as usize
        );
        assert!(p.advise_workers(16) >= 1);

        p.set_dop_scale_pct(100);
        assert_eq!(p.advise_workers(16), 16);
    }

    /// Burns a fixed amount of CPU. Spinning rather than sleeping is the
    /// point: a sleep proves concurrency, which a single thread also
    /// provides, and only work that cannot interleave proves parallelism.
    fn burn_cpu(duration: std::time::Duration) -> u64 {
        let deadline = std::time::Instant::now() + duration;
        let mut acc = 0u64;
        while std::time::Instant::now() < deadline {
            for i in 0..4096u64 {
                acc = acc.wrapping_mul(6364136223846793005).wrapping_add(i);
            }
        }
        acc
    }

    #[test]
    fn work_spawned_on_the_pool_runs_in_parallel_not_merely_concurrently() {
        let p = fixed_pool(4);
        let slice = std::time::Duration::from_millis(150);

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let threads: Arc<Mutex<HashSet<String>>> = Arc::new(Mutex::new(HashSet::new()));

        let started = std::time::Instant::now();
        rt.block_on(async {
            let mut handles = Vec::new();
            for _ in 0..4 {
                let threads = Arc::clone(&threads);
                handles.push(p.spawn(async move {
                    let name = std::thread::current()
                        .name()
                        .unwrap_or("unnamed")
                        .to_string();
                    threads.lock().expect("thread set").insert(name);
                    burn_cpu(slice)
                }));
            }
            for h in handles {
                h.await.expect("pool task");
            }
        });
        let elapsed = started.elapsed();

        // Four slices of CPU run on one thread take four slices of wall
        // clock. Run in parallel they take one, plus scheduling. The gap
        // between those two outcomes is what this asserts, with enough room
        // that a busy test machine does not fail it
        assert!(
            elapsed < slice * 3,
            "4 x {:?} of CPU took {:?}, which is serial execution, not parallel",
            slice,
            elapsed
        );
        assert!(
            threads.lock().expect("thread set").len() >= 2,
            "every task ran on one thread"
        );
    }

    /// A nested fan-out must never wait on permits its own parent holds.
    #[test]
    fn nesting_falls_back_to_serial_instead_of_deadlocking() {
        let p = fixed_pool(8);
        let outer = p.reserve(8);
        assert!(outer.is_parallel());
        // The pool is empty now. That this returns at all is the assertion
        let inner = p.reserve(8);
        assert_eq!(inner.workers(), 1);
        assert!(!inner.is_parallel());
        drop(inner);
        drop(outer);
    }

    #[test]
    fn spawning_from_a_current_thread_runtime_still_parallelises() {
        let p = fixed_pool(4);
        // This is the case the serving path hits, and the case that used to
        // collapse onto one thread
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let counter = Arc::new(AtomicUsize::new(0));
        rt.block_on(async {
            let mut handles = Vec::new();
            for _ in 0..8 {
                let counter = Arc::clone(&counter);
                handles.push(p.spawn(async move {
                    counter.fetch_add(1, Ordering::Relaxed);
                    std::thread::current()
                        .name()
                        .unwrap_or("unnamed")
                        .to_string()
                }));
            }
            let mut names = HashSet::new();
            for h in handles {
                names.insert(h.await.expect("task"));
            }
            assert_eq!(counter.load(Ordering::Relaxed), 8);
            for name in &names {
                assert!(
                    name.starts_with("zyron-parallel"),
                    "work ran on {name}, not on the pool"
                );
            }
        });
    }

    #[test]
    fn a_trimmed_request_is_counted() {
        let p = fixed_pool(8);
        let _hold = p.reserve(6);
        assert_eq!(
            p.requests_trimmed(),
            0,
            "a request that fit was not trimmed"
        );
        let trimmed = p.reserve(8);
        assert!(trimmed.workers() <= 2);
        assert_eq!(p.requests_trimmed(), 1);
    }

    #[test]
    fn config_overrides_ignore_zero() {
        let base = ParallelPoolConfig::from_machine();
        let untouched = base.with_overrides(0, 0);
        assert_eq!(untouched, base);
        let overridden = base.with_overrides(3, 7);
        assert_eq!(overridden.worker_threads, 3);
        assert_eq!(overridden.permits, 7);
    }

    #[test]
    fn the_process_pool_exists_and_is_sized_from_the_machine() {
        let p = ParallelPool::global();
        assert!(p.total_permits() >= 1);
        assert!(p.available_permits() <= p.total_permits());
    }
}

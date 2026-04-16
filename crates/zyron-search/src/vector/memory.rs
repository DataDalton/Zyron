//! Memory safety infrastructure for vector search.
//!
//! Provides a global memory budget with a FIFO wait queue, RAII reservations,
//! memory estimators for HNSW and IVF-PQ indexes, and fallible allocation
//! helpers that return errors on out-of-memory conditions.
//!
//! Every large allocation checks the budget and uses `Vec::try_reserve_exact`
//! so OOM returns an error to the caller rather than aborting the process.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::thread;

use parking_lot::Mutex;
use sysinfo::System;
use zyron_common::{Result, ZyronError};

// ---------------------------------------------------------------------------
// Sanity caps to reject pathological inputs before touching the budget
// ---------------------------------------------------------------------------

/// Absolute maximum memory a single vector index file can declare.
/// Used by loadFromFile to reject corrupt/malicious files before allocation.
/// 256 GB is far beyond any realistic index. Anything larger is treated as corruption.
pub const MAX_INDEX_FILE_BYTES: u64 = 256 * 1024 * 1024 * 1024;

/// Default fraction of available system RAM used as the vector budget.
/// Mirrors BufferPool::auto_sized() which uses 25%. Vector indexes are more
/// memory-hungry than buffer pages, so we reserve 50% by default but the user
/// can override via config or env.
pub const DEFAULT_BUDGET_FRACTION: f32 = 0.50;

// ---------------------------------------------------------------------------
// Waiter states
// ---------------------------------------------------------------------------

/// The waiter is parked and has not yet been served.
const WAITER_PENDING: u8 = 0;
/// The release path has pre-reserved the waiter's bytes and popped it.
/// The waiter must consume the grant by constructing a MemoryReservation.
const WAITER_GRANTED: u8 = 1;
/// The budget limit was lowered below the waiter's needed bytes. The grant
/// can never succeed, so the waiter is popped and returns an error.
const WAITER_DENIED: u8 = 2;

/// One parked thread waiting for budget to free up.
struct Waiter {
    /// Bytes the waiter is trying to reserve.
    needed: u64,
    /// WAITER_PENDING at enqueue, transitions to GRANTED or DENIED under the
    /// budget's internal mutex and is read by the waiter after unpark.
    state: AtomicU8,
    /// Handle used by the release path to wake this specific waiter.
    thread: thread::Thread,
}

/// Mutex-protected state for the budget: currently reserved bytes and the
/// FIFO wait queue. Every mutation of these fields happens under this lock
/// so grants and releases are serialized.
struct BudgetInner {
    reserved: u64,
    waiters: VecDeque<Arc<Waiter>>,
}

// ---------------------------------------------------------------------------
// VectorMemoryBudget - global memory accountant with FIFO wait queue
// ---------------------------------------------------------------------------

/// Tracks total memory reserved by all vector indexes and queries.
/// Enforces a configurable upper limit and queues requests that cannot be
/// served immediately. Admission is strictly FIFO: a large request at the
/// head of the queue blocks smaller requests behind it until memory frees,
/// so a flood of small requests cannot starve a large one.
///
/// The only failure mode is "requested bytes exceed the total limit", which
/// can never be satisfied no matter how long the caller waits. Any request
/// that fits in the configured limit will eventually succeed once queued
/// reservations release enough memory.
pub struct VectorMemoryBudget {
    limit_bytes: AtomicU64,
    /// Lock-free cache of `inner.reserved` for metrics reads. Kept in sync
    /// with `inner.reserved` under the mutex.
    reserved_bytes_cache: AtomicU64,
    inner: Mutex<BudgetInner>,
    high_water_bytes: AtomicU64,
    reservation_count: AtomicU64,
}

impl VectorMemoryBudget {
    /// Creates a budget with explicit limit in bytes.
    pub fn with_limit(limit_bytes: u64) -> Self {
        Self {
            limit_bytes: AtomicU64::new(limit_bytes),
            reserved_bytes_cache: AtomicU64::new(0),
            inner: Mutex::new(BudgetInner {
                reserved: 0,
                waiters: VecDeque::new(),
            }),
            high_water_bytes: AtomicU64::new(0),
            reservation_count: AtomicU64::new(0),
        }
    }

    /// Creates a budget sized to a fraction of available system RAM.
    /// Queries the system via sysinfo, computes `available * fraction`, and
    /// uses that as the limit. Minimum 256 MB to ensure viability on
    /// low-memory systems.
    pub fn auto_sized(fraction: f32) -> Self {
        let mut sys = System::new();
        sys.refresh_memory();
        let available = sys.available_memory();
        let target = (available as f64 * fraction as f64) as u64;
        let limit = target.max(256 * 1024 * 1024);
        Self::with_limit(limit)
    }

    /// Non-blocking reservation. Returns immediately, either with a
    /// MemoryReservation or a `VectorMemoryBudgetExceeded` error. Respects
    /// FIFO: refuses to jump ahead of any already-queued waiter, so calling
    /// this while other threads are waiting never causes starvation.
    pub fn try_reserve(self: &Arc<Self>, bytes: u64) -> Result<MemoryReservation> {
        if bytes == 0 {
            return Ok(MemoryReservation::zero(Arc::clone(self)));
        }
        let limit = self.limit_bytes.load(Ordering::Acquire);
        if bytes > limit {
            return Err(ZyronError::VectorIndexTooLarge {
                estimated: bytes,
                limit,
            });
        }

        let mut inner = self.inner.lock();
        if !inner.waiters.is_empty() {
            let available = limit.saturating_sub(inner.reserved);
            return Err(ZyronError::VectorMemoryBudgetExceeded {
                requested: bytes,
                available,
            });
        }
        let new_total = inner.reserved.saturating_add(bytes);
        if new_total > limit {
            return Err(ZyronError::VectorMemoryBudgetExceeded {
                requested: bytes,
                available: limit.saturating_sub(inner.reserved),
            });
        }
        inner.reserved = new_total;
        self.reserved_bytes_cache
            .store(new_total, Ordering::Release);
        drop(inner);
        self.bump_stats(new_total);
        Ok(MemoryReservation {
            budget: Arc::clone(self),
            bytes,
            released: false,
        })
    }

    /// Blocking reservation. Waits in FIFO order until the budget can serve
    /// the request, then returns a MemoryReservation. The caller's thread is
    /// parked while it waits, so the CPU cost during the wait is zero.
    ///
    /// Returns `VectorIndexTooLarge` immediately if `bytes` exceeds the
    /// configured limit, since no amount of waiting can make the request fit.
    /// The same error is returned from a parked call if the limit is later
    /// lowered below the waiter's needed bytes via `set_limit`.
    pub fn reserve_blocking(self: &Arc<Self>, bytes: u64) -> Result<MemoryReservation> {
        if bytes == 0 {
            return Ok(MemoryReservation::zero(Arc::clone(self)));
        }
        let limit = self.limit_bytes.load(Ordering::Acquire);
        if bytes > limit {
            return Err(ZyronError::VectorIndexTooLarge {
                estimated: bytes,
                limit,
            });
        }

        // Fast path: no one is queued and the budget has room right now.
        let waiter = {
            let mut inner = self.inner.lock();
            let new_total = inner.reserved.saturating_add(bytes);
            if inner.waiters.is_empty() && new_total <= limit {
                inner.reserved = new_total;
                self.reserved_bytes_cache
                    .store(new_total, Ordering::Release);
                drop(inner);
                self.bump_stats(new_total);
                return Ok(MemoryReservation {
                    budget: Arc::clone(self),
                    bytes,
                    released: false,
                });
            }
            let w = Arc::new(Waiter {
                needed: bytes,
                state: AtomicU8::new(WAITER_PENDING),
                thread: thread::current(),
            });
            inner.waiters.push_back(Arc::clone(&w));
            w
        };

        // Park until the release path grants or denies us. park() may return
        // spuriously, so the state check is inside a loop.
        loop {
            thread::park();
            match waiter.state.load(Ordering::Acquire) {
                WAITER_GRANTED => {
                    // The release path already added `bytes` to `reserved`
                    // on our behalf and popped us from the queue. Publishing
                    // the high-water update here keeps stats consistent.
                    let current = self.reserved_bytes_cache.load(Ordering::Acquire);
                    self.bump_stats(current);
                    return Ok(MemoryReservation {
                        budget: Arc::clone(self),
                        bytes,
                        released: false,
                    });
                }
                WAITER_DENIED => {
                    return Err(ZyronError::VectorIndexTooLarge {
                        estimated: bytes,
                        limit: self.limit_bytes.load(Ordering::Acquire),
                    });
                }
                _ => {
                    // Spurious wake, park again.
                }
            }
        }
    }

    /// Attempts to reserve with graceful downgrade. If the initial request
    /// fails, calls `scale_down` to compute a smaller request and retries.
    /// Returns the reservation and the actual number of bytes reserved
    /// (which may be smaller than the initial request).
    ///
    /// Used by index builders to adaptively reduce m/efConstruction/centroids
    /// until the build fits in the budget without blocking.
    pub fn try_reserve_adaptive<F>(
        self: &Arc<Self>,
        initial_bytes: u64,
        scale_down: F,
    ) -> Result<(MemoryReservation, u64)>
    where
        F: Fn(u64) -> Option<u64>,
    {
        let mut request = initial_bytes;
        loop {
            match self.try_reserve(request) {
                Ok(r) => return Ok((r, request)),
                Err(_) => match scale_down(request) {
                    Some(smaller) if smaller < request && smaller > 0 => {
                        request = smaller;
                        continue;
                    }
                    _ => {
                        return Err(ZyronError::VectorIndexTooLarge {
                            estimated: initial_bytes,
                            limit: self.limit_bytes.load(Ordering::Acquire),
                        });
                    }
                },
            }
        }
    }

    /// Current reserved bytes.
    pub fn reserved_bytes(&self) -> u64 {
        self.reserved_bytes_cache.load(Ordering::Acquire)
    }

    /// Limit in bytes.
    pub fn limit_bytes(&self) -> u64 {
        self.limit_bytes.load(Ordering::Acquire)
    }

    /// Bytes still available in the budget.
    pub fn available_bytes(&self) -> u64 {
        self.limit_bytes().saturating_sub(self.reserved_bytes())
    }

    /// Peak reserved bytes since creation (for metrics).
    pub fn high_water_bytes(&self) -> u64 {
        self.high_water_bytes.load(Ordering::Acquire)
    }

    /// Total reservations made since creation (for metrics).
    pub fn reservation_count(&self) -> u64 {
        self.reservation_count.load(Ordering::Acquire)
    }

    /// Number of callers currently parked in the wait queue.
    pub fn waiter_count(&self) -> usize {
        self.inner.lock().waiters.len()
    }

    /// Updates the limit at runtime. Existing reservations are unaffected.
    /// If the limit is raised, any queued waiters that now fit are granted.
    /// If the limit is lowered below a queued waiter's needed bytes, that
    /// waiter is woken with a `VectorIndexTooLarge` error.
    pub fn set_limit(&self, limit_bytes: u64) {
        self.limit_bytes.store(limit_bytes, Ordering::Release);
        let mut inner = self.inner.lock();
        self.drain_waiters(&mut inner);
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Updates the high-water mark and increments the reservation counter.
    /// Called from the success path of both try_reserve and reserve_blocking.
    fn bump_stats(&self, current_reserved: u64) {
        let mut hw = self.high_water_bytes.load(Ordering::Relaxed);
        while current_reserved > hw {
            match self.high_water_bytes.compare_exchange_weak(
                hw,
                current_reserved,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => hw = actual,
            }
        }
        self.reservation_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Releases `bytes` back to the budget and drains as many waiters as can
    /// now be served. Only called from `MemoryReservation::release_internal`.
    fn release_bytes(&self, bytes: u64) {
        let mut inner = self.inner.lock();
        inner.reserved = inner.reserved.saturating_sub(bytes);
        self.drain_waiters(&mut inner);
    }

    /// Walks the head of the wait queue and either grants or denies each
    /// waiter in FIFO order. Grants are served while the budget has room.
    /// Waiters whose needed bytes exceed the current limit are denied and
    /// skipped without blocking the queue. Stops at the first waiter that
    /// would fit in the limit but does not fit in the currently-available
    /// space, preserving strict FIFO fairness.
    ///
    /// Must be called with `self.inner` locked.
    fn drain_waiters(&self, inner: &mut BudgetInner) {
        let limit = self.limit_bytes.load(Ordering::Acquire);
        while let Some(head) = inner.waiters.front() {
            if head.needed > limit {
                head.state.store(WAITER_DENIED, Ordering::Release);
                let t = head.thread.clone();
                inner.waiters.pop_front();
                t.unpark();
                continue;
            }
            let new_total = inner.reserved.saturating_add(head.needed);
            if new_total > limit {
                break;
            }
            inner.reserved = new_total;
            head.state.store(WAITER_GRANTED, Ordering::Release);
            let t = head.thread.clone();
            inner.waiters.pop_front();
            t.unpark();
        }
        self.reserved_bytes_cache
            .store(inner.reserved, Ordering::Release);
    }
}

impl Default for VectorMemoryBudget {
    fn default() -> Self {
        Self::auto_sized(DEFAULT_BUDGET_FRACTION)
    }
}

// ---------------------------------------------------------------------------
// MemoryReservation - RAII guard that releases on Drop
// ---------------------------------------------------------------------------

/// RAII guard holding a memory reservation against a VectorMemoryBudget.
/// The reserved bytes are released back to the budget when this struct is
/// dropped, whether via normal scope exit, error propagation, or panic.
/// Release also wakes any queued waiters that can now be served.
pub struct MemoryReservation {
    budget: Arc<VectorMemoryBudget>,
    bytes: u64,
    released: bool,
}

impl MemoryReservation {
    /// Creates a reservation that holds zero bytes. Used for requests of
    /// size zero so callers can rely on the RAII contract uniformly.
    fn zero(budget: Arc<VectorMemoryBudget>) -> Self {
        Self {
            budget,
            bytes: 0,
            released: true,
        }
    }

    /// Returns the number of bytes held by this reservation.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Releases the reservation early. Normally handled by Drop.
    pub fn release(mut self) {
        self.release_internal();
    }

    fn release_internal(&mut self) {
        if !self.released {
            self.budget.release_bytes(self.bytes);
            self.released = true;
        }
    }
}

impl Drop for MemoryReservation {
    fn drop(&mut self) {
        self.release_internal();
    }
}

// ---------------------------------------------------------------------------
// Memory estimators - compute index footprints before allocation
// ---------------------------------------------------------------------------

/// Estimates the steady-state memory footprint of an HNSW index.
/// Formula components match the NodeStore flat-arena layout:
///   - vector arena: n * dims * 4 bytes
///   - quantized arena: n * dims * 1 byte
///   - node metadata arrays: ids (n * 8), levels (n * 1), deleted (n * 1)
///   - conn_data: layer 0 always holds a (2m + 1)-u32 slot. Each higher
///     layer a node reaches adds (m + 1) u32s. The per-node upper-layer
///     count follows a geometric distribution with P(level >= l) = m^-l,
///     whose sum over l >= 1 is 1 / (m - 1). Per-node expected slot
///     count is therefore (2m + 1) + (m + 1) / (m - 1). The estimator
///     uses a slightly larger fixed overhead of 3 * (m + 1) to stay on
///     the safe side of graphs that happen to produce above-expected
///     levels, making the return value an upper bound on typical builds.
///   - node_offsets table: n * 8 bytes
///   - idMap: per-entry overhead from the scc::HashMap implementation
pub fn estimate_hnsw_memory(n: usize, dims: usize, m: u16) -> u64 {
    let n = n as u64;
    let dims = dims as u64;
    let m = m as u64;

    let arena = n * dims * 4;
    let quantized = n * dims;
    // NodeStore stores ids, levels, and deleted as three separate Vecs.
    // Combined per-node footprint is 8 + 1 + 1 = 10 bytes, rounded to
    // 16 to leave slack for Vec growth and allocator alignment.
    let node_meta = n * 16;
    // Upper bound on per-node slot count: layer 0 plus a conservative
    // fixed three upper-layer slots. See doc comment above for why this
    // overshoots the geometric expectation.
    let per_node_slots = (2 * m + 1) + 3 * (m + 1);
    let conn_data = n * per_node_slots * 4;
    let node_offsets = n * 8;
    let id_map = n * 32;

    arena + quantized + node_meta + conn_data + node_offsets + id_map
}

/// Estimates peak memory during HNSW parallel build.
/// Peak includes the steady-state footprint plus transient build state:
///   - BuildGraph conn_data of AtomicU32 cells coexists with the finalized
///     NodeStore during the flatten step, bounded above by a second copy
///     of conn_data.
///   - Per-node RwLock instances in BuildGraph: n * 8 bytes.
///   - Per-thread VisitedSet is sized by efConstruction, not by n, and is
///     a fixed tens-of-KB per thread. Included in the workspace allowance.
///   - Per-thread SearchState heaps and scratch buffers.
/// The 30% margin accounts for allocator fragmentation and Vec growth
/// slack beyond these line items.
pub fn estimate_hnsw_build_peak(n: usize, dims: usize, m: u16, cores: usize) -> u64 {
    let steady = estimate_hnsw_memory(n, dims, m);
    let n_u64 = n as u64;

    // BuildGraph conn_data coexists with the finalized NodeStore during
    // flatten. Upper bound: a second copy of conn_data.
    let build_transients = n_u64 * ((2 * m as u64 + 1) + 3 * (m as u64 + 1)) * 4;

    // Per-node RwLock instances in BuildGraph.
    let node_locks = n_u64 * 8;

    // Search-state heaps, visited sets, and scratch buffers across all threads.
    let workspace = (cores as u64 * 4 * 1024 * 1024).max(64 * 1024 * 1024);

    let peak = steady + build_transients + node_locks + workspace;
    peak + peak / 3
}

/// Estimates the steady-state memory footprint of an IVF-PQ index.
/// Components:
///   - vector arena: n * dims * 4 bytes (kept for exact rerank)
///   - centroids: numCentroids * dims * 4 bytes
///   - residual centroids: ~512 * dims * 4 bytes
///   - codebooks: numSub * 256 * (dims/numSub) * 4 bytes
///   - inverted lists: n * (8 + 4 + 2 + numSub + 24) bytes (per entry)
///   - idToArenaIdx (scc::HashMap): n * 32 bytes
pub fn estimate_ivfpq_memory(n: usize, dims: usize, num_centroids: u32, num_sub: u16) -> u64 {
    let n = n as u64;
    let dims = dims as u64;
    let nc = num_centroids as u64;
    let ns = num_sub as u64;

    let arena = n * dims * 4;
    let centroids = nc * dims * 4;
    let residual_centroids = 512 * dims * 4;
    let codebooks = ns * 256 * dims * 4; // approx
    let inverted_entry_size = 8 + 4 + 2 + ns + 24;
    let inverted_lists = n * inverted_entry_size;
    let id_map = n * 32;

    arena + centroids + residual_centroids + codebooks + inverted_lists + id_map
}

/// Estimates peak memory during IVF-PQ build.
/// Peak includes steady-state plus the transient residuals1 and residuals2
/// arrays that coexist during stage 2 quantization. Each residuals array is
/// n * dims * 4 bytes (the same size as the arena).
pub fn estimate_ivfpq_build_peak(
    n: usize,
    dims: usize,
    num_centroids: u32,
    num_sub: u16,
    cores: usize,
) -> u64 {
    let steady = estimate_ivfpq_memory(n, dims, num_centroids, num_sub);
    let n_u64 = n as u64;
    let dims_u64 = dims as u64;

    // Transient residual arrays (stage 1 + stage 2). Kept in memory
    // simultaneously in the current implementation.
    let residuals = 2 * n_u64 * dims_u64 * 4;

    // K-means parallel partial sums: cores threads each holding k * dims * 4 bytes
    let kmeans_partials = cores as u64 * num_centroids as u64 * dims_u64 * 4;

    // Assignments array: n * 8 bytes
    let assignments = n_u64 * 8;

    let peak = steady + residuals + kmeans_partials + assignments;
    // 30% safety margin.
    peak + peak / 3
}

// ---------------------------------------------------------------------------
// Fallible allocation helpers
// ---------------------------------------------------------------------------

/// Allocates a Vec with the given capacity. Returns a MemoryAllocationFailed
/// error instead of aborting the process if allocation fails.
/// Use this instead of `Vec::with_capacity` for any user-input-driven size.
pub fn try_alloc_vec<T>(capacity: usize) -> Result<Vec<T>> {
    let mut v: Vec<T> = Vec::new();
    if capacity == 0 {
        return Ok(v);
    }
    v.try_reserve_exact(capacity)
        .map_err(|_| ZyronError::MemoryAllocationFailed {
            bytes: (capacity as u64).saturating_mul(std::mem::size_of::<T>() as u64),
        })?;
    Ok(v)
}

/// Allocates and fills a Vec with `size` copies of `value`. Fallible.
pub fn try_alloc_filled<T: Clone>(size: usize, value: T) -> Result<Vec<T>> {
    let mut v = try_alloc_vec(size)?;
    v.resize(size, value);
    Ok(v)
}

/// Allocates a Vec filled with zeros (default values). Fallible.
pub fn try_alloc_default<T: Default + Clone>(size: usize) -> Result<Vec<T>> {
    try_alloc_filled(size, T::default())
}

/// Validates that a declared size (typically from a file header) is within
/// reasonable bounds before allocating based on it. Rejects sizes exceeding
/// the absolute sanity cap.
pub fn validate_file_size(declared_bytes: u64) -> Result<()> {
    if declared_bytes > MAX_INDEX_FILE_BYTES {
        return Err(ZyronError::VectorIndexFileCorrupt {
            declared: declared_bytes,
        });
    }
    Ok(())
}

/// Queries the system for currently available memory in bytes.
/// Used by tests and pre-allocation checks.
pub fn available_system_memory() -> u64 {
    let mut sys = System::new();
    sys.refresh_memory();
    sys.available_memory()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;
    use std::thread;
    use std::time::{Duration, Instant};

    #[test]
    fn tryReserveRejectsOverLimit() {
        let budget = Arc::new(VectorMemoryBudget::with_limit(1000));
        let _r1 = budget
            .try_reserve(500)
            .expect("first reserve should succeed");
        let result = budget.try_reserve(600);
        assert!(matches!(
            result,
            Err(ZyronError::VectorMemoryBudgetExceeded { .. })
        ));
    }

    #[test]
    fn tryReserveRejectsOversizedSingleRequest() {
        let budget = Arc::new(VectorMemoryBudget::with_limit(1000));
        let result = budget.try_reserve(1500);
        assert!(matches!(
            result,
            Err(ZyronError::VectorIndexTooLarge { .. })
        ));
    }

    #[test]
    fn reservationReleasesOnDrop() {
        let budget = Arc::new(VectorMemoryBudget::with_limit(1000));
        {
            let _r = budget.try_reserve(500).expect("reserve");
            assert_eq!(budget.reserved_bytes(), 500);
        }
        assert_eq!(budget.reserved_bytes(), 0);
    }

    #[test]
    fn reservationReleasesOnExplicitRelease() {
        let budget = Arc::new(VectorMemoryBudget::with_limit(1000));
        let r = budget.try_reserve(500).expect("reserve");
        r.release();
        assert_eq!(budget.reserved_bytes(), 0);
    }

    #[test]
    fn highWaterTracksPeak() {
        let budget = Arc::new(VectorMemoryBudget::with_limit(10000));
        let r1 = budget.try_reserve(3000).expect("r1");
        let r2 = budget.try_reserve(4000).expect("r2");
        assert_eq!(budget.high_water_bytes(), 7000);
        drop(r2);
        assert_eq!(budget.high_water_bytes(), 7000); // peak preserved
        drop(r1);
        assert_eq!(budget.reserved_bytes(), 0);
    }

    #[test]
    fn adaptiveDowngradeSucceeds() {
        let budget = Arc::new(VectorMemoryBudget::with_limit(1000));
        let (_reservation, actual) = budget
            .try_reserve_adaptive(5000, |current| {
                if current > 500 {
                    Some(current / 2)
                } else {
                    None
                }
            })
            .expect("adaptive should succeed");
        assert!(actual <= 1000);
        assert!(actual > 0);
    }

    #[test]
    fn adaptiveGivesUpWhenScaleDownReturnsNone() {
        let budget = Arc::new(VectorMemoryBudget::with_limit(100));
        let result = budget.try_reserve_adaptive(5000, |_| None);
        assert!(matches!(
            result,
            Err(ZyronError::VectorIndexTooLarge { .. })
        ));
    }

    #[test]
    fn estimateHnswScalesWithN() {
        let small = estimate_hnsw_memory(1000, 128, 32);
        let large = estimate_hnsw_memory(1_000_000, 128, 32);
        assert!(large > small * 500);
    }

    #[test]
    fn estimateHnswBuildPeakExceedsSteady() {
        let steady = estimate_hnsw_memory(100_000, 128, 32);
        let peak = estimate_hnsw_build_peak(100_000, 128, 32, 8);
        assert!(peak > steady);
    }

    #[test]
    fn estimateIvfpqScalesWithN() {
        let small = estimate_ivfpq_memory(1000, 128, 32, 16);
        let large = estimate_ivfpq_memory(1_000_000, 128, 256, 16);
        assert!(large > small);
    }

    #[test]
    fn tryAllocVecSucceeds() {
        let v: Vec<u32> = try_alloc_vec(100).expect("small alloc");
        assert_eq!(v.capacity(), 100);
        assert_eq!(v.len(), 0);
    }

    #[test]
    fn tryAllocFilledSucceeds() {
        let v: Vec<u32> = try_alloc_filled(10, 42).expect("fill");
        assert_eq!(v.len(), 10);
        assert!(v.iter().all(|&x| x == 42));
    }

    #[test]
    fn tryAllocRejectsImpossibleSize() {
        // Request so large that try_reserve_exact must fail.
        let result: Result<Vec<u64>> = try_alloc_vec(usize::MAX / 2);
        assert!(result.is_err());
    }

    #[test]
    fn validateFileSizeRejectsPathological() {
        assert!(validate_file_size(1024).is_ok());
        assert!(validate_file_size(MAX_INDEX_FILE_BYTES + 1).is_err());
    }

    #[test]
    fn autoSizedProducesPositiveBudget() {
        let budget = VectorMemoryBudget::auto_sized(0.5);
        assert!(budget.limit_bytes() >= 256 * 1024 * 1024);
    }

    #[test]
    fn concurrentTryReservesAreAtomic() {
        let budget = Arc::new(VectorMemoryBudget::with_limit(1_000_000));
        let mut handles = Vec::new();
        for _ in 0..16 {
            let b = Arc::clone(&budget);
            handles.push(thread::spawn(move || {
                let mut reservations = Vec::new();
                for _ in 0..10 {
                    if let Ok(r) = b.try_reserve(1000) {
                        reservations.push(r);
                    }
                }
                reservations
            }));
        }
        let all: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        let total_reserved: usize = all.iter().map(|v| v.len()).sum();
        assert_eq!(budget.reserved_bytes(), (total_reserved * 1000) as u64);
        drop(all);
        assert_eq!(budget.reserved_bytes(), 0);
    }

    #[test]
    fn blockingReserveServesInFifoOrder() {
        // Budget limit is 1000. Thread A reserves 700. B and C each need 600,
        // which is larger than the remaining 300 and also more than half the
        // total budget, so only one of them can run at a time. B is queued
        // first and C second, so when A releases its 700, the drain loop
        // serves B but stops because C no longer fits (600 + 600 > 1000).
        // Only after B releases its 600 can C be granted. This makes the
        // grant order observable through a shared ledger: B must push 'B'
        // before C pushes 'C'.
        let budget = Arc::new(VectorMemoryBudget::with_limit(1000));
        let r_a = budget.try_reserve(700).expect("A reserves 700");

        let order = Arc::new(Mutex::new(Vec::<char>::new()));

        let b_budget = Arc::clone(&budget);
        let b_order = Arc::clone(&order);
        let handle_b = thread::spawn(move || {
            let r = b_budget.reserve_blocking(600).expect("B reserved");
            b_order.lock().push('B');
            // Hold the reservation briefly so C cannot be granted until B
            // finishes. Dropping the guard releases the 600 bytes.
            thread::sleep(Duration::from_millis(50));
            drop(r);
        });
        wait_until(|| budget.waiter_count() >= 1, Duration::from_secs(2));

        let c_budget = Arc::clone(&budget);
        let c_order = Arc::clone(&order);
        let handle_c = thread::spawn(move || {
            let _r = c_budget.reserve_blocking(600).expect("C reserved");
            c_order.lock().push('C');
        });
        wait_until(|| budget.waiter_count() >= 2, Duration::from_secs(2));

        drop(r_a);
        handle_b.join().expect("B join");
        handle_c.join().expect("C join");

        let final_order = order.lock().clone();
        assert_eq!(final_order, vec!['B', 'C']);
    }

    #[test]
    fn blockingReserveWakesMultipleWaitersOnLargeRelease() {
        // A holds 900. B (300), C (300), D (300) are queued. Releasing A
        // frees all 900 bytes, which can serve B, C, and D at once. All
        // three should complete without further releases.
        let budget = Arc::new(VectorMemoryBudget::with_limit(900));
        let r_a = budget.try_reserve(900).expect("A reserves 900");

        let done = Arc::new(AtomicU64::new(0));
        let mut handles = Vec::new();
        for _ in 0..3 {
            let b = Arc::clone(&budget);
            let d = Arc::clone(&done);
            handles.push(thread::spawn(move || {
                let _r = b.reserve_blocking(300).expect("reserved");
                d.fetch_add(1, Ordering::AcqRel);
                thread::sleep(Duration::from_millis(100));
            }));
        }
        wait_until(|| budget.waiter_count() == 3, Duration::from_secs(2));

        drop(r_a);
        for h in handles {
            h.join().expect("join");
        }
        assert_eq!(done.load(Ordering::Acquire), 3);
    }

    #[test]
    fn blockingReserveRejectsOversizedImmediately() {
        let budget = Arc::new(VectorMemoryBudget::with_limit(1000));
        let started = Instant::now();
        let result = budget.reserve_blocking(2000);
        assert!(matches!(
            result,
            Err(ZyronError::VectorIndexTooLarge { .. })
        ));
        // Should return without blocking.
        assert!(started.elapsed() < Duration::from_millis(200));
    }

    #[test]
    fn loweringLimitDeniesWaiterWhoNoLongerFits() {
        // B reserves 300 and holds it. A queues for 800 (fits at original
        // limit 1000). Lowering the limit to 400 makes A's request
        // impossible. A should be woken with VectorIndexTooLarge.
        let budget = Arc::new(VectorMemoryBudget::with_limit(1000));
        let r_b = budget.try_reserve(300).expect("B holds 300");

        let a_budget = Arc::clone(&budget);
        let a_result: Arc<Mutex<Option<Result<()>>>> = Arc::new(Mutex::new(None));
        let a_result_clone = Arc::clone(&a_result);
        let handle_a = thread::spawn(move || {
            let r = a_budget.reserve_blocking(800);
            *a_result_clone.lock() = Some(r.map(|_| ()));
        });

        wait_until(|| budget.waiter_count() == 1, Duration::from_secs(2));
        budget.set_limit(400);
        handle_a.join().expect("A join");
        drop(r_b);

        let guard = a_result.lock();
        assert!(matches!(
            guard.as_ref().expect("A finished"),
            Err(ZyronError::VectorIndexTooLarge { .. })
        ));
    }

    #[test]
    fn raisingLimitDrainsQueuedWaiters() {
        // Budget starts at 400. A reserves 400 and holds it. B queues
        // for 800 (which is larger than the current limit but not the
        // raised limit). The initial bytes > limit check rejects B
        // immediately, so to exercise the drain-on-raise path we first
        // raise the limit, then try_reserve, then lower, which is the
        // set_limit ordering real callers hit.
        //
        // Concretely: budget = 1000. A holds 700. B (500) queues because
        // only 300 is free. We raise limit to 1500: now 800 is free, B
        // fits, drain_waiters unparks B. B completes without A releasing.
        let budget = Arc::new(VectorMemoryBudget::with_limit(1000));
        let r_a = budget.try_reserve(700).expect("A reserves 700");

        let b_budget = Arc::clone(&budget);
        let b_done = Arc::new(AtomicBool::new(false));
        let b_done_clone = Arc::clone(&b_done);
        let handle_b = thread::spawn(move || {
            let _r = b_budget.reserve_blocking(500).expect("B reserved");
            b_done_clone.store(true, Ordering::Release);
        });

        wait_until(|| budget.waiter_count() == 1, Duration::from_secs(2));
        assert!(!b_done.load(Ordering::Acquire));

        budget.set_limit(1500);
        handle_b.join().expect("B join");
        assert!(b_done.load(Ordering::Acquire));
        drop(r_a);
    }

    #[test]
    fn tryReserveRefusesToJumpQueue() {
        // A queues. A later try_reserve from another thread must not
        // jump ahead of the waiter even if its bytes would fit.
        let budget = Arc::new(VectorMemoryBudget::with_limit(1000));
        let r_big = budget.try_reserve(900).expect("holds 900");

        let w_budget = Arc::clone(&budget);
        let handle = thread::spawn(move || {
            let _r = w_budget.reserve_blocking(300).expect("waiter reserved");
        });
        wait_until(|| budget.waiter_count() == 1, Duration::from_secs(2));

        // Only 100 free, but even a 50-byte try should refuse (FIFO).
        let jumper = budget.try_reserve(50);
        assert!(matches!(
            jumper,
            Err(ZyronError::VectorMemoryBudgetExceeded { .. })
        ));

        drop(r_big);
        handle.join().expect("waiter join");
    }

    #[test]
    fn zeroByteReservationSucceeds() {
        let budget = Arc::new(VectorMemoryBudget::with_limit(1000));
        let r = budget.try_reserve(0).expect("zero try_reserve");
        assert_eq!(r.bytes(), 0);
        let r2 = budget.reserve_blocking(0).expect("zero blocking");
        assert_eq!(r2.bytes(), 0);
        drop(r);
        drop(r2);
        assert_eq!(budget.reserved_bytes(), 0);
    }

    // Polls `cond` until it returns true or `timeout` elapses. Panics on
    // timeout so test failures surface promptly instead of hanging.
    fn wait_until<F: Fn() -> bool>(cond: F, timeout: Duration) {
        let start = Instant::now();
        while !cond() {
            if start.elapsed() > timeout {
                panic!("wait_until timed out after {:?}", timeout);
            }
            thread::sleep(Duration::from_millis(2));
        }
    }
}

//! Scheduling, rate limiting, and quotas
//!
//! Re-exports the cron and pure-math rate-limit primitives from earlier
//! phases, then adds stateful key-based registries:
//!
//!   RateLimiterRegistry: lock-free sliding-window rate limiter keyed by
//!   tenant string. Uses scc::HashMap for per-key state and atomic counters
//!   for the bucketed window so checks are wait-free in the common path
//!
//!   QuotaRegistry: lock-free atomic-counter quota store keyed by string
//!   quota_increment uses a CAS loop so quota_check and quota_increment can
//!   race without losing updates
//!
//!   TokenBucketBurstRegistry: token bucket with burst credit (item L). Any
//!   unused refill capacity accrues into a bounded credit pool, allowing
//!   short bursts above steady-state without rejecting load

pub use crate::cron::{
    CronExpr, cron_between as cron_list, cron_human_readable, cron_matches, cron_next, cron_parse,
    cron_prev,
};
pub use crate::rate_limit::{
    LeakyBucket, TokenBucket, fixed_window_count, leaky_bucket_add, leaky_bucket_create,
    sliding_window_check, sliding_window_count, token_bucket_available, token_bucket_consume,
    token_bucket_create,
};

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use zyron_common::{Result, ZyronError};

// ---------------------------------------------------------------------------
// Sliding-window rate limiter, key based
// ---------------------------------------------------------------------------

/// Number of sub-buckets per window. More sub-buckets means smoother
/// enforcement at the cost of a slightly larger per-key footprint. 10 gives
/// 10-percent granularity which is the standard pick
const SUB_BUCKETS: usize = 10;

/// One bucketed window for a single key, accessed concurrently by all
/// requests for that key. Each sub-bucket is an AtomicU32 counter, the
/// bucket epoch is an AtomicU64 so we can detect when the window has rolled
/// over and reset stale buckets without taking a lock
struct AtomicSlidingWindow {
    /// Counters for each sub-bucket
    counters: [AtomicU32; SUB_BUCKETS],
    /// Epoch (window-bucket index in nanoseconds / sub_bucket_size_ns) the
    /// active bucket was last touched. Lets us reset stale buckets lazily
    epochs: [AtomicU64; SUB_BUCKETS],
}

impl AtomicSlidingWindow {
    fn new() -> Self {
        Self {
            counters: std::array::from_fn(|_| AtomicU32::new(0)),
            epochs: std::array::from_fn(|_| AtomicU64::new(0)),
        }
    }

    /// Returns the total count over the active window, advancing stale
    /// buckets to zero. now_ns is monotonic nanoseconds, window_ns is the
    /// width of the window in nanoseconds
    fn current_count(&self, now_ns: u64, window_ns: u64) -> u32 {
        let bucket_ns = window_ns / SUB_BUCKETS as u64;
        let active_epoch_floor = now_ns.saturating_sub(window_ns) / bucket_ns;
        let mut total = 0u32;
        for i in 0..SUB_BUCKETS {
            let bucket_epoch = self.epochs[i].load(Ordering::Acquire);
            if bucket_epoch >= active_epoch_floor {
                total = total.saturating_add(self.counters[i].load(Ordering::Acquire));
            }
        }
        total
    }

    /// Atomically advances the bucket and tries to consume one slot. Returns
    /// (allowed, observed_count_including_this_attempt)
    #[inline]
    fn check_and_consume(&self, now_ns: u64, window_ns: u64, max: u32) -> (bool, u32) {
        let bucket_ns = window_ns / SUB_BUCKETS as u64;
        if bucket_ns == 0 {
            return (true, 1);
        }
        let now_epoch = now_ns / bucket_ns;
        let slot = (now_epoch as usize) % SUB_BUCKETS;
        // CAS the epoch, if it has rolled over we reset the counter to 0
        let prev = self.epochs[slot].swap(now_epoch, Ordering::AcqRel);
        if prev != now_epoch {
            self.counters[slot].store(0, Ordering::Release);
        }
        // Bump the active bucket atomically and read back the new value so
        // we don't have to re-sum buckets in the common allowed path. The
        // returned `prev_in_slot + 1` is the count for this bucket only;
        // the total across buckets is at least that and at most max+1
        let after = self.counters[slot].fetch_add(1, Ordering::AcqRel) + 1;
        // Fast accept: if this bucket alone is below max and no neighbouring
        // bucket can be active (window collapses to one bucket), short-circuit.
        // Else fall through to the full sum
        if after <= max {
            // Optimistic accept; verify with a cheaper bucket-bounded sum
            let total = self.current_count(now_ns, window_ns);
            if total <= max {
                return (true, total);
            }
            // Over the limit even with the wider window, roll back our slot
            self.counters[slot].fetch_sub(1, Ordering::AcqRel);
            (false, total - 1)
        } else {
            // Even our slot alone exceeds max, must reject
            self.counters[slot].fetch_sub(1, Ordering::AcqRel);
            (false, after - 1)
        }
    }
}

/// Lock-free key-based sliding-window rate limiter
pub struct RateLimiterRegistry {
    inner: scc::HashMap<String, Arc<AtomicSlidingWindow>>,
    baseline: Instant,
}

impl Default for RateLimiterRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl RateLimiterRegistry {
    pub fn new() -> Self {
        Self {
            inner: scc::HashMap::new(),
            baseline: Instant::now(),
        }
    }

    #[inline(always)]
    fn now_ns(&self) -> u64 {
        // Avoid Duration::as_nanos() which returns u128 and forces a wide
        // multiply on the hot path. Compose from secs + subsec_nanos in u64
        let d = self.baseline.elapsed();
        d.as_secs()
            .wrapping_mul(1_000_000_000)
            .wrapping_add(d.subsec_nanos() as u64)
    }

    fn window_for(&self, key: &str) -> Arc<AtomicSlidingWindow> {
        if let Some(found) = self.inner.read_sync(key, |_, w| Arc::clone(w)) {
            return found;
        }
        // Insert if missing, otherwise pick up the racing insert
        let new_w = Arc::new(AtomicSlidingWindow::new());
        match self.inner.entry_sync(key.to_string()) {
            scc::hash_map::Entry::Occupied(o) => Arc::clone(o.get()),
            scc::hash_map::Entry::Vacant(v) => {
                v.insert_entry(Arc::clone(&new_w));
                new_w
            }
        }
    }

    /// Returns true if a request for key is allowed under max_requests per
    /// window. Bumps the counter when allowed, decrements on rejection
    pub fn check(&self, key: &str, max_requests: u32, window: Duration) -> bool {
        let now_ns = self.now_ns();
        let win_ns = window
            .as_secs()
            .wrapping_mul(1_000_000_000)
            .wrapping_add(window.subsec_nanos() as u64);
        // Run check_and_consume inside the read_sync closure to avoid
        // Arc::clone on the hot path. Falls through to insert+retry only
        // when the key is not yet registered (cold path)
        if let Some(allowed) = self.inner.read_sync(key, |_, w| {
            w.check_and_consume(now_ns, win_ns, max_requests).0
        }) {
            return allowed;
        }
        // Cold path: insert and retry
        let w = self.window_for(key);
        w.check_and_consume(now_ns, win_ns, max_requests).0
    }

    /// Returns a structured decision including retry_after_ms (item K)
    pub fn check_with_decision(
        &self,
        key: &str,
        max_requests: u32,
        window: Duration,
    ) -> RateLimitDecision {
        let w = self.window_for(key);
        let now_ns = self.now_ns();
        let win_ns = window.as_nanos() as u64;
        let (allowed, observed) = w.check_and_consume(now_ns, win_ns, max_requests);
        let remaining = max_requests.saturating_sub(observed);
        let retry_after_ms = if allowed {
            0
        } else {
            // Estimate when one sub-bucket will roll out of the window
            let bucket_ns = win_ns / SUB_BUCKETS as u64;
            (bucket_ns / 1_000_000).max(1) as i64
        };
        RateLimitDecision {
            allowed,
            limit: max_requests,
            remaining,
            retry_after_ms,
        }
    }

    /// Returns the current observed count for key over the given window
    pub fn current_count(&self, key: &str, window: Duration) -> u32 {
        let w = self.window_for(key);
        w.current_count(self.now_ns(), window.as_nanos() as u64)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RateLimitDecision {
    pub allowed: bool,
    pub limit: u32,
    pub remaining: u32,
    pub retry_after_ms: i64,
}

// ---------------------------------------------------------------------------
// Quota registry
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuotaResult {
    pub allowed: bool,
    pub used: u64,
    pub limit: u64,
    pub remaining: u64,
}

pub struct QuotaRegistry {
    inner: scc::HashMap<String, Arc<AtomicU64>>,
}

impl Default for QuotaRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl QuotaRegistry {
    pub fn new() -> Self {
        Self {
            inner: scc::HashMap::new(),
        }
    }

    fn counter(&self, key: &str) -> Arc<AtomicU64> {
        if let Some(found) = self.inner.read_sync(key, |_, c| Arc::clone(c)) {
            return found;
        }
        let new_c = Arc::new(AtomicU64::new(0));
        match self.inner.entry_sync(key.to_string()) {
            scc::hash_map::Entry::Occupied(o) => Arc::clone(o.get()),
            scc::hash_map::Entry::Vacant(v) => {
                v.insert_entry(Arc::clone(&new_c));
                new_c
            }
        }
    }

    /// Read-only quota check
    pub fn check(&self, key: &str, limit: u64) -> QuotaResult {
        let used = self
            .inner
            .read_sync(key, |_, c| c.load(Ordering::Acquire))
            .unwrap_or(0);
        QuotaResult {
            allowed: used < limit,
            used,
            limit,
            remaining: limit.saturating_sub(used),
        }
    }

    /// Atomically consumes amount against the quota for key. Returns the new
    /// total or an error when the increment would exceed limit. Uses a CAS
    /// loop so concurrent callers cannot exceed the limit
    pub fn increment(&self, key: &str, amount: u64, limit: u64) -> Result<u64> {
        let counter = self.counter(key);
        loop {
            let cur = counter.load(Ordering::Acquire);
            let new = cur
                .checked_add(amount)
                .ok_or_else(|| ZyronError::ExecutionError("quota counter overflow".to_string()))?;
            if new > limit {
                return Err(ZyronError::ExecutionError(format!(
                    "quota exceeded: {}/{}",
                    new, limit
                )));
            }
            match counter.compare_exchange_weak(cur, new, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => return Ok(new),
                Err(_) => continue,
            }
        }
    }

    /// Releases amount from the quota for key. Saturates at zero. No-op if
    /// the key is unknown
    pub fn release(&self, key: &str, amount: u64) {
        self.inner.read_sync(key, |_, c| {
            loop {
                let cur = c.load(Ordering::Acquire);
                let new = cur.saturating_sub(amount);
                if c.compare_exchange_weak(cur, new, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    break;
                }
            }
        });
    }

    /// Snapshot of all quota keys and their current usage. Used by gossip
    /// (item O) and admin views
    pub fn snapshot(&self) -> Vec<(String, u64)> {
        let mut out = Vec::new();
        self.inner.iter_sync(|k, v| {
            out.push((k.clone(), v.load(Ordering::Acquire)));
            true
        });
        out
    }

    /// Applies a remote snapshot delta. Used by gossip to converge across
    /// replicas (item O). For each (key, value) the local counter is set to
    /// the maximum of the local and remote value, which is monotone and
    /// commutative so concurrent gossip rounds always converge
    pub fn merge_remote(&self, remote: &[(String, u64)]) {
        for (k, remote_v) in remote {
            let counter = self.counter(k);
            loop {
                let cur = counter.load(Ordering::Acquire);
                if *remote_v <= cur {
                    break;
                }
                if counter
                    .compare_exchange_weak(cur, *remote_v, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    break;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Token bucket with burst credit (item L)
// ---------------------------------------------------------------------------

/// Token bucket whose unused refill capacity accrues into a credit pool
/// capped at credit_cap tokens, allowing short bursts beyond capacity
pub struct BurstTokenBucket {
    capacity: f64,
    refill_rate: f64,
    credit_cap: f64,
    state: parking_lot::Mutex<BurstState>,
}

struct BurstState {
    current: f64,
    credit: f64,
    last_refill: i64,
}

impl BurstTokenBucket {
    pub fn new(capacity: f64, refill_rate: f64, credit_cap_multiplier: f64) -> Self {
        Self {
            capacity,
            refill_rate,
            credit_cap: capacity * credit_cap_multiplier.max(0.0),
            state: parking_lot::Mutex::new(BurstState {
                current: capacity,
                credit: 0.0,
                last_refill: 0,
            }),
        }
    }

    /// Attempts to consume the requested tokens. Returns true on success
    pub fn consume(&self, tokens: f64, now_micros: i64) -> bool {
        let mut s = self.state.lock();
        let elapsed_secs = ((now_micros - s.last_refill).max(0) as f64) / 1_000_000.0;
        let refill = elapsed_secs * self.refill_rate;
        let raw_after = s.current + refill;
        if raw_after > self.capacity {
            // Excess refill goes into the credit pool, capped
            let excess = raw_after - self.capacity;
            s.current = self.capacity;
            s.credit = (s.credit + excess).min(self.credit_cap);
        } else {
            s.current = raw_after;
        }
        s.last_refill = now_micros;
        if s.current >= tokens {
            s.current -= tokens;
            return true;
        }
        // Tap the credit pool to make up the shortfall
        let need = tokens - s.current;
        if s.credit >= need {
            s.credit -= need;
            s.current = 0.0;
            true
        } else {
            false
        }
    }

    pub fn snapshot(&self) -> (f64, f64) {
        let s = self.state.lock();
        (s.current, s.credit)
    }
}

// ---------------------------------------------------------------------------
// Free function adapters used by the SQL executor
// ---------------------------------------------------------------------------

/// rate_limit_check using a registry, suitable for SQL dispatch
pub fn rate_limit_check(
    reg: &RateLimiterRegistry,
    key: &str,
    max_requests: u32,
    window_micros: i64,
) -> bool {
    if window_micros <= 0 {
        return true;
    }
    reg.check(
        key,
        max_requests,
        Duration::from_micros(window_micros as u64),
    )
}

pub fn quota_check(reg: &QuotaRegistry, key: &str, limit: u64) -> QuotaResult {
    reg.check(key, limit)
}

pub fn quota_increment(reg: &QuotaRegistry, key: &str, amount: u64, limit: u64) -> Result<u64> {
    reg.increment(key, amount, limit)
}

// ---------------------------------------------------------------------------
// Monotonic clock helper for the executor
// ---------------------------------------------------------------------------

/// Returns a monotonic timestamp in microseconds, suitable as now_micros for
/// the rate-limit and bucket primitives
pub fn monotonic_now_micros() -> i64 {
    static BASELINE: parking_lot::Mutex<Option<Instant>> = parking_lot::Mutex::new(None);
    let baseline = {
        let mut guard = BASELINE.lock();
        *guard.get_or_insert_with(Instant::now)
    };
    baseline.elapsed().as_micros() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_limit_allows_within_limit() {
        let reg = RateLimiterRegistry::new();
        for _ in 0..10 {
            assert!(reg.check("user_1", 10, Duration::from_secs(1)));
        }
        // 11th should be rejected
        assert!(!reg.check("user_1", 10, Duration::from_secs(1)));
    }

    #[test]
    fn rate_limit_decision_returns_retry_after() {
        let reg = RateLimiterRegistry::new();
        for _ in 0..5 {
            let d = reg.check_with_decision("api", 5, Duration::from_secs(1));
            assert!(d.allowed);
        }
        let d = reg.check_with_decision("api", 5, Duration::from_secs(1));
        assert!(!d.allowed);
        assert!(d.retry_after_ms > 0);
        assert_eq!(d.limit, 5);
    }

    #[test]
    fn quota_increment_and_release() {
        let reg = QuotaRegistry::new();
        assert_eq!(reg.increment("u1", 100, 1000).unwrap(), 100);
        assert_eq!(reg.increment("u1", 200, 1000).unwrap(), 300);
        let q = reg.check("u1", 1000);
        assert_eq!(q.used, 300);
        assert_eq!(q.remaining, 700);
        assert!(q.allowed);
        reg.release("u1", 100);
        assert_eq!(reg.check("u1", 1000).used, 200);
    }

    #[test]
    fn quota_increment_rejects_over_limit() {
        let reg = QuotaRegistry::new();
        reg.increment("u1", 900, 1000).unwrap();
        assert!(reg.increment("u1", 200, 1000).is_err());
        assert_eq!(reg.check("u1", 1000).used, 900);
    }

    #[test]
    fn quota_merge_remote_takes_max() {
        let reg = QuotaRegistry::new();
        reg.increment("k", 50, 1000).unwrap();
        reg.merge_remote(&[("k".to_string(), 80)]);
        assert_eq!(reg.check("k", 1000).used, 80);
        // Stale remote does not roll backwards
        reg.merge_remote(&[("k".to_string(), 30)]);
        assert_eq!(reg.check("k", 1000).used, 80);
    }

    #[test]
    fn burst_bucket_allows_bursts_from_credit() {
        let bucket = BurstTokenBucket::new(10.0, 1.0, 2.0);
        // Drain to zero then wait for a refill that overflows into credit
        assert!(bucket.consume(10.0, 0));
        // After 15 seconds we should have 10 in the bucket and 5 in credit
        let _ = bucket.consume(0.0, 15_000_000);
        let (cur, credit) = bucket.snapshot();
        assert_eq!(cur, 10.0);
        assert!(credit >= 5.0);
        // Burst 12 tokens, draws 10 from bucket and 2 from credit
        assert!(bucket.consume(12.0, 15_000_000));
    }

    #[test]
    fn cron_list_re_export_works() {
        let expr = cron_parse("0 9 * * 1-5").unwrap();
        let list = cron_list(&expr, 0, 10_000_000_000_000).unwrap();
        // Sanity, returns at least one fire time
        assert!(!list.is_empty());
    }

    #[test]
    fn monotonic_now_micros_advances() {
        let a = monotonic_now_micros();
        std::thread::sleep(Duration::from_micros(50));
        let b = monotonic_now_micros();
        assert!(b > a);
    }
}

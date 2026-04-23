// -----------------------------------------------------------------------------
// Token-bucket rate limiter.
//
// Buckets are keyed by scope (global, per-IP, per-user, per-API-key). Tokens
// use fixed-point (1 token = 1000 subtokens) so fractional refill rates behave
// correctly at sub-second intervals. Refill is computed lazily on each check.
// -----------------------------------------------------------------------------

use std::net::IpAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use parking_lot::Mutex;

const FIXED_POINT: u64 = 1000;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RateLimitKey {
    Global(String),
    PerIp(String, IpAddr),
    PerUser(String, u32),
    PerApiKey(String, String),
}

struct Bucket {
    tokens: AtomicU64,
    last_refill_micros: AtomicU64,
    capacity: u64,
    refill_per_sec: u64,
}

impl Bucket {
    fn new(capacity: u64, refill_per_sec: u64, start_micros: u64) -> Self {
        Self {
            tokens: AtomicU64::new(capacity * FIXED_POINT),
            last_refill_micros: AtomicU64::new(start_micros),
            capacity,
            refill_per_sec,
        }
    }

    fn try_consume(&self, cost: u64, now_micros: u64) -> bool {
        let last = self.last_refill_micros.swap(now_micros, Ordering::AcqRel);
        let elapsed_micros = now_micros.saturating_sub(last);
        let added = (elapsed_micros * self.refill_per_sec * FIXED_POINT) / 1_000_000;
        let cap = self.capacity * FIXED_POINT;
        loop {
            let current = self.tokens.load(Ordering::Acquire);
            let refilled = (current + added).min(cap);
            let need = cost * FIXED_POINT;
            if refilled < need {
                self.tokens.store(refilled, Ordering::Release);
                return false;
            }
            let after = refilled - need;
            if self
                .tokens
                .compare_exchange(current, after, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return true;
            }
        }
    }
}

/// Rate limiter keyed by scope+endpoint. Buckets are created lazily on first
/// access and held for the lifetime of the limiter.
pub struct RateLimiter {
    buckets: Mutex<Vec<(RateLimitKey, Bucket)>>,
    start: Instant,
}

impl RateLimiter {
    pub fn new() -> Self {
        Self {
            buckets: Mutex::new(Vec::new()),
            start: Instant::now(),
        }
    }

    /// Attempts to consume `cost` tokens from the bucket for `key`. Creates a
    /// new bucket if none exists. Returns true when the request fits under the
    /// configured rate.
    pub fn check(&self, key: RateLimitKey, capacity: u64, refill_per_sec: u64, cost: u64) -> bool {
        if capacity == 0 {
            return true;
        }
        let now_micros = self.start.elapsed().as_micros() as u64;
        let mut buckets = self.buckets.lock();
        for (k, b) in buckets.iter() {
            if *k == key {
                return b.try_consume(cost, now_micros);
            }
        }
        let bucket = Bucket::new(capacity, refill_per_sec, now_micros);
        let ok = bucket.try_consume(cost, now_micros);
        buckets.push((key, bucket));
        ok
    }

    /// Drops every stored bucket. Used by admin surfaces and tests.
    pub fn reset(&self) {
        self.buckets.lock().clear();
    }
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    #[test]
    fn allows_under_capacity() {
        let r = RateLimiter::new();
        let k = RateLimitKey::Global("ep".into());
        for _ in 0..5 {
            assert!(r.check(k.clone(), 5, 1, 1));
        }
    }

    #[test]
    fn blocks_over_capacity() {
        let r = RateLimiter::new();
        let k = RateLimitKey::Global("ep".into());
        for _ in 0..3 {
            assert!(r.check(k.clone(), 3, 1, 1));
        }
        assert!(!r.check(k, 3, 1, 1));
    }

    #[test]
    fn disabled_with_zero_capacity() {
        let r = RateLimiter::new();
        let k = RateLimitKey::Global("ep".into());
        for _ in 0..100 {
            assert!(r.check(k.clone(), 0, 0, 1));
        }
    }

    #[test]
    fn per_ip_isolated() {
        let r = RateLimiter::new();
        let a = RateLimitKey::PerIp("ep".into(), IpAddr::V4(Ipv4Addr::new(1, 1, 1, 1)));
        let b = RateLimitKey::PerIp("ep".into(), IpAddr::V4(Ipv4Addr::new(2, 2, 2, 2)));
        assert!(r.check(a.clone(), 1, 1, 1));
        assert!(!r.check(a, 1, 1, 1));
        assert!(r.check(b, 1, 1, 1));
    }

    #[test]
    fn reset_clears() {
        let r = RateLimiter::new();
        let k = RateLimitKey::Global("ep".into());
        assert!(r.check(k.clone(), 1, 0, 1));
        assert!(!r.check(k.clone(), 1, 0, 1));
        r.reset();
        assert!(r.check(k, 1, 0, 1));
    }
}

//! Cleanup governor: a lock-free token bucket plus a preferred-hours window
//! so TTL/tier/purge scans never degrade foreground throughput.

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Rate-limited, time-windowed admission control for cleanup batches.
pub struct CleanupGovernor {
    /// Available row budget (tokens). Refilled by `tick`.
    tokens: AtomicI64,
    /// Max tokens (burst) and refill per second.
    burst: i64,
    refill_per_sec: i64,
    last_refill_us: AtomicI64,
    /// Bitmask of allowed hours (bit h set => hour h is allowed). 0 => all.
    allowed_hours_mask: AtomicU64,
    /// Max rows per single batch.
    max_batch: i64,
}

impl CleanupGovernor {
    pub fn new(burst: i64, refill_per_sec: i64, max_batch: i64) -> Self {
        Self {
            tokens: AtomicI64::new(burst.max(0)),
            burst: burst.max(0),
            refill_per_sec: refill_per_sec.max(0),
            last_refill_us: AtomicI64::new(now_us()),
            allowed_hours_mask: AtomicU64::new(0),
            max_batch: max_batch.max(1),
        }
    }

    /// Restricts cleanup to the given UTC hours (e.g. [2,3,4]). Empty => any.
    pub fn set_preferred_hours(&self, hours: &[u8]) {
        let mut mask = 0u64;
        for h in hours {
            if *h < 24 {
                mask |= 1u64 << h;
            }
        }
        self.allowed_hours_mask.store(mask, Ordering::Release);
    }

    fn in_window(&self, now: i64) -> bool {
        let mask = self.allowed_hours_mask.load(Ordering::Acquire);
        if mask == 0 {
            return true;
        }
        let secs = now / 1_000_000;
        let hour = ((secs / 3600) % 24) as u32;
        (mask >> hour) & 1 == 1
    }

    fn refill(&self, now: i64) {
        let last = self.last_refill_us.load(Ordering::Acquire);
        let elapsed_s = (now - last) / 1_000_000;
        if elapsed_s <= 0 {
            return;
        }
        if self
            .last_refill_us
            .compare_exchange(last, now, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            let add = elapsed_s.saturating_mul(self.refill_per_sec);
            let mut cur = self.tokens.load(Ordering::Acquire);
            loop {
                let next = (cur + add).min(self.burst);
                match self
                    .tokens
                    .compare_exchange(cur, next, Ordering::AcqRel, Ordering::Acquire)
                {
                    Ok(_) => break,
                    Err(observed) => cur = observed,
                }
            }
        }
    }

    /// Attempts to acquire budget for up to `requested` rows. Returns the
    /// number of rows the caller may process this batch (0 = wait/skip,
    /// outside window or no tokens).
    pub fn acquire(&self, requested: i64) -> i64 {
        let now = now_us();
        if !self.in_window(now) {
            return 0;
        }
        self.refill(now);
        let want = requested.clamp(0, self.max_batch);
        if want == 0 {
            return 0;
        }
        let mut cur = self.tokens.load(Ordering::Acquire);
        loop {
            if cur <= 0 {
                return 0;
            }
            let grant = want.min(cur);
            match self.tokens.compare_exchange(
                cur,
                cur - grant,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return grant,
                Err(observed) => cur = observed,
            }
        }
    }
}

impl Default for CleanupGovernor {
    /// 100k row burst, 100k rows/sec refill, 10k per batch.
    fn default() -> Self {
        Self::new(100_000, 100_000, 10_000)
    }
}

fn now_us() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
}

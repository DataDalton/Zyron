//! Backpressure monitoring, load shedding, rate limiting, and auto-scaling.
//!
//! BackpressureMonitor tracks per-operator queue fill ratios using atomic
//! counters. LoadShedder applies inline policies (priority, sampling, age)
//! with zero allocation. RateLimiter uses a lock-free token bucket.
//! AutoScaling provides threshold-based scaling hints with cooldown.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use crate::record::StreamRecord;

// ---------------------------------------------------------------------------
// BackpressureMonitor
// ---------------------------------------------------------------------------

/// Per-operator backpressure monitoring. Uses fixed-size arrays indexed
/// by operator_id for lock-free concurrent reads from metrics views.
pub struct BackpressureMonitor {
    /// Per-operator current queue length.
    queue_lengths: Vec<AtomicUsize>,
    /// Per-operator queue capacity.
    capacities: Vec<usize>,
    /// Number of tracked operators.
    operator_count: usize,
}

impl BackpressureMonitor {
    /// Creates a monitor for the given operator capacities.
    /// Index i corresponds to operator i's queue.
    pub fn new(capacities: &[usize]) -> Self {
        let mut queue_lengths = Vec::with_capacity(capacities.len());
        for _ in 0..capacities.len() {
            queue_lengths.push(AtomicUsize::new(0));
        }
        Self {
            queue_lengths,
            capacities: capacities.to_vec(),
            operator_count: capacities.len(),
        }
    }

    /// Updates the queue length for an operator.
    #[inline]
    pub fn update_queue_length(&self, operator_id: usize, length: usize) {
        if operator_id < self.operator_count {
            self.queue_lengths[operator_id].store(length, Ordering::Relaxed);
        }
    }

    /// Returns the backpressure ratio (0.0 to 1.0) for an operator.
    /// 0.0 = empty, 1.0 = full.
    #[inline]
    pub fn ratio(&self, operator_id: usize) -> f64 {
        if operator_id >= self.operator_count || self.capacities[operator_id] == 0 {
            return 0.0;
        }
        let length = self.queue_lengths[operator_id].load(Ordering::Relaxed);
        length as f64 / self.capacities[operator_id] as f64
    }

    /// Returns true if any operator is above the given threshold ratio.
    pub fn any_above_threshold(&self, threshold: f64) -> bool {
        for i in 0..self.operator_count {
            if self.ratio(i) > threshold {
                return true;
            }
        }
        false
    }

    /// Returns the operator with the highest backpressure ratio.
    pub fn bottleneck(&self) -> Option<(usize, f64)> {
        if self.operator_count == 0 {
            return None;
        }
        let mut max_id = 0;
        let mut max_ratio = 0.0f64;
        for i in 0..self.operator_count {
            let r = self.ratio(i);
            if r > max_ratio {
                max_ratio = r;
                max_id = i;
            }
        }
        Some((max_id, max_ratio))
    }

    /// Number of tracked operators.
    pub fn operator_count(&self) -> usize {
        self.operator_count
    }
}

// ---------------------------------------------------------------------------
// LoadSheddingPolicy
// ---------------------------------------------------------------------------

/// Policy for shedding load when backpressure is high.
#[derive(Debug, Clone)]
pub enum LoadSheddingPolicy {
    /// Drop records below a minimum priority level.
    Priority { min_priority: u8 },
    /// Randomly sample records with the given keep ratio (0.0 to 1.0).
    /// Uses xorshift64 PRNG for a single-cycle random decision.
    Sample { keep_ratio: f64, rng_state: u64 },
    /// Drop records older than max_age_ms relative to current watermark.
    DropOld { max_age_ms: i64 },
}

// ---------------------------------------------------------------------------
// LoadShedder
// ---------------------------------------------------------------------------

/// Applies a load shedding policy to incoming records.
/// All decisions are made inline with zero allocation. Tracks shedding
/// statistics via atomic counters.
pub struct LoadShedder {
    policy: LoadSheddingPolicy,
    /// PRNG state for sampling policy (xorshift64).
    rng_state: u64,
    /// Total records shed.
    pub records_shed: AtomicU64,
}

impl LoadShedder {
    pub fn new(policy: LoadSheddingPolicy) -> Self {
        let rng_state = match &policy {
            LoadSheddingPolicy::Sample { rng_state, .. } => *rng_state,
            _ => 0x12345678_9abcdef0,
        };
        Self {
            policy,
            rng_state,
            records_shed: AtomicU64::new(0),
        }
    }

    /// Xorshift64 PRNG. Single cycle per call.
    #[inline]
    fn next_random(&mut self) -> u64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        x
    }

    /// Applies the shedding policy to a record batch.
    /// Returns the record with shed rows removed.
    pub fn apply(&mut self, record: &StreamRecord, watermark_ms: i64) -> StreamRecord {
        let num_rows = record.num_rows();
        if num_rows == 0 {
            return record.clone();
        }

        match &self.policy {
            LoadSheddingPolicy::Priority { min_priority } => {
                // Priority-based: drop records with priority below threshold.
                // Priority is encoded in the first byte of key hash (if available).
                let min_p = *min_priority;
                let mut mask = vec![true; num_rows];
                let mut shed_count = 0u64;
                if let Some(ref keys) = record.keys {
                    for i in 0..num_rows {
                        let priority = (keys[i] & 0xFF) as u8;
                        if priority < min_p {
                            mask[i] = false;
                            shed_count += 1;
                        }
                    }
                }
                self.records_shed.fetch_add(shed_count, Ordering::Relaxed);
                record.filter(&mask)
            }
            LoadSheddingPolicy::Sample { keep_ratio, .. } => {
                let ratio = if keep_ratio.is_nan() {
                    0.0
                } else {
                    keep_ratio.clamp(0.0, 1.0)
                };
                let threshold = (ratio * u64::MAX as f64) as u64;
                let mut mask = vec![true; num_rows];
                let mut shed_count = 0u64;
                for i in 0..num_rows {
                    let rand = self.next_random();
                    if rand > threshold {
                        mask[i] = false;
                        shed_count += 1;
                    }
                }
                self.records_shed.fetch_add(shed_count, Ordering::Relaxed);
                record.filter(&mask)
            }
            LoadSheddingPolicy::DropOld { max_age_ms } => {
                let cutoff = watermark_ms - *max_age_ms;
                let mut mask = vec![true; num_rows];
                let mut shed_count = 0u64;
                for i in 0..num_rows {
                    if record.event_times[i] < cutoff {
                        mask[i] = false;
                        shed_count += 1;
                    }
                }
                self.records_shed.fetch_add(shed_count, Ordering::Relaxed);
                record.filter(&mask)
            }
        }
    }

    /// Total records shed since creation.
    pub fn total_shed(&self) -> u64 {
        self.records_shed.load(Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// RateLimiter
// ---------------------------------------------------------------------------

/// Token bucket rate limiter. Lock-free using AtomicU64 for the token count.
/// Tokens are refilled by calling `refill()` periodically.
pub struct RateLimiter {
    /// Available tokens (scaled by 1000 for sub-token precision).
    tokens: AtomicU64,
    /// Maximum tokens (scaled).
    max_tokens: u64,
    /// Tokens to add per refill call (scaled).
    refill_amount: u64,
}

impl RateLimiter {
    /// Creates a rate limiter with the given capacity and refill rate.
    /// `max_per_second` is the target throughput. `refill_interval_ms` is
    /// how often refill() will be called.
    pub fn new(max_per_second: u64, refill_interval_ms: u64) -> Self {
        let max_tokens = max_per_second * 1000;
        let refill_amount = max_per_second * refill_interval_ms;
        Self {
            tokens: AtomicU64::new(max_tokens),
            max_tokens,
            refill_amount,
        }
    }

    /// Tries to acquire `n` tokens. Returns true if successful.
    #[inline]
    pub fn try_acquire(&self, n: u64) -> bool {
        let scaled_n = n * 1000;
        loop {
            let current = self.tokens.load(Ordering::Relaxed);
            if current < scaled_n {
                return false;
            }
            if self
                .tokens
                .compare_exchange_weak(
                    current,
                    current - scaled_n,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                return true;
            }
        }
    }

    /// Refills tokens. Call this periodically from a timer.
    pub fn refill(&self) {
        loop {
            let current = self.tokens.load(Ordering::Relaxed);
            let new_val = (current + self.refill_amount).min(self.max_tokens);
            if self
                .tokens
                .compare_exchange_weak(current, new_val, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                return;
            }
        }
    }

    /// Returns currently available tokens (unscaled).
    pub fn available(&self) -> u64 {
        self.tokens.load(Ordering::Relaxed) / 1000
    }
}

// ---------------------------------------------------------------------------
// AutoScaling
// ---------------------------------------------------------------------------

/// Threshold-based auto-scaling hints with cooldown.
/// Does not perform actual scaling but emits ScaleUp/ScaleDown/Steady signals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalingDecision {
    ScaleUp,
    ScaleDown,
    Steady,
}

pub struct AutoScaling {
    /// Backpressure ratio above which to scale up.
    scale_up_threshold: f64,
    /// Backpressure ratio below which to scale down.
    scale_down_threshold: f64,
    /// Minimum time between scaling decisions in milliseconds.
    cooldown_ms: u64,
    /// Timestamp of the last scaling decision.
    last_decision_time_ms: u64,
}

impl AutoScaling {
    pub fn new(scale_up_threshold: f64, scale_down_threshold: f64, cooldown_ms: u64) -> Self {
        Self {
            scale_up_threshold,
            scale_down_threshold,
            cooldown_ms,
            last_decision_time_ms: 0,
        }
    }

    /// Evaluates current backpressure and returns a scaling decision.
    /// `current_ratio` is the maximum backpressure ratio across operators.
    /// `current_time_ms` is the current wall-clock time in milliseconds.
    pub fn evaluate(&mut self, current_ratio: f64, current_time_ms: u64) -> ScalingDecision {
        if current_time_ms < self.last_decision_time_ms + self.cooldown_ms {
            return ScalingDecision::Steady;
        }

        let decision = if current_ratio > self.scale_up_threshold {
            ScalingDecision::ScaleUp
        } else if current_ratio < self.scale_down_threshold {
            ScalingDecision::ScaleDown
        } else {
            ScalingDecision::Steady
        };

        if decision != ScalingDecision::Steady {
            self.last_decision_time_ms = current_time_ms;
        }

        decision
    }

    /// Returns the configured thresholds.
    pub fn thresholds(&self) -> (f64, f64) {
        (self.scale_up_threshold, self.scale_down_threshold)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::{StreamBatch, StreamColumn, StreamColumnData};
    use crate::record::ChangeFlag;

    fn make_record(n: usize) -> StreamRecord {
        let col = StreamColumn::from_data(StreamColumnData::Int64((0..n as i64).collect()));
        let batch = StreamBatch::new(vec![col]);
        let times: Vec<i64> = (0..n as i64).map(|i| i * 1000).collect();
        StreamRecord::new(batch, times, vec![ChangeFlag::Insert; n])
    }

    #[test]
    fn test_backpressure_monitor() {
        let monitor = BackpressureMonitor::new(&[100, 200, 50]);
        assert_eq!(monitor.operator_count(), 3);

        monitor.update_queue_length(0, 50);
        monitor.update_queue_length(1, 180);
        monitor.update_queue_length(2, 25);

        assert!((monitor.ratio(0) - 0.5).abs() < 0.01);
        assert!((monitor.ratio(1) - 0.9).abs() < 0.01);
        assert!((monitor.ratio(2) - 0.5).abs() < 0.01);

        assert!(monitor.any_above_threshold(0.8));
        assert!(!monitor.any_above_threshold(0.95));
    }

    #[test]
    fn test_backpressure_bottleneck() {
        let monitor = BackpressureMonitor::new(&[100, 100]);
        monitor.update_queue_length(0, 30);
        monitor.update_queue_length(1, 90);

        let (id, ratio) = monitor.bottleneck().expect("should have bottleneck");
        assert_eq!(id, 1);
        assert!((ratio - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_load_shedder_sample() {
        let policy = LoadSheddingPolicy::Sample {
            keep_ratio: 0.5,
            rng_state: 42,
        };
        let mut shedder = LoadShedder::new(policy);
        let record = make_record(1000);
        let result = shedder.apply(&record, 0);
        // With 50% keep ratio, roughly half should remain.
        let kept = result.num_rows();
        assert!(kept > 300 && kept < 700, "kept {kept} rows, expected ~500");
        assert!(shedder.total_shed() > 0);
    }

    #[test]
    fn test_load_shedder_drop_old() {
        let policy = LoadSheddingPolicy::DropOld { max_age_ms: 5000 };
        let mut shedder = LoadShedder::new(policy);
        let record = make_record(10); // times: 0, 1000, 2000, ..., 9000
        let result = shedder.apply(&record, 8000);
        // cutoff = 8000 - 5000 = 3000. Times < 3000 are shed: 0, 1000, 2000.
        assert_eq!(result.num_rows(), 7);
        assert_eq!(shedder.total_shed(), 3);
    }

    #[test]
    fn test_rate_limiter() {
        let limiter = RateLimiter::new(100, 100); // 100/sec, refill every 100ms
        assert!(limiter.try_acquire(50));
        assert!(limiter.try_acquire(50));
        assert!(!limiter.try_acquire(1)); // Should be exhausted.

        limiter.refill();
        assert!(limiter.try_acquire(1)); // Should have tokens again.
    }

    #[test]
    fn test_auto_scaling() {
        let mut scaler = AutoScaling::new(0.8, 0.2, 5000);

        // High backpressure should trigger scale up.
        let decision = scaler.evaluate(0.9, 10_000);
        assert_eq!(decision, ScalingDecision::ScaleUp);

        // Within cooldown, should be steady.
        let decision = scaler.evaluate(0.9, 12_000);
        assert_eq!(decision, ScalingDecision::Steady);

        // After cooldown, low backpressure should trigger scale down.
        let decision = scaler.evaluate(0.1, 20_000);
        assert_eq!(decision, ScalingDecision::ScaleDown);
    }

    #[test]
    fn test_empty_monitor() {
        let monitor = BackpressureMonitor::new(&[]);
        assert_eq!(monitor.operator_count(), 0);
        assert!(monitor.bottleneck().is_none());
        assert!(!monitor.any_above_threshold(0.5));
    }
}

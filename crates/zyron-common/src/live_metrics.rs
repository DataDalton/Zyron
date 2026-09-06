//! Live query metrics every connection records and the health probe reads.
//!
//! Counters, a latency histogram with quantile estimates, and a sliding
//! window of per-second buckets, so throughput and error rate are answered
//! from the last minute rather than from a lifetime average. Everything is
//! lock free, one atomic add per event on the query path, and the readers
//! are the metrics endpoint and the upgrade driver's health probe, both of
//! which run at controller cadence

use std::sync::atomic::{AtomicU64, Ordering};

use crate::format::upgrade::HealthBaseline;

/// Bucket upper bounds in microseconds, one millisecond to ten seconds
pub const LATENCY_BUCKETS_US: &[u64] = &[
    1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000,
];

/// Seconds the rate window covers
pub const RATE_WINDOW_SECS: u64 = 60;

/// Lock-free latency histogram with fixed exponential buckets
pub struct LatencyHistogram {
    /// One counter per boundary plus one for everything past the last
    buckets: Vec<AtomicU64>,
    sum_us: AtomicU64,
    count: AtomicU64,
}

impl LatencyHistogram {
    pub fn new() -> Self {
        let mut buckets = Vec::with_capacity(LATENCY_BUCKETS_US.len() + 1);
        for _ in 0..=LATENCY_BUCKETS_US.len() {
            buckets.push(AtomicU64::new(0));
        }
        Self {
            buckets,
            sum_us: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }

    /// Records one observation in microseconds
    pub fn record(&self, duration_us: u64) {
        self.sum_us.fetch_add(duration_us, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
        let idx = LATENCY_BUCKETS_US
            .iter()
            .position(|&bound| duration_us <= bound)
            .unwrap_or(LATENCY_BUCKETS_US.len());
        self.buckets[idx].fetch_add(1, Ordering::Relaxed);
    }

    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    pub fn sum_us(&self) -> u64 {
        self.sum_us.load(Ordering::Relaxed)
    }

    /// Cumulative counts per boundary, the last entry being the total
    pub fn cumulative(&self) -> Vec<u64> {
        let mut running = 0u64;
        self.buckets
            .iter()
            .map(|bucket| {
                running += bucket.load(Ordering::Relaxed);
                running
            })
            .collect()
    }

    /// The upper bound of the bucket holding the given quantile, zero when
    /// nothing has been recorded so an idle node reads as no latency
    pub fn quantile_estimate_us(&self, quantile: f64) -> u64 {
        let total = self.count.load(Ordering::Relaxed);
        if total == 0 {
            return 0;
        }
        let target = ((total as f64) * quantile.clamp(0.0, 1.0)).ceil().max(1.0) as u64;
        let mut cumulative = 0u64;
        for (i, &bound) in LATENCY_BUCKETS_US.iter().enumerate() {
            cumulative += self.buckets[i].load(Ordering::Relaxed);
            if cumulative >= target {
                return bound;
            }
        }
        LATENCY_BUCKETS_US.last().copied().unwrap_or(u64::MAX)
    }

    pub fn p50_estimate_us(&self) -> u64 {
        self.quantile_estimate_us(0.50)
    }

    pub fn p99_estimate_us(&self) -> u64 {
        self.quantile_estimate_us(0.99)
    }
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self::new()
    }
}

/// One second of the rate window
struct RateSlot {
    second: AtomicU64,
    queries: AtomicU64,
    errors: AtomicU64,
}

/// What the connections record, read by the metrics endpoint and the
/// health probe
pub struct QueryMetrics {
    pub connections_total: AtomicU64,
    pub queries_total: AtomicU64,
    pub errors_total: AtomicU64,
    pub latency: LatencyHistogram,
    /// The unix second of the first recorded query, zero before any
    first_record_secs: AtomicU64,
    slots: Vec<RateSlot>,
}

impl QueryMetrics {
    pub fn new() -> Self {
        let slots = (0..RATE_WINDOW_SECS)
            .map(|_| RateSlot {
                second: AtomicU64::new(0),
                queries: AtomicU64::new(0),
                errors: AtomicU64::new(0),
            })
            .collect();
        Self {
            connections_total: AtomicU64::new(0),
            queries_total: AtomicU64::new(0),
            errors_total: AtomicU64::new(0),
            latency: LatencyHistogram::new(),
            first_record_secs: AtomicU64::new(0),
            slots,
        }
    }

    pub fn connection_opened(&self) {
        self.connections_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Records one finished statement.
    ///
    /// The window slot for this second is reset when it still holds an older
    /// second. Two recorders crossing a second boundary at once can lose a
    /// count to the reset, which moves a rate by one event in sixty seconds
    /// and is not worth a lock on the query path
    pub fn record_query(&self, now_secs: u64, duration_us: u64, failed: bool) {
        self.queries_total.fetch_add(1, Ordering::Relaxed);
        if failed {
            self.errors_total.fetch_add(1, Ordering::Relaxed);
        }
        self.latency.record(duration_us);
        let _ = self.first_record_secs.compare_exchange(
            0,
            now_secs.max(1),
            Ordering::Relaxed,
            Ordering::Relaxed,
        );
        let slot = &self.slots[(now_secs % RATE_WINDOW_SECS) as usize];
        if slot.second.load(Ordering::Relaxed) != now_secs {
            slot.queries.store(0, Ordering::Relaxed);
            slot.errors.store(0, Ordering::Relaxed);
            slot.second.store(now_secs, Ordering::Relaxed);
        }
        slot.queries.fetch_add(1, Ordering::Relaxed);
        if failed {
            slot.errors.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Queries per second and the failed fraction over the last window,
    /// scaled to the seconds the node has actually been recording when that
    /// is shorter than the window
    pub fn rates(&self, now_secs: u64) -> (f64, f64) {
        let (queries, errors, window) = self.window_counts(now_secs);
        (queries as f64 / window, error_fraction(queries, errors))
    }

    /// Queries and failures inside the window, and the seconds of it the
    /// node has been recording for
    fn window_counts(&self, now_secs: u64) -> (u64, u64, f64) {
        let first = self.first_record_secs.load(Ordering::Relaxed);
        if first == 0 {
            return (0, 0, 1.0);
        }
        let oldest = now_secs.saturating_sub(RATE_WINDOW_SECS - 1);
        let mut queries = 0u64;
        let mut errors = 0u64;
        for slot in &self.slots {
            let second = slot.second.load(Ordering::Relaxed);
            if second >= oldest && second <= now_secs {
                queries += slot.queries.load(Ordering::Relaxed);
                errors += slot.errors.load(Ordering::Relaxed);
            }
        }
        let recording_for = now_secs.saturating_sub(first) + 1;
        let window = recording_for.clamp(1, RATE_WINDOW_SECS) as f64;
        (queries, errors, window)
    }

    /// The node's health as the upgrade driver judges it
    pub fn sample(&self, now_secs: u64, active_connections: u64) -> HealthBaseline {
        let (queries, errors, window) = self.window_counts(now_secs);
        HealthBaseline {
            p50_latency_us: self.latency.p50_estimate_us(),
            p99_latency_us: self.latency.p99_estimate_us(),
            throughput_per_sec: queries as f64 / window,
            error_rate: error_fraction(queries, errors),
            active_connections,
            queries_in_window: queries,
        }
    }
}

/// The failed share of the queries, zero when there were none
fn error_fraction(queries: u64, errors: u64) -> f64 {
    if queries == 0 {
        0.0
    } else {
        errors as f64 / queries as f64
    }
}

impl Default for QueryMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_histogram_reads_zero_at_every_quantile() {
        let h = LatencyHistogram::new();
        assert_eq!(h.p50_estimate_us(), 0);
        assert_eq!(h.p99_estimate_us(), 0);
        assert_eq!(h.count(), 0);
    }

    #[test]
    fn quantiles_land_in_the_bucket_holding_them() {
        let h = LatencyHistogram::new();
        for _ in 0..99 {
            h.record(500);
        }
        h.record(2_000_000);
        assert_eq!(h.p50_estimate_us(), 1_000);
        assert_eq!(h.p99_estimate_us(), 1_000);
        assert_eq!(h.quantile_estimate_us(1.0), 5_000_000);
        assert_eq!(h.count(), 100);
        assert_eq!(h.sum_us(), 99 * 500 + 2_000_000);
    }

    #[test]
    fn rates_cover_the_last_window_only() {
        let m = QueryMetrics::new();
        for second in 100..110 {
            m.record_query(second, 100, false);
            m.record_query(second, 100, second % 2 == 0);
        }
        let (throughput, error_rate) = m.rates(109);
        // Ten seconds of recording, twenty queries, one failed on each of the
        // five even seconds
        assert!((throughput - 2.0).abs() < 1e-9, "{throughput}");
        assert!((error_rate - 0.25).abs() < 1e-9, "{error_rate}");

        // Two minutes later those seconds are outside the window
        let (later, _) = m.rates(300);
        assert_eq!(later, 0.0);
        assert_eq!(m.queries_total.load(Ordering::Relaxed), 20);
        assert_eq!(m.errors_total.load(Ordering::Relaxed), 5);
    }

    #[test]
    fn a_reused_slot_forgets_the_older_second() {
        let m = QueryMetrics::new();
        m.record_query(5, 100, false);
        m.record_query(5 + RATE_WINDOW_SECS, 100, false);
        let (throughput, _) = m.rates(5 + RATE_WINDOW_SECS);
        // The older second landed in the same slot and was cleared, so one
        // query over a full window
        assert!((throughput - 1.0 / RATE_WINDOW_SECS as f64).abs() < 1e-9);
    }

    #[test]
    fn a_sample_carries_every_field_the_judge_reads() {
        let m = QueryMetrics::new();
        m.connection_opened();
        for _ in 0..10 {
            m.record_query(50, 3_000, false);
        }
        let sample = m.sample(50, 7);
        assert_eq!(sample.p50_latency_us, 5_000);
        assert_eq!(sample.p99_latency_us, 5_000);
        assert!((sample.throughput_per_sec - 10.0).abs() < 1e-9);
        assert_eq!(sample.error_rate, 0.0);
        assert_eq!(sample.active_connections, 7);
        assert_eq!(sample.queries_in_window, 10);
    }
}

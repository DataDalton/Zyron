//! Lock-free metrics collection with Prometheus text exposition.
//!
//! All counters and gauges use atomic operations for zero-contention updates
//! from connection handler threads. The histogram uses per-bucket atomic
//! counters for recording query latency without locks.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::session::SessionManager;

/// Lock-free latency histogram with fixed exponential buckets.
/// Each bucket boundary is in microseconds. The record() method
/// atomically increments the correct bucket counter.
pub struct LatencyHistogram {
    /// Bucket upper bounds in microseconds.
    boundaries: &'static [u64],
    /// Per-bucket counters (one extra for +Inf).
    buckets: Vec<AtomicU64>,
    /// Running sum of all recorded values in microseconds.
    sum: AtomicU64,
    /// Total number of observations.
    count: AtomicU64,
}

/// Prometheus bucket boundaries in microseconds:
/// 1ms, 5ms, 10ms, 50ms, 100ms, 500ms, 1s, 5s, 10s
static LATENCY_BUCKETS_US: &[u64] = &[
    1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000,
];

impl LatencyHistogram {
    /// Creates a new histogram with the default bucket boundaries.
    pub fn new() -> Self {
        let boundaries = LATENCY_BUCKETS_US;
        let mut buckets = Vec::with_capacity(boundaries.len() + 1);
        for _ in 0..=boundaries.len() {
            buckets.push(AtomicU64::new(0));
        }
        Self {
            boundaries,
            buckets,
            sum: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }

    /// Records a latency observation in microseconds.
    pub fn record(&self, duration_us: u64) {
        self.sum.fetch_add(duration_us, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);

        // Find the first bucket where the value fits
        let idx = self
            .boundaries
            .iter()
            .position(|&b| duration_us <= b)
            .unwrap_or(self.boundaries.len());
        self.buckets[idx].fetch_add(1, Ordering::Relaxed);
    }

    /// Estimates the 99th percentile latency in microseconds from the
    /// cumulative bucket counts. Returns the upper bound of the bucket that
    /// contains the p99 observation (standard histogram-quantile bucket
    /// estimate). Returns 0 when no observations have been recorded so the
    /// caller treats an idle system as no backpressure.
    pub fn p99_estimate_us(&self) -> u64 {
        let total = self.count.load(Ordering::Relaxed);
        if total == 0 {
            return 0;
        }
        let target = ((total as f64) * 0.99).ceil() as u64;
        let mut cumulative: u64 = 0;
        for (i, &boundary) in self.boundaries.iter().enumerate() {
            cumulative += self.buckets[i].load(Ordering::Relaxed);
            if cumulative >= target {
                return boundary;
            }
        }
        // p99 falls in the +Inf bucket: return the largest boundary as the
        // best finite estimate.
        self.boundaries.last().copied().unwrap_or(u64::MAX)
    }

    /// Renders histogram lines in Prometheus text format.
    fn render_prometheus(&self, name: &str, help: &str, out: &mut String) {
        out.push_str(&format!("# HELP {} {}\n", name, help));
        out.push_str(&format!("# TYPE {} histogram\n", name));

        let mut cumulative: u64 = 0;
        for (i, &boundary) in self.boundaries.iter().enumerate() {
            cumulative += self.buckets[i].load(Ordering::Relaxed);
            let le_seconds = boundary as f64 / 1_000_000.0;
            out.push_str(&format!(
                "{}{{le=\"{}\"}} {}\n",
                name, le_seconds, cumulative
            ));
        }
        // +Inf bucket
        cumulative += self.buckets[self.boundaries.len()].load(Ordering::Relaxed);
        out.push_str(&format!("{}{{le=\"+Inf\"}} {}\n", name, cumulative));

        let sum_seconds = self.sum.load(Ordering::Relaxed) as f64 / 1_000_000.0;
        out.push_str(&format!("{}_sum {}\n", name, sum_seconds));
        out.push_str(&format!(
            "{}_count {}\n",
            name,
            self.count.load(Ordering::Relaxed)
        ));
    }
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self::new()
    }
}

/// Central metrics registry. All fields use atomic operations for
/// lock-free concurrent updates from any thread.
pub struct MetricsRegistry {
    // Counters (monotonically increasing)
    pub connections_total: AtomicU64,
    pub queries_total: AtomicU64,
    pub errors_total: AtomicU64,
    pub transactions_committed: AtomicU64,
    pub transactions_aborted: AtomicU64,
    pub bytes_sent: AtomicU64,
    pub bytes_received: AtomicU64,

    // Histograms
    pub query_duration: LatencyHistogram,

    // Labeled metric series shared with emit sites in lower crates
    pub labeled: Arc<zyron_common::LabeledMetrics>,

    // References for gauge sampling
    session_mgr: Arc<SessionManager>,
}

impl MetricsRegistry {
    /// Creates a new metrics registry with all counters at zero.
    pub fn new(
        session_mgr: Arc<SessionManager>,
        labeled: Arc<zyron_common::LabeledMetrics>,
    ) -> Self {
        Self {
            connections_total: AtomicU64::new(0),
            queries_total: AtomicU64::new(0),
            errors_total: AtomicU64::new(0),
            transactions_committed: AtomicU64::new(0),
            transactions_aborted: AtomicU64::new(0),
            bytes_sent: AtomicU64::new(0),
            bytes_received: AtomicU64::new(0),
            query_duration: LatencyHistogram::new(),
            labeled,
            session_mgr,
        }
    }

    /// Renders all metrics in Prometheus text exposition format.
    pub fn render_prometheus(&self) -> String {
        let mut out = String::with_capacity(4096);

        // Counters
        render_counter(
            &mut out,
            "zyron_connections_total",
            "Total connections accepted since server start",
            self.connections_total.load(Ordering::Relaxed),
        );
        render_counter(
            &mut out,
            "zyron_queries_total",
            "Total queries executed",
            self.queries_total.load(Ordering::Relaxed),
        );
        render_counter(
            &mut out,
            "zyron_errors_total",
            "Total query errors",
            self.errors_total.load(Ordering::Relaxed),
        );
        render_counter(
            &mut out,
            "zyron_transactions_committed_total",
            "Total committed transactions",
            self.transactions_committed.load(Ordering::Relaxed),
        );
        render_counter(
            &mut out,
            "zyron_transactions_aborted_total",
            "Total aborted transactions",
            self.transactions_aborted.load(Ordering::Relaxed),
        );
        render_counter(
            &mut out,
            "zyron_bytes_sent_total",
            "Total bytes sent to clients",
            self.bytes_sent.load(Ordering::Relaxed),
        );
        render_counter(
            &mut out,
            "zyron_bytes_received_total",
            "Total bytes received from clients",
            self.bytes_received.load(Ordering::Relaxed),
        );

        // Gauges
        render_gauge(
            &mut out,
            "zyron_active_connections",
            "Current number of active connections",
            self.session_mgr.active_count() as u64,
        );
        render_gauge(
            &mut out,
            "zyron_max_connections",
            "Maximum allowed connections",
            self.session_mgr.max_connections() as u64,
        );

        // Histogram
        self.query_duration.render_prometheus(
            "zyron_query_duration_seconds",
            "Query execution duration in seconds",
            &mut out,
        );

        // Labeled publication/subscription/credential/TLS series
        self.labeled.render_prometheus(&mut out);

        out
    }
}

fn render_counter(out: &mut String, name: &str, help: &str, value: u64) {
    out.push_str(&format!("# HELP {} {}\n", name, help));
    out.push_str(&format!("# TYPE {} counter\n", name));
    out.push_str(&format!("{} {}\n", name, value));
}

fn render_gauge(out: &mut String, name: &str, help: &str, value: u64) {
    out.push_str(&format!("# HELP {} {}\n", name, help));
    out.push_str(&format!("# TYPE {} gauge\n", name));
    out.push_str(&format!("{} {}\n", name, value));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_record() {
        let h = LatencyHistogram::new();
        h.record(500); // 0.5ms, bucket[0] (<=1ms)
        h.record(3000); // 3ms, bucket[1] (<=5ms)
        h.record(50000); // 50ms, bucket[3] (<=50ms)
        h.record(20_000_000); // 20s, +Inf bucket

        assert_eq!(h.count.load(Ordering::Relaxed), 4);
        assert_eq!(
            h.sum.load(Ordering::Relaxed),
            500 + 3000 + 50000 + 20_000_000
        );
        assert_eq!(h.buckets[0].load(Ordering::Relaxed), 1); // <=1ms
        assert_eq!(h.buckets[1].load(Ordering::Relaxed), 1); // <=5ms
        assert_eq!(h.buckets[3].load(Ordering::Relaxed), 1); // <=50ms
        assert_eq!(h.buckets[9].load(Ordering::Relaxed), 1); // +Inf
    }

    #[test]
    fn test_histogram_prometheus_format() {
        let h = LatencyHistogram::new();
        h.record(1_000); // exactly 1ms
        h.record(100_000); // 100ms

        let mut out = String::new();
        h.render_prometheus("test_metric", "A test metric", &mut out);

        assert!(out.contains("# HELP test_metric A test metric"));
        assert!(out.contains("# TYPE test_metric histogram"));
        assert!(out.contains("test_metric{le=\"+Inf\"} 2"));
        assert!(out.contains("test_metric_count 2"));
    }

    #[test]
    fn test_metrics_registry_render() {
        let session_mgr = Arc::new(SessionManager::new(100, 0));
        let labeled = Arc::new(zyron_common::LabeledMetrics::new());
        let registry = MetricsRegistry::new(session_mgr, labeled);

        registry.connections_total.fetch_add(10, Ordering::Relaxed);
        registry.queries_total.fetch_add(50, Ordering::Relaxed);
        registry.errors_total.fetch_add(2, Ordering::Relaxed);

        let output = registry.render_prometheus();
        assert!(output.contains("zyron_connections_total 10"));
        assert!(output.contains("zyron_queries_total 50"));
        assert!(output.contains("zyron_errors_total 2"));
        assert!(output.contains("zyron_active_connections 0"));
        assert!(output.contains("zyron_max_connections 100"));
    }

    #[test]
    fn test_labeled_families_render_and_accumulate() {
        use zyron_common::TlsDirection;
        let session_mgr = Arc::new(SessionManager::new(100, 0));
        let labeled = Arc::new(zyron_common::LabeledMetrics::new());
        let registry = MetricsRegistry::new(session_mgr, labeled.clone());

        labeled.pubSubscribersInc("pub1");
        labeled.pubBytesSent("pub1", 4096);
        labeled.pubRetentionLagSet("pub1", 12);
        labeled.subLagLsnSet("sub1", 7);
        labeled.subLastPollSet("sub1", 1_700_000_000);
        labeled.subReconnectInc("sub1");
        labeled.credCacheHit("aws_sts");
        labeled.credCacheMiss("aws_sts");
        labeled.credRefresh("aws_sts");
        labeled.tlsHandshake(TlsDirection::Inbound, true);
        labeled.tlsHandshake(TlsDirection::Outbound, false);
        labeled.tlsSessionResumed();

        let out = registry.render_prometheus();
        assert!(out.contains("zyron_publication_active_subscribers{publication=\"pub1\"} 1"));
        assert!(out.contains("zyron_publication_bytes_sent_total{publication=\"pub1\"} 4096"));
        assert!(out.contains("zyron_publication_retention_lag_seconds{publication=\"pub1\"} 12"));
        assert!(out.contains("zyron_subscription_lag_lsn{subscription=\"sub1\"} 7"));
        assert!(
            out.contains(
                "zyron_subscription_last_poll_timestamp{subscription=\"sub1\"} 1700000000"
            )
        );
        assert!(out.contains("zyron_subscription_reconnects_total{subscription=\"sub1\"} 1"));
        assert!(out.contains("zyron_credential_cache_hits_total{provider=\"aws_sts\"} 1"));
        assert!(out.contains("zyron_credential_cache_misses_total{provider=\"aws_sts\"} 1"));
        assert!(out.contains("zyron_credential_refreshes_total{provider=\"aws_sts\"} 1"));
        assert!(out.contains("zyron_tls_handshakes_total{direction=\"inbound\",result=\"ok\"} 1"));
        assert!(
            out.contains("zyron_tls_handshakes_total{direction=\"outbound\",result=\"fail\"} 1")
        );
        assert!(out.contains("zyron_tls_session_resumptions_total 1"));

        // Counter accumulates rather than overwrites.
        labeled.pubBytesSent("pub1", 1000);
        let out2 = registry.render_prometheus();
        assert!(out2.contains("zyron_publication_bytes_sent_total{publication=\"pub1\"} 5096"));
    }
}

//! Lock-free metrics collection with Prometheus text exposition.
//!
//! The query counters and the latency histogram are the `QueryMetrics` every
//! connection records into, shared with the upgrade driver's health probe so
//! the exposition and the probe read one set of numbers. Everything else is
//! an atomic counter updated from any thread without a lock.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use zyron_common::QueryMetrics;
use zyron_common::live_metrics::LATENCY_BUCKETS_US;

use crate::session::SessionManager;

/// Central metrics registry. All fields use atomic operations for
/// lock-free concurrent updates from any thread.
pub struct MetricsRegistry {
    /// Connections, queries, errors, and query latency, recorded by the
    /// wire layer and read by the upgrade driver as well as by the
    /// exposition
    pub query: Arc<QueryMetrics>,

    // Counters (monotonically increasing)
    pub transactions_committed: AtomicU64,
    pub transactions_aborted: AtomicU64,
    pub bytes_sent: AtomicU64,
    pub bytes_received: AtomicU64,

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
        query: Arc<QueryMetrics>,
    ) -> Self {
        Self {
            query,
            transactions_committed: AtomicU64::new(0),
            transactions_aborted: AtomicU64::new(0),
            bytes_sent: AtomicU64::new(0),
            bytes_received: AtomicU64::new(0),
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
            self.query.connections_total.load(Ordering::Relaxed),
        );
        render_counter(
            &mut out,
            "zyron_queries_total",
            "Total queries executed",
            self.query.queries_total.load(Ordering::Relaxed),
        );
        render_counter(
            &mut out,
            "zyron_errors_total",
            "Total query errors",
            self.query.errors_total.load(Ordering::Relaxed),
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
            "Connection ceiling derived from the memory the node measured",
            self.session_mgr.max_connections() as u64,
        );

        // Histogram
        render_histogram(
            &mut out,
            "zyron_query_duration_seconds",
            "Query execution duration in seconds",
            &self.query,
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

/// Renders the query latency histogram in Prometheus text format, the
/// bucket bounds converted from microseconds to the seconds Prometheus
/// expects
fn render_histogram(out: &mut String, name: &str, help: &str, query: &QueryMetrics) {
    out.push_str(&format!("# HELP {} {}\n", name, help));
    out.push_str(&format!("# TYPE {} histogram\n", name));
    let cumulative = query.latency.cumulative();
    for (i, &boundary) in LATENCY_BUCKETS_US.iter().enumerate() {
        let le_seconds = boundary as f64 / 1_000_000.0;
        let count = cumulative.get(i).copied().unwrap_or(0);
        out.push_str(&format!("{}{{le=\"{}\"}} {}\n", name, le_seconds, count));
    }
    let total = cumulative.last().copied().unwrap_or(0);
    out.push_str(&format!("{}{{le=\"+Inf\"}} {}\n", name, total));
    let sum_seconds = query.latency.sum_us() as f64 / 1_000_000.0;
    out.push_str(&format!("{}_sum {}\n", name, sum_seconds));
    out.push_str(&format!("{}_count {}\n", name, query.latency.count()));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry() -> (MetricsRegistry, Arc<zyron_common::LabeledMetrics>) {
        let session_mgr = Arc::new(SessionManager::new(0));
        let labeled = Arc::new(zyron_common::LabeledMetrics::new());
        let query = Arc::new(QueryMetrics::new());
        (
            MetricsRegistry::new(session_mgr, labeled.clone(), query),
            labeled,
        )
    }

    #[test]
    fn test_histogram_prometheus_format() {
        let (registry, _) = registry();
        registry.query.latency.record(1_000);
        registry.query.latency.record(100_000);

        let mut out = String::new();
        render_histogram(&mut out, "test_metric", "A test metric", &registry.query);

        assert!(out.contains("# HELP test_metric A test metric"));
        assert!(out.contains("# TYPE test_metric histogram"));
        assert!(out.contains("test_metric{le=\"0.001\"} 1"));
        assert!(out.contains("test_metric{le=\"0.1\"} 2"));
        assert!(out.contains("test_metric{le=\"+Inf\"} 2"));
        assert!(out.contains("test_metric_count 2"));
    }

    #[test]
    fn test_metrics_registry_render() {
        let (registry, _) = registry();
        registry
            .query
            .connections_total
            .fetch_add(10, Ordering::Relaxed);
        registry
            .query
            .queries_total
            .fetch_add(50, Ordering::Relaxed);
        registry.query.errors_total.fetch_add(2, Ordering::Relaxed);

        let output = registry.render_prometheus();
        assert!(output.contains("zyron_connections_total 10"));
        assert!(output.contains("zyron_queries_total 50"));
        assert!(output.contains("zyron_errors_total 2"));
        assert!(output.contains("zyron_active_connections 0"));
        // The ceiling is what the node's memory affords, not a configured
        // count, so the metric is checked against the gauge that enforces it
        let ceiling = zyron_pressure::pressure_control::PressureController::global()
            .connections()
            .ceiling();
        assert!(
            output.contains(&format!("zyron_max_connections {ceiling}")),
            "the exported ceiling disagrees with the one being enforced"
        );
        assert!(ceiling >= 1);
    }

    #[test]
    fn test_labeled_families_render_and_accumulate() {
        use zyron_common::TlsDirection;
        let (registry, labeled) = registry();

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

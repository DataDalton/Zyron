// -----------------------------------------------------------------------------
// Gateway metrics.
//
// Tracks per-endpoint counters (requests, errors, rate-limited, circuit-open)
// and a coarse latency histogram. GatewayMetrics holds the set of endpoints
// registered with the router. Rendering concatenates every endpoint's
// counters in Prometheus text format.
// -----------------------------------------------------------------------------

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;

/// Histogram buckets, in microseconds.
pub const LATENCY_BUCKETS_US: [u64; 11] = [
    100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000,
];

/// Per-endpoint counter set.
#[derive(Debug)]
pub struct EndpointMetrics {
    pub requests_total: AtomicU64,
    pub errors_total: AtomicU64,
    pub rate_limited_total: AtomicU64,
    pub circuit_open_total: AtomicU64,
    pub ws_active: AtomicU64,
    pub ws_messages_total: AtomicU64,
    pub ws_bytes_total: AtomicU64,
    pub latency_buckets: [AtomicU64; LATENCY_BUCKETS_US.len()],
    pub latency_sum_us: AtomicU64,
    pub last_status: AtomicU64,
}

impl EndpointMetrics {
    pub fn new() -> Self {
        let buckets: [AtomicU64; LATENCY_BUCKETS_US.len()] = Default::default();
        Self {
            requests_total: AtomicU64::new(0),
            errors_total: AtomicU64::new(0),
            rate_limited_total: AtomicU64::new(0),
            circuit_open_total: AtomicU64::new(0),
            ws_active: AtomicU64::new(0),
            ws_messages_total: AtomicU64::new(0),
            ws_bytes_total: AtomicU64::new(0),
            latency_buckets: buckets,
            latency_sum_us: AtomicU64::new(0),
            last_status: AtomicU64::new(0),
        }
    }

    pub fn record(&self, latency_us: u64, status: u16, is_error: bool) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
        if is_error {
            self.errors_total.fetch_add(1, Ordering::Relaxed);
        }
        self.last_status.store(status as u64, Ordering::Relaxed);
        self.latency_sum_us.fetch_add(latency_us, Ordering::Relaxed);
        for (i, edge) in LATENCY_BUCKETS_US.iter().enumerate() {
            if latency_us <= *edge {
                self.latency_buckets[i].fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    pub fn inc_rate_limited(&self) {
        self.rate_limited_total.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_circuit_open(&self) {
        self.circuit_open_total.fetch_add(1, Ordering::Relaxed);
    }

    pub fn ws_conn_open(&self) {
        self.ws_active.fetch_add(1, Ordering::Relaxed);
    }

    pub fn ws_conn_close(&self) {
        self.ws_active.fetch_sub(1, Ordering::Relaxed);
    }

    pub fn ws_message(&self, bytes: u64) {
        self.ws_messages_total.fetch_add(1, Ordering::Relaxed);
        self.ws_bytes_total.fetch_add(bytes, Ordering::Relaxed);
    }
}

impl Default for EndpointMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Aggregated registry of per-endpoint metrics. Renders Prometheus exposition
/// text for every known endpoint.
pub struct GatewayMetrics {
    endpoints: RwLock<Vec<(String, Arc<EndpointMetrics>)>>,
}

impl GatewayMetrics {
    pub fn new() -> Self {
        Self {
            endpoints: RwLock::new(Vec::new()),
        }
    }

    pub fn register(&self, name: String, metrics: Arc<EndpointMetrics>) {
        let mut guard = self.endpoints.write();
        if let Some(slot) = guard.iter_mut().find(|(n, _)| n == &name) {
            slot.1 = metrics;
        } else {
            guard.push((name, metrics));
        }
    }

    pub fn unregister(&self, name: &str) {
        let mut guard = self.endpoints.write();
        guard.retain(|(n, _)| n != name);
    }

    pub fn render_prometheus(&self) -> String {
        let mut out = String::with_capacity(4096);
        out.push_str("# HELP zyron_endpoint_http_requests_total Dynamic endpoint HTTP requests.\n");
        out.push_str("# TYPE zyron_endpoint_http_requests_total counter\n");
        let guard = self.endpoints.read();
        for (name, m) in guard.iter() {
            let _ = std::fmt::Write::write_fmt(
                &mut out,
                format_args!(
                    "zyron_endpoint_http_requests_total{{endpoint=\"{}\"}} {}\n",
                    escape_label(name),
                    m.requests_total.load(Ordering::Relaxed)
                ),
            );
        }
        out.push_str("# HELP zyron_endpoint_http_errors_total Dynamic endpoint errors.\n");
        out.push_str("# TYPE zyron_endpoint_http_errors_total counter\n");
        for (name, m) in guard.iter() {
            let _ = std::fmt::Write::write_fmt(
                &mut out,
                format_args!(
                    "zyron_endpoint_http_errors_total{{endpoint=\"{}\"}} {}\n",
                    escape_label(name),
                    m.errors_total.load(Ordering::Relaxed)
                ),
            );
        }
        out.push_str("# HELP zyron_endpoint_rate_limited_total Requests blocked by rate limit.\n");
        out.push_str("# TYPE zyron_endpoint_rate_limited_total counter\n");
        for (name, m) in guard.iter() {
            let _ = std::fmt::Write::write_fmt(
                &mut out,
                format_args!(
                    "zyron_endpoint_rate_limited_total{{endpoint=\"{}\"}} {}\n",
                    escape_label(name),
                    m.rate_limited_total.load(Ordering::Relaxed)
                ),
            );
        }
        out.push_str("# HELP zyron_endpoint_circuit_open_total Circuit-breaker trips.\n");
        out.push_str("# TYPE zyron_endpoint_circuit_open_total counter\n");
        for (name, m) in guard.iter() {
            let _ = std::fmt::Write::write_fmt(
                &mut out,
                format_args!(
                    "zyron_endpoint_circuit_open_total{{endpoint=\"{}\"}} {}\n",
                    escape_label(name),
                    m.circuit_open_total.load(Ordering::Relaxed)
                ),
            );
        }
        out.push_str("# HELP zyron_endpoint_ws_active_connections Active WebSocket connections.\n");
        out.push_str("# TYPE zyron_endpoint_ws_active_connections gauge\n");
        for (name, m) in guard.iter() {
            let _ = std::fmt::Write::write_fmt(
                &mut out,
                format_args!(
                    "zyron_endpoint_ws_active_connections{{endpoint=\"{}\"}} {}\n",
                    escape_label(name),
                    m.ws_active.load(Ordering::Relaxed)
                ),
            );
        }
        out.push_str("# HELP zyron_endpoint_ws_messages_total WebSocket messages delivered.\n");
        out.push_str("# TYPE zyron_endpoint_ws_messages_total counter\n");
        for (name, m) in guard.iter() {
            let _ = std::fmt::Write::write_fmt(
                &mut out,
                format_args!(
                    "zyron_endpoint_ws_messages_total{{endpoint=\"{}\"}} {}\n",
                    escape_label(name),
                    m.ws_messages_total.load(Ordering::Relaxed)
                ),
            );
        }
        out.push_str("# HELP zyron_endpoint_ws_bytes_total WebSocket bytes delivered.\n");
        out.push_str("# TYPE zyron_endpoint_ws_bytes_total counter\n");
        for (name, m) in guard.iter() {
            let _ = std::fmt::Write::write_fmt(
                &mut out,
                format_args!(
                    "zyron_endpoint_ws_bytes_total{{endpoint=\"{}\"}} {}\n",
                    escape_label(name),
                    m.ws_bytes_total.load(Ordering::Relaxed)
                ),
            );
        }
        out
    }
}

impl Default for GatewayMetrics {
    fn default() -> Self {
        Self::new()
    }
}

fn escape_label(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_bumps_counters() {
        let m = EndpointMetrics::new();
        m.record(1_000, 200, false);
        m.record(1_000_000, 500, true);
        assert_eq!(m.requests_total.load(Ordering::Relaxed), 2);
        assert_eq!(m.errors_total.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn render_contains_registered_endpoint() {
        let gm = GatewayMetrics::new();
        let em = Arc::new(EndpointMetrics::new());
        em.inc_rate_limited();
        gm.register("my_ep".into(), em);
        let s = gm.render_prometheus();
        assert!(s.contains("zyron_endpoint_rate_limited_total{endpoint=\"my_ep\"} 1"));
    }

    #[test]
    fn unregister_drops_entry() {
        let gm = GatewayMetrics::new();
        gm.register("x".into(), Arc::new(EndpointMetrics::new()));
        gm.unregister("x");
        let s = gm.render_prometheus();
        assert!(!s.contains("endpoint=\"x\""));
    }
}

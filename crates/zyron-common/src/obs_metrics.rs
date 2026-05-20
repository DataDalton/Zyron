#![allow(non_snake_case)]
//! Lock-free labeled metric series rendered in Prometheus text format.
//!
//! The fixed scalar metrics live in the server MetricsRegistry. These series
//! carry a label set (publication, subscription, provider, or TLS direction
//! and result) and are emitted from crates below zyron-server in the
//! dependency graph, so they live here in zyron-common where every emit site
//! can reach them. Updates use scc::HashMap read_sync fast path and
//! insert_sync cold path, no Mutex or RwLock on the update path.

use std::sync::atomic::{AtomicU64, Ordering};

use scc::HashMap as SccHashMap;

/// TLS handshake direction label value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TlsDirection {
    Inbound,
    Outbound,
}

impl TlsDirection {
    pub fn as_str(&self) -> &'static str {
        match self {
            TlsDirection::Inbound => "inbound",
            TlsDirection::Outbound => "outbound",
        }
    }
}

/// Separator placed between the direction and result components of the
/// tls_handshakes_total composite key. ASCII 0x01 cannot appear in either
/// component since both are fixed static strings.
const TLS_KEY_SEP: char = '\u{1}';

/// Bucket boundaries for the reaper-pass histogram, in microseconds. Matches
/// the cadence of zyrondb_query_duration_seconds so dashboards can reuse the
/// same axis.
const REAP_BUCKETS_US: [u64; 9] = [
    1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000,
];

/// Labeled metric series shared across crates. Counters are accumulated with
/// fetch_add, gauges are overwritten with store.
#[derive(Debug)]
pub struct LabeledMetrics {
    publicationActiveSubscribers: SccHashMap<String, AtomicU64>,
    publicationBytesSentTotal: SccHashMap<String, AtomicU64>,
    publicationRetentionLagSeconds: SccHashMap<String, AtomicU64>,
    subscriptionLagLsn: SccHashMap<String, AtomicU64>,
    subscriptionLastPollTimestamp: SccHashMap<String, AtomicU64>,
    subscriptionReconnectsTotal: SccHashMap<String, AtomicU64>,
    credentialCacheHitsTotal: SccHashMap<String, AtomicU64>,
    credentialCacheMissesTotal: SccHashMap<String, AtomicU64>,
    credentialRefreshesTotal: SccHashMap<String, AtomicU64>,
    tlsHandshakesTotal: SccHashMap<String, AtomicU64>,
    tlsSessionResumptionsTotal: AtomicU64,
    subscriptionReapsTotal: SccHashMap<String, AtomicU64>,
    subscriptionReapSecondsBuckets: [AtomicU64; 10],
    subscriptionReapSecondsCount: AtomicU64,
    subscriptionReapSecondsSumUs: AtomicU64,
}

impl LabeledMetrics {
    pub fn new() -> Self {
        Self {
            publicationActiveSubscribers: SccHashMap::new(),
            publicationBytesSentTotal: SccHashMap::new(),
            publicationRetentionLagSeconds: SccHashMap::new(),
            subscriptionLagLsn: SccHashMap::new(),
            subscriptionLastPollTimestamp: SccHashMap::new(),
            subscriptionReconnectsTotal: SccHashMap::new(),
            credentialCacheHitsTotal: SccHashMap::new(),
            credentialCacheMissesTotal: SccHashMap::new(),
            credentialRefreshesTotal: SccHashMap::new(),
            tlsHandshakesTotal: SccHashMap::new(),
            tlsSessionResumptionsTotal: AtomicU64::new(0),
            subscriptionReapsTotal: SccHashMap::new(),
            subscriptionReapSecondsBuckets: Default::default(),
            subscriptionReapSecondsCount: AtomicU64::new(0),
            subscriptionReapSecondsSumUs: AtomicU64::new(0),
        }
    }

    /// Increments zyron_subscription_reaps_total{result} once per reaped
    /// subscription. result is "success" when the catalog update lands,
    /// "persist_error" when it fails.
    pub fn subscriptionReap(&self, result: &str) {
        Self::addBy(&self.subscriptionReapsTotal, result, 1);
    }

    /// Observes one full reaper pass on the zyron_subscription_reap_seconds
    /// histogram. Called exactly once per pass regardless of how many
    /// subscriptions were reaped.
    pub fn subscriptionReapPassObserved(&self, duration_us: u64) {
        self.subscriptionReapSecondsCount
            .fetch_add(1, Ordering::Relaxed);
        self.subscriptionReapSecondsSumUs
            .fetch_add(duration_us, Ordering::Relaxed);
        let mut placed = false;
        for (i, &edge) in REAP_BUCKETS_US.iter().enumerate() {
            if duration_us <= edge {
                self.subscriptionReapSecondsBuckets[i].fetch_add(1, Ordering::Relaxed);
                placed = true;
                break;
            }
        }
        if !placed {
            self.subscriptionReapSecondsBuckets[REAP_BUCKETS_US.len()]
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Returns the histogram observation count (test inspection only).
    pub fn subscriptionReapSecondsCount(&self) -> u64 {
        self.subscriptionReapSecondsCount.load(Ordering::Relaxed)
    }

    /// Returns the count of reaper outcomes with the given result label.
    pub fn subscriptionReapsTotalFor(&self, result: &str) -> u64 {
        self.subscriptionReapsTotal
            .read_sync(result, |_, v| v.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    // ----- shared map operations -----

    fn addBy(map: &SccHashMap<String, AtomicU64>, key: &str, delta: u64) {
        if map
            .read_sync(key, |_, v| {
                v.fetch_add(delta, Ordering::Relaxed);
            })
            .is_some()
        {
            return;
        }
        // Cold path: another writer may have created the key between the read
        // and the insert. insert_sync fails when the key exists, so fall back
        // to a second read that adds onto the now-present counter.
        if map
            .insert_sync(key.to_string(), AtomicU64::new(delta))
            .is_err()
        {
            let _ = map.read_sync(key, |_, v| {
                v.fetch_add(delta, Ordering::Relaxed);
            });
        }
    }

    fn setValue(map: &SccHashMap<String, AtomicU64>, key: &str, value: u64) {
        if map
            .read_sync(key, |_, v| {
                v.store(value, Ordering::Relaxed);
            })
            .is_some()
        {
            return;
        }
        if map
            .insert_sync(key.to_string(), AtomicU64::new(value))
            .is_err()
        {
            let _ = map.read_sync(key, |_, v| {
                v.store(value, Ordering::Relaxed);
            });
        }
    }

    fn subSaturating(map: &SccHashMap<String, AtomicU64>, key: &str, delta: u64) {
        if map
            .read_sync(key, |_, v| {
                let mut cur = v.load(Ordering::Relaxed);
                loop {
                    let next = cur.saturating_sub(delta);
                    match v.compare_exchange_weak(cur, next, Ordering::Relaxed, Ordering::Relaxed) {
                        Ok(_) => break,
                        Err(observed) => cur = observed,
                    }
                }
            })
            .is_some()
        {
            return;
        }
        let _ = map.insert_sync(key.to_string(), AtomicU64::new(0));
    }

    // ----- typed entry points -----

    pub fn pubSubscribersInc(&self, publication: &str) {
        Self::addBy(&self.publicationActiveSubscribers, publication, 1);
    }

    pub fn pubSubscribersDec(&self, publication: &str) {
        Self::subSaturating(&self.publicationActiveSubscribers, publication, 1);
    }

    pub fn pubBytesSent(&self, publication: &str, bytes: u64) {
        Self::addBy(&self.publicationBytesSentTotal, publication, bytes);
    }

    pub fn pubRetentionLagSet(&self, publication: &str, secs: u64) {
        Self::setValue(&self.publicationRetentionLagSeconds, publication, secs);
    }

    pub fn subLagLsnSet(&self, subscription: &str, lag: u64) {
        Self::setValue(&self.subscriptionLagLsn, subscription, lag);
    }

    pub fn subLastPollSet(&self, subscription: &str, unixSecs: u64) {
        Self::setValue(&self.subscriptionLastPollTimestamp, subscription, unixSecs);
    }

    pub fn subReconnectInc(&self, subscription: &str) {
        Self::addBy(&self.subscriptionReconnectsTotal, subscription, 1);
    }

    pub fn credCacheHit(&self, provider: &str) {
        Self::addBy(&self.credentialCacheHitsTotal, provider, 1);
    }

    pub fn credCacheMiss(&self, provider: &str) {
        Self::addBy(&self.credentialCacheMissesTotal, provider, 1);
    }

    pub fn credRefresh(&self, provider: &str) {
        Self::addBy(&self.credentialRefreshesTotal, provider, 1);
    }

    pub fn tlsHandshake(&self, direction: TlsDirection, ok: bool) {
        let result = if ok { "ok" } else { "fail" };
        let key = format!("{}{}{}", direction.as_str(), TLS_KEY_SEP, result);
        Self::addBy(&self.tlsHandshakesTotal, &key, 1);
    }

    pub fn tlsSessionResumed(&self) {
        self.tlsSessionResumptionsTotal
            .fetch_add(1, Ordering::Relaxed);
    }

    // ----- rendering -----

    fn renderSingleLabel(
        out: &mut String,
        name: &str,
        help: &str,
        type_: &str,
        label: &str,
        map: &SccHashMap<String, AtomicU64>,
    ) {
        out.push_str(&format!("# HELP {name} {help}\n"));
        out.push_str(&format!("# TYPE {name} {type_}\n"));
        map.iter_sync(|k, v| {
            let val = v.load(Ordering::Relaxed);
            out.push_str(&format!("{name}{{{label}=\"{}\"}} {val}\n", escapeLabel(k)));
            true
        });
    }

    /// Appends every labeled family to the Prometheus text buffer.
    pub fn render_prometheus(&self, out: &mut String) {
        Self::renderSingleLabel(
            out,
            "zyron_publication_active_subscribers",
            "Active subscribers per publication",
            "gauge",
            "publication",
            &self.publicationActiveSubscribers,
        );
        Self::renderSingleLabel(
            out,
            "zyron_publication_bytes_sent_total",
            "Total bytes sent per publication",
            "counter",
            "publication",
            &self.publicationBytesSentTotal,
        );
        Self::renderSingleLabel(
            out,
            "zyron_publication_retention_lag_seconds",
            "Publication CDF retention lag in seconds",
            "gauge",
            "publication",
            &self.publicationRetentionLagSeconds,
        );
        Self::renderSingleLabel(
            out,
            "zyron_subscription_lag_lsn",
            "Subscription LSN lag, publisher head minus subscriber acked",
            "gauge",
            "subscription",
            &self.subscriptionLagLsn,
        );
        Self::renderSingleLabel(
            out,
            "zyron_subscription_last_poll_timestamp",
            "Unix timestamp of the last subscription poll",
            "gauge",
            "subscription",
            &self.subscriptionLastPollTimestamp,
        );
        Self::renderSingleLabel(
            out,
            "zyron_subscription_reconnects_total",
            "Total subscription reconnects",
            "counter",
            "subscription",
            &self.subscriptionReconnectsTotal,
        );
        Self::renderSingleLabel(
            out,
            "zyron_credential_cache_hits_total",
            "Credential cache hits per provider",
            "counter",
            "provider",
            &self.credentialCacheHitsTotal,
        );
        Self::renderSingleLabel(
            out,
            "zyron_credential_cache_misses_total",
            "Credential cache misses per provider",
            "counter",
            "provider",
            &self.credentialCacheMissesTotal,
        );
        Self::renderSingleLabel(
            out,
            "zyron_credential_refreshes_total",
            "Credential proactive refreshes per provider",
            "counter",
            "provider",
            &self.credentialRefreshesTotal,
        );

        out.push_str("# HELP zyron_tls_handshakes_total TLS handshakes by direction and result\n");
        out.push_str("# TYPE zyron_tls_handshakes_total counter\n");
        self.tlsHandshakesTotal.iter_sync(|k, v| {
            let val = v.load(Ordering::Relaxed);
            let mut parts = k.split(TLS_KEY_SEP);
            let direction = parts.next().unwrap_or("");
            let result = parts.next().unwrap_or("");
            out.push_str(&format!(
                "zyron_tls_handshakes_total{{direction=\"{}\",result=\"{}\"}} {val}\n",
                escapeLabel(direction),
                escapeLabel(result)
            ));
            true
        });

        out.push_str(
            "# HELP zyron_tls_session_resumptions_total TLS sessions resumed via session tickets\n",
        );
        out.push_str("# TYPE zyron_tls_session_resumptions_total counter\n");
        out.push_str(&format!(
            "zyron_tls_session_resumptions_total {}\n",
            self.tlsSessionResumptionsTotal.load(Ordering::Relaxed)
        ));

        // Subscription reaper counter, labeled by outcome.
        Self::renderSingleLabel(
            out,
            "zyron_subscription_reaps_total",
            "Subscription reaper passes, labeled by outcome",
            "counter",
            "result",
            &self.subscriptionReapsTotal,
        );

        // Subscription reap-pass duration histogram.
        out.push_str(
            "# HELP zyron_subscription_reap_seconds Duration of one reaper pass, in seconds\n",
        );
        out.push_str("# TYPE zyron_subscription_reap_seconds histogram\n");
        let mut cumulative: u64 = 0;
        for (i, &edge) in REAP_BUCKETS_US.iter().enumerate() {
            cumulative += self.subscriptionReapSecondsBuckets[i].load(Ordering::Relaxed);
            let le_seconds = edge as f64 / 1_000_000.0;
            out.push_str(&format!(
                "zyron_subscription_reap_seconds_bucket{{le=\"{}\"}} {}\n",
                le_seconds, cumulative
            ));
        }
        cumulative +=
            self.subscriptionReapSecondsBuckets[REAP_BUCKETS_US.len()].load(Ordering::Relaxed);
        out.push_str(&format!(
            "zyron_subscription_reap_seconds_bucket{{le=\"+Inf\"}} {}\n",
            cumulative
        ));
        let sum_seconds =
            self.subscriptionReapSecondsSumUs.load(Ordering::Relaxed) as f64 / 1_000_000.0;
        out.push_str(&format!(
            "zyron_subscription_reap_seconds_sum {}\n",
            sum_seconds
        ));
        out.push_str(&format!(
            "zyron_subscription_reap_seconds_count {}\n",
            self.subscriptionReapSecondsCount.load(Ordering::Relaxed)
        ));
    }
}

impl Default for LabeledMetrics {
    fn default() -> Self {
        Self::new()
    }
}

fn escapeLabel(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counter_accumulates_gauge_overwrites() {
        let m = LabeledMetrics::new();
        m.pubBytesSent("pub1", 4096);
        m.pubBytesSent("pub1", 1000);
        m.subLagLsnSet("sub1", 9);
        m.subLagLsnSet("sub1", 7);

        let mut out = String::new();
        m.render_prometheus(&mut out);
        assert!(out.contains("zyron_publication_bytes_sent_total{publication=\"pub1\"} 5096"));
        assert!(out.contains("zyron_subscription_lag_lsn{subscription=\"sub1\"} 7"));
    }

    #[test]
    fn subscribers_gauge_saturates_at_zero() {
        let m = LabeledMetrics::new();
        m.pubSubscribersInc("p");
        m.pubSubscribersDec("p");
        m.pubSubscribersDec("p");
        let mut out = String::new();
        m.render_prometheus(&mut out);
        assert!(out.contains("zyron_publication_active_subscribers{publication=\"p\"} 0"));
    }

    #[test]
    fn label_values_are_escaped() {
        let m = LabeledMetrics::new();
        m.credCacheHit("a\"b\\c");
        let mut out = String::new();
        m.render_prometheus(&mut out);
        assert!(out.contains("zyron_credential_cache_hits_total{provider=\"a\\\"b\\\\c\"} 1"));
    }

    #[test]
    fn tls_composite_key_splits_into_two_labels() {
        let m = LabeledMetrics::new();
        m.tlsHandshake(TlsDirection::Inbound, true);
        m.tlsHandshake(TlsDirection::Outbound, false);
        m.tlsSessionResumed();
        let mut out = String::new();
        m.render_prometheus(&mut out);
        assert!(out.contains("zyron_tls_handshakes_total{direction=\"inbound\",result=\"ok\"} 1"));
        assert!(
            out.contains("zyron_tls_handshakes_total{direction=\"outbound\",result=\"fail\"} 1")
        );
        assert!(out.contains("zyron_tls_session_resumptions_total 1"));
    }
}

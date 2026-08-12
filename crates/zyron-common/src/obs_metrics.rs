#![allow(non_snake_case)]
//! Lock-free labeled metric series rendered in Prometheus text format.
//!
//! The fixed scalar metrics live in the server MetricsRegistry. These series
//! carry a label set (publication, subscription, provider, or TLS direction
//! and result) and are emitted from crates below zyron-server in the
//! dependency graph, so they live here in zyron-common where every emit site
//! can reach them. Updates use scc::HashMap read_sync fast path and
//! insert_sync cold path, no Mutex or RwLock on the update path.

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

use scc::HashMap as SccHashMap;

use crate::cluster::{ClusterDecision, ClusterMode, ClusteringSchedule};

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

/// Separator between the table and decision components of the
/// clustering_proposals_total composite key. ASCII 0x01 cannot appear in
/// a table name, and the decision component is a fixed static string.
const CLUSTER_KEY_SEP: char = '\u{1}';

/// Bucket boundaries for the speculative skip-rate delta histogram, in
/// thousandths. A delta is a change in skip rate so it spans -1 to 1, and
/// the negative buckets matter: a proposal the gate refused for reading
/// more bytes should be visible as such rather than absent.
const SKIPRATE_DELTA_BUCKETS_MILLI: [i64; 10] = [-1000, -500, -100, -10, 0, 10, 50, 100, 250, 500];

/// Per-table histogram of proposed skip-rate deltas. One instance per
/// table, held in a map, so the fixed-array pattern still applies with no
/// allocation on the observe path
#[derive(Debug, Default)]
struct DeltaHistogram {
    buckets: [AtomicU64; SKIPRATE_DELTA_BUCKETS_MILLI.len() + 1],
    count: AtomicU64,
    /// Sum in thousandths, signed because a refused proposal contributes
    /// a negative delta
    sum_milli: AtomicI64,
}

impl DeltaHistogram {
    fn observe(&self, delta: f64) {
        let milli = (delta * 1000.0).round() as i64;
        self.count.fetch_add(1, Ordering::Relaxed);
        self.sum_milli.fetch_add(milli, Ordering::Relaxed);
        for (i, &edge) in SKIPRATE_DELTA_BUCKETS_MILLI.iter().enumerate() {
            if milli <= edge {
                self.buckets[i].fetch_add(1, Ordering::Relaxed);
                return;
            }
        }
        self.buckets[SKIPRATE_DELTA_BUCKETS_MILLI.len()].fetch_add(1, Ordering::Relaxed);
    }
}

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
    clusteringFilesClusteredTotal: SccHashMap<String, AtomicU64>,
    clusteringFilesRewrittenTotal: SccHashMap<String, AtomicU64>,
    clusteringBytesRewrittenTotal: SccHashMap<String, AtomicU64>,
    clusteringProposalsTotal: SccHashMap<String, AtomicU64>,
    clusteringSkiprateDelta: SccHashMap<String, DeltaHistogram>,
    clusteringSkipRate: SccHashMap<String, AtomicU64>,
    clusteringPendingProposals: SccHashMap<String, AtomicU64>,
    clusteringWorkloadWindowSize: SccHashMap<String, AtomicU64>,
    clusteringMode: SccHashMap<String, AtomicU64>,
    clusteringSchedule: SccHashMap<String, AtomicU64>,
    clusteringLastPassSeconds: SccHashMap<String, AtomicU64>,
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
            clusteringFilesClusteredTotal: SccHashMap::new(),
            clusteringFilesRewrittenTotal: SccHashMap::new(),
            clusteringBytesRewrittenTotal: SccHashMap::new(),
            clusteringProposalsTotal: SccHashMap::new(),
            clusteringSkiprateDelta: SccHashMap::new(),
            clusteringSkipRate: SccHashMap::new(),
            clusteringPendingProposals: SccHashMap::new(),
            clusteringWorkloadWindowSize: SccHashMap::new(),
            clusteringMode: SccHashMap::new(),
            clusteringSchedule: SccHashMap::new(),
            clusteringLastPassSeconds: SccHashMap::new(),
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

    // ----- clustering -----

    /// Records one finished clustering pass. Called once per pass, on the
    /// maintenance thread, never on a query path.
    ///
    /// `files_in` is what the pass read and `files_out` what it wrote, so
    /// a rejected pass still reports the bytes it spent: work refused by
    /// the gate is real cost and hiding it would make the gate look free
    pub fn clusteringPass(
        &self,
        table: &str,
        decision: ClusterDecision,
        files_in: u64,
        files_out: u64,
        bytes_written: u64,
        skiprate_delta: f64,
        duration_us: u64,
    ) {
        let key = format!("{}{}{}", table, CLUSTER_KEY_SEP, decision.as_str());
        Self::addBy(&self.clusteringProposalsTotal, &key, 1);
        Self::addBy(&self.clusteringFilesRewrittenTotal, table, files_in);
        Self::addBy(&self.clusteringBytesRewrittenTotal, table, bytes_written);
        if decision == ClusterDecision::Accepted {
            Self::addBy(&self.clusteringFilesClusteredTotal, table, files_out);
        }
        Self::setValue(
            &self.clusteringLastPassSeconds,
            table,
            duration_us / 1_000_000,
        );
        if self
            .clusteringSkiprateDelta
            .read_sync(table, |_, h| h.observe(skiprate_delta))
            .is_none()
        {
            let fresh = DeltaHistogram::default();
            fresh.observe(skiprate_delta);
            if self
                .clusteringSkiprateDelta
                .insert_sync(table.to_string(), fresh)
                .is_err()
            {
                let _ = self
                    .clusteringSkiprateDelta
                    .read_sync(table, |_, h| h.observe(skiprate_delta));
            }
        }
    }

    /// Sets the measured byte-weighted skip rate for a table, in
    /// thousandths so the gauge stays an integer counter
    pub fn clusteringSkipRateSet(&self, table: &str, rate: f64) {
        let milli = (rate.clamp(0.0, 1.0) * 1000.0).round() as u64;
        Self::setValue(&self.clusteringSkipRate, table, milli);
    }

    pub fn clusteringPendingProposalsSet(&self, table: &str, pending: u64) {
        Self::setValue(&self.clusteringPendingProposals, table, pending);
    }

    /// Number of distinct predicate terms the observer currently holds
    /// for this table, which is how much evidence the gate has to judge on
    pub fn clusteringWorkloadWindowSet(&self, table: &str, terms: u64) {
        Self::setValue(&self.clusteringWorkloadWindowSize, table, terms);
    }

    pub fn clusteringPolicySet(
        &self,
        table: &str,
        mode: ClusterMode,
        schedule: ClusteringSchedule,
    ) {
        Self::setValue(&self.clusteringMode, table, mode.to_u8() as u64);
        Self::setValue(&self.clusteringSchedule, table, schedule.to_u8() as u64);
    }

    /// Returns the proposal count for one table and decision, so a test
    /// can assert what a pass recorded
    pub fn clusteringProposalsFor(&self, table: &str, decision: ClusterDecision) -> u64 {
        let key = format!("{}{}{}", table, CLUSTER_KEY_SEP, decision.as_str());
        self.clusteringProposalsTotal
            .read_sync(&key, |_, v| v.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    /// Returns the skip-rate delta observation count for one table
    pub fn clusteringSkiprateDeltaCount(&self, table: &str) -> u64 {
        self.clusteringSkiprateDelta
            .read_sync(table, |_, h| h.count.load(Ordering::Relaxed))
            .unwrap_or(0)
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

        self.renderClustering(out);
    }

    /// Appends the clustering families. Modes and schedules render as
    /// their persisted codes rather than as extra labels, so a table that
    /// changes mode replaces its own sample instead of leaving a stale
    /// label set behind forever
    fn renderClustering(&self, out: &mut String) {
        Self::renderSingleLabel(
            out,
            "zyron_clustering_files_clustered_total",
            "Data files an accepted clustering pass wrote, per table",
            "counter",
            "table",
            &self.clusteringFilesClusteredTotal,
        );
        Self::renderSingleLabel(
            out,
            "zyron_clustering_files_rewritten_total",
            "Data files clustering passes read, accepted or refused, per table",
            "counter",
            "table",
            &self.clusteringFilesRewrittenTotal,
        );
        Self::renderSingleLabel(
            out,
            "zyron_clustering_bytes_rewritten_total",
            "Bytes clustering passes staged, accepted or refused, per table",
            "counter",
            "table",
            &self.clusteringBytesRewrittenTotal,
        );

        out.push_str(
            "# HELP zyron_clustering_proposals_total Clustering proposals by table and decision\n",
        );
        out.push_str("# TYPE zyron_clustering_proposals_total counter\n");
        self.clusteringProposalsTotal.iter_sync(|k, v| {
            let val = v.load(Ordering::Relaxed);
            let mut parts = k.split(CLUSTER_KEY_SEP);
            let table = parts.next().unwrap_or("");
            let decision = parts.next().unwrap_or("");
            out.push_str(&format!(
                "zyron_clustering_proposals_total{{table=\"{}\",decision=\"{}\"}} {val}\n",
                escapeLabel(table),
                escapeLabel(decision)
            ));
            true
        });

        out.push_str(
            "# HELP zyron_clustering_speculative_skiprate_delta Proposed skip-rate change per pass\n",
        );
        out.push_str("# TYPE zyron_clustering_speculative_skiprate_delta histogram\n");
        self.clusteringSkiprateDelta.iter_sync(|table, hist| {
            let label = escapeLabel(table);
            let mut cumulative = 0u64;
            for (i, &edge) in SKIPRATE_DELTA_BUCKETS_MILLI.iter().enumerate() {
                cumulative += hist.buckets[i].load(Ordering::Relaxed);
                out.push_str(&format!(
                    "zyron_clustering_speculative_skiprate_delta_bucket{{table=\"{}\",le=\"{}\"}} {}\n",
                    label,
                    edge as f64 / 1000.0,
                    cumulative
                ));
            }
            cumulative += hist.buckets[SKIPRATE_DELTA_BUCKETS_MILLI.len()].load(Ordering::Relaxed);
            out.push_str(&format!(
                "zyron_clustering_speculative_skiprate_delta_bucket{{table=\"{}\",le=\"+Inf\"}} {}\n",
                label, cumulative
            ));
            out.push_str(&format!(
                "zyron_clustering_speculative_skiprate_delta_sum{{table=\"{}\"}} {}\n",
                label,
                hist.sum_milli.load(Ordering::Relaxed) as f64 / 1000.0
            ));
            out.push_str(&format!(
                "zyron_clustering_speculative_skiprate_delta_count{{table=\"{}\"}} {}\n",
                label,
                hist.count.load(Ordering::Relaxed)
            ));
            true
        });

        Self::renderSingleLabel(
            out,
            "zyron_clustering_skip_rate",
            "Measured byte-weighted skip rate per table, in thousandths",
            "gauge",
            "table",
            &self.clusteringSkipRate,
        );
        Self::renderSingleLabel(
            out,
            "zyron_clustering_pending_proposals",
            "Clustering proposals waiting for a maintenance slot, per table",
            "gauge",
            "table",
            &self.clusteringPendingProposals,
        );
        Self::renderSingleLabel(
            out,
            "zyron_clustering_workload_window_size",
            "Distinct predicate terms the observer holds, per table",
            "gauge",
            "table",
            &self.clusteringWorkloadWindowSize,
        );
        Self::renderSingleLabel(
            out,
            "zyron_clustering_mode",
            "Clustering mode per table, 0 force, 1 auto, 2 hybrid",
            "gauge",
            "table",
            &self.clusteringMode,
        );
        Self::renderSingleLabel(
            out,
            "zyron_clustering_schedule",
            "Clustering schedule per table, 0 on demand, 1 incremental, 2 continuous",
            "gauge",
            "table",
            &self.clusteringSchedule,
        );
        Self::renderSingleLabel(
            out,
            "zyron_clustering_last_pass_seconds",
            "Duration of the last clustering pass per table, in seconds",
            "gauge",
            "table",
            &self.clusteringLastPassSeconds,
        );
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
mod clustering_tests {
    use super::*;

    /// A refused pass has to be visible. Reporting only accepted passes
    /// would make the gate look free when refused work costs real IO
    #[test]
    fn test_a_refused_pass_still_reports_the_work_it_spent() {
        let m = LabeledMetrics::new();
        m.clusteringPass(
            "events",
            ClusterDecision::Accepted,
            4,
            2,
            8_192,
            0.42,
            1_500_000,
        );
        m.clusteringPass(
            "events",
            ClusterDecision::RejectedWorse,
            3,
            3,
            4_096,
            -0.10,
            2_000_000,
        );

        assert_eq!(
            m.clusteringProposalsFor("events", ClusterDecision::Accepted),
            1
        );
        assert_eq!(
            m.clusteringProposalsFor("events", ClusterDecision::RejectedWorse),
            1
        );
        assert_eq!(m.clusteringSkiprateDeltaCount("events"), 2);

        let mut out = String::new();
        m.render_prometheus(&mut out);
        // Both passes read files and staged bytes
        assert!(out.contains("zyron_clustering_files_rewritten_total{table=\"events\"} 7"));
        assert!(out.contains("zyron_clustering_bytes_rewritten_total{table=\"events\"} 12288"));
        // Only the accepted one produced live clustered files
        assert!(out.contains("zyron_clustering_files_clustered_total{table=\"events\"} 2"));
        // The negative delta lands in a negative bucket, not below zero
        assert!(out.contains(
            "zyron_clustering_speculative_skiprate_delta_bucket{table=\"events\",le=\"-0.1\"} 1"
        ));
        assert!(
            out.contains("zyron_clustering_speculative_skiprate_delta_sum{table=\"events\"} 0.32")
        );
        assert!(
            out.contains("zyron_clustering_speculative_skiprate_delta_count{table=\"events\"} 2")
        );
        // The last pass overwrites rather than accumulating
        assert!(out.contains("zyron_clustering_last_pass_seconds{table=\"events\"} 2"));
    }

    #[test]
    fn test_policy_gauges_replace_rather_than_accumulate() {
        let m = LabeledMetrics::new();
        m.clusteringPolicySet("events", ClusterMode::Auto, ClusteringSchedule::Continuous);
        m.clusteringPolicySet("events", ClusterMode::Hybrid, ClusteringSchedule::OnDemand);
        m.clusteringSkipRateSet("events", 0.875);
        m.clusteringPendingProposalsSet("events", 3);
        m.clusteringWorkloadWindowSet("events", 17);

        let mut out = String::new();
        m.render_prometheus(&mut out);
        assert!(out.contains("zyron_clustering_mode{table=\"events\"} 2"));
        assert!(out.contains("zyron_clustering_schedule{table=\"events\"} 0"));
        assert!(out.contains("zyron_clustering_skip_rate{table=\"events\"} 875"));
        assert!(out.contains("zyron_clustering_pending_proposals{table=\"events\"} 3"));
        assert!(out.contains("zyron_clustering_workload_window_size{table=\"events\"} 17"));
    }

    /// Every family must announce its type, or a scrape drops it
    #[test]
    fn test_every_clustering_family_is_declared() {
        let m = LabeledMetrics::new();
        m.clusteringPass("t", ClusterDecision::Accepted, 1, 1, 1, 0.5, 1);
        m.clusteringPolicySet("t", ClusterMode::Force, ClusteringSchedule::Incremental);
        m.clusteringSkipRateSet("t", 0.5);
        m.clusteringPendingProposalsSet("t", 0);
        m.clusteringWorkloadWindowSet("t", 0);
        let mut out = String::new();
        m.render_prometheus(&mut out);
        for family in [
            "zyron_clustering_files_clustered_total",
            "zyron_clustering_files_rewritten_total",
            "zyron_clustering_bytes_rewritten_total",
            "zyron_clustering_proposals_total",
            "zyron_clustering_speculative_skiprate_delta",
            "zyron_clustering_skip_rate",
            "zyron_clustering_pending_proposals",
            "zyron_clustering_workload_window_size",
            "zyron_clustering_mode",
            "zyron_clustering_schedule",
            "zyron_clustering_last_pass_seconds",
        ] {
            assert!(
                out.contains(&format!("# TYPE {family} ")),
                "{family} has no TYPE line"
            );
            assert!(
                out.contains(&format!("# HELP {family} ")),
                "{family} has no HELP line"
            );
        }
    }
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

//! The change stream sweeper.
//!
//! Every pass reads each stream's state off the feed counters, marks the
//! streams whose position names changes a feed no longer holds as stale
//! before a reader arrives to find out, records every stream's pending rows
//! and lag as metrics, and raises the alerts the change feed templates
//! declare through the operator's contact channels. Nothing here opens a
//! change file. Staleness is a position against a purge floor and lag is a
//! counter against a timestamp.
//!
//! An alert fires once per condition per stream and again only after the
//! condition cleared and returned, so a stream that stays behind is reported
//! when it falls behind rather than on every pass

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use zyron_cdc::change_stream::{self, AlertFiring, ChangeStreamRuntime};
use zyron_wire::connection::ServerState;

use crate::upgrade::notification::{AlertEvent, Notifier};

/// The thresholds one pass compares against
#[derive(Debug, Clone)]
pub struct SweeperConfig {
    pub interval_secs: u64,
    pub lag_rows: u64,
    pub lag_seconds: u64,
    pub retention_margin_secs: u64,
}

impl Default for SweeperConfig {
    fn default() -> Self {
        Self {
            interval_secs: 10,
            lag_rows: 1_000_000,
            lag_seconds: 3600,
            retention_margin_secs: 3600,
        }
    }
}

/// What one pass did, for the caller and the tests
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SweepOutcome {
    pub streams: usize,
    pub went_stale: usize,
    pub recovered: usize,
    pub alerts: Vec<AlertFiring>,
}

/// The conditions currently raised, so a condition that holds across passes
/// is reported once
#[derive(Default)]
pub struct Raised {
    active: parking_lot::Mutex<HashSet<(u32, &'static str)>>,
    /// Apply runs already reported, by their start instant
    apply_failures: parking_lot::Mutex<HashSet<i64>>,
}

pub async fn change_stream_sweeper_loop(
    server: Arc<ServerState>,
    shutdown: Arc<AtomicBool>,
    wake: Arc<tokio::sync::Notify>,
    config: SweeperConfig,
    notifier: Option<Arc<Notifier>>,
    labeled: Option<Arc<zyron_common::LabeledMetrics>>,
) {
    let mut ticker = tokio::time::interval(Duration::from_secs(config.interval_secs.max(1)));
    let raised = Raised::default();
    while super::tick_until_shutdown(&mut ticker, &shutdown, &wake).await {
        let outcome = sweep_once(&server, &config, &raised, labeled.as_deref()).await;
        if let Some(notifier) = notifier.as_ref() {
            dispatch(notifier, &outcome.alerts).await;
        }
    }
}

/// Runs one pass, staleness, metrics and alert evaluation
pub async fn sweep_once(
    server: &Arc<ServerState>,
    config: &SweeperConfig,
    raised: &Raised,
    labeled: Option<&zyron_common::LabeledMetrics>,
) -> SweepOutcome {
    let mut outcome = SweepOutcome::default();
    let Some(feeds) = server.cdc_registry.as_ref() else {
        return outcome;
    };
    let runtime = ChangeStreamRuntime::new(Arc::clone(feeds));
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0);

    // Staleness first, so the status each stream reports below is the one
    // the catalog now holds
    let entries = server.catalog.list_change_streams();
    outcome.streams = entries.len();
    for changed in runtime.sweep(&entries) {
        let event = if changed.stale {
            outcome.went_stale += 1;
            "ChangeStreamWentStale"
        } else {
            outcome.recovered += 1;
            "ChangeStreamRecovered"
        };
        let name = changed.name.clone();
        let reason = changed.stale_reason.clone();
        match server.catalog.update_change_stream(changed).await {
            Ok(()) => tracing::info!(
                target: "zyron::audit",
                event,
                stream = %name,
                reason = %reason,
            ),
            Err(e) => tracing::warn!(
                error = %e,
                stream = %name,
                "the sweeper could not record a change stream's staleness"
            ),
        }
    }

    let margin_micros = (config.retention_margin_secs as i64).saturating_mul(1_000_000);
    let mut now_active: HashSet<(u32, &'static str)> = HashSet::new();
    for entry in server.catalog.list_change_streams() {
        let status = runtime.status(&entry, now);
        if let Some(labeled) = labeled {
            labeled.changeStreamSet(
                &status.name,
                status.pending_rows,
                status.pending_versions,
                status.lag_seconds.max(0) as u64,
                status.stale,
            );
        }
        let mut firing =
            change_stream::evaluate_alerts(&status, config.lag_rows, config.lag_seconds as i64);
        if let Some(pressure) =
            change_stream::retention_pressure(&runtime, &entry, margin_micros, now)
        {
            firing.push(pressure);
        }
        for alert in firing {
            now_active.insert((alert.stream_id, alert.template));
            outcome.alerts.push(alert);
        }
    }

    // An apply run that failed is reported once, keyed by when it started
    for run in zyron_wire::change_stream_dispatch::apply_runs().failures() {
        if raised.apply_failures.lock().insert(run.started_at) {
            outcome.alerts.push(AlertFiring {
                template: "cdc_apply_failed",
                stream_id: 0,
                stream: run.target.clone(),
                detail: run.error.clone(),
            });
        }
    }

    if let Some(labeled) = labeled {
        for table_id in feeds.table_ids() {
            if let Some(feed) = feeds.get_feed(table_id) {
                let name = server
                    .catalog
                    .get_table_by_id(zyron_catalog::TableId(table_id))
                    .map(|t| t.name.clone())
                    .unwrap_or_else(|_| table_id.to_string());
                labeled.changeFeedSet(&name, feed.file_size_bytes(), feed.record_count());
            }
        }
    }

    // A condition already raised on the previous pass is not raised again
    // until it clears
    let mut active = raised.active.lock();
    outcome.alerts.retain(|alert| {
        alert.template == "cdc_apply_failed" || !active.contains(&(alert.stream_id, alert.template))
    });
    *active = now_active;
    if let Some(labeled) = labeled {
        for alert in &outcome.alerts {
            labeled.changeStreamAlert(alert.template);
        }
    }
    outcome
}

/// Delivers the alerts a pass raised through the contact channels
async fn dispatch(notifier: &Notifier, alerts: &[AlertFiring]) {
    let now_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    for alert in alerts {
        let event = AlertEvent {
            template: alert.template.to_string(),
            subsystem: "cdc",
            object: alert.stream.clone(),
            detail: alert.detail.clone(),
            urgent: matches!(
                alert.template,
                "cdc_stream_stale" | "cdc_stream_needs_attention" | "cdc_apply_failed"
            ),
        };
        let (_, deliveries) = notifier.emit(&event, now_secs).await;
        for delivery in deliveries.iter().filter(|d| !d.delivered) {
            tracing::warn!(
                channel = %delivery.channel,
                alert = alert.template,
                detail = %delivery.detail,
                "a change stream alert was not delivered"
            );
        }
        tracing::info!(
            target: "zyron::audit",
            event = "ChangeStreamAlert",
            template = alert.template,
            stream = %alert.stream,
            detail = %alert.detail,
        );
    }
}

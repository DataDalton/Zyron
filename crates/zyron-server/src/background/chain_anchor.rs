//! The chain anchoring pass.
//!
//! Every interval, the head of each verified table whose chain has moved
//! since its last anchor is recorded, as an anchor this node holds and as a
//! `ChainAnchored` event in the audit log. Anchoring is what makes a
//! truncation detectable: a chain with its last entries removed is still a
//! consistent chain, and only an anchor naming a head it no longer reaches
//! contradicts it.
//!
//! A head that has gone unanchored for longer than twice the interval
//! raises `chain_not_anchored`, once per table until it is anchored again.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use zyron_wire::connection::ServerState;
use zyron_wire::system_verify_views::ALERT_TEMPLATES;

use crate::upgrade::notification::{AlertEvent, Notifier};

/// What one pass compares against
#[derive(Debug, Clone)]
pub struct AnchorConfig {
    /// Seconds between passes, which is also the window the overdue alert
    /// is measured in twice over
    pub interval_secs: u64,
}

impl Default for AnchorConfig {
    fn default() -> Self {
        Self {
            interval_secs: zyron_lifecycle::verify::anchor::DEFAULT_ANCHOR_INTERVAL_SECS,
        }
    }
}

/// What one pass did, for the caller and the tests
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AnchorOutcome {
    /// Tables whose head was recorded this pass
    pub anchored: Vec<String>,
    /// Tables whose head has gone unanchored past twice the interval
    pub overdue: Vec<String>,
    /// Verifications that found a chain not intact, each reported once
    pub failed: Vec<(String, String)>,
}

/// The conditions already raised, so one that holds across passes is
/// reported once
#[derive(Default)]
pub struct Raised {
    unanchored: parking_lot::Mutex<HashSet<u32>>,
    /// Failed verifications already reported, by the instant each started
    failures: parking_lot::Mutex<HashSet<i64>>,
}

pub async fn chain_anchor_loop(
    server: Arc<ServerState>,
    shutdown: Arc<AtomicBool>,
    wake: Arc<tokio::sync::Notify>,
    config: AnchorConfig,
    notifier: Option<Arc<Notifier>>,
) {
    let mut ticker = tokio::time::interval(Duration::from_secs(config.interval_secs.max(1)));
    let raised = Raised::default();
    while super::tick_until_shutdown(&mut ticker, &shutdown, &wake).await {
        let outcome = anchor_once(&server, &config, &raised).await;
        if let Some(notifier) = notifier.as_ref() {
            dispatch(notifier, &outcome).await;
        }
    }
}

/// Anchors every verified table whose head has moved, and reports the ones
/// whose head is overdue.
pub async fn anchor_once(
    server: &Arc<ServerState>,
    config: &AnchorConfig,
    raised: &Raised,
) -> AnchorOutcome {
    let mut outcome = AnchorOutcome::default();
    for anchor in zyron_wire::verify_dispatch::anchor_due_tables(server).await {
        outcome.anchored.push(anchor.table_name);
    }
    let overdue = zyron_wire::verify_dispatch::unanchored_tables(server, config.interval_secs);
    let now_overdue: HashSet<u32> = overdue.iter().map(|(id, _)| *id).collect();
    {
        let mut active = raised.unanchored.lock();
        for (table_id, name) in &overdue {
            if active.insert(*table_id) {
                outcome.overdue.push(name.clone());
            }
        }
        active.retain(|table_id| now_overdue.contains(table_id));
    }
    // A verification that found a chain not intact is news whoever ran it
    // already has. Reporting it here as well is what reaches the operator's
    // contact channels, once per run rather than once per pass
    if let Some(registry) = server.chain_registry.as_ref() {
        let mut reported = raised.failures.lock();
        for run in registry.runs().all() {
            if run.intact {
                continue;
            }
            if reported.insert(run.started_at) {
                outcome
                    .failed
                    .push((run.table_name.clone(), run.finding.clone()));
            }
        }
    }
    outcome
}

/// Delivers what a pass raised through the contact channels
async fn dispatch(notifier: &Notifier, outcome: &AnchorOutcome) {
    let now_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    for (table, detail) in &outcome.failed {
        report_failed_verification(notifier, table, detail, now_secs).await;
    }
    for table in &outcome.overdue {
        let event = AlertEvent {
            template: "chain_not_anchored".to_string(),
            subsystem: "verify",
            object: table.clone(),
            detail: format!(
                "the commit chain of {table} has moved since its last anchor and has not been \
                 anchored within twice the interval, so a truncation of it would not be detected"
            ),
            urgent: true,
        };
        let (_, deliveries) = notifier.emit(&event, now_secs).await;
        for delivery in deliveries.iter().filter(|d| !d.delivered) {
            tracing::warn!(
                target: "zyron::verify",
                channel = %delivery.channel,
                table = %table,
                detail = %delivery.detail,
                "an unanchored chain alert was not delivered"
            );
        }
    }
}

/// Raises `verify_failed` for a verification that found a chain not intact.
///
/// Called by whoever ran the verification rather than by a pass of its own,
/// because a failed verification is news the moment it is known
pub async fn report_failed_verification(
    notifier: &Notifier,
    table: &str,
    detail: &str,
    now_secs: u64,
) {
    let event = AlertEvent {
        template: "verify_failed".to_string(),
        subsystem: "verify",
        object: table.to_string(),
        detail: detail.to_string(),
        urgent: true,
    };
    let (_, deliveries) = notifier.emit(&event, now_secs).await;
    for delivery in deliveries.iter().filter(|d| !d.delivered) {
        tracing::warn!(
            target: "zyron::verify",
            channel = %delivery.channel,
            table,
            detail = %delivery.detail,
            "a failed verification alert was not delivered"
        );
    }
}

/// Every alert this pass and the verification path raise, by name
pub fn alert_names() -> Vec<&'static str> {
    ALERT_TEMPLATES.iter().map(|t| t.name).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_the_alerts_this_worker_raises_are_the_ones_declared() {
        assert_eq!(alert_names(), vec!["verify_failed", "chain_not_anchored"]);
    }

    #[test]
    fn test_the_interval_defaults_to_the_one_the_anchor_store_documents() {
        assert_eq!(
            AnchorConfig::default().interval_secs,
            zyron_lifecycle::verify::anchor::DEFAULT_ANCHOR_INTERVAL_SECS
        );
    }

    /// A condition that holds across passes is reported once, and reported
    /// again only after it cleared and came back
    #[test]
    fn test_a_condition_already_raised_is_not_raised_again() {
        let raised = Raised::default();
        let mut active = raised.unanchored.lock();
        assert!(active.insert(7));
        assert!(!active.insert(7));
        active.retain(|id| *id != 7);
        assert!(active.insert(7));
    }
}

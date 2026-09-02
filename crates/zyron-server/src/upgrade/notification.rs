//! Upgrade notification.
//!
//! Every step of an upgrade is written to the audit hash chain and delivered
//! to whatever contact channels the operator configured. The audit entry is
//! the record that cannot be edited; the notification is the copy that
//! reaches a person. A channel that fails delivery never fails the upgrade,
//! it is recorded as undelivered so the gap is visible rather than silent

use std::sync::Arc;

use zyron_common::format::{UpgradeOutcome, UpgradePhase};

/// What happened, in the words a person reads
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UpgradeEvent {
    /// A release was found on the channel
    PendingDetected {
        from_version: String,
        to_version: String,
        gate_summary: String,
    },
    /// The sequence started
    Started {
        from_version: String,
        to_version: String,
    },
    /// One node finished
    NodeCompleted {
        node_id: String,
        to_version: String,
        nodes_remaining: u32,
    },
    /// A node was put back and the sequence stopped
    RolledBack { node_id: String, reason: String },
    /// An operator or a health check paused the sequence
    Paused { reason: String },
    /// The gate refused
    Blocked { reason: String },
    /// The sequence finished
    Completed {
        to_version: String,
        outcome: UpgradeOutcome,
        detail: String,
    },
}

impl UpgradeEvent {
    /// The subject line a channel carries
    pub fn subject(&self) -> String {
        match self {
            UpgradeEvent::PendingDetected { to_version, .. } => {
                format!("Zyron upgrade to {to_version} is available")
            }
            UpgradeEvent::Started { to_version, .. } => {
                format!("Zyron upgrade to {to_version} started")
            }
            UpgradeEvent::NodeCompleted {
                node_id,
                to_version,
                ..
            } => format!("Zyron node {node_id} is on {to_version}"),
            UpgradeEvent::RolledBack { node_id, .. } => {
                format!("Zyron node {node_id} was rolled back, upgrade paused")
            }
            UpgradeEvent::Paused { .. } => "Zyron upgrade paused".to_string(),
            UpgradeEvent::Blocked { .. } => "Zyron upgrade blocked".to_string(),
            UpgradeEvent::Completed { to_version, .. } => {
                format!("Zyron upgrade to {to_version} finished")
            }
        }
    }

    /// The body a channel carries
    pub fn body(&self) -> String {
        match self {
            UpgradeEvent::PendingDetected {
                from_version,
                to_version,
                gate_summary,
            } => format!(
                "A release is available on this cluster's channel.\n\
                 Current version {from_version}, target {to_version}.\n\
                 Compatibility gate: {gate_summary}"
            ),
            UpgradeEvent::Started {
                from_version,
                to_version,
            } => format!(
                "The rolling upgrade from {from_version} to {to_version} has started. \
                 Nodes drain and restart one at a time, the leader last."
            ),
            UpgradeEvent::NodeCompleted {
                node_id,
                to_version,
                nodes_remaining,
            } => format!(
                "Node {node_id} is healthy on {to_version}. {nodes_remaining} node(s) to go."
            ),
            UpgradeEvent::RolledBack { node_id, reason } => format!(
                "Node {node_id} did not reach its health baseline after restarting and was \
                 returned to the previous binary. The upgrade is paused for review.\n\
                 Reason: {reason}"
            ),
            UpgradeEvent::Paused { reason } => {
                format!("The upgrade is paused.\nReason: {reason}")
            }
            UpgradeEvent::Blocked { reason } => {
                format!("The compatibility gate refused this upgrade.\nReason: {reason}")
            }
            UpgradeEvent::Completed {
                to_version,
                outcome,
                detail,
            } => format!("The upgrade to {to_version} ended {outcome}.\n{detail}"),
        }
    }

    /// The audit event type this maps to, so the chain records what class of
    /// thing happened rather than only its text
    pub const fn audit_event_type(&self) -> u8 {
        // Reuses the compliance log's event type space, where the upgrade
        // substrate holds 20 through 26
        match self {
            UpgradeEvent::PendingDetected { .. } => 20,
            UpgradeEvent::Started { .. } => 21,
            UpgradeEvent::NodeCompleted { .. } => 22,
            UpgradeEvent::RolledBack { .. } => 23,
            UpgradeEvent::Paused { .. } => 24,
            UpgradeEvent::Blocked { .. } => 25,
            UpgradeEvent::Completed { .. } => 26,
        }
    }

    /// The phase this event puts the sequence in
    pub const fn phase(&self) -> UpgradePhase {
        match self {
            UpgradeEvent::PendingDetected { .. } => UpgradePhase::Detected,
            UpgradeEvent::Started { .. } => UpgradePhase::Rolling,
            UpgradeEvent::NodeCompleted { .. } => UpgradePhase::Rolling,
            UpgradeEvent::RolledBack { .. } => UpgradePhase::RollingBack,
            UpgradeEvent::Paused { .. } => UpgradePhase::Paused,
            UpgradeEvent::Blocked { .. } => UpgradePhase::Blocked,
            UpgradeEvent::Completed { .. } => UpgradePhase::Completed,
        }
    }
}

/// Where a notification goes
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContactChannel {
    Email { address: String },
    Webhook { url: String },
    Slack { webhook_url: String },
    PagerDuty { routing_key: String },
}

impl ContactChannel {
    pub fn describe(&self) -> String {
        match self {
            ContactChannel::Email { address } => format!("email {address}"),
            ContactChannel::Webhook { url } => format!("webhook {url}"),
            ContactChannel::Slack { .. } => "slack".to_string(),
            ContactChannel::PagerDuty { .. } => "pagerduty".to_string(),
        }
    }
}

/// One delivery attempt's outcome
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Delivery {
    pub channel: String,
    pub subject: String,
    pub delivered: bool,
    pub detail: String,
    pub at_secs: u64,
}

/// Delivers a notification to one channel
#[async_trait::async_trait]
pub trait NotificationSink: Send + Sync {
    async fn deliver(&self, channel: &ContactChannel, event: &UpgradeEvent) -> Delivery;
    fn describe(&self) -> String;
}

/// Sends webhook, Slack, and PagerDuty notifications over HTTPS, and records
/// email as undelivered because this node has no mail transport
pub struct HttpNotificationSink {
    client: reqwest::Client,
}

impl HttpNotificationSink {
    pub fn new(timeout_secs: u64) -> std::result::Result<Self, String> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(timeout_secs))
            .build()
            .map_err(|e| format!("notification client, {e}"))?;
        Ok(Self { client })
    }

    async fn post(&self, url: &str, event: &UpgradeEvent, channel: &str, now: u64) -> Delivery {
        let payload = serde_json::json!({
            "subject": event.subject(),
            "body": event.body(),
            "phase": event.phase().label(),
        });
        match self.client.post(url).json(&payload).send().await {
            Ok(response) if response.status().is_success() => Delivery {
                channel: channel.to_string(),
                subject: event.subject(),
                delivered: true,
                detail: format!("answered {}", response.status()),
                at_secs: now,
            },
            Ok(response) => Delivery {
                channel: channel.to_string(),
                subject: event.subject(),
                delivered: false,
                detail: format!("answered {}", response.status()),
                at_secs: now,
            },
            Err(e) => Delivery {
                channel: channel.to_string(),
                subject: event.subject(),
                delivered: false,
                detail: e.to_string(),
                at_secs: now,
            },
        }
    }
}

#[async_trait::async_trait]
impl NotificationSink for HttpNotificationSink {
    async fn deliver(&self, channel: &ContactChannel, event: &UpgradeEvent) -> Delivery {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        match channel {
            ContactChannel::Webhook { url } => {
                self.post(url, event, &channel.describe(), now).await
            }
            ContactChannel::Slack { webhook_url } => {
                self.post(webhook_url, event, &channel.describe(), now)
                    .await
            }
            ContactChannel::PagerDuty { routing_key } => {
                let url =
                    format!("https://events.pagerduty.com/v2/enqueue?routing_key={routing_key}");
                self.post(&url, event, &channel.describe(), now).await
            }
            ContactChannel::Email { address } => Delivery {
                channel: channel.describe(),
                subject: event.subject(),
                delivered: false,
                detail: format!(
                    "this node has no mail transport, so {address} was not written to. \
                     Configure a webhook channel to receive upgrade notifications"
                ),
                at_secs: now,
            },
        }
    }

    fn describe(&self) -> String {
        "https notification sink".to_string()
    }
}

/// Records deliveries without sending anything, which is what a dry run and
/// the tests use
#[derive(Debug, Default)]
pub struct RecordingSink {
    pub deliveries: parking_lot::Mutex<Vec<Delivery>>,
}

impl RecordingSink {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn recorded(&self) -> Vec<Delivery> {
        self.deliveries.lock().clone()
    }
}

#[async_trait::async_trait]
impl NotificationSink for RecordingSink {
    async fn deliver(&self, channel: &ContactChannel, event: &UpgradeEvent) -> Delivery {
        let delivery = Delivery {
            channel: channel.describe(),
            subject: event.subject(),
            delivered: true,
            detail: event.body(),
            at_secs: 0,
        };
        self.deliveries.lock().push(delivery.clone());
        delivery
    }

    fn describe(&self) -> String {
        "recording sink".to_string()
    }
}

/// Fans one event out to every configured channel and audits it.
///
/// The audit entry is written whether or not any channel took the message,
/// so an upgrade step is on the record even when nobody was reachable
pub struct Notifier {
    channels: Vec<ContactChannel>,
    sink: Arc<dyn NotificationSink>,
    chain: Arc<zyron_lifecycle::audit_chain::AuditChain>,
}

impl Notifier {
    pub fn new(channels: Vec<ContactChannel>, sink: Arc<dyn NotificationSink>) -> Self {
        Self {
            channels,
            sink,
            chain: Arc::new(zyron_lifecycle::audit_chain::AuditChain::new()),
        }
    }

    /// The chain entry each notification produced, in order
    pub fn chain(&self) -> Arc<zyron_lifecycle::audit_chain::AuditChain> {
        Arc::clone(&self.chain)
    }

    /// Audits an event and delivers it
    pub async fn emit(
        &self,
        event: &UpgradeEvent,
        now_secs: u64,
    ) -> (zyron_catalog::schema::ComplianceLogEntry, Vec<Delivery>) {
        let entry = self.chain.next_entry(
            event.audit_event_type(),
            "upgrade".to_string(),
            0,
            now_secs as i64,
            format!("{} :: {}", event.subject(), event.body()),
        );
        let mut deliveries = Vec::with_capacity(self.channels.len());
        for channel in &self.channels {
            deliveries.push(self.sink.deliver(channel, event).await);
        }
        (entry, deliveries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyron_lifecycle::audit_chain::AuditChain;

    fn channels() -> Vec<ContactChannel> {
        vec![
            ContactChannel::Email {
                address: "ops@example.com".to_string(),
            },
            ContactChannel::Webhook {
                url: "https://example/hook".to_string(),
            },
        ]
    }

    #[tokio::test]
    async fn test_every_channel_gets_the_event_and_the_chain_records_it() {
        let sink = RecordingSink::new();
        let notifier = Notifier::new(channels(), sink.clone());
        let event = UpgradeEvent::Started {
            from_version: "0.11.0".to_string(),
            to_version: "0.12.0".to_string(),
        };
        let (entry, deliveries) = notifier.emit(&event, 1_000).await;
        assert_eq!(deliveries.len(), 2);
        assert!(deliveries.iter().all(|d| d.delivered));
        assert_eq!(entry.event_type, 21);
        assert!(entry.detail.contains("upgrade from 0.11.0 to 0.12.0"));
        assert_eq!(sink.recorded().len(), 2);
    }

    #[tokio::test]
    async fn test_the_chain_stays_intact_across_a_whole_sequence() {
        let sink = RecordingSink::new();
        let notifier = Notifier::new(channels(), sink);
        let events = vec![
            UpgradeEvent::PendingDetected {
                from_version: "0.11.0".to_string(),
                to_version: "0.12.0".to_string(),
                gate_summary: "0 blockers".to_string(),
            },
            UpgradeEvent::Started {
                from_version: "0.11.0".to_string(),
                to_version: "0.12.0".to_string(),
            },
            UpgradeEvent::NodeCompleted {
                node_id: "node-2".to_string(),
                to_version: "0.12.0".to_string(),
                nodes_remaining: 2,
            },
            UpgradeEvent::Completed {
                to_version: "0.12.0".to_string(),
                outcome: UpgradeOutcome::Completed,
                detail: "3 nodes".to_string(),
            },
        ];
        let mut entries = Vec::new();
        for (index, event) in events.iter().enumerate() {
            let (entry, _) = notifier.emit(event, 1_000 + index as u64).await;
            entries.push(entry);
        }
        let (verified, intact) = AuditChain::verify(&entries);
        assert_eq!(verified, 4);
        assert!(intact, "every upgrade step is on an unbroken chain");
    }

    #[tokio::test]
    async fn test_a_tampered_entry_breaks_the_chain() {
        let sink = RecordingSink::new();
        let notifier = Notifier::new(channels(), sink);
        let (first, _) = notifier
            .emit(
                &UpgradeEvent::Started {
                    from_version: "0.11.0".to_string(),
                    to_version: "0.12.0".to_string(),
                },
                0,
            )
            .await;
        let (second, _) = notifier
            .emit(
                &UpgradeEvent::Paused {
                    reason: "operator".to_string(),
                },
                1,
            )
            .await;
        let mut tampered = vec![first, second];
        tampered[0].detail = "something else".to_string();
        let (_, intact) = AuditChain::verify(&tampered);
        assert!(!intact);
    }

    #[test]
    fn test_every_event_has_a_distinct_audit_type_and_a_readable_subject() {
        let events = vec![
            UpgradeEvent::PendingDetected {
                from_version: "a".into(),
                to_version: "b".into(),
                gate_summary: "c".into(),
            },
            UpgradeEvent::Started {
                from_version: "a".into(),
                to_version: "b".into(),
            },
            UpgradeEvent::NodeCompleted {
                node_id: "n".into(),
                to_version: "b".into(),
                nodes_remaining: 1,
            },
            UpgradeEvent::RolledBack {
                node_id: "n".into(),
                reason: "slow".into(),
            },
            UpgradeEvent::Paused {
                reason: "operator".into(),
            },
            UpgradeEvent::Blocked {
                reason: "unsafe rewrite".into(),
            },
            UpgradeEvent::Completed {
                to_version: "b".into(),
                outcome: UpgradeOutcome::Completed,
                detail: "done".into(),
            },
        ];
        let mut types: Vec<u8> = events.iter().map(|e| e.audit_event_type()).collect();
        types.sort_unstable();
        types.dedup();
        assert_eq!(types.len(), events.len(), "types are distinct");
        for event in &events {
            assert!(!event.subject().is_empty());
            assert!(!event.body().is_empty());
        }
    }

    #[tokio::test]
    async fn test_an_email_channel_reports_that_it_could_not_deliver() {
        let sink = HttpNotificationSink::new(1).expect("builds");
        let delivery = sink
            .deliver(
                &ContactChannel::Email {
                    address: "ops@example.com".to_string(),
                },
                &UpgradeEvent::Paused {
                    reason: "operator".to_string(),
                },
            )
            .await;
        assert!(!delivery.delivered);
        assert!(
            delivery.detail.contains("no mail transport"),
            "{}",
            delivery.detail
        );
    }
}

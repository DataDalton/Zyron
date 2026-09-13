//! Notification through the operator's contact channels.
//!
//! Every step of an upgrade, and every alert a subsystem raises, is written
//! to the audit hash chain and delivered to whatever contact channels the
//! operator configured. The audit entry is the record that cannot be edited,
//! and the notification is the copy that reaches a person. A channel that fails
//! delivery never fails the work that raised the event, it is recorded as
//! undelivered so the gap is visible rather than silent.
//!
//! The channels and the sink carry any [`NotificationEvent`]. The upgrade
//! events are one implementation and the alert templates other subsystems
//! declare are another, so an alert reaches the same webhook, Slack or
//! Discord channel an upgrade step does

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

    /// The color a Discord embed carries, as the integer Discord reads.
    /// Blue is informational, green is progress that landed, orange asks
    /// for attention, red is an alarm
    pub const fn embed_color(&self) -> u32 {
        match self {
            UpgradeEvent::PendingDetected { .. } => 0x3498DB,
            UpgradeEvent::Started { .. } => 0x3498DB,
            UpgradeEvent::NodeCompleted { .. } => 0x2ECC71,
            UpgradeEvent::RolledBack { .. } => 0xE74C3C,
            UpgradeEvent::Paused { .. } => 0xF39C12,
            UpgradeEvent::Blocked { .. } => 0xE74C3C,
            UpgradeEvent::Completed { .. } => 0x2ECC71,
        }
    }

    /// The structured details an embed carries beside its title and body,
    /// one labeled pair per detail this event holds. An event holding none
    /// of them yields an empty list
    pub fn detail_fields(&self) -> Vec<(&'static str, String)> {
        match self {
            UpgradeEvent::PendingDetected {
                from_version,
                to_version,
                gate_summary,
            } => vec![
                ("From version", from_version.clone()),
                ("To version", to_version.clone()),
                ("Compatibility gate", gate_summary.clone()),
            ],
            UpgradeEvent::Started {
                from_version,
                to_version,
            } => vec![
                ("From version", from_version.clone()),
                ("To version", to_version.clone()),
            ],
            UpgradeEvent::NodeCompleted {
                node_id,
                to_version,
                ..
            } => vec![
                ("Node", node_id.clone()),
                ("To version", to_version.clone()),
            ],
            UpgradeEvent::RolledBack { node_id, .. } => vec![("Node", node_id.clone())],
            UpgradeEvent::Paused { .. } => Vec::new(),
            UpgradeEvent::Blocked { .. } => Vec::new(),
            UpgradeEvent::Completed {
                to_version,
                outcome,
                ..
            } => vec![
                ("To version", to_version.clone()),
                ("Outcome", outcome.to_string()),
            ],
        }
    }
}

/// Anything a contact channel carries, what to say, how to classify it
/// and how to record it
pub trait NotificationEvent: Send + Sync {
    /// The subject line a channel carries
    fn subject(&self) -> String;
    /// The body a channel carries
    fn body(&self) -> String;
    /// The word a structured payload files the event under, an upgrade
    /// phase or an alert template name
    fn category(&self) -> String;
    /// The subsystem the event came from, which the audit chain records
    /// and a Discord footer names
    fn source(&self) -> &'static str;
    /// The audit event type the chain records
    fn audit_event_type(&self) -> u8;
    /// The color a Discord embed carries
    fn embed_color(&self) -> u32;
    /// Labeled details beside the title and body
    fn detail_fields(&self) -> Vec<(&'static str, String)>;
}

impl NotificationEvent for UpgradeEvent {
    fn subject(&self) -> String {
        UpgradeEvent::subject(self)
    }

    fn body(&self) -> String {
        UpgradeEvent::body(self)
    }

    fn category(&self) -> String {
        self.phase().label().to_string()
    }

    fn source(&self) -> &'static str {
        "upgrade"
    }

    fn audit_event_type(&self) -> u8 {
        UpgradeEvent::audit_event_type(self)
    }

    fn embed_color(&self) -> u32 {
        UpgradeEvent::embed_color(self)
    }

    fn detail_fields(&self) -> Vec<(&'static str, String)> {
        UpgradeEvent::detail_fields(self)
    }
}

/// An alert a subsystem raised from one of the templates it declares
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlertEvent {
    /// The template name, such as `cdc_stream_lag`
    pub template: String,
    /// The subsystem that declares the template
    pub subsystem: &'static str,
    /// The object the alert is about
    pub object: String,
    /// What was measured
    pub detail: String,
    /// True for a condition that stops work until someone acts, which is
    /// what turns the embed red rather than orange
    pub urgent: bool,
}

/// The compliance log event type every alert notification records under
pub const ALERT_AUDIT_EVENT_TYPE: u8 = 27;

impl NotificationEvent for AlertEvent {
    fn subject(&self) -> String {
        format!("Zyron alert {} on {}", self.template, self.object)
    }

    fn body(&self) -> String {
        self.detail.clone()
    }

    fn category(&self) -> String {
        self.template.clone()
    }

    fn source(&self) -> &'static str {
        self.subsystem
    }

    fn audit_event_type(&self) -> u8 {
        ALERT_AUDIT_EVENT_TYPE
    }

    fn embed_color(&self) -> u32 {
        if self.urgent { 0xE74C3C } else { 0xF39C12 }
    }

    fn detail_fields(&self) -> Vec<(&'static str, String)> {
        vec![
            ("Template", self.template.clone()),
            ("Object", self.object.clone()),
        ]
    }
}

/// Where a notification goes
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContactChannel {
    Email { address: String },
    Webhook { url: String },
    Slack { webhook_url: String },
    Discord { webhook_url: String },
}

impl ContactChannel {
    /// Builds a Discord channel from a webhook address, refusing an address
    /// outside the Discord webhook shape. The variant holds a plain field,
    /// so this is the path that checks an address before it becomes a
    /// channel
    pub fn discord(webhook_url: impl Into<String>) -> std::result::Result<ContactChannel, String> {
        let webhook_url = webhook_url.into();
        if !discord_webhook_url_is_wellformed(&webhook_url) {
            return Err(format!(
                "`{webhook_url}` is not a Discord webhook address. The shape is \
                 https://discord.com/api/webhooks/<id>/<token>, optionally on the canary. or \
                 ptb. host or the discordapp.com domain, where <id> is digits and <token> is \
                 letters, digits, underscores, and hyphens"
            ));
        }
        Ok(ContactChannel::Discord { webhook_url })
    }

    pub fn describe(&self) -> String {
        match self {
            ContactChannel::Email { address } => format!("email {address}"),
            ContactChannel::Webhook { url } => format!("webhook {url}"),
            ContactChannel::Slack { .. } => "slack".to_string(),
            ContactChannel::Discord { .. } => "discord".to_string(),
        }
    }
}

/// Whether an address is a Discord webhook, matching
/// `^https://(canary\.|ptb\.)?discord(app)?\.com/api/webhooks/\d+/[A-Za-z0-9_-]+$`
/// by walking the address once
pub fn discord_webhook_url_is_wellformed(url: &str) -> bool {
    let Some(rest) = url.strip_prefix("https://") else {
        return false;
    };
    let rest = rest
        .strip_prefix("canary.")
        .or_else(|| rest.strip_prefix("ptb."))
        .unwrap_or(rest);
    let Some(rest) = rest
        .strip_prefix("discordapp.com/api/webhooks/")
        .or_else(|| rest.strip_prefix("discord.com/api/webhooks/"))
    else {
        return false;
    };
    // The token runs to the end of the address, so a second separator
    // leaves a character the token charset refuses and the whole address
    // fails
    let Some((id, token)) = rest.split_once('/') else {
        return false;
    };
    !id.is_empty()
        && id.bytes().all(|b| b.is_ascii_digit())
        && !token.is_empty()
        && token
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
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
    async fn deliver(&self, channel: &ContactChannel, event: &dyn NotificationEvent) -> Delivery;
    fn describe(&self) -> String;
}

/// The footer a Discord embed carries, so an event is attributable in a
/// channel that carries more than one system's messages
fn discord_footer(event: &dyn NotificationEvent) -> String {
    match event.source() {
        "upgrade" => "Zyron auto-upgrade".to_string(),
        source => format!("Zyron {source}"),
    }
}

/// The longest `retry-after` a Discord delivery waits out. A channel asking
/// for longer is reported undelivered, because one channel's rate limit
/// never holds an upgrade step
const DISCORD_MAX_RETRY_AFTER_SECS: u64 = 60;

/// Sends webhook, Slack, and Discord notifications over HTTPS, and records
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

    async fn post(
        &self,
        url: &str,
        event: &dyn NotificationEvent,
        channel: &str,
        now: u64,
    ) -> Delivery {
        let payload = serde_json::json!({
            "subject": event.subject(),
            "body": event.body(),
            "phase": event.category(),
            "source": event.source(),
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

    /// Posts one embed to a Discord webhook, waiting out a rate limit once.
    ///
    /// A 429 is the only answer that is tried again, and only when the
    /// channel names an interval this sink is willing to hold for. Every
    /// other answer is reported as it came back
    async fn post_discord(
        &self,
        url: &str,
        event: &dyn NotificationEvent,
        channel: &str,
        now: u64,
    ) -> Delivery {
        let payload = discord_payload(event, now);
        let response = match self.client.post(url).json(&payload).send().await {
            Ok(response) => response,
            Err(e) => return settled(channel, event, false, e.to_string(), now),
        };
        let status = response.status();
        if status != reqwest::StatusCode::TOO_MANY_REQUESTS {
            return settled(
                channel,
                event,
                status.is_success(),
                format!("answered {status}"),
                now,
            );
        }

        let Some(retry_after) = retry_after_secs(response.headers()) else {
            return settled(
                channel,
                event,
                false,
                format!(
                    "answered {status} without a usable retry-after header, so there is no \
                     interval to wait out and the event went undelivered"
                ),
                now,
            );
        };
        if retry_after > DISCORD_MAX_RETRY_AFTER_SECS {
            return settled(
                channel,
                event,
                false,
                format!(
                    "answered {status} asking for {retry_after} seconds, past the \
                     {DISCORD_MAX_RETRY_AFTER_SECS} second cap this sink waits, so the event \
                     went undelivered rather than holding the upgrade step"
                ),
                now,
            );
        }

        tokio::time::sleep(std::time::Duration::from_secs(retry_after)).await;
        match self.client.post(url).json(&payload).send().await {
            Ok(response) => {
                let retried = response.status();
                settled(
                    channel,
                    event,
                    retried.is_success(),
                    format!(
                        "answered {status}, waited {retry_after}s, second attempt answered \
                         {retried}"
                    ),
                    now,
                )
            }
            Err(e) => settled(
                channel,
                event,
                false,
                format!("answered {status}, waited {retry_after}s, second attempt failed, {e}"),
                now,
            ),
        }
    }
}

/// One delivery outcome in the shape every caller reports it in
fn settled(
    channel: &str,
    event: &dyn NotificationEvent,
    delivered: bool,
    detail: String,
    now: u64,
) -> Delivery {
    Delivery {
        channel: channel.to_string(),
        subject: event.subject(),
        delivered,
        detail,
        at_secs: now,
    }
}

/// The whole-second interval a `retry-after` header asks for. Discord sends
/// this as seconds and may send a fraction, which rounds up so the wait is
/// never shorter than what was asked for
fn retry_after_secs(headers: &reqwest::header::HeaderMap) -> Option<u64> {
    let raw = headers.get(reqwest::header::RETRY_AFTER)?.to_str().ok()?;
    let raw = raw.trim();
    if let Ok(secs) = raw.parse::<u64>() {
        return Some(secs);
    }
    let seconds = raw.parse::<f64>().ok()?;
    if !seconds.is_finite() || seconds < 0.0 {
        return None;
    }
    Some(seconds.ceil() as u64)
}

/// The embed body a Discord webhook takes for one event
pub fn discord_payload(event: &dyn NotificationEvent, now: u64) -> serde_json::Value {
    let fields: Vec<serde_json::Value> = event
        .detail_fields()
        .into_iter()
        .map(|(name, value)| serde_json::json!({ "name": name, "value": value, "inline": true }))
        .collect();
    serde_json::json!({
        "embeds": [{
            "title": event.subject(),
            "description": event.body(),
            "color": event.embed_color(),
            "fields": fields,
            "footer": { "text": discord_footer(event) },
            "timestamp": iso8601_utc(now),
        }]
    })
}

/// Formats Unix seconds as an ISO-8601 UTC instant, `YYYY-MM-DDTHH:MM:SSZ`
fn iso8601_utc(unix_secs: u64) -> String {
    let secs = unix_secs as i64;
    let (year, month, day) = civil_from_days(secs.div_euclid(86_400));
    let second_of_day = secs.rem_euclid(86_400);
    let hour = second_of_day / 3_600;
    let minute = (second_of_day % 3_600) / 60;
    let second = second_of_day % 60;
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

/// Converts a day count since the Unix epoch to a civil year, month, and
/// day using Howard Hinnant's algorithm
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let day_of_era = z - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * day_of_year + 2) / 153;
    let day = (day_of_year - (153 * shifted_month + 2) / 5 + 1) as u32;
    let month = if shifted_month < 10 {
        shifted_month + 3
    } else {
        shifted_month - 9
    } as u32;
    (if month <= 2 { year + 1 } else { year }, month, day)
}

#[async_trait::async_trait]
impl NotificationSink for HttpNotificationSink {
    async fn deliver(&self, channel: &ContactChannel, event: &dyn NotificationEvent) -> Delivery {
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
            ContactChannel::Discord { webhook_url } => {
                self.post_discord(webhook_url, event, &channel.describe(), now)
                    .await
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
    async fn deliver(&self, channel: &ContactChannel, event: &dyn NotificationEvent) -> Delivery {
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
        event: &dyn NotificationEvent,
        now_secs: u64,
    ) -> (zyron_catalog::schema::ComplianceLogEntry, Vec<Delivery>) {
        let entry = self.chain.next_entry(
            event.audit_event_type(),
            event.source().to_string(),
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

    /// One channel of every kind the sink delivers to, so an assertion over
    /// this list covers each of them
    fn channels() -> Vec<ContactChannel> {
        vec![
            ContactChannel::Email {
                address: "ops@example.com".to_string(),
            },
            ContactChannel::Webhook {
                url: "https://example/hook".to_string(),
            },
            ContactChannel::Discord {
                webhook_url: "https://discord.com/api/webhooks/1/abc".to_string(),
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
        assert_eq!(deliveries.len(), 3);
        assert!(deliveries.iter().all(|d| d.delivered));
        let reached: Vec<String> = deliveries.iter().map(|d| d.channel.clone()).collect();
        assert!(
            reached.iter().any(|c| c.starts_with("email")),
            "{reached:?}"
        );
        assert!(
            reached.iter().any(|c| c.starts_with("webhook")),
            "{reached:?}"
        );
        assert!(reached.iter().any(|c| c == "discord"), "{reached:?}");
        assert_eq!(entry.event_type, 21);
        assert!(entry.detail.contains("upgrade from 0.11.0 to 0.12.0"));
        assert_eq!(sink.recorded().len(), 3);
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

    /// The event a Discord test delivers, chosen because it carries every
    /// field shape an embed renders
    fn discord_event() -> UpgradeEvent {
        UpgradeEvent::NodeCompleted {
            node_id: "node-2".to_string(),
            to_version: "0.13.0".to_string(),
            nodes_remaining: 1,
        }
    }

    #[test]
    fn test_a_discord_channel_builds_an_embed_and_a_bad_webhook_url_refuses() {
        let good = ContactChannel::discord("https://discord.com/api/webhooks/123/tok-EN_9")
            .expect("a Discord webhook address is accepted");
        match &good {
            ContactChannel::Discord { webhook_url } => {
                assert_eq!(webhook_url, "https://discord.com/api/webhooks/123/tok-EN_9");
            }
            other => panic!("built {other:?}"),
        }
        assert_eq!(good.describe(), "discord");

        let refused = ContactChannel::discord("https://example.com/api/webhooks/123/tok")
            .expect_err("another host is refused");
        assert!(
            refused.contains("is not a Discord webhook address"),
            "{refused}"
        );
        assert!(
            refused.contains("https://discord.com/api/webhooks/<id>/<token>"),
            "{refused}"
        );

        // Every host the shape allows
        for accepted in [
            "https://discord.com/api/webhooks/1/a",
            "https://canary.discord.com/api/webhooks/1/a",
            "https://ptb.discord.com/api/webhooks/1/a",
            "https://discordapp.com/api/webhooks/1/a",
            "https://canary.discordapp.com/api/webhooks/1/a",
            "https://discord.com/api/webhooks/98765432109876543/aB9_-zZ",
        ] {
            assert!(
                discord_webhook_url_is_wellformed(accepted),
                "{accepted} is a Discord webhook address"
            );
        }
        for rejected in [
            // Plain HTTP, the scheme Discord does not serve webhooks on
            "http://discord.com/api/webhooks/1/a",
            // Another host wearing the path
            "https://discord.evil.com/api/webhooks/1/a",
            "https://discordapp.co/api/webhooks/1/a",
            // No token
            "https://discord.com/api/webhooks/1",
            "https://discord.com/api/webhooks/1/",
            // No id, or an id that is not digits
            "https://discord.com/api/webhooks//a",
            "https://discord.com/api/webhooks/abc/a",
            // Empty path and trailing garbage
            "https://discord.com/api/webhooks/",
            "https://discord.com/api/webhooks/1/a/extra",
            "https://discord.com/api/webhooks/1/a?wait=true",
            "",
        ] {
            assert!(
                !discord_webhook_url_is_wellformed(rejected),
                "{rejected} is not a Discord webhook address"
            );
            assert!(ContactChannel::discord(rejected).is_err(), "{rejected}");
        }
    }

    #[test]
    fn test_a_discord_embed_carries_the_event_its_color_and_its_details() {
        let payload = discord_payload(&discord_event(), 1_757_251_845);
        let embed = &payload["embeds"][0];
        assert_eq!(embed["title"], "Zyron node node-2 is on 0.13.0");
        assert!(
            embed["description"]
                .as_str()
                .unwrap_or_default()
                .contains("Node node-2 is healthy on 0.13.0"),
            "{embed}"
        );
        assert_eq!(embed["color"], 0x2ECC71);
        assert_eq!(embed["footer"]["text"], "Zyron auto-upgrade");
        assert_eq!(embed["timestamp"], "2025-09-07T13:30:45Z");
        let fields = embed["fields"].as_array().expect("fields is an array");
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0]["name"], "Node");
        assert_eq!(fields[0]["value"], "node-2");
        assert_eq!(fields[0]["inline"], true);
        assert_eq!(fields[1]["name"], "To version");
        assert_eq!(fields[1]["value"], "0.13.0");

        // Every event kind maps to the color its class carries
        let colors = [
            (
                UpgradeEvent::PendingDetected {
                    from_version: "a".into(),
                    to_version: "b".into(),
                    gate_summary: "c".into(),
                },
                0x3498DB,
            ),
            (
                UpgradeEvent::Started {
                    from_version: "a".into(),
                    to_version: "b".into(),
                },
                0x3498DB,
            ),
            (discord_event(), 0x2ECC71),
            (
                UpgradeEvent::RolledBack {
                    node_id: "n".into(),
                    reason: "slow".into(),
                },
                0xE74C3C,
            ),
            (
                UpgradeEvent::Paused {
                    reason: "operator".into(),
                },
                0xF39C12,
            ),
            (
                UpgradeEvent::Blocked {
                    reason: "unsafe rewrite".into(),
                },
                0xE74C3C,
            ),
            (
                UpgradeEvent::Completed {
                    to_version: "b".into(),
                    outcome: UpgradeOutcome::Completed,
                    detail: "done".into(),
                },
                0x2ECC71,
            ),
        ];
        for (event, color) in colors {
            assert_eq!(event.embed_color(), color, "{event:?}");
            assert_eq!(
                discord_payload(&event, 0)["embeds"][0]["color"],
                color,
                "{event:?}"
            );
        }
    }

    #[tokio::test]
    async fn test_a_discord_channel_survives_a_429_by_honoring_retry_after_once() {
        let server = httpmock::MockServer::start_async().await;
        let limited = server
            .mock_async(|when, then| {
                when.method(httpmock::Method::POST).path("/hook");
                then.status(429).header("retry-after", "1");
            })
            .await;
        let sink = HttpNotificationSink::new(10).expect("builds");
        let channel = ContactChannel::Discord {
            webhook_url: server.url("/hook"),
        };
        let started = std::time::Instant::now();
        let event = discord_event();

        // The first answer is a rate limit, the second is taken
        let delivery = tokio::join!(sink.deliver(&channel, &event), async {
            tokio::time::sleep(std::time::Duration::from_millis(300)).await;
            limited.delete_async().await;
            server
                .mock_async(|when, then| {
                    when.method(httpmock::Method::POST).path("/hook");
                    then.status(200);
                })
                .await;
        })
        .0;

        assert!(delivery.delivered, "{}", delivery.detail);
        assert_eq!(delivery.channel, "discord");
        assert!(
            delivery.detail.contains("second attempt answered 200"),
            "{}",
            delivery.detail
        );
        assert!(delivery.detail.contains("waited 1s"), "{}", delivery.detail);
        assert!(
            started.elapsed() >= std::time::Duration::from_secs(1),
            "the interval the channel asked for was waited out"
        );
    }

    #[tokio::test]
    async fn test_a_discord_channel_gives_up_when_retry_after_exceeds_the_cap() {
        let server = httpmock::MockServer::start_async().await;
        server
            .mock_async(|when, then| {
                when.method(httpmock::Method::POST).path("/hook");
                then.status(429).header("retry-after", "120");
            })
            .await;
        let sink = HttpNotificationSink::new(10).expect("builds");
        let started = std::time::Instant::now();
        let delivery = sink
            .deliver(
                &ContactChannel::Discord {
                    webhook_url: server.url("/hook"),
                },
                &discord_event(),
            )
            .await;
        assert!(!delivery.delivered, "{}", delivery.detail);
        assert!(
            delivery.detail.contains("asking for 120 seconds"),
            "{}",
            delivery.detail
        );
        assert!(
            delivery.detail.contains("past the 60 second cap"),
            "{}",
            delivery.detail
        );
        assert!(
            started.elapsed() < std::time::Duration::from_secs(60),
            "an interval past the cap is refused rather than waited out"
        );
    }

    #[tokio::test]
    async fn test_a_discord_channel_gives_up_immediately_on_a_bare_429() {
        let server = httpmock::MockServer::start_async().await;
        server
            .mock_async(|when, then| {
                when.method(httpmock::Method::POST).path("/hook");
                then.status(429);
            })
            .await;
        let sink = HttpNotificationSink::new(10).expect("builds");
        let started = std::time::Instant::now();
        let delivery = sink
            .deliver(
                &ContactChannel::Discord {
                    webhook_url: server.url("/hook"),
                },
                &discord_event(),
            )
            .await;
        assert!(!delivery.delivered, "{}", delivery.detail);
        assert!(
            delivery
                .detail
                .contains("without a usable retry-after header"),
            "{}",
            delivery.detail
        );
        assert!(
            started.elapsed() < std::time::Duration::from_secs(1),
            "a rate limit with no interval is not slept on"
        );
    }

    #[test]
    fn test_the_embed_timestamp_is_iso_8601_utc() {
        assert_eq!(iso8601_utc(0), "1970-01-01T00:00:00Z");
        assert_eq!(iso8601_utc(1), "1970-01-01T00:00:01Z");
        // A leap day, which is where a civil-date conversion goes wrong
        assert_eq!(iso8601_utc(1_709_164_800), "2024-02-29T00:00:00Z");
        assert_eq!(iso8601_utc(1_757_251_845), "2025-09-07T13:30:45Z");
    }
}

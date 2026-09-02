//! Auto-upgrade types.
//!
//! Channels, release manifests, the health baseline a rolling upgrade is
//! judged against, maintenance windows, and the state each node publishes.
//! The orchestration that drives these lives in the server, the shapes live
//! here so the catalog views, the CLI, and the wire layer all read the same
//! definitions

use std::fmt;

use super::deprecation::BinaryVersion;
use super::rewrite::UserObjectRewritePolicy;

/// Which release stream a cluster follows
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum UpgradeChannel {
    /// Production-tested releases
    #[default]
    Stable,
    /// Pre-release for early feedback
    Beta,
    /// Bleeding edge
    Canary,
    /// Locked to one version
    Pinned(String),
}

impl UpgradeChannel {
    pub fn label(&self) -> &'static str {
        match self {
            UpgradeChannel::Stable => "stable",
            UpgradeChannel::Beta => "beta",
            UpgradeChannel::Canary => "canary",
            UpgradeChannel::Pinned(_) => "pinned",
        }
    }

    /// Parses a channel name. `pinned` arrives without its version, which
    /// the separate `pinned_version` setting supplies
    pub fn parse(name: &str) -> Option<UpgradeChannel> {
        match name.trim().to_ascii_lowercase().as_str() {
            "stable" => Some(UpgradeChannel::Stable),
            "beta" => Some(UpgradeChannel::Beta),
            "canary" => Some(UpgradeChannel::Canary),
            "pinned" => Some(UpgradeChannel::Pinned(String::new())),
            _ => None,
        }
    }

    /// The path segment the release feed is fetched from
    pub fn feed_segment(&self) -> &str {
        match self {
            UpgradeChannel::Stable => "stable",
            UpgradeChannel::Beta => "beta",
            UpgradeChannel::Canary => "canary",
            UpgradeChannel::Pinned(_) => "stable",
        }
    }

    /// The version this channel is pinned to, when it is pinned
    pub fn pinned_version(&self) -> Option<&str> {
        match self {
            UpgradeChannel::Pinned(version) if !version.is_empty() => Some(version),
            _ => None,
        }
    }
}

impl fmt::Display for UpgradeChannel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UpgradeChannel::Pinned(version) if !version.is_empty() => {
                write!(f, "pinned({version})")
            }
            other => f.write_str(other.label()),
        }
    }
}

/// Where an upgrade has got to
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UpgradePhase {
    /// No upgrade is pending
    #[default]
    Idle,
    /// A release was found on the channel
    Detected,
    /// The compatibility gate is running
    GateRunning,
    /// The gate found rewrites needing an admin
    AwaitingAck,
    /// The gate refused to proceed
    Blocked,
    /// Waiting for the maintenance window to open
    Queued,
    /// Downloading and verifying the binary
    Staging,
    /// Nodes are draining and restarting one at a time
    Rolling,
    /// Post-upgrade format, catalog, and user-object migrations are running
    Migrating,
    /// Halted by an operator or by a failed health check
    Paused,
    /// A node is being returned to the previous binary
    RollingBack,
    /// Finished
    Completed,
    /// Ended without completing
    Failed,
}

impl UpgradePhase {
    pub const fn label(self) -> &'static str {
        match self {
            UpgradePhase::Idle => "Idle",
            UpgradePhase::Detected => "Detected",
            UpgradePhase::GateRunning => "GateRunning",
            UpgradePhase::AwaitingAck => "AwaitingAck",
            UpgradePhase::Blocked => "Blocked",
            UpgradePhase::Queued => "Queued",
            UpgradePhase::Staging => "Staging",
            UpgradePhase::Rolling => "Rolling",
            UpgradePhase::Migrating => "Migrating",
            UpgradePhase::Paused => "Paused",
            UpgradePhase::RollingBack => "RollingBack",
            UpgradePhase::Completed => "Completed",
            UpgradePhase::Failed => "Failed",
        }
    }

    /// Whether the sequence can still move forward from here on its own
    #[inline]
    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            UpgradePhase::Completed | UpgradePhase::Failed | UpgradePhase::Idle
        )
    }

    /// Whether an operator has to act before the sequence resumes
    #[inline]
    pub const fn awaits_operator(self) -> bool {
        matches!(
            self,
            UpgradePhase::AwaitingAck | UpgradePhase::Blocked | UpgradePhase::Paused
        )
    }
}

impl fmt::Display for UpgradePhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// What one node reports about the upgrade it is part of
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeUpgradeState {
    pub node_id: String,
    pub from_version: String,
    pub to_version: String,
    pub phase: UpgradePhase,
    pub started_at_secs: u64,
    pub updated_at_secs: u64,
    /// Whether this node currently leads the consensus group, which decides
    /// where it sits in the sequence
    pub is_leader: bool,
    /// The last thing that happened, printed by `SHOW UPGRADE STATE`
    pub message: String,
}

/// One finished upgrade, held for `zyron_sys.upgrade.history`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpgradeHistoryEntry {
    pub upgrade_id: u64,
    pub from_version: String,
    pub to_version: String,
    pub channel: String,
    pub started_at_secs: u64,
    pub finished_at_secs: u64,
    pub outcome: UpgradeOutcome,
    pub nodes_upgraded: u32,
    pub format_migrations_run: u32,
    pub catalog_migrations_run: u32,
    pub rewrites_applied: u32,
    /// Whether every migration in the upgrade could be undone
    pub reversible: bool,
    pub detail: String,
}

/// How an upgrade ended
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpgradeOutcome {
    Completed,
    RolledBack,
    Paused,
    Blocked,
    Failed,
}

impl UpgradeOutcome {
    pub const fn label(self) -> &'static str {
        match self {
            UpgradeOutcome::Completed => "completed",
            UpgradeOutcome::RolledBack => "rolled_back",
            UpgradeOutcome::Paused => "paused",
            UpgradeOutcome::Blocked => "blocked",
            UpgradeOutcome::Failed => "failed",
        }
    }
}

impl fmt::Display for UpgradeOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// The metrics a node is judged against after it restarts
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct HealthBaseline {
    pub p50_latency_us: u64,
    pub p99_latency_us: u64,
    pub throughput_per_sec: f64,
    pub error_rate: f64,
    pub active_connections: u64,
}

/// How far a restarted node is allowed to drift from the baseline
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HealthThreshold {
    /// Multiple of baseline latency still counted as healthy
    pub latency_multiplier: f64,
    /// Fraction of baseline throughput still counted as healthy
    pub throughput_floor: f64,
    /// Absolute error rate ceiling
    pub error_rate_ceiling: f64,
}

impl Default for HealthThreshold {
    fn default() -> Self {
        Self {
            latency_multiplier: 2.0,
            throughput_floor: 0.5,
            error_rate_ceiling: 0.01,
        }
    }
}

/// Why a restarted node failed its health check
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HealthVerdict {
    Healthy,
    LatencyRegressed { baseline_us: u64, observed_us: u64 },
    ThroughputRegressed { baseline: u64, observed: u64 },
    ErrorRateRegressed { ceiling_ppm: u64, observed_ppm: u64 },
}

impl HealthVerdict {
    #[inline]
    pub fn is_healthy(&self) -> bool {
        matches!(self, HealthVerdict::Healthy)
    }

    pub fn reason(&self) -> String {
        match self {
            HealthVerdict::Healthy => "healthy".to_string(),
            HealthVerdict::LatencyRegressed {
                baseline_us,
                observed_us,
            } => format!("p99 latency {observed_us}us against a {baseline_us}us baseline"),
            HealthVerdict::ThroughputRegressed { baseline, observed } => {
                format!("throughput {observed} per second against a {baseline} per second baseline")
            }
            HealthVerdict::ErrorRateRegressed {
                ceiling_ppm,
                observed_ppm,
            } => format!(
                "error rate {observed_ppm} per million against a {ceiling_ppm} per million \
                 ceiling"
            ),
        }
    }
}

impl HealthBaseline {
    /// Judges an observation against this baseline.
    ///
    /// Latency and error rate are checked first because a node that answers
    /// slowly or wrongly is unhealthy whatever its throughput says
    pub fn judge(&self, observed: &HealthBaseline, threshold: HealthThreshold) -> HealthVerdict {
        if self.p99_latency_us > 0 {
            let ceiling = (self.p99_latency_us as f64 * threshold.latency_multiplier) as u64;
            if observed.p99_latency_us > ceiling {
                return HealthVerdict::LatencyRegressed {
                    baseline_us: self.p99_latency_us,
                    observed_us: observed.p99_latency_us,
                };
            }
        }
        if observed.error_rate > threshold.error_rate_ceiling {
            return HealthVerdict::ErrorRateRegressed {
                ceiling_ppm: (threshold.error_rate_ceiling * 1_000_000.0) as u64,
                observed_ppm: (observed.error_rate * 1_000_000.0) as u64,
            };
        }
        if self.throughput_per_sec > 0.0 {
            let floor = self.throughput_per_sec * threshold.throughput_floor;
            if observed.throughput_per_sec < floor {
                return HealthVerdict::ThroughputRegressed {
                    baseline: self.throughput_per_sec as u64,
                    observed: observed.throughput_per_sec as u64,
                };
            }
        }
        HealthVerdict::Healthy
    }
}

/// One maintenance window, in UTC minutes from midnight
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaintenanceWindow {
    pub start_minute: u32,
    pub end_minute: u32,
}

impl MaintenanceWindow {
    /// Parses `HH:MM-HH:MM UTC`. Only UTC is accepted, so a window means the
    /// same thing on every node whatever its local clock says
    pub fn parse(text: &str) -> Result<MaintenanceWindow, String> {
        let trimmed = text.trim();
        let body = trimmed
            .strip_suffix("UTC")
            .or_else(|| trimmed.strip_suffix("utc"))
            .unwrap_or(trimmed)
            .trim();
        let (start, end) = body
            .split_once('-')
            .ok_or_else(|| format!("maintenance window `{text}` is not `HH:MM-HH:MM UTC`"))?;
        let start_minute = parse_hhmm(start.trim())
            .ok_or_else(|| format!("maintenance window `{text}` has a bad start time"))?;
        let end_minute = parse_hhmm(end.trim())
            .ok_or_else(|| format!("maintenance window `{text}` has a bad end time"))?;
        if start_minute == end_minute {
            return Err(format!(
                "maintenance window `{text}` starts and ends at the same minute"
            ));
        }
        Ok(MaintenanceWindow {
            start_minute,
            end_minute,
        })
    }

    /// Whether a UTC minute-of-day sits inside the window. A window whose
    /// end is before its start wraps past midnight
    pub fn contains_minute(&self, minute_of_day: u32) -> bool {
        if self.start_minute <= self.end_minute {
            minute_of_day >= self.start_minute && minute_of_day < self.end_minute
        } else {
            minute_of_day >= self.start_minute || minute_of_day < self.end_minute
        }
    }

    /// Whether a unix second sits inside the window
    pub fn contains_unix(&self, unix_secs: u64) -> bool {
        self.contains_minute(minute_of_day(unix_secs))
    }
}

impl fmt::Display for MaintenanceWindow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:02}:{:02}-{:02}:{:02} UTC",
            self.start_minute / 60,
            self.start_minute % 60,
            self.end_minute / 60,
            self.end_minute % 60
        )
    }
}

fn parse_hhmm(text: &str) -> Option<u32> {
    let (hh, mm) = text.split_once(':')?;
    let hours: u32 = hh.parse().ok()?;
    let minutes: u32 = mm.parse().ok()?;
    if hours > 23 || minutes > 59 {
        return None;
    }
    Some(hours * 60 + minutes)
}

/// UTC minute of day for a unix second
#[inline]
pub fn minute_of_day(unix_secs: u64) -> u32 {
    ((unix_secs % 86_400) / 60) as u32
}

/// Parses a duration written the way an `OVERLAP` clause writes one.
///
/// Accepts a bare number of seconds, or a number with a `s`, `m`, `h`, `d`,
/// or `w` suffix. A duration with no unit is seconds, which is what a caller
/// writing `OVERLAP 3600` means
pub fn parse_duration_secs(text: &str) -> Result<u64, String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err("duration is empty".to_string());
    }
    let (digits, unit) = match trimmed.char_indices().find(|(_, c)| !c.is_ascii_digit()) {
        Some((index, _)) => trimmed.split_at(index),
        None => (trimmed, ""),
    };
    if digits.is_empty() {
        return Err(format!(
            "duration `{text}` does not start with a number, write it like `24h` or `3600`"
        ));
    }
    let value: u64 = digits
        .parse()
        .map_err(|_| format!("duration `{text}` has a number too large to hold"))?;
    let multiplier = match unit.trim().to_ascii_lowercase().as_str() {
        "" | "s" | "sec" | "secs" | "second" | "seconds" => 1,
        "m" | "min" | "mins" | "minute" | "minutes" => 60,
        "h" | "hr" | "hrs" | "hour" | "hours" => 3_600,
        "d" | "day" | "days" => 86_400,
        "w" | "week" | "weeks" => 604_800,
        other => {
            return Err(format!(
                "duration `{text}` has unit `{other}`, which is not one of s, m, h, d, w"
            ));
        }
    };
    value
        .checked_mul(multiplier)
        .ok_or_else(|| format!("duration `{text}` overflows"))
}

/// A set of maintenance windows. Empty means any time
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MaintenanceSchedule {
    pub windows: Vec<MaintenanceWindow>,
}

impl MaintenanceSchedule {
    /// Parses a comma-separated list of windows. An empty string is any
    /// time, which is the default
    pub fn parse(text: &str) -> Result<MaintenanceSchedule, String> {
        let trimmed = text.trim();
        if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("any") {
            return Ok(MaintenanceSchedule::default());
        }
        let mut windows = Vec::new();
        for part in trimmed.split(',') {
            if part.trim().is_empty() {
                continue;
            }
            windows.push(MaintenanceWindow::parse(part)?);
        }
        Ok(MaintenanceSchedule { windows })
    }

    /// Whether an upgrade may be triggered now
    pub fn is_open(&self, unix_secs: u64) -> bool {
        self.windows.is_empty() || self.windows.iter().any(|w| w.contains_unix(unix_secs))
    }

    /// Seconds until the next window opens, or 0 when one is open now
    pub fn secs_until_open(&self, unix_secs: u64) -> u64 {
        if self.is_open(unix_secs) {
            return 0;
        }
        let now = minute_of_day(unix_secs);
        let wait = self
            .windows
            .iter()
            .map(|w| {
                if w.start_minute >= now {
                    w.start_minute - now
                } else {
                    1_440 - now + w.start_minute
                }
            })
            .min()
            .unwrap_or(0);
        // Land on the first second of the window rather than mid-minute
        (wait as u64) * 60 - (unix_secs % 60)
    }
}

impl fmt::Display for MaintenanceSchedule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.windows.is_empty() {
            return f.write_str("any");
        }
        let rendered: Vec<String> = self.windows.iter().map(|w| w.to_string()).collect();
        f.write_str(&rendered.join(", "))
    }
}

/// One release the feed offers
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReleaseEntry {
    pub version: String,
    /// Where the binary is fetched from
    pub artifact_url: String,
    /// Hex SHA-256 of the binary
    pub sha256: String,
    /// The scheme that signed the manifest
    pub signature_scheme: String,
    /// Hex signature over the canonical manifest bytes
    pub signature: String,
    /// Versions that must be passed through to reach this one, oldest first
    pub upgrade_chain: Vec<String>,
    /// Whether this release changes any format's writer version
    pub carries_format_bump: bool,
    /// Release notes URL
    pub notes_url: String,
}

/// The signed release feed for one channel
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReleaseManifest {
    pub channel: String,
    pub generated_at_secs: u64,
    pub releases: Vec<ReleaseEntry>,
    /// The scheme that signed this manifest
    pub signature_scheme: String,
    /// Hex signature over the canonical manifest bytes
    pub signature: String,
}

impl ReleaseManifest {
    /// The newest release on the feed that is ahead of a running version
    pub fn newest_after(&self, running: &str) -> Option<&ReleaseEntry> {
        let running = BinaryVersion::parse(running)?;
        self.releases
            .iter()
            .filter(|r| {
                BinaryVersion::parse(&r.version)
                    .map(|v| v > running)
                    .unwrap_or(false)
            })
            .max_by_key(|r| BinaryVersion::parse(&r.version).unwrap_or_default())
    }

    /// One release by version
    pub fn release(&self, version: &str) -> Option<&ReleaseEntry> {
        self.releases.iter().find(|r| r.version == version)
    }

    /// The versions to pass through to get from a running version to a
    /// target, target last. Empty when the target is not on the feed
    pub fn plan_chain(&self, running: &str, target: &str) -> Vec<String> {
        let Some(entry) = self.release(target) else {
            return Vec::new();
        };
        let Some(running) = BinaryVersion::parse(running) else {
            return Vec::new();
        };
        let mut chain: Vec<String> = entry
            .upgrade_chain
            .iter()
            .filter(|step| {
                BinaryVersion::parse(step)
                    .map(|v| v > running)
                    .unwrap_or(false)
            })
            .cloned()
            .collect();
        chain.sort_by_key(|v| BinaryVersion::parse(v).unwrap_or_default());
        chain.push(target.to_string());
        chain
    }

    /// The bytes a signature covers, which is every field except the
    /// signature itself, in a fixed order
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(256);
        out.extend_from_slice(self.channel.as_bytes());
        out.push(b'\n');
        out.extend_from_slice(self.generated_at_secs.to_string().as_bytes());
        out.push(b'\n');
        for release in &self.releases {
            out.extend_from_slice(release.version.as_bytes());
            out.push(b'\t');
            out.extend_from_slice(release.artifact_url.as_bytes());
            out.push(b'\t');
            out.extend_from_slice(release.sha256.as_bytes());
            out.push(b'\t');
            out.extend_from_slice(release.upgrade_chain.join(",").as_bytes());
            out.push(b'\t');
            out.extend_from_slice(if release.carries_format_bump {
                b"format_bump"
            } else {
                b"no_format_bump"
            });
            out.push(b'\n');
        }
        out
    }
}

/// The settings the upgrade substrate reads
#[derive(Debug, Clone, PartialEq)]
pub struct UpgradeSettings {
    pub auto_upgrade_enabled: bool,
    pub channel: UpgradeChannel,
    pub pinned_version: Option<String>,
    pub window: MaintenanceSchedule,
    pub paused: bool,
    pub user_object_rewrite_policy: UserObjectRewritePolicy,
    /// Fraction of node memory an eager sweep may use
    pub format_migration_budget_memory_fraction: f64,
    /// Seconds one format's migration may run for
    pub format_migration_budget_time_secs: u64,
    /// Multiple of source size the migration may occupy on disk
    pub format_migration_budget_disk_multiple: f64,
    pub rollback_on_health_fail: bool,
    pub federation_coordination_timeout_secs: u64,
    /// Take a snapshot before a major-version upgrade
    pub pre_upgrade_backup_snapshot: bool,
    pub deprecation_warning_rate_limit_per_hour: u32,
    pub release_feed_poll_interval_secs: u64,
    /// Seconds a node has to reach its health baseline after restart
    pub health_recovery_timeout_secs: u64,
    /// Where the signed release feed is fetched from
    pub release_feed_url: String,
}

impl Default for UpgradeSettings {
    fn default() -> Self {
        Self {
            auto_upgrade_enabled: true,
            channel: UpgradeChannel::Stable,
            pinned_version: None,
            window: MaintenanceSchedule::default(),
            paused: false,
            user_object_rewrite_policy: UserObjectRewritePolicy::AutoSafe,
            format_migration_budget_memory_fraction: 0.25,
            format_migration_budget_time_secs: 6 * 3_600,
            format_migration_budget_disk_multiple: 2.0,
            rollback_on_health_fail: true,
            federation_coordination_timeout_secs: 30 * 60,
            pre_upgrade_backup_snapshot: true,
            deprecation_warning_rate_limit_per_hour:
                super::deprecation::DEFAULT_WARNING_RATE_LIMIT_PER_HOUR,
            release_feed_poll_interval_secs: 4 * 3_600,
            health_recovery_timeout_secs: 5 * 60,
            release_feed_url: "https://releases.zyron.dev/feed".to_string(),
        }
    }
}

impl UpgradeSettings {
    /// The channel with its pinned version filled in, which is the form the
    /// poller uses
    pub fn effective_channel(&self) -> UpgradeChannel {
        match (&self.channel, &self.pinned_version) {
            (UpgradeChannel::Pinned(_), Some(version)) => UpgradeChannel::Pinned(version.clone()),
            (other, _) => other.clone(),
        }
    }

    /// Whether an upgrade may start right now
    pub fn may_start(&self, now_secs: u64) -> Result<(), String> {
        if !self.auto_upgrade_enabled {
            return Err("auto_upgrade_enabled is false".to_string());
        }
        if self.paused {
            return Err("auto_upgrade_paused is true".to_string());
        }
        if !self.window.is_open(now_secs) {
            return Err(format!(
                "outside the maintenance window {}, {} seconds until it opens",
                self.window,
                self.window.secs_until_open(now_secs)
            ));
        }
        Ok(())
    }

    /// The full feed URL for the effective channel
    pub fn feed_url(&self) -> String {
        format!(
            "{}/{}.manifest",
            self.release_feed_url.trim_end_matches('/'),
            self.channel.feed_segment()
        )
    }
}

/// One user-authored object an upgrade would rewrite
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RewriteRecord {
    pub object_name: String,
    pub object_kind: super::rewrite::ObjectKind,
    pub rewriter_name: String,
    pub category: super::rewrite::RewriteCategory,
    pub status: super::rewrite::RewriteStatus,
    /// Hash of the statement before the rewrite
    pub before_hash: u32,
    /// Hash of the statement after it
    pub after_hash: u32,
    /// Who acknowledged it, empty while nobody has
    pub acknowledged_by: String,
    pub updated_at_secs: u64,
    /// The human-readable diff the admin sees
    pub diff: String,
}

/// The node's observable upgrade state.
///
/// The orchestrator writes here and the catalog views, the CLI, and the DDL
/// surface read from here, so all four see one state rather than each
/// keeping a copy. Every field is behind its own lock and touched once per
/// upgrade step, never on a query path
#[derive(Debug, Default)]
pub struct UpgradeBoard {
    settings: std::sync::Mutex<UpgradeSettings>,
    nodes: std::sync::Mutex<Vec<NodeUpgradeState>>,
    history: std::sync::Mutex<Vec<UpgradeHistoryEntry>>,
    rewrites: std::sync::Mutex<Vec<RewriteRecord>>,
}

/// How many finished upgrades the board holds before the oldest fall off
const MAX_HISTORY_ENTRIES: usize = 256;

impl UpgradeBoard {
    pub fn new() -> Self {
        Self::default()
    }

    /// The settings in force, which `ALTER SYSTEM` changes
    pub fn settings(&self) -> UpgradeSettings {
        self.settings.lock().map(|s| s.clone()).unwrap_or_default()
    }

    /// Applies one change to the settings and hands back what they became
    pub fn update_settings(&self, change: impl FnOnce(&mut UpgradeSettings)) -> UpgradeSettings {
        match self.settings.lock() {
            Ok(mut slot) => {
                change(&mut slot);
                slot.clone()
            }
            Err(_) => UpgradeSettings::default(),
        }
    }

    /// Publishes one node's state, replacing what that node published before
    pub fn set_node_state(&self, state: NodeUpgradeState) {
        if let Ok(mut nodes) = self.nodes.lock() {
            match nodes.iter_mut().find(|n| n.node_id == state.node_id) {
                Some(existing) => *existing = state,
                None => nodes.push(state),
            }
            nodes.sort_by(|a, b| a.node_id.cmp(&b.node_id));
        }
    }

    /// Every node's state, node id order
    pub fn node_states(&self) -> Vec<NodeUpgradeState> {
        self.nodes.lock().map(|n| n.clone()).unwrap_or_default()
    }

    /// The phase the cluster is in, which is the least advanced phase any
    /// node reports. A sequence is only as far along as its slowest node
    pub fn cluster_phase(&self) -> UpgradePhase {
        let nodes = self.node_states();
        if nodes.is_empty() {
            return UpgradePhase::Idle;
        }
        // An operator-blocking phase on any node is the cluster's phase,
        // because the sequence has stopped
        for node in &nodes {
            if node.phase.awaits_operator() {
                return node.phase;
            }
        }
        if nodes.iter().all(|n| n.phase == UpgradePhase::Completed) {
            return UpgradePhase::Completed;
        }
        nodes
            .iter()
            .map(|n| n.phase)
            .find(|p| !p.is_terminal())
            .unwrap_or(UpgradePhase::Idle)
    }

    /// Records a finished upgrade.
    ///
    /// The oldest entries fall off past `MAX_HISTORY_ENTRIES`. This board is
    /// the live view a running server answers from, not the durable record,
    /// so it stays bounded however long the process runs and however many
    /// upgrades it sees
    pub fn push_history(&self, entry: UpgradeHistoryEntry) {
        if let Ok(mut history) = self.history.lock() {
            history.push(entry);
            let excess = history.len().saturating_sub(MAX_HISTORY_ENTRIES);
            if excess > 0 {
                history.drain(..excess);
            }
        }
    }

    /// Past upgrades, newest first, at most `limit`
    pub fn history(&self, limit: usize) -> Vec<UpgradeHistoryEntry> {
        let mut entries = self.history.lock().map(|h| h.clone()).unwrap_or_default();
        entries.sort_by(|a, b| b.upgrade_id.cmp(&a.upgrade_id));
        entries.truncate(limit);
        entries
    }

    /// The id the next upgrade takes
    pub fn next_upgrade_id(&self) -> u64 {
        self.history
            .lock()
            .map(|h| h.iter().map(|e| e.upgrade_id).max().unwrap_or(0) + 1)
            .unwrap_or(1)
    }

    /// Replaces the rewrite queue, which the compatibility gate does
    pub fn set_rewrites(&self, records: Vec<RewriteRecord>) {
        if let Ok(mut slot) = self.rewrites.lock() {
            *slot = records;
        }
    }

    /// The rewrite queue
    pub fn rewrites(&self) -> Vec<RewriteRecord> {
        self.rewrites.lock().map(|r| r.clone()).unwrap_or_default()
    }

    /// Marks every queued rewrite of one category acknowledged, returning how
    /// many moved
    pub fn acknowledge(
        &self,
        category: super::rewrite::RewriteCategory,
        actor: &str,
        now_secs: u64,
    ) -> usize {
        let Ok(mut records) = self.rewrites.lock() else {
            return 0;
        };
        let mut moved = 0;
        for record in records.iter_mut() {
            if record.category == category
                && record.status == super::rewrite::RewriteStatus::Pending
            {
                record.status = super::rewrite::RewriteStatus::Acknowledged;
                record.acknowledged_by = actor.to_string();
                record.updated_at_secs = now_secs;
                moved += 1;
            }
        }
        moved
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_parses_and_renders() {
        assert_eq!(
            UpgradeChannel::parse("STABLE"),
            Some(UpgradeChannel::Stable)
        );
        assert_eq!(UpgradeChannel::parse("beta"), Some(UpgradeChannel::Beta));
        assert_eq!(
            UpgradeChannel::parse("canary"),
            Some(UpgradeChannel::Canary)
        );
        assert_eq!(
            UpgradeChannel::parse("pinned"),
            Some(UpgradeChannel::Pinned(String::new()))
        );
        assert_eq!(UpgradeChannel::parse("nope"), None);
        assert_eq!(
            UpgradeChannel::Pinned("2.3.1".to_string()).to_string(),
            "pinned(2.3.1)"
        );
    }

    #[test]
    fn test_feed_url_follows_the_channel() {
        let mut settings = UpgradeSettings::default();
        assert!(settings.feed_url().ends_with("/stable.manifest"));
        settings.channel = UpgradeChannel::Beta;
        assert!(settings.feed_url().ends_with("/beta.manifest"));
        settings.channel = UpgradeChannel::Stable;
        assert!(settings.feed_url().ends_with("/stable.manifest"));
    }

    #[test]
    fn test_maintenance_window_parses_and_contains() {
        let window = MaintenanceWindow::parse("02:00-04:00 UTC").expect("parses");
        assert_eq!(window.start_minute, 120);
        assert_eq!(window.end_minute, 240);
        assert!(window.contains_minute(150));
        assert!(!window.contains_minute(900));
        assert_eq!(window.to_string(), "02:00-04:00 UTC");
        assert!(MaintenanceWindow::parse("bad").is_err());
        assert!(MaintenanceWindow::parse("25:00-26:00 UTC").is_err());
    }

    #[test]
    fn test_maintenance_window_wraps_past_midnight() {
        let window = MaintenanceWindow::parse("23:00-01:00 UTC").expect("parses");
        assert!(window.contains_minute(23 * 60 + 30));
        assert!(window.contains_minute(30));
        assert!(!window.contains_minute(12 * 60));
    }

    #[test]
    fn test_schedule_queues_outside_the_window() {
        let schedule = MaintenanceSchedule::parse("02:00-04:00 UTC").expect("parses");
        // 15:00 UTC
        let at_1500 = 15 * 3_600;
        assert!(!schedule.is_open(at_1500));
        // 02:00 UTC the next day
        let at_0200 = 2 * 3_600;
        assert!(schedule.is_open(at_0200));
        assert_eq!(schedule.secs_until_open(at_0200), 0);
        assert!(schedule.secs_until_open(at_1500) > 0);
    }

    #[test]
    fn test_empty_schedule_is_always_open() {
        let schedule = MaintenanceSchedule::parse("").expect("parses");
        assert!(schedule.is_open(0));
        assert!(schedule.is_open(12 * 3_600));
        assert_eq!(schedule.to_string(), "any");
    }

    #[test]
    fn test_multiple_windows_are_accepted() {
        let schedule =
            MaintenanceSchedule::parse("02:00-04:00 UTC, 14:00-15:00 UTC").expect("parses");
        assert_eq!(schedule.windows.len(), 2);
        assert!(schedule.is_open(14 * 3_600 + 600));
        assert!(!schedule.is_open(20 * 3_600));
    }

    #[test]
    fn test_may_start_names_the_reason_it_cannot() {
        let mut settings = UpgradeSettings::default();
        assert!(settings.may_start(0).is_ok());
        settings.paused = true;
        assert!(
            settings
                .may_start(0)
                .expect_err("paused")
                .contains("auto_upgrade_paused")
        );
        settings.paused = false;
        settings.auto_upgrade_enabled = false;
        assert!(
            settings
                .may_start(0)
                .expect_err("disabled")
                .contains("auto_upgrade_enabled")
        );
        settings.auto_upgrade_enabled = true;
        settings.window = MaintenanceSchedule::parse("02:00-04:00 UTC").expect("parses");
        assert!(
            settings
                .may_start(15 * 3_600)
                .expect_err("outside window")
                .contains("maintenance window")
        );
    }

    #[test]
    fn test_health_verdict_catches_each_regression() {
        let baseline = HealthBaseline {
            p50_latency_us: 100,
            p99_latency_us: 1_000,
            throughput_per_sec: 10_000.0,
            error_rate: 0.0,
            active_connections: 50,
        };
        let threshold = HealthThreshold::default();
        assert!(baseline.judge(&baseline, threshold).is_healthy());

        let slow = HealthBaseline {
            p99_latency_us: 5_000,
            ..baseline
        };
        assert!(matches!(
            baseline.judge(&slow, threshold),
            HealthVerdict::LatencyRegressed { .. }
        ));

        let erroring = HealthBaseline {
            error_rate: 0.5,
            ..baseline
        };
        assert!(matches!(
            baseline.judge(&erroring, threshold),
            HealthVerdict::ErrorRateRegressed { .. }
        ));

        let starved = HealthBaseline {
            throughput_per_sec: 100.0,
            ..baseline
        };
        assert!(matches!(
            baseline.judge(&starved, threshold),
            HealthVerdict::ThroughputRegressed { .. }
        ));
    }

    fn manifest() -> ReleaseManifest {
        ReleaseManifest {
            channel: "stable".to_string(),
            generated_at_secs: 1_000,
            releases: vec![
                ReleaseEntry {
                    version: "2.1.0".to_string(),
                    artifact_url: "https://example/2.1.0".to_string(),
                    sha256: "aa".to_string(),
                    signature_scheme: "Ed25519".to_string(),
                    signature: "00".to_string(),
                    upgrade_chain: vec![],
                    carries_format_bump: false,
                    notes_url: String::new(),
                },
                ReleaseEntry {
                    version: "2.3.0".to_string(),
                    artifact_url: "https://example/2.3.0".to_string(),
                    sha256: "bb".to_string(),
                    signature_scheme: "Ed25519".to_string(),
                    signature: "00".to_string(),
                    upgrade_chain: vec!["2.1.0".to_string(), "2.2.0".to_string()],
                    carries_format_bump: true,
                    notes_url: String::new(),
                },
            ],
            signature_scheme: "Ed25519".to_string(),
            signature: "ff".to_string(),
        }
    }

    #[test]
    fn test_newest_after_picks_the_highest_release() {
        let manifest = manifest();
        assert_eq!(
            manifest.newest_after("2.0.0").expect("found").version,
            "2.3.0"
        );
        assert!(manifest.newest_after("3.0.0").is_none());
    }

    #[test]
    fn test_chained_upgrade_is_planned_from_the_manifest() {
        let manifest = manifest();
        assert_eq!(
            manifest.plan_chain("2.0.0", "2.3.0"),
            vec![
                "2.1.0".to_string(),
                "2.2.0".to_string(),
                "2.3.0".to_string()
            ]
        );
        // Already past the first step, so it drops out of the chain
        assert_eq!(
            manifest.plan_chain("2.1.0", "2.3.0"),
            vec!["2.2.0".to_string(), "2.3.0".to_string()]
        );
        assert!(manifest.plan_chain("2.0.0", "9.9.9").is_empty());
    }

    #[test]
    fn test_canonical_bytes_exclude_the_signature() {
        let mut manifest = manifest();
        let before = manifest.canonical_bytes();
        manifest.signature = "different".to_string();
        assert_eq!(before, manifest.canonical_bytes());
        manifest.releases[0].sha256 = "cc".to_string();
        assert_ne!(before, manifest.canonical_bytes());
    }

    #[test]
    fn test_duration_parsing() {
        assert_eq!(parse_duration_secs("3600"), Ok(3_600));
        assert_eq!(parse_duration_secs("24h"), Ok(86_400));
        assert_eq!(parse_duration_secs("30m"), Ok(1_800));
        assert_eq!(parse_duration_secs("7d"), Ok(604_800));
        assert_eq!(parse_duration_secs("2w"), Ok(1_209_600));
        assert_eq!(parse_duration_secs(" 45 s "), Ok(45));
        assert!(parse_duration_secs("").is_err());
        assert!(parse_duration_secs("h").is_err());
        let err = parse_duration_secs("24y").expect_err("bad unit");
        assert!(err.contains("s, m, h, d, w"), "{err}");
    }

    #[test]
    fn test_phase_classification() {
        assert!(UpgradePhase::Completed.is_terminal());
        assert!(!UpgradePhase::Rolling.is_terminal());
        assert!(UpgradePhase::AwaitingAck.awaits_operator());
        assert!(UpgradePhase::Paused.awaits_operator());
        assert!(!UpgradePhase::Staging.awaits_operator());
    }

    /// A server that runs for years and upgrades often must not accumulate
    /// history without bound, and the newest entries are the ones kept
    #[test]
    fn test_history_is_bounded_and_keeps_the_newest() {
        let board = UpgradeBoard::default();
        let total = MAX_HISTORY_ENTRIES + 50;
        for id in 0..total as u64 {
            board.push_history(UpgradeHistoryEntry {
                upgrade_id: id,
                from_version: "0.11.0".to_string(),
                to_version: "0.12.0".to_string(),
                channel: "stable".to_string(),
                started_at_secs: id,
                finished_at_secs: id + 1,
                outcome: UpgradeOutcome::Completed,
                nodes_upgraded: 1,
                format_migrations_run: 0,
                catalog_migrations_run: 0,
                rewrites_applied: 0,
                reversible: true,
                detail: String::new(),
            });
        }
        let all = board.history(usize::MAX);
        assert_eq!(all.len(), MAX_HISTORY_ENTRIES);
        assert_eq!(
            all.first().map(|e| e.upgrade_id),
            Some(total as u64 - 1),
            "history reads newest first"
        );
        assert_eq!(
            all.last().map(|e| e.upgrade_id),
            Some((total - MAX_HISTORY_ENTRIES) as u64),
            "the oldest entries fell off, not the newest"
        );
    }
}

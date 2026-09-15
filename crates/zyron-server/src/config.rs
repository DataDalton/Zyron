//! TOML configuration loading for Zyron.
//!
//! Reads a zyron.toml file, applies environment variable overrides,
//! validates settings, and maps sections to the existing ServerConfig
//! and StorageConfig structs.

use serde::Deserialize;
use std::path::{Path, PathBuf};
use zyron_common::format::FormatKind;
use zyron_common::format::text_envelope;
use zyron_common::{Result, ZyronError};

/// Serializes every rewrite of zyron.auto.conf in this process
static AUTO_CONF_WRITE: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

/// Writes a file through a sibling and a rename. The sibling is synced
/// before the rename and the directory after it, so once this returns the
/// new contents are on disk under the final name, and a crash before that
/// leaves the previous file untouched
fn write_replacing(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write;

    let file_name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let sibling = path.with_file_name(format!("{file_name}.tmp"));
    let written = (|| -> std::io::Result<()> {
        let mut file = std::fs::File::create(&sibling)?;
        file.write_all(bytes)?;
        file.sync_all()
    })();
    if let Err(e) = written {
        let _ = std::fs::remove_file(&sibling);
        return Err(ZyronError::Internal(format!(
            "{} could not be written, {e}",
            sibling.display()
        )));
    }
    if let Err(e) = std::fs::rename(&sibling, path) {
        let _ = std::fs::remove_file(&sibling);
        return Err(ZyronError::Internal(format!(
            "{} could not replace {}, {e}",
            sibling.display(),
            path.display()
        )));
    }
    #[cfg(unix)]
    if let Some(directory) = path.parent() {
        std::fs::File::open(directory)
            .and_then(|dir| dir.sync_all())
            .map_err(|e| {
                ZyronError::Internal(format!(
                    "{} could not be synced after writing {}, {e}",
                    directory.display(),
                    path.display()
                ))
            })?;
    }
    Ok(())
}

/// Top-level server configuration loaded from zyron.toml.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ZyronConfig {
    pub server: ServerSection,
    pub storage: StorageSection,
    pub wal: WalSection,
    pub checkpoint: CheckpointSection,
    pub auth: AuthSection,
    pub logging: LoggingSection,
    pub metrics: MetricsSection,
    pub compaction: CompactionSection,
    pub vacuum: VacuumSection,
    pub query: QuerySection,
    /// How this deployment acquires and releases nodes. Lives with the
    /// provisioner so that nothing outside it reads how the node was
    /// registered
    pub mesh: zyron_pressure::provisioner::MeshSection,
    /// The consensus group this node belongs to. Off unless an operator
    /// describes one, and a node with it off behaves exactly as it did
    /// before there were groups
    pub cluster: ClusterSection,
    /// External media tooling: paths to the binaries media operations
    /// invoke when configured
    pub media: MediaSection,
    /// How this node finds, checks, and installs releases. The upgrade board
    /// is seeded from here at boot and `ALTER SYSTEM SET` writes back here,
    /// so the settings an operator sees are the ones the next boot reads
    pub upgrade: UpgradeSection,
    /// Which signature scheme signs each artifact kind. The scheme registry
    /// is seeded from here at boot and `SET SIGNATURE SCHEME` writes back
    /// here through the replicated log, so a scheme an operator chose is
    /// still in force after a restart and on every member of the group
    pub crypto: CryptoSection,
    /// Change data feeds and change streams, the caps every feed's
    /// retention and size stay under, how many streams a table may carry,
    /// and the thresholds the stream alerts fire on
    pub cdc: CdcSection,
    /// Verifiable tables, how often a chain's head is anchored and how many
    /// commits a sampled verification reads the rows of
    pub verify: VerifySection,
}

/// [verify] section of the config file
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct VerifySection {
    /// Seconds between anchoring passes. An anchor is what makes a chain's
    /// truncation detectable, and `chain_not_anchored` fires on a head that
    /// has gone twice this long without one
    pub anchor_interval_secs: u64,
    /// Commits a verification reads the rows of when the statement names no
    /// number. Reading every commit's rows is `rows => 'all'`, which takes
    /// MANAGE_VERIFICATION
    pub default_sample: u64,
}

impl Default for VerifySection {
    fn default() -> Self {
        Self {
            anchor_interval_secs: zyron_lifecycle::verify::anchor::DEFAULT_ANCHOR_INTERVAL_SECS,
            default_sample: 64,
        }
    }
}

/// [cdc] section of the config file
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct CdcSection {
    /// Bytes one table's change feed may hold before its oldest changes are
    /// purged ahead of their retention. Zero sets no cap. A feed at its cap
    /// purges rather than refusing the table's writes
    pub cdf_max_bytes_per_table: u64,
    /// The longest retention any feed may be given, in seconds. Zero sets no
    /// cap. A feed asked for more is refused naming this
    pub cdf_max_retention_secs: u64,
    /// Change streams one table may carry before another is refused
    pub change_streams_per_table: u32,
    /// Pending rows at which cdc_stream_lag fires
    pub stream_lag_rows: u64,
    /// Age in seconds of the oldest unconsumed change at which
    /// cdc_stream_lag fires
    pub stream_lag_seconds: u64,
    /// How close, in seconds, retention may come to reclaiming a stream's
    /// unconsumed changes before cdf_retention_pressure fires
    pub retention_margin_secs: u64,
    /// Seconds between passes of the staleness sweeper and the alert
    /// evaluation
    pub sweep_interval_secs: u64,
}

impl Default for CdcSection {
    fn default() -> Self {
        Self {
            cdf_max_bytes_per_table: 0,
            cdf_max_retention_secs: 0,
            change_streams_per_table: 64,
            stream_lag_rows: 1_000_000,
            stream_lag_seconds: 3600,
            retention_margin_secs: 3600,
            sweep_interval_secs: 10,
        }
    }
}

/// [crypto] section of the config file.
///
/// One entry per artifact kind, keyed by the kind's catalog name lowercased,
/// holding the binding in the form the scheme registry reads. The kinds are
/// not fields because the set of them belongs to the registry, so a kind
/// added there needs no change here
#[derive(Debug, Clone, Default, Deserialize)]
pub struct CryptoSection {
    #[serde(flatten)]
    pub artifact_schemes: std::collections::BTreeMap<String, String>,
}

/// [upgrade] section of the config file.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct UpgradeSection {
    pub auto_upgrade_enabled: bool,
    /// stable, beta, canary, or pinned
    pub channel: String,
    /// The version a pinned channel holds at
    pub pinned_version: Option<String>,
    /// Comma separated `HH:MM-HH:MM UTC` windows, empty for any time
    pub window: String,
    /// Set by an operator or by a failed health check, and cleared only by
    /// an operator
    pub paused: bool,
    /// auto_safe, notify_all, or manual_only
    pub user_object_rewrite_policy: String,
    pub format_migration_budget_memory_fraction: f64,
    pub format_migration_budget_time_secs: u64,
    pub format_migration_budget_disk_multiple: f64,
    pub rollback_on_health_fail: bool,
    pub federation_coordination_timeout_secs: u64,
    pub pre_upgrade_backup_snapshot: bool,
    pub deprecation_warning_rate_limit_per_hour: u32,
    pub release_feed_poll_interval_secs: u64,
    /// Seconds a restarted node has to reach the health baseline
    pub health_recovery_timeout_secs: u64,
    /// Seconds between health observations while a node recovers
    pub health_poll_interval_secs: u64,
    /// Multiple of the baseline p99 still counted as healthy
    pub health_latency_multiplier: f64,
    /// Fraction of the baseline throughput still counted as healthy
    pub health_throughput_floor: f64,
    /// Absolute error rate ceiling
    pub health_error_rate_ceiling: f64,
    /// Seconds a node has to finish its in-flight work before a restart
    pub drain_timeout_secs: u64,
    /// Seconds a node has to stage a release once asked
    pub stage_timeout_secs: u64,
    /// Where the signed release feed lives. Empty is the vendor's feed, an
    /// `http://` or `https://` URL is another feed, and a directory is a
    /// feed an operator publishes into by hand with no remote at all. The
    /// `releases` directory under the data directory is read ahead of a
    /// remote feed either way
    pub release_feed_url: String,
    /// The public half of the key releases are signed with, as hex. Empty
    /// uses the vendor's key built into this binary. Set it to verify
    /// releases signed with a key of your own
    pub release_signing_key: String,
    /// The signature scheme a configured release key belongs to
    pub release_signing_scheme: String,
    /// Where upgrade notifications are posted, empty for none
    pub notify_webhook_url: String,
    /// A Slack incoming webhook that receives upgrade notifications, empty
    /// for none
    pub notify_slack_webhook_url: String,
    /// A Discord channel webhook that receives upgrade notifications as an
    /// embed, empty for none
    pub notify_discord_webhook_url: String,
}

impl Default for UpgradeSection {
    fn default() -> Self {
        let settings = zyron_common::format::UpgradeSettings::default();
        Self {
            auto_upgrade_enabled: settings.auto_upgrade_enabled,
            channel: settings.channel.label().to_string(),
            pinned_version: None,
            window: String::new(),
            paused: false,
            user_object_rewrite_policy: settings.user_object_rewrite_policy.label().to_string(),
            format_migration_budget_memory_fraction: settings
                .format_migration_budget_memory_fraction,
            format_migration_budget_time_secs: settings.format_migration_budget_time_secs,
            format_migration_budget_disk_multiple: settings.format_migration_budget_disk_multiple,
            rollback_on_health_fail: settings.rollback_on_health_fail,
            federation_coordination_timeout_secs: settings.federation_coordination_timeout_secs,
            pre_upgrade_backup_snapshot: settings.pre_upgrade_backup_snapshot,
            deprecation_warning_rate_limit_per_hour: settings
                .deprecation_warning_rate_limit_per_hour,
            release_feed_poll_interval_secs: settings.release_feed_poll_interval_secs,
            health_recovery_timeout_secs: settings.health_recovery_timeout_secs,
            health_poll_interval_secs: 5,
            health_latency_multiplier: 2.0,
            health_throughput_floor: 0.5,
            health_error_rate_ceiling: 0.01,
            drain_timeout_secs: 300,
            stage_timeout_secs: 600,
            release_feed_url: String::new(),
            release_signing_key: String::new(),
            release_signing_scheme: "Ed25519".to_string(),
            notify_webhook_url: String::new(),
            notify_slack_webhook_url: String::new(),
            notify_discord_webhook_url: String::new(),
        }
    }
}

impl UpgradeSection {
    /// Refuses a section the board could not be seeded from
    pub fn validate(&self) -> Result<()> {
        use zyron_common::format::rewrite::UserObjectRewritePolicy;
        use zyron_common::format::{BinaryVersion, MaintenanceSchedule, UpgradeChannel};

        if UpgradeChannel::parse(&self.channel).is_none() {
            return Err(ZyronError::ConfigError(format!(
                "upgrade.channel \"{}\" is not a channel, use stable, beta, canary, or pinned",
                self.channel
            )));
        }
        if let Some(pinned) = &self.pinned_version {
            if BinaryVersion::parse(pinned).is_none() {
                return Err(ZyronError::ConfigError(format!(
                    "upgrade.pinned_version \"{pinned}\" is not a major.minor.patch version"
                )));
            }
        }
        if self.channel.eq_ignore_ascii_case("pinned") && self.pinned_version.is_none() {
            return Err(ZyronError::ConfigError(
                "upgrade.channel is pinned but upgrade.pinned_version names no version".into(),
            ));
        }
        MaintenanceSchedule::parse(&self.window)
            .map_err(|e| ZyronError::ConfigError(format!("upgrade.window, {e}")))?;
        if UserObjectRewritePolicy::parse(&self.user_object_rewrite_policy).is_none() {
            return Err(ZyronError::ConfigError(format!(
                "upgrade.user_object_rewrite_policy \"{}\" is not a policy, use auto_safe, \
                 notify_all, or manual_only",
                self.user_object_rewrite_policy
            )));
        }
        if !(0.0..=1.0).contains(&self.format_migration_budget_memory_fraction) {
            return Err(ZyronError::ConfigError(
                "upgrade.format_migration_budget_memory_fraction must be between 0 and 1".into(),
            ));
        }
        if self.format_migration_budget_disk_multiple < 1.0 {
            return Err(ZyronError::ConfigError(
                "upgrade.format_migration_budget_disk_multiple must be at least 1".into(),
            ));
        }
        if self.health_latency_multiplier < 1.0 {
            return Err(ZyronError::ConfigError(
                "upgrade.health_latency_multiplier must be at least 1".into(),
            ));
        }
        if !(0.0..=1.0).contains(&self.health_throughput_floor) {
            return Err(ZyronError::ConfigError(
                "upgrade.health_throughput_floor must be between 0 and 1".into(),
            ));
        }
        if !(0.0..=1.0).contains(&self.health_error_rate_ceiling) {
            return Err(ZyronError::ConfigError(
                "upgrade.health_error_rate_ceiling must be between 0 and 1".into(),
            ));
        }
        if self.health_poll_interval_secs == 0 {
            return Err(ZyronError::ConfigError(
                "upgrade.health_poll_interval_secs must be at least 1".into(),
            ));
        }
        if !self.release_signing_key.is_empty() {
            self.release_key_material()?;
        }
        Ok(())
    }

    /// The release signing key as verifying material: the configured key
    /// when one is set, otherwise the vendor's key built into this binary
    pub fn release_key_material(&self) -> Result<zyron_auth::signature::VerifyingMaterial> {
        use zyron_auth::signature::VerifyingMaterial;

        if self.release_signing_key.trim().is_empty() {
            return crate::upgrade::release_key::built_in();
        }
        let bytes =
            crate::upgrade::feed::decode_hex(self.release_signing_key.trim()).ok_or_else(|| {
                ZyronError::ConfigError(
                    "upgrade.release_signing_key must be the verifying key as hex".into(),
                )
            })?;
        let scheme = self.release_signing_scheme.trim();
        let material = match scheme.to_ascii_uppercase().as_str() {
            "ED25519" => {
                let key: [u8; 32] = bytes.as_slice().try_into().map_err(|_| {
                    ZyronError::ConfigError(
                        "upgrade.release_signing_key must be the 32 byte Ed25519 verifying \
                         key as 64 hex characters"
                            .into(),
                    )
                })?;
                VerifyingMaterial::Ed25519(key)
            }
            "ES256" => VerifyingMaterial::Es256(bytes),
            "RS256" => VerifyingMaterial::Rs256(bytes),
            other => {
                return Err(ZyronError::ConfigError(format!(
                    "upgrade.release_signing_scheme \"{other}\" is not a scheme a release key \
                     belongs to, use Ed25519, ES256, or RS256"
                )));
            }
        };
        Ok(material)
    }

    /// The board settings this section describes
    pub fn to_settings(&self, data_dir: &Path) -> zyron_common::format::UpgradeSettings {
        use zyron_common::format::rewrite::UserObjectRewritePolicy;
        use zyron_common::format::{MaintenanceSchedule, UpgradeChannel};

        let channel = match UpgradeChannel::parse(&self.channel) {
            Some(UpgradeChannel::Pinned(_)) => {
                UpgradeChannel::Pinned(self.pinned_version.clone().unwrap_or_default())
            }
            Some(channel) => channel,
            None => UpgradeChannel::Stable,
        };
        zyron_common::format::UpgradeSettings {
            auto_upgrade_enabled: self.auto_upgrade_enabled,
            channel,
            pinned_version: self.pinned_version.clone(),
            window: MaintenanceSchedule::parse(&self.window).unwrap_or_default(),
            paused: self.paused,
            user_object_rewrite_policy: UserObjectRewritePolicy::parse(
                &self.user_object_rewrite_policy,
            )
            .unwrap_or_default(),
            format_migration_budget_memory_fraction: self.format_migration_budget_memory_fraction,
            format_migration_budget_time_secs: self.format_migration_budget_time_secs,
            format_migration_budget_disk_multiple: self.format_migration_budget_disk_multiple,
            rollback_on_health_fail: self.rollback_on_health_fail,
            federation_coordination_timeout_secs: self.federation_coordination_timeout_secs,
            pre_upgrade_backup_snapshot: self.pre_upgrade_backup_snapshot,
            deprecation_warning_rate_limit_per_hour: self.deprecation_warning_rate_limit_per_hour,
            release_feed_poll_interval_secs: self.release_feed_poll_interval_secs,
            health_recovery_timeout_secs: self.health_recovery_timeout_secs,
            release_feed_url: self.release_feed_url(data_dir),
        }
    }

    /// The scheme the effective release key belongs to
    pub fn release_signing_scheme_name(&self) -> String {
        if self.release_signing_key.trim().is_empty() {
            crate::upgrade::release_key::BUILT_IN_SCHEME.to_string()
        } else {
            self.release_signing_scheme.trim().to_string()
        }
    }

    /// The directory an operator delivers releases into with `zyron-ctl
    /// release stage`. It is read ahead of any remote feed, so a release
    /// placed by hand is the one the node sees. The setting names it when
    /// it holds a directory, otherwise it is `releases` under the data
    /// directory
    pub fn local_feed_dir(&self, data_dir: &Path) -> PathBuf {
        let url = self.release_feed_url.trim();
        if url.is_empty() || self.feed_is_remote() {
            data_dir.join("releases")
        } else {
            PathBuf::from(url)
        }
    }

    /// The remote feed, when there is one: the vendor's feed by default,
    /// the configured URL when the setting holds one, and none when the
    /// setting names a directory, which is what an air-gapped node sets
    pub fn remote_feed_url(&self) -> Option<String> {
        let url = self.release_feed_url.trim();
        if url.is_empty() {
            Some(zyron_common::format::UpgradeSettings::default().release_feed_url)
        } else if self.feed_is_remote() {
            Some(url.to_string())
        } else {
            None
        }
    }

    /// The feed the settings report, the remote one when there is one and
    /// the local directory otherwise
    pub fn release_feed_url(&self, data_dir: &Path) -> String {
        self.remote_feed_url()
            .unwrap_or_else(|| self.local_feed_dir(data_dir).display().to_string())
    }

    /// Whether the setting holds a URL rather than a directory
    pub fn feed_is_remote(&self) -> bool {
        let url = self.release_feed_url.trim().to_ascii_lowercase();
        url.starts_with("http://") || url.starts_with("https://")
    }
}

/// External tool paths for media operations. An unset path makes the
/// operation answer with an actionable error naming this key
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct MediaSection {
    pub ffmpeg_path: Option<PathBuf>,
    pub tesseract_path: Option<PathBuf>,
}

impl Default for ZyronConfig {
    fn default() -> Self {
        Self {
            server: ServerSection::default(),
            storage: StorageSection::default(),
            wal: WalSection::default(),
            checkpoint: CheckpointSection::default(),
            auth: AuthSection::default(),
            logging: LoggingSection::default(),
            metrics: MetricsSection::default(),
            compaction: CompactionSection::default(),
            vacuum: VacuumSection::default(),
            query: QuerySection::default(),
            mesh: zyron_pressure::provisioner::MeshSection::default(),
            cluster: ClusterSection::default(),
            media: MediaSection::default(),
            upgrade: UpgradeSection::default(),
            crypto: CryptoSection::default(),
            cdc: CdcSection::default(),
            verify: VerifySection::default(),
        }
    }
}

/// One member of the consensus group.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct ClusterPeerSection {
    /// The name an operator gave the node. The consensus id is derived from
    /// it, so every member arrives at the same id for it without asking
    pub name: String,
    /// Where that node serves consensus, as `host:port`
    pub address: String,
}

/// The consensus group, described entirely by configuration.
///
/// Nothing about the group is discovered. A node that guessed at its own
/// membership could form a second group beside the real one, so every member
/// including this node is listed, and this node's own name has to be among
/// them
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ClusterSection {
    pub enabled: bool,
    /// This node's name, which must appear in `peers`
    pub node_name: String,
    /// Where this node serves consensus
    pub listen: String,
    /// Every member of the group, this node included
    pub peers: Vec<ClusterPeerSection>,
    /// Stretches the timers for a group whose members are in different
    /// regions, where a round trip is tens of milliseconds
    pub multi_region: bool,
    /// How far the log may grow past the last snapshot, in entries
    pub snapshot_threshold: u64,
    /// How far the log may grow past the last snapshot, in bytes. Zero keeps
    /// the built-in figure. Counting entries alone is right for a log of keys
    /// and wrong for a log of transactions
    pub snapshot_threshold_bytes: u64,
    /// Bytes of decoded log entries held in memory. Zero keeps the built-in
    /// figure. Past this the oldest are read back from the file when a
    /// follower asks for them
    pub resident_log_bytes: u64,
    /// Bytes a transaction buffers before a chunk of it is replicated. A bulk
    /// load past this replicates while it runs rather than arriving as one
    /// entry the size of the load
    pub chunk_bytes: u64,
    /// The table committed Put and Delete commands are applied to
    pub replicated_table: String,
}

impl Default for ClusterSection {
    fn default() -> Self {
        Self {
            enabled: false,
            node_name: String::new(),
            listen: "0.0.0.0:5434".to_string(),
            peers: Vec::new(),
            multi_region: false,
            snapshot_threshold: 10_000,
            snapshot_threshold_bytes: 0,
            resident_log_bytes: 0,
            chunk_bytes: 1024 * 1024,
            replicated_table: "raft_kv".to_string(),
        }
    }
}

impl ClusterSection {
    /// Refuses a group description that cannot produce a working node.
    ///
    /// Checked at startup rather than on the first election, because a node
    /// that is not in its own group would come up, campaign for nothing, and
    /// look like a network fault
    pub fn validate(&self) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        if self.node_name.trim().is_empty() {
            return Err(ZyronError::ConfigError(
                "cluster.node_name is required when cluster.enabled is true".into(),
            ));
        }
        if self.listen.trim().is_empty() {
            return Err(ZyronError::ConfigError(
                "cluster.listen is required when cluster.enabled is true".into(),
            ));
        }
        if self.peers.is_empty() {
            return Err(ZyronError::ConfigError(
                "cluster.peers must name every member of the group, this node included".into(),
            ));
        }
        for peer in &self.peers {
            if peer.name.trim().is_empty() || peer.address.trim().is_empty() {
                return Err(ZyronError::ConfigError(
                    "every entry in cluster.peers needs a name and an address".into(),
                ));
            }
        }
        if !self.peers.iter().any(|p| p.name == self.node_name) {
            return Err(ZyronError::ConfigError(format!(
                "cluster.node_name \"{}\" is not among cluster.peers, so this node is not in its own group",
                self.node_name
            )));
        }
        if self.replicated_table.trim().is_empty() {
            return Err(ZyronError::ConfigError(
                "cluster.replicated_table cannot be empty".into(),
            ));
        }
        Ok(())
    }
}

/// What a config still carrying the removed connection cap is told.
const REMOVED_MAX_CONNECTIONS: &str = "server.max_connections has been removed. \
Connections are bounded by the memory the node measured, and the live ceiling \
is in zyron_sys.pressure.node_capabilities";

impl ZyronConfig {
    /// Loads configuration from a TOML file at the given path.
    /// Falls back to defaults for any missing fields.
    pub fn load(path: &Path) -> Result<Self> {
        let contents = std::fs::read_to_string(path).map_err(|e| {
            ZyronError::Internal(format!(
                "Failed to read config file {}: {}",
                path.display(),
                e
            ))
        })?;
        Self::from_toml(&contents, path)
    }

    /// Parses a config document, checking its declared format envelope
    /// before any key is read.
    ///
    /// A file with no `[format]` section is at the version this binary
    /// writes: the section is stamped the first time the file is rewritten,
    /// which `ALTER SYSTEM` does, and a hand-written config is not expected
    /// to carry one. A section that names a different version fails closed,
    /// because a value whose meaning changed would otherwise be read under
    /// the wrong one.
    pub fn from_toml(contents: &str, path: &Path) -> Result<Self> {
        match text_envelope::parse(contents) {
            Ok((kind, version)) => {
                if kind != FormatKind::ZyronTomlConfig {
                    return Err(ZyronError::Internal(format!(
                        "{} declares kind {}, which is not a server configuration",
                        path.display(),
                        kind
                    )));
                }
                if version != crate::format::CONFIG_FORMAT_VERSION {
                    return Err(ZyronError::Internal(format!(
                        "{} is at format version {}, this binary writes and reads {}. Upgrade \
                         through a release that still reads {} to move it forward first",
                        path.display(),
                        version,
                        crate::format::CONFIG_FORMAT_VERSION,
                        version
                    )));
                }
            }
            Err(zyron_common::format::TextEnvelopeError::Missing) => {}
            Err(e) => {
                return Err(ZyronError::Internal(format!(
                    "{} has an unreadable [format] section, {e}",
                    path.display()
                )));
            }
        }
        let config: ZyronConfig = toml::from_str(contents)
            .map_err(|e| ZyronError::Internal(format!("Failed to parse config file: {}", e)))?;
        config.validate()?;
        Ok(config)
    }

    /// Stamps a config document with the current format envelope, which is
    /// what a rewrite does so the version on disk always names itself.
    pub fn stamp_format(contents: &str) -> String {
        text_envelope::with_header(
            FormatKind::ZyronTomlConfig,
            crate::format::CONFIG_FORMAT_VERSION,
            contents,
        )
    }

    /// Loads configuration with the following priority:
    /// 1. Defaults
    /// 2. Explicit path or ./zyron.toml
    /// 3. zyron.auto.conf (persistent ALTER SYSTEM overrides)
    /// 4. ZYRON_* environment variables
    pub fn load_with_overrides(path: Option<&Path>) -> Result<Self> {
        let mut config = if let Some(p) = path {
            Self::load(p)?
        } else {
            let default_path = PathBuf::from("zyron.toml");
            if default_path.exists() {
                Self::load(&default_path)?
            } else {
                Self::default()
            }
        };

        // Apply persistent overrides from ALTER SYSTEM
        config.apply_auto_conf(&config.storage.data_dir.clone())?;
        config.apply_env_overrides();
        config.validate()?;
        Ok(config)
    }

    /// Loads zyron.auto.conf (TOML fragment) from the data directory and merges overrides.
    /// This file is written by ALTER SYSTEM SET commands.
    pub fn apply_auto_conf(&mut self, data_dir: &Path) -> Result<()> {
        let auto_path = data_dir.join("zyron.auto.conf");
        if !auto_path.exists() {
            return Ok(());
        }
        let contents = std::fs::read_to_string(&auto_path).map_err(|e| {
            ZyronError::Internal(format!(
                "Failed to read auto.conf {}: {}",
                auto_path.display(),
                e
            ))
        })?;
        // The declared envelope decides which reader runs, the same as for
        // zyron.toml. An override file with no section predates the stamp and
        // is at the version this binary writes
        match text_envelope::parse(&contents) {
            Ok((kind, version)) => {
                if kind != FormatKind::ZyronTomlConfig
                    || version != crate::format::CONFIG_FORMAT_VERSION
                {
                    return Err(ZyronError::Internal(format!(
                        "{} declares {kind} at format version {version}, this binary writes \
                         and reads {} at {}. Upgrade through a release that still reads \
                         {version} to move it forward first",
                        auto_path.display(),
                        FormatKind::ZyronTomlConfig,
                        crate::format::CONFIG_FORMAT_VERSION
                    )));
                }
            }
            Err(zyron_common::format::TextEnvelopeError::Missing) => {}
            Err(e) => {
                return Err(ZyronError::Internal(format!(
                    "{} has an unreadable [format] section, {e}",
                    auto_path.display()
                )));
            }
        }
        let mut overrides: toml::Table = toml::from_str(&contents)
            .map_err(|e| ZyronError::Internal(format!("Failed to parse zyron.auto.conf: {}", e)))?;
        // The envelope is the file's own identity, not an override
        overrides.remove(zyron_common::format::TEXT_ENVELOPE_SECTION);
        self.apply_overrides_from_table(&overrides).map_err(|e| {
            ZyronError::Internal(format!(
                "{} refers to {}, fix or remove the entry to boot",
                auto_path.display(),
                e
            ))
        })
    }

    /// Applies dotted key overrides from a TOML table. An unknown section
    /// or key, a non-scalar value, or an unparseable value is an error, a
    /// persisted override that cannot take effect must never vanish silently
    fn apply_overrides_from_table(&mut self, table: &toml::Table) -> Result<()> {
        for (section, value) in table {
            let toml::Value::Table(sub) = value else {
                return Err(ZyronError::Internal(format!(
                    "config section '{}' which is not a table of keys",
                    section
                )));
            };
            for (key, val) in sub {
                let val_str = match val {
                    toml::Value::String(s) => s.clone(),
                    toml::Value::Integer(i) => i.to_string(),
                    toml::Value::Float(f) => f.to_string(),
                    toml::Value::Boolean(b) => b.to_string(),
                    other => {
                        return Err(ZyronError::Internal(format!(
                            "config key {}.{} holding a {} value, expected a scalar",
                            section,
                            key,
                            other.type_str()
                        )));
                    }
                };
                self.set_config_value(section, key, &val_str)?;
            }
        }
        Ok(())
    }

    /// Applies one "section.field" override, refusing unknown keys and
    /// unparseable values.
    pub fn apply_override(&mut self, key: &str, value: &str) -> Result<()> {
        let Some((section, field)) = key.split_once('.') else {
            return Err(ZyronError::Internal(format!(
                "invalid config key format '{}', expected 'section.field'",
                key
            )));
        };
        self.set_config_value(section, field, value)
    }

    /// Sets a config value by section and key name. Unknown sections,
    /// unknown keys, and unparseable values are errors.
    fn set_config_value(&mut self, section: &str, key: &str, value: &str) -> Result<()> {
        fn parsed<T: std::str::FromStr>(section: &str, key: &str, value: &str) -> Result<T> {
            value.parse().map_err(|_| {
                ZyronError::Internal(format!(
                    "invalid value '{}' for config key {}.{}",
                    value, section, key
                ))
            })
        }
        fn parsed_size(section: &str, key: &str, value: &str) -> Result<usize> {
            parse_size(value).map_err(|_| {
                ZyronError::Internal(format!(
                    "invalid size '{}' for config key {}.{}",
                    value, section, key
                ))
            })
        }
        match section {
            "server" => match key {
                "host" => self.server.host = value.into(),
                "port" => self.server.port = parsed(section, key, value)?,
                "max_connections" => {
                    return Err(ZyronError::Internal(REMOVED_MAX_CONNECTIONS.into()));
                }
                "connection_timeout_secs" => {
                    self.server.connection_timeout_secs = parsed(section, key, value)?;
                }
                "worker_threads" => self.server.worker_threads = parsed(section, key, value)?,
                "tls_enabled" => self.server.tls_enabled = parsed(section, key, value)?,
                "tls_cert_path" => self.server.tls_cert_path = Some(PathBuf::from(value)),
                "tls_key_path" => self.server.tls_key_path = Some(PathBuf::from(value)),
                "quic_enabled" => self.server.quic_enabled = parsed(section, key, value)?,
                "quic_port" => self.server.quic_port = Some(parsed(section, key, value)?),
                "quic_zero_rtt" => self.server.quic_zero_rtt = parsed(section, key, value)?,
                "quic_idle_timeout_secs" => {
                    self.server.quic_idle_timeout_secs = parsed(section, key, value)?;
                }
                "dual_stack" => self.server.dual_stack = parsed(section, key, value)?,
                _ => return unknown_key(section, key),
            },
            "storage" => match key {
                "data_dir" => self.storage.data_dir = PathBuf::from(value),
                "page_size" => self.storage.page_size = parsed(section, key, value)?,
                "buffer_pool_size" => {
                    self.storage.buffer_pool_size = parsed_size(section, key, value)?;
                }
                "deployment_mode" => self.storage.deployment_mode = value.into(),
                "page_checksum_verify" => self.storage.page_checksum_verify = value.into(),
                _ => return unknown_key(section, key),
            },
            "wal" => match key {
                "wal_dir" => self.wal.wal_dir = Some(PathBuf::from(value)),
                "segment_size" => self.wal.segment_size = parsed_size(section, key, value)?,
                "sync_mode" => self.wal.sync_mode = value.into(),
                "ring_buffer_capacity" => {
                    self.wal.ring_buffer_capacity = parsed_size(section, key, value)?;
                }
                _ => return unknown_key(section, key),
            },
            "checkpoint" => match key {
                "wal_bytes_threshold" => {
                    self.checkpoint.wal_bytes_threshold = parse_size_u64(value).map_err(|_| {
                        ZyronError::Internal(format!(
                            "invalid size '{}' for config key {}.{}",
                            value, section, key
                        ))
                    })?;
                }
                "max_interval_secs" => {
                    self.checkpoint.max_interval_secs = parsed(section, key, value)?;
                }
                "min_interval_secs" => {
                    self.checkpoint.min_interval_secs = parsed(section, key, value)?;
                }
                _ => return unknown_key(section, key),
            },
            "auth" => match key {
                "method" => self.auth.method = value.into(),
                "password_encryption" => self.auth.password_encryption = value.into(),
                "balloon_space_cost" => {
                    self.auth.balloon_space_cost = Some(parsed(section, key, value)?);
                }
                "balloon_time_cost" => {
                    self.auth.balloon_time_cost = Some(parsed(section, key, value)?);
                }
                "jwt_secret" => self.auth.jwt_secret = Some(value.into()),
                "jwt_algorithm" => self.auth.jwt_algorithm = Some(value.into()),
                "jwt_issuer" => self.auth.jwt_issuer = Some(value.into()),
                "brute_force_enabled" => {
                    self.auth.brute_force_enabled = Some(parsed(section, key, value)?);
                }
                "lockout_threshold" => {
                    self.auth.lockout_threshold = Some(parsed(section, key, value)?);
                }
                "lockout_duration_secs" => {
                    self.auth.lockout_duration_secs = Some(parsed(section, key, value)?);
                }
                "ip_block_threshold" => {
                    self.auth.ip_block_threshold = Some(parsed(section, key, value)?);
                }
                "failure_window_secs" => {
                    self.auth.failure_window_secs = Some(parsed(section, key, value)?);
                }
                "ip_block_duration_secs" => {
                    self.auth.ip_block_duration_secs = Some(parsed(section, key, value)?);
                }
                "min_attempt_interval_ms" => {
                    self.auth.min_attempt_interval_ms = Some(parsed(section, key, value)?);
                }
                "webauthn_rp_id" => self.auth.webauthn_rp_id = Some(value.into()),
                "webauthn_rp_name" => self.auth.webauthn_rp_name = Some(value.into()),
                "webauthn_origin" => self.auth.webauthn_origin = Some(value.into()),
                "webauthn_challenge_timeout" => {
                    self.auth.webauthn_challenge_timeout = Some(parsed(section, key, value)?);
                }
                "tls_required" => self.auth.tls_required = parsed(section, key, value)?,
                _ => return unknown_key(section, key),
            },
            "logging" => match key {
                "level" => self.logging.level = value.into(),
                "format" => self.logging.format = value.into(),
                "output" => self.logging.output = value.into(),
                "file_path" => self.logging.file_path = Some(PathBuf::from(value)),
                _ => return unknown_key(section, key),
            },
            "media" => match key {
                "ffmpeg_path" => self.media.ffmpeg_path = Some(PathBuf::from(value)),
                "tesseract_path" => self.media.tesseract_path = Some(PathBuf::from(value)),
                _ => return unknown_key(section, key),
            },
            "metrics" => match key {
                "enabled" => self.metrics.enabled = parsed(section, key, value)?,
                "host" => self.metrics.host = value.into(),
                "port" => self.metrics.port = parsed(section, key, value)?,
                "path" => self.metrics.path = value.into(),
                "dual_stack" => self.metrics.dual_stack = parsed(section, key, value)?,
                _ => return unknown_key(section, key),
            },
            "compaction" => match key {
                "enabled" => self.compaction.enabled = parsed(section, key, value)?,
                "threshold_rows" => {
                    self.compaction.threshold_rows = parsed(section, key, value)?;
                }
                "max_concurrent" => self.compaction.max_concurrent = parsed(section, key, value)?,
                "rate_limit_mbps" => {
                    self.compaction.rate_limit_mbps = parsed(section, key, value)?;
                }
                "interval_secs" => self.compaction.interval_secs = parsed(section, key, value)?,
                "oltp_p99_threshold_us" => {
                    self.compaction.oltp_p99_threshold_us = parsed(section, key, value)?;
                }
                "max_rows_per_file" => {
                    self.compaction.max_rows_per_file = parsed(section, key, value)?;
                }
                _ => return unknown_key(section, key),
            },
            "vacuum" => match key {
                "enabled" => self.vacuum.enabled = parsed(section, key, value)?,
                "interval_secs" => self.vacuum.interval_secs = parsed(section, key, value)?,
                "dead_tuple_threshold" => {
                    self.vacuum.dead_tuple_threshold = parsed(section, key, value)?;
                }
                _ => return unknown_key(section, key),
            },
            "query" => match key {
                "default_isolation" => self.query.default_isolation = value.into(),
                "statement_timeout_secs" => {
                    self.query.statement_timeout_secs = parsed(section, key, value)?;
                }
                "max_result_rows" => self.query.max_result_rows = parsed(section, key, value)?,
                "max_memory_bytes" => {
                    self.query.max_memory_bytes = parse_size_u64(value).map_err(|_| {
                        ZyronError::Internal(format!(
                            "invalid size '{}' for config key {}.{}",
                            value, section, key
                        ))
                    })?;
                }
                _ => return unknown_key(section, key),
            },
            "mesh" => match key {
                "node_registration_mode" => {
                    self.mesh.node_registration_mode = value.into();
                }
                "warm_pool_max_nodes" => {
                    self.mesh.warm_pool_max_nodes = parsed(section, key, value)?;
                }
                "provision_latency_secs" => {
                    self.mesh.provision_latency_secs = parsed(section, key, value)?;
                }
                "hot_set_pages" => self.mesh.hot_set_pages = parsed(section, key, value)?,
                "hot_set_queries" => self.mesh.hot_set_queries = parsed(section, key, value)?,
                _ => return unknown_key(section, key),
            },
            "upgrade" => match key {
                "auto_upgrade_enabled" => {
                    self.upgrade.auto_upgrade_enabled = parsed(section, key, value)?;
                }
                "channel" => self.upgrade.channel = value.into(),
                "pinned_version" => {
                    self.upgrade.pinned_version = if value.trim().is_empty() {
                        None
                    } else {
                        Some(value.into())
                    };
                }
                "window" => self.upgrade.window = value.into(),
                "paused" => self.upgrade.paused = parsed(section, key, value)?,
                "user_object_rewrite_policy" => {
                    self.upgrade.user_object_rewrite_policy = value.into();
                }
                "format_migration_budget_memory_fraction" => {
                    self.upgrade.format_migration_budget_memory_fraction =
                        parsed(section, key, value)?;
                }
                "format_migration_budget_time_secs" => {
                    self.upgrade.format_migration_budget_time_secs = parsed(section, key, value)?;
                }
                "format_migration_budget_disk_multiple" => {
                    self.upgrade.format_migration_budget_disk_multiple =
                        parsed(section, key, value)?;
                }
                "rollback_on_health_fail" => {
                    self.upgrade.rollback_on_health_fail = parsed(section, key, value)?;
                }
                "federation_coordination_timeout_secs" => {
                    self.upgrade.federation_coordination_timeout_secs =
                        parsed(section, key, value)?;
                }
                "pre_upgrade_backup_snapshot" => {
                    self.upgrade.pre_upgrade_backup_snapshot = parsed(section, key, value)?;
                }
                "deprecation_warning_rate_limit_per_hour" => {
                    self.upgrade.deprecation_warning_rate_limit_per_hour =
                        parsed(section, key, value)?;
                }
                "release_feed_poll_interval_secs" => {
                    self.upgrade.release_feed_poll_interval_secs = parsed(section, key, value)?;
                }
                "health_recovery_timeout_secs" => {
                    self.upgrade.health_recovery_timeout_secs = parsed(section, key, value)?;
                }
                "health_poll_interval_secs" => {
                    self.upgrade.health_poll_interval_secs = parsed(section, key, value)?;
                }
                "health_latency_multiplier" => {
                    self.upgrade.health_latency_multiplier = parsed(section, key, value)?;
                }
                "health_throughput_floor" => {
                    self.upgrade.health_throughput_floor = parsed(section, key, value)?;
                }
                "health_error_rate_ceiling" => {
                    self.upgrade.health_error_rate_ceiling = parsed(section, key, value)?;
                }
                "drain_timeout_secs" => {
                    self.upgrade.drain_timeout_secs = parsed(section, key, value)?;
                }
                "stage_timeout_secs" => {
                    self.upgrade.stage_timeout_secs = parsed(section, key, value)?;
                }
                "release_feed_url" => self.upgrade.release_feed_url = value.into(),
                "release_signing_key" => self.upgrade.release_signing_key = value.into(),
                "release_signing_scheme" => self.upgrade.release_signing_scheme = value.into(),
                "notify_webhook_url" => self.upgrade.notify_webhook_url = value.into(),
                "notify_slack_webhook_url" => {
                    self.upgrade.notify_slack_webhook_url = value.into();
                }
                // The address is checked here, so a malformed one is
                // refused where it is set and the next restart builds the
                // channel from a value that already passed
                "notify_discord_webhook_url" => {
                    let address = value.trim();
                    if !address.is_empty() {
                        crate::upgrade::notification::ContactChannel::discord(address).map_err(
                            |reason| {
                                ZyronError::Internal(format!(
                                    "invalid value for config key {section}.{key}, {reason}"
                                ))
                            },
                        )?;
                    }
                    self.upgrade.notify_discord_webhook_url = address.into();
                }
                _ => return unknown_key(section, key),
            },
            // One entry per artifact kind, and the kinds belong to the
            // signature registry rather than to this file, so the key is
            // checked against the registry instead of against a list here
            // that would have to be kept in step with it
            "crypto" => {
                if !crate::crypto_settings::is_crypto_setting(&format!("{section}.{key}")) {
                    return Err(ZyronError::Internal(format!(
                        "`{key}` is not an artifact kind, so `{section}.{key}` binds nothing"
                    )));
                }
                self.crypto
                    .artifact_schemes
                    .insert(key.to_ascii_lowercase(), value.to_string());
            }
            _ => {
                return Err(ZyronError::Internal(format!(
                    "unknown config section '{}'",
                    section
                )));
            }
        }
        Ok(())
    }

    /// Writes a single key-value override to zyron.auto.conf.
    /// The key should be in "section.field" format (e.g. "server.port").
    ///
    /// The whole file is read, changed, and written back under one
    /// process-wide lock, because `ALTER SYSTEM SET`, a replicated setting
    /// being applied, and the upgrade service's pause each call this from
    /// their own thread and two at once would lose one key. The write goes
    /// through a sibling file and a rename, so a crash at any point leaves
    /// the previous file or the new one and never a partial one the next
    /// boot refuses
    pub fn write_auto_conf(data_dir: &Path, key: &str, value: &str) -> Result<()> {
        let _writing = AUTO_CONF_WRITE.lock();
        let auto_path = data_dir.join("zyron.auto.conf");
        let mut table: toml::Table = if auto_path.exists() {
            let contents = std::fs::read_to_string(&auto_path).map_err(|e| {
                ZyronError::Internal(format!("{} is not readable, {e}", auto_path.display()))
            })?;
            toml::from_str(&contents).map_err(|e| {
                ZyronError::Internal(format!(
                    "{} is not readable TOML, {e}. Fix or remove the file before setting {key}",
                    auto_path.display()
                ))
            })?
        } else {
            toml::Table::new()
        };

        // Parse "section.field" into nested TOML table
        if let Some((section, field)) = key.split_once('.') {
            let section_table = table
                .entry(section.to_string())
                .or_insert_with(|| toml::Value::Table(toml::Table::new()));
            if let toml::Value::Table(t) = section_table {
                // Try to store as the most specific type
                if let Ok(v) = value.parse::<i64>() {
                    t.insert(field.to_string(), toml::Value::Integer(v));
                } else if let Ok(v) = value.parse::<f64>() {
                    t.insert(field.to_string(), toml::Value::Float(v));
                } else if let Ok(v) = value.parse::<bool>() {
                    t.insert(field.to_string(), toml::Value::Boolean(v));
                } else {
                    t.insert(field.to_string(), toml::Value::String(value.to_string()));
                }
            }
        } else {
            return Err(ZyronError::Internal(format!(
                "Invalid config key format '{}', expected 'section.field'",
                key
            )));
        }

        // The file is rewritten in full on every ALTER SYSTEM, which is what
        // makes the config format's eager policy free: the envelope is
        // stamped here and the next boot reads the version it declares
        table.remove(zyron_common::format::TEXT_ENVELOPE_SECTION);
        let serialized = toml::to_string_pretty(&table)
            .map_err(|e| ZyronError::Internal(format!("Failed to serialize auto.conf: {}", e)))?;
        let stamped = Self::stamp_format(&serialized);
        write_replacing(&auto_path, stamped.as_bytes())
    }

    /// Applies environment variable overrides to the config.
    /// Logs a warning if an env var is set but cannot be parsed.
    fn apply_env_overrides(&mut self) {
        if let Ok(val) = std::env::var("ZYRON_HOST") {
            self.server.host = val;
        }
        if let Ok(val) = std::env::var("ZYRON_PORT") {
            match val.parse() {
                Ok(port) => self.server.port = port,
                Err(_) => {
                    tracing::warn!("ZYRON_PORT='{}' is not a valid port number, ignoring", val)
                }
            }
        }
        if let Ok(val) = std::env::var("ZYRON_DATA_DIR") {
            self.storage.data_dir = PathBuf::from(val);
        }
        if let Ok(val) = std::env::var("ZYRON_WAL_DIR") {
            self.wal.wal_dir = Some(PathBuf::from(val));
        }
        if let Ok(val) = std::env::var("ZYRON_LOG_LEVEL") {
            self.logging.level = val;
        }
        if std::env::var("ZYRON_MAX_CONNECTIONS").is_ok() {
            tracing::warn!("{}", REMOVED_MAX_CONNECTIONS);
        }
        if let Ok(val) = std::env::var("ZYRON_BUFFER_POOL_SIZE") {
            match parse_size(&val) {
                Ok(v) => self.storage.buffer_pool_size = v,
                Err(_) => tracing::warn!("ZYRON_BUFFER_POOL_SIZE='{}' is not valid, ignoring", val),
            }
        }
        if let Ok(val) = std::env::var("ZYRON_DEPLOYMENT_MODE") {
            match zyron_common::DeploymentMode::parse(&val) {
                Some(mode) => self.storage.deployment_mode = mode.as_str().into(),
                None => tracing::warn!(
                    "ZYRON_DEPLOYMENT_MODE='{}' is not 'db', 'lake' or 'unified', ignoring",
                    val
                ),
            }
        }
        if let Ok(val) = std::env::var("ZYRON_METRICS_ENABLED") {
            if let Ok(v) = val.parse() {
                self.metrics.enabled = v;
            }
        }
        if let Ok(val) = std::env::var("ZYRON_METRICS_PORT") {
            match val.parse() {
                Ok(v) => self.metrics.port = v,
                Err(_) => tracing::warn!("ZYRON_METRICS_PORT='{}' is not valid, ignoring", val),
            }
        }
        if let Ok(val) = std::env::var("ZYRON_VACUUM_ENABLED") {
            if let Ok(v) = val.parse() {
                self.vacuum.enabled = v;
            }
        }
        if let Ok(val) = std::env::var("ZYRON_VACUUM_INTERVAL") {
            match val.parse() {
                Ok(v) => self.vacuum.interval_secs = v,
                Err(_) => tracing::warn!("ZYRON_VACUUM_INTERVAL='{}' is not valid, ignoring", val),
            }
        }
        if let Ok(val) = std::env::var("ZYRON_COMPACTION_ENABLED") {
            if let Ok(v) = val.parse() {
                self.compaction.enabled = v;
            }
        }
        if let Ok(val) = std::env::var("ZYRON_QUERY_STATEMENT_TIMEOUT") {
            match val.parse() {
                Ok(v) => self.query.statement_timeout_secs = v,
                Err(_) => tracing::warn!(
                    "ZYRON_QUERY_STATEMENT_TIMEOUT='{}' is not valid, ignoring",
                    val
                ),
            }
        }
        if let Ok(val) = std::env::var("ZYRON_QUERY_MAX_RESULT_ROWS") {
            match val.parse() {
                Ok(v) => self.query.max_result_rows = v,
                Err(_) => tracing::warn!(
                    "ZYRON_QUERY_MAX_RESULT_ROWS='{}' is not valid, ignoring",
                    val
                ),
            }
        }
        if let Ok(val) = std::env::var("ZYRON_LOGGING_OUTPUT") {
            self.logging.output = val;
        }
        if let Ok(val) = std::env::var("ZYRON_LOGGING_FILE_PATH") {
            self.logging.file_path = Some(PathBuf::from(val));
        }
        if let Ok(val) = std::env::var("ZYRON_AUTH_TLS_REQUIRED") {
            if let Ok(v) = val.parse() {
                self.auth.tls_required = v;
            }
        }
    }

    /// Validates the config for logical consistency.
    pub(crate) fn validate(&self) -> Result<()> {
        if self.server.port == 0 {
            return Err(ZyronError::Internal("Server port cannot be 0".into()));
        }
        if self.server.max_connections.is_some() {
            return Err(ZyronError::Internal(REMOVED_MAX_CONNECTIONS.into()));
        }
        self.mesh.validate()?;
        self.upgrade.validate()?;
        if !matches!(
            self.auth.password_encryption.as_str(),
            "balloon-sha-256" | "scram-sha-256" | "md5"
        ) {
            return Err(ZyronError::Internal(format!(
                "auth.password_encryption '{}' is not one of 'balloon-sha-256', 'scram-sha-256' or 'md5'",
                self.auth.password_encryption
            )));
        }
        if self.server.worker_threads == 0 {
            return Err(ZyronError::Internal("worker_threads cannot be 0".into()));
        }
        if self.storage.buffer_pool_size == 0 {
            return Err(ZyronError::Internal("buffer_pool_size cannot be 0".into()));
        }
        if zyron_common::DeploymentMode::parse(&self.storage.deployment_mode).is_none() {
            return Err(ZyronError::Internal(format!(
                "Invalid storage.deployment_mode '{}', expected 'db', 'lake' or 'unified'",
                self.storage.deployment_mode
            )));
        }
        match zyron_storage::PageChecksumVerify::parse(&self.storage.page_checksum_verify) {
            Some(zyron_storage::PageChecksumVerify::Off) => {
                return Err(ZyronError::Internal(
                    "storage.page_checksum_verify = 'off' is not accepted: a server never runs \
                     with an unverified page read path. Use 'always' (the default) or 'sampled'"
                        .into(),
                ));
            }
            Some(_) => {}
            None => {
                return Err(ZyronError::Internal(format!(
                    "Invalid storage.page_checksum_verify '{}', expected 'always' or 'sampled'",
                    self.storage.page_checksum_verify
                )));
            }
        }
        if self.wal.segment_size == 0 {
            return Err(ZyronError::Internal("WAL segment_size cannot be 0".into()));
        }
        // A ring smaller than a segment can fill with one segment's records
        // and deadlock rotation against producers waiting for ring space
        if self.wal.ring_buffer_capacity < self.wal.segment_size {
            return Err(ZyronError::Internal(format!(
                "wal.ring_buffer_capacity ({}) must be at least wal.segment_size ({})",
                self.wal.ring_buffer_capacity, self.wal.segment_size
            )));
        }
        if self.checkpoint.wal_bytes_threshold == 0 {
            return Err(ZyronError::Internal(
                "wal_bytes_threshold cannot be 0".into(),
            ));
        }
        if self.checkpoint.min_interval_secs >= self.checkpoint.max_interval_secs {
            return Err(ZyronError::Internal(
                "checkpoint min_interval_secs must be less than max_interval_secs".into(),
            ));
        }
        if self.server.tls_enabled {
            if self.server.tls_cert_path.is_none() || self.server.tls_key_path.is_none() {
                return Err(ZyronError::Internal(
                    "TLS enabled but tls_cert_path or tls_key_path not set".into(),
                ));
            }
        }
        // WAL sync mode validation
        match self.wal.sync_mode.as_str() {
            "fsync" | "fdatasync" | "none" => {}
            other => {
                return Err(ZyronError::Internal(format!(
                    "Invalid wal.sync_mode '{}', expected 'fsync', 'fdatasync', or 'none'",
                    other
                )));
            }
        }
        // Compaction section
        if self.compaction.max_concurrent == 0 {
            return Err(ZyronError::Internal(
                "compaction.max_concurrent must be at least 1".into(),
            ));
        }
        if self.compaction.rate_limit_mbps == 0 {
            return Err(ZyronError::Internal(
                "compaction.rate_limit_mbps must be greater than 0".into(),
            ));
        }
        // Vacuum section
        if !(0.0..=1.0).contains(&self.vacuum.dead_tuple_threshold) {
            return Err(ZyronError::Internal(
                "vacuum.dead_tuple_threshold must be between 0.0 and 1.0".into(),
            ));
        }
        // Change data feed section
        if self.cdc.change_streams_per_table == 0 {
            return Err(ZyronError::Internal(
                "cdc.change_streams_per_table must be at least 1".into(),
            ));
        }
        if self.cdc.sweep_interval_secs == 0 {
            return Err(ZyronError::Internal(
                "cdc.sweep_interval_secs must be at least 1".into(),
            ));
        }
        // Query section
        match self.query.default_isolation.as_str() {
            "snapshot" | "read_committed" => {}
            other => {
                return Err(ZyronError::Internal(format!(
                    "Invalid query.default_isolation '{}', expected 'snapshot' or 'read_committed'",
                    other
                )));
            }
        }
        // Logging section
        match self.logging.output.as_str() {
            "stdout" | "file" => {}
            other => {
                return Err(ZyronError::Internal(format!(
                    "Invalid logging.output '{}', expected 'stdout' or 'file'",
                    other
                )));
            }
        }
        if self.logging.output == "file" && self.logging.file_path.is_none() {
            return Err(ZyronError::Internal(
                "logging.file_path is required when logging.output = 'file'".into(),
            ));
        }
        // Auth TLS requirement check
        if self.auth.tls_required && !self.server.tls_enabled {
            tracing::warn!("auth.tls_required is true but server.tls_enabled is false");
        }
        Ok(())
    }

    /// Returns the typed deployment mode. Validation restricts the source
    /// string at load, so an unrecognized value here falls back to the
    /// engine default rather than failing a running server.
    pub fn deployment_mode(&self) -> zyron_common::DeploymentMode {
        zyron_common::DeploymentMode::parse(&self.storage.deployment_mode).unwrap_or_default()
    }

    /// Converts the server section to the common ServerConfig.
    pub fn to_server_config(&self) -> zyron_common::ServerConfig {
        zyron_common::ServerConfig {
            host: self.server.host.clone(),
            port: self.server.port,
            connection_timeout_secs: self.server.connection_timeout_secs,
            worker_threads: self.server.worker_threads,
            tls_enabled: self.server.tls_enabled,
            tls_cert_path: self.server.tls_cert_path.clone(),
            tls_key_path: self.server.tls_key_path.clone(),
            quic_enabled: self.server.quic_enabled,
            quic_port: self.server.quic_port,
            quic_zero_rtt: self.server.quic_zero_rtt,
            quic_idle_timeout_secs: self.server.quic_idle_timeout_secs,
            dual_stack: self.server.dual_stack,
        }
    }

    /// Converts the storage section to the common StorageConfig.
    pub fn to_storage_config(&self) -> zyron_common::StorageConfig {
        let wal_dir = self
            .wal
            .wal_dir
            .clone()
            .unwrap_or_else(|| self.storage.data_dir.join("wal"));
        zyron_common::StorageConfig {
            data_dir: self.storage.data_dir.clone(),
            wal_dir,
            page_size: self.storage.page_size,
            buffer_pool_pages: self.storage.buffer_pool_size / self.storage.page_size,
            wal_segment_size: self.wal.segment_size,
            checkpoint_interval_secs: self.checkpoint.max_interval_secs,
            fsync_enabled: self.wal.sync_mode == "fsync",
            direct_io: false,
        }
    }

    /// Returns the effective WAL directory.
    pub fn wal_dir(&self) -> PathBuf {
        self.wal
            .wal_dir
            .clone()
            .unwrap_or_else(|| self.storage.data_dir.join("wal"))
    }

    /// Looks up a config value by dotted key (e.g. "server.port").
    /// Returns the current value as a string, or None if the key is not recognized.
    pub fn get_config_value(&self, key: &str) -> Option<String> {
        match key {
            // Server
            "server.host" => Some(self.server.host.clone()),
            "server.port" => Some(self.server.port.to_string()),
            "server.connection_timeout_secs" => {
                Some(self.server.connection_timeout_secs.to_string())
            }
            "server.worker_threads" => Some(self.server.worker_threads.to_string()),
            "server.tls_enabled" => Some(self.server.tls_enabled.to_string()),
            "server.dual_stack" => Some(self.server.dual_stack.to_string()),
            "metrics.host" => Some(self.metrics.host.clone()),
            "metrics.dual_stack" => Some(self.metrics.dual_stack.to_string()),
            // Storage
            "storage.data_dir" => Some(self.storage.data_dir.display().to_string()),
            "storage.page_size" => Some(self.storage.page_size.to_string()),
            "storage.buffer_pool_size" => Some(self.storage.buffer_pool_size.to_string()),
            "storage.deployment_mode" => Some(self.storage.deployment_mode.clone()),
            // WAL
            "wal.wal_dir" => Some(self.wal_dir().display().to_string()),
            "wal.segment_size" => Some(self.wal.segment_size.to_string()),
            "wal.sync_mode" => Some(self.wal.sync_mode.clone()),
            "wal.ring_buffer_capacity" => Some(self.wal.ring_buffer_capacity.to_string()),
            // Checkpoint
            "checkpoint.wal_bytes_threshold" => {
                Some(self.checkpoint.wal_bytes_threshold.to_string())
            }
            "checkpoint.max_interval_secs" => Some(self.checkpoint.max_interval_secs.to_string()),
            "checkpoint.min_interval_secs" => Some(self.checkpoint.min_interval_secs.to_string()),
            // Auth
            "auth.method" => Some(self.auth.method.clone()),
            "auth.password_encryption" => Some(self.auth.password_encryption.clone()),
            "auth.tls_required" => Some(self.auth.tls_required.to_string()),
            // Logging
            "logging.level" => Some(self.logging.level.clone()),
            "logging.format" => Some(self.logging.format.clone()),
            "logging.output" => Some(self.logging.output.clone()),
            "logging.file_path" => Some(
                self.logging
                    .file_path
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_default(),
            ),
            // Metrics
            "metrics.enabled" => Some(self.metrics.enabled.to_string()),
            "metrics.port" => Some(self.metrics.port.to_string()),
            "metrics.path" => Some(self.metrics.path.clone()),
            // Media tooling
            "media.ffmpeg_path" => self
                .media
                .ffmpeg_path
                .as_ref()
                .map(|p| p.display().to_string()),
            "media.tesseract_path" => self
                .media
                .tesseract_path
                .as_ref()
                .map(|p| p.display().to_string()),
            // Compaction
            "compaction.enabled" => Some(self.compaction.enabled.to_string()),
            "compaction.threshold_rows" => Some(self.compaction.threshold_rows.to_string()),
            "compaction.max_concurrent" => Some(self.compaction.max_concurrent.to_string()),
            "compaction.rate_limit_mbps" => Some(self.compaction.rate_limit_mbps.to_string()),
            // Vacuum
            "vacuum.enabled" => Some(self.vacuum.enabled.to_string()),
            "vacuum.interval_secs" => Some(self.vacuum.interval_secs.to_string()),
            "vacuum.dead_tuple_threshold" => Some(self.vacuum.dead_tuple_threshold.to_string()),
            // Change data feeds and streams
            "cdc.cdf_max_bytes_per_table" => Some(self.cdc.cdf_max_bytes_per_table.to_string()),
            "cdc.cdf_max_retention_secs" => Some(self.cdc.cdf_max_retention_secs.to_string()),
            "verify.anchor_interval_secs" => Some(self.verify.anchor_interval_secs.to_string()),
            "verify.default_sample" => Some(self.verify.default_sample.to_string()),
            "cdc.change_streams_per_table" => Some(self.cdc.change_streams_per_table.to_string()),
            "cdc.stream_lag_rows" => Some(self.cdc.stream_lag_rows.to_string()),
            "cdc.stream_lag_seconds" => Some(self.cdc.stream_lag_seconds.to_string()),
            "cdc.retention_margin_secs" => Some(self.cdc.retention_margin_secs.to_string()),
            "cdc.sweep_interval_secs" => Some(self.cdc.sweep_interval_secs.to_string()),
            // Query
            "query.default_isolation" => Some(self.query.default_isolation.clone()),
            "query.statement_timeout_secs" => Some(self.query.statement_timeout_secs.to_string()),
            "query.max_result_rows" => Some(self.query.max_result_rows.to_string()),
            "query.max_memory_bytes" => Some(self.query.max_memory_bytes.to_string()),
            "mesh.node_registration_mode" => Some(self.mesh.node_registration_mode.clone()),
            "mesh.warm_pool_max_nodes" => Some(self.mesh.warm_pool_max_nodes.to_string()),
            "mesh.provision_latency_secs" => Some(self.mesh.provision_latency_secs.to_string()),
            "mesh.hot_set_pages" => Some(self.mesh.hot_set_pages.to_string()),
            "mesh.hot_set_queries" => Some(self.mesh.hot_set_queries.to_string()),
            "upgrade.auto_upgrade_enabled" => Some(self.upgrade.auto_upgrade_enabled.to_string()),
            "upgrade.channel" => Some(self.upgrade.channel.clone()),
            "upgrade.pinned_version" => {
                Some(self.upgrade.pinned_version.clone().unwrap_or_default())
            }
            "upgrade.window" => Some(self.upgrade.window.clone()),
            "upgrade.paused" => Some(self.upgrade.paused.to_string()),
            "upgrade.user_object_rewrite_policy" => {
                Some(self.upgrade.user_object_rewrite_policy.clone())
            }
            "upgrade.format_migration_budget_memory_fraction" => Some(
                self.upgrade
                    .format_migration_budget_memory_fraction
                    .to_string(),
            ),
            "upgrade.format_migration_budget_time_secs" => {
                Some(self.upgrade.format_migration_budget_time_secs.to_string())
            }
            "upgrade.format_migration_budget_disk_multiple" => Some(
                self.upgrade
                    .format_migration_budget_disk_multiple
                    .to_string(),
            ),
            "upgrade.rollback_on_health_fail" => {
                Some(self.upgrade.rollback_on_health_fail.to_string())
            }
            "upgrade.federation_coordination_timeout_secs" => Some(
                self.upgrade
                    .federation_coordination_timeout_secs
                    .to_string(),
            ),
            "upgrade.pre_upgrade_backup_snapshot" => {
                Some(self.upgrade.pre_upgrade_backup_snapshot.to_string())
            }
            "upgrade.deprecation_warning_rate_limit_per_hour" => Some(
                self.upgrade
                    .deprecation_warning_rate_limit_per_hour
                    .to_string(),
            ),
            "upgrade.release_feed_poll_interval_secs" => {
                Some(self.upgrade.release_feed_poll_interval_secs.to_string())
            }
            "upgrade.health_recovery_timeout_secs" => {
                Some(self.upgrade.health_recovery_timeout_secs.to_string())
            }
            "upgrade.health_poll_interval_secs" => {
                Some(self.upgrade.health_poll_interval_secs.to_string())
            }
            "upgrade.health_latency_multiplier" => {
                Some(self.upgrade.health_latency_multiplier.to_string())
            }
            "upgrade.health_throughput_floor" => {
                Some(self.upgrade.health_throughput_floor.to_string())
            }
            "upgrade.health_error_rate_ceiling" => {
                Some(self.upgrade.health_error_rate_ceiling.to_string())
            }
            "upgrade.drain_timeout_secs" => Some(self.upgrade.drain_timeout_secs.to_string()),
            "upgrade.stage_timeout_secs" => Some(self.upgrade.stage_timeout_secs.to_string()),
            "upgrade.release_feed_url" => {
                Some(self.upgrade.release_feed_url(&self.storage.data_dir))
            }
            // The key is a secret's public half, but a config value that
            // prints in full is one an operator can compare against the
            // release page, which is the point of publishing it
            "upgrade.release_signing_key" => Some(self.upgrade.release_signing_key.clone()),
            "upgrade.release_signing_scheme" => Some(self.upgrade.release_signing_scheme.clone()),
            "upgrade.notify_webhook_url" => Some(self.upgrade.notify_webhook_url.clone()),
            "upgrade.notify_slack_webhook_url" => {
                Some(self.upgrade.notify_slack_webhook_url.clone())
            }
            "upgrade.notify_discord_webhook_url" => {
                Some(self.upgrade.notify_discord_webhook_url.clone())
            }
            // Also support shorthand aliases
            "server_version" => Some(env!("CARGO_PKG_VERSION").to_string()),
            "port" => Some(self.server.port.to_string()),
            "data_dir" | "data_directory" => Some(self.storage.data_dir.display().to_string()),
            // A scheme binding answers from the config rather than from the
            // registry, so what this reports is what the next boot loads
            other if crate::crypto_settings::is_crypto_setting(other) => other
                .split_once('.')
                .and_then(|(_, field)| self.crypto.artifact_schemes.get(field))
                .cloned(),
            _ => None,
        }
    }

    /// Returns all config entries as (key, value, description) tuples for SHOW ALL.
    pub fn all_config_entries(&self) -> Vec<(String, String, String)> {
        vec![
            (
                "server_version".into(),
                env!("CARGO_PKG_VERSION").into(),
                "Server version".into(),
            ),
            (
                "server.host".into(),
                self.server.host.clone(),
                "Bind address".into(),
            ),
            (
                "server.port".into(),
                self.server.port.to_string(),
                "Listen port".into(),
            ),
            (
                "server.connection_timeout_secs".into(),
                self.server.connection_timeout_secs.to_string(),
                "Idle connection timeout in seconds".into(),
            ),
            (
                "server.worker_threads".into(),
                self.server.worker_threads.to_string(),
                "Worker thread count".into(),
            ),
            (
                "server.tls_enabled".into(),
                self.server.tls_enabled.to_string(),
                "TLS enabled".into(),
            ),
            (
                "server.dual_stack".into(),
                self.server.dual_stack.to_string(),
                "Accept IPv4 connections on IPv6 wildcard binds (V6ONLY off)".into(),
            ),
            (
                "storage.data_dir".into(),
                self.storage.data_dir.display().to_string(),
                "Data file directory".into(),
            ),
            (
                "storage.page_size".into(),
                self.storage.page_size.to_string(),
                "Page size in bytes".into(),
            ),
            (
                "storage.buffer_pool_size".into(),
                self.storage.buffer_pool_size.to_string(),
                "Buffer pool size in bytes".into(),
            ),
            (
                "storage.deployment_mode".into(),
                self.storage.deployment_mode.clone(),
                "Storage tiers this node runs (db, lake or unified)".into(),
            ),
            (
                "wal.wal_dir".into(),
                self.wal_dir().display().to_string(),
                "WAL segment directory".into(),
            ),
            (
                "wal.segment_size".into(),
                self.wal.segment_size.to_string(),
                "WAL segment size in bytes".into(),
            ),
            (
                "wal.sync_mode".into(),
                self.wal.sync_mode.clone(),
                "WAL sync mode (fsync, fdatasync, none)".into(),
            ),
            (
                "wal.ring_buffer_capacity".into(),
                self.wal.ring_buffer_capacity.to_string(),
                "WAL ring buffer capacity in bytes".into(),
            ),
            (
                "checkpoint.wal_bytes_threshold".into(),
                self.checkpoint.wal_bytes_threshold.to_string(),
                "WAL bytes before checkpoint trigger".into(),
            ),
            (
                "checkpoint.max_interval_secs".into(),
                self.checkpoint.max_interval_secs.to_string(),
                "Maximum seconds between checkpoints".into(),
            ),
            (
                "checkpoint.min_interval_secs".into(),
                self.checkpoint.min_interval_secs.to_string(),
                "Minimum seconds between checkpoints".into(),
            ),
            (
                "auth.method".into(),
                self.auth.method.clone(),
                "Authentication method".into(),
            ),
            (
                "auth.password_encryption".into(),
                self.auth.password_encryption.clone(),
                "Password hashing algorithm".into(),
            ),
            (
                "auth.tls_required".into(),
                self.auth.tls_required.to_string(),
                "Require TLS for all connections".into(),
            ),
            (
                "logging.level".into(),
                self.logging.level.clone(),
                "Log level (debug, info, warn, error)".into(),
            ),
            (
                "logging.format".into(),
                self.logging.format.clone(),
                "Log format (text or json)".into(),
            ),
            (
                "logging.output".into(),
                self.logging.output.clone(),
                "Log output (stdout or file)".into(),
            ),
            (
                "metrics.enabled".into(),
                self.metrics.enabled.to_string(),
                "Metrics collection enabled".into(),
            ),
            (
                "metrics.port".into(),
                self.metrics.port.to_string(),
                "Metrics HTTP port".into(),
            ),
            (
                "metrics.path".into(),
                self.metrics.path.clone(),
                "Metrics endpoint path".into(),
            ),
            (
                "compaction.enabled".into(),
                self.compaction.enabled.to_string(),
                "Auto compaction enabled".into(),
            ),
            (
                "compaction.threshold_rows".into(),
                self.compaction.threshold_rows.to_string(),
                "Row count threshold for compaction".into(),
            ),
            (
                "compaction.max_concurrent".into(),
                self.compaction.max_concurrent.to_string(),
                "Maximum concurrent compaction tasks".into(),
            ),
            (
                "compaction.rate_limit_mbps".into(),
                self.compaction.rate_limit_mbps.to_string(),
                "Compaction IO rate limit in MB/s".into(),
            ),
            (
                "vacuum.enabled".into(),
                self.vacuum.enabled.to_string(),
                "Auto vacuum enabled".into(),
            ),
            (
                "vacuum.interval_secs".into(),
                self.vacuum.interval_secs.to_string(),
                "Vacuum check interval in seconds".into(),
            ),
            (
                "vacuum.dead_tuple_threshold".into(),
                self.vacuum.dead_tuple_threshold.to_string(),
                "Dead tuple fraction before vacuum triggers".into(),
            ),
            (
                "cdc.cdf_max_bytes_per_table".into(),
                self.cdc.cdf_max_bytes_per_table.to_string(),
                "Bytes a change feed may hold before it purges oldest first, 0 for no cap".into(),
            ),
            (
                "cdc.cdf_max_retention_secs".into(),
                self.cdc.cdf_max_retention_secs.to_string(),
                "Longest change feed retention in seconds, 0 for no cap".into(),
            ),
            (
                "cdc.change_streams_per_table".into(),
                self.cdc.change_streams_per_table.to_string(),
                "Change streams one table may carry".into(),
            ),
            (
                "cdc.stream_lag_rows".into(),
                self.cdc.stream_lag_rows.to_string(),
                "Pending rows at which cdc_stream_lag fires".into(),
            ),
            (
                "cdc.stream_lag_seconds".into(),
                self.cdc.stream_lag_seconds.to_string(),
                "Age of the oldest unconsumed change at which cdc_stream_lag fires".into(),
            ),
            (
                "cdc.retention_margin_secs".into(),
                self.cdc.retention_margin_secs.to_string(),
                "Seconds before reclamation at which cdf_retention_pressure fires".into(),
            ),
            (
                "cdc.sweep_interval_secs".into(),
                self.cdc.sweep_interval_secs.to_string(),
                "Seconds between staleness sweeps and alert evaluations".into(),
            ),
            (
                "query.default_isolation".into(),
                self.query.default_isolation.clone(),
                "Default transaction isolation level".into(),
            ),
            (
                "query.statement_timeout_secs".into(),
                self.query.statement_timeout_secs.to_string(),
                "Default statement timeout in seconds".into(),
            ),
            (
                "query.max_result_rows".into(),
                self.query.max_result_rows.to_string(),
                "Maximum result rows per query (0 = no limit)".into(),
            ),
            (
                "query.max_memory_bytes".into(),
                self.query.max_memory_bytes.to_string(),
                "Maximum bytes one query may materialize (0 = no limit)".into(),
            ),
            (
                "mesh.node_registration_mode".into(),
                self.mesh.node_registration_mode.clone(),
                "How this deployment acquires machines: none, static, cloud, hypervisor, \
                 kubernetes, or ipmi"
                    .into(),
            ),
            (
                "mesh.warm_pool_max_nodes".into(),
                self.mesh.warm_pool_max_nodes.to_string(),
                "Largest pool of idle nodes the operator will pay to keep ready (0 = none)".into(),
            ),
            (
                "mesh.provision_latency_secs".into(),
                self.mesh.provision_latency_secs.to_string(),
                "Overrides the measured provision latency (0 = use the measurement)".into(),
            ),
            (
                "mesh.hot_set_pages".into(),
                self.mesh.hot_set_pages.to_string(),
                "Page identifiers a draining node hands to its survivors".into(),
            ),
            (
                "mesh.hot_set_queries".into(),
                self.mesh.hot_set_queries.to_string(),
                "Query shapes a draining node hands to its survivors".into(),
            ),
            (
                "upgrade.auto_upgrade_enabled".into(),
                self.upgrade.auto_upgrade_enabled.to_string(),
                "Whether this cluster installs releases from its channel on its own".into(),
            ),
            (
                "upgrade.channel".into(),
                self.upgrade.channel.clone(),
                "Release channel: stable, beta, canary, or pinned".into(),
            ),
            (
                "upgrade.pinned_version".into(),
                self.upgrade.pinned_version.clone().unwrap_or_default(),
                "The version a pinned channel holds at".into(),
            ),
            (
                "upgrade.window".into(),
                self.upgrade.window.clone(),
                "Maintenance windows as HH:MM-HH:MM UTC, comma separated, empty for any time"
                    .into(),
            ),
            (
                "upgrade.paused".into(),
                self.upgrade.paused.to_string(),
                "Whether upgrades are halted, set by an operator or by a failed health check"
                    .into(),
            ),
            (
                "upgrade.user_object_rewrite_policy".into(),
                self.upgrade.user_object_rewrite_policy.clone(),
                "What happens to user objects an upgrade rewrites: auto_safe, notify_all, or \
                 manual_only"
                    .into(),
            ),
            (
                "upgrade.format_migration_budget_memory_fraction".into(),
                self.upgrade
                    .format_migration_budget_memory_fraction
                    .to_string(),
                "Fraction of node memory an eager format sweep may use".into(),
            ),
            (
                "upgrade.format_migration_budget_time_secs".into(),
                self.upgrade.format_migration_budget_time_secs.to_string(),
                "Seconds one format's sweep may run for".into(),
            ),
            (
                "upgrade.format_migration_budget_disk_multiple".into(),
                self.upgrade
                    .format_migration_budget_disk_multiple
                    .to_string(),
                "Multiple of a file's size a sweep may occupy on disk while it works".into(),
            ),
            (
                "upgrade.rollback_on_health_fail".into(),
                self.upgrade.rollback_on_health_fail.to_string(),
                "Whether a node that fails its health check is put back on the previous binary"
                    .into(),
            ),
            (
                "upgrade.federation_coordination_timeout_secs".into(),
                self.upgrade
                    .federation_coordination_timeout_secs
                    .to_string(),
                "Seconds a federated peer has to answer the compatibility gate".into(),
            ),
            (
                "upgrade.pre_upgrade_backup_snapshot".into(),
                self.upgrade.pre_upgrade_backup_snapshot.to_string(),
                "Whether a physical backup is taken before a major version upgrade".into(),
            ),
            (
                "upgrade.deprecation_warning_rate_limit_per_hour".into(),
                self.upgrade
                    .deprecation_warning_rate_limit_per_hour
                    .to_string(),
                "Deprecation warnings emitted per item per hour".into(),
            ),
            (
                "upgrade.release_feed_poll_interval_secs".into(),
                self.upgrade.release_feed_poll_interval_secs.to_string(),
                "Seconds between polls of the release feed".into(),
            ),
            (
                "upgrade.health_recovery_timeout_secs".into(),
                self.upgrade.health_recovery_timeout_secs.to_string(),
                "Seconds a restarted node has to reach the health baseline".into(),
            ),
            (
                "upgrade.health_poll_interval_secs".into(),
                self.upgrade.health_poll_interval_secs.to_string(),
                "Seconds between health observations while a node recovers".into(),
            ),
            (
                "upgrade.health_latency_multiplier".into(),
                self.upgrade.health_latency_multiplier.to_string(),
                "Multiple of the baseline p99 still counted as healthy".into(),
            ),
            (
                "upgrade.health_throughput_floor".into(),
                self.upgrade.health_throughput_floor.to_string(),
                "Fraction of the baseline throughput still counted as healthy".into(),
            ),
            (
                "upgrade.health_error_rate_ceiling".into(),
                self.upgrade.health_error_rate_ceiling.to_string(),
                "Error rate above which a restarted node is unhealthy".into(),
            ),
            (
                "upgrade.drain_timeout_secs".into(),
                self.upgrade.drain_timeout_secs.to_string(),
                "Seconds a node has to finish in-flight work before it restarts".into(),
            ),
            (
                "upgrade.stage_timeout_secs".into(),
                self.upgrade.stage_timeout_secs.to_string(),
                "Seconds a node has to fetch and verify a release once asked".into(),
            ),
            (
                "upgrade.release_feed_url".into(),
                self.upgrade.release_feed_url(&self.storage.data_dir),
                "The signed release feed: the vendor's by default, another URL, or a directory \
                 releases are delivered into by hand"
                    .into(),
            ),
            (
                "upgrade.release_signing_key".into(),
                self.upgrade.release_signing_key.clone(),
                "The public half of the key releases are checked against as hex, empty uses the \
                 key built into this binary"
                    .into(),
            ),
            (
                "upgrade.release_signing_scheme".into(),
                self.upgrade.release_signing_scheme.clone(),
                "The signature scheme a configured release signing key belongs to".into(),
            ),
            (
                "upgrade.notify_webhook_url".into(),
                self.upgrade.notify_webhook_url.clone(),
                "Webhook that receives upgrade notifications, empty for none".into(),
            ),
            (
                "upgrade.notify_slack_webhook_url".into(),
                self.upgrade.notify_slack_webhook_url.clone(),
                "Slack incoming webhook that receives upgrade notifications, empty for none".into(),
            ),
            (
                "upgrade.notify_discord_webhook_url".into(),
                self.upgrade.notify_discord_webhook_url.clone(),
                "Discord channel webhook that receives upgrade notifications as an embed, \
                 empty for none"
                    .into(),
            ),
        ]
    }
}

/// [server] section of the config file.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ServerSection {
    /// Bind address for the main wire protocol listener. Accepts any string
    /// `TcpListener::bind` understands. Examples: `"127.0.0.1"` (IPv4 loopback),
    /// `"0.0.0.0"` (IPv4 wildcard, IPv4 only), `"::1"` (IPv6 loopback),
    /// `"[::]"` (IPv6 wildcard, also accepts IPv4 on dual-stack OSes when
    /// `dual_stack = true`)
    pub host: String,
    pub port: u16,
    pub connection_timeout_secs: u32,
    /// Present only to refuse a config file that still sets it. Connections
    /// are bounded by measured memory now, so a configured count would be a
    /// ceiling the hardware never asked for, and silently ignoring the key
    /// would leave an operator believing they had set one
    #[serde(default, skip_serializing)]
    pub max_connections: Option<toml::Value>,
    pub worker_threads: usize,
    pub tls_enabled: bool,
    pub tls_cert_path: Option<PathBuf>,
    pub tls_key_path: Option<PathBuf>,
    pub quic_enabled: bool,
    pub quic_port: Option<u16>,
    pub quic_zero_rtt: bool,
    pub quic_idle_timeout_secs: u32,
    /// When the bind address is an IPv6 wildcard (`::` or `[::]`), accept
    /// IPv4 connections too via IPv4-mapped IPv6 addresses. Linux kernel
    /// defaults V6ONLY to false; Windows defaults to true. Setting this true
    /// applies V6ONLY=false explicitly via socket2 so behaviour is identical
    /// across platforms. Set false to bind IPv6-only when needed
    pub dual_stack: bool,
}

impl Default for ServerSection {
    fn default() -> Self {
        Self {
            // Dual-stack default so IPv6 clients work out of the box.
            // Operators wanting IPv4-only can set host = "0.0.0.0"
            host: "[::]".into(),
            port: 5432,
            connection_timeout_secs: 30,
            max_connections: None,
            worker_threads: std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(1),
            tls_enabled: false,
            tls_cert_path: None,
            tls_key_path: None,
            // HTTP/3 over QUIC is the primary transport
            quic_enabled: true,
            quic_port: None,
            quic_zero_rtt: false,
            quic_idle_timeout_secs: 300,
            dual_stack: true,
        }
    }
}

/// [storage] section of the config file.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct StorageSection {
    pub data_dir: PathBuf,
    pub page_size: usize,
    /// Buffer pool size in bytes. Accepts human-readable strings via parse_size.
    #[serde(deserialize_with = "deserialize_size")]
    pub buffer_pool_size: usize,
    /// Storage tiers this node runs: "db" for heap tables only, "lake" for
    /// ZyronLake tables only, "unified" for both with heap as the default.
    /// Picks the format CREATE TABLE uses without a USING clause, refuses DDL
    /// naming the other format, and gates lake startup recovery and workers.
    pub deployment_mode: String,
    /// Read-side page checksum verification: "always" verifies every heap
    /// page read, "sampled" verifies roughly one in a hundred, for
    /// benchmark investigations that isolate verification cost. "off" is
    /// rejected by the validator, a server never runs with an unverified
    /// read path. Writes stamp checksums unconditionally.
    pub page_checksum_verify: String,
}

impl Default for StorageSection {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("./data"),
            page_size: 16384,
            buffer_pool_size: 128 * 1024 * 1024, // 128 MB
            deployment_mode: "unified".into(),
            page_checksum_verify: "always".into(),
        }
    }
}

/// [wal] section of the config file.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct WalSection {
    pub wal_dir: Option<PathBuf>,
    /// Segment size in bytes. Accepts human-readable strings via parse_size.
    #[serde(deserialize_with = "deserialize_size")]
    pub segment_size: usize,
    /// Sync mode: "fsync" for durable writes, "fdatasync" for data-only sync, "none" for no sync.
    pub sync_mode: String,
    /// Ring buffer capacity in bytes. Accepts human-readable strings.
    #[serde(deserialize_with = "deserialize_size")]
    pub ring_buffer_capacity: usize,
}

impl Default for WalSection {
    fn default() -> Self {
        Self {
            wal_dir: None,
            segment_size: 16 * 1024 * 1024, // 16 MB
            sync_mode: "fsync".into(),
            ring_buffer_capacity: 16 * 1024 * 1024, // 16 MB
        }
    }
}

/// [checkpoint] section of the config file.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct CheckpointSection {
    /// WAL bytes accumulated before triggering a checkpoint.
    #[serde(deserialize_with = "deserialize_size_u64")]
    pub wal_bytes_threshold: u64,
    /// Maximum seconds between checkpoints (fallback timer for idle systems).
    pub max_interval_secs: u32,
    /// Minimum seconds between checkpoints (prevents thrashing).
    pub min_interval_secs: u32,
}

impl Default for CheckpointSection {
    fn default() -> Self {
        Self {
            wal_bytes_threshold: 64 * 1024 * 1024, // 64 MB
            max_interval_secs: 600,
            min_interval_secs: 5,
        }
    }
}

/// [auth] section of the config file.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct AuthSection {
    pub method: String,
    pub password_encryption: String,
    pub balloon_space_cost: Option<usize>,
    pub balloon_time_cost: Option<usize>,
    pub jwt_secret: Option<String>,
    pub jwt_algorithm: Option<String>,
    pub jwt_issuer: Option<String>,
    pub brute_force_enabled: Option<bool>,
    pub lockout_threshold: Option<u32>,
    pub lockout_duration_secs: Option<u64>,
    pub ip_block_threshold: Option<u32>,
    pub failure_window_secs: Option<u64>,
    pub ip_block_duration_secs: Option<u64>,
    pub min_attempt_interval_ms: Option<u64>,
    /// WebAuthn relying party ID (domain name, e.g. "db.example.com").
    pub webauthn_rp_id: Option<String>,
    /// WebAuthn relying party display name.
    pub webauthn_rp_name: Option<String>,
    /// WebAuthn expected origin (e.g. "https://db.example.com").
    pub webauthn_origin: Option<String>,
    /// WebAuthn challenge timeout in seconds (default 60).
    pub webauthn_challenge_timeout: Option<u64>,
    /// Require TLS for all client connections.
    pub tls_required: bool,
}

impl Default for AuthSection {
    fn default() -> Self {
        Self {
            method: "trust".into(),
            password_encryption: "balloon-sha-256".into(),
            balloon_space_cost: None,
            balloon_time_cost: None,
            jwt_secret: None,
            jwt_algorithm: None,
            jwt_issuer: None,
            brute_force_enabled: None,
            lockout_threshold: None,
            lockout_duration_secs: None,
            ip_block_threshold: None,
            failure_window_secs: None,
            ip_block_duration_secs: None,
            min_attempt_interval_ms: None,
            webauthn_rp_id: None,
            webauthn_rp_name: None,
            webauthn_origin: None,
            webauthn_challenge_timeout: None,
            tls_required: false,
        }
    }
}

/// [logging] section of the config file.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct LoggingSection {
    pub level: String,
    pub format: String,
    /// Output target: "stdout" (default) or "file".
    pub output: String,
    /// Log file path. Required when output = "file".
    pub file_path: Option<PathBuf>,
}

impl Default for LoggingSection {
    fn default() -> Self {
        Self {
            level: "info".into(),
            format: "text".into(),
            output: "stdout".into(),
            file_path: None,
        }
    }
}

/// [metrics] section of the config file.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct MetricsSection {
    pub enabled: bool,
    /// Bind address for the health/metrics HTTP server. Accepts any string
    /// `TcpListener::bind` understands. Examples: `"127.0.0.1"` (IPv4 loopback),
    /// `"0.0.0.0"` (IPv4 wildcard), `"::1"` (IPv6 loopback), `"[::]"` (IPv6
    /// wildcard, also accepts IPv4 on dual-stack OSes when `dual_stack=true`)
    pub host: String,
    /// Port for health and metrics HTTP server.
    pub port: u16,
    /// Metrics endpoint path.
    pub path: String,
    /// When `host` is an IPv6 wildcard, accept IPv4 connections too via
    /// IPv4-mapped IPv6 addresses. Linux defaults V6ONLY to false (so this
    /// matches the kernel default), Windows defaults to true (so this flag
    /// explicitly overrides via socket2 to give consistent dual-stack behaviour)
    pub dual_stack: bool,
}

impl Default for MetricsSection {
    fn default() -> Self {
        Self {
            enabled: true,
            // Dual-stack default so IPv6 clients work out of the box
            host: "[::]".into(),
            port: 9090,
            path: "/metrics".into(),
            dual_stack: true,
        }
    }
}

/// [compaction] section of the config file.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct CompactionSection {
    pub enabled: bool,
    pub threshold_rows: u64,
    pub max_concurrent: usize,
    /// Rate limit for compaction IO in megabytes per second.
    pub rate_limit_mbps: u64,
    /// Seconds between compaction cycles.
    pub interval_secs: u64,
    /// Skip a cycle when measured query p99 exceeds this many microseconds.
    pub oltp_p99_threshold_us: u64,
    /// Maximum rows written into one .zyr segment file.
    pub max_rows_per_file: u64,
}

impl Default for CompactionSection {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold_rows: 100_000,
            max_concurrent: 2,
            rate_limit_mbps: 100,
            interval_secs: 30,
            oltp_p99_threshold_us: 1_000,
            max_rows_per_file: 1_000_000,
        }
    }
}

/// [vacuum] section of the config file.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct VacuumSection {
    pub enabled: bool,
    pub interval_secs: u64,
    /// Fraction of dead tuples before vacuum triggers (0.0 to 1.0).
    pub dead_tuple_threshold: f64,
}

impl Default for VacuumSection {
    fn default() -> Self {
        Self {
            enabled: true,
            interval_secs: 60,
            dead_tuple_threshold: 0.2,
        }
    }
}

/// [query] section of the config file.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct QuerySection {
    /// Default transaction isolation level: "snapshot" or "read_committed".
    pub default_isolation: String,
    /// Maximum seconds a statement can run before being canceled. 0 = no limit.
    pub statement_timeout_secs: u64,
    /// Maximum rows a single query can return. 0 = no limit.
    pub max_result_rows: u64,
    /// Maximum bytes one query's materializing operators (sorts, hash
    /// joins, aggregations, window buffers, set operations, the result
    /// set) may hold at once. Past it an operator that can spill does, and
    /// one that cannot fails. 0 = no limit and no spilling.
    pub max_memory_bytes: u64,
    /// Maximum bytes every spilling query on this node may hold on disk at
    /// once. A query that would take the node past it is refused rather than
    /// filling the disk, because a full disk stops every query rather than
    /// one. 0 = spilling disabled, and operators fail at the memory budget
    /// the way they did before spilling existed.
    pub spill_quota_bytes: u64,
}

impl Default for QuerySection {
    fn default() -> Self {
        Self {
            default_isolation: "snapshot".into(),
            statement_timeout_secs: 300,
            max_result_rows: 1_000_000,
            max_memory_bytes: 0,
            // Sixteen gigabytes, which is enough for a sort or a join far
            // larger than any node's memory and small enough that a runaway
            // query cannot take the disk the data is on. It only ever applies
            // when a memory budget is set, because without one nothing spills
            spill_quota_bytes: 16 * 1024 * 1024 * 1024,
        }
    }
}

/// Parses a human-readable size string into bytes.
/// Supports: "128MB", "1GB", "16KB", "1024" (plain bytes).
/// Case-insensitive. Allows optional space between number and unit.
/// The error every unknown-key arm returns, naming the exact key so a typo
/// in ALTER SYSTEM or a hand edited auto.conf points at itself
fn unknown_key(section: &str, key: &str) -> Result<()> {
    Err(ZyronError::Internal(format!(
        "unknown config key {}.{}",
        section, key
    )))
}

pub fn parse_size(s: &str) -> std::result::Result<usize, String> {
    let s = s.trim();
    if s.is_empty() {
        return Err("empty size string".into());
    }

    // Find where the numeric part ends
    let num_end = s
        .find(|c: char| !c.is_ascii_digit() && c != '.')
        .unwrap_or(s.len());

    let num_str = s[..num_end].trim();
    let unit_str = s[num_end..].trim().to_uppercase();

    let num: f64 = num_str
        .parse()
        .map_err(|_| format!("invalid number in size string: {}", num_str))?;

    let multiplier: f64 = match unit_str.as_str() {
        "" | "B" => 1.0,
        "KB" | "K" => 1024.0,
        "MB" | "M" => 1024.0 * 1024.0,
        "GB" | "G" => 1024.0 * 1024.0 * 1024.0,
        "TB" | "T" => 1024.0 * 1024.0 * 1024.0 * 1024.0,
        "PB" | "P" => 1024.0 * 1024.0 * 1024.0 * 1024.0 * 1024.0,
        "ZB" | "Z" => 1024.0 * 1024.0 * 1024.0 * 1024.0 * 1024.0 * 1024.0 * 1024.0,
        _ => return Err(format!("unknown size unit: {}", unit_str)),
    };

    let result = num * multiplier;
    if !result.is_finite() || result < 0.0 || result > usize::MAX as f64 {
        return Err(format!("size value overflows: {}", s));
    }

    Ok(result as usize)
}

/// Parses a human-readable size string into bytes as u64.
pub fn parse_size_u64(s: &str) -> std::result::Result<u64, String> {
    parse_size(s).map(|v| v as u64)
}

/// Serde deserializer that accepts either an integer or a size string.
fn deserialize_size<'de, D>(deserializer: D) -> std::result::Result<usize, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de;

    struct SizeVisitor;

    impl<'de> de::Visitor<'de> for SizeVisitor {
        type Value = usize;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("an integer or a size string like \"128MB\"")
        }

        fn visit_u64<E: de::Error>(self, value: u64) -> std::result::Result<usize, E> {
            Ok(value as usize)
        }

        fn visit_i64<E: de::Error>(self, value: i64) -> std::result::Result<usize, E> {
            if value < 0 {
                return Err(E::custom("size cannot be negative"));
            }
            Ok(value as usize)
        }

        fn visit_str<E: de::Error>(self, value: &str) -> std::result::Result<usize, E> {
            parse_size(value).map_err(E::custom)
        }
    }

    deserializer.deserialize_any(SizeVisitor)
}

/// Serde deserializer that accepts either an integer or a size string, returns u64.
fn deserialize_size_u64<'de, D>(deserializer: D) -> std::result::Result<u64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de;

    struct SizeVisitorU64;

    impl<'de> de::Visitor<'de> for SizeVisitorU64 {
        type Value = u64;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("an integer or a size string like \"64MB\"")
        }

        fn visit_u64<E: de::Error>(self, value: u64) -> std::result::Result<u64, E> {
            Ok(value)
        }

        fn visit_i64<E: de::Error>(self, value: i64) -> std::result::Result<u64, E> {
            if value < 0 {
                return Err(E::custom("size cannot be negative"));
            }
            Ok(value as u64)
        }

        fn visit_str<E: de::Error>(self, value: &str) -> std::result::Result<u64, E> {
            parse_size_u64(value).map_err(E::custom)
        }
    }

    deserializer.deserialize_any(SizeVisitorU64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_size_bytes() {
        assert_eq!(parse_size("1024").unwrap(), 1024);
        assert_eq!(parse_size("0").unwrap(), 0);
    }

    #[test]
    fn test_parse_size_kb() {
        assert_eq!(parse_size("16KB").unwrap(), 16 * 1024);
        assert_eq!(parse_size("16 KB").unwrap(), 16 * 1024);
        assert_eq!(parse_size("16kb").unwrap(), 16 * 1024);
        assert_eq!(parse_size("16K").unwrap(), 16 * 1024);
    }

    #[test]
    fn test_parse_size_mb() {
        assert_eq!(parse_size("128MB").unwrap(), 128 * 1024 * 1024);
        assert_eq!(parse_size("64 MB").unwrap(), 64 * 1024 * 1024);
        assert_eq!(parse_size("1M").unwrap(), 1024 * 1024);
    }

    #[test]
    fn test_parse_size_gb() {
        assert_eq!(parse_size("1GB").unwrap(), 1024 * 1024 * 1024);
        assert_eq!(parse_size("2 GB").unwrap(), 2 * 1024 * 1024 * 1024);
        assert_eq!(parse_size("1G").unwrap(), 1024 * 1024 * 1024);
    }

    #[test]
    fn test_parse_size_errors() {
        assert!(parse_size("").is_err());
        assert!(parse_size("abc").is_err());
        assert!(parse_size("16XB").is_err());
        assert!(parse_size("999999999TB").is_err()); // overflow
    }

    #[test]
    fn test_parse_size_pb_zb() {
        assert_eq!(parse_size("1PB").unwrap(), 1024 * 1024 * 1024 * 1024 * 1024);
        assert_eq!(parse_size("1P").unwrap(), 1024 * 1024 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_page_checksum_verify_validation() {
        // The default is always-on verification
        let config = ZyronConfig::default();
        assert_eq!(config.storage.page_checksum_verify, "always");
        assert!(config.validate().is_ok());

        // Sampled exists for benchmark investigations
        let mut config = ZyronConfig::default();
        config.storage.page_checksum_verify = "sampled".into();
        assert!(config.validate().is_ok());

        // Off is rejected, a server never runs an unverified read path.
        // The error names the setting and the acceptable values
        let mut config = ZyronConfig::default();
        config.storage.page_checksum_verify = "off".into();
        let err = config.validate().expect_err("'off' must be rejected");
        let message = err.to_string();
        assert!(message.contains("page_checksum_verify"), "{message}");
        assert!(message.contains("always"), "{message}");
        assert!(message.contains("sampled"), "{message}");

        // Unknown values are rejected with the same guidance
        let mut config = ZyronConfig::default();
        config.storage.page_checksum_verify = "sometimes".into();
        let err = config
            .validate()
            .expect_err("unknown value must be rejected");
        assert!(err.to_string().contains("page_checksum_verify"));
    }

    #[test]
    fn test_default_config() {
        let config = ZyronConfig::default();
        assert_eq!(config.server.port, 5432);
        assert_eq!(config.storage.buffer_pool_size, 128 * 1024 * 1024);
        assert_eq!(config.wal.segment_size, 16 * 1024 * 1024);
        assert_eq!(config.wal.ring_buffer_capacity, 16 * 1024 * 1024);
        assert_eq!(config.checkpoint.wal_bytes_threshold, 64 * 1024 * 1024);
        assert_eq!(config.checkpoint.max_interval_secs, 600);
        assert_eq!(config.checkpoint.min_interval_secs, 5);
        assert_eq!(config.auth.method, "trust");
        assert!(!config.auth.tls_required);
        assert_eq!(config.logging.level, "info");
        assert_eq!(config.logging.output, "stdout");
        assert!(config.logging.file_path.is_none());
        // New sections
        assert!(config.metrics.enabled);
        assert_eq!(config.metrics.port, 9090);
        assert_eq!(config.metrics.path, "/metrics");
        assert!(config.compaction.enabled);
        assert_eq!(config.compaction.threshold_rows, 100_000);
        assert_eq!(config.compaction.max_concurrent, 2);
        assert_eq!(config.compaction.rate_limit_mbps, 100);
        assert!(config.vacuum.enabled);
        assert_eq!(config.vacuum.interval_secs, 60);
        assert!((config.vacuum.dead_tuple_threshold - 0.2).abs() < f64::EPSILON);
        assert_eq!(config.query.default_isolation, "snapshot");
        assert_eq!(config.query.statement_timeout_secs, 300);
        assert_eq!(config.query.max_result_rows, 1_000_000);
    }

    #[test]
    fn test_load_toml_string() {
        let toml_str = r#"
[server]
port = 5433
max_connections = 500
host = "0.0.0.0"

[storage]
data_dir = "/var/lib/zyron"
buffer_pool_size = "1GB"

[wal]
segment_size = "32MB"
sync_mode = "fsync"

[checkpoint]
wal_bytes_threshold = "128MB"
max_interval_secs = 300
min_interval_secs = 3

[auth]
method = "scram-sha-256"

[logging]
level = "debug"
format = "json"
"#;
        let config: ZyronConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.server.port, 5433);
        assert_eq!(config.server.host, "0.0.0.0");
        // The file still carries the removed connection cap, so it is refused
        // rather than parsed and forgotten
        let refused = config.validate().expect_err("stale key must be refused");
        assert!(
            refused
                .to_string()
                .contains("max_connections has been removed"),
            "unhelpful refusal: {refused}"
        );
        assert_eq!(config.storage.data_dir, PathBuf::from("/var/lib/zyron"));
        assert_eq!(config.storage.buffer_pool_size, 1024 * 1024 * 1024);
        assert_eq!(config.wal.segment_size, 32 * 1024 * 1024);
        assert_eq!(config.checkpoint.wal_bytes_threshold, 128 * 1024 * 1024);
        assert_eq!(config.checkpoint.max_interval_secs, 300);
        assert_eq!(config.checkpoint.min_interval_secs, 3);
        assert_eq!(config.auth.method, "scram-sha-256");
        assert_eq!(config.logging.level, "debug");
        assert_eq!(config.logging.format, "json");
    }

    #[test]
    fn test_partial_toml() {
        let toml_str = r#"
[server]
port = 9999
"#;
        let config: ZyronConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.server.port, 9999);
        // All other sections should be defaults
        assert_eq!(config.storage.buffer_pool_size, 128 * 1024 * 1024);
        assert_eq!(config.checkpoint.max_interval_secs, 600);
    }

    #[test]
    fn test_integer_sizes() {
        let toml_str = r#"
[storage]
buffer_pool_size = 67108864

[wal]
segment_size = 8388608

[checkpoint]
wal_bytes_threshold = 33554432
"#;
        let config: ZyronConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.storage.buffer_pool_size, 64 * 1024 * 1024);
        assert_eq!(config.wal.segment_size, 8 * 1024 * 1024);
        assert_eq!(config.checkpoint.wal_bytes_threshold, 32 * 1024 * 1024);
    }

    #[test]
    fn test_to_server_config() {
        let config = ZyronConfig::default();
        let server_cfg = config.to_server_config();
        // Default is dual-stack IPv6 wildcard
        assert_eq!(server_cfg.host, "[::]");
        assert!(server_cfg.dual_stack);
        assert_eq!(server_cfg.port, 5432);
    }

    #[test]
    fn test_to_storage_config() {
        let config = ZyronConfig::default();
        let storage_cfg = config.to_storage_config();
        assert_eq!(storage_cfg.data_dir, PathBuf::from("./data"));
        assert_eq!(storage_cfg.wal_dir, PathBuf::from("./data/wal"));
        assert_eq!(storage_cfg.buffer_pool_pages, 128 * 1024 * 1024 / 16384);
        assert_eq!(storage_cfg.wal_segment_size, 16 * 1024 * 1024);
    }

    #[test]
    fn test_validation_errors() {
        let mut config = ZyronConfig::default();
        config.server.port = 0;
        assert!(config.validate().is_err());

        let mut config = ZyronConfig::default();
        config.server.max_connections = Some(toml::Value::Integer(500));
        assert!(config.validate().is_err());

        let mut config = ZyronConfig::default();
        config.checkpoint.min_interval_secs = 600;
        config.checkpoint.max_interval_secs = 600;
        assert!(config.validate().is_err());

        let mut config = ZyronConfig::default();
        config.server.tls_enabled = true;
        assert!(config.validate().is_err());

        // New section validations
        let mut config = ZyronConfig::default();
        config.wal.sync_mode = "invalid".into();
        assert!(config.validate().is_err());

        let mut config = ZyronConfig::default();
        config.compaction.max_concurrent = 0;
        assert!(config.validate().is_err());

        let mut config = ZyronConfig::default();
        config.compaction.rate_limit_mbps = 0;
        assert!(config.validate().is_err());

        let mut config = ZyronConfig::default();
        config.vacuum.dead_tuple_threshold = 1.5;
        assert!(config.validate().is_err());

        let mut config = ZyronConfig::default();
        config.query.default_isolation = "none".into();
        assert!(config.validate().is_err());

        let mut config = ZyronConfig::default();
        config.logging.output = "file".into();
        config.logging.file_path = None;
        assert!(config.validate().is_err());

        // Valid file logging
        let mut config = ZyronConfig::default();
        config.logging.output = "file".into();
        config.logging.file_path = Some(PathBuf::from("/var/log/zyron.log"));
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_deployment_mode_reads_from_toml_and_rejects_unknown_names() {
        // Unset means unified, which runs both formats with heap as the
        // default so an existing deployment keeps its behavior
        let config = ZyronConfig::default();
        assert_eq!(config.storage.deployment_mode, "unified");
        assert_eq!(
            config.deployment_mode(),
            zyron_common::DeploymentMode::Unified
        );
        assert!(config.validate().is_ok());

        let config: ZyronConfig = toml::from_str(
            r#"
[storage]
deployment_mode = "lake"
"#,
        )
        .unwrap();
        assert_eq!(config.deployment_mode(), zyron_common::DeploymentMode::Lake);
        assert!(config.validate().is_ok());
        assert_eq!(
            config
                .get_config_value("storage.deployment_mode")
                .as_deref(),
            Some("lake")
        );

        let mut config = ZyronConfig::default();
        config.storage.deployment_mode = "hybrid".into();
        let err = config.validate().expect_err("unknown mode must be refused");
        assert!(err.to_string().contains("storage.deployment_mode"));
        assert!(err.to_string().contains("'db', 'lake' or 'unified'"));

        // Runtime override path, the one zyron.auto.conf goes through
        let mut config = ZyronConfig::default();
        config
            .set_config_value("storage", "deployment_mode", "db")
            .unwrap();
        assert_eq!(config.deployment_mode(), zyron_common::DeploymentMode::Db);
    }

    // ALTER SYSTEM overrides are refused up front instead of being dropped
    // silently at the next boot
    #[test]
    fn test_a_discord_notify_url_is_checked_where_it_is_set() {
        let mut config = ZyronConfig::default();
        config
            .apply_override(
                "upgrade.notify_discord_webhook_url",
                " https://discord.com/api/webhooks/1/abc ",
            )
            .expect("a Discord webhook address is taken, trimmed");
        assert_eq!(
            config.upgrade.notify_discord_webhook_url,
            "https://discord.com/api/webhooks/1/abc"
        );
        assert_eq!(
            config.get_config_value("upgrade.notify_discord_webhook_url"),
            Some("https://discord.com/api/webhooks/1/abc".to_string())
        );

        let err = config
            .apply_override("upgrade.notify_discord_webhook_url", "https://example/hook")
            .expect_err("an address that is not a Discord webhook must be refused");
        assert!(
            err.to_string()
                .contains("upgrade.notify_discord_webhook_url"),
            "{err}"
        );
        assert!(
            err.to_string().contains("is not a Discord webhook address"),
            "{err}"
        );
        assert_eq!(
            config.upgrade.notify_discord_webhook_url, "https://discord.com/api/webhooks/1/abc",
            "a refused address leaves the configured one in place"
        );

        // Empty clears the channel, so an operator can turn it off
        config
            .apply_override("upgrade.notify_discord_webhook_url", "")
            .expect("empty is accepted");
        assert!(config.upgrade.notify_discord_webhook_url.is_empty());
    }

    #[test]
    fn test_apply_override_refuses_unknown_and_unparseable() {
        let mut config = ZyronConfig::default();

        let err = config
            .apply_override("server.prot", "5433")
            .expect_err("typo key must be refused");
        assert!(err.to_string().contains("server.prot"));

        let err = config
            .apply_override("nosuch.key", "1")
            .expect_err("unknown section must be refused");
        assert!(err.to_string().contains("nosuch"));

        let err = config
            .apply_override("server.port", "not_a_port")
            .expect_err("unparseable value must be refused");
        assert!(err.to_string().contains("server.port"));

        let err = config
            .apply_override("port", "5433")
            .expect_err("undotted key must be refused");
        assert!(err.to_string().contains("section.field"));

        config.apply_override("server.port", "5433").unwrap();
        assert_eq!(config.server.port, 5433);
        config
            .apply_override("compaction.interval_secs", "120")
            .unwrap();
        assert_eq!(config.compaction.interval_secs, 120);
        config
            .apply_override("compaction.max_rows_per_file", "500000")
            .unwrap();
        assert_eq!(config.compaction.max_rows_per_file, 500000);
    }

    // A bad entry in auto.conf names itself in the boot error instead of
    // vanishing, so the operator knows exactly what to fix
    #[test]
    fn test_auto_conf_bad_entry_errors_with_key_name() {
        let tmp = tempfile::TempDir::new().unwrap();
        let dir = tmp.path().to_path_buf();
        std::fs::write(dir.join("zyron.auto.conf"), "[server]\nprot = 5433\n").unwrap();

        let mut config = ZyronConfig::default();
        let err = config
            .apply_auto_conf(&dir)
            .expect_err("unknown persisted key must fail the boot loudly");
        let msg = err.to_string();
        assert!(msg.contains("zyron.auto.conf"));
        assert!(msg.contains("server.prot"));
    }

    #[test]
    fn test_new_sections_toml() {
        let toml_str = r#"
[metrics]
enabled = false
port = 8080
path = "/prom"

[compaction]
enabled = false
threshold_rows = 50000
max_concurrent = 4
rate_limit_mbps = 200

[vacuum]
enabled = false
interval_secs = 120
dead_tuple_threshold = 0.3

[query]
default_isolation = "read_committed"
statement_timeout_secs = 60
max_result_rows = 500000

[logging]
output = "file"
file_path = "/var/log/zyron.log"

[auth]
tls_required = true

[wal]
ring_buffer_capacity = "32MB"
sync_mode = "fdatasync"
"#;
        let config: ZyronConfig = toml::from_str(toml_str).unwrap();
        assert!(!config.metrics.enabled);
        assert_eq!(config.metrics.port, 8080);
        assert_eq!(config.metrics.path, "/prom");
        assert!(!config.compaction.enabled);
        assert_eq!(config.compaction.threshold_rows, 50_000);
        assert_eq!(config.compaction.max_concurrent, 4);
        assert_eq!(config.compaction.rate_limit_mbps, 200);
        assert!(!config.vacuum.enabled);
        assert_eq!(config.vacuum.interval_secs, 120);
        assert!((config.vacuum.dead_tuple_threshold - 0.3).abs() < f64::EPSILON);
        assert_eq!(config.query.default_isolation, "read_committed");
        assert_eq!(config.query.statement_timeout_secs, 60);
        assert_eq!(config.query.max_result_rows, 500_000);
        assert_eq!(config.logging.output, "file");
        assert_eq!(
            config.logging.file_path,
            Some(PathBuf::from("/var/log/zyron.log"))
        );
        assert!(config.auth.tls_required);
        assert_eq!(config.wal.ring_buffer_capacity, 32 * 1024 * 1024);
        assert_eq!(config.wal.sync_mode, "fdatasync");
    }

    #[test]
    fn test_get_config_value() {
        let config = ZyronConfig::default();
        assert_eq!(config.get_config_value("server.port"), Some("5432".into()));
        assert_eq!(
            config.get_config_value("wal.sync_mode"),
            Some("fsync".into())
        );
        assert_eq!(
            config.get_config_value("vacuum.dead_tuple_threshold"),
            Some("0.2".into())
        );
        assert_eq!(
            config.get_config_value("query.default_isolation"),
            Some("snapshot".into())
        );
        assert_eq!(
            config.get_config_value("metrics.enabled"),
            Some("true".into())
        );
        assert!(config.get_config_value("server_version").is_some());
        assert!(config.get_config_value("nonexistent.key").is_none());
    }

    #[test]
    fn test_all_config_entries() {
        let config = ZyronConfig::default();
        let entries = config.all_config_entries();
        assert!(entries.len() >= 30);
        // Verify server_version is first
        assert_eq!(entries[0].0, "server_version");
        // Verify all entries have non-empty key, value, description
        for (key, _value, desc) in &entries {
            assert!(!key.is_empty());
            assert!(!desc.is_empty());
        }
    }

    #[test]
    fn test_auto_conf_round_trip() {
        let dir = std::env::temp_dir().join("zyron_test_auto_conf");
        let _ = std::fs::create_dir_all(&dir);

        // Write an override
        ZyronConfig::write_auto_conf(&dir, "server.port", "9999").unwrap();
        ZyronConfig::write_auto_conf(&dir, "vacuum.enabled", "false").unwrap();
        ZyronConfig::write_auto_conf(&dir, "query.default_isolation", "read_committed").unwrap();

        // Load and apply
        let mut config = ZyronConfig::default();
        config.apply_auto_conf(&dir).unwrap();

        assert_eq!(config.server.port, 9999);
        assert!(!config.vacuum.enabled);
        assert_eq!(config.query.default_isolation, "read_committed");

        // Clean up
        let _ = std::fs::remove_dir_all(&dir);
    }

    // The rewrite goes through a sibling and a rename, so nothing but the
    // file itself is left behind, and a file that is not TOML is refused
    // with its name rather than silently replaced
    #[test]
    fn test_auto_conf_rewrite_leaves_only_the_file_and_refuses_a_corrupt_one() {
        let tmp = tempfile::TempDir::new().unwrap();
        let dir = tmp.path();
        ZyronConfig::write_auto_conf(dir, "server.port", "9999").unwrap();
        ZyronConfig::write_auto_conf(dir, "vacuum.enabled", "false").unwrap();
        let names: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, vec!["zyron.auto.conf".to_string()], "{names:?}");

        std::fs::write(dir.join("zyron.auto.conf"), "[server\nport = ").unwrap();
        let err = ZyronConfig::write_auto_conf(dir, "server.port", "1")
            .expect_err("a corrupt file is not overwritten")
            .to_string();
        assert!(err.contains("zyron.auto.conf"), "{err}");
        assert!(err.contains("server.port"), "{err}");
        assert_eq!(
            std::fs::read_to_string(dir.join("zyron.auto.conf")).unwrap(),
            "[server\nport = ",
            "the corrupt file is left for the operator"
        );
    }

    // Every writer rewrites the whole file, so writers from several threads
    // at once keep every key only because the rewrite is serialized
    #[test]
    fn test_auto_conf_concurrent_writers_lose_no_key() {
        let tmp = tempfile::TempDir::new().unwrap();
        let dir = tmp.path();
        let keys: Vec<String> = (0..16).map(|i| format!("section_{i}.field")).collect();
        std::thread::scope(|scope| {
            for key in &keys {
                scope.spawn(move || {
                    ZyronConfig::write_auto_conf(dir, key, "1").unwrap();
                });
            }
        });
        let contents = std::fs::read_to_string(dir.join("zyron.auto.conf")).unwrap();
        let table: toml::Table = toml::from_str(&contents).unwrap();
        for key in &keys {
            let (section, field) = key.split_once('.').unwrap();
            let value = table
                .get(section)
                .and_then(|s| s.get(field))
                .and_then(|v| v.as_integer());
            assert_eq!(value, Some(1), "{key} was lost, file holds {contents}");
        }
    }

    #[test]
    fn test_env_overrides() {
        // SAFETY: test runs single-threaded, env vars are cleaned up after.
        unsafe {
            std::env::set_var("ZYRON_PORT", "9876");
            std::env::set_var("ZYRON_HOST", "0.0.0.0");
            std::env::set_var("ZYRON_DATA_DIR", "/tmp/zyron");
            std::env::set_var("ZYRON_LOG_LEVEL", "debug");
            std::env::set_var("ZYRON_MAX_CONNECTIONS", "2000");
        }

        let mut config = ZyronConfig::default();
        config.apply_env_overrides();

        assert_eq!(config.server.port, 9876);
        assert_eq!(config.server.host, "0.0.0.0");
        assert_eq!(config.storage.data_dir, PathBuf::from("/tmp/zyron"));
        assert_eq!(config.logging.level, "debug");

        // Clean up
        unsafe {
            std::env::remove_var("ZYRON_PORT");
            std::env::remove_var("ZYRON_HOST");
            std::env::remove_var("ZYRON_DATA_DIR");
            std::env::remove_var("ZYRON_LOG_LEVEL");
            std::env::remove_var("ZYRON_MAX_CONNECTIONS");
        }
    }
}

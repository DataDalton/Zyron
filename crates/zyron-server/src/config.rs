//! TOML configuration loading for ZyronDB.
//!
//! Reads a zyrondb.toml file, applies environment variable overrides,
//! validates settings, and maps sections to the existing ServerConfig
//! and StorageConfig structs.

use serde::Deserialize;
use std::path::{Path, PathBuf};
use zyron_common::{Result, ZyronError};

/// Top-level server configuration loaded from zyrondb.toml.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ZyronConfig {
    pub server: ServerSection,
    pub storage: StorageSection,
    pub wal: WalSection,
    pub checkpoint: CheckpointSection,
    pub auth: AuthSection,
    pub logging: LoggingSection,
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
        }
    }
}

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
        let config: ZyronConfig = toml::from_str(&contents)
            .map_err(|e| ZyronError::Internal(format!("Failed to parse config file: {}", e)))?;
        config.validate()?;
        Ok(config)
    }

    /// Loads configuration with the following priority:
    /// 1. Explicit path (if provided)
    /// 2. ./zyrondb.toml
    /// 3. Default values
    ///
    /// Environment variables override file values:
    /// ZYRON_HOST, ZYRON_PORT, ZYRON_DATA_DIR, ZYRON_WAL_DIR, ZYRON_LOG_LEVEL,
    /// ZYRON_MAX_CONNECTIONS
    pub fn load_with_overrides(path: Option<&Path>) -> Result<Self> {
        let mut config = if let Some(p) = path {
            Self::load(p)?
        } else {
            let default_path = PathBuf::from("zyrondb.toml");
            if default_path.exists() {
                Self::load(&default_path)?
            } else {
                Self::default()
            }
        };

        config.apply_env_overrides();
        config.validate()?;
        Ok(config)
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
        if let Ok(val) = std::env::var("ZYRON_MAX_CONNECTIONS") {
            match val.parse() {
                Ok(n) => self.server.max_connections = n,
                Err(_) => tracing::warn!(
                    "ZYRON_MAX_CONNECTIONS='{}' is not a valid number, ignoring",
                    val
                ),
            }
        }
    }

    /// Validates the config for logical consistency.
    fn validate(&self) -> Result<()> {
        if self.server.port == 0 {
            return Err(ZyronError::Internal("Server port cannot be 0".into()));
        }
        if self.server.max_connections == 0 {
            return Err(ZyronError::Internal("max_connections cannot be 0".into()));
        }
        if self.server.worker_threads == 0 {
            return Err(ZyronError::Internal("worker_threads cannot be 0".into()));
        }
        if self.storage.buffer_pool_size == 0 {
            return Err(ZyronError::Internal("buffer_pool_size cannot be 0".into()));
        }
        if self.wal.segment_size == 0 {
            return Err(ZyronError::Internal("WAL segment_size cannot be 0".into()));
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
        Ok(())
    }

    /// Converts the server section to the common ServerConfig.
    pub fn to_server_config(&self) -> zyron_common::ServerConfig {
        zyron_common::ServerConfig {
            host: self.server.host.clone(),
            port: self.server.port,
            max_connections: self.server.max_connections,
            connection_timeout_secs: self.server.connection_timeout_secs,
            statement_timeout_secs: self.server.statement_timeout_secs,
            worker_threads: self.server.worker_threads,
            tls_enabled: self.server.tls_enabled,
            tls_cert_path: self.server.tls_cert_path.clone(),
            tls_key_path: self.server.tls_key_path.clone(),
            quic_enabled: self.server.quic_enabled,
            quic_port: self.server.quic_port,
            quic_zero_rtt: self.server.quic_zero_rtt,
            quic_idle_timeout_secs: self.server.quic_idle_timeout_secs,
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
}

/// [server] section of the config file.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ServerSection {
    pub host: String,
    pub port: u16,
    pub max_connections: u32,
    pub connection_timeout_secs: u32,
    pub statement_timeout_secs: u32,
    pub worker_threads: usize,
    pub tls_enabled: bool,
    pub tls_cert_path: Option<PathBuf>,
    pub tls_key_path: Option<PathBuf>,
    pub quic_enabled: bool,
    pub quic_port: Option<u16>,
    pub quic_zero_rtt: bool,
    pub quic_idle_timeout_secs: u32,
    /// Port for the health/metrics HTTP server.
    pub health_port: u16,
}

impl Default for ServerSection {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: 5432,
            max_connections: 1000,
            connection_timeout_secs: 30,
            statement_timeout_secs: 0,
            worker_threads: std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(1),
            tls_enabled: false,
            tls_cert_path: None,
            tls_key_path: None,
            quic_enabled: false,
            quic_port: None,
            quic_zero_rtt: false,
            quic_idle_timeout_secs: 300,
            health_port: 9090,
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
}

impl Default for StorageSection {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("./data"),
            page_size: 16384,
            buffer_pool_size: 128 * 1024 * 1024, // 128 MB
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
    /// Sync mode: "fsync" for durable writes, "none" for no sync.
    pub sync_mode: String,
}

impl Default for WalSection {
    fn default() -> Self {
        Self {
            wal_dir: None,
            segment_size: 16 * 1024 * 1024, // 16 MB
            sync_mode: "fsync".into(),
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
    pub password_file: Option<PathBuf>,
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
}

impl Default for AuthSection {
    fn default() -> Self {
        Self {
            method: "trust".into(),
            password_file: None,
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
        }
    }
}

/// [logging] section of the config file.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct LoggingSection {
    pub level: String,
    pub format: String,
}

impl Default for LoggingSection {
    fn default() -> Self {
        Self {
            level: "info".into(),
            format: "text".into(),
        }
    }
}

/// Parses a human-readable size string into bytes.
/// Supports: "128MB", "1GB", "16KB", "1024" (plain bytes).
/// Case-insensitive. Allows optional space between number and unit.
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
    fn test_default_config() {
        let config = ZyronConfig::default();
        assert_eq!(config.server.port, 5432);
        assert_eq!(config.server.max_connections, 1000);
        assert_eq!(config.storage.buffer_pool_size, 128 * 1024 * 1024);
        assert_eq!(config.wal.segment_size, 16 * 1024 * 1024);
        assert_eq!(config.checkpoint.wal_bytes_threshold, 64 * 1024 * 1024);
        assert_eq!(config.checkpoint.max_interval_secs, 600);
        assert_eq!(config.checkpoint.min_interval_secs, 5);
        assert_eq!(config.auth.method, "trust");
        assert_eq!(config.logging.level, "info");
    }

    #[test]
    fn test_load_toml_string() {
        let toml_str = r#"
[server]
port = 5433
max_connections = 500
host = "0.0.0.0"

[storage]
data_dir = "/var/lib/zyrondb"
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
password_file = "/etc/zyrondb/passwords"

[logging]
level = "debug"
format = "json"
"#;
        let config: ZyronConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.server.port, 5433);
        assert_eq!(config.server.max_connections, 500);
        assert_eq!(config.server.host, "0.0.0.0");
        assert_eq!(config.storage.data_dir, PathBuf::from("/var/lib/zyrondb"));
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
        assert_eq!(config.server.max_connections, 1000);
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
        assert_eq!(server_cfg.host, "127.0.0.1");
        assert_eq!(server_cfg.port, 5432);
        assert_eq!(server_cfg.max_connections, 1000);
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
        config.server.max_connections = 0;
        assert!(config.validate().is_err());

        let mut config = ZyronConfig::default();
        config.checkpoint.min_interval_secs = 600;
        config.checkpoint.max_interval_secs = 600;
        assert!(config.validate().is_err());

        let mut config = ZyronConfig::default();
        config.server.tls_enabled = true;
        assert!(config.validate().is_err());
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
        assert_eq!(config.server.max_connections, 2000);

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

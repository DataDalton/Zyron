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
    pub buffer: BufferSection,
    pub wal: WalSection,
    pub checkpoint: CheckpointSection,
    pub auth: AuthSection,
    pub logging: LoggingSection,
    pub metrics: MetricsSection,
    pub compaction: CompactionSection,
    pub vacuum: VacuumSection,
    pub query: QuerySection,
}

impl Default for ZyronConfig {
    fn default() -> Self {
        Self {
            server: ServerSection::default(),
            storage: StorageSection::default(),
            buffer: BufferSection::default(),
            wal: WalSection::default(),
            checkpoint: CheckpointSection::default(),
            auth: AuthSection::default(),
            logging: LoggingSection::default(),
            metrics: MetricsSection::default(),
            compaction: CompactionSection::default(),
            vacuum: VacuumSection::default(),
            query: QuerySection::default(),
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
    /// 1. Defaults
    /// 2. Explicit path or ./zyrondb.toml
    /// 3. zyrondb.auto.conf (persistent ALTER SYSTEM overrides)
    /// 4. ZYRON_* environment variables
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

        // Apply persistent overrides from ALTER SYSTEM
        config.apply_auto_conf(&config.storage.data_dir.clone())?;
        config.apply_env_overrides();
        config.validate()?;
        Ok(config)
    }

    /// Loads zyrondb.auto.conf (TOML fragment) from the data directory and merges overrides.
    /// This file is written by ALTER SYSTEM SET commands.
    pub fn apply_auto_conf(&mut self, data_dir: &Path) -> Result<()> {
        let auto_path = data_dir.join("zyrondb.auto.conf");
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
        let overrides: toml::Table = toml::from_str(&contents).map_err(|e| {
            ZyronError::Internal(format!("Failed to parse zyrondb.auto.conf: {}", e))
        })?;
        self.apply_overrides_from_table(&overrides);
        Ok(())
    }

    /// Applies dotted key overrides from a TOML table.
    fn apply_overrides_from_table(&mut self, table: &toml::Table) {
        for (section, value) in table {
            if let toml::Value::Table(sub) = value {
                for (key, val) in sub {
                    let val_str = match val {
                        toml::Value::String(s) => s.clone(),
                        toml::Value::Integer(i) => i.to_string(),
                        toml::Value::Float(f) => f.to_string(),
                        toml::Value::Boolean(b) => b.to_string(),
                        _ => continue,
                    };
                    self.set_config_value(section, key, &val_str);
                }
            }
        }
    }

    /// Sets a config value by section and key name.
    fn set_config_value(&mut self, section: &str, key: &str, value: &str) {
        match section {
            "server" => match key {
                "host" => self.server.host = value.into(),
                "port" => {
                    if let Ok(v) = value.parse() {
                        self.server.port = v;
                    }
                }
                "max_connections" => {
                    if let Ok(v) = value.parse() {
                        self.server.max_connections = v;
                    }
                }
                "connection_timeout_secs" => {
                    if let Ok(v) = value.parse() {
                        self.server.connection_timeout_secs = v;
                    }
                }
                "statement_timeout_secs" => {
                    if let Ok(v) = value.parse() {
                        self.server.statement_timeout_secs = v;
                    }
                }
                "worker_threads" => {
                    if let Ok(v) = value.parse() {
                        self.server.worker_threads = v;
                    }
                }
                "tls_enabled" => {
                    if let Ok(v) = value.parse() {
                        self.server.tls_enabled = v;
                    }
                }
                "health_port" => {
                    if let Ok(v) = value.parse() {
                        self.server.health_port = v;
                    }
                }
                _ => {}
            },
            "storage" => match key {
                "data_dir" => self.storage.data_dir = PathBuf::from(value),
                "page_size" => {
                    if let Ok(v) = value.parse() {
                        self.storage.page_size = v;
                    }
                }
                "buffer_pool_size" => {
                    if let Ok(v) = parse_size(value) {
                        self.storage.buffer_pool_size = v;
                    }
                }
                "temp_dir" => self.storage.temp_dir = Some(PathBuf::from(value)),
                _ => {}
            },
            "buffer" => match key {
                "pool_size" => {
                    if let Ok(v) = parse_size(value) {
                        self.buffer.pool_size = v;
                    }
                }
                "eviction_policy" => self.buffer.eviction_policy = value.into(),
                _ => {}
            },
            "wal" => match key {
                "segment_size" => {
                    if let Ok(v) = parse_size(value) {
                        self.wal.segment_size = v;
                    }
                }
                "sync_mode" => self.wal.sync_mode = value.into(),
                "ring_buffer_capacity" => {
                    if let Ok(v) = parse_size(value) {
                        self.wal.ring_buffer_capacity = v;
                    }
                }
                _ => {}
            },
            "checkpoint" => match key {
                "wal_bytes_threshold" => {
                    if let Ok(v) = parse_size_u64(value) {
                        self.checkpoint.wal_bytes_threshold = v;
                    }
                }
                "max_interval_secs" => {
                    if let Ok(v) = value.parse() {
                        self.checkpoint.max_interval_secs = v;
                    }
                }
                "min_interval_secs" => {
                    if let Ok(v) = value.parse() {
                        self.checkpoint.min_interval_secs = v;
                    }
                }
                _ => {}
            },
            "auth" => match key {
                "method" => self.auth.method = value.into(),
                "tls_required" => {
                    if let Ok(v) = value.parse() {
                        self.auth.tls_required = v;
                    }
                }
                _ => {}
            },
            "logging" => match key {
                "level" => self.logging.level = value.into(),
                "format" => self.logging.format = value.into(),
                "output" => self.logging.output = value.into(),
                _ => {}
            },
            "metrics" => match key {
                "enabled" => {
                    if let Ok(v) = value.parse() {
                        self.metrics.enabled = v;
                    }
                }
                "port" => {
                    if let Ok(v) = value.parse() {
                        self.metrics.port = v;
                    }
                }
                "path" => self.metrics.path = value.into(),
                _ => {}
            },
            "compaction" => match key {
                "enabled" => {
                    if let Ok(v) = value.parse() {
                        self.compaction.enabled = v;
                    }
                }
                "threshold_rows" => {
                    if let Ok(v) = value.parse() {
                        self.compaction.threshold_rows = v;
                    }
                }
                "max_concurrent" => {
                    if let Ok(v) = value.parse() {
                        self.compaction.max_concurrent = v;
                    }
                }
                "rate_limit_mbps" => {
                    if let Ok(v) = value.parse() {
                        self.compaction.rate_limit_mbps = v;
                    }
                }
                _ => {}
            },
            "vacuum" => match key {
                "enabled" => {
                    if let Ok(v) = value.parse() {
                        self.vacuum.enabled = v;
                    }
                }
                "interval_secs" => {
                    if let Ok(v) = value.parse() {
                        self.vacuum.interval_secs = v;
                    }
                }
                "dead_tuple_threshold" => {
                    if let Ok(v) = value.parse() {
                        self.vacuum.dead_tuple_threshold = v;
                    }
                }
                _ => {}
            },
            "query" => match key {
                "default_isolation" => self.query.default_isolation = value.into(),
                "statement_timeout_secs" => {
                    if let Ok(v) = value.parse() {
                        self.query.statement_timeout_secs = v;
                    }
                }
                "max_result_rows" => {
                    if let Ok(v) = value.parse() {
                        self.query.max_result_rows = v;
                    }
                }
                _ => {}
            },
            _ => {}
        }
    }

    /// Writes a single key-value override to zyrondb.auto.conf.
    /// The key should be in "section.field" format (e.g. "server.port").
    pub fn write_auto_conf(data_dir: &Path, key: &str, value: &str) -> Result<()> {
        let auto_path = data_dir.join("zyrondb.auto.conf");
        let mut table: toml::Table = if auto_path.exists() {
            let contents = std::fs::read_to_string(&auto_path)
                .map_err(|e| ZyronError::Internal(format!("Failed to read auto.conf: {}", e)))?;
            toml::from_str(&contents).unwrap_or_default()
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

        let serialized = toml::to_string_pretty(&table)
            .map_err(|e| ZyronError::Internal(format!("Failed to serialize auto.conf: {}", e)))?;
        std::fs::write(&auto_path, serialized)
            .map_err(|e| ZyronError::Internal(format!("Failed to write auto.conf: {}", e)))?;
        Ok(())
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
        if let Ok(val) = std::env::var("ZYRON_BUFFER_POOL_SIZE") {
            match parse_size(&val) {
                Ok(v) => self.buffer.pool_size = v,
                Err(_) => tracing::warn!("ZYRON_BUFFER_POOL_SIZE='{}' is not valid, ignoring", val),
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
        if let Ok(val) = std::env::var("ZYRON_STORAGE_TEMP_DIR") {
            self.storage.temp_dir = Some(PathBuf::from(val));
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
        // Buffer section
        if self.buffer.pool_size == 0 {
            return Err(ZyronError::Internal("buffer.pool_size cannot be 0".into()));
        }
        match self.buffer.eviction_policy.as_str() {
            "clock" | "lru" => {}
            other => {
                return Err(ZyronError::Internal(format!(
                    "Invalid buffer.eviction_policy '{}', expected 'clock' or 'lru'",
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
        // Query section
        match self.query.default_isolation.as_str() {
            "snapshot" | "read_committed" | "serializable" => {}
            other => {
                return Err(ZyronError::Internal(format!(
                    "Invalid query.default_isolation '{}', expected 'snapshot', 'read_committed', or 'serializable'",
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

    /// Returns the effective temp directory.
    pub fn temp_dir(&self) -> PathBuf {
        self.storage
            .temp_dir
            .clone()
            .unwrap_or_else(|| self.storage.data_dir.join("tmp"))
    }

    /// Looks up a config value by dotted key (e.g. "server.port").
    /// Returns the current value as a string, or None if the key is not recognized.
    pub fn get_config_value(&self, key: &str) -> Option<String> {
        match key {
            // Server
            "server.host" => Some(self.server.host.clone()),
            "server.port" => Some(self.server.port.to_string()),
            "server.max_connections" => Some(self.server.max_connections.to_string()),
            "server.connection_timeout_secs" => {
                Some(self.server.connection_timeout_secs.to_string())
            }
            "server.statement_timeout_secs" => Some(self.server.statement_timeout_secs.to_string()),
            "server.worker_threads" => Some(self.server.worker_threads.to_string()),
            "server.tls_enabled" => Some(self.server.tls_enabled.to_string()),
            "server.health_port" => Some(self.server.health_port.to_string()),
            // Storage
            "storage.data_dir" => Some(self.storage.data_dir.display().to_string()),
            "storage.page_size" => Some(self.storage.page_size.to_string()),
            "storage.buffer_pool_size" => Some(self.storage.buffer_pool_size.to_string()),
            "storage.temp_dir" => Some(self.temp_dir().display().to_string()),
            // Buffer
            "buffer.pool_size" => Some(self.buffer.pool_size.to_string()),
            "buffer.eviction_policy" => Some(self.buffer.eviction_policy.clone()),
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
            // Compaction
            "compaction.enabled" => Some(self.compaction.enabled.to_string()),
            "compaction.threshold_rows" => Some(self.compaction.threshold_rows.to_string()),
            "compaction.max_concurrent" => Some(self.compaction.max_concurrent.to_string()),
            "compaction.rate_limit_mbps" => Some(self.compaction.rate_limit_mbps.to_string()),
            // Vacuum
            "vacuum.enabled" => Some(self.vacuum.enabled.to_string()),
            "vacuum.interval_secs" => Some(self.vacuum.interval_secs.to_string()),
            "vacuum.dead_tuple_threshold" => Some(self.vacuum.dead_tuple_threshold.to_string()),
            // Query
            "query.default_isolation" => Some(self.query.default_isolation.clone()),
            "query.statement_timeout_secs" => Some(self.query.statement_timeout_secs.to_string()),
            "query.max_result_rows" => Some(self.query.max_result_rows.to_string()),
            // Also support shorthand aliases
            "server_version" => Some(env!("CARGO_PKG_VERSION").to_string()),
            "port" => Some(self.server.port.to_string()),
            "max_connections" => Some(self.server.max_connections.to_string()),
            "data_dir" | "data_directory" => Some(self.storage.data_dir.display().to_string()),
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
                "server.max_connections".into(),
                self.server.max_connections.to_string(),
                "Maximum concurrent connections".into(),
            ),
            (
                "server.connection_timeout_secs".into(),
                self.server.connection_timeout_secs.to_string(),
                "Idle connection timeout in seconds".into(),
            ),
            (
                "server.statement_timeout_secs".into(),
                self.server.statement_timeout_secs.to_string(),
                "Statement timeout in seconds (0 = no limit)".into(),
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
                "server.health_port".into(),
                self.server.health_port.to_string(),
                "Health/metrics HTTP port".into(),
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
                "storage.temp_dir".into(),
                self.temp_dir().display().to_string(),
                "Temporary file directory".into(),
            ),
            (
                "buffer.pool_size".into(),
                self.buffer.pool_size.to_string(),
                "Buffer pool size in bytes".into(),
            ),
            (
                "buffer.eviction_policy".into(),
                self.buffer.eviction_policy.clone(),
                "Buffer eviction policy (clock or lru)".into(),
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
        ]
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
    /// Temporary file directory. Falls back to data_dir/tmp when not set.
    pub temp_dir: Option<PathBuf>,
}

impl Default for StorageSection {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("./data"),
            page_size: 16384,
            buffer_pool_size: 128 * 1024 * 1024, // 128 MB
            temp_dir: None,
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
    /// Require TLS for all client connections.
    pub tls_required: bool,
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

/// [buffer] section of the config file.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct BufferSection {
    /// Buffer pool size in bytes. Accepts human-readable strings like "1GB".
    #[serde(deserialize_with = "deserialize_size")]
    pub pool_size: usize,
    /// Eviction policy: "clock" (default) or "lru".
    pub eviction_policy: String,
}

impl Default for BufferSection {
    fn default() -> Self {
        Self {
            pool_size: 128 * 1024 * 1024, // 128 MB
            eviction_policy: "clock".into(),
        }
    }
}

/// [metrics] section of the config file.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct MetricsSection {
    pub enabled: bool,
    /// Port for health and metrics HTTP server.
    pub port: u16,
    /// Metrics endpoint path.
    pub path: String,
}

impl Default for MetricsSection {
    fn default() -> Self {
        Self {
            enabled: true,
            port: 9090,
            path: "/metrics".into(),
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
}

impl Default for CompactionSection {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold_rows: 100_000,
            max_concurrent: 2,
            rate_limit_mbps: 100,
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
    /// Default transaction isolation level: "snapshot", "read_committed", "serializable".
    pub default_isolation: String,
    /// Maximum seconds a statement can run before being canceled. 0 = no limit.
    pub statement_timeout_secs: u64,
    /// Maximum rows a single query can return. 0 = no limit.
    pub max_result_rows: u64,
}

impl Default for QuerySection {
    fn default() -> Self {
        Self {
            default_isolation: "snapshot".into(),
            statement_timeout_secs: 300,
            max_result_rows: 1_000_000,
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
        assert_eq!(config.buffer.pool_size, 128 * 1024 * 1024);
        assert_eq!(config.buffer.eviction_policy, "clock");
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
        assert!(config.storage.temp_dir.is_none());
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

        // New section validations
        let mut config = ZyronConfig::default();
        config.wal.sync_mode = "invalid".into();
        assert!(config.validate().is_err());

        let mut config = ZyronConfig::default();
        config.buffer.eviction_policy = "random".into();
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
    fn test_new_sections_toml() {
        let toml_str = r#"
[buffer]
pool_size = "2GB"
eviction_policy = "lru"

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
default_isolation = "serializable"
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

[storage]
temp_dir = "/tmp/zyron"
"#;
        let config: ZyronConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.buffer.pool_size, 2 * 1024 * 1024 * 1024);
        assert_eq!(config.buffer.eviction_policy, "lru");
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
        assert_eq!(config.query.default_isolation, "serializable");
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
        assert_eq!(config.storage.temp_dir, Some(PathBuf::from("/tmp/zyron")));
    }

    #[test]
    fn test_get_config_value() {
        let config = ZyronConfig::default();
        assert_eq!(config.get_config_value("server.port"), Some("5432".into()));
        assert_eq!(
            config.get_config_value("buffer.eviction_policy"),
            Some("clock".into())
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
        ZyronConfig::write_auto_conf(&dir, "query.default_isolation", "serializable").unwrap();

        // Load and apply
        let mut config = ZyronConfig::default();
        config.apply_auto_conf(&dir).unwrap();

        assert_eq!(config.server.port, 9999);
        assert!(!config.vacuum.enabled);
        assert_eq!(config.query.default_isolation, "serializable");

        // Clean up
        let _ = std::fs::remove_dir_all(&dir);
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

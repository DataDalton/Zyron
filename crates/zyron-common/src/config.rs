//! Configuration structures for ZyronDB.

use crate::page::PAGE_SIZE;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Server configuration for the ZyronDB instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Host address to bind to. Accepts IPv4 (`127.0.0.1`, `0.0.0.0`),
    /// IPv6 (`::1`, `::`, `[::]`), and hostnames. The `[]` brackets around an
    /// IPv6 literal are optional and stripped automatically
    pub host: String,
    /// Port number to listen on.
    pub port: u16,
    /// Maximum number of concurrent connections.
    pub max_connections: u32,
    /// Connection timeout in seconds.
    pub connection_timeout_secs: u32,
    /// Statement timeout in seconds (0 = no timeout).
    pub statement_timeout_secs: u32,
    /// Number of worker threads for query execution.
    pub worker_threads: usize,
    /// Enable TLS for connections.
    pub tls_enabled: bool,
    /// Path to TLS certificate file.
    pub tls_cert_path: Option<PathBuf>,
    /// Path to TLS key file.
    pub tls_key_path: Option<PathBuf>,
    /// Enable QUIC transport (requires tls_enabled with cert/key paths).
    pub quic_enabled: bool,
    /// UDP port for QUIC connections. Defaults to tcp port + 1 if not set.
    pub quic_port: Option<u16>,
    /// Enable 0-RTT connection resumption for QUIC.
    /// Faster reconnects but replay-vulnerable for the initial data.
    pub quic_zero_rtt: bool,
    /// QUIC idle timeout in seconds before closing inactive connections.
    pub quic_idle_timeout_secs: u32,
    /// When `host` resolves to an IPv6 wildcard (`::`), accept IPv4
    /// connections too via IPv4-mapped IPv6 addresses. Linux kernel defaults
    /// V6ONLY to false (matches this default); Windows defaults to true so
    /// the listener applies V6ONLY=false explicitly via socket2 to give
    /// consistent behaviour across platforms. Set false to bind IPv6-only
    #[serde(default = "default_dual_stack")]
    pub dual_stack: bool,
}

fn default_dual_stack() -> bool {
    true
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            // Dual-stack default so IPv6 clients work out of the box. Operators
            // wanting IPv4-only can set host = "0.0.0.0"
            host: "[::]".to_string(),
            port: 5432,
            max_connections: 100,
            connection_timeout_secs: 30,
            statement_timeout_secs: 0,
            worker_threads: num_cpus(),
            tls_enabled: false,
            tls_cert_path: None,
            tls_key_path: None,
            quic_enabled: false,
            quic_port: None,
            quic_zero_rtt: false,
            quic_idle_timeout_secs: 300,
            dual_stack: true,
        }
    }
}

impl ServerConfig {
    /// Returns the QUIC UDP port (defaults to tcp port + 1).
    pub fn quic_listen_port(&self) -> u16 {
        self.quic_port.unwrap_or(self.port + 1)
    }
}

/// Storage tiers a node runs, set by `storage.deployment_mode`.
///
/// The mode picks the format `CREATE TABLE` uses when the statement carries
/// no `USING` clause and refuses DDL naming the format the node does not run,
/// so nobody is handed a table whose commit-rate profile they did not ask for.
/// It also gates startup work: a `db` node opens no lake transaction log and
/// runs no lake worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DeploymentMode {
    /// Heap tables only, the latency-critical shape for an application's own
    /// database
    Db,
    /// ZyronLake tables only, the centrally managed lakehouse shape
    Lake,
    /// Both formats on one node, heap by default, so a heap table and a lake
    /// table join locally with no federation hop
    Unified,
}

impl Default for DeploymentMode {
    fn default() -> Self {
        Self::Unified
    }
}

impl DeploymentMode {
    /// Parses a configured mode name, case-insensitive. None on anything else
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "db" => Some(Self::Db),
            "lake" => Some(Self::Lake),
            "unified" => Some(Self::Unified),
            _ => None,
        }
    }

    /// The configured spelling of this mode
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Db => "db",
            Self::Lake => "lake",
            Self::Unified => "unified",
        }
    }

    /// True when a CREATE TABLE with no USING clause creates a lake table
    pub fn defaults_to_lake(self) -> bool {
        matches!(self, Self::Lake)
    }

    /// True when a table may be created in the ZyronLake format
    pub fn allows_lake(self) -> bool {
        matches!(self, Self::Lake | Self::Unified)
    }

    /// True when a table may be created in the heap format
    pub fn allows_heap(self) -> bool {
        matches!(self, Self::Db | Self::Unified)
    }

    /// True when the node opens lake transaction logs at startup and runs the
    /// lake background workers
    pub fn runs_lake_tier(self) -> bool {
        self.allows_lake()
    }
}

/// Storage configuration for the database engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Directory for data files.
    pub data_dir: PathBuf,
    /// Directory for WAL files.
    pub wal_dir: PathBuf,
    /// Page size in bytes.
    pub page_size: usize,
    /// Buffer pool size in number of pages.
    pub buffer_pool_pages: usize,
    /// WAL segment size in bytes.
    pub wal_segment_size: usize,
    /// Checkpoint interval in seconds.
    pub checkpoint_interval_secs: u32,
    /// Enable fsync for durability.
    pub fsync_enabled: bool,
    /// Enable direct I/O (bypass OS page cache).
    pub direct_io: bool,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("./data"),
            wal_dir: PathBuf::from("./data/wal"),
            page_size: PAGE_SIZE,
            buffer_pool_pages: 8192,            // 128 MB with 16 KB pages
            wal_segment_size: 16 * 1024 * 1024, // 16 MB
            checkpoint_interval_secs: 300,      // 5 minutes
            fsync_enabled: true,
            direct_io: false,
        }
    }
}

impl StorageConfig {
    /// Returns the total buffer pool size in bytes.
    pub fn buffer_pool_size_bytes(&self) -> usize {
        self.buffer_pool_pages * self.page_size
    }
}

/// Returns the number of available CPUs.
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_config_defaults() {
        let config = ServerConfig::default();
        // Dual-stack IPv6 wildcard accepts both IPv4 and IPv6 clients out of
        // the box on Linux (V6ONLY=false default) and on Windows via the
        // explicit V6ONLY=false applied by socket2 in the wire crate
        assert_eq!(config.host, "[::]");
        assert!(config.dual_stack);
        assert_eq!(config.port, 5432);
        assert_eq!(config.max_connections, 100);
        assert_eq!(config.connection_timeout_secs, 30);
        assert_eq!(config.statement_timeout_secs, 0);
        assert!(config.worker_threads >= 1);
        assert!(!config.tls_enabled);
        assert!(config.tls_cert_path.is_none());
        assert!(config.tls_key_path.is_none());
        assert!(!config.quic_enabled);
        assert!(config.quic_port.is_none());
        assert!(!config.quic_zero_rtt);
        assert_eq!(config.quic_idle_timeout_secs, 300);
        assert_eq!(config.quic_listen_port(), 5433);
    }

    #[test]
    fn test_server_config_custom() {
        let config = ServerConfig {
            host: "0.0.0.0".to_string(),
            port: 5433,
            max_connections: 500,
            connection_timeout_secs: 60,
            statement_timeout_secs: 300,
            worker_threads: 8,
            tls_enabled: true,
            tls_cert_path: Some(PathBuf::from("/etc/ssl/cert.pem")),
            tls_key_path: Some(PathBuf::from("/etc/ssl/key.pem")),
            quic_enabled: true,
            quic_port: Some(5444),
            quic_zero_rtt: true,
            quic_idle_timeout_secs: 120,
            dual_stack: false,
        };

        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.port, 5433);
        assert_eq!(config.max_connections, 500);
        assert!(config.tls_enabled);
        assert!(config.tls_cert_path.is_some());
    }

    #[test]
    fn test_server_config_clone() {
        let config1 = ServerConfig::default();
        let config2 = config1.clone();
        assert_eq!(config1.port, config2.port);
        assert_eq!(config1.host, config2.host);
    }

    #[test]
    fn test_server_config_serde_roundtrip() {
        let original = ServerConfig::default();
        let serialized = serde_json::to_string(&original).unwrap();
        let deserialized: ServerConfig = serde_json::from_str(&serialized).unwrap();

        assert_eq!(original.host, deserialized.host);
        assert_eq!(original.port, deserialized.port);
        assert_eq!(original.max_connections, deserialized.max_connections);
        assert_eq!(original.tls_enabled, deserialized.tls_enabled);
    }

    #[test]
    fn test_storage_config_defaults() {
        let config = StorageConfig::default();
        assert_eq!(config.data_dir, PathBuf::from("./data"));
        assert_eq!(config.wal_dir, PathBuf::from("./data/wal"));
        assert_eq!(config.page_size, PAGE_SIZE);
        assert_eq!(config.page_size, 16384);
        assert_eq!(config.buffer_pool_pages, 8192);
        assert_eq!(config.wal_segment_size, 16 * 1024 * 1024);
        assert_eq!(config.checkpoint_interval_secs, 300);
        assert!(config.fsync_enabled);
        assert!(!config.direct_io);
    }

    #[test]
    fn test_storage_config_custom() {
        let config = StorageConfig {
            data_dir: PathBuf::from("/var/lib/zyrondb"),
            wal_dir: PathBuf::from("/var/lib/zyrondb/wal"),
            page_size: 8192,
            buffer_pool_pages: 16384,
            wal_segment_size: 64 * 1024 * 1024,
            checkpoint_interval_secs: 600,
            fsync_enabled: true,
            direct_io: true,
        };

        assert_eq!(config.data_dir, PathBuf::from("/var/lib/zyrondb"));
        assert_eq!(config.page_size, 8192);
        assert!(config.direct_io);
    }

    #[test]
    fn test_buffer_pool_size_bytes() {
        let config = StorageConfig::default();
        let expected = config.buffer_pool_pages * config.page_size;
        assert_eq!(config.buffer_pool_size_bytes(), expected);

        // 8192 pages * 16384 bytes = 128 MB
        assert_eq!(config.buffer_pool_size_bytes(), 8192 * 16384);
        assert_eq!(config.buffer_pool_size_bytes(), 134_217_728);
    }

    #[test]
    fn test_buffer_pool_size_bytes_custom() {
        let mut config = StorageConfig::default();
        config.buffer_pool_pages = 1024;
        config.page_size = 8192;

        assert_eq!(config.buffer_pool_size_bytes(), 1024 * 8192);
        assert_eq!(config.buffer_pool_size_bytes(), 8_388_608); // 8 MB
    }

    #[test]
    fn test_storage_config_clone() {
        let config1 = StorageConfig::default();
        let config2 = config1.clone();
        assert_eq!(config1.page_size, config2.page_size);
        assert_eq!(config1.data_dir, config2.data_dir);
    }

    #[test]
    fn test_storage_config_serde_roundtrip() {
        let original = StorageConfig::default();
        let serialized = serde_json::to_string(&original).unwrap();
        let deserialized: StorageConfig = serde_json::from_str(&serialized).unwrap();

        assert_eq!(original.data_dir, deserialized.data_dir);
        assert_eq!(original.page_size, deserialized.page_size);
        assert_eq!(original.buffer_pool_pages, deserialized.buffer_pool_pages);
    }

    #[test]
    fn test_deployment_mode_parses_every_name_case_insensitively() {
        assert_eq!(DeploymentMode::parse("db"), Some(DeploymentMode::Db));
        assert_eq!(DeploymentMode::parse("LAKE"), Some(DeploymentMode::Lake));
        assert_eq!(
            DeploymentMode::parse(" Unified "),
            Some(DeploymentMode::Unified)
        );
        assert_eq!(DeploymentMode::parse("hybrid"), None);
        assert_eq!(DeploymentMode::parse(""), None);
        for mode in [
            DeploymentMode::Db,
            DeploymentMode::Lake,
            DeploymentMode::Unified,
        ] {
            assert_eq!(DeploymentMode::parse(mode.as_str()), Some(mode));
        }
    }

    #[test]
    fn test_deployment_mode_gates_match_the_deployment_table() {
        // db stores heap only, lake stores ZyronLake only, unified runs both
        // with heap as the unqualified default
        let db = DeploymentMode::Db;
        assert!(db.allows_heap() && !db.allows_lake());
        assert!(!db.defaults_to_lake());
        assert!(!db.runs_lake_tier());

        let lake = DeploymentMode::Lake;
        assert!(lake.allows_lake() && !lake.allows_heap());
        assert!(lake.defaults_to_lake());
        assert!(lake.runs_lake_tier());

        let unified = DeploymentMode::Unified;
        assert!(unified.allows_lake() && unified.allows_heap());
        assert!(!unified.defaults_to_lake());
        assert!(unified.runs_lake_tier());

        assert_eq!(DeploymentMode::default(), DeploymentMode::Unified);
    }

    #[test]
    fn test_num_cpus() {
        let cpus = num_cpus();
        assert!(cpus >= 1, "Should have at least 1 CPU");
    }

    #[test]
    fn test_server_config_with_tls() {
        let config = ServerConfig {
            tls_enabled: true,
            tls_cert_path: Some(PathBuf::from("/path/to/cert.pem")),
            tls_key_path: Some(PathBuf::from("/path/to/key.pem")),
            ..Default::default()
        };

        assert!(config.tls_enabled);
        assert_eq!(
            config.tls_cert_path,
            Some(PathBuf::from("/path/to/cert.pem"))
        );
        assert_eq!(config.tls_key_path, Some(PathBuf::from("/path/to/key.pem")));
    }
}

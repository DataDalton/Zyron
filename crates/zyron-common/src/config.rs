//! Configuration structures for ZyronDB.

use crate::page::PAGE_SIZE;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Server configuration for the ZyronDB instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Host address to bind to.
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
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 5432,
            max_connections: 100,
            connection_timeout_secs: 30,
            statement_timeout_secs: 0,
            worker_threads: num_cpus(),
            tls_enabled: false,
            tls_cert_path: None,
            tls_key_path: None,
        }
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
    /// Compression algorithm for pages.
    pub compression: CompressionType,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("./data"),
            wal_dir: PathBuf::from("./data/wal"),
            page_size: PAGE_SIZE,
            buffer_pool_pages: 8192, // 128 MB with 16 KB pages
            wal_segment_size: 16 * 1024 * 1024, // 16 MB
            checkpoint_interval_secs: 300, // 5 minutes
            fsync_enabled: true,
            direct_io: false,
            compression: CompressionType::None,
        }
    }
}

impl StorageConfig {
    /// Returns the total buffer pool size in bytes.
    pub fn buffer_pool_size_bytes(&self) -> usize {
        self.buffer_pool_pages * self.page_size
    }
}

/// Compression algorithm for page data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum CompressionType {
    /// No compression.
    #[default]
    None,
    /// LZ4 compression (fast, moderate ratio).
    Lz4,
    /// Zstd compression (slower, better ratio).
    Zstd,
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
        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.port, 5432);
        assert_eq!(config.max_connections, 100);
        assert_eq!(config.connection_timeout_secs, 30);
        assert_eq!(config.statement_timeout_secs, 0);
        assert!(config.worker_threads >= 1);
        assert!(!config.tls_enabled);
        assert!(config.tls_cert_path.is_none());
        assert!(config.tls_key_path.is_none());
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
        assert_eq!(config.compression, CompressionType::None);
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
            compression: CompressionType::Lz4,
        };

        assert_eq!(config.data_dir, PathBuf::from("/var/lib/zyrondb"));
        assert_eq!(config.page_size, 8192);
        assert_eq!(config.compression, CompressionType::Lz4);
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
        assert_eq!(original.compression, deserialized.compression);
    }

    #[test]
    fn test_compression_type_default() {
        let compression = CompressionType::default();
        assert_eq!(compression, CompressionType::None);
    }

    #[test]
    fn test_compression_type_variants() {
        assert_ne!(CompressionType::None, CompressionType::Lz4);
        assert_ne!(CompressionType::Lz4, CompressionType::Zstd);
        assert_ne!(CompressionType::None, CompressionType::Zstd);
    }

    #[test]
    fn test_compression_type_clone_copy() {
        let c1 = CompressionType::Lz4;
        let c2 = c1; // Copy
        let c3 = c1.clone(); // Clone
        assert_eq!(c1, c2);
        assert_eq!(c1, c3);
    }

    #[test]
    fn test_compression_type_serde_roundtrip() {
        for compression in [
            CompressionType::None,
            CompressionType::Lz4,
            CompressionType::Zstd,
        ] {
            let serialized = serde_json::to_string(&compression).unwrap();
            let deserialized: CompressionType = serde_json::from_str(&serialized).unwrap();
            assert_eq!(compression, deserialized);
        }
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
        assert_eq!(
            config.tls_key_path,
            Some(PathBuf::from("/path/to/key.pem"))
        );
    }

    #[test]
    fn test_storage_config_with_compression() {
        let config_lz4 = StorageConfig {
            compression: CompressionType::Lz4,
            ..Default::default()
        };
        assert_eq!(config_lz4.compression, CompressionType::Lz4);

        let config_zstd = StorageConfig {
            compression: CompressionType::Zstd,
            ..Default::default()
        };
        assert_eq!(config_zstd.compression, CompressionType::Zstd);
    }
}

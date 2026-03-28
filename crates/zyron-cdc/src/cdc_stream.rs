//! Outbound CDC streams that deliver change events to external sinks.
//!
//! Each stream is backed by a replication slot and a configurable sink
//! (Kafka, S3, or Webhook). Changes are batched in memory until batch_size
//! or batch_interval triggers a flush to the sink.

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use bytes::Bytes;
use scc::HashMap as SccHashMap;
use serde::{Deserialize, Serialize};
use zyron_common::{Result, ZyronError};

// ---------------------------------------------------------------------------
// OutputFormat
// ---------------------------------------------------------------------------

/// Output format for CDC data written to sinks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputFormat {
    Json,
    Parquet,
    Avro,
}

// ---------------------------------------------------------------------------
// CdcSinkConfig
// ---------------------------------------------------------------------------

/// Sink configuration for outbound CDC streams.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CdcSinkConfig {
    Kafka {
        brokers: String,
        topic: String,
        key_columns: Vec<String>,
    },
    S3 {
        bucket: String,
        prefix: String,
        region: String,
        format: OutputFormat,
        partition_by: Option<String>,
    },
    Webhook {
        url: String,
        headers: Vec<(String, String)>,
        batch_size: usize,
    },
}

// ---------------------------------------------------------------------------
// SinkCheckpoint
// ---------------------------------------------------------------------------

/// Tracks delivery progress for exactly-once semantics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SinkCheckpoint {
    pub stream_name: String,
    pub last_confirmed_lsn: u64,
    pub sink_specific_offset: Option<String>,
    pub last_flush_timestamp: i64,
}

// ---------------------------------------------------------------------------
// StreamRetryPolicy
// ---------------------------------------------------------------------------

/// Retry policy for failed sink writes with exponential backoff.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamRetryPolicy {
    pub max_retries: u32,
    pub initial_backoff_ms: u64,
    pub max_backoff_ms: u64,
    pub backoff_multiplier: f64,
}

impl Default for StreamRetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 10,
            initial_backoff_ms: 100,
            max_backoff_ms: 30_000,
            backoff_multiplier: 2.0,
        }
    }
}

// ---------------------------------------------------------------------------
// StreamStatus
// ---------------------------------------------------------------------------

/// Runtime status of a CDC stream.
#[derive(Debug, Clone)]
pub struct StreamStatus {
    pub name: String,
    pub active: bool,
    pub last_lsn: u64,
    pub records_sent: u64,
    pub last_error: Option<String>,
}

// ---------------------------------------------------------------------------
// CdcSink trait
// ---------------------------------------------------------------------------

/// Trait for CDC sink implementations that receive change batches.
pub trait CdcSink: Send + Sync {
    /// Writes a batch of serialized changes to the sink.
    fn write_batch(&self, changes: &[Bytes]) -> Result<()>;

    /// Flushes any buffered data.
    fn flush(&self) -> Result<()>;

    /// Returns the current checkpoint (delivery progress).
    fn checkpoint(&self) -> Result<SinkCheckpoint>;
}

// ---------------------------------------------------------------------------
// Stub sink implementations
// ---------------------------------------------------------------------------

/// Stub Kafka sink. Real implementation connects to Kafka brokers.
pub struct KafkaSink {
    pub config: CdcSinkConfig,
    stream_name: String,
    last_lsn: AtomicU64,
}

impl KafkaSink {
    pub fn new(config: CdcSinkConfig, stream_name: String) -> Self {
        Self {
            config,
            stream_name,
            last_lsn: AtomicU64::new(0),
        }
    }
}

impl CdcSink for KafkaSink {
    fn write_batch(&self, _changes: &[Bytes]) -> Result<()> {
        // Stub: real implementation produces to Kafka topic.
        Ok(())
    }

    fn flush(&self) -> Result<()> {
        Ok(())
    }

    fn checkpoint(&self) -> Result<SinkCheckpoint> {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as i64;
        Ok(SinkCheckpoint {
            stream_name: self.stream_name.clone(),
            last_confirmed_lsn: self.last_lsn.load(Ordering::Relaxed),
            sink_specific_offset: None,
            last_flush_timestamp: ts,
        })
    }
}

/// Stub S3 sink. Real implementation writes to S3 bucket.
pub struct S3Sink {
    pub config: CdcSinkConfig,
    stream_name: String,
    last_lsn: AtomicU64,
}

impl S3Sink {
    pub fn new(config: CdcSinkConfig, stream_name: String) -> Self {
        Self {
            config,
            stream_name,
            last_lsn: AtomicU64::new(0),
        }
    }
}

impl CdcSink for S3Sink {
    fn write_batch(&self, _changes: &[Bytes]) -> Result<()> {
        Ok(())
    }

    fn flush(&self) -> Result<()> {
        Ok(())
    }

    fn checkpoint(&self) -> Result<SinkCheckpoint> {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as i64;
        Ok(SinkCheckpoint {
            stream_name: self.stream_name.clone(),
            last_confirmed_lsn: self.last_lsn.load(Ordering::Relaxed),
            sink_specific_offset: None,
            last_flush_timestamp: ts,
        })
    }
}

/// Stub Webhook sink. Real implementation POSTs to HTTP endpoint.
pub struct WebhookSink {
    pub config: CdcSinkConfig,
    stream_name: String,
    last_lsn: AtomicU64,
}

impl WebhookSink {
    pub fn new(config: CdcSinkConfig, stream_name: String) -> Self {
        Self {
            config,
            stream_name,
            last_lsn: AtomicU64::new(0),
        }
    }
}

impl CdcSink for WebhookSink {
    fn write_batch(&self, _changes: &[Bytes]) -> Result<()> {
        Ok(())
    }

    fn flush(&self) -> Result<()> {
        Ok(())
    }

    fn checkpoint(&self) -> Result<SinkCheckpoint> {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as i64;
        Ok(SinkCheckpoint {
            stream_name: self.stream_name.clone(),
            last_confirmed_lsn: self.last_lsn.load(Ordering::Relaxed),
            sink_specific_offset: None,
            last_flush_timestamp: ts,
        })
    }
}

// ---------------------------------------------------------------------------
// CdcOutputStream
// ---------------------------------------------------------------------------

/// An outbound CDC stream definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdcOutputStream {
    pub name: String,
    pub table_id: u32,
    pub slot_name: String,
    pub sink: CdcSinkConfig,
    pub decoder_plugin: crate::decoder::DecoderPlugin,
    pub filter: Option<String>,
    pub include_columns: Option<Vec<String>>,
    pub batch_size: usize,
    pub batch_interval_ms: u64,
    pub active: bool,
    pub retry_policy: StreamRetryPolicy,
}

// ---------------------------------------------------------------------------
// CdcStreamManager
// ---------------------------------------------------------------------------

/// Manages outbound CDC streams.
pub struct CdcStreamManager {
    streams: SccHashMap<String, CdcOutputStream>,
    state_file: PathBuf,
}

impl CdcStreamManager {
    /// Opens or creates the stream manager, loading persisted state.
    pub fn new(data_dir: &Path) -> Result<Self> {
        let state_file = data_dir.join(".zystreams");

        let streams = SccHashMap::new();

        if state_file.exists() {
            let mut file = File::open(&state_file)?;
            let mut data = Vec::new();
            file.read_to_end(&mut data)?;
            if !data.is_empty() {
                let list: Vec<CdcOutputStream> = serde_json::from_slice(&data).map_err(|e| {
                    ZyronError::CdcStreamError(format!("failed to parse stream state: {e}"))
                })?;
                for stream in list {
                    let _ = streams.insert_sync(stream.name.clone(), stream);
                }
            }
        }

        Ok(Self {
            streams,
            state_file,
        })
    }

    /// Creates a new outbound CDC stream.
    pub fn create_stream(&self, stream: CdcOutputStream) -> Result<()> {
        if self
            .streams
            .insert_sync(stream.name.clone(), stream)
            .is_err()
        {
            return Err(ZyronError::CdcStreamError("stream already exists".into()));
        }
        self.persist()?;
        Ok(())
    }

    /// Drops an outbound CDC stream.
    pub fn drop_stream(&self, name: &str) -> Result<()> {
        self.streams
            .remove_sync(name)
            .ok_or_else(|| ZyronError::CdcStreamError(format!("stream not found: {name}")))?;
        self.persist()?;
        Ok(())
    }

    /// Lists all outbound CDC streams.
    pub fn list_streams(&self) -> Vec<CdcOutputStream> {
        let mut result = Vec::new();
        self.streams.iter_sync(|_name, stream| {
            result.push(stream.clone());
            true
        });
        result
    }

    /// Gets a stream by name.
    pub fn get_stream(&self, name: &str) -> Result<CdcOutputStream> {
        self.streams
            .read_sync(name, |_, stream| stream.clone())
            .ok_or_else(|| ZyronError::CdcStreamError(format!("stream not found: {name}")))
    }

    /// Removes all streams targeting the given table_id.
    pub fn remove_streams_for_table(&self, table_id: u32) -> Result<Vec<String>> {
        let mut to_remove = Vec::new();
        self.streams.iter_sync(|name, stream| {
            if stream.table_id == table_id {
                to_remove.push(name.clone());
            }
            true
        });

        for name in &to_remove {
            let _ = self.streams.remove_sync(name);
        }

        if !to_remove.is_empty() {
            self.persist()?;
        }

        Ok(to_remove)
    }

    /// Persists stream state to disk using atomic rename.
    fn persist(&self) -> Result<()> {
        let streams = self.list_streams();
        let data = serde_json::to_vec(&streams).map_err(|e| {
            ZyronError::CdcStreamError(format!("failed to serialize stream state: {e}"))
        })?;

        let tmp_path = self.state_file.with_extension("zystreams.tmp");
        {
            let mut tmp = File::create(&tmp_path)?;
            tmp.write_all(&data)?;
            tmp.sync_all()?;
        }

        fs::rename(&tmp_path, &self.state_file)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::DecoderPlugin;
    use tempfile::TempDir;

    fn sample_stream() -> CdcOutputStream {
        CdcOutputStream {
            name: "test_stream".into(),
            table_id: 42,
            slot_name: "test_slot".into(),
            sink: CdcSinkConfig::Kafka {
                brokers: "localhost:9092".into(),
                topic: "cdc_events".into(),
                key_columns: vec!["id".into()],
            },
            decoder_plugin: DecoderPlugin::Debezium,
            filter: None,
            include_columns: None,
            batch_size: 1000,
            batch_interval_ms: 100,
            active: true,
            retry_policy: StreamRetryPolicy::default(),
        }
    }

    #[test]
    fn test_create_and_list_streams() {
        let tmp = TempDir::new().unwrap();
        let mgr = CdcStreamManager::new(tmp.path()).unwrap();

        mgr.create_stream(sample_stream()).unwrap();
        let streams = mgr.list_streams();
        assert_eq!(streams.len(), 1);
        assert_eq!(streams[0].name, "test_stream");
    }

    #[test]
    fn test_create_duplicate_stream_fails() {
        let tmp = TempDir::new().unwrap();
        let mgr = CdcStreamManager::new(tmp.path()).unwrap();

        mgr.create_stream(sample_stream()).unwrap();
        assert!(mgr.create_stream(sample_stream()).is_err());
    }

    #[test]
    fn test_drop_stream() {
        let tmp = TempDir::new().unwrap();
        let mgr = CdcStreamManager::new(tmp.path()).unwrap();

        mgr.create_stream(sample_stream()).unwrap();
        mgr.drop_stream("test_stream").unwrap();
        assert!(mgr.list_streams().is_empty());

        assert!(mgr.drop_stream("nonexistent").is_err());
    }

    #[test]
    fn test_persistence() {
        let tmp = TempDir::new().unwrap();

        {
            let mgr = CdcStreamManager::new(tmp.path()).unwrap();
            mgr.create_stream(sample_stream()).unwrap();
        }

        let mgr = CdcStreamManager::new(tmp.path()).unwrap();
        let streams = mgr.list_streams();
        assert_eq!(streams.len(), 1);
        assert_eq!(streams[0].name, "test_stream");
    }

    #[test]
    fn test_remove_streams_for_table() {
        let tmp = TempDir::new().unwrap();
        let mgr = CdcStreamManager::new(tmp.path()).unwrap();

        let mut s1 = sample_stream();
        s1.name = "s1".into();
        s1.table_id = 42;

        let mut s2 = sample_stream();
        s2.name = "s2".into();
        s2.table_id = 43;

        mgr.create_stream(s1).unwrap();
        mgr.create_stream(s2).unwrap();

        let removed = mgr.remove_streams_for_table(42).unwrap();
        assert_eq!(removed, vec!["s1"]);
        assert_eq!(mgr.list_streams().len(), 1);
    }

    #[test]
    fn test_sink_stubs() {
        let kafka = KafkaSink::new(
            CdcSinkConfig::Kafka {
                brokers: "localhost:9092".into(),
                topic: "test".into(),
                key_columns: vec![],
            },
            "test".into(),
        );
        kafka.write_batch(&[]).unwrap();
        kafka.flush().unwrap();
        let cp = kafka.checkpoint().unwrap();
        assert_eq!(cp.stream_name, "test");

        let s3 = S3Sink::new(
            CdcSinkConfig::S3 {
                bucket: "bucket".into(),
                prefix: "prefix".into(),
                region: "us-east-1".into(),
                format: OutputFormat::Json,
                partition_by: None,
            },
            "s3_test".into(),
        );
        s3.write_batch(&[]).unwrap();

        let wh = WebhookSink::new(
            CdcSinkConfig::Webhook {
                url: "http://localhost:8080".into(),
                headers: vec![],
                batch_size: 100,
            },
            "wh_test".into(),
        );
        wh.write_batch(&[]).unwrap();
    }

    #[test]
    fn test_retry_policy_default() {
        let policy = StreamRetryPolicy::default();
        assert_eq!(policy.max_retries, 10);
        assert_eq!(policy.initial_backoff_ms, 100);
        assert_eq!(policy.max_backoff_ms, 30_000);
    }
}

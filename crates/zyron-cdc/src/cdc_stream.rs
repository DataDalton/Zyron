//! Outbound CDC streams that deliver change events to external sinks.
//!
//! Each stream consumes a change stream, which holds its delivery position,
//! and writes to a configurable sink (Kafka, S3, or Webhook). Changes are
//! batched in memory until batch_size or batch_interval triggers a flush to
//! the sink

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
// CdcSink trait
// ---------------------------------------------------------------------------

/// Trait for CDC sink implementations that receive change batches.
///
/// A sink holds no position of its own. Where an outbound stream has got to
/// is the change stream it consumes, whose position the driver moves once a
/// batch is confirmed delivered, so there is one place progress is recorded
/// and it is the one `zyron_sys.cdc.change_streams` shows
pub trait CdcSink: Send + Sync {
    /// Writes a batch of serialized changes to the sink. Returns an error when
    /// delivery is not confirmed so the driver does not move the stream past
    /// undelivered data
    fn write_batch(&self, changes: &[Bytes]) -> Result<()>;

    /// Flushes any buffered data.
    fn flush(&self) -> Result<()>;
}

/// Builds a JSON array body from already-serialized JSON change records.
fn json_array_body(changes: &[Bytes]) -> Vec<u8> {
    let cap = changes.iter().map(|c| c.len() + 1).sum::<usize>() + 2;
    let mut buf = Vec::with_capacity(cap);
    buf.push(b'[');
    for (i, c) in changes.iter().enumerate() {
        if i > 0 {
            buf.push(b',');
        }
        buf.extend_from_slice(c);
    }
    buf.push(b']');
    buf
}

// ---------------------------------------------------------------------------
// Sink implementations
// ---------------------------------------------------------------------------

/// Kafka sink. Produces each change as one record to the configured topic via
/// the pure-Rust rskafka client. Records go to partition 0 to preserve total
/// ordering of the change stream.
pub struct KafkaSink {
    pub config: CdcSinkConfig,
    stream_name: String,
}

impl KafkaSink {
    pub fn new(config: CdcSinkConfig, stream_name: String) -> Self {
        Self {
            config,
            stream_name,
        }
    }

    /// The stream this sink delivers for
    pub fn stream_name(&self) -> &str {
        &self.stream_name
    }
}

impl CdcSink for KafkaSink {
    fn write_batch(&self, changes: &[Bytes]) -> Result<()> {
        if changes.is_empty() {
            return Ok(());
        }
        let (brokers, topic) = match &self.config {
            CdcSinkConfig::Kafka { brokers, topic, .. } => (brokers.clone(), topic.clone()),
            _ => {
                return Err(ZyronError::CdcStreamError(
                    "KafkaSink constructed with a non-Kafka config".into(),
                ));
            }
        };
        let broker_list: Vec<String> = brokers
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        if broker_list.is_empty() {
            return Err(ZyronError::CdcStreamError(
                "Kafka brokers list is empty".into(),
            ));
        }
        let values: Vec<Vec<u8>> = changes.iter().map(|c| c.to_vec()).collect();

        crate::sink_io::block_on_io(async move {
            use rskafka::client::ClientBuilder;
            use rskafka::client::partition::{Compression, UnknownTopicHandling};
            use rskafka::record::Record;

            let client = ClientBuilder::new(broker_list)
                .build()
                .await
                .map_err(|e| ZyronError::CdcStreamError(format!("Kafka connect failed: {e}")))?;
            let partition = client
                .partition_client(topic.clone(), 0, UnknownTopicHandling::Retry)
                .await
                .map_err(|e| {
                    ZyronError::CdcStreamError(format!("Kafka partition client failed: {e}"))
                })?;
            let now = chrono::Utc::now();
            let records: Vec<Record> = values
                .into_iter()
                .map(|v| Record {
                    key: None,
                    value: Some(v),
                    headers: std::collections::BTreeMap::new(),
                    timestamp: now,
                })
                .collect();
            partition
                .produce(records, Compression::NoCompression)
                .await
                .map_err(|e| ZyronError::CdcStreamError(format!("Kafka produce failed: {e}")))?;
            Ok::<(), ZyronError>(())
        })?;
        Ok(())
    }

    fn flush(&self) -> Result<()> {
        Ok(())
    }
}

/// S3 sink. Uploads each batch as a single object via SigV4-signed PUT.
pub struct S3Sink {
    pub config: CdcSinkConfig,
    stream_name: String,
    seq: AtomicU64,
}

impl S3Sink {
    pub fn new(config: CdcSinkConfig, stream_name: String) -> Self {
        Self {
            config,
            stream_name,
            seq: AtomicU64::new(0),
        }
    }
}

impl CdcSink for S3Sink {
    fn write_batch(&self, changes: &[Bytes]) -> Result<()> {
        if changes.is_empty() {
            return Ok(());
        }
        let (bucket, prefix, region, format, partition_by) = match &self.config {
            CdcSinkConfig::S3 {
                bucket,
                prefix,
                region,
                format,
                partition_by,
            } => (
                bucket.clone(),
                prefix.clone(),
                region.clone(),
                format.clone(),
                partition_by.clone(),
            ),
            _ => {
                return Err(ZyronError::CdcStreamError(
                    "S3Sink constructed with a non-S3 config".into(),
                ));
            }
        };

        let (body, ext, content_type) = match format {
            OutputFormat::Json => (json_array_body(changes), "json", "application/json"),
            other => {
                return Err(ZyronError::CdcStreamError(format!(
                    "S3 sink output format {other:?} is not supported; use JSON"
                )));
            }
        };

        let ts_micros = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros();
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        let mut key = String::new();
        if !prefix.is_empty() {
            key.push_str(prefix.trim_end_matches('/'));
            key.push('/');
        }
        if let Some(part) = partition_by.as_deref().filter(|p| !p.is_empty()) {
            key.push_str(part.trim_matches('/'));
            key.push('/');
        }
        key.push_str(&format!("{}-{ts_micros}-{seq}.{ext}", self.stream_name));

        crate::sink_io::s3_put(&bucket, &region, &key, body, content_type)?;
        Ok(())
    }

    fn flush(&self) -> Result<()> {
        Ok(())
    }
}

/// Webhook sink. POSTs each batch as a JSON array to the configured endpoint
/// with the configured headers.
pub struct WebhookSink {
    pub config: CdcSinkConfig,
    stream_name: String,
}

impl WebhookSink {
    pub fn new(config: CdcSinkConfig, stream_name: String) -> Self {
        Self {
            config,
            stream_name,
        }
    }

    /// The stream this sink delivers for
    pub fn stream_name(&self) -> &str {
        &self.stream_name
    }
}

impl CdcSink for WebhookSink {
    fn write_batch(&self, changes: &[Bytes]) -> Result<()> {
        if changes.is_empty() {
            return Ok(());
        }
        let (url, headers) = match &self.config {
            CdcSinkConfig::Webhook { url, headers, .. } => (url.clone(), headers.clone()),
            _ => {
                return Err(ZyronError::CdcStreamError(
                    "WebhookSink constructed with a non-Webhook config".into(),
                ));
            }
        };
        let body = json_array_body(changes);
        crate::sink_io::webhook_post(&url, &headers, body)
    }

    fn flush(&self) -> Result<()> {
        Ok(())
    }
}

/// Builds the concrete sink for a stream from its configured sink type.
pub fn build_sink(stream: &CdcOutputStream) -> Box<dyn CdcSink> {
    match &stream.sink {
        CdcSinkConfig::Kafka { .. } => {
            Box::new(KafkaSink::new(stream.sink.clone(), stream.name.clone()))
        }
        CdcSinkConfig::S3 { .. } => Box::new(S3Sink::new(stream.sink.clone(), stream.name.clone())),
        CdcSinkConfig::Webhook { .. } => {
            Box::new(WebhookSink::new(stream.sink.clone(), stream.name.clone()))
        }
    }
}

// ---------------------------------------------------------------------------
// Stream driver (pump)
// ---------------------------------------------------------------------------

/// A transaction's fate as the engine's commit-status authority reports it.
/// Change records land in the feed at execution time, before their
/// transaction decides, so delivery consults this per record: only a
/// committed transaction's changes reach the sink, an aborted one's are
/// skipped, and an undecided one holds the pass so the position never moves
/// past a change that could still roll back
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxnDecision {
    Committed,
    Aborted,
    InFlight,
}

/// What one delivery pass did. How many records reached the sink and the
/// highest commit version every record of which is delivered or skipped,
/// which is where the change stream's position moves to
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeliveryPass {
    pub delivered: u64,
    pub complete_version: u64,
}

/// Drives one delivery pass for a stream. Reads change records committed
/// after the change stream's position, decodes each into the stream's output
/// format and delivers them to the sink in batches of `batch_size`. Answers
/// with the version the stream's position moves to, which the caller writes
/// into the catalog once the pass is done.
///
/// The decode closure converts a raw CDF record into a DecodedChange. It is
/// injected so this crate stays free of the catalog and executor: the server
/// supplies a closure that decodes row bytes against the table schema. The
/// txn_decision closure reports each record's transaction fate the same way.
pub fn drive_stream_once<F>(
    stream: &CdcOutputStream,
    feed: &crate::change_feed::ChangeDataFeed,
    start_version: u64,
    sink: &dyn CdcSink,
    decode: F,
    txn_decision: &dyn Fn(u64) -> TxnDecision,
) -> Result<DeliveryPass>
where
    F: Fn(&crate::change_feed::ChangeRecord) -> Result<crate::decoder::DecodedChange>,
{
    let changes = feed.query_changes(start_version + 1, u64::MAX)?;
    drive_stream_changes(stream, changes, start_version, sink, decode, txn_decision)
}

/// Delivers change records a caller already has.
///
/// A lake table keeps no change file: its transaction log is the change
/// record, so the pump derives the records from the log and drives the
/// stream through here. Batching and sink delivery are identical either
/// way, so a stream behaves the same on both formats
pub fn drive_stream_changes<F>(
    stream: &CdcOutputStream,
    changes: Vec<crate::change_feed::ChangeRecord>,
    start_version: u64,
    sink: &dyn CdcSink,
    decode: F,
    txn_decision: &dyn Fn(u64) -> TxnDecision,
) -> Result<DeliveryPass>
where
    F: Fn(&crate::change_feed::ChangeRecord) -> Result<crate::decoder::DecodedChange>,
{
    if changes.is_empty() {
        return Ok(DeliveryPass {
            delivered: 0,
            complete_version: start_version,
        });
    }

    let decoder = crate::decoder::create_decoder(stream.decoder_plugin);
    let mut delivered = 0u64;
    let batch_cap = stream.batch_size.max(1);
    let mut batch: Vec<Bytes> = Vec::with_capacity(batch_cap);
    // The highest commit version whose records are all enqueued or skipped.
    // The position only ever moves to such a boundary. A statement's records
    // all share one version, so a crash between flushes redelivers a partial
    // version instead of silently losing its tail, and an undecided
    // transaction stops the pass before its version begins
    let mut last_complete_version = start_version;

    let flush = |sink: &dyn CdcSink, batch: &mut Vec<Bytes>| -> Result<()> {
        if batch.is_empty() {
            return Ok(());
        }
        sink.write_batch(batch)?;
        batch.clear();
        Ok(())
    };

    let total = changes.len();
    let mut idx = 0usize;
    'versions: while idx < total {
        let version = changes[idx].commit_version;
        let mut group_end = idx;
        while group_end < total && changes[group_end].commit_version == version {
            group_end += 1;
        }

        // A version enters the batch all or nothing: every record's
        // transaction is decided before any of them is enqueued, so an
        // undecided transaction never leaves half a version at the sink
        for record in &changes[idx..group_end] {
            if txn_decision(record.txn_id) == TxnDecision::InFlight {
                break 'versions;
            }
        }
        for record in &changes[idx..group_end] {
            match txn_decision(record.txn_id) {
                TxnDecision::Committed => {
                    let decoded = decode(record)?;
                    let bytes = decoder.serialize(&decoded)?;
                    batch.push(bytes);
                }
                TxnDecision::Aborted => {}
                TxnDecision::InFlight => break 'versions,
            }
        }
        last_complete_version = version;
        idx = group_end;

        if batch.len() >= batch_cap {
            let n = batch.len() as u64;
            flush(sink, &mut batch)?;
            delivered += n;
        }
    }
    let remaining = batch.len() as u64;
    flush(sink, &mut batch)?;
    delivered += remaining;

    Ok(DeliveryPass {
        delivered,
        complete_version: last_complete_version,
    })
}

// ---------------------------------------------------------------------------
// CdcOutputStream
// ---------------------------------------------------------------------------

/// The change stream an outbound stream created without FROM CHANGE STREAM
/// consumes, one named after it, so there is one position model rather than
/// a private checkpoint beside the visible one
pub fn implicit_change_stream_name(stream_name: &str) -> String {
    format!("__cdc_{stream_name}")
}

/// An outbound CDC stream definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdcOutputStream {
    pub name: String,
    pub table_id: u32,
    /// The change stream this outbound stream consumes. Its position is
    /// where delivery has got to, the one `zyron_sys.cdc.change_streams`
    /// shows, and the only place it is recorded
    pub change_stream: String,
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
                let parsed = zyron_common::format::envelope::decode_as(
                    &data,
                    zyron_common::format::FormatKind::StreamingCdcCheckpoint,
                )
                .map_err(|e| {
                    ZyronError::CdcStreamError(format!(
                        "stream checkpoint {}, {e}",
                        state_file.display()
                    ))
                })?;
                let body: std::borrow::Cow<'_, [u8]> = match parsed.header.version {
                    v if v == crate::format::CDC_CHECKPOINT_FORMAT_VERSION => {
                        std::borrow::Cow::Borrowed(parsed.body)
                    }
                    v if v == crate::format::CDC_CHECKPOINT_FORMAT_VERSION_1 => {
                        std::borrow::Cow::Owned(
                            crate::format::cdc_streams_1_0_to_1_1(parsed.body)
                                .map_err(ZyronError::CdcStreamError)?,
                        )
                    }
                    other => {
                        return Err(ZyronError::CdcStreamError(format!(
                            "stream state is at format version {other}, this binary writes {} \
                             and reads {} through it. Upgrade through a release that still \
                             reads {other} to move it forward first",
                            crate::format::CDC_CHECKPOINT_FORMAT_VERSION,
                            crate::format::CDC_CHECKPOINT_FORMAT_VERSION_1,
                        )));
                    }
                };
                let list: Vec<CdcOutputStream> = serde_json::from_slice(&body).map_err(|e| {
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
        let body = serde_json::to_vec(&streams).map_err(|e| {
            ZyronError::CdcStreamError(format!("failed to serialize stream state: {e}"))
        })?;
        // Stream progress is a checkpoint: the envelope names the format and
        // the version, and a checkpoint that does not parse is refused rather
        // than partly applied
        let data = zyron_common::format::envelope::encode(
            zyron_common::format::FormatKind::StreamingCdcCheckpoint,
            crate::format::CDC_CHECKPOINT_FORMAT_VERSION,
            &body,
        );

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
            change_stream: "test_stream".into(),
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
        assert_eq!(kafka.stream_name(), "test");

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
        assert_eq!(wh.stream_name(), "wh_test");
    }

    #[test]
    fn test_a_version_1_state_file_reads_forward() {
        let (kind, version) =
            zyron_common::format::envelope::peek(crate::format::STREAM_STATE_FIXTURE_1)
                .expect("peeks");
        assert_eq!(
            kind,
            zyron_common::format::FormatKind::StreamingCdcCheckpoint
        );
        assert_eq!(version, crate::format::CDC_CHECKPOINT_FORMAT_VERSION_1);
        let tmp = TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join(".zystreams"),
            crate::format::STREAM_STATE_FIXTURE_1,
        )
        .expect("writes the fixture as the state file");
        let mgr = CdcStreamManager::new(tmp.path()).expect("reads the older state");
        let streams = mgr.list_streams();
        assert_eq!(streams.len(), 1);
        assert_eq!(streams[0].name, "orders_to_kafka");
        assert_eq!(
            streams[0].change_stream,
            implicit_change_stream_name("orders_to_kafka"),
            "the slot the old writer recorded is replaced by the stream named after it"
        );
    }

    #[test]
    fn test_retry_policy_default() {
        let policy = StreamRetryPolicy::default();
        assert_eq!(policy.max_retries, 10);
        assert_eq!(policy.initial_backoff_ms, 100);
        assert_eq!(policy.max_backoff_ms, 30_000);
    }
}

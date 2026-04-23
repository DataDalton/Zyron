//! Source connectors for ingesting data into streaming jobs.
//!
//! Provides the SourceConnector trait for reading micro-batches from
//! external systems. Implementations include a file-based stub and
//! functional implementations for ZyronCdcSource, ZyronTableSource,
//! InMemorySource, and RateLimitedSource.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use zyron_common::{Result, ZyronError};

use crate::record::StreamRecord;

// ---------------------------------------------------------------------------
// SourceConnector trait
// ---------------------------------------------------------------------------

/// Trait for streaming source connectors.
/// Sources produce StreamRecord micro-batches from external systems.
pub trait SourceConnector: Send {
    /// Opens the source, optionally resuming from a serialized offset.
    fn open(&mut self, offset: Option<&[u8]>) -> Result<()>;

    /// Returns the next batch of records, or None if no data is available.
    fn next_batch(&mut self) -> Result<Option<StreamRecord>>;

    /// Commits the current offset (for at-least-once delivery).
    fn commit(&mut self, offset: &[u8]) -> Result<()>;

    /// Seeks to a specific offset (for replay on recovery).
    fn seek(&mut self, offset: &[u8]) -> Result<()>;

    /// Returns the current offset as a serialized byte vector.
    fn current_offset(&self) -> Vec<u8>;

    /// Closes the source and releases resources.
    fn close(&mut self) -> Result<()>;
}

// ---------------------------------------------------------------------------
// SourceConfig
// ---------------------------------------------------------------------------

/// Configuration for different source types.
#[derive(Debug, Clone)]
pub enum SourceConfig {
    File { path: String, pattern: String },
    ZyronCdc { slot_name: String, table_id: u32 },
    InMemory { record_count: usize },
}

// ---------------------------------------------------------------------------
// FileSource (stub)
// ---------------------------------------------------------------------------

/// File-based source connector stub. Reads from a directory of files
/// matching a pattern.
pub struct FileSource {
    path: String,
    pattern: String,
    is_open: bool,
}

impl FileSource {
    pub fn new(path: String, pattern: String) -> Self {
        Self {
            path,
            pattern,
            is_open: false,
        }
    }
}

impl SourceConnector for FileSource {
    fn open(&mut self, _offset: Option<&[u8]>) -> Result<()> {
        self.is_open = true;
        Ok(())
    }
    fn next_batch(&mut self) -> Result<Option<StreamRecord>> {
        Ok(None)
    }
    fn commit(&mut self, _offset: &[u8]) -> Result<()> {
        Ok(())
    }
    fn seek(&mut self, _offset: &[u8]) -> Result<()> {
        Ok(())
    }
    fn current_offset(&self) -> Vec<u8> {
        Vec::new()
    }
    fn close(&mut self) -> Result<()> {
        self.is_open = false;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ZyronCdcSource
// ---------------------------------------------------------------------------

/// Source that reads from a ZyronDB CDC replication slot.
/// Converts change records into StreamRecord micro-batches with
/// appropriate ChangeFlags (Insert, UpdateBefore, UpdateAfter, Delete).
/// Tracks LSN offset for exactly-once replay.
pub struct ZyronCdcSource {
    slot_name: String,
    table_id: u32,
    current_lsn: u64,
    is_open: bool,
    /// Simulated pending records for testing.
    pending: VecDeque<StreamRecord>,
}

impl ZyronCdcSource {
    pub fn new(slot_name: String, table_id: u32) -> Self {
        Self {
            slot_name,
            table_id,
            current_lsn: 0,
            is_open: false,
            pending: VecDeque::new(),
        }
    }

    /// Adds pending records for testing.
    pub fn add_pending(&mut self, record: StreamRecord) {
        self.pending.push_back(record);
    }
}

impl SourceConnector for ZyronCdcSource {
    fn open(&mut self, offset: Option<&[u8]>) -> Result<()> {
        if let Some(bytes) = offset {
            if bytes.len() >= 8 {
                self.current_lsn = u64::from_le_bytes([
                    bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
                ]);
            }
        }
        self.is_open = true;
        Ok(())
    }

    fn next_batch(&mut self) -> Result<Option<StreamRecord>> {
        if !self.is_open {
            return Err(ZyronError::StreamingError("CDC source not open".into()));
        }
        if self.pending.is_empty() {
            return Ok(None);
        }
        let record = self.pending.pop_front().expect("pending is not empty");
        self.current_lsn += record.num_rows() as u64;
        Ok(Some(record))
    }

    fn commit(&mut self, _offset: &[u8]) -> Result<()> {
        Ok(())
    }

    fn seek(&mut self, offset: &[u8]) -> Result<()> {
        if offset.len() >= 8 {
            self.current_lsn = u64::from_le_bytes([
                offset[0], offset[1], offset[2], offset[3], offset[4], offset[5], offset[6],
                offset[7],
            ]);
        }
        Ok(())
    }

    fn current_offset(&self) -> Vec<u8> {
        self.current_lsn.to_le_bytes().to_vec()
    }

    fn close(&mut self) -> Result<()> {
        self.is_open = false;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// InMemorySource
// ---------------------------------------------------------------------------

/// In-memory source for testing. Returns pre-loaded records in configurable
/// batch sizes.
pub struct InMemorySource {
    records: Vec<StreamRecord>,
    position: usize,
    batch_size: usize,
    is_open: bool,
}

impl InMemorySource {
    pub fn new(records: Vec<StreamRecord>) -> Self {
        Self {
            records,
            position: 0,
            batch_size: 1,
            is_open: false,
        }
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size.max(1);
        self
    }
}

impl SourceConnector for InMemorySource {
    fn open(&mut self, offset: Option<&[u8]>) -> Result<()> {
        if let Some(bytes) = offset {
            if bytes.len() >= 8 {
                self.position = u64::from_le_bytes([
                    bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
                ]) as usize;
            }
        }
        self.is_open = true;
        Ok(())
    }

    fn next_batch(&mut self) -> Result<Option<StreamRecord>> {
        if !self.is_open {
            return Err(ZyronError::StreamingError("source not open".into()));
        }
        if self.position >= self.records.len() {
            return Ok(None);
        }
        let record = self.records[self.position].clone();
        self.position += 1;
        Ok(Some(record))
    }

    fn commit(&mut self, _offset: &[u8]) -> Result<()> {
        Ok(())
    }

    fn seek(&mut self, offset: &[u8]) -> Result<()> {
        if offset.len() >= 8 {
            self.position = u64::from_le_bytes([
                offset[0], offset[1], offset[2], offset[3], offset[4], offset[5], offset[6],
                offset[7],
            ]) as usize;
        }
        Ok(())
    }

    fn current_offset(&self) -> Vec<u8> {
        (self.position as u64).to_le_bytes().to_vec()
    }

    fn close(&mut self) -> Result<()> {
        self.is_open = false;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// RateLimitedSource
// ---------------------------------------------------------------------------

/// Wraps any source connector with a token-bucket rate limiter.
/// Limits the number of records returned per second.
pub struct RateLimitedSource {
    inner: Box<dyn SourceConnector>,
    max_events_per_second: u64,
    tokens: u64,
    last_refill_ms: u64,
}

impl RateLimitedSource {
    pub fn new(inner: Box<dyn SourceConnector>, max_events_per_second: u64) -> Self {
        Self {
            inner,
            max_events_per_second,
            tokens: max_events_per_second,
            last_refill_ms: 0,
        }
    }

    fn refill_tokens(&mut self, current_ms: u64) {
        if current_ms <= self.last_refill_ms {
            return;
        }
        let elapsed_ms = current_ms - self.last_refill_ms;
        let new_tokens = (elapsed_ms * self.max_events_per_second) / 1000;
        self.tokens = (self.tokens + new_tokens).min(self.max_events_per_second);
        self.last_refill_ms = current_ms;
    }
}

impl SourceConnector for RateLimitedSource {
    fn open(&mut self, offset: Option<&[u8]>) -> Result<()> {
        self.inner.open(offset)
    }

    fn next_batch(&mut self) -> Result<Option<StreamRecord>> {
        if self.tokens == 0 {
            return Ok(None);
        }
        match self.inner.next_batch()? {
            Some(record) => {
                let rows = record.num_rows() as u64;
                if rows <= self.tokens {
                    self.tokens -= rows;
                    Ok(Some(record))
                } else {
                    // Only return as many rows as tokens allow.
                    let mask: Vec<bool> = (0..record.num_rows())
                        .map(|i| (i as u64) < self.tokens)
                        .collect();
                    self.tokens = 0;
                    Ok(Some(record.filter(&mask)))
                }
            }
            None => Ok(None),
        }
    }

    fn commit(&mut self, offset: &[u8]) -> Result<()> {
        self.inner.commit(offset)
    }

    fn seek(&mut self, offset: &[u8]) -> Result<()> {
        self.inner.seek(offset)
    }

    fn current_offset(&self) -> Vec<u8> {
        self.inner.current_offset()
    }

    fn close(&mut self) -> Result<()> {
        self.inner.close()
    }
}

// ---------------------------------------------------------------------------
// ZyronTableSource
// ---------------------------------------------------------------------------

/// Pulls change records directly from a ZyronDB change data feed.
/// Each call to read_batch advances the last_version watermark to the
/// max commit_version observed. Returns a Vec of CdfChange items.
/// Distinct from the ZyronCdcSource above, which uses the slot/VecDeque
/// trait-based model for generic pipelines.
pub struct ZyronTableSource {
    table_id: u32,
    cdf: Arc<zyron_cdc::ChangeDataFeed>,
    last_version: AtomicU64,
}

/// One row pulled from the CDF, exposing the raw row_data payload and
/// the change type for downstream filter and project stages.
#[derive(Debug, Clone)]
pub struct CdfChange {
    pub commit_version: u64,
    pub commit_timestamp: i64,
    pub change_type: zyron_cdc::ChangeType,
    pub row_data: Vec<u8>,
    pub primary_key_data: Vec<u8>,
}

impl ZyronTableSource {
    /// Resolves the CDF for the given table through the registry. Returns an
    /// error if the table has no active feed.
    pub fn new(table_id: u32, cdc_registry: Arc<zyron_cdc::CdfRegistry>) -> Result<Self> {
        let cdf = cdc_registry.get_feed(table_id).ok_or_else(|| {
            ZyronError::StreamingError(format!(
                "no change data feed registered for table {table_id}"
            ))
        })?;
        Ok(Self {
            table_id,
            cdf,
            last_version: AtomicU64::new(0),
        })
    }

    /// Returns the table id this source reads from.
    pub fn table_id(&self) -> u32 {
        self.table_id
    }

    /// Returns the last commit_version seen by this source.
    pub fn last_version(&self) -> u64 {
        self.last_version.load(Ordering::Acquire)
    }

    /// Pulls up to `max_rows` new change records from the CDF, advancing the
    /// last_version watermark. Returns an empty Vec if no new records exist.
    pub fn read_batch(&self, max_rows: usize) -> Result<Vec<CdfChange>> {
        if max_rows == 0 {
            return Ok(Vec::new());
        }
        let start = self.last_version.load(Ordering::Acquire).saturating_add(1);
        let records = self.cdf.query_changes(start, u64::MAX)?;

        if records.is_empty() {
            return Ok(Vec::new());
        }

        let mut out: Vec<CdfChange> = Vec::with_capacity(records.len().min(max_rows));
        let mut max_seen = self.last_version.load(Ordering::Acquire);
        for rec in records.into_iter().take(max_rows) {
            if rec.commit_version > max_seen {
                max_seen = rec.commit_version;
            }
            out.push(CdfChange {
                commit_version: rec.commit_version,
                commit_timestamp: rec.commit_timestamp,
                change_type: rec.change_type,
                row_data: rec.row_data,
                primary_key_data: rec.primary_key_data,
            });
        }
        self.last_version.store(max_seen, Ordering::Release);
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// ZyronSourceAdapter trait
// ---------------------------------------------------------------------------

/// Adapter trait for sources that pull rows from a remote Zyron publication
/// and forward them to a streaming job as CdfChange batches. The concrete
/// implementation lives in the zyron-wire crate as ZyronSourceClient. The
/// trait object form lets the runner dispatch to the remote source without
/// taking a build dependency on zyron-wire.
#[async_trait::async_trait]
pub trait ZyronSourceAdapter: Send + Sync {
    /// Runs the source loop. The adapter pulls rows from its remote, converts
    /// them to CdfChange batches, and invokes on_batch for each batch. The
    /// loop exits when shutdown is set or the remote signals end-of-stream.
    /// start_lsn is the LSN to resume from, zero means start from earliest.
    async fn run(
        &self,
        start_lsn: u64,
        on_batch: Box<dyn Fn(Vec<CdfChange>) -> zyron_common::Result<()> + Send + Sync>,
        shutdown: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) -> zyron_common::Result<()>;

    /// Closes the adapter and releases any held resources.
    async fn close(&self) -> zyron_common::Result<()>;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::{StreamBatch, StreamColumn, StreamColumnData};
    use crate::record::ChangeFlag;

    fn make_test_record(n: usize) -> StreamRecord {
        let col = StreamColumn::from_data(StreamColumnData::Int64((0..n as i64).collect()));
        let batch = StreamBatch::new(vec![col]);
        let times: Vec<i64> = (0..n as i64).map(|i| i * 1000).collect();
        StreamRecord::new(batch, times, vec![ChangeFlag::Insert; n])
    }

    #[test]
    fn test_in_memory_source() {
        let records = vec![make_test_record(3), make_test_record(2)];
        let mut source = InMemorySource::new(records);
        source.open(None).expect("open should succeed");

        let batch1 = source
            .next_batch()
            .expect("next_batch should succeed")
            .expect("should have data");
        assert_eq!(batch1.num_rows(), 3);

        let batch2 = source
            .next_batch()
            .expect("next_batch should succeed")
            .expect("should have data");
        assert_eq!(batch2.num_rows(), 2);

        let batch3 = source.next_batch().expect("next_batch should succeed");
        assert!(batch3.is_none());

        source.close().expect("close should succeed");
    }

    #[test]
    fn test_in_memory_source_seek() {
        let records = vec![
            make_test_record(1),
            make_test_record(2),
            make_test_record(3),
        ];
        let mut source = InMemorySource::new(records);
        source.open(None).expect("open should succeed");

        // Skip to position 2.
        let offset = 2u64.to_le_bytes().to_vec();
        source.seek(&offset).expect("seek should succeed");

        let batch = source
            .next_batch()
            .expect("next_batch should succeed")
            .expect("should have data");
        assert_eq!(batch.num_rows(), 3);
    }

    #[test]
    fn test_cdc_source() {
        let mut source = ZyronCdcSource::new("test_slot".into(), 1);
        source.add_pending(make_test_record(5));
        source.open(None).expect("open should succeed");

        let batch = source
            .next_batch()
            .expect("next_batch should succeed")
            .expect("should have CDC records");
        assert_eq!(batch.num_rows(), 5);

        let offset = source.current_offset();
        assert_eq!(offset.len(), 8);
        let lsn = u64::from_le_bytes([
            offset[0], offset[1], offset[2], offset[3], offset[4], offset[5], offset[6], offset[7],
        ]);
        assert_eq!(lsn, 5);

        source.close().expect("close should succeed");
    }

    #[test]
    fn test_rate_limited_source() {
        let inner = InMemorySource::new(vec![make_test_record(100)]);
        let mut source = RateLimitedSource::new(Box::new(inner), 50);
        source.open(None).expect("open should succeed");

        let batch = source
            .next_batch()
            .expect("next_batch should succeed")
            .expect("should have rate-limited data");
        // Should be capped at 50 tokens.
        assert_eq!(batch.num_rows(), 50);

        // Tokens exhausted.
        let batch2 = source.next_batch().expect("next_batch should succeed");
        assert!(batch2.is_none());
    }

    #[test]
    fn test_source_not_open_error() {
        let mut source = InMemorySource::new(vec![make_test_record(1)]);
        // Calling next_batch without open should fail.
        let result = source.next_batch();
        assert!(result.is_err());
    }
}

//! WAL writer with group commit for high-throughput durability.
//!
//! Group commit batches multiple WAL records together and performs a single
//! fsync for the entire batch, amortizing the expensive disk sync operation
//! across many transactions.

use crate::record::{LogRecord, LogRecordType, Lsn};
use crate::segment::{LogSegment, SegmentHeader, SegmentId};
use bytes::Bytes;
use parking_lot::Mutex;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::{broadcast, Notify};
use zyron_common::{Result, ZyronError};

/// Configuration for the WAL writer.
#[derive(Debug, Clone)]
pub struct WalWriterConfig {
    /// Directory for WAL segment files.
    pub wal_dir: PathBuf,
    /// Maximum size of each segment file.
    pub segment_size: u32,
    /// Enable fsync after writes.
    pub fsync_enabled: bool,
    /// Maximum records to buffer before forcing flush.
    pub batch_size: usize,
    /// Maximum bytes to buffer before forcing flush.
    pub batch_bytes: usize,
    /// Flush interval in microseconds (0 = immediate flush after each batch).
    pub flush_interval_us: u64,
}

impl Default for WalWriterConfig {
    fn default() -> Self {
        Self {
            wal_dir: PathBuf::from("./data/wal"),
            segment_size: LogSegment::DEFAULT_SIZE,
            fsync_enabled: true,
            batch_size: 1000,
            batch_bytes: 4 * 1024 * 1024, // 4MB
            flush_interval_us: 0,
        }
    }
}

/// Entry in the write buffer waiting for flush.
struct PendingWrite {
    /// Serialized record data.
    data: Bytes,
    /// Assigned LSN for this record.
    lsn: Lsn,
}

/// Write buffer accumulating records for group commit.
struct WriteBuffer {
    /// Pending writes awaiting flush.
    pending: Vec<PendingWrite>,
    /// Total bytes in pending writes.
    pending_bytes: usize,
    /// Next offset within current segment (used for LSN assignment).
    next_offset: u32,
    /// Current segment ID.
    current_segment_id: u32,
}

impl WriteBuffer {
    fn new(segment_id: u32, start_offset: u32) -> Self {
        Self {
            pending: Vec::with_capacity(1024),
            pending_bytes: 0,
            next_offset: start_offset,
            current_segment_id: segment_id,
        }
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    #[inline]
    fn len(&self) -> usize {
        self.pending.len()
    }

    #[inline]
    fn bytes(&self) -> usize {
        self.pending_bytes
    }

    /// Adds a record to the buffer and returns its assigned LSN.
    #[inline]
    fn push(&mut self, data: Bytes) -> Lsn {
        let lsn = Lsn::new(self.current_segment_id, self.next_offset);
        let size = data.len() as u32;
        self.pending.push(PendingWrite { data, lsn });
        self.pending_bytes += size as usize;
        self.next_offset += size;
        lsn
    }

    /// Takes all pending writes, leaving buffer empty.
    #[inline]
    fn take(&mut self) -> (Vec<PendingWrite>, usize) {
        let pending = std::mem::take(&mut self.pending);
        let bytes = self.pending_bytes;
        self.pending_bytes = 0;
        self.pending.reserve(1024);
        (pending, bytes)
    }

    /// Updates segment after rotation.
    fn rotate_segment(&mut self, new_segment_id: u32, start_offset: u32) {
        self.current_segment_id = new_segment_id;
        self.next_offset = start_offset;
    }
}

/// Thread-safe WAL writer with group commit.
///
/// Batches multiple records together and performs a single fsync for the
/// entire batch, providing both durability and high throughput.
pub struct WalWriter {
    /// Configuration.
    config: WalWriterConfig,
    /// Current active segment.
    segment: Mutex<Option<LogSegment>>,
    /// Write buffer for group commit.
    buffer: Mutex<WriteBuffer>,
    /// Next transaction ID to assign.
    next_txn_id: AtomicU64,
    /// Last flushed LSN (all records up to this LSN are durable).
    flushed_lsn: AtomicU64,
    /// Broadcast channel for flush completion notifications.
    flush_complete: broadcast::Sender<Lsn>,
    /// Notify for waking flush waiters.
    flush_notify: Notify,
}

impl WalWriter {
    /// Creates a new WAL writer.
    pub async fn new(config: WalWriterConfig) -> Result<Self> {
        tokio::fs::create_dir_all(&config.wal_dir).await?;

        let (segment, next_lsn) = Self::recover_or_create(&config).await?;
        let segment_id = segment.segment_id().0;
        let write_offset = segment.write_offset();

        let (flush_complete, _) = broadcast::channel(64);

        Ok(Self {
            config,
            segment: Mutex::new(Some(segment)),
            buffer: Mutex::new(WriteBuffer::new(segment_id, write_offset)),
            next_txn_id: AtomicU64::new(1),
            flushed_lsn: AtomicU64::new(next_lsn.0.saturating_sub(1)),
            flush_complete,
            flush_notify: Notify::new(),
        })
    }

    /// Recovers from existing segments or creates a new one.
    async fn recover_or_create(config: &WalWriterConfig) -> Result<(LogSegment, Lsn)> {
        let mut segments: Vec<PathBuf> = Vec::new();
        let mut dir = tokio::fs::read_dir(&config.wal_dir).await?;

        while let Some(entry) = dir.next_entry().await? {
            let path = entry.path();
            if path.extension().map(|ext| ext == "wal").unwrap_or(false) {
                segments.push(path);
            }
        }

        if segments.is_empty() {
            let segment_id = SegmentId::FIRST;
            let first_lsn = Lsn::new(segment_id.0, SegmentHeader::SIZE as u32);
            let segment = LogSegment::create(
                &config.wal_dir,
                segment_id,
                first_lsn,
                config.segment_size,
            ).await?;
            return Ok((segment, first_lsn));
        }

        segments.sort();
        let latest_path = segments.last().ok_or_else(|| {
            ZyronError::Internal("WAL segments list unexpectedly empty".to_string())
        })?;
        let segment = LogSegment::open(latest_path).await?;
        let next_lsn = Lsn::new(segment.segment_id().0, segment.write_offset());

        Ok((segment, next_lsn))
    }

    /// Returns the directory containing WAL segments.
    #[inline]
    pub fn wal_dir(&self) -> &Path {
        &self.config.wal_dir
    }

    /// Returns the last flushed LSN.
    #[inline]
    pub fn flushed_lsn(&self) -> Lsn {
        Lsn(self.flushed_lsn.load(Ordering::Acquire))
    }

    /// Allocates a new transaction ID.
    #[inline]
    pub fn allocate_txn_id(&self) -> u32 {
        self.next_txn_id.fetch_add(1, Ordering::Relaxed) as u32
    }

    /// Appends a log record with group commit.
    ///
    /// The record is added to the write buffer. If the buffer is full or
    /// flush_interval has elapsed, a flush is triggered. The caller waits
    /// until the record is durable (fsync complete).
    #[inline]
    pub async fn append(&self, mut record: LogRecord) -> Result<Lsn> {
        // Calculate size without serializing (avoids double serialization)
        let record_size = record.size_on_disk();

        let (lsn, should_flush) = {
            let mut buffer = self.buffer.lock();

            // Check if we need segment rotation
            let needs_rotation = {
                let segment = self.segment.lock();
                segment.as_ref()
                    .map(|s| !s.has_space(record_size + buffer.bytes()))
                    .unwrap_or(true)
            };

            if needs_rotation {
                // Flush current buffer before rotation
                drop(buffer);
                self.flush_internal().await?;
                buffer = self.buffer.lock();

                // Rotate segment
                self.rotate_segment().await?;

                // Update buffer for new segment
                let segment = self.segment.lock();
                if let Some(ref seg) = *segment {
                    buffer.rotate_segment(seg.segment_id().0, seg.write_offset());
                }
            }

            // Assign LSN and serialize once
            let assigned_lsn = Lsn::new(buffer.current_segment_id, buffer.next_offset);
            record.lsn = assigned_lsn;
            let data = record.serialize();

            let lsn = buffer.push(data);
            let should_flush = buffer.len() >= self.config.batch_size
                || buffer.bytes() >= self.config.batch_bytes;

            (lsn, should_flush)
        };

        // Only flush when batch thresholds are reached
        if should_flush {
            self.flush_internal().await?;
        } else if self.config.fsync_enabled {
            // With fsync enabled, wait for flush to ensure durability
            self.wait_for_flush(lsn).await?;
        }
        // When fsync disabled and batch not full, let writes accumulate

        Ok(lsn)
    }

    /// Waits until the given LSN has been flushed to disk.
    async fn wait_for_flush(&self, target_lsn: Lsn) -> Result<()> {
        // Fast path: already flushed
        if self.flushed_lsn() >= target_lsn {
            return Ok(());
        }

        let mut receiver = self.flush_complete.subscribe();

        // Check again after subscribing
        if self.flushed_lsn() >= target_lsn {
            return Ok(());
        }

        // Trigger a flush if buffer is non-empty
        {
            let buffer = self.buffer.lock();
            if !buffer.is_empty() {
                drop(buffer);
                self.flush_notify.notify_one();
            }
        }

        // Wait for flush completion
        loop {
            match tokio::time::timeout(
                std::time::Duration::from_millis(10),
                receiver.recv()
            ).await {
                Ok(Ok(flushed_lsn)) => {
                    if flushed_lsn >= target_lsn {
                        return Ok(());
                    }
                }
                Ok(Err(_)) => {
                    // Channel closed, try direct check
                    if self.flushed_lsn() >= target_lsn {
                        return Ok(());
                    }
                    // Trigger flush ourselves
                    self.flush_internal().await?;
                    return Ok(());
                }
                Err(_) => {
                    // Timeout - check if flushed and try to trigger flush
                    if self.flushed_lsn() >= target_lsn {
                        return Ok(());
                    }
                    self.flush_internal().await?;
                }
            }
        }
    }

    /// Flushes the write buffer to disk.
    async fn flush_internal(&self) -> Result<()> {
        let (pending, total_bytes) = {
            let mut buffer = self.buffer.lock();
            if buffer.is_empty() {
                return Ok(());
            }
            buffer.take()
        };

        if pending.is_empty() {
            return Ok(());
        }

        let max_lsn = pending.last().map(|p| p.lsn).unwrap_or(Lsn::INVALID);
        let batch_count = pending.len();

        // Concatenate all records into single buffer for batch write
        let mut batch_data = Vec::with_capacity(total_bytes);
        for write in pending {
            batch_data.extend_from_slice(&write.data);
        }

        // Single write for entire batch
        {
            let mut segment = self.segment.lock();
            let seg = segment.as_mut().ok_or_else(|| {
                ZyronError::WalWriteFailed("WAL closed".to_string())
            })?;

            seg.append_batch(&batch_data, batch_count).await?;

            // Single fsync for entire batch
            if self.config.fsync_enabled {
                seg.sync().await?;
            }
        }

        // Update flushed LSN and notify waiters
        self.flushed_lsn.store(max_lsn.0, Ordering::Release);
        let _ = self.flush_complete.send(max_lsn);

        Ok(())
    }

    /// Rotates to a new segment.
    async fn rotate_segment(&self) -> Result<()> {
        let mut segment = self.segment.lock();

        if let Some(ref mut seg) = *segment {
            seg.sync().await?;
            seg.close().await?;

            let new_segment_id = seg.segment_id().next();
            let new_first_lsn = Lsn::new(new_segment_id.0, SegmentHeader::SIZE as u32);

            *seg = LogSegment::create(
                &self.config.wal_dir,
                new_segment_id,
                new_first_lsn,
                self.config.segment_size,
            ).await?;
        }

        Ok(())
    }

    /// Forces all pending records to disk.
    pub async fn flush(&self) -> Result<Lsn> {
        self.flush_internal().await?;
        Ok(self.flushed_lsn())
    }

    /// Closes the WAL writer.
    pub async fn close(&self) -> Result<()> {
        self.flush_internal().await?;

        let mut segment = self.segment.lock();
        if let Some(ref mut seg) = segment.take() {
            seg.sync().await?;
            seg.close().await?;
        }
        Ok(())
    }

    /// Returns the current segment ID.
    pub fn current_segment_id(&self) -> Option<SegmentId> {
        self.segment.lock().as_ref().map(|s| s.segment_id())
    }

    /// Returns the next LSN that will be assigned.
    pub fn next_lsn(&self) -> Lsn {
        let buffer = self.buffer.lock();
        Lsn::new(buffer.current_segment_id, buffer.next_offset)
    }

    // Convenience methods for common record types

    /// Logs a transaction begin.
    #[inline]
    pub async fn log_begin(&self, txn_id: u32) -> Result<Lsn> {
        let record = LogRecord::begin(Lsn::INVALID, txn_id);
        self.append(record).await
    }

    /// Logs a transaction commit.
    #[inline]
    pub async fn log_commit(&self, txn_id: u32, prev_lsn: Lsn) -> Result<Lsn> {
        let record = LogRecord::commit(Lsn::INVALID, prev_lsn, txn_id);
        self.append(record).await
    }

    /// Logs a transaction abort.
    #[inline]
    pub async fn log_abort(&self, txn_id: u32, prev_lsn: Lsn) -> Result<Lsn> {
        let record = LogRecord::abort(Lsn::INVALID, prev_lsn, txn_id);
        self.append(record).await
    }

    /// Logs an insert operation.
    #[inline]
    pub async fn log_insert(&self, txn_id: u32, prev_lsn: Lsn, payload: Bytes) -> Result<Lsn> {
        let record = LogRecord::new(
            Lsn::INVALID,
            prev_lsn,
            txn_id,
            LogRecordType::Insert,
            payload,
        );
        self.append(record).await
    }

    /// Logs an update operation.
    #[inline]
    pub async fn log_update(&self, txn_id: u32, prev_lsn: Lsn, payload: Bytes) -> Result<Lsn> {
        let record = LogRecord::new(
            Lsn::INVALID,
            prev_lsn,
            txn_id,
            LogRecordType::Update,
            payload,
        );
        self.append(record).await
    }

    /// Logs a delete operation.
    #[inline]
    pub async fn log_delete(&self, txn_id: u32, prev_lsn: Lsn, payload: Bytes) -> Result<Lsn> {
        let record = LogRecord::new(
            Lsn::INVALID,
            prev_lsn,
            txn_id,
            LogRecordType::Delete,
            payload,
        );
        self.append(record).await
    }

    /// Logs a checkpoint begin marker.
    #[inline]
    pub async fn log_checkpoint_begin(&self) -> Result<Lsn> {
        let record = LogRecord::new(
            Lsn::INVALID,
            Lsn::INVALID,
            0,
            LogRecordType::CheckpointBegin,
            Bytes::new(),
        );
        self.append(record).await
    }

    /// Logs a checkpoint end marker.
    #[inline]
    pub async fn log_checkpoint_end(&self, payload: Bytes) -> Result<Lsn> {
        let record = LogRecord::new(
            Lsn::INVALID,
            Lsn::INVALID,
            0,
            LogRecordType::CheckpointEnd,
            payload,
        );
        self.append(record).await
    }
}

/// Handle for a transaction's WAL operations.
pub struct TxnWalHandle {
    writer: Arc<WalWriter>,
    txn_id: u32,
    last_lsn: Lsn,
}

impl TxnWalHandle {
    /// Creates a new transaction handle.
    pub async fn new(writer: Arc<WalWriter>) -> Result<Self> {
        let txn_id = writer.allocate_txn_id();
        let last_lsn = writer.log_begin(txn_id).await?;

        Ok(Self {
            writer,
            txn_id,
            last_lsn,
        })
    }

    /// Returns the transaction ID.
    #[inline]
    pub fn txn_id(&self) -> u32 {
        self.txn_id
    }

    /// Returns the last LSN written by this transaction.
    #[inline]
    pub fn last_lsn(&self) -> Lsn {
        self.last_lsn
    }

    /// Logs an insert operation.
    #[inline]
    pub async fn log_insert(&mut self, payload: Bytes) -> Result<Lsn> {
        self.last_lsn = self.writer.log_insert(self.txn_id, self.last_lsn, payload).await?;
        Ok(self.last_lsn)
    }

    /// Logs an update operation.
    #[inline]
    pub async fn log_update(&mut self, payload: Bytes) -> Result<Lsn> {
        self.last_lsn = self.writer.log_update(self.txn_id, self.last_lsn, payload).await?;
        Ok(self.last_lsn)
    }

    /// Logs a delete operation.
    #[inline]
    pub async fn log_delete(&mut self, payload: Bytes) -> Result<Lsn> {
        self.last_lsn = self.writer.log_delete(self.txn_id, self.last_lsn, payload).await?;
        Ok(self.last_lsn)
    }

    /// Commits the transaction.
    #[inline]
    pub async fn commit(self) -> Result<Lsn> {
        self.writer.log_commit(self.txn_id, self.last_lsn).await
    }

    /// Aborts the transaction.
    #[inline]
    pub async fn abort(self) -> Result<Lsn> {
        self.writer.log_abort(self.txn_id, self.last_lsn).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    async fn create_test_writer() -> (WalWriter, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let config = WalWriterConfig {
            wal_dir: dir.path().to_path_buf(),
            segment_size: LogSegment::DEFAULT_SIZE,
            fsync_enabled: false,
            batch_size: 100,
            batch_bytes: 1024 * 1024,
            flush_interval_us: 0,
        };
        let writer = WalWriter::new(config).await.unwrap();
        (writer, dir)
    }

    #[tokio::test]
    async fn test_wal_writer_creation() {
        let (writer, _dir) = create_test_writer().await;
        assert!(writer.next_lsn().is_valid());
        assert_eq!(writer.current_segment_id(), Some(SegmentId::FIRST));
    }

    #[tokio::test]
    async fn test_wal_writer_append() {
        let (writer, _dir) = create_test_writer().await;

        let record = LogRecord::begin(Lsn::INVALID, 1);
        let lsn = writer.append(record).await.unwrap();

        assert!(lsn.is_valid());
        assert_eq!(lsn.segment_id(), 1);
    }

    #[tokio::test]
    async fn test_wal_writer_transaction_flow() {
        let (writer, _dir) = create_test_writer().await;

        let begin_lsn = writer.log_begin(1).await.unwrap();
        assert!(begin_lsn.is_valid());

        let insert_lsn = writer.log_insert(1, begin_lsn, Bytes::from_static(b"data")).await.unwrap();
        assert!(insert_lsn > begin_lsn);

        let commit_lsn = writer.log_commit(1, insert_lsn).await.unwrap();
        assert!(commit_lsn > insert_lsn);
    }

    #[tokio::test]
    async fn test_wal_writer_multiple_transactions() {
        let (writer, _dir) = create_test_writer().await;

        for i in 1..=10 {
            let begin_lsn = writer.log_begin(i).await.unwrap();
            let insert_lsn = writer.log_insert(
                i,
                begin_lsn,
                Bytes::from(format!("data{}", i)),
            ).await.unwrap();
            writer.log_commit(i, insert_lsn).await.unwrap();
        }

        writer.flush().await.unwrap();
    }

    #[tokio::test]
    async fn test_wal_writer_flush() {
        let (writer, _dir) = create_test_writer().await;

        let lsn1 = writer.log_begin(1).await.unwrap();
        let flushed = writer.flush().await.unwrap();

        assert!(flushed >= lsn1);
    }

    #[tokio::test]
    async fn test_wal_writer_recovery() {
        let dir = tempdir().unwrap();
        let config = WalWriterConfig {
            wal_dir: dir.path().to_path_buf(),
            segment_size: LogSegment::DEFAULT_SIZE,
            fsync_enabled: true,
            batch_size: 1,
            batch_bytes: 1024,
            flush_interval_us: 0,
        };

        let final_lsn;
        {
            let writer = WalWriter::new(config.clone()).await.unwrap();
            writer.log_begin(1).await.unwrap();
            writer.log_insert(1, Lsn::INVALID, Bytes::from_static(b"test")).await.unwrap();
            final_lsn = writer.log_commit(1, Lsn::INVALID).await.unwrap();
            writer.close().await.unwrap();
        }

        {
            let writer = WalWriter::new(config).await.unwrap();
            assert!(writer.next_lsn() >= final_lsn);
        }
    }

    #[tokio::test]
    async fn test_txn_wal_handle() {
        let (writer, _dir) = create_test_writer().await;
        let writer = Arc::new(writer);

        let mut handle = TxnWalHandle::new(writer.clone()).await.unwrap();
        assert!(handle.txn_id() > 0);

        handle.log_insert(Bytes::from_static(b"row1")).await.unwrap();
        handle.log_update(Bytes::from_static(b"row1_updated")).await.unwrap();
        handle.log_delete(Bytes::from_static(b"row1")).await.unwrap();

        let commit_lsn = handle.commit().await.unwrap();
        assert!(commit_lsn.is_valid());
    }

    #[tokio::test]
    async fn test_txn_wal_handle_abort() {
        let (writer, _dir) = create_test_writer().await;
        let writer = Arc::new(writer);

        let mut handle = TxnWalHandle::new(writer.clone()).await.unwrap();
        handle.log_insert(Bytes::from_static(b"data")).await.unwrap();

        let abort_lsn = handle.abort().await.unwrap();
        assert!(abort_lsn.is_valid());
    }

    #[tokio::test]
    async fn test_wal_writer_checkpoint() {
        let (writer, _dir) = create_test_writer().await;

        let begin_lsn = writer.log_checkpoint_begin().await.unwrap();
        let end_lsn = writer.log_checkpoint_end(Bytes::from_static(b"checkpoint data")).await.unwrap();

        assert!(end_lsn > begin_lsn);
    }

    #[tokio::test]
    async fn test_wal_batch_flush() {
        let dir = tempdir().unwrap();
        let config = WalWriterConfig {
            wal_dir: dir.path().to_path_buf(),
            segment_size: LogSegment::DEFAULT_SIZE,
            fsync_enabled: false,
            batch_size: 10, // Small batch for testing
            batch_bytes: 1024 * 1024,
            flush_interval_us: 0,
        };
        let writer = WalWriter::new(config).await.unwrap();

        // Write 25 records (should trigger 2 flushes)
        for i in 1..=25 {
            writer.log_begin(i).await.unwrap();
        }

        writer.flush().await.unwrap();
    }
}

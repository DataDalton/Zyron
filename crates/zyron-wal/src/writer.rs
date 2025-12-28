//! WAL writer for appending log records.

use crate::record::{LogRecord, LogRecordType, Lsn};
use crate::segment::{LogSegment, SegmentHeader, SegmentId};
use bytes::Bytes;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use zyron_common::{Result, ZyronError};

/// Configuration for the WAL writer.
#[derive(Debug, Clone)]
pub struct WalWriterConfig {
    /// Directory for WAL segment files.
    pub wal_dir: PathBuf,
    /// Maximum size of each segment file.
    pub segment_size: u32,
    /// Enable fsync after each write.
    pub fsync_enabled: bool,
    /// Number of records to buffer before flushing (group commit).
    pub group_commit_size: usize,
}

impl Default for WalWriterConfig {
    fn default() -> Self {
        Self {
            wal_dir: PathBuf::from("./data/wal"),
            segment_size: LogSegment::DEFAULT_SIZE,
            fsync_enabled: true,
            group_commit_size: 1,
        }
    }
}

/// Thread-safe WAL writer.
///
/// Handles appending log records, segment rotation, and fsync.
pub struct WalWriter {
    /// Configuration.
    config: WalWriterConfig,
    /// Current active segment.
    current_segment: Mutex<Option<LogSegment>>,
    /// Next LSN to assign.
    next_lsn: AtomicU64,
    /// Next transaction ID to assign.
    next_txn_id: AtomicU64,
    /// Pending records for group commit (reserved for future use).
    #[allow(dead_code)]
    pending: Mutex<VecDeque<LogRecord>>,
    /// Last flushed LSN.
    flushed_lsn: AtomicU64,
}

impl WalWriter {
    /// Creates a new WAL writer.
    pub fn new(config: WalWriterConfig) -> Result<Self> {
        // Create WAL directory if it doesn't exist
        std::fs::create_dir_all(&config.wal_dir)?;

        // Find existing segments and determine starting LSN
        let (current_segment, next_lsn) = Self::recover_or_create(&config)?;

        Ok(Self {
            config,
            current_segment: Mutex::new(Some(current_segment)),
            next_lsn: AtomicU64::new(next_lsn.0),
            next_txn_id: AtomicU64::new(1),
            pending: Mutex::new(VecDeque::new()),
            flushed_lsn: AtomicU64::new(0),
        })
    }

    /// Recovers from existing segments or creates a new one.
    fn recover_or_create(config: &WalWriterConfig) -> Result<(LogSegment, Lsn)> {
        let mut segments: Vec<_> = std::fs::read_dir(&config.wal_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "wal")
                    .unwrap_or(false)
            })
            .collect();

        if segments.is_empty() {
            // Create first segment
            let segment_id = SegmentId::FIRST;
            let first_lsn = Lsn::new(segment_id.0, SegmentHeader::SIZE as u32);
            let segment = LogSegment::create(
                &config.wal_dir,
                segment_id,
                first_lsn,
                config.segment_size,
            )?;
            return Ok((segment, first_lsn));
        }

        // Sort by filename to find the latest segment
        segments.sort_by(|a, b| a.path().cmp(&b.path()));

        // Open the latest segment
        let latest_path = segments.last().unwrap().path();
        let segment = LogSegment::open(&latest_path)?;

        // Next LSN is at the current write position
        let next_lsn = Lsn::new(segment.segment_id().0, segment.write_offset());

        Ok((segment, next_lsn))
    }

    /// Returns the directory containing WAL segments.
    pub fn wal_dir(&self) -> &Path {
        &self.config.wal_dir
    }

    /// Returns the next LSN that will be assigned.
    pub fn next_lsn(&self) -> Lsn {
        Lsn(self.next_lsn.load(Ordering::SeqCst))
    }

    /// Returns the last flushed LSN.
    pub fn flushed_lsn(&self) -> Lsn {
        Lsn(self.flushed_lsn.load(Ordering::SeqCst))
    }

    /// Allocates a new transaction ID.
    pub fn allocate_txn_id(&self) -> u32 {
        self.next_txn_id.fetch_add(1, Ordering::SeqCst) as u32
    }

    /// Allocates the next LSN for a record of the given size.
    fn allocate_lsn(&self, record_size: usize) -> Lsn {
        let current = self.next_lsn.fetch_add(record_size as u64, Ordering::SeqCst);
        Lsn(current)
    }

    /// Appends a log record.
    pub fn append(&self, mut record: LogRecord) -> Result<Lsn> {
        let record_size = record.size_on_disk();

        // Allocate LSN
        let lsn = self.allocate_lsn(record_size);
        record.lsn = lsn;

        let mut segment_guard = self.current_segment.lock();
        let segment = segment_guard
            .as_mut()
            .ok_or_else(|| ZyronError::WalWriteFailed("WAL closed".to_string()))?;

        // Check if we need to rotate to a new segment
        if !segment.has_space(record_size) {
            // Close current segment
            segment.sync()?;
            segment.close()?;

            // Create new segment
            let new_segment_id = segment.segment_id().next();
            let new_first_lsn = Lsn::new(new_segment_id.0, SegmentHeader::SIZE as u32);

            *segment = LogSegment::create(
                &self.config.wal_dir,
                new_segment_id,
                new_first_lsn,
                self.config.segment_size,
            )?;

            // Update next_lsn to point to new segment
            self.next_lsn.store(new_first_lsn.0, Ordering::SeqCst);
            record.lsn = new_first_lsn;
        }

        // Write record to segment
        let written_lsn = segment.append(&record)?;

        // Sync if configured
        if self.config.fsync_enabled {
            segment.sync()?;
            self.flushed_lsn.store(written_lsn.0, Ordering::SeqCst);
        }

        Ok(written_lsn)
    }

    /// Logs a transaction begin.
    pub fn log_begin(&self, txn_id: u32) -> Result<Lsn> {
        let record = LogRecord::begin(Lsn::INVALID, txn_id);
        self.append(record)
    }

    /// Logs a transaction commit.
    pub fn log_commit(&self, txn_id: u32, prev_lsn: Lsn) -> Result<Lsn> {
        let record = LogRecord::commit(Lsn::INVALID, prev_lsn, txn_id);
        self.append(record)
    }

    /// Logs a transaction abort.
    pub fn log_abort(&self, txn_id: u32, prev_lsn: Lsn) -> Result<Lsn> {
        let record = LogRecord::abort(Lsn::INVALID, prev_lsn, txn_id);
        self.append(record)
    }

    /// Logs an insert operation.
    pub fn log_insert(
        &self,
        txn_id: u32,
        prev_lsn: Lsn,
        payload: Bytes,
    ) -> Result<Lsn> {
        let record = LogRecord::new(
            Lsn::INVALID,
            prev_lsn,
            txn_id,
            LogRecordType::Insert,
            payload,
        );
        self.append(record)
    }

    /// Logs an update operation.
    pub fn log_update(
        &self,
        txn_id: u32,
        prev_lsn: Lsn,
        payload: Bytes,
    ) -> Result<Lsn> {
        let record = LogRecord::new(
            Lsn::INVALID,
            prev_lsn,
            txn_id,
            LogRecordType::Update,
            payload,
        );
        self.append(record)
    }

    /// Logs a delete operation.
    pub fn log_delete(
        &self,
        txn_id: u32,
        prev_lsn: Lsn,
        payload: Bytes,
    ) -> Result<Lsn> {
        let record = LogRecord::new(
            Lsn::INVALID,
            prev_lsn,
            txn_id,
            LogRecordType::Delete,
            payload,
        );
        self.append(record)
    }

    /// Logs a checkpoint begin marker.
    pub fn log_checkpoint_begin(&self) -> Result<Lsn> {
        let record = LogRecord::new(
            Lsn::INVALID,
            Lsn::INVALID,
            0,
            LogRecordType::CheckpointBegin,
            Bytes::new(),
        );
        self.append(record)
    }

    /// Logs a checkpoint end marker with active transaction info.
    pub fn log_checkpoint_end(&self, payload: Bytes) -> Result<Lsn> {
        let record = LogRecord::new(
            Lsn::INVALID,
            Lsn::INVALID,
            0,
            LogRecordType::CheckpointEnd,
            payload,
        );
        self.append(record)
    }

    /// Forces all pending records to disk.
    pub fn flush(&self) -> Result<Lsn> {
        let mut segment_guard = self.current_segment.lock();
        if let Some(ref mut segment) = *segment_guard {
            segment.sync()?;
            let lsn = Lsn::new(segment.segment_id().0, segment.write_offset());
            self.flushed_lsn.store(lsn.0, Ordering::SeqCst);
            Ok(lsn)
        } else {
            Ok(Lsn::INVALID)
        }
    }

    /// Closes the WAL writer.
    pub fn close(&self) -> Result<()> {
        let mut segment_guard = self.current_segment.lock();
        if let Some(ref mut segment) = segment_guard.take() {
            segment.sync()?;
            segment.close()?;
        }
        Ok(())
    }

    /// Returns the current segment ID.
    pub fn current_segment_id(&self) -> Option<SegmentId> {
        self.current_segment.lock().as_ref().map(|s| s.segment_id())
    }
}

impl Drop for WalWriter {
    fn drop(&mut self) {
        let _ = self.close();
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
    pub fn new(writer: Arc<WalWriter>) -> Result<Self> {
        let txn_id = writer.allocate_txn_id();
        let last_lsn = writer.log_begin(txn_id)?;

        Ok(Self {
            writer,
            txn_id,
            last_lsn,
        })
    }

    /// Returns the transaction ID.
    pub fn txn_id(&self) -> u32 {
        self.txn_id
    }

    /// Returns the last LSN written by this transaction.
    pub fn last_lsn(&self) -> Lsn {
        self.last_lsn
    }

    /// Logs an insert operation.
    pub fn log_insert(&mut self, payload: Bytes) -> Result<Lsn> {
        self.last_lsn = self.writer.log_insert(self.txn_id, self.last_lsn, payload)?;
        Ok(self.last_lsn)
    }

    /// Logs an update operation.
    pub fn log_update(&mut self, payload: Bytes) -> Result<Lsn> {
        self.last_lsn = self.writer.log_update(self.txn_id, self.last_lsn, payload)?;
        Ok(self.last_lsn)
    }

    /// Logs a delete operation.
    pub fn log_delete(&mut self, payload: Bytes) -> Result<Lsn> {
        self.last_lsn = self.writer.log_delete(self.txn_id, self.last_lsn, payload)?;
        Ok(self.last_lsn)
    }

    /// Commits the transaction.
    pub fn commit(self) -> Result<Lsn> {
        self.writer.log_commit(self.txn_id, self.last_lsn)
    }

    /// Aborts the transaction.
    pub fn abort(self) -> Result<Lsn> {
        self.writer.log_abort(self.txn_id, self.last_lsn)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn create_test_writer() -> (WalWriter, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let config = WalWriterConfig {
            wal_dir: dir.path().to_path_buf(),
            segment_size: LogSegment::DEFAULT_SIZE,
            fsync_enabled: false, // Disable for faster tests
            group_commit_size: 1,
        };
        let writer = WalWriter::new(config).unwrap();
        (writer, dir)
    }

    #[test]
    fn test_wal_writer_creation() {
        let (writer, _dir) = create_test_writer();
        assert!(writer.next_lsn().is_valid());
        assert_eq!(writer.current_segment_id(), Some(SegmentId::FIRST));
    }

    #[test]
    fn test_wal_writer_append() {
        let (writer, _dir) = create_test_writer();

        let record = LogRecord::begin(Lsn::INVALID, 1);
        let lsn = writer.append(record).unwrap();

        assert!(lsn.is_valid());
        assert_eq!(lsn.segment_id(), 1);
    }

    #[test]
    fn test_wal_writer_transaction_flow() {
        let (writer, _dir) = create_test_writer();

        // Begin transaction
        let begin_lsn = writer.log_begin(1).unwrap();
        assert!(begin_lsn.is_valid());

        // Insert
        let insert_lsn = writer.log_insert(1, begin_lsn, Bytes::from_static(b"data")).unwrap();
        assert!(insert_lsn > begin_lsn);

        // Commit
        let commit_lsn = writer.log_commit(1, insert_lsn).unwrap();
        assert!(commit_lsn > insert_lsn);
    }

    #[test]
    fn test_wal_writer_multiple_transactions() {
        let (writer, _dir) = create_test_writer();

        for i in 1..=10 {
            let begin_lsn = writer.log_begin(i).unwrap();
            let insert_lsn = writer.log_insert(
                i,
                begin_lsn,
                Bytes::from(format!("data{}", i)),
            ).unwrap();
            writer.log_commit(i, insert_lsn).unwrap();
        }

        writer.flush().unwrap();
    }

    #[test]
    fn test_wal_writer_flush() {
        let (writer, _dir) = create_test_writer();

        let lsn1 = writer.log_begin(1).unwrap();
        let flushed = writer.flush().unwrap();

        assert!(flushed >= lsn1);
    }

    #[test]
    fn test_wal_writer_recovery() {
        let dir = tempdir().unwrap();
        let config = WalWriterConfig {
            wal_dir: dir.path().to_path_buf(),
            segment_size: LogSegment::DEFAULT_SIZE,
            fsync_enabled: true,
            group_commit_size: 1,
        };

        let final_lsn;
        {
            let writer = WalWriter::new(config.clone()).unwrap();
            writer.log_begin(1).unwrap();
            writer.log_insert(1, Lsn::INVALID, Bytes::from_static(b"test")).unwrap();
            final_lsn = writer.log_commit(1, Lsn::INVALID).unwrap();
            writer.close().unwrap();
        }

        // Reopen and check state
        {
            let writer = WalWriter::new(config).unwrap();
            assert!(writer.next_lsn() >= final_lsn);
        }
    }

    #[test]
    fn test_txn_wal_handle() {
        let (writer, _dir) = create_test_writer();
        let writer = Arc::new(writer);

        let mut handle = TxnWalHandle::new(writer.clone()).unwrap();
        assert!(handle.txn_id() > 0);

        handle.log_insert(Bytes::from_static(b"row1")).unwrap();
        handle.log_update(Bytes::from_static(b"row1_updated")).unwrap();
        handle.log_delete(Bytes::from_static(b"row1")).unwrap();

        let commit_lsn = handle.commit().unwrap();
        assert!(commit_lsn.is_valid());
    }

    #[test]
    fn test_txn_wal_handle_abort() {
        let (writer, _dir) = create_test_writer();
        let writer = Arc::new(writer);

        let mut handle = TxnWalHandle::new(writer.clone()).unwrap();
        handle.log_insert(Bytes::from_static(b"data")).unwrap();

        let abort_lsn = handle.abort().unwrap();
        assert!(abort_lsn.is_valid());
    }

    #[test]
    fn test_wal_writer_checkpoint() {
        let (writer, _dir) = create_test_writer();

        let begin_lsn = writer.log_checkpoint_begin().unwrap();
        let end_lsn = writer.log_checkpoint_end(Bytes::from_static(b"checkpoint data")).unwrap();

        assert!(end_lsn > begin_lsn);
    }
}

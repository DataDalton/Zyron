//! Lock-free WAL writer with group commit for high-throughput durability.
//!
//! Uses atomic operations for LSN assignment and a ring buffer for buffering
//! records. A dedicated flush thread writes records to disk in order,
//! amortizing fsync across batches (group commit).
//!
//! The hot path (append) is lock-free, making this writer scale well under
//! concurrent load from multiple transactions.

use crate::record::{LogRecord, LogRecordType, Lsn};
use crate::ring_buffer::RingBuffer;
use crate::segment::{LogSegment, SegmentHeader, SegmentId};
use crate::sequencer::LsnSequencer;
use bytes::Bytes;
use parking_lot::{Condvar, Mutex};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::JoinHandle;
use tokio::sync::Notify;
use zyron_common::{Result, ZyronError};

/// Shared state for coordinating segment rotation between `append()` and the flush thread.
struct RotationState {
    /// Rotation has been requested.
    pending: bool,
    /// The segment ID that is full and needs to rotate away.
    old_segment_id: u32,
}

/// Configuration for the WAL writer.
#[derive(Debug, Clone)]
pub struct WalWriterConfig {
    /// Directory for WAL segment files.
    pub wal_dir: PathBuf,
    /// Maximum size of each segment file.
    pub segment_size: u32,
    /// Enable fsync after writes.
    pub fsync_enabled: bool,
    /// Ring buffer capacity in bytes.
    pub ring_buffer_capacity: usize,
}

impl Default for WalWriterConfig {
    fn default() -> Self {
        Self {
            wal_dir: PathBuf::from("./data/wal"),
            segment_size: LogSegment::DEFAULT_SIZE,
            fsync_enabled: true,
            ring_buffer_capacity: 16 * 1024 * 1024, // 16MB
        }
    }
}

/// Lock-free WAL writer for high-throughput concurrent workloads.
///
/// Uses atomic operations for LSN assignment and a ring buffer for
/// buffering records. A dedicated flush thread writes records to disk
/// in LSN order.
///
/// The hot path (`append`) is lock-free:
/// 1. Pre-serialize record (outside any lock)
/// 2. Reserve LSN atomically (CAS loop)
/// 3. Finalize serialization (patch LSN + CRC32)
/// 4. Insert into ring buffer
/// 5. Notify flush thread
pub struct WalWriter {
    /// Lock-free LSN sequencer, shared with the flush thread for rotation.
    sequencer: Arc<LsnSequencer>,
    /// Ring buffer for pending records.
    ring_buffer: Arc<RingBuffer>,
    /// Notification to wake flush thread.
    flush_notify: Arc<Notify>,
    /// Notification when flush completes.
    flush_done: Arc<Notify>,
    /// Shutdown flag.
    shutdown: Arc<AtomicBool>,
    /// Flush thread handle.
    flush_thread: Option<JoinHandle<()>>,
    /// Current segment (only accessed by flush thread).
    segment: Arc<Mutex<Option<LogSegment>>>,
    /// Last flushed LSN.
    flushed_lsn: Arc<AtomicU64>,
    /// Next transaction ID.
    next_txn_id: AtomicU64,
    /// Configuration.
    config: WalWriterConfig,
    /// Segment rotation coordination between append() and the flush thread.
    rotation: Arc<(Mutex<RotationState>, Condvar)>,
}

impl WalWriter {
    /// Creates a new WAL writer.
    pub async fn new(config: WalWriterConfig) -> Result<Self> {
        tokio::fs::create_dir_all(&config.wal_dir).await?;

        let (segment, initial_lsn) = Self::recover_or_create(&config).await?;
        let segment_id = segment.segment_id().0;
        let write_offset = segment.write_offset();

        let sequencer = Arc::new(LsnSequencer::new(
            segment_id,
            write_offset,
            config.segment_size,
        ));
        let ring_buffer = Arc::new(RingBuffer::new(config.ring_buffer_capacity));
        let flush_notify = Arc::new(Notify::new());
        let flush_done = Arc::new(Notify::new());
        let shutdown = Arc::new(AtomicBool::new(false));
        let segment = Arc::new(Mutex::new(Some(segment)));
        let flushed_lsn = Arc::new(AtomicU64::new(initial_lsn.0.saturating_sub(1)));
        let rotation = Arc::new((
            Mutex::new(RotationState {
                pending: false,
                old_segment_id: 0,
            }),
            Condvar::new(),
        ));

        let flush_thread = Self::spawn_flush_thread(
            ring_buffer.clone(),
            segment.clone(),
            flush_notify.clone(),
            flush_done.clone(),
            shutdown.clone(),
            flushed_lsn.clone(),
            config.fsync_enabled,
            sequencer.clone(),
            rotation.clone(),
            config.wal_dir.clone(),
            config.segment_size,
        );

        Ok(Self {
            sequencer,
            ring_buffer,
            flush_notify,
            flush_done,
            shutdown,
            flush_thread: Some(flush_thread),
            segment,
            flushed_lsn,
            next_txn_id: AtomicU64::new(1),
            config,
            rotation,
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
            let segment =
                LogSegment::create(&config.wal_dir, segment_id, first_lsn, config.segment_size)
                    .await?;
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

    /// Spawns the background flush thread.
    fn spawn_flush_thread(
        ring_buffer: Arc<RingBuffer>,
        segment: Arc<Mutex<Option<LogSegment>>>,
        notify: Arc<Notify>,
        flush_done: Arc<Notify>,
        shutdown: Arc<AtomicBool>,
        flushed_lsn: Arc<AtomicU64>,
        fsync_enabled: bool,
        sequencer: Arc<LsnSequencer>,
        rotation: Arc<(Mutex<RotationState>, Condvar)>,
        wal_dir: PathBuf,
        segment_size: u32,
    ) -> JoinHandle<()> {
        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();

            rt.block_on(async {
                let mut batch_buffer = Vec::with_capacity(64 * 1024);

                loop {
                    // Wait for notification
                    notify.notified().await;

                    // Check shutdown
                    if shutdown.load(Ordering::Acquire) {
                        // Final drain before exit
                        Self::flush_records(
                            &ring_buffer,
                            &segment,
                            &mut batch_buffer,
                            &flushed_lsn,
                            &flush_done,
                            fsync_enabled,
                        )
                        .await;
                        break;
                    }

                    // Flush any pending records
                    Self::flush_records(
                        &ring_buffer,
                        &segment,
                        &mut batch_buffer,
                        &flushed_lsn,
                        &flush_done,
                        fsync_enabled,
                    )
                    .await;

                    // Handle segment rotation if requested by append()
                    Self::handle_rotation(
                        &rotation,
                        &ring_buffer,
                        &segment,
                        &sequencer,
                        &wal_dir,
                        segment_size,
                        fsync_enabled,
                    )
                    .await;
                }
            });
        })
    }

    /// Creates a new segment and advances the sequencer to complete rotation.
    ///
    /// Called by the flush thread after draining the ring buffer. Before creating
    /// the new segment, spins until all in-flight writes to the old segment commit,
    /// then does a final drain to capture any bytes committed after the main flush.
    /// This prevents cross-segment contamination caused by delayed commit_write calls.
    async fn handle_rotation(
        rotation: &Arc<(Mutex<RotationState>, Condvar)>,
        ring_buffer: &RingBuffer,
        segment: &Mutex<Option<LogSegment>>,
        sequencer: &Arc<LsnSequencer>,
        wal_dir: &Path,
        segment_size: u32,
        fsync_enabled: bool,
    ) {
        let (lock, condvar) = rotation.as_ref();
        let old_segment_id = {
            let state = lock.lock();
            if !state.pending {
                return;
            }
            state.old_segment_id
        };

        // Wait for all in-flight writes to the old segment to commit their bytes.
        // This covers threads that called write_record() but haven't called commit_write() yet.
        ring_buffer.wait_until_committed();

        // Drain any bytes committed after flush_records() ran. These belong to the old
        // segment and must be written before the new segment is installed.
        let mut residual = Vec::new();
        ring_buffer.drain_into(&mut residual);
        if !residual.is_empty() {
            let mut seg_guard = segment.lock();
            if let Some(ref mut seg) = *seg_guard {
                let _ = seg.append_batch(&residual).await;
            }
        }

        let new_segment_id = old_segment_id + 1;
        let first_lsn = Lsn::new(new_segment_id, SegmentHeader::SIZE as u32);

        // Sync old segment before switching
        {
            let mut seg_guard = segment.lock();
            if let Some(ref mut seg) = *seg_guard {
                if fsync_enabled {
                    let _ = seg.sync().await;
                }
            }
        }

        // Create the new segment file
        match LogSegment::create(wal_dir, SegmentId(new_segment_id), first_lsn, segment_size).await
        {
            Ok(new_seg) => {
                // Install new segment
                {
                    let mut seg_guard = segment.lock();
                    *seg_guard = Some(new_seg);
                }
                // Advance sequencer so append() callers can reserve space
                sequencer.advance_segment(new_segment_id);
                // Signal all threads waiting on rotation
                let mut state = lock.lock();
                state.pending = false;
                condvar.notify_all();
            }
            Err(e) => {
                eprintln!("WAL segment rotation error: {:?}", e);
                // Clear pending state and wake all waiters to prevent deadlock.
                // Callers will retry rotation on the next append.
                let mut state = lock.lock();
                state.pending = false;
                condvar.notify_all();
            }
        }
    }

    /// Flushes records from ring buffer to disk.
    async fn flush_records(
        ring_buffer: &RingBuffer,
        segment: &Mutex<Option<LogSegment>>,
        batch_buffer: &mut Vec<u8>,
        flushed_lsn: &AtomicU64,
        flush_done: &Notify,
        fsync_enabled: bool,
    ) {
        batch_buffer.clear();
        let max_lsn = ring_buffer.drain_into(batch_buffer);

        if batch_buffer.is_empty() {
            flush_done.notify_waiters();
            return;
        }

        // Write to segment
        {
            let mut seg_guard = segment.lock();
            if let Some(ref mut seg) = *seg_guard {
                if let Err(e) = seg.append_batch(batch_buffer).await {
                    eprintln!("WAL flush error: {:?}", e);
                    flush_done.notify_waiters();
                    return;
                }

                if fsync_enabled {
                    if let Err(e) = seg.sync().await {
                        eprintln!("WAL sync error: {:?}", e);
                        flush_done.notify_waiters();
                        return;
                    }
                }
            }
        }

        flushed_lsn.store(max_lsn.0, Ordering::Release);
        flush_done.notify_waiters();
    }

    /// Appends a log record. Zero-allocation hot path.
    ///
    /// The record is serialized directly into a pre-allocated ring buffer slot,
    /// eliminating per-record allocation overhead. When the current segment is
    /// full, this method blocks until the flush thread completes segment rotation.
    #[inline]
    pub fn append(&self, record: &LogRecord) -> Result<Lsn> {
        let record_size = record.size_on_disk() as u32;

        loop {
            // Reserve LSN atomically (lock-free)
            let (lsn, needs_rotation) = self.sequencer.reserve(record_size);

            if needs_rotation {
                // Request rotation and block until the flush thread completes it
                let (lock, condvar) = self.rotation.as_ref();
                let mut state = lock.lock();

                // Check again under the lock: another thread may have already rotated
                let (current_lsn, still_full) = self.sequencer.reserve(record_size);
                if !still_full {
                    // Rotation already happened, write the record
                    drop(state);
                    let lsn = current_lsn;
                    unsafe {
                        let buf = self.ring_buffer.write_record(record_size as usize, lsn);
                        record.serialize_into(buf, lsn);
                    }
                    self.ring_buffer.commit_write(record_size as usize, lsn);
                    self.flush_notify.notify_one();
                    return Ok(lsn);
                }

                if !state.pending {
                    state.old_segment_id = lsn.segment_id();
                    state.pending = true;
                    self.flush_notify.notify_one();
                }

                // Wait for the flush thread to complete rotation
                condvar.wait(&mut state);
                drop(state);
                continue;
            }

            // Normal path: claim space and write directly into ring buffer
            // SAFETY: record fits in buffer
            unsafe {
                let buf = self.ring_buffer.write_record(record_size as usize, lsn);
                record.serialize_into(buf, lsn);
            }

            self.ring_buffer.commit_write(record_size as usize, lsn);
            self.flush_notify.notify_one();
            return Ok(lsn);
        }
    }

    /// Appends a log record and waits for it to be flushed to disk.
    pub async fn append_durable(&self, record: &LogRecord) -> Result<Lsn> {
        let lsn = self.append(record)?;
        self.wait_for_flush(lsn).await?;
        Ok(lsn)
    }

    /// Waits until the given LSN has been flushed to disk.
    pub async fn wait_for_flush(&self, target_lsn: Lsn) -> Result<()> {
        loop {
            // Check if already flushed
            if self.flushed_lsn() >= target_lsn {
                return Ok(());
            }

            // Trigger flush thread and wait for completion
            self.flush_notify.notify_one();
            self.flush_done.notified().await;
        }
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

    /// Returns the next LSN that will be assigned.
    #[inline]
    pub fn next_lsn(&self) -> Lsn {
        self.sequencer.current()
    }

    /// Returns the current segment ID according to the sequencer.
    pub fn current_segment_id(&self) -> Result<SegmentId> {
        Ok(SegmentId(self.sequencer.current_segment_id()))
    }

    /// Returns the WAL directory.
    #[inline]
    pub fn wal_dir(&self) -> &Path {
        &self.config.wal_dir
    }

    /// Forces a flush and waits for completion.
    pub async fn flush(&self) -> Result<Lsn> {
        loop {
            // Check if nothing pending
            if self.ring_buffer.is_empty() {
                return Ok(self.flushed_lsn());
            }

            // Trigger flush thread and wait for completion
            self.flush_notify.notify_one();
            self.flush_done.notified().await;
        }
    }

    /// Closes the WAL writer.
    pub async fn close(&self) -> Result<()> {
        // First flush any pending records
        self.flush().await?;

        // Signal shutdown
        self.shutdown.store(true, Ordering::Release);
        self.flush_notify.notify_one();

        // Close segment
        let mut seg_guard = self.segment.lock();
        if let Some(ref mut seg) = seg_guard.take() {
            seg.sync().await?;
            seg.close().await?;
        }

        Ok(())
    }

    // Convenience methods for common record types

    /// Logs a transaction begin.
    #[inline]
    pub fn log_begin(&self, txn_id: u32) -> Result<Lsn> {
        let record = LogRecord::begin(Lsn::INVALID, txn_id);
        self.append(&record)
    }

    /// Logs a transaction commit.
    #[inline]
    pub fn log_commit(&self, txn_id: u32, prev_lsn: Lsn) -> Result<Lsn> {
        let record = LogRecord::commit(Lsn::INVALID, prev_lsn, txn_id);
        self.append(&record)
    }

    /// Logs a transaction abort.
    #[inline]
    pub fn log_abort(&self, txn_id: u32, prev_lsn: Lsn) -> Result<Lsn> {
        let record = LogRecord::abort(Lsn::INVALID, prev_lsn, txn_id);
        self.append(&record)
    }

    /// Logs an insert operation.
    #[inline]
    pub fn log_insert(&self, txn_id: u32, prev_lsn: Lsn, payload: Bytes) -> Result<Lsn> {
        let record = LogRecord::new(
            Lsn::INVALID,
            prev_lsn,
            txn_id,
            LogRecordType::Insert,
            payload,
        );
        self.append(&record)
    }

    /// Logs an update operation.
    #[inline]
    pub fn log_update(&self, txn_id: u32, prev_lsn: Lsn, payload: Bytes) -> Result<Lsn> {
        let record = LogRecord::new(
            Lsn::INVALID,
            prev_lsn,
            txn_id,
            LogRecordType::Update,
            payload,
        );
        self.append(&record)
    }

    /// Logs a delete operation.
    #[inline]
    pub fn log_delete(&self, txn_id: u32, prev_lsn: Lsn, payload: Bytes) -> Result<Lsn> {
        let record = LogRecord::new(
            Lsn::INVALID,
            prev_lsn,
            txn_id,
            LogRecordType::Delete,
            payload,
        );
        self.append(&record)
    }

    /// Logs a checkpoint begin marker.
    #[inline]
    pub fn log_checkpoint_begin(&self) -> Result<Lsn> {
        let record = LogRecord::new(
            Lsn::INVALID,
            Lsn::INVALID,
            0,
            LogRecordType::CheckpointBegin,
            Bytes::new(),
        );
        self.append(&record)
    }

    /// Logs a checkpoint end marker.
    #[inline]
    pub fn log_checkpoint_end(&self, payload: Bytes) -> Result<Lsn> {
        let record = LogRecord::new(
            Lsn::INVALID,
            Lsn::INVALID,
            0,
            LogRecordType::CheckpointEnd,
            payload,
        );
        self.append(&record)
    }
}

impl Drop for WalWriter {
    fn drop(&mut self) {
        // Signal shutdown
        self.shutdown.store(true, Ordering::Release);
        self.flush_notify.notify_one();

        // Wait for flush thread
        if let Some(handle) = self.flush_thread.take() {
            let _ = handle.join();
        }
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
    pub fn log_insert(&mut self, payload: Bytes) -> Result<Lsn> {
        self.last_lsn = self
            .writer
            .log_insert(self.txn_id, self.last_lsn, payload)?;
        Ok(self.last_lsn)
    }

    /// Logs an update operation.
    #[inline]
    pub fn log_update(&mut self, payload: Bytes) -> Result<Lsn> {
        self.last_lsn = self
            .writer
            .log_update(self.txn_id, self.last_lsn, payload)?;
        Ok(self.last_lsn)
    }

    /// Logs a delete operation.
    #[inline]
    pub fn log_delete(&mut self, payload: Bytes) -> Result<Lsn> {
        self.last_lsn = self
            .writer
            .log_delete(self.txn_id, self.last_lsn, payload)?;
        Ok(self.last_lsn)
    }

    /// Commits the transaction.
    #[inline]
    pub fn commit(self) -> Result<Lsn> {
        self.writer.log_commit(self.txn_id, self.last_lsn)
    }

    /// Aborts the transaction.
    #[inline]
    pub fn abort(self) -> Result<Lsn> {
        self.writer.log_abort(self.txn_id, self.last_lsn)
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
            ring_buffer_capacity: 1024 * 1024, // 1MB
        };
        let writer = WalWriter::new(config).await.unwrap();
        (writer, dir)
    }

    #[tokio::test]
    async fn test_wal_writer_creation() {
        let (writer, _dir) = create_test_writer().await;
        assert!(writer.next_lsn().is_valid());
        assert_eq!(writer.current_segment_id().unwrap(), SegmentId::FIRST);
    }

    #[tokio::test]
    async fn test_wal_writer_append() {
        let (writer, _dir) = create_test_writer().await;

        let record = LogRecord::begin(Lsn::INVALID, 1);
        let lsn = writer.append(&record).unwrap();

        assert!(lsn.is_valid());
        assert_eq!(lsn.segment_id(), 1);

        writer.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_wal_writer_transaction_flow() {
        let (writer, _dir) = create_test_writer().await;

        let begin_lsn = writer.log_begin(1).unwrap();
        assert!(begin_lsn.is_valid());

        let insert_lsn = writer
            .log_insert(1, begin_lsn, Bytes::from_static(b"data"))
            .unwrap();
        assert!(insert_lsn > begin_lsn);

        let commit_lsn = writer.log_commit(1, insert_lsn).unwrap();
        assert!(commit_lsn > insert_lsn);

        writer.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_wal_writer_multiple_transactions() {
        let (writer, _dir) = create_test_writer().await;

        for i in 1..=10 {
            let begin_lsn = writer.log_begin(i).unwrap();
            let insert_lsn = writer
                .log_insert(i, begin_lsn, Bytes::from(format!("data{}", i)))
                .unwrap();
            writer.log_commit(i, insert_lsn).unwrap();
        }

        writer.flush().await.unwrap();
        writer.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_wal_writer_flush() {
        let (writer, _dir) = create_test_writer().await;

        let lsn1 = writer.log_begin(1).unwrap();
        let flushed = writer.flush().await.unwrap();

        assert!(flushed >= lsn1);
        writer.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_wal_writer_recovery() {
        let dir = tempdir().unwrap();
        let config = WalWriterConfig {
            wal_dir: dir.path().to_path_buf(),
            segment_size: LogSegment::DEFAULT_SIZE,
            fsync_enabled: true,
            ring_buffer_capacity: 1024 * 1024, // 1MB
        };

        let final_lsn;
        {
            let writer = WalWriter::new(config.clone()).await.unwrap();
            writer.log_begin(1).unwrap();
            writer
                .log_insert(1, Lsn::INVALID, Bytes::from_static(b"test"))
                .unwrap();
            final_lsn = writer.log_commit(1, Lsn::INVALID).unwrap();
            writer.close().await.unwrap();
        }

        {
            let writer = WalWriter::new(config).await.unwrap();
            assert!(writer.next_lsn() >= final_lsn);
            writer.close().await.unwrap();
        }
    }

    #[tokio::test]
    async fn test_txn_wal_handle() {
        let (writer, _dir) = create_test_writer().await;
        let writer = Arc::new(writer);

        let mut handle = TxnWalHandle::new(writer.clone()).unwrap();
        assert!(handle.txn_id() > 0);

        handle.log_insert(Bytes::from_static(b"row1")).unwrap();
        handle
            .log_update(Bytes::from_static(b"row1_updated"))
            .unwrap();
        handle.log_delete(Bytes::from_static(b"row1")).unwrap();

        let commit_lsn = handle.commit().unwrap();
        assert!(commit_lsn.is_valid());

        writer.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_txn_wal_handle_abort() {
        let (writer, _dir) = create_test_writer().await;
        let writer = Arc::new(writer);

        let mut handle = TxnWalHandle::new(writer.clone()).unwrap();
        handle.log_insert(Bytes::from_static(b"data")).unwrap();

        let abort_lsn = handle.abort().unwrap();
        assert!(abort_lsn.is_valid());

        writer.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_wal_writer_checkpoint() {
        let (writer, _dir) = create_test_writer().await;

        let begin_lsn = writer.log_checkpoint_begin().unwrap();
        let end_lsn = writer
            .log_checkpoint_end(Bytes::from_static(b"checkpoint data"))
            .unwrap();

        assert!(end_lsn > begin_lsn);
        writer.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_wal_batch_flush() {
        let dir = tempdir().unwrap();
        let config = WalWriterConfig {
            wal_dir: dir.path().to_path_buf(),
            segment_size: LogSegment::DEFAULT_SIZE,
            fsync_enabled: false,
            ring_buffer_capacity: 1024 * 1024, // 1MB
        };
        let writer = WalWriter::new(config).await.unwrap();

        // Write 25 records
        for i in 1..=25 {
            writer.log_begin(i).unwrap();
        }

        writer.flush().await.unwrap();
        writer.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_segment_rotation() {
        use crate::reader::WalReader;

        let dir = tempdir().unwrap();
        // 64KB segment, 200-byte payload records = 228 bytes each
        // 287 records per segment
        let config = WalWriterConfig {
            wal_dir: dir.path().to_path_buf(),
            segment_size: 64 * 1024,
            fsync_enabled: false,
            ring_buffer_capacity: 1024 * 1024, // 1MB
        };

        let writer = WalWriter::new(config).await.unwrap();
        let initial_seg = writer.current_segment_id().unwrap();

        for i in 0..1000 {
            let payload = Bytes::from(vec![0u8; 200]);
            writer.log_insert(1, Lsn::INVALID, payload).unwrap();
            if i % 100 == 99 {
                let seg = writer.current_segment_id().unwrap();
                println!("After record {}: segment {}", i + 1, seg.0);
            }
        }

        let final_seg = writer.current_segment_id().unwrap();
        writer.close().await.unwrap();

        println!("Rotated from seg {} to seg {}", initial_seg.0, final_seg.0);
        assert!(final_seg.0 > initial_seg.0, "Expected rotation");

        // List files on disk
        let mut files: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| {
                let e = e.unwrap();
                (e.file_name(), e.metadata().unwrap().len())
            })
            .collect();
        files.sort();
        for (name, size) in &files {
            println!("  {:?} size={}", name, size);
        }

        let reader = WalReader::new(dir.path()).unwrap();
        println!("Segment count: {}", reader.segment_count());
        let records = reader.scan_all().unwrap();
        println!("Total records: {}", records.len());
        assert_eq!(records.len(), 1000, "Expected 1000 records");
    }
}

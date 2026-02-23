//! Lock-free WAL writer with group commit for high-throughput durability.
//!
//! Uses atomic operations for LSN assignment and a ring buffer for buffering
//! records. A dedicated flush thread writes records to disk in order,
//! amortizing fsync across batches (group commit).
//!
//! The hot path (append) is lock-free, making this writer scale well under
//! concurrent load from multiple transactions.

use crate::record::{LogRecordType, Lsn, record_size_for_payload, serialize_raw};
use crate::ring_buffer::RingBuffer;
use crate::segment::{LogSegment, SegmentHeader, SegmentId};
use crate::sequencer::LsnSequencer;
use parking_lot::{Condvar, Mutex};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::JoinHandle;
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
/// 1. Reserve LSN atomically (CAS loop)
/// 2. Serialize header + payload + checksum directly into ring buffer
/// 3. Commit write (atomic cursor advance)
/// 4. Conditionally wake flush thread via thread::unpark
pub struct WalWriter {
    /// Lock-free LSN sequencer, shared with the flush thread for rotation.
    sequencer: Arc<LsnSequencer>,
    /// Ring buffer for pending records.
    ring_buffer: Arc<RingBuffer>,
    /// Flush thread handle for unpark-based wakeup. Set by the flush thread on
    /// startup via OnceLock, so get() is a single atomic load (~1ns).
    flush_thread_waker: Arc<OnceLock<std::thread::Thread>>,
    /// Signal from flush thread to waiters when a flush cycle completes.
    flush_done: Arc<(Mutex<bool>, Condvar)>,
    /// Shutdown flag.
    shutdown: Arc<AtomicBool>,
    /// Flush thread join handle.
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
    /// Creates a new WAL writer. All I/O is synchronous.
    pub fn new(config: WalWriterConfig) -> Result<Self> {
        std::fs::create_dir_all(&config.wal_dir)?;

        let (segment, initial_lsn) = Self::recover_or_create(&config)?;
        let segment_id = segment.segment_id().0;
        let write_offset = segment.write_offset();

        let sequencer = Arc::new(LsnSequencer::new(
            segment_id,
            write_offset,
            config.segment_size,
        ));
        let ring_buffer = Arc::new(RingBuffer::new(config.ring_buffer_capacity));
        let flush_thread_waker = Arc::new(OnceLock::new());
        let flush_done = Arc::new((Mutex::new(false), Condvar::new()));
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
            flush_thread_waker.clone(),
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
            flush_thread_waker,
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
    fn recover_or_create(config: &WalWriterConfig) -> Result<(LogSegment, Lsn)> {
        let mut segments: Vec<PathBuf> = Vec::new();

        for entry in std::fs::read_dir(&config.wal_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().map(|ext| ext == "wal").unwrap_or(false) {
                segments.push(path);
            }
        }

        if segments.is_empty() {
            let segment_id = SegmentId::FIRST;
            let first_lsn = Lsn::new(segment_id.0, SegmentHeader::SIZE as u32);
            let segment =
                LogSegment::create(&config.wal_dir, segment_id, first_lsn, config.segment_size)?;
            return Ok((segment, first_lsn));
        }

        segments.sort();
        let latest_path = segments.last().ok_or_else(|| {
            ZyronError::Internal("WAL segments list unexpectedly empty".to_string())
        })?;
        let segment = LogSegment::open(latest_path)?;
        let next_lsn = Lsn::new(segment.segment_id().0, segment.write_offset());

        Ok((segment, next_lsn))
    }

    /// Spawns the background flush thread.
    ///
    /// Uses std::thread::park/unpark for lightweight sleep/wake with no runtime
    /// overhead. The thread registers its handle via OnceLock so that append()
    /// can call unpark() with a single atomic load.
    fn spawn_flush_thread(
        ring_buffer: Arc<RingBuffer>,
        segment: Arc<Mutex<Option<LogSegment>>>,
        flush_thread_waker: Arc<OnceLock<std::thread::Thread>>,
        flush_done: Arc<(Mutex<bool>, Condvar)>,
        shutdown: Arc<AtomicBool>,
        flushed_lsn: Arc<AtomicU64>,
        fsync_enabled: bool,
        sequencer: Arc<LsnSequencer>,
        rotation: Arc<(Mutex<RotationState>, Condvar)>,
        wal_dir: PathBuf,
        segment_size: u32,
    ) -> JoinHandle<()> {
        std::thread::spawn(move || {
            // Register this thread for unpark wakeup. OnceLock::get() is a
            // single atomic Acquire load, so the append() hot path pays ~1ns
            // instead of tokio::sync::Notify's internal locking (~15-25ns).
            flush_thread_waker.set(std::thread::current()).ok();

            let mut batch_buffer = Vec::with_capacity(64 * 1024);

            loop {
                // Check for work before parking. This prevents missed wakeups:
                // if unpark() fires between our last drain_into() and park(),
                // the token is consumed and park() returns immediately.
                let has_data = !ring_buffer.is_empty();
                let has_rotation = {
                    let state = rotation.0.lock();
                    state.pending
                };

                if !has_data && !has_rotation && !shutdown.load(Ordering::Acquire) {
                    // Short park with timeout so records below the 4KB wakeup threshold
                    // still get flushed promptly.
                    std::thread::park_timeout(std::time::Duration::from_micros(50));
                }

                if shutdown.load(Ordering::Acquire) {
                    // Final drain before exit
                    Self::flush_records_sync(
                        &ring_buffer,
                        &segment,
                        &mut batch_buffer,
                        &flushed_lsn,
                        &flush_done,
                        fsync_enabled,
                    );
                    break;
                }

                // Flush any pending records
                Self::flush_records_sync(
                    &ring_buffer,
                    &segment,
                    &mut batch_buffer,
                    &flushed_lsn,
                    &flush_done,
                    fsync_enabled,
                );

                // Handle segment rotation if requested by append()
                Self::handle_rotation_sync(
                    &rotation,
                    &ring_buffer,
                    &segment,
                    &sequencer,
                    &wal_dir,
                    segment_size,
                    fsync_enabled,
                );
            }
        })
    }

    /// Creates a new segment and advances the sequencer to complete rotation.
    ///
    /// Called by the flush thread after draining the ring buffer. Before creating
    /// the new segment, spins until all in-flight writes to the old segment commit,
    /// then does a final drain to capture any bytes committed after the main flush.
    /// This prevents cross-segment contamination caused by delayed commit_write calls.
    fn handle_rotation_sync(
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

        // Drain any bytes committed after flush_records_sync() ran. These belong to the old
        // segment and must be written before the new segment is installed.
        let mut residual = Vec::new();
        ring_buffer.drain_into(&mut residual);
        if !residual.is_empty() {
            let mut seg_guard = segment.lock();
            if let Some(ref mut seg) = *seg_guard {
                let _ = seg.append_batch(&residual);
            }
        }

        let new_segment_id = old_segment_id + 1;
        let first_lsn = Lsn::new(new_segment_id, SegmentHeader::SIZE as u32);

        // Sync old segment before switching
        {
            let mut seg_guard = segment.lock();
            if let Some(ref mut seg) = *seg_guard {
                if fsync_enabled {
                    let _ = seg.sync();
                }
            }
        }

        // Create the new segment file
        match LogSegment::create(wal_dir, SegmentId(new_segment_id), first_lsn, segment_size) {
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

    /// Flushes records from ring buffer to disk. Fully synchronous.
    fn flush_records_sync(
        ring_buffer: &RingBuffer,
        segment: &Mutex<Option<LogSegment>>,
        batch_buffer: &mut Vec<u8>,
        flushed_lsn: &AtomicU64,
        flush_done: &Arc<(Mutex<bool>, Condvar)>,
        fsync_enabled: bool,
    ) {
        batch_buffer.clear();
        let max_lsn = ring_buffer.drain_into(batch_buffer);

        if batch_buffer.is_empty() {
            // Signal waiters even when empty so flush() callers don't hang
            let (lock, condvar) = flush_done.as_ref();
            let mut done = lock.lock();
            *done = true;
            condvar.notify_all();
            return;
        }

        // Write to segment
        {
            let mut seg_guard = segment.lock();
            if let Some(ref mut seg) = *seg_guard {
                if let Err(e) = seg.append_batch(batch_buffer) {
                    eprintln!("WAL flush error: {:?}", e);
                    let (lock, condvar) = flush_done.as_ref();
                    let mut done = lock.lock();
                    *done = true;
                    condvar.notify_all();
                    return;
                }

                if fsync_enabled {
                    if let Err(e) = seg.sync() {
                        eprintln!("WAL sync error: {:?}", e);
                        let (lock, condvar) = flush_done.as_ref();
                        let mut done = lock.lock();
                        *done = true;
                        condvar.notify_all();
                        return;
                    }
                }
            }
        }

        flushed_lsn.store(max_lsn.0, Ordering::Release);

        // Signal flush waiters
        let (lock, condvar) = flush_done.as_ref();
        let mut done = lock.lock();
        *done = true;
        condvar.notify_all();
    }

    /// Appends a log record. Zero-allocation hot path.
    ///
    /// Serializes header + payload + checksum directly into a ring buffer slot
    /// using serialize_raw, with no intermediate struct construction. When the
    /// current segment is full, blocks until the flush thread completes rotation.
    #[inline]
    fn append(
        &self,
        txn_id: u32,
        prev_lsn: Lsn,
        record_type: LogRecordType,
        flags: u8,
        payload: &[u8],
    ) -> Result<Lsn> {
        if payload.len() > crate::constants::MAX_PAYLOAD_SIZE {
            return Err(ZyronError::Internal(format!(
                "payload {} bytes exceeds MAX_PAYLOAD_SIZE {}",
                payload.len(),
                crate::constants::MAX_PAYLOAD_SIZE,
            )));
        }
        let record_size = record_size_for_payload(payload.len()) as u32;

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
                    drop(state);
                    let lsn = current_lsn;
                    unsafe {
                        let buf = self.ring_buffer.write_record(record_size as usize, lsn);
                        serialize_raw(
                            buf,
                            lsn,
                            prev_lsn,
                            txn_id,
                            record_type as u8,
                            flags,
                            payload,
                        );
                    }
                    self.ring_buffer.commit_write(record_size as usize, lsn);
                    self.wake_flush_thread();
                    return Ok(lsn);
                }

                if !state.pending {
                    state.old_segment_id = lsn.segment_id();
                    state.pending = true;
                    self.wake_flush_thread();
                }

                condvar.wait(&mut state);
                drop(state);
                continue;
            }

            // Normal path: serialize directly into ring buffer
            unsafe {
                let buf = self.ring_buffer.write_record(record_size as usize, lsn);
                serialize_raw(
                    buf,
                    lsn,
                    prev_lsn,
                    txn_id,
                    record_type as u8,
                    flags,
                    payload,
                );
            }

            self.ring_buffer.commit_write(record_size as usize, lsn);
            self.maybe_wake_flush_thread(record_size as usize);
            return Ok(lsn);
        }
    }

    /// Wakes the flush thread via std::thread::unpark.
    /// OnceLock::get() is a single atomic Acquire load (~1ns).
    #[inline(always)]
    fn wake_flush_thread(&self) {
        if let Some(t) = self.flush_thread_waker.get() {
            t.unpark();
        }
    }

    /// Wakes the flush thread only for large records that should be flushed
    /// immediately. Small records rely on the flush thread's park_timeout(50us)
    /// for batching, avoiding per-record unpark syscalls.
    #[inline(always)]
    fn maybe_wake_flush_thread(&self, record_size: usize) {
        if record_size >= 4096 {
            self.wake_flush_thread();
        }
    }

    /// Waits until the given LSN has been flushed to disk.
    pub fn wait_for_flush(&self, target_lsn: Lsn) -> Result<()> {
        loop {
            if self.flushed_lsn() >= target_lsn {
                return Ok(());
            }
            self.wake_flush_thread();
            let (lock, condvar) = self.flush_done.as_ref();
            let mut done = lock.lock();
            // Short timeout to recheck flushed_lsn
            condvar.wait_for(&mut done, std::time::Duration::from_micros(100));
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
    pub fn flush(&self) -> Result<Lsn> {
        loop {
            if self.ring_buffer.is_empty() {
                return Ok(self.flushed_lsn());
            }
            self.wake_flush_thread();
            let (lock, condvar) = self.flush_done.as_ref();
            let mut done = lock.lock();
            condvar.wait_for(&mut done, std::time::Duration::from_micros(100));
        }
    }

    /// Closes the WAL writer.
    pub fn close(&self) -> Result<()> {
        // First flush any pending records
        self.flush()?;

        // Signal shutdown
        self.shutdown.store(true, Ordering::Release);
        self.wake_flush_thread();

        // Close segment
        let mut seg_guard = self.segment.lock();
        if let Some(ref mut seg) = seg_guard.take() {
            seg.sync()?;
            seg.close()?;
        }

        Ok(())
    }

    // Convenience methods for common record types

    /// Logs a transaction begin.
    #[inline]
    pub fn log_begin(&self, txn_id: u32) -> Result<Lsn> {
        self.append(txn_id, Lsn::INVALID, LogRecordType::Begin, 0, &[])
    }

    /// Logs a transaction commit.
    #[inline]
    pub fn log_commit(&self, txn_id: u32, prev_lsn: Lsn) -> Result<Lsn> {
        self.append(txn_id, prev_lsn, LogRecordType::Commit, 0, &[])
    }

    /// Logs a transaction abort.
    #[inline]
    pub fn log_abort(&self, txn_id: u32, prev_lsn: Lsn) -> Result<Lsn> {
        self.append(txn_id, prev_lsn, LogRecordType::Abort, 0, &[])
    }

    /// Logs an insert operation.
    #[inline]
    pub fn log_insert(&self, txn_id: u32, prev_lsn: Lsn, payload: &[u8]) -> Result<Lsn> {
        self.append(txn_id, prev_lsn, LogRecordType::Insert, 0, payload)
    }

    /// Logs a batch of insert operations with amortized atomic overhead.
    ///
    /// Reserves space for all records in one CAS, serializes them contiguously,
    /// and commits once. Reduces atomic operations from 3N to 3 per batch.
    /// Falls back to per-record append at segment boundaries.
    #[inline]
    pub fn log_insert_batch(&self, inserts: &[(u32, &[u8])]) -> Result<Vec<Lsn>> {
        if inserts.is_empty() {
            return Ok(Vec::new());
        }

        let mut lsns = Vec::with_capacity(inserts.len());
        let mut idx = 0;

        while idx < inserts.len() {
            let mut batch_size: u32 = 0;
            let batch_start = idx;
            let mut batch_end = idx;

            for (_, payload) in &inserts[idx..] {
                let rsize = record_size_for_payload(payload.len()) as u32;
                let new_total = batch_size + rsize;
                if new_total > 256 * 1024 && batch_end > batch_start {
                    break;
                }
                batch_size = new_total;
                batch_end += 1;
            }

            let (base_lsn, needs_rotation) = self.sequencer.reserve(batch_size);

            if needs_rotation {
                let (txn_id, payload) = inserts[idx];
                let lsn = self.append(txn_id, Lsn::INVALID, LogRecordType::Insert, 0, payload)?;
                lsns.push(lsn);
                idx += 1;
                continue;
            }

            let buf_start = unsafe { self.ring_buffer.write_record(batch_size as usize, base_lsn) };

            let mut buf_offset: u32 = 0;
            for &(txn_id, payload) in &inserts[batch_start..batch_end] {
                let rsize = record_size_for_payload(payload.len()) as u32;
                let record_lsn = Lsn::new(base_lsn.segment_id(), base_lsn.offset() + buf_offset);

                unsafe {
                    let buf_ptr = buf_start.add(buf_offset as usize);
                    serialize_raw(
                        buf_ptr,
                        record_lsn,
                        Lsn::INVALID,
                        txn_id,
                        LogRecordType::Insert as u8,
                        0,
                        payload,
                    );
                }

                lsns.push(record_lsn);
                buf_offset += rsize;
            }

            self.ring_buffer
                .commit_write(batch_size as usize, *lsns.last().unwrap());
            self.maybe_wake_flush_thread(batch_size as usize);

            idx = batch_end;
        }

        Ok(lsns)
    }

    /// Logs an update operation.
    #[inline]
    pub fn log_update(&self, txn_id: u32, prev_lsn: Lsn, payload: &[u8]) -> Result<Lsn> {
        self.append(txn_id, prev_lsn, LogRecordType::Update, 0, payload)
    }

    /// Logs a delete operation.
    #[inline]
    pub fn log_delete(&self, txn_id: u32, prev_lsn: Lsn, payload: &[u8]) -> Result<Lsn> {
        self.append(txn_id, prev_lsn, LogRecordType::Delete, 0, payload)
    }

    /// Logs a checkpoint begin marker.
    #[inline]
    pub fn log_checkpoint_begin(&self) -> Result<Lsn> {
        self.append(0, Lsn::INVALID, LogRecordType::CheckpointBegin, 0, &[])
    }

    /// Logs a checkpoint end marker.
    #[inline]
    pub fn log_checkpoint_end(&self, payload: &[u8]) -> Result<Lsn> {
        self.append(0, Lsn::INVALID, LogRecordType::CheckpointEnd, 0, payload)
    }
}

impl Drop for WalWriter {
    fn drop(&mut self) {
        // Signal shutdown
        self.shutdown.store(true, Ordering::Release);
        self.wake_flush_thread();

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
    pub fn log_insert(&mut self, payload: &[u8]) -> Result<Lsn> {
        self.last_lsn = self
            .writer
            .log_insert(self.txn_id, self.last_lsn, payload)?;
        Ok(self.last_lsn)
    }

    /// Logs an update operation.
    #[inline]
    pub fn log_update(&mut self, payload: &[u8]) -> Result<Lsn> {
        self.last_lsn = self
            .writer
            .log_update(self.txn_id, self.last_lsn, payload)?;
        Ok(self.last_lsn)
    }

    /// Logs a delete operation.
    #[inline]
    pub fn log_delete(&mut self, payload: &[u8]) -> Result<Lsn> {
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

    fn create_test_writer() -> (WalWriter, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let config = WalWriterConfig {
            wal_dir: dir.path().to_path_buf(),
            segment_size: LogSegment::DEFAULT_SIZE,
            fsync_enabled: false,
            ring_buffer_capacity: 1024 * 1024, // 1MB
        };
        let writer = WalWriter::new(config).unwrap();
        (writer, dir)
    }

    #[test]
    fn test_wal_writer_creation() {
        let (writer, _dir) = create_test_writer();
        assert!(writer.next_lsn().is_valid());
        assert_eq!(writer.current_segment_id().unwrap(), SegmentId::FIRST);
    }

    #[test]
    fn test_wal_writer_append() {
        let (writer, _dir) = create_test_writer();

        let lsn = writer.log_begin(1).unwrap();

        assert!(lsn.is_valid());
        assert_eq!(lsn.segment_id(), 1);

        writer.close().unwrap();
    }

    #[test]
    fn test_wal_writer_transaction_flow() {
        let (writer, _dir) = create_test_writer();

        let begin_lsn = writer.log_begin(1).unwrap();
        assert!(begin_lsn.is_valid());

        let insert_lsn = writer.log_insert(1, begin_lsn, b"data").unwrap();
        assert!(insert_lsn > begin_lsn);

        let commit_lsn = writer.log_commit(1, insert_lsn).unwrap();
        assert!(commit_lsn > insert_lsn);

        writer.close().unwrap();
    }

    #[test]
    fn test_wal_writer_multiple_transactions() {
        let (writer, _dir) = create_test_writer();

        for i in 1..=10 {
            let begin_lsn = writer.log_begin(i).unwrap();
            let data = format!("data{}", i);
            let insert_lsn = writer.log_insert(i, begin_lsn, data.as_bytes()).unwrap();
            writer.log_commit(i, insert_lsn).unwrap();
        }

        writer.flush().unwrap();
        writer.close().unwrap();
    }

    #[test]
    fn test_wal_writer_flush() {
        let (writer, _dir) = create_test_writer();

        let lsn1 = writer.log_begin(1).unwrap();
        let flushed = writer.flush().unwrap();

        assert!(flushed >= lsn1);
        writer.close().unwrap();
    }

    #[test]
    fn test_wal_writer_recovery() {
        let dir = tempdir().unwrap();
        let config = WalWriterConfig {
            wal_dir: dir.path().to_path_buf(),
            segment_size: LogSegment::DEFAULT_SIZE,
            fsync_enabled: true,
            ring_buffer_capacity: 1024 * 1024, // 1MB
        };

        let final_lsn;
        {
            let writer = WalWriter::new(config.clone()).unwrap();
            writer.log_begin(1).unwrap();
            writer.log_insert(1, Lsn::INVALID, b"test").unwrap();
            final_lsn = writer.log_commit(1, Lsn::INVALID).unwrap();
            writer.close().unwrap();
        }

        {
            let writer = WalWriter::new(config).unwrap();
            assert!(writer.next_lsn() >= final_lsn);
            writer.close().unwrap();
        }
    }

    #[test]
    fn test_txn_wal_handle() {
        let (writer, _dir) = create_test_writer();
        let writer = Arc::new(writer);

        let mut handle = TxnWalHandle::new(writer.clone()).unwrap();
        assert!(handle.txn_id() > 0);

        handle.log_insert(b"row1").unwrap();
        handle.log_update(b"row1_updated").unwrap();
        handle.log_delete(b"row1").unwrap();

        let commit_lsn = handle.commit().unwrap();
        assert!(commit_lsn.is_valid());

        writer.close().unwrap();
    }

    #[test]
    fn test_txn_wal_handle_abort() {
        let (writer, _dir) = create_test_writer();
        let writer = Arc::new(writer);

        let mut handle = TxnWalHandle::new(writer.clone()).unwrap();
        handle.log_insert(b"data").unwrap();

        let abort_lsn = handle.abort().unwrap();
        assert!(abort_lsn.is_valid());

        writer.close().unwrap();
    }

    #[test]
    fn test_wal_writer_checkpoint() {
        let (writer, _dir) = create_test_writer();

        let begin_lsn = writer.log_checkpoint_begin().unwrap();
        let end_lsn = writer.log_checkpoint_end(b"checkpoint data").unwrap();

        assert!(end_lsn > begin_lsn);
        writer.close().unwrap();
    }

    #[test]
    fn test_wal_batch_flush() {
        let dir = tempdir().unwrap();
        let config = WalWriterConfig {
            wal_dir: dir.path().to_path_buf(),
            segment_size: LogSegment::DEFAULT_SIZE,
            fsync_enabled: false,
            ring_buffer_capacity: 1024 * 1024, // 1MB
        };
        let writer = WalWriter::new(config).unwrap();

        // Write 25 records
        for i in 1..=25 {
            writer.log_begin(i).unwrap();
        }

        writer.flush().unwrap();
        writer.close().unwrap();
    }

    #[test]
    fn test_segment_rotation() {
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

        let writer = WalWriter::new(config).unwrap();
        let initial_seg = writer.current_segment_id().unwrap();

        for i in 0..1000 {
            writer.log_insert(1, Lsn::INVALID, &[0u8; 200]).unwrap();
            if i % 100 == 99 {
                let seg = writer.current_segment_id().unwrap();
                println!("After record {}: segment {}", i + 1, seg.0);
            }
        }

        let final_seg = writer.current_segment_id().unwrap();
        writer.close().unwrap();

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

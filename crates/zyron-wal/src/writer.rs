//! Lock-free WAL writer with group commit for high-throughput durability.
//!
//! Uses atomic operations for LSN assignment and a ring buffer for buffering
//! records. A dedicated flush thread writes records to disk in order,
//! amortizing fsync across batches (group commit).
//!
//! The hot path (append) is lock-free, making this writer scale well under
//! concurrent load from multiple transactions.

use crate::durability::DurabilityNotifier;
use crate::record::{
    LogRecordType, Lsn, backfill_checksums, record_size_for_payload, serialize_raw_deferred,
};
use crate::ring_buffer::RingBuffer;
use crate::segment::{LogSegment, SegmentHeader, SegmentId};
use crate::sequencer::LsnSequencer;
use parking_lot::Mutex;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::JoinHandle;
use zyron_common::profile::{self, Phase};
use zyron_common::{Result, ZyronError};

/// Atomic state machine for coordinating segment rotation between append() and the flush thread.
///
/// Packs state into a single AtomicU64:
/// - Bits 0..1: rotation phase (0=Idle, 1=Requested, 2=InProgress, 3=Done)
/// - Bits 2..31: old_segment_id (30 bits, max ~1 billion segments)
/// - Bits 32..63: generation counter (32 bits) for ABA prevention
struct AtomicRotationState {
    packed: AtomicU64,
}

const ROTATION_IDLE: u64 = 0;
const ROTATION_REQUESTED: u64 = 1;
const ROTATION_IN_PROGRESS: u64 = 2;
const ROTATION_DONE: u64 = 3;
const STATE_MASK: u64 = 0b11;
const SEGMENT_SHIFT: u32 = 2;
const SEGMENT_MASK: u64 = 0x3FFF_FFFC; // bits 2..31
const GENERATION_SHIFT: u32 = 32;

impl AtomicRotationState {
    fn new() -> Self {
        Self {
            packed: AtomicU64::new(0),
        }
    }

    /// Packs state, segment_id, and generation into a u64.
    #[inline]
    fn pack(state: u64, segment_id: u32, generation: u32) -> u64 {
        (state & STATE_MASK)
            | (((segment_id as u64) << SEGMENT_SHIFT) & SEGMENT_MASK)
            | ((generation as u64) << GENERATION_SHIFT)
    }

    /// Extracts the rotation phase from a packed value.
    #[inline]
    fn phase(val: u64) -> u64 {
        val & STATE_MASK
    }

    /// Extracts the segment_id from a packed value.
    #[inline]
    fn segment_id(val: u64) -> u32 {
        ((val & SEGMENT_MASK) >> SEGMENT_SHIFT) as u32
    }

    /// Extracts the generation counter from a packed value.
    #[inline]
    fn generation(val: u64) -> u32 {
        (val >> GENERATION_SHIFT) as u32
    }

    /// Attempts to transition Idle -> Requested with the given segment_id.
    /// Returns true if this thread won the CAS race.
    fn request_rotation(&self, old_segment_id: u32) -> bool {
        let current = self.packed.load(Ordering::Acquire);
        if Self::phase(current) != ROTATION_IDLE {
            return false;
        }
        let generation = Self::generation(current);
        let new_val = Self::pack(ROTATION_REQUESTED, old_segment_id, generation);
        self.packed
            .compare_exchange(current, new_val, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok()
    }

    /// Transitions Requested -> InProgress. Called by the flush thread.
    /// Returns the old_segment_id on success.
    fn start_rotation(&self) -> Option<u32> {
        let current = self.packed.load(Ordering::Acquire);
        if Self::phase(current) != ROTATION_REQUESTED {
            return None;
        }
        let segment_id = Self::segment_id(current);
        let generation = Self::generation(current);
        let new_val = Self::pack(ROTATION_IN_PROGRESS, segment_id, generation);
        if self
            .packed
            .compare_exchange(current, new_val, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok()
        {
            Some(segment_id)
        } else {
            None
        }
    }

    /// Transitions InProgress -> Done. Called by the flush thread after rotation completes.
    fn complete_rotation(&self) {
        let current = self.packed.load(Ordering::Acquire);
        let generation = Self::generation(current).wrapping_add(1);
        let new_val = Self::pack(ROTATION_DONE, 0, generation);
        self.packed.store(new_val, Ordering::Release);
    }

    /// Transitions Done -> Idle. Called by waiting append() threads after observing Done.
    fn acknowledge_done(&self) {
        let current = self.packed.load(Ordering::Acquire);
        if Self::phase(current) != ROTATION_DONE {
            return;
        }
        let generation = Self::generation(current);
        let new_val = Self::pack(ROTATION_IDLE, 0, generation);
        // Best-effort CAS. Multiple threads may race, only one wins. That is fine
        // because all threads observe Done and proceed to retry their append.
        let _ = self
            .packed
            .compare_exchange(current, new_val, Ordering::AcqRel, Ordering::Relaxed);
    }

    /// Returns true if a rotation is pending (Requested or InProgress).
    /// Relaxed ordering is sufficient: this is a hint check, the actual
    /// state transition uses CAS with Acquire/Release.
    #[inline]
    fn is_rotating(&self) -> bool {
        let phase = Self::phase(self.packed.load(Ordering::Relaxed));
        phase == ROTATION_REQUESTED || phase == ROTATION_IN_PROGRESS
    }

    /// Returns true if the rotation is done and awaiting acknowledgment.
    #[inline]
    fn is_done(&self) -> bool {
        Self::phase(self.packed.load(Ordering::Relaxed)) == ROTATION_DONE
    }
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
    /// startup via OnceLock so get() is a single atomic Acquire load.
    flush_thread_waker: Arc<OnceLock<std::thread::Thread>>,
    /// Shutdown flag.
    shutdown: Arc<AtomicBool>,
    /// Flush thread join handle. Wrapped in Mutex so close() can join via &self.
    flush_thread: Mutex<Option<JoinHandle<()>>>,
    /// Current segment (only accessed by flush thread).
    segment: Arc<Mutex<Option<LogSegment>>>,
    /// Last flushed LSN.
    flushed_lsn: Arc<AtomicU64>,
    /// Next transaction ID.
    next_txn_id: AtomicU64,
    /// Configuration.
    config: WalWriterConfig,
    /// Segment rotation coordination between append() and the flush thread.
    rotation: Arc<AtomicRotationState>,
    /// Set to true by the flush thread when an I/O error occurs during flush.
    /// Checked by append() to fail fast instead of buffering into a broken WAL.
    flush_io_error: Arc<AtomicBool>,
    /// Total fsync calls (for zyron_stat_wal). Arc-shared with the flush thread.
    pub wal_syncs: Arc<AtomicU64>,
    /// Retention hook that returns the minimum LSN that must be retained.
    /// Used by replication slots to prevent WAL segment deletion.
    retention_hook: parking_lot::RwLock<Option<Arc<dyn Fn() -> Option<Lsn> + Send + Sync>>>,
    /// Called by the flush thread after each flush so an async durability
    /// waiter can be woken without this crate depending on an async runtime.
    /// Set once via register_flush_waker.
    durable_waker: Arc<OnceLock<FlushWaker>>,
    /// Wakes sync durability waiters off the flush thread's critical path. The
    /// flush thread pokes it after each flush; it resumes only the committers a
    /// flush satisfied, overlapped with the next device write.
    notifier: Arc<DurabilityNotifier>,
    /// Notifier thread join handle, joined in close().
    notifier_thread: Mutex<Option<JoinHandle<()>>>,
}

/// Callback invoked by the flush thread after every flush cycle, used to wake
/// transaction-commit durability waiters. Kept as a plain closure so the WAL
/// crate stays free of any async-runtime dependency.
pub type FlushWaker = Arc<dyn Fn() + Send + Sync>;

/// A batch append at or above this many bytes wakes the flush thread eagerly,
/// to bound ring-buffer occupancy. Smaller batches are left for the commit's
/// leader flush so a batch costs one device write instead of two.
const EAGER_FLUSH_BYTES: u32 = 256 * 1024;

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
        let shutdown = Arc::new(AtomicBool::new(false));
        let segment = Arc::new(Mutex::new(Some(segment)));
        let next_segment = Arc::new(Mutex::new(None));
        let flushed_lsn = Arc::new(AtomicU64::new(initial_lsn.0.saturating_sub(1)));
        let rotation = Arc::new(AtomicRotationState::new());
        let flush_io_error = Arc::new(AtomicBool::new(false));
        let wal_syncs = Arc::new(AtomicU64::new(0));
        let durable_waker: Arc<OnceLock<FlushWaker>> = Arc::new(OnceLock::new());
        let notifier = DurabilityNotifier::new(
            flushed_lsn.clone(),
            flush_io_error.clone(),
            shutdown.clone(),
            durable_waker.clone(),
        );
        let notifier_thread = notifier.spawn();
        let flush_thread = Self::spawn_flush_thread(
            ring_buffer.clone(),
            segment.clone(),
            next_segment.clone(),
            flush_thread_waker.clone(),
            shutdown.clone(),
            flushed_lsn.clone(),
            config.fsync_enabled,
            sequencer.clone(),
            rotation.clone(),
            config.wal_dir.clone(),
            config.segment_size,
            flush_io_error.clone(),
            wal_syncs.clone(),
            notifier.clone(),
        );

        Ok(Self {
            sequencer,
            ring_buffer,
            flush_thread_waker,
            shutdown,
            flush_thread: Mutex::new(Some(flush_thread)),
            segment,
            flushed_lsn,
            next_txn_id: AtomicU64::new(1),
            config,
            rotation,
            flush_io_error,
            wal_syncs,
            retention_hook: parking_lot::RwLock::new(None),
            durable_waker,
            notifier,
            notifier_thread: Mutex::new(Some(notifier_thread)),
        })
    }

    /// Address used as the parking key for sync durability waiters. Stable for
    /// the writer's lifetime (the flushed-LSN atomic lives on the heap behind an
    /// Arc), and identical to the address flush_records_sync wakes by.
    #[inline]
    fn flushed_lsn_key(&self) -> usize {
        Arc::as_ptr(&self.flushed_lsn) as usize
    }

    /// Registers a callback the flush thread invokes after each flush cycle,
    /// used to wake transaction-commit durability waiters. Set once.
    pub fn register_flush_waker(&self, waker: FlushWaker) {
        let _ = self.durable_waker.set(waker);
    }

    /// Wakes the flush thread so a pending commit record is fsync'd promptly,
    /// rather than waiting for the idle park timeout. Called by a committer
    /// awaiting durability.
    pub fn request_flush(&self) {
        self.wake_flush_thread();
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
    #[allow(clippy::too_many_arguments)]
    fn spawn_flush_thread(
        ring_buffer: Arc<RingBuffer>,
        segment: Arc<Mutex<Option<LogSegment>>>,
        next_segment: Arc<Mutex<Option<LogSegment>>>,
        flush_thread_waker: Arc<OnceLock<std::thread::Thread>>,
        shutdown: Arc<AtomicBool>,
        flushed_lsn: Arc<AtomicU64>,
        fsync_enabled: bool,
        sequencer: Arc<LsnSequencer>,
        rotation: Arc<AtomicRotationState>,
        wal_dir: PathBuf,
        segment_size: u32,
        flush_io_error: Arc<AtomicBool>,
        wal_syncs_counter: Arc<AtomicU64>,
        notifier: Arc<DurabilityNotifier>,
    ) -> JoinHandle<()> {
        std::thread::spawn(move || {
            // Register this thread for unpark wakeup. The waker is stored in
            // an OnceLock so the append() hot path reads it with a single
            // atomic Acquire load and calls std::thread::unpark directly.
            flush_thread_waker.set(std::thread::current()).ok();

            // Flush-thread-private scratch buffers. The flush thread is the sole
            // flusher, so these need no synchronization: `batch_buffer` stages
            // the bytes for one device write, and `leftover` carries the tail
            // that did not fit the current segment across to the next flush.
            let mut batch_buffer: Vec<u8> = Vec::with_capacity(64 * 1024);
            let mut leftover: Vec<u8> = Vec::new();

            // Idle backoff state, the park_timeout grows exponentially when
            // no work arrives so a quiet WAL costs near-zero CPU. The hot
            // base interval is 50us so sub-4KB writes still flush within an
            // ms of the writer's commit, an unpark from the writer (via
            // wake_flush_thread) immediately returns from park_timeout
            // regardless of the timeout so latency on first write is
            // unaffected by the backoff
            const PARK_BASE_US: u64 = 50;
            const PARK_MAX_US: u64 = 10_000;
            // Bounded busy-poll before parking. A committer awaiting durability
            // appends its record then nudges this thread; if the thread is
            // already on-core spinning, it serves that record without paying an
            // unpark plus scheduler dispatch (~10us on Windows), which is the
            // dominant non-fsync commit latency under active load. Sized to a
            // few microseconds so a truly idle WAL still falls through to the
            // park backoff and costs near-zero CPU.
            const SPIN_LIMIT: u32 = 512;
            let mut park_us: u64 = PARK_BASE_US;

            loop {
                // Wait for work before flushing. Spin-poll first so a just
                // committed record is picked up without an unpark dispatch;
                // only park (with exponential backoff) once the spin window
                // expires with no work. Checking for work before parking also
                // prevents missed wakeups: an unpark arriving between the last
                // drain and the park is held as a token and returns immediately.
                let mut spun: u32 = 0;
                loop {
                    if shutdown.load(Ordering::Acquire) {
                        park_us = PARK_BASE_US;
                        break;
                    }
                    // Advance the contiguous watermark over records producers
                    // published since the last drain. Producers publish
                    // wait-free; the flush thread is the sole consumer that
                    // computes the watermark, so it must scan here before
                    // deciding there is drainable work. An overflow always also
                    // requests rotation, so a non-empty leftover implies
                    // is_rotating(); checking rotation covers a pending leftover.
                    // While a device write is in flight, concurrent committers
                    // pile into the ring, so the next drain batches the whole
                    // group in one write (group commit with no artificial window).
                    ring_buffer.advance_committed();
                    if ring_buffer.has_drainable() || rotation.is_rotating() {
                        park_us = PARK_BASE_US;
                        break;
                    }
                    if ring_buffer.has_pending() {
                        // Producers have claimed ring space but not yet published
                        // (mid-write between the claim and the marker store).
                        // Cede the core so they finish, then re-scan, rather than
                        // parking on work that is about to appear or burning a
                        // full core in a tight spin.
                        std::thread::yield_now();
                        continue;
                    }
                    // Fully quiescent: nothing claimed, no rotation, no shutdown.
                    if spun >= SPIN_LIMIT {
                        std::thread::park_timeout(std::time::Duration::from_micros(park_us));
                        park_us = (park_us * 2).min(PARK_MAX_US);
                        break;
                    }
                    spun += 1;
                    std::hint::spin_loop();
                }

                // Capture the rotation state before flushing. An overflow inside
                // flush_records_sync may request a fresh rotation and stage the
                // tail in leftover; that tail must be written to the old segment
                // on a later iteration before the new segment is installed, so
                // rotation is handled the iteration after it is observed here,
                // never in the same flush that staged the overflow.
                let has_rotation = rotation.is_rotating();

                if shutdown.load(Ordering::Acquire) {
                    // Ensure every claimed slot has committed so the final drain
                    // sees all in-flight writes, then drain the ring and any
                    // carried-over leftover to disk, rotating as many times as
                    // needed. A single flush is not enough: if it overflows the
                    // current segment it stages the tail in `leftover`, which a
                    // following rotation + flush must write, or those acked
                    // records are lost at shutdown.
                    ring_buffer.wait_until_committed();
                    loop {
                        if rotation.is_rotating() {
                            Self::handle_rotation_sync(
                                &rotation,
                                &ring_buffer,
                                &segment,
                                &next_segment,
                                &sequencer,
                                &wal_dir,
                                segment_size,
                                fsync_enabled,
                                &mut leftover,
                            );
                        }
                        // Ack a completed rotation back to Idle so a following
                        // overflow's request_rotation can fire (see the main
                        // loop). Without this the shutdown drain could spin
                        // forever on un-writable leftover.
                        rotation.acknowledge_done();
                        Self::flush_records_sync(
                            &ring_buffer,
                            &segment,
                            &mut batch_buffer,
                            &mut leftover,
                            &rotation,
                            &flushed_lsn,
                            fsync_enabled,
                            &flush_io_error,
                            &wal_syncs_counter,
                        );
                        // Done once the ring is drained, nothing is staged for a
                        // following segment, and no rotation is pending.
                        if ring_buffer.is_empty() && leftover.is_empty() && !rotation.is_rotating()
                        {
                            break;
                        }
                        if flush_io_error.load(Ordering::Acquire) {
                            break;
                        }
                    }
                    // Wake any commit awaiting durability before the thread
                    // exits. The notifier drains satisfied waiters and the async
                    // waker; close() also pokes it and joins it.
                    notifier.poke();
                    break;
                }

                // Drain and write all committed records in one batch. The
                // device-write latency is itself the group-commit window:
                // committers that arrive during the previous write are swept
                // into this one. No artificial timer, no committer-led flushing.
                Self::flush_records_sync(
                    &ring_buffer,
                    &segment,
                    &mut batch_buffer,
                    &mut leftover,
                    &rotation,
                    &flushed_lsn,
                    fsync_enabled,
                    &flush_io_error,
                    &wal_syncs_counter,
                );

                // Poke the notifier so it wakes the committers this flush
                // satisfied, off this thread's critical path. A single cheap
                // unpark; the per-committer resumes happen on the notifier
                // thread, overlapped with the next device write below. The
                // notifier also drives the async durability waker. A spurious
                // poke when nothing flushed is harmless: the notifier re-reads
                // flushed_lsn and wakes no one.
                notifier.poke();

                // Pre-allocate the next segment when the live segment is
                // 75% full so the eventual rotation does not pay file
                // create + fsync inside the append-path stall. Cheap to
                // check, only fires once per ~12MB of WAL traffic with
                // the default 16MB segment.
                Self::maybe_prealloc_next_segment(
                    &segment,
                    &next_segment,
                    &sequencer,
                    &wal_dir,
                    segment_size,
                );

                // Handle segment rotation only when it was already pending at
                // the start of this iteration. Using the pre-flush snapshot
                // defers a rotation that this flush's overflow just requested to
                // the next iteration, after the staged leftover is written.
                if has_rotation {
                    Self::handle_rotation_sync(
                        &rotation,
                        &ring_buffer,
                        &segment,
                        &next_segment,
                        &sequencer,
                        &wal_dir,
                        segment_size,
                        fsync_enabled,
                        &mut leftover,
                    );
                    // The flush thread acknowledges its own completed rotation
                    // back to Idle. A rotation the flush requested for its own
                    // segment overflow has no append() waiter to ack the Done
                    // state; leaving it in Done would make the next
                    // request_rotation (CAS Idle->Requested) fail, so the flush
                    // could never get a segment for the staged leftover and
                    // would spin forever while committers wait on a flushed_lsn
                    // that never advances. Writers only need is_rotating()==false
                    // and the advanced sequencer, both true at Idle, so acking
                    // here does not strand a writer that requested the rotation.
                    rotation.acknowledge_done();
                }

                // Re-evaluate any flush()/wait_for_flush waiters once per
                // iteration. flush_records_sync already unparks after a
                // non-empty drain; this also covers the empty-drain and
                // post-rotation cases (a flush() caller waiting for the ring to
                // drain or a rotation to settle), so no waiter is stranded when
                // state changed without records being written. No-op when no
                // thread is parked on the key.
                // SAFETY: key is the address of the flushed_lsn atomic, the same
                // key flush_records_sync and the waiters use.
                unsafe {
                    parking_lot_core::unpark_all(
                        Arc::as_ptr(&flushed_lsn) as usize,
                        parking_lot_core::DEFAULT_UNPARK_TOKEN,
                    );
                }
            }
        })
    }

    /// If the live segment has crossed PREALLOC_THRESHOLD and we do not
    /// already hold a pre-built next segment, create one on the flush
    /// thread so the eventual rotation skips file create + fsync.
    fn maybe_prealloc_next_segment(
        segment: &Mutex<Option<LogSegment>>,
        next_segment: &Mutex<Option<LogSegment>>,
        sequencer: &Arc<LsnSequencer>,
        wal_dir: &Path,
        segment_size: u32,
    ) {
        const PREALLOC_NUMER: u32 = 3;
        const PREALLOC_DENOM: u32 = 4;

        if next_segment.lock().is_some() {
            return;
        }

        let (current_segment_id, current_write_offset) = {
            let guard = segment.lock();
            match guard.as_ref() {
                Some(s) => (s.segment_id().0, s.write_offset()),
                None => return,
            }
        };

        let threshold = (segment_size / PREALLOC_DENOM) * PREALLOC_NUMER;
        if current_write_offset < threshold {
            return;
        }
        // Skip if the sequencer has already advanced past this segment, the
        // rotation finished before pre-allocation got scheduled.
        if sequencer.current_segment_id() != current_segment_id {
            return;
        }

        let new_segment_id = current_segment_id + 1;
        let first_lsn = Lsn::new(new_segment_id, SegmentHeader::SIZE as u32);
        match LogSegment::create(wal_dir, SegmentId(new_segment_id), first_lsn, segment_size) {
            Ok(seg) => {
                *next_segment.lock() = Some(seg);
            }
            Err(e) => {
                eprintln!("WAL pre-allocate segment failed: {:?}", e);
            }
        }
    }

    /// Creates a new segment and advances the sequencer to complete rotation.
    ///
    /// Called by the flush thread after draining the ring buffer. Before creating
    /// the new segment, spins until all in-flight writes to the old segment commit,
    /// then does a final drain to capture any bytes committed after the main flush.
    /// This prevents cross-segment contamination caused by delayed commit_write calls.
    #[allow(clippy::too_many_arguments)]
    fn handle_rotation_sync(
        rotation: &Arc<AtomicRotationState>,
        ring_buffer: &RingBuffer,
        segment: &Mutex<Option<LogSegment>>,
        next_segment: &Mutex<Option<LogSegment>>,
        sequencer: &Arc<LsnSequencer>,
        wal_dir: &Path,
        segment_size: u32,
        fsync_enabled: bool,
        leftover: &mut Vec<u8>,
    ) {
        // Transition Requested -> InProgress. If no rotation was requested, return.
        let old_segment_id = match rotation.start_rotation() {
            Some(id) => id,
            None => return,
        };

        // Wait for all in-flight writes to the old segment to commit their bytes.
        // This covers threads that called write_record() but haven't called commit_write() yet.
        ring_buffer.wait_until_committed();

        // Drain any bytes committed after flush_records_sync() ran. They were
        // assigned LSNs in the old segment's range but their checksums have not
        // been backfilled yet (the flush thread does that just before writing),
        // so they are staged into `leftover` rather than written here. The next
        // flush backfills and writes the whole batch to the freshly rotated
        // segment. Writing them to the old segment now would persist a stale
        // placeholder checksum (recovery would stop at it), and dropping them
        // would let flushed_lsn advance over a record not on disk. drain_into
        // appends, so the residual lands after any overflow already staged this
        // iteration, preserving LSN order.
        ring_buffer.drain_into(leftover);

        let new_segment_id = old_segment_id + 1;

        // Sync old segment before switching
        {
            let mut seg_guard = segment.lock();
            if let Some(ref mut seg) = *seg_guard
                && fsync_enabled
            {
                let _ = seg.sync();
            }
        }

        // Prefer the pre-allocated next segment if it matches the rotation
        // target. Falling back to LogSegment::create here covers the rare
        // case where rotation fires before pre-allocation got scheduled.
        let new_seg_result = {
            let mut next_guard = next_segment.lock();
            match next_guard.take() {
                Some(seg) if seg.segment_id().0 == new_segment_id => Ok(seg),
                Some(_stale) => {
                    // Stale pre-allocated segment, create on demand.
                    let first_lsn = Lsn::new(new_segment_id, SegmentHeader::SIZE as u32);
                    LogSegment::create(wal_dir, SegmentId(new_segment_id), first_lsn, segment_size)
                }
                None => {
                    let first_lsn = Lsn::new(new_segment_id, SegmentHeader::SIZE as u32);
                    LogSegment::create(wal_dir, SegmentId(new_segment_id), first_lsn, segment_size)
                }
            }
        };

        match new_seg_result {
            Ok(new_seg) => {
                {
                    let mut seg_guard = segment.lock();
                    *seg_guard = Some(new_seg);
                }
                sequencer.advance_segment(new_segment_id);
            }
            Err(e) => {
                eprintln!("WAL segment rotation error: {:?}", e);
                // Rotation failure is fatal. New appends will fail fast via flush_io_error.
            }
        }

        // Signal InProgress -> Done. Waiting append() threads will observe this
        // and transition Done -> Idle before retrying.
        rotation.complete_rotation();
    }

    /// Flushes records from ring buffer to disk. Fully synchronous.
    ///
    /// `leftover` carries bytes that did not fit in the previous flush's
    /// segment. They are prepended to the current drain and written to the
    /// freshly rotated segment. When the combined batch overflows the current
    /// segment (which can happen when ring-buffer straddle padding inflates
    /// drained bytes beyond the LSN range), the function writes what fits,
    /// stores the tail in `leftover`, and triggers rotation
    #[allow(clippy::too_many_arguments)]
    fn flush_records_sync(
        ring_buffer: &RingBuffer,
        segment: &Mutex<Option<LogSegment>>,
        batch_buffer: &mut Vec<u8>,
        leftover: &mut Vec<u8>,
        rotation: &Arc<AtomicRotationState>,
        flushed_lsn: &AtomicU64,
        fsync_enabled: bool,
        flush_io_error: &AtomicBool,
        wal_syncs_counter: &AtomicU64,
    ) {
        batch_buffer.clear();
        // Stage previous-iteration overflow bytes ahead of any new drain
        if !leftover.is_empty() {
            batch_buffer.extend_from_slice(leftover);
            leftover.clear();
        }
        let _ = ring_buffer.drain_into(batch_buffer);

        if batch_buffer.is_empty() {
            return;
        }

        // Count whole records drained this flush, for the per-flush amortization
        // denominator. Only when profiling is enabled.
        if profile::is_enabled() {
            use crate::constants::{CHECKSUM_SIZE, HEADER_SIZE, OFF_PAYLOAD_LEN};
            let mut off = 0usize;
            let mut n = 0u64;
            while off + HEADER_SIZE + CHECKSUM_SIZE <= batch_buffer.len() {
                let pl = u16::from_le_bytes([
                    batch_buffer[off + OFF_PAYLOAD_LEN],
                    batch_buffer[off + OFF_PAYLOAD_LEN + 1],
                ]) as usize;
                off += HEADER_SIZE + pl + CHECKSUM_SIZE;
                n += 1;
            }
            profile::record_value(Phase::FlushBatchRecords, n);
        }

        // Compute checksums for all records in the batch before writing to disk.
        // Deferred from append() hot path to amortize checksum cost in the flush thread
        {
            let _s = profile::scope(Phase::FlushChecksum);
            backfill_checksums(batch_buffer);
        }

        let mut current_seg_id: u32 = 0;
        let mut overflow = false;
        // Highest LSN among records actually written to a segment this flush.
        let mut written_max_lsn: u64 = 0;
        {
            let mut seg_guard = segment.lock();
            if let Some(ref mut seg) = *seg_guard {
                current_seg_id = seg.segment_id().0;
                let remaining = seg.remaining_space() as usize;
                // Write only whole records that fit in the segment's remaining
                // space. A record is never split across the boundary: the tail
                // records are carried to `leftover` and written to the next
                // segment. Splitting a record would leave a torn head in this
                // segment (recovery stops at it) and a headerless tail in the
                // next, losing every record after the split point.
                let (to_write_len, prefix_max_lsn) =
                    Self::record_aligned_prefix_len(batch_buffer, remaining);
                written_max_lsn = prefix_max_lsn;

                if to_write_len > 0 {
                    let _s = profile::scope(Phase::FlushSegWrite);
                    if let Err(e) = seg.append_batch(&batch_buffer[..to_write_len]) {
                        eprintln!(
                            "WAL flush error seg={} write_off={} chunk={} err={:?}",
                            current_seg_id,
                            seg.write_offset(),
                            to_write_len,
                            e,
                        );
                        flush_io_error.store(true, Ordering::Release);
                        return;
                    }
                }

                if to_write_len < batch_buffer.len() {
                    // Tail overflowed, stash it and trigger rotation
                    leftover.extend_from_slice(&batch_buffer[to_write_len..]);
                    overflow = true;
                }

                if fsync_enabled && to_write_len > 0 {
                    let _s = profile::scope(Phase::FlushFsync);
                    if let Err(e) = seg.sync() {
                        eprintln!("WAL sync error: {:?}", e);
                        flush_io_error.store(true, Ordering::Release);
                        return;
                    }
                    wal_syncs_counter.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        if overflow {
            rotation.request_rotation(current_seg_id);
        }

        // Advance the durable watermark to the highest LSN actually written to
        // disk this flush. Records that overflowed into `leftover` are NOT on
        // disk yet, so they do not advance flushed_lsn until a later flush
        // writes them. Crucially, when the batch is leftover-only (the ring
        // drained nothing new), this still advances flushed_lsn to the leftover
        // records now on disk, instead of leaving it frozen below their LSN and
        // parking their committers forever. fetch_max keeps it monotonic against
        // a concurrent higher store.
        if written_max_lsn > 0 {
            let mut cur = flushed_lsn.load(Ordering::Acquire);
            while written_max_lsn > cur {
                match flushed_lsn.compare_exchange_weak(
                    cur,
                    written_max_lsn,
                    Ordering::Release,
                    Ordering::Acquire,
                ) {
                    Ok(_) => break,
                    Err(observed) => cur = observed,
                }
            }
        }

        // Wake any flush() callers parked on the flushed-LSN address. Sync commit
        // durability waiters no longer park here (they use the notifier), so this
        // bucket is normally empty and the call is a cheap lock-and-check. The
        // store above is Release and unpark_all takes the parking bucket lock,
        // while a parked flush() caller re-reads its drained predicate under that
        // same bucket lock, so a state change is never lost to a wake that fires
        // before it parks.
        // SAFETY: the key is the address of this same flushed_lsn atomic, which
        // is exactly the key flush() callers park on.
        unsafe {
            parking_lot_core::unpark_all(
                flushed_lsn as *const AtomicU64 as usize,
                parking_lot_core::DEFAULT_UNPARK_TOKEN,
            );
        }
    }

    /// Returns the byte length of the largest prefix of `batch` made up of
    /// whole records that fits within `limit` bytes. Records are length-prefixed
    /// by the header's payload_len, so the prefix always ends on a record
    /// boundary and a record is never split across `limit`. Returns 0 when even
    /// the first record exceeds `limit` (the whole batch then rotates to the
    /// next segment). The batch is a clean stream of whole records here, since
    /// it is the drained ring contents plus any prior whole-record leftover.
    /// Returns (prefix_len, max_lsn) for the largest whole-record prefix of
    /// `batch` that fits in `limit`. `max_lsn` is the highest LSN among the
    /// non-padding records in that prefix, i.e. the highest LSN actually written
    /// to disk; flushed_lsn is driven from it so durability is acked only for
    /// records on disk (and never for a record that overflowed into leftover).
    fn record_aligned_prefix_len(batch: &[u8], limit: usize) -> (usize, u64) {
        use crate::constants::{
            CHECKSUM_SIZE, HEADER_SIZE, OFF_LSN, OFF_PAYLOAD_LEN, OFF_RECORD_TYPE,
        };
        let mut offset = 0;
        let mut max_lsn = 0u64;
        while offset + HEADER_SIZE + CHECKSUM_SIZE <= batch.len() {
            let payload_len = u16::from_le_bytes([
                batch[offset + OFF_PAYLOAD_LEN],
                batch[offset + OFF_PAYLOAD_LEN + 1],
            ]) as usize;
            let record_size = HEADER_SIZE + payload_len + CHECKSUM_SIZE;
            if offset + record_size > batch.len() || offset + record_size > limit {
                break;
            }
            // Padding records (LogRecordType::Invalid == 0) carry no meaningful
            // LSN, so they do not advance the durable watermark.
            if batch[offset + OFF_RECORD_TYPE] != 0 {
                let lsn = u64::from_le_bytes(
                    batch[offset + OFF_LSN..offset + OFF_LSN + 8]
                        .try_into()
                        .unwrap(),
                );
                if lsn > max_lsn {
                    max_lsn = lsn;
                }
            }
            offset += record_size;
        }
        (offset, max_lsn)
    }

    /// Cold error path for payload size validation. Separated from append()
    /// to keep the hot path small and branch-predictor-friendly.
    #[cold]
    #[inline(never)]
    fn payload_too_large(len: usize) -> Result<Lsn> {
        Err(ZyronError::Internal(format!(
            "payload {} bytes exceeds MAX_PAYLOAD_SIZE {}",
            len,
            crate::constants::MAX_PAYLOAD_SIZE,
        )))
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
        if self.flush_io_error.load(Ordering::Acquire) {
            return Err(ZyronError::WalWriteFailed(
                "flush thread encountered an I/O error".into(),
            ));
        }
        if payload.len() > crate::constants::MAX_PAYLOAD_SIZE {
            return Self::payload_too_large(payload.len());
        }
        let record_size = record_size_for_payload(payload.len()) as u32;

        loop {
            // Reserve LSN atomically (lock-free)
            let (lsn, needs_rotation) = self.sequencer.reserve(record_size);

            if needs_rotation {
                // If rotation is already done by a previous cycle, acknowledge and retry.
                if self.rotation.is_done() {
                    self.rotation.acknowledge_done();
                    continue;
                }

                // Try to reserve again, another thread may have completed rotation.
                let (current_lsn, still_full) = self.sequencer.reserve(record_size);
                if !still_full {
                    let lsn = current_lsn;
                    let claim_offset;
                    unsafe {
                        let (buf, off) = self.ring_buffer.write_record(record_size as usize);
                        claim_offset = off;
                        serialize_raw_deferred(
                            buf,
                            lsn,
                            prev_lsn,
                            txn_id,
                            record_type as u8,
                            flags,
                            payload,
                        );
                    }
                    self.ring_buffer.publish(claim_offset);
                    self.wake_flush_thread();
                    return Ok(lsn);
                }

                // Request rotation (CAS Idle -> Requested). Only one thread wins.
                self.rotation.request_rotation(lsn.segment_id());
                self.wake_flush_thread();

                // Spin while rotation is in progress (REQUESTED or IN_PROGRESS)
                // Exit on either DONE (we got the wakeup) or IDLE (another waiter
                // already observed DONE and acknowledged it). Spinning on
                // !is_done() alone deadlocks late waiters, the first waiter
                // would ack DONE -> IDLE, and the rest would never see DONE again
                while self.rotation.is_rotating() {
                    std::thread::park_timeout(std::time::Duration::from_micros(10));
                }

                // Acknowledge Done -> Idle so the next rotation cycle can proceed.
                // Late waiters observing IDLE short-circuit inside acknowledge_done.
                self.rotation.acknowledge_done();
                continue;
            }

            // Normal path: serialize into ring buffer with deferred checksum
            let claim_offset;
            unsafe {
                let (buf, off) = self.ring_buffer.write_record(record_size as usize);
                claim_offset = off;
                serialize_raw_deferred(
                    buf,
                    lsn,
                    prev_lsn,
                    txn_id,
                    record_type as u8,
                    flags,
                    payload,
                );
            }

            self.ring_buffer.publish(claim_offset);
            self.maybe_wake_flush_thread(record_size as usize);
            return Ok(lsn);
        }
    }

    /// Wakes the flush thread via std::thread::unpark.
    /// OnceLock::get() is a single atomic Acquire load on the hot path.
    #[inline(always)]
    fn wake_flush_thread(&self) {
        if let Some(t) = self.flush_thread_waker.get() {
            t.unpark();
        }
    }

    /// Wakes the flush thread only for large records that should be flushed
    /// immediately. Small records rely on the flush thread's park_timeout
    /// loop for batching, avoiding per-record unpark syscalls.
    #[inline(always)]
    fn maybe_wake_flush_thread(&self, record_size: usize) {
        if record_size >= 4096 {
            self.wake_flush_thread();
        }
    }

    /// Blocks until the WAL has durably flushed at least up to `target_lsn`.
    ///
    /// Nudges the flush thread so the committer's record is written, then waits
    /// via the durability notifier: a bounded pre-spin catches the fastest
    /// commits, otherwise the committer registers its target LSN and parks. The
    /// flush thread, after storing flushed_lsn, pokes the notifier, which wakes
    /// only the committers a flush satisfied, off the flush thread's critical
    /// path. No broadcast wake of every parked committer, no flush-thread stall
    /// behind the wakes.
    pub fn wait_for_flush(&self, target_lsn: Lsn) -> Result<()> {
        // Ensure the flush thread runs even if it had parked on an idle ring.
        self.wake_flush_thread();
        self.notifier.wait(target_lsn)
    }

    /// Returns the last flushed LSN.
    #[inline]
    pub fn flushed_lsn(&self) -> Lsn {
        Lsn(self.flushed_lsn.load(Ordering::Acquire))
    }

    /// Total bytes written across the writer's lifetime. Served from the
    /// ring buffer's existing committed-cursor atomic so the hot write path
    /// doesn't maintain a duplicate counter.
    #[inline]
    pub fn wal_bytes_written(&self) -> u64 {
        self.ring_buffer.total_committed_bytes()
    }

    /// Total records written across the writer's lifetime. Updated
    /// synchronously in commit_write so observers see the count immediately
    #[inline]
    pub fn wal_records_written(&self) -> u64 {
        self.ring_buffer.total_committed_records()
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

    /// Returns the current segment ID as a raw u32 without error wrapping.
    /// Cheap atomic read, used by the checkpoint scheduler for WAL growth triggers.
    #[inline]
    pub fn segment_id(&self) -> u32 {
        self.sequencer.current_segment_id()
    }

    /// Returns the WAL directory.
    #[inline]
    pub fn wal_dir(&self) -> &Path {
        &self.config.wal_dir
    }

    /// Sets a retention hook that returns the minimum LSN that must be retained.
    /// Used by replication slots to prevent WAL segment deletion.
    pub fn set_retention_hook(&self, hook: Arc<dyn Fn() -> Option<Lsn> + Send + Sync>) {
        *self.retention_hook.write() = Some(hook);
    }

    /// Deletes WAL segment files whose records are fully covered by a checkpoint.
    ///
    /// Segments with segment_id strictly less than the checkpoint LSN's segment are
    /// fully below the checkpoint and safe to delete. The segment containing the
    /// checkpoint LSN is kept because recovery replays from that offset.
    ///
    /// Returns the number of segments deleted.
    pub fn cleanup_old_segments(&self, checkpoint_lsn: Lsn) -> Result<usize> {
        let checkpoint_segment_id = checkpoint_lsn.segment_id();

        // Respect replication slot retention: do not delete segments
        // that any active slot still needs.
        // Clone the Arc out of the Mutex so the lock is not held during the hook call.
        let hook_fn = self.retention_hook.read().clone();
        let effective_segment_id = if let Some(ref hook_fn) = hook_fn {
            if let Some(min_lsn) = hook_fn() {
                checkpoint_segment_id.min(min_lsn.segment_id())
            } else {
                checkpoint_segment_id
            }
        } else {
            checkpoint_segment_id
        };

        let mut deleted = 0;

        for entry in std::fs::read_dir(&self.config.wal_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().map(|ext| ext == "wal").unwrap_or(false)
                && let Some(stem) = path.file_stem()
                && let Ok(id) = stem.to_string_lossy().parse::<u32>()
                && id < effective_segment_id
            {
                std::fs::remove_file(&path)?;
                deleted += 1;
            }
        }

        Ok(deleted)
    }

    /// Forces a flush and waits for completion.
    /// Waits until the ring buffer is empty and no segment rotation is in progress.
    /// The dedicated flush thread is the sole flusher; this nudges it and parks
    /// on the flushed-LSN address (advanced on every drain) until drained.
    pub fn flush(&self) -> Result<Lsn> {
        let key = self.flushed_lsn_key();
        let drained = |s: &Self| {
            s.ring_buffer.is_empty() && !s.rotation.is_rotating() && !s.rotation.is_done()
        };
        self.wake_flush_thread();
        loop {
            if drained(self) {
                return Ok(self.flushed_lsn());
            }
            if self.flush_io_error.load(Ordering::Acquire) {
                return Err(ZyronError::WalWriteFailed(
                    "flush thread encountered an I/O error".into(),
                ));
            }
            // SAFETY: key is the address of self.flushed_lsn.
            unsafe {
                parking_lot_core::park(
                    key,
                    || !drained(self) && !self.flush_io_error.load(Ordering::Acquire),
                    || self.wake_flush_thread(),
                    |_, _| {},
                    parking_lot_core::DEFAULT_PARK_TOKEN,
                    None,
                );
            }
        }
    }

    /// Closes the WAL writer.
    ///
    /// Flushes pending records, signals the flush thread to exit, joins it
    /// to guarantee all in-flight writes and rotations complete, then closes
    /// the final segment file.
    pub fn close(&self) -> Result<()> {
        // Flush pending records and wait for rotation to settle
        self.flush()?;

        // Signal shutdown and wake the flush thread
        self.shutdown.store(true, Ordering::Release);
        self.wake_flush_thread();

        // Join the flush thread to guarantee it has finished all writes.
        // Without this, close() can take the segment out of the Option while
        // the flush thread is between drain_into() and seg.append_batch(),
        // causing the flush thread to see None and silently drop the batch.
        if let Some(handle) = self.flush_thread.lock().take() {
            let _ = handle.join();
        }

        // Wake and join the notifier thread after the flush thread has settled,
        // so its final drain delivers any last durability wakes before exit.
        self.notifier.poke();
        if let Some(handle) = self.notifier_thread.lock().take() {
            let _ = handle.join();
        }

        // Close segment (flush thread has exited, safe to take ownership)
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

    /// Logs a batch of insert operations with amortized atomic overhead
    ///
    /// Reserves space for all records in one CAS, serializes them contiguously,
    /// and commits once. Reduces atomic operations from 3N to 3 per batch
    /// Falls back to per-record append at segment boundaries
    #[inline]
    pub fn log_insert_batch(&self, inserts: &[(u32, &[u8])]) -> Result<Vec<Lsn>> {
        if inserts.is_empty() {
            return Ok(Vec::new());
        }
        let mut lsns = Vec::with_capacity(inserts.len());
        self.log_insert_batch_inner(inserts, |lsn| lsns.push(lsn))?;
        Ok(lsns)
    }

    /// Logs a batch of insert operations and returns only the last LSN
    ///
    /// Same batching machinery as log_insert_batch but skips the per-record
    /// Vec<Lsn> allocation, callers like the executor's INSERT operator only
    /// need the last LSN to chain to the Commit record so the per-row LSNs
    /// are pure overhead, this variant runs zero allocations on the success
    /// path
    #[inline]
    pub fn log_insert_batch_last_lsn(&self, inserts: &[(u32, &[u8])]) -> Result<Lsn> {
        if inserts.is_empty() {
            return Ok(Lsn::INVALID);
        }
        let mut last = Lsn::INVALID;
        self.log_insert_batch_inner(inserts, |lsn| last = lsn)?;
        Ok(last)
    }

    /// Shared inner loop, invokes the callback once per record with the
    /// assigned LSN. The callback either pushes into a Vec or remembers the
    /// last value, both compile down to a tight inline loop
    fn log_insert_batch_inner(
        &self,
        inserts: &[(u32, &[u8])],
        mut on_lsn: impl FnMut(Lsn),
    ) -> Result<()> {
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
                on_lsn(lsn);
                idx += 1;
                continue;
            }

            let (buf_start, claim_offset) =
                unsafe { self.ring_buffer.write_record(batch_size as usize) };

            let mut buf_offset: u32 = 0;
            for &(txn_id, payload) in &inserts[batch_start..batch_end] {
                let rsize = record_size_for_payload(payload.len()) as u32;
                let record_lsn = Lsn::new(base_lsn.segment_id(), base_lsn.offset() + buf_offset);

                unsafe {
                    let buf_ptr = buf_start.add(buf_offset as usize);
                    serialize_raw_deferred(
                        buf_ptr,
                        record_lsn,
                        Lsn::INVALID,
                        txn_id,
                        LogRecordType::Insert as u8,
                        0,
                        payload,
                    );
                }
                // Publish each record at its own offset, wait-free, after its
                // bytes are written. The flush thread advances the watermark
                // over the contiguous run of published records.
                self.ring_buffer.publish(claim_offset + buf_offset as u64);

                on_lsn(record_lsn);
                buf_offset += rsize;
            }
            // Do not eagerly flush a normal-sized insert batch. These records
            // are not durability-required until commit, and the commit's leader
            // flush sweeps them into a single device write; eagerly flushing
            // them here would cost a second write-through per batch. Only wake
            // for very large batches, to bound ring-buffer occupancy.
            if batch_size >= EAGER_FLUSH_BYTES {
                self.wake_flush_thread();
            }

            idx = batch_end;
        }

        Ok(())
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

    /// Logs a batch of delete operations with amortized atomic overhead.
    ///
    /// Same batching strategy as log_insert_batch: one CAS reserve, one
    /// commit for the entire batch. Falls back to per-record append at
    /// segment boundaries.
    #[inline]
    pub fn log_delete_batch(&self, deletes: &[(u32, &[u8])]) -> Result<Vec<Lsn>> {
        if deletes.is_empty() {
            return Ok(Vec::new());
        }

        let mut lsns = Vec::with_capacity(deletes.len());
        let mut idx = 0;

        while idx < deletes.len() {
            let mut batch_size: u32 = 0;
            let batch_start = idx;
            let mut batch_end = idx;

            for (_, payload) in &deletes[idx..] {
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
                let (txn_id, payload) = deletes[idx];
                let lsn = self.append(txn_id, Lsn::INVALID, LogRecordType::Delete, 0, payload)?;
                lsns.push(lsn);
                idx += 1;
                continue;
            }

            let (buf_start, claim_offset) =
                unsafe { self.ring_buffer.write_record(batch_size as usize) };

            let mut buf_offset: u32 = 0;
            for &(txn_id, payload) in &deletes[batch_start..batch_end] {
                let rsize = record_size_for_payload(payload.len()) as u32;
                let record_lsn = Lsn::new(base_lsn.segment_id(), base_lsn.offset() + buf_offset);

                unsafe {
                    let buf_ptr = buf_start.add(buf_offset as usize);
                    serialize_raw_deferred(
                        buf_ptr,
                        record_lsn,
                        Lsn::INVALID,
                        txn_id,
                        LogRecordType::Delete as u8,
                        0,
                        payload,
                    );
                }
                // Publish each record at its own offset, wait-free.
                self.ring_buffer.publish(claim_offset + buf_offset as u64);

                lsns.push(record_lsn);
                buf_offset += rsize;
            }
            // See log_insert_batch_last_lsn: defer to the commit's leader flush
            // so a batch is one device write, not two. Only very large batches
            // wake the flush thread eagerly to bound ring-buffer occupancy.
            if batch_size >= EAGER_FLUSH_BYTES {
                self.wake_flush_thread();
            }

            idx = batch_end;
        }

        Ok(lsns)
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

    /// Logs a columnar compaction begin marker.
    #[inline]
    pub fn log_compaction_begin(&self, payload: &[u8]) -> Result<Lsn> {
        self.append(0, Lsn::INVALID, LogRecordType::CompactionBegin, 0, payload)
    }

    /// Logs a columnar compaction end marker. This is the fold commit point.
    #[inline]
    pub fn log_compaction_end(&self, payload: &[u8]) -> Result<Lsn> {
        self.append(0, Lsn::INVALID, LogRecordType::CompactionEnd, 0, payload)
    }

    /// Logs a columnar merge begin marker.
    #[inline]
    pub fn log_merge_begin(&self, payload: &[u8]) -> Result<Lsn> {
        self.append(0, Lsn::INVALID, LogRecordType::MergeBegin, 0, payload)
    }

    /// Logs a columnar merge end marker. This is the merge commit point.
    #[inline]
    pub fn log_merge_end(&self, payload: &[u8]) -> Result<Lsn> {
        self.append(0, Lsn::INVALID, LogRecordType::MergeEnd, 0, payload)
    }

    /// Logs an epoch-tagged columnar value patch.
    #[inline]
    pub fn log_columnar_patch(&self, payload: &[u8]) -> Result<Lsn> {
        self.append(0, Lsn::INVALID, LogRecordType::ColumnarPatch, 0, payload)
    }

    /// Logs a columnar supersede (delete of a columnar-resident row).
    #[inline]
    pub fn log_columnar_supersede(&self, payload: &[u8]) -> Result<Lsn> {
        self.append(
            0,
            Lsn::INVALID,
            LogRecordType::ColumnarSupersede,
            0,
            payload,
        )
    }
}

impl Drop for WalWriter {
    fn drop(&mut self) {
        // Signal shutdown
        self.shutdown.store(true, Ordering::Release);
        self.wake_flush_thread();

        // Wait for flush thread (get_mut is safe in Drop since we have &mut self)
        if let Some(handle) = self.flush_thread.get_mut().take() {
            let _ = handle.join();
        }

        // Wake and join the notifier thread.
        self.notifier.poke();
        if let Some(handle) = self.notifier_thread.get_mut().take() {
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

    // Reproduces the commit_blocking hot loop: a single thread logs a record
    // then waits for its durability, repeatedly. Guards against a regression
    // that would hang or catastrophically slow wait_for_flush; expected runtime
    // is well under the assertion bound on fsync-disabled in-memory flushes.
    #[test]
    fn many_sequential_wait_for_flush() {
        let (writer, _dir) = create_test_writer();
        let mut prev = Lsn::INVALID;
        let start = std::time::Instant::now();
        for _ in 0..20_000 {
            let lsn = writer.log_insert(1, prev, b"x").unwrap();
            writer.wait_for_flush(lsn).unwrap();
            prev = lsn;
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed < std::time::Duration::from_secs(10),
            "20K sequential wait_for_flush took {:?}, regression suspected",
            elapsed
        );
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

    // Regression: a record committed in the window between a flush drain and a
    // segment rotation (the rotation "residual") must never be dropped when the
    // old segment is full. Dropping it lets flushed_lsn advance over a record
    // that is not on disk, so a committer is acked for a write that a crash
    // would lose. Many concurrent writers against a small segment hit that
    // window repeatedly; every acked record must be recoverable.
    #[test]
    fn test_concurrent_rotation_no_residual_loss() {
        use crate::reader::WalReader;
        use std::collections::HashSet;

        let dir = tempdir().unwrap();
        let config = WalWriterConfig {
            wal_dir: dir.path().to_path_buf(),
            // Small segment so the writers force hundreds of rotations.
            segment_size: 32 * 1024,
            fsync_enabled: false,
            ring_buffer_capacity: 1024 * 1024,
        };
        let writer = Arc::new(WalWriter::new(config).unwrap());

        const WRITERS: u32 = 8;
        const PER_WRITER: u32 = 3000;

        let mut handles = Vec::new();
        for t in 0..WRITERS {
            let w = Arc::clone(&writer);
            handles.push(std::thread::spawn(move || {
                for s in 0..PER_WRITER {
                    // Unique (writer, seq) key in the payload; 200 bytes so
                    // records straddle the 32KB segment boundary often.
                    let mut payload = [0u8; 200];
                    payload[..4].copy_from_slice(&t.to_le_bytes());
                    payload[4..8].copy_from_slice(&s.to_le_bytes());
                    let lsn = w.log_insert(t + 1, Lsn::INVALID, &payload).unwrap();
                    // Acknowledge durability: after this returns, the record
                    // must survive recovery.
                    w.wait_for_flush(lsn).unwrap();
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        writer.close().unwrap();

        // Recover every record from disk and confirm no acked record was lost.
        let reader = WalReader::new(dir.path()).unwrap();
        let records = reader.scan_all().unwrap();
        let mut found: HashSet<(u32, u32)> = HashSet::new();
        for rec in &records {
            if rec.record_type == LogRecordType::Insert && rec.payload.len() >= 8 {
                let t = u32::from_le_bytes(rec.payload[..4].try_into().unwrap());
                let s = u32::from_le_bytes(rec.payload[4..8].try_into().unwrap());
                found.insert((t, s));
            }
        }
        let mut missing: Vec<(u32, u32)> = Vec::new();
        for t in 0..WRITERS {
            for s in 0..PER_WRITER {
                if !found.contains(&(t, s)) {
                    missing.push((t, s));
                }
            }
        }
        println!(
            "scanned {} records, found {} unique, missing {} of {}",
            records.len(),
            found.len(),
            missing.len(),
            WRITERS * PER_WRITER
        );
        if !missing.is_empty() {
            println!("first missing: {:?}", &missing[..missing.len().min(20)]);
        }
        assert!(
            missing.is_empty(),
            "{} acked records lost after recovery",
            missing.len()
        );
    }
}

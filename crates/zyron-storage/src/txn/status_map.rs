//! Lock-free, durable transaction commit-status map (the commit log / CLOG).
//!
//! MVCC visibility must tell a committed transaction's writes from an aborted
//! one's after the transaction has left the active set. This map records each
//! transaction's final status so a snapshot can make that distinction with one
//! atomic load. Two status bits per transaction id are packed into atomic words
//! grouped into lazily allocated fixed segments, so a status read is a pointer
//! load plus a masked word load with no lock and no allocation.
//!
//! The map is persisted at each checkpoint and at shutdown and reloaded at
//! startup, so an aborted transaction's writes that reached disk before a crash
//! stay invisible after recovery (the engine performs no physical undo).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};

use zyron_common::{Result, ZyronError};

/// Two-bit status codes. The default 0b00 means active / not yet recorded.
const STATUS_COMMITTED: u64 = 0b01;
const STATUS_ABORTED: u64 = 0b10;

/// Transactions tracked per segment. 1,048,576 ids at 2 bits each is 256 KiB.
const TXNS_PER_SEGMENT: u64 = 1 << 20;
/// 64-bit words per segment (32 transactions per word).
const WORDS_PER_SEGMENT: usize = (TXNS_PER_SEGMENT / 32) as usize;
/// Maximum segments. 4096 * 1,048,576 spans the full u32 transaction id space
/// the tuple header stores.
const MAX_SEGMENTS: usize = 4096;

/// Final status of a transaction as recorded in the commit log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxnStatus {
    /// In progress, or never recorded: not yet committed.
    Active,
    Committed,
    Aborted,
}

struct Segment {
    words: Box<[AtomicU64]>,
}

fn new_segment() -> Box<Segment> {
    let mut words = Vec::with_capacity(WORDS_PER_SEGMENT);
    for _ in 0..WORDS_PER_SEGMENT {
        words.push(AtomicU64::new(0));
    }
    Box::new(Segment {
        words: words.into_boxed_slice(),
    })
}

/// One commit-LSN per transaction id in the segment's range. Used by time-travel
/// to date each transaction's durability point: a tuple is visible at version N
/// when its inserter committed at an LSN <= N and its deleter (if any) committed
/// at an LSN > N. Allocated only when commit-LSN tracking is enabled, so a
/// workload that never reads a past version pays nothing.
struct LsnSegment {
    lsns: Box<[AtomicU64]>,
    /// Largest commit LSN recorded in this segment. Lets the dawn watermark
    /// advance past a whole segment only once every transaction in it committed
    /// at or below the retention floor, which is correct even though commit LSN
    /// is not monotonic in transaction id (a long-running transaction keeps the
    /// max high until the floor passes its commit).
    max_lsn: AtomicU64,
}

fn new_lsn_segment() -> Box<LsnSegment> {
    let mut lsns = Vec::with_capacity(TXNS_PER_SEGMENT as usize);
    for _ in 0..TXNS_PER_SEGMENT {
        lsns.push(AtomicU64::new(0));
    }
    Box::new(LsnSegment {
        lsns: lsns.into_boxed_slice(),
        max_lsn: AtomicU64::new(0),
    })
}

/// Durable commit-status map.
pub struct TxnStatusMap {
    /// Lazily allocated segments. A null pointer means no transaction in that
    /// segment's id range has recorded a status yet.
    segments: Box<[AtomicPtr<Segment>]>,
    /// When true, every transaction is reported committed. Used for the frozen
    /// bootstrap map and for tests that do not exercise abort visibility.
    all_committed: bool,
    /// Smallest transaction id that has ever aborted (u64::MAX if none). A
    /// transaction below this is guaranteed not aborted, so visibility can treat
    /// it as committed without a per-row status lookup once it is also below the
    /// active horizon.
    min_aborted: AtomicU64,
    /// Highest transaction id below which vacuum has reclaimed every aborted
    /// tuple, so all remaining tuples there are committed. Lets the frozen
    /// horizon advance past aborts once their rows are gone.
    vacuum_frozen: AtomicU64,
    /// Segment-aligned id below which segments have been freed. Every
    /// transaction below it is committed (vacuum reclaimed aborted inserts and
    /// cleared aborted-delete stamps before the horizon advanced past it), so a
    /// status read short-circuits to committed without touching a freed segment.
    truncated_below: AtomicU64,
    /// Per-transaction commit LSN, parallel to `segments`. Lazily allocated and
    /// populated only while commit-LSN tracking is enabled. A null segment or a
    /// zero entry means the transaction committed before tracking began and is
    /// dated at the dawn of recorded history (LSN 0), which is at or before every
    /// version tag, so time-travel treats it as already present.
    lsn_segments: Box<[AtomicPtr<LsnSegment>]>,
    /// When true, commits record their LSN into `lsn_segments`. Enabled the first
    /// time a retained version exists (a CREATE VERSION tag or a configured
    /// retention window), so a database that never time-travels allocates no
    /// commit-LSN memory.
    lsn_tracking: std::sync::atomic::AtomicBool,
    /// Smallest version LSN that must remain reconstructable. A committed-delete
    /// tuple whose deleter committed above this floor is still visible at a
    /// retained version, so vacuum must keep it. `u64::MAX` means no version is
    /// retained, so pruning reclaims dead tuples with no time-travel constraint.
    version_retention_floor: AtomicU64,
    /// Segment-aligned transaction id below which commit LSNs have been freed
    /// and `commit_lsn` returns 0 (the dawn). Distinct from `truncated_below`,
    /// which advances with the active horizon for commit-status reads: this one
    /// advances only past transactions that all committed at or below the
    /// retention floor, so dating a transaction as the dawn never crosses a
    /// retained version or window. Bounds commit-LSN memory to the retention
    /// window without mis-dating a transaction that committed after it.
    lsn_dawn_below: AtomicU64,
}

impl TxnStatusMap {
    /// Creates an empty status map: every transaction is Active until recorded.
    pub fn new() -> Self {
        let mut v = Vec::with_capacity(MAX_SEGMENTS);
        for _ in 0..MAX_SEGMENTS {
            v.push(AtomicPtr::new(std::ptr::null_mut()));
        }
        let mut lsn_v = Vec::with_capacity(MAX_SEGMENTS);
        for _ in 0..MAX_SEGMENTS {
            lsn_v.push(AtomicPtr::new(std::ptr::null_mut()));
        }
        Self {
            segments: v.into_boxed_slice(),
            all_committed: false,
            min_aborted: AtomicU64::new(u64::MAX),
            vacuum_frozen: AtomicU64::new(0),
            truncated_below: AtomicU64::new(0),
            lsn_segments: lsn_v.into_boxed_slice(),
            lsn_tracking: std::sync::atomic::AtomicBool::new(false),
            version_retention_floor: AtomicU64::new(u64::MAX),
            lsn_dawn_below: AtomicU64::new(0),
        }
    }

    /// Creates a map that reports every transaction committed. Used as a frozen
    /// bootstrap map and by tests that do not exercise abort visibility.
    pub fn all_committed() -> Self {
        Self {
            segments: Vec::new().into_boxed_slice(),
            all_committed: true,
            min_aborted: AtomicU64::new(u64::MAX),
            vacuum_frozen: AtomicU64::new(0),
            truncated_below: AtomicU64::new(0),
            lsn_segments: Vec::new().into_boxed_slice(),
            lsn_tracking: std::sync::atomic::AtomicBool::new(false),
            version_retention_floor: AtomicU64::new(u64::MAX),
            lsn_dawn_below: AtomicU64::new(0),
        }
    }

    /// Returns the frozen horizon for a snapshot whose oldest active transaction
    /// id is `oldest_active`: every transaction below the returned value is
    /// committed and ended before the snapshot, so visibility can treat it as
    /// visible with a single comparison and no status lookup. Safe because the
    /// horizon never crosses an abort that still has live tuples: it is capped at
    /// the smallest-ever aborted id, raised only by vacuum once it has reclaimed
    /// the aborted rows below a point.
    #[inline]
    pub fn frozen_below(&self, oldest_active: u64) -> u64 {
        if self.all_committed {
            return oldest_active;
        }
        let by_abort = oldest_active.min(self.min_aborted.load(Ordering::Relaxed));
        by_abort.max(self.vacuum_frozen.load(Ordering::Relaxed))
    }

    /// Records that vacuum has reclaimed every aborted tuple with id below
    /// `xid`, allowing the frozen horizon to advance past earlier aborts.
    pub fn advance_vacuum_frozen(&self, xid: u64) {
        self.vacuum_frozen.fetch_max(xid, Ordering::Relaxed);
    }

    #[inline]
    fn locate(txn_id: u64) -> Option<(usize, usize, u32)> {
        let seg = (txn_id / TXNS_PER_SEGMENT) as usize;
        if seg >= MAX_SEGMENTS {
            return None;
        }
        let within = txn_id % TXNS_PER_SEGMENT;
        let word = (within / 32) as usize;
        let shift = ((within % 32) * 2) as u32;
        Some((seg, word, shift))
    }

    /// Returns the segment for `seg`, allocating it if absent. The loser of an
    /// allocation race frees its segment and uses the winner's.
    fn get_or_alloc_segment(&self, seg: usize) -> &Segment {
        let slot = &self.segments[seg];
        let p = slot.load(Ordering::Acquire);
        if !p.is_null() {
            return unsafe { &*p };
        }
        let raw = Box::into_raw(new_segment());
        match slot.compare_exchange(
            std::ptr::null_mut(),
            raw,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => unsafe { &*raw },
            Err(existing) => {
                unsafe { drop(Box::from_raw(raw)) };
                unsafe { &*existing }
            }
        }
    }

    /// Writes the two status bits for a transaction.
    fn set_status(&self, txn_id: u64, status: u64) {
        if self.all_committed {
            return;
        }
        let Some((seg, word, shift)) = Self::locate(txn_id) else {
            return;
        };
        let segment = self.get_or_alloc_segment(seg);
        let w = &segment.words[word];
        let mask = 0b11u64 << shift;
        let want = status << shift;
        let mut cur = w.load(Ordering::Relaxed);
        loop {
            let new = (cur & !mask) | want;
            if new == cur {
                return;
            }
            match w.compare_exchange_weak(cur, new, Ordering::AcqRel, Ordering::Relaxed) {
                Ok(_) => return,
                Err(observed) => cur = observed,
            }
        }
    }

    /// Records that a transaction committed.
    #[inline]
    pub fn record_committed(&self, txn_id: u64) {
        self.set_status(txn_id, STATUS_COMMITTED);
    }

    /// Records that a transaction aborted.
    #[inline]
    pub fn record_aborted(&self, txn_id: u64) {
        self.set_status(txn_id, STATUS_ABORTED);
        // Cap the frozen horizon at the smallest aborted id so visibility never
        // treats an aborted transaction's still-live tuples as committed.
        self.min_aborted.fetch_min(txn_id, Ordering::Relaxed);
    }

    /// Enables commit-LSN tracking. Subsequent commits record their LSN so
    /// time-travel can date them. Called when the first retained version exists.
    /// Idempotent.
    #[inline]
    pub fn enable_lsn_tracking(&self) {
        self.lsn_tracking.store(true, Ordering::Release);
    }

    /// Returns true when commit-LSN tracking is active.
    #[inline]
    pub fn lsn_tracking_enabled(&self) -> bool {
        self.lsn_tracking.load(Ordering::Acquire)
    }

    /// Returns the commit-LSN segment for `seg`, allocating it if absent. The
    /// loser of an allocation race frees its segment and uses the winner's.
    fn get_or_alloc_lsn_segment(&self, seg: usize) -> &LsnSegment {
        let slot = &self.lsn_segments[seg];
        let p = slot.load(Ordering::Acquire);
        if !p.is_null() {
            return unsafe { &*p };
        }
        let raw = Box::into_raw(new_lsn_segment());
        match slot.compare_exchange(
            std::ptr::null_mut(),
            raw,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => unsafe { &*raw },
            Err(existing) => {
                unsafe { drop(Box::from_raw(raw)) };
                unsafe { &*existing }
            }
        }
    }

    /// Records that a transaction committed at the given WAL LSN. Marks the
    /// status committed and, when commit-LSN tracking is enabled, stores the LSN
    /// so time-travel can date the transaction. The commit path always supplies
    /// the LSN of the commit record.
    #[inline]
    pub fn record_committed_at(&self, txn_id: u64, commit_lsn: u64) {
        self.set_status(txn_id, STATUS_COMMITTED);
        if !self.lsn_tracking.load(Ordering::Acquire) || self.all_committed || txn_id == 0 {
            return;
        }
        let Some((seg, _, _)) = Self::locate(txn_id) else {
            return;
        };
        let within = (txn_id % TXNS_PER_SEGMENT) as usize;
        let segment = self.get_or_alloc_lsn_segment(seg);
        segment.lsns[within].store(commit_lsn, Ordering::Release);
        segment.max_lsn.fetch_max(commit_lsn, Ordering::AcqRel);
    }

    /// Returns the commit LSN of a transaction for time-travel dating.
    ///
    /// `Some(lsn)` is the durability point of a committed transaction. `Some(0)`
    /// means committed before tracking began or below the commit-LSN dawn: dated
    /// at the dawn of recorded history, which the dawn watermark guarantees is at
    /// or below every retained version or window. `None` means active or aborted:
    /// never visible at any past version. The dawn uses `lsn_dawn_below`, not the
    /// status `truncated_below`: the latter advances with the active horizon and
    /// could otherwise date a transaction that committed after a retained version
    /// as the dawn, making a not-yet-existing row appear in an AS OF read.
    #[inline]
    pub fn commit_lsn(&self, txn_id: u64) -> Option<u64> {
        if self.all_committed || txn_id == 0 {
            return Some(0);
        }
        if txn_id < self.lsn_dawn_below.load(Ordering::Acquire) {
            return Some(0);
        }
        if self.status(txn_id) != TxnStatus::Committed {
            return None;
        }
        let Some((seg, _, _)) = Self::locate(txn_id) else {
            return Some(0);
        };
        let p = self.lsn_segments[seg].load(Ordering::Acquire);
        if p.is_null() {
            return Some(0);
        }
        let within = (txn_id % TXNS_PER_SEGMENT) as usize;
        let segment = unsafe { &*p };
        Some(segment.lsns[within].load(Ordering::Acquire))
    }

    /// Time-travel visibility at version `version` (a WAL LSN). A tuple is visible
    /// when its inserting transaction committed at an LSN <= version and its
    /// deleting transaction either never committed or committed at an LSN >
    /// version. This is MVCC visibility dated by commit LSN instead of by the
    /// reader's live snapshot.
    #[inline]
    pub fn is_visible_at_version(&self, xmin: u64, xmax: u64, version: u64) -> bool {
        let Some(cl_min) = self.commit_lsn(xmin) else {
            return false;
        };
        if cl_min > version {
            return false;
        }
        if xmax == 0 {
            return true;
        }
        match self.commit_lsn(xmax) {
            Some(cl_max) => cl_max > version,
            None => true,
        }
    }

    /// Lowers the version retention floor to include `version_lsn`, so vacuum
    /// keeps every tuple still visible at that version. Called when a version is
    /// tagged. Monotone downward via fetch_min.
    #[inline]
    pub fn retain_version(&self, version_lsn: u64) {
        self.version_retention_floor
            .fetch_min(version_lsn, Ordering::AcqRel);
    }

    /// Sets the version retention floor to an absolute value. Used to recompute
    /// the floor from the remaining version tags after one is dropped; pass
    /// `u64::MAX` when no version remains.
    #[inline]
    pub fn set_version_retention_floor(&self, floor: u64) {
        self.version_retention_floor.store(floor, Ordering::Release);
    }

    /// Returns the current version retention floor.
    #[inline]
    pub fn version_retention_floor(&self) -> u64 {
        self.version_retention_floor.load(Ordering::Acquire)
    }

    /// Returns the commit-LSN dawn watermark: transactions below it are dated at
    /// the dawn (commit LSN 0).
    #[inline]
    pub fn commit_lsn_dawn(&self) -> u64 {
        self.lsn_dawn_below.load(Ordering::Acquire)
    }

    /// Advances the commit-LSN dawn watermark and frees the commit-LSN segments
    /// it passes, bounding commit-LSN memory to the retention window. A segment
    /// is freed only when every transaction in it committed at or below `floor`
    /// (the global minimum retention floor across version tags and time windows).
    /// Advancing stops at the first segment still holding a transaction that
    /// committed after the floor, so the watermark never dates such a transaction
    /// as the dawn. It is also capped at `truncated_below`: below the status
    /// truncation watermark vacuum has reclaimed every aborted tuple, so dating a
    /// transaction there as the dawn (committed at 0) cannot expose an aborted
    /// insert. Returns the number of segments freed.
    pub fn advance_commit_lsn_dawn(&self, floor: u64) -> usize {
        if self.all_committed {
            return 0;
        }
        let limit_seg = (self.truncated_below.load(Ordering::Acquire) / TXNS_PER_SEGMENT) as usize;
        let start_seg = (self.lsn_dawn_below.load(Ordering::Acquire) / TXNS_PER_SEGMENT) as usize;
        if start_seg >= limit_seg {
            return 0;
        }
        let guard = crossbeam::epoch::pin();
        let mut freed = 0;
        let mut advanced_to = start_seg;
        for seg in start_seg..limit_seg {
            let p = self.lsn_segments[seg].load(Ordering::Acquire);
            if !p.is_null() {
                // Stop before a segment that still holds a post-floor commit, so
                // the dawn never crosses a transaction a retained version sees.
                if unsafe { &*p }.max_lsn.load(Ordering::Acquire) > floor {
                    break;
                }
                let old = self.lsn_segments[seg].swap(std::ptr::null_mut(), Ordering::AcqRel);
                if !old.is_null() {
                    unsafe {
                        guard.defer_unchecked(move || drop(Box::from_raw(old)));
                    }
                    freed += 1;
                }
            }
            advanced_to = seg + 1;
        }
        if advanced_to > start_seg {
            self.lsn_dawn_below
                .fetch_max((advanced_to as u64) * TXNS_PER_SEGMENT, Ordering::AcqRel);
        }
        freed
    }

    /// Returns true when a committed-delete tuple with the given `xmax` may be
    /// physically reclaimed without breaking time-travel: either no version is
    /// retained, or the deleter committed at or before the retention floor so no
    /// retained version still sees the row alive. An uncommitted deleter (an
    /// aborted delete) leaves the row live and is handled by the caller, not
    /// here; this method conservatively keeps it.
    #[inline]
    pub fn version_reclaimable(&self, xmax: u64) -> bool {
        self.is_reclaimable_below(xmax, self.version_retention_floor.load(Ordering::Acquire))
    }

    /// Returns true when a committed-delete tuple with the given `xmax` may be
    /// physically reclaimed under an arbitrary retention floor: the deleter
    /// committed at or before the floor, so no retained version sees the row
    /// alive. `u64::MAX` means no retention (always reclaimable). The vacuum
    /// worker passes a per-table effective floor (the lower of the version-tag
    /// floor and the time-based floor) here.
    #[inline]
    pub fn is_reclaimable_below(&self, xmax: u64, floor: u64) -> bool {
        if floor == u64::MAX {
            return true;
        }
        match self.commit_lsn(xmax) {
            Some(cl) => cl <= floor,
            None => false,
        }
    }

    /// Reads a transaction's recorded status.
    #[inline]
    pub fn status(&self, txn_id: u64) -> TxnStatus {
        if self.all_committed || txn_id == 0 {
            return TxnStatus::Committed;
        }
        // Below the truncation point every transaction is committed and its
        // segment may have been freed; short-circuit without touching it.
        if txn_id < self.truncated_below.load(Ordering::Acquire) {
            return TxnStatus::Committed;
        }
        let Some((seg, word, shift)) = Self::locate(txn_id) else {
            return TxnStatus::Active;
        };
        // Pin the epoch so a concurrent truncation cannot free the segment while
        // it is being read; the free is deferred until no reader holds a guard.
        let _guard = crossbeam::epoch::pin();
        let p = self.segments[seg].load(Ordering::Acquire);
        if p.is_null() {
            return TxnStatus::Active;
        }
        let segment = unsafe { &*p };
        let bits = (segment.words[word].load(Ordering::Acquire) >> shift) & 0b11;
        match bits {
            STATUS_COMMITTED => TxnStatus::Committed,
            STATUS_ABORTED => TxnStatus::Aborted,
            _ => TxnStatus::Active,
        }
    }

    /// Returns true only when the transaction is recorded committed. Transaction
    /// id 0 is the frozen bootstrap sentinel and is always committed.
    #[inline]
    pub fn is_committed(&self, txn_id: u64) -> bool {
        if self.all_committed || txn_id == 0 {
            return true;
        }
        if txn_id < self.truncated_below.load(Ordering::Acquire) {
            return true;
        }
        let Some((seg, word, shift)) = Self::locate(txn_id) else {
            return false;
        };
        let _guard = crossbeam::epoch::pin();
        let p = self.segments[seg].load(Ordering::Acquire);
        if p.is_null() {
            return false;
        }
        let segment = unsafe { &*p };
        ((segment.words[word].load(Ordering::Acquire) >> shift) & 0b11) == STATUS_COMMITTED
    }

    /// Frees commit-status segments that lie entirely below `watermark`,
    /// reclaiming their memory. Safe only when `watermark <= vacuum_frozen`: by
    /// then vacuum has reclaimed every aborted insert and cleared every
    /// aborted-delete stamp below it, so no live tuple references a transaction
    /// there and a status read can answer committed without the segment. The
    /// watermark is published first so readers short-circuit, then segments are
    /// freed through the epoch collector so an in-flight reader is never left
    /// with a dangling pointer. Returns the number of segments freed.
    pub fn truncate_below(&self, watermark: u64) -> usize {
        if self.all_committed {
            return 0;
        }
        // Only whole segments fully below the watermark are safe to free.
        let safe_seg = (watermark / TXNS_PER_SEGMENT) as usize;
        if safe_seg == 0 {
            return 0;
        }
        let aligned = (safe_seg as u64) * TXNS_PER_SEGMENT;
        // Publish the truncation point before freeing so new reads short-circuit.
        let prev = self.truncated_below.fetch_max(aligned, Ordering::AcqRel);
        if aligned <= prev {
            return 0;
        }
        let guard = crossbeam::epoch::pin();
        // Frees only commit-status segments. Commit-LSN segments are freed
        // separately by advance_commit_lsn_dawn, which advances on the retention
        // floor rather than the active horizon, so a retained version's delete is
        // never dated as the dawn.
        let mut freed = 0;
        for seg in (prev / TXNS_PER_SEGMENT) as usize..safe_seg {
            let old = self.segments[seg].swap(std::ptr::null_mut(), Ordering::AcqRel);
            if !old.is_null() {
                unsafe {
                    guard.defer_unchecked(move || drop(Box::from_raw(old)));
                }
                freed += 1;
            }
        }
        freed
    }

    /// Returns true when the transaction is recorded aborted.
    #[inline]
    pub fn is_aborted(&self, txn_id: u64) -> bool {
        self.status(txn_id) == TxnStatus::Aborted
    }

    fn file_path(dir: &Path) -> PathBuf {
        dir.join(".zyclog")
    }

    /// Persists the allocated segments to disk with an atomic rename so a crash
    /// mid-write never leaves a torn file.
    pub fn persist(&self, dir: &Path) -> Result<()> {
        if self.all_committed {
            return Ok(());
        }
        // Pin the epoch so segments are not freed by a concurrent truncation
        // while their words are being serialized.
        let _guard = crossbeam::epoch::pin();
        // Collect allocated segments.
        let mut present: Vec<(u32, &Segment)> = Vec::new();
        for (idx, slot) in self.segments.iter().enumerate() {
            let p = slot.load(Ordering::Acquire);
            if !p.is_null() {
                present.push((idx as u32, unsafe { &*p }));
            }
        }
        let mut buf = Vec::with_capacity(28 + present.len() * (4 + WORDS_PER_SEGMENT * 8));
        // Header: frozen-horizon watermarks, then the segment count.
        buf.extend_from_slice(&self.min_aborted.load(Ordering::Relaxed).to_le_bytes());
        buf.extend_from_slice(&self.vacuum_frozen.load(Ordering::Relaxed).to_le_bytes());
        buf.extend_from_slice(&self.truncated_below.load(Ordering::Relaxed).to_le_bytes());
        buf.extend_from_slice(&(present.len() as u32).to_le_bytes());
        for (idx, seg) in &present {
            buf.extend_from_slice(&idx.to_le_bytes());
            for w in seg.words.iter() {
                buf.extend_from_slice(&w.load(Ordering::Relaxed).to_le_bytes());
            }
        }

        // Commit-LSN section: tracking flag, then allocated commit-LSN segments.
        let mut lsn_present: Vec<(u32, &LsnSegment)> = Vec::new();
        for (idx, slot) in self.lsn_segments.iter().enumerate() {
            let p = slot.load(Ordering::Acquire);
            if !p.is_null() {
                lsn_present.push((idx as u32, unsafe { &*p }));
            }
        }
        buf.extend_from_slice(&(self.lsn_tracking.load(Ordering::Relaxed) as u64).to_le_bytes());
        buf.extend_from_slice(&self.lsn_dawn_below.load(Ordering::Relaxed).to_le_bytes());
        buf.extend_from_slice(&(lsn_present.len() as u32).to_le_bytes());
        for (idx, seg) in &lsn_present {
            buf.extend_from_slice(&idx.to_le_bytes());
            for l in seg.lsns.iter() {
                buf.extend_from_slice(&l.load(Ordering::Relaxed).to_le_bytes());
            }
        }

        let path = Self::file_path(dir);
        let tmp = path.with_extension("zyclog.tmp");
        {
            use std::io::Write;
            let mut f = std::fs::File::create(&tmp).map_err(ZyronError::Io)?;
            f.write_all(&buf).map_err(ZyronError::Io)?;
            f.sync_all().map_err(ZyronError::Io)?;
        }
        std::fs::rename(&tmp, &path).map_err(ZyronError::Io)?;
        Ok(())
    }

    /// Loads persisted segment status from disk. A missing file is a clean no-op.
    pub fn load(&self, dir: &Path) -> Result<()> {
        if self.all_committed {
            return Ok(());
        }
        let path = Self::file_path(dir);
        let data = match std::fs::read(&path) {
            Ok(d) => d,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(ZyronError::Io(e)),
        };
        if data.len() < 28 {
            return Ok(());
        }
        let mut off = 0usize;
        self.min_aborted
            .store(read_u64(&data, &mut off)?, Ordering::Relaxed);
        self.vacuum_frozen
            .store(read_u64(&data, &mut off)?, Ordering::Relaxed);
        self.truncated_below
            .store(read_u64(&data, &mut off)?, Ordering::Relaxed);
        let count = read_u32(&data, &mut off)? as usize;
        let seg_bytes = WORDS_PER_SEGMENT * 8;
        for _ in 0..count {
            let idx = read_u32(&data, &mut off)? as usize;
            if idx >= MAX_SEGMENTS {
                return Err(ZyronError::Internal(
                    "clog segment index out of range".into(),
                ));
            }
            let end = off + seg_bytes;
            let slice = data
                .get(off..end)
                .ok_or_else(|| ZyronError::Internal("clog truncated".into()))?;
            let segment = self.get_or_alloc_segment(idx);
            for (w, chunk) in segment.words.iter().zip(slice.chunks_exact(8)) {
                w.store(
                    u64::from_le_bytes(chunk.try_into().unwrap()),
                    Ordering::Relaxed,
                );
            }
            off = end;
        }
        // Commit-LSN section, present in files written after commit-LSN tracking
        // was added. A file without it leaves tracking disabled and no segments.
        if off < data.len() {
            let tracking = read_u64(&data, &mut off)? != 0;
            self.lsn_tracking.store(tracking, Ordering::Relaxed);
            self.lsn_dawn_below
                .store(read_u64(&data, &mut off)?, Ordering::Relaxed);
            let lsn_count = read_u32(&data, &mut off)? as usize;
            let lsn_seg_bytes = TXNS_PER_SEGMENT as usize * 8;
            for _ in 0..lsn_count {
                let idx = read_u32(&data, &mut off)? as usize;
                if idx >= MAX_SEGMENTS {
                    return Err(ZyronError::Internal(
                        "clog commit-lsn segment index out of range".into(),
                    ));
                }
                let end = off + lsn_seg_bytes;
                let slice = data
                    .get(off..end)
                    .ok_or_else(|| ZyronError::Internal("clog commit-lsn truncated".into()))?;
                let segment = self.get_or_alloc_lsn_segment(idx);
                let mut max_lsn = 0u64;
                for (l, chunk) in segment.lsns.iter().zip(slice.chunks_exact(8)) {
                    let v = u64::from_le_bytes(chunk.try_into().unwrap());
                    l.store(v, Ordering::Relaxed);
                    max_lsn = max_lsn.max(v);
                }
                segment.max_lsn.store(max_lsn, Ordering::Relaxed);
                off = end;
            }
        }
        Ok(())
    }
}

impl Default for TxnStatusMap {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for TxnStatusMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let allocated = self
            .segments
            .iter()
            .filter(|s| !s.load(Ordering::Relaxed).is_null())
            .count();
        f.debug_struct("TxnStatusMap")
            .field("all_committed", &self.all_committed)
            .field("allocated_segments", &allocated)
            .finish()
    }
}

impl Drop for TxnStatusMap {
    fn drop(&mut self) {
        for slot in self.segments.iter() {
            let p = slot.load(Ordering::Relaxed);
            if !p.is_null() {
                unsafe { drop(Box::from_raw(p)) };
            }
        }
        for slot in self.lsn_segments.iter() {
            let p = slot.load(Ordering::Relaxed);
            if !p.is_null() {
                unsafe { drop(Box::from_raw(p)) };
            }
        }
    }
}

fn read_u32(data: &[u8], off: &mut usize) -> Result<u32> {
    let end = *off + 4;
    let slice = data
        .get(*off..end)
        .ok_or_else(|| ZyronError::Internal("clog truncated".into()))?;
    *off = end;
    Ok(u32::from_le_bytes(slice.try_into().unwrap()))
}

fn read_u64(data: &[u8], off: &mut usize) -> Result<u64> {
    let end = *off + 8;
    let slice = data
        .get(*off..end)
        .ok_or_else(|| ZyronError::Internal("clog truncated".into()))?;
    *off = end;
    Ok(u64::from_le_bytes(slice.try_into().unwrap()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_active_then_committed_aborted() {
        let m = TxnStatusMap::new();
        assert_eq!(m.status(5), TxnStatus::Active);
        assert!(!m.is_committed(5));
        m.record_committed(5);
        assert!(m.is_committed(5));
        assert_eq!(m.status(5), TxnStatus::Committed);
        m.record_aborted(6);
        assert!(!m.is_committed(6));
        assert!(m.is_aborted(6));
    }

    #[test]
    fn test_zero_is_frozen_committed() {
        let m = TxnStatusMap::new();
        assert!(m.is_committed(0));
    }

    #[test]
    fn test_all_committed_map() {
        let m = TxnStatusMap::all_committed();
        assert!(m.is_committed(1));
        assert!(m.is_committed(999_999));
        // Recording is a no-op and never reports aborted.
        m.record_aborted(7);
        assert!(m.is_committed(7));
    }

    #[test]
    fn test_commit_lsn_tracking_off_dates_at_dawn() {
        let m = TxnStatusMap::new();
        // Tracking disabled: a committed transaction is dated at the dawn so it
        // is visible at every version, and the database allocates no lsn memory.
        m.record_committed_at(5, 1000);
        assert_eq!(m.commit_lsn(5), Some(0));
        assert!(
            m.lsn_segments
                .iter()
                .all(|s| s.load(Ordering::Relaxed).is_null())
        );
    }

    #[test]
    fn test_commit_lsn_tracking_on_records_lsn() {
        let m = TxnStatusMap::new();
        m.enable_lsn_tracking();
        m.record_committed_at(5, 1000);
        assert_eq!(m.commit_lsn(5), Some(1000));
        // Active and aborted transactions have no commit LSN.
        assert_eq!(m.commit_lsn(6), None);
        m.record_aborted(7);
        assert_eq!(m.commit_lsn(7), None);
    }

    #[test]
    fn test_is_visible_at_version() {
        let m = TxnStatusMap::new();
        m.enable_lsn_tracking();
        // Inserted by txn 1 at LSN 10, deleted by txn 2 at LSN 20.
        m.record_committed_at(1, 10);
        m.record_committed_at(2, 20);
        // Before insert: not visible.
        assert!(!m.is_visible_at_version(1, 0, 5));
        // After insert, before delete: visible.
        assert!(m.is_visible_at_version(1, 2, 15));
        // At/after delete: not visible (deleted at 20, version 20 sees the delete).
        assert!(!m.is_visible_at_version(1, 2, 20));
        assert!(!m.is_visible_at_version(1, 2, 25));
        // A live row (xmax 0) inserted at 10 is visible at any version >= 10.
        assert!(m.is_visible_at_version(1, 0, 10));
        assert!(m.is_visible_at_version(1, 0, 99));
    }

    #[test]
    fn test_version_reclaimable_respects_floor() {
        let m = TxnStatusMap::new();
        m.enable_lsn_tracking();
        m.record_committed_at(2, 20); // deleter committed at LSN 20
        // No floor: always reclaimable.
        assert!(m.version_reclaimable(2));
        // Floor at 15: deleter committed at 20 > 15, still visible at the floor.
        m.retain_version(15);
        assert!(!m.version_reclaimable(2));
        // Floor lowered to 25: deleter committed at 20 <= 25, reclaimable.
        m.retain_version(25);
        // fetch_min keeps the floor at 15, so still not reclaimable.
        assert_eq!(m.version_retention_floor(), 15);
        m.set_version_retention_floor(25);
        assert!(m.version_reclaimable(2));
    }

    #[test]
    fn advance_commit_lsn_dawn_respects_floor_and_is_settled() {
        let m = TxnStatusMap::new();
        m.enable_lsn_tracking();
        // A low-id transaction that committed late (high LSN) in segment 0, and a
        // higher-id transaction that committed early (low LSN) in segment 1: commit
        // LSN is deliberately out of transaction-id order.
        m.record_committed_at(5, 1000);
        m.record_committed_at(TXNS_PER_SEGMENT + 7, 200);
        // Status truncation has reclaimed aborts through segment 2, so the dawn
        // may advance up to there once the floor allows.
        m.truncate_below(TXNS_PER_SEGMENT * 3);

        // Floor below segment 0's max commit: the segment is kept and the late
        // transaction keeps its real commit LSN (never mis-dated as the dawn).
        assert_eq!(m.advance_commit_lsn_dawn(500), 0);
        assert_eq!(m.commit_lsn(5), Some(1000));
        assert_eq!(m.commit_lsn_dawn(), 0);

        // Floor at/above both segments' max commit: both are freed and their
        // transactions date as the dawn, which is now safe (committed <= floor).
        assert_eq!(m.advance_commit_lsn_dawn(1000), 2);
        assert_eq!(m.commit_lsn(5), Some(0));
        assert_eq!(m.commit_lsn(TXNS_PER_SEGMENT + 7), Some(0));
        assert_eq!(m.commit_lsn_dawn(), TXNS_PER_SEGMENT * 3);
    }

    #[test]
    fn advance_commit_lsn_dawn_capped_at_truncated_below() {
        let m = TxnStatusMap::new();
        m.enable_lsn_tracking();
        m.record_committed_at(5, 100);
        // No status truncation yet, so the dawn cannot advance even though the
        // floor is above the commit: aborts below it are not proven reclaimed.
        assert_eq!(m.advance_commit_lsn_dawn(u64::MAX), 0);
        assert_eq!(m.commit_lsn(5), Some(100));
    }

    #[test]
    fn test_persist_load_round_trips_commit_lsns() {
        let dir = tempfile::TempDir::new().unwrap();
        let m = TxnStatusMap::new();
        m.enable_lsn_tracking();
        m.record_committed_at(3, 300);
        m.record_committed_at(TXNS_PER_SEGMENT + 9, 900);
        m.persist(dir.path()).unwrap();

        let loaded = TxnStatusMap::new();
        loaded.load(dir.path()).unwrap();
        assert!(loaded.lsn_tracking_enabled());
        assert_eq!(loaded.commit_lsn(3), Some(300));
        assert_eq!(loaded.commit_lsn(TXNS_PER_SEGMENT + 9), Some(900));
    }

    #[test]
    fn test_cross_segment_ids() {
        let m = TxnStatusMap::new();
        let a = TXNS_PER_SEGMENT + 3;
        let b = TXNS_PER_SEGMENT * 2 + 100;
        m.record_committed(a);
        m.record_aborted(b);
        assert!(m.is_committed(a));
        assert!(m.is_aborted(b));
        assert_eq!(m.status(TXNS_PER_SEGMENT + 4), TxnStatus::Active);
    }

    #[test]
    fn test_truncate_frees_low_segments_and_reports_committed() {
        let m = TxnStatusMap::new();
        // Record statuses in segment 0 (below) and segment 2 (above the cut).
        m.record_committed(5);
        m.record_aborted(9);
        let high = TXNS_PER_SEGMENT * 2 + 7;
        m.record_aborted(high);

        // Truncate below segment 2: segments 0 and 1 are freed.
        let freed = m.truncate_below(TXNS_PER_SEGMENT * 2);
        assert_eq!(freed, 1); // only segment 0 was allocated

        // Below the watermark everything reports committed without the segment.
        assert!(m.is_committed(5));
        assert!(m.is_committed(9));
        assert!(!m.is_aborted(9));
        assert_eq!(m.status(5), TxnStatus::Committed);
        // Above the watermark the recorded status is intact.
        assert!(m.is_aborted(high));
        assert!(!m.is_committed(high));
    }

    #[test]
    fn test_truncate_below_segment_boundary_is_noop() {
        let m = TxnStatusMap::new();
        m.record_aborted(3);
        // A watermark inside segment 0 frees nothing (no whole segment below it).
        assert_eq!(m.truncate_below(100), 0);
        assert!(m.is_aborted(3));
    }

    #[test]
    fn test_persist_and_load() {
        let dir = tempfile::tempdir().unwrap();
        {
            let m = TxnStatusMap::new();
            m.record_committed(1);
            m.record_committed(2);
            m.record_aborted(3);
            m.record_committed(TXNS_PER_SEGMENT + 9);
            m.persist(dir.path()).unwrap();
        }
        let loaded = TxnStatusMap::new();
        loaded.load(dir.path()).unwrap();
        assert!(loaded.is_committed(1));
        assert!(loaded.is_committed(2));
        assert!(loaded.is_aborted(3));
        assert!(loaded.is_committed(TXNS_PER_SEGMENT + 9));
        assert_eq!(loaded.status(4), TxnStatus::Active);
    }
}

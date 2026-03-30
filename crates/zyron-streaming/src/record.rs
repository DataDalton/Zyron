//! StreamRecord micro-batch, ChangeFlag, ChangelogMode, and StreamRecordPool.
//!
//! A StreamRecord wraps a columnar StreamBatch with per-row event timestamps,
//! optional pre-computed key hashes, and per-row change flags. The
//! StreamRecordPool provides lock-free buffer reuse via a Treiber stack
//! to eliminate allocation on the hot path.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use crate::column::{STREAM_BATCH_SIZE, StreamBatch};

// ---------------------------------------------------------------------------
// ChangeFlag
// ---------------------------------------------------------------------------

/// Per-row change type for changelog (retract) streams.
/// +I = insert, -U = update before, +U = update after, -D = delete.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ChangeFlag {
    Insert = 0,
    UpdateBefore = 1,
    UpdateAfter = 2,
    Delete = 3,
}

impl ChangeFlag {
    /// Returns true if this flag represents a retraction (negative record).
    #[inline]
    pub fn is_retraction(&self) -> bool {
        matches!(self, ChangeFlag::UpdateBefore | ChangeFlag::Delete)
    }

    /// Serialize to a single byte.
    #[inline]
    pub fn to_byte(self) -> u8 {
        self as u8
    }

    /// Deserialize from a byte.
    #[inline]
    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            0 => Some(ChangeFlag::Insert),
            1 => Some(ChangeFlag::UpdateBefore),
            2 => Some(ChangeFlag::UpdateAfter),
            3 => Some(ChangeFlag::Delete),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// ChangelogMode
// ---------------------------------------------------------------------------

/// Determines whether a stream carries only inserts or full retractions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangelogMode {
    /// Stream contains only +I records (append-only).
    InsertOnly,
    /// Stream may contain +I, -U, +U, -D records.
    Retract,
}

// ---------------------------------------------------------------------------
// StreamRecord
// ---------------------------------------------------------------------------

/// A micro-batch of rows flowing through the streaming operator DAG.
/// Default batch size is 1024 rows, amortizing per-record overhead and
/// enabling vectorized operations on contiguous typed arrays.
#[derive(Debug, Clone)]
pub struct StreamRecord {
    /// Columnar micro-batch data.
    pub batch: StreamBatch,
    /// Per-row event timestamps in milliseconds.
    pub event_times: Vec<i64>,
    /// Pre-computed key hashes for joins and aggregations.
    /// None if key hashing has not been applied yet.
    pub keys: Option<Vec<u64>>,
    /// Per-row change flags for retract streams.
    pub change_flags: Vec<ChangeFlag>,
}

impl StreamRecord {
    /// Creates a new StreamRecord from a batch with event times and change flags.
    pub fn new(batch: StreamBatch, event_times: Vec<i64>, change_flags: Vec<ChangeFlag>) -> Self {
        debug_assert_eq!(batch.num_rows, event_times.len());
        debug_assert_eq!(batch.num_rows, change_flags.len());
        Self {
            batch,
            event_times,
            keys: None,
            change_flags,
        }
    }

    /// Creates an insert-only record (all rows flagged as Insert).
    pub fn new_insert(batch: StreamBatch, event_times: Vec<i64>) -> Self {
        let num_rows = batch.num_rows;
        debug_assert_eq!(num_rows, event_times.len());
        Self {
            batch,
            event_times,
            keys: None,
            change_flags: vec![ChangeFlag::Insert; num_rows],
        }
    }

    /// Creates an empty record used as a poison pill to signal shutdown.
    pub fn empty() -> Self {
        Self {
            batch: StreamBatch::empty(),
            event_times: Vec::new(),
            keys: None,
            change_flags: Vec::new(),
        }
    }

    /// Returns true if this record contains no rows.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.batch.is_empty()
    }

    /// Number of rows in this micro-batch.
    #[inline]
    pub fn num_rows(&self) -> usize {
        self.batch.num_rows
    }

    /// Filters rows by a boolean mask. Returns a new record containing only
    /// rows where mask[i] is true. Builds an index array once, then gathers
    /// all components using take/index operations.
    pub fn filter(&self, mask: &[bool]) -> Self {
        debug_assert_eq!(mask.len(), self.num_rows());

        // Build index array once, then gather all components.
        let indices: Vec<u32> = mask
            .iter()
            .enumerate()
            .filter(|&(_, &keep)| keep)
            .map(|(i, _)| i as u32)
            .collect();

        let batch = self.batch.take(&indices);
        let event_times: Vec<i64> = indices
            .iter()
            .map(|&i| self.event_times[i as usize])
            .collect();
        let keys = self
            .keys
            .as_ref()
            .map(|k| indices.iter().map(|&i| k[i as usize]).collect());
        let change_flags: Vec<ChangeFlag> = indices
            .iter()
            .map(|&i| self.change_flags[i as usize])
            .collect();

        Self {
            batch,
            event_times,
            keys,
            change_flags,
        }
    }

    /// Resets this record to an empty state for reuse from the pool.
    pub fn clear(&mut self) {
        self.batch = StreamBatch::empty();
        self.event_times.clear();
        self.keys = None;
        self.change_flags.clear();
    }
}

// ---------------------------------------------------------------------------
// StreamRecordPool: lock-free Treiber stack with ABA prevention
// ---------------------------------------------------------------------------

/// Sentinel value indicating end of stack (no next slot).
const POOL_NULL: u32 = u32::MAX;

/// Lock-free pool of pre-allocated StreamRecord buffers.
/// Uses an array-based Treiber stack with a generation counter packed
/// into the head to prevent the ABA problem. pop() returns a cleared
/// record, push() returns one for reuse.
pub struct StreamRecordPool {
    /// Slot storage. Each slot holds an Option<StreamRecord>.
    slots: Vec<parking_lot::Mutex<Option<StreamRecord>>>,
    /// Per-slot next pointer for the free-list chain.
    next: Vec<AtomicU32>,
    /// Stack head: upper 32 bits = generation, lower 32 bits = slot index.
    /// POOL_NULL in lower bits means empty stack.
    head: AtomicU64,
    pool_size: AtomicU64,
    max_pool_size: u64,
}

unsafe impl Send for StreamRecordPool {}
unsafe impl Sync for StreamRecordPool {}

impl StreamRecordPool {
    #[inline(always)]
    fn pack(generation: u32, index: u32) -> u64 {
        ((generation as u64) << 32) | (index as u64)
    }

    #[inline(always)]
    fn unpack(packed: u64) -> (u32, u32) {
        ((packed >> 32) as u32, packed as u32)
    }

    /// Creates a new pool with pre-allocated records.
    pub fn new(pre_allocate: usize) -> Self {
        let capacity = if pre_allocate == 0 {
            64
        } else {
            pre_allocate * 4
        };
        let mut slots = Vec::with_capacity(capacity);
        let mut next_vec = Vec::with_capacity(capacity);

        for i in 0..capacity {
            if i < pre_allocate {
                slots.push(parking_lot::Mutex::new(Some(StreamRecord {
                    batch: StreamBatch::empty(),
                    event_times: Vec::with_capacity(STREAM_BATCH_SIZE),
                    keys: None,
                    change_flags: Vec::with_capacity(STREAM_BATCH_SIZE),
                })));
            } else {
                slots.push(parking_lot::Mutex::new(None));
            }

            if i == 0 {
                next_vec.push(AtomicU32::new(POOL_NULL));
            } else {
                next_vec.push(AtomicU32::new((i - 1) as u32));
            }
        }

        // Stack top is the last pre-allocated slot, or POOL_NULL if none.
        let top = if pre_allocate > 0 {
            (pre_allocate - 1) as u32
        } else {
            POOL_NULL
        };

        Self {
            slots,
            next: next_vec,
            head: AtomicU64::new(Self::pack(0, top)),
            pool_size: AtomicU64::new(pre_allocate as u64),
            max_pool_size: capacity as u64,
        }
    }

    /// Takes a record from the pool. Returns a fresh empty record if the
    /// pool is exhausted.
    pub fn pop(&self) -> StreamRecord {
        loop {
            let current = self.head.load(Ordering::Acquire);
            let (generation, top) = Self::unpack(current);

            if top == POOL_NULL {
                return StreamRecord {
                    batch: StreamBatch::empty(),
                    event_times: Vec::with_capacity(STREAM_BATCH_SIZE),
                    keys: None,
                    change_flags: Vec::with_capacity(STREAM_BATCH_SIZE),
                };
            }

            let next_top = self.next[top as usize].load(Ordering::Acquire);
            let new_head = Self::pack(generation.wrapping_add(1), next_top);

            match self.head.compare_exchange_weak(
                current,
                new_head,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    self.pool_size.fetch_sub(1, Ordering::Relaxed);
                    let mut slot = self.slots[top as usize].lock();
                    let mut record = slot.take().unwrap_or_else(|| StreamRecord {
                        batch: StreamBatch::empty(),
                        event_times: Vec::with_capacity(STREAM_BATCH_SIZE),
                        keys: None,
                        change_flags: Vec::with_capacity(STREAM_BATCH_SIZE),
                    });
                    record.clear();
                    return record;
                }
                Err(_) => std::hint::spin_loop(),
            }
        }
    }

    /// Returns a record to the pool for reuse. If the pool exceeds
    /// max_pool_size the record is dropped instead of returned.
    pub fn push(&self, record: StreamRecord) {
        if self.pool_size.load(Ordering::Relaxed) >= self.max_pool_size {
            return;
        }

        // Find a free slot to store the record. Scan for an empty slot.
        let mut slot_idx = None;
        for (i, slot) in self.slots.iter().enumerate() {
            let guard = slot.lock();
            if guard.is_none() {
                drop(guard);
                slot_idx = Some(i);
                break;
            }
        }

        let idx = match slot_idx {
            Some(i) => i,
            None => return, // No free slots available, drop the record.
        };

        // Store the record in the slot.
        {
            let mut slot = self.slots[idx].lock();
            *slot = Some(record);
        }

        // Push the slot index onto the stack.
        loop {
            let current = self.head.load(Ordering::Acquire);
            let (generation, top) = Self::unpack(current);

            self.next[idx].store(top, Ordering::Release);
            let new_head = Self::pack(generation.wrapping_add(1), idx as u32);

            match self.head.compare_exchange_weak(
                current,
                new_head,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    self.pool_size.fetch_add(1, Ordering::Relaxed);
                    return;
                }
                Err(_) => std::hint::spin_loop(),
            }
        }
    }

    /// Number of records currently in the pool.
    pub fn available(&self) -> u64 {
        self.pool_size.load(Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::{StreamBatch, StreamColumn, StreamColumnData};

    fn make_test_record(n: usize) -> StreamRecord {
        let col = StreamColumn::from_data(StreamColumnData::Int64((0..n as i64).collect()));
        let batch = StreamBatch::new(vec![col]);
        let times: Vec<i64> = (0..n as i64).map(|i| i * 1000).collect();
        StreamRecord::new_insert(batch, times)
    }

    #[test]
    fn test_stream_record_new() {
        let record = make_test_record(5);
        assert_eq!(record.num_rows(), 5);
        assert!(!record.is_empty());
        assert!(record.keys.is_none());
        assert_eq!(record.change_flags.len(), 5);
        assert!(record.change_flags.iter().all(|f| *f == ChangeFlag::Insert));
    }

    #[test]
    fn test_stream_record_filter() {
        let record = make_test_record(4);
        let mask = [true, false, true, false];
        let filtered = record.filter(&mask);
        assert_eq!(filtered.num_rows(), 2);
        assert_eq!(filtered.event_times, vec![0, 2000]);
    }

    #[test]
    fn test_stream_record_empty() {
        let record = StreamRecord::empty();
        assert!(record.is_empty());
        assert_eq!(record.num_rows(), 0);
    }

    #[test]
    fn test_change_flag_serialization() {
        for flag in [
            ChangeFlag::Insert,
            ChangeFlag::UpdateBefore,
            ChangeFlag::UpdateAfter,
            ChangeFlag::Delete,
        ] {
            let b = flag.to_byte();
            let restored = ChangeFlag::from_byte(b);
            assert_eq!(restored, Some(flag));
        }
        assert_eq!(ChangeFlag::from_byte(255), None);
    }

    #[test]
    fn test_change_flag_retraction() {
        assert!(!ChangeFlag::Insert.is_retraction());
        assert!(ChangeFlag::UpdateBefore.is_retraction());
        assert!(!ChangeFlag::UpdateAfter.is_retraction());
        assert!(ChangeFlag::Delete.is_retraction());
    }

    #[test]
    fn test_record_pool_pop_push() {
        let pool = StreamRecordPool::new(4);
        assert_eq!(pool.available(), 4);

        let r1 = pool.pop();
        assert!(r1.is_empty());
        assert_eq!(pool.available(), 3);

        let r2 = pool.pop();
        assert_eq!(pool.available(), 2);
        pool.push(r1);
        assert_eq!(pool.available(), 3);
        pool.push(r2);
        assert_eq!(pool.available(), 4);
    }

    #[test]
    fn test_record_pool_empty_fallback() {
        let pool = StreamRecordPool::new(0);
        // Pool is empty, should allocate a fresh record.
        let record = pool.pop();
        assert!(record.is_empty());
    }

    #[test]
    fn test_changelog_mode() {
        assert_eq!(ChangelogMode::InsertOnly, ChangelogMode::InsertOnly);
        assert_ne!(ChangelogMode::InsertOnly, ChangelogMode::Retract);
    }
}

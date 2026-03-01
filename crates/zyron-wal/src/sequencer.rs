//! Lock-free LSN sequencer for concurrent WAL access.
//!
//! Uses atomic compare-and-swap operations to assign globally-ordered LSNs
//! without locks. Multiple writers can reserve LSN space concurrently.

use crate::record::Lsn;
use crate::segment::SegmentHeader;
use std::sync::atomic::{AtomicU64, Ordering};

/// Lock-free LSN sequencer.
///
/// Assigns monotonically increasing LSNs using atomic operations.
/// The LSN encodes both segment ID (upper 32 bits) and offset (lower 32 bits).
pub struct LsnSequencer {
    /// Packed LSN: upper 32 bits = segment_id, lower 32 bits = offset.
    next_lsn: AtomicU64,
    /// Maximum segment size in bytes.
    segment_size: u32,
}

impl LsnSequencer {
    /// Creates a new sequencer starting at the given segment and offset.
    pub fn new(segment_id: u32, initial_offset: u32, segment_size: u32) -> Self {
        let packed = ((segment_id as u64) << 32) | (initial_offset as u64);
        Self {
            next_lsn: AtomicU64::new(packed),
            segment_size,
        }
    }

    /// Reserves space for a record atomically.
    ///
    /// Returns `(assigned_lsn, needs_rotation)`:
    /// - `assigned_lsn`: The LSN where this record should be written
    /// - `needs_rotation`: True if segment is full and rotation is required
    ///
    /// If `needs_rotation` is true, the caller must coordinate segment rotation
    /// before retrying. The LSN returned in this case is the current position
    /// (not advanced).
    ///
    /// Uses fetch_add instead of CAS for the hot path. fetch_add always succeeds
    /// on the first attempt, eliminating the retry loop and the Acquire load.
    /// Rotation detection uses the post-add offset: if the new offset exceeds the
    /// segment boundary, the record_size is subtracted back (another fetch_add)
    /// and rotation is signaled.
    #[inline]
    pub fn reserve(&self, record_size: u32) -> (Lsn, bool) {
        let prev = self
            .next_lsn
            .fetch_add(record_size as u64, Ordering::Relaxed);
        let segment_id = (prev >> 32) as u32;
        let offset = prev as u32;

        if offset + record_size > self.segment_size {
            // Record does not fit. Undo the advance so the offset stays at the
            // segment boundary for the rotation path.
            self.next_lsn
                .fetch_sub(record_size as u64, Ordering::Relaxed);
            return (Lsn::new(segment_id, offset), true);
        }

        (Lsn::new(segment_id, offset), false)
    }

    /// Advances to the next segment.
    ///
    /// Called after segment rotation completes. Sets the offset to the
    /// initial position after the segment header.
    pub fn advance_segment(&self, new_segment_id: u32) {
        let initial_offset = SegmentHeader::SIZE as u32;
        let new_packed = ((new_segment_id as u64) << 32) | (initial_offset as u64);
        self.next_lsn.store(new_packed, Ordering::Release);
    }

    /// Returns the current LSN position without advancing.
    #[inline]
    pub fn current(&self) -> Lsn {
        let packed = self.next_lsn.load(Ordering::Acquire);
        let segment_id = (packed >> 32) as u32;
        let offset = packed as u32;
        Lsn::new(segment_id, offset)
    }

    /// Returns the current segment ID.
    #[inline]
    pub fn current_segment_id(&self) -> u32 {
        let packed = self.next_lsn.load(Ordering::Acquire);
        (packed >> 32) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_sequencer_basic() {
        let seq = LsnSequencer::new(0, 64, 1024);

        let (lsn, needs_rotation) = seq.reserve(32);
        assert!(!needs_rotation);
        assert_eq!(lsn.segment_id(), 0);
        assert_eq!(lsn.offset(), 64);

        let (lsn2, needs_rotation) = seq.reserve(32);
        assert!(!needs_rotation);
        assert_eq!(lsn2.segment_id(), 0);
        assert_eq!(lsn2.offset(), 96);
    }

    #[test]
    fn test_sequencer_rotation() {
        let seq = LsnSequencer::new(0, 64, 128);

        // First record fits
        let (_, needs_rotation) = seq.reserve(32);
        assert!(!needs_rotation);

        // Second record fits
        let (_, needs_rotation) = seq.reserve(32);
        assert!(!needs_rotation);

        // Third record would exceed segment size
        let (lsn, needs_rotation) = seq.reserve(64);
        assert!(needs_rotation);
        assert_eq!(lsn.offset(), 128); // Position not advanced
    }

    #[test]
    fn test_sequencer_advance_segment() {
        let seq = LsnSequencer::new(0, 64, 128);
        seq.reserve(64).0;

        seq.advance_segment(1);

        let current = seq.current();
        assert_eq!(current.segment_id(), 1);
        assert_eq!(current.offset(), SegmentHeader::SIZE as u32);
    }

    #[test]
    fn test_sequencer_concurrent() {
        // 8 threads * 10_000 records * 32 bytes = 2,560,000 bytes
        // Segment must be large enough to fit all records
        let seq = Arc::new(LsnSequencer::new(0, 64, 4_000_000));
        let record_size = 32;
        let threads = 8;
        let records_per_thread = 10_000;

        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let seq = Arc::clone(&seq);
                thread::spawn(move || {
                    let mut lsns = Vec::with_capacity(records_per_thread);
                    for _ in 0..records_per_thread {
                        let (lsn, _) = seq.reserve(record_size);
                        lsns.push(lsn);
                    }
                    lsns
                })
            })
            .collect();

        let mut all_lsns: Vec<Lsn> = handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect();

        // Verify all LSNs are unique
        all_lsns.sort();
        for i in 1..all_lsns.len() {
            assert_ne!(all_lsns[i], all_lsns[i - 1], "duplicate LSN detected");
        }

        // Verify correct count
        assert_eq!(all_lsns.len(), threads * records_per_thread);
    }
}

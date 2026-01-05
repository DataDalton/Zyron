//! Page replacement policies for the buffer pool.

use crate::frame::FrameId;
use parking_lot::Mutex;
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};

/// Trait for page replacement algorithms.
pub trait Replacer: Send + Sync {
    /// Records that the given frame was accessed.
    fn record_access(&self, frame_id: FrameId);

    /// Marks a frame as evictable (unpinned).
    fn set_evictable(&self, frame_id: FrameId, evictable: bool);

    /// Combined operation: records access and pins the frame (sets non-evictable).
    /// Single lock acquisition instead of two separate calls.
    fn access_and_pin(&self, frame_id: FrameId);

    /// Selects a victim frame for eviction.
    ///
    /// Returns None if no frames are evictable.
    fn evict(&self) -> Option<FrameId>;

    /// Removes a frame from the replacer.
    fn remove(&self, frame_id: FrameId);

    /// Returns the number of evictable frames.
    fn size(&self) -> usize;
}

/// Clock replacement algorithm implementation.
///
/// Uses atomic reference bits for lock-free access recording.
/// Only takes the mutex lock for evictable set modifications.
pub struct ClockReplacer {
    /// Number of frames.
    num_frames: usize,
    /// Reference bits for each frame (atomic for lock-free access).
    reference_bits: Vec<AtomicBool>,
    /// Internal state protected by mutex (evictable set and clock hand).
    inner: Mutex<ClockReplacerInner>,
}

struct ClockReplacerInner {
    /// Set of evictable frame IDs.
    evictable: HashSet<FrameId>,
    /// Current clock hand position.
    clock_hand: usize,
}

impl ClockReplacer {
    /// Creates a new clock replacer with the given number of frames.
    pub fn new(num_frames: usize) -> Self {
        let reference_bits: Vec<AtomicBool> = (0..num_frames)
            .map(|_| AtomicBool::new(false))
            .collect();

        Self {
            num_frames,
            reference_bits,
            inner: Mutex::new(ClockReplacerInner {
                evictable: HashSet::new(),
                clock_hand: 0,
            }),
        }
    }

    /// Returns the total capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.num_frames
    }
}

impl Replacer for ClockReplacer {
    #[inline]
    fn record_access(&self, frame_id: FrameId) {
        let idx = frame_id.0 as usize;
        if idx < self.num_frames {
            // Lock-free atomic write
            self.reference_bits[idx].store(true, Ordering::Relaxed);
        }
    }

    #[inline]
    fn set_evictable(&self, frame_id: FrameId, evictable: bool) {
        if (frame_id.0 as usize) >= self.num_frames {
            return;
        }

        let mut inner = self.inner.lock();
        if evictable {
            inner.evictable.insert(frame_id);
        } else {
            inner.evictable.remove(&frame_id);
        }
    }

    #[inline]
    fn access_and_pin(&self, frame_id: FrameId) {
        let idx = frame_id.0 as usize;
        if idx < self.num_frames {
            // Lock-free atomic write for reference bit
            self.reference_bits[idx].store(true, Ordering::Relaxed);
            // Only take lock for evictable set
            self.inner.lock().evictable.remove(&frame_id);
        }
    }

    fn evict(&self) -> Option<FrameId> {
        let mut inner = self.inner.lock();

        if inner.evictable.is_empty() {
            return None;
        }

        let num_frames = self.num_frames;

        // Make at most 2 full rotations to find a victim
        for _ in 0..(2 * num_frames) {
            let hand = inner.clock_hand;
            let frame_id = FrameId(hand as u32);

            if inner.evictable.contains(&frame_id) {
                if !self.reference_bits[hand].load(Ordering::Relaxed) {
                    // Found victim: evictable and reference bit is 0
                    inner.evictable.remove(&frame_id);
                    inner.clock_hand = (hand + 1) % num_frames;
                    return Some(frame_id);
                } else {
                    // Clear reference bit and continue
                    self.reference_bits[hand].store(false, Ordering::Relaxed);
                }
            }

            inner.clock_hand = (hand + 1) % num_frames;

        }

        // If we still haven't found one, just pick any evictable frame
        if let Some(&frame_id) = inner.evictable.iter().next() {
            inner.evictable.remove(&frame_id);
            return Some(frame_id);
        }

        None
    }

    fn remove(&self, frame_id: FrameId) {
        let idx = frame_id.0 as usize;
        if idx < self.num_frames {
            self.inner.lock().evictable.remove(&frame_id);
            self.reference_bits[idx].store(false, Ordering::Relaxed);
        }
    }

    fn size(&self) -> usize {
        self.inner.lock().evictable.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clock_replacer_new() {
        let replacer = ClockReplacer::new(10);
        assert_eq!(replacer.capacity(), 10);
        assert_eq!(replacer.size(), 0);
    }

    #[test]
    fn test_clock_replacer_set_evictable() {
        let replacer = ClockReplacer::new(10);

        replacer.set_evictable(FrameId(0), true);
        replacer.set_evictable(FrameId(1), true);
        replacer.set_evictable(FrameId(2), true);

        assert_eq!(replacer.size(), 3);

        replacer.set_evictable(FrameId(1), false);
        assert_eq!(replacer.size(), 2);
    }

    #[test]
    fn test_clock_replacer_evict_empty() {
        let replacer = ClockReplacer::new(10);
        assert!(replacer.evict().is_none());
    }

    #[test]
    fn test_clock_replacer_evict_single() {
        let replacer = ClockReplacer::new(10);

        replacer.set_evictable(FrameId(5), true);
        assert_eq!(replacer.size(), 1);

        let victim = replacer.evict();
        assert_eq!(victim, Some(FrameId(5)));
        assert_eq!(replacer.size(), 0);
    }

    #[test]
    fn test_clock_replacer_evict_with_reference_bits() {
        let replacer = ClockReplacer::new(10);

        // Add evictable frames
        replacer.set_evictable(FrameId(0), true);
        replacer.set_evictable(FrameId(1), true);
        replacer.set_evictable(FrameId(2), true);

        // Set reference bits on frames 0 and 1
        replacer.record_access(FrameId(0));
        replacer.record_access(FrameId(1));

        // Frame 2 should be evicted first (no reference bit)
        let victim = replacer.evict();
        assert_eq!(victim, Some(FrameId(2)));
    }

    #[test]
    fn test_clock_replacer_evict_all_referenced() {
        let replacer = ClockReplacer::new(3);

        replacer.set_evictable(FrameId(0), true);
        replacer.set_evictable(FrameId(1), true);
        replacer.set_evictable(FrameId(2), true);

        // Reference all frames
        replacer.record_access(FrameId(0));
        replacer.record_access(FrameId(1));
        replacer.record_access(FrameId(2));

        // Should still be able to evict (after clearing reference bits)
        let victim = replacer.evict();
        assert!(victim.is_some());
        assert_eq!(replacer.size(), 2);
    }

    #[test]
    fn test_clock_replacer_remove() {
        let replacer = ClockReplacer::new(10);

        replacer.set_evictable(FrameId(0), true);
        replacer.set_evictable(FrameId(1), true);
        assert_eq!(replacer.size(), 2);

        replacer.remove(FrameId(0));
        assert_eq!(replacer.size(), 1);

        let victim = replacer.evict();
        assert_eq!(victim, Some(FrameId(1)));
    }

    #[test]
    fn test_clock_replacer_record_access() {
        let replacer = ClockReplacer::new(10);

        replacer.set_evictable(FrameId(0), true);
        replacer.set_evictable(FrameId(1), true);

        // Access frame 0, giving it a second chance
        replacer.record_access(FrameId(0));

        // Frame 1 should be evicted (frame 0 has reference bit set)
        let victim = replacer.evict();
        assert_eq!(victim, Some(FrameId(1)));
    }

    #[test]
    fn test_clock_replacer_out_of_bounds() {
        let replacer = ClockReplacer::new(5);

        // These should not panic
        replacer.set_evictable(FrameId(100), true);
        replacer.record_access(FrameId(100));
        replacer.remove(FrameId(100));

        assert_eq!(replacer.size(), 0);
    }

    #[test]
    fn test_clock_replacer_fifo_like_behavior() {
        let replacer = ClockReplacer::new(5);

        // Add frames in order without accessing them
        for i in 0..5 {
            replacer.set_evictable(FrameId(i), true);
        }

        // Should evict in roughly FIFO order when no reference bits are set
        let v1 = replacer.evict();
        let v2 = replacer.evict();
        let v3 = replacer.evict();

        // All should be valid evictions
        assert!(v1.is_some());
        assert!(v2.is_some());
        assert!(v3.is_some());
        assert_eq!(replacer.size(), 2);
    }

    #[test]
    fn test_clock_replacer_pin_unpin_cycle() {
        let replacer = ClockReplacer::new(3);

        // Initially all evictable
        replacer.set_evictable(FrameId(0), true);
        replacer.set_evictable(FrameId(1), true);
        replacer.set_evictable(FrameId(2), true);
        assert_eq!(replacer.size(), 3);

        // Pin frame 1 (not evictable)
        replacer.set_evictable(FrameId(1), false);
        assert_eq!(replacer.size(), 2);

        // Evict should skip frame 1
        let victim = replacer.evict();
        assert!(victim.is_some());
        assert_ne!(victim, Some(FrameId(1)));

        // Unpin frame 1
        replacer.set_evictable(FrameId(1), true);
        assert_eq!(replacer.size(), 2); // One was evicted, one re-added
    }
}

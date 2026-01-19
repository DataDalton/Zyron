//! Page replacement policies for the buffer pool.

use crate::frame::FrameId;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

/// Trait for page replacement algorithms.
pub trait Replacer: Send + Sync {
    /// Records that the given frame was accessed (sets reference bit).
    fn record_access(&self, frame_id: FrameId);

    /// Selects a victim frame for eviction.
    /// The is_evictable closure checks if a frame can be evicted (pin_count == 0).
    /// Returns None if no frames are evictable.
    fn evict<F>(&self, is_evictable: F) -> Option<FrameId>
    where
        F: Fn(FrameId) -> bool;

    /// Removes a frame from the replacer (clears reference bit).
    fn remove(&self, frame_id: FrameId);
}

/// Clock replacement algorithm implementation.
///
/// Uses atomic reference bits for the clock algorithm.
/// Evictability is determined by checking pin_count directly during eviction,
/// eliminating redundant state tracking.
pub struct ClockReplacer {
    /// Number of frames.
    num_frames: usize,
    /// Reference bits for each frame (atomic for lock-free access).
    reference_bits: Vec<AtomicBool>,
    /// Clock hand position (protected by mutex during eviction).
    clock_hand: Mutex<usize>,
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
            clock_hand: Mutex::new(0),
        }
    }

    /// Returns the total capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.num_frames
    }
}

impl Replacer for ClockReplacer {
    #[inline(always)]
    fn record_access(&self, frame_id: FrameId) {
        let idx = frame_id.0 as usize;
        if idx < self.num_frames {
            self.reference_bits[idx].store(true, Ordering::Relaxed);
        }
    }

    fn evict<F>(&self, is_evictable: F) -> Option<FrameId>
    where
        F: Fn(FrameId) -> bool,
    {
        let mut hand = self.clock_hand.lock();
        let num_frames = self.num_frames;

        // Clock sweep: 2 full rotations to find a victim
        for _ in 0..(2 * num_frames) {
            let idx = *hand;
            *hand = (idx + 1) % num_frames;
            let frame_id = FrameId(idx as u32);

            // Check if frame is evictable (pin_count == 0)
            if is_evictable(frame_id) {
                if !self.reference_bits[idx].load(Ordering::Relaxed) {
                    // Found victim: evictable and reference bit is 0
                    return Some(frame_id);
                } else {
                    // Clear reference bit, give second chance
                    self.reference_bits[idx].store(false, Ordering::Relaxed);
                }
            }
        }

        None
    }

    #[inline]
    fn remove(&self, frame_id: FrameId) {
        let idx = frame_id.0 as usize;
        if idx < self.num_frames {
            self.reference_bits[idx].store(false, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clock_replacer_new() {
        let replacer = ClockReplacer::new(10);
        assert_eq!(replacer.capacity(), 10);
    }

    #[test]
    fn test_clock_replacer_evict_empty() {
        let replacer = ClockReplacer::new(10);
        // No frames are evictable
        assert!(replacer.evict(|_| false).is_none());
    }

    #[test]
    fn test_clock_replacer_evict_single() {
        let replacer = ClockReplacer::new(10);

        // Only frame 5 is evictable
        let victim = replacer.evict(|fid| fid.0 == 5);
        assert_eq!(victim, Some(FrameId(5)));
    }

    #[test]
    fn test_clock_replacer_evict_with_reference_bits() {
        let replacer = ClockReplacer::new(10);

        // Set reference bits on frames 0 and 1
        replacer.record_access(FrameId(0));
        replacer.record_access(FrameId(1));

        // Frames 0, 1, 2 are evictable, but 0 and 1 have reference bits
        // Frame 2 should be evicted first
        let victim = replacer.evict(|fid| fid.0 <= 2);
        assert_eq!(victim, Some(FrameId(2)));
    }

    #[test]
    fn test_clock_replacer_evict_all_referenced() {
        let replacer = ClockReplacer::new(3);

        // Reference all frames
        replacer.record_access(FrameId(0));
        replacer.record_access(FrameId(1));
        replacer.record_access(FrameId(2));

        // All evictable - should clear reference bits and evict one
        let victim = replacer.evict(|_| true);
        assert!(victim.is_some());
    }

    #[test]
    fn test_clock_replacer_remove() {
        let replacer = ClockReplacer::new(10);

        replacer.record_access(FrameId(0));
        replacer.remove(FrameId(0));

        // Frame 0 should be evicted immediately (reference bit cleared)
        let victim = replacer.evict(|fid| fid.0 == 0);
        assert_eq!(victim, Some(FrameId(0)));
    }

    #[test]
    fn test_clock_replacer_record_access() {
        let replacer = ClockReplacer::new(10);

        // Frame 0 and 1 are evictable
        // Access frame 0, giving it a second chance
        replacer.record_access(FrameId(0));

        // Frame 1 should be evicted (frame 0 has reference bit set)
        let victim = replacer.evict(|fid| fid.0 <= 1);
        assert_eq!(victim, Some(FrameId(1)));
    }

    #[test]
    fn test_clock_replacer_out_of_bounds() {
        let replacer = ClockReplacer::new(5);

        // These should not panic
        replacer.record_access(FrameId(100));
        replacer.remove(FrameId(100));
    }

    #[test]
    fn test_clock_replacer_fifo_like_behavior() {
        let replacer = ClockReplacer::new(5);

        // All frames evictable, no reference bits
        // Should evict in order starting from clock hand
        let v1 = replacer.evict(|_| true);
        let v2 = replacer.evict(|_| true);
        let v3 = replacer.evict(|_| true);

        assert_eq!(v1, Some(FrameId(0)));
        assert_eq!(v2, Some(FrameId(1)));
        assert_eq!(v3, Some(FrameId(2)));
    }

    #[test]
    fn test_clock_replacer_pin_unpin_simulation() {
        let replacer = ClockReplacer::new(3);

        // Simulate: frames 0, 1, 2 exist, frame 1 is pinned
        let is_evictable = |fid: FrameId| fid.0 != 1;

        // Should skip frame 1
        let victim = replacer.evict(is_evictable);
        assert!(victim.is_some());
        assert_ne!(victim, Some(FrameId(1)));
    }
}

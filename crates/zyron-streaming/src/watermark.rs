//! Streaming watermark generation and tracking.
//!
//! Watermarks represent event-time progress and indicate that the system
//! believes all events up to the watermark timestamp have arrived. Used
//! to trigger window closure and late data detection.

use std::sync::atomic::{AtomicI64, Ordering};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Watermark
// ---------------------------------------------------------------------------

/// A streaming watermark representing event-time progress.
/// All events with timestamp <= watermark.timestamp_ms are considered arrived.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Watermark {
    pub timestamp_ms: i64,
}

impl Watermark {
    pub const MIN: Watermark = Watermark {
        timestamp_ms: i64::MIN,
    };

    #[inline]
    pub fn new(timestamp_ms: i64) -> Self {
        Self { timestamp_ms }
    }
}

// ---------------------------------------------------------------------------
// WatermarkGenerator trait
// ---------------------------------------------------------------------------

/// Generates watermarks from observed event timestamps.
pub trait WatermarkGenerator: Send + Sync {
    /// Called for each event to update internal state.
    fn on_event(&mut self, event_time_ms: i64);

    /// Returns the current watermark based on observed events.
    fn current_watermark(&self) -> Watermark;
}

// ---------------------------------------------------------------------------
// BoundedOutOfOrderWatermark
// ---------------------------------------------------------------------------

/// Bounded out-of-order watermark generator.
/// Watermark = max(observed_timestamps) - max_out_of_orderness.
/// SQL: WATERMARK FOR event_time AS event_time - INTERVAL '5 minutes'
pub struct BoundedOutOfOrderWatermark {
    max_observed_ms: i64,
    max_out_of_orderness_ms: i64,
}

impl BoundedOutOfOrderWatermark {
    pub fn new(max_out_of_orderness_ms: i64) -> Self {
        Self {
            max_observed_ms: i64::MIN,
            max_out_of_orderness_ms,
        }
    }
}

impl WatermarkGenerator for BoundedOutOfOrderWatermark {
    #[inline]
    fn on_event(&mut self, event_time_ms: i64) {
        if event_time_ms > self.max_observed_ms {
            self.max_observed_ms = event_time_ms;
        }
    }

    fn current_watermark(&self) -> Watermark {
        if self.max_observed_ms == i64::MIN {
            Watermark::MIN
        } else {
            Watermark::new(self.max_observed_ms - self.max_out_of_orderness_ms)
        }
    }
}

// ---------------------------------------------------------------------------
// PeriodicWatermark
// ---------------------------------------------------------------------------

/// Emits watermarks at fixed event count intervals.
/// Wraps an inner WatermarkGenerator and only updates the emitted
/// watermark every N events.
pub struct PeriodicWatermark {
    inner: Box<dyn WatermarkGenerator>,
    emit_interval: u64,
    event_count: u64,
    last_emitted: Watermark,
}

impl PeriodicWatermark {
    pub fn new(inner: Box<dyn WatermarkGenerator>, emit_interval: u64) -> Self {
        Self {
            inner,
            emit_interval: emit_interval.max(1),
            event_count: 0,
            last_emitted: Watermark::MIN,
        }
    }
}

impl WatermarkGenerator for PeriodicWatermark {
    fn on_event(&mut self, event_time_ms: i64) {
        self.inner.on_event(event_time_ms);
        self.event_count += 1;
        if self.event_count % self.emit_interval == 0 {
            self.last_emitted = self.inner.current_watermark();
        }
    }

    fn current_watermark(&self) -> Watermark {
        self.last_emitted
    }
}

// ---------------------------------------------------------------------------
// IdleSourceWatermark
// ---------------------------------------------------------------------------

/// Advances watermark when a source is idle (no events for a timeout period).
/// This prevents idle sources from holding back the global watermark.
pub struct IdleSourceWatermark {
    inner: Box<dyn WatermarkGenerator>,
    idle_timeout_ms: u64,
    last_event_wall_clock: Instant,
    last_event_time_ms: i64,
    idle_advance_ms: i64,
}

impl IdleSourceWatermark {
    pub fn new(inner: Box<dyn WatermarkGenerator>, idle_timeout_ms: u64) -> Self {
        Self {
            inner,
            idle_timeout_ms,
            last_event_wall_clock: Instant::now(),
            last_event_time_ms: i64::MIN,
            idle_advance_ms: 0,
        }
    }

    /// Call periodically (e.g., from a timer) to check for idle sources.
    pub fn check_idle(&mut self) {
        let elapsed = self.last_event_wall_clock.elapsed().as_millis() as u64;
        if elapsed > self.idle_timeout_ms && self.last_event_time_ms != i64::MIN {
            self.idle_advance_ms = (elapsed - self.idle_timeout_ms) as i64;
        }
    }
}

impl WatermarkGenerator for IdleSourceWatermark {
    fn on_event(&mut self, event_time_ms: i64) {
        self.inner.on_event(event_time_ms);
        self.last_event_wall_clock = Instant::now();
        self.last_event_time_ms = event_time_ms;
        self.idle_advance_ms = 0;
    }

    fn current_watermark(&self) -> Watermark {
        let base = self.inner.current_watermark();
        if self.idle_advance_ms > 0 {
            Watermark::new(base.timestamp_ms + self.idle_advance_ms)
        } else {
            base
        }
    }
}

// ---------------------------------------------------------------------------
// StreamWatermarkTracker: per-source atomic watermark tracking
// ---------------------------------------------------------------------------

/// Tracks watermarks from multiple parallel sources.
/// Uses a fixed-size AtomicI64 array (one per source) with lock-free reads.
/// The global minimum is recomputed by scanning the array.
pub struct StreamWatermarkTracker {
    /// Per-source watermark. Index = source_id (0-based).
    source_watermarks: Vec<AtomicI64>,
    /// Number of active sources.
    source_count: usize,
}

impl StreamWatermarkTracker {
    /// Creates a tracker for the given number of sources.
    /// All sources start at i64::MIN (no watermark).
    pub fn new(source_count: usize) -> Self {
        let mut source_watermarks = Vec::with_capacity(source_count);
        for _ in 0..source_count {
            source_watermarks.push(AtomicI64::new(i64::MIN));
        }
        Self {
            source_watermarks,
            source_count,
        }
    }

    /// Updates the watermark for a specific source.
    /// Only advances (never decreases).
    #[inline]
    pub fn advance(&self, source_id: usize, timestamp_ms: i64) {
        debug_assert!(source_id < self.source_count);
        // fetch_max ensures monotonic advance.
        self.source_watermarks[source_id].fetch_max(timestamp_ms, Ordering::Relaxed);
    }

    /// Returns the combined minimum watermark across all sources.
    /// This is the global watermark: events up to this time are considered complete.
    pub fn combined_watermark(&self) -> Watermark {
        let mut min = i64::MAX;
        for wm in &self.source_watermarks {
            let v = wm.load(Ordering::Relaxed);
            if v < min {
                min = v;
            }
        }
        if min == i64::MAX {
            Watermark::MIN
        } else {
            Watermark::new(min)
        }
    }

    /// Returns the watermark for a specific source.
    pub fn source_watermark(&self, source_id: usize) -> Watermark {
        Watermark::new(self.source_watermarks[source_id].load(Ordering::Relaxed))
    }

    pub fn source_count(&self) -> usize {
        self.source_count
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounded_out_of_order() {
        let mut wm_gen = BoundedOutOfOrderWatermark::new(5000); // 5s lateness
        assert_eq!(wm_gen.current_watermark(), Watermark::MIN);

        wm_gen.on_event(10_000);
        assert_eq!(wm_gen.current_watermark(), Watermark::new(5_000));

        wm_gen.on_event(15_000);
        assert_eq!(wm_gen.current_watermark(), Watermark::new(10_000));

        // Out-of-order event should not decrease watermark.
        wm_gen.on_event(8_000);
        assert_eq!(wm_gen.current_watermark(), Watermark::new(10_000));
    }

    #[test]
    fn test_periodic_watermark() {
        let inner = Box::new(BoundedOutOfOrderWatermark::new(0));
        let mut wm_gen = PeriodicWatermark::new(inner, 3);

        wm_gen.on_event(100);
        assert_eq!(wm_gen.current_watermark(), Watermark::MIN);

        wm_gen.on_event(200);
        assert_eq!(wm_gen.current_watermark(), Watermark::MIN);

        // Third event triggers emission.
        wm_gen.on_event(300);
        assert_eq!(wm_gen.current_watermark(), Watermark::new(300));
    }

    #[test]
    fn test_stream_watermark_tracker() {
        let tracker = StreamWatermarkTracker::new(3);

        tracker.advance(0, 100);
        tracker.advance(1, 200);
        tracker.advance(2, 150);

        // Global min should be source 0's watermark.
        assert_eq!(tracker.combined_watermark(), Watermark::new(100));

        // Advance source 0.
        tracker.advance(0, 300);
        assert_eq!(tracker.combined_watermark(), Watermark::new(150));
    }

    #[test]
    fn test_watermark_monotonic_advance() {
        let tracker = StreamWatermarkTracker::new(1);
        tracker.advance(0, 100);
        tracker.advance(0, 50); // Should not decrease.
        assert_eq!(tracker.source_watermark(0), Watermark::new(100));
    }

    #[test]
    fn test_watermark_ordering() {
        let a = Watermark::new(100);
        let b = Watermark::new(200);
        assert!(a < b);
        assert!(b > a);
        assert_eq!(a, Watermark::new(100));
    }

    #[test]
    fn test_tracker_uninitialized() {
        let tracker = StreamWatermarkTracker::new(2);
        // All sources at MIN, combined should be MIN.
        assert_eq!(tracker.combined_watermark(), Watermark::MIN);
    }
}

//! Window types and assignment for streaming aggregation.
//!
//! Provides tumbling, sliding, session, and cumulative window assigners,
//! a session window merger using union-find with path compression,
//! and SQL-compatible window accessor functions.

use zyron_common::Result;

// ---------------------------------------------------------------------------
// WindowRange
// ---------------------------------------------------------------------------

/// Half-open time range [start_ms, end_ms) for a window instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WindowRange {
    pub start_ms: i64,
    pub end_ms: i64,
}

impl WindowRange {
    #[inline]
    pub fn new(start_ms: i64, end_ms: i64) -> Self {
        Self { start_ms, end_ms }
    }

    /// Duration of the window in milliseconds.
    #[inline]
    pub fn duration_ms(&self) -> i64 {
        self.end_ms - self.start_ms
    }

    /// Returns true if this window contains the given timestamp.
    #[inline]
    pub fn contains(&self, timestamp_ms: i64) -> bool {
        timestamp_ms >= self.start_ms && timestamp_ms < self.end_ms
    }

    /// Returns true if two windows overlap.
    #[inline]
    pub fn overlaps(&self, other: &WindowRange) -> bool {
        self.start_ms < other.end_ms && other.start_ms < self.end_ms
    }

    /// Merges two overlapping or adjacent windows.
    #[inline]
    pub fn merge(&self, other: &WindowRange) -> WindowRange {
        WindowRange {
            start_ms: self.start_ms.min(other.start_ms),
            end_ms: self.end_ms.max(other.end_ms),
        }
    }
}

// ---------------------------------------------------------------------------
// WindowType
// ---------------------------------------------------------------------------

/// The four supported streaming window types.
#[derive(Debug, Clone, PartialEq)]
pub enum WindowType {
    /// Fixed-size, non-overlapping windows.
    Tumbling { size_ms: i64 },
    /// Overlapping windows with configurable slide interval.
    Sliding { size_ms: i64, slide_ms: i64 },
    /// Activity-based windows that merge on proximity.
    Session { gap_ms: i64 },
    /// Running totals within fixed periods at step intervals.
    Cumulative { max_size_ms: i64, step_ms: i64 },
}

// ---------------------------------------------------------------------------
// WindowAssigner trait
// ---------------------------------------------------------------------------

/// Assigns events to windows based on their event time.
pub trait WindowAssigner: Send + Sync {
    /// Returns the window(s) an event at the given time belongs to.
    /// For tumbling windows this returns exactly 1 window.
    /// For sliding windows this returns ceil(size/slide) windows.
    fn assign_windows(&self, event_time_ms: i64) -> Vec<WindowRange>;

    /// Assigns windows, writing results into the provided buffer.
    /// Returns the number of windows assigned. Clears the buffer first.
    /// Default implementation calls assign_windows() and copies results.
    fn assign_windows_into(&self, event_time_ms: i64, buf: &mut Vec<WindowRange>) -> usize {
        let windows = self.assign_windows(event_time_ms);
        let n = windows.len();
        buf.clear();
        buf.extend_from_slice(&windows);
        n
    }

    /// Maximum possible duration of any window produced by this assigner.
    fn max_window_duration_ms(&self) -> i64;
}

// ---------------------------------------------------------------------------
// TumblingWindowAssigner
// ---------------------------------------------------------------------------

/// Fixed-size, non-overlapping windows. Assignment is a single integer division.
pub struct TumblingWindowAssigner {
    pub size_ms: i64,
}

impl TumblingWindowAssigner {
    pub fn new(size_ms: i64) -> Self {
        assert!(size_ms > 0, "window size must be positive");
        Self { size_ms }
    }
}

impl WindowAssigner for TumblingWindowAssigner {
    #[inline(always)]
    fn assign_windows(&self, event_time_ms: i64) -> Vec<WindowRange> {
        let start = event_time_ms - event_time_ms.rem_euclid(self.size_ms);
        vec![WindowRange::new(start, start + self.size_ms)]
    }

    #[inline]
    fn assign_windows_into(&self, event_time_ms: i64, buf: &mut Vec<WindowRange>) -> usize {
        buf.clear();
        let start = event_time_ms - event_time_ms.rem_euclid(self.size_ms);
        buf.push(WindowRange::new(start, start + self.size_ms));
        1
    }

    fn max_window_duration_ms(&self) -> i64 {
        self.size_ms
    }
}

// ---------------------------------------------------------------------------
// SlidingWindowAssigner
// ---------------------------------------------------------------------------

/// Overlapping windows with configurable slide interval.
pub struct SlidingWindowAssigner {
    pub size_ms: i64,
    pub slide_ms: i64,
}

impl SlidingWindowAssigner {
    pub fn new(size_ms: i64, slide_ms: i64) -> Self {
        assert!(size_ms > 0, "window size must be positive");
        assert!(slide_ms > 0, "slide interval must be positive");
        Self { size_ms, slide_ms }
    }
}

impl WindowAssigner for SlidingWindowAssigner {
    #[inline(always)]
    fn assign_windows(&self, event_time_ms: i64) -> Vec<WindowRange> {
        // Walk forward from the earliest window that contains this event.
        // Earliest start = last_start - (windows_per_event - 1) * slide.
        // But clamped to >= 0.
        let last_start = event_time_ms - event_time_ms.rem_euclid(self.slide_ms);

        // Find the earliest window start that still contains event_time.
        // A window [s, s+size) contains event_time when s + size > event_time,
        // i.e. s > event_time - size.
        let earliest = (event_time_ms - self.size_ms + 1).max(0);
        // Align earliest up to a slide boundary.
        let first_start =
            earliest + (self.slide_ms - earliest.rem_euclid(self.slide_ms)) % self.slide_ms;

        // Number of windows = (last_start - first_start) / slide + 1.
        let count = if first_start > last_start {
            0
        } else {
            ((last_start - first_start) / self.slide_ms + 1) as usize
        };

        let mut windows = Vec::with_capacity(count);
        let mut start = first_start;
        while start <= last_start {
            windows.push(WindowRange::new(start, start + self.size_ms));
            start += self.slide_ms;
        }
        windows
    }

    #[inline(always)]
    fn assign_windows_into(&self, event_time_ms: i64, buf: &mut Vec<WindowRange>) -> usize {
        buf.clear();
        let last_start = event_time_ms - event_time_ms.rem_euclid(self.slide_ms);
        let earliest = (event_time_ms - self.size_ms + 1).max(0);
        let first_start =
            earliest + (self.slide_ms - earliest.rem_euclid(self.slide_ms)) % self.slide_ms;

        let mut start = first_start;
        while start <= last_start {
            buf.push(WindowRange::new(start, start + self.size_ms));
            start += self.slide_ms;
        }
        buf.len()
    }

    fn max_window_duration_ms(&self) -> i64 {
        self.size_ms
    }
}

// ---------------------------------------------------------------------------
// SessionWindowAssigner
// ---------------------------------------------------------------------------

/// Activity-based windows defined by an inactivity gap.
/// Returns a provisional single-event window. Actual merging is performed
/// by the SessionMerger when events bridge separate sessions.
pub struct SessionWindowAssigner {
    pub gap_ms: i64,
}

impl SessionWindowAssigner {
    pub fn new(gap_ms: i64) -> Self {
        assert!(gap_ms > 0, "session gap must be positive");
        Self { gap_ms }
    }
}

impl WindowAssigner for SessionWindowAssigner {
    #[inline(always)]
    fn assign_windows(&self, event_time_ms: i64) -> Vec<WindowRange> {
        vec![WindowRange::new(event_time_ms, event_time_ms + self.gap_ms)]
    }

    #[inline(always)]
    fn assign_windows_into(&self, event_time_ms: i64, buf: &mut Vec<WindowRange>) -> usize {
        buf.clear();
        buf.push(WindowRange::new(event_time_ms, event_time_ms + self.gap_ms));
        1
    }

    fn max_window_duration_ms(&self) -> i64 {
        // Session windows have no fixed max duration.
        i64::MAX
    }
}

// ---------------------------------------------------------------------------
// CumulativeWindowAssigner
// ---------------------------------------------------------------------------

/// Running totals within fixed periods at step intervals.
/// For example, daily cumulative at hourly steps: max_size = 24h, step = 1h.
pub struct CumulativeWindowAssigner {
    pub max_size_ms: i64,
    pub step_ms: i64,
}

impl CumulativeWindowAssigner {
    pub fn new(max_size_ms: i64, step_ms: i64) -> Self {
        assert!(max_size_ms > 0, "max size must be positive");
        assert!(step_ms > 0, "step must be positive");
        assert!(max_size_ms >= step_ms, "max size must be >= step");
        Self {
            max_size_ms,
            step_ms,
        }
    }
}

impl WindowAssigner for CumulativeWindowAssigner {
    #[inline(always)]
    fn assign_windows(&self, event_time_ms: i64) -> Vec<WindowRange> {
        let period_start = event_time_ms - event_time_ms.rem_euclid(self.max_size_ms);
        let offset = event_time_ms - period_start;
        let step_idx = offset / self.step_ms;
        let max_steps = self.max_size_ms / self.step_ms;
        let mut windows = Vec::with_capacity((max_steps - step_idx) as usize);
        for i in step_idx..=(max_steps - 1) {
            let end = period_start + (i + 1) * self.step_ms;
            windows.push(WindowRange::new(period_start, end));
        }
        windows
    }

    #[inline(always)]
    fn assign_windows_into(&self, event_time_ms: i64, buf: &mut Vec<WindowRange>) -> usize {
        buf.clear();
        let period_start = event_time_ms - event_time_ms.rem_euclid(self.max_size_ms);
        let offset = event_time_ms - period_start;
        let step_idx = offset / self.step_ms;
        let max_steps = self.max_size_ms / self.step_ms;
        buf.reserve((max_steps - step_idx) as usize);
        for i in step_idx..=(max_steps - 1) {
            let end = period_start + (i + 1) * self.step_ms;
            buf.push(WindowRange::new(period_start, end));
        }
        buf.len()
    }

    fn max_window_duration_ms(&self) -> i64 {
        self.max_size_ms
    }
}

// ---------------------------------------------------------------------------
// SessionMerger: union-find for merging overlapping session windows
// ---------------------------------------------------------------------------

/// Union-find structure for merging session windows.
/// Uses path halving (iterative) for find and union by rank.
pub struct SessionMerger {
    /// Maps key hash to session index.
    key_to_session: hashbrown::HashMap<u64, u32>,
    /// Union-find parent array.
    parent: Vec<u32>,
    /// Rank for union by rank.
    rank: Vec<u32>,
    /// Window range per session (at root).
    ranges: Vec<WindowRange>,
    /// Number of active sessions.
    count: usize,
}

impl SessionMerger {
    pub fn new() -> Self {
        Self {
            key_to_session: hashbrown::HashMap::new(),
            parent: Vec::new(),
            rank: Vec::new(),
            ranges: Vec::new(),
            count: 0,
        }
    }

    /// Adds an event and returns the root session index + merged window range.
    /// If the event bridges two existing sessions, they are merged.
    pub fn add_event(
        &mut self,
        key_hash: u64,
        event_time_ms: i64,
        gap_ms: i64,
    ) -> (u32, WindowRange) {
        let new_range = WindowRange::new(event_time_ms, event_time_ms + gap_ms);

        if let Some(&existing_idx) = self.key_to_session.get(&key_hash) {
            let root = self.find(existing_idx);
            let merged = self.ranges[root as usize].merge(&new_range);
            self.ranges[root as usize] = merged;
            (root, merged)
        } else {
            let idx = self.parent.len() as u32;
            self.parent.push(idx);
            self.rank.push(0);
            self.ranges.push(new_range);
            self.key_to_session.insert(key_hash, idx);
            self.count += 1;
            (idx, new_range)
        }
    }

    /// Merges two sessions by key hash. Returns the merged window range.
    pub fn merge_sessions(&mut self, key_a: u64, key_b: u64) -> Result<WindowRange> {
        let idx_a = *self
            .key_to_session
            .get(&key_a)
            .ok_or_else(|| zyron_common::ZyronError::StreamingError("session not found".into()))?;
        let idx_b = *self
            .key_to_session
            .get(&key_b)
            .ok_or_else(|| zyron_common::ZyronError::StreamingError("session not found".into()))?;

        let root = self.union(idx_a, idx_b);
        Ok(self.ranges[root as usize])
    }

    /// Find with path halving (iterative, no recursion).
    #[inline]
    pub fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            let grandparent = self.parent[self.parent[x as usize] as usize];
            self.parent[x as usize] = grandparent;
            x = grandparent;
        }
        x
    }

    /// Union by rank. Returns the new root.
    fn union(&mut self, a: u32, b: u32) -> u32 {
        let root_a = self.find(a);
        let root_b = self.find(b);
        if root_a == root_b {
            return root_a;
        }

        let merged_range = self.ranges[root_a as usize].merge(&self.ranges[root_b as usize]);

        let new_root = if self.rank[root_a as usize] < self.rank[root_b as usize] {
            self.parent[root_a as usize] = root_b;
            self.ranges[root_b as usize] = merged_range;
            root_b
        } else if self.rank[root_a as usize] > self.rank[root_b as usize] {
            self.parent[root_b as usize] = root_a;
            self.ranges[root_a as usize] = merged_range;
            root_a
        } else {
            self.parent[root_b as usize] = root_a;
            self.rank[root_a as usize] += 1;
            self.ranges[root_a as usize] = merged_range;
            root_a
        };

        self.count -= 1;
        new_root
    }

    /// Returns the window range for the session containing this key.
    pub fn get_range(&mut self, key_hash: u64) -> Option<WindowRange> {
        let idx = *self.key_to_session.get(&key_hash)?;
        let root = self.find(idx);
        Some(self.ranges[root as usize])
    }

    /// Number of distinct sessions.
    pub fn session_count(&self) -> usize {
        self.count
    }

    /// Removes expired sessions where window end < cutoff.
    pub fn expire_before(&mut self, cutoff_ms: i64) {
        let expired_keys: Vec<u64> = self
            .key_to_session
            .iter()
            .filter(|(_, idx)| {
                // Check root range. Note: find needs &mut self, so we check
                // parent chain manually without path compression for this scan.
                let mut x = **idx;
                while self.parent[x as usize] != x {
                    x = self.parent[x as usize];
                }
                self.ranges[x as usize].end_ms < cutoff_ms
            })
            .map(|(k, _)| *k)
            .collect();

        for key in expired_keys {
            self.key_to_session.remove(&key);
        }
    }
}

impl Default for SessionMerger {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SessionAssigner: per-key stateful session window tracker
// ---------------------------------------------------------------------------

/// Per-key stateful session window assigner. Keeps one open WindowRange per
/// key. Each observed event extends the key's open window to cover the event
/// and the trailing inactivity gap. Closed sessions are drained via
/// drain_closed once the watermark passes their end.
///
/// This is distinct from SessionWindowAssigner (which is stateless and used by
/// the older operator pipeline). SessionAssigner matches the streaming runner
/// contract where each event routes to exactly one live session per key.
pub struct SessionAssigner {
    gap_ms: i64,
    /// Per-key open-window tracker. Maps the caller-supplied key bytes to the
    /// currently open window.
    open: scc::HashMap<Vec<u8>, WindowRange>,
}

impl SessionAssigner {
    /// Creates a new assigner. gap_ms is the inactivity gap after which a
    /// session closes. Panics if gap_ms is not positive.
    pub fn new(gap_ms: i64) -> Self {
        assert!(gap_ms > 0, "session gap must be positive");
        Self {
            gap_ms,
            open: scc::HashMap::new(),
        }
    }

    /// Extends the open window for `key` to include event_time_ms. Returns the
    /// updated WindowRange. If no open window exists, or if the previous
    /// window's end is at or before the event time, starts a new session
    /// covering [event_time_ms, event_time_ms + gap_ms).
    pub fn assign(&self, key: &[u8], event_time_ms: i64) -> WindowRange {
        let mut result = WindowRange::new(event_time_ms, event_time_ms + self.gap_ms);
        let key_vec = key.to_vec();
        // entry_sync gives us upsert semantics.
        let entry = self.open.entry_sync(key_vec);
        match entry {
            scc::hash_map::Entry::Occupied(mut occ) => {
                let existing = *occ.get();
                if existing.end_ms <= event_time_ms {
                    // Previous session already lapsed, start a fresh one.
                    occ.insert(result);
                } else {
                    // Extend to cover the new event plus a trailing gap.
                    let extended = WindowRange::new(
                        existing.start_ms.min(event_time_ms),
                        (event_time_ms + self.gap_ms).max(existing.end_ms),
                    );
                    occ.insert(extended);
                    result = extended;
                }
            }
            scc::hash_map::Entry::Vacant(vac) => {
                vac.insert_entry(result);
            }
        }
        result
    }

    /// Returns and removes any sessions where end_ms is at or before the
    /// watermark. Drained entries are returned as (key, window) pairs so the
    /// caller can emit final aggregates and release state.
    pub fn drain_closed(&self, watermark_ms: i64) -> Vec<(Vec<u8>, WindowRange)> {
        let mut to_remove = Vec::new();
        self.open.iter_sync(|k, v| {
            if v.end_ms <= watermark_ms {
                to_remove.push((k.clone(), *v));
            }
            true
        });
        let mut drained = Vec::with_capacity(to_remove.len());
        for (k, v) in to_remove {
            if self.open.remove_sync(&k).is_some() {
                drained.push((k, v));
            }
        }
        drained
    }

    /// Number of currently open sessions.
    pub fn open_count(&self) -> usize {
        self.open.len()
    }
}

// ---------------------------------------------------------------------------
// Watermark-based closure check
// ---------------------------------------------------------------------------

/// Returns true if `window` has been fully covered by the watermark, i.e. the
/// window's end is at or before the watermark and no further events can arrive
/// that would land inside it.
#[inline]
pub fn window_is_closed(window: &WindowRange, watermark_ms: i64) -> bool {
    window.end_ms <= watermark_ms
}

// ---------------------------------------------------------------------------
// SQL window accessor functions
// ---------------------------------------------------------------------------

/// SQL: TUMBLE_START(), HOP_START(), SESSION_START(), CUMULATE_START().
#[inline]
pub fn window_start(range: &WindowRange) -> i64 {
    range.start_ms
}

/// SQL: TUMBLE_END(), HOP_END(), SESSION_END(), CUMULATE_END().
#[inline]
pub fn window_end(range: &WindowRange) -> i64 {
    range.end_ms
}

/// SQL: TUMBLE_ROWTIME(), HOP_ROWTIME(). Returns end - 1 (last valid timestamp).
#[inline]
pub fn window_rowtime(range: &WindowRange) -> i64 {
    range.end_ms - 1
}

/// SQL: SESSION_ID(). Returns the start timestamp as a unique session identifier.
#[inline]
pub fn session_id(range: &WindowRange) -> i64 {
    range.start_ms
}

// ---------------------------------------------------------------------------
// Convenience constructors matching SQL syntax
// ---------------------------------------------------------------------------

/// SQL: TUMBLE(time_col, interval).
#[inline]
pub fn tumble(event_time_ms: i64, size_ms: i64) -> WindowRange {
    let start = event_time_ms - event_time_ms.rem_euclid(size_ms);
    WindowRange::new(start, start + size_ms)
}

/// SQL: HOP(time_col, slide, size). Returns all windows containing this event.
pub fn hop(event_time_ms: i64, slide_ms: i64, size_ms: i64) -> Vec<WindowRange> {
    let assigner = SlidingWindowAssigner::new(size_ms, slide_ms);
    assigner.assign_windows(event_time_ms)
}

/// SQL: SESSION(time_col, gap). Returns provisional window for this event.
#[inline]
pub fn session_assign(event_time_ms: i64, gap_ms: i64) -> WindowRange {
    WindowRange::new(event_time_ms, event_time_ms + gap_ms)
}

/// SQL: CUMULATE(time_col, step, max_size).
pub fn cumulate(event_time_ms: i64, step_ms: i64, max_size_ms: i64) -> Vec<WindowRange> {
    let assigner = CumulativeWindowAssigner::new(max_size_ms, step_ms);
    assigner.assign_windows(event_time_ms)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tumbling_window() {
        let assigner = TumblingWindowAssigner::new(60_000); // 1 minute
        let windows = assigner.assign_windows(123_456);
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0], WindowRange::new(120_000, 180_000));
    }

    #[test]
    fn test_tumbling_window_boundary() {
        let assigner = TumblingWindowAssigner::new(1000);
        let windows = assigner.assign_windows(1000);
        assert_eq!(windows[0], WindowRange::new(1000, 2000));
    }

    #[test]
    fn test_sliding_window() {
        let assigner = SlidingWindowAssigner::new(300_000, 60_000); // 5min window, 1min slide
        let windows = assigner.assign_windows(150_000);
        // Event at 150s should be in windows starting at: 0, 60000, 120000
        assert!(windows.len() >= 2);
        for w in &windows {
            assert!(w.contains(150_000));
        }
    }

    #[test]
    fn test_session_window() {
        let assigner = SessionWindowAssigner::new(30_000);
        let windows = assigner.assign_windows(100_000);
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0], WindowRange::new(100_000, 130_000));
    }

    #[test]
    fn test_cumulative_window() {
        // 24h period, 1h steps. Event at 2.5 hours.
        let assigner = CumulativeWindowAssigner::new(86_400_000, 3_600_000);
        let windows = assigner.assign_windows(9_000_000); // 2.5 hours
        // Should be in cumulative windows from step 2 onwards.
        assert!(!windows.is_empty());
        for w in &windows {
            assert_eq!(w.start_ms, 0);
            assert!(w.contains(9_000_000));
        }
    }

    #[test]
    fn test_session_merger_basic() {
        let mut merger = SessionMerger::new();
        let gap = 30_000;

        // First event for user A at time 100.
        let (_, r1) = merger.add_event(1, 100_000, gap);
        assert_eq!(r1, WindowRange::new(100_000, 130_000));

        // Second event for user A at time 110 (within gap).
        let (_, r2) = merger.add_event(1, 110_000, gap);
        // Should merge: [100000, 140000).
        assert_eq!(r2.start_ms, 100_000);
        assert_eq!(r2.end_ms, 140_000);
    }

    #[test]
    fn test_session_merger_separate_keys() {
        let mut merger = SessionMerger::new();
        merger.add_event(1, 100_000, 30_000);
        merger.add_event(2, 200_000, 30_000);
        assert_eq!(merger.session_count(), 2);
    }

    #[test]
    fn test_window_range_merge() {
        let a = WindowRange::new(0, 100);
        let b = WindowRange::new(50, 200);
        let merged = a.merge(&b);
        assert_eq!(merged, WindowRange::new(0, 200));
    }

    #[test]
    fn test_window_range_overlaps() {
        let a = WindowRange::new(0, 100);
        let b = WindowRange::new(50, 200);
        let c = WindowRange::new(200, 300);
        assert!(a.overlaps(&b));
        assert!(!a.overlaps(&c));
        assert!(!b.overlaps(&c));
    }

    #[test]
    fn test_sql_functions() {
        let w = tumble(123_456, 60_000);
        assert_eq!(window_start(&w), 120_000);
        assert_eq!(window_end(&w), 180_000);
        assert_eq!(window_rowtime(&w), 179_999);
        assert_eq!(session_id(&w), 120_000);
    }

    #[test]
    fn tumbling_assigns_one_window_per_event() {
        let a = TumblingWindowAssigner::new(1_000);
        for t in [0, 1, 999, 1_000, 5_500] {
            let ws = a.assign_windows(t);
            assert_eq!(ws.len(), 1, "tumbling must produce exactly one window");
        }
    }

    #[test]
    fn hopping_overlap_at_size_gt_slide() {
        // Size 300, slide 100: each event lands in 3 overlapping windows.
        let a = SlidingWindowAssigner::new(300, 100);
        let ws = a.assign_windows(500);
        assert!(ws.len() >= 2, "overlap requires at least two windows");
        for w in &ws {
            assert!(w.contains(500));
        }
        // When slide == size, behaves like tumbling (one window).
        let same = SlidingWindowAssigner::new(200, 200);
        assert_eq!(same.assign_windows(500).len(), 1);
    }

    #[test]
    fn session_extends_open_window_within_gap() {
        let a = SessionAssigner::new(50);
        let r1 = a.assign(b"user-1", 100);
        assert_eq!(r1, WindowRange::new(100, 150));
        // Event within the gap extends the window.
        let r2 = a.assign(b"user-1", 120);
        assert_eq!(r2.start_ms, 100);
        assert_eq!(r2.end_ms, 170);
    }

    #[test]
    fn session_starts_new_window_after_gap() {
        let a = SessionAssigner::new(50);
        let _ = a.assign(b"user-1", 100);
        // Event past end (150) starts a fresh session.
        let r = a.assign(b"user-1", 200);
        assert_eq!(r, WindowRange::new(200, 250));
    }

    #[test]
    fn watermark_closes_past_windows() {
        let w = WindowRange::new(100, 200);
        assert!(!window_is_closed(&w, 150));
        assert!(!window_is_closed(&w, 199));
        assert!(window_is_closed(&w, 200));
        assert!(window_is_closed(&w, 201));

        // SessionAssigner drains closed sessions.
        let a = SessionAssigner::new(30);
        a.assign(b"k1", 100);
        a.assign(b"k2", 500);
        let drained = a.drain_closed(200);
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].0, b"k1".to_vec());
        assert_eq!(a.open_count(), 1);
    }

    #[test]
    fn test_hop_function() {
        let windows = hop(150_000, 60_000, 300_000);
        assert!(!windows.is_empty());
        for w in &windows {
            assert!(w.contains(150_000));
        }
    }
}

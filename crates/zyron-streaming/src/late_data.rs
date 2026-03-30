//! Late data handling, allowed lateness, and event deduplication.
//!
//! Provides LateDataPolicy (Update, Drop, SideOutput), AllowedLateness
//! configuration, LateDataHandler with atomic counters for tracking
//! dropped/updated/side-output records, and DeduplicationFilter with
//! configurable dedup windows and periodic cleanup of expired entries.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::record::StreamRecord;
use crate::spsc::SpscSender;
use crate::watermark::Watermark;

// ---------------------------------------------------------------------------
// LateDataPolicy
// ---------------------------------------------------------------------------

/// Determines how late-arriving data is handled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LateDataPolicy {
    /// Emit retraction (-U) and updated result (+U) for affected windows.
    Update,
    /// Silently discard late records.
    Drop,
    /// Route late records to a side output channel.
    SideOutput,
}

// ---------------------------------------------------------------------------
// AllowedLateness
// ---------------------------------------------------------------------------

/// Configuration for how late data can arrive beyond the watermark.
#[derive(Debug, Clone)]
pub struct AllowedLateness {
    /// Maximum allowed lateness in milliseconds.
    pub lateness_ms: i64,
}

impl AllowedLateness {
    pub fn new(lateness_ms: i64) -> Self {
        Self { lateness_ms }
    }

    /// Returns true if an event at `event_time` is considered late
    /// given the current watermark and this lateness configuration.
    #[inline]
    pub fn is_late(&self, event_time_ms: i64, watermark: &Watermark) -> bool {
        event_time_ms < watermark.timestamp_ms - self.lateness_ms
    }

    /// No lateness allowed. Any event before the watermark is late.
    pub fn none() -> Self {
        Self { lateness_ms: 0 }
    }
}

/// Inline check: returns true if the event is late given watermark and
/// allowed lateness.
#[inline]
pub fn is_late(event_time_ms: i64, watermark: &Watermark, allowed_lateness_ms: i64) -> bool {
    event_time_ms < watermark.timestamp_ms - allowed_lateness_ms
}

// ---------------------------------------------------------------------------
// LateDataHandler
// ---------------------------------------------------------------------------

/// Handles late data according to the configured policy.
/// Tracks statistics via atomic counters for monitoring.
pub struct LateDataHandler {
    policy: LateDataPolicy,
    allowed_lateness: AllowedLateness,
    /// Side output channel for SideOutput policy.
    side_output: Option<SpscSender<StreamRecord>>,
    /// Count of records dropped due to lateness.
    pub dropped_count: AtomicU64,
    /// Count of records that triggered window updates.
    pub updated_count: AtomicU64,
    /// Count of records routed to the side output.
    pub side_output_count: AtomicU64,
}

impl LateDataHandler {
    /// Creates a handler with the given policy and lateness configuration.
    pub fn new(policy: LateDataPolicy, allowed_lateness: AllowedLateness) -> Self {
        Self {
            policy,
            allowed_lateness,
            side_output: None,
            dropped_count: AtomicU64::new(0),
            updated_count: AtomicU64::new(0),
            side_output_count: AtomicU64::new(0),
        }
    }

    /// Sets the side output channel for SideOutput policy.
    pub fn with_side_output(mut self, sender: SpscSender<StreamRecord>) -> Self {
        self.side_output = Some(sender);
        self
    }

    /// Returns the configured policy.
    pub fn policy(&self) -> LateDataPolicy {
        self.policy
    }

    /// Returns true if the event at the given time is considered late.
    #[inline]
    pub fn is_late(&self, event_time_ms: i64, watermark: &Watermark) -> bool {
        self.allowed_lateness.is_late(event_time_ms, watermark)
    }

    /// Handles a single late record according to the policy.
    /// Returns true if the record should be processed (Update policy),
    /// false if it was dropped or side-output.
    pub fn handle_late_record(
        &self,
        record: StreamRecord,
        _event_time_ms: i64,
        _watermark: &Watermark,
    ) -> bool {
        match self.policy {
            LateDataPolicy::Update => {
                self.updated_count.fetch_add(1, Ordering::Relaxed);
                true
            }
            LateDataPolicy::Drop => {
                self.dropped_count.fetch_add(1, Ordering::Relaxed);
                false
            }
            LateDataPolicy::SideOutput => {
                self.side_output_count.fetch_add(1, Ordering::Relaxed);
                if let Some(ref sender) = self.side_output {
                    sender.send(record);
                }
                false
            }
        }
    }

    /// Handles a batch of records, filtering late ones.
    /// Returns the non-late portion of the record (rows that pass).
    pub fn filter_late(&self, record: &StreamRecord, watermark: &Watermark) -> StreamRecord {
        let num_rows = record.num_rows();
        if num_rows == 0 {
            return record.clone();
        }

        // Count late records before allocating a mask.
        let threshold = watermark.timestamp_ms - self.allowed_lateness.lateness_ms;
        let mut late_count = 0u64;
        for i in 0..num_rows {
            if record.event_times[i] < threshold {
                late_count += 1;
            }
        }

        // Fast path: no late records, return clone without building a mask.
        if late_count == 0 {
            return record.clone();
        }

        match self.policy {
            LateDataPolicy::Update => {
                self.updated_count.fetch_add(late_count, Ordering::Relaxed);
                // For Update policy, return all records (late ones trigger updates).
                return record.clone();
            }
            LateDataPolicy::Drop => {
                self.dropped_count.fetch_add(late_count, Ordering::Relaxed);
            }
            LateDataPolicy::SideOutput => {
                self.side_output_count
                    .fetch_add(late_count, Ordering::Relaxed);
            }
        }

        // Build the mask only when filtering is needed.
        let mut mask = vec![true; num_rows];
        for i in 0..num_rows {
            if record.event_times[i] < threshold {
                mask[i] = false;
            }
        }

        if self.policy == LateDataPolicy::SideOutput {
            let late_mask: Vec<bool> = mask.iter().map(|&keep| !keep).collect();
            let late_record = record.filter(&late_mask);
            if let Some(ref sender) = self.side_output {
                sender.send(late_record);
            }
        }

        record.filter(&mask)
    }

    /// Returns total number of dropped records.
    pub fn total_dropped(&self) -> u64 {
        self.dropped_count.load(Ordering::Relaxed)
    }

    /// Returns total number of updated records.
    pub fn total_updated(&self) -> u64 {
        self.updated_count.load(Ordering::Relaxed)
    }

    /// Returns total number of side-output records.
    pub fn total_side_output(&self) -> u64 {
        self.side_output_count.load(Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// DeduplicationFilter
// ---------------------------------------------------------------------------

/// Event deduplication filter using a hashmap of event_id_hash -> expiry_time.
/// Configurable dedup window with periodic cleanup of expired entries.
pub struct DeduplicationFilter {
    /// Maps event ID hash to expiry timestamp (ms).
    seen: hashbrown::HashMap<u64, i64>,
    /// Dedup window in milliseconds. Events within this window are deduplicated.
    dedup_window_ms: i64,
    /// How often to run cleanup (in number of insert calls).
    cleanup_interval: u64,
    /// Counter for cleanup scheduling.
    insert_count: u64,
}

impl DeduplicationFilter {
    /// Creates a new dedup filter with the given window.
    pub fn new(dedup_window_ms: i64) -> Self {
        Self {
            seen: hashbrown::HashMap::new(),
            dedup_window_ms,
            cleanup_interval: 10_000,
            insert_count: 0,
        }
    }

    /// Creates a filter with a custom cleanup interval.
    pub fn with_cleanup_interval(mut self, interval: u64) -> Self {
        self.cleanup_interval = interval.max(1);
        self
    }

    /// Returns true if this event ID has already been seen (is a duplicate).
    /// If not seen, records it with an expiry time.
    #[inline]
    pub fn is_duplicate(&mut self, event_id_hash: u64, current_time_ms: i64) -> bool {
        if let Some(&expiry) = self.seen.get(&event_id_hash) {
            if current_time_ms < expiry {
                return true; // Duplicate within the dedup window.
            }
            // Expired entry, update it.
        }

        self.seen
            .insert(event_id_hash, current_time_ms + self.dedup_window_ms);
        self.insert_count += 1;

        if self.insert_count % self.cleanup_interval == 0 {
            self.cleanup(current_time_ms);
        }

        false
    }

    /// Removes all expired entries from the filter.
    pub fn cleanup(&mut self, current_time_ms: i64) {
        self.seen.retain(|_, &mut expiry| expiry > current_time_ms);
    }

    /// Number of tracked event IDs.
    pub fn len(&self) -> usize {
        self.seen.len()
    }

    /// Returns true if no events are being tracked.
    pub fn is_empty(&self) -> bool {
        self.seen.is_empty()
    }

    /// Clears all tracked events.
    pub fn clear(&mut self) {
        self.seen.clear();
        self.insert_count = 0;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::{StreamBatch, StreamColumn, StreamColumnData};
    use crate::record::ChangeFlag;

    fn make_record_with_times(times: Vec<i64>) -> StreamRecord {
        let n = times.len();
        let col = StreamColumn::from_data(StreamColumnData::Int64((0..n as i64).collect()));
        let batch = StreamBatch::new(vec![col]);
        StreamRecord::new(batch, times, vec![ChangeFlag::Insert; n])
    }

    #[test]
    fn test_is_late_function() {
        let wm = Watermark::new(10_000);
        assert!(!is_late(9_500, &wm, 1_000)); // Within lateness.
        assert!(is_late(8_000, &wm, 1_000)); // Too late.
        assert!(!is_late(10_000, &wm, 0)); // Exactly at watermark.
        assert!(is_late(9_999, &wm, 0)); // Just before watermark with no lateness.
    }

    #[test]
    fn test_allowed_lateness() {
        let al = AllowedLateness::new(5_000);
        let wm = Watermark::new(20_000);
        assert!(!al.is_late(16_000, &wm)); // 20000 - 5000 = 15000, 16000 >= 15000
        assert!(al.is_late(14_000, &wm)); // 14000 < 15000
    }

    #[test]
    fn test_late_data_handler_drop() {
        let handler = LateDataHandler::new(LateDataPolicy::Drop, AllowedLateness::none());
        let wm = Watermark::new(10_000);
        let record = make_record_with_times(vec![5_000, 10_500, 8_000, 12_000]);
        let result = handler.filter_late(&record, &wm);
        // Rows 0 (5000) and 2 (8000) are late (< 10000 with lateness 0).
        assert_eq!(result.num_rows(), 2);
        assert_eq!(handler.total_dropped(), 2);
    }

    #[test]
    fn test_late_data_handler_update() {
        let handler = LateDataHandler::new(LateDataPolicy::Update, AllowedLateness::none());
        let wm = Watermark::new(10_000);
        let record = make_record_with_times(vec![5_000, 10_500]);
        let result = handler.filter_late(&record, &wm);
        // Update policy returns all records.
        assert_eq!(result.num_rows(), 2);
        assert_eq!(handler.total_updated(), 1);
    }

    #[test]
    fn test_late_data_handler_side_output() {
        let (tx, rx) = crate::spsc::spsc_channel(16);
        let handler = LateDataHandler::new(LateDataPolicy::SideOutput, AllowedLateness::none())
            .with_side_output(tx);

        let wm = Watermark::new(10_000);
        let record = make_record_with_times(vec![5_000, 12_000]);
        let result = handler.filter_late(&record, &wm);
        assert_eq!(result.num_rows(), 1); // Only 12_000 passes.
        assert_eq!(handler.total_side_output(), 1);

        // Side output should have the late record.
        let side = rx.try_recv();
        assert!(side.is_some());
        assert_eq!(side.map(|r| r.num_rows()), Some(1));
    }

    #[test]
    fn test_dedup_filter_basic() {
        let mut filter = DeduplicationFilter::new(10_000);
        assert!(!filter.is_duplicate(42, 1_000)); // First time.
        assert!(filter.is_duplicate(42, 2_000)); // Duplicate within window.
        assert!(!filter.is_duplicate(43, 2_000)); // Different ID.
    }

    #[test]
    fn test_dedup_filter_expiry() {
        let mut filter = DeduplicationFilter::new(5_000);
        assert!(!filter.is_duplicate(42, 1_000)); // Expires at 6_000.
        assert!(filter.is_duplicate(42, 5_000)); // Still valid.
        assert!(!filter.is_duplicate(42, 7_000)); // Expired, treated as new.
    }

    #[test]
    fn test_dedup_filter_cleanup() {
        let mut filter = DeduplicationFilter::new(5_000);
        filter.is_duplicate(1, 1_000); // Expires at 6_000
        filter.is_duplicate(2, 2_000); // Expires at 7_000
        filter.is_duplicate(3, 3_000); // Expires at 8_000

        assert_eq!(filter.len(), 3);
        filter.cleanup(6_500); // Should remove entry 1 (expired at 6_000).
        assert_eq!(filter.len(), 2);
    }

    #[test]
    fn test_no_late_data() {
        let handler = LateDataHandler::new(LateDataPolicy::Drop, AllowedLateness::new(5_000));
        let wm = Watermark::new(10_000);
        let record = make_record_with_times(vec![6_000, 8_000, 10_000]);
        let result = handler.filter_late(&record, &wm);
        // All records are within lateness (watermark 10000 - lateness 5000 = 5000).
        assert_eq!(result.num_rows(), 3);
        assert_eq!(handler.total_dropped(), 0);
    }
}

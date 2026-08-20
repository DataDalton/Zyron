//! What unasked compactions did, so an operator can see them.
//!
//! A compaction the operator did not ask for rewrites files and changes
//! how much a scan reads, and a maintenance loop that does that silently
//! is one nobody can reason about: a table that got faster overnight and a
//! table that got slower look identical from the outside. This records
//! every run, what tripped it, and what it moved.
//!
//! Process wide, like the workload observer and the log registry, because
//! maintenance belongs to the node rather than to a session. Held in a
//! fixed ring so a node that has been up for months carries a bounded
//! amount of it: this is an operator's recent history, not an audit log,
//! and the durable record of what happened is the transaction log itself.

use std::sync::{Mutex, OnceLock};

use crate::manifest::CompactionTrigger;

/// Runs the ring keeps.
///
/// One tick can compact at most one table, so this is at least a few
/// hundred maintenance cycles on a node with a handful of lake tables,
/// which is the window an operator asks about
pub const HISTORY_CAPACITY: usize = 256;

/// One compaction that ran without being asked
#[derive(Debug, Clone, PartialEq)]
pub struct CompactionRecord {
    pub table_id: u32,
    /// Table name as the catalog had it when the run started
    pub table_name: String,
    pub trigger: CompactionTrigger,
    /// Microseconds since the epoch
    pub triggered_at_us: i64,
    pub files_before: usize,
    pub files_after: usize,
    /// Rows the rewrite removed physically that a delete had already
    /// removed logically
    pub dead_rows_reclaimed: u64,
    /// Share of files that were below a quarter of the target, in
    /// thousandths, so the view stays integer typed
    pub small_file_ratio_milli: u32,
    /// Share of rows that were deleted and not yet rewritten, in
    /// thousandths
    pub dead_row_ratio_milli: u32,
    /// Log version the rewrite committed, None when it changed nothing
    pub version: Option<u64>,
}

/// The node's recent unasked compactions, newest last
#[derive(Debug, Default)]
pub struct CompactionHistory {
    runs: Mutex<std::collections::VecDeque<CompactionRecord>>,
}

impl CompactionHistory {
    pub fn record(&self, run: CompactionRecord) {
        let mut runs = self.runs.lock().unwrap_or_else(|e| e.into_inner());
        if runs.len() == HISTORY_CAPACITY {
            runs.pop_front();
        }
        runs.push_back(run);
    }

    /// Every run still in the ring, oldest first
    pub fn runs(&self) -> Vec<CompactionRecord> {
        self.runs
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .cloned()
            .collect()
    }

    /// Runs for one table, oldest first
    pub fn runs_for(&self, table_id: u32) -> Vec<CompactionRecord> {
        self.runs
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|r| r.table_id == table_id)
            .cloned()
            .collect()
    }

    pub fn len(&self) -> usize {
        self.runs.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    pub fn is_empty(&self) -> bool {
        self.runs
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_empty()
    }
}

static HISTORY: OnceLock<CompactionHistory> = OnceLock::new();

/// The node's compaction history
#[inline]
pub fn compaction_history() -> &'static CompactionHistory {
    HISTORY.get_or_init(CompactionHistory::default)
}

/// A ratio as thousandths, saturating rather than wrapping.
///
/// The view is integer typed, matching how the clustering metrics report
/// their rates, so a rate crosses into it here rather than at every
/// reader
#[inline]
pub fn ratio_milli(numerator: u64, denominator: u64) -> u32 {
    if denominator == 0 {
        return 0;
    }
    let scaled = numerator as u128 * 1000 / denominator as u128;
    u32::try_from(scaled).unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(table_id: u32, at: i64) -> CompactionRecord {
        CompactionRecord {
            table_id,
            table_name: format!("t{table_id}"),
            trigger: CompactionTrigger::SmallFiles,
            triggered_at_us: at,
            files_before: 8,
            files_after: 1,
            dead_rows_reclaimed: 0,
            small_file_ratio_milli: 875,
            dead_row_ratio_milli: 0,
            version: Some(4),
        }
    }

    /// A node that has been up for months holds a bounded amount of this,
    /// and what it drops is the oldest rather than the newest: an operator
    /// asking what just happened is the case this exists for
    #[test]
    fn test_the_ring_keeps_the_newest_runs() {
        let history = CompactionHistory::default();
        for i in 0..(HISTORY_CAPACITY as i64 + 10) {
            history.record(record(1, i));
        }
        assert_eq!(history.len(), HISTORY_CAPACITY);
        let runs = history.runs();
        assert_eq!(runs.first().expect("a run").triggered_at_us, 10);
        assert_eq!(
            runs.last().expect("a run").triggered_at_us,
            HISTORY_CAPACITY as i64 + 9
        );
    }

    #[test]
    fn test_runs_filter_by_table() {
        let history = CompactionHistory::default();
        history.record(record(1, 0));
        history.record(record(2, 1));
        history.record(record(1, 2));
        assert_eq!(history.runs_for(1).len(), 2);
        assert_eq!(history.runs_for(2).len(), 1);
        assert!(history.runs_for(3).is_empty());
    }

    /// A rate crosses into the integer form once, here, so two readers
    /// cannot round it differently
    #[test]
    fn test_ratio_milli_is_exact_at_the_ends_and_rounds_down_between() {
        assert_eq!(ratio_milli(0, 10), 0);
        assert_eq!(ratio_milli(10, 10), 1000);
        assert_eq!(ratio_milli(1, 3), 333);
        assert_eq!(ratio_milli(7, 8), 875);
        // A denominator of zero is no observation rather than a division
        assert_eq!(ratio_milli(5, 0), 0);
    }
}

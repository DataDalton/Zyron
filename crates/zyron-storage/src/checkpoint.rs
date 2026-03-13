//! Per-table checkpoint state tracking.
//!
//! Tracks the last checkpoint LSN for each table so hot tables can checkpoint
//! frequently while cold tables rarely. WAL segment cleanup uses the global
//! minimum checkpoint LSN across all tables as the safe deletion boundary.

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Checkpoint state for a single table.
pub struct TableCheckpointState {
    /// All pages for this table with dirty_lsn <= this value are on disk.
    pub last_checkpoint_lsn: AtomicU64,
    /// File IDs belonging to this table (heap_file_id, fsm_file_id).
    pub file_ids: Vec<u32>,
}

/// Tracks checkpoint progress for all registered tables.
///
/// The global minimum checkpoint LSN determines which WAL segments
/// can be safely deleted (all segments with max LSN below this value).
pub struct CheckpointTracker {
    /// Per-table checkpoint state, keyed by table_id.
    tables: RwLock<HashMap<u32, TableCheckpointState>>,
    /// Reverse map: file_id -> table_id for fast lookup.
    file_to_table: RwLock<HashMap<u32, u32>>,
    /// Minimum checkpoint LSN across all tables.
    /// WAL segments below this LSN are safe to delete.
    global_min_checkpoint_lsn: AtomicU64,
}

impl CheckpointTracker {
    /// Creates a new empty tracker.
    pub fn new() -> Self {
        Self {
            tables: RwLock::new(HashMap::new()),
            file_to_table: RwLock::new(HashMap::new()),
            global_min_checkpoint_lsn: AtomicU64::new(0),
        }
    }

    /// Registers a table for checkpoint tracking.
    pub fn register_table(&self, table_id: u32, file_ids: &[u32]) {
        let state = TableCheckpointState {
            last_checkpoint_lsn: AtomicU64::new(0),
            file_ids: file_ids.to_vec(),
        };

        let mut ftable = self.file_to_table.write();
        for &fid in file_ids {
            ftable.insert(fid, table_id);
        }

        self.tables.write().insert(table_id, state);
    }

    /// Unregisters a table (on DROP TABLE).
    pub fn unregister_table(&self, table_id: u32) {
        let mut tables = self.tables.write();
        if let Some(state) = tables.remove(&table_id) {
            let mut ftable = self.file_to_table.write();
            for &fid in &state.file_ids {
                ftable.remove(&fid);
            }
        }
        drop(tables);
        self.recompute_global_min();
    }

    /// Returns the last checkpoint LSN for a specific table.
    pub fn table_checkpoint_lsn(&self, table_id: u32) -> u64 {
        let tables = self.tables.read();
        tables
            .get(&table_id)
            .map(|s| s.last_checkpoint_lsn.load(Ordering::Acquire))
            .unwrap_or(0)
    }

    /// Advances a single table's checkpoint LSN.
    pub fn advance_table_checkpoint(&self, table_id: u32, lsn: u64) {
        let tables = self.tables.read();
        if let Some(state) = tables.get(&table_id) {
            state.last_checkpoint_lsn.store(lsn, Ordering::Release);
        }
        drop(tables);
        self.recompute_global_min();
    }

    /// Advances all registered tables to the given checkpoint LSN.
    pub fn advance_all_tables(&self, lsn: u64) {
        let tables = self.tables.read();
        for state in tables.values() {
            state.last_checkpoint_lsn.store(lsn, Ordering::Release);
        }
        drop(tables);
        self.global_min_checkpoint_lsn.store(lsn, Ordering::Release);
    }

    /// Recomputes the global minimum checkpoint LSN by scanning all tables.
    pub fn recompute_global_min(&self) {
        let tables = self.tables.read();
        let min = tables
            .values()
            .map(|s| s.last_checkpoint_lsn.load(Ordering::Acquire))
            .min()
            .unwrap_or(0);
        self.global_min_checkpoint_lsn.store(min, Ordering::Release);
    }

    /// Returns the global minimum checkpoint LSN.
    /// WAL segments with max LSN below this value can be safely deleted.
    pub fn global_min_checkpoint_lsn(&self) -> u64 {
        self.global_min_checkpoint_lsn.load(Ordering::Acquire)
    }

    /// Returns table IDs where the gap between current_lsn and their
    /// last checkpoint LSN exceeds the threshold.
    pub fn tables_needing_checkpoint(&self, current_lsn: u64, threshold: u64) -> Vec<u32> {
        let tables = self.tables.read();
        tables
            .iter()
            .filter_map(|(&tid, state)| {
                let last = state.last_checkpoint_lsn.load(Ordering::Acquire);
                if current_lsn.saturating_sub(last) > threshold {
                    Some(tid)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Looks up the table_id for a given file_id.
    pub fn table_for_file(&self, file_id: u32) -> Option<u32> {
        self.file_to_table.read().get(&file_id).copied()
    }

    /// Returns the number of registered tables.
    pub fn table_count(&self) -> usize {
        self.tables.read().len()
    }

    /// Returns a snapshot of (table_id, last_checkpoint_lsn) for all registered tables.
    /// Used by the checkpoint coordinator to serialize per-table state into the payload.
    pub fn snapshot_table_lsns(&self) -> Vec<(u32, u64)> {
        let tables = self.tables.read();
        tables
            .iter()
            .map(|(&tid, state)| (tid, state.last_checkpoint_lsn.load(Ordering::Acquire)))
            .collect()
    }
}

impl Default for CheckpointTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_lookup() {
        let tracker = CheckpointTracker::new();
        tracker.register_table(1, &[200, 201]);
        tracker.register_table(2, &[202, 203]);

        assert_eq!(tracker.table_count(), 2);
        assert_eq!(tracker.table_for_file(200), Some(1));
        assert_eq!(tracker.table_for_file(201), Some(1));
        assert_eq!(tracker.table_for_file(202), Some(2));
        assert_eq!(tracker.table_for_file(999), None);
    }

    #[test]
    fn test_advance_and_global_min() {
        let tracker = CheckpointTracker::new();
        tracker.register_table(1, &[200, 201]);
        tracker.register_table(2, &[202, 203]);

        // Both at 0
        assert_eq!(tracker.global_min_checkpoint_lsn(), 0);

        // Advance table 1 to 500
        tracker.advance_table_checkpoint(1, 500);
        // Global min is still 0 (table 2 is at 0)
        assert_eq!(tracker.global_min_checkpoint_lsn(), 0);

        // Advance table 2 to 300
        tracker.advance_table_checkpoint(2, 300);
        // Global min is 300
        assert_eq!(tracker.global_min_checkpoint_lsn(), 300);

        // Advance all to 1000
        tracker.advance_all_tables(1000);
        assert_eq!(tracker.global_min_checkpoint_lsn(), 1000);
        assert_eq!(tracker.table_checkpoint_lsn(1), 1000);
        assert_eq!(tracker.table_checkpoint_lsn(2), 1000);
    }

    #[test]
    fn test_unregister_table() {
        let tracker = CheckpointTracker::new();
        tracker.register_table(1, &[200, 201]);
        tracker.register_table(2, &[202, 203]);

        tracker.advance_table_checkpoint(1, 500);
        tracker.advance_table_checkpoint(2, 300);

        tracker.unregister_table(2);
        assert_eq!(tracker.table_count(), 1);
        assert_eq!(tracker.table_for_file(202), None);
        // Global min recalculated: only table 1 at 500
        assert_eq!(tracker.global_min_checkpoint_lsn(), 500);
    }

    #[test]
    fn test_tables_needing_checkpoint() {
        let tracker = CheckpointTracker::new();
        tracker.register_table(1, &[200]);
        tracker.register_table(2, &[202]);
        tracker.register_table(3, &[204]);

        tracker.advance_table_checkpoint(1, 100);
        tracker.advance_table_checkpoint(2, 900);
        tracker.advance_table_checkpoint(3, 500);

        // current_lsn = 1000, threshold = 200
        // Table 1: gap = 900 > 200 -> needs checkpoint
        // Table 2: gap = 100 <= 200 -> ok
        // Table 3: gap = 500 > 200 -> needs checkpoint
        let mut needing = tracker.tables_needing_checkpoint(1000, 200);
        needing.sort();
        assert_eq!(needing, vec![1, 3]);
    }

    #[test]
    fn test_empty_tracker() {
        let tracker = CheckpointTracker::new();
        assert_eq!(tracker.global_min_checkpoint_lsn(), 0);
        assert_eq!(tracker.table_count(), 0);
        assert!(tracker.tables_needing_checkpoint(1000, 100).is_empty());
    }
}

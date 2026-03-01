//! MVCC snapshot for transaction visibility.
//!
//! A Snapshot captures the set of active transactions at BEGIN time.
//! It provides visibility checks following standard MVCC rules:
//! - Tuples inserted by committed transactions before the snapshot are visible.
//! - Tuples inserted by active (uncommitted) transactions are invisible.
//! - Tuples deleted by committed transactions before the snapshot are invisible.
//! - Tuples inserted or deleted by the owning transaction follow self-visibility rules.

/// Immutable snapshot of active transactions taken at BEGIN time.
///
/// Used for MVCC visibility checks. The active_txn_ids list is sorted
/// for binary search during visibility checks.
#[derive(Debug, Clone)]
pub struct Snapshot {
    /// Transaction ID of the owning transaction.
    pub txn_id: u64,
    /// Sorted list of transaction IDs that were active at snapshot time.
    active_txn_ids: Vec<u64>,
}

impl Snapshot {
    /// Creates a new snapshot with the given active transaction set.
    /// Sorts the active list for binary search.
    pub fn new(txn_id: u64, mut active_txns: Vec<u64>) -> Self {
        active_txns.sort_unstable();
        Self {
            txn_id,
            active_txn_ids: active_txns,
        }
    }

    /// Checks if a tuple version is visible to this snapshot.
    ///
    /// MVCC visibility rules:
    /// 1. xmin == own txn_id: visible (own insert), unless xmax == own txn_id (self-deleted)
    /// 2. xmin < own txn_id and xmin not in active set: visible (committed before snapshot)
    /// 3. xmax == 0: tuple is live (not deleted)
    /// 4. xmax != 0 and xmax committed (not in active set) and xmax <= own txn_id: invisible (deleted)
    #[inline]
    pub fn is_visible(&self, xmin: u64, xmax: u64) -> bool {
        // Case 1: Own transaction inserted this tuple
        if xmin == self.txn_id {
            // Visible unless we also deleted it
            return xmax != self.txn_id;
        }

        // Case 2: Check if inserting transaction is visible
        // Invisible if xmin >= our txn_id (started after us)
        if xmin >= self.txn_id {
            return false;
        }

        // Fast path: when no active transactions at snapshot time, skip binary searches.
        // Common case for read-only workloads and low-concurrency OLTP.
        if self.active_txn_ids.is_empty() {
            return xmax == 0 || xmax >= self.txn_id;
        }

        // Invisible if xmin is still active (uncommitted at snapshot time)
        if self.is_txn_active(xmin) {
            return false;
        }

        // xmin is committed and started before us, so tuple was validly inserted.

        // Case 3: Check deletion
        if xmax == 0 {
            // Not deleted
            return true;
        }

        // Case 4: Check if deleting transaction is visible to us
        if xmax == self.txn_id {
            // We deleted it ourselves
            return false;
        }

        if xmax >= self.txn_id {
            // Deleting txn started after us, so deletion is invisible. Tuple is still visible.
            return true;
        }

        if self.is_txn_active(xmax) {
            // Deleting txn is still active (uncommitted), so deletion is invisible. Tuple is still visible.
            return true;
        }

        // Deleting txn committed before our snapshot. Tuple is deleted (invisible).
        false
    }

    /// Returns true if the given transaction ID was active at snapshot time.
    /// Uses binary search on the sorted active list.
    #[inline]
    pub fn is_txn_active(&self, txn_id: u64) -> bool {
        self.active_txn_ids.binary_search(&txn_id).is_ok()
    }

    /// Returns the number of active transactions at snapshot time.
    pub fn active_count(&self) -> usize {
        self.active_txn_ids.len()
    }

    /// Returns the active transaction IDs.
    pub fn active_txn_ids(&self) -> &[u64] {
        &self.active_txn_ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_own_insert_visible() {
        let snapshot = Snapshot::new(10, vec![]);
        // xmin=10 (own txn), xmax=0 (not deleted)
        assert!(snapshot.is_visible(10, 0));
    }

    #[test]
    fn test_own_insert_then_delete_invisible() {
        let snapshot = Snapshot::new(10, vec![]);
        // xmin=10 (own txn), xmax=10 (self-deleted)
        assert!(!snapshot.is_visible(10, 10));
    }

    #[test]
    fn test_committed_insert_visible() {
        let snapshot = Snapshot::new(10, vec![]);
        // xmin=5 (committed before us), xmax=0 (not deleted)
        assert!(snapshot.is_visible(5, 0));
    }

    #[test]
    fn test_active_insert_invisible() {
        let snapshot = Snapshot::new(10, vec![5]);
        // xmin=5 (still active at snapshot time), xmax=0
        assert!(!snapshot.is_visible(5, 0));
    }

    #[test]
    fn test_future_insert_invisible() {
        let snapshot = Snapshot::new(10, vec![]);
        // xmin=15 (started after us), xmax=0
        assert!(!snapshot.is_visible(15, 0));
    }

    #[test]
    fn test_committed_delete_invisible() {
        let snapshot = Snapshot::new(10, vec![]);
        // xmin=3 (committed), xmax=7 (committed delete before us)
        assert!(!snapshot.is_visible(3, 7));
    }

    #[test]
    fn test_active_delete_still_visible() {
        let snapshot = Snapshot::new(10, vec![7]);
        // xmin=3 (committed), xmax=7 (active, so deletion not yet committed)
        assert!(snapshot.is_visible(3, 7));
    }

    #[test]
    fn test_future_delete_still_visible() {
        let snapshot = Snapshot::new(10, vec![]);
        // xmin=3 (committed), xmax=15 (started after us, deletion invisible)
        assert!(snapshot.is_visible(3, 15));
    }

    #[test]
    fn test_is_txn_active_binary_search() {
        let snapshot = Snapshot::new(100, vec![5, 10, 20, 50]);

        assert!(snapshot.is_txn_active(5));
        assert!(snapshot.is_txn_active(10));
        assert!(snapshot.is_txn_active(20));
        assert!(snapshot.is_txn_active(50));

        assert!(!snapshot.is_txn_active(1));
        assert!(!snapshot.is_txn_active(15));
        assert!(!snapshot.is_txn_active(100));
    }

    #[test]
    fn test_empty_active_set() {
        let snapshot = Snapshot::new(10, vec![]);
        assert_eq!(snapshot.active_count(), 0);
        assert!(!snapshot.is_txn_active(5));
    }

    #[test]
    fn test_snapshot_sorts_active_ids() {
        let snapshot = Snapshot::new(100, vec![50, 10, 30, 20]);
        assert_eq!(snapshot.active_txn_ids(), &[10, 20, 30, 50]);
    }

    #[test]
    fn test_complex_visibility_scenario() {
        // Txn 10 sees committed txns 1-5, active txns 6,8, committed txn 7
        let snapshot = Snapshot::new(10, vec![6, 8]);

        // Committed insert by txn 3, not deleted
        assert!(snapshot.is_visible(3, 0));

        // Committed insert by txn 3, deleted by committed txn 7
        assert!(!snapshot.is_visible(3, 7));

        // Committed insert by txn 3, deleted by active txn 6 (not yet committed)
        assert!(snapshot.is_visible(3, 6));

        // Active insert by txn 6 (not committed)
        assert!(!snapshot.is_visible(6, 0));

        // Insert by txn 8 (active, not committed)
        assert!(!snapshot.is_visible(8, 0));
    }
}

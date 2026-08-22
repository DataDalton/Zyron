//! MVCC snapshot for transaction visibility.
//!
//! A Snapshot captures the set of active transactions at BEGIN time.
//! It provides visibility checks following standard MVCC rules:
//! - Tuples inserted by committed transactions before the snapshot are visible.
//! - Tuples inserted by active (uncommitted) transactions are invisible.
//! - Tuples deleted by committed transactions before the snapshot are invisible.
//! - Tuples inserted or deleted by the owning transaction follow self-visibility rules.

use std::sync::Arc;

use super::status_map::TxnStatusMap;

/// Immutable snapshot of active transactions taken at BEGIN time.
///
/// Used for MVCC visibility checks. The active_txn_ids list is sorted
/// for binary search during visibility checks. The shared commit-status map
/// lets visibility distinguish a committed transaction's writes from an aborted
/// one's once the transaction has left the active set.
#[derive(Debug, Clone)]
pub struct Snapshot {
    /// Transaction ID of the owning transaction.
    pub txn_id: u64,
    /// Sorted list of transaction IDs that were active at snapshot time.
    active_txn_ids: Vec<u64>,
    /// Shared transaction commit-status map (cheap Arc clone per snapshot).
    status: Arc<TxnStatusMap>,
    /// Every transaction below this id is committed and ended before this
    /// snapshot, so visibility resolves it with a single comparison and no
    /// status-map lookup. Captured once at snapshot creation.
    frozen_below: u64,
}

impl Snapshot {
    /// Creates a new snapshot with the given active transaction set and the
    /// shared commit-status map.
    ///
    /// `active_txns` must already be sorted in ascending order; the visibility
    /// fast paths use binary search and corruption is silent if the invariant
    /// is broken. The transaction manager is the only production producer and
    /// returns a pre-sorted vec, so this constructor does not re-sort.
    pub fn new(txn_id: u64, active_txns: Vec<u64>, status: Arc<TxnStatusMap>) -> Self {
        debug_assert!(
            active_txns.windows(2).all(|w| w[0] <= w[1]),
            "Snapshot::new requires a sorted active_txn list"
        );
        // Oldest transaction still in flight (or this txn if none): every txn
        // below the frozen horizon committed and ended before this snapshot.
        let oldest_active = active_txns.first().copied().unwrap_or(txn_id);
        let frozen_below = status.frozen_below(oldest_active);
        Self {
            txn_id,
            active_txn_ids: active_txns,
            status,
            frozen_below,
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

        // Invisible if xmin started after us.
        if xmin >= self.txn_id {
            return false;
        }

        // Resolve whether xmin committed before this snapshot. The frozen-horizon
        // fast path (a single comparison, no atomic load) covers the dominant
        // case of scanning rows older than any active transaction; only newer
        // rows fall back to the active-set + status-map check.
        let xmin_committed = if xmin < self.frozen_below {
            true
        } else if self.is_txn_active(xmin) {
            return false;
        } else {
            self.status.is_committed(xmin)
        };
        if !xmin_committed {
            return false;
        }

        // Case 3: not deleted.
        if xmax == 0 {
            return true;
        }

        // Case 4: is the deletion visible to us?
        if xmax == self.txn_id {
            // We deleted it ourselves.
            return false;
        }
        if xmax >= self.txn_id {
            // Deleter started after us: deletion invisible, tuple still visible.
            return true;
        }
        // Deleter ended before our snapshot. Below the frozen horizon it is
        // committed (the tuple is deleted); otherwise consult the active set and
        // status map. An aborted delete leaves the tuple visible.
        if xmax < self.frozen_below {
            return false;
        }
        if self.is_txn_active(xmax) {
            return true;
        }
        !self.status.is_committed(xmax)
    }

    /// Returns true if a tuple is live as of NOW rather than as of this
    /// snapshot, from the point of view of transaction `self_txn`.
    ///
    /// Used by unique-constraint enforcement, which must consider the latest
    /// committed state rather than the inserting transaction's frozen
    /// snapshot, and must also see what `self_txn` has already done. Its own
    /// writes are not committed yet but are certain to it: a row it inserted
    /// is a conflict for its next statement, and a row it deleted is free for
    /// that statement to reuse the key of. Judging by commit status alone
    /// would get both backwards, admitting a duplicate written twice in one
    /// transaction and refusing a delete-then-reinsert of the same key.
    #[inline]
    pub fn is_live_latest(&self, xmin: u64, xmax: u64, self_txn: u64) -> bool {
        let inserted = xmin == self_txn || self.status.is_committed(xmin);
        let deleted = xmax != 0 && (xmax == self_txn || self.status.is_committed(xmax));
        inserted && !deleted
    }

    /// Returns the prune horizon for this snapshot: every transaction below it
    /// committed and ended before the oldest transaction active when this
    /// snapshot was taken. A version with `xmax` below this horizon is a
    /// committed delete invisible to every live snapshot, so on-access pruning
    /// can reclaim it. Because the active set is system-wide, this horizon is
    /// globally safe, not merely safe for this snapshot.
    #[inline]
    pub fn prune_horizon(&self) -> u64 {
        self.frozen_below
    }

    /// Returns the shared commit-status map, so callers that prune dead tuples
    /// can test whether an inserter aborted.
    #[inline]
    pub fn status_map(&self) -> &Arc<TxnStatusMap> {
        &self.status
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
    use crate::txn::status_map::TxnStatusMap;

    /// A status map that reports every transaction committed, for tests that do
    /// not exercise abort visibility.
    fn committed() -> Arc<TxnStatusMap> {
        Arc::new(TxnStatusMap::all_committed())
    }

    /// A real status map seeded with explicit committed and aborted ids.
    fn status(commit: &[u64], abort: &[u64]) -> Arc<TxnStatusMap> {
        let m = TxnStatusMap::new();
        for &c in commit {
            m.record_committed(c);
        }
        for &a in abort {
            m.record_aborted(a);
        }
        Arc::new(m)
    }

    #[test]
    fn test_own_insert_visible() {
        let snapshot = Snapshot::new(10, vec![], committed());
        assert!(snapshot.is_visible(10, 0));
    }

    #[test]
    fn test_own_insert_then_delete_invisible() {
        let snapshot = Snapshot::new(10, vec![], committed());
        assert!(!snapshot.is_visible(10, 10));
    }

    #[test]
    fn test_committed_insert_visible() {
        let snapshot = Snapshot::new(10, vec![], committed());
        assert!(snapshot.is_visible(5, 0));
    }

    #[test]
    fn test_active_insert_invisible() {
        let snapshot = Snapshot::new(10, vec![5], committed());
        assert!(!snapshot.is_visible(5, 0));
    }

    #[test]
    fn test_future_insert_invisible() {
        let snapshot = Snapshot::new(10, vec![], committed());
        assert!(!snapshot.is_visible(15, 0));
    }

    #[test]
    fn test_committed_delete_invisible() {
        let snapshot = Snapshot::new(10, vec![], committed());
        assert!(!snapshot.is_visible(3, 7));
    }

    #[test]
    fn test_active_delete_still_visible() {
        let snapshot = Snapshot::new(10, vec![7], committed());
        assert!(snapshot.is_visible(3, 7));
    }

    #[test]
    fn test_future_delete_still_visible() {
        let snapshot = Snapshot::new(10, vec![], committed());
        assert!(snapshot.is_visible(3, 15));
    }

    // An aborted inserter's row is invisible even though it left the active set.
    #[test]
    fn test_aborted_insert_invisible() {
        let snapshot = Snapshot::new(10, vec![], status(&[], &[5]));
        assert!(!snapshot.is_visible(5, 0));
    }

    // A row whose deleter aborted stays visible (the delete did not happen).
    #[test]
    fn test_aborted_delete_keeps_row_visible() {
        let snapshot = Snapshot::new(10, vec![], status(&[3], &[7]));
        assert!(snapshot.is_visible(3, 7));
    }

    // A committed insert later deleted by an aborted txn remains visible.
    #[test]
    fn test_committed_insert_aborted_delete_visible() {
        let snapshot = Snapshot::new(10, vec![], status(&[2], &[8]));
        assert!(snapshot.is_visible(2, 8));
        // Same row deleted by a committed txn is invisible.
        let snapshot2 = Snapshot::new(10, vec![], status(&[2, 8], &[]));
        assert!(!snapshot2.is_visible(2, 8));
    }

    // A transaction at or above the frozen horizon that never recorded a commit
    // (e.g. in flight) is treated as not committed, so its inserts are invisible.
    // An early abort (id 1) caps the frozen horizon so id 5 is above it and must
    // take the status-map path rather than the frozen fast path.
    #[test]
    fn test_unrecorded_insert_invisible() {
        let snapshot = Snapshot::new(10, vec![], status(&[], &[1]));
        assert!(!snapshot.is_visible(5, 0));
    }

    // The frozen-horizon fast path: a transaction below the horizon is treated
    // as committed without a status-map entry (recovery records every ended
    // transaction, so nothing below the horizon is silently uncommitted).
    #[test]
    fn test_frozen_below_horizon_visible() {
        let snapshot = Snapshot::new(10, vec![], status(&[], &[]));
        // No aborts and no active txns: horizon is the snapshot txn id, so id 5
        // resolves visible via the fast path.
        assert!(snapshot.is_visible(5, 0));
    }

    #[test]
    fn test_is_txn_active_binary_search() {
        let snapshot = Snapshot::new(100, vec![5, 10, 20, 50], committed());
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
        let snapshot = Snapshot::new(10, vec![], committed());
        assert_eq!(snapshot.active_count(), 0);
        assert!(!snapshot.is_txn_active(5));
    }

    #[test]
    fn test_snapshot_preserves_sorted_input() {
        let snapshot = Snapshot::new(100, vec![10, 20, 30, 50], committed());
        assert_eq!(snapshot.active_txn_ids(), &[10, 20, 30, 50]);
    }

    #[test]
    fn test_complex_visibility_scenario() {
        let snapshot = Snapshot::new(10, vec![6, 8], committed());
        assert!(snapshot.is_visible(3, 0));
        assert!(!snapshot.is_visible(3, 7));
        assert!(snapshot.is_visible(3, 6));
        assert!(!snapshot.is_visible(6, 0));
        assert!(!snapshot.is_visible(8, 0));
    }
}

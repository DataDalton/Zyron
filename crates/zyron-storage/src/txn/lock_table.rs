//! Row-level write-write conflict detection.
//!
//! The LockTable tracks which transactions hold write locks on specific
//! heap rows. If two transactions attempt to write the same row,
//! the second one receives a TransactionConflict error.

use crate::tuple::TupleId;
use zyron_common::{Result, ZyronError};

/// Composite key for a row lock: (table_id, page_num, slot_id).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct RowLockKey {
    table_id: u32,
    page_num: u64,
    slot_id: u16,
}

impl RowLockKey {
    fn new(table_id: u32, rid: TupleId) -> Self {
        Self {
            table_id,
            page_num: rid.page_id.page_num,
            slot_id: rid.slot_id,
        }
    }
}

/// Row-level lock table for write-write conflict detection.
///
/// Uses scc::HashMap for lock-free concurrent access. Each entry maps
/// a (table_id, page_num, slot_id) composite key to the txn_id holding
/// the lock. A per-txn inverse map tracks which keys each transaction
/// holds, enabling O(k) unlock instead of O(n) full-table scan where
/// k = locks held by the transaction and n = total locks in the table.
pub struct LockTable {
    /// Maps row lock keys to the holding txn_id.
    locks: scc::HashMap<RowLockKey, u64>,
    /// Inverse map: txn_id -> keys held by that transaction.
    /// Enables O(k) unlock_all instead of O(n) retain_sync.
    txn_locks: scc::HashMap<u64, Vec<RowLockKey>>,
}

impl LockTable {
    /// Creates a new empty lock table.
    pub fn new() -> Self {
        Self {
            locks: scc::HashMap::new(),
            txn_locks: scc::HashMap::new(),
        }
    }

    /// Acquires a write lock on a row for the given transaction.
    ///
    /// Returns Ok(()) if the lock was acquired or was already held by the same txn.
    /// Returns TransactionConflict if another transaction holds the lock.
    pub fn lock_row(&self, txn_id: u64, table_id: u32, rid: TupleId) -> Result<()> {
        let key = RowLockKey::new(table_id, rid);

        // Try to insert the lock. If the key already exists, check the holder.
        match self.locks.entry_sync(key) {
            scc::hash_map::Entry::Occupied(entry) => {
                let holder = *entry.get();
                if holder == txn_id {
                    // Already held by same txn (idempotent)
                    Ok(())
                } else {
                    Err(ZyronError::TransactionConflict {
                        txn_id,
                        reason: format!(
                            "row {}:{} locked by txn {}",
                            rid.page_id, rid.slot_id, holder
                        ),
                    })
                }
            }
            scc::hash_map::Entry::Vacant(entry) => {
                entry.insert_entry(txn_id);
                // Track in inverse map for O(k) unlock
                self.txn_locks
                    .entry_sync(txn_id)
                    .or_default()
                    .get_mut()
                    .push(key);
                Ok(())
            }
        }
    }

    /// Releases all locks held by a transaction.
    /// Uses the per-txn inverse map for O(k) removal where k = locks held,
    /// instead of O(n) full-table scan.
    pub fn unlock_all(&self, txn_id: u64) {
        if let Some((_, keys)) = self.txn_locks.remove_sync(&txn_id) {
            for key in keys {
                let _ = self.locks.remove_sync(&key);
            }
        }
    }

    /// Returns the txn_id holding the lock on a row, if any.
    pub fn is_locked_by(&self, table_id: u32, rid: TupleId) -> Option<u64> {
        let key = RowLockKey::new(table_id, rid);
        self.locks.get_sync(&key).map(|entry| *entry.get())
    }

    /// Returns the number of active locks.
    pub fn lock_count(&self) -> usize {
        self.locks.len()
    }
}

impl Default for LockTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyron_common::page::PageId;

    fn make_rid(page_num: u64, slot_id: u16) -> TupleId {
        TupleId::new(PageId::new(0, page_num), slot_id)
    }

    #[test]
    fn test_lock_row_success() {
        let table = LockTable::new();
        let rid = make_rid(1, 0);
        assert!(table.lock_row(1, 0, rid).is_ok());
        assert_eq!(table.lock_count(), 1);
    }

    #[test]
    fn test_lock_row_idempotent() {
        let table = LockTable::new();
        let rid = make_rid(1, 0);
        table.lock_row(1, 0, rid).unwrap();
        // Same txn locking same row is idempotent
        assert!(table.lock_row(1, 0, rid).is_ok());
        assert_eq!(table.lock_count(), 1);
    }

    #[test]
    fn test_lock_row_conflict() {
        let table = LockTable::new();
        let rid = make_rid(1, 0);
        table.lock_row(1, 0, rid).unwrap();

        // Different txn trying to lock same row
        let result = table.lock_row(2, 0, rid);
        assert!(result.is_err());
        match result.unwrap_err() {
            ZyronError::TransactionConflict { txn_id, .. } => {
                assert_eq!(txn_id, 2);
            }
            other => panic!("expected TransactionConflict, got: {:?}", other),
        }
    }

    #[test]
    fn test_lock_different_rows() {
        let table = LockTable::new();
        let rid1 = make_rid(1, 0);
        let rid2 = make_rid(1, 1);
        let rid3 = make_rid(2, 0);

        table.lock_row(1, 0, rid1).unwrap();
        table.lock_row(2, 0, rid2).unwrap();
        table.lock_row(3, 0, rid3).unwrap();

        assert_eq!(table.lock_count(), 3);
    }

    #[test]
    fn test_unlock_all() {
        let table = LockTable::new();
        let rid1 = make_rid(1, 0);
        let rid2 = make_rid(1, 1);
        let rid3 = make_rid(2, 0);

        table.lock_row(1, 0, rid1).unwrap();
        table.lock_row(1, 0, rid2).unwrap();
        table.lock_row(2, 0, rid3).unwrap();

        assert_eq!(table.lock_count(), 3);

        table.unlock_all(1);
        assert_eq!(table.lock_count(), 1);

        // Row previously locked by txn 1 is now available
        assert!(table.lock_row(3, 0, rid1).is_ok());
    }

    #[test]
    fn test_is_locked_by() {
        let table = LockTable::new();
        let rid = make_rid(1, 0);

        assert!(table.is_locked_by(0, rid).is_none());

        table.lock_row(42, 0, rid).unwrap();
        assert_eq!(table.is_locked_by(0, rid), Some(42));
    }

    #[test]
    fn test_different_tables_independent() {
        let table = LockTable::new();
        let rid = make_rid(1, 0);

        // Same row in different tables should not conflict
        table.lock_row(1, 0, rid).unwrap();
        assert!(table.lock_row(2, 1, rid).is_ok());
        assert_eq!(table.lock_count(), 2);
    }
}

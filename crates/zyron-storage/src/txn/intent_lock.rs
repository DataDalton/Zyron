//! B+Tree key-level write-write conflict detection.
//!
//! The IntentLockTable tracks which transactions hold write locks on
//! specific B+Tree keys. Separate from the row-level LockTable because
//! intent locks protect index keys while LockTable protects heap rows.
//! Both must be acquired for a transactional write.

use zyron_common::{Result, ZyronError};

/// Intent lock table for B+Tree key-level conflict detection.
///
/// Uses scc::HashMap for lock-free concurrent access. Each entry maps
/// a hashed (table_id, key) pair to the txn_id holding the lock.
/// Key hashing uses xxh3 for fast, high-quality distribution.
pub struct IntentLockTable {
    /// Maps hashed intent lock keys to the holding txn_id.
    /// Key: xxh3 hash of (table_id, key_bytes), Value: txn_id.
    locks: scc::HashMap<u64, u64>,
    /// Inverse map: txn_id -> hashed keys held by that transaction.
    /// Enables O(k) unlock_all instead of O(n) retain_sync.
    txn_locks: scc::HashMap<u64, Vec<u64>>,
}

impl IntentLockTable {
    /// Creates a new empty intent lock table.
    pub fn new() -> Self {
        Self {
            locks: scc::HashMap::new(),
            txn_locks: scc::HashMap::new(),
        }
    }

    /// Acquires an intent lock on a B+Tree key for the given transaction.
    ///
    /// Returns Ok(()) if the lock was acquired or was already held by the same txn.
    /// Returns TransactionConflict if another transaction holds the lock.
    pub fn lock_key(&self, txn_id: u64, table_id: u32, key: &[u8]) -> Result<()> {
        let hash = Self::hash_key(table_id, key);

        match self.locks.entry_sync(hash) {
            scc::hash_map::Entry::Occupied(entry) => {
                let holder = *entry.get();
                if holder == txn_id {
                    // Already held by same txn (idempotent)
                    Ok(())
                } else {
                    Err(ZyronError::TransactionConflict {
                        txn_id,
                        reason: format!("index key in table {} locked by txn {}", table_id, holder),
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
                    .push(hash);
                Ok(())
            }
        }
    }

    /// Releases all intent locks held by a transaction.
    /// Uses the per-txn inverse map for O(k) removal where k = locks held,
    /// instead of O(n) full-table scan.
    pub fn unlock_all(&self, txn_id: u64) {
        if let Some((_, keys)) = self.txn_locks.remove_sync(&txn_id) {
            for key in keys {
                let _ = self.locks.remove_sync(&key);
            }
        }
    }

    /// Returns the txn_id holding the intent lock on a key, if any.
    pub fn is_locked_by(&self, table_id: u32, key: &[u8]) -> Option<u64> {
        let hash = Self::hash_key(table_id, key);
        self.locks.read_sync(&hash, |_, v| *v)
    }

    /// Returns the number of active intent locks.
    pub fn lock_count(&self) -> usize {
        self.locks.len()
    }

    /// Hashes a (table_id, key) pair to a u64 using the central hash primitive.
    #[inline]
    fn hash_key(table_id: u32, key: &[u8]) -> u64 {
        // Seed with table_id so identical keys in different tables produce different hashes.
        zyron_common::hash64_seeded(key, table_id as u64)
    }
}

impl Default for IntentLockTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lock_key_success() {
        let table = IntentLockTable::new();
        assert!(table.lock_key(1, 0, b"key1").is_ok());
        assert_eq!(table.lock_count(), 1);
    }

    #[test]
    fn test_lock_key_idempotent() {
        let table = IntentLockTable::new();
        table.lock_key(1, 0, b"key1").unwrap();
        // Same txn locking same key is idempotent
        assert!(table.lock_key(1, 0, b"key1").is_ok());
        assert_eq!(table.lock_count(), 1);
    }

    #[test]
    fn test_lock_key_conflict() {
        let table = IntentLockTable::new();
        table.lock_key(1, 0, b"key1").unwrap();

        // Different txn trying to lock same key
        let result = table.lock_key(2, 0, b"key1");
        assert!(result.is_err());
        match result.unwrap_err() {
            ZyronError::TransactionConflict { txn_id, .. } => {
                assert_eq!(txn_id, 2);
            }
            other => panic!("expected TransactionConflict, got: {:?}", other),
        }
    }

    #[test]
    fn test_lock_different_keys() {
        let table = IntentLockTable::new();
        table.lock_key(1, 0, b"key1").unwrap();
        table.lock_key(2, 0, b"key2").unwrap();
        table.lock_key(3, 0, b"key3").unwrap();

        assert_eq!(table.lock_count(), 3);
    }

    #[test]
    fn test_unlock_all() {
        let table = IntentLockTable::new();
        table.lock_key(1, 0, b"key1").unwrap();
        table.lock_key(1, 0, b"key2").unwrap();
        table.lock_key(2, 0, b"key3").unwrap();

        assert_eq!(table.lock_count(), 3);

        table.unlock_all(1);
        assert_eq!(table.lock_count(), 1);

        // Key previously locked by txn 1 is now available
        assert!(table.lock_key(3, 0, b"key1").is_ok());
    }

    #[test]
    fn test_is_locked_by() {
        let table = IntentLockTable::new();

        assert!(table.is_locked_by(0, b"key1").is_none());

        table.lock_key(42, 0, b"key1").unwrap();
        assert_eq!(table.is_locked_by(0, b"key1"), Some(42));
    }

    #[test]
    fn test_different_tables_independent() {
        let table = IntentLockTable::new();

        // Same key in different tables should not conflict
        table.lock_key(1, 0, b"key1").unwrap();
        assert!(table.lock_key(2, 1, b"key1").is_ok());
        assert_eq!(table.lock_count(), 2);
    }

    #[test]
    fn test_empty_key() {
        let table = IntentLockTable::new();
        assert!(table.lock_key(1, 0, b"").is_ok());
        assert_eq!(table.lock_count(), 1);
    }

    #[test]
    fn test_large_key() {
        let table = IntentLockTable::new();
        let large_key = vec![0xFFu8; 1024];
        assert!(table.lock_key(1, 0, &large_key).is_ok());
        assert_eq!(table.is_locked_by(0, &large_key), Some(1));
    }
}

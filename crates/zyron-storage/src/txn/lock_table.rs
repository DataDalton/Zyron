//! Row level lock table for FOR UPDATE/SHARE and DML write-write conflicts.
//!
//! Keys on RowLocator so heap, columnar and lake resident rows all lock
//! through one table. Exclusive locks are held by a single transaction,
//! shared locks by any number. DML takes exclusive locks before stamping
//! or patching a row, SELECT FOR UPDATE takes exclusive, FOR SHARE takes
//! shared. Locks release at commit or abort via unlock_all, ROLLBACK TO
//! SAVEPOINT releases the tail via unlock_after.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Notify;
use zyron_common::{Result, RowLocator, ZyronError};

use super::deadlock::WaitForGraph;

/// Upper bound on a blocking lock wait before the request fails with a
/// conflict error, backstops deadlocks between mutually blocked waiters
const LOCK_WAIT_TIMEOUT: Duration = Duration::from_secs(10);

/// Lock strength requested for one row
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockMode {
    /// Compatible with other shared holders, blocks exclusive
    Shared,
    /// Sole holder, blocks every other transaction
    Exclusive,
}

/// Composite key, the table plus the storage agnostic row identity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct RowLockKey {
    table_id: u32,
    locator: RowLocator,
}

/// Current holders of one row lock. A Shared entry never persists empty,
/// release removes the map entry when the last holder leaves
#[derive(Debug)]
enum LockState {
    Exclusive(u64),
    Shared(Vec<u64>),
}

/// Row-level lock table shared by SELECT FOR UPDATE/SHARE and the DML
/// write paths.
///
/// Uses scc::HashMap for lock-free concurrent access. Each entry maps a
/// (table_id, RowLocator) composite key to its holder set. A per-txn
/// inverse map tracks which keys each transaction holds, enabling O(k)
/// unlock instead of O(n) full-table scan where k = locks held by the
/// transaction and n = total locks in the table. Waiters park on a
/// per-key Notify that release paths fire, so a blocking FOR UPDATE
/// wakes as soon as the holder commits or aborts.
pub struct LockTable {
    /// Maps row lock keys to their holder set
    locks: scc::HashMap<RowLockKey, LockState>,
    /// Inverse map: txn_id -> keys the transaction appears on, in
    /// acquisition order so savepoint release can truncate the tail
    txn_locks: scc::HashMap<u64, Vec<RowLockKey>>,
    /// Park points for blocked acquisitions, one Notify per contended key,
    /// removed when a release wakes the key so idle keys carry no state
    waiters: scc::HashMap<RowLockKey, Arc<Notify>>,
    /// Wait-for graph fed by blocking acquisitions. Shared with the
    /// transaction manager, which clears a txn's edges at commit and abort
    wait_graph: Arc<WaitForGraph>,
}

impl LockTable {
    /// Creates a new empty lock table.
    pub fn new() -> Self {
        Self {
            locks: scc::HashMap::new(),
            txn_locks: scc::HashMap::new(),
            waiters: scc::HashMap::new(),
            wait_graph: Arc::new(WaitForGraph::new()),
        }
    }

    /// The wait-for graph blocking acquisitions feed. The transaction
    /// manager shares this instance so commit and abort clear a txn's edges.
    pub fn wait_graph(&self) -> &Arc<WaitForGraph> {
        &self.wait_graph
    }

    /// Acquires a row lock without waiting.
    ///
    /// Returns Ok(()) if the lock was granted or the transaction already
    /// holds it at sufficient strength. Returns TransactionConflict naming
    /// the holder if another transaction blocks it. This is the NOWAIT
    /// path and the DML write path.
    pub fn lock_row(
        &self,
        txn_id: u64,
        table_id: u32,
        locator: RowLocator,
        mode: LockMode,
    ) -> Result<()> {
        let key = RowLockKey { table_id, locator };
        match self.try_acquire(txn_id, key, mode) {
            Ok(()) => Ok(()),
            Err(holder) => Err(ZyronError::TransactionConflict {
                txn_id,
                reason: format!("row {locator} in table {table_id} locked by txn {holder}"),
            }),
        }
    }

    /// Attempts a row lock without waiting, reporting the blocking holder's
    /// txn_id on refusal. Used by the row-locking operator, which needs the
    /// holder to decide whether a completed wait means the row changed.
    pub fn lock_row_or_holder(
        &self,
        txn_id: u64,
        table_id: u32,
        locator: RowLocator,
        mode: LockMode,
    ) -> std::result::Result<(), u64> {
        self.try_acquire(txn_id, RowLockKey { table_id, locator }, mode)
    }

    /// Attempts a row lock without waiting, reporting success as a bool.
    /// This is the SKIP LOCKED path, a refusal filters the row out of the
    /// result instead of erroring.
    pub fn try_lock_row(
        &self,
        txn_id: u64,
        table_id: u32,
        locator: RowLocator,
        mode: LockMode,
    ) -> bool {
        self.try_acquire(txn_id, RowLockKey { table_id, locator }, mode)
            .is_ok()
    }

    /// Acquires a row lock, parking until the holder releases it or the
    /// wait times out. Registers on the key's Notify before each re-check
    /// so a release landing between the check and the await still wakes
    /// this waiter. Each park registers a waiter-to-holder edge in the
    /// wait-for graph, an edge that closes a cycle fails this request
    /// immediately, the requester breaking the cycle it would complete.
    /// The timeout backstops anything the graph misses.
    pub async fn lock_row_wait(
        &self,
        txn_id: u64,
        table_id: u32,
        locator: RowLocator,
        mode: LockMode,
    ) -> Result<()> {
        let key = RowLockKey { table_id, locator };
        if self.try_acquire(txn_id, key, mode).is_ok() {
            return Ok(());
        }
        let mut holder;
        let deadline = tokio::time::Instant::now() + LOCK_WAIT_TIMEOUT;
        loop {
            let notify = self.waiter_handle(key);
            let notified = notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            match self.try_acquire(txn_id, key, mode) {
                Ok(()) => {
                    self.wait_graph.remove_edge(txn_id);
                    return Ok(());
                }
                Err(h) => holder = h,
            }
            // re-point the edge at the current holder, add_edge does not
            // replace an existing edge so the stale one is removed first
            self.wait_graph.remove_edge(txn_id);
            if self.wait_graph.add_edge(txn_id, holder).is_some() {
                return Err(ZyronError::TransactionConflict {
                    txn_id,
                    reason: format!(
                        "deadlock detected, txn {txn_id} waiting on row {locator} in table {table_id} held by txn {holder} closes a wait cycle"
                    ),
                });
            }
            if tokio::time::timeout_at(deadline, notified).await.is_err() {
                self.wait_graph.remove_edge(txn_id);
                return Err(ZyronError::TransactionConflict {
                    txn_id,
                    reason: format!(
                        "lock wait timeout on row {locator} in table {table_id} held by txn {holder}"
                    ),
                });
            }
        }
    }

    /// Core acquisition. Ok on grant or already-held-at-strength, Err with
    /// a blocking holder's txn_id otherwise. A sole shared holder upgrades
    /// in place to exclusive. An exclusive holder satisfies a shared
    /// request without a second entry.
    fn try_acquire(
        &self,
        txn_id: u64,
        key: RowLockKey,
        mode: LockMode,
    ) -> std::result::Result<(), u64> {
        match self.locks.entry_sync(key) {
            scc::hash_map::Entry::Occupied(mut entry) => {
                let state = entry.get_mut();
                match state {
                    LockState::Exclusive(holder) => {
                        if *holder == txn_id {
                            Ok(())
                        } else {
                            Err(*holder)
                        }
                    }
                    LockState::Shared(holders) => match mode {
                        LockMode::Shared => {
                            if !holders.contains(&txn_id) {
                                holders.push(txn_id);
                                self.track(txn_id, key);
                            }
                            Ok(())
                        }
                        LockMode::Exclusive => {
                            if holders.len() == 1 && holders[0] == txn_id {
                                *state = LockState::Exclusive(txn_id);
                                Ok(())
                            } else {
                                let blocker = holders
                                    .iter()
                                    .copied()
                                    .find(|h| *h != txn_id)
                                    .unwrap_or(txn_id);
                                Err(blocker)
                            }
                        }
                    },
                }
            }
            scc::hash_map::Entry::Vacant(entry) => {
                entry.insert_entry(match mode {
                    LockMode::Exclusive => LockState::Exclusive(txn_id),
                    LockMode::Shared => LockState::Shared(vec![txn_id]),
                });
                self.track(txn_id, key);
                Ok(())
            }
        }
    }

    /// Records the key in the transaction's inverse list, exactly once per
    /// (txn, key). Callers only invoke this on the txn's first appearance
    /// on the key, an in-place shared-to-exclusive upgrade keeps the
    /// original entry.
    fn track(&self, txn_id: u64, key: RowLockKey) {
        self.txn_locks
            .entry_sync(txn_id)
            .or_default()
            .get_mut()
            .push(key);
    }

    /// Returns the per-key Notify waiters park on, creating it on first
    /// contention.
    fn waiter_handle(&self, key: RowLockKey) -> Arc<Notify> {
        match self.waiters.entry_sync(key) {
            scc::hash_map::Entry::Occupied(entry) => Arc::clone(entry.get()),
            scc::hash_map::Entry::Vacant(entry) => {
                let notify = Arc::new(Notify::new());
                entry.insert_entry(Arc::clone(&notify));
                notify
            }
        }
    }

    /// Removes the transaction from one key's holder set and wakes waiters
    /// if anything was released. Waiters holding the removed Notify still
    /// receive the wakeup, a later waiter creates a fresh one.
    fn release_key(&self, key: RowLockKey, txn_id: u64) {
        let mut released = false;
        if let scc::hash_map::Entry::Occupied(mut entry) = self.locks.entry_sync(key) {
            let mut drop_entry = false;
            match entry.get_mut() {
                LockState::Exclusive(holder) => {
                    if *holder == txn_id {
                        released = true;
                        drop_entry = true;
                    }
                }
                LockState::Shared(holders) => {
                    let before = holders.len();
                    holders.retain(|h| *h != txn_id);
                    released = holders.len() != before;
                    drop_entry = holders.is_empty();
                }
            }
            if drop_entry {
                let _ = entry.remove();
            }
        }
        if released && !self.waiters.is_empty() {
            if let Some((_, notify)) = self.waiters.remove_sync(&key) {
                notify.notify_waiters();
            }
        }
    }

    /// Releases all locks held by a transaction and wakes their waiters
    /// Uses the per-txn inverse map for O(k) removal where k = locks held,
    /// instead of O(n) full-table scan
    /// Fast-path checks the txn_locks size with a single atomic load before
    /// running a hash bucket lookup, this skips ~50ns of scc work per call
    /// for the common read-only case where the txn never acquired any locks
    pub fn unlock_all(&self, txn_id: u64) {
        if self.txn_locks.is_empty() {
            return;
        }
        if let Some((_, keys)) = self.txn_locks.remove_sync(&txn_id) {
            for key in keys {
                self.release_key(key, txn_id);
            }
        }
    }

    /// Returns the number of row locks the transaction currently holds. Captured
    /// at SAVEPOINT so a later ROLLBACK TO SAVEPOINT can release exactly the
    /// locks acquired after the savepoint.
    pub fn current_count(&self, txn_id: u64) -> usize {
        self.txn_locks
            .read_sync(&txn_id, |_, keys| keys.len())
            .unwrap_or(0)
    }

    /// Releases the row locks the transaction acquired after `keep`. The per-txn
    /// key list is in acquisition order, so the locks beyond index `keep` are the
    /// ones taken after the savepoint that recorded that count. A shared lock
    /// upgraded to exclusive after the savepoint stays exclusive, strength is
    /// never downgraded before commit.
    pub fn unlock_after(&self, txn_id: u64, keep: usize) {
        if self.txn_locks.is_empty() {
            return;
        }
        let mut dropped: Vec<RowLockKey> = Vec::new();
        self.txn_locks.update_sync(&txn_id, |_, keys| {
            if keep < keys.len() {
                dropped = keys.drain(keep..).collect();
            }
        });
        for key in dropped {
            self.release_key(key, txn_id);
        }
    }

    /// Returns a txn_id holding a lock on the row, if any. For a shared
    /// lock this is the first holder, callers use it for diagnostics only.
    pub fn is_locked_by(&self, table_id: u32, locator: RowLocator) -> Option<u64> {
        self.locks
            .read_sync(&RowLockKey { table_id, locator }, |_, state| match state {
                LockState::Exclusive(holder) => *holder,
                LockState::Shared(holders) => holders.first().copied().unwrap_or(0),
            })
    }

    /// Returns the number of rows currently carrying at least one lock.
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

    fn heap_loc(page_num: u64, slot: u16) -> RowLocator {
        RowLocator::Heap {
            page: PageId::new(0, page_num),
            slot,
        }
    }

    fn col_loc(file_id: u64, sys_rowid: u64) -> RowLocator {
        RowLocator::Columnar { file_id, sys_rowid }
    }

    #[test]
    fn test_lock_row_success() {
        let table = LockTable::new();
        assert!(
            table
                .lock_row(1, 0, heap_loc(1, 0), LockMode::Exclusive)
                .is_ok()
        );
        assert_eq!(table.lock_count(), 1);
    }

    #[test]
    fn test_lock_row_idempotent() {
        let table = LockTable::new();
        let loc = heap_loc(1, 0);
        table.lock_row(1, 0, loc, LockMode::Exclusive).unwrap();
        assert!(table.lock_row(1, 0, loc, LockMode::Exclusive).is_ok());
        assert_eq!(table.lock_count(), 1);
        assert_eq!(table.current_count(1), 1);
    }

    #[test]
    fn test_lock_row_conflict() {
        let table = LockTable::new();
        let loc = heap_loc(1, 0);
        table.lock_row(1, 0, loc, LockMode::Exclusive).unwrap();

        let result = table.lock_row(2, 0, loc, LockMode::Exclusive);
        assert!(result.is_err());
        match result.unwrap_err() {
            ZyronError::TransactionConflict { txn_id, .. } => {
                assert_eq!(txn_id, 2);
            }
            other => panic!("expected TransactionConflict, got: {:?}", other),
        }
    }

    #[test]
    fn test_columnar_locator_locks_independently_of_heap() {
        let table = LockTable::new();
        table
            .lock_row(1, 0, col_loc(3, 42), LockMode::Exclusive)
            .unwrap();
        // a heap row in the same table does not collide with the columnar key
        assert!(
            table
                .lock_row(2, 0, heap_loc(3, 42), LockMode::Exclusive)
                .is_ok()
        );
        // a second txn on the same columnar row conflicts
        assert!(
            table
                .lock_row(2, 0, col_loc(3, 42), LockMode::Exclusive)
                .is_err()
        );
        assert_eq!(table.is_locked_by(0, col_loc(3, 42)), Some(1));
    }

    #[test]
    fn test_shared_lock_compatibility_and_exclusive_block() {
        let table = LockTable::new();
        let loc = heap_loc(1, 0);
        table.lock_row(1, 0, loc, LockMode::Shared).unwrap();
        // second shared holder is compatible
        assert!(table.lock_row(2, 0, loc, LockMode::Shared).is_ok());
        // exclusive blocked while any other shared holder remains
        assert!(table.lock_row(3, 0, loc, LockMode::Exclusive).is_err());
        assert!(table.lock_row(1, 0, loc, LockMode::Exclusive).is_err());
        // once the other holder releases, the sole holder upgrades in place
        table.unlock_all(2);
        assert!(table.lock_row(1, 0, loc, LockMode::Exclusive).is_ok());
        assert!(table.lock_row(2, 0, loc, LockMode::Shared).is_err());
        assert_eq!(table.lock_count(), 1);
    }

    #[test]
    fn test_exclusive_holder_satisfies_shared_request() {
        let table = LockTable::new();
        let loc = heap_loc(1, 0);
        table.lock_row(1, 0, loc, LockMode::Exclusive).unwrap();
        assert!(table.lock_row(1, 0, loc, LockMode::Shared).is_ok());
        assert_eq!(table.current_count(1), 1);
    }

    #[test]
    fn test_try_lock_row_skip_semantics() {
        let table = LockTable::new();
        let loc = heap_loc(1, 0);
        table.lock_row(1, 0, loc, LockMode::Exclusive).unwrap();
        assert!(!table.try_lock_row(2, 0, loc, LockMode::Exclusive));
        assert!(table.try_lock_row(2, 0, heap_loc(1, 1), LockMode::Exclusive));
        assert_eq!(table.current_count(2), 1);
    }

    #[test]
    fn test_unlock_all_releases_shared_membership() {
        let table = LockTable::new();
        let loc = heap_loc(1, 0);
        table.lock_row(1, 0, loc, LockMode::Shared).unwrap();
        table.lock_row(2, 0, loc, LockMode::Shared).unwrap();
        table.unlock_all(1);
        // the entry survives with the remaining holder
        assert_eq!(table.lock_count(), 1);
        assert_eq!(table.is_locked_by(0, loc), Some(2));
        table.unlock_all(2);
        assert_eq!(table.lock_count(), 0);
    }

    #[test]
    fn test_unlock_all() {
        let table = LockTable::new();
        let l1 = heap_loc(1, 0);
        let l2 = heap_loc(1, 1);
        let l3 = heap_loc(2, 0);

        table.lock_row(1, 0, l1, LockMode::Exclusive).unwrap();
        table.lock_row(1, 0, l2, LockMode::Exclusive).unwrap();
        table.lock_row(2, 0, l3, LockMode::Exclusive).unwrap();

        assert_eq!(table.lock_count(), 3);

        table.unlock_all(1);
        assert_eq!(table.lock_count(), 1);

        assert!(table.lock_row(3, 0, l1, LockMode::Exclusive).is_ok());
    }

    #[test]
    fn test_unlock_after_releases_savepoint_tail() {
        let table = LockTable::new();
        table
            .lock_row(1, 0, heap_loc(1, 0), LockMode::Exclusive)
            .unwrap();
        let keep = table.current_count(1);
        table
            .lock_row(1, 0, heap_loc(1, 1), LockMode::Exclusive)
            .unwrap();
        table
            .lock_row(1, 0, col_loc(4, 7), LockMode::Exclusive)
            .unwrap();
        assert_eq!(table.current_count(1), keep + 2);

        table.unlock_after(1, keep);
        assert_eq!(table.current_count(1), keep);
        assert!(
            table
                .lock_row(2, 0, heap_loc(1, 1), LockMode::Exclusive)
                .is_ok()
        );
        assert!(
            table
                .lock_row(2, 0, col_loc(4, 7), LockMode::Exclusive)
                .is_ok()
        );
        assert!(
            table
                .lock_row(2, 0, heap_loc(1, 0), LockMode::Exclusive)
                .is_err()
        );
    }

    #[test]
    fn test_is_locked_by() {
        let table = LockTable::new();
        let loc = heap_loc(1, 0);

        assert!(table.is_locked_by(0, loc).is_none());

        table.lock_row(42, 0, loc, LockMode::Exclusive).unwrap();
        assert_eq!(table.is_locked_by(0, loc), Some(42));
    }

    #[test]
    fn test_different_tables_independent() {
        let table = LockTable::new();
        let loc = heap_loc(1, 0);

        table.lock_row(1, 0, loc, LockMode::Exclusive).unwrap();
        assert!(table.lock_row(2, 1, loc, LockMode::Exclusive).is_ok());
        assert_eq!(table.lock_count(), 2);
    }

    #[tokio::test]
    async fn test_lock_row_wait_wakes_on_release() {
        let table = std::sync::Arc::new(LockTable::new());
        let loc = heap_loc(9, 3);
        table.lock_row(1, 0, loc, LockMode::Exclusive).unwrap();

        let t2 = {
            let table = std::sync::Arc::clone(&table);
            tokio::spawn(async move { table.lock_row_wait(2, 0, loc, LockMode::Exclusive).await })
        };
        // let the waiter park, then release the holder
        tokio::time::sleep(Duration::from_millis(20)).await;
        table.unlock_all(1);
        t2.await
            .expect("join")
            .expect("waiter acquires after release");
        assert_eq!(table.is_locked_by(0, loc), Some(2));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_crossed_waiters_detect_deadlock() {
        let table = std::sync::Arc::new(LockTable::new());
        let r1 = heap_loc(1, 0);
        let r2 = heap_loc(1, 1);
        table.lock_row(1, 0, r1, LockMode::Exclusive).unwrap();
        table.lock_row(2, 0, r2, LockMode::Exclusive).unwrap();

        // txn 1 parks on r2, then txn 2 requests r1 and closes the cycle
        let w1 = {
            let table = std::sync::Arc::clone(&table);
            tokio::spawn(async move { table.lock_row_wait(1, 0, r2, LockMode::Exclusive).await })
        };
        tokio::time::sleep(Duration::from_millis(20)).await;
        let w2 = table.lock_row_wait(2, 0, r1, LockMode::Exclusive).await;
        let err = w2.expect_err("cycle-closing waiter fails fast");
        match err {
            ZyronError::TransactionConflict { reason, .. } => {
                assert!(reason.contains("deadlock"), "got: {reason}");
            }
            other => panic!("expected TransactionConflict, got: {:?}", other),
        }
        // the failed txn aborts, releasing its lock and waking the survivor
        table.unlock_all(2);
        table.wait_graph().remove_transaction(2);
        w1.await
            .expect("join")
            .expect("surviving waiter acquires after victim release");
    }

    #[tokio::test]
    async fn test_lock_row_wait_immediate_when_free() {
        let table = LockTable::new();
        table
            .lock_row_wait(5, 0, col_loc(1, 1), LockMode::Exclusive)
            .await
            .unwrap();
        assert_eq!(table.is_locked_by(0, col_loc(1, 1)), Some(5));
    }
}

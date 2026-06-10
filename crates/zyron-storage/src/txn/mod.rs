//! Transaction management for MVCC-based concurrency control.
//!
//! Provides snapshot isolation and write-write conflict detection.
//! Transaction logic lives inside zyron-storage because it needs direct
//! access to heap tuple headers (xmin/xmax), B+tree latch ordering,
//! and buffer pool pin/unpin.

mod deadlock;
mod durability;
mod gc;
mod intent_lock;
mod isolation;
mod lock_table;
mod proc_array;
mod snapshot;

pub use deadlock::WaitForGraph;
use durability::DurabilityQueue;
pub use gc::{GcStats, MvccGc};
pub use intent_lock::IntentLockTable;
pub use isolation::IsolationLevel;
pub use lock_table::LockTable;
pub use proc_array::ProcArray;
pub use snapshot::Snapshot;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use zyron_common::profile::{self, Phase};
use zyron_common::{Result, ZyronError};
use zyron_wal::WalWriter;
use zyron_wal::record::Lsn;

/// Status of a transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransactionStatus {
    /// Transaction is currently executing.
    Active,
    /// Transaction has been committed.
    Committed,
    /// Transaction has been aborted.
    Aborted,
}

/// A saved transaction state for partial rollback.
pub struct Savepoint {
    /// User-specified savepoint name.
    pub name: String,
    /// Snapshot at the time the savepoint was created.
    pub snapshot: Snapshot,
    /// Number of row locks held when the savepoint was created.
    /// Used to determine which locks were acquired after the savepoint.
    pub lock_count: usize,
}

/// A database transaction with MVCC snapshot isolation.
pub struct Transaction {
    /// Monotonically increasing transaction ID.
    pub txn_id: u64,
    /// Isolation level for this transaction.
    pub isolation: IsolationLevel,
    /// Snapshot of active transactions at BEGIN time.
    pub snapshot: Snapshot,
    /// Current transaction status.
    pub status: TransactionStatus,
    /// Last LSN written by this transaction (for WAL chaining).
    last_lsn: Lsn,
    /// Stack of savepoints for partial rollback.
    savepoints: Vec<Savepoint>,
    /// Index of the proc-array slot this transaction occupies, returned to
    /// the pool on commit or abort.
    slot_idx: usize,
    /// True once this transaction has appended a WAL data record. A
    /// transaction that never wrote commits without a commit record or a
    /// flush wait, since it has nothing to make durable.
    wrote_data: bool,
}

impl Transaction {
    /// Returns the transaction ID.
    #[inline]
    pub fn txn_id(&self) -> u64 {
        self.txn_id
    }

    /// Marks that this transaction has appended a WAL data record. Set by the
    /// server after a statement logs an insert, update, or delete.
    #[inline]
    pub fn mark_wrote_data(&mut self) {
        self.wrote_data = true;
    }

    /// Returns true if this transaction has appended a WAL data record.
    #[inline]
    pub fn wrote_data(&self) -> bool {
        self.wrote_data
    }

    /// Returns the last LSN written by this transaction.
    #[inline]
    pub fn last_lsn(&self) -> Lsn {
        self.last_lsn
    }

    /// Sets the last LSN for WAL chaining.
    #[inline]
    pub fn set_last_lsn(&mut self, lsn: Lsn) {
        self.last_lsn = lsn;
    }

    /// Returns true if this transaction is active.
    #[inline]
    pub fn is_active(&self) -> bool {
        self.status == TransactionStatus::Active
    }

    /// Returns the transaction's txn_id as u32 for WAL/TupleHeader writes.
    /// Errors if txn_id exceeds u32::MAX, which the sequencer does not permit.
    #[inline]
    pub fn txn_id_u32(&self) -> Result<u32> {
        u32::try_from(self.txn_id).map_err(|_| {
            ZyronError::Internal(format!(
                "txn_id {} exceeds u32::MAX, on-disk format widening required",
                self.txn_id
            ))
        })
    }

    /// Creates a savepoint with the given name.
    /// Captures the current snapshot and lock count for later rollback.
    pub fn savepoint(&mut self, name: String, current_lock_count: usize) {
        self.savepoints.push(Savepoint {
            name,
            snapshot: self.snapshot.clone(),
            lock_count: current_lock_count,
        });
    }

    /// Rolls back to the named savepoint.
    /// Restores the snapshot to the savepoint's state and returns the
    /// lock count at the savepoint (caller is responsible for releasing
    /// locks acquired after this count).
    /// Returns None if no savepoint with that name exists.
    pub fn rollback_to_savepoint(&mut self, name: &str) -> Option<usize> {
        // Find the savepoint, keeping all savepoints at or before it
        let idx = self.savepoints.iter().rposition(|sp| sp.name == name)?;
        let sp = &self.savepoints[idx];
        let lock_count = sp.lock_count;
        self.snapshot = sp.snapshot.clone();
        // Remove savepoints created after this one (but keep the named one)
        self.savepoints.truncate(idx + 1);
        Some(lock_count)
    }

    /// Releases a savepoint by name. Locks acquired after the savepoint persist.
    /// Returns false if the savepoint was not found.
    pub fn release_savepoint(&mut self, name: &str) -> bool {
        if let Some(idx) = self.savepoints.iter().rposition(|sp| sp.name == name) {
            self.savepoints.remove(idx);
            true
        } else {
            false
        }
    }

    /// Returns the number of active savepoints.
    pub fn savepoint_count(&self) -> usize {
        self.savepoints.len()
    }
}

/// Manages transaction lifecycle: begin, commit, abort.
///
/// The active-transaction set lives in a lock-free `ProcArray` of
/// cache-line-padded atomic slots. begin claims a slot via CAS, commit and
/// abort release it with an atomic store. Snapshots iterate the slot table
/// with Acquire loads, no shared lock is taken on the hot path.
pub struct TransactionManager {
    /// Monotonically increasing transaction ID counter.
    next_txn_id: AtomicU64,
    /// Lock-free table of active transaction slots.
    proc_array: ProcArray,
    /// WAL writer for durability.
    wal: Arc<WalWriter>,
    /// Row-level lock table for write-write conflict detection.
    lock_table: LockTable,
    /// Intent lock table for B+Tree key-level conflict detection.
    intent_locks: IntentLockTable,
    /// Wait-for graph for deadlock detection.
    wait_for_graph: WaitForGraph,
    /// Targeted commit-durability wakeups. A committing transaction registers
    /// its commit LSN and is woken only when a flush satisfies that LSN, instead
    /// of waking on every flush. Driven by the WAL flush thread via the
    /// registered flush waker.
    durability: Arc<DurabilityQueue>,
}

impl TransactionManager {
    /// Creates a new transaction manager.
    pub fn new(wal: Arc<WalWriter>) -> Self {
        let durability = Self::register_durability(&wal);
        Self {
            next_txn_id: AtomicU64::new(1),
            proc_array: ProcArray::new(),
            wal,
            lock_table: LockTable::new(),
            intent_locks: IntentLockTable::new(),
            wait_for_graph: WaitForGraph::new(),
            durability,
        }
    }

    /// Creates a transaction manager with a starting txn_id.
    /// Used for recovery to resume from the last known txn_id.
    pub fn with_start_txn_id(wal: Arc<WalWriter>, start_txn_id: u64) -> Self {
        let durability = Self::register_durability(&wal);
        Self {
            next_txn_id: AtomicU64::new(start_txn_id),
            proc_array: ProcArray::new(),
            wal,
            lock_table: LockTable::new(),
            intent_locks: IntentLockTable::new(),
            wait_for_graph: WaitForGraph::new(),
            durability,
        }
    }

    /// Builds the durability queue and registers a flush waker that drains it.
    /// After each flush the WAL calls the waker, which wakes only the committers
    /// whose target LSN the flush satisfied. The closure holds an Arc to the
    /// queue, keeping it alive for the WAL's lifetime; the manager holds another.
    fn register_durability(wal: &Arc<WalWriter>) -> Arc<DurabilityQueue> {
        let queue = DurabilityQueue::new(Arc::clone(wal));
        let queue_for_waker = Arc::clone(&queue);
        wal.register_flush_waker(Arc::new(move || {
            queue_for_waker.wake_satisfied();
        }));
        queue
    }

    /// Begins a new transaction with the given isolation level.
    ///
    /// Allocates a fresh txn_id, claims a proc-array slot lock-free, takes a
    /// snapshot of the other live transactions, then writes the WAL Begin
    /// record. If the WAL append fails the slot is released so the table
    /// does not leak.
    pub fn begin(&self, isolation: IsolationLevel) -> Result<Transaction> {
        let _s = profile::scope(Phase::TxnBegin);
        let txn_id = self.next_txn_id.fetch_add(1, Ordering::Relaxed);

        let slot_idx = self.proc_array.claim(txn_id)?;

        // Empty until snapshot_into pushes live txn ids. At low concurrency the
        // active set is usually empty, so starting empty skips a heap allocation
        // on the common begin; snapshot_into grows it only when peers are live.
        let mut active_ids: Vec<u64> = Vec::new();
        self.proc_array.snapshot_into(txn_id, &mut active_ids);
        let snapshot = Snapshot::new(txn_id, active_ids);

        let txn_id_u32 = match u32::try_from(txn_id) {
            Ok(v) => v,
            Err(_) => {
                self.proc_array.release(slot_idx);
                return Err(ZyronError::Internal(format!(
                    "txn_id {} exceeds u32::MAX",
                    txn_id
                )));
            }
        };
        let lsn = match self.wal.log_begin(txn_id_u32) {
            Ok(lsn) => lsn,
            Err(e) => {
                self.proc_array.release(slot_idx);
                return Err(e);
            }
        };

        Ok(Transaction {
            txn_id,
            isolation,
            snapshot,
            status: TransactionStatus::Active,
            last_lsn: lsn,
            savepoints: Vec::new(),
            slot_idx,
            wrote_data: false,
        })
    }

    /// Commits a transaction.
    ///
    /// Writes a Commit record to the WAL, releases all locks, and frees the
    /// proc-array slot.
    /// Writes the commit record, releases the transaction's locks and proc-array
    /// slot, and marks it committed. Returns the commit LSN. The caller must
    /// then await durability (via `commit` or `commit_blocking`) before
    /// acknowledging the commit; this method alone does not guarantee the record
    /// is on stable storage.
    fn commit_inner(&self, txn: &mut Transaction) -> Result<Lsn> {
        if txn.status != TransactionStatus::Active {
            return Err(ZyronError::TransactionAborted(format!(
                "transaction {} is not active (status: {:?})",
                txn.txn_id, txn.status
            )));
        }

        let txn_id_u32 = txn.txn_id_u32()?;
        let lsn = {
            let _s = profile::scope(Phase::CommitRecordAppend);
            self.wal.log_commit(txn_id_u32, txn.last_lsn)?
        };
        txn.last_lsn = lsn;
        txn.status = TransactionStatus::Committed;

        {
            let _s = profile::scope(Phase::LockRelease);
            self.lock_table.unlock_all(txn.txn_id);
            self.intent_locks.unlock_all(txn.txn_id);
            self.wait_for_graph.remove_transaction(txn.txn_id);
        }

        {
            let _s = profile::scope(Phase::ProcArrayRelease);
            self.proc_array.release(txn.slot_idx);
        }

        Ok(lsn)
    }

    /// Commits a transaction durably for async callers (the server hot path):
    /// writes the commit record, then awaits until it is fsync'd before
    /// returning, yielding the runtime worker instead of blocking it. Durability
    /// is unconditional, so a returned Ok means the commit survives a crash.
    /// Concurrent commits share one fsync via the flush thread (group commit),
    /// so the cost is one flush of latency, not one fsync per transaction.
    pub async fn commit(&self, txn: &mut Transaction) -> Result<()> {
        let lsn = self.commit_inner(txn)?;
        {
            let _s = profile::scope(Phase::DurabilityWait);
            self.wait_durable(lsn).await;
        }
        Ok(())
    }

    /// Commits a read-only transaction: one that appended no WAL data record.
    /// Such a transaction has nothing to make durable, so it writes no commit
    /// record and does no flush wait. It still releases locks, the wait-for
    /// graph edges, and the proc-array slot, and marks itself committed.
    /// Matches the standard read-only commit fast path: a transaction that
    /// only read leaves no trace that recovery must reconstruct.
    pub fn commit_read_only(&self, txn: &mut Transaction) -> Result<()> {
        if txn.status != TransactionStatus::Active {
            return Err(ZyronError::TransactionAborted(format!(
                "transaction {} is not active (status: {:?})",
                txn.txn_id, txn.status
            )));
        }

        txn.status = TransactionStatus::Committed;

        self.lock_table.unlock_all(txn.txn_id);
        self.intent_locks.unlock_all(txn.txn_id);
        self.wait_for_graph.remove_transaction(txn.txn_id);

        self.proc_array.release(txn.slot_idx);

        Ok(())
    }

    /// Commits a transaction durably for synchronous callers (background worker
    /// threads, tests). Blocks the calling thread until the commit record is
    /// fsync'd. Identical durability guarantee to `commit`; it differs only in
    /// waiting by blocking the thread rather than yielding an async task, which
    /// suits non-async call sites.
    pub fn commit_blocking(&self, txn: &mut Transaction) -> Result<()> {
        let lsn = self.commit_inner(txn)?;
        // The dedicated WAL flush thread is the sole flusher and group-commit
        // leader; wait_for_flush nudges it and parks until the commit record is
        // durable. Concurrent committers are batched into one device write.
        {
            let _s = profile::scope(Phase::DurabilityWait);
            self.wal.wait_for_flush(lsn)?;
        }
        Ok(())
    }

    /// Awaits until the WAL has durably flushed at least up to `target`,
    /// yielding the runtime worker instead of blocking it. Nudges the flush
    /// thread once, then registers on the DurabilityQueue for a targeted wakeup
    /// the flush thread issues after the flush that satisfies `target`. Group
    /// commit means many awaiting committers wake from one device write.
    async fn wait_durable(&self, target: Lsn) {
        // Fast path: already durable. Under group commit a peer's flush often
        // advanced flushed_lsn past our LSN before we even check.
        if self.wal.flushed_lsn() >= target {
            return;
        }
        // Nudge the sole flush thread so it drains and writes our record, then
        // await the targeted wakeup it fires once flushed_lsn covers target.
        self.wal.request_flush();
        self.durability.wait(target).await;
    }

    /// Aborts a transaction.
    ///
    /// Writes an Abort record to the WAL, releases all locks, and frees the
    /// proc-array slot.
    pub fn abort(&self, txn: &mut Transaction) -> Result<()> {
        if txn.status != TransactionStatus::Active {
            return Err(ZyronError::TransactionAborted(format!(
                "transaction {} is not active (status: {:?})",
                txn.txn_id, txn.status
            )));
        }

        let txn_id_u32 = txn.txn_id_u32()?;
        let lsn = self.wal.log_abort(txn_id_u32, txn.last_lsn)?;
        txn.last_lsn = lsn;
        txn.status = TransactionStatus::Aborted;

        self.lock_table.unlock_all(txn.txn_id);
        self.intent_locks.unlock_all(txn.txn_id);
        self.wait_for_graph.remove_transaction(txn.txn_id);

        self.proc_array.release(txn.slot_idx);

        Ok(())
    }

    /// Returns a sorted snapshot of currently active transaction IDs.
    pub fn active_txn_ids(&self) -> Vec<u64> {
        self.proc_array.active_txn_ids()
    }

    /// Refreshes the snapshot for a ReadCommitted transaction.
    /// Returns a new snapshot reflecting the current active transaction set.
    pub fn refresh_snapshot(&self, txn: &Transaction) -> Snapshot {
        let mut active_ids: Vec<u64> = Vec::with_capacity(16);
        self.proc_array.snapshot_into(txn.txn_id, &mut active_ids);
        Snapshot::new(txn.txn_id, active_ids)
    }

    /// Returns a reference to the row-level lock table.
    pub fn lock_table(&self) -> &LockTable {
        &self.lock_table
    }

    /// Returns a reference to the intent lock table.
    pub fn intent_locks(&self) -> &IntentLockTable {
        &self.intent_locks
    }

    /// Returns a reference to the wait-for graph for deadlock detection.
    pub fn wait_for_graph(&self) -> &WaitForGraph {
        &self.wait_for_graph
    }

    /// Returns the number of currently active transactions.
    pub fn active_count(&self) -> usize {
        self.proc_array.active_count()
    }

    /// Returns the next txn_id that will be assigned.
    pub fn next_txn_id(&self) -> u64 {
        self.next_txn_id.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use zyron_wal::{WalWriterConfig, segment::LogSegment};

    fn create_test_manager() -> (TransactionManager, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let config = WalWriterConfig {
            wal_dir: dir.path().to_path_buf(),
            segment_size: LogSegment::DEFAULT_SIZE,
            fsync_enabled: false,
            ring_buffer_capacity: 1024 * 1024,
        };
        let writer = WalWriter::new(config).unwrap();
        let mgr = TransactionManager::new(Arc::new(writer));
        (mgr, dir)
    }

    #[test]
    fn test_begin_transaction() {
        let (mgr, _dir) = create_test_manager();

        let txn = mgr.begin(IsolationLevel::SnapshotIsolation).unwrap();
        assert_eq!(txn.txn_id, 1);
        assert!(txn.is_active());
        assert_eq!(txn.isolation, IsolationLevel::SnapshotIsolation);
        assert_eq!(mgr.active_count(), 1);
    }

    #[test]
    fn test_monotonic_txn_ids() {
        let (mgr, _dir) = create_test_manager();

        let txn1 = mgr.begin(IsolationLevel::SnapshotIsolation).unwrap();
        let txn2 = mgr.begin(IsolationLevel::SnapshotIsolation).unwrap();
        let txn3 = mgr.begin(IsolationLevel::SnapshotIsolation).unwrap();

        assert_eq!(txn1.txn_id, 1);
        assert_eq!(txn2.txn_id, 2);
        assert_eq!(txn3.txn_id, 3);
        assert_eq!(mgr.active_count(), 3);
    }

    #[test]
    fn test_commit_transaction() {
        let (mgr, _dir) = create_test_manager();

        let mut txn = mgr.begin(IsolationLevel::SnapshotIsolation).unwrap();
        assert_eq!(mgr.active_count(), 1);

        mgr.commit_blocking(&mut txn).unwrap();
        assert_eq!(txn.status, TransactionStatus::Committed);
        assert_eq!(mgr.active_count(), 0);
    }

    // Exercises the async durable commit path end to end.
    #[tokio::test]
    async fn test_commit_transaction_async_durable() {
        let (mgr, _dir) = create_test_manager();
        let mut txn = mgr.begin(IsolationLevel::SnapshotIsolation).unwrap();
        mgr.commit(&mut txn).await.unwrap();
        assert_eq!(txn.status, TransactionStatus::Committed);
        assert_eq!(mgr.active_count(), 0);
    }

    // Hammers the async durable commit path under concurrency with fsync on,
    // the exact path the targeted-wakeup durability queue serves. A lost wakeup
    // would hang one of the tasks; the outer timeout turns that into a failure
    // instead of a stuck test. All commits must complete and each commit LSN
    // must be durably flushed.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_durable_commits_all_wake() {
        let dir = tempdir().unwrap();
        let config = WalWriterConfig {
            wal_dir: dir.path().to_path_buf(),
            segment_size: LogSegment::DEFAULT_SIZE,
            fsync_enabled: true,
            ring_buffer_capacity: 1024 * 1024,
        };
        let mgr = Arc::new(TransactionManager::new(Arc::new(
            WalWriter::new(config).unwrap(),
        )));

        let run = async {
            let mut handles = Vec::new();
            for _ in 0..256 {
                let mgr = Arc::clone(&mgr);
                handles.push(tokio::spawn(async move {
                    let mut txn = mgr.begin(IsolationLevel::SnapshotIsolation).unwrap();
                    let target = txn.last_lsn();
                    mgr.commit(&mut txn).await.unwrap();
                    assert_eq!(txn.status, TransactionStatus::Committed);
                    assert!(mgr.wal.flushed_lsn() >= target);
                }));
            }
            for h in handles {
                h.await.unwrap();
            }
        };

        tokio::time::timeout(std::time::Duration::from_secs(20), run)
            .await
            .expect("concurrent durable commits did not all complete: lost wakeup");
        assert_eq!(mgr.active_count(), 0);
    }

    #[test]
    fn test_abort_transaction() {
        let (mgr, _dir) = create_test_manager();

        let mut txn = mgr.begin(IsolationLevel::SnapshotIsolation).unwrap();
        assert_eq!(mgr.active_count(), 1);

        mgr.abort(&mut txn).unwrap();
        assert_eq!(txn.status, TransactionStatus::Aborted);
        assert_eq!(mgr.active_count(), 0);
    }

    #[test]
    fn test_double_commit_fails() {
        let (mgr, _dir) = create_test_manager();

        let mut txn = mgr.begin(IsolationLevel::SnapshotIsolation).unwrap();
        mgr.commit_blocking(&mut txn).unwrap();

        let result = mgr.commit_blocking(&mut txn);
        assert!(result.is_err());
    }

    #[test]
    fn test_snapshot_captures_active_set() {
        let (mgr, _dir) = create_test_manager();

        let txn1 = mgr.begin(IsolationLevel::SnapshotIsolation).unwrap();
        let txn2 = mgr.begin(IsolationLevel::SnapshotIsolation).unwrap();

        // txn2's snapshot should see txn1 as active
        assert!(txn2.snapshot.is_txn_active(txn1.txn_id));
        // txn1's snapshot should NOT see txn2 (txn2 started after txn1)
        assert!(!txn1.snapshot.is_txn_active(txn2.txn_id));
    }

    #[test]
    fn test_active_txn_ids() {
        let (mgr, _dir) = create_test_manager();

        let _txn1 = mgr.begin(IsolationLevel::SnapshotIsolation).unwrap();
        let _txn2 = mgr.begin(IsolationLevel::ReadCommitted).unwrap();

        let active = mgr.active_txn_ids();
        assert_eq!(active.len(), 2);
        assert!(active.contains(&1));
        assert!(active.contains(&2));
    }

    #[test]
    fn test_refresh_snapshot() {
        let (mgr, _dir) = create_test_manager();

        let txn1 = mgr.begin(IsolationLevel::ReadCommitted).unwrap();
        let mut txn2 = mgr.begin(IsolationLevel::SnapshotIsolation).unwrap();

        // txn1's original snapshot was taken before txn2 existed, so txn2 is not in it.
        // But a refreshed snapshot should see txn2 as active (it is currently running).
        let refreshed = mgr.refresh_snapshot(&txn1);
        assert!(refreshed.is_txn_active(txn2.txn_id));

        // Commit txn2
        mgr.commit_blocking(&mut txn2).unwrap();

        // Refresh again, txn2 is no longer active
        let refreshed2 = mgr.refresh_snapshot(&txn1);
        assert!(!refreshed2.is_txn_active(txn2.txn_id));
    }

    #[test]
    fn test_txn_id_u32_conversion() {
        let (mgr, _dir) = create_test_manager();
        let txn = mgr.begin(IsolationLevel::SnapshotIsolation).unwrap();
        assert_eq!(txn.txn_id_u32().unwrap(), 1u32);
    }

    #[test]
    fn test_savepoint_basic() {
        let (mgr, _dir) = create_test_manager();
        let mut txn = mgr.begin(IsolationLevel::SnapshotIsolation).unwrap();

        assert_eq!(txn.savepoint_count(), 0);

        txn.savepoint("sp1".into(), 0);
        assert_eq!(txn.savepoint_count(), 1);

        txn.savepoint("sp2".into(), 5);
        assert_eq!(txn.savepoint_count(), 2);
    }

    #[test]
    fn test_savepoint_rollback() {
        let (mgr, _dir) = create_test_manager();
        let mut txn = mgr.begin(IsolationLevel::SnapshotIsolation).unwrap();

        txn.savepoint("sp1".into(), 3);
        txn.savepoint("sp2".into(), 7);

        // Rollback to sp1 should remove sp2 and return lock count 3
        let lock_count = txn.rollback_to_savepoint("sp1");
        assert_eq!(lock_count, Some(3));
        assert_eq!(txn.savepoint_count(), 1); // sp1 retained

        // Rollback to non-existent savepoint
        assert!(txn.rollback_to_savepoint("sp3").is_none());
    }

    #[test]
    fn test_savepoint_release() {
        let (mgr, _dir) = create_test_manager();
        let mut txn = mgr.begin(IsolationLevel::SnapshotIsolation).unwrap();

        txn.savepoint("sp1".into(), 0);
        txn.savepoint("sp2".into(), 5);

        // Release sp1
        assert!(txn.release_savepoint("sp1"));
        assert_eq!(txn.savepoint_count(), 1);

        // Release non-existent
        assert!(!txn.release_savepoint("sp3"));
    }

    #[test]
    fn test_wait_for_graph_accessor() {
        let (mgr, _dir) = create_test_manager();
        assert_eq!(mgr.wait_for_graph().edge_count(), 0);
    }

    #[test]
    fn test_commit_cleans_wait_for_graph() {
        let (mgr, _dir) = create_test_manager();

        let txn1 = mgr.begin(IsolationLevel::SnapshotIsolation).unwrap();
        let mut txn2 = mgr.begin(IsolationLevel::SnapshotIsolation).unwrap();

        // Simulate: txn2 is waiting for txn1
        mgr.wait_for_graph().add_edge(txn2.txn_id, txn1.txn_id);
        assert_eq!(mgr.wait_for_graph().edge_count(), 1);

        // Committing txn2 should clean up its edges
        mgr.commit_blocking(&mut txn2).unwrap();
        assert_eq!(mgr.wait_for_graph().edge_count(), 0);
    }
}

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
mod retention_clock;
mod snapshot;
mod status_map;
mod undo;

pub use deadlock::WaitForGraph;
use durability::DurabilityQueue;
pub use gc::{GcStats, MvccGc};
pub use intent_lock::IntentLockTable;
pub use isolation::IsolationLevel;
pub use lock_table::LockTable;
pub use proc_array::ProcArray;
pub use retention_clock::RetentionClock;
pub use snapshot::Snapshot;
pub use status_map::{TxnStatus, TxnStatusMap};
pub use undo::{TxnUndoLog, UndoEntry};

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
    /// Number of row locks held when the savepoint was created. Locks beyond
    /// this count were acquired after the savepoint and are released on rollback.
    pub row_lock_count: usize,
    /// Number of intent (B+tree key) locks held when the savepoint was created.
    pub intent_lock_count: usize,
    /// Length of the undo log when the savepoint was created. Rolling back to
    /// this savepoint reverses every undo entry recorded at or after this mark.
    pub undo_high_water: usize,
}

/// Result of rolling back to a savepoint. The undo entries are returned in the
/// order they must be applied (last write first) so the caller reverses each at
/// the heap-tuple level. The lock counts are the held counts captured when the
/// savepoint was taken; locks acquired after it are released.
pub struct SavepointRollback {
    /// Row-lock count recorded at the savepoint.
    pub row_lock_count: usize,
    /// Intent-lock count recorded at the savepoint.
    pub intent_lock_count: usize,
    /// Undo entries to reverse, last-recorded first.
    pub undo: Vec<UndoEntry>,
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
    /// True when the transaction was started READ ONLY. Write statements are
    /// rejected before they touch the heap.
    read_only: bool,
    /// Ordered undo log of this transaction's own writes, shared with the
    /// execution context that performs them. Entries are recorded only while a
    /// savepoint is open and reversed on ROLLBACK TO SAVEPOINT.
    undo_log: Arc<TxnUndoLog>,
    /// Shared cleanup handles cloned from the manager at begin time. They let
    /// Drop release this transaction's resources if it is dropped while still
    /// active, so a client disconnect or an early-return error path that never
    /// reached commit or abort cannot leak the proc-array slot or its locks.
    proc_array: Arc<ProcArray>,
    lock_table: Arc<LockTable>,
    intent_locks: Arc<IntentLockTable>,
    wait_for_graph: Arc<WaitForGraph>,
    status_map: Arc<TxnStatusMap>,
}

impl Drop for Transaction {
    /// Releases the transaction's resources if it is dropped without an explicit
    /// commit or abort. A client that disconnects mid-transaction, or an error
    /// path that returns before committing, would otherwise leak its proc-array
    /// slot (which pins the MVCC vacuum horizon) and its held locks (which block
    /// other transactions) for the life of the process. Cleanup is in memory
    /// only and writes no WAL abort record, recovery already treats a
    /// transaction with no commit record as aborted, so the durable side is
    /// correct without it and Drop stays synchronous and infallible. commit and
    /// abort set status to Committed or Aborted before releasing, so this is a
    /// no-op after either and never double-frees a slot.
    fn drop(&mut self) {
        if self.status != TransactionStatus::Active {
            return;
        }
        self.status = TransactionStatus::Aborted;
        self.undo_log.clear();
        self.lock_table.unlock_all(self.txn_id);
        self.intent_locks.unlock_all(self.txn_id);
        self.wait_for_graph.remove_transaction(self.txn_id);
        self.status_map.record_aborted(self.txn_id);
        self.proc_array.release(self.slot_idx);
    }
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

    /// Marks this transaction READ ONLY so write statements are rejected.
    #[inline]
    pub fn set_read_only(&mut self, read_only: bool) {
        self.read_only = read_only;
    }

    /// Returns true if this transaction was started READ ONLY.
    #[inline]
    pub fn read_only(&self) -> bool {
        self.read_only
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

    /// Returns the shared undo log handle. The execution context clones this so
    /// DML operators record reversible writes into the same log this transaction
    /// reverses on ROLLBACK TO SAVEPOINT.
    #[inline]
    pub fn undo_log(&self) -> Arc<TxnUndoLog> {
        Arc::clone(&self.undo_log)
    }

    /// Creates a savepoint with the given name. Captures the current snapshot,
    /// the current row and intent lock counts, and the current undo-log length so
    /// a later rollback reverses only writes made after this point and releases
    /// only locks taken after it. Opening a savepoint turns on undo recording for
    /// subsequent writes.
    pub fn savepoint(&mut self, name: String, row_lock_count: usize, intent_lock_count: usize) {
        let undo_high_water = self.undo_log.len();
        self.savepoints.push(Savepoint {
            name,
            snapshot: self.snapshot.clone(),
            row_lock_count,
            intent_lock_count,
            undo_high_water,
        });
        self.undo_log.enter_savepoint();
    }

    /// Rolls back to the named savepoint. Returns the lock count recorded at the
    /// savepoint and the undo entries to reverse (last write first), truncating
    /// the undo log and discarding savepoints created after the target. The
    /// transaction stays open and its read snapshot is unchanged: a savepoint
    /// rollback undoes writes, not the visible committed-data snapshot.
    /// Returns None if no savepoint with that name exists.
    pub fn rollback_to_savepoint(&mut self, name: &str) -> Option<SavepointRollback> {
        // Innermost match by name, keeping savepoints at or before it.
        let idx = self.savepoints.iter().rposition(|sp| sp.name == name)?;
        let row_lock_count = self.savepoints[idx].row_lock_count;
        let intent_lock_count = self.savepoints[idx].intent_lock_count;
        let high_water = self.savepoints[idx].undo_high_water;
        // Reverse every write recorded at or after this savepoint.
        let undo = self.undo_log.drain_from(high_water);
        // Drop savepoints created after the target. The target stays open, so
        // the open-savepoint count falls by the number discarded.
        let discarded = self.savepoints.len() - (idx + 1);
        self.savepoints.truncate(idx + 1);
        if discarded > 0 {
            self.undo_log.leave_savepoints(discarded);
        }
        Some(SavepointRollback {
            row_lock_count,
            intent_lock_count,
            undo,
        })
    }

    /// Releases a savepoint by name. Its undo entries remain in the log so an
    /// outer savepoint rollback still reverses them; only the marker is removed.
    /// Locks acquired after the savepoint persist. Returns false if not found.
    pub fn release_savepoint(&mut self, name: &str) -> bool {
        if let Some(idx) = self.savepoints.iter().rposition(|sp| sp.name == name) {
            self.savepoints.remove(idx);
            self.undo_log.leave_savepoints(1);
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
    /// Lock-free table of active transaction slots. Shared into every
    /// transaction so a dropped-while-active transaction releases its slot.
    proc_array: Arc<ProcArray>,
    /// WAL writer for durability.
    wal: Arc<WalWriter>,
    /// Row-level lock table for write-write conflict detection. Shared into
    /// every transaction so a dropped-while-active transaction releases locks.
    lock_table: Arc<LockTable>,
    /// Intent lock table for B+Tree key-level conflict detection. Shared so the
    /// executor can take key locks (e.g. to serialize unique-index inserts of
    /// the same value); released by commit/abort via `unlock_all`.
    intent_locks: Arc<IntentLockTable>,
    /// Wait-for graph for deadlock detection. Shared into every transaction so
    /// a dropped-while-active transaction removes its edges.
    wait_for_graph: Arc<WaitForGraph>,
    /// Durable commit-status map consulted by MVCC visibility so an aborted
    /// transaction's writes never appear committed. Shared into every snapshot.
    status_map: Arc<TxnStatusMap>,
    /// Wall-clock to WAL-LSN sample log for time-based time-travel retention.
    /// Sampled by the vacuum worker, read by vacuum and manual VACUUM to turn a
    /// retention window into a floor LSN.
    retention_clock: Arc<RetentionClock>,
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
            proc_array: Arc::new(ProcArray::new()),
            wal,
            lock_table: Arc::new(LockTable::new()),
            intent_locks: Arc::new(IntentLockTable::new()),
            wait_for_graph: Arc::new(WaitForGraph::new()),
            status_map: Arc::new(TxnStatusMap::new()),
            retention_clock: Arc::new(RetentionClock::new()),
            durability,
        }
    }

    /// Creates a transaction manager with a starting txn_id.
    /// Used for recovery to resume from the last known txn_id.
    pub fn with_start_txn_id(wal: Arc<WalWriter>, start_txn_id: u64) -> Self {
        let durability = Self::register_durability(&wal);
        Self {
            next_txn_id: AtomicU64::new(start_txn_id),
            proc_array: Arc::new(ProcArray::new()),
            wal,
            lock_table: Arc::new(LockTable::new()),
            intent_locks: Arc::new(IntentLockTable::new()),
            wait_for_graph: Arc::new(WaitForGraph::new()),
            status_map: Arc::new(TxnStatusMap::new()),
            retention_clock: Arc::new(RetentionClock::new()),
            durability,
        }
    }

    /// Returns the commit-status map for recovery reconstruction, checkpoint
    /// persistence, vacuum, and unique-constraint probes.
    #[inline]
    pub fn status_map(&self) -> &Arc<TxnStatusMap> {
        &self.status_map
    }

    /// Returns the wall-clock to WAL-LSN sample log used by time-based
    /// time-travel retention.
    pub fn retention_clock(&self) -> &Arc<RetentionClock> {
        &self.retention_clock
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
        let snapshot = Snapshot::new(txn_id, active_ids, Arc::clone(&self.status_map));

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
            read_only: false,
            undo_log: Arc::new(TxnUndoLog::new()),
            proc_array: self.proc_array_shared(),
            lock_table: Arc::clone(&self.lock_table),
            intent_locks: Arc::clone(&self.intent_locks),
            wait_for_graph: Arc::clone(&self.wait_for_graph),
            status_map: Arc::clone(&self.status_map),
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

        // The transaction's writes are now permanent, so the undo log is dropped
        // without replay.
        txn.undo_log.clear();

        {
            let _s = profile::scope(Phase::LockRelease);
            self.lock_table.unlock_all(txn.txn_id);
            self.intent_locks.unlock_all(txn.txn_id);
            self.wait_for_graph.remove_transaction(txn.txn_id);
        }

        // Publish the committed status BEFORE leaving the active set, so no
        // snapshot can observe this transaction as neither active nor committed.
        // The commit-record LSN dates the transaction for time-travel; it is
        // stored only while commit-LSN tracking is enabled.
        self.status_map.record_committed_at(txn.txn_id, lsn.0);

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

        txn.undo_log.clear();

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

        // A full abort hides every write via the status map, so the undo log is
        // discarded without replay.
        txn.undo_log.clear();

        self.lock_table.unlock_all(txn.txn_id);
        self.intent_locks.unlock_all(txn.txn_id);
        self.wait_for_graph.remove_transaction(txn.txn_id);

        // Record the abort so this transaction's writes stay invisible after it
        // leaves the active set. The engine performs no physical undo.
        self.status_map.record_aborted(txn.txn_id);

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
        Snapshot::new(txn.txn_id, active_ids, Arc::clone(&self.status_map))
    }

    /// Returns a reference to the row-level lock table.
    pub fn lock_table(&self) -> &LockTable {
        &self.lock_table
    }

    /// Returns the shared proc array. Cloned into each transaction so a
    /// dropped-while-active transaction can release its slot.
    #[inline]
    fn proc_array_shared(&self) -> Arc<ProcArray> {
        Arc::clone(&self.proc_array)
    }

    /// Returns the shared intent lock table. The executor takes key locks
    /// through this (e.g. to serialize concurrent unique-index inserts of the
    /// same value); the locks are released by this transaction's commit/abort.
    pub fn intent_locks(&self) -> &Arc<IntentLockTable> {
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

    // A transaction dropped without commit or abort (a client disconnect or an
    // error path that never finalized it) releases its proc-array slot and
    // records itself aborted through the Drop net, so a leaked transaction
    // cannot pin the vacuum horizon or hide as perpetually active.
    #[test]
    fn dropped_active_transaction_releases_slot() {
        let (mgr, _dir) = create_test_manager();
        let txn_id = {
            let txn = mgr.begin(IsolationLevel::SnapshotIsolation).unwrap();
            assert_eq!(mgr.active_count(), 1);
            assert!(!mgr.status_map().is_aborted(txn.txn_id));
            txn.txn_id
        };
        assert_eq!(mgr.active_count(), 0);
        assert!(mgr.status_map().is_aborted(txn_id));
    }

    // The Drop net releases locks a leaked transaction held, so they do not
    // block other transactions. After the drop a fresh transaction acquires the
    // same row lock cleanly.
    #[test]
    fn dropped_active_transaction_releases_locks() {
        let (mgr, _dir) = create_test_manager();
        let rid = crate::tuple::TupleId::new(zyron_common::page::PageId::new(0, 1), 0);
        let txn_id = {
            let txn = mgr.begin(IsolationLevel::SnapshotIsolation).unwrap();
            mgr.lock_table().lock_row(txn.txn_id, 7, rid).unwrap();
            assert_eq!(mgr.lock_table().current_count(txn.txn_id), 1);
            txn.txn_id
        };
        assert_eq!(mgr.lock_table().current_count(txn_id), 0);

        let txn2 = mgr.begin(IsolationLevel::SnapshotIsolation).unwrap();
        mgr.lock_table().lock_row(txn2.txn_id, 7, rid).unwrap();
        assert_eq!(mgr.lock_table().current_count(txn2.txn_id), 1);
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

        txn.savepoint("sp1".into(), 0, 0);
        assert_eq!(txn.savepoint_count(), 1);

        txn.savepoint("sp2".into(), 5, 0);
        assert_eq!(txn.savepoint_count(), 2);
    }

    #[test]
    fn test_savepoint_rollback() {
        let (mgr, _dir) = create_test_manager();
        let mut txn = mgr.begin(IsolationLevel::SnapshotIsolation).unwrap();

        txn.savepoint("sp1".into(), 3, 2);
        txn.savepoint("sp2".into(), 7, 4);

        // Rollback to sp1 should remove sp2 and return the counts captured at sp1
        let rb = txn.rollback_to_savepoint("sp1").expect("sp1 exists");
        assert_eq!(rb.row_lock_count, 3);
        assert_eq!(rb.intent_lock_count, 2);
        assert!(rb.undo.is_empty()); // no writes recorded
        assert_eq!(txn.savepoint_count(), 1); // sp1 retained

        // Rollback to non-existent savepoint
        assert!(txn.rollback_to_savepoint("sp3").is_none());
    }

    #[test]
    fn test_savepoint_release() {
        let (mgr, _dir) = create_test_manager();
        let mut txn = mgr.begin(IsolationLevel::SnapshotIsolation).unwrap();

        txn.savepoint("sp1".into(), 0, 0);
        txn.savepoint("sp2".into(), 5, 0);

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

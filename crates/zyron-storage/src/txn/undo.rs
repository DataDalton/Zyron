//! Per-transaction undo log for ROLLBACK TO SAVEPOINT.
//!
//! The engine model is abort-marks-writes-invisible via the status map with no
//! physical undo. Partial rollback needs to reverse a transaction's OWN writes
//! made after a savepoint, which is safe because a transaction's uncommitted
//! writes are invisible to every other transaction. Each entry reverses one
//! write at the tuple-header level:
//!   - ReverseInsert: the txn inserted a tuple; undo self-deletes it by stamping
//!     xmax = the txn id, so the txn stops seeing it and after commit no one does
//!   - ReverseDelete: the txn deleted a pre-existing tuple (stamped xmax = txn
//!     id); undo clears xmax back to 0, restoring the row
//! An UPDATE is delete-old + insert-new, so it records ReverseDelete(old) then
//! ReverseInsert(new).
//!
//! Recording is gated on the transaction having at least one open savepoint, so
//! a transaction with no savepoint records nothing and pays no overhead.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::tuple::TupleId;

/// One reversible write recorded while a savepoint is open. Carries the heap and
/// free-space-map file ids so a partial rollback can address the heap that holds
/// the tuple without a catalog lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UndoEntry {
    /// The transaction inserted this tuple. Reverse by stamping xmax = txn id.
    ReverseInsert {
        heap_file_id: u32,
        fsm_file_id: u32,
        tid: TupleId,
    },
    /// The transaction deleted this pre-existing tuple (stamped its xmax = txn
    /// id). Reverse by clearing xmax back to 0.
    ReverseDelete {
        heap_file_id: u32,
        fsm_file_id: u32,
        tid: TupleId,
    },
}

/// Ordered undo log shared between the owning Transaction and the execution
/// context that performs its writes. The entry vector is mutated behind a Mutex
/// because the execution context appends while it holds only a shared reference.
/// `active_savepoints` is an atomic so the write path's gate check is a single
/// relaxed load with no lock.
pub struct TxnUndoLog {
    entries: Mutex<Vec<UndoEntry>>,
    active_savepoints: AtomicUsize,
}

impl TxnUndoLog {
    /// Creates an empty undo log with no open savepoints.
    pub fn new() -> Self {
        Self {
            entries: Mutex::new(Vec::new()),
            active_savepoints: AtomicUsize::new(0),
        }
    }

    /// Returns true when at least one savepoint is open. The DML write path
    /// checks this before recording an entry so a transaction with no open
    /// savepoint records nothing.
    #[inline]
    pub fn has_active_savepoint(&self) -> bool {
        self.active_savepoints.load(Ordering::Relaxed) > 0
    }

    /// Records a reversible write. The caller gates this on
    /// `has_active_savepoint`, so entries accumulate only while a savepoint is
    /// open.
    #[inline]
    pub fn record(&self, entry: UndoEntry) {
        self.entries.lock().expect("undo log poisoned").push(entry);
    }

    /// Returns the current number of recorded entries. Captured by `savepoint`
    /// as the high-water mark to roll back to.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.lock().expect("undo log poisoned").len()
    }

    /// Returns true when no entries are recorded.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Increments the open-savepoint count. Called by `Transaction::savepoint`.
    #[inline]
    pub fn enter_savepoint(&self) {
        self.active_savepoints.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrements the open-savepoint count, saturating at zero. Called when a
    /// savepoint is released or discarded by an outer rollback.
    #[inline]
    pub fn leave_savepoints(&self, count: usize) {
        let mut current = self.active_savepoints.load(Ordering::Relaxed);
        loop {
            let next = current.saturating_sub(count);
            match self.active_savepoints.compare_exchange_weak(
                current,
                next,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(observed) => current = observed,
            }
        }
    }

    /// Removes and returns every entry recorded at or after `high_water`, in
    /// reverse (last-recorded first) order so the caller applies each reverse op
    /// in the correct undo order. The log is truncated to `high_water`.
    pub fn drain_from(&self, high_water: usize) -> Vec<UndoEntry> {
        let mut entries = self.entries.lock().expect("undo log poisoned");
        if high_water >= entries.len() {
            return Vec::new();
        }
        let mut tail: Vec<UndoEntry> = entries.split_off(high_water);
        tail.reverse();
        tail
    }

    /// Discards all entries and resets the open-savepoint count. Called on commit
    /// (writes become permanent) and on full abort (status map hides them).
    pub fn clear(&self) {
        self.entries.lock().expect("undo log poisoned").clear();
        self.active_savepoints.store(0, Ordering::Relaxed);
    }
}

impl Default for TxnUndoLog {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for TxnUndoLog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TxnUndoLog")
            .field("len", &self.len())
            .field(
                "active_savepoints",
                &self.active_savepoints.load(Ordering::Relaxed),
            )
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyron_common::page::PageId;

    fn rid(page: u64, slot: u16) -> TupleId {
        TupleId::new(PageId::new(0, page), slot)
    }

    #[test]
    fn records_only_when_savepoint_open() {
        let log = TxnUndoLog::new();
        assert!(!log.has_active_savepoint());
        log.enter_savepoint();
        assert!(log.has_active_savepoint());
        log.record(UndoEntry::ReverseInsert {
            heap_file_id: 0,
            fsm_file_id: 1,
            tid: rid(0, 0),
        });
        assert_eq!(log.len(), 1);
    }

    #[test]
    fn drain_from_returns_reversed_tail() {
        let log = TxnUndoLog::new();
        log.enter_savepoint();
        let a = UndoEntry::ReverseInsert {
            heap_file_id: 0,
            fsm_file_id: 1,
            tid: rid(0, 0),
        };
        let b = UndoEntry::ReverseInsert {
            heap_file_id: 0,
            fsm_file_id: 1,
            tid: rid(0, 1),
        };
        let c = UndoEntry::ReverseDelete {
            heap_file_id: 0,
            fsm_file_id: 1,
            tid: rid(0, 2),
        };
        log.record(a);
        let hw = log.len();
        log.record(b);
        log.record(c);
        let drained = log.drain_from(hw);
        assert_eq!(drained, vec![c, b]);
        assert_eq!(log.len(), 1);
    }

    #[test]
    fn leave_savepoints_saturates() {
        let log = TxnUndoLog::new();
        log.enter_savepoint();
        log.enter_savepoint();
        log.leave_savepoints(5);
        assert!(!log.has_active_savepoint());
    }

    #[test]
    fn clear_resets_everything() {
        let log = TxnUndoLog::new();
        log.enter_savepoint();
        log.record(UndoEntry::ReverseInsert {
            heap_file_id: 0,
            fsm_file_id: 1,
            tid: rid(0, 0),
        });
        log.clear();
        assert!(log.is_empty());
        assert!(!log.has_active_savepoint());
    }
}

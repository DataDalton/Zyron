//! Lock-free active-transaction registry (ProcArray).
//!
//! Replaces a `parking_lot::Mutex<BTreeSet<u64>>` that was acquired on
//! every begin and commit. The mutex serialized all transaction starts
//! and ends under contention, capping concurrent OLTP throughput.
//!
//! Layout
//! ------
//! A fixed-size array of cache-line-padded `AtomicU64` slots. Each slot is
//! either `FREE` (u64::MAX) or holds an active txn_id. begin claims a slot
//! with `compare_exchange`, commit/abort releases it with a relaxed store.
//! Snapshots iterate the slots with Acquire loads, no locking.
//!
//! Capacity
//! --------
//! `MAX_SLOTS = 4096`. At 64 bytes per slot the table is 256 KB, paid once
//! at server startup. ZyronDB rejects connections above this ceiling
//! upstream, so claim cannot legitimately fail under correct operation,
//! a full table returns an internal error.
//!
//! Snapshot semantics
//! ------------------
//! The existing semantics are preserved: snapshot scan happens after
//! `next_txn_id` is advanced and after the slot is claimed, so any
//! transaction with a smaller txn_id that has already claimed a slot is
//! visible to the new transaction. The pre-existing benign race window
//! between fetch_add of `next_txn_id` and slot claim is unchanged in
//! width (it was bounded by the mutex's critical section before).

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use zyron_common::{Result, ZyronError};

/// Number of slots in the ProcArray. Sized for max concurrent connections.
pub const MAX_SLOTS: usize = 4096;

const FREE: u64 = u64::MAX;

#[repr(align(64))]
struct PaddedSlot(AtomicU64);

/// Lock-free table of active transaction IDs.
///
/// `slots` is the actual storage. `active_count` is an authoritative count
/// of slots currently holding a txn_id, used by `snapshot_into` to early
/// exit the scan once it has located every active txn. Without this hint
/// every begin() would scan all 4096 atomic slots, which at high
/// commit/sec (100K+) saturates multiple CPU cores on the scan alone and
/// starves the tokio runtime.
pub struct ProcArray {
    slots: Box<[PaddedSlot]>,
    active_count: AtomicUsize,
}

impl ProcArray {
    pub fn new() -> Self {
        let mut slots: Vec<PaddedSlot> = Vec::with_capacity(MAX_SLOTS);
        for _ in 0..MAX_SLOTS {
            slots.push(PaddedSlot(AtomicU64::new(FREE)));
        }
        Self {
            slots: slots.into_boxed_slice(),
            active_count: AtomicUsize::new(0),
        }
    }

    /// Claims a free slot and stores `txn_id` into it atomically.
    /// Probes from slot 0 so that occupied slots stay clustered near the
    /// start of the table. `snapshot_into` then early exits as soon as it
    /// has found every active txn, which is the hot path the whole array
    /// exists to keep cheap.
    pub fn claim(&self, txn_id: u64) -> Result<usize> {
        for i in 0..MAX_SLOTS {
            let slot = &self.slots[i].0;
            if slot.load(Ordering::Relaxed) != FREE {
                continue;
            }
            if slot
                .compare_exchange(FREE, txn_id, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                self.active_count.fetch_add(1, Ordering::Release);
                return Ok(i);
            }
        }
        Err(ZyronError::Internal(format!(
            "proc array full, no free slot for txn {}",
            txn_id
        )))
    }

    /// Releases the slot. Caller must own the slot index returned from claim.
    pub fn release(&self, slot_idx: usize) {
        self.slots[slot_idx].0.store(FREE, Ordering::Release);
        self.active_count.fetch_sub(1, Ordering::Release);
    }

    /// Fills `into` with the txn_ids of all active transactions, excluding
    /// `exclude_txn_id`. The result is sorted ascending so callers can
    /// binary search. Scans only as far as needed to locate every active
    /// slot, using `active_count` as the termination hint, so the cost is
    /// O(active_txns) in steady state, not O(MAX_SLOTS).
    pub fn snapshot_into(&self, exclude_txn_id: u64, into: &mut Vec<u64>) {
        into.clear();
        let target = self.active_count.load(Ordering::Acquire);
        if target == 0 {
            return;
        }
        let mut found = 0usize;
        for s in self.slots.iter() {
            if found >= target {
                break;
            }
            let v = s.0.load(Ordering::Acquire);
            if v != FREE {
                found += 1;
                if v != exclude_txn_id {
                    into.push(v);
                }
            }
        }
        into.sort_unstable();
    }

    /// Returns a fresh sorted Vec of currently active transaction ids.
    pub fn active_txn_ids(&self) -> Vec<u64> {
        let target = self.active_count.load(Ordering::Acquire);
        if target == 0 {
            return Vec::new();
        }
        let mut v = Vec::with_capacity(target);
        let mut found = 0usize;
        for s in self.slots.iter() {
            if found >= target {
                break;
            }
            let val = s.0.load(Ordering::Acquire);
            if val != FREE {
                v.push(val);
                found += 1;
            }
        }
        v.sort_unstable();
        v
    }

    /// Returns the count of slots holding an active transaction.
    pub fn active_count(&self) -> usize {
        self.active_count.load(Ordering::Acquire)
    }
}

impl Default for ProcArray {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claim_and_release_roundtrip() {
        let p = ProcArray::new();
        assert_eq!(p.active_count(), 0);
        let s = p.claim(7).unwrap();
        assert_eq!(p.active_count(), 1);
        let v = p.active_txn_ids();
        assert_eq!(v, vec![7u64]);
        p.release(s);
        assert_eq!(p.active_count(), 0);
        assert!(p.active_txn_ids().is_empty());
    }

    #[test]
    fn snapshot_excludes_self() {
        let p = ProcArray::new();
        let _ = p.claim(10).unwrap();
        let s2 = p.claim(11).unwrap();
        let mut buf = Vec::new();
        p.snapshot_into(11, &mut buf);
        assert_eq!(buf, vec![10u64]);
        let _ = s2;
    }

    #[test]
    fn snapshot_is_sorted() {
        let p = ProcArray::new();
        let _ = p.claim(50).unwrap();
        let _ = p.claim(20).unwrap();
        let _ = p.claim(40).unwrap();
        let mut buf = Vec::new();
        p.snapshot_into(u64::MAX, &mut buf);
        assert_eq!(buf, vec![20u64, 40, 50]);
    }

    #[test]
    fn release_makes_slot_reusable() {
        let p = ProcArray::new();
        let s1 = p.claim(100).unwrap();
        p.release(s1);
        let s2 = p.claim(101).unwrap();
        assert_eq!(p.active_count(), 1);
        p.release(s2);
    }

    #[test]
    fn concurrent_claim_release_threadsafe() {
        use std::sync::Arc;
        use std::thread;

        let p = Arc::new(ProcArray::new());
        let threads: Vec<_> = (0..16)
            .map(|t| {
                let p = Arc::clone(&p);
                thread::spawn(move || {
                    for i in 0..200 {
                        let txn_id = (t as u64) * 10_000 + i;
                        let s = p.claim(txn_id).unwrap();
                        p.release(s);
                    }
                })
            })
            .collect();
        for h in threads {
            h.join().unwrap();
        }
        assert_eq!(p.active_count(), 0);
    }
}

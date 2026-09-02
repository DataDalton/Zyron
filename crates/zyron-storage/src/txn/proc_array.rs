//! Lock-free active-transaction registry (ProcArray).
//!
//! Replaces a `parking_lot::Mutex<BTreeSet<u64>>` that was acquired on
//! every begin and commit. The mutex serialized all transaction starts
//! and ends under contention, capping concurrent OLTP throughput.
//!
//! Layout
//! ------
//! A fixed-size array of cache-line-padded slots. Each slot holds an active
//! txn_id or `FREE` (u64::MAX), alongside the lowest `xmax` that
//! transaction can still need to see. begin claims a slot with
//! `compare_exchange`, commit/abort releases it with a relaxed store.
//! Snapshots iterate the slots with Acquire loads, no locking.
//!
//! Two questions are answered off this array, and they have opposite safety
//! requirements. Which transactions are active is an exact-membership
//! question, where a missing id reads as committed. The prune horizon is a
//! minimum, where a value that is too high physically reclaims tuples a live
//! reader is still entitled to. The horizon is therefore taken over
//! published per-transaction floors rather than over txn ids, which is not
//! the same number.
//!
//! Capacity
//! --------
//! `MAX_SLOTS = 4096`. At 64 bytes per slot the table is 256 KB, paid once
//! at server startup. Zyron rejects connections above this ceiling
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

/// Horizon value published by a slot that has claimed an id but has not yet
/// captured its snapshot. Pruning cannot know what such a transaction will
/// need, so any slot carrying it forces the global horizon to zero and
/// on-access pruning stands down until the transaction publishes
const HORIZON_UNSET: u64 = 0;

#[repr(align(64))]
struct PaddedSlot {
    /// Active transaction id, or `FREE`
    txn: AtomicU64,
    /// Lowest `xmax` this transaction can still need to see, published once
    /// its snapshot is taken, `HORIZON_UNSET` until then
    horizon: AtomicU64,
}

/// Lock-free table of active transaction IDs.
///
/// `slots` is the actual storage. `active_count` tracks how many slots
/// currently hold a txn_id and sizes the snapshot result Vec. It is not a
/// scan termination hint, a concurrent claim/release can change it between
/// the count load and the slot scan, so a scan that stopped on it could miss
/// a still-active txn in a higher slot. Scans bound themselves with
/// `high_water` instead, which only ever rises.
pub struct ProcArray {
    slots: Box<[PaddedSlot]>,
    active_count: AtomicUsize,
    /// Highest slot index any claim has probed, so a slot above it has never
    /// been claimed and no scan needs to visit it. Raised before the claiming
    /// compare_exchange, which is what makes it safe to stop at: a scan that
    /// reads it cannot miss a slot the claim is about to take.
    ///
    /// It never falls, so the cost of a scan tracks peak concurrency rather
    /// than the fixed 4096 slots. Measured at 1.0 ns for one live slot, 4.9
    /// at 16, 41.6 at 128, and 1730.3 for the full table, which is what every
    /// scan paid before the bound existed. Lowering it is not safe the
    /// obvious way: a scan that observes the top slots free and CAS-lowers
    /// can erase a raise from a claim that already landed, and the next scan
    /// then misses that live transaction
    high_water: AtomicUsize,
}

impl ProcArray {
    pub fn new() -> Self {
        let mut slots: Vec<PaddedSlot> = Vec::with_capacity(MAX_SLOTS);
        for _ in 0..MAX_SLOTS {
            slots.push(PaddedSlot {
                txn: AtomicU64::new(FREE),
                horizon: AtomicU64::new(HORIZON_UNSET),
            });
        }
        Self {
            slots: slots.into_boxed_slice(),
            active_count: AtomicUsize::new(0),
            high_water: AtomicUsize::new(0),
        }
    }

    /// Claims a free slot and stores `txn_id` into it atomically.
    /// Probes from slot 0 so that occupied slots stay clustered near the
    /// start of the table, which keeps `high_water` close to the real peak
    /// concurrency and the scans that stop at it short.
    pub fn claim(&self, txn_id: u64) -> Result<usize> {
        for i in 0..MAX_SLOTS {
            let slot = &self.slots[i].txn;
            if slot.load(Ordering::Relaxed) != FREE {
                continue;
            }
            // Raise the scan bound before taking the slot. A horizon scan that
            // reads the bound after this point covers slot i, so it cannot
            // treat a transaction claiming here as absent
            self.high_water.fetch_max(i, Ordering::Release);
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
    ///
    /// The horizon is cleared before the id, so a slot observed as claimed
    /// never carries the previous owner's horizon. The next claim's
    /// compare_exchange reads the freeing store, so a scanner that sees the
    /// new id sees `HORIZON_UNSET` or the new owner's own value
    pub fn release(&self, slot_idx: usize) {
        self.slots[slot_idx]
            .horizon
            .store(HORIZON_UNSET, Ordering::Release);
        self.slots[slot_idx].txn.store(FREE, Ordering::Release);
        self.active_count.fetch_sub(1, Ordering::Release);
    }

    /// Replaces the id a claimed slot publishes. Caller must own the slot.
    /// begin claims with a conservative placeholder id before allocating
    /// the real one, then publishes the real id here, so a concurrent
    /// horizon scan never finds the id space advanced past a transaction
    /// it cannot see
    pub fn set(&self, slot_idx: usize, txn_id: u64) {
        self.slots[slot_idx].txn.store(txn_id, Ordering::Release);
    }

    /// Publishes the lowest `xmax` this transaction can still need to see.
    /// Caller must own the slot. Until this lands the slot reads as
    /// `HORIZON_UNSET` and holds the global horizon at zero, which stops
    /// on-access pruning rather than letting it guess
    pub fn publish_horizon(&self, slot_idx: usize, horizon: u64) {
        self.slots[slot_idx]
            .horizon
            .store(horizon, Ordering::Release);
    }

    /// Lowest `xmax` that no live transaction can still need to see, so a
    /// committed delete below it is invisible everywhere and its tuple can be
    /// reclaimed.
    ///
    /// This is the minimum over the published horizons of every live slot,
    /// not the minimum active txn id. The two differ, and using the id is
    /// wrong. A transaction still in flight when a reader took its snapshot
    /// sits in that reader's active set, so the reader keeps seeing rows that
    /// transaction deleted even after it commits and leaves the array. Its id
    /// is then absent from the array while the reader still needs the rows
    ///
    /// Returns 0 when any live slot has yet to publish, which prunes nothing,
    /// and `u64::MAX` when no slot is live, since a transaction starting
    /// later takes a snapshot with an empty active set and needs nothing
    /// already committed
    pub fn global_prune_horizon(&self) -> u64 {
        let bound = self.high_water.load(Ordering::Acquire);
        let mut horizon = u64::MAX;
        for slot in self.slots[..=bound].iter() {
            if slot.txn.load(Ordering::Acquire) == FREE {
                continue;
            }
            let published = slot.horizon.load(Ordering::Acquire);
            if published == HORIZON_UNSET {
                return 0;
            }
            horizon = horizon.min(published);
        }
        horizon
    }

    /// Fills `into` with the txn_ids of all active transactions, excluding
    /// `exclude_txn_id`. The result is sorted ascending so callers can
    /// binary search.
    ///
    /// Stopping early on a racy `active_count` would skip a still-active txn
    /// in a higher slot whenever a concurrent claim/release shifts the count
    /// between the load and the scan, which would treat an uncommitted txn as
    /// committed. `active_count` is used only to size the result.
    ///
    /// The scan bound is `high_water`, which is sound rather than a guess. A
    /// transaction P this caller must see has `P.id < txn_id`, so P won the
    /// id it holds before this caller won its own, and P raised `high_water`
    /// past its slot before claiming that slot. Both ids come from
    /// compare_exchange on one SeqCst counter, so P's raise precedes this
    /// caller's id, which precedes this load: the bound already covers P's
    /// slot. A transaction claiming concurrently takes a higher id and is
    /// fenced by visibility instead.
    pub fn snapshot_into(&self, exclude_txn_id: u64, into: &mut Vec<u64>) {
        into.clear();
        into.reserve(self.active_count.load(Ordering::Acquire));
        let bound = self.high_water.load(Ordering::Acquire);
        for s in self.slots[..=bound].iter() {
            let v = s.txn.load(Ordering::Acquire);
            if v != FREE && v != exclude_txn_id {
                into.push(v);
            }
        }
        into.sort_unstable();
    }

    /// Returns a fresh sorted Vec of currently active transaction ids.
    /// Scans every slot for a consistent snapshot, the same reason
    /// `snapshot_into` does, `active_count` only sizes the Vec.
    pub fn active_txn_ids(&self) -> Vec<u64> {
        let mut v = Vec::with_capacity(self.active_count.load(Ordering::Acquire));
        for s in self.slots.iter() {
            let val = s.txn.load(Ordering::Acquire);
            if val != FREE {
                v.push(val);
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

impl std::fmt::Debug for ProcArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProcArray")
            .field("active_count", &self.active_count())
            .field("prune_horizon", &self.global_prune_horizon())
            .finish()
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
    fn snapshot_includes_high_slot_after_low_release() {
        // A txn in a higher slot must stay visible after a lower slot is
        // released, even though active_count drops to 1. An early-exit scan
        // keyed on active_count would stop before reaching the higher slot
        // and wrongly omit the still-active txn.
        let p = ProcArray::new();
        let s_low = p.claim(100).unwrap();
        let _s_high = p.claim(200).unwrap();
        assert_eq!(s_low, 0);
        p.release(s_low);
        assert_eq!(p.active_count(), 1);
        let mut buf = Vec::new();
        p.snapshot_into(u64::MAX, &mut buf);
        assert_eq!(buf, vec![200u64]);
        assert_eq!(p.active_txn_ids(), vec![200u64]);
    }

    #[test]
    fn concurrent_release_never_omits_active_txn() {
        // Stress the count-vs-scan race: one thread churns low slots while
        // another snapshots. A pinned high-slot txn must appear in every
        // snapshot regardless of the concurrent count changes.
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering as AtOrd};
        use std::thread;

        let p = Arc::new(ProcArray::new());
        let _churn_slot = p.claim(10).unwrap();
        let pinned_slot = p.claim(u64::MAX - 1).unwrap();
        assert!(pinned_slot > 0);

        let stop = Arc::new(AtomicBool::new(false));
        let churn_p = Arc::clone(&p);
        let churn_stop = Arc::clone(&stop);
        let churn = thread::spawn(move || {
            while !churn_stop.load(AtOrd::Relaxed) {
                let s = churn_p.claim(11).unwrap();
                churn_p.release(s);
            }
        });

        let mut buf = Vec::new();
        for _ in 0..50_000 {
            p.snapshot_into(u64::MAX, &mut buf);
            assert!(
                buf.contains(&(u64::MAX - 1)),
                "pinned high-slot txn missing from snapshot"
            );
        }
        stop.store(true, AtOrd::Relaxed);
        churn.join().unwrap();
        p.release(pinned_slot);
    }

    #[test]
    fn horizon_is_unpublished_until_the_snapshot_is_taken() {
        // A claimed slot that has not published holds the horizon down, so a
        // concurrent delete cannot prune during the window between claiming
        // the slot and capturing the active set
        let p = ProcArray::new();
        let s = p.claim(10).unwrap();
        assert_eq!(p.global_prune_horizon(), 0);
        p.publish_horizon(s, 10);
        assert_eq!(p.global_prune_horizon(), 10);
        p.release(s);
        assert_eq!(p.global_prune_horizon(), u64::MAX);
    }

    #[test]
    fn horizon_holds_at_an_older_readers_floor_not_the_writers() {
        // The interleaving that physically removed rows out from under a live
        // reader. Delete 5 is in flight when reader 10 takes its snapshot, so
        // 5 lands in the reader's active set and the reader keeps seeing the
        // rows 5 deleted. Once 5 commits and leaves the array, writer 20 sees
        // only {10} active, so its own frozen horizon is 10. Pruning on that
        // reclaims the rows with xmax 5 while the reader still needs them
        let p = ProcArray::new();
        let reader = p.claim(10).unwrap();
        // Reader began while 5 was still active, so its floor sits at 5
        p.publish_horizon(reader, 5);

        let writer = p.claim(20).unwrap();
        p.publish_horizon(writer, 10);

        let horizon = p.global_prune_horizon();
        assert_eq!(
            horizon, 5,
            "horizon must fall back to the oldest live reader's floor"
        );
        assert!(
            !(5 < horizon),
            "a version deleted by 5 must not be reclaimable while reader 10 is live"
        );

        p.release(reader);
        assert_eq!(p.global_prune_horizon(), 10);
        p.release(writer);
    }

    #[test]
    fn horizon_scan_covers_slots_above_a_released_low_slot() {
        // The scan bound must still reach a live high slot after a lower one
        // is released, the same requirement the active-set scan has
        let p = ProcArray::new();
        let low = p.claim(10).unwrap();
        let high = p.claim(20).unwrap();
        p.publish_horizon(low, 10);
        p.publish_horizon(high, 4);
        assert!(high > low);
        p.release(low);
        assert_eq!(
            p.global_prune_horizon(),
            4,
            "released low slot must not hide the live high slot's horizon"
        );
        p.release(high);
    }

    #[test]
    fn released_slot_does_not_leak_its_horizon_to_the_next_owner() {
        let p = ProcArray::new();
        let s = p.claim(10).unwrap();
        p.publish_horizon(s, 10);
        p.release(s);
        let reused = p.claim(30).unwrap();
        assert_eq!(reused, s, "claim probes from slot 0 so the slot is reused");
        assert_eq!(
            p.global_prune_horizon(),
            0,
            "reused slot must read as unpublished, not carry the old horizon"
        );
        p.release(reused);
    }

    #[test]
    fn horizon_never_passes_a_live_slot_under_churn() {
        // The scan bound is what a missed slot would come through. Each
        // thread publishes a floor and then checks the global horizon has not
        // risen above it while it is live, which is exactly the guarantee
        // on-access pruning relies on
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering as AtOrd};
        use std::thread;

        let p = Arc::new(ProcArray::new());
        let stop = Arc::new(AtomicBool::new(false));

        // Push high_water up, then free the slots so later claims reuse them
        let warm: Vec<usize> = (0..64)
            .map(|i| p.claim(9_000 + i as u64).unwrap())
            .collect();
        for s in warm {
            p.release(s);
        }

        let threads: Vec<_> = (0..8)
            .map(|t| {
                let p = Arc::clone(&p);
                let stop = Arc::clone(&stop);
                thread::spawn(move || {
                    for i in 0..4_000u64 {
                        let id = 1_000_000 + t * 100_000 + i;
                        let floor = id - 500;
                        let slot = p.claim(id).unwrap();
                        p.publish_horizon(slot, floor);
                        let seen = p.global_prune_horizon();
                        assert!(
                            seen <= floor,
                            "horizon {seen} passed a live slot whose floor is {floor}"
                        );
                        p.release(slot);
                    }
                    stop.store(true, AtOrd::Relaxed);
                })
            })
            .collect();
        for h in threads {
            h.join().unwrap();
        }
        assert_eq!(p.active_count(), 0);
    }

    #[test]
    #[ignore]
    fn measure_scan_cost_by_high_water() {
        use std::time::Instant;
        const OPS: usize = 200_000;
        for peak in [1usize, 16, 128, 512, 4096] {
            let p = ProcArray::new();
            // Drive high_water to the peak, then leave four live so the scan
            // has something to find, which is the steady state after a burst
            let mut slots = Vec::new();
            for i in 0..peak {
                slots.push(p.claim(1000 + i as u64).unwrap());
            }
            for (i, s) in slots.iter().enumerate() {
                p.publish_horizon(*s, 1000 + i as u64);
            }
            for s in slots.iter().skip(4) {
                p.release(*s);
            }

            let mut buf = Vec::new();
            for _ in 0..1000 {
                p.snapshot_into(0, &mut buf);
                std::hint::black_box(p.global_prune_horizon());
            }

            let t = Instant::now();
            for _ in 0..OPS {
                p.snapshot_into(0, &mut buf);
                std::hint::black_box(&buf);
            }
            let snap_ns = t.elapsed().as_nanos() as f64 / OPS as f64;

            let t = Instant::now();
            for _ in 0..OPS {
                std::hint::black_box(p.global_prune_horizon());
            }
            let horizon_ns = t.elapsed().as_nanos() as f64 / OPS as f64;

            println!(
                "peak {peak:5}  snapshot_into {snap_ns:9.1} ns  global_prune_horizon {horizon_ns:9.1} ns"
            );
        }
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

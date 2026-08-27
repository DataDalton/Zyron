//! Lock-free page table for buffer pool page ID to frame ID mapping.

use crate::frame::FrameId;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use zyron_common::page::PageId;

/// Direct array size for file_id=0 pages. Covers first 16384 pages (~256MB at 16KB pages).
const DIRECT_PATH_SIZE: usize = 16384;

/// Sentinel value for empty slots in direct path.
const EMPTY_FRAME: u32 = u32::MAX;

/// Sentinel value for empty key slots in hash table.
const EMPTY_KEY: u64 = u64::MAX;

/// Sentinel value for deleted key slots (tombstone).
const TOMBSTONE_KEY: u64 = u64::MAX - 1;

/// Sentinel marking a slot reserved by an in-progress insert_if_absent.
/// The value is staged under this marker before the real key is published, so a
/// reader never sees a real key paired with a stale value.
const INPROGRESS_KEY: u64 = u64::MAX - 2;

/// What a probe found in the slot it stopped on.
#[derive(Clone, Copy)]
enum Slot {
    /// Free and never used, the slot that ends a run
    Empty,
    /// Freed by a remove, reusable but not proof the key is absent
    Tombstone,
    /// Already holds the key, carrying this frame id
    Holds(u32),
}

/// Result of an atomic insert-if-absent.
///
/// `Inserted` means this call installed the frame. `Existing` means another frame
/// was already mapped for the page id, the caller must release the frame it tried
/// to install and use the returned winner instead. `TableFull` means the hash
/// table has no slot.
pub enum InsertOutcome {
    Inserted,
    Existing(FrameId),
    TableFull,
}

/// Lock-free page table mapping PageId to FrameId.
///
/// Uses two-tier lookup:
/// - Direct array for file_id=0, page_num < 16384 (~2-3ns lookup)
/// - Open-addressing hash table with linear probing for overflow (~8-12ns lookup)
pub struct PageTable {
    /// Direct array for file_id=0 pages. Stores frame_id directly.
    direct_path: Box<[AtomicU32; DIRECT_PATH_SIZE]>,
    /// Hash table keys (full 64-bit PageId).
    hash_keys: Box<[AtomicU64]>,
    /// Hash table values (frame_id).
    hash_values: Box<[AtomicU32]>,
    /// Bitmask for hash table indexing (hash_size - 1).
    hash_mask: usize,
}

impl PageTable {
    /// Creates a new page table with capacity for the given number of frames.
    pub fn new(capacity: usize) -> Self {
        // Hash table size = next power of 2, 2x capacity for ~50% load factor
        let hash_size = (capacity * 2).next_power_of_two().max(1024);

        // Initialize direct path array with empty sentinel
        let direct_path: Box<[AtomicU32; DIRECT_PATH_SIZE]> = {
            let mut v = Vec::with_capacity(DIRECT_PATH_SIZE);
            for _ in 0..DIRECT_PATH_SIZE {
                v.push(AtomicU32::new(EMPTY_FRAME));
            }
            v.into_boxed_slice().try_into().unwrap()
        };

        // Initialize separate key and value arrays for hash table
        let hash_keys: Box<[AtomicU64]> =
            (0..hash_size).map(|_| AtomicU64::new(EMPTY_KEY)).collect();

        let hash_values: Box<[AtomicU32]> = (0..hash_size)
            .map(|_| AtomicU32::new(EMPTY_FRAME))
            .collect();

        Self {
            direct_path,
            hash_keys,
            hash_values,
            hash_mask: hash_size - 1,
        }
    }

    /// Looks up a page ID and returns its frame ID if present.
    #[inline(always)]
    pub fn get(&self, page_id: PageId) -> Option<FrameId> {
        if page_id.file_id == 0 && (page_id.page_num as usize) < DIRECT_PATH_SIZE {
            let val = self.direct_path[page_id.page_num as usize].load(Ordering::Acquire);
            if val != EMPTY_FRAME {
                return Some(FrameId(val));
            }
            return None;
        }
        self.get_from_hash(page_id)
    }

    #[inline]
    fn get_from_hash(&self, page_id: PageId) -> Option<FrameId> {
        let key = page_id.as_u64();
        let mut idx = self.hash_index(key);

        for _ in 0..self.hash_keys.len() {
            let stored_key = self.hash_keys[idx].load(Ordering::Acquire);
            if stored_key == EMPTY_KEY {
                return None;
            }
            if stored_key == key {
                let frame_id = self.hash_values[idx].load(Ordering::Acquire);
                return Some(FrameId(frame_id));
            }
            // Tombstones and in-progress reservations are probed past rather
            // than waited on. This is the pool's hottest path, and a lookup
            // racing an insert that has not published its key yet is entitled
            // to report the page absent: the insert has not taken effect. A
            // published entry further along is still found, because probing
            // stops only at an empty slot and one key occupies one slot
            idx = (idx + 1) & self.hash_mask;
        }
        None
    }

    /// Inserts a page ID to frame ID mapping. Returns true on success.
    pub fn insert(&self, page_id: PageId, frame_id: FrameId) -> bool {
        if page_id.file_id == 0 && (page_id.page_num as usize) < DIRECT_PATH_SIZE {
            self.direct_path[page_id.page_num as usize].store(frame_id.0, Ordering::Release);
            return true;
        }
        self.insert_to_hash(page_id, frame_id)
    }

    /// Inserts a page ID to frame ID mapping only if the page id is absent.
    ///
    /// Resolves concurrent inserts for the same page id to a single winner so two
    /// new_page calls for one id cannot install two frames. Returns the existing
    /// frame when another inserter won the race.
    pub fn insert_if_absent(&self, page_id: PageId, frame_id: FrameId) -> InsertOutcome {
        if page_id.file_id == 0 && (page_id.page_num as usize) < DIRECT_PATH_SIZE {
            match self.direct_path[page_id.page_num as usize].compare_exchange(
                EMPTY_FRAME,
                frame_id.0,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => InsertOutcome::Inserted,
                Err(existing) => InsertOutcome::Existing(FrameId(existing)),
            }
        } else {
            self.insert_if_absent_hash(page_id, frame_id)
        }
    }

    fn insert_if_absent_hash(&self, page_id: PageId, frame_id: FrameId) -> InsertOutcome {
        let key = page_id.as_u64();
        let capacity = self.hash_keys.len();

        for _ in 0..capacity {
            let Some((claim, expect)) = self.probe_for_claim(key) else {
                return InsertOutcome::TableFull;
            };
            if let Slot::Holds(existing) = expect {
                return InsertOutcome::Existing(FrameId(existing));
            }
            if !self.reserve(claim, expect) {
                // Lost the slot to another inserter. Probe again from the
                // start, the run has changed and the new occupant may be
                // this very key
                continue;
            }
            self.publish(claim, key, frame_id);

            // Two inserters can still both publish when a removal turns a
            // live slot into a tombstone between their reads of it. Each
            // then looks for the first slot in the run holding the key and
            // the later one stands down, so exactly one copy survives and
            // both callers agree on which
            match self.first_slot_holding(key) {
                Some(winner) if winner != claim => {
                    let existing = self.hash_values[winner].load(Ordering::Acquire);
                    self.stand_down(claim);
                    return InsertOutcome::Existing(FrameId(existing));
                }
                _ => return InsertOutcome::Inserted,
            }
        }
        InsertOutcome::TableFull
    }

    /// What a probe found in a slot it may claim.
    fn probe_for_claim(&self, key: u64) -> Option<(usize, Slot)> {
        let mut idx = self.hash_index(key);
        let mut first_tomb = None;
        for _ in 0..self.hash_keys.len() {
            // A reserved slot has to be resolved before this slot is judged.
            // The reservation may be about to publish this very key, and
            // probing past it would let both inserters take a slot for the
            // same key, leaving one page id mapped to two frames and two
            // copies of the page in the pool.
            //
            // The wait is bounded by two stores in the reserving thread,
            // with no I/O and no lock in between
            let stored_key = self.await_slot_resolution(idx);
            if stored_key == key {
                let existing = self.hash_values[idx].load(Ordering::Acquire);
                return Some((idx, Slot::Holds(existing)));
            }
            if stored_key == TOMBSTONE_KEY {
                // Recorded, not taken. A tombstone is not proof the key is
                // absent: the key can sit past it, put there while this slot
                // still held a live one. Only the empty slot that ends the
                // run proves absence, so the walk continues to it
                if first_tomb.is_none() {
                    first_tomb = Some(idx);
                }
            } else if stored_key == EMPTY_KEY {
                // A tombstone is reused ahead of the empty slot, so a table
                // that has been evicting for a long time does not fill with
                // dead slots and start reporting itself full
                return Some(match first_tomb {
                    Some(tomb) => (tomb, Slot::Tombstone),
                    None => (idx, Slot::Empty),
                });
            }
            idx = (idx + 1) & self.hash_mask;
        }
        first_tomb.map(|tomb| (tomb, Slot::Tombstone))
    }

    /// Takes a slot exclusively with the in-progress marker.
    #[inline]
    fn reserve(&self, idx: usize, expect: Slot) -> bool {
        let prior = match expect {
            Slot::Tombstone => TOMBSTONE_KEY,
            Slot::Empty => EMPTY_KEY,
            Slot::Holds(_) => return false,
        };
        self.hash_keys[idx]
            .compare_exchange(prior, INPROGRESS_KEY, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    /// Gives up a published slot, always as a tombstone.
    ///
    /// Never as an empty slot, even when the claim displaced one. Another
    /// key can have been placed past this slot while it was published, and
    /// an empty here would end the run in front of it and lose it
    #[inline]
    fn stand_down(&self, idx: usize) {
        self.hash_keys[idx].store(TOMBSTONE_KEY, Ordering::Release);
    }

    /// Stages the value under the reservation, then publishes the key, so a
    /// reader that loads the key also loads the matching value.
    #[inline]
    fn publish(&self, idx: usize, key: u64, frame_id: FrameId) {
        self.hash_values[idx].store(frame_id.0, Ordering::Release);
        self.hash_keys[idx].store(key, Ordering::Release);
    }

    /// The first slot in the key's run that holds it, in the same order
    /// `get` probes, so it names the copy every reader will find.
    fn first_slot_holding(&self, key: u64) -> Option<usize> {
        let mut idx = self.hash_index(key);
        for _ in 0..self.hash_keys.len() {
            let stored_key = self.await_slot_resolution(idx);
            if stored_key == key {
                return Some(idx);
            }
            if stored_key == EMPTY_KEY {
                return None;
            }
            idx = (idx + 1) & self.hash_mask;
        }
        None
    }

    /// Reads a slot's key, waiting out an in-progress reservation so the
    /// caller never judges a slot whose key is about to change.
    ///
    /// Returns the resolved key, which is never `INPROGRESS_KEY`.
    #[inline]
    fn await_slot_resolution(&self, idx: usize) -> u64 {
        loop {
            let stored_key = self.hash_keys[idx].load(Ordering::Acquire);
            if stored_key != INPROGRESS_KEY {
                return stored_key;
            }
            std::hint::spin_loop();
        }
    }

    fn insert_to_hash(&self, page_id: PageId, frame_id: FrameId) -> bool {
        let key = page_id.as_u64();
        let capacity = self.hash_keys.len();

        for _ in 0..capacity {
            let Some((claim, expect)) = self.probe_for_claim(key) else {
                return false;
            };
            if matches!(expect, Slot::Holds(_)) {
                self.hash_values[claim].store(frame_id.0, Ordering::Release);
                return true;
            }
            if !self.reserve(claim, expect) {
                continue;
            }
            self.publish(claim, key, frame_id);

            // Same stand-down rule the insert-if-absent path uses, so a race
            // cannot leave one page id in two slots. The earlier slot keeps
            // the mapping and takes this frame id, because a plain insert
            // overwrites whatever was mapped
            if let Some(winner) = self.first_slot_holding(key) {
                if winner != claim {
                    self.hash_values[winner].store(frame_id.0, Ordering::Release);
                    self.stand_down(claim);
                }
            }
            return true;
        }
        false
    }

    /// Removes a page ID mapping. Returns the frame ID if it was present.
    pub fn remove(&self, page_id: PageId) -> Option<FrameId> {
        if page_id.file_id == 0 && (page_id.page_num as usize) < DIRECT_PATH_SIZE {
            let old =
                self.direct_path[page_id.page_num as usize].swap(EMPTY_FRAME, Ordering::AcqRel);
            if old != EMPTY_FRAME {
                return Some(FrameId(old));
            }
            return None;
        }
        self.remove_from_hash(page_id)
    }

    fn remove_from_hash(&self, page_id: PageId) -> Option<FrameId> {
        let key = page_id.as_u64();
        let mut idx = self.hash_index(key);

        for _ in 0..self.hash_keys.len() {
            // A reservation resolves to either this key or another one, and
            // skipping it would report the entry absent while it is being
            // published, leaving the mapping in place after a remove
            let stored_key = self.await_slot_resolution(idx);
            if stored_key == EMPTY_KEY {
                return None;
            }
            if stored_key == key {
                let frame_id = self.hash_values[idx].load(Ordering::Acquire);
                self.hash_keys[idx].store(TOMBSTONE_KEY, Ordering::Release);
                return Some(FrameId(frame_id));
            }
            idx = (idx + 1) & self.hash_mask;
        }
        None
    }

    /// Computes hash table index for a key.
    #[inline(always)]
    fn hash_index(&self, key: u64) -> usize {
        // FxHash-style multiply for distribution
        let hash = key.wrapping_mul(0x517cc1b727220a95);
        (hash as usize) & self.hash_mask
    }

    /// Returns true if the page ID is in the table.
    pub fn contains(&self, page_id: PageId) -> bool {
        self.get(page_id).is_some()
    }

    /// Returns the number of entries in the table.
    pub fn len(&self) -> usize {
        let mut count = 0;
        for slot in self.direct_path.iter() {
            let val = slot.load(Ordering::Relaxed);
            if val != EMPTY_FRAME {
                count += 1;
            }
        }
        for slot in self.hash_keys.iter() {
            let val = slot.load(Ordering::Relaxed);
            if val != EMPTY_KEY && val != TOMBSTONE_KEY && val != INPROGRESS_KEY {
                count += 1;
            }
        }
        count
    }

    /// Returns true if the table is empty.
    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterates over all entries, calling the provided function for each.
    /// Returns early if the function returns false.
    ///
    /// Keys and values load Acquire so a published entry is observed with
    /// its matching frame: a Relaxed pair could tear against a concurrent
    /// publish and hand a flusher a stale or unrelated frame id, which
    /// would write another page's bytes to this page's disk slot.
    pub fn for_each<F>(&self, mut f: F)
    where
        F: FnMut(PageId, FrameId) -> bool,
    {
        // Iterate direct path
        for (page_num, slot) in self.direct_path.iter().enumerate() {
            let val = slot.load(Ordering::Acquire);
            if val != EMPTY_FRAME {
                let page_id = PageId::new(0, page_num as u64);
                if !f(page_id, FrameId(val)) {
                    return;
                }
            }
        }

        // Iterate hash table
        for (idx, key_slot) in self.hash_keys.iter().enumerate() {
            let key = key_slot.load(Ordering::Acquire);
            if key != EMPTY_KEY && key != TOMBSTONE_KEY && key != INPROGRESS_KEY {
                let frame_id = self.hash_values[idx].load(Ordering::Acquire);
                let page_id = PageId::from_u64(key);
                if !f(page_id, FrameId(frame_id)) {
                    return;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_direct_path_insert_get() {
        let table = PageTable::new(100);
        let page_id = PageId::new(0, 42);
        let frame_id = FrameId(7);

        assert!(table.insert(page_id, frame_id));
        assert_eq!(table.get(page_id), Some(frame_id));
        assert!(table.contains(page_id));
    }

    #[test]
    fn test_direct_path_remove() {
        let table = PageTable::new(100);
        let page_id = PageId::new(0, 42);
        let frame_id = FrameId(7);

        table.insert(page_id, frame_id);
        assert_eq!(table.remove(page_id), Some(frame_id));
        assert_eq!(table.get(page_id), None);
        assert!(!table.contains(page_id));
    }

    #[test]
    fn test_hash_path_insert_get() {
        let table = PageTable::new(100);
        // Use file_id > 0 to go through hash path
        let page_id = PageId::new(1, 42);
        let frame_id = FrameId(7);

        assert!(table.insert(page_id, frame_id));
        assert_eq!(table.get(page_id), Some(frame_id));
    }

    #[test]
    fn test_hash_path_remove() {
        let table = PageTable::new(100);
        let page_id = PageId::new(1, 42);
        let frame_id = FrameId(7);

        table.insert(page_id, frame_id);
        assert_eq!(table.remove(page_id), Some(frame_id));
        assert_eq!(table.get(page_id), None);
    }

    #[test]
    fn test_high_page_num_uses_hash() {
        let table = PageTable::new(100);
        // page_num >= DIRECT_PATH_SIZE goes through hash
        let page_id = PageId::new(0, DIRECT_PATH_SIZE as u64 + 100);
        let frame_id = FrameId(5);

        assert!(table.insert(page_id, frame_id));
        assert_eq!(table.get(page_id), Some(frame_id));
    }

    #[test]
    fn test_len() {
        let table = PageTable::new(100);

        assert_eq!(table.len(), 0);
        assert!(table.is_empty());

        table.insert(PageId::new(0, 1), FrameId(1));
        table.insert(PageId::new(0, 2), FrameId(2));
        table.insert(PageId::new(1, 1), FrameId(3));

        assert_eq!(table.len(), 3);
        assert!(!table.is_empty());
    }

    #[test]
    fn test_update_existing() {
        let table = PageTable::new(100);
        let page_id = PageId::new(0, 42);

        table.insert(page_id, FrameId(1));
        assert_eq!(table.get(page_id), Some(FrameId(1)));

        table.insert(page_id, FrameId(2));
        assert_eq!(table.get(page_id), Some(FrameId(2)));
        assert_eq!(table.len(), 1);
    }

    /// Concurrent inserts of one hash-path key produce exactly one winner.
    ///
    /// An inserter reserves its slot with an in-progress marker before
    /// publishing the real key. A second inserter that probed past that
    /// marker took the next free slot, so both got `Inserted` and the key
    /// occupied two slots: one page id mapped to two buffer frames, which is
    /// two copies of the page and a lost write when one of them is flushed.
    ///
    /// Only the hash path is affected. The direct path is a single
    /// compare-exchange with no reservation window, and is included here so a
    /// regression on either side is caught.
    #[test]
    fn test_concurrent_insert_if_absent_yields_one_winner() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering as O};

        // Enough rounds and threads that the reservation window is hit. The
        // defect this guards reproduced in roughly one run in thirteen with a
        // single round, so the rounds are what make the test decisive
        for round in 0..400u64 {
            for page_id in [PageId::new(0, 5), PageId::new(7, 100_000 + round)] {
                let table = Arc::new(PageTable::new(64));
                let barrier = Arc::new(std::sync::Barrier::new(8));
                let inserted = Arc::new(AtomicUsize::new(0));
                let mut handles = Vec::new();

                for t in 0..8u32 {
                    let table = Arc::clone(&table);
                    let barrier = Arc::clone(&barrier);
                    let inserted = Arc::clone(&inserted);
                    handles.push(std::thread::spawn(move || {
                        barrier.wait();
                        match table.insert_if_absent(page_id, FrameId(t)) {
                            InsertOutcome::Inserted => {
                                inserted.fetch_add(1, O::Relaxed);
                                FrameId(t)
                            }
                            InsertOutcome::Existing(winner) => winner,
                            InsertOutcome::TableFull => panic!("table has room for one key"),
                        }
                    }));
                }
                let seen: Vec<FrameId> = handles.into_iter().map(|h| h.join().unwrap()).collect();

                assert_eq!(
                    inserted.load(O::Relaxed),
                    1,
                    "exactly one inserter may win {page_id:?}"
                );
                assert_eq!(table.len(), 1, "one key occupies one slot");
                let winner = table.get(page_id).expect("the winner is readable");
                for got in seen {
                    assert_eq!(got, winner, "every caller resolves to the winning frame");
                }
            }
        }
    }

    /// Two page ids that land on the same slot, so the second sits one past
    /// the first.
    fn colliding_pair(table: &PageTable) -> (PageId, PageId) {
        let first = PageId::new(1, 1);
        let target = table.hash_index(first.as_u64());
        for n in 2..2_000_000u64 {
            let candidate = PageId::new(1, n);
            if table.hash_index(candidate.as_u64()) == target {
                return (first, candidate);
            }
        }
        panic!("no colliding page id found");
    }

    /// Every slot in the table that holds this key. The table's whole
    /// contract is that the answer is never longer than one entry, and the
    /// pool depends on it: two frames for one page id means a write through
    /// one is invisible through the other.
    fn slots_holding(table: &PageTable, page_id: PageId) -> Vec<usize> {
        let key = page_id.as_u64();
        (0..table.hash_keys.len())
            .filter(|&i| table.hash_keys[i].load(Ordering::Relaxed) == key)
            .collect()
    }

    /// A tombstone in front of a live key is not proof the key is absent.
    /// An insert that stops there installs a second frame for a page that
    /// is already mapped, and the remove that follows tombstones only the
    /// copy it reaches, leaving the other one answering lookups.
    #[test]
    fn insert_if_absent_refuses_a_page_that_sits_behind_a_tombstone() {
        let table = PageTable::new(64);
        let (front, behind) = colliding_pair(&table);

        assert!(table.insert(front, FrameId(1)));
        assert!(matches!(
            table.insert_if_absent(behind, FrameId(2)),
            InsertOutcome::Inserted
        ));
        assert_eq!(table.remove(front), Some(FrameId(1)));

        match table.insert_if_absent(behind, FrameId(3)) {
            InsertOutcome::Existing(frame) => assert_eq!(frame, FrameId(2)),
            InsertOutcome::Inserted => panic!("one page id was mapped to two frames"),
            InsertOutcome::TableFull => panic!("table reported full with 62 slots free"),
        }
        assert_eq!(slots_holding(&table, behind).len(), 1);
        assert_eq!(table.get(behind), Some(FrameId(2)));
        assert_eq!(table.remove(behind), Some(FrameId(2)));
        assert_eq!(table.get(behind), None, "remove left a stale mapping");
    }

    /// The plain insert has the same run to walk, and updating in place is
    /// the only outcome that keeps one slot per key.
    #[test]
    fn insert_updates_a_page_that_sits_behind_a_tombstone() {
        let table = PageTable::new(64);
        let (front, behind) = colliding_pair(&table);

        assert!(table.insert(front, FrameId(1)));
        assert!(table.insert(behind, FrameId(2)));
        assert_eq!(table.remove(front), Some(FrameId(1)));

        assert!(table.insert(behind, FrameId(3)));
        assert_eq!(slots_holding(&table, behind).len(), 1);
        assert_eq!(table.get(behind), Some(FrameId(3)));
        assert_eq!(table.remove(behind), Some(FrameId(3)));
        assert_eq!(table.get(behind), None);
    }

    /// A pool evicts for as long as it runs, so a table that only ever
    /// spends slots would report itself full long before it is.
    #[test]
    fn tombstones_are_reused_rather_than_accumulated() {
        let table = PageTable::new(64);
        for round in 0..10_000u64 {
            let page = PageId::new(1, round + 1);
            assert!(
                matches!(
                    table.insert_if_absent(page, FrameId(round as u32 % 64)),
                    InsertOutcome::Inserted
                ),
                "round {round} could not install"
            );
            assert_eq!(table.get(page), Some(FrameId(round as u32 % 64)));
            assert!(table.remove(page).is_some());
        }
        assert_eq!(table.len(), 0);
    }

    /// Concurrent installs of one page id resolve to a single winner while
    /// removals keep turning slots into tombstones underneath them.
    #[test]
    fn concurrent_install_and_evict_never_doubles_a_page() {
        use std::sync::Arc;
        use std::sync::atomic::AtomicUsize;

        let table = Arc::new(PageTable::new(256));
        let contested: Vec<PageId> = (1..=8u64).map(|n| PageId::new(1, n * 4099)).collect();
        let inserted = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::new();
        for thread in 0..6u32 {
            let table = Arc::clone(&table);
            let pages = contested.clone();
            let inserted = Arc::clone(&inserted);
            handles.push(std::thread::spawn(move || {
                for round in 0..4_000u32 {
                    let page = pages[(round as usize + thread as usize) % pages.len()];
                    match table.insert_if_absent(page, FrameId(thread)) {
                        InsertOutcome::Inserted => {
                            inserted.fetch_add(1, Ordering::Relaxed);
                            table.remove(page);
                        }
                        InsertOutcome::Existing(_) => {}
                        InsertOutcome::TableFull => panic!("table full at 256 frames"),
                    }
                }
            }));
        }
        for handle in handles {
            handle.join().expect("worker");
        }
        assert!(
            inserted.load(Ordering::Relaxed) > 0,
            "nothing was installed"
        );
        for page in &contested {
            let held = slots_holding(&table, *page);
            assert!(
                held.len() <= 1,
                "page {:?} ended up in slots {:?}",
                page,
                held
            );
        }
    }
}

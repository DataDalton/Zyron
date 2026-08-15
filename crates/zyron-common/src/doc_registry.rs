//! Per-table document identity for search indexes.
//!
//! FTS, vector and spatial indexes address rows through a dense per-table
//! ordinal DocId instead of a heap page and slot. The registry maps each
//! ordinal to the row's current RowLocator and back, so an indexed row keeps
//! its DocId when the background fold moves it from the heap to a columnar
//! segment and its heap slot is later vacuumed. Postings stay u64 and BM25
//! keeps scoring dense arrays, ordinals only grow.
//!
//! Ordinals are never reused. A deleted document leaves a None slot in the
//! forward table, which also removes the heap slot-reuse hazard that forced
//! index deletes to run before inserts on UPDATE.

use std::sync::{Arc, RwLock};

use crate::RowLocator;

/// Registry over every table's document identity map
pub struct DocRegistry {
    tables: scc::HashMap<u32, Arc<TableDocRegistry>>,
}

/// One table's document identity map. The forward table index IS the
/// ordinal, allocation appends under the write lock so the invariant
/// cannot drift. The reverse map serves DML, which knows a row's locator
/// and needs its DocId to maintain the indexes.
pub struct TableDocRegistry {
    forward: RwLock<Vec<Option<RowLocator>>>,
    reverse: scc::HashMap<RowLocator, u64>,
}

impl TableDocRegistry {
    fn new() -> Self {
        Self {
            forward: RwLock::new(Vec::new()),
            reverse: scc::HashMap::new(),
        }
    }
}

impl DocRegistry {
    pub fn new() -> Self {
        Self {
            tables: scc::HashMap::new(),
        }
    }

    fn table(&self, table_id: u32) -> Arc<TableDocRegistry> {
        match self.tables.entry_sync(table_id) {
            scc::hash_map::Entry::Occupied(e) => Arc::clone(e.get()),
            scc::hash_map::Entry::Vacant(e) => {
                let t = Arc::new(TableDocRegistry::new());
                e.insert_entry(Arc::clone(&t));
                t
            }
        }
    }

    /// Allocates the next ordinal for a row entering the table's search
    /// indexes. One ordinal per row, shared by FTS, vector and spatial.
    /// Re-inserting a locator that already has a live ordinal returns the
    /// existing one, so the three index kinds share cleanly.
    pub fn allocate(&self, table_id: u32, locator: RowLocator) -> u64 {
        let t = self.table(table_id);
        if let Some(existing) = t.reverse.read_sync(&locator, |_, v| *v) {
            return existing;
        }
        let mut fwd = t.forward.write().expect("doc registry forward lock");
        // re-check under the lock, a racing allocate may have won
        if let Some(existing) = t.reverse.read_sync(&locator, |_, v| *v) {
            return existing;
        }
        let ordinal = fwd.len() as u64;
        fwd.push(Some(locator));
        let _ = t.reverse.insert_sync(locator, ordinal);
        ordinal
    }

    /// The row locator currently behind a DocId. None for a deleted
    /// document or an ordinal the table never allocated.
    pub fn locator(&self, table_id: u32, doc: u64) -> Option<RowLocator> {
        let t = self.tables.read_sync(&table_id, |_, t| Arc::clone(t))?;
        let fwd = t.forward.read().expect("doc registry forward lock");
        fwd.get(doc as usize).copied().flatten()
    }

    /// Maps a batch of DocIds to locators under one read lock. Output is
    /// cleared and refilled positionally.
    pub fn map_docs(&self, table_id: u32, docs: &[u64], out: &mut Vec<Option<RowLocator>>) {
        out.clear();
        let Some(t) = self.tables.read_sync(&table_id, |_, t| Arc::clone(t)) else {
            out.resize(docs.len(), None);
            return;
        };
        let fwd = t.forward.read().expect("doc registry forward lock");
        out.extend(docs.iter().map(|d| fwd.get(*d as usize).copied().flatten()));
    }

    /// The live DocId for a row, if it has one.
    pub fn doc_for(&self, table_id: u32, locator: RowLocator) -> Option<u64> {
        let t = self.tables.read_sync(&table_id, |_, t| Arc::clone(t))?;
        t.reverse.read_sync(&locator, |_, v| *v)
    }

    /// Removes a row's document identity on DELETE, returning the DocId the
    /// caller must delete from each index. The ordinal is never reused.
    ///
    /// The forward write lock is taken before the reverse map is touched, so
    /// the removal is one step against a concurrent `allocate` for the same
    /// row. Reading the reverse map first let an allocate return the ordinal
    /// this call was in the middle of retiring, handing the caller a DocId
    /// whose forward slot had already been cleared.
    pub fn take(&self, table_id: u32, locator: RowLocator) -> Option<u64> {
        let t = self.tables.read_sync(&table_id, |_, t| Arc::clone(t))?;
        let mut fwd = t.forward.write().expect("doc registry forward lock");
        let (_, ordinal) = t.reverse.remove_sync(&locator)?;
        if let Some(slot) = fwd.get_mut(ordinal as usize) {
            *slot = None;
        }
        Some(ordinal)
    }

    /// Re-points a document at the row's new storage location, keeping its
    /// DocId. Called by the fold when a heap row moves into a columnar
    /// segment, this is what keeps search indexes valid across folding.
    /// Returns false when the old locator had no live document.
    ///
    /// The whole re-point runs under the forward write lock, which is the
    /// lock `allocate` holds while it re-checks the reverse map. Publishing
    /// the new locator after releasing it let an `allocate` for that same
    /// locator slip in between and the row came away with two ordinals. The
    /// fold writes columnar locators here while an index build allocates for
    /// the rows it scans, which is where the two meet.
    ///
    /// Holding the lock is not enough on its own: when the allocate runs
    /// first, `new` already carries a document, and moving `old`'s onto it
    /// would still make two. The invariant is one locator, one document, so
    /// the destination's existing document wins and the old ordinal is
    /// retired. A search then returns the row once, and postings left behind
    /// under the retired ordinal resolve to a dead slot exactly as they do
    /// after a delete.
    pub fn repoint(&self, table_id: u32, old: RowLocator, new: RowLocator) -> bool {
        let Some(t) = self.tables.read_sync(&table_id, |_, t| Arc::clone(t)) else {
            return false;
        };
        let mut fwd = t.forward.write().expect("doc registry forward lock");
        let Some((_, ordinal)) = t.reverse.remove_sync(&old) else {
            return false;
        };
        if t.reverse.read_sync(&new, |_, v| *v).is_some() {
            // The destination is already identified. Retire the old ordinal
            // rather than pointing a second one at the same row
            if let Some(slot) = fwd.get_mut(ordinal as usize) {
                *slot = None;
            }
            return true;
        }
        if let Some(slot) = fwd.get_mut(ordinal as usize) {
            *slot = Some(new);
        }
        let _ = t.reverse.insert_sync(new, ordinal);
        true
    }

    /// One past the highest ordinal ever allocated, sizes dense score
    /// accumulators.
    pub fn ordinal_count(&self, table_id: u32) -> u64 {
        self.tables
            .read_sync(&table_id, |_, t| {
                t.forward.read().expect("doc registry forward lock").len() as u64
            })
            .unwrap_or(0)
    }

    /// Drops a table's whole map on DROP TABLE.
    pub fn drop_table(&self, table_id: u32) {
        let _ = self.tables.remove_sync(&table_id);
    }

    /// Serializes every table map. Layout: magic "ZYDOC" + version u8 +
    /// table count u32, then per table: table_id u32 + forward len u64 +
    /// entries as kind u8 (0 dead, 1 heap, 2 columnar, 3 lake) + payload
    /// (heap: file u64 + page u64 + slot u16; columnar/lake: two u64).
    /// Dead slots persist so ordinals stay stable across restart.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(64);
        out.extend_from_slice(b"ZYDOC");
        out.push(1u8);
        let mut ids: Vec<u32> = Vec::new();
        self.tables.iter_sync(|id, _| {
            ids.push(*id);
            true
        });
        ids.sort_unstable();
        out.extend_from_slice(&(ids.len() as u32).to_le_bytes());
        for id in ids {
            let Some(t) = self.tables.read_sync(&id, |_, t| Arc::clone(t)) else {
                out.extend_from_slice(&id.to_le_bytes());
                out.extend_from_slice(&0u64.to_le_bytes());
                continue;
            };
            let fwd = t.forward.read().expect("doc registry forward lock");
            out.extend_from_slice(&id.to_le_bytes());
            out.extend_from_slice(&(fwd.len() as u64).to_le_bytes());
            for slot in fwd.iter() {
                match slot {
                    None => out.push(0),
                    Some(RowLocator::Heap { page, slot }) => {
                        out.push(1);
                        out.extend_from_slice(&(page.file_id as u64).to_le_bytes());
                        out.extend_from_slice(&page.page_num.to_le_bytes());
                        out.extend_from_slice(&slot.to_le_bytes());
                    }
                    Some(RowLocator::Columnar { file_id, sys_rowid }) => {
                        out.push(2);
                        out.extend_from_slice(&file_id.to_le_bytes());
                        out.extend_from_slice(&sys_rowid.to_le_bytes());
                    }
                    Some(RowLocator::Lake { file_id, ordinal }) => {
                        out.push(3);
                        out.extend_from_slice(&file_id.to_le_bytes());
                        out.extend_from_slice(&ordinal.to_le_bytes());
                    }
                }
            }
        }
        out
    }

    /// Rebuilds a registry from encode() output. A short or corrupt buffer
    /// yields None, the caller starts empty like a missing snapshot.
    pub fn decode(data: &[u8]) -> Option<Self> {
        let mut p = 0usize;
        let need = |p: usize, n: usize| p.checked_add(n).filter(|e| *e <= data.len());
        need(p, 6)?;
        if &data[..5] != b"ZYDOC" || data[5] != 1 {
            return None;
        }
        p = 6;
        let rd_u16 =
            |p: usize| -> Option<u16> { Some(u16::from_le_bytes(data[p..p + 2].try_into().ok()?)) };
        let rd_u32 =
            |p: usize| -> Option<u32> { Some(u32::from_le_bytes(data[p..p + 4].try_into().ok()?)) };
        let rd_u64 =
            |p: usize| -> Option<u64> { Some(u64::from_le_bytes(data[p..p + 8].try_into().ok()?)) };
        let registry = Self::new();
        let table_count = {
            need(p, 4)?;
            let v = rd_u32(p)?;
            p += 4;
            v
        };
        for _ in 0..table_count {
            need(p, 12)?;
            let table_id = rd_u32(p)?;
            let len = rd_u64(p + 4)? as usize;
            p += 12;
            let t = registry.table(table_id);
            let mut fwd = t.forward.write().expect("doc registry forward lock");
            fwd.reserve(len);
            for ordinal in 0..len {
                need(p, 1)?;
                let kind = data[p];
                p += 1;
                let loc = match kind {
                    0 => None,
                    1 => {
                        need(p, 18)?;
                        let file_id = rd_u64(p)?;
                        let page_num = rd_u64(p + 8)?;
                        let slot = rd_u16(p + 16)?;
                        p += 18;
                        Some(RowLocator::Heap {
                            page: crate::page::PageId::new(file_id as u32, page_num),
                            slot,
                        })
                    }
                    2 => {
                        need(p, 16)?;
                        let file_id = rd_u64(p)?;
                        let sys_rowid = rd_u64(p + 8)?;
                        p += 16;
                        Some(RowLocator::Columnar { file_id, sys_rowid })
                    }
                    3 => {
                        need(p, 16)?;
                        let file_id = rd_u64(p)?;
                        let ordinal = rd_u64(p + 8)?;
                        p += 16;
                        Some(RowLocator::Lake { file_id, ordinal })
                    }
                    _ => return None,
                };
                if let Some(loc) = loc {
                    let _ = t.reverse.insert_sync(loc, ordinal as u64);
                }
                fwd.push(loc);
            }
        }
        Some(registry)
    }
}

impl Default for DocRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::page::PageId;

    fn heap(page: u64, slot: u16) -> RowLocator {
        RowLocator::Heap {
            page: PageId::new(7, page),
            slot,
        }
    }

    #[test]
    fn allocate_is_dense_and_idempotent_per_locator() {
        let r = DocRegistry::new();
        assert_eq!(r.allocate(1, heap(0, 0)), 0);
        assert_eq!(r.allocate(1, heap(0, 1)), 1);
        assert_eq!(r.allocate(1, heap(0, 0)), 0, "same locator keeps its doc");
        assert_eq!(r.allocate(2, heap(0, 0)), 0, "tables are independent");
        assert_eq!(r.ordinal_count(1), 2);
    }

    #[test]
    fn take_tombstones_and_never_reuses_the_ordinal() {
        let r = DocRegistry::new();
        let d0 = r.allocate(1, heap(0, 0));
        assert_eq!(r.take(1, heap(0, 0)), Some(d0));
        assert_eq!(r.locator(1, d0), None);
        assert_eq!(r.doc_for(1, heap(0, 0)), None);
        // a new row in the reused heap slot gets a fresh ordinal, the old
        // doc id can never resolve to the new row
        let d1 = r.allocate(1, heap(0, 0));
        assert_ne!(d0, d1);
        assert_eq!(r.locator(1, d1), Some(heap(0, 0)));
    }

    #[test]
    fn repoint_keeps_the_doc_across_a_fold() {
        let r = DocRegistry::new();
        let doc = r.allocate(1, heap(3, 4));
        let folded = RowLocator::Columnar {
            file_id: 9,
            sys_rowid: 42,
        };
        assert!(r.repoint(1, heap(3, 4), folded));
        assert_eq!(r.locator(1, doc), Some(folded));
        assert_eq!(r.doc_for(1, folded), Some(doc));
        assert_eq!(r.doc_for(1, heap(3, 4)), None);
        assert!(!r.repoint(1, heap(3, 4), folded), "old locator is gone");
    }

    #[test]
    fn map_docs_resolves_batches_positionally() {
        let r = DocRegistry::new();
        let a = r.allocate(1, heap(0, 0));
        let b = r.allocate(1, heap(0, 1));
        r.take(1, heap(0, 1));
        let mut out = Vec::new();
        r.map_docs(1, &[b, a, 99], &mut out);
        assert_eq!(out, vec![None, Some(heap(0, 0)), None]);
    }

    #[test]
    fn encode_decode_round_trips_all_kinds_and_tombstones() {
        let r = DocRegistry::new();
        let d0 = r.allocate(5, heap(1, 2));
        let d1 = r.allocate(5, heap(1, 3));
        r.take(5, heap(1, 3));
        let col = RowLocator::Columnar {
            file_id: 11,
            sys_rowid: 7,
        };
        let d2 = r.allocate(5, col);
        let lake = RowLocator::Lake {
            file_id: 2,
            ordinal: 100,
        };
        let d3 = r.allocate(6, lake);

        let bytes = r.encode();
        let back = DocRegistry::decode(&bytes).expect("decode");
        assert_eq!(back.locator(5, d0), Some(heap(1, 2)));
        assert_eq!(back.locator(5, d1), None, "tombstone survives");
        assert_eq!(back.locator(5, d2), Some(col));
        assert_eq!(back.doc_for(5, col), Some(d2));
        assert_eq!(back.locator(6, d3), Some(lake));
        assert_eq!(
            back.ordinal_count(5),
            3,
            "next ordinal stable across restart"
        );
        // an ordinal allocated after restart does not collide
        assert_eq!(back.allocate(5, heap(9, 9)), 3);
    }

    #[test]
    fn decode_rejects_corrupt_input() {
        assert!(DocRegistry::decode(b"").is_none());
        assert!(DocRegistry::decode(b"ZYDOX\x01\x00\x00\x00\x00").is_none());
        let r = DocRegistry::new();
        r.allocate(1, heap(0, 0));
        let mut bytes = r.encode();
        bytes.truncate(bytes.len() - 3);
        assert!(DocRegistry::decode(&bytes).is_none());
    }

    /// How many forward slots currently point at `locator`. One row has one
    /// document, so this is 1 for a live row and 0 for a retired one.
    fn documents_for(r: &DocRegistry, table_id: u32, locator: RowLocator) -> usize {
        (0..r.ordinal_count(table_id))
            .filter(|d| r.locator(table_id, *d) == Some(locator))
            .count()
    }

    /// A fold re-pointing a row and an index build allocating for that same
    /// row must not leave it with two documents.
    ///
    /// `repoint` used to publish the new locator after releasing the forward
    /// lock. An `allocate` for that locator could land in the gap, take a
    /// fresh ordinal, and leave two forward slots pointing at one row: the
    /// row comes back twice from a search, and the loser is unreachable
    /// through the reverse map so no later `take` can free it. The fold
    /// writes columnar locators while an index build allocates for the rows
    /// it scans, which is where the two meet in production.
    #[test]
    fn concurrent_repoint_and_allocate_leave_one_document_per_row() {
        use std::sync::Arc;

        for round in 0..300u64 {
            let r = Arc::new(DocRegistry::new());
            let old = heap(round, 1);
            let new = RowLocator::Columnar {
                file_id: 9,
                sys_rowid: round,
            };
            let doc = r.allocate(1, old);

            let barrier = Arc::new(std::sync::Barrier::new(2));
            let repointer = {
                let (r, barrier) = (Arc::clone(&r), Arc::clone(&barrier));
                std::thread::spawn(move || {
                    barrier.wait();
                    r.repoint(1, old, new)
                })
            };
            let allocator = {
                let (r, barrier) = (Arc::clone(&r), Arc::clone(&barrier));
                std::thread::spawn(move || {
                    barrier.wait();
                    r.allocate(1, new)
                })
            };
            let repointed = repointer.join().expect("repoint thread");
            let allocated = allocator.join().expect("allocate thread");

            assert!(repointed, "the old locator had a live document");
            assert_eq!(
                documents_for(&r, 1, new),
                1,
                "round {round}: one row, one document"
            );
            assert_eq!(
                documents_for(&r, 1, old),
                0,
                "round {round}: the old locator is vacated"
            );
            // Whichever order the two ran in, the reverse map and the forward
            // array agree, and the allocator was handed the live ordinal
            assert_eq!(r.doc_for(1, new), Some(allocated), "round {round}");
            assert_eq!(r.locator(1, allocated), Some(new), "round {round}");
            // When the re-point won it carried the original document across.
            // When the allocate won, the row was already identified and the
            // original was retired instead of doubling up
            if allocated != doc {
                assert_eq!(
                    r.locator(1, doc),
                    None,
                    "round {round}: the superseded document is retired"
                );
            }
        }
    }

    /// Concurrent allocates for one row hand back one ordinal, and for
    /// distinct rows hand back distinct ones.
    #[test]
    fn concurrent_allocate_is_one_document_per_row() {
        use std::sync::Arc;

        for round in 0..200u64 {
            let r = Arc::new(DocRegistry::new());
            let shared = heap(round, 0);
            let barrier = Arc::new(std::sync::Barrier::new(8));
            let mut handles = Vec::new();
            for t in 0..8u16 {
                let (r, barrier) = (Arc::clone(&r), Arc::clone(&barrier));
                handles.push(std::thread::spawn(move || {
                    barrier.wait();
                    // Half contend on one row, half take rows of their own
                    let loc = if t % 2 == 0 { shared } else { heap(round, t) };
                    (loc, r.allocate(1, loc))
                }));
            }
            let got: Vec<(RowLocator, u64)> =
                handles.into_iter().map(|h| h.join().expect("thread")).collect();

            for (loc, doc) in &got {
                assert_eq!(r.locator(1, *doc), Some(*loc), "round {round}");
                assert_eq!(documents_for(&r, 1, *loc), 1, "round {round}");
            }
            let shared_docs: Vec<u64> = got
                .iter()
                .filter(|(l, _)| *l == shared)
                .map(|(_, d)| *d)
                .collect();
            assert!(
                shared_docs.windows(2).all(|w| w[0] == w[1]),
                "round {round}: one row cannot have two documents: {shared_docs:?}"
            );
        }
    }

    /// A delete racing an allocate for the same row never hands the caller a
    /// document whose forward slot has already been cleared.
    #[test]
    fn concurrent_take_and_allocate_stay_consistent() {
        use std::sync::Arc;

        for round in 0..300u64 {
            let r = Arc::new(DocRegistry::new());
            let loc = heap(round, 3);
            r.allocate(1, loc);

            let barrier = Arc::new(std::sync::Barrier::new(2));
            let taker = {
                let (r, barrier) = (Arc::clone(&r), Arc::clone(&barrier));
                std::thread::spawn(move || {
                    barrier.wait();
                    r.take(1, loc)
                })
            };
            let allocator = {
                let (r, barrier) = (Arc::clone(&r), Arc::clone(&barrier));
                std::thread::spawn(move || {
                    barrier.wait();
                    r.allocate(1, loc)
                })
            };
            let taken = taker.join().expect("take thread");
            let allocated = allocator.join().expect("allocate thread");

            assert!(taken.is_some(), "round {round}: the row had a document");
            // The allocate either ran first and got the doc the take then
            // retired, or ran after and minted a fresh one. Either way the
            // document it was handed has to describe the row it asked about
            match r.locator(1, allocated) {
                Some(l) => assert_eq!(l, loc, "round {round}"),
                None => assert_eq!(
                    Some(allocated),
                    taken,
                    "round {round}: only the retired document may be empty"
                ),
            }
            assert!(
                documents_for(&r, 1, loc) <= 1,
                "round {round}: a row never holds two documents"
            );
        }
    }
}

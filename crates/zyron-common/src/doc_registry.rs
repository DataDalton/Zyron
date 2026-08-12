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
    pub fn take(&self, table_id: u32, locator: RowLocator) -> Option<u64> {
        let t = self.tables.read_sync(&table_id, |_, t| Arc::clone(t))?;
        let (_, ordinal) = t.reverse.remove_sync(&locator)?;
        let mut fwd = t.forward.write().expect("doc registry forward lock");
        if let Some(slot) = fwd.get_mut(ordinal as usize) {
            *slot = None;
        }
        Some(ordinal)
    }

    /// Re-points a document at the row's new storage location, keeping its
    /// DocId. Called by the fold when a heap row moves into a columnar
    /// segment, this is what keeps search indexes valid across folding.
    /// Returns false when the old locator had no live document.
    pub fn repoint(&self, table_id: u32, old: RowLocator, new: RowLocator) -> bool {
        let Some(t) = self.tables.read_sync(&table_id, |_, t| Arc::clone(t)) else {
            return false;
        };
        let Some((_, ordinal)) = t.reverse.remove_sync(&old) else {
            return false;
        };
        let mut fwd = t.forward.write().expect("doc registry forward lock");
        if let Some(slot) = fwd.get_mut(ordinal as usize) {
            *slot = Some(new);
        }
        drop(fwd);
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
}

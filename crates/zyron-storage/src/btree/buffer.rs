//! Write buffer with Swiss Tables hash table for B+Tree.

use std::simd::prelude::*;
use crate::tuple::TupleId;
use super::arena_index::BTreeArenaIndex;
use super::constants::{CTRL_DELETED, CTRL_EMPTY, GROUP_SIZE, HASH_TABLE_SIZE, WRITE_BUFFER_CAPACITY};
use super::types::InsertStats;
use zyron_common::Result;

struct WriteBufferEntry {
    key: u64,
    packed_tuple_id: u64,
}

/// Swiss Tables-style hash table with control bytes.
/// Uses 32-way parallel control byte comparison via portable SIMD.
///
/// Memory layout:
/// - ctrl: 1 byte per slot (h2 hash or empty/deleted marker)
/// - keys: u64 per slot
/// - values: u64 per slot
///
/// Probe control bytes first (cache-friendly), compare full keys on match.
#[repr(C, align(32))]
struct SwissTable {
    /// Control bytes: h2 (7-bit hash) or CTRL_EMPTY/CTRL_DELETED.
    ctrl: Box<[u8]>,
    /// Keys array.
    keys: Box<[u64]>,
    /// Values array (packed tuple IDs).
    values: Box<[u64]>,
    /// Number of entries.
    len: usize,
}

impl SwissTable {
    /// Creates a new empty hash table.
    fn new() -> Self {
        Self {
            ctrl: vec![CTRL_EMPTY; HASH_TABLE_SIZE].into_boxed_slice(),
            keys: vec![0u64; HASH_TABLE_SIZE].into_boxed_slice(),
            values: vec![0u64; HASH_TABLE_SIZE].into_boxed_slice(),
            len: 0,
        }
    }

    /// Primary hash (h1) - determines slot index.
    #[inline(always)]
    fn h1(hash: u64) -> usize {
        hash as usize & (HASH_TABLE_SIZE - 1)
    }

    /// Secondary hash (h2) - 7-bit control byte value (0-127).
    #[inline(always)]
    fn h2(hash: u64) -> u8 {
        // Use top 7 bits for h2 (different bits than h1)
        ((hash >> 57) & 0x7F) as u8
    }

    /// Full hash using FxHash-style multiply-XOR.
    /// Faster than Fibonacci for random input while maintaining good distribution.
    #[inline(always)]
    fn hash(key: u64) -> u64 {
        // FxHash constant: highly non-linear mixing
        const K: u64 = 0x517cc1b727220a95;
        key.wrapping_mul(K)
    }

    /// Returns the number of entries.
    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }

    /// Returns true if at capacity.
    #[inline(always)]
    fn is_full(&self) -> bool {
        self.len >= WRITE_BUFFER_CAPACITY
    }

    /// SIMD search using control bytes. Compares 32 control bytes at once.
    #[inline(always)]
    fn search(&self, key: u64) -> Option<u64> {
        let hash = Self::hash(key);
        let h1 = Self::h1(hash);
        let h2 = Self::h2(hash);

        // Create SIMD vector of h2 for comparison
        let h2_vec: Simd<u8, GROUP_SIZE> = Simd::splat(h2);
        let empty_vec: Simd<u8, GROUP_SIZE> = Simd::splat(CTRL_EMPTY);

        let mut group_idx = h1 & !(GROUP_SIZE - 1);
        let start_group = group_idx;

        loop {
            // Load 32 control bytes at once
            let ctrl_slice = &self.ctrl[group_idx..group_idx + GROUP_SIZE];
            let ctrl_vec = Simd::from_slice(ctrl_slice);

            // Find slots with matching h2 (potential key matches)
            let match_mask = ctrl_vec.simd_eq(h2_vec);

            // Check each potential match
            let mut bits = match_mask.to_bitmask();
            while bits != 0 {
                let lane = bits.trailing_zeros() as usize;
                let slot = group_idx + lane;

                // Compare full key only on h2 match
                if self.keys[slot] == key {
                    return Some(self.values[slot]);
                }

                bits &= bits - 1; // Clear lowest set bit
            }

            // Check for empty slots (key definitely not present)
            let empty_mask = ctrl_vec.simd_eq(empty_vec);
            if empty_mask.any() {
                return None;
            }

            // Move to next group
            group_idx = (group_idx + GROUP_SIZE) & (HASH_TABLE_SIZE - 1);

            // Full table scan completed
            if group_idx == start_group {
                return None;
            }
        }
    }

    /// Insert a key-value pair using SIMD group-based probing.
    /// Pure SIMD path without branching fast-paths for consistent performance.
    #[inline(always)]
    fn insert(&mut self, key: u64, value: u64) {
        let hash = Self::hash(key);
        let h1 = Self::h1(hash);
        let h2 = Self::h2(hash);

        let h2_vec: Simd<u8, GROUP_SIZE> = Simd::splat(h2);
        let empty_vec: Simd<u8, GROUP_SIZE> = Simd::splat(CTRL_EMPTY);

        let mut group_idx = h1 & !(GROUP_SIZE - 1);

        loop {
            let ctrl_slice = &self.ctrl[group_idx..group_idx + GROUP_SIZE];
            let ctrl_vec = Simd::from_slice(ctrl_slice);

            // Check for existing key (same h2, then compare full key)
            let match_mask = ctrl_vec.simd_eq(h2_vec);
            let mut bits = match_mask.to_bitmask();
            while bits != 0 {
                let lane = bits.trailing_zeros() as usize;
                let slot = group_idx + lane;
                if self.keys[slot] == key {
                    self.values[slot] = value;
                    return;
                }
                bits &= bits - 1;
            }

            // Find first empty slot in this group
            let empty_mask = ctrl_vec.simd_eq(empty_vec);
            if empty_mask.any() {
                let bits = empty_mask.to_bitmask();
                let lane = bits.trailing_zeros() as usize;
                let slot = group_idx + lane;
                self.ctrl[slot] = h2;
                self.keys[slot] = key;
                self.values[slot] = value;
                self.len += 1;
                return;
            }

            group_idx = (group_idx + GROUP_SIZE) & (HASH_TABLE_SIZE - 1);
        }
    }

    /// Drains all entries sorted by key (for merging into B+Tree).
    fn drain_sorted(&mut self) -> Vec<WriteBufferEntry> {
        let mut entries = Vec::with_capacity(self.len);

        for i in 0..HASH_TABLE_SIZE {
            let ctrl = self.ctrl[i];
            if ctrl != CTRL_EMPTY && ctrl != CTRL_DELETED {
                entries.push(WriteBufferEntry {
                    key: self.keys[i],
                    packed_tuple_id: self.values[i],
                });
                self.ctrl[i] = CTRL_EMPTY;
            }
        }

        self.len = 0;
        entries.sort_unstable_by_key(|e| e.key);
        entries
    }

    /// Returns an iterator over entries (unsorted).
    fn iter(&self) -> impl Iterator<Item = WriteBufferEntry> + '_ {
        (0..HASH_TABLE_SIZE)
            .filter(move |&i| self.ctrl[i] != CTRL_EMPTY && self.ctrl[i] != CTRL_DELETED)
            .map(move |i| WriteBufferEntry {
                key: self.keys[i],
                packed_tuple_id: self.values[i],
            })
    }
}

/// Write buffer using Swiss Tables-style hash table.
/// Provides O(1) insert and lookup with 32-way parallel control byte comparison.
struct WriteBuffer {
    table: SwissTable,
}

impl WriteBuffer {
    /// Creates a new write buffer.
    fn new() -> Self {
        Self {
            table: SwissTable::new(),
        }
    }

    /// Returns true if buffer is at capacity.
    #[inline(always)]
    fn is_full(&self) -> bool {
        self.table.is_full()
    }

    /// Returns the number of entries.
    #[inline(always)]
    fn len(&self) -> usize {
        self.table.len()
    }

    /// O(1) lookup with 32-way SIMD control byte comparison.
    #[inline(always)]
    fn search(&self, key: u64) -> Option<u64> {
        self.table.search(key)
    }

    /// O(1) insert a key-value pair.
    #[inline(always)]
    fn insert(&mut self, key: u64, packed_tuple_id: u64) {
        self.table.insert(key, packed_tuple_id);
    }

    /// Returns entries sorted by key (for merging into B+Tree).
    fn drain_sorted(&mut self) -> Vec<WriteBufferEntry> {
        self.table.drain_sorted()
    }

    /// Returns an iterator over entries (unsorted, for range scans).
    fn iter(&self) -> impl Iterator<Item = WriteBufferEntry> + '_ {
        self.table.iter()
    }
}

/// Buffered B+Tree index with write buffer for high insert throughput.
///
/// Architecture:
/// - Writes go to an in-memory write buffer (~100ns insert)
/// - When buffer is full, entries are bulk-merged into the B+Tree
/// - Reads check the write buffer first, then the B+Tree
/// - B+Tree uses 32KB nodes for shallow height (height=2 for 1M keys)
///
/// Performance characteristics:
/// - Insert: ~100ns (buffer insert with binary search)
/// - Lookup: ~80ns (buffer check + B+Tree search)
/// - Range scan: Merges results from buffer and B+Tree
pub struct BufferedBTreeIndex {
    /// The underlying B+Tree (read-optimized with 32KB nodes).
    btree: BTreeArenaIndex,
    /// Write buffer for absorbing inserts.
    buffer: WriteBuffer,
    /// Fast empty-check flag. Avoids HashMap.len() call on every search.
    buffer_has_data: bool,
    /// Performance statistics for profiling.
    stats: InsertStats,
}

impl BufferedBTreeIndex {
    /// Creates a new buffered B+Tree index.
    /// Capacity is the maximum number of B+Tree nodes (each 32KB).
    pub fn new(capacity_nodes: usize) -> Self {
        Self {
            btree: BTreeArenaIndex::new(capacity_nodes),
            buffer: WriteBuffer::new(),
            buffer_has_data: false,
            stats: InsertStats::default(),
        }
    }

    /// Returns performance statistics for profiling.
    pub fn stats(&self) -> &InsertStats {
        &self.stats
    }

    /// Resets performance statistics.
    pub fn reset_stats(&mut self) {
        self.stats = InsertStats::default();
    }

    /// Returns the tree height (of the underlying B+Tree).
    #[inline]
    pub fn height(&self) -> u64 {
        self.btree.height()
    }

    /// Returns the number of entries in the buffer.
    #[inline]
    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }

    /// Search for a key. Checks write buffer first, then B+Tree.
    #[inline(always)]
    pub fn search(&self, key: &[u8]) -> Option<TupleId> {
        // Fast path: skip buffer check when empty (common after flush)
        // Uses bool flag instead of HashMap.len() for ~15ns savings
        if self.buffer_has_data {
            let search_key = self.key_to_u64(key);
            if let Some(packed) = self.buffer.search(search_key) {
                return Some(BTreeArenaIndex::unpack_tuple_id(packed));
            }
        }

        // Fall through to B+Tree
        self.btree.search(key)
    }

    /// Insert a key-value pair.
    /// Goes to write buffer first; flushes to B+Tree when buffer is full.
    #[inline(always)]
    pub fn insert(&mut self, key: &[u8], tuple_id: TupleId) -> Result<()> {
        // Check if buffer is full and needs flush (rare case)
        if self.buffer.is_full() {
            self.flush_buffer()?;
        }

        // Convert key and pack tuple_id
        let search_key = self.key_to_u64(key);
        let packed = BTreeArenaIndex::pack_tuple_id(tuple_id);

        // Insert into buffer and mark as non-empty
        self.buffer.insert(search_key, packed);
        self.buffer_has_data = true;
        Ok(())
    }

    /// Flush write buffer to B+Tree.
    /// Drains and sorts buffer entries, then bulk inserts into the tree.
    fn flush_buffer(&mut self) -> Result<()> {
        if !self.buffer_has_data {
            return Ok(());
        }

        let flush_start = std::time::Instant::now();

        // Drain and sort entries for sequential B+Tree insertion
        let drain_start = std::time::Instant::now();
        let entries = self.buffer.drain_sorted();
        let drain_elapsed = drain_start.elapsed();

        // Bulk insert using cursor-based API (passes u64 directly, no conversion)
        let btree_start = std::time::Instant::now();
        let bulk_entries: Vec<(u64, u64)> = entries
            .iter()
            .map(|e| (e.key, e.packed_tuple_id))
            .collect();
        self.btree.insert_bulk_sorted(&bulk_entries)?;
        let btree_elapsed = btree_start.elapsed();

        // Update stats
        self.stats.flush_count += 1;
        self.stats.flush_time_ns += flush_start.elapsed().as_nanos() as u64;
        self.stats.drain_time_ns += drain_elapsed.as_nanos() as u64;
        self.stats.btree_insert_time_ns += btree_elapsed.as_nanos() as u64;

        // Mark buffer as empty
        self.buffer_has_data = false;
        Ok(())
    }

    /// Force flush the write buffer to B+Tree.
    pub fn flush(&mut self) -> Result<()> {
        self.flush_buffer()
    }

    /// Range scan from start_key to end_key (inclusive).
    /// Merges results from write buffer and B+Tree.
    #[inline]
    pub fn range_scan(
        &self,
        start_key: Option<&[u8]>,
        end_key: Option<&[u8]>,
    ) -> Vec<(u64, TupleId)> {
        // Fast path: skip merge when buffer is empty (common after flush)
        if !self.buffer_has_data {
            return self.btree.range_scan(start_key, end_key);
        }

        let start = start_key.map(|k| self.key_to_u64(k)).unwrap_or(0);
        let end = end_key.map(|k| self.key_to_u64(k)).unwrap_or(u64::MAX);

        // Get results from B+Tree (already sorted)
        let btree_results = self.btree.range_scan(start_key, end_key);

        // Get results from buffer that fall in range, then sort
        let mut buffer_results: Vec<_> = self
            .buffer
            .iter()
            .filter(|e| e.key >= start && e.key <= end)
            .map(|e| (e.key, BTreeArenaIndex::unpack_tuple_id(e.packed_tuple_id)))
            .collect();
        buffer_results.sort_unstable_by_key(|(k, _)| *k);

        // Merge sorted results (buffer entries override B+Tree for same key)
        let mut merged = Vec::with_capacity(btree_results.len() + buffer_results.len());
        let mut btree_iter = btree_results.into_iter().peekable();
        let mut buffer_iter = buffer_results.into_iter().peekable();

        loop {
            match (btree_iter.peek(), buffer_iter.peek()) {
                (Some(&(bk, _)), Some(&(bufk, _))) => {
                    if bk < bufk {
                        merged.push(btree_iter.next().unwrap());
                    } else if bk > bufk {
                        merged.push(buffer_iter.next().unwrap());
                    } else {
                        // Same key - buffer wins (more recent)
                        btree_iter.next();
                        merged.push(buffer_iter.next().unwrap());
                    }
                }
                (Some(_), None) => {
                    merged.push(btree_iter.next().unwrap());
                }
                (None, Some(_)) => {
                    merged.push(buffer_iter.next().unwrap());
                }
                (None, None) => break,
            }
        }

        merged
    }

    /// Returns the total number of entries (buffer + B+Tree).
    pub fn count(&self) -> usize {
        self.buffer.len() + self.btree.count()
    }

    /// Convert key bytes to u64 (big-endian for sort order).
    #[inline(always)]
    fn key_to_u64(&self, key: &[u8]) -> u64 {
        if key.len() >= 8 {
            u64::from_be_bytes([
                key[0], key[1], key[2], key[3], key[4], key[5], key[6], key[7],
            ])
        } else {
            let mut padded = [0u8; 8];
            padded[..key.len()].copy_from_slice(key);
            u64::from_be_bytes(padded)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::page::{BTreeInternalPage, BTreeLeafPage};
    use super::super::types::{DeleteResult, InternalEntry, InternalPageHeader, LeafEntry, LeafPageHeader};
    use bytes::Bytes;
    use zyron_common::page::{PageId, PAGE_SIZE};
    use zyron_common::ZyronError;

    #[test]
    fn test_leaf_header_roundtrip() {
        let header = LeafPageHeader {
            num_slots: 42,
            data_end: 100,
            next_leaf: 12345,
            reserved: 0,
        };

        let bytes = header.to_bytes();
        let recovered = LeafPageHeader::from_bytes(&bytes);

        assert_eq!(recovered.num_slots, 42);
        assert_eq!(recovered.data_end, 100);
        assert_eq!(recovered.next_leaf, 12345);
    }

    #[test]
    fn test_internal_header_roundtrip() {
        let header = InternalPageHeader {
            num_keys: 10,
            free_space_offset: 200,
            level: 2,
            reserved: [0; 10],
        };

        let bytes = header.to_bytes();
        let recovered = InternalPageHeader::from_bytes(&bytes);

        assert_eq!(recovered.num_keys, 10);
        assert_eq!(recovered.free_space_offset, 200);
        assert_eq!(recovered.level, 2);
    }

    #[test]
    fn test_leaf_entry_roundtrip() {
        let entry = LeafEntry {
            key: Bytes::from_static(b"test_key"),
            tuple_id: TupleId::new(PageId::new(1, 42), 5),
        };

        let bytes = entry.to_bytes();
        let (recovered, consumed) = LeafEntry::from_bytes(&bytes).unwrap();

        assert_eq!(recovered.key, entry.key);
        assert_eq!(recovered.tuple_id.page_id.file_id, 1);
        assert_eq!(recovered.tuple_id.page_id.page_num, 42);
        assert_eq!(recovered.tuple_id.slot_id, 5);
        assert_eq!(consumed, bytes.len());
    }

    #[test]
    fn test_internal_entry_roundtrip() {
        let entry = InternalEntry {
            key: Bytes::from_static(b"separator"),
            child_page_id: PageId::new(2, 100),
        };

        let bytes = entry.to_bytes();
        let (recovered, consumed) = InternalEntry::from_bytes(&bytes).unwrap();

        assert_eq!(recovered.key, entry.key);
        assert_eq!(recovered.child_page_id.file_id, 2);
        assert_eq!(recovered.child_page_id.page_num, 100);
        assert_eq!(consumed, bytes.len());
    }

    #[test]
    fn test_leaf_page_new() {
        let page = BTreeLeafPage::new(PageId::new(0, 0));

        assert_eq!(page.num_entries(), 0);
        assert!(page.free_space() > 0);
        assert!(page.next_leaf().is_none());
    }

    #[test]
    fn test_leaf_page_insert_and_get() {
        let mut page = BTreeLeafPage::new(PageId::new(0, 0));

        let key = Bytes::from_static(b"hello");
        let tuple_id = TupleId::new(PageId::new(1, 10), 5);

        page.insert(key.clone(), tuple_id).unwrap();

        assert_eq!(page.num_entries(), 1);
        assert_eq!(page.get(&key), Some(tuple_id));
    }

    #[test]
    fn test_leaf_page_insert_multiple_sorted() {
        let mut page = BTreeLeafPage::new(PageId::new(0, 0));

        // Insert in random order
        page.insert(
            Bytes::from_static(b"charlie"),
            TupleId::new(PageId::new(0, 0), 3),
        )
        .unwrap();
        page.insert(
            Bytes::from_static(b"alpha"),
            TupleId::new(PageId::new(0, 0), 1),
        )
        .unwrap();
        page.insert(
            Bytes::from_static(b"bravo"),
            TupleId::new(PageId::new(0, 0), 2),
        )
        .unwrap();

        assert_eq!(page.num_entries(), 3);

        // Should be stored in sorted order
        let entries = page.entries();
        assert_eq!(entries[0].key.as_ref(), b"alpha");
        assert_eq!(entries[1].key.as_ref(), b"bravo");
        assert_eq!(entries[2].key.as_ref(), b"charlie");
    }

    #[test]
    fn test_leaf_page_duplicate_key() {
        let mut page = BTreeLeafPage::new(PageId::new(0, 0));

        page.insert(
            Bytes::from_static(b"key"),
            TupleId::new(PageId::new(0, 0), 1),
        )
        .unwrap();

        let result = page.insert(
            Bytes::from_static(b"key"),
            TupleId::new(PageId::new(0, 0), 2),
        );
        assert!(matches!(result, Err(ZyronError::DuplicateKey)));
    }

    #[test]
    fn test_leaf_page_delete() {
        let mut page = BTreeLeafPage::new(PageId::new(0, 0));

        page.insert(
            Bytes::from_static(b"key1"),
            TupleId::new(PageId::new(0, 0), 1),
        )
        .unwrap();
        page.insert(
            Bytes::from_static(b"key2"),
            TupleId::new(PageId::new(0, 0), 2),
        )
        .unwrap();

        // Delete returns DeleteResult, check for Underfull since page has little data
        let result = page.delete(b"key1");
        assert!(result == DeleteResult::Ok || result == DeleteResult::Underfull);
        assert_eq!(page.num_entries(), 1);
        assert!(page.get(b"key1").is_none());
        assert!(page.get(b"key2").is_some());

        assert_eq!(page.delete(b"nonexistent"), DeleteResult::NotFound);
    }

    #[test]
    fn test_leaf_page_is_underfull() {
        let mut page = BTreeLeafPage::new(PageId::new(0, 0));

        // Empty page with 0 entries is not considered underfull (special case)
        assert!(!page.is_underfull());

        // Insert a single small entry
        page.insert(
            Bytes::from_static(b"key"),
            TupleId::new(PageId::new(0, 0), 1),
        )
        .unwrap();

        // A page with only one small entry is underfull (below 50% capacity)
        assert!(page.is_underfull());
    }

    #[test]
    fn test_leaf_page_borrow_from_right() {
        let mut left = BTreeLeafPage::new(PageId::new(0, 0));
        let mut right = BTreeLeafPage::new(PageId::new(0, 1));

        // Set up left page with one entry
        left.insert(Bytes::from_static(b"a"), TupleId::new(PageId::new(0, 0), 1))
            .unwrap();

        // Set up right page with multiple entries
        right
            .insert(Bytes::from_static(b"b"), TupleId::new(PageId::new(0, 0), 2))
            .unwrap();
        right
            .insert(Bytes::from_static(b"c"), TupleId::new(PageId::new(0, 0), 3))
            .unwrap();

        // Borrow from right
        let new_separator = left.borrow_from_right(&mut right);
        assert!(new_separator.is_some());
        assert_eq!(new_separator.unwrap().as_ref(), b"c");

        // Left should now have 2 entries
        assert_eq!(left.num_entries(), 2);
        // Right should now have 1 entry
        assert_eq!(right.num_entries(), 1);
    }

    #[test]
    fn test_leaf_page_merge_with_right() {
        let mut left = BTreeLeafPage::new(PageId::new(0, 0));
        let mut right = BTreeLeafPage::new(PageId::new(0, 1));

        left.insert(Bytes::from_static(b"a"), TupleId::new(PageId::new(0, 0), 1))
            .unwrap();
        right
            .insert(Bytes::from_static(b"b"), TupleId::new(PageId::new(0, 0), 2))
            .unwrap();

        // Link pages
        left.set_next_leaf(Some(PageId::new(0, 1)));
        right.set_next_leaf(Some(PageId::new(0, 2)));

        // Merge right into left
        assert!(left.merge_with_right(&mut right));

        // Left should now have both entries
        assert_eq!(left.num_entries(), 2);
        // Left's next should point past the merged sibling
        assert_eq!(left.next_leaf(), Some(PageId::new(0, 2)));
    }

    #[test]
    fn test_internal_page_is_underfull() {
        let mut page = BTreeInternalPage::new(PageId::new(0, 0), 0);
        page.set_leftmost_child(PageId::new(1, 0));

        // Empty internal page (no keys) is not considered underfull
        assert!(!page.is_underfull());

        // Insert a single small entry
        page.insert(Bytes::from_static(b"key"), PageId::new(1, 1))
            .unwrap();

        // A page with only one small entry is underfull (below 50% capacity)
        assert!(page.is_underfull());
    }

    #[test]
    fn test_internal_page_delete() {
        let mut page = BTreeInternalPage::new(PageId::new(0, 0), 0);
        page.set_leftmost_child(PageId::new(1, 0));

        page.insert(Bytes::from_static(b"key1"), PageId::new(1, 1))
            .unwrap();
        page.insert(Bytes::from_static(b"key2"), PageId::new(1, 2))
            .unwrap();

        assert_eq!(page.num_keys(), 2);

        // Delete a key
        let result = page.delete(b"key1");
        assert!(result == DeleteResult::Ok || result == DeleteResult::Underfull);
        assert_eq!(page.num_keys(), 1);

        // Delete nonexistent key
        assert_eq!(page.delete(b"nonexistent"), DeleteResult::NotFound);
    }

    #[test]
    fn test_leaf_page_split() {
        let mut page = BTreeLeafPage::new(PageId::new(0, 0));

        // Insert many entries
        for i in 0..100 {
            let key = Bytes::from(format!("key_{:03}", i));
            let _ = page.insert(key, TupleId::new(PageId::new(0, 0), i as u16));
        }

        let entries_before = page.num_entries();
        let (split_key, right_page) = page.split(PageId::new(0, 1));

        let left_entries = page.num_entries();
        let right_entries = right_page.num_entries();

        assert_eq!(left_entries + right_entries, entries_before);
        assert!(left_entries > 0);
        assert!(right_entries > 0);

        // Split key should be the first key of right page
        let right_first = right_page.entries()[0].key.clone();
        assert_eq!(split_key, right_first);

        // Pages should be linked
        assert_eq!(page.next_leaf(), Some(PageId::new(0, 1)));
    }

    #[test]
    fn test_leaf_page_from_bytes() {
        let mut page = BTreeLeafPage::new(PageId::new(0, 0));
        page.insert(
            Bytes::from_static(b"test"),
            TupleId::new(PageId::new(1, 2), 3),
        )
        .unwrap();

        let bytes = *page.as_bytes();
        let recovered = BTreeLeafPage::from_bytes(bytes);

        assert_eq!(recovered.num_entries(), 1);
        assert_eq!(
            recovered.get(b"test"),
            Some(TupleId::new(PageId::new(1, 2), 3))
        );
    }

    #[test]
    fn test_internal_page_new() {
        let page = BTreeInternalPage::new(PageId::new(0, 0), 1);

        assert_eq!(page.num_keys(), 0);
        assert_eq!(page.level(), 1);
        assert!(page.free_space() > 0);
    }

    #[test]
    fn test_internal_page_leftmost_child() {
        let mut page = BTreeInternalPage::new(PageId::new(0, 0), 0);
        page.set_leftmost_child(PageId::new(1, 100));

        assert_eq!(page.leftmost_child(), PageId::new(1, 100));
    }

    #[test]
    fn test_internal_page_insert_and_find() {
        let mut page = BTreeInternalPage::new(PageId::new(0, 0), 0);
        page.set_leftmost_child(PageId::new(1, 0));

        page.insert(Bytes::from_static(b"key1"), PageId::new(1, 1))
            .unwrap();
        page.insert(Bytes::from_static(b"key2"), PageId::new(1, 2))
            .unwrap();

        assert_eq!(page.num_keys(), 2);

        // Keys less than "key1" go to leftmost child
        assert_eq!(page.find_child(b"aaa"), PageId::new(1, 0));

        // Keys >= "key2" go to the right child
        assert_eq!(page.find_child(b"key2"), PageId::new(1, 2));
        assert_eq!(page.find_child(b"zzz"), PageId::new(1, 2));
    }

    #[test]
    fn test_internal_page_split() {
        let mut page = BTreeInternalPage::new(PageId::new(0, 0), 0);
        page.set_leftmost_child(PageId::new(1, 0));

        for i in 0..50 {
            let key = Bytes::from(format!("key_{:03}", i));
            let _ = page.insert(key, PageId::new(1, i + 1));
        }

        let keys_before = page.num_keys();
        let (promoted_key, right_page) = page.split(PageId::new(0, 1));

        let left_keys = page.num_keys();
        let right_keys = right_page.num_keys();

        // The promoted key is not counted in either side
        assert_eq!(left_keys + right_keys + 1, keys_before);
        assert!(left_keys > 0);
        assert!(right_keys > 0);
        assert!(!promoted_key.is_empty());
    }

    #[test]
    fn test_leaf_page_next_leaf_chain() {
        let mut page1 = BTreeLeafPage::new(PageId::new(0, 0));
        let mut page2 = BTreeLeafPage::new(PageId::new(0, 1));
        let page3 = BTreeLeafPage::new(PageId::new(0, 2));

        page1.set_next_leaf(Some(PageId::new(0, 1)));
        page2.set_next_leaf(Some(PageId::new(0, 2)));

        assert_eq!(page1.next_leaf(), Some(PageId::new(0, 1)));
        assert_eq!(page2.next_leaf(), Some(PageId::new(0, 2)));
        assert_eq!(page3.next_leaf(), None);
    }

    #[test]
    fn test_leaf_entry_size_on_disk() {
        let entry = LeafEntry {
            key: Bytes::from_static(b"hello"),
            tuple_id: TupleId::new(PageId::new(0, 0), 0),
        };

        // 2 (key_len) + 5 (key) + 10 (tuple_id) = 17
        assert_eq!(entry.size_on_disk(), 17);
    }

    #[test]
    fn test_internal_entry_size_on_disk() {
        let entry = InternalEntry {
            key: Bytes::from_static(b"hello"),
            child_page_id: PageId::new(0, 0),
        };

        // 2 (key_len) + 5 (key) + 8 (page_id) = 15
        assert_eq!(entry.size_on_disk(), 15);
    }

    #[test]
    fn test_leaf_page_can_fit() {
        let page = BTreeLeafPage::new(PageId::new(0, 0));

        // Should be able to fit small entries
        assert!(page.can_fit(100));

        // Should not be able to fit entries larger than page
        assert!(!page.can_fit(PAGE_SIZE));
    }

    #[test]
    fn test_internal_page_can_fit() {
        let page = BTreeInternalPage::new(PageId::new(0, 0), 0);

        // Should be able to fit small entries
        assert!(page.can_fit(100));

        // Should not be able to fit entries larger than page
        assert!(!page.can_fit(PAGE_SIZE));
    }

    // =========================================================================
    // Arena-Based B+Tree Tests
    // =========================================================================

    #[test]
    fn test_arena_btree_basic_operations() {
        let mut tree = BTreeArenaIndex::new(1024);

        // Insert some keys
        let tid1 = TupleId::new(PageId::new(0, 1), 1);
        let tid2 = TupleId::new(PageId::new(0, 2), 2);
        let tid3 = TupleId::new(PageId::new(0, 3), 3);

        tree.insert(&100u64.to_be_bytes(), tid1).unwrap();
        tree.insert(&200u64.to_be_bytes(), tid2).unwrap();
        tree.insert(&50u64.to_be_bytes(), tid3).unwrap();

        // Verify count
        assert_eq!(tree.count(), 3);

        // Search for keys
        assert_eq!(tree.search(&100u64.to_be_bytes()), Some(tid1));
        assert_eq!(tree.search(&200u64.to_be_bytes()), Some(tid2));
        assert_eq!(tree.search(&50u64.to_be_bytes()), Some(tid3));

        // Non-existent key
        assert_eq!(tree.search(&999u64.to_be_bytes()), None);
    }

    #[test]
    fn test_arena_btree_many_inserts() {
        let mut tree = BTreeArenaIndex::new(8192);

        // Insert 1000 keys
        for i in 0..1000u64 {
            let tid = TupleId::new(PageId::new(0, i as u32), i as u16);
            tree.insert(&i.to_be_bytes(), tid).unwrap();
        }

        assert_eq!(tree.count(), 1000);

        // Verify all keys can be found
        for i in 0..1000u64 {
            let tid = TupleId::new(PageId::new(0, i as u32), i as u16);
            assert_eq!(tree.search(&i.to_be_bytes()), Some(tid));
        }
    }

    #[test]
    fn test_arena_btree_range_scan() {
        let mut tree = BTreeArenaIndex::new(4096);

        // Insert keys 0, 10, 20, ..., 100
        for i in (0..=100u64).step_by(10) {
            let tid = TupleId::new(PageId::new(0, i as u32), 0);
            tree.insert(&i.to_be_bytes(), tid).unwrap();
        }

        // Range scan from 25 to 75 (should get 30, 40, 50, 60, 70)
        let results = tree.range_scan(Some(&25u64.to_be_bytes()), Some(&75u64.to_be_bytes()));
        assert_eq!(results.len(), 5);

        let keys: Vec<u64> = results.iter().map(|(k, _)| *k).collect();
        assert_eq!(keys, vec![30, 40, 50, 60, 70]);
    }
}

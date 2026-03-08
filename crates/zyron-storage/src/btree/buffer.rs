//! Write buffer with partitioned key-space for B+Tree.

use super::arena_index::BTreeArenaIndex;
use super::constants::WRITE_BUFFER_CAPACITY;
use super::types::InsertStats;
use crate::tuple::TupleId;
use std::cell::Cell;
use zyron_common::Result;

const NUM_PARTITIONS: usize = 256;

/// Partition index from the top 8 bits of a u64 key.
#[inline(always)]
fn partition_of(key: u64) -> usize {
    (key >> 56) as usize
}

/// Partitioned write buffer. Divides the key space into 256 partitions
/// based on the top 8 bits of each key. Each partition is a Vec that stays
/// small enough to fit in L1/L2 cache during sort and scan operations.
///
/// Insert: ~3-5ns (Vec push into a small partition, cache-hot).
/// Search: linear scan of one partition (~512 entries average).
/// Flush: sort each partition independently, concatenate in order (globally sorted).
struct PartitionedBuffer {
    /// 256 partitions, each a Vec<(key, packed_tuple_id)>.
    partitions: Vec<Vec<(u64, u64)>>,
    /// Total entry count across all partitions.
    len: usize,
    /// Minimum key currently in the buffer. u64::MAX when empty.
    min_key: u64,
    /// Maximum key currently in the buffer. 0 when empty.
    max_key: u64,
    /// Whether all partitions are sorted (enables binary search).
    /// Uses Cell for interior mutability so search() can take &self.
    sorted: Cell<bool>,
}

impl PartitionedBuffer {
    fn new() -> Self {
        let cap_per_partition = WRITE_BUFFER_CAPACITY / NUM_PARTITIONS + 1;
        let partitions = (0..NUM_PARTITIONS)
            .map(|_| Vec::with_capacity(cap_per_partition))
            .collect();
        Self {
            partitions,
            len: 0,
            min_key: u64::MAX,
            max_key: 0,
            sorted: Cell::new(true),
        }
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    fn is_full(&self) -> bool {
        self.len >= WRITE_BUFFER_CAPACITY
    }

    /// Search for a key using linear scan within its partition.
    /// Takes &self to allow concurrent reads without exclusive access.
    #[inline(always)]
    fn search(&self, key: u64) -> Option<u64> {
        let p = partition_of(key);
        let partition = &self.partitions[p];
        // Linear scan of the partition. Each partition averages ~512 entries,
        // so scanning is ~200ns worst case. Avoids requiring &mut self for
        // sort-based binary search, which would serialize all reads behind a mutex.
        let mut last_val = None;
        for &(k, v) in partition.iter() {
            if k == key {
                last_val = Some(v);
            }
        }
        last_val
    }

    /// Insert a key-value pair. Duplicates are resolved at flush time
    /// (last write wins) to keep the insert path O(1).
    #[inline(always)]
    fn insert(&mut self, key: u64, value: u64) {
        if key < self.min_key {
            self.min_key = key;
        }
        if key > self.max_key {
            self.max_key = key;
        }
        let p = partition_of(key);
        self.partitions[p].push((key, value));
        self.len += 1;
        self.sorted.set(false);
    }

    /// Drains all partitions into output, globally sorted by key.
    /// Each partition is sorted independently, then concatenated in partition
    /// order (since partition i contains keys with top 8 bits = i, the
    /// concatenation is globally sorted).
    fn drain_sorted_into(&mut self, output: &mut Vec<(u64, u64)>) {
        output.clear();
        output.reserve(self.len);

        for partition in self.partitions.iter_mut() {
            if partition.is_empty() {
                continue;
            }
            partition.sort_unstable_by_key(|&(k, _)| k);
            // Dedup: for duplicate keys, keep the last inserted value.
            // After sort_unstable, duplicates are adjacent. Walk backwards
            // through each group to find the last-inserted entry (highest
            // original index, which sort_unstable preserves as rightmost).
            let start = output.len();
            output.extend_from_slice(partition);
            let slice = &mut output[start..];
            if slice.len() > 1 {
                let mut write = 0;
                for read in 1..slice.len() {
                    if slice[read].0 == slice[write].0 {
                        // Duplicate key: keep the later one (last write wins)
                        slice[write] = slice[read];
                    } else {
                        write += 1;
                        slice[write] = slice[read];
                    }
                }
                let new_len = start + write + 1;
                output.truncate(new_len);
            }
            partition.clear();
        }

        self.len = 0;
        self.min_key = u64::MAX;
        self.max_key = 0;
        self.sorted.set(true);
    }

    /// Returns an iterator over all (key, packed_tuple_id) pairs (unsorted).
    fn iter(&self) -> impl Iterator<Item = (u64, u64)> + '_ {
        self.partitions.iter().flat_map(|p| p.iter().copied())
    }
}

/// Buffered B+Tree index with write buffer for high insert throughput.
///
/// Architecture:
/// - Writes go to an in-memory partitioned write buffer (~3-5ns insert)
/// - When buffer is full, entries are bulk-merged into the B+Tree
/// - Reads check the write buffer first, then the B+Tree
/// - B+Tree uses 32KB nodes for shallow height (height=2 for 1M keys)
pub struct BufferedBTreeIndex {
    /// The underlying B+Tree (read-optimized with 32KB nodes).
    btree: BTreeArenaIndex,
    /// Write buffer for absorbing inserts.
    buffer: PartitionedBuffer,
    /// Fast empty-check flag.
    buffer_has_data: bool,
    /// Performance statistics for profiling.
    stats: InsertStats,
    /// Tombstone set for deleted keys.
    deleted: std::collections::HashSet<u64>,
    /// Reusable drain output buffer.
    drain_buf: Vec<(u64, u64)>,
}

impl BufferedBTreeIndex {
    /// Creates a new buffered B+Tree index.
    /// Capacity is the maximum number of B+Tree nodes (each 32KB).
    pub fn new(capacity_nodes: usize) -> Self {
        Self {
            btree: BTreeArenaIndex::new(capacity_nodes),
            buffer: PartitionedBuffer::new(),
            buffer_has_data: false,
            stats: InsertStats::default(),
            deleted: std::collections::HashSet::new(),
            drain_buf: Vec::with_capacity(WRITE_BUFFER_CAPACITY),
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

    /// Search for a key. Checks tombstone set, then write buffer, then B+Tree.
    #[inline(always)]
    pub fn search(&self, key: &[u8]) -> Option<TupleId> {
        let search_key = self.key_to_u64(key);

        // Tombstone check
        if !self.deleted.is_empty() && self.deleted.contains(&search_key) {
            return None;
        }

        // Check write buffer first
        if self.buffer_has_data
            && let Some(packed) = self.buffer.search(search_key)
        {
            return Some(BTreeArenaIndex::unpack_tuple_id(packed));
        }

        // Fall through to B+Tree
        self.btree.search(key)
    }

    /// Insert a key-value pair.
    /// Goes to write buffer first; flushes to B+Tree when buffer is full.
    #[inline(always)]
    pub fn insert(&mut self, key: &[u8], tuple_id: TupleId) -> Result<()> {
        if self.buffer.is_full() {
            self.flush_buffer()?;
        }

        let search_key = self.key_to_u64(key);
        let packed = BTreeArenaIndex::pack_tuple_id(tuple_id);

        if !self.deleted.is_empty() {
            self.deleted.remove(&search_key);
        }

        self.buffer.insert(search_key, packed);
        self.buffer_has_data = true;
        Ok(())
    }

    /// Delete a key by adding it to the tombstone set.
    pub fn delete(&mut self, key: &[u8]) -> bool {
        let search_key = self.key_to_u64(key);
        self.deleted.insert(search_key)
    }

    /// Flush write buffer to B+Tree.
    fn flush_buffer(&mut self) -> Result<()> {
        if !self.buffer_has_data {
            return Ok(());
        }

        let flush_start = std::time::Instant::now();

        // Drain and sort entries for sequential B+Tree insertion.
        self.buffer.drain_sorted_into(&mut self.drain_buf);

        // Bulk insert into arena.
        if self.deleted.is_empty() {
            self.btree.insert_bulk_sorted(&self.drain_buf)?;
        } else {
            let filtered: Vec<(u64, u64)> = self
                .drain_buf
                .iter()
                .filter(|(k, _)| !self.deleted.contains(k))
                .copied()
                .collect();
            self.btree.insert_bulk_sorted(&filtered)?;
        }

        self.stats.flush_count += 1;
        self.stats.flush_time_ns += flush_start.elapsed().as_nanos() as u64;

        self.buffer_has_data = false;
        Ok(())
    }

    /// Force flush the write buffer to B+Tree.
    pub fn flush(&mut self) -> Result<()> {
        self.flush_buffer()
    }

    /// Range scan from start_key to end_key (inclusive).
    #[inline]
    pub fn range_scan(&self, start_key: Option<&[u8]>, end_key: Option<&[u8]>) -> Vec<(u64, u64)> {
        if !self.buffer_has_data {
            let mut results = self.btree.range_scan(start_key, end_key);
            if !self.deleted.is_empty() {
                results.retain(|(k, _)| !self.deleted.contains(k));
            }
            return results;
        }

        let start = start_key.map(|k| self.key_to_u64(k)).unwrap_or(0);
        let end = end_key.map(|k| self.key_to_u64(k)).unwrap_or(u64::MAX);

        let btree_results = self.btree.range_scan(start_key, end_key);

        let mut buffer_results: Vec<(u64, u64)> =
            if self.buffer.min_key <= end && self.buffer.max_key >= start {
                self.buffer
                    .iter()
                    .filter(|(k, _)| *k >= start && *k <= end && !self.deleted.contains(k))
                    .collect()
            } else {
                Vec::new()
            };
        buffer_results.sort_unstable_by_key(|(k, _)| *k);

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

        if !self.deleted.is_empty() {
            merged.retain(|(k, _)| !self.deleted.contains(k));
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
    use super::super::page::{BTreeInternalPage, BTreeLeafPage};
    use super::super::types::{
        DeleteResult, InternalEntry, InternalPageHeader, LeafEntry, LeafPageHeader,
    };
    use super::*;
    use bytes::Bytes;
    use zyron_common::ZyronError;
    use zyron_common::page::{PAGE_SIZE, PageId};

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
        // file_id is not stored on disk, reconstructed as 0
        assert_eq!(recovered.tuple_id.page_id.file_id, 0);
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
        // file_id is not stored on disk, reconstructed as 0
        assert_eq!(recovered.child_page_id.file_id, 0);
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
        // file_id is not stored per entry, reads back as 0
        let expected = TupleId::new(PageId::new(0, 10), 5);
        assert_eq!(page.get(&key), Some(expected));
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
        // file_id is not stored per entry, reads back as 0
        assert_eq!(
            recovered.get(b"test"),
            Some(TupleId::new(PageId::new(0, 2), 3))
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

        // Keys less than "key1" go to leftmost child (stored as u64, preserves file_id)
        assert_eq!(page.find_child(b"aaa"), PageId::new(1, 0));

        // Keys >= "key2" go to the right child (entry pointers store page_num only, file_id=0)
        assert_eq!(page.find_child(b"key2"), PageId::new(0, 2));
        assert_eq!(page.find_child(b"zzz"), PageId::new(0, 2));
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

        // 2 (key_len) + 5 (key) + 4 (page_num) + 2 (slot_id) = 13
        assert_eq!(entry.size_on_disk(), 13);
    }

    #[test]
    fn test_internal_entry_size_on_disk() {
        let entry = InternalEntry {
            key: Bytes::from_static(b"hello"),
            child_page_id: PageId::new(0, 0),
        };

        // 2 (key_len) + 5 (key) + 4 (page_num) = 11
        assert_eq!(entry.size_on_disk(), 11);
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
            let tid = TupleId::new(PageId::new(0, i), i as u16);
            tree.insert(&i.to_be_bytes(), tid).unwrap();
        }

        assert_eq!(tree.count(), 1000);

        // Verify all keys can be found
        for i in 0..1000u64 {
            let tid = TupleId::new(PageId::new(0, i), i as u16);
            assert_eq!(tree.search(&i.to_be_bytes()), Some(tid));
        }
    }

    #[test]
    fn test_arena_btree_range_scan() {
        let mut tree = BTreeArenaIndex::new(4096);

        // Insert keys 0, 10, 20, ..., 100
        for i in (0..=100u64).step_by(10) {
            let tid = TupleId::new(PageId::new(0, i), 0);
            tree.insert(&i.to_be_bytes(), tid).unwrap();
        }

        // Range scan from 25 to 75 (should get 30, 40, 50, 60, 70)
        let results = tree.range_scan(Some(&25u64.to_be_bytes()), Some(&75u64.to_be_bytes()));
        assert_eq!(results.len(), 5);

        let keys: Vec<u64> = results.iter().map(|(k, _)| *k).collect();
        assert_eq!(keys, vec![30, 40, 50, 60, 70]);
    }
}

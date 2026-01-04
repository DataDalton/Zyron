//! B+ tree index implementation.
//!
//! This module provides a disk-based B+ tree for indexing tuples.
//! The tree stores keys in sorted order with values (TupleIds) in leaf nodes.
//!
//! Page layout for leaf nodes:
//! ```text
//! +------------------+
//! | Page Header (32) |
//! +------------------+
//! | Leaf Header (16) |
//! +------------------+
//! | Key-Value Pairs  |
//! | (variable size)  |
//! +------------------+
//! ```
//!
//! Page layout for internal nodes:
//! ```text
//! +------------------+
//! | Page Header (32) |
//! +------------------+
//! | Internal Hdr(16) |
//! +------------------+
//! | Keys & Pointers  |
//! | (variable size)  |
//! +------------------+
//! ```

use crate::disk::DiskManager;
use crate::tuple::TupleId;
use bytes::{Bytes, BytesMut};
use std::sync::Arc;
use zyron_buffer::{BufferPool, EvictedPage};
use zyron_common::page::{PageHeader, PageId, PageType, PAGE_SIZE};
use zyron_common::{Result, ZyronError};

/// Maximum key size in bytes.
pub const MAX_KEY_SIZE: usize = 256;

/// Minimum fill factor for B+ tree nodes (50%).
pub const MIN_FILL_FACTOR: f64 = 0.5;

/// Result of a delete operation indicating whether rebalancing may be needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeleteResult {
    /// Key was deleted and node has sufficient entries.
    Ok,
    /// Key was deleted but node is now underfull (below MIN_FILL_FACTOR).
    Underfull,
    /// Key was not found.
    NotFound,
}

/// Header for B+ tree leaf pages.
///
/// Layout (16 bytes):
/// - num_keys: 2 bytes
/// - free_space_offset: 2 bytes
/// - next_leaf: 8 bytes (PageId as u64, for range scans)
/// - reserved: 4 bytes
#[derive(Debug, Clone, Copy)]
pub struct LeafPageHeader {
    /// Number of key-value pairs in this leaf.
    pub num_keys: u16,
    /// Offset to free space (from page start).
    pub free_space_offset: u16,
    /// Page ID of the next leaf (for range scans).
    pub next_leaf: u64,
    /// Reserved for future use.
    pub reserved: u32,
}

impl LeafPageHeader {
    /// Size of the leaf header in bytes.
    pub const SIZE: usize = 16;

    /// Offset of leaf header in page (after PageHeader).
    pub const OFFSET: usize = PageHeader::SIZE;

    /// Creates a new leaf header.
    pub fn new() -> Self {
        Self {
            num_keys: 0,
            free_space_offset: (PageHeader::SIZE + Self::SIZE) as u16,
            next_leaf: u64::MAX,
            reserved: 0,
        }
    }

    /// Serializes to bytes.
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..2].copy_from_slice(&self.num_keys.to_le_bytes());
        buf[2..4].copy_from_slice(&self.free_space_offset.to_le_bytes());
        buf[4..12].copy_from_slice(&self.next_leaf.to_le_bytes());
        buf[12..16].copy_from_slice(&self.reserved.to_le_bytes());
        buf
    }

    /// Deserializes from bytes.
    pub fn from_bytes(buf: &[u8]) -> Self {
        Self {
            num_keys: u16::from_le_bytes([buf[0], buf[1]]),
            free_space_offset: u16::from_le_bytes([buf[2], buf[3]]),
            next_leaf: u64::from_le_bytes([
                buf[4], buf[5], buf[6], buf[7], buf[8], buf[9], buf[10], buf[11],
            ]),
            reserved: u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]),
        }
    }
}

impl Default for LeafPageHeader {
    fn default() -> Self {
        Self::new()
    }
}

/// Header for B+ tree internal pages.
///
/// Layout (16 bytes):
/// - num_keys: 2 bytes
/// - free_space_offset: 2 bytes
/// - level: 2 bytes (0 = just above leaves)
/// - reserved: 10 bytes
#[derive(Debug, Clone, Copy)]
pub struct InternalPageHeader {
    /// Number of keys in this internal node.
    pub num_keys: u16,
    /// Offset to free space (from page start).
    pub free_space_offset: u16,
    /// Level in the tree (0 = just above leaves).
    pub level: u16,
    /// Reserved for future use.
    pub reserved: [u8; 10],
}

impl InternalPageHeader {
    /// Size of the internal header in bytes.
    pub const SIZE: usize = 16;

    /// Offset of internal header in page (after PageHeader).
    pub const OFFSET: usize = PageHeader::SIZE;

    /// Creates a new internal header.
    pub fn new(level: u16) -> Self {
        Self {
            num_keys: 0,
            free_space_offset: (PageHeader::SIZE + Self::SIZE) as u16,
            level,
            reserved: [0; 10],
        }
    }

    /// Serializes to bytes.
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..2].copy_from_slice(&self.num_keys.to_le_bytes());
        buf[2..4].copy_from_slice(&self.free_space_offset.to_le_bytes());
        buf[4..6].copy_from_slice(&self.level.to_le_bytes());
        buf[6..16].copy_from_slice(&self.reserved);
        buf
    }

    /// Deserializes from bytes.
    pub fn from_bytes(buf: &[u8]) -> Self {
        let mut reserved = [0u8; 10];
        reserved.copy_from_slice(&buf[6..16]);
        Self {
            num_keys: u16::from_le_bytes([buf[0], buf[1]]),
            free_space_offset: u16::from_le_bytes([buf[2], buf[3]]),
            level: u16::from_le_bytes([buf[4], buf[5]]),
            reserved,
        }
    }
}

impl Default for InternalPageHeader {
    fn default() -> Self {
        Self::new(0)
    }
}

/// A key-value entry in a leaf page.
///
/// Layout:
/// - key_len: 2 bytes
/// - key: variable
/// - tuple_id: 8 bytes (page_id as u64 + slot_id as u16 packed)
#[derive(Debug, Clone)]
pub struct LeafEntry {
    /// The key bytes.
    pub key: Bytes,
    /// The tuple ID this key points to.
    pub tuple_id: TupleId,
}

impl LeafEntry {
    /// Size of this entry on disk.
    pub fn size_on_disk(&self) -> usize {
        2 + self.key.len() + 10 // key_len + key + tuple_id (file_id:4 + page_num:4 + slot_id:2)
    }

    /// Serializes the entry to bytes.
    pub fn to_bytes(&self) -> Bytes {
        let mut buf = BytesMut::with_capacity(self.size_on_disk());
        buf.extend_from_slice(&(self.key.len() as u16).to_le_bytes());
        buf.extend_from_slice(&self.key);
        buf.extend_from_slice(&self.tuple_id.page_id.file_id.to_le_bytes());
        buf.extend_from_slice(&self.tuple_id.page_id.page_num.to_le_bytes());
        buf.extend_from_slice(&self.tuple_id.slot_id.to_le_bytes());
        buf.freeze()
    }

    /// Deserializes an entry from bytes. Returns (entry, bytes_consumed).
    pub fn from_bytes(buf: &[u8]) -> Option<(Self, usize)> {
        if buf.len() < 12 {
            return None;
        }

        let key_len = u16::from_le_bytes([buf[0], buf[1]]) as usize;
        if buf.len() < 2 + key_len + 10 {
            return None;
        }

        let key = Bytes::copy_from_slice(&buf[2..2 + key_len]);
        let file_id = u32::from_le_bytes([
            buf[2 + key_len],
            buf[3 + key_len],
            buf[4 + key_len],
            buf[5 + key_len],
        ]);
        let page_num = u32::from_le_bytes([
            buf[6 + key_len],
            buf[7 + key_len],
            buf[8 + key_len],
            buf[9 + key_len],
        ]);
        let slot_id = u16::from_le_bytes([buf[10 + key_len], buf[11 + key_len]]);

        let tuple_id = TupleId::new(PageId::new(file_id, page_num), slot_id);
        Some((Self { key, tuple_id }, 2 + key_len + 10))
    }
}

/// A key-pointer entry in an internal page.
///
/// Layout:
/// - key_len: 2 bytes
/// - key: variable
/// - child_page_id: 8 bytes
#[derive(Debug, Clone)]
pub struct InternalEntry {
    /// The key bytes.
    pub key: Bytes,
    /// The child page ID.
    pub child_page_id: PageId,
}

impl InternalEntry {
    /// Size of this entry on disk.
    pub fn size_on_disk(&self) -> usize {
        2 + self.key.len() + 8 // key_len + key + page_id
    }

    /// Serializes the entry to bytes.
    pub fn to_bytes(&self) -> Bytes {
        let mut buf = BytesMut::with_capacity(self.size_on_disk());
        buf.extend_from_slice(&(self.key.len() as u16).to_le_bytes());
        buf.extend_from_slice(&self.key);
        buf.extend_from_slice(&self.child_page_id.as_u64().to_le_bytes());
        buf.freeze()
    }

    /// Deserializes an entry from bytes. Returns (entry, bytes_consumed).
    pub fn from_bytes(buf: &[u8]) -> Option<(Self, usize)> {
        if buf.len() < 10 {
            return None;
        }

        let key_len = u16::from_le_bytes([buf[0], buf[1]]) as usize;
        if buf.len() < 2 + key_len + 8 {
            return None;
        }

        let key = Bytes::copy_from_slice(&buf[2..2 + key_len]);
        let child_page_id = PageId::from_u64(u64::from_le_bytes([
            buf[2 + key_len],
            buf[3 + key_len],
            buf[4 + key_len],
            buf[5 + key_len],
            buf[6 + key_len],
            buf[7 + key_len],
            buf[8 + key_len],
            buf[9 + key_len],
        ]));

        Some((Self { key, child_page_id }, 2 + key_len + 8))
    }
}

/// B+ tree leaf page.
pub struct BTreeLeafPage {
    /// Page data buffer.
    data: Box<[u8; PAGE_SIZE]>,
}

impl BTreeLeafPage {
    /// Data start offset after headers.
    const DATA_START: usize = PageHeader::SIZE + LeafPageHeader::SIZE;

    /// Creates a new empty leaf page.
    pub fn new(page_id: PageId) -> Self {
        let mut data = Box::new([0u8; PAGE_SIZE]);

        // Initialize page header
        let page_header = PageHeader::new(page_id, PageType::BTreeLeaf);
        data[..PageHeader::SIZE].copy_from_slice(&page_header.to_bytes());

        // Initialize leaf header
        let leaf_header = LeafPageHeader::new();
        let offset = LeafPageHeader::OFFSET;
        data[offset..offset + LeafPageHeader::SIZE].copy_from_slice(&leaf_header.to_bytes());

        Self { data }
    }

    /// Creates a leaf page from raw bytes.
    pub fn from_bytes(data: [u8; PAGE_SIZE]) -> Self {
        Self {
            data: Box::new(data),
        }
    }

    /// Returns the raw page data.
    pub fn as_bytes(&self) -> &[u8; PAGE_SIZE] {
        &self.data
    }

    /// Returns the leaf header.
    fn leaf_header(&self) -> LeafPageHeader {
        let offset = LeafPageHeader::OFFSET;
        LeafPageHeader::from_bytes(&self.data[offset..offset + LeafPageHeader::SIZE])
    }

    /// Writes the leaf header.
    fn set_leaf_header(&mut self, header: LeafPageHeader) {
        let offset = LeafPageHeader::OFFSET;
        self.data[offset..offset + LeafPageHeader::SIZE].copy_from_slice(&header.to_bytes());
    }

    /// Returns the number of entries in this leaf.
    pub fn num_entries(&self) -> u16 {
        self.leaf_header().num_keys
    }

    /// Returns the amount of free space available.
    pub fn free_space(&self) -> usize {
        PAGE_SIZE - self.leaf_header().free_space_offset as usize
    }

    /// Returns the next leaf page ID.
    pub fn next_leaf(&self) -> Option<PageId> {
        let next = self.leaf_header().next_leaf;
        if next == u64::MAX {
            None
        } else {
            Some(PageId::from_u64(next))
        }
    }

    /// Sets the next leaf page ID.
    pub fn set_next_leaf(&mut self, page_id: Option<PageId>) {
        let mut header = self.leaf_header();
        header.next_leaf = page_id.map(|p| p.as_u64()).unwrap_or(u64::MAX);
        self.set_leaf_header(header);
    }

    /// Reads all entries from the leaf.
    pub fn entries(&self) -> Vec<LeafEntry> {
        let header = self.leaf_header();
        let mut entries = Vec::with_capacity(header.num_keys as usize);
        let mut offset = Self::DATA_START;

        for _ in 0..header.num_keys {
            if let Some((entry, consumed)) = LeafEntry::from_bytes(&self.data[offset..]) {
                entries.push(entry);
                offset += consumed;
            } else {
                break;
            }
        }

        entries
    }

    /// Binary search for a key. Returns Ok(index) if found, Err(index) for insertion point.
    pub fn search(&self, key: &[u8]) -> std::result::Result<usize, usize> {
        let entries = self.entries();
        entries.binary_search_by(|e| e.key.as_ref().cmp(key))
    }

    /// Inserts a key-value pair into the leaf. Returns error if page is full.
    pub fn insert(&mut self, key: Bytes, tuple_id: TupleId) -> Result<()> {
        let entry = LeafEntry { key, tuple_id };
        let entry_size = entry.size_on_disk();

        if self.free_space() < entry_size {
            return Err(ZyronError::NodeFull);
        }

        // Find insertion point
        let insert_pos = match self.search(&entry.key) {
            Ok(_) => return Err(ZyronError::DuplicateKey),
            Err(pos) => pos,
        };

        // Read existing entries
        let mut entries = self.entries();
        entries.insert(insert_pos, entry);

        // Rewrite all entries
        self.write_entries(&entries)
    }

    /// Writes entries to the page.
    fn write_entries(&mut self, entries: &[LeafEntry]) -> Result<()> {
        let mut header = self.leaf_header();
        let mut offset = Self::DATA_START;

        for entry in entries {
            let bytes = entry.to_bytes();
            if offset + bytes.len() > PAGE_SIZE {
                return Err(ZyronError::NodeFull);
            }
            self.data[offset..offset + bytes.len()].copy_from_slice(&bytes);
            offset += bytes.len();
        }

        header.num_keys = entries.len() as u16;
        header.free_space_offset = offset as u16;
        self.set_leaf_header(header);
        Ok(())
    }

    /// Gets the value for a key.
    pub fn get(&self, key: &[u8]) -> Option<TupleId> {
        Self::get_in_slice(&*self.data, key)
    }

    /// Gets the value for a key directly from raw page data.
    /// Zero-copy version that avoids allocations.
    pub fn get_in_slice(data: &[u8], key: &[u8]) -> Option<TupleId> {
        // Parse header
        let header_offset = LeafPageHeader::OFFSET;
        let num_keys = u16::from_le_bytes([data[header_offset], data[header_offset + 1]]) as usize;

        // Binary search through entries
        let mut offset = Self::DATA_START;
        for _ in 0..num_keys {
            // Parse entry: key_len (2) + key (var) + page_id (8) + slot_id (2)
            let key_len = u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
            let entry_key = &data[offset + 2..offset + 2 + key_len];

            match key.cmp(entry_key) {
                std::cmp::Ordering::Equal => {
                    // Found it - parse tuple_id
                    let tuple_offset = offset + 2 + key_len;
                    let page_id = PageId::from_u64(u64::from_le_bytes([
                        data[tuple_offset], data[tuple_offset + 1],
                        data[tuple_offset + 2], data[tuple_offset + 3],
                        data[tuple_offset + 4], data[tuple_offset + 5],
                        data[tuple_offset + 6], data[tuple_offset + 7],
                    ]));
                    let slot_id = u16::from_le_bytes([
                        data[tuple_offset + 8], data[tuple_offset + 9],
                    ]);
                    return Some(TupleId::new(page_id, slot_id));
                }
                std::cmp::Ordering::Less => {
                    // Key would be before this entry, so not found (entries are sorted ascending)
                    return None;
                }
                std::cmp::Ordering::Greater => {
                    // Key is after this entry, continue searching
                    offset += 2 + key_len + 10; // key_len + key + page_id(8) + slot_id(2)
                }
            }
        }
        None
    }

    /// Inserts a key-value pair directly into raw page data.
    /// Returns Ok(()) on success, Err(NodeFull) if page is full, Err(DuplicateKey) if key exists.
    pub fn insert_in_slice(data: &mut [u8], key: &[u8], tuple_id: TupleId) -> Result<()> {
        // Parse header
        let header_offset = LeafPageHeader::OFFSET;
        let num_keys = u16::from_le_bytes([data[header_offset], data[header_offset + 1]]) as usize;
        let raw_free_offset = u16::from_le_bytes([data[header_offset + 2], data[header_offset + 3]]) as usize;

        // Handle uninitialized pages (free_space_offset == 0 means page was never written)
        let free_space_offset = if raw_free_offset < Self::DATA_START {
            Self::DATA_START
        } else {
            raw_free_offset
        };

        // Entry size: key_len(2) + key + page_id(8) + slot_id(2)
        let entry_size = 2 + key.len() + 10;
        let free_space = PAGE_SIZE - free_space_offset;

        if free_space < entry_size {
            return Err(ZyronError::NodeFull);
        }

        // Find insertion point and check for duplicates
        let mut insert_offset = Self::DATA_START;
        let mut offset = Self::DATA_START;

        for _ in 0..num_keys {
            let key_len = u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
            let entry_key = &data[offset + 2..offset + 2 + key_len];
            let entry_total = 2 + key_len + 10;

            match key.cmp(entry_key) {
                std::cmp::Ordering::Equal => return Err(ZyronError::DuplicateKey),
                std::cmp::Ordering::Less => {
                    insert_offset = offset;
                    break;
                }
                std::cmp::Ordering::Greater => {
                    offset += entry_total;
                    insert_offset = offset;
                }
            }
        }

        // Shift existing entries to make room
        let bytes_to_shift = free_space_offset - insert_offset;
        if bytes_to_shift > 0 {
            data.copy_within(insert_offset..free_space_offset, insert_offset + entry_size);
        }

        // Write the new entry
        let mut write_offset = insert_offset;
        data[write_offset..write_offset + 2].copy_from_slice(&(key.len() as u16).to_le_bytes());
        write_offset += 2;
        data[write_offset..write_offset + key.len()].copy_from_slice(key);
        write_offset += key.len();
        data[write_offset..write_offset + 8].copy_from_slice(&tuple_id.page_id.as_u64().to_le_bytes());
        write_offset += 8;
        data[write_offset..write_offset + 2].copy_from_slice(&tuple_id.slot_id.to_le_bytes());

        // Update header
        let new_num_keys = (num_keys + 1) as u16;
        let new_free_offset = (free_space_offset + entry_size) as u16;
        data[header_offset..header_offset + 2].copy_from_slice(&new_num_keys.to_le_bytes());
        data[header_offset + 2..header_offset + 4].copy_from_slice(&new_free_offset.to_le_bytes());

        Ok(())
    }

    /// Deletes a key from the leaf. Returns DeleteResult indicating outcome.
    pub fn delete(&mut self, key: &[u8]) -> DeleteResult {
        match self.search(key) {
            Ok(idx) => {
                let mut entries = self.entries();
                entries.remove(idx);
                if self.write_entries(&entries).is_ok() {
                    if self.is_underfull() {
                        DeleteResult::Underfull
                    } else {
                        DeleteResult::Ok
                    }
                } else {
                    DeleteResult::NotFound
                }
            }
            Err(_) => DeleteResult::NotFound,
        }
    }

    /// Returns true if this leaf is underfull (below MIN_FILL_FACTOR capacity).
    ///
    /// An underfull node should trigger rebalancing (borrowing from siblings
    /// or merging with a sibling) to maintain B+ tree balance invariants.
    pub fn is_underfull(&self) -> bool {
        let used_space = self.leaf_header().free_space_offset as usize - Self::DATA_START;
        let total_data_space = PAGE_SIZE - Self::DATA_START;
        let fill_ratio = used_space as f64 / total_data_space as f64;
        fill_ratio < MIN_FILL_FACTOR && self.num_entries() > 0
    }

    /// Returns the minimum number of bytes that should be used to avoid underflow.
    pub fn min_used_space(&self) -> usize {
        let total_data_space = PAGE_SIZE - Self::DATA_START;
        (total_data_space as f64 * MIN_FILL_FACTOR) as usize
    }

    /// Borrows entries from a right sibling to fix underflow.
    ///
    /// Returns the new separator key that should replace the old separator
    /// in the parent node, or None if borrowing is not possible.
    pub fn borrow_from_right(&mut self, right_sibling: &mut BTreeLeafPage) -> Option<Bytes> {
        if right_sibling.num_entries() <= 1 {
            return None; // Can't borrow if sibling would become empty
        }

        let mut right_entries = right_sibling.entries();
        let borrowed = right_entries.remove(0);

        let mut my_entries = self.entries();
        my_entries.push(borrowed);

        // Write updated entries
        if self.write_entries(&my_entries).is_err() {
            return None;
        }
        if right_sibling.write_entries(&right_entries).is_err() {
            return None;
        }

        // New separator is the first key of the right sibling after borrowing
        right_entries.first().map(|e| e.key.clone())
    }

    /// Borrows entries from a left sibling to fix underflow.
    ///
    /// Returns the new separator key that should replace the old separator
    /// in the parent node, or None if borrowing is not possible.
    pub fn borrow_from_left(&mut self, left_sibling: &mut BTreeLeafPage) -> Option<Bytes> {
        if left_sibling.num_entries() <= 1 {
            return None; // Can't borrow if sibling would become empty
        }

        let mut left_entries = left_sibling.entries();
        let borrowed = left_entries.pop()?;

        let mut my_entries = self.entries();
        my_entries.insert(0, borrowed);

        // Write updated entries
        if self.write_entries(&my_entries).is_err() {
            return None;
        }
        if left_sibling.write_entries(&left_entries).is_err() {
            return None;
        }

        // New separator is the first key of this node after borrowing
        my_entries.first().map(|e| e.key.clone())
    }

    /// Merges this leaf with its right sibling.
    ///
    /// All entries from right_sibling are moved into this leaf.
    /// The right sibling becomes empty and should be deallocated.
    /// Returns true if merge succeeded.
    pub fn merge_with_right(&mut self, right_sibling: &mut BTreeLeafPage) -> bool {
        let mut my_entries = self.entries();
        let right_entries = right_sibling.entries();

        my_entries.extend(right_entries);

        // Update next_leaf pointer to skip the merged sibling
        let new_next = right_sibling.next_leaf();
        self.set_next_leaf(new_next);

        self.write_entries(&my_entries).is_ok()
    }

    /// Returns true if this leaf can fit another entry of the given size.
    pub fn can_fit(&self, entry_size: usize) -> bool {
        self.free_space() >= entry_size
    }

    /// Splits this leaf into two. Returns (split_key, new_right_page).
    pub fn split(&mut self, new_page_id: PageId) -> (Bytes, BTreeLeafPage) {
        let entries = self.entries();
        let mid = entries.len() / 2;

        let left_entries: Vec<_> = entries[..mid].to_vec();
        let right_entries: Vec<_> = entries[mid..].to_vec();

        // The split key is the first key of the right page
        let split_key = right_entries[0].key.clone();

        // Rewrite left page
        let _ = self.write_entries(&left_entries);

        // Create right page
        let mut right_page = BTreeLeafPage::new(new_page_id);
        let _ = right_page.write_entries(&right_entries);

        // Link pages
        let old_next = self.next_leaf();
        self.set_next_leaf(Some(new_page_id));
        right_page.set_next_leaf(old_next);

        (split_key, right_page)
    }
}

/// B+ tree internal page.
pub struct BTreeInternalPage {
    /// Page data buffer.
    data: Box<[u8; PAGE_SIZE]>,
}

impl BTreeInternalPage {
    /// Data start offset after headers.
    const DATA_START: usize = PageHeader::SIZE + InternalPageHeader::SIZE;

    /// Size of the leftmost child pointer.
    const LEFTMOST_PTR_SIZE: usize = 8;

    /// Creates a new empty internal page.
    pub fn new(page_id: PageId, level: u16) -> Self {
        let mut data = Box::new([0u8; PAGE_SIZE]);

        // Initialize page header
        let page_header = PageHeader::new(page_id, PageType::BTreeInternal);
        data[..PageHeader::SIZE].copy_from_slice(&page_header.to_bytes());

        // Initialize internal header
        let internal_header = InternalPageHeader::new(level);
        let offset = InternalPageHeader::OFFSET;
        data[offset..offset + InternalPageHeader::SIZE]
            .copy_from_slice(&internal_header.to_bytes());

        Self { data }
    }

    /// Creates an internal page from raw bytes.
    pub fn from_bytes(data: [u8; PAGE_SIZE]) -> Self {
        Self {
            data: Box::new(data),
        }
    }

    /// Returns the raw page data.
    pub fn as_bytes(&self) -> &[u8; PAGE_SIZE] {
        &self.data
    }

    /// Returns the internal header.
    fn internal_header(&self) -> InternalPageHeader {
        let offset = InternalPageHeader::OFFSET;
        InternalPageHeader::from_bytes(&self.data[offset..offset + InternalPageHeader::SIZE])
    }

    /// Writes the internal header.
    fn set_internal_header(&mut self, header: InternalPageHeader) {
        let offset = InternalPageHeader::OFFSET;
        self.data[offset..offset + InternalPageHeader::SIZE]
            .copy_from_slice(&header.to_bytes());
    }

    /// Returns the number of keys in this internal node.
    pub fn num_keys(&self) -> u16 {
        self.internal_header().num_keys
    }

    /// Returns the level of this internal node.
    pub fn level(&self) -> u16 {
        self.internal_header().level
    }

    /// Returns the amount of free space available.
    pub fn free_space(&self) -> usize {
        PAGE_SIZE - self.internal_header().free_space_offset as usize
    }

    /// Gets the leftmost child pointer.
    pub fn leftmost_child(&self) -> PageId {
        let offset = Self::DATA_START;
        let bytes = &self.data[offset..offset + 8];
        PageId::from_u64(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    /// Sets the leftmost child pointer.
    pub fn set_leftmost_child(&mut self, page_id: PageId) {
        let offset = Self::DATA_START;
        self.data[offset..offset + 8].copy_from_slice(&page_id.as_u64().to_le_bytes());

        // Update header if this is the first entry
        let mut header = self.internal_header();
        if header.free_space_offset == Self::DATA_START as u16 {
            header.free_space_offset = (Self::DATA_START + Self::LEFTMOST_PTR_SIZE) as u16;
            self.set_internal_header(header);
        }
    }

    /// Reads all entries from the internal node.
    pub fn entries(&self) -> Vec<InternalEntry> {
        let header = self.internal_header();
        let mut entries = Vec::with_capacity(header.num_keys as usize);
        let mut offset = Self::DATA_START + Self::LEFTMOST_PTR_SIZE;

        for _ in 0..header.num_keys {
            if let Some((entry, consumed)) = InternalEntry::from_bytes(&self.data[offset..]) {
                entries.push(entry);
                offset += consumed;
            } else {
                break;
            }
        }

        entries
    }

    /// Finds the child page for a given key.
    pub fn find_child(&self, key: &[u8]) -> PageId {
        Self::find_child_in_slice(&*self.data, key)
    }

    /// Finds the child page for a given key directly from raw page data.
    /// Zero-copy version that avoids Box allocation and page struct creation.
    pub fn find_child_in_slice(data: &[u8], key: &[u8]) -> PageId {
        // Parse header to get num_keys
        let header_offset = InternalPageHeader::OFFSET;
        let num_keys = u16::from_le_bytes([data[header_offset], data[header_offset + 1]]) as usize;

        // Leftmost child pointer
        let leftmost_offset = Self::DATA_START;
        let leftmost = PageId::from_u64(u64::from_le_bytes([
            data[leftmost_offset],
            data[leftmost_offset + 1],
            data[leftmost_offset + 2],
            data[leftmost_offset + 3],
            data[leftmost_offset + 4],
            data[leftmost_offset + 5],
            data[leftmost_offset + 6],
            data[leftmost_offset + 7],
        ]));

        if num_keys == 0 {
            return leftmost;
        }

        // Scan through entries to find correct child
        let mut offset = Self::DATA_START + Self::LEFTMOST_PTR_SIZE;
        let mut prev_child = leftmost;

        for _ in 0..num_keys {
            // Parse entry: key_len (2) + key (var) + child_page_id (8)
            let key_len = u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
            let entry_key = &data[offset + 2..offset + 2 + key_len];
            let child_offset = offset + 2 + key_len;
            let child = PageId::from_u64(u64::from_le_bytes([
                data[child_offset],
                data[child_offset + 1],
                data[child_offset + 2],
                data[child_offset + 3],
                data[child_offset + 4],
                data[child_offset + 5],
                data[child_offset + 6],
                data[child_offset + 7],
            ]));

            if key < entry_key {
                return prev_child;
            }
            prev_child = child;
            offset = child_offset + 8;
        }

        prev_child
    }

    /// Inserts a key and right child pointer.
    pub fn insert(&mut self, key: Bytes, right_child: PageId) -> Result<()> {
        let entry = InternalEntry {
            key,
            child_page_id: right_child,
        };
        let entry_size = entry.size_on_disk();

        if self.free_space() < entry_size {
            return Err(ZyronError::NodeFull);
        }

        let mut entries = self.entries();

        // Find insertion point
        let insert_pos = entries
            .iter()
            .position(|e| entry.key.as_ref() < e.key.as_ref())
            .unwrap_or(entries.len());

        entries.insert(insert_pos, entry);

        // Rewrite all entries
        self.write_entries(&entries)
    }

    /// Writes entries to the page.
    fn write_entries(&mut self, entries: &[InternalEntry]) -> Result<()> {
        let mut header = self.internal_header();
        let mut offset = Self::DATA_START + Self::LEFTMOST_PTR_SIZE;

        for entry in entries {
            let bytes = entry.to_bytes();
            if offset + bytes.len() > PAGE_SIZE {
                return Err(ZyronError::NodeFull);
            }
            self.data[offset..offset + bytes.len()].copy_from_slice(&bytes);
            offset += bytes.len();
        }

        header.num_keys = entries.len() as u16;
        header.free_space_offset = offset as u16;
        self.set_internal_header(header);
        Ok(())
    }

    /// Splits this internal node. Returns (promoted_key, new_right_page).
    pub fn split(&mut self, new_page_id: PageId) -> (Bytes, BTreeInternalPage) {
        let entries = self.entries();
        let mid = entries.len() / 2;

        let left_entries: Vec<_> = entries[..mid].to_vec();
        let promoted_key = entries[mid].key.clone();
        let right_first_child = entries[mid].child_page_id;
        let right_entries: Vec<_> = entries[mid + 1..].to_vec();

        // Rewrite left page
        let _ = self.write_entries(&left_entries);

        // Create right page
        let mut right_page = BTreeInternalPage::new(new_page_id, self.level());
        right_page.set_leftmost_child(right_first_child);
        let _ = right_page.write_entries(&right_entries);

        (promoted_key, right_page)
    }

    /// Returns true if this node can fit another entry of the given size.
    pub fn can_fit(&self, entry_size: usize) -> bool {
        self.free_space() >= entry_size
    }

    /// Returns true if this internal node is underfull (below MIN_FILL_FACTOR capacity).
    ///
    /// An underfull node should trigger rebalancing (borrowing from siblings
    /// or merging with a sibling) to maintain B+ tree balance invariants.
    pub fn is_underfull(&self) -> bool {
        let used_space = self.internal_header().free_space_offset as usize
            - Self::DATA_START
            - Self::LEFTMOST_PTR_SIZE;
        let total_data_space = PAGE_SIZE - Self::DATA_START - Self::LEFTMOST_PTR_SIZE;
        let fill_ratio = used_space as f64 / total_data_space as f64;
        fill_ratio < MIN_FILL_FACTOR && self.num_keys() > 0
    }

    /// Returns the minimum number of bytes that should be used to avoid underflow.
    pub fn min_used_space(&self) -> usize {
        let total_data_space = PAGE_SIZE - Self::DATA_START - Self::LEFTMOST_PTR_SIZE;
        (total_data_space as f64 * MIN_FILL_FACTOR) as usize
    }

    /// Deletes a key from the internal node. Returns DeleteResult indicating outcome.
    pub fn delete(&mut self, key: &[u8]) -> DeleteResult {
        let entries = self.entries();
        let pos = entries.iter().position(|e| e.key.as_ref() == key);

        match pos {
            Some(idx) => {
                let mut entries = entries;
                entries.remove(idx);
                if self.write_entries(&entries).is_ok() {
                    if self.is_underfull() {
                        DeleteResult::Underfull
                    } else {
                        DeleteResult::Ok
                    }
                } else {
                    DeleteResult::NotFound
                }
            }
            None => DeleteResult::NotFound,
        }
    }

    /// Borrows an entry from a right sibling to fix underflow.
    ///
    /// The separator_key is the key in the parent that separates this node from the sibling.
    /// Returns the new separator key that should replace the old one in the parent,
    /// or None if borrowing is not possible.
    pub fn borrow_from_right(
        &mut self,
        right_sibling: &mut BTreeInternalPage,
        separator_key: Bytes,
    ) -> Option<Bytes> {
        if right_sibling.num_keys() <= 1 {
            return None; // Can't borrow if sibling would become too empty
        }

        let mut right_entries = right_sibling.entries();
        let borrowed = right_entries.remove(0);

        // The separator comes down to become a key in this node
        let new_entry = InternalEntry {
            key: separator_key,
            child_page_id: right_sibling.leftmost_child(),
        };

        let mut my_entries = self.entries();
        my_entries.push(new_entry);

        // Update right sibling's leftmost child to the borrowed entry's child
        right_sibling.set_leftmost_child(borrowed.child_page_id);

        // Write updated entries
        if self.write_entries(&my_entries).is_err() {
            return None;
        }
        if right_sibling.write_entries(&right_entries).is_err() {
            return None;
        }

        // The borrowed key becomes the new separator in the parent
        Some(borrowed.key)
    }

    /// Borrows an entry from a left sibling to fix underflow.
    ///
    /// The separator_key is the key in the parent that separates this node from the sibling.
    /// Returns the new separator key that should replace the old one in the parent,
    /// or None if borrowing is not possible.
    pub fn borrow_from_left(
        &mut self,
        left_sibling: &mut BTreeInternalPage,
        separator_key: Bytes,
    ) -> Option<Bytes> {
        if left_sibling.num_keys() <= 1 {
            return None; // Can't borrow if sibling would become too empty
        }

        let mut left_entries = left_sibling.entries();
        let borrowed = left_entries.pop()?;

        // The separator comes down to become a key in this node
        let new_entry = InternalEntry {
            key: separator_key,
            child_page_id: self.leftmost_child(),
        };

        let mut my_entries = self.entries();
        my_entries.insert(0, new_entry);

        // Update this node's leftmost child to the borrowed entry's child
        self.set_leftmost_child(borrowed.child_page_id);

        // Write updated entries
        if self.write_entries(&my_entries).is_err() {
            return None;
        }
        if left_sibling.write_entries(&left_entries).is_err() {
            return None;
        }

        // The borrowed key becomes the new separator in the parent
        Some(borrowed.key)
    }

    /// Merges this internal node with its right sibling.
    ///
    /// The separator_key is the key from the parent that separates the two nodes.
    /// All entries from right_sibling are moved into this node.
    /// Returns true if merge succeeded.
    pub fn merge_with_right(
        &mut self,
        right_sibling: &BTreeInternalPage,
        separator_key: Bytes,
    ) -> bool {
        let mut my_entries = self.entries();

        // The separator key comes down with the right sibling's leftmost child
        let separator_entry = InternalEntry {
            key: separator_key,
            child_page_id: right_sibling.leftmost_child(),
        };
        my_entries.push(separator_entry);

        // Add all entries from the right sibling
        my_entries.extend(right_sibling.entries());

        self.write_entries(&my_entries).is_ok()
    }
}

/// B+ tree index.
pub struct BTree {
    /// Root page ID.
    root_page_id: PageId,
    /// File ID for this index.
    file_id: u32,
    /// Tree height (1 = just root as leaf).
    height: u32,
}

impl BTree {
    /// Creates a new B+ tree with an empty root leaf.
    pub fn new(root_page_id: PageId, file_id: u32) -> Self {
        Self {
            root_page_id,
            file_id,
            height: 1,
        }
    }

    /// Returns the root page ID.
    pub fn root_page_id(&self) -> PageId {
        self.root_page_id
    }

    /// Returns the file ID.
    pub fn file_id(&self) -> u32 {
        self.file_id
    }

    /// Returns the tree height.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Sets the root page ID (used after root split).
    pub fn set_root_page_id(&mut self, page_id: PageId) {
        self.root_page_id = page_id;
    }

    /// Increments the tree height (used after root split).
    pub fn increment_height(&mut self) {
        self.height += 1;
    }
}

/// B+ tree index with disk I/O operations.
///
/// Provides tree-level operations (search, insert, delete, range_scan)
/// that coordinate page-level operations with disk I/O through BufferPool.
/// Pages are cached in the buffer pool for fast repeated access.
pub struct BTreeIndex {
    /// Tree metadata.
    tree: BTree,
    /// Disk manager for page I/O.
    disk: Arc<DiskManager>,
    /// Buffer pool for page caching.
    pool: Arc<BufferPool>,
}

impl BTreeIndex {
    /// Creates a new B+ tree index.
    ///
    /// Creates an empty root leaf page and caches it in the buffer pool.
    pub async fn create(
        disk: Arc<DiskManager>,
        pool: Arc<BufferPool>,
        file_id: u32,
    ) -> Result<Self> {
        // Allocate root page
        let root_page_id = disk.allocate_page(file_id).await?;

        // Create empty leaf as root and add to buffer pool
        let root_page = BTreeLeafPage::new(root_page_id);
        let (_, evicted) = pool.load_page(root_page_id, root_page.as_bytes())?;
        pool.unpin_page(root_page_id, true); // Mark dirty for eventual disk write

        // Flush any evicted dirty page
        if let Some(evicted) = evicted {
            disk.write_page(evicted.page_id, &*evicted.data).await?;
        }

        let tree = BTree::new(root_page_id, file_id);
        Ok(Self { tree, disk, pool })
    }

    /// Opens an existing B+ tree index.
    ///
    /// Reads the root page to verify the index exists.
    pub async fn open(
        disk: Arc<DiskManager>,
        pool: Arc<BufferPool>,
        file_id: u32,
        root_page_id: PageId,
        height: u32,
    ) -> Result<Self> {
        // Verify root page exists by reading it
        disk.read_page(root_page_id).await?;

        let mut tree = BTree::new(root_page_id, file_id);
        for _ in 1..height {
            tree.increment_height();
        }

        Ok(Self { tree, disk, pool })
    }

    /// Returns the root page ID.
    pub fn root_page_id(&self) -> PageId {
        self.tree.root_page_id()
    }

    /// Returns the file ID.
    pub fn file_id(&self) -> u32 {
        self.tree.file_id()
    }

    /// Returns the tree height.
    pub fn height(&self) -> u32 {
        self.tree.height()
    }

    // =========================================================================
    // Cached Page Access Helpers
    // =========================================================================

    /// Flushes an evicted dirty page to disk.
    async fn flush_evicted(&self, evicted: EvictedPage) -> Result<()> {
        self.disk.write_page(evicted.page_id, &*evicted.data).await
    }

    /// Reads a page, checking buffer pool first, loading from disk on miss.
    async fn read_cached_page(&self, page_id: PageId) -> Result<[u8; PAGE_SIZE]> {
        // Check buffer pool first
        if let Some(frame) = self.pool.fetch_page(page_id) {
            let data = frame.read_data();
            let mut result = [0u8; PAGE_SIZE];
            result.copy_from_slice(&**data);
            self.pool.unpin_page(page_id, false);
            return Ok(result);
        }

        // Cache miss - load from disk
        let data = self.disk.read_page(page_id).await?;
        let (_, evicted) = self.pool.load_page(page_id, &data)?;

        // Flush evicted dirty page if any
        if let Some(evicted) = evicted {
            self.flush_evicted(evicted).await?;
        }

        self.pool.unpin_page(page_id, false);
        Ok(data)
    }

    /// Writes a page to buffer pool only (marks dirty). Disk write deferred until eviction.
    async fn write_cached_page(&self, page_id: PageId, data: &[u8; PAGE_SIZE]) -> Result<()> {
        // Write to buffer pool only - disk write happens on eviction or flush
        if let Some(frame) = self.pool.fetch_page(page_id) {
            frame.copy_from(data);
            frame.set_dirty(true);
            self.pool.unpin_page(page_id, true);
        } else {
            // Page not in pool - load it and mark dirty
            let (_, evicted) = self.pool.load_page(page_id, data)?;

            // Flush evicted dirty page if any
            if let Some(evicted) = evicted {
                self.flush_evicted(evicted).await?;
            }

            self.pool.unpin_page(page_id, true);
        }

        Ok(())
    }

    /// Allocates a new page via disk manager and adds to buffer pool.
    async fn allocate_cached_page(&self) -> Result<PageId> {
        let page_id = self.disk.allocate_page(self.tree.file_id()).await?;
        // New pages will be added to pool when first written
        Ok(page_id)
    }

    /// Flushes all dirty pages belonging to this tree's file to disk.
    pub async fn flush(&self) -> Result<()> {
        let file_id = self.tree.file_id();
        let disk = self.disk.clone();

        self.pool.flush_all(|page_id, data| {
            if page_id.file_id == file_id {
                // Synchronous write - flush_all expects sync callback
                // For async, we'd need a different approach
                let data_copy: [u8; PAGE_SIZE] = data.try_into().map_err(|_| {
                    ZyronError::IoError("invalid page size".to_string())
                })?;
                // Use blocking for now - proper async flush would need redesign
                tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async {
                        disk.write_page(page_id, &data_copy).await
                    })
                })?;
            }
            Ok(())
        })?;

        Ok(())
    }

    /// Searches for a key in the B+ tree.
    ///
    /// Returns the TupleId if found, None otherwise.
    pub async fn search(&self, key: &[u8]) -> Result<Option<TupleId>> {
        let leaf_page_id = self.find_leaf(key).await?;

        // Try buffer pool first (zero-copy path)
        if let Some(guard) = self.pool.read_page(leaf_page_id) {
            let data = guard.data();
            return Ok(BTreeLeafPage::get_in_slice(&**data, key));
        }

        // Cache miss - load from disk
        let page_data = self.disk.read_page(leaf_page_id).await?;
        let (frame, evicted) = self.pool.load_page(leaf_page_id, &page_data)?;
        if let Some(evicted) = evicted {
            self.flush_evicted(evicted).await?;
        }
        let data = frame.read_data();
        let result = BTreeLeafPage::get_in_slice(&**data, key);
        drop(data);
        self.pool.unpin_page(leaf_page_id, false);
        Ok(result)
    }

    /// Finds the leaf page that should contain the given key.
    /// Uses zero-copy access to buffer pool frames for internal node traversal.
    async fn find_leaf(&self, key: &[u8]) -> Result<PageId> {
        let mut current_page_id = self.tree.root_page_id();

        // If tree height is 1, root is a leaf
        if self.tree.height() == 1 {
            return Ok(current_page_id);
        }

        // Traverse internal nodes using zero-copy access
        for _ in 0..(self.tree.height() - 1) {
            let next_child = if let Some(guard) = self.pool.read_page(current_page_id) {
                let data = guard.data();
                BTreeInternalPage::find_child_in_slice(&**data, key)
            } else {
                // Cache miss - load from disk
                let page_data = self.disk.read_page(current_page_id).await?;
                let (frame, evicted) = self.pool.load_page(current_page_id, &page_data)?;
                if let Some(evicted) = evicted {
                    self.flush_evicted(evicted).await?;
                }
                // Use frame directly (already pinned by load_page)
                let data = frame.read_data();
                let result = BTreeInternalPage::find_child_in_slice(&**data, key);
                drop(data);
                self.pool.unpin_page(current_page_id, false);
                result
            };
            current_page_id = next_child;
        }

        Ok(current_page_id)
    }

    /// Inserts a key-value pair into the B+ tree.
    ///
    /// Uses in-place operations on buffer pool frames to avoid data copying.
    /// Handles page splits as needed.
    pub async fn insert(&mut self, key: Bytes, tuple_id: TupleId) -> Result<()> {
        if key.len() > MAX_KEY_SIZE {
            return Err(ZyronError::KeyTooLarge {
                size: key.len(),
                max: MAX_KEY_SIZE,
            });
        }

        // Find the path from root to the target leaf
        let path = self.find_path(&key).await?;

        // Get the leaf page ID
        let leaf_page_id = *path.last().ok_or_else(|| {
            ZyronError::BTreeCorrupted("empty path".to_string())
        })?;

        // Try in-place insert via buffer pool
        let insert_result = if let Some(guard) = self.pool.write_page(leaf_page_id) {
            // Cache hit - insert directly into buffer pool frame
            let mut data = guard.data_mut();
            let result = BTreeLeafPage::insert_in_slice(&mut **data, key.as_ref(), tuple_id);
            if result.is_ok() {
                guard.set_dirty();
            }
            result
        } else {
            // Cache miss - load page then insert in place
            let page_data = self.disk.read_page(leaf_page_id).await?;
            let (frame, evicted) = self.pool.load_page(leaf_page_id, &page_data)?;
            if let Some(evicted) = evicted {
                self.flush_evicted(evicted).await?;
            }
            let mut data = frame.write_data();
            let result = BTreeLeafPage::insert_in_slice(&mut **data, key.as_ref(), tuple_id);
            let is_ok = result.is_ok();
            drop(data);
            self.pool.unpin_page(leaf_page_id, is_ok);
            result
        };

        match insert_result {
            Ok(()) => Ok(()),
            Err(ZyronError::NodeFull) => {
                // Need to split - fall back to copying path for split handling
                self.insert_with_split(key, tuple_id, path).await
            }
            Err(e) => Err(e),
        }
    }

    /// Finds the path from root to the leaf containing the key.
    /// Uses zero-copy access to buffer pool frames for internal node traversal.
    async fn find_path(&self, key: &[u8]) -> Result<Vec<PageId>> {
        let mut path = Vec::with_capacity(self.tree.height() as usize);
        let mut current_page_id = self.tree.root_page_id();

        path.push(current_page_id);

        // If tree height is 1, root is a leaf
        if self.tree.height() == 1 {
            return Ok(path);
        }

        // Traverse internal nodes using zero-copy access
        for _ in 0..(self.tree.height() - 1) {
            // Try buffer pool first (zero-copy path)
            let next_child = if let Some(guard) = self.pool.read_page(current_page_id) {
                let data = guard.data();
                BTreeInternalPage::find_child_in_slice(&**data, key)
            } else {
                // Cache miss - load from disk
                let page_data = self.disk.read_page(current_page_id).await?;
                let (frame, evicted) = self.pool.load_page(current_page_id, &page_data)?;
                if let Some(evicted) = evicted {
                    self.flush_evicted(evicted).await?;
                }
                // Use frame directly (already pinned by load_page)
                let data = frame.read_data();
                let result = BTreeInternalPage::find_child_in_slice(&**data, key);
                drop(data);
                self.pool.unpin_page(current_page_id, false);
                result
            };

            current_page_id = next_child;
            path.push(current_page_id);
        }

        Ok(path)
    }

    /// Inserts with split handling.
    async fn insert_with_split(
        &mut self,
        key: Bytes,
        tuple_id: TupleId,
        path: Vec<PageId>,
    ) -> Result<()> {
        let leaf_page_id = *path.last().ok_or_else(|| {
            ZyronError::BTreeCorrupted("empty path".to_string())
        })?;

        // Read and split the leaf
        let page_data = self.read_cached_page(leaf_page_id).await?;
        let mut leaf = BTreeLeafPage::from_bytes(page_data);

        // Allocate new page for right sibling
        let new_page_id = self.allocate_cached_page().await?;
        let (split_key, mut right_leaf) = leaf.split(new_page_id);

        // Insert the new key into appropriate leaf
        if key.as_ref() < split_key.as_ref() {
            leaf.insert(key, tuple_id)?;
        } else {
            right_leaf.insert(key, tuple_id)?;
        }

        // Write both leaves
        self.write_cached_page(leaf_page_id, leaf.as_bytes()).await?;
        self.write_cached_page(new_page_id, right_leaf.as_bytes()).await?;

        // Propagate the split up the tree
        // When path.len() is 1, the root is a leaf and we need to create a new root
        if path.len() < 2 {
            self.create_new_root(split_key, new_page_id).await
        } else {
            self.insert_into_parent(split_key, new_page_id, &path, path.len() - 2).await
        }
    }

    /// Inserts a separator key and child pointer into parent nodes.
    /// Uses iteration instead of recursion to avoid async boxing.
    async fn insert_into_parent(
        &mut self,
        key: Bytes,
        right_child: PageId,
        path: &[PageId],
        start_parent_idx: usize,
    ) -> Result<()> {
        let mut current_key = key;
        let mut current_child = right_child;
        let mut parent_idx = start_parent_idx;

        loop {
            // If we've reached above the root, create a new root
            if parent_idx >= path.len() {
                return self.create_new_root(current_key, current_child).await;
            }

            let parent_page_id = path[parent_idx];
            let page_data = self.read_cached_page(parent_page_id).await?;
            let mut parent = BTreeInternalPage::from_bytes(page_data);

            // Try to insert into parent
            match parent.insert(current_key.clone(), current_child) {
                Ok(()) => {
                    self.write_cached_page(parent_page_id, parent.as_bytes()).await?;
                    return Ok(());
                }
                Err(ZyronError::NodeFull) => {
                    // Split the internal node
                    let new_page_id = self.allocate_cached_page().await?;
                    let (promoted_key, mut right_internal) = parent.split(new_page_id);

                    // Insert into appropriate side
                    if current_key.as_ref() < promoted_key.as_ref() {
                        parent.insert(current_key, current_child)?;
                    } else {
                        right_internal.insert(current_key, current_child)?;
                    }

                    // Write both pages - split creates new right page
                    self.write_cached_page(new_page_id, right_internal.as_bytes()).await?;
                    self.write_cached_page(parent_page_id, parent.as_bytes()).await?;

                    // Continue propagating up
                    if parent_idx == 0 {
                        return self.create_new_root(promoted_key, new_page_id).await;
                    }

                    // Set up for next iteration
                    current_key = promoted_key;
                    current_child = new_page_id;
                    parent_idx -= 1;
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Creates a new root when the current root splits.
    async fn create_new_root(&mut self, key: Bytes, right_child: PageId) -> Result<()> {
        let new_root_id = self.allocate_cached_page().await?;
        let old_root_id = self.tree.root_page_id();

        let mut new_root = BTreeInternalPage::new(new_root_id, self.tree.height() as u16);
        new_root.set_leftmost_child(old_root_id);
        new_root.insert(key, right_child)?;

        self.write_cached_page(new_root_id, new_root.as_bytes()).await?;

        self.tree.set_root_page_id(new_root_id);
        self.tree.increment_height();

        Ok(())
    }

    /// Deletes a key from the B+ tree.
    ///
    /// Returns true if the key was found and deleted.
    pub async fn delete(&mut self, key: &[u8]) -> Result<bool> {
        let path = self.find_path(key).await?;

        let leaf_page_id = *path.last().ok_or_else(|| {
            ZyronError::BTreeCorrupted("empty path".to_string())
        })?;

        let page_data = self.read_cached_page(leaf_page_id).await?;
        let mut leaf = BTreeLeafPage::from_bytes(page_data);

        match leaf.delete(key) {
            DeleteResult::Ok => {
                self.write_cached_page(leaf_page_id, leaf.as_bytes()).await?;
                Ok(true)
            }
            DeleteResult::Underfull => {
                self.write_cached_page(leaf_page_id, leaf.as_bytes()).await?;
                // For simplicity, we don't rebalance on delete in this implementation.
                // A production implementation would handle underflow here.
                Ok(true)
            }
            DeleteResult::NotFound => Ok(false),
        }
    }

    /// Performs a range scan from start_key to end_key (inclusive).
    ///
    /// Returns all matching (key, tuple_id) pairs in sorted order.
    pub async fn range_scan(
        &self,
        start_key: Option<&[u8]>,
        end_key: Option<&[u8]>,
    ) -> Result<Vec<(Bytes, TupleId)>> {
        let mut results = Vec::new();

        // Find starting leaf
        let start_leaf_id = match start_key {
            Some(key) => self.find_leaf(key).await?,
            None => self.find_leftmost_leaf().await?,
        };

        let mut current_page_id = Some(start_leaf_id);

        while let Some(page_id) = current_page_id {
            let page_data = self.read_cached_page(page_id).await?;
            let leaf = BTreeLeafPage::from_bytes(page_data);

            for entry in leaf.entries() {
                // Check start bound
                if let Some(start) = start_key {
                    if entry.key.as_ref() < start {
                        continue;
                    }
                }

                // Check end bound
                if let Some(end) = end_key {
                    if entry.key.as_ref() > end {
                        return Ok(results);
                    }
                }

                results.push((entry.key.clone(), entry.tuple_id));
            }

            current_page_id = leaf.next_leaf();
        }

        Ok(results)
    }

    /// Finds the leftmost leaf page (for full scans).
    async fn find_leftmost_leaf(&self) -> Result<PageId> {
        let mut current_page_id = self.tree.root_page_id();

        // If tree height is 1, root is a leaf
        if self.tree.height() == 1 {
            return Ok(current_page_id);
        }

        // Traverse leftmost path
        for _ in 0..(self.tree.height() - 1) {
            let page_data = self.read_cached_page(current_page_id).await?;
            let internal = BTreeInternalPage::from_bytes(page_data);
            current_page_id = internal.leftmost_child();
        }

        Ok(current_page_id)
    }

    /// Returns an iterator over all entries in key order.
    pub async fn scan_all(&self) -> Result<Vec<(Bytes, TupleId)>> {
        self.range_scan(None, None).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leaf_header_roundtrip() {
        let header = LeafPageHeader {
            num_keys: 42,
            free_space_offset: 100,
            next_leaf: 12345,
            reserved: 0,
        };

        let bytes = header.to_bytes();
        let recovered = LeafPageHeader::from_bytes(&bytes);

        assert_eq!(recovered.num_keys, 42);
        assert_eq!(recovered.free_space_offset, 100);
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
        left.insert(
            Bytes::from_static(b"a"),
            TupleId::new(PageId::new(0, 0), 1),
        )
        .unwrap();

        // Set up right page with multiple entries
        right
            .insert(
                Bytes::from_static(b"b"),
                TupleId::new(PageId::new(0, 0), 2),
            )
            .unwrap();
        right
            .insert(
                Bytes::from_static(b"c"),
                TupleId::new(PageId::new(0, 0), 3),
            )
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

        left.insert(
            Bytes::from_static(b"a"),
            TupleId::new(PageId::new(0, 0), 1),
        )
        .unwrap();
        right
            .insert(
                Bytes::from_static(b"b"),
                TupleId::new(PageId::new(0, 0), 2),
            )
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
    fn test_btree_new() {
        let btree = BTree::new(PageId::new(1, 0), 1);

        assert_eq!(btree.root_page_id(), PageId::new(1, 0));
        assert_eq!(btree.file_id(), 1);
        assert_eq!(btree.height(), 1);
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
}

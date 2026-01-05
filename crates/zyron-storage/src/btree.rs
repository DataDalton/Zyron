//! B+ tree index implementation.
//!
//! This module provides an in-memory B+ tree for indexing tuples.
//! All index nodes are stored in RAM for fast access (~40ns lookup target).
//! Durability is achieved through WAL logging and periodic checkpoints.
//!
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
//!
//! Memory usage: ~5MB per million keys (internal nodes only, leaf nodes scale with key size).

use crate::disk::DiskManager;
use crate::tuple::TupleId;
use bytes::{Bytes, BytesMut};
use parking_lot::RwLock;
use std::sync::Arc;
use zyron_buffer::BufferPool;
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

/// Header for B+ tree leaf pages (slotted page format).
///
/// Layout (16 bytes):
/// - num_slots: 2 bytes (number of entries)
/// - data_end: 2 bytes (offset where entry data begins, grows backward from PAGE_SIZE)
/// - next_leaf: 8 bytes (PageId as u64, for range scans)
/// - reserved: 4 bytes
///
/// Page layout:
/// ```text
/// +------------------------+ 0
/// | Page Header (32 bytes) |
/// +------------------------+ 32
/// | Leaf Header (16 bytes) |
/// +------------------------+ 48 (SLOT_ARRAY_START)
/// | Slot Array             |
/// | [offset:2, len:2] * n  |  <- grows forward
/// +------------------------+ 48 + 4*n
/// |      Free Space        |
/// +------------------------+ data_end
/// | Entry Data             |
/// | (key_len:2 + key + tid)|  <- grows backward from PAGE_SIZE
/// +------------------------+ PAGE_SIZE
/// ```
#[derive(Debug, Clone, Copy)]
pub struct LeafPageHeader {
    /// Number of entries (slots) in this leaf.
    pub num_slots: u16,
    /// Offset where entry data begins (grows backward from PAGE_SIZE).
    pub data_end: u16,
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
            num_slots: 0,
            data_end: PAGE_SIZE as u16, // Data grows backward from end
            next_leaf: u64::MAX,
            reserved: 0,
        }
    }

    /// Serializes to bytes.
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..2].copy_from_slice(&self.num_slots.to_le_bytes());
        buf[2..4].copy_from_slice(&self.data_end.to_le_bytes());
        buf[4..12].copy_from_slice(&self.next_leaf.to_le_bytes());
        buf[12..16].copy_from_slice(&self.reserved.to_le_bytes());
        buf
    }

    /// Deserializes from bytes.
    pub fn from_bytes(buf: &[u8]) -> Self {
        Self {
            num_slots: u16::from_le_bytes([buf[0], buf[1]]),
            data_end: u16::from_le_bytes([buf[2], buf[3]]),
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

/// B+ tree leaf page (slotted page format).
pub struct BTreeLeafPage {
    /// Page data buffer.
    data: Box<[u8; PAGE_SIZE]>,
}

impl BTreeLeafPage {
    /// Slot array start offset after headers.
    const SLOT_ARRAY_START: usize = PageHeader::SIZE + LeafPageHeader::SIZE;

    /// Size of each slot (offset:2 + len:2).
    const SLOT_SIZE: usize = 4;

    /// Creates a new empty leaf page.
    pub fn new(page_id: PageId) -> Self {
        let mut data = Box::new([0u8; PAGE_SIZE]);

        // Initialize page header
        let page_header = PageHeader::new(page_id, PageType::BTreeLeaf);
        data[..PageHeader::SIZE].copy_from_slice(&page_header.to_bytes());

        // Initialize leaf header with data_end at PAGE_SIZE
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
        self.leaf_header().num_slots
    }

    /// Returns the amount of free space available.
    /// Free space = data_end - (SLOT_ARRAY_START + num_slots * SLOT_SIZE)
    pub fn free_space(&self) -> usize {
        let header = self.leaf_header();
        let slot_array_end = Self::SLOT_ARRAY_START + (header.num_slots as usize * Self::SLOT_SIZE);
        (header.data_end as usize).saturating_sub(slot_array_end)
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

    /// Reads all entries from the leaf (via slot array).
    pub fn entries(&self) -> Vec<LeafEntry> {
        let header = self.leaf_header();
        let num_slots = header.num_slots as usize;
        let mut entries = Vec::with_capacity(num_slots);

        for slot_idx in 0..num_slots {
            let slot_offset = Self::SLOT_ARRAY_START + slot_idx * Self::SLOT_SIZE;
            let entry_offset = u16::from_le_bytes([
                self.data[slot_offset],
                self.data[slot_offset + 1],
            ]) as usize;

            if let Some((entry, _)) = LeafEntry::from_bytes(&self.data[entry_offset..]) {
                entries.push(entry);
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
    /// Uses single-pass in-place insertion for efficiency.
    #[inline]
    pub fn insert(&mut self, key: Bytes, tuple_id: TupleId) -> Result<()> {
        Self::insert_in_slice(&mut *self.data, &key, tuple_id)
    }

    /// Writes entries to the page using slotted format.
    fn write_entries(&mut self, entries: &[LeafEntry]) -> Result<()> {
        let num_entries = entries.len();

        // Calculate total space needed
        let slot_space = num_entries * Self::SLOT_SIZE;
        let entry_space: usize = entries.iter().map(|e| e.size_on_disk()).sum();
        let slot_array_end = Self::SLOT_ARRAY_START + slot_space;

        if slot_array_end + entry_space > PAGE_SIZE {
            return Err(ZyronError::NodeFull);
        }

        // Write entries backward from end and slots forward from start
        let mut data_end = PAGE_SIZE;

        for (slot_idx, entry) in entries.iter().enumerate() {
            let bytes = entry.to_bytes();
            data_end -= bytes.len();
            self.data[data_end..data_end + bytes.len()].copy_from_slice(&bytes);

            // Write slot
            let slot_offset = Self::SLOT_ARRAY_START + slot_idx * Self::SLOT_SIZE;
            self.data[slot_offset..slot_offset + 2].copy_from_slice(&(data_end as u16).to_le_bytes());
            self.data[slot_offset + 2..slot_offset + 4].copy_from_slice(&(bytes.len() as u16).to_le_bytes());
        }

        // Update header
        let mut header = self.leaf_header();
        header.num_slots = num_entries as u16;
        header.data_end = data_end as u16;
        self.set_leaf_header(header);
        Ok(())
    }

    /// Gets the value for a key.
    pub fn get(&self, key: &[u8]) -> Option<TupleId> {
        Self::get_in_slice(&*self.data, key)
    }

    /// Gets the value for a key using slotted page format.
    /// Binary search directly on slot array for O(log n) lookup - no offset building needed.
    #[inline]
    pub fn get_in_slice(data: &[u8], key: &[u8]) -> Option<TupleId> {
        // Parse header
        let header_offset = LeafPageHeader::OFFSET;
        let num_slots = u16::from_le_bytes([data[header_offset], data[header_offset + 1]]) as usize;

        if num_slots == 0 {
            return None;
        }

        // Binary search directly on slot array
        let mut low = 0usize;
        let mut high = num_slots;

        while low < high {
            let mid = low + (high - low) / 2;
            let slot_off = Self::SLOT_ARRAY_START + mid * Self::SLOT_SIZE;
            let entry_off = u16::from_le_bytes([data[slot_off], data[slot_off + 1]]) as usize;
            let key_len = u16::from_le_bytes([data[entry_off], data[entry_off + 1]]) as usize;
            let entry_key = &data[entry_off + 2..entry_off + 2 + key_len];

            match key.cmp(entry_key) {
                std::cmp::Ordering::Equal => {
                    let tuple_offset = entry_off + 2 + key_len;
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
                std::cmp::Ordering::Less => high = mid,
                std::cmp::Ordering::Greater => low = mid + 1,
            }
        }
        None
    }

    /// Inserts a key-value pair using slotted page format.
    /// Binary search for O(log n) lookup, only shift 4-byte slots instead of full entries.
    /// Returns Ok(()) on success, Err(NodeFull) if page is full, Err(DuplicateKey) if key exists.
    #[inline]
    pub fn insert_in_slice(data: &mut [u8], key: &[u8], tuple_id: TupleId) -> Result<()> {
        // Parse header
        let header_offset = LeafPageHeader::OFFSET;
        let num_slots = u16::from_le_bytes([data[header_offset], data[header_offset + 1]]) as usize;
        let raw_data_end = u16::from_le_bytes([data[header_offset + 2], data[header_offset + 3]]) as usize;

        // Handle uninitialized pages (data_end == 0 means page was never written)
        let data_end = if raw_data_end == 0 || raw_data_end > PAGE_SIZE {
            PAGE_SIZE
        } else {
            raw_data_end
        };

        // Entry size: key_len(2) + key + page_id(8) + slot_id(2)
        let entry_size = 2 + key.len() + 10;

        // Calculate free space: between slot array end and data start
        let slot_array_end = Self::SLOT_ARRAY_START + num_slots * Self::SLOT_SIZE;
        let free_space = data_end.saturating_sub(slot_array_end);

        // Need space for both entry data and new slot
        if free_space < entry_size + Self::SLOT_SIZE {
            return Err(ZyronError::NodeFull);
        }

        // Binary search through slot array to find insertion point
        let mut low = 0usize;
        let mut high = num_slots;

        while low < high {
            let mid = low + (high - low) / 2;
            let slot_off = Self::SLOT_ARRAY_START + mid * Self::SLOT_SIZE;
            let entry_off = u16::from_le_bytes([data[slot_off], data[slot_off + 1]]) as usize;
            let key_len = u16::from_le_bytes([data[entry_off], data[entry_off + 1]]) as usize;
            let entry_key = &data[entry_off + 2..entry_off + 2 + key_len];

            match key.cmp(entry_key) {
                std::cmp::Ordering::Equal => return Err(ZyronError::DuplicateKey),
                std::cmp::Ordering::Less => high = mid,
                std::cmp::Ordering::Greater => low = mid + 1,
            }
        }

        let insert_slot_idx = low;

        // Write entry data at the end (grows backward)
        let new_data_end = data_end - entry_size;
        let mut write_offset = new_data_end;
        data[write_offset..write_offset + 2].copy_from_slice(&(key.len() as u16).to_le_bytes());
        write_offset += 2;
        data[write_offset..write_offset + key.len()].copy_from_slice(key);
        write_offset += key.len();
        data[write_offset..write_offset + 8].copy_from_slice(&tuple_id.page_id.as_u64().to_le_bytes());
        write_offset += 8;
        data[write_offset..write_offset + 2].copy_from_slice(&tuple_id.slot_id.to_le_bytes());

        // Shift slots forward to make room for new slot (only 4 bytes per slot)
        let insert_slot_offset = Self::SLOT_ARRAY_START + insert_slot_idx * Self::SLOT_SIZE;
        let slots_to_shift = num_slots - insert_slot_idx;
        if slots_to_shift > 0 {
            let shift_start = insert_slot_offset;
            let shift_end = shift_start + slots_to_shift * Self::SLOT_SIZE;
            data.copy_within(shift_start..shift_end, shift_start + Self::SLOT_SIZE);
        }

        // Write new slot (offset:2 + len:2)
        data[insert_slot_offset..insert_slot_offset + 2].copy_from_slice(&(new_data_end as u16).to_le_bytes());
        data[insert_slot_offset + 2..insert_slot_offset + 4].copy_from_slice(&(entry_size as u16).to_le_bytes());

        // Update header
        let new_num_slots = (num_slots + 1) as u16;
        data[header_offset..header_offset + 2].copy_from_slice(&new_num_slots.to_le_bytes());
        data[header_offset + 2..header_offset + 4].copy_from_slice(&(new_data_end as u16).to_le_bytes());

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
        let header = self.leaf_header();
        let entry_data_space = PAGE_SIZE - header.data_end as usize;
        let slot_space = header.num_slots as usize * Self::SLOT_SIZE;
        let used_space = entry_data_space + slot_space;
        let total_data_space = PAGE_SIZE - Self::SLOT_ARRAY_START;
        let fill_ratio = used_space as f64 / total_data_space as f64;
        fill_ratio < MIN_FILL_FACTOR && self.num_entries() > 0
    }

    /// Returns the minimum number of bytes that should be used to avoid underflow.
    pub fn min_used_space(&self) -> usize {
        let total_data_space = PAGE_SIZE - Self::SLOT_ARRAY_START;
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
    /// Uses linear search for small pages (common case), binary search for large ones.
    #[inline]
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

        // For internal nodes with <= 8 entries, linear search is faster
        // (avoids 8KB offset array allocation)
        if num_keys <= 8 {
            let mut offset = Self::DATA_START + Self::LEFTMOST_PTR_SIZE;
            let mut last_child = leftmost;

            for _ in 0..num_keys {
                let key_len = u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
                let entry_key = &data[offset + 2..offset + 2 + key_len];

                if key < entry_key {
                    return last_child;
                }

                // Update last_child to this entry's child pointer
                let child_offset = offset + 2 + key_len;
                last_child = PageId::from_u64(u64::from_le_bytes([
                    data[child_offset],
                    data[child_offset + 1],
                    data[child_offset + 2],
                    data[child_offset + 3],
                    data[child_offset + 4],
                    data[child_offset + 5],
                    data[child_offset + 6],
                    data[child_offset + 7],
                ]));

                offset += 2 + key_len + 8;
            }

            return last_child;
        }

        // For larger pages, use binary search with offset indexing
        let mut offsets = [0usize; 1024];
        let limit = num_keys.min(1024);
        let mut offset = Self::DATA_START + Self::LEFTMOST_PTR_SIZE;

        for i in 0..limit {
            offsets[i] = offset;
            let key_len = u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
            offset += 2 + key_len + 8;
        }

        let mut low = 0usize;
        let mut high = limit;

        while low < high {
            let mid = low + (high - low) / 2;
            let entry_offset = offsets[mid];
            let key_len = u16::from_le_bytes([data[entry_offset], data[entry_offset + 1]]) as usize;
            let entry_key = &data[entry_offset + 2..entry_offset + 2 + key_len];

            if key < entry_key {
                high = mid;
            } else {
                low = mid + 1;
            }
        }

        if low == 0 {
            leftmost
        } else {
            let entry_offset = offsets[low - 1];
            let key_len = u16::from_le_bytes([data[entry_offset], data[entry_offset + 1]]) as usize;
            let child_offset = entry_offset + 2 + key_len;
            PageId::from_u64(u64::from_le_bytes([
                data[child_offset],
                data[child_offset + 1],
                data[child_offset + 2],
                data[child_offset + 3],
                data[child_offset + 4],
                data[child_offset + 5],
                data[child_offset + 6],
                data[child_offset + 7],
            ]))
        }
    }

    /// Inserts a key and right child pointer.
    /// Uses in-place insertion for efficiency.
    #[inline]
    pub fn insert(&mut self, key: Bytes, right_child: PageId) -> Result<()> {
        Self::insert_in_slice(&mut *self.data, key.as_ref(), right_child)
    }

    /// Inserts a key and child pointer directly into raw page data.
    /// Returns Ok(()) on success, Err(NodeFull) if page is full.
    pub fn insert_in_slice(data: &mut [u8], key: &[u8], right_child: PageId) -> Result<()> {
        // Parse header
        let header_offset = InternalPageHeader::OFFSET;
        let num_keys = u16::from_le_bytes([data[header_offset], data[header_offset + 1]]) as usize;
        let raw_free_offset = u16::from_le_bytes([data[header_offset + 2], data[header_offset + 3]]) as usize;

        // Handle uninitialized pages
        let free_space_offset = if raw_free_offset < Self::DATA_START + Self::LEFTMOST_PTR_SIZE {
            Self::DATA_START + Self::LEFTMOST_PTR_SIZE
        } else {
            raw_free_offset
        };

        // Entry size: key_len(2) + key + page_id(8)
        let entry_size = 2 + key.len() + 8;
        let free_space = PAGE_SIZE - free_space_offset;

        if free_space < entry_size {
            return Err(ZyronError::NodeFull);
        }

        // Find insertion point using linear scan (internal nodes have fewer entries)
        let mut insert_offset = Self::DATA_START + Self::LEFTMOST_PTR_SIZE;
        let mut offset = insert_offset;

        for _ in 0..num_keys {
            let key_len = u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
            let entry_key = &data[offset + 2..offset + 2 + key_len];
            let entry_total = 2 + key_len + 8;

            if key < entry_key {
                insert_offset = offset;
                break;
            }
            offset += entry_total;
            insert_offset = offset;
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
        data[write_offset..write_offset + 8].copy_from_slice(&right_child.as_u64().to_le_bytes());

        // Update header
        let new_num_keys = (num_keys + 1) as u16;
        let new_free_offset = (free_space_offset + entry_size) as u16;
        data[header_offset..header_offset + 2].copy_from_slice(&new_num_keys.to_le_bytes());
        data[header_offset + 2..header_offset + 4].copy_from_slice(&new_free_offset.to_le_bytes());

        Ok(())
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

/// In-memory page storage for B+Tree nodes.
///
/// All pages are stored in RAM in a Vec. Page numbers map directly to Vec indices.
/// This eliminates disk I/O overhead and BufferPool lock contention from the hot path.
pub struct InMemoryPageStore {
    /// Pages stored by page number (index = page_num).
    pages: Vec<Box<[u8; PAGE_SIZE]>>,
}

impl InMemoryPageStore {
    /// Creates a new empty page store.
    fn new() -> Self {
        Self { pages: Vec::new() }
    }

    /// Allocates a new page and returns its page number.
    #[inline]
    fn allocate(&mut self) -> u32 {
        let page_num = self.pages.len() as u32;
        self.pages.push(Box::new([0u8; PAGE_SIZE]));
        page_num
    }

    /// Gets a page by page number (read-only).
    #[inline]
    fn get(&self, page_num: u32) -> Option<&[u8; PAGE_SIZE]> {
        self.pages.get(page_num as usize).map(|p| &**p)
    }

    /// Gets a mutable page by page number.
    #[inline]
    fn get_mut(&mut self, page_num: u32) -> Option<&mut [u8; PAGE_SIZE]> {
        self.pages.get_mut(page_num as usize).map(|p| &mut **p)
    }

    /// Returns total number of pages.
    #[inline]
    fn len(&self) -> usize {
        self.pages.len()
    }

    /// Writes page data at a specific page number.
    #[inline]
    fn write(&mut self, page_num: u32, data: &[u8; PAGE_SIZE]) {
        if let Some(page) = self.pages.get_mut(page_num as usize) {
            page.copy_from_slice(data);
        }
    }
}

/// B+ tree index with fully in-memory storage.
///
/// All nodes (internal and leaf) are stored in RAM for minimal lookup latency.
/// Provides tree-level operations (search, insert, delete, range_scan).
///
/// Performance characteristics:
/// - Lookup: O(log n) with ~40ns per level (direct memory access)
/// - Insert: O(log n) amortized, may trigger splits
/// - Memory: ~5MB per million keys (depends on key size)
///
/// Durability: WAL logs modifications, checkpoint persists full tree to disk.
/// Recovery: Load checkpoint, replay WAL.
pub struct BTreeIndex {
    /// In-memory page storage (all nodes stored here).
    pages: RwLock<InMemoryPageStore>,
    /// Root page number (index into pages).
    root_page_num: std::sync::atomic::AtomicU32,
    /// Tree height (1 = just root as leaf).
    height: std::sync::atomic::AtomicU32,
    /// File ID for this index (used for PageId construction).
    file_id: u32,
    /// Disk manager for checkpoint I/O (not used in hot path).
    #[allow(dead_code)]
    disk: Arc<DiskManager>,
    /// Buffer pool reference (kept for compatibility, not used in hot path).
    #[allow(dead_code)]
    pool: Arc<BufferPool>,
}

impl BTreeIndex {
    /// Maximum B+Tree height (supports billions of keys).
    const MAX_HEIGHT: usize = 16;

    /// Creates a new B+ tree index with fully in-memory storage.
    ///
    /// All pages are stored in RAM. Disk/BufferPool are kept for compatibility
    /// but not used in the hot path.
    pub async fn create(
        disk: Arc<DiskManager>,
        pool: Arc<BufferPool>,
        file_id: u32,
    ) -> Result<Self> {
        use std::sync::atomic::AtomicU32;

        // Create in-memory page store
        let mut store = InMemoryPageStore::new();

        // Allocate root leaf page (page 0)
        let root_page_num = store.allocate();
        let root_page_id = PageId::new(file_id, root_page_num);

        // Initialize root as empty leaf
        let root_page = BTreeLeafPage::new(root_page_id);
        store.write(root_page_num, root_page.as_bytes());

        Ok(Self {
            pages: RwLock::new(store),
            root_page_num: AtomicU32::new(root_page_num),
            height: AtomicU32::new(1),
            file_id,
            disk,
            pool,
        })
    }

    /// Opens an existing B+ tree index (placeholder - would load from checkpoint).
    pub async fn open(
        disk: Arc<DiskManager>,
        pool: Arc<BufferPool>,
        file_id: u32,
        _root_page_id: PageId,
        height: u32,
    ) -> Result<Self> {
        use std::sync::atomic::AtomicU32;

        // Create empty in-memory store (in production, would load from checkpoint)
        let mut store = InMemoryPageStore::new();
        let root_page_num = store.allocate();
        let root_page_id = PageId::new(file_id, root_page_num);
        let root_page = BTreeLeafPage::new(root_page_id);
        store.write(root_page_num, root_page.as_bytes());

        Ok(Self {
            pages: RwLock::new(store),
            root_page_num: AtomicU32::new(root_page_num),
            height: AtomicU32::new(height),
            file_id,
            disk,
            pool,
        })
    }

    /// Returns the root page ID.
    #[inline]
    pub fn root_page_id(&self) -> PageId {
        PageId::new(
            self.file_id,
            self.root_page_num.load(std::sync::atomic::Ordering::Acquire),
        )
    }

    /// Returns the file ID.
    #[inline]
    pub fn file_id(&self) -> u32 {
        self.file_id
    }

    /// Returns the tree height.
    #[inline]
    pub fn height(&self) -> u32 {
        self.height.load(std::sync::atomic::Ordering::Acquire)
    }

    // =========================================================================
    // Core In-Memory Operations (Synchronous)
    // =========================================================================

    /// Finds the leaf page number for a given key using existing pages reference.
    #[inline]
    fn find_leaf_in_pages(&self, pages: &InMemoryPageStore, key: &[u8]) -> u32 {
        let height = self.height.load(std::sync::atomic::Ordering::Relaxed);
        let mut current = self.root_page_num.load(std::sync::atomic::Ordering::Relaxed);

        // Height 1 means root is the leaf
        if height == 1 {
            return current;
        }

        // Traverse internal nodes
        for _ in 0..(height - 1) {
            if let Some(data) = pages.get(current) {
                let child_page_id = BTreeInternalPage::find_child_in_slice(data, key);
                current = child_page_id.page_num;
            } else {
                break;
            }
        }

        current
    }

    /// Finds the leaf page number for a given key (takes read lock).
    #[inline]
    fn find_leaf_page_num(&self, key: &[u8]) -> u32 {
        let pages = self.pages.read();
        self.find_leaf_in_pages(&pages, key)
    }

    /// Searches for a key synchronously. All data is in RAM.
    /// Single lock acquisition for entire operation.
    #[inline]
    pub fn search_sync(&self, key: &[u8]) -> Option<TupleId> {
        let pages = self.pages.read();
        let leaf_page_num = self.find_leaf_in_pages(&pages, key);

        if let Some(data) = pages.get(leaf_page_num) {
            BTreeLeafPage::get_in_slice(data, key)
        } else {
            None
        }
    }

    /// Inserts a key-value pair synchronously. All data is in RAM.
    /// Single lock acquisition for entire operation (no split case).
    #[inline]
    pub fn insert_sync(&self, key: &[u8], tuple_id: TupleId) -> Result<()> {
        if key.len() > MAX_KEY_SIZE {
            return Err(ZyronError::KeyTooLarge {
                size: key.len(),
                max: MAX_KEY_SIZE,
            });
        }

        // Fast path: single write lock for find + insert
        {
            let mut pages = self.pages.write();
            let leaf_page_num = self.find_leaf_in_pages(&pages, key);

            if let Some(data) = pages.get_mut(leaf_page_num) {
                match BTreeLeafPage::insert_in_slice(data, key, tuple_id) {
                    Ok(()) => return Ok(()),
                    Err(ZyronError::NodeFull) => {
                        // Need split - fall through
                    }
                    Err(e) => return Err(e),
                }
            }
        }

        // Slow path: need to split
        self.insert_with_split_sync(Bytes::copy_from_slice(key), tuple_id)
    }

    /// Inserts a key-value pair with exclusive access (no locking).
    /// Use when caller has &mut BTreeIndex for maximum performance.
    /// Fully inlined fast path for minimal overhead.
    #[inline(always)]
    pub fn insert_exclusive(&mut self, key: &[u8], tuple_id: TupleId) -> Result<()> {
        if key.len() > MAX_KEY_SIZE {
            return Err(ZyronError::KeyTooLarge {
                size: key.len(),
                max: MAX_KEY_SIZE,
            });
        }

        let pages = self.pages.get_mut();
        let height = *self.height.get_mut();
        let root = *self.root_page_num.get_mut();

        // Inline leaf finding for fast path
        let mut current = root;
        let mut path = [0u32; Self::MAX_HEIGHT];
        let mut path_len = 0;

        path[path_len] = current;
        path_len += 1;

        if height > 1 {
            for _ in 0..(height - 1) {
                if let Some(data) = pages.get(current) {
                    let child = BTreeInternalPage::find_child_in_slice(data, key);
                    current = child.page_num;
                    path[path_len] = current;
                    path_len += 1;
                } else {
                    return Err(ZyronError::BTreeCorrupted("internal node not found".to_string()));
                }
            }
        }

        // Try insert in leaf (fast path - most common)
        if let Some(data) = pages.get_mut(current) {
            match BTreeLeafPage::insert_in_slice(data, key, tuple_id) {
                Ok(()) => return Ok(()),
                Err(ZyronError::NodeFull) => {
                    // Fall through to split path
                }
                Err(e) => return Err(e),
            }
        } else {
            return Err(ZyronError::BTreeCorrupted("leaf not found".to_string()));
        }

        // Split handling (rare path)
        self.insert_with_split_exclusive(Bytes::copy_from_slice(key), tuple_id, &path[..path_len])
    }

    /// Finds path with exclusive access.
    #[inline]
    fn find_path_exclusive(&mut self, key: &[u8]) -> ([u32; Self::MAX_HEIGHT], usize) {
        let pages = self.pages.get_mut();
        let height = *self.height.get_mut();
        let root = *self.root_page_num.get_mut();

        let mut path = [0u32; Self::MAX_HEIGHT];
        let mut path_len = 0;
        let mut current = root;

        path[path_len] = current;
        path_len += 1;

        if height == 1 {
            return (path, path_len);
        }

        for _ in 0..(height - 1) {
            if let Some(data) = pages.get(current) {
                let child_page_id = BTreeInternalPage::find_child_in_slice(data, key);
                current = child_page_id.page_num;
                path[path_len] = current;
                path_len += 1;
            } else {
                break;
            }
        }

        (path, path_len)
    }

    /// Insert with split using exclusive access (no locking).
    fn insert_with_split_exclusive(&mut self, key: Bytes, tuple_id: TupleId, path: &[u32]) -> Result<()> {
        let leaf_page_num = path[path.len() - 1];

        // Get direct access to pages
        let pages = self.pages.get_mut();

        // Read and split the leaf
        let leaf_data = *pages.get(leaf_page_num)
            .ok_or_else(|| ZyronError::BTreeCorrupted("leaf not found".to_string()))?;
        let mut leaf = BTreeLeafPage::from_bytes(leaf_data);

        // Allocate new page
        let new_page_num = pages.allocate();
        let new_page_id = PageId::new(self.file_id, new_page_num);
        let (split_key, mut right_leaf) = leaf.split(new_page_id);

        // Insert into appropriate leaf
        if key.as_ref() < split_key.as_ref() {
            leaf.insert(key, tuple_id)?;
        } else {
            right_leaf.insert(key, tuple_id)?;
        }

        // Write both leaves
        pages.write(leaf_page_num, leaf.as_bytes());
        pages.write(new_page_num, right_leaf.as_bytes());

        // Propagate split up
        if path.len() < 2 {
            // Root was a leaf, create new root
            self.create_new_root_exclusive(split_key, new_page_num)
        } else {
            self.propagate_split_exclusive(split_key, new_page_num, path)
        }
    }

    /// Propagate split with exclusive access.
    fn propagate_split_exclusive(&mut self, key: Bytes, new_child: u32, path: &[u32]) -> Result<()> {
        let mut current_key = key;
        let mut current_child = new_child;
        let mut parent_idx = path.len() - 2;

        loop {
            let parent_page_num = path[parent_idx];
            let pages = self.pages.get_mut();

            let parent_data = *pages.get(parent_page_num)
                .ok_or_else(|| ZyronError::BTreeCorrupted("parent not found".to_string()))?;
            let mut parent = BTreeInternalPage::from_bytes(parent_data);

            let new_child_page_id = PageId::new(self.file_id, current_child);

            match parent.insert(current_key.clone(), new_child_page_id) {
                Ok(()) => {
                    pages.write(parent_page_num, parent.as_bytes());
                    return Ok(());
                }
                Err(ZyronError::NodeFull) => {
                    // Split the internal node
                    let new_page_num = pages.allocate();
                    let new_page_id = PageId::new(self.file_id, new_page_num);
                    let (promoted_key, mut right_internal) = parent.split(new_page_id);

                    if current_key.as_ref() < promoted_key.as_ref() {
                        parent.insert(current_key, new_child_page_id)?;
                    } else {
                        right_internal.insert(current_key, new_child_page_id)?;
                    }

                    pages.write(parent_page_num, parent.as_bytes());
                    pages.write(new_page_num, right_internal.as_bytes());

                    if parent_idx == 0 {
                        return self.create_new_root_exclusive(promoted_key, new_page_num);
                    }

                    current_key = promoted_key;
                    current_child = new_page_num;
                    parent_idx -= 1;
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Create new root with exclusive access.
    fn create_new_root_exclusive(&mut self, key: Bytes, right_child: u32) -> Result<()> {
        let pages = self.pages.get_mut();
        let old_root = *self.root_page_num.get_mut();
        let height = *self.height.get_mut();

        let new_root_num = pages.allocate();
        let new_root_id = PageId::new(self.file_id, new_root_num);
        let old_root_id = PageId::new(self.file_id, old_root);
        let right_child_id = PageId::new(self.file_id, right_child);

        let mut new_root = BTreeInternalPage::new(new_root_id, height as u16);
        new_root.set_leftmost_child(old_root_id);
        new_root.insert(key, right_child_id)?;

        pages.write(new_root_num, new_root.as_bytes());

        *self.root_page_num.get_mut() = new_root_num;
        *self.height.get_mut() = height + 1;

        Ok(())
    }

    /// Searches with exclusive access (no locking).
    #[inline]
    pub fn search_exclusive(&mut self, key: &[u8]) -> Option<TupleId> {
        let pages = self.pages.get_mut();
        let height = *self.height.get_mut();
        let root = *self.root_page_num.get_mut();

        let leaf_page_num = Self::find_leaf_direct(pages, height, root, key);

        if let Some(data) = pages.get(leaf_page_num) {
            BTreeLeafPage::get_in_slice(data, key)
        } else {
            None
        }
    }

    /// Direct leaf lookup without any locking.
    #[inline]
    fn find_leaf_direct(pages: &InMemoryPageStore, height: u32, root: u32, key: &[u8]) -> u32 {
        let mut current = root;

        if height == 1 {
            return current;
        }

        for _ in 0..(height - 1) {
            if let Some(data) = pages.get(current) {
                let child_page_id = BTreeInternalPage::find_child_in_slice(data, key);
                current = child_page_id.page_num;
            } else {
                break;
            }
        }

        current
    }

    /// Finds the path from root to leaf (returns page numbers).
    fn find_path_sync(&self, key: &[u8]) -> ([u32; Self::MAX_HEIGHT], usize) {
        let pages = self.pages.read();
        let height = self.height.load(std::sync::atomic::Ordering::Acquire);
        let root = self.root_page_num.load(std::sync::atomic::Ordering::Acquire);

        let mut path = [0u32; Self::MAX_HEIGHT];
        let mut path_len = 0;
        let mut current = root;

        path[path_len] = current;
        path_len += 1;

        if height == 1 {
            return (path, path_len);
        }

        for _ in 0..(height - 1) {
            if let Some(data) = pages.get(current) {
                let child_page_id = BTreeInternalPage::find_child_in_slice(data, key);
                current = child_page_id.page_num;
                path[path_len] = current;
                path_len += 1;
            } else {
                break;
            }
        }

        (path, path_len)
    }

    /// Insert with split handling (synchronous).
    fn insert_with_split_sync(&self, key: Bytes, tuple_id: TupleId) -> Result<()> {
        let (path, path_len) = self.find_path_sync(&key);
        if path_len == 0 {
            return Err(ZyronError::BTreeCorrupted("empty path".to_string()));
        }

        let leaf_page_num = path[path_len - 1];

        let mut pages = self.pages.write();

        // Read the leaf page
        let leaf_data = pages.get(leaf_page_num)
            .ok_or_else(|| ZyronError::BTreeCorrupted("leaf not found".to_string()))?;
        let mut leaf = BTreeLeafPage::from_bytes(*leaf_data);

        // Allocate new page for right sibling
        let new_page_num = pages.allocate();
        let new_page_id = PageId::new(self.file_id, new_page_num);
        let (split_key, mut right_leaf) = leaf.split(new_page_id);

        // Insert the new key into appropriate leaf
        if key.as_ref() < split_key.as_ref() {
            leaf.insert(key, tuple_id)?;
        } else {
            right_leaf.insert(key, tuple_id)?;
        }

        // Write both leaves
        pages.write(leaf_page_num, leaf.as_bytes());
        pages.write(new_page_num, right_leaf.as_bytes());

        // Propagate split up the tree
        drop(pages); // Release lock before recursive call
        self.propagate_split_sync(split_key, new_page_num, &path[..path_len])
    }

    /// Propagate split up the tree.
    fn propagate_split_sync(&self, key: Bytes, new_child: u32, path: &[u32]) -> Result<()> {
        if path.len() < 2 {
            // Root was a leaf, create new root
            return self.create_new_root_sync(key, new_child);
        }

        let mut current_key = key;
        let mut current_child = new_child;
        let mut parent_idx = path.len() - 2;

        loop {
            let parent_page_num = path[parent_idx];

            let mut pages = self.pages.write();

            let parent_data = pages.get(parent_page_num)
                .ok_or_else(|| ZyronError::BTreeCorrupted("parent not found".to_string()))?;
            let mut parent = BTreeInternalPage::from_bytes(*parent_data);

            let new_child_page_id = PageId::new(self.file_id, current_child);

            match parent.insert(current_key.clone(), new_child_page_id) {
                Ok(()) => {
                    pages.write(parent_page_num, parent.as_bytes());
                    return Ok(());
                }
                Err(ZyronError::NodeFull) => {
                    // Split the internal node
                    let new_page_num = pages.allocate();
                    let new_page_id = PageId::new(self.file_id, new_page_num);
                    let (promoted_key, mut right_internal) = parent.split(new_page_id);

                    // Insert into appropriate side
                    if current_key.as_ref() < promoted_key.as_ref() {
                        parent.insert(current_key, new_child_page_id)?;
                    } else {
                        right_internal.insert(current_key, new_child_page_id)?;
                    }

                    // Write both pages
                    pages.write(parent_page_num, parent.as_bytes());
                    pages.write(new_page_num, right_internal.as_bytes());

                    drop(pages);

                    // Continue propagating up
                    if parent_idx == 0 {
                        return self.create_new_root_sync(promoted_key, new_page_num);
                    }

                    current_key = promoted_key;
                    current_child = new_page_num;
                    parent_idx -= 1;
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Creates a new root when the current root splits.
    fn create_new_root_sync(&self, key: Bytes, right_child: u32) -> Result<()> {
        let mut pages = self.pages.write();
        let old_root = self.root_page_num.load(std::sync::atomic::Ordering::Acquire);
        let height = self.height.load(std::sync::atomic::Ordering::Acquire);

        let new_root_num = pages.allocate();
        let new_root_id = PageId::new(self.file_id, new_root_num);
        let old_root_id = PageId::new(self.file_id, old_root);
        let right_child_id = PageId::new(self.file_id, right_child);

        let mut new_root = BTreeInternalPage::new(new_root_id, height as u16);
        new_root.set_leftmost_child(old_root_id);
        new_root.insert(key, right_child_id)?;

        pages.write(new_root_num, new_root.as_bytes());

        self.root_page_num.store(new_root_num, std::sync::atomic::Ordering::Release);
        self.height.store(height + 1, std::sync::atomic::Ordering::Release);

        Ok(())
    }

    /// Deletes a key synchronously.
    pub fn delete_sync(&self, key: &[u8]) -> bool {
        let leaf_page_num = self.find_leaf_page_num(key);

        let mut pages = self.pages.write();
        if let Some(data) = pages.get(leaf_page_num) {
            let mut leaf = BTreeLeafPage::from_bytes(*data);
            match leaf.delete(key) {
                DeleteResult::Ok | DeleteResult::Underfull => {
                    pages.write(leaf_page_num, leaf.as_bytes());
                    true
                }
                DeleteResult::NotFound => false,
            }
        } else {
            false
        }
    }

    /// Range scan synchronously.
    pub fn range_scan_sync(
        &self,
        start_key: Option<&[u8]>,
        end_key: Option<&[u8]>,
    ) -> Vec<(Bytes, TupleId)> {
        let pages = self.pages.read();
        let mut results = Vec::new();

        // Find starting leaf
        let start_leaf_num = match start_key {
            Some(key) => self.find_leaf_page_num(key),
            None => self.find_leftmost_leaf_num(&pages),
        };

        let mut current_page_num = Some(start_leaf_num);

        while let Some(page_num) = current_page_num {
            if let Some(data) = pages.get(page_num) {
                let leaf = BTreeLeafPage::from_bytes(*data);

                for entry in leaf.entries() {
                    if let Some(start) = start_key {
                        if entry.key.as_ref() < start {
                            continue;
                        }
                    }

                    if let Some(end) = end_key {
                        if entry.key.as_ref() > end {
                            return results;
                        }
                    }

                    results.push((entry.key.clone(), entry.tuple_id));
                }

                current_page_num = leaf.next_leaf().map(|p| p.page_num);
            } else {
                break;
            }
        }

        results
    }

    /// Find leftmost leaf page number.
    fn find_leftmost_leaf_num(&self, pages: &InMemoryPageStore) -> u32 {
        let height = self.height.load(std::sync::atomic::Ordering::Acquire);
        let mut current = self.root_page_num.load(std::sync::atomic::Ordering::Acquire);

        if height == 1 {
            return current;
        }

        for _ in 0..(height - 1) {
            if let Some(data) = pages.get(current) {
                let internal = BTreeInternalPage::from_bytes(*data);
                current = internal.leftmost_child().page_num;
            } else {
                break;
            }
        }

        current
    }

    // =========================================================================
    // Async Wrappers (for API compatibility)
    // =========================================================================

    /// Searches for a key. Async wrapper around sync operation.
    pub async fn search(&self, key: &[u8]) -> Result<Option<TupleId>> {
        Ok(self.search_sync(key))
    }

    /// Inserts a key-value pair. Uses lock-free path since we have &mut self.
    pub async fn insert(&mut self, key: Bytes, tuple_id: TupleId) -> Result<()> {
        self.insert_exclusive(key.as_ref(), tuple_id)
    }

    /// Deletes a key. Async wrapper around sync operation.
    pub async fn delete(&mut self, key: &[u8]) -> Result<bool> {
        Ok(self.delete_sync(key))
    }

    /// Range scan. Async wrapper around sync operation.
    pub async fn range_scan(
        &self,
        start_key: Option<&[u8]>,
        end_key: Option<&[u8]>,
    ) -> Result<Vec<(Bytes, TupleId)>> {
        Ok(self.range_scan_sync(start_key, end_key))
    }

    /// Scan all entries. Async wrapper around sync operation.
    pub async fn scan_all(&self) -> Result<Vec<(Bytes, TupleId)>> {
        Ok(self.range_scan_sync(None, None))
    }

    /// Flush is a no-op for in-memory B+Tree (persistence handled by checkpoint).
    pub async fn flush(&self) -> Result<()> {
        Ok(())
    }

    /// Batch insert multiple entries.
    pub async fn insert_batch(&mut self, entries: Vec<(Bytes, TupleId)>) -> Result<usize> {
        let mut inserted = 0;
        for (key, tuple_id) in entries {
            if key.len() <= MAX_KEY_SIZE {
                self.insert_sync(key.as_ref(), tuple_id)?;
                inserted += 1;
            }
        }
        Ok(inserted)
    }

    /// Warm cache is a no-op for in-memory B+Tree (all data already in RAM).
    pub async fn warm_cache(&self) -> Result<usize> {
        Ok(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

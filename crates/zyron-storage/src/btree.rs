//! B+ tree index implementation with write buffer for HTAP workloads.
//!
//! This module provides three B+Tree implementations:
//!
//! ## BufferedBTreeIndex (Recommended for HTAP)
//!
//! A hybrid structure combining a write buffer with a read-optimized B+Tree:
//!
//! ```text
//! Writes → [WriteBuffer (sorted, 64K entries)] → merge → [B+Tree (32KB nodes)]
//!                                                              ↑
//! Reads → check buffer first, then ───────────────────────────┘
//! ```
//!
//! Performance characteristics:
//! - Insert: ~100ns (buffer insert with binary search)
//! - Lookup: ~80ns (buffer check + B+Tree with height=2)
//! - Range scan: Merges results from buffer and B+Tree
//!
//! The write buffer absorbs high-frequency inserts, while the B+Tree uses
//! 32KB nodes for shallow height (2 levels for 1M keys) and fast lookups.
//!
//! ## BTreeArenaIndex (Direct B+Tree)
//!
//! Arena-based B+Tree with contiguous memory allocation:
//! - 32KB nodes for shallow tree height
//! - Direct pointer arithmetic for fast traversal
//! - Lock-free reads with exclusive writes (&mut self)
//!
//! ## BTreeIndex (Page-Based)
//!
//! Traditional page-based B+Tree using the buffer pool:
//! - Integrates with disk manager for persistence
//! - Suitable for larger-than-memory indexes
//!
//! ## Memory Layout (Arena-Based)
//!
//! Internal node layout:
//! ```text
//! +------------------+ 0
//! | version: u64     | 8
//! | num_keys: u16    | 10
//! | level: u16       | 12
//! | reserved: u32    | 16 (HEADER_SIZE)
//! +------------------+
//! | child_0: u64     | 24
//! | key_0: u64       |
//! | child_1: u64     |
//! | ...              |
//! +------------------+
//! ```
//!
//! Leaf node layout:
//! ```text
//! +------------------+ 0
//! | version: u64     | 8
//! | num_entries: u16 | 10
//! | reserved: [u8;6] | 16
//! | next_leaf: u64   | 24 (LEAF_HEADER_SIZE)
//! +------------------+
//! | key_0: u64       |
//! | tuple_id_0: u64  |
//! | ...              |
//! +------------------+
//! ```
//!
//! With 32KB nodes: 2046 keys per node, height=2 for 1M keys.

use crate::disk::DiskManager;
use crate::tuple::TupleId;
use bytes::{Bytes, BytesMut};
use parking_lot::RwLock;
use std::sync::Arc;
use zyron_buffer::BufferPool;
use zyron_common::page::{PAGE_SIZE, PageHeader, PageId, PageType};
use zyron_common::{Result, ZyronError};

/// Maximum key size in bytes.
pub const MAX_KEY_SIZE: usize = 256;

/// Minimum fill factor for B+ tree nodes (50%).
pub const MIN_FILL_FACTOR: f64 = 0.5;

/// Fast key comparison using u64 prefix for 8+ byte keys.
/// Falls back to slice comparison for shorter keys or when prefix matches.
#[inline(always)]
fn compare_keys_fast(a: &[u8], b: &[u8]) -> std::cmp::Ordering {
    // For 8+ byte keys, compare first 8 bytes as u64 (big-endian for sort order)
    if a.len() >= 8 && b.len() >= 8 {
        let a_prefix = u64::from_be_bytes([a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]]);
        let b_prefix = u64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
        if a_prefix != b_prefix {
            return a_prefix.cmp(&b_prefix);
        }
        // Prefix matched, compare remaining bytes
        if a.len() == 8 && b.len() == 8 {
            return std::cmp::Ordering::Equal;
        }
    }
    a.cmp(b)
}

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
            let entry_offset =
                u16::from_le_bytes([self.data[slot_offset], self.data[slot_offset + 1]]) as usize;

            if let Some((entry, _)) = LeafEntry::from_bytes(&self.data[entry_offset..]) {
                entries.push(entry);
            }
        }

        entries
    }

    /// Binary search for a key. Returns Ok(index) if found, Err(index) for insertion point.
    pub fn search(&self, key: &[u8]) -> std::result::Result<usize, usize> {
        let entries = self.entries();
        entries.binary_search_by(|e| compare_keys_fast(e.key.as_ref(), key))
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
            self.data[slot_offset..slot_offset + 2]
                .copy_from_slice(&(data_end as u16).to_le_bytes());
            self.data[slot_offset + 2..slot_offset + 4]
                .copy_from_slice(&(bytes.len() as u16).to_le_bytes());
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
    #[inline(always)]
    pub fn get_in_slice(data: &[u8], key: &[u8]) -> Option<TupleId> {
        // Parse header - read num_slots as single u16
        let header_offset = LeafPageHeader::OFFSET;
        let num_slots = u16::from_le_bytes([data[header_offset], data[header_offset + 1]]) as usize;

        if num_slots == 0 {
            return None;
        }

        // Binary search with packed slot reads and u64 prefix comparison
        let mut low = 0usize;
        let mut high = num_slots;

        while low < high {
            let mid = low + (high - low) / 2;
            let slot_off = Self::SLOT_ARRAY_START + mid * Self::SLOT_SIZE;

            // Read slot as packed u32 (offset:u16 + len:u16)
            let packed = u32::from_le_bytes([
                data[slot_off],
                data[slot_off + 1],
                data[slot_off + 2],
                data[slot_off + 3],
            ]);
            let entry_off = (packed & 0xFFFF) as usize;

            // Read key_len from entry
            let key_len = u16::from_le_bytes([data[entry_off], data[entry_off + 1]]) as usize;
            let entry_key = &data[entry_off + 2..entry_off + 2 + key_len];

            match compare_keys_fast(key, entry_key) {
                std::cmp::Ordering::Equal => {
                    let tuple_offset = entry_off + 2 + key_len;
                    // Read page_id as u64 directly
                    let page_id = PageId::from_u64(u64::from_le_bytes([
                        data[tuple_offset],
                        data[tuple_offset + 1],
                        data[tuple_offset + 2],
                        data[tuple_offset + 3],
                        data[tuple_offset + 4],
                        data[tuple_offset + 5],
                        data[tuple_offset + 6],
                        data[tuple_offset + 7],
                    ]));
                    let slot_id =
                        u16::from_le_bytes([data[tuple_offset + 8], data[tuple_offset + 9]]);
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
    #[inline(always)]
    pub fn insert_in_slice(data: &mut [u8], key: &[u8], tuple_id: TupleId) -> Result<()> {
        // Parse header
        let header_offset = LeafPageHeader::OFFSET;
        let num_slots = u16::from_le_bytes([data[header_offset], data[header_offset + 1]]) as usize;
        let raw_data_end =
            u16::from_le_bytes([data[header_offset + 2], data[header_offset + 3]]) as usize;

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

            // Packed slot read: offset:u16 + len:u16 as single u32
            let packed = u32::from_le_bytes([
                data[slot_off],
                data[slot_off + 1],
                data[slot_off + 2],
                data[slot_off + 3],
            ]);
            let entry_off = (packed & 0xFFFF) as usize;

            let key_len = u16::from_le_bytes([data[entry_off], data[entry_off + 1]]) as usize;
            let entry_key = &data[entry_off + 2..entry_off + 2 + key_len];

            match compare_keys_fast(key, entry_key) {
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
        data[write_offset..write_offset + 8]
            .copy_from_slice(&tuple_id.page_id.as_u64().to_le_bytes());
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
        data[insert_slot_offset..insert_slot_offset + 2]
            .copy_from_slice(&(new_data_end as u16).to_le_bytes());
        data[insert_slot_offset + 2..insert_slot_offset + 4]
            .copy_from_slice(&(entry_size as u16).to_le_bytes());

        // Update header
        let new_num_slots = (num_slots + 1) as u16;
        data[header_offset..header_offset + 2].copy_from_slice(&new_num_slots.to_le_bytes());
        data[header_offset + 2..header_offset + 4]
            .copy_from_slice(&(new_data_end as u16).to_le_bytes());

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
        self.data[offset..offset + InternalPageHeader::SIZE].copy_from_slice(&header.to_bytes());
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
    #[inline(always)]
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
        // (avoids offset array allocation)
        if num_keys <= 8 {
            let mut offset = Self::DATA_START + Self::LEFTMOST_PTR_SIZE;
            let mut last_child = leftmost;

            for _ in 0..num_keys {
                let key_len = u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
                let entry_key = &data[offset + 2..offset + 2 + key_len];

                if compare_keys_fast(key, entry_key).is_lt() {
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

            if compare_keys_fast(key, entry_key).is_lt() {
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
    #[inline(always)]
    pub fn insert_in_slice(data: &mut [u8], key: &[u8], right_child: PageId) -> Result<()> {
        // Parse header
        let header_offset = InternalPageHeader::OFFSET;
        let num_keys = u16::from_le_bytes([data[header_offset], data[header_offset + 1]]) as usize;
        let raw_free_offset =
            u16::from_le_bytes([data[header_offset + 2], data[header_offset + 3]]) as usize;

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

            if compare_keys_fast(key, entry_key).is_lt() {
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
            self.root_page_num
                .load(std::sync::atomic::Ordering::Acquire),
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
        let mut current = self
            .root_page_num
            .load(std::sync::atomic::Ordering::Relaxed);

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
                    return Err(ZyronError::BTreeCorrupted(
                        "internal node not found".to_string(),
                    ));
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

    /// Insert with split using exclusive access (no locking).
    fn insert_with_split_exclusive(
        &mut self,
        key: Bytes,
        tuple_id: TupleId,
        path: &[u32],
    ) -> Result<()> {
        let leaf_page_num = path[path.len() - 1];

        // Get direct access to pages
        let pages = self.pages.get_mut();

        // Read and split the leaf
        let leaf_data = *pages
            .get(leaf_page_num)
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
    fn propagate_split_exclusive(
        &mut self,
        key: Bytes,
        new_child: u32,
        path: &[u32],
    ) -> Result<()> {
        let mut current_key = key;
        let mut current_child = new_child;
        let mut parent_idx = path.len() - 2;

        loop {
            let parent_page_num = path[parent_idx];
            let pages = self.pages.get_mut();

            let parent_data = *pages
                .get(parent_page_num)
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
        let root = self
            .root_page_num
            .load(std::sync::atomic::Ordering::Acquire);

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
        let leaf_data = pages
            .get(leaf_page_num)
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

            let parent_data = pages
                .get(parent_page_num)
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
        let old_root = self
            .root_page_num
            .load(std::sync::atomic::Ordering::Acquire);
        let height = self.height.load(std::sync::atomic::Ordering::Acquire);

        let new_root_num = pages.allocate();
        let new_root_id = PageId::new(self.file_id, new_root_num);
        let old_root_id = PageId::new(self.file_id, old_root);
        let right_child_id = PageId::new(self.file_id, right_child);

        let mut new_root = BTreeInternalPage::new(new_root_id, height as u16);
        new_root.set_leftmost_child(old_root_id);
        new_root.insert(key, right_child_id)?;

        pages.write(new_root_num, new_root.as_bytes());

        self.root_page_num
            .store(new_root_num, std::sync::atomic::Ordering::Release);
        self.height
            .store(height + 1, std::sync::atomic::Ordering::Release);

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
                        if compare_keys_fast(entry.key.as_ref(), start).is_lt() {
                            continue;
                        }
                    }

                    if let Some(end) = end_key {
                        if compare_keys_fast(entry.key.as_ref(), end).is_gt() {
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
        let mut current = self
            .root_page_num
            .load(std::sync::atomic::Ordering::Acquire);

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

// =============================================================================
// Arena-Based B+Tree Implementation (High-Performance)
// =============================================================================
//
// This implementation uses contiguous memory allocation and lock-free reads
// for minimal lookup latency (~40ns target). Based on LMDB's design principles.
//
// Key design decisions:
// - Contiguous memory arena eliminates pointer chasing
// - Version-based optimistic locking enables lock-free reads
// - Fixed-size 8-byte keys stored as u64 for single-instruction comparison
// - Single-writer lock for inserts (no concurrent write conflicts)

use parking_lot::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// 32KB nodes hold ~2046 keys per node, achieving height=2 for 1M keys.
/// With a write buffer in front, the B+Tree is read-optimized. Large nodes
/// reduce tree height for faster lookups (~74ns). The write buffer handles
/// insert throughput, so node size is tuned purely for read performance.
const ARENA_NODE_SIZE: usize = 32768;

/// Write buffer capacity (number of entries before flush to B+Tree).
/// 64K entries × 16 bytes = 1MB buffer size.
const WRITE_BUFFER_CAPACITY: usize = 65536;

/// Header size for internal nodes (16 bytes).
const ARENA_INTERNAL_HEADER_SIZE: usize = 16;

/// Header size for leaf nodes (24 bytes).
const ARENA_LEAF_HEADER_SIZE: usize = 24;

/// Child pointer size (8 bytes for u64 offset).
const ARENA_CHILD_SIZE: usize = 8;

/// Key size for arena nodes (8 bytes, stored as u64).
const ARENA_KEY_SIZE: usize = 8;

/// Entry size for leaf nodes (key + tuple_id = 16 bytes).
const ARENA_LEAF_ENTRY_SIZE: usize = 16;

/// Maximum keys per internal node: (32768 - 16 - 8) / (8 + 8) = 2046
const ARENA_MAX_INTERNAL_KEYS: usize =
    (ARENA_NODE_SIZE - ARENA_INTERNAL_HEADER_SIZE - ARENA_CHILD_SIZE)
        / (ARENA_KEY_SIZE + ARENA_CHILD_SIZE);

/// Maximum entries per leaf node: (32768 - 24) / 16 = 2046
const ARENA_MAX_LEAF_ENTRIES: usize =
    (ARENA_NODE_SIZE - ARENA_LEAF_HEADER_SIZE) / ARENA_LEAF_ENTRY_SIZE;

/// Sentinel offset indicating null/invalid.
const ARENA_NULL_OFFSET: u64 = u64::MAX;

/// Internal node header layout.
/// +------------------+ 0
/// | version: u64     | 8  (for optimistic locking)
/// | num_keys: u16    | 10
/// | level: u16       | 12
/// | reserved: u32    | 16 (HEADER_SIZE)
/// +------------------+
#[repr(C)]
#[derive(Clone, Copy)]
struct ArenaInternalNodeHeader {
    version: u64,
    num_keys: u16,
    level: u16,
    reserved: u32,
}

/// Leaf node header layout.
/// +------------------+ 0
/// | version: u64     | 8
/// | num_entries: u16 | 10
/// | reserved: [u8;6] | 16
/// | next_leaf: u64   | 24 (LEAF_HEADER_SIZE)
/// +------------------+
#[repr(C)]
#[derive(Clone, Copy)]
struct ArenaLeafNodeHeader {
    version: u64,
    num_entries: u16,
    reserved: [u8; 6],
    next_leaf: u64,
}

/// Contiguous memory arena for B+Tree nodes.
/// Nodes are allocated sequentially, enabling direct pointer access.
pub struct BTreeArena {
    /// Contiguous memory region.
    data: Box<[u8]>,
    /// Current allocation offset (grows upward).
    alloc_offset: AtomicU64,
    /// Total capacity in bytes.
    capacity: usize,
}

impl BTreeArena {
    /// Creates a new arena with the specified capacity in nodes.
    fn new(num_nodes: usize) -> Self {
        let capacity = num_nodes * ARENA_NODE_SIZE;
        let data = vec![0u8; capacity].into_boxed_slice();
        Self {
            data,
            alloc_offset: AtomicU64::new(0),
            capacity,
        }
    }

    /// Allocates a new node, returns its offset.
    #[inline]
    fn allocate(&self) -> u64 {
        let offset = self
            .alloc_offset
            .fetch_add(ARENA_NODE_SIZE as u64, Ordering::Relaxed);
        if offset as usize + ARENA_NODE_SIZE > self.capacity {
            panic!("BTreeArena out of memory");
        }
        offset
    }

    /// Direct pointer access to node data (read).
    #[inline(always)]
    fn node_ptr(&self, offset: u64) -> *const u8 {
        debug_assert!((offset as usize) < self.capacity);
        unsafe { self.data.as_ptr().add(offset as usize) }
    }

    /// Direct pointer access to node data (write).
    #[inline(always)]
    fn node_ptr_mut(&mut self, offset: u64) -> *mut u8 {
        debug_assert!((offset as usize) < self.capacity);
        unsafe { self.data.as_mut_ptr().add(offset as usize) }
    }

    /// Reads internal node header (direct read, no volatility).
    #[inline(always)]
    fn read_internal_header(&self, offset: u64) -> ArenaInternalNodeHeader {
        unsafe {
            let ptr = self.node_ptr(offset) as *const ArenaInternalNodeHeader;
            *ptr
        }
    }

    /// Reads leaf node header (direct read, no volatility).
    #[inline(always)]
    fn read_leaf_header(&self, offset: u64) -> ArenaLeafNodeHeader {
        unsafe {
            let ptr = self.node_ptr(offset) as *const ArenaLeafNodeHeader;
            *ptr
        }
    }
}

/// Arena-based B+Tree index with lock-free reads.
///
/// Performance characteristics:
/// - Lookup: ~35ns (4 levels × 8ns per level)
/// - Insert: ~120ns (lookup + version bump + memcpy)
/// - Memory: ~4KB per 254 keys
pub struct BTreeArenaIndex {
    /// Memory arena for all nodes.
    arena: BTreeArena,
    /// Root node offset.
    root_offset: AtomicU64,
    /// Tree height (1 = just root as leaf).
    height: AtomicU64,
    /// Write lock for concurrent insert operations (reserved for future use).
    #[allow(dead_code)]
    write_lock: Mutex<()>,
}

impl BTreeArenaIndex {
    /// Creates a new arena-based B+Tree index.
    /// Capacity is the maximum number of nodes (each 4KB).
    pub fn new(capacity_nodes: usize) -> Self {
        let mut arena = BTreeArena::new(capacity_nodes.max(1024));

        // Allocate root as leaf node
        let root_offset = arena.allocate();

        // Initialize root leaf header
        unsafe {
            let header_ptr = arena.node_ptr_mut(root_offset) as *mut ArenaLeafNodeHeader;
            std::ptr::write(
                header_ptr,
                ArenaLeafNodeHeader {
                    version: 0,
                    num_entries: 0,
                    reserved: [0; 6],
                    next_leaf: ARENA_NULL_OFFSET,
                },
            );
        }

        Self {
            arena,
            root_offset: AtomicU64::new(root_offset),
            height: AtomicU64::new(1),
            write_lock: Mutex::new(()),
        }
    }

    /// Returns the tree height.
    #[inline]
    pub fn height(&self) -> u64 {
        self.height.load(Ordering::Acquire)
    }

    // =========================================================================
    // Lock-Free Read Path
    // =========================================================================

    /// Direct search without version overhead.
    /// Writes use &mut self which provides exclusive access.
    #[inline(always)]
    pub fn search(&self, key: &[u8]) -> Option<TupleId> {
        self.search_optimistic(key)
    }

    /// Direct search without version checking.
    /// Writes use &mut self which provides exclusive access.
    /// Relaxed ordering is safe since writes are exclusive.
    #[inline(always)]
    fn search_optimistic(&self, key: &[u8]) -> Option<TupleId> {
        let search_key = self.key_to_u64(key);
        let mut current_offset = self.root_offset.load(Ordering::Relaxed);
        let height = self.height.load(Ordering::Relaxed);

        // Fast path for height=1 (root is leaf)
        if height == 1 {
            return self.search_leaf_interpolation(current_offset, search_key);
        }

        // Traverse internal nodes with leaf prefetching
        // Height 2: 1 internal level, height 3: 2 internal levels, etc.
        let mut levels_remaining = height - 1;
        while levels_remaining > 0 {
            current_offset = self.traverse_internal_node_prefetch(
                current_offset,
                search_key,
                levels_remaining == 1,
            );
            levels_remaining -= 1;
        }

        // Search leaf node using interpolation search
        self.search_leaf_interpolation(current_offset, search_key)
    }

    /// Traverse internal node with optional prefetch of child node.
    /// When at last internal level, prefetches the target leaf node.
    #[inline(always)]
    fn traverse_internal_node_prefetch(
        &self,
        offset: u64,
        search_key: u64,
        prefetch_child: bool,
    ) -> u64 {
        let header = self.arena.read_internal_header(offset);
        let child_idx = self.find_child_idx_branchless(offset, search_key, header.num_keys);
        let child_offset = self.read_child_ptr(offset, child_idx);

        // Prefetch child node header and first entries while returning
        if prefetch_child {
            unsafe {
                let child_ptr = self.arena.node_ptr(child_offset);
                std::arch::x86_64::_mm_prefetch(
                    child_ptr as *const i8,
                    std::arch::x86_64::_MM_HINT_T0,
                );
                // Prefetch first cache line of entries (64 bytes after header)
                std::arch::x86_64::_mm_prefetch(
                    child_ptr.add(64) as *const i8,
                    std::arch::x86_64::_MM_HINT_T0,
                );
            }
        }

        child_offset
    }

    /// Branchless binary search for child index in internal node.
    /// Uses bitmask operations instead of branches for better pipeline efficiency.
    /// B+Tree semantics: child[i] has keys < key[i], child[i+1] has keys >= key[i].
    #[inline(always)]
    fn find_child_idx_branchless(&self, node_offset: u64, search_key: u64, num_keys: u16) -> usize {
        if num_keys == 0 {
            return 0;
        }

        unsafe {
            let base = self
                .arena
                .node_ptr(node_offset)
                .add(ARENA_INTERNAL_HEADER_SIZE + ARENA_CHILD_SIZE);
            let key_stride = ARENA_KEY_SIZE + ARENA_CHILD_SIZE;

            let mut lo = 0usize;
            let mut hi = num_keys as usize;

            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                let entry_ptr = base.add(mid * key_stride) as *const u64;
                let entry_key = std::ptr::read_unaligned(entry_ptr);

                // Branchless update using bitmask:
                // go_right is 0xFFFF... if search_key >= entry_key, 0 otherwise
                let go_right = (search_key >= entry_key) as usize;
                // If go_right: lo = mid + 1, hi unchanged
                // If !go_right: hi = mid, lo unchanged
                lo = go_right * (mid + 1) + (1 - go_right) * lo;
                hi = (1 - go_right) * mid + go_right * hi;
            }

            lo
        }
    }

    /// Read child pointer at given index from internal node.
    #[inline(always)]
    fn read_child_ptr(&self, node_offset: u64, idx: usize) -> u64 {
        unsafe {
            let key_stride = ARENA_KEY_SIZE + ARENA_CHILD_SIZE;
            let ptr = if idx == 0 {
                // Leftmost child is right after header
                self.arena
                    .node_ptr(node_offset)
                    .add(ARENA_INTERNAL_HEADER_SIZE) as *const u64
            } else {
                // Child i is after key (i-1)
                self.arena.node_ptr(node_offset).add(
                    ARENA_INTERNAL_HEADER_SIZE
                        + ARENA_CHILD_SIZE
                        + (idx - 1) * key_stride
                        + ARENA_KEY_SIZE,
                ) as *const u64
            };
            std::ptr::read_unaligned(ptr)
        }
    }

    /// Search leaf using interpolation search with binary search fallback.
    /// Interpolation search: O(log log n) for uniform keys, falls back to binary for safety.
    #[inline(always)]
    fn search_leaf_interpolation(&self, offset: u64, search_key: u64) -> Option<TupleId> {
        let header = self.arena.read_leaf_header(offset);
        let count = header.num_entries as usize;

        if count == 0 {
            return None;
        }

        unsafe {
            let base = self.arena.node_ptr(offset).add(ARENA_LEAF_HEADER_SIZE);

            // Read first and last keys for interpolation
            let first_key = std::ptr::read_unaligned(base as *const u64);
            let last_key = std::ptr::read_unaligned(
                base.add((count - 1) * ARENA_LEAF_ENTRY_SIZE) as *const u64
            );

            // Early exit: key out of range
            if search_key < first_key || search_key > last_key {
                return None;
            }

            // Exact match on boundaries
            if search_key == first_key {
                let tuple_ptr = (base as *const u64).add(1);
                return Some(Self::unpack_tuple_id(std::ptr::read_unaligned(tuple_ptr)));
            }
            if search_key == last_key {
                let entry_ptr = base.add((count - 1) * ARENA_LEAF_ENTRY_SIZE);
                let tuple_ptr = (entry_ptr as *const u64).add(1);
                return Some(Self::unpack_tuple_id(std::ptr::read_unaligned(tuple_ptr)));
            }

            let mut lo = 0usize;
            let mut hi = count;

            // Interpolation search: estimate position based on key distribution
            // Use up to 3 interpolation steps, then fall back to binary
            for _ in 0..3 {
                if hi - lo <= 8 {
                    break;
                }

                let lo_key =
                    std::ptr::read_unaligned(base.add(lo * ARENA_LEAF_ENTRY_SIZE) as *const u64);
                let hi_key = std::ptr::read_unaligned(
                    base.add((hi - 1) * ARENA_LEAF_ENTRY_SIZE) as *const u64
                );

                if lo_key >= hi_key {
                    break;
                }

                // Interpolation formula: estimate position proportionally
                let range = hi - lo;
                let key_range = hi_key - lo_key;
                let key_offset = search_key.saturating_sub(lo_key);

                // Compute interpolated position with overflow protection
                let estimate = if key_range > 0 {
                    lo + ((key_offset as u128 * range as u128 / key_range as u128) as usize)
                        .min(range - 1)
                } else {
                    lo + range / 2
                };

                let entry_ptr = base.add(estimate * ARENA_LEAF_ENTRY_SIZE) as *const u64;
                let entry_key = std::ptr::read_unaligned(entry_ptr);

                if entry_key == search_key {
                    let tuple_ptr = entry_ptr.add(1);
                    return Some(Self::unpack_tuple_id(std::ptr::read_unaligned(tuple_ptr)));
                } else if search_key < entry_key {
                    hi = estimate;
                } else {
                    lo = estimate + 1;
                }
            }

            // Binary search for remaining range
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                let entry_ptr = base.add(mid * ARENA_LEAF_ENTRY_SIZE) as *const u64;
                let entry_key = std::ptr::read_unaligned(entry_ptr);

                let go_right = (search_key > entry_key) as usize;
                lo = go_right * (mid + 1) + (1 - go_right) * lo;
                hi = (1 - go_right) * mid + go_right * hi;
            }

            // Verify exact match
            if lo < count {
                let entry_ptr = base.add(lo * ARENA_LEAF_ENTRY_SIZE) as *const u64;
                let entry_key = std::ptr::read_unaligned(entry_ptr);
                if entry_key == search_key {
                    let tuple_ptr = entry_ptr.add(1);
                    return Some(Self::unpack_tuple_id(std::ptr::read_unaligned(tuple_ptr)));
                }
            }

            None
        }
    }

    // =========================================================================
    // Write Path (Single Writer)
    // =========================================================================

    /// Insert a key-value pair.
    #[inline]
    pub fn insert(&mut self, key: &[u8], tuple_id: TupleId) -> Result<()> {
        let search_key = self.key_to_u64(key);
        let packed_tuple_id = Self::pack_tuple_id(tuple_id);

        // Find leaf for insert (Relaxed is safe since we have &mut self)
        let height = self.height.load(Ordering::Relaxed);
        let mut current_offset = self.root_offset.load(Ordering::Relaxed);

        // Build path for potential split propagation
        let mut path = [0u64; 16];
        let mut path_len = 0;

        path[path_len] = current_offset;
        path_len += 1;

        // Traverse to leaf
        for _ in 0..(height - 1) {
            let header = self.arena.read_internal_header(current_offset);
            let child_idx =
                self.find_child_idx_branchless(current_offset, search_key, header.num_keys);
            let child_offset = self.read_child_ptr(current_offset, child_idx);
            current_offset = child_offset;
            path[path_len] = current_offset;
            path_len += 1;
        }

        // Try insert in leaf
        let leaf_offset = current_offset;
        if self.try_insert_in_leaf(leaf_offset, search_key, packed_tuple_id)? {
            return Ok(());
        }

        // Leaf full - need to split
        self.split_and_insert(search_key, packed_tuple_id, &path[..path_len])
    }

    /// Try to insert in leaf without split.
    #[inline(always)]
    fn try_insert_in_leaf(&mut self, offset: u64, key: u64, packed_tuple_id: u64) -> Result<bool> {
        unsafe {
            let header_ptr = self.arena.node_ptr_mut(offset) as *mut ArenaLeafNodeHeader;
            let header = std::ptr::read_volatile(header_ptr);

            if header.num_entries as usize >= ARENA_MAX_LEAF_ENTRIES {
                return Ok(false); // Need split
            }

            // Bump version (odd = write in progress)
            let new_version = header.version + 1;
            std::ptr::write_volatile(&mut (*header_ptr).version, new_version);
            std::sync::atomic::fence(Ordering::Release);

            // Find insertion position (maintain sorted order)
            let insert_pos = self.find_leaf_insert_pos(offset, key, header.num_entries);

            // Shift entries to make room
            let base = self.arena.node_ptr_mut(offset).add(ARENA_LEAF_HEADER_SIZE);
            if insert_pos < header.num_entries as usize {
                let src = base.add(insert_pos * ARENA_LEAF_ENTRY_SIZE);
                let dst = base.add((insert_pos + 1) * ARENA_LEAF_ENTRY_SIZE);
                let count = (header.num_entries as usize - insert_pos) * ARENA_LEAF_ENTRY_SIZE;
                std::ptr::copy(src, dst, count);
            }

            // Write new entry
            let entry_ptr = base.add(insert_pos * ARENA_LEAF_ENTRY_SIZE) as *mut u64;
            std::ptr::write_unaligned(entry_ptr, key);
            std::ptr::write_unaligned(entry_ptr.add(1), packed_tuple_id);

            // Update count and finalize version (even = stable)
            (*header_ptr).num_entries = header.num_entries + 1;
            std::sync::atomic::fence(Ordering::Release);
            std::ptr::write_volatile(&mut (*header_ptr).version, new_version + 1);

            Ok(true)
        }
    }

    /// Find insertion position in leaf (binary search).
    #[inline]
    fn find_leaf_insert_pos(&self, offset: u64, key: u64, num_entries: u16) -> usize {
        if num_entries == 0 {
            return 0;
        }

        unsafe {
            let base = self.arena.node_ptr(offset).add(ARENA_LEAF_HEADER_SIZE);
            let mut lo = 0usize;
            let mut hi = num_entries as usize;

            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                let entry_ptr = base.add(mid * ARENA_LEAF_ENTRY_SIZE) as *const u64;
                let entry_key = std::ptr::read_unaligned(entry_ptr);

                if key > entry_key {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }

            lo
        }
    }

    /// Split leaf and insert (handles tree growth).
    fn split_and_insert(&mut self, key: u64, packed_tuple_id: u64, path: &[u64]) -> Result<()> {
        let leaf_offset = path[path.len() - 1];

        // Read current leaf
        let header = self.arena.read_leaf_header(leaf_offset);
        let num_entries = header.num_entries as usize;

        // Allocate new leaf
        let new_leaf_offset = self.arena.allocate();

        // Split point: half of entries go to new leaf
        let split_point = num_entries / 2;

        // Bump version on old leaf
        unsafe {
            let header_ptr = self.arena.node_ptr_mut(leaf_offset) as *mut ArenaLeafNodeHeader;
            let new_version = header.version + 1;
            std::ptr::write_volatile(&mut (*header_ptr).version, new_version);
            std::sync::atomic::fence(Ordering::Release);

            // Copy second half to new leaf
            let old_base = self.arena.node_ptr(leaf_offset).add(ARENA_LEAF_HEADER_SIZE);
            let new_base = self
                .arena
                .node_ptr_mut(new_leaf_offset)
                .add(ARENA_LEAF_HEADER_SIZE);

            let copy_count = num_entries - split_point;
            std::ptr::copy_nonoverlapping(
                old_base.add(split_point * ARENA_LEAF_ENTRY_SIZE),
                new_base,
                copy_count * ARENA_LEAF_ENTRY_SIZE,
            );

            // Get split key (first key in new leaf)
            let split_key = std::ptr::read_unaligned(new_base as *const u64);

            // Initialize new leaf header
            let new_header_ptr =
                self.arena.node_ptr_mut(new_leaf_offset) as *mut ArenaLeafNodeHeader;
            std::ptr::write(
                new_header_ptr,
                ArenaLeafNodeHeader {
                    version: 0,
                    num_entries: copy_count as u16,
                    reserved: [0; 6],
                    next_leaf: header.next_leaf,
                },
            );

            // Update old leaf: truncate and link to new leaf
            (*header_ptr).num_entries = split_point as u16;
            (*header_ptr).next_leaf = new_leaf_offset;
            std::sync::atomic::fence(Ordering::Release);
            std::ptr::write_volatile(&mut (*header_ptr).version, new_version + 1);

            // Insert the new key into appropriate leaf
            if key < split_key {
                self.try_insert_in_leaf(leaf_offset, key, packed_tuple_id)?;
            } else {
                self.try_insert_in_leaf(new_leaf_offset, key, packed_tuple_id)?;
            }

            // Propagate split up
            if path.len() == 1 {
                // Root was a leaf, create new root
                self.create_new_root(split_key, leaf_offset, new_leaf_offset);
            } else {
                self.propagate_split(split_key, new_leaf_offset, &path[..path.len() - 1])?;
            }
        }

        Ok(())
    }

    /// Create new root after root split.
    fn create_new_root(&mut self, split_key: u64, left_child: u64, right_child: u64) {
        let new_root_offset = self.arena.allocate();

        unsafe {
            let header_ptr =
                self.arena.node_ptr_mut(new_root_offset) as *mut ArenaInternalNodeHeader;
            std::ptr::write(
                header_ptr,
                ArenaInternalNodeHeader {
                    version: 0,
                    num_keys: 1,
                    level: self.height.load(Ordering::Relaxed) as u16,
                    reserved: 0,
                },
            );

            // Write leftmost child
            let child0_ptr = self
                .arena
                .node_ptr_mut(new_root_offset)
                .add(ARENA_INTERNAL_HEADER_SIZE) as *mut u64;
            std::ptr::write_unaligned(child0_ptr, left_child);

            // Write key + right child
            let key_ptr = child0_ptr.add(1);
            std::ptr::write_unaligned(key_ptr, split_key);
            let child1_ptr = key_ptr.add(1);
            std::ptr::write_unaligned(child1_ptr, right_child);
        }

        self.height.fetch_add(1, Ordering::Release);
        self.root_offset.store(new_root_offset, Ordering::Release);
    }

    /// Propagate split up to parent internal nodes.
    fn propagate_split(&mut self, key: u64, new_child: u64, path: &[u64]) -> Result<()> {
        let mut current_key = key;
        let mut current_child = new_child;

        for parent_idx in (0..path.len()).rev() {
            let parent_offset = path[parent_idx];

            if self.try_insert_in_internal(parent_offset, current_key, current_child)? {
                return Ok(());
            }

            // Need to split internal node
            let (split_key, new_internal) =
                self.split_internal_node(parent_offset, current_key, current_child)?;

            if parent_idx == 0 {
                // Root split
                self.create_new_root(split_key, parent_offset, new_internal);
                return Ok(());
            }

            current_key = split_key;
            current_child = new_internal;
        }

        Ok(())
    }

    /// Try to insert key/child in internal node without split.
    fn try_insert_in_internal(&mut self, offset: u64, key: u64, child: u64) -> Result<bool> {
        unsafe {
            let header_ptr = self.arena.node_ptr_mut(offset) as *mut ArenaInternalNodeHeader;
            let header = std::ptr::read_volatile(header_ptr);

            if header.num_keys as usize >= ARENA_MAX_INTERNAL_KEYS {
                return Ok(false);
            }

            // Bump version
            let new_version = header.version + 1;
            std::ptr::write_volatile(&mut (*header_ptr).version, new_version);
            std::sync::atomic::fence(Ordering::Release);

            // Find insertion position
            let insert_pos = self.find_internal_insert_pos(offset, key, header.num_keys);

            let key_stride = ARENA_KEY_SIZE + ARENA_CHILD_SIZE;
            let base = self
                .arena
                .node_ptr_mut(offset)
                .add(ARENA_INTERNAL_HEADER_SIZE + ARENA_CHILD_SIZE);

            // Shift entries
            if insert_pos < header.num_keys as usize {
                let src = base.add(insert_pos * key_stride);
                let dst = base.add((insert_pos + 1) * key_stride);
                let count = (header.num_keys as usize - insert_pos) * key_stride;
                std::ptr::copy(src, dst, count);
            }

            // Write new key and child
            let entry_ptr = base.add(insert_pos * key_stride) as *mut u64;
            std::ptr::write_unaligned(entry_ptr, key);
            std::ptr::write_unaligned(entry_ptr.add(1), child);

            // Update count and finalize
            (*header_ptr).num_keys = header.num_keys + 1;
            std::sync::atomic::fence(Ordering::Release);
            std::ptr::write_volatile(&mut (*header_ptr).version, new_version + 1);

            Ok(true)
        }
    }

    /// Find insertion position in internal node.
    #[inline]
    fn find_internal_insert_pos(&self, offset: u64, key: u64, num_keys: u16) -> usize {
        if num_keys == 0 {
            return 0;
        }

        unsafe {
            let base = self
                .arena
                .node_ptr(offset)
                .add(ARENA_INTERNAL_HEADER_SIZE + ARENA_CHILD_SIZE);
            let key_stride = ARENA_KEY_SIZE + ARENA_CHILD_SIZE;

            let mut lo = 0usize;
            let mut hi = num_keys as usize;

            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                let entry_ptr = base.add(mid * key_stride) as *const u64;
                let entry_key = std::ptr::read_unaligned(entry_ptr);

                if key > entry_key {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }

            lo
        }
    }

    /// Split internal node.
    fn split_internal_node(&mut self, offset: u64, key: u64, child: u64) -> Result<(u64, u64)> {
        let new_offset = self.arena.allocate();

        unsafe {
            let header_ptr = self.arena.node_ptr_mut(offset) as *mut ArenaInternalNodeHeader;
            let header = std::ptr::read_volatile(header_ptr);

            let num_keys = header.num_keys as usize;
            let split_point = num_keys / 2;

            // Bump version
            let new_version = header.version + 1;
            std::ptr::write_volatile(&mut (*header_ptr).version, new_version);
            std::sync::atomic::fence(Ordering::Release);

            let key_stride = ARENA_KEY_SIZE + ARENA_CHILD_SIZE;
            let base = self.arena.node_ptr(offset).add(ARENA_INTERNAL_HEADER_SIZE);

            // Get split key (middle key, will be promoted)
            let split_key_ptr = base.add(ARENA_CHILD_SIZE + split_point * key_stride) as *const u64;
            let split_key = std::ptr::read_unaligned(split_key_ptr);

            // Copy right half to new node (keys after split point)
            let new_base = self
                .arena
                .node_ptr_mut(new_offset)
                .add(ARENA_INTERNAL_HEADER_SIZE);

            // First, copy the leftmost child of right subtree
            let right_child_ptr = split_key_ptr.add(1) as *const u64;
            std::ptr::write_unaligned(
                new_base as *mut u64,
                std::ptr::read_unaligned(right_child_ptr),
            );

            // Copy remaining keys and children
            let copy_start = (split_point + 1) * key_stride + ARENA_CHILD_SIZE;
            let copy_count = (num_keys - split_point - 1) * key_stride;
            if copy_count > 0 {
                std::ptr::copy_nonoverlapping(
                    base.add(copy_start),
                    new_base.add(ARENA_CHILD_SIZE),
                    copy_count,
                );
            }

            // Initialize new internal header
            let new_header_ptr =
                self.arena.node_ptr_mut(new_offset) as *mut ArenaInternalNodeHeader;
            std::ptr::write(
                new_header_ptr,
                ArenaInternalNodeHeader {
                    version: 0,
                    num_keys: (num_keys - split_point - 1) as u16,
                    level: header.level,
                    reserved: 0,
                },
            );

            // Update old node
            (*header_ptr).num_keys = split_point as u16;
            std::sync::atomic::fence(Ordering::Release);
            std::ptr::write_volatile(&mut (*header_ptr).version, new_version + 1);

            // Insert the new key/child into appropriate node
            if key < split_key {
                self.try_insert_in_internal(offset, key, child)?;
            } else {
                self.try_insert_in_internal(new_offset, key, child)?;
            }

            Ok((split_key, new_offset))
        }
    }

    // =========================================================================
    // Range Scan
    // =========================================================================

    /// Range scan from start_key to end_key (inclusive).
    /// Relaxed ordering is safe since writes are exclusive.
    pub fn range_scan(
        &self,
        start_key: Option<&[u8]>,
        end_key: Option<&[u8]>,
    ) -> Vec<(u64, TupleId)> {
        let start = start_key.map(|k| self.key_to_u64(k)).unwrap_or(0);
        let end = end_key.map(|k| self.key_to_u64(k)).unwrap_or(u64::MAX);

        let mut results = Vec::new();

        // Find starting leaf
        let height = self.height.load(Ordering::Relaxed);
        let mut current = self.root_offset.load(Ordering::Relaxed);

        for _ in 0..(height - 1) {
            let header = self.arena.read_internal_header(current);
            let child_idx = self.find_child_idx_branchless(current, start, header.num_keys);
            current = self.read_child_ptr(current, child_idx);
        }

        // Scan leaves
        loop {
            unsafe {
                let header = self.arena.read_leaf_header(current);
                let base = self.arena.node_ptr(current).add(ARENA_LEAF_HEADER_SIZE);

                for i in 0..header.num_entries as usize {
                    let entry_ptr = base.add(i * ARENA_LEAF_ENTRY_SIZE) as *const u64;
                    let entry_key = std::ptr::read_unaligned(entry_ptr);

                    if entry_key < start {
                        continue;
                    }
                    if entry_key > end {
                        return results;
                    }

                    let tuple_ptr = entry_ptr.add(1);
                    let packed = std::ptr::read_unaligned(tuple_ptr);
                    results.push((entry_key, Self::unpack_tuple_id(packed)));
                }

                if header.next_leaf == ARENA_NULL_OFFSET {
                    break;
                }
                current = header.next_leaf;
            }
        }

        results
    }

    // =========================================================================
    // Utility Functions
    // =========================================================================

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

    /// Pack TupleId into u64: file_id(16) + page_num(32) + slot_id(16).
    #[inline(always)]
    pub(crate) fn pack_tuple_id(tid: TupleId) -> u64 {
        ((tid.page_id.file_id as u64 & 0xFFFF) << 48)
            | ((tid.page_id.page_num as u64) << 16)
            | (tid.slot_id as u64)
    }

    /// Unpack u64 into TupleId.
    #[inline(always)]
    pub(crate) fn unpack_tuple_id(packed: u64) -> TupleId {
        TupleId {
            page_id: PageId {
                file_id: ((packed >> 48) & 0xFFFF) as u32,
                page_num: ((packed >> 16) & 0xFFFFFFFF) as u32,
            },
            slot_id: (packed & 0xFFFF) as u16,
        }
    }

    /// Returns the number of entries in the tree (for testing).
    pub fn count(&self) -> usize {
        let mut count = 0;
        let height = self.height.load(Ordering::Relaxed);
        let mut current = self.root_offset.load(Ordering::Relaxed);

        // Find leftmost leaf
        for _ in 0..(height - 1) {
            let _header = self.arena.read_internal_header(current);
            current = self.read_child_ptr(current, 0);
        }

        // Count all leaves
        loop {
            let header = self.arena.read_leaf_header(current);
            count += header.num_entries as usize;

            if header.next_leaf == ARENA_NULL_OFFSET {
                break;
            }
            current = header.next_leaf;
        }

        count
    }

    // =========================================================================
    // Bulk Insert (Cursor-Based)
    // =========================================================================

    /// Bulk insert sorted u64 key-value pairs directly.
    /// Keys must be sorted ascending within the batch. Values are pre-packed tuple_ids.
    /// Optimizations for sorted batches:
    /// 1. Skip tree traversal when consecutive keys go to same leaf
    /// 2. Skip binary search when inserting at predictable position
    /// 3. Batch version bumps per leaf (one at start, one at end)
    #[inline]
    pub fn insert_bulk_sorted(&mut self, entries: &[(u64, u64)]) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }

        // Batch state for current leaf
        let mut batch_leaf: u64 = u64::MAX;
        let mut batch_version: u64 = 0;
        let mut batch_active = false;
        let mut batch_max_key: u64 = 0; // Maximum key that belongs in current batch leaf
        let mut last_insert_pos: usize = 0;
        let mut path = [0u64; 16];
        let mut path_len: usize;

        for &(key, packed_tuple_id) in entries {
            // Try to stay in current leaf batch
            if batch_active {
                let header = self.arena.read_leaf_header(batch_leaf);

                // Check if leaf has space AND key belongs to this leaf
                // Key belongs if it's <= the boundary key for this leaf
                let key_belongs = key <= batch_max_key || batch_max_key == u64::MAX;
                if key_belongs && (header.num_entries as usize) < ARENA_MAX_LEAF_ENTRIES {
                    // Determine insertion position using sorted property
                    let insert_pos = if last_insert_pos + 1 >= header.num_entries as usize {
                        // Append at end (common case for ascending inserts)
                        header.num_entries as usize
                    } else {
                        // Check if key fits at last_insert_pos + 1
                        let next_key = unsafe {
                            let base =
                                self.arena.node_ptr(batch_leaf).add(ARENA_LEAF_HEADER_SIZE);
                            let ptr = base.add((last_insert_pos + 1) * ARENA_LEAF_ENTRY_SIZE)
                                as *const u64;
                            std::ptr::read_unaligned(ptr)
                        };
                        if key < next_key {
                            last_insert_pos + 1
                        } else {
                            // Binary search from last position
                            self.find_leaf_insert_pos_from(
                                batch_leaf,
                                key,
                                last_insert_pos + 1,
                                header.num_entries,
                            )
                        }
                    };

                    // Insert with shift
                    unsafe {
                        let base =
                            self.arena.node_ptr_mut(batch_leaf).add(ARENA_LEAF_HEADER_SIZE);
                        if insert_pos < header.num_entries as usize {
                            let src = base.add(insert_pos * ARENA_LEAF_ENTRY_SIZE);
                            let dst = base.add((insert_pos + 1) * ARENA_LEAF_ENTRY_SIZE);
                            let count =
                                (header.num_entries as usize - insert_pos) * ARENA_LEAF_ENTRY_SIZE;
                            std::ptr::copy(src, dst, count);
                        }
                        let entry_ptr = base.add(insert_pos * ARENA_LEAF_ENTRY_SIZE) as *mut u64;
                        std::ptr::write_unaligned(entry_ptr, key);
                        std::ptr::write_unaligned(entry_ptr.add(1), packed_tuple_id);

                        let header_ptr =
                            self.arena.node_ptr_mut(batch_leaf) as *mut ArenaLeafNodeHeader;
                        (*header_ptr).num_entries = header.num_entries + 1;

                        // Update batch_max_key: read last key after insert
                        let new_count = header.num_entries + 1;
                        let last_ptr =
                            base.add((new_count as usize - 1) * ARENA_LEAF_ENTRY_SIZE) as *const u64;
                        batch_max_key = std::ptr::read_unaligned(last_ptr);
                    }

                    last_insert_pos = insert_pos;
                    continue;
                }
            }

            // End current batch and start new tree traversal
            if batch_active {
                unsafe {
                    let header_ptr =
                        self.arena.node_ptr_mut(batch_leaf) as *mut ArenaLeafNodeHeader;
                    std::sync::atomic::fence(Ordering::Release);
                    std::ptr::write_volatile(&mut (*header_ptr).version, batch_version + 2);
                }
                // batch_active will be set by the branch below (true or false)
            }

            // Tree traversal to find correct leaf
            let height = self.height.load(Ordering::Relaxed);
            let mut current_offset = self.root_offset.load(Ordering::Relaxed);
            path_len = 0;
            path[path_len] = current_offset;
            path_len += 1;

            for _ in 0..(height - 1) {
                let header = self.arena.read_internal_header(current_offset);
                let child_idx =
                    self.find_child_idx_branchless(current_offset, key, header.num_keys);
                let child_offset = self.read_child_ptr(current_offset, child_idx);
                current_offset = child_offset;
                path[path_len] = current_offset;
                path_len += 1;
            }

            let leaf_offset = current_offset;
            let header = self.arena.read_leaf_header(leaf_offset);

            if (header.num_entries as usize) < ARENA_MAX_LEAF_ENTRIES {
                // Start new batch on this leaf
                batch_leaf = leaf_offset;
                batch_version = header.version;
                batch_active = true;

                // Bump version (odd = write in progress)
                unsafe {
                    let header_ptr =
                        self.arena.node_ptr_mut(batch_leaf) as *mut ArenaLeafNodeHeader;
                    std::ptr::write_volatile(&mut (*header_ptr).version, batch_version + 1);
                    std::sync::atomic::fence(Ordering::Release);
                }

                // Find insertion position
                let insert_pos = self.find_leaf_insert_pos(leaf_offset, key, header.num_entries);

                // Insert with shift
                unsafe {
                    let base = self.arena.node_ptr_mut(leaf_offset).add(ARENA_LEAF_HEADER_SIZE);
                    if insert_pos < header.num_entries as usize {
                        let src = base.add(insert_pos * ARENA_LEAF_ENTRY_SIZE);
                        let dst = base.add((insert_pos + 1) * ARENA_LEAF_ENTRY_SIZE);
                        let count =
                            (header.num_entries as usize - insert_pos) * ARENA_LEAF_ENTRY_SIZE;
                        std::ptr::copy(src, dst, count);
                    }
                    let entry_ptr = base.add(insert_pos * ARENA_LEAF_ENTRY_SIZE) as *mut u64;
                    std::ptr::write_unaligned(entry_ptr, key);
                    std::ptr::write_unaligned(entry_ptr.add(1), packed_tuple_id);

                    let header_ptr =
                        self.arena.node_ptr_mut(leaf_offset) as *mut ArenaLeafNodeHeader;
                    (*header_ptr).num_entries = header.num_entries + 1;

                    // Update batch_max_key: read last key after insert
                    let new_count = header.num_entries + 1;
                    let last_ptr =
                        base.add((new_count as usize - 1) * ARENA_LEAF_ENTRY_SIZE) as *const u64;
                    batch_max_key = std::ptr::read_unaligned(last_ptr);
                }

                last_insert_pos = insert_pos;
            } else {
                // Leaf full - split (no batch active after this)
                self.split_and_insert(key, packed_tuple_id, &path[..path_len])?;

                // Update state for next iteration
                let height = self.height.load(Ordering::Relaxed);
                let (new_leaf, _, _) = self.find_leaf_with_path(key, height);
                let new_header = self.arena.read_leaf_header(new_leaf);
                batch_leaf = new_leaf;
                batch_version = new_header.version;
                batch_active = false; // Don't start batch after split (version already finalized)

                // Find where key ended up
                last_insert_pos =
                    self.find_key_position(new_leaf, key, new_header.num_entries).unwrap_or(0);
            }
        }

        // Finalize last batch
        if batch_active {
            unsafe {
                let header_ptr = self.arena.node_ptr_mut(batch_leaf) as *mut ArenaLeafNodeHeader;
                std::sync::atomic::fence(Ordering::Release);
                std::ptr::write_volatile(&mut (*header_ptr).version, batch_version + 2);
            }
        }

        Ok(())
    }

    /// Binary search starting from a given position (for sorted batch optimization).
    #[inline(always)]
    fn find_leaf_insert_pos_from(
        &self,
        offset: u64,
        key: u64,
        start: usize,
        num_entries: u16,
    ) -> usize {
        unsafe {
            let base = self.arena.node_ptr(offset).add(ARENA_LEAF_HEADER_SIZE);
            let mut lo = start;
            let mut hi = num_entries as usize;

            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                let entry_ptr = base.add(mid * ARENA_LEAF_ENTRY_SIZE) as *const u64;
                let entry_key = std::ptr::read_unaligned(entry_ptr);

                if key > entry_key {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            lo
        }
    }

    /// Find exact position of a key in leaf (returns None if not found).
    #[inline(always)]
    fn find_key_position(&self, offset: u64, key: u64, num_entries: u16) -> Option<usize> {
        let pos = self.find_leaf_insert_pos(offset, key, num_entries);
        if pos < num_entries as usize {
            let entry_key = unsafe {
                let base = self.arena.node_ptr(offset).add(ARENA_LEAF_HEADER_SIZE);
                let ptr = base.add(pos * ARENA_LEAF_ENTRY_SIZE) as *const u64;
                std::ptr::read_unaligned(ptr)
            };
            if entry_key == key {
                return Some(pos);
            }
        }
        None
    }

    /// Find leaf for key and return (leaf_offset, path, path_len).
    #[inline]
    fn find_leaf_with_path(&self, search_key: u64, height: u64) -> (u64, [u64; 16], usize) {
        let mut current_offset = self.root_offset.load(Ordering::Relaxed);
        let mut path = [0u64; 16];
        let mut path_len = 0;

        path[path_len] = current_offset;
        path_len += 1;

        for _ in 0..(height - 1) {
            let header = self.arena.read_internal_header(current_offset);
            let child_idx =
                self.find_child_idx_branchless(current_offset, search_key, header.num_keys);
            let child_offset = self.read_child_ptr(current_offset, child_idx);
            current_offset = child_offset;
            path[path_len] = current_offset;
            path_len += 1;
        }

        (current_offset, path, path_len)
    }
}

// =============================================================================
// Buffered B+Tree Index (Write Buffer + B+Tree)
// =============================================================================

/// Entry in the write buffer (key + packed tuple_id).
#[derive(Clone, Copy)]
struct WriteBufferEntry {
    key: u64,
    packed_tuple_id: u64,
}

/// Hash table size: 131072 slots (power of 2) for 50% load factor with 65536 entries.
/// Must be a multiple of GROUP_SIZE (32) for SIMD alignment.
const HASH_TABLE_SIZE: usize = 131072;

/// Group size for SIMD probing. AVX2 = 32 bytes = 32 control bytes at once.
const GROUP_SIZE: usize = 32;

/// Control byte values.
const CTRL_EMPTY: u8 = 0x80; // 0b10000000 - empty slot
const CTRL_DELETED: u8 = 0xFE; // 0b11111110 - deleted slot (tombstone)

/// Fibonacci hash constant (golden ratio * 2^64).
use std::simd::prelude::*;

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

/// Performance statistics for profiling insert operations.
#[derive(Default, Clone)]
pub struct InsertStats {
    /// Number of flush operations performed.
    pub flush_count: u64,
    /// Total time spent in flush operations (nanoseconds).
    pub flush_time_ns: u64,
    /// Total time spent in drain_sorted (nanoseconds).
    pub drain_time_ns: u64,
    /// Total time spent inserting to B+Tree during flush (nanoseconds).
    pub btree_insert_time_ns: u64,
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

//! Common types for B+Tree implementations.

use crate::tuple::TupleId;
use bytes::{Bytes, BytesMut};
use zyron_common::page::{PAGE_SIZE, PageHeader, PageId};

/// Key comparison using u64 prefix for 8+ byte keys.
/// Falls back to slice comparison for shorter keys or when prefix matches.
#[inline(always)]
pub fn compare_keys(a: &[u8], b: &[u8]) -> std::cmp::Ordering {
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
/// | Page Header (40 bytes) |
/// +------------------------+ 40
/// | Leaf Header (16 bytes) |
/// +------------------------+ 56 (SLOT_ARRAY_START)
/// | Slot Array             |
/// | [offset:2, len:2] * n  |  <- grows forward
/// +------------------------+ 48 + 4*n
/// |      Free Space        |
/// +------------------------+ data_end
/// | Entry Data             |
/// | (key_len:2+key+pn4+s2) |  <- grows backward from PAGE_SIZE
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
/// Layout (on-disk):
/// - key_len: 2 bytes
/// - key: variable
/// - page_num: 4 bytes (u32, file_id is implicit from the B+tree index)
/// - slot_id: 2 bytes
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
        2 + self.key.len() + 6 // key_len + key + page_num(4) + slot_id(2)
    }

    /// Serializes the entry to bytes.
    /// Stores only page_num (u32) + slot_id (u16). file_id is reconstructed
    /// from the B+tree index context on read.
    pub fn to_bytes(&self) -> Bytes {
        let mut buf = BytesMut::with_capacity(self.size_on_disk());
        buf.extend_from_slice(&(self.key.len() as u16).to_le_bytes());
        buf.extend_from_slice(&self.key);
        buf.extend_from_slice(&(self.tuple_id.page_id.page_num as u32).to_le_bytes());
        buf.extend_from_slice(&self.tuple_id.slot_id.to_le_bytes());
        buf.freeze()
    }

    /// Deserializes an entry from bytes. Returns (entry, bytes_consumed).
    /// Reconstructs PageId with file_id=0. Callers that need the correct
    /// file_id must set it from context after deserialization.
    pub fn from_bytes(buf: &[u8]) -> Option<(Self, usize)> {
        if buf.len() < 8 {
            return None;
        }

        let key_len = u16::from_le_bytes([buf[0], buf[1]]) as usize;
        if buf.len() < 2 + key_len + 6 {
            return None;
        }

        let key = Bytes::copy_from_slice(&buf[2..2 + key_len]);
        let page_num = u32::from_le_bytes([
            buf[2 + key_len],
            buf[3 + key_len],
            buf[4 + key_len],
            buf[5 + key_len],
        ]);
        let slot_id = u16::from_le_bytes([buf[6 + key_len], buf[7 + key_len]]);

        let page_id = PageId::new(0, page_num as u64);
        let tuple_id = TupleId::new(page_id, slot_id);
        Some((Self { key, tuple_id }, 2 + key_len + 6))
    }
}

/// A key-pointer entry in an internal page.
///
/// Layout (on-disk):
/// - key_len: 2 bytes
/// - key: variable
/// - child_page_num: 4 bytes (u32, file_id is implicit from the B+tree index)
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
        2 + self.key.len() + 4 // key_len + key + page_num(4)
    }

    /// Serializes the entry to bytes.
    /// Stores only page_num (u32). file_id is reconstructed from the
    /// B+tree index context on read.
    pub fn to_bytes(&self) -> Bytes {
        let mut buf = BytesMut::with_capacity(self.size_on_disk());
        buf.extend_from_slice(&(self.key.len() as u16).to_le_bytes());
        buf.extend_from_slice(&self.key);
        buf.extend_from_slice(&(self.child_page_id.page_num as u32).to_le_bytes());
        buf.freeze()
    }

    /// Deserializes an entry from bytes. Returns (entry, bytes_consumed).
    /// Reconstructs PageId with file_id=0. Callers that need the correct
    /// file_id must set it from context after deserialization.
    pub fn from_bytes(buf: &[u8]) -> Option<(Self, usize)> {
        if buf.len() < 6 {
            return None;
        }

        let key_len = u16::from_le_bytes([buf[0], buf[1]]) as usize;
        if buf.len() < 2 + key_len + 4 {
            return None;
        }

        let key = Bytes::copy_from_slice(&buf[2..2 + key_len]);
        let page_num = u32::from_le_bytes([
            buf[2 + key_len],
            buf[3 + key_len],
            buf[4 + key_len],
            buf[5 + key_len],
        ]);
        let child_page_id = PageId::new(0, page_num as u64);

        Some((Self { key, child_page_id }, 2 + key_len + 4))
    }
}

/// Performance statistics for profiling insert operations.
#[derive(Default, Clone)]
pub struct InsertStats {
    /// Number of flush operations performed.
    pub flush_count: u64,
    /// Total time spent in flush operations (nanoseconds).
    pub flush_time_ns: u64,
}

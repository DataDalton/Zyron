//! Common types for B+Tree implementations.

use crate::tuple::TupleId;
use bytes::{Bytes, BytesMut};
use zyron_common::page::{PAGE_SIZE, PageHeader, PageId};
use zyron_common::zerocopy::{AsBytes, FromBytes};

/// Packed 16-byte leaf page header for single-memcpy serialization.
#[repr(C, packed)]
struct PackedLeafHeader {
    num_slots: u16,
    data_end: u16,
    next_leaf: u64,
    reserved: u32,
}

const _: () = {
    assert!(std::mem::size_of::<PackedLeafHeader>() == 2 + 2 + 8 + 4);
    assert!(std::mem::align_of::<PackedLeafHeader>() == 1);
};

unsafe impl AsBytes for PackedLeafHeader {}
unsafe impl FromBytes for PackedLeafHeader {}

/// Packed 16-byte internal page header for single-memcpy serialization.
#[repr(C, packed)]
struct PackedInternalHeader {
    num_keys: u16,
    free_space_offset: u16,
    level: u16,
    reserved: [u8; 10],
}

const _INTERNAL: () = {
    assert!(std::mem::size_of::<PackedInternalHeader>() == 2 + 2 + 2 + 10);
    assert!(std::mem::align_of::<PackedInternalHeader>() == 1);
};

unsafe impl AsBytes for PackedInternalHeader {}
unsafe impl FromBytes for PackedInternalHeader {}

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

    /// Serializes to bytes via single memcpy.
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let packed = PackedLeafHeader {
            num_slots: self.num_slots.to_le(),
            data_end: self.data_end.to_le(),
            next_leaf: self.next_leaf.to_le(),
            reserved: self.reserved.to_le(),
        };
        let mut buf = [0u8; Self::SIZE];
        packed.write_to(&mut buf, 0);
        buf
    }

    /// Deserializes from bytes via single unaligned read.
    pub fn from_bytes(buf: &[u8]) -> Self {
        let packed = PackedLeafHeader::read_from(buf, 0);
        Self {
            num_slots: u16::from_le(packed.num_slots),
            data_end: u16::from_le(packed.data_end),
            next_leaf: u64::from_le(packed.next_leaf),
            reserved: u32::from_le(packed.reserved),
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

    /// Serializes to bytes via single memcpy.
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let packed = PackedInternalHeader {
            num_keys: self.num_keys.to_le(),
            free_space_offset: self.free_space_offset.to_le(),
            level: self.level.to_le(),
            reserved: self.reserved,
        };
        let mut buf = [0u8; Self::SIZE];
        packed.write_to(&mut buf, 0);
        buf
    }

    /// Deserializes from bytes via single unaligned read.
    pub fn from_bytes(buf: &[u8]) -> Self {
        let packed = PackedInternalHeader::read_from(buf, 0);
        Self {
            num_keys: u16::from_le(packed.num_keys),
            free_space_offset: u16::from_le(packed.free_space_offset),
            level: u16::from_le(packed.level),
            reserved: packed.reserved,
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

/// Zero-copy view of a leaf entry. Borrows key from page buffer.
#[derive(Debug, Clone, Copy)]
pub struct LeafEntryView<'a> {
    pub key: &'a [u8],
    pub tuple_id: TupleId,
}

impl<'a> LeafEntryView<'a> {
    /// Parses a leaf entry view from a byte slice without copying the key.
    pub fn from_bytes(buf: &'a [u8]) -> Option<(Self, usize)> {
        if buf.len() < 8 {
            return None;
        }
        let key_len = u16::from_le_bytes([buf[0], buf[1]]) as usize;
        let total = 2 + key_len + 6;
        if buf.len() < total {
            return None;
        }
        let key = &buf[2..2 + key_len];
        let page_num = u32::from_le_bytes([
            buf[2 + key_len],
            buf[3 + key_len],
            buf[4 + key_len],
            buf[5 + key_len],
        ]);
        let slot_id = u16::from_le_bytes([buf[6 + key_len], buf[7 + key_len]]);
        let page_id = PageId::new(0, page_num as u64);
        let tuple_id = TupleId::new(page_id, slot_id);
        Some((Self { key, tuple_id }, total))
    }

    /// Size of this entry on disk.
    pub fn size_on_disk(&self) -> usize {
        2 + self.key.len() + 6
    }

    /// Converts to an owned LeafEntry by copying the key.
    pub fn to_owned(&self) -> LeafEntry {
        LeafEntry {
            key: Bytes::copy_from_slice(self.key),
            tuple_id: self.tuple_id,
        }
    }

    /// Writes this entry directly to a byte slice at the given offset.
    /// Returns the number of bytes written.
    #[inline]
    pub fn write_to_slice(&self, buf: &mut [u8], offset: usize) -> usize {
        let kl = self.key.len();
        buf[offset..offset + 2].copy_from_slice(&(kl as u16).to_le_bytes());
        buf[offset + 2..offset + 2 + kl].copy_from_slice(self.key);
        let vo = offset + 2 + kl;
        buf[vo..vo + 4].copy_from_slice(&(self.tuple_id.page_id.page_num as u32).to_le_bytes());
        buf[vo + 4..vo + 6].copy_from_slice(&self.tuple_id.slot_id.to_le_bytes());
        2 + kl + 6
    }
}

impl LeafEntry {
    /// Writes this entry directly to a byte slice at the given offset.
    /// Returns the number of bytes written. Avoids BytesMut allocation.
    #[inline]
    pub fn write_to_slice(&self, buf: &mut [u8], offset: usize) -> usize {
        let kl = self.key.len();
        buf[offset..offset + 2].copy_from_slice(&(kl as u16).to_le_bytes());
        buf[offset + 2..offset + 2 + kl].copy_from_slice(&self.key);
        let vo = offset + 2 + kl;
        buf[vo..vo + 4].copy_from_slice(&(self.tuple_id.page_id.page_num as u32).to_le_bytes());
        buf[vo + 4..vo + 6].copy_from_slice(&self.tuple_id.slot_id.to_le_bytes());
        2 + kl + 6
    }
}

/// Zero-copy view of an internal entry. Borrows key from page buffer.
#[derive(Debug, Clone, Copy)]
pub struct InternalEntryView<'a> {
    pub key: &'a [u8],
    pub child_page_id: PageId,
}

impl<'a> InternalEntryView<'a> {
    /// Parses an internal entry view from a byte slice without copying the key.
    pub fn from_bytes(buf: &'a [u8]) -> Option<(Self, usize)> {
        if buf.len() < 6 {
            return None;
        }
        let key_len = u16::from_le_bytes([buf[0], buf[1]]) as usize;
        let total = 2 + key_len + 4;
        if buf.len() < total {
            return None;
        }
        let key = &buf[2..2 + key_len];
        let page_num = u32::from_le_bytes([
            buf[2 + key_len],
            buf[3 + key_len],
            buf[4 + key_len],
            buf[5 + key_len],
        ]);
        let child_page_id = PageId::new(0, page_num as u64);
        Some((Self { key, child_page_id }, total))
    }

    /// Size of this entry on disk.
    pub fn size_on_disk(&self) -> usize {
        2 + self.key.len() + 4
    }

    /// Converts to an owned InternalEntry by copying the key.
    pub fn to_owned(&self) -> InternalEntry {
        InternalEntry {
            key: Bytes::copy_from_slice(self.key),
            child_page_id: self.child_page_id,
        }
    }

    /// Writes this entry directly to a byte slice at the given offset.
    /// Returns the number of bytes written.
    #[inline]
    pub fn write_to_slice(&self, buf: &mut [u8], offset: usize) -> usize {
        let kl = self.key.len();
        buf[offset..offset + 2].copy_from_slice(&(kl as u16).to_le_bytes());
        buf[offset + 2..offset + 2 + kl].copy_from_slice(self.key);
        let vo = offset + 2 + kl;
        buf[vo..vo + 4].copy_from_slice(&(self.child_page_id.page_num as u32).to_le_bytes());
        2 + kl + 4
    }
}

impl InternalEntry {
    /// Writes this entry directly to a byte slice at the given offset.
    /// Returns the number of bytes written. Avoids BytesMut allocation.
    #[inline]
    pub fn write_to_slice(&self, buf: &mut [u8], offset: usize) -> usize {
        let kl = self.key.len();
        buf[offset..offset + 2].copy_from_slice(&(kl as u16).to_le_bytes());
        buf[offset + 2..offset + 2 + kl].copy_from_slice(&self.key);
        let vo = offset + 2 + kl;
        buf[vo..vo + 4].copy_from_slice(&(self.child_page_id.page_num as u32).to_le_bytes());
        2 + kl + 4
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

#[cfg(test)]
mod tests {
    use super::*;

    // Packed struct fields can't be referenced in assert_eq! due to alignment.
    // Copy fields to locals before comparing.

    #[test]
    fn test_packed_leaf_header_roundtrip() {
        let header = PackedLeafHeader {
            num_slots: 42,
            data_end: 8192,
            next_leaf: 0x0102030405060708,
            reserved: 0,
        };

        let bytes = header.as_bytes();
        assert_eq!(bytes.len(), 16);

        let r = PackedLeafHeader::read_from(bytes, 0);
        let (ns, de, nl, rv) = (r.num_slots, r.data_end, r.next_leaf, r.reserved);
        assert_eq!(ns, 42);
        assert_eq!(de, 8192);
        assert_eq!(nl, 0x0102030405060708);
        assert_eq!(rv, 0);
    }

    #[test]
    fn test_packed_leaf_header_write_to() {
        let header = PackedLeafHeader {
            num_slots: 10,
            data_end: 4096,
            next_leaf: 999,
            reserved: 0,
        };

        let mut buf = [0u8; 32];
        let written = header.write_to(&mut buf, 8);
        assert_eq!(written, 16);

        let r = PackedLeafHeader::read_from(&buf, 8);
        let (ns, de, nl) = (r.num_slots, r.data_end, r.next_leaf);
        assert_eq!(ns, 10);
        assert_eq!(de, 4096);
        assert_eq!(nl, 999);
    }

    #[test]
    fn test_packed_internal_header_roundtrip() {
        let header = PackedInternalHeader {
            num_keys: 100,
            free_space_offset: 2048,
            level: 3,
            reserved: [0; 10],
        };

        let bytes = header.as_bytes();
        assert_eq!(bytes.len(), 16);

        let r = PackedInternalHeader::read_from(bytes, 0);
        let (nk, fso, lv) = (r.num_keys, r.free_space_offset, r.level);
        assert_eq!(nk, 100);
        assert_eq!(fso, 2048);
        assert_eq!(lv, 3);
    }

    #[test]
    fn test_packed_internal_header_all_bits() {
        let header = PackedInternalHeader {
            num_keys: u16::MAX,
            free_space_offset: u16::MAX,
            level: u16::MAX,
            reserved: [0xFF; 10],
        };

        let r = PackedInternalHeader::read_from(header.as_bytes(), 0);
        let (nk, fso, lv) = (r.num_keys, r.free_space_offset, r.level);
        assert_eq!(nk, u16::MAX);
        assert_eq!(fso, u16::MAX);
        assert_eq!(lv, u16::MAX);
    }

    #[test]
    fn test_leaf_page_header_roundtrip() {
        let header = LeafPageHeader {
            num_slots: 5,
            data_end: 1024,
            next_leaf: 42,
            reserved: 0,
        };

        let bytes = header.to_bytes();
        let restored = LeafPageHeader::from_bytes(&bytes);
        assert_eq!(restored.num_slots, 5);
        assert_eq!(restored.data_end, 1024);
        assert_eq!(restored.next_leaf, 42);
    }

    #[test]
    fn test_internal_page_header_roundtrip() {
        let header = InternalPageHeader {
            num_keys: 20,
            free_space_offset: 512,
            level: 2,
            reserved: [0; 10],
        };

        let bytes = header.to_bytes();
        let restored = InternalPageHeader::from_bytes(&bytes);
        assert_eq!(restored.num_keys, 20);
        assert_eq!(restored.free_space_offset, 512);
        assert_eq!(restored.level, 2);
    }

    #[test]
    fn test_leaf_entry_roundtrip() {
        let entry = LeafEntry {
            key: Bytes::from(vec![1, 2, 3, 4]),
            tuple_id: TupleId::new(PageId::new(0, 10), 5),
        };

        let bytes = entry.to_bytes();
        let (restored, _size) = LeafEntry::from_bytes(&bytes).unwrap();
        assert_eq!(restored.key, entry.key);
        assert_eq!(restored.tuple_id.page_id.page_num, 10);
        assert_eq!(restored.tuple_id.slot_id, 5);
    }

    #[test]
    fn test_internal_entry_roundtrip() {
        let entry = InternalEntry {
            key: Bytes::from(vec![10, 20, 30]),
            child_page_id: PageId::new(0, 77),
        };

        let bytes = entry.to_bytes();
        let (restored, size) = InternalEntry::from_bytes(&bytes).unwrap();
        assert_eq!(restored.key, entry.key);
        assert_eq!(restored.child_page_id.page_num, 77);
        assert_eq!(size, 2 + 3 + 4); // key_len(2) + key(3) + page_num(4)
    }

    #[test]
    fn test_compare_keys_short() {
        assert_eq!(compare_keys(b"abc", b"abc"), std::cmp::Ordering::Equal);
        assert_eq!(compare_keys(b"abc", b"abd"), std::cmp::Ordering::Less);
        assert_eq!(compare_keys(b"abd", b"abc"), std::cmp::Ordering::Greater);
    }

    #[test]
    fn test_compare_keys_long_prefix_optimization() {
        let a = b"12345678_suffix_a";
        let b = b"12345678_suffix_b";
        // Same 8-byte prefix, falls back to slice comparison
        assert_eq!(compare_keys(a, b), std::cmp::Ordering::Less);
    }

    #[test]
    fn test_compare_keys_long_different_prefix() {
        let a = b"00000000rest";
        let b = b"11111111rest";
        // Different prefix, u64 comparison catches it
        assert_eq!(compare_keys(a, b), std::cmp::Ordering::Less);
    }
}

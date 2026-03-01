//! Page structures for ZyronDB storage.

use serde::{Deserialize, Serialize};

/// Default page size in bytes (16 KB).
pub const PAGE_SIZE: usize = 16 * 1024;

/// Unique identifier for a page within a file.
///
/// PageId consists of a file ID and a logical page number. The page_num is u64
/// to support segmented storage with unlimited table sizes. For buffer pool
/// packing into u64, only the lower 32 bits of page_num are used (segment
/// offsets fit within u32).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PageId {
    /// File identifier (0 = data file, 1+ = index files).
    pub file_id: u32,
    /// Logical page number within the file (0-indexed).
    pub page_num: u64,
}

impl PageId {
    /// Creates a new PageId.
    pub fn new(file_id: u32, page_num: u64) -> Self {
        Self { file_id, page_num }
    }

    /// Packs the PageId into a u64 for buffer pool frame lookup.
    /// Uses file_id in upper 32 bits and lower 32 bits of page_num.
    /// For segmented storage, page_num within a segment always fits in u32.
    pub fn as_u64(&self) -> u64 {
        ((self.file_id as u64) << 32) | (self.page_num & 0xFFFF_FFFF)
    }

    /// Creates a PageId from a u64 representation.
    /// Recovers file_id from upper 32 bits and page_num from lower 32 bits.
    pub fn from_u64(value: u64) -> Self {
        Self {
            file_id: (value >> 32) as u32,
            page_num: (value as u32) as u64,
        }
    }

    /// Returns the segment ID for this page given a segment size in pages.
    #[inline]
    pub fn segment_id(&self, segment_size: u64) -> u64 {
        self.page_num / segment_size
    }

    /// Returns the offset within the segment for this page.
    #[inline]
    pub fn offset_in_segment(&self, segment_size: u64) -> u32 {
        (self.page_num % segment_size) as u32
    }
}

impl std::fmt::Display for PageId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.file_id, self.page_num)
    }
}

/// Page types in ZyronDB storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum PageType {
    /// Unallocated/free page.
    Free = 0,
    /// Heap data page containing tuples.
    Heap = 1,
    /// B+ tree leaf page.
    BTreeLeaf = 2,
    /// B+ tree internal page.
    BTreeInternal = 3,
    /// Free space map page.
    FreeSpaceMap = 4,
    /// Visibility map page.
    VisibilityMap = 5,
    /// Overflow page for large values.
    Overflow = 6,
}

/// Header structure at the beginning of every page.
///
/// Layout (40 bytes total, format v2):
/// - file_id: 4 bytes
/// - page_num: 8 bytes (u64, was u32 in v1)
/// - lsn: 8 bytes (log sequence number for recovery)
/// - page_type: 1 byte
/// - flags: 1 byte
/// - free_space_offset: 2 bytes
/// - tuple_count: 2 bytes
/// - checksum: 4 bytes
/// - format_version: 1 byte
/// - reserved: 9 bytes
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub struct PageHeader {
    /// Unique page identifier.
    pub page_id: PageId,
    /// Log sequence number of the last modification.
    pub lsn: u64,
    /// Type of this page.
    pub page_type: PageType,
    /// Page flags (dirty, pinned, etc.).
    pub flags: PageFlags,
    /// Offset to the start of free space within the page.
    pub free_space_offset: u16,
    /// Number of tuples/entries on this page.
    pub tuple_count: u16,
    /// CRC32 checksum of the page contents (excluding this field).
    pub checksum: u32,
    /// Page header format version (2 = u64 page_num).
    pub format_version: u8,
}

/// Current page header format version.
pub const PAGE_HEADER_FORMAT_VERSION: u8 = 2;

impl PageHeader {
    /// Size of the page header in bytes.
    pub const SIZE: usize = 40;

    /// Creates a new page header.
    pub fn new(page_id: PageId, page_type: PageType) -> Self {
        Self {
            page_id,
            lsn: 0,
            page_type,
            flags: PageFlags::empty(),
            free_space_offset: Self::SIZE as u16,
            tuple_count: 0,
            checksum: 0,
            format_version: PAGE_HEADER_FORMAT_VERSION,
        }
    }

    /// Returns the amount of free space available on this page.
    pub fn free_space(&self) -> usize {
        PAGE_SIZE - self.free_space_offset as usize
    }

    /// Serializes the header to bytes.
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..4].copy_from_slice(&self.page_id.file_id.to_le_bytes());
        buf[4..12].copy_from_slice(&self.page_id.page_num.to_le_bytes());
        buf[12..20].copy_from_slice(&self.lsn.to_le_bytes());
        buf[20] = self.page_type as u8;
        buf[21] = self.flags.0;
        buf[22..24].copy_from_slice(&self.free_space_offset.to_le_bytes());
        buf[24..26].copy_from_slice(&self.tuple_count.to_le_bytes());
        buf[26..30].copy_from_slice(&self.checksum.to_le_bytes());
        buf[30] = self.format_version;
        // bytes 31-39 are reserved (already zeroed)
        buf
    }

    /// Deserializes the header from bytes.
    pub fn from_bytes(buf: &[u8]) -> Self {
        let file_id = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        let page_num = u64::from_le_bytes([
            buf[4], buf[5], buf[6], buf[7], buf[8], buf[9], buf[10], buf[11],
        ]);
        let lsn = u64::from_le_bytes([
            buf[12], buf[13], buf[14], buf[15], buf[16], buf[17], buf[18], buf[19],
        ]);
        let page_type = match buf[20] {
            0 => PageType::Free,
            1 => PageType::Heap,
            2 => PageType::BTreeLeaf,
            3 => PageType::BTreeInternal,
            4 => PageType::FreeSpaceMap,
            5 => PageType::VisibilityMap,
            6 => PageType::Overflow,
            _ => PageType::Free,
        };
        let flags = PageFlags(buf[21]);
        let free_space_offset = u16::from_le_bytes([buf[22], buf[23]]);
        let tuple_count = u16::from_le_bytes([buf[24], buf[25]]);
        let checksum = u32::from_le_bytes([buf[26], buf[27], buf[28], buf[29]]);
        let format_version = if buf.len() > 30 { buf[30] } else { 1 };

        Self {
            page_id: PageId::new(file_id, page_num),
            lsn,
            page_type,
            flags,
            free_space_offset,
            tuple_count,
            checksum,
            format_version,
        }
    }
}

/// Flags for page state.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct PageFlags(u8);

impl PageFlags {
    /// No flags set.
    pub const fn empty() -> Self {
        Self(0)
    }

    /// Page has been modified and needs to be written to disk.
    pub const DIRTY: u8 = 0b0000_0001;
    /// Page is pinned in the buffer pool.
    pub const PINNED: u8 = 0b0000_0010;
    /// Page is being written to disk.
    pub const FLUSHING: u8 = 0b0000_0100;

    /// Returns true if the dirty flag is set.
    pub fn is_dirty(&self) -> bool {
        self.0 & Self::DIRTY != 0
    }

    /// Sets the dirty flag.
    pub fn set_dirty(&mut self, dirty: bool) {
        if dirty {
            self.0 |= Self::DIRTY;
        } else {
            self.0 &= !Self::DIRTY;
        }
    }

    /// Returns true if the pinned flag is set.
    pub fn is_pinned(&self) -> bool {
        self.0 & Self::PINNED != 0
    }

    /// Sets the pinned flag.
    pub fn set_pinned(&mut self, pinned: bool) {
        if pinned {
            self.0 |= Self::PINNED;
        } else {
            self.0 &= !Self::PINNED;
        }
    }

    /// Returns true if the flushing flag is set.
    pub fn is_flushing(&self) -> bool {
        self.0 & Self::FLUSHING != 0
    }

    /// Sets the flushing flag.
    pub fn set_flushing(&mut self, flushing: bool) {
        if flushing {
            self.0 |= Self::FLUSHING;
        } else {
            self.0 &= !Self::FLUSHING;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_size_constant() {
        assert_eq!(PAGE_SIZE, 16 * 1024);
        assert_eq!(PAGE_SIZE, 16384);
    }

    #[test]
    fn test_page_id_new() {
        let page_id = PageId::new(1, 100);
        assert_eq!(page_id.file_id, 1);
        assert_eq!(page_id.page_num, 100);
    }

    #[test]
    fn test_page_id_roundtrip() {
        let page_id = PageId::new(42, 1000);
        let as_u64 = page_id.as_u64();
        let recovered = PageId::from_u64(as_u64);
        assert_eq!(page_id, recovered);
    }

    #[test]
    fn test_page_id_roundtrip_edge_cases() {
        // Zero values
        let page_id = PageId::new(0, 0);
        assert_eq!(page_id, PageId::from_u64(page_id.as_u64()));

        // Max u32 range values (fits in lower 32 bits of as_u64)
        let page_id = PageId::new(u32::MAX, u32::MAX as u64);
        assert_eq!(page_id, PageId::from_u64(page_id.as_u64()));

        // Mixed values
        let page_id = PageId::new(0, u32::MAX as u64);
        assert_eq!(page_id, PageId::from_u64(page_id.as_u64()));

        let page_id = PageId::new(u32::MAX, 0);
        assert_eq!(page_id, PageId::from_u64(page_id.as_u64()));
    }

    #[test]
    fn test_page_id_as_u64_bit_layout() {
        let page_id = PageId::new(1, 2);
        let as_u64 = page_id.as_u64();
        // file_id (1) in upper 32 bits, page_num (2) in lower 32 bits
        assert_eq!(as_u64, (1u64 << 32) | 2);
    }

    #[test]
    fn test_page_id_segment_helpers() {
        let segment_size = 65536u64;
        // Page in first segment
        let page = PageId::new(0, 100);
        assert_eq!(page.segment_id(segment_size), 0);
        assert_eq!(page.offset_in_segment(segment_size), 100);

        // Page in second segment
        let page = PageId::new(0, 65536);
        assert_eq!(page.segment_id(segment_size), 1);
        assert_eq!(page.offset_in_segment(segment_size), 0);

        // Page in third segment with offset
        let page = PageId::new(0, 131073);
        assert_eq!(page.segment_id(segment_size), 2);
        assert_eq!(page.offset_in_segment(segment_size), 1);
    }

    #[test]
    fn test_page_id_display() {
        let page_id = PageId::new(5, 123);
        assert_eq!(page_id.to_string(), "5:123");

        let page_id = PageId::new(0, 0);
        assert_eq!(page_id.to_string(), "0:0");
    }

    #[test]
    fn test_page_id_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(PageId::new(1, 1));
        set.insert(PageId::new(1, 2));
        set.insert(PageId::new(1, 1)); // Duplicate

        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_page_type_repr() {
        assert_eq!(PageType::Free as u8, 0);
        assert_eq!(PageType::Heap as u8, 1);
        assert_eq!(PageType::BTreeLeaf as u8, 2);
        assert_eq!(PageType::BTreeInternal as u8, 3);
        assert_eq!(PageType::FreeSpaceMap as u8, 4);
        assert_eq!(PageType::VisibilityMap as u8, 5);
        assert_eq!(PageType::Overflow as u8, 6);
    }

    #[test]
    fn test_page_header_new() {
        let page_id = PageId::new(0, 42);
        let header = PageHeader::new(page_id, PageType::Heap);

        assert_eq!(header.page_id, page_id);
        assert_eq!(header.lsn, 0);
        assert_eq!(header.page_type, PageType::Heap);
        assert!(!header.flags.is_dirty());
        assert!(!header.flags.is_pinned());
        assert_eq!(header.free_space_offset, PageHeader::SIZE as u16);
        assert_eq!(header.tuple_count, 0);
        assert_eq!(header.checksum, 0);
        assert_eq!(header.format_version, PAGE_HEADER_FORMAT_VERSION);
    }

    #[test]
    fn test_page_header_size() {
        assert_eq!(PageHeader::SIZE, 40);
    }

    #[test]
    fn test_page_header_free_space() {
        let page_id = PageId::new(0, 0);
        let header = PageHeader::new(page_id, PageType::Heap);

        // Initial free space = PAGE_SIZE - header size
        assert_eq!(header.free_space(), PAGE_SIZE - PageHeader::SIZE);
        assert_eq!(header.free_space(), 16384 - 40);
        assert_eq!(header.free_space(), 16344);
    }

    #[test]
    fn test_page_header_free_space_after_allocation() {
        let page_id = PageId::new(0, 0);
        let mut header = PageHeader::new(page_id, PageType::Heap);

        // Simulate allocating 100 bytes
        header.free_space_offset += 100;
        assert_eq!(
            header.free_space(),
            PAGE_SIZE - header.free_space_offset as usize
        );
        assert_eq!(header.free_space(), 16344 - 100);
    }

    #[test]
    fn test_page_flags_empty() {
        let flags = PageFlags::empty();
        assert!(!flags.is_dirty());
        assert!(!flags.is_pinned());
        assert!(!flags.is_flushing());
    }

    #[test]
    fn test_page_flags_dirty() {
        let mut flags = PageFlags::empty();

        flags.set_dirty(true);
        assert!(flags.is_dirty());
        assert!(!flags.is_pinned());
        assert!(!flags.is_flushing());

        flags.set_dirty(false);
        assert!(!flags.is_dirty());
    }

    #[test]
    fn test_page_flags_pinned() {
        let mut flags = PageFlags::empty();

        flags.set_pinned(true);
        assert!(!flags.is_dirty());
        assert!(flags.is_pinned());
        assert!(!flags.is_flushing());

        flags.set_pinned(false);
        assert!(!flags.is_pinned());
    }

    #[test]
    fn test_page_flags_flushing() {
        let mut flags = PageFlags::empty();

        flags.set_flushing(true);
        assert!(!flags.is_dirty());
        assert!(!flags.is_pinned());
        assert!(flags.is_flushing());

        flags.set_flushing(false);
        assert!(!flags.is_flushing());
    }

    #[test]
    fn test_page_flags_combined() {
        let mut flags = PageFlags::empty();

        // Set all flags
        flags.set_dirty(true);
        flags.set_pinned(true);
        flags.set_flushing(true);

        assert!(flags.is_dirty());
        assert!(flags.is_pinned());
        assert!(flags.is_flushing());

        // Clear one flag, others remain
        flags.set_dirty(false);
        assert!(!flags.is_dirty());
        assert!(flags.is_pinned());
        assert!(flags.is_flushing());
    }

    #[test]
    fn test_page_flags_default() {
        let flags = PageFlags::default();
        assert!(!flags.is_dirty());
        assert!(!flags.is_pinned());
        assert!(!flags.is_flushing());
    }

    #[test]
    fn test_page_flags_bit_values() {
        assert_eq!(PageFlags::DIRTY, 0b0000_0001);
        assert_eq!(PageFlags::PINNED, 0b0000_0010);
        assert_eq!(PageFlags::FLUSHING, 0b0000_0100);
    }

    #[test]
    fn test_page_id_serde_roundtrip() {
        let original = PageId::new(10, 500);
        let serialized = serde_json::to_string(&original).unwrap();
        let deserialized: PageId = serde_json::from_str(&serialized).unwrap();
        assert_eq!(original, deserialized);
    }

    #[test]
    fn test_page_type_serde_roundtrip() {
        for page_type in [
            PageType::Free,
            PageType::Heap,
            PageType::BTreeLeaf,
            PageType::BTreeInternal,
            PageType::FreeSpaceMap,
            PageType::VisibilityMap,
            PageType::Overflow,
        ] {
            let serialized = serde_json::to_string(&page_type).unwrap();
            let deserialized: PageType = serde_json::from_str(&serialized).unwrap();
            assert_eq!(page_type, deserialized);
        }
    }

    #[test]
    fn test_page_header_serde_roundtrip() {
        let page_id = PageId::new(3, 999);
        let mut header = PageHeader::new(page_id, PageType::BTreeLeaf);
        header.lsn = 12345;
        header.tuple_count = 50;
        header.checksum = 0xDEADBEEF;
        header.flags.set_dirty(true);

        let serialized = serde_json::to_string(&header).unwrap();
        let deserialized: PageHeader = serde_json::from_str(&serialized).unwrap();

        assert_eq!(header.page_id, deserialized.page_id);
        assert_eq!(header.lsn, deserialized.lsn);
        assert_eq!(header.page_type, deserialized.page_type);
        assert_eq!(header.tuple_count, deserialized.tuple_count);
        assert_eq!(header.checksum, deserialized.checksum);
    }
}

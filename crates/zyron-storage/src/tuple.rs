//! Tuple representation and serialization.

use zyron_common::page::PageId;

/// Unique identifier for a tuple within the database.
///
/// Combines a PageId with a slot number to uniquely identify
/// where a tuple is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TupleId {
    /// Page containing this tuple.
    pub page_id: PageId,
    /// Slot number within the page.
    pub slot_id: u16,
}

impl TupleId {
    /// Creates a new tuple ID.
    pub fn new(page_id: PageId, slot_id: u16) -> Self {
        Self { page_id, slot_id }
    }

    /// Invalid tuple ID.
    pub const INVALID: TupleId = TupleId {
        page_id: PageId {
            file_id: u32::MAX,
            page_num: u32::MAX,
        },
        slot_id: u16::MAX,
    };

    /// Returns true if this is a valid tuple ID.
    pub fn is_valid(&self) -> bool {
        self.page_id.file_id != u32::MAX
    }
}

impl std::fmt::Display for TupleId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.page_id, self.slot_id)
    }
}

/// A tuple (row) stored in the database.
///
/// Tuples are stored as variable-length byte sequences with a header
/// containing metadata about the tuple's state.
#[derive(Debug, Clone)]
pub struct Tuple {
    /// Tuple header.
    header: TupleHeader,
    /// Tuple data.
    data: Vec<u8>,
}

/// Header for a tuple.
///
/// Layout (12 bytes):
/// - flags: 2 bytes
/// - data_len: 2 bytes
/// - xmin: 4 bytes (transaction that created this tuple)
/// - xmax: 4 bytes (transaction that deleted/updated this tuple, 0 if live)
#[derive(Debug, Clone, Copy, Default)]
pub struct TupleHeader {
    /// Tuple flags.
    pub flags: TupleFlags,
    /// Length of tuple data in bytes.
    pub data_len: u16,
    /// Transaction ID that created this tuple.
    pub xmin: u32,
    /// Transaction ID that deleted or updated this tuple (0 if still live).
    pub xmax: u32,
}

impl TupleHeader {
    /// Size of the tuple header in bytes.
    pub const SIZE: usize = 12;

    /// Creates a new tuple header.
    pub fn new(data_len: u16, xmin: u32) -> Self {
        Self {
            flags: TupleFlags::empty(),
            data_len,
            xmin,
            xmax: 0,
        }
    }

    /// Creates a tuple header with both xmin and xmax.
    pub fn with_xmax(data_len: u16, xmin: u32, xmax: u32) -> Self {
        Self {
            flags: TupleFlags::empty(),
            data_len,
            xmin,
            xmax,
        }
    }

    /// Returns true if this tuple is visible to the given transaction.
    /// A tuple is visible if:
    /// - xmin is committed and less than or equal to the snapshot
    /// - xmax is either 0 (not deleted) or greater than the snapshot
    pub fn is_visible(&self, snapshot_xid: u32) -> bool {
        self.xmin <= snapshot_xid && (self.xmax == 0 || self.xmax > snapshot_xid)
    }

    /// Serializes the header to bytes.
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..2].copy_from_slice(&self.flags.0.to_le_bytes());
        buf[2..4].copy_from_slice(&self.data_len.to_le_bytes());
        buf[4..8].copy_from_slice(&self.xmin.to_le_bytes());
        buf[8..12].copy_from_slice(&self.xmax.to_le_bytes());
        buf
    }

    /// Deserializes the header from bytes.
    pub fn from_bytes(buf: &[u8]) -> Self {
        Self {
            flags: TupleFlags(u16::from_le_bytes([buf[0], buf[1]])),
            data_len: u16::from_le_bytes([buf[2], buf[3]]),
            xmin: u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]),
            xmax: u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]),
        }
    }

    /// Deserializes the header from bytes without bounds checks.
    ///
    /// # Safety
    /// Caller must ensure buf has at least SIZE (12) bytes.
    #[inline(always)]
    pub unsafe fn from_bytes_unchecked(buf: &[u8]) -> Self {
        unsafe {
            Self {
                flags: TupleFlags(u16::from_le_bytes([
                    *buf.get_unchecked(0),
                    *buf.get_unchecked(1),
                ])),
                data_len: u16::from_le_bytes([*buf.get_unchecked(2), *buf.get_unchecked(3)]),
                xmin: u32::from_le_bytes([
                    *buf.get_unchecked(4),
                    *buf.get_unchecked(5),
                    *buf.get_unchecked(6),
                    *buf.get_unchecked(7),
                ]),
                xmax: u32::from_le_bytes([
                    *buf.get_unchecked(8),
                    *buf.get_unchecked(9),
                    *buf.get_unchecked(10),
                    *buf.get_unchecked(11),
                ]),
            }
        }
    }
}

/// Flags for tuple state.
#[derive(Debug, Clone, Copy, Default)]
pub struct TupleFlags(pub u16);

impl TupleFlags {
    /// No flags set.
    pub const fn empty() -> Self {
        Self(0)
    }

    /// Tuple has been deleted.
    pub const DELETED: u16 = 0x0001;
    /// Tuple is being updated (locked).
    pub const LOCKED: u16 = 0x0002;
    /// Tuple has null values.
    pub const HAS_NULLS: u16 = 0x0004;
    /// Tuple has variable-length fields.
    pub const HAS_VARLEN: u16 = 0x0008;

    /// Returns true if the deleted flag is set.
    pub fn is_deleted(&self) -> bool {
        self.0 & Self::DELETED != 0
    }

    /// Sets the deleted flag.
    pub fn set_deleted(&mut self, deleted: bool) {
        if deleted {
            self.0 |= Self::DELETED;
        } else {
            self.0 &= !Self::DELETED;
        }
    }

    /// Returns true if the locked flag is set.
    pub fn is_locked(&self) -> bool {
        self.0 & Self::LOCKED != 0
    }

    /// Sets the locked flag.
    pub fn set_locked(&mut self, locked: bool) {
        if locked {
            self.0 |= Self::LOCKED;
        } else {
            self.0 &= !Self::LOCKED;
        }
    }

    /// Returns true if the tuple has null values.
    pub fn has_nulls(&self) -> bool {
        self.0 & Self::HAS_NULLS != 0
    }

    /// Sets the has_nulls flag.
    pub fn set_has_nulls(&mut self, has_nulls: bool) {
        if has_nulls {
            self.0 |= Self::HAS_NULLS;
        } else {
            self.0 &= !Self::HAS_NULLS;
        }
    }
}

impl Tuple {
    /// Creates a new tuple from raw data.
    pub fn new(data: Vec<u8>, xmin: u32) -> Self {
        let header = TupleHeader::new(data.len() as u16, xmin);
        Self { header, data }
    }

    /// Creates a tuple with a specific header.
    pub fn with_header(header: TupleHeader, data: Vec<u8>) -> Self {
        Self { header, data }
    }

    /// Returns the tuple header.
    pub fn header(&self) -> &TupleHeader {
        &self.header
    }

    /// Returns a mutable reference to the header.
    pub fn header_mut(&mut self) -> &mut TupleHeader {
        &mut self.header
    }

    /// Returns the tuple data.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Returns the total size of this tuple on disk (header + data).
    pub fn size_on_disk(&self) -> usize {
        TupleHeader::SIZE + self.data.len()
    }

    /// Returns true if this tuple is marked as deleted.
    pub fn is_deleted(&self) -> bool {
        self.header.flags.is_deleted()
    }

    /// Marks this tuple as deleted.
    pub fn set_deleted(&mut self, deleted: bool) {
        self.header.flags.set_deleted(deleted);
    }

    /// Serializes the tuple to bytes.
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.size_on_disk());
        buf.extend_from_slice(&self.header.to_bytes());
        buf.extend_from_slice(&self.data);
        buf
    }

    /// Deserializes a tuple from bytes.
    #[inline]
    pub fn deserialize(buf: &[u8]) -> Option<Self> {
        if buf.len() < TupleHeader::SIZE {
            return None;
        }

        let header = TupleHeader::from_bytes(&buf[..TupleHeader::SIZE]);
        let data_end = TupleHeader::SIZE + header.data_len as usize;

        if buf.len() < data_end {
            return None;
        }

        let data = buf[TupleHeader::SIZE..data_end].to_vec();
        Some(Self { header, data })
    }
}

/// Zero-copy view into tuple data stored in a page buffer.
///
/// Holds a borrowed reference to tuple data, avoiding heap allocation.
/// The lifetime is tied to the page buffer that holds the data.
#[derive(Debug, Clone, Copy)]
pub struct TupleView<'a> {
    /// Tuple header (copied, 12 bytes).
    pub header: TupleHeader,
    /// Reference to tuple data in the page buffer.
    pub data: &'a [u8],
}

impl<'a> TupleView<'a> {
    /// Creates a new tuple view.
    #[inline]
    pub fn new(header: TupleHeader, data: &'a [u8]) -> Self {
        Self { header, data }
    }

    /// Converts this view to an owned Tuple by copying the data.
    #[inline]
    pub fn to_owned(&self) -> Tuple {
        Tuple::with_header(self.header, self.data.to_vec())
    }

    /// Returns the tuple data length.
    #[inline]
    pub fn data_len(&self) -> usize {
        self.data.len()
    }

    /// Returns total size on disk (header + data).
    #[inline]
    pub fn size_on_disk(&self) -> usize {
        TupleHeader::SIZE + self.data.len()
    }

    /// Returns true if this tuple is marked as deleted.
    #[inline]
    pub fn is_deleted(&self) -> bool {
        self.header.flags.is_deleted()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tuple_id_new() {
        let page_id = PageId::new(1, 42);
        let tuple_id = TupleId::new(page_id, 5);

        assert_eq!(tuple_id.page_id, page_id);
        assert_eq!(tuple_id.slot_id, 5);
        assert!(tuple_id.is_valid());
    }

    #[test]
    fn test_tuple_id_invalid() {
        let invalid = TupleId::INVALID;
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_tuple_id_display() {
        let tuple_id = TupleId::new(PageId::new(1, 42), 5);
        assert_eq!(tuple_id.to_string(), "1:42:5");
    }

    #[test]
    fn test_tuple_header_roundtrip() {
        let header = TupleHeader {
            flags: TupleFlags(0x0003),
            data_len: 256,
            xmin: 12345,
            xmax: 67890,
        };

        let bytes = header.to_bytes();
        let recovered = TupleHeader::from_bytes(&bytes);

        assert_eq!(recovered.flags.0, header.flags.0);
        assert_eq!(recovered.data_len, header.data_len);
        assert_eq!(recovered.xmin, header.xmin);
        assert_eq!(recovered.xmax, header.xmax);
    }

    #[test]
    fn test_tuple_flags() {
        let mut flags = TupleFlags::empty();

        assert!(!flags.is_deleted());
        assert!(!flags.is_locked());
        assert!(!flags.has_nulls());

        flags.set_deleted(true);
        assert!(flags.is_deleted());

        flags.set_locked(true);
        assert!(flags.is_locked());

        flags.set_has_nulls(true);
        assert!(flags.has_nulls());

        flags.set_deleted(false);
        assert!(!flags.is_deleted());
        assert!(flags.is_locked()); // Other flags unaffected
    }

    #[test]
    fn test_tuple_new() {
        let data = b"hello world".to_vec();
        let tuple = Tuple::new(data.clone(), 100);

        assert_eq!(tuple.data(), &data);
        assert_eq!(tuple.header().data_len, 11);
        assert_eq!(tuple.header().xmin, 100);
        assert!(!tuple.is_deleted());
    }

    #[test]
    fn test_tuple_size_on_disk() {
        let data = vec![0u8; 100];
        let tuple = Tuple::new(data, 1);

        assert_eq!(tuple.size_on_disk(), TupleHeader::SIZE + 100);
    }

    #[test]
    fn test_tuple_serialize_deserialize() {
        let data = b"test data 123".to_vec();
        let tuple = Tuple::new(data.clone(), 999);

        let serialized = tuple.serialize();
        let recovered = Tuple::deserialize(&serialized).unwrap();

        assert_eq!(recovered.data(), &data);
        assert_eq!(recovered.header().xmin, 999);
        assert_eq!(recovered.header().data_len, data.len() as u16);
    }

    #[test]
    fn test_tuple_delete_flag() {
        let data = b"data".to_vec();
        let mut tuple = Tuple::new(data, 1);

        assert!(!tuple.is_deleted());

        tuple.set_deleted(true);
        assert!(tuple.is_deleted());

        // Serialize and deserialize preserves flag
        let serialized = tuple.serialize();
        let recovered = Tuple::deserialize(&serialized).unwrap();
        assert!(recovered.is_deleted());
    }

    #[test]
    fn test_tuple_deserialize_too_short() {
        let short_buf = [0u8; 4]; // Less than header size
        assert!(Tuple::deserialize(&short_buf).is_none());
    }

    #[test]
    fn test_tuple_deserialize_truncated_data() {
        let data = b"data".to_vec();
        let tuple = Tuple::new(data, 1);
        let serialized = tuple.serialize();

        // Truncate the serialized data
        let truncated = &serialized[..serialized.len() - 2];
        assert!(Tuple::deserialize(truncated).is_none());
    }

    #[test]
    fn test_tuple_with_header() {
        let header = TupleHeader {
            flags: TupleFlags(TupleFlags::HAS_NULLS),
            data_len: 5,
            xmin: 42,
            xmax: 0,
        };
        let data = b"hello".to_vec();

        let tuple = Tuple::with_header(header, data.clone());

        assert!(tuple.header().flags.has_nulls());
        assert_eq!(tuple.header().xmin, 42);
        assert_eq!(tuple.header().xmax, 0);
        assert_eq!(tuple.data(), &data);
    }

    #[test]
    fn test_tuple_visibility() {
        // Tuple created by txn 10, still live (xmax = 0)
        let header = TupleHeader::new(5, 10);
        assert!(header.is_visible(10)); // visible to creator
        assert!(header.is_visible(15)); // visible to later txns
        assert!(!header.is_visible(5)); // not visible to earlier txns

        // Tuple deleted by txn 20
        let header = TupleHeader::with_xmax(5, 10, 20);
        assert!(header.is_visible(15)); // visible between xmin and xmax
        assert!(!header.is_visible(25)); // not visible after xmax
        assert!(!header.is_visible(5)); // not visible before xmin
    }

    #[test]
    fn test_tuple_header_mut() {
        let data = b"data".to_vec();
        let mut tuple = Tuple::new(data, 1);

        tuple.header_mut().xmin = 999;
        assert_eq!(tuple.header().xmin, 999);

        tuple.header_mut().flags.set_locked(true);
        assert!(tuple.header().flags.is_locked());
    }
}

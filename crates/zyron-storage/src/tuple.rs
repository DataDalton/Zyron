//! Tuple representation and serialization.

use crate::txn::Snapshot;
use zyron_common::page::PageId;
use zyron_common::zerocopy::{AsBytes, FromBytes};

/// Packed 12-byte tuple header for single-memcpy serialization.
/// All fields stored in little-endian format.
#[repr(C, packed)]
struct PackedTupleHeader {
    flags: u16,
    data_len: u16,
    xmin: u32,
    xmax: u32,
}

const _: () = {
    assert!(std::mem::size_of::<PackedTupleHeader>() == 2 + 2 + 4 + 4);
    assert!(std::mem::align_of::<PackedTupleHeader>() == 1);
};

// Safety: PackedTupleHeader is repr(C, packed) with no padding (verified above).
unsafe impl AsBytes for PackedTupleHeader {}
unsafe impl FromBytes for PackedTupleHeader {}

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
            page_num: u64::MAX,
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

    /// Returns true if this tuple is visible to the given MVCC snapshot.
    ///
    /// Uses full MVCC visibility rules with active transaction tracking.
    /// Widens u32 xmin/xmax to u64 for the Snapshot check.
    #[inline]
    pub fn is_visible_to(&self, snapshot: &Snapshot) -> bool {
        snapshot.is_visible(self.xmin as u64, self.xmax as u64)
    }

    /// Serializes the header to bytes via single memcpy.
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let packed = PackedTupleHeader {
            flags: self.flags.0.to_le(),
            data_len: self.data_len.to_le(),
            xmin: self.xmin.to_le(),
            xmax: self.xmax.to_le(),
        };
        let mut buf = [0u8; Self::SIZE];
        packed.write_to(&mut buf, 0);
        buf
    }

    /// Deserializes the header from bytes via single unaligned read.
    pub fn from_bytes(buf: &[u8]) -> Self {
        let packed = PackedTupleHeader::read_from(buf, 0);
        Self {
            flags: TupleFlags(u16::from_le(packed.flags)),
            data_len: u16::from_le(packed.data_len),
            xmin: u32::from_le(packed.xmin),
            xmax: u32::from_le(packed.xmax),
        }
    }

    /// Deserializes the header from bytes without bounds checks.
    ///
    /// # Safety
    /// Caller must ensure buf has at least SIZE (12) bytes.
    #[inline(always)]
    pub unsafe fn from_bytes_unchecked(buf: &[u8]) -> Self {
        let packed = unsafe { std::ptr::read_unaligned(buf.as_ptr() as *const PackedTupleHeader) };
        Self {
            flags: TupleFlags(u16::from_le(packed.flags)),
            data_len: u16::from_le(packed.data_len),
            xmin: u32::from_le(packed.xmin),
            xmax: u32::from_le(packed.xmax),
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
    /// Tuple carries version metadata (version_id + deleted_at_version).
    pub const HAS_VERSION: u16 = 0x0010;

    /// Returns true if this tuple has version metadata.
    pub fn has_version(&self) -> bool {
        self.0 & Self::HAS_VERSION != 0
    }

    /// Sets the has_version flag.
    pub fn set_has_version(&mut self, val: bool) {
        if val {
            self.0 |= Self::HAS_VERSION;
        } else {
            self.0 &= !Self::HAS_VERSION;
        }
    }

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
    ///
    /// Panics if data exceeds 65535 bytes (u16::MAX), the maximum tuple data length.
    pub fn new(data: Vec<u8>, xmin: u32) -> Self {
        assert!(
            data.len() <= u16::MAX as usize,
            "tuple data length {} exceeds maximum {} bytes",
            data.len(),
            u16::MAX
        );
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
    #[allow(clippy::wrong_self_convention)]
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

// ---------------------------------------------------------------------------
// Versioned tuple header (28 bytes): base header + version tracking
// ---------------------------------------------------------------------------

/// Packed 28-byte versioned tuple header for single-memcpy serialization.
#[repr(C, packed)]
struct PackedVersionedTupleHeader {
    flags: u16,
    data_len: u16,
    xmin: u32,
    xmax: u32,
    version_id: u64,
    deleted_at_version: u64,
}

const _: () = {
    assert!(std::mem::size_of::<PackedVersionedTupleHeader>() == 28);
    assert!(std::mem::align_of::<PackedVersionedTupleHeader>() == 1);
};

unsafe impl AsBytes for PackedVersionedTupleHeader {}
unsafe impl FromBytes for PackedVersionedTupleHeader {}

/// Size of the versioned tuple header in bytes.
pub const VERSIONED_TUPLE_HEADER_SIZE: usize = 28;

/// Extended tuple header with version tracking.
///
/// Layout (28 bytes):
/// - base: 12 bytes (flags, data_len, xmin, xmax)
/// - version_id: 8 bytes (version that created this tuple)
/// - deleted_at_version: 8 bytes (version that deleted this tuple, 0 if live)
#[derive(Debug, Clone, Copy, Default)]
pub struct VersionedTupleHeader {
    /// Base tuple header containing flags, data_len, xmin, xmax.
    pub base: TupleHeader,
    /// Version that created this tuple.
    pub version_id: u64,
    /// Version that deleted this tuple (0 if still live).
    pub deleted_at_version: u64,
}

impl VersionedTupleHeader {
    /// Size of the versioned tuple header in bytes.
    pub const SIZE: usize = VERSIONED_TUPLE_HEADER_SIZE;

    /// Creates a new versioned tuple header.
    pub fn new(data_len: u16, xmin: u32, version_id: u64) -> Self {
        let mut base = TupleHeader::new(data_len, xmin);
        base.flags.set_has_version(true);
        Self {
            base,
            version_id,
            deleted_at_version: 0,
        }
    }

    /// Creates a versioned tuple header with all fields.
    pub fn with_deletion(
        data_len: u16,
        xmin: u32,
        xmax: u32,
        version_id: u64,
        deleted_at_version: u64,
    ) -> Self {
        let mut base = TupleHeader::with_xmax(data_len, xmin, xmax);
        base.flags.set_has_version(true);
        Self {
            base,
            version_id,
            deleted_at_version,
        }
    }

    /// Returns true if this tuple is visible at the given version.
    #[inline]
    pub fn is_visible_at_version(&self, target_version: u64) -> bool {
        self.version_id <= target_version
            && (self.deleted_at_version == 0 || self.deleted_at_version > target_version)
    }

    /// Serializes the header to bytes via single memcpy.
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let packed = PackedVersionedTupleHeader {
            flags: self.base.flags.0.to_le(),
            data_len: self.base.data_len.to_le(),
            xmin: self.base.xmin.to_le(),
            xmax: self.base.xmax.to_le(),
            version_id: self.version_id.to_le(),
            deleted_at_version: self.deleted_at_version.to_le(),
        };
        let mut buf = [0u8; Self::SIZE];
        packed.write_to(&mut buf, 0);
        buf
    }

    /// Deserializes the header from bytes via single unaligned read.
    pub fn from_bytes(buf: &[u8]) -> Self {
        let packed = PackedVersionedTupleHeader::read_from(buf, 0);
        Self {
            base: TupleHeader {
                flags: TupleFlags(u16::from_le(packed.flags)),
                data_len: u16::from_le(packed.data_len),
                xmin: u32::from_le(packed.xmin),
                xmax: u32::from_le(packed.xmax),
            },
            version_id: u64::from_le(packed.version_id),
            deleted_at_version: u64::from_le(packed.deleted_at_version),
        }
    }

    /// Deserializes the header from bytes without bounds checks.
    ///
    /// # Safety
    /// Caller must ensure buf has at least SIZE (28) bytes.
    #[inline(always)]
    pub unsafe fn from_bytes_unchecked(buf: &[u8]) -> Self {
        let packed =
            unsafe { std::ptr::read_unaligned(buf.as_ptr() as *const PackedVersionedTupleHeader) };
        Self {
            base: TupleHeader {
                flags: TupleFlags(u16::from_le(packed.flags)),
                data_len: u16::from_le(packed.data_len),
                xmin: u32::from_le(packed.xmin),
                xmax: u32::from_le(packed.xmax),
            },
            version_id: u64::from_le(packed.version_id),
            deleted_at_version: u64::from_le(packed.deleted_at_version),
        }
    }
}

/// Zero-copy view into versioned tuple data stored in a page buffer.
#[derive(Debug, Clone, Copy)]
pub struct VersionedTupleView<'a> {
    /// Versioned tuple header (copied, 28 bytes).
    pub header: VersionedTupleHeader,
    /// Reference to tuple data in the page buffer.
    pub data: &'a [u8],
}

impl<'a> VersionedTupleView<'a> {
    /// Creates a new versioned tuple view.
    #[inline]
    pub fn new(header: VersionedTupleHeader, data: &'a [u8]) -> Self {
        Self { header, data }
    }

    /// Converts this view to an owned Tuple by copying the data.
    #[inline]
    #[allow(clippy::wrong_self_convention)]
    pub fn to_owned(&self) -> Tuple {
        Tuple::with_header(self.header.base, self.data.to_vec())
    }

    /// Returns the tuple data length.
    #[inline]
    pub fn data_len(&self) -> usize {
        self.data.len()
    }

    /// Returns total size on disk (versioned header + data).
    #[inline]
    pub fn size_on_disk(&self) -> usize {
        VersionedTupleHeader::SIZE + self.data.len()
    }

    /// Returns true if this tuple is marked as deleted.
    #[inline]
    pub fn is_deleted(&self) -> bool {
        self.header.base.flags.is_deleted()
    }

    /// Returns true if this tuple is visible at the given version.
    #[inline]
    pub fn is_visible_at_version(&self, target_version: u64) -> bool {
        self.header.is_visible_at_version(target_version)
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

    #[test]
    fn test_has_version_flag() {
        let mut flags = TupleFlags::empty();
        assert!(!flags.has_version());

        flags.set_has_version(true);
        assert!(flags.has_version());
        assert!(!flags.is_deleted()); // other flags unaffected

        flags.set_has_version(false);
        assert!(!flags.has_version());
    }

    #[test]
    fn test_versioned_tuple_header_new() {
        let header = VersionedTupleHeader::new(100, 42, 7);

        assert_eq!(header.base.data_len, 100);
        assert_eq!(header.base.xmin, 42);
        assert_eq!(header.base.xmax, 0);
        assert!(header.base.flags.has_version());
        assert_eq!(header.version_id, 7);
        assert_eq!(header.deleted_at_version, 0);
    }

    #[test]
    fn test_versioned_tuple_header_roundtrip() {
        let header = VersionedTupleHeader::with_deletion(256, 12345, 67890, 999, 1050);

        let bytes = header.to_bytes();
        assert_eq!(bytes.len(), VersionedTupleHeader::SIZE);

        let recovered = VersionedTupleHeader::from_bytes(&bytes);

        assert_eq!(recovered.base.flags.0, header.base.flags.0);
        assert_eq!(recovered.base.data_len, 256);
        assert_eq!(recovered.base.xmin, 12345);
        assert_eq!(recovered.base.xmax, 67890);
        assert_eq!(recovered.version_id, 999);
        assert_eq!(recovered.deleted_at_version, 1050);
    }

    #[test]
    fn test_versioned_tuple_header_unchecked() {
        let header = VersionedTupleHeader::new(50, 100, 42);
        let bytes = header.to_bytes();

        let recovered = unsafe { VersionedTupleHeader::from_bytes_unchecked(&bytes) };

        assert_eq!(recovered.base.data_len, 50);
        assert_eq!(recovered.base.xmin, 100);
        assert_eq!(recovered.version_id, 42);
        assert_eq!(recovered.deleted_at_version, 0);
    }

    #[test]
    fn test_versioned_visibility_at_version() {
        // Live tuple created at version 10
        let header = VersionedTupleHeader::new(5, 1, 10);
        assert!(header.is_visible_at_version(10)); // visible at creation version
        assert!(header.is_visible_at_version(100)); // visible at later versions
        assert!(!header.is_visible_at_version(5)); // not visible before creation

        // Tuple created at version 10, deleted at version 20
        let header = VersionedTupleHeader::with_deletion(5, 1, 2, 10, 20);
        assert!(header.is_visible_at_version(10)); // visible at creation
        assert!(header.is_visible_at_version(15)); // visible between create and delete
        assert!(header.is_visible_at_version(19)); // visible just before deletion
        assert!(!header.is_visible_at_version(20)); // not visible at deletion version
        assert!(!header.is_visible_at_version(25)); // not visible after deletion
        assert!(!header.is_visible_at_version(5)); // not visible before creation
    }

    #[test]
    fn test_versioned_tuple_view() {
        let header = VersionedTupleHeader::new(5, 1, 42);
        let data = b"hello";
        let view = VersionedTupleView::new(header, data);

        assert_eq!(view.data_len(), 5);
        assert_eq!(view.size_on_disk(), VersionedTupleHeader::SIZE + 5);
        assert!(!view.is_deleted());
        assert!(view.is_visible_at_version(42));
        assert!(!view.is_visible_at_version(41));

        let owned = view.to_owned();
        assert_eq!(owned.data(), data);
        assert_eq!(owned.header().xmin, 1);
    }

    #[test]
    fn test_versioned_header_size_constant() {
        assert_eq!(VERSIONED_TUPLE_HEADER_SIZE, 28);
        assert_eq!(VersionedTupleHeader::SIZE, 28);
        assert_eq!(
            VersionedTupleHeader::SIZE,
            TupleHeader::SIZE + std::mem::size_of::<u64>() * 2
        );
    }
}

//! Storage agnostic row identity.
//!
//! Every subsystem that needs to point at a row (indexes, triggers, CDC,
//! constraints, locking, undo) addresses it through RowLocator instead of
//! assuming heap pages. Heap rows live in mutable pages, columnar rows in
//! immutable .zyr segments keyed by stable sys_rowid, lake rows in
//! manifest tracked files keyed by ordinal position.

use crate::page::PageId;

/// Identifies one row independent of the storage format holding it
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RowLocator {
    /// Mutable heap page slot
    Heap { page: PageId, slot: u16 },
    /// Immutable .zyr segment row, sys_rowid survives merges
    Columnar { file_id: u64, sys_rowid: u64 },
    /// Lake data file row addressed by ordinal within the file
    Lake { file_id: u64, ordinal: u64 },
}

impl RowLocator {
    /// Byte width of the fixed codecs below, one tag byte plus two u64 words
    pub const ENCODED_LEN: usize = 17;

    const TAG_HEAP: u8 = 0;
    const TAG_COLUMNAR: u8 = 1;
    const TAG_LAKE: u8 = 2;

    /// The (tag, a, b) words of the fixed codec. Heap drops its file_id,
    /// callers normalize it to 0 before encoding and re-stamp from context
    /// after decoding
    #[inline]
    fn codec_words(&self) -> (u8, u64, u64) {
        match *self {
            RowLocator::Heap { page, slot } => (Self::TAG_HEAP, page.page_num, slot as u64),
            RowLocator::Columnar { file_id, sys_rowid } => (Self::TAG_COLUMNAR, file_id, sys_rowid),
            RowLocator::Lake { file_id, ordinal } => (Self::TAG_LAKE, file_id, ordinal),
        }
    }

    /// Appends the order-preserving big-endian form used as a btree composite
    /// key suffix. Fixed width, so all-0x00 and all-0xFF suffixes bracket
    /// every real suffix inside one value's key range
    #[inline]
    pub fn append_key_suffix(&self, buf: &mut Vec<u8>) {
        let (tag, a, b) = self.codec_words();
        buf.push(tag);
        buf.extend_from_slice(&a.to_be_bytes());
        buf.extend_from_slice(&b.to_be_bytes());
    }

    /// Writes the little-endian payload form, exactly ENCODED_LEN bytes
    #[inline]
    pub fn write_payload(&self, out: &mut [u8]) {
        let (tag, a, b) = self.codec_words();
        out[0] = tag;
        out[1..9].copy_from_slice(&a.to_le_bytes());
        out[9..17].copy_from_slice(&b.to_le_bytes());
    }

    /// Reads the payload form. Heap decodes with file_id 0, callers re-stamp
    /// the owning heap file from context. Returns None on a short buffer or
    /// an unknown tag
    #[inline]
    pub fn read_payload(buf: &[u8]) -> Option<RowLocator> {
        if buf.len() < Self::ENCODED_LEN {
            return None;
        }
        let a = u64::from_le_bytes(buf[1..9].try_into().ok()?);
        let b = u64::from_le_bytes(buf[9..17].try_into().ok()?);
        match buf[0] {
            Self::TAG_HEAP => Some(RowLocator::Heap {
                page: PageId::new(0, a),
                slot: b as u16,
            }),
            Self::TAG_COLUMNAR => Some(RowLocator::Columnar {
                file_id: a,
                sys_rowid: b,
            }),
            Self::TAG_LAKE => Some(RowLocator::Lake {
                file_id: a,
                ordinal: b,
            }),
            _ => None,
        }
    }

    #[inline]
    pub fn is_heap(&self) -> bool {
        matches!(self, RowLocator::Heap { .. })
    }

    #[inline]
    pub fn is_columnar(&self) -> bool {
        matches!(self, RowLocator::Columnar { .. })
    }

    #[inline]
    pub fn is_lake(&self) -> bool {
        matches!(self, RowLocator::Lake { .. })
    }

    /// The (file_id, sys_rowid) pair for a columnar row
    #[inline]
    pub fn columnar_pair(&self) -> Option<(u64, u64)> {
        match *self {
            RowLocator::Columnar { file_id, sys_rowid } => Some((file_id, sys_rowid)),
            _ => None,
        }
    }
}

impl std::fmt::Display for RowLocator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RowLocator::Heap { page, slot } => write!(f, "heap:{page}:{slot}"),
            RowLocator::Columnar { file_id, sys_rowid } => {
                write!(f, "columnar:{file_id}:{sys_rowid}")
            }
            RowLocator::Lake { file_id, ordinal } => write!(f, "lake:{file_id}:{ordinal}"),
        }
    }
}

//! The envelope carried by page-resident formats.
//!
//! Heap pages, B+tree pages, and free space map pages are fixed-size records
//! inside a larger file rather than files of their own. They carry the same
//! identity the file envelope carries, magic and major and minor version,
//! packed into the nine reserved bytes at the tail of `PageHeader` instead of
//! a header of their own, so a page keeps every byte of its payload capacity.
//!
//! ```text
//! [0..4)  magic          format-kind identifier, 'ZHEP', 'ZBPI', 'ZBPL', 'ZFSM'
//! [4..6)  version_major  u16 little endian
//! [6..8)  version_minor  u16 little endian
//! [8]     flags          per-page flag byte
//! ```
//!
//! Integrity comes from the page checksum, which covers every page byte
//! except its own four, the stamp included. A page has one checksum rather
//! than a header checksum and a footer checksum because it is written and
//! read as a single unit

use super::envelope::EnvelopeError;
use super::kind::FormatKind;
use super::version::FormatVersion;

/// Bytes the page stamp occupies
pub const FORMAT_STAMP_LEN: usize = 9;

/// Page stamp flag bits, distinct from `PageFlags`, which describe the page
/// contents rather than its encoding
pub mod stamp_flags {
    /// The page body is compressed
    pub const COMPRESSED: u8 = 1 << 0;
    /// The page body is encrypted
    pub const ENCRYPTED: u8 = 1 << 1;
}

/// The identity stamp a page-resident format carries
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FormatStamp {
    pub kind: FormatKind,
    pub version: FormatVersion,
    pub flags: u8,
}

impl FormatStamp {
    pub const fn new(kind: FormatKind, version: FormatVersion) -> Self {
        Self {
            kind,
            version,
            flags: 0,
        }
    }

    /// Packs the stamp into its nine bytes
    #[inline]
    pub fn to_bytes(self) -> [u8; FORMAT_STAMP_LEN] {
        let mut out = [0u8; FORMAT_STAMP_LEN];
        out[0..4].copy_from_slice(&self.kind.magic());
        out[4..8].copy_from_slice(&self.version.to_le_bytes());
        out[8] = self.flags;
        out
    }

    /// The same packing in a const context, so a format can hold its stamp
    /// bytes as a constant and compare against them without building one
    pub const fn to_bytes_const(self) -> [u8; FORMAT_STAMP_LEN] {
        let magic = self.kind.magic();
        let version = self.version.to_le_bytes();
        [
            magic[0], magic[1], magic[2], magic[3], version[0], version[1], version[2], version[3],
            self.flags,
        ]
    }

    /// Reads a stamp back, refusing bytes that address no registered format.
    ///
    /// An all-zero stamp reads as `UnknownMagic` with a zero magic, which is
    /// what an unstamped page looks like, so a caller that wants to treat a
    /// blank page as blank checks `is_blank` first
    #[inline]
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, EnvelopeError> {
        if bytes.len() < FORMAT_STAMP_LEN {
            return Err(EnvelopeError::TooShort {
                have: bytes.len(),
                need: FORMAT_STAMP_LEN,
            });
        }
        let magic = [bytes[0], bytes[1], bytes[2], bytes[3]];
        let Some(kind) = FormatKind::from_magic(magic) else {
            return Err(EnvelopeError::UnknownMagic { magic });
        };
        Ok(Self {
            kind,
            version: FormatVersion::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            flags: bytes[8],
        })
    }

    /// True when the nine bytes are all zero, which is an unallocated page
    /// rather than a corrupted one
    #[inline]
    pub fn is_blank(bytes: &[u8]) -> bool {
        bytes.len() >= FORMAT_STAMP_LEN && bytes[..FORMAT_STAMP_LEN].iter().all(|b| *b == 0)
    }

    /// Reads only the magic and version, which is what a page read does
    /// before deciding whether any migration applies
    #[inline]
    pub fn peek(bytes: &[u8]) -> Result<(FormatKind, FormatVersion), EnvelopeError> {
        let stamp = Self::from_bytes(bytes)?;
        Ok((stamp.kind, stamp.version))
    }

    #[inline]
    pub const fn is_compressed(&self) -> bool {
        self.flags & stamp_flags::COMPRESSED != 0
    }

    #[inline]
    pub const fn is_encrypted(&self) -> bool {
        self.flags & stamp_flags::ENCRYPTED != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stamp_round_trips_for_every_page_format() {
        for kind in [
            FormatKind::HeapPage,
            FormatKind::BTreeInternal,
            FormatKind::BTreeLeaf,
            FormatKind::Fsm,
        ] {
            let stamp = FormatStamp {
                kind,
                version: FormatVersion::new(3, 9),
                flags: stamp_flags::COMPRESSED,
            };
            let bytes = stamp.to_bytes();
            assert_eq!(bytes.len(), FORMAT_STAMP_LEN);
            let read = FormatStamp::from_bytes(&bytes).expect("reads back");
            assert_eq!(read, stamp);
            assert!(read.is_compressed());
            assert!(!read.is_encrypted());
        }
    }

    #[test]
    fn test_blank_stamp_is_recognized_and_not_decoded() {
        let blank = [0u8; FORMAT_STAMP_LEN];
        assert!(FormatStamp::is_blank(&blank));
        assert!(matches!(
            FormatStamp::from_bytes(&blank),
            Err(EnvelopeError::UnknownMagic { .. })
        ));
    }

    #[test]
    fn test_short_stamp_is_refused() {
        assert!(matches!(
            FormatStamp::from_bytes(&[0u8; 4]),
            Err(EnvelopeError::TooShort { .. })
        ));
    }

    #[test]
    fn test_peek_matches_full_read() {
        let stamp = FormatStamp::new(FormatKind::Fsm, FormatVersion::new(1, 4));
        let bytes = stamp.to_bytes();
        assert_eq!(
            FormatStamp::peek(&bytes).expect("peeks"),
            (FormatKind::Fsm, FormatVersion::new(1, 4))
        );
    }
}

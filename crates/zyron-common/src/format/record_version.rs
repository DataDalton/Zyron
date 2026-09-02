//! Per-record version tags.
//!
//! A file envelope names the version of the container. Records inside a
//! container written across a rolling upgrade can be of different versions,
//! so WAL segments, Raft log entries, replication apply entries, and audit
//! chain events tag each record with its own version. The tag is one byte
//! where 255 versions are enough and two bytes where they are not, so the
//! cost per record stays at the smallest thing that can carry the answer

use super::envelope::EnvelopeError;
use super::version::FormatVersion;

/// A one-byte record version tag.
///
/// Value 0 is reserved and never written, so a zero-filled region reads as
/// absent rather than as version zero
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RecordVersion(u8);

/// The reserved tag value that marks an unwritten record slot
pub const RECORD_VERSION_ABSENT: u8 = 0;

impl RecordVersion {
    /// The first version a record format carries
    pub const V1: RecordVersion = RecordVersion(1);

    /// Builds a tag, refusing the reserved zero
    pub const fn new(value: u8) -> Option<RecordVersion> {
        if value == RECORD_VERSION_ABSENT {
            None
        } else {
            Some(RecordVersion(value))
        }
    }

    #[inline]
    pub const fn get(self) -> u8 {
        self.0
    }

    /// Reads a tag from the first byte of a record, refusing a zero byte
    #[inline]
    pub fn read(bytes: &[u8]) -> Result<RecordVersion, EnvelopeError> {
        let Some(first) = bytes.first().copied() else {
            return Err(EnvelopeError::TooShort { have: 0, need: 1 });
        };
        match RecordVersion::new(first) {
            Some(tag) => Ok(tag),
            None => Err(EnvelopeError::UnknownMagic { magic: [0; 4] }),
        }
    }

    /// The equivalent format version, which is a minor step inside major 1
    /// for record streams, so a record tag orders against a file version
    #[inline]
    pub const fn as_format_version(self) -> FormatVersion {
        FormatVersion::new(1, self.0 as u16)
    }
}

/// A two-byte record version tag, for record formats that outlive 255
/// versions
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WideRecordVersion(u16);

impl WideRecordVersion {
    pub const V1: WideRecordVersion = WideRecordVersion(1);

    pub const fn new(value: u16) -> Option<WideRecordVersion> {
        if value == 0 {
            None
        } else {
            Some(WideRecordVersion(value))
        }
    }

    #[inline]
    pub const fn get(self) -> u16 {
        self.0
    }

    #[inline]
    pub const fn to_le_bytes(self) -> [u8; 2] {
        self.0.to_le_bytes()
    }

    #[inline]
    pub fn read(bytes: &[u8]) -> Result<WideRecordVersion, EnvelopeError> {
        if bytes.len() < 2 {
            return Err(EnvelopeError::TooShort {
                have: bytes.len(),
                need: 2,
            });
        }
        match WideRecordVersion::new(u16::from_le_bytes([bytes[0], bytes[1]])) {
            Some(tag) => Ok(tag),
            None => Err(EnvelopeError::UnknownMagic { magic: [0; 4] }),
        }
    }

    #[inline]
    pub const fn as_format_version(self) -> FormatVersion {
        FormatVersion::new(1, self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_is_reserved() {
        assert!(RecordVersion::new(0).is_none());
        assert!(WideRecordVersion::new(0).is_none());
        assert!(RecordVersion::read(&[0]).is_err());
        assert!(WideRecordVersion::read(&[0, 0]).is_err());
    }

    #[test]
    fn test_tags_round_trip() {
        for raw in 1u8..=255 {
            let tag = RecordVersion::new(raw).expect("nonzero");
            assert_eq!(tag.get(), raw);
            assert_eq!(RecordVersion::read(&[raw]).expect("reads"), tag);
        }
        let wide = WideRecordVersion::new(4096).expect("nonzero");
        assert_eq!(
            WideRecordVersion::read(&wide.to_le_bytes()).expect("reads"),
            wide
        );
    }

    #[test]
    fn test_tags_order_against_format_versions() {
        assert!(
            RecordVersion::new(3).expect("nonzero").as_format_version()
                > RecordVersion::V1.as_format_version()
        );
        assert_eq!(
            WideRecordVersion::V1.as_format_version(),
            FormatVersion::new(1, 1)
        );
    }

    #[test]
    fn test_empty_input_is_refused() {
        assert!(matches!(
            RecordVersion::read(&[]),
            Err(EnvelopeError::TooShort { .. })
        ));
    }
}

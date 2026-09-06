//! Per-record version tags.
//!
//! A file envelope names the version of the container. Records inside a
//! container written across a rolling upgrade can be of different versions,
//! so WAL segments, Raft log entries, replication apply entries, and audit
//! chain events tag each record with its own version. The tag is one byte
//! where 254 versions are enough and two bytes where they are not, so the
//! cost per record stays at the smallest thing that can carry the answer.
//!
//! Two byte values are reserved and never name a version:
//!
//! - `0` marks an unwritten slot, so a zero filled region reads as absent
//!   rather than as version zero
//! - `255` marks the wide escape, meaning the version did not fit one byte
//!   and a `WideRecordVersion` carries it instead
//!
//! The escape is what keeps the one byte tag from being a dead end. Without
//! it a stream reaching 255 would have no value left to say "the layout
//! changed again", and moving to a wider tag would need a flag day where
//! every reader is replaced at once. With it, a reader that predates the
//! change still recognizes the escape and refuses the record by name
//! instead of misparsing it. Nothing writes the escape today, and the
//! inline range stays 1 through 254

use super::envelope::EnvelopeError;
use super::version::FormatVersion;

/// A one-byte record version tag, holding 1 through 254.
///
/// Value 0 is reserved and never written, so a zero-filled region reads as
/// absent rather than as version zero. Value 255 is reserved as the wide
/// escape, so a stream that outgrows one byte has a value left to say so
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RecordVersion(u8);

/// The reserved tag value that marks an unwritten record slot
pub const RECORD_VERSION_ABSENT: u8 = 0;

/// The reserved tag value meaning the version did not fit one byte.
///
/// A record carrying this byte is followed by a `WideRecordVersion` at a
/// position its own format defines. A length prefixed record puts it
/// immediately after the escape, which is what `RecordTag::read` expects. A
/// record with a fixed header, the WAL entry being the one in the tree,
/// names its own position when it gets there
pub const RECORD_VERSION_WIDE_ESCAPE: u8 = 255;

/// The highest version that fits the one byte tag
pub const RECORD_VERSION_MAX_INLINE: u8 = RECORD_VERSION_WIDE_ESCAPE - 1;

impl RecordVersion {
    /// The first version a record format carries
    pub const V1: RecordVersion = RecordVersion(1);

    /// Builds a tag, refusing both reserved values
    pub const fn new(value: u8) -> Option<RecordVersion> {
        if value == RECORD_VERSION_ABSENT || value == RECORD_VERSION_WIDE_ESCAPE {
            None
        } else {
            Some(RecordVersion(value))
        }
    }

    #[inline]
    pub const fn get(self) -> u8 {
        self.0
    }

    /// Whether a raw byte is the wide escape rather than a version
    #[inline]
    pub const fn is_wide_escape(value: u8) -> bool {
        value == RECORD_VERSION_WIDE_ESCAPE
    }

    /// Reads an inline tag from the first byte of a record.
    ///
    /// The wide escape is refused here rather than resolved, because this
    /// call has only the one byte to work with. A caller that can read the
    /// bytes behind it uses `RecordTag::read` instead
    #[inline]
    pub fn read(bytes: &[u8]) -> Result<RecordVersion, EnvelopeError> {
        let Some(first) = bytes.first().copied() else {
            return Err(EnvelopeError::TooShort { have: 0, need: 1 });
        };
        match RecordVersion::new(first) {
            Some(tag) => Ok(tag),
            None => Err(EnvelopeError::BadRecordVersion { found: first }),
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
            None => Err(EnvelopeError::BadRecordVersion { found: 0 }),
        }
    }

    #[inline]
    pub const fn as_format_version(self) -> FormatVersion {
        FormatVersion::new(1, self.0)
    }
}

/// What the leading version bytes of a record decode to.
///
/// A record stream that can read the bytes behind its tag resolves the wide
/// escape through this rather than through `RecordVersion::read`, which sees
/// only the one byte and has to refuse it
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordTag {
    /// The version fit the single byte
    Inline(RecordVersion),
    /// The byte was the escape and a two byte version followed it
    Wide(WideRecordVersion),
}

impl RecordTag {
    /// Reads a tag, resolving the wide escape by reading the two bytes
    /// behind it. Returns the tag and how many bytes it consumed, one for an
    /// inline version and three for a wide one
    #[inline]
    pub fn read(bytes: &[u8]) -> Result<(RecordTag, usize), EnvelopeError> {
        let Some(first) = bytes.first().copied() else {
            return Err(EnvelopeError::TooShort { have: 0, need: 1 });
        };
        if RecordVersion::is_wide_escape(first) {
            let wide = WideRecordVersion::read(bytes.get(1..).unwrap_or(&[]))?;
            return Ok((RecordTag::Wide(wide), 3));
        }
        match RecordVersion::new(first) {
            Some(tag) => Ok((RecordTag::Inline(tag), 1)),
            None => Err(EnvelopeError::BadRecordVersion { found: first }),
        }
    }

    /// The version this tag names, widened so an inline and a wide tag order
    /// against each other
    #[inline]
    pub const fn as_format_version(self) -> FormatVersion {
        match self {
            RecordTag::Inline(tag) => tag.as_format_version(),
            RecordTag::Wide(tag) => tag.as_format_version(),
        }
    }

    /// Bytes this tag occupies when written
    #[inline]
    pub const fn encoded_len(self) -> usize {
        match self {
            RecordTag::Inline(_) => 1,
            RecordTag::Wide(_) => 3,
        }
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

    /// The escape is what stops the one byte tag being a dead end, so it can
    /// never be handed out as an ordinary version
    #[test]
    fn test_the_wide_escape_is_reserved() {
        assert!(RecordVersion::new(RECORD_VERSION_WIDE_ESCAPE).is_none());
        assert!(RecordVersion::new(RECORD_VERSION_MAX_INLINE).is_some());
        assert_eq!(RECORD_VERSION_MAX_INLINE, 254);
        assert!(RecordVersion::is_wide_escape(RECORD_VERSION_WIDE_ESCAPE));
        assert!(!RecordVersion::is_wide_escape(RECORD_VERSION_MAX_INLINE));

        // A reader with only the one byte refuses it by name rather than
        // reading it as version 255
        let err = RecordVersion::read(&[RECORD_VERSION_WIDE_ESCAPE]).expect_err("refused");
        assert!(matches!(
            err,
            EnvelopeError::BadRecordVersion { found: 255 }
        ));
        assert!(err.to_string().contains("wide escape"), "{err}");
    }

    /// A stream that can read behind the escape resolves it to the wide tag
    /// and reports the three bytes it consumed
    #[test]
    fn test_the_escape_resolves_to_a_wide_tag() {
        let wide = WideRecordVersion::new(300).expect("nonzero");
        let mut bytes = vec![RECORD_VERSION_WIDE_ESCAPE];
        bytes.extend_from_slice(&wide.to_le_bytes());
        let (tag, used) = RecordTag::read(&bytes).expect("reads");
        assert_eq!(tag, RecordTag::Wide(wide));
        assert_eq!(used, 3);
        assert_eq!(tag.encoded_len(), 3);

        let (inline, used) = RecordTag::read(&[2]).expect("reads");
        assert_eq!(inline, RecordTag::Inline(RecordVersion::new(2).unwrap()));
        assert_eq!(used, 1);

        // A wide tag orders above every inline one, so the two are
        // comparable across the change that introduces the escape
        assert!(
            RecordTag::Wide(wide).as_format_version()
                > RecordTag::Inline(RecordVersion::new(RECORD_VERSION_MAX_INLINE).unwrap())
                    .as_format_version()
        );
    }

    /// An escape with nothing behind it is truncation, not a version
    #[test]
    fn test_a_truncated_escape_is_refused() {
        assert!(matches!(
            RecordTag::read(&[RECORD_VERSION_WIDE_ESCAPE]),
            Err(EnvelopeError::TooShort { .. })
        ));
        assert!(matches!(
            RecordTag::read(&[RECORD_VERSION_WIDE_ESCAPE, 1]),
            Err(EnvelopeError::TooShort { .. })
        ));
    }

    #[test]
    fn test_tags_round_trip() {
        for raw in 1u8..=RECORD_VERSION_MAX_INLINE {
            let tag = RecordVersion::new(raw).expect("usable");
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

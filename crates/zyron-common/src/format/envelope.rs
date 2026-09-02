//! The universal on-disk format envelope.
//!
//! Every standalone file Zyron writes opens with a fixed 20-byte header and
//! closes with a 4-byte footer checksum. The header identifies the format
//! kind by magic and the format version by major and minor, so a reader can
//! decide what to do with a file having read 8 bytes of it, and a file whose
//! version the binary has no reader for is refused instead of guessed at.
//!
//! ```text
//! [0..4)   magic            format-kind identifier, 'ZWAL', 'ZHEP', ...
//! [4..6)   version_major    u16 little endian
//! [6..8)   version_minor    u16 little endian
//! [8..12)  header_length    u32 little endian, 20 plus any extension bytes
//! [12..16) flags            u32 little endian, compression / encryption / per-format
//! [16..20) header_checksum  u32 little endian over [0..16) and the extension
//! [20..header_length)       per-format header extension, optional
//! [header_length..len-4)    body
//! [len-4..len)             footer_checksum  u32 little endian over the body
//! ```
//!
//! Both checksums are required. The header checksum catches a torn or
//! corrupted header before any length in it is trusted, and the footer
//! checksum covers the body

use crate::checksum::hash32;

use super::kind::FormatKind;
use super::version::FormatVersion;

/// Bytes of the fixed part of the envelope header
pub const ENVELOPE_HEADER_LEN: usize = 20;

/// Bytes of the trailing body checksum
pub const ENVELOPE_FOOTER_LEN: usize = 4;

/// Smallest byte count that can hold an envelope, an empty body included
pub const ENVELOPE_MIN_LEN: usize = ENVELOPE_HEADER_LEN + ENVELOPE_FOOTER_LEN;

/// Bytes a reader needs before it can name the format kind and version
pub const ENVELOPE_PEEK_LEN: usize = 8;

/// Envelope flag bits.
///
/// The low byte is reserved for substrate-wide meanings, the upper three
/// bytes belong to the format that set them
pub mod flags {
    /// The body is compressed
    pub const COMPRESSED: u32 = 1 << 0;
    /// The body is encrypted
    pub const ENCRYPTED: u32 = 1 << 1;
    /// The body was written by a downgrade-write, for export to an older
    /// cluster, rather than by the current writer
    pub const DOWNGRADE_WRITE: u32 = 1 << 2;
    /// Mask of the bits the substrate owns
    pub const SUBSTRATE_MASK: u32 = 0x0000_00FF;
    /// Mask of the bits a format may set for itself
    pub const PER_FORMAT_MASK: u32 = 0xFFFF_FF00;
}

/// What went wrong reading an envelope
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnvelopeError {
    /// The slice is shorter than the smallest possible envelope
    TooShort { have: usize, need: usize },
    /// The leading four bytes address no registered format
    UnknownMagic { magic: [u8; 4] },
    /// The leading four bytes address a different format than the caller
    /// asked for
    MagicMismatch {
        expected: FormatKind,
        found: [u8; 4],
    },
    /// The declared header length does not fit the slice or is below the
    /// fixed minimum
    BadHeaderLength { declared: u32, have: usize },
    /// The header checksum does not match the header bytes
    HeaderChecksumMismatch { stored: u32, computed: u32 },
    /// The footer checksum does not match the body bytes
    FooterChecksumMismatch { stored: u32, computed: u32 },
    /// The version is outside the reader window this binary carries
    UnknownVersion {
        kind: FormatKind,
        found: FormatVersion,
        oldest_supported: FormatVersion,
        newest_supported: FormatVersion,
    },
}

impl std::fmt::Display for EnvelopeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EnvelopeError::TooShort { have, need } => write!(
                f,
                "format envelope truncated, {have} bytes present and {need} needed"
            ),
            EnvelopeError::UnknownMagic { magic } => write!(
                f,
                "format magic {} addresses no registered format",
                printable_magic(magic)
            ),
            EnvelopeError::MagicMismatch { expected, found } => write!(
                f,
                "expected a {} file with magic {}, found magic {}",
                expected,
                printable_magic(&expected.magic()),
                printable_magic(found)
            ),
            EnvelopeError::BadHeaderLength { declared, have } => write!(
                f,
                "format envelope declares a {declared} byte header, which does not fit \
                 the {have} bytes present"
            ),
            EnvelopeError::HeaderChecksumMismatch { stored, computed } => write!(
                f,
                "format envelope header checksum mismatch, stored {stored:#010x} \
                 computed {computed:#010x}"
            ),
            EnvelopeError::FooterChecksumMismatch { stored, computed } => write!(
                f,
                "format envelope body checksum mismatch, stored {stored:#010x} \
                 computed {computed:#010x}"
            ),
            EnvelopeError::UnknownVersion {
                kind,
                found,
                oldest_supported,
                newest_supported,
            } => write!(
                f,
                "{kind} file is at format version {found}, which this binary cannot read. \
                 Readers are carried for {oldest_supported} through {newest_supported}. \
                 Upgrade through a release that still reads {found} to move the file \
                 forward first"
            ),
        }
    }
}

impl std::error::Error for EnvelopeError {}

/// Renders a magic for an error message, falling back to hex when the bytes
/// are not printable
pub fn printable_magic(magic: &[u8; 4]) -> String {
    if magic.iter().all(|b| b.is_ascii_graphic()) {
        format!("'{}'", String::from_utf8_lossy(magic))
    } else {
        format!("{magic:02x?}")
    }
}

/// The parsed fixed header of an envelope
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnvelopeHeader {
    pub kind: FormatKind,
    pub version: FormatVersion,
    pub header_length: u32,
    pub flags: u32,
    pub header_checksum: u32,
}

impl EnvelopeHeader {
    /// Byte offset of the body, which is the declared header length
    #[inline]
    pub const fn body_offset(&self) -> usize {
        self.header_length as usize
    }

    #[inline]
    pub const fn is_compressed(&self) -> bool {
        self.flags & flags::COMPRESSED != 0
    }

    #[inline]
    pub const fn is_encrypted(&self) -> bool {
        self.flags & flags::ENCRYPTED != 0
    }

    #[inline]
    pub const fn is_downgrade_write(&self) -> bool {
        self.flags & flags::DOWNGRADE_WRITE != 0
    }
}

/// What a decoded envelope hands back, the header plus borrowed body bytes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Envelope<'a> {
    pub header: EnvelopeHeader,
    /// The per-format header extension, empty when the header is the fixed
    /// 20 bytes
    pub extension: &'a [u8],
    pub body: &'a [u8],
    pub footer_checksum: u32,
}

/// Reads the format kind and version out of the first 8 bytes.
///
/// This is the hot path. Opening a file peeks these bytes, compares the
/// version against the writer's current version, and takes the current
/// reader path on a match without touching the rest of the envelope
#[inline]
pub fn peek(bytes: &[u8]) -> Result<(FormatKind, FormatVersion), EnvelopeError> {
    if bytes.len() < ENVELOPE_PEEK_LEN {
        return Err(EnvelopeError::TooShort {
            have: bytes.len(),
            need: ENVELOPE_PEEK_LEN,
        });
    }
    let magic = [bytes[0], bytes[1], bytes[2], bytes[3]];
    let Some(kind) = FormatKind::from_magic(magic) else {
        return Err(EnvelopeError::UnknownMagic { magic });
    };
    let version = FormatVersion::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    Ok((kind, version))
}

/// Reads the version of a file already known to be of one kind, refusing a
/// file whose magic says otherwise
#[inline]
pub fn peek_version_of(bytes: &[u8], expected: FormatKind) -> Result<FormatVersion, EnvelopeError> {
    if bytes.len() < ENVELOPE_PEEK_LEN {
        return Err(EnvelopeError::TooShort {
            have: bytes.len(),
            need: ENVELOPE_PEEK_LEN,
        });
    }
    let magic = [bytes[0], bytes[1], bytes[2], bytes[3]];
    if magic != expected.magic() {
        return Err(EnvelopeError::MagicMismatch {
            expected,
            found: magic,
        });
    }
    Ok(FormatVersion::from_le_bytes([
        bytes[4], bytes[5], bytes[6], bytes[7],
    ]))
}

/// Computes the header checksum over the fixed bytes ahead of the checksum
/// field and any extension bytes behind it
#[inline]
fn header_checksum_of(fixed: &[u8], extension: &[u8]) -> u32 {
    let mut hasher = crate::checksum::Hasher::new();
    hasher.update(&fixed[..16]);
    hasher.update(extension);
    hasher.finish32()
}

/// Writes an envelope header into a 20-byte buffer.
///
/// The extension bytes are not written here, the caller appends them, but
/// they are folded into the header checksum so a torn extension is caught
/// with the header rather than with the body
pub fn encode_header(
    kind: FormatKind,
    version: FormatVersion,
    flags: u32,
    extension: &[u8],
) -> [u8; ENVELOPE_HEADER_LEN] {
    let mut header = [0u8; ENVELOPE_HEADER_LEN];
    header[0..4].copy_from_slice(&kind.magic());
    header[4..8].copy_from_slice(&version.to_le_bytes());
    let header_length = (ENVELOPE_HEADER_LEN + extension.len()) as u32;
    header[8..12].copy_from_slice(&header_length.to_le_bytes());
    header[12..16].copy_from_slice(&flags.to_le_bytes());
    let checksum = header_checksum_of(&header, extension);
    header[16..20].copy_from_slice(&checksum.to_le_bytes());
    header
}

/// Wraps a body in a complete envelope
pub fn encode(kind: FormatKind, version: FormatVersion, body: &[u8]) -> Vec<u8> {
    encode_with(kind, version, 0, &[], body)
}

/// Wraps a body in a complete envelope, with flags and a per-format header
/// extension
pub fn encode_with(
    kind: FormatKind,
    version: FormatVersion,
    flags: u32,
    extension: &[u8],
    body: &[u8],
) -> Vec<u8> {
    let mut out = Vec::with_capacity(
        ENVELOPE_HEADER_LEN + extension.len() + body.len() + ENVELOPE_FOOTER_LEN,
    );
    out.extend_from_slice(&encode_header(kind, version, flags, extension));
    out.extend_from_slice(extension);
    out.extend_from_slice(body);
    let footer = hash32(body);
    out.extend_from_slice(&footer.to_le_bytes());
    out
}

/// Parses the fixed header and verifies its checksum.
///
/// The extension is needed for the checksum, so the whole header has to be
/// present. The body is not touched
pub fn decode_header(bytes: &[u8]) -> Result<(EnvelopeHeader, &[u8]), EnvelopeError> {
    if bytes.len() < ENVELOPE_HEADER_LEN {
        return Err(EnvelopeError::TooShort {
            have: bytes.len(),
            need: ENVELOPE_HEADER_LEN,
        });
    }
    let magic = [bytes[0], bytes[1], bytes[2], bytes[3]];
    let Some(kind) = FormatKind::from_magic(magic) else {
        return Err(EnvelopeError::UnknownMagic { magic });
    };
    let version = FormatVersion::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    let header_length = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    if (header_length as usize) < ENVELOPE_HEADER_LEN || header_length as usize > bytes.len() {
        return Err(EnvelopeError::BadHeaderLength {
            declared: header_length,
            have: bytes.len(),
        });
    }
    let flags = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
    let stored = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
    let extension = &bytes[ENVELOPE_HEADER_LEN..header_length as usize];
    let computed = header_checksum_of(&bytes[..ENVELOPE_HEADER_LEN], extension);
    if computed != stored {
        return Err(EnvelopeError::HeaderChecksumMismatch { stored, computed });
    }
    Ok((
        EnvelopeHeader {
            kind,
            version,
            header_length,
            flags,
            header_checksum: stored,
        },
        extension,
    ))
}

/// Parses a whole envelope and verifies both checksums
pub fn decode(bytes: &[u8]) -> Result<Envelope<'_>, EnvelopeError> {
    if bytes.len() < ENVELOPE_MIN_LEN {
        return Err(EnvelopeError::TooShort {
            have: bytes.len(),
            need: ENVELOPE_MIN_LEN,
        });
    }
    let (header, extension) = decode_header(bytes)?;
    let body_start = header.body_offset();
    let body_end = bytes.len() - ENVELOPE_FOOTER_LEN;
    if body_start > body_end {
        return Err(EnvelopeError::BadHeaderLength {
            declared: header.header_length,
            have: bytes.len(),
        });
    }
    let body = &bytes[body_start..body_end];
    let stored = u32::from_le_bytes([
        bytes[body_end],
        bytes[body_end + 1],
        bytes[body_end + 2],
        bytes[body_end + 3],
    ]);
    let computed = hash32(body);
    if computed != stored {
        return Err(EnvelopeError::FooterChecksumMismatch { stored, computed });
    }
    Ok(Envelope {
        header,
        extension,
        body,
        footer_checksum: stored,
    })
}

/// Parses a whole envelope and refuses one that is not of the expected kind
pub fn decode_as(bytes: &[u8], expected: FormatKind) -> Result<Envelope<'_>, EnvelopeError> {
    let envelope = decode(bytes)?;
    if envelope.header.kind != expected {
        return Err(EnvelopeError::MagicMismatch {
            expected,
            found: envelope.header.kind.magic(),
        });
    }
    Ok(envelope)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::kind::ALL_FORMAT_KINDS;

    #[test]
    fn test_round_trip_for_every_format_kind() {
        for kind in ALL_FORMAT_KINDS {
            let body = format!("body of {kind}").into_bytes();
            let version = FormatVersion::new(2, 5);
            let encoded = encode(*kind, version, &body);
            let decoded = decode(&encoded).expect("decodes");
            assert_eq!(decoded.header.kind, *kind);
            assert_eq!(decoded.header.version, version);
            assert_eq!(decoded.header.flags, 0);
            assert_eq!(decoded.header.header_length as usize, ENVELOPE_HEADER_LEN);
            assert!(decoded.extension.is_empty());
            assert_eq!(decoded.body, &body[..]);
        }
    }

    #[test]
    fn test_round_trip_with_extension_and_flags() {
        let extension = b"per-format header bytes";
        let body = b"the body";
        let encoded = encode_with(
            FormatKind::ZyrColumnar,
            FormatVersion::new(1, 3),
            flags::COMPRESSED | 0x0000_0100,
            extension,
            body,
        );
        let decoded = decode(&encoded).expect("decodes");
        assert!(decoded.header.is_compressed());
        assert!(!decoded.header.is_encrypted());
        assert_eq!(decoded.extension, extension);
        assert_eq!(decoded.body, body);
        assert_eq!(
            decoded.header.header_length as usize,
            ENVELOPE_HEADER_LEN + extension.len()
        );
    }

    #[test]
    fn test_peek_reads_kind_and_version_from_eight_bytes() {
        let encoded = encode(FormatKind::WalSegment, FormatVersion::new(4, 1), b"x");
        let (kind, version) = peek(&encoded[..ENVELOPE_PEEK_LEN]).expect("peeks");
        assert_eq!(kind, FormatKind::WalSegment);
        assert_eq!(version, FormatVersion::new(4, 1));
    }

    #[test]
    fn test_empty_body_round_trips() {
        let encoded = encode(FormatKind::Fsm, FormatVersion::V1, &[]);
        assert_eq!(encoded.len(), ENVELOPE_MIN_LEN);
        let decoded = decode(&encoded).expect("decodes");
        assert!(decoded.body.is_empty());
    }

    #[test]
    fn test_corruption_at_any_byte_is_caught() {
        let body: Vec<u8> = (0u8..64).collect();
        let encoded = encode(FormatKind::HeapPage, FormatVersion::new(1, 2), &body);
        for index in 0..encoded.len() {
            let mut corrupted = encoded.clone();
            corrupted[index] ^= 0x01;
            assert!(
                decode(&corrupted).is_err(),
                "flipping byte {index} was not caught"
            );
        }
    }

    #[test]
    fn test_unknown_magic_is_refused() {
        let mut encoded = encode(FormatKind::HeapPage, FormatVersion::V1, b"x");
        encoded[0..4].copy_from_slice(b"QQQQ");
        match decode(&encoded) {
            Err(EnvelopeError::UnknownMagic { magic }) => assert_eq!(&magic, b"QQQQ"),
            other => panic!("expected UnknownMagic, got {other:?}"),
        }
    }

    #[test]
    fn test_wrong_kind_is_refused_by_decode_as() {
        let encoded = encode(FormatKind::HeapPage, FormatVersion::V1, b"x");
        match decode_as(&encoded, FormatKind::BTreeLeaf) {
            Err(EnvelopeError::MagicMismatch { expected, found }) => {
                assert_eq!(expected, FormatKind::BTreeLeaf);
                assert_eq!(found, FormatKind::HeapPage.magic());
            }
            other => panic!("expected MagicMismatch, got {other:?}"),
        }
    }

    #[test]
    fn test_declared_header_length_past_the_slice_is_refused() {
        let mut encoded = encode(FormatKind::HeapPage, FormatVersion::V1, b"body");
        encoded[8..12].copy_from_slice(&9999u32.to_le_bytes());
        assert!(matches!(
            decode(&encoded),
            Err(EnvelopeError::BadHeaderLength { .. })
        ));
    }

    #[test]
    fn test_short_input_is_refused() {
        assert!(matches!(
            decode(&[0u8; 4]),
            Err(EnvelopeError::TooShort { .. })
        ));
        assert!(matches!(
            peek(&[0u8; 3]),
            Err(EnvelopeError::TooShort { .. })
        ));
    }

    #[test]
    fn test_peek_version_of_refuses_a_foreign_file() {
        let encoded = encode(FormatKind::RaftLog, FormatVersion::new(2, 0), b"x");
        assert_eq!(
            peek_version_of(&encoded, FormatKind::RaftLog).expect("peeks"),
            FormatVersion::new(2, 0)
        );
        assert!(peek_version_of(&encoded, FormatKind::WalSegment).is_err());
    }
}

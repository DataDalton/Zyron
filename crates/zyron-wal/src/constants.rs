//! WAL constants for record format and segment configuration.

// Log record header constants

/// Size of the record header in bytes.
pub const HEADER_SIZE: usize = 24;

/// Size of the checksum in bytes.
pub const CHECKSUM_SIZE: usize = 4;

/// Maximum payload size (64 KB).
pub const MAX_PAYLOAD_SIZE: usize = 64 * 1024;

// Header field offsets for pointer-based parsing

/// Offset of LSN field in header.
pub const OFF_LSN: usize = 0;

/// Offset of previous LSN field in header.
pub const OFF_PREV_LSN: usize = 8;

/// Offset of transaction ID field in header.
pub const OFF_TXN_ID: usize = 16;

/// Offset of record type field in header.
pub const OFF_RECORD_TYPE: usize = 20;

/// Offset of flags field in header.
pub const OFF_FLAGS: usize = 21;

/// Offset of payload length field in header.
pub const OFF_PAYLOAD_LEN: usize = 22;

// Segment constants

/// Default segment size (16 MB).
pub const DEFAULT_SEGMENT_SIZE: u32 = 16 * 1024 * 1024;

/// Segment header size in bytes.
pub const SEGMENT_HEADER_SIZE: usize = 32;

/// Magic bytes identifying a WAL segment.
pub const SEGMENT_MAGIC: [u8; 4] = *b"ZWAL";

/// Current format version.
pub const SEGMENT_VERSION: u32 = 1;

//! WAL constants for record format and segment configuration.

// Log record header constants

/// Size of the record header in bytes.
pub const HEADER_SIZE: usize = 24;

/// Size of the checksum in bytes.
pub const CHECKSUM_SIZE: usize = 4;

/// Maximum payload size. Capped at u16::MAX (65535) because the on-disk
/// record header stores payload length as a 2-byte field.
pub const MAX_PAYLOAD_SIZE: usize = u16::MAX as usize;

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

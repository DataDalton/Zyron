//! WAL log record format.

use crate::constants::{
    CHECKSUM_SIZE, HEADER_SIZE, MAX_PAYLOAD_SIZE, OFF_FLAGS, OFF_LSN, OFF_PAYLOAD_LEN,
    OFF_PREV_LSN, OFF_RECORD_TYPE, OFF_TXN_ID,
};
use bytes::{BufMut, Bytes, BytesMut};
use serde::{Deserialize, Serialize};
use zyron_common::{Result, ZyronError};

/// Log Sequence Number - unique identifier for each log record.
///
/// LSN is a monotonically increasing 64-bit value that identifies
/// the position of a record in the WAL. It encodes both the segment
/// ID and offset within the segment.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
pub struct Lsn(pub u64);

impl Lsn {
    /// Invalid/uninitialized LSN.
    pub const INVALID: Lsn = Lsn(0);

    /// First valid LSN.
    pub const FIRST: Lsn = Lsn(1);

    /// Creates a new LSN from segment ID and offset.
    #[inline]
    pub fn new(segment_id: u32, offset: u32) -> Self {
        Self(((segment_id as u64) << 32) | (offset as u64))
    }

    /// Returns the segment ID portion of this LSN.
    #[inline]
    pub fn segment_id(&self) -> u32 {
        (self.0 >> 32) as u32
    }

    /// Returns the offset within the segment.
    #[inline]
    pub fn offset(&self) -> u32 {
        self.0 as u32
    }

    /// Returns true if this is a valid LSN.
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.0 > 0
    }

    /// Returns the next LSN after advancing by the given number of bytes.
    #[inline]
    pub fn advance(&self, bytes: u32) -> Self {
        Self(self.0 + bytes as u64)
    }
}

impl std::fmt::Display for Lsn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.segment_id(), self.offset())
    }
}

/// Types of log records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum LogRecordType {
    /// Invalid/padding record.
    Invalid = 0,
    /// Transaction begin.
    Begin = 1,
    /// Transaction commit.
    Commit = 2,
    /// Transaction abort/rollback.
    Abort = 3,
    /// Page insert operation.
    Insert = 10,
    /// Page update operation.
    Update = 11,
    /// Page delete operation.
    Delete = 12,
    /// Full page image (for recovery).
    FullPage = 20,
    /// Checkpoint begin marker.
    CheckpointBegin = 30,
    /// Checkpoint end marker.
    CheckpointEnd = 31,
    /// Compensation log record (for undo).
    Clr = 40,
    /// Compaction begin marker. Payload: table_id(8) + output file path.
    CompactionBegin = 50,
    /// Compaction end marker. Payload: table_id(8) + file_size(8) + row_count(8) + output file path.
    CompactionEnd = 51,
}

impl TryFrom<u8> for LogRecordType {
    type Error = ZyronError;

    fn try_from(value: u8) -> Result<Self> {
        match value {
            0 => Ok(LogRecordType::Invalid),
            1 => Ok(LogRecordType::Begin),
            2 => Ok(LogRecordType::Commit),
            3 => Ok(LogRecordType::Abort),
            10 => Ok(LogRecordType::Insert),
            11 => Ok(LogRecordType::Update),
            12 => Ok(LogRecordType::Delete),
            20 => Ok(LogRecordType::FullPage),
            30 => Ok(LogRecordType::CheckpointBegin),
            31 => Ok(LogRecordType::CheckpointEnd),
            40 => Ok(LogRecordType::Clr),
            50 => Ok(LogRecordType::CompactionBegin),
            51 => Ok(LogRecordType::CompactionEnd),
            _ => Err(ZyronError::WalCorrupted {
                lsn: 0,
                reason: format!("invalid record type: {}", value),
            }),
        }
    }
}

/// A single log record in the WAL.
///
/// Record format on disk:
/// - header (24 bytes):
///   - lsn: 8 bytes
///   - prev_lsn: 8 bytes (for transaction chaining)
///   - txn_id: 4 bytes
///   - record_type: 1 byte
///   - flags: 1 byte
///   - payload_len: 2 bytes
/// - payload: variable length
/// - checksum: 4 bytes (custom WAL checksum of header + payload)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogRecord {
    /// LSN of this record.
    pub lsn: Lsn,
    /// LSN of the previous record in this transaction.
    pub prev_lsn: Lsn,
    /// Transaction ID.
    pub txn_id: u32,
    /// Type of this record.
    pub record_type: LogRecordType,
    /// Record flags.
    pub flags: u8,
    /// Record payload.
    pub payload: Bytes,
}

impl LogRecord {
    // Re-export constants for API compatibility
    pub const HEADER_SIZE: usize = HEADER_SIZE;
    pub const CHECKSUM_SIZE: usize = CHECKSUM_SIZE;
    pub const MAX_PAYLOAD_SIZE: usize = MAX_PAYLOAD_SIZE;

    /// Creates a new log record.
    #[inline]
    pub fn new(
        lsn: Lsn,
        prev_lsn: Lsn,
        txn_id: u32,
        record_type: LogRecordType,
        payload: Bytes,
    ) -> Self {
        Self {
            lsn,
            prev_lsn,
            txn_id,
            record_type,
            flags: 0,
            payload,
        }
    }

    /// Creates a transaction begin record.
    #[inline]
    pub fn begin(lsn: Lsn, txn_id: u32) -> Self {
        Self::new(
            lsn,
            Lsn::INVALID,
            txn_id,
            LogRecordType::Begin,
            Bytes::new(),
        )
    }

    /// Creates a transaction commit record.
    #[inline]
    pub fn commit(lsn: Lsn, prev_lsn: Lsn, txn_id: u32) -> Self {
        Self::new(lsn, prev_lsn, txn_id, LogRecordType::Commit, Bytes::new())
    }

    /// Creates a transaction abort record.
    #[inline]
    pub fn abort(lsn: Lsn, prev_lsn: Lsn, txn_id: u32) -> Self {
        Self::new(lsn, prev_lsn, txn_id, LogRecordType::Abort, Bytes::new())
    }

    /// Returns the total size of this record on disk.
    #[inline]
    pub fn size_on_disk(&self) -> usize {
        HEADER_SIZE + self.payload.len() + CHECKSUM_SIZE
    }

    /// Serializes this record to bytes.
    #[inline]
    pub fn serialize(&self) -> Bytes {
        use crate::checksum::WalHasher;

        let total_size = self.size_on_disk();
        let mut buf = BytesMut::with_capacity(total_size);

        let payload_len = self.payload.len() as u16;
        let data_len = HEADER_SIZE + self.payload.len();

        // Compute checksum from struct fields (no buffer re-read needed)
        let mut hasher = WalHasher::new(data_len);
        hasher.write_header_fields(
            self.lsn.0,
            self.prev_lsn.0,
            self.txn_id,
            self.record_type as u8,
            self.flags,
            payload_len,
        );
        hasher.write_payload(&self.payload);
        let checksum = hasher.finish();

        // Write header
        buf.put_u64_le(self.lsn.0);
        buf.put_u64_le(self.prev_lsn.0);
        buf.put_u32_le(self.txn_id);
        buf.put_u8(self.record_type as u8);
        buf.put_u8(self.flags);
        buf.put_u16_le(payload_len);

        // Write payload
        buf.put_slice(&self.payload);

        // Write checksum
        buf.put_u32_le(checksum);

        buf.freeze()
    }

    /// Deserializes a record from bytes with checksum verification.
    /// Verifies checksum directly from input bytes without re-serialization.
    #[inline]
    pub fn deserialize(data: &[u8]) -> Result<Self> {
        if data.len() < HEADER_SIZE + CHECKSUM_SIZE {
            return Err(ZyronError::WalCorrupted {
                lsn: 0,
                reason: "record too short".to_string(),
            });
        }

        // Parse header fields directly from slice
        let lsn = Lsn(u64::from_le_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ]));
        let prev_lsn = Lsn(u64::from_le_bytes([
            data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15],
        ]));
        let txn_id = u32::from_le_bytes([data[16], data[17], data[18], data[19]]);
        let record_type = LogRecordType::try_from(data[20])?;
        let flags = data[21];
        let payload_len = u16::from_le_bytes([data[22], data[23]]) as usize;

        if payload_len > MAX_PAYLOAD_SIZE {
            return Err(ZyronError::WalCorrupted {
                lsn: lsn.0,
                reason: format!("payload too large: {}", payload_len),
            });
        }

        let total_size = HEADER_SIZE + payload_len + CHECKSUM_SIZE;
        if data.len() < total_size {
            return Err(ZyronError::WalCorrupted {
                lsn: lsn.0,
                reason: "truncated record".to_string(),
            });
        }

        // Verify checksum over header + payload
        let checksum_offset = HEADER_SIZE + payload_len;
        let stored_checksum = u32::from_le_bytes([
            data[checksum_offset],
            data[checksum_offset + 1],
            data[checksum_offset + 2],
            data[checksum_offset + 3],
        ]);
        let computed_checksum =
            crate::checksum::wal_checksum(&data[..checksum_offset], HEADER_SIZE);

        if stored_checksum != computed_checksum {
            return Err(ZyronError::WalCorrupted {
                lsn: lsn.0,
                reason: format!(
                    "checksum mismatch: stored={}, computed={}",
                    stored_checksum, computed_checksum
                ),
            });
        }

        // Extract payload
        let payload = Bytes::copy_from_slice(&data[HEADER_SIZE..HEADER_SIZE + payload_len]);

        Ok(Self {
            lsn,
            prev_lsn,
            txn_id,
            record_type,
            flags,
            payload,
        })
    }

    /// Parses all records from a contiguous Bytes buffer with checksum verification.
    /// Uses zero-copy slicing for payload data to avoid per-record allocation.
    /// Pointer-based parsing eliminates bounds checks after initial validation.
    #[inline]
    pub fn parse_all(data: Bytes) -> Result<Vec<Self>> {
        let mut records = Vec::with_capacity(data.len() / 64);
        let mut offset = 0;
        let data_len = data.len();
        let base_ptr = data.as_ptr();

        while offset + HEADER_SIZE + CHECKSUM_SIZE <= data_len {
            // SAFETY: bounds check above ensures we can read HEADER_SIZE bytes
            let (lsn_raw, prev_lsn_raw, txn_id, record_type_byte, flags, payload_len) = unsafe {
                let ptr = base_ptr.add(offset);
                let lsn_raw = std::ptr::read_unaligned(ptr.add(OFF_LSN) as *const u64);
                let prev_lsn_raw = std::ptr::read_unaligned(ptr.add(OFF_PREV_LSN) as *const u64);
                let txn_id = std::ptr::read_unaligned(ptr.add(OFF_TXN_ID) as *const u32);
                let record_type_byte = *ptr.add(OFF_RECORD_TYPE);
                let flags = *ptr.add(OFF_FLAGS);
                let payload_len =
                    std::ptr::read_unaligned(ptr.add(OFF_PAYLOAD_LEN) as *const u16) as usize;
                (
                    u64::from_le(lsn_raw),
                    u64::from_le(prev_lsn_raw),
                    u32::from_le(txn_id),
                    record_type_byte,
                    flags,
                    payload_len,
                )
            };

            let record_size = HEADER_SIZE + payload_len + CHECKSUM_SIZE;
            if offset + record_size > data_len {
                break;
            }

            // Verify checksum over header + payload
            let checksum_offset = offset + HEADER_SIZE + payload_len;
            // SAFETY: bounds check above ensures checksum_offset + 4 <= data_len
            let stored_checksum = unsafe {
                let ptr = base_ptr.add(checksum_offset);
                u32::from_le(std::ptr::read_unaligned(ptr as *const u32))
            };
            // SAFETY: bounds check ensures valid slice
            let data_slice = unsafe {
                std::slice::from_raw_parts(base_ptr.add(offset), checksum_offset - offset)
            };
            let computed_checksum = crate::checksum::wal_checksum(data_slice, HEADER_SIZE);
            if stored_checksum != computed_checksum {
                break;
            }

            let record_type = match LogRecordType::try_from(record_type_byte) {
                Ok(rt) => rt,
                Err(_) => break,
            };

            // For non-empty payloads, zero-copy slice shares underlying buffer via Arc.
            // Empty payloads use Bytes::new() to skip the Arc refcount increment.
            let payload = if payload_len > 0 {
                let payload_start = offset + HEADER_SIZE;
                data.slice(payload_start..payload_start + payload_len)
            } else {
                Bytes::new()
            };

            records.push(Self {
                lsn: Lsn(lsn_raw),
                prev_lsn: Lsn(prev_lsn_raw),
                txn_id,
                record_type,
                flags,
                payload,
            });

            offset += record_size;
        }

        Ok(records)
    }

    /// Parses all records from a contiguous Bytes buffer without checksum verification.
    /// For trusted data (just written, no crash recovery needed). The existing
    /// `parse_all` with checksums remains for crash recovery.
    #[inline]
    pub fn parse_all_trusted(data: Bytes) -> Vec<Self> {
        let mut records = Vec::with_capacity(data.len() / 64);
        let mut offset = 0;
        let data_len = data.len();
        let base_ptr = data.as_ptr();

        while offset + HEADER_SIZE + CHECKSUM_SIZE <= data_len {
            let (lsn_raw, prev_lsn_raw, txn_id, record_type_byte, flags, payload_len) = unsafe {
                let ptr = base_ptr.add(offset);
                let lsn_raw = std::ptr::read_unaligned(ptr.add(OFF_LSN) as *const u64);
                let prev_lsn_raw = std::ptr::read_unaligned(ptr.add(OFF_PREV_LSN) as *const u64);
                let txn_id = std::ptr::read_unaligned(ptr.add(OFF_TXN_ID) as *const u32);
                let record_type_byte = *ptr.add(OFF_RECORD_TYPE);
                let flags = *ptr.add(OFF_FLAGS);
                let payload_len =
                    std::ptr::read_unaligned(ptr.add(OFF_PAYLOAD_LEN) as *const u16) as usize;
                (
                    u64::from_le(lsn_raw),
                    u64::from_le(prev_lsn_raw),
                    u32::from_le(txn_id),
                    record_type_byte,
                    flags,
                    payload_len,
                )
            };

            let record_size = HEADER_SIZE + payload_len + CHECKSUM_SIZE;
            if offset + record_size > data_len {
                break;
            }

            let record_type = match LogRecordType::try_from(record_type_byte) {
                Ok(rt) => rt,
                Err(_) => break,
            };

            let payload = if payload_len > 0 {
                let payload_start = offset + HEADER_SIZE;
                data.slice(payload_start..payload_start + payload_len)
            } else {
                Bytes::new()
            };

            records.push(Self {
                lsn: Lsn(lsn_raw),
                prev_lsn: Lsn(prev_lsn_raw),
                txn_id,
                record_type,
                flags,
                payload,
            });

            offset += record_size;
        }

        records
    }

    /// Parses records from a contiguous Bytes buffer, calling f for each valid record.
    ///
    /// Same parsing and checksum logic as parse_all but with no intermediate Vec.
    /// Each record is handed to the callback immediately after parsing.
    #[inline]
    pub fn for_each<F>(data: Bytes, mut f: F)
    where
        F: FnMut(LogRecord),
    {
        let mut offset = 0;
        let data_len = data.len();
        let base_ptr = data.as_ptr();

        while offset + HEADER_SIZE + CHECKSUM_SIZE <= data_len {
            // SAFETY: bounds check above ensures we can read HEADER_SIZE bytes.
            let (lsn_raw, prev_lsn_raw, txn_id, record_type_byte, flags, payload_len) = unsafe {
                let ptr = base_ptr.add(offset);
                let lsn_raw = std::ptr::read_unaligned(ptr.add(OFF_LSN) as *const u64);
                let prev_lsn_raw = std::ptr::read_unaligned(ptr.add(OFF_PREV_LSN) as *const u64);
                let txn_id = std::ptr::read_unaligned(ptr.add(OFF_TXN_ID) as *const u32);
                let record_type_byte = *ptr.add(OFF_RECORD_TYPE);
                let flags = *ptr.add(OFF_FLAGS);
                let payload_len =
                    std::ptr::read_unaligned(ptr.add(OFF_PAYLOAD_LEN) as *const u16) as usize;
                (
                    u64::from_le(lsn_raw),
                    u64::from_le(prev_lsn_raw),
                    u32::from_le(txn_id),
                    record_type_byte,
                    flags,
                    payload_len,
                )
            };

            let record_size = HEADER_SIZE + payload_len + CHECKSUM_SIZE;
            if offset + record_size > data_len {
                break;
            }

            let checksum_offset = offset + HEADER_SIZE + payload_len;
            // SAFETY: bounds check above ensures checksum_offset + 4 <= data_len.
            let stored_checksum = unsafe {
                let ptr = base_ptr.add(checksum_offset);
                u32::from_le(std::ptr::read_unaligned(ptr as *const u32))
            };
            // SAFETY: bounds check ensures valid slice.
            let data_slice = unsafe {
                std::slice::from_raw_parts(base_ptr.add(offset), checksum_offset - offset)
            };
            let computed_checksum = crate::checksum::wal_checksum(data_slice, HEADER_SIZE);
            if stored_checksum != computed_checksum {
                break;
            }

            let record_type = match LogRecordType::try_from(record_type_byte) {
                Ok(rt) => rt,
                Err(_) => break,
            };

            let payload = if payload_len > 0 {
                let payload_start = offset + HEADER_SIZE;
                data.slice(payload_start..payload_start + payload_len)
            } else {
                Bytes::new()
            };

            f(Self {
                lsn: Lsn(lsn_raw),
                prev_lsn: Lsn(prev_lsn_raw),
                txn_id,
                record_type,
                flags,
                payload,
            });

            offset += record_size;
        }
    }
}

/// Lightweight record reference that defers payload slicing until needed.
/// Avoids Arc refcount increment per record during parsing. Payload is
/// accessed on demand via `payload()`.
#[derive(Debug, Clone)]
pub struct LazyLogRecord {
    pub lsn: Lsn,
    pub prev_lsn: Lsn,
    pub txn_id: u32,
    pub record_type: LogRecordType,
    pub flags: u8,
    /// Offset of the payload within the original Bytes buffer.
    payload_offset: u32,
    /// Length of the payload.
    payload_len: u16,
}

impl LazyLogRecord {
    /// Returns the payload by slicing the original buffer on demand.
    #[inline]
    pub fn payload(&self, data: &Bytes) -> Bytes {
        if self.payload_len > 0 {
            let start = self.payload_offset as usize;
            data.slice(start..start + self.payload_len as usize)
        } else {
            Bytes::new()
        }
    }

    /// Returns the payload length without slicing.
    #[inline]
    pub fn payload_len(&self) -> usize {
        self.payload_len as usize
    }
}

/// Parses all records from a contiguous Bytes buffer without checksum
/// verification and without slicing payloads. Returns the buffer alongside
/// lightweight record references for on-demand payload access.
pub fn parse_all_lazy(data: Bytes) -> (Bytes, Vec<LazyLogRecord>) {
    let mut records = Vec::with_capacity(data.len() / 64);
    let mut offset = 0;
    let data_len = data.len();
    let base_ptr = data.as_ptr();

    while offset + HEADER_SIZE + CHECKSUM_SIZE <= data_len {
        let (lsn_raw, prev_lsn_raw, txn_id, record_type_byte, flags, payload_len) = unsafe {
            let ptr = base_ptr.add(offset);
            let lsn_raw = std::ptr::read_unaligned(ptr.add(OFF_LSN) as *const u64);
            let prev_lsn_raw = std::ptr::read_unaligned(ptr.add(OFF_PREV_LSN) as *const u64);
            let txn_id = std::ptr::read_unaligned(ptr.add(OFF_TXN_ID) as *const u32);
            let record_type_byte = *ptr.add(OFF_RECORD_TYPE);
            let flags = *ptr.add(OFF_FLAGS);
            let payload_len =
                std::ptr::read_unaligned(ptr.add(OFF_PAYLOAD_LEN) as *const u16) as usize;
            (
                u64::from_le(lsn_raw),
                u64::from_le(prev_lsn_raw),
                u32::from_le(txn_id),
                record_type_byte,
                flags,
                payload_len,
            )
        };

        let record_size = HEADER_SIZE + payload_len + CHECKSUM_SIZE;
        if offset + record_size > data_len {
            break;
        }

        let record_type = match LogRecordType::try_from(record_type_byte) {
            Ok(rt) => rt,
            Err(_) => break,
        };

        records.push(LazyLogRecord {
            lsn: Lsn(lsn_raw),
            prev_lsn: Lsn(prev_lsn_raw),
            txn_id,
            record_type,
            flags,
            payload_offset: (offset + HEADER_SIZE) as u32,
            payload_len: payload_len as u16,
        });

        offset += record_size;
    }

    (data, records)
}

/// Returns the on-disk size for a record with the given payload length.
#[inline(always)]
pub const fn record_size_for_payload(payload_len: usize) -> usize {
    HEADER_SIZE + payload_len + CHECKSUM_SIZE
}

/// Serializes a WAL record directly from individual fields and a payload slice.
///
/// This is the single write-path serialization function. Computes the checksum
/// incrementally from the provided field values and writes header + payload +
/// checksum into the destination buffer in one pass. No intermediate struct
/// construction or buffer re-read.
///
/// Returns the number of bytes written (HEADER_SIZE + payload.len() + CHECKSUM_SIZE).
///
/// # Safety
/// Caller must ensure `buf` has at least `record_size_for_payload(payload.len())`
/// bytes available.
#[inline]
pub unsafe fn serialize_raw(
    buf: *mut u8,
    lsn: Lsn,
    prev_lsn: Lsn,
    txn_id: u32,
    record_type: u8,
    flags: u8,
    payload: &[u8],
) -> usize {
    use crate::checksum::WalHasher;

    debug_assert!(
        payload.len() <= MAX_PAYLOAD_SIZE,
        "payload {} bytes exceeds MAX_PAYLOAD_SIZE {}",
        payload.len(),
        MAX_PAYLOAD_SIZE,
    );

    let payload_len = payload.len() as u16;
    let data_len = HEADER_SIZE + payload.len();

    // Compute checksum from field values, not from the output buffer
    let mut hasher = WalHasher::new(data_len);
    hasher.write_header_fields(lsn.0, prev_lsn.0, txn_id, record_type, flags, payload_len);
    hasher.write_payload(payload);
    let checksum = hasher.finish();

    // Pack the 24-byte header for a single memcpy
    #[repr(C, packed)]
    struct PackedHeader {
        lsn: u64,
        prev_lsn: u64,
        txn_id: u32,
        record_type: u8,
        flags: u8,
        payload_len: u16,
    }

    let header = PackedHeader {
        lsn: lsn.0.to_le(),
        prev_lsn: prev_lsn.0.to_le(),
        txn_id: txn_id.to_le(),
        record_type,
        flags,
        payload_len: payload_len.to_le(),
    };

    unsafe {
        std::ptr::copy_nonoverlapping(
            &header as *const PackedHeader as *const u8,
            buf,
            HEADER_SIZE,
        );
    }

    let mut offset = HEADER_SIZE;

    if !payload.is_empty() {
        unsafe {
            std::ptr::copy_nonoverlapping(payload.as_ptr(), buf.add(offset), payload.len());
        }
        offset += payload.len();
    }

    unsafe {
        std::ptr::copy_nonoverlapping(checksum.to_le_bytes().as_ptr(), buf.add(offset), 4);
    }
    offset + CHECKSUM_SIZE
}

/// Serializes a WAL record without computing the checksum.
///
/// Writes header + payload + 4 zero bytes (checksum placeholder) into the buffer.
/// The checksum must be backfilled later by `backfill_checksums` before the data
/// reaches disk. This defers the checksum cost from the append() hot path to the
/// flush thread.
///
/// Returns the number of bytes written (HEADER_SIZE + payload.len() + CHECKSUM_SIZE).
///
/// # Safety
/// Caller must ensure `buf` has at least `record_size_for_payload(payload.len())`
/// bytes available.
#[inline]
pub unsafe fn serialize_raw_deferred(
    buf: *mut u8,
    lsn: Lsn,
    prev_lsn: Lsn,
    txn_id: u32,
    record_type: u8,
    flags: u8,
    payload: &[u8],
) -> usize {
    debug_assert!(
        payload.len() <= MAX_PAYLOAD_SIZE,
        "payload {} bytes exceeds MAX_PAYLOAD_SIZE {}",
        payload.len(),
        MAX_PAYLOAD_SIZE,
    );

    let payload_len = payload.len() as u16;

    #[repr(C, packed)]
    struct PackedHeader {
        lsn: u64,
        prev_lsn: u64,
        txn_id: u32,
        record_type: u8,
        flags: u8,
        payload_len: u16,
    }

    let header = PackedHeader {
        lsn: lsn.0.to_le(),
        prev_lsn: prev_lsn.0.to_le(),
        txn_id: txn_id.to_le(),
        record_type,
        flags,
        payload_len: payload_len.to_le(),
    };

    unsafe {
        std::ptr::copy_nonoverlapping(
            &header as *const PackedHeader as *const u8,
            buf,
            HEADER_SIZE,
        );
    }

    let mut offset = HEADER_SIZE;

    if !payload.is_empty() {
        unsafe {
            std::ptr::copy_nonoverlapping(payload.as_ptr(), buf.add(offset), payload.len());
        }
        offset += payload.len();
    }

    // Write zero checksum placeholder
    unsafe {
        std::ptr::write_unaligned(buf.add(offset) as *mut u32, 0u32);
    }
    offset + CHECKSUM_SIZE
}

/// Walks a contiguous buffer of serialized WAL records and computes + writes
/// the checksum for each record. Called by the flush thread after drain_into
/// and before writing to disk.
///
/// Each record's checksum is computed from its header + payload bytes using
/// `wal_checksum`, then written into the 4-byte placeholder at the end of
/// each record.
pub fn backfill_checksums(buf: &mut [u8]) {
    let len = buf.len();
    let mut offset = 0;

    while offset + HEADER_SIZE + CHECKSUM_SIZE <= len {
        // Read payload_len from header
        let payload_len = u16::from_le_bytes([
            buf[offset + OFF_PAYLOAD_LEN],
            buf[offset + OFF_PAYLOAD_LEN + 1],
        ]) as usize;

        let record_size = HEADER_SIZE + payload_len + CHECKSUM_SIZE;
        if offset + record_size > len {
            break;
        }

        // Compute checksum over header + payload
        let data_end = offset + HEADER_SIZE + payload_len;
        let checksum = crate::checksum::wal_checksum(&buf[offset..data_end], HEADER_SIZE);

        // Write checksum into placeholder
        let checksum_offset = data_end;
        buf[checksum_offset..checksum_offset + 4].copy_from_slice(&checksum.to_le_bytes());

        offset += record_size;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lsn_new() {
        let lsn = Lsn::new(5, 1000);
        assert_eq!(lsn.segment_id(), 5);
        assert_eq!(lsn.offset(), 1000);
    }

    #[test]
    fn test_lsn_roundtrip() {
        let lsn = Lsn::new(123, 456789);
        let segment = lsn.segment_id();
        let offset = lsn.offset();
        let recovered = Lsn::new(segment, offset);
        assert_eq!(lsn, recovered);
    }

    #[test]
    fn test_lsn_advance() {
        let lsn = Lsn::new(1, 100);
        let advanced = lsn.advance(50);
        assert_eq!(advanced.offset(), 150);
    }

    #[test]
    fn test_lsn_ordering() {
        let lsn1 = Lsn::new(1, 100);
        let lsn2 = Lsn::new(1, 200);
        let lsn3 = Lsn::new(2, 50);

        assert!(lsn1 < lsn2);
        assert!(lsn2 < lsn3);
        assert!(lsn1 < lsn3);
    }

    #[test]
    fn test_lsn_display() {
        let lsn = Lsn::new(3, 1024);
        assert_eq!(lsn.to_string(), "3/1024");
    }

    #[test]
    fn test_lsn_validity() {
        assert!(!Lsn::INVALID.is_valid());
        assert!(Lsn::FIRST.is_valid());
        assert!(Lsn::new(1, 100).is_valid());
    }

    #[test]
    fn test_log_record_type_conversion() {
        assert_eq!(LogRecordType::try_from(0).unwrap(), LogRecordType::Invalid);
        assert_eq!(LogRecordType::try_from(1).unwrap(), LogRecordType::Begin);
        assert_eq!(LogRecordType::try_from(2).unwrap(), LogRecordType::Commit);
        assert_eq!(LogRecordType::try_from(10).unwrap(), LogRecordType::Insert);
        assert!(LogRecordType::try_from(255).is_err());
    }

    #[test]
    fn test_log_record_serialization() {
        let record = LogRecord::new(
            Lsn::new(1, 100),
            Lsn::INVALID,
            42,
            LogRecordType::Begin,
            Bytes::from_static(b"test payload"),
        );

        let serialized = record.serialize();
        let deserialized = LogRecord::deserialize(&serialized).unwrap();

        assert_eq!(deserialized.lsn, record.lsn);
        assert_eq!(deserialized.prev_lsn, record.prev_lsn);
        assert_eq!(deserialized.txn_id, record.txn_id);
        assert_eq!(deserialized.record_type, record.record_type);
        assert_eq!(deserialized.payload, record.payload);
    }

    #[test]
    fn test_log_record_size() {
        let record = LogRecord::new(
            Lsn::new(1, 0),
            Lsn::INVALID,
            1,
            LogRecordType::Begin,
            Bytes::from_static(b"hello"),
        );

        let expected_size = LogRecord::HEADER_SIZE + 5 + LogRecord::CHECKSUM_SIZE;
        assert_eq!(record.size_on_disk(), expected_size);
    }

    #[test]
    fn test_begin_commit_abort_records() {
        let begin = LogRecord::begin(Lsn::new(1, 0), 100);
        assert_eq!(begin.record_type, LogRecordType::Begin);
        assert_eq!(begin.txn_id, 100);

        let commit = LogRecord::commit(Lsn::new(1, 100), Lsn::new(1, 0), 100);
        assert_eq!(commit.record_type, LogRecordType::Commit);
        assert_eq!(commit.prev_lsn, Lsn::new(1, 0));

        let abort = LogRecord::abort(Lsn::new(1, 200), Lsn::new(1, 100), 100);
        assert_eq!(abort.record_type, LogRecordType::Abort);
    }

    #[test]
    fn test_corrupted_record_detection() {
        let record = LogRecord::begin(Lsn::new(1, 0), 1);
        let mut serialized = record.serialize().to_vec();

        // Corrupt a byte in the payload area
        if serialized.len() > 20 {
            serialized[20] ^= 0xFF;
        }

        let result = LogRecord::deserialize(&serialized);
        assert!(result.is_err());
    }
}

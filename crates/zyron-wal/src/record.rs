//! WAL log record format.

use bytes::{Buf, BufMut, Bytes, BytesMut};
use serde::{Deserialize, Serialize};
use zyron_common::{PageId, Result, ZyronError};

/// Log Sequence Number - unique identifier for each log record.
///
/// LSN is a monotonically increasing 64-bit value that identifies
/// the position of a record in the WAL. It encodes both the segment
/// ID and offset within the segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize)]
pub struct Lsn(pub u64);

impl Lsn {
    /// Invalid/uninitialized LSN.
    pub const INVALID: Lsn = Lsn(0);

    /// First valid LSN.
    pub const FIRST: Lsn = Lsn(1);

    /// Creates a new LSN from segment ID and offset.
    pub fn new(segment_id: u32, offset: u32) -> Self {
        Self(((segment_id as u64) << 32) | (offset as u64))
    }

    /// Returns the segment ID portion of this LSN.
    pub fn segment_id(&self) -> u32 {
        (self.0 >> 32) as u32
    }

    /// Returns the offset within the segment.
    pub fn offset(&self) -> u32 {
        self.0 as u32
    }

    /// Returns true if this is a valid LSN.
    pub fn is_valid(&self) -> bool {
        self.0 > 0
    }

    /// Returns the next LSN after advancing by the given number of bytes.
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
/// - checksum: 4 bytes (CRC32 of header + payload)
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
    /// Size of the record header in bytes.
    pub const HEADER_SIZE: usize = 24;
    /// Size of the checksum in bytes.
    pub const CHECKSUM_SIZE: usize = 4;
    /// Maximum payload size (64 KB).
    pub const MAX_PAYLOAD_SIZE: usize = 64 * 1024;

    /// Creates a new log record.
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
    pub fn begin(lsn: Lsn, txn_id: u32) -> Self {
        Self::new(lsn, Lsn::INVALID, txn_id, LogRecordType::Begin, Bytes::new())
    }

    /// Creates a transaction commit record.
    pub fn commit(lsn: Lsn, prev_lsn: Lsn, txn_id: u32) -> Self {
        Self::new(lsn, prev_lsn, txn_id, LogRecordType::Commit, Bytes::new())
    }

    /// Creates a transaction abort record.
    pub fn abort(lsn: Lsn, prev_lsn: Lsn, txn_id: u32) -> Self {
        Self::new(lsn, prev_lsn, txn_id, LogRecordType::Abort, Bytes::new())
    }

    /// Returns the total size of this record on disk.
    pub fn size_on_disk(&self) -> usize {
        Self::HEADER_SIZE + self.payload.len() + Self::CHECKSUM_SIZE
    }

    /// Serializes this record to bytes.
    pub fn serialize(&self) -> Bytes {
        let total_size = self.size_on_disk();
        let mut buf = BytesMut::with_capacity(total_size);

        // Write header
        buf.put_u64_le(self.lsn.0);
        buf.put_u64_le(self.prev_lsn.0);
        buf.put_u32_le(self.txn_id);
        buf.put_u8(self.record_type as u8);
        buf.put_u8(self.flags);
        buf.put_u16_le(self.payload.len() as u16);

        // Write payload
        buf.put_slice(&self.payload);

        // Compute and write checksum
        let checksum = crc32fast::hash(&buf);
        buf.put_u32_le(checksum);

        buf.freeze()
    }

    /// Deserializes a record from bytes.
    pub fn deserialize(mut data: &[u8]) -> Result<Self> {
        if data.len() < Self::HEADER_SIZE + Self::CHECKSUM_SIZE {
            return Err(ZyronError::WalCorrupted {
                lsn: 0,
                reason: "record too short".to_string(),
            });
        }

        // Read header
        let lsn = Lsn(data.get_u64_le());
        let prev_lsn = Lsn(data.get_u64_le());
        let txn_id = data.get_u32_le();
        let record_type = LogRecordType::try_from(data.get_u8())?;
        let flags = data.get_u8();
        let payload_len = data.get_u16_le() as usize;

        if payload_len > Self::MAX_PAYLOAD_SIZE {
            return Err(ZyronError::WalCorrupted {
                lsn: lsn.0,
                reason: format!("payload too large: {}", payload_len),
            });
        }

        if data.len() < payload_len + Self::CHECKSUM_SIZE {
            return Err(ZyronError::WalCorrupted {
                lsn: lsn.0,
                reason: "truncated record".to_string(),
            });
        }

        // Read payload
        let payload = Bytes::copy_from_slice(&data[..payload_len]);
        data.advance(payload_len);

        // Read and verify checksum
        let stored_checksum = data.get_u32_le();
        let record = Self {
            lsn,
            prev_lsn,
            txn_id,
            record_type,
            flags,
            payload,
        };

        // Recompute checksum for verification
        let serialized = record.serialize();
        let computed_checksum = (&serialized[serialized.len() - 4..]).get_u32_le();

        if stored_checksum != computed_checksum {
            return Err(ZyronError::WalCorrupted {
                lsn: lsn.0,
                reason: format!(
                    "checksum mismatch: stored={}, computed={}",
                    stored_checksum, computed_checksum
                ),
            });
        }

        Ok(record)
    }
}

/// Payload for insert/update/delete operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPayload {
    /// Page being modified.
    pub page_id: PageId,
    /// Slot number within the page.
    pub slot: u16,
    /// Old data (for update/delete, used in undo).
    pub old_data: Option<Bytes>,
    /// New data (for insert/update, used in redo).
    pub new_data: Option<Bytes>,
}

impl DataPayload {
    /// Serializes the payload to bytes.
    pub fn serialize(&self) -> Bytes {
        let mut buf = BytesMut::new();

        buf.put_u32_le(self.page_id.file_id);
        buf.put_u32_le(self.page_id.page_num);
        buf.put_u16_le(self.slot);

        // Old data
        if let Some(ref old) = self.old_data {
            buf.put_u32_le(old.len() as u32);
            buf.put_slice(old);
        } else {
            buf.put_u32_le(0);
        }

        // New data
        if let Some(ref new) = self.new_data {
            buf.put_u32_le(new.len() as u32);
            buf.put_slice(new);
        } else {
            buf.put_u32_le(0);
        }

        buf.freeze()
    }

    /// Deserializes the payload from bytes.
    pub fn deserialize(mut data: &[u8]) -> Result<Self> {
        if data.len() < 14 {
            return Err(ZyronError::WalCorrupted {
                lsn: 0,
                reason: "data payload too short".to_string(),
            });
        }

        let file_id = data.get_u32_le();
        let page_num = data.get_u32_le();
        let page_id = PageId::new(file_id, page_num);
        let slot = data.get_u16_le();

        // Old data
        let old_len = data.get_u32_le() as usize;
        let old_data = if old_len > 0 {
            if data.len() < old_len {
                return Err(ZyronError::WalCorrupted {
                    lsn: 0,
                    reason: "truncated old data".to_string(),
                });
            }
            let old = Bytes::copy_from_slice(&data[..old_len]);
            data.advance(old_len);
            Some(old)
        } else {
            None
        };

        // New data
        let new_len = data.get_u32_le() as usize;
        let new_data = if new_len > 0 {
            if data.len() < new_len {
                return Err(ZyronError::WalCorrupted {
                    lsn: 0,
                    reason: "truncated new data".to_string(),
                });
            }
            let new = Bytes::copy_from_slice(&data[..new_len]);
            Some(new)
        } else {
            None
        };

        Ok(Self {
            page_id,
            slot,
            old_data,
            new_data,
        })
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
    fn test_data_payload_serialization() {
        let payload = DataPayload {
            page_id: PageId::new(1, 42),
            slot: 5,
            old_data: Some(Bytes::from_static(b"old")),
            new_data: Some(Bytes::from_static(b"new value")),
        };

        let serialized = payload.serialize();
        let deserialized = DataPayload::deserialize(&serialized).unwrap();

        assert_eq!(deserialized.page_id, payload.page_id);
        assert_eq!(deserialized.slot, payload.slot);
        assert_eq!(deserialized.old_data, payload.old_data);
        assert_eq!(deserialized.new_data, payload.new_data);
    }

    #[test]
    fn test_data_payload_no_old_data() {
        let payload = DataPayload {
            page_id: PageId::new(0, 0),
            slot: 0,
            old_data: None,
            new_data: Some(Bytes::from_static(b"inserted")),
        };

        let serialized = payload.serialize();
        let deserialized = DataPayload::deserialize(&serialized).unwrap();

        assert!(deserialized.old_data.is_none());
        assert!(deserialized.new_data.is_some());
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

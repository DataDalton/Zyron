//! WAL segment management.

use crate::record::{LogRecord, Lsn};
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use zyron_common::{Result, ZyronError};

/// Unique identifier for a WAL segment file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SegmentId(pub u32);

impl SegmentId {
    /// First segment ID.
    pub const FIRST: SegmentId = SegmentId(1);

    /// Returns the next segment ID.
    pub fn next(&self) -> Self {
        SegmentId(self.0 + 1)
    }

    /// Generates the filename for this segment.
    pub fn filename(&self) -> String {
        format!("{:016}.wal", self.0)
    }
}

impl std::fmt::Display for SegmentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:08}", self.0)
    }
}

/// Header at the beginning of each segment file.
///
/// Layout (32 bytes):
/// - magic: 4 bytes ("ZWAL")
/// - version: 4 bytes
/// - segment_id: 4 bytes
/// - segment_size: 4 bytes
/// - first_lsn: 8 bytes
/// - flags: 4 bytes
/// - checksum: 4 bytes
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub struct SegmentHeader {
    /// Magic bytes for identification.
    pub magic: [u8; 4],
    /// Format version.
    pub version: u32,
    /// Segment ID.
    pub segment_id: SegmentId,
    /// Maximum size of this segment.
    pub segment_size: u32,
    /// First LSN in this segment.
    pub first_lsn: Lsn,
    /// Segment flags.
    pub flags: u32,
    /// Header checksum.
    pub checksum: u32,
}

impl SegmentHeader {
    /// Magic bytes identifying a WAL segment.
    pub const MAGIC: [u8; 4] = *b"ZWAL";
    /// Current format version.
    pub const VERSION: u32 = 1;
    /// Size of the header in bytes.
    pub const SIZE: usize = 32;

    /// Creates a new segment header.
    pub fn new(segment_id: SegmentId, segment_size: u32, first_lsn: Lsn) -> Self {
        let mut header = Self {
            magic: Self::MAGIC,
            version: Self::VERSION,
            segment_id,
            segment_size,
            first_lsn,
            flags: 0,
            checksum: 0,
        };
        header.checksum = header.compute_checksum();
        header
    }

    /// Computes the checksum for this header.
    fn compute_checksum(&self) -> u32 {
        let mut data = [0u8; Self::SIZE - 4];
        data[0..4].copy_from_slice(&self.magic);
        data[4..8].copy_from_slice(&self.version.to_le_bytes());
        data[8..12].copy_from_slice(&self.segment_id.0.to_le_bytes());
        data[12..16].copy_from_slice(&self.segment_size.to_le_bytes());
        data[16..24].copy_from_slice(&self.first_lsn.0.to_le_bytes());
        data[24..28].copy_from_slice(&self.flags.to_le_bytes());
        crc32fast::hash(&data)
    }

    /// Validates this header.
    pub fn validate(&self) -> Result<()> {
        if self.magic != Self::MAGIC {
            return Err(ZyronError::WalCorrupted {
                lsn: self.first_lsn.0,
                reason: "invalid magic bytes".to_string(),
            });
        }
        if self.version != Self::VERSION {
            return Err(ZyronError::WalCorrupted {
                lsn: self.first_lsn.0,
                reason: format!("unsupported version: {}", self.version),
            });
        }
        let expected_checksum = self.compute_checksum();
        if self.checksum != expected_checksum {
            return Err(ZyronError::WalCorrupted {
                lsn: self.first_lsn.0,
                reason: "header checksum mismatch".to_string(),
            });
        }
        Ok(())
    }

    /// Serializes the header to bytes.
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut data = [0u8; Self::SIZE];
        data[0..4].copy_from_slice(&self.magic);
        data[4..8].copy_from_slice(&self.version.to_le_bytes());
        data[8..12].copy_from_slice(&self.segment_id.0.to_le_bytes());
        data[12..16].copy_from_slice(&self.segment_size.to_le_bytes());
        data[16..24].copy_from_slice(&self.first_lsn.0.to_le_bytes());
        data[24..28].copy_from_slice(&self.flags.to_le_bytes());
        data[28..32].copy_from_slice(&self.checksum.to_le_bytes());
        data
    }

    /// Deserializes the header from bytes.
    pub fn from_bytes(data: &[u8; Self::SIZE]) -> Self {
        let mut magic = [0u8; 4];
        magic.copy_from_slice(&data[0..4]);

        Self {
            magic,
            version: u32::from_le_bytes(data[4..8].try_into().unwrap()),
            segment_id: SegmentId(u32::from_le_bytes(data[8..12].try_into().unwrap())),
            segment_size: u32::from_le_bytes(data[12..16].try_into().unwrap()),
            first_lsn: Lsn(u64::from_le_bytes(data[16..24].try_into().unwrap())),
            flags: u32::from_le_bytes(data[24..28].try_into().unwrap()),
            checksum: u32::from_le_bytes(data[28..32].try_into().unwrap()),
        }
    }
}

/// A single WAL segment file.
pub struct LogSegment {
    /// Path to the segment file.
    path: PathBuf,
    /// Segment header.
    header: SegmentHeader,
    /// Current write position within the segment.
    write_offset: u32,
    /// File handle for writing.
    file: Option<File>,
}

impl LogSegment {
    /// Default segment size (16 MB).
    pub const DEFAULT_SIZE: u32 = 16 * 1024 * 1024;

    /// Creates a new segment file.
    pub fn create(
        wal_dir: &Path,
        segment_id: SegmentId,
        first_lsn: Lsn,
        segment_size: u32,
    ) -> Result<Self> {
        let path = wal_dir.join(segment_id.filename());
        let header = SegmentHeader::new(segment_id, segment_size, first_lsn);

        let mut file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(true)
            .open(&path)?;

        // Write header
        file.write_all(&header.to_bytes())?;
        file.sync_all()?;

        Ok(Self {
            path,
            header,
            write_offset: SegmentHeader::SIZE as u32,
            file: Some(file),
        })
    }

    /// Opens an existing segment file.
    pub fn open(path: &Path) -> Result<Self> {
        let mut file = OpenOptions::new().read(true).write(true).open(path)?;

        // Read and validate header
        let mut header_bytes = [0u8; SegmentHeader::SIZE];
        file.read_exact(&mut header_bytes)?;
        let header = SegmentHeader::from_bytes(&header_bytes);
        header.validate()?;

        // Find current write position by seeking to end
        let file_size = file.seek(SeekFrom::End(0))?;
        let write_offset = file_size as u32;

        Ok(Self {
            path: path.to_path_buf(),
            header,
            write_offset,
            file: Some(file),
        })
    }

    /// Returns the segment ID.
    pub fn segment_id(&self) -> SegmentId {
        self.header.segment_id
    }

    /// Returns the first LSN in this segment.
    pub fn first_lsn(&self) -> Lsn {
        self.header.first_lsn
    }

    /// Returns the current write offset.
    pub fn write_offset(&self) -> u32 {
        self.write_offset
    }

    /// Returns the remaining space in this segment.
    pub fn remaining_space(&self) -> u32 {
        self.header.segment_size.saturating_sub(self.write_offset)
    }

    /// Returns true if this segment has space for a record of the given size.
    pub fn has_space(&self, record_size: usize) -> bool {
        self.remaining_space() >= record_size as u32
    }

    /// Returns the path to this segment file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Appends a record to this segment.
    pub fn append(&mut self, record: &LogRecord) -> Result<Lsn> {
        let data = record.serialize();
        let record_size = data.len() as u32;

        if !self.has_space(data.len()) {
            return Err(ZyronError::WalWriteFailed(
                "segment full".to_string(),
            ));
        }

        let file = self.file.as_mut().ok_or_else(|| {
            ZyronError::WalWriteFailed("segment not open for writing".to_string())
        })?;

        file.seek(SeekFrom::Start(self.write_offset as u64))?;
        file.write_all(&data)?;

        let lsn = Lsn::new(self.header.segment_id.0, self.write_offset);
        self.write_offset += record_size;

        Ok(lsn)
    }

    /// Syncs the segment to disk.
    pub fn sync(&mut self) -> Result<()> {
        if let Some(ref file) = self.file {
            file.sync_all()?;
        }
        Ok(())
    }

    /// Closes the segment.
    pub fn close(&mut self) -> Result<()> {
        if let Some(file) = self.file.take() {
            file.sync_all()?;
        }
        Ok(())
    }

    /// Reads a record at the given offset.
    pub fn read_at(&mut self, offset: u32) -> Result<LogRecord> {
        let file = self.file.as_mut().ok_or_else(|| {
            ZyronError::WalCorrupted {
                lsn: 0,
                reason: "segment not open".to_string(),
            }
        })?;

        file.seek(SeekFrom::Start(offset as u64))?;

        // Read header first to get payload length
        let mut header_buf = [0u8; LogRecord::HEADER_SIZE];
        file.read_exact(&mut header_buf)?;

        let payload_len = u16::from_le_bytes([header_buf[22], header_buf[23]]) as usize;
        let total_size = LogRecord::HEADER_SIZE + payload_len + LogRecord::CHECKSUM_SIZE;

        // Read full record
        let mut record_buf = vec![0u8; total_size];
        file.seek(SeekFrom::Start(offset as u64))?;
        file.read_exact(&mut record_buf)?;

        LogRecord::deserialize(&record_buf)
    }
}

/// Iterator over records in a segment.
pub struct SegmentIterator<'a> {
    segment: &'a mut LogSegment,
    offset: u32,
    end_offset: u32,
}

impl<'a> SegmentIterator<'a> {
    /// Creates a new iterator starting from the given offset.
    pub fn new(segment: &'a mut LogSegment, start_offset: u32) -> Self {
        let end_offset = segment.write_offset;
        Self {
            segment,
            offset: start_offset,
            end_offset,
        }
    }
}

impl Iterator for SegmentIterator<'_> {
    type Item = Result<LogRecord>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset >= self.end_offset {
            return None;
        }

        match self.segment.read_at(self.offset) {
            Ok(record) => {
                self.offset += record.size_on_disk() as u32;
                Some(Ok(record))
            }
            Err(e) => Some(Err(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use crate::record::LogRecordType;
    use tempfile::tempdir;

    #[test]
    fn test_segment_id() {
        let id = SegmentId(42);
        assert_eq!(id.next(), SegmentId(43));
        assert_eq!(id.filename(), "0000000000000042.wal");
    }

    #[test]
    fn test_segment_header_roundtrip() {
        let header = SegmentHeader::new(
            SegmentId(1),
            LogSegment::DEFAULT_SIZE,
            Lsn::new(1, 0),
        );

        let bytes = header.to_bytes();
        let recovered = SegmentHeader::from_bytes(&bytes);

        assert_eq!(recovered.magic, SegmentHeader::MAGIC);
        assert_eq!(recovered.version, SegmentHeader::VERSION);
        assert_eq!(recovered.segment_id, header.segment_id);
        assert_eq!(recovered.segment_size, header.segment_size);
        assert_eq!(recovered.first_lsn, header.first_lsn);
        recovered.validate().unwrap();
    }

    #[test]
    fn test_segment_header_validation() {
        let mut header = SegmentHeader::new(
            SegmentId(1),
            LogSegment::DEFAULT_SIZE,
            Lsn::new(1, 0),
        );

        // Valid header
        assert!(header.validate().is_ok());

        // Invalid magic
        header.magic = *b"XXXX";
        assert!(header.validate().is_err());
    }

    #[test]
    fn test_segment_create_and_open() {
        let dir = tempdir().unwrap();
        let segment_id = SegmentId::FIRST;
        let first_lsn = Lsn::new(1, SegmentHeader::SIZE as u32);

        // Create segment
        {
            let mut segment = LogSegment::create(
                dir.path(),
                segment_id,
                first_lsn,
                LogSegment::DEFAULT_SIZE,
            ).unwrap();

            assert_eq!(segment.segment_id(), segment_id);
            assert_eq!(segment.first_lsn(), first_lsn);
            segment.close().unwrap();
        }

        // Reopen segment
        {
            let path = dir.path().join(segment_id.filename());
            let segment = LogSegment::open(&path).unwrap();
            assert_eq!(segment.segment_id(), segment_id);
            assert_eq!(segment.first_lsn(), first_lsn);
        }
    }

    #[test]
    fn test_segment_append_and_read() {
        let dir = tempdir().unwrap();
        let segment_id = SegmentId::FIRST;

        let mut segment = LogSegment::create(
            dir.path(),
            segment_id,
            Lsn::new(1, SegmentHeader::SIZE as u32),
            LogSegment::DEFAULT_SIZE,
        ).unwrap();

        // Append a record
        let record = LogRecord::new(
            Lsn::INVALID, // Will be assigned by append
            Lsn::INVALID,
            1,
            LogRecordType::Begin,
            Bytes::from_static(b"test"),
        );

        let lsn = segment.append(&record).unwrap();
        segment.sync().unwrap();

        // Read it back
        let read_record = segment.read_at(lsn.offset()).unwrap();
        assert_eq!(read_record.txn_id, 1);
        assert_eq!(read_record.record_type, LogRecordType::Begin);
        assert_eq!(read_record.payload, Bytes::from_static(b"test"));
    }

    #[test]
    fn test_segment_remaining_space() {
        let dir = tempdir().unwrap();
        let small_size = 1024u32;

        let segment = LogSegment::create(
            dir.path(),
            SegmentId::FIRST,
            Lsn::FIRST,
            small_size,
        ).unwrap();

        let expected_remaining = small_size - SegmentHeader::SIZE as u32;
        assert_eq!(segment.remaining_space(), expected_remaining);
    }

    #[test]
    fn test_segment_full() {
        let dir = tempdir().unwrap();
        let small_size = 128u32; // Very small segment

        let mut segment = LogSegment::create(
            dir.path(),
            SegmentId::FIRST,
            Lsn::FIRST,
            small_size,
        ).unwrap();

        // Try to write a record larger than remaining space
        let large_payload = Bytes::from(vec![0u8; 200]);
        let record = LogRecord::new(
            Lsn::INVALID,
            Lsn::INVALID,
            1,
            LogRecordType::FullPage,
            large_payload,
        );

        let result = segment.append(&record);
        assert!(result.is_err());
    }
}

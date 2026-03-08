//! Reader and writer for the .zyr columnar file format.
//!
//! File layout:
//!   [0x0000] FILE HEADER (PAGE_SIZE = 16384 bytes)
//!     [0..8]     magic: "ZYRCOL\0\0"
//!     [8..12]    format_version: u32
//!     [12..16]   header_checksum: u32
//!     [16..20]   column_count: u32
//!     [20..28]   row_count: u64
//!     [28..36]   table_id: u64
//!     [36..44]   xmin_range_lo: u64
//!     [44..52]   xmin_range_hi: u64
//!     [52..60]   xmax_range_lo: u64
//!     [60..68]   xmax_range_hi: u64
//!     [68..72]   primary_key_column_id: u32
//!     [72]       sort_order: u8
//!     [73..128]  reserved (zeroed)
//!     [128..PAGE_SIZE] padding
//!
//!   [PAGE_SIZE+] COLUMN SEGMENTS (each page-aligned)
//!     SegmentHeader (128 bytes) + bloom + zone_maps + encoded_data + padding
//!
//!   FOOTER:
//!     Segment index: column_count * 20 bytes
//!       column_id(4) + offset(8) + size(8) per entry
//!     segment_index_offset: u64
//!     magic repeat: "ZYRCOL\0\0"
//!     file_checksum: u32

use super::constants::{
    FILE_HEADER_METADATA_SIZE, FILE_HEADER_SIZE, FOOTER_SIZE, SEGMENT_HEADER_SIZE,
    SEGMENT_INDEX_ENTRY_SIZE, ZYR_FORMAT_VERSION, ZYR_MAGIC,
};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use zyron_common::page::PAGE_SIZE;
use zyron_common::{Result, ZyronError};

// ---------------------------------------------------------------------------
// SortOrder
// ---------------------------------------------------------------------------

/// Sort order for the primary key column in a .zyr file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SortOrder {
    /// No guaranteed sort order.
    None = 0,
    /// Rows sorted in ascending key order.
    Asc = 1,
    /// Rows sorted in descending key order.
    Desc = 2,
}

impl SortOrder {
    fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(SortOrder::None),
            1 => Ok(SortOrder::Asc),
            2 => Ok(SortOrder::Desc),
            other => Err(ZyronError::InvalidZyrFile(format!(
                "unknown sort_order value: {}",
                other
            ))),
        }
    }
}

// ---------------------------------------------------------------------------
// ZyrFileHeader
// ---------------------------------------------------------------------------

/// Metadata stored in the first PAGE_SIZE bytes of a .zyr file.
#[derive(Debug, Clone)]
pub struct ZyrFileHeader {
    pub format_version: u32,
    pub column_count: u32,
    pub row_count: u64,
    pub table_id: u64,
    pub xmin_range_lo: u64,
    pub xmin_range_hi: u64,
    pub xmax_range_lo: u64,
    pub xmax_range_hi: u64,
    pub primary_key_column_id: u32,
    pub sort_order: SortOrder,
}

impl ZyrFileHeader {
    /// Serializes the header into a full PAGE_SIZE buffer.
    /// The header_checksum field at bytes [12..16] covers bytes [0..12] and
    /// [16..FILE_HEADER_METADATA_SIZE].
    pub fn to_bytes(&self) -> [u8; PAGE_SIZE] {
        let mut buf = [0u8; PAGE_SIZE];

        buf[0..8].copy_from_slice(&ZYR_MAGIC);
        buf[8..12].copy_from_slice(&self.format_version.to_le_bytes());
        // [12..16] = header_checksum, filled below.
        buf[16..20].copy_from_slice(&self.column_count.to_le_bytes());
        buf[20..28].copy_from_slice(&self.row_count.to_le_bytes());
        buf[28..36].copy_from_slice(&self.table_id.to_le_bytes());
        buf[36..44].copy_from_slice(&self.xmin_range_lo.to_le_bytes());
        buf[44..52].copy_from_slice(&self.xmin_range_hi.to_le_bytes());
        buf[52..60].copy_from_slice(&self.xmax_range_lo.to_le_bytes());
        buf[60..68].copy_from_slice(&self.xmax_range_hi.to_le_bytes());
        buf[68..72].copy_from_slice(&self.primary_key_column_id.to_le_bytes());
        buf[72] = self.sort_order as u8;
        // [73..128] reserved, already zeroed.
        // [128..PAGE_SIZE] padding, already zeroed.

        // Checksum covers magic+version [0..12] and metadata [16..FILE_HEADER_METADATA_SIZE].
        let checksum = header_checksum(&buf);
        buf[12..16].copy_from_slice(&checksum.to_le_bytes());

        buf
    }

    /// Deserializes a header from a PAGE_SIZE buffer. Validates magic, version,
    /// and checksum before returning.
    pub fn from_bytes(buf: &[u8; PAGE_SIZE]) -> Result<Self> {
        if buf[0..8] != ZYR_MAGIC {
            return Err(ZyronError::InvalidZyrFile(
                "invalid magic bytes in file header".into(),
            ));
        }

        let formatVersion = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);
        if formatVersion != ZYR_FORMAT_VERSION {
            return Err(ZyronError::InvalidZyrFile(format!(
                "unsupported format version: {} (expected {})",
                formatVersion, ZYR_FORMAT_VERSION
            )));
        }

        let storedChecksum = u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]);
        let computedChecksum = header_checksum(buf);
        if storedChecksum != computedChecksum {
            return Err(ZyronError::InvalidZyrFile(format!(
                "header checksum mismatch: stored 0x{:08x}, computed 0x{:08x}",
                storedChecksum, computedChecksum
            )));
        }

        let columnCount = u32::from_le_bytes([buf[16], buf[17], buf[18], buf[19]]);
        let rowCount = u64::from_le_bytes(
            buf[20..28]
                .try_into()
                .map_err(|_| ZyronError::InvalidZyrFile("failed to read row_count".into()))?,
        );
        let tableId = u64::from_le_bytes(
            buf[28..36]
                .try_into()
                .map_err(|_| ZyronError::InvalidZyrFile("failed to read table_id".into()))?,
        );
        let xminRangeLo = u64::from_le_bytes(
            buf[36..44]
                .try_into()
                .map_err(|_| ZyronError::InvalidZyrFile("failed to read xmin_range_lo".into()))?,
        );
        let xminRangeHi = u64::from_le_bytes(
            buf[44..52]
                .try_into()
                .map_err(|_| ZyronError::InvalidZyrFile("failed to read xmin_range_hi".into()))?,
        );
        let xmaxRangeLo = u64::from_le_bytes(
            buf[52..60]
                .try_into()
                .map_err(|_| ZyronError::InvalidZyrFile("failed to read xmax_range_lo".into()))?,
        );
        let xmaxRangeHi = u64::from_le_bytes(
            buf[60..68]
                .try_into()
                .map_err(|_| ZyronError::InvalidZyrFile("failed to read xmax_range_hi".into()))?,
        );
        let primaryKeyColumnId = u32::from_le_bytes([buf[68], buf[69], buf[70], buf[71]]);
        let sortOrder = SortOrder::from_u8(buf[72])?;

        Ok(Self {
            format_version: formatVersion,
            column_count: columnCount,
            row_count: rowCount,
            table_id: tableId,
            xmin_range_lo: xminRangeLo,
            xmin_range_hi: xminRangeHi,
            xmax_range_lo: xmaxRangeLo,
            xmax_range_hi: xmaxRangeHi,
            primary_key_column_id: primaryKeyColumnId,
            sort_order: sortOrder,
        })
    }
}

/// Computes header checksum over [0..12] and [16..FILE_HEADER_METADATA_SIZE],
/// skipping the checksum field itself at [12..16].
fn header_checksum(buf: &[u8; PAGE_SIZE]) -> u32 {
    // Concatenate the two regions logically and pass to zyr_checksum.
    // The header metadata is small (124 bytes), so a stack copy is fine.
    let mut data = [0u8; FILE_HEADER_METADATA_SIZE - 4];
    data[0..12].copy_from_slice(&buf[0..12]);
    data[12..].copy_from_slice(&buf[16..FILE_HEADER_METADATA_SIZE]);
    zyr_checksum(&data)
}

// ---------------------------------------------------------------------------
// zyr_checksum - 4-lane parallel multiply-XOR
// ---------------------------------------------------------------------------

/// 4-lane parallel multiply-XOR checksum.
/// Processes 32 bytes per iteration with independent multiply chains.
/// The CPU pipelines all 4 lanes (3-cycle multiply latency, 4 independent
/// chains). Covers all bytes with no sampling.
pub fn zyr_checksum(data: &[u8]) -> u32 {
    const P0: u64 = 0x517cc1b727220a95;
    const P1: u64 = 0x6c62272e07bb0143;
    const P2: u64 = 0x8ebc6af09c88c6e3;
    const P3: u64 = 0x305f1d4b1e0e2a6f;
    const FINAL: u64 = 0xff51afd7ed558ccd;

    // Seed lane 0 with data length to distinguish different-sized inputs.
    let mut s0: u64 = P0 ^ (data.len() as u64);
    let mut s1: u64 = P1;
    let mut s2: u64 = P2;
    let mut s3: u64 = P3;

    let ptr = data.as_ptr();
    let len = data.len();
    let mut i = 0;

    // Process 32-byte chunks across 4 independent lanes.
    while i + 32 <= len {
        unsafe {
            s0 = (s0 ^ (ptr.add(i) as *const u64).read_unaligned()).wrapping_mul(P0);
            s1 = (s1 ^ (ptr.add(i + 8) as *const u64).read_unaligned()).wrapping_mul(P1);
            s2 = (s2 ^ (ptr.add(i + 16) as *const u64).read_unaligned()).wrapping_mul(P2);
            s3 = (s3 ^ (ptr.add(i + 24) as *const u64).read_unaligned()).wrapping_mul(P3);
        }
        i += 32;
    }

    // Tail: remaining 8-byte words into lane 0.
    while i + 8 <= len {
        s0 = (s0 ^ unsafe { (ptr.add(i) as *const u64).read_unaligned() }).wrapping_mul(P0);
        i += 8;
    }

    // Tail: remaining < 8 bytes.
    if i < len {
        let mut tail: u64 = 0;
        unsafe {
            std::ptr::copy_nonoverlapping(ptr.add(i), &mut tail as *mut u64 as *mut u8, len - i);
        }
        s0 = (s0 ^ tail).wrapping_mul(P0);
    }

    // Combine all 4 lanes and finalize to 32 bits.
    let mut h = s0 ^ s1 ^ s2 ^ s3;
    h ^= h >> 33;
    h = h.wrapping_mul(FINAL);
    h ^= h >> 33;
    h as u32
}

// ---------------------------------------------------------------------------
// Segment index entry (in-memory representation)
// ---------------------------------------------------------------------------

/// In-memory representation of one segment index entry from the footer.
#[derive(Debug, Clone)]
struct SegmentIndexEntry {
    columnId: u32,
    offset: u64,
    size: u64,
}

// ---------------------------------------------------------------------------
// ZyrFileWriter
// ---------------------------------------------------------------------------

/// Writes a .zyr columnar file using a temporary path for atomic rename.
pub struct ZyrFileWriter {
    writer: BufWriter<File>,
    tmpPath: PathBuf,
    finalPath: PathBuf,
    #[allow(dead_code)]
    header: ZyrFileHeader,
    segmentIndex: Vec<SegmentIndexEntry>,
    currentOffset: u64,
}

impl ZyrFileWriter {
    /// Creates a new writer. Writes the file header to a temporary file
    /// at `path.with_extension("zyr.tmp")`.
    pub fn create(path: &Path, header: ZyrFileHeader) -> Result<Self> {
        let finalPath = path.to_path_buf();
        let tmpPath = path.with_extension("zyr.tmp");

        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmpPath)
            .map_err(|e| {
                ZyronError::IoError(format!(
                    "failed to create temp file {}: {}",
                    tmpPath.display(),
                    e
                ))
            })?;

        let mut writer = BufWriter::new(file);

        let headerBytes = header.to_bytes();
        writer
            .write_all(&headerBytes)
            .map_err(|e| ZyronError::IoError(format!("failed to write file header: {}", e)))?;

        Ok(Self {
            writer,
            tmpPath,
            finalPath,
            header,
            segmentIndex: Vec::new(),
            currentOffset: FILE_HEADER_SIZE as u64,
        })
    }

    /// Writes a column segment. The segment consists of a 128-byte header,
    /// optional bloom filter bytes, zone map bytes, and encoded column data.
    /// The combined output is padded to the next PAGE_SIZE boundary.
    pub fn write_segment(
        &mut self,
        columnId: u32,
        headerBytes: &[u8; SEGMENT_HEADER_SIZE],
        bloomBytes: Option<&[u8]>,
        zoneMapBytes: &[u8],
        encodedData: &[u8],
    ) -> Result<()> {
        let segmentStart = self.currentOffset;

        // Write segment header.
        self.writer
            .write_all(headerBytes)
            .map_err(|e| ZyronError::IoError(format!("failed to write segment header: {}", e)))?;

        // Write bloom filter if present.
        let bloomLen = if let Some(bloom) = bloomBytes {
            self.writer
                .write_all(bloom)
                .map_err(|e| ZyronError::IoError(format!("failed to write bloom filter: {}", e)))?;
            bloom.len()
        } else {
            0
        };

        // Write zone map data.
        self.writer
            .write_all(zoneMapBytes)
            .map_err(|e| ZyronError::IoError(format!("failed to write zone map: {}", e)))?;

        // Write encoded column data.
        self.writer
            .write_all(encodedData)
            .map_err(|e| ZyronError::IoError(format!("failed to write encoded data: {}", e)))?;

        let rawLen = SEGMENT_HEADER_SIZE + bloomLen + zoneMapBytes.len() + encodedData.len();
        let paddedLen = round_up_to_page(rawLen);
        let padBytes = paddedLen - rawLen;

        if padBytes > 0 {
            // Write zeroed padding. Stack buffer for small pads, heap for larger.
            let zeroes = vec![0u8; padBytes];
            self.writer.write_all(&zeroes).map_err(|e| {
                ZyronError::IoError(format!("failed to write segment padding: {}", e))
            })?;
        }

        self.segmentIndex.push(SegmentIndexEntry {
            columnId,
            offset: segmentStart,
            size: paddedLen as u64,
        });

        self.currentOffset += paddedLen as u64;
        Ok(())
    }

    /// Writes the footer (segment index + trailer), flushes, optionally fsyncs,
    /// and renames the temp file to the final path. Returns the final file size.
    pub fn finalize(mut self, fsync: bool) -> Result<u64> {
        let segmentIndexOffset = self.currentOffset;

        // Write segment index entries.
        for entry in &self.segmentIndex {
            let mut entryBuf = [0u8; SEGMENT_INDEX_ENTRY_SIZE];
            entryBuf[0..4].copy_from_slice(&entry.columnId.to_le_bytes());
            entryBuf[4..12].copy_from_slice(&entry.offset.to_le_bytes());
            entryBuf[12..20].copy_from_slice(&entry.size.to_le_bytes());
            self.writer.write_all(&entryBuf).map_err(|e| {
                ZyronError::IoError(format!("failed to write segment index entry: {}", e))
            })?;
        }

        // Write segment_index_offset.
        self.writer
            .write_all(&segmentIndexOffset.to_le_bytes())
            .map_err(|e| {
                ZyronError::IoError(format!("failed to write segment_index_offset: {}", e))
            })?;

        // Write magic repeat.
        self.writer
            .write_all(&ZYR_MAGIC)
            .map_err(|e| ZyronError::IoError(format!("failed to write footer magic: {}", e)))?;

        // Compute file checksum over all bytes written so far.
        // Flush the buffer first, then re-read the file for checksumming.
        self.writer
            .flush()
            .map_err(|e| ZyronError::IoError(format!("failed to flush before checksum: {}", e)))?;

        let fileChecksum = compute_file_checksum(&self.tmpPath)?;

        self.writer
            .write_all(&fileChecksum.to_le_bytes())
            .map_err(|e| ZyronError::IoError(format!("failed to write file checksum: {}", e)))?;

        self.writer
            .flush()
            .map_err(|e| ZyronError::IoError(format!("failed to flush writer: {}", e)))?;

        if fsync {
            self.writer
                .get_ref()
                .sync_all()
                .map_err(|e| ZyronError::IoError(format!("failed to fsync temp file: {}", e)))?;
        }

        // Get final file size before dropping writer.
        let fileSize = self
            .writer
            .get_ref()
            .metadata()
            .map_err(|e| ZyronError::IoError(format!("failed to read temp file metadata: {}", e)))?
            .len();

        // Drop writer to release file handle before rename.
        drop(self.writer);

        std::fs::rename(&self.tmpPath, &self.finalPath).map_err(|e| {
            ZyronError::IoError(format!(
                "failed to rename {} to {}: {}",
                self.tmpPath.display(),
                self.finalPath.display(),
                e
            ))
        })?;

        if fsync {
            // Fsync the parent directory to persist the rename.
            if let Some(parentDir) = self.finalPath.parent()
                && let Ok(dir) = File::open(parentDir)
            {
                let _ = dir.sync_all();
            }
        }

        Ok(fileSize)
    }
}

/// Computes zyr_checksum over the entire file contents at the given path.
fn compute_file_checksum(path: &Path) -> Result<u32> {
    let data = std::fs::read(path).map_err(|e| {
        ZyronError::IoError(format!(
            "failed to read file for checksum {}: {}",
            path.display(),
            e
        ))
    })?;
    Ok(zyr_checksum(&data))
}

/// Rounds `size` up to the next PAGE_SIZE multiple.
#[inline]
fn round_up_to_page(size: usize) -> usize {
    (size + PAGE_SIZE - 1) & !(PAGE_SIZE - 1)
}

// ---------------------------------------------------------------------------
// ZyrFileReader
// ---------------------------------------------------------------------------

/// Reads a .zyr columnar file. Validates header and footer on open.
pub struct ZyrFileReader {
    path: PathBuf,
    header: ZyrFileHeader,
    segmentIndex: Vec<SegmentIndexEntry>,
    #[allow(dead_code)]
    fileSize: u64,
}

impl ZyrFileReader {
    /// Opens a .zyr file at `path`. Reads and validates the header and footer.
    pub fn open(path: &Path) -> Result<Self> {
        let filePath = path.to_path_buf();
        let mut file = BufReader::new(File::open(path).map_err(|e| {
            ZyronError::IoError(format!(
                "failed to open .zyr file {}: {}",
                path.display(),
                e
            ))
        })?);

        let fileSize = file
            .get_ref()
            .metadata()
            .map_err(|e| {
                ZyronError::IoError(format!(
                    "failed to read .zyr file metadata {}: {}",
                    path.display(),
                    e
                ))
            })?
            .len();

        let minSize = (FILE_HEADER_SIZE + FOOTER_SIZE) as u64;
        if fileSize < minSize {
            return Err(ZyronError::InvalidZyrFile(format!(
                "file too small: {} bytes (minimum {})",
                fileSize, minSize
            )));
        }

        // Read file header (first PAGE_SIZE bytes).
        let mut headerBuf = [0u8; PAGE_SIZE];
        file.read_exact(&mut headerBuf)
            .map_err(|e| ZyronError::IoError(format!("failed to read file header: {}", e)))?;
        let header = ZyrFileHeader::from_bytes(&headerBuf)?;

        // Read footer trailer: last FOOTER_SIZE bytes = segment_index_offset(8) + magic(8) + checksum(4).
        let trailerStart = fileSize - FOOTER_SIZE as u64;
        file.seek(SeekFrom::Start(trailerStart))
            .map_err(|e| ZyronError::IoError(format!("failed to seek to footer: {}", e)))?;
        let mut trailerBuf = [0u8; FOOTER_SIZE];
        file.read_exact(&mut trailerBuf)
            .map_err(|e| ZyronError::IoError(format!("failed to read footer trailer: {}", e)))?;

        // Parse trailer fields.
        let segmentIndexOffset = u64::from_le_bytes(trailerBuf[0..8].try_into().map_err(|_| {
            ZyronError::InvalidZyrFile("failed to read segment_index_offset".into())
        })?);

        let footerMagic: [u8; 8] = trailerBuf[8..16]
            .try_into()
            .map_err(|_| ZyronError::InvalidZyrFile("failed to read footer magic".into()))?;
        if footerMagic != ZYR_MAGIC {
            return Err(ZyronError::InvalidZyrFile(
                "invalid magic bytes in footer".into(),
            ));
        }

        let storedFileChecksum = u32::from_le_bytes([
            trailerBuf[16],
            trailerBuf[17],
            trailerBuf[18],
            trailerBuf[19],
        ]);

        // Validate file checksum. The checksum covers all bytes before the
        // checksum field itself (file_size - 4).
        let checksumLen = fileSize - 4;
        file.seek(SeekFrom::Start(0)).map_err(|e| {
            ZyronError::IoError(format!("failed to seek for checksum validation: {}", e))
        })?;
        let mut checksumData = vec![0u8; checksumLen as usize];
        file.read_exact(&mut checksumData).map_err(|e| {
            ZyronError::IoError(format!(
                "failed to read file for checksum validation: {}",
                e
            ))
        })?;
        let computedFileChecksum = zyr_checksum(&checksumData);
        if storedFileChecksum != computedFileChecksum {
            return Err(ZyronError::InvalidZyrFile(format!(
                "file checksum mismatch: stored 0x{:08x}, computed 0x{:08x}",
                storedFileChecksum, computedFileChecksum
            )));
        }

        // Read segment index entries.
        let indexRegionSize = (trailerStart - segmentIndexOffset) as usize;
        if !indexRegionSize.is_multiple_of(SEGMENT_INDEX_ENTRY_SIZE) {
            return Err(ZyronError::InvalidZyrFile(format!(
                "segment index region size {} is not a multiple of entry size {}",
                indexRegionSize, SEGMENT_INDEX_ENTRY_SIZE
            )));
        }
        let entryCount = indexRegionSize / SEGMENT_INDEX_ENTRY_SIZE;

        file.seek(SeekFrom::Start(segmentIndexOffset))
            .map_err(|e| ZyronError::IoError(format!("failed to seek to segment index: {}", e)))?;

        let mut segmentIndex = Vec::with_capacity(entryCount);
        for _ in 0..entryCount {
            let mut entryBuf = [0u8; SEGMENT_INDEX_ENTRY_SIZE];
            file.read_exact(&mut entryBuf).map_err(|e| {
                ZyronError::IoError(format!("failed to read segment index entry: {}", e))
            })?;
            let columnId = u32::from_le_bytes([entryBuf[0], entryBuf[1], entryBuf[2], entryBuf[3]]);
            let offset = u64::from_le_bytes(entryBuf[4..12].try_into().map_err(|_| {
                ZyronError::InvalidZyrFile("failed to parse segment offset".into())
            })?);
            let size =
                u64::from_le_bytes(entryBuf[12..20].try_into().map_err(|_| {
                    ZyronError::InvalidZyrFile("failed to parse segment size".into())
                })?);
            segmentIndex.push(SegmentIndexEntry {
                columnId,
                offset,
                size,
            });
        }

        Ok(Self {
            path: filePath,
            header,
            segmentIndex,
            fileSize,
        })
    }

    /// Returns a reference to the file header.
    pub fn header(&self) -> &ZyrFileHeader {
        &self.header
    }

    /// Returns the number of column segments in the file.
    pub fn segment_count(&self) -> usize {
        self.segmentIndex.len()
    }

    /// Reads the raw segment bytes for the given column_id. Returns the full
    /// page-aligned segment data (header + bloom + zone maps + encoded data +
    /// padding).
    pub fn read_segment_raw(&self, columnId: u32) -> Result<Vec<u8>> {
        let entry = self
            .segmentIndex
            .iter()
            .find(|e| e.columnId == columnId)
            .ok_or_else(|| {
                ZyronError::InvalidZyrFile(format!("no segment found for column_id {}", columnId))
            })?;

        let mut file = BufReader::new(File::open(&self.path).map_err(|e| {
            ZyronError::IoError(format!(
                "failed to reopen .zyr file {}: {}",
                self.path.display(),
                e
            ))
        })?);

        file.seek(SeekFrom::Start(entry.offset)).map_err(|e| {
            ZyronError::IoError(format!(
                "failed to seek to segment for column {}: {}",
                columnId, e
            ))
        })?;

        let mut buf = vec![0u8; entry.size as usize];
        file.read_exact(&mut buf).map_err(|e| {
            ZyronError::IoError(format!(
                "failed to read segment for column {}: {}",
                columnId, e
            ))
        })?;

        Ok(buf)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    /// Builds a deterministic 128-byte segment header with the column_id
    /// stamped in the first 4 bytes.
    fn make_segment_header(columnId: u32) -> [u8; SEGMENT_HEADER_SIZE] {
        let mut hdr = [0u8; SEGMENT_HEADER_SIZE];
        hdr[0..4].copy_from_slice(&columnId.to_le_bytes());
        hdr
    }

    #[test]
    fn test_roundtrip_header_and_two_segments() {
        let dir = tempdir().expect("failed to create temp dir");
        let filePath = dir.path().join("test.zyr");

        let header = ZyrFileHeader {
            format_version: ZYR_FORMAT_VERSION,
            column_count: 2,
            row_count: 1000,
            table_id: 42,
            xmin_range_lo: 100,
            xmin_range_hi: 500,
            xmax_range_lo: 0,
            xmax_range_hi: u64::MAX,
            primary_key_column_id: 0,
            sort_order: SortOrder::Asc,
        };

        // Write the file.
        {
            let mut writer =
                ZyrFileWriter::create(&filePath, header.clone()).expect("create writer");

            // Segment 0: 128-byte header + 256 bytes zone map + 4096 bytes data.
            let segHdr0 = make_segment_header(0);
            let zoneMap0 = vec![0xAAu8; 256];
            let data0 = vec![0x11u8; 4096];
            writer
                .write_segment(0, &segHdr0, None, &zoneMap0, &data0)
                .expect("write segment 0");

            // Segment 1: 128-byte header + 64 bytes bloom + 128 bytes zone map + 2048 bytes data.
            let segHdr1 = make_segment_header(1);
            let bloom1 = vec![0xBBu8; 64];
            let zoneMap1 = vec![0xCCu8; 128];
            let data1 = vec![0x22u8; 2048];
            writer
                .write_segment(1, &segHdr1, Some(&bloom1), &zoneMap1, &data1)
                .expect("write segment 1");

            let fileSize = writer.finalize(false).expect("finalize");
            assert!(fileSize > 0);
        }

        // Read the file back.
        let reader = ZyrFileReader::open(&filePath).expect("open reader");

        // Validate header fields.
        let rh = reader.header();
        assert_eq!(rh.format_version, ZYR_FORMAT_VERSION);
        assert_eq!(rh.column_count, 2);
        assert_eq!(rh.row_count, 1000);
        assert_eq!(rh.table_id, 42);
        assert_eq!(rh.xmin_range_lo, 100);
        assert_eq!(rh.xmin_range_hi, 500);
        assert_eq!(rh.xmax_range_lo, 0);
        assert_eq!(rh.xmax_range_hi, u64::MAX);
        assert_eq!(rh.primary_key_column_id, 0);
        assert_eq!(rh.sort_order, SortOrder::Asc);

        // Validate segment count.
        assert_eq!(reader.segment_count(), 2);

        // Read segment 0 and verify contents.
        let seg0 = reader.read_segment_raw(0).expect("read segment 0");
        assert_eq!(seg0.len() % PAGE_SIZE, 0, "segment 0 not page-aligned");
        // First 4 bytes = column_id.
        assert_eq!(u32::from_le_bytes([seg0[0], seg0[1], seg0[2], seg0[3]]), 0);
        // Zone map starts at offset 128 (after segment header, no bloom).
        assert_eq!(seg0[SEGMENT_HEADER_SIZE], 0xAA);
        assert_eq!(seg0[SEGMENT_HEADER_SIZE + 255], 0xAA);
        // Encoded data starts at 128 + 256 = 384.
        let dataStart0 = SEGMENT_HEADER_SIZE + 256;
        assert_eq!(seg0[dataStart0], 0x11);
        assert_eq!(seg0[dataStart0 + 4095], 0x11);

        // Read segment 1 and verify contents.
        let seg1 = reader.read_segment_raw(1).expect("read segment 1");
        assert_eq!(seg1.len() % PAGE_SIZE, 0, "segment 1 not page-aligned");
        assert_eq!(u32::from_le_bytes([seg1[0], seg1[1], seg1[2], seg1[3]]), 1);
        // Bloom starts at 128.
        assert_eq!(seg1[SEGMENT_HEADER_SIZE], 0xBB);
        assert_eq!(seg1[SEGMENT_HEADER_SIZE + 63], 0xBB);
        // Zone map starts at 128 + 64 = 192.
        let zmStart1 = SEGMENT_HEADER_SIZE + 64;
        assert_eq!(seg1[zmStart1], 0xCC);
        assert_eq!(seg1[zmStart1 + 127], 0xCC);
        // Encoded data starts at 128 + 64 + 128 = 320.
        let dataStart1 = SEGMENT_HEADER_SIZE + 64 + 128;
        assert_eq!(seg1[dataStart1], 0x22);
        assert_eq!(seg1[dataStart1 + 2047], 0x22);
    }

    #[test]
    fn test_checksum_deterministic_and_detects_changes() {
        let data = b"hello world, this is a checksum test with enough bytes to exercise lanes";
        let c1 = zyr_checksum(data);
        let c2 = zyr_checksum(data);
        assert_eq!(c1, c2, "checksum must be deterministic");

        // Flipping one bit should change the checksum.
        let mut modified = data.to_vec();
        modified[10] ^= 0x01;
        let c3 = zyr_checksum(&modified);
        assert_ne!(c1, c3, "checksum should detect single-bit flip");

        // Empty data should produce a valid checksum.
        let c4 = zyr_checksum(&[]);
        let c5 = zyr_checksum(&[0u8; 1]);
        assert_ne!(c4, c5, "empty vs single-zero should differ");

        // Different lengths should produce different checksums.
        let c6 = zyr_checksum(&[0x42; 32]);
        let c7 = zyr_checksum(&[0x42; 33]);
        assert_ne!(
            c6, c7,
            "different lengths should produce different checksums"
        );
    }

    #[test]
    fn test_invalid_magic_detection() {
        let dir = tempdir().expect("failed to create temp dir");
        let filePath = dir.path().join("bad_magic.zyr");

        // Write a valid file first.
        let header = ZyrFileHeader {
            format_version: ZYR_FORMAT_VERSION,
            column_count: 0,
            row_count: 0,
            table_id: 1,
            xmin_range_lo: 0,
            xmin_range_hi: 0,
            xmax_range_lo: 0,
            xmax_range_hi: 0,
            primary_key_column_id: 0,
            sort_order: SortOrder::None,
        };

        let writer = ZyrFileWriter::create(&filePath, header).expect("create writer");
        writer.finalize(false).expect("finalize");

        // Corrupt the magic bytes in the file header.
        let mut fileData = std::fs::read(&filePath).expect("read file");
        fileData[0] = b'X';
        std::fs::write(&filePath, &fileData).expect("write corrupted file");

        let result = ZyrFileReader::open(&filePath);
        assert!(result.is_err());
        let errMsg = format!("{}", result.err().expect("expected error"));
        assert!(
            errMsg.contains("invalid magic"),
            "error should mention invalid magic, got: {}",
            errMsg
        );
    }

    #[test]
    fn test_file_checksum_detects_corruption() {
        let dir = tempdir().expect("failed to create temp dir");
        let filePath = dir.path().join("corrupt.zyr");

        let header = ZyrFileHeader {
            format_version: ZYR_FORMAT_VERSION,
            column_count: 1,
            row_count: 10,
            table_id: 99,
            xmin_range_lo: 0,
            xmin_range_hi: 100,
            xmax_range_lo: 0,
            xmax_range_hi: 0,
            primary_key_column_id: 0,
            sort_order: SortOrder::Desc,
        };

        let mut writer = ZyrFileWriter::create(&filePath, header).expect("create writer");
        let segHdr = make_segment_header(0);
        let zoneMap = vec![0xFFu8; 64];
        let data = vec![0xEEu8; 512];
        writer
            .write_segment(0, &segHdr, None, &zoneMap, &data)
            .expect("write segment");
        writer.finalize(false).expect("finalize");

        // Corrupt a byte in the middle of the segment data region.
        let mut fileData = std::fs::read(&filePath).expect("read file");
        let corruptOffset = PAGE_SIZE + SEGMENT_HEADER_SIZE + 100;
        if corruptOffset < fileData.len() {
            fileData[corruptOffset] ^= 0xFF;
            std::fs::write(&filePath, &fileData).expect("write corrupted file");

            let result = ZyrFileReader::open(&filePath);
            assert!(result.is_err());
            let errMsg = format!("{}", result.err().expect("expected error"));
            assert!(
                errMsg.contains("checksum mismatch"),
                "error should mention checksum mismatch, got: {}",
                errMsg
            );
        }
    }

    #[test]
    fn test_header_serialization_roundtrip() {
        let header = ZyrFileHeader {
            format_version: ZYR_FORMAT_VERSION,
            column_count: 5,
            row_count: 123456789,
            table_id: u64::MAX,
            xmin_range_lo: 1,
            xmin_range_hi: 999,
            xmax_range_lo: 50,
            xmax_range_hi: 500,
            primary_key_column_id: 3,
            sort_order: SortOrder::Desc,
        };

        let bytes = header.to_bytes();
        let recovered = ZyrFileHeader::from_bytes(&bytes).expect("from_bytes");

        assert_eq!(recovered.format_version, header.format_version);
        assert_eq!(recovered.column_count, header.column_count);
        assert_eq!(recovered.row_count, header.row_count);
        assert_eq!(recovered.table_id, header.table_id);
        assert_eq!(recovered.xmin_range_lo, header.xmin_range_lo);
        assert_eq!(recovered.xmin_range_hi, header.xmin_range_hi);
        assert_eq!(recovered.xmax_range_lo, header.xmax_range_lo);
        assert_eq!(recovered.xmax_range_hi, header.xmax_range_hi);
        assert_eq!(
            recovered.primary_key_column_id,
            header.primary_key_column_id
        );
        assert_eq!(recovered.sort_order, header.sort_order);
    }

    #[test]
    fn test_sort_order_from_u8() {
        assert_eq!(SortOrder::from_u8(0).expect("0"), SortOrder::None);
        assert_eq!(SortOrder::from_u8(1).expect("1"), SortOrder::Asc);
        assert_eq!(SortOrder::from_u8(2).expect("2"), SortOrder::Desc);
        assert!(SortOrder::from_u8(3).is_err());
        assert!(SortOrder::from_u8(255).is_err());
    }

    #[test]
    fn test_round_up_to_page() {
        assert_eq!(round_up_to_page(0), 0);
        assert_eq!(round_up_to_page(1), PAGE_SIZE);
        assert_eq!(round_up_to_page(PAGE_SIZE), PAGE_SIZE);
        assert_eq!(round_up_to_page(PAGE_SIZE + 1), PAGE_SIZE * 2);
        assert_eq!(round_up_to_page(PAGE_SIZE * 3), PAGE_SIZE * 3);
        assert_eq!(round_up_to_page(PAGE_SIZE * 3 - 1), PAGE_SIZE * 3);
    }

    #[test]
    fn test_missing_column_returns_error() {
        let dir = tempdir().expect("failed to create temp dir");
        let filePath = dir.path().join("no_col.zyr");

        let header = ZyrFileHeader {
            format_version: ZYR_FORMAT_VERSION,
            column_count: 1,
            row_count: 0,
            table_id: 1,
            xmin_range_lo: 0,
            xmin_range_hi: 0,
            xmax_range_lo: 0,
            xmax_range_hi: 0,
            primary_key_column_id: 0,
            sort_order: SortOrder::None,
        };

        let mut writer = ZyrFileWriter::create(&filePath, header).expect("create writer");
        let segHdr = make_segment_header(0);
        writer
            .write_segment(0, &segHdr, None, &[], &[0u8; 128])
            .expect("write segment");
        writer.finalize(false).expect("finalize");

        let reader = ZyrFileReader::open(&filePath).expect("open reader");
        assert!(reader.read_segment_raw(0).is_ok());
        assert!(reader.read_segment_raw(999).is_err());
    }
}

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

use super::bloom::BloomFilter;
use super::constants::{
    FILE_HEADER_METADATA_SIZE, FILE_HEADER_SIZE, FOOTER_SIZE, SEGMENT_HEADER_SIZE,
    SEGMENT_INDEX_ENTRY_SIZE, ZYR_FORMAT_VERSION, ZYR_MAGIC,
};
use super::segment::{SegmentHeader, ZoneMapEntry};
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
        let checksum = {
            let mut h = zyron_common::Hasher::new();
            h.update(&buf[0..12]);
            h.finish_phase();
            h.update(&buf[16..FILE_HEADER_METADATA_SIZE]);
            h.finish32()
        };
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
        let computedChecksum = {
            let mut h = zyron_common::Hasher::new();
            h.update(&buf[0..12]);
            h.finish_phase();
            h.update(&buf[16..FILE_HEADER_METADATA_SIZE]);
            h.finish32()
        };
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
    /// optional bloom filter bytes, zone map bytes, the null bitmap, and
    /// encoded column data, in that order. The combined output is padded to
    /// the next PAGE_SIZE boundary. The null bitmap is empty when the column
    /// has no nulls; readers derive its length from the header null_count and
    /// the file row_count.
    pub fn write_segment(
        &mut self,
        columnId: u32,
        headerBytes: &[u8; SEGMENT_HEADER_SIZE],
        bloomBytes: Option<&[u8]>,
        zoneMapBytes: &[u8],
        nullBitmap: &[u8],
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

        // Write the null bitmap (empty when the column has no nulls).
        self.writer
            .write_all(nullBitmap)
            .map_err(|e| ZyronError::IoError(format!("failed to write null bitmap: {}", e)))?;

        // Write encoded column data.
        self.writer
            .write_all(encodedData)
            .map_err(|e| ZyronError::IoError(format!("failed to write encoded data: {}", e)))?;

        let rawLen = SEGMENT_HEADER_SIZE
            + bloomLen
            + zoneMapBytes.len()
            + nullBitmap.len()
            + encodedData.len();
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

        // Write segment index entries, CRCing them as they go. The footer
        // checksum covers only this index region (offsets/sizes), not the
        // bulk payload: per-segment data is protected by SegmentHeader
        // data_checksum, the file header by its own header_checksum, so a
        // read never needs a whole-file pass to be integrity-safe.
        let mut indexHasher = zyron_common::Hasher::new();
        for entry in &self.segmentIndex {
            let mut entryBuf = [0u8; SEGMENT_INDEX_ENTRY_SIZE];
            entryBuf[0..4].copy_from_slice(&entry.columnId.to_le_bytes());
            entryBuf[4..12].copy_from_slice(&entry.offset.to_le_bytes());
            entryBuf[12..20].copy_from_slice(&entry.size.to_le_bytes());
            indexHasher.update(&entryBuf);
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

        // Footer checksum = CRC of the segment-index region only (computed
        // in memory above, no file re-read).
        let indexChecksum = indexHasher.finish32();
        self.writer
            .write_all(&indexChecksum.to_le_bytes())
            .map_err(|e| ZyronError::IoError(format!("failed to write index checksum: {}", e)))?;

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

/// Where each part of one raw segment buffer starts and ends.
///
/// The layout is header, bloom, zone maps, null bitmap, encoded payload,
/// padding. Every reader that wants one part has to skip the ones before
/// it, so the arithmetic lives here once rather than being repeated and
/// drifting between the paths that decode a column, evaluate a predicate
/// on it, and read it back through the scan's single-open batch reader
pub struct SegmentRegions {
    pub header: SegmentHeader,
    pub null_bitmap: std::ops::Range<usize>,
    pub encoded: std::ops::Range<usize>,
}

impl SegmentRegions {
    /// The encoded payload, checked against the checksum the writer
    /// recorded over exactly these bytes.
    ///
    /// Granular integrity: the bytes are already in memory and are exactly
    /// the ones about to be decoded, so this costs no extra IO, and the
    /// header self-verified its own checksum when it was parsed
    pub fn verified_payload<'a>(&self, raw: &'a [u8], column_id: u32) -> Result<&'a [u8]> {
        let enc = &raw[self.encoded.clone()];
        let crc = zyron_common::hash32(enc);
        if crc != self.header.data_checksum {
            return Err(ZyronError::InvalidZyrFile(format!(
                "segment payload checksum mismatch for column {}: stored 0x{:08x}, computed 0x{:08x}",
                column_id, self.header.data_checksum, crc
            )));
        }
        Ok(enc)
    }
}

/// Splits one raw segment buffer into its parts. `column_id` names the
/// column in error messages only
pub fn segment_regions(raw: &[u8], column_id: u32, row_count: usize) -> Result<SegmentRegions> {
    use super::constants::{ZONE_MAP_BATCH_SIZE, ZONE_MAP_ENTRY_SIZE};
    if raw.len() < SEGMENT_HEADER_SIZE {
        return Err(ZyronError::InvalidZyrFile(format!(
            "segment for column {} is {} bytes, shorter than its header",
            column_id,
            raw.len()
        )));
    }
    let mut header_bytes = [0u8; SEGMENT_HEADER_SIZE];
    header_bytes.copy_from_slice(&raw[..SEGMENT_HEADER_SIZE]);
    let header = SegmentHeader::from_bytes(&header_bytes)?;
    let zones = row_count.div_ceil(ZONE_MAP_BATCH_SIZE as usize);
    let null_start =
        SEGMENT_HEADER_SIZE + header.bloom_filter_size as usize + zones * ZONE_MAP_ENTRY_SIZE;
    let encoded_start = null_start
        + if header.null_count > 0 {
            row_count.div_ceil(8)
        } else {
            0
        };
    let encoded_end = encoded_start + header.encoded_size as usize;
    if raw.len() < encoded_end {
        return Err(ZyronError::InvalidZyrFile(format!(
            "segment for column {} truncated: {} bytes, need {}",
            column_id,
            raw.len(),
            encoded_end
        )));
    }
    Ok(SegmentRegions {
        header,
        null_bitmap: null_start..encoded_start,
        encoded: encoded_start..encoded_end,
    })
}

impl ZyrFileReader {
    /// Opens a .zyr file with granular integrity validation and no
    /// whole-file pass: the file header is checked by its own
    /// header_checksum, the footer magic and the segment-index region by the
    /// footer checksum, and each SegmentHeader self-verifies its header_crc
    /// when parsed. A segment's encoded payload is verified by the reader
    /// against SegmentHeader.data_checksum over exactly the bytes it reads to
    /// decode (zero extra IO), so a metadata aggregate that never reads a
    /// payload never pays for it, while corruption is still detected on any
    /// byte actually consumed.
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

        // Read the segment-index region once, verify the footer checksum
        // over exactly that region (offsets/sizes only — small, bounded by
        // metadata not data, no whole-file pass), then parse entries from
        // the in-memory buffer.
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
        let mut indexBytes = vec![0u8; indexRegionSize];
        file.read_exact(&mut indexBytes)
            .map_err(|e| ZyronError::IoError(format!("failed to read segment index: {}", e)))?;
        let computedIndexChecksum = zyron_common::hash32(&indexBytes);
        if storedFileChecksum != computedIndexChecksum {
            return Err(ZyronError::InvalidZyrFile(format!(
                "segment index checksum mismatch: stored 0x{:08x}, computed 0x{:08x}",
                storedFileChecksum, computedIndexChecksum
            )));
        }

        let mut segmentIndex = Vec::with_capacity(entryCount);
        for k in 0..entryCount {
            let entryBuf: [u8; SEGMENT_INDEX_ENTRY_SIZE] = indexBytes
                [k * SEGMENT_INDEX_ENTRY_SIZE..(k + 1) * SEGMENT_INDEX_ENTRY_SIZE]
                .try_into()
                .map_err(|_| {
                    ZyronError::InvalidZyrFile("failed to slice segment index entry".into())
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

    /// Whether this file holds a segment for one column.
    ///
    /// Answered from the segment index the open already read, so a column
    /// the file predates is detected without touching the filesystem
    pub fn has_segment(&self, column_id: u32) -> bool {
        self.segmentIndex.iter().any(|e| e.columnId == column_id)
    }

    /// Bytes on disk of one column's segment, header through padding.
    ///
    /// Answered from the segment index the open already read, so a scan can
    /// report what a column cost it without a second pass over the file. Zero
    /// for a column this file predates, which is also what reading it costs.
    pub fn segment_bytes(&self, column_id: u32) -> u64 {
        self.segmentIndex
            .iter()
            .find(|e| e.columnId == column_id)
            .map(|e| e.size)
            .unwrap_or(0)
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

    /// Reads and fully decodes one column segment, returning the decoded
    /// value bytes and the null bitmap (empty when the segment has no nulls).
    /// Verifies the encoded payload checksum before decoding. value_size is
    /// the fixed cell width, 0 for varlen columns
    pub fn decode_column(
        &self,
        column_id: u32,
        row_count: usize,
        value_size: usize,
    ) -> Result<(Vec<u8>, Vec<u8>)> {
        let raw = self.read_segment_raw(column_id)?;
        let regions = segment_regions(&raw, column_id, row_count)?;
        let null_bitmap = raw[regions.null_bitmap.clone()].to_vec();
        let enc = regions.verified_payload(&raw, column_id)?;
        let decoded = crate::encoding::create_encoding(regions.header.encoding_type)
            .decode(enc, row_count, value_size)?;
        Ok((decoded, null_bitmap))
    }

    /// Reads and decodes rows `start..end` of one column segment.
    ///
    /// A point read wants a few rows out of a segment holding millions.
    /// Decoding the whole segment to reach them makes the cost of one row
    /// the cost of the column, which is what an index is supposed to
    /// remove. Encodings that can address a row without replaying the ones
    /// before it pay for the range alone, and the rest fall back to the
    /// decode they would have done anyway.
    ///
    /// The segment payload is still read from disk in full, because it is
    /// one contiguous region and a partial read would cost a second seek
    /// for no gain at these sizes. What this removes is the decode
    pub fn decode_column_range(
        &self,
        column_id: u32,
        row_count: usize,
        value_size: usize,
        start: usize,
        end: usize,
    ) -> Result<(Vec<u8>, Vec<u8>)> {
        let raw = self.read_segment_raw(column_id)?;
        let regions = segment_regions(&raw, column_id, row_count)?;
        let (start, end) = crate::encoding::clamp_range(row_count, start, end);
        let null_bitmap = raw[regions.null_bitmap.clone()].to_vec();
        let enc = regions.verified_payload(&raw, column_id)?;
        let decoded = crate::encoding::create_encoding(regions.header.encoding_type)
            .decode_range(enc, row_count, value_size, start, end)?;
        Ok((decoded, null_bitmap))
    }

    /// Reads one column's header and zone maps without touching its
    /// payload, in a single file open.
    ///
    /// A zone covers ZONE_MAP_BATCH_SIZE rows and holds their bounds, and
    /// a segment's own bounds are the union of its zones, so a segment can
    /// admit a range that no zone holds. Rejecting it there costs the
    /// header and the zone region, which is bounded by the row count
    /// rather than by the data. Bounds are raw little endian value bytes
    /// in a 32-byte slot, compared with `compare_value_to_slot` under the
    /// column's own signedness.
    ///
    /// The two are returned together because the zone region's offset
    /// depends on the header's bloom size, so reading one already pays for
    /// the other.
    pub fn read_segment_metadata(
        &self,
        column_id: u32,
        row_count: usize,
    ) -> Result<(SegmentHeader, Vec<ZoneMapEntry>)> {
        use super::constants::{ZONE_MAP_BATCH_SIZE, ZONE_MAP_ENTRY_SIZE};
        let entry = self
            .segmentIndex
            .iter()
            .find(|e| e.columnId == column_id)
            .ok_or_else(|| {
                ZyronError::InvalidZyrFile(format!("no segment found for column_id {}", column_id))
            })?;
        let mut file = File::open(&self.path).map_err(|e| {
            ZyronError::IoError(format!(
                "failed to reopen .zyr file {}: {}",
                self.path.display(),
                e
            ))
        })?;
        file.seek(SeekFrom::Start(entry.offset))
            .map_err(|e| ZyronError::IoError(format!("failed to seek to segment header: {}", e)))?;
        let mut header_bytes = [0u8; SEGMENT_HEADER_SIZE];
        file.read_exact(&mut header_bytes)
            .map_err(|e| ZyronError::IoError(format!("failed to read segment header: {}", e)))?;
        let header = SegmentHeader::from_bytes(&header_bytes)?;

        let zones = row_count.div_ceil(ZONE_MAP_BATCH_SIZE as usize);
        let region = zones * ZONE_MAP_ENTRY_SIZE;
        let start = (SEGMENT_HEADER_SIZE + header.bloom_filter_size as usize) as u64;
        if start + region as u64 > entry.size {
            return Err(ZyronError::InvalidZyrFile(format!(
                "zone maps for column {} run past their segment: need {} bytes of {}",
                column_id,
                start + region as u64,
                entry.size
            )));
        }
        // The header sits at the segment start and the bloom follows it, so
        // the zone region begins wherever the bloom ends
        if header.bloom_filter_size > 0 {
            file.seek(SeekFrom::Start(entry.offset + start))
                .map_err(|e| {
                    ZyronError::IoError(format!(
                        "failed to seek to zone maps for column {}: {}",
                        column_id, e
                    ))
                })?;
        }
        let mut buf = vec![0u8; region];
        file.read_exact(&mut buf).map_err(|e| {
            ZyronError::IoError(format!(
                "failed to read zone maps for column {}: {}",
                column_id, e
            ))
        })?;
        let mut out = Vec::with_capacity(zones);
        for z in 0..zones {
            let slice: [u8; ZONE_MAP_ENTRY_SIZE] = buf
                [z * ZONE_MAP_ENTRY_SIZE..(z + 1) * ZONE_MAP_ENTRY_SIZE]
                .try_into()
                .map_err(|_| {
                    ZyronError::InvalidZyrFile("failed to slice zone map entry".into())
                })?;
            out.push(ZoneMapEntry::from_bytes(&slice));
        }
        Ok((header, out))
    }

    /// Evaluates a predicate against one column without decoding it,
    /// returning a packed bitmask of ceil(row_count / 8) bytes.
    ///
    /// Dictionary, run length and constant encodings answer from their
    /// compact form, so a column whose decoded size is orders of magnitude
    /// larger than its encoded size costs only the encoded size here. An
    /// encoding with no such shortcut falls back to the trait's default,
    /// which decodes, and the caller is no worse off than if it had
    /// decoded itself
    pub fn eval_column_predicate(
        &self,
        column_id: u32,
        row_count: usize,
        value_size: usize,
        predicate: &crate::encoding::Predicate<'_>,
    ) -> Result<Vec<u8>> {
        let raw = self.read_segment_raw(column_id)?;
        let regions = segment_regions(&raw, column_id, row_count)?;
        let enc = regions.verified_payload(&raw, column_id)?;
        crate::encoding::create_encoding(regions.header.encoding_type).eval_predicate(
            enc,
            row_count,
            value_size,
            predicate,
        )
    }


    /// Reads only the SEGMENT_HEADER_SIZE-byte header for a column, without
    /// pulling the encoded data. This is the metadata-only path used by
    /// aggregate pushdown (MIN/MAX/COUNT answered from the header), so a
    /// `SELECT MAX(c)` over a clean segment costs one small header read
    /// instead of decoding every row.
    pub fn read_segment_header_bytes(&self, columnId: u32) -> Result<[u8; SEGMENT_HEADER_SIZE]> {
        let entry = self
            .segmentIndex
            .iter()
            .find(|e| e.columnId == columnId)
            .ok_or_else(|| {
                ZyronError::InvalidZyrFile(format!("no segment found for column_id {}", columnId))
            })?;
        let mut file = File::open(&self.path).map_err(|e| {
            ZyronError::IoError(format!(
                "failed to reopen .zyr file {}: {}",
                self.path.display(),
                e
            ))
        })?;
        file.seek(SeekFrom::Start(entry.offset))
            .map_err(|e| ZyronError::IoError(format!("failed to seek to segment header: {}", e)))?;
        let mut buf = [0u8; SEGMENT_HEADER_SIZE];
        file.read_exact(&mut buf)
            .map_err(|e| ZyronError::IoError(format!("failed to read segment header: {}", e)))?;
        Ok(buf)
    }

    /// Reads and parses one column's segment header, without its data.
    ///
    /// This is the metadata-only path an aggregate takes when MIN, MAX or
    /// COUNT is answerable from the header alone, so `SELECT MAX(c)` over
    /// a clean segment costs one small read instead of decoding every row
    pub fn read_segment_header(&self, column_id: u32) -> Result<SegmentHeader> {
        let bytes = self.read_segment_header_bytes(column_id)?;
        SegmentHeader::from_bytes(&bytes)
    }

    /// Reads a column's value bloom filter without touching its data.
    ///
    /// Returns None when the segment carries no bloom, which is the answer
    /// for a low-cardinality or dictionary-encoded column. The bloom offset
    /// in the header is segment relative, so the read lands at the segment's
    /// file offset plus that value.
    pub fn read_bloom(&self, columnId: u32) -> Result<Option<BloomFilter>> {
        let entry = self
            .segmentIndex
            .iter()
            .find(|e| e.columnId == columnId)
            .ok_or_else(|| {
                ZyronError::InvalidZyrFile(format!("no segment found for column_id {}", columnId))
            })?;
        let headerBytes = self.read_segment_header_bytes(columnId)?;
        let header = super::segment::SegmentHeader::from_bytes(&headerBytes)?;
        if header.bloom_filter_size == 0 {
            return Ok(None);
        }
        let end = header.bloom_filter_offset + header.bloom_filter_size as u64;
        if end > entry.size {
            return Err(ZyronError::InvalidZyrFile(format!(
                "bloom filter for column {} runs past its segment: needs {} bytes of {}",
                columnId, end, entry.size
            )));
        }

        let mut file = File::open(&self.path).map_err(|e| {
            ZyronError::IoError(format!(
                "failed to reopen .zyr file {}: {}",
                self.path.display(),
                e
            ))
        })?;
        file.seek(SeekFrom::Start(entry.offset + header.bloom_filter_offset))
            .map_err(|e| {
                ZyronError::IoError(format!(
                    "failed to seek to bloom filter for column {}: {}",
                    columnId, e
                ))
            })?;
        let mut buf = vec![0u8; header.bloom_filter_size as usize];
        file.read_exact(&mut buf).map_err(|e| {
            ZyronError::IoError(format!(
                "failed to read bloom filter for column {}: {}",
                columnId, e
            ))
        })?;
        BloomFilter::from_bytes(&buf).map(Some)
    }

    /// Reads several column segments with a single file open, invoking `f`
    /// with the raw bytes for each requested column as soon as it is read.
    /// The raw buffer is reused across columns, so peak raw memory is one
    /// segment instead of the sum of every requested segment. The callback
    /// decodes the bytes before the next column overwrites the buffer. `f`
    /// receives the request-order index and `None` when the column is absent.
    pub fn read_segments_each<F>(&self, column_ids: &[u32], mut f: F) -> Result<()>
    where
        F: FnMut(usize, Option<&[u8]>) -> Result<()>,
    {
        let mut file = BufReader::new(File::open(&self.path).map_err(|e| {
            ZyronError::IoError(format!(
                "failed to reopen .zyr file {}: {}",
                self.path.display(),
                e
            ))
        })?);
        let mut buf: Vec<u8> = Vec::new();
        for (idx, &cid) in column_ids.iter().enumerate() {
            match self.segmentIndex.iter().find(|e| e.columnId == cid) {
                Some(entry) => {
                    file.seek(SeekFrom::Start(entry.offset)).map_err(|e| {
                        ZyronError::IoError(format!("failed to seek to segment: {}", e))
                    })?;
                    let need = entry.size as usize;
                    buf.clear();
                    buf.reserve(need);
                    // SAFETY: read_exact writes exactly `need` bytes into the
                    // slice or returns an error, so no element is observed
                    // before initialization. Avoids the resize zero-fill on
                    // the scan hot path.
                    #[allow(clippy::uninit_vec)]
                    unsafe {
                        buf.set_len(need);
                    }
                    file.read_exact(&mut buf).map_err(|e| {
                        ZyronError::IoError(format!("failed to read segment: {}", e))
                    })?;
                    f(idx, Some(&buf))?;
                }
                None => f(idx, None)?,
            }
        }
        Ok(())
    }

    /// Returns the file-level row count from the header.
    pub fn row_count(&self) -> u64 {
        self.header.row_count
    }

    /// What the file's rows are ordered by, if anything.
    ///
    /// `Asc` means `primary_key_column_id` is genuinely sorted and can be
    /// binary searched. A file laid out by a multi-dimensional curve
    /// reports `None`, because no single column is sorted in it
    pub fn sort_order(&self) -> SortOrder {
        self.header.sort_order
    }

    /// The column `sort_order` refers to, meaningless when that is `None`.
    pub fn primary_key_column_id(&self) -> u32 {
        self.header.primary_key_column_id
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
                .write_segment(0, &segHdr0, None, &zoneMap0, &[], &data0)
                .expect("write segment 0");

            // Segment 1: 128-byte header + 64 bytes bloom + 128 bytes zone map + 2048 bytes data.
            let segHdr1 = make_segment_header(1);
            let bloom1 = vec![0xBBu8; 64];
            let zoneMap1 = vec![0xCCu8; 128];
            let data1 = vec![0x22u8; 2048];
            writer
                .write_segment(1, &segHdr1, Some(&bloom1), &zoneMap1, &[], &data1)
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
        let c1 = zyron_common::hash32(data);
        let c2 = zyron_common::hash32(data);
        assert_eq!(c1, c2, "checksum must be deterministic");

        // Flipping one bit should change the checksum.
        let mut modified = data.to_vec();
        modified[10] ^= 0x01;
        let c3 = zyron_common::hash32(&modified);
        assert_ne!(c1, c3, "checksum should detect single-bit flip");

        // Empty data should produce a valid checksum.
        let c4 = zyron_common::hash32(&[]);
        let c5 = zyron_common::hash32(&[0u8; 1]);
        assert_ne!(c4, c5, "empty vs single-zero should differ");

        // Different lengths should produce different checksums.
        let c6 = zyron_common::hash32(&[0x42; 32]);
        let c7 = zyron_common::hash32(&[0x42; 33]);
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
    fn test_segment_index_corruption_detected_at_open() {
        // Granular integrity model: open() validates the file header (its
        // own header_checksum) and the segment-index region (footer
        // checksum). Corruption inside the index region must be caught at
        // open. Segment-payload corruption is caught when the segment is
        // read (SegmentHeader.data_checksum), not at open, so it is not part
        // of this test.
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
            .write_segment(0, &segHdr, None, &zoneMap, &[], &data)
            .expect("write segment");
        writer.finalize(false).expect("finalize");

        // A clean file opens.
        ZyrFileReader::open(&filePath).expect("clean file opens");

        // Corrupt a byte in the segment-index region (the
        // SEGMENT_INDEX_ENTRY_SIZE bytes immediately before the 20-byte
        // trailer). open() must reject it.
        let mut fileData = std::fs::read(&filePath).expect("read file");
        let idxByte = fileData.len() - FOOTER_SIZE - 1;
        fileData[idxByte] ^= 0xFF;
        std::fs::write(&filePath, &fileData).expect("write corrupted file");

        let result = ZyrFileReader::open(&filePath);
        assert!(result.is_err(), "index corruption must be detected");
        let errMsg = format!("{}", result.err().expect("expected error"));
        assert!(
            errMsg.contains("index checksum mismatch"),
            "error should mention index checksum mismatch, got: {}",
            errMsg
        );
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
            .write_segment(0, &segHdr, None, &[], &[], &[0u8; 128])
            .expect("write segment");
        writer.finalize(false).expect("finalize");

        let reader = ZyrFileReader::open(&filePath).expect("open reader");
        assert!(reader.read_segment_raw(0).is_ok());
        assert!(reader.read_segment_raw(999).is_err());
    }
}

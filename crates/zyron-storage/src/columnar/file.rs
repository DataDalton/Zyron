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
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
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
    /// Byte offset of the segment index, and its size in bytes.
    ///
    /// The index sits at the end of the file, ahead of the footer trailer,
    /// and where it lands is only known once every segment is written. The
    /// writer records it here after the fact, by rewriting the header page
    /// before the file is fsynced and renamed into place.
    ///
    /// Carrying it in the header is what lets an open be three system calls
    /// rather than five. Without it a reader asks the filesystem how large
    /// the file is, reads the trailer at the end to learn where the index
    /// starts, and only then reads the index. With it, the header page says
    /// where the index is and one further read takes the index and the
    /// trailer together. An open costs the same whether the file holds a
    /// thousand rows or a million, so on a table of small files it is the
    /// scan, and every call crosses whatever filter driver the platform has
    /// in the path.
    ///
    /// Zero in both means the header was written before its index existed,
    /// which is every header until the file it belongs to is finalized
    pub segment_index_offset: u64,
    pub segment_index_size: u32,
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
        // [73..80] reserved, already zeroed.
        buf[80..88].copy_from_slice(&self.segment_index_offset.to_le_bytes());
        buf[88..92].copy_from_slice(&self.segment_index_size.to_le_bytes());
        // [92..128] reserved, already zeroed.
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

        let segmentIndexOffset = u64::from_le_bytes(buf[80..88].try_into().map_err(|_| {
            ZyronError::InvalidZyrFile("failed to read segment_index_offset".into())
        })?);
        let segmentIndexSize = u32::from_le_bytes([buf[88], buf[89], buf[90], buf[91]]);

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
            segment_index_offset: segmentIndexOffset,
            segment_index_size: segmentIndexSize,
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
    ///
    /// Returns the padded region this column occupies, which is what a
    /// reader pays to read the column and what a cost model comparing two
    /// access paths has to compare. Deriving it later needs the file open,
    /// and the writer already knows it
    pub fn write_segment(
        &mut self,
        columnId: u32,
        headerBytes: &[u8; SEGMENT_HEADER_SIZE],
        bloomBytes: Option<&[u8]>,
        zoneMapBytes: &[u8],
        nullBitmap: &[u8],
        encodedData: &[u8],
    ) -> Result<u64> {
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
        Ok(paddedLen as u64)
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
        let indexSize = (self.segmentIndex.len() * SEGMENT_INDEX_ENTRY_SIZE) as u32;
        let indexChecksum = indexHasher.finish32();
        self.writer
            .write_all(&indexChecksum.to_le_bytes())
            .map_err(|e| ZyronError::IoError(format!("failed to write index checksum: {}", e)))?;

        // Rewrite the header page now that the index has a position, so a
        // reader learns where it is from the header rather than by asking
        // the filesystem for the file size and reading the trailer to find
        // out. This is one buffered write and a seek, paid once when the
        // file is built, against two system calls saved on every open the
        // file ever serves
        self.header.segment_index_offset = segmentIndexOffset;
        self.header.segment_index_size = indexSize;
        self.writer
            .flush()
            .map_err(|e| ZyronError::IoError(format!("failed to flush writer: {}", e)))?;
        self.writer
            .get_mut()
            .seek(SeekFrom::Start(0))
            .map_err(|e| ZyronError::IoError(format!("failed to seek to file header: {}", e)))?;
        self.writer
            .write_all(&self.header.to_bytes())
            .map_err(|e| ZyronError::IoError(format!("failed to rewrite file header: {}", e)))?;
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
            sync_parent_directory(&self.finalPath)?;
        }

        Ok(fileSize)
    }
}

/// Persists the directory entry a rename created.
///
/// Flushing a file covers its bytes, not the name that reaches them. On a
/// filesystem that can lose a directory entry independently of file
/// contents, a crash between the rename and the next metadata flush leaves
/// a committed version referencing a data file whose name never landed, so
/// this runs before the caller is told the file is durable.
///
/// Errors are returned rather than swallowed. A durability step that
/// quietly does nothing is worse than one that was never attempted, because
/// every caller above it believes the guarantee holds
#[cfg(not(windows))]
fn sync_parent_directory(path: &Path) -> Result<()> {
    let Some(parent) = path.parent() else {
        return Ok(());
    };
    let dir = File::open(parent).map_err(|e| {
        ZyronError::IoError(format!(
            "failed to open {} to persist a rename: {}",
            parent.display(),
            e
        ))
    })?;
    dir.sync_all().map_err(|e| {
        ZyronError::IoError(format!(
            "failed to fsync {} to persist a rename: {}",
            parent.display(),
            e
        ))
    })
}

/// Windows exposes no way to flush a directory, so there is nothing to call.
///
/// A directory handle opened with backup semantics is accepted and
/// `FlushFileBuffers` on it fails with access denied, measured rather than
/// assumed. Rename durability there comes from NTFS journaling the
/// operation as metadata and replaying it from the volume log, which is the
/// same guarantee the explicit flush reaches for elsewhere. Forcing it
/// would mean flushing a volume handle, which needs administrator rights
/// and flushes every other file on the volume with it.
///
/// This previously called `File::open` on the directory and discarded both
/// the open error and the flush error. `File::open` on a directory fails on
/// Windows, so the step did nothing on every Windows build while reading as
/// though it had run
#[cfg(windows)]
fn sync_parent_directory(_path: &Path) -> Result<()> {
    Ok(())
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
///
/// The handle stays open for the reader's life and every read is
/// positional, so reading N columns costs one file open rather than N. A
/// point lookup reads a handful of small regions out of two files, which
/// made the open the dominant cost of answering it.
///
/// Holding the handle does not pin the file. The format reclaims files
/// underneath readers, vacuum unlinking a version's files and compaction
/// and tier moves renaming them, and both keep working. Unix keeps the
/// inode alive until the last handle closes, and Rust opens with
/// `FILE_SHARE_DELETE` on Windows, which gives the same behaviour there.
/// `test_a_reader_survives_its_file_being_deleted` pins it
pub struct ZyrFileReader {
    path: PathBuf,
    /// Held open so no read has to reopen the file. Reads are positional,
    /// so this is shared rather than owned by one read at a time
    file: File,
    header: ZyrFileHeader,
    segmentIndex: Vec<SegmentIndexEntry>,
    /// Column id paired with its position in `segmentIndex`, sorted by id so
    /// a lookup is a binary search rather than a walk of every segment.
    ///
    /// A scan asks for a named column of every file it opens, so the walk is
    /// quadratic in the column count across a wide table. The index itself
    /// keeps the order the writer laid segments down, because a reader that
    /// wants them in file order still gets them that way
    segmentByColumn: Vec<(u32, u32)>,
    /// Bytes this reader has pulled off the file, counted where the reads
    /// happen rather than estimated from what a column occupies on disk.
    ///
    /// A decode reads a segment's tail and not its bloom, so a count derived
    /// from the segment size would report bytes no read asked for, and the
    /// figure is an enforced bound rather than a note
    ioBytes: AtomicU64,
    /// Bytes of column data read, meaning null bitmaps and encoded payloads
    /// and not the headers, blooms or zone maps that decide what to read.
    ///
    /// Kept apart from `ioBytes` because it is the quantity a cross format
    /// comparison rests on: the heap columnar scan excludes its own header
    /// and zone reads too, so the two report bytes of column data the query
    /// had to touch rather than everything the file system was asked for
    dataBytes: AtomicU64,
    #[allow(dead_code)]
    fileSize: u64,
}

/// Fills `buf` from `offset`, naming the region in any error so a failure
/// says which part of which file could not be read
/// What a read was for, named only when the read fails.
///
/// These are the reads a scan makes for every column of every file, and
/// building the description at the call site allocates and formats a string
/// on the success path to carry text nothing reads. Naming the purpose by
/// value costs nothing until an error branch asks it to render itself
#[derive(Clone, Copy)]
enum ReadPurpose {
    FileHeader,
    SegmentIndexAndFooter,
    SegmentHeader,
    Segment(u32),
    ZoneMaps(u32),
    BloomFilter(u32),
}

impl std::fmt::Display for ReadPurpose {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FileHeader => f.write_str("file header"),
            Self::SegmentIndexAndFooter => f.write_str("segment index and footer"),
            Self::SegmentHeader => f.write_str("segment header"),
            Self::Segment(column) => write!(f, "segment for column {}", column),
            Self::ZoneMaps(column) => write!(f, "zone maps for column {}", column),
            Self::BloomFilter(column) => write!(f, "bloom filter for column {}", column),
        }
    }
}

fn read_exact_at(
    file: &File,
    buf: &mut [u8],
    offset: u64,
    path: &Path,
    what: ReadPurpose,
) -> Result<()> {
    crate::disk::positional_read_exact(file, buf, offset).map_err(|e| {
        ZyronError::IoError(format!(
            "failed to read {} of .zyr file {} at offset {}: {}",
            what,
            path.display(),
            offset,
            e
        ))
    })
}

/// Reads `len` bytes at `offset` into a fresh buffer.
///
/// The buffer is not zeroed first. The read overwrites every byte of it or
/// fails, so a zero fill is pure cost, and a segment is a page or more. A
/// failed read drops the buffer without anything having looked at it
fn read_vec_at(
    file: &File,
    len: usize,
    offset: u64,
    path: &Path,
    what: ReadPurpose,
) -> Result<Vec<u8>> {
    let mut buf: Vec<u8> = Vec::with_capacity(len);
    // SAFETY: read_exact_at writes exactly `len` bytes into the slice or
    // returns an error, so no element is observed before initialization
    #[allow(clippy::uninit_vec)]
    unsafe {
        buf.set_len(len);
    }
    read_exact_at(file, &mut buf, offset, path, what)?;
    Ok(buf)
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
        let file = File::open(path).map_err(|e| {
            ZyronError::IoError(format!(
                "failed to open .zyr file {}: {}",
                path.display(),
                e
            ))
        })?;

        // Read file header (first PAGE_SIZE bytes). It names where the
        // segment index sits, which is what keeps this open to three system
        // calls: without it the file size has to be asked for and the
        // trailer read to find the index, and only then the index itself
        let mut headerBuf = [0u8; PAGE_SIZE];
        read_exact_at(&file, &mut headerBuf, 0, path, ReadPurpose::FileHeader)?;
        let header = ZyrFileHeader::from_bytes(&headerBuf)?;

        let segmentIndexOffset = header.segment_index_offset;
        let indexRegionSize = header.segment_index_size as usize;
        if segmentIndexOffset < FILE_HEADER_SIZE as u64 {
            return Err(ZyronError::InvalidZyrFile(format!(
                "segment index offset {} overlaps the file header",
                segmentIndexOffset
            )));
        }
        if !indexRegionSize.is_multiple_of(SEGMENT_INDEX_ENTRY_SIZE) {
            return Err(ZyronError::InvalidZyrFile(format!(
                "segment index region size {} is not a multiple of entry size {}",
                indexRegionSize, SEGMENT_INDEX_ENTRY_SIZE
            )));
        }
        let entryCount = indexRegionSize / SEGMENT_INDEX_ENTRY_SIZE;
        let trailerStart = segmentIndexOffset + indexRegionSize as u64;
        let fileSize = trailerStart + FOOTER_SIZE as u64;

        // The index and the trailer that describes it are adjacent, so one
        // read takes both. The checksum still covers exactly the index
        // region (offsets and sizes only, bounded by metadata rather than
        // data), so integrity costs no whole-file pass
        let tail = read_vec_at(
            &file,
            indexRegionSize + FOOTER_SIZE,
            segmentIndexOffset,
            path,
            ReadPurpose::SegmentIndexAndFooter,
        )?;
        let (indexBytes, trailerBuf) = tail.split_at(indexRegionSize);

        let footerMagic: [u8; 8] = trailerBuf[8..16]
            .try_into()
            .map_err(|_| ZyronError::InvalidZyrFile("failed to read footer magic".into()))?;
        if footerMagic != ZYR_MAGIC {
            return Err(ZyronError::InvalidZyrFile(
                "invalid magic bytes in footer".into(),
            ));
        }

        // The trailer carries its own copy of where the index starts. Two
        // records of one fact disagreeing means the file is damaged, and
        // reading them in the same call makes the cross-check free
        let trailerIndexOffset = u64::from_le_bytes(trailerBuf[0..8].try_into().map_err(|_| {
            ZyronError::InvalidZyrFile("failed to read segment_index_offset".into())
        })?);
        if trailerIndexOffset != segmentIndexOffset {
            return Err(ZyronError::InvalidZyrFile(format!(
                "segment index offset disagrees between header and footer: {} and {}",
                segmentIndexOffset, trailerIndexOffset
            )));
        }

        let storedFileChecksum = u32::from_le_bytes([
            trailerBuf[16],
            trailerBuf[17],
            trailerBuf[18],
            trailerBuf[19],
        ]);
        let computedIndexChecksum = zyron_common::hash32(indexBytes);
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

        // Column ids paired with their positions, sorted for lookup. Two
        // segments claiming one column would make every read of it depend on
        // which was found first, so the file is refused instead
        let mut segmentByColumn: Vec<(u32, u32)> = segmentIndex
            .iter()
            .enumerate()
            .map(|(i, e)| (e.columnId, i as u32))
            .collect();
        segmentByColumn.sort_unstable_by_key(|(id, _)| *id);
        if let Some(pair) = segmentByColumn.windows(2).find(|w| w[0].0 == w[1].0) {
            return Err(ZyronError::InvalidZyrFile(format!(
                "segment index names column {} more than once",
                pair[0].0
            )));
        }

        Ok(Self {
            path: filePath,
            file,
            header,
            segmentIndex,
            segmentByColumn,
            ioBytes: AtomicU64::new(PAGE_SIZE as u64 + indexRegionSize as u64 + FOOTER_SIZE as u64),
            dataBytes: AtomicU64::new(0),
            fileSize,
        })
    }

    /// Bytes read off this file since it was opened, including the header
    /// page and segment index the open itself read
    pub fn io_bytes(&self) -> u64 {
        self.ioBytes.load(Ordering::Relaxed)
    }

    /// Bytes of column data this reader has decoded or evaluated against,
    /// excluding the metadata reads that decide what to touch
    pub fn column_data_bytes(&self) -> u64 {
        self.dataBytes.load(Ordering::Relaxed)
    }

    #[inline]
    fn count_read(&self, bytes: usize) {
        self.ioBytes.fetch_add(bytes as u64, Ordering::Relaxed);
    }

    #[inline]
    fn count_data_read(&self, bytes: usize) {
        self.ioBytes.fetch_add(bytes as u64, Ordering::Relaxed);
        self.dataBytes.fetch_add(bytes as u64, Ordering::Relaxed);
    }

    /// The segment holding one column, found by binary search
    fn segment_for(&self, column_id: u32) -> Option<&SegmentIndexEntry> {
        let at = self
            .segmentByColumn
            .binary_search_by_key(&column_id, |(id, _)| *id)
            .ok()?;
        self.segmentIndex.get(self.segmentByColumn[at].1 as usize)
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
        self.segment_for(column_id).is_some()
    }

    /// Bytes on disk of one column's segment, header through padding.
    ///
    /// Answered from the segment index the open already read, so a scan can
    /// report what a column cost it without a second pass over the file. Zero
    /// for a column this file predates, which is also what reading it costs.
    pub fn segment_bytes(&self, column_id: u32) -> u64 {
        self.segment_for(column_id).map(|e| e.size).unwrap_or(0)
    }

    /// Reads the raw segment bytes for the given column_id. Returns the full
    /// page-aligned segment data (header + bloom + zone maps + encoded data +
    /// padding).
    pub fn read_segment_raw(&self, columnId: u32) -> Result<Vec<u8>> {
        let entry = self.segment_for(columnId).ok_or_else(|| {
            ZyronError::InvalidZyrFile(format!("no segment found for column_id {}", columnId))
        })?;

        self.count_read(entry.size as usize);
        read_vec_at(
            &self.file,
            entry.size as usize,
            entry.offset,
            &self.path,
            ReadPurpose::Segment(columnId),
        )
    }

    /// Reads only the parts of a segment a decode consumes: the null bitmap
    /// and the encoded payload, with the segment header that describes them.
    ///
    /// A segment is laid out header, bloom, zone maps, null bitmap, payload,
    /// so the two a decode wants are its tail and the two it does not are
    /// its head. Reading the whole segment to decode it therefore pays for a
    /// bloom sized at ten bits per row and a zone region sized at sixty four
    /// bytes per thousand, neither of which the decode looks at. On a
    /// quarter million row column that is most of the read: the bloom alone
    /// is over three hundred kilobytes where a well compressed payload is a
    /// few dozen bytes.
    ///
    /// Two positional reads replace one. The header has to be read before
    /// the tail can be located, and both are small next to what skipping the
    /// bloom saves
    fn read_segment_payload(
        &self,
        column_id: u32,
        row_count: usize,
    ) -> Result<(SegmentHeader, Vec<u8>, Vec<u8>)> {
        use super::constants::{ZONE_MAP_BATCH_SIZE, ZONE_MAP_ENTRY_SIZE};
        let entry = self.segment_for(column_id).ok_or_else(|| {
            ZyronError::InvalidZyrFile(format!("no segment found for column_id {}", column_id))
        })?;

        let mut header_bytes = [0u8; SEGMENT_HEADER_SIZE];
        read_exact_at(
            &self.file,
            &mut header_bytes,
            entry.offset,
            &self.path,
            ReadPurpose::SegmentHeader,
        )?;
        let header = SegmentHeader::from_bytes(&header_bytes)?;

        let zones = row_count.div_ceil(ZONE_MAP_BATCH_SIZE as usize);
        let null_start =
            SEGMENT_HEADER_SIZE + header.bloom_filter_size as usize + zones * ZONE_MAP_ENTRY_SIZE;
        let null_len = if header.null_count > 0 {
            row_count.div_ceil(8)
        } else {
            0
        };
        let tail_len = null_len + header.encoded_size as usize;
        if null_start as u64 + tail_len as u64 > entry.size {
            return Err(ZyronError::InvalidZyrFile(format!(
                "segment for column {} truncated: {} bytes, need {}",
                column_id,
                entry.size,
                null_start + tail_len
            )));
        }
        let tail = read_vec_at(
            &self.file,
            tail_len,
            entry.offset + null_start as u64,
            &self.path,
            ReadPurpose::Segment(column_id),
        )?;

        // The header locates the tail, so it is IO but not column data
        self.count_read(SEGMENT_HEADER_SIZE);
        self.count_data_read(tail_len);
        let (null_bitmap, encoded) = tail.split_at(null_len);
        let crc = zyron_common::hash32(encoded);
        if crc != header.data_checksum {
            return Err(ZyronError::InvalidZyrFile(format!(
                "segment payload checksum mismatch for column {}: stored 0x{:08x}, computed 0x{:08x}",
                column_id, header.data_checksum, crc
            )));
        }
        Ok((header, null_bitmap.to_vec(), encoded.to_vec()))
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
        let (header, null_bitmap, enc) = self.read_segment_payload(column_id, row_count)?;
        let decoded = crate::encoding::create_encoding(header.encoding_type)
            .decode(&enc, row_count, value_size)?;
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
        let (header, null_bitmap, enc) = self.read_segment_payload(column_id, row_count)?;
        let (start, end) = crate::encoding::clamp_range(row_count, start, end);
        let decoded = crate::encoding::create_encoding(header.encoding_type)
            .decode_range(&enc, row_count, value_size, start, end)?;
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
        let entry = self.segment_for(column_id).ok_or_else(|| {
            ZyronError::InvalidZyrFile(format!("no segment found for column_id {}", column_id))
        })?;
        let mut header_bytes = [0u8; SEGMENT_HEADER_SIZE];
        read_exact_at(
            &self.file,
            &mut header_bytes,
            entry.offset,
            &self.path,
            ReadPurpose::SegmentHeader,
        )?;
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
        let buf = read_vec_at(
            &self.file,
            region,
            entry.offset + start,
            &self.path,
            ReadPurpose::ZoneMaps(column_id),
        )?;
        let mut out = Vec::with_capacity(zones);
        for z in 0..zones {
            let slice: [u8; ZONE_MAP_ENTRY_SIZE] = buf
                [z * ZONE_MAP_ENTRY_SIZE..(z + 1) * ZONE_MAP_ENTRY_SIZE]
                .try_into()
                .map_err(|_| ZyronError::InvalidZyrFile("failed to slice zone map entry".into()))?;
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
        let (header, _, enc) = self.read_segment_payload(column_id, row_count)?;
        crate::encoding::create_encoding(header.encoding_type)
            .eval_predicate(&enc, row_count, value_size, predicate)
    }

    /// Evaluates a predicate over rows `start..end`, returning a keep mask
    /// of `ceil((end - start) / 8)` bytes covering that range alone.
    ///
    /// A caller that has already rejected most of a file from its zone maps
    /// needs an answer for the rows the surviving zones cover and nothing
    /// else. Every encoding can decode a range without replaying the rows
    /// before it, so answering for one zone costs one zone rather than the
    /// whole column.
    ///
    /// A full range keeps the encoding's own predicate evaluator, which is
    /// specialized per encoding and beats decoding and then comparing. The
    /// ranged path decodes because a predicate evaluator that works on
    /// encoded bytes has no way to start partway through a segment
    pub fn eval_column_predicate_rows(
        &self,
        column_id: u32,
        row_count: usize,
        value_size: usize,
        predicate: &crate::encoding::Predicate<'_>,
        start: usize,
        end: usize,
    ) -> Result<Vec<u8>> {
        if start == 0 && end >= row_count {
            return self.eval_column_predicate(column_id, row_count, value_size, predicate);
        }
        if start >= end {
            return Ok(Vec::new());
        }
        let (header, _, enc) = self.read_segment_payload(column_id, row_count)?;
        let decoded = crate::encoding::create_encoding(header.encoding_type)
            .decode_range(&enc, row_count, value_size, start, end)?;
        crate::encoding::eval_predicate_on_raw(&decoded, end - start, value_size, predicate)
    }

    /// Reads only the SEGMENT_HEADER_SIZE-byte header for a column, without
    /// pulling the encoded data. This is the metadata-only path used by
    /// aggregate pushdown (MIN/MAX/COUNT answered from the header), so a
    /// `SELECT MAX(c)` over a clean segment costs one small header read
    /// instead of decoding every row.
    pub fn read_segment_header_bytes(&self, columnId: u32) -> Result<[u8; SEGMENT_HEADER_SIZE]> {
        let entry = self.segment_for(columnId).ok_or_else(|| {
            ZyronError::InvalidZyrFile(format!("no segment found for column_id {}", columnId))
        })?;
        let mut buf = [0u8; SEGMENT_HEADER_SIZE];
        read_exact_at(
            &self.file,
            &mut buf,
            entry.offset,
            &self.path,
            ReadPurpose::SegmentHeader,
        )?;
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
        let entry = self.segment_for(columnId).ok_or_else(|| {
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

        let buf = read_vec_at(
            &self.file,
            header.bloom_filter_size as usize,
            entry.offset + header.bloom_filter_offset,
            &self.path,
            ReadPurpose::BloomFilter(columnId),
        )?;
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
        let mut buf: Vec<u8> = Vec::new();
        for (idx, &cid) in column_ids.iter().enumerate() {
            match self.segment_for(cid) {
                Some(entry) => {
                    let need = entry.size as usize;
                    buf.clear();
                    buf.reserve(need);
                    // SAFETY: positional_read_exact writes exactly `need`
                    // bytes into the slice or returns an error, so no
                    // element is observed before initialization. Avoids the
                    // resize zero-fill on the scan hot path.
                    #[allow(clippy::uninit_vec)]
                    unsafe {
                        buf.set_len(need);
                    }
                    read_exact_at(
                        &self.file,
                        &mut buf,
                        entry.offset,
                        &self.path,
                        ReadPurpose::Segment(entry.columnId),
                    )?;
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

    /// The reader holds its file open for its whole life, and the format
    /// deletes files underneath readers. Vacuum unlinks the files of a
    /// version nothing needs any more while a scan started before it may
    /// still be reading them, so an open reader must not make the unlink
    /// fail, and it must keep answering from the bytes it opened.
    ///
    /// This is what makes holding the handle safe rather than a change of
    /// reclamation semantics, and it is platform behaviour rather than
    /// anything this file does, which is exactly why it needs a test
    #[test]
    fn test_a_reader_survives_its_file_being_deleted() {
        let dir = tempdir().expect("failed to create temp dir");
        let filePath = dir.path().join("reclaimed.zyr");

        let header = ZyrFileHeader {
            format_version: ZYR_FORMAT_VERSION,
            column_count: 1,
            row_count: 64,
            table_id: 7,
            xmin_range_lo: 0,
            xmin_range_hi: 0,
            xmax_range_lo: 0,
            xmax_range_hi: u64::MAX,
            primary_key_column_id: 0,
            sort_order: SortOrder::None,
            segment_index_offset: 0,
            segment_index_size: 0,
        };
        let data = vec![0x5Au8; 4096];
        {
            let mut writer = ZyrFileWriter::create(&filePath, header).expect("create writer");
            writer
                .write_segment(0, &make_segment_header(0), None, &[], &[], &data)
                .expect("write segment");
            writer.finalize(false).expect("finalize");
        }

        let reader = ZyrFileReader::open(&filePath).expect("open reader");
        std::fs::remove_file(&filePath).expect("an open reader must not block reclamation");
        assert!(!filePath.exists());

        // The reader still answers, from metadata it already holds and from
        // bytes it reads after the unlink
        assert_eq!(reader.row_count(), 64);
        let raw = reader
            .read_segment_raw(0)
            .expect("read segment after unlink");
        assert_eq!(
            &raw[SEGMENT_HEADER_SIZE..SEGMENT_HEADER_SIZE + data.len()],
            &data[..]
        );
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
            segment_index_offset: 0,
            segment_index_size: 0,
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
            segment_index_offset: 0,
            segment_index_size: 0,
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
            segment_index_offset: 0,
            segment_index_size: 0,
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
            segment_index_offset: 0,
            segment_index_size: 0,
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
            segment_index_offset: 0,
            segment_index_size: 0,
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

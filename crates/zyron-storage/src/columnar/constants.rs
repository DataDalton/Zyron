//! Byte-level constants for the .zyr columnar file format.

use zyron_common::format::FormatVersion;
use zyron_common::format::version::VersionWindow;

/// Sentinel repeated in the trailer of a .zyr file. Not the file's format
/// identity, which the envelope in the header carries. This marks where the
/// trailer is so a truncated tail is told apart from a healthy one.
pub const ZYR_FOOTER_SENTINEL: [u8; 8] = *b"ZYRCOL\0\0";

/// Current .zyr format version, registered in the format registry.
///
/// 1.1 keeps the file header in 512 bytes and starts every column segment
/// on a 64-byte boundary. 1.0 padded the header and every segment to a
/// 16KB page
pub const ZYR_FORMAT_VERSION: FormatVersion = FormatVersion::new(1, 1);

/// The page-padded layout, which the reader still opens and the migration
/// moves forward
pub const ZYR_FORMAT_VERSION_1_0: FormatVersion = FormatVersion::V1;

/// Versions this binary reads. Every offset a reader uses comes from the
/// segment index, so one reader serves both layouts
pub const ZYR_READER_WINDOW: VersionWindow =
    VersionWindow::new(ZYR_FORMAT_VERSION_1_0, ZYR_FORMAT_VERSION);

/// Bytes the file header occupies before the first column segment.
///
/// The header itself is FILE_HEADER_METADATA_SIZE bytes and the rest is
/// room to grow. Nothing derives a segment's position from this, because
/// the segment index records an explicit offset and size for every column,
/// so the only thing it fixes is where the first segment can start
pub const FILE_HEADER_SIZE: usize = 512;

/// Byte boundary each column segment starts on.
///
/// A segment's header, bloom blocks and zone entries are all multiples of a
/// cache line, so aligning the segment start keeps every one of them
/// aligned within the file. Nothing reads a segment as pages: the header
/// read lands at the segment offset and every other read lands at an
/// offset past a bloom and a zone region whose sizes are the column's own,
/// so padding a segment out to a page boundary bought alignment for one
/// 128-byte read and spent up to a page per column to do it
pub const SEGMENT_ALIGNMENT: usize = 64;

/// Bytes of metadata in the file header before the padding region. The
/// first 20 are the format envelope, the rest are the file's own header
/// extension, which the envelope's header checksum covers.
pub const FILE_HEADER_METADATA_SIZE: usize = 128;

/// On-disk size of a SegmentHeader.
pub const SEGMENT_HEADER_SIZE: usize = 128;

/// Fixed-size slot for min/max stat values in segment headers.
pub const STAT_VALUE_SIZE: usize = 32;

/// Number of rows per zone map micro-batch.
pub const ZONE_MAP_BATCH_SIZE: u32 = 1024;

/// Size of one zone map entry: min(32) + max(32).
pub const ZONE_MAP_ENTRY_SIZE: usize = 64;

/// Segment index entry: column_id(4) + offset(8) + size(8).
pub const SEGMENT_INDEX_ENTRY_SIZE: usize = 20;

/// Footer: segment_index_offset(8) + magic(8) + file_checksum(4).
pub const FOOTER_SIZE: usize = 20;

/// Bloom filter bits per element for ~1% false positive rate.
pub const BLOOM_BITS_PER_ELEMENT: usize = 10;

/// Number of hash functions for bloom filter at 10 bits/element.
pub const BLOOM_HASH_COUNT: u32 = 7;

/// Minimum distinct value count to build a bloom filter. Below this
/// threshold, dictionary encoding provides exact membership lookup,
/// making a bloom filter redundant.
pub const BLOOM_MIN_CARDINALITY: u64 = 64;

/// Block size for split-block bloom filter (cache-line aligned).
pub const BLOOM_BLOCK_SIZE: usize = 64;

// ---------------------------------------------------------------------------
// Columnar-MVCC system columns
// ---------------------------------------------------------------------------
//
// Every .zyr carries three hidden system columns beside the user columns.
// Their column ids live in a reserved high range that user column ordinals
// never reach, so the segment index keys do not collide. Each is an ordinary
// encoded column with its own zone map, so per-zone MVCC min/max comes for
// free from the sys_xmin and sys_supersede zone maps.

/// Per-table monotonic row identity. Survives merges. Encodes as constant-step.
pub const SYS_COL_ROWID: u32 = u32::MAX;

/// Creating transaction id, widened from the heap u32 tuple header.
pub const SYS_COL_XMIN: u32 = u32::MAX - 1;

/// Transaction id that superseded this columnar version, 0 if never.
pub const SYS_COL_SUPERSEDE: u32 = u32::MAX - 2;

/// Lowest reserved system column id. User column ids are catalog ordinals and
/// never reach this range.
pub const SYS_COL_MIN: u32 = u32::MAX - 2;

/// First column id a shredded VARIANT path takes.
///
/// User column ids are catalog ordinals, which are u16, so the range above
/// them is free. Shredded ids are handed out in order within one fold and
/// recorded on the segment beside the path they hold, so nothing has to
/// derive an id from a path and no two paths can land on the same column.
pub const SHRED_COL_BASE: u32 = 1 << 20;

/// Highest column id a shredded path may take, one below the system range.
pub const SHRED_COL_MAX: u32 = SYS_COL_MIN - 1;

/// All three system columns are 8-byte values.
pub const SYS_COL_VALUE_SIZE: usize = 8;

// ---------------------------------------------------------------------------
// Columnar patch log
// ---------------------------------------------------------------------------

/// File extension for the per-table append-only columnar patch log. UPDATE and
/// DELETE of a columnar-resident row append epoch-tagged entries here. Never a
/// .zyr rewrite, never a heap round trip. Folded into base at merge.
pub const ZYRPATCH_EXTENSION: &str = "zyrpatch";

/// Magic bytes identifying a .zyrpatch log file.
pub const ZYRPATCH_MAGIC: [u8; 8] = *b"ZYRPT2\0\0";

/// Patch log record kinds.
pub const PATCH_KIND_VALUE: u8 = 1;
pub const PATCH_KIND_SUPERSEDE: u8 = 2;

/// Revokes one earlier value patch, written by ROLLBACK TO SAVEPOINT
pub const PATCH_KIND_REVOKE_VALUE: u8 = 3;

/// Revokes one earlier supersede, written by ROLLBACK TO SAVEPOINT
pub const PATCH_KIND_REVOKE_SUPERSEDE: u8 = 4;

/// Discards every overlay entry of one branch, written on DROP BRANCH and
/// after MERGE BRANCH folds the branch rows into the main line
pub const PATCH_KIND_BRANCH_CLEAR: u8 = 5;

/// Copies one row's main line overlay into a branch, written before the
/// branch's first write to that row so pre fork patches stay visible on
/// the branch while later main line writes to the row do not
pub const PATCH_KIND_BRANCH_COPY: u8 = 6;

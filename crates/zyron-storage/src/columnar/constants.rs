//! Byte-level constants for the .zyr columnar file format.

use zyron_common::page::PAGE_SIZE;

/// Magic bytes identifying a .zyr columnar file.
pub const ZYR_MAGIC: [u8; 8] = *b"ZYRCOL\0\0";

/// Current .zyr format version. The field exists so the format can be
/// versioned in the future. Only the current version is supported (reads of
/// any other version are rejected).
pub const ZYR_FORMAT_VERSION: u32 = 1;

/// File header occupies one full page for alignment.
pub const FILE_HEADER_SIZE: usize = PAGE_SIZE;

/// Bytes of metadata in the file header before the padding region.
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

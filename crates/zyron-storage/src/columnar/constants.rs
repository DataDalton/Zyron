//! Byte-level constants for the .zyr columnar file format.

use zyron_common::page::PAGE_SIZE;

/// Magic bytes identifying a .zyr columnar file.
pub const ZYR_MAGIC: [u8; 8] = *b"ZYRCOL\0\0";

/// Current .zyr format version.
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

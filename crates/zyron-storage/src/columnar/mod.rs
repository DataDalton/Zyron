#![allow(non_snake_case)]
//! Columnar storage for analytical workloads.
//!
//! Converts heap tuple data into column-oriented .zyr files for
//! scan queries. Background compaction materializes rows, sorts by
//! primary key, encodes per-column with type-specific strategies, and
//! writes page-aligned files with bloom filters and zone maps for
//! segment pruning.

pub mod bloom;
pub mod cache;
pub mod compaction;
pub mod constants;
pub mod file;
pub mod segment;
pub mod sorted;

pub use bloom::BloomFilter;
pub use cache::{SegmentCache, SegmentCacheKey, SegmentCacheStats};
pub use compaction::{
    ColumnDescriptor, CompactionConfig, CompactionInput, CompactionResult, CompactionThread,
};
pub use constants::*;
pub use file::{SortOrder, ZyrFileHeader, ZyrFileReader, ZyrFileWriter};
pub use segment::{
    ColumnSegment, SegmentHeader, ZoneMapEntry, compare_le_bytes, compare_stat_slots,
    value_to_stat_slot,
};
pub use sorted::{SortedSegmentEntry, SortedSegmentIndex};

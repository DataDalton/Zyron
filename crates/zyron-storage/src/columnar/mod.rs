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
pub mod patch;
pub mod segment;
pub mod sketch;
pub mod sorted;
pub mod tier;
pub mod wal_payload;

pub use bloom::{BloomFilter, might_contain_serialized};
pub use cache::{SegmentCache, SegmentCacheKey, SegmentCacheStats};
pub use compaction::{
    ColumnDescriptor, CompactionConfig, CompactionInput, CompactionResult, FileOrdering,
    cluster_curve, cluster_order, encode_and_write, run_compaction_cycle,
};
pub use constants::*;
pub use file::{
    SegmentRegions, SortOrder, ZyrFileHeader, ZyrFileReader, ZyrFileWriter, segment_regions,
};
pub use patch::{ColumnarPatchManager, PatchStore, RowOverlay, ValuePatch};
pub use segment::{
    BloomPolicy, ColumnSegment, SegmentHeader, SegmentOptions, SlotOrder, ZoneMapEntry,
    compare_le_bytes, compare_stat_slots, compare_stat_slots_typed, compare_value_to_slot,
    compare_values_ordered, slot_order, stat_slot_is_signed, value_to_stat_slot, varlen_upper_slot,
};
pub use sketch::DistinctSketch;
pub use sorted::{MergeScanIterator, SortedSegmentEntry, SortedSegmentIndex};
pub use tier::{TIER_DIR_NAME, columnar_root_for_segment, tier_segment_dir};
pub use wal_payload::{
    ColumnarBranchClearPayload, ColumnarPatchRevokePayload, ColumnarSupersedePayload,
    ColumnarValuePatchPayload,
};

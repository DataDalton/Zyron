//! B+Tree constants for page-based and arena-based implementations.

// Page-based B+Tree constants

/// Maximum key size in bytes.
pub const MAX_KEY_SIZE: usize = 256;

/// Minimum fill factor for B+ tree nodes (50%).
pub const MIN_FILL_FACTOR: f64 = 0.5;

// Arena-based B+Tree constants

/// 32KB nodes. With a write buffer in front, the B+Tree is read-optimized.
/// Large nodes keep the tree shallow so lookups traverse fewer levels.
/// The write buffer handles insert throughput, so node size is tuned purely
/// for read performance.
pub(crate) const ARENA_NODE_SIZE: usize = 32768;

/// Write buffer capacity (number of entries before flush to B+Tree).
/// 128K entries x 16 bytes = 2MB buffer size. Larger buffer reduces flush
/// frequency, halving radix sort + tree insert overhead for large key sets.
pub(crate) const WRITE_BUFFER_CAPACITY: usize = 131072;

/// Header size for internal nodes (16 bytes).
pub(crate) const ARENA_INTERNAL_HEADER_SIZE: usize = 16;

/// Header size for leaf nodes (24 bytes).
pub(crate) const ARENA_LEAF_HEADER_SIZE: usize = 24;

/// Child pointer size (8 bytes for u64 offset).
pub(crate) const ARENA_CHILD_SIZE: usize = 8;

/// Key size for arena nodes (8 bytes, stored as u64).
pub(crate) const ARENA_KEY_SIZE: usize = 8;

/// Entry size for leaf nodes (key + tuple_id = 16 bytes).
pub(crate) const ARENA_LEAF_ENTRY_SIZE: usize = 16;

/// Maximum keys per internal node: (32768 - 16 - 8) / (8 + 8) = 2046
pub(crate) const ARENA_MAX_INTERNAL_KEYS: usize =
    (ARENA_NODE_SIZE - ARENA_INTERNAL_HEADER_SIZE - ARENA_CHILD_SIZE)
        / (ARENA_KEY_SIZE + ARENA_CHILD_SIZE);

/// Maximum entries per leaf node: (32768 - 24) / 16 = 2046
pub(crate) const ARENA_MAX_LEAF_ENTRIES: usize =
    (ARENA_NODE_SIZE - ARENA_LEAF_HEADER_SIZE) / ARENA_LEAF_ENTRY_SIZE;

/// Sentinel offset indicating null/invalid.
pub(crate) const ARENA_NULL_OFFSET: u64 = u64::MAX;

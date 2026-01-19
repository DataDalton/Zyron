//! B+ tree index implementation with write buffer for HTAP workloads.
//!
//! This module provides three B+Tree implementations:
//!
//! ## BufferedBTreeIndex (Recommended for HTAP)
//!
//! A hybrid structure combining a write buffer with a read-optimized B+Tree:
//!
//! ```text
//! Writes → [WriteBuffer (sorted, 64K entries)] → merge → [B+Tree (32KB nodes)]
//!                                                              ↑
//! Reads → check buffer first, then ───────────────────────────┘
//! ```
//!
//! Performance characteristics:
//! - Insert: ~100ns (buffer insert with binary search)
//! - Lookup: ~80ns (buffer check + B+Tree with height=2)
//! - Range scan: Merges results from buffer and B+Tree
//!
//! The write buffer absorbs high-frequency inserts, while the B+Tree uses
//! 32KB nodes for shallow height (2 levels for 1M keys) and fast lookups.
//!
//! ## BTreeArenaIndex (Direct B+Tree)
//!
//! Arena-based B+Tree with contiguous memory allocation:
//! - 32KB nodes for shallow tree height
//! - Direct pointer arithmetic for fast traversal
//! - Lock-free reads with exclusive writes (&mut self)
//!
//! ## BTreeIndex (Page-Based)
//!
//! Traditional page-based B+Tree using the buffer pool:
//! - Integrates with disk manager for persistence
//! - Suitable for larger-than-memory indexes
//!
//! ## Memory Layout (Arena-Based)
//!
//! Internal node layout:
//! ```text
//! +------------------+ 0
//! | version: u64     | 8
//! | num_keys: u16    | 10
//! | level: u16       | 12
//! | reserved: u32    | 16 (HEADER_SIZE)
//! +------------------+
//! | child_0: u64     | 24
//! | key_0: u64       |
//! | child_1: u64     |
//! | ...              |
//! +------------------+
//! ```
//!
//! Leaf node layout:
//! ```text
//! +------------------+ 0
//! | version: u64     | 8
//! | num_entries: u16 | 10
//! | reserved: [u8;6] | 16
//! | next_leaf: u64   | 24 (LEAF_HEADER_SIZE)
//! +------------------+
//! | key_0: u64       |
//! | tuple_id_0: u64  |
//! | ...              |
//! +------------------+
//! ```
//!
//! With 32KB nodes: 2046 keys per node, height=2 for 1M keys.

// Submodules
pub mod arena;
pub mod arena_index;
pub mod buffer;
pub mod constants;
pub mod index;
pub mod page;
pub mod store;
pub mod types;

// Re-exports for public API
pub use arena_index::BTreeArenaIndex;
pub use buffer::BufferedBTreeIndex;
pub use constants::{MAX_KEY_SIZE, MIN_FILL_FACTOR};
pub use index::BTreeIndex;
pub use page::{BTreeInternalPage, BTreeLeafPage};
pub use types::{DeleteResult, InternalEntry, InternalPageHeader, LeafEntry, LeafPageHeader};

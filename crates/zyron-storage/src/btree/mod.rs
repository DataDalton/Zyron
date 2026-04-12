//! B+ tree index implementation with write buffer for HTAP workloads.
//!
//! This module provides three B+Tree implementations:
//!
//! ## BufferedBTreeIndex (Recommended for HTAP)
//!
//! A hybrid structure combining a write buffer with a read-optimized B+Tree:
//!
//! ```text
//! Writes -> [WriteBuffer (sorted, 64K entries)] -> merge -> [B+Tree (32KB nodes)]
//!                                                              ↑
//! Reads -> check buffer first, then ───────────────────────────┘
//! ```
//!
//! Operation shape:
//! - Insert: buffered insert into a partitioned write buffer.
//! - Lookup: buffer check, then a B+Tree traversal on miss.
//! - Range scan: merges results from the buffer and the B+Tree.
//!
//! The write buffer absorbs high-frequency inserts. The B+Tree uses 32KB
//! nodes so the tree stays shallow, which keeps the lookup path short.
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
pub mod checkpoint;
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

//! Arena-based memory allocation for B+Tree nodes.

use super::constants::ARENA_NODE_SIZE;
use std::sync::atomic::{AtomicU64, Ordering};

/// Internal node header layout.
/// +------------------+ 0
/// | version: u64     | 8  (for optimistic locking)
/// | num_keys: u16    | 10
/// | level: u16       | 12
/// | reserved: u32    | 16 (HEADER_SIZE)
/// +------------------+
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct ArenaInternalNodeHeader {
    pub(crate) version: u64,
    pub(crate) num_keys: u16,
    pub(crate) level: u16,
    pub(crate) reserved: u32,
}

/// Leaf node header layout.
/// +------------------+ 0
/// | version: u64     | 8
/// | num_entries: u16 | 10
/// | reserved: [u8;6] | 16
/// | next_leaf: u64   | 24 (LEAF_HEADER_SIZE)
/// +------------------+
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct ArenaLeafNodeHeader {
    pub(crate) version: u64,
    pub(crate) num_entries: u16,
    pub(crate) reserved: [u8; 6],
    pub(crate) next_leaf: u64,
}

/// Contiguous memory arena for B+Tree nodes.
/// Nodes are allocated sequentially, enabling direct pointer access.
pub struct BTreeArena {
    /// Contiguous memory region.
    data: Box<[u8]>,
    /// Current allocation offset (grows upward).
    alloc_offset: AtomicU64,
    /// Total capacity in bytes.
    capacity: usize,
}

impl BTreeArena {
    /// Creates a new arena with the specified capacity in nodes.
    pub fn new(num_nodes: usize) -> Self {
        let capacity = num_nodes * ARENA_NODE_SIZE;
        let data = vec![0u8; capacity].into_boxed_slice();
        Self {
            data,
            alloc_offset: AtomicU64::new(0),
            capacity,
        }
    }

    /// Allocates a new node, returns its offset.
    #[inline]
    pub fn allocate(&self) -> u64 {
        let offset = self
            .alloc_offset
            .fetch_add(ARENA_NODE_SIZE as u64, Ordering::Relaxed);
        if offset as usize + ARENA_NODE_SIZE > self.capacity {
            panic!("BTreeArena out of memory");
        }
        offset
    }

    /// Direct pointer access to node data (read).
    #[inline(always)]
    pub fn node_ptr(&self, offset: u64) -> *const u8 {
        debug_assert!((offset as usize) < self.capacity);
        unsafe { self.data.as_ptr().add(offset as usize) }
    }

    /// Direct pointer access to node data (write).
    #[inline(always)]
    pub fn node_ptr_mut(&mut self, offset: u64) -> *mut u8 {
        debug_assert!((offset as usize) < self.capacity);
        unsafe { self.data.as_mut_ptr().add(offset as usize) }
    }

    /// Reads internal node header (direct read, no volatility).
    #[inline(always)]
    pub fn read_internal_header(&self, offset: u64) -> ArenaInternalNodeHeader {
        unsafe {
            let ptr = self.node_ptr(offset) as *const ArenaInternalNodeHeader;
            *ptr
        }
    }

    /// Reads leaf node header (direct read, no volatility).
    #[inline(always)]
    pub fn read_leaf_header(&self, offset: u64) -> ArenaLeafNodeHeader {
        unsafe {
            let ptr = self.node_ptr(offset) as *const ArenaLeafNodeHeader;
            *ptr
        }
    }
}

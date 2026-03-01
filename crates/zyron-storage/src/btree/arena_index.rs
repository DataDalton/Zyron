//! Arena-based B+Tree index with bulk insert optimization.

use super::arena::{ArenaInternalNodeHeader, ArenaLeafNodeHeader, BTreeArena};
use super::constants::{
    ARENA_CHILD_SIZE, ARENA_INTERNAL_HEADER_SIZE, ARENA_KEY_SIZE, ARENA_LEAF_ENTRY_SIZE,
    ARENA_LEAF_HEADER_SIZE, ARENA_MAX_INTERNAL_KEYS, ARENA_MAX_LEAF_ENTRIES, ARENA_NULL_OFFSET,
};
use crate::tuple::TupleId;
use std::sync::atomic::{AtomicU64, Ordering};
use zyron_common::Result;
use zyron_common::page::PageId;

pub struct BTreeArenaIndex {
    /// Memory arena for all nodes.
    arena: BTreeArena,
    /// Root node offset.
    root_offset: AtomicU64,
    /// Tree height (1 = just root as leaf).
    height: AtomicU64,
}

impl BTreeArenaIndex {
    /// Creates a new arena-based B+Tree index.
    /// Capacity is the maximum number of nodes (each 4KB).
    pub fn new(capacity_nodes: usize) -> Self {
        let mut arena = BTreeArena::new(capacity_nodes.max(1024));

        // Allocate root as leaf node
        let root_offset = arena.allocate();

        // Initialize root leaf header
        unsafe {
            let header_ptr = arena.node_ptr_mut(root_offset) as *mut ArenaLeafNodeHeader;
            std::ptr::write(
                header_ptr,
                ArenaLeafNodeHeader {
                    version: 0,
                    num_entries: 0,
                    reserved: [0; 6],
                    next_leaf: ARENA_NULL_OFFSET,
                },
            );
        }

        Self {
            arena,
            root_offset: AtomicU64::new(root_offset),
            height: AtomicU64::new(1),
        }
    }

    /// Returns the tree height.
    #[inline]
    pub fn height(&self) -> u64 {
        self.height.load(Ordering::Acquire)
    }

    // =========================================================================
    // Lock-Free Read Path
    // =========================================================================

    /// Direct search without version overhead.
    /// Writes use &mut self which provides exclusive access.
    #[inline(always)]
    pub fn search(&self, key: &[u8]) -> Option<TupleId> {
        self.search_optimistic(key)
    }

    /// Direct search without version checking.
    /// Writes use &mut self which provides exclusive access.
    /// Relaxed ordering is safe since writes are exclusive.
    #[inline(always)]
    fn search_optimistic(&self, key: &[u8]) -> Option<TupleId> {
        let search_key = self.key_to_u64(key);
        let mut current_offset = self.root_offset.load(Ordering::Relaxed);
        let height = self.height.load(Ordering::Relaxed);

        // Fast path for height=1 (root is leaf)
        if height == 1 {
            return self.search_leaf_interpolation(current_offset, search_key);
        }

        // Traverse internal nodes with leaf prefetching
        // Height 2: 1 internal level, height 3: 2 internal levels, etc.
        let mut levels_remaining = height - 1;
        while levels_remaining > 0 {
            current_offset = self.traverse_internal_node_prefetch(
                current_offset,
                search_key,
                levels_remaining == 1,
            );
            levels_remaining -= 1;
        }

        // Search leaf node using interpolation search
        self.search_leaf_interpolation(current_offset, search_key)
    }

    /// Traverse internal node with optional prefetch of child node.
    /// When at last internal level, prefetches the target leaf node.
    #[inline(always)]
    fn traverse_internal_node_prefetch(
        &self,
        offset: u64,
        search_key: u64,
        prefetch_child: bool,
    ) -> u64 {
        let header = self.arena.read_internal_header(offset);
        let child_idx = self.find_child_idx_branchless(offset, search_key, header.num_keys);
        let child_offset = self.read_child_ptr(offset, child_idx);

        // Prefetch child node header and first entries while returning
        if prefetch_child {
            unsafe {
                let child_ptr = self.arena.node_ptr(child_offset);
                std::arch::x86_64::_mm_prefetch(
                    child_ptr as *const i8,
                    std::arch::x86_64::_MM_HINT_T0,
                );
                // Prefetch first cache line of entries (64 bytes after header)
                std::arch::x86_64::_mm_prefetch(
                    child_ptr.add(64) as *const i8,
                    std::arch::x86_64::_MM_HINT_T0,
                );
            }
        }

        child_offset
    }

    /// Branchless binary search for child index in internal node.
    /// Uses bitmask operations instead of branches for better pipeline efficiency.
    /// B+Tree semantics: child[i] has keys < key[i], child[i+1] has keys >= key[i].
    #[inline(always)]
    fn find_child_idx_branchless(&self, node_offset: u64, search_key: u64, num_keys: u16) -> usize {
        if num_keys == 0 {
            return 0;
        }

        unsafe {
            let base = self
                .arena
                .node_ptr(node_offset)
                .add(ARENA_INTERNAL_HEADER_SIZE + ARENA_CHILD_SIZE);
            let key_stride = ARENA_KEY_SIZE + ARENA_CHILD_SIZE;

            let mut lo = 0usize;
            let mut hi = num_keys as usize;

            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                let entry_ptr = base.add(mid * key_stride) as *const u64;
                let entry_key = std::ptr::read_unaligned(entry_ptr);

                // Branchless update using bitmask:
                // go_right is 0xFFFF... if search_key >= entry_key, 0 otherwise
                let go_right = (search_key >= entry_key) as usize;
                // If go_right: lo = mid + 1, hi unchanged
                // If !go_right: hi = mid, lo unchanged
                lo = go_right * (mid + 1) + (1 - go_right) * lo;
                hi = (1 - go_right) * mid + go_right * hi;
            }

            lo
        }
    }

    /// Read the separator key at position key_idx from an internal node.
    /// key[key_idx] is the separator to the right of child[key_idx].
    #[inline(always)]
    fn read_separator_key(&self, node_offset: u64, key_idx: usize) -> u64 {
        unsafe {
            let key_stride = ARENA_KEY_SIZE + ARENA_CHILD_SIZE;
            let ptr = self
                .arena
                .node_ptr(node_offset)
                .add(ARENA_INTERNAL_HEADER_SIZE + ARENA_CHILD_SIZE + key_idx * key_stride)
                as *const u64;
            std::ptr::read_unaligned(ptr)
        }
    }

    /// Read child pointer at given index from internal node.
    #[inline(always)]
    fn read_child_ptr(&self, node_offset: u64, idx: usize) -> u64 {
        unsafe {
            let key_stride = ARENA_KEY_SIZE + ARENA_CHILD_SIZE;
            let ptr = if idx == 0 {
                // Leftmost child is right after header
                self.arena
                    .node_ptr(node_offset)
                    .add(ARENA_INTERNAL_HEADER_SIZE) as *const u64
            } else {
                // Child i is after key (i-1)
                self.arena.node_ptr(node_offset).add(
                    ARENA_INTERNAL_HEADER_SIZE
                        + ARENA_CHILD_SIZE
                        + (idx - 1) * key_stride
                        + ARENA_KEY_SIZE,
                ) as *const u64
            };
            std::ptr::read_unaligned(ptr)
        }
    }

    /// Search leaf using interpolation search with binary search fallback.
    /// Interpolation search: O(log log n) for uniform keys, falls back to binary for safety.
    #[inline(always)]
    fn search_leaf_interpolation(&self, offset: u64, search_key: u64) -> Option<TupleId> {
        let header = self.arena.read_leaf_header(offset);
        let count = header.num_entries as usize;

        if count == 0 {
            return None;
        }

        unsafe {
            let base = self.arena.node_ptr(offset).add(ARENA_LEAF_HEADER_SIZE);

            // Read first and last keys for interpolation
            let first_key = std::ptr::read_unaligned(base as *const u64);
            let last_key = std::ptr::read_unaligned(
                base.add((count - 1) * ARENA_LEAF_ENTRY_SIZE) as *const u64
            );

            // Early exit: key out of range
            if search_key < first_key || search_key > last_key {
                return None;
            }

            // Exact match on boundaries
            if search_key == first_key {
                let tuple_ptr = (base as *const u64).add(1);
                return Some(Self::unpack_tuple_id(std::ptr::read_unaligned(tuple_ptr)));
            }
            if search_key == last_key {
                let entry_ptr = base.add((count - 1) * ARENA_LEAF_ENTRY_SIZE);
                let tuple_ptr = (entry_ptr as *const u64).add(1);
                return Some(Self::unpack_tuple_id(std::ptr::read_unaligned(tuple_ptr)));
            }

            let mut lo = 0usize;
            let mut hi = count;

            // Interpolation search: estimate position based on key distribution
            // Use up to 3 interpolation steps, then fall back to binary
            for _ in 0..3 {
                if hi - lo <= 8 {
                    break;
                }

                let lo_key =
                    std::ptr::read_unaligned(base.add(lo * ARENA_LEAF_ENTRY_SIZE) as *const u64);
                let hi_key = std::ptr::read_unaligned(
                    base.add((hi - 1) * ARENA_LEAF_ENTRY_SIZE) as *const u64
                );

                if lo_key >= hi_key {
                    break;
                }

                // Interpolation formula: estimate position proportionally
                let range = hi - lo;
                let key_range = hi_key - lo_key;
                let key_offset = search_key.saturating_sub(lo_key);

                // Compute interpolated position with overflow protection
                let estimate = if key_range > 0 {
                    lo + ((key_offset as u128 * range as u128 / key_range as u128) as usize)
                        .min(range - 1)
                } else {
                    lo + range / 2
                };

                let entry_ptr = base.add(estimate * ARENA_LEAF_ENTRY_SIZE) as *const u64;
                let entry_key = std::ptr::read_unaligned(entry_ptr);

                if entry_key == search_key {
                    let tuple_ptr = entry_ptr.add(1);
                    return Some(Self::unpack_tuple_id(std::ptr::read_unaligned(tuple_ptr)));
                } else if search_key < entry_key {
                    hi = estimate;
                } else {
                    lo = estimate + 1;
                }
            }

            // Binary search for remaining range
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                let entry_ptr = base.add(mid * ARENA_LEAF_ENTRY_SIZE) as *const u64;
                let entry_key = std::ptr::read_unaligned(entry_ptr);

                let go_right = (search_key > entry_key) as usize;
                lo = go_right * (mid + 1) + (1 - go_right) * lo;
                hi = (1 - go_right) * mid + go_right * hi;
            }

            // Verify exact match
            if lo < count {
                let entry_ptr = base.add(lo * ARENA_LEAF_ENTRY_SIZE) as *const u64;
                let entry_key = std::ptr::read_unaligned(entry_ptr);
                if entry_key == search_key {
                    let tuple_ptr = entry_ptr.add(1);
                    return Some(Self::unpack_tuple_id(std::ptr::read_unaligned(tuple_ptr)));
                }
            }

            None
        }
    }

    // =========================================================================
    // Write Path (Single Writer)
    // =========================================================================

    /// Insert a key-value pair.
    #[inline]
    pub fn insert(&mut self, key: &[u8], tuple_id: TupleId) -> Result<()> {
        let search_key = self.key_to_u64(key);
        let packed_tuple_id = Self::pack_tuple_id(tuple_id);

        // Find leaf for insert (Relaxed is safe since we have &mut self)
        let height = self.height.load(Ordering::Relaxed);
        let mut current_offset = self.root_offset.load(Ordering::Relaxed);

        // Build path for potential split propagation
        let mut path = [0u64; 16];
        let mut path_len = 0;

        path[path_len] = current_offset;
        path_len += 1;

        // Traverse to leaf
        for _ in 0..(height - 1) {
            let header = self.arena.read_internal_header(current_offset);
            let child_idx =
                self.find_child_idx_branchless(current_offset, search_key, header.num_keys);
            let child_offset = self.read_child_ptr(current_offset, child_idx);
            current_offset = child_offset;
            path[path_len] = current_offset;
            path_len += 1;
        }

        // Try insert in leaf
        let leaf_offset = current_offset;
        if self.try_insert_in_leaf(leaf_offset, search_key, packed_tuple_id)? {
            return Ok(());
        }

        // Leaf full - need to split
        self.split_and_insert(search_key, packed_tuple_id, &path[..path_len])
    }

    /// Try to insert in leaf without split.
    #[inline(always)]
    fn try_insert_in_leaf(&mut self, offset: u64, key: u64, packed_tuple_id: u64) -> Result<bool> {
        unsafe {
            let header_ptr = self.arena.node_ptr_mut(offset) as *mut ArenaLeafNodeHeader;
            let header = std::ptr::read_volatile(header_ptr);

            if header.num_entries as usize >= ARENA_MAX_LEAF_ENTRIES {
                return Ok(false); // Need split
            }

            // Bump version (odd = write in progress)
            let new_version = header.version + 1;
            std::ptr::write_volatile(&mut (*header_ptr).version, new_version);
            std::sync::atomic::fence(Ordering::Release);

            // Find insertion position (maintain sorted order)
            let insert_pos = self.find_leaf_insert_pos(offset, key, header.num_entries);

            // Shift entries to make room
            let base = self.arena.node_ptr_mut(offset).add(ARENA_LEAF_HEADER_SIZE);
            if insert_pos < header.num_entries as usize {
                let src = base.add(insert_pos * ARENA_LEAF_ENTRY_SIZE);
                let dst = base.add((insert_pos + 1) * ARENA_LEAF_ENTRY_SIZE);
                let count = (header.num_entries as usize - insert_pos) * ARENA_LEAF_ENTRY_SIZE;
                std::ptr::copy(src, dst, count);
            }

            // Write new entry
            let entry_ptr = base.add(insert_pos * ARENA_LEAF_ENTRY_SIZE) as *mut u64;
            std::ptr::write_unaligned(entry_ptr, key);
            std::ptr::write_unaligned(entry_ptr.add(1), packed_tuple_id);

            // Update count and finalize version (even = stable)
            (*header_ptr).num_entries = header.num_entries + 1;
            std::sync::atomic::fence(Ordering::Release);
            std::ptr::write_volatile(&mut (*header_ptr).version, new_version + 1);

            Ok(true)
        }
    }

    /// Find insertion position in leaf (binary search).
    #[inline]
    fn find_leaf_insert_pos(&self, offset: u64, key: u64, num_entries: u16) -> usize {
        if num_entries == 0 {
            return 0;
        }

        unsafe {
            let base = self.arena.node_ptr(offset).add(ARENA_LEAF_HEADER_SIZE);
            let mut lo = 0usize;
            let mut hi = num_entries as usize;

            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                let entry_ptr = base.add(mid * ARENA_LEAF_ENTRY_SIZE) as *const u64;
                let entry_key = std::ptr::read_unaligned(entry_ptr);

                if key > entry_key {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }

            lo
        }
    }

    /// Split leaf and insert (handles tree growth).
    fn split_and_insert(&mut self, key: u64, packed_tuple_id: u64, path: &[u64]) -> Result<()> {
        let leaf_offset = path[path.len() - 1];

        // Read current leaf
        let header = self.arena.read_leaf_header(leaf_offset);
        let num_entries = header.num_entries as usize;

        // Allocate new leaf
        let new_leaf_offset = self.arena.allocate();

        // Split point: half of entries go to new leaf
        let split_point = num_entries / 2;

        // Bump version on old leaf
        unsafe {
            let header_ptr = self.arena.node_ptr_mut(leaf_offset) as *mut ArenaLeafNodeHeader;
            let new_version = header.version + 1;
            std::ptr::write_volatile(&mut (*header_ptr).version, new_version);
            std::sync::atomic::fence(Ordering::Release);

            // Copy second half to new leaf
            let old_base = self.arena.node_ptr(leaf_offset).add(ARENA_LEAF_HEADER_SIZE);
            let new_base = self
                .arena
                .node_ptr_mut(new_leaf_offset)
                .add(ARENA_LEAF_HEADER_SIZE);

            let copy_count = num_entries - split_point;
            std::ptr::copy_nonoverlapping(
                old_base.add(split_point * ARENA_LEAF_ENTRY_SIZE),
                new_base,
                copy_count * ARENA_LEAF_ENTRY_SIZE,
            );

            // Get split key (first key in new leaf)
            let split_key = std::ptr::read_unaligned(new_base as *const u64);

            // Initialize new leaf header
            let new_header_ptr =
                self.arena.node_ptr_mut(new_leaf_offset) as *mut ArenaLeafNodeHeader;
            std::ptr::write(
                new_header_ptr,
                ArenaLeafNodeHeader {
                    version: 0,
                    num_entries: copy_count as u16,
                    reserved: [0; 6],
                    next_leaf: header.next_leaf,
                },
            );

            // Update old leaf: truncate and link to new leaf
            (*header_ptr).num_entries = split_point as u16;
            (*header_ptr).next_leaf = new_leaf_offset;
            std::sync::atomic::fence(Ordering::Release);
            std::ptr::write_volatile(&mut (*header_ptr).version, new_version + 1);

            // Insert the new key into appropriate leaf
            if key < split_key {
                self.try_insert_in_leaf(leaf_offset, key, packed_tuple_id)?;
            } else {
                self.try_insert_in_leaf(new_leaf_offset, key, packed_tuple_id)?;
            }

            // Propagate split up
            if path.len() == 1 {
                // Root was a leaf, create new root
                self.create_new_root(split_key, leaf_offset, new_leaf_offset);
            } else {
                self.propagate_split(split_key, new_leaf_offset, &path[..path.len() - 1])?;
            }
        }

        Ok(())
    }

    /// Create new root after root split.
    fn create_new_root(&mut self, split_key: u64, left_child: u64, right_child: u64) {
        let new_root_offset = self.arena.allocate();

        unsafe {
            let header_ptr =
                self.arena.node_ptr_mut(new_root_offset) as *mut ArenaInternalNodeHeader;
            std::ptr::write(
                header_ptr,
                ArenaInternalNodeHeader {
                    version: 0,
                    num_keys: 1,
                    level: self.height.load(Ordering::Relaxed) as u16,
                    reserved: 0,
                },
            );

            // Write leftmost child
            let child0_ptr = self
                .arena
                .node_ptr_mut(new_root_offset)
                .add(ARENA_INTERNAL_HEADER_SIZE) as *mut u64;
            std::ptr::write_unaligned(child0_ptr, left_child);

            // Write key + right child
            let key_ptr = child0_ptr.add(1);
            std::ptr::write_unaligned(key_ptr, split_key);
            let child1_ptr = key_ptr.add(1);
            std::ptr::write_unaligned(child1_ptr, right_child);
        }

        self.height.fetch_add(1, Ordering::Release);
        self.root_offset.store(new_root_offset, Ordering::Release);
    }

    /// Propagate split up to parent internal nodes.
    fn propagate_split(&mut self, key: u64, new_child: u64, path: &[u64]) -> Result<()> {
        let mut current_key = key;
        let mut current_child = new_child;

        for parent_idx in (0..path.len()).rev() {
            let parent_offset = path[parent_idx];

            if self.try_insert_in_internal(parent_offset, current_key, current_child)? {
                return Ok(());
            }

            // Need to split internal node
            let (split_key, new_internal) =
                self.split_internal_node(parent_offset, current_key, current_child)?;

            if parent_idx == 0 {
                // Root split
                self.create_new_root(split_key, parent_offset, new_internal);
                return Ok(());
            }

            current_key = split_key;
            current_child = new_internal;
        }

        Ok(())
    }

    /// Try to insert key/child in internal node without split.
    fn try_insert_in_internal(&mut self, offset: u64, key: u64, child: u64) -> Result<bool> {
        unsafe {
            let header_ptr = self.arena.node_ptr_mut(offset) as *mut ArenaInternalNodeHeader;
            let header = std::ptr::read_volatile(header_ptr);

            if header.num_keys as usize >= ARENA_MAX_INTERNAL_KEYS {
                return Ok(false);
            }

            // Bump version
            let new_version = header.version + 1;
            std::ptr::write_volatile(&mut (*header_ptr).version, new_version);
            std::sync::atomic::fence(Ordering::Release);

            // Find insertion position
            let insert_pos = self.find_internal_insert_pos(offset, key, header.num_keys);

            let key_stride = ARENA_KEY_SIZE + ARENA_CHILD_SIZE;
            let base = self
                .arena
                .node_ptr_mut(offset)
                .add(ARENA_INTERNAL_HEADER_SIZE + ARENA_CHILD_SIZE);

            // Shift entries
            if insert_pos < header.num_keys as usize {
                let src = base.add(insert_pos * key_stride);
                let dst = base.add((insert_pos + 1) * key_stride);
                let count = (header.num_keys as usize - insert_pos) * key_stride;
                std::ptr::copy(src, dst, count);
            }

            // Write new key and child
            let entry_ptr = base.add(insert_pos * key_stride) as *mut u64;
            std::ptr::write_unaligned(entry_ptr, key);
            std::ptr::write_unaligned(entry_ptr.add(1), child);

            // Update count and finalize
            (*header_ptr).num_keys = header.num_keys + 1;
            std::sync::atomic::fence(Ordering::Release);
            std::ptr::write_volatile(&mut (*header_ptr).version, new_version + 1);

            Ok(true)
        }
    }

    /// Find insertion position in internal node.
    #[inline]
    fn find_internal_insert_pos(&self, offset: u64, key: u64, num_keys: u16) -> usize {
        if num_keys == 0 {
            return 0;
        }

        unsafe {
            let base = self
                .arena
                .node_ptr(offset)
                .add(ARENA_INTERNAL_HEADER_SIZE + ARENA_CHILD_SIZE);
            let key_stride = ARENA_KEY_SIZE + ARENA_CHILD_SIZE;

            let mut lo = 0usize;
            let mut hi = num_keys as usize;

            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                let entry_ptr = base.add(mid * key_stride) as *const u64;
                let entry_key = std::ptr::read_unaligned(entry_ptr);

                if key > entry_key {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }

            lo
        }
    }

    /// Split internal node.
    fn split_internal_node(&mut self, offset: u64, key: u64, child: u64) -> Result<(u64, u64)> {
        let new_offset = self.arena.allocate();

        unsafe {
            let header_ptr = self.arena.node_ptr_mut(offset) as *mut ArenaInternalNodeHeader;
            let header = std::ptr::read_volatile(header_ptr);

            let num_keys = header.num_keys as usize;
            let split_point = num_keys / 2;

            // Bump version
            let new_version = header.version + 1;
            std::ptr::write_volatile(&mut (*header_ptr).version, new_version);
            std::sync::atomic::fence(Ordering::Release);

            let key_stride = ARENA_KEY_SIZE + ARENA_CHILD_SIZE;
            let base = self.arena.node_ptr(offset).add(ARENA_INTERNAL_HEADER_SIZE);

            // Get split key (middle key, will be promoted)
            let split_key_ptr = base.add(ARENA_CHILD_SIZE + split_point * key_stride) as *const u64;
            let split_key = std::ptr::read_unaligned(split_key_ptr);

            // Copy right half to new node (keys after split point)
            let new_base = self
                .arena
                .node_ptr_mut(new_offset)
                .add(ARENA_INTERNAL_HEADER_SIZE);

            // First, copy the leftmost child of right subtree
            let right_child_ptr = split_key_ptr.add(1) as *const u64;
            std::ptr::write_unaligned(
                new_base as *mut u64,
                std::ptr::read_unaligned(right_child_ptr),
            );

            // Copy remaining keys and children
            let copy_start = (split_point + 1) * key_stride + ARENA_CHILD_SIZE;
            let copy_count = (num_keys - split_point - 1) * key_stride;
            if copy_count > 0 {
                std::ptr::copy_nonoverlapping(
                    base.add(copy_start),
                    new_base.add(ARENA_CHILD_SIZE),
                    copy_count,
                );
            }

            // Initialize new internal header
            let new_header_ptr =
                self.arena.node_ptr_mut(new_offset) as *mut ArenaInternalNodeHeader;
            std::ptr::write(
                new_header_ptr,
                ArenaInternalNodeHeader {
                    version: 0,
                    num_keys: (num_keys - split_point - 1) as u16,
                    level: header.level,
                    reserved: 0,
                },
            );

            // Update old node
            (*header_ptr).num_keys = split_point as u16;
            std::sync::atomic::fence(Ordering::Release);
            std::ptr::write_volatile(&mut (*header_ptr).version, new_version + 1);

            // Insert the new key/child into appropriate node
            if key < split_key {
                self.try_insert_in_internal(offset, key, child)?;
            } else {
                self.try_insert_in_internal(new_offset, key, child)?;
            }

            Ok((split_key, new_offset))
        }
    }

    // =========================================================================
    // Range Scan
    // =========================================================================

    /// Range scan from start_key to end_key (inclusive).
    /// Returns packed (key, packed_tuple_id) pairs to avoid unpacking overhead.
    /// Callers use `unpack_tuple_id()` to decode when needed.
    ///
    /// Arena leaf entries are stored as contiguous [key:u64][packed_tid:u64] pairs,
    /// matching the (u64, u64) output layout. This enables bulk memcpy of entire
    /// entry ranges instead of per-entry reads.
    pub fn range_scan(&self, start_key: Option<&[u8]>, end_key: Option<&[u8]>) -> Vec<(u64, u64)> {
        let start = start_key.map(|k| self.key_to_u64(k)).unwrap_or(0);
        let end = end_key.map(|k| self.key_to_u64(k)).unwrap_or(u64::MAX);

        // Pre-allocate to avoid repeated reallocation under concurrent allocator contention.
        let capacity = (end.saturating_sub(start).saturating_add(1) as usize).min(8192);
        let mut results: Vec<(u64, u64)> = Vec::with_capacity(capacity);

        // Find starting leaf
        let height = self.height.load(Ordering::Relaxed);
        let mut current = self.root_offset.load(Ordering::Relaxed);

        for _ in 0..(height - 1) {
            let header = self.arena.read_internal_header(current);
            let child_idx = self.find_child_idx_branchless(current, start, header.num_keys);
            current = self.read_child_ptr(current, child_idx);
        }

        // First leaf: binary search for scan_start position.
        unsafe {
            let header = self.arena.read_leaf_header(current);
            let num_entries = header.num_entries as usize;
            let base = self.arena.node_ptr(current).add(ARENA_LEAF_HEADER_SIZE);

            let scan_start = if start > 0 {
                let mut lo = 0usize;
                let mut hi = num_entries;
                while lo < hi {
                    let mid = lo + (hi - lo) / 2;
                    let mid_key = std::ptr::read_unaligned(
                        base.add(mid * ARENA_LEAF_ENTRY_SIZE) as *const u64
                    );
                    if mid_key < start {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                lo
            } else {
                0
            };

            // Binary search for scan_end position.
            let scan_end = {
                let mut lo = scan_start;
                let mut hi = num_entries;
                while lo < hi {
                    let mid = lo + (hi - lo) / 2;
                    let mid_key = std::ptr::read_unaligned(
                        base.add(mid * ARENA_LEAF_ENTRY_SIZE) as *const u64
                    );
                    if mid_key <= end {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                lo
            };

            // Bulk copy: arena entry layout [key:u64][tid:u64] matches (u64, u64) tuple layout.
            let count = scan_end - scan_start;
            if count > 0 {
                let src = base.add(scan_start * ARENA_LEAF_ENTRY_SIZE);
                let old_len = results.len();
                let new_len = old_len + count;
                if new_len > results.capacity() {
                    results.reserve(count);
                }
                std::ptr::copy_nonoverlapping(
                    src,
                    (results.as_mut_ptr() as *mut u8).add(old_len * 16),
                    count * ARENA_LEAF_ENTRY_SIZE,
                );
                results.set_len(new_len);
            }

            // Single-leaf fast path: end key found in this leaf or no next leaf.
            if scan_end < num_entries || header.next_leaf == ARENA_NULL_OFFSET {
                return results;
            }
            current = header.next_leaf;
        }

        // Subsequent leaves: scan_start is always 0 (all entries >= start).
        loop {
            unsafe {
                let header = self.arena.read_leaf_header(current);
                let num_entries = header.num_entries as usize;
                let base = self.arena.node_ptr(current).add(ARENA_LEAF_HEADER_SIZE);

                // Check if end key falls within this leaf by reading the last entry.
                // If last_key <= end, the entire leaf is in range (skip binary search).
                let last_key = if num_entries > 0 {
                    std::ptr::read_unaligned(
                        base.add((num_entries - 1) * ARENA_LEAF_ENTRY_SIZE) as *const u64
                    )
                } else {
                    0
                };

                let scan_end = if last_key <= end {
                    num_entries
                } else {
                    // Binary search only on the last leaf that intersects the end boundary.
                    let mut lo = 0usize;
                    let mut hi = num_entries;
                    while lo < hi {
                        let mid = lo + (hi - lo) / 2;
                        let mid_key = std::ptr::read_unaligned(
                            base.add(mid * ARENA_LEAF_ENTRY_SIZE) as *const u64,
                        );
                        if mid_key <= end {
                            lo = mid + 1;
                        } else {
                            hi = mid;
                        }
                    }
                    lo
                };

                // Bulk copy entire range from this leaf.
                if scan_end > 0 {
                    let old_len = results.len();
                    let new_len = old_len + scan_end;
                    if new_len > results.capacity() {
                        results.reserve(scan_end);
                    }
                    std::ptr::copy_nonoverlapping(
                        base,
                        (results.as_mut_ptr() as *mut u8).add(old_len * 16),
                        scan_end * ARENA_LEAF_ENTRY_SIZE,
                    );
                    results.set_len(new_len);
                }

                if scan_end < num_entries || header.next_leaf == ARENA_NULL_OFFSET {
                    break;
                }
                current = header.next_leaf;
            }
        }

        results
    }

    // =========================================================================
    // Utility Functions
    // =========================================================================

    /// Convert key bytes to u64 (big-endian for sort order).
    #[inline(always)]
    fn key_to_u64(&self, key: &[u8]) -> u64 {
        if key.len() >= 8 {
            u64::from_be_bytes([
                key[0], key[1], key[2], key[3], key[4], key[5], key[6], key[7],
            ])
        } else {
            let mut padded = [0u8; 8];
            padded[..key.len()].copy_from_slice(key);
            u64::from_be_bytes(padded)
        }
    }

    /// Pack TupleId into u64: file_id(16) + page_num(32) + slot_id(16).
    #[inline(always)]
    pub(crate) fn pack_tuple_id(tid: TupleId) -> u64 {
        ((tid.page_id.file_id as u64 & 0xFFFF) << 48)
            | ((tid.page_id.page_num as u64) << 16)
            | (tid.slot_id as u64)
    }

    /// Unpack u64 into TupleId.
    #[inline(always)]
    pub(crate) fn unpack_tuple_id(packed: u64) -> TupleId {
        TupleId {
            page_id: PageId {
                file_id: ((packed >> 48) & 0xFFFF) as u32,
                page_num: ((packed >> 16) & 0xFFFFFFFF) as u64,
            },
            slot_id: (packed & 0xFFFF) as u16,
        }
    }

    /// Returns the number of entries in the tree (for testing).
    pub fn count(&self) -> usize {
        let mut count = 0;
        let height = self.height.load(Ordering::Relaxed);
        let mut current = self.root_offset.load(Ordering::Relaxed);

        // Find leftmost leaf
        for _ in 0..(height - 1) {
            current = self.read_child_ptr(current, 0);
        }

        // Count all leaves
        loop {
            let header = self.arena.read_leaf_header(current);
            count += header.num_entries as usize;

            if header.next_leaf == ARENA_NULL_OFFSET {
                break;
            }
            current = header.next_leaf;
        }

        count
    }

    // =========================================================================
    // Bulk Insert (Cursor-Based)
    // =========================================================================

    /// Bulk insert sorted u64 key-value pairs directly.
    /// Keys must be sorted ascending within the batch. Values are pre-packed tuple_ids.
    /// Optimizations for sorted batches:
    /// 1. Skip tree traversal when consecutive keys go to same leaf
    /// 2. Skip binary search when inserting at predictable position
    /// 3. Batch version bumps per leaf (one at start, one at end)
    #[inline]
    pub fn insert_bulk_sorted(&mut self, entries: &[(u64, u64)]) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }

        // Batch state for current leaf
        let mut batch_leaf: u64 = u64::MAX;
        let mut batch_version: u64 = 0;
        let mut batch_active = false;
        let mut batch_max_key: u64 = 0; // Upper bound key for the current batch leaf (separator from parent)
        let mut last_insert_pos: usize = 0;
        let mut path = [0u64; 16];
        let mut path_len: usize;

        // Cached raw pointer to the current batch leaf node. Avoids repeated node_ptr_mut
        // calls in the fast append path. Valid only when batch_active == true.
        let mut batch_leaf_ptr: *mut u8 = std::ptr::null_mut();
        // Cached num_entries for the current batch leaf. Avoids read_leaf_header per append.
        let mut cached_num_entries: u16 = 0;

        for &(key, packed_tuple_id) in entries {
            // Try to stay in current leaf batch.
            // cached_num_entries and batch_leaf_ptr avoid header reads and pointer recomputation.
            if batch_active {
                let key_belongs = key <= batch_max_key || batch_max_key == u64::MAX;
                if key_belongs && (cached_num_entries as usize) < ARENA_MAX_LEAF_ENTRIES {
                    // Determine insertion position using sorted property
                    let insert_pos = if last_insert_pos + 1 >= cached_num_entries as usize {
                        // Append at end (common case for ascending inserts)
                        cached_num_entries as usize
                    } else {
                        // Check if key fits at last_insert_pos + 1
                        let next_key = unsafe {
                            let base = (batch_leaf_ptr as *const u8).add(ARENA_LEAF_HEADER_SIZE);
                            let ptr = base.add((last_insert_pos + 1) * ARENA_LEAF_ENTRY_SIZE)
                                as *const u64;
                            std::ptr::read_unaligned(ptr)
                        };
                        if key < next_key {
                            last_insert_pos + 1
                        } else {
                            // Binary search from last position
                            self.find_leaf_insert_pos_from(
                                batch_leaf,
                                key,
                                last_insert_pos + 1,
                                cached_num_entries,
                            )
                        }
                    };

                    // Insert with shift. Uses cached batch_leaf_ptr: no repeated node_ptr_mut.
                    unsafe {
                        let base = batch_leaf_ptr.add(ARENA_LEAF_HEADER_SIZE);
                        if insert_pos < cached_num_entries as usize {
                            let src = base.add(insert_pos * ARENA_LEAF_ENTRY_SIZE);
                            let dst = base.add((insert_pos + 1) * ARENA_LEAF_ENTRY_SIZE);
                            let count =
                                (cached_num_entries as usize - insert_pos) * ARENA_LEAF_ENTRY_SIZE;
                            std::ptr::copy(src, dst, count);
                        }
                        let entry_ptr = base.add(insert_pos * ARENA_LEAF_ENTRY_SIZE) as *mut u64;
                        std::ptr::write_unaligned(entry_ptr, key);
                        std::ptr::write_unaligned(entry_ptr.add(1), packed_tuple_id);

                        let header_ptr = batch_leaf_ptr as *mut ArenaLeafNodeHeader;
                        (*header_ptr).num_entries = cached_num_entries + 1;
                    }

                    cached_num_entries += 1;
                    last_insert_pos = insert_pos;
                    continue;
                }
            }

            // End current batch and start new tree traversal.
            // No volatile/fence needed: &mut self guarantees exclusive access.
            if batch_active {
                unsafe {
                    let header_ptr = batch_leaf_ptr as *mut ArenaLeafNodeHeader;
                    (*header_ptr).version = batch_version + 2;
                }
            }

            // Tree traversal to find correct leaf, tracking separator key for batch bound
            let height = self.height.load(Ordering::Relaxed);
            let mut current_offset = self.root_offset.load(Ordering::Relaxed);
            path_len = 0;
            path[path_len] = current_offset;
            path_len += 1;

            // leaf_separator: upper bound of the target leaf (from direct parent separator key).
            // u64::MAX when the leaf is the rightmost child at that level.
            let mut leaf_separator = u64::MAX;

            for _ in 0..(height - 1) {
                let header = self.arena.read_internal_header(current_offset);
                let child_idx =
                    self.find_child_idx_branchless(current_offset, key, header.num_keys);
                leaf_separator = if (child_idx as u16) < header.num_keys {
                    self.read_separator_key(current_offset, child_idx)
                } else {
                    u64::MAX
                };
                let child_offset = self.read_child_ptr(current_offset, child_idx);
                current_offset = child_offset;
                path[path_len] = current_offset;
                path_len += 1;
            }

            let leaf_offset = current_offset;
            let header = self.arena.read_leaf_header(leaf_offset);

            if (header.num_entries as usize) < ARENA_MAX_LEAF_ENTRIES {
                // Start new batch on this leaf. Cache pointer and count for hot-path use.
                batch_leaf = leaf_offset;
                batch_version = header.version;
                batch_active = true;
                batch_leaf_ptr = self.arena.node_ptr_mut(leaf_offset);

                // Bump version (odd = write in progress).
                // No volatile/fence needed: &mut self guarantees exclusive access.
                unsafe {
                    let header_ptr = batch_leaf_ptr as *mut ArenaLeafNodeHeader;
                    (*header_ptr).version = batch_version + 1;
                }

                // Find insertion position
                let insert_pos = self.find_leaf_insert_pos(leaf_offset, key, header.num_entries);

                // Insert with shift
                unsafe {
                    let base = batch_leaf_ptr.add(ARENA_LEAF_HEADER_SIZE);
                    if insert_pos < header.num_entries as usize {
                        let src = base.add(insert_pos * ARENA_LEAF_ENTRY_SIZE);
                        let dst = base.add((insert_pos + 1) * ARENA_LEAF_ENTRY_SIZE);
                        let count =
                            (header.num_entries as usize - insert_pos) * ARENA_LEAF_ENTRY_SIZE;
                        std::ptr::copy(src, dst, count);
                    }
                    let entry_ptr = base.add(insert_pos * ARENA_LEAF_ENTRY_SIZE) as *mut u64;
                    std::ptr::write_unaligned(entry_ptr, key);
                    std::ptr::write_unaligned(entry_ptr.add(1), packed_tuple_id);

                    let header_ptr = batch_leaf_ptr as *mut ArenaLeafNodeHeader;
                    (*header_ptr).num_entries = header.num_entries + 1;
                }

                // Upper bound for this leaf comes from the separator in its parent.
                // Stays constant for the entire batch on this leaf.
                batch_max_key = leaf_separator;
                cached_num_entries = header.num_entries + 1;
                last_insert_pos = insert_pos;
            } else {
                // Leaf full - split (no batch active after this)
                self.split_and_insert(key, packed_tuple_id, &path[..path_len])?;

                // Update state for next iteration
                let height = self.height.load(Ordering::Relaxed);
                let (new_leaf, _, _) = self.find_leaf_with_path(key, height);
                let new_header = self.arena.read_leaf_header(new_leaf);
                batch_leaf = new_leaf;
                batch_version = new_header.version;
                batch_active = false; // Don't start batch after split (version already finalized)

                // Find where key ended up
                last_insert_pos = self
                    .find_key_position(new_leaf, key, new_header.num_entries)
                    .unwrap_or(0);
            }
        }

        // Finalize last batch. No volatile/fence: &mut self guarantees exclusive access.
        if batch_active {
            unsafe {
                let header_ptr = batch_leaf_ptr as *mut ArenaLeafNodeHeader;
                (*header_ptr).version = batch_version + 2;
            }
        }

        Ok(())
    }

    /// Binary search starting from a given position (for sorted batch optimization).
    #[inline(always)]
    fn find_leaf_insert_pos_from(
        &self,
        offset: u64,
        key: u64,
        start: usize,
        num_entries: u16,
    ) -> usize {
        unsafe {
            let base = self.arena.node_ptr(offset).add(ARENA_LEAF_HEADER_SIZE);
            let mut lo = start;
            let mut hi = num_entries as usize;

            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                let entry_ptr = base.add(mid * ARENA_LEAF_ENTRY_SIZE) as *const u64;
                let entry_key = std::ptr::read_unaligned(entry_ptr);

                if key > entry_key {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            lo
        }
    }

    /// Find exact position of a key in leaf (returns None if not found).
    #[inline(always)]
    fn find_key_position(&self, offset: u64, key: u64, num_entries: u16) -> Option<usize> {
        let pos = self.find_leaf_insert_pos(offset, key, num_entries);
        if pos < num_entries as usize {
            let entry_key = unsafe {
                let base = self.arena.node_ptr(offset).add(ARENA_LEAF_HEADER_SIZE);
                let ptr = base.add(pos * ARENA_LEAF_ENTRY_SIZE) as *const u64;
                std::ptr::read_unaligned(ptr)
            };
            if entry_key == key {
                return Some(pos);
            }
        }
        None
    }

    /// Find leaf for key and return (leaf_offset, path, path_len).
    #[inline]
    fn find_leaf_with_path(&self, search_key: u64, height: u64) -> (u64, [u64; 16], usize) {
        let mut current_offset = self.root_offset.load(Ordering::Relaxed);
        let mut path = [0u64; 16];
        let mut path_len = 0;

        path[path_len] = current_offset;
        path_len += 1;

        for _ in 0..(height - 1) {
            let header = self.arena.read_internal_header(current_offset);
            let child_idx =
                self.find_child_idx_branchless(current_offset, search_key, header.num_keys);
            let child_offset = self.read_child_ptr(current_offset, child_idx);
            current_offset = child_offset;
            path[path_len] = current_offset;
            path_len += 1;
        }

        (current_offset, path, path_len)
    }
}

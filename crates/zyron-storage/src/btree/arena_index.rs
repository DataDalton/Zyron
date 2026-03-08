//! Arena-based B+Tree index with bulk insert optimization.

use super::arena::{ArenaInternalNodeHeader, ArenaLeafNodeHeader, BTreeArena};
use super::constants::{
    ARENA_CHILD_SIZE, ARENA_INTERNAL_HEADER_SIZE, ARENA_KEY_SIZE, ARENA_LEAF_HEADER_SIZE,
    ARENA_MAX_INTERNAL_KEYS, ARENA_MAX_LEAF_ENTRIES, ARENA_NULL_OFFSET,
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
    /// When at last internal level, prefetches the target leaf node header
    /// and first cache line of entries.
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

        if prefetch_child {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                let child_ptr = self.arena.node_ptr(child_offset);
                std::arch::x86_64::_mm_prefetch(
                    child_ptr as *const i8,
                    std::arch::x86_64::_MM_HINT_T0,
                );
                std::arch::x86_64::_mm_prefetch(
                    child_ptr.add(64) as *const i8,
                    std::arch::x86_64::_MM_HINT_T0,
                );
            }
        }

        child_offset
    }

    /// Branchless binary search for child index in internal node.
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

                let go_right = (search_key >= entry_key) as usize;
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

    /// Interpolation search with binary search fallback on sorted interleaved leaf.
    /// Each entry is 16 bytes: [key: u64, value: u64].
    /// Interpolation converges in O(log log n) steps for uniform keys (~3 steps
    /// for 2046 entries vs 11 for binary search), then binary search for remainder.
    #[inline(always)]
    fn search_leaf_interpolation(&self, offset: u64, search_key: u64) -> Option<TupleId> {
        let header = self.arena.read_leaf_header(offset);
        let count = header.num_entries as usize;

        if count == 0 {
            return None;
        }

        unsafe {
            let base = self.arena.node_ptr(offset).add(ARENA_LEAF_HEADER_SIZE);
            let entries = base as *const u64;

            let first_key = std::ptr::read_unaligned(entries);
            let last_key = std::ptr::read_unaligned(entries.add((count - 1) * 2));

            if search_key < first_key || search_key > last_key {
                return None;
            }
            if search_key == first_key {
                return Some(Self::unpack_tuple_id(std::ptr::read_unaligned(
                    entries.add(1),
                )));
            }
            if search_key == last_key {
                return Some(Self::unpack_tuple_id(std::ptr::read_unaligned(
                    entries.add((count - 1) * 2 + 1),
                )));
            }

            let mut lo = 0usize;
            let mut hi = count;

            // Interpolation search: estimate position from key distribution.
            // Up to 3 steps, then fall back to binary search.
            for _ in 0..3 {
                if hi - lo <= 8 {
                    break;
                }

                let lo_key = std::ptr::read_unaligned(entries.add(lo * 2));
                let hi_key = std::ptr::read_unaligned(entries.add((hi - 1) * 2));

                if lo_key >= hi_key {
                    break;
                }

                let range = hi - lo;
                let key_range = hi_key - lo_key;
                let key_offset = search_key.saturating_sub(lo_key);

                let estimate = if key_range > 0 {
                    lo + ((key_offset as u128 * range as u128 / key_range as u128) as usize)
                        .min(range - 1)
                } else {
                    lo + range / 2
                };

                let entry_key = std::ptr::read_unaligned(entries.add(estimate * 2));

                if entry_key == search_key {
                    return Some(Self::unpack_tuple_id(std::ptr::read_unaligned(
                        entries.add(estimate * 2 + 1),
                    )));
                } else if search_key < entry_key {
                    hi = estimate;
                } else {
                    lo = estimate + 1;
                }
            }

            // Binary search fallback for remaining range
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                let entry_key = std::ptr::read_unaligned(entries.add(mid * 2));

                let go_right = (search_key > entry_key) as usize;
                lo = go_right * (mid + 1) + (1 - go_right) * lo;
                hi = (1 - go_right) * mid + go_right * hi;
            }

            if lo < count {
                let found_key = std::ptr::read_unaligned(entries.add(lo * 2));
                if found_key == search_key {
                    return Some(Self::unpack_tuple_id(std::ptr::read_unaligned(
                        entries.add(lo * 2 + 1),
                    )));
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
    /// Writes in sorted interleaved order.
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

            let n = header.num_entries as usize;
            let base = self.arena.node_ptr_mut(offset).add(ARENA_LEAF_HEADER_SIZE);
            let entries = base as *mut u64;
            // Interleaved layout: entry i at entries[2*i] (key), entries[2*i+1] (value)

            // Find insertion position (binary search over sorted keys)
            let insert_pos = self.find_leaf_insert_pos(offset, key, header.num_entries);

            // Shift entries (key+value pairs) to make room
            if insert_pos < n {
                let shift_count = n - insert_pos;
                std::ptr::copy(
                    entries.add(insert_pos * 2),
                    entries.add((insert_pos + 1) * 2),
                    shift_count * 2,
                );
            }

            // Write new key and value
            std::ptr::write_unaligned(entries.add(insert_pos * 2), key);
            std::ptr::write_unaligned(entries.add(insert_pos * 2 + 1), packed_tuple_id);

            (*header_ptr).num_entries = header.num_entries + 1;
            std::sync::atomic::fence(Ordering::Release);
            std::ptr::write_volatile(&mut (*header_ptr).version, new_version + 1);

            Ok(true)
        }
    }

    /// Find insertion position in leaf (binary search over keys array).
    #[inline]
    fn find_leaf_insert_pos(&self, offset: u64, key: u64, num_entries: u16) -> usize {
        if num_entries == 0 {
            return 0;
        }

        unsafe {
            let entries = self.arena.node_ptr(offset).add(ARENA_LEAF_HEADER_SIZE) as *const u64;
            let mut lo = 0usize;
            let mut hi = num_entries as usize;

            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                let entry_key = std::ptr::read_unaligned(entries.add(mid * 2));

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
    /// Writes halves in sorted interleaved order.
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

            let old_entries =
                self.arena.node_ptr(leaf_offset).add(ARENA_LEAF_HEADER_SIZE) as *const u64;

            // Get split key (first key in right half, interleaved: key at 2*i)
            let split_key = std::ptr::read_unaligned(old_entries.add(split_point * 2));

            // Copy right half to new leaf (interleaved entries)
            let new_base = self
                .arena
                .node_ptr_mut(new_leaf_offset)
                .add(ARENA_LEAF_HEADER_SIZE);
            let new_entries = new_base as *mut u64;
            let copy_count = num_entries - split_point;
            std::ptr::copy_nonoverlapping(
                old_entries.add(split_point * 2),
                new_entries,
                copy_count * 2,
            );
            // Left half stays in place (already at old_entries[0..split_point*2])

            // Initialize new leaf header (layout = SORTED)
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

            // Update old leaf: truncate, link to new leaf
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
            let right_child_ptr = split_key_ptr.add(1);
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
    /// Sorted interleaved leaf layout with bulk memcpy.
    pub fn range_scan(&self, start_key: Option<&[u8]>, end_key: Option<&[u8]>) -> Vec<(u64, u64)> {
        let start = start_key.map(|k| self.key_to_u64(k)).unwrap_or(0);
        let end = end_key.map(|k| self.key_to_u64(k)).unwrap_or(u64::MAX);

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

        // First leaf: binary search for start/end positions, bulk memcpy
        unsafe {
            let header = self.arena.read_leaf_header(current);
            let num_entries = header.num_entries as usize;
            let base = self.arena.node_ptr(current).add(ARENA_LEAF_HEADER_SIZE);

            let scan_start = if start > 0 {
                let mut lo = 0usize;
                let mut hi = num_entries;
                while lo < hi {
                    let mid = lo + (hi - lo) / 2;
                    let mid_key = std::ptr::read_unaligned(base.add(mid * 16) as *const u64);
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

            let scan_end = {
                let mut lo = scan_start;
                let mut hi = num_entries;
                while lo < hi {
                    let mid = lo + (hi - lo) / 2;
                    let mid_key = std::ptr::read_unaligned(base.add(mid * 16) as *const u64);
                    if mid_key <= end {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                lo
            };

            // Bulk copy: interleaved [key:u64][val:u64] matches (u64, u64) layout.
            let count = scan_end - scan_start;
            if count > 0 {
                let src = base.add(scan_start * 16);
                let old_len = results.len();
                let new_len = old_len + count;
                if new_len > results.capacity() {
                    results.reserve(count);
                }
                std::ptr::copy_nonoverlapping(
                    src,
                    (results.as_mut_ptr() as *mut u8).add(old_len * 16),
                    count * 16,
                );
                results.set_len(new_len);
            }

            if scan_end < num_entries || header.next_leaf == ARENA_NULL_OFFSET {
                return results;
            }
            current = header.next_leaf;
        }

        // Subsequent leaves: bulk memcpy
        loop {
            unsafe {
                let header = self.arena.read_leaf_header(current);
                let num_entries = header.num_entries as usize;
                let base = self.arena.node_ptr(current).add(ARENA_LEAF_HEADER_SIZE);

                let last_key = if num_entries > 0 {
                    std::ptr::read_unaligned(base.add((num_entries - 1) * 16) as *const u64)
                } else {
                    0
                };

                let scan_end = if last_key <= end {
                    num_entries
                } else {
                    let mut lo = 0usize;
                    let mut hi = num_entries;
                    while lo < hi {
                        let mid = lo + (hi - lo) / 2;
                        let mid_key = std::ptr::read_unaligned(base.add(mid * 16) as *const u64);
                        if mid_key <= end {
                            lo = mid + 1;
                        } else {
                            hi = mid;
                        }
                    }
                    lo
                };

                if scan_end > 0 {
                    let old_len = results.len();
                    let new_len = old_len + scan_end;
                    if new_len > results.capacity() {
                        results.reserve(scan_end);
                    }
                    std::ptr::copy_nonoverlapping(
                        base,
                        (results.as_mut_ptr() as *mut u8).add(old_len * 16),
                        scan_end * 16,
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
    /// Uses a direct 8-byte read for the common case (single load + bswap).
    #[inline(always)]
    fn key_to_u64(&self, key: &[u8]) -> u64 {
        if key.len() >= 8 {
            u64::from_be_bytes(unsafe { *(key.as_ptr() as *const [u8; 8]) })
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
            | (tid.page_id.page_num << 16)
            | (tid.slot_id as u64)
    }

    /// Unpack u64 into TupleId.
    #[inline(always)]
    pub(crate) fn unpack_tuple_id(packed: u64) -> TupleId {
        TupleId {
            page_id: PageId {
                file_id: ((packed >> 48) & 0xFFFF) as u32,
                page_num: (packed >> 16) & 0xFFFFFFFF,
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
        let mut batch_version: u64 = 0;
        let mut batch_active = false;
        let mut batch_max_key: u64 = 0;
        let mut path = [0u64; 16];
        let mut path_len: usize;

        // Cached raw pointer to the current batch leaf node
        let mut batch_leaf_ptr: *mut u8 = std::ptr::null_mut();
        // Cached num_entries for the current batch leaf (existing entries count at batch start)
        let mut leaf_existing_count: u16 = 0;

        // Pending new entries for the current leaf, collected for merge-sort flush
        const MAX_PENDING: usize = 2048;
        let mut pending_keys = [0u64; MAX_PENDING];
        let mut pending_vals = [0u64; MAX_PENDING];
        let mut pending_count: usize = 0;

        for &(key, packed_tuple_id) in entries {
            // Try to stay in current leaf batch
            if batch_active {
                let key_belongs = key <= batch_max_key || batch_max_key == u64::MAX;
                let total = leaf_existing_count as usize + pending_count + 1;
                if key_belongs && total <= ARENA_MAX_LEAF_ENTRIES {
                    pending_keys[pending_count] = key;
                    pending_vals[pending_count] = packed_tuple_id;
                    pending_count += 1;
                    continue;
                }
            }

            // Flush pending entries to current leaf via merge, then start new leaf
            if batch_active && pending_count > 0 {
                unsafe {
                    self.merge_flush_leaf(
                        batch_leaf_ptr,
                        leaf_existing_count,
                        &pending_keys[..pending_count],
                        &pending_vals[..pending_count],
                    );
                }
            }

            // Finalize current batch version
            if batch_active {
                unsafe {
                    let header_ptr = batch_leaf_ptr as *mut ArenaLeafNodeHeader;
                    (*header_ptr).version = batch_version + 2;
                }
            }

            // Tree traversal to find correct leaf
            let height = self.height.load(Ordering::Relaxed);
            let mut current_offset = self.root_offset.load(Ordering::Relaxed);
            path_len = 0;
            path[path_len] = current_offset;
            path_len += 1;

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
                // Start new batch on this leaf
                batch_version = header.version;
                batch_active = true;
                batch_leaf_ptr = self.arena.node_ptr_mut(leaf_offset);
                leaf_existing_count = header.num_entries;

                // Bump version (odd = write in progress)
                unsafe {
                    let header_ptr = batch_leaf_ptr as *mut ArenaLeafNodeHeader;
                    (*header_ptr).version = batch_version + 1;
                }

                batch_max_key = leaf_separator;

                // Add first entry to pending buffer
                pending_keys[0] = key;
                pending_vals[0] = packed_tuple_id;
                pending_count = 1;
            } else {
                // Leaf full, split
                batch_active = false;
                self.split_and_insert(key, packed_tuple_id, &path[..path_len])?;

                // Re-find the leaf after split for potential next batch
                let height = self.height.load(Ordering::Relaxed);
                let (new_leaf, _, _) = self.find_leaf_with_path(key, height);
                let new_header = self.arena.read_leaf_header(new_leaf);
                batch_version = new_header.version;
                leaf_existing_count = new_header.num_entries;
                pending_count = 0;
            }
        }

        // Flush remaining pending entries
        if batch_active && pending_count > 0 {
            unsafe {
                self.merge_flush_leaf(
                    batch_leaf_ptr,
                    leaf_existing_count,
                    &pending_keys[..pending_count],
                    &pending_vals[..pending_count],
                );
            }
        }

        // Finalize last batch version
        if batch_active {
            unsafe {
                let header_ptr = batch_leaf_ptr as *mut ArenaLeafNodeHeader;
                (*header_ptr).version = batch_version + 2;
            }
        }

        Ok(())
    }

    /// Merge-flush pending sorted entries into a leaf node using interleaved layout.
    /// Each entry is [key: u64, val: u64] = 16 bytes, contiguous in sorted order.
    #[inline(always)]
    unsafe fn merge_flush_leaf(
        &self,
        leaf_ptr: *mut u8,
        existing_count: u16,
        new_keys: &[u64],
        new_vals: &[u64],
    ) {
        unsafe {
            let base = leaf_ptr.add(ARENA_LEAF_HEADER_SIZE);
            let existing = existing_count as usize;
            let new_count = new_keys.len();
            let total = existing + new_count;
            let header_ptr = leaf_ptr as *mut ArenaLeafNodeHeader;
            let leaf_entries = base as *mut u64;

            // Fast path: empty leaf, write interleaved entries directly
            if existing == 0 {
                for i in 0..new_count {
                    *leaf_entries.add(i * 2) = new_keys[i];
                    *leaf_entries.add(i * 2 + 1) = new_vals[i];
                }
                (*header_ptr).num_entries = new_count as u16;
                return;
            }

            // Extract existing entries from interleaved layout into temp buffers
            let mut sorted_keys = [0u64; ARENA_MAX_LEAF_ENTRIES];
            let mut sorted_vals = [0u64; ARENA_MAX_LEAF_ENTRIES];
            for i in 0..existing {
                sorted_keys[i] = *leaf_entries.add(i * 2);
                sorted_vals[i] = *leaf_entries.add(i * 2 + 1);
            }

            // Merge existing + new into interleaved output directly on the leaf
            let mut ei = 0usize;
            let mut ni = 0usize;
            let mut out = 0usize;

            while ei < existing && ni < new_count {
                if sorted_keys[ei] <= new_keys[ni] {
                    *leaf_entries.add(out * 2) = sorted_keys[ei];
                    *leaf_entries.add(out * 2 + 1) = sorted_vals[ei];
                    ei += 1;
                } else {
                    *leaf_entries.add(out * 2) = new_keys[ni];
                    *leaf_entries.add(out * 2 + 1) = new_vals[ni];
                    ni += 1;
                }
                out += 1;
            }

            while ei < existing {
                *leaf_entries.add(out * 2) = sorted_keys[ei];
                *leaf_entries.add(out * 2 + 1) = sorted_vals[ei];
                ei += 1;
                out += 1;
            }

            while ni < new_count {
                *leaf_entries.add(out * 2) = new_keys[ni];
                *leaf_entries.add(out * 2 + 1) = new_vals[ni];
                ni += 1;
                out += 1;
            }

            (*header_ptr).num_entries = total as u16;
        }
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

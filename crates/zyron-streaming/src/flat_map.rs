//! Flat hash tables for streaming join builds and keyed state.
//!
//! `FlatU64Map` is a SIMD-accelerated open-addressing table for pre-hashed
//! u64 keys, `FlatHashTable` a flat bucket array with prev-chains for join
//! builds. Both take keys already hashed by the canonical primitives in
//! zyron_common::checksum (hash_int, fnv1a_64, hash_combine) or by the
//! column batch hashing in crate::column.

// ---------------------------------------------------------------------------
// FlatU64Map<V>: SIMD-accelerated open-addressing table for u64 keys
// ---------------------------------------------------------------------------

/// Empty slot sentinel. Keys equal to u64::MAX cannot be stored in this map.
/// All hash functions feeding this map produce well-distributed values where
/// u64::MAX is statistically unreachable (probability ~5.4e-20 per hash).
const U64MAP_EMPTY: u64 = u64::MAX;

/// Maximum load factor: 75%.
const U64MAP_LOAD_NUMER: usize = 3;
const U64MAP_LOAD_DENOM: usize = 4;

// ---------------------------------------------------------------------------
// SIMD group comparison: platform-specific, best available instruction set.
//
// x86_64: AVX2 (4 keys per compare) as default. At runtime, if AVX-512 is
//         detected, the map upgrades to 8 keys per compare. The group_size
//         field on each map instance controls which path is used.
// aarch64: SVE (hardware-adaptive width, 2-32 keys depending on chip).
//          Falls back to NEON (2 keys) if SVE is not available.
// other:   Scalar (1 key per compare).
// ---------------------------------------------------------------------------

/// AVX2: compare 4 u64 keys in one instruction.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn group_match_4(keys_ptr: *const u64, target: u64) -> u32 {
    use std::arch::x86_64::*;
    // SAFETY: caller guarantees 4 readable u64s at keys_ptr and an AVX2 CPU
    unsafe {
        let group = _mm256_loadu_si256(keys_ptr as *const __m256i);
        let needle = _mm256_set1_epi64x(target as i64);
        let cmp = _mm256_cmpeq_epi64(group, needle);
        _mm256_movemask_pd(_mm256_castsi256_pd(cmp)) as u32
    }
}

/// AVX-512: compare 8 u64 keys in one instruction.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn group_match_8(keys_ptr: *const u64, target: u64) -> u32 {
    use std::arch::x86_64::*;
    // SAFETY: caller guarantees 8 readable u64s at keys_ptr
    let group = unsafe { _mm512_loadu_si512(keys_ptr as *const __m512i) };
    let needle = _mm512_set1_epi64(target as i64);
    _mm512_cmpeq_epi64_mask(group, needle) as u32
}

/// NEON: compare 2 u64 keys.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn group_match_2(keys_ptr: *const u64, target: u64) -> u32 {
    use std::arch::aarch64::*;
    // SAFETY: caller guarantees 2 readable u64s at keys_ptr
    unsafe {
        let group = vld1q_u64(keys_ptr);
        let needle = vdupq_n_u64(target);
        let cmp = vceqq_u64(group, needle);
        let b0 = if vgetq_lane_u64(cmp, 0) != 0 { 1u32 } else { 0 };
        let b1 = if vgetq_lane_u64(cmp, 1) != 0 { 2u32 } else { 0 };
        b0 | b1
    }
}

/// Function pointer type for SIMD group match. Set once at map creation.
type GroupMatchFn = unsafe fn(*const u64, u64) -> u32;

/// Returns the best group_match function for this CPU.
#[cfg(target_arch = "x86_64")]
fn select_group_match_fn() -> GroupMatchFn {
    if is_x86_feature_detected!("avx512f") {
        group_match_8
    } else {
        group_match_4
    }
}

#[cfg(target_arch = "aarch64")]
fn select_group_match_fn() -> GroupMatchFn {
    group_match_2
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn select_group_match_fn() -> GroupMatchFn {
    #[inline(always)]
    unsafe fn scalar_match(keys_ptr: *const u64, target: u64) -> u32 {
        // SAFETY: caller guarantees 1 readable u64 at keys_ptr
        if unsafe { *keys_ptr } == target { 1 } else { 0 }
    }
    scalar_match
}

/// Detects the best SIMD group size for this CPU at runtime.
fn detect_group_size() -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return 8;
        }
        return 4; // AVX2 is baseline for all modern x86_64.
    }
    #[cfg(target_arch = "aarch64")]
    {
        return 2;
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        return 1;
    }
}

/// Cached group size, detected once per process.
static DETECTED_GROUP_SIZE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

fn get_group_size() -> usize {
    *DETECTED_GROUP_SIZE.get_or_init(detect_group_size)
}

/// SIMD-accelerated open-addressing hash map for u64 keys.
///
/// Keys are their own hash (no hashing step). Probes in groups using the
/// best available SIMD: AVX-512 (8 keys), AVX2 (4 keys), or NEON/SVE (2 keys).
/// Detected once at runtime, cached for the process lifetime. No control byte
/// indirection (unlike Swiss Table). Direct u64 key match per SIMD instruction.
///
/// Values stored in parallel MaybeUninit array. Single-threaded, zero locks.
pub struct FlatU64Map<V> {
    keys: Vec<u64>,
    values: Vec<std::mem::MaybeUninit<V>>,
    capacity: usize,
    mask: usize,
    group_size: usize,
    /// SIMD compare function, selected once at creation. Indirect call avoids
    /// per-lookup branch on group_size.
    match_fn: GroupMatchFn,
    len: usize,
}

impl<V> FlatU64Map<V> {
    pub fn new() -> Self {
        Self::with_capacity(16)
    }

    pub fn with_capacity(min_cap: usize) -> Self {
        let gs = get_group_size();
        let raw = (min_cap * U64MAP_LOAD_DENOM / U64MAP_LOAD_NUMER + 1)
            .next_power_of_two()
            .max(16);
        // Round up to multiple of group_size.
        let capacity = (raw + (gs - 1)) & !(gs - 1);
        let mut keys = Vec::with_capacity(capacity);
        keys.resize(capacity, U64MAP_EMPTY);
        let mut values: Vec<std::mem::MaybeUninit<V>> = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            values.push(std::mem::MaybeUninit::uninit());
        }
        Self {
            keys,
            values,
            capacity,
            mask: capacity - 1,
            group_size: gs,
            match_fn: select_group_match_fn(),
            len: 0,
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Finds the slot index for a key, or None.
    /// SIMD-accelerated: checks group_size consecutive slots per iteration
    /// via stored function pointer (no per-lookup branch).
    #[inline(always)]
    fn find_slot(&self, key: u64) -> Option<usize> {
        let mut idx = (key as usize) & self.mask;
        let keys_ptr = self.keys.as_ptr();
        let gs = self.group_size;
        let mfn = self.match_fn;
        let mut probed = 0usize;
        while probed < self.capacity {
            if idx + gs <= self.capacity {
                let match_mask = unsafe { (mfn)(keys_ptr.add(idx), key) };
                if match_mask != 0 {
                    return Some(idx + match_mask.trailing_zeros() as usize);
                }
                let empty_mask = unsafe { (mfn)(keys_ptr.add(idx), U64MAP_EMPTY) };
                if empty_mask != 0 {
                    return None;
                }
                idx = (idx + gs) & self.mask;
                probed += gs;
            } else {
                let k = self.keys[idx];
                if k == key {
                    return Some(idx);
                }
                if k == U64MAP_EMPTY {
                    return None;
                }
                idx = (idx + 1) & self.mask;
                probed += 1;
            }
        }
        None
    }

    /// Finds the first empty slot starting from the ideal position.
    #[inline(always)]
    fn find_empty_slot(&self, key: u64) -> usize {
        let mut idx = (key as usize) & self.mask;
        loop {
            if self.keys[idx] == U64MAP_EMPTY {
                return idx;
            }
            idx = (idx + 1) & self.mask;
        }
    }

    #[inline]
    pub fn get(&self, key: u64) -> Option<&V> {
        self.find_slot(key)
            .map(|idx| unsafe { self.values[idx].assume_init_ref() })
    }

    #[inline]
    pub fn get_mut(&mut self, key: u64) -> Option<&mut V> {
        self.find_slot(key)
            .map(|idx| unsafe { self.values[idx].assume_init_mut() })
    }

    #[inline]
    pub fn get_or_insert_with(&mut self, key: u64, make_value: impl FnOnce() -> V) -> &mut V {
        if self.len * U64MAP_LOAD_DENOM >= self.capacity * U64MAP_LOAD_NUMER {
            self.grow();
        }
        if let Some(idx) = self.find_slot(key) {
            return unsafe { self.values[idx].assume_init_mut() };
        }
        let idx = self.find_empty_slot(key);
        self.keys[idx] = key;
        self.values[idx] = std::mem::MaybeUninit::new(make_value());
        self.len += 1;
        unsafe { self.values[idx].assume_init_mut() }
    }

    #[inline]
    pub fn insert(&mut self, key: u64, value: V) {
        if self.len * U64MAP_LOAD_DENOM >= self.capacity * U64MAP_LOAD_NUMER {
            self.grow();
        }
        if let Some(idx) = self.find_slot(key) {
            unsafe {
                self.values[idx].assume_init_drop();
            }
            self.values[idx] = std::mem::MaybeUninit::new(value);
            return;
        }
        let idx = self.find_empty_slot(key);
        self.keys[idx] = key;
        self.values[idx] = std::mem::MaybeUninit::new(value);
        self.len += 1;
    }

    pub fn remove(&mut self, key: u64) -> bool {
        let idx = match self.find_slot(key) {
            Some(i) => i,
            None => return false,
        };
        unsafe {
            self.values[idx].assume_init_drop();
        }
        self.keys[idx] = U64MAP_EMPTY;
        self.len -= 1;
        // Backward-shift deletion to maintain probe chains.
        let mut prev = idx;
        let mut cur = (idx + 1) & self.mask;
        loop {
            let ck = self.keys[cur];
            if ck == U64MAP_EMPTY {
                break;
            }
            let ideal = (ck as usize) & self.mask;
            let should_shift = if prev < cur {
                ideal <= prev || ideal > cur
            } else {
                ideal <= prev && ideal > cur
            };
            if !should_shift {
                break;
            }
            self.keys[prev] = self.keys[cur];
            self.values.swap(prev, cur);
            self.keys[cur] = U64MAP_EMPTY;
            prev = cur;
            cur = (cur + 1) & self.mask;
        }
        true
    }

    #[inline]
    pub fn iter(&self, mut f: impl FnMut(u64, &V)) {
        for i in 0..self.capacity {
            if self.keys[i] != U64MAP_EMPTY {
                f(self.keys[i], unsafe { self.values[i].assume_init_ref() });
            }
        }
    }

    #[inline]
    pub fn iter_mut(&mut self, mut f: impl FnMut(u64, &mut V)) {
        for i in 0..self.capacity {
            if self.keys[i] != U64MAP_EMPTY {
                f(self.keys[i], unsafe { self.values[i].assume_init_mut() });
            }
        }
    }

    pub fn retain(&mut self, mut pred: impl FnMut(u64, &mut V) -> bool) {
        let mut to_remove = Vec::new();
        for i in 0..self.capacity {
            if self.keys[i] != U64MAP_EMPTY {
                if !pred(self.keys[i], unsafe { self.values[i].assume_init_mut() }) {
                    to_remove.push(self.keys[i]);
                }
            }
        }
        for key in to_remove {
            self.remove(key);
        }
    }

    pub fn clear(&mut self) {
        for i in 0..self.capacity {
            if self.keys[i] != U64MAP_EMPTY {
                unsafe {
                    self.values[i].assume_init_drop();
                }
                self.keys[i] = U64MAP_EMPTY;
            }
        }
        self.len = 0;
    }

    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.keys.iter().enumerate().filter_map(move |(i, &k)| {
            if k != U64MAP_EMPTY {
                Some(unsafe { self.values[i].assume_init_ref() })
            } else {
                None
            }
        })
    }

    fn grow(&mut self) {
        let new_cap = self.capacity * 2;
        let mut new_keys = Vec::with_capacity(new_cap);
        new_keys.resize(new_cap, U64MAP_EMPTY);
        let mut new_values: Vec<std::mem::MaybeUninit<V>> = Vec::with_capacity(new_cap);
        for _ in 0..new_cap {
            new_values.push(std::mem::MaybeUninit::uninit());
        }
        let new_mask = new_cap - 1;

        let old_keys = std::mem::replace(&mut self.keys, new_keys);
        let old_values = std::mem::replace(&mut self.values, new_values);
        self.capacity = new_cap;
        self.mask = new_mask;
        self.len = 0;

        for (i, &key) in old_keys.iter().enumerate() {
            if key != U64MAP_EMPTY {
                let mut idx = (key as usize) & new_mask;
                loop {
                    if self.keys[idx] == U64MAP_EMPTY {
                        self.keys[idx] = key;
                        // Safety: old slot was occupied. Move the value out.
                        // MaybeUninit<V> is Copy-like in memory, no double-drop.
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                old_values[i].as_ptr(),
                                self.values[idx].as_mut_ptr(),
                                1,
                            );
                        }
                        self.len += 1;
                        break;
                    }
                    idx = (idx + 1) & new_mask;
                }
            }
        }
        // old_values Vec drops MaybeUninit<V> wrappers (no-op, no destructor).
        // old_keys Vec drops u64 values (no-op).
    }
}

impl<V> Default for FlatU64Map<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V> Drop for FlatU64Map<V> {
    fn drop(&mut self) {
        // Only drop occupied values.
        for i in 0..self.capacity {
            if self.keys[i] != U64MAP_EMPTY {
                unsafe {
                    self.values[i].assume_init_drop();
                }
            }
        }
        // MaybeUninit<V> has no Drop, so Vec<MaybeUninit<V>> drop is safe.
    }
}

// ---------------------------------------------------------------------------
// FlatHashTable: flat bucket array with prev-chain for join builds
// ---------------------------------------------------------------------------

/// Sentinel value indicating end of chain.
const FLAT_NULL: u32 = u32::MAX;

/// Flat hash table optimized for join build phases.
///
/// Uses a power-of-2 bucket array where each bucket stores the head of a
/// chain. Entries are stored externally in a Vec with (next_index, hash_hi32)
/// pairs. Insert is a single swap of the bucket head.
pub struct FlatHashTable {
    /// Bucket array: bucket[hash & mask] = head entry index.
    buckets: Vec<u32>,
    /// Mask for bucket index (capacity - 1).
    mask: u32,
    /// Entry chain: (next_entry_index, upper_32_bits_of_hash).
    entries: Vec<(u32, u32)>,
}

impl FlatHashTable {
    /// Creates a new table with the given capacity (rounded up to power of 2).
    pub fn new(expected_entries: usize) -> Self {
        let capacity = (expected_entries * 2).next_power_of_two().max(16);
        Self {
            buckets: vec![FLAT_NULL; capacity],
            mask: (capacity - 1) as u32,
            entries: Vec::with_capacity(expected_entries),
        }
    }

    /// Inserts a new entry. Returns the entry index.
    #[inline]
    pub fn insert(&mut self, hash: u64) -> u32 {
        let bucket_idx = (hash as u32) & self.mask;
        let hash_hi32 = (hash >> 32) as u32;
        let entry_idx = self.entries.len() as u32;
        let prev_head = self.buckets[bucket_idx as usize];
        self.buckets[bucket_idx as usize] = entry_idx;
        self.entries.push((prev_head, hash_hi32));
        entry_idx
    }

    /// Returns an iterator over entry indices matching the given hash.
    #[inline]
    pub fn get(&self, hash: u64) -> FlatHashIter<'_> {
        let bucket_idx = (hash as u32) & self.mask;
        let hash_hi32 = (hash >> 32) as u32;
        FlatHashIter {
            entries: &self.entries,
            cursor: self.buckets[bucket_idx as usize],
            hash_hi32,
        }
    }

    /// Prefetch the bucket for a given hash (for pipelined probing).
    #[inline]
    pub fn prefetch(&self, hash: u64) {
        let bucket_idx = ((hash as u32) & self.mask) as usize;
        // Safety: bucket_idx is always within bounds due to mask.
        let ptr = unsafe { self.buckets.as_ptr().add(bucket_idx) };
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = ptr;
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Iterator over entries in a FlatHashTable chain that match a specific hash.
pub struct FlatHashIter<'a> {
    entries: &'a [(u32, u32)],
    cursor: u32,
    hash_hi32: u32,
}

impl Iterator for FlatHashIter<'_> {
    type Item = u32;

    #[inline]
    fn next(&mut self) -> Option<u32> {
        while self.cursor != FLAT_NULL {
            let idx = self.cursor;
            let (next, stored_hi32) = self.entries[idx as usize];
            self.cursor = next;
            if stored_hi32 == self.hash_hi32 {
                return Some(idx);
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use zyron_common::hash_int;

    #[test]
    fn test_flat_u64_map_basic() {
        let mut map: FlatU64Map<&str> = FlatU64Map::new();
        map.insert(hash_int(42), "hello");
        map.insert(hash_int(99), "world");
        assert_eq!(map.get(hash_int(42)), Some(&"hello"));
        assert_eq!(map.get(hash_int(99)), Some(&"world"));
        assert_eq!(map.get(hash_int(0)), None);
        assert_eq!(map.len(), 2);

        // Overwrite.
        map.insert(hash_int(42), "updated");
        assert_eq!(map.get(hash_int(42)), Some(&"updated"));
        assert_eq!(map.len(), 2);

        // Remove.
        assert!(map.remove(hash_int(42)));
        assert_eq!(map.get(hash_int(42)), None);
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn test_flat_u64_map_grow() {
        let mut map: FlatU64Map<i64> = FlatU64Map::with_capacity(4);
        for i in 0..100i64 {
            map.insert(hash_int(i as u64), i * 10);
        }
        assert_eq!(map.len(), 100);
        for i in 0..100i64 {
            assert_eq!(map.get(hash_int(i as u64)), Some(&(i * 10)));
        }
    }

    #[test]
    fn test_flat_hash_table_insert_get() {
        let mut table = FlatHashTable::new(100);
        let h1 = hash_int(1);
        let h2 = hash_int(2);
        let h3 = hash_int(3);

        let idx1 = table.insert(h1);
        let idx2 = table.insert(h2);
        let idx3 = table.insert(h3);

        assert_eq!(idx1, 0);
        assert_eq!(idx2, 1);
        assert_eq!(idx3, 2);

        let results: Vec<u32> = table.get(h1).collect();
        assert_eq!(results, vec![0]);

        let results: Vec<u32> = table.get(h2).collect();
        assert_eq!(results, vec![1]);
    }

    #[test]
    fn test_flat_hash_table_collision_chain() {
        let mut table = FlatHashTable::new(100);
        // Insert multiple entries with the same hash to test chaining.
        let h = hash_int(42);
        table.insert(h);
        table.insert(h);
        table.insert(h);

        let results: Vec<u32> = table.get(h).collect();
        assert_eq!(results.len(), 3);
        // Should return in reverse insertion order (stack behavior).
        assert_eq!(results, vec![2, 1, 0]);
    }
}

//! Hashing kernels for streaming operators.
//!
//! Provides fibonacci hashing, boost-style hash combining, MurmurHash3
//! finalization, FNV-1a for strings, batch column hashing, an identity
//! hasher for pre-computed keys, and a flat hash table for join builds.
//! All functions match the algorithms used in zyron-executor/compute.rs
//! for consistency across the engine.

use crate::column::{StreamColumn, StreamColumnData};

// ---------------------------------------------------------------------------
// Hash constants
// ---------------------------------------------------------------------------

/// Fibonacci hashing constant (golden ratio * 2^64).
const HASH_GOLDEN: u64 = 0x9e3779b97f4a7c15;

/// FNV-1a offset basis.
const FNV_OFFSET: u64 = 0xcbf29ce484222325;

/// FNV-1a prime.
const FNV_PRIME: u64 = 0x100000001b3;

/// Sentinel hash for null values.
pub const HASH_NULL_SENTINEL: u64 = 0xdeadbeefcafebabe;

// ---------------------------------------------------------------------------
// Core hash functions
// ---------------------------------------------------------------------------

/// Fibonacci hashing for integer keys. Produces well-distributed 64-bit hashes
/// from integer values with a single multiply and shift.
#[inline(always)]
pub fn hash_int(v: i64) -> u64 {
    let h = (v as u64).wrapping_mul(HASH_GOLDEN);
    h ^ (h >> 32)
}

/// Boost-style hash combiner. Mixes a new value into an existing seed.
#[inline(always)]
pub fn hash_combine(seed: u64, value: u64) -> u64 {
    seed ^ (value
        .wrapping_add(HASH_GOLDEN)
        .wrapping_add(seed << 6)
        .wrapping_add(seed >> 2))
}

/// MurmurHash3 64-bit finalizer. Achieves full avalanche (every input bit
/// affects every output bit) in 3 multiply-xorshift steps.
#[inline(always)]
pub fn hash_finalize(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    x
}

/// FNV-1a hash for byte slices. Good distribution for strings and binary data.
#[inline]
pub fn hash_bytes_fnv(bytes: &[u8]) -> u64 {
    let mut h: u64 = FNV_OFFSET;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

// ---------------------------------------------------------------------------
// Batch column hashing
// ---------------------------------------------------------------------------

/// Hashes an entire column into a Vec<u64>, one hash per row.
/// Typed dispatch avoids per-row ScalarValue creation.
pub fn hash_column_batch(col: &StreamColumn, num_rows: usize) -> Vec<u64> {
    let mut hashes = Vec::with_capacity(num_rows);
    hash_column_batch_into(col, num_rows, &mut hashes);
    hashes
}

/// Hashes a column into an existing buffer, clearing it first.
/// Avoids allocation when the caller can reuse a buffer across calls.
pub fn hash_column_batch_into(col: &StreamColumn, num_rows: usize, hashes: &mut Vec<u64>) {
    hashes.clear();
    hashes.reserve(num_rows.saturating_sub(hashes.capacity()));
    let has_nulls = col.nulls.has_nulls();

    match &col.data {
        StreamColumnData::Int64(v) => {
            if has_nulls {
                for i in 0..num_rows {
                    if col.nulls.is_null(i) {
                        hashes.push(HASH_NULL_SENTINEL);
                    } else {
                        hashes.push(hash_int(v[i]));
                    }
                }
            } else {
                for i in 0..num_rows {
                    hashes.push(hash_int(v[i]));
                }
            }
        }
        StreamColumnData::Int32(v) => {
            if has_nulls {
                for i in 0..num_rows {
                    if col.nulls.is_null(i) {
                        hashes.push(HASH_NULL_SENTINEL);
                    } else {
                        hashes.push(hash_int(v[i] as i64));
                    }
                }
            } else {
                for i in 0..num_rows {
                    hashes.push(hash_int(v[i] as i64));
                }
            }
        }
        StreamColumnData::Int16(v) => {
            if has_nulls {
                for i in 0..num_rows {
                    if col.nulls.is_null(i) {
                        hashes.push(HASH_NULL_SENTINEL);
                    } else {
                        hashes.push(hash_int(v[i] as i64));
                    }
                }
            } else {
                for i in 0..num_rows {
                    hashes.push(hash_int(v[i] as i64));
                }
            }
        }
        StreamColumnData::Int8(v) => {
            if has_nulls {
                for i in 0..num_rows {
                    if col.nulls.is_null(i) {
                        hashes.push(HASH_NULL_SENTINEL);
                    } else {
                        hashes.push(hash_int(v[i] as i64));
                    }
                }
            } else {
                for i in 0..num_rows {
                    hashes.push(hash_int(v[i] as i64));
                }
            }
        }
        StreamColumnData::Int128(v) => {
            if has_nulls {
                for i in 0..num_rows {
                    if col.nulls.is_null(i) {
                        hashes.push(HASH_NULL_SENTINEL);
                    } else {
                        let lo = v[i] as u64;
                        let hi = (v[i] >> 64) as u64;
                        hashes.push(hash_combine(hash_int(lo as i64), hi));
                    }
                }
            } else {
                for i in 0..num_rows {
                    let lo = v[i] as u64;
                    let hi = (v[i] >> 64) as u64;
                    hashes.push(hash_combine(hash_int(lo as i64), hi));
                }
            }
        }
        StreamColumnData::UInt64(v) => {
            if has_nulls {
                for i in 0..num_rows {
                    if col.nulls.is_null(i) {
                        hashes.push(HASH_NULL_SENTINEL);
                    } else {
                        hashes.push(hash_int(v[i] as i64));
                    }
                }
            } else {
                for i in 0..num_rows {
                    hashes.push(hash_int(v[i] as i64));
                }
            }
        }
        StreamColumnData::UInt32(v) => {
            if has_nulls {
                for i in 0..num_rows {
                    if col.nulls.is_null(i) {
                        hashes.push(HASH_NULL_SENTINEL);
                    } else {
                        hashes.push(hash_int(v[i] as i64));
                    }
                }
            } else {
                for i in 0..num_rows {
                    hashes.push(hash_int(v[i] as i64));
                }
            }
        }
        StreamColumnData::UInt16(v) => {
            if has_nulls {
                for i in 0..num_rows {
                    if col.nulls.is_null(i) {
                        hashes.push(HASH_NULL_SENTINEL);
                    } else {
                        hashes.push(hash_int(v[i] as i64));
                    }
                }
            } else {
                for i in 0..num_rows {
                    hashes.push(hash_int(v[i] as i64));
                }
            }
        }
        StreamColumnData::UInt8(v) => {
            if has_nulls {
                for i in 0..num_rows {
                    if col.nulls.is_null(i) {
                        hashes.push(HASH_NULL_SENTINEL);
                    } else {
                        hashes.push(hash_int(v[i] as i64));
                    }
                }
            } else {
                for i in 0..num_rows {
                    hashes.push(hash_int(v[i] as i64));
                }
            }
        }
        StreamColumnData::Float64(v) => {
            if has_nulls {
                for i in 0..num_rows {
                    if col.nulls.is_null(i) {
                        hashes.push(HASH_NULL_SENTINEL);
                    } else {
                        hashes.push(hash_int(v[i].to_bits() as i64));
                    }
                }
            } else {
                for i in 0..num_rows {
                    hashes.push(hash_int(v[i].to_bits() as i64));
                }
            }
        }
        StreamColumnData::Float32(v) => {
            if has_nulls {
                for i in 0..num_rows {
                    if col.nulls.is_null(i) {
                        hashes.push(HASH_NULL_SENTINEL);
                    } else {
                        hashes.push(hash_int(v[i].to_bits() as i64));
                    }
                }
            } else {
                for i in 0..num_rows {
                    hashes.push(hash_int(v[i].to_bits() as i64));
                }
            }
        }
        StreamColumnData::Boolean(v) => {
            if has_nulls {
                for i in 0..num_rows {
                    if col.nulls.is_null(i) {
                        hashes.push(HASH_NULL_SENTINEL);
                    } else {
                        hashes.push(hash_int(v[i] as i64));
                    }
                }
            } else {
                for i in 0..num_rows {
                    hashes.push(hash_int(v[i] as i64));
                }
            }
        }
        StreamColumnData::Utf8(v) => {
            if has_nulls {
                for i in 0..num_rows {
                    if col.nulls.is_null(i) {
                        hashes.push(HASH_NULL_SENTINEL);
                    } else {
                        hashes.push(hash_bytes_fnv(v[i].as_bytes()));
                    }
                }
            } else {
                for i in 0..num_rows {
                    hashes.push(hash_bytes_fnv(v[i].as_bytes()));
                }
            }
        }
        StreamColumnData::Binary(v) => {
            if has_nulls {
                for i in 0..num_rows {
                    if col.nulls.is_null(i) {
                        hashes.push(HASH_NULL_SENTINEL);
                    } else {
                        hashes.push(hash_bytes_fnv(&v[i]));
                    }
                }
            } else {
                for i in 0..num_rows {
                    hashes.push(hash_bytes_fnv(&v[i]));
                }
            }
        }
    }
}

/// Combines hashes from multiple columns into a single hash per row.
/// Uses hash_combine to fold each column's hash into the running seed.
pub fn hash_multi_column_batch(cols: &[&StreamColumn], num_rows: usize) -> Vec<u64> {
    let mut hashes = Vec::with_capacity(num_rows);
    hash_multi_column_batch_into(cols, num_rows, &mut hashes);
    hashes
}

/// Combines hashes from multiple columns into an existing buffer.
/// Clears the buffer first. Avoids allocation when reusing across calls.
pub fn hash_multi_column_batch_into(
    cols: &[&StreamColumn],
    num_rows: usize,
    hashes: &mut Vec<u64>,
) {
    if cols.is_empty() {
        hashes.clear();
        hashes.resize(num_rows, 0);
        return;
    }

    hash_column_batch_into(cols[0], num_rows, hashes);
    let mut col_hashes = Vec::with_capacity(num_rows);
    for col in &cols[1..] {
        hash_column_batch_into(col, num_rows, &mut col_hashes);
        for (h, ch) in hashes.iter_mut().zip(col_hashes.iter()) {
            *h = hash_combine(*h, *ch);
        }
    }

    // Finalize all hashes for better distribution.
    for h in hashes.iter_mut() {
        *h = hash_finalize(*h);
    }
}

// ---------------------------------------------------------------------------
// FlatU64Map<V>: SIMD-accelerated open-addressing table for u64 keys
// ---------------------------------------------------------------------------

/// Empty slot sentinel. Keys equal to u64::MAX cannot be stored in this map.
/// All hash functions in this crate produce well-distributed values where
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
    let group = _mm256_loadu_si256(keys_ptr as *const __m256i);
    let needle = _mm256_set1_epi64x(target as i64);
    let cmp = _mm256_cmpeq_epi64(group, needle);
    _mm256_movemask_pd(_mm256_castsi256_pd(cmp)) as u32
}

/// AVX-512: compare 8 u64 keys in one instruction.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn group_match_8(keys_ptr: *const u64, target: u64) -> u32 {
    use std::arch::x86_64::*;
    let group = _mm512_loadu_si512(keys_ptr as *const __m512i);
    let needle = _mm512_set1_epi64(target as i64);
    _mm512_cmpeq_epi64_mask(group, needle) as u32
}

/// NEON: compare 2 u64 keys.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn group_match_2(keys_ptr: *const u64, target: u64) -> u32 {
    use std::arch::aarch64::*;
    let group = vld1q_u64(keys_ptr);
    let needle = vdupq_n_u64(target);
    let cmp = vceqq_u64(group, needle);
    let b0 = if vgetq_lane_u64(cmp, 0) != 0 { 1u32 } else { 0 };
    let b1 = if vgetq_lane_u64(cmp, 1) != 0 { 2u32 } else { 0 };
    b0 | b1
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
        if *keys_ptr == target { 1 } else { 0 }
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
    use crate::column::{NullBitmap, StreamColumn, StreamColumnData};

    #[test]
    fn test_hash_int_distribution() {
        // Consecutive integers should produce different hashes.
        let h0 = hash_int(0);
        let h1 = hash_int(1);
        let h2 = hash_int(2);
        assert_ne!(h0, h1);
        assert_ne!(h1, h2);
        assert_ne!(h0, h2);
    }

    #[test]
    fn test_hash_combine_order_matters() {
        let a = hash_combine(100, 200);
        let b = hash_combine(200, 100);
        assert_ne!(a, b);
    }

    #[test]
    fn test_hash_finalize_avalanche() {
        let h1 = hash_finalize(1);
        let h2 = hash_finalize(2);
        // Differ in many bits, not just 1.
        assert!((h1 ^ h2).count_ones() > 10);
    }

    #[test]
    fn test_hash_bytes_fnv() {
        let h1 = hash_bytes_fnv(b"hello");
        let h2 = hash_bytes_fnv(b"world");
        assert_ne!(h1, h2);
        // Same input, same hash.
        assert_eq!(hash_bytes_fnv(b"hello"), h1);
    }

    #[test]
    fn test_hash_column_batch_int64() {
        let col = StreamColumn::from_data(StreamColumnData::Int64(vec![10, 20, 30]));
        let hashes = hash_column_batch(&col, 3);
        assert_eq!(hashes.len(), 3);
        assert_ne!(hashes[0], hashes[1]);
        assert_ne!(hashes[1], hashes[2]);
    }

    #[test]
    fn test_hash_column_batch_with_nulls() {
        let mut nulls = NullBitmap::new_valid(3);
        nulls.set_null(1);
        let col = StreamColumn::new(StreamColumnData::Int64(vec![10, 20, 30]), nulls);
        let hashes = hash_column_batch(&col, 3);
        assert_eq!(hashes[1], HASH_NULL_SENTINEL);
        assert_ne!(hashes[0], HASH_NULL_SENTINEL);
    }

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
            map.insert(hash_int(i), i * 10);
        }
        assert_eq!(map.len(), 100);
        for i in 0..100i64 {
            assert_eq!(map.get(hash_int(i)), Some(&(i * 10)));
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

    #[test]
    fn test_multi_column_hash() {
        let col1 = StreamColumn::from_data(StreamColumnData::Int64(vec![1, 2, 3]));
        let col2 = StreamColumn::from_data(StreamColumnData::Utf8(vec![
            "a".into(),
            "b".into(),
            "c".into(),
        ]));
        let hashes = hash_multi_column_batch(&[&col1, &col2], 3);
        assert_eq!(hashes.len(), 3);
        // Each row should have a distinct combined hash.
        assert_ne!(hashes[0], hashes[1]);
        assert_ne!(hashes[1], hashes[2]);
    }
}

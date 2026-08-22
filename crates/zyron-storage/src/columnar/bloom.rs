//! Split-block bloom filter for segment-level membership pruning.
//!
//! Each probe touches exactly one 64-byte cache-line-aligned block.
//! Keys hash with the canonical hot-path 128-bit hash. The low half selects
//! the block, the high half is the in-block probe base, and an odd step
//! from the low half advances the double-hashing sequence, see
//! `probe_params` for why the base and the block selection must come from
//! different halves. The hash carries no runtime dispatch and no lane
//! setup, the property the previous local fork existed for: the central
//! AES Hasher's dispatch and lane initialization measured +78% per probe
//! at this site.
//!
//! Multi-key callers use the batch probes, which hash a group of keys on
//! independent chains and prefetch every key's block before testing any
//! bits, so the block cache misses overlap instead of serializing.

use crate::columnar::constants::*;
use zyron_common::checksum::hot::hot_hash128;
use zyron_common::{Result, ZyronError};

/// Hashes one key to the 128-bit (block, probe) pair
#[inline(always)]
fn bloom_hash(data: &[u8]) -> u128 {
    hot_hash128(data)
}

/// Derives the probe parameters from one 128-bit hash: block start from the
/// low half, in-block probe base from the high half, odd probe step from
/// the low half.
///
/// The base must come from a different half than the block selection.
/// Selecting the block fixes h1 modulo num_blocks, and whenever
/// gcd(num_blocks, 512) > 1 an h1-derived base is then confined to a
/// fraction of the block's bit positions, which measured at 3-5x the
/// natural false positive rate. The odd step is coprime to the 512
/// block bits, so the probe sequence visits distinct positions
#[inline(always)]
fn probe_params(hash: u128, num_blocks: u32) -> (usize, u64, u64) {
    let h1 = hash as u64;
    let h2 = (hash >> 64) as u64;
    let block_start = (h1 % num_blocks as u64) as usize * BLOOM_BLOCK_SIZE;
    (block_start, h2, h1 | 1)
}

/// Prefetches the cache line holding a block into L1. The block size is one
/// cache line, so one prefetch covers every probe of that key
#[inline(always)]
fn prefetch_block(bits: &[u8], block_start: usize) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: block_start is within bits, computed as block index times
    // BLOOM_BLOCK_SIZE against a validated length. Prefetch has no
    // architectural effect beyond the cache
    unsafe {
        std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(
            bits.as_ptr().add(block_start) as *const i8,
        );
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (bits, block_start);
    }
}

/// Tests every probe position of one key without early exit. The branchless
/// accumulation keeps a batch of keys advancing without per-probe branch
/// mispredictions, and all probes land in one already-prefetched cache line
#[inline(always)]
fn probe_block_all(bits: &[u8], block_start: usize, base: u64, step: u64, hash_count: u32) -> bool {
    let ptr = bits.as_ptr();
    let mut all = true;
    let mut probe = base;
    for _ in 0..hash_count {
        let bit_pos = probe % BLOCK_BITS as u64;
        // SAFETY: bit_pos < 512 so the byte offset is within the 64-byte
        // block at block_start, which the caller derived from a validated
        // bit array length
        let byte = unsafe { *ptr.add(block_start + (bit_pos >> 3) as usize) };
        all &= (byte >> (bit_pos & 7)) & 1 == 1;
        probe = probe.wrapping_add(step);
    }
    all
}

/// Keys probed per prefetch group in the batch paths. Four independent
/// hash chains fill the multiplier pipeline and four outstanding block
/// prefetches overlap their potential cache misses
const BATCH_GROUP: usize = 4;

/// Probes a group of already-hashed keys against a validated bit array:
/// compute all block addresses, prefetch all blocks, then test bits
#[inline(always)]
fn probe_hashed_group(
    bits: &[u8],
    num_blocks: u32,
    hash_count: u32,
    hashes: &[u128],
    results: &mut [bool],
) {
    // One probe_params call per key: the derivation lives in one place and
    // the block-selecting division runs once, with the prefetch issued as
    // soon as each key's block is known
    let mut params = [(0usize, 0u64, 0u64); BATCH_GROUP];
    for (j, h) in hashes.iter().enumerate() {
        params[j] = probe_params(*h, num_blocks);
        prefetch_block(bits, params[j].0);
    }
    for (j, &(block_start, base, step)) in params.iter().take(hashes.len()).enumerate() {
        results[j] = probe_block_all(bits, block_start, base, step, hash_count);
    }
}

/// Batch probe core over a validated bit array. Hashes and probes keys in
/// groups of BATCH_GROUP so hash chains and block fetches overlap, with a
/// sequential tail for the remainder. `hash_at` supplies the hash of the
/// key at an index, so the u64 and byte-slice entry points share this one
/// body and cannot drift apart
fn probe_batch(
    bits: &[u8],
    num_blocks: u32,
    hash_count: u32,
    count: usize,
    hash_at: impl Fn(usize) -> u128,
    results: &mut [bool],
) {
    let mut i = 0;
    while i + BATCH_GROUP <= count {
        let mut hashes = [0u128; BATCH_GROUP];
        for j in 0..BATCH_GROUP {
            hashes[j] = hash_at(i + j);
        }
        probe_hashed_group(
            bits,
            num_blocks,
            hash_count,
            &hashes,
            &mut results[i..i + BATCH_GROUP],
        );
        i += BATCH_GROUP;
    }
    while i < count {
        let (block_start, base, step) = probe_params(hash_at(i), num_blocks);
        results[i] = probe_block_all(bits, block_start, base, step, hash_count);
        i += 1;
    }
}

/// Version of the key hash and probe derivation baked into serialized
/// filters. A stored filter's bits are only meaningful to the scheme that
/// set them, so every reader checks this field before trusting a probe
/// answer. Version 2 is the canonical 128-bit key hash with the decoupled
/// probe derivation in probe_params
pub const BLOOM_ALGORITHM_VERSION: u32 = 2;

/// Serialization header size: algorithm_version(4) + hash_count(4) +
/// num_blocks(4) + num_elements(8)
const HEADER_SIZE: usize = 20;

/// Bits per block (BLOOM_BLOCK_SIZE * 8).
const BLOCK_BITS: u32 = (BLOOM_BLOCK_SIZE * 8) as u32;

/// Split-block bloom filter with cache-line aligned blocks.
///
/// The bit array length is always a multiple of BLOOM_BLOCK_SIZE (64 bytes).
/// Each insert or probe hashes the value once with the central 128-bit
/// hash, selects a single block, then sets or checks BLOOM_HASH_COUNT bit
/// positions within that block via double hashing: bit_i = (base + i *
/// step) % 512 with base and step from `probe_params`.
pub struct BloomFilter {
    /// Bit array. Length is always num_blocks * BLOOM_BLOCK_SIZE.
    bits: Vec<u8>,
    /// Number of 64-byte blocks in the filter.
    numBlocks: u32,
    /// Number of hash probes per element.
    hashCount: u32,
    /// Count of elements inserted.
    numElements: u64,
}

impl BloomFilter {
    /// Allocates a bloom filter sized for the given number of expected elements.
    ///
    /// Total bits = expected_elements * BLOOM_BITS_PER_ELEMENT, rounded up to
    /// the nearest multiple of BLOOM_BLOCK_SIZE * 8 (512 bits per block).
    /// The minimum allocation is one block.
    pub fn new(expectedElements: u64) -> Self {
        let totalBits = expectedElements.saturating_mul(BLOOM_BITS_PER_ELEMENT as u64);
        let bitsPerBlock = BLOCK_BITS as u64;

        // Round up to the nearest whole block count, minimum 1 block.
        let numBlocks = totalBits.div_ceil(bitsPerBlock).max(1) as u32;
        let byteCount = numBlocks as usize * BLOOM_BLOCK_SIZE;

        Self {
            bits: vec![0u8; byteCount],
            numBlocks,
            hashCount: BLOOM_HASH_COUNT,
            numElements: 0,
        }
    }

    /// Inserts a value into the bloom filter.
    ///
    /// Hashes the value with the central 128-bit hash, selects a block via
    /// the lower 64 bits, then sets BLOOM_HASH_COUNT bit positions within
    /// that block using double hashing: bit_i = (base + i * step) % 512
    /// with base and step from `probe_params`.
    pub fn insert(&mut self, value: &[u8]) {
        let hash = bloom_hash(value);
        let (blockStart, base, step) = probe_params(hash, self.numBlocks);

        for i in 0..self.hashCount {
            let bitPos = base.wrapping_add((i as u64).wrapping_mul(step)) % BLOCK_BITS as u64;
            let byteOffset = blockStart + (bitPos / 8) as usize;
            let bitMask = 1u8 << (bitPos % 8);
            self.bits[byteOffset] |= bitMask;
        }

        self.numElements += 1;
    }

    /// Checks whether a value might be in the set.
    ///
    /// Returns true if all probed bits are set (may false positive).
    /// Returns false if any probed bit is unset (no false negatives).
    #[inline]
    pub fn might_contain(&self, value: &[u8]) -> bool {
        let hash = bloom_hash(value);
        let (blockStart, base, step) = probe_params(hash, self.numBlocks);
        let ptr = self.bits.as_ptr();

        // Unrolled probe loop using raw pointer reads to skip bounds checks.
        // The block is always BLOOM_BLOCK_SIZE (64) bytes, and bitPos is always
        // mod 512, so byteOffset is always within [blockStart, blockStart+63].
        for i in 0..self.hashCount {
            let bitPos = base.wrapping_add((i as u64).wrapping_mul(step)) % BLOCK_BITS as u64;
            let byteOffset = blockStart + (bitPos >> 3) as usize;
            let bitMask = 1u8 << (bitPos & 7);
            let byte = unsafe { *ptr.add(byteOffset) };
            if byte & bitMask == 0 {
                return false;
            }
        }

        true
    }

    /// Batch membership test for u64 keys, writing one answer per key.
    ///
    /// Each key is hashed exactly as `might_contain(&key.to_le_bytes())`
    /// and the answers match the sequential probes bit for bit. Keys are
    /// processed in groups whose hash chains run independently and whose
    /// blocks are all prefetched before any bit test, so a batch of cache
    /// misses overlaps instead of serializing. Callers probing several keys
    /// per iteration use this instead of a might_contain loop.
    ///
    /// `results` must be the same length as `keys`.
    pub fn might_contain_batch(&self, keys: &[u64], results: &mut [bool]) {
        assert_eq!(
            keys.len(),
            results.len(),
            "results length must match keys length"
        );
        probe_batch(
            &self.bits,
            self.numBlocks,
            self.hashCount,
            keys.len(),
            |i| bloom_hash(&keys[i].to_le_bytes()),
            results,
        );
    }

    /// Batch membership test for byte-slice keys, the same contract as
    /// [`Self::might_contain_batch`] with each value hashed exactly as
    /// `might_contain(value)`.
    pub fn might_contain_batch_bytes(&self, values: &[&[u8]], results: &mut [bool]) {
        assert_eq!(
            values.len(),
            results.len(),
            "results length must match values length"
        );
        probe_batch(
            &self.bits,
            self.numBlocks,
            self.hashCount,
            values.len(),
            |i| bloom_hash(values[i]),
            results,
        );
    }

    /// Serializes the bloom filter to bytes.
    ///
    /// Layout: algorithm_version(4 LE) + hash_count(4 LE) +
    /// num_blocks(4 LE) + num_elements(8 LE) + bits.
    pub fn to_bytes(&self) -> Vec<u8> {
        let totalSize = HEADER_SIZE + self.bits.len();
        let mut buf = Vec::with_capacity(totalSize);

        buf.extend_from_slice(&BLOOM_ALGORITHM_VERSION.to_le_bytes());
        buf.extend_from_slice(&self.hashCount.to_le_bytes());
        buf.extend_from_slice(&self.numBlocks.to_le_bytes());
        buf.extend_from_slice(&self.numElements.to_le_bytes());
        buf.extend_from_slice(&self.bits);

        buf
    }

    /// Deserializes a bloom filter from bytes.
    ///
    /// Validates header fields and buffer length. Returns an error if the
    /// data is truncated, has zero blocks, or has a mismatched bit array length.
    pub fn from_bytes(buf: &[u8]) -> Result<Self> {
        if buf.len() < HEADER_SIZE {
            return Err(ZyronError::DecodingFailed(format!(
                "bloom filter buffer too small: {} bytes, need at least {}",
                buf.len(),
                HEADER_SIZE
            )));
        }

        let version = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        if version != BLOOM_ALGORITHM_VERSION {
            return Err(ZyronError::DecodingFailed(format!(
                "bloom filter algorithm version {} is not the supported version {}",
                version, BLOOM_ALGORITHM_VERSION
            )));
        }
        let hashCount = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
        let numBlocks = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);
        let numElements = u64::from_le_bytes([
            buf[12], buf[13], buf[14], buf[15], buf[16], buf[17], buf[18], buf[19],
        ]);

        if numBlocks == 0 {
            return Err(ZyronError::DecodingFailed(
                "bloom filter num_blocks is zero".to_string(),
            ));
        }

        let expectedBitsLen = numBlocks as usize * BLOOM_BLOCK_SIZE;
        let actualBitsLen = buf.len() - HEADER_SIZE;

        if actualBitsLen != expectedBitsLen {
            return Err(ZyronError::DecodingFailed(format!(
                "bloom filter bit array length mismatch: expected {} bytes ({} blocks * {}), got {}",
                expectedBitsLen, numBlocks, BLOOM_BLOCK_SIZE, actualBitsLen
            )));
        }

        if hashCount == 0 {
            return Err(ZyronError::DecodingFailed(
                "bloom filter hash_count is zero".to_string(),
            ));
        }

        if hashCount > BLOOM_HASH_COUNT * 2 {
            return Err(ZyronError::DecodingFailed(format!(
                "bloom filter hash count {} exceeds maximum {}",
                hashCount,
                BLOOM_HASH_COUNT * 2
            )));
        }

        let bits = buf[HEADER_SIZE..].to_vec();

        Ok(Self {
            bits,
            numBlocks,
            hashCount,
            numElements,
        })
    }

    /// Returns the total serialized byte count: 20-byte header + bit array.
    pub fn on_disk_size(&self) -> usize {
        HEADER_SIZE + self.bits.len()
    }
}

/// Probes serialized bloom bytes in place, without copying the bit array.
///
/// This is the pruning-path probe: a manifest carries a file's bloom bytes
/// verbatim, and deciding whether to open that file must cost no allocation.
/// Returns false only when the value is provably absent, so a truncated or
/// otherwise unreadable buffer answers true and prunes nothing, since a
/// false negative would silently drop rows.
#[inline]
pub fn might_contain_serialized(buf: &[u8], value: &[u8]) -> bool {
    let Some((bits, numBlocks, hashCount)) = validate_serialized(buf) else {
        return true;
    };

    let hash = bloom_hash(value);
    let (blockStart, base, step) = probe_params(hash, numBlocks);

    for i in 0..hashCount {
        let bitPos = base.wrapping_add((i as u64).wrapping_mul(step)) % BLOCK_BITS as u64;
        let byteOffset = blockStart + (bitPos >> 3) as usize;
        let bitMask = 1u8 << (bitPos & 7);
        match bits.get(byteOffset) {
            Some(byte) if byte & bitMask != 0 => {}
            // An out-of-range offset cannot happen for a validated length,
            // treat it as unknown rather than as proof of absence
            Some(_) => return false,
            None => return true,
        }
    }
    true
}

/// Batch probe of serialized bloom bytes in place, one answer per value.
///
/// Each answer matches `might_contain_serialized(buf, value)` bit for bit,
/// including the conservative rules: a header that fails validation answers
/// true for every value and prunes nothing. The header validates once for
/// the whole batch instead of once per value, and the group prefetch
/// overlaps the block fetches the sequential loop would serialize.
///
/// `results` must be the same length as `values`.
#[inline]
pub fn might_contain_serialized_batch(buf: &[u8], values: &[&[u8]], results: &mut [bool]) {
    assert_eq!(
        values.len(),
        results.len(),
        "results length must match values length"
    );
    let Some((bits, numBlocks, hashCount)) = validate_serialized(buf) else {
        results.fill(true);
        return;
    };
    probe_batch(
        bits,
        numBlocks,
        hashCount,
        values.len(),
        |i| bloom_hash(values[i]),
        results,
    );
}

/// Validates a serialized filter's header for probing. None means the
/// buffer proves nothing absent: too short, zero or out-of-range header
/// fields, a mismatched bit array length, or bits set by a different
/// algorithm version
#[inline]
fn validate_serialized(buf: &[u8]) -> Option<(&[u8], u32, u32)> {
    if buf.len() < HEADER_SIZE {
        return None;
    }
    let version = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    let hashCount = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
    let numBlocks = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);
    if version != BLOOM_ALGORITHM_VERSION
        || numBlocks == 0
        || hashCount == 0
        || hashCount > BLOOM_HASH_COUNT * 2
    {
        return None;
    }
    let bits = &buf[HEADER_SIZE..];
    if bits.len() != numBlocks as usize * BLOOM_BLOCK_SIZE {
        return None;
    }
    Some((bits, numBlocks, hashCount))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_probe() {
        let mut filter = BloomFilter::new(100);
        filter.insert(b"hello");
        filter.insert(b"world");

        assert!(filter.might_contain(b"hello"));
        assert!(filter.might_contain(b"world"));
        assert_eq!(filter.numElements, 2);
    }

    #[test]
    fn test_no_false_negatives() {
        // Insert 1000 known values and verify every one is found.
        let elementCount = 1000u64;
        let mut filter = BloomFilter::new(elementCount);

        for i in 0..elementCount {
            let key = format!("key_{}", i);
            filter.insert(key.as_bytes());
        }

        for i in 0..elementCount {
            let key = format!("key_{}", i);
            assert!(
                filter.might_contain(key.as_bytes()),
                "false negative for {}",
                key
            );
        }
    }

    #[test]
    fn test_false_positive_rate() {
        // Insert 10,000 elements, then probe 100,000 non-inserted values.
        // At 10 bits/element with 7 hashes, the expected FPR is ~0.82%.
        // Allow up to 2% to account for split-block variance.
        let insertCount = 10_000u64;
        let probeCount = 100_000u64;

        let mut filter = BloomFilter::new(insertCount);

        for i in 0..insertCount {
            let key = format!("inserted_{}", i);
            filter.insert(key.as_bytes());
        }

        let mut falsePositives = 0u64;
        for i in 0..probeCount {
            let key = format!("absent_{}", i);
            if filter.might_contain(key.as_bytes()) {
                falsePositives += 1;
            }
        }

        let fpr = falsePositives as f64 / probeCount as f64;
        // Split-block bloom filters run above the classic uniform-fill rate
        // due to block confinement. With the probe base decoupled from the
        // block selection this configuration measures ~1-2%, allow 3%
        assert!(
            fpr < 0.03,
            "false positive rate too high: {:.4} ({} / {})",
            fpr,
            falsePositives,
            probeCount
        );
    }

    #[test]
    fn test_empty_filter() {
        let filter = BloomFilter::new(100);
        assert!(!filter.might_contain(b"anything"));
        assert!(!filter.might_contain(b""));
        assert_eq!(filter.numElements, 0);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut filter = BloomFilter::new(500);
        for i in 0..500u64 {
            let key = i.to_le_bytes();
            filter.insert(&key);
        }

        let serialized = filter.to_bytes();
        assert_eq!(serialized.len(), filter.on_disk_size());

        let restored =
            BloomFilter::from_bytes(&serialized).expect("deserialization should succeed");

        assert_eq!(restored.numBlocks, filter.numBlocks);
        assert_eq!(restored.hashCount, filter.hashCount);
        assert_eq!(restored.numElements, filter.numElements);
        assert_eq!(restored.bits, filter.bits);

        // Verify functional equivalence after roundtrip.
        for i in 0..500u64 {
            let key = i.to_le_bytes();
            assert!(restored.might_contain(&key));
        }
    }

    #[test]
    fn test_from_bytes_too_small() {
        let buf = vec![0u8; 10];
        let result = BloomFilter::from_bytes(&buf);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_bytes_zero_blocks() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&BLOOM_ALGORITHM_VERSION.to_le_bytes());
        buf.extend_from_slice(&7u32.to_le_bytes()); // hash_count
        buf.extend_from_slice(&0u32.to_le_bytes()); // num_blocks = 0
        buf.extend_from_slice(&0u64.to_le_bytes()); // num_elements

        let result = BloomFilter::from_bytes(&buf);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_bytes_length_mismatch() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&BLOOM_ALGORITHM_VERSION.to_le_bytes());
        buf.extend_from_slice(&7u32.to_le_bytes()); // hash_count
        buf.extend_from_slice(&2u32.to_le_bytes()); // num_blocks = 2
        buf.extend_from_slice(&0u64.to_le_bytes()); // num_elements
        buf.extend_from_slice(&[0u8; 50]); // 50 bytes, but 2 blocks need 128

        let result = BloomFilter::from_bytes(&buf);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_bytes_zero_hash_count() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&BLOOM_ALGORITHM_VERSION.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // hash_count = 0
        buf.extend_from_slice(&1u32.to_le_bytes()); // num_blocks = 1
        buf.extend_from_slice(&0u64.to_le_bytes()); // num_elements
        buf.extend_from_slice(&[0u8; BLOOM_BLOCK_SIZE]); // 64 bytes for 1 block

        let result = BloomFilter::from_bytes(&buf);
        assert!(result.is_err());
    }

    /// A filter tagged with a different algorithm version proves nothing
    /// absent: loading errors, probing prunes nothing
    #[test]
    fn test_wrong_algorithm_version_is_conservative() {
        let mut filter = BloomFilter::new(100);
        filter.insert(b"present");
        let mut serialized = filter.to_bytes();
        serialized[0..4].copy_from_slice(&(BLOOM_ALGORITHM_VERSION + 1).to_le_bytes());

        assert!(BloomFilter::from_bytes(&serialized).is_err());
        assert!(might_contain_serialized(&serialized, b"present"));
        assert!(might_contain_serialized(&serialized, b"never inserted"));

        let values: Vec<&[u8]> = vec![b"present", b"never inserted"];
        let mut results = vec![false; values.len()];
        might_contain_serialized_batch(&serialized, &values, &mut results);
        assert!(results.iter().all(|&r| r));
    }

    #[test]
    fn test_on_disk_size() {
        let filter = BloomFilter::new(1000);
        let expectedBitsLen = filter.numBlocks as usize * BLOOM_BLOCK_SIZE;
        assert_eq!(filter.on_disk_size(), HEADER_SIZE + expectedBitsLen);
    }

    #[test]
    fn test_minimum_one_block() {
        // Even with 0 expected elements, the filter allocates at least one block.
        let filter = BloomFilter::new(0);
        assert_eq!(filter.numBlocks, 1);
        assert_eq!(filter.bits.len(), BLOOM_BLOCK_SIZE);
    }

    #[test]
    fn test_block_alignment() {
        // The bit array length is always a multiple of BLOOM_BLOCK_SIZE.
        for count in [1, 10, 100, 1000, 50_000] {
            let filter = BloomFilter::new(count);
            assert_eq!(
                filter.bits.len() % BLOOM_BLOCK_SIZE,
                0,
                "misaligned for {} elements",
                count
            );
        }
    }

    #[test]
    fn test_single_byte_value() {
        let mut filter = BloomFilter::new(10);
        filter.insert(&[0xFF]);
        assert!(filter.might_contain(&[0xFF]));
        assert!(!filter.might_contain(&[0xFE]));
    }

    #[test]
    fn test_empty_value() {
        let mut filter = BloomFilter::new(10);
        filter.insert(b"");
        assert!(filter.might_contain(b""));
    }

    /// Deterministic pseudo-random test keys from the canonical splitmix64
    fn next_key(state: &mut u64) -> u64 {
        *state = zyron_common::splitmix64(*state);
        *state
    }

    /// The batch probe must agree with the sequential probe on every key,
    /// present or absent, across group boundaries and the remainder tail
    #[test]
    fn test_batch_matches_sequential_on_10k_random_keys() {
        let mut filter = BloomFilter::new(4096);
        let mut state = 0xB100_F11E_D5EEDu64;
        let inserted: Vec<u64> = (0..4096).map(|_| next_key(&mut state)).collect();
        for k in &inserted {
            filter.insert(&k.to_le_bytes());
        }

        // Mix of present and absent keys, lengths exercising every tail size
        let mut keys: Vec<u64> = (0..6000).map(|_| next_key(&mut state)).collect();
        keys.extend(inserted.iter().take(4000));

        for probe_len in [0usize, 1, 3, 4, 5, 7, 8, 10_000] {
            let subset = &keys[..probe_len];
            let mut batch = vec![false; probe_len];
            filter.might_contain_batch(subset, &mut batch);
            for (i, k) in subset.iter().enumerate() {
                assert_eq!(
                    batch[i],
                    filter.might_contain(&k.to_le_bytes()),
                    "key {k} at index {i} of batch len {probe_len}"
                );
            }
        }
    }

    #[test]
    fn test_batch_bytes_matches_sequential() {
        let mut filter = BloomFilter::new(1000);
        for i in 0..1000u64 {
            filter.insert(format!("key_{i}").as_bytes());
        }
        let owned: Vec<String> = (0..2000).map(|i| format!("key_{i}")).collect();
        let values: Vec<&[u8]> = owned.iter().map(|s| s.as_bytes()).collect();
        let mut batch = vec![false; values.len()];
        filter.might_contain_batch_bytes(&values, &mut batch);
        for (i, v) in values.iter().enumerate() {
            assert_eq!(batch[i], filter.might_contain(v), "value index {i}");
        }
    }

    #[test]
    fn test_serialized_batch_matches_sequential() {
        let mut filter = BloomFilter::new(500);
        for i in 0..500u64 {
            filter.insert(&i.to_le_bytes());
        }
        let serialized = filter.to_bytes();

        let owned: Vec<[u8; 8]> = (0..1000u64).map(|i| i.to_le_bytes()).collect();
        let values: Vec<&[u8]> = owned.iter().map(|v| v.as_slice()).collect();
        let mut batch = vec![false; values.len()];
        might_contain_serialized_batch(&serialized, &values, &mut batch);
        for (i, v) in values.iter().enumerate() {
            assert_eq!(
                batch[i],
                might_contain_serialized(&serialized, v),
                "value index {i}"
            );
        }

        // A header that fails validation answers true for every value,
        // the same conservative answer the sequential probe gives
        let truncated = &serialized[..HEADER_SIZE - 1];
        let mut conservative = vec![false; values.len()];
        might_contain_serialized_batch(truncated, &values, &mut conservative);
        assert!(conservative.iter().all(|&r| r));
    }
}

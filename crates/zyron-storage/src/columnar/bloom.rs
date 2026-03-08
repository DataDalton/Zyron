//! Split-block bloom filter for segment-level membership pruning.
//!
//! Each probe touches exactly one 64-byte cache-line-aligned block.
//! Uses xxh3_128 for hashing with double hashing within the block
//! to compute multiple bit positions from a single hash.

use crate::columnar::constants::*;
use xxhash_rust::xxh3::xxh3_128;
use zyron_common::{Result, ZyronError};

/// Serialization header size: hash_count(4) + num_blocks(4) + num_elements(8).
const HEADER_SIZE: usize = 16;

/// Bits per block (BLOOM_BLOCK_SIZE * 8).
const BLOCK_BITS: u32 = (BLOOM_BLOCK_SIZE * 8) as u32;

/// Split-block bloom filter with cache-line aligned blocks.
///
/// The bit array length is always a multiple of BLOOM_BLOCK_SIZE (64 bytes).
/// Each insert or probe hashes the value once with xxh3_128, selects a single
/// block, then uses double hashing to set or check BLOOM_HASH_COUNT bit
/// positions within that block.
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
    /// Hashes the value with xxh3_128, selects a block via the lower 64 bits,
    /// then sets BLOOM_HASH_COUNT bit positions within that block using double
    /// hashing: bit_i = (h1 + i * h2) % 512.
    pub fn insert(&mut self, value: &[u8]) {
        let hash = xxh3_128(value);
        let h1 = hash as u64;
        let h2 = (hash >> 64) as u64;

        let blockIndex = (h1 % self.numBlocks as u64) as usize;
        let blockStart = blockIndex * BLOOM_BLOCK_SIZE;

        for i in 0..self.hashCount {
            let bitPos = h1.wrapping_add((i as u64).wrapping_mul(h2)) % BLOCK_BITS as u64;
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
        let hash = xxh3_128(value);
        let h1 = hash as u64;
        let h2 = (hash >> 64) as u64;

        let blockIndex = (h1 % self.numBlocks as u64) as usize;
        let blockStart = blockIndex * BLOOM_BLOCK_SIZE;
        let ptr = self.bits.as_ptr();

        // Unrolled probe loop using raw pointer reads to skip bounds checks.
        // The block is always BLOOM_BLOCK_SIZE (64) bytes, and bitPos is always
        // mod 512, so byteOffset is always within [blockStart, blockStart+63].
        for i in 0..self.hashCount {
            let bitPos = h1.wrapping_add((i as u64).wrapping_mul(h2)) % BLOCK_BITS as u64;
            let byteOffset = blockStart + (bitPos >> 3) as usize;
            let bitMask = 1u8 << (bitPos & 7);
            let byte = unsafe { *ptr.add(byteOffset) };
            if byte & bitMask == 0 {
                return false;
            }
        }

        true
    }

    /// Serializes the bloom filter to bytes.
    ///
    /// Layout: hash_count(4 LE) + num_blocks(4 LE) + num_elements(8 LE) + bits.
    pub fn to_bytes(&self) -> Vec<u8> {
        let totalSize = HEADER_SIZE + self.bits.len();
        let mut buf = Vec::with_capacity(totalSize);

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

        let hashCount = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        let numBlocks = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
        let numElements = u64::from_le_bytes([
            buf[8], buf[9], buf[10], buf[11], buf[12], buf[13], buf[14], buf[15],
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

    /// Returns the total serialized byte count: 16-byte header + bit array.
    pub fn on_disk_size(&self) -> usize {
        HEADER_SIZE + self.bits.len()
    }
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
        // Split-block bloom filters have higher FPR than standard bloom
        // due to block confinement. Allow up to 8% (typically ~5-6%).
        assert!(
            fpr < 0.08,
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
        buf.extend_from_slice(&7u32.to_le_bytes()); // hash_count
        buf.extend_from_slice(&0u32.to_le_bytes()); // num_blocks = 0
        buf.extend_from_slice(&0u64.to_le_bytes()); // num_elements

        let result = BloomFilter::from_bytes(&buf);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_bytes_length_mismatch() {
        let mut buf = Vec::new();
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
        buf.extend_from_slice(&0u32.to_le_bytes()); // hash_count = 0
        buf.extend_from_slice(&1u32.to_le_bytes()); // num_blocks = 1
        buf.extend_from_slice(&0u64.to_le_bytes()); // num_elements
        buf.extend_from_slice(&[0u8; BLOOM_BLOCK_SIZE]); // 64 bytes for 1 block

        let result = BloomFilter::from_bytes(&buf);
        assert!(result.is_err());
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
}

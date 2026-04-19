//! Cryptographic and hash functions.
//!
//! SHA-256/384/512 (via sha2 crate), HMAC-SHA256,
//! hash_combine for composite keys, and jump consistent hashing for sharding.

use sha2::{Digest, Sha256, Sha384, Sha512};

/// Computes SHA-256 of the given data. Returns 32 bytes.
pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&result);
    out
}

/// Computes SHA-384 of the given data. Returns 48 bytes.
pub fn sha384(data: &[u8]) -> [u8; 48] {
    let mut hasher = Sha384::new();
    hasher.update(data);
    let result = hasher.finalize();
    let mut out = [0u8; 48];
    out.copy_from_slice(&result);
    out
}

/// Computes SHA-512 of the given data. Returns 64 bytes.
pub fn sha512(data: &[u8]) -> [u8; 64] {
    let mut hasher = Sha512::new();
    hasher.update(data);
    let result = hasher.finalize();
    let mut out = [0u8; 64];
    out.copy_from_slice(&result);
    out
}

/// Computes HMAC-SHA256 of data with the given key. Returns 32 bytes.
/// Implementation follows RFC 2104: HMAC = H((key XOR opad) || H((key XOR ipad) || message))
pub fn hmac_sha256(data: &[u8], key: &[u8]) -> [u8; 32] {
    let block_size = 64; // SHA-256 block size

    // If key is longer than block size, hash it first
    let key_bytes = if key.len() > block_size {
        let h = sha256(key);
        h.to_vec()
    } else {
        key.to_vec()
    };

    // Pad key to block size
    let mut padded_key = vec![0u8; block_size];
    padded_key[..key_bytes.len()].copy_from_slice(&key_bytes);

    // Inner padding (key XOR 0x36)
    let mut inner = Vec::with_capacity(block_size + data.len());
    for &k in &padded_key {
        inner.push(k ^ 0x36);
    }
    inner.extend_from_slice(data);
    let inner_hash = sha256(&inner);

    // Outer padding (key XOR 0x5C)
    let mut outer = Vec::with_capacity(block_size + 32);
    for &k in &padded_key {
        outer.push(k ^ 0x5C);
    }
    outer.extend_from_slice(&inner_hash);
    sha256(&outer)
}

/// Combines two hash values into one using boost::hash_combine style mixing.
/// Produces a deterministic combined hash suitable for composite keys.
pub fn hash_combine(a: u64, b: u64) -> u64 {
    // Uses the golden ratio constant and bit mixing from boost::hash_combine
    let phi: u64 = 0x9E3779B97F4A7C15; // 2^64 / golden ratio
    a ^ (b
        .wrapping_add(phi)
        .wrapping_add(a << 6)
        .wrapping_add(a >> 2))
}

/// Jump consistent hashing (Lamping and Veach, 2014).
/// Maps a key to one of num_buckets buckets with minimal disruption
/// when the number of buckets changes.
pub fn consistent_hash(key: &[u8], num_buckets: u32) -> u32 {
    if num_buckets == 0 {
        return 0;
    }

    // Hash the key to a u64 seed
    let mut seed = zyron_common::hash64(key);

    let mut b: i64 = -1;
    let mut j: i64 = 0;

    while j < num_buckets as i64 {
        b = j;
        // LCG-style PRNG step
        seed = seed.wrapping_mul(2862933555777941757).wrapping_add(1);
        j = ((b.wrapping_add(1) as f64) * ((1i64 << 31) as f64)
            / ((seed >> 33).wrapping_add(1) as f64)) as i64;
    }

    b as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sha256_empty() {
        let hash = sha256(b"");
        let hex: String = hash.iter().map(|b| format!("{:02x}", b)).collect();
        assert_eq!(
            hex,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn test_sha256_hello() {
        let hash = sha256(b"hello");
        let hex: String = hash.iter().map(|b| format!("{:02x}", b)).collect();
        assert_eq!(
            hex,
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
    }

    #[test]
    fn test_sha384_empty() {
        let hash = sha384(b"");
        assert_eq!(hash.len(), 48);
        let hex: String = hash.iter().map(|b| format!("{:02x}", b)).collect();
        assert!(hex.starts_with("38b060a751ac9638"));
    }

    #[test]
    fn test_sha512_empty() {
        let hash = sha512(b"");
        assert_eq!(hash.len(), 64);
        let hex: String = hash.iter().map(|b| format!("{:02x}", b)).collect();
        assert!(hex.starts_with("cf83e1357eefb8bd"));
    }

    #[test]
    fn test_sha256_deterministic() {
        assert_eq!(sha256(b"test"), sha256(b"test"));
    }

    #[test]
    fn test_sha256_different_inputs() {
        assert_ne!(sha256(b"hello"), sha256(b"world"));
    }

    #[test]
    fn test_hmac_sha256_known() {
        // RFC 4231 Test Case 1
        let key = [0x0b; 20];
        let data = b"Hi There";
        let mac = hmac_sha256(data, &key);
        let hex: String = mac.iter().map(|b| format!("{:02x}", b)).collect();
        assert_eq!(
            hex,
            "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
        );
    }

    #[test]
    fn test_hmac_sha256_short_key() {
        let mac = hmac_sha256(b"message", b"key");
        assert_eq!(mac.len(), 32);
    }

    #[test]
    fn test_hmac_sha256_long_key() {
        // Key longer than 64 bytes should be hashed first
        let long_key = vec![0xAA; 131];
        let mac = hmac_sha256(
            b"Test Using Larger Than Block-Size Key - Hash Key First",
            &long_key,
        );
        assert_eq!(mac.len(), 32);
    }

    #[test]
    fn test_hmac_sha256_deterministic() {
        let mac1 = hmac_sha256(b"data", b"key");
        let mac2 = hmac_sha256(b"data", b"key");
        assert_eq!(mac1, mac2);
    }

    #[test]
    fn test_hmac_sha256_different_keys() {
        let mac1 = hmac_sha256(b"data", b"key1");
        let mac2 = hmac_sha256(b"data", b"key2");
        assert_ne!(mac1, mac2);
    }

    #[test]
    fn test_hash_combine_deterministic() {
        let h = hash_combine(42, 99);
        assert_eq!(h, hash_combine(42, 99));
    }

    #[test]
    fn test_hash_combine_order_matters() {
        assert_ne!(hash_combine(1, 2), hash_combine(2, 1));
    }

    #[test]
    fn test_hash_combine_zero() {
        // Combining with zero should still produce a non-trivial hash
        let h = hash_combine(0, 42);
        assert_ne!(h, 0);
        assert_ne!(h, 42);
    }

    #[test]
    fn test_consistent_hash_range() {
        for i in 0..100u64 {
            let bucket = consistent_hash(&i.to_le_bytes(), 10);
            assert!(bucket < 10);
        }
    }

    #[test]
    fn test_consistent_hash_deterministic() {
        let a = consistent_hash(b"key", 100);
        let b = consistent_hash(b"key", 100);
        assert_eq!(a, b);
    }

    #[test]
    fn test_consistent_hash_single_bucket() {
        assert_eq!(consistent_hash(b"anything", 1), 0);
    }

    #[test]
    fn test_consistent_hash_zero_buckets() {
        assert_eq!(consistent_hash(b"key", 0), 0);
    }

    #[test]
    fn test_consistent_hash_distribution() {
        // Verify rough uniformity: 1000 keys across 10 buckets
        let mut counts = [0u32; 10];
        for i in 0..1000u64 {
            let bucket = consistent_hash(&i.to_le_bytes(), 10);
            counts[bucket as usize] += 1;
        }
        // Each bucket should have roughly 100 keys (allow 50% deviation)
        for count in counts {
            assert!(count > 50, "Bucket too empty: {}", count);
            assert!(count < 200, "Bucket too full: {}", count);
        }
    }

    #[test]
    fn test_consistent_hash_minimal_disruption() {
        // Adding a bucket should only move ~1/n of keys
        let mut moves = 0;
        for i in 0..1000u64 {
            let old = consistent_hash(&i.to_le_bytes(), 10);
            let new = consistent_hash(&i.to_le_bytes(), 11);
            if old != new {
                moves += 1;
            }
        }
        // Expect roughly 1000/11 = ~91 moves. Allow generous margin.
        assert!(moves < 200, "Too many keys moved: {}", moves);
    }
}

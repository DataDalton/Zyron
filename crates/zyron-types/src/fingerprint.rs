//! Document fingerprinting: MinHash (LSH) and SimHash (Charikar).
//!
//! MinHash: preserves Jaccard similarity between token sets via hash signatures.
//! SimHash: 64-bit Charikar fingerprint for fast near-duplicate detection
//! via Hamming distance.

use crate::encoding::xxhash64;
use zyron_common::{Result, ZyronError};

// ---------------------------------------------------------------------------
// MinHash
// ---------------------------------------------------------------------------

/// Computes a MinHash signature of the given tokens using num_hashes hash functions.
/// Uses the permutation approach: h_i(x) = (a_i * h(x) + b_i) mod p, where
/// p is a large prime and (a_i, b_i) are random per-slot constants.
pub fn minhash_signature(tokens: &[&str], num_hashes: u32) -> Vec<u64> {
    if num_hashes == 0 {
        return Vec::new();
    }

    let mut signature = vec![u64::MAX; num_hashes as usize];
    // Large prime near 2^61 - 1 for Mersenne prime arithmetic
    const PRIME: u64 = (1u64 << 61) - 1;

    // Pre-compute (a, b) coefficients for each hash slot deterministically
    let mut coefficients: Vec<(u64, u64)> = Vec::with_capacity(num_hashes as usize);
    for i in 0..num_hashes as u64 {
        // Use xxhash to generate pseudorandom but deterministic coefficients
        let a = xxhash64(&(i * 2).to_le_bytes()) | 1; // odd to ensure coprime
        let b = xxhash64(&(i * 2 + 1).to_le_bytes());
        coefficients.push((a % PRIME, b % PRIME));
    }

    for token in tokens {
        let h = xxhash64(token.as_bytes()) % PRIME;
        for (i, &(a, b)) in coefficients.iter().enumerate() {
            // Mod-p multiplication with overflow protection
            let h_i = mul_mod_p(a, h, PRIME).wrapping_add(b) % PRIME;
            if h_i < signature[i] {
                signature[i] = h_i;
            }
        }
    }

    signature
}

#[inline]
fn mul_mod_p(a: u64, b: u64, p: u64) -> u64 {
    // Use u128 for overflow-free multiplication
    ((a as u128) * (b as u128) % (p as u128)) as u64
}

/// Approximates Jaccard similarity between two MinHash signatures.
pub fn minhash_similarity(a: &[u64], b: &[u64]) -> Result<f64> {
    if a.len() != b.len() {
        return Err(ZyronError::ExecutionError(format!(
            "MinHash signature length mismatch: {} vs {}",
            a.len(),
            b.len()
        )));
    }
    if a.is_empty() {
        return Ok(0.0);
    }
    let matches = a.iter().zip(b).filter(|(x, y)| x == y).count();
    Ok(matches as f64 / a.len() as f64)
}

/// Encodes a MinHash signature as bytes (little-endian u64 array).
pub fn minhash_encode(signature: &[u64]) -> Vec<u8> {
    let mut result = Vec::with_capacity(signature.len() * 8);
    for &val in signature {
        result.extend_from_slice(&val.to_le_bytes());
    }
    result
}

/// Decodes bytes into a MinHash signature.
pub fn minhash_decode(bytes: &[u8]) -> Result<Vec<u64>> {
    if bytes.len() % 8 != 0 {
        return Err(ZyronError::ExecutionError(
            "MinHash bytes must be multiple of 8".into(),
        ));
    }
    let mut result = Vec::with_capacity(bytes.len() / 8);
    for chunk in bytes.chunks_exact(8) {
        let val = u64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]);
        result.push(val);
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// SimHash (Charikar)
// ---------------------------------------------------------------------------

/// Computes a 64-bit SimHash fingerprint of a string by tokenizing on whitespace.
pub fn simhash(text: &str) -> u64 {
    let tokens: Vec<&str> = text.split_whitespace().collect();
    simhash_tokens(&tokens)
}

/// Computes a 64-bit SimHash fingerprint from pre-tokenized input.
pub fn simhash_tokens(tokens: &[&str]) -> u64 {
    if tokens.is_empty() {
        return 0;
    }

    let mut vector = [0i32; 64];

    for token in tokens {
        let h = xxhash64(token.as_bytes());
        for bit in 0..64 {
            if (h >> bit) & 1 == 1 {
                vector[bit] += 1;
            } else {
                vector[bit] -= 1;
            }
        }
    }

    let mut fingerprint = 0u64;
    for (bit, &count) in vector.iter().enumerate() {
        if count > 0 {
            fingerprint |= 1u64 << bit;
        }
    }
    fingerprint
}

/// Computes the Hamming distance between two SimHash fingerprints.
pub fn simhash_distance(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// Returns true if two SimHash fingerprints are within `threshold` Hamming distance.
pub fn simhash_similar(a: u64, b: u64, threshold: u32) -> bool {
    simhash_distance(a, b) <= threshold
}

// ---------------------------------------------------------------------------
// k-Shingles
// ---------------------------------------------------------------------------

/// Generates k-character shingles from text for use as MinHash input tokens.
pub fn shingle(text: &str, k: usize) -> Vec<String> {
    if k == 0 {
        return Vec::new();
    }
    let chars: Vec<char> = text.chars().collect();
    if chars.len() < k {
        return Vec::new();
    }
    (0..=chars.len() - k)
        .map(|i| chars[i..i + k].iter().collect())
        .collect()
}

/// Generates k-word shingles from text.
pub fn word_shingle(text: &str, k: usize) -> Vec<String> {
    if k == 0 {
        return Vec::new();
    }
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < k {
        return Vec::new();
    }
    (0..=words.len() - k)
        .map(|i| words[i..i + k].join(" "))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minhash_length() {
        let tokens = vec!["the", "quick", "brown", "fox"];
        let sig = minhash_signature(&tokens, 128);
        assert_eq!(sig.len(), 128);
    }

    #[test]
    fn test_minhash_deterministic() {
        let tokens = vec!["hello", "world"];
        let s1 = minhash_signature(&tokens, 64);
        let s2 = minhash_signature(&tokens, 64);
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_minhash_similarity_identical() {
        let tokens = vec!["a", "b", "c"];
        let sig = minhash_signature(&tokens, 128);
        let sim = minhash_similarity(&sig, &sig).unwrap();
        assert_eq!(sim, 1.0);
    }

    #[test]
    fn test_minhash_similarity_disjoint() {
        let a = minhash_signature(&["a", "b", "c"], 128);
        let b = minhash_signature(&["x", "y", "z"], 128);
        let sim = minhash_similarity(&a, &b).unwrap();
        assert!(
            sim < 0.1,
            "Disjoint sets should have low similarity: {}",
            sim
        );
    }

    #[test]
    fn test_minhash_similarity_overlap() {
        // Sets with ~50% overlap
        let a = minhash_signature(&["a", "b", "c", "d"], 256);
        let b = minhash_signature(&["c", "d", "e", "f"], 256);
        let sim = minhash_similarity(&a, &b).unwrap();
        // True Jaccard = 2/6 = 0.333
        assert!((sim - 0.333).abs() < 0.15, "Expected ~0.333, got {}", sim);
    }

    #[test]
    fn test_minhash_length_mismatch() {
        let a = vec![1u64, 2, 3];
        let b = vec![1u64, 2];
        assert!(minhash_similarity(&a, &b).is_err());
    }

    #[test]
    fn test_minhash_encode_decode() {
        let sig = vec![1u64, 2, 3, 4, 5];
        let bytes = minhash_encode(&sig);
        let decoded = minhash_decode(&bytes).unwrap();
        assert_eq!(sig, decoded);
    }

    #[test]
    fn test_minhash_decode_invalid() {
        assert!(minhash_decode(&[1, 2, 3]).is_err());
    }

    #[test]
    fn test_simhash_deterministic() {
        let h1 = simhash("hello world");
        let h2 = simhash("hello world");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_simhash_different() {
        let h1 = simhash("hello world");
        let h2 = simhash("completely different text");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_simhash_similar_documents() {
        let doc1 = simhash("The quick brown fox jumps over the lazy dog");
        let doc2 = simhash("The quick brown fox jumps over the sleeping dog");
        let dist = simhash_distance(doc1, doc2);
        assert!(
            dist < 20,
            "Similar documents should have low Hamming distance: {}",
            dist
        );
    }

    #[test]
    fn test_simhash_distance_identical() {
        let h = simhash("hello world");
        assert_eq!(simhash_distance(h, h), 0);
    }

    #[test]
    fn test_simhash_distance_different() {
        let h1 = simhash("alpha beta gamma");
        let h2 = simhash("xyz qrs tuv");
        let dist = simhash_distance(h1, h2);
        assert!(dist > 10);
    }

    #[test]
    fn test_simhash_similar_threshold() {
        let h1 = simhash("hello world foo bar");
        let h2 = simhash("hello world foo baz");
        assert!(simhash_similar(h1, h2, 20));
    }

    #[test]
    fn test_shingle() {
        let s = shingle("hello", 2);
        assert_eq!(s, vec!["he", "el", "ll", "lo"]);
    }

    #[test]
    fn test_shingle_too_short() {
        assert!(shingle("hi", 5).is_empty());
    }

    #[test]
    fn test_shingle_zero_k() {
        assert!(shingle("hello", 0).is_empty());
    }

    #[test]
    fn test_word_shingle() {
        let s = word_shingle("the quick brown fox", 2);
        assert_eq!(s, vec!["the quick", "quick brown", "brown fox"]);
    }

    #[test]
    fn test_simhash_empty() {
        assert_eq!(simhash(""), 0);
    }

    #[test]
    fn test_minhash_empty() {
        let sig = minhash_signature(&[], 16);
        // All positions remain u64::MAX (no tokens seen)
        assert_eq!(sig.len(), 16);
        assert!(sig.iter().all(|&v| v == u64::MAX));
    }
}

//! Shared primitives for batch column hashing.
//!
//! The executor and streaming crates each hash entire column batches into
//! per-row u64 arrays: typed dispatch once per column, then a tight
//! auto-vectorizable loop of [`hash_int`] or [`super::helpers::fnv1a_64`]
//! per row, combined across columns with [`super::helpers::hash_combine`]
//! and finished with [`crate::mix_finalize_3round`]. The per-value
//! primitives those loops share live here as the single canonical copies.
//!
//! The dispatch loops themselves stay in their own crates: each operates
//! on that crate's column representation (zyron-executor::column::Column,
//! zyron-streaming::column::StreamColumn) and on that crate's null
//! convention, and those types cannot move below their crates. What was
//! duplicated, and is now consolidated, is the arithmetic the loops apply
//! per value

/// Golden ratio constant shared by fibonacci hashing and hash_combine
pub const HASH_GOLDEN: u64 = 0x9e3779b97f4a7c15;

/// Fibonacci hash for a single integer: multiply by the golden ratio
/// constant, then fold high bits into low bits for bucket distribution.
///
/// A bijection on u64, distinct inputs produce distinct outputs, so hash
/// equality implies key equality with zero false positives. The fused join
/// paths rely on that. Signed values hash via their two's complement bits,
/// `hash_int(v as u64)`
#[inline(always)]
pub fn hash_int(v: u64) -> u64 {
    let h = v.wrapping_mul(HASH_GOLDEN);
    h ^ (h >> 32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_int_is_bijective_on_samples() {
        // The inverse of multiplication by an odd constant exists mod 2^64
        // and h ^ (h >> 32) is self-inverse composed twice over the high
        // half, spot-check no collisions across a dense sample
        let mut seen = std::collections::HashSet::new();
        for v in 0..100_000u64 {
            assert!(seen.insert(hash_int(v)), "collision at {v}");
        }
    }

    #[test]
    fn hash_int_matches_duplicated_copies() {
        // Transcription of the body the executor and streaming copies shared
        fn old(v: u64) -> u64 {
            let h = v.wrapping_mul(0x9e3779b97f4a7c15);
            h ^ (h >> 32)
        }
        for v in [0u64, 1, 42, i64::MAX as u64, u64::MAX] {
            assert_eq!(hash_int(v), old(v));
        }
    }
}

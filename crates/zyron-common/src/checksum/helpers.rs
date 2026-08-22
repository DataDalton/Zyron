//! Single canonical copies of the small hand-rolled hash helpers.
//!
//! - [`fnv1a_64`] serves zyron-types::checksum (the SQL function),
//!   zyron-streaming, zyron-executor::compute and zyron-lake::predicate
//! - [`hash_combine`] serves zyron-types::crypto (the SQL function),
//!   zyron-streaming and zyron-executor::compute
//! - [`hash_fold`] serves the wire statement cache, plan cache shard
//!   index, auto-param type-state hash and session identity hash
//! - [`splitmix64`] is the pure one-shot form of the prng seed expander,
//!   serving zyron-lake::transaction_log's partition bloom
//!
//! Each function is pinned by a transcription test below, so persisted
//! values that depend on it (lake commit headers, partition blooms, SQL
//! function output) cannot silently change

/// FNV-1a 64-bit offset basis. The only definition in the workspace
pub const FNV_OFFSET: u64 = 0xcbf29ce484222325;

/// FNV-1a 64-bit prime. The only definition in the workspace
pub const FNV_PRIME: u64 = 0x100000001b3;

/// FNV-1a 64-bit hash, the canonical byte-by-byte multiply-xor form.
///
/// This is the standard algorithm and its output is externally observable:
/// the SQL function fnv1a_64() dispatches here and users compare its result
/// against other FNV-1a implementations, so the recurrence must stay exact.
///
/// There is no vector form of this function and one cannot exist with equal
/// output. Each step multiplies the running state, and integer
/// multiplication does not distribute over the xor that injects the next
/// byte, so the byte-serial xor-multiply chain cannot be split into lanes
/// that recombine to the same value. Reformulating the recurrence as
/// independent per-byte products still leaves an equally long serial chain
/// through the low byte of the state, so the dependency floor of roughly
/// one multiply latency per byte is a property of the algorithm, not of
/// this implementation. `fnv1a_64_matches_spec` below pins the recurrence
/// so an attempted lane decomposition fails there instead of silently
/// changing SQL-visible output
#[inline]
pub fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// Boost-style seed combiner: folds `value` into `seed` with the golden
/// ratio constant and two shifted copies of the seed.
///
/// Order matters, hash_combine(a, b) != hash_combine(b, a), which composite
/// key hashing relies on
#[inline(always)]
pub fn hash_combine(seed: u64, value: u64) -> u64 {
    seed ^ (value
        .wrapping_add(0x9e3779b97f4a7c15)
        .wrapping_add(seed << 6)
        .wrapping_add(seed >> 2))
}

/// Rotate-xor fold: rotates the running seed and xors in the next value.
///
/// The cheapest order-sensitive fold for combining values that are already
/// well distributed (existing hashes, ids, discriminants). Not a mixer, a
/// low-entropy `x` stays low-entropy in the output, so feed it hashes
#[inline(always)]
pub fn hash_fold(seed: u64, x: u64) -> u64 {
    seed.rotate_left(5) ^ x
}

/// splitmix64 as a pure function of its input.
///
/// Same constants and rounds as [`crate::prng::splitMix64`], which advances
/// a &mut state for seed expansion. This form hashes one u64 to one u64,
/// which is what hash uses (partition-id blooms) want
#[inline]
pub fn splitmix64(x: u64) -> u64 {
    let mut state = x;
    crate::prng::splitMix64(&mut state)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins fnv1a_64 to an independent transcription of the FNV-1a
    /// specification. The SQL surface exposes this hash, so any deviation,
    /// including a lane-decomposed reimplementation, is user-visible
    #[test]
    fn fnv1a_64_matches_spec() {
        fn spec(bytes: &[u8]) -> u64 {
            let mut h: u64 = 0xcbf29ce484222325;
            for &b in bytes {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
            h
        }
        assert_eq!(fnv1a_64(b""), FNV_OFFSET);
        // Known vector: fnv1a_64("a") = (offset ^ 0x61) * prime
        assert_eq!(fnv1a_64(b"a"), (FNV_OFFSET ^ 0x61).wrapping_mul(FNV_PRIME));
        let mut state = 0x12345678u64;
        for len in [0usize, 1, 7, 8, 31, 32, 33, 100, 1000] {
            let data: Vec<u8> = (0..len)
                .map(|_| {
                    state = splitmix64(state);
                    state as u8
                })
                .collect();
            assert_eq!(fnv1a_64(&data), spec(&data), "len {len}");
        }
    }

    #[test]
    fn hash_combine_matches_duplicated_copies() {
        // Transcription of the body the three deleted copies shared
        fn old(seed: u64, value: u64) -> u64 {
            let phi: u64 = 0x9e3779b97f4a7c15;
            seed ^ (value
                .wrapping_add(phi)
                .wrapping_add(seed << 6)
                .wrapping_add(seed >> 2))
        }
        for (a, b) in [(0u64, 0u64), (1, 2), (42, 99), (u64::MAX, 1), (7, u64::MAX)] {
            assert_eq!(hash_combine(a, b), old(a, b));
        }
        assert_ne!(hash_combine(1, 2), hash_combine(2, 1));
    }

    #[test]
    fn hash_fold_is_rotate_xor() {
        assert_eq!(hash_fold(1, 0), 1u64.rotate_left(5));
        assert_eq!(hash_fold(0, 7), 7);
        let seed = 0xdead_beef_cafe_f00du64;
        assert_eq!(hash_fold(seed, 3), seed.rotate_left(5) ^ 3);
    }

    /// The one-shot form must equal the copy zyron-lake::transaction_log
    /// carried, because the partition bloom it feeds is persisted in commit
    /// headers
    #[test]
    fn splitmix64_matches_lake_copy() {
        fn lake_copy(mut x: u64) -> u64 {
            x = x.wrapping_add(0x9E3779B97F4A7C15);
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
            x ^ (x >> 31)
        }
        for x in [0u64, 1, 42, 0x9E3779B97F4A7C15, u64::MAX] {
            assert_eq!(splitmix64(x), lake_copy(x));
        }
    }
}

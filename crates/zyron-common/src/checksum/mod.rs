//! Central hash and checksum primitive for ZyronDB.
//!
//! AES-round-based mixing across 8 parallel 128-bit lanes. Used everywhere
//! the database needs a non-cryptographic hash:
//!
//! - File / record integrity checksums (WAL records, columnar files,
//!   B-tree checkpoints, version records, CDC records, backup files,
//!   spatial index snapshots).
//! - Hash-map and Bloom filter key hashing (intent locks, columnar bloom,
//!   probabilistic data structures, fingerprint hashes, hash joins,
//!   streaming state hashes, query pattern hashes).
//!
//! Three output widths are exposed: `hash32` (integrity-style 32-bit),
//! `hash64` (hash-map keys), `hash128` (Bloom double-hashing). All are
//! derived from the same internal 128-bit state, so the algorithm is
//! defined once.
//!
//! Hardware acceleration is available where the host CPU supports AES
//! instructions (AES-NI on x86, AES extension on ARMv8). When neither is
//! present, a pure-software AES round implementation runs as a fallback;
//! output is bit-identical to every other tier.
//!
//! Streaming use cases (incremental hashing of WAL records as fields are
//! emitted from registers) use the `Hasher` type. `Hasher::finish_phase()`
//! mixes a phase separator into all lanes so structural boundaries (e.g.
//! WAL header vs payload) are detected separately from byte-level changes.
//!
//! See `checksum/spec.rs` for the algorithm constants and `checksum.md`
//! (alongside this file) for the byte-level reference specification.

pub mod scalar;
pub mod spec;

#[cfg(target_arch = "x86_64")]
pub mod x86_aes;

#[cfg(target_arch = "aarch64")]
pub mod aarch64_aes;

pub use spec::ALGORITHM_VERSION;

use scalar::{Lanes, hash_scalar};
use spec::{CHUNK_BYTES, SMALL_INPUT_THRESHOLD};
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Tier dispatch
// ---------------------------------------------------------------------------

/// Identifier for the active hash implementation, exposed as a metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    Scalar,
    X86Aes,
    AArch64Aes,
}

impl Tier {
    pub fn name(&self) -> &'static str {
        match self {
            Tier::Scalar => "scalar",
            Tier::X86Aes => "x86_aes",
            Tier::AArch64Aes => "aarch64_aes",
        }
    }
}

#[derive(Clone, Copy)]
struct Dispatch {
    tier: Tier,
    one_shot: fn(&[u8], u128) -> u128,
    absorb_chunk: fn(&mut Lanes, &[u8; CHUNK_BYTES]),
    finish_phase: fn(&mut Lanes),
    finalize: fn(&Lanes) -> u128,
    small_input: fn(&[u8], u128) -> u128,
}

static DISPATCH: OnceLock<Dispatch> = OnceLock::new();

fn dispatch() -> &'static Dispatch {
    DISPATCH.get_or_init(detect_tier)
}

fn detect_tier() -> Dispatch {
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    {
        if x86_aes::is_available() {
            return Dispatch {
                tier: Tier::X86Aes,
                one_shot: |bytes, seed| unsafe { x86_aes::hash_x86_aes(bytes, seed) },
                absorb_chunk: |lanes, chunk| unsafe { x86_aes::absorb_chunk(lanes, chunk) },
                finish_phase: |lanes| unsafe { x86_aes::finish_phase(lanes) },
                finalize: |lanes| unsafe { x86_aes::finalize(lanes) },
                small_input: |bytes, seed| unsafe { x86_aes::small_input(bytes, seed) },
            };
        }
    }
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    {
        if aarch64_aes::is_available() {
            return Dispatch {
                tier: Tier::AArch64Aes,
                one_shot: |bytes, seed| unsafe { aarch64_aes::hash_aarch64_aes(bytes, seed) },
                absorb_chunk: |lanes, chunk| unsafe { aarch64_aes::absorb_chunk(lanes, chunk) },
                finish_phase: |lanes| unsafe { aarch64_aes::finish_phase(lanes) },
                finalize: |lanes| unsafe { aarch64_aes::finalize(lanes) },
                small_input: |bytes, seed| unsafe { aarch64_aes::small_input(bytes, seed) },
            };
        }
    }
    Dispatch {
        tier: Tier::Scalar,
        one_shot: hash_scalar,
        absorb_chunk: |lanes, chunk| lanes.absorb_chunk(chunk),
        finish_phase: |lanes| lanes.finish_phase(),
        finalize: |lanes| lanes.finalize(),
        small_input: scalar::small_input,
    }
}

/// Returns which implementation tier the dispatcher selected at startup.
/// Useful as a metric so operators can verify hardware acceleration is
/// active in production.
pub fn active_tier() -> Tier {
    dispatch().tier
}

// ---------------------------------------------------------------------------
// One-shot hash API
// ---------------------------------------------------------------------------

/// Hashes the input with seed 0 and returns a 32-bit checksum suitable for
/// integrity verification (file, record, page checksums).
#[inline]
pub fn hash32(bytes: &[u8]) -> u32 {
    fold_to_u32((dispatch().one_shot)(bytes, 0))
}

/// Hashes the input with seed 0 and returns 64 bits suitable for hash-map
/// or Bloom filter key hashing.
#[inline]
pub fn hash64(bytes: &[u8]) -> u64 {
    fold_to_u64((dispatch().one_shot)(bytes, 0))
}

/// Hashes the input with seed 0 and returns the full 128-bit state. Used
/// by Bloom filter double-hashing (split into two 64-bit values).
#[inline]
pub fn hash128(bytes: &[u8]) -> u128 {
    (dispatch().one_shot)(bytes, 0)
}

/// Seeded variant of [`hash32`]. The seed mixes into every lane and the
/// finalization, so two hashes with different seeds are statistically
/// independent.
#[inline]
pub fn hash32_seeded(bytes: &[u8], seed: u64) -> u32 {
    fold_to_u32((dispatch().one_shot)(bytes, seed as u128))
}

/// Seeded variant of [`hash64`].
#[inline]
pub fn hash64_seeded(bytes: &[u8], seed: u64) -> u64 {
    fold_to_u64((dispatch().one_shot)(bytes, seed as u128))
}

/// Seeded variant of [`hash128`]. Accepts a 128-bit seed for callers that
/// want to fully randomize the output space (e.g. randomized hashing).
#[inline]
pub fn hash128_seeded(bytes: &[u8], seed: u128) -> u128 {
    (dispatch().one_shot)(bytes, seed)
}

/// Folds 128-bit state down to 64 bits via xor of high and low halves.
#[inline(always)]
fn fold_to_u64(state: u128) -> u64 {
    (state as u64) ^ ((state >> 64) as u64)
}

/// Folds 128-bit state down to 32 bits via xor of all four 32-bit halves.
#[inline(always)]
fn fold_to_u32(state: u128) -> u32 {
    let lo = state as u64;
    let hi = (state >> 64) as u64;
    (lo as u32) ^ ((lo >> 32) as u32) ^ (hi as u32) ^ ((hi >> 32) as u32)
}

// ---------------------------------------------------------------------------
// Streaming Hasher
// ---------------------------------------------------------------------------

/// Streaming hasher. Produces the same output as the one-shot API for the
/// concatenation of all `update` calls, regardless of how the input was
/// chunked.
///
/// Reusable via [`Hasher::reset`]; no allocation in the hot path. Aligned
/// to a cache line so concurrent use across threads (one hasher per thread)
/// avoids false sharing.
/// Streaming hasher for the 8-lane AES pipeline. Used for file checksums,
/// record integrity over KB+ of data, and hashmap/Bloom keys. Per-call
/// setup cost is fixed (~300-500 ns) and amortizes to nothing on large
/// buffers. Sub-KB hot-path workloads should use a domain-specific
/// primitive instead — see `zyron-wal::checksum::WalHasher` for the WAL
/// shape.
#[repr(align(64))]
pub struct Hasher {
    lanes: Lanes,
    buf: [u8; CHUNK_BYTES],
    buf_len: usize,
    total_len: u64,
    seed: u128,
}

impl Hasher {
    /// Creates a new hasher with seed 0.
    #[inline]
    pub fn new() -> Self {
        Self::with_seed(0)
    }

    /// Creates a new hasher with the given 128-bit seed.
    #[inline]
    pub fn with_seed(seed: u128) -> Self {
        Self {
            lanes: Lanes::init(seed),
            buf: [0u8; CHUNK_BYTES],
            buf_len: 0,
            total_len: 0,
            seed,
        }
    }

    /// Adds bytes to the running hash. Cheap to call with arbitrary chunk
    /// sizes; bytes are buffered up to a chunk boundary then absorbed via
    /// the active SIMD tier.
    pub fn update(&mut self, mut bytes: &[u8]) {
        self.total_len = self.total_len.wrapping_add(bytes.len() as u64);
        let d = dispatch();

        if self.buf_len > 0 {
            let need = CHUNK_BYTES - self.buf_len;
            if bytes.len() < need {
                self.buf[self.buf_len..self.buf_len + bytes.len()].copy_from_slice(bytes);
                self.buf_len += bytes.len();
                return;
            }
            self.buf[self.buf_len..].copy_from_slice(&bytes[..need]);
            let buf_copy = self.buf;
            (d.absorb_chunk)(&mut self.lanes, &buf_copy);
            self.buf_len = 0;
            bytes = &bytes[need..];
        }

        while bytes.len() >= CHUNK_BYTES {
            let chunk: &[u8; CHUNK_BYTES] = bytes[..CHUNK_BYTES]
                .try_into()
                .expect("slice is exactly CHUNK_BYTES");
            (d.absorb_chunk)(&mut self.lanes, chunk);
            bytes = &bytes[CHUNK_BYTES..];
        }

        if !bytes.is_empty() {
            self.buf[..bytes.len()].copy_from_slice(bytes);
            self.buf_len = bytes.len();
        }
    }

    /// Mixes a phase separator across all 8 lanes. Lets callers detect
    /// data that crosses a logical boundary differently than expected.
    /// Any buffered bytes are flushed first so the separator applies
    /// cleanly between phases.
    pub fn finish_phase(&mut self) {
        let d = dispatch();
        if self.buf_len > 0 {
            let mut tail = [0u8; CHUNK_BYTES];
            tail[..self.buf_len].copy_from_slice(&self.buf[..self.buf_len]);
            let len_bytes = self.total_len.to_le_bytes();
            for j in 0..8 {
                tail[CHUNK_BYTES - 8 + j] ^= len_bytes[j];
            }
            (d.absorb_chunk)(&mut self.lanes, &tail);
            self.buf_len = 0;
        }
        (d.finish_phase)(&mut self.lanes);
    }

    /// Consumes the hasher and returns the 128-bit state.
    pub fn finish128(mut self) -> u128 {
        let d = dispatch();
        if self.total_len as usize <= SMALL_INPUT_THRESHOLD
            && self.total_len as usize == self.buf_len
        {
            return (d.small_input)(&self.buf[..self.buf_len], self.seed);
        }
        self.flush_tail_dispatched(d);
        (d.finalize)(&self.lanes)
    }

    /// Consumes the hasher and returns 64 bits.
    #[inline]
    pub fn finish64(self) -> u64 {
        fold_to_u64(self.finish128())
    }

    /// Consumes the hasher and returns 32 bits.
    #[inline]
    pub fn finish32(self) -> u32 {
        fold_to_u32(self.finish128())
    }

    /// Returns the current 128-bit state without consuming.
    pub fn peek128(&self) -> u128 {
        let d = dispatch();
        if self.total_len as usize <= SMALL_INPUT_THRESHOLD
            && self.total_len as usize == self.buf_len
        {
            return (d.small_input)(&self.buf[..self.buf_len], self.seed);
        }
        let mut copy = Self {
            lanes: self.lanes.clone(),
            buf: self.buf,
            buf_len: self.buf_len,
            total_len: self.total_len,
            seed: self.seed,
        };
        copy.flush_tail_dispatched(d);
        (d.finalize)(&copy.lanes)
    }

    /// Resets the hasher to its initial state. The seed is preserved.
    pub fn reset(&mut self) {
        self.lanes = Lanes::init(self.seed);
        self.buf_len = 0;
        self.total_len = 0;
    }

    /// Returns the seed this hasher was constructed with.
    #[inline]
    pub fn seed(&self) -> u128 {
        self.seed
    }

    /// Bytes seen so far.
    #[inline]
    pub fn bytes_consumed(&self) -> u64 {
        self.total_len
    }

    /// Internal: emit the final tail chunk with length mixed in, via the
    /// active SIMD tier.
    #[inline]
    fn flush_tail_dispatched(&mut self, d: &Dispatch) {
        let mut tail = [0u8; CHUNK_BYTES];
        tail[..self.buf_len].copy_from_slice(&self.buf[..self.buf_len]);
        let len_bytes = self.total_len.to_le_bytes();
        for j in 0..8 {
            tail[CHUNK_BYTES - 8 + j] ^= len_bytes[j];
        }
        (d.absorb_chunk)(&mut self.lanes, &tail);
        self.buf_len = 0;
    }
}

impl Default for Hasher {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for Hasher {
    fn clone(&self) -> Self {
        Self {
            lanes: self.lanes.clone(),
            buf: self.buf,
            buf_len: self.buf_len,
            total_len: self.total_len,
            seed: self.seed,
        }
    }
}

// ---------------------------------------------------------------------------
// std::hash integration: lets the central hasher be used with HashMap.
// ---------------------------------------------------------------------------

impl std::hash::Hasher for Hasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        self.update(bytes);
    }

    /// Returns the 64-bit fold without consuming. Std's Hasher trait
    /// requires `&self` semantics here so we use the cloning peek path.
    fn finish(&self) -> u64 {
        fold_to_u64(self.peek128())
    }
}

/// `BuildHasher` adapter so `HashMap<K, V, ZyBuildHasher>` works.
#[derive(Clone, Copy, Default)]
pub struct ZyBuildHasher;

impl std::hash::BuildHasher for ZyBuildHasher {
    type Hasher = Hasher;

    #[inline]
    fn build_hasher(&self) -> Hasher {
        Hasher::new()
    }
}

/// Seeded `BuildHasher` for callers that want randomized hash-map hashing.
#[derive(Clone, Copy)]
pub struct ZyBuildHasherSeeded {
    seed: u128,
}

impl ZyBuildHasherSeeded {
    /// Creates a `BuildHasher` that constructs hashers with the given seed.
    #[inline]
    pub const fn new(seed: u128) -> Self {
        Self { seed }
    }
}

impl std::hash::BuildHasher for ZyBuildHasherSeeded {
    type Hasher = Hasher;

    #[inline]
    fn build_hasher(&self) -> Hasher {
        Hasher::with_seed(self.seed)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ----- One-shot API -----

    #[test]
    fn hash32_deterministic() {
        assert_eq!(hash32(b"hello"), hash32(b"hello"));
    }

    #[test]
    fn hash32_seed_changes_output() {
        assert_ne!(hash32_seeded(b"x", 1), hash32_seeded(b"x", 2));
    }

    #[test]
    fn output_widths_are_consistent_folds() {
        let bytes = b"the quick brown fox jumps over the lazy dog";
        let h128 = hash128(bytes);
        let h64 = hash64(bytes);
        let h32 = hash32(bytes);
        assert_eq!(h64, fold_to_u64(h128));
        assert_eq!(h32, fold_to_u32(h128));
    }

    #[test]
    fn empty_inputs_are_deterministic_and_nonzero() {
        let a = hash32(b"");
        let b = hash32(b"");
        assert_eq!(a, b);
        assert_ne!(a, 0, "empty hash should not be zero");
    }

    // ----- Streaming Hasher -----

    #[test]
    fn streaming_matches_oneshot_byte_by_byte() {
        let data: Vec<u8> = (0..500).map(|i| (i * 31) as u8).collect();
        let one = hash128(&data);

        let mut h = Hasher::new();
        for b in &data {
            h.update(std::slice::from_ref(b));
        }
        assert_eq!(h.finish128(), one);
    }

    #[test]
    fn streaming_matches_oneshot_arbitrary_chunks() {
        let data: Vec<u8> = (0..1000).map(|i| (i * 13 + 7) as u8).collect();
        let one = hash128(&data);

        let chunk_sizes = [1usize, 7, 31, 64, 128, 129, 256, 333];
        for &cs in &chunk_sizes {
            let mut h = Hasher::new();
            for chunk in data.chunks(cs) {
                h.update(chunk);
            }
            assert_eq!(h.finish128(), one, "chunk size {}", cs);
        }
    }

    #[test]
    fn streaming_seed_propagates() {
        let data = b"some test data";
        let one = hash128_seeded(data, 0xdeadbeef_cafebabe);
        let mut h = Hasher::with_seed(0xdeadbeef_cafebabe);
        h.update(data);
        assert_eq!(h.finish128(), one);
    }

    #[test]
    fn reset_returns_to_initial_state() {
        let mut h = Hasher::with_seed(42);
        h.update(b"first stream of data");
        let _ = h.peek128();
        h.reset();
        h.update(b"second");
        let one = hash128_seeded(b"second", 42);
        assert_eq!(h.finish128(), one);
    }

    #[test]
    fn peek_does_not_consume() {
        let mut h = Hasher::new();
        h.update(b"hello world");
        let p1 = h.peek128();
        let p2 = h.peek128();
        assert_eq!(p1, p2, "peek must not change state");

        h.update(b"!");
        let p3 = h.peek128();
        assert_ne!(p1, p3, "peek after more input must change");
    }

    #[test]
    fn finish_phase_changes_subsequent_output() {
        let mut a = Hasher::new();
        a.update(b"aaaa");
        a.update(b"bbbb");
        let no_phase = a.finish128();

        let mut b = Hasher::new();
        b.update(b"aaaa");
        b.finish_phase();
        b.update(b"bbbb");
        let with_phase = b.finish128();

        assert_ne!(no_phase, with_phase);
    }

    // ----- std::hash integration -----

    #[test]
    fn works_with_hashmap() {
        use std::collections::HashMap;
        let mut map: HashMap<String, u32, ZyBuildHasher> = HashMap::with_hasher(ZyBuildHasher);
        for i in 0..1000u32 {
            map.insert(format!("key-{}", i), i);
        }
        for i in 0..1000u32 {
            assert_eq!(map.get(&format!("key-{}", i)), Some(&i));
        }
    }

    // ----- Property tests -----

    #[test]
    fn no_collision_on_distinct_keys() {
        // Modest distribution check: 10K distinct strings produce 10K
        // distinct hash64 outputs (no collisions in this small space).
        let mut seen = std::collections::HashSet::new();
        for i in 0..10_000u32 {
            let key = format!("user-{:08}", i);
            let h = hash64(key.as_bytes());
            assert!(seen.insert(h), "collision on key {}", key);
        }
    }

    #[test]
    fn avalanche_on_small_inputs() {
        // Flipping any bit of a small input must flip a substantial number
        // of output bits (avalanche property).
        let base = hash128(&[0x42u8; 8]);
        for byte in 0..8 {
            for bit in 0..8 {
                let mut data = [0x42u8; 8];
                data[byte] ^= 1u8 << bit;
                let alt = hash128(&data);
                let diff = (base ^ alt).count_ones();
                assert!(
                    diff > 32 && diff < 96,
                    "weak avalanche at byte {} bit {}: diff={}",
                    byte,
                    bit,
                    diff
                );
            }
        }
    }

    #[test]
    fn identical_inputs_identical_outputs_across_widths() {
        for size in [0, 1, 7, 16, 32, 33, 64, 127, 128, 129, 1000, 10_000] {
            let data: Vec<u8> = (0..size).map(|i| i as u8).collect();
            let h32_a = hash32(&data);
            let h32_b = hash32(&data);
            let h64_a = hash64(&data);
            let h64_b = hash64(&data);
            let h128_a = hash128(&data);
            let h128_b = hash128(&data);
            assert_eq!(h32_a, h32_b, "size {}", size);
            assert_eq!(h64_a, h64_b, "size {}", size);
            assert_eq!(h128_a, h128_b, "size {}", size);
        }
    }
}

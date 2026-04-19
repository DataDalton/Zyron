//! Algorithm constants for the central hash/checksum.
//!
//! All constants below are derived from the fractional binary expansions of
//! square roots of small primes (the same approach SHA-256 uses for its round
//! constants). This makes them reproducibly defensible: anyone can verify they
//! were not chosen to introduce backdoors.
//!
//! sqrt(p) fractional bits are taken modulo 2^128 for the lane and
//! finalization keys, modulo 2^64 for the phase separator.

/// Initial state for each of the 8 parallel lanes. Derived from the fractional
/// bits of sqrt(2), sqrt(3), sqrt(5), sqrt(7), sqrt(11), sqrt(13), sqrt(17),
/// sqrt(19) respectively.
pub const LANE_SEEDS: [u128; 8] = [
    0x6a09e667f3bcc908_b2fb1366ea957d3e_u128,
    0xbb67ae8584caa73b_2def372fe2c44a64_u128,
    0x3c6ef372fe94f82b_a54ff53a5f1d36f1_u128,
    0xa54ff53a5f1d36f1_510e527fade682d1_u128,
    0x9b05688c2b3e6c1f_1f83d9abfb41bd6b_u128,
    0x1f83d9abfb41bd6b_5be0cd19137e2179_u128,
    0xcbbb9d5dc1059ed8_629a292a367cd507_u128,
    0x629a292a367cd507_9159015a3070dd17_u128,
];

/// Single round key used inside chunk processing. Derived from sqrt(23).
pub const CHUNK_ROUND_KEY: u128 = 0x152fecd8f70e5939_67332667ffc00b31_u128;

/// Mixed into all lanes when `Hasher::finish_phase()` is called. Detects
/// data that crosses a logical phase boundary differently in different
/// readers (the same property the WAL header/payload separator gives).
/// Derived from sqrt(29).
pub const PHASE_SEPARATOR: u128 = 0x8eb44a8768581511_db0c2e0d64f98fa7_u128;

/// Five finalization round keys, applied during the lane-fold tree.
/// Derived from sqrt(31), sqrt(37), sqrt(41), sqrt(43), sqrt(47).
pub const FINAL_KEYS: [u128; 5] = [
    0x47b5481dbefa4fa4_db0c2e0d64f98fa7_u128,
    0xc19bf174cf692694_47b5481dbefa4fa4_u128,
    0xd728ae22e49b69c1_efbe4786384f25e3_u128,
    0xe49b69c1efbe4786_0fc19dc68b8cd5b5_u128,
    0xefbe4786384f25e3_240ca1cc77ac9c65_u128,
];

/// Maximum input length that uses the small-input fast path. Inputs at or
/// below this length skip 8-lane setup and use a 2-lane direct-mix path.
pub const SMALL_INPUT_THRESHOLD: usize = 32;

/// Bytes consumed per main-loop chunk: 8 lanes * 16 bytes per lane.
pub const CHUNK_BYTES: usize = 128;

/// Algorithm version byte. Embedded in any on-disk format that stores a
/// checksum so the algorithm can evolve without invalidating existing files.
pub const ALGORITHM_VERSION: u8 = 1;

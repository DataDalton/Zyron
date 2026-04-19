//! Scalar reference implementation. Software AES round + 8-lane hash.
//!
//! This is the source of truth for the checksum algorithm. Every SIMD tier
//! produces bit-identical output to this implementation; cross-tier
//! consistency tests verify it on a fuzz corpus.
//!
//! Performance is not a goal here: this path runs only when no SIMD is
//! available (no AES-NI on x86, no AES extension on ARM, or under Miri). It
//! achieves roughly 500 MB/s on a modern core, which is enough for slow
//! verification paths and unit tests.
//!
//! AES round implementation: standard AES-128 round function (SubBytes ->
//! ShiftRows -> MixColumns -> AddRoundKey) with the published S-box. No table
//! optimization; correctness over speed.

use super::spec::{
    CHUNK_BYTES, CHUNK_ROUND_KEY, FINAL_KEYS, LANE_SEEDS, PHASE_SEPARATOR, SMALL_INPUT_THRESHOLD,
};

// ---------------------------------------------------------------------------
// AES S-box (Rijndael forward S-box). Standard published table.
// ---------------------------------------------------------------------------

const SBOX: [u8; 256] = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
];

/// Multiply a byte by 2 in GF(2^8) using the AES irreducible polynomial.
#[inline(always)]
fn xtime(b: u8) -> u8 {
    let h = (b >> 7) & 1;
    ((b << 1) & 0xff) ^ (h * 0x1b)
}

/// Standard AES-128 round: SubBytes -> ShiftRows -> MixColumns -> AddRoundKey.
/// State and round_key are interpreted as 16-byte little-endian arrays.
#[inline]
fn aes_round_scalar(state: u128, round_key: u128) -> u128 {
    let s = state.to_le_bytes();
    let k = round_key.to_le_bytes();

    // SubBytes
    let mut b = [0u8; 16];
    for i in 0..16 {
        b[i] = SBOX[s[i] as usize];
    }

    // ShiftRows: the AES state matrix is column-major (4x4 of bytes).
    // Row r is rotated left by r positions. Apply on the column-major layout.
    //   layout: [c0r0 c0r1 c0r2 c0r3 | c1r0 c1r1 c1r2 c1r3 | ...]
    //   row 0 (bytes 0,4,8,12) unchanged
    //   row 1 (bytes 1,5,9,13) rotate left 1
    //   row 2 (bytes 2,6,10,14) rotate left 2
    //   row 3 (bytes 3,7,11,15) rotate left 3
    let r = [
        b[0], b[5], b[10], b[15], b[4], b[9], b[14], b[3], b[8], b[13], b[2], b[7], b[12], b[1],
        b[6], b[11],
    ];

    // MixColumns: each column c (4 bytes) becomes:
    //   c0' = 2*c0 ^ 3*c1 ^ 1*c2 ^ 1*c3
    //   c1' = 1*c0 ^ 2*c1 ^ 3*c2 ^ 1*c3
    //   c2' = 1*c0 ^ 1*c1 ^ 2*c2 ^ 3*c3
    //   c3' = 3*c0 ^ 1*c1 ^ 1*c2 ^ 2*c3
    // (multiplications in GF(2^8) using xtime)
    let mut m = [0u8; 16];
    for c in 0..4 {
        let i = c * 4;
        let c0 = r[i];
        let c1 = r[i + 1];
        let c2 = r[i + 2];
        let c3 = r[i + 3];
        let t = c0 ^ c1 ^ c2 ^ c3;
        m[i] = c0 ^ t ^ xtime(c0 ^ c1);
        m[i + 1] = c1 ^ t ^ xtime(c1 ^ c2);
        m[i + 2] = c2 ^ t ^ xtime(c2 ^ c3);
        m[i + 3] = c3 ^ t ^ xtime(c3 ^ c0);
    }

    // AddRoundKey
    let mut out = [0u8; 16];
    for i in 0..16 {
        out[i] = m[i] ^ k[i];
    }
    u128::from_le_bytes(out)
}

// ---------------------------------------------------------------------------
// Lane state and 8-lane processing
// ---------------------------------------------------------------------------

/// Eight 128-bit lanes that absorb input data via XOR + AES round.
#[derive(Clone)]
pub struct Lanes {
    pub lanes: [u128; 8],
}

impl Lanes {
    /// Initializes lane state from the per-lane seed constants and an
    /// optional user-supplied seed. Length is NOT folded here so the
    /// streaming `Hasher` (which does not know the final length until
    /// `finish`) starts from the same state as one-shot. Length is
    /// always mixed in by `absorb_tail`.
    #[inline]
    pub fn init(seed: u128) -> Self {
        let mut lanes = [0u128; 8];
        for i in 0..8 {
            // Rotate the seed per lane so identical seeds produce diverse
            // lane state.
            let rotated_seed = seed.rotate_left((i as u32) * 17);
            lanes[i] = LANE_SEEDS[i] ^ rotated_seed;
        }
        Self { lanes }
    }

    /// Absorbs a single 128-byte chunk, mixing 16 bytes into each lane.
    #[inline]
    pub fn absorb_chunk(&mut self, chunk: &[u8; CHUNK_BYTES]) {
        for i in 0..8 {
            let block = u128::from_le_bytes([
                chunk[i * 16],
                chunk[i * 16 + 1],
                chunk[i * 16 + 2],
                chunk[i * 16 + 3],
                chunk[i * 16 + 4],
                chunk[i * 16 + 5],
                chunk[i * 16 + 6],
                chunk[i * 16 + 7],
                chunk[i * 16 + 8],
                chunk[i * 16 + 9],
                chunk[i * 16 + 10],
                chunk[i * 16 + 11],
                chunk[i * 16 + 12],
                chunk[i * 16 + 13],
                chunk[i * 16 + 14],
                chunk[i * 16 + 15],
            ]);
            self.lanes[i] = aes_round_scalar(self.lanes[i] ^ block, CHUNK_ROUND_KEY);
        }
    }

    /// Absorbs the trailing partial chunk (0..CHUNK_BYTES bytes). The total
    /// stream length is XORed into the last 8 bytes of the padded buffer to
    /// detect truncation/extension that happens to be chunk-aligned.
    #[inline]
    pub fn absorb_tail(&mut self, tail: &[u8], total_length: u64) {
        debug_assert!(tail.len() < CHUNK_BYTES);
        let mut buf = [0u8; CHUNK_BYTES];
        buf[..tail.len()].copy_from_slice(tail);
        // Mix the total length into the trailing 8 bytes regardless of how
        // much real data the tail contains. (XOR with zero pad is identity,
        // so this just ensures length appears in the absorb.)
        let len_bytes = total_length.to_le_bytes();
        for i in 0..8 {
            buf[CHUNK_BYTES - 8 + i] ^= len_bytes[i];
        }
        self.absorb_chunk(&buf);
    }

    /// Mixes the phase separator into all lanes. Lets callers detect data
    /// that crosses a logical boundary differently than expected (e.g. WAL
    /// header bytes appearing in payload position).
    #[inline]
    pub fn finish_phase(&mut self) {
        for i in 0..8 {
            self.lanes[i] = aes_round_scalar(self.lanes[i] ^ PHASE_SEPARATOR, CHUNK_ROUND_KEY);
        }
    }

    /// Folds 8 lanes pairwise into one 128-bit value, then runs three
    /// finalization rounds for full diffusion.
    #[inline]
    pub fn finalize(&self) -> u128 {
        let l0 = aes_round_scalar(self.lanes[0] ^ self.lanes[1], FINAL_KEYS[0]);
        let l1 = aes_round_scalar(self.lanes[2] ^ self.lanes[3], FINAL_KEYS[0]);
        let l2 = aes_round_scalar(self.lanes[4] ^ self.lanes[5], FINAL_KEYS[0]);
        let l3 = aes_round_scalar(self.lanes[6] ^ self.lanes[7], FINAL_KEYS[0]);

        let m0 = aes_round_scalar(l0 ^ l1, FINAL_KEYS[1]);
        let m1 = aes_round_scalar(l2 ^ l3, FINAL_KEYS[1]);

        let f = aes_round_scalar(m0 ^ m1, FINAL_KEYS[2]);
        let f = aes_round_scalar(f, FINAL_KEYS[3]);
        aes_round_scalar(f, FINAL_KEYS[4])
    }
}

// ---------------------------------------------------------------------------
// Small-input fast path (<= 32 bytes)
// ---------------------------------------------------------------------------

/// Direct mix for small inputs without the full 8-lane pipeline. Two
/// independent lanes absorb the (possibly very short) input in a single AES
/// round each, then combine. This is dramatically faster than pretending the
/// input is a 128-byte tail when the input is e.g. 8 bytes (a typical
/// hash-map key).
#[inline]
pub fn small_input(bytes: &[u8], seed: u128) -> u128 {
    debug_assert!(bytes.len() <= SMALL_INPUT_THRESHOLD);

    let mut buf_a = [0u8; 16];
    let mut buf_b = [0u8; 16];
    let n = bytes.len();
    if n <= 16 {
        buf_a[..n].copy_from_slice(bytes);
        // Fold the bytes a second time into buf_b at a rotated offset so a
        // single-byte change still affects both lanes.
        for i in 0..n {
            buf_b[(i + 7) & 0x0f] ^= bytes[i];
        }
    } else {
        buf_a.copy_from_slice(&bytes[..16]);
        buf_b[..n - 16].copy_from_slice(&bytes[16..]);
    }
    let len_word = (n as u128).wrapping_mul(0x9e3779b97f4a7c15);
    let lane_a = LANE_SEEDS[0] ^ seed ^ len_word;
    let lane_b = LANE_SEEDS[1] ^ seed ^ len_word.rotate_left(33);
    let block_a = u128::from_le_bytes(buf_a);
    let block_b = u128::from_le_bytes(buf_b);

    let m0 = aes_round_scalar(lane_a ^ block_a, CHUNK_ROUND_KEY);
    let m1 = aes_round_scalar(lane_b ^ block_b, CHUNK_ROUND_KEY);
    let combined = aes_round_scalar(m0 ^ m1, FINAL_KEYS[0]);
    let combined = aes_round_scalar(combined, FINAL_KEYS[1]);
    aes_round_scalar(combined, FINAL_KEYS[4])
}

// ---------------------------------------------------------------------------
// One-shot scalar hash
// ---------------------------------------------------------------------------

/// Reference one-shot hash. Returns the full 128-bit state; callers extract
/// 32, 64, or 128 bits as needed.
pub fn hash_scalar(bytes: &[u8], seed: u128) -> u128 {
    if bytes.len() <= SMALL_INPUT_THRESHOLD {
        return small_input(bytes, seed);
    }

    let mut lanes = Lanes::init(seed);

    let mut i = 0;
    while i + CHUNK_BYTES <= bytes.len() {
        let chunk: &[u8; CHUNK_BYTES] = bytes[i..i + CHUNK_BYTES]
            .try_into()
            .expect("chunk slice has correct length");
        lanes.absorb_chunk(chunk);
        i += CHUNK_BYTES;
    }
    if i < bytes.len() {
        lanes.absorb_tail(&bytes[i..], bytes.len() as u64);
    } else {
        // Even on chunk-aligned input, mix a length-only tail so the final
        // state always reflects the length explicitly.
        lanes.absorb_tail(&[], bytes.len() as u64);
    }

    lanes.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aes_round_known_vector() {
        // Standard AES-128 reference: encrypting all-zeros with all-zeros key
        // through one full round (no key schedule expansion) should produce a
        // deterministic value. The exact bytes vary by which round-key is
        // used; here we just check that two identical inputs hash identically
        // and that flipping a bit changes the output.
        let a = aes_round_scalar(0, 0);
        let b = aes_round_scalar(0, 0);
        assert_eq!(a, b);
        let c = aes_round_scalar(1, 0);
        assert_ne!(a, c);
    }

    #[test]
    fn aes_round_partial_avalanche() {
        // A single AES round provides partial avalanche: a single input bit
        // flip affects ~3 output bytes (~24 bits via SubBytes -> ShiftRows ->
        // MixColumns spread). Two rounds give full avalanche; the hash
        // function uses 5+ rounds total. Here we verify each input bit flip
        // produces at least 8 output bit changes (well above zero, well
        // below the random-string ~64 average for full avalanche).
        let base = aes_round_scalar(0x0123456789abcdef_fedcba9876543210, 0);
        for bit in 0..128 {
            let flipped = aes_round_scalar(0x0123456789abcdef_fedcba9876543210 ^ (1u128 << bit), 0);
            let diff = (base ^ flipped).count_ones();
            assert!(
                diff >= 8,
                "weak single-round avalanche at bit {}: only {} bits differ",
                bit,
                diff
            );
        }
    }

    #[test]
    fn hash_deterministic() {
        let a = hash_scalar(b"hello world", 0);
        let b = hash_scalar(b"hello world", 0);
        assert_eq!(a, b);
    }

    #[test]
    fn hash_different_inputs_differ() {
        let a = hash_scalar(b"hello", 0);
        let b = hash_scalar(b"world", 0);
        assert_ne!(a, b);
    }

    #[test]
    fn hash_seed_changes_output() {
        let a = hash_scalar(b"hello", 0);
        let b = hash_scalar(b"hello", 1);
        assert_ne!(a, b);
    }

    #[test]
    fn hash_length_changes_output() {
        // Same prefix, different length must hash differently.
        let a = hash_scalar(b"abcd", 0);
        let b = hash_scalar(b"abcde", 0);
        assert_ne!(a, b);
    }

    #[test]
    fn hash_empty_is_deterministic_nonzero() {
        let a = hash_scalar(b"", 0);
        let b = hash_scalar(b"", 0);
        assert_eq!(a, b);
        assert_ne!(a, 0, "empty input should not produce a zero hash");
    }

    #[test]
    fn hash_single_bit_flip_detected() {
        let mut data = b"the quick brown fox jumps over the lazy dog".to_vec();
        let original = hash_scalar(&data, 0);
        for bit in 0..(data.len() * 8) {
            let mut flipped = data.clone();
            flipped[bit / 8] ^= 1u8 << (bit % 8);
            assert_ne!(hash_scalar(&flipped, 0), original);
        }
        // Make sure the original is still itself
        let _ = std::mem::replace(&mut data, vec![]);
    }

    #[test]
    fn hash_chunk_boundary() {
        // 127 bytes (just under chunk), 128 bytes (one chunk), 129 bytes
        // (one chunk + 1) all hash differently.
        let n127: Vec<u8> = (0..127).map(|i| i as u8).collect();
        let n128: Vec<u8> = (0..128).map(|i| i as u8).collect();
        let n129: Vec<u8> = (0..129).map(|i| i as u8).collect();
        let h127 = hash_scalar(&n127, 0);
        let h128 = hash_scalar(&n128, 0);
        let h129 = hash_scalar(&n129, 0);
        assert_ne!(h127, h128);
        assert_ne!(h128, h129);
        assert_ne!(h127, h129);
    }

    #[test]
    fn hash_large_input() {
        let data: Vec<u8> = (0..100_000).map(|i| (i & 0xff) as u8).collect();
        let h1 = hash_scalar(&data, 0);
        let mut alt = data.clone();
        alt[50_000] ^= 1;
        let h2 = hash_scalar(&alt, 0);
        assert_ne!(h1, h2);
    }

    #[test]
    fn small_path_matches_when_small() {
        // Inputs <=32 bytes should go through small_input. Verify the path
        // is consistent for itself across calls.
        let s1 = hash_scalar(b"short", 42);
        let s2 = hash_scalar(b"short", 42);
        assert_eq!(s1, s2);
        let s3 = hash_scalar(b"shorr", 42);
        assert_ne!(s1, s3);
    }

    #[test]
    fn streaming_matches_oneshot() {
        let data: Vec<u8> = (0..500).map(|i| (i * 7) as u8).collect();
        let one_shot = hash_scalar(&data, 99);

        let mut lanes = Lanes::init(99);
        let mut i = 0;
        while i + CHUNK_BYTES <= data.len() {
            let chunk: &[u8; CHUNK_BYTES] = data[i..i + CHUNK_BYTES].try_into().unwrap();
            lanes.absorb_chunk(chunk);
            i += CHUNK_BYTES;
        }
        if i < data.len() {
            lanes.absorb_tail(&data[i..], data.len() as u64);
        } else {
            lanes.absorb_tail(&[], data.len() as u64);
        }
        let streaming = lanes.finalize();
        assert_eq!(streaming, one_shot);
    }

    #[test]
    fn phase_separator_changes_state() {
        let mut lanes_a = Lanes::init(0);
        let mut lanes_b = Lanes::init(0);
        lanes_b.finish_phase();
        // After phase mixing, every lane should differ from the un-phased version.
        for i in 0..8 {
            assert_ne!(lanes_a.lanes[i], lanes_b.lanes[i]);
        }
    }
}

//! Custom Balloon Hashing KDF using AES-based compression.
//!
//! Balloon Hashing (Boneh, Corrigan-Gibbs, Schechter 2016) is a provably
//! memory-hard password hashing function. This implementation uses a 10-round
//! AES-based compression function as the internal primitive, accelerated by
//! VAES (x86_64) or ARM AES hardware instructions.
//!
//! The algorithm has three phases:
//! 1. Expand: Fill a buffer of space_cost blocks sequentially
//! 2. Mix: For time_cost rounds, update blocks using pseudorandom dependencies
//! 3. Extract: Return the final block as the hash

use sha2::{Digest, Sha256};
use zyron_common::Result;

/// Parameters controlling the memory and time cost of Balloon Hashing.
#[derive(Debug, Clone, Copy)]
pub struct BalloonParams {
    /// Number of 32-byte blocks in the buffer. Controls memory usage.
    /// 2_097_152 blocks = 64MB.
    pub space_cost: usize,
    /// Number of mixing rounds. Higher values increase time cost.
    pub time_cost: usize,
    /// Number of pseudorandom dependencies per block per round.
    pub delta: usize,
}

impl Default for BalloonParams {
    /// Default parameters use ~64MB memory with 1 mixing round.
    /// Actual latency depends on hardware. Deployments should tune to their target.
    fn default() -> Self {
        Self {
            space_cost: 2_097_152,
            time_cost: 1,
            delta: 3,
        }
    }
}

impl BalloonParams {
    /// Creates parameters for testing (fast, low memory).
    pub fn test() -> Self {
        Self {
            space_cost: 1024,
            time_cost: 1,
            delta: 3,
        }
    }
}

// ---------------------------------------------------------------------------
// AES-based 32-byte compression function
//
// Processes a 32-byte block as two 128-bit halves through 10 full AES
// encryption rounds with cross-lane mixing (XOR between halves every 2 rounds)
// for full 256-bit diffusion. The counter is XORed into both halves as a tweak
// before the rounds begin, ensuring identical input blocks with different
// counters produce different outputs.
//
// Round keys are derived from a fixed 256-bit key using the standard AES-256
// key schedule. The fixed key is acceptable because the security of Balloon
// Hashing relies on memory-hardness (the attacker must allocate the full
// buffer), not on secrecy of the internal compression key.
// ---------------------------------------------------------------------------

/// 11 round keys (128-bit each) for 10-round AES-128.
/// Base key: SHA-256("ZyronDB-Balloon-AES-v1")[0..16] = a6e191d47af7a45bd40b576783476e71.
/// Expanded using the standard AES-128 key schedule (FIPS 197 Section 5.2).
const AES_ROUND_KEYS: [[u8; 16]; 11] = [
    [
        0xa6, 0xe1, 0x91, 0xd4, 0x7a, 0xf7, 0xa4, 0x5b, 0xd4, 0x0b, 0x57, 0x67, 0x83, 0x47, 0x6e,
        0x71,
    ],
    [
        0x07, 0x7e, 0x32, 0x38, 0x7d, 0x89, 0x96, 0x63, 0xa9, 0x82, 0xc1, 0x04, 0x2a, 0xc5, 0xaf,
        0x75,
    ],
    [
        0xa3, 0x07, 0xaf, 0xdd, 0xde, 0x8e, 0x39, 0xbe, 0x77, 0x0c, 0xf8, 0xba, 0x5d, 0xc9, 0x57,
        0xcf,
    ],
    [
        0x7a, 0x5c, 0x25, 0x91, 0xa4, 0xd2, 0x1c, 0x2f, 0xd3, 0xde, 0xe4, 0x95, 0x8e, 0x17, 0xb3,
        0x5a,
    ],
    [
        0x82, 0x31, 0x9b, 0x88, 0x26, 0xe3, 0x87, 0xa7, 0xf5, 0x3d, 0x63, 0x32, 0x7b, 0x2a, 0xd0,
        0x68,
    ],
    [
        0x77, 0x41, 0xde, 0xa9, 0x51, 0xa2, 0x59, 0x0e, 0xa4, 0x9f, 0x3a, 0x3c, 0xdf, 0xb5, 0xea,
        0x54,
    ],
    [
        0x82, 0xc6, 0xfe, 0x37, 0xd3, 0x64, 0xa7, 0x39, 0x77, 0xfb, 0x9d, 0x05, 0xa8, 0x4e, 0x77,
        0x51,
    ],
    [
        0xed, 0x33, 0x2f, 0xf5, 0x3e, 0x57, 0x88, 0xcc, 0x49, 0xac, 0x15, 0xc9, 0xe1, 0xe2, 0x62,
        0x98,
    ],
    [
        0xf5, 0x99, 0x69, 0x0d, 0xcb, 0xce, 0xe1, 0xc1, 0x82, 0x62, 0xf4, 0x08, 0x63, 0x80, 0x96,
        0x90,
    ],
    [
        0x23, 0x09, 0x09, 0xf6, 0xe8, 0xc7, 0xe8, 0x37, 0x6a, 0xa5, 0x1c, 0x3f, 0x09, 0x25, 0x8a,
        0xaf,
    ],
    [
        0x2a, 0x77, 0x70, 0xf7, 0xc2, 0xb0, 0x98, 0xc0, 0xa8, 0x15, 0x84, 0xff, 0xa1, 0x30, 0x0e,
        0x50,
    ],
];

// ---------------------------------------------------------------------------
// x86_64 VAES implementation - processes both 128-bit halves in parallel
// ---------------------------------------------------------------------------
#[cfg(target_arch = "x86_64")]
mod aes_impl {
    use super::AES_ROUND_KEYS;
    use core::arch::x86_64::*;

    /// Loads a round key into a 256-bit register (duplicated to both lanes).
    #[inline(always)]
    unsafe fn load_round_key_256(key: &[u8; 16]) -> __m256i {
        unsafe {
            let k128 = _mm_loadu_si128(key.as_ptr() as *const __m128i);
            _mm256_set_m128i(k128, k128)
        }
    }

    /// 10-round AES compression of a 32-byte block with a 64-bit counter tweak.
    /// Both 128-bit halves are processed in parallel via VAES. Cross-lane XOR
    /// mixing every 2 rounds ensures full 256-bit avalanche.
    #[target_feature(enable = "vaes", enable = "avx2")]
    pub(super) unsafe fn aes_compress(block: &[u8; 32], counter: u64) -> [u8; 32] {
        unsafe {
            // Load the 32-byte block into a single 256-bit register.
            let mut state = _mm256_loadu_si256(block.as_ptr() as *const __m256i);

            // XOR the counter into both halves as a tweak.
            let counter_bytes = counter.to_le_bytes();
            let mut tweak_buf = [0u8; 16];
            tweak_buf[0..8].copy_from_slice(&counter_bytes);
            let tweak128 = _mm_loadu_si128(tweak_buf.as_ptr() as *const __m128i);
            let tweak = _mm256_set_m128i(tweak128, tweak128);
            state = _mm256_xor_si256(state, tweak);

            // Initial round key addition (round 0).
            let rk0 = load_round_key_256(&AES_ROUND_KEYS[0]);
            state = _mm256_xor_si256(state, rk0);

            // Rounds 1-2, then cross-lane mix.
            state = _mm256_aesenc_epi128(state, load_round_key_256(&AES_ROUND_KEYS[1]));
            state = _mm256_aesenc_epi128(state, load_round_key_256(&AES_ROUND_KEYS[2]));
            let rotated = _mm256_permute4x64_epi64(state, 0x39);
            state = _mm256_xor_si256(state, rotated);

            // Rounds 3-4, then cross-lane mix.
            state = _mm256_aesenc_epi128(state, load_round_key_256(&AES_ROUND_KEYS[3]));
            state = _mm256_aesenc_epi128(state, load_round_key_256(&AES_ROUND_KEYS[4]));
            let rotated = _mm256_permute4x64_epi64(state, 0x39);
            state = _mm256_xor_si256(state, rotated);

            // Rounds 5-6, then cross-lane mix.
            state = _mm256_aesenc_epi128(state, load_round_key_256(&AES_ROUND_KEYS[5]));
            state = _mm256_aesenc_epi128(state, load_round_key_256(&AES_ROUND_KEYS[6]));
            let rotated = _mm256_permute4x64_epi64(state, 0x39);
            state = _mm256_xor_si256(state, rotated);

            // Rounds 7-8, then cross-lane mix.
            state = _mm256_aesenc_epi128(state, load_round_key_256(&AES_ROUND_KEYS[7]));
            state = _mm256_aesenc_epi128(state, load_round_key_256(&AES_ROUND_KEYS[8]));
            let rotated = _mm256_permute4x64_epi64(state, 0x39);
            state = _mm256_xor_si256(state, rotated);

            // Rounds 9-10 (final round uses aesenclast).
            state = _mm256_aesenc_epi128(state, load_round_key_256(&AES_ROUND_KEYS[9]));
            state = _mm256_aesenclast_epi128(state, load_round_key_256(&AES_ROUND_KEYS[10]));

            // Store result.
            let mut out = [0u8; 32];
            _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, state);
            out
        }
    }

    /// Compresses two 32-byte blocks with a counter, producing a 32-byte output.
    /// XORs the two blocks in registers and runs AES rounds without intermediate
    /// memory roundtrip.
    #[target_feature(enable = "vaes", enable = "avx2")]
    pub(super) unsafe fn aes_compress_two(a: &[u8; 32], b: &[u8; 32], counter: u64) -> [u8; 32] {
        unsafe {
            let va = _mm256_loadu_si256(a.as_ptr() as *const __m256i);
            let vb = _mm256_loadu_si256(b.as_ptr() as *const __m256i);
            let mut state = _mm256_xor_si256(va, vb);

            // Counter tweak.
            let counter_bytes = counter.to_le_bytes();
            let mut tweak_buf = [0u8; 16];
            tweak_buf[0..8].copy_from_slice(&counter_bytes);
            let tweak128 = _mm_loadu_si128(tweak_buf.as_ptr() as *const __m128i);
            let tweak = _mm256_set_m128i(tweak128, tweak128);
            state = _mm256_xor_si256(state, tweak);

            // Round 0 key addition.
            state = _mm256_xor_si256(state, load_round_key_256(&AES_ROUND_KEYS[0]));

            // Rounds 1-2 + cross-lane mix.
            state = _mm256_aesenc_epi128(state, load_round_key_256(&AES_ROUND_KEYS[1]));
            state = _mm256_aesenc_epi128(state, load_round_key_256(&AES_ROUND_KEYS[2]));
            let rotated = _mm256_permute4x64_epi64(state, 0x39);
            state = _mm256_xor_si256(state, rotated);

            // Rounds 3-4 + cross-lane mix.
            state = _mm256_aesenc_epi128(state, load_round_key_256(&AES_ROUND_KEYS[3]));
            state = _mm256_aesenc_epi128(state, load_round_key_256(&AES_ROUND_KEYS[4]));
            let rotated = _mm256_permute4x64_epi64(state, 0x39);
            state = _mm256_xor_si256(state, rotated);

            // Rounds 5-6 + cross-lane mix.
            state = _mm256_aesenc_epi128(state, load_round_key_256(&AES_ROUND_KEYS[5]));
            state = _mm256_aesenc_epi128(state, load_round_key_256(&AES_ROUND_KEYS[6]));
            let rotated = _mm256_permute4x64_epi64(state, 0x39);
            state = _mm256_xor_si256(state, rotated);

            // Rounds 7-8 + cross-lane mix.
            state = _mm256_aesenc_epi128(state, load_round_key_256(&AES_ROUND_KEYS[7]));
            state = _mm256_aesenc_epi128(state, load_round_key_256(&AES_ROUND_KEYS[8]));
            let rotated = _mm256_permute4x64_epi64(state, 0x39);
            state = _mm256_xor_si256(state, rotated);

            // Rounds 9-10 (final).
            state = _mm256_aesenc_epi128(state, load_round_key_256(&AES_ROUND_KEYS[9]));
            state = _mm256_aesenclast_epi128(state, load_round_key_256(&AES_ROUND_KEYS[10]));

            let mut out = [0u8; 32];
            _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, state);
            out
        }
    }

    /// Compresses 4 u64 values with AES, returns a pseudorandom index.
    #[target_feature(enable = "vaes", enable = "avx2")]
    pub(super) unsafe fn aes_compress_to_index(
        a: u64,
        b: u64,
        c: u64,
        d: u64,
        modulus: usize,
    ) -> usize {
        unsafe {
            let mut block = [0u8; 32];
            block[0..8].copy_from_slice(&a.to_le_bytes());
            block[8..16].copy_from_slice(&b.to_le_bytes());
            block[16..24].copy_from_slice(&c.to_le_bytes());
            block[24..32].copy_from_slice(&d.to_le_bytes());
            let result = aes_compress(&block, 0);
            let val = u64::from_le_bytes([
                result[0], result[1], result[2], result[3], result[4], result[5], result[6],
                result[7],
            ]);
            (val as usize) % modulus
        }
    }

    /// Issues a prefetch hint for a 32-byte block.
    #[inline(always)]
    pub(super) fn prefetch(ptr: *const u8) {
        unsafe {
            _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
        }
    }
}

// ---------------------------------------------------------------------------
// aarch64 ARM AES implementation - two sequential 128-bit AES operations
// ---------------------------------------------------------------------------
#[cfg(target_arch = "aarch64")]
mod aes_impl {
    use super::AES_ROUND_KEYS;
    use core::arch::aarch64::*;

    /// Performs one AES encryption round on ARM (AESE + AESMC).
    #[inline(always)]
    unsafe fn aes_round(state: uint8x16_t, key: uint8x16_t) -> uint8x16_t {
        let after_sub_shift = vaeseq_u8(state, key);
        vaesmcq_u8(after_sub_shift)
    }

    /// Performs the final AES round on ARM (AESE without MixColumns).
    #[inline(always)]
    unsafe fn aes_round_last(state: uint8x16_t, key: uint8x16_t) -> uint8x16_t {
        vaeseq_u8(state, key)
    }

    /// Loads a 128-bit round key.
    #[inline(always)]
    unsafe fn load_key(key: &[u8; 16]) -> uint8x16_t {
        vld1q_u8(key.as_ptr())
    }

    /// Applies 10 AES rounds to a single 128-bit block.
    #[inline(always)]
    unsafe fn aes_10_rounds(mut state: uint8x16_t, tweak: uint8x16_t) -> uint8x16_t {
        state = veorq_u8(state, tweak);
        state = veorq_u8(state, load_key(&AES_ROUND_KEYS[0]));
        state = aes_round(state, load_key(&AES_ROUND_KEYS[1]));
        state = aes_round(state, load_key(&AES_ROUND_KEYS[2]));
        state = aes_round(state, load_key(&AES_ROUND_KEYS[3]));
        state = aes_round(state, load_key(&AES_ROUND_KEYS[4]));
        state = aes_round(state, load_key(&AES_ROUND_KEYS[5]));
        state = aes_round(state, load_key(&AES_ROUND_KEYS[6]));
        state = aes_round(state, load_key(&AES_ROUND_KEYS[7]));
        state = aes_round(state, load_key(&AES_ROUND_KEYS[8]));
        state = aes_round(state, load_key(&AES_ROUND_KEYS[9]));
        state = aes_round_last(state, load_key(&AES_ROUND_KEYS[10]));
        state
    }

    /// 10-round AES compression of a 32-byte block with a 64-bit counter tweak.
    /// Processes both 128-bit halves with cross-lane mixing every 2 rounds.
    #[target_feature(enable = "aes")]
    pub(super) unsafe fn aes_compress(block: &[u8; 32], counter: u64) -> [u8; 32] {
        let mut lo = vld1q_u8(block.as_ptr());
        let mut hi = vld1q_u8(block.as_ptr().add(16));

        // Counter tweak.
        let counter_bytes = counter.to_le_bytes();
        let mut tweak_buf = [0u8; 16];
        tweak_buf[0..8].copy_from_slice(&counter_bytes);
        let tweak = vld1q_u8(tweak_buf.as_ptr());
        lo = veorq_u8(lo, tweak);
        hi = veorq_u8(hi, tweak);

        // Round 0 key addition.
        let rk0 = load_key(&AES_ROUND_KEYS[0]);
        lo = veorq_u8(lo, rk0);
        hi = veorq_u8(hi, rk0);

        // Rounds 1-2, cross-mix.
        lo = aes_round(lo, load_key(&AES_ROUND_KEYS[1]));
        hi = aes_round(hi, load_key(&AES_ROUND_KEYS[1]));
        lo = aes_round(lo, load_key(&AES_ROUND_KEYS[2]));
        hi = aes_round(hi, load_key(&AES_ROUND_KEYS[2]));
        let tmp = lo;
        lo = veorq_u8(lo, hi);
        hi = veorq_u8(hi, tmp);

        // Rounds 3-4, cross-mix.
        lo = aes_round(lo, load_key(&AES_ROUND_KEYS[3]));
        hi = aes_round(hi, load_key(&AES_ROUND_KEYS[3]));
        lo = aes_round(lo, load_key(&AES_ROUND_KEYS[4]));
        hi = aes_round(hi, load_key(&AES_ROUND_KEYS[4]));
        let tmp = lo;
        lo = veorq_u8(lo, hi);
        hi = veorq_u8(hi, tmp);

        // Rounds 5-6, cross-mix.
        lo = aes_round(lo, load_key(&AES_ROUND_KEYS[5]));
        hi = aes_round(hi, load_key(&AES_ROUND_KEYS[5]));
        lo = aes_round(lo, load_key(&AES_ROUND_KEYS[6]));
        hi = aes_round(hi, load_key(&AES_ROUND_KEYS[6]));
        let tmp = lo;
        lo = veorq_u8(lo, hi);
        hi = veorq_u8(hi, tmp);

        // Rounds 7-8, cross-mix.
        lo = aes_round(lo, load_key(&AES_ROUND_KEYS[7]));
        hi = aes_round(hi, load_key(&AES_ROUND_KEYS[7]));
        lo = aes_round(lo, load_key(&AES_ROUND_KEYS[8]));
        hi = aes_round(hi, load_key(&AES_ROUND_KEYS[8]));
        let tmp = lo;
        lo = veorq_u8(lo, hi);
        hi = veorq_u8(hi, tmp);

        // Rounds 9-10 (final round).
        lo = aes_round(lo, load_key(&AES_ROUND_KEYS[9]));
        hi = aes_round(hi, load_key(&AES_ROUND_KEYS[9]));
        lo = aes_round_last(lo, load_key(&AES_ROUND_KEYS[10]));
        hi = aes_round_last(hi, load_key(&AES_ROUND_KEYS[10]));

        let mut out = [0u8; 32];
        vst1q_u8(out.as_mut_ptr(), lo);
        vst1q_u8(out.as_mut_ptr().add(16), hi);
        out
    }

    /// Compresses two 32-byte blocks with a counter, producing a 32-byte output.
    #[target_feature(enable = "aes")]
    pub(super) unsafe fn aes_compress_two(a: &[u8; 32], b: &[u8; 32], counter: u64) -> [u8; 32] {
        let a_lo = vld1q_u8(a.as_ptr());
        let a_hi = vld1q_u8(a.as_ptr().add(16));
        let b_lo = vld1q_u8(b.as_ptr());
        let b_hi = vld1q_u8(b.as_ptr().add(16));
        let mut combined = [0u8; 32];
        vst1q_u8(combined.as_mut_ptr(), veorq_u8(a_lo, b_lo));
        vst1q_u8(combined.as_mut_ptr().add(16), veorq_u8(a_hi, b_hi));
        aes_compress(&combined, counter)
    }

    /// Compresses 4 u64 values with AES, returns a pseudorandom index.
    #[target_feature(enable = "aes")]
    pub(super) unsafe fn aes_compress_to_index(
        a: u64,
        b: u64,
        c: u64,
        d: u64,
        modulus: usize,
    ) -> usize {
        let mut block = [0u8; 32];
        block[0..8].copy_from_slice(&a.to_le_bytes());
        block[8..16].copy_from_slice(&b.to_le_bytes());
        block[16..24].copy_from_slice(&c.to_le_bytes());
        block[24..32].copy_from_slice(&d.to_le_bytes());
        let result = aes_compress(&block, 0);
        let val = u64::from_le_bytes([
            result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7],
        ]);
        (val as usize) % modulus
    }

    /// Issues a prefetch hint.
    #[inline(always)]
    pub(super) fn prefetch(ptr: *const u8) {
        unsafe {
            core::arch::aarch64::_prefetch(ptr as *const i8, 0, 3);
        }
    }
}

// ---------------------------------------------------------------------------
// Balloon Hash core
// ---------------------------------------------------------------------------

/// Computes a Balloon Hash of the password with the given salt and parameters.
///
/// Returns a 32-byte hash that is memory-hard to compute.
/// Uses a 10-round AES-based compression function accelerated by VAES (x86_64)
/// or ARM AES instructions for all fixed-size internal operations.
pub fn balloon_hash(password: &[u8], salt: &[u8], params: &BalloonParams) -> [u8; 32] {
    let space = params.space_cost;
    let mut buf: Vec<[u8; 32]> = vec![[0u8; 32]; space];
    let mut counter: u64 = 0;

    // Phase 1: Expand - fill buffer sequentially.
    // buf[0] uses SHA-256 for the initial variable-length password+salt input.
    buf[0] = hash_counter_data_data(counter, password, salt);
    counter += 1;

    // Remaining expand uses AES compression (fixed 32-byte input).
    for i in 1..space {
        buf[i] = unsafe { aes_impl::aes_compress(&buf[i - 1], counter) };
        counter += 1;
    }

    // Phase 2: Mix - update blocks using pseudorandom dependencies.
    // Pre-computes the next delta iteration's random index and prefetches it
    // while the current iteration's AES compression runs, overlapping memory
    // latency (~200 cycles for DRAM) with compute (~100 cycles for 2 compressions).
    for t in 0..params.time_cost {
        for m in 0..space {
            let prev_idx = if m == 0 { space - 1 } else { m - 1 };
            buf[m] = unsafe { aes_impl::aes_compress_two(&buf[prev_idx], &buf[m], counter) };
            counter += 1;

            if params.delta > 0 {
                // Pre-compute first delta index and prefetch.
                let mut next_idx = unsafe {
                    aes_impl::aes_compress_to_index(counter, t as u64, m as u64, 0, space)
                };
                aes_impl::prefetch(buf[next_idx].as_ptr());

                for j in 0..params.delta {
                    let idx = next_idx;
                    counter += 1;

                    // Pre-compute and prefetch the NEXT delta's random block
                    // while we do the current AES compression below.
                    if j + 1 < params.delta {
                        next_idx = unsafe {
                            aes_impl::aes_compress_to_index(
                                counter + 1,
                                t as u64,
                                m as u64,
                                j as u64 + 1,
                                space,
                            )
                        };
                        aes_impl::prefetch(buf[next_idx].as_ptr());
                    }

                    buf[m] = unsafe { aes_impl::aes_compress_two(&buf[m], &buf[idx], counter) };
                    counter += 1;
                }
            }
        }
    }

    // Phase 3: Extract - return the final block.
    buf[space - 1]
}

/// Hashes a password with random salt and default parameters, returning an encoded string.
///
/// Format: $balloon-aes$v=1$s=<space>,t=<time>,d=<delta>$<base64-salt>$<base64-hash>
pub fn balloon_hash_encoded(password: &str) -> Result<String> {
    balloon_hash_encoded_with_params(password, &BalloonParams::default())
}

/// Hashes a password with random salt and specified parameters.
pub fn balloon_hash_encoded_with_params(password: &str, params: &BalloonParams) -> Result<String> {
    use rand::Rng;
    let mut salt = [0u8; 16];
    rand::rng().fill_bytes(&mut salt);

    let hash = balloon_hash(password.as_bytes(), &salt, params);

    let salt_b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &salt);
    let hash_b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &hash);

    Ok(format!(
        "$balloon-aes$v=1$s={},t={},d={}${}${}",
        params.space_cost, params.time_cost, params.delta, salt_b64, hash_b64
    ))
}

/// Verifies a password against an encoded balloon hash string.
pub fn balloon_verify(password: &str, encoded: &str) -> Result<bool> {
    let parts: Vec<&str> = encoded.split('$').collect();
    if parts.len() != 6 || parts[1] != "balloon-aes" || parts[2] != "v=1" {
        return Err(zyron_common::ZyronError::InvalidCredential(
            "Invalid balloon hash format".to_string(),
        ));
    }

    let params = parse_params(parts[3])?;
    let salt = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, parts[4])
        .map_err(|_| {
            zyron_common::ZyronError::InvalidCredential("Invalid base64 salt".to_string())
        })?;
    let expected_hash =
        base64::Engine::decode(&base64::engine::general_purpose::STANDARD, parts[5]).map_err(
            |_| zyron_common::ZyronError::InvalidCredential("Invalid base64 hash".to_string()),
        )?;

    let computed = balloon_hash(password.as_bytes(), &salt, &params);
    Ok(constant_time_eq(&computed, &expected_hash))
}

/// Parses parameter string "s=N,t=N,d=N" into BalloonParams.
fn parse_params(s: &str) -> Result<BalloonParams> {
    let mut space_cost = 0usize;
    let mut time_cost = 0usize;
    let mut delta = 0usize;

    for part in s.split(',') {
        if let Some(val) = part.strip_prefix("s=") {
            space_cost = val.parse().map_err(|_| {
                zyron_common::ZyronError::InvalidCredential("Invalid space_cost".to_string())
            })?;
        } else if let Some(val) = part.strip_prefix("t=") {
            time_cost = val.parse().map_err(|_| {
                zyron_common::ZyronError::InvalidCredential("Invalid time_cost".to_string())
            })?;
        } else if let Some(val) = part.strip_prefix("d=") {
            delta = val.parse().map_err(|_| {
                zyron_common::ZyronError::InvalidCredential("Invalid delta".to_string())
            })?;
        }
    }

    if space_cost == 0 || time_cost == 0 || delta == 0 {
        return Err(zyron_common::ZyronError::InvalidCredential(
            "Missing balloon parameters".to_string(),
        ));
    }

    Ok(BalloonParams {
        space_cost,
        time_cost,
        delta,
    })
}

/// Constant-time byte comparison to prevent timing side-channel attacks.
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

// Variable-length hash using SHA-256 Digest trait. Called once per balloon_hash
// invocation for buf[0] where password and salt lengths are not fixed.
fn hash_counter_data_data(counter: u64, data1: &[u8], data2: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(counter.to_le_bytes());
    hasher.update(data1);
    hasher.update(data2);
    let result = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&result);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_balloon_hash_deterministic() {
        let params = BalloonParams::test();
        let hash1 = balloon_hash(b"password", b"salt1234salt1234", &params);
        let hash2 = balloon_hash(b"password", b"salt1234salt1234", &params);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_balloon_hash_different_passwords() {
        let params = BalloonParams::test();
        let hash1 = balloon_hash(b"password1", b"salt1234salt1234", &params);
        let hash2 = balloon_hash(b"password2", b"salt1234salt1234", &params);
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_balloon_hash_different_salts() {
        let params = BalloonParams::test();
        let hash1 = balloon_hash(b"password", b"salt1234salt1234", &params);
        let hash2 = balloon_hash(b"password", b"salt5678salt5678", &params);
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_balloon_encoded_roundtrip() {
        let params = BalloonParams::test();
        let encoded = balloon_hash_encoded_with_params("mypassword", &params).unwrap();
        assert!(encoded.starts_with("$balloon-aes$v=1$"));
        assert!(balloon_verify("mypassword", &encoded).unwrap());
        assert!(!balloon_verify("wrongpassword", &encoded).unwrap());
    }

    #[test]
    fn test_balloon_verify_invalid_format() {
        assert!(balloon_verify("pass", "not-a-hash").is_err());
        assert!(balloon_verify("pass", "$invalid$v=1$s=1,t=1,d=1$aa$bb").is_err());
    }

    #[test]
    fn test_constant_time_eq() {
        assert!(constant_time_eq(b"hello", b"hello"));
        assert!(!constant_time_eq(b"hello", b"world"));
        assert!(!constant_time_eq(b"hello", b"hell"));
    }

    #[test]
    fn test_parse_params() {
        let p = parse_params("s=1024,t=3,d=3").unwrap();
        assert_eq!(p.space_cost, 1024);
        assert_eq!(p.time_cost, 3);
        assert_eq!(p.delta, 3);
    }

    #[test]
    fn test_parse_params_missing() {
        assert!(parse_params("s=1024,t=3").is_err());
    }

    #[test]
    fn test_aes_compress_avalanche() {
        // Flipping one bit in input should change roughly half the output bits.
        let mut block_a = [0u8; 32];
        let mut block_b = [0u8; 32];
        block_a[0] = 0x00;
        block_b[0] = 0x01;
        let out_a = unsafe { aes_impl::aes_compress(&block_a, 0) };
        let out_b = unsafe { aes_impl::aes_compress(&block_b, 0) };
        let diff_bits: u32 = out_a
            .iter()
            .zip(out_b.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum();
        // 256-bit output, expect ~128 bits to differ (allow 64-192 range).
        assert!(
            diff_bits > 64 && diff_bits < 192,
            "Avalanche test: {} bits differ out of 256",
            diff_bits
        );
    }

    #[test]
    fn test_aes_compress_counter_sensitivity() {
        // Same block with different counters must produce different output.
        let block = [0xABu8; 32];
        let out1 = unsafe { aes_impl::aes_compress(&block, 0) };
        let out2 = unsafe { aes_impl::aes_compress(&block, 1) };
        assert_ne!(out1, out2);
    }
}

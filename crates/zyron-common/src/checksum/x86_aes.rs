//! x86_64 AES-NI hardware-accelerated tier.
//!
//! Uses `_mm_aesenc_si128` to perform a single AES round per lane in 1 cycle
//! throughput on modern Intel/AMD cores. Eight 128-bit lanes live in eight
//! XMM registers; the chunk loop issues 8 independent AES rounds per
//! 128-byte chunk, which the OoO core schedules in parallel for ~10 GB/s
//! single-threaded.
//!
//! Output is bit-identical to the scalar reference (verified by cross-tier
//! consistency tests). Determinism is guaranteed because AES-NI implements
//! the same AES-128 round function the scalar code computes by hand.

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::{
    __m128i, _mm_aesenc_si128, _mm_loadu_si128, _mm_set_epi64x, _mm_storeu_si128, _mm_xor_si128,
};

use super::scalar::Lanes;
use super::spec::{
    CHUNK_BYTES, CHUNK_ROUND_KEY, FINAL_KEYS, LANE_SEEDS, PHASE_SEPARATOR, SMALL_INPUT_THRESHOLD,
};

/// Returns true if the running CPU supports the AES-NI + SSE2 instructions
/// required by this tier.
#[inline]
pub fn is_available() -> bool {
    is_x86_feature_detected!("aes") && is_x86_feature_detected!("sse2")
}

// ---------------------------------------------------------------------------
// u128 <-> __m128i helpers
// ---------------------------------------------------------------------------

#[inline(always)]
unsafe fn u128_to_m128i(v: u128) -> __m128i {
    // _mm_set_epi64x takes (hi, lo) signed 64-bit args and packs them into
    // an XMM register. Cast through u64 to preserve the bit pattern.
    let lo = v as u64;
    let hi = (v >> 64) as u64;
    _mm_set_epi64x(hi as i64, lo as i64)
}

#[inline(always)]
unsafe fn m128i_to_u128(v: __m128i) -> u128 {
    let mut bytes = [0u8; 16];
    _mm_storeu_si128(bytes.as_mut_ptr() as *mut __m128i, v);
    u128::from_le_bytes(bytes)
}

// ---------------------------------------------------------------------------
// AES-accelerated chunk absorption + finalization
// ---------------------------------------------------------------------------

#[target_feature(enable = "aes,sse2")]
#[inline]
unsafe fn absorb_chunk_aes(lanes: &mut Lanes, chunk: &[u8; CHUNK_BYTES]) {
    let key = u128_to_m128i(CHUNK_ROUND_KEY);
    let ptr = chunk.as_ptr() as *const __m128i;

    // Load 8 lanes into XMM
    let mut l0 = u128_to_m128i(lanes.lanes[0]);
    let mut l1 = u128_to_m128i(lanes.lanes[1]);
    let mut l2 = u128_to_m128i(lanes.lanes[2]);
    let mut l3 = u128_to_m128i(lanes.lanes[3]);
    let mut l4 = u128_to_m128i(lanes.lanes[4]);
    let mut l5 = u128_to_m128i(lanes.lanes[5]);
    let mut l6 = u128_to_m128i(lanes.lanes[6]);
    let mut l7 = u128_to_m128i(lanes.lanes[7]);

    // Load 8 input blocks
    let b0 = _mm_loadu_si128(ptr);
    let b1 = _mm_loadu_si128(ptr.add(1));
    let b2 = _mm_loadu_si128(ptr.add(2));
    let b3 = _mm_loadu_si128(ptr.add(3));
    let b4 = _mm_loadu_si128(ptr.add(4));
    let b5 = _mm_loadu_si128(ptr.add(5));
    let b6 = _mm_loadu_si128(ptr.add(6));
    let b7 = _mm_loadu_si128(ptr.add(7));

    // XOR input into lane state
    l0 = _mm_xor_si128(l0, b0);
    l1 = _mm_xor_si128(l1, b1);
    l2 = _mm_xor_si128(l2, b2);
    l3 = _mm_xor_si128(l3, b3);
    l4 = _mm_xor_si128(l4, b4);
    l5 = _mm_xor_si128(l5, b5);
    l6 = _mm_xor_si128(l6, b6);
    l7 = _mm_xor_si128(l7, b7);

    // AES round per lane. _mm_aesenc_si128(state, key) computes
    //   MixColumns(ShiftRows(SubBytes(state))) XOR key
    // which matches the scalar aes_round_scalar definition exactly.
    l0 = _mm_aesenc_si128(l0, key);
    l1 = _mm_aesenc_si128(l1, key);
    l2 = _mm_aesenc_si128(l2, key);
    l3 = _mm_aesenc_si128(l3, key);
    l4 = _mm_aesenc_si128(l4, key);
    l5 = _mm_aesenc_si128(l5, key);
    l6 = _mm_aesenc_si128(l6, key);
    l7 = _mm_aesenc_si128(l7, key);

    // Store back
    lanes.lanes[0] = m128i_to_u128(l0);
    lanes.lanes[1] = m128i_to_u128(l1);
    lanes.lanes[2] = m128i_to_u128(l2);
    lanes.lanes[3] = m128i_to_u128(l3);
    lanes.lanes[4] = m128i_to_u128(l4);
    lanes.lanes[5] = m128i_to_u128(l5);
    lanes.lanes[6] = m128i_to_u128(l6);
    lanes.lanes[7] = m128i_to_u128(l7);
}

#[target_feature(enable = "aes,sse2")]
#[inline]
unsafe fn finish_phase_aes(lanes: &mut Lanes) {
    let key = u128_to_m128i(CHUNK_ROUND_KEY);
    let sep = u128_to_m128i(PHASE_SEPARATOR);

    for i in 0..8 {
        let l = u128_to_m128i(lanes.lanes[i]);
        let mixed = _mm_xor_si128(l, sep);
        lanes.lanes[i] = m128i_to_u128(_mm_aesenc_si128(mixed, key));
    }
}

#[target_feature(enable = "aes,sse2")]
#[inline]
unsafe fn finalize_aes(lanes: &Lanes) -> u128 {
    let k0 = u128_to_m128i(FINAL_KEYS[0]);
    let k1 = u128_to_m128i(FINAL_KEYS[1]);
    let k2 = u128_to_m128i(FINAL_KEYS[2]);
    let k3 = u128_to_m128i(FINAL_KEYS[3]);
    let k4 = u128_to_m128i(FINAL_KEYS[4]);

    let l0 = u128_to_m128i(lanes.lanes[0]);
    let l1 = u128_to_m128i(lanes.lanes[1]);
    let l2 = u128_to_m128i(lanes.lanes[2]);
    let l3 = u128_to_m128i(lanes.lanes[3]);
    let l4 = u128_to_m128i(lanes.lanes[4]);
    let l5 = u128_to_m128i(lanes.lanes[5]);
    let l6 = u128_to_m128i(lanes.lanes[6]);
    let l7 = u128_to_m128i(lanes.lanes[7]);

    let p0 = _mm_aesenc_si128(_mm_xor_si128(l0, l1), k0);
    let p1 = _mm_aesenc_si128(_mm_xor_si128(l2, l3), k0);
    let p2 = _mm_aesenc_si128(_mm_xor_si128(l4, l5), k0);
    let p3 = _mm_aesenc_si128(_mm_xor_si128(l6, l7), k0);

    let q0 = _mm_aesenc_si128(_mm_xor_si128(p0, p1), k1);
    let q1 = _mm_aesenc_si128(_mm_xor_si128(p2, p3), k1);

    let r = _mm_aesenc_si128(_mm_xor_si128(q0, q1), k2);
    let r = _mm_aesenc_si128(r, k3);
    m128i_to_u128(_mm_aesenc_si128(r, k4))
}

#[target_feature(enable = "aes,sse2")]
#[inline]
unsafe fn small_input_aes(bytes: &[u8], seed: u128) -> u128 {
    debug_assert!(bytes.len() <= SMALL_INPUT_THRESHOLD);
    let n = bytes.len();
    let mut buf_a = [0u8; 16];
    let mut buf_b = [0u8; 16];
    if n <= 16 {
        buf_a[..n].copy_from_slice(bytes);
        for i in 0..n {
            buf_b[(i + 7) & 0x0f] ^= bytes[i];
        }
    } else {
        buf_a.copy_from_slice(&bytes[..16]);
        buf_b[..n - 16].copy_from_slice(&bytes[16..]);
    }
    let len_word = (n as u128).wrapping_mul(0x9e3779b97f4a7c15);

    let lane_a = u128_to_m128i(LANE_SEEDS[0] ^ seed ^ len_word);
    let lane_b = u128_to_m128i(LANE_SEEDS[1] ^ seed ^ len_word.rotate_left(33));
    let block_a = u128_to_m128i(u128::from_le_bytes(buf_a));
    let block_b = u128_to_m128i(u128::from_le_bytes(buf_b));
    let chunk_key = u128_to_m128i(CHUNK_ROUND_KEY);
    let k0 = u128_to_m128i(FINAL_KEYS[0]);
    let k1 = u128_to_m128i(FINAL_KEYS[1]);
    let k4 = u128_to_m128i(FINAL_KEYS[4]);

    let m0 = _mm_aesenc_si128(_mm_xor_si128(lane_a, block_a), chunk_key);
    let m1 = _mm_aesenc_si128(_mm_xor_si128(lane_b, block_b), chunk_key);
    let combined = _mm_aesenc_si128(_mm_xor_si128(m0, m1), k0);
    let combined = _mm_aesenc_si128(combined, k1);
    m128i_to_u128(_mm_aesenc_si128(combined, k4))
}

// ---------------------------------------------------------------------------
// Public AES-NI hash + dispatch hooks
// ---------------------------------------------------------------------------

/// One-shot hash on the AES-NI tier. Output equals `scalar::hash_scalar` for
/// the same input.
///
/// # Safety
/// Caller must have verified `is_available()` returns true.
#[target_feature(enable = "aes,sse2")]
pub unsafe fn hash_x86_aes(bytes: &[u8], seed: u128) -> u128 {
    if bytes.len() <= SMALL_INPUT_THRESHOLD {
        return small_input_aes(bytes, seed);
    }

    let mut lanes = Lanes::init(seed);
    let mut i = 0;
    while i + CHUNK_BYTES <= bytes.len() {
        let chunk: &[u8; CHUNK_BYTES] = bytes[i..i + CHUNK_BYTES]
            .try_into()
            .expect("chunk length matches");
        absorb_chunk_aes(&mut lanes, chunk);
        i += CHUNK_BYTES;
    }
    let mut tail = [0u8; CHUNK_BYTES];
    let tail_bytes = &bytes[i..];
    tail[..tail_bytes.len()].copy_from_slice(tail_bytes);
    let len_bytes = (bytes.len() as u64).to_le_bytes();
    for j in 0..8 {
        tail[CHUNK_BYTES - 8 + j] ^= len_bytes[j];
    }
    absorb_chunk_aes(&mut lanes, &tail);

    finalize_aes(&lanes)
}

/// Streaming chunk absorption. `# Safety`: AES-NI must be available.
#[inline]
pub unsafe fn absorb_chunk(lanes: &mut Lanes, chunk: &[u8; CHUNK_BYTES]) {
    absorb_chunk_aes(lanes, chunk)
}

/// Streaming phase-separator mix. `# Safety`: AES-NI must be available.
#[inline]
pub unsafe fn finish_phase(lanes: &mut Lanes) {
    finish_phase_aes(lanes)
}

/// Streaming finalization. `# Safety`: AES-NI must be available.
#[inline]
pub unsafe fn finalize(lanes: &Lanes) -> u128 {
    finalize_aes(lanes)
}

/// Small-input fast path. `# Safety`: AES-NI must be available.
#[inline]
pub unsafe fn small_input(bytes: &[u8], seed: u128) -> u128 {
    small_input_aes(bytes, seed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checksum::scalar;

    fn aes_or_skip() -> bool {
        if !is_available() {
            eprintln!("AES-NI not available on this host; skipping AES tier test");
            return false;
        }
        true
    }

    #[test]
    fn aes_matches_scalar_for_various_sizes() {
        if !aes_or_skip() {
            return;
        }
        for size in [
            0, 1, 7, 16, 31, 32, 33, 64, 127, 128, 129, 256, 1023, 1024, 4096, 12345,
        ] {
            let data: Vec<u8> = (0..size).map(|i| ((i * 31) & 0xff) as u8).collect();
            let s = scalar::hash_scalar(&data, 0);
            let a = unsafe { hash_x86_aes(&data, 0) };
            assert_eq!(s, a, "size {} mismatch (scalar vs AES-NI)", size);
        }
    }

    #[test]
    fn aes_matches_scalar_with_seed() {
        if !aes_or_skip() {
            return;
        }
        let data: Vec<u8> = (0..512).map(|i| (i * 7 + 11) as u8).collect();
        for seed in [0u128, 1, 42, 0xdeadbeef, u128::MAX, u128::MAX - 1] {
            assert_eq!(
                scalar::hash_scalar(&data, seed),
                unsafe { hash_x86_aes(&data, seed) },
                "seed {} mismatch",
                seed
            );
        }
    }

    #[test]
    fn aes_streaming_matches_scalar_streaming() {
        if !aes_or_skip() {
            return;
        }
        let data: Vec<u8> = (0..1000).map(|i| (i * 13) as u8).collect();

        let mut scalar_lanes = Lanes::init(99);
        let mut aes_lanes = Lanes::init(99);

        let mut i = 0;
        while i + CHUNK_BYTES <= data.len() {
            let chunk: &[u8; CHUNK_BYTES] = data[i..i + CHUNK_BYTES].try_into().unwrap();
            scalar_lanes.absorb_chunk(chunk);
            unsafe { absorb_chunk(&mut aes_lanes, chunk) };
            i += CHUNK_BYTES;
        }
        for j in 0..8 {
            assert_eq!(
                scalar_lanes.lanes[j], aes_lanes.lanes[j],
                "lane {} diverged after chunk absorption",
                j
            );
        }

        scalar_lanes.absorb_tail(&data[i..], data.len() as u64);
        // For AES tier, replicate the same tail handling
        let mut tail = [0u8; CHUNK_BYTES];
        tail[..data.len() - i].copy_from_slice(&data[i..]);
        let len_bytes = (data.len() as u64).to_le_bytes();
        for j in 0..8 {
            tail[CHUNK_BYTES - 8 + j] ^= len_bytes[j];
        }
        unsafe { absorb_chunk(&mut aes_lanes, &tail) };

        for j in 0..8 {
            assert_eq!(
                scalar_lanes.lanes[j], aes_lanes.lanes[j],
                "lane {} diverged after tail",
                j
            );
        }

        let s = scalar_lanes.finalize();
        let a = unsafe { finalize(&aes_lanes) };
        assert_eq!(s, a);
    }

    #[test]
    fn aes_phase_matches_scalar_phase() {
        if !aes_or_skip() {
            return;
        }
        let mut s = Lanes::init(7);
        let mut a = Lanes::init(7);
        s.finish_phase();
        unsafe { finish_phase(&mut a) };
        for i in 0..8 {
            assert_eq!(s.lanes[i], a.lanes[i], "phase mix differs at lane {}", i);
        }
    }

    #[test]
    fn aes_small_input_matches_scalar() {
        if !aes_or_skip() {
            return;
        }
        for size in 0..=SMALL_INPUT_THRESHOLD {
            let data: Vec<u8> = (0..size).map(|i| (i + 7) as u8).collect();
            let s = scalar::small_input(&data, 0);
            let a = unsafe { small_input(&data, 0) };
            assert_eq!(s, a, "small_input mismatch at size {}", size);
        }
    }
}

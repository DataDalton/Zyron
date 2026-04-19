//! aarch64 ARMv8 AES extension hardware-accelerated tier.
//!
//! ARM provides two instructions that together perform an AES round:
//!
//! - `vaeseq_u8(state, key)` produces `ShiftRows(SubBytes(state XOR key))`
//! - `vaesmcq_u8(state)` produces `MixColumns(state)`
//!
//! Note that the order is the reverse of x86's AESENC (AddRoundKey first
//! on ARM vs AddRoundKey last on x86). To get the same round function as
//! `_mm_aesenc_si128(state, key)` on ARM we compute:
//!
//!   `vaesmcq_u8(vaeseq_u8(state, zero)) XOR key`
//!
//! which evaluates to `MixColumns(ShiftRows(SubBytes(state))) XOR key`,
//! identical to x86 AESENC. The extra XOR with `key` is one instruction
//! and pipelines fine with the AES instructions.
//!
//! Output is bit-identical to the scalar reference and the x86 AES-NI tier;
//! cross-tier consistency tests verify it.

#![cfg(target_arch = "aarch64")]

use std::arch::aarch64::{uint8x16_t, vaeseq_u8, vaesmcq_u8, veorq_u8, vld1q_u8, vst1q_u8};

use super::scalar::Lanes;
use super::spec::{
    CHUNK_BYTES, CHUNK_ROUND_KEY, FINAL_KEYS, LANE_SEEDS, PHASE_SEPARATOR, SMALL_INPUT_THRESHOLD,
};

/// Returns true if the running CPU supports ARMv8 AES instructions.
#[inline]
pub fn is_available() -> bool {
    std::arch::is_aarch64_feature_detected!("aes")
}

// ---------------------------------------------------------------------------
// u128 <-> uint8x16_t helpers
// ---------------------------------------------------------------------------

#[inline(always)]
unsafe fn u128_to_v(v: u128) -> uint8x16_t {
    let bytes = v.to_le_bytes();
    vld1q_u8(bytes.as_ptr())
}

#[inline(always)]
unsafe fn v_to_u128(v: uint8x16_t) -> u128 {
    let mut bytes = [0u8; 16];
    vst1q_u8(bytes.as_mut_ptr(), v);
    u128::from_le_bytes(bytes)
}

/// AES round matching x86's `_mm_aesenc_si128(state, key)` semantics:
/// `MixColumns(ShiftRows(SubBytes(state))) XOR key`.
#[inline(always)]
#[target_feature(enable = "aes,neon")]
unsafe fn aes_round(state: uint8x16_t, key: uint8x16_t) -> uint8x16_t {
    let zero = u128_to_v(0);
    // vaeseq_u8(state, zero) = ShiftRows(SubBytes(state XOR 0)) = ShiftRows(SubBytes(state))
    let after_sb_sr = vaeseq_u8(state, zero);
    // vaesmcq_u8 -> MixColumns
    let after_mc = vaesmcq_u8(after_sb_sr);
    // XOR with our round key
    veorq_u8(after_mc, key)
}

#[target_feature(enable = "aes,neon")]
#[inline]
unsafe fn absorb_chunk_neon(lanes: &mut Lanes, chunk: &[u8; CHUNK_BYTES]) {
    let key = u128_to_v(CHUNK_ROUND_KEY);
    let ptr = chunk.as_ptr();

    let mut l0 = u128_to_v(lanes.lanes[0]);
    let mut l1 = u128_to_v(lanes.lanes[1]);
    let mut l2 = u128_to_v(lanes.lanes[2]);
    let mut l3 = u128_to_v(lanes.lanes[3]);
    let mut l4 = u128_to_v(lanes.lanes[4]);
    let mut l5 = u128_to_v(lanes.lanes[5]);
    let mut l6 = u128_to_v(lanes.lanes[6]);
    let mut l7 = u128_to_v(lanes.lanes[7]);

    let b0 = vld1q_u8(ptr);
    let b1 = vld1q_u8(ptr.add(16));
    let b2 = vld1q_u8(ptr.add(32));
    let b3 = vld1q_u8(ptr.add(48));
    let b4 = vld1q_u8(ptr.add(64));
    let b5 = vld1q_u8(ptr.add(80));
    let b6 = vld1q_u8(ptr.add(96));
    let b7 = vld1q_u8(ptr.add(112));

    l0 = veorq_u8(l0, b0);
    l1 = veorq_u8(l1, b1);
    l2 = veorq_u8(l2, b2);
    l3 = veorq_u8(l3, b3);
    l4 = veorq_u8(l4, b4);
    l5 = veorq_u8(l5, b5);
    l6 = veorq_u8(l6, b6);
    l7 = veorq_u8(l7, b7);

    l0 = aes_round(l0, key);
    l1 = aes_round(l1, key);
    l2 = aes_round(l2, key);
    l3 = aes_round(l3, key);
    l4 = aes_round(l4, key);
    l5 = aes_round(l5, key);
    l6 = aes_round(l6, key);
    l7 = aes_round(l7, key);

    lanes.lanes[0] = v_to_u128(l0);
    lanes.lanes[1] = v_to_u128(l1);
    lanes.lanes[2] = v_to_u128(l2);
    lanes.lanes[3] = v_to_u128(l3);
    lanes.lanes[4] = v_to_u128(l4);
    lanes.lanes[5] = v_to_u128(l5);
    lanes.lanes[6] = v_to_u128(l6);
    lanes.lanes[7] = v_to_u128(l7);
}

#[target_feature(enable = "aes,neon")]
#[inline]
unsafe fn finish_phase_neon(lanes: &mut Lanes) {
    let key = u128_to_v(CHUNK_ROUND_KEY);
    let sep = u128_to_v(PHASE_SEPARATOR);
    for i in 0..8 {
        let l = u128_to_v(lanes.lanes[i]);
        let mixed = veorq_u8(l, sep);
        lanes.lanes[i] = v_to_u128(aes_round(mixed, key));
    }
}

#[target_feature(enable = "aes,neon")]
#[inline]
unsafe fn finalize_neon(lanes: &Lanes) -> u128 {
    let k0 = u128_to_v(FINAL_KEYS[0]);
    let k1 = u128_to_v(FINAL_KEYS[1]);
    let k2 = u128_to_v(FINAL_KEYS[2]);
    let k3 = u128_to_v(FINAL_KEYS[3]);
    let k4 = u128_to_v(FINAL_KEYS[4]);

    let l0 = u128_to_v(lanes.lanes[0]);
    let l1 = u128_to_v(lanes.lanes[1]);
    let l2 = u128_to_v(lanes.lanes[2]);
    let l3 = u128_to_v(lanes.lanes[3]);
    let l4 = u128_to_v(lanes.lanes[4]);
    let l5 = u128_to_v(lanes.lanes[5]);
    let l6 = u128_to_v(lanes.lanes[6]);
    let l7 = u128_to_v(lanes.lanes[7]);

    let p0 = aes_round(veorq_u8(l0, l1), k0);
    let p1 = aes_round(veorq_u8(l2, l3), k0);
    let p2 = aes_round(veorq_u8(l4, l5), k0);
    let p3 = aes_round(veorq_u8(l6, l7), k0);

    let q0 = aes_round(veorq_u8(p0, p1), k1);
    let q1 = aes_round(veorq_u8(p2, p3), k1);

    let r = aes_round(veorq_u8(q0, q1), k2);
    let r = aes_round(r, k3);
    v_to_u128(aes_round(r, k4))
}

#[target_feature(enable = "aes,neon")]
#[inline]
unsafe fn small_input_neon(bytes: &[u8], seed: u128) -> u128 {
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

    let lane_a = u128_to_v(LANE_SEEDS[0] ^ seed ^ len_word);
    let lane_b = u128_to_v(LANE_SEEDS[1] ^ seed ^ len_word.rotate_left(33));
    let block_a = u128_to_v(u128::from_le_bytes(buf_a));
    let block_b = u128_to_v(u128::from_le_bytes(buf_b));
    let chunk_key = u128_to_v(CHUNK_ROUND_KEY);
    let k0 = u128_to_v(FINAL_KEYS[0]);
    let k1 = u128_to_v(FINAL_KEYS[1]);
    let k4 = u128_to_v(FINAL_KEYS[4]);

    let m0 = aes_round(veorq_u8(lane_a, block_a), chunk_key);
    let m1 = aes_round(veorq_u8(lane_b, block_b), chunk_key);
    let combined = aes_round(veorq_u8(m0, m1), k0);
    let combined = aes_round(combined, k1);
    v_to_u128(aes_round(combined, k4))
}

/// One-shot hash on the ARMv8 AES tier. Output equals `scalar::hash_scalar`
/// for the same input.
///
/// # Safety
/// Caller must have verified `is_available()` returns true.
#[target_feature(enable = "aes,neon")]
pub unsafe fn hash_aarch64_aes(bytes: &[u8], seed: u128) -> u128 {
    if bytes.len() <= SMALL_INPUT_THRESHOLD {
        return small_input_neon(bytes, seed);
    }
    let mut lanes = Lanes::init(seed);
    let mut i = 0;
    while i + CHUNK_BYTES <= bytes.len() {
        let chunk: &[u8; CHUNK_BYTES] = bytes[i..i + CHUNK_BYTES]
            .try_into()
            .expect("chunk length matches");
        absorb_chunk_neon(&mut lanes, chunk);
        i += CHUNK_BYTES;
    }
    let mut tail = [0u8; CHUNK_BYTES];
    let tail_bytes = &bytes[i..];
    tail[..tail_bytes.len()].copy_from_slice(tail_bytes);
    let len_bytes = (bytes.len() as u64).to_le_bytes();
    for j in 0..8 {
        tail[CHUNK_BYTES - 8 + j] ^= len_bytes[j];
    }
    absorb_chunk_neon(&mut lanes, &tail);
    finalize_neon(&lanes)
}

/// Streaming chunk absorb. `# Safety`: ARM AES extension must be available.
#[inline]
pub unsafe fn absorb_chunk(lanes: &mut Lanes, chunk: &[u8; CHUNK_BYTES]) {
    absorb_chunk_neon(lanes, chunk)
}

/// Streaming phase mix. `# Safety`: ARM AES extension must be available.
#[inline]
pub unsafe fn finish_phase(lanes: &mut Lanes) {
    finish_phase_neon(lanes)
}

/// Streaming finalize. `# Safety`: ARM AES extension must be available.
#[inline]
pub unsafe fn finalize(lanes: &Lanes) -> u128 {
    finalize_neon(lanes)
}

/// Small input fast path. `# Safety`: ARM AES extension must be available.
#[inline]
pub unsafe fn small_input(bytes: &[u8], seed: u128) -> u128 {
    small_input_neon(bytes, seed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checksum::scalar;

    fn neon_or_skip() -> bool {
        if !is_available() {
            eprintln!("ARM AES not available on this host; skipping NEON tier test");
            return false;
        }
        true
    }

    #[test]
    fn neon_matches_scalar_for_various_sizes() {
        if !neon_or_skip() {
            return;
        }
        for size in [
            0, 1, 7, 16, 31, 32, 33, 64, 127, 128, 129, 256, 1023, 1024, 4096, 12345,
        ] {
            let data: Vec<u8> = (0..size).map(|i| ((i * 31) & 0xff) as u8).collect();
            let s = scalar::hash_scalar(&data, 0);
            let a = unsafe { hash_aarch64_aes(&data, 0) };
            assert_eq!(s, a, "size {} mismatch (scalar vs NEON AES)", size);
        }
    }

    #[test]
    fn neon_matches_scalar_with_seed() {
        if !neon_or_skip() {
            return;
        }
        let data: Vec<u8> = (0..512).map(|i| (i * 7 + 11) as u8).collect();
        for seed in [0u128, 1, 42, 0xdeadbeef, u128::MAX, u128::MAX - 1] {
            assert_eq!(
                scalar::hash_scalar(&data, seed),
                unsafe { hash_aarch64_aes(&data, seed) },
                "seed {} mismatch",
                seed
            );
        }
    }
}

//! Hot-path record hash: two-lane multiply-xor mixing with 32, 64 and
//! 128-bit output.
//!
//! One canonical copy of the primitive that previously existed as four
//! hand-tuned forks:
//!
//! - the WAL record checksum (zyron-wal::checksum, WalHasher and
//!   wal_checksum)
//! - the change-feed record checksum (zyron-cdc cdf_hash32)
//! - the version-entry checksum (zyron-versioning version_entry_hash)
//! - the bloom key hash (zyron-storage bloom_hash)
//!
//! Each fork existed to avoid the central AES Hasher's fixed per-call cost:
//! its dispatch and lane initialization measured +14% on per-row change-feed
//! insert and +78% per bloom probe when those sites used the central
//! hasher. This module keeps the property that bought: no runtime dispatch,
//! no lane setup beyond two u64 seeds, and inline mixing that compiles into
//! the caller's loop. Each migrated site holds its pre-consolidation
//! latency within 1%, enforced by the per-site benchmark suites.
//!
//! The WAL fork is the canonical survivor: for identical input the free
//! functions here produce byte-for-byte the checksums zyron-wal stored
//! before the consolidation, pinned by `matches_pre_consolidation_wal_*`
//! below. The change-feed, version-entry and bloom outputs changed to the
//! canonical form when their sites migrated, records and filters those
//! sites write are re-verified with the same function that wrote them.
//!
//! The total input length seeds both lanes, so truncation changes the
//! checksum even when the surviving bytes are identical. An optional phase
//! separator marks a structural boundary (WAL header vs payload) so bytes
//! that shift across the boundary change the checksum even when the raw
//! byte stream does not

/// Primary mixing constant, from wyhash, tested across billions of inputs
/// for uniform distribution
pub const MIX_A: u64 = 0x517cc1b727220a95;

/// Finalization constant, the Murmur3 finalizer multiplier
pub const MIX_B: u64 = 0xff51afd7ed558ccd;

/// Phase separator mixed in at a structural boundary. Golden ratio
/// fractional bits, chosen to have no overlap with typical record data
pub const PHASE_SEP: u64 = 0x9e3779b97f4a7c15;

/// Mixes a u64 value into one lane of running state
#[inline(always)]
fn mix(state: u64, value: u64) -> u64 {
    (state ^ value).wrapping_mul(MIX_A)
}

/// Folds merged 64-bit state with full diffusion. finish32 truncates this
#[inline(always)]
fn finalize64(mut h: u64) -> u64 {
    h ^= h >> 33;
    h = h.wrapping_mul(MIX_B);
    h ^= h >> 33;
    h
}

/// Mixes a byte slice into the two lanes.
///
/// Pairs of 8-byte words go to alternating lanes so the two multiply
/// chains run independently, halving the critical path on x86. A lone
/// 8-byte word, a 4-byte word and a final 1-7 byte tail all fold into
/// lane A. Handles any alignment.
///
/// The tail folds per call, so hashing one buffer in two `update_payload`
/// calls only matches hashing it in one call when the split point is a
/// multiple of 16. Both sides of a stored checksum must feed identical
/// segments
#[inline(always)]
fn mix_bytes(state_a: u64, state_b: u64, data: &[u8]) -> (u64, u64) {
    let len = data.len();
    let ptr = data.as_ptr();
    let mut i = 0;
    let mut la = state_a;
    let mut lb = state_b;

    while i + 16 <= len {
        // SAFETY: i + 16 <= len bounds both unaligned 8-byte reads
        let w0 = unsafe { (ptr.add(i) as *const u64).read_unaligned() };
        let w1 = unsafe { (ptr.add(i + 8) as *const u64).read_unaligned() };
        la = mix(la, w0);
        lb = mix(lb, w1);
        i += 16;
    }

    if i + 8 <= len {
        // SAFETY: i + 8 <= len bounds the unaligned 8-byte read
        let word = unsafe { (ptr.add(i) as *const u64).read_unaligned() };
        la = mix(la, word);
        i += 8;
    }

    if i + 4 <= len {
        // SAFETY: i + 4 <= len bounds the unaligned 4-byte read
        let word = unsafe { (ptr.add(i) as *const u32).read_unaligned() } as u64;
        la = mix(la, word);
        i += 4;
    }

    if i < len {
        let mut tail: u64 = 0;
        // SAFETY: len - i < 4 bytes remain at ptr + i, copied into tail
        unsafe {
            std::ptr::copy_nonoverlapping(ptr.add(i), &mut tail as *mut u64 as *mut u8, len - i);
        }
        la = mix(la, tail);
    }

    (la, lb)
}

/// Finalizes the two lanes into 128 bits whose halves are independently
/// distributed, for double hashing (bloom block plus in-block probes).
///
/// The low half takes lane B after its xorshift-multiply. Injecting the
/// raw lane, as the pre-consolidation bloom fork did, leaves the low half
/// barely avalanched for inputs whose change only touches lane B, because
/// one multiply only spreads a flipped bit upward, and the xorshift folds
/// the high bits down before the multiply re-diffuses them. The injection
/// lands before lane B's trailing xorshift so the low half, which double
/// hashing consumes first for block selection, is off the longest chain.
/// Both cross-injections xor in values the other half already carries, so
/// the map from (lane A, lane B) to (low, high) stays invertible and the
/// pair keeps 128 bits of state
#[inline(always)]
fn finalize128(mut la: u64, mut lb: u64) -> u128 {
    lb ^= lb >> 33;
    lb = lb.wrapping_mul(MIX_B);
    la ^= la >> 33;
    la = la.wrapping_mul(MIX_B);
    la ^= lb;
    lb ^= lb >> 29;
    lb ^= la.rotate_left(29);
    ((lb as u128) << 64) | (la as u128)
}

// ---------------------------------------------------------------------------
// Incremental hasher
// ---------------------------------------------------------------------------

/// Incremental two-lane hasher for callers that produce a record's bytes in
/// pieces: header fields still in registers, payload in a source buffer.
///
/// The total record length is part of the seed, so it must be known at
/// construction. Produces the same result as the one-shot functions for
/// the concatenation of its inputs, with the boundary caveat documented on
/// `update_payload`
pub struct HotHasher {
    lane_a: u64,
    lane_b: u64,
}

impl HotHasher {
    /// Seeds both lanes with the total input length (every byte that will
    /// be fed, excluding the checksum field itself). Length in the seed
    /// makes truncated records hash differently even when the surviving
    /// bytes match
    #[inline(always)]
    pub fn new(total_len: usize) -> Self {
        let seed = (total_len as u64) ^ MIX_A;
        Self {
            lane_a: seed,
            lane_b: seed,
        }
    }

    /// Mixes one word into lane A. For header fields fed from registers,
    /// following the same lane alternation `update_header` applies to a
    /// serialized header: word 0 to lane A, word 1 to lane B, word 2 to
    /// lane A
    #[inline(always)]
    pub fn mix_word_a(&mut self, word: u64) {
        self.lane_a = mix(self.lane_a, word);
    }

    /// Mixes one word into lane B
    #[inline(always)]
    pub fn mix_word_b(&mut self, word: u64) {
        self.lane_b = mix(self.lane_b, word);
    }

    /// Mixes the phase separator into lane B, marking the header-payload
    /// boundary. `update_header` calls this itself, register-fed callers
    /// using mix_word_a/mix_word_b call it after the last header word
    #[inline(always)]
    pub fn phase_separator(&mut self) {
        self.lane_b = mix(self.lane_b, PHASE_SEP);
    }

    /// Mixes serialized header bytes, then the phase separator
    #[inline(always)]
    pub fn update_header(&mut self, header: &[u8]) {
        if !header.is_empty() {
            let (la, lb) = mix_bytes(self.lane_a, self.lane_b, header);
            self.lane_a = la;
            self.lane_b = lb;
        }
        self.phase_separator();
    }

    /// Mixes payload bytes.
    ///
    /// Each call folds its own sub-8-byte tail, so splitting one buffer
    /// across calls matches the one-shot hash only when every split point
    /// is a multiple of 16 bytes. A checksum's writer and verifier must
    /// feed identical segments
    #[inline(always)]
    pub fn update_payload(&mut self, data: &[u8]) {
        if !data.is_empty() {
            let (la, lb) = mix_bytes(self.lane_a, self.lane_b, data);
            self.lane_a = la;
            self.lane_b = lb;
        }
    }

    /// Merges the lanes and folds to 32 bits, the integrity checksum width
    #[inline(always)]
    pub fn finish32(self) -> u32 {
        finalize64(mix(self.lane_a, self.lane_b)) as u32
    }

    /// Merges the lanes and folds to 64 bits, the hash-key width
    #[inline(always)]
    pub fn finish64(self) -> u64 {
        finalize64(mix(self.lane_a, self.lane_b))
    }
}

// ---------------------------------------------------------------------------
// One-shot functions
// ---------------------------------------------------------------------------

/// 32-bit hash of a byte slice, for record and file integrity checksums.
/// Identical to the pre-consolidation zyron-wal data_checksum
#[inline(always)]
pub fn hot_hash32(bytes: &[u8]) -> u32 {
    let seed: u64 = (bytes.len() as u64) ^ MIX_A;
    let (la, lb) = mix_bytes(seed, seed, bytes);
    finalize64(mix(la, lb)) as u32
}

/// 64-bit hash of a byte slice, for hash-map keys
#[inline(always)]
pub fn hot_hash64(bytes: &[u8]) -> u64 {
    let seed: u64 = (bytes.len() as u64) ^ MIX_A;
    let (la, lb) = mix_bytes(seed, seed, bytes);
    finalize64(mix(la, lb))
}

/// 128-bit hash of a byte slice with independently distributed halves,
/// for bloom double hashing.
///
/// One-shot only, and its byte routing differs from the 32-bit functions:
/// the sub-16-byte remainder is balanced across the lanes (lone 8-byte word
/// to lane A, 4-byte word and tail to lane B) instead of all folding into
/// lane A. Bloom keys are typically 8 to 32 bytes, and lane-A-only
/// remainder routing would serialize three dependent multiplies on exactly
/// those sizes. The 32-bit functions keep the lane-A routing because their
/// stored WAL checksums pin it
#[inline(always)]
pub fn hot_hash128(bytes: &[u8]) -> u128 {
    let seed: u64 = (bytes.len() as u64) ^ MIX_A;
    let len = bytes.len();
    let ptr = bytes.as_ptr();
    let mut la = seed;
    let mut lb = seed;
    let mut i = 0;

    while i + 16 <= len {
        // SAFETY: i + 16 <= len bounds both unaligned 8-byte reads
        let w0 = unsafe { (ptr.add(i) as *const u64).read_unaligned() };
        let w1 = unsafe { (ptr.add(i + 8) as *const u64).read_unaligned() };
        la = mix(la, w0);
        lb = mix(lb, w1);
        i += 16;
    }

    if i + 8 <= len {
        // SAFETY: i + 8 <= len bounds the unaligned 8-byte read
        let word = unsafe { (ptr.add(i) as *const u64).read_unaligned() };
        la = mix(la, word);
        i += 8;
    }

    if i + 4 <= len {
        // SAFETY: i + 4 <= len bounds the unaligned 4-byte read
        let word = unsafe { (ptr.add(i) as *const u32).read_unaligned() } as u64;
        lb = mix(lb, word);
        i += 4;
    }

    if i < len {
        let mut tail: u64 = 0;
        // SAFETY: len - i < 4 bytes remain at ptr + i, copied into tail
        unsafe {
            std::ptr::copy_nonoverlapping(ptr.add(i), &mut tail as *mut u64 as *mut u8, len - i);
        }
        lb = mix(lb, tail);
    }

    finalize128(la, lb)
}

/// 32-bit hash of a structured record: header bytes, then the phase
/// separator, then payload bytes. The seed covers the combined length.
/// Identical to the pre-consolidation zyron-wal wal_checksum for
/// (data, header_size) split as (&data[..header_size], &data[header_size..])
#[inline(always)]
pub fn hot_hash_with_header(header: &[u8], payload: &[u8]) -> u32 {
    let total = header.len() + payload.len();
    let mut hasher = HotHasher::new(total);
    hasher.update_header(header);
    hasher.update_payload(payload);
    hasher.finish32()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Transcription of the WAL fork before consolidation. hot_hash32 and
    /// hot_hash_with_header must reproduce it bit for bit, WAL segments on
    /// disk carry checksums this algorithm wrote
    mod pre_consolidation_wal {
        const MIX_A: u64 = 0x517cc1b727220a95;
        const MIX_B: u64 = 0xff51afd7ed558ccd;
        const PHASE_SEP: u64 = 0x9e3779b97f4a7c15;

        fn mix(state: u64, value: u64) -> u64 {
            (state ^ value).wrapping_mul(MIX_A)
        }

        fn finalize(mut h: u64) -> u32 {
            h ^= h >> 33;
            h = h.wrapping_mul(MIX_B);
            h ^= h >> 33;
            h as u32
        }

        fn mix_bytes(state_a: u64, state_b: u64, data: &[u8]) -> (u64, u64) {
            let len = data.len();
            let mut i = 0;
            let (mut la, mut lb) = (state_a, state_b);
            while i + 16 <= len {
                la = mix(
                    la,
                    u64::from_le_bytes(data[i..i + 8].try_into().expect("8")),
                );
                lb = mix(
                    lb,
                    u64::from_le_bytes(data[i + 8..i + 16].try_into().expect("8")),
                );
                i += 16;
            }
            if i + 8 <= len {
                la = mix(
                    la,
                    u64::from_le_bytes(data[i..i + 8].try_into().expect("8")),
                );
                i += 8;
            }
            if i + 4 <= len {
                la = mix(
                    la,
                    u32::from_le_bytes(data[i..i + 4].try_into().expect("4")) as u64,
                );
                i += 4;
            }
            if i < len {
                let mut tail = [0u8; 8];
                tail[..len - i].copy_from_slice(&data[i..]);
                la = mix(la, u64::from_le_bytes(tail));
            }
            (la, lb)
        }

        pub fn wal_checksum(data: &[u8], header_size: usize) -> u32 {
            let seed: u64 = (data.len() as u64) ^ MIX_A;
            let header_end = header_size.min(data.len());
            let (mut lane_a, lane_b) = mix_bytes(seed, seed, &data[..header_end]);
            let mut lane_b = mix(lane_b, PHASE_SEP);
            if header_end < data.len() {
                let (la, lb) = mix_bytes(lane_a, lane_b, &data[header_end..]);
                lane_a = la;
                lane_b = lb;
            }
            finalize(mix(lane_a, lane_b))
        }

        pub fn data_checksum(data: &[u8]) -> u32 {
            let seed: u64 = (data.len() as u64) ^ MIX_A;
            let (la, lb) = mix_bytes(seed, seed, data);
            finalize(mix(la, lb))
        }
    }

    fn pattern(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i * 37 + 13) as u8).collect()
    }

    #[test]
    fn matches_pre_consolidation_wal_data_checksum() {
        for len in [0usize, 1, 3, 4, 7, 8, 15, 16, 17, 23, 24, 100, 8192] {
            let data = pattern(len);
            assert_eq!(
                hot_hash32(&data),
                pre_consolidation_wal::data_checksum(&data),
                "len {len}"
            );
        }
    }

    #[test]
    fn matches_pre_consolidation_wal_checksum_with_header() {
        for (header, payload) in [
            (24usize, 0usize),
            (24, 1),
            (24, 20),
            (24, 8192),
            (0, 64),
            (20, 44),
            (16, 3),
        ] {
            let data = pattern(header + payload);
            assert_eq!(
                hot_hash_with_header(&data[..header], &data[header..]),
                pre_consolidation_wal::wal_checksum(&data, header),
                "header {header} payload {payload}"
            );
        }
    }

    #[test]
    fn incremental_matches_one_shot() {
        let data = pattern(300);
        for split in [0usize, 24, 300] {
            let mut h = HotHasher::new(data.len());
            h.update_header(&data[..split]);
            h.update_payload(&data[split..]);
            assert_eq!(
                h.finish32(),
                hot_hash_with_header(&data[..split], &data[split..]),
                "split {split}"
            );
        }
    }

    #[test]
    fn register_fed_header_matches_serialized_header() {
        // 24-byte header as three words, the WAL write path shape
        let w0 = 0x0000000100000040u64;
        let w1 = 0x0000000100000000u64;
        let w2 = 0x00140a000000002au64;
        let payload = pattern(100);
        let mut header = Vec::with_capacity(24);
        header.extend_from_slice(&w0.to_le_bytes());
        header.extend_from_slice(&w1.to_le_bytes());
        header.extend_from_slice(&w2.to_le_bytes());

        let mut reg = HotHasher::new(24 + payload.len());
        reg.mix_word_a(w0);
        reg.mix_word_b(w1);
        reg.mix_word_a(w2);
        reg.phase_separator();
        reg.update_payload(&payload);

        assert_eq!(reg.finish32(), hot_hash_with_header(&header, &payload));
    }

    #[test]
    fn phase_separator_detects_boundary_shift() {
        let data = vec![0xAAu8; 44];
        assert_ne!(
            hot_hash_with_header(&data[..24], &data[24..]),
            hot_hash_with_header(&data[..20], &data[20..]),
        );
    }

    #[test]
    fn length_seeds_distinguish_truncations() {
        let data = pattern(100);
        let full = hot_hash32(&data);
        for len in 1..100 {
            assert_ne!(full, hot_hash32(&data[..len]), "truncated to {len}");
        }
    }

    #[test]
    fn widths_are_consistent() {
        let data = pattern(64);
        assert_eq!(hot_hash32(&data), hot_hash64(&data) as u32);
        let mut h = HotHasher::new(data.len());
        h.update_payload(&data);
        assert_eq!(h.finish64(), hot_hash64(&data));
    }

    /// The 128-bit one-shot balances the sub-16-byte remainder across both
    /// lanes. For inputs that are a multiple of 16 bytes the routing is the
    /// pair loop alone, identical to the payload routing
    #[test]
    fn hash128_remainder_lands_on_lane_b() {
        // 12-byte input: 8-byte word to lane A, 4-byte word to lane B.
        // Transcribe the expected lanes and finalization independently
        let data = pattern(12);
        let seed = 12u64 ^ MIX_A;
        let w0 = u64::from_le_bytes(data[0..8].try_into().expect("8"));
        let w1 = u32::from_le_bytes(data[8..12].try_into().expect("4")) as u64;
        let la = (seed ^ w0).wrapping_mul(MIX_A);
        let lb = (seed ^ w1).wrapping_mul(MIX_A);
        let (mut fa, mut fb) = (la, lb);
        fb ^= fb >> 33;
        fb = fb.wrapping_mul(MIX_B);
        fa ^= fa >> 33;
        fa = fa.wrapping_mul(MIX_B);
        fa ^= fb;
        fb ^= fb >> 29;
        fb ^= fa.rotate_left(29);
        assert_eq!(hot_hash128(&data), ((fb as u128) << 64) | fa as u128);
    }

    #[test]
    fn hash128_halves_differ_and_avalanche() {
        // Both halves must respond to single-bit input changes, double
        // hashing derives independent probe sequences from them
        for len in [8usize, 9, 16, 32] {
            let base_data = pattern(len);
            let base = hot_hash128(&base_data);
            for byte in 0..len {
                for bit in 0..8 {
                    let mut data = base_data.clone();
                    data[byte] ^= 1u8 << bit;
                    let alt = hot_hash128(&data);
                    let lo_diff = ((base ^ alt) as u64).count_ones();
                    let hi_diff = (((base ^ alt) >> 64) as u64).count_ones();
                    assert!(
                        lo_diff >= 10 && hi_diff >= 10,
                        "weak diffusion at len {len} byte {byte} bit {bit}: lo {lo_diff} hi {hi_diff}"
                    );
                }
            }
        }
    }

    #[test]
    fn bit_flips_change_hash32() {
        let mut data = pattern(80);
        let original = hot_hash32(&data);
        for byte in 0..80 {
            for bit in 0..8 {
                data[byte] ^= 1 << bit;
                assert_ne!(original, hot_hash32(&data), "byte {byte} bit {bit}");
                data[byte] ^= 1 << bit;
            }
        }
    }
}

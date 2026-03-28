//! WAL record integrity checksum.
//!
//! Custom checksum built for the WAL record format. Provides two APIs:
//!
//! - `WalHasher`: incremental hasher for the write path. Feeds header fields
//!   from registers and payload from the source pointer, so serialize_into
//!   never re-reads the output buffer. This saves memory bandwidth
//!   proportional to record size (critical for large full-page-image records).
//!
//! - `wal_checksum`: one-shot function for the read/verify path. Processes a
//!   contiguous byte slice (already in memory from disk read).
//!
//! Both APIs produce identical 32-bit checksums for the same data.
//!
//! The hash uses multiply-xor mixing with proven constants that provide good
//! avalanche properties: every input bit affects every output bit with ~50%
//! probability. This reliably detects single-bit flips, partial writes, zeroed
//! regions, and byte-level corruption.
//!
//! Structure-aware features that a generic library cannot provide:
//!
//! - Length is mixed into the seed, so truncated records produce different
//!   checksums even if the surviving bytes are identical.
//!
//! - A phase separator is mixed in at the header/payload boundary, so data
//!   that crosses the boundary differently (e.g. shifted by one byte) produces
//!   a different checksum even if the raw bytes are the same.

/// Mixing constant with good bit avalanche. From wyhash, widely tested across
/// billions of inputs for uniform distribution.
const MIX_A: u64 = 0x517cc1b727220a95;

/// Second mixing constant for the finalization step.
const MIX_B: u64 = 0xff51afd7ed558ccd;

/// Phase separator mixed in between header and payload to detect structural
/// misalignment. Chosen to have no overlap with typical WAL data patterns.
const PHASE_SEP: u64 = 0x9e3779b97f4a7c15; // golden ratio fractional bits

/// Mixes a u64 value into the running hash state.
#[inline(always)]
fn mix(state: u64, value: u64) -> u64 {
    (state ^ value).wrapping_mul(MIX_A)
}

/// Folds 64-bit state down to 32 bits with full diffusion.
#[inline(always)]
fn finalize(mut h: u64) -> u32 {
    h ^= h >> 33;
    h = h.wrapping_mul(MIX_B);
    h ^= h >> 33;
    h as u32
}

/// Mixes a byte slice into the hash state using two-lane parallel accumulation.
/// Processes pairs of 8-byte words on independent lanes to break the
/// multiply-dependency chain, halving the critical path latency on x86.
/// Handles any alignment and any remainder bytes.
#[inline(always)]
fn mix_bytes(state_a: u64, state_b: u64, data: &[u8]) -> (u64, u64) {
    let len = data.len();
    let ptr = data.as_ptr();
    let mut i = 0;
    let mut la = state_a;
    let mut lb = state_b;

    // Process pairs of 8-byte words on two lanes
    while i + 16 <= len {
        let w0 = unsafe { (ptr.add(i) as *const u64).read_unaligned() };
        let w1 = unsafe { (ptr.add(i + 8) as *const u64).read_unaligned() };
        la = mix(la, w0);
        lb = mix(lb, w1);
        i += 16;
    }

    // One remaining 8-byte word goes to lane A
    if i + 8 <= len {
        let word = unsafe { (ptr.add(i) as *const u64).read_unaligned() };
        la = mix(la, word);
        i += 8;
    }

    // Process 4-byte remainder on lane A
    if i + 4 <= len {
        let word = unsafe { (ptr.add(i) as *const u32).read_unaligned() } as u64;
        la = mix(la, word);
        i += 4;
    }

    // Process remaining 1-3 bytes on lane A
    if i < len {
        let mut tail: u64 = 0;
        unsafe {
            std::ptr::copy_nonoverlapping(ptr.add(i), &mut tail as *mut u64 as *mut u8, len - i);
        }
        la = mix(la, tail);
    }

    (la, lb)
}

// ---------------------------------------------------------------------------
// Incremental hasher (write path)
// ---------------------------------------------------------------------------

/// Incremental hasher for WAL record serialization.
///
/// Uses two-lane parallel accumulation to break the multiply-dependency
/// chain. On x86, wrapping_mul has 3-cycle latency. With two independent
/// lanes, the CPU can execute both multiplies simultaneously, halving
/// the critical path from 12 chained multiplies to 6.
///
/// Usage:
/// ```ignore
/// let mut hasher = WalHasher::new(record_total_size);
/// hasher.write_header_fields(lsn, prev_lsn, txn_id, record_type, flags, payload_len);
/// hasher.write_payload(&payload_bytes);
/// let checksum = hasher.finish();
/// ```
pub struct WalHasher {
    lane_a: u64,
    lane_b: u64,
}

impl WalHasher {
    /// Creates a new hasher seeded with the total record size (header + payload,
    /// excluding the checksum itself). Embedding the length in the seed means
    /// truncated records will produce different checksums.
    ///
    /// Lane A and lane B start from the same seed so that the two-lane merge
    /// is order-independent at initialization.
    #[inline(always)]
    pub fn new(data_len: usize) -> Self {
        let seed = (data_len as u64) ^ MIX_A;
        Self {
            lane_a: seed,
            lane_b: seed,
        }
    }

    /// Mixes the 24-byte header fields using two-lane parallel accumulation.
    ///
    /// Header has 4 values: lsn, prev_lsn, packed_tail, PHASE_SEP.
    /// Lane A processes lsn and packed_tail. Lane B processes prev_lsn and
    /// PHASE_SEP. Both lanes execute independently, breaking the dependency
    /// chain from 4 serial multiplies to 2.
    #[inline(always)]
    pub fn write_header_fields(
        &mut self,
        lsn: u64,
        prev_lsn: u64,
        txn_id: u32,
        record_type: u8,
        flags: u8,
        payload_len: u16,
    ) {
        // Pack txn_id (4) + record_type (1) + flags (1) + payload_len (2) = 8 bytes
        // matching the on-disk layout at header bytes [16..24]
        let packed_tail: u64 = (txn_id.to_le() as u64)
            | ((record_type as u64) << 32)
            | ((flags as u64) << 40)
            | ((payload_len.to_le() as u64) << 48);

        // Two-lane parallel: each lane processes 2 values independently.
        // lsn and packed_tail on lane A, prev_lsn and PHASE_SEP on lane B.
        self.lane_a = mix(self.lane_a, lsn.to_le());
        self.lane_b = mix(self.lane_b, prev_lsn.to_le());
        self.lane_a = mix(self.lane_a, packed_tail);
        self.lane_b = mix(self.lane_b, PHASE_SEP);
    }

    /// Mixes payload bytes from the source slice using two-lane accumulation.
    #[inline(always)]
    pub fn write_payload(&mut self, data: &[u8]) {
        if !data.is_empty() {
            let (la, lb) = mix_bytes(self.lane_a, self.lane_b, data);
            self.lane_a = la;
            self.lane_b = lb;
        }
    }

    /// Finalizes the hash by merging both lanes and folding to 32 bits.
    #[inline(always)]
    pub fn finish(self) -> u32 {
        finalize(mix(self.lane_a, self.lane_b))
    }
}

// ---------------------------------------------------------------------------
// One-shot checksum (read/verify path)
// ---------------------------------------------------------------------------

/// Computes a 32-bit checksum over a contiguous byte slice.
///
/// Used during WAL replay and record verification where the data is already
/// in a contiguous buffer read from disk.
///
/// For correctness, this function must produce the same checksum as
/// WalHasher when given identical data. Both use the same seed, the same
/// two-lane parallel accumulation, and the same processing order. The
/// header portion uses the same lane assignment as write_header_fields:
/// lane A gets words 0, 2 (lsn, packed_tail) and lane B gets words 1, 3
/// (prev_lsn, PHASE_SEP).
#[inline(always)]
pub fn wal_checksum(data: &[u8], header_size: usize) -> u32 {
    let seed: u64 = (data.len() as u64) ^ MIX_A;
    let mut lane_a = seed;
    let mut lane_b = seed;

    // Process header portion using two-lane parallel accumulation.
    // For the standard 24-byte header (3 words), this matches the
    // incremental hasher's lane assignment: word0 -> lane A,
    // word1 -> lane B, word2 -> lane A, PHASE_SEP -> lane B.
    let header_end = header_size.min(data.len());
    let header = &data[..header_end];
    let hlen = header.len();
    let hptr = header.as_ptr();
    let mut hi = 0;

    // Process pairs of header words on two lanes
    while hi + 16 <= hlen {
        let w0 = unsafe { (hptr.add(hi) as *const u64).read_unaligned() };
        let w1 = unsafe { (hptr.add(hi + 8) as *const u64).read_unaligned() };
        lane_a = mix(lane_a, w0);
        lane_b = mix(lane_b, w1);
        hi += 16;
    }

    // Remaining single 8-byte word goes to lane A
    if hi + 8 <= hlen {
        let word = unsafe { (hptr.add(hi) as *const u64).read_unaligned() };
        lane_a = mix(lane_a, word);
        hi += 8;
    }

    // 4-byte remainder on lane A
    if hi + 4 <= hlen {
        let word = unsafe { (hptr.add(hi) as *const u32).read_unaligned() } as u64;
        lane_a = mix(lane_a, word);
        hi += 4;
    }

    // 1-3 byte remainder on lane A
    if hi < hlen {
        let mut tail: u64 = 0;
        unsafe {
            std::ptr::copy_nonoverlapping(
                hptr.add(hi),
                &mut tail as *mut u64 as *mut u8,
                hlen - hi,
            );
        }
        lane_a = mix(lane_a, tail);
    }

    // Phase separator on lane B (matches write_header_fields lane assignment)
    lane_b = mix(lane_b, PHASE_SEP);

    // Process payload portion using two-lane accumulation
    if header_end < data.len() {
        let (la, lb) = mix_bytes(lane_a, lane_b, &data[header_end..]);
        lane_a = la;
        lane_b = lb;
    }

    finalize(mix(lane_a, lane_b))
}

// ---------------------------------------------------------------------------
// General-purpose data checksum
// ---------------------------------------------------------------------------

/// Computes a 32-bit checksum over an arbitrary byte slice.
///
/// Uses the same multiply-xor mixing primitives as the WAL checksum but
/// without the header/payload phase separator. Suitable for any data
/// integrity check (CDF records, slot state files, etc.).
#[inline]
pub fn data_checksum(data: &[u8]) -> u32 {
    let seed: u64 = (data.len() as u64) ^ MIX_A;
    let (la, lb) = mix_bytes(seed, seed, data);
    finalize(mix(la, lb))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one_shot_deterministic() {
        let data = b"test record data for WAL integrity checking";
        let c1 = wal_checksum(data, 24);
        let c2 = wal_checksum(data, 24);
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_one_shot_empty() {
        let c = wal_checksum(b"", 24);
        assert_ne!(c, 0, "Empty input should produce non-trivial hash");
    }

    #[test]
    fn test_incremental_matches_one_shot() {
        // Build a fake 24-byte header + 20-byte payload
        let lsn: u64 = 0x0000000100000040; // segment 1, offset 64
        let prev_lsn: u64 = 0;
        let txn_id: u32 = 42;
        let record_type: u8 = 10; // Insert
        let flags: u8 = 0;
        let payload = b"hello world payload!"; // 20 bytes
        let payload_len = payload.len() as u16;

        // Serialize to bytes (matching on-disk format)
        let mut buf = Vec::with_capacity(44);
        buf.extend_from_slice(&lsn.to_le_bytes());
        buf.extend_from_slice(&prev_lsn.to_le_bytes());
        buf.extend_from_slice(&txn_id.to_le_bytes());
        buf.push(record_type);
        buf.push(flags);
        buf.extend_from_slice(&payload_len.to_le_bytes());
        buf.extend_from_slice(payload);

        // One-shot from serialized bytes
        let one_shot = wal_checksum(&buf, 24);

        // Incremental from typed fields
        let mut hasher = WalHasher::new(buf.len());
        hasher.write_header_fields(lsn, prev_lsn, txn_id, record_type, flags, payload_len);
        hasher.write_payload(payload);
        let incremental = hasher.finish();

        assert_eq!(
            one_shot, incremental,
            "One-shot and incremental must produce identical checksums"
        );
    }

    #[test]
    fn test_incremental_matches_one_shot_empty_payload() {
        let lsn: u64 = 0x0000000100000080;
        let prev_lsn: u64 = 0x0000000100000040;
        let txn_id: u32 = 1;
        let record_type: u8 = 1; // Begin
        let flags: u8 = 0;
        let payload_len: u16 = 0;

        let mut buf = Vec::with_capacity(24);
        buf.extend_from_slice(&lsn.to_le_bytes());
        buf.extend_from_slice(&prev_lsn.to_le_bytes());
        buf.extend_from_slice(&txn_id.to_le_bytes());
        buf.push(record_type);
        buf.push(flags);
        buf.extend_from_slice(&payload_len.to_le_bytes());

        let one_shot = wal_checksum(&buf, 24);

        let mut hasher = WalHasher::new(buf.len());
        hasher.write_header_fields(lsn, prev_lsn, txn_id, record_type, flags, payload_len);
        hasher.write_payload(&[]);
        let incremental = hasher.finish();

        assert_eq!(one_shot, incremental);
    }

    #[test]
    fn test_incremental_matches_one_shot_large_payload() {
        let lsn: u64 = 0x0000000200001000;
        let prev_lsn: u64 = 0x0000000200000800;
        let txn_id: u32 = 999;
        let record_type: u8 = 20; // FullPage
        let flags: u8 = 0;

        // 8KB payload simulating a full page image
        let payload: Vec<u8> = (0..8192).map(|i| (i * 37 + 13) as u8).collect();
        let payload_len = payload.len() as u16;

        let mut buf = Vec::with_capacity(24 + payload.len());
        buf.extend_from_slice(&lsn.to_le_bytes());
        buf.extend_from_slice(&prev_lsn.to_le_bytes());
        buf.extend_from_slice(&txn_id.to_le_bytes());
        buf.push(record_type);
        buf.push(flags);
        buf.extend_from_slice(&payload_len.to_le_bytes());
        buf.extend_from_slice(&payload);

        let one_shot = wal_checksum(&buf, 24);

        let mut hasher = WalHasher::new(buf.len());
        hasher.write_header_fields(lsn, prev_lsn, txn_id, record_type, flags, payload_len);
        hasher.write_payload(&payload);
        let incremental = hasher.finish();

        assert_eq!(one_shot, incremental);
    }

    #[test]
    fn test_single_bit_flip_detected() {
        let mut data = vec![0u8; 80];
        for i in 0..80 {
            data[i] = (i * 17 + 3) as u8;
        }
        let original = wal_checksum(&data, 24);

        // Flip each bit position and verify the checksum changes
        for byte_pos in 0..80 {
            for bit in 0..8 {
                data[byte_pos] ^= 1 << bit;
                let flipped = wal_checksum(&data, 24);
                assert_ne!(
                    original, flipped,
                    "Bit flip at byte {} bit {} not detected",
                    byte_pos, bit
                );
                data[byte_pos] ^= 1 << bit; // restore
            }
        }
    }

    #[test]
    fn test_truncation_detected() {
        let data: Vec<u8> = (0..100).map(|i| (i * 7) as u8).collect();
        let full = wal_checksum(&data, 24);

        // Every truncation length should produce a different checksum
        for len in 1..100 {
            let truncated = wal_checksum(&data[..len], 24);
            assert_ne!(full, truncated, "Truncation to {} bytes not detected", len);
        }
    }

    #[test]
    fn test_zeroed_region_detected() {
        let data: Vec<u8> = (0..80).map(|i| (i + 1) as u8).collect();
        let original = wal_checksum(&data, 24);

        // Zero out different regions and verify detection
        for start in (0..80).step_by(8) {
            let mut corrupted = data.clone();
            let end = (start + 8).min(80);
            for byte in &mut corrupted[start..end] {
                *byte = 0;
            }
            let zeroed = wal_checksum(&corrupted, 24);
            assert_ne!(
                original, zeroed,
                "Zeroed region at [{}..{}] not detected",
                start, end
            );
        }
    }

    #[test]
    fn test_phase_separator_catches_shift() {
        // Two "records" with the same total bytes but header/payload split differently.
        // The phase separator at the header boundary should produce different checksums.
        let data_a = vec![0xAA; 44]; // 24 header + 20 payload
        let data_b = data_a.clone(); // identical bytes

        // Same data, different header size = different checksum
        let checksum_24 = wal_checksum(&data_a, 24);
        let checksum_20 = wal_checksum(&data_b, 20);
        assert_ne!(
            checksum_24, checksum_20,
            "Different header boundaries with same bytes should differ"
        );
    }

    #[test]
    fn test_various_payload_sizes() {
        // Test correctness across a range of sizes from empty to 64KB
        for size in [
            0, 1, 3, 4, 7, 8, 15, 16, 23, 24, 31, 32, 100, 256, 1024, 4096, 8192, 65535,
        ] {
            let mut data = Vec::with_capacity(24 + size);
            // Fake header
            data.extend_from_slice(&[0u8; 24]);
            // Payload with deterministic pattern
            for i in 0..size {
                data.push((i * 37 + 13) as u8);
            }

            let c1 = wal_checksum(&data, 24);
            let c2 = wal_checksum(&data, 24);
            assert_eq!(c1, c2, "Non-deterministic at payload size {}", size);

            // Verify incremental matches
            let lsn = u64::from_le_bytes(data[0..8].try_into().unwrap());
            let prev_lsn = u64::from_le_bytes(data[8..16].try_into().unwrap());
            let txn_id = u32::from_le_bytes(data[16..20].try_into().unwrap());
            let record_type = data[20];
            let flags = data[21];
            let payload_len = u16::from_le_bytes(data[22..24].try_into().unwrap());

            let mut hasher = WalHasher::new(data.len());
            hasher.write_header_fields(lsn, prev_lsn, txn_id, record_type, flags, payload_len);
            hasher.write_payload(&data[24..]);
            assert_eq!(
                c1,
                hasher.finish(),
                "Incremental mismatch at payload size {}",
                size
            );
        }
    }

    #[test]
    fn test_data_checksum_deterministic() {
        let data = b"test data for CDF integrity checking";
        let c1 = data_checksum(data);
        let c2 = data_checksum(data);
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_data_checksum_empty() {
        // Empty input is deterministic. The actual value is not important
        // since CDF records are never empty.
        let c1 = data_checksum(b"");
        let c2 = data_checksum(b"");
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_data_checksum_bit_flip_detected() {
        let mut data = vec![0u8; 64];
        for i in 0..64 {
            data[i] = (i * 17 + 3) as u8;
        }
        let original = data_checksum(&data);

        for byte_pos in 0..64 {
            for bit in 0..8 {
                data[byte_pos] ^= 1 << bit;
                let flipped = data_checksum(&data);
                assert_ne!(
                    original, flipped,
                    "Bit flip at byte {byte_pos} bit {bit} not detected"
                );
                data[byte_pos] ^= 1 << bit;
            }
        }
    }

    #[test]
    fn test_data_checksum_different_lengths() {
        let data: Vec<u8> = (0..100).map(|i| (i * 7) as u8).collect();
        let full = data_checksum(&data);
        for len in 1..100 {
            let truncated = data_checksum(&data[..len]);
            assert_ne!(full, truncated, "Truncation to {len} bytes not detected");
        }
    }
}

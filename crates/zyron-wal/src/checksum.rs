//! WAL record integrity checksum.
//!
//! Thin adapters over the canonical hot-path hash in
//! `zyron_common::checksum::hot`, which consolidated this file's previous
//! hand-rolled two-lane mixer with the change-feed, version-entry and bloom
//! forks. Output is byte-for-byte what the pre-consolidation code produced,
//! pinned by the matches_pre_consolidation_wal tests beside the canonical
//! implementation, so WAL segments on disk keep their stored checksums.
//!
//! Two shapes are exposed:
//!
//! - `WalHasher`: incremental hasher for the write path. Feeds header fields
//!   from registers and payload from the source pointer, so serialize_into
//!   never re-reads the output buffer. This saves memory bandwidth
//!   proportional to record size (critical for large full-page-image records).
//!
//! - `wal_checksum`: one-shot function for the read/verify path. Processes a
//!   contiguous byte slice (already in memory from disk read).
//!
//! Both produce identical 32-bit checksums for the same data.
//!
//! Structure-aware features carried by the canonical primitive:
//!
//! - Length is mixed into the seed, so truncated records produce different
//!   checksums even if the surviving bytes are identical.
//!
//! - A phase separator is mixed in at the header/payload boundary, so data
//!   that crosses the boundary differently (e.g. shifted by one byte) produces
//!   a different checksum even if the raw bytes are the same

use zyron_common::checksum::hot::{HotHasher, hot_hash_with_header, hot_hash32};

// ---------------------------------------------------------------------------
// Incremental hasher (write path)
// ---------------------------------------------------------------------------

/// Incremental hasher for WAL record serialization.
///
/// Wraps the canonical two-lane hasher, mapping the WAL's typed header
/// fields onto its register-fed word API so the write path still hashes
/// straight from registers without materializing header bytes.
///
/// Usage:
/// ```ignore
/// let mut hasher = WalHasher::new(record_total_size);
/// hasher.write_header_fields(lsn, prev_lsn, txn_id, record_type, flags, payload_len);
/// hasher.write_payload(&payload_bytes);
/// let checksum = hasher.finish();
/// ```
pub struct WalHasher(HotHasher);

impl WalHasher {
    /// Creates a new hasher seeded with the total record size (header + payload,
    /// excluding the checksum itself). Embedding the length in the seed means
    /// truncated records will produce different checksums
    #[inline(always)]
    pub fn new(data_len: usize) -> Self {
        Self(HotHasher::new(data_len))
    }

    /// Mixes the 24-byte header fields using two-lane parallel accumulation.
    ///
    /// Word order matches the on-disk header layout so the one-shot verify
    /// over serialized bytes computes the same value: lsn to lane A,
    /// prev_lsn to lane B, the packed tail to lane A, then the phase
    /// separator marks the header/payload boundary
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

        self.0.mix_word_a(lsn.to_le());
        self.0.mix_word_b(prev_lsn.to_le());
        self.0.mix_word_a(packed_tail);
        self.0.phase_separator();
    }

    /// Mixes payload bytes from the source slice using two-lane accumulation
    #[inline(always)]
    pub fn write_payload(&mut self, data: &[u8]) {
        self.0.update_payload(data);
    }

    /// Finalizes the hash by merging both lanes and folding to 32 bits
    #[inline(always)]
    pub fn finish(self) -> u32 {
        self.0.finish32()
    }
}

// ---------------------------------------------------------------------------
// One-shot checksum (read/verify path)
// ---------------------------------------------------------------------------

/// Computes a 32-bit checksum over a contiguous byte slice whose first
/// `header_size` bytes are the record header.
///
/// Produces the same checksum as WalHasher for identical data: the header
/// portion gets the same lane assignment as write_header_fields (word 0 to
/// lane A, word 1 to lane B, word 2 to lane A, phase separator to lane B)
#[inline(always)]
pub fn wal_checksum(data: &[u8], header_size: usize) -> u32 {
    let header_end = header_size.min(data.len());
    hot_hash_with_header(&data[..header_end], &data[header_end..])
}

// ---------------------------------------------------------------------------
// General-purpose data checksum
// ---------------------------------------------------------------------------

/// Computes a 32-bit checksum over an arbitrary byte slice.
///
/// The canonical hash without the header/payload phase separator. Suitable
/// for any data integrity check (CDF records, slot state files, etc.)
#[inline]
pub fn data_checksum(data: &[u8]) -> u32 {
    hot_hash32(data)
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
        // The phase separator at the header boundary should produce different checksums
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
        // since CDF records are never empty
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

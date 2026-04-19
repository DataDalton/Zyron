//! ID generation functions.
//!
//! UUID v4 (random), UUID v7 (time-ordered, RFC 9562), ULID, Snowflake,
//! CUID2, NanoID, KSUID, TSID. All functions are thread-safe.

use rand::RngExt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// UUID v4 (random)
// ---------------------------------------------------------------------------

/// Generates a random UUID v4. Returns 16 bytes.
/// Sets version bits (4) and variant bits (RFC 9562).
pub fn uuid_v4() -> [u8; 16] {
    let mut rng = rand::rng();
    let mut bytes = [0u8; 16];
    rng.fill(&mut bytes);

    // Set version 4 (bits 48-51)
    bytes[6] = (bytes[6] & 0x0F) | 0x40;
    // Set variant 10 (bits 64-65)
    bytes[8] = (bytes[8] & 0x3F) | 0x80;

    bytes
}

// ---------------------------------------------------------------------------
// UUID v7 (time-ordered, RFC 9562)
// ---------------------------------------------------------------------------

/// Generates a time-ordered UUID v7 per RFC 9562.
/// Layout: 48-bit Unix timestamp (ms) + 4-bit version (7) + 12-bit random
/// + 2-bit variant (10) + 62-bit random.
pub fn uuid_v7() -> [u8; 16] {
    let mut rng = rand::rng();

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let millis = now.as_millis() as u64;

    let mut bytes = [0u8; 16];

    // Bytes 0-5: 48-bit Unix timestamp in milliseconds (big-endian)
    bytes[0] = ((millis >> 40) & 0xFF) as u8;
    bytes[1] = ((millis >> 32) & 0xFF) as u8;
    bytes[2] = ((millis >> 24) & 0xFF) as u8;
    bytes[3] = ((millis >> 16) & 0xFF) as u8;
    bytes[4] = ((millis >> 8) & 0xFF) as u8;
    bytes[5] = (millis & 0xFF) as u8;

    // Bytes 6-15: random
    let random_bytes: [u8; 10] = rng.random();
    bytes[6..16].copy_from_slice(&random_bytes);

    // Set version 7 (bits 48-51)
    bytes[6] = (bytes[6] & 0x0F) | 0x70;
    // Set variant 10 (bits 64-65)
    bytes[8] = (bytes[8] & 0x3F) | 0x80;

    bytes
}

/// Formats UUID bytes as "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx".
pub fn uuid_to_string(bytes: &[u8; 16]) -> String {
    let hex_chars = b"0123456789abcdef";
    let mut s = String::with_capacity(36);
    for (i, &b) in bytes.iter().enumerate() {
        if i == 4 || i == 6 || i == 8 || i == 10 {
            s.push('-');
        }
        s.push(hex_chars[(b >> 4) as usize] as char);
        s.push(hex_chars[(b & 0x0F) as usize] as char);
    }
    s
}

// ---------------------------------------------------------------------------
// ULID (Universally Unique Lexicographically Sortable Identifier)
// ---------------------------------------------------------------------------

const CROCKFORD_BASE32: &[u8] = b"0123456789ABCDEFGHJKMNPQRSTVWXYZ";

/// Generates a ULID: 10 characters of timestamp (ms) + 16 characters of randomness.
/// Crockford's Base32 encoding, 26 characters total, time-ordered.
pub fn ulid() -> String {
    let mut rng = rand::rng();

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let millis = now.as_millis() as u64;

    let mut result = [0u8; 26];

    // Encode 48-bit timestamp into first 10 characters (Crockford Base32)
    let mut ts = millis;
    for i in (0..10).rev() {
        result[i] = CROCKFORD_BASE32[(ts & 0x1F) as usize];
        ts >>= 5;
    }

    // Encode 80 bits of randomness into last 16 characters
    let random_bytes: [u8; 10] = rng.random();
    let mut bits: u128 = 0;
    for &b in &random_bytes {
        bits = (bits << 8) | b as u128;
    }
    for i in (10..26).rev() {
        result[i] = CROCKFORD_BASE32[(bits & 0x1F) as usize];
        bits >>= 5;
    }

    String::from_utf8(result.to_vec()).unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Snowflake ID (Twitter-style)
// ---------------------------------------------------------------------------

/// Generates a Snowflake ID: 41-bit timestamp + 10-bit machine + 12-bit sequence.
/// machine_id: 0-1023 (10 bits).
/// state: packed (timestamp_ms << 12 | sequence). Spin-waits to the next ms
/// when the 12-bit sequence is exhausted within a single millisecond, so IDs
/// are always unique and monotonically increasing.
/// The timestamp epoch is 2020-01-01 00:00:00 UTC.
pub fn snowflake(machine_id: u16, state: &AtomicU64) -> i64 {
    let custom_epoch = 1_577_836_800_000u64; // 2020-01-01 00:00:00 UTC in ms

    loop {
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let now_rel = now_ms.saturating_sub(custom_epoch);

        let prev = state.load(Ordering::Acquire);
        let prev_ts = prev >> 12;
        let prev_seq = prev & 0x0FFF;

        let (next_ts, next_seq) = if now_rel > prev_ts {
            (now_rel, 0u64)
        } else if prev_seq + 1 < 0x1000 {
            (prev_ts, prev_seq + 1)
        } else {
            // Sequence exhausted within this ms: spin until the clock advances.
            std::hint::spin_loop();
            continue;
        };

        let next = (next_ts << 12) | next_seq;
        if state
            .compare_exchange(prev, next, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            let id = ((next_ts & 0x1FFFFFFFFFF) << 22)
                | (((machine_id & 0x03FF) as u64) << 12)
                | next_seq;
            return id as i64;
        }
    }
}

// ---------------------------------------------------------------------------
// CUID2
// ---------------------------------------------------------------------------

/// Generates a CUID2: collision-resistant unique identifier.
/// Starts with a letter, followed by random characters.
/// Uses a hash-based approach for collision resistance.
pub fn cuid2() -> String {
    let mut rng = rand::rng();
    let length = 24;

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let timestamp = now.as_millis() as u64;

    // Build entropy from timestamp + random data
    let random_bytes: [u8; 32] = rng.random();
    let mut input = Vec::with_capacity(40);
    input.extend_from_slice(&timestamp.to_le_bytes());
    input.extend_from_slice(&random_bytes);

    // Hash with SHA-256 to mix entropy
    let hash = crate::crypto::sha256(&input);

    // Encode as base36 (0-9, a-z)
    let base36 = b"0123456789abcdefghijklmnopqrstuvwxyz";
    let mut result = String::with_capacity(length);

    // First character must be a letter (a-z)
    result.push(base36[(hash[0] % 26 + 10) as usize] as char);

    // Remaining characters from hash bytes
    for i in 1..length {
        let byte_idx = i % hash.len();
        result.push(base36[(hash[byte_idx].wrapping_add(i as u8)) as usize % 36] as char);
    }

    result
}

// ---------------------------------------------------------------------------
// NanoID
// ---------------------------------------------------------------------------

const NANOID_ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-";

/// Generates a NanoID: URL-safe random identifier of the given length.
/// Default length is 21 if 0 is passed.
pub fn nanoid(len: usize) -> String {
    let length = if len == 0 { 21 } else { len };
    let mut rng = rand::rng();
    let mut result = String::with_capacity(length);

    for _ in 0..length {
        let idx: usize = rng.random_range(0..NANOID_ALPHABET.len());
        result.push(NANOID_ALPHABET[idx] as char);
    }
    result
}

// ---------------------------------------------------------------------------
// KSUID (K-Sortable Unique ID)
// ---------------------------------------------------------------------------

const KSUID_EPOCH: u64 = 1_400_000_000; // Custom epoch: 2014-05-13T16:53:20Z

/// Generates a KSUID: 4-byte timestamp + 16-byte random payload.
/// Base62 encoded to 27 characters.
pub fn ksuid() -> String {
    let mut rng = rand::rng();

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let seconds = now.as_secs().saturating_sub(KSUID_EPOCH);

    let mut bytes = [0u8; 20];
    bytes[0] = ((seconds >> 24) & 0xFF) as u8;
    bytes[1] = ((seconds >> 16) & 0xFF) as u8;
    bytes[2] = ((seconds >> 8) & 0xFF) as u8;
    bytes[3] = (seconds & 0xFF) as u8;

    let random: [u8; 16] = rng.random();
    bytes[4..20].copy_from_slice(&random);

    base62_encode(&bytes)
}

fn base62_encode(data: &[u8]) -> String {
    const BASE62: &[u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

    if data.is_empty() {
        return String::new();
    }

    let mut digits: Vec<u8> = Vec::with_capacity(data.len() * 2);
    for &byte in data {
        let mut carry = byte as u32;
        for d in digits.iter_mut() {
            carry += (*d as u32) << 8;
            *d = (carry % 62) as u8;
            carry /= 62;
        }
        while carry > 0 {
            digits.push((carry % 62) as u8);
            carry /= 62;
        }
    }

    // Pad to 27 characters for KSUID
    while digits.len() < 27 {
        digits.push(0);
    }

    digits
        .iter()
        .rev()
        .map(|&d| BASE62[d as usize] as char)
        .collect()
}

// ---------------------------------------------------------------------------
// TSID (Time-Sorted ID)
// ---------------------------------------------------------------------------

/// Generates a TSID: 42-bit timestamp (ms) + 22-bit random.
/// Simpler than Snowflake, does not require machine_id configuration.
pub fn tsid() -> i64 {
    let mut rng = rand::rng();

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let millis = now.as_millis() as u64;

    let random: u32 = rng.random_range(0..(1 << 22));

    let id = ((millis & 0x3FFFFFFFFFF) << 22) | (random as u64);
    id as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    // UUID v4
    #[test]
    fn test_uuid_v4_version() {
        let uuid = uuid_v4();
        assert_eq!((uuid[6] >> 4) & 0x0F, 4);
    }

    #[test]
    fn test_uuid_v4_variant() {
        let uuid = uuid_v4();
        assert_eq!((uuid[8] >> 6) & 0x03, 2); // variant 10
    }

    #[test]
    fn test_uuid_v4_unique() {
        let a = uuid_v4();
        let b = uuid_v4();
        assert_ne!(a, b);
    }

    #[test]
    fn test_uuid_v4_format() {
        let uuid = uuid_v4();
        let s = uuid_to_string(&uuid);
        assert_eq!(s.len(), 36);
        assert_eq!(s.chars().nth(8).unwrap(), '-');
        assert_eq!(s.chars().nth(13).unwrap(), '-');
        assert_eq!(s.chars().nth(18).unwrap(), '-');
        assert_eq!(s.chars().nth(23).unwrap(), '-');
    }

    // UUID v7
    #[test]
    fn test_uuid_v7_version() {
        let uuid = uuid_v7();
        assert_eq!((uuid[6] >> 4) & 0x0F, 7);
    }

    #[test]
    fn test_uuid_v7_variant() {
        let uuid = uuid_v7();
        assert_eq!((uuid[8] >> 6) & 0x03, 2);
    }

    #[test]
    fn test_uuid_v7_time_ordered() {
        let a = uuid_v7();
        let b = uuid_v7();
        // UUIDs generated later should sort after earlier ones
        // Compare the first 6 bytes (timestamp)
        assert!(a[..6] <= b[..6]);
    }

    #[test]
    fn test_uuid_v7_unique() {
        let a = uuid_v7();
        let b = uuid_v7();
        assert_ne!(a, b);
    }

    #[test]
    fn test_uuid_v7_has_timestamp() {
        let uuid = uuid_v7();
        // First 6 bytes should be non-zero (current timestamp)
        let ts_bytes = &uuid[..6];
        assert!(ts_bytes.iter().any(|&b| b != 0));
    }

    // UUID to string
    #[test]
    fn test_uuid_to_string_format() {
        let bytes = [
            0x55, 0x0e, 0x84, 0x00, 0xe2, 0x9b, 0x41, 0xd4, 0xa7, 0x16, 0x44, 0x66, 0x55, 0x44,
            0x00, 0x00,
        ];
        assert_eq!(
            uuid_to_string(&bytes),
            "550e8400-e29b-41d4-a716-446655440000"
        );
    }

    // ULID
    #[test]
    fn test_ulid_length() {
        let id = ulid();
        assert_eq!(id.len(), 26);
    }

    #[test]
    fn test_ulid_unique() {
        let a = ulid();
        let b = ulid();
        assert_ne!(a, b);
    }

    #[test]
    fn test_ulid_crockford_chars() {
        let id = ulid();
        for c in id.chars() {
            assert!(
                c.is_ascii_alphanumeric(),
                "ULID contains non-alphanumeric character: {}",
                c
            );
            // Crockford Base32 excludes I, L, O, U
            assert!(c != 'I' && c != 'L' && c != 'O' && c != 'U');
        }
    }

    #[test]
    fn test_ulid_time_prefix_ordered() {
        let a = ulid();
        let b = ulid();
        // The timestamp portion (first 10 chars) should be non-decreasing.
        // Within the same millisecond, the random portion can vary.
        assert!(a[..10] <= b[..10]);
    }

    // Snowflake
    #[test]
    fn test_snowflake_positive() {
        let seq = AtomicU64::new(0);
        let id = snowflake(1, &seq);
        assert!(id > 0);
    }

    #[test]
    fn test_snowflake_unique() {
        let seq = AtomicU64::new(0);
        let a = snowflake(1, &seq);
        let b = snowflake(1, &seq);
        assert_ne!(a, b);
    }

    #[test]
    fn test_snowflake_sequence_increments() {
        let seq = AtomicU64::new(0);
        let a = snowflake(1, &seq);
        let b = snowflake(1, &seq);
        // With same timestamp, b should have sequence+1 from a
        let seq_a = a & 0xFFF;
        let seq_b = b & 0xFFF;
        assert_eq!(seq_b, seq_a + 1);
    }

    #[test]
    fn test_snowflake_machine_id() {
        let seq = AtomicU64::new(0);
        let id = snowflake(42, &seq);
        let extracted_machine = ((id >> 12) & 0x3FF) as u16;
        assert_eq!(extracted_machine, 42);
    }

    // CUID2
    #[test]
    fn test_cuid2_length() {
        let id = cuid2();
        assert_eq!(id.len(), 24);
    }

    #[test]
    fn test_cuid2_starts_with_letter() {
        let id = cuid2();
        assert!(id.chars().next().unwrap().is_ascii_lowercase());
    }

    #[test]
    fn test_cuid2_unique() {
        let a = cuid2();
        let b = cuid2();
        assert_ne!(a, b);
    }

    // NanoID
    #[test]
    fn test_nanoid_default_length() {
        let id = nanoid(0);
        assert_eq!(id.len(), 21);
    }

    #[test]
    fn test_nanoid_custom_length() {
        let id = nanoid(10);
        assert_eq!(id.len(), 10);
    }

    #[test]
    fn test_nanoid_url_safe() {
        let id = nanoid(100);
        for c in id.chars() {
            assert!(
                c.is_ascii_alphanumeric() || c == '_' || c == '-',
                "NanoID contains unsafe character: {}",
                c
            );
        }
    }

    #[test]
    fn test_nanoid_unique() {
        let a = nanoid(21);
        let b = nanoid(21);
        assert_ne!(a, b);
    }

    // KSUID
    #[test]
    fn test_ksuid_length() {
        let id = ksuid();
        assert_eq!(id.len(), 27);
    }

    #[test]
    fn test_ksuid_unique() {
        let a = ksuid();
        let b = ksuid();
        assert_ne!(a, b);
    }

    #[test]
    fn test_ksuid_valid_format() {
        let a = ksuid();
        let b = ksuid();
        // Both should be 27 characters of base62
        assert_eq!(a.len(), 27);
        assert_eq!(b.len(), 27);
        assert!(a.chars().all(|c| c.is_ascii_alphanumeric()));
        assert!(b.chars().all(|c| c.is_ascii_alphanumeric()));
    }

    // TSID
    #[test]
    fn test_tsid_positive() {
        let id = tsid();
        assert!(id > 0);
    }

    #[test]
    fn test_tsid_unique() {
        let a = tsid();
        let b = tsid();
        assert_ne!(a, b);
    }

    #[test]
    fn test_tsid_timestamp_component() {
        let a = tsid();
        let b = tsid();
        // The timestamp portion (upper 42 bits) should be non-decreasing.
        // The random portion (lower 22 bits) can vary.
        let ts_a = a >> 22;
        let ts_b = b >> 22;
        assert!(ts_b >= ts_a);
    }
}

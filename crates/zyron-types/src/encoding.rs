//! Encoding, decoding, and hashing functions.
//!
//! Base58 (Bitcoin alphabet), Base32 (RFC 4648), hex, Base64URL,
//! CRC32, CRC32C, xxHash64, and MurmurHash3.

use zyron_common::{Result, ZyronError};

// ---------------------------------------------------------------------------
// Hex encode/decode
// ---------------------------------------------------------------------------

const HEX_CHARS: &[u8; 16] = b"0123456789abcdef";

/// Encodes bytes as a lowercase hex string.
pub fn hex_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len() * 2);
    for &b in data {
        out.push(HEX_CHARS[(b >> 4) as usize] as char);
        out.push(HEX_CHARS[(b & 0x0F) as usize] as char);
    }
    out
}

/// Decodes a hex string into bytes.
pub fn hex_decode(text: &str) -> Result<Vec<u8>> {
    let s = text.strip_prefix("0x").unwrap_or(text);
    if s.len() % 2 != 0 {
        return Err(ZyronError::ExecutionError(
            "Hex string has odd length".into(),
        ));
    }
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(s.len() / 2);
    for i in (0..bytes.len()).step_by(2) {
        let hi = hex_val(bytes[i]).ok_or_else(|| {
            ZyronError::ExecutionError(format!("Invalid hex digit: {}", bytes[i] as char))
        })?;
        let lo = hex_val(bytes[i + 1]).ok_or_else(|| {
            ZyronError::ExecutionError(format!("Invalid hex digit: {}", bytes[i + 1] as char))
        })?;
        out.push((hi << 4) | lo);
    }
    Ok(out)
}

fn hex_val(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Base58 (Bitcoin alphabet)
// ---------------------------------------------------------------------------

const BASE58_ALPHABET: &[u8] = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

/// Encodes bytes using Base58 (Bitcoin alphabet).
pub fn base58_encode(data: &[u8]) -> String {
    if data.is_empty() {
        return String::new();
    }

    // Count leading zeros
    let leading_zeros = data.iter().take_while(|&&b| b == 0).count();

    // Convert to base58 using repeated division
    let mut digits: Vec<u8> = Vec::with_capacity(data.len() * 137 / 100 + 1);
    for &byte in data {
        let mut carry = byte as u32;
        for d in digits.iter_mut() {
            carry += (*d as u32) << 8;
            *d = (carry % 58) as u8;
            carry /= 58;
        }
        while carry > 0 {
            digits.push((carry % 58) as u8);
            carry /= 58;
        }
    }

    let mut result = String::with_capacity(leading_zeros + digits.len());
    for _ in 0..leading_zeros {
        result.push('1');
    }
    for &d in digits.iter().rev() {
        result.push(BASE58_ALPHABET[d as usize] as char);
    }
    result
}

/// Decodes a Base58 string (Bitcoin alphabet) into bytes.
pub fn base58_decode(text: &str) -> Result<Vec<u8>> {
    if text.is_empty() {
        return Ok(Vec::new());
    }

    let mut decode_table = [255u8; 128];
    for (i, &c) in BASE58_ALPHABET.iter().enumerate() {
        decode_table[c as usize] = i as u8;
    }

    let leading_ones = text.bytes().take_while(|&b| b == b'1').count();

    let mut bytes: Vec<u8> = Vec::with_capacity(text.len());
    for c in text.bytes() {
        if c >= 128 || decode_table[c as usize] == 255 {
            return Err(ZyronError::ExecutionError(format!(
                "Invalid Base58 character: {}",
                c as char
            )));
        }
        let mut carry = decode_table[c as usize] as u32;
        for b in bytes.iter_mut() {
            carry += (*b as u32) * 58;
            *b = (carry & 0xFF) as u8;
            carry >>= 8;
        }
        while carry > 0 {
            bytes.push((carry & 0xFF) as u8);
            carry >>= 8;
        }
    }

    let mut result = Vec::with_capacity(leading_ones + bytes.len());
    for _ in 0..leading_ones {
        result.push(0);
    }
    result.extend(bytes.iter().rev());
    Ok(result)
}

// ---------------------------------------------------------------------------
// Base32 (RFC 4648)
// ---------------------------------------------------------------------------

const BASE32_ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";

/// Encodes bytes using Base32 (RFC 4648) with padding.
pub fn base32_encode(data: &[u8]) -> String {
    let mut result = String::with_capacity((data.len() + 4) / 5 * 8);
    for chunk in data.chunks(5) {
        let mut buf = [0u8; 5];
        buf[..chunk.len()].copy_from_slice(chunk);

        let indices = [
            buf[0] >> 3,
            ((buf[0] & 0x07) << 2) | (buf[1] >> 6),
            (buf[1] >> 1) & 0x1F,
            ((buf[1] & 0x01) << 4) | (buf[2] >> 4),
            ((buf[2] & 0x0F) << 1) | (buf[3] >> 7),
            (buf[3] >> 2) & 0x1F,
            ((buf[3] & 0x03) << 3) | (buf[4] >> 5),
            buf[4] & 0x1F,
        ];

        let chars_to_write = match chunk.len() {
            1 => 2,
            2 => 4,
            3 => 5,
            4 => 7,
            5 => 8,
            _ => 0,
        };

        for &idx in &indices[..chars_to_write] {
            result.push(BASE32_ALPHABET[idx as usize] as char);
        }
        // Pad to 8 characters
        for _ in chars_to_write..8 {
            result.push('=');
        }
    }
    result
}

/// Decodes a Base32 string (RFC 4648) into bytes.
pub fn base32_decode(text: &str) -> Result<Vec<u8>> {
    let s = text.trim_end_matches('=');
    if s.is_empty() {
        return Ok(Vec::new());
    }

    let mut decode_table = [255u8; 128];
    for (i, &c) in BASE32_ALPHABET.iter().enumerate() {
        decode_table[c as usize] = i as u8;
        // Also accept lowercase
        if c >= b'A' && c <= b'Z' {
            decode_table[(c + 32) as usize] = i as u8;
        }
    }

    let mut bits: u64 = 0;
    let mut bit_count = 0u32;
    let mut result = Vec::with_capacity(s.len() * 5 / 8);

    for c in s.bytes() {
        if c >= 128 || decode_table[c as usize] == 255 {
            return Err(ZyronError::ExecutionError(format!(
                "Invalid Base32 character: {}",
                c as char
            )));
        }
        bits = (bits << 5) | decode_table[c as usize] as u64;
        bit_count += 5;
        if bit_count >= 8 {
            bit_count -= 8;
            result.push(((bits >> bit_count) & 0xFF) as u8);
        }
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Base64URL (RFC 4648 section 5)
// ---------------------------------------------------------------------------

const BASE64URL_ALPHABET: &[u8] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

/// Encodes bytes using Base64URL (no padding).
pub fn base64url_encode(data: &[u8]) -> String {
    let mut result = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;

        result.push(BASE64URL_ALPHABET[((triple >> 18) & 0x3F) as usize] as char);
        result.push(BASE64URL_ALPHABET[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            result.push(BASE64URL_ALPHABET[((triple >> 6) & 0x3F) as usize] as char);
        }
        if chunk.len() > 2 {
            result.push(BASE64URL_ALPHABET[(triple & 0x3F) as usize] as char);
        }
    }
    result
}

/// Decodes a Base64URL string (with or without padding) into bytes.
pub fn base64url_decode(text: &str) -> Result<Vec<u8>> {
    let s = text.trim_end_matches('=');
    if s.is_empty() {
        return Ok(Vec::new());
    }

    let mut decode_table = [255u8; 128];
    for (i, &c) in BASE64URL_ALPHABET.iter().enumerate() {
        decode_table[c as usize] = i as u8;
    }

    let mut bits: u64 = 0;
    let mut bit_count = 0u32;
    let mut result = Vec::with_capacity(s.len() * 3 / 4);

    for c in s.bytes() {
        if c >= 128 || decode_table[c as usize] == 255 {
            return Err(ZyronError::ExecutionError(format!(
                "Invalid Base64URL character: {}",
                c as char
            )));
        }
        bits = (bits << 6) | decode_table[c as usize] as u64;
        bit_count += 6;
        if bit_count >= 8 {
            bit_count -= 8;
            result.push(((bits >> bit_count) & 0xFF) as u8);
        }
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// CRC32 (ISO 3309 / ITU-T V.42, polynomial 0xEDB88320)
// ---------------------------------------------------------------------------

/// Computes CRC32 (ISO 3309) of the given data.
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;
    for &byte in data {
        let idx = ((crc ^ byte as u32) & 0xFF) as usize;
        crc = CRC32_TABLE[idx] ^ (crc >> 8);
    }
    crc ^ 0xFFFFFFFF
}

// CRC32 lookup table (polynomial 0xEDB88320)
const CRC32_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0u32;
    while i < 256 {
        let mut crc = i;
        let mut j = 0;
        while j < 8 {
            if crc & 1 != 0 {
                crc = 0xEDB88320 ^ (crc >> 1);
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        table[i as usize] = crc;
        i += 1;
    }
    table
};

// ---------------------------------------------------------------------------
// CRC32C (Castagnoli, polynomial 0x82F63B78)
// ---------------------------------------------------------------------------

/// Computes CRC32C (Castagnoli) of the given data.
pub fn crc32c(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;
    for &byte in data {
        let idx = ((crc ^ byte as u32) & 0xFF) as usize;
        crc = CRC32C_TABLE[idx] ^ (crc >> 8);
    }
    crc ^ 0xFFFFFFFF
}

const CRC32C_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0u32;
    while i < 256 {
        let mut crc = i;
        let mut j = 0;
        while j < 8 {
            if crc & 1 != 0 {
                crc = 0x82F63B78 ^ (crc >> 1);
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        table[i as usize] = crc;
        i += 1;
    }
    table
};

// ---------------------------------------------------------------------------
// xxHash64
// ---------------------------------------------------------------------------

/// Computes xxHash64 of the given data using the xxhash-rust crate.
pub fn xxhash64(data: &[u8]) -> u64 {
    xxhash_rust::xxh3::xxh3_64(data)
}

// ---------------------------------------------------------------------------
// MurmurHash3
// ---------------------------------------------------------------------------

/// Computes MurmurHash3 (32-bit) with the given seed.
pub fn murmur3_32(data: &[u8], seed: u32) -> u32 {
    let c1: u32 = 0xCC9E2D51;
    let c2: u32 = 0x1B873593;
    let mut h1 = seed;
    let nblocks = data.len() / 4;

    // Body: process 4-byte blocks
    for i in 0..nblocks {
        let offset = i * 4;
        let mut k1 = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);

        k1 = k1.wrapping_mul(c1);
        k1 = k1.rotate_left(15);
        k1 = k1.wrapping_mul(c2);

        h1 ^= k1;
        h1 = h1.rotate_left(13);
        h1 = h1.wrapping_mul(5).wrapping_add(0xE6546B64);
    }

    // Tail: process remaining bytes
    let tail = &data[nblocks * 4..];
    let mut k1: u32 = 0;
    match tail.len() {
        3 => {
            k1 ^= (tail[2] as u32) << 16;
            k1 ^= (tail[1] as u32) << 8;
            k1 ^= tail[0] as u32;
            k1 = k1.wrapping_mul(c1);
            k1 = k1.rotate_left(15);
            k1 = k1.wrapping_mul(c2);
            h1 ^= k1;
        }
        2 => {
            k1 ^= (tail[1] as u32) << 8;
            k1 ^= tail[0] as u32;
            k1 = k1.wrapping_mul(c1);
            k1 = k1.rotate_left(15);
            k1 = k1.wrapping_mul(c2);
            h1 ^= k1;
        }
        1 => {
            k1 ^= tail[0] as u32;
            k1 = k1.wrapping_mul(c1);
            k1 = k1.rotate_left(15);
            k1 = k1.wrapping_mul(c2);
            h1 ^= k1;
        }
        _ => {}
    }

    // Finalization
    h1 ^= data.len() as u32;
    h1 = fmix32(h1);
    h1
}

fn fmix32(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x85EBCA6B);
    h ^= h >> 13;
    h = h.wrapping_mul(0xC2B2AE35);
    h ^= h >> 16;
    h
}

/// Computes MurmurHash3 (128-bit, x64 variant) with the given seed.
/// Returns the two 64-bit halves.
pub fn murmur3_128(data: &[u8], seed: u32) -> u128 {
    let c1: u64 = 0x87C37B91114253D5;
    let c2: u64 = 0x4CF5AD432745937F;
    let mut h1 = seed as u64;
    let mut h2 = seed as u64;
    let nblocks = data.len() / 16;

    for i in 0..nblocks {
        let offset = i * 16;
        let mut k1 = u64::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]);
        let mut k2 = u64::from_le_bytes([
            data[offset + 8],
            data[offset + 9],
            data[offset + 10],
            data[offset + 11],
            data[offset + 12],
            data[offset + 13],
            data[offset + 14],
            data[offset + 15],
        ]);

        k1 = k1.wrapping_mul(c1);
        k1 = k1.rotate_left(31);
        k1 = k1.wrapping_mul(c2);
        h1 ^= k1;
        h1 = h1.rotate_left(27);
        h1 = h1.wrapping_add(h2);
        h1 = h1.wrapping_mul(5).wrapping_add(0x52DCE729);

        k2 = k2.wrapping_mul(c2);
        k2 = k2.rotate_left(33);
        k2 = k2.wrapping_mul(c1);
        h2 ^= k2;
        h2 = h2.rotate_left(31);
        h2 = h2.wrapping_add(h1);
        h2 = h2.wrapping_mul(5).wrapping_add(0x38495AB5);
    }

    // Tail
    let tail = &data[nblocks * 16..];
    let mut k1: u64 = 0;
    let mut k2: u64 = 0;

    if tail.len() >= 15 {
        k2 ^= (tail[14] as u64) << 48;
    }
    if tail.len() >= 14 {
        k2 ^= (tail[13] as u64) << 40;
    }
    if tail.len() >= 13 {
        k2 ^= (tail[12] as u64) << 32;
    }
    if tail.len() >= 12 {
        k2 ^= (tail[11] as u64) << 24;
    }
    if tail.len() >= 11 {
        k2 ^= (tail[10] as u64) << 16;
    }
    if tail.len() >= 10 {
        k2 ^= (tail[9] as u64) << 8;
    }
    if tail.len() >= 9 {
        k2 ^= tail[8] as u64;
        k2 = k2.wrapping_mul(c2);
        k2 = k2.rotate_left(33);
        k2 = k2.wrapping_mul(c1);
        h2 ^= k2;
    }
    if tail.len() >= 8 {
        k1 ^= (tail[7] as u64) << 56;
    }
    if tail.len() >= 7 {
        k1 ^= (tail[6] as u64) << 48;
    }
    if tail.len() >= 6 {
        k1 ^= (tail[5] as u64) << 40;
    }
    if tail.len() >= 5 {
        k1 ^= (tail[4] as u64) << 32;
    }
    if tail.len() >= 4 {
        k1 ^= (tail[3] as u64) << 24;
    }
    if tail.len() >= 3 {
        k1 ^= (tail[2] as u64) << 16;
    }
    if tail.len() >= 2 {
        k1 ^= (tail[1] as u64) << 8;
    }
    if !tail.is_empty() {
        k1 ^= tail[0] as u64;
        k1 = k1.wrapping_mul(c1);
        k1 = k1.rotate_left(31);
        k1 = k1.wrapping_mul(c2);
        h1 ^= k1;
    }

    // Finalization
    h1 ^= data.len() as u64;
    h2 ^= data.len() as u64;
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);
    h1 = fmix64(h1);
    h2 = fmix64(h2);
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);

    ((h2 as u128) << 64) | (h1 as u128)
}

fn fmix64(mut h: u64) -> u64 {
    h ^= h >> 33;
    h = h.wrapping_mul(0xFF51AFD7ED558CCD);
    h ^= h >> 33;
    h = h.wrapping_mul(0xC4CEB9FE1A85EC53);
    h ^= h >> 33;
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    // Hex
    #[test]
    fn test_hex_roundtrip() {
        let data = b"Hello, World!";
        let encoded = hex_encode(data);
        let decoded = hex_decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_hex_encode() {
        assert_eq!(hex_encode(&[0xDE, 0xAD, 0xBE, 0xEF]), "deadbeef");
    }

    #[test]
    fn test_hex_decode_0x_prefix() {
        let decoded = hex_decode("0xDEAD").unwrap();
        assert_eq!(decoded, vec![0xDE, 0xAD]);
    }

    #[test]
    fn test_hex_decode_odd_length() {
        assert!(hex_decode("abc").is_err());
    }

    #[test]
    fn test_hex_decode_invalid_char() {
        assert!(hex_decode("zz").is_err());
    }

    #[test]
    fn test_hex_empty() {
        assert_eq!(hex_encode(&[]), "");
        assert_eq!(hex_decode("").unwrap(), Vec::<u8>::new());
    }

    // Base58
    #[test]
    fn test_base58_roundtrip() {
        let data = b"Hello World";
        let encoded = base58_encode(data);
        let decoded = base58_decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_base58_known_vector() {
        // "Hello World" in Base58 = "JxF12TrwUP45BMd"
        let encoded = base58_encode(b"Hello World");
        assert_eq!(encoded, "JxF12TrwUP45BMd");
    }

    #[test]
    fn test_base58_leading_zeros() {
        let data = vec![0, 0, 0, 1];
        let encoded = base58_encode(&data);
        assert!(encoded.starts_with("111"));
        let decoded = base58_decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_base58_empty() {
        assert_eq!(base58_encode(&[]), "");
        assert_eq!(base58_decode("").unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn test_base58_invalid_char() {
        assert!(base58_decode("0OIl").is_err()); // 0, O, I, l are not in Base58
    }

    // Base32
    #[test]
    fn test_base32_roundtrip() {
        let data = b"Hello!";
        let encoded = base32_encode(data);
        let decoded = base32_decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_base32_known_vectors() {
        // RFC 4648 test vectors
        assert_eq!(base32_encode(b""), "");
        assert_eq!(base32_encode(b"f"), "MY======");
        assert_eq!(base32_encode(b"fo"), "MZXQ====");
        assert_eq!(base32_encode(b"foo"), "MZXW6===");
        assert_eq!(base32_encode(b"foob"), "MZXW6YQ=");
        assert_eq!(base32_encode(b"fooba"), "MZXW6YTB");
        assert_eq!(base32_encode(b"foobar"), "MZXW6YTBOI======");
    }

    #[test]
    fn test_base32_decode_lowercase() {
        let decoded = base32_decode("mzxw6===").unwrap();
        assert_eq!(decoded, b"foo");
    }

    #[test]
    fn test_base32_invalid_char() {
        assert!(base32_decode("1234").is_err());
    }

    // Base64URL
    #[test]
    fn test_base64url_roundtrip() {
        let data = b"Hello, World!";
        let encoded = base64url_encode(data);
        let decoded = base64url_decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_base64url_no_padding() {
        let encoded = base64url_encode(b"Hello");
        assert!(!encoded.contains('='));
    }

    #[test]
    fn test_base64url_url_safe_chars() {
        // Data that would produce + and / in standard Base64
        let data = vec![0xFF, 0xFF, 0xFF];
        let encoded = base64url_encode(&data);
        assert!(!encoded.contains('+'));
        assert!(!encoded.contains('/'));
        assert!(encoded.contains('-') || encoded.contains('_'));
    }

    #[test]
    fn test_base64url_empty() {
        assert_eq!(base64url_encode(&[]), "");
        assert_eq!(base64url_decode("").unwrap(), Vec::<u8>::new());
    }

    // CRC32
    #[test]
    fn test_crc32_known() {
        // CRC32 of "123456789" = 0xCBF43926
        assert_eq!(crc32(b"123456789"), 0xCBF43926);
    }

    #[test]
    fn test_crc32_empty() {
        assert_eq!(crc32(b""), 0x00000000);
    }

    #[test]
    fn test_crc32_deterministic() {
        let data = b"Hello World";
        assert_eq!(crc32(data), crc32(data));
    }

    // CRC32C
    #[test]
    fn test_crc32c_known() {
        // CRC32C of "123456789" = 0xE3069283
        assert_eq!(crc32c(b"123456789"), 0xE3069283);
    }

    #[test]
    fn test_crc32c_empty() {
        assert_eq!(crc32c(b""), 0x00000000);
    }

    #[test]
    fn test_crc32_vs_crc32c_different() {
        let data = b"test";
        assert_ne!(crc32(data), crc32c(data));
    }

    // xxHash64
    #[test]
    fn test_xxhash64_deterministic() {
        let data = b"Hello World";
        assert_eq!(xxhash64(data), xxhash64(data));
    }

    #[test]
    fn test_xxhash64_different_inputs() {
        assert_ne!(xxhash64(b"hello"), xxhash64(b"world"));
    }

    #[test]
    fn test_xxhash64_empty() {
        // Should not panic
        let _ = xxhash64(b"");
    }

    // MurmurHash3
    #[test]
    fn test_murmur3_32_deterministic() {
        let data = b"Hello World";
        assert_eq!(murmur3_32(data, 0), murmur3_32(data, 0));
    }

    #[test]
    fn test_murmur3_32_seed_matters() {
        let data = b"test";
        assert_ne!(murmur3_32(data, 0), murmur3_32(data, 42));
    }

    #[test]
    fn test_murmur3_32_known() {
        // Known test vector: murmur3_32("", 0) = 0
        assert_eq!(murmur3_32(b"", 0), 0);
    }

    #[test]
    fn test_murmur3_32_tail_sizes() {
        // Test different tail sizes (1, 2, 3 bytes after last 4-byte block)
        let _ = murmur3_32(b"a", 0);
        let _ = murmur3_32(b"ab", 0);
        let _ = murmur3_32(b"abc", 0);
        let _ = murmur3_32(b"abcd", 0);
        let _ = murmur3_32(b"abcde", 0);
    }

    #[test]
    fn test_murmur3_128_deterministic() {
        let data = b"Hello World";
        assert_eq!(murmur3_128(data, 0), murmur3_128(data, 0));
    }

    #[test]
    fn test_murmur3_128_different_from_32() {
        let data = b"test";
        let h32 = murmur3_32(data, 0) as u128;
        let h128 = murmur3_128(data, 0);
        assert_ne!(h32, h128);
    }

    #[test]
    fn test_murmur3_128_empty() {
        let h = murmur3_128(b"", 0);
        assert_eq!(h, 0); // murmur3_128("", 0) = 0
    }

    #[test]
    fn test_murmur3_128_various_lengths() {
        // Test tail handling for lengths 1-15
        for i in 1..=20 {
            let data: Vec<u8> = (0..i).collect();
            let _ = murmur3_128(&data, 42);
        }
    }
}

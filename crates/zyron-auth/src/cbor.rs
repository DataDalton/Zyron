//! Minimal CBOR decoder for WebAuthn COSE key and attestation parsing.
//!
//! Implements a subset of RFC 8949 sufficient for decoding COSE public keys
//! (EC2 and OKP key types) and WebAuthn attestation objects. Supports major
//! types 0-7: unsigned integers, negative integers, byte strings, text strings,
//! arrays, maps, simple values (bool, null), and tagged values (tag is skipped,
//! inner value is returned).

use zyron_common::{Result, ZyronError};

/// A decoded CBOR value.
#[derive(Debug, Clone)]
pub enum CborValue {
    UnsignedInt(u64),
    NegativeInt(i64),
    ByteString(Vec<u8>),
    TextString(String),
    Array(Vec<CborValue>),
    Map(Vec<(CborValue, CborValue)>),
    Bool(bool),
    Null,
}

impl CborValue {
    /// Returns the value as a map slice, if it is a Map.
    pub fn as_map(&self) -> Option<&[(CborValue, CborValue)]> {
        match self {
            CborValue::Map(entries) => Some(entries),
            _ => None,
        }
    }

    /// Returns the value as a byte slice, if it is a ByteString.
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            CborValue::ByteString(b) => Some(b),
            _ => None,
        }
    }

    /// Returns the value as a signed integer (works for both UnsignedInt and NegativeInt).
    pub fn as_int(&self) -> Option<i64> {
        match self {
            CborValue::UnsignedInt(n) => i64::try_from(*n).ok(),
            CborValue::NegativeInt(n) => Some(*n),
            _ => None,
        }
    }

    /// Returns the value as an unsigned integer, if it is an UnsignedInt.
    pub fn as_unsigned(&self) -> Option<u64> {
        match self {
            CborValue::UnsignedInt(n) => Some(*n),
            _ => None,
        }
    }

    /// Returns the value as a string slice, if it is a TextString.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            CborValue::TextString(s) => Some(s),
            _ => None,
        }
    }

    /// Looks up a value in a CBOR map by integer key.
    /// COSE keys use integer map keys (e.g., 1 for kty, 3 for alg, -1 for crv).
    pub fn map_get_int(&self, key: i64) -> Option<&CborValue> {
        let entries = self.as_map()?;
        for (k, v) in entries {
            if k.as_int() == Some(key) {
                return Some(v);
            }
        }
        None
    }
}

/// Maximum nesting depth for CBOR decoding to prevent stack overflow.
const MAX_DECODE_DEPTH: usize = 32;

/// Decodes a single CBOR value from the byte slice.
/// Returns the parsed value and the number of bytes consumed.
pub fn decode(data: &[u8]) -> Result<(CborValue, usize)> {
    decode_with_depth(data, 0)
}

fn decode_with_depth(data: &[u8], depth: usize) -> Result<(CborValue, usize)> {
    if depth > MAX_DECODE_DEPTH {
        return Err(ZyronError::DecodingFailed(
            "CBOR: nesting depth exceeds maximum".to_string(),
        ));
    }
    if data.is_empty() {
        return Err(ZyronError::DecodingFailed("CBOR: empty input".to_string()));
    }

    let initial_byte = data[0];
    let major_type = initial_byte >> 5;
    let additional_info = initial_byte & 0x1f;

    match major_type {
        0 => {
            // Unsigned integer
            let (value, consumed) = decode_uint(data)?;
            Ok((CborValue::UnsignedInt(value), consumed))
        }
        1 => {
            // Negative integer: -1 - n
            let (n, consumed) = decode_uint(data)?;
            let n_i64 = i64::try_from(n).map_err(|_| {
                ZyronError::DecodingFailed(format!(
                    "CBOR: negative integer value {} overflows i64",
                    n
                ))
            })?;
            let value = -1i64 - n_i64;
            Ok((CborValue::NegativeInt(value), consumed))
        }
        2 => {
            // Byte string
            let (len, header_size) = decode_length(data)?;
            let total = header_size + len;
            if data.len() < total {
                return Err(ZyronError::DecodingFailed(format!(
                    "CBOR: byte string needs {} bytes, got {}",
                    total,
                    data.len()
                )));
            }
            let bytes = data[header_size..total].to_vec();
            Ok((CborValue::ByteString(bytes), total))
        }
        3 => {
            // Text string
            let (len, header_size) = decode_length(data)?;
            let total = header_size + len;
            if data.len() < total {
                return Err(ZyronError::DecodingFailed(format!(
                    "CBOR: text string needs {} bytes, got {}",
                    total,
                    data.len()
                )));
            }
            let text = std::str::from_utf8(&data[header_size..total])
                .map_err(|_| {
                    ZyronError::DecodingFailed("CBOR: invalid UTF-8 in text string".to_string())
                })?
                .to_string();
            Ok((CborValue::TextString(text), total))
        }
        4 => {
            // Array
            let (count, mut pos) = decode_length(data)?;
            let mut items = Vec::with_capacity(count.min(64));
            for _ in 0..count {
                let (value, consumed) = decode_with_depth(&data[pos..], depth + 1)?;
                pos += consumed;
                items.push(value);
            }
            Ok((CborValue::Array(items), pos))
        }
        5 => {
            // Map
            let (count, mut pos) = decode_length(data)?;
            let mut entries = Vec::with_capacity(count.min(32));
            for _ in 0..count {
                let (key, kc) = decode_with_depth(&data[pos..], depth + 1)?;
                pos += kc;
                let (val, vc) = decode_with_depth(&data[pos..], depth + 1)?;
                pos += vc;
                entries.push((key, val));
            }
            Ok((CborValue::Map(entries), pos))
        }
        6 => {
            // Tagged value: skip the tag number and decode the inner value
            let (_tag, header_size) = decode_length(data)?;
            let (value, inner_consumed) = decode_with_depth(&data[header_size..], depth + 1)?;
            Ok((value, header_size + inner_consumed))
        }
        7 => {
            // Simple values and floats
            match additional_info {
                20 => Ok((CborValue::Bool(false), 1)),
                21 => Ok((CborValue::Bool(true), 1)),
                22 => Ok((CborValue::Null, 1)),
                _ => Err(ZyronError::DecodingFailed(format!(
                    "CBOR: unsupported simple value {}",
                    additional_info
                ))),
            }
        }
        _ => Err(ZyronError::DecodingFailed(format!(
            "CBOR: unknown major type {}",
            major_type
        ))),
    }
}

/// Decodes the unsigned integer value from the initial byte and following bytes.
/// Returns (value, bytes_consumed).
fn decode_uint(data: &[u8]) -> Result<(u64, usize)> {
    if data.is_empty() {
        return Err(ZyronError::DecodingFailed(
            "CBOR: empty data for uint".to_string(),
        ));
    }
    let additional = data[0] & 0x1f;
    match additional {
        0..=23 => Ok((additional as u64, 1)),
        24 => {
            if data.len() < 2 {
                return Err(ZyronError::DecodingFailed(
                    "CBOR: truncated 1-byte uint".to_string(),
                ));
            }
            Ok((data[1] as u64, 2))
        }
        25 => {
            if data.len() < 3 {
                return Err(ZyronError::DecodingFailed(
                    "CBOR: truncated 2-byte uint".to_string(),
                ));
            }
            Ok((u16::from_be_bytes([data[1], data[2]]) as u64, 3))
        }
        26 => {
            if data.len() < 5 {
                return Err(ZyronError::DecodingFailed(
                    "CBOR: truncated 4-byte uint".to_string(),
                ));
            }
            Ok((
                u32::from_be_bytes([data[1], data[2], data[3], data[4]]) as u64,
                5,
            ))
        }
        27 => {
            if data.len() < 9 {
                return Err(ZyronError::DecodingFailed(
                    "CBOR: truncated 8-byte uint".to_string(),
                ));
            }
            Ok((
                u64::from_be_bytes([
                    data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8],
                ]),
                9,
            ))
        }
        _ => Err(ZyronError::DecodingFailed(format!(
            "CBOR: invalid additional info {} for uint",
            additional
        ))),
    }
}

/// Decodes the length field (used by byte strings, text strings, arrays, maps, tags).
/// Returns (length_value, header_bytes_consumed).
fn decode_length(data: &[u8]) -> Result<(usize, usize)> {
    let (value, consumed) = decode_uint(data)?;
    Ok((value as usize, consumed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_unsigned_small() {
        // CBOR: 0x05 = unsigned int 5
        let (val, consumed) = decode(&[0x05]).expect("decode");
        assert_eq!(val.as_unsigned(), Some(5));
        assert_eq!(consumed, 1);
    }

    #[test]
    fn test_decode_unsigned_one_byte() {
        // CBOR: 0x18 0x64 = unsigned int 100
        let (val, consumed) = decode(&[0x18, 0x64]).expect("decode");
        assert_eq!(val.as_unsigned(), Some(100));
        assert_eq!(consumed, 2);
    }

    #[test]
    fn test_decode_unsigned_two_byte() {
        // CBOR: 0x19 0x01 0x00 = unsigned int 256
        let (val, consumed) = decode(&[0x19, 0x01, 0x00]).expect("decode");
        assert_eq!(val.as_unsigned(), Some(256));
        assert_eq!(consumed, 3);
    }

    #[test]
    fn test_decode_negative_int() {
        // CBOR: 0x20 = negative int -1 (encoded as -1 - 0)
        let (val, consumed) = decode(&[0x20]).expect("decode");
        assert_eq!(val.as_int(), Some(-1));
        assert_eq!(consumed, 1);
    }

    #[test]
    fn test_decode_negative_int_larger() {
        // CBOR: 0x38 0x06 = negative int -7 (encoded as -1 - 6)
        let (val, consumed) = decode(&[0x38, 0x06]).expect("decode");
        assert_eq!(val.as_int(), Some(-7));
        assert_eq!(consumed, 2);
    }

    #[test]
    fn test_decode_byte_string() {
        // CBOR: 0x43 followed by 3 bytes
        let data = [0x43, 0x01, 0x02, 0x03];
        let (val, consumed) = decode(&data).expect("decode");
        assert_eq!(val.as_bytes(), Some(&[0x01, 0x02, 0x03][..]));
        assert_eq!(consumed, 4);
    }

    #[test]
    fn test_decode_text_string() {
        // CBOR: 0x65 "hello"
        let data = [0x65, b'h', b'e', b'l', b'l', b'o'];
        let (val, consumed) = decode(&data).expect("decode");
        assert_eq!(val.as_text(), Some("hello"));
        assert_eq!(consumed, 6);
    }

    #[test]
    fn test_decode_array() {
        // CBOR: [1, 2, 3] = 0x83, 0x01, 0x02, 0x03
        let data = [0x83, 0x01, 0x02, 0x03];
        let (val, consumed) = decode(&data).expect("decode");
        match &val {
            CborValue::Array(items) => {
                assert_eq!(items.len(), 3);
                assert_eq!(items[0].as_unsigned(), Some(1));
                assert_eq!(items[2].as_unsigned(), Some(3));
            }
            _ => panic!("expected array"),
        }
        assert_eq!(consumed, 4);
    }

    #[test]
    fn test_decode_map() {
        // CBOR: {1: 2, 3: 4} = 0xa2, 0x01, 0x02, 0x03, 0x04
        let data = [0xa2, 0x01, 0x02, 0x03, 0x04];
        let (val, consumed) = decode(&data).expect("decode");
        assert_eq!(val.map_get_int(1).and_then(|v| v.as_unsigned()), Some(2));
        assert_eq!(val.map_get_int(3).and_then(|v| v.as_unsigned()), Some(4));
        assert_eq!(consumed, 5);
    }

    #[test]
    fn test_decode_bool_true() {
        let (val, consumed) = decode(&[0xf5]).expect("decode");
        assert!(matches!(val, CborValue::Bool(true)));
        assert_eq!(consumed, 1);
    }

    #[test]
    fn test_decode_bool_false() {
        let (val, consumed) = decode(&[0xf4]).expect("decode");
        assert!(matches!(val, CborValue::Bool(false)));
        assert_eq!(consumed, 1);
    }

    #[test]
    fn test_decode_null() {
        let (val, consumed) = decode(&[0xf6]).expect("decode");
        assert!(matches!(val, CborValue::Null));
        assert_eq!(consumed, 1);
    }

    #[test]
    fn test_decode_empty_input() {
        assert!(decode(&[]).is_err());
    }

    #[test]
    fn test_decode_truncated_byte_string() {
        // Claims 5 bytes but only has 2
        assert!(decode(&[0x45, 0x01, 0x02]).is_err());
    }

    #[test]
    fn test_map_get_int_missing_key() {
        let data = [0xa1, 0x01, 0x02]; // {1: 2}
        let (val, _) = decode(&data).expect("decode");
        assert!(val.map_get_int(99).is_none());
    }

    #[test]
    fn test_map_get_int_negative_key() {
        // COSE uses negative keys: {-1: 1} = 0xa1, 0x20, 0x01
        let data = [0xa1, 0x20, 0x01];
        let (val, _) = decode(&data).expect("decode");
        assert_eq!(val.map_get_int(-1).and_then(|v| v.as_unsigned()), Some(1));
    }

    #[test]
    fn test_decode_cose_like_map() {
        // Simulates a COSE EC2 key structure:
        // {1: 2, 3: -7, -1: 1, -2: bytes(32), -3: bytes(32)}
        let mut data = Vec::new();
        data.push(0xa5); // map of 5 items

        // 1: 2 (kty: EC2)
        data.push(0x01);
        data.push(0x02);

        // 3: -7 (alg: ES256) -> -7 is encoded as 0x26 (major type 1, value 6)
        data.push(0x03);
        data.push(0x26);

        // -1: 1 (crv: P-256) -> -1 is encoded as 0x20
        data.push(0x20);
        data.push(0x01);

        // -2: 32 bytes (x coordinate)
        data.push(0x21); // -2
        data.push(0x58); // byte string, 1-byte length
        data.push(32); // length = 32
        data.extend_from_slice(&[0xAA; 32]);

        // -3: 32 bytes (y coordinate)
        data.push(0x22); // -3
        data.push(0x58);
        data.push(32);
        data.extend_from_slice(&[0xBB; 32]);

        let (val, _) = decode(&data).expect("decode");

        // Verify kty
        assert_eq!(val.map_get_int(1).and_then(|v| v.as_unsigned()), Some(2));
        // Verify alg
        assert_eq!(val.map_get_int(3).and_then(|v| v.as_int()), Some(-7));
        // Verify crv
        assert_eq!(val.map_get_int(-1).and_then(|v| v.as_unsigned()), Some(1));
        // Verify x
        let x = val
            .map_get_int(-2)
            .and_then(|v| v.as_bytes())
            .expect("x bytes");
        assert_eq!(x.len(), 32);
        assert_eq!(x[0], 0xAA);
        // Verify y
        let y = val
            .map_get_int(-3)
            .and_then(|v| v.as_bytes())
            .expect("y bytes");
        assert_eq!(y.len(), 32);
        assert_eq!(y[0], 0xBB);
    }
}

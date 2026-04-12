//! PostgreSQL type OID mapping and value serialization.
//!
//! Maps ZyronDB TypeId values to PostgreSQL type OIDs and provides
//! text/binary format conversion between ScalarValue and wire bytes.

use bytes::{BufMut, BytesMut};

use crate::messages::ProtocolError;
use zyron_common::TypeId;
use zyron_executor::column::ScalarValue;

// PostgreSQL type OID constants.
pub const PG_BOOL_OID: i32 = 16;
pub const PG_BYTEA_OID: i32 = 17;
pub const PG_INT8_OID: i32 = 20; // bigint
pub const PG_INT2_OID: i32 = 21; // smallint
pub const PG_INT4_OID: i32 = 23; // integer
pub const PG_TEXT_OID: i32 = 25;
pub const PG_FLOAT4_OID: i32 = 700;
pub const PG_FLOAT8_OID: i32 = 701;
pub const PG_CHAR_OID: i32 = 1042; // bpchar
pub const PG_VARCHAR_OID: i32 = 1043;
pub const PG_DATE_OID: i32 = 1082;
pub const PG_TIME_OID: i32 = 1083;
pub const PG_TIMESTAMP_OID: i32 = 1114;
pub const PG_TIMESTAMPTZ_OID: i32 = 1184;
pub const PG_INTERVAL_OID: i32 = 1186;
pub const PG_NUMERIC_OID: i32 = 1700;
pub const PG_UUID_OID: i32 = 2950;
pub const PG_JSON_OID: i32 = 114;
pub const PG_JSONB_OID: i32 = 3802;

/// Maps a ZyronDB TypeId to the corresponding PostgreSQL type OID.
pub fn type_id_to_pg_oid(type_id: TypeId) -> i32 {
    match type_id {
        TypeId::Null => 0,
        TypeId::Boolean => PG_BOOL_OID,
        TypeId::Int8 | TypeId::UInt8 => PG_INT2_OID,
        TypeId::Int16 => PG_INT2_OID,
        TypeId::Int32 | TypeId::UInt16 => PG_INT4_OID,
        TypeId::Int64 | TypeId::UInt32 => PG_INT8_OID,
        TypeId::Int128 | TypeId::UInt64 | TypeId::UInt128 => PG_NUMERIC_OID,
        TypeId::Float32 => PG_FLOAT4_OID,
        TypeId::Float64 => PG_FLOAT8_OID,
        TypeId::Decimal => PG_NUMERIC_OID,
        TypeId::Char => PG_CHAR_OID,
        TypeId::Varchar => PG_VARCHAR_OID,
        TypeId::Text => PG_TEXT_OID,
        TypeId::Binary | TypeId::Varbinary | TypeId::Bytea => PG_BYTEA_OID,
        TypeId::Date => PG_DATE_OID,
        TypeId::Time => PG_TIME_OID,
        TypeId::Timestamp => PG_TIMESTAMP_OID,
        TypeId::TimestampTz => PG_TIMESTAMPTZ_OID,
        TypeId::Interval => PG_INTERVAL_OID,
        TypeId::Uuid => PG_UUID_OID,
        TypeId::Json => PG_JSON_OID,
        TypeId::Jsonb => PG_JSONB_OID,
        TypeId::Array | TypeId::Composite => PG_TEXT_OID,
        // Custom OID for vector type (matches pgvector convention).
        TypeId::Vector => 16385,
    }
}

/// Returns the PostgreSQL type size for RowDescription.
/// Fixed-size types return their byte width, variable-length types return -1.
pub fn pg_type_size(type_id: TypeId) -> i16 {
    match type_id {
        TypeId::Boolean => 1,
        TypeId::Int8 | TypeId::UInt8 => 2, // mapped to int2
        TypeId::Int16 => 2,
        TypeId::Int32 | TypeId::UInt16 => 4,
        TypeId::Int64 | TypeId::UInt32 => 8,
        TypeId::Float32 => 4,
        TypeId::Float64 => 8,
        TypeId::Date => 4,
        TypeId::Time | TypeId::Timestamp | TypeId::TimestampTz => 8,
        TypeId::Uuid => 16,
        TypeId::Interval => 16,
        _ => -1, // variable-length
    }
}

/// Formats a float as PostgreSQL text format bytes.
/// Handles NaN/Infinity specially (ryu uses "inf", PostgreSQL uses "Infinity").
#[inline]
fn float_to_pg_vec(v: f64) -> Vec<u8> {
    if v.is_nan() {
        return b"NaN".to_vec();
    }
    if v.is_infinite() {
        return if v.is_sign_positive() {
            b"Infinity".to_vec()
        } else {
            b"-Infinity".to_vec()
        };
    }
    let mut buf = ryu::Buffer::new();
    buf.format(v).as_bytes().to_vec()
}

/// Converts a ScalarValue to its PostgreSQL text format representation.
/// Returns None for SQL NULL. Common types bypass BytesMut and construct
/// Vec directly from stack buffers to avoid per-scalar allocation overhead.
#[inline]
pub fn scalar_to_text(scalar: &ScalarValue) -> Option<Vec<u8>> {
    match scalar {
        ScalarValue::Null => None,
        ScalarValue::Boolean(v) => Some(vec![if *v { b't' } else { b'f' }]),
        ScalarValue::Int8(v) => {
            let mut b = itoa::Buffer::new();
            Some(b.format(*v).as_bytes().to_vec())
        }
        ScalarValue::Int16(v) => {
            let mut b = itoa::Buffer::new();
            Some(b.format(*v).as_bytes().to_vec())
        }
        ScalarValue::Int32(v) => {
            let mut b = itoa::Buffer::new();
            Some(b.format(*v).as_bytes().to_vec())
        }
        ScalarValue::Int64(v) => {
            let mut b = itoa::Buffer::new();
            Some(b.format(*v).as_bytes().to_vec())
        }
        ScalarValue::Int128(v) => {
            let mut b = itoa::Buffer::new();
            Some(b.format(*v).as_bytes().to_vec())
        }
        ScalarValue::UInt8(v) => {
            let mut b = itoa::Buffer::new();
            Some(b.format(*v).as_bytes().to_vec())
        }
        ScalarValue::UInt16(v) => {
            let mut b = itoa::Buffer::new();
            Some(b.format(*v).as_bytes().to_vec())
        }
        ScalarValue::UInt32(v) => {
            let mut b = itoa::Buffer::new();
            Some(b.format(*v).as_bytes().to_vec())
        }
        ScalarValue::UInt64(v) => {
            let mut b = itoa::Buffer::new();
            Some(b.format(*v).as_bytes().to_vec())
        }
        ScalarValue::Float32(v) => Some(float_to_pg_vec(*v as f64)),
        ScalarValue::Float64(v) => Some(float_to_pg_vec(*v)),
        ScalarValue::Utf8(v) => Some(v.clone().into_bytes()),
        _ => {
            let mut buf = BytesMut::with_capacity(32);
            scalar_write_text(scalar, &mut buf);
            Some(buf.into())
        }
    }
}

/// Converts a ScalarValue to PostgreSQL binary wire format.
/// Returns None for SQL NULL. Common fixed-size types construct Vec directly
/// from stack arrays to avoid BytesMut allocation overhead.
#[inline]
pub fn scalar_to_binary(scalar: &ScalarValue) -> Option<Vec<u8>> {
    match scalar {
        ScalarValue::Null => None,
        ScalarValue::Boolean(v) => Some(vec![if *v { 1 } else { 0 }]),
        ScalarValue::Int8(v) => Some((*v as i16).to_be_bytes().to_vec()),
        ScalarValue::Int16(v) => Some(v.to_be_bytes().to_vec()),
        ScalarValue::Int32(v) => Some(v.to_be_bytes().to_vec()),
        ScalarValue::Int64(v) => Some(v.to_be_bytes().to_vec()),
        ScalarValue::UInt8(v) => Some((*v as i16).to_be_bytes().to_vec()),
        ScalarValue::UInt16(v) => Some((*v as i32).to_be_bytes().to_vec()),
        ScalarValue::UInt32(v) => Some((*v as i64).to_be_bytes().to_vec()),
        ScalarValue::Float32(v) => Some(v.to_be_bytes().to_vec()),
        ScalarValue::Float64(v) => Some(v.to_be_bytes().to_vec()),
        ScalarValue::Utf8(v) => Some(v.as_bytes().to_vec()),
        _ => {
            let mut buf = BytesMut::with_capacity(16);
            scalar_write_binary(scalar, &mut buf);
            Some(buf.into())
        }
    }
}

/// Writes a ScalarValue in PG text format directly into the buffer.
/// Returns false for NULL (caller should write -1 length), true for non-NULL.
/// Zero heap allocations for all types.
pub fn scalar_write_text(scalar: &ScalarValue, buf: &mut BytesMut) -> bool {
    match scalar {
        ScalarValue::Null => false,
        ScalarValue::Boolean(v) => {
            buf.put_u8(if *v { b't' } else { b'f' });
            true
        }
        ScalarValue::Int8(v) => {
            let mut b = itoa::Buffer::new();
            buf.extend_from_slice(b.format(*v).as_bytes());
            true
        }
        ScalarValue::Int16(v) => {
            let mut b = itoa::Buffer::new();
            buf.extend_from_slice(b.format(*v).as_bytes());
            true
        }
        ScalarValue::Int32(v) => {
            let mut b = itoa::Buffer::new();
            buf.extend_from_slice(b.format(*v).as_bytes());
            true
        }
        ScalarValue::Int64(v) => {
            let mut b = itoa::Buffer::new();
            buf.extend_from_slice(b.format(*v).as_bytes());
            true
        }
        ScalarValue::Int128(v) => {
            let mut b = itoa::Buffer::new();
            buf.extend_from_slice(b.format(*v).as_bytes());
            true
        }
        ScalarValue::UInt8(v) => {
            let mut b = itoa::Buffer::new();
            buf.extend_from_slice(b.format(*v).as_bytes());
            true
        }
        ScalarValue::UInt16(v) => {
            let mut b = itoa::Buffer::new();
            buf.extend_from_slice(b.format(*v).as_bytes());
            true
        }
        ScalarValue::UInt32(v) => {
            let mut b = itoa::Buffer::new();
            buf.extend_from_slice(b.format(*v).as_bytes());
            true
        }
        ScalarValue::UInt64(v) => {
            let mut b = itoa::Buffer::new();
            buf.extend_from_slice(b.format(*v).as_bytes());
            true
        }
        ScalarValue::Float32(v) => {
            write_float(*v as f64, buf);
            true
        }
        ScalarValue::Float64(v) => {
            write_float(*v, buf);
            true
        }
        ScalarValue::Utf8(v) => {
            buf.extend_from_slice(v.as_bytes());
            true
        }
        ScalarValue::Binary(v) => {
            write_bytea_hex(v, buf);
            true
        }
        ScalarValue::FixedBinary16(v) => {
            write_uuid(v, buf);
            true
        }
    }
}

/// Writes a ScalarValue in PG binary format directly into the buffer.
/// Returns false for NULL, true for non-NULL.
/// Zero heap allocations for all types.
pub fn scalar_write_binary(scalar: &ScalarValue, buf: &mut BytesMut) -> bool {
    match scalar {
        ScalarValue::Null => false,
        ScalarValue::Boolean(v) => {
            buf.put_u8(if *v { 1 } else { 0 });
            true
        }
        ScalarValue::Int8(v) => {
            buf.put_i16(*v as i16);
            true
        }
        ScalarValue::Int16(v) => {
            buf.put_i16(*v);
            true
        }
        ScalarValue::Int32(v) => {
            buf.put_i32(*v);
            true
        }
        ScalarValue::Int64(v) => {
            buf.put_i64(*v);
            true
        }
        ScalarValue::Int128(v) => {
            // Numeric types sent as text representation in binary mode.
            let mut b = itoa::Buffer::new();
            buf.extend_from_slice(b.format(*v).as_bytes());
            true
        }
        ScalarValue::UInt8(v) => {
            buf.put_i16(*v as i16);
            true
        }
        ScalarValue::UInt16(v) => {
            buf.put_i32(*v as i32);
            true
        }
        ScalarValue::UInt32(v) => {
            buf.put_i64(*v as i64);
            true
        }
        ScalarValue::UInt64(v) => {
            let mut b = itoa::Buffer::new();
            buf.extend_from_slice(b.format(*v).as_bytes());
            true
        }
        ScalarValue::Float32(v) => {
            buf.put_f32(*v);
            true
        }
        ScalarValue::Float64(v) => {
            buf.put_f64(*v);
            true
        }
        ScalarValue::Utf8(v) => {
            buf.extend_from_slice(v.as_bytes());
            true
        }
        ScalarValue::Binary(v) => {
            buf.extend_from_slice(v);
            true
        }
        ScalarValue::FixedBinary16(v) => {
            buf.extend_from_slice(v);
            true
        }
    }
}

/// Writes a float value in PG text format directly into the buffer.
/// Uses ryu for fast formatting with zero heap allocations.
fn write_float(v: f64, buf: &mut BytesMut) {
    if v.is_nan() {
        buf.extend_from_slice(b"NaN");
        return;
    }
    if v.is_infinite() {
        if v.is_sign_positive() {
            buf.extend_from_slice(b"Infinity");
        } else {
            buf.extend_from_slice(b"-Infinity");
        }
        return;
    }
    let mut ryu_buf = ryu::Buffer::new();
    let s = ryu_buf.format(v);
    buf.extend_from_slice(s.as_bytes());
    // ryu always includes a decimal point or exponent for non-integer values,
    // but integer-valued floats like 1.0 may be formatted as "1.0" by ryu.
}

/// Writes UUID bytes as "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx" directly into buf.
/// Uses a stack buffer with hex nibble lookup. Zero heap allocations.
/// Formats UUID bytes into a 36-byte stack buffer (xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx).
fn format_uuid_stack(bytes: &[u8; 16], out: &mut [u8; 36]) {
    let mut pos = 0;
    for (i, &b) in bytes.iter().enumerate() {
        if i == 4 || i == 6 || i == 8 || i == 10 {
            out[pos] = b'-';
            pos += 1;
        }
        out[pos] = HEX_CHARS[(b >> 4) as usize];
        out[pos + 1] = HEX_CHARS[(b & 0x0f) as usize];
        pos += 2;
    }
}

fn write_uuid(bytes: &[u8; 16], buf: &mut BytesMut) {
    let mut out = [0u8; 36];
    format_uuid_stack(bytes, &mut out);
    buf.extend_from_slice(&out);
}

/// Writes bytea value as hex format (\\x followed by hex pairs) directly into buf.
/// Processes 8 bytes at a time to reduce extend_from_slice call count.
fn write_bytea_hex(bytes: &[u8], buf: &mut BytesMut) {
    buf.reserve(2 + bytes.len() * 2);
    buf.extend_from_slice(b"\\x");

    let mut hex = [0u8; 16];
    for chunk in bytes.chunks(8) {
        for (i, &b) in chunk.iter().enumerate() {
            hex[i * 2] = HEX_CHARS[(b >> 4) as usize];
            hex[i * 2 + 1] = HEX_CHARS[(b & 0x0f) as usize];
        }
        buf.extend_from_slice(&hex[..chunk.len() * 2]);
    }
}

/// Writes a vector value in text format as bracket notation: [0.1,0.2,0.3].
/// The input bytes are raw little-endian f32 values from the storage layer.
/// A trailing partial f32 (bytes.len() % 4 != 0) is ignored rather than
/// panicking so a corrupt row can't crash the wire layer.
pub fn write_vector_text(bytes: &[u8], buf: &mut BytesMut) {
    buf.put_u8(b'[');
    let mut rb = ryu::Buffer::new();
    let mut first = true;
    for chunk in bytes.chunks_exact(4) {
        if !first {
            buf.put_u8(b',');
        }
        first = false;
        let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        buf.extend_from_slice(rb.format(val as f64).as_bytes());
    }
    buf.put_u8(b']');
}

/// Writes a vector value in binary format: u16 dimension count followed by
/// big-endian f32 values (matching pgvector binary format). Partial trailing
/// bytes are ignored.
pub fn write_vector_binary(bytes: &[u8], buf: &mut BytesMut) {
    let dims = (bytes.len() / 4) as u16;
    buf.put_u16(dims);
    for chunk in bytes.chunks_exact(4) {
        let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        buf.put_f32(val);
    }
}

/// Parses a text-format parameter value into a ScalarValue.
pub fn text_to_scalar(bytes: &[u8], type_oid: i32) -> Result<ScalarValue, ProtocolError> {
    let text = std::str::from_utf8(bytes)
        .map_err(|e| ProtocolError::Malformed(format!("Invalid UTF-8 in parameter: {}", e)))?;

    match type_oid {
        0 => Ok(ScalarValue::Utf8(text.to_string())),
        PG_BOOL_OID => match text {
            "t" | "true" | "TRUE" | "1" | "yes" | "on" => Ok(ScalarValue::Boolean(true)),
            "f" | "false" | "FALSE" | "0" | "no" | "off" => Ok(ScalarValue::Boolean(false)),
            _ => Err(ProtocolError::Malformed(format!(
                "Invalid boolean: {}",
                text
            ))),
        },
        PG_INT2_OID => text
            .parse::<i16>()
            .map(ScalarValue::Int16)
            .map_err(|e| ProtocolError::Malformed(format!("Invalid int2: {}", e))),
        PG_INT4_OID => text
            .parse::<i32>()
            .map(ScalarValue::Int32)
            .map_err(|e| ProtocolError::Malformed(format!("Invalid int4: {}", e))),
        PG_INT8_OID => text
            .parse::<i64>()
            .map(ScalarValue::Int64)
            .map_err(|e| ProtocolError::Malformed(format!("Invalid int8: {}", e))),
        PG_FLOAT4_OID => text
            .parse::<f32>()
            .map(ScalarValue::Float32)
            .map_err(|e| ProtocolError::Malformed(format!("Invalid float4: {}", e))),
        PG_FLOAT8_OID => text
            .parse::<f64>()
            .map(ScalarValue::Float64)
            .map_err(|e| ProtocolError::Malformed(format!("Invalid float8: {}", e))),
        PG_NUMERIC_OID => {
            // Try integer first, then fall back to float
            if let Ok(v) = text.parse::<i64>() {
                Ok(ScalarValue::Int64(v))
            } else if let Ok(v) = text.parse::<i128>() {
                Ok(ScalarValue::Int128(v))
            } else if let Ok(v) = text.parse::<f64>() {
                Ok(ScalarValue::Float64(v))
            } else {
                Err(ProtocolError::Malformed(format!(
                    "Invalid numeric: {}",
                    text
                )))
            }
        }
        PG_TEXT_OID | PG_VARCHAR_OID | PG_CHAR_OID => Ok(ScalarValue::Utf8(text.to_string())),
        PG_BYTEA_OID => Ok(ScalarValue::Binary(hex_to_bytea(text)?)),
        PG_UUID_OID => {
            let uuid_bytes = parse_uuid(text)?;
            Ok(ScalarValue::FixedBinary16(uuid_bytes))
        }
        PG_JSON_OID | PG_JSONB_OID => Ok(ScalarValue::Utf8(text.to_string())),
        PG_DATE_OID | PG_TIME_OID | PG_TIMESTAMP_OID | PG_TIMESTAMPTZ_OID | PG_INTERVAL_OID => {
            // Store temporal types as strings for now.
            // Full temporal type handling requires date/time parsing.
            Ok(ScalarValue::Utf8(text.to_string()))
        }
        _ => Ok(ScalarValue::Utf8(text.to_string())),
    }
}

/// Parses a binary-format parameter value into a ScalarValue.
pub fn binary_to_scalar(bytes: &[u8], type_oid: i32) -> Result<ScalarValue, ProtocolError> {
    match type_oid {
        PG_BOOL_OID => {
            if bytes.len() != 1 {
                return Err(ProtocolError::Malformed("Bool requires 1 byte".into()));
            }
            Ok(ScalarValue::Boolean(bytes[0] != 0))
        }
        PG_INT2_OID => {
            if bytes.len() != 2 {
                return Err(ProtocolError::Malformed("Int2 requires 2 bytes".into()));
            }
            Ok(ScalarValue::Int16(i16::from_be_bytes([bytes[0], bytes[1]])))
        }
        PG_INT4_OID => {
            if bytes.len() != 4 {
                return Err(ProtocolError::Malformed("Int4 requires 4 bytes".into()));
            }
            Ok(ScalarValue::Int32(i32::from_be_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3],
            ])))
        }
        PG_INT8_OID => {
            if bytes.len() != 8 {
                return Err(ProtocolError::Malformed("Int8 requires 8 bytes".into()));
            }
            Ok(ScalarValue::Int64(i64::from_be_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            ])))
        }
        PG_FLOAT4_OID => {
            if bytes.len() != 4 {
                return Err(ProtocolError::Malformed("Float4 requires 4 bytes".into()));
            }
            Ok(ScalarValue::Float32(f32::from_be_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3],
            ])))
        }
        PG_FLOAT8_OID => {
            if bytes.len() != 8 {
                return Err(ProtocolError::Malformed("Float8 requires 8 bytes".into()));
            }
            Ok(ScalarValue::Float64(f64::from_be_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            ])))
        }
        PG_TEXT_OID | PG_VARCHAR_OID | PG_CHAR_OID | PG_JSON_OID | PG_JSONB_OID => {
            let s = std::str::from_utf8(bytes)
                .map_err(|e| ProtocolError::Malformed(format!("Invalid UTF-8: {}", e)))?;
            Ok(ScalarValue::Utf8(s.to_string()))
        }
        PG_BYTEA_OID => Ok(ScalarValue::Binary(bytes.to_vec())),
        PG_UUID_OID => {
            if bytes.len() != 16 {
                return Err(ProtocolError::Malformed("UUID requires 16 bytes".into()));
            }
            let mut arr = [0u8; 16];
            arr.copy_from_slice(bytes);
            Ok(ScalarValue::FixedBinary16(arr))
        }
        _ => {
            // Fall back to text interpretation. Validate UTF-8 in-place, then allocate once.
            let s = std::str::from_utf8(bytes)
                .map_err(|e| ProtocolError::Malformed(format!("Invalid UTF-8: {}", e)))?;
            Ok(ScalarValue::Utf8(s.to_string()))
        }
    }
}

/// Parses a UUID string into 16 bytes. Accepts "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx".
/// Parses hex directly without intermediate String allocation.
fn parse_uuid(s: &str) -> Result<[u8; 16], ProtocolError> {
    let b = s.as_bytes();
    let mut bytes = [0u8; 16];
    let mut bi = 0; // byte index into output
    let mut i = 0; // index into input

    while i < b.len() && bi < 16 {
        if b[i] == b'-' {
            i += 1;
            continue;
        }
        if i + 1 >= b.len() {
            return Err(ProtocolError::Malformed(
                "Invalid UUID: truncated hex pair".into(),
            ));
        }
        let hi = hex_nibble(b[i])
            .ok_or_else(|| ProtocolError::Malformed("Invalid UUID: bad hex digit".into()))?;
        let lo = hex_nibble(b[i + 1])
            .ok_or_else(|| ProtocolError::Malformed("Invalid UUID: bad hex digit".into()))?;
        bytes[bi] = (hi << 4) | lo;
        bi += 1;
        i += 2;
    }

    if bi != 16 {
        return Err(ProtocolError::Malformed(
            "Invalid UUID: wrong length".into(),
        ));
    }
    Ok(bytes)
}

/// Converts a hex ASCII byte to its nibble value (0-15), or None if invalid.
#[inline]
fn hex_nibble(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

const HEX_CHARS: [u8; 16] = *b"0123456789abcdef";

/// Parses hex-encoded bytea (\\xABCD or \xABCD) back to bytes.
fn hex_to_bytea(text: &str) -> Result<Vec<u8>, ProtocolError> {
    let hex = text
        .strip_prefix("\\x")
        .or_else(|| text.strip_prefix("\\\\x"))
        .unwrap_or(text);

    if hex.len() % 2 != 0 {
        return Err(ProtocolError::Malformed(
            "Odd-length hex string for bytea".into(),
        ));
    }

    let mut bytes = Vec::with_capacity(hex.len() / 2);
    for i in (0..hex.len()).step_by(2) {
        let byte = u8::from_str_radix(&hex[i..i + 2], 16)
            .map_err(|_| ProtocolError::Malformed("Invalid hex in bytea".into()))?;
        bytes.push(byte);
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test helper: converts bytea to hex format for roundtrip testing.
    fn bytea_to_hex(bytes: &[u8]) -> Vec<u8> {
        let mut result = Vec::with_capacity(2 + bytes.len() * 2);
        result.extend_from_slice(b"\\x");
        for b in bytes {
            result.push(HEX_CHARS[(b >> 4) as usize]);
            result.push(HEX_CHARS[(b & 0x0f) as usize]);
        }
        result
    }

    /// Test helper: formats UUID from 16 bytes to string.
    fn format_uuid(bytes: &[u8; 16]) -> String {
        let mut out = [0u8; 36];
        format_uuid_stack(bytes, &mut out);
        String::from_utf8(out.to_vec()).unwrap()
    }

    /// Test helper: formats float matching PostgreSQL output.
    fn format_float(v: f64) -> String {
        if v.is_nan() {
            return "NaN".to_string();
        }
        if v.is_infinite() {
            return if v.is_sign_positive() {
                "Infinity".to_string()
            } else {
                "-Infinity".to_string()
            };
        }
        let mut buf = ryu::Buffer::new();
        buf.format(v).to_string()
    }

    #[test]
    fn test_type_id_to_pg_oid() {
        assert_eq!(type_id_to_pg_oid(TypeId::Boolean), PG_BOOL_OID);
        assert_eq!(type_id_to_pg_oid(TypeId::Int32), PG_INT4_OID);
        assert_eq!(type_id_to_pg_oid(TypeId::Int64), PG_INT8_OID);
        assert_eq!(type_id_to_pg_oid(TypeId::Float64), PG_FLOAT8_OID);
        assert_eq!(type_id_to_pg_oid(TypeId::Varchar), PG_VARCHAR_OID);
        assert_eq!(type_id_to_pg_oid(TypeId::Text), PG_TEXT_OID);
        assert_eq!(type_id_to_pg_oid(TypeId::Uuid), PG_UUID_OID);
        assert_eq!(type_id_to_pg_oid(TypeId::Json), PG_JSON_OID);
        assert_eq!(type_id_to_pg_oid(TypeId::Jsonb), PG_JSONB_OID);
        assert_eq!(type_id_to_pg_oid(TypeId::Bytea), PG_BYTEA_OID);
        assert_eq!(type_id_to_pg_oid(TypeId::Null), 0);
    }

    #[test]
    fn test_unsigned_type_mapping() {
        assert_eq!(type_id_to_pg_oid(TypeId::UInt8), PG_INT2_OID);
        assert_eq!(type_id_to_pg_oid(TypeId::UInt16), PG_INT4_OID);
        assert_eq!(type_id_to_pg_oid(TypeId::UInt32), PG_INT8_OID);
        assert_eq!(type_id_to_pg_oid(TypeId::UInt64), PG_NUMERIC_OID);
        assert_eq!(type_id_to_pg_oid(TypeId::UInt128), PG_NUMERIC_OID);
    }

    #[test]
    fn test_pg_type_size() {
        assert_eq!(pg_type_size(TypeId::Boolean), 1);
        assert_eq!(pg_type_size(TypeId::Int16), 2);
        assert_eq!(pg_type_size(TypeId::Int32), 4);
        assert_eq!(pg_type_size(TypeId::Int64), 8);
        assert_eq!(pg_type_size(TypeId::Float32), 4);
        assert_eq!(pg_type_size(TypeId::Float64), 8);
        assert_eq!(pg_type_size(TypeId::Uuid), 16);
        assert_eq!(pg_type_size(TypeId::Varchar), -1);
        assert_eq!(pg_type_size(TypeId::Text), -1);
    }

    #[test]
    fn test_scalar_to_text_integers() {
        assert_eq!(
            scalar_to_text(&ScalarValue::Int32(42)),
            Some(b"42".to_vec())
        );
        assert_eq!(
            scalar_to_text(&ScalarValue::Int64(-100)),
            Some(b"-100".to_vec())
        );
        assert_eq!(
            scalar_to_text(&ScalarValue::Int128(999999999999)),
            Some(b"999999999999".to_vec())
        );
    }

    #[test]
    fn test_scalar_to_text_boolean() {
        assert_eq!(
            scalar_to_text(&ScalarValue::Boolean(true)),
            Some(b"t".to_vec())
        );
        assert_eq!(
            scalar_to_text(&ScalarValue::Boolean(false)),
            Some(b"f".to_vec())
        );
    }

    #[test]
    fn test_scalar_to_text_null() {
        assert_eq!(scalar_to_text(&ScalarValue::Null), None);
    }

    #[test]
    fn test_scalar_to_text_string() {
        assert_eq!(
            scalar_to_text(&ScalarValue::Utf8("hello".into())),
            Some(b"hello".to_vec())
        );
    }

    #[test]
    fn test_scalar_to_text_float() {
        let result = scalar_to_text(&ScalarValue::Float64(3.14));
        assert!(result.is_some());
        let s = String::from_utf8(result.unwrap()).unwrap();
        assert!(s.starts_with("3.14"));
    }

    #[test]
    fn test_scalar_to_text_float_special() {
        assert_eq!(
            scalar_to_text(&ScalarValue::Float64(f64::NAN)),
            Some(b"NaN".to_vec())
        );
        assert_eq!(
            scalar_to_text(&ScalarValue::Float64(f64::INFINITY)),
            Some(b"Infinity".to_vec())
        );
        assert_eq!(
            scalar_to_text(&ScalarValue::Float64(f64::NEG_INFINITY)),
            Some(b"-Infinity".to_vec())
        );
    }

    #[test]
    fn test_scalar_to_text_uuid() {
        let uuid = [
            0x55, 0x0e, 0x84, 0x00, 0xe2, 0x9b, 0x41, 0xd4, 0xa7, 0x16, 0x44, 0x66, 0x55, 0x44,
            0x00, 0x00,
        ];
        let result = scalar_to_text(&ScalarValue::FixedBinary16(uuid)).unwrap();
        let s = String::from_utf8(result).unwrap();
        assert_eq!(s, "550e8400-e29b-41d4-a716-446655440000");
    }

    #[test]
    fn test_scalar_to_text_bytea() {
        let result = scalar_to_text(&ScalarValue::Binary(vec![0xDE, 0xAD, 0xBE, 0xEF])).unwrap();
        assert_eq!(result, b"\\xdeadbeef");
    }

    #[test]
    fn test_scalar_to_binary_int32() {
        let result = scalar_to_binary(&ScalarValue::Int32(42)).unwrap();
        assert_eq!(result, 42_i32.to_be_bytes());
    }

    #[test]
    fn test_scalar_to_binary_int64() {
        let result = scalar_to_binary(&ScalarValue::Int64(-1)).unwrap();
        assert_eq!(result, (-1_i64).to_be_bytes());
    }

    #[test]
    fn test_scalar_to_binary_bool() {
        assert_eq!(
            scalar_to_binary(&ScalarValue::Boolean(true)).unwrap(),
            vec![1]
        );
        assert_eq!(
            scalar_to_binary(&ScalarValue::Boolean(false)).unwrap(),
            vec![0]
        );
    }

    #[test]
    fn test_scalar_to_binary_null() {
        assert_eq!(scalar_to_binary(&ScalarValue::Null), None);
    }

    #[test]
    fn test_text_to_scalar_int() {
        let result = text_to_scalar(b"42", PG_INT4_OID).unwrap();
        assert_eq!(result, ScalarValue::Int32(42));
    }

    #[test]
    fn test_text_to_scalar_bool() {
        assert_eq!(
            text_to_scalar(b"true", PG_BOOL_OID).unwrap(),
            ScalarValue::Boolean(true)
        );
        assert_eq!(
            text_to_scalar(b"f", PG_BOOL_OID).unwrap(),
            ScalarValue::Boolean(false)
        );
    }

    #[test]
    fn test_text_to_scalar_text() {
        let result = text_to_scalar(b"hello", PG_TEXT_OID).unwrap();
        assert_eq!(result, ScalarValue::Utf8("hello".into()));
    }

    #[test]
    fn test_text_to_scalar_uuid() {
        let result = text_to_scalar(b"550e8400-e29b-41d4-a716-446655440000", PG_UUID_OID).unwrap();
        match result {
            ScalarValue::FixedBinary16(bytes) => {
                assert_eq!(bytes[0], 0x55);
                assert_eq!(bytes[3], 0x00);
            }
            _ => panic!("Expected FixedBinary16"),
        }
    }

    #[test]
    fn test_binary_to_scalar_int32() {
        let result = binary_to_scalar(&42_i32.to_be_bytes(), PG_INT4_OID).unwrap();
        assert_eq!(result, ScalarValue::Int32(42));
    }

    #[test]
    fn test_binary_to_scalar_wrong_size() {
        let result = binary_to_scalar(&[0, 0, 0], PG_INT4_OID);
        assert!(result.is_err());
    }

    #[test]
    fn test_format_uuid() {
        let bytes = [
            0x55, 0x0e, 0x84, 0x00, 0xe2, 0x9b, 0x41, 0xd4, 0xa7, 0x16, 0x44, 0x66, 0x55, 0x44,
            0x00, 0x00,
        ];
        assert_eq!(format_uuid(&bytes), "550e8400-e29b-41d4-a716-446655440000");
    }

    #[test]
    fn test_parse_uuid() {
        let bytes = parse_uuid("550e8400-e29b-41d4-a716-446655440000").unwrap();
        assert_eq!(bytes[0], 0x55);
        assert_eq!(bytes[4], 0xe2);
    }

    #[test]
    fn test_bytea_roundtrip() {
        let original = vec![0xDE, 0xAD, 0xBE, 0xEF];
        let hex = bytea_to_hex(&original);
        let hex_str = std::str::from_utf8(&hex).unwrap();
        let recovered = hex_to_bytea(hex_str).unwrap();
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_format_float_integer() {
        // Integer-valued floats should have ".0"
        assert_eq!(format_float(1.0), "1.0");
        assert_eq!(format_float(0.0), "0.0");
    }

    #[test]
    fn test_format_float_decimal() {
        let s = format_float(3.14);
        assert!(s.contains("3.14"));
    }

    #[test]
    fn test_text_to_scalar_invalid_int() {
        let result = text_to_scalar(b"not_a_number", PG_INT4_OID);
        assert!(result.is_err());
    }
}

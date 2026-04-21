// -----------------------------------------------------------------------------
// Shared type mapping helpers
// -----------------------------------------------------------------------------
//
// Conversion routines used by multiple format implementations. These cover
// TypeId to Arrow DataType mapping, and StreamValue to and from serde_json
// Value coercion. Binary payloads move through JSON as base64 strings,
// Int128 values move as decimal strings to avoid JSON number precision loss.

use crate::row_codec::StreamValue;
use arrow::datatypes::{DataType as ArrowDataType, TimeUnit};
use zyron_common::{Result, TypeId, ZyronError};

// -----------------------------------------------------------------------------
// TypeId to Arrow
// -----------------------------------------------------------------------------

/// Maps a ZyronDB TypeId onto an Arrow DataType. Types with no direct Arrow
/// analogue fall back to Binary or Utf8 as appropriate. Decimal and Int128
/// use Decimal128 with precision 38 and scale 0, a reasonable default.
pub fn type_id_to_arrow(t: TypeId) -> ArrowDataType {
    match t {
        TypeId::Null => ArrowDataType::Null,
        TypeId::Boolean => ArrowDataType::Boolean,
        TypeId::Int8 => ArrowDataType::Int8,
        TypeId::Int16 => ArrowDataType::Int16,
        TypeId::Int32 => ArrowDataType::Int32,
        TypeId::Int64 => ArrowDataType::Int64,
        TypeId::Int128 | TypeId::Decimal | TypeId::UInt128 => ArrowDataType::Decimal128(38, 0),
        TypeId::UInt8 => ArrowDataType::UInt8,
        TypeId::UInt16 => ArrowDataType::UInt16,
        TypeId::UInt32 => ArrowDataType::UInt32,
        TypeId::UInt64 => ArrowDataType::UInt64,
        TypeId::Float32 => ArrowDataType::Float32,
        TypeId::Float64 => ArrowDataType::Float64,
        TypeId::Char | TypeId::Varchar | TypeId::Text | TypeId::Json | TypeId::Jsonb => {
            ArrowDataType::Utf8
        }
        TypeId::Binary | TypeId::Varbinary | TypeId::Bytea => ArrowDataType::Binary,
        TypeId::Date => ArrowDataType::Date32,
        TypeId::Time => ArrowDataType::Time64(TimeUnit::Microsecond),
        TypeId::Timestamp => ArrowDataType::Timestamp(TimeUnit::Microsecond, None),
        TypeId::TimestampTz => ArrowDataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
        TypeId::Uuid => ArrowDataType::FixedSizeBinary(16),
        TypeId::Interval => ArrowDataType::FixedSizeBinary(16),
        TypeId::Array | TypeId::Composite | TypeId::Vector => ArrowDataType::Binary,
        // Extended types fall back to Binary for the variable length cases
        // and FixedSizeBinary for the fixed width cases.
        TypeId::Geometry
        | TypeId::Matrix
        | TypeId::Range
        | TypeId::HyperLogLog
        | TypeId::BloomFilter
        | TypeId::TDigest
        | TypeId::CountMinSketch => ArrowDataType::Binary,
        TypeId::Color => ArrowDataType::FixedSizeBinary(4),
        TypeId::SemVer | TypeId::Money | TypeId::Bitfield | TypeId::Quantity => {
            ArrowDataType::FixedSizeBinary(8)
        }
        TypeId::Inet | TypeId::Cidr | TypeId::MacAddr => ArrowDataType::Binary,
    }
}

// -----------------------------------------------------------------------------
// Arrow to TypeId
// -----------------------------------------------------------------------------

/// Maps an Arrow DataType back to a ZyronDB TypeId. Returns an error for
/// Arrow types that have no direct Zyron equivalent. Used when inferring a
/// schema from a Parquet or Arrow IPC file.
pub fn arrow_to_type_id(dt: &ArrowDataType) -> Result<TypeId> {
    let t = match dt {
        ArrowDataType::Null => TypeId::Null,
        ArrowDataType::Boolean => TypeId::Boolean,
        ArrowDataType::Int8 => TypeId::Int8,
        ArrowDataType::Int16 => TypeId::Int16,
        ArrowDataType::Int32 => TypeId::Int32,
        ArrowDataType::Int64 => TypeId::Int64,
        ArrowDataType::UInt8 => TypeId::UInt8,
        ArrowDataType::UInt16 => TypeId::UInt16,
        ArrowDataType::UInt32 => TypeId::UInt32,
        ArrowDataType::UInt64 => TypeId::UInt64,
        ArrowDataType::Float16 | ArrowDataType::Float32 => TypeId::Float32,
        ArrowDataType::Float64 => TypeId::Float64,
        ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View => TypeId::Text,
        ArrowDataType::Binary
        | ArrowDataType::LargeBinary
        | ArrowDataType::BinaryView
        | ArrowDataType::FixedSizeBinary(_) => TypeId::Binary,
        ArrowDataType::Date32 | ArrowDataType::Date64 => TypeId::Date,
        ArrowDataType::Time32(_) | ArrowDataType::Time64(_) => TypeId::Time,
        ArrowDataType::Timestamp(_, None) => TypeId::Timestamp,
        ArrowDataType::Timestamp(_, Some(_)) => TypeId::TimestampTz,
        ArrowDataType::Duration(_) | ArrowDataType::Interval(_) => TypeId::Interval,
        ArrowDataType::Decimal128(_, _) => TypeId::Decimal,
        ArrowDataType::Decimal256(precision, scale) => {
            if *precision > 38 {
                return Err(ZyronError::StreamingError(format!(
                    "format inference does not support Decimal256 with precision {precision} scale {scale}, cap is 38"
                )));
            }
            TypeId::Decimal
        }
        ArrowDataType::List(_)
        | ArrowDataType::LargeList(_)
        | ArrowDataType::FixedSizeList(_, _)
        | ArrowDataType::ListView(_)
        | ArrowDataType::LargeListView(_) => TypeId::Array,
        ArrowDataType::Struct(_) => TypeId::Composite,
        ArrowDataType::Union(_, _) => {
            return Err(ZyronError::StreamingError(
                "format inference does not support Union".to_string(),
            ));
        }
        ArrowDataType::Map(_, _) => {
            return Err(ZyronError::StreamingError(
                "format inference does not support Map".to_string(),
            ));
        }
        ArrowDataType::Dictionary(_, _) => {
            return Err(ZyronError::StreamingError(
                "format inference does not support Dictionary".to_string(),
            ));
        }
        ArrowDataType::RunEndEncoded(_, _) => {
            return Err(ZyronError::StreamingError(
                "format inference does not support RunEndEncoded".to_string(),
            ));
        }
    };
    Ok(t)
}

// -----------------------------------------------------------------------------
// Avro to TypeId
// -----------------------------------------------------------------------------

/// Maps an Avro schema node onto a ZyronDB TypeId. Handles primitives,
/// logical types (date, timestamps, decimal, uuid), and the [null, T] union
/// idiom that Avro uses for nullable fields. Errors out for shapes with no
/// direct mapping such as enums, maps, and fixed unions.
pub fn avro_to_type_id(schema: &apache_avro::Schema) -> Result<TypeId> {
    use apache_avro::Schema as AS;
    let t = match schema {
        AS::Null => TypeId::Null,
        AS::Boolean => TypeId::Boolean,
        AS::Int => TypeId::Int32,
        AS::Long => TypeId::Int64,
        AS::Float => TypeId::Float32,
        AS::Double => TypeId::Float64,
        AS::String => TypeId::Text,
        AS::Bytes => TypeId::Binary,
        AS::Date => TypeId::Date,
        AS::TimestampMillis | AS::TimestampMicros | AS::TimestampNanos => TypeId::Timestamp,
        AS::TimeMillis | AS::TimeMicros => TypeId::Time,
        AS::Decimal(_) | AS::BigDecimal => TypeId::Decimal,
        AS::Uuid => TypeId::Uuid,
        AS::LocalTimestampMillis | AS::LocalTimestampMicros | AS::LocalTimestampNanos => {
            TypeId::Timestamp
        }
        AS::Duration => TypeId::Interval,
        AS::Fixed(_) => TypeId::Binary,
        AS::Record(_) => TypeId::Composite,
        AS::Array(_) => TypeId::Array,
        AS::Union(u) => {
            // Accept the [null, T] nullable idiom, recurse on the non-null
            // branch. Reject arbitrary unions with no direct mapping.
            let variants = u.variants();
            if variants.len() == 2 {
                let (null_idx, other_idx) = match (&variants[0], &variants[1]) {
                    (AS::Null, _) => (0usize, 1usize),
                    (_, AS::Null) => (1usize, 0usize),
                    _ => {
                        return Err(ZyronError::StreamingError(
                            "format inference does not support non-nullable Avro unions"
                                .to_string(),
                        ));
                    }
                };
                let _ = null_idx;
                return avro_to_type_id(&variants[other_idx]);
            }
            return Err(ZyronError::StreamingError(
                "format inference does not support Avro unions beyond [null, T]".to_string(),
            ));
        }
        AS::Map(_) => {
            return Err(ZyronError::StreamingError(
                "format inference does not support Avro Map".to_string(),
            ));
        }
        AS::Enum(_) => {
            return Err(ZyronError::StreamingError(
                "format inference does not support Avro Enum".to_string(),
            ));
        }
        AS::Ref { .. } => {
            return Err(ZyronError::StreamingError(
                "format inference cannot resolve Avro schema reference without context".to_string(),
            ));
        }
    };
    Ok(t)
}

// -----------------------------------------------------------------------------
// StreamValue to JSON
// -----------------------------------------------------------------------------

/// Converts a StreamValue into a serde_json Value using the provided TypeId
/// as a hint. Binary payloads are encoded as base64 strings. Int128 values
/// are encoded as decimal strings.
pub fn stream_value_to_json(v: &StreamValue, _t: TypeId) -> serde_json::Value {
    match v {
        StreamValue::Null => serde_json::Value::Null,
        StreamValue::Bool(b) => serde_json::Value::Bool(*b),
        StreamValue::I64(n) => serde_json::Value::Number((*n).into()),
        StreamValue::I128(n) => serde_json::Value::String(n.to_string()),
        StreamValue::F64(n) => serde_json::Number::from_f64(*n)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        StreamValue::Utf8(s) => serde_json::Value::String(s.clone()),
        StreamValue::Binary(b) => serde_json::Value::String(base64_encode(b)),
    }
}

// -----------------------------------------------------------------------------
// JSON to StreamValue
// -----------------------------------------------------------------------------

/// Coerces a serde_json Value into a StreamValue using the target TypeId.
/// Strings are parsed into numbers for integer and float targets, numbers
/// are accepted as-is, null produces StreamValue::Null. Binary targets
/// accept either base64 strings or JSON arrays of byte integers.
pub fn json_to_stream_value(v: &serde_json::Value, t: TypeId) -> Result<StreamValue> {
    if v.is_null() {
        return Ok(StreamValue::Null);
    }
    match t {
        TypeId::Boolean => match v {
            serde_json::Value::Bool(b) => Ok(StreamValue::Bool(*b)),
            serde_json::Value::String(s) => parse_bool(s).map(StreamValue::Bool),
            serde_json::Value::Number(n) => Ok(StreamValue::Bool(n.as_i64().unwrap_or(0) != 0)),
            _ => Err(type_err("bool", v)),
        },
        TypeId::Int8
        | TypeId::Int16
        | TypeId::Int32
        | TypeId::Int64
        | TypeId::UInt8
        | TypeId::UInt16
        | TypeId::UInt32
        | TypeId::UInt64
        | TypeId::Date
        | TypeId::Time
        | TypeId::Timestamp
        | TypeId::TimestampTz => match v {
            serde_json::Value::Number(n) => n
                .as_i64()
                .map(StreamValue::I64)
                .ok_or_else(|| type_err("integer", v)),
            serde_json::Value::String(s) => s
                .parse::<i64>()
                .map(StreamValue::I64)
                .map_err(|_| type_err("integer string", v)),
            _ => Err(type_err("integer", v)),
        },
        TypeId::Int128 | TypeId::Decimal | TypeId::UInt128 => match v {
            serde_json::Value::String(s) => s
                .parse::<i128>()
                .map(StreamValue::I128)
                .map_err(|_| type_err("i128 string", v)),
            serde_json::Value::Number(n) => n
                .as_i64()
                .map(|i| StreamValue::I128(i as i128))
                .ok_or_else(|| type_err("i128 number", v)),
            _ => Err(type_err("i128", v)),
        },
        TypeId::Float32 | TypeId::Float64 => match v {
            serde_json::Value::Number(n) => n
                .as_f64()
                .map(StreamValue::F64)
                .ok_or_else(|| type_err("float", v)),
            serde_json::Value::String(s) => s
                .parse::<f64>()
                .map(StreamValue::F64)
                .map_err(|_| type_err("float string", v)),
            _ => Err(type_err("float", v)),
        },
        TypeId::Char | TypeId::Varchar | TypeId::Text | TypeId::Json | TypeId::Jsonb => match v {
            serde_json::Value::String(s) => Ok(StreamValue::Utf8(s.clone())),
            other => Ok(StreamValue::Utf8(other.to_string())),
        },
        TypeId::Binary
        | TypeId::Varbinary
        | TypeId::Bytea
        | TypeId::Uuid
        | TypeId::Interval
        | TypeId::Array
        | TypeId::Composite
        | TypeId::Vector
        | TypeId::Geometry
        | TypeId::Matrix
        | TypeId::Color
        | TypeId::SemVer
        | TypeId::Inet
        | TypeId::Cidr
        | TypeId::MacAddr
        | TypeId::Money
        | TypeId::Range
        | TypeId::HyperLogLog
        | TypeId::BloomFilter
        | TypeId::TDigest
        | TypeId::CountMinSketch
        | TypeId::Bitfield
        | TypeId::Quantity => match v {
            serde_json::Value::String(s) => base64_decode(s).map(StreamValue::Binary),
            serde_json::Value::Array(arr) => {
                let mut out = Vec::with_capacity(arr.len());
                for item in arr {
                    let byte = item
                        .as_u64()
                        .and_then(|u| u8::try_from(u).ok())
                        .ok_or_else(|| type_err("byte", item))?;
                    out.push(byte);
                }
                Ok(StreamValue::Binary(out))
            }
            _ => Err(type_err("binary", v)),
        },
        TypeId::Null => Ok(StreamValue::Null),
    }
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

fn parse_bool(s: &str) -> Result<bool> {
    match s.to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "y" | "t" => Ok(true),
        "false" | "0" | "no" | "n" | "f" => Ok(false),
        _ => Err(ZyronError::StreamingError(format!(
            "cannot parse bool from '{s}'"
        ))),
    }
}

fn type_err(want: &str, got: &serde_json::Value) -> ZyronError {
    ZyronError::StreamingError(format!("expected {want}, got JSON {got}"))
}

// Minimal base64 encode and decode implementations. Avoids pulling in a new
// crate dependency for what is a simple transform.

const BASE64_ALPHABET: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

pub fn base64_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity((bytes.len() + 2) / 3 * 4);
    let mut i = 0;
    while i + 3 <= bytes.len() {
        let n = ((bytes[i] as u32) << 16) | ((bytes[i + 1] as u32) << 8) | bytes[i + 2] as u32;
        out.push(BASE64_ALPHABET[((n >> 18) & 63) as usize] as char);
        out.push(BASE64_ALPHABET[((n >> 12) & 63) as usize] as char);
        out.push(BASE64_ALPHABET[((n >> 6) & 63) as usize] as char);
        out.push(BASE64_ALPHABET[(n & 63) as usize] as char);
        i += 3;
    }
    let rem = bytes.len() - i;
    if rem == 1 {
        let n = (bytes[i] as u32) << 16;
        out.push(BASE64_ALPHABET[((n >> 18) & 63) as usize] as char);
        out.push(BASE64_ALPHABET[((n >> 12) & 63) as usize] as char);
        out.push('=');
        out.push('=');
    } else if rem == 2 {
        let n = ((bytes[i] as u32) << 16) | ((bytes[i + 1] as u32) << 8);
        out.push(BASE64_ALPHABET[((n >> 18) & 63) as usize] as char);
        out.push(BASE64_ALPHABET[((n >> 12) & 63) as usize] as char);
        out.push(BASE64_ALPHABET[((n >> 6) & 63) as usize] as char);
        out.push('=');
    }
    out
}

pub fn base64_decode(s: &str) -> Result<Vec<u8>> {
    let trimmed = s.trim();
    let bytes = trimmed.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    let mut buf = [0u8; 4];
    let mut bi = 0;
    for &c in bytes {
        if c == b'=' {
            break;
        }
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b' ' | b'\n' | b'\r' | b'\t' => continue,
            _ => {
                return Err(ZyronError::StreamingError(format!(
                    "invalid base64 char: {}",
                    c as char
                )));
            }
        };
        buf[bi] = v;
        bi += 1;
        if bi == 4 {
            let n = ((buf[0] as u32) << 18)
                | ((buf[1] as u32) << 12)
                | ((buf[2] as u32) << 6)
                | buf[3] as u32;
            out.push(((n >> 16) & 0xff) as u8);
            out.push(((n >> 8) & 0xff) as u8);
            out.push((n & 0xff) as u8);
            bi = 0;
        }
    }
    if bi == 2 {
        let n = ((buf[0] as u32) << 18) | ((buf[1] as u32) << 12);
        out.push(((n >> 16) & 0xff) as u8);
    } else if bi == 3 {
        let n = ((buf[0] as u32) << 18) | ((buf[1] as u32) << 12) | ((buf[2] as u32) << 6);
        out.push(((n >> 16) & 0xff) as u8);
        out.push(((n >> 8) & 0xff) as u8);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arrow_to_type_id_primitives() {
        use arrow::datatypes::TimeUnit;
        assert_eq!(
            arrow_to_type_id(&ArrowDataType::Null).unwrap(),
            TypeId::Null
        );
        assert_eq!(
            arrow_to_type_id(&ArrowDataType::Boolean).unwrap(),
            TypeId::Boolean
        );
        assert_eq!(
            arrow_to_type_id(&ArrowDataType::Int8).unwrap(),
            TypeId::Int8
        );
        assert_eq!(
            arrow_to_type_id(&ArrowDataType::Int64).unwrap(),
            TypeId::Int64
        );
        assert_eq!(
            arrow_to_type_id(&ArrowDataType::UInt32).unwrap(),
            TypeId::UInt32
        );
        assert_eq!(
            arrow_to_type_id(&ArrowDataType::Float64).unwrap(),
            TypeId::Float64
        );
        assert_eq!(
            arrow_to_type_id(&ArrowDataType::Utf8).unwrap(),
            TypeId::Text
        );
        assert_eq!(
            arrow_to_type_id(&ArrowDataType::LargeUtf8).unwrap(),
            TypeId::Text
        );
        assert_eq!(
            arrow_to_type_id(&ArrowDataType::Binary).unwrap(),
            TypeId::Binary
        );
        assert_eq!(
            arrow_to_type_id(&ArrowDataType::FixedSizeBinary(16)).unwrap(),
            TypeId::Binary
        );
        assert_eq!(
            arrow_to_type_id(&ArrowDataType::Date32).unwrap(),
            TypeId::Date
        );
        assert_eq!(
            arrow_to_type_id(&ArrowDataType::Date64).unwrap(),
            TypeId::Date
        );
        assert_eq!(
            arrow_to_type_id(&ArrowDataType::Time64(TimeUnit::Microsecond)).unwrap(),
            TypeId::Time
        );
        assert_eq!(
            arrow_to_type_id(&ArrowDataType::Timestamp(TimeUnit::Microsecond, None)).unwrap(),
            TypeId::Timestamp
        );
        assert_eq!(
            arrow_to_type_id(&ArrowDataType::Timestamp(
                TimeUnit::Microsecond,
                Some("UTC".into())
            ))
            .unwrap(),
            TypeId::TimestampTz
        );
        assert_eq!(
            arrow_to_type_id(&ArrowDataType::Decimal128(10, 2)).unwrap(),
            TypeId::Decimal
        );
    }

    #[test]
    fn arrow_to_type_id_unsupported_errors() {
        // Dictionary encoding has no direct Zyron analogue.
        let dt = ArrowDataType::Dictionary(
            Box::new(ArrowDataType::Int32),
            Box::new(ArrowDataType::Utf8),
        );
        assert!(arrow_to_type_id(&dt).is_err());
    }

    #[test]
    fn avro_to_type_id_primitives() {
        use apache_avro::Schema as AS;
        assert_eq!(avro_to_type_id(&AS::Null).unwrap(), TypeId::Null);
        assert_eq!(avro_to_type_id(&AS::Boolean).unwrap(), TypeId::Boolean);
        assert_eq!(avro_to_type_id(&AS::Int).unwrap(), TypeId::Int32);
        assert_eq!(avro_to_type_id(&AS::Long).unwrap(), TypeId::Int64);
        assert_eq!(avro_to_type_id(&AS::Float).unwrap(), TypeId::Float32);
        assert_eq!(avro_to_type_id(&AS::Double).unwrap(), TypeId::Float64);
        assert_eq!(avro_to_type_id(&AS::String).unwrap(), TypeId::Text);
        assert_eq!(avro_to_type_id(&AS::Bytes).unwrap(), TypeId::Binary);
        assert_eq!(avro_to_type_id(&AS::Date).unwrap(), TypeId::Date);
        assert_eq!(
            avro_to_type_id(&AS::TimestampMillis).unwrap(),
            TypeId::Timestamp
        );
        assert_eq!(avro_to_type_id(&AS::Uuid).unwrap(), TypeId::Uuid);
    }

    #[test]
    fn avro_to_type_id_nullable_union() {
        let union_json = r#"["null", "long"]"#;
        let schema = apache_avro::Schema::parse_str(union_json).unwrap();
        assert_eq!(avro_to_type_id(&schema).unwrap(), TypeId::Int64);
    }

    #[test]
    fn base64_roundtrip() {
        let cases: &[&[u8]] = &[b"", b"f", b"fo", b"foo", b"foob", b"fooba", b"foobar"];
        for c in cases {
            let enc = base64_encode(c);
            let dec = base64_decode(&enc).unwrap();
            assert_eq!(&dec[..], *c, "roundtrip failed for {:?}", c);
        }
    }
}

// -----------------------------------------------------------------------------
// Protobuf wire-format reader and writer
// -----------------------------------------------------------------------------
//
// Without a compiled .proto schema, result sets are encoded self-descriptively:
// the stream is a sequence of length-delimited Row messages (field 1, repeated),
// and each Row encodes its columns by field number = column_index + 1. Wire
// types follow the column value: varint for integers and booleans, fixed64 for
// floats, length-delimited for strings and binary. A 128-bit integer has no
// native protobuf scalar and is carried as 16 big-endian bytes. A NULL column
// is omitted, matching protobuf's absent-field semantics. The output is valid
// protobuf decodable by any library that maps field numbers to columns.

use super::{ColumnSpec, FormatReader, FormatWriter};
use crate::row_codec::StreamValue;
use zyron_common::{Result, TypeId, ZyronError};

const WIRE_VARINT: u8 = 0;
const WIRE_FIXED64: u8 = 1;
const WIRE_LEN: u8 = 2;
const WIRE_FIXED32: u8 = 5;

// -----------------------------------------------------------------------------
// Writer
// -----------------------------------------------------------------------------

pub struct ProtobufWriter;

fn encode_varint(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let mut byte = (v & 0x7F) as u8;
        v >>= 7;
        if v != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if v == 0 {
            break;
        }
    }
}

fn encode_tag(out: &mut Vec<u8>, field: u32, wire: u8) {
    encode_varint(out, ((field as u64) << 3) | wire as u64);
}

fn encode_len_field(out: &mut Vec<u8>, field: u32, bytes: &[u8]) {
    encode_tag(out, field, WIRE_LEN);
    encode_varint(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

fn encode_row(row: &[StreamValue]) -> Vec<u8> {
    let mut buf = Vec::new();
    for (i, value) in row.iter().enumerate() {
        let field = (i + 1) as u32;
        match value {
            StreamValue::Null => {} // omitted: absent field == null
            StreamValue::Bool(b) => {
                encode_tag(&mut buf, field, WIRE_VARINT);
                encode_varint(&mut buf, u64::from(*b));
            }
            StreamValue::I64(n) => {
                encode_tag(&mut buf, field, WIRE_VARINT);
                // protobuf int64 is two's-complement in a varint
                encode_varint(&mut buf, *n as u64);
            }
            StreamValue::I128(n) => {
                encode_len_field(&mut buf, field, &n.to_be_bytes());
            }
            StreamValue::F64(f) => {
                encode_tag(&mut buf, field, WIRE_FIXED64);
                buf.extend_from_slice(&f.to_le_bytes());
            }
            StreamValue::Utf8(s) => encode_len_field(&mut buf, field, s.as_bytes()),
            StreamValue::Binary(b) => encode_len_field(&mut buf, field, b),
        }
    }
    buf
}

impl FormatWriter for ProtobufWriter {
    fn write_rows(&mut self, rows: &[Vec<StreamValue>], _schema: &[ColumnSpec]) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        for row in rows {
            let row_bytes = encode_row(row);
            // Field 1 (rows), repeated, length-delimited.
            encode_len_field(&mut out, 1, &row_bytes);
        }
        Ok(out)
    }
}

// -----------------------------------------------------------------------------
// Reader
// -----------------------------------------------------------------------------

pub struct ProtobufReader;

fn read_varint(bytes: &[u8], pos: &mut usize) -> Result<u64> {
    let mut result = 0u64;
    let mut shift = 0u32;
    loop {
        if *pos >= bytes.len() {
            return Err(ZyronError::ExecutionError(
                "protobuf: truncated varint".to_string(),
            ));
        }
        let byte = bytes[*pos];
        *pos += 1;
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
        if shift >= 64 {
            return Err(ZyronError::ExecutionError(
                "protobuf: varint overflow".to_string(),
            ));
        }
    }
    Ok(result)
}

fn skip_field(bytes: &[u8], pos: &mut usize, wire: u8) -> Result<()> {
    match wire {
        WIRE_VARINT => {
            read_varint(bytes, pos)?;
        }
        WIRE_FIXED64 => *pos += 8,
        WIRE_LEN => {
            let len = read_varint(bytes, pos)? as usize;
            *pos += len;
        }
        WIRE_FIXED32 => *pos += 4,
        _ => {
            return Err(ZyronError::ExecutionError(format!(
                "protobuf: unsupported wire type {wire}"
            )));
        }
    }
    if *pos > bytes.len() {
        return Err(ZyronError::ExecutionError(
            "protobuf: truncated field".to_string(),
        ));
    }
    Ok(())
}

fn decode_row(bytes: &[u8], schema: &[ColumnSpec]) -> Result<Vec<StreamValue>> {
    let mut row = vec![StreamValue::Null; schema.len()];
    let mut pos = 0;
    while pos < bytes.len() {
        let tag = read_varint(bytes, &mut pos)?;
        let field = (tag >> 3) as usize;
        let wire = (tag & 0x7) as u8;
        if field == 0 || field > schema.len() {
            // Field outside the known schema: skip to stay wire-compatible.
            skip_field(bytes, &mut pos, wire)?;
            continue;
        }
        let col = field - 1;
        let type_id = schema[col].type_id;
        let value = match wire {
            WIRE_VARINT => {
                let v = read_varint(bytes, &mut pos)?;
                if type_id == TypeId::Boolean {
                    StreamValue::Bool(v != 0)
                } else {
                    StreamValue::I64(v as i64)
                }
            }
            WIRE_FIXED64 => {
                if pos + 8 > bytes.len() {
                    return Err(ZyronError::ExecutionError(
                        "protobuf: truncated fixed64".to_string(),
                    ));
                }
                let arr: [u8; 8] = bytes[pos..pos + 8].try_into().unwrap();
                pos += 8;
                StreamValue::F64(f64::from_le_bytes(arr))
            }
            WIRE_LEN => {
                let len = read_varint(bytes, &mut pos)? as usize;
                if pos + len > bytes.len() {
                    return Err(ZyronError::ExecutionError(
                        "protobuf: truncated length-delimited field".to_string(),
                    ));
                }
                let data = &bytes[pos..pos + len];
                pos += len;
                if type_id == TypeId::Int128 && data.len() == 16 {
                    let mut arr = [0u8; 16];
                    arr.copy_from_slice(data);
                    StreamValue::I128(i128::from_be_bytes(arr))
                } else if type_id.is_binary() {
                    StreamValue::Binary(data.to_vec())
                } else {
                    StreamValue::Utf8(String::from_utf8_lossy(data).into_owned())
                }
            }
            other => {
                return Err(ZyronError::ExecutionError(format!(
                    "protobuf: unsupported wire type {other}"
                )));
            }
        };
        row[col] = value;
    }
    Ok(row)
}

impl FormatReader for ProtobufReader {
    fn read_rows(&mut self, bytes: &[u8], schema: &[ColumnSpec]) -> Result<Vec<Vec<StreamValue>>> {
        let mut rows = Vec::new();
        let mut pos = 0;
        while pos < bytes.len() {
            let tag = read_varint(bytes, &mut pos)?;
            let field = (tag >> 3) as u32;
            let wire = (tag & 0x7) as u8;
            if field != 1 || wire != WIRE_LEN {
                return Err(ZyronError::ExecutionError(
                    "protobuf: expected repeated row field 1 (length-delimited)".to_string(),
                ));
            }
            let len = read_varint(bytes, &mut pos)? as usize;
            if pos + len > bytes.len() {
                return Err(ZyronError::ExecutionError(
                    "protobuf: truncated row message".to_string(),
                ));
            }
            let row_bytes = &bytes[pos..pos + len];
            pos += len;
            rows.push(decode_row(row_bytes, schema)?);
        }
        Ok(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema() -> Vec<ColumnSpec> {
        vec![
            ColumnSpec::new("id", TypeId::Int64),
            ColumnSpec::new("name", TypeId::Varchar),
            ColumnSpec::new("active", TypeId::Boolean),
            ColumnSpec::new("score", TypeId::Float64),
        ]
    }

    #[test]
    fn round_trips_rows_including_null() {
        let rows = vec![
            vec![
                StreamValue::I64(1),
                StreamValue::Utf8("alpha".to_string()),
                StreamValue::Bool(true),
                StreamValue::F64(1.5),
            ],
            vec![
                StreamValue::I64(-2),
                StreamValue::Null,
                StreamValue::Bool(false),
                StreamValue::F64(-3.25),
            ],
        ];
        let bytes = ProtobufWriter.write_rows(&rows, &schema()).unwrap();
        let decoded = ProtobufReader.read_rows(&bytes, &schema()).unwrap();
        assert_eq!(decoded.len(), 2);
        assert!(matches!(decoded[0][0], StreamValue::I64(1)));
        assert!(matches!(&decoded[0][1], StreamValue::Utf8(s) if s == "alpha"));
        assert!(matches!(decoded[0][2], StreamValue::Bool(true)));
        // Omitted (null) name decodes back to Null.
        assert!(matches!(decoded[1][1], StreamValue::Null));
        assert!(matches!(decoded[1][0], StreamValue::I64(-2)));
    }
}

// -----------------------------------------------------------------------------
// Avro format
// -----------------------------------------------------------------------------
//
// Reads and writes Apache Avro object container files. The schema is built
// from ColumnSpec entries, fields are wrapped in a union with null so every
// column can carry a null. Unsigned integers widen to long since Avro has
// no unsigned types. Decimal, Int128, Uuid, Interval, Array, Composite, and
// Vector columns are encoded as bytes.

use super::{ColumnSpec, FormatReader, FormatWriter};
use crate::row_codec::StreamValue;
use apache_avro::{
    Reader as AvroReaderCore, Schema as AvroSchema, Writer as AvroWriterCore,
    types::Value as AvroValue,
};
use zyron_common::{Result, TypeId, ZyronError};

pub struct AvroReader;

impl FormatReader for AvroReader {
    fn read_rows(&mut self, bytes: &[u8], schema: &[ColumnSpec]) -> Result<Vec<Vec<StreamValue>>> {
        let reader = AvroReaderCore::new(bytes)
            .map_err(|e| ZyronError::StreamingError(format!("avro: reader init error: {e}")))?;
        let mut rows = Vec::new();
        for value in reader {
            let value =
                value.map_err(|e| ZyronError::StreamingError(format!("avro: read error: {e}")))?;
            rows.push(record_to_row(&value, schema)?);
        }
        Ok(rows)
    }
}

pub struct AvroWriter;

impl FormatWriter for AvroWriter {
    fn write_rows(&mut self, rows: &[Vec<StreamValue>], schema: &[ColumnSpec]) -> Result<Vec<u8>> {
        let schema_json = build_schema_json(schema);
        let avro_schema = AvroSchema::parse_str(&schema_json)
            .map_err(|e| ZyronError::StreamingError(format!("avro: schema parse error: {e}")))?;
        let mut writer = AvroWriterCore::new(&avro_schema, Vec::new());
        for row in rows {
            let mut fields: Vec<(String, AvroValue)> = Vec::with_capacity(schema.len());
            for (col, v) in schema.iter().zip(row.iter()) {
                fields.push((col.name.clone(), stream_value_to_avro(v, col.type_id)?));
            }
            writer
                .append(AvroValue::Record(fields))
                .map_err(|e| ZyronError::StreamingError(format!("avro: append error: {e}")))?;
        }
        writer
            .into_inner()
            .map_err(|e| ZyronError::StreamingError(format!("avro: finalize error: {e}")))
    }
}

// -----------------------------------------------------------------------------
// Schema construction
// -----------------------------------------------------------------------------

fn build_schema_json(schema: &[ColumnSpec]) -> String {
    let mut fields = String::new();
    for (i, col) in schema.iter().enumerate() {
        if i > 0 {
            fields.push(',');
        }
        let t = avro_type_json(col.type_id);
        fields.push_str(&format!(
            "{{\"name\":\"{}\",\"type\":[\"null\",{}]}}",
            col.name, t
        ));
    }
    format!(
        "{{\"type\":\"record\",\"name\":\"ZyronRow\",\"fields\":[{}]}}",
        fields
    )
}

fn avro_type_json(t: TypeId) -> &'static str {
    match t {
        TypeId::Boolean => "\"boolean\"",
        TypeId::Int8 | TypeId::Int16 | TypeId::Int32 => "\"int\"",
        TypeId::Int64
        | TypeId::UInt8
        | TypeId::UInt16
        | TypeId::UInt32
        | TypeId::UInt64
        | TypeId::Date
        | TypeId::Time
        | TypeId::Timestamp
        | TypeId::TimestampTz => "\"long\"",
        TypeId::Float32 => "\"float\"",
        TypeId::Float64 => "\"double\"",
        TypeId::Char | TypeId::Varchar | TypeId::Text | TypeId::Json | TypeId::Jsonb => {
            "\"string\""
        }
        TypeId::Int128
        | TypeId::Decimal
        | TypeId::UInt128
        | TypeId::Binary
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
        | TypeId::Quantity
        | TypeId::Null => "\"bytes\"",
    }
}

// -----------------------------------------------------------------------------
// StreamValue to Avro
// -----------------------------------------------------------------------------

fn stream_value_to_avro(v: &StreamValue, t: TypeId) -> Result<AvroValue> {
    if matches!(v, StreamValue::Null) {
        return Ok(AvroValue::Union(0, Box::new(AvroValue::Null)));
    }
    let inner = match (t, v) {
        (TypeId::Boolean, StreamValue::Bool(b)) => AvroValue::Boolean(*b),
        (TypeId::Int8, StreamValue::I64(n))
        | (TypeId::Int16, StreamValue::I64(n))
        | (TypeId::Int32, StreamValue::I64(n)) => AvroValue::Int(*n as i32),
        (TypeId::Int64, StreamValue::I64(n))
        | (TypeId::UInt8, StreamValue::I64(n))
        | (TypeId::UInt16, StreamValue::I64(n))
        | (TypeId::UInt32, StreamValue::I64(n))
        | (TypeId::UInt64, StreamValue::I64(n))
        | (TypeId::Date, StreamValue::I64(n))
        | (TypeId::Time, StreamValue::I64(n))
        | (TypeId::Timestamp, StreamValue::I64(n))
        | (TypeId::TimestampTz, StreamValue::I64(n)) => AvroValue::Long(*n),
        (TypeId::Float32, StreamValue::F64(n)) => AvroValue::Float(*n as f32),
        (TypeId::Float64, StreamValue::F64(n)) => AvroValue::Double(*n),
        (TypeId::Float32, StreamValue::I64(n)) => AvroValue::Float(*n as f32),
        (TypeId::Float64, StreamValue::I64(n)) => AvroValue::Double(*n as f64),
        (TypeId::Char, StreamValue::Utf8(s))
        | (TypeId::Varchar, StreamValue::Utf8(s))
        | (TypeId::Text, StreamValue::Utf8(s))
        | (TypeId::Json, StreamValue::Utf8(s))
        | (TypeId::Jsonb, StreamValue::Utf8(s)) => AvroValue::String(s.clone()),
        (TypeId::Int128, StreamValue::I128(n))
        | (TypeId::Decimal, StreamValue::I128(n))
        | (TypeId::UInt128, StreamValue::I128(n)) => AvroValue::Bytes(n.to_le_bytes().to_vec()),
        (TypeId::Binary, StreamValue::Binary(b))
        | (TypeId::Varbinary, StreamValue::Binary(b))
        | (TypeId::Bytea, StreamValue::Binary(b))
        | (TypeId::Uuid, StreamValue::Binary(b))
        | (TypeId::Interval, StreamValue::Binary(b))
        | (TypeId::Array, StreamValue::Binary(b))
        | (TypeId::Composite, StreamValue::Binary(b))
        | (TypeId::Vector, StreamValue::Binary(b)) => AvroValue::Bytes(b.clone()),
        _ => {
            return Err(ZyronError::StreamingError(format!(
                "avro: cannot encode {v:?} as {t:?}"
            )));
        }
    };
    Ok(AvroValue::Union(1, Box::new(inner)))
}

// -----------------------------------------------------------------------------
// Avro to StreamValue
// -----------------------------------------------------------------------------

fn record_to_row(value: &AvroValue, schema: &[ColumnSpec]) -> Result<Vec<StreamValue>> {
    let fields = match value {
        AvroValue::Record(fs) => fs,
        _ => {
            return Err(ZyronError::StreamingError(
                "avro: expected record value".to_string(),
            ));
        }
    };
    let mut row = Vec::with_capacity(schema.len());
    for col in schema {
        let v = fields
            .iter()
            .find(|(n, _)| n == &col.name)
            .map(|(_, v)| v)
            .ok_or_else(|| {
                ZyronError::StreamingError(format!("avro: missing field '{}' in record", col.name))
            })?;
        row.push(avro_to_stream_value(v, col.type_id)?);
    }
    Ok(row)
}

fn avro_to_stream_value(v: &AvroValue, t: TypeId) -> Result<StreamValue> {
    let inner = match v {
        AvroValue::Union(_, boxed) => boxed.as_ref(),
        other => other,
    };
    Ok(match (t, inner) {
        (_, AvroValue::Null) => StreamValue::Null,
        (TypeId::Boolean, AvroValue::Boolean(b)) => StreamValue::Bool(*b),
        (_, AvroValue::Int(n)) => StreamValue::I64(*n as i64),
        (_, AvroValue::Long(n)) => StreamValue::I64(*n),
        (TypeId::Float32, AvroValue::Float(n)) => StreamValue::F64(*n as f64),
        (TypeId::Float64, AvroValue::Double(n)) => StreamValue::F64(*n),
        (_, AvroValue::Float(n)) => StreamValue::F64(*n as f64),
        (_, AvroValue::Double(n)) => StreamValue::F64(*n),
        (_, AvroValue::String(s)) => StreamValue::Utf8(s.clone()),
        (TypeId::Int128, AvroValue::Bytes(b))
        | (TypeId::Decimal, AvroValue::Bytes(b))
        | (TypeId::UInt128, AvroValue::Bytes(b))
            if b.len() == 16 =>
        {
            let mut arr = [0u8; 16];
            arr.copy_from_slice(b);
            StreamValue::I128(i128::from_le_bytes(arr))
        }
        (_, AvroValue::Bytes(b)) => StreamValue::Binary(b.clone()),
        (_, other) => {
            return Err(ZyronError::StreamingError(format!(
                "avro: cannot decode {other:?} as {t:?}"
            )));
        }
    })
}

// -----------------------------------------------------------------------------
// Schema inference
// -----------------------------------------------------------------------------

/// Reads the Avro object container header from an in-memory buffer and
/// returns the inferred column list. Field order matches the writer schema.
pub fn infer_avro_schema(bytes: &[u8]) -> Result<Vec<ColumnSpec>> {
    use super::schema::avro_to_type_id;
    let reader = AvroReaderCore::new(bytes)
        .map_err(|e| ZyronError::StreamingError(format!("avro: schema read error: {e}")))?;
    let writer_schema = reader.writer_schema();
    let record_fields = match writer_schema {
        AvroSchema::Record(r) => &r.fields,
        _ => {
            return Err(ZyronError::StreamingError(
                "avro: top-level schema must be a record for schema inference".to_string(),
            ));
        }
    };
    let mut cols = Vec::with_capacity(record_fields.len());
    for field in record_fields {
        let type_id = avro_to_type_id(&field.schema)?;
        cols.push(ColumnSpec {
            name: field.name.clone(),
            type_id,
        });
    }
    Ok(cols)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{assert_rows_equal, sample_rows, sample_schema};

    #[test]
    fn avro_roundtrip() {
        let schema = sample_schema();
        let rows = sample_rows();
        let mut writer = AvroWriter;
        let bytes = writer.write_rows(&rows, &schema).unwrap();
        let mut reader = AvroReader;
        let decoded = reader.read_rows(&bytes, &schema).unwrap();
        assert_rows_equal(&decoded, &rows);
    }

    #[test]
    fn avro_infer_schema_matches_writer() {
        let schema = sample_schema();
        let rows = sample_rows();
        let mut writer = AvroWriter;
        let bytes = writer.write_rows(&rows, &schema).unwrap();
        let inferred = infer_avro_schema(&bytes).unwrap();
        assert_eq!(inferred.len(), schema.len());
        assert_eq!(inferred[0].name, "id");
        // Avro field types round-trip through the union [null, T] idiom.
        // Varchar is written as string which maps back to Text.
        assert_eq!(inferred[0].type_id, TypeId::Int64);
        assert_eq!(inferred[1].type_id, TypeId::Text);
        assert_eq!(inferred[2].type_id, TypeId::Boolean);
        assert_eq!(inferred[3].type_id, TypeId::Float64);
    }
}

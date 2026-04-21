// -----------------------------------------------------------------------------
// CSV format
// -----------------------------------------------------------------------------
//
// Comma-separated values with a header row. The reader matches headers to
// schema names and coerces each field to StreamValue per the declared
// TypeId. The writer emits a header row followed by one data row per input
// row. Null values become empty cells. Vector, Array, Composite, Uuid,
// Interval, and Binary columns are unsupported and yield an error.

use super::{ColumnSpec, FormatReader, FormatWriter};
use crate::row_codec::StreamValue;
use zyron_common::{Result, TypeId, ZyronError};

// -----------------------------------------------------------------------------
// Reader
// -----------------------------------------------------------------------------

pub struct CsvReader;

impl FormatReader for CsvReader {
    fn read_rows(
        &mut self,
        bytes: &[u8],
        schema: &[ColumnSpec],
    ) -> Result<Vec<Vec<StreamValue>>> {
        let mut rdr = ::csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(bytes);
        let headers = rdr
            .headers()
            .map_err(|e| ZyronError::StreamingError(format!("csv: header error: {e}")))?
            .clone();
        let mut index_map = Vec::with_capacity(schema.len());
        for col in schema {
            let idx = headers.iter().position(|h| h == col.name);
            index_map.push(idx);
        }
        let mut out = Vec::new();
        for (ri, record) in rdr.records().enumerate() {
            let record = record.map_err(|e| {
                ZyronError::StreamingError(format!("csv: record {ri} read error: {e}"))
            })?;
            let mut row = Vec::with_capacity(schema.len());
            for (col, idx) in schema.iter().zip(index_map.iter()) {
                let text = match idx {
                    Some(i) => record.get(*i).unwrap_or(""),
                    None => "",
                };
                row.push(text_to_value(text, col.type_id)?);
            }
            out.push(row);
        }
        Ok(out)
    }
}

// -----------------------------------------------------------------------------
// Writer
// -----------------------------------------------------------------------------

pub struct CsvWriter;

impl FormatWriter for CsvWriter {
    fn write_rows(
        &mut self,
        rows: &[Vec<StreamValue>],
        schema: &[ColumnSpec],
    ) -> Result<Vec<u8>> {
        let mut wtr = ::csv::WriterBuilder::new().from_writer(Vec::new());
        let header: Vec<&str> = schema.iter().map(|c| c.name.as_str()).collect();
        wtr.write_record(&header).map_err(|e| {
            ZyronError::StreamingError(format!("csv: header write error: {e}"))
        })?;
        for row in rows {
            if row.len() != schema.len() {
                return Err(ZyronError::StreamingError(format!(
                    "csv: row arity {} does not match schema arity {}",
                    row.len(),
                    schema.len()
                )));
            }
            let mut fields: Vec<String> = Vec::with_capacity(schema.len());
            for (col, v) in schema.iter().zip(row.iter()) {
                fields.push(value_to_text(v, col.type_id)?);
            }
            wtr.write_record(&fields).map_err(|e| {
                ZyronError::StreamingError(format!("csv: row write error: {e}"))
            })?;
        }
        wtr.flush()
            .map_err(|e| ZyronError::StreamingError(format!("csv: flush error: {e}")))?;
        wtr.into_inner()
            .map_err(|e| ZyronError::StreamingError(format!("csv: finalize error: {e}")))
    }
}

// -----------------------------------------------------------------------------
// Coercion
// -----------------------------------------------------------------------------

fn text_to_value(text: &str, t: TypeId) -> Result<StreamValue> {
    if text.is_empty() {
        return Ok(StreamValue::Null);
    }
    match t {
        TypeId::Boolean => match text.to_ascii_lowercase().as_str() {
            "true" | "1" | "yes" | "y" | "t" => Ok(StreamValue::Bool(true)),
            "false" | "0" | "no" | "n" | "f" => Ok(StreamValue::Bool(false)),
            _ => Err(ZyronError::StreamingError(format!(
                "csv: cannot parse bool from '{text}'"
            ))),
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
        | TypeId::TimestampTz => text
            .parse::<i64>()
            .map(StreamValue::I64)
            .map_err(|_| ZyronError::StreamingError(format!("csv: bad integer '{text}'"))),
        TypeId::Int128 | TypeId::Decimal | TypeId::UInt128 => text
            .parse::<i128>()
            .map(StreamValue::I128)
            .map_err(|_| ZyronError::StreamingError(format!("csv: bad i128 '{text}'"))),
        TypeId::Float32 | TypeId::Float64 => text
            .parse::<f64>()
            .map(StreamValue::F64)
            .map_err(|_| ZyronError::StreamingError(format!("csv: bad float '{text}'"))),
        TypeId::Char | TypeId::Varchar | TypeId::Text | TypeId::Json | TypeId::Jsonb => {
            Ok(StreamValue::Utf8(text.to_string()))
        }
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
        | TypeId::Quantity
        | TypeId::Null => Err(ZyronError::StreamingError(format!(
            "csv: type {t:?} is not supported"
        ))),
    }
}

fn value_to_text(v: &StreamValue, t: TypeId) -> Result<String> {
    match v {
        StreamValue::Null => Ok(String::new()),
        StreamValue::Bool(b) => Ok(if *b { "true".to_string() } else { "false".to_string() }),
        StreamValue::I64(n) => Ok(n.to_string()),
        StreamValue::I128(n) => Ok(n.to_string()),
        StreamValue::F64(n) => Ok(n.to_string()),
        StreamValue::Utf8(s) => Ok(s.clone()),
        StreamValue::Binary(_) => Err(ZyronError::StreamingError(format!(
            "csv: binary type {t:?} is not supported"
        ))),
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{assert_rows_equal, sample_rows, sample_schema};

    #[test]
    fn csv_roundtrip() {
        let schema = sample_schema();
        let rows = sample_rows();
        let mut writer = CsvWriter;
        let bytes = writer.write_rows(&rows, &schema).unwrap();
        let mut reader = CsvReader;
        let decoded = reader.read_rows(&bytes, &schema).unwrap();
        assert_rows_equal(&decoded, &rows);
    }
}

// -----------------------------------------------------------------------------
// JSON Lines format
// -----------------------------------------------------------------------------
//
// One JSON object per line. The reader splits on newlines and parses each
// non-empty line as an object, coercing fields to StreamValue per TypeId.
// The writer emits one compact JSON object per row followed by a newline.

use super::{ColumnSpec, FormatReader, FormatWriter};
use super::schema::{json_to_stream_value, stream_value_to_json};
use crate::row_codec::StreamValue;
use zyron_common::{Result, ZyronError};

// -----------------------------------------------------------------------------
// Reader
// -----------------------------------------------------------------------------

pub struct JsonLinesReader;

impl FormatReader for JsonLinesReader {
    fn read_rows(
        &mut self,
        bytes: &[u8],
        schema: &[ColumnSpec],
    ) -> Result<Vec<Vec<StreamValue>>> {
        let text = std::str::from_utf8(bytes).map_err(|e| {
            ZyronError::StreamingError(format!("jsonl: invalid UTF-8: {e}"))
        })?;
        let mut rows = Vec::new();
        for (i, line) in text.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let value: serde_json::Value = serde_json::from_str(trimmed).map_err(|e| {
                ZyronError::StreamingError(format!("jsonl: parse error at line {i}: {e}"))
            })?;
            rows.push(object_to_row(&value, schema)?);
        }
        Ok(rows)
    }
}

// -----------------------------------------------------------------------------
// Writer
// -----------------------------------------------------------------------------

pub struct JsonLinesWriter;

impl FormatWriter for JsonLinesWriter {
    fn write_rows(
        &mut self,
        rows: &[Vec<StreamValue>],
        schema: &[ColumnSpec],
    ) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        for row in rows {
            let obj = row_to_object(row, schema)?;
            let s = serde_json::to_string(&obj).map_err(|e| {
                ZyronError::StreamingError(format!("jsonl: serialize error: {e}"))
            })?;
            out.extend_from_slice(s.as_bytes());
            out.push(b'\n');
        }
        Ok(out)
    }
}

// -----------------------------------------------------------------------------
// Helpers shared with json.rs
// -----------------------------------------------------------------------------

pub(crate) fn object_to_row(
    value: &serde_json::Value,
    schema: &[ColumnSpec],
) -> Result<Vec<StreamValue>> {
    let obj = value.as_object().ok_or_else(|| {
        ZyronError::StreamingError("jsonl: expected JSON object per record".to_string())
    })?;
    let mut row = Vec::with_capacity(schema.len());
    for col in schema {
        match obj.get(&col.name) {
            Some(v) => row.push(json_to_stream_value(v, col.type_id)?),
            None => row.push(StreamValue::Null),
        }
    }
    Ok(row)
}

pub(crate) fn row_to_object(
    row: &[StreamValue],
    schema: &[ColumnSpec],
) -> Result<serde_json::Value> {
    if row.len() != schema.len() {
        return Err(ZyronError::StreamingError(format!(
            "row arity {} does not match schema arity {}",
            row.len(),
            schema.len()
        )));
    }
    let mut map = serde_json::Map::with_capacity(schema.len());
    for (col, v) in schema.iter().zip(row.iter()) {
        map.insert(col.name.clone(), stream_value_to_json(v, col.type_id));
    }
    Ok(serde_json::Value::Object(map))
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{assert_rows_equal, sample_rows, sample_schema};

    #[test]
    fn jsonl_roundtrip() {
        let schema = sample_schema();
        let rows = sample_rows();
        let mut writer = JsonLinesWriter;
        let bytes = writer.write_rows(&rows, &schema).unwrap();
        let mut reader = JsonLinesReader;
        let decoded = reader.read_rows(&bytes, &schema).unwrap();
        assert_rows_equal(&decoded, &rows);
    }
}

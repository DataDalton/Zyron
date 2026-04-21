// -----------------------------------------------------------------------------
// JSON array format
// -----------------------------------------------------------------------------
//
// A single JSON array containing objects. Reads parse the whole buffer,
// writes emit a pretty-printed array. Field coercion is shared with jsonl.

use super::jsonl::{object_to_row, row_to_object};
use super::{ColumnSpec, FormatReader, FormatWriter};
use crate::row_codec::StreamValue;
use zyron_common::{Result, ZyronError};

// -----------------------------------------------------------------------------
// Reader
// -----------------------------------------------------------------------------

pub struct JsonReader;

impl FormatReader for JsonReader {
    fn read_rows(&mut self, bytes: &[u8], schema: &[ColumnSpec]) -> Result<Vec<Vec<StreamValue>>> {
        let value: serde_json::Value = serde_json::from_slice(bytes)
            .map_err(|e| ZyronError::StreamingError(format!("json: parse error: {e}")))?;
        let arr = value.as_array().ok_or_else(|| {
            ZyronError::StreamingError("json: expected top-level array".to_string())
        })?;
        let mut rows = Vec::with_capacity(arr.len());
        for item in arr {
            rows.push(object_to_row(item, schema)?);
        }
        Ok(rows)
    }
}

// -----------------------------------------------------------------------------
// Writer
// -----------------------------------------------------------------------------

pub struct JsonWriter;

impl FormatWriter for JsonWriter {
    fn write_rows(&mut self, rows: &[Vec<StreamValue>], schema: &[ColumnSpec]) -> Result<Vec<u8>> {
        let mut arr = Vec::with_capacity(rows.len());
        for row in rows {
            arr.push(row_to_object(row, schema)?);
        }
        let value = serde_json::Value::Array(arr);
        serde_json::to_vec(&value)
            .map_err(|e| ZyronError::StreamingError(format!("json: serialize error: {e}")))
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
    fn json_roundtrip() {
        let schema = sample_schema();
        let rows = sample_rows();
        let mut writer = JsonWriter;
        let bytes = writer.write_rows(&rows, &schema).unwrap();
        let mut reader = JsonReader;
        let decoded = reader.read_rows(&bytes, &schema).unwrap();
        assert_rows_equal(&decoded, &rows);
    }
}

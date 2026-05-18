// -----------------------------------------------------------------------------
// Arrow IPC stream format
// -----------------------------------------------------------------------------
//
// Reads and writes Arrow IPC streams. Schema generation uses the shared
// TypeId to DataType mapping in format/schema.rs. All rows are written as a
// single RecordBatch and readers iterate over batches to handle files that
// contain more than one batch.

use super::{ColumnSpec, FormatReader, FormatWriter};
use super::schema::{arrow_to_type_id, timestamp_arrow_type};
use crate::row_codec::StreamValue;
use arrow::array::RecordBatch;
use arrow::datatypes::{Field, Schema};
use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::StreamWriter;
use std::io::Cursor;
use std::sync::Arc;
use zyron_common::{Result, ZyronError};

pub struct ArrowIpcReader;

impl FormatReader for ArrowIpcReader {
    fn read_rows(
        &mut self,
        bytes: &[u8],
        schema: &[ColumnSpec],
    ) -> Result<Vec<Vec<StreamValue>>> {
        let cursor = Cursor::new(bytes);
        let reader = StreamReader::try_new(cursor, None).map_err(|e| {
            ZyronError::StreamingError(format!("arrow_ipc: open error: {e}"))
        })?;
        let mut rows = Vec::new();
        for batch in reader {
            let batch = batch.map_err(|e| {
                ZyronError::StreamingError(format!("arrow_ipc: read error: {e}"))
            })?;
            super::record_batch::batch_to_rows(&batch, schema, &mut rows)?;
        }
        Ok(rows)
    }
}

pub struct ArrowIpcWriter;

impl FormatWriter for ArrowIpcWriter {
    fn write_rows(
        &mut self,
        rows: &[Vec<StreamValue>],
        schema: &[ColumnSpec],
    ) -> Result<Vec<u8>> {
        let fields: Vec<Field> = schema
            .iter()
            .map(|c| {
                Field::new(
                    &c.name,
                    timestamp_arrow_type(c.type_id, c.ts_precision),
                    true,
                )
            })
            .collect();
        let arrow_schema = Arc::new(Schema::new(fields));
        let batch: RecordBatch =
            super::record_batch::rows_to_batch(rows, schema, arrow_schema.clone())?;
        let mut buf: Vec<u8> = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut buf, &arrow_schema).map_err(|e| {
                ZyronError::StreamingError(format!("arrow_ipc: writer init error: {e}"))
            })?;
            writer.write(&batch).map_err(|e| {
                ZyronError::StreamingError(format!("arrow_ipc: write error: {e}"))
            })?;
            writer.finish().map_err(|e| {
                ZyronError::StreamingError(format!("arrow_ipc: finish error: {e}"))
            })?;
        }
        Ok(buf)
    }
}

// -----------------------------------------------------------------------------
// Schema inference
// -----------------------------------------------------------------------------

/// Reads the Arrow IPC stream header and returns the inferred column list.
/// Field order matches the file's Arrow schema.
pub fn infer_arrow_ipc_schema(bytes: &[u8]) -> Result<Vec<ColumnSpec>> {
    let cursor = Cursor::new(bytes);
    let reader = StreamReader::try_new(cursor, None).map_err(|e| {
        ZyronError::StreamingError(format!("arrow_ipc: schema read error: {e}"))
    })?;
    let schema = reader.schema();
    let mut cols = Vec::with_capacity(schema.fields().len());
    for field in schema.fields() {
        let type_id = arrow_to_type_id(field.data_type())?;
        cols.push(ColumnSpec::new(field.name().to_string(), type_id));
    }
    Ok(cols)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{assert_rows_equal, sample_rows, sample_schema};
    use zyron_common::TypeId;

    #[test]
    fn arrow_ipc_roundtrip() {
        let schema = sample_schema();
        let rows = sample_rows();
        let mut writer = ArrowIpcWriter;
        let bytes = writer.write_rows(&rows, &schema).unwrap();
        let mut reader = ArrowIpcReader;
        let decoded = reader.read_rows(&bytes, &schema).unwrap();
        assert_rows_equal(&decoded, &rows);
    }

    #[test]
    fn arrow_ipc_ps_timestamp_exports_as_nanosecond_lossy() {
        use crate::format::ColumnSpec;
        use crate::row_codec::StreamValue;
        use arrow::array::{Array, TimestampNanosecondArray};
        use arrow::datatypes::{DataType as AT, TimeUnit};
        use arrow::ipc::reader::StreamReader;

        // A TIMESTAMP(9) column: i128 picoseconds. Arrow has no ps unit, so it
        // must export as Timestamp(Nanosecond) with values truncated ps/1000.
        let schema = vec![ColumnSpec::with_precision(
            "t9",
            TypeId::Timestamp,
            Some(9),
        )];
        // 1_700_000_000_123_456_789_000 ps -> 1_700_000_000_123_456_789 ns.
        // 1_999 ps -> 1 ns (truncating, not rounding). A null passes through.
        let rows = vec![
            vec![StreamValue::I128(1_700_000_000_123_456_789_000)],
            vec![StreamValue::I128(1_999)],
            vec![StreamValue::Null],
        ];
        let mut writer = ArrowIpcWriter;
        let bytes = writer.write_rows(&rows, &schema).unwrap();

        let mut rdr = StreamReader::try_new(std::io::Cursor::new(bytes), None).unwrap();
        let batch = rdr.next().unwrap().unwrap();
        assert_eq!(
            batch.schema().field(0).data_type(),
            &AT::Timestamp(TimeUnit::Nanosecond, None),
            "ps column must export as a Nanosecond timestamp"
        );
        let a = batch
            .column(0)
            .as_any()
            .downcast_ref::<TimestampNanosecondArray>()
            .expect("nanosecond timestamp array");
        assert_eq!(a.value(0), 1_700_000_000_123_456_789);
        assert_eq!(a.value(1), 1, "ps->ns truncates the low 3 digits");
        assert!(a.is_null(2), "null passes through");
    }

    #[test]
    fn arrow_ipc_infer_schema_matches_writer() {
        let schema = sample_schema();
        let rows = sample_rows();
        let mut writer = ArrowIpcWriter;
        let bytes = writer.write_rows(&rows, &schema).unwrap();
        let inferred = infer_arrow_ipc_schema(&bytes).unwrap();
        assert_eq!(inferred.len(), schema.len());
        assert_eq!(inferred[0].name, "id");
        assert_eq!(inferred[0].type_id, TypeId::Int64);
        assert_eq!(inferred[1].type_id, TypeId::Text);
        assert_eq!(inferred[2].type_id, TypeId::Boolean);
        assert_eq!(inferred[3].type_id, TypeId::Float64);
    }
}

// -----------------------------------------------------------------------------
// Parquet format
// -----------------------------------------------------------------------------
//
// Reads and writes Apache Parquet files. The reader uses the Arrow record
// batch reader over an in-memory cursor, the writer packs the rows into a
// single RecordBatch and passes it to ArrowWriter. Schema mapping follows
// format/schema.rs, and column conversion routes through record_batch.rs.

use super::{ColumnSpec, FormatReader, FormatWriter};
use super::record_batch::{batch_to_rows, rows_to_batch};
use super::schema::{arrow_to_type_id, type_id_to_arrow};
use crate::row_codec::StreamValue;
use arrow::array::RecordBatch;
use arrow::datatypes::{Field, Schema};
use bytes::Bytes;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use std::sync::Arc;
use zyron_common::{Result, ZyronError};

pub struct ParquetReader;

impl FormatReader for ParquetReader {
    fn read_rows(
        &mut self,
        bytes: &[u8],
        schema: &[ColumnSpec],
    ) -> Result<Vec<Vec<StreamValue>>> {
        let data = Bytes::copy_from_slice(bytes);
        let builder = ParquetRecordBatchReaderBuilder::try_new(data).map_err(|e| {
            ZyronError::StreamingError(format!("parquet: reader init error: {e}"))
        })?;
        let reader = builder.build().map_err(|e| {
            ZyronError::StreamingError(format!("parquet: reader build error: {e}"))
        })?;
        let mut rows = Vec::new();
        for batch in reader {
            let batch = batch.map_err(|e| {
                ZyronError::StreamingError(format!("parquet: read error: {e}"))
            })?;
            batch_to_rows(&batch, schema, &mut rows)?;
        }
        Ok(rows)
    }
}

pub struct ParquetWriter;

impl FormatWriter for ParquetWriter {
    fn write_rows(
        &mut self,
        rows: &[Vec<StreamValue>],
        schema: &[ColumnSpec],
    ) -> Result<Vec<u8>> {
        let fields: Vec<Field> = schema
            .iter()
            .map(|c| Field::new(&c.name, type_id_to_arrow(c.type_id), true))
            .collect();
        let arrow_schema = Arc::new(Schema::new(fields));
        let batch: RecordBatch = rows_to_batch(rows, schema, arrow_schema.clone())?;
        let mut buf: Vec<u8> = Vec::new();
        {
            let mut writer = ArrowWriter::try_new(&mut buf, arrow_schema.clone(), None)
                .map_err(|e| {
                    ZyronError::StreamingError(format!("parquet: writer init error: {e}"))
                })?;
            writer.write(&batch).map_err(|e| {
                ZyronError::StreamingError(format!("parquet: write error: {e}"))
            })?;
            writer.close().map_err(|e| {
                ZyronError::StreamingError(format!("parquet: close error: {e}"))
            })?;
        }
        Ok(buf)
    }
}

// -----------------------------------------------------------------------------
// Schema inference
// -----------------------------------------------------------------------------

/// Reads the Parquet footer from an in-memory buffer and returns the inferred
/// column list. Field order matches the file's Arrow schema.
pub fn infer_parquet_schema(bytes: &[u8]) -> Result<Vec<ColumnSpec>> {
    let data = Bytes::copy_from_slice(bytes);
    let builder = ParquetRecordBatchReaderBuilder::try_new(data).map_err(|e| {
        ZyronError::StreamingError(format!("parquet: schema read error: {e}"))
    })?;
    let schema = builder.schema();
    let mut cols = Vec::with_capacity(schema.fields().len());
    for field in schema.fields() {
        let type_id = arrow_to_type_id(field.data_type())?;
        cols.push(ColumnSpec { name: field.name().to_string(), type_id });
    }
    Ok(cols)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{assert_rows_equal, sample_rows, sample_schema};
    use zyron_common::TypeId;

    #[test]
    fn parquet_roundtrip() {
        let schema = sample_schema();
        let rows = sample_rows();
        let mut writer = ParquetWriter;
        let bytes = writer.write_rows(&rows, &schema).unwrap();
        let mut reader = ParquetReader;
        let decoded = reader.read_rows(&bytes, &schema).unwrap();
        assert_rows_equal(&decoded, &rows);
    }

    #[test]
    fn parquet_infer_schema_matches_writer() {
        let schema = sample_schema();
        let rows = sample_rows();
        let mut writer = ParquetWriter;
        let bytes = writer.write_rows(&rows, &schema).unwrap();
        let inferred = infer_parquet_schema(&bytes).unwrap();
        assert_eq!(inferred.len(), schema.len());
        for (a, b) in inferred.iter().zip(schema.iter()) {
            assert_eq!(a.name, b.name);
        }
        // The sample schema uses Int64, Varchar, Boolean, Float64. Varchar
        // is written as Arrow Utf8 and inference maps that back to Text.
        assert_eq!(inferred[0].type_id, TypeId::Int64);
        assert_eq!(inferred[1].type_id, TypeId::Text);
        assert_eq!(inferred[2].type_id, TypeId::Boolean);
        assert_eq!(inferred[3].type_id, TypeId::Float64);
    }
}

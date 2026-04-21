// -----------------------------------------------------------------------------
// Format adapters for streaming external sources and sinks
// -----------------------------------------------------------------------------
//
// Each format implements read_rows, taking bytes in and emitting
// Vec<Vec<StreamValue>>, and write_rows, taking rows in and emitting bytes.
// The caller supplies a schema of ColumnSpec entries drawn from the catalog's
// column types. The format layer does not know about OpenDAL, cloud storage,
// or catalog entries, it only converts between raw bytes and row values.

use crate::row_codec::StreamValue;
use zyron_common::{Result, TypeId};

pub mod arrow_ipc;
pub mod avro;
pub mod csv;
pub mod json;
pub mod jsonl;
pub mod parquet;
pub mod record_batch;
pub mod schema;

pub use arrow_ipc::infer_arrow_ipc_schema;
pub use avro::infer_avro_schema;
pub use parquet::infer_parquet_schema;
use zyron_common::ZyronError;

// -----------------------------------------------------------------------------
// FormatKind
// -----------------------------------------------------------------------------

/// Identifies which on-disk format a reader or writer handles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormatKind {
    Json,
    JsonLines,
    Csv,
    Parquet,
    ArrowIpc,
    Avro,
}

// -----------------------------------------------------------------------------
// ColumnSpec
// -----------------------------------------------------------------------------

/// Column descriptor for format readers and writers. The name drives JSON
/// and CSV header matching, and Parquet, Arrow, or Avro schema generation.
/// The type_id drives coercion to and from StreamValue.
#[derive(Debug, Clone)]
pub struct ColumnSpec {
    pub name: String,
    pub type_id: TypeId,
}

// -----------------------------------------------------------------------------
// FormatReader / FormatWriter traits
// -----------------------------------------------------------------------------

/// Streaming reader trait. Implementations consume bytes and emit rows.
/// Record-delimited formats like JSON Lines or CSV can be chunked by the
/// caller. Parquet, Arrow IPC, and Avro are self-framing and read the whole
/// buffer at once.
pub trait FormatReader: Send {
    fn read_rows(&mut self, bytes: &[u8], schema: &[ColumnSpec]) -> Result<Vec<Vec<StreamValue>>>;
}

/// Streaming writer trait. Implementations convert rows into bytes. For
/// columnar formats this is a whole-file encode, for record-delimited
/// formats each row is appended in turn.
pub trait FormatWriter: Send {
    fn write_rows(&mut self, rows: &[Vec<StreamValue>], schema: &[ColumnSpec]) -> Result<Vec<u8>>;
}

// -----------------------------------------------------------------------------
// Factories
// -----------------------------------------------------------------------------

/// Builds a reader for the given FormatKind.
pub fn reader_for(kind: FormatKind) -> Box<dyn FormatReader> {
    match kind {
        FormatKind::Json => Box::new(json::JsonReader),
        FormatKind::JsonLines => Box::new(jsonl::JsonLinesReader),
        FormatKind::Csv => Box::new(csv::CsvReader),
        FormatKind::Parquet => Box::new(parquet::ParquetReader),
        FormatKind::ArrowIpc => Box::new(arrow_ipc::ArrowIpcReader),
        FormatKind::Avro => Box::new(avro::AvroReader),
    }
}

/// Reads a file's embedded schema and returns the inferred column list.
/// Supported only for self-describing formats (Parquet, Arrow IPC, Avro).
/// Record-delimited text formats (JSON, JSONL, CSV) have no embedded schema
/// and return an error.
pub fn infer_schema(kind: FormatKind, bytes: &[u8]) -> Result<Vec<ColumnSpec>> {
    match kind {
        FormatKind::Parquet => infer_parquet_schema(bytes),
        FormatKind::ArrowIpc => infer_arrow_ipc_schema(bytes),
        FormatKind::Avro => infer_avro_schema(bytes),
        FormatKind::Json | FormatKind::JsonLines | FormatKind::Csv => Err(ZyronError::PlanError(
            "schema inference requires a self-describing format (Parquet, Arrow IPC, or Avro)"
                .to_string(),
        )),
    }
}

/// Builds a writer for the given FormatKind.
pub fn writer_for(kind: FormatKind) -> Box<dyn FormatWriter> {
    match kind {
        FormatKind::Json => Box::new(json::JsonWriter),
        FormatKind::JsonLines => Box::new(jsonl::JsonLinesWriter),
        FormatKind::Csv => Box::new(csv::CsvWriter),
        FormatKind::Parquet => Box::new(parquet::ParquetWriter),
        FormatKind::ArrowIpc => Box::new(arrow_ipc::ArrowIpcWriter),
        FormatKind::Avro => Box::new(avro::AvroWriter),
    }
}

// -----------------------------------------------------------------------------
// Shared test helpers
// -----------------------------------------------------------------------------

#[cfg(test)]
pub(crate) fn sample_schema() -> Vec<ColumnSpec> {
    vec![
        ColumnSpec {
            name: "id".to_string(),
            type_id: TypeId::Int64,
        },
        ColumnSpec {
            name: "name".to_string(),
            type_id: TypeId::Varchar,
        },
        ColumnSpec {
            name: "active".to_string(),
            type_id: TypeId::Boolean,
        },
        ColumnSpec {
            name: "score".to_string(),
            type_id: TypeId::Float64,
        },
    ]
}

#[cfg(test)]
pub(crate) fn sample_rows() -> Vec<Vec<StreamValue>> {
    vec![
        vec![
            StreamValue::I64(1),
            StreamValue::Utf8("alpha".to_string()),
            StreamValue::Bool(true),
            StreamValue::F64(1.5),
        ],
        vec![
            StreamValue::I64(-2),
            StreamValue::Utf8("ünîçødé".to_string()),
            StreamValue::Bool(false),
            StreamValue::F64(-3.25),
        ],
        vec![
            StreamValue::I64(3),
            StreamValue::Null,
            StreamValue::Bool(true),
            StreamValue::F64(0.0),
        ],
    ]
}

#[cfg(test)]
pub(crate) fn assert_rows_equal(actual: &[Vec<StreamValue>], expected: &[Vec<StreamValue>]) {
    assert_eq!(actual.len(), expected.len(), "row count mismatch");
    for (ri, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(a.len(), e.len(), "column count mismatch at row {ri}");
        for (ci, (av, ev)) in a.iter().zip(e.iter()).enumerate() {
            let eq = match (av, ev) {
                (StreamValue::Null, StreamValue::Null) => true,
                (StreamValue::Bool(x), StreamValue::Bool(y)) => x == y,
                (StreamValue::I64(x), StreamValue::I64(y)) => x == y,
                (StreamValue::I128(x), StreamValue::I128(y)) => x == y,
                (StreamValue::F64(x), StreamValue::F64(y)) => {
                    (x - y).abs() < 1e-9 || (x.is_nan() && y.is_nan())
                }
                (StreamValue::Utf8(x), StreamValue::Utf8(y)) => x == y,
                (StreamValue::Binary(x), StreamValue::Binary(y)) => x == y,
                _ => false,
            };
            assert!(
                eq,
                "row {ri} col {ci} mismatch: actual={:?} expected={:?}",
                av, ev
            );
        }
    }
}

#[cfg(test)]
mod inference_tests {
    use super::*;

    #[test]
    fn infer_schema_rejects_record_delimited_formats() {
        // JSON, JSONL, and CSV do not carry schema in their bytes, inference
        // must surface an error so callers know to supply COLUMNS (...).
        for kind in [FormatKind::Json, FormatKind::JsonLines, FormatKind::Csv] {
            let result = infer_schema(kind, b"id,name\n1,a\n");
            assert!(result.is_err(), "expected error for {:?}", kind);
        }
    }
}

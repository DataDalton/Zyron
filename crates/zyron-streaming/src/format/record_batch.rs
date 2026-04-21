// -----------------------------------------------------------------------------
// RecordBatch bridge
// -----------------------------------------------------------------------------
//
// Shared conversion routines between Vec<Vec<StreamValue>> and Arrow
// RecordBatch. Used by the Parquet and Arrow IPC format implementations.
// Builder arms are dispatched per TypeId and append StreamValue into the
// matching typed Arrow array builder.

use super::ColumnSpec;
use crate::row_codec::StreamValue;
use arrow::array::{
    Array, ArrayRef, BinaryArray, BinaryBuilder, BooleanArray, BooleanBuilder, Decimal128Array,
    Decimal128Builder, Float32Array, Float32Builder, Float64Array, Float64Builder, Int8Array,
    Int8Builder, Int16Array, Int16Builder, Int32Array, Int32Builder, Int64Array, Int64Builder,
    RecordBatch, StringArray, StringBuilder, UInt8Array, UInt8Builder, UInt16Array, UInt16Builder,
    UInt32Array, UInt32Builder, UInt64Array, UInt64Builder,
};
use arrow::datatypes::Schema;
use std::sync::Arc;
use zyron_common::{Result, TypeId, ZyronError};

// -----------------------------------------------------------------------------
// rows_to_batch
// -----------------------------------------------------------------------------

pub fn rows_to_batch(
    rows: &[Vec<StreamValue>],
    schema: &[ColumnSpec],
    arrow_schema: Arc<Schema>,
) -> Result<RecordBatch> {
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(schema.len());
    for (ci, col) in schema.iter().enumerate() {
        arrays.push(build_column(ci, col.type_id, rows)?);
    }
    RecordBatch::try_new(arrow_schema, arrays)
        .map_err(|e| ZyronError::StreamingError(format!("record_batch: build error: {e}")))
}

fn build_column(ci: usize, t: TypeId, rows: &[Vec<StreamValue>]) -> Result<ArrayRef> {
    let n = rows.len();
    match t {
        TypeId::Boolean => {
            let mut b = BooleanBuilder::with_capacity(n);
            for row in rows {
                match &row[ci] {
                    StreamValue::Null => b.append_null(),
                    StreamValue::Bool(v) => b.append_value(*v),
                    other => return Err(col_type_err(ci, t, other)),
                }
            }
            Ok(Arc::new(b.finish()) as ArrayRef)
        }
        TypeId::Int8 => build_i8(ci, rows),
        TypeId::Int16 => build_i16(ci, rows),
        TypeId::Int32 | TypeId::Date => build_i32(ci, rows),
        TypeId::Int64 | TypeId::Time | TypeId::Timestamp | TypeId::TimestampTz => {
            build_i64(ci, rows)
        }
        TypeId::UInt8 => build_u8(ci, rows),
        TypeId::UInt16 => build_u16(ci, rows),
        TypeId::UInt32 => build_u32(ci, rows),
        TypeId::UInt64 => build_u64(ci, rows),
        TypeId::Float32 => {
            let mut b = Float32Builder::with_capacity(n);
            for row in rows {
                match &row[ci] {
                    StreamValue::Null => b.append_null(),
                    StreamValue::F64(v) => b.append_value(*v as f32),
                    StreamValue::I64(v) => b.append_value(*v as f32),
                    other => return Err(col_type_err(ci, t, other)),
                }
            }
            Ok(Arc::new(b.finish()) as ArrayRef)
        }
        TypeId::Float64 => {
            let mut b = Float64Builder::with_capacity(n);
            for row in rows {
                match &row[ci] {
                    StreamValue::Null => b.append_null(),
                    StreamValue::F64(v) => b.append_value(*v),
                    StreamValue::I64(v) => b.append_value(*v as f64),
                    other => return Err(col_type_err(ci, t, other)),
                }
            }
            Ok(Arc::new(b.finish()) as ArrayRef)
        }
        TypeId::Int128 | TypeId::Decimal | TypeId::UInt128 => {
            let mut b = Decimal128Builder::with_capacity(n)
                .with_precision_and_scale(38, 0)
                .map_err(|e| {
                    ZyronError::StreamingError(format!("record_batch: decimal setup error: {e}"))
                })?;
            for row in rows {
                match &row[ci] {
                    StreamValue::Null => b.append_null(),
                    StreamValue::I128(v) => b.append_value(*v),
                    StreamValue::I64(v) => b.append_value(*v as i128),
                    other => return Err(col_type_err(ci, t, other)),
                }
            }
            Ok(Arc::new(b.finish()) as ArrayRef)
        }
        TypeId::Char | TypeId::Varchar | TypeId::Text | TypeId::Json | TypeId::Jsonb => {
            let mut b = StringBuilder::with_capacity(n, n * 8);
            for row in rows {
                match &row[ci] {
                    StreamValue::Null => b.append_null(),
                    StreamValue::Utf8(s) => b.append_value(s),
                    other => return Err(col_type_err(ci, t, other)),
                }
            }
            Ok(Arc::new(b.finish()) as ArrayRef)
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
        | TypeId::Quantity => {
            let mut b = BinaryBuilder::with_capacity(n, n * 8);
            for row in rows {
                match &row[ci] {
                    StreamValue::Null => b.append_null(),
                    StreamValue::Binary(bs) => b.append_value(bs),
                    other => return Err(col_type_err(ci, t, other)),
                }
            }
            Ok(Arc::new(b.finish()) as ArrayRef)
        }
        TypeId::Null => Err(ZyronError::StreamingError(format!(
            "record_batch: Null column type at index {ci} is not supported"
        ))),
    }
}

macro_rules! build_int {
    ($fnname:ident, $builder:ident, $cast:ty) => {
        fn $fnname(ci: usize, rows: &[Vec<StreamValue>]) -> Result<ArrayRef> {
            let mut b = $builder::with_capacity(rows.len());
            for row in rows {
                match &row[ci] {
                    StreamValue::Null => b.append_null(),
                    StreamValue::I64(v) => b.append_value(*v as $cast),
                    StreamValue::Bool(v) => b.append_value(if *v { 1 } else { 0 }),
                    other => {
                        return Err(ZyronError::StreamingError(format!(
                            "record_batch: col {ci} expected integer, got {other:?}"
                        )));
                    }
                }
            }
            Ok(Arc::new(b.finish()) as ArrayRef)
        }
    };
}

build_int!(build_i8, Int8Builder, i8);
build_int!(build_i16, Int16Builder, i16);
build_int!(build_i32, Int32Builder, i32);
build_int!(build_i64, Int64Builder, i64);
build_int!(build_u8, UInt8Builder, u8);
build_int!(build_u16, UInt16Builder, u16);
build_int!(build_u32, UInt32Builder, u32);
build_int!(build_u64, UInt64Builder, u64);

fn col_type_err(ci: usize, t: TypeId, got: &StreamValue) -> ZyronError {
    ZyronError::StreamingError(format!(
        "record_batch: col {ci} expected {t:?}, got {got:?}"
    ))
}

// -----------------------------------------------------------------------------
// batch_to_rows
// -----------------------------------------------------------------------------

pub fn batch_to_rows(
    batch: &RecordBatch,
    schema: &[ColumnSpec],
    out: &mut Vec<Vec<StreamValue>>,
) -> Result<()> {
    if batch.num_columns() != schema.len() {
        return Err(ZyronError::StreamingError(format!(
            "record_batch: column count {} does not match schema {}",
            batch.num_columns(),
            schema.len()
        )));
    }
    let n = batch.num_rows();
    let start = out.len();
    for _ in 0..n {
        out.push(Vec::with_capacity(schema.len()));
    }
    for (ci, col) in schema.iter().enumerate() {
        let arr = batch.column(ci);
        extract_column(col.type_id, arr.as_ref(), &mut out[start..])?;
    }
    Ok(())
}

fn extract_column(t: TypeId, arr: &dyn Array, rows: &mut [Vec<StreamValue>]) -> Result<()> {
    let n = arr.len();
    for i in 0..n {
        let v = if arr.is_null(i) {
            StreamValue::Null
        } else {
            extract_scalar(t, arr, i)?
        };
        rows[i].push(v);
    }
    Ok(())
}

fn extract_scalar(t: TypeId, arr: &dyn Array, i: usize) -> Result<StreamValue> {
    match t {
        TypeId::Boolean => Ok(StreamValue::Bool(
            arr.as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| dc_err("Boolean"))?
                .value(i),
        )),
        TypeId::Int8 => Ok(StreamValue::I64(
            arr.as_any()
                .downcast_ref::<Int8Array>()
                .ok_or_else(|| dc_err("Int8"))?
                .value(i) as i64,
        )),
        TypeId::Int16 => Ok(StreamValue::I64(
            arr.as_any()
                .downcast_ref::<Int16Array>()
                .ok_or_else(|| dc_err("Int16"))?
                .value(i) as i64,
        )),
        TypeId::Int32 | TypeId::Date => Ok(StreamValue::I64(
            arr.as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| dc_err("Int32"))?
                .value(i) as i64,
        )),
        TypeId::Int64 | TypeId::Time | TypeId::Timestamp | TypeId::TimestampTz => {
            Ok(StreamValue::I64(
                arr.as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| dc_err("Int64"))?
                    .value(i),
            ))
        }
        TypeId::UInt8 => Ok(StreamValue::I64(
            arr.as_any()
                .downcast_ref::<UInt8Array>()
                .ok_or_else(|| dc_err("UInt8"))?
                .value(i) as i64,
        )),
        TypeId::UInt16 => Ok(StreamValue::I64(
            arr.as_any()
                .downcast_ref::<UInt16Array>()
                .ok_or_else(|| dc_err("UInt16"))?
                .value(i) as i64,
        )),
        TypeId::UInt32 => Ok(StreamValue::I64(
            arr.as_any()
                .downcast_ref::<UInt32Array>()
                .ok_or_else(|| dc_err("UInt32"))?
                .value(i) as i64,
        )),
        TypeId::UInt64 => Ok(StreamValue::I64(
            arr.as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| dc_err("UInt64"))?
                .value(i) as i64,
        )),
        TypeId::Float32 => Ok(StreamValue::F64(
            arr.as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| dc_err("Float32"))?
                .value(i) as f64,
        )),
        TypeId::Float64 => Ok(StreamValue::F64(
            arr.as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| dc_err("Float64"))?
                .value(i),
        )),
        TypeId::Int128 | TypeId::Decimal | TypeId::UInt128 => Ok(StreamValue::I128(
            arr.as_any()
                .downcast_ref::<Decimal128Array>()
                .ok_or_else(|| dc_err("Decimal128"))?
                .value(i),
        )),
        TypeId::Char | TypeId::Varchar | TypeId::Text | TypeId::Json | TypeId::Jsonb => {
            Ok(StreamValue::Utf8(
                arr.as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| dc_err("String"))?
                    .value(i)
                    .to_string(),
            ))
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
        | TypeId::Quantity => Ok(StreamValue::Binary(
            arr.as_any()
                .downcast_ref::<BinaryArray>()
                .ok_or_else(|| dc_err("Binary"))?
                .value(i)
                .to_vec(),
        )),
        TypeId::Null => Ok(StreamValue::Null),
    }
}

fn dc_err(kind: &str) -> ZyronError {
    ZyronError::StreamingError(format!("record_batch: downcast to {kind} failed"))
}

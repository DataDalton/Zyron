//! DataBatch: columnar batch of rows for vectorized query execution.
//!
//! Provides the DataBatch type that holds typed columns with null bitmaps,
//! and conversion functions between the NSM (N-ary Storage Model) tuple
//! format used by the storage engine and the columnar batch format used
//! for query processing.

use zyron_catalog::ColumnEntry;
use zyron_common::TypeId;
use zyron_planner::logical::LogicalColumn;
use zyron_storage::Tuple;

use crate::column::{Column, ColumnData, NullBitmap, ScalarValue};

/// Number of rows per execution batch.
pub const BATCH_SIZE: usize = 1024;

// ---------------------------------------------------------------------------
// DataBatch
// ---------------------------------------------------------------------------

/// A columnar batch of rows. Each column holds a typed vector of values
/// with a null bitmap. All columns have the same number of rows.
#[derive(Debug, Clone)]
pub struct DataBatch {
    pub columns: Vec<Column>,
    pub num_rows: usize,
}

impl DataBatch {
    /// Creates a batch from pre-built columns. All columns must have the same length.
    pub fn new(columns: Vec<Column>) -> Self {
        let num_rows = columns.first().map_or(0, |c| c.len());
        debug_assert!(columns.iter().all(|c| c.len() == num_rows));
        Self { columns, num_rows }
    }

    /// Creates an empty batch with no rows and no columns.
    pub fn empty() -> Self {
        Self {
            columns: Vec::new(),
            num_rows: 0,
        }
    }

    /// Returns a single column by index.
    pub fn column(&self, idx: usize) -> &Column {
        &self.columns[idx]
    }

    /// Number of columns.
    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    /// Selects rows where mask[i] is true.
    pub fn filter(&self, mask: &[bool]) -> Self {
        let columns: Vec<Column> = self.columns.iter().map(|c| c.filter(mask)).collect();
        let num_rows = columns.first().map_or(0, |c| c.len());
        Self { columns, num_rows }
    }

    /// Reorders rows by indices.
    pub fn take(&self, indices: &[u32]) -> Self {
        let columns: Vec<Column> = self.columns.iter().map(|c| c.take(indices)).collect();
        let num_rows = indices.len();
        Self { columns, num_rows }
    }

    /// Extracts a contiguous sub-range.
    pub fn slice(&self, offset: usize, len: usize) -> Self {
        let actual_len = len.min(self.num_rows.saturating_sub(offset));
        let columns: Vec<Column> = self
            .columns
            .iter()
            .map(|c| c.slice(offset, actual_len))
            .collect();
        Self {
            columns,
            num_rows: actual_len,
        }
    }
}

// ---------------------------------------------------------------------------
// Column builders for incremental construction
// ---------------------------------------------------------------------------

/// Builder for constructing columns row by row during tuple decoding.
pub struct ColumnBuilder {
    data: ColumnData,
    nulls: NullBitmap,
    type_id: TypeId,
}

impl ColumnBuilder {
    pub fn new(type_id: TypeId, capacity: usize) -> Self {
        Self {
            data: ColumnData::with_capacity(type_id, capacity),
            nulls: NullBitmap::empty(),
            type_id,
        }
    }

    pub fn push(&mut self, scalar: &ScalarValue) {
        let is_null = scalar.is_null();
        self.nulls.push(is_null);
        self.data.push_scalar(scalar);
    }

    pub fn push_null(&mut self) {
        self.nulls.push(true);
        self.data.push_scalar(&ScalarValue::Null);
    }

    pub fn finish(self) -> Column {
        Column::with_nulls(self.data, self.nulls, self.type_id)
    }
}

/// Creates a vector of column builders for the given logical columns.
pub fn create_builders(columns: &[LogicalColumn], capacity: usize) -> Vec<ColumnBuilder> {
    columns
        .iter()
        .map(|col| ColumnBuilder::new(col.type_id, capacity))
        .collect()
}

/// Finalizes builders into a DataBatch.
pub fn finalize_builders(builders: Vec<ColumnBuilder>) -> DataBatch {
    let columns: Vec<Column> = builders.into_iter().map(|b| b.finish()).collect();
    DataBatch::new(columns)
}

// ---------------------------------------------------------------------------
// Tuple decode: NSM bytes -> column builders
// ---------------------------------------------------------------------------

/// Decodes one tuple's data bytes into column builders.
///
/// Tuple data layout (NSM, little-endian):
/// - Null bitmap: ceil(num_columns / 8) bytes, bit N set = column N is null
/// - Column values in ordinal order:
///   - Fixed-size types: inline at TypeId::fixed_size() bytes (zeroed if null)
///   - Variable-length types: 4-byte LE length prefix + data bytes (length=0, no data if null)
pub fn decode_tuple_into_builders(
    data: &[u8],
    columns: &[ColumnEntry],
    builders: &mut [ColumnBuilder],
) {
    let num_cols = columns.len();
    let null_bitmap_len = (num_cols + 7) / 8;
    let null_bitmap = &data[..null_bitmap_len];
    let mut offset = null_bitmap_len;

    for (i, col) in columns.iter().enumerate() {
        let is_null = (null_bitmap[i / 8] >> (i % 8)) & 1 == 1;

        if let Some(fixed_size) = col.type_id.fixed_size() {
            if is_null {
                builders[i].push_null();
                offset += fixed_size;
            } else {
                let value_bytes = &data[offset..offset + fixed_size];
                let scalar = decode_fixed_scalar(col.type_id, value_bytes);
                builders[i].push(&scalar);
                offset += fixed_size;
            }
        } else {
            // Variable-length: 4-byte LE length prefix
            let len = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]) as usize;
            offset += 4;

            if is_null {
                builders[i].push_null();
                offset += len;
            } else {
                let value_bytes = &data[offset..offset + len];
                let scalar = decode_varlen_scalar(col.type_id, value_bytes);
                builders[i].push(&scalar);
                offset += len;
            }
        }
    }
}

/// Decodes a fixed-size value from raw bytes into a ScalarValue.
fn decode_fixed_scalar(type_id: TypeId, bytes: &[u8]) -> ScalarValue {
    match type_id {
        TypeId::Null => ScalarValue::Null,
        TypeId::Boolean => ScalarValue::Boolean(bytes[0] != 0),
        TypeId::Int8 => ScalarValue::Int8(i8::from_le_bytes([bytes[0]])),
        TypeId::Int16 => ScalarValue::Int16(i16::from_le_bytes(bytes[..2].try_into().unwrap())),
        TypeId::Int32 | TypeId::Date => {
            ScalarValue::Int32(i32::from_le_bytes(bytes[..4].try_into().unwrap()))
        }
        TypeId::Int64 | TypeId::Time | TypeId::Timestamp | TypeId::TimestampTz => {
            ScalarValue::Int64(i64::from_le_bytes(bytes[..8].try_into().unwrap()))
        }
        TypeId::Int128 | TypeId::Decimal => {
            ScalarValue::Int128(i128::from_le_bytes(bytes[..16].try_into().unwrap()))
        }
        TypeId::UInt8 => ScalarValue::UInt8(bytes[0]),
        TypeId::UInt16 => ScalarValue::UInt16(u16::from_le_bytes(bytes[..2].try_into().unwrap())),
        TypeId::UInt32 => ScalarValue::UInt32(u32::from_le_bytes(bytes[..4].try_into().unwrap())),
        TypeId::UInt64 => ScalarValue::UInt64(u64::from_le_bytes(bytes[..8].try_into().unwrap())),
        TypeId::UInt128 => {
            ScalarValue::Int128(i128::from_le_bytes(bytes[..16].try_into().unwrap()))
        }
        TypeId::Float32 => ScalarValue::Float32(f32::from_le_bytes(bytes[..4].try_into().unwrap())),
        TypeId::Float64 => ScalarValue::Float64(f64::from_le_bytes(bytes[..8].try_into().unwrap())),
        TypeId::Uuid => ScalarValue::FixedBinary16(bytes[..16].try_into().unwrap()),
        TypeId::Interval => {
            let arr: [u8; 16] = bytes[..16].try_into().unwrap();
            ScalarValue::Interval(zyron_common::Interval::from_le_bytes(&arr))
        }
        _ => ScalarValue::Null,
    }
}

/// Decodes a variable-length value from raw bytes into a ScalarValue.
fn decode_varlen_scalar(type_id: TypeId, bytes: &[u8]) -> ScalarValue {
    match type_id {
        TypeId::Char | TypeId::Varchar | TypeId::Text | TypeId::Json | TypeId::Jsonb => {
            ScalarValue::Utf8(String::from_utf8_lossy(bytes).into_owned())
        }
        TypeId::Binary
        | TypeId::Varbinary
        | TypeId::Bytea
        | TypeId::Array
        | TypeId::Composite
        | TypeId::Vector => ScalarValue::Binary(bytes.to_vec()),
        _ => ScalarValue::Null,
    }
}

// ---------------------------------------------------------------------------
// Tuple encode: DataBatch row -> NSM bytes
// ---------------------------------------------------------------------------

/// Encodes one row from a DataBatch into tuple data bytes (NSM format).
pub fn encode_row(batch: &DataBatch, row_idx: usize, columns: &[ColumnEntry]) -> Vec<u8> {
    let num_cols = columns.len();
    let null_bitmap_len = (num_cols + 7) / 8;
    let mut buf = Vec::with_capacity(null_bitmap_len + num_cols * 8);
    buf.resize(null_bitmap_len, 0u8);

    for (i, col) in columns.iter().enumerate() {
        let column = &batch.columns[i];
        let is_null = column.is_null(row_idx);

        if is_null {
            buf[i / 8] |= 1 << (i % 8);
        }

        if let Some(fixed_size) = col.type_id.fixed_size() {
            if is_null {
                buf.extend(std::iter::repeat(0u8).take(fixed_size));
            } else {
                encode_fixed_scalar(&mut buf, col.type_id, &column.data.get_scalar(row_idx));
            }
        } else if is_null {
            buf.extend_from_slice(&0u32.to_le_bytes());
        } else {
            encode_varlen_scalar(&mut buf, col.type_id, &column.data.get_scalar(row_idx));
        }
    }

    buf
}

/// Encodes a fixed-size scalar value into the output buffer.
fn encode_fixed_scalar(buf: &mut Vec<u8>, type_id: TypeId, scalar: &ScalarValue) {
    match (type_id, scalar) {
        (TypeId::Null, _) => {}
        (TypeId::Boolean, ScalarValue::Boolean(v)) => buf.push(if *v { 1 } else { 0 }),
        (TypeId::Int8, ScalarValue::Int8(v)) => buf.extend_from_slice(&v.to_le_bytes()),
        (TypeId::Int16, ScalarValue::Int16(v)) => buf.extend_from_slice(&v.to_le_bytes()),
        (TypeId::Int32 | TypeId::Date, ScalarValue::Int32(v)) => {
            buf.extend_from_slice(&v.to_le_bytes())
        }
        (
            TypeId::Int64 | TypeId::Time | TypeId::Timestamp | TypeId::TimestampTz,
            ScalarValue::Int64(v),
        ) => buf.extend_from_slice(&v.to_le_bytes()),
        (TypeId::Int128 | TypeId::Decimal | TypeId::UInt128, ScalarValue::Int128(v)) => {
            buf.extend_from_slice(&v.to_le_bytes())
        }
        (TypeId::UInt8, ScalarValue::UInt8(v)) => buf.extend_from_slice(&v.to_le_bytes()),
        (TypeId::UInt16, ScalarValue::UInt16(v)) => buf.extend_from_slice(&v.to_le_bytes()),
        (TypeId::UInt32, ScalarValue::UInt32(v)) => buf.extend_from_slice(&v.to_le_bytes()),
        (TypeId::UInt64, ScalarValue::UInt64(v)) => buf.extend_from_slice(&v.to_le_bytes()),
        (TypeId::Float32, ScalarValue::Float32(v)) => buf.extend_from_slice(&v.to_le_bytes()),
        (TypeId::Float64, ScalarValue::Float64(v)) => buf.extend_from_slice(&v.to_le_bytes()),
        (TypeId::Uuid, ScalarValue::FixedBinary16(v)) => buf.extend_from_slice(v),
        (TypeId::Interval, ScalarValue::Interval(i)) => buf.extend_from_slice(&i.to_le_bytes()),
        _ => {
            if let Some(size) = type_id.fixed_size() {
                buf.extend(std::iter::repeat(0u8).take(size));
            }
        }
    }
}

/// Encodes a variable-length scalar value with 4-byte LE length prefix.
fn encode_varlen_scalar(buf: &mut Vec<u8>, type_id: TypeId, scalar: &ScalarValue) {
    match (type_id, scalar) {
        (
            TypeId::Char | TypeId::Varchar | TypeId::Text | TypeId::Json | TypeId::Jsonb,
            ScalarValue::Utf8(s),
        ) => {
            let bytes = s.as_bytes();
            buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            buf.extend_from_slice(bytes);
        }
        (
            TypeId::Binary
            | TypeId::Varbinary
            | TypeId::Bytea
            | TypeId::Array
            | TypeId::Composite
            | TypeId::Vector,
            ScalarValue::Binary(b),
        ) => {
            buf.extend_from_slice(&(b.len() as u32).to_le_bytes());
            buf.extend_from_slice(b);
        }
        _ => {
            buf.extend_from_slice(&0u32.to_le_bytes());
        }
    }
}

/// Converts an entire DataBatch to storage Tuples.
pub fn batch_to_tuples(batch: &DataBatch, columns: &[ColumnEntry], xmin: u32) -> Vec<Tuple> {
    let mut tuples = Vec::with_capacity(batch.num_rows);
    for row_idx in 0..batch.num_rows {
        let data = encode_row(batch, row_idx, columns);
        tuples.push(Tuple::new(data, xmin));
    }
    tuples
}

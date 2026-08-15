//! DataBatch: columnar batch of rows for vectorized query execution.
//!
//! Provides the DataBatch type that holds typed columns with null bitmaps,
//! and conversion functions between the NSM (N-ary Storage Model) tuple
//! format used by the storage engine and the columnar batch format used
//! for query processing.

use zyron_catalog::{ColumnEntry, ColumnId};
use zyron_common::{Result, TypeId};
use zyron_planner::binder::BoundExpr;
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

    /// Creates a batch carrying a row count but no column data, for the
    /// `COUNT(*)`-style scan fast path where the consumer needs only the
    /// number of visible rows.
    pub fn with_row_count(num_rows: usize) -> Self {
        Self {
            columns: Vec::new(),
            num_rows,
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
    /// Logical type reported on the finished Column (e.g. TimestampTz even
    /// when the physical buffer is Int128 picoseconds).
    type_id: TypeId,
    /// Fractional-second precision carried onto the finished Column so a
    /// physical i128 is known to be a logical ps timestamp.
    fractional_digits: Option<u8>,
}

impl ColumnBuilder {
    pub fn new(type_id: TypeId, capacity: usize) -> Self {
        Self {
            data: ColumnData::with_capacity(type_id, capacity),
            nulls: NullBitmap::empty(),
            type_id,
            fractional_digits: None,
        }
    }

    /// Builder for a timestamp column: the physical buffer is sized for
    /// `physical_type` (Int128 when p>6) while the finished Column reports the
    /// `logical_type` and carries `fractional_digits`.
    pub fn new_ts(
        logical_type: TypeId,
        physical_type: TypeId,
        fractional_digits: Option<u8>,
        capacity: usize,
    ) -> Self {
        Self {
            data: ColumnData::with_capacity(physical_type, capacity),
            nulls: NullBitmap::empty(),
            type_id: logical_type,
            fractional_digits,
        }
    }

    pub fn push(&mut self, scalar: &ScalarValue) {
        let is_null = scalar.is_null();
        self.nulls.push(is_null);
        self.data.push_scalar(scalar);
    }

    /// Appends a value the caller is done with, moving a text or binary
    /// cell's allocation into the column rather than copying it.
    pub fn push_owned(&mut self, scalar: ScalarValue) {
        self.nulls.push(scalar.is_null());
        self.data.push_scalar_owned(scalar);
    }

    pub fn push_null(&mut self) {
        self.nulls.push(true);
        self.data.push_scalar(&ScalarValue::Null);
    }

    pub fn finish(self) -> Column {
        Column::with_nulls_ts(self.data, self.nulls, self.type_id, self.fractional_digits)
    }
}

/// Creates a vector of column builders for the given logical columns. A
/// TIMESTAMP(p) column with p>6 gets an i128 physical buffer while the
/// finished Column keeps its logical timestamp type and precision.
pub fn create_builders(columns: &[LogicalColumn], capacity: usize) -> Vec<ColumnBuilder> {
    columns
        .iter()
        .map(|col| {
            let phys = TypeId::timestamp_physical_type_id(col.type_id, col.fractional_digits);
            if phys != col.type_id || col.fractional_digits.is_some() {
                ColumnBuilder::new_ts(col.type_id, phys, col.fractional_digits, capacity)
            } else {
                ColumnBuilder::new(col.type_id, capacity)
            }
        })
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

/// Builds the per-column-ordinal lookup table used by
/// `decode_tuple_into_builders`: index `i` is `Some(b)` if the table column
/// at ordinal `i` maps to builder `b`, or `None` if the projection skips it.
///
/// The decoder iterates table columns in declaration order, so this map turns
/// the per-row "is column projected?" question into an O(1) array lookup.
pub fn build_column_to_builder_map(
    columns: &[ColumnEntry],
    output_column_ids: &[ColumnId],
) -> Vec<Option<u16>> {
    let mut map = vec![None; columns.len()];
    for (b, oid) in output_column_ids.iter().enumerate() {
        if let Some(i) = columns.iter().position(|c| c.id == *oid) {
            map[i] = Some(b as u16);
        }
    }
    map
}

/// Decodes one tuple's data bytes into column builders.
///
/// Tuple data layout (NSM, little-endian):
/// - Null bitmap: ceil(num_columns / 8) bytes, bit N set = column N is null
/// - Column values in ordinal order:
///   - Fixed-size types: inline at TypeId::fixed_size() bytes (zeroed if null)
///   - Variable-length types: 4-byte LE length prefix + data bytes (length=0, no data if null)
///
/// `column_to_builder` is the precomputed per-ordinal lookup produced by
/// `build_column_to_builder_map`. The decoder walks every table column to
/// keep the offset cursor aligned with the encoded row, but only touches
/// `builders[b]` when `column_to_builder[i] == Some(b)`. Pass the table's
/// full column list as `columns` even when the scan projects a subset.
pub fn decode_tuple_into_builders(
    data: &[u8],
    columns: &[ColumnEntry],
    column_to_builder: &[Option<u16>],
    builders: &mut [ColumnBuilder],
) {
    debug_assert_eq!(column_to_builder.len(), columns.len());
    let num_cols = columns.len();
    let null_bitmap_len = (num_cols + 7) / 8;
    let null_bitmap = &data[..null_bitmap_len];
    let mut offset = null_bitmap_len;

    for (i, col) in columns.iter().enumerate() {
        let is_null = (null_bitmap[i / 8] >> (i % 8)) & 1 == 1;
        let builder_idx = column_to_builder[i].map(|b| b as usize);
        // Physical type drives byte layout: a TIMESTAMP(p>6) column is stored
        // as 16-byte i128 picoseconds even though its logical type is a
        // timestamp.
        let phys_type = col.physical_type_id();

        if let Some(fixed_size) = phys_type.fixed_size() {
            if is_null {
                if let Some(b) = builder_idx {
                    builders[b].push_null();
                }
                offset += fixed_size;
            } else {
                let value_bytes = &data[offset..offset + fixed_size];
                if let Some(b) = builder_idx {
                    let scalar = decode_fixed_scalar(phys_type, value_bytes);
                    builders[b].push(&scalar);
                }
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
                if let Some(b) = builder_idx {
                    builders[b].push_null();
                }
                offset += len;
            } else {
                let value_bytes = &data[offset..offset + len];
                if let Some(b) = builder_idx {
                    let scalar = decode_varlen_scalar(col.type_id, value_bytes);
                    builders[b].push(&scalar);
                }
                offset += len;
            }
        }
    }
}

/// Evaluates a bound predicate against a set of encoded tuple rows and returns
/// a keep mask (true means the row satisfies the predicate). The rows are
/// decoded once into a columnar batch and the predicate is evaluated
/// vectorized, so the per-row cost is amortized across the whole set rather
/// than paid as a scalar evaluation per row.
///
/// `output_columns` is the logical schema the predicate was bound against (its
/// ColumnRefs resolve by position in this slice). `table_columns` is the full
/// table schema used to decode the NSM tuple bytes; only the columns present
/// in `output_columns` are materialized.
pub fn evaluate_row_filter(
    output_columns: &[LogicalColumn],
    table_columns: &[ColumnEntry],
    predicate: &BoundExpr,
    rows: &[&[u8]],
) -> Result<Vec<bool>> {
    if rows.is_empty() {
        return Ok(Vec::new());
    }
    let output_ids: Vec<ColumnId> = output_columns.iter().map(|c| c.column_id).collect();
    let column_to_builder = build_column_to_builder_map(table_columns, &output_ids);

    // Fail closed on a row whose bytes do not span the full schema: a truncated
    // or malformed change record is dropped (mask = false) rather than panicking
    // the decoder or evaluating a garbage predicate result. Only well-formed
    // rows are decoded into the batch; their predicate results are scattered
    // back to their original positions.
    let mut keep = vec![false; rows.len()];
    let mut decodable: Vec<usize> = Vec::with_capacity(rows.len());
    let mut builders = create_builders(output_columns, rows.len());
    for (i, row) in rows.iter().enumerate() {
        if tuple_decodes_within_bounds(row, table_columns) {
            decode_tuple_into_builders(row, table_columns, &column_to_builder, &mut builders);
            decodable.push(i);
        }
    }
    let batch = finalize_builders(builders);
    if batch.num_rows == 0 {
        return Ok(keep);
    }
    let mask_col = crate::expr::evaluate(predicate, &batch, output_columns, &[])?;
    let sub = crate::compute::column_to_mask(&mask_col);
    for (j, &i) in decodable.iter().enumerate() {
        keep[i] = sub.get(j).copied().unwrap_or(false);
    }
    Ok(keep)
}

/// Returns whether `data` holds a complete NSM tuple for `columns` without
/// reading past its end. MUST mirror the offset advancement in
/// `decode_tuple_into_builders` (null bitmap, fixed sizes by physical type,
/// 4-byte length prefix for variable-length columns) so a row that passes here
/// decodes without an out-of-bounds index.
fn tuple_decodes_within_bounds(data: &[u8], columns: &[ColumnEntry]) -> bool {
    let num_cols = columns.len();
    let null_bitmap_len = num_cols.div_ceil(8);
    if data.len() < null_bitmap_len {
        return false;
    }
    let mut offset = null_bitmap_len;
    for col in columns {
        let phys_type = col.physical_type_id();
        if let Some(fixed_size) = phys_type.fixed_size() {
            offset += fixed_size;
            if offset > data.len() {
                return false;
            }
        } else {
            if offset + 4 > data.len() {
                return false;
            }
            let len = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]) as usize;
            offset += 4 + len;
            if offset > data.len() {
                return false;
            }
        }
    }
    true
}

/// Decodes a fixed-size value from raw bytes into a ScalarValue.
pub fn decode_fixed_scalar(type_id: TypeId, bytes: &[u8]) -> ScalarValue {
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
        TypeId::Int128 | TypeId::Decimal | TypeId::Hlc => {
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
pub fn decode_varlen_scalar(type_id: TypeId, bytes: &[u8]) -> ScalarValue {
    match type_id {
        TypeId::Char | TypeId::Varchar | TypeId::Text | TypeId::Json | TypeId::Jsonb => {
            ScalarValue::Utf8(String::from_utf8_lossy(bytes).into_owned())
        }
        // Every other variable-length type (geometry, matrix, range, the
        // sketch family, and future additions) is byte-backed. A type list
        // here would silently turn unlisted values into NULL
        _ => ScalarValue::Binary(bytes.to_vec()),
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

        // Physical type drives byte layout (TIMESTAMP(p>6) = 16-byte i128 ps).
        let phys_type = col.physical_type_id();
        if let Some(fixed_size) = phys_type.fixed_size() {
            if is_null {
                buf.extend(std::iter::repeat(0u8).take(fixed_size));
            } else {
                encode_fixed_scalar(&mut buf, phys_type, &column.data.get_scalar(row_idx));
            }
        } else if is_null {
            buf.extend_from_slice(&0u32.to_le_bytes());
        } else {
            encode_varlen_scalar(&mut buf, &column.data.get_scalar(row_idx));
        }
    }

    buf
}

/// Encodes one scalar into the raw columnar value form: a fixed-width LE
/// value when `value_size > 0`, or the bare variable-length bytes (no length
/// prefix) when `value_size == 0`. This is the exact inverse of
/// `decode_fixed_scalar` / `decode_varlen_scalar`, so a value written here by
/// the columnar patch path round-trips through the columnar read path.
pub(crate) fn encode_scalar_value(
    type_id: TypeId,
    scalar: &ScalarValue,
    value_size: usize,
) -> Vec<u8> {
    if value_size == 0 {
        return match scalar {
            ScalarValue::Utf8(s) => s.as_bytes().to_vec(),
            ScalarValue::Binary(b) => b.clone(),
            ScalarValue::Null => Vec::new(),
            _ => Vec::new(),
        };
    }
    let mut buf = Vec::with_capacity(value_size);
    encode_fixed_scalar(&mut buf, type_id, scalar);
    if buf.len() < value_size {
        buf.resize(value_size, 0);
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
        (
            TypeId::Int128 | TypeId::Decimal | TypeId::UInt128 | TypeId::Hlc,
            ScalarValue::Int128(v),
        ) => buf.extend_from_slice(&v.to_le_bytes()),
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
fn encode_varlen_scalar(buf: &mut Vec<u8>, scalar: &ScalarValue) {
    // Encode by the scalar's representation, not by a type list. Every
    // variable-length column materializes as Utf8 or Binary, and a type
    // enumeration here silently wrote empty cells for unlisted types
    // (geometry, matrix, range, the sketch family), losing the payload on
    // every heap insert
    match scalar {
        ScalarValue::Utf8(s) => {
            let bytes = s.as_bytes();
            buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            buf.extend_from_slice(bytes);
        }
        ScalarValue::Binary(b) => {
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

#[cfg(test)]
mod row_filter_tests {
    use super::*;
    use zyron_catalog::TableId;
    use zyron_parser::ast::{BinaryOperator, LiteralValue};
    use zyron_planner::binder::{BoundExpr, ColumnRef};

    fn col(id: u16, name: &str, type_id: TypeId, ordinal: u16) -> ColumnEntry {
        ColumnEntry {
            id: ColumnId(id),
            table_id: TableId(1),
            name: name.to_string(),
            type_id,
            ordinal,
            nullable: false,
            default_expr: None,
            max_length: None,
            fractional_digits: None,
            tz_offset_secs: None,
            element_type: None,
        }
    }

    fn lcol(id: u16, name: &str, type_id: TypeId) -> LogicalColumn {
        LogicalColumn {
            table_idx: Some(0),
            column_id: ColumnId(id),
            name: name.to_string(),
            type_id,
            nullable: false,
            fractional_digits: None,
        }
    }

    // evaluate_row_filter must decode encoded rows and return a keep mask that
    // matches the predicate, so the publication stream drops rows the policy
    // hides.
    #[test]
    fn row_filter_masks_by_predicate() {
        let table_columns = vec![
            col(0, "id", TypeId::Int64, 0),
            col(1, "region", TypeId::Text, 1),
        ];

        // Three rows: us, eu, us.
        let batch = DataBatch::new(vec![
            Column::new(ColumnData::Int64(vec![1, 2, 3]), TypeId::Int64),
            Column::new(
                ColumnData::Utf8(vec!["us".into(), "eu".into(), "us".into()]),
                TypeId::Text,
            ),
        ]);
        let encoded: Vec<Vec<u8>> = (0..batch.num_rows)
            .map(|i| encode_row(&batch, i, &table_columns))
            .collect();
        let rows: Vec<&[u8]> = encoded.iter().map(|v| v.as_slice()).collect();

        let output_columns = vec![
            lcol(0, "id", TypeId::Int64),
            lcol(1, "region", TypeId::Text),
        ];

        // region = 'us'
        let predicate = BoundExpr::BinaryOp {
            left: Box::new(BoundExpr::ColumnRef(ColumnRef {
                table_idx: 0,
                column_id: ColumnId(1),
                type_id: TypeId::Text,
                nullable: false,
                fractional_digits: None,
            })),
            op: BinaryOperator::Eq,
            right: Box::new(BoundExpr::Literal {
                value: LiteralValue::String("us".into()),
                type_id: TypeId::Text,
            }),
            type_id: TypeId::Boolean,
        };

        let mask = evaluate_row_filter(&output_columns, &table_columns, &predicate, &rows).unwrap();
        assert_eq!(mask, vec![true, false, true]);

        // Empty input is a no-op.
        let empty = evaluate_row_filter(&output_columns, &table_columns, &predicate, &[]).unwrap();
        assert!(empty.is_empty());
    }

    // A truncated / malformed row image must be dropped (fail closed), never
    // panic the decoder, and must not affect the verdict on well-formed rows.
    #[test]
    fn row_filter_drops_truncated_rows() {
        let table_columns = vec![
            col(0, "id", TypeId::Int64, 0),
            col(1, "region", TypeId::Text, 1),
        ];
        let batch = DataBatch::new(vec![
            Column::new(ColumnData::Int64(vec![1]), TypeId::Int64),
            Column::new(ColumnData::Utf8(vec!["us".into()]), TypeId::Text),
        ]);
        let good = encode_row(&batch, 0, &table_columns);
        let truncated: Vec<u8> = vec![0u8]; // far too short for the schema
        let empty: Vec<u8> = Vec::new();
        let rows: Vec<&[u8]> = vec![good.as_slice(), truncated.as_slice(), empty.as_slice()];

        let output_columns = vec![
            lcol(0, "id", TypeId::Int64),
            lcol(1, "region", TypeId::Text),
        ];
        let predicate = BoundExpr::BinaryOp {
            left: Box::new(BoundExpr::ColumnRef(ColumnRef {
                table_idx: 0,
                column_id: ColumnId(1),
                type_id: TypeId::Text,
                nullable: false,
                fractional_digits: None,
            })),
            op: BinaryOperator::Eq,
            right: Box::new(BoundExpr::Literal {
                value: LiteralValue::String("us".into()),
                type_id: TypeId::Text,
            }),
            type_id: TypeId::Boolean,
        };

        let mask = evaluate_row_filter(&output_columns, &table_columns, &predicate, &rows).unwrap();
        // Good row passes; the two malformed rows are dropped.
        assert_eq!(mask, vec![true, false, false]);
    }
}

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
use crate::epoch_decode::EpochDecoder;

/// Number of rows per execution batch.
pub const BATCH_SIZE: usize = 1024;

// ---------------------------------------------------------------------------
// DataBatch
// ---------------------------------------------------------------------------

/// One JSON path a scan already pulled out of a variant column, held
/// alongside the batch it belongs to.
///
/// A columnar segment can store a promoted path as a column of its own, in
/// which case the scan reads the values instead of walking every document.
/// The values it carries are what the walk would have produced, so an
/// expression served from here and the same expression served from the
/// document give the same answer
#[derive(Debug, Clone)]
pub struct ResolvedPath {
    /// Which table in the batch's schema the variant column belongs to,
    /// spelled the way a `ColumnRef` spells it
    pub table_idx: usize,
    /// The variant column the path was read out of
    pub column_id: u16,
    /// Dotted path, as `variant_extract` spells it
    pub path: String,
    /// One value per row of the batch, null where the document had no
    /// scalar at the path
    pub values: Column,
}

/// A columnar batch of rows. Each column holds a typed vector of values
/// with a null bitmap. All columns have the same number of rows.
#[derive(Debug, Clone)]
pub struct DataBatch {
    pub columns: Vec<Column>,
    pub num_rows: usize,
    /// Variant paths already extracted for these rows, in the same row order
    /// as `columns`. Empty on every batch not built by a scan reading a
    /// segment that stores the path, and an empty list costs an expression
    /// only the document walk it would have done anyway
    pub resolved: Vec<ResolvedPath>,
}

impl DataBatch {
    /// Approximate heap bytes this batch holds, column payloads plus one
    /// bit per row per null bitmap. Used by the query memory budget.
    pub fn approx_bytes(&self) -> u64 {
        self.columns
            .iter()
            .map(|c| &c.data)
            .chain(self.resolved.iter().map(|r| &r.values.data))
            .map(|d| d.approx_bytes() + (self.num_rows as u64).div_ceil(8))
            .sum()
    }

    /// Creates a batch from pre-built columns. All columns must have the same length.
    pub fn new(columns: Vec<Column>) -> Self {
        let num_rows = columns.first().map_or(0, |c| c.len());
        debug_assert!(columns.iter().all(|c| c.len() == num_rows));
        Self {
            columns,
            num_rows,
            resolved: Vec::new(),
        }
    }

    /// Creates an empty batch with no rows and no columns.
    pub fn empty() -> Self {
        Self {
            columns: Vec::new(),
            num_rows: 0,
            resolved: Vec::new(),
        }
    }

    /// Creates a batch carrying a row count but no column data, for the
    /// `COUNT(*)`-style scan fast path where the consumer needs only the
    /// number of visible rows.
    pub fn with_row_count(num_rows: usize) -> Self {
        Self {
            columns: Vec::new(),
            num_rows,
            resolved: Vec::new(),
        }
    }

    /// Attaches variant paths a scan resolved for these rows.
    ///
    /// Every entry holds one value per row of the batch, in the same order,
    /// or the substitution would answer with another row's value
    pub fn with_resolved(mut self, resolved: Vec<ResolvedPath>) -> Self {
        debug_assert!(resolved.iter().all(|r| r.values.len() == self.num_rows));
        self.resolved = resolved;
        self
    }

    /// The values a scan already extracted for one variant column and path,
    /// or None when nothing resolved it and the caller has to read the
    /// documents.
    ///
    /// An entry whose length is not this batch's row count is not offered.
    /// It would answer with another row's value, and reading the documents is
    /// always available and always right
    pub fn resolved_path(&self, table_idx: usize, column_id: u16, path: &str) -> Option<&Column> {
        if self.num_rows == 0 {
            return None;
        }
        self.resolved
            .iter()
            .find(|r| r.column_id == column_id && r.table_idx == table_idx && r.path == path)
            .map(|r| &r.values)
            .filter(|values| values.len() == self.num_rows)
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
    ///
    /// Resolved paths take the same selection as the columns, which is what
    /// keeps a value lined up with the row it came from
    pub fn filter(&self, mask: &[bool]) -> Self {
        let columns: Vec<Column> = self.columns.iter().map(|c| c.filter(mask)).collect();
        let num_rows = columns.first().map_or(0, |c| c.len());
        Self {
            columns,
            num_rows,
            resolved: self.map_resolved(|c| c.filter(mask)),
        }
    }

    /// Reorders rows by indices.
    pub fn take(&self, indices: &[u32]) -> Self {
        let columns: Vec<Column> = self.columns.iter().map(|c| c.take(indices)).collect();
        let num_rows = indices.len();
        Self {
            columns,
            num_rows,
            resolved: self.map_resolved(|c| c.take(indices)),
        }
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
            resolved: self.map_resolved(|c| c.slice(offset, actual_len)),
        }
    }

    /// Builds an empty batch shaped like this one, to be filled one row at a
    /// time by [`DataBatch::load_row`].
    ///
    /// An expression that has to be evaluated per row needs a batch to read,
    /// and slicing a fresh one row batch per row allocated a buffer and a null
    /// bitmap for every column of every row. One view built here and refilled
    /// in place allocates on the first row and reuses those buffers for the
    /// rest of the batch.
    pub fn row_view(&self) -> Self {
        // A zero length slice carries the column's physical variant, which the
        // logical type alone does not name: a p>6 timestamp is i128 under a
        // Timestamp type id, and a view built from the type id would be i64
        Self {
            columns: self
                .columns
                .iter()
                .map(|c| {
                    Column::with_nulls_ts(
                        c.data.slice(0, 0),
                        NullBitmap::empty(),
                        c.type_id,
                        c.fractional_digits,
                    )
                })
                .collect(),
            num_rows: 0,
            resolved: self
                .resolved
                .iter()
                .map(|r| ResolvedPath {
                    table_idx: r.table_idx,
                    column_id: r.column_id,
                    path: r.path.clone(),
                    values: Column::with_nulls_ts(
                        r.values.data.slice(0, 0),
                        NullBitmap::empty(),
                        r.values.type_id,
                        r.values.fractional_digits,
                    ),
                })
                .collect(),
        }
    }

    /// Replaces a view's contents with one row of this batch.
    ///
    /// The view has to have come from [`DataBatch::row_view`] on this batch, so
    /// that its columns line up with these in both count and type.
    pub fn load_row(&self, row: usize, view: &mut Self) {
        self.load_row_at(row, view, 0);
        for (dst, src) in view.resolved.iter_mut().zip(self.resolved.iter()) {
            dst.values.data.truncate(0);
            dst.values.data.push_from(&src.values.data, row);
            dst.values.nulls.clear();
            dst.values.nulls.push_from(&src.values.nulls, row);
        }
    }

    /// Writes one row of this batch into a view's columns starting at `at`.
    ///
    /// A joined row is assembled from two batches, so each side fills its own
    /// span of one view rather than each producing a batch of its own for the
    /// two to be concatenated. Only the columns are written, so a view filled
    /// this way carries no resolved variant paths, which is what a freshly
    /// concatenated joined batch carried as well.
    pub fn load_row_at(&self, row: usize, view: &mut Self, at: usize) {
        for (offset, src) in self.columns.iter().enumerate() {
            let Some(dst) = view.columns.get_mut(at + offset) else {
                break;
            };
            dst.data.truncate(0);
            dst.data.push_from(&src.data, row);
            dst.nulls.clear();
            dst.nulls.push_from(&src.nulls, row);
        }
        view.num_rows = 1;
    }

    /// Applies one row transform to every resolved path, skipping the work
    /// for the batches that carry none
    fn map_resolved(&self, f: impl Fn(&Column) -> Column) -> Vec<ResolvedPath> {
        if self.resolved.is_empty() {
            return Vec::new();
        }
        self.resolved
            .iter()
            .map(|r| ResolvedPath {
                table_idx: r.table_idx,
                column_id: r.column_id,
                path: r.path.clone(),
                values: f(&r.values),
            })
            .collect()
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

    /// Appends one present fixed-width cell straight into the typed buffer.
    ///
    /// The per-cell counterpart of `extend_fixed`, for a decoder that walks
    /// a row and meets one cell of each column at a time. Building a
    /// scalar to push it costs a construction and two dispatches per cell,
    /// this is one dispatch and a store. Returns false, having pushed
    /// nothing, for a pairing this does not carry, and the caller pushes
    /// the scalar instead. Every arm lands the value `decode_fixed_scalar`
    /// would
    #[inline]
    pub fn push_fixed(&mut self, physical: TypeId, bytes: &[u8]) -> bool {
        macro_rules! one {
            ($buf:expr, $width:expr, $decode:expr) => {{
                let Some(cell) = bytes.get(..$width) else {
                    return false;
                };
                let decode: fn(&[u8]) -> _ = $decode;
                $buf.push(decode(cell));
                self.nulls.push(false);
                true
            }};
        }

        match (&mut self.data, physical) {
            (ColumnData::Boolean(v), TypeId::Boolean) => one!(v, 1, |b| b[0] != 0),
            (ColumnData::Int8(v), TypeId::Int8) => one!(v, 1, |b| i8::from_le_bytes([b[0]])),
            (ColumnData::Int16(v), TypeId::Int16) => {
                one!(v, 2, |b| i16::from_le_bytes(b.try_into().unwrap()))
            }
            (ColumnData::Int32(v), TypeId::Int32 | TypeId::Date) => {
                one!(v, 4, |b| i32::from_le_bytes(b.try_into().unwrap()))
            }
            (
                ColumnData::Int64(v),
                TypeId::Int64 | TypeId::Time | TypeId::Timestamp | TypeId::TimestampTz,
            ) => {
                one!(v, 8, |b| i64::from_le_bytes(b.try_into().unwrap()))
            }
            (
                ColumnData::Int128(v),
                TypeId::Int128 | TypeId::Decimal | TypeId::Hlc | TypeId::UInt128,
            ) => {
                one!(v, 16, |b| i128::from_le_bytes(b.try_into().unwrap()))
            }
            (ColumnData::UInt8(v), TypeId::UInt8) => one!(v, 1, |b| b[0]),
            (ColumnData::UInt16(v), TypeId::UInt16) => {
                one!(v, 2, |b| u16::from_le_bytes(b.try_into().unwrap()))
            }
            (ColumnData::UInt32(v), TypeId::UInt32) => {
                one!(v, 4, |b| u32::from_le_bytes(b.try_into().unwrap()))
            }
            (ColumnData::UInt64(v), TypeId::UInt64) => {
                one!(v, 8, |b| u64::from_le_bytes(b.try_into().unwrap()))
            }
            (ColumnData::Float32(v), TypeId::Float32) => {
                one!(v, 4, |b| f32::from_le_bytes(b.try_into().unwrap()))
            }
            (ColumnData::Float64(v), TypeId::Float64) => {
                one!(v, 8, |b| f64::from_le_bytes(b.try_into().unwrap()))
            }
            _ => false,
        }
    }

    /// Appends a run of fixed-width cells straight into the typed buffer,
    /// `None` meaning null.
    ///
    /// The per-value path decides two things at runtime for every single
    /// value: which `ScalarValue` to build from the bytes, and which buffer
    /// that scalar belongs in. Both answers are the same for every value in a
    /// column, so this settles them once and leaves a load and a store in the
    /// loop. Scanning a decoded column is where that difference is the whole
    /// cost, since the bytes are already contiguous and typed.
    ///
    /// Returns false, having consumed nothing, when the buffer and the type
    /// are a pairing this does not carry. The caller falls back to pushing
    /// scalars, which is what makes an unlisted type slow rather than wrong.
    ///
    /// Every arm must land the same value the per-value path would:
    /// `decode_fixed_scalar` for a present cell, and for an absent one the
    /// buffer's own filler with the null flag set.
    pub fn extend_fixed<'a>(
        &mut self,
        physical: TypeId,
        cells: impl Iterator<Item = Option<&'a [u8]>>,
    ) -> bool {
        macro_rules! run {
            ($buf:expr, $width:expr, $decode:expr, $filler:expr) => {{
                let buf = $buf;
                for cell in cells {
                    match cell {
                        Some(bytes) => {
                            self.nulls.push(false);
                            let decode: fn(&[u8]) -> _ = $decode;
                            buf.push(decode(&bytes[..$width]));
                        }
                        None => {
                            self.nulls.push(true);
                            buf.push($filler);
                        }
                    }
                }
                true
            }};
        }

        match (&mut self.data, physical) {
            (ColumnData::Boolean(v), TypeId::Boolean) => run!(v, 1, |b| b[0] != 0, false),
            (ColumnData::Int8(v), TypeId::Int8) => {
                run!(v, 1, |b| i8::from_le_bytes([b[0]]), 0)
            }
            (ColumnData::Int16(v), TypeId::Int16) => {
                run!(v, 2, |b| i16::from_le_bytes(b.try_into().unwrap()), 0)
            }
            (ColumnData::Int32(v), TypeId::Int32 | TypeId::Date) => {
                run!(v, 4, |b| i32::from_le_bytes(b.try_into().unwrap()), 0)
            }
            (
                ColumnData::Int64(v),
                TypeId::Int64 | TypeId::Time | TypeId::Timestamp | TypeId::TimestampTz,
            ) => {
                run!(v, 8, |b| i64::from_le_bytes(b.try_into().unwrap()), 0)
            }
            (
                ColumnData::Int128(v),
                TypeId::Int128 | TypeId::Decimal | TypeId::Hlc | TypeId::UInt128,
            ) => {
                run!(v, 16, |b| i128::from_le_bytes(b.try_into().unwrap()), 0)
            }
            (ColumnData::UInt8(v), TypeId::UInt8) => run!(v, 1, |b| b[0], 0),
            (ColumnData::UInt16(v), TypeId::UInt16) => {
                run!(v, 2, |b| u16::from_le_bytes(b.try_into().unwrap()), 0)
            }
            (ColumnData::UInt32(v), TypeId::UInt32) => {
                run!(v, 4, |b| u32::from_le_bytes(b.try_into().unwrap()), 0)
            }
            (ColumnData::UInt64(v), TypeId::UInt64) => {
                run!(v, 8, |b| u64::from_le_bytes(b.try_into().unwrap()), 0)
            }
            (ColumnData::Float32(v), TypeId::Float32) => {
                run!(v, 4, |b| f32::from_le_bytes(b.try_into().unwrap()), 0.0)
            }
            (ColumnData::Float64(v), TypeId::Float64) => {
                run!(v, 8, |b| f64::from_le_bytes(b.try_into().unwrap()), 0.0)
            }
            _ => false,
        }
    }

    /// Appends `rows` cells laid out end to end, none of them null.
    ///
    /// The gather above asks two questions per value: whether the ordinal
    /// falls inside the decoded range, and whether the cell is present. A
    /// run has answered both for all of them before the loop starts, so
    /// what is left is a decode over a slice and one bulk step for the
    /// null bitmap. This is the shape every scan that filtered nothing
    /// hands over, which is every full scan and every aggregate over one
    pub fn extend_fixed_run(&mut self, physical: TypeId, data: &[u8], rows: usize) -> bool {
        macro_rules! run {
            ($buf:expr, $width:expr, $decode:expr) => {{
                let buf = $buf;
                if data.len() < rows * $width {
                    return false;
                }
                buf.reserve(rows);
                for cell in data.chunks_exact($width).take(rows) {
                    let decode: fn(&[u8]) -> _ = $decode;
                    buf.push(decode(cell));
                }
                self.nulls.extend_valid(rows);
                true
            }};
        }

        match (&mut self.data, physical) {
            (ColumnData::Boolean(v), TypeId::Boolean) => run!(v, 1, |b| b[0] != 0),
            (ColumnData::Int8(v), TypeId::Int8) => run!(v, 1, |b| i8::from_le_bytes([b[0]])),
            (ColumnData::Int16(v), TypeId::Int16) => {
                run!(v, 2, |b| i16::from_le_bytes(b.try_into().unwrap()))
            }
            (ColumnData::Int32(v), TypeId::Int32 | TypeId::Date) => {
                run!(v, 4, |b| i32::from_le_bytes(b.try_into().unwrap()))
            }
            (
                ColumnData::Int64(v),
                TypeId::Int64 | TypeId::Time | TypeId::Timestamp | TypeId::TimestampTz,
            ) => {
                run!(v, 8, |b| i64::from_le_bytes(b.try_into().unwrap()))
            }
            (
                ColumnData::Int128(v),
                TypeId::Int128 | TypeId::Decimal | TypeId::Hlc | TypeId::UInt128,
            ) => {
                run!(v, 16, |b| i128::from_le_bytes(b.try_into().unwrap()))
            }
            (ColumnData::UInt8(v), TypeId::UInt8) => run!(v, 1, |b| b[0]),
            (ColumnData::UInt16(v), TypeId::UInt16) => {
                run!(v, 2, |b| u16::from_le_bytes(b.try_into().unwrap()))
            }
            (ColumnData::UInt32(v), TypeId::UInt32) => {
                run!(v, 4, |b| u32::from_le_bytes(b.try_into().unwrap()))
            }
            (ColumnData::UInt64(v), TypeId::UInt64) => {
                run!(v, 8, |b| u64::from_le_bytes(b.try_into().unwrap()))
            }
            (ColumnData::Float32(v), TypeId::Float32) => {
                run!(v, 4, |b| f32::from_le_bytes(b.try_into().unwrap()))
            }
            (ColumnData::Float64(v), TypeId::Float64) => {
                run!(v, 8, |b| f64::from_le_bytes(b.try_into().unwrap()))
            }
            _ => false,
        }
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
/// - Null bitmap: ceil(columns in the epoch's layout / 8) bytes, bit N set =
///   the layout's column N is null
/// - Column values in the epoch's positional order:
///   - Fixed-size types: inline at TypeId::fixed_size() bytes (zeroed if null)
///   - Variable-length types: 4-byte LE length prefix + data bytes (length=0, no data if null)
///
/// `epoch` is the layout the row was written under, read from its slot. The
/// decoder resolves it against the plans `decoder` holds, so a row older than
/// a column fills that column from its recorded absent value and a row that
/// still carries a dropped column walks past it. `at` names the row for the
/// corruption report an epoch with no recorded layout produces.
pub fn decode_tuple_into_builders(
    data: &[u8],
    decoder: &EpochDecoder,
    epoch: u16,
    at: Option<zyron_common::RowLocator>,
    builders: &mut [ColumnBuilder],
) -> Result<()> {
    decoder.decode(epoch, data, at, builders)
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
    table: &zyron_catalog::TableEntry,
    epoch: u16,
    predicate: &BoundExpr,
    rows: &[&[u8]],
) -> Result<Vec<bool>> {
    if rows.is_empty() {
        return Ok(Vec::new());
    }
    let output_ids: Vec<ColumnId> = output_columns.iter().map(|c| c.column_id).collect();
    let decoder = EpochDecoder::new(table, &output_ids);

    // Fail closed on a row whose bytes do not span the epoch's layout: a
    // truncated or malformed change record is dropped (mask = false) rather
    // than panicking the decoder or evaluating a garbage predicate result.
    // Only well-formed rows are decoded into the batch; their predicate
    // results are scattered back to their original positions.
    let mut keep = vec![false; rows.len()];
    let mut decodable: Vec<usize> = Vec::with_capacity(rows.len());
    let mut builders = create_builders(output_columns, rows.len());
    for (i, row) in rows.iter().enumerate() {
        if decoder.try_decode(epoch, row, &mut builders) {
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
        TypeId::Char
        | TypeId::Varchar
        | TypeId::Text
        | TypeId::Json
        | TypeId::Jsonb
        | TypeId::Variant
        | TypeId::Ltree => ScalarValue::Utf8(String::from_utf8_lossy(bytes).into_owned()),
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
    let null_bitmap_len = num_cols.div_ceil(8);
    let mut buf = Vec::with_capacity(null_bitmap_len + num_cols * 8);
    encode_row_into(&mut buf, batch, row_idx, columns);
    buf
}

/// Encodes one row onto the end of an existing buffer.
///
/// The whole-batch callers go through this: a replication capture encodes a
/// thousand rows into the buffer that becomes the log record, and a
/// whole-image match scan encodes every row it inspects into one reused
/// scratch buffer. A vector per row on those paths is an allocation and a
/// copy that buys nothing
pub fn encode_row_into(
    buf: &mut Vec<u8>,
    batch: &DataBatch,
    row_idx: usize,
    columns: &[ColumnEntry],
) {
    let num_cols = columns.len();
    let null_bitmap_len = num_cols.div_ceil(8);
    let base = buf.len();
    buf.resize(base + null_bitmap_len, 0u8);

    for (i, col) in columns.iter().enumerate() {
        let column = &batch.columns[i];
        let is_null = column.is_null(row_idx);

        if is_null {
            buf[base + i / 8] |= 1 << (i % 8);
        }

        // Physical type drives byte layout (TIMESTAMP(p>6) = 16-byte i128 ps).
        let phys_type = col.physical_type_id();
        if let Some(fixed_size) = phys_type.fixed_size() {
            if is_null {
                buf.extend(std::iter::repeat_n(0u8, fixed_size));
            } else {
                encode_fixed_scalar(buf, phys_type, &column.data.get_scalar(row_idx));
            }
        } else if is_null {
            buf.extend_from_slice(&0u32.to_le_bytes());
        } else {
            // Varlen payloads are borrowed straight from the column, the
            // length prefix and bytes land in buf without a scalar copy
            match &column.data {
                ColumnData::Utf8(v) => encode_varlen_bytes(buf, v[row_idx].as_bytes()),
                ColumnData::Binary(v) => encode_varlen_bytes(buf, &v[row_idx]),
                other => encode_varlen_scalar(buf, &other.get_scalar(row_idx)),
            }
        }
    }
}

/// Writes one length-prefixed variable-length payload, the borrowed
/// counterpart of `encode_varlen_scalar`
#[inline]
fn encode_varlen_bytes(buf: &mut Vec<u8>, bytes: &[u8]) {
    buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    buf.extend_from_slice(bytes);
}

/// Encodes one scalar into the raw columnar value form: a fixed-width LE
/// value when `value_size > 0`, or the bare variable-length bytes (no length
/// prefix) when `value_size == 0`. This is the exact inverse of
/// `decode_fixed_scalar` / `decode_varlen_scalar`, so a value written here by
/// the columnar patch path round-trips through the columnar read path.
pub fn encode_scalar_value(type_id: TypeId, scalar: &ScalarValue, value_size: usize) -> Vec<u8> {
    let mut buf = Vec::with_capacity(value_size);
    encode_scalar_value_into(&mut buf, type_id, scalar, value_size);
    buf
}

/// Encodes one scalar into the end of `buf` rather than into a buffer of its
/// own.
///
/// A caller filling a column reuses one buffer across every cell, which is
/// the difference between one allocation per cell and none: a ten thousand
/// row batch of two columns spent 415us building twenty thousand of them
#[inline]
pub(crate) fn encode_scalar_value_into(
    buf: &mut Vec<u8>,
    type_id: TypeId,
    scalar: &ScalarValue,
    value_size: usize,
) {
    if value_size == 0 {
        match scalar {
            ScalarValue::Utf8(s) => buf.extend_from_slice(s.as_bytes()),
            ScalarValue::Binary(b) => buf.extend_from_slice(b),
            _ => {}
        }
        return;
    }
    let start = buf.len();
    encode_fixed_scalar(buf, type_id, scalar);
    if buf.len() - start < value_size {
        buf.resize(start + value_size, 0);
    }
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
        // A TIMESTAMP(p>6) column carries i128 picosecond values under its
        // logical timestamp type, stored at the 16-byte physical width
        (TypeId::Timestamp | TypeId::TimestampTz, ScalarValue::Int128(v)) => {
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
                buf.extend(std::iter::repeat_n(0u8, size));
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
///
/// Every tuple is stamped with `schema_epoch`, which is the layout `columns`
/// describes. A row written without its epoch could not be read back: the
/// decoder would have no way to know how many columns its null bitmap covers.
pub fn batch_to_tuples(
    batch: &DataBatch,
    columns: &[ColumnEntry],
    xmin: u64,
    schema_epoch: u16,
) -> Vec<Tuple> {
    let mut tuples = Vec::with_capacity(batch.num_rows);
    for row_idx in 0..batch.num_rows {
        let data = encode_row(batch, row_idx, columns);
        tuples.push(Tuple::with_epoch(data, xmin, schema_epoch));
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
            attrs: Default::default(),
            absent_value: None,
            dropped: false,
        }
    }

    /// A table entry over `columns`, sealed at its first epoch, which is what
    /// the filter reads a row image through.
    fn table_of(columns: Vec<ColumnEntry>) -> zyron_catalog::TableEntry {
        let mut entry = zyron_catalog::TableEntry {
            id: TableId(1),
            schema_id: zyron_catalog::SchemaId(1),
            name: "t".to_string(),
            heap_file_id: 1,
            fsm_file_id: 2,
            columns,
            constraints: Vec::new(),
            created_at: 0,
            versioning_enabled: false,
            scd_type: None,
            system_versioned: false,
            history_table_id: None,
            cdf_enabled: false,
            cdf_retention_days: 0,
            lifecycle: Default::default(),
            columnar: Default::default(),
            dropped_at: None,
            expectations: Vec::new(),
            time_travel_retention_secs: 0,
            lake: Default::default(),
            cluster: Default::default(),
            foreign: Default::default(),
            schema_epoch: 0,
            schema_epochs: Vec::new(),
            pre_stamp_columns: Vec::new(),
        };
        entry.seal_initial_epoch();
        entry
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

        let table = table_of(table_columns.clone());
        let mask = evaluate_row_filter(
            &output_columns,
            &table,
            table.schema_epoch,
            &predicate,
            &rows,
        )
        .unwrap();
        assert_eq!(mask, vec![true, false, true]);

        // Empty input is a no-op.
        let empty =
            evaluate_row_filter(&output_columns, &table, table.schema_epoch, &predicate, &[])
                .unwrap();
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

        let table = table_of(table_columns.clone());
        let mask = evaluate_row_filter(
            &output_columns,
            &table,
            table.schema_epoch,
            &predicate,
            &rows,
        )
        .unwrap();
        // Good row passes; the two malformed rows are dropped.
        assert_eq!(mask, vec![true, false, false]);
    }
}

#[cfg(test)]
mod extend_fixed_tests {
    use super::*;

    /// Every fixed-width physical type a lake column can decode to.
    const FIXED_TYPES: &[TypeId] = &[
        TypeId::Boolean,
        TypeId::Int8,
        TypeId::Int16,
        TypeId::Int32,
        TypeId::Date,
        TypeId::Int64,
        TypeId::Time,
        TypeId::Timestamp,
        TypeId::TimestampTz,
        TypeId::Int128,
        TypeId::Decimal,
        TypeId::Hlc,
        TypeId::UInt8,
        TypeId::UInt16,
        TypeId::UInt32,
        TypeId::UInt64,
        TypeId::UInt128,
        TypeId::Float32,
        TypeId::Float64,
        TypeId::Uuid,
        TypeId::Interval,
    ];

    /// The bulk append exists only to be faster, so it has to land exactly
    /// what the per-value path lands, values and null flags alike. This runs
    /// both over the same cells and compares the finished column.
    ///
    /// A type the bulk path declines is still covered: it reports false and
    /// the comparison then holds trivially, which is the fallback the caller
    /// depends on.
    #[test]
    fn extend_fixed_matches_the_per_value_path() {
        for &physical in FIXED_TYPES {
            let Some(width) = physical.fixed_size() else {
                continue;
            };
            if width == 0 {
                continue;
            }

            // A mix of bit patterns, including all-zero, all-ones and a
            // high-bit-set value so sign handling shows up
            let patterns: Vec<Option<Vec<u8>>> = (0..40u8)
                .map(|i| {
                    if i % 7 == 3 {
                        None
                    } else {
                        Some(
                            (0..width)
                                .map(|b| i.wrapping_mul(31).wrapping_add(b as u8))
                                .collect(),
                        )
                    }
                })
                .chain([
                    Some(vec![0u8; width]),
                    Some(vec![0xFFu8; width]),
                    None,
                    Some({
                        let mut v = vec![0u8; width];
                        v[width - 1] = 0x80;
                        v
                    }),
                ])
                .collect();

            let mut bulk = ColumnBuilder::new(physical, patterns.len());
            let took = bulk.extend_fixed(
                physical,
                patterns.iter().map(|p| p.as_ref().map(|v| v.as_slice())),
            );

            let mut per_value = ColumnBuilder::new(physical, patterns.len());
            for p in &patterns {
                let sv = match p {
                    None => ScalarValue::Null,
                    Some(bytes) => decode_fixed_scalar(physical, bytes),
                };
                per_value.push_owned(sv);
            }

            if !took {
                // Declined, so the caller uses the per-value path and there
                // is nothing to compare
                continue;
            }

            let a = bulk.finish();
            let b = per_value.finish();
            assert_eq!(
                format!("{:?}", a.data),
                format!("{:?}", b.data),
                "{physical:?} values differ between the bulk and per-value paths"
            );
            assert_eq!(
                format!("{:?}", a.nulls),
                format!("{:?}", b.nulls),
                "{physical:?} null flags differ between the bulk and per-value paths"
            );
        }
    }

    /// The bulk path must not consume anything when it declines, or the
    /// caller's fallback would start partway through the column.
    #[test]
    fn a_declined_type_consumes_no_cells() {
        let cells = [Some([1u8, 2, 3, 4].as_slice()), None];
        let mut builder = ColumnBuilder::new(TypeId::Int32, 2);
        // Int32 buffer offered a type it does not carry
        assert!(!builder.extend_fixed(TypeId::Float64, cells.iter().copied()));
        let finished = builder.finish();
        assert_eq!(finished.nulls.len(), 0, "declining still pushed nulls");
    }
}

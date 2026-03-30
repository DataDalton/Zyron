//! Lightweight columnar types for streaming micro-batch processing.
//!
//! Provides StreamBatch, StreamColumn, StreamColumnData, NullBitmap,
//! and ScalarValue types that mirror the shapes from zyron-executor
//! but live locally to avoid pulling in the full executor dependency
//! chain (executor -> planner -> catalog -> storage -> buffer -> wal).
//! Conversion to/from DataBatch happens at integration boundaries.

use std::fmt;

// ---------------------------------------------------------------------------
// NullBitmap: packed u64 words, 1 bit per value (bit set = null)
// ---------------------------------------------------------------------------

/// Packed null bitmap using u64 words. Bit set at position i means row i is null.
#[derive(Debug, Clone)]
pub struct NullBitmap {
    words: Vec<u64>,
    len: usize,
}

impl NullBitmap {
    /// Creates a bitmap with all values valid (no nulls).
    pub fn new_valid(len: usize) -> Self {
        let word_count = (len + 63) / 64;
        Self {
            words: vec![0; word_count],
            len,
        }
    }

    /// Creates an empty bitmap.
    pub fn empty() -> Self {
        Self {
            words: Vec::new(),
            len: 0,
        }
    }

    #[inline(always)]
    pub fn is_null(&self, idx: usize) -> bool {
        debug_assert!(idx < self.len);
        (self.words[idx / 64] >> (idx % 64)) & 1 == 1
    }

    #[inline(always)]
    pub fn is_valid(&self, idx: usize) -> bool {
        !self.is_null(idx)
    }

    #[inline]
    pub fn set_null(&mut self, idx: usize) {
        debug_assert!(idx < self.len);
        self.words[idx / 64] |= 1u64 << (idx % 64);
    }

    #[inline]
    pub fn set_valid(&mut self, idx: usize) {
        debug_assert!(idx < self.len);
        self.words[idx / 64] &= !(1u64 << (idx % 64));
    }

    pub fn push(&mut self, is_null: bool) {
        let idx = self.len;
        self.len += 1;
        if idx / 64 >= self.words.len() {
            self.words.push(0);
        }
        if is_null {
            self.words[idx / 64] |= 1u64 << (idx % 64);
        }
    }

    /// Returns true if any value is null.
    pub fn has_nulls(&self) -> bool {
        self.words.iter().any(|&w| w != 0)
    }

    /// Count of null values via popcount.
    pub fn null_count(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Filter rows by boolean mask. Pre-allocates the result bitmap.
    pub fn filter(&self, mask: &[bool]) -> Self {
        debug_assert_eq!(mask.len(), self.len);
        let count = mask.iter().filter(|&&b| b).count();
        let mut result = Self::new_valid(count);
        let mut dst = 0;
        for (i, &keep) in mask.iter().enumerate() {
            if keep {
                if self.is_null(i) {
                    result.set_null(dst);
                }
                dst += 1;
            }
        }
        result
    }

    /// Gather rows by indices. Pre-allocates the result bitmap.
    pub fn take(&self, indices: &[u32]) -> Self {
        let mut result = Self::new_valid(indices.len());
        for (dst, &idx) in indices.iter().enumerate() {
            if self.is_null(idx as usize) {
                result.set_null(dst);
            }
        }
        result
    }

    /// Extract a contiguous sub-range.
    pub fn slice(&self, offset: usize, len: usize) -> Self {
        let actual_len = len.min(self.len.saturating_sub(offset));
        let mut result = Self::new_valid(actual_len);
        for i in 0..actual_len {
            if self.is_null(offset + i) {
                result.set_null(i);
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// ScalarValue: lightweight runtime-typed single value
// ---------------------------------------------------------------------------

/// Lightweight scalar value for accumulator finalization and fallback paths.
/// Not used on hot paths (accumulators use typed column access).
#[derive(Debug, Clone, PartialEq)]
pub enum ScalarValue {
    Null,
    Boolean(bool),
    Int8(i8),
    Int16(i16),
    Int32(i32),
    Int64(i64),
    Int128(i128),
    UInt8(u8),
    UInt16(u16),
    UInt32(u32),
    UInt64(u64),
    Float32(f32),
    Float64(f64),
    Utf8(String),
    Binary(Vec<u8>),
}

impl ScalarValue {
    pub fn is_null(&self) -> bool {
        matches!(self, ScalarValue::Null)
    }

    /// Convert numeric scalars to f64 for aggregate operations.
    pub fn to_f64(&self) -> Option<f64> {
        match self {
            ScalarValue::Int8(v) => Some(*v as f64),
            ScalarValue::Int16(v) => Some(*v as f64),
            ScalarValue::Int32(v) => Some(*v as f64),
            ScalarValue::Int64(v) => Some(*v as f64),
            ScalarValue::Int128(v) => Some(*v as f64),
            ScalarValue::UInt8(v) => Some(*v as f64),
            ScalarValue::UInt16(v) => Some(*v as f64),
            ScalarValue::UInt32(v) => Some(*v as f64),
            ScalarValue::UInt64(v) => Some(*v as f64),
            ScalarValue::Float32(v) => Some(*v as f64),
            ScalarValue::Float64(v) => Some(*v),
            _ => None,
        }
    }
}

impl fmt::Display for ScalarValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScalarValue::Null => write!(f, "NULL"),
            ScalarValue::Boolean(v) => write!(f, "{v}"),
            ScalarValue::Int8(v) => write!(f, "{v}"),
            ScalarValue::Int16(v) => write!(f, "{v}"),
            ScalarValue::Int32(v) => write!(f, "{v}"),
            ScalarValue::Int64(v) => write!(f, "{v}"),
            ScalarValue::Int128(v) => write!(f, "{v}"),
            ScalarValue::UInt8(v) => write!(f, "{v}"),
            ScalarValue::UInt16(v) => write!(f, "{v}"),
            ScalarValue::UInt32(v) => write!(f, "{v}"),
            ScalarValue::UInt64(v) => write!(f, "{v}"),
            ScalarValue::Float32(v) => write!(f, "{v}"),
            ScalarValue::Float64(v) => write!(f, "{v}"),
            ScalarValue::Utf8(v) => write!(f, "{v}"),
            ScalarValue::Binary(v) => write!(f, "<{} bytes>", v.len()),
        }
    }
}

// ---------------------------------------------------------------------------
// StreamColumnData: typed columnar storage
// ---------------------------------------------------------------------------

/// Typed column storage. Each variant holds a contiguous Vec of values.
/// Dispatch via the `dispatch_column!` macro for operations.
#[derive(Debug, Clone)]
pub enum StreamColumnData {
    Boolean(Vec<bool>),
    Int8(Vec<i8>),
    Int16(Vec<i16>),
    Int32(Vec<i32>),
    Int64(Vec<i64>),
    Int128(Vec<i128>),
    UInt8(Vec<u8>),
    UInt16(Vec<u16>),
    UInt32(Vec<u32>),
    UInt64(Vec<u64>),
    Float32(Vec<f32>),
    Float64(Vec<f64>),
    Utf8(Vec<String>),
    Binary(Vec<Vec<u8>>),
}

/// Applies an operation to the inner Vec of a StreamColumnData variant.
macro_rules! dispatch_column {
    ($col:expr, $vec:ident => $body:expr) => {
        match $col {
            StreamColumnData::Boolean($vec) => $body,
            StreamColumnData::Int8($vec) => $body,
            StreamColumnData::Int16($vec) => $body,
            StreamColumnData::Int32($vec) => $body,
            StreamColumnData::Int64($vec) => $body,
            StreamColumnData::Int128($vec) => $body,
            StreamColumnData::UInt8($vec) => $body,
            StreamColumnData::UInt16($vec) => $body,
            StreamColumnData::UInt32($vec) => $body,
            StreamColumnData::UInt64($vec) => $body,
            StreamColumnData::Float32($vec) => $body,
            StreamColumnData::Float64($vec) => $body,
            StreamColumnData::Utf8($vec) => $body,
            StreamColumnData::Binary($vec) => $body,
        }
    };
}

/// Applies an operation that rebuilds the same-typed StreamColumnData.
macro_rules! dispatch_column_rebuild {
    ($col:expr, $vec:ident => $body:expr) => {
        match $col {
            StreamColumnData::Boolean($vec) => StreamColumnData::Boolean($body),
            StreamColumnData::Int8($vec) => StreamColumnData::Int8($body),
            StreamColumnData::Int16($vec) => StreamColumnData::Int16($body),
            StreamColumnData::Int32($vec) => StreamColumnData::Int32($body),
            StreamColumnData::Int64($vec) => StreamColumnData::Int64($body),
            StreamColumnData::Int128($vec) => StreamColumnData::Int128($body),
            StreamColumnData::UInt8($vec) => StreamColumnData::UInt8($body),
            StreamColumnData::UInt16($vec) => StreamColumnData::UInt16($body),
            StreamColumnData::UInt32($vec) => StreamColumnData::UInt32($body),
            StreamColumnData::UInt64($vec) => StreamColumnData::UInt64($body),
            StreamColumnData::Float32($vec) => StreamColumnData::Float32($body),
            StreamColumnData::Float64($vec) => StreamColumnData::Float64($body),
            StreamColumnData::Utf8($vec) => StreamColumnData::Utf8($body),
            StreamColumnData::Binary($vec) => StreamColumnData::Binary($body),
        }
    };
}

impl StreamColumnData {
    #[inline]
    pub fn len(&self) -> usize {
        dispatch_column!(self, v => v.len())
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        dispatch_column!(self, v => v.is_empty())
    }

    /// Filter rows by boolean mask.
    pub fn filter(&self, mask: &[bool]) -> Self {
        dispatch_column_rebuild!(self, v => {
            v.iter()
                .zip(mask.iter())
                .filter_map(|(val, &keep)| if keep { Some(val.clone()) } else { None })
                .collect()
        })
    }

    /// Gather rows by indices.
    pub fn take(&self, indices: &[u32]) -> Self {
        dispatch_column_rebuild!(self, v => {
            indices.iter().map(|&idx| v[idx as usize].clone()).collect()
        })
    }

    /// Extract a contiguous sub-range.
    pub fn slice(&self, offset: usize, len: usize) -> Self {
        dispatch_column_rebuild!(self, v => {
            let end = (offset + len).min(v.len());
            v[offset..end].to_vec()
        })
    }

    /// Get scalar value at row index.
    pub fn get_scalar(&self, row: usize) -> ScalarValue {
        match self {
            StreamColumnData::Boolean(v) => ScalarValue::Boolean(v[row]),
            StreamColumnData::Int8(v) => ScalarValue::Int8(v[row]),
            StreamColumnData::Int16(v) => ScalarValue::Int16(v[row]),
            StreamColumnData::Int32(v) => ScalarValue::Int32(v[row]),
            StreamColumnData::Int64(v) => ScalarValue::Int64(v[row]),
            StreamColumnData::Int128(v) => ScalarValue::Int128(v[row]),
            StreamColumnData::UInt8(v) => ScalarValue::UInt8(v[row]),
            StreamColumnData::UInt16(v) => ScalarValue::UInt16(v[row]),
            StreamColumnData::UInt32(v) => ScalarValue::UInt32(v[row]),
            StreamColumnData::UInt64(v) => ScalarValue::UInt64(v[row]),
            StreamColumnData::Float32(v) => ScalarValue::Float32(v[row]),
            StreamColumnData::Float64(v) => ScalarValue::Float64(v[row]),
            StreamColumnData::Utf8(v) => ScalarValue::Utf8(v[row].clone()),
            StreamColumnData::Binary(v) => ScalarValue::Binary(v[row].clone()),
        }
    }

    /// Type identifier byte matching zyron-common TypeId values.
    pub fn type_id(&self) -> u8 {
        match self {
            StreamColumnData::Boolean(_) => 1,
            StreamColumnData::Int8(_) => 2,
            StreamColumnData::Int16(_) => 3,
            StreamColumnData::Int32(_) => 4,
            StreamColumnData::Int64(_) => 5,
            StreamColumnData::Int128(_) => 6,
            StreamColumnData::UInt8(_) => 7,
            StreamColumnData::UInt16(_) => 8,
            StreamColumnData::UInt32(_) => 9,
            StreamColumnData::UInt64(_) => 10,
            StreamColumnData::Float32(_) => 11,
            StreamColumnData::Float64(_) => 12,
            StreamColumnData::Utf8(_) => 13,
            StreamColumnData::Binary(_) => 14,
        }
    }

    /// Creates an empty column of the same type.
    pub fn empty_like(&self) -> Self {
        match self {
            StreamColumnData::Boolean(_) => StreamColumnData::Boolean(Vec::new()),
            StreamColumnData::Int8(_) => StreamColumnData::Int8(Vec::new()),
            StreamColumnData::Int16(_) => StreamColumnData::Int16(Vec::new()),
            StreamColumnData::Int32(_) => StreamColumnData::Int32(Vec::new()),
            StreamColumnData::Int64(_) => StreamColumnData::Int64(Vec::new()),
            StreamColumnData::Int128(_) => StreamColumnData::Int128(Vec::new()),
            StreamColumnData::UInt8(_) => StreamColumnData::UInt8(Vec::new()),
            StreamColumnData::UInt16(_) => StreamColumnData::UInt16(Vec::new()),
            StreamColumnData::UInt32(_) => StreamColumnData::UInt32(Vec::new()),
            StreamColumnData::UInt64(_) => StreamColumnData::UInt64(Vec::new()),
            StreamColumnData::Float32(_) => StreamColumnData::Float32(Vec::new()),
            StreamColumnData::Float64(_) => StreamColumnData::Float64(Vec::new()),
            StreamColumnData::Utf8(_) => StreamColumnData::Utf8(Vec::new()),
            StreamColumnData::Binary(_) => StreamColumnData::Binary(Vec::new()),
        }
    }

    /// Creates an empty column of the same type with pre-allocated capacity.
    pub fn empty_like_with_capacity(&self, capacity: usize) -> Self {
        match self {
            StreamColumnData::Boolean(_) => StreamColumnData::Boolean(Vec::with_capacity(capacity)),
            StreamColumnData::Int8(_) => StreamColumnData::Int8(Vec::with_capacity(capacity)),
            StreamColumnData::Int16(_) => StreamColumnData::Int16(Vec::with_capacity(capacity)),
            StreamColumnData::Int32(_) => StreamColumnData::Int32(Vec::with_capacity(capacity)),
            StreamColumnData::Int64(_) => StreamColumnData::Int64(Vec::with_capacity(capacity)),
            StreamColumnData::Int128(_) => StreamColumnData::Int128(Vec::with_capacity(capacity)),
            StreamColumnData::UInt8(_) => StreamColumnData::UInt8(Vec::with_capacity(capacity)),
            StreamColumnData::UInt16(_) => StreamColumnData::UInt16(Vec::with_capacity(capacity)),
            StreamColumnData::UInt32(_) => StreamColumnData::UInt32(Vec::with_capacity(capacity)),
            StreamColumnData::UInt64(_) => StreamColumnData::UInt64(Vec::with_capacity(capacity)),
            StreamColumnData::Float32(_) => StreamColumnData::Float32(Vec::with_capacity(capacity)),
            StreamColumnData::Float64(_) => StreamColumnData::Float64(Vec::with_capacity(capacity)),
            StreamColumnData::Utf8(_) => StreamColumnData::Utf8(Vec::with_capacity(capacity)),
            StreamColumnData::Binary(_) => StreamColumnData::Binary(Vec::with_capacity(capacity)),
        }
    }

    /// Pushes a scalar value into this column. The scalar type must match the
    /// column type. Mismatched types are silently ignored.
    pub fn push_scalar(&mut self, scalar: &ScalarValue) {
        match (self, scalar) {
            (StreamColumnData::Boolean(v), ScalarValue::Boolean(s)) => v.push(*s),
            (StreamColumnData::Int8(v), ScalarValue::Int8(s)) => v.push(*s),
            (StreamColumnData::Int16(v), ScalarValue::Int16(s)) => v.push(*s),
            (StreamColumnData::Int32(v), ScalarValue::Int32(s)) => v.push(*s),
            (StreamColumnData::Int64(v), ScalarValue::Int64(s)) => v.push(*s),
            (StreamColumnData::Int128(v), ScalarValue::Int128(s)) => v.push(*s),
            (StreamColumnData::UInt8(v), ScalarValue::UInt8(s)) => v.push(*s),
            (StreamColumnData::UInt16(v), ScalarValue::UInt16(s)) => v.push(*s),
            (StreamColumnData::UInt32(v), ScalarValue::UInt32(s)) => v.push(*s),
            (StreamColumnData::UInt64(v), ScalarValue::UInt64(s)) => v.push(*s),
            (StreamColumnData::Float32(v), ScalarValue::Float32(s)) => v.push(*s),
            (StreamColumnData::Float64(v), ScalarValue::Float64(s)) => v.push(*s),
            (StreamColumnData::Utf8(v), ScalarValue::Utf8(s)) => v.push(s.clone()),
            (StreamColumnData::Binary(v), ScalarValue::Binary(s)) => v.push(s.clone()),
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// StreamColumn: typed data + null bitmap
// ---------------------------------------------------------------------------

/// A single column of typed data with a null bitmap.
#[derive(Debug, Clone)]
pub struct StreamColumn {
    pub data: StreamColumnData,
    pub nulls: NullBitmap,
    pub type_id: u8,
}

impl StreamColumn {
    pub fn new(data: StreamColumnData, nulls: NullBitmap) -> Self {
        let type_id = data.type_id();
        debug_assert_eq!(data.len(), nulls.len());
        Self {
            data,
            nulls,
            type_id,
        }
    }

    /// Creates a column with no nulls.
    pub fn from_data(data: StreamColumnData) -> Self {
        let len = data.len();
        let type_id = data.type_id();
        Self {
            data,
            nulls: NullBitmap::new_valid(len),
            type_id,
        }
    }

    #[inline(always)]
    pub fn is_null(&self, idx: usize) -> bool {
        self.nulls.is_null(idx)
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get scalar value at row, returning Null if the row is null.
    pub fn get_scalar(&self, row: usize) -> ScalarValue {
        if self.nulls.is_null(row) {
            ScalarValue::Null
        } else {
            self.data.get_scalar(row)
        }
    }

    pub fn filter(&self, mask: &[bool]) -> Self {
        Self {
            data: self.data.filter(mask),
            nulls: self.nulls.filter(mask),
            type_id: self.type_id,
        }
    }

    pub fn take(&self, indices: &[u32]) -> Self {
        Self {
            data: self.data.take(indices),
            nulls: self.nulls.take(indices),
            type_id: self.type_id,
        }
    }

    pub fn slice(&self, offset: usize, len: usize) -> Self {
        Self {
            data: self.data.slice(offset, len),
            nulls: self.nulls.slice(offset, len),
            type_id: self.type_id,
        }
    }
}

// ---------------------------------------------------------------------------
// StreamBatch: columnar micro-batch
// ---------------------------------------------------------------------------

/// Number of rows per streaming micro-batch.
pub const STREAM_BATCH_SIZE: usize = 1024;

/// A columnar batch of rows for streaming operators.
/// All columns have the same number of rows.
#[derive(Debug, Clone)]
pub struct StreamBatch {
    pub columns: Vec<StreamColumn>,
    pub num_rows: usize,
}

impl StreamBatch {
    pub fn new(columns: Vec<StreamColumn>) -> Self {
        let num_rows = columns.first().map_or(0, |c| c.len());
        debug_assert!(columns.iter().all(|c| c.len() == num_rows));
        Self { columns, num_rows }
    }

    pub fn empty() -> Self {
        Self {
            columns: Vec::new(),
            num_rows: 0,
        }
    }

    pub fn column(&self, idx: usize) -> &StreamColumn {
        &self.columns[idx]
    }

    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    pub fn is_empty(&self) -> bool {
        self.num_rows == 0
    }

    pub fn filter(&self, mask: &[bool]) -> Self {
        let columns: Vec<StreamColumn> = self.columns.iter().map(|c| c.filter(mask)).collect();
        let num_rows = columns.first().map_or(0, |c| c.len());
        Self { columns, num_rows }
    }

    pub fn take(&self, indices: &[u32]) -> Self {
        let columns: Vec<StreamColumn> = self.columns.iter().map(|c| c.take(indices)).collect();
        let num_rows = indices.len();
        Self { columns, num_rows }
    }

    pub fn slice(&self, offset: usize, len: usize) -> Self {
        let actual_len = len.min(self.num_rows.saturating_sub(offset));
        let columns: Vec<StreamColumn> = self
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_null_bitmap_basic() {
        let mut bm = NullBitmap::new_valid(10);
        assert!(!bm.has_nulls());
        assert_eq!(bm.null_count(), 0);

        bm.set_null(3);
        bm.set_null(7);
        assert!(bm.has_nulls());
        assert_eq!(bm.null_count(), 2);
        assert!(bm.is_null(3));
        assert!(bm.is_null(7));
        assert!(bm.is_valid(0));
        assert!(bm.is_valid(5));
    }

    #[test]
    fn test_null_bitmap_push() {
        let mut bm = NullBitmap::empty();
        for i in 0..100 {
            bm.push(i % 10 == 0);
        }
        assert_eq!(bm.len(), 100);
        assert_eq!(bm.null_count(), 10);
        assert!(bm.is_null(0));
        assert!(bm.is_null(10));
        assert!(bm.is_valid(1));
    }

    #[test]
    fn test_null_bitmap_filter() {
        let mut bm = NullBitmap::new_valid(4);
        bm.set_null(1);
        let filtered = bm.filter(&[true, false, true, true]);
        assert_eq!(filtered.len(), 3);
        assert!(filtered.is_valid(0));
        assert!(filtered.is_valid(1));
        assert!(filtered.is_valid(2));
    }

    #[test]
    fn test_column_data_operations() {
        let data = StreamColumnData::Int64(vec![10, 20, 30, 40, 50]);
        assert_eq!(data.len(), 5);

        let filtered = data.filter(&[true, false, true, false, true]);
        if let StreamColumnData::Int64(v) = &filtered {
            assert_eq!(v, &[10, 30, 50]);
        } else {
            panic!("wrong type");
        }

        let taken = data.take(&[4, 2, 0]);
        if let StreamColumnData::Int64(v) = &taken {
            assert_eq!(v, &[50, 30, 10]);
        } else {
            panic!("wrong type");
        }

        let sliced = data.slice(1, 3);
        if let StreamColumnData::Int64(v) = &sliced {
            assert_eq!(v, &[20, 30, 40]);
        } else {
            panic!("wrong type");
        }
    }

    #[test]
    fn test_stream_column_get_scalar() {
        let mut nulls = NullBitmap::new_valid(3);
        nulls.set_null(1);
        let col = StreamColumn::new(StreamColumnData::Int64(vec![10, 20, 30]), nulls);
        assert_eq!(col.get_scalar(0), ScalarValue::Int64(10));
        assert_eq!(col.get_scalar(1), ScalarValue::Null);
        assert_eq!(col.get_scalar(2), ScalarValue::Int64(30));
    }

    #[test]
    fn test_stream_batch_basic() {
        let col1 = StreamColumn::from_data(StreamColumnData::Int64(vec![1, 2, 3]));
        let col2 = StreamColumn::from_data(StreamColumnData::Float64(vec![1.0, 2.0, 3.0]));
        let batch = StreamBatch::new(vec![col1, col2]);

        assert_eq!(batch.num_rows, 3);
        assert_eq!(batch.num_columns(), 2);

        let sliced = batch.slice(1, 2);
        assert_eq!(sliced.num_rows, 2);
        if let StreamColumnData::Int64(v) = &sliced.column(0).data {
            assert_eq!(v, &[2, 3]);
        }
    }

    #[test]
    fn test_scalar_value_to_f64() {
        assert_eq!(ScalarValue::Int64(42).to_f64(), Some(42.0));
        assert_eq!(ScalarValue::Float64(3.14).to_f64(), Some(3.14));
        assert_eq!(ScalarValue::Utf8("hello".into()).to_f64(), None);
        assert_eq!(ScalarValue::Null.to_f64(), None);
    }

    #[test]
    fn test_empty_batch() {
        let batch = StreamBatch::empty();
        assert!(batch.is_empty());
        assert_eq!(batch.num_rows, 0);
        assert_eq!(batch.num_columns(), 0);
    }
}

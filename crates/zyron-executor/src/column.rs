//! Columnar data types for vectorized query execution.
//!
//! Provides typed column vectors with packed null bitmaps, a runtime
//! ScalarValue for row-level operations, and the Column wrapper that
//! pairs data with nullability metadata.

use std::fmt;
use zyron_common::TypeId;

// ---------------------------------------------------------------------------
// Null bitmap - packed u64 words, bit set = null
// ---------------------------------------------------------------------------

/// Packed null bitmap using 64-bit words. A set bit indicates a null value.
#[derive(Debug, Clone)]
pub struct NullBitmap {
    words: Vec<u64>,
    len: usize,
}

impl NullBitmap {
    /// Creates a bitmap with no nulls.
    pub fn none(len: usize) -> Self {
        let word_count = (len + 63) / 64;
        Self {
            words: vec![0; word_count],
            len,
        }
    }

    /// Creates a bitmap where every value is null.
    pub fn all_null(len: usize) -> Self {
        let word_count = (len + 63) / 64;
        let mut words = vec![u64::MAX; word_count];
        // Clear bits beyond len in the last word.
        let remainder = len % 64;
        if remainder > 0 && !words.is_empty() {
            let last = words.len() - 1;
            words[last] = (1u64 << remainder) - 1;
        }
        Self { words, len }
    }

    /// Creates an empty bitmap.
    pub fn empty() -> Self {
        Self {
            words: Vec::new(),
            len: 0,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn is_null(&self, idx: usize) -> bool {
        debug_assert!(idx < self.len);
        (self.words[idx / 64] >> (idx % 64)) & 1 == 1
    }

    #[inline]
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

    /// Appends a null/valid indicator.
    pub fn push(&mut self, is_null: bool) {
        if self.len % 64 == 0 {
            self.words.push(0);
        }
        if is_null {
            let word_idx = self.len / 64;
            let bit_idx = self.len % 64;
            self.words[word_idx] |= 1u64 << bit_idx;
        }
        self.len += 1;
    }

    /// Returns true if any value is null.
    pub fn has_nulls(&self) -> bool {
        self.words.iter().any(|&w| w != 0)
    }

    /// Counts the number of null values.
    pub fn null_count(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Selects rows where mask[i] is true.
    pub fn filter(&self, mask: &[bool]) -> Self {
        if !self.has_nulls() {
            let count = mask.iter().filter(|&&m| m).count();
            return Self::none(count);
        }
        let mut result = Self::empty();
        for (i, &keep) in mask.iter().enumerate() {
            if keep {
                result.push(self.is_null(i));
            }
        }
        result
    }

    /// Reorders rows by index.
    pub fn take(&self, indices: &[u32]) -> Self {
        if !self.has_nulls() {
            return Self::none(indices.len());
        }
        let mut result = Self::empty();
        for &idx in indices {
            result.push(self.is_null(idx as usize));
        }
        result
    }

    /// Extracts a contiguous sub-range.
    pub fn slice(&self, offset: usize, len: usize) -> Self {
        if !self.has_nulls() {
            return Self::none(len);
        }
        // Word-aligned fast path: copy whole words directly.
        if offset % 64 == 0 {
            let word_start = offset / 64;
            let word_count = (len + 63) / 64;
            let words = self.words[word_start..word_start + word_count].to_vec();
            return Self { words, len };
        }
        let mut result = Self::empty();
        for i in offset..offset + len {
            result.push(self.is_null(i));
        }
        result
    }

    /// Appends all bits from another bitmap.
    pub fn extend_from(&mut self, other: &NullBitmap) {
        // Fast path: if current bitmap is word-aligned, use bulk operations.
        if self.len % 64 == 0 && !other.words.is_empty() {
            self.words.extend_from_slice(&other.words);
            self.len += other.len;
            return;
        }
        // Slow path for unaligned case.
        self.words.reserve((other.len + 63) / 64);
        for i in 0..other.len {
            self.push(other.is_null(i));
        }
    }

    /// Appends a single bit from another bitmap at the given index.
    #[inline]
    pub fn push_from(&mut self, other: &NullBitmap, idx: usize) {
        self.push(other.is_null(idx));
    }

    /// Gathers bits at the given indices from another bitmap.
    /// Fast path: if source has no nulls, extends with zeros (all valid).
    pub fn gather_from(&mut self, other: &NullBitmap, indices: &[u32]) {
        if !other.has_nulls() {
            let new_len = self.len + indices.len();
            self.words.resize((new_len + 63) / 64, 0);
            self.len = new_len;
            return;
        }
        for &idx in indices {
            self.push(other.is_null(idx as usize));
        }
    }
}

// ---------------------------------------------------------------------------
// ScalarValue - runtime typed single value
// ---------------------------------------------------------------------------

/// A single typed value for row-level operations, grouping keys,
/// and accumulator state.
#[derive(Debug, Clone)]
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
    FixedBinary16([u8; 16]),
}

impl PartialEq for ScalarValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ScalarValue::Null, ScalarValue::Null) => true,
            (ScalarValue::Boolean(a), ScalarValue::Boolean(b)) => a == b,
            (ScalarValue::Int8(a), ScalarValue::Int8(b)) => a == b,
            (ScalarValue::Int16(a), ScalarValue::Int16(b)) => a == b,
            (ScalarValue::Int32(a), ScalarValue::Int32(b)) => a == b,
            (ScalarValue::Int64(a), ScalarValue::Int64(b)) => a == b,
            (ScalarValue::Int128(a), ScalarValue::Int128(b)) => a == b,
            (ScalarValue::UInt8(a), ScalarValue::UInt8(b)) => a == b,
            (ScalarValue::UInt16(a), ScalarValue::UInt16(b)) => a == b,
            (ScalarValue::UInt32(a), ScalarValue::UInt32(b)) => a == b,
            (ScalarValue::UInt64(a), ScalarValue::UInt64(b)) => a == b,
            (ScalarValue::Float32(a), ScalarValue::Float32(b)) => a.to_bits() == b.to_bits(),
            (ScalarValue::Float64(a), ScalarValue::Float64(b)) => a.to_bits() == b.to_bits(),
            (ScalarValue::Utf8(a), ScalarValue::Utf8(b)) => a == b,
            (ScalarValue::Binary(a), ScalarValue::Binary(b)) => a == b,
            (ScalarValue::FixedBinary16(a), ScalarValue::FixedBinary16(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for ScalarValue {}

impl std::hash::Hash for ScalarValue {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            ScalarValue::Null => {}
            ScalarValue::Boolean(v) => v.hash(state),
            ScalarValue::Int8(v) => v.hash(state),
            ScalarValue::Int16(v) => v.hash(state),
            ScalarValue::Int32(v) => v.hash(state),
            ScalarValue::Int64(v) => v.hash(state),
            ScalarValue::Int128(v) => v.hash(state),
            ScalarValue::UInt8(v) => v.hash(state),
            ScalarValue::UInt16(v) => v.hash(state),
            ScalarValue::UInt32(v) => v.hash(state),
            ScalarValue::UInt64(v) => v.hash(state),
            ScalarValue::Float32(v) => v.to_bits().hash(state),
            ScalarValue::Float64(v) => v.to_bits().hash(state),
            ScalarValue::Utf8(v) => v.hash(state),
            ScalarValue::Binary(v) => v.hash(state),
            ScalarValue::FixedBinary16(v) => v.hash(state),
        }
    }
}

impl PartialOrd for ScalarValue {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (ScalarValue::Null, ScalarValue::Null) => Some(std::cmp::Ordering::Equal),
            (ScalarValue::Null, _) => Some(std::cmp::Ordering::Less),
            (_, ScalarValue::Null) => Some(std::cmp::Ordering::Greater),
            (ScalarValue::Boolean(a), ScalarValue::Boolean(b)) => a.partial_cmp(b),
            (ScalarValue::Int8(a), ScalarValue::Int8(b)) => a.partial_cmp(b),
            (ScalarValue::Int16(a), ScalarValue::Int16(b)) => a.partial_cmp(b),
            (ScalarValue::Int32(a), ScalarValue::Int32(b)) => a.partial_cmp(b),
            (ScalarValue::Int64(a), ScalarValue::Int64(b)) => a.partial_cmp(b),
            (ScalarValue::Int128(a), ScalarValue::Int128(b)) => a.partial_cmp(b),
            (ScalarValue::UInt8(a), ScalarValue::UInt8(b)) => a.partial_cmp(b),
            (ScalarValue::UInt16(a), ScalarValue::UInt16(b)) => a.partial_cmp(b),
            (ScalarValue::UInt32(a), ScalarValue::UInt32(b)) => a.partial_cmp(b),
            (ScalarValue::UInt64(a), ScalarValue::UInt64(b)) => a.partial_cmp(b),
            (ScalarValue::Float32(a), ScalarValue::Float32(b)) => a.partial_cmp(b),
            (ScalarValue::Float64(a), ScalarValue::Float64(b)) => a.partial_cmp(b),
            (ScalarValue::Utf8(a), ScalarValue::Utf8(b)) => a.partial_cmp(b),
            (ScalarValue::Binary(a), ScalarValue::Binary(b)) => a.partial_cmp(b),
            (ScalarValue::FixedBinary16(a), ScalarValue::FixedBinary16(b)) => a.partial_cmp(b),
            _ => None,
        }
    }
}

impl ScalarValue {
    /// Returns the TypeId of this scalar.
    pub fn type_id(&self) -> TypeId {
        match self {
            ScalarValue::Null => TypeId::Null,
            ScalarValue::Boolean(_) => TypeId::Boolean,
            ScalarValue::Int8(_) => TypeId::Int8,
            ScalarValue::Int16(_) => TypeId::Int16,
            ScalarValue::Int32(_) => TypeId::Int32,
            ScalarValue::Int64(_) => TypeId::Int64,
            ScalarValue::Int128(_) => TypeId::Int128,
            ScalarValue::UInt8(_) => TypeId::UInt8,
            ScalarValue::UInt16(_) => TypeId::UInt16,
            ScalarValue::UInt32(_) => TypeId::UInt32,
            ScalarValue::UInt64(_) => TypeId::UInt64,
            ScalarValue::Float32(_) => TypeId::Float32,
            ScalarValue::Float64(_) => TypeId::Float64,
            ScalarValue::Utf8(_) => TypeId::Text,
            ScalarValue::Binary(_) => TypeId::Bytea,
            ScalarValue::FixedBinary16(_) => TypeId::Uuid,
        }
    }

    /// Converts to f64 for numeric accumulators.
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

    /// Returns true if this is a Null variant.
    pub fn is_null(&self) -> bool {
        matches!(self, ScalarValue::Null)
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
            ScalarValue::FixedBinary16(_) => write!(f, "<16 bytes>"),
        }
    }
}

// ---------------------------------------------------------------------------
// ColumnData - typed storage for column values
// ---------------------------------------------------------------------------

/// Typed storage for column values. Each variant holds a Vec of the
/// corresponding Rust type.
#[derive(Debug, Clone)]
pub enum ColumnData {
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
    FixedBinary16(Vec<[u8; 16]>),
}

/// Applies an operation to each ColumnData variant, returning a new ColumnData.
macro_rules! dispatch_column {
    ($self:expr, $v:ident, $body:expr) => {
        match $self {
            ColumnData::Boolean($v) => ColumnData::Boolean($body),
            ColumnData::Int8($v) => ColumnData::Int8($body),
            ColumnData::Int16($v) => ColumnData::Int16($body),
            ColumnData::Int32($v) => ColumnData::Int32($body),
            ColumnData::Int64($v) => ColumnData::Int64($body),
            ColumnData::Int128($v) => ColumnData::Int128($body),
            ColumnData::UInt8($v) => ColumnData::UInt8($body),
            ColumnData::UInt16($v) => ColumnData::UInt16($body),
            ColumnData::UInt32($v) => ColumnData::UInt32($body),
            ColumnData::UInt64($v) => ColumnData::UInt64($body),
            ColumnData::Float32($v) => ColumnData::Float32($body),
            ColumnData::Float64($v) => ColumnData::Float64($body),
            ColumnData::Utf8($v) => ColumnData::Utf8($body),
            ColumnData::Binary($v) => ColumnData::Binary($body),
            ColumnData::FixedBinary16($v) => ColumnData::FixedBinary16($body),
        }
    };
}

impl ColumnData {
    /// Creates an empty column of the given type.
    pub fn empty(type_id: TypeId) -> Self {
        Self::with_capacity(type_id, 0)
    }

    /// Creates an empty column with pre-allocated capacity.
    pub fn with_capacity(type_id: TypeId, cap: usize) -> Self {
        match type_id {
            TypeId::Boolean => ColumnData::Boolean(Vec::with_capacity(cap)),
            TypeId::Int8 => ColumnData::Int8(Vec::with_capacity(cap)),
            TypeId::Int16 => ColumnData::Int16(Vec::with_capacity(cap)),
            TypeId::Int32 | TypeId::Date => ColumnData::Int32(Vec::with_capacity(cap)),
            TypeId::Int64 | TypeId::Time | TypeId::Timestamp | TypeId::TimestampTz => {
                ColumnData::Int64(Vec::with_capacity(cap))
            }
            TypeId::Int128 | TypeId::Decimal => ColumnData::Int128(Vec::with_capacity(cap)),
            TypeId::UInt8 => ColumnData::UInt8(Vec::with_capacity(cap)),
            TypeId::UInt16 => ColumnData::UInt16(Vec::with_capacity(cap)),
            TypeId::UInt32 => ColumnData::UInt32(Vec::with_capacity(cap)),
            TypeId::UInt64 => ColumnData::UInt64(Vec::with_capacity(cap)),
            TypeId::UInt128 => ColumnData::Int128(Vec::with_capacity(cap)),
            TypeId::Float32 => ColumnData::Float32(Vec::with_capacity(cap)),
            TypeId::Float64 => ColumnData::Float64(Vec::with_capacity(cap)),
            TypeId::Char | TypeId::Varchar | TypeId::Text | TypeId::Json | TypeId::Jsonb => {
                ColumnData::Utf8(Vec::with_capacity(cap))
            }
            TypeId::Binary
            | TypeId::Varbinary
            | TypeId::Bytea
            | TypeId::Array
            | TypeId::Composite
            | TypeId::Vector => ColumnData::Binary(Vec::with_capacity(cap)),
            TypeId::Uuid | TypeId::Interval => ColumnData::FixedBinary16(Vec::with_capacity(cap)),
            TypeId::Null => ColumnData::Boolean(Vec::with_capacity(cap)),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            ColumnData::Boolean(v) => v.len(),
            ColumnData::Int8(v) => v.len(),
            ColumnData::Int16(v) => v.len(),
            ColumnData::Int32(v) => v.len(),
            ColumnData::Int64(v) => v.len(),
            ColumnData::Int128(v) => v.len(),
            ColumnData::UInt8(v) => v.len(),
            ColumnData::UInt16(v) => v.len(),
            ColumnData::UInt32(v) => v.len(),
            ColumnData::UInt64(v) => v.len(),
            ColumnData::Float32(v) => v.len(),
            ColumnData::Float64(v) => v.len(),
            ColumnData::Utf8(v) => v.len(),
            ColumnData::Binary(v) => v.len(),
            ColumnData::FixedBinary16(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Selects rows where mask[i] is true.
    pub fn filter(&self, mask: &[bool]) -> Self {
        dispatch_column!(self, v, {
            v.iter()
                .zip(mask)
                .filter_map(|(val, &keep)| if keep { Some(val.clone()) } else { None })
                .collect()
        })
    }

    /// Reorders rows by indices.
    pub fn take(&self, indices: &[u32]) -> Self {
        dispatch_column!(self, v, {
            indices.iter().map(|&i| v[i as usize].clone()).collect()
        })
    }

    /// Extracts a contiguous sub-range.
    pub fn slice(&self, offset: usize, len: usize) -> Self {
        dispatch_column!(self, v, v[offset..offset + len].to_vec())
    }

    /// Extracts a scalar value at the given row.
    pub fn get_scalar(&self, row: usize) -> ScalarValue {
        match self {
            ColumnData::Boolean(v) => ScalarValue::Boolean(v[row]),
            ColumnData::Int8(v) => ScalarValue::Int8(v[row]),
            ColumnData::Int16(v) => ScalarValue::Int16(v[row]),
            ColumnData::Int32(v) => ScalarValue::Int32(v[row]),
            ColumnData::Int64(v) => ScalarValue::Int64(v[row]),
            ColumnData::Int128(v) => ScalarValue::Int128(v[row]),
            ColumnData::UInt8(v) => ScalarValue::UInt8(v[row]),
            ColumnData::UInt16(v) => ScalarValue::UInt16(v[row]),
            ColumnData::UInt32(v) => ScalarValue::UInt32(v[row]),
            ColumnData::UInt64(v) => ScalarValue::UInt64(v[row]),
            ColumnData::Float32(v) => ScalarValue::Float32(v[row]),
            ColumnData::Float64(v) => ScalarValue::Float64(v[row]),
            ColumnData::Utf8(v) => ScalarValue::Utf8(v[row].clone()),
            ColumnData::Binary(v) => ScalarValue::Binary(v[row].clone()),
            ColumnData::FixedBinary16(v) => ScalarValue::FixedBinary16(v[row]),
        }
    }

    /// Appends a scalar value. Pushes a zero/empty default if the type does not match.
    pub fn push_scalar(&mut self, scalar: &ScalarValue) {
        match (self, scalar) {
            (ColumnData::Boolean(v), ScalarValue::Boolean(s)) => v.push(*s),
            (ColumnData::Boolean(v), _) => v.push(false),
            (ColumnData::Int8(v), ScalarValue::Int8(s)) => v.push(*s),
            (ColumnData::Int8(v), _) => v.push(0),
            (ColumnData::Int16(v), ScalarValue::Int16(s)) => v.push(*s),
            (ColumnData::Int16(v), _) => v.push(0),
            (ColumnData::Int32(v), ScalarValue::Int32(s)) => v.push(*s),
            (ColumnData::Int32(v), _) => v.push(0),
            (ColumnData::Int64(v), ScalarValue::Int64(s)) => v.push(*s),
            (ColumnData::Int64(v), _) => v.push(0),
            (ColumnData::Int128(v), ScalarValue::Int128(s)) => v.push(*s),
            (ColumnData::Int128(v), _) => v.push(0),
            (ColumnData::UInt8(v), ScalarValue::UInt8(s)) => v.push(*s),
            (ColumnData::UInt8(v), _) => v.push(0),
            (ColumnData::UInt16(v), ScalarValue::UInt16(s)) => v.push(*s),
            (ColumnData::UInt16(v), _) => v.push(0),
            (ColumnData::UInt32(v), ScalarValue::UInt32(s)) => v.push(*s),
            (ColumnData::UInt32(v), _) => v.push(0),
            (ColumnData::UInt64(v), ScalarValue::UInt64(s)) => v.push(*s),
            (ColumnData::UInt64(v), _) => v.push(0),
            (ColumnData::Float32(v), ScalarValue::Float32(s)) => v.push(*s),
            (ColumnData::Float32(v), _) => v.push(0.0),
            (ColumnData::Float64(v), ScalarValue::Float64(s)) => v.push(*s),
            (ColumnData::Float64(v), _) => v.push(0.0),
            (ColumnData::Utf8(v), ScalarValue::Utf8(s)) => v.push(s.clone()),
            (ColumnData::Utf8(v), _) => v.push(String::new()),
            (ColumnData::Binary(v), ScalarValue::Binary(s)) => v.push(s.clone()),
            (ColumnData::Binary(v), _) => v.push(Vec::new()),
            (ColumnData::FixedBinary16(v), ScalarValue::FixedBinary16(s)) => v.push(*s),
            (ColumnData::FixedBinary16(v), _) => v.push([0u8; 16]),
        }
    }

    /// Appends all values from another ColumnData of the same variant.
    /// Panics if the variants do not match.
    pub fn extend_from(&mut self, other: &ColumnData) {
        match (self, other) {
            (ColumnData::Boolean(v), ColumnData::Boolean(o)) => v.extend_from_slice(o),
            (ColumnData::Int8(v), ColumnData::Int8(o)) => v.extend_from_slice(o),
            (ColumnData::Int16(v), ColumnData::Int16(o)) => v.extend_from_slice(o),
            (ColumnData::Int32(v), ColumnData::Int32(o)) => v.extend_from_slice(o),
            (ColumnData::Int64(v), ColumnData::Int64(o)) => v.extend_from_slice(o),
            (ColumnData::Int128(v), ColumnData::Int128(o)) => v.extend_from_slice(o),
            (ColumnData::UInt8(v), ColumnData::UInt8(o)) => v.extend_from_slice(o),
            (ColumnData::UInt16(v), ColumnData::UInt16(o)) => v.extend_from_slice(o),
            (ColumnData::UInt32(v), ColumnData::UInt32(o)) => v.extend_from_slice(o),
            (ColumnData::UInt64(v), ColumnData::UInt64(o)) => v.extend_from_slice(o),
            (ColumnData::Float32(v), ColumnData::Float32(o)) => v.extend_from_slice(o),
            (ColumnData::Float64(v), ColumnData::Float64(o)) => v.extend_from_slice(o),
            (ColumnData::Utf8(v), ColumnData::Utf8(o)) => v.extend(o.iter().cloned()),
            (ColumnData::Binary(v), ColumnData::Binary(o)) => v.extend(o.iter().cloned()),
            (ColumnData::FixedBinary16(v), ColumnData::FixedBinary16(o)) => v.extend_from_slice(o),
            _ => panic!("ColumnData::extend_from: type mismatch"),
        }
    }

    /// Appends a single value at `idx` from another ColumnData of the same variant.
    /// No ScalarValue intermediary.
    #[inline]
    pub fn push_from(&mut self, other: &ColumnData, idx: usize) {
        match (self, other) {
            (ColumnData::Boolean(v), ColumnData::Boolean(o)) => v.push(o[idx]),
            (ColumnData::Int8(v), ColumnData::Int8(o)) => v.push(o[idx]),
            (ColumnData::Int16(v), ColumnData::Int16(o)) => v.push(o[idx]),
            (ColumnData::Int32(v), ColumnData::Int32(o)) => v.push(o[idx]),
            (ColumnData::Int64(v), ColumnData::Int64(o)) => v.push(o[idx]),
            (ColumnData::Int128(v), ColumnData::Int128(o)) => v.push(o[idx]),
            (ColumnData::UInt8(v), ColumnData::UInt8(o)) => v.push(o[idx]),
            (ColumnData::UInt16(v), ColumnData::UInt16(o)) => v.push(o[idx]),
            (ColumnData::UInt32(v), ColumnData::UInt32(o)) => v.push(o[idx]),
            (ColumnData::UInt64(v), ColumnData::UInt64(o)) => v.push(o[idx]),
            (ColumnData::Float32(v), ColumnData::Float32(o)) => v.push(o[idx]),
            (ColumnData::Float64(v), ColumnData::Float64(o)) => v.push(o[idx]),
            (ColumnData::Utf8(v), ColumnData::Utf8(o)) => v.push(o[idx].clone()),
            (ColumnData::Binary(v), ColumnData::Binary(o)) => v.push(o[idx].clone()),
            (ColumnData::FixedBinary16(v), ColumnData::FixedBinary16(o)) => v.push(o[idx]),
            _ => panic!("ColumnData::push_from: type mismatch"),
        }
    }

    /// Appends values at the given indices from another ColumnData.
    pub fn gather_from(&mut self, other: &ColumnData, indices: &[u32]) {
        match (self, other) {
            (ColumnData::Boolean(v), ColumnData::Boolean(o)) => {
                v.reserve(indices.len());
                for &i in indices {
                    v.push(o[i as usize]);
                }
            }
            (ColumnData::Int8(v), ColumnData::Int8(o)) => {
                v.reserve(indices.len());
                for &i in indices {
                    v.push(o[i as usize]);
                }
            }
            (ColumnData::Int16(v), ColumnData::Int16(o)) => {
                v.reserve(indices.len());
                for &i in indices {
                    v.push(o[i as usize]);
                }
            }
            (ColumnData::Int32(v), ColumnData::Int32(o)) => {
                v.reserve(indices.len());
                for &i in indices {
                    v.push(o[i as usize]);
                }
            }
            (ColumnData::Int64(v), ColumnData::Int64(o)) => {
                v.reserve(indices.len());
                for &i in indices {
                    v.push(o[i as usize]);
                }
            }
            (ColumnData::Int128(v), ColumnData::Int128(o)) => {
                v.reserve(indices.len());
                for &i in indices {
                    v.push(o[i as usize]);
                }
            }
            (ColumnData::UInt8(v), ColumnData::UInt8(o)) => {
                v.reserve(indices.len());
                for &i in indices {
                    v.push(o[i as usize]);
                }
            }
            (ColumnData::UInt16(v), ColumnData::UInt16(o)) => {
                v.reserve(indices.len());
                for &i in indices {
                    v.push(o[i as usize]);
                }
            }
            (ColumnData::UInt32(v), ColumnData::UInt32(o)) => {
                v.reserve(indices.len());
                for &i in indices {
                    v.push(o[i as usize]);
                }
            }
            (ColumnData::UInt64(v), ColumnData::UInt64(o)) => {
                v.reserve(indices.len());
                for &i in indices {
                    v.push(o[i as usize]);
                }
            }
            (ColumnData::Float32(v), ColumnData::Float32(o)) => {
                v.reserve(indices.len());
                for &i in indices {
                    v.push(o[i as usize]);
                }
            }
            (ColumnData::Float64(v), ColumnData::Float64(o)) => {
                v.reserve(indices.len());
                for &i in indices {
                    v.push(o[i as usize]);
                }
            }
            (ColumnData::Utf8(v), ColumnData::Utf8(o)) => {
                v.reserve(indices.len());
                for &i in indices {
                    v.push(o[i as usize].clone());
                }
            }
            (ColumnData::Binary(v), ColumnData::Binary(o)) => {
                v.reserve(indices.len());
                for &i in indices {
                    v.push(o[i as usize].clone());
                }
            }
            (ColumnData::FixedBinary16(v), ColumnData::FixedBinary16(o)) => {
                v.reserve(indices.len());
                for &i in indices {
                    v.push(o[i as usize]);
                }
            }
            _ => panic!("ColumnData::gather_from: type mismatch"),
        }
    }

    /// Creates a column filled with default values for the given type.
    /// Used for null columns without per-row push_scalar overhead.
    pub fn null_fill(type_id: TypeId, len: usize) -> Self {
        match type_id {
            TypeId::Boolean => ColumnData::Boolean(vec![false; len]),
            TypeId::Int8 => ColumnData::Int8(vec![0; len]),
            TypeId::Int16 => ColumnData::Int16(vec![0; len]),
            TypeId::Int32 | TypeId::Date => ColumnData::Int32(vec![0; len]),
            TypeId::Int64 | TypeId::Time | TypeId::Timestamp | TypeId::TimestampTz => {
                ColumnData::Int64(vec![0; len])
            }
            TypeId::Int128 | TypeId::Decimal | TypeId::UInt128 => ColumnData::Int128(vec![0; len]),
            TypeId::UInt8 => ColumnData::UInt8(vec![0; len]),
            TypeId::UInt16 => ColumnData::UInt16(vec![0; len]),
            TypeId::UInt32 => ColumnData::UInt32(vec![0; len]),
            TypeId::UInt64 => ColumnData::UInt64(vec![0; len]),
            TypeId::Float32 => ColumnData::Float32(vec![0.0; len]),
            TypeId::Float64 => ColumnData::Float64(vec![0.0; len]),
            TypeId::Char | TypeId::Varchar | TypeId::Text | TypeId::Json | TypeId::Jsonb => {
                ColumnData::Utf8(vec![String::new(); len])
            }
            TypeId::Binary
            | TypeId::Varbinary
            | TypeId::Bytea
            | TypeId::Array
            | TypeId::Composite
            | TypeId::Vector => ColumnData::Binary(vec![Vec::new(); len]),
            TypeId::Uuid | TypeId::Interval => ColumnData::FixedBinary16(vec![[0u8; 16]; len]),
            TypeId::Null => ColumnData::Boolean(vec![false; len]),
        }
    }

    /// Pushes a typed default value (zero/empty). Used for null padding.
    #[inline]
    pub fn push_default(&mut self) {
        match self {
            ColumnData::Boolean(v) => v.push(false),
            ColumnData::Int8(v) => v.push(0),
            ColumnData::Int16(v) => v.push(0),
            ColumnData::Int32(v) => v.push(0),
            ColumnData::Int64(v) => v.push(0),
            ColumnData::Int128(v) => v.push(0),
            ColumnData::UInt8(v) => v.push(0),
            ColumnData::UInt16(v) => v.push(0),
            ColumnData::UInt32(v) => v.push(0),
            ColumnData::UInt64(v) => v.push(0),
            ColumnData::Float32(v) => v.push(0.0),
            ColumnData::Float64(v) => v.push(0.0),
            ColumnData::Utf8(v) => v.push(String::new()),
            ColumnData::Binary(v) => v.push(Vec::new()),
            ColumnData::FixedBinary16(v) => v.push([0u8; 16]),
        }
    }

    /// Creates a column filled with the same scalar value repeated `len` times.
    pub fn from_scalar(scalar: &ScalarValue, len: usize) -> Self {
        match scalar {
            ScalarValue::Null => ColumnData::Boolean(vec![false; len]),
            ScalarValue::Boolean(v) => ColumnData::Boolean(vec![*v; len]),
            ScalarValue::Int8(v) => ColumnData::Int8(vec![*v; len]),
            ScalarValue::Int16(v) => ColumnData::Int16(vec![*v; len]),
            ScalarValue::Int32(v) => ColumnData::Int32(vec![*v; len]),
            ScalarValue::Int64(v) => ColumnData::Int64(vec![*v; len]),
            ScalarValue::Int128(v) => ColumnData::Int128(vec![*v; len]),
            ScalarValue::UInt8(v) => ColumnData::UInt8(vec![*v; len]),
            ScalarValue::UInt16(v) => ColumnData::UInt16(vec![*v; len]),
            ScalarValue::UInt32(v) => ColumnData::UInt32(vec![*v; len]),
            ScalarValue::UInt64(v) => ColumnData::UInt64(vec![*v; len]),
            ScalarValue::Float32(v) => ColumnData::Float32(vec![*v; len]),
            ScalarValue::Float64(v) => ColumnData::Float64(vec![*v; len]),
            ScalarValue::Utf8(v) => ColumnData::Utf8(vec![v.clone(); len]),
            ScalarValue::Binary(v) => ColumnData::Binary(vec![v.clone(); len]),
            ScalarValue::FixedBinary16(v) => ColumnData::FixedBinary16(vec![*v; len]),
        }
    }
}

// ---------------------------------------------------------------------------
// Column - data + null bitmap + type metadata
// ---------------------------------------------------------------------------

/// A column of typed values with a null bitmap.
#[derive(Debug, Clone)]
pub struct Column {
    pub data: ColumnData,
    pub nulls: NullBitmap,
    pub type_id: TypeId,
}

impl Column {
    /// Creates a column with no nulls.
    pub fn new(data: ColumnData, type_id: TypeId) -> Self {
        let len = data.len();
        Self {
            data,
            nulls: NullBitmap::none(len),
            type_id,
        }
    }

    /// Creates a column with explicit null bitmap.
    pub fn with_nulls(data: ColumnData, nulls: NullBitmap, type_id: TypeId) -> Self {
        debug_assert_eq!(data.len(), nulls.len());
        Self {
            data,
            nulls,
            type_id,
        }
    }

    /// Creates an all-null column of the given type and length.
    pub fn null_column(type_id: TypeId, len: usize) -> Self {
        Self {
            data: ColumnData::null_fill(type_id, len),
            nulls: NullBitmap::all_null(len),
            type_id,
        }
    }

    /// Appends all rows from another column. No ScalarValue intermediary.
    pub fn extend_from(&mut self, other: &Column) {
        self.data.extend_from(&other.data);
        self.nulls.extend_from(&other.nulls);
    }

    /// Appends a single row from another column. No ScalarValue intermediary.
    #[inline]
    pub fn push_row_from(&mut self, other: &Column, idx: usize) {
        self.data.push_from(&other.data, idx);
        self.nulls.push_from(&other.nulls, idx);
    }

    /// Appends a null value with the correct typed default.
    #[inline]
    pub fn push_null(&mut self) {
        self.data.push_default();
        self.nulls.push(true);
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Selects rows where mask[i] is true.
    pub fn filter(&self, mask: &[bool]) -> Self {
        Self {
            data: self.data.filter(mask),
            nulls: self.nulls.filter(mask),
            type_id: self.type_id,
        }
    }

    /// Reorders rows by indices.
    pub fn take(&self, indices: &[u32]) -> Self {
        Self {
            data: self.data.take(indices),
            nulls: self.nulls.take(indices),
            type_id: self.type_id,
        }
    }

    /// Extracts a contiguous sub-range.
    pub fn slice(&self, offset: usize, len: usize) -> Self {
        Self {
            data: self.data.slice(offset, len),
            nulls: self.nulls.slice(offset, len),
            type_id: self.type_id,
        }
    }

    /// Returns true if the value at the given row is null.
    #[inline]
    pub fn is_null(&self, row: usize) -> bool {
        self.nulls.is_null(row)
    }

    /// Extracts a scalar value at the given row, returning Null if the value is null.
    pub fn get_scalar(&self, row: usize) -> ScalarValue {
        if self.nulls.is_null(row) {
            ScalarValue::Null
        } else {
            self.data.get_scalar(row)
        }
    }

    /// Extracts the boolean value at a row. Panics if not a boolean column.
    #[inline]
    pub fn get_bool(&self, row: usize) -> bool {
        match &self.data {
            ColumnData::Boolean(v) => v[row],
            _ => panic!("get_bool called on non-boolean column"),
        }
    }

    /// Returns the boolean values as a slice. Panics if not a boolean column.
    pub fn as_bools(&self) -> &[bool] {
        match &self.data {
            ColumnData::Boolean(v) => v,
            _ => panic!("as_bools called on non-boolean column"),
        }
    }
}

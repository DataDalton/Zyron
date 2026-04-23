//! Streaming accumulators for windowed aggregation.
//!
//! Provides the StreamAccumulator trait with typed fast paths for update,
//! merge (parallel aggregation), and serialize/deserialize (checkpointing).
//! Implementations: Count, Sum, Avg, Min, Max with typed column access
//! that bypasses ScalarValue on the hot path.

use crate::column::{ScalarValue, StreamColumn, StreamColumnData};
use crate::row_codec::StreamValue;
use zyron_common::{Result, TypeId, ZyronError};

// ---------------------------------------------------------------------------
// StreamAccumulator trait
// ---------------------------------------------------------------------------

/// Trait for streaming aggregation accumulators.
/// Supports typed batch updates, scalar fallback, parallel merge,
/// and checkpoint serialization.
pub trait StreamAccumulator: Send + Sync {
    /// Update accumulator from a typed column at a specific row.
    /// Uses direct typed Vec access without ScalarValue conversion.
    fn update_typed(&mut self, col: &StreamColumn, row: usize);

    /// Update accumulator from a scalar value (fallback path).
    fn update_scalar(&mut self, val: &ScalarValue);

    /// Produce the final aggregated value.
    fn finalize(&self) -> ScalarValue;

    /// Merge another accumulator of the same type (for parallel aggregation).
    fn merge(&mut self, other: &dyn StreamAccumulator);

    /// Serialize accumulator state for checkpointing.
    fn serialize(&self) -> Vec<u8>;

    /// Restore accumulator state from serialized bytes (checkpoint recovery).
    fn deserialize(&mut self, bytes: &[u8]) -> bool;

    /// Reset the accumulator to its initial state.
    fn reset(&mut self);

    /// Clone into a boxed trait object.
    fn clone_box(&self) -> Box<dyn StreamAccumulator>;

    /// Returns the accumulator type name for debugging.
    fn name(&self) -> &'static str;
}

// ---------------------------------------------------------------------------
// CountAccumulator
// ---------------------------------------------------------------------------

/// Counts non-null values.
#[derive(Debug, Clone)]
pub struct CountAccumulator {
    count: i64,
}

impl CountAccumulator {
    pub fn new() -> Self {
        Self { count: 0 }
    }

    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 8 {
            return None;
        }
        let count = i64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]);
        Some(Self { count })
    }
}

impl Default for CountAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamAccumulator for CountAccumulator {
    #[inline]
    fn update_typed(&mut self, col: &StreamColumn, row: usize) {
        if !col.is_null(row) {
            self.count += 1;
        }
    }

    fn update_scalar(&mut self, val: &ScalarValue) {
        if !val.is_null() {
            self.count += 1;
        }
    }

    fn finalize(&self) -> ScalarValue {
        ScalarValue::Int64(self.count)
    }

    fn merge(&mut self, other: &dyn StreamAccumulator) {
        let result = other.finalize();
        if let ScalarValue::Int64(c) = result {
            self.count += c;
        }
    }

    fn serialize(&self) -> Vec<u8> {
        self.count.to_le_bytes().to_vec()
    }

    fn deserialize(&mut self, bytes: &[u8]) -> bool {
        if let Some(restored) = CountAccumulator::from_bytes(bytes) {
            self.count = restored.count;
            true
        } else {
            false
        }
    }

    fn reset(&mut self) {
        self.count = 0;
    }

    fn clone_box(&self) -> Box<dyn StreamAccumulator> {
        Box::new(self.clone())
    }

    fn name(&self) -> &'static str {
        "count"
    }
}

// ---------------------------------------------------------------------------
// SumAccumulator
// ---------------------------------------------------------------------------

/// Sums numeric values. Uses f64 internally to handle mixed numeric types.
#[derive(Debug, Clone)]
pub struct SumAccumulator {
    sum: f64,
    has_value: bool,
}

impl SumAccumulator {
    pub fn new() -> Self {
        Self {
            sum: 0.0,
            has_value: false,
        }
    }

    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 9 {
            return None;
        }
        let sum = f64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]);
        let has_value = bytes[8] != 0;
        Some(Self { sum, has_value })
    }

    #[inline]
    fn add_f64(&mut self, v: f64) {
        self.sum += v;
        self.has_value = true;
    }
}

impl Default for SumAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamAccumulator for SumAccumulator {
    #[inline]
    fn update_typed(&mut self, col: &StreamColumn, row: usize) {
        if col.is_null(row) {
            return;
        }
        match &col.data {
            StreamColumnData::Int64(v) => self.add_f64(v[row] as f64),
            StreamColumnData::Int32(v) => self.add_f64(v[row] as f64),
            StreamColumnData::Int16(v) => self.add_f64(v[row] as f64),
            StreamColumnData::Int8(v) => self.add_f64(v[row] as f64),
            StreamColumnData::Float64(v) => self.add_f64(v[row]),
            StreamColumnData::Float32(v) => self.add_f64(v[row] as f64),
            StreamColumnData::UInt64(v) => self.add_f64(v[row] as f64),
            StreamColumnData::UInt32(v) => self.add_f64(v[row] as f64),
            StreamColumnData::UInt16(v) => self.add_f64(v[row] as f64),
            StreamColumnData::UInt8(v) => self.add_f64(v[row] as f64),
            StreamColumnData::Int128(v) => self.add_f64(v[row] as f64),
            _ => {}
        }
    }

    fn update_scalar(&mut self, val: &ScalarValue) {
        if let Some(f) = val.to_f64() {
            self.add_f64(f);
        }
    }

    fn finalize(&self) -> ScalarValue {
        if self.has_value {
            ScalarValue::Float64(self.sum)
        } else {
            ScalarValue::Null
        }
    }

    fn merge(&mut self, other: &dyn StreamAccumulator) {
        let result = other.finalize();
        if let ScalarValue::Float64(s) = result {
            self.sum += s;
            self.has_value = true;
        }
    }

    fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(9);
        buf.extend_from_slice(&self.sum.to_le_bytes());
        buf.push(if self.has_value { 1 } else { 0 });
        buf
    }

    fn deserialize(&mut self, bytes: &[u8]) -> bool {
        if let Some(restored) = SumAccumulator::from_bytes(bytes) {
            self.sum = restored.sum;
            self.has_value = restored.has_value;
            true
        } else {
            false
        }
    }

    fn reset(&mut self) {
        self.sum = 0.0;
        self.has_value = false;
    }

    fn clone_box(&self) -> Box<dyn StreamAccumulator> {
        Box::new(self.clone())
    }

    fn name(&self) -> &'static str {
        "sum"
    }
}

// ---------------------------------------------------------------------------
// AvgAccumulator
// ---------------------------------------------------------------------------

/// Computes average of numeric values.
#[derive(Debug, Clone)]
pub struct AvgAccumulator {
    sum: f64,
    count: i64,
}

impl AvgAccumulator {
    pub fn new() -> Self {
        Self { sum: 0.0, count: 0 }
    }

    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 16 {
            return None;
        }
        let sum = f64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]);
        let count = i64::from_le_bytes([
            bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
        ]);
        Some(Self { sum, count })
    }
}

impl Default for AvgAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamAccumulator for AvgAccumulator {
    #[inline]
    fn update_typed(&mut self, col: &StreamColumn, row: usize) {
        if col.is_null(row) {
            return;
        }
        let val = match &col.data {
            StreamColumnData::Int64(v) => v[row] as f64,
            StreamColumnData::Int32(v) => v[row] as f64,
            StreamColumnData::Int16(v) => v[row] as f64,
            StreamColumnData::Int8(v) => v[row] as f64,
            StreamColumnData::Float64(v) => v[row],
            StreamColumnData::Float32(v) => v[row] as f64,
            StreamColumnData::UInt64(v) => v[row] as f64,
            StreamColumnData::UInt32(v) => v[row] as f64,
            StreamColumnData::UInt16(v) => v[row] as f64,
            StreamColumnData::UInt8(v) => v[row] as f64,
            StreamColumnData::Int128(v) => v[row] as f64,
            _ => return,
        };
        self.sum += val;
        self.count += 1;
    }

    fn update_scalar(&mut self, val: &ScalarValue) {
        if let Some(f) = val.to_f64() {
            self.sum += f;
            self.count += 1;
        }
    }

    fn finalize(&self) -> ScalarValue {
        if self.count > 0 {
            ScalarValue::Float64(self.sum / self.count as f64)
        } else {
            ScalarValue::Null
        }
    }

    fn merge(&mut self, other: &dyn StreamAccumulator) {
        // Avg merge requires accessing sum and count from the other accumulator.
        // We serialize/deserialize to get the components.
        let bytes = other.serialize();
        if let Some(other_avg) = AvgAccumulator::from_bytes(&bytes) {
            self.sum += other_avg.sum;
            self.count += other_avg.count;
        }
    }

    fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(16);
        buf.extend_from_slice(&self.sum.to_le_bytes());
        buf.extend_from_slice(&self.count.to_le_bytes());
        buf
    }

    fn deserialize(&mut self, bytes: &[u8]) -> bool {
        if let Some(restored) = AvgAccumulator::from_bytes(bytes) {
            self.sum = restored.sum;
            self.count = restored.count;
            true
        } else {
            false
        }
    }

    fn reset(&mut self) {
        self.sum = 0.0;
        self.count = 0;
    }

    fn clone_box(&self) -> Box<dyn StreamAccumulator> {
        Box::new(self.clone())
    }

    fn name(&self) -> &'static str {
        "avg"
    }
}

// ---------------------------------------------------------------------------
// MinAccumulator
// ---------------------------------------------------------------------------

/// Tracks the minimum value. Uses f64 internally.
#[derive(Debug, Clone)]
pub struct MinAccumulator {
    min: f64,
    has_value: bool,
}

impl MinAccumulator {
    pub fn new() -> Self {
        Self {
            min: f64::INFINITY,
            has_value: false,
        }
    }

    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 9 {
            return None;
        }
        let min = f64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]);
        let has_value = bytes[8] != 0;
        Some(Self { min, has_value })
    }

    #[inline]
    fn check_value(&mut self, v: f64) {
        if v < self.min {
            self.min = v;
        }
        self.has_value = true;
    }
}

impl Default for MinAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamAccumulator for MinAccumulator {
    #[inline]
    fn update_typed(&mut self, col: &StreamColumn, row: usize) {
        if col.is_null(row) {
            return;
        }
        match &col.data {
            StreamColumnData::Int64(v) => self.check_value(v[row] as f64),
            StreamColumnData::Int32(v) => self.check_value(v[row] as f64),
            StreamColumnData::Int16(v) => self.check_value(v[row] as f64),
            StreamColumnData::Int8(v) => self.check_value(v[row] as f64),
            StreamColumnData::Float64(v) => self.check_value(v[row]),
            StreamColumnData::Float32(v) => self.check_value(v[row] as f64),
            StreamColumnData::UInt64(v) => self.check_value(v[row] as f64),
            StreamColumnData::UInt32(v) => self.check_value(v[row] as f64),
            StreamColumnData::UInt16(v) => self.check_value(v[row] as f64),
            StreamColumnData::UInt8(v) => self.check_value(v[row] as f64),
            StreamColumnData::Int128(v) => self.check_value(v[row] as f64),
            _ => {}
        }
    }

    fn update_scalar(&mut self, val: &ScalarValue) {
        if let Some(f) = val.to_f64() {
            self.check_value(f);
        }
    }

    fn finalize(&self) -> ScalarValue {
        if self.has_value {
            ScalarValue::Float64(self.min)
        } else {
            ScalarValue::Null
        }
    }

    fn merge(&mut self, other: &dyn StreamAccumulator) {
        let result = other.finalize();
        if let ScalarValue::Float64(v) = result {
            self.check_value(v);
        }
    }

    fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(9);
        buf.extend_from_slice(&self.min.to_le_bytes());
        buf.push(if self.has_value { 1 } else { 0 });
        buf
    }

    fn deserialize(&mut self, bytes: &[u8]) -> bool {
        if let Some(restored) = MinAccumulator::from_bytes(bytes) {
            self.min = restored.min;
            self.has_value = restored.has_value;
            true
        } else {
            false
        }
    }

    fn reset(&mut self) {
        self.min = f64::INFINITY;
        self.has_value = false;
    }

    fn clone_box(&self) -> Box<dyn StreamAccumulator> {
        Box::new(self.clone())
    }

    fn name(&self) -> &'static str {
        "min"
    }
}

// ---------------------------------------------------------------------------
// MaxAccumulator
// ---------------------------------------------------------------------------

/// Tracks the maximum value. Uses f64 internally.
#[derive(Debug, Clone)]
pub struct MaxAccumulator {
    max: f64,
    has_value: bool,
}

impl MaxAccumulator {
    pub fn new() -> Self {
        Self {
            max: f64::NEG_INFINITY,
            has_value: false,
        }
    }

    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 9 {
            return None;
        }
        let max = f64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]);
        let has_value = bytes[8] != 0;
        Some(Self { max, has_value })
    }

    #[inline]
    fn check_value(&mut self, v: f64) {
        if v > self.max {
            self.max = v;
        }
        self.has_value = true;
    }
}

impl Default for MaxAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamAccumulator for MaxAccumulator {
    #[inline]
    fn update_typed(&mut self, col: &StreamColumn, row: usize) {
        if col.is_null(row) {
            return;
        }
        match &col.data {
            StreamColumnData::Int64(v) => self.check_value(v[row] as f64),
            StreamColumnData::Int32(v) => self.check_value(v[row] as f64),
            StreamColumnData::Int16(v) => self.check_value(v[row] as f64),
            StreamColumnData::Int8(v) => self.check_value(v[row] as f64),
            StreamColumnData::Float64(v) => self.check_value(v[row]),
            StreamColumnData::Float32(v) => self.check_value(v[row] as f64),
            StreamColumnData::UInt64(v) => self.check_value(v[row] as f64),
            StreamColumnData::UInt32(v) => self.check_value(v[row] as f64),
            StreamColumnData::UInt16(v) => self.check_value(v[row] as f64),
            StreamColumnData::UInt8(v) => self.check_value(v[row] as f64),
            StreamColumnData::Int128(v) => self.check_value(v[row] as f64),
            _ => {}
        }
    }

    fn update_scalar(&mut self, val: &ScalarValue) {
        if let Some(f) = val.to_f64() {
            self.check_value(f);
        }
    }

    fn finalize(&self) -> ScalarValue {
        if self.has_value {
            ScalarValue::Float64(self.max)
        } else {
            ScalarValue::Null
        }
    }

    fn merge(&mut self, other: &dyn StreamAccumulator) {
        let result = other.finalize();
        if let ScalarValue::Float64(v) = result {
            self.check_value(v);
        }
    }

    fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(9);
        buf.extend_from_slice(&self.max.to_le_bytes());
        buf.push(if self.has_value { 1 } else { 0 });
        buf
    }

    fn deserialize(&mut self, bytes: &[u8]) -> bool {
        if let Some(restored) = MaxAccumulator::from_bytes(bytes) {
            self.max = restored.max;
            self.has_value = restored.has_value;
            true
        } else {
            false
        }
    }

    fn reset(&mut self) {
        self.max = f64::NEG_INFINITY;
        self.has_value = false;
    }

    fn clone_box(&self) -> Box<dyn StreamAccumulator> {
        Box::new(self.clone())
    }

    fn name(&self) -> &'static str {
        "max"
    }
}

// ---------------------------------------------------------------------------
// WindowAccumulator trait and registry for streaming aggregates
// ---------------------------------------------------------------------------
//
// This is the runner-facing accumulator surface. Each window-key pair holds
// an opaque Vec<u8> state. The runner decodes state, calls update with one
// decoded StreamValue, and writes the updated state back to WindowStateStore.
// At window close, finalize produces the StreamValue the sink receives. The
// typed ScalarValue accumulators above serve a separate vectorized operator
// pipeline and are kept intact.

/// Window-aggregate accumulator contract used by the streaming runner.
pub trait WindowAccumulator: Send + Sync {
    /// Initial state bytes for a fresh (window, key) pair.
    fn init(&self) -> Vec<u8>;

    /// Folds one input value into the existing state bytes.
    fn update(&self, state: &mut Vec<u8>, value: &StreamValue) -> Result<()>;

    /// Produces the finalized StreamValue from the state at window close.
    fn finalize(&self, state: &[u8]) -> Result<StreamValue>;

    /// TypeId of the finalized output column.
    fn output_type(&self) -> TypeId;
}

/// Returns the window accumulator for a named aggregate function and its
/// input TypeId. Returns None for unknown names or unsupported type pairs.
/// Names are matched case-insensitively. COUNT(*) is represented by name
/// "COUNT" with a TypeId::Null input to distinguish it from COUNT(col).
pub fn get_accumulator(name: &str, input: TypeId) -> Option<Box<dyn WindowAccumulator>> {
    let upper = name.to_ascii_uppercase();
    match upper.as_str() {
        "COUNT" => {
            if input == TypeId::Null {
                Some(Box::new(CountStarWindowAcc))
            } else {
                Some(Box::new(CountColWindowAcc))
            }
        }
        "SUM" => {
            if is_float(input) {
                Some(Box::new(SumF64WindowAcc))
            } else if is_int(input) {
                Some(Box::new(SumI64WindowAcc))
            } else {
                None
            }
        }
        "AVG" => {
            if is_numeric(input) {
                Some(Box::new(AvgWindowAcc))
            } else {
                None
            }
        }
        "MIN" => {
            if is_numeric(input) {
                Some(Box::new(MinWindowAcc {
                    use_float: is_float(input),
                    out_type: narrow_numeric(input),
                }))
            } else {
                None
            }
        }
        "MAX" => {
            if is_numeric(input) {
                Some(Box::new(MaxWindowAcc {
                    use_float: is_float(input),
                    out_type: narrow_numeric(input),
                }))
            } else {
                None
            }
        }
        "FIRST" => Some(Box::new(FirstWindowAcc { out_type: input })),
        "LAST" => Some(Box::new(LastWindowAcc { out_type: input })),
        _ => None,
    }
}

#[inline]
fn is_int(t: TypeId) -> bool {
    matches!(
        t,
        TypeId::Int8
            | TypeId::Int16
            | TypeId::Int32
            | TypeId::Int64
            | TypeId::UInt8
            | TypeId::UInt16
            | TypeId::UInt32
            | TypeId::UInt64
    )
}

#[inline]
fn is_float(t: TypeId) -> bool {
    matches!(t, TypeId::Float32 | TypeId::Float64)
}

#[inline]
fn is_numeric(t: TypeId) -> bool {
    is_int(t) || is_float(t)
}

#[inline]
fn narrow_numeric(t: TypeId) -> TypeId {
    if is_float(t) {
        TypeId::Float64
    } else {
        TypeId::Int64
    }
}

fn sv_as_i64(v: &StreamValue) -> Option<i64> {
    match v {
        StreamValue::I64(x) => Some(*x),
        StreamValue::I128(x) => i64::try_from(*x).ok(),
        StreamValue::F64(x) => Some(*x as i64),
        StreamValue::Bool(b) => Some(if *b { 1 } else { 0 }),
        _ => None,
    }
}

fn sv_as_f64(v: &StreamValue) -> Option<f64> {
    match v {
        StreamValue::F64(x) => Some(*x),
        StreamValue::I64(x) => Some(*x as f64),
        StreamValue::I128(x) => Some(*x as f64),
        _ => None,
    }
}

// -----------------------------------------------------------------------------
// COUNT(*) accumulator. 8-byte little-endian i64 counter.
// -----------------------------------------------------------------------------

pub struct CountStarWindowAcc;

impl WindowAccumulator for CountStarWindowAcc {
    fn init(&self) -> Vec<u8> {
        0i64.to_le_bytes().to_vec()
    }
    fn update(&self, state: &mut Vec<u8>, _value: &StreamValue) -> Result<()> {
        let mut c = read_i64(state)?;
        c += 1;
        write_i64(state, c);
        Ok(())
    }
    fn finalize(&self, state: &[u8]) -> Result<StreamValue> {
        Ok(StreamValue::I64(read_i64(state)?))
    }
    fn output_type(&self) -> TypeId {
        TypeId::Int64
    }
}

// -----------------------------------------------------------------------------
// COUNT(col) accumulator. Skips nulls.
// -----------------------------------------------------------------------------

pub struct CountColWindowAcc;

impl WindowAccumulator for CountColWindowAcc {
    fn init(&self) -> Vec<u8> {
        0i64.to_le_bytes().to_vec()
    }
    fn update(&self, state: &mut Vec<u8>, value: &StreamValue) -> Result<()> {
        if matches!(value, StreamValue::Null) {
            return Ok(());
        }
        let mut c = read_i64(state)?;
        c += 1;
        write_i64(state, c);
        Ok(())
    }
    fn finalize(&self, state: &[u8]) -> Result<StreamValue> {
        Ok(StreamValue::I64(read_i64(state)?))
    }
    fn output_type(&self) -> TypeId {
        TypeId::Int64
    }
}

// -----------------------------------------------------------------------------
// SUM over integers. i64 state.
// -----------------------------------------------------------------------------

pub struct SumI64WindowAcc;

impl WindowAccumulator for SumI64WindowAcc {
    fn init(&self) -> Vec<u8> {
        0i64.to_le_bytes().to_vec()
    }
    fn update(&self, state: &mut Vec<u8>, value: &StreamValue) -> Result<()> {
        if matches!(value, StreamValue::Null) {
            return Ok(());
        }
        let n = sv_as_i64(value)
            .ok_or_else(|| ZyronError::StreamingError("SUM input is not integral".to_string()))?;
        let mut s = read_i64(state)?;
        s = s.wrapping_add(n);
        write_i64(state, s);
        Ok(())
    }
    fn finalize(&self, state: &[u8]) -> Result<StreamValue> {
        Ok(StreamValue::I64(read_i64(state)?))
    }
    fn output_type(&self) -> TypeId {
        TypeId::Int64
    }
}

// -----------------------------------------------------------------------------
// SUM over floats. f64 state.
// -----------------------------------------------------------------------------

pub struct SumF64WindowAcc;

impl WindowAccumulator for SumF64WindowAcc {
    fn init(&self) -> Vec<u8> {
        0.0f64.to_le_bytes().to_vec()
    }
    fn update(&self, state: &mut Vec<u8>, value: &StreamValue) -> Result<()> {
        if matches!(value, StreamValue::Null) {
            return Ok(());
        }
        let n = sv_as_f64(value)
            .ok_or_else(|| ZyronError::StreamingError("SUM input is not numeric".to_string()))?;
        let mut s = read_f64(state)?;
        s += n;
        write_f64(state, s);
        Ok(())
    }
    fn finalize(&self, state: &[u8]) -> Result<StreamValue> {
        Ok(StreamValue::F64(read_f64(state)?))
    }
    fn output_type(&self) -> TypeId {
        TypeId::Float64
    }
}

// -----------------------------------------------------------------------------
// AVG accumulator. 16 bytes: i64 count + f64 sum.
// -----------------------------------------------------------------------------

pub struct AvgWindowAcc;

impl WindowAccumulator for AvgWindowAcc {
    fn init(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(16);
        v.extend_from_slice(&0i64.to_le_bytes());
        v.extend_from_slice(&0.0f64.to_le_bytes());
        v
    }
    fn update(&self, state: &mut Vec<u8>, value: &StreamValue) -> Result<()> {
        if matches!(value, StreamValue::Null) {
            return Ok(());
        }
        let n = sv_as_f64(value)
            .ok_or_else(|| ZyronError::StreamingError("AVG input is not numeric".to_string()))?;
        if state.len() < 16 {
            return Err(ZyronError::StreamingError(
                "AVG state truncated".to_string(),
            ));
        }
        let mut count = i64::from_le_bytes(state[0..8].try_into().unwrap());
        let mut sum = f64::from_le_bytes(state[8..16].try_into().unwrap());
        count += 1;
        sum += n;
        state[0..8].copy_from_slice(&count.to_le_bytes());
        state[8..16].copy_from_slice(&sum.to_le_bytes());
        Ok(())
    }
    fn finalize(&self, state: &[u8]) -> Result<StreamValue> {
        if state.len() < 16 {
            return Err(ZyronError::StreamingError(
                "AVG state truncated".to_string(),
            ));
        }
        let count = i64::from_le_bytes(state[0..8].try_into().unwrap());
        let sum = f64::from_le_bytes(state[8..16].try_into().unwrap());
        if count == 0 {
            Ok(StreamValue::Null)
        } else {
            Ok(StreamValue::F64(sum / count as f64))
        }
    }
    fn output_type(&self) -> TypeId {
        TypeId::Float64
    }
}

// -----------------------------------------------------------------------------
// MIN accumulator. State is 1 byte has_value flag + f64 or i64 current best.
// -----------------------------------------------------------------------------

pub struct MinWindowAcc {
    use_float: bool,
    out_type: TypeId,
}

impl WindowAccumulator for MinWindowAcc {
    fn init(&self) -> Vec<u8> {
        vec![0u8; 9]
    }
    fn update(&self, state: &mut Vec<u8>, value: &StreamValue) -> Result<()> {
        if matches!(value, StreamValue::Null) {
            return Ok(());
        }
        if state.len() < 9 {
            return Err(ZyronError::StreamingError(
                "MIN state truncated".to_string(),
            ));
        }
        let has = state[0] != 0;
        if self.use_float {
            let incoming = sv_as_f64(value).ok_or_else(|| {
                ZyronError::StreamingError("MIN input is not numeric".to_string())
            })?;
            let current = f64::from_le_bytes(state[1..9].try_into().unwrap());
            let new = if !has || incoming < current {
                incoming
            } else {
                current
            };
            state[0] = 1;
            state[1..9].copy_from_slice(&new.to_le_bytes());
        } else {
            let incoming = sv_as_i64(value).ok_or_else(|| {
                ZyronError::StreamingError("MIN input is not integral".to_string())
            })?;
            let current = i64::from_le_bytes(state[1..9].try_into().unwrap());
            let new = if !has || incoming < current {
                incoming
            } else {
                current
            };
            state[0] = 1;
            state[1..9].copy_from_slice(&new.to_le_bytes());
        }
        Ok(())
    }
    fn finalize(&self, state: &[u8]) -> Result<StreamValue> {
        if state.len() < 9 {
            return Err(ZyronError::StreamingError(
                "MIN state truncated".to_string(),
            ));
        }
        if state[0] == 0 {
            return Ok(StreamValue::Null);
        }
        if self.use_float {
            Ok(StreamValue::F64(f64::from_le_bytes(
                state[1..9].try_into().unwrap(),
            )))
        } else {
            Ok(StreamValue::I64(i64::from_le_bytes(
                state[1..9].try_into().unwrap(),
            )))
        }
    }
    fn output_type(&self) -> TypeId {
        self.out_type
    }
}

// -----------------------------------------------------------------------------
// MAX accumulator. Mirror of MinWindowAcc.
// -----------------------------------------------------------------------------

pub struct MaxWindowAcc {
    use_float: bool,
    out_type: TypeId,
}

impl WindowAccumulator for MaxWindowAcc {
    fn init(&self) -> Vec<u8> {
        vec![0u8; 9]
    }
    fn update(&self, state: &mut Vec<u8>, value: &StreamValue) -> Result<()> {
        if matches!(value, StreamValue::Null) {
            return Ok(());
        }
        if state.len() < 9 {
            return Err(ZyronError::StreamingError(
                "MAX state truncated".to_string(),
            ));
        }
        let has = state[0] != 0;
        if self.use_float {
            let incoming = sv_as_f64(value).ok_or_else(|| {
                ZyronError::StreamingError("MAX input is not numeric".to_string())
            })?;
            let current = f64::from_le_bytes(state[1..9].try_into().unwrap());
            let new = if !has || incoming > current {
                incoming
            } else {
                current
            };
            state[0] = 1;
            state[1..9].copy_from_slice(&new.to_le_bytes());
        } else {
            let incoming = sv_as_i64(value).ok_or_else(|| {
                ZyronError::StreamingError("MAX input is not integral".to_string())
            })?;
            let current = i64::from_le_bytes(state[1..9].try_into().unwrap());
            let new = if !has || incoming > current {
                incoming
            } else {
                current
            };
            state[0] = 1;
            state[1..9].copy_from_slice(&new.to_le_bytes());
        }
        Ok(())
    }
    fn finalize(&self, state: &[u8]) -> Result<StreamValue> {
        if state.len() < 9 {
            return Err(ZyronError::StreamingError(
                "MAX state truncated".to_string(),
            ));
        }
        if state[0] == 0 {
            return Ok(StreamValue::Null);
        }
        if self.use_float {
            Ok(StreamValue::F64(f64::from_le_bytes(
                state[1..9].try_into().unwrap(),
            )))
        } else {
            Ok(StreamValue::I64(i64::from_le_bytes(
                state[1..9].try_into().unwrap(),
            )))
        }
    }
    fn output_type(&self) -> TypeId {
        self.out_type
    }
}

// -----------------------------------------------------------------------------
// FIRST and LAST accumulators. State format:
//   [1 byte has_value] [1 byte tag] [variable payload]
// Tags: 0 = Null, 1 = Bool, 2 = I64, 3 = F64, 4 = I128, 5 = Utf8, 6 = Binary
// -----------------------------------------------------------------------------

pub struct FirstWindowAcc {
    out_type: TypeId,
}

impl WindowAccumulator for FirstWindowAcc {
    fn init(&self) -> Vec<u8> {
        vec![0u8]
    }
    fn update(&self, state: &mut Vec<u8>, value: &StreamValue) -> Result<()> {
        if !state.is_empty() && state[0] != 0 {
            return Ok(());
        }
        let mut buf = Vec::with_capacity(32);
        buf.push(1u8);
        encode_sv(&mut buf, value);
        *state = buf;
        Ok(())
    }
    fn finalize(&self, state: &[u8]) -> Result<StreamValue> {
        if state.is_empty() || state[0] == 0 {
            return Ok(StreamValue::Null);
        }
        decode_sv(&state[1..])
    }
    fn output_type(&self) -> TypeId {
        self.out_type
    }
}

pub struct LastWindowAcc {
    out_type: TypeId,
}

impl WindowAccumulator for LastWindowAcc {
    fn init(&self) -> Vec<u8> {
        vec![0u8]
    }
    fn update(&self, state: &mut Vec<u8>, value: &StreamValue) -> Result<()> {
        let mut buf = Vec::with_capacity(32);
        buf.push(1u8);
        encode_sv(&mut buf, value);
        *state = buf;
        Ok(())
    }
    fn finalize(&self, state: &[u8]) -> Result<StreamValue> {
        if state.is_empty() || state[0] == 0 {
            return Ok(StreamValue::Null);
        }
        decode_sv(&state[1..])
    }
    fn output_type(&self) -> TypeId {
        self.out_type
    }
}

// -----------------------------------------------------------------------------
// Low-level state helpers
// -----------------------------------------------------------------------------

#[inline]
fn read_i64(state: &[u8]) -> Result<i64> {
    if state.len() < 8 {
        return Err(ZyronError::StreamingError(
            "accumulator state too short for i64".to_string(),
        ));
    }
    Ok(i64::from_le_bytes(state[..8].try_into().unwrap()))
}

#[inline]
fn write_i64(state: &mut Vec<u8>, v: i64) {
    if state.len() < 8 {
        state.resize(8, 0);
    }
    state[..8].copy_from_slice(&v.to_le_bytes());
}

#[inline]
fn read_f64(state: &[u8]) -> Result<f64> {
    if state.len() < 8 {
        return Err(ZyronError::StreamingError(
            "accumulator state too short for f64".to_string(),
        ));
    }
    Ok(f64::from_le_bytes(state[..8].try_into().unwrap()))
}

#[inline]
fn write_f64(state: &mut Vec<u8>, v: f64) {
    if state.len() < 8 {
        state.resize(8, 0);
    }
    state[..8].copy_from_slice(&v.to_le_bytes());
}

fn encode_sv(buf: &mut Vec<u8>, v: &StreamValue) {
    match v {
        StreamValue::Null => buf.push(0),
        StreamValue::Bool(b) => {
            buf.push(1);
            buf.push(if *b { 1 } else { 0 });
        }
        StreamValue::I64(x) => {
            buf.push(2);
            buf.extend_from_slice(&x.to_le_bytes());
        }
        StreamValue::F64(x) => {
            buf.push(3);
            buf.extend_from_slice(&x.to_le_bytes());
        }
        StreamValue::I128(x) => {
            buf.push(4);
            buf.extend_from_slice(&x.to_le_bytes());
        }
        StreamValue::Utf8(s) => {
            buf.push(5);
            let bytes = s.as_bytes();
            buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            buf.extend_from_slice(bytes);
        }
        StreamValue::Binary(b) => {
            buf.push(6);
            buf.extend_from_slice(&(b.len() as u32).to_le_bytes());
            buf.extend_from_slice(b);
        }
    }
}

fn decode_sv(bytes: &[u8]) -> Result<StreamValue> {
    if bytes.is_empty() {
        return Err(ZyronError::StreamingError("empty StreamValue".to_string()));
    }
    match bytes[0] {
        0 => Ok(StreamValue::Null),
        1 => {
            if bytes.len() < 2 {
                return Err(ZyronError::StreamingError("Bool truncated".to_string()));
            }
            Ok(StreamValue::Bool(bytes[1] != 0))
        }
        2 => {
            if bytes.len() < 9 {
                return Err(ZyronError::StreamingError("I64 truncated".to_string()));
            }
            Ok(StreamValue::I64(i64::from_le_bytes(
                bytes[1..9].try_into().unwrap(),
            )))
        }
        3 => {
            if bytes.len() < 9 {
                return Err(ZyronError::StreamingError("F64 truncated".to_string()));
            }
            Ok(StreamValue::F64(f64::from_le_bytes(
                bytes[1..9].try_into().unwrap(),
            )))
        }
        4 => {
            if bytes.len() < 17 {
                return Err(ZyronError::StreamingError("I128 truncated".to_string()));
            }
            Ok(StreamValue::I128(i128::from_le_bytes(
                bytes[1..17].try_into().unwrap(),
            )))
        }
        5 => {
            if bytes.len() < 5 {
                return Err(ZyronError::StreamingError("Utf8 truncated".to_string()));
            }
            let len = u32::from_le_bytes(bytes[1..5].try_into().unwrap()) as usize;
            if bytes.len() < 5 + len {
                return Err(ZyronError::StreamingError(
                    "Utf8 payload truncated".to_string(),
                ));
            }
            Ok(StreamValue::Utf8(
                String::from_utf8_lossy(&bytes[5..5 + len]).into_owned(),
            ))
        }
        6 => {
            if bytes.len() < 5 {
                return Err(ZyronError::StreamingError("Binary truncated".to_string()));
            }
            let len = u32::from_le_bytes(bytes[1..5].try_into().unwrap()) as usize;
            if bytes.len() < 5 + len {
                return Err(ZyronError::StreamingError(
                    "Binary payload truncated".to_string(),
                ));
            }
            Ok(StreamValue::Binary(bytes[5..5 + len].to_vec()))
        }
        other => Err(ZyronError::StreamingError(format!(
            "unknown StreamValue tag {other}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod window_accumulator_tests {
    use super::*;

    #[test]
    fn count_star_increments_on_every_row() {
        let acc = get_accumulator("COUNT", TypeId::Null).unwrap();
        let mut s = acc.init();
        for _ in 0..5 {
            acc.update(&mut s, &StreamValue::Null).unwrap();
        }
        let v = acc.finalize(&s).unwrap();
        assert!(matches!(v, StreamValue::I64(5)));
    }

    #[test]
    fn count_col_skips_null() {
        let acc = get_accumulator("COUNT", TypeId::Int64).unwrap();
        let mut s = acc.init();
        acc.update(&mut s, &StreamValue::I64(1)).unwrap();
        acc.update(&mut s, &StreamValue::Null).unwrap();
        acc.update(&mut s, &StreamValue::I64(3)).unwrap();
        assert!(matches!(acc.finalize(&s).unwrap(), StreamValue::I64(2)));
    }

    #[test]
    fn sum_i64_accumulates() {
        let acc = get_accumulator("SUM", TypeId::Int64).unwrap();
        let mut s = acc.init();
        for v in [1i64, 2, 3, 4] {
            acc.update(&mut s, &StreamValue::I64(v)).unwrap();
        }
        assert!(matches!(acc.finalize(&s).unwrap(), StreamValue::I64(10)));
    }

    #[test]
    fn sum_f64_accumulates() {
        let acc = get_accumulator("SUM", TypeId::Float64).unwrap();
        let mut s = acc.init();
        for v in [0.5f64, 1.5, 2.0] {
            acc.update(&mut s, &StreamValue::F64(v)).unwrap();
        }
        match acc.finalize(&s).unwrap() {
            StreamValue::F64(x) => assert!((x - 4.0).abs() < 1e-9),
            other => panic!("expected F64, got {other:?}"),
        }
    }

    #[test]
    fn avg_returns_null_when_empty() {
        let acc = get_accumulator("AVG", TypeId::Int64).unwrap();
        let s = acc.init();
        assert!(matches!(acc.finalize(&s).unwrap(), StreamValue::Null));
    }

    #[test]
    fn avg_divides_sum_by_count() {
        let acc = get_accumulator("AVG", TypeId::Int64).unwrap();
        let mut s = acc.init();
        for v in [10i64, 20, 30] {
            acc.update(&mut s, &StreamValue::I64(v)).unwrap();
        }
        match acc.finalize(&s).unwrap() {
            StreamValue::F64(x) => assert!((x - 20.0).abs() < 1e-9),
            other => panic!("expected F64, got {other:?}"),
        }
    }

    #[test]
    fn min_tracks_smallest() {
        let acc = get_accumulator("MIN", TypeId::Int64).unwrap();
        let mut s = acc.init();
        for v in [30i64, 10, 20] {
            acc.update(&mut s, &StreamValue::I64(v)).unwrap();
        }
        assert!(matches!(acc.finalize(&s).unwrap(), StreamValue::I64(10)));
    }

    #[test]
    fn max_tracks_largest() {
        let acc = get_accumulator("MAX", TypeId::Float64).unwrap();
        let mut s = acc.init();
        for v in [1.0f64, 5.5, 3.0] {
            acc.update(&mut s, &StreamValue::F64(v)).unwrap();
        }
        match acc.finalize(&s).unwrap() {
            StreamValue::F64(x) => assert!((x - 5.5).abs() < 1e-9),
            other => panic!("expected F64, got {other:?}"),
        }
    }

    #[test]
    fn first_retains_earliest_value() {
        let acc = get_accumulator("FIRST", TypeId::Int64).unwrap();
        let mut s = acc.init();
        acc.update(&mut s, &StreamValue::I64(7)).unwrap();
        acc.update(&mut s, &StreamValue::I64(11)).unwrap();
        assert!(matches!(acc.finalize(&s).unwrap(), StreamValue::I64(7)));
    }

    #[test]
    fn last_retains_final_value() {
        let acc = get_accumulator("LAST", TypeId::Varchar).unwrap();
        let mut s = acc.init();
        acc.update(&mut s, &StreamValue::Utf8("a".to_string()))
            .unwrap();
        acc.update(&mut s, &StreamValue::Utf8("z".to_string()))
            .unwrap();
        match acc.finalize(&s).unwrap() {
            StreamValue::Utf8(ref z) => assert_eq!(z, "z"),
            other => panic!("expected Utf8, got {other:?}"),
        }
    }

    #[test]
    fn unknown_aggregate_returns_none() {
        assert!(get_accumulator("WEIRD", TypeId::Int64).is_none());
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::{NullBitmap, StreamColumn, StreamColumnData};

    fn int64_col(values: Vec<i64>) -> StreamColumn {
        StreamColumn::from_data(StreamColumnData::Int64(values))
    }

    fn int64_col_with_null(values: Vec<i64>, null_idx: usize) -> StreamColumn {
        let len = values.len();
        let mut nulls = NullBitmap::new_valid(len);
        nulls.set_null(null_idx);
        StreamColumn::new(StreamColumnData::Int64(values), nulls)
    }

    #[test]
    fn test_count_accumulator() {
        let mut acc = CountAccumulator::new();
        let col = int64_col(vec![10, 20, 30]);
        for i in 0..3 {
            acc.update_typed(&col, i);
        }
        assert_eq!(acc.finalize(), ScalarValue::Int64(3));
    }

    #[test]
    fn test_count_with_nulls() {
        let mut acc = CountAccumulator::new();
        let col = int64_col_with_null(vec![10, 20, 30], 1);
        for i in 0..3 {
            acc.update_typed(&col, i);
        }
        assert_eq!(acc.finalize(), ScalarValue::Int64(2));
    }

    #[test]
    fn test_sum_accumulator() {
        let mut acc = SumAccumulator::new();
        let col = int64_col(vec![10, 20, 30]);
        for i in 0..3 {
            acc.update_typed(&col, i);
        }
        assert_eq!(acc.finalize(), ScalarValue::Float64(60.0));
    }

    #[test]
    fn test_sum_empty() {
        let acc = SumAccumulator::new();
        assert_eq!(acc.finalize(), ScalarValue::Null);
    }

    #[test]
    fn test_avg_accumulator() {
        let mut acc = AvgAccumulator::new();
        let col = int64_col(vec![10, 20, 30]);
        for i in 0..3 {
            acc.update_typed(&col, i);
        }
        assert_eq!(acc.finalize(), ScalarValue::Float64(20.0));
    }

    #[test]
    fn test_avg_empty() {
        let acc = AvgAccumulator::new();
        assert_eq!(acc.finalize(), ScalarValue::Null);
    }

    #[test]
    fn test_min_accumulator() {
        let mut acc = MinAccumulator::new();
        let col = int64_col(vec![30, 10, 20]);
        for i in 0..3 {
            acc.update_typed(&col, i);
        }
        assert_eq!(acc.finalize(), ScalarValue::Float64(10.0));
    }

    #[test]
    fn test_max_accumulator() {
        let mut acc = MaxAccumulator::new();
        let col = int64_col(vec![10, 30, 20]);
        for i in 0..3 {
            acc.update_typed(&col, i);
        }
        assert_eq!(acc.finalize(), ScalarValue::Float64(30.0));
    }

    #[test]
    fn test_merge_count() {
        let mut acc1 = CountAccumulator::new();
        let mut acc2 = CountAccumulator::new();
        let col = int64_col(vec![1, 2, 3]);
        for i in 0..3 {
            acc1.update_typed(&col, i);
            acc2.update_typed(&col, i);
        }
        acc1.merge(&acc2);
        assert_eq!(acc1.finalize(), ScalarValue::Int64(6));
    }

    #[test]
    fn test_serialize_deserialize_count() {
        let mut acc = CountAccumulator::new();
        acc.update_scalar(&ScalarValue::Int64(1));
        acc.update_scalar(&ScalarValue::Int64(2));
        let bytes = acc.serialize();
        let restored = CountAccumulator::from_bytes(&bytes);
        assert!(restored.is_some());
        assert_eq!(restored.map(|a| a.finalize()), Some(ScalarValue::Int64(2)));
    }

    #[test]
    fn test_serialize_deserialize_sum() {
        let mut acc = SumAccumulator::new();
        acc.update_scalar(&ScalarValue::Float64(3.14));
        let bytes = acc.serialize();
        let restored = SumAccumulator::from_bytes(&bytes);
        assert!(restored.is_some());
    }

    #[test]
    fn test_reset() {
        let mut acc = CountAccumulator::new();
        acc.update_scalar(&ScalarValue::Int64(1));
        assert_eq!(acc.finalize(), ScalarValue::Int64(1));
        acc.reset();
        assert_eq!(acc.finalize(), ScalarValue::Int64(0));
    }

    #[test]
    fn test_float64_column() {
        let mut acc = SumAccumulator::new();
        let col = StreamColumn::from_data(StreamColumnData::Float64(vec![1.5, 2.5, 3.0]));
        for i in 0..3 {
            acc.update_typed(&col, i);
        }
        assert_eq!(acc.finalize(), ScalarValue::Float64(7.0));
    }
}

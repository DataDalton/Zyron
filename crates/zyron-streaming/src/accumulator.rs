//! Streaming accumulators for windowed aggregation.
//!
//! Provides the StreamAccumulator trait with typed fast paths for update,
//! merge (parallel aggregation), and serialize/deserialize (checkpointing).
//! Implementations: Count, Sum, Avg, Min, Max with typed column access
//! that bypasses ScalarValue on the hot path.

use crate::column::{ScalarValue, StreamColumn, StreamColumnData};

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

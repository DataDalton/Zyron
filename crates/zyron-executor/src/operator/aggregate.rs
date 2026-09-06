//! Aggregation operators and accumulator infrastructure.
//!
//! Provides hash-based and sort-based aggregation with pluggable accumulators
//! for COUNT, SUM, AVG, MIN, MAX. Uses typed column access to avoid
//! ScalarValue allocation in hot paths.

use std::borrow::Cow;
use std::sync::Arc;

use zyron_catalog::{ColumnId, TableEntry, TableId};
use zyron_common::{Result, TypeId, ZyronError};
use zyron_planner::binder::{BoundExpr, BoundUda};
use zyron_planner::logical::{AggregateExpr, LogicalColumn};

use crate::batch::DataBatch;
use crate::column::{Column, ColumnData, NullBitmap, ScalarValue};
use crate::compute;
use crate::context::ExecutionContext;
use crate::expr::evaluate_borrowed;
use crate::operator::scan::PageRangeScanner;
use crate::operator::{ExecutionBatch, Operator, OperatorResult};

// ---------------------------------------------------------------------------
// Accumulator trait and built-in implementations
// ---------------------------------------------------------------------------

pub(crate) trait Accumulator: std::any::Any + Send {
    fn update(&mut self, value: &ScalarValue);

    /// Typed update directly from a column at a given row, avoiding ScalarValue.
    fn update_typed(&mut self, col: &Column, row: usize) {
        self.update(&col.get_scalar(row));
    }

    /// Folds `n` argument-less rows in one call (`COUNT(*)` over a batch).
    /// The default replays `update` per row; count accumulators override it
    /// with a single add so a 180K-row batch is one operation, not 180K.
    fn add_count(&mut self, n: usize) {
        for _ in 0..n {
            self.update(&ScalarValue::Int64(1));
        }
    }

    fn finalize(&self) -> ScalarValue;

    /// Combines a partial accumulator of the same concrete type into this one
    /// for parallel partial aggregation. The default panics; only accumulators
    /// that return true from supports_parallel_merge override it, and the
    /// parallel path only runs when all accumulators support merging.
    fn merge(&mut self, _other: &dyn Accumulator) {
        unreachable!("merge called on an accumulator without parallel-merge support");
    }

    /// Whether partial states from disjoint partitions can be combined without
    /// changing the result. Order-sensitive accumulators (first/last) and ones
    /// without a defined parallel combine return false, forcing serial
    /// aggregation.
    fn supports_parallel_merge(&self) -> bool {
        false
    }

    /// Fallible counterparts used by the operators so a user-defined aggregate
    /// can surface an evaluation error from its state or final function instead
    /// of losing it. Built-in accumulators inherit the infallible defaults.
    fn update_checked(&mut self, value: &ScalarValue) -> Result<()> {
        self.update(value);
        Ok(())
    }

    fn update_typed_checked(&mut self, col: &Column, row: usize) -> Result<()> {
        self.update_typed(col, row);
        Ok(())
    }

    /// Folds one row of an aggregate taking more than one argument.
    ///
    /// Most aggregates read a single column and the default hands them that
    /// one, so only the accumulators that correlate two columns override
    /// this. A correlation cannot be computed from either column alone, so
    /// the pairing has to survive as far as the accumulator
    fn update_row_checked(&mut self, cols: &[Cow<'_, Column>], row: usize) -> Result<()> {
        match cols.first() {
            Some(col) => self.update_typed_checked(col.as_ref(), row),
            None => self.update_checked(&ScalarValue::Int64(1)),
        }
    }

    fn finalize_checked(&self) -> Result<ScalarValue> {
        Ok(self.finalize())
    }

    /// Returns the accumulator to its freshly built state so one allocation
    /// serves many folds. Window frames refold each row's range through this
    /// instead of building a new boxed accumulator per row
    fn reset(&mut self);
}

/// Downcasts a partial accumulator to its concrete type for merging. Safe
/// because the parallel aggregate builds every partition's accumulators from
/// the same aggregate definition, so paired slots always share a type.
fn merge_peer<T: Accumulator>(other: &dyn Accumulator) -> &T {
    (other as &dyn std::any::Any)
        .downcast_ref::<T>()
        .expect("parallel aggregate merged mismatched accumulator types")
}

struct CountAccumulator {
    count: i64,
}

impl Accumulator for CountAccumulator {
    fn update(&mut self, value: &ScalarValue) {
        if !value.is_null() {
            self.count += 1;
        }
    }
    fn update_typed(&mut self, col: &Column, row: usize) {
        if !col.is_null(row) {
            self.count += 1;
        }
    }
    fn finalize(&self) -> ScalarValue {
        ScalarValue::Int64(self.count)
    }
    fn merge(&mut self, other: &dyn Accumulator) {
        self.count += merge_peer::<CountAccumulator>(other).count;
    }
    fn supports_parallel_merge(&self) -> bool {
        true
    }
    fn reset(&mut self) {
        self.count = 0;
    }
}

struct CountStarAccumulator {
    count: i64,
}

impl Accumulator for CountStarAccumulator {
    fn update(&mut self, _value: &ScalarValue) {
        self.count += 1;
    }
    fn update_typed(&mut self, _col: &Column, _row: usize) {
        self.count += 1;
    }
    fn add_count(&mut self, n: usize) {
        self.count += n as i64;
    }
    fn finalize(&self) -> ScalarValue {
        ScalarValue::Int64(self.count)
    }
    fn merge(&mut self, other: &dyn Accumulator) {
        self.count += merge_peer::<CountStarAccumulator>(other).count;
    }
    fn reset(&mut self) {
        self.count = 0;
    }
    fn supports_parallel_merge(&self) -> bool {
        true
    }
}

/// Sums numeric input. Integer inputs accumulate exactly in i128 so a large
/// integer sum keeps full precision an f64 accumulator would lose past 2^53;
/// floating inputs accumulate in f64. finalize yields the natural scalar (an
/// i128 for integer-only input, an f64 once any floating input is seen);
/// finalize_groups then coerces it to the aggregate's declared output type.
struct SumAccumulator {
    int_sum: i128,
    float_sum: f64,
    saw_float: bool,
    has_value: bool,
    /// The 128-bit accumulator overflowed. Recorded here because the per-row
    /// updates are infallible, and surfaced as an error from
    /// finalize_checked so the statement fails instead of returning a
    /// wrapped sum
    overflowed: bool,
}

impl SumAccumulator {
    fn finalize_inner(&self) -> ScalarValue {
        if !self.has_value {
            ScalarValue::Null
        } else if self.saw_float {
            ScalarValue::Float64(self.float_sum + self.int_sum as f64)
        } else {
            ScalarValue::Int128(self.int_sum)
        }
    }
}

impl Accumulator for SumAccumulator {
    fn update(&mut self, value: &ScalarValue) {
        match value {
            ScalarValue::Float32(_) | ScalarValue::Float64(_) => {
                if let Some(f) = value.to_f64() {
                    self.float_sum += f;
                    self.saw_float = true;
                    self.has_value = true;
                }
            }
            _ => {
                if let Some(i) = value.to_i128() {
                    match self.int_sum.checked_add(i) {
                        Some(s) => self.int_sum = s,
                        None => self.overflowed = true,
                    }
                    self.has_value = true;
                }
            }
        }
    }
    fn update_typed(&mut self, col: &Column, row: usize) {
        if col.is_null(row) {
            return;
        }
        match &col.data {
            ColumnData::Int64(v) => {
                self.int_sum += v[row] as i128;
                self.has_value = true;
            }
            ColumnData::Int32(v) => {
                self.int_sum += v[row] as i128;
                self.has_value = true;
            }
            ColumnData::Float64(v) => {
                self.float_sum += v[row];
                self.saw_float = true;
                self.has_value = true;
            }
            ColumnData::Float32(v) => {
                self.float_sum += v[row] as f64;
                self.saw_float = true;
                self.has_value = true;
            }
            // Other numeric widths (Int8/16/128, UInt*) route through the scalar
            // path, which sorts them into the integer or floating accumulator.
            _ => self.update(&col.get_scalar(row)),
        }
    }
    fn finalize(&self) -> ScalarValue {
        self.finalize_inner()
    }
    fn finalize_checked(&self) -> Result<ScalarValue> {
        if self.overflowed {
            return Err(ZyronError::ExecutionError(
                "SUM overflowed its 128-bit accumulator".to_string(),
            ));
        }
        Ok(self.finalize_inner())
    }
    fn merge(&mut self, other: &dyn Accumulator) {
        let o = merge_peer::<SumAccumulator>(other);
        self.overflowed |= o.overflowed;
        if o.has_value {
            match self.int_sum.checked_add(o.int_sum) {
                Some(s) => self.int_sum = s,
                None => self.overflowed = true,
            }
            self.float_sum += o.float_sum;
            self.saw_float |= o.saw_float;
            self.has_value = true;
        }
    }
    fn supports_parallel_merge(&self) -> bool {
        true
    }
    fn reset(&mut self) {
        self.int_sum = 0;
        self.float_sum = 0.0;
        self.saw_float = false;
        self.has_value = false;
        self.overflowed = false;
    }
}

/// The numeric value at (col, row) as an f64, dividing a decimal's raw
/// scaled integer back onto its value scale. A plain 128-bit integer folds
/// as its numeric value. None for NULL and non-numeric data
fn numeric_value_f64(col: &Column, row: usize) -> Option<f64> {
    if col.is_null(row) {
        return None;
    }
    if let ColumnData::Int128(v) = &col.data {
        let raw = v[row];
        return Some(if col.type_id == zyron_common::TypeId::Decimal {
            let scale = col.fractional_digits.unwrap_or(0);
            raw as f64 / 10f64.powi(scale as i32)
        } else {
            raw as f64
        });
    }
    col.get_scalar(row).to_f64()
}

struct AvgAccumulator {
    sum: f64,
    count: i64,
}

impl Accumulator for AvgAccumulator {
    fn update(&mut self, value: &ScalarValue) {
        if let Some(f) = value.to_f64() {
            self.sum += f;
            self.count += 1;
        }
    }
    fn update_typed(&mut self, col: &Column, row: usize) {
        if col.is_null(row) {
            return;
        }
        match &col.data {
            ColumnData::Int64(v) => {
                self.sum += v[row] as f64;
                self.count += 1;
            }
            ColumnData::Float64(v) => {
                self.sum += v[row];
                self.count += 1;
            }
            ColumnData::Int32(v) => {
                self.sum += v[row] as f64;
                self.count += 1;
            }
            ColumnData::Float32(v) => {
                self.sum += v[row] as f64;
                self.count += 1;
            }
            // Decimals average on the value scale, not the raw scaled int
            ColumnData::Int128(_) => {
                if let Some(x) = numeric_value_f64(col, row) {
                    self.sum += x;
                    self.count += 1;
                }
            }
            _ => self.update(&col.get_scalar(row)),
        }
    }
    fn finalize(&self) -> ScalarValue {
        if self.count > 0 {
            ScalarValue::Float64(self.sum / self.count as f64)
        } else {
            ScalarValue::Null
        }
    }
    fn merge(&mut self, other: &dyn Accumulator) {
        let o = merge_peer::<AvgAccumulator>(other);
        self.sum += o.sum;
        self.count += o.count;
    }
    fn reset(&mut self) {
        self.sum = 0.0;
        self.count = 0;
    }
    fn supports_parallel_merge(&self) -> bool {
        true
    }
}

/// Compares two non-null scalars for MIN/MAX with the float total order,
/// NaN greater than every number and equal to itself. partial_cmp returns
/// None for a NaN pair, which made the running extreme depend on the order
/// values arrived in
fn cmp_scalar_total(a: &ScalarValue, b: &ScalarValue) -> Option<std::cmp::Ordering> {
    match (a, b) {
        (ScalarValue::Float32(x), ScalarValue::Float32(y)) => {
            Some(crate::compute::cmp_f32_total(*x, *y))
        }
        (ScalarValue::Float64(x), ScalarValue::Float64(y)) => {
            Some(crate::compute::cmp_f64_total(*x, *y))
        }
        _ => a.partial_cmp(b),
    }
}

struct MinAccumulator {
    min: Option<ScalarValue>,
}

impl Accumulator for MinAccumulator {
    fn update(&mut self, value: &ScalarValue) {
        if value.is_null() {
            return;
        }
        self.min = Some(match &self.min {
            None => value.clone(),
            Some(current) => {
                if cmp_scalar_total(value, current).is_some_and(|o| o == std::cmp::Ordering::Less) {
                    value.clone()
                } else {
                    current.clone()
                }
            }
        });
    }
    fn finalize(&self) -> ScalarValue {
        self.min.clone().unwrap_or(ScalarValue::Null)
    }
    fn merge(&mut self, other: &dyn Accumulator) {
        if let Some(v) = &merge_peer::<MinAccumulator>(other).min {
            self.update(v);
        }
    }
    fn reset(&mut self) {
        self.min = None;
    }
    fn supports_parallel_merge(&self) -> bool {
        true
    }
}

struct MaxAccumulator {
    max: Option<ScalarValue>,
}

impl Accumulator for MaxAccumulator {
    fn update(&mut self, value: &ScalarValue) {
        if value.is_null() {
            return;
        }
        self.max = Some(match &self.max {
            None => value.clone(),
            Some(current) => {
                if cmp_scalar_total(value, current)
                    .is_some_and(|o| o == std::cmp::Ordering::Greater)
                {
                    value.clone()
                } else {
                    current.clone()
                }
            }
        });
    }
    fn finalize(&self) -> ScalarValue {
        self.max.clone().unwrap_or(ScalarValue::Null)
    }
    fn merge(&mut self, other: &dyn Accumulator) {
        if let Some(v) = &merge_peer::<MaxAccumulator>(other).max {
            self.update(v);
        }
    }
    fn reset(&mut self) {
        self.max = None;
    }
    fn supports_parallel_merge(&self) -> bool {
        true
    }
}

/// Wraps another accumulator to implement `agg(DISTINCT x)`: each input value
/// is folded into the inner accumulator only the first time it is seen, so
/// COUNT/SUM/AVG over distinct values are correct. Not parallel-mergeable
/// (merging two inners would double-count values seen in both partitions), so
/// distinct aggregates run on the serial path.
struct DistinctAccumulator {
    seen: std::collections::HashSet<ScalarValue>,
    inner: Box<dyn Accumulator>,
}

/// True for variants whose clone allocates. These check set membership
/// before cloning so a repeated value never pays an allocation, while
/// fixed-size values keep the single-hash insert whose clone is a copy
fn scalar_owns_heap(value: &ScalarValue) -> bool {
    matches!(value, ScalarValue::Utf8(_) | ScalarValue::Binary(_))
}

impl Accumulator for DistinctAccumulator {
    fn update(&mut self, value: &ScalarValue) {
        if value.is_null() {
            return;
        }
        if scalar_owns_heap(value) {
            if !self.seen.contains(value) {
                self.inner.update(value);
                self.seen.insert(value.clone());
            }
        } else if self.seen.insert(value.clone()) {
            self.inner.update(value);
        }
    }
    fn update_typed(&mut self, col: &Column, row: usize) {
        if col.is_null(row) {
            return;
        }
        // The scalar is only the dedup key. The fold goes through the
        // typed path so the inner accumulator keeps the column context a
        // bare scalar loses, a decimal's scale above all
        let value = col.get_scalar(row);
        if scalar_owns_heap(&value) {
            if !self.seen.contains(&value) {
                self.inner.update_typed(col, row);
                self.seen.insert(value);
            }
        } else if self.seen.insert(value) {
            self.inner.update_typed(col, row);
        }
    }
    fn finalize(&self) -> ScalarValue {
        self.inner.finalize()
    }
    fn update_checked(&mut self, value: &ScalarValue) -> Result<()> {
        if value.is_null() {
            return Ok(());
        }
        if scalar_owns_heap(value) {
            if !self.seen.contains(value) {
                self.inner.update_checked(value)?;
                self.seen.insert(value.clone());
            }
        } else if self.seen.insert(value.clone()) {
            self.inner.update_checked(value)?;
        }
        Ok(())
    }
    fn update_typed_checked(&mut self, col: &Column, row: usize) -> Result<()> {
        if col.is_null(row) {
            return Ok(());
        }
        let value = col.get_scalar(row);
        if scalar_owns_heap(&value) {
            if !self.seen.contains(&value) {
                self.inner.update_typed_checked(col, row)?;
                self.seen.insert(value);
            }
        } else if self.seen.insert(value) {
            self.inner.update_typed_checked(col, row)?;
        }
        Ok(())
    }
    fn finalize_checked(&self) -> Result<ScalarValue> {
        self.inner.finalize_checked()
    }
    fn reset(&mut self) {
        self.seen.clear();
        self.inner.reset();
    }
}

/// Builds a synthetic logical column for a user-defined aggregate's state or
/// input, addressed by the column id the binder used (0 = state, 1 = input).
fn uda_column(column_id: u16, type_id: TypeId) -> LogicalColumn {
    LogicalColumn {
        table_idx: Some(0),
        column_id: ColumnId(column_id),
        name: String::new(),
        type_id,
        nullable: true,
        fractional_digits: None,
    }
}

/// Builds a one-row column holding a single scalar, carrying a null bitmap when
/// the value is NULL so the state or input reaches the bound function as NULL.
fn scalar_to_col(value: &ScalarValue, type_id: TypeId) -> Column {
    if value.is_null() {
        Column::null_column(type_id, 1)
    } else {
        Column::new(ColumnData::from_scalar(value, 1), type_id)
    }
}

/// Reads a one-row column's value, returning NULL when the row is null.
fn col_scalar(col: &Column, row: usize) -> ScalarValue {
    if col.is_null(row) {
        ScalarValue::Null
    } else {
        col.data.get_scalar(row)
    }
}

/// Evaluates a bound constant expression (no input columns) to a scalar.
fn eval_const(expr: &BoundExpr) -> Result<ScalarValue> {
    let batch = DataBatch {
        columns: Vec::new(),
        num_rows: 1,
        resolved: Vec::new(),
    };
    let col = crate::expr::evaluate(expr, &batch, &[], &[])?;
    Ok(col_scalar(&col, 0))
}

/// Accumulator for a user-defined aggregate. Holds the running state and folds
/// each input value by evaluating the bound state-transition function over a
/// one-row (state, input) batch. The optional final function runs once at
/// finalize. NULL inputs are skipped, matching built-in aggregate semantics.
/// The first evaluation error is retained and surfaced through the fallible
/// accumulator methods so a query fails rather than returning a wrong result.
struct UdaAccumulator {
    sfunc: BoundExpr,
    finalfunc: Option<BoundExpr>,
    state_type: TypeId,
    input_type: TypeId,
    state: ScalarValue,
    sfunc_schema: Vec<LogicalColumn>,
    final_schema: Vec<LogicalColumn>,
    error: Option<String>,
    /// The evaluated init state and any init evaluation failure, kept so
    /// reset restores exactly the freshly built condition
    initial_state: ScalarValue,
    initial_error: Option<String>,
}

impl UdaAccumulator {
    fn new(uda: &BoundUda) -> Self {
        let state_type = uda.state_type;
        let input_type = uda.input_types.first().copied().unwrap_or(TypeId::Null);
        let (state, error) = match &uda.init {
            Some(init_expr) => match eval_const(init_expr) {
                Ok(v) => (v, None),
                Err(e) => (ScalarValue::Null, Some(e.to_string())),
            },
            None => (ScalarValue::Null, None),
        };
        Self {
            sfunc: uda.sfunc.clone(),
            finalfunc: uda.finalfunc.clone(),
            state_type,
            input_type,
            initial_state: state.clone(),
            initial_error: error.clone(),
            state,
            sfunc_schema: vec![uda_column(0, state_type), uda_column(1, input_type)],
            final_schema: vec![uda_column(0, state_type)],
            error,
        }
    }

    fn fold(&mut self, value: &ScalarValue) -> Result<()> {
        if let Some(e) = &self.error {
            return Err(ZyronError::ExecutionError(e.clone()));
        }
        let state_col = scalar_to_col(&self.state, self.state_type);
        let input_col = Column::new(ColumnData::from_scalar(value, 1), self.input_type);
        let batch = DataBatch {
            columns: vec![state_col, input_col],
            num_rows: 1,
            resolved: Vec::new(),
        };
        match crate::expr::evaluate(&self.sfunc, &batch, &self.sfunc_schema, &[]) {
            Ok(col) => {
                self.state = col_scalar(&col, 0);
                Ok(())
            }
            Err(e) => {
                self.error = Some(e.to_string());
                Err(e)
            }
        }
    }

    fn finalize_inner(&self) -> Result<ScalarValue> {
        if let Some(e) = &self.error {
            return Err(ZyronError::ExecutionError(e.clone()));
        }
        match &self.finalfunc {
            Some(ff) => {
                let state_col = scalar_to_col(&self.state, self.state_type);
                let batch = DataBatch {
                    columns: vec![state_col],
                    num_rows: 1,
                    resolved: Vec::new(),
                };
                let col = crate::expr::evaluate(ff, &batch, &self.final_schema, &[])?;
                Ok(col_scalar(&col, 0))
            }
            None => Ok(self.state.clone()),
        }
    }
}

impl Accumulator for UdaAccumulator {
    fn update(&mut self, value: &ScalarValue) {
        if value.is_null() {
            return;
        }
        let _ = self.fold(value);
    }
    fn update_typed(&mut self, col: &Column, row: usize) {
        if col.is_null(row) {
            return;
        }
        let _ = self.fold(&col.data.get_scalar(row));
    }
    fn finalize(&self) -> ScalarValue {
        self.finalize_inner().unwrap_or(ScalarValue::Null)
    }
    fn update_checked(&mut self, value: &ScalarValue) -> Result<()> {
        if value.is_null() {
            return Ok(());
        }
        self.fold(value)
    }
    fn update_typed_checked(&mut self, col: &Column, row: usize) -> Result<()> {
        if col.is_null(row) {
            return Ok(());
        }
        self.fold(&col.data.get_scalar(row))
    }
    fn finalize_checked(&self) -> Result<ScalarValue> {
        self.finalize_inner()
    }
    fn reset(&mut self) {
        self.state = self.initial_state.clone();
        self.error = self.initial_error.clone();
    }
}

fn create_accumulator(agg: &AggregateExpr) -> Box<dyn Accumulator> {
    let inner: Box<dyn Accumulator> = match &agg.uda {
        Some(uda) => Box::new(UdaAccumulator::new(uda)),
        None => build_accumulator(&agg.function_name, agg.args.len()),
    };
    // DISTINCT only applies to aggregates over an argument; COUNT(*) has no
    // argument to deduplicate.
    if agg.distinct && !agg.args.is_empty() {
        Box::new(DistinctAccumulator {
            seen: std::collections::HashSet::new(),
            inner,
        })
    } else {
        inner
    }
}

/// Which sketch a merging accumulator folds, and the primitive that folds
/// two of them
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SketchKind {
    HyperLogLog,
    Bloom,
    TDigest,
    CountMinSketch,
}

impl SketchKind {
    fn merge_bytes(self, a: &[u8], b: &[u8]) -> Result<Vec<u8>> {
        match self {
            SketchKind::HyperLogLog => zyron_types::probabilistic::hll_merge(a, b),
            SketchKind::Bloom => zyron_types::probabilistic::bloom_merge(a, b),
            SketchKind::TDigest => zyron_types::probabilistic::tdigest_merge(a, b),
            SketchKind::CountMinSketch => zyron_types::probabilistic::cms_merge(a, b),
        }
    }

    fn name(self) -> &'static str {
        match self {
            SketchKind::HyperLogLog => "hll_merge_agg",
            SketchKind::Bloom => "bloom_merge_agg",
            SketchKind::TDigest => "tdigest_merge_agg",
            SketchKind::CountMinSketch => "cms_merge_agg",
        }
    }

    fn from_name(name: &str) -> Option<Self> {
        match name {
            "hll_merge_agg" => Some(SketchKind::HyperLogLog),
            "bloom_merge_agg" => Some(SketchKind::Bloom),
            "tdigest_merge_agg" => Some(SketchKind::TDigest),
            "cms_merge_agg" => Some(SketchKind::CountMinSketch),
            _ => None,
        }
    }
}

/// Folds a group's sketches into one.
///
/// A sketch exists so a partial result can be combined without revisiting
/// the rows behind it, which is what a rollup over pre built sketches needs.
/// This keeps one merged image and folds each incoming value into it, so its
/// memory is the size of a single sketch however many rows the group holds,
/// and merging is associative, so partitions combine without changing the
/// answer.
///
/// A value that is not a sketch of this kind fails the aggregate rather than
/// being skipped. A merge that quietly dropped an unreadable input would
/// report a smaller population than the data holds, and a sketch nobody can
/// tell is wrong is worse than an error.
pub(crate) struct SketchMergeAccumulator {
    kind: SketchKind,
    merged: Option<Vec<u8>>,
    error: Option<String>,
}

impl SketchMergeAccumulator {
    pub(crate) fn new(kind: SketchKind) -> Self {
        Self {
            kind,
            merged: None,
            error: None,
        }
    }

    fn fold(&mut self, bytes: &[u8]) {
        match self.merged.take() {
            None => self.merged = Some(bytes.to_vec()),
            Some(current) => match self.kind.merge_bytes(&current, bytes) {
                Ok(merged) => self.merged = Some(merged),
                Err(e) => self.error = Some(format!("{}: {e}", self.kind.name())),
            },
        }
    }
}

impl Accumulator for SketchMergeAccumulator {
    fn update(&mut self, value: &ScalarValue) {
        if self.error.is_some() {
            return;
        }
        match value {
            ScalarValue::Null => {}
            ScalarValue::Binary(b) => {
                let bytes = b.clone();
                self.fold(&bytes);
            }
            other => {
                self.error = Some(format!(
                    "{} takes a sketch value, got {other:?}",
                    self.kind.name()
                ));
            }
        }
    }

    fn finalize(&self) -> ScalarValue {
        match &self.merged {
            Some(bytes) if self.error.is_none() => ScalarValue::Binary(bytes.clone()),
            _ => ScalarValue::Null,
        }
    }

    fn finalize_checked(&self) -> Result<ScalarValue> {
        match &self.error {
            Some(message) => Err(ZyronError::ExecutionError(message.clone())),
            None => Ok(self.finalize()),
        }
    }

    fn merge(&mut self, other: &dyn Accumulator) {
        let peer: &SketchMergeAccumulator = merge_peer(other);
        if self.error.is_some() {
            return;
        }
        if let Some(message) = &peer.error {
            self.error = Some(message.clone());
            return;
        }
        if let Some(bytes) = &peer.merged {
            let bytes = bytes.clone();
            self.fold(&bytes);
        }
    }

    fn supports_parallel_merge(&self) -> bool {
        true
    }

    fn reset(&mut self) {
        self.merged = None;
        self.error = None;
    }
}

/// A row's numeric value, or None when it is null or not a number
fn numeric_at(col: &Column, row: usize) -> Option<f64> {
    if col.is_null(row) {
        return None;
    }
    match col.get_scalar(row) {
        ScalarValue::Int8(v) => Some(v as f64),
        ScalarValue::Int16(v) => Some(v as f64),
        ScalarValue::Int32(v) => Some(v as f64),
        ScalarValue::Int64(v) => Some(v as f64),
        ScalarValue::Int128(v) => Some(v as f64),
        ScalarValue::UInt8(v) => Some(v as f64),
        ScalarValue::UInt16(v) => Some(v as f64),
        ScalarValue::UInt32(v) => Some(v as f64),
        ScalarValue::UInt64(v) => Some(v as f64),
        ScalarValue::Float32(v) => Some(v as f64),
        ScalarValue::Float64(v) => Some(v),
        _ => None,
    }
}

/// Which statistic a paired accumulator reports from the same sums
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PairStatistic {
    Correlation,
    Covariance,
}

/// Correlation and covariance over two columns.
///
/// Both read the same running sums, so they share one accumulator and
/// differ only in what they report at the end. The sums are kept in the
/// shifted form, taking the first pair as the origin, because summing raw
/// products of large values loses the precision the difference of means
/// depends on, and a shift costs one subtraction per row.
///
/// A row is folded only when both values are present. A pair is what the
/// statistic is defined over, so a row holding one half of one says
/// nothing and is skipped rather than counted as a zero.
pub(crate) struct PairAccumulator {
    statistic: PairStatistic,
    count: u64,
    origin: Option<(f64, f64)>,
    sum_x: f64,
    sum_y: f64,
    sum_xx: f64,
    sum_yy: f64,
    sum_xy: f64,
}

impl PairAccumulator {
    pub(crate) fn new(statistic: PairStatistic) -> Self {
        Self {
            statistic,
            count: 0,
            origin: None,
            sum_x: 0.0,
            sum_y: 0.0,
            sum_xx: 0.0,
            sum_yy: 0.0,
            sum_xy: 0.0,
        }
    }

    fn fold(&mut self, x: f64, y: f64) {
        let (ox, oy) = *self.origin.get_or_insert((x, y));
        let dx = x - ox;
        let dy = y - oy;
        self.count += 1;
        self.sum_x += dx;
        self.sum_y += dy;
        self.sum_xx += dx * dx;
        self.sum_yy += dy * dy;
        self.sum_xy += dx * dy;
    }

    /// Sum of products of deviations from the mean, in the shifted frame
    fn centered(&self) -> (f64, f64, f64) {
        let n = self.count as f64;
        let sxx = self.sum_xx - self.sum_x * self.sum_x / n;
        let syy = self.sum_yy - self.sum_y * self.sum_y / n;
        let sxy = self.sum_xy - self.sum_x * self.sum_y / n;
        (sxx, syy, sxy)
    }
}

impl Accumulator for PairAccumulator {
    fn update(&mut self, _value: &ScalarValue) {
        // A single value is half a pair and says nothing on its own. The
        // operator drives this through update_row_checked instead
    }

    fn update_row_checked(&mut self, cols: &[Cow<'_, Column>], row: usize) -> Result<()> {
        let (Some(x), Some(y)) = (cols.first(), cols.get(1)) else {
            return Err(ZyronError::ExecutionError(
                "correlation and covariance take two arguments".to_string(),
            ));
        };
        if let (Some(xv), Some(yv)) = (numeric_at(x.as_ref(), row), numeric_at(y.as_ref(), row)) {
            self.fold(xv, yv);
        }
        Ok(())
    }

    fn finalize(&self) -> ScalarValue {
        match self.statistic {
            // A sample covariance needs two pairs to have a denominator
            PairStatistic::Covariance => {
                if self.count < 2 {
                    return ScalarValue::Null;
                }
                let (_, _, sxy) = self.centered();
                ScalarValue::Float64(sxy / (self.count as f64 - 1.0))
            }
            // A correlation is undefined when either side never varies,
            // because the ratio divides by that variation
            PairStatistic::Correlation => {
                if self.count < 2 {
                    return ScalarValue::Null;
                }
                let (sxx, syy, sxy) = self.centered();
                if sxx <= 0.0 || syy <= 0.0 {
                    return ScalarValue::Null;
                }
                ScalarValue::Float64(sxy / (sxx * syy).sqrt())
            }
        }
    }

    fn reset(&mut self) {
        self.count = 0;
        self.origin = None;
        self.sum_x = 0.0;
        self.sum_y = 0.0;
        self.sum_xx = 0.0;
        self.sum_yy = 0.0;
        self.sum_xy = 0.0;
    }
}

/// Time weighted average of a value over the interval each reading covers.
///
/// A reading taken every second and one taken every hour do not carry the
/// same weight in an average over time, so each value is weighted by the
/// gap to the next reading. The last reading closes no interval and so
/// contributes no weight, which is what keeps the result the average over
/// the observed span rather than over an interval that has not finished.
///
/// Readings are folded in the order the operator delivers them, so an
/// ORDER BY on the time column is what makes the weights the real gaps.
pub(crate) struct TimeWeightAccumulator {
    previous: Option<(f64, f64)>,
    weighted: f64,
    span: f64,
}

impl TimeWeightAccumulator {
    pub(crate) fn new() -> Self {
        Self {
            previous: None,
            weighted: 0.0,
            span: 0.0,
        }
    }
}

impl Accumulator for TimeWeightAccumulator {
    fn update(&mut self, _value: &ScalarValue) {
        // A value without its timestamp carries no weight
    }

    fn update_row_checked(&mut self, cols: &[Cow<'_, Column>], row: usize) -> Result<()> {
        let (Some(value), Some(time)) = (cols.first(), cols.get(1)) else {
            return Err(ZyronError::ExecutionError(
                "time_weight takes a value and a time".to_string(),
            ));
        };
        let (Some(v), Some(t)) = (
            numeric_at(value.as_ref(), row),
            numeric_at(time.as_ref(), row),
        ) else {
            return Ok(());
        };
        if let Some((pv, pt)) = self.previous {
            let width = t - pt;
            // Readings out of order would subtract span from the average,
            // so a step backwards contributes nothing rather than a
            // negative weight
            if width > 0.0 {
                self.weighted += pv * width;
                self.span += width;
            }
        }
        self.previous = Some((v, t));
        Ok(())
    }

    fn finalize(&self) -> ScalarValue {
        if self.span <= 0.0 {
            // One reading covers no interval, so its value is the answer
            return match self.previous {
                Some((v, _)) => ScalarValue::Float64(v),
                None => ScalarValue::Null,
            };
        }
        ScalarValue::Float64(self.weighted / self.span)
    }

    fn reset(&mut self) {
        self.previous = None;
        self.weighted = 0.0;
        self.span = 0.0;
    }
}

pub(crate) fn build_accumulator(name: &str, args_count: usize) -> Box<dyn Accumulator> {
    match name.to_lowercase().as_str() {
        "count" => {
            if args_count == 0 {
                Box::new(CountStarAccumulator { count: 0 })
            } else {
                Box::new(CountAccumulator { count: 0 })
            }
        }
        "sum" => Box::new(SumAccumulator {
            int_sum: 0,
            float_sum: 0.0,
            saw_float: false,
            has_value: false,
            overflowed: false,
        }),
        "avg" => Box::new(AvgAccumulator { sum: 0.0, count: 0 }),
        "min" => Box::new(MinAccumulator { min: None }),
        "max" => Box::new(MaxAccumulator { max: None }),
        "first" => Box::new(FirstAccumulator { value: None }),
        "last" => Box::new(LastAccumulator { value: None }),
        "stddev_agg" | "stddev" | "stddev_sample_agg" => Box::new(StddevAccumulator {
            count: 0,
            mean: 0.0,
            m2: 0.0,
        }),
        "variance_agg" | "variance" => Box::new(VarianceAccumulator {
            count: 0,
            mean: 0.0,
            m2: 0.0,
        }),
        "correlation_agg" => Box::new(PairAccumulator::new(PairStatistic::Correlation)),
        "covariance_agg" => Box::new(PairAccumulator::new(PairStatistic::Covariance)),
        "time_weight" => Box::new(TimeWeightAccumulator::new()),
        other => match SketchKind::from_name(other) {
            Some(kind) => Box::new(SketchMergeAccumulator::new(kind)),
            None => Box::new(CountAccumulator { count: 0 }),
        },
    }
}

/// Whether `create_accumulator` has a real implementation for this function.
/// MUST list the same set as the matched arms above; the catch-all there
/// returns a COUNT accumulator, so callers validate up front and error rather
/// than silently computing a COUNT for an unimplemented aggregate.
pub fn is_supported_aggregate(name: &str) -> bool {
    matches!(
        name.to_lowercase().as_str(),
        "count"
            | "sum"
            | "avg"
            | "min"
            | "max"
            | "first"
            | "last"
            | "stddev_agg"
            | "stddev"
            | "stddev_sample_agg"
            | "variance_agg"
            | "variance"
            | "hll_merge_agg"
            | "bloom_merge_agg"
            | "tdigest_merge_agg"
            | "cms_merge_agg"
            | "correlation_agg"
            | "covariance_agg"
            | "time_weight"
    )
}

/// Validates every aggregate has a real implementation, returning an error
/// instead of letting an unimplemented function silently degrade to COUNT.
fn validate_aggregates(aggregates: &[AggregateExpr]) -> Result<()> {
    for agg in aggregates {
        // A user-defined aggregate carries its bound state/final functions, so
        // it is always executable regardless of the built-in name set.
        if agg.uda.is_some() {
            continue;
        }
        if !is_supported_aggregate(&agg.function_name) {
            return Err(ZyronError::ExecutionError(format!(
                "aggregate function '{}' is not implemented",
                agg.function_name
            )));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Extended accumulators (first/last, approximate, temporal)
// ---------------------------------------------------------------------------

/// First value seen (in input order). For two-arg first(value, time),
/// the operator currently passes only the first arg.
struct FirstAccumulator {
    value: Option<ScalarValue>,
}

impl Accumulator for FirstAccumulator {
    fn update(&mut self, value: &ScalarValue) {
        if self.value.is_none() && !value.is_null() {
            self.value = Some(value.clone());
        }
    }
    fn finalize(&self) -> ScalarValue {
        self.value.clone().unwrap_or(ScalarValue::Null)
    }
    fn reset(&mut self) {
        self.value = None;
    }
}

/// Last value seen (in input order).
struct LastAccumulator {
    value: Option<ScalarValue>,
}

impl Accumulator for LastAccumulator {
    fn update(&mut self, value: &ScalarValue) {
        if !value.is_null() {
            self.value = Some(value.clone());
        }
    }
    fn finalize(&self) -> ScalarValue {
        self.value.clone().unwrap_or(ScalarValue::Null)
    }
    fn reset(&mut self) {
        self.value = None;
    }
}

/// Sample standard deviation via Welford's online algorithm.
struct StddevAccumulator {
    count: u64,
    mean: f64,
    m2: f64,
}

impl StddevAccumulator {
    fn accept(&mut self, x: f64) {
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }
}

impl Accumulator for StddevAccumulator {
    fn update(&mut self, value: &ScalarValue) {
        let x = match value {
            ScalarValue::Float64(v) => *v,
            ScalarValue::Float32(v) => *v as f64,
            ScalarValue::Int64(v) => *v as f64,
            ScalarValue::Int32(v) => *v as f64,
            ScalarValue::Int16(v) => *v as f64,
            ScalarValue::Int8(v) => *v as f64,
            ScalarValue::Int128(v) => *v as f64,
            _ => return,
        };
        self.accept(x);
    }
    fn update_typed(&mut self, col: &Column, row: usize) {
        // A decimal folds on its value scale, which only the column knows
        if let ColumnData::Int128(_) = &col.data {
            if let Some(x) = numeric_value_f64(col, row) {
                self.accept(x);
            }
            return;
        }
        self.update(&col.get_scalar(row));
    }
    fn finalize(&self) -> ScalarValue {
        if self.count < 2 {
            return ScalarValue::Null;
        }
        let variance = self.m2 / (self.count - 1) as f64;
        ScalarValue::Float64(variance.sqrt())
    }
    fn reset(&mut self) {
        self.count = 0;
        self.mean = 0.0;
        self.m2 = 0.0;
    }
}

/// Sample variance via Welford's online algorithm.
struct VarianceAccumulator {
    count: u64,
    mean: f64,
    m2: f64,
}

impl VarianceAccumulator {
    fn accept(&mut self, x: f64) {
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }
}

impl Accumulator for VarianceAccumulator {
    fn update(&mut self, value: &ScalarValue) {
        let x = match value {
            ScalarValue::Float64(v) => *v,
            ScalarValue::Float32(v) => *v as f64,
            ScalarValue::Int64(v) => *v as f64,
            ScalarValue::Int32(v) => *v as f64,
            ScalarValue::Int16(v) => *v as f64,
            ScalarValue::Int8(v) => *v as f64,
            ScalarValue::Int128(v) => *v as f64,
            _ => return,
        };
        self.accept(x);
    }
    fn update_typed(&mut self, col: &Column, row: usize) {
        // A decimal folds on its value scale, which only the column knows
        if let ColumnData::Int128(_) = &col.data {
            if let Some(x) = numeric_value_f64(col, row) {
                self.accept(x);
            }
            return;
        }
        self.update(&col.get_scalar(row));
    }
    fn finalize(&self) -> ScalarValue {
        if self.count < 2 {
            return ScalarValue::Null;
        }
        ScalarValue::Float64(self.m2 / (self.count - 1) as f64)
    }
    fn reset(&mut self) {
        self.count = 0;
        self.mean = 0.0;
        self.m2 = 0.0;
    }
}

// ---------------------------------------------------------------------------
// HashAggregateOperator
// ---------------------------------------------------------------------------

/// Hash-based aggregation. Drains all input, groups rows by key,
/// feeds argument values into per-group accumulators, and emits
/// the finalized results. Uses typed hashing to avoid per-row
/// Vec<ScalarValue> allocation for group keys.
pub struct HashAggregateOperator {
    child: Box<dyn Operator>,
    group_by: Vec<BoundExpr>,
    aggregates: Vec<AggregateExpr>,
    input_schema: Vec<LogicalColumn>,
    output_schema: Vec<LogicalColumn>,
    finished: bool,
    result: Option<DataBatch>,
    output_cursor: usize,
    /// Query memory budget the accumulated group state reserves against,
    /// approximated by input batch size. None runs unbudgeted.
    memory_budget: Option<Arc<crate::context::QueryMemoryBudget>>,
    /// Where rows for groups that did not fit are put. None means the
    /// aggregate fails at the budget the way it did before spilling existed
    spill: Option<Arc<crate::spill::SpillDirectory>>,
    /// Bytes of group state the aggregate may hold. Zero means never spill
    spill_threshold_bytes: u64,
    /// Partitions of rows still to be aggregated, taken from the back so a
    /// partition split again is finished before its siblings are started
    pending: Vec<PendingPartition>,
    /// True once the input has been read and the resident groups emitted
    input_drained: bool,
}

/// One partition of input rows waiting to be aggregated on its own.
struct PendingPartition {
    reader: crate::spill::SpillReader,
    /// How many times the rows in it have already been re-partitioned, which
    /// decides the hash seed the next split uses
    depth: u32,
}

impl HashAggregateOperator {
    pub fn new(
        child: Box<dyn Operator>,
        group_by: Vec<BoundExpr>,
        aggregates: Vec<AggregateExpr>,
        input_schema: Vec<LogicalColumn>,
        output_schema: Vec<LogicalColumn>,
    ) -> Self {
        Self {
            child,
            group_by,
            aggregates,
            input_schema,
            output_schema,
            finished: false,
            result: None,
            output_cursor: 0,
            memory_budget: None,
            spill: None,
            spill_threshold_bytes: 0,
            pending: Vec::new(),
            input_drained: false,
        }
    }

    /// Gives the aggregate somewhere to put the groups that do not fit.
    ///
    /// The threshold is what the query may hold, which is the point the
    /// aggregate used to fail at.
    pub fn set_spill(
        &mut self,
        directory: Option<Arc<crate::spill::SpillDirectory>>,
        threshold_bytes: u64,
    ) {
        self.spill = directory;
        self.spill_threshold_bytes = threshold_bytes;
    }

    /// Attaches the query memory budget. Set by the operator builder from
    /// the execution context.
    pub fn set_memory_budget(&mut self, budget: Option<Arc<crate::context::QueryMemoryBudget>>) {
        self.memory_budget = budget;
    }

    /// A place to route the groups that will not fit, or None when this
    /// aggregate has nowhere to put them.
    fn new_router(&self) -> Option<crate::operator::grace::PartitionWriterSet> {
        let directory = self.spill.as_ref()?;
        if self.spill_threshold_bytes == 0 {
            return None;
        }
        Some(crate::operator::grace::PartitionWriterSet::new(
            Arc::clone(directory),
            self.spill_threshold_bytes,
        ))
    }

    /// Closes a router and queues whatever it wrote.
    fn queue_partitions(
        &mut self,
        router: crate::operator::grace::PartitionWriterSet,
        depth: u32,
    ) -> Result<()> {
        if router.routed() == 0 {
            return Ok(());
        }
        if depth == 1 {
            crate::spill::SpillStats::global()
                .aggregates_spilled
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        for side in router.finish()? {
            if let Some(reader) = side.into_reader() {
                self.pending.push(PendingPartition { reader, depth });
            }
        }
        Ok(())
    }

    /// Aggregates one partition on its own.
    ///
    /// Its groups are disjoint from every other partition's and from the
    /// resident table's, so this is a whole aggregation over a subset of the
    /// rows and its output needs nothing done to it. If it does not fit
    /// either, it splits again under a different seed, which is the same
    /// mechanism one level down.
    fn aggregate_partition(&mut self, partition: PendingPartition) -> Result<Option<DataBatch>> {
        let mut state = GroupAccumulatorState::new(&self.aggregates);
        let mut router = if partition.depth < MAX_AGGREGATE_SPILL_DEPTH {
            self.new_router()
        } else {
            // Deep enough that splitting again is not what is wrong. What is
            // left is aggregated in memory, and the budget answers for it
            None
        };
        let seed = zyron_common::checksum::splitmix64(partition.depth as u64 + 1);
        let mut reader = partition.reader;
        while let Some(batch) = reader.read_batch()? {
            state.ingest_bounded(
                &batch,
                &self.group_by,
                &self.aggregates,
                &self.input_schema,
                router.as_mut(),
                seed,
                self.spill_threshold_bytes,
            )?;
        }
        // The file goes before its children are closed, so the rows it held
        // are never counted against the spill quota twice
        drop(reader);
        if let Some(router) = router.take() {
            self.queue_partitions(router, partition.depth + 1)?;
        }
        if state.num_groups == 0 {
            return Ok(None);
        }
        Ok(Some(finalize_groups(
            &state,
            self.group_by.len(),
            &self.output_schema,
        )?))
    }

    async fn materialize(&mut self) -> Result<()> {
        validate_aggregates(&self.aggregates)?;
        let num_group_cols = self.group_by.len();

        // Materialized group state. A grouped aggregate builds it up by
        // hashing keys and resolving collisions. A global aggregate has its
        // one group from the start and never hashes
        let mut state = GroupAccumulatorState::new(&self.aggregates);

        // Global-aggregate fast path: no GROUP BY means exactly one output row,
        // which exists whether or not any input arrives, and the inner loop
        // skips every hashing, group-store and collision-check step
        if num_group_cols == 0 {
            state.ensure_ungrouped_row();

            loop {
                match self.child.next().await? {
                    Some(eb) => {
                        let batch = &eb.batch;
                        let num_rows = batch.num_rows;
                        if num_rows == 0 {
                            continue;
                        }

                        // Resolve each aggregate's input column. ColumnRef
                        // arguments borrow directly from the batch so SUM,
                        // AVG, etc. on a base column do not allocate.
                        let agg_arg_cols: Vec<Vec<Cow<'_, Column>>> = self
                            .aggregates
                            .iter()
                            .map(|agg| {
                                agg.args
                                    .iter()
                                    .map(|arg| {
                                        evaluate_borrowed(arg, batch, &self.input_schema, &[])
                                    })
                                    .collect::<Result<Vec<_>>>()
                            })
                            .collect::<Result<Vec<_>>>()?;

                        state.ingest_ungrouped(&agg_arg_cols, num_rows)?;
                    }
                    None => break,
                }
            }
        } else {
            // Grouped aggregation runs through the shared GroupAccumulatorState
            // so the serial path and the parallel partial-aggregate path use one
            // grouping and one find-or-create implementation.
            let mut router = self.new_router();
            loop {
                match self.child.next().await? {
                    Some(eb) => {
                        // Without somewhere to put the overflow the budget is
                        // the hard limit it always was. Group state grows by
                        // at most the batch it ingests, and a high-cardinality
                        // GROUP BY approaches that bound, so the batch size is
                        // the reservation proxy
                        if router.is_none() {
                            if let Some(budget) = &self.memory_budget {
                                budget.reserve(eb.batch.approx_bytes())?;
                            }
                        }
                        // Around the fold alone, not the pull above it, so
                        // this and the scan phases add up rather than nest
                        let _fold = zyron_common::profile::scope(
                            zyron_common::profile::Phase::ExecAggregateFold,
                        );
                        state.ingest_bounded(
                            &eb.batch,
                            &self.group_by,
                            &self.aggregates,
                            &self.input_schema,
                            router.as_mut(),
                            0,
                            self.spill_threshold_bytes,
                        )?;
                    }
                    None => break,
                }
            }
            if let Some(router) = router {
                self.queue_partitions(router, 1)?;
            }
        }

        if state.num_groups == 0 {
            return Ok(());
        }

        self.result = Some(finalize_groups(
            &state,
            num_group_cols,
            &self.output_schema,
        )?);
        Ok(())
    }
}

/// Builds the output batch from materialized group keys and accumulators.
/// Shared by the serial and parallel aggregate paths so both emit identical
/// column layouts.
/// Coerces an aggregate's finalized scalar to the output column's declared type.
/// Accumulators finalize to their natural scalar (SUM over integers yields an
/// i128, for example) while the output column is built from the binder's
/// declared aggregate return type. This casts a numeric result to that type so
/// the column data matches its schema; a value that does not fit the target
/// integer width is an overflow error, not a silent truncation. A value that
/// already matches, a NULL, or a target with no numeric scalar form (MIN/MAX
/// over a temporal column) passes through unchanged.
pub(crate) fn coerce_aggregate_scalar(val: ScalarValue, target: TypeId) -> Result<ScalarValue> {
    if matches!(val, ScalarValue::Null) || val.type_id() == target {
        return Ok(val);
    }
    let src = val.type_id();
    let int_val = val
        .to_i128()
        .or_else(|| val.to_f64().map(|f| f.round() as i128));
    let float_val = val.to_f64();
    let fit = |o: Option<ScalarValue>| {
        o.ok_or_else(|| {
            ZyronError::ExecutionError(format!(
                "aggregate result of type {src:?} does not fit output type {target:?}"
            ))
        })
    };
    match target {
        TypeId::Int8 => fit(int_val
            .and_then(|i| i8::try_from(i).ok())
            .map(ScalarValue::Int8)),
        TypeId::Int16 => fit(int_val
            .and_then(|i| i16::try_from(i).ok())
            .map(ScalarValue::Int16)),
        TypeId::Int32 => fit(int_val
            .and_then(|i| i32::try_from(i).ok())
            .map(ScalarValue::Int32)),
        TypeId::Int64 => fit(int_val
            .and_then(|i| i64::try_from(i).ok())
            .map(ScalarValue::Int64)),
        TypeId::Int128 | TypeId::Decimal => fit(int_val.map(ScalarValue::Int128)),
        TypeId::UInt8 => fit(int_val
            .and_then(|i| u8::try_from(i).ok())
            .map(ScalarValue::UInt8)),
        TypeId::UInt16 => fit(int_val
            .and_then(|i| u16::try_from(i).ok())
            .map(ScalarValue::UInt16)),
        TypeId::UInt32 => fit(int_val
            .and_then(|i| u32::try_from(i).ok())
            .map(ScalarValue::UInt32)),
        TypeId::UInt64 => fit(int_val
            .and_then(|i| u64::try_from(i).ok())
            .map(ScalarValue::UInt64)),
        TypeId::Float32 => fit(float_val.map(|f| ScalarValue::Float32(f as f32))),
        TypeId::Float64 => fit(float_val.map(ScalarValue::Float64)),
        _ => Ok(val),
    }
}

fn finalize_groups(
    state: &GroupAccumulatorState,
    num_group_cols: usize,
    output_schema: &[LogicalColumn],
) -> Result<DataBatch> {
    let num_groups = state.num_groups;
    let mut columns: Vec<Column> = Vec::with_capacity(output_schema.len());
    for (i, col_def) in output_schema.iter().enumerate().take(num_group_cols) {
        let store_col = &state.group_key_store[i];
        let mut data = ColumnData::with_capacity(col_def.type_id, num_groups);
        let mut nulls = NullBitmap::empty();
        for gidx in 0..num_groups {
            nulls.push(store_col.is_null(gidx));
            data.push_from(&store_col.data, gidx);
        }
        // The declared fractional digits ride along so a decimal key column
        // compares and renders on its value scale
        columns.push(Column::with_nulls_ts(
            data,
            nulls,
            col_def.type_id,
            col_def.fractional_digits,
        ));
    }
    for (acc, col_def) in state
        .accumulators
        .iter()
        .zip(&output_schema[num_group_cols..])
    {
        columns.push(acc.finalize_column(num_groups, col_def)?);
    }
    Ok(DataBatch::new(columns))
}

/// Group index of a row the table had no room for, which no fold touches
const ROUTED: u32 = u32::MAX;

/// Calls `fold(row, group)` for every row of a batch that has a group and,
/// when a column is given, a value in it. Both checks are hoisted out of
/// the loop, so the common batch, with nothing routed and no nulls, runs a
/// plain loop over the rows
#[inline(always)]
fn for_each_folded_row(
    row_groups: &[u32],
    any_routed: bool,
    nulls: Option<&NullBitmap>,
    mut fold: impl FnMut(usize, usize),
) {
    match (any_routed, nulls.filter(|n| n.has_nulls())) {
        (false, None) => {
            for (row, &g) in row_groups.iter().enumerate() {
                fold(row, g as usize);
            }
        }
        (true, None) => {
            for (row, &g) in row_groups.iter().enumerate() {
                if g != ROUTED {
                    fold(row, g as usize);
                }
            }
        }
        (false, Some(nulls)) => {
            for (row, &g) in row_groups.iter().enumerate() {
                if !nulls.is_null(row) {
                    fold(row, g as usize);
                }
            }
        }
        (true, Some(nulls)) => {
            for (row, &g) in row_groups.iter().enumerate() {
                if g != ROUTED && !nulls.is_null(row) {
                    fold(row, g as usize);
                }
            }
        }
    }
}

/// Calls `fold` on every present value of a column, in row order.
///
/// The null bitmap is walked a word at a time, so a block of sixty four
/// rows with nothing null runs a plain loop over the slice and the bit
/// test is paid only inside a block that has a null in it
#[inline(always)]
fn for_each_present<'a, T>(values: &'a [T], nulls: &NullBitmap, mut fold: impl FnMut(&'a T)) {
    let words = nulls.words();
    for (block, chunk) in values.chunks(64).enumerate() {
        match words.get(block).copied().unwrap_or(0) {
            0 => chunk.iter().for_each(&mut fold),
            word => {
                for (bit, value) in chunk.iter().enumerate() {
                    if (word >> bit) & 1 == 0 {
                        fold(value);
                    }
                }
            }
        }
    }
}

/// Rows of a column that hold a value, from the null bitmap's set bits
#[inline]
fn present_count(nulls: &NullBitmap, num_rows: usize) -> usize {
    let absent: usize = nulls.words().iter().map(|w| w.count_ones() as usize).sum();
    num_rows.saturating_sub(absent)
}

/// The exact sum and the count of a column's present integers, None when
/// none is present.
///
/// Two accumulators inside a block with no null in it, so consecutive adds
/// do not wait on each other. Sixty four values widened from sixty four
/// bits cannot overflow the wide accumulator, the checked add happens once
/// when the block total lands in the group's state
#[inline(always)]
fn sum_present<T: Copy>(
    values: &[T],
    nulls: &NullBitmap,
    widen: impl Fn(T) -> i128,
) -> Option<(i128, usize)> {
    let words = nulls.words();
    let mut total: i128 = 0;
    let mut count = 0usize;
    for (block, chunk) in values.chunks(64).enumerate() {
        match words.get(block).copied().unwrap_or(0) {
            0 => {
                let mut even: i128 = 0;
                let mut odd: i128 = 0;
                let (pairs, rest) = chunk.as_chunks::<2>();
                for pair in pairs {
                    even += widen(pair[0]);
                    odd += widen(pair[1]);
                }
                for &value in rest {
                    even += widen(value);
                }
                total += even + odd;
                count += chunk.len();
            }
            word => {
                for (bit, &value) in chunk.iter().enumerate() {
                    if (word >> bit) & 1 == 0 {
                        total += widen(value);
                        count += 1;
                    }
                }
            }
        }
    }
    (count > 0).then_some((total, count))
}

/// Folds a whole column into one group's running extreme.
///
/// The batch's own best is found first and compared with the stored value
/// once, so a text column clones one string per batch rather than one per
/// row that beat the last. Ties keep what arrived first, as the grouped
/// fold does
#[inline(always)]
fn fold_extreme_all<'a, T: Clone>(
    slot: &mut Option<T>,
    values: &'a [T],
    nulls: &NullBitmap,
    is_max: bool,
    cmp: impl Fn(&T, &T) -> std::cmp::Ordering,
) {
    let wanted = if is_max {
        std::cmp::Ordering::Greater
    } else {
        std::cmp::Ordering::Less
    };
    let mut best: Option<&'a T> = None;
    for_each_present(values, nulls, |value| {
        let replace = match best {
            Some(current) => cmp(value, current) == wanted,
            None => true,
        };
        if replace {
            best = Some(value);
        }
    });
    if let Some(value) = best {
        let replace = match slot {
            Some(current) => cmp(value, current) == wanted,
            None => true,
        };
        if replace {
            *slot = Some(value.clone());
        }
    }
}

/// A group's running SUM. Integers accumulate exactly in i128 and floating
/// input in f64, and the flags record whether anything was seen, whether
/// any of it was floating, and whether the integer sum overflowed
#[derive(Clone, Copy, Default)]
struct SumState {
    int_sum: i128,
    float_sum: f64,
    flags: u8,
}

const SUM_HAS_VALUE: u8 = 1;
const SUM_SAW_FLOAT: u8 = 1 << 1;
const SUM_OVERFLOWED: u8 = 1 << 2;

impl SumState {
    #[inline]
    fn add_int(&mut self, v: i128) {
        match self.int_sum.checked_add(v) {
            Some(s) => self.int_sum = s,
            None => self.flags |= SUM_OVERFLOWED,
        }
        self.flags |= SUM_HAS_VALUE;
    }

    #[inline]
    fn add_float(&mut self, v: f64) {
        self.float_sum += v;
        self.flags |= SUM_SAW_FLOAT | SUM_HAS_VALUE;
    }

    /// Folds a value of any type, sorting it into the integer or the
    /// floating sum. A value with no numeric form is not a value
    fn add_scalar(&mut self, value: &ScalarValue) {
        match value {
            ScalarValue::Float32(_) | ScalarValue::Float64(_) => {
                if let Some(f) = value.to_f64() {
                    self.add_float(f);
                }
            }
            _ => {
                if let Some(i) = value.to_i128() {
                    self.add_int(i);
                }
            }
        }
    }

    fn merge(&mut self, other: SumState) {
        self.flags |= other.flags & SUM_OVERFLOWED;
        if other.flags & SUM_HAS_VALUE != 0 {
            self.add_int(other.int_sum);
            self.float_sum += other.float_sum;
            self.flags |= other.flags & SUM_SAW_FLOAT;
        }
    }

    /// The natural scalar of the sum: NULL with nothing seen, an f64 once
    /// any floating input was, an i128 otherwise. An overflowed sum fails
    /// the statement rather than returning a wrapped value
    fn finalize(&self) -> Result<ScalarValue> {
        if self.flags & SUM_OVERFLOWED != 0 {
            return Err(ZyronError::ExecutionError(
                "SUM overflowed its 128-bit accumulator".to_string(),
            ));
        }
        Ok(if self.flags & SUM_HAS_VALUE == 0 {
            ScalarValue::Null
        } else if self.flags & SUM_SAW_FLOAT != 0 {
            ScalarValue::Float64(self.float_sum + self.int_sum as f64)
        } else {
            ScalarValue::Int128(self.int_sum)
        })
    }
}

/// A group's running AVG, on the value scale
#[derive(Clone, Copy, Default)]
struct AvgState {
    sum: f64,
    count: i64,
}

impl AvgState {
    #[inline]
    fn add(&mut self, v: f64) {
        self.sum += v;
        self.count += 1;
    }

    fn finalize(&self) -> ScalarValue {
        if self.count > 0 {
            ScalarValue::Float64(self.sum / self.count as f64)
        } else {
            ScalarValue::Null
        }
    }
}

/// The running MIN or MAX of every group, typed by the argument column so
/// a row compares against the stored value directly rather than through a
/// scalar built for the comparison
enum Extremes {
    Int64(Vec<Option<i64>>),
    Int32(Vec<Option<i32>>),
    Float64(Vec<Option<f64>>),
    Float32(Vec<Option<f32>>),
    Utf8(Vec<Option<String>>),
}

/// Folds a batch of values into their groups' running extremes. A value
/// replaces the stored one only when it strictly beats it, so ties keep
/// what arrived first
#[inline(always)]
fn fold_extremes<T: Clone>(
    slots: &mut [Option<T>],
    values: &[T],
    row_groups: &[u32],
    any_routed: bool,
    nulls: &NullBitmap,
    is_max: bool,
    cmp: impl Fn(&T, &T) -> std::cmp::Ordering,
) {
    let wanted = if is_max {
        std::cmp::Ordering::Greater
    } else {
        std::cmp::Ordering::Less
    };
    for_each_folded_row(row_groups, any_routed, Some(nulls), |row, g| {
        let v = &values[row];
        let replace = match &slots[g] {
            Some(cur) => cmp(v, cur) == wanted,
            None => true,
        };
        if replace {
            slots[g] = Some(v.clone());
        }
    });
}

/// Folds another partition's extremes in, group `og` of theirs landing in
/// group `mapping[og]` here
fn merge_extremes<T>(
    slots: &mut [Option<T>],
    theirs: Vec<Option<T>>,
    mapping: &[u32],
    is_max: bool,
    cmp: impl Fn(&T, &T) -> std::cmp::Ordering,
) {
    let wanted = if is_max {
        std::cmp::Ordering::Greater
    } else {
        std::cmp::Ordering::Less
    };
    for (og, v) in theirs.into_iter().enumerate() {
        let Some(v) = v else {
            continue;
        };
        let slot = &mut slots[mapping[og] as usize];
        let replace = match slot {
            Some(cur) => cmp(&v, cur) == wanted,
            None => true,
        };
        if replace {
            *slot = Some(v);
        }
    }
}

/// The output column of a fixed-width extreme, null where a group saw no
/// value
fn extremes_column<T: Copy + Default>(
    slots: &[Option<T>],
    num_groups: usize,
    wrap: impl FnOnce(Vec<T>) -> ColumnData,
    col_def: &LogicalColumn,
) -> Column {
    let mut values = Vec::with_capacity(num_groups);
    let mut nulls = NullBitmap::none(num_groups);
    for (g, slot) in slots[..num_groups].iter().enumerate() {
        match slot {
            Some(v) => values.push(*v),
            None => {
                nulls.set_null(g);
                values.push(T::default());
            }
        }
    }
    Column::with_nulls_ts(
        wrap(values),
        nulls,
        col_def.type_id,
        col_def.fractional_digits,
    )
}

fn extreme_type_mismatch() -> ZyronError {
    ZyronError::ExecutionError(
        "MIN or MAX received a column of a type other than the one it was planned over".to_string(),
    )
}

impl Extremes {
    fn grow_to(&mut self, num_groups: usize) {
        match self {
            Extremes::Int64(v) => v.resize(num_groups, None),
            Extremes::Int32(v) => v.resize(num_groups, None),
            Extremes::Float64(v) => v.resize(num_groups, None),
            Extremes::Float32(v) => v.resize(num_groups, None),
            Extremes::Utf8(v) => v.resize(num_groups, None),
        }
    }

    fn bytes_per_group(&self) -> u64 {
        match self {
            Extremes::Int64(_) | Extremes::Float64(_) => 16,
            Extremes::Int32(_) | Extremes::Float32(_) => 8,
            Extremes::Utf8(_) => 32,
        }
    }

    fn fold_batch(
        &mut self,
        row_groups: &[u32],
        any_routed: bool,
        col: &Column,
        is_max: bool,
    ) -> Result<()> {
        let nulls = &col.nulls;
        match (self, &col.data) {
            (Extremes::Int64(slots), ColumnData::Int64(v)) => {
                fold_extremes(slots, v, row_groups, any_routed, nulls, is_max, |a, b| {
                    a.cmp(b)
                });
            }
            (Extremes::Int32(slots), ColumnData::Int32(v)) => {
                fold_extremes(slots, v, row_groups, any_routed, nulls, is_max, |a, b| {
                    a.cmp(b)
                });
            }
            (Extremes::Float64(slots), ColumnData::Float64(v)) => {
                fold_extremes(slots, v, row_groups, any_routed, nulls, is_max, |a, b| {
                    crate::compute::cmp_f64_total(*a, *b)
                });
            }
            (Extremes::Float32(slots), ColumnData::Float32(v)) => {
                fold_extremes(slots, v, row_groups, any_routed, nulls, is_max, |a, b| {
                    crate::compute::cmp_f32_total(*a, *b)
                });
            }
            (Extremes::Utf8(slots), ColumnData::Utf8(v)) => {
                fold_extremes(slots, v, row_groups, any_routed, nulls, is_max, |a, b| {
                    a.cmp(b)
                });
            }
            _ => return Err(extreme_type_mismatch()),
        }
        Ok(())
    }

    /// Folds a whole column into one group, for an aggregate with no
    /// grouping
    fn fold_all(&mut self, gidx: usize, col: &Column, is_max: bool) -> Result<()> {
        let nulls = &col.nulls;
        match (self, &col.data) {
            (Extremes::Int64(slots), ColumnData::Int64(v)) => {
                fold_extreme_all(&mut slots[gidx], v, nulls, is_max, |a, b| a.cmp(b));
            }
            (Extremes::Int32(slots), ColumnData::Int32(v)) => {
                fold_extreme_all(&mut slots[gidx], v, nulls, is_max, |a, b| a.cmp(b));
            }
            (Extremes::Float64(slots), ColumnData::Float64(v)) => {
                fold_extreme_all(&mut slots[gidx], v, nulls, is_max, |a, b| {
                    crate::compute::cmp_f64_total(*a, *b)
                });
            }
            (Extremes::Float32(slots), ColumnData::Float32(v)) => {
                fold_extreme_all(&mut slots[gidx], v, nulls, is_max, |a, b| {
                    crate::compute::cmp_f32_total(*a, *b)
                });
            }
            (Extremes::Utf8(slots), ColumnData::Utf8(v)) => {
                fold_extreme_all(&mut slots[gidx], v, nulls, is_max, |a, b| a.cmp(b));
            }
            _ => return Err(extreme_type_mismatch()),
        }
        Ok(())
    }

    fn merge_from(&mut self, other: Extremes, mapping: &[u32], is_max: bool) -> Result<()> {
        match (self, other) {
            (Extremes::Int64(slots), Extremes::Int64(theirs)) => {
                merge_extremes(slots, theirs, mapping, is_max, |a, b| a.cmp(b));
            }
            (Extremes::Int32(slots), Extremes::Int32(theirs)) => {
                merge_extremes(slots, theirs, mapping, is_max, |a, b| a.cmp(b));
            }
            (Extremes::Float64(slots), Extremes::Float64(theirs)) => {
                merge_extremes(slots, theirs, mapping, is_max, |a, b| {
                    crate::compute::cmp_f64_total(*a, *b)
                });
            }
            (Extremes::Float32(slots), Extremes::Float32(theirs)) => {
                merge_extremes(slots, theirs, mapping, is_max, |a, b| {
                    crate::compute::cmp_f32_total(*a, *b)
                });
            }
            (Extremes::Utf8(slots), Extremes::Utf8(theirs)) => {
                merge_extremes(slots, theirs, mapping, is_max, |a, b| a.cmp(b));
            }
            _ => return Err(extreme_type_mismatch()),
        }
        Ok(())
    }

    fn scalar(&self, g: usize) -> ScalarValue {
        match self {
            Extremes::Int64(v) => v[g].map_or(ScalarValue::Null, ScalarValue::Int64),
            Extremes::Int32(v) => v[g].map_or(ScalarValue::Null, ScalarValue::Int32),
            Extremes::Float64(v) => v[g].map_or(ScalarValue::Null, ScalarValue::Float64),
            Extremes::Float32(v) => v[g].map_or(ScalarValue::Null, ScalarValue::Float32),
            Extremes::Utf8(v) => v[g].clone().map_or(ScalarValue::Null, ScalarValue::Utf8),
        }
    }

    /// The output column built straight from the store when the declared
    /// type is the stored one, or None when it needs the scalar path
    fn column_if_direct(&self, num_groups: usize, col_def: &LogicalColumn) -> Option<Column> {
        match (self, col_def.type_id) {
            (Extremes::Int64(slots), TypeId::Int64) => Some(extremes_column(
                slots,
                num_groups,
                ColumnData::Int64,
                col_def,
            )),
            (Extremes::Int32(slots), TypeId::Int32) => Some(extremes_column(
                slots,
                num_groups,
                ColumnData::Int32,
                col_def,
            )),
            (Extremes::Float64(slots), TypeId::Float64) => Some(extremes_column(
                slots,
                num_groups,
                ColumnData::Float64,
                col_def,
            )),
            (Extremes::Float32(slots), TypeId::Float32) => Some(extremes_column(
                slots,
                num_groups,
                ColumnData::Float32,
                col_def,
            )),
            (Extremes::Utf8(slots), TypeId::Text | TypeId::Varchar) => {
                let mut values = Vec::with_capacity(num_groups);
                let mut nulls = NullBitmap::none(num_groups);
                for (g, slot) in slots[..num_groups].iter().enumerate() {
                    match slot {
                        Some(s) => values.push(s.clone()),
                        None => {
                            nulls.set_null(g);
                            values.push(String::new());
                        }
                    }
                }
                Some(Column::with_nulls_ts(
                    ColumnData::Utf8(values),
                    nulls,
                    col_def.type_id,
                    col_def.fractional_digits,
                ))
            }
            _ => None,
        }
    }
}

/// The one argument column of an aggregate that takes exactly one
fn argument<'a>(cols: &'a [Cow<'a, Column>]) -> Result<&'a Column> {
    cols.first()
        .map(|c| c.as_ref())
        .ok_or_else(|| ZyronError::ExecutionError("aggregate is missing its argument".to_string()))
}

/// The state of one aggregate for every group, one array per aggregate
/// rather than a box per group per aggregate.
///
/// A row's fold is then an index into a typed array. Behind a box per
/// group it was a load of the group's vector, a load of the box, a virtual
/// call and a match on the column type, each of them its own cache line
/// once the groups outnumber a cache level, and the match repeated for
/// every row. Here a batch is folded one aggregate at a time, the type is
/// matched once per batch, and the loop over the rows touches the state
/// array and the argument column. The aggregates whose state is a fixed
/// scalar take this form. Everything else keeps a box per group and the
/// accumulator it always had
enum GroupedAccumulator {
    /// COUNT(*) counts every row, COUNT(expr) the rows where it is not null
    Count {
        counts: Vec<i64>,
        of_argument: bool,
    },
    Sum(Vec<SumState>),
    Avg(Vec<AvgState>),
    /// MIN or MAX over one column of a type the store has an array for
    Extreme {
        slots: Extremes,
        is_max: bool,
    },
    /// A box per group, built from the aggregate's definition as groups
    /// appear
    Boxed {
        accs: Vec<Box<dyn Accumulator>>,
        template: AggregateExpr,
    },
}

impl GroupedAccumulator {
    fn new(agg: &AggregateExpr) -> Self {
        let boxed = || GroupedAccumulator::Boxed {
            accs: Vec::new(),
            template: agg.clone(),
        };
        // DISTINCT wraps the accumulator and a user-defined one is its own
        // state, so both keep the box
        if agg.uda.is_some() || (agg.distinct && !agg.args.is_empty()) {
            return boxed();
        }
        match agg.function_name.to_lowercase().as_str() {
            "count" => GroupedAccumulator::Count {
                counts: Vec::new(),
                of_argument: !agg.args.is_empty(),
            },
            "sum" if agg.args.len() == 1 => GroupedAccumulator::Sum(Vec::new()),
            "avg" if agg.args.len() == 1 => GroupedAccumulator::Avg(Vec::new()),
            name @ ("min" | "max") if agg.args.len() == 1 => {
                let slots = match agg.args[0].type_id() {
                    TypeId::Int64 => Extremes::Int64(Vec::new()),
                    TypeId::Int32 => Extremes::Int32(Vec::new()),
                    TypeId::Float64 => Extremes::Float64(Vec::new()),
                    TypeId::Float32 => Extremes::Float32(Vec::new()),
                    TypeId::Text | TypeId::Varchar => Extremes::Utf8(Vec::new()),
                    _ => return boxed(),
                };
                GroupedAccumulator::Extreme {
                    slots,
                    is_max: name == "max",
                }
            }
            _ => boxed(),
        }
    }

    /// Makes room for groups up to `num_groups`, the new ones at their
    /// fresh state
    fn grow_to(&mut self, num_groups: usize) {
        match self {
            GroupedAccumulator::Count { counts, .. } => counts.resize(num_groups, 0),
            GroupedAccumulator::Sum(states) => states.resize(num_groups, SumState::default()),
            GroupedAccumulator::Avg(states) => states.resize(num_groups, AvgState::default()),
            GroupedAccumulator::Extreme { slots, .. } => slots.grow_to(num_groups),
            GroupedAccumulator::Boxed { accs, template } => {
                if accs.len() < num_groups {
                    accs.resize_with(num_groups, || create_accumulator(template));
                }
            }
        }
    }

    /// Bytes one group's state costs
    fn bytes_per_group(&self) -> u64 {
        match self {
            GroupedAccumulator::Count { .. } => std::mem::size_of::<i64>() as u64,
            GroupedAccumulator::Sum(_) => std::mem::size_of::<SumState>() as u64,
            GroupedAccumulator::Avg(_) => std::mem::size_of::<AvgState>() as u64,
            GroupedAccumulator::Extreme { slots, .. } => slots.bytes_per_group(),
            GroupedAccumulator::Boxed { .. } => ACCUMULATOR_BYTES,
        }
    }

    /// Folds every row of a batch into its group. `row_groups` holds the
    /// group of each row, or `ROUTED` for a row that went to a partition
    fn fold_batch(
        &mut self,
        row_groups: &[u32],
        any_routed: bool,
        cols: &[Cow<'_, Column>],
    ) -> Result<()> {
        match self {
            GroupedAccumulator::Count {
                counts,
                of_argument: false,
            } => {
                for_each_folded_row(row_groups, any_routed, None, |_, g| counts[g] += 1);
            }
            GroupedAccumulator::Count {
                counts,
                of_argument: true,
            } => {
                let col = argument(cols)?;
                for_each_folded_row(row_groups, any_routed, Some(&col.nulls), |_, g| {
                    counts[g] += 1
                });
            }
            GroupedAccumulator::Sum(states) => {
                let col = argument(cols)?;
                let nulls = Some(&col.nulls);
                match &col.data {
                    ColumnData::Int64(v) => {
                        for_each_folded_row(row_groups, any_routed, nulls, |row, g| {
                            states[g].add_int(v[row] as i128)
                        });
                    }
                    ColumnData::Int32(v) => {
                        for_each_folded_row(row_groups, any_routed, nulls, |row, g| {
                            states[g].add_int(v[row] as i128)
                        });
                    }
                    ColumnData::Int128(v) => {
                        for_each_folded_row(row_groups, any_routed, nulls, |row, g| {
                            states[g].add_int(v[row])
                        });
                    }
                    ColumnData::Float64(v) => {
                        for_each_folded_row(row_groups, any_routed, nulls, |row, g| {
                            states[g].add_float(v[row])
                        });
                    }
                    ColumnData::Float32(v) => {
                        for_each_folded_row(row_groups, any_routed, nulls, |row, g| {
                            states[g].add_float(v[row] as f64)
                        });
                    }
                    // Other numeric widths go through the scalar, which
                    // sorts them into the integer or the floating sum
                    data => {
                        for_each_folded_row(row_groups, any_routed, nulls, |row, g| {
                            states[g].add_scalar(&data.get_scalar(row))
                        });
                    }
                }
            }
            GroupedAccumulator::Avg(states) => {
                let col = argument(cols)?;
                let nulls = Some(&col.nulls);
                match &col.data {
                    ColumnData::Int64(v) => {
                        for_each_folded_row(row_groups, any_routed, nulls, |row, g| {
                            states[g].add(v[row] as f64)
                        });
                    }
                    ColumnData::Int32(v) => {
                        for_each_folded_row(row_groups, any_routed, nulls, |row, g| {
                            states[g].add(v[row] as f64)
                        });
                    }
                    ColumnData::Float64(v) => {
                        for_each_folded_row(row_groups, any_routed, nulls, |row, g| {
                            states[g].add(v[row])
                        });
                    }
                    ColumnData::Float32(v) => {
                        for_each_folded_row(row_groups, any_routed, nulls, |row, g| {
                            states[g].add(v[row] as f64)
                        });
                    }
                    // Decimals average on the value scale, not the raw
                    // scaled integer
                    ColumnData::Int128(_) => {
                        for_each_folded_row(row_groups, any_routed, nulls, |row, g| {
                            if let Some(x) = numeric_value_f64(col, row) {
                                states[g].add(x);
                            }
                        });
                    }
                    data => {
                        for_each_folded_row(row_groups, any_routed, nulls, |row, g| {
                            if let Some(x) = data.get_scalar(row).to_f64() {
                                states[g].add(x);
                            }
                        });
                    }
                }
            }
            GroupedAccumulator::Extreme { slots, is_max } => {
                slots.fold_batch(row_groups, any_routed, argument(cols)?, *is_max)?;
            }
            GroupedAccumulator::Boxed { accs, .. } => {
                for (row, &g) in row_groups.iter().enumerate() {
                    if g != ROUTED {
                        accs[g as usize].update_row_checked(cols, row)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Folds every row of a batch into one group, for an aggregate with no
    /// grouping.
    ///
    /// One group means there is no row to group lookup to make, so a count,
    /// a sum, an average and an extreme run over the argument as a slice
    /// with the null check settled a word at a time. A global aggregate
    /// over a whole table takes this path for every row it reads, and
    /// walking each row through a group index was most of what it cost.
    /// A shape with no typed form here takes the grouped fold with the
    /// group held constant, which lands the same value more slowly
    fn fold_all_into(
        &mut self,
        gidx: usize,
        cols: &[Cow<'_, Column>],
        num_rows: usize,
        row_groups: &mut Vec<u32>,
    ) -> Result<()> {
        if self.fold_all_typed(gidx, cols, num_rows)? {
            return Ok(());
        }
        row_groups.clear();
        row_groups.resize(num_rows, gidx as u32);
        self.fold_batch(row_groups, false, cols)
    }

    /// The typed whole-column folds, true when one of them took the batch.
    ///
    /// Integer sums and averages total the batch exactly and land once in
    /// the group's state, floating ones add value by value in row order so
    /// the result is the one the grouped fold produces
    fn fold_all_typed(
        &mut self,
        gidx: usize,
        cols: &[Cow<'_, Column>],
        num_rows: usize,
    ) -> Result<bool> {
        match self {
            GroupedAccumulator::Count {
                counts,
                of_argument: false,
            } => {
                counts[gidx] += num_rows as i64;
            }
            GroupedAccumulator::Count {
                counts,
                of_argument: true,
            } => {
                let col = argument(cols)?;
                counts[gidx] += present_count(&col.nulls, num_rows) as i64;
            }
            GroupedAccumulator::Sum(states) => {
                let col = argument(cols)?;
                let nulls = &col.nulls;
                match &col.data {
                    ColumnData::Int64(v) => {
                        if let Some((total, _)) = sum_present(v, nulls, |x| x as i128) {
                            states[gidx].add_int(total);
                        }
                    }
                    ColumnData::Int32(v) => {
                        if let Some((total, _)) = sum_present(v, nulls, |x| x as i128) {
                            states[gidx].add_int(total);
                        }
                    }
                    ColumnData::Float64(v) => {
                        for_each_present(v, nulls, |x| states[gidx].add_float(*x));
                    }
                    ColumnData::Float32(v) => {
                        for_each_present(v, nulls, |x| states[gidx].add_float(*x as f64));
                    }
                    _ => return Ok(false),
                }
            }
            GroupedAccumulator::Avg(states) => {
                let col = argument(cols)?;
                let nulls = &col.nulls;
                match &col.data {
                    ColumnData::Int64(v) => {
                        if let Some((total, count)) = sum_present(v, nulls, |x| x as i128) {
                            states[gidx].sum += total as f64;
                            states[gidx].count += count as i64;
                        }
                    }
                    ColumnData::Int32(v) => {
                        if let Some((total, count)) = sum_present(v, nulls, |x| x as i128) {
                            states[gidx].sum += total as f64;
                            states[gidx].count += count as i64;
                        }
                    }
                    ColumnData::Float64(v) => {
                        for_each_present(v, nulls, |x| states[gidx].add(*x));
                    }
                    ColumnData::Float32(v) => {
                        for_each_present(v, nulls, |x| states[gidx].add(*x as f64));
                    }
                    _ => return Ok(false),
                }
            }
            GroupedAccumulator::Extreme { slots, is_max } => {
                slots.fold_all(gidx, argument(cols)?, *is_max)?;
            }
            GroupedAccumulator::Boxed { .. } => return Ok(false),
        }
        Ok(true)
    }

    /// Folds another partition's state for the same aggregate in, group
    /// `og` of the other landing in group `mapping[og]` here. Partitions
    /// are built from one aggregate list, so the shapes always pair
    fn merge_from(&mut self, other: GroupedAccumulator, mapping: &[u32]) -> Result<()> {
        match (self, other) {
            (
                GroupedAccumulator::Count { counts, .. },
                GroupedAccumulator::Count { counts: theirs, .. },
            ) => {
                for (og, c) in theirs.iter().enumerate() {
                    counts[mapping[og] as usize] += c;
                }
            }
            (GroupedAccumulator::Sum(states), GroupedAccumulator::Sum(theirs)) => {
                for (og, s) in theirs.into_iter().enumerate() {
                    states[mapping[og] as usize].merge(s);
                }
            }
            (GroupedAccumulator::Avg(states), GroupedAccumulator::Avg(theirs)) => {
                for (og, s) in theirs.into_iter().enumerate() {
                    let mine = &mut states[mapping[og] as usize];
                    mine.sum += s.sum;
                    mine.count += s.count;
                }
            }
            (
                GroupedAccumulator::Extreme { slots, is_max },
                GroupedAccumulator::Extreme { slots: theirs, .. },
            ) => slots.merge_from(theirs, mapping, *is_max)?,
            (
                GroupedAccumulator::Boxed { accs, .. },
                GroupedAccumulator::Boxed { accs: theirs, .. },
            ) => {
                for (og, acc) in theirs.iter().enumerate() {
                    accs[mapping[og] as usize].merge(acc.as_ref());
                }
            }
            _ => {
                return Err(ZyronError::ExecutionError(
                    "parallel aggregate merged partitions of different shapes".to_string(),
                ));
            }
        }
        Ok(())
    }

    /// One group's result as its natural scalar, before the output type
    /// is applied
    fn finalize_scalar(&self, g: usize) -> Result<ScalarValue> {
        match self {
            GroupedAccumulator::Count { counts, .. } => Ok(ScalarValue::Int64(counts[g])),
            GroupedAccumulator::Sum(states) => states[g].finalize(),
            GroupedAccumulator::Avg(states) => Ok(states[g].finalize()),
            GroupedAccumulator::Extreme { slots, .. } => Ok(slots.scalar(g)),
            GroupedAccumulator::Boxed { accs, .. } => accs[g].finalize_checked(),
        }
    }

    /// The output column of this aggregate over every group, in the
    /// declared type. A store whose type is the declared one becomes the
    /// column directly, the rest go a scalar per group through the same
    /// coercion the scalar path applies
    fn finalize_column(&self, num_groups: usize, col_def: &LogicalColumn) -> Result<Column> {
        let target = col_def.type_id;
        match self {
            GroupedAccumulator::Count { counts, .. } if target == TypeId::Int64 => {
                return Ok(Column::new_ts(
                    ColumnData::Int64(counts[..num_groups].to_vec()),
                    target,
                    col_def.fractional_digits,
                ));
            }
            GroupedAccumulator::Sum(states)
                if target == TypeId::Int64
                    && states[..num_groups]
                        .iter()
                        .all(|s| s.flags & (SUM_SAW_FLOAT | SUM_OVERFLOWED) == 0) =>
            {
                let mut values = Vec::with_capacity(num_groups);
                let mut nulls = NullBitmap::none(num_groups);
                for (g, s) in states[..num_groups].iter().enumerate() {
                    if s.flags & SUM_HAS_VALUE == 0 {
                        nulls.set_null(g);
                        values.push(0);
                    } else {
                        values.push(i64::try_from(s.int_sum).map_err(|_| {
                            ZyronError::ExecutionError(format!(
                                "aggregate result of type {:?} does not fit output type {target:?}",
                                TypeId::Int128
                            ))
                        })?);
                    }
                }
                return Ok(Column::with_nulls_ts(
                    ColumnData::Int64(values),
                    nulls,
                    target,
                    col_def.fractional_digits,
                ));
            }
            GroupedAccumulator::Sum(states)
                if target == TypeId::Float64
                    && states[..num_groups]
                        .iter()
                        .all(|s| s.flags & SUM_OVERFLOWED == 0) =>
            {
                let mut values = Vec::with_capacity(num_groups);
                let mut nulls = NullBitmap::none(num_groups);
                for (g, s) in states[..num_groups].iter().enumerate() {
                    if s.flags & SUM_HAS_VALUE == 0 {
                        nulls.set_null(g);
                        values.push(0.0);
                    } else {
                        values.push(s.float_sum + s.int_sum as f64);
                    }
                }
                return Ok(Column::with_nulls_ts(
                    ColumnData::Float64(values),
                    nulls,
                    target,
                    col_def.fractional_digits,
                ));
            }
            GroupedAccumulator::Avg(states) if target == TypeId::Float64 => {
                let mut values = Vec::with_capacity(num_groups);
                let mut nulls = NullBitmap::none(num_groups);
                for (g, s) in states[..num_groups].iter().enumerate() {
                    if s.count > 0 {
                        values.push(s.sum / s.count as f64);
                    } else {
                        nulls.set_null(g);
                        values.push(0.0);
                    }
                }
                return Ok(Column::with_nulls_ts(
                    ColumnData::Float64(values),
                    nulls,
                    target,
                    col_def.fractional_digits,
                ));
            }
            GroupedAccumulator::Extreme { slots, .. } => {
                if let Some(column) = slots.column_if_direct(num_groups, col_def) {
                    return Ok(column);
                }
            }
            _ => {}
        }

        let mut data = ColumnData::with_capacity(target, num_groups);
        let mut nulls = NullBitmap::empty();
        for g in 0..num_groups {
            let val = coerce_aggregate_scalar(self.finalize_scalar(g)?, target)?;
            nulls.push(val.is_null());
            data.push_scalar(&val);
        }
        // The declared fractional digits ride along so a decimal aggregate's
        // output column compares and renders on its value scale
        Ok(Column::with_nulls_ts(
            data,
            nulls,
            target,
            col_def.fractional_digits,
        ))
    }
}

/// Partial grouped-aggregation state for one partition. The parallel aggregate
/// builds one per worker over a disjoint page range, then merges them; the
/// serial aggregate uses a single instance over all input. One grouping and
/// find-or-create implementation backs both so they cannot diverge.
struct GroupAccumulatorState {
    group_key_store: Vec<Column>,
    index: GroupIndex,
    /// One per aggregate, each holding every group's state
    accumulators: Vec<GroupedAccumulator>,
    num_groups: usize,
    /// Bytes the grouping keys hold, counted as they are stored rather than
    /// measured afterwards: measuring a text key store means walking every
    /// group, and the answer is wanted once per batch
    key_bytes: u64,
    /// True once the table has reached its budget and stopped taking new
    /// groups. Rows for groups already here still fold into them
    frozen: bool,
    /// Rows of the batch in hand whose group is not resident, reused across
    /// batches so a routed aggregate does not allocate one per batch
    unresident: Vec<u32>,
    /// The group of each row of the batch in hand, filled by one pass over
    /// the rows and read by a pass per aggregate, reused across batches
    row_groups: Vec<u32>,
}

/// Bytes one group costs beyond its key and its accumulator state: its
/// bucket slot, chain link and hash in the index, and the slack of the
/// arrays that grow to hold it.
const GROUP_OVERHEAD_BYTES: u64 = 64;

/// Where a group is found from its hash.
///
/// A flat bucket table chained through an array, which is what the hash
/// join builds its side with and the set operations keep their rows in. A
/// map from hash to a vector of group indices allocates one vector per
/// distinct hash, so a grouping over a hundred thousand keys made a hundred
/// thousand short-lived allocations to hold one index each, and every
/// lookup chased the map's node and then the vector's heap block. The
/// window operator groups its partitions through the same index
pub(crate) struct GroupIndex {
    /// Bucket heads, chained backwards through `chain`
    table: compute::FlatHashTable,
    /// Per group, the group that held its bucket before it
    chain: Vec<u32>,
    /// Per group, its full hash, so a chain walk rejects a collision
    /// without touching the key store and growing the table is a pass
    /// over these rather than a rehash of every stored key
    hashes: Vec<u64>,
    /// Groups the table is sized for. Past it the table doubles and rebuilds
    capacity: usize,
}

/// Groups a fresh index is sized for, before any growth
const GROUP_INDEX_INITIAL_GROUPS: usize = 1024;

/// Rows ahead of the one being folded whose bucket is prefetched, so the
/// table access for a row is in cache by the time the fold reaches it
const GROUP_PREFETCH_DISTANCE: usize = 16;

impl GroupIndex {
    pub(crate) fn new() -> Self {
        Self {
            table: compute::FlatHashTable::with_capacity(GROUP_INDEX_INITIAL_GROUPS),
            chain: Vec::new(),
            hashes: Vec::new(),
            capacity: GROUP_INDEX_INITIAL_GROUPS,
        }
    }

    /// The group whose hash matches and whose key `equals` accepts, or None
    #[inline]
    pub(crate) fn find(&self, hash: u64, equals: impl Fn(usize) -> bool) -> Option<usize> {
        let mut idx = self.table.get(hash);
        while idx != u32::MAX {
            let at = idx as usize;
            if self.hashes[at] == hash && equals(at) {
                return Some(at);
            }
            idx = self.chain[at];
        }
        None
    }

    /// Records a new group under its hash. The group's index is the number
    /// of groups recorded before it
    pub(crate) fn insert(&mut self, hash: u64) {
        if self.hashes.len() >= self.capacity {
            self.grow();
        }
        let gidx = self.hashes.len() as u32;
        let prev = self.table.insert(hash, gidx);
        self.chain.push(prev);
        self.hashes.push(hash);
    }

    /// Doubles the bucket table, rebuilding the chains from the hashes
    /// already stored
    fn grow(&mut self) {
        self.capacity *= 2;
        self.table = compute::FlatHashTable::with_capacity(self.capacity);
        for (gidx, hash) in self.hashes.iter().enumerate() {
            self.chain[gidx] = self.table.insert(*hash, gidx as u32);
        }
    }

    #[inline]
    pub(crate) fn prefetch(&self, hash: u64) {
        self.table.prefetch(hash);
    }
}

/// Bytes one boxed accumulator costs: the box, and the state of the widest
/// fixed-size accumulator there is.
const ACCUMULATOR_BYTES: u64 = 48;

/// Times a partition may be split again before it is aggregated in memory
/// whatever its size.
///
/// Each level absorbs at least one group into its resident table before it
/// routes anything, so the recursion ends on its own. The cap is for file
/// descriptors: sixteen to the eighth is more partitions than any real
/// grouping produces, and a threshold small enough to keep splitting past
/// that is a threshold too small to hold one group.
const MAX_AGGREGATE_SPILL_DEPTH: u32 = 8;

impl GroupAccumulatorState {
    fn new(aggregates: &[AggregateExpr]) -> Self {
        Self {
            group_key_store: Vec::new(),
            index: GroupIndex::new(),
            accumulators: aggregates.iter().map(GroupedAccumulator::new).collect(),
            num_groups: 0,
            key_bytes: 0,
            frozen: false,
            unresident: Vec::new(),
            row_groups: Vec::new(),
        }
    }

    /// What the table holds, near enough to decide when to stop growing.
    ///
    /// The keys and the flat accumulators are counted exactly. A boxed
    /// accumulator is estimated, because its state sits behind a trait
    /// object with no size to ask for, and a fixed figure is right for every
    /// one whose state is fixed, which is all of them but DISTINCT and a
    /// user-defined one. Those two keep growing after the table is frozen,
    /// and there is nothing to do about it without writing accumulator
    /// state to disk, which would need a merge that those two do not have.
    fn resident_bytes(&self) -> u64 {
        let per_group: u64 = self.accumulators.iter().map(|a| a.bytes_per_group()).sum();
        self.key_bytes + self.num_groups as u64 * (GROUP_OVERHEAD_BYTES + per_group)
    }

    /// Gives an aggregate with no GROUP BY its one group, which exists
    /// whether or not any row arrives
    fn ensure_ungrouped_row(&mut self) {
        if self.num_groups == 0 {
            self.num_groups = 1;
            for acc in &mut self.accumulators {
                acc.grow_to(1);
            }
        }
    }

    /// Folds a batch into the one group of an aggregate with no GROUP BY
    fn ingest_ungrouped(
        &mut self,
        agg_arg_cols: &[Vec<Cow<'_, Column>>],
        num_rows: usize,
    ) -> Result<()> {
        self.ensure_ungrouped_row();
        let Self {
            accumulators,
            row_groups,
            ..
        } = self;
        for (acc, cols) in accumulators.iter_mut().zip(agg_arg_cols) {
            acc.fold_all_into(0, cols, num_rows, row_groups)?;
        }
        Ok(())
    }

    /// Folds one input batch into the partition's group state.
    fn ingest(
        &mut self,
        batch: &DataBatch,
        group_by: &[BoundExpr],
        aggregates: &[AggregateExpr],
        input_schema: &[LogicalColumn],
    ) -> Result<()> {
        self.ingest_bounded(batch, group_by, aggregates, input_schema, None, 0, 0)
    }

    /// Folds one input batch in, sending the rows it has no room for to disk.
    ///
    /// With no router and no threshold this is the unbounded ingest above, and
    /// the two share one implementation so the grouping and the find-or-create
    /// cannot drift apart between the path that spills and the path that does
    /// not.
    fn ingest_bounded(
        &mut self,
        batch: &DataBatch,
        group_by: &[BoundExpr],
        aggregates: &[AggregateExpr],
        input_schema: &[LogicalColumn],
        router: Option<&mut crate::operator::grace::PartitionWriterSet>,
        seed: u64,
        threshold_bytes: u64,
    ) -> Result<()> {
        let num_rows = batch.num_rows;
        if num_rows == 0 {
            return Ok(());
        }

        // GROUP BY columns: ColumnRef paths borrow from the batch so a string
        // key column is not cloned per batch.
        let group_cols: Vec<Cow<'_, Column>> = group_by
            .iter()
            .map(|expr| evaluate_borrowed(expr, batch, input_schema, &[]))
            .collect::<Result<Vec<_>>>()?;
        let agg_arg_cols: Vec<Vec<Cow<'_, Column>>> = aggregates
            .iter()
            .map(|agg| {
                agg.args
                    .iter()
                    .map(|arg| evaluate_borrowed(arg, batch, input_schema, &[]))
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;

        let group_refs: Vec<&Column> = group_cols.iter().map(|c| c.as_ref()).collect();
        let hashes = compute::hash_column_batch(&group_refs, num_rows);

        if self.group_key_store.is_empty() {
            for gc in &group_cols {
                self.group_key_store.push(Column::new_ts(
                    ColumnData::with_capacity(gc.type_id, 64),
                    gc.type_id,
                    gc.fractional_digits,
                ));
            }
        }

        let Self {
            group_key_store,
            index,
            accumulators,
            num_groups,
            key_bytes,
            frozen,
            unresident,
            row_groups,
        } = self;

        // Taken so the routing pass below can hand the batch to the writer
        // while the table's own fields are still borrowed
        let mut routed = std::mem::take(unresident);
        routed.clear();
        row_groups.clear();
        row_groups.reserve(num_rows);

        // First the group of every row, so each accumulator can then be
        // folded over the whole batch in one pass of its own
        for row in 0..num_rows {
            // The hashes are all in hand, so the bucket a later row needs
            // is fetched while this one is looked up
            let ahead = row + GROUP_PREFETCH_DISTANCE;
            if ahead < num_rows {
                index.prefetch(hashes[ahead]);
            }
            let gidx = if *frozen {
                match find_group(index, group_key_store, &group_refs, row, hashes[row]) {
                    Some(gidx) => gidx,
                    None => {
                        // A group this table has no room for. Its rows go to
                        // a partition, all of them, so the group is whole
                        // wherever it ends up
                        routed.push(row as u32);
                        row_groups.push(ROUTED);
                        continue;
                    }
                }
            } else {
                find_or_create_group(
                    index,
                    group_key_store,
                    num_groups,
                    key_bytes,
                    &group_refs,
                    row,
                    hashes[row],
                )
            };
            row_groups.push(gidx as u32);
        }

        let any_routed = !routed.is_empty();
        for (acc, cols) in accumulators.iter_mut().zip(&agg_arg_cols) {
            acc.grow_to(*num_groups);
            acc.fold_batch(row_groups, any_routed, cols)?;
        }

        if let Some(router) = router {
            router.push_selected(batch, &hashes, &routed, seed)?;
        } else if !routed.is_empty() {
            return Err(ZyronError::ExecutionError(
                "the aggregate stopped taking groups with nowhere to put them".into(),
            ));
        }
        self.unresident = routed;

        // Checked once per batch rather than per row: the table grows by at
        // most one group per row, and a batch of overshoot is a batch of rows
        // worth of state, not a multiple of the budget
        if !self.frozen && threshold_bytes > 0 && self.resident_bytes() >= threshold_bytes {
            self.frozen = true;
        }
        Ok(())
    }

    /// Combines another partition's partial state into this one. Group keys are
    /// re-hashed from the other partition's key store and matched against this
    /// partition's groups; paired accumulators are merged associatively.
    fn merge(&mut self, other: GroupAccumulatorState) -> Result<()> {
        if other.num_groups == 0 {
            return Ok(());
        }
        if self.group_key_store.is_empty() {
            for c in &other.group_key_store {
                self.group_key_store.push(Column::new(
                    ColumnData::with_capacity(c.type_id, 64),
                    c.type_id,
                ));
            }
        }

        let other_refs: Vec<&Column> = other.group_key_store.iter().collect();
        let hashes = compute::hash_column_batch(&other_refs, other.num_groups);

        let Self {
            group_key_store,
            index,
            accumulators,
            num_groups,
            key_bytes,
            ..
        } = self;

        // Where each of the other's groups lands here
        let mut mapping: Vec<u32> = Vec::with_capacity(other.num_groups);
        for ogidx in 0..other.num_groups {
            let gidx = find_or_create_group(
                index,
                group_key_store,
                num_groups,
                key_bytes,
                &other_refs,
                ogidx,
                hashes[ogidx],
            );
            mapping.push(gidx as u32);
        }
        for (mine, theirs) in accumulators.iter_mut().zip(other.accumulators) {
            mine.grow_to(*num_groups);
            mine.merge_from(theirs, &mapping)?;
        }
        Ok(())
    }
}

/// Whether a stored group's key equals the key at a row of the input.
#[inline]
fn group_key_equals(
    group_key_store: &[Column],
    key_cols: &[&Column],
    row: usize,
    gidx: usize,
) -> bool {
    for (ci, kc) in key_cols.iter().enumerate() {
        let store_col = &group_key_store[ci];
        let a_null = kc.is_null(row);
        let b_null = store_col.is_null(gidx);
        if a_null != b_null {
            return false;
        }
        if a_null {
            continue;
        }
        if !column_values_equal_cross(&kc.data, row, &store_col.data, gidx) {
            return false;
        }
    }
    true
}

/// Finds a row's group, without creating one.
///
/// What a frozen table does: a hit folds the row in for free, and a miss is a
/// group this table is not going to hold.
#[inline]
fn find_group(
    index: &GroupIndex,
    group_key_store: &[Column],
    key_cols: &[&Column],
    row: usize,
    hash: u64,
) -> Option<usize> {
    index.find(hash, |gidx| {
        group_key_equals(group_key_store, key_cols, row, gidx)
    })
}

/// Bytes one row of a key column contributes to the group key store.
#[inline]
fn key_row_bytes(col: &Column, row: usize) -> u64 {
    match &col.data {
        ColumnData::Utf8(v) => v.get(row).map(|s| s.len() as u64 + 24).unwrap_or(24),
        ColumnData::Binary(v) => v.get(row).map(|b| b.len() as u64 + 24).unwrap_or(24),
        ColumnData::Boolean(_) | ColumnData::Int8(_) | ColumnData::UInt8(_) => 1,
        ColumnData::Int16(_) | ColumnData::UInt16(_) => 2,
        ColumnData::Int32(_) | ColumnData::UInt32(_) | ColumnData::Float32(_) => 4,
        ColumnData::Int64(_) | ColumnData::UInt64(_) | ColumnData::Float64(_) => 8,
        ColumnData::Int128(_) | ColumnData::FixedBinary16(_) | ColumnData::Interval(_) => 16,
    }
}

/// Finds the group matching `key_cols[..][row]` by hash and equality, or
/// creates it, copying the key into the store. The accumulators make room
/// for it afterwards, once per batch rather than per group. The disjoint
/// `&mut` parameters let one implementation serve both batch ingest and
/// partition merge without a borrow conflict.
#[allow(clippy::too_many_arguments)]
fn find_or_create_group(
    index: &mut GroupIndex,
    group_key_store: &mut [Column],
    num_groups: &mut usize,
    key_bytes: &mut u64,
    key_cols: &[&Column],
    row: usize,
    hash: u64,
) -> usize {
    if let Some(gidx) = index.find(hash, |gidx| {
        group_key_equals(group_key_store, key_cols, row, gidx)
    }) {
        return gidx;
    }
    let gidx = *num_groups;
    *num_groups += 1;
    index.insert(hash);
    for (ci, kc) in key_cols.iter().enumerate() {
        group_key_store[ci].push_row_from(kc, row);
        *key_bytes += key_row_bytes(kc, row);
    }
    gidx
}

/// Compares values at different indices across two ColumnData instances of the same type.
#[inline]
fn column_values_equal_cross(a: &ColumnData, a_idx: usize, b: &ColumnData, b_idx: usize) -> bool {
    match (a, b) {
        (ColumnData::Boolean(va), ColumnData::Boolean(vb)) => va[a_idx] == vb[b_idx],
        (ColumnData::Int8(va), ColumnData::Int8(vb)) => va[a_idx] == vb[b_idx],
        (ColumnData::Int16(va), ColumnData::Int16(vb)) => va[a_idx] == vb[b_idx],
        (ColumnData::Int32(va), ColumnData::Int32(vb)) => va[a_idx] == vb[b_idx],
        (ColumnData::Int64(va), ColumnData::Int64(vb)) => va[a_idx] == vb[b_idx],
        (ColumnData::Int128(va), ColumnData::Int128(vb)) => va[a_idx] == vb[b_idx],
        (ColumnData::UInt8(va), ColumnData::UInt8(vb)) => va[a_idx] == vb[b_idx],
        (ColumnData::UInt16(va), ColumnData::UInt16(vb)) => va[a_idx] == vb[b_idx],
        (ColumnData::UInt32(va), ColumnData::UInt32(vb)) => va[a_idx] == vb[b_idx],
        (ColumnData::UInt64(va), ColumnData::UInt64(vb)) => va[a_idx] == vb[b_idx],
        (ColumnData::Float32(va), ColumnData::Float32(vb)) => {
            crate::compute::f32_key_eq(va[a_idx], vb[b_idx])
        }
        (ColumnData::Float64(va), ColumnData::Float64(vb)) => {
            crate::compute::f64_key_eq(va[a_idx], vb[b_idx])
        }
        (ColumnData::Utf8(va), ColumnData::Utf8(vb)) => va[a_idx] == vb[b_idx],
        (ColumnData::Binary(va), ColumnData::Binary(vb)) => va[a_idx] == vb[b_idx],
        (ColumnData::FixedBinary16(va), ColumnData::FixedBinary16(vb)) => va[a_idx] == vb[b_idx],
        // hash_column_batch hashes intervals, so interval keys land in the same
        // bucket and must compare here too, otherwise GROUP BY on an interval
        // column makes one group per row.
        (ColumnData::Interval(va), ColumnData::Interval(vb)) => va[a_idx] == vb[b_idx],
        _ => false,
    }
}

impl Operator for HashAggregateOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            loop {
                if self.finished {
                    return Ok(None);
                }

                // Whatever stage produced the batch in hand, it leaves a slice
                // at a time so a partition holding millions of groups is not
                // one output batch
                if let Some(result) = self.result.as_ref() {
                    if self.output_cursor < result.num_rows {
                        let remaining = result.num_rows - self.output_cursor;
                        let chunk = remaining.min(crate::batch::BATCH_SIZE);
                        let batch = result.slice(self.output_cursor, chunk);
                        self.output_cursor += chunk;
                        return Ok(Some(ExecutionBatch::new(batch)));
                    }
                    self.result = None;
                    self.output_cursor = 0;
                }

                if !self.input_drained {
                    let mut timer = crate::calibrate::BatchTimer::start(
                        zyron_pressure::capability::OperatorKind::Aggregate,
                    );
                    self.materialize().await?;
                    self.input_drained = true;
                    timer.rows(self.result.as_ref().map(|b| b.num_rows).unwrap_or(0) as u64);
                    continue;
                }

                // Then the groups that did not fit, one partition at a time.
                // Taken from the back, so a partition that split again is
                // finished before its siblings start and its files are freed
                match self.pending.pop() {
                    Some(partition) => {
                        self.result = self.aggregate_partition(partition)?;
                        self.output_cursor = 0;
                    }
                    None => {
                        self.finished = true;
                        return Ok(None);
                    }
                }
            }
        })
    }
}

// ---------------------------------------------------------------------------
// SortAggregateOperator
// ---------------------------------------------------------------------------

/// Sort-based aggregation. Currently delegates to HashAggregateOperator.
pub struct SortAggregateOperator {
    inner: HashAggregateOperator,
}

impl SortAggregateOperator {
    pub fn new(
        child: Box<dyn Operator>,
        group_by: Vec<BoundExpr>,
        aggregates: Vec<AggregateExpr>,
        input_schema: Vec<LogicalColumn>,
        output_schema: Vec<LogicalColumn>,
    ) -> Self {
        Self {
            inner: HashAggregateOperator::new(
                child,
                group_by,
                aggregates,
                input_schema,
                output_schema,
            ),
        }
    }
}

impl SortAggregateOperator {
    /// Attaches the query memory budget to the aggregate underneath.
    pub fn set_memory_budget(&mut self, budget: Option<Arc<crate::context::QueryMemoryBudget>>) {
        self.inner.set_memory_budget(budget);
    }

    /// Gives the aggregate underneath somewhere to spill.
    pub fn set_spill(
        &mut self,
        directory: Option<Arc<crate::spill::SpillDirectory>>,
        threshold_bytes: u64,
    ) {
        self.inner.set_spill(directory, threshold_bytes);
    }
}

impl Operator for SortAggregateOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        self.inner.next()
    }
}

// ---------------------------------------------------------------------------
// ParallelHashAggregateOperator
// ---------------------------------------------------------------------------

/// Whether an aggregate function can be combined across disjoint partitions
/// without changing the result. An explicit whitelist of the functions whose
/// accumulators implement an associative merge. Unknown names return false so
/// they take the serial path instead of being silently treated as parallel via
/// the create_accumulator catch-all. Kept in sync with the accumulators that
/// override Accumulator::supports_parallel_merge.
/// Whether an aggregate named in a plan can be computed as partials over
/// disjoint ranges and merged.
///
/// The accumulator itself answers, because it is the thing that implements
/// `merge` and knows whether combining partial states changes the result. A
/// second list of names here would be a copy of that knowledge, and the day
/// the two disagreed the plan would either lose parallelism it could have had
/// or call `merge` on an accumulator whose default body is `unreachable!`.
///
/// The probe accumulator is built and dropped; it holds no state before its
/// first update, so this costs one allocation at plan time.
pub fn aggregate_supports_parallel(function_name: &str) -> bool {
    // An unimplemented name has no accumulator of its own, and probing it
    // would land on the catch-all COUNT and answer for the wrong aggregate.
    // `validate_aggregates` rejects such a plan before it runs, so the only
    // honest answer here is that it is not something to parallelize
    if !is_supported_aggregate(function_name) {
        return false;
    }
    // Argument count only distinguishes COUNT(*) from COUNT(expr), and both
    // merge, so either probe answers for the name
    build_accumulator(function_name, 1).supports_parallel_merge()
}

/// Aggregates a contiguous page range into one partial group state. Each
/// parallel worker runs this over a disjoint range, then the operator merges
/// the partials.
async fn aggregate_page_range(
    ctx: Arc<ExecutionContext>,
    table_entry: Arc<TableEntry>,
    columns: Vec<LogicalColumn>,
    predicate: Option<BoundExpr>,
    group_by: Vec<BoundExpr>,
    aggregates: Vec<AggregateExpr>,
    input_schema: Vec<LogicalColumn>,
    claims: Arc<crate::operator::scan::PageClaims>,
) -> Result<GroupAccumulatorState> {
    let mut scanner =
        PageRangeScanner::claiming(&ctx, &table_entry, &columns, predicate.as_ref(), claims);
    let mut state = GroupAccumulatorState::new(&aggregates);
    while let Some(batch) = scanner.next_batch().await? {
        state.ingest(&batch, &group_by, &aggregates, &input_schema)?;
    }
    Ok(state)
}

/// Rows per group below which splitting a grouped aggregate across workers
/// costs more than it saves.
///
/// Every worker builds a group table of its own and the merge is one pass
/// per worker over as many groups as the data holds. Against a fold over
/// the rows that is cheap when rows greatly outnumber groups, and it is the
/// whole cost when they do not: a group per row leaves every worker holding
/// the entire table and the merge repeating the aggregation
const SPLIT_MIN_ROWS_PER_GROUP: u64 = 16;

/// Whether a grouped aggregate over a lake table is worth splitting.
///
/// The manifest records a distinct estimate per file per column, and their
/// sum is an upper bound on the table's distinct count because a value in
/// two files is counted twice. That is the conservative direction for this
/// question: a small bound proves the grouping is narrow, while a large one
/// only fails to prove it, and both answers then fall the safe way.
///
/// A grouping key that is not a plain column, or a column no file has an
/// estimate for, is not judged and does not split
pub(crate) fn worth_splitting(manifest: &zyron_lake::ManifestFile, group_by: &[BoundExpr]) -> bool {
    let rows: u64 = manifest.entries.iter().map(|e| e.row_count).sum();
    if rows == 0 {
        return false;
    }
    let mut distinct_upper: u64 = 1;
    for key in group_by {
        let BoundExpr::ColumnRef(cr) = key else {
            return false;
        };
        let column_id = cr.column_id.0 as u32;
        let mut column_upper: u64 = 0;
        for entry in &manifest.entries {
            match entry.stats_for(column_id).and_then(|s| s.ndv) {
                Some(ndv) => column_upper = column_upper.saturating_add(ndv),
                None => return false,
            }
        }
        // Two keys multiply into at most the product of their distinct
        // counts, capped at the rows that could carry them
        distinct_upper = distinct_upper.saturating_mul(column_upper).min(rows);
    }
    distinct_upper.saturating_mul(SPLIT_MIN_ROWS_PER_GROUP) <= rows
}

/// Folds one worker's share of a lake table's files into a local group
/// table.
///
/// The scan is rebuilt per worker rather than shared, because a scan owns
/// its decode buffers and its position in the file list. Building it costs
/// a manifest lookup, which is tens of nanoseconds, against a share of the
/// table it then reads alone
#[allow(clippy::too_many_arguments)]
async fn aggregate_lake_files(
    ctx: Arc<ExecutionContext>,
    table_id: TableId,
    columns: Vec<LogicalColumn>,
    predicate: Option<BoundExpr>,
    lowered: Option<zyron_lake::LakePredicate>,
    as_of: Option<zyron_planner::logical::AsOfTarget>,
    group_by: Vec<BoundExpr>,
    aggregates: Vec<AggregateExpr>,
    input_schema: Vec<LogicalColumn>,
    files: Arc<Vec<u64>>,
    cursor: Arc<std::sync::atomic::AtomicUsize>,
) -> Result<GroupAccumulatorState> {
    let mut scan = crate::operator::lake_scan::LakeScanOperator::new(
        ctx, table_id, columns, predicate, lowered, as_of,
    )?;
    scan.share_files(&files, cursor);
    let mut state = GroupAccumulatorState::new(&aggregates);
    while let Some(eb) = scan.next().await? {
        state.ingest(&eb.batch, &group_by, &aggregates, &input_schema)?;
    }
    Ok(state)
}

/// Grouped aggregation fused with a parallel lake scan. Each worker reads a
/// disjoint set of the table's data files and builds a local group table,
/// and the partials merge exactly as the heap's do.
///
/// A lake table's files are already the unit the scan reads one at a time,
/// so they split without any of the range arithmetic a heap needs, and a
/// file is never read by two workers
pub struct ParallelLakeAggregateOperator {
    ctx: Arc<ExecutionContext>,
    table_id: TableId,
    columns: Vec<LogicalColumn>,
    predicate: Option<BoundExpr>,
    lowered: Option<zyron_lake::LakePredicate>,
    as_of: Option<zyron_planner::logical::AsOfTarget>,
    group_by: Vec<BoundExpr>,
    aggregates: Vec<AggregateExpr>,
    input_schema: Vec<LogicalColumn>,
    output_schema: Vec<LogicalColumn>,
    /// The pruned file set, decided by the caller so pruning is not
    /// repeated once here and again in every worker
    files: Vec<u64>,
    /// Rows those files hold by their manifest entries, which sizes the
    /// fan-out
    rows: u64,
    finished: bool,
    result: Option<DataBatch>,
    output_cursor: usize,
}

/// Rows of a lake table below which one more worker is not worth waking,
/// for an aggregate with no grouping.
///
/// A decoded lake row costs a nanosecond or two to read and fold into one
/// accumulator, so fifty thousand of them are around a hundred
/// microseconds of work, enough that the wake and the start of a pool
/// thread are a small part of it
const LAKE_MIN_ROWS_PER_WORKER: u64 = 50_000;

/// The same floor for a grouped aggregate, whose rows each hash their key
/// and find their group, several times the cost of a plain fold. Ten
/// thousand rows is the same hundred microseconds of work
const LAKE_MIN_ROWS_PER_WORKER_GROUPED: u64 = 10_000;

impl ParallelLakeAggregateOperator {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: Arc<ExecutionContext>,
        table_id: TableId,
        columns: Vec<LogicalColumn>,
        predicate: Option<BoundExpr>,
        lowered: Option<zyron_lake::LakePredicate>,
        as_of: Option<zyron_planner::logical::AsOfTarget>,
        group_by: Vec<BoundExpr>,
        aggregates: Vec<AggregateExpr>,
        input_schema: Vec<LogicalColumn>,
        output_schema: Vec<LogicalColumn>,
        files: Vec<u64>,
        rows: u64,
    ) -> Self {
        Self {
            ctx,
            table_id,
            columns,
            predicate,
            lowered,
            as_of,
            group_by,
            aggregates,
            input_schema,
            output_schema,
            files,
            rows,
            finished: false,
            result: None,
            output_cursor: 0,
        }
    }

    async fn materialize(&mut self) -> Result<()> {
        validate_aggregates(&self.aggregates)?;
        let files = std::mem::take(&mut self.files);
        if files.is_empty() {
            self.finished = true;
            return Ok(());
        }

        // The file count bounds the split the data supports, at a floor of
        // rows per worker below which waking one costs more than its share
        // of the read, and the pool bounds what the machine can spare. The
        // grant is held until every worker has been joined. Workers claim
        // files from one cursor, so the read ends when the files run out
        // rather than when the slowest worker finishes its share
        let floor = if self.group_by.is_empty() {
            LAKE_MIN_ROWS_PER_WORKER
        } else {
            LAKE_MIN_ROWS_PER_WORKER_GROUPED
        };
        let natural_workers = (self.rows.div_ceil(floor).max(1) as usize).min(files.len());
        let grant = crate::parallel_pool::reserve(natural_workers);
        let num_workers = grant.workers().min(natural_workers).max(1);
        let files = Arc::new(files);
        let cursor = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let mut merged = GroupAccumulatorState::new(&self.aggregates);
        if num_workers <= 1 {
            // One worker is this thread. Waking a pool thread to do what
            // the caller can do itself costs the wake and buys nothing
            drop(grant);
            let state = aggregate_lake_files(
                self.ctx.clone(),
                self.table_id,
                self.columns.clone(),
                self.predicate.clone(),
                self.lowered.clone(),
                self.as_of.clone(),
                self.group_by.clone(),
                self.aggregates.clone(),
                self.input_schema.clone(),
                files,
                cursor,
            )
            .await?;
            merged.merge(state)?;
        } else {
            let mut handles = Vec::with_capacity(num_workers);
            for _ in 0..num_workers {
                handles.push(crate::parallel_pool::spawn(aggregate_lake_files(
                    self.ctx.clone(),
                    self.table_id,
                    self.columns.clone(),
                    self.predicate.clone(),
                    self.lowered.clone(),
                    self.as_of.clone(),
                    self.group_by.clone(),
                    self.aggregates.clone(),
                    self.input_schema.clone(),
                    Arc::clone(&files),
                    Arc::clone(&cursor),
                )));
            }
            for handle in handles {
                let state = handle.await.map_err(|e| {
                    ZyronError::ExecutionError(format!(
                        "parallel lake aggregate worker failed: {e}"
                    ))
                })??;
                merged.merge(state)?;
            }
            drop(grant);
        }

        if merged.num_groups == 0 {
            self.finished = true;
            return Ok(());
        }

        self.result = Some(finalize_groups(
            &merged,
            self.group_by.len(),
            &self.output_schema,
        )?);
        Ok(())
    }
}

impl Operator for ParallelLakeAggregateOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            if self.finished {
                return Ok(None);
            }
            if self.result.is_none() {
                self.materialize().await?;
                if self.finished {
                    return Ok(None);
                }
            }
            let Some(result) = self.result.as_ref() else {
                self.finished = true;
                return Ok(None);
            };
            if self.output_cursor >= result.num_rows {
                self.finished = true;
                return Ok(None);
            }
            let remaining = result.num_rows - self.output_cursor;
            let chunk = remaining.min(crate::batch::BATCH_SIZE);
            let batch = result.slice(self.output_cursor, chunk);
            self.output_cursor += chunk;
            Ok(Some(ExecutionBatch::new(batch)))
        })
    }
}

/// Grouped aggregation fused with a parallel heap scan. Each worker scans a
/// disjoint page range and builds a local group table; the operator merges the
/// partials into the final result. Used when the child is a plain heap scan
/// large enough for parallelism and every aggregate supports a parallel merge.
pub struct ParallelHashAggregateOperator {
    ctx: Arc<ExecutionContext>,
    table_id: TableId,
    columns: Vec<LogicalColumn>,
    predicate: Option<BoundExpr>,
    group_by: Vec<BoundExpr>,
    aggregates: Vec<AggregateExpr>,
    input_schema: Vec<LogicalColumn>,
    output_schema: Vec<LogicalColumn>,
    finished: bool,
    result: Option<DataBatch>,
    output_cursor: usize,
}

impl ParallelHashAggregateOperator {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: Arc<ExecutionContext>,
        table_id: TableId,
        columns: Vec<LogicalColumn>,
        predicate: Option<BoundExpr>,
        group_by: Vec<BoundExpr>,
        aggregates: Vec<AggregateExpr>,
        input_schema: Vec<LogicalColumn>,
        output_schema: Vec<LogicalColumn>,
    ) -> Self {
        Self {
            ctx,
            table_id,
            columns,
            predicate,
            group_by,
            aggregates,
            input_schema,
            output_schema,
            finished: false,
            result: None,
            output_cursor: 0,
        }
    }

    async fn materialize(&mut self) -> Result<()> {
        validate_aggregates(&self.aggregates)?;
        let table_entry = self.ctx.get_table_entry(self.table_id)?;
        let num_pages = self
            .ctx
            .get_heap_file(self.table_id)
            .await?
            .num_pages_cached() as u64;
        // One scan, however many workers divide it. Each worker's scanner folds
        // in its own row and byte totals.
        if let Some(stats) = self.ctx.table_io_stats_for(self.table_id.0) {
            stats.record_seq_scan();
        }

        // The pages bound the split the data supports, at a floor of work
        // per worker, and the pool bounds what the machine can spare for it
        // right now. The grant is held until every worker has been joined.
        // Workers claim runs of pages from one cursor, so the scan ends
        // when the pages run out rather than when the slowest worker does
        let natural_workers = crate::operator::scan::parallel_workers_for_pages(num_pages);
        let grant = crate::parallel_pool::reserve(natural_workers);
        let num_workers = grant.workers().min(natural_workers).max(1);
        let claims = Arc::new(crate::operator::scan::PageClaims::new(0, num_pages));

        let mut merged = GroupAccumulatorState::new(&self.aggregates);
        if num_workers <= 1 {
            // One worker is this thread. Waking a pool thread to do what
            // the caller can do itself costs the wake and buys nothing
            drop(grant);
            let state = aggregate_page_range(
                self.ctx.clone(),
                table_entry.clone(),
                self.columns.clone(),
                self.predicate.clone(),
                self.group_by.clone(),
                self.aggregates.clone(),
                self.input_schema.clone(),
                claims,
            )
            .await?;
            merged.merge(state)?;
        } else {
            let mut handles = Vec::with_capacity(num_workers);
            for _ in 0..num_workers {
                handles.push(crate::parallel_pool::spawn(aggregate_page_range(
                    self.ctx.clone(),
                    table_entry.clone(),
                    self.columns.clone(),
                    self.predicate.clone(),
                    self.group_by.clone(),
                    self.aggregates.clone(),
                    self.input_schema.clone(),
                    Arc::clone(&claims),
                )));
            }
            for handle in handles {
                let state = handle.await.map_err(|e| {
                    ZyronError::ExecutionError(format!("parallel aggregate worker failed: {e}"))
                })??;
                merged.merge(state)?;
            }
            drop(grant);
        }

        if merged.num_groups == 0 {
            self.finished = true;
            return Ok(());
        }

        self.result = Some(finalize_groups(
            &merged,
            self.group_by.len(),
            &self.output_schema,
        )?);
        Ok(())
    }
}

impl Operator for ParallelHashAggregateOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            if self.finished {
                return Ok(None);
            }

            if self.result.is_none() && self.output_cursor == 0 {
                self.materialize().await?;
            }

            let Some(ref result) = self.result else {
                self.finished = true;
                return Ok(None);
            };

            if self.output_cursor >= result.num_rows {
                self.finished = true;
                return Ok(None);
            }

            let remaining = result.num_rows - self.output_cursor;
            let chunk = remaining.min(crate::batch::BATCH_SIZE);
            let batch = result.slice(self.output_cursor, chunk);
            self.output_cursor += chunk;

            Ok(Some(ExecutionBatch::new(batch)))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A manifest of `files` data files, each holding `rows_per_file` rows
    /// and reporting `ndv_per_file` distinct values in column zero
    fn manifest_with_ndv(
        files: usize,
        rows_per_file: u64,
        ndv_per_file: Option<u64>,
    ) -> zyron_lake::ManifestFile {
        let entries = (0..files)
            .map(|i| zyron_lake::PartitionEntry {
                partition_id: i as u64,
                size_bytes: rows_per_file * 8,
                row_count: rows_per_file,
                added_version: 1,
                cluster_spec_id: 0,
                column_stats: std::sync::Arc::new(vec![zyron_lake::ColumnStatsEntry {
                    column_id: 0,
                    bounds: zyron_lake::ColumnBounds {
                        min: None,
                        max: None,
                        null_count: 0,
                        row_count: rows_per_file,
                    },
                    bloom: None,
                    ndv: ndv_per_file,
                    size_bytes: None,
                    sum: None,
                }]),
                delete_predicate_ids: Vec::new(),
            })
            .collect();
        zyron_lake::ManifestFile {
            snapshot_id: 1,
            parent_snapshot_id: 0,
            timestamp_us: 0,
            schema: zyron_lake::LakeSchema {
                columns: Vec::new(),
                next_column_id: 1,
                derived: Vec::new(),
                schema_id: 0,
            },
            cluster_spec: zyron_lake::ClusterSpec::none(),
            entries,
            delete_predicates: Vec::new(),
            properties: std::collections::BTreeMap::new(),
            indexes: Vec::new(),
            index_files: Vec::new(),
        }
    }

    fn key_on_column_zero() -> Vec<BoundExpr> {
        vec![BoundExpr::ColumnRef(zyron_planner::binder::ColumnRef {
            table_idx: 0,
            column_id: zyron_catalog::ColumnId(0),
            type_id: TypeId::Int64,
            nullable: true,
            fractional_digits: None,
        })]
    }

    /// The split pays for itself when many rows share a group and not when
    /// each row is its own, which is the whole judgement the gate makes
    #[test]
    fn test_a_grouping_splits_only_when_rows_outnumber_groups() {
        let narrow = manifest_with_ndv(4, 500, Some(8));
        assert!(
            worth_splitting(&narrow, &key_on_column_zero()),
            "thirty two distinct values at most over two thousand rows is worth dividing"
        );

        // A distinct value per row: every worker ends up holding the whole
        // table and the merge repeats the aggregation
        let wide = manifest_with_ndv(4, 500, Some(500));
        assert!(!worth_splitting(&wide, &key_on_column_zero()));

        // Right at the boundary, where each group holds exactly the rows the
        // threshold asks for
        let edge = manifest_with_ndv(1, 1_600, Some(100));
        assert!(worth_splitting(&edge, &key_on_column_zero()));
        let past_edge = manifest_with_ndv(1, 1_600, Some(101));
        assert!(!worth_splitting(&past_edge, &key_on_column_zero()));
    }

    /// Nothing to judge from means no split, rather than a guess
    #[test]
    fn test_a_grouping_with_no_estimate_is_not_split() {
        let no_ndv = manifest_with_ndv(4, 500, None);
        assert!(
            !worth_splitting(&no_ndv, &key_on_column_zero()),
            "a file with no distinct estimate leaves the question unanswered"
        );

        let empty = manifest_with_ndv(0, 0, Some(1));
        assert!(!worth_splitting(&empty, &key_on_column_zero()));

        // A key that is not a plain column has no per column estimate
        let narrow = manifest_with_ndv(4, 500, Some(8));
        let computed = vec![BoundExpr::Literal {
            value: zyron_parser::ast::LiteralValue::Integer(1),
            type_id: TypeId::Int64,
        }];
        assert!(!worth_splitting(&narrow, &computed));
    }

    /// The accumulator decides whether an aggregate can be split across
    /// ranges, so adding one that merges is enough to make plans parallelize
    /// it and there is no second list to keep in step
    #[test]
    fn test_parallel_eligibility_comes_from_the_accumulator() {
        for name in ["count", "sum", "avg", "min", "max", "COUNT", "Sum"] {
            assert!(
                aggregate_supports_parallel(name),
                "{name} merges, so a plan may split it"
            );
            assert!(build_accumulator(name, 1).supports_parallel_merge());
        }

        // An accumulator whose default `merge` is `unreachable!` must never be
        // reported as splittable, or a parallel plan would panic on it
        for name in ["first", "last", "stddev", "variance"] {
            assert!(
                is_supported_aggregate(name),
                "{name} is a real aggregate, just not a splittable one"
            );
            assert_eq!(
                aggregate_supports_parallel(name),
                build_accumulator(name, 1).supports_parallel_merge(),
                "{name}: the plan-time answer must be the accumulator's answer"
            );
            assert!(
                !aggregate_supports_parallel(name),
                "{name} defines no parallel combine"
            );
        }

        // A name with no implementation resolves to the catch-all COUNT
        // accumulator, which does merge. Answering from that probe would be
        // answering for the wrong aggregate, so the guard runs first
        for name in ["string_agg", "array_agg", "no_such_aggregate"] {
            assert!(!is_supported_aggregate(name));
            assert!(
                build_accumulator(name, 1).supports_parallel_merge(),
                "{name} lands on the COUNT catch-all, which is why the guard exists"
            );
            assert!(
                !aggregate_supports_parallel(name),
                "{name} has no implementation, so it is not something to split"
            );
        }
    }

    use super::*;
    use zyron_catalog::ColumnId;
    use zyron_planner::binder::ColumnRef;

    // DISTINCT aggregates must fold each value only once.
    fn one_arg_agg(name: &str, distinct: bool) -> AggregateExpr {
        AggregateExpr {
            function_name: name.to_string(),
            args: vec![col_ref(0, TypeId::Int64)],
            distinct,
            return_type: TypeId::Int64,
            uda: None,
        }
    }

    #[test]
    fn distinct_aggregates_dedup() {
        let vals = [1i64, 1, 2, 3, 3];

        let mut count_distinct = create_accumulator(&one_arg_agg("count", true));
        let mut sum_distinct = create_accumulator(&one_arg_agg("sum", true));
        let mut count_all = create_accumulator(&one_arg_agg("count", false));
        for &v in &vals {
            count_distinct.update(&ScalarValue::Int64(v));
            sum_distinct.update(&ScalarValue::Int64(v));
            count_all.update(&ScalarValue::Int64(v));
        }
        // 3 distinct values {1,2,3}; sum of distinct = 6; non-distinct counts all 5.
        // SUM accumulates integers exactly, so the natural finalize is an i128;
        // finalize_groups coerces it to the aggregate's declared output type.
        assert_eq!(count_distinct.finalize(), ScalarValue::Int64(3));
        assert_eq!(sum_distinct.finalize(), ScalarValue::Int128(6));
        assert_eq!(count_all.finalize(), ScalarValue::Int64(5));

        // NULLs are ignored by distinct too.
        let mut count_with_nulls = create_accumulator(&one_arg_agg("count", true));
        count_with_nulls.update(&ScalarValue::Null);
        count_with_nulls.update(&ScalarValue::Int64(7));
        count_with_nulls.update(&ScalarValue::Null);
        assert_eq!(count_with_nulls.finalize(), ScalarValue::Int64(1));
    }

    fn int_col(values: Vec<i64>) -> Column {
        Column::new(ColumnData::Int64(values), TypeId::Int64)
    }

    fn batch(keys: Vec<i64>, vals: Vec<i64>) -> DataBatch {
        DataBatch::new(vec![int_col(keys), int_col(vals)])
    }

    fn col_ref(column_id: u16, type_id: TypeId) -> BoundExpr {
        BoundExpr::ColumnRef(ColumnRef {
            table_idx: 0,
            column_id: ColumnId(column_id),
            type_id,
            nullable: false,
            fractional_digits: None,
        })
    }

    fn count_star() -> AggregateExpr {
        AggregateExpr {
            function_name: "count".into(),
            args: vec![],
            distinct: false,
            return_type: TypeId::Int64,
            uda: None,
        }
    }

    fn sum_of(column_id: u16) -> AggregateExpr {
        AggregateExpr {
            function_name: "sum".into(),
            args: vec![col_ref(column_id, TypeId::Int64)],
            distinct: false,
            return_type: TypeId::Float64,
            uda: None,
        }
    }

    fn input_schema() -> Vec<LogicalColumn> {
        vec![
            LogicalColumn {
                table_idx: Some(0),
                column_id: ColumnId(0),
                name: "k".into(),
                type_id: TypeId::Int64,
                nullable: false,
                fractional_digits: None,
            },
            LogicalColumn {
                table_idx: Some(0),
                column_id: ColumnId(1),
                name: "v".into(),
                type_id: TypeId::Int64,
                nullable: false,
                fractional_digits: None,
            },
        ]
    }

    fn output_schema_for(
        group_by: &[BoundExpr],
        aggregates: &[AggregateExpr],
    ) -> Vec<LogicalColumn> {
        let mut schema = Vec::new();
        for (i, _) in group_by.iter().enumerate() {
            schema.push(LogicalColumn {
                table_idx: Some(0),
                column_id: ColumnId(i as u16),
                name: format!("g{i}"),
                type_id: TypeId::Int64,
                nullable: false,
                fractional_digits: None,
            });
        }
        for (i, agg) in aggregates.iter().enumerate() {
            schema.push(LogicalColumn {
                table_idx: None,
                column_id: ColumnId(100 + i as u16),
                name: agg.function_name.clone(),
                type_id: agg.return_type,
                nullable: true,
                fractional_digits: None,
            });
        }
        schema
    }

    // Reads a finalized GROUP BY result into a key -> (count, sum) map so the
    // comparison ignores group emission order, which differs between the serial
    // and merged paths.
    fn result_map(b: &DataBatch) -> std::collections::BTreeMap<i64, (i64, i64)> {
        let ColumnData::Int64(keys) = &b.columns[0].data else {
            panic!("key column is not Int64");
        };
        let ColumnData::Int64(counts) = &b.columns[1].data else {
            panic!("count column is not Int64");
        };
        let ColumnData::Float64(sums) = &b.columns[2].data else {
            panic!("sum column is not Float64");
        };
        let mut map = std::collections::BTreeMap::new();
        for i in 0..b.num_rows {
            map.insert(keys[i], (counts[i], sums[i] as i64));
        }
        map
    }

    // Partial aggregation merged across partitions must equal aggregating the
    // same rows in one partition. Guards the parallel aggregate's merge path,
    // which the end-to-end bench exercises but does not value-check.
    #[test]
    fn parallel_merge_matches_serial() {
        let group_by = vec![col_ref(0, TypeId::Int64)];
        let aggregates = vec![count_star(), sum_of(1)];
        let schema = input_schema();
        let out_schema = output_schema_for(&group_by, &aggregates);

        let batches = [
            batch(vec![1, 2, 1], vec![10, 20, 30]),
            batch(vec![3, 2, 1], vec![5, 15, 25]),
            batch(vec![3, 3, 2], vec![1, 2, 3]),
        ];

        // Serial: one partition over every batch.
        let mut serial = GroupAccumulatorState::new(&aggregates);
        for b in &batches {
            serial.ingest(b, &group_by, &aggregates, &schema).unwrap();
        }
        let serial_out = finalize_groups(&serial, 1, &out_schema).expect("finalize serial");

        // Parallel: one partition per batch, then merge.
        let mut merged = GroupAccumulatorState::new(&aggregates);
        for b in &batches {
            let mut part = GroupAccumulatorState::new(&aggregates);
            part.ingest(b, &group_by, &aggregates, &schema).unwrap();
            merged.merge(part).expect("merge");
        }
        let merged_out = finalize_groups(&merged, 1, &out_schema).expect("finalize merged");

        assert_eq!(result_map(&serial_out), result_map(&merged_out));
        // Known answer: key 1 -> count 3 sum 65, key 2 -> 3 38, key 3 -> 3 8.
        let m = result_map(&merged_out);
        assert_eq!(m[&1], (3, 65));
        assert_eq!(m[&2], (3, 38));
        assert_eq!(m[&3], (3, 8));
    }
}

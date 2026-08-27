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
use crate::compute::{self, PreHashMap};
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

    fn add_count_checked(&mut self, n: usize) -> Result<()> {
        self.add_count(n);
        Ok(())
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
        _ => Box::new(CountAccumulator { count: 0 }),
    }
}

/// Whether `create_accumulator` has a real implementation for this function.
/// MUST list the same set as the matched arms above; the catch-all there
/// returns a COUNT accumulator, so callers validate up front and error rather
/// than silently computing a COUNT for an unimplemented aggregate.
pub(crate) fn is_supported_aggregate(name: &str) -> bool {
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
        let mut state = GroupAccumulatorState::new();
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
            &state.group_key_store,
            &state.group_accumulators,
            state.num_groups,
            self.group_by.len(),
            &self.output_schema,
        )?))
    }

    async fn materialize(&mut self) -> Result<()> {
        validate_aggregates(&self.aggregates)?;
        let num_group_cols = self.group_by.len();

        // Materialized group state. For grouped aggregates this is built up by
        // hashing keys and resolving collisions; for global aggregates the
        // store stays empty and only `group_accumulators[0]` is touched.
        let mut group_key_store: Vec<Column> = Vec::new();
        let mut group_accumulators: Vec<Vec<Box<dyn Accumulator>>> = Vec::new();
        let num_groups;

        // Global-aggregate fast path: no GROUP BY means exactly one output row.
        // Pre-create the single accumulator vector and skip every hashing,
        // group-store, and collision-check step in the inner loop.
        if num_group_cols == 0 {
            let accs: Vec<Box<dyn Accumulator>> =
                self.aggregates.iter().map(create_accumulator).collect();
            group_accumulators.push(accs);
            num_groups = 1;

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
                        let agg_arg_cols: Vec<Option<Cow<'_, Column>>> = self
                            .aggregates
                            .iter()
                            .map(|agg| {
                                if agg.args.is_empty() {
                                    Ok(None)
                                } else {
                                    Ok(Some(evaluate_borrowed(
                                        &agg.args[0],
                                        batch,
                                        &self.input_schema,
                                        &[],
                                    )?))
                                }
                            })
                            .collect::<Result<Vec<_>>>()?;

                        let accs = &mut group_accumulators[0];
                        for (i, acc) in accs.iter_mut().enumerate() {
                            match &agg_arg_cols[i] {
                                // COUNT(*)-style: fold the whole batch in one
                                // add instead of a per-row loop.
                                None => acc.add_count_checked(num_rows)?,
                                // Aggregates over a column still walk rows so
                                // nulls and per-value math are handled.
                                Some(col) => {
                                    let c = col.as_ref();
                                    for row in 0..num_rows {
                                        acc.update_typed_checked(c, row)?;
                                    }
                                }
                            }
                        }
                    }
                    None => break,
                }
            }
        } else {
            // Grouped aggregation runs through the shared GroupAccumulatorState
            // so the serial path and the parallel partial-aggregate path use one
            // grouping and one find-or-create implementation.
            let mut state = GroupAccumulatorState::new();
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
            group_key_store = state.group_key_store;
            group_accumulators = state.group_accumulators;
            num_groups = state.num_groups;
        }

        if num_groups == 0 {
            return Ok(());
        }

        self.result = Some(finalize_groups(
            &group_key_store,
            &group_accumulators,
            num_groups,
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
    group_key_store: &[Column],
    group_accumulators: &[Vec<Box<dyn Accumulator>>],
    num_groups: usize,
    num_group_cols: usize,
    output_schema: &[LogicalColumn],
) -> Result<DataBatch> {
    let mut col_builders: Vec<(ColumnData, NullBitmap, TypeId, Option<u8>)> =
        Vec::with_capacity(output_schema.len());
    for col_def in output_schema {
        col_builders.push((
            ColumnData::with_capacity(col_def.type_id, num_groups),
            NullBitmap::empty(),
            col_def.type_id,
            col_def.fractional_digits,
        ));
    }

    for gidx in 0..num_groups {
        for i in 0..num_group_cols {
            let (data, nulls, _, _) = &mut col_builders[i];
            let store_col = &group_key_store[i];
            nulls.push(store_col.is_null(gidx));
            data.push_from(&store_col.data, gidx);
        }
        for (i, acc) in group_accumulators[gidx].iter().enumerate() {
            let raw = acc.finalize_checked()?;
            let (data, nulls, ty, _) = &mut col_builders[num_group_cols + i];
            let val = coerce_aggregate_scalar(raw, *ty)?;
            nulls.push(val.is_null());
            data.push_scalar(&val);
        }
    }

    // The declared fractional digits ride along so a decimal aggregate's
    // output column compares and renders on its value scale
    let columns: Vec<Column> = col_builders
        .into_iter()
        .map(|(data, nulls, type_id, fractional_digits)| {
            Column::with_nulls_ts(data, nulls, type_id, fractional_digits)
        })
        .collect();

    Ok(DataBatch::new(columns))
}

/// Partial grouped-aggregation state for one partition. The parallel aggregate
/// builds one per worker over a disjoint page range, then merges them; the
/// serial aggregate uses a single instance over all input. One grouping and
/// find-or-create implementation backs both so they cannot diverge.
struct GroupAccumulatorState {
    group_key_store: Vec<Column>,
    hash_to_groups: PreHashMap<u64, Vec<usize>>,
    group_accumulators: Vec<Vec<Box<dyn Accumulator>>>,
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
}

/// Bytes one group costs beyond its key: the hash entry, its place in the
/// collision list, and the vector of accumulator boxes.
const GROUP_OVERHEAD_BYTES: u64 = 64;

/// Bytes one accumulator costs: the box, and the state of the widest
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
    fn new() -> Self {
        Self {
            group_key_store: Vec::new(),
            hash_to_groups: PreHashMap::default(),
            group_accumulators: Vec::new(),
            num_groups: 0,
            key_bytes: 0,
            frozen: false,
            unresident: Vec::new(),
        }
    }

    /// What the table holds, near enough to decide when to stop growing.
    ///
    /// The keys are counted exactly. The accumulators are estimated, because
    /// their state sits behind a trait object with no size to ask for, and a
    /// count times a fixed figure is right for every accumulator whose state
    /// is fixed, which is all of them but DISTINCT and a user-defined one.
    /// Those two keep growing after the table is frozen, and there is nothing
    /// to do about it without writing accumulator state to disk, which would
    /// need a merge that those two do not have.
    fn resident_bytes(&self) -> u64 {
        let per_group = self
            .group_accumulators
            .first()
            .map(|a| a.len())
            .unwrap_or(0) as u64;
        self.key_bytes
            + self.num_groups as u64 * (GROUP_OVERHEAD_BYTES + per_group * ACCUMULATOR_BYTES)
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
        let agg_arg_cols: Vec<Option<Cow<'_, Column>>> = aggregates
            .iter()
            .map(|agg| {
                if agg.args.is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(evaluate_borrowed(
                        &agg.args[0],
                        batch,
                        input_schema,
                        &[],
                    )?))
                }
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
            hash_to_groups,
            group_accumulators,
            num_groups,
            key_bytes,
            frozen,
            unresident,
        } = self;

        // Taken so the routing pass below can hand the batch to the writer
        // while the table's own fields are still borrowed
        let mut routed = std::mem::take(unresident);
        routed.clear();

        for row in 0..num_rows {
            let gidx = if *frozen {
                match find_group(
                    hash_to_groups,
                    group_key_store,
                    &group_refs,
                    row,
                    hashes[row],
                ) {
                    Some(gidx) => gidx,
                    None => {
                        // A group this table has no room for. Its rows go to
                        // a partition, all of them, so the group is whole
                        // wherever it ends up
                        routed.push(row as u32);
                        continue;
                    }
                }
            } else {
                find_or_create_group(
                    hash_to_groups,
                    group_key_store,
                    group_accumulators,
                    num_groups,
                    key_bytes,
                    &group_refs,
                    row,
                    hashes[row],
                    || aggregates.iter().map(create_accumulator).collect(),
                )
            };
            let accs = &mut group_accumulators[gidx];
            for (i, acc) in accs.iter_mut().enumerate() {
                match &agg_arg_cols[i] {
                    Some(col) => acc.update_typed_checked(col.as_ref(), row)?,
                    None => acc.update_checked(&ScalarValue::Int64(1))?,
                }
            }
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
    fn merge(&mut self, other: GroupAccumulatorState, aggregates: &[AggregateExpr]) {
        if other.num_groups == 0 {
            return;
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
            hash_to_groups,
            group_accumulators,
            num_groups,
            key_bytes,
            ..
        } = self;

        for ogidx in 0..other.num_groups {
            let gidx = find_or_create_group(
                hash_to_groups,
                group_key_store,
                group_accumulators,
                num_groups,
                key_bytes,
                &other_refs,
                ogidx,
                hashes[ogidx],
                || aggregates.iter().map(create_accumulator).collect(),
            );
            let dst = &mut group_accumulators[gidx];
            for (i, acc) in dst.iter_mut().enumerate() {
                acc.merge(other.group_accumulators[ogidx][i].as_ref());
            }
        }
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
    hash_to_groups: &PreHashMap<u64, Vec<usize>>,
    group_key_store: &[Column],
    key_cols: &[&Column],
    row: usize,
    hash: u64,
) -> Option<usize> {
    let candidates = hash_to_groups.get(&hash)?;
    candidates
        .iter()
        .copied()
        .find(|&gidx| group_key_equals(group_key_store, key_cols, row, gidx))
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
/// creates it, copying the key into the store and building fresh
/// accumulators. The disjoint `&mut` parameters let one implementation serve
/// both batch ingest and partition merge without a borrow conflict.
#[allow(clippy::too_many_arguments)]
fn find_or_create_group(
    hash_to_groups: &mut PreHashMap<u64, Vec<usize>>,
    group_key_store: &mut [Column],
    group_accumulators: &mut Vec<Vec<Box<dyn Accumulator>>>,
    num_groups: &mut usize,
    key_bytes: &mut u64,
    key_cols: &[&Column],
    row: usize,
    hash: u64,
    make_accs: impl Fn() -> Vec<Box<dyn Accumulator>>,
) -> usize {
    let candidates = hash_to_groups.entry(hash).or_default();
    for &gidx in candidates.iter() {
        if group_key_equals(group_key_store, key_cols, row, gidx) {
            return gidx;
        }
    }
    let gidx = *num_groups;
    *num_groups += 1;
    candidates.push(gidx);
    for (ci, kc) in key_cols.iter().enumerate() {
        group_key_store[ci].push_row_from(kc, row);
        *key_bytes += key_row_bytes(kc, row);
    }
    group_accumulators.push(make_accs());
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
    start_page: u64,
    end_page: u64,
) -> Result<GroupAccumulatorState> {
    let mut scanner = PageRangeScanner::new(
        &ctx,
        &table_entry,
        &columns,
        predicate.as_ref(),
        start_page,
        end_page,
    );
    let mut state = GroupAccumulatorState::new();
    while let Some(batch) = scanner.next_batch().await? {
        state.ingest(&batch, &group_by, &aggregates, &input_schema)?;
    }
    Ok(state)
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

        // The pages bound the split the data supports; the pool bounds what
        // the machine can spare for it right now. The grant is held until
        // every worker has been joined
        let natural_workers = num_pages.max(1) as usize;
        let grant = crate::parallel_pool::reserve(natural_workers);
        let num_workers = grant.workers().min(natural_workers).max(1);
        let pages_per_worker = num_pages.div_ceil(num_workers as u64);

        let mut handles = Vec::with_capacity(num_workers);
        for worker_id in 0..num_workers {
            let start_page = worker_id as u64 * pages_per_worker;
            let end_page = ((worker_id as u64 + 1) * pages_per_worker).min(num_pages);
            if start_page >= end_page {
                continue;
            }
            handles.push(crate::parallel_pool::spawn(aggregate_page_range(
                self.ctx.clone(),
                table_entry.clone(),
                self.columns.clone(),
                self.predicate.clone(),
                self.group_by.clone(),
                self.aggregates.clone(),
                self.input_schema.clone(),
                start_page,
                end_page,
            )));
        }

        let mut merged = GroupAccumulatorState::new();
        for handle in handles {
            let state = handle.await.map_err(|e| {
                ZyronError::ExecutionError(format!("parallel aggregate worker failed: {e}"))
            })??;
            merged.merge(state, &self.aggregates);
        }
        drop(grant);

        if merged.num_groups == 0 {
            self.finished = true;
            return Ok(());
        }

        self.result = Some(finalize_groups(
            &merged.group_key_store,
            &merged.group_accumulators,
            merged.num_groups,
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
        let mut serial = GroupAccumulatorState::new();
        for b in &batches {
            serial.ingest(b, &group_by, &aggregates, &schema).unwrap();
        }
        let serial_out = finalize_groups(
            &serial.group_key_store,
            &serial.group_accumulators,
            serial.num_groups,
            1,
            &out_schema,
        )
        .expect("finalize serial");

        // Parallel: one partition per batch, then merge.
        let mut merged = GroupAccumulatorState::new();
        for b in &batches {
            let mut part = GroupAccumulatorState::new();
            part.ingest(b, &group_by, &aggregates, &schema).unwrap();
            merged.merge(part, &aggregates);
        }
        let merged_out = finalize_groups(
            &merged.group_key_store,
            &merged.group_accumulators,
            merged.num_groups,
            1,
            &out_schema,
        )
        .expect("finalize merged");

        assert_eq!(result_map(&serial_out), result_map(&merged_out));
        // Known answer: key 1 -> count 3 sum 65, key 2 -> 3 38, key 3 -> 3 8.
        let m = result_map(&merged_out);
        assert_eq!(m[&1], (3, 65));
        assert_eq!(m[&2], (3, 38));
        assert_eq!(m[&3], (3, 8));
    }
}

//! Aggregation operators and accumulator infrastructure.
//!
//! Provides hash-based and sort-based aggregation with pluggable accumulators
//! for COUNT, SUM, AVG, MIN, MAX. Uses typed column access to avoid
//! ScalarValue allocation in hot paths.

use std::borrow::Cow;
use std::sync::Arc;

use zyron_catalog::{TableEntry, TableId};
use zyron_common::{Result, TypeId, ZyronError};
use zyron_planner::binder::BoundExpr;
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

trait Accumulator: std::any::Any + Send {
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
    fn supports_parallel_merge(&self) -> bool {
        true
    }
}

struct SumAccumulator {
    sum: f64,
    has_value: bool,
}

impl Accumulator for SumAccumulator {
    fn update(&mut self, value: &ScalarValue) {
        if let Some(f) = value.to_f64() {
            self.sum += f;
            self.has_value = true;
        }
    }
    fn update_typed(&mut self, col: &Column, row: usize) {
        if col.is_null(row) {
            return;
        }
        match &col.data {
            ColumnData::Int64(v) => {
                self.sum += v[row] as f64;
                self.has_value = true;
            }
            ColumnData::Float64(v) => {
                self.sum += v[row];
                self.has_value = true;
            }
            ColumnData::Int32(v) => {
                self.sum += v[row] as f64;
                self.has_value = true;
            }
            ColumnData::Float32(v) => {
                self.sum += v[row] as f64;
                self.has_value = true;
            }
            _ => self.update(&col.get_scalar(row)),
        }
    }
    fn finalize(&self) -> ScalarValue {
        if self.has_value {
            ScalarValue::Float64(self.sum)
        } else {
            ScalarValue::Null
        }
    }
    fn merge(&mut self, other: &dyn Accumulator) {
        let o = merge_peer::<SumAccumulator>(other);
        if o.has_value {
            self.sum += o.sum;
            self.has_value = true;
        }
    }
    fn supports_parallel_merge(&self) -> bool {
        true
    }
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
    fn supports_parallel_merge(&self) -> bool {
        true
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
                if value
                    .partial_cmp(current)
                    .is_some_and(|o| o == std::cmp::Ordering::Less)
                {
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
                if value
                    .partial_cmp(current)
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

impl Accumulator for DistinctAccumulator {
    fn update(&mut self, value: &ScalarValue) {
        if !value.is_null() && self.seen.insert(value.clone()) {
            self.inner.update(value);
        }
    }
    fn update_typed(&mut self, col: &Column, row: usize) {
        if col.is_null(row) {
            return;
        }
        let value = col.get_scalar(row);
        if self.seen.insert(value.clone()) {
            self.inner.update(&value);
        }
    }
    fn finalize(&self) -> ScalarValue {
        self.inner.finalize()
    }
}

fn create_accumulator(name: &str, args_count: usize, distinct: bool) -> Box<dyn Accumulator> {
    let inner = build_accumulator(name, args_count);
    // DISTINCT only applies to aggregates over an argument; COUNT(*) has no
    // argument to deduplicate.
    if distinct && args_count > 0 {
        Box::new(DistinctAccumulator {
            seen: std::collections::HashSet::new(),
            inner,
        })
    } else {
        inner
    }
}

fn build_accumulator(name: &str, args_count: usize) -> Box<dyn Accumulator> {
    match name.to_lowercase().as_str() {
        "count" => {
            if args_count == 0 {
                Box::new(CountStarAccumulator { count: 0 })
            } else {
                Box::new(CountAccumulator { count: 0 })
            }
        }
        "sum" => Box::new(SumAccumulator {
            sum: 0.0,
            has_value: false,
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
fn is_supported_aggregate(name: &str) -> bool {
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
}

/// Sample standard deviation via Welford's online algorithm.
struct StddevAccumulator {
    count: u64,
    mean: f64,
    m2: f64,
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
            _ => return,
        };
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }
    fn finalize(&self) -> ScalarValue {
        if self.count < 2 {
            return ScalarValue::Null;
        }
        let variance = self.m2 / (self.count - 1) as f64;
        ScalarValue::Float64(variance.sqrt())
    }
}

/// Sample variance via Welford's online algorithm.
struct VarianceAccumulator {
    count: u64,
    mean: f64,
    m2: f64,
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
            _ => return,
        };
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }
    fn finalize(&self) -> ScalarValue {
        if self.count < 2 {
            return ScalarValue::Null;
        }
        ScalarValue::Float64(self.m2 / (self.count - 1) as f64)
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
        }
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
            let accs: Vec<Box<dyn Accumulator>> = self
                .aggregates
                .iter()
                .map(|agg| create_accumulator(&agg.function_name, agg.args.len(), agg.distinct))
                .collect();
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
                                None => acc.add_count(num_rows),
                                // Aggregates over a column still walk rows so
                                // nulls and per-value math are handled.
                                Some(col) => {
                                    let c = col.as_ref();
                                    for row in 0..num_rows {
                                        acc.update_typed(c, row);
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
            loop {
                match self.child.next().await? {
                    Some(eb) => {
                        state.ingest(
                            &eb.batch,
                            &self.group_by,
                            &self.aggregates,
                            &self.input_schema,
                        )?;
                    }
                    None => break,
                }
            }
            group_key_store = state.group_key_store;
            group_accumulators = state.group_accumulators;
            num_groups = state.num_groups;
        }

        if num_groups == 0 {
            self.finished = true;
            return Ok(());
        }

        self.result = Some(finalize_groups(
            &group_key_store,
            &group_accumulators,
            num_groups,
            num_group_cols,
            &self.output_schema,
        ));
        Ok(())
    }
}

/// Builds the output batch from materialized group keys and accumulators.
/// Shared by the serial and parallel aggregate paths so both emit identical
/// column layouts.
fn finalize_groups(
    group_key_store: &[Column],
    group_accumulators: &[Vec<Box<dyn Accumulator>>],
    num_groups: usize,
    num_group_cols: usize,
    output_schema: &[LogicalColumn],
) -> DataBatch {
    let mut col_builders: Vec<(ColumnData, NullBitmap, TypeId)> =
        Vec::with_capacity(output_schema.len());
    for col_def in output_schema {
        col_builders.push((
            ColumnData::with_capacity(col_def.type_id, num_groups),
            NullBitmap::empty(),
            col_def.type_id,
        ));
    }

    for gidx in 0..num_groups {
        for i in 0..num_group_cols {
            let (data, nulls, _) = &mut col_builders[i];
            let store_col = &group_key_store[i];
            nulls.push(store_col.is_null(gidx));
            data.push_from(&store_col.data, gidx);
        }
        for (i, acc) in group_accumulators[gidx].iter().enumerate() {
            let val = acc.finalize();
            let (data, nulls, _) = &mut col_builders[num_group_cols + i];
            nulls.push(val.is_null());
            data.push_scalar(&val);
        }
    }

    let columns: Vec<Column> = col_builders
        .into_iter()
        .map(|(data, nulls, type_id)| Column::with_nulls(data, nulls, type_id))
        .collect();

    DataBatch::new(columns)
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
}

impl GroupAccumulatorState {
    fn new() -> Self {
        Self {
            group_key_store: Vec::new(),
            hash_to_groups: PreHashMap::default(),
            group_accumulators: Vec::new(),
            num_groups: 0,
        }
    }

    /// Folds one input batch into the partition's group state.
    fn ingest(
        &mut self,
        batch: &DataBatch,
        group_by: &[BoundExpr],
        aggregates: &[AggregateExpr],
        input_schema: &[LogicalColumn],
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
                self.group_key_store.push(Column::new(
                    ColumnData::with_capacity(gc.type_id, 64),
                    gc.type_id,
                ));
            }
        }

        let Self {
            group_key_store,
            hash_to_groups,
            group_accumulators,
            num_groups,
        } = self;

        for row in 0..num_rows {
            let gidx = find_or_create_group(
                hash_to_groups,
                group_key_store,
                group_accumulators,
                num_groups,
                &group_refs,
                row,
                hashes[row],
                || {
                    aggregates
                        .iter()
                        .map(|agg| {
                            create_accumulator(&agg.function_name, agg.args.len(), agg.distinct)
                        })
                        .collect()
                },
            );
            let accs = &mut group_accumulators[gidx];
            for (i, acc) in accs.iter_mut().enumerate() {
                match &agg_arg_cols[i] {
                    Some(col) => acc.update_typed(col.as_ref(), row),
                    None => acc.update(&ScalarValue::Int64(1)),
                }
            }
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
        } = self;

        for ogidx in 0..other.num_groups {
            let gidx = find_or_create_group(
                hash_to_groups,
                group_key_store,
                group_accumulators,
                num_groups,
                &other_refs,
                ogidx,
                hashes[ogidx],
                || {
                    aggregates
                        .iter()
                        .map(|agg| {
                            create_accumulator(&agg.function_name, agg.args.len(), agg.distinct)
                        })
                        .collect()
                },
            );
            let dst = &mut group_accumulators[gidx];
            for (i, acc) in dst.iter_mut().enumerate() {
                acc.merge(other.group_accumulators[ogidx][i].as_ref());
            }
        }
    }
}

/// Finds the group matching `key_cols[.. ][row]` by hash and equality, or
/// creates it (copying the key into the store and building fresh
/// accumulators). The disjoint `&mut` parameters let one implementation serve
/// both batch ingest and partition merge without a borrow conflict.
#[allow(clippy::too_many_arguments)]
fn find_or_create_group(
    hash_to_groups: &mut PreHashMap<u64, Vec<usize>>,
    group_key_store: &mut [Column],
    group_accumulators: &mut Vec<Vec<Box<dyn Accumulator>>>,
    num_groups: &mut usize,
    key_cols: &[&Column],
    row: usize,
    hash: u64,
    make_accs: impl Fn() -> Vec<Box<dyn Accumulator>>,
) -> usize {
    let candidates = hash_to_groups.entry(hash).or_default();
    for &gidx in candidates.iter() {
        let mut eq = true;
        for (ci, kc) in key_cols.iter().enumerate() {
            let store_col = &group_key_store[ci];
            let a_null = kc.is_null(row);
            let b_null = store_col.is_null(gidx);
            if a_null != b_null {
                eq = false;
                break;
            }
            if a_null {
                continue;
            }
            if !column_values_equal_cross(&kc.data, row, &store_col.data, gidx) {
                eq = false;
                break;
            }
        }
        if eq {
            return gidx;
        }
    }
    let gidx = *num_groups;
    *num_groups += 1;
    candidates.push(gidx);
    for (ci, kc) in key_cols.iter().enumerate() {
        group_key_store[ci].push_row_from(kc, row);
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
            va[a_idx].to_bits() == vb[b_idx].to_bits()
        }
        (ColumnData::Float64(va), ColumnData::Float64(vb)) => {
            va[a_idx].to_bits() == vb[b_idx].to_bits()
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
pub fn aggregate_supports_parallel(function_name: &str) -> bool {
    matches!(
        function_name.to_lowercase().as_str(),
        "count" | "sum" | "avg" | "min" | "max"
    )
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

        let num_workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .min(num_pages.max(1) as usize)
            .max(1);
        let pages_per_worker = num_pages.div_ceil(num_workers as u64);

        let mut handles = Vec::with_capacity(num_workers);
        for worker_id in 0..num_workers {
            let start_page = worker_id as u64 * pages_per_worker;
            let end_page = ((worker_id as u64 + 1) * pages_per_worker).min(num_pages);
            if start_page >= end_page {
                continue;
            }
            handles.push(tokio::spawn(aggregate_page_range(
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
        ));
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
    use zyron_catalog::ColumnId;
    use zyron_planner::binder::ColumnRef;

    // DISTINCT aggregates must fold each value only once.
    #[test]
    fn distinct_aggregates_dedup() {
        let vals = [1i64, 1, 2, 3, 3];

        let mut count_distinct = create_accumulator("count", 1, true);
        let mut sum_distinct = create_accumulator("sum", 1, true);
        let mut count_all = create_accumulator("count", 1, false);
        for &v in &vals {
            count_distinct.update(&ScalarValue::Int64(v));
            sum_distinct.update(&ScalarValue::Int64(v));
            count_all.update(&ScalarValue::Int64(v));
        }
        // 3 distinct values {1,2,3}; sum of distinct = 6; non-distinct counts all 5.
        assert_eq!(count_distinct.finalize(), ScalarValue::Int64(3));
        assert_eq!(sum_distinct.finalize(), ScalarValue::Float64(6.0));
        assert_eq!(count_all.finalize(), ScalarValue::Int64(5));

        // NULLs are ignored by distinct too.
        let mut count_with_nulls = create_accumulator("count", 1, true);
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
            ts_precision: None,
        })
    }

    fn count_star() -> AggregateExpr {
        AggregateExpr {
            function_name: "count".into(),
            args: vec![],
            distinct: false,
            return_type: TypeId::Int64,
        }
    }

    fn sum_of(column_id: u16) -> AggregateExpr {
        AggregateExpr {
            function_name: "sum".into(),
            args: vec![col_ref(column_id, TypeId::Int64)],
            distinct: false,
            return_type: TypeId::Float64,
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
                ts_precision: None,
            },
            LogicalColumn {
                table_idx: Some(0),
                column_id: ColumnId(1),
                name: "v".into(),
                type_id: TypeId::Int64,
                nullable: false,
                ts_precision: None,
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
                ts_precision: None,
            });
        }
        for (i, agg) in aggregates.iter().enumerate() {
            schema.push(LogicalColumn {
                table_idx: None,
                column_id: ColumnId(100 + i as u16),
                name: agg.function_name.clone(),
                type_id: agg.return_type,
                nullable: true,
                ts_precision: None,
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
        );

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
        );

        assert_eq!(result_map(&serial_out), result_map(&merged_out));
        // Known answer: key 1 -> count 3 sum 65, key 2 -> 3 38, key 3 -> 3 8.
        let m = result_map(&merged_out);
        assert_eq!(m[&1], (3, 65));
        assert_eq!(m[&2], (3, 38));
        assert_eq!(m[&3], (3, 8));
    }
}

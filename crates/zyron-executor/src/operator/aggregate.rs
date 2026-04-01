//! Aggregation operators and accumulator infrastructure.
//!
//! Provides hash-based and sort-based aggregation with pluggable accumulators
//! for COUNT, SUM, AVG, MIN, MAX. Uses typed column access to avoid
//! ScalarValue allocation in hot paths.

use zyron_common::{Result, TypeId};
use zyron_planner::binder::BoundExpr;
use zyron_planner::logical::{AggregateExpr, LogicalColumn};

use crate::batch::DataBatch;
use crate::column::{Column, ColumnData, NullBitmap, ScalarValue};
use crate::compute::{self, PreHashMap};
use crate::expr::evaluate;
use crate::operator::{ExecutionBatch, Operator, OperatorResult};

// ---------------------------------------------------------------------------
// Accumulator trait and built-in implementations
// ---------------------------------------------------------------------------

trait Accumulator: Send {
    fn update(&mut self, value: &ScalarValue);

    /// Typed update directly from a column at a given row, avoiding ScalarValue.
    fn update_typed(&mut self, col: &Column, row: usize) {
        self.update(&col.get_scalar(row));
    }

    fn finalize(&self) -> ScalarValue;
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
    fn finalize(&self) -> ScalarValue {
        ScalarValue::Int64(self.count)
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
}

fn create_accumulator(name: &str, args_count: usize) -> Box<dyn Accumulator> {
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
        _ => Box::new(CountAccumulator { count: 0 }),
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
        // group_index maps hash -> list of (group_idx, first occurrence row data).
        // We store group keys in columnar builders for collision resolution.
        let num_group_cols = self.group_by.len();
        let num_agg_cols = self.aggregates.len();

        // Accumulated group key columns for collision checking.
        let mut group_key_store: Vec<Column> = Vec::new();
        let mut group_accumulators: Vec<Vec<Box<dyn Accumulator>>> = Vec::new();
        let mut hash_to_groups: PreHashMap<u64, Vec<usize>> = PreHashMap::default();
        let mut num_groups = 0usize;

        loop {
            match self.child.next().await? {
                Some(eb) => {
                    let batch = &eb.batch;
                    let num_rows = batch.num_rows;

                    let group_cols: Vec<Column> = self
                        .group_by
                        .iter()
                        .map(|expr| evaluate(expr, batch, &self.input_schema, &[]))
                        .collect::<Result<Vec<_>>>()?;

                    let agg_arg_cols: Vec<Option<Column>> = self
                        .aggregates
                        .iter()
                        .map(|agg| {
                            if agg.args.is_empty() {
                                Ok(None)
                            } else {
                                Ok(Some(evaluate(
                                    &agg.args[0],
                                    batch,
                                    &self.input_schema,
                                    &[],
                                )?))
                            }
                        })
                        .collect::<Result<Vec<_>>>()?;

                    // Batch hash group keys.
                    let group_refs: Vec<&Column> = group_cols.iter().collect();
                    let hashes = if !group_cols.is_empty() {
                        compute::hash_column_batch(&group_refs, num_rows)
                    } else {
                        vec![0u64; num_rows]
                    };

                    // Initialize group key store on first batch.
                    if group_key_store.is_empty() && !group_cols.is_empty() {
                        for gc in &group_cols {
                            group_key_store.push(Column::new(
                                ColumnData::with_capacity(gc.type_id, 64),
                                gc.type_id,
                            ));
                        }
                    }

                    for row in 0..num_rows {
                        let hash = hashes[row];

                        // Find or create group.
                        let group_idx = if num_group_cols == 0 {
                            // Global aggregate: single group.
                            if num_groups == 0 {
                                let accs = self
                                    .aggregates
                                    .iter()
                                    .map(|agg| {
                                        create_accumulator(&agg.function_name, agg.args.len())
                                    })
                                    .collect();
                                group_accumulators.push(accs);
                                num_groups = 1;
                            }
                            0
                        } else {
                            // Lookup by hash, then check for collision.
                            let candidates = hash_to_groups.entry(hash).or_default();
                            let mut found = None;
                            for &gidx in candidates.iter() {
                                let mut eq = true;
                                for (ci, gc) in group_cols.iter().enumerate() {
                                    let store_col = &group_key_store[ci];
                                    // Compare group_cols[ci][row] vs store_col[gidx].
                                    let a_null = gc.is_null(row);
                                    let b_null = store_col.is_null(gidx);
                                    if a_null != b_null {
                                        eq = false;
                                        break;
                                    }
                                    if a_null {
                                        continue;
                                    }
                                    if !column_values_equal_cross(
                                        &gc.data,
                                        row,
                                        &store_col.data,
                                        gidx,
                                    ) {
                                        eq = false;
                                        break;
                                    }
                                }
                                if eq {
                                    found = Some(gidx);
                                    break;
                                }
                            }
                            match found {
                                Some(gidx) => gidx,
                                None => {
                                    let gidx = num_groups;
                                    num_groups += 1;
                                    candidates.push(gidx);
                                    // Append group key to store.
                                    for (ci, gc) in group_cols.iter().enumerate() {
                                        group_key_store[ci].push_row_from(gc, row);
                                    }
                                    let accs = self
                                        .aggregates
                                        .iter()
                                        .map(|agg| {
                                            create_accumulator(&agg.function_name, agg.args.len())
                                        })
                                        .collect();
                                    group_accumulators.push(accs);
                                    gidx
                                }
                            }
                        };

                        // Update accumulators using typed access.
                        let accs = &mut group_accumulators[group_idx];
                        for (i, acc) in accs.iter_mut().enumerate() {
                            match &agg_arg_cols[i] {
                                Some(col) => acc.update_typed(col, row),
                                None => acc.update(&ScalarValue::Int64(1)),
                            }
                        }
                    }
                }
                None => break,
            }
        }

        // For global aggregates with no groups, insert a single entry.
        if num_groups == 0 && self.group_by.is_empty() {
            let accs: Vec<Box<dyn Accumulator>> = self
                .aggregates
                .iter()
                .map(|agg| create_accumulator(&agg.function_name, agg.args.len()))
                .collect();
            group_accumulators.push(accs);
            num_groups = 1;
        }

        if num_groups == 0 {
            self.finished = true;
            return Ok(());
        }

        // Build output columns.
        let total_cols = num_group_cols + num_agg_cols;

        let mut col_builders: Vec<(ColumnData, NullBitmap, TypeId)> =
            Vec::with_capacity(total_cols);
        for col_def in &self.output_schema {
            col_builders.push((
                ColumnData::with_capacity(col_def.type_id, num_groups),
                NullBitmap::empty(),
                col_def.type_id,
            ));
        }

        for gidx in 0..num_groups {
            // Emit group key columns from store.
            for i in 0..num_group_cols {
                let (data, nulls, _) = &mut col_builders[i];
                let store_col = &group_key_store[i];
                nulls.push(store_col.is_null(gidx));
                data.push_from(&store_col.data, gidx);
            }
            // Emit aggregate results.
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

        self.result = Some(DataBatch::new(columns));
        Ok(())
    }
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

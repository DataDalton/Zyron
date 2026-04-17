//! Window function operator.
//!
//! Drains all input, sorts by (partition_by, order_by), iterates partition
//! boundaries, and applies each window function per-partition. Appends one
//! result column per window expression to the output batch.
//!
//! Currently supports the unbounded full-partition frame (default for
//! analytical functions without explicit ROWS/RANGE BETWEEN).

use zyron_common::{Result, TypeId, ZyronError};
use zyron_parser::ast::{WindowFrame, WindowFrameBound, WindowFrameDirection, WindowFrameMode};
use zyron_planner::binder::{BoundExpr, BoundOrderBy};
use zyron_planner::logical::LogicalColumn;

use crate::batch::DataBatch;
use crate::column::{Column, ColumnData, NullBitmap, ScalarValue};
use crate::expr::evaluate;
use crate::operator::{ExecutionBatch, Operator, OperatorResult};

/// Per-partition window function evaluator.
pub struct WindowOperator {
    child: Box<dyn Operator>,
    window_exprs: Vec<BoundExpr>,
    input_schema: Vec<LogicalColumn>,
    /// Fully materialized output with input columns + window columns.
    result: Option<DataBatch>,
    output_cursor: usize,
    finished: bool,
}

impl WindowOperator {
    pub fn new(
        child: Box<dyn Operator>,
        window_exprs: Vec<BoundExpr>,
        input_schema: Vec<LogicalColumn>,
    ) -> Self {
        Self {
            child,
            window_exprs,
            input_schema,
            result: None,
            output_cursor: 0,
            finished: false,
        }
    }

    async fn materialize(&mut self) -> Result<()> {
        // Drain all input batches into a single combined batch.
        let mut combined_columns: Vec<Vec<Column>> = Vec::new();
        let mut total_rows = 0usize;

        while let Some(eb) = self.child.next().await? {
            total_rows += eb.batch.num_rows;
            if combined_columns.is_empty() {
                combined_columns.resize_with(eb.batch.num_columns(), Vec::new);
            }
            for (i, col) in eb.batch.columns.into_iter().enumerate() {
                combined_columns[i].push(col);
            }
        }

        if total_rows == 0 {
            self.finished = true;
            return Ok(());
        }

        // Concatenate all column pieces.
        let mut input_columns: Vec<Column> =
            combined_columns.into_iter().map(concat_columns).collect();

        // Determine sort key from the first window expression's partition_by + order_by.
        // All window functions currently share the same implicit frame; we process
        // each independently so they can have different partition/order keys.
        let mut output_columns = input_columns.clone();

        for window_expr in &self.window_exprs {
            let (function, partition_by, order_by, frame) = match window_expr {
                BoundExpr::WindowFunction {
                    function,
                    partition_by,
                    order_by,
                    frame,
                    ..
                } => (function.as_ref(), partition_by, order_by, frame.as_ref()),
                _ => {
                    return Err(ZyronError::ExecutionError(
                        "Window expression must be BoundExpr::WindowFunction".into(),
                    ));
                }
            };

            // Build a sort key: partition keys (ascending) then order keys.
            let sort_indices = compute_sort_indices(
                &input_columns,
                &self.input_schema,
                partition_by,
                order_by,
                total_rows,
            )?;

            // Evaluate the window function in sorted order, then unsort back.
            let sorted_input = reorder_batch(&input_columns, &sort_indices);

            // Identify partition boundaries.
            let partition_boundaries = find_partition_boundaries(
                &sorted_input,
                &self.input_schema,
                partition_by,
                total_rows,
            )?;

            // Compute window function output per partition.
            let window_output = evaluate_window_function(
                function,
                &sorted_input,
                &self.input_schema,
                &partition_boundaries,
                order_by,
                frame,
            )?;

            // Unsort: scatter the window values back to their original row positions.
            let unsorted = unsort_column(&window_output, &sort_indices);
            output_columns.push(unsorted);
        }

        self.result = Some(DataBatch::new(output_columns));
        Ok(())
    }
}

impl Operator for WindowOperator {
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
            let batch = self
                .result
                .as_ref()
                .expect("result populated after materialize");
            let total = batch.num_rows;
            if self.output_cursor >= total {
                self.finished = true;
                return Ok(None);
            }
            let emit_rows = (total - self.output_cursor).min(1024);
            let slice = slice_batch(batch, self.output_cursor, emit_rows);
            self.output_cursor += emit_rows;
            if self.output_cursor >= total {
                self.finished = true;
            }
            Ok(Some(ExecutionBatch::new(slice)))
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn concat_columns(parts: Vec<Column>) -> Column {
    if parts.len() == 1 {
        return parts.into_iter().next().expect("single part");
    }
    let type_id = parts[0].type_id;
    let mut iter = parts.into_iter();
    let first = iter.next().expect("at least one part");
    let mut acc = first;
    for p in iter {
        acc.extend_from(&p);
    }
    acc.type_id = type_id;
    acc
}

fn slice_batch(batch: &DataBatch, offset: usize, len: usize) -> DataBatch {
    let cols: Vec<Column> = batch
        .columns
        .iter()
        .map(|c| Column {
            data: c.data.slice(offset, len),
            nulls: c.nulls.slice(offset, len),
            type_id: c.type_id,
        })
        .collect();
    DataBatch::new(cols)
}

/// Computes sort indices that group partitions together and order rows within
/// each partition. Uses a stable sort by (partition_keys..., order_keys...).
fn compute_sort_indices(
    columns: &[Column],
    schema: &[LogicalColumn],
    partition_by: &[BoundExpr],
    order_by: &[BoundOrderBy],
    total_rows: usize,
) -> Result<Vec<u32>> {
    let mut indices: Vec<u32> = (0..total_rows as u32).collect();

    // Materialize evaluation of partition keys and order keys.
    let batch = DataBatch::new(columns.to_vec());
    let mut partition_cols: Vec<Column> = Vec::with_capacity(partition_by.len());
    for expr in partition_by {
        partition_cols.push(evaluate(expr, &batch, schema, &[])?);
    }
    let mut order_cols: Vec<Column> = Vec::with_capacity(order_by.len());
    for ob in order_by {
        order_cols.push(evaluate(&ob.expr, &batch, schema, &[])?);
    }

    indices.sort_by(|&a, &b| {
        // Compare partition keys first (ascending).
        for col in &partition_cols {
            let ord = compare_col_rows(col, a as usize, b as usize);
            if ord != std::cmp::Ordering::Equal {
                return ord;
            }
        }
        // Then order keys respecting direction.
        for (i, col) in order_cols.iter().enumerate() {
            let ord = compare_col_rows(col, a as usize, b as usize);
            let ord = if order_by[i].asc { ord } else { ord.reverse() };
            if ord != std::cmp::Ordering::Equal {
                return ord;
            }
        }
        std::cmp::Ordering::Equal
    });

    Ok(indices)
}

fn compare_col_rows(col: &Column, a: usize, b: usize) -> std::cmp::Ordering {
    let a_null = col.nulls.is_null(a);
    let b_null = col.nulls.is_null(b);
    match (a_null, b_null) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Less,
        (false, true) => std::cmp::Ordering::Greater,
        (false, false) => compare_col_values(&col.data, a, b),
    }
}

fn compare_col_values(data: &ColumnData, a: usize, b: usize) -> std::cmp::Ordering {
    match data {
        ColumnData::Boolean(v) => v[a].cmp(&v[b]),
        ColumnData::Int8(v) => v[a].cmp(&v[b]),
        ColumnData::Int16(v) => v[a].cmp(&v[b]),
        ColumnData::Int32(v) => v[a].cmp(&v[b]),
        ColumnData::Int64(v) => v[a].cmp(&v[b]),
        ColumnData::Int128(v) => v[a].cmp(&v[b]),
        ColumnData::UInt8(v) => v[a].cmp(&v[b]),
        ColumnData::UInt16(v) => v[a].cmp(&v[b]),
        ColumnData::UInt32(v) => v[a].cmp(&v[b]),
        ColumnData::UInt64(v) => v[a].cmp(&v[b]),
        ColumnData::Float32(v) => v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal),
        ColumnData::Float64(v) => v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal),
        ColumnData::Utf8(v) => v[a].cmp(&v[b]),
        ColumnData::Binary(v) => v[a].cmp(&v[b]),
        ColumnData::FixedBinary16(v) => v[a].cmp(&v[b]),
        ColumnData::Interval(v) => v[a].cmp(&v[b]),
    }
}

fn reorder_batch(columns: &[Column], indices: &[u32]) -> Vec<Column> {
    columns.iter().map(|c| reorder_column(c, indices)).collect()
}

fn reorder_column(col: &Column, indices: &[u32]) -> Column {
    let new_data = col.data.take(indices);
    let new_nulls = col.nulls.take(indices);
    Column::with_nulls(new_data, new_nulls, col.type_id)
}

/// Scatters sorted-position results back to original row positions.
/// Given that indices[i] = original_row_idx for sorted row i, this produces
/// a column where result[indices[i]] = sorted[i].
fn unsort_column(sorted: &Column, indices: &[u32]) -> Column {
    let n = indices.len();
    let mut inverse = vec![0u32; n];
    for (sorted_pos, &orig_pos) in indices.iter().enumerate() {
        inverse[orig_pos as usize] = sorted_pos as u32;
    }
    reorder_column(sorted, &inverse)
}

/// Returns partition start offsets in sorted order. Always includes 0 and total_rows as endpoints.
/// Partitions are ranges [starts[i], starts[i+1]).
fn find_partition_boundaries(
    sorted_columns: &[Column],
    schema: &[LogicalColumn],
    partition_by: &[BoundExpr],
    total_rows: usize,
) -> Result<Vec<usize>> {
    let mut boundaries = vec![0usize];

    if partition_by.is_empty() || total_rows == 0 {
        boundaries.push(total_rows);
        return Ok(boundaries);
    }

    let batch = DataBatch::new(sorted_columns.to_vec());
    let mut partition_cols: Vec<Column> = Vec::with_capacity(partition_by.len());
    for expr in partition_by {
        partition_cols.push(evaluate(expr, &batch, schema, &[])?);
    }

    for row in 1..total_rows {
        let mut same = true;
        for col in &partition_cols {
            if compare_col_rows(col, row - 1, row) != std::cmp::Ordering::Equal {
                same = false;
                break;
            }
        }
        if !same {
            boundaries.push(row);
        }
    }
    boundaries.push(total_rows);
    Ok(boundaries)
}

/// Identifies which output column type a window function produces.
enum WindowOutputKind {
    /// Float64 column with null bitmap.
    Float64,
    /// Int64 column for ranking/row number functions.
    Int64,
    /// Columns whose type matches the first argument (LAG, LEAD, FIRST_VALUE, LAST_VALUE, NTH_VALUE).
    MatchFirstArg,
}

fn window_function_kind(name: &str) -> Result<WindowOutputKind> {
    match name {
        "ema"
        | "rate"
        | "delta"
        | "derivative"
        | "moving_average"
        | "moving_avg"
        | "exponential_smoothing"
        | "percent_rank"
        | "cume_dist" => Ok(WindowOutputKind::Float64),
        "row_number" | "rank" | "dense_rank" | "ntile" => Ok(WindowOutputKind::Int64),
        "lag" | "lead" | "first_value" | "last_value" | "nth_value" => {
            Ok(WindowOutputKind::MatchFirstArg)
        }
        _ => Err(ZyronError::ExecutionError(format!(
            "unsupported window function: {}",
            name
        ))),
    }
}

/// Evaluates a window function over each partition. Dispatches on function name.
fn evaluate_window_function(
    function: &BoundExpr,
    sorted_columns: &[Column],
    schema: &[LogicalColumn],
    partition_boundaries: &[usize],
    order_by: &[BoundOrderBy],
    frame: Option<&WindowFrame>,
) -> Result<Column> {
    let (name, args) = match function {
        BoundExpr::Function { name, args, .. } => (name.to_lowercase(), args),
        BoundExpr::AggregateFunction { name, args, .. } => (name.to_lowercase(), args),
        _ => {
            return Err(ZyronError::ExecutionError(
                "Window expects Function or AggregateFunction inner node".into(),
            ));
        }
    };

    let total_rows = sorted_columns.first().map(|c| c.data.len()).unwrap_or(0);

    // Evaluate argument columns in sorted order.
    let batch = DataBatch::new(sorted_columns.to_vec());
    let mut arg_cols: Vec<Column> = Vec::with_capacity(args.len());
    for a in args {
        arg_cols.push(evaluate(a, &batch, schema, &[])?);
    }

    // Evaluate the ORDER BY expression (used as the time axis for rate/derivative
    // and as the ranking key for rank/dense_rank).
    let time_col: Option<Column> = if let Some(ob) = order_by.first() {
        Some(evaluate(&ob.expr, &batch, schema, &[])?)
    } else {
        None
    };

    let kind = window_function_kind(&name)?;

    match kind {
        WindowOutputKind::Int64 => {
            let mut result_data = vec![0i64; total_rows];
            let result_nulls = NullBitmap::none(total_rows);
            for window in partition_boundaries.windows(2) {
                let start = window[0];
                let end = window[1];
                if end <= start {
                    continue;
                }
                match name.as_str() {
                    "row_number" => {
                        for i in start..end {
                            result_data[i] = (i - start + 1) as i64;
                        }
                    }
                    "rank" => {
                        compute_rank(
                            &mut result_data,
                            start,
                            end,
                            order_by,
                            &batch,
                            schema,
                            false,
                        )?;
                    }
                    "dense_rank" => {
                        compute_rank(&mut result_data, start, end, order_by, &batch, schema, true)?;
                    }
                    "ntile" => {
                        let n = extract_i64_scalar(&arg_cols.first(), start)?
                            .unwrap_or(1)
                            .max(1) as usize;
                        let partition_len = end - start;
                        for i in start..end {
                            let pos = i - start;
                            let bucket = (pos * n) / partition_len;
                            result_data[i] = (bucket + 1) as i64;
                        }
                    }
                    _ => {
                        return Err(ZyronError::ExecutionError(format!(
                            "internal: unexpected Int64 window function: {}",
                            name
                        )));
                    }
                }
            }
            Ok(Column::with_nulls(
                ColumnData::Int64(result_data),
                result_nulls,
                TypeId::Int64,
            ))
        }
        WindowOutputKind::Float64 => {
            let mut result_data = vec![0.0f64; total_rows];
            let mut result_nulls = NullBitmap::none(total_rows);

            for window in partition_boundaries.windows(2) {
                let start = window[0];
                let end = window[1];
                if end <= start {
                    continue;
                }
                match name.as_str() {
                    "ema" => {
                        let values = extract_f64_range(&arg_cols[0], start, end)?;
                        let alpha = extract_f64_scalar(&arg_cols.get(1), start)?.unwrap_or(0.5);
                        let out = zyron_types::timeseries::ema(&values, alpha);
                        for (i, v) in out.iter().enumerate() {
                            result_data[start + i] = *v;
                        }
                    }
                    "delta" => {
                        let values = extract_f64_range(&arg_cols[0], start, end)?;
                        let out = zyron_types::timeseries::delta(&values);
                        for (i, v) in out.iter().enumerate() {
                            match v {
                                Some(x) => result_data[start + i] = *x,
                                None => result_nulls.set_null(start + i),
                            }
                        }
                    }
                    "rate" => {
                        let values = extract_f64_range(&arg_cols[0], start, end)?;
                        let t_col = time_col.as_ref().ok_or_else(|| {
                            ZyronError::ExecutionError("rate() requires ORDER BY".into())
                        })?;
                        let times = extract_i64_range(t_col, start, end)?;
                        let out = zyron_types::timeseries::rate(&times, &values);
                        for (i, v) in out.iter().enumerate() {
                            match v {
                                Some(x) => result_data[start + i] = *x,
                                None => result_nulls.set_null(start + i),
                            }
                        }
                    }
                    "derivative" => {
                        let values = extract_f64_range(&arg_cols[0], start, end)?;
                        let time_src = if arg_cols.len() >= 2 {
                            &arg_cols[1]
                        } else {
                            time_col.as_ref().ok_or_else(|| {
                                ZyronError::ExecutionError(
                                    "derivative() requires time argument or ORDER BY".into(),
                                )
                            })?
                        };
                        let times = extract_i64_range(time_src, start, end)?;
                        let out = zyron_types::timeseries::derivative(&times, &values);
                        for (i, v) in out.iter().enumerate() {
                            match v {
                                Some(x) => result_data[start + i] = *x,
                                None => result_nulls.set_null(start + i),
                            }
                        }
                    }
                    "moving_average" | "moving_avg" => {
                        // Frame handling:
                        // - ROWS BETWEEN N PRECEDING ...: row-offset frame.
                        // - RANGE BETWEEN N/INTERVAL PRECEDING ...: value/time-offset frame.
                        // - No frame: fall back to fixed window size from args[1].
                        if let Some(f) = frame {
                            let values = extract_f64_range(&arg_cols[0], start, end)?;
                            match f.mode {
                                WindowFrameMode::Rows => {
                                    for i in 0..(end - start) {
                                        let (lo, hi) = resolve_row_frame(i, end - start, f);
                                        if hi > lo {
                                            let sum: f64 = values[lo..hi].iter().sum();
                                            result_data[start + i] = sum / (hi - lo) as f64;
                                        }
                                    }
                                    continue;
                                }
                                WindowFrameMode::Range => {
                                    let order_col = match time_col.as_ref() {
                                        Some(c) => c,
                                        None => {
                                            return Err(ZyronError::ExecutionError(
                                                "RANGE frame requires ORDER BY".into(),
                                            ));
                                        }
                                    };
                                    let order_values = extract_order_values(order_col, start, end);
                                    for i in 0..(end - start) {
                                        let (lo, hi) =
                                            resolve_range_frame(i, end - start, f, &order_values);
                                        if hi > lo {
                                            let sum: f64 = values[lo..hi].iter().sum();
                                            result_data[start + i] = sum / (hi - lo) as f64;
                                        }
                                    }
                                    continue;
                                }
                            }
                        }
                        let values = extract_f64_range(&arg_cols[0], start, end)?;
                        let window_size =
                            extract_i64_scalar(&arg_cols.get(1), start)?.unwrap_or(3) as usize;
                        let out = zyron_types::statistics::moving_average(&values, window_size);
                        for (i, v) in out.iter().enumerate() {
                            result_data[start + i] = *v;
                        }
                    }
                    "exponential_smoothing" => {
                        let values = extract_f64_range(&arg_cols[0], start, end)?;
                        let alpha = extract_f64_scalar(&arg_cols.get(1), start)?.unwrap_or(0.5);
                        let out = zyron_types::statistics::exponential_smoothing(&values, alpha);
                        for (i, v) in out.iter().enumerate() {
                            result_data[start + i] = *v;
                        }
                    }
                    "percent_rank" => {
                        // PERCENT_RANK = (rank - 1) / (total - 1) within partition.
                        let partition_len = end - start;
                        if partition_len == 1 {
                            result_data[start] = 0.0;
                            continue;
                        }
                        let mut ranks = vec![0i64; total_rows];
                        compute_rank(&mut ranks, start, end, order_by, &batch, schema, false)?;
                        let denom = (partition_len - 1) as f64;
                        for i in start..end {
                            result_data[i] = (ranks[i] - 1) as f64 / denom;
                        }
                    }
                    "cume_dist" => {
                        // CUME_DIST = count of peers including current / partition size.
                        // Using rank to determine peer count.
                        let partition_len = end - start;
                        let mut ranks = vec![0i64; total_rows];
                        compute_rank(&mut ranks, start, end, order_by, &batch, schema, false)?;
                        // count how many rows have rank <= current row's rank
                        for i in start..end {
                            let r = ranks[i];
                            let cnt = ranks[start..end].iter().filter(|&&x| x <= r).count();
                            result_data[i] = cnt as f64 / partition_len as f64;
                        }
                    }
                    _ => unreachable!("handled by kind dispatch"),
                }
            }

            Ok(Column::with_nulls(
                ColumnData::Float64(result_data),
                result_nulls,
                TypeId::Float64,
            ))
        }
        WindowOutputKind::MatchFirstArg => {
            // Build an output column matching the type of arg 0 by scattering.
            if arg_cols.is_empty() {
                return Err(ZyronError::ExecutionError(format!(
                    "{} requires at least one argument",
                    name
                )));
            }
            let src = &arg_cols[0];
            let mut indices: Vec<i64> = vec![-1; total_rows]; // source index per output row; -1 = null
            for window in partition_boundaries.windows(2) {
                let start = window[0];
                let end = window[1];
                if end <= start {
                    continue;
                }
                match name.as_str() {
                    "lag" => {
                        let offset =
                            extract_i64_scalar(&arg_cols.get(1), start)?.unwrap_or(1) as isize;
                        for i in start..end {
                            let src_pos = (i as isize) - offset;
                            if src_pos >= start as isize && src_pos < end as isize {
                                indices[i] = src_pos as i64;
                            }
                        }
                    }
                    "lead" => {
                        let offset =
                            extract_i64_scalar(&arg_cols.get(1), start)?.unwrap_or(1) as isize;
                        for i in start..end {
                            let src_pos = (i as isize) + offset;
                            if src_pos >= start as isize && src_pos < end as isize {
                                indices[i] = src_pos as i64;
                            }
                        }
                    }
                    "first_value" => {
                        for i in start..end {
                            indices[i] = start as i64;
                        }
                    }
                    "last_value" => {
                        // Depends on frame: default for last_value is CURRENT ROW (unlike first_value which is UNBOUNDED PRECEDING).
                        // Apply the specified frame if present; otherwise use end-1 (full partition).
                        if let Some(f) = frame {
                            let len = end - start;
                            match f.mode {
                                WindowFrameMode::Rows => {
                                    for i in 0..len {
                                        let (lo, hi) = resolve_row_frame(i, len, f);
                                        if hi > lo {
                                            indices[start + i] = (start + hi - 1) as i64;
                                        }
                                    }
                                    continue;
                                }
                                WindowFrameMode::Range => {
                                    if let Some(order_col) = time_col.as_ref() {
                                        let order_values =
                                            extract_order_values(order_col, start, end);
                                        for i in 0..len {
                                            let (lo, hi) =
                                                resolve_range_frame(i, len, f, &order_values);
                                            if hi > lo {
                                                indices[start + i] = (start + hi - 1) as i64;
                                            }
                                        }
                                        continue;
                                    }
                                }
                            }
                        }
                        for i in start..end {
                            indices[i] = (end - 1) as i64;
                        }
                    }
                    "nth_value" => {
                        let n = extract_i64_scalar(&arg_cols.get(1), start)?
                            .unwrap_or(1)
                            .max(1) as usize;
                        if (end - start) >= n {
                            let nth_src = (start + n - 1) as i64;
                            for i in start..end {
                                indices[i] = nth_src;
                            }
                        }
                    }
                    _ => unreachable!("handled by kind dispatch"),
                }
            }
            scatter_column_by_index(src, &indices)
        }
    }
}

/// Computes dense or sparse ranks within a single partition range [start, end).
fn compute_rank(
    out: &mut [i64],
    start: usize,
    end: usize,
    order_by: &[BoundOrderBy],
    batch: &DataBatch,
    schema: &[LogicalColumn],
    dense: bool,
) -> Result<()> {
    if end == start {
        return Ok(());
    }
    if order_by.is_empty() {
        // Without ORDER BY, all rows tie at rank 1.
        for i in start..end {
            out[i] = 1;
        }
        return Ok(());
    }

    let order_cols: Vec<Column> = order_by
        .iter()
        .map(|ob| evaluate(&ob.expr, batch, schema, &[]))
        .collect::<Result<Vec<_>>>()?;

    let mut current_rank: i64 = 1;
    out[start] = 1;
    let mut tie_base: i64 = 1;
    for i in (start + 1)..end {
        let mut tied = true;
        for col in &order_cols {
            if compare_col_rows(col, i - 1, i) != std::cmp::Ordering::Equal {
                tied = false;
                break;
            }
        }
        if tied {
            out[i] = tie_base;
        } else {
            if dense {
                current_rank += 1;
            } else {
                current_rank = (i - start + 1) as i64;
            }
            out[i] = current_rank;
            tie_base = current_rank;
        }
    }
    Ok(())
}

/// Resolves a row-frame (WindowFrameMode::Rows) to (lower_inclusive, upper_exclusive)
/// within a partition of the given length, for the given row position in the partition.
fn resolve_row_frame(pos: usize, partition_len: usize, frame: &WindowFrame) -> (usize, usize) {
    let start = bound_to_offset(pos, partition_len, frame.start, true);
    let end = frame
        .end
        .map(|b| bound_to_offset(pos, partition_len, b, false))
        .unwrap_or((pos + 1) as isize);
    let lo = start.max(0) as usize;
    let hi = end.min(partition_len as isize).max(0) as usize;
    (lo.min(partition_len), hi.min(partition_len))
}

/// Resolves a RANGE frame to (lower_inclusive, upper_exclusive) partition-row indices.
/// Supports both numeric RANGE (offset compared against order-by numeric values) and
/// interval RANGE (calendar-aware timestamp arithmetic when ORDER BY is temporal).
///
/// order_values must be the ORDER BY column values in sorted order, indexed relative
/// to `partition_start`. For each row `pos` in [0, partition_len), find all rows
/// whose order value falls within the computed bounds.
fn resolve_range_frame(
    pos: usize,
    partition_len: usize,
    frame: &WindowFrame,
    order_values: &[i64],
) -> (usize, usize) {
    // Current row's order value is the anchor.
    let anchor = order_values[pos];

    // Resolve lower bound order value
    let (lo_val, lo_inclusive) = resolve_range_bound(anchor, frame.start, true);
    // Resolve upper bound. If frame.end is None, use CURRENT ROW.
    let end_bound = frame.end.unwrap_or(WindowFrameBound::CurrentRow);
    let (hi_val, hi_inclusive) = resolve_range_bound(anchor, end_bound, false);

    // Binary search for the lo and hi indices (order_values is sorted).
    let lo_idx = match lo_val {
        None => 0,
        Some(v) => {
            let search =
                order_values.partition_point(|x| if lo_inclusive { *x < v } else { *x <= v });
            search.min(partition_len)
        }
    };
    let hi_idx = match hi_val {
        None => partition_len,
        Some(v) => {
            let search =
                order_values.partition_point(|x| if hi_inclusive { *x <= v } else { *x < v });
            search.min(partition_len)
        }
    };

    (lo_idx, hi_idx)
}

/// Resolves a RANGE frame bound to an order-value threshold.
/// Returns (Some(value), inclusive) or (None, _) for unbounded.
/// When is_start is true, the value is the lower edge; when false, the upper edge.
fn resolve_range_bound(
    anchor: i64,
    bound: WindowFrameBound,
    is_start: bool,
) -> (Option<i64>, bool) {
    match bound {
        WindowFrameBound::CurrentRow => (Some(anchor), true),
        WindowFrameBound::Unbounded(_) => (None, true),
        WindowFrameBound::Offset(n, WindowFrameDirection::Preceding) => {
            let offset = anchor.saturating_sub(n as i64);
            let _ = is_start;
            (Some(offset), true)
        }
        WindowFrameBound::Offset(n, WindowFrameDirection::Following) => {
            let offset = anchor.saturating_add(n as i64);
            let _ = is_start;
            (Some(offset), true)
        }
        WindowFrameBound::IntervalBound(interval, WindowFrameDirection::Preceding) => {
            // anchor - interval (calendar-aware timestamp math)
            let result = interval.subtract_from_timestamp_micros(anchor);
            (Some(result), true)
        }
        WindowFrameBound::IntervalBound(interval, WindowFrameDirection::Following) => {
            // anchor + interval
            let result = interval.add_to_timestamp_micros(anchor);
            (Some(result), true)
        }
    }
}

/// Extracts ORDER BY column values as i64 (microseconds for timestamps, or numeric values).
/// Partition slice [start..end) of the sorted column.
fn extract_order_values(order_col: &Column, start: usize, end: usize) -> Vec<i64> {
    let mut result = Vec::with_capacity(end - start);
    for i in start..end {
        let v = match &order_col.data {
            ColumnData::Int64(v) => v[i],
            ColumnData::Int32(v) => v[i] as i64,
            ColumnData::Int16(v) => v[i] as i64,
            ColumnData::Int8(v) => v[i] as i64,
            ColumnData::UInt32(v) => v[i] as i64,
            ColumnData::UInt64(v) => v[i] as i64,
            ColumnData::Float32(v) => v[i] as i64,
            ColumnData::Float64(v) => v[i] as i64,
            _ => 0,
        };
        result.push(v);
    }
    result
}

/// Converts a WindowFrameBound to an offset within the partition.
/// When `is_start` is true, returns the inclusive lower bound.
/// When false, returns the exclusive upper bound.
fn bound_to_offset(
    pos: usize,
    partition_len: usize,
    bound: WindowFrameBound,
    is_start: bool,
) -> isize {
    match bound {
        WindowFrameBound::CurrentRow => {
            if is_start {
                pos as isize
            } else {
                (pos + 1) as isize
            }
        }
        WindowFrameBound::Unbounded(WindowFrameDirection::Preceding) => 0,
        WindowFrameBound::Unbounded(WindowFrameDirection::Following) => partition_len as isize,
        WindowFrameBound::Offset(n, WindowFrameDirection::Preceding) => {
            let offset = pos as isize - n as isize;
            if is_start { offset } else { offset + 1 }
        }
        WindowFrameBound::Offset(n, WindowFrameDirection::Following) => {
            let offset = pos as isize + n as isize;
            if is_start { offset } else { offset + 1 }
        }
        // IntervalBound is only valid in RANGE mode and is resolved via
        // resolve_range_frame (Step 9). Treat as unbounded here for ROW mode to
        // avoid surprising truncation; the RANGE code path never calls this.
        WindowFrameBound::IntervalBound(_, WindowFrameDirection::Preceding) => 0,
        WindowFrameBound::IntervalBound(_, WindowFrameDirection::Following) => {
            partition_len as isize
        }
    }
}

/// Scatters values from `src` into a new column of the same type, using per-row source indices.
/// An index of -1 produces a null in the output.
fn scatter_column_by_index(src: &Column, indices: &[i64]) -> Result<Column> {
    let n = indices.len();
    let mut nulls = NullBitmap::none(n);

    // Collect non-negative indices for the reorder call, mapping -1 to a placeholder.
    let reorder_indices: Vec<u32> = indices
        .iter()
        .map(|&idx| if idx < 0 { 0 } else { idx as u32 })
        .collect();
    let reordered_data = src.data.take(&reorder_indices);

    // Mark nulls for -1 index positions and preserve source nulls for valid picks.
    for (i, &idx) in indices.iter().enumerate() {
        if idx < 0 {
            nulls.set_null(i);
        } else if src.nulls.is_null(idx as usize) {
            nulls.set_null(i);
        }
    }

    Ok(Column::with_nulls(reordered_data, nulls, src.type_id))
}

fn extract_f64_range(col: &Column, start: usize, end: usize) -> Result<Vec<f64>> {
    let mut out = Vec::with_capacity(end - start);
    for i in start..end {
        if col.nulls.is_null(i) {
            out.push(0.0);
            continue;
        }
        let val = match &col.data {
            ColumnData::Float64(v) => v[i],
            ColumnData::Float32(v) => v[i] as f64,
            ColumnData::Int64(v) => v[i] as f64,
            ColumnData::Int32(v) => v[i] as f64,
            ColumnData::Int16(v) => v[i] as f64,
            ColumnData::Int8(v) => v[i] as f64,
            ColumnData::UInt32(v) => v[i] as f64,
            ColumnData::UInt64(v) => v[i] as f64,
            _ => {
                return Err(ZyronError::ExecutionError(
                    "window function requires numeric column".into(),
                ));
            }
        };
        out.push(val);
    }
    Ok(out)
}

fn extract_i64_range(col: &Column, start: usize, end: usize) -> Result<Vec<i64>> {
    let mut out = Vec::with_capacity(end - start);
    for i in start..end {
        if col.nulls.is_null(i) {
            out.push(0);
            continue;
        }
        let val = match &col.data {
            ColumnData::Int64(v) => v[i],
            ColumnData::Int32(v) => v[i] as i64,
            ColumnData::Int16(v) => v[i] as i64,
            ColumnData::UInt32(v) => v[i] as i64,
            ColumnData::UInt64(v) => v[i] as i64,
            _ => {
                return Err(ZyronError::ExecutionError(
                    "window function requires integer time column".into(),
                ));
            }
        };
        out.push(val);
    }
    Ok(out)
}

fn extract_f64_scalar(col_opt: &Option<&Column>, row: usize) -> Result<Option<f64>> {
    let col = match col_opt {
        Some(c) => *c,
        None => return Ok(None),
    };
    if col.nulls.is_null(row) {
        return Ok(None);
    }
    let val = match &col.data {
        ColumnData::Float64(v) => v[row],
        ColumnData::Float32(v) => v[row] as f64,
        ColumnData::Int64(v) => v[row] as f64,
        ColumnData::Int32(v) => v[row] as f64,
        _ => return Ok(None),
    };
    Ok(Some(val))
}

fn extract_i64_scalar(col_opt: &Option<&Column>, row: usize) -> Result<Option<i64>> {
    let col = match col_opt {
        Some(c) => *c,
        None => return Ok(None),
    };
    if col.nulls.is_null(row) {
        return Ok(None);
    }
    let val = match &col.data {
        ColumnData::Int64(v) => v[row],
        ColumnData::Int32(v) => v[row] as i64,
        _ => return Ok(None),
    };
    Ok(Some(val))
}

// Suppress unused warnings for ScalarValue import - it's kept for future extension.
#[allow(dead_code)]
fn _use_scalar(_: ScalarValue) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_row_frame_unbounded_preceding_to_current() {
        let frame = WindowFrame {
            mode: WindowFrameMode::Rows,
            start: WindowFrameBound::Unbounded(WindowFrameDirection::Preceding),
            end: Some(WindowFrameBound::CurrentRow),
        };
        assert_eq!(resolve_row_frame(0, 10, &frame), (0, 1));
        assert_eq!(resolve_row_frame(5, 10, &frame), (0, 6));
        assert_eq!(resolve_row_frame(9, 10, &frame), (0, 10));
    }

    #[test]
    fn test_resolve_row_frame_n_preceding_to_current() {
        let frame = WindowFrame {
            mode: WindowFrameMode::Rows,
            start: WindowFrameBound::Offset(2, WindowFrameDirection::Preceding),
            end: Some(WindowFrameBound::CurrentRow),
        };
        // At pos 5 in partition of 10: rows 3..=5 (exclusive end 6)
        assert_eq!(resolve_row_frame(5, 10, &frame), (3, 6));
        // At pos 1: rows 0..=1 (clamped to partition start)
        assert_eq!(resolve_row_frame(1, 10, &frame), (0, 2));
        // At pos 0: just row 0
        assert_eq!(resolve_row_frame(0, 10, &frame), (0, 1));
    }

    #[test]
    fn test_resolve_row_frame_current_to_n_following() {
        let frame = WindowFrame {
            mode: WindowFrameMode::Rows,
            start: WindowFrameBound::CurrentRow,
            end: Some(WindowFrameBound::Offset(2, WindowFrameDirection::Following)),
        };
        // At pos 5 in partition of 10: rows 5..=7 (exclusive end 8)
        assert_eq!(resolve_row_frame(5, 10, &frame), (5, 8));
        // At pos 8: rows 8..=9 clamped to partition end
        assert_eq!(resolve_row_frame(8, 10, &frame), (8, 10));
    }

    #[test]
    fn test_resolve_row_frame_unbounded_both() {
        let frame = WindowFrame {
            mode: WindowFrameMode::Rows,
            start: WindowFrameBound::Unbounded(WindowFrameDirection::Preceding),
            end: Some(WindowFrameBound::Unbounded(WindowFrameDirection::Following)),
        };
        assert_eq!(resolve_row_frame(0, 10, &frame), (0, 10));
        assert_eq!(resolve_row_frame(5, 10, &frame), (0, 10));
    }

    #[test]
    fn test_bound_to_offset_current_start() {
        let b = WindowFrameBound::CurrentRow;
        assert_eq!(bound_to_offset(3, 10, b, true), 3);
        assert_eq!(bound_to_offset(3, 10, b, false), 4);
    }

    #[test]
    fn test_bound_to_offset_unbounded_preceding() {
        let b = WindowFrameBound::Unbounded(WindowFrameDirection::Preceding);
        assert_eq!(bound_to_offset(5, 10, b, true), 0);
    }

    #[test]
    fn test_bound_to_offset_unbounded_following() {
        let b = WindowFrameBound::Unbounded(WindowFrameDirection::Following);
        assert_eq!(bound_to_offset(5, 10, b, false), 10);
    }

    #[test]
    fn test_bound_to_offset_n_preceding() {
        let b = WindowFrameBound::Offset(3, WindowFrameDirection::Preceding);
        assert_eq!(bound_to_offset(5, 10, b, true), 2);
    }

    #[test]
    fn test_bound_to_offset_n_following() {
        let b = WindowFrameBound::Offset(2, WindowFrameDirection::Following);
        assert_eq!(bound_to_offset(5, 10, b, false), 8);
    }

    #[test]
    fn test_window_kind_returns() {
        assert!(matches!(
            window_function_kind("row_number").unwrap(),
            WindowOutputKind::Int64
        ));
        assert!(matches!(
            window_function_kind("rank").unwrap(),
            WindowOutputKind::Int64
        ));
        assert!(matches!(
            window_function_kind("ema").unwrap(),
            WindowOutputKind::Float64
        ));
        assert!(matches!(
            window_function_kind("lag").unwrap(),
            WindowOutputKind::MatchFirstArg
        ));
        assert!(window_function_kind("nonexistent").is_err());
    }

    #[test]
    fn test_window_kind_lead_lag_first_last() {
        assert!(matches!(
            window_function_kind("lead").unwrap(),
            WindowOutputKind::MatchFirstArg
        ));
        assert!(matches!(
            window_function_kind("first_value").unwrap(),
            WindowOutputKind::MatchFirstArg
        ));
        assert!(matches!(
            window_function_kind("last_value").unwrap(),
            WindowOutputKind::MatchFirstArg
        ));
        assert!(matches!(
            window_function_kind("nth_value").unwrap(),
            WindowOutputKind::MatchFirstArg
        ));
    }

    #[test]
    fn test_window_kind_ntile_cume_percent() {
        assert!(matches!(
            window_function_kind("ntile").unwrap(),
            WindowOutputKind::Int64
        ));
        assert!(matches!(
            window_function_kind("cume_dist").unwrap(),
            WindowOutputKind::Float64
        ));
        assert!(matches!(
            window_function_kind("percent_rank").unwrap(),
            WindowOutputKind::Float64
        ));
    }

    // ----- RANGE frame resolution -----

    #[test]
    fn test_resolve_range_frame_numeric_preceding() {
        // order_values: [10, 20, 30, 40, 50]
        // frame: RANGE BETWEEN 15 PRECEDING AND CURRENT ROW
        // At pos=2 (value 30): include rows where order >= 15 and <= 30 -> indices [1, 2, 3) (values 20, 30)
        let order_values = vec![10i64, 20, 30, 40, 50];
        let frame = WindowFrame {
            mode: WindowFrameMode::Range,
            start: WindowFrameBound::Offset(15, WindowFrameDirection::Preceding),
            end: Some(WindowFrameBound::CurrentRow),
        };
        let (lo, hi) = resolve_range_frame(2, 5, &frame, &order_values);
        assert_eq!(lo, 1);
        assert_eq!(hi, 3);
    }

    #[test]
    fn test_resolve_range_frame_unbounded_both() {
        let order_values = vec![10i64, 20, 30, 40, 50];
        let frame = WindowFrame {
            mode: WindowFrameMode::Range,
            start: WindowFrameBound::Unbounded(WindowFrameDirection::Preceding),
            end: Some(WindowFrameBound::Unbounded(WindowFrameDirection::Following)),
        };
        let (lo, hi) = resolve_range_frame(2, 5, &frame, &order_values);
        assert_eq!(lo, 0);
        assert_eq!(hi, 5);
    }

    #[test]
    fn test_resolve_range_frame_interval_preceding() {
        // Timestamps in microseconds: 0, 1h, 2h, 3h, 4h
        let hour_us: i64 = 3_600_000_000;
        let order_values = vec![0, hour_us, 2 * hour_us, 3 * hour_us, 4 * hour_us];
        // frame: RANGE BETWEEN INTERVAL '1 hour' PRECEDING AND CURRENT ROW
        let frame = WindowFrame {
            mode: WindowFrameMode::Range,
            start: WindowFrameBound::IntervalBound(
                zyron_common::Interval::from_nanoseconds(3_600_000_000_000),
                WindowFrameDirection::Preceding,
            ),
            end: Some(WindowFrameBound::CurrentRow),
        };
        // At pos=2 (2h): anchor=2h, lower=1h => rows [1, 2, 3) (1h, 2h)
        let (lo, hi) = resolve_range_frame(2, 5, &frame, &order_values);
        assert_eq!(lo, 1);
        assert_eq!(hi, 3);
    }

    #[test]
    fn test_resolve_range_frame_current_row_only() {
        let order_values = vec![10i64, 20, 30, 40, 50];
        let frame = WindowFrame {
            mode: WindowFrameMode::Range,
            start: WindowFrameBound::CurrentRow,
            end: Some(WindowFrameBound::CurrentRow),
        };
        let (lo, hi) = resolve_range_frame(2, 5, &frame, &order_values);
        assert_eq!(lo, 2);
        assert_eq!(hi, 3);
    }

    #[test]
    fn test_resolve_range_frame_ties() {
        // Multiple rows share the same order value: all should be included.
        // order_values: [10, 20, 20, 20, 30]
        // frame: RANGE BETWEEN CURRENT ROW AND CURRENT ROW
        // At pos=1 (value 20): should include all rows with order=20 -> [1, 4)
        let order_values = vec![10i64, 20, 20, 20, 30];
        let frame = WindowFrame {
            mode: WindowFrameMode::Range,
            start: WindowFrameBound::CurrentRow,
            end: Some(WindowFrameBound::CurrentRow),
        };
        let (lo, hi) = resolve_range_frame(1, 5, &frame, &order_values);
        assert_eq!(lo, 1);
        assert_eq!(hi, 4);
    }
}

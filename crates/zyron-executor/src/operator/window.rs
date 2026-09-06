//! Window function operator.
//!
//! Drains all input into one batch, and for each window expression lays
//! the rows out by partition and ORDER BY key, applies the function per
//! partition, and scatters the result back to input order. Appends one
//! result column per window expression to the output batch.
//!
//! The layout groups partitions by hash rather than sorting by the
//! partition key, since the output goes back to input order and only the
//! grouping matters, orders the rows by the ORDER BY keys once, and gathers
//! only the columns the function reads. The running aggregates over one
//! numeric argument fold in typed loops that write their output column
//! directly, and every other shape folds through the aggregate operator's
//! accumulators so the two never disagree.

use std::borrow::Cow;

use zyron_common::{Result, TypeId, ZyronError};
use zyron_parser::ast::{WindowFrame, WindowFrameBound, WindowFrameDirection, WindowFrameMode};
use zyron_planner::binder::{BoundExpr, BoundOrderBy};
use zyron_planner::logical::LogicalColumn;

use crate::batch::DataBatch;
use crate::column::{Column, ColumnData, NullBitmap, ScalarValue};
use crate::compute;
use crate::expr::evaluate_borrowed;
use crate::operator::aggregate::{
    Accumulator, GroupIndex, build_accumulator, coerce_aggregate_scalar, is_supported_aggregate,
};
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
    /// Query memory budget the buffered input reserves against. None runs
    /// unbudgeted.
    memory_budget: Option<std::sync::Arc<crate::context::QueryMemoryBudget>>,
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
            memory_budget: None,
        }
    }

    /// Attaches the query memory budget. Set by the operator builder from
    /// the execution context.
    pub fn set_memory_budget(
        &mut self,
        budget: Option<std::sync::Arc<crate::context::QueryMemoryBudget>>,
    ) {
        self.memory_budget = budget;
    }

    async fn materialize(&mut self) -> Result<()> {
        // Drain all input batches into a single combined batch.
        let mut combined_columns: Vec<Vec<Column>> = Vec::new();
        let mut total_rows = 0usize;

        while let Some(eb) = self.child.next().await? {
            if let Some(budget) = &self.memory_budget {
                budget.reserve(eb.batch.approx_bytes())?;
            }
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

        // The whole input as one batch, which every window expression
        // reads in place. Each has its own partition and order keys, so
        // each lays the rows out for itself
        let input = DataBatch::new(combined_columns.into_iter().map(concat_columns).collect());
        let mut window_columns: Vec<Column> = Vec::with_capacity(self.window_exprs.len());

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

            let layout =
                PartitionLayout::build(&input, &self.input_schema, partition_by, order_by)?;
            let window_output = evaluate_window_function(
                function,
                &input,
                &self.input_schema,
                &layout,
                order_by,
                frame,
                window_expr.type_id(),
            )?;
            // Scatter the window values back to their input positions
            let _scatter =
                zyron_common::profile::scope(zyron_common::profile::Phase::ExecWindowScatter);
            window_columns.push(unsort_column(&window_output, &layout.sorted));
        }

        let mut output_columns = input.columns;
        output_columns.extend(window_columns);
        self.result = Some(DataBatch::new(output_columns));
        Ok(())
    }
}

/// The rows of the input arranged for one window: grouped by partition,
/// each partition in ORDER BY order, with the partition boundaries in that
/// arrangement and the ORDER BY keys gathered into it
struct PartitionLayout {
    /// Input row of each sorted position
    sorted: Vec<u32>,
    /// Partition starts in sorted positions, ending with the row count
    boundaries: Vec<usize>,
    /// The ORDER BY key columns in sorted order, evaluated once
    order_cols: Vec<Column>,
}

impl PartitionLayout {
    fn build(
        input: &DataBatch,
        schema: &[LogicalColumn],
        partition_by: &[BoundExpr],
        order_by: &[BoundOrderBy],
    ) -> Result<Self> {
        let total = input.num_rows;
        let partition =
            zyron_common::profile::scope(zyron_common::profile::Phase::ExecWindowPartition);
        let (ids, partitions) = partition_ids(input, schema, partition_by)?;
        drop(partition);
        let _order = zyron_common::profile::scope(zyron_common::profile::Phase::ExecWindowOrder);

        let keys: Vec<Cow<'_, Column>> = order_by
            .iter()
            .map(|ob| evaluate_borrowed(&ob.expr, input, schema, &[]))
            .collect::<Result<Vec<_>>>()?;
        let ordered: Vec<u32> = if keys.is_empty() {
            (0..total as u32).collect()
        } else {
            let refs: Vec<&Column> = keys.iter().map(|c| c.as_ref()).collect();
            let ascending: Vec<bool> = order_by.iter().map(|ob| ob.asc).collect();
            let nulls_first: Vec<bool> = order_by.iter().map(|ob| ob.nulls_first).collect();
            compute::sort_indices_stable(&refs, &ascending, &nulls_first, total)
        };

        // A counting sort by partition id over the ordered rows. It is
        // stable, so it groups the partitions and keeps each one's order,
        // and its offsets are the partition boundaries
        let mut boundaries = vec![0usize; partitions + 1];
        for &row in &ordered {
            boundaries[ids[row as usize] as usize + 1] += 1;
        }
        for p in 0..partitions {
            boundaries[p + 1] += boundaries[p];
        }
        let mut next: Vec<usize> = boundaries[..partitions].to_vec();
        let mut sorted = vec![0u32; total];
        for &row in &ordered {
            let p = ids[row as usize] as usize;
            sorted[next[p]] = row;
            next[p] += 1;
        }

        let order_cols: Vec<Column> = keys.iter().map(|k| k.take(&sorted)).collect();
        Ok(Self {
            sorted,
            boundaries,
            order_cols,
        })
    }
}

/// Rows ahead of the one being placed whose bucket is prefetched
const PARTITION_PREFETCH_DISTANCE: usize = 16;

/// A dense partition id per input row, in order of first appearance, and
/// how many partitions there are. Rows with equal keys share an id, two
/// NULL keys counting as equal, which is how GROUP BY groups them
fn partition_ids(
    input: &DataBatch,
    schema: &[LogicalColumn],
    partition_by: &[BoundExpr],
) -> Result<(Vec<u32>, usize)> {
    let total = input.num_rows;
    if partition_by.is_empty() {
        return Ok((vec![0; total], 1));
    }
    let keys: Vec<Cow<'_, Column>> = partition_by
        .iter()
        .map(|expr| evaluate_borrowed(expr, input, schema, &[]))
        .collect::<Result<Vec<_>>>()?;
    let refs: Vec<&Column> = keys.iter().map(|c| c.as_ref()).collect();
    let hashes = compute::hash_column_batch(&refs, total);

    let mut index = GroupIndex::new();
    // The first row seen of each partition, which is what a later row's
    // key is compared against
    let mut representatives: Vec<u32> = Vec::new();
    let mut ids: Vec<u32> = Vec::with_capacity(total);
    for row in 0..total {
        let ahead = row + PARTITION_PREFETCH_DISTANCE;
        if ahead < total {
            index.prefetch(hashes[ahead]);
        }
        let id = match index.find(hashes[row], |p| {
            compute::rows_equal_typed(&refs, row, representatives[p] as usize)
        }) {
            Some(p) => p,
            None => {
                index.insert(hashes[row]);
                representatives.push(row as u32);
                representatives.len() - 1
            }
        };
        ids.push(id as u32);
    }
    Ok((ids, representatives.len()))
}

/// How two adjacent sorted rows are compared for a tie on the ORDER BY
/// keys, resolved once per fold rather than per row: whether a column has
/// nulls is a scan of its bitmap, which asked per pair is a pass over the
/// column for every row. One integer key with no nulls, the common case,
/// compares the values directly
enum PeerKey<'a> {
    Int64(&'a [i64]),
    General(&'a [Column]),
}

impl<'a> PeerKey<'a> {
    fn of(order_cols: &'a [Column]) -> Self {
        if let [col] = order_cols
            && let ColumnData::Int64(v) = &col.data
            && !col.nulls.has_nulls()
        {
            return PeerKey::Int64(v);
        }
        PeerKey::General(order_cols)
    }

    #[inline]
    fn tie(&self, a: usize, b: usize) -> bool {
        match self {
            PeerKey::Int64(v) => v[a] == v[b],
            PeerKey::General(cols) => cols
                .iter()
                .all(|c| compare_col_rows(c, a, b) == std::cmp::Ordering::Equal),
        }
    }
}

/// How the rows of a partition accumulate into each output row
#[derive(Clone, Copy)]
enum RunningShape {
    /// Every row of the partition folds into every output row
    Whole,
    /// Rows fold in one at a time, each output row seeing the rows up to
    /// and including itself
    Rows,
    /// Rows fold in a peer group at a time, rows tied on the ORDER BY keys
    /// sharing one output value
    Peers,
}

/// The running shape a frame describes, or None for a frame the
/// accumulator fold has to resolve row by row. A RANGE frame with no ORDER
/// BY is left to that fold as well, which is where it is refused
fn running_shape(order_cols: &[Column], frame: Option<&WindowFrame>) -> Option<RunningShape> {
    match frame {
        None if order_cols.is_empty() => Some(RunningShape::Whole),
        None => Some(RunningShape::Peers),
        Some(f) => {
            if !matches!(
                f.start,
                WindowFrameBound::Unbounded(WindowFrameDirection::Preceding)
            ) {
                return None;
            }
            match (f.mode, f.end) {
                (WindowFrameMode::Rows, None | Some(WindowFrameBound::CurrentRow)) => {
                    Some(RunningShape::Rows)
                }
                (WindowFrameMode::Range, None | Some(WindowFrameBound::CurrentRow))
                    if !order_cols.is_empty() =>
                {
                    Some(RunningShape::Peers)
                }
                _ => None,
            }
        }
    }
}

/// Folds each partition's rows into its output rows under a running shape,
/// `step` absorbing one row and `emit` reading the state, rows that share a
/// peer group taking one value. None from `emit` is a NULL output
#[allow(clippy::too_many_arguments)]
fn fold_running<S, T: Copy>(
    boundaries: &[usize],
    order_cols: &[Column],
    shape: RunningShape,
    mut fresh: impl FnMut() -> S,
    mut step: impl FnMut(&mut S, usize),
    mut emit: impl FnMut(&S) -> Result<Option<T>>,
    out: &mut [T],
    nulls: &mut NullBitmap,
) -> Result<()> {
    let mut write = |row: usize, value: Option<T>| match value {
        Some(v) => out[row] = v,
        None => nulls.set_null(row),
    };
    let peers = PeerKey::of(order_cols);
    for window in boundaries.windows(2) {
        let (start, end) = (window[0], window[1]);
        if end <= start {
            continue;
        }
        let mut state = fresh();
        match shape {
            RunningShape::Whole => {
                for row in start..end {
                    step(&mut state, row);
                }
                let value = emit(&state)?;
                for row in start..end {
                    write(row, value);
                }
            }
            RunningShape::Rows => {
                for row in start..end {
                    step(&mut state, row);
                    write(row, emit(&state)?);
                }
            }
            RunningShape::Peers => {
                let mut pos = start;
                while pos < end {
                    let mut peer_end = pos + 1;
                    while peer_end < end && peers.tie(peer_end - 1, peer_end) {
                        peer_end += 1;
                    }
                    for row in pos..peer_end {
                        step(&mut state, row);
                    }
                    let value = emit(&state)?;
                    for row in pos..peer_end {
                        write(row, value);
                    }
                    pos = peer_end;
                }
            }
        }
    }
    Ok(())
}

/// The running aggregates whose state is one number, folded in a typed
/// loop that writes the output column directly. None hands the shape to
/// the accumulator fold, which covers every aggregate and every frame
#[allow(clippy::too_many_arguments)]
fn running_typed(
    name: &str,
    args_len: usize,
    arg_col: Option<&Column>,
    boundaries: &[usize],
    order_cols: &[Column],
    frame: Option<&WindowFrame>,
    total_rows: usize,
    out_type: TypeId,
) -> Result<Option<Column>> {
    let Some(shape) = running_shape(order_cols, frame) else {
        return Ok(None);
    };
    let mut nulls = NullBitmap::none(total_rows);
    let column = match (name, arg_col, out_type) {
        ("count", None, TypeId::Int64) if args_len == 0 => {
            let mut out = vec![0i64; total_rows];
            fold_running(
                boundaries,
                order_cols,
                shape,
                || 0i64,
                |n, _| *n += 1,
                |n| Ok(Some(*n)),
                &mut out,
                &mut nulls,
            )?;
            Column::with_nulls(ColumnData::Int64(out), nulls, out_type)
        }
        ("count", Some(col), TypeId::Int64) => {
            let mut out = vec![0i64; total_rows];
            fold_running(
                boundaries,
                order_cols,
                shape,
                || 0i64,
                |n, row| {
                    if !col.is_null(row) {
                        *n += 1;
                    }
                },
                |n| Ok(Some(*n)),
                &mut out,
                &mut nulls,
            )?;
            Column::with_nulls(ColumnData::Int64(out), nulls, out_type)
        }
        ("sum", Some(col), TypeId::Int64) => {
            let ColumnData::Int64(v) = &col.data else {
                return Ok(None);
            };
            let mut out = vec![0i64; total_rows];
            fold_running(
                boundaries,
                order_cols,
                shape,
                || (0i128, false),
                |(sum, seen), row| {
                    if !col.is_null(row) {
                        *sum += v[row] as i128;
                        *seen = true;
                    }
                },
                |(sum, seen)| {
                    if !*seen {
                        return Ok(None);
                    }
                    i64::try_from(*sum).map(Some).map_err(|_| {
                        ZyronError::ExecutionError(format!(
                            "aggregate result of type {:?} does not fit output type {out_type:?}",
                            TypeId::Int128
                        ))
                    })
                },
                &mut out,
                &mut nulls,
            )?;
            Column::with_nulls(ColumnData::Int64(out), nulls, out_type)
        }
        ("sum", Some(col), TypeId::Float64) => {
            let ColumnData::Float64(v) = &col.data else {
                return Ok(None);
            };
            let mut out = vec![0f64; total_rows];
            fold_running(
                boundaries,
                order_cols,
                shape,
                || (0f64, false),
                |(sum, seen), row| {
                    if !col.is_null(row) {
                        *sum += v[row];
                        *seen = true;
                    }
                },
                |(sum, seen)| Ok(seen.then_some(*sum)),
                &mut out,
                &mut nulls,
            )?;
            Column::with_nulls(ColumnData::Float64(out), nulls, out_type)
        }
        ("avg", Some(col), TypeId::Float64) => {
            let mut out = vec![0f64; total_rows];
            let emit = |(sum, count): &(f64, i64)| Ok((*count > 0).then(|| *sum / *count as f64));
            match &col.data {
                ColumnData::Int64(v) => fold_running(
                    boundaries,
                    order_cols,
                    shape,
                    || (0f64, 0i64),
                    |(sum, count), row| {
                        if !col.is_null(row) {
                            *sum += v[row] as f64;
                            *count += 1;
                        }
                    },
                    emit,
                    &mut out,
                    &mut nulls,
                )?,
                ColumnData::Float64(v) => fold_running(
                    boundaries,
                    order_cols,
                    shape,
                    || (0f64, 0i64),
                    |(sum, count), row| {
                        if !col.is_null(row) {
                            *sum += v[row];
                            *count += 1;
                        }
                    },
                    emit,
                    &mut out,
                    &mut nulls,
                )?,
                _ => return Ok(None),
            }
            Column::with_nulls(ColumnData::Float64(out), nulls, out_type)
        }
        ("min" | "max", Some(col), TypeId::Int64) => {
            let ColumnData::Int64(v) = &col.data else {
                return Ok(None);
            };
            let is_max = name == "max";
            let mut out = vec![0i64; total_rows];
            fold_running(
                boundaries,
                order_cols,
                shape,
                || None::<i64>,
                |extreme, row| {
                    if !col.is_null(row) {
                        let x = v[row];
                        *extreme = Some(match *extreme {
                            None => x,
                            Some(cur) if is_max => cur.max(x),
                            Some(cur) => cur.min(x),
                        });
                    }
                },
                |extreme| Ok(*extreme),
                &mut out,
                &mut nulls,
            )?;
            Column::with_nulls(ColumnData::Int64(out), nulls, out_type)
        }
        ("min" | "max", Some(col), TypeId::Float64) => {
            let ColumnData::Float64(v) = &col.data else {
                return Ok(None);
            };
            // The float total order, NaN above every number and equal to
            // itself, so the running extreme does not depend on arrival
            let wanted = if name == "max" {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Less
            };
            let mut out = vec![0f64; total_rows];
            fold_running(
                boundaries,
                order_cols,
                shape,
                || None::<f64>,
                |extreme, row| {
                    if !col.is_null(row) {
                        let x = v[row];
                        *extreme = Some(match *extreme {
                            None => x,
                            Some(cur) if compute::cmp_f64_total(x, cur) == wanted => x,
                            Some(cur) => cur,
                        });
                    }
                },
                |extreme| Ok(*extreme),
                &mut out,
                &mut nulls,
            )?;
            Column::with_nulls(ColumnData::Float64(out), nulls, out_type)
        }
        _ => return Ok(None),
    };
    Ok(Some(column))
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
            fractional_digits: c.fractional_digits,
        })
        .collect();
    DataBatch::new(cols)
}

/// Computes sort indices that group partitions together and order rows within
/// each partition. Uses a stable sort by (partition_keys..., order_keys...).
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
        ColumnData::Float32(v) => crate::compute::cmp_f32_total(v[a], v[b]),
        ColumnData::Float64(v) => crate::compute::cmp_f64_total(v[a], v[b]),
        ColumnData::Utf8(v) => v[a].cmp(&v[b]),
        ColumnData::Binary(v) => v[a].cmp(&v[b]),
        ColumnData::FixedBinary16(v) => v[a].cmp(&v[b]),
        ColumnData::Interval(v) => v[a].cmp(&v[b]),
    }
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

/// Identifies which output column type a window function produces.
enum WindowOutputKind {
    /// Float64 column with null bitmap.
    Float64,
    /// Int64 column for ranking/row number functions.
    Int64,
    /// Columns whose type matches the first argument (LAG, LEAD, FIRST_VALUE, LAST_VALUE, NTH_VALUE).
    MatchFirstArg,
}

/// Computes a built-in aggregate (SUM/COUNT/AVG/MIN/MAX, etc.) as a window
/// function over each partition, reusing the same accumulators as the aggregate
/// operator so results never diverge. Produces a column of `out_type` in sorted
/// order (the caller unsorts it back to input order).
///
/// Frame semantics:
/// - explicit ROWS/RANGE frame: aggregate over each row's resolved frame.
/// - no frame, no ORDER BY: whole-partition aggregate broadcast to every row.
/// - no frame, with ORDER BY: running aggregate over RANGE UNBOUNDED PRECEDING
///   AND CURRENT ROW, so peer rows (equal order keys) share the cumulative value.
#[allow(clippy::too_many_arguments)]
fn compute_window_aggregate(
    name: &str,
    args_len: usize,
    arg_col: Option<&Column>,
    partition_boundaries: &[usize],
    order_by: &[BoundOrderBy],
    order_cols: &[Column],
    frame: Option<&WindowFrame>,
    total_rows: usize,
    out_type: TypeId,
) -> Result<Column> {
    if let Some(column) = running_typed(
        name,
        args_len,
        arg_col,
        partition_boundaries,
        order_cols,
        frame,
        total_rows,
        out_type,
    )? {
        return Ok(column);
    }

    let mut data = ColumnData::with_capacity(out_type, total_rows);
    let mut nulls = NullBitmap::empty();

    // Aggregate the argument column over the absolute row range [lo, hi).
    let agg_range = |lo: usize, hi: usize| -> Result<ScalarValue> {
        let mut acc = build_accumulator(name, args_len);
        if args_len == 0 {
            // COUNT(*): every row in the frame counts.
            acc.add_count(hi - lo);
        } else if let Some(col) = arg_col {
            for r in lo..hi {
                acc.update_typed(col, r);
            }
        }
        coerce_aggregate_scalar(acc.finalize(), out_type)
    };

    let push = |val: ScalarValue, data: &mut ColumnData, nulls: &mut NullBitmap| {
        nulls.push(val.is_null());
        data.push_scalar(&val);
    };

    for window in partition_boundaries.windows(2) {
        let start = window[0];
        let end = window[1];
        if end <= start {
            continue;
        }
        let plen = end - start;

        if let Some(f) = frame {
            // Explicit frame: aggregate each row's resolved [lo, hi)
            let range_state = if matches!(f.mode, WindowFrameMode::Range) {
                let oc = order_cols.first().ok_or_else(|| {
                    ZyronError::ExecutionError("RANGE frame requires ORDER BY".into())
                })?;
                let axis = build_range_axis(order_by, oc)?;
                let (values, value_nulls) = extract_order_values(oc, start, end)?;
                Some((values, value_nulls, axis))
            } else {
                None
            };
            let resolve = |pos: usize| -> Result<(usize, usize)> {
                match (f.mode, &range_state) {
                    (WindowFrameMode::Rows, _) => Ok(resolve_row_frame(pos, plen, f)),
                    (WindowFrameMode::Range, Some((values, value_nulls, axis))) => {
                        resolve_range_frame(pos, plen, f, values, value_nulls, axis)
                    }
                    (WindowFrameMode::Range, None) => Err(ZyronError::ExecutionError(
                        "RANGE frame requires ORDER BY".into(),
                    )),
                }
            };
            if matches!(
                f.start,
                WindowFrameBound::Unbounded(WindowFrameDirection::Preceding)
            ) {
                // A frame anchored at UNBOUNDED PRECEDING makes every row a
                // prefix fold, so one running accumulator absorbs each row
                // once and finalizes per row. The running fold feeds the same
                // rows in the same order as a fresh fold of [0, hi), so the
                // results are identical while the partition cost drops from
                // quadratic to linear. The refold accumulator covers an upper
                // edge that steps backward, which only occurs when RANGE
                // order values are not ascending
                let mut running = build_accumulator(name, args_len);
                let mut refold = build_accumulator(name, args_len);
                let mut absorbed = 0usize;
                for pos in 0..plen {
                    let (lo, hi) = resolve(pos)?;
                    let val = if hi > lo {
                        if hi >= absorbed {
                            while absorbed < hi {
                                if args_len == 0 {
                                    running.add_count(1);
                                } else if let Some(col) = arg_col {
                                    running.update_typed(col, start + absorbed);
                                }
                                absorbed += 1;
                            }
                            coerce_aggregate_scalar(running.finalize(), out_type)?
                        } else {
                            refold.reset();
                            fold_frame(refold.as_mut(), args_len, arg_col, start + lo, start + hi);
                            coerce_aggregate_scalar(refold.finalize(), out_type)?
                        }
                    } else {
                        ScalarValue::Null
                    };
                    push(val, &mut data, &mut nulls);
                }
            } else {
                // A sliding lower edge refolds each row's own range: float
                // aggregation is fold-order sensitive, so the overlap between
                // consecutive frames cannot be reused without changing
                // results. One reset accumulator keeps the refold free of
                // per-row allocation
                let mut acc = build_accumulator(name, args_len);
                for pos in 0..plen {
                    let (lo, hi) = resolve(pos)?;
                    let val = if hi > lo {
                        acc.reset();
                        fold_frame(acc.as_mut(), args_len, arg_col, start + lo, start + hi);
                        coerce_aggregate_scalar(acc.finalize(), out_type)?
                    } else {
                        ScalarValue::Null
                    };
                    push(val, &mut data, &mut nulls);
                }
            }
        } else if order_by.is_empty() {
            // Whole partition: one fold, broadcast to every row.
            let val = agg_range(start, end)?;
            for _ in 0..plen {
                push(val.clone(), &mut data, &mut nulls);
            }
        } else {
            // Running RANGE UNBOUNDED PRECEDING AND CURRENT ROW: advance one
            // accumulator across the partition, grouping peers so equal order
            // keys share the cumulative value.
            let mut acc = build_accumulator(name, args_len);
            let peers = PeerKey::of(order_cols);
            let mut pos = start;
            while pos < end {
                let mut peer_end = pos + 1;
                while peer_end < end && peers.tie(peer_end - 1, peer_end) {
                    peer_end += 1;
                }
                if args_len == 0 {
                    acc.add_count(peer_end - pos);
                } else if let Some(col) = arg_col {
                    for r in pos..peer_end {
                        acc.update_typed(col, r);
                    }
                }
                let val = coerce_aggregate_scalar(acc.finalize(), out_type)?;
                for _ in pos..peer_end {
                    push(val.clone(), &mut data, &mut nulls);
                }
                pos = peer_end;
            }
        }
    }

    Ok(Column::with_nulls(data, nulls, out_type))
}

/// Folds the absolute row range [lo, hi) into an existing accumulator
fn fold_frame(
    acc: &mut dyn Accumulator,
    args_len: usize,
    arg_col: Option<&Column>,
    lo: usize,
    hi: usize,
) {
    if args_len == 0 {
        acc.add_count(hi - lo);
    } else if let Some(col) = arg_col {
        for r in lo..hi {
            acc.update_typed(col, r);
        }
    }
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
#[allow(clippy::too_many_arguments)]
fn evaluate_window_function(
    function: &BoundExpr,
    input: &DataBatch,
    schema: &[LogicalColumn],
    layout: &PartitionLayout,
    order_by: &[BoundOrderBy],
    frame: Option<&WindowFrame>,
    result_type: TypeId,
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

    let total_rows = input.num_rows;

    // Argument columns, evaluated on the input as it arrived and gathered
    // into the window's order, so only what the function reads is moved
    let gather = zyron_common::profile::scope(zyron_common::profile::Phase::ExecWindowArgs);
    let mut arg_cols: Vec<Column> = Vec::with_capacity(args.len());
    for a in args {
        arg_cols.push(evaluate_borrowed(a, input, schema, &[])?.take(&layout.sorted));
    }
    drop(gather);
    let _fold = zyron_common::profile::scope(zyron_common::profile::Phase::ExecWindowFold);
    let partition_boundaries = layout.boundaries.as_slice();
    let order_cols = layout.order_cols.as_slice();

    // A built-in aggregate used as a window function (SUM/COUNT/AVG/MIN/MAX OVER)
    // folds the aggregate over each partition's frame. result_type is the
    // aggregate's declared output type.
    if is_supported_aggregate(&name) {
        return compute_window_aggregate(
            &name,
            args.len(),
            arg_cols.first(),
            partition_boundaries,
            order_by,
            order_cols,
            frame,
            total_rows,
            result_type,
        );
    }

    // The leading ORDER BY key, the time axis for rate and derivative and
    // the ranking key for rank and dense_rank
    let time_col: Option<&Column> = order_cols.first();

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
                        compute_rank(&mut result_data, start, end, order_cols, false);
                    }
                    "dense_rank" => {
                        compute_rank(&mut result_data, start, end, order_cols, true);
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
                                    let axis = build_range_axis(order_by, order_col)?;
                                    let (order_values, order_nulls) =
                                        extract_order_values(order_col, start, end)?;
                                    for i in 0..(end - start) {
                                        let (lo, hi) = resolve_range_frame(
                                            i,
                                            end - start,
                                            f,
                                            &order_values,
                                            &order_nulls,
                                            &axis,
                                        )?;
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
                        compute_rank(&mut ranks, start, end, order_cols, false);
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
                        compute_rank(&mut ranks, start, end, order_cols, false);
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
                                        let axis = build_range_axis(order_by, order_col)?;
                                        let (order_values, order_nulls) =
                                            extract_order_values(order_col, start, end)?;
                                        for i in 0..len {
                                            let (lo, hi) = resolve_range_frame(
                                                i,
                                                len,
                                                f,
                                                &order_values,
                                                &order_nulls,
                                                &axis,
                                            )?;
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
fn compute_rank(out: &mut [i64], start: usize, end: usize, order_cols: &[Column], dense: bool) {
    if end == start {
        return;
    }
    if order_cols.is_empty() {
        // Without ORDER BY, all rows tie at rank 1.
        for i in start..end {
            out[i] = 1;
        }
        return;
    }

    let peers = PeerKey::of(order_cols);
    let mut current_rank: i64 = 1;
    out[start] = 1;
    let mut tie_base: i64 = 1;
    for i in (start + 1)..end {
        if peers.tie(i - 1, i) {
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
/// The value axis a RANGE frame measures on: the ORDER BY key's direction,
/// the multiplier a plain-number offset scales by (ten to the scale for a
/// decimal, one million for a picosecond timestamp whose offsets are
/// microseconds), and the timestamp width interval bounds shift at
struct RangeAxis {
    asc: bool,
    offset_mul: i128,
    ts_us: bool,
    ts_ps: bool,
}

fn build_range_axis(order_by: &[BoundOrderBy], col: &Column) -> Result<RangeAxis> {
    let asc = order_by.first().map(|o| o.asc).unwrap_or(true);
    let is_ts = matches!(
        col.type_id,
        zyron_common::TypeId::Timestamp | zyron_common::TypeId::TimestampTz
    );
    let ps = is_ts && col.fractional_digits.unwrap_or(6) > 6;
    let offset_mul = if col.type_id == zyron_common::TypeId::Decimal {
        zyron_common::decimal::scale_factor(col.fractional_digits.unwrap_or(0))?
    } else if ps {
        1_000_000
    } else {
        1
    };
    Ok(RangeAxis {
        asc,
        offset_mul,
        ts_us: is_ts && !ps,
        ts_ps: ps,
    })
}

/// ORDER BY values for one partition, integer or float axis, with per-row
/// null flags. A non-numeric order key cannot measure a RANGE distance
/// and errors loudly instead of measuring everything as zero
enum AxisValues {
    Int(Vec<i128>),
    Float(Vec<f64>),
}

fn extract_order_values(
    order_col: &Column,
    start: usize,
    end: usize,
) -> Result<(AxisValues, Vec<bool>)> {
    let mut is_null = Vec::with_capacity(end - start);
    for i in start..end {
        is_null.push(order_col.is_null(i));
    }
    let values = match &order_col.data {
        ColumnData::Int64(v) => AxisValues::Int(v[start..end].iter().map(|&x| x as i128).collect()),
        ColumnData::Int32(v) => AxisValues::Int(v[start..end].iter().map(|&x| x as i128).collect()),
        ColumnData::Int16(v) => AxisValues::Int(v[start..end].iter().map(|&x| x as i128).collect()),
        ColumnData::Int8(v) => AxisValues::Int(v[start..end].iter().map(|&x| x as i128).collect()),
        ColumnData::UInt8(v) => AxisValues::Int(v[start..end].iter().map(|&x| x as i128).collect()),
        ColumnData::UInt16(v) => {
            AxisValues::Int(v[start..end].iter().map(|&x| x as i128).collect())
        }
        ColumnData::UInt32(v) => {
            AxisValues::Int(v[start..end].iter().map(|&x| x as i128).collect())
        }
        ColumnData::UInt64(v) => {
            AxisValues::Int(v[start..end].iter().map(|&x| x as i128).collect())
        }
        ColumnData::Int128(v) => AxisValues::Int(v[start..end].to_vec()),
        ColumnData::Float32(v) => {
            AxisValues::Float(v[start..end].iter().map(|&x| x as f64).collect())
        }
        ColumnData::Float64(v) => AxisValues::Float(v[start..end].to_vec()),
        _ => {
            return Err(ZyronError::ExecutionError(
                "RANGE frame requires a numeric or temporal ORDER BY key".into(),
            ));
        }
    };
    Ok((values, is_null))
}

/// Shifts an integer-axis anchor by an interval, calendar aware, at the
/// axis's timestamp width.
fn shift_anchor_by_interval(
    anchor: i128,
    interval: &zyron_common::Interval,
    add: bool,
    axis: &RangeAxis,
) -> Result<i128> {
    if axis.ts_ps {
        let us = anchor.div_euclid(1_000_000) as i64;
        let frac = anchor.rem_euclid(1_000_000);
        let calendar = zyron_common::Interval {
            months: interval.months,
            days: interval.days,
            nanoseconds: interval.nanoseconds - (interval.nanoseconds % 1_000),
        };
        let sub_us_ps = i128::from(interval.nanoseconds % 1_000) * 1_000;
        let shifted = if add {
            calendar.add_to_timestamp_micros(us)
        } else {
            calendar.subtract_from_timestamp_micros(us)
        };
        Ok(i128::from(shifted) * 1_000_000 + frac + if add { sub_us_ps } else { -sub_us_ps })
    } else if axis.ts_us {
        let base = anchor as i64;
        let shifted = if add {
            interval.add_to_timestamp_micros(base)
        } else {
            interval.subtract_from_timestamp_micros(base)
        };
        Ok(i128::from(shifted))
    } else {
        Err(ZyronError::ExecutionError(
            "a RANGE INTERVAL bound requires a timestamp ORDER BY key".into(),
        ))
    }
}

/// The value threshold one frame bound resolves to on the axis, None for
/// unbounded. `toward_start` is true for the frame's lower edge. Under a
/// descending order the value axis runs backward, so PRECEDING moves
/// toward larger values and FOLLOWING toward smaller
fn range_bound_threshold_int(
    anchor: i128,
    bound: WindowFrameBound,
    axis: &RangeAxis,
) -> Result<Option<i128>> {
    let scaled = |n: u64| (n as i128).saturating_mul(axis.offset_mul);
    Ok(match bound {
        WindowFrameBound::CurrentRow => Some(anchor),
        WindowFrameBound::Unbounded(_) => None,
        WindowFrameBound::Offset(n, WindowFrameDirection::Preceding) => Some(if axis.asc {
            anchor.saturating_sub(scaled(n))
        } else {
            anchor.saturating_add(scaled(n))
        }),
        WindowFrameBound::Offset(n, WindowFrameDirection::Following) => Some(if axis.asc {
            anchor.saturating_add(scaled(n))
        } else {
            anchor.saturating_sub(scaled(n))
        }),
        WindowFrameBound::IntervalBound(interval, WindowFrameDirection::Preceding) => Some(
            shift_anchor_by_interval(anchor, &interval, !axis.asc, axis)?,
        ),
        WindowFrameBound::IntervalBound(interval, WindowFrameDirection::Following) => {
            Some(shift_anchor_by_interval(anchor, &interval, axis.asc, axis)?)
        }
    })
}

fn range_bound_threshold_float(
    anchor: f64,
    bound: WindowFrameBound,
    axis: &RangeAxis,
) -> Result<Option<f64>> {
    Ok(match bound {
        WindowFrameBound::CurrentRow => Some(anchor),
        WindowFrameBound::Unbounded(_) => None,
        WindowFrameBound::Offset(n, WindowFrameDirection::Preceding) => Some(if axis.asc {
            anchor - n as f64
        } else {
            anchor + n as f64
        }),
        WindowFrameBound::Offset(n, WindowFrameDirection::Following) => Some(if axis.asc {
            anchor + n as f64
        } else {
            anchor - n as f64
        }),
        WindowFrameBound::IntervalBound(..) => {
            return Err(ZyronError::ExecutionError(
                "a RANGE INTERVAL bound requires a timestamp ORDER BY key".into(),
            ));
        }
    })
}

/// Resolves one row's RANGE frame to [lo, hi) partition indices.
///
/// The partition's order values are sorted in the query's declared
/// direction, so the searches run forward under ASC and reversed under
/// DESC. Nulls sort as one contiguous block at whichever end the null
/// placement put them: a null anchor's frame is its peer block (extended
/// by unbounded edges), a value bound from a non-null anchor never
/// reaches into the null block, and an unbounded edge always does
fn resolve_range_frame(
    pos: usize,
    partition_len: usize,
    frame: &WindowFrame,
    values: &AxisValues,
    nulls: &[bool],
    axis: &RangeAxis,
) -> Result<(usize, usize)> {
    let end_bound = frame.end.unwrap_or(WindowFrameBound::CurrentRow);
    // The null block is contiguous at the front or the back of the
    // partition, whichever the sort placed it at
    let leading_nulls = nulls.iter().take_while(|&&n| n).count();
    let trailing_nulls = if leading_nulls == partition_len {
        0
    } else {
        nulls.iter().rev().take_while(|&&n| n).count()
    };
    let nn_lo = leading_nulls;
    let nn_hi = partition_len - trailing_nulls;

    if nulls[pos] {
        let (block_lo, block_hi) = if pos < nn_lo {
            (0, nn_lo)
        } else {
            (nn_hi, partition_len)
        };
        let lo = match frame.start {
            WindowFrameBound::Unbounded(WindowFrameDirection::Preceding) => 0,
            _ => block_lo,
        };
        let hi = match end_bound {
            WindowFrameBound::Unbounded(WindowFrameDirection::Following) => partition_len,
            _ => block_hi,
        };
        return Ok((lo, hi));
    }

    match values {
        AxisValues::Int(v) => {
            let anchor = v[pos];
            let nonnull = &v[nn_lo..nn_hi];
            let lo = match range_bound_threshold_int(anchor, frame.start, axis)? {
                None => 0,
                Some(t) => {
                    nn_lo
                        + if axis.asc {
                            nonnull.partition_point(|x| *x < t)
                        } else {
                            nonnull.partition_point(|x| *x > t)
                        }
                }
            };
            let hi = match range_bound_threshold_int(anchor, end_bound, axis)? {
                None => partition_len,
                Some(t) => {
                    nn_lo
                        + if axis.asc {
                            nonnull.partition_point(|x| *x <= t)
                        } else {
                            nonnull.partition_point(|x| *x >= t)
                        }
                }
            };
            Ok((lo.min(partition_len), hi.min(partition_len)))
        }
        AxisValues::Float(v) => {
            let anchor = v[pos];
            let nonnull = &v[nn_lo..nn_hi];
            let lo = match range_bound_threshold_float(anchor, frame.start, axis)? {
                None => 0,
                Some(t) => {
                    nn_lo
                        + if axis.asc {
                            nonnull.partition_point(|x| *x < t)
                        } else {
                            nonnull.partition_point(|x| *x > t)
                        }
                }
            };
            let hi = match range_bound_threshold_float(anchor, end_bound, axis)? {
                None => partition_len,
                Some(t) => {
                    nn_lo
                        + if axis.asc {
                            nonnull.partition_point(|x| *x <= t)
                        } else {
                            nonnull.partition_point(|x| *x >= t)
                        }
                }
            };
            Ok((lo.min(partition_len), hi.min(partition_len)))
        }
    }
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

    /// Test axis: ascending, no scaling, microsecond timestamps.
    fn test_axis() -> RangeAxis {
        RangeAxis {
            asc: true,
            offset_mul: 1,
            ts_us: true,
            ts_ps: false,
        }
    }

    fn resolve_int(
        pos: usize,
        plen: usize,
        frame: &WindowFrame,
        order_values: &[i64],
    ) -> (usize, usize) {
        let values = AxisValues::Int(order_values.iter().map(|&x| x as i128).collect());
        let nulls = vec![false; order_values.len()];
        resolve_range_frame(pos, plen, frame, &values, &nulls, &test_axis()).unwrap()
    }

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
        let (lo, hi) = resolve_int(2, 5, &frame, &order_values);
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
        let (lo, hi) = resolve_int(2, 5, &frame, &order_values);
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
        let (lo, hi) = resolve_int(2, 5, &frame, &order_values);
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
        let (lo, hi) = resolve_int(2, 5, &frame, &order_values);
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
        let (lo, hi) = resolve_int(1, 5, &frame, &order_values);
        assert_eq!(lo, 1);
        assert_eq!(hi, 4);
    }
}

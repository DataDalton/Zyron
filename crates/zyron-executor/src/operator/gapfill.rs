//! Time-bucket gap fill.
//!
//! Densifies a time-bucketed aggregate: after the child (an aggregate grouped
//! by `time_bucket_gapfill(width, ts)`) produces one row per present bucket,
//! this operator emits a row for every bucket in the observed [min, max]
//! range stepping by `width`. Buckets with no input row get the bucket value
//! and NULL for every other column. The range is data-derived (bounded by the
//! observed extent) and capped so a pathological span cannot explode.

use std::collections::HashMap;

use zyron_common::{Result, ZyronError};

use crate::batch::{ColumnBuilder, DataBatch};
use crate::column::{Column, ColumnData, ScalarValue};
use crate::operator::{ExecutionBatch, Operator, OperatorResult};

/// Hard ceiling on synthesized buckets. A larger span is a query error rather
/// than an unbounded materialization.
const MAX_GAPFILL_BUCKETS: i128 = 5_000_000;

pub struct GapFillOperator {
    child: Box<dyn Operator>,
    bucket_col: usize,
    width: i128,
    finished: bool,
    result: Option<DataBatch>,
    output_cursor: usize,
}

impl GapFillOperator {
    pub fn new(child: Box<dyn Operator>, bucket_col: usize, width: i128) -> Self {
        Self {
            child,
            bucket_col,
            width,
            finished: false,
            result: None,
            output_cursor: 0,
        }
    }

    /// Reads the bucket value at `row` as an i128 (Int64 or Int128 column).
    fn bucket_at(col: &Column, row: usize) -> Option<i128> {
        if col.is_null(row) {
            return None;
        }
        match &col.data {
            ColumnData::Int64(v) => Some(v[row] as i128),
            ColumnData::Int128(v) => Some(v[row]),
            _ => None,
        }
    }

    async fn materialize(&mut self) -> Result<()> {
        if self.width <= 0 {
            return Err(ZyronError::ExecutionError(
                "time_bucket_gapfill width must be positive".to_string(),
            ));
        }
        // Drain the child (an aggregate, so one row per present bucket) while
        // folding the bucket index and [min,max] in the same pass. The dense
        // span is checked as it grows so a pathological range errors before
        // the full result is materialized, instead of after a second O(n)
        // pass. The good path is also one pass instead of two.
        let mut cols: Vec<Column> = Vec::new();
        let mut by_bucket: HashMap<i128, usize> = HashMap::new();
        let mut min_b: Option<i128> = None;
        let mut max_b: Option<i128> = None;
        let mut base_row: usize = 0;
        loop {
            match self.child.next().await? {
                Some(eb) => {
                    let b = eb.batch;
                    if b.num_rows == 0 {
                        continue;
                    }
                    if self.bucket_col < b.columns.len() {
                        let bc = &b.columns[self.bucket_col];
                        let n = bc.len();
                        for r in 0..n {
                            if let Some(bk) = Self::bucket_at(bc, r) {
                                by_bucket.entry(bk).or_insert(base_row + r);
                                min_b = Some(min_b.map_or(bk, |m| m.min(bk)));
                                max_b = Some(max_b.map_or(bk, |m| m.max(bk)));
                            }
                        }
                        if let (Some(a), Some(z)) = (min_b, max_b) {
                            // Early bound: reject before holding the full
                            // result if the observed span already exceeds the
                            // synthesizable cap.
                            let dense = (z - a) / self.width + 1;
                            if dense > MAX_GAPFILL_BUCKETS {
                                return Err(ZyronError::ExecutionError(format!(
                                    "time_bucket_gapfill would synthesize {dense} buckets \
                                     (cap {MAX_GAPFILL_BUCKETS}); narrow the range or \
                                     widen the bucket"
                                )));
                            }
                        }
                    }
                    base_row += b.num_rows;
                    if cols.is_empty() {
                        cols = b.columns;
                    } else {
                        for (i, c) in b.columns.into_iter().enumerate() {
                            cols[i].extend_from(&c);
                        }
                    }
                }
                None => break,
            }
        }

        let ncols = cols.len();
        if ncols == 0 || self.bucket_col >= ncols {
            self.result = Some(DataBatch::empty());
            return Ok(());
        }
        let nrows = cols[0].len();
        if nrows == 0 {
            self.result = Some(DataBatch::new(cols));
            return Ok(());
        }

        let (min_b, max_b) = match (min_b, max_b) {
            (Some(a), Some(b)) => (a, b),
            // No non-null bucket: nothing to densify.
            _ => {
                self.result = Some(DataBatch::new(cols));
                return Ok(());
            }
        };
        // Final dense count (the per-batch early check already bounded this).
        let span = max_b - min_b;
        let dense = span / self.width + 1;
        if dense > MAX_GAPFILL_BUCKETS {
            return Err(ZyronError::ExecutionError(format!(
                "time_bucket_gapfill would synthesize {dense} buckets \
                 (cap {MAX_GAPFILL_BUCKETS}); narrow the range or widen the bucket"
            )));
        }

        // Build the dense output via per-column builders.
        let mut builders: Vec<ColumnBuilder> = cols
            .iter()
            .map(|c| ColumnBuilder::new(c.type_id, dense as usize))
            .collect();
        let bucket_is_i128 = matches!(cols[self.bucket_col].data, ColumnData::Int128(_));

        let mut b = min_b;
        while b <= max_b {
            match by_bucket.get(&b) {
                Some(&src) => {
                    for (i, c) in cols.iter().enumerate() {
                        if c.is_null(src) {
                            builders[i].push_null();
                        } else {
                            builders[i].push(&c.data.get_scalar(src));
                        }
                    }
                }
                None => {
                    for (i, builder) in builders.iter_mut().enumerate() {
                        if i == self.bucket_col {
                            let bv = if bucket_is_i128 {
                                ScalarValue::Int128(b)
                            } else {
                                ScalarValue::Int64(b as i64)
                            };
                            builder.push(&bv);
                        } else {
                            // Absent bucket: aggregates are NULL.
                            builder.push_null();
                        }
                    }
                }
            }
            b += self.width;
        }

        let out_cols: Vec<Column> = builders
            .into_iter()
            .zip(cols.iter())
            .map(|(bld, src)| {
                let mut col = bld.finish();
                // Preserve timestamp precision metadata on the bucket column.
                col.ts_precision = src.ts_precision;
                col
            })
            .collect();
        self.result = Some(DataBatch::new(out_cols));
        Ok(())
    }
}

impl Operator for GapFillOperator {
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

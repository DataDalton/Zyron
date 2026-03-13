//! Sort operator for ordering results.
//!
//! Materializes all child output, computes sort indices, reorders
//! all data in a single take() pass, then emits output via cheap
//! contiguous slice() calls. Uses radix sort for integer key types.
//! Supports top-N via optional limit parameter.

use zyron_common::Result;
use zyron_planner::binder::{BoundExpr, BoundOrderBy};
use zyron_planner::logical::LogicalColumn;

use crate::batch::DataBatch;
use crate::column::Column;
use crate::compute;
use crate::expr::{evaluate, resolve_column_index};
use crate::operator::{ExecutionBatch, Operator, OperatorResult};

/// Sorts all input rows by the given order-by expressions.
/// Materializes the entire input before producing output.
/// If limit is set, only the top-N rows are retained.
pub struct SortOperator {
    child: Box<dyn Operator>,
    order_by: Vec<BoundOrderBy>,
    input_schema: Vec<LogicalColumn>,
    limit: Option<u64>,
    /// Fully sorted batch. Output is emitted via contiguous slice().
    sorted_batch: Option<DataBatch>,
    total_rows: usize,
    output_cursor: usize,
    finished: bool,
}

impl SortOperator {
    pub fn new(
        child: Box<dyn Operator>,
        order_by: Vec<BoundOrderBy>,
        input_schema: Vec<LogicalColumn>,
        limit: Option<u64>,
    ) -> Self {
        Self {
            child,
            order_by,
            input_schema,
            limit,
            sorted_batch: None,
            total_rows: 0,
            output_cursor: 0,
            finished: false,
        }
    }

    async fn materialize(&mut self) -> Result<()> {
        // Collect all input batches.
        let mut all_columns: Vec<Vec<Column>> = Vec::new();
        let mut total_rows = 0usize;

        loop {
            match self.child.next().await? {
                Some(eb) => {
                    total_rows += eb.batch.num_rows;
                    if all_columns.is_empty() {
                        all_columns.resize_with(eb.batch.num_columns(), Vec::new);
                    }
                    for (i, col) in eb.batch.columns.into_iter().enumerate() {
                        all_columns[i].push(col);
                    }
                }
                None => break,
            }
        }

        if total_rows == 0 {
            self.finished = true;
            return Ok(());
        }

        // Single-key integer ColumnRef: radix sort directly from batches.
        // Avoids concat (reads batch columns in-place) and avoids take
        // (extracts sorted values via reverse XOR transform).
        if self.order_by.len() == 1 {
            if let BoundExpr::ColumnRef(cr) = &self.order_by[0].expr {
                let key_idx = resolve_column_index(cr.table_idx, cr.column_id, &self.input_schema)?;
                let num_cols = all_columns.len();
                let key_batches = &all_columns[key_idx];
                let has_nulls = key_batches.iter().any(|c| c.nulls.has_nulls());

                if !has_nulls && num_cols == 1 {
                    // Single column: sort values in-place, no indices needed.
                    let type_id = key_batches[0].type_id;
                    let mut merged = concat_columns(key_batches);
                    compute::sort_column_inplace(&mut merged.data, self.order_by[0].asc);
                    if let Some(limit) = self.limit {
                        let limit = limit as usize;
                        if total_rows > limit {
                            merged.data = merged.data.slice(0, limit);
                            merged.nulls = crate::column::NullBitmap::none(limit);
                            merged.type_id = type_id;
                            self.total_rows = limit;
                            self.sorted_batch = Some(DataBatch::new(vec![merged]));
                            return Ok(());
                        }
                    }
                    self.total_rows = total_rows;
                    self.sorted_batch = Some(DataBatch::new(vec![merged]));
                    return Ok(());
                }

                if !has_nulls {
                    // Multi-column: radix sort with value extraction for key,
                    // take() for non-key columns.
                    let merged_key = concat_columns(key_batches);
                    if let Some((mut indices, sorted_key)) =
                        compute::radix_sort_contiguous(&merged_key, self.order_by[0].asc)
                    {
                        if let Some(limit) = self.limit {
                            indices.truncate(limit as usize);
                        }
                        let final_len = indices.len();
                        let key_type = all_columns[key_idx][0].type_id;
                        let mut sorted_key_opt = Some(if final_len < total_rows {
                            sorted_key.slice(0, final_len)
                        } else {
                            sorted_key
                        });
                        let idx_slice = &indices[..final_len];
                        let mut result_columns = Vec::with_capacity(num_cols);
                        for (col_idx, col_batches) in all_columns.iter().enumerate() {
                            if col_idx == key_idx {
                                result_columns
                                    .push(Column::new(sorted_key_opt.take().unwrap(), key_type));
                            } else {
                                let merged = concat_columns(col_batches);
                                result_columns.push(merged.take(idx_slice));
                            }
                        }
                        self.total_rows = final_len;
                        self.sorted_batch = Some(DataBatch::new(result_columns));
                        return Ok(());
                    }
                }
            }
        }

        // Fallback: concat all columns, sort_indices, take.
        let mut merged_columns: Vec<Column> = Vec::with_capacity(all_columns.len());
        for col_batches in &all_columns {
            merged_columns.push(concat_columns(col_batches));
        }
        let merged = DataBatch::new(merged_columns);

        // Evaluate sort key columns. For ColumnRef expressions, borrow
        // directly from the merged batch to avoid cloning the data.
        let mut ascending = Vec::with_capacity(self.order_by.len());
        let mut nulls_first = Vec::with_capacity(self.order_by.len());

        let mut key_sources: Vec<Option<usize>> = Vec::with_capacity(self.order_by.len());
        let mut owned_sort_columns: Vec<Column> = Vec::new();
        for ob in &self.order_by {
            ascending.push(ob.asc);
            nulls_first.push(ob.nulls_first);
            if let BoundExpr::ColumnRef(cr) = &ob.expr {
                let idx = resolve_column_index(cr.table_idx, cr.column_id, &self.input_schema)?;
                key_sources.push(Some(idx));
            } else {
                let col = evaluate(&ob.expr, &merged, &self.input_schema)?;
                key_sources.push(None);
                owned_sort_columns.push(col);
            }
        }

        let mut owned_idx = 0;
        let sort_refs: Vec<&Column> = key_sources
            .iter()
            .map(|src| match src {
                Some(idx) => &merged.columns[*idx],
                None => {
                    let col = &owned_sort_columns[owned_idx];
                    owned_idx += 1;
                    col
                }
            })
            .collect();

        let mut indices = compute::sort_indices(&sort_refs, &ascending, &nulls_first, total_rows);

        // Apply top-N limit.
        if let Some(limit) = self.limit {
            let limit = limit as usize;
            if indices.len() > limit {
                indices.truncate(limit);
            }
        }

        // Single full take() pass: reorder all data once. Output is then
        // emitted via contiguous slice() calls which are cheap memcpy.
        self.total_rows = indices.len();
        self.sorted_batch = Some(merged.take(&indices));
        Ok(())
    }
}

impl Operator for SortOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            if self.finished {
                return Ok(None);
            }

            if self.sorted_batch.is_none() && self.output_cursor == 0 {
                self.materialize().await?;
            }

            let Some(ref sorted) = self.sorted_batch else {
                self.finished = true;
                return Ok(None);
            };

            if self.output_cursor >= self.total_rows {
                self.finished = true;
                self.sorted_batch = None;
                return Ok(None);
            }

            let remaining = self.total_rows - self.output_cursor;
            // Use larger output chunks to reduce slice() allocation overhead.
            // Sort materializes all data up front, so emitting larger batches
            // reduces per-batch overhead without affecting streaming behavior.
            const SORT_OUTPUT_CHUNK: usize = 8192;
            let chunk_size = remaining.min(SORT_OUTPUT_CHUNK);
            let batch = sorted.slice(self.output_cursor, chunk_size);
            self.output_cursor += chunk_size;

            Ok(Some(ExecutionBatch::new(batch)))
        })
    }
}

/// Concatenates multiple columns of the same type into one.
/// Uses typed bulk extend_from to avoid per-row ScalarValue allocation.
fn concat_columns(columns: &[Column]) -> Column {
    if columns.is_empty() {
        return Column::null_column(zyron_common::TypeId::Null, 0);
    }
    if columns.len() == 1 {
        return columns[0].clone();
    }

    let type_id = columns[0].type_id;
    let total_len: usize = columns.iter().map(|c| c.len()).sum();

    let mut data = crate::column::ColumnData::with_capacity(type_id, total_len);
    let mut nulls = crate::column::NullBitmap::empty();

    for col in columns {
        data.extend_from(&col.data);
        nulls.extend_from(&col.nulls);
    }

    Column::with_nulls(data, nulls, type_id)
}

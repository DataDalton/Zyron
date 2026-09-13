//! Hash-based distinct operator for duplicate elimination.
//!
//! Uses typed hashing and columnar storage for seen rows instead of
//! HashSet<Vec<ScalarValue>>. Rows are hashed in batches and stored in
//! column builders for collision resolution via typed equality, and found
//! again through the flat chained index the aggregate keeps its groups in.

use crate::column::Column;
use crate::compute;
use crate::operator::aggregate::GroupIndex;
use crate::operator::{ExecutionBatch, Operator, OperatorResult};

/// Rows ahead of the one being checked whose bucket is prefetched
const SEEN_PREFETCH_DISTANCE: usize = 16;

/// Eliminates duplicate rows using typed hash-based deduplication.
/// Stores seen rows in columnar format for collision resolution.
pub struct HashDistinctOperator {
    child: Box<dyn Operator>,
    /// Where a seen row is found from its hash
    index: GroupIndex,
    /// Columnar storage of all distinct rows seen so far.
    seen_columns: Vec<Column>,
    seen_count: usize,
    initialized: bool,
}

impl HashDistinctOperator {
    pub fn new(child: Box<dyn Operator>) -> Self {
        Self {
            child,
            index: GroupIndex::new(),
            seen_columns: Vec::new(),
            seen_count: 0,
            initialized: false,
        }
    }
}

impl Operator for HashDistinctOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            loop {
                let input = self.child.next().await?;
                let Some(exec_batch) = input else {
                    return Ok(None);
                };

                let batch = &exec_batch.batch;
                let num_cols = batch.num_columns();
                let num_rows = batch.num_rows;

                // Initialize seen columns on first batch.
                if !self.initialized {
                    self.seen_columns = batch
                        .columns
                        .iter()
                        .map(|c| {
                            Column::new_ts(
                                crate::column::ColumnData::with_capacity_for(
                                    c.type_id,
                                    c.fractional_digits,
                                    64,
                                ),
                                c.type_id,
                                c.fractional_digits,
                            )
                        })
                        .collect();
                    self.initialized = true;
                }

                // Batch hash all rows.
                let col_refs: Vec<&Column> = batch.columns.iter().collect();
                let hashes = compute::hash_column_batch(&col_refs, num_rows);

                let mut mask = Vec::with_capacity(num_rows);

                for row in 0..num_rows {
                    // The hashes are all in hand, so the bucket a later row
                    // needs is fetched while this one is compared
                    let ahead = row + SEEN_PREFETCH_DISTANCE;
                    if ahead < num_rows {
                        self.index.prefetch(hashes[ahead]);
                    }
                    let hash = hashes[row];
                    let seen_columns = &self.seen_columns;
                    let found = self
                        .index
                        .find(hash, |seen_idx| {
                            seen_row_equals(batch, row, seen_columns, seen_idx, num_cols)
                        })
                        .is_some();

                    if found {
                        mask.push(false);
                    } else {
                        // Add to seen store.
                        self.index.insert(hash);
                        for ci in 0..num_cols {
                            self.seen_columns[ci].push_row_from(&batch.columns[ci], row);
                        }
                        self.seen_count += 1;
                        mask.push(true);
                    }
                }

                let filtered = batch.filter(&mask);
                if filtered.num_rows == 0 {
                    continue;
                }

                return Ok(Some(ExecutionBatch::new(filtered)));
            }
        })
    }
}

/// Whether a batch row equals a stored row, two NULLs counting as equal
/// the way DISTINCT reads them
#[inline]
fn seen_row_equals(
    batch: &crate::batch::DataBatch,
    row: usize,
    seen_columns: &[Column],
    seen_idx: usize,
    num_cols: usize,
) -> bool {
    for ci in 0..num_cols {
        let a_null = batch.columns[ci].is_null(row);
        let b_null = seen_columns[ci].is_null(seen_idx);
        if a_null != b_null {
            return false;
        }
        if a_null {
            continue;
        }
        if !column_values_equal_cross(
            &batch.columns[ci].data,
            row,
            &seen_columns[ci].data,
            seen_idx,
        ) {
            return false;
        }
    }
    true
}

/// Compares values at different indices across two ColumnData instances of the same type.
#[inline]
fn column_values_equal_cross(
    a: &crate::column::ColumnData,
    a_idx: usize,
    b: &crate::column::ColumnData,
    b_idx: usize,
) -> bool {
    use crate::column::ColumnData;
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
        _ => false,
    }
}

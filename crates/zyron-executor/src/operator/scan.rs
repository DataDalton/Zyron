//! Sequential, parallel, and index scan operators.
//!
//! SeqScanOperator reads heap pages one at a time, decodes visible tuples
//! into columnar batches, and optionally applies a predicate filter.
//! ParallelSeqScanOperator splits the page range across multiple tokio tasks
//! for multi-core throughput on large tables.
//! IndexScanOperator uses B+ tree range scans to collect matching TupleIds,
//! then fetches only those tuples from the heap. Falls back to sequential
//! scan with predicate filtering when no B+ tree instance is registered.

use std::sync::Arc;

use zyron_catalog::{IndexEntry, TableEntry};
use zyron_common::Result;
use zyron_common::page::{PAGE_SIZE, PageId};
use zyron_parser::ast::{BinaryOperator, LiteralValue};
use zyron_planner::binder::BoundExpr;
use zyron_planner::logical::LogicalColumn;
use zyron_storage::{BTreeIndex, HeapPage, TupleId};

use crate::batch::{
    BATCH_SIZE, DataBatch, create_builders, decode_tuple_into_builders, finalize_builders,
};
use crate::compute::column_to_mask;
use crate::context::ExecutionContext;
use crate::expr::evaluate;
use crate::operator::{ExecutionBatch, Operator, OperatorResult};

/// Minimum number of pages before parallel scan is used.
/// Below this threshold, the task spawn overhead outweighs the benefit.
const PARALLEL_SCAN_MIN_PAGES: u64 = 64;

// ---------------------------------------------------------------------------
// Sequential scan
// ---------------------------------------------------------------------------

/// Reads all visible tuples from a heap file, one page at a time.
/// Produces DataBatch batches of up to batch_size rows. An optional
/// predicate is evaluated after decoding and applied as a post-filter.
pub struct SeqScanOperator {
    ctx: Arc<ExecutionContext>,
    table_entry: Arc<TableEntry>,
    output_columns: Vec<LogicalColumn>,
    predicate: Option<BoundExpr>,
    page_cursor: u64,
    num_pages: u64,
    finished: bool,
    track_tuple_ids: bool,
    /// When set, use version-based visibility instead of MVCC snapshot.
    as_of_version: Option<u64>,
}

impl SeqScanOperator {
    /// Creates a new sequential scan operator for the given table.
    pub async fn new(
        ctx: Arc<ExecutionContext>,
        table_id: zyron_catalog::TableId,
        columns: Vec<LogicalColumn>,
        predicate: Option<BoundExpr>,
        track_tuple_ids: bool,
        as_of_version: Option<u64>,
    ) -> Result<Self> {
        let table_entry = ctx.get_table_entry(table_id)?;
        let num_pages = ctx.disk_manager.num_pages(table_entry.heap_file_id).await?;

        Ok(Self {
            ctx,
            table_entry,
            output_columns: columns,
            predicate,
            page_cursor: 0,
            num_pages,
            finished: false,
            track_tuple_ids,
            as_of_version,
        })
    }
}

impl Operator for SeqScanOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            if self.finished {
                return Ok(None);
            }
            self.ctx.check_cancelled()?;

            let batch_size = self.ctx.batch_size;
            let mut builders = create_builders(&self.output_columns, batch_size);
            let mut tuple_ids: Vec<TupleId> = if self.track_tuple_ids {
                Vec::with_capacity(batch_size)
            } else {
                Vec::new()
            };
            let mut row_count: usize = 0;

            while row_count < batch_size && self.page_cursor < self.num_pages {
                let page_id = PageId::new(self.table_entry.heap_file_id, self.page_cursor);
                self.page_cursor += 1;

                let page_data: [u8; PAGE_SIZE] = self.ctx.disk_manager.read_page(page_id).await?;
                let page = HeapPage::from_bytes(page_data);

                for (slot_id, tuple) in page.iter() {
                    if tuple.is_deleted() {
                        continue;
                    }
                    // Version-based visibility for time travel queries,
                    // MVCC snapshot visibility for normal queries.
                    //
                    // When as_of_version is set, the tuple's base header
                    // xmin/xmax are reinterpreted as version bounds via
                    // is_visible (version_id <= target, deleted_at > target).
                    // This works because versioned tables store version_id
                    // in xmin and deleted_at_version in xmax on the base
                    // TupleHeader, keeping the same visibility predicate shape.
                    //
                    // Non-versioned tuples in a time travel query fall back
                    // to MVCC snapshot visibility as a safety measure.
                    if let Some(target_version) = self.as_of_version {
                        if tuple.header().flags.has_version() {
                            if !tuple.header().is_visible(target_version as u32) {
                                continue;
                            }
                        } else if !tuple.header().is_visible_to(&self.ctx.snapshot) {
                            continue;
                        }
                    } else if !tuple.header().is_visible_to(&self.ctx.snapshot) {
                        continue;
                    }

                    decode_tuple_into_builders(
                        tuple.data(),
                        &self.table_entry.columns,
                        &mut builders,
                    );

                    if self.track_tuple_ids {
                        tuple_ids.push(TupleId::new(page_id, slot_id.0));
                    }

                    row_count += 1;
                    if row_count >= batch_size {
                        break;
                    }
                }
            }

            if row_count == 0 {
                self.finished = true;
                return Ok(None);
            }

            let batch = finalize_builders(builders);

            // Apply predicate filter if present.
            if let Some(ref predicate) = self.predicate {
                let mask_col = evaluate(predicate, &batch, &self.output_columns, &[])?;
                let mask = column_to_mask(&mask_col);

                let filtered = batch.filter(&mask);

                if self.track_tuple_ids {
                    let filtered_ids: Vec<TupleId> = mask
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &keep)| if keep { Some(tuple_ids[i]) } else { None })
                        .collect();
                    return Ok(Some(ExecutionBatch::with_tuple_ids(filtered, filtered_ids)));
                }

                return Ok(Some(ExecutionBatch::new(filtered)));
            }

            if self.track_tuple_ids {
                Ok(Some(ExecutionBatch::with_tuple_ids(batch, tuple_ids)))
            } else {
                Ok(Some(ExecutionBatch::new(batch)))
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Parallel sequential scan
// ---------------------------------------------------------------------------

/// Multi-threaded sequential scan that divides the page range across
/// multiple tokio tasks. Each worker scans its assigned pages, decodes
/// visible tuples, applies the predicate, and sends result batches
/// through an MPSC channel. The operator's next() receives from the
/// channel, providing multi-core throughput for large table scans.
///
/// Not used for tuple ID tracking (DML operations need ordered IDs).
pub struct ParallelSeqScanOperator {
    receiver: tokio::sync::mpsc::Receiver<Result<DataBatch>>,
    finished: bool,
}

impl ParallelSeqScanOperator {
    /// Creates a parallel scan operator. Spawns worker tasks immediately.
    /// Each worker scans a contiguous slice of the table's pages.
    pub async fn new(
        ctx: Arc<ExecutionContext>,
        table_id: zyron_catalog::TableId,
        columns: Vec<LogicalColumn>,
        predicate: Option<BoundExpr>,
    ) -> Result<Self> {
        let table_entry = ctx.get_table_entry(table_id)?;
        let num_pages = ctx.disk_manager.num_pages(table_entry.heap_file_id).await?;

        let num_workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .min(num_pages as usize)
            .max(1);

        let pages_per_worker = (num_pages + num_workers as u64 - 1) / num_workers as u64;

        // Channel capacity: 2 batches per worker to keep workers busy
        // without unbounded buffering.
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<DataBatch>>(num_workers * 2);

        for worker_id in 0..num_workers {
            let start_page = worker_id as u64 * pages_per_worker;
            let end_page = ((worker_id as u64 + 1) * pages_per_worker).min(num_pages);

            if start_page >= end_page {
                continue;
            }

            let tx = tx.clone();
            let ctx = ctx.clone();
            let table_entry = table_entry.clone();
            let columns = columns.clone();
            let predicate = predicate.clone();

            tokio::spawn(async move {
                let result = scan_page_range(
                    &ctx,
                    &table_entry,
                    &columns,
                    predicate.as_ref(),
                    start_page,
                    end_page,
                    &tx,
                )
                .await;

                // If the scan itself errored, send the error through the channel.
                if let Err(e) = result {
                    let _ = tx.send(Err(e)).await;
                }
            });
        }

        Ok(Self {
            receiver: rx,
            finished: false,
        })
    }
}

/// Scans a contiguous range of pages, decodes visible tuples, applies
/// the predicate filter, and sends result batches through the channel.
async fn scan_page_range(
    ctx: &ExecutionContext,
    table_entry: &TableEntry,
    output_columns: &[LogicalColumn],
    predicate: Option<&BoundExpr>,
    start_page: u64,
    end_page: u64,
    tx: &tokio::sync::mpsc::Sender<Result<DataBatch>>,
) -> Result<()> {
    let batch_size = BATCH_SIZE;
    let mut page_cursor = start_page;

    while page_cursor < end_page {
        ctx.check_cancelled()?;

        let mut builders = create_builders(output_columns, batch_size);
        let mut row_count = 0usize;

        while row_count < batch_size && page_cursor < end_page {
            let page_id = PageId::new(table_entry.heap_file_id, page_cursor);
            page_cursor += 1;

            let page_data: [u8; PAGE_SIZE] = ctx.disk_manager.read_page(page_id).await?;
            let page = HeapPage::from_bytes(page_data);

            for (_slot_id, tuple) in page.iter() {
                if tuple.is_deleted() {
                    continue;
                }
                if !tuple.header().is_visible_to(&ctx.snapshot) {
                    continue;
                }

                decode_tuple_into_builders(tuple.data(), &table_entry.columns, &mut builders);

                row_count += 1;
                if row_count >= batch_size {
                    break;
                }
            }
        }

        if row_count == 0 {
            break;
        }

        let batch = finalize_builders(builders);

        // Apply predicate filter if present.
        let output = if let Some(pred) = predicate {
            let mask_col = evaluate(pred, &batch, output_columns, &[])?;
            let mask = column_to_mask(&mask_col);
            let filtered = batch.filter(&mask);
            if filtered.num_rows == 0 {
                continue;
            }
            filtered
        } else {
            batch
        };

        // If the receiver has been dropped (query cancelled), stop scanning.
        if tx.send(Ok(output)).await.is_err() {
            break;
        }
    }

    Ok(())
}

impl Operator for ParallelSeqScanOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            if self.finished {
                return Ok(None);
            }

            match self.receiver.recv().await {
                Some(Ok(batch)) => Ok(Some(ExecutionBatch::new(batch))),
                Some(Err(e)) => {
                    self.finished = true;
                    Err(e)
                }
                None => {
                    self.finished = true;
                    Ok(None)
                }
            }
        })
    }
}

/// Determines whether a parallel scan should be used for the given table.
/// Returns true when the table has enough pages to benefit from parallelism
/// and tuple ID tracking is not required.
pub fn should_use_parallel_scan(num_pages: u64, track_tuple_ids: bool) -> bool {
    !track_tuple_ids && num_pages >= PARALLEL_SCAN_MIN_PAGES
}

// ---------------------------------------------------------------------------
// Index scan
// ---------------------------------------------------------------------------

/// Key range bounds extracted from an index predicate for B+ tree lookup.
struct ScanBounds {
    start_key: Option<Vec<u8>>,
    end_key: Option<Vec<u8>>,
}

/// Serializes a LiteralValue to big-endian bytes for B+ tree key comparison.
/// Returns None for types that cannot be used as index keys.
fn literal_to_key_bytes(value: &LiteralValue) -> Option<Vec<u8>> {
    match value {
        LiteralValue::Integer(v) => Some((*v as u64).to_be_bytes().to_vec()),
        LiteralValue::Float(v) => {
            // IEEE 754 float-to-sortable-bytes encoding.
            let bits = v.to_bits();
            let sortable = if bits >> 63 == 1 {
                !bits
            } else {
                bits ^ (1u64 << 63)
            };
            Some(sortable.to_be_bytes().to_vec())
        }
        LiteralValue::String(s) => Some(s.as_bytes().to_vec()),
        LiteralValue::Boolean(b) => Some(vec![*b as u8]),
        LiteralValue::Null => None,
        LiteralValue::Interval(i) => Some(i.to_le_bytes().to_vec()),
    }
}

/// Extracts start/end key bounds from an index predicate.
/// Handles equality, less-than, greater-than, and BETWEEN on a single
/// column matching the first column of the index.
///
/// Predicates that cannot be decomposed into range bounds (complex AND
/// trees, OR, functions) return an unbounded scan, letting the remaining
/// predicate handle correctness via post-filtering.
fn extract_scan_bounds(predicate: &BoundExpr, index: &IndexEntry) -> ScanBounds {
    if index.columns.is_empty() {
        return ScanBounds {
            start_key: None,
            end_key: None,
        };
    }
    let index_col_id = index.columns[0].column_id;

    match predicate {
        // col = literal or literal = col
        BoundExpr::BinaryOp {
            left,
            op: BinaryOperator::Eq,
            right,
            ..
        } => {
            if let Some(bytes) = match_column_literal(left, right, index_col_id) {
                return ScanBounds {
                    start_key: Some(bytes.clone()),
                    end_key: Some(bytes),
                };
            }
        }
        // col > literal
        BoundExpr::BinaryOp {
            left,
            op: BinaryOperator::Gt,
            right,
            ..
        } => {
            if let Some(bytes) = match_column_op_literal(left, right, index_col_id) {
                // Start just after this key. For integer keys, increment by 1.
                let start = increment_key(&bytes);
                return ScanBounds {
                    start_key: Some(start),
                    end_key: None,
                };
            }
            // literal > col means col < literal
            if let Some(bytes) = match_literal_op_column(left, right, index_col_id) {
                let end = decrement_key(&bytes);
                return ScanBounds {
                    start_key: None,
                    end_key: Some(end),
                };
            }
        }
        // col >= literal
        BoundExpr::BinaryOp {
            left,
            op: BinaryOperator::GtEq,
            right,
            ..
        } => {
            if let Some(bytes) = match_column_op_literal(left, right, index_col_id) {
                return ScanBounds {
                    start_key: Some(bytes),
                    end_key: None,
                };
            }
            if let Some(bytes) = match_literal_op_column(left, right, index_col_id) {
                return ScanBounds {
                    start_key: None,
                    end_key: Some(bytes),
                };
            }
        }
        // col < literal
        BoundExpr::BinaryOp {
            left,
            op: BinaryOperator::Lt,
            right,
            ..
        } => {
            if let Some(bytes) = match_column_op_literal(left, right, index_col_id) {
                let end = decrement_key(&bytes);
                return ScanBounds {
                    start_key: None,
                    end_key: Some(end),
                };
            }
            if let Some(bytes) = match_literal_op_column(left, right, index_col_id) {
                let start = increment_key(&bytes);
                return ScanBounds {
                    start_key: Some(start),
                    end_key: None,
                };
            }
        }
        // col <= literal
        BoundExpr::BinaryOp {
            left,
            op: BinaryOperator::LtEq,
            right,
            ..
        } => {
            if let Some(bytes) = match_column_op_literal(left, right, index_col_id) {
                return ScanBounds {
                    start_key: None,
                    end_key: Some(bytes),
                };
            }
            if let Some(bytes) = match_literal_op_column(left, right, index_col_id) {
                return ScanBounds {
                    start_key: Some(bytes),
                    end_key: None,
                };
            }
        }
        // col BETWEEN low AND high
        BoundExpr::Between {
            expr,
            low,
            high,
            negated: false,
        } => {
            if matches_index_column(expr, index_col_id) {
                let start = extract_literal_bytes(low);
                let end = extract_literal_bytes(high);
                if start.is_some() || end.is_some() {
                    return ScanBounds {
                        start_key: start,
                        end_key: end,
                    };
                }
            }
        }
        // AND: intersect bounds from both sides
        BoundExpr::BinaryOp {
            left,
            op: BinaryOperator::And,
            right,
            ..
        } => {
            let left_bounds = extract_scan_bounds(left, index);
            let right_bounds = extract_scan_bounds(right, index);
            return ScanBounds {
                start_key: pick_later_key(left_bounds.start_key, right_bounds.start_key),
                end_key: pick_earlier_key(left_bounds.end_key, right_bounds.end_key),
            };
        }
        _ => {}
    }

    ScanBounds {
        start_key: None,
        end_key: None,
    }
}

/// Returns true if the expression is a ColumnRef matching the given column ID.
fn matches_index_column(expr: &BoundExpr, col_id: zyron_catalog::ColumnId) -> bool {
    matches!(expr, BoundExpr::ColumnRef(cr) if cr.column_id == col_id)
}

/// Checks if left is a ColumnRef matching col_id and right is a literal.
/// Returns the literal serialized as key bytes.
fn match_column_op_literal(
    left: &BoundExpr,
    right: &BoundExpr,
    col_id: zyron_catalog::ColumnId,
) -> Option<Vec<u8>> {
    if matches_index_column(left, col_id) {
        return extract_literal_bytes(right);
    }
    None
}

/// Checks if left is a literal and right is a ColumnRef matching col_id.
/// Returns the literal serialized as key bytes.
fn match_literal_op_column(
    left: &BoundExpr,
    right: &BoundExpr,
    col_id: zyron_catalog::ColumnId,
) -> Option<Vec<u8>> {
    if matches_index_column(right, col_id) {
        return extract_literal_bytes(left);
    }
    None
}

/// Matches col = literal or literal = col patterns.
fn match_column_literal(
    left: &BoundExpr,
    right: &BoundExpr,
    col_id: zyron_catalog::ColumnId,
) -> Option<Vec<u8>> {
    match_column_op_literal(left, right, col_id)
        .or_else(|| match_literal_op_column(left, right, col_id))
}

/// Extracts literal bytes from a BoundExpr::Literal.
fn extract_literal_bytes(expr: &BoundExpr) -> Option<Vec<u8>> {
    if let BoundExpr::Literal { value, .. } = expr {
        literal_to_key_bytes(value)
    } else {
        None
    }
}

/// Picks the later (larger) of two optional start keys.
fn pick_later_key(a: Option<Vec<u8>>, b: Option<Vec<u8>>) -> Option<Vec<u8>> {
    match (a, b) {
        (Some(a), Some(b)) => Some(if a >= b { a } else { b }),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

/// Picks the earlier (smaller) of two optional end keys.
fn pick_earlier_key(a: Option<Vec<u8>>, b: Option<Vec<u8>>) -> Option<Vec<u8>> {
    match (a, b) {
        (Some(a), Some(b)) => Some(if a <= b { a } else { b }),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

/// Increments a big-endian byte key by 1. Used for exclusive lower bounds (>).
fn increment_key(key: &[u8]) -> Vec<u8> {
    let mut result = key.to_vec();
    for byte in result.iter_mut().rev() {
        if *byte < 255 {
            *byte += 1;
            return result;
        }
        *byte = 0;
    }
    // Overflow: push an extra byte (handles max key edge case).
    result.push(0);
    result
}

/// Decrements a big-endian byte key by 1. Used for exclusive upper bounds (<).
fn decrement_key(key: &[u8]) -> Vec<u8> {
    let mut result = key.to_vec();
    for byte in result.iter_mut().rev() {
        if *byte > 0 {
            *byte -= 1;
            return result;
        }
        *byte = 255;
    }
    result
}

/// Index-guided scan operator. Uses a B+ tree index to look up matching
/// TupleIds, then fetches only those tuples from the heap file.
///
/// When no BTreeIndex instance is registered in the ExecutionContext,
/// falls back to a sequential scan with predicate filtering.
pub struct IndexScanOperator {
    /// B+ tree index scan state. None when falling back to seq scan.
    index_state: Option<IndexScanState>,
    /// Fallback sequential scan when no index instance is available.
    fallback: Option<SeqScanOperator>,
}

/// State for an active B+ tree index scan.
struct IndexScanState {
    ctx: Arc<ExecutionContext>,
    table_entry: Arc<TableEntry>,
    output_columns: Vec<LogicalColumn>,
    remaining_predicate: Option<BoundExpr>,
    track_tuple_ids: bool,
    /// Pre-collected TupleIds from the B+ tree range scan.
    tuple_ids: Vec<TupleId>,
    /// Current position in the tuple_ids vector.
    cursor: usize,
    finished: bool,
}

impl IndexScanOperator {
    /// Creates an index scan operator. When a BTreeIndex instance is
    /// registered in the ExecutionContext for the given index_id, performs
    /// an actual B+ tree range scan. Otherwise falls back to sequential
    /// scan with the predicate applied as a post-filter.
    pub async fn new(
        ctx: Arc<ExecutionContext>,
        table_id: zyron_catalog::TableId,
        index: Option<Arc<IndexEntry>>,
        btree: Option<Arc<BTreeIndex>>,
        columns: Vec<LogicalColumn>,
        predicate: BoundExpr,
        remaining_predicate: Option<BoundExpr>,
        track_tuple_ids: bool,
    ) -> Result<Self> {
        // Try B+ tree path when both index metadata and a live tree are available.
        if let (Some(index_entry), Some(btree_index)) = (&index, &btree) {
            let bounds = extract_scan_bounds(&predicate, index_entry);

            let table_entry = ctx.get_table_entry(table_id)?;
            let heap_file_id = table_entry.heap_file_id;

            // Collect matching TupleIds from the B+ tree.
            let mut tuple_ids = Vec::new();
            btree_index.range_scan_for_each(
                bounds.start_key.as_deref(),
                bounds.end_key.as_deref(),
                |_key, tid| {
                    // The B+ tree stores page numbers with file_id=0.
                    // Reconstruct the correct PageId using the heap file ID.
                    let corrected_tid =
                        TupleId::new(PageId::new(heap_file_id, tid.page_id.page_num), tid.slot_id);
                    tuple_ids.push(corrected_tid);
                    true
                },
            );

            return Ok(Self {
                index_state: Some(IndexScanState {
                    ctx,
                    table_entry,
                    output_columns: columns,
                    remaining_predicate,
                    track_tuple_ids,
                    tuple_ids,
                    cursor: 0,
                    finished: false,
                }),
                fallback: None,
            });
        }

        // Fallback: no B+ tree instance available. Use sequential scan
        // with the full predicate as a post-filter.
        let combined = match remaining_predicate {
            Some(rest) => BoundExpr::BinaryOp {
                left: Box::new(predicate),
                op: BinaryOperator::And,
                right: Box::new(rest),
                type_id: zyron_common::TypeId::Boolean,
            },
            None => predicate,
        };

        let inner = SeqScanOperator::new(
            ctx,
            table_id,
            columns,
            Some(combined),
            track_tuple_ids,
            None,
        )
        .await?;

        Ok(Self {
            index_state: None,
            fallback: Some(inner),
        })
    }
}

impl Operator for IndexScanOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        // Delegate to fallback sequential scan if no index state.
        if let Some(ref mut fallback) = self.fallback {
            return fallback.next();
        }

        Box::pin(async move {
            let state = self.index_state.as_mut().unwrap();

            if state.finished {
                return Ok(None);
            }
            state.ctx.check_cancelled()?;

            let batch_size = state.ctx.batch_size;
            let mut builders = create_builders(&state.output_columns, batch_size);
            let mut result_tuple_ids: Vec<TupleId> = if state.track_tuple_ids {
                Vec::with_capacity(batch_size)
            } else {
                Vec::new()
            };
            let mut row_count: usize = 0;

            // Fetch tuples from the heap using pre-collected TupleIds.
            while row_count < batch_size && state.cursor < state.tuple_ids.len() {
                let tid = state.tuple_ids[state.cursor];
                state.cursor += 1;

                let page_data: [u8; PAGE_SIZE] =
                    state.ctx.disk_manager.read_page(tid.page_id).await?;
                let page = HeapPage::from_bytes(page_data);

                let slot_id = zyron_storage::SlotId(tid.slot_id);
                let Some(tuple) = page.get_tuple(slot_id) else {
                    continue;
                };

                if tuple.is_deleted() {
                    continue;
                }
                if !tuple.header().is_visible_to(&state.ctx.snapshot) {
                    continue;
                }

                decode_tuple_into_builders(tuple.data(), &state.table_entry.columns, &mut builders);

                if state.track_tuple_ids {
                    result_tuple_ids.push(tid);
                }

                row_count += 1;
            }

            if row_count == 0 {
                state.finished = true;
                return Ok(None);
            }

            let batch = finalize_builders(builders);

            // Apply remaining predicate as a post-filter.
            if let Some(ref pred) = state.remaining_predicate {
                let mask_col = evaluate(pred, &batch, &state.output_columns, &[])?;
                let mask = column_to_mask(&mask_col);
                let filtered = batch.filter(&mask);

                if state.track_tuple_ids {
                    let filtered_ids: Vec<TupleId> = mask
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &keep)| {
                            if keep {
                                Some(result_tuple_ids[i])
                            } else {
                                None
                            }
                        })
                        .collect();
                    return Ok(Some(ExecutionBatch::with_tuple_ids(filtered, filtered_ids)));
                }

                return Ok(Some(ExecutionBatch::new(filtered)));
            }

            if state.track_tuple_ids {
                Ok(Some(ExecutionBatch::with_tuple_ids(
                    batch,
                    result_tuple_ids,
                )))
            } else {
                Ok(Some(ExecutionBatch::new(batch)))
            }
        })
    }
}

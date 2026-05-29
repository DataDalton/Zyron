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

use zyron_buffer::BufferPool;
use zyron_catalog::{IndexEntry, TableEntry};
use zyron_common::Result;
use zyron_common::TypeId;
use zyron_common::page::{PAGE_SIZE, PageId};
use zyron_parser::ast::{BinaryOperator, LiteralValue};
use zyron_planner::binder::BoundExpr;
use zyron_planner::logical::LogicalColumn;
use zyron_storage::{BTreeIndex, DiskManager, HeapPage, TupleId};

use crate::batch::{
    BATCH_SIZE, ColumnBuilder, DataBatch, build_column_to_builder_map, create_builders,
    decode_tuple_into_builders, finalize_builders,
};
use crate::column::ScalarValue;
use crate::compute::column_to_mask;
use crate::context::ExecutionContext;
use crate::expr::evaluate;
use crate::operator::{ExecutionBatch, Operator, OperatorResult};

/// Reads a heap page through the buffer pool: serves the buffer-pool copy
/// when present (and freshly inserted pages live there before the background
/// writer flushes them), otherwise loads from disk into the pool and returns
/// the loaded data. Any dirty page evicted during the load is written back.
async fn read_page_through_pool(
    pool: &BufferPool,
    disk: &DiskManager,
    page_id: PageId,
) -> Result<[u8; PAGE_SIZE]> {
    if let Some(frame) = pool.fetch_page(page_id) {
        let guard = frame.read_data();
        let data: [u8; PAGE_SIZE] = **guard;
        drop(guard);
        pool.unpin_page(page_id, false);
        return Ok(data);
    }
    let disk_data = disk.read_page(page_id).await?;
    let (frame, evicted) = pool.load_page(page_id, &disk_data)?;
    if let Some(evicted_page) = evicted {
        disk.write_page(evicted_page.page_id, &evicted_page.data)
            .await?;
    }
    let guard = frame.read_data();
    let data: [u8; PAGE_SIZE] = **guard;
    drop(guard);
    pool.unpin_page(page_id, false);
    Ok(data)
}

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
    /// Per-table-column index into `output_columns`. Built once at
    /// construction so the per-row decoder does an O(1) lookup instead of
    /// scanning the projection list.
    column_to_builder: Vec<Option<u16>>,
    predicate: Option<BoundExpr>,
    page_cursor: u64,
    /// Resume position within the current page when a previous next() call
    /// stopped mid-page after filling its output batch. Zero means start from
    /// the first slot of the page identified by page_cursor.
    slot_cursor: u16,
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
        // cached atomic load instead of disk_manager.num_pages which would
        // queue on the per-file tokio Mutex under concurrency
        let hf = ctx.get_heap_file(table_id).await?;
        let num_pages = hf.num_pages_cached() as u64;
        let output_ids: Vec<zyron_catalog::ColumnId> =
            columns.iter().map(|c| c.column_id).collect();
        let column_to_builder = build_column_to_builder_map(&table_entry.columns, &output_ids);

        Ok(Self {
            ctx,
            table_entry,
            output_columns: columns,
            column_to_builder,
            predicate,
            page_cursor: 0,
            slot_cursor: 0,
            num_pages,
            finished: false,
            track_tuple_ids,
            as_of_version,
        })
    }

    /// Enforces column-level security on a result batch. Delegates to the
    /// shared operator-level policy so heap and columnar scans behave
    /// identically.
    fn apply_column_security(&self, batch: DataBatch) -> DataBatch {
        crate::operator::apply_column_security(
            &self.ctx,
            self.table_entry.id.0,
            &self.output_columns,
            batch,
        )
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
            let count_only =
                self.output_columns.is_empty() && self.predicate.is_none() && !self.track_tuple_ids;
            let mut builders = create_builders(&self.output_columns, batch_size);
            let mut tuple_ids: Vec<TupleId> = if self.track_tuple_ids {
                Vec::with_capacity(batch_size)
            } else {
                Vec::new()
            };
            let mut row_count: usize = 0;

            while row_count < batch_size && self.page_cursor < self.num_pages {
                let page_id = PageId::new(self.table_entry.heap_file_id, self.page_cursor);

                let page_data: [u8; PAGE_SIZE] =
                    read_page_through_pool(&self.ctx.buffer_pool, &self.ctx.disk_manager, page_id)
                        .await?;

                // Empty page fast path, avoid HeapPage box allocation when
                // the page has zero slots (freshly allocated, never written)
                let header = HeapPage::heap_header_from_slice(&page_data);
                if header.slot_count == 0 {
                    self.page_cursor += 1;
                    self.slot_cursor = 0;
                    continue;
                }

                let page = HeapPage::from_bytes(page_data);
                let slot_count = header.slot_count;
                let mut slot_idx = self.slot_cursor;
                let mut filled_batch = false;

                while slot_idx < slot_count {
                    let slot_id = zyron_storage::SlotId(slot_idx);
                    slot_idx += 1;
                    let Some(tuple) = page.get_tuple_view(slot_id) else {
                        continue;
                    };
                    if tuple.is_deleted() {
                        continue;
                    }
                    // Version-based visibility for time travel queries,
                    // MVCC snapshot visibility for normal queries
                    //
                    // When as_of_version is set, the tuple's base header
                    // xmin/xmax are reinterpreted as version bounds via
                    // is_visible (version_id <= target, deleted_at > target)
                    // This works because versioned tables store version_id
                    // in xmin and deleted_at_version in xmax on the base
                    // TupleHeader, keeping the same visibility predicate shape
                    //
                    // Non-versioned tuples in a time travel query fall back
                    // to MVCC snapshot visibility as a safety measure
                    let hdr = tuple.header;
                    if let Some(target_version) = self.as_of_version {
                        if hdr.flags.has_version() {
                            if !hdr.is_visible(target_version as u32) {
                                continue;
                            }
                        } else if !hdr.is_visible_to(&self.ctx.snapshot) {
                            continue;
                        }
                    } else if !hdr.is_visible_to(&self.ctx.snapshot) {
                        continue;
                    }

                    if !count_only {
                        decode_tuple_into_builders(
                            tuple.data,
                            &self.table_entry.columns,
                            &self.column_to_builder,
                            &mut builders,
                        );

                        if self.track_tuple_ids {
                            tuple_ids.push(TupleId::new(page_id, slot_id.0));
                        }
                    }

                    row_count += 1;
                    if row_count >= batch_size {
                        filled_batch = true;
                        break;
                    }
                }

                if filled_batch && slot_idx < slot_count {
                    // Resume from slot_idx on this same page in the next batch
                    self.slot_cursor = slot_idx;
                } else {
                    self.page_cursor += 1;
                    self.slot_cursor = 0;
                }
            }

            if row_count == 0 {
                self.finished = true;
                return Ok(None);
            }

            // Count-only path emits the visible-row count with no column data.
            if count_only {
                return Ok(Some(ExecutionBatch::new(DataBatch::with_row_count(
                    row_count,
                ))));
            }

            let batch = finalize_builders(builders);

            // Apply predicate filter if present. The predicate runs on the
            // real (unmasked) values; column-level security is applied to the
            // surviving rows afterward so masking never changes filtering.
            if let Some(ref predicate) = self.predicate {
                let mask_col = evaluate(predicate, &batch, &self.output_columns, &self.ctx.params)?;
                let mask = column_to_mask(&mask_col);

                let filtered = batch.filter(&mask);
                let secured = self.apply_column_security(filtered);

                if self.track_tuple_ids {
                    let filtered_ids: Vec<TupleId> = mask
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &keep)| if keep { Some(tuple_ids[i]) } else { None })
                        .collect();
                    return Ok(Some(ExecutionBatch::with_tuple_ids(secured, filtered_ids)));
                }

                return Ok(Some(ExecutionBatch::new(secured)));
            }

            let secured = self.apply_column_security(batch);
            if self.track_tuple_ids {
                Ok(Some(ExecutionBatch::with_tuple_ids(secured, tuple_ids)))
            } else {
                Ok(Some(ExecutionBatch::new(secured)))
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
    /// Worker join handles, retained so a worker that panics (rather than
    /// returning Err through the channel) is detected as an error instead of
    /// being mistaken for clean end-of-stream, which would silently truncate
    /// the result set.
    workers: Vec<tokio::task::JoinHandle<()>>,
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
        let num_pages = ctx.get_heap_file(table_id).await?.num_pages_cached() as u64;

        let num_workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .min(num_pages as usize)
            .max(1);

        let pages_per_worker = (num_pages + num_workers as u64 - 1) / num_workers as u64;

        // Channel capacity: 2 batches per worker to keep workers busy
        // without unbounded buffering.
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<DataBatch>>(num_workers * 2);

        let mut workers = Vec::with_capacity(num_workers);
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

            workers.push(tokio::spawn(async move {
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
            }));
        }

        Ok(Self {
            receiver: rx,
            finished: false,
            workers,
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
    let mut scanner = PageRangeScanner::new(
        ctx,
        table_entry,
        output_columns,
        predicate,
        start_page,
        end_page,
    );
    while let Some(batch) = scanner.next_batch().await? {
        if tx.send(Ok(batch)).await.is_err() {
            break;
        }
    }
    Ok(())
}

/// Pull-based scanner over a contiguous page range. The single
/// decode/visibility/count-only path shared by the parallel scan and the
/// parallel aggregate, so both consume rows identically.
///
/// Count-only mirrors SeqScanOperator. When no columns are projected and no
/// predicate filters rows, COUNT(*) needs only the visible-row count, so the
/// batch carries num_rows with no column data. Without this the empty builders
/// finalize to a zero-column batch whose num_rows is 0, dropping every row.
pub(crate) struct PageRangeScanner<'a> {
    ctx: &'a ExecutionContext,
    table_entry: &'a TableEntry,
    output_columns: &'a [LogicalColumn],
    predicate: Option<&'a BoundExpr>,
    column_to_builder: Vec<Option<u16>>,
    count_only: bool,
    page_cursor: u64,
    end_page: u64,
    // Resume position within the current page when a batch fills mid-page.
    // Without this, slots after the break would be skipped because page_cursor
    // already advanced.
    slot_cursor: u16,
}

impl<'a> PageRangeScanner<'a> {
    pub(crate) fn new(
        ctx: &'a ExecutionContext,
        table_entry: &'a TableEntry,
        output_columns: &'a [LogicalColumn],
        predicate: Option<&'a BoundExpr>,
        start_page: u64,
        end_page: u64,
    ) -> Self {
        let output_ids: Vec<zyron_catalog::ColumnId> =
            output_columns.iter().map(|c| c.column_id).collect();
        let column_to_builder = build_column_to_builder_map(&table_entry.columns, &output_ids);
        let count_only = output_columns.is_empty() && predicate.is_none();
        Self {
            ctx,
            table_entry,
            output_columns,
            predicate,
            column_to_builder,
            count_only,
            page_cursor: start_page,
            end_page,
            slot_cursor: 0,
        }
    }

    /// Produces the next result batch, or None when the range is exhausted.
    pub(crate) async fn next_batch(&mut self) -> Result<Option<DataBatch>> {
        let batch_size = self.ctx.batch_size;

        while self.page_cursor < self.end_page {
            self.ctx.check_cancelled()?;

            let mut builders = create_builders(self.output_columns, batch_size);
            let mut row_count = 0usize;

            while row_count < batch_size && self.page_cursor < self.end_page {
                let page_id = PageId::new(self.table_entry.heap_file_id, self.page_cursor);

                let page_data: [u8; PAGE_SIZE] =
                    read_page_through_pool(&self.ctx.buffer_pool, &self.ctx.disk_manager, page_id)
                        .await?;

                let header = HeapPage::heap_header_from_slice(&page_data);
                if header.slot_count == 0 {
                    self.page_cursor += 1;
                    self.slot_cursor = 0;
                    continue;
                }

                let page = HeapPage::from_bytes(page_data);
                let slot_count = header.slot_count;
                let mut slot_idx = self.slot_cursor;
                let mut filled_batch = false;

                while slot_idx < slot_count {
                    let slot_id = zyron_storage::SlotId(slot_idx);
                    slot_idx += 1;
                    let Some(tuple) = page.get_tuple_view(slot_id) else {
                        continue;
                    };
                    if tuple.is_deleted() {
                        continue;
                    }
                    let hdr = tuple.header;
                    if !hdr.is_visible_to(&self.ctx.snapshot) {
                        continue;
                    }

                    if !self.count_only {
                        decode_tuple_into_builders(
                            tuple.data,
                            &self.table_entry.columns,
                            &self.column_to_builder,
                            &mut builders,
                        );
                    }

                    row_count += 1;
                    if row_count >= batch_size {
                        filled_batch = true;
                        break;
                    }
                }

                if filled_batch && slot_idx < slot_count {
                    self.slot_cursor = slot_idx;
                } else {
                    self.page_cursor += 1;
                    self.slot_cursor = 0;
                }
            }

            if row_count == 0 {
                break;
            }

            if self.count_only {
                return Ok(Some(DataBatch::with_row_count(row_count)));
            }

            let batch = finalize_builders(builders);
            if let Some(pred) = self.predicate {
                let mask_col = evaluate(pred, &batch, self.output_columns, &self.ctx.params)?;
                let mask = column_to_mask(&mask_col);
                let filtered = batch.filter(&mask);
                if filtered.num_rows == 0 {
                    continue;
                }
                return Ok(Some(filtered));
            }
            return Ok(Some(batch));
        }

        Ok(None)
    }
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
                    // The channel closed because every worker's sender dropped.
                    // Distinguish clean completion from a worker that panicked
                    // (which would otherwise look like an early end-of-stream
                    // and silently truncate results). Join the handles and
                    // surface a panic as an execution error.
                    for handle in self.workers.drain(..) {
                        if handle.await.is_err() {
                            return Err(zyron_common::ZyronError::ExecutionError(
                                "parallel scan worker panicked".to_string(),
                            ));
                        }
                    }
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
fn extract_scan_bounds(
    predicate: &BoundExpr,
    index: &IndexEntry,
    params: &[ScalarValue],
) -> ScanBounds {
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
            if let Some(bytes) = match_column_literal(left, right, index_col_id, params) {
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
            if let Some(bytes) = match_column_op_literal(left, right, index_col_id, params) {
                // Start just after this key. For integer keys, increment by 1.
                let start = increment_key(&bytes);
                return ScanBounds {
                    start_key: Some(start),
                    end_key: None,
                };
            }
            // literal > col means col < literal
            if let Some(bytes) = match_literal_op_column(left, right, index_col_id, params) {
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
            if let Some(bytes) = match_column_op_literal(left, right, index_col_id, params) {
                return ScanBounds {
                    start_key: Some(bytes),
                    end_key: None,
                };
            }
            if let Some(bytes) = match_literal_op_column(left, right, index_col_id, params) {
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
            if let Some(bytes) = match_column_op_literal(left, right, index_col_id, params) {
                let end = decrement_key(&bytes);
                return ScanBounds {
                    start_key: None,
                    end_key: Some(end),
                };
            }
            if let Some(bytes) = match_literal_op_column(left, right, index_col_id, params) {
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
            if let Some(bytes) = match_column_op_literal(left, right, index_col_id, params) {
                return ScanBounds {
                    start_key: None,
                    end_key: Some(bytes),
                };
            }
            if let Some(bytes) = match_literal_op_column(left, right, index_col_id, params) {
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
                let col_ty = column_type_id(expr);
                let start = extract_constant_bytes(low, params, col_ty);
                let end = extract_constant_bytes(high, params, col_ty);
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
            let left_bounds = extract_scan_bounds(left, index, params);
            let right_bounds = extract_scan_bounds(right, index, params);
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

/// Returns the TypeId carried by a ColumnRef expression when one is
/// present, otherwise None. Used so index bound encoding can coerce a
/// parameter scalar to the same byte layout the indexer used at INSERT
/// time even when the wire layer decoded the parameter as Utf8 (which
/// happens whenever the client sent Parse with zero param type hints).
fn column_type_id(expr: &BoundExpr) -> Option<TypeId> {
    if let BoundExpr::ColumnRef(cr) = expr {
        Some(cr.type_id)
    } else {
        None
    }
}

/// Checks if left is a ColumnRef matching col_id and right is a constant.
/// Returns the constant serialized as key bytes.
fn match_column_op_literal(
    left: &BoundExpr,
    right: &BoundExpr,
    col_id: zyron_catalog::ColumnId,
    params: &[ScalarValue],
) -> Option<Vec<u8>> {
    if matches_index_column(left, col_id) {
        return extract_constant_bytes(right, params, column_type_id(left));
    }
    None
}

/// Checks if left is a constant and right is a ColumnRef matching col_id.
/// Returns the constant serialized as key bytes.
fn match_literal_op_column(
    left: &BoundExpr,
    right: &BoundExpr,
    col_id: zyron_catalog::ColumnId,
    params: &[ScalarValue],
) -> Option<Vec<u8>> {
    if matches_index_column(right, col_id) {
        return extract_constant_bytes(left, params, column_type_id(right));
    }
    None
}

/// Matches col = constant or constant = col patterns.
fn match_column_literal(
    left: &BoundExpr,
    right: &BoundExpr,
    col_id: zyron_catalog::ColumnId,
    params: &[ScalarValue],
) -> Option<Vec<u8>> {
    match_column_op_literal(left, right, col_id, params)
        .or_else(|| match_literal_op_column(left, right, col_id, params))
}

/// Extracts constant bytes from a `BoundExpr::Literal` or a
/// `BoundExpr::Parameter` resolved against the executor's bind parameters.
/// `column_ty` is the TypeId of the indexed column being compared, used to
/// coerce a Utf8-decoded parameter back to the column's wire encoding.
fn extract_constant_bytes(
    expr: &BoundExpr,
    params: &[ScalarValue],
    column_ty: Option<TypeId>,
) -> Option<Vec<u8>> {
    match expr {
        BoundExpr::Literal { value, .. } => literal_to_key_bytes(value),
        // PG parameter indexes are 1-based ($1, $2, ...) while the params
        // slice is 0-based. The mismatch silently produced empty bounds and
        // an open range scan for every prepared point lookup.
        BoundExpr::Parameter { index, .. } if *index >= 1 => params
            .get(*index - 1)
            .and_then(|v| scalar_to_key_bytes(v, column_ty)),
        _ => None,
    }
}

/// Encodes a bound parameter into the same big-endian, order-preserving
/// byte layout that `encode_btree_key_into` uses when an indexed row is
/// indexed. `column_ty` is the indexed column's TypeId, used to coerce
/// from the parameter's wire-decoded scalar shape (often Utf8 when the
/// client did not send param type hints during Parse) to the layout the
/// indexer wrote.
fn scalar_to_key_bytes(value: &ScalarValue, column_ty: Option<TypeId>) -> Option<Vec<u8>> {
    if let (ScalarValue::Utf8(s), Some(ty)) = (value, column_ty) {
        if let Some(bytes) = coerce_text_to_key_bytes(s, ty) {
            return Some(bytes);
        }
    }
    match value {
        ScalarValue::Null => None,
        ScalarValue::Boolean(b) => Some(vec![*b as u8]),
        ScalarValue::Int8(v) => Some((*v as i64 as u64).to_be_bytes().to_vec()),
        ScalarValue::Int16(v) => Some((*v as i64 as u64).to_be_bytes().to_vec()),
        ScalarValue::Int32(v) => Some((*v as i64 as u64).to_be_bytes().to_vec()),
        ScalarValue::Int64(v) => Some((*v as u64).to_be_bytes().to_vec()),
        ScalarValue::Int128(v) => {
            let key = (*v as u128) ^ (1u128 << 127);
            Some(key.to_be_bytes().to_vec())
        }
        ScalarValue::UInt8(v) => Some((*v as u64).to_be_bytes().to_vec()),
        ScalarValue::UInt16(v) => Some((*v as u64).to_be_bytes().to_vec()),
        ScalarValue::UInt32(v) => Some((*v as u64).to_be_bytes().to_vec()),
        ScalarValue::UInt64(v) => Some(v.to_be_bytes().to_vec()),
        ScalarValue::Float32(v) => {
            let bits = (*v as f64).to_bits();
            let sortable = if bits >> 63 == 1 {
                !bits
            } else {
                bits ^ (1u64 << 63)
            };
            Some(sortable.to_be_bytes().to_vec())
        }
        ScalarValue::Float64(v) => {
            let bits = v.to_bits();
            let sortable = if bits >> 63 == 1 {
                !bits
            } else {
                bits ^ (1u64 << 63)
            };
            Some(sortable.to_be_bytes().to_vec())
        }
        ScalarValue::Utf8(s) => Some(s.as_bytes().to_vec()),
        ScalarValue::Binary(b) => Some(b.clone()),
        ScalarValue::FixedBinary16(b) => Some(b.to_vec()),
        ScalarValue::Interval(i) => Some(i.to_le_bytes().to_vec()),
    }
}

/// Coerces a text-encoded parameter to the byte layout the indexer wrote
/// for a column of the given TypeId. Returns None when the text cannot be
/// parsed to that type, which causes the caller to fall back to the raw
/// Utf8 byte path and the index scan to come up empty.
fn coerce_text_to_key_bytes(text: &str, ty: TypeId) -> Option<Vec<u8>> {
    match ty {
        TypeId::Int8 | TypeId::Int16 | TypeId::Int32 | TypeId::Int64 => text
            .trim()
            .parse::<i64>()
            .ok()
            .map(|v| (v as u64).to_be_bytes().to_vec()),
        TypeId::UInt8 | TypeId::UInt16 | TypeId::UInt32 | TypeId::UInt64 => text
            .trim()
            .parse::<u64>()
            .ok()
            .map(|v| v.to_be_bytes().to_vec()),
        TypeId::Float32 | TypeId::Float64 => text.trim().parse::<f64>().ok().map(|v| {
            let bits = v.to_bits();
            let sortable = if bits >> 63 == 1 {
                !bits
            } else {
                bits ^ (1u64 << 63)
            };
            sortable.to_be_bytes().to_vec()
        }),
        TypeId::Boolean => match text.trim() {
            "t" | "T" | "true" | "TRUE" | "1" | "yes" | "on" => Some(vec![1]),
            "f" | "F" | "false" | "FALSE" | "0" | "no" | "off" => Some(vec![0]),
            _ => None,
        },
        _ => None,
    }
}

/// Extracts literal bytes from a `BoundExpr::Literal`. Kept for callers
/// that already have a literal in hand and do not need parameter lookup.
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
    /// Per-table-column index into `output_columns`, precomputed once.
    column_to_builder: Vec<Option<u16>>,
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
            let bounds = extract_scan_bounds(&predicate, index_entry, &ctx.params);

            // If bounds extraction could not narrow the scan to a single
            // key range, the predicate still has to filter the result so
            // we do not return rows that do not match. Push it into the
            // remaining_predicate so the post-filter catches them. This
            // keeps correctness intact for any predicate shape that the
            // bounds extractor does not yet decompose (function calls,
            // OR, unsupported scalar shapes) without silently degrading
            // an indexed lookup into an open scan.
            let effective_remaining = if bounds.start_key.is_none() && bounds.end_key.is_none() {
                match remaining_predicate {
                    Some(rest) => Some(BoundExpr::BinaryOp {
                        left: Box::new(predicate.clone()),
                        op: BinaryOperator::And,
                        right: Box::new(rest),
                        type_id: zyron_common::TypeId::Boolean,
                    }),
                    None => Some(predicate.clone()),
                }
            } else {
                remaining_predicate
            };

            let table_entry = ctx.get_table_entry(table_id)?;
            let heap_file_id = table_entry.heap_file_id;

            let mut tuple_ids = Vec::new();
            btree_index.range_scan_for_each(
                bounds.start_key.as_deref(),
                bounds.end_key.as_deref(),
                |_key, tid| {
                    let corrected_tid =
                        TupleId::new(PageId::new(heap_file_id, tid.page_id.page_num), tid.slot_id);
                    tuple_ids.push(corrected_tid);
                    true
                },
            );

            let output_ids: Vec<zyron_catalog::ColumnId> =
                columns.iter().map(|c| c.column_id).collect();
            let column_to_builder = build_column_to_builder_map(&table_entry.columns, &output_ids);

            return Ok(Self {
                index_state: Some(IndexScanState {
                    ctx,
                    table_entry,
                    output_columns: columns,
                    column_to_builder,
                    remaining_predicate: effective_remaining,
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
            // Read directly from the buffer pool frame's data via the read
            // lock, avoids the 16KB stack copy + Box allocation that
            // read_page_through_pool would do per call. Concurrent atomic
            // inserts coordinate via the slot's AtomicU32 commit so our
            // read sees either uncommitted (length=0, skip) or committed
            // bytes consistently
            while row_count < batch_size && state.cursor < state.tuple_ids.len() {
                let tid = state.tuple_ids[state.cursor];
                state.cursor += 1;

                let frame_present = state.ctx.buffer_pool.fetch_page(tid.page_id).is_some();
                if !frame_present {
                    let disk_data = state.ctx.disk_manager.read_page(tid.page_id).await?;
                    let (_, evicted) = state.ctx.buffer_pool.load_page(tid.page_id, &disk_data)?;
                    if let Some(ev) = evicted {
                        state
                            .ctx
                            .disk_manager
                            .write_page(ev.page_id, &ev.data)
                            .await?;
                    }
                    // load_page pinned, frame_present path's fetch_page also
                    // pinned, in both cases we have one extra pin to balance
                }

                let frame = state
                    .ctx
                    .buffer_pool
                    .fetch_page(tid.page_id)
                    .expect("just pinned this page");
                state.ctx.buffer_pool.unpin_page(tid.page_id, false);

                let visible = {
                    let guard = frame.read_data();
                    let slot_id = zyron_storage::SlotId(tid.slot_id);
                    match HeapPage::get_tuple_view_from_slice(&**guard, slot_id) {
                        None => false,
                        Some(view) => {
                            if view.is_deleted() || !view.header.is_visible_to(&state.ctx.snapshot)
                            {
                                false
                            } else {
                                decode_tuple_into_builders(
                                    view.data,
                                    &state.table_entry.columns,
                                    &state.column_to_builder,
                                    &mut builders,
                                );
                                true
                            }
                        }
                    }
                };
                state.ctx.buffer_pool.unpin_page(tid.page_id, false);

                if visible {
                    if state.track_tuple_ids {
                        result_tuple_ids.push(tid);
                    }
                    row_count += 1;
                }
            }

            if row_count == 0 {
                state.finished = true;
                return Ok(None);
            }

            let batch = finalize_builders(builders);

            // Apply remaining predicate as a post-filter.
            if let Some(ref pred) = state.remaining_predicate {
                let mask_col = evaluate(pred, &batch, &state.output_columns, &state.ctx.params)?;
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

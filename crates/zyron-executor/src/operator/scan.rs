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
    DataBatch, build_column_to_builder_map, create_builders, decode_tuple_into_builders,
    finalize_builders,
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
pub(crate) async fn read_page_through_pool(
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

/// Resolves a branch's append overlay file id and page count for a table, or
/// (None, 0) on the main line. The scan reads this range after the main range.
fn branch_append_range(
    ctx: &ExecutionContext,
    branch_id: Option<u64>,
    heap_file_id: u32,
) -> (Option<u32>, u64) {
    match (branch_id, &ctx.branch_catalog) {
        (Some(bid), Some(cat)) => {
            let files = cat.branch_files_for(bid, heap_file_id);
            (
                Some(files.append_file_id),
                cat.append_page_count(bid, heap_file_id),
            )
        }
        _ => (None, 0),
    }
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
    /// Effective branch for this scan. When set, each page id is resolved
    /// through the branch override chain before reading.
    branch_id: Option<u64>,
    /// Append overlay file id for the active branch, holding rows the branch
    /// inserted. None on the main line. Scanned as a second range after the
    /// main range.
    branch_append_file_id: Option<u32>,
    /// Number of pages in the branch append file to scan.
    num_append_pages: u64,
    /// False while scanning the main range, true once scanning the append range.
    in_append_phase: bool,
    /// When true the main range is skipped entirely and only the branch append
    /// range is scanned. Used by the branch-aware index scan to read the insert
    /// delta after draining the main index.
    append_only: bool,
    /// This table's IO counters, resolved once at construction. Updated once
    /// per batch with the rows produced and the page bytes read to produce them.
    io_stats: Option<Arc<zyron_common::TableIOStats>>,
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
        let branch_id = ctx.active_branch_id;
        let table_entry = ctx.get_table_entry(table_id)?;
        // cached atomic load instead of disk_manager.num_pages which would
        // queue on the per-file tokio Mutex under concurrency
        let hf = ctx.get_heap_file(table_id).await?;
        let num_pages = hf.num_pages_cached() as u64;
        let output_ids: Vec<zyron_catalog::ColumnId> =
            columns.iter().map(|c| c.column_id).collect();
        let column_to_builder = build_column_to_builder_map(&table_entry.columns, &output_ids);
        let (branch_append_file_id, num_append_pages) =
            branch_append_range(&ctx, branch_id, table_entry.heap_file_id);
        let io_stats = ctx.table_io_stats_for(table_id.0);
        if let Some(stats) = &io_stats {
            stats.record_seq_scan();
        }

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
            branch_id,
            branch_append_file_id,
            num_append_pages,
            in_append_phase: false,
            append_only: false,
            io_stats,
        })
    }

    /// Overrides the scan's branch (used for a per-query `IN BRANCH name` that
    /// differs from the session's active branch). Recomputes the append range
    /// for the new branch.
    pub fn with_branch(mut self, branch_id: Option<u64>) -> Self {
        self.branch_id = branch_id;
        let (af, np) = branch_append_range(&self.ctx, branch_id, self.table_entry.heap_file_id);
        self.branch_append_file_id = af;
        self.num_append_pages = np;
        self
    }

    /// Restricts the scan to the branch append range, skipping the main range.
    /// The branch-aware index scan uses this to read the insert delta the main
    /// index does not cover.
    pub fn append_only(mut self) -> Self {
        self.append_only = true;
        self.in_append_phase = true;
        self
    }

    /// Points the scan at an explicit append file range, skipping the main
    /// range, independent of any session branch. MERGE uses this to read a
    /// branch's inserted rows so it can replay them onto the main line.
    pub fn scan_append_file(mut self, append_file_id: u32, num_pages: u64) -> Self {
        self.branch_append_file_id = Some(append_file_id);
        self.num_append_pages = num_pages;
        self.append_only = true;
        self.in_append_phase = true;
        self
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
            // Pages fetched for this batch, accumulated locally and folded into
            // the table counters once when the batch is done. A page revisited
            // because a previous batch filled mid-page counts again, which is
            // what happened: it was fetched again.
            let mut pages_read: u64 = 0;

            while row_count < batch_size {
                // The main range resolves each page through the branch override
                // chain; once it is exhausted the scan continues into the
                // branch append range, read directly with snapshot visibility.
                let page_id = if !self.in_append_phase {
                    if self.page_cursor >= self.num_pages {
                        if self.branch_append_file_id.is_some() && self.num_append_pages > 0 {
                            self.in_append_phase = true;
                            self.page_cursor = 0;
                            self.slot_cursor = 0;
                            continue;
                        }
                        break;
                    }
                    self.ctx.resolve_branch_page(
                        self.branch_id,
                        PageId::new(self.table_entry.heap_file_id, self.page_cursor),
                    )
                } else {
                    if self.page_cursor >= self.num_append_pages {
                        break;
                    }
                    PageId::new(self.branch_append_file_id.unwrap(), self.page_cursor)
                };

                let page_data: [u8; PAGE_SIZE] =
                    read_page_through_pool(&self.ctx.buffer_pool, &self.ctx.disk_manager, page_id)
                        .await?;
                pages_read += 1;

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
                    // Time-travel visibility dates each tuple by its
                    // transactions' commit LSNs: visible at version N when the
                    // inserter committed at an LSN <= N and the deleter (if any)
                    // committed at an LSN > N. This reconstructs the committed
                    // state as of N from the MVCC versions the heap already
                    // holds. Normal queries use live-snapshot MVCC visibility.
                    let hdr = tuple.header;
                    if let Some(target_version) = self.as_of_version {
                        if !self.ctx.snapshot.status_map().is_visible_at_version(
                            hdr.xmin as u64,
                            hdr.xmax as u64,
                            target_version,
                        ) {
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

            if let Some(stats) = &self.io_stats {
                stats.record_seq_batch(row_count as u64, pages_read * PAGE_SIZE as u64);
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
        // One scan, however many workers divide it. Each worker's scanner folds
        // in its own row and byte totals.
        if let Some(stats) = ctx.table_io_stats_for(table_id.0) {
            stats.record_seq_scan();
        }

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
    /// This table's IO counters. Each worker holds its own Arc to the same
    /// entry and folds its batch totals in, so the table's counters are the sum
    /// across workers. Scan initiation is recorded by the owning operator, once
    /// for the whole parallel scan rather than once per worker.
    io_stats: Option<Arc<zyron_common::TableIOStats>>,
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
        let io_stats = ctx.table_io_stats_for(table_entry.id.0);
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
            io_stats,
        }
    }

    /// Produces the next result batch, or None when the range is exhausted.
    pub(crate) async fn next_batch(&mut self) -> Result<Option<DataBatch>> {
        let batch_size = self.ctx.batch_size;

        while self.page_cursor < self.end_page {
            self.ctx.check_cancelled()?;

            let mut builders = create_builders(self.output_columns, batch_size);
            let mut row_count = 0usize;
            // Pages fetched for this batch, folded into the table counters once
            // when the batch is done rather than once per page.
            let mut pages_read: u64 = 0;

            while row_count < batch_size && self.page_cursor < self.end_page {
                let page_id = self.ctx.resolve_branch_page(
                    self.ctx.active_branch_id,
                    PageId::new(self.table_entry.heap_file_id, self.page_cursor),
                );

                let page_data: [u8; PAGE_SIZE] =
                    read_page_through_pool(&self.ctx.buffer_pool, &self.ctx.disk_manager, page_id)
                        .await?;
                pages_read += 1;

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

            if let Some(stats) = &self.io_stats {
                stats.record_seq_batch(row_count as u64, pages_read * PAGE_SIZE as u64);
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
        // Matches the sixteen byte key the indexer writes for a 128 bit
        // value, sign bit flipped so negatives order below positives
        LiteralValue::Int128(v) => Some(((*v as u128) ^ (1u128 << 127)).to_be_bytes().to_vec()),
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
    /// Branch insert delta scan, drained after the main index path. Present only
    /// when a branch is active: the main B+ tree does not index branch-inserted
    /// rows, so they are read from the append range with the predicate applied.
    append_delta: Option<SeqScanOperator>,
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
    /// Pre-collected row locators from the B+ tree range scan, heap entries
    /// re-stamped with the table's heap file id.
    locators: Vec<zyron_common::RowLocator>,
    /// Pre-fetched values for columnar-resident entries, batched per segment.
    /// None when every entry is heap resident.
    columnar: Option<crate::operator::doc_fetch::DocRowFetcher>,
    /// Current position in the locators vector.
    cursor: usize,
    /// Active branch for this scan. Main index tids are resolved through this
    /// branch's override chain so rows the branch deleted or modified (their
    /// cow page slot is tombstoned) drop out on fetch.
    branch_id: Option<u64>,
    finished: bool,
    /// Table and index IO counters, updated per batch with the rows fetched
    /// and the page bytes read to fetch them. The scan count and the entries
    /// the range scan examined are recorded when it is built, because the
    /// range scan runs to completion there.
    io_stats: crate::operator::IndexScanStats,
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
        descending: bool,
    ) -> Result<Self> {
        // Try B+ tree path when both index metadata and a live tree are available.
        if let (Some(index_entry), Some(btree_index)) = (&index, &btree) {
            let bounds = extract_scan_bounds(&predicate, index_entry, &ctx.params);

            // Always re-apply the index predicate as a post-filter, never trust
            // the index entry alone. Under MVCC, deleted rows keep their index
            // entries (filtered here by the per-row visibility check) and a
            // vacuumed-then-reused heap slot can be reached through a stale
            // entry whose key no longer matches the row now in that slot; the
            // post-filter rechecks the actual column value so such a row is
            // dropped. This also covers predicate shapes the bounds extractor
            // cannot decompose (functions, OR, unsupported scalars).
            let effective_remaining = match remaining_predicate {
                Some(rest) => Some(BoundExpr::BinaryOp {
                    left: Box::new(predicate.clone()),
                    op: BinaryOperator::And,
                    right: Box::new(rest),
                    type_id: zyron_common::TypeId::Boolean,
                }),
                None => Some(predicate.clone()),
            };

            let table_entry = ctx.get_table_entry(table_id)?;
            let heap_file_id = table_entry.heap_file_id;

            // A stored key is the leading indexed value, then any further key
            // components, then a locator suffix so duplicate values coexist.
            // Bounds are built on the leading value alone, so the lower bound
            // is the value itself (every entry for it sorts at or above) and
            // the upper bound is the value's successor, which sorts above
            // every entry for it whatever follows in the key.
            let start_key = bounds.start_key.clone();
            let end_key = bounds
                .end_key
                .as_ref()
                .and_then(|k| crate::operator::modify::index_key_upper_bound(k));

            let mut locators: Vec<zyron_common::RowLocator> = Vec::new();
            let mut has_columnar = false;
            btree_index.range_scan_for_each(
                start_key.as_deref(),
                end_key.as_deref(),
                |_key, loc| {
                    match loc {
                        // heap entries store no file id, re-stamp the table's
                        zyron_common::RowLocator::Heap { page, slot } => {
                            locators.push(zyron_common::RowLocator::Heap {
                                page: PageId::new(heap_file_id, page.page_num),
                                slot,
                            });
                        }
                        other => {
                            has_columnar = true;
                            locators.push(other);
                        }
                    }
                    true
                },
            );

            // Columnar-resident entries are pre-fetched in one batched pass
            // through the columnar scan machinery, visibility included
            let columnar = if has_columnar {
                Some(
                    crate::operator::doc_fetch::DocRowFetcher::prepare_columnar_only(
                        &ctx, table_id, &columns, &locators, None,
                    )
                    .await?,
                )
            } else {
                None
            };

            let output_ids: Vec<zyron_catalog::ColumnId> =
                columns.iter().map(|c| c.column_id).collect();
            let column_to_builder = build_column_to_builder_map(&table_entry.columns, &output_ids);

            // With a branch active, the main index does not cover rows the
            // branch inserted, so scan the append delta with the full predicate
            // after the main index path. Index scans are only branch-accelerated
            // for reads; DML (track_tuple_ids) stays on the sequential path.
            let branch_id = ctx.active_branch_id;
            let append_delta = if branch_id.is_some() && !track_tuple_ids {
                let full_predicate = match &effective_remaining {
                    Some(rest) => BoundExpr::BinaryOp {
                        left: Box::new(predicate.clone()),
                        op: BinaryOperator::And,
                        right: Box::new(rest.clone()),
                        type_id: zyron_common::TypeId::Boolean,
                    },
                    None => predicate.clone(),
                };
                let delta = SeqScanOperator::new(
                    ctx.clone(),
                    table_id,
                    columns.clone(),
                    Some(full_predicate),
                    false,
                    None,
                )
                .await?
                .append_only();
                Some(delta)
            } else {
                None
            };

            // The B+tree yields entries in ascending key order. A descending
            // scan reads the same entries the other way, which is what lets
            // an ORDER BY ... DESC be answered without a sort. The list is
            // already materialized, so this is a reversal rather than a
            // second traversal
            if descending {
                locators.reverse();
            }

            // The range scan already ran to completion above, so the number
            // of entries it examined is known here and recorded once
            let io_stats = crate::operator::IndexScanStats::open(
                &ctx,
                table_id.0,
                index_entry.id.0,
                locators.len(),
            );

            return Ok(Self {
                index_state: Some(IndexScanState {
                    ctx,
                    table_entry,
                    output_columns: columns,
                    column_to_builder,
                    remaining_predicate: effective_remaining,
                    track_tuple_ids,
                    locators,
                    columnar,
                    cursor: 0,
                    branch_id,
                    finished: false,
                    io_stats,
                }),
                fallback: None,
                append_delta,
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
            append_delta: None,
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
            // Drain the main index path first; when it is exhausted, drain the
            // branch insert delta (present only for branch-active reads).
            if let Some(state) = self.index_state.as_mut() {
                if !state.finished {
                    if let Some(batch) = state.next_batch().await? {
                        return Ok(Some(batch));
                    }
                    state.finished = true;
                }
            }
            if let Some(delta) = self.append_delta.as_mut() {
                return delta.next().await;
            }
            Ok(None)
        })
    }
}

impl IndexScanState {
    /// Produces the next batch from the pre-collected index tuple ids, or None
    /// when they are exhausted. Each tuple id is resolved through the active
    /// branch's override chain before fetch so branch deletes and modifications
    /// (their cow page slot is tombstoned) drop out.
    async fn next_batch(&mut self) -> Result<Option<ExecutionBatch>> {
        self.ctx.check_cancelled()?;

        let batch_size = self.ctx.batch_size;
        let mut builders = create_builders(&self.output_columns, batch_size);
        let mut result_locators: Vec<zyron_common::RowLocator> = if self.track_tuple_ids {
            Vec::with_capacity(batch_size)
        } else {
            Vec::new()
        };
        let mut row_count: usize = 0;

        // Fetch rows using the pre-collected locators. Heap entries read
        // directly from the buffer pool frame's data via the read lock,
        // avoiding the 16KB stack copy + Box allocation that
        // read_page_through_pool would do per call. Concurrent atomic
        // inserts coordinate via the slot's AtomicU32 commit so our read
        // sees either uncommitted (length=0, skip) or committed bytes
        // consistently. Columnar entries were pre-fetched in one batched
        // pass with visibility applied
        // DML consumers route a batch to the heap or the columnar mutation
        // path as a whole, so tracked batches stay homogeneous per storage
        // kind, a kind change closes the batch and the next call continues
        let mut batch_kind: Option<u8> = None;
        // Heap pages fetched to resolve this batch's locators, folded into the
        // table counters once when the batch is done.
        let mut pages_read: u64 = 0;
        while row_count < batch_size && self.cursor < self.locators.len() {
            let loc = self.locators[self.cursor];
            if self.track_tuple_ids {
                let kind = match loc {
                    zyron_common::RowLocator::Heap { .. } => 0u8,
                    zyron_common::RowLocator::Columnar { .. } => 1,
                    zyron_common::RowLocator::Lake { .. } => 2,
                };
                match batch_kind {
                    None => batch_kind = Some(kind),
                    Some(k) if k != kind => break,
                    _ => {}
                }
            }
            self.cursor += 1;

            let visible = match loc {
                zyron_common::RowLocator::Heap { page, slot } => {
                    // Resolve to the branch-local page when the branch copied
                    // it; the slot id is preserved by the page copy.
                    let phys_page = self.ctx.resolve_branch_page(self.branch_id, page);
                    pages_read += 1;

                    let frame_present = self.ctx.buffer_pool.fetch_page(phys_page).is_some();
                    if !frame_present {
                        let disk_data = self.ctx.disk_manager.read_page(phys_page).await?;
                        let (_, evicted) = self.ctx.buffer_pool.load_page(phys_page, &disk_data)?;
                        if let Some(ev) = evicted {
                            self.ctx
                                .disk_manager
                                .write_page(ev.page_id, &ev.data)
                                .await?;
                        }
                        // load_page pinned, frame_present path's fetch_page also
                        // pinned, in both cases we have one extra pin to balance
                    }

                    let frame = self
                        .ctx
                        .buffer_pool
                        .fetch_page(phys_page)
                        .expect("just pinned this page");
                    self.ctx.buffer_pool.unpin_page(phys_page, false);

                    let visible = {
                        let guard = frame.read_data();
                        let slot_id = zyron_storage::SlotId(slot);
                        match HeapPage::get_tuple_view_from_slice(&**guard, slot_id) {
                            None => false,
                            Some(view) => {
                                if view.is_deleted()
                                    || !view.header.is_visible_to(&self.ctx.snapshot)
                                {
                                    false
                                } else {
                                    decode_tuple_into_builders(
                                        view.data,
                                        &self.table_entry.columns,
                                        &self.column_to_builder,
                                        &mut builders,
                                    );
                                    true
                                }
                            }
                        }
                    };
                    self.ctx.buffer_pool.unpin_page(phys_page, false);
                    visible
                }
                zyron_common::RowLocator::Columnar { file_id, sys_rowid } => {
                    match self
                        .columnar
                        .as_ref()
                        .and_then(|f| f.columnar_row(file_id, sys_rowid))
                    {
                        Some(vals) => {
                            for (b, v) in builders.iter_mut().zip(vals.iter()) {
                                b.push(v);
                            }
                            true
                        }
                        // superseded, invisible to this snapshot, or reclaimed
                        None => false,
                    }
                }
                zyron_common::RowLocator::Lake { .. } => false,
            };

            if visible {
                if self.track_tuple_ids {
                    result_locators.push(loc);
                }
                row_count += 1;
            }
        }

        self.io_stats
            .record_batch(row_count as u64, pages_read * PAGE_SIZE as u64);

        if row_count == 0 {
            return Ok(None);
        }

        let batch = finalize_builders(builders);

        // Apply remaining predicate as a post-filter.
        if let Some(ref pred) = self.remaining_predicate {
            let mask_col = evaluate(pred, &batch, &self.output_columns, &self.ctx.params)?;
            let mask = column_to_mask(&mask_col);
            let filtered = batch.filter(&mask);

            if self.track_tuple_ids {
                let filtered_ids: Vec<zyron_common::RowLocator> = mask
                    .iter()
                    .enumerate()
                    .filter_map(
                        |(i, &keep)| {
                            if keep { Some(result_locators[i]) } else { None }
                        },
                    )
                    .collect();
                return Ok(Some(ExecutionBatch::with_locators(filtered, filtered_ids)));
            }

            return Ok(Some(ExecutionBatch::new(filtered)));
        }

        if self.track_tuple_ids {
            Ok(Some(ExecutionBatch::with_locators(batch, result_locators)))
        } else {
            Ok(Some(ExecutionBatch::new(batch)))
        }
    }
}

//! Sequential, parallel, and index scan operators.
//!
//! SeqScanOperator reads heap pages one at a time, decodes visible tuples
//! into columnar batches, and optionally applies a predicate filter.
//! ParallelSeqScanOperator splits the page range across multiple tokio tasks
//! for multi-core throughput on large tables.
//! IndexScanOperator provides the same interface using index-guided access
//! (currently falls back to sequential scan with predicate filtering).

use std::sync::Arc;

use zyron_catalog::TableEntry;
use zyron_common::Result;
use zyron_common::page::{PAGE_SIZE, PageId};
use zyron_planner::binder::BoundExpr;
use zyron_planner::logical::LogicalColumn;
use zyron_storage::{HeapPage, TupleId};

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
                let mask_col = evaluate(predicate, &batch, &self.output_columns)?;
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
            let mask_col = evaluate(pred, &batch, output_columns)?;
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

/// Index-guided scan operator. Accepts a predicate that would normally be
/// resolved through a B+ tree index lookup to produce a set of TupleIds.
///
/// TODO: Wire up actual B+ tree range scan to collect matching TupleIds,
/// then fetch tuples by TupleId from the heap. Currently falls back to a
/// full sequential scan with predicate filtering applied after decode.
pub struct IndexScanOperator {
    inner: SeqScanOperator,
}

impl IndexScanOperator {
    /// Creates an index scan operator. The predicate is applied as a
    /// post-filter on a sequential scan until B+ tree lookup is integrated.
    pub async fn new(
        ctx: Arc<ExecutionContext>,
        table_id: zyron_catalog::TableId,
        columns: Vec<LogicalColumn>,
        predicate: BoundExpr,
        remaining_predicate: Option<BoundExpr>,
        track_tuple_ids: bool,
    ) -> Result<Self> {
        // Combine the index predicate and any remaining predicate into one.
        let combined = match remaining_predicate {
            Some(rest) => BoundExpr::BinaryOp {
                left: Box::new(predicate),
                op: zyron_parser::ast::BinaryOperator::And,
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

        Ok(Self { inner })
    }
}

impl Operator for IndexScanOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        self.inner.next()
    }
}

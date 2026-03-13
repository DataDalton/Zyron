//! Sequential and index scan operators.
//!
//! SeqScanOperator reads heap pages one at a time, decodes visible tuples
//! into columnar batches, and optionally applies a predicate filter.
//! IndexScanOperator provides the same interface using index-guided access
//! (currently falls back to sequential scan with predicate filtering).

use std::sync::Arc;

use zyron_catalog::TableEntry;
use zyron_common::Result;
use zyron_common::page::{PAGE_SIZE, PageId};
use zyron_planner::binder::BoundExpr;
use zyron_planner::logical::LogicalColumn;
use zyron_storage::{HeapPage, TupleId};

use crate::batch::{create_builders, decode_tuple_into_builders, finalize_builders};
use crate::compute::column_to_mask;
use crate::context::ExecutionContext;
use crate::expr::evaluate;
use crate::operator::{ExecutionBatch, Operator, OperatorResult};

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
}

impl SeqScanOperator {
    /// Creates a new sequential scan operator for the given table.
    pub async fn new(
        ctx: Arc<ExecutionContext>,
        table_id: zyron_catalog::TableId,
        columns: Vec<LogicalColumn>,
        predicate: Option<BoundExpr>,
        track_tuple_ids: bool,
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
        })
    }
}

impl Operator for SeqScanOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            if self.finished {
                return Ok(None);
            }

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
                    if !tuple.header().is_visible_to(&self.ctx.snapshot) {
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

        let inner =
            SeqScanOperator::new(ctx, table_id, columns, Some(combined), track_tuple_ids).await?;

        Ok(Self { inner })
    }
}

impl Operator for IndexScanOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        self.inner.next()
    }
}

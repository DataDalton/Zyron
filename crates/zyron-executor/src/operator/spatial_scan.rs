//! Spatial index scan operator.
//!
//! Executes one of three query patterns against an R-tree:
//!
//! - **KNN**: returns the k entries with the smallest distance from a query
//!   point. Backed by `RTree::knn`. Used by `ORDER BY ST_Distance(col, q)
//!   LIMIT k`.
//! - **DWithin**: returns entries within a radius of a query point. Backed
//!   by `RTree::dwithin`. Used by `WHERE ST_DWithin(col, q, radius)`.
//! - **Range**: returns entries whose MBR intersects a query rectangle.
//!   Backed by `RTree::range`. Used by `WHERE ST_Intersects(col, env)`.
//!
//! After the spatial filter the operator decodes the rowid back to a
//! TupleId and fetches the heap tuple, applying any remaining predicate
//! and visibility checks downstream. SELECT privilege on the underlying
//! table is enforced at the planner stage; this operator does no extra
//! gating beyond the standard tuple-visibility logic in heap fetch.

use std::sync::Arc;

use zyron_catalog::IndexId;
use zyron_common::Result;
use zyron_common::RowLocator;
use zyron_common::ZyronError;
use zyron_planner::logical::LogicalColumn;
use zyron_planner::physical::SpatialScanKind;
use zyron_storage::HeapPage;
use zyron_types::spatial_index::Mbr;

use crate::batch::{
    build_column_to_builder_map, create_builders, decode_tuple_into_builders, finalize_builders,
};
use crate::context::ExecutionContext;
use crate::operator::doc_fetch::DocRowFetcher;
use crate::operator::{ExecutionBatch, Operator, OperatorResult};

#[allow(unused_imports)]
use crate::column::{Column, ColumnData, NullBitmap};

/// Operator that runs a spatial query against an in-memory R-tree, then
/// fetches matching heap tuples in result order.
pub struct SpatialScanOperator {
    ctx: Arc<ExecutionContext>,
    table_id: zyron_catalog::TableId,
    output_columns: Vec<LogicalColumn>,
    /// Pre-computed (rowid, optional_distance_squared) results.
    results: Vec<(u64, Option<f64>)>,
    /// Resolved locators plus pre-fetched columnar hits, result-aligned.
    fetcher: DocRowFetcher,
    /// Current position in the results vector.
    cursor: usize,
    finished: bool,
    /// When set, heap hits are dated by commit LSN at this version instead of
    /// being resolved against the live snapshot
    heap_version: Option<u64>,
    /// Table and index IO counters. The tree query ran to completion before
    /// any row was fetched, so the entries it examined were recorded then
    io_stats: crate::operator::IndexScanStats,
}

impl SpatialScanOperator {
    /// Creates a new spatial scan operator. Looks up the live R-tree from
    /// the context's spatial manager and runs the appropriate query method
    /// based on `kind`. Results are pre-collected so iteration is sequential
    /// thereafter.
    pub async fn new(
        ctx: Arc<ExecutionContext>,
        table_id: zyron_catalog::TableId,
        index_id: IndexId,
        columns: Vec<LogicalColumn>,
        kind: SpatialScanKind,
        as_of: Option<zyron_planner::logical::AsOfTarget>,
    ) -> Result<Self> {
        let spatial_mgr = ctx.spatial_manager.as_ref().ok_or_else(|| {
            ZyronError::ExecutionError(
                "spatial manager not configured on execution context".to_string(),
            )
        })?;

        let tree = spatial_mgr.get(index_id.0).ok_or_else(|| {
            ZyronError::ExecutionError(format!(
                "spatial index id {} not found in manager",
                index_id.0
            ))
        })?;

        let results: Vec<(u64, Option<f64>)> = match kind {
            SpatialScanKind::Knn { query_point, k } => tree
                .knn(&query_point, k)
                .into_iter()
                .map(|hit| (hit.entry.data, Some(hit.dist_sq)))
                .collect(),
            SpatialScanKind::DWithin {
                query_point,
                radius_meters,
            } => tree
                .dwithin(&query_point, radius_meters)
                .into_iter()
                .map(|entry| (entry.data, None))
                .collect(),
            SpatialScanKind::Range { mbr_min, mbr_max } => {
                let q = Mbr::from_extents(&mbr_min, &mbr_max);
                tree.range(&q)
                    .into_iter()
                    .map(|entry| (entry.data, None))
                    .collect()
            }
        };

        let _table_entry = ctx.get_table_entry(table_id)?;

        let docs: Vec<u64> = results.iter().map(|r| r.0).collect();
        let fetcher =
            DocRowFetcher::prepare(&ctx, table_id, &columns, &docs, as_of.as_ref()).await?;
        let io_stats =
            crate::operator::IndexScanStats::open(&ctx, table_id.0, index_id.0, results.len());

        Ok(Self {
            ctx,
            table_id,
            output_columns: columns,
            results,
            fetcher,
            cursor: 0,
            finished: false,
            heap_version: crate::operator::doc_fetch::heap_as_of_version(as_of.as_ref()),
            io_stats,
        })
    }
}

impl Operator for SpatialScanOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            if self.finished {
                return Ok(None);
            }
            self.ctx.check_cancelled()?;

            let batch_size = self.ctx.batch_size;
            let table_entry = self.ctx.get_table_entry(self.table_id)?;

            let mut builders = create_builders(&self.output_columns, batch_size);
            let output_ids: Vec<zyron_catalog::ColumnId> =
                self.output_columns.iter().map(|c| c.column_id).collect();
            let column_to_builder = build_column_to_builder_map(&table_entry.columns, &output_ids);
            let mut row_count = 0usize;
            // Heap pages fetched to resolve this batch's hits, folded into
            // the table counters once when the batch is done
            let mut pages_read = 0u64;

            while row_count < batch_size && self.cursor < self.results.len() {
                let locator = self.fetcher.locators[self.cursor];
                self.cursor += 1;

                match locator {
                    // dead document, deleted after the index entry landed
                    None => continue,
                    Some(RowLocator::Heap { page, slot }) => {
                        // pool-first read: a committed row can live in a
                        // dirty buffer frame long before any flush
                        let data = crate::operator::scan::read_page_through_pool(
                            &self.ctx.buffer_pool,
                            &self.ctx.disk_manager,
                            page,
                        )
                        .await?;
                        pages_read += 1;
                        let Some(view) =
                            HeapPage::get_tuple_view_from_slice(&data, zyron_storage::SlotId(slot))
                        else {
                            continue;
                        };
                        if !crate::operator::doc_fetch::heap_row_visible(
                            &self.ctx,
                            self.heap_version,
                            view.header.xmin as u64,
                            view.header.xmax as u64,
                        ) {
                            continue;
                        }
                        decode_tuple_into_builders(
                            view.data,
                            &table_entry.columns,
                            &column_to_builder,
                            &mut builders,
                        );
                    }
                    Some(RowLocator::Columnar { file_id, sys_rowid }) => {
                        let Some(vals) = self.fetcher.columnar_row(file_id, sys_rowid) else {
                            continue;
                        };
                        for (b, v) in builders.iter_mut().zip(vals.iter()) {
                            b.push(v);
                        }
                    }
                    Some(RowLocator::Lake { file_id, ordinal }) => {
                        let Some(vals) = self.fetcher.lake_row(file_id, ordinal) else {
                            continue;
                        };
                        for (b, v) in builders.iter_mut().zip(vals.iter()) {
                            b.push(v);
                        }
                    }
                }
                row_count += 1;
            }

            self.io_stats.record_batch(
                row_count as u64,
                pages_read * zyron_common::page::PAGE_SIZE as u64,
            );

            if row_count == 0 {
                self.finished = true;
                return Ok(None);
            }

            let batch = finalize_builders(builders);
            Ok(Some(ExecutionBatch::new(batch)))
        })
    }
}

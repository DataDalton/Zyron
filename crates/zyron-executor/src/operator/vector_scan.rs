//! Vector similarity search scan operator.
//!
//! Uses a vector index (HNSW or IVF-PQ) to find nearest neighbors by distance.
//! Fetches heap tuples by converting VectorId back to TupleId and appends a
//! distance column to each output batch. Enforces VectorSearch privilege
//! and applies ABAC/row-ownership filtering.

use std::sync::Arc;

use zyron_catalog::IndexId;
use zyron_common::Result;
use zyron_common::RowLocator;
use zyron_common::ZyronError;
use zyron_planner::logical::LogicalColumn;
use zyron_storage::HeapPage;

use crate::batch::{
    build_column_to_builder_map, create_builders, decode_tuple_into_builders, finalize_builders,
};
use crate::context::ExecutionContext;
use crate::operator::doc_fetch::DocRowFetcher;
use crate::operator::{ExecutionBatch, Operator, OperatorResult};

/// Operator that executes a vector similarity search against a vector index,
/// then fetches matching heap tuples ordered by distance.
pub struct VectorScanOperator {
    ctx: Arc<ExecutionContext>,
    table_id: zyron_catalog::TableId,
    output_columns: Vec<LogicalColumn>,
    /// Pre-computed (vector_id, distance) results from the vector index.
    results: Vec<(u64, f32)>,
    /// Resolved locators plus pre-fetched columnar hits, result-aligned.
    fetcher: DocRowFetcher,
    /// Current position in the results vector.
    cursor: usize,
    finished: bool,
    /// When set, heap hits are dated by commit LSN at this version instead of
    /// being resolved against the live snapshot
    heap_version: Option<u64>,
    /// Table and index IO counters. The search ran to completion before any
    /// row was fetched, so the entries it examined were recorded then
    io_stats: crate::operator::IndexScanStats,
}

impl VectorScanOperator {
    /// Creates a new vector scan operator. Checks VectorSearch privilege,
    /// extracts the query vector, runs the search, and pre-collects results.
    pub async fn new(
        ctx: Arc<ExecutionContext>,
        table_id: zyron_catalog::TableId,
        index_id: IndexId,
        columns: Vec<LogicalColumn>,
        query_vector: Vec<f32>,
        k: usize,
        ef_search: u16,
        as_of: Option<zyron_planner::logical::AsOfTarget>,
    ) -> Result<Self> {
        // Privilege check: require VectorSearch on the table
        ctx.check_search_privilege(zyron_auth::PrivilegeType::VectorSearch, table_id.0)?;

        let vec_index = ctx
            .get_vector_index(index_id.0)
            .ok_or_else(|| ZyronError::VectorIndexNotFound(format!("IndexId({})", index_id.0)))?;

        // Execute the vector search
        let results = zyron_search::vector::VectorSearch::search(
            vec_index.as_ref(),
            &query_vector,
            k,
            ef_search,
        )?;

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

impl Operator for VectorScanOperator {
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
            let mut distances: Vec<f32> = Vec::with_capacity(batch_size);
            let mut row_count = 0usize;
            // Heap pages fetched to resolve this batch's hits, folded into
            // the table counters once when the batch is done
            let mut pages_read = 0u64;

            while row_count < batch_size && self.cursor < self.results.len() {
                let (_vec_id, distance) = self.results[self.cursor];
                let locator = self.fetcher.locators[self.cursor];
                self.cursor += 1;

                match locator {
                    // dead document, deleted after the index entry was scored
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
                distances.push(distance);
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

            let mut batch = finalize_builders(builders);
            // Append distance as an additional Float32 column stored as Float64
            let float_distances: Vec<f64> = distances.iter().map(|&d| d as f64).collect();
            batch.columns.push(crate::column::Column::new(
                crate::column::ColumnData::Float64(float_distances),
                zyron_common::TypeId::Float64,
            ));
            Ok(Some(ExecutionBatch::new(batch)))
        })
    }
}

//! Vector similarity search scan operator.
//!
//! Uses a vector index (HNSW or IVF-PQ) to find nearest neighbors by distance.
//! Fetches heap tuples by converting VectorId back to TupleId and appends a
//! distance column to each output batch. Enforces VectorSearch privilege
//! and applies ABAC/row-ownership filtering.

use std::sync::Arc;

use zyron_catalog::IndexId;
use zyron_common::Result;
use zyron_common::ZyronError;
use zyron_common::page::{PAGE_SIZE, PageId};
use zyron_planner::binder::BoundExpr;
use zyron_planner::logical::LogicalColumn;
use zyron_search::decode_doc_id;
use zyron_storage::{HeapPage, TupleId};

use crate::batch::{create_builders, decode_tuple_into_builders, finalize_builders};
use crate::context::ExecutionContext;
use crate::operator::{ExecutionBatch, Operator, OperatorResult};

/// Operator that executes a vector similarity search against a vector index,
/// then fetches matching heap tuples ordered by distance.
pub struct VectorScanOperator {
    ctx: Arc<ExecutionContext>,
    table_id: zyron_catalog::TableId,
    output_columns: Vec<LogicalColumn>,
    /// Pre-computed (vector_id, distance) results from the vector index.
    results: Vec<(u64, f32)>,
    /// Current position in the results vector.
    cursor: usize,
    finished: bool,
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

        Ok(Self {
            ctx,
            table_id,
            output_columns: columns,
            results,
            cursor: 0,
            finished: false,
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
            let mut distances: Vec<f32> = Vec::with_capacity(batch_size);
            let mut row_count = 0usize;

            while row_count < batch_size && self.cursor < self.results.len() {
                let (vec_id, distance) = self.results[self.cursor];
                self.cursor += 1;

                // Convert VectorId back to TupleId (same encoding as DocId)
                let (page_num, slot_id) = decode_doc_id(vec_id);

                let page_id = PageId::new(table_entry.heap_file_id, page_num);
                let tid = TupleId::new(page_id, slot_id);

                // Read heap page and extract tuple
                let page_data: [u8; PAGE_SIZE] = self.ctx.disk_manager.read_page(page_id).await?;
                let page = HeapPage::from_bytes(page_data);

                let slot = zyron_storage::SlotId(tid.slot_id);
                let Some(tuple) = page.get_tuple(slot) else {
                    continue;
                };

                if tuple.is_deleted() {
                    continue;
                }
                if !tuple.header().is_visible_to(&self.ctx.snapshot) {
                    continue;
                }

                decode_tuple_into_builders(tuple.data(), &table_entry.columns, &mut builders);
                distances.push(distance);
                row_count += 1;
            }

            if row_count == 0 {
                self.finished = true;
                return Ok(None);
            }

            let mut batch = finalize_builders(builders);
            // Append distance as an additional Float32 column stored as Float64
            let float_distances: Vec<f64> = distances.iter().map(|&d| d as f64).collect();
            batch.columns.push(crate::column::Column {
                data: crate::column::ColumnData::Float64(float_distances),
                nulls: crate::column::NullBitmap::none(row_count),
                type_id: zyron_common::TypeId::Float64,
            });
            Ok(Some(ExecutionBatch::new(batch)))
        })
    }
}

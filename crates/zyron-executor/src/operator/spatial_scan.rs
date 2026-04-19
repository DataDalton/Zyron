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
use zyron_common::ZyronError;
use zyron_common::page::{PAGE_SIZE, PageId};
use zyron_planner::logical::LogicalColumn;
use zyron_planner::physical::SpatialScanKind;
use zyron_search::decode_doc_id;
use zyron_storage::{HeapPage, TupleId};
use zyron_types::spatial_index::Mbr;

use crate::batch::{create_builders, decode_tuple_into_builders, finalize_builders};
use crate::context::ExecutionContext;
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
    /// Current position in the results vector.
    cursor: usize,
    finished: bool,
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
            let mut row_count = 0usize;

            while row_count < batch_size && self.cursor < self.results.len() {
                let (rowid, _) = self.results[self.cursor];
                self.cursor += 1;

                let (page_num, slot_id) = decode_doc_id(rowid);
                let page_id = PageId::new(table_entry.heap_file_id, page_num);
                let tid = TupleId::new(page_id, slot_id);

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
                row_count += 1;
            }

            if row_count == 0 {
                self.finished = true;
                return Ok(None);
            }

            let batch = finalize_builders(builders);
            Ok(Some(ExecutionBatch::new(batch)))
        })
    }
}

//! Graph algorithm execution operator.
//!
//! Builds a CSR representation from the graph schema's backing tables,
//! dispatches to the selected SIMD-accelerated algorithm, and returns
//! results as columnar batches. Enforces GraphTraverse or GraphAlgorithm
//! privileges depending on the operation type.

use std::sync::Arc;

use zyron_common::{Result, ZyronError};
use zyron_planner::logical::LogicalColumn;

use crate::batch::{ColumnBuilder, DataBatch, decode_tuple_into_builders};
use crate::column::{Column, ColumnData, NullBitmap};
use crate::context::ExecutionContext;
use crate::operator::scan::read_page_through_pool;
use crate::operator::{ExecutionBatch, Operator, OperatorResult};

/// The type of graph algorithm to execute.
#[derive(Debug, Clone)]
pub enum GraphAlgorithmKind {
    PageRank { damping: f64, iterations: usize },
    ShortestPath { source_id: u64, target_id: u64 },
    Bfs { source_id: u64, max_depth: u32 },
    ConnectedComponents,
    CommunityDetection,
    BetweennessCentrality,
}

/// Operator that executes a graph algorithm and returns results as batches.
pub struct GraphAlgorithmOperator {
    /// Pre-computed result batches from the algorithm.
    batches: Vec<DataBatch>,
    /// Current batch index.
    cursor: usize,
    finished: bool,
}

impl GraphAlgorithmOperator {
    /// Creates a new graph algorithm operator. Checks privileges, builds CSR,
    /// executes the algorithm, and pre-computes result batches.
    pub async fn new(
        ctx: Arc<ExecutionContext>,
        schema_name: String,
        algorithm: GraphAlgorithmKind,
        output_columns: Vec<LogicalColumn>,
    ) -> Result<Self> {
        // Privilege check: GraphAlgorithm for compute-heavy operations
        let privilege = match &algorithm {
            GraphAlgorithmKind::ShortestPath { .. } | GraphAlgorithmKind::Bfs { .. } => {
                zyron_auth::PrivilegeType::GraphTraverse
            }
            _ => zyron_auth::PrivilegeType::GraphAlgorithm,
        };

        // Check privilege on schema (object_id 0 since we check by name-based policy)
        // For graph schemas, the privilege is checked at the schema level
        ctx.check_search_privilege(privilege, 0)?;

        let graph_mgr = ctx.graph_manager.clone().ok_or_else(|| {
            ZyronError::GraphSchemaNotFound("graph manager not configured".to_string())
        })?;

        let schema = graph_mgr
            .get_schema(&schema_name)
            .ok_or_else(|| ZyronError::GraphSchemaNotFound(schema_name.clone()))?;

        // Use the cached CSR when present, otherwise load the edges from the
        // backing edge tables' heaps (honoring the statement snapshot) and build
        // the CSR. The cache is invalidated by DML on the backing tables, so a
        // cold or post-mutation query rebuilds it here rather than failing.
        let csr = match graph_mgr.get_cached_csr(&schema_name) {
            Some(cached) => cached,
            None => {
                let edges = load_graph_edges(&ctx, &schema).await?;
                graph_mgr.get_or_build_csr(&schema_name, &edges)?
            }
        };

        // Execute the algorithm
        let batches = match algorithm {
            GraphAlgorithmKind::PageRank {
                damping,
                iterations,
            } => {
                let results = zyron_search::graph::algorithms::pagerank(&csr, damping, iterations)?;
                build_node_score_batches(&results, &output_columns)
            }
            GraphAlgorithmKind::ShortestPath {
                source_id,
                target_id,
            } => {
                let path =
                    zyron_search::graph::algorithms::shortest_path(&csr, source_id, target_id)?;
                match path {
                    Some(p) => build_path_batches(&p, &output_columns),
                    None => vec![],
                }
            }
            GraphAlgorithmKind::Bfs {
                source_id,
                max_depth,
            } => {
                let pairs = zyron_search::graph::algorithms::bfs(&csr, source_id, max_depth)?;
                build_node_depth_batches(&pairs, &output_columns)
            }
            GraphAlgorithmKind::ConnectedComponents => {
                let components = zyron_search::graph::algorithms::connected_components(&csr)?;
                build_component_batches(&components, &output_columns)
            }
            GraphAlgorithmKind::CommunityDetection => {
                let communities = zyron_search::graph::algorithms::community_detection(&csr)?;
                build_component_batches(&communities, &output_columns)
            }
            GraphAlgorithmKind::BetweennessCentrality => {
                let results = zyron_search::graph::algorithms::betweenness_centrality(&csr)?;
                build_node_score_batches(&results, &output_columns)
            }
        };

        Ok(Self {
            batches,
            cursor: 0,
            finished: false,
        })
    }
}

impl Operator for GraphAlgorithmOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            if self.finished || self.cursor >= self.batches.len() {
                self.finished = true;
                return Ok(None);
            }
            let batch = self.batches[self.cursor].clone();
            self.cursor += 1;
            Ok(Some(ExecutionBatch::new(batch)))
        })
    }
}

/// Loads every edge of a graph schema from its backing edge tables' heaps.
/// Each edge table follows the graph convention of `from_node`/`to_node`
/// node-id columns plus an optional `weight` column. Only rows visible to the
/// statement snapshot are read. An undirected edge label contributes both
/// directions so directed algorithms traverse it symmetrically.
async fn load_graph_edges(
    ctx: &Arc<ExecutionContext>,
    schema: &zyron_search::graph::GraphSchema,
) -> Result<Vec<(u64, u64, Option<f64>)>> {
    let mut edges: Vec<(u64, u64, Option<f64>)> = Vec::new();

    for edge_label in &schema.edge_labels {
        let table = ctx
            .catalog
            .get_table_by_id(zyron_catalog::TableId(edge_label.edge_table_id))?;

        let column_pos = |name: &str| table.columns.iter().position(|c| c.name == name);
        let from_pos = column_pos("from_node").ok_or_else(|| {
            ZyronError::GraphAlgorithmError(format!(
                "edge table for label '{}' has no from_node column",
                edge_label.name
            ))
        })?;
        let to_pos = column_pos("to_node").ok_or_else(|| {
            ZyronError::GraphAlgorithmError(format!(
                "edge table for label '{}' has no to_node column",
                edge_label.name
            ))
        })?;
        let weight_pos = column_pos("weight");

        // Decode only the node-id (and optional weight) columns: map every other
        // table column to None so the decoder skips it.
        let mut col_to_builder: Vec<Option<u16>> = vec![None; table.columns.len()];
        let mut builders: Vec<ColumnBuilder> = Vec::with_capacity(3);
        col_to_builder[from_pos] = Some(builders.len() as u16);
        builders.push(ColumnBuilder::new(table.columns[from_pos].type_id, 0));
        col_to_builder[to_pos] = Some(builders.len() as u16);
        builders.push(ColumnBuilder::new(table.columns[to_pos].type_id, 0));
        let weight_builder = weight_pos.map(|wp| {
            let idx = builders.len();
            col_to_builder[wp] = Some(idx as u16);
            builders.push(ColumnBuilder::new(table.columns[wp].type_id, 0));
            idx
        });

        let heap = ctx.get_heap_file(table.id).await?;
        let num_pages = heap.num_pages_cached() as u32;
        for page_num in 0..num_pages {
            ctx.check_cancelled()?;
            let page_id = zyron_common::page::PageId::new(table.heap_file_id, page_num as u64);
            let page_data =
                read_page_through_pool(&ctx.buffer_pool, &ctx.disk_manager, page_id).await?;
            let header = zyron_storage::HeapPage::heap_header_from_slice(&page_data);
            if header.slot_count == 0 {
                continue;
            }
            let page = zyron_storage::HeapPage::from_bytes(page_data);
            for slot in 0..header.slot_count {
                let Some(view) = page.get_tuple_view(zyron_storage::SlotId(slot)) else {
                    continue;
                };
                if view.is_deleted() || !view.header.is_visible_to(&ctx.snapshot) {
                    continue;
                }
                decode_tuple_into_builders(
                    view.data,
                    &table.columns,
                    &col_to_builder,
                    &mut builders,
                );
            }
        }

        let columns: Vec<Column> = builders.into_iter().map(|b| b.finish()).collect();
        let from_col = &columns[0];
        let to_col = &columns[1];
        let weight_col = weight_builder.map(|i| &columns[i]);
        let row_count = from_col.len();
        for row in 0..row_count {
            if from_col.is_null(row) || to_col.is_null(row) {
                continue;
            }
            let (Some(from), Some(to)) = (
                scalar_to_node_id(&from_col.get_scalar(row)),
                scalar_to_node_id(&to_col.get_scalar(row)),
            ) else {
                continue;
            };
            let weight = weight_col.and_then(|c| {
                if c.is_null(row) {
                    None
                } else {
                    scalar_to_weight(&c.get_scalar(row))
                }
            });
            edges.push((from, to, weight));
            if !edge_label.directed {
                edges.push((to, from, weight));
            }
        }
    }

    Ok(edges)
}

/// Reads a node-id column value as a u64, accepting any integer width.
fn scalar_to_node_id(s: &crate::column::ScalarValue) -> Option<u64> {
    use crate::column::ScalarValue as V;
    match s {
        V::Int8(v) => Some(*v as u64),
        V::Int16(v) => Some(*v as u64),
        V::Int32(v) => Some(*v as u64),
        V::Int64(v) => Some(*v as u64),
        V::UInt8(v) => Some(*v as u64),
        V::UInt16(v) => Some(*v as u64),
        V::UInt32(v) => Some(*v as u64),
        V::UInt64(v) => Some(*v),
        _ => None,
    }
}

/// Reads an edge-weight column value as f64, accepting float or integer types.
fn scalar_to_weight(s: &crate::column::ScalarValue) -> Option<f64> {
    use crate::column::ScalarValue as V;
    match s {
        V::Float32(v) => Some(*v as f64),
        V::Float64(v) => Some(*v),
        V::Int8(v) => Some(*v as f64),
        V::Int16(v) => Some(*v as f64),
        V::Int32(v) => Some(*v as f64),
        V::Int64(v) => Some(*v as f64),
        V::UInt8(v) => Some(*v as f64),
        V::UInt16(v) => Some(*v as f64),
        V::UInt32(v) => Some(*v as f64),
        V::UInt64(v) => Some(*v as f64),
        _ => None,
    }
}

/// Builds batches for (node_id, score) results. Output schema from the binder
/// is (node_id: Int64, score: Float64) for PageRank and
/// (node_id: Int64, centrality: Float64) for betweenness centrality.
fn build_node_score_batches(
    results: &[(u64, f64)],
    _output_columns: &[LogicalColumn],
) -> Vec<DataBatch> {
    if results.is_empty() {
        return vec![];
    }

    let row_count = results.len();
    let mut node_ids: Vec<i64> = Vec::with_capacity(row_count);
    let mut scores: Vec<f64> = Vec::with_capacity(row_count);

    for &(nid, score) in results {
        node_ids.push(nid as i64);
        scores.push(score);
    }

    let batch = DataBatch::new(vec![
        Column::new(ColumnData::Int64(node_ids), zyron_common::TypeId::Int64),
        Column::new(ColumnData::Float64(scores), zyron_common::TypeId::Float64),
    ]);
    vec![batch]
}

/// Builds batches for a shortest-path result. Output schema from the binder
/// is (step: Int32, node_id: Int64). Step starts at 0 at the source.
fn build_path_batches(path: &[u64], _output_columns: &[LogicalColumn]) -> Vec<DataBatch> {
    if path.is_empty() {
        return vec![];
    }

    let row_count = path.len();
    let mut steps: Vec<i32> = Vec::with_capacity(row_count);
    let mut node_ids: Vec<i64> = Vec::with_capacity(row_count);

    for (i, &nid) in path.iter().enumerate() {
        steps.push(i as i32);
        node_ids.push(nid as i64);
    }

    let batch = DataBatch::new(vec![
        Column::new(ColumnData::Int32(steps), zyron_common::TypeId::Int32),
        Column::new(ColumnData::Int64(node_ids), zyron_common::TypeId::Int64),
    ]);
    vec![batch]
}

/// Builds batches for BFS results. Output schema from the binder is
/// (node_id: Int64, depth: Int32).
fn build_node_depth_batches(
    pairs: &[(u64, u32)],
    _output_columns: &[LogicalColumn],
) -> Vec<DataBatch> {
    if pairs.is_empty() {
        return vec![];
    }

    let row_count = pairs.len();
    let mut node_ids: Vec<i64> = Vec::with_capacity(row_count);
    let mut depths: Vec<i32> = Vec::with_capacity(row_count);

    for &(nid, depth) in pairs {
        node_ids.push(nid as i64);
        depths.push(depth as i32);
    }

    let batch = DataBatch::new(vec![
        Column::new(ColumnData::Int64(node_ids), zyron_common::TypeId::Int64),
        Column::new(ColumnData::Int32(depths), zyron_common::TypeId::Int32),
    ]);
    vec![batch]
}

/// Builds batches for component/community results.
/// Each row is (node_id, component_id).
fn build_component_batches(
    components: &[Vec<u64>],
    _output_columns: &[LogicalColumn],
) -> Vec<DataBatch> {
    let total_rows: usize = components.iter().map(|c| c.len()).sum();
    if total_rows == 0 {
        return vec![];
    }

    let mut node_ids: Vec<i64> = Vec::with_capacity(total_rows);
    let mut component_ids: Vec<i64> = Vec::with_capacity(total_rows);

    for (comp_id, component) in components.iter().enumerate() {
        for &nid in component {
            node_ids.push(nid as i64);
            component_ids.push(comp_id as i64);
        }
    }

    let batch = DataBatch::new(vec![
        Column::new(ColumnData::Int64(node_ids), zyron_common::TypeId::Int64),
        Column::new(
            ColumnData::Int64(component_ids),
            zyron_common::TypeId::Int64,
        ),
    ]);
    vec![batch]
}

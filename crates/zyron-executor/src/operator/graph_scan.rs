//! Graph algorithm execution operator.
//!
//! Builds a CSR representation from the graph schema's backing tables,
//! dispatches to the selected SIMD-accelerated algorithm, and returns
//! results as columnar batches. Enforces GraphTraverse or GraphAlgorithm
//! privileges depending on the operation type.

use std::sync::Arc;

use zyron_common::{Result, ZyronError};
use zyron_planner::logical::LogicalColumn;

use crate::batch::DataBatch;
use crate::column::{Column, ColumnData, NullBitmap};
use crate::context::ExecutionContext;
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

        let graph_mgr = ctx.graph_manager.as_ref().ok_or_else(|| {
            ZyronError::GraphSchemaNotFound("graph manager not configured".to_string())
        })?;

        let _schema = graph_mgr
            .get_schema(&schema_name)
            .ok_or_else(|| ZyronError::GraphSchemaNotFound(schema_name.clone()))?;

        // Build or retrieve cached CSR
        // For now, we need edge data from the backing tables. This would normally
        // be loaded from the heap pages. As a placeholder, use the cached CSR
        // or return an error if not cached.
        let csr = graph_mgr.get_cached_csr(&schema_name).ok_or_else(|| {
            ZyronError::GraphAlgorithmError(format!(
                "no cached CSR for graph schema '{}'. Run a graph query to build it first.",
                schema_name
            ))
        })?;

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
        Column {
            data: ColumnData::Int64(node_ids),
            nulls: NullBitmap::none(row_count),
            type_id: zyron_common::TypeId::Int64,
        },
        Column {
            data: ColumnData::Float64(scores),
            nulls: NullBitmap::none(row_count),
            type_id: zyron_common::TypeId::Float64,
        },
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
        Column {
            data: ColumnData::Int32(steps),
            nulls: NullBitmap::none(row_count),
            type_id: zyron_common::TypeId::Int32,
        },
        Column {
            data: ColumnData::Int64(node_ids),
            nulls: NullBitmap::none(row_count),
            type_id: zyron_common::TypeId::Int64,
        },
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
        Column {
            data: ColumnData::Int64(node_ids),
            nulls: NullBitmap::none(row_count),
            type_id: zyron_common::TypeId::Int64,
        },
        Column {
            data: ColumnData::Int32(depths),
            nulls: NullBitmap::none(row_count),
            type_id: zyron_common::TypeId::Int32,
        },
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
        Column {
            data: ColumnData::Int64(node_ids),
            nulls: NullBitmap::none(total_rows),
            type_id: zyron_common::TypeId::Int64,
        },
        Column {
            data: ColumnData::Int64(component_ids),
            nulls: NullBitmap::none(total_rows),
            type_id: zyron_common::TypeId::Int64,
        },
    ]);
    vec![batch]
}

//! Physical execution plan representation.
//!
//! Physical plans map logical operators to concrete execution strategies.
//! Each variant specifies how to implement the operator (e.g., HashJoin vs MergeJoin)
//! and carries cost estimates for plan comparison.

pub mod builder;

use crate::binder::{BoundAssignment, BoundExpr, BoundOrderBy};
use crate::cost::PlanCost;
use crate::logical::{AggregateExpr, LogicalColumn};
use std::sync::Arc;
use zyron_catalog::{ColumnId, IndexEntry, IndexId, TableId};
use zyron_parser::ast::{JoinType, SetOpType};

/// One aggregate answered from columnar segment metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetaAggKind {
    /// COUNT(*): sum of segment row counts plus heap residual.
    CountStar,
    /// COUNT(col): row count minus that column's null count.
    CountCol,
    /// MIN(col) from per-segment header min (fixed-width columns only).
    Min,
    /// MAX(col) from per-segment header max (fixed-width columns only).
    Max,
}

/// Specification of one metadata-pushdown aggregate output column.
#[derive(Debug, Clone)]
pub struct MetaAggSpec {
    pub kind: MetaAggKind,
    /// Target column id for CountCol/Min/Max; ignored for CountStar.
    pub column_id: Option<ColumnId>,
    /// Result type id.
    pub return_type: zyron_common::types::TypeId,
    /// Output column name.
    pub name: String,
}

/// Physical execution plan. Each variant maps to a concrete operator
/// and carries cost estimates.
#[derive(Debug, Clone)]
pub enum PhysicalPlan {
    /// Full table scan reading all pages sequentially.
    SeqScan {
        table_id: TableId,
        columns: Vec<LogicalColumn>,
        predicate: Option<BoundExpr>,
        cost: PlanCost,
        /// Time travel target for versioned table scans.
        as_of: Option<super::logical::AsOfTarget>,
    },

    /// Hybrid scan: the union of the table's registered .zyr columnar
    /// segments and the heap residual, reconciled by snapshot visibility.
    /// A folded row exists in exactly one store, so the union never double
    /// counts. Chosen when the table has registered columnar segments.
    HybridScan {
        table_id: TableId,
        columns: Vec<LogicalColumn>,
        predicate: Option<BoundExpr>,
        cost: PlanCost,
    },

    /// MIN/MAX/COUNT answered from columnar segment headers plus the heap
    /// residual, with no decode of the folded rows. Emitted only when the
    /// table has registered segments, there is no GROUP BY, no predicate, no
    /// DISTINCT, and the patch overlay is consulted at execution so the
    /// metadata is MVCC-safe (a non-clean segment falls back to a scan).
    ColumnarMetadataAggregate {
        table_id: TableId,
        specs: Vec<MetaAggSpec>,
        schema: Vec<LogicalColumn>,
        cost: PlanCost,
    },

    /// Index-based scan for selective predicates.
    IndexScan {
        table_id: TableId,
        index_id: IndexId,
        index: Arc<IndexEntry>,
        columns: Vec<LogicalColumn>,
        predicate: BoundExpr,
        remaining_predicate: Option<BoundExpr>,
        scan_direction: ScanDirection,
        cost: PlanCost,
        /// Time travel target for versioned table scans.
        as_of: Option<super::logical::AsOfTarget>,
    },

    /// Filter rows by predicate.
    Filter {
        predicate: BoundExpr,
        child: Box<PhysicalPlan>,
        cost: PlanCost,
    },

    /// Project output columns.
    Project {
        expressions: Vec<BoundExpr>,
        aliases: Vec<Option<String>>,
        child: Box<PhysicalPlan>,
        cost: PlanCost,
    },

    /// Nested loop join.
    NestedLoopJoin {
        left: Box<PhysicalPlan>,
        right: Box<PhysicalPlan>,
        join_type: JoinType,
        condition: Option<BoundExpr>,
        cost: PlanCost,
    },

    /// Hash join. Build hash table on the smaller side, probe with the larger.
    HashJoin {
        left: Box<PhysicalPlan>,
        right: Box<PhysicalPlan>,
        join_type: JoinType,
        left_keys: Vec<BoundExpr>,
        right_keys: Vec<BoundExpr>,
        remaining_condition: Option<BoundExpr>,
        cost: PlanCost,
    },

    /// Sort-merge join. Both sides pre-sorted on join keys.
    MergeJoin {
        left: Box<PhysicalPlan>,
        right: Box<PhysicalPlan>,
        join_type: JoinType,
        left_keys: Vec<BoundExpr>,
        right_keys: Vec<BoundExpr>,
        cost: PlanCost,
    },

    /// Hash-based aggregation.
    HashAggregate {
        group_by: Vec<BoundExpr>,
        aggregates: Vec<AggregateExpr>,
        child: Box<PhysicalPlan>,
        cost: PlanCost,
    },

    /// Sort-based aggregation (used when input is already sorted on group keys).
    SortAggregate {
        group_by: Vec<BoundExpr>,
        aggregates: Vec<AggregateExpr>,
        child: Box<PhysicalPlan>,
        cost: PlanCost,
    },

    /// Time-bucket gap fill. Densifies a time-bucketed aggregate by emitting a
    /// row for every bucket in the observed [min, max] range stepping by
    /// `width`. Absent buckets get the bucket value and NULL for all other
    /// columns. `bucket_col` is the index of the time_bucket_gapfill grouping
    /// column in the child output.
    GapFill {
        bucket_col: usize,
        width: i128,
        child: Box<PhysicalPlan>,
        cost: PlanCost,
    },

    /// External sort (top-N uses a bounded heap when limit is present).
    Sort {
        order_by: Vec<BoundOrderBy>,
        child: Box<PhysicalPlan>,
        limit: Option<u64>,
        cost: PlanCost,
    },

    /// Limit and offset.
    Limit {
        limit: Option<u64>,
        offset: Option<u64>,
        child: Box<PhysicalPlan>,
        cost: PlanCost,
    },

    /// Distinct via hash set.
    HashDistinct {
        child: Box<PhysicalPlan>,
        cost: PlanCost,
    },

    /// Set operation (UNION, INTERSECT, EXCEPT).
    SetOp {
        op: SetOpType,
        all: bool,
        left: Box<PhysicalPlan>,
        right: Box<PhysicalPlan>,
        cost: PlanCost,
    },

    /// Insert rows into a table.
    Insert {
        table_id: TableId,
        target_columns: Vec<ColumnId>,
        source: Box<PhysicalPlan>,
        cost: PlanCost,
    },

    /// Inline values (produces rows from constants).
    Values {
        rows: Vec<Vec<BoundExpr>>,
        schema: Vec<LogicalColumn>,
        cost: PlanCost,
    },

    /// Update matching rows.
    Update {
        table_id: TableId,
        assignments: Vec<BoundAssignment>,
        child: Box<PhysicalPlan>,
        cost: PlanCost,
    },

    /// Delete matching rows.
    Delete {
        table_id: TableId,
        child: Box<PhysicalPlan>,
        cost: PlanCost,
    },

    /// Parallel sequential scan distributing page ranges across workers.
    ParallelSeqScan {
        table_id: TableId,
        columns: Vec<LogicalColumn>,
        predicate: Option<BoundExpr>,
        num_workers: usize,
        cost: PlanCost,
    },

    /// Parallel hash join with partitioned build and probe phases.
    ParallelHashJoin {
        left: Box<PhysicalPlan>,
        right: Box<PhysicalPlan>,
        join_type: JoinType,
        left_keys: Vec<BoundExpr>,
        right_keys: Vec<BoundExpr>,
        remaining_condition: Option<BoundExpr>,
        num_workers: usize,
        cost: PlanCost,
    },

    /// Exchange operator: gathers partitioned streams into one.
    Gather {
        child: Box<PhysicalPlan>,
        num_workers: usize,
        cost: PlanCost,
    },

    /// Exchange operator: repartitions data by hash for parallel joins.
    Repartition {
        child: Box<PhysicalPlan>,
        partition_keys: Vec<BoundExpr>,
        num_partitions: usize,
        cost: PlanCost,
    },

    /// Exchange operator: broadcasts small table to all workers.
    Broadcast {
        child: Box<PhysicalPlan>,
        num_workers: usize,
        cost: PlanCost,
    },

    /// Full-text search scan using an inverted index.
    FulltextScan {
        table_id: TableId,
        index_id: IndexId,
        columns: Vec<LogicalColumn>,
        /// The match_against function call containing column refs and query.
        match_expr: BoundExpr,
        /// Additional predicates to apply after FTS scoring.
        remaining_predicate: Option<BoundExpr>,
        cost: PlanCost,
    },

    /// Approximate nearest neighbor scan using a vector index.
    VectorScan {
        table_id: TableId,
        index_id: IndexId,
        columns: Vec<LogicalColumn>,
        query_vector: Vec<f32>,
        metric: u8,
        k: usize,
        remaining_predicate: Option<BoundExpr>,
        cost: PlanCost,
    },

    /// Spatial scan over an R-tree index. Serves KNN, ST_DWithin,
    /// ST_Intersects, and bounding-box range queries.
    SpatialScan {
        table_id: TableId,
        index_id: IndexId,
        columns: Vec<LogicalColumn>,
        kind: SpatialScanKind,
        remaining_predicate: Option<BoundExpr>,
        cost: PlanCost,
    },

    /// Graph algorithm execution over a schema's edge/vertex tables.
    GraphAlgorithm {
        algorithm: GraphAlgorithmType,
        schema_name: String,
        params: Vec<(String, BoundExpr)>,
        output_columns: Vec<LogicalColumn>,
        cost: PlanCost,
    },

    /// Analytics table-returning function (COHORT_RETENTION,
    /// FUNNEL_ANALYSIS, DATA_PROFILE, COLUMN_PROFILE, CORRELATION_MATRIX).
    /// Resolved by the binder against the analytics function registry.
    AnalyticsTableFunction {
        function_name: String,
        named_args: Vec<(String, BoundExpr)>,
        positional_args: Vec<BoundExpr>,
        output_columns: Vec<LogicalColumn>,
        cost: PlanCost,
    },

    /// Window function evaluation over partitioned and ordered input.
    /// Drains the child, sorts by (partition_by, order_by), applies each
    /// window function per-partition, and appends result columns.
    Window {
        /// Each entry is a `BoundExpr::WindowFunction` node describing
        /// a function call with its partition_by/order_by keys.
        window_exprs: Vec<BoundExpr>,
        /// Column name for each window expression output.
        window_names: Vec<String>,
        child: Box<PhysicalPlan>,
        cost: PlanCost,
    },
}

/// Mode of a spatial index scan, set at plan time from the matched predicate
/// or sort+limit pattern. The executor reads this to dispatch the right
/// R-tree query method.
#[derive(Debug, Clone)]
pub enum SpatialScanKind {
    /// K-nearest-neighbor: ORDER BY ST_Distance(col, query) [ASC] LIMIT k.
    Knn { query_point: Vec<f64>, k: usize },
    /// Distance filter: WHERE ST_DWithin(col, query, radius).
    DWithin {
        query_point: Vec<f64>,
        radius_meters: f64,
    },
    /// Bounding-box range: WHERE ST_Intersects(col, env_mbr) or equivalent.
    Range {
        mbr_min: Vec<f64>,
        mbr_max: Vec<f64>,
    },
}

#[derive(Debug, Clone)]
pub enum GraphAlgorithmType {
    PageRank,
    ShortestPath,
    Bfs,
    ConnectedComponents,
    CommunityDetection,
    BetweennessCentrality,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanDirection {
    Forward,
    Backward,
}

impl PhysicalPlan {
    /// Returns the estimated cost of this plan node.
    pub fn cost(&self) -> &PlanCost {
        match self {
            PhysicalPlan::SeqScan { cost, .. }
            | PhysicalPlan::HybridScan { cost, .. }
            | PhysicalPlan::ColumnarMetadataAggregate { cost, .. }
            | PhysicalPlan::IndexScan { cost, .. }
            | PhysicalPlan::Filter { cost, .. }
            | PhysicalPlan::Project { cost, .. }
            | PhysicalPlan::NestedLoopJoin { cost, .. }
            | PhysicalPlan::HashJoin { cost, .. }
            | PhysicalPlan::MergeJoin { cost, .. }
            | PhysicalPlan::HashAggregate { cost, .. }
            | PhysicalPlan::SortAggregate { cost, .. }
            | PhysicalPlan::Sort { cost, .. }
            | PhysicalPlan::Limit { cost, .. }
            | PhysicalPlan::HashDistinct { cost, .. }
            | PhysicalPlan::SetOp { cost, .. }
            | PhysicalPlan::Insert { cost, .. }
            | PhysicalPlan::Values { cost, .. }
            | PhysicalPlan::Update { cost, .. }
            | PhysicalPlan::Delete { cost, .. }
            | PhysicalPlan::ParallelSeqScan { cost, .. }
            | PhysicalPlan::ParallelHashJoin { cost, .. }
            | PhysicalPlan::Gather { cost, .. }
            | PhysicalPlan::Repartition { cost, .. }
            | PhysicalPlan::Broadcast { cost, .. }
            | PhysicalPlan::FulltextScan { cost, .. }
            | PhysicalPlan::VectorScan { cost, .. }
            | PhysicalPlan::SpatialScan { cost, .. }
            | PhysicalPlan::GraphAlgorithm { cost, .. }
            | PhysicalPlan::AnalyticsTableFunction { cost, .. }
            | PhysicalPlan::GapFill { cost, .. }
            | PhysicalPlan::Window { cost, .. } => cost,
        }
    }

    /// Returns the output schema of this plan node.
    pub fn output_schema(&self) -> Vec<LogicalColumn> {
        match self {
            PhysicalPlan::SeqScan { columns, .. }
            | PhysicalPlan::HybridScan { columns, .. }
            | PhysicalPlan::IndexScan { columns, .. }
            | PhysicalPlan::ParallelSeqScan { columns, .. } => columns.clone(),
            PhysicalPlan::ColumnarMetadataAggregate { schema, .. } => schema.clone(),
            PhysicalPlan::Filter { child, .. } => child.output_schema(),
            PhysicalPlan::Project {
                expressions,
                aliases,
                ..
            } => expressions
                .iter()
                .enumerate()
                .map(|(i, expr)| {
                    let name = aliases
                        .get(i)
                        .and_then(|a| a.clone())
                        .unwrap_or_else(|| format!("col{}", i));
                    LogicalColumn {
                        table_idx: None,
                        column_id: ColumnId(i as u16),
                        name,
                        type_id: expr.type_id(),
                        nullable: expr.nullable(),
                        ts_precision: None,
                    }
                })
                .collect(),
            PhysicalPlan::NestedLoopJoin { left, right, .. }
            | PhysicalPlan::HashJoin { left, right, .. }
            | PhysicalPlan::MergeJoin { left, right, .. }
            | PhysicalPlan::ParallelHashJoin { left, right, .. } => {
                let mut schema = left.output_schema();
                schema.extend(right.output_schema());
                schema
            }
            PhysicalPlan::HashAggregate {
                group_by,
                aggregates,
                ..
            }
            | PhysicalPlan::SortAggregate {
                group_by,
                aggregates,
                ..
            } => {
                let mut schema = Vec::new();
                for (i, expr) in group_by.iter().enumerate() {
                    schema.push(LogicalColumn {
                        table_idx: Some(crate::logical::AGGREGATE_TABLE_IDX),
                        column_id: ColumnId(i as u16),
                        name: format!("group{}", i),
                        type_id: expr.type_id(),
                        nullable: expr.nullable(),
                        ts_precision: None,
                    });
                }
                for (i, agg) in aggregates.iter().enumerate() {
                    let idx = group_by.len() + i;
                    schema.push(LogicalColumn {
                        table_idx: Some(crate::logical::AGGREGATE_TABLE_IDX),
                        column_id: ColumnId(idx as u16),
                        name: agg.function_name.clone(),
                        type_id: agg.return_type,
                        nullable: true,
                        ts_precision: None,
                    });
                }
                schema
            }
            PhysicalPlan::Sort { child, .. }
            | PhysicalPlan::Limit { child, .. }
            | PhysicalPlan::HashDistinct { child, .. }
            | PhysicalPlan::Gather { child, .. }
            | PhysicalPlan::Repartition { child, .. }
            | PhysicalPlan::GapFill { child, .. }
            | PhysicalPlan::Broadcast { child, .. } => child.output_schema(),
            PhysicalPlan::SetOp { left, .. } => left.output_schema(),
            PhysicalPlan::Insert { .. }
            | PhysicalPlan::Update { .. }
            | PhysicalPlan::Delete { .. } => Vec::new(),
            PhysicalPlan::Values { schema, .. } => schema.clone(),
            PhysicalPlan::FulltextScan { columns, .. }
            | PhysicalPlan::VectorScan { columns, .. }
            | PhysicalPlan::SpatialScan { columns, .. } => columns.clone(),
            PhysicalPlan::GraphAlgorithm { output_columns, .. } => output_columns.clone(),
            PhysicalPlan::AnalyticsTableFunction { output_columns, .. } => output_columns.clone(),
            PhysicalPlan::Window {
                window_exprs,
                window_names,
                child,
                ..
            } => {
                let mut schema = child.output_schema();
                for (i, expr) in window_exprs.iter().enumerate() {
                    let name = window_names
                        .get(i)
                        .cloned()
                        .unwrap_or_else(|| format!("window{}", i));
                    schema.push(LogicalColumn {
                        table_idx: None,
                        column_id: ColumnId((schema.len()) as u16),
                        name,
                        type_id: expr.type_id(),
                        nullable: true,
                        ts_precision: None,
                    });
                }
                schema
            }
        }
    }

    /// Returns the total cost of this node plus all children.
    pub fn total_cost(&self) -> PlanCost {
        let own = *self.cost();
        let children_cost = match self {
            PhysicalPlan::SeqScan { .. }
            | PhysicalPlan::HybridScan { .. }
            | PhysicalPlan::ColumnarMetadataAggregate { .. }
            | PhysicalPlan::IndexScan { .. }
            | PhysicalPlan::Values { .. }
            | PhysicalPlan::ParallelSeqScan { .. }
            | PhysicalPlan::FulltextScan { .. }
            | PhysicalPlan::VectorScan { .. }
            | PhysicalPlan::SpatialScan { .. }
            | PhysicalPlan::GraphAlgorithm { .. }
            | PhysicalPlan::AnalyticsTableFunction { .. } => PlanCost::zero(),
            PhysicalPlan::Filter { child, .. }
            | PhysicalPlan::Project { child, .. }
            | PhysicalPlan::HashAggregate { child, .. }
            | PhysicalPlan::SortAggregate { child, .. }
            | PhysicalPlan::Sort { child, .. }
            | PhysicalPlan::Limit { child, .. }
            | PhysicalPlan::HashDistinct { child, .. }
            | PhysicalPlan::Insert { source: child, .. }
            | PhysicalPlan::Update { child, .. }
            | PhysicalPlan::Delete { child, .. }
            | PhysicalPlan::Gather { child, .. }
            | PhysicalPlan::Repartition { child, .. }
            | PhysicalPlan::Broadcast { child, .. }
            | PhysicalPlan::GapFill { child, .. }
            | PhysicalPlan::Window { child, .. } => child.total_cost(),
            PhysicalPlan::NestedLoopJoin { left, right, .. }
            | PhysicalPlan::HashJoin { left, right, .. }
            | PhysicalPlan::MergeJoin { left, right, .. }
            | PhysicalPlan::SetOp { left, right, .. }
            | PhysicalPlan::ParallelHashJoin { left, right, .. } => {
                left.total_cost().add(&right.total_cost())
            }
        };
        PlanCost {
            io_cost: own.io_cost + children_cost.io_cost,
            cpu_cost: own.cpu_cost + children_cost.cpu_cost,
            row_count: own.row_count,
        }
    }
}

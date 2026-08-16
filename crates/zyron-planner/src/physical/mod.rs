//! Physical execution plan representation.
//!
//! Physical plans map logical operators to concrete execution strategies.
//! Each variant specifies how to implement the operator (e.g., HashJoin vs MergeJoin)
//! and carries cost estimates for plan comparison.

pub mod builder;

use crate::binder::{BoundAssignment, BoundExpr, BoundOrderBy};
use crate::cost::PlanCost;
use crate::logical::{AggregateExpr, AsOfTarget, LogicalColumn};
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
        /// Time-travel target. None for a current read; Some(Version) routes the
        /// columnar and heap scans to commit-LSN version visibility so a query
        /// as of a past version sees folded rows too.
        as_of: Option<super::logical::AsOfTarget>,
        cost: PlanCost,
    },

    /// Scan of a lake table, driven by its transaction log manifest. A
    /// lake table has no heap rows and no MVCC system columns, visibility
    /// is the manifest at the resolved log version, so no other scan node
    /// can serve it. Chosen whenever the catalog marks the table lake.
    LakeScan {
        table_id: TableId,
        columns: Vec<LogicalColumn>,
        predicate: Option<BoundExpr>,
        /// The predicate lowered to the lake IR, present only when it has an
        /// exact equivalent. The operator prunes files with it, and its
        /// absence is why a scan reads every file, so EXPLAIN reports it.
        lowered: Option<zyron_lake::LakePredicate>,
        /// Time-travel target resolved by the operator against the log,
        /// version directly and timestamp through commit timestamps.
        as_of: Option<super::logical::AsOfTarget>,
        cost: PlanCost,
    },

    /// Scan of a table that lives on a peer. The projection, the filter and
    /// the row cap travel to the remote inside `request`, because a
    /// federated scan that fetched everything and filtered here would pay
    /// the network for work the peer could have skipped.
    ///
    /// `residual` is the part of the predicate with no faithful SQL
    /// rendering, evaluated locally on the rows that come back. A conjunct
    /// is in exactly one of the two places, so no row is filtered twice and
    /// none is missed.
    ForeignScan {
        table_id: TableId,
        columns: Vec<LogicalColumn>,
        residual: Option<BoundExpr>,
        request: zyron_common::ForeignRequest,
        cost: PlanCost,
    },

    /// Predicate delete on a lake table. The predicate is recorded in the
    /// table's log rather than applied row by row: files it fully covers
    /// are dropped whole with no data IO, files it may match carry it
    /// until a later optimize rewrites them. None deletes every row.
    LakeDelete {
        table_id: TableId,
        predicate: Option<zyron_lake::LakePredicate>,
        /// The same row-selecting predicate in bound form, so referential
        /// enforcement can gather exactly the rows the delete removes
        bound_predicate: Option<BoundExpr>,
        /// The predicate's SQL text, recorded in the manifest
        sql: String,
        cost: PlanCost,
    },

    /// Update of a lake table: the matching rows are read through the
    /// child scan, the assignments produce their new images, and one
    /// commit removes the old rows and adds the new ones. The child
    /// projects every column so the new image is complete.
    LakeUpdate {
        table_id: TableId,
        assignments: Vec<crate::binder::BoundAssignment>,
        check_constraints: Vec<BoundExpr>,
        /// Lowered form of the child's predicate, what removes the old
        /// rows. None updates every row.
        predicate: Option<zyron_lake::LakePredicate>,
        sql: String,
        child: Box<PhysicalPlan>,
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
        /// Which way the index is walked. Backward serves an ORDER BY that
        /// runs opposite to the index's declared key direction.
        scan_direction: ScanDirection,
        /// The ordering this scan is relied on to produce, set when a Sort
        /// above it was removed because the index already yields that order.
        /// The executor rebuilds the Sort if it has to fall back to a path
        /// that does not read the index in order, so losing the index at
        /// runtime costs speed rather than correctness.
        ordered_by: Option<Vec<crate::binder::BoundOrderBy>>,
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
        /// Table index stamped on output columns when this projection
        /// materializes a derived table (view or FROM-subquery), so the
        /// enclosing query addresses them by `(table_idx, ordinal)`.
        output_table_idx: Option<usize>,
    },

    /// Nested loop join.
    NestedLoopJoin {
        left: Box<PhysicalPlan>,
        right: Box<PhysicalPlan>,
        join_type: JoinType,
        condition: Option<BoundExpr>,
        cost: PlanCost,
    },

    /// LATERAL join. The right side is a subquery executed once per left row
    /// with the referenced left columns bound as parameters. Held as a bound
    /// select so the executor parameterizes and plans it against the current
    /// outer row. join_type is Inner/Cross (drop left rows with no match) or
    /// Left (NULL-extend); condition is the optional ON predicate.
    LateralJoin {
        left: Box<PhysicalPlan>,
        /// Boxed because a `BoundSelect` is by far the widest thing any plan
        /// node carries, and inlining it here would set the size of every
        /// `PhysicalPlan` value in the planner and the executor
        subquery: Box<crate::binder::BoundSelect>,
        subquery_table_idx: usize,
        join_type: JoinType,
        condition: Option<BoundExpr>,
        left_schema: Vec<LogicalColumn>,
        right_schema: Vec<LogicalColumn>,
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

    /// Row locking for SELECT ... FOR UPDATE/SHARE. The executor builds its
    /// child through the locator-tracking scan path so every row carries a
    /// storage locator to lock.
    LockRows {
        table_id: TableId,
        mode: crate::binder::RowLockMode,
        wait: crate::binder::RowLockWait,
        /// literal LIMIT plus OFFSET, locking stops after this many rows
        cap: Option<u64>,
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
        /// Bound default expressions for omitted columns that carry a DEFAULT.
        column_defaults: Vec<(ColumnId, crate::binder::BoundExpr)>,
        /// CHECK constraint predicates (bound at table_idx 0) to enforce per row.
        check_constraints: Vec<crate::binder::BoundExpr>,
        /// Data-quality expectations (bound at table_idx 0) applied per row.
        expectations: Vec<crate::binder::BoundExpectation>,
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
        /// CHECK constraint predicates (bound at table_idx 0) to enforce per row.
        check_constraints: Vec<crate::binder::BoundExpr>,
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
        /// Time travel qualifier. A hit resolves to a row in the store the
        /// row lives in, and this is the version that fetch reads at.
        as_of: Option<AsOfTarget>,
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
        /// Time travel qualifier, see FulltextScan
        as_of: Option<AsOfTarget>,
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
        /// Time travel qualifier, see FulltextScan
        as_of: Option<AsOfTarget>,
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
            | PhysicalPlan::LakeScan { cost, .. }
            | PhysicalPlan::ForeignScan { cost, .. }
            | PhysicalPlan::LakeDelete { cost, .. }
            | PhysicalPlan::LakeUpdate { cost, .. }
            | PhysicalPlan::ColumnarMetadataAggregate { cost, .. }
            | PhysicalPlan::IndexScan { cost, .. }
            | PhysicalPlan::Filter { cost, .. }
            | PhysicalPlan::Project { cost, .. }
            | PhysicalPlan::NestedLoopJoin { cost, .. }
            | PhysicalPlan::LateralJoin { cost, .. }
            | PhysicalPlan::HashJoin { cost, .. }
            | PhysicalPlan::MergeJoin { cost, .. }
            | PhysicalPlan::HashAggregate { cost, .. }
            | PhysicalPlan::SortAggregate { cost, .. }
            | PhysicalPlan::Sort { cost, .. }
            | PhysicalPlan::Limit { cost, .. }
            | PhysicalPlan::HashDistinct { cost, .. }
            | PhysicalPlan::LockRows { cost, .. }
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
            | PhysicalPlan::LakeScan { columns, .. }
            | PhysicalPlan::ForeignScan { columns, .. }
            | PhysicalPlan::IndexScan { columns, .. }
            | PhysicalPlan::ParallelSeqScan { columns, .. } => columns.clone(),
            PhysicalPlan::ColumnarMetadataAggregate { schema, .. } => schema.clone(),
            PhysicalPlan::Filter { child, .. } => child.output_schema(),
            PhysicalPlan::Project {
                expressions,
                aliases,
                output_table_idx,
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
                        table_idx: *output_table_idx,
                        column_id: ColumnId(i as u16),
                        name,
                        type_id: expr.type_id(),
                        nullable: expr.nullable(),
                        fractional_digits: expr.fractional_digits(),
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
            PhysicalPlan::LateralJoin {
                left_schema,
                right_schema,
                ..
            } => {
                let mut schema = left_schema.clone();
                schema.extend(right_schema.clone());
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
                        fractional_digits: expr.fractional_digits(),
                    });
                }
                for (i, agg) in aggregates.iter().enumerate() {
                    let idx = group_by.len() + i;
                    // A decimal aggregate keeps its argument's scale, so
                    // the output column compares and renders the value
                    // rather than the raw scaled integer
                    let fractional_digits = if agg.return_type == zyron_common::TypeId::Decimal {
                        agg.args.first().and_then(|a| a.fractional_digits())
                    } else {
                        None
                    };
                    schema.push(LogicalColumn {
                        table_idx: Some(crate::logical::AGGREGATE_TABLE_IDX),
                        column_id: ColumnId(idx as u16),
                        name: agg.function_name.clone(),
                        type_id: agg.return_type,
                        nullable: true,
                        fractional_digits,
                    });
                }
                schema
            }
            PhysicalPlan::Sort { child, .. }
            | PhysicalPlan::Limit { child, .. }
            | PhysicalPlan::HashDistinct { child, .. }
            | PhysicalPlan::LockRows { child, .. }
            | PhysicalPlan::Gather { child, .. }
            | PhysicalPlan::Repartition { child, .. }
            | PhysicalPlan::GapFill { child, .. }
            | PhysicalPlan::Broadcast { child, .. } => child.output_schema(),
            PhysicalPlan::SetOp { left, .. } => left.output_schema(),
            PhysicalPlan::Insert { .. }
            | PhysicalPlan::Update { .. }
            | PhysicalPlan::Delete { .. }
            | PhysicalPlan::LakeDelete { .. }
            | PhysicalPlan::LakeUpdate { .. } => Vec::new(),
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
                // Window outputs are appended after the input columns and
                // addressed by (WINDOW_TABLE_IDX, window index). Using the index
                // (not a positional column_id) keeps the address stable and free
                // of collisions with input column ids when projection pushdown
                // trims the input. Must match rewrite_window_refs in the builder.
                for (i, expr) in window_exprs.iter().enumerate() {
                    let name = window_names
                        .get(i)
                        .cloned()
                        .unwrap_or_else(|| format!("window{}", i));
                    schema.push(LogicalColumn {
                        table_idx: Some(super::logical::WINDOW_TABLE_IDX),
                        column_id: ColumnId(i as u16),
                        name,
                        type_id: expr.type_id(),
                        nullable: true,
                        fractional_digits: expr.fractional_digits(),
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
            | PhysicalPlan::LakeScan { .. }
            | PhysicalPlan::ForeignScan { .. }
            | PhysicalPlan::LakeDelete { .. }
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
            | PhysicalPlan::LockRows { child, .. }
            | PhysicalPlan::Insert { source: child, .. }
            | PhysicalPlan::Update { child, .. }
            | PhysicalPlan::Delete { child, .. }
            | PhysicalPlan::LakeUpdate { child, .. }
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
            // The lateral subquery is planned at execution time, so only the
            // left input contributes child cost here.
            PhysicalPlan::LateralJoin { left, .. } => left.total_cost(),
        };
        PlanCost {
            io_cost: own.io_cost + children_cost.io_cost,
            cpu_cost: own.cpu_cost + children_cost.cpu_cost,
            row_count: own.row_count,
        }
    }
}

#[cfg(test)]
mod plan_width {
    use super::*;

    /// Every construction, move, match arm and `Vec` element in the planner
    /// and the executor pays this width, and the operator tree is built by
    /// recursion, so it is also multiplied by plan depth on the stack. It was
    /// 928 bytes while `LateralJoin` held a `BoundSelect` inline, which alone
    /// is 704. Raising this is a real cost, so it is pinned rather than left
    /// to drift
    #[test]
    fn a_physical_plan_node_stays_narrow() {
        let width = std::mem::size_of::<PhysicalPlan>();
        assert!(
            width <= 384,
            "PhysicalPlan grew to {} bytes, over the 384 byte budget.              Box the widest field of the variant that grew rather than              widening every plan node in the tree",
            width
        );
        assert!(
            std::mem::size_of::<crate::binder::BoundSelect>() > width,
            "BoundSelect is no longer wider than a plan node, so the reason              LateralJoin boxes it should be rechecked"
        );
    }
}

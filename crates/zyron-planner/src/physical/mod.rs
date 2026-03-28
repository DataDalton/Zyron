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
            | PhysicalPlan::Broadcast { cost, .. } => cost,
        }
    }

    /// Returns the output schema of this plan node.
    pub fn output_schema(&self) -> Vec<LogicalColumn> {
        match self {
            PhysicalPlan::SeqScan { columns, .. }
            | PhysicalPlan::IndexScan { columns, .. }
            | PhysicalPlan::ParallelSeqScan { columns, .. } => columns.clone(),
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
                        table_idx: None,
                        column_id: ColumnId(i as u16),
                        name: format!("group{}", i),
                        type_id: expr.type_id(),
                        nullable: expr.nullable(),
                    });
                }
                for (i, agg) in aggregates.iter().enumerate() {
                    let idx = group_by.len() + i;
                    schema.push(LogicalColumn {
                        table_idx: None,
                        column_id: ColumnId(idx as u16),
                        name: agg.function_name.clone(),
                        type_id: agg.return_type,
                        nullable: true,
                    });
                }
                schema
            }
            PhysicalPlan::Sort { child, .. }
            | PhysicalPlan::Limit { child, .. }
            | PhysicalPlan::HashDistinct { child, .. }
            | PhysicalPlan::Gather { child, .. }
            | PhysicalPlan::Repartition { child, .. }
            | PhysicalPlan::Broadcast { child, .. } => child.output_schema(),
            PhysicalPlan::SetOp { left, .. } => left.output_schema(),
            PhysicalPlan::Insert { .. }
            | PhysicalPlan::Update { .. }
            | PhysicalPlan::Delete { .. } => Vec::new(),
            PhysicalPlan::Values { schema, .. } => schema.clone(),
        }
    }

    /// Returns the total cost of this node plus all children.
    pub fn total_cost(&self) -> PlanCost {
        let own = *self.cost();
        let children_cost = match self {
            PhysicalPlan::SeqScan { .. }
            | PhysicalPlan::IndexScan { .. }
            | PhysicalPlan::Values { .. }
            | PhysicalPlan::ParallelSeqScan { .. } => PlanCost::zero(),
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
            | PhysicalPlan::Broadcast { child, .. } => child.total_cost(),
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

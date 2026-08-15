//! Logical plan representation for query planning.
//!
//! Logical plans form a relational algebra tree that describes what data
//! operations to perform without specifying how to execute them.
//! The optimizer transforms logical plans, and the physical planner
//! converts them into executable physical plans.

pub mod builder;

use crate::binder::{BoundAssignment, BoundExpr, BoundOrderBy};
use crate::optimizer::rules::encoding_pushdown::EncodingHint;
use std::sync::Arc;
use zyron_catalog::{ColumnId, TableId};
use zyron_common::TypeId;
use zyron_parser::ast::{JoinType, SetOpType};

// ---------------------------------------------------------------------------
// Logical column
// ---------------------------------------------------------------------------

/// Synthetic table index assigned to every column produced by an Aggregate
/// node's output schema. The post-aggregate projection rewriter creates
/// `ColumnRef`s with this `table_idx` so the executor's resolver matches
/// them to the aggregate's output regardless of the original table layout.
///
/// `usize::MAX` is well outside any real bind-context table index space, so
/// it cannot collide with a base-table or subquery alias.
pub const AGGREGATE_TABLE_IDX: usize = usize::MAX;

/// Synthetic `table_idx` for window-function output columns, appended to the
/// Window node's output schema after the input columns. The physical planner
/// rewrites each `WindowFunction` in the projection to a `ColumnRef` with this
/// `table_idx` and a `column_id` equal to the window's index, so the executor's
/// resolver matches it to the appended output column regardless of how many
/// input columns survived projection pushdown. Distinct from
/// `AGGREGATE_TABLE_IDX` so the two never alias.
pub const WINDOW_TABLE_IDX: usize = usize::MAX - 1;

/// A column in the output schema of a logical plan node.
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalColumn {
    pub table_idx: Option<usize>,
    pub column_id: ColumnId,
    pub name: String,
    /// Logical type. For a TIMESTAMP(p)/TIMESTAMPTZ(p) column this stays the
    /// logical timestamp type; fractional_digits records p so the executor can
    /// pick the i128 picosecond physical buffer for p>6 while keeping the
    /// logical identity for compare/cast/presentation.
    pub type_id: TypeId,
    pub nullable: bool,
    /// Digits after the decimal point: fractional seconds for a TIMESTAMP(p),
    /// scale for a DECIMAL(p,s), None for every other type.
    pub fractional_digits: Option<u8>,
}

// ---------------------------------------------------------------------------
// Time travel target
// ---------------------------------------------------------------------------

/// Target for time travel queries on scan nodes.
#[derive(Debug, Clone, PartialEq)]
pub enum AsOfTarget {
    /// Query table at a specific version number.
    Version(u64),
    /// Query table at a specific timestamp (microseconds since epoch).
    Timestamp(i64),
    /// Query table on a named branch. The executor resolves the name to
    /// the branch's pinned VersionId via BranchManager at scan time
    Branch(String),
}

// ---------------------------------------------------------------------------
// Logical plan
// ---------------------------------------------------------------------------

/// Relational algebra tree for query plans.
#[derive(Debug, Clone, PartialEq)]
pub enum LogicalPlan {
    /// Sequential scan of a base table.
    Scan {
        table_id: TableId,
        table_idx: usize,
        columns: Vec<LogicalColumn>,
        alias: String,
        /// Encoding optimization hints set by the encoding pushdown rule.
        encoding_hints: Option<EncodingHint>,
        /// Time travel target for versioned table scans.
        as_of: Option<AsOfTarget>,
    },

    /// Predicate filter.
    Filter {
        predicate: BoundExpr,
        child: Arc<LogicalPlan>,
    },

    /// Column projection.
    Project {
        expressions: Vec<BoundExpr>,
        aliases: Vec<Option<String>>,
        child: Arc<LogicalPlan>,
        /// When set, the projection's output columns carry this table index so
        /// an enclosing query can address them by `(table_idx, ordinal)`. Used
        /// to relabel a derived table (view or FROM-subquery) under the table
        /// index the binder allocated for it. None for an ordinary final
        /// projection, whose outputs are positional and unaddressable.
        output_table_idx: Option<usize>,
    },

    /// Join two relations.
    Join {
        left: Arc<LogicalPlan>,
        right: Arc<LogicalPlan>,
        join_type: JoinType,
        condition: JoinCondition,
    },

    /// LATERAL join: the right side is a subquery that may reference columns
    /// from the left, so it is executed once per left row with those columns
    /// bound as parameters. The subquery is held as a bound select (not a
    /// LogicalPlan child) because it is planned and parameterized at execution
    /// time against the current outer row. join_type is Inner for a comma or
    /// CROSS JOIN LATERAL and Left for a LEFT JOIN LATERAL; condition is the
    /// optional ON predicate.
    LateralJoin {
        left: Arc<LogicalPlan>,
        subquery: LateralSubquery,
        subquery_table_idx: usize,
        join_type: JoinType,
        condition: Option<BoundExpr>,
    },

    /// Group-by aggregation.
    Aggregate {
        group_by: Vec<BoundExpr>,
        aggregates: Vec<AggregateExpr>,
        child: Arc<LogicalPlan>,
    },

    /// Sort by order-by expressions.
    Sort {
        order_by: Vec<BoundOrderBy>,
        child: Arc<LogicalPlan>,
    },

    /// Limit and/or offset.
    Limit {
        limit: Option<u64>,
        offset: Option<u64>,
        child: Arc<LogicalPlan>,
    },

    /// Distinct elimination.
    Distinct { child: Arc<LogicalPlan> },

    /// Row locking for SELECT ... FOR UPDATE/SHARE. Sits directly above the
    /// locked table's row-producing subtree, below Project, so every row it
    /// sees still carries a storage locator.
    LockRows {
        table_id: TableId,
        mode: crate::binder::RowLockMode,
        wait: crate::binder::RowLockWait,
        /// LIMIT plus OFFSET when both are literal. The nodes between this
        /// one and the Limit preserve row count, so locking stops once this
        /// many rows are emitted. Keeps FOR UPDATE SKIP LOCKED LIMIT n
        /// locking exactly n rows instead of a whole batch
        cap: Option<u64>,
        child: Arc<LogicalPlan>,
    },

    /// Set operations (UNION, INTERSECT, EXCEPT).
    SetOp {
        op: SetOpType,
        all: bool,
        left: Arc<LogicalPlan>,
        right: Arc<LogicalPlan>,
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
        source: Arc<LogicalPlan>,
    },

    /// Inline values (for INSERT ... VALUES or standalone VALUES).
    Values {
        rows: Vec<Vec<BoundExpr>>,
        schema: Vec<LogicalColumn>,
    },

    /// Update rows.
    Update {
        table_id: TableId,
        assignments: Vec<BoundAssignment>,
        /// CHECK constraint predicates (bound at table_idx 0) to enforce per row.
        check_constraints: Vec<crate::binder::BoundExpr>,
        child: Arc<LogicalPlan>,
    },

    /// Delete rows.
    Delete {
        table_id: TableId,
        child: Arc<LogicalPlan>,
    },

    /// Graph algorithm execution over a named graph schema.
    GraphAlgorithm {
        schema_name: String,
        algorithm: String,
        params: Vec<(String, BoundExpr)>,
        output_columns: Vec<LogicalColumn>,
    },

    /// Analytics table-returning function. Resolved by the binder against
    /// the analytics function registry. The executor dispatches by name
    /// to the corresponding zyron-analytics implementation.
    AnalyticsTableFunction {
        function_name: String,
        named_args: Vec<(String, BoundExpr)>,
        positional_args: Vec<BoundExpr>,
        output_columns: Vec<LogicalColumn>,
    },
}

/// Holds a LATERAL subquery's bound plan inside a LogicalPlan node. BoundSelect
/// has no PartialEq (it holds Arc<TableEntry>), and no optimization rule mutates
/// a lateral subquery, so it is invariant under optimization and compares equal.
/// Real plan changes are still detected through the LateralJoin node's left and
/// condition fields, which compare normally.
#[derive(Debug, Clone)]
pub struct LateralSubquery(pub Box<crate::binder::BoundSelect>);

impl PartialEq for LateralSubquery {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// Join condition
// ---------------------------------------------------------------------------

/// Join condition for logical join nodes.
#[derive(Debug, Clone, PartialEq)]
pub enum JoinCondition {
    On(BoundExpr),
    Using(Vec<ColumnId>),
    Natural,
    Cross,
}

// ---------------------------------------------------------------------------
// Aggregate expression
// ---------------------------------------------------------------------------

/// An aggregate expression within an Aggregate node.
#[derive(Debug, Clone, PartialEq)]
pub struct AggregateExpr {
    pub function_name: String,
    pub args: Vec<BoundExpr>,
    pub distinct: bool,
    pub return_type: TypeId,
    /// Set for a user-defined aggregate, carrying the bound state-transition
    /// and final functions. None for built-in aggregates.
    pub uda: Option<Box<crate::binder::BoundUda>>,
}

// ---------------------------------------------------------------------------
// LogicalPlan helpers
// ---------------------------------------------------------------------------

impl LogicalPlan {
    /// Returns the output schema of this plan node.
    pub fn output_schema(&self) -> Vec<LogicalColumn> {
        match self {
            LogicalPlan::Scan { columns, .. } => columns.clone(),
            LogicalPlan::Filter { child, .. } => child.output_schema(),
            LogicalPlan::Project {
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
            LogicalPlan::Join { left, right, .. } => {
                let mut schema = left.output_schema();
                schema.extend(right.output_schema());
                schema
            }
            LogicalPlan::LateralJoin {
                left,
                subquery,
                subquery_table_idx,
                join_type,
                ..
            } => {
                let mut schema = left.output_schema();
                // A LEFT JOIN LATERAL produces NULLs when the subquery yields no
                // rows for an outer row, so its right columns are nullable.
                let force_nullable = matches!(join_type, JoinType::Left | JoinType::Full);
                for (i, col) in subquery.0.output_schema.iter().enumerate() {
                    schema.push(LogicalColumn {
                        table_idx: Some(*subquery_table_idx),
                        column_id: ColumnId(i as u16),
                        name: col.name.clone(),
                        type_id: col.type_id,
                        nullable: col.nullable || force_nullable,
                        fractional_digits: col.fractional_digits,
                    });
                }
                schema
            }
            LogicalPlan::Aggregate {
                group_by,
                aggregates,
                ..
            } => {
                let mut schema = Vec::with_capacity(group_by.len() + aggregates.len());
                for (i, expr) in group_by.iter().enumerate() {
                    schema.push(LogicalColumn {
                        table_idx: None,
                        column_id: ColumnId(i as u16),
                        name: format!("group{}", i),
                        type_id: expr.type_id(),
                        nullable: expr.nullable(),
                        fractional_digits: expr.fractional_digits(),
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
                        // Aggregate-result precision finalized in B5.
                        fractional_digits: None,
                    });
                }
                schema
            }
            LogicalPlan::Sort { child, .. } => child.output_schema(),
            LogicalPlan::Limit { child, .. } => child.output_schema(),
            LogicalPlan::Distinct { child } => child.output_schema(),
            LogicalPlan::LockRows { child, .. } => child.output_schema(),
            LogicalPlan::SetOp { left, .. } => left.output_schema(),
            LogicalPlan::Insert { .. } => Vec::new(),
            LogicalPlan::Values { schema, .. } => schema.clone(),
            LogicalPlan::Update { .. } => Vec::new(),
            LogicalPlan::Delete { .. } => Vec::new(),
            LogicalPlan::GraphAlgorithm { output_columns, .. } => output_columns.clone(),
            LogicalPlan::AnalyticsTableFunction { output_columns, .. } => output_columns.clone(),
        }
    }

    /// Returns all child plan nodes.
    pub fn children(&self) -> Vec<&LogicalPlan> {
        match self {
            LogicalPlan::Scan { .. }
            | LogicalPlan::Values { .. }
            | LogicalPlan::GraphAlgorithm { .. }
            | LogicalPlan::AnalyticsTableFunction { .. } => vec![],
            LogicalPlan::Filter { child, .. }
            | LogicalPlan::Project { child, .. }
            | LogicalPlan::Aggregate { child, .. }
            | LogicalPlan::Sort { child, .. }
            | LogicalPlan::Limit { child, .. }
            | LogicalPlan::Distinct { child }
            | LogicalPlan::LockRows { child, .. }
            | LogicalPlan::Insert { source: child, .. }
            | LogicalPlan::Update { child, .. }
            | LogicalPlan::Delete { child, .. } => vec![child],
            LogicalPlan::Join { left, right, .. } | LogicalPlan::SetOp { left, right, .. } => {
                vec![left, right]
            }
            // The lateral subquery is not a LogicalPlan child; it is planned at
            // execution time, so only the left input is a child here.
            LogicalPlan::LateralJoin { left, .. } => vec![left],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyron_common::TypeId;

    #[test]
    fn test_scan_output_schema() {
        let plan = LogicalPlan::Scan {
            table_id: TableId(1),
            table_idx: 0,
            columns: vec![
                LogicalColumn {
                    table_idx: Some(0),
                    column_id: ColumnId(0),
                    name: "id".to_string(),
                    type_id: TypeId::Int64,
                    nullable: false,
                    fractional_digits: None,
                },
                LogicalColumn {
                    table_idx: Some(0),
                    column_id: ColumnId(1),
                    name: "name".to_string(),
                    type_id: TypeId::Varchar,
                    nullable: true,
                    fractional_digits: None,
                },
            ],
            alias: "users".to_string(),
            encoding_hints: None,
            as_of: None,
        };
        let schema = plan.output_schema();
        assert_eq!(schema.len(), 2);
        assert_eq!(schema[0].name, "id");
        assert_eq!(schema[1].name, "name");
    }

    #[test]
    fn test_filter_preserves_schema() {
        let scan = LogicalPlan::Scan {
            table_id: TableId(1),
            table_idx: 0,
            columns: vec![LogicalColumn {
                table_idx: Some(0),
                column_id: ColumnId(0),
                name: "id".to_string(),
                type_id: TypeId::Int64,
                nullable: false,
                fractional_digits: None,
            }],
            alias: "t".to_string(),
            encoding_hints: None,
            as_of: None,
        };
        let filter = LogicalPlan::Filter {
            predicate: BoundExpr::Literal {
                value: zyron_parser::ast::LiteralValue::Boolean(true),
                type_id: TypeId::Boolean,
            },
            child: Arc::new(scan),
        };
        let schema = filter.output_schema();
        assert_eq!(schema.len(), 1);
        assert_eq!(schema[0].name, "id");
    }

    #[test]
    fn test_join_merges_schemas() {
        let left = LogicalPlan::Scan {
            table_id: TableId(1),
            table_idx: 0,
            columns: vec![LogicalColumn {
                table_idx: Some(0),
                column_id: ColumnId(0),
                name: "a".to_string(),
                type_id: TypeId::Int64,
                nullable: false,
                fractional_digits: None,
            }],
            alias: "l".to_string(),
            encoding_hints: None,
            as_of: None,
        };
        let right = LogicalPlan::Scan {
            table_id: TableId(2),
            table_idx: 1,
            columns: vec![LogicalColumn {
                table_idx: Some(1),
                column_id: ColumnId(0),
                name: "b".to_string(),
                type_id: TypeId::Int64,
                nullable: false,
                fractional_digits: None,
            }],
            alias: "r".to_string(),
            encoding_hints: None,
            as_of: None,
        };
        let join = LogicalPlan::Join {
            left: Arc::new(left),
            right: Arc::new(right),
            join_type: JoinType::Inner,
            condition: JoinCondition::Cross,
        };
        let schema = join.output_schema();
        assert_eq!(schema.len(), 2);
        assert_eq!(schema[0].name, "a");
        assert_eq!(schema[1].name, "b");
    }

    #[test]
    fn test_children_count() {
        let scan = LogicalPlan::Scan {
            table_id: TableId(1),
            table_idx: 0,
            columns: vec![],
            alias: "t".to_string(),
            encoding_hints: None,
            as_of: None,
        };
        assert_eq!(scan.children().len(), 0);

        let filter = LogicalPlan::Filter {
            predicate: BoundExpr::Literal {
                value: zyron_parser::ast::LiteralValue::Boolean(true),
                type_id: TypeId::Boolean,
            },
            child: Arc::new(scan),
        };
        assert_eq!(filter.children().len(), 1);
    }
}

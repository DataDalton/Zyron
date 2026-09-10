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

/// Synthetic `table_idx` the head branch's projection takes when set
/// operations follow it. A compound query's trailing ORDER BY addresses the
/// combined result like a derived table: the deferred sort keys become
/// `ColumnRef`s with this `table_idx` and a `column_id` equal to the output
/// position, and the projection's output schema carries the same identity.
/// Distinct from `AGGREGATE_TABLE_IDX` and `WINDOW_TABLE_IDX` so the three
/// never alias.
pub const SET_OP_TABLE_IDX: usize = usize::MAX - 2;

/// Synthetic `table_idx` a lambda's parameter binds under.
///
/// `array_transform(arr, x -> x * 2)` names one element, which is not a
/// column of any relation the query reads. Binding it as a column of a
/// one-column pseudo relation lets the executor evaluate the body over a
/// batch of elements through the same evaluator every other expression uses.
/// Distinct from the three above so the four never alias.
pub const LAMBDA_TABLE_IDX: usize = usize::MAX - 3;

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
        /// Stored generated columns (bound at table_idx 0) computed per row.
        generated_columns: Vec<crate::binder::BoundGeneratedColumn>,
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
        /// STORED generated columns, recomputed from the updated row image so
        /// a generated value never outlives the columns it derives from.
        generated_columns: Vec<crate::binder::BoundGeneratedColumn>,
        child: Arc<LogicalPlan>,
    },

    /// Delete rows.
    Delete {
        table_id: TableId,
        child: Arc<LogicalPlan>,
    },

    /// DML against a view with an INSTEAD OF trigger. `source` produces one
    /// parameter row per affected row; the executor runs the trigger body per
    /// row instead of writing storage. `param_map` routes each trigger
    /// parameter position to a source column, None filling NULL.
    ViewTriggerWrite {
        view_id: u32,
        event: u8,
        param_map: Vec<Option<usize>>,
        source: Arc<LogicalPlan>,
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

    /// Expands each input row into zero or more output rows.
    ///
    /// UNNEST walks arrays, FLATTEN walks a VARIANT document and UNPIVOT
    /// walks a fixed list of column groups. All three repeat the input row's
    /// columns beside what they produce, so one node serves them and the
    /// executor builds an output batch from a repeat vector rather than
    /// copying a row per produced row.
    ExpandRows {
        child: Arc<LogicalPlan>,
        /// Boxed for the reason `AsofJoin`'s match is: the widest spec arm
        /// would otherwise set the width of every node in the tree
        spec: Box<ExpandSpec>,
        /// The child column each carried output column comes from, in output
        /// order, ahead of the produced columns.
        ///
        /// Carried by identity rather than by position, because projection
        /// pushdown prunes the child and every position after a pruned
        /// column would shift. Empty when the input is a synthesized one-row
        /// relation with nothing to carry
        carry: Vec<crate::binder::ColumnRef>,
        /// Every column the node outputs, the carried ones then the produced
        /// ones, already carrying the identity an enclosing query addresses
        /// them by
        output_columns: Vec<LogicalColumn>,
        /// True when an input row that produced no output row is still
        /// emitted once with the produced columns null. LEFT JOIN LATERAL
        /// asks for this, and so does FLATTEN's `outer => true`
        outer_input: bool,
    },

    /// Joins each left row to the nearest right row in one direction along
    /// an ordered column, within the equality group the ON clause names.
    ///
    /// Both inputs arrive sorted by (equality keys, match column); the
    /// physical builder elides a sort whose input already holds that order.
    AsofJoin {
        left: Arc<LogicalPlan>,
        right: Arc<LogicalPlan>,
        /// Boxed so the match expressions and the equality keys do not set
        /// the width of every node in the tree. The optimizer rebuilds the
        /// whole logical plan by value on each rule pass, so this variant's
        /// width is paid by every node of every rebuild
        match_on: Box<AsofMatchOn>,
    },
}

/// What one `AsofJoin` matches on.
#[derive(Debug, Clone, PartialEq)]
pub struct AsofMatchOn {
    /// The ON clause's equalities, as (left side, right side) pairs
    pub equality_keys: Vec<(BoundExpr, BoundExpr)>,
    /// The two sides of the match condition's inequality
    pub match_left: BoundExpr,
    pub match_right: BoundExpr,
    pub direction: AsofDirection,
    /// How far a match may reach, as the bound written after AND in the
    /// match condition. None leaves the reach unbounded
    pub tolerance: Option<AsofTolerance>,
    /// Inner drops an unmatched left row, Left keeps it with the right
    /// columns null
    pub join_type: JoinType,
}

/// What one `ExpandRows` node expands.
#[derive(Debug, Clone, PartialEq)]
pub enum ExpandSpec {
    /// One column per array, zipped to the longest with the shorter padded
    /// with NULL, plus a BIGINT position column when `with_ordinality`
    Unnest {
        arrays: Vec<BoundExpr>,
        with_ordinality: bool,
    },
    /// The six columns seq, key, path, index, value and this, one row per
    /// member the walk reaches
    Flatten {
        document: BoundExpr,
        /// The document position the walk starts from. None starts at the
        /// root
        path: Option<String>,
        /// True walks nested arrays and objects depth first
        recursive: bool,
    },
    /// One row per group, carrying the group's label and its values
    Unpivot {
        /// One entry per group, each holding one expression per value column
        groups: Vec<Vec<BoundExpr>>,
        /// The label each group takes in the name column
        labels: Vec<BoundExpr>,
        /// False drops a group whose every value is null
        include_nulls: bool,
    },
}

/// How far back or forward an ASOF match may reach.
///
/// The bound is compared against the distance between the two match columns,
/// so a left row whose nearest right row lies further away than this is left
/// unmatched.
#[derive(Debug, Clone, PartialEq)]
pub struct AsofTolerance {
    pub bound: BoundExpr,
    /// True when the bound was written with `<=`, so a distance exactly
    /// equal to it still matches
    pub inclusive: bool,
}

/// Which way an ASOF join reaches for its match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsofDirection {
    /// `left >= right`, the greatest right value at or below the left value
    Backward,
    /// `left > right`, the greatest right value strictly below it
    BackwardStrict,
    /// `left <= right`, the least right value at or above the left value
    Forward,
    /// `left < right`, the least right value strictly above it
    ForwardStrict,
}

impl AsofDirection {
    /// True when the match reaches toward smaller right values.
    pub fn is_backward(self) -> bool {
        matches!(
            self,
            AsofDirection::Backward | AsofDirection::BackwardStrict
        )
    }

    /// True when a right value equal to the left value matches.
    pub fn allows_equal(self) -> bool {
        matches!(self, AsofDirection::Backward | AsofDirection::Forward)
    }

    /// The operator the match condition was written with.
    pub fn operator(self) -> &'static str {
        match self {
            AsofDirection::Backward => ">=",
            AsofDirection::BackwardStrict => ">",
            AsofDirection::Forward => "<=",
            AsofDirection::ForwardStrict => "<",
        }
    }
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
                    // A decimal aggregate keeps its argument's scale, so
                    // the output column compares and renders the value
                    // rather than the raw scaled integer
                    let fractional_digits = if agg.return_type == zyron_common::TypeId::Decimal {
                        agg.args.first().and_then(|a| a.fractional_digits())
                    } else {
                        None
                    };
                    schema.push(LogicalColumn {
                        table_idx: None,
                        column_id: ColumnId(idx as u16),
                        name: agg.function_name.clone(),
                        type_id: agg.return_type,
                        nullable: true,
                        fractional_digits,
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
            LogicalPlan::ViewTriggerWrite { .. } => Vec::new(),
            LogicalPlan::GraphAlgorithm { output_columns, .. } => output_columns.clone(),
            LogicalPlan::AnalyticsTableFunction { output_columns, .. } => output_columns.clone(),
            LogicalPlan::ExpandRows { output_columns, .. } => output_columns.clone(),
            LogicalPlan::AsofJoin {
                left,
                right,
                match_on,
            } => {
                let join_type = &match_on.join_type;
                let mut schema = left.output_schema();
                // An unmatched left row carries NULLs on the right side, so
                // the right columns are nullable under the LEFT form
                let force_nullable = matches!(join_type, JoinType::Left);
                for col in right.output_schema() {
                    schema.push(LogicalColumn {
                        nullable: col.nullable || force_nullable,
                        ..col
                    });
                }
                schema
            }
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
            | LogicalPlan::ViewTriggerWrite { source: child, .. }
            | LogicalPlan::Update { child, .. }
            | LogicalPlan::Delete { child, .. }
            | LogicalPlan::ExpandRows { child, .. } => vec![child],
            LogicalPlan::Join { left, right, .. }
            | LogicalPlan::SetOp { left, right, .. }
            | LogicalPlan::AsofJoin { left, right, .. } => {
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

    /// The logical tree is rebuilt by value on every optimizer rule pass, so
    /// the widest variant's width is paid by every node of every rebuild and
    /// by every Arc allocation the rebuild makes. It reached 448 bytes while
    /// `AsofJoin` held three `BoundExpr` inline, which is wider than the
    /// physical node it becomes. Raising this is a real cost, so it is pinned
    /// rather than left to drift
    #[test]
    fn a_logical_plan_node_stays_narrow() {
        let width = std::mem::size_of::<LogicalPlan>();
        assert!(
            width <= 256,
            "LogicalPlan grew to {width} bytes, over the 256 byte budget.              Box the widest field of the variant that grew rather than              widening every plan node in the tree"
        );
    }
}

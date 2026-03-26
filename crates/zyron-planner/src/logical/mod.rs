//! Logical plan representation for query planning.
//!
//! Logical plans form a relational algebra tree that describes what data
//! operations to perform without specifying how to execute them.
//! The optimizer transforms logical plans, and the physical planner
//! converts them into executable physical plans.

pub mod builder;

use crate::binder::{BoundAssignment, BoundExpr, BoundOrderBy};
use crate::optimizer::rules::encoding_pushdown::EncodingHint;
use zyron_catalog::{ColumnId, TableId};
use zyron_common::TypeId;
use zyron_parser::ast::{JoinType, SetOpType};

// ---------------------------------------------------------------------------
// Logical column
// ---------------------------------------------------------------------------

/// A column in the output schema of a logical plan node.
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalColumn {
    pub table_idx: Option<usize>,
    pub column_id: ColumnId,
    pub name: String,
    pub type_id: TypeId,
    pub nullable: bool,
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
    },

    /// Predicate filter.
    Filter {
        predicate: BoundExpr,
        child: Box<LogicalPlan>,
    },

    /// Column projection.
    Project {
        expressions: Vec<BoundExpr>,
        aliases: Vec<Option<String>>,
        child: Box<LogicalPlan>,
    },

    /// Join two relations.
    Join {
        left: Box<LogicalPlan>,
        right: Box<LogicalPlan>,
        join_type: JoinType,
        condition: JoinCondition,
    },

    /// Group-by aggregation.
    Aggregate {
        group_by: Vec<BoundExpr>,
        aggregates: Vec<AggregateExpr>,
        child: Box<LogicalPlan>,
    },

    /// Sort by order-by expressions.
    Sort {
        order_by: Vec<BoundOrderBy>,
        child: Box<LogicalPlan>,
    },

    /// Limit and/or offset.
    Limit {
        limit: Option<u64>,
        offset: Option<u64>,
        child: Box<LogicalPlan>,
    },

    /// Distinct elimination.
    Distinct { child: Box<LogicalPlan> },

    /// Set operations (UNION, INTERSECT, EXCEPT).
    SetOp {
        op: SetOpType,
        all: bool,
        left: Box<LogicalPlan>,
        right: Box<LogicalPlan>,
    },

    /// Insert rows into a table.
    Insert {
        table_id: TableId,
        target_columns: Vec<ColumnId>,
        source: Box<LogicalPlan>,
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
        child: Box<LogicalPlan>,
    },

    /// Delete rows.
    Delete {
        table_id: TableId,
        child: Box<LogicalPlan>,
    },
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
            LogicalPlan::Join { left, right, .. } => {
                let mut schema = left.output_schema();
                schema.extend(right.output_schema());
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
            LogicalPlan::Sort { child, .. } => child.output_schema(),
            LogicalPlan::Limit { child, .. } => child.output_schema(),
            LogicalPlan::Distinct { child } => child.output_schema(),
            LogicalPlan::SetOp { left, .. } => left.output_schema(),
            LogicalPlan::Insert { .. } => Vec::new(),
            LogicalPlan::Values { schema, .. } => schema.clone(),
            LogicalPlan::Update { .. } => Vec::new(),
            LogicalPlan::Delete { .. } => Vec::new(),
        }
    }

    /// Returns all child plan nodes.
    pub fn children(&self) -> Vec<&LogicalPlan> {
        match self {
            LogicalPlan::Scan { .. } | LogicalPlan::Values { .. } => vec![],
            LogicalPlan::Filter { child, .. }
            | LogicalPlan::Project { child, .. }
            | LogicalPlan::Aggregate { child, .. }
            | LogicalPlan::Sort { child, .. }
            | LogicalPlan::Limit { child, .. }
            | LogicalPlan::Distinct { child }
            | LogicalPlan::Insert { source: child, .. }
            | LogicalPlan::Update { child, .. }
            | LogicalPlan::Delete { child, .. } => vec![child],
            LogicalPlan::Join { left, right, .. } | LogicalPlan::SetOp { left, right, .. } => {
                vec![left, right]
            }
        }
    }

    /// Returns mutable references to all child plan nodes.
    pub fn children_mut(&mut self) -> Vec<&mut LogicalPlan> {
        match self {
            LogicalPlan::Scan { .. } | LogicalPlan::Values { .. } => vec![],
            LogicalPlan::Filter { child, .. }
            | LogicalPlan::Project { child, .. }
            | LogicalPlan::Aggregate { child, .. }
            | LogicalPlan::Sort { child, .. }
            | LogicalPlan::Limit { child, .. }
            | LogicalPlan::Distinct { child }
            | LogicalPlan::Insert { source: child, .. }
            | LogicalPlan::Update { child, .. }
            | LogicalPlan::Delete { child, .. } => vec![child],
            LogicalPlan::Join { left, right, .. } | LogicalPlan::SetOp { left, right, .. } => {
                vec![left, right]
            }
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
                },
                LogicalColumn {
                    table_idx: Some(0),
                    column_id: ColumnId(1),
                    name: "name".to_string(),
                    type_id: TypeId::Varchar,
                    nullable: true,
                },
            ],
            alias: "users".to_string(),
            encoding_hints: None,
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
            }],
            alias: "t".to_string(),
            encoding_hints: None,
        };
        let filter = LogicalPlan::Filter {
            predicate: BoundExpr::Literal {
                value: zyron_parser::ast::LiteralValue::Boolean(true),
                type_id: TypeId::Boolean,
            },
            child: Box::new(scan),
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
            }],
            alias: "l".to_string(),
            encoding_hints: None,
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
            }],
            alias: "r".to_string(),
            encoding_hints: None,
        };
        let join = LogicalPlan::Join {
            left: Box::new(left),
            right: Box::new(right),
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
        };
        assert_eq!(scan.children().len(), 0);

        let filter = LogicalPlan::Filter {
            predicate: BoundExpr::Literal {
                value: zyron_parser::ast::LiteralValue::Boolean(true),
                type_id: TypeId::Boolean,
            },
            child: Box::new(scan),
        };
        assert_eq!(filter.children().len(), 1);
    }
}

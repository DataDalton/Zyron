//! Converts bound AST into logical plan trees.
//!
//! The builder takes a BoundStatement and produces a LogicalPlan tree
//! following standard relational algebra construction:
//! FROM -> Filter -> Aggregate -> Having -> Project -> Distinct -> Sort -> Limit

use crate::binder::*;
use crate::logical::*;
use zyron_catalog::ColumnId;
use zyron_common::{Result, TypeId};
use zyron_parser::ast::LiteralValue;

/// Converts a bound statement into a logical plan tree.
pub fn build_logical_plan(bound: &BoundStatement) -> Result<LogicalPlan> {
    match bound {
        BoundStatement::Select(select) => build_select_plan(select),
        BoundStatement::Insert(insert) => build_insert_plan(insert),
        BoundStatement::Update(update) => build_update_plan(update),
        BoundStatement::Delete(delete) => build_delete_plan(delete),
    }
}

// ---------------------------------------------------------------------------
// SELECT plan construction
// ---------------------------------------------------------------------------

fn build_select_plan(select: &BoundSelect) -> Result<LogicalPlan> {
    // 1. FROM clause -> base plan (scans and joins)
    let mut plan = if select.from.is_empty() {
        // SELECT without FROM (e.g., SELECT 1+1)
        LogicalPlan::Values {
            rows: vec![vec![BoundExpr::Literal {
                value: LiteralValue::Null,
                type_id: TypeId::Null,
            }]],
            schema: vec![LogicalColumn {
                table_idx: None,
                column_id: ColumnId(0),
                name: "".to_string(),
                type_id: TypeId::Null,
                nullable: true,
            }],
        }
    } else {
        let mut from_plan: Option<LogicalPlan> = None;
        for item in &select.from {
            let item_plan = build_from_item(item)?;
            from_plan = Some(match from_plan {
                None => item_plan,
                Some(left) => LogicalPlan::Join {
                    left: Box::new(left),
                    right: Box::new(item_plan),
                    join_type: zyron_parser::ast::JoinType::Cross,
                    condition: JoinCondition::Cross,
                },
            });
        }
        from_plan.unwrap()
    };

    // 2. WHERE -> Filter
    if let Some(predicate) = &select.where_clause {
        plan = LogicalPlan::Filter {
            predicate: predicate.clone(),
            child: Box::new(plan),
        };
    }

    // 3. GROUP BY + aggregates -> Aggregate
    let (has_aggregates, aggregates) = extract_aggregates(&select.projections);
    if !select.group_by.is_empty() || has_aggregates {
        plan = LogicalPlan::Aggregate {
            group_by: select.group_by.clone(),
            aggregates,
            child: Box::new(plan),
        };
    }

    // 4. HAVING -> Filter on top of Aggregate
    if let Some(having) = &select.having {
        plan = LogicalPlan::Filter {
            predicate: having.clone(),
            child: Box::new(plan),
        };
    }

    // 5. SELECT -> Project
    let (expressions, aliases) = build_projection_list(&select.projections);
    if !expressions.is_empty() {
        plan = LogicalPlan::Project {
            expressions,
            aliases,
            child: Box::new(plan),
        };
    }

    // 6. DISTINCT -> Distinct
    if select.distinct {
        plan = LogicalPlan::Distinct {
            child: Box::new(plan),
        };
    }

    // 7. ORDER BY -> Sort
    if !select.order_by.is_empty() {
        plan = LogicalPlan::Sort {
            order_by: select.order_by.clone(),
            child: Box::new(plan),
        };
    }

    // 8. LIMIT/OFFSET -> Limit
    let limit_val = extract_u64_literal(&select.limit);
    let offset_val = extract_u64_literal(&select.offset);
    if limit_val.is_some() || offset_val.is_some() {
        plan = LogicalPlan::Limit {
            limit: limit_val,
            offset: offset_val,
            child: Box::new(plan),
        };
    }

    // 9. Set operations
    for set_op in &select.set_ops {
        let right_plan = build_select_plan(&set_op.right)?;
        plan = LogicalPlan::SetOp {
            op: set_op.op,
            all: set_op.all,
            left: Box::new(plan),
            right: Box::new(right_plan),
        };
    }

    Ok(plan)
}

// ---------------------------------------------------------------------------
// FROM item construction
// ---------------------------------------------------------------------------

fn build_from_item(item: &BoundFromItem) -> Result<LogicalPlan> {
    match item {
        BoundFromItem::BaseTable {
            table_idx,
            table_id,
            entry,
        } => {
            let columns: Vec<LogicalColumn> = entry
                .columns
                .iter()
                .map(|c| LogicalColumn {
                    table_idx: Some(*table_idx),
                    column_id: c.id,
                    name: c.name.clone(),
                    type_id: c.type_id,
                    nullable: c.nullable,
                })
                .collect();
            Ok(LogicalPlan::Scan {
                table_id: *table_id,
                table_idx: *table_idx,
                columns,
                alias: entry.name.clone(),
                encoding_hints: None,
                as_of: None,
            })
        }
        BoundFromItem::Join {
            left,
            join_type,
            right,
            condition,
        } => {
            let left_plan = build_from_item(left)?;
            let right_plan = build_from_item(right)?;
            let join_condition = match condition {
                BoundJoinCondition::On(expr) => JoinCondition::On(expr.clone()),
                BoundJoinCondition::Using(cols) => JoinCondition::Using(cols.clone()),
                BoundJoinCondition::Natural => JoinCondition::Natural,
                BoundJoinCondition::None => JoinCondition::Cross,
            };
            Ok(LogicalPlan::Join {
                left: Box::new(left_plan),
                right: Box::new(right_plan),
                join_type: *join_type,
                condition: join_condition,
            })
        }
        BoundFromItem::Subquery { query, .. } => build_select_plan(query),
        BoundFromItem::GraphQuery {
            schema_name,
            algorithm,
            params,
            output_columns,
            ..
        } => Ok(LogicalPlan::GraphAlgorithm {
            schema_name: schema_name.clone(),
            algorithm: algorithm.clone(),
            params: params.clone(),
            output_columns: output_columns.clone(),
        }),
    }
}

// ---------------------------------------------------------------------------
// Aggregate extraction
// ---------------------------------------------------------------------------

/// Walks projection items to find aggregate functions.
/// Returns (has_any_aggregates, list_of_aggregate_exprs).
fn extract_aggregates(projections: &[BoundSelectItem]) -> (bool, Vec<AggregateExpr>) {
    let mut aggregates = Vec::new();
    let mut has_agg = false;

    for item in projections {
        if let BoundSelectItem::Expr(expr, _) = item {
            collect_aggregates_from_expr(expr, &mut aggregates, &mut has_agg);
        }
    }

    (has_agg, aggregates)
}

fn collect_aggregates_from_expr(
    expr: &BoundExpr,
    out: &mut Vec<AggregateExpr>,
    has_agg: &mut bool,
) {
    match expr {
        BoundExpr::AggregateFunction {
            name,
            args,
            distinct,
            return_type,
        } => {
            *has_agg = true;
            out.push(AggregateExpr {
                function_name: name.clone(),
                args: args.clone(),
                distinct: *distinct,
                return_type: *return_type,
            });
        }
        BoundExpr::BinaryOp { left, right, .. } => {
            collect_aggregates_from_expr(left, out, has_agg);
            collect_aggregates_from_expr(right, out, has_agg);
        }
        BoundExpr::UnaryOp { expr, .. } => {
            collect_aggregates_from_expr(expr, out, has_agg);
        }
        BoundExpr::Function { args, .. } => {
            for arg in args {
                collect_aggregates_from_expr(arg, out, has_agg);
            }
        }
        BoundExpr::Nested(inner) => {
            collect_aggregates_from_expr(inner, out, has_agg);
        }
        BoundExpr::Case {
            operand,
            conditions,
            else_result,
            ..
        } => {
            if let Some(op) = operand {
                collect_aggregates_from_expr(op, out, has_agg);
            }
            for wc in conditions {
                collect_aggregates_from_expr(&wc.condition, out, has_agg);
                collect_aggregates_from_expr(&wc.result, out, has_agg);
            }
            if let Some(e) = else_result {
                collect_aggregates_from_expr(e, out, has_agg);
            }
        }
        BoundExpr::Cast { expr, .. } => {
            collect_aggregates_from_expr(expr, out, has_agg);
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Projection list construction
// ---------------------------------------------------------------------------

fn build_projection_list(projections: &[BoundSelectItem]) -> (Vec<BoundExpr>, Vec<Option<String>>) {
    let mut expressions = Vec::new();
    let mut aliases = Vec::new();

    for item in projections {
        match item {
            BoundSelectItem::Expr(expr, alias) => {
                expressions.push(expr.clone());
                aliases.push(alias.clone());
            }
            BoundSelectItem::Wildcard | BoundSelectItem::AllColumns(_) => {
                // Wildcards are expanded during binding into the output schema.
                // At plan level, they become pass-through (no explicit Project needed).
            }
        }
    }

    (expressions, aliases)
}

// ---------------------------------------------------------------------------
// DML plan construction
// ---------------------------------------------------------------------------

fn build_insert_plan(insert: &BoundInsert) -> Result<LogicalPlan> {
    let source = match &insert.source {
        BoundInsertSource::Values(rows) => {
            let schema: Vec<LogicalColumn> = insert
                .table_entry
                .columns
                .iter()
                .enumerate()
                .filter(|(i, _)| insert.target_columns.contains(&ColumnId(*i as u16)))
                .map(|(_i, c)| LogicalColumn {
                    table_idx: None,
                    column_id: c.id,
                    name: c.name.clone(),
                    type_id: c.type_id,
                    nullable: c.nullable,
                })
                .collect();
            LogicalPlan::Values {
                rows: rows.clone(),
                schema,
            }
        }
        BoundInsertSource::Query(query) => build_select_plan(query)?,
    };

    Ok(LogicalPlan::Insert {
        table_id: insert.table_id,
        target_columns: insert.target_columns.clone(),
        source: Box::new(source),
    })
}

fn build_update_plan(update: &BoundUpdate) -> Result<LogicalPlan> {
    // Scan the target table
    let columns: Vec<LogicalColumn> = update
        .table_entry
        .columns
        .iter()
        .map(|c| LogicalColumn {
            table_idx: None,
            column_id: c.id,
            name: c.name.clone(),
            type_id: c.type_id,
            nullable: c.nullable,
        })
        .collect();

    let mut plan = LogicalPlan::Scan {
        table_id: update.table_id,
        table_idx: 0,
        columns,
        alias: update.table_entry.name.clone(),
        encoding_hints: None,
        as_of: None,
    };

    // Apply WHERE filter
    if let Some(predicate) = &update.where_clause {
        plan = LogicalPlan::Filter {
            predicate: predicate.clone(),
            child: Box::new(plan),
        };
    }

    Ok(LogicalPlan::Update {
        table_id: update.table_id,
        assignments: update.assignments.clone(),
        child: Box::new(plan),
    })
}

fn build_delete_plan(delete: &BoundDelete) -> Result<LogicalPlan> {
    let columns: Vec<LogicalColumn> = delete
        .table_entry
        .columns
        .iter()
        .map(|c| LogicalColumn {
            table_idx: None,
            column_id: c.id,
            name: c.name.clone(),
            type_id: c.type_id,
            nullable: c.nullable,
        })
        .collect();

    let mut plan = LogicalPlan::Scan {
        table_id: delete.table_id,
        table_idx: 0,
        columns,
        alias: delete.table_entry.name.clone(),
        encoding_hints: None,
        as_of: None,
    };

    if let Some(predicate) = &delete.where_clause {
        plan = LogicalPlan::Filter {
            predicate: predicate.clone(),
            child: Box::new(plan),
        };
    }

    Ok(LogicalPlan::Delete {
        table_id: delete.table_id,
        child: Box::new(plan),
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extracts a u64 from a bound literal expression (for LIMIT/OFFSET).
fn extract_u64_literal(expr: &Option<BoundExpr>) -> Option<u64> {
    match expr {
        Some(BoundExpr::Literal {
            value: LiteralValue::Integer(n),
            ..
        }) => Some(*n as u64),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyron_parser::ast::LiteralValue;

    #[test]
    fn test_extract_u64_literal() {
        let expr = Some(BoundExpr::Literal {
            value: LiteralValue::Integer(10),
            type_id: TypeId::Int64,
        });
        assert_eq!(extract_u64_literal(&expr), Some(10));

        let none_expr: Option<BoundExpr> = None;
        assert_eq!(extract_u64_literal(&none_expr), None);
    }

    #[test]
    fn test_extract_aggregates_finds_count() {
        let projections = vec![BoundSelectItem::Expr(
            BoundExpr::AggregateFunction {
                name: "count".to_string(),
                args: vec![BoundExpr::Literal {
                    value: LiteralValue::Integer(1),
                    type_id: TypeId::Int64,
                }],
                distinct: false,
                return_type: TypeId::Int64,
            },
            Some("cnt".to_string()),
        )];
        let (has_agg, aggs) = extract_aggregates(&projections);
        assert!(has_agg);
        assert_eq!(aggs.len(), 1);
        assert_eq!(aggs[0].function_name, "count");
    }

    #[test]
    fn test_extract_aggregates_no_aggregates() {
        let projections = vec![BoundSelectItem::Expr(
            BoundExpr::Literal {
                value: LiteralValue::Integer(1),
                type_id: TypeId::Int64,
            },
            None,
        )];
        let (has_agg, aggs) = extract_aggregates(&projections);
        assert!(!has_agg);
        assert!(aggs.is_empty());
    }
}

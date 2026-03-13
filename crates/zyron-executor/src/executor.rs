//! Executor driver that converts a PhysicalPlan tree into an Operator tree
//! and drains it to produce result batches.

use std::sync::Arc;

use zyron_common::Result;
use zyron_planner::physical::PhysicalPlan;

use crate::batch::DataBatch;
use crate::context::ExecutionContext;
use crate::operator::Operator;
use crate::operator::aggregate::{HashAggregateOperator, SortAggregateOperator};
use crate::operator::distinct::HashDistinctOperator;
use crate::operator::filter::FilterOperator;
use crate::operator::join::{HashJoinOperator, MergeJoinOperator, NestedLoopJoinOperator};
use crate::operator::limit::LimitOperator;
use crate::operator::modify::{DeleteOperator, InsertOperator, UpdateOperator, ValuesOperator};
use crate::operator::project::ProjectOperator;
use crate::operator::scan::{IndexScanOperator, SeqScanOperator};
use crate::operator::setop::SetOpOperator;
use crate::operator::sort::SortOperator;

/// Recursively converts a PhysicalPlan into an executable Operator tree.
fn build_operator_tree(
    plan: PhysicalPlan,
    ctx: &Arc<ExecutionContext>,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Box<dyn Operator>>> + Send + '_>> {
    Box::pin(async move {
        let result: Result<Box<dyn Operator>> = match plan {
            PhysicalPlan::SeqScan {
                table_id,
                columns,
                predicate,
                ..
            } => {
                let op =
                    SeqScanOperator::new(ctx.clone(), table_id, columns, predicate, false).await?;
                Ok(Box::new(op))
            }

            PhysicalPlan::IndexScan {
                table_id,
                columns,
                predicate,
                remaining_predicate,
                ..
            } => {
                let op = IndexScanOperator::new(
                    ctx.clone(),
                    table_id,
                    columns,
                    predicate,
                    remaining_predicate,
                    false,
                )
                .await?;
                Ok(Box::new(op))
            }

            PhysicalPlan::Filter {
                predicate, child, ..
            } => {
                let input_schema = child.output_schema();
                let child_op = build_operator_tree(*child, ctx).await?;
                Ok(Box::new(FilterOperator::new(
                    child_op,
                    predicate,
                    input_schema,
                )))
            }

            PhysicalPlan::Project {
                expressions, child, ..
            } => {
                let input_schema = child.output_schema();
                let child_op = build_operator_tree(*child, ctx).await?;
                Ok(Box::new(ProjectOperator::new(
                    child_op,
                    expressions,
                    input_schema,
                )))
            }

            PhysicalPlan::NestedLoopJoin {
                left,
                right,
                join_type,
                condition,
                ..
            } => {
                let left_schema = left.output_schema();
                let right_schema = right.output_schema();
                let left_op = build_operator_tree(*left, ctx).await?;
                let right_op = build_operator_tree(*right, ctx).await?;
                Ok(Box::new(NestedLoopJoinOperator::new(
                    left_op,
                    right_op,
                    join_type,
                    condition,
                    left_schema,
                    right_schema,
                )))
            }

            PhysicalPlan::HashJoin {
                left,
                right,
                join_type,
                left_keys,
                right_keys,
                remaining_condition,
                ..
            } => {
                let left_schema = left.output_schema();
                let right_schema = right.output_schema();
                let left_op = build_operator_tree(*left, ctx).await?;
                let right_op = build_operator_tree(*right, ctx).await?;
                Ok(Box::new(HashJoinOperator::new(
                    left_op,
                    right_op,
                    join_type,
                    left_keys,
                    right_keys,
                    remaining_condition,
                    left_schema,
                    right_schema,
                )))
            }

            PhysicalPlan::MergeJoin {
                left,
                right,
                join_type,
                left_keys,
                right_keys,
                ..
            } => {
                let left_schema = left.output_schema();
                let right_schema = right.output_schema();
                let left_op = build_operator_tree(*left, ctx).await?;
                let right_op = build_operator_tree(*right, ctx).await?;
                Ok(Box::new(MergeJoinOperator::new(
                    left_op,
                    right_op,
                    join_type,
                    left_keys,
                    right_keys,
                    left_schema,
                    right_schema,
                )))
            }

            PhysicalPlan::HashAggregate {
                group_by,
                aggregates,
                child,
                ..
            } => {
                let input_schema = child.output_schema();
                let output_schema = {
                    let mut schema = Vec::new();
                    for (i, expr) in group_by.iter().enumerate() {
                        schema.push(zyron_planner::logical::LogicalColumn {
                            table_idx: None,
                            column_id: zyron_catalog::ColumnId(i as u16),
                            name: format!("group{}", i),
                            type_id: expr.type_id(),
                            nullable: expr.nullable(),
                        });
                    }
                    for (i, agg) in aggregates.iter().enumerate() {
                        let idx = group_by.len() + i;
                        schema.push(zyron_planner::logical::LogicalColumn {
                            table_idx: None,
                            column_id: zyron_catalog::ColumnId(idx as u16),
                            name: agg.function_name.clone(),
                            type_id: agg.return_type,
                            nullable: true,
                        });
                    }
                    schema
                };
                let child_op = build_operator_tree(*child, ctx).await?;
                Ok(Box::new(HashAggregateOperator::new(
                    child_op,
                    group_by,
                    aggregates,
                    input_schema,
                    output_schema,
                )))
            }

            PhysicalPlan::SortAggregate {
                group_by,
                aggregates,
                child,
                ..
            } => {
                let input_schema = child.output_schema();
                let output_schema = {
                    let mut schema = Vec::new();
                    for (i, expr) in group_by.iter().enumerate() {
                        schema.push(zyron_planner::logical::LogicalColumn {
                            table_idx: None,
                            column_id: zyron_catalog::ColumnId(i as u16),
                            name: format!("group{}", i),
                            type_id: expr.type_id(),
                            nullable: expr.nullable(),
                        });
                    }
                    for (i, agg) in aggregates.iter().enumerate() {
                        let idx = group_by.len() + i;
                        schema.push(zyron_planner::logical::LogicalColumn {
                            table_idx: None,
                            column_id: zyron_catalog::ColumnId(idx as u16),
                            name: agg.function_name.clone(),
                            type_id: agg.return_type,
                            nullable: true,
                        });
                    }
                    schema
                };
                let child_op = build_operator_tree(*child, ctx).await?;
                Ok(Box::new(SortAggregateOperator::new(
                    child_op,
                    group_by,
                    aggregates,
                    input_schema,
                    output_schema,
                )))
            }

            PhysicalPlan::Sort {
                order_by,
                child,
                limit,
                ..
            } => {
                let input_schema = child.output_schema();
                let child_op = build_operator_tree(*child, ctx).await?;
                Ok(Box::new(SortOperator::new(
                    child_op,
                    order_by,
                    input_schema,
                    limit,
                )))
            }

            PhysicalPlan::Limit {
                limit,
                offset,
                child,
                ..
            } => {
                let child_op = build_operator_tree(*child, ctx).await?;
                Ok(Box::new(LimitOperator::new(child_op, limit, offset)))
            }

            PhysicalPlan::HashDistinct { child, .. } => {
                let child_op = build_operator_tree(*child, ctx).await?;
                Ok(Box::new(HashDistinctOperator::new(child_op)))
            }

            PhysicalPlan::SetOp {
                op,
                all,
                left,
                right,
                ..
            } => {
                let left_op = build_operator_tree(*left, ctx).await?;
                let right_op = build_operator_tree(*right, ctx).await?;
                Ok(Box::new(SetOpOperator::new(left_op, right_op, op, all)))
            }

            PhysicalPlan::Values { rows, schema, .. } => {
                Ok(Box::new(ValuesOperator::new(rows, schema)))
            }

            PhysicalPlan::Insert {
                table_id, source, ..
            } => {
                let source_op = build_operator_tree(*source, ctx).await?;
                Ok(Box::new(InsertOperator::new(
                    source_op,
                    ctx.clone(),
                    table_id,
                )))
            }

            PhysicalPlan::Delete {
                table_id, child, ..
            } => {
                // Build a scan that tracks tuple IDs for deletion.
                let child_op = build_scan_with_tuple_ids(*child, ctx).await?;
                Ok(Box::new(DeleteOperator::new(
                    child_op,
                    ctx.clone(),
                    table_id,
                )))
            }

            PhysicalPlan::Update {
                table_id,
                assignments,
                child,
                ..
            } => {
                let input_schema = child.output_schema();
                let child_op = build_scan_with_tuple_ids(*child, ctx).await?;
                Ok(Box::new(UpdateOperator::new(
                    child_op,
                    ctx.clone(),
                    table_id,
                    assignments,
                    input_schema,
                )))
            }
        };
        result
    })
}

/// Builds an operator tree where the leaf scan tracks tuple IDs.
/// Used by DELETE and UPDATE to identify which heap rows to modify.
fn build_scan_with_tuple_ids(
    plan: PhysicalPlan,
    ctx: &Arc<ExecutionContext>,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Box<dyn Operator>>> + Send + '_>> {
    Box::pin(async move {
        let result: Result<Box<dyn Operator>> = match plan {
            PhysicalPlan::SeqScan {
                table_id,
                columns,
                predicate,
                ..
            } => {
                let op =
                    SeqScanOperator::new(ctx.clone(), table_id, columns, predicate, true).await?;
                Ok(Box::new(op) as Box<dyn Operator>)
            }

            PhysicalPlan::IndexScan {
                table_id,
                columns,
                predicate,
                remaining_predicate,
                ..
            } => {
                let op = IndexScanOperator::new(
                    ctx.clone(),
                    table_id,
                    columns,
                    predicate,
                    remaining_predicate,
                    true,
                )
                .await?;
                Ok(Box::new(op) as Box<dyn Operator>)
            }

            PhysicalPlan::Filter {
                predicate, child, ..
            } => {
                let input_schema = child.output_schema();
                let child_op = build_scan_with_tuple_ids(*child, ctx).await?;
                Ok(Box::new(FilterOperator::new(
                    child_op,
                    predicate,
                    input_schema,
                )))
            }

            PhysicalPlan::Limit {
                limit,
                offset,
                child,
                ..
            } => {
                let child_op = build_scan_with_tuple_ids(*child, ctx).await?;
                Ok(Box::new(LimitOperator::new(child_op, limit, offset)))
            }

            other => {
                // Fall back to normal build for non-scan nodes.
                build_operator_tree(other, ctx).await
            }
        };
        result
    })
}

/// Executes a PhysicalPlan and collects all result batches.
pub async fn execute(plan: PhysicalPlan, ctx: &Arc<ExecutionContext>) -> Result<Vec<DataBatch>> {
    let mut root = build_operator_tree(plan, ctx).await?;
    let mut results = Vec::new();

    loop {
        match root.next().await? {
            Some(exec_batch) => results.push(exec_batch.batch),
            None => break,
        }
    }

    Ok(results)
}

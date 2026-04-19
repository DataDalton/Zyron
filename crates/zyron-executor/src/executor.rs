//! Executor driver that converts a PhysicalPlan tree into an Operator tree
//! and drains it to produce result batches.

use std::sync::Arc;

use zyron_common::Result;
use zyron_planner::logical::AsOfTarget;
use zyron_planner::physical::PhysicalPlan;

use crate::batch::DataBatch;
use crate::context::ExecutionContext;
use crate::operator::aggregate::{HashAggregateOperator, SortAggregateOperator};
use crate::operator::distinct::HashDistinctOperator;
use crate::operator::filter::FilterOperator;
use crate::operator::join::{HashJoinOperator, MergeJoinOperator, NestedLoopJoinOperator};
use crate::operator::limit::LimitOperator;
use crate::operator::modify::{DeleteOperator, InsertOperator, UpdateOperator, ValuesOperator};
use crate::operator::project::ProjectOperator;
use crate::operator::scan::{
    IndexScanOperator, ParallelSeqScanOperator, SeqScanOperator, should_use_parallel_scan,
};
use crate::operator::setop::SetOpOperator;
use crate::operator::sort::SortOperator;
use crate::operator::{MetricsOperator, Operator, OperatorMetrics};

/// Result of building an operator tree: the operator plus optional metrics
/// (populated only when analyze mode is enabled on the ExecutionContext).
struct BuildResult {
    op: Box<dyn Operator>,
    metrics: Option<Arc<OperatorMetrics>>,
}

impl BuildResult {
    fn new(op: Box<dyn Operator>) -> Self {
        Self { op, metrics: None }
    }

    /// Wraps the operator with a MetricsOperator if analyze is enabled.
    fn with_metrics(
        mut self,
        name: &str,
        analyze: bool,
        child_metrics: Vec<Arc<OperatorMetrics>>,
    ) -> Self {
        if analyze {
            let metrics = OperatorMetrics::with_children(name, child_metrics);
            self.op = Box::new(MetricsOperator::new(self.op, metrics.clone()));
            self.metrics = Some(metrics);
        }
        self
    }
}

/// Helper to collect child metrics into a Vec, filtering out None values.
fn collect_metrics(items: &[&Option<Arc<OperatorMetrics>>]) -> Vec<Arc<OperatorMetrics>> {
    items.iter().filter_map(|m| m.as_ref().cloned()).collect()
}

/// Recursively converts a PhysicalPlan into an executable Operator tree.
/// When analyze mode is enabled on the context, each operator is wrapped
/// with a MetricsOperator that collects timing and row count stats.
fn build_operator_tree(
    plan: PhysicalPlan,
    ctx: &Arc<ExecutionContext>,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<BuildResult>> + Send + '_>> {
    Box::pin(async move {
        let analyze = ctx.analyze;
        let result: Result<BuildResult> = match plan {
            PhysicalPlan::SeqScan {
                table_id,
                columns,
                predicate,
                as_of,
                ..
            } => {
                let as_of_version = match &as_of {
                    Some(AsOfTarget::Version(v)) => Some(*v),
                    _ => None,
                };

                // Use parallel scan for large tables when not tracking tuple IDs.
                // Parallel scan does not support as_of yet, fall back to serial.
                let table_entry = ctx.get_table_entry(table_id)?;
                let num_pages = ctx.disk_manager.num_pages(table_entry.heap_file_id).await?;

                if as_of_version.is_none() && should_use_parallel_scan(num_pages, false) {
                    let op =
                        ParallelSeqScanOperator::new(ctx.clone(), table_id, columns, predicate)
                            .await?;
                    let br = BuildResult::new(Box::new(op));
                    Ok(br.with_metrics("ParallelSeqScan", analyze, vec![]))
                } else {
                    let op = SeqScanOperator::new(
                        ctx.clone(),
                        table_id,
                        columns,
                        predicate,
                        false,
                        as_of_version,
                    )
                    .await?;
                    let br = BuildResult::new(Box::new(op));
                    Ok(br.with_metrics("SeqScan", analyze, vec![]))
                }
            }

            PhysicalPlan::IndexScan {
                table_id,
                index_id,
                index,
                columns,
                predicate,
                remaining_predicate,
                ..
            } => {
                let btree = ctx.get_index(index_id);
                let op = IndexScanOperator::new(
                    ctx.clone(),
                    table_id,
                    Some(index),
                    btree,
                    columns,
                    predicate,
                    remaining_predicate,
                    false,
                )
                .await?;
                let br = BuildResult::new(Box::new(op));
                Ok(br.with_metrics("IndexScan", analyze, vec![]))
            }

            PhysicalPlan::FulltextScan {
                table_id,
                index_id,
                columns,
                match_expr,
                remaining_predicate,
                ..
            } => {
                let output_schema = columns.clone();
                let op = crate::operator::fts_scan::FulltextScanOperator::new(
                    ctx.clone(),
                    table_id,
                    index_id,
                    columns,
                    match_expr,
                )
                .await?;
                let mut br = BuildResult::new(Box::new(op));
                // Apply remaining predicate as a filter on top of the FTS scan.
                if let Some(pred) = remaining_predicate {
                    br =
                        BuildResult::new(Box::new(FilterOperator::new(br.op, pred, output_schema)));
                }
                Ok(br.with_metrics("FulltextScan", analyze, vec![]))
            }

            PhysicalPlan::VectorScan {
                table_id,
                index_id,
                columns,
                query_vector,
                k,
                remaining_predicate,
                ..
            } => {
                let output_schema = columns.clone();
                let ef_search = 64u16; // default ef_search
                let op = crate::operator::vector_scan::VectorScanOperator::new(
                    ctx.clone(),
                    table_id,
                    index_id,
                    columns,
                    query_vector,
                    k,
                    ef_search,
                )
                .await?;
                let mut br = BuildResult::new(Box::new(op));
                if let Some(pred) = remaining_predicate {
                    br =
                        BuildResult::new(Box::new(FilterOperator::new(br.op, pred, output_schema)));
                }
                Ok(br.with_metrics("VectorScan", analyze, vec![]))
            }

            PhysicalPlan::SpatialScan {
                table_id,
                index_id,
                columns,
                kind,
                remaining_predicate,
                ..
            } => {
                let output_schema = columns.clone();
                let op = crate::operator::spatial_scan::SpatialScanOperator::new(
                    ctx.clone(),
                    table_id,
                    index_id,
                    columns,
                    kind,
                )
                .await?;
                let mut br = BuildResult::new(Box::new(op));
                if let Some(pred) = remaining_predicate {
                    br =
                        BuildResult::new(Box::new(FilterOperator::new(br.op, pred, output_schema)));
                }
                Ok(br.with_metrics("SpatialScan", analyze, vec![]))
            }

            PhysicalPlan::GraphAlgorithm {
                algorithm,
                schema_name,
                params,
                output_columns,
                ..
            } => {
                use crate::operator::graph_scan::{GraphAlgorithmKind, GraphAlgorithmOperator};
                use zyron_planner::physical::GraphAlgorithmType;

                // Extract algorithm-specific parameters from bound expressions.
                let extract_f64 = |ps: &[(String, zyron_planner::binder::BoundExpr)],
                                   name: &str,
                                   default: f64|
                 -> f64 {
                    ps.iter()
                        .find(|(n, _)| n == name)
                        .and_then(|(_, e)| match e {
                            zyron_planner::binder::BoundExpr::Literal {
                                value: zyron_parser::ast::LiteralValue::Float(v),
                                ..
                            } => Some(*v),
                            zyron_planner::binder::BoundExpr::Literal {
                                value: zyron_parser::ast::LiteralValue::Integer(v),
                                ..
                            } => Some(*v as f64),
                            _ => None,
                        })
                        .unwrap_or(default)
                };
                let extract_u64 = |ps: &[(String, zyron_planner::binder::BoundExpr)],
                                   name: &str,
                                   default: u64|
                 -> u64 {
                    ps.iter()
                        .find(|(n, _)| n == name)
                        .and_then(|(_, e)| match e {
                            zyron_planner::binder::BoundExpr::Literal {
                                value: zyron_parser::ast::LiteralValue::Integer(v),
                                ..
                            } => Some(*v as u64),
                            _ => None,
                        })
                        .unwrap_or(default)
                };

                let kind = match algorithm {
                    GraphAlgorithmType::PageRank => GraphAlgorithmKind::PageRank {
                        damping: extract_f64(&params, "damping", 0.85),
                        iterations: extract_f64(&params, "iterations", 20.0) as usize,
                    },
                    GraphAlgorithmType::ShortestPath => GraphAlgorithmKind::ShortestPath {
                        source_id: extract_u64(&params, "source", 0),
                        target_id: extract_u64(&params, "target", 0),
                    },
                    GraphAlgorithmType::Bfs => GraphAlgorithmKind::Bfs {
                        source_id: extract_u64(&params, "source", 0),
                        max_depth: extract_u64(&params, "max_depth", 100) as u32,
                    },
                    GraphAlgorithmType::ConnectedComponents => {
                        GraphAlgorithmKind::ConnectedComponents
                    }
                    GraphAlgorithmType::CommunityDetection => {
                        GraphAlgorithmKind::CommunityDetection
                    }
                    GraphAlgorithmType::BetweennessCentrality => {
                        GraphAlgorithmKind::BetweennessCentrality
                    }
                };

                let op = GraphAlgorithmOperator::new(
                    Arc::clone(&ctx),
                    schema_name,
                    kind,
                    output_columns,
                )
                .await?;
                let br = BuildResult::new(Box::new(op));
                Ok(br.with_metrics("GraphAlgorithm", analyze, vec![]))
            }

            PhysicalPlan::Filter {
                predicate, child, ..
            } => {
                let input_schema = child.output_schema();
                let child_br = build_operator_tree(*child, ctx).await?;
                let child_m = collect_metrics(&[&child_br.metrics]);
                let br = BuildResult::new(Box::new(FilterOperator::new(
                    child_br.op,
                    predicate,
                    input_schema,
                )));
                Ok(br.with_metrics("Filter", analyze, child_m))
            }

            PhysicalPlan::Project {
                expressions, child, ..
            } => {
                let input_schema = child.output_schema();
                let child_br = build_operator_tree(*child, ctx).await?;
                let child_m = collect_metrics(&[&child_br.metrics]);
                let br = BuildResult::new(Box::new(ProjectOperator::new(
                    child_br.op,
                    expressions,
                    input_schema,
                )));
                Ok(br.with_metrics("Project", analyze, child_m))
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
                let left_br = build_operator_tree(*left, ctx).await?;
                let right_br = build_operator_tree(*right, ctx).await?;
                let child_m = collect_metrics(&[&left_br.metrics, &right_br.metrics]);
                let br = BuildResult::new(Box::new(NestedLoopJoinOperator::new(
                    left_br.op,
                    right_br.op,
                    join_type,
                    condition,
                    left_schema,
                    right_schema,
                )));
                Ok(br.with_metrics("NestedLoopJoin", analyze, child_m))
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
                let left_br = build_operator_tree(*left, ctx).await?;
                let right_br = build_operator_tree(*right, ctx).await?;
                let child_m = collect_metrics(&[&left_br.metrics, &right_br.metrics]);
                let br = BuildResult::new(Box::new(HashJoinOperator::new(
                    left_br.op,
                    right_br.op,
                    join_type,
                    left_keys,
                    right_keys,
                    remaining_condition,
                    left_schema,
                    right_schema,
                )));
                Ok(br.with_metrics("HashJoin", analyze, child_m))
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
                let left_br = build_operator_tree(*left, ctx).await?;
                let right_br = build_operator_tree(*right, ctx).await?;
                let child_m = collect_metrics(&[&left_br.metrics, &right_br.metrics]);
                let br = BuildResult::new(Box::new(MergeJoinOperator::new(
                    left_br.op,
                    right_br.op,
                    join_type,
                    left_keys,
                    right_keys,
                    left_schema,
                    right_schema,
                )));
                Ok(br.with_metrics("MergeJoin", analyze, child_m))
            }

            PhysicalPlan::HashAggregate {
                group_by,
                aggregates,
                child,
                ..
            } => {
                let input_schema = child.output_schema();
                let output_schema = build_aggregate_schema(&group_by, &aggregates);
                let child_br = build_operator_tree(*child, ctx).await?;
                let child_m = collect_metrics(&[&child_br.metrics]);
                let br = BuildResult::new(Box::new(HashAggregateOperator::new(
                    child_br.op,
                    group_by,
                    aggregates,
                    input_schema,
                    output_schema,
                )));
                Ok(br.with_metrics("HashAggregate", analyze, child_m))
            }

            PhysicalPlan::SortAggregate {
                group_by,
                aggregates,
                child,
                ..
            } => {
                let input_schema = child.output_schema();
                let output_schema = build_aggregate_schema(&group_by, &aggregates);
                let child_br = build_operator_tree(*child, ctx).await?;
                let child_m = collect_metrics(&[&child_br.metrics]);
                let br = BuildResult::new(Box::new(SortAggregateOperator::new(
                    child_br.op,
                    group_by,
                    aggregates,
                    input_schema,
                    output_schema,
                )));
                Ok(br.with_metrics("SortAggregate", analyze, child_m))
            }

            PhysicalPlan::Sort {
                order_by,
                child,
                limit,
                ..
            } => {
                let input_schema = child.output_schema();
                let child_br = build_operator_tree(*child, ctx).await?;
                let child_m = collect_metrics(&[&child_br.metrics]);
                let br = BuildResult::new(Box::new(SortOperator::new(
                    child_br.op,
                    order_by,
                    input_schema,
                    limit,
                )));
                Ok(br.with_metrics("Sort", analyze, child_m))
            }

            PhysicalPlan::Limit {
                limit,
                offset,
                child,
                ..
            } => {
                let child_br = build_operator_tree(*child, ctx).await?;
                let child_m = collect_metrics(&[&child_br.metrics]);
                let br = BuildResult::new(Box::new(LimitOperator::new(child_br.op, limit, offset)));
                Ok(br.with_metrics("Limit", analyze, child_m))
            }

            PhysicalPlan::HashDistinct { child, .. } => {
                let child_br = build_operator_tree(*child, ctx).await?;
                let child_m = collect_metrics(&[&child_br.metrics]);
                let br = BuildResult::new(Box::new(HashDistinctOperator::new(child_br.op)));
                Ok(br.with_metrics("HashDistinct", analyze, child_m))
            }

            PhysicalPlan::SetOp {
                op,
                all,
                left,
                right,
                ..
            } => {
                let left_br = build_operator_tree(*left, ctx).await?;
                let right_br = build_operator_tree(*right, ctx).await?;
                let child_m = collect_metrics(&[&left_br.metrics, &right_br.metrics]);
                let br = BuildResult::new(Box::new(SetOpOperator::new(
                    left_br.op,
                    right_br.op,
                    op,
                    all,
                )));
                Ok(br.with_metrics("SetOp", analyze, child_m))
            }

            PhysicalPlan::Values { rows, schema, .. } => {
                let br = BuildResult::new(Box::new(ValuesOperator::new(rows, schema)));
                Ok(br.with_metrics("Values", analyze, vec![]))
            }

            PhysicalPlan::Insert {
                table_id, source, ..
            } => {
                let source_br = build_operator_tree(*source, ctx).await?;
                let child_m = collect_metrics(&[&source_br.metrics]);
                let br = BuildResult::new(Box::new(InsertOperator::new(
                    source_br.op,
                    ctx.clone(),
                    table_id,
                )));
                Ok(br.with_metrics("Insert", analyze, child_m))
            }

            PhysicalPlan::Delete {
                table_id, child, ..
            } => {
                let child_br = build_scan_with_tuple_ids(*child, ctx).await?;
                let child_m = collect_metrics(&[&child_br.metrics]);
                let br = BuildResult::new(Box::new(DeleteOperator::new(
                    child_br.op,
                    ctx.clone(),
                    table_id,
                )));
                Ok(br.with_metrics("Delete", analyze, child_m))
            }

            PhysicalPlan::Update {
                table_id,
                assignments,
                child,
                ..
            } => {
                let input_schema = child.output_schema();
                let child_br = build_scan_with_tuple_ids(*child, ctx).await?;
                let child_m = collect_metrics(&[&child_br.metrics]);
                let br = BuildResult::new(Box::new(UpdateOperator::new(
                    child_br.op,
                    ctx.clone(),
                    table_id,
                    assignments,
                    input_schema,
                )));
                Ok(br.with_metrics("Update", analyze, child_m))
            }

            // Parallel scan: reuse the existing ParallelSeqScanOperator
            PhysicalPlan::ParallelSeqScan {
                table_id,
                columns,
                predicate,
                ..
            } => {
                let op =
                    ParallelSeqScanOperator::new(ctx.clone(), table_id, columns, predicate).await?;
                let br = BuildResult::new(Box::new(op));
                Ok(br.with_metrics("ParallelSeqScan", analyze, Vec::new()))
            }

            // Parallel hash join: fall back to serial hash join for now.
            // The planner marks it parallel for cost estimation purposes.
            PhysicalPlan::ParallelHashJoin {
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
                let left_br = build_operator_tree(*left, ctx).await?;
                let right_br = build_operator_tree(*right, ctx).await?;
                let child_m = collect_metrics(&[&left_br.metrics, &right_br.metrics]);
                let br = BuildResult::new(Box::new(HashJoinOperator::new(
                    left_br.op,
                    right_br.op,
                    join_type,
                    left_keys,
                    right_keys,
                    remaining_condition,
                    left_schema,
                    right_schema,
                )));
                Ok(br.with_metrics("ParallelHashJoin", analyze, child_m))
            }

            // Gather: passes through to child, wraps with metrics for EXPLAIN ANALYZE alignment
            PhysicalPlan::Gather { child, .. } => {
                let child_br = build_operator_tree(*child, ctx).await?;
                let child_m = collect_metrics(&[&child_br.metrics]);
                Ok(BuildResult::new(child_br.op).with_metrics("Gather", analyze, child_m))
            }

            // Repartition: passes through to child (partitioning is a future extension)
            PhysicalPlan::Repartition { child, .. } => {
                let child_br = build_operator_tree(*child, ctx).await?;
                let child_m = collect_metrics(&[&child_br.metrics]);
                Ok(BuildResult::new(child_br.op).with_metrics("Repartition", analyze, child_m))
            }

            // Broadcast: passes through to child
            PhysicalPlan::Broadcast { child, .. } => {
                let child_br = build_operator_tree(*child, ctx).await?;
                let child_m = collect_metrics(&[&child_br.metrics]);
                Ok(BuildResult::new(child_br.op).with_metrics("Broadcast", analyze, child_m))
            }

            PhysicalPlan::Window {
                window_exprs,
                child,
                ..
            } => {
                let input_schema = child.output_schema();
                let child_br = build_operator_tree(*child, ctx).await?;
                let child_m = collect_metrics(&[&child_br.metrics]);
                let op = crate::operator::window::WindowOperator::new(
                    child_br.op,
                    window_exprs,
                    input_schema,
                );
                Ok(BuildResult::new(Box::new(op)).with_metrics("Window", analyze, child_m))
            }
        };
        result
    })
}

/// Builds the output schema for aggregate operators.
fn build_aggregate_schema(
    group_by: &[zyron_planner::binder::BoundExpr],
    aggregates: &[zyron_planner::logical::AggregateExpr],
) -> Vec<zyron_planner::logical::LogicalColumn> {
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
}

/// Builds an operator tree where the leaf scan tracks tuple IDs.
/// Used by DELETE and UPDATE to identify which heap rows to modify.
fn build_scan_with_tuple_ids(
    plan: PhysicalPlan,
    ctx: &Arc<ExecutionContext>,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<BuildResult>> + Send + '_>> {
    Box::pin(async move {
        let analyze = ctx.analyze;
        let result: Result<BuildResult> = match plan {
            PhysicalPlan::SeqScan {
                table_id,
                columns,
                predicate,
                as_of,
                ..
            } => {
                let as_of_version = match &as_of {
                    Some(AsOfTarget::Version(v)) => Some(*v),
                    _ => None,
                };
                let op = SeqScanOperator::new(
                    ctx.clone(),
                    table_id,
                    columns,
                    predicate,
                    true,
                    as_of_version,
                )
                .await?;
                let br = BuildResult::new(Box::new(op) as Box<dyn Operator>);
                Ok(br.with_metrics("SeqScan", analyze, vec![]))
            }

            PhysicalPlan::IndexScan {
                table_id,
                index_id,
                index,
                columns,
                predicate,
                remaining_predicate,
                ..
            } => {
                let btree = ctx.get_index(index_id);
                let op = IndexScanOperator::new(
                    ctx.clone(),
                    table_id,
                    Some(index),
                    btree,
                    columns,
                    predicate,
                    remaining_predicate,
                    true,
                )
                .await?;
                let br = BuildResult::new(Box::new(op) as Box<dyn Operator>);
                Ok(br.with_metrics("IndexScan", analyze, vec![]))
            }

            PhysicalPlan::Filter {
                predicate, child, ..
            } => {
                let input_schema = child.output_schema();
                let child_br = build_scan_with_tuple_ids(*child, ctx).await?;
                let child_m = collect_metrics(&[&child_br.metrics]);
                let br = BuildResult::new(Box::new(FilterOperator::new(
                    child_br.op,
                    predicate,
                    input_schema,
                )));
                Ok(br.with_metrics("Filter", analyze, child_m))
            }

            PhysicalPlan::Limit {
                limit,
                offset,
                child,
                ..
            } => {
                let child_br = build_scan_with_tuple_ids(*child, ctx).await?;
                let child_m = collect_metrics(&[&child_br.metrics]);
                let br = BuildResult::new(Box::new(LimitOperator::new(child_br.op, limit, offset)));
                Ok(br.with_metrics("Limit", analyze, child_m))
            }

            other => build_operator_tree(other, ctx).await,
        };
        result
    })
}

/// Executes a PhysicalPlan and collects all result batches.
/// Checks for query cancellation between each batch.
pub async fn execute(plan: PhysicalPlan, ctx: &Arc<ExecutionContext>) -> Result<Vec<DataBatch>> {
    let br = build_operator_tree(plan, ctx).await?;
    let mut root = br.op;
    let mut results = Vec::new();

    loop {
        ctx.check_cancelled()?;
        match root.next().await? {
            Some(exec_batch) => results.push(exec_batch.batch),
            None => break,
        }
    }

    Ok(results)
}

/// Executes a PhysicalPlan with EXPLAIN ANALYZE, returning both the result
/// batches and the per-operator metrics tree.
pub async fn execute_analyze(
    plan: PhysicalPlan,
    ctx: &Arc<ExecutionContext>,
) -> Result<(Vec<DataBatch>, Option<Arc<OperatorMetrics>>)> {
    let br = build_operator_tree(plan, ctx).await?;
    let root_metrics = br.metrics.clone();
    let mut root = br.op;
    let mut results = Vec::new();

    loop {
        ctx.check_cancelled()?;
        match root.next().await? {
            Some(exec_batch) => results.push(exec_batch.batch),
            None => break,
        }
    }

    Ok((results, root_metrics))
}

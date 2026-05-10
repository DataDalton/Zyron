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
                // Resolve the AS OF qualifier into a concrete scan parameter
                let mut effective_predicate = predicate;
                let as_of_version = match &as_of {
                    Some(AsOfTarget::Version(v)) => Some(*v),
                    Some(AsOfTarget::Timestamp(ts)) => {
                        // Translate AS OF TIMESTAMP into a sys_start/sys_end
                        // predicate on a system-versioned table. Without this
                        // the scan would silently return current data
                        let entry = ctx.catalog.get_table_by_id(table_id)?;
                        if !entry.system_versioned {
                            return Err(zyron_common::ZyronError::ExecutionError(format!(
                                "AS OF TIMESTAMP requires a system-versioned table, table {} is not system-versioned. \
                                 Enable with ALTER TABLE ... SET (system_versioning = on) or use AS OF VERSION instead",
                                entry.name
                            )));
                        }
                        let predicate_for_ts = build_system_time_predicate(&entry, *ts)?;
                        effective_predicate = Some(match effective_predicate {
                            Some(existing) => combine_with_and(existing, predicate_for_ts),
                            None => predicate_for_ts,
                        });
                        None
                    }
                    Some(AsOfTarget::Branch(name)) => {
                        return Err(zyron_common::ZyronError::ExecutionError(format!(
                            "IN BRANCH '{}' is parsed and bound but executor branch lookup is not yet wired",
                            name
                        )));
                    }
                    None => None,
                };

                // ParallelSeqScan does not support as_of, fall back to serial in that case
                let num_pages = ctx.get_heap_file(table_id).await?.num_pages_cached() as u64;

                if as_of_version.is_none()
                    && as_of.is_none()
                    && should_use_parallel_scan(num_pages, false)
                {
                    let op = ParallelSeqScanOperator::new(
                        ctx.clone(),
                        table_id,
                        columns,
                        effective_predicate,
                    )
                    .await?;
                    let br = BuildResult::new(Box::new(op));
                    Ok(br.with_metrics("ParallelSeqScan", analyze, vec![]))
                } else {
                    let op = SeqScanOperator::new(
                        ctx.clone(),
                        table_id,
                        columns,
                        effective_predicate,
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
                    br = BuildResult::new(Box::new(FilterOperator::with_params(
                        br.op,
                        pred,
                        output_schema,
                        ctx.params.clone(),
                    )));
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
                    br = BuildResult::new(Box::new(FilterOperator::with_params(
                        br.op,
                        pred,
                        output_schema,
                        ctx.params.clone(),
                    )));
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
                    br = BuildResult::new(Box::new(FilterOperator::with_params(
                        br.op,
                        pred,
                        output_schema,
                        ctx.params.clone(),
                    )));
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

            PhysicalPlan::AnalyticsTableFunction {
                function_name,
                named_args,
                positional_args,
                output_columns,
                ..
            } => {
                use crate::operator::analytics_table_fn::AnalyticsTableFunctionOperator;
                let op = AnalyticsTableFunctionOperator::new(
                    Arc::clone(&ctx),
                    function_name,
                    named_args,
                    positional_args,
                    output_columns,
                );
                let br = BuildResult::new(Box::new(op));
                Ok(br.with_metrics("AnalyticsTableFunction", analyze, vec![]))
            }

            PhysicalPlan::Filter {
                predicate, child, ..
            } => {
                let input_schema = child.output_schema();
                let params = ctx.params.clone();
                let child_br = build_operator_tree(*child, ctx).await?;
                let child_m = collect_metrics(&[&child_br.metrics]);
                let br = BuildResult::new(Box::new(FilterOperator::with_params(
                    child_br.op,
                    predicate,
                    input_schema,
                    params,
                )));
                Ok(br.with_metrics("Filter", analyze, child_m))
            }

            PhysicalPlan::Project {
                expressions, child, ..
            } => {
                let input_schema = child.output_schema();
                let params = ctx.params.clone();
                let child_br = build_operator_tree(*child, ctx).await?;
                let child_m = collect_metrics(&[&child_br.metrics]);
                let br = BuildResult::new(Box::new(ProjectOperator::with_params(
                    child_br.op,
                    expressions,
                    input_schema,
                    params,
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
                let br = BuildResult::new(Box::new(ValuesOperator::with_params(
                    rows,
                    schema,
                    ctx.params.clone(),
                )));
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
            table_idx: Some(zyron_planner::logical::AGGREGATE_TABLE_IDX),
            column_id: zyron_catalog::ColumnId(i as u16),
            name: format!("group{}", i),
            type_id: expr.type_id(),
            nullable: expr.nullable(),
        });
    }
    for (i, agg) in aggregates.iter().enumerate() {
        let idx = group_by.len() + i;
        schema.push(zyron_planner::logical::LogicalColumn {
            table_idx: Some(zyron_planner::logical::AGGREGATE_TABLE_IDX),
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
                let mut effective_predicate = predicate;
                let as_of_version = match &as_of {
                    Some(AsOfTarget::Version(v)) => Some(*v),
                    Some(AsOfTarget::Timestamp(ts)) => {
                        let entry = ctx.catalog.get_table_by_id(table_id)?;
                        if !entry.system_versioned {
                            return Err(zyron_common::ZyronError::ExecutionError(format!(
                                "AS OF TIMESTAMP requires a system-versioned table, table {} is not system-versioned",
                                entry.name
                            )));
                        }
                        let predicate_for_ts = build_system_time_predicate(&entry, *ts)?;
                        effective_predicate = Some(match effective_predicate {
                            Some(existing) => combine_with_and(existing, predicate_for_ts),
                            None => predicate_for_ts,
                        });
                        None
                    }
                    Some(AsOfTarget::Branch(name)) => {
                        return Err(zyron_common::ZyronError::ExecutionError(format!(
                            "IN BRANCH '{}' is parsed and bound but executor branch lookup is not yet wired",
                            name
                        )));
                    }
                    None => None,
                };
                let op = SeqScanOperator::new(
                    ctx.clone(),
                    table_id,
                    columns,
                    effective_predicate,
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
                let params = ctx.params.clone();
                let child_br = build_scan_with_tuple_ids(*child, ctx).await?;
                let child_m = collect_metrics(&[&child_br.metrics]);
                let br = BuildResult::new(Box::new(FilterOperator::with_params(
                    child_br.op,
                    predicate,
                    input_schema,
                    params,
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

/// Builds the visibility predicate for AS OF TIMESTAMP <ts> on a system-versioned
/// table. Filters tuples whose system-time period [sys_start, sys_end) contains ts
/// `sys_end IS NULL` is treated as "still current" so currently-live rows match
fn build_system_time_predicate(
    entry: &zyron_catalog::TableEntry,
    ts_micros: i64,
) -> Result<zyron_planner::binder::BoundExpr> {
    use zyron_common::TypeId;
    use zyron_parser::ast::{BinaryOperator, LiteralValue};
    use zyron_planner::binder::{BoundExpr, ColumnRef};

    let sys_start = entry
        .columns
        .iter()
        .find(|c| c.name == "sys_start")
        .ok_or_else(|| {
            zyron_common::ZyronError::ExecutionError(format!(
                "system-versioned table {} is missing the sys_start column",
                entry.name
            ))
        })?;
    let sys_end = entry
        .columns
        .iter()
        .find(|c| c.name == "sys_end")
        .ok_or_else(|| {
            zyron_common::ZyronError::ExecutionError(format!(
                "system-versioned table {} is missing the sys_end column",
                entry.name
            ))
        })?;

    let table_idx = 0usize;
    let ts_lit = BoundExpr::Literal {
        value: LiteralValue::Integer(ts_micros),
        type_id: TypeId::Timestamp,
    };
    let sys_start_ref = BoundExpr::ColumnRef(ColumnRef {
        table_idx,
        column_id: sys_start.id,
        type_id: sys_start.type_id,
        nullable: sys_start.nullable,
    });
    let sys_end_ref = BoundExpr::ColumnRef(ColumnRef {
        table_idx,
        column_id: sys_end.id,
        type_id: sys_end.type_id,
        nullable: sys_end.nullable,
    });

    // sys_start <= ts
    let start_le = BoundExpr::BinaryOp {
        left: Box::new(sys_start_ref),
        op: BinaryOperator::LtEq,
        right: Box::new(ts_lit.clone()),
        type_id: TypeId::Boolean,
    };
    // sys_end > ts
    let end_gt = BoundExpr::BinaryOp {
        left: Box::new(sys_end_ref.clone()),
        op: BinaryOperator::Gt,
        right: Box::new(ts_lit),
        type_id: TypeId::Boolean,
    };
    // sys_end IS NULL (currently-live rows have sys_end = MAX_TIMESTAMP per
    // SystemVersionedTable::on_insert_defaults, but defensive-IS-NULL keeps
    // schemas that store NULL for the live row also working)
    let end_is_null = BoundExpr::IsNull {
        expr: Box::new(sys_end_ref),
        negated: false,
    };
    // (sys_end > ts OR sys_end IS NULL)
    let end_visible = BoundExpr::BinaryOp {
        left: Box::new(end_gt),
        op: BinaryOperator::Or,
        right: Box::new(end_is_null),
        type_id: TypeId::Boolean,
    };
    // sys_start <= ts AND (sys_end > ts OR sys_end IS NULL)
    Ok(BoundExpr::BinaryOp {
        left: Box::new(start_le),
        op: BinaryOperator::And,
        right: Box::new(end_visible),
        type_id: TypeId::Boolean,
    })
}

/// Combines two BoundExpr predicates with logical AND
fn combine_with_and(
    left: zyron_planner::binder::BoundExpr,
    right: zyron_planner::binder::BoundExpr,
) -> zyron_planner::binder::BoundExpr {
    use zyron_common::TypeId;
    use zyron_parser::ast::BinaryOperator;
    use zyron_planner::binder::BoundExpr;
    BoundExpr::BinaryOp {
        left: Box::new(left),
        op: BinaryOperator::And,
        right: Box::new(right),
        type_id: TypeId::Boolean,
    }
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

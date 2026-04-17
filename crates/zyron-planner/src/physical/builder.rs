//! Converts optimized logical plans into physical execution plans.
//!
//! Makes cost-based decisions for operator selection:
//! - SeqScan vs IndexScan (index scan preferred when selectivity < 10%)
//! - HashJoin vs MergeJoin vs NestedLoopJoin
//! - HashAggregate vs SortAggregate

use crate::binder::BoundExpr;
use crate::cost::{CostModel, EncodingCostParameters, INDEX_SCAN_SELECTIVITY_THRESHOLD, PlanCost};
use crate::logical::{JoinCondition, LogicalPlan};
use crate::optimizer::rules::{encoding_pushdown, parallel_plan};
use crate::physical::*;
use std::sync::Arc;
use zyron_catalog::{Catalog, IndexEntry};
use zyron_common::{Result, TypeId};
use zyron_parser::ast::{BinaryOperator, JoinType};

/// Converts an optimized logical plan into a physical plan using cost-based decisions.
pub fn build_physical_plan(logical: LogicalPlan, catalog: &Catalog) -> Result<PhysicalPlan> {
    let cost_model = CostModel::default();
    PhysicalPlanner::new(catalog, cost_model).plan(logical)
}

struct PhysicalPlanner<'a> {
    catalog: &'a Catalog,
    cost_model: CostModel,
}

impl<'a> PhysicalPlanner<'a> {
    fn new(catalog: &'a Catalog, cost_model: CostModel) -> Self {
        Self {
            catalog,
            cost_model,
        }
    }

    fn plan(&self, logical: LogicalPlan) -> Result<PhysicalPlan> {
        match logical {
            LogicalPlan::Scan {
                table_id,
                columns,
                encoding_hints,
                as_of,
                ..
            } => self.plan_scan(table_id, columns, None, encoding_hints, as_of),
            LogicalPlan::Filter { predicate, child } => {
                // Try to push the filter into a scan (index scan opportunity)
                if let LogicalPlan::Scan {
                    table_id,
                    columns,
                    encoding_hints,
                    as_of,
                    ..
                } = *child
                {
                    return self.plan_scan(
                        table_id,
                        columns,
                        Some(predicate),
                        encoding_hints,
                        as_of,
                    );
                }

                let child_plan = self.plan(*child)?;
                let child_cost = *child_plan.cost();
                let selectivity = self.cost_model.estimate_selectivity(&predicate, None, None);
                let cost = PlanCost {
                    io_cost: 0.0,
                    cpu_cost: child_cost.row_count * self.cost_model.cpu_operator_cost,
                    row_count: (child_cost.row_count * selectivity).max(1.0),
                };
                Ok(PhysicalPlan::Filter {
                    predicate,
                    child: Box::new(child_plan),
                    cost,
                })
            }
            LogicalPlan::Project {
                expressions,
                aliases,
                child,
            } => {
                let child_plan = self.plan(*child)?;
                let child_cost = *child_plan.cost();
                let cost = PlanCost {
                    io_cost: 0.0,
                    cpu_cost: child_cost.row_count * self.cost_model.cpu_operator_cost,
                    row_count: child_cost.row_count,
                };

                // Collect window functions from expressions. If any exist, we
                // insert a Window node between the child and Project, and rewrite
                // each WindowFunction occurrence as a ColumnRef pointing to the
                // appended window output column.
                let mut window_exprs: Vec<crate::binder::BoundExpr> = Vec::new();
                let mut window_names: Vec<String> = Vec::new();
                let input_schema = child_plan.output_schema();

                let rewritten: Vec<crate::binder::BoundExpr> = expressions
                    .iter()
                    .enumerate()
                    .map(|(i, e)| {
                        rewrite_window_refs(
                            e,
                            &mut window_exprs,
                            &mut window_names,
                            input_schema.len(),
                            aliases.get(i).and_then(|a| a.clone()),
                        )
                    })
                    .collect();

                if !window_exprs.is_empty() {
                    let window_cost = PlanCost {
                        io_cost: 0.0,
                        cpu_cost: child_cost.row_count
                            * self.cost_model.cpu_operator_cost
                            * window_exprs.len() as f64,
                        row_count: child_cost.row_count,
                    };
                    let window_plan = PhysicalPlan::Window {
                        window_exprs,
                        window_names,
                        child: Box::new(child_plan),
                        cost: window_cost,
                    };
                    return Ok(PhysicalPlan::Project {
                        expressions: rewritten,
                        aliases,
                        child: Box::new(window_plan),
                        cost,
                    });
                }

                Ok(PhysicalPlan::Project {
                    expressions,
                    aliases,
                    child: Box::new(child_plan),
                    cost,
                })
            }
            LogicalPlan::Join {
                left,
                right,
                join_type,
                condition,
            } => self.plan_join(*left, *right, join_type, condition),
            LogicalPlan::Aggregate {
                group_by,
                aggregates,
                child,
            } => self.plan_aggregate(group_by, aggregates, *child),
            LogicalPlan::Sort { order_by, child } => {
                let child_plan = self.plan(*child)?;
                let child_cost = *child_plan.cost();
                let sort_cost = self.cost_model.cost_sort(&child_cost);
                Ok(PhysicalPlan::Sort {
                    order_by,
                    child: Box::new(child_plan),
                    limit: None,
                    cost: PlanCost {
                        io_cost: 0.0,
                        cpu_cost: sort_cost.cpu_cost - child_cost.cpu_cost,
                        row_count: child_cost.row_count,
                    },
                })
            }
            LogicalPlan::Limit {
                limit,
                offset,
                child,
            } => {
                // Check if there's a Sort below for top-N optimization
                let child_plan = self.plan(*child)?;
                let rows = limit
                    .map(|l| (l as f64).min(child_plan.cost().row_count))
                    .unwrap_or(child_plan.cost().row_count);
                let cost = PlanCost {
                    io_cost: 0.0,
                    cpu_cost: rows * self.cost_model.cpu_tuple_cost,
                    row_count: rows,
                };
                Ok(PhysicalPlan::Limit {
                    limit,
                    offset,
                    child: Box::new(child_plan),
                    cost,
                })
            }
            LogicalPlan::Distinct { child } => {
                let child_plan = self.plan(*child)?;
                let child_cost = *child_plan.cost();
                let cost = PlanCost {
                    io_cost: 0.0,
                    cpu_cost: child_cost.row_count * self.cost_model.cpu_operator_cost,
                    row_count: child_cost.row_count * 0.8,
                };
                Ok(PhysicalPlan::HashDistinct {
                    child: Box::new(child_plan),
                    cost,
                })
            }
            LogicalPlan::SetOp {
                op,
                all,
                left,
                right,
            } => {
                let left_plan = self.plan(*left)?;
                let right_plan = self.plan(*right)?;
                let cost = PlanCost {
                    io_cost: 0.0,
                    cpu_cost: (left_plan.cost().row_count + right_plan.cost().row_count)
                        * self.cost_model.cpu_tuple_cost,
                    row_count: left_plan.cost().row_count + right_plan.cost().row_count,
                };
                Ok(PhysicalPlan::SetOp {
                    op,
                    all,
                    left: Box::new(left_plan),
                    right: Box::new(right_plan),
                    cost,
                })
            }
            LogicalPlan::Insert {
                table_id,
                target_columns,
                source,
            } => {
                let source_plan = self.plan(*source)?;
                let cost = *source_plan.cost();
                Ok(PhysicalPlan::Insert {
                    table_id,
                    target_columns,
                    source: Box::new(source_plan),
                    cost,
                })
            }
            LogicalPlan::Values { rows, schema } => {
                let cost = PlanCost {
                    io_cost: 0.0,
                    cpu_cost: rows.len() as f64 * self.cost_model.cpu_tuple_cost,
                    row_count: rows.len() as f64,
                };
                Ok(PhysicalPlan::Values { rows, schema, cost })
            }
            LogicalPlan::Update {
                table_id,
                assignments,
                child,
            } => {
                let child_plan = self.plan(*child)?;
                let cost = *child_plan.cost();
                Ok(PhysicalPlan::Update {
                    table_id,
                    assignments,
                    child: Box::new(child_plan),
                    cost,
                })
            }
            LogicalPlan::Delete { table_id, child } => {
                let child_plan = self.plan(*child)?;
                let cost = *child_plan.cost();
                Ok(PhysicalPlan::Delete {
                    table_id,
                    child: Box::new(child_plan),
                    cost,
                })
            }
            LogicalPlan::GraphAlgorithm {
                schema_name,
                algorithm,
                params,
                output_columns,
            } => {
                let algo_type = match algorithm.as_str() {
                    "pagerank" => GraphAlgorithmType::PageRank,
                    "shortest_path" => GraphAlgorithmType::ShortestPath,
                    "bfs" => GraphAlgorithmType::Bfs,
                    "connected_components" => GraphAlgorithmType::ConnectedComponents,
                    "community_detection" => GraphAlgorithmType::CommunityDetection,
                    "betweenness_centrality" => GraphAlgorithmType::BetweennessCentrality,
                    other => {
                        return Err(zyron_common::ZyronError::PlanError(format!(
                            "unknown graph algorithm '{}'",
                            other
                        )));
                    }
                };

                // Cost estimates are tied to big-O complexity of each algorithm.
                // Without edge/node counts at plan time, we use a nominal graph
                // size of V=10_000 nodes and E=100_000 edges so the optimizer
                // can at least rank graph queries against each other.
                let v: f64 = 10_000.0;
                let e: f64 = 100_000.0;
                let pagerank_iters: f64 = 20.0;
                let (cpu, row_count) = match algo_type {
                    // O(iter * (V + E))
                    GraphAlgorithmType::PageRank => (pagerank_iters * (v + e), v),
                    // O(V + E) Dijkstra-equivalent, one path out
                    GraphAlgorithmType::ShortestPath => (v + e, v.sqrt()),
                    // O(V + E) level-limited, bounded by reachable subgraph
                    GraphAlgorithmType::Bfs => (v + e, v),
                    // O(V + E) union-find
                    GraphAlgorithmType::ConnectedComponents => (v + e, v),
                    // O(iter * (V + E)) Louvain-style
                    GraphAlgorithmType::CommunityDetection => (10.0 * (v + e), v),
                    // O(V * (V + E)) Brandes' algorithm, worst-case of the set
                    GraphAlgorithmType::BetweennessCentrality => (v * (v + e), v),
                };
                let cost = PlanCost {
                    io_cost: v, // single pass to read the graph backing tables
                    cpu_cost: cpu,
                    row_count,
                };

                Ok(PhysicalPlan::GraphAlgorithm {
                    algorithm: algo_type,
                    schema_name,
                    params,
                    output_columns,
                    cost,
                })
            }
        }
    }

    // -----------------------------------------------------------------------
    // Scan planning: SeqScan vs IndexScan
    // -----------------------------------------------------------------------

    fn plan_scan(
        &self,
        table_id: zyron_catalog::TableId,
        columns: Vec<crate::logical::LogicalColumn>,
        predicate: Option<BoundExpr>,
        encoding_hints: Option<encoding_pushdown::EncodingHint>,
        as_of: Option<crate::logical::AsOfTarget>,
    ) -> Result<PhysicalPlan> {
        // Get table stats
        let table_stats = self.catalog.get_stats(table_id);

        // Get available indexes
        let indexes = self.catalog.get_indexes_for_table(table_id);

        // Check for full-text search predicates (MATCH AGAINST -> match_against function)
        if let Some(pred) = &predicate {
            if let Some((fts_expr, remaining)) = extract_match_against(pred) {
                // Find a Fulltext index covering the referenced columns
                for index in &indexes {
                    if index.index_type == zyron_catalog::IndexType::Fulltext {
                        let cost = PlanCost {
                            io_cost: 1.0,
                            cpu_cost: 10.0,
                            row_count: 100.0,
                        };
                        return Ok(PhysicalPlan::FulltextScan {
                            table_id,
                            index_id: index.id,
                            columns,
                            match_expr: fts_expr.clone(),
                            remaining_predicate: remaining.cloned(),
                            cost,
                        });
                    }
                }
            }
        }

        // Check for vector distance predicates (vector_distance_* function calls)
        if let Some(pred) = &predicate {
            if let Some((vec_expr, remaining)) = extract_vector_distance(pred) {
                for index in &indexes {
                    if index.index_type == zyron_catalog::IndexType::Vector {
                        let cost = PlanCost {
                            io_cost: 1.0,
                            cpu_cost: 5.0,
                            row_count: 10.0,
                        };
                        let _ = vec_expr; // Referenced columns extracted at execution time
                        return Ok(PhysicalPlan::VectorScan {
                            table_id,
                            index_id: index.id,
                            columns,
                            query_vector: Vec::new(), // Populated at execution time from bound expr
                            metric: 0,                // Populated from index config
                            k: 10,                    // Default, overridden by LIMIT
                            remaining_predicate: remaining.cloned(),
                            cost,
                        });
                    }
                }
            }
        }

        // Try to find a B-tree index scan opportunity
        if let Some(pred) = &predicate {
            if let Some((ts, cs)) = &table_stats {
                for index in &indexes {
                    if let Some((index_pred, remaining)) = match_index(pred, index) {
                        let selectivity =
                            self.cost_model
                                .estimate_selectivity(&index_pred, Some(ts), Some(cs));

                        if selectivity < INDEX_SCAN_SELECTIVITY_THRESHOLD {
                            let cost = self.cost_model.cost_index_scan(ts, selectivity);
                            return Ok(PhysicalPlan::IndexScan {
                                table_id,
                                index_id: index.id,
                                index: Arc::clone(index),
                                columns,
                                predicate: index_pred,
                                remaining_predicate: remaining,
                                scan_direction: ScanDirection::Forward,
                                cost,
                                as_of: as_of.clone(),
                            });
                        }
                    }
                }
            }
        }

        // Compute sequential scan cost
        let seq_cost = if let Some((ts, _)) = &table_stats {
            let mut scan_cost = self.cost_model.cost_seq_scan(ts);
            if let Some(pred) = &predicate {
                let selectivity = self.cost_model.estimate_selectivity(
                    pred,
                    table_stats.as_ref().map(|(ts, _)| ts),
                    table_stats.as_ref().map(|(_, cs)| cs.as_slice()),
                );
                scan_cost.row_count = (scan_cost.row_count * selectivity).max(1.0);
            }
            scan_cost
        } else {
            PlanCost {
                io_cost: 10.0,
                cpu_cost: 1000.0 * self.cost_model.cpu_tuple_cost,
                row_count: if predicate.is_some() { 100.0 } else { 1000.0 },
            }
        };

        // Consider parallel scan for large tables.
        // Use the predicate-adjusted row count from seq_cost to decide threshold,
        // since filtering reduces the effective work.
        if let Some((ts, _)) = &table_stats {
            if parallel_plan::should_parallelize(ts.row_count as f64) {
                let num_workers = parallel_plan::compute_worker_count(ts.page_count);
                if num_workers > 1 {
                    let mut parallel_cost = self.cost_model.cost_parallel_scan(ts, num_workers);
                    // Apply predicate selectivity to parallel scan row count
                    if predicate.is_some() {
                        parallel_cost.row_count = seq_cost.row_count;
                    }
                    // Gather node adds minimal coordination cost (parallel_tuple_cost
                    // is already included in cost_parallel_scan)
                    let gather_cost = PlanCost {
                        io_cost: 0.0,
                        cpu_cost: 0.0,
                        row_count: parallel_cost.row_count,
                    };
                    let total_parallel = parallel_cost.total() + gather_cost.total();
                    if total_parallel < seq_cost.total() {
                        let parallel_scan = PhysicalPlan::ParallelSeqScan {
                            table_id,
                            columns: columns.clone(),
                            predicate: predicate.clone(),
                            num_workers,
                            cost: parallel_cost,
                        };
                        return Ok(PhysicalPlan::Gather {
                            child: Box::new(parallel_scan),
                            num_workers,
                            cost: gather_cost,
                        });
                    }
                }
            }
        }

        // Consider encoding-aware scan cost if hints are present
        if let Some(hints) = &encoding_hints {
            if hints.any_applicable() {
                if let Some((ts, _)) = &table_stats {
                    let skip_rate = hints.estimated_skip_rate();
                    let params = EncodingCostParameters {
                        skip_rate,
                        ..EncodingCostParameters::default()
                    };
                    let encoded_cost = self.cost_model.cost_encoded_scan(ts, &params);
                    // If encoded scan is cheaper, adjust the sequential scan cost
                    if encoded_cost.total() < seq_cost.total() {
                        return Ok(PhysicalPlan::SeqScan {
                            table_id,
                            columns,
                            predicate,
                            cost: encoded_cost,
                            as_of: as_of.clone(),
                        });
                    }
                }
            }
        }

        Ok(PhysicalPlan::SeqScan {
            table_id,
            columns,
            predicate,
            cost: seq_cost,
            as_of,
        })
    }

    // -----------------------------------------------------------------------
    // Join planning: Hash vs Merge vs Nested Loop
    // -----------------------------------------------------------------------

    fn plan_join(
        &self,
        left: LogicalPlan,
        right: LogicalPlan,
        join_type: JoinType,
        condition: JoinCondition,
    ) -> Result<PhysicalPlan> {
        let left_plan = self.plan(left)?;
        let right_plan = self.plan(right)?;
        let left_cost = *left_plan.cost();
        let right_cost = *right_plan.cost();

        match &condition {
            JoinCondition::On(expr) => {
                // Try to extract equi-join keys
                if let Some((left_keys, right_keys, remaining)) = extract_equi_keys(expr) {
                    // Cost all three strategies
                    let hash_cost = self.cost_model.cost_hash_join(&left_cost, &right_cost);
                    let merge_cost_base = self.cost_model.cost_merge_join(&left_cost, &right_cost);
                    let nl_cost = self
                        .cost_model
                        .cost_nested_loop_join(&left_cost, &right_cost);

                    // Add sort cost to merge join if needed
                    let left_sort_cost = self.cost_model.cost_sort(&left_cost);
                    let right_sort_cost = self.cost_model.cost_sort(&right_cost);
                    let merge_total =
                        merge_cost_base.total() + left_sort_cost.total() + right_sort_cost.total();

                    // Pick cheapest
                    if nl_cost.total() < hash_cost.total()
                        && nl_cost.total() < merge_total
                        && right_cost.row_count < 100.0
                    {
                        // Nested loop for small right side
                        Ok(PhysicalPlan::NestedLoopJoin {
                            left: Box::new(left_plan),
                            right: Box::new(right_plan),
                            join_type,
                            condition: Some(expr.clone()),
                            cost: nl_cost,
                        })
                    } else if merge_total < hash_cost.total() {
                        // Merge join
                        Ok(PhysicalPlan::MergeJoin {
                            left: Box::new(left_plan),
                            right: Box::new(right_plan),
                            join_type,
                            left_keys,
                            right_keys,
                            cost: merge_cost_base,
                        })
                    } else {
                        // Consider parallel hash join for large inputs
                        if parallel_plan::should_parallelize(left_cost.row_count)
                            || parallel_plan::should_parallelize(right_cost.row_count)
                        {
                            // Approximate page count from row counts, clamped to u32 range
                            let approx_pages =
                                ((left_cost.row_count + right_cost.row_count) / 100.0)
                                    .min(u32::MAX as f64) as u32;
                            let num_workers = parallel_plan::compute_worker_count(approx_pages);
                            if num_workers > 1 {
                                let parallel_cost = self.cost_model.cost_parallel_hash_join(
                                    &left_cost,
                                    &right_cost,
                                    num_workers,
                                );
                                if parallel_cost.total() < hash_cost.total() {
                                    let gather_cost = PlanCost {
                                        io_cost: 0.0,
                                        cpu_cost: 0.0,
                                        row_count: parallel_cost.row_count,
                                    };
                                    let par_join = PhysicalPlan::ParallelHashJoin {
                                        left: Box::new(left_plan),
                                        right: Box::new(right_plan),
                                        join_type,
                                        left_keys,
                                        right_keys,
                                        remaining_condition: remaining,
                                        num_workers,
                                        cost: parallel_cost,
                                    };
                                    return Ok(PhysicalPlan::Gather {
                                        child: Box::new(par_join),
                                        num_workers,
                                        cost: gather_cost,
                                    });
                                }
                            }
                        }

                        // Hash join (default for equi-joins)
                        Ok(PhysicalPlan::HashJoin {
                            left: Box::new(left_plan),
                            right: Box::new(right_plan),
                            join_type,
                            left_keys,
                            right_keys,
                            remaining_condition: remaining,
                            cost: hash_cost,
                        })
                    }
                } else {
                    // Non-equi join: must use nested loop
                    let cost = self
                        .cost_model
                        .cost_nested_loop_join(&left_cost, &right_cost);
                    Ok(PhysicalPlan::NestedLoopJoin {
                        left: Box::new(left_plan),
                        right: Box::new(right_plan),
                        join_type,
                        condition: Some(expr.clone()),
                        cost,
                    })
                }
            }
            JoinCondition::Using(_) | JoinCondition::Natural => {
                // Treat as hash join with the condition being equality on shared columns
                let cost = self.cost_model.cost_hash_join(&left_cost, &right_cost);
                Ok(PhysicalPlan::HashJoin {
                    left: Box::new(left_plan),
                    right: Box::new(right_plan),
                    join_type,
                    left_keys: vec![],
                    right_keys: vec![],
                    remaining_condition: None,
                    cost,
                })
            }
            JoinCondition::Cross => {
                let cost = self
                    .cost_model
                    .cost_nested_loop_join(&left_cost, &right_cost);
                Ok(PhysicalPlan::NestedLoopJoin {
                    left: Box::new(left_plan),
                    right: Box::new(right_plan),
                    join_type,
                    condition: None,
                    cost,
                })
            }
        }
    }

    // -----------------------------------------------------------------------
    // Aggregate planning: Hash vs Sort
    // -----------------------------------------------------------------------

    fn plan_aggregate(
        &self,
        group_by: Vec<BoundExpr>,
        aggregates: Vec<crate::logical::AggregateExpr>,
        child: LogicalPlan,
    ) -> Result<PhysicalPlan> {
        let child_plan = self.plan(child)?;
        let child_cost = *child_plan.cost();

        let group_count = if group_by.is_empty() {
            1.0
        } else {
            child_cost.row_count.sqrt().max(1.0)
        };

        let cost = self
            .cost_model
            .cost_hash_aggregate(&child_cost, group_count);

        // Use HashAggregate by default (better for random group distributions)
        Ok(PhysicalPlan::HashAggregate {
            group_by,
            aggregates,
            child: Box::new(child_plan),
            cost: PlanCost {
                io_cost: 0.0,
                cpu_cost: cost.cpu_cost - child_cost.cpu_cost,
                row_count: group_count,
            },
        })
    }
}

// ---------------------------------------------------------------------------
// Index matching
// ---------------------------------------------------------------------------

/// Checks if a predicate matches an index's leading column(s).
/// Returns (index_predicate, remaining_predicate) if a match is found.
fn match_index(
    predicate: &BoundExpr,
    index: &IndexEntry,
) -> Option<(BoundExpr, Option<BoundExpr>)> {
    if index.columns.is_empty() {
        return None;
    }

    let leading_col = index.columns[0].column_id;

    // Check if the predicate references the leading index column
    match predicate {
        BoundExpr::BinaryOp {
            left,
            op:
                BinaryOperator::Eq
                | BinaryOperator::Lt
                | BinaryOperator::Gt
                | BinaryOperator::LtEq
                | BinaryOperator::GtEq,
            right,
            ..
        } => {
            let left_col = extract_column_id_from_expr(left);
            let right_col = extract_column_id_from_expr(right);

            if left_col == Some(leading_col) || right_col == Some(leading_col) {
                return Some((predicate.clone(), None));
            }
            None
        }
        // AND: check if any conjunct matches the index
        BoundExpr::BinaryOp {
            left,
            op: BinaryOperator::And,
            right,
            ..
        } => {
            let left_match = match_index(left, index);
            let right_match = match_index(right, index);

            match (left_match, right_match) {
                (Some((l_pred, _)), Some((r_pred, _))) => {
                    // Both sides match: combine as the index predicate
                    Some((
                        BoundExpr::BinaryOp {
                            left: Box::new(l_pred),
                            op: BinaryOperator::And,
                            right: Box::new(r_pred),
                            type_id: TypeId::Boolean,
                        },
                        None,
                    ))
                }
                (Some((idx_pred, _)), None) => Some((idx_pred, Some(right.as_ref().clone()))),
                (None, Some((idx_pred, _))) => Some((idx_pred, Some(left.as_ref().clone()))),
                (None, None) => None,
            }
        }
        BoundExpr::Between {
            expr,
            negated: false,
            ..
        } => {
            if extract_column_id_from_expr(expr) == Some(leading_col) {
                Some((predicate.clone(), None))
            } else {
                None
            }
        }
        BoundExpr::InList {
            expr,
            negated: false,
            ..
        } => {
            if extract_column_id_from_expr(expr) == Some(leading_col) {
                Some((predicate.clone(), None))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Extracts a match_against function call from a predicate tree.
/// Returns the FTS expression and any remaining non-FTS predicate.
fn extract_match_against(predicate: &BoundExpr) -> Option<(&BoundExpr, Option<&BoundExpr>)> {
    match predicate {
        BoundExpr::Function { name, .. } if name == "match_against" => Some((predicate, None)),
        BoundExpr::BinaryOp {
            left,
            op: BinaryOperator::And,
            right,
            ..
        } => {
            if let Some((fts, _)) = extract_match_against(left) {
                return Some((fts, Some(right)));
            }
            if let Some((fts, _)) = extract_match_against(right) {
                return Some((fts, Some(left)));
            }
            None
        }
        _ => None,
    }
}

/// Extracts a vector_distance_* function call from a predicate tree.
/// Returns the vector distance expression and any remaining non-vector predicate.
fn extract_vector_distance(predicate: &BoundExpr) -> Option<(&BoundExpr, Option<&BoundExpr>)> {
    match predicate {
        BoundExpr::Function { name, .. } if name.starts_with("vector_distance_") => {
            Some((predicate, None))
        }
        BoundExpr::BinaryOp {
            left,
            op: BinaryOperator::And,
            right,
            ..
        } => {
            if let Some((vec_expr, _)) = extract_vector_distance(left) {
                return Some((vec_expr, Some(right)));
            }
            if let Some((vec_expr, _)) = extract_vector_distance(right) {
                return Some((vec_expr, Some(left)));
            }
            None
        }
        _ => None,
    }
}

fn extract_column_id_from_expr(expr: &BoundExpr) -> Option<zyron_catalog::ColumnId> {
    match expr {
        BoundExpr::ColumnRef(cr) => Some(cr.column_id),
        BoundExpr::Nested(inner) => extract_column_id_from_expr(inner),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Equi-join key extraction
// ---------------------------------------------------------------------------

/// Extracts equi-join keys from a conjunction.
/// Given `a.x = b.y AND a.z = b.w AND a.q > 5`,
/// returns (vec![a.x, a.z], vec![b.y, b.w], Some(a.q > 5)).
fn extract_equi_keys(
    expr: &BoundExpr,
) -> Option<(Vec<BoundExpr>, Vec<BoundExpr>, Option<BoundExpr>)> {
    let conjuncts = split_conjuncts(expr);
    let mut left_keys = Vec::new();
    let mut right_keys = Vec::new();
    let mut remaining = Vec::new();

    for conj in conjuncts {
        if let BoundExpr::BinaryOp {
            left,
            op: BinaryOperator::Eq,
            right,
            ..
        } = &conj
        {
            if is_column_ref(left) && is_column_ref(right) {
                left_keys.push(left.as_ref().clone());
                right_keys.push(right.as_ref().clone());
                continue;
            }
        }
        remaining.push(conj);
    }

    if left_keys.is_empty() {
        return None;
    }

    let remaining_expr = if remaining.is_empty() {
        None
    } else {
        Some(combine_conjuncts(remaining))
    };

    Some((left_keys, right_keys, remaining_expr))
}

fn split_conjuncts(expr: &BoundExpr) -> Vec<BoundExpr> {
    match expr {
        BoundExpr::BinaryOp {
            left,
            op: BinaryOperator::And,
            right,
            ..
        } => {
            let mut result = split_conjuncts(left);
            result.extend(split_conjuncts(right));
            result
        }
        other => vec![other.clone()],
    }
}

fn combine_conjuncts(mut conjuncts: Vec<BoundExpr>) -> BoundExpr {
    if conjuncts.len() == 1 {
        return conjuncts.remove(0);
    }
    let mut result = conjuncts.remove(0);
    for conj in conjuncts {
        result = BoundExpr::BinaryOp {
            left: Box::new(result),
            op: BinaryOperator::And,
            right: Box::new(conj),
            type_id: TypeId::Boolean,
        };
    }
    result
}

fn is_column_ref(expr: &BoundExpr) -> bool {
    matches!(expr, BoundExpr::ColumnRef(_))
}

/// Walks a BoundExpr tree and replaces WindowFunction nodes with ColumnRefs
/// pointing to positions in an auxiliary window-output column list.
/// Collects the WindowFunction expressions into `collected`.
fn rewrite_window_refs(
    expr: &BoundExpr,
    collected: &mut Vec<BoundExpr>,
    names: &mut Vec<String>,
    input_schema_len: usize,
    alias_hint: Option<String>,
) -> BoundExpr {
    use crate::binder::{BoundExpr as BE, BoundWhen, ColumnRef};
    use zyron_catalog::ColumnId;

    match expr {
        BE::WindowFunction { type_id, .. } => {
            let idx = collected.len();
            collected.push(expr.clone());
            let name = alias_hint.unwrap_or_else(|| format!("window_{}", idx));
            names.push(name);
            // Window output columns are appended after input_schema_len input columns.
            // Column indices in the batch are 0..N, but ColumnRef is by (table_idx, column_id).
            // The executor's expr resolver uses column_id to index into schema. Window
            // columns in the Window output schema have column_id = (input_schema_len + idx).
            BE::ColumnRef(ColumnRef {
                table_idx: 0,
                column_id: ColumnId((input_schema_len + idx) as u16),
                type_id: *type_id,
                nullable: true,
            })
        }
        BE::ColumnRef(_) | BE::Literal { .. } | BE::Parameter { .. } => expr.clone(),
        BE::BinaryOp {
            left,
            op,
            right,
            type_id,
        } => BE::BinaryOp {
            left: Box::new(rewrite_window_refs(
                left,
                collected,
                names,
                input_schema_len,
                None,
            )),
            op: op.clone(),
            right: Box::new(rewrite_window_refs(
                right,
                collected,
                names,
                input_schema_len,
                None,
            )),
            type_id: *type_id,
        },
        BE::UnaryOp {
            op,
            expr: inner,
            type_id,
        } => BE::UnaryOp {
            op: op.clone(),
            expr: Box::new(rewrite_window_refs(
                inner,
                collected,
                names,
                input_schema_len,
                None,
            )),
            type_id: *type_id,
        },
        BE::IsNull {
            expr: inner,
            negated,
        } => BE::IsNull {
            expr: Box::new(rewrite_window_refs(
                inner,
                collected,
                names,
                input_schema_len,
                None,
            )),
            negated: *negated,
        },
        BE::InList {
            expr: inner,
            list,
            negated,
        } => BE::InList {
            expr: Box::new(rewrite_window_refs(
                inner,
                collected,
                names,
                input_schema_len,
                None,
            )),
            list: list
                .iter()
                .map(|e| rewrite_window_refs(e, collected, names, input_schema_len, None))
                .collect(),
            negated: *negated,
        },
        BE::Between {
            expr: inner,
            low,
            high,
            negated,
        } => BE::Between {
            expr: Box::new(rewrite_window_refs(
                inner,
                collected,
                names,
                input_schema_len,
                None,
            )),
            low: Box::new(rewrite_window_refs(
                low,
                collected,
                names,
                input_schema_len,
                None,
            )),
            high: Box::new(rewrite_window_refs(
                high,
                collected,
                names,
                input_schema_len,
                None,
            )),
            negated: *negated,
        },
        BE::Like {
            expr: inner,
            pattern,
            negated,
        } => BE::Like {
            expr: Box::new(rewrite_window_refs(
                inner,
                collected,
                names,
                input_schema_len,
                None,
            )),
            pattern: Box::new(rewrite_window_refs(
                pattern,
                collected,
                names,
                input_schema_len,
                None,
            )),
            negated: *negated,
        },
        BE::ILike {
            expr: inner,
            pattern,
            negated,
        } => BE::ILike {
            expr: Box::new(rewrite_window_refs(
                inner,
                collected,
                names,
                input_schema_len,
                None,
            )),
            pattern: Box::new(rewrite_window_refs(
                pattern,
                collected,
                names,
                input_schema_len,
                None,
            )),
            negated: *negated,
        },
        BE::Function {
            name,
            args,
            return_type,
            distinct,
        } => BE::Function {
            name: name.clone(),
            args: args
                .iter()
                .map(|a| rewrite_window_refs(a, collected, names, input_schema_len, None))
                .collect(),
            return_type: *return_type,
            distinct: *distinct,
        },
        BE::AggregateFunction {
            name,
            args,
            distinct,
            return_type,
        } => BE::AggregateFunction {
            name: name.clone(),
            args: args
                .iter()
                .map(|a| rewrite_window_refs(a, collected, names, input_schema_len, None))
                .collect(),
            distinct: *distinct,
            return_type: *return_type,
        },
        BE::Cast {
            expr: inner,
            target_type,
        } => BE::Cast {
            expr: Box::new(rewrite_window_refs(
                inner,
                collected,
                names,
                input_schema_len,
                None,
            )),
            target_type: *target_type,
        },
        BE::Case {
            operand,
            conditions,
            else_result,
            type_id,
        } => BE::Case {
            operand: operand.as_ref().map(|o| {
                Box::new(rewrite_window_refs(
                    o,
                    collected,
                    names,
                    input_schema_len,
                    None,
                ))
            }),
            conditions: conditions
                .iter()
                .map(|w| BoundWhen {
                    condition: rewrite_window_refs(
                        &w.condition,
                        collected,
                        names,
                        input_schema_len,
                        None,
                    ),
                    result: rewrite_window_refs(
                        &w.result,
                        collected,
                        names,
                        input_schema_len,
                        None,
                    ),
                })
                .collect(),
            else_result: else_result.as_ref().map(|e| {
                Box::new(rewrite_window_refs(
                    e,
                    collected,
                    names,
                    input_schema_len,
                    None,
                ))
            }),
            type_id: *type_id,
        },
        BE::Nested(inner) => BE::Nested(Box::new(rewrite_window_refs(
            inner,
            collected,
            names,
            input_schema_len,
            None,
        ))),
        // Subqueries don't participate in window rewriting at this level.
        BE::Subquery { .. } | BE::Exists { .. } | BE::InSubquery { .. } => expr.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binder::ColumnRef;
    use zyron_catalog::ColumnId;

    #[test]
    fn test_extract_equi_keys() {
        let left_col = BoundExpr::ColumnRef(ColumnRef {
            table_idx: 0,
            column_id: ColumnId(0),
            type_id: TypeId::Int64,
            nullable: false,
        });
        let right_col = BoundExpr::ColumnRef(ColumnRef {
            table_idx: 1,
            column_id: ColumnId(0),
            type_id: TypeId::Int64,
            nullable: false,
        });
        let eq = BoundExpr::BinaryOp {
            left: Box::new(left_col.clone()),
            op: BinaryOperator::Eq,
            right: Box::new(right_col.clone()),
            type_id: TypeId::Boolean,
        };

        let result = extract_equi_keys(&eq);
        assert!(result.is_some());
        let (lk, rk, rem) = result.unwrap();
        assert_eq!(lk.len(), 1);
        assert_eq!(rk.len(), 1);
        assert!(rem.is_none());
    }

    #[test]
    fn test_extract_equi_keys_with_remaining() {
        let left_col = BoundExpr::ColumnRef(ColumnRef {
            table_idx: 0,
            column_id: ColumnId(0),
            type_id: TypeId::Int64,
            nullable: false,
        });
        let right_col = BoundExpr::ColumnRef(ColumnRef {
            table_idx: 1,
            column_id: ColumnId(0),
            type_id: TypeId::Int64,
            nullable: false,
        });
        let eq = BoundExpr::BinaryOp {
            left: Box::new(left_col.clone()),
            op: BinaryOperator::Eq,
            right: Box::new(right_col.clone()),
            type_id: TypeId::Boolean,
        };
        let extra = BoundExpr::BinaryOp {
            left: Box::new(left_col.clone()),
            op: BinaryOperator::Gt,
            right: Box::new(BoundExpr::Literal {
                value: zyron_parser::ast::LiteralValue::Integer(5),
                type_id: TypeId::Int64,
            }),
            type_id: TypeId::Boolean,
        };
        let combined = BoundExpr::BinaryOp {
            left: Box::new(eq),
            op: BinaryOperator::And,
            right: Box::new(extra),
            type_id: TypeId::Boolean,
        };

        let result = extract_equi_keys(&combined);
        assert!(result.is_some());
        let (lk, rk, rem) = result.unwrap();
        assert_eq!(lk.len(), 1);
        assert_eq!(rk.len(), 1);
        assert!(rem.is_some());
    }

    #[test]
    fn test_no_equi_keys() {
        let expr = BoundExpr::BinaryOp {
            left: Box::new(BoundExpr::ColumnRef(ColumnRef {
                table_idx: 0,
                column_id: ColumnId(0),
                type_id: TypeId::Int64,
                nullable: false,
            })),
            op: BinaryOperator::Gt,
            right: Box::new(BoundExpr::Literal {
                value: zyron_parser::ast::LiteralValue::Integer(5),
                type_id: TypeId::Int64,
            }),
            type_id: TypeId::Boolean,
        };
        assert!(extract_equi_keys(&expr).is_none());
    }
}

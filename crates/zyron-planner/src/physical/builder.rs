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
use zyron_common::{Result, TypeId, ZyronError};
use zyron_parser::ast::{BinaryOperator, JoinType};

/// Converts an optimized logical plan into a physical plan using cost-based decisions.
///
/// `peers` is this node's view of the mesh, needed to cost a scan of a table
/// that lives on one. It is passed rather than looked up, because peer
/// membership is node-local state the catalog deliberately does not hold,
/// and a planner reaching for a global would make two plans of the same
/// query differ by which node built them without saying so. None costs
/// every peer as unknown, which is the conservative estimate.
pub fn build_physical_plan(
    logical: LogicalPlan,
    catalog: &Catalog,
    peers: Option<&zyron_common::PeerRegistry>,
) -> Result<PhysicalPlan> {
    let cost_model = CostModel::default();
    PhysicalPlanner::new(catalog, cost_model, peers).plan(logical)
}

struct PhysicalPlanner<'a> {
    catalog: &'a Catalog,
    cost_model: CostModel,
    peers: Option<&'a zyron_common::PeerRegistry>,
}

impl<'a> PhysicalPlanner<'a> {
    fn new(
        catalog: &'a Catalog,
        cost_model: CostModel,
        peers: Option<&'a zyron_common::PeerRegistry>,
    ) -> Self {
        Self {
            catalog,
            cost_model,
            peers,
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
            } => self.plan_scan(table_id, columns, None, encoding_hints, as_of, None),
            LogicalPlan::Filter { predicate, child } => {
                // Try to push the filter into a scan (index scan opportunity).
                // Probe by reference, then take ownership only on the matching
                // path so the shared Arc child is moved out at most once.
                if matches!(child.as_ref(), LogicalPlan::Scan { .. }) {
                    if let LogicalPlan::Scan {
                        table_id,
                        columns,
                        encoding_hints,
                        as_of,
                        ..
                    } = Arc::unwrap_or_clone(child)
                    {
                        // A subquery in the predicate is evaluated by a Filter
                        // operator, not the scan: a correlated subquery needs the
                        // outer row the scan does not carry, and an uncorrelated
                        // one is folded at the Filter. Split the conjuncts so
                        // subquery-free ones still push into the scan and only the
                        // subquery-bearing ones stay above it.
                        let conjuncts = split_conjuncts(&predicate);
                        let (sub_conjuncts, simple_conjuncts): (Vec<_>, Vec<_>) =
                            conjuncts.into_iter().partition(predicate_has_subquery);
                        if sub_conjuncts.is_empty() {
                            return self.plan_scan(
                                table_id,
                                columns,
                                Some(predicate),
                                encoding_hints,
                                as_of,
                                None,
                            );
                        }
                        let scan_predicate = if simple_conjuncts.is_empty() {
                            None
                        } else {
                            Some(combine_conjuncts(simple_conjuncts))
                        };
                        let scan = self.plan_scan(
                            table_id,
                            columns,
                            scan_predicate,
                            encoding_hints,
                            as_of,
                            None,
                        )?;
                        let scan_cost = *scan.cost();
                        let filter_predicate = combine_conjuncts(sub_conjuncts);
                        let selectivity =
                            self.cost_model
                                .estimate_selectivity(&filter_predicate, None, None);
                        let cost = PlanCost {
                            io_cost: 0.0,
                            cpu_cost: scan_cost.row_count * self.cost_model.cpu_operator_cost,
                            row_count: (scan_cost.row_count * selectivity).max(1.0),
                        };
                        return Ok(PhysicalPlan::Filter {
                            predicate: filter_predicate,
                            child: Box::new(scan),
                            cost,
                        });
                    }
                    unreachable!("child matched Scan above");
                }

                let child_plan = self.plan(Arc::unwrap_or_clone(child))?;
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
                output_table_idx,
            } => {
                let child_plan = self.plan(Arc::unwrap_or_clone(child))?;
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

                let rewritten: Vec<crate::binder::BoundExpr> = expressions
                    .iter()
                    .enumerate()
                    .map(|(i, e)| {
                        rewrite_window_refs(
                            e,
                            &mut window_exprs,
                            &mut window_names,
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
                        output_table_idx,
                    });
                }

                Ok(PhysicalPlan::Project {
                    expressions,
                    aliases,
                    child: Box::new(child_plan),
                    cost,
                    output_table_idx,
                })
            }
            LogicalPlan::Join {
                left,
                right,
                join_type,
                condition,
            } => self.plan_join(
                Arc::unwrap_or_clone(left),
                Arc::unwrap_or_clone(right),
                join_type,
                condition,
            ),
            LogicalPlan::LateralJoin {
                left,
                subquery,
                subquery_table_idx,
                join_type,
                condition,
            } => {
                let left_plan = self.plan(Arc::unwrap_or_clone(left))?;
                let left_cost = *left_plan.cost();
                let left_schema = left_plan.output_schema();
                // The right schema is the lateral output relabeled under the
                // subquery's table index, NULLable when the join is a LEFT join.
                let force_nullable = matches!(join_type, JoinType::Left | JoinType::Full);
                let right_schema: Vec<LogicalColumn> = subquery
                    .0
                    .output_schema
                    .iter()
                    .enumerate()
                    .map(|(i, col)| LogicalColumn {
                        table_idx: Some(subquery_table_idx),
                        column_id: zyron_catalog::ColumnId(i as u16),
                        name: col.name.clone(),
                        type_id: col.type_id,
                        nullable: col.nullable || force_nullable,
                        ts_precision: col.ts_precision,
                    })
                    .collect();
                let cost = PlanCost {
                    io_cost: left_cost.io_cost,
                    cpu_cost: left_cost.cpu_cost
                        + left_cost.row_count * self.cost_model.cpu_operator_cost * 4.0,
                    row_count: left_cost.row_count,
                };
                Ok(PhysicalPlan::LateralJoin {
                    left: Box::new(left_plan),
                    subquery: *subquery.0,
                    subquery_table_idx,
                    join_type,
                    condition,
                    left_schema,
                    right_schema,
                    cost,
                })
            }
            LogicalPlan::Aggregate {
                group_by,
                aggregates,
                child,
            } => self.plan_aggregate(group_by, aggregates, Arc::unwrap_or_clone(child)),
            LogicalPlan::Sort { order_by, child } => {
                let child_plan = self.plan(Arc::unwrap_or_clone(child))?;
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
                // When the limit sits directly over a scan (optionally through a
                // Filter), thread k into the scan so a vector-distance predicate
                // becomes a real KNN VectorScan. When it sits over a
                // Sort(ST_Distance) the three nodes collapse into a spatial KNN
                // scan. Otherwise the child plans normally.
                let limit_k = limit.map(|l| l as usize);
                let scan_child = if offset.unwrap_or(0) == 0 {
                    self.try_plan_knn_under_limit(child.as_ref(), limit_k)?
                } else {
                    None
                };
                let mut child_plan = match scan_child {
                    Some(plan) => plan,
                    None => self.plan(Arc::unwrap_or_clone(child))?,
                };
                // A foreign scan can serve the cap itself. Every row the
                // offset skips still has to arrive, so the peer is asked for
                // both
                if let Some(rows) = limit {
                    let cap = (rows as usize).saturating_add(offset.unwrap_or(0) as usize);
                    push_limit_to_foreign_scan(&mut child_plan, cap);
                }
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
                let child_plan = self.plan(Arc::unwrap_or_clone(child))?;
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
            LogicalPlan::LockRows {
                table_id,
                mode,
                wait,
                cap,
                child,
            } => {
                let child_plan = self.plan(Arc::unwrap_or_clone(child))?;
                let child_cost = *child_plan.cost();
                let rows = cap
                    .map(|c| (c as f64).min(child_cost.row_count))
                    .unwrap_or(child_cost.row_count);
                // one lock table insert per emitted row
                let cost = PlanCost {
                    io_cost: 0.0,
                    cpu_cost: rows * self.cost_model.cpu_operator_cost,
                    row_count: rows,
                };
                Ok(PhysicalPlan::LockRows {
                    table_id,
                    mode,
                    wait,
                    cap,
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
                let left_plan = self.plan(Arc::unwrap_or_clone(left))?;
                let right_plan = self.plan(Arc::unwrap_or_clone(right))?;
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
                column_defaults,
                check_constraints,
                expectations,
                source,
            } => {
                let source_plan = self.plan(Arc::unwrap_or_clone(source))?;
                let cost = *source_plan.cost();
                Ok(PhysicalPlan::Insert {
                    table_id,
                    target_columns,
                    column_defaults,
                    check_constraints,
                    expectations,
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
                check_constraints,
                child,
            } => {
                let child_plan = self.plan(Arc::unwrap_or_clone(child))?;
                if let Some(lake) = self.plan_lake_update(
                    table_id,
                    assignments.clone(),
                    check_constraints.clone(),
                    &child_plan,
                )? {
                    return Ok(lake);
                }
                let cost = *child_plan.cost();
                Ok(PhysicalPlan::Update {
                    table_id,
                    assignments,
                    check_constraints,
                    child: Box::new(child_plan),
                    cost,
                })
            }
            LogicalPlan::Delete { table_id, child } => {
                let child_plan = self.plan(Arc::unwrap_or_clone(child))?;
                if let Some(lake) = self.plan_lake_delete(table_id, &child_plan)? {
                    return Ok(lake);
                }
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
            LogicalPlan::AnalyticsTableFunction {
                function_name,
                named_args,
                positional_args,
                output_columns,
            } => {
                // Cost: a single scan over the input source plus per-row work
                // proportional to the function. Without source size info we
                // use a nominal 10k-row estimate so the optimizer has a value
                // to compare against.
                let nominal_rows: f64 = 10_000.0;
                let cost = PlanCost {
                    io_cost: nominal_rows,
                    cpu_cost: nominal_rows * 4.0,
                    row_count: match function_name.as_str() {
                        "DATA_PROFILE" | "COLUMN_PROFILE" => output_columns.len() as f64,
                        "CORRELATION_MATRIX" => positional_args.len().pow(2) as f64,
                        _ => nominal_rows,
                    },
                };
                Ok(PhysicalPlan::AnalyticsTableFunction {
                    function_name,
                    named_args,
                    positional_args,
                    output_columns,
                    cost,
                })
            }
        }
    }

    /// Recognizes a KNN-shaped plan directly under a LIMIT and, when matched,
    /// produces the scan with k threaded in. Two shapes are handled: a vector
    /// distance predicate (`Limit -> [Filter] -> Scan`) becomes a VectorScan
    /// serving k rows, and an `ORDER BY ST_Distance(col, q)` over a spatial
    /// index (`Limit -> Sort -> [Project] -> [Filter] -> Scan`) becomes a
    /// spatial KNN scan. Returns None when no pattern matches so the caller
    /// plans the child normally.
    fn try_plan_knn_under_limit(
        &self,
        child: &LogicalPlan,
        k: Option<usize>,
    ) -> Result<Option<PhysicalPlan>> {
        match child {
            // Vector distance predicate: thread k into the scan so it serves a
            // bounded nearest-neighbor search.
            LogicalPlan::Scan {
                table_id,
                columns,
                encoding_hints,
                as_of,
                ..
            } => {
                let plan = self.plan_scan(
                    *table_id,
                    columns.clone(),
                    None,
                    encoding_hints.clone(),
                    as_of.clone(),
                    k,
                )?;
                Ok(matches!(plan, PhysicalPlan::VectorScan { .. }).then_some(plan))
            }
            LogicalPlan::Filter { predicate, child } => {
                if let LogicalPlan::Scan {
                    table_id,
                    columns,
                    encoding_hints,
                    as_of,
                    ..
                } = child.as_ref()
                {
                    let plan = self.plan_scan(
                        *table_id,
                        columns.clone(),
                        Some(predicate.clone()),
                        encoding_hints.clone(),
                        as_of.clone(),
                        k,
                    )?;
                    return Ok(matches!(plan, PhysicalPlan::VectorScan { .. }).then_some(plan));
                }
                Ok(None)
            }
            // ORDER BY ST_Distance(col, q) over a spatial index.
            LogicalPlan::Sort { order_by, child } => {
                let Some(k) = k else { return Ok(None) };
                if order_by.len() != 1 || !order_by[0].asc {
                    return Ok(None);
                }
                let Some(query_point) = extract_st_distance_point(&order_by[0].expr) else {
                    return Ok(None);
                };
                // Peel an optional Project then optional Filter to reach the scan.
                let mut node = child.as_ref();
                if let LogicalPlan::Project { child, .. } = node {
                    node = child.as_ref();
                }
                let mut remaining_predicate = None;
                if let LogicalPlan::Filter { predicate, child } = node {
                    remaining_predicate = Some(predicate.clone());
                    node = child.as_ref();
                }
                let LogicalPlan::Scan {
                    table_id,
                    columns,
                    as_of,
                    ..
                } = node
                else {
                    return Ok(None);
                };
                // Same routing rule the predicate path uses: a lake table
                // answers a search at any version, and on the other stores a
                // time-travel read stays on the storage scan
                let searchable = as_of.is_none()
                    || self
                        .catalog
                        .get_table_by_id(*table_id)
                        .map(|te| te.lake.is_lake())
                        .unwrap_or(false);
                if !searchable {
                    return Ok(None);
                }
                for index in &self.catalog.get_indexes_for_table(*table_id) {
                    if index.index_type == zyron_catalog::IndexType::Spatial {
                        let cost = PlanCost {
                            io_cost: 2.0,
                            cpu_cost: 8.0,
                            row_count: k as f64,
                        };
                        return Ok(Some(PhysicalPlan::SpatialScan {
                            table_id: *table_id,
                            index_id: index.id,
                            columns: columns.clone(),
                            kind: SpatialScanKind::Knn { query_point, k },
                            remaining_predicate,
                            as_of: as_of.clone(),
                            cost,
                        }));
                    }
                }
                Ok(None)
            }
            _ => Ok(None),
        }
    }

    // -----------------------------------------------------------------------
    // Scan planning: SeqScan vs IndexScan
    // -----------------------------------------------------------------------

    /// Routes a DELETE on a lake table to the predicate-delete node.
    /// Returns None for a heap or columnar table, so the row-addressed
    /// delete path is untouched. A predicate with no exact lake
    /// equivalent is refused rather than approximated
    fn plan_lake_delete(
        &self,
        table_id: zyron_catalog::TableId,
        child: &PhysicalPlan,
    ) -> Result<Option<PhysicalPlan>> {
        let Ok(te) = self.catalog.get_table_by_id(table_id) else {
            return Ok(None);
        };
        if !te.lake.is_lake() {
            return Ok(None);
        }
        let (bound, lowered) = self.lake_dml_predicate(&te, child, "DELETE")?;
        let _ = bound;
        let sql = lowered
            .as_ref()
            .map(|p| crate::lake_predicate::render_sql(p, &te.columns))
            .unwrap_or_else(|| "TRUE".to_string());
        Ok(Some(PhysicalPlan::LakeDelete {
            table_id,
            predicate: lowered,
            sql,
            cost: *child.cost(),
        }))
    }

    /// Extracts the row-selecting predicate from a lake DML child plan and
    /// lowers it. Returns the bound form, for a scan that must reproduce
    /// exactly the same rows, alongside the lake form that records the
    /// mutation. A predicate with no exact lake equivalent is refused
    fn lake_dml_predicate(
        &self,
        te: &zyron_catalog::TableEntry,
        child: &PhysicalPlan,
        what: &str,
    ) -> Result<(Option<BoundExpr>, Option<zyron_lake::LakePredicate>)> {
        // The child is the table's own scan, optionally under one residual
        // filter the scan could not absorb
        let (scan_predicate, residual) = match child {
            PhysicalPlan::LakeScan { predicate, .. } => (predicate.as_ref(), None),
            PhysicalPlan::Filter {
                child: inner,
                predicate,
                ..
            } => match inner.as_ref() {
                PhysicalPlan::LakeScan {
                    predicate: scan_pred,
                    ..
                } => (scan_pred.as_ref(), Some(predicate)),
                _ => {
                    return Err(ZyronError::PlanError(format!(
                        "{} on lake table \"{}\" requires a predicate over its own columns",
                        what, te.name
                    )));
                }
            },
            _ => {
                return Err(ZyronError::PlanError(format!(
                    "{} on lake table \"{}\" requires a predicate over its own columns",
                    what, te.name
                )));
            }
        };

        let mut bound: Option<BoundExpr> = None;
        let mut lowered: Option<zyron_lake::LakePredicate> = None;
        for part in [scan_predicate, residual].into_iter().flatten() {
            let one =
                crate::lake_predicate::lower_predicate(part, &te.columns).ok_or_else(|| {
                    ZyronError::PlanError(format!(
                        "{} on lake table \"{}\" needs a predicate the lake format can record, \
                     this one uses a construct with no exact equivalent",
                        what, te.name
                    ))
                })?;
            lowered = Some(match lowered {
                Some(existing) => zyron_lake::LakePredicate::And(vec![existing, one]),
                None => one,
            });
            bound = Some(match bound {
                Some(existing) => BoundExpr::BinaryOp {
                    left: Box::new(existing),
                    op: zyron_parser::ast::BinaryOperator::And,
                    right: Box::new(part.clone()),
                    type_id: TypeId::Boolean,
                },
                None => part.clone(),
            });
        }
        Ok((bound, lowered))
    }

    /// Routes an UPDATE on a lake table to the read-assign-replace node.
    /// Returns None for a heap or columnar table
    fn plan_lake_update(
        &self,
        table_id: zyron_catalog::TableId,
        assignments: Vec<crate::binder::BoundAssignment>,
        check_constraints: Vec<BoundExpr>,
        child: &PhysicalPlan,
    ) -> Result<Option<PhysicalPlan>> {
        let Ok(te) = self.catalog.get_table_by_id(table_id) else {
            return Ok(None);
        };
        if !te.lake.is_lake() {
            return Ok(None);
        }
        let (bound, lowered) = self.lake_dml_predicate(&te, child, "UPDATE")?;
        let sql = lowered
            .as_ref()
            .map(|p| crate::lake_predicate::render_sql(p, &te.columns))
            .unwrap_or_else(|| "TRUE".to_string());

        // The new row image needs every column, so the scan is rebuilt
        // over the full projection rather than the narrower one the
        // logical plan asked for
        let table_idx = match child {
            PhysicalPlan::LakeScan { columns, .. } => {
                columns.first().and_then(|c| c.table_idx).unwrap_or(0)
            }
            _ => 0,
        };
        let columns: Vec<crate::logical::LogicalColumn> = te
            .columns
            .iter()
            .map(|c| crate::logical::LogicalColumn {
                table_idx: Some(table_idx),
                column_id: c.id,
                name: c.name.clone(),
                type_id: c.type_id,
                nullable: c.nullable,
                ts_precision: c.ts_precision,
            })
            .collect();
        let cost = *child.cost();
        let scan = PhysicalPlan::LakeScan {
            table_id,
            columns,
            predicate: bound,
            lowered: lowered.clone(),
            as_of: None,
            cost,
        };
        Ok(Some(PhysicalPlan::LakeUpdate {
            table_id,
            assignments,
            check_constraints,
            predicate: lowered,
            sql,
            child: Box::new(scan),
            cost,
        }))
    }

    /// Builds the scan of a table that lives on a peer.
    ///
    /// The predicate is split rather than chosen between: each conjunct that
    /// renders faithfully as SQL goes to the remote, and each one that does
    /// not stays here as a residual. Splitting is what makes a filter like
    /// `region = 'west' AND f(x)` push its cheap half instead of neither, and
    /// a conjunct never lands in both places, so no row is filtered twice.
    ///
    /// A LIMIT travels only when nothing is left to filter locally: a remote
    /// row cap applied before a residual would cut rows the residual was
    /// going to keep, which changes the answer rather than the cost.
    fn plan_foreign_scan(
        &self,
        te: &zyron_catalog::TableEntry,
        columns: Vec<crate::logical::LogicalColumn>,
        predicate: Option<BoundExpr>,
        as_of: Option<crate::logical::AsOfTarget>,
        limit: Option<usize>,
    ) -> Result<PhysicalPlan> {
        let Some((peer, remote_table)) = te.foreign.remote() else {
            return Err(ZyronError::PlanError(format!(
                "table \"{}\" is marked foreign but names no peer",
                te.name
            )));
        };
        // Time travel is a property of a table's own history, and this node
        // holds none for a table it does not store. Reporting that beats
        // returning the peer's current rows under an AS OF the peer never saw
        if as_of.is_some() {
            return Err(ZyronError::PlanError(format!(
                "AS OF on foreign table \"{}\" is not available, its history belongs to \
                 peer \"{}\" and this node holds none of it",
                te.name, peer
            )));
        }
        if columns.is_empty() {
            return Err(ZyronError::PlanError(format!(
                "scan of foreign table \"{}\" projects no column",
                te.name
            )));
        }

        // Only the projected columns cross the wire, named as the peer knows
        // them. A column the catalog declares but this plan does not read is
        // never fetched
        let mut names = Vec::with_capacity(columns.len());
        let mut types = Vec::with_capacity(columns.len());
        for column in &columns {
            let entry = te
                .columns
                .iter()
                .find(|c| c.id == column.column_id)
                .ok_or_else(|| {
                    ZyronError::PlanError(format!(
                        "foreign table \"{}\" has no column with id {}",
                        te.name, column.column_id.0
                    ))
                })?;
            names.push(entry.name.clone());
            types.push(entry.type_id);
        }

        // Conjunct by conjunct: what renders goes to the peer, what does not
        // stays here
        let (pushed, residual) = match &predicate {
            None => (None, None),
            Some(p) => {
                let (mut sendable, mut local) = (Vec::new(), Vec::new());
                for conjunct in split_conjuncts(p) {
                    match crate::bound_predicate_sql::bound_predicate_to_sql(&conjunct, &te.columns)
                    {
                        Some(sql) => sendable.push(sql),
                        None => local.push(conjunct),
                    }
                }
                (
                    (!sendable.is_empty()).then(|| sendable.join(" AND ")),
                    (!local.is_empty()).then(|| combine_conjuncts(local)),
                )
            }
        };

        let request = zyron_common::ForeignRequest {
            peer: peer.to_string(),
            table: remote_table.to_string(),
            columns: names,
            column_types: types,
            predicate: pushed,
            limit: residual.is_none().then_some(limit).flatten(),
        };

        // What the peer said it stores, which is what decides whether the
        // pushed filter saves it file reads or index walks
        let mode = self
            .peers
            .and_then(|registry| registry.get(peer))
            .and_then(|entry| entry.effective_mode());
        let stats = self.catalog.get_stats(te.id);
        let selectivity = match &predicate {
            None => 1.0,
            Some(p) => self.cost_model.estimate_selectivity(p, None, None),
        };
        let cost = match &stats {
            Some(s) => self.cost_model.cost_foreign_scan(
                mode,
                &s.0,
                selectivity,
                columns.len(),
                request.is_filtered(),
            ),
            // Nothing local was ever analyzed for a table this node does not
            // store, so the shape of the estimate comes from the request
            // rather than from statistics that do not exist
            None => PlanCost {
                io_cost: 100.0 + columns.len() as f64,
                cpu_cost: 10.0,
                row_count: 1000.0 * selectivity.clamp(0.0, 1.0).max(0.01),
            },
        };

        Ok(PhysicalPlan::ForeignScan {
            table_id: te.id,
            columns,
            residual,
            request,
            cost,
        })
    }

    fn plan_scan(
        &self,
        table_id: zyron_catalog::TableId,
        columns: Vec<crate::logical::LogicalColumn>,
        predicate: Option<BoundExpr>,
        encoding_hints: Option<encoding_pushdown::EncodingHint>,
        as_of: Option<crate::logical::AsOfTarget>,
        limit: Option<usize>,
    ) -> Result<PhysicalPlan> {
        // Get table stats
        let table_stats = self.catalog.get_stats(table_id);

        // Foreign routing gate, ahead of every local access path. A foreign
        // table has no rows here at all, so a heap, columnar, lake or index
        // node would read storage that was never written
        if let Ok(te) = self.catalog.get_table_by_id(table_id) {
            if te.foreign.is_foreign() {
                return self.plan_foreign_scan(&te, columns, predicate, as_of, limit);
            }
        }

        // Whether this table's rows live in a lake log rather than the heap.
        // Search routing below is shared by both, the lake gate after it is
        // what diverges
        let is_lake = self
            .catalog
            .get_table_by_id(table_id)
            .map(|te| te.lake.is_lake())
            .unwrap_or(false);

        // Get available indexes. Search predicates (FTS, vector, spatial)
        // route to their index operators BEFORE the lake gate and before
        // the hybrid columnar check: the search operators resolve heap,
        // columnar and lake hits through the document registry, so where a
        // row is stored never decides whether a search can find it.
        //
        // A time-travel read routes by what the store can answer. Lake
        // postings are only ever added and lake data files are immutable and
        // versioned, so a hit resolves against the manifest at the requested
        // version and the answer is exact. On the heap and columnar stores a
        // delete retires the document, so rows live at a past version are no
        // longer in the index and the storage scan is what can answer
        let indexes = self.catalog.get_indexes_for_table(table_id);
        let search_routable = as_of.is_none() || is_lake;

        // Check for full-text search predicates (MATCH AGAINST -> match_against function)
        if search_routable && let Some(pred) = &predicate {
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
                            as_of: as_of.clone(),
                            cost,
                        });
                    }
                }
            }
        }

        // Check for vector distance predicates (vector_distance_* function calls)
        if search_routable && let Some(pred) = &predicate {
            if let Some((vec_expr, remaining)) = extract_vector_distance(pred) {
                for index in &indexes {
                    if index.index_type == zyron_catalog::IndexType::Vector {
                        // Extract the literal query vector from the non-column
                        // argument of the distance call. Without one the index
                        // cannot serve the predicate, so fall through to the
                        // SeqScan path.
                        let Some(query_vector) = extract_query_vector(vec_expr) else {
                            break;
                        };
                        // The distance metric is intrinsic to the built index
                        // and applied inside the search at execution time, so it
                        // is carried as 0 here and not re-derived at plan time.
                        let metric = 0;
                        // k comes from the enclosing LIMIT; default to 10.
                        let k = limit.unwrap_or(10);
                        let cost = PlanCost {
                            io_cost: 1.0,
                            cpu_cost: 5.0,
                            row_count: k as f64,
                        };
                        return Ok(PhysicalPlan::VectorScan {
                            table_id,
                            index_id: index.id,
                            columns,
                            query_vector,
                            metric,
                            k,
                            remaining_predicate: remaining.cloned(),
                            as_of: as_of.clone(),
                            cost,
                        });
                    }
                }
            }
        }

        // Check for spatial predicates (st_dwithin / st_intersects / st_contains)
        // and route to a SpatialScan when the predicate shape is recognized
        // and a Spatial index is present on the table.
        if search_routable && let Some(pred) = &predicate {
            if let Some((sp_expr, remaining)) = extract_spatial_predicate(pred) {
                if let Some(kind) = build_spatial_scan_kind(sp_expr) {
                    for index in &indexes {
                        if index.index_type == zyron_catalog::IndexType::Spatial {
                            // KNN cost: exactly k rows by definition.
                            // DWithin / Range cost: start with a static
                            // default and let the executor refine at runtime.
                            // A full cardinality walk would require reaching
                            // into the live spatial manager here, which the
                            // planner does not currently reference; the cost
                            // model is good enough to prefer the index over
                            // SeqScan when a predicate matches.
                            let row_count = match &kind {
                                SpatialScanKind::Knn { k, .. } => *k as f64,
                                SpatialScanKind::DWithin { .. } => 100.0,
                                SpatialScanKind::Range { .. } => 100.0,
                            };
                            let cost = PlanCost {
                                io_cost: 2.0,
                                cpu_cost: 8.0,
                                row_count,
                            };
                            return Ok(PhysicalPlan::SpatialScan {
                                table_id,
                                index_id: index.id,
                                columns,
                                kind,
                                remaining_predicate: remaining.cloned(),
                                as_of: as_of.clone(),
                                cost,
                            });
                        }
                    }
                }
            }
        }

        // No index took the search predicate, so the storage scan evaluates
        // it row by row. A match call yields a relevance value rather than a
        // truth value, which is what an index operator consumes but not what
        // a filter does, so in boolean position it becomes a comparison
        let predicate = predicate.map(compare_bare_match_to_zero);

        // Lake routing gate. A lake table's rows live in its transaction
        // log, and past the search operators above no heap, columnar or
        // index node can see them, so the lake scan is the access path.
        // Every time-travel qualifier resolves inside the operator. A
        // version or timestamp resolves against the log, a branch by
        // opening that head instead of main's
        if let Ok(te) = self.catalog.get_table_by_id(table_id) {
            if te.lake.is_lake() {
                let cost = match &table_stats {
                    Some(s) => self.cost_model.cost_seq_scan(&s.0),
                    None => PlanCost {
                        io_cost: 10.0,
                        cpu_cost: 10.0,
                        row_count: 1000.0,
                    },
                };
                // Lower once here so the operator prunes with it instead of
                // repeating the work, and EXPLAIN can say whether the filter
                // skips files at all
                let lowered = predicate
                    .as_ref()
                    .and_then(|p| crate::lake_predicate::lower_predicate(p, &te.columns));
                return Ok(PhysicalPlan::LakeScan {
                    table_id,
                    columns,
                    predicate,
                    lowered,
                    as_of,
                    cost,
                });
            }
        }

        // Columnar correctness gate. When the table has registered .zyr
        // segments, folded rows were physically deleted from the heap, so a
        // heap-only SeqScan or IndexScan would silently miss them. Only the
        // hybrid scan reads both stores. A current read and an AS OF VERSION
        // read both use it: the hybrid scan dates the columnar and heap rows by
        // commit LSN under AS OF, so a past version sees folded rows too. AS OF
        // TIMESTAMP and AS OF BRANCH stay on the heap path (their predicate and
        // branch-overlay handling is heap-only).
        // A point lookup through a B+tree index beats a full hybrid scan on
        // a segment-bearing table: entries are locator-keyed, so the index
        // serves heap and folded rows alike. Only equality shapes route
        // here, range predicates keep the columnar scan's zone-map pruning.
        // The executor falls back to the hybrid union when no live tree is
        // registered, so folded rows are never dropped
        if as_of.is_none()
            && let Some(pred) = &predicate
            && let Ok(te) = self.catalog.get_table_by_id(table_id)
            && !te.columnar.segments.is_empty()
        {
            for index in &indexes {
                if index.index_type != zyron_catalog::IndexType::BTree {
                    continue;
                }
                if let Some((index_pred, remaining)) = match_index(pred, index)
                    && matches!(
                        &index_pred,
                        BoundExpr::BinaryOp {
                            op: BinaryOperator::Eq,
                            ..
                        }
                    )
                {
                    let cost = PlanCost {
                        io_cost: 1.0,
                        cpu_cost: 5.0,
                        row_count: 10.0,
                    };
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

        // Tables with columnar segments route to the hybrid scan, which
        // reads both stores under one MVCC view.
        let as_of_hybrid = matches!(as_of, None | Some(crate::logical::AsOfTarget::Version(_)));
        if as_of_hybrid
            && let Ok(te) = self.catalog.get_table_by_id(table_id)
            && !te.columnar.segments.is_empty()
        {
            let mut cost = match &table_stats {
                Some(s) => self.cost_model.cost_seq_scan(&s.0),
                None => PlanCost::zero(),
            };
            // Wire the zone-map / bloom skip-rate hint to the real columnar
            // path: a higher provable skip rate lowers the scanned cost.
            if let Some(hints) = &encoding_hints
                && hints.any_applicable()
            {
                let keep = 1.0 - hints.estimated_skip_rate();
                cost = PlanCost {
                    io_cost: cost.io_cost * keep,
                    cpu_cost: cost.cpu_cost * keep,
                    row_count: cost.row_count,
                };
            }
            return Ok(PhysicalPlan::HybridScan {
                table_id,
                columns,
                predicate,
                as_of,
                cost,
            });
        }

        // Try to find a B-tree index scan opportunity. Equality on indexed
        // columns prefers IndexScan even without stats, since index lookup
        // is by definition more selective than a full table scan.
        // Time travel (AS OF) stays on the sequential heap path: visibility is
        // dated per tuple by commit LSN, so every version of a row must be
        // examined, and the index carries no version information. This also
        // keeps AS OF TIMESTAMP and AS OF BRANCH on their predicate/overlay
        // handling, which only the sequential scan applies.
        if as_of.is_none() {
            if let Some(pred) = &predicate {
                for index in &indexes {
                    if let Some((index_pred, remaining)) = match_index(pred, index) {
                        let (selectivity, cost) = if let Some(s) = &table_stats {
                            let (ts, cs) = (&s.0, &s.1);
                            let sel = self.cost_model.estimate_selectivity(
                                &index_pred,
                                Some(ts),
                                Some(cs),
                            );
                            (sel, self.cost_model.cost_index_scan(ts, sel))
                        } else {
                            let sel = 0.001f64;
                            (
                                sel,
                                PlanCost {
                                    io_cost: 1.0,
                                    cpu_cost: 5.0,
                                    row_count: 10.0,
                                },
                            )
                        };
                        if selectivity < INDEX_SCAN_SELECTIVITY_THRESHOLD {
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
        let seq_cost = if let Some(s) = &table_stats {
            let ts = &s.0;
            let mut scan_cost = self.cost_model.cost_seq_scan(ts);
            if let Some(pred) = &predicate {
                let selectivity = self.cost_model.estimate_selectivity(
                    pred,
                    table_stats.as_ref().map(|s| &s.0),
                    table_stats.as_ref().map(|s| s.1.as_slice()),
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
        if let Some(s) = &table_stats {
            let ts = &s.0;
            if parallel_plan::should_parallelize(ts.row_count as f64) {
                let num_workers = parallel_plan::compute_worker_count(ts.page_count);
                if num_workers > 1 {
                    let mut parallel_cost = self.cost_model.cost_parallel_scan(ts, num_workers);
                    // Apply predicate selectivity to parallel scan row count
                    if predicate.is_some() {
                        parallel_cost.row_count = seq_cost.row_count;
                    }
                    // Gather rolls up the child's cost so the plan-level
                    // cost reflects the actual work, the child's parallel
                    // cost already includes parallel_tuple_cost so Gather
                    // adds no further coordination overhead
                    let gather_cost = PlanCost {
                        io_cost: parallel_cost.io_cost,
                        cpu_cost: parallel_cost.cpu_cost,
                        row_count: parallel_cost.row_count,
                    };
                    let total_parallel = gather_cost.total();
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
                if let Some(s) = &table_stats {
                    let ts = &s.0;
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
                // A subquery in the ON predicate cannot be a hash or merge key and
                // is evaluated per joined row by the nested-loop operator's
                // correlated path. Force a nested loop with the full condition so
                // no equi-key extraction strands the subquery on an unsupported
                // operator. (Inner joins are already lowered to Cross + Filter in
                // the logical builder, so this path is reached by outer joins.)
                if predicate_has_subquery(expr) {
                    let nl_cost = self
                        .cost_model
                        .cost_nested_loop_join(&left_cost, &right_cost);
                    return Ok(PhysicalPlan::NestedLoopJoin {
                        left: Box::new(left_plan),
                        right: Box::new(right_plan),
                        join_type,
                        condition: Some(expr.clone()),
                        cost: nl_cost,
                    });
                }
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
                                    // Gather rolls up the child's cost so plan-level
                                    // cost reflects the actual work
                                    let gather_cost = PlanCost {
                                        io_cost: parallel_cost.io_cost,
                                        cpu_cost: parallel_cost.cpu_cost,
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

    /// Maps a set of ungrouped aggregates to metadata-pushdown specs, or
    /// None if any aggregate is not metadata-answerable. MIN/MAX require a
    /// fixed-width column (a variable-length header stores only a truncated
    /// byte prefix, which is not an exact extremum). DISTINCT disqualifies.
    fn try_meta_agg_specs(
        &self,
        table_id: zyron_catalog::TableId,
        aggregates: &[crate::logical::AggregateExpr],
    ) -> Option<Vec<crate::physical::MetaAggSpec>> {
        if aggregates.is_empty() {
            return None;
        }
        let te = self.catalog.get_table_by_id(table_id).ok()?;
        let mut specs = Vec::with_capacity(aggregates.len());
        for a in aggregates {
            if a.distinct {
                return None;
            }
            let fname = a.function_name.to_lowercase();
            let (kind, col) = match fname.as_str() {
                "count" => {
                    let col = a.args.first().and_then(extract_column_id_from_expr);
                    match col {
                        Some(c) => (crate::physical::MetaAggKind::CountCol, Some(c)),
                        None => (crate::physical::MetaAggKind::CountStar, None),
                    }
                }
                "min" | "max" => {
                    let c = a.args.first().and_then(extract_column_id_from_expr)?;
                    let ce = te.columns.iter().find(|x| x.id == c)?;
                    // Fixed-width only: a var-len header min/max is a
                    // truncated prefix, not the exact value.
                    ce.physical_type_id().fixed_size()?;
                    let k = if fname == "min" {
                        crate::physical::MetaAggKind::Min
                    } else {
                        crate::physical::MetaAggKind::Max
                    };
                    (k, Some(c))
                }
                _ => return None,
            };
            specs.push(crate::physical::MetaAggSpec {
                kind,
                column_id: col,
                return_type: a.return_type,
                name: a.function_name.clone(),
            });
        }
        Some(specs)
    }

    fn plan_aggregate(
        &self,
        group_by: Vec<BoundExpr>,
        aggregates: Vec<crate::logical::AggregateExpr>,
        child: LogicalPlan,
    ) -> Result<PhysicalPlan> {
        let child_plan = self.plan(child)?;
        let child_cost = *child_plan.cost();

        // Metadata aggregate pushdown: ungrouped MIN/MAX/COUNT over a table
        // with columnar segments and no predicate can be answered from
        // segment headers (with an MVCC patch guard at execution) instead of
        // decoding the folded rows.
        if group_by.is_empty()
            && let PhysicalPlan::HybridScan {
                table_id,
                predicate: None,
                ..
            } = &child_plan
        {
            let tid = *table_id;
            if let Some(specs) = self.try_meta_agg_specs(tid, &aggregates) {
                let schema: Vec<crate::logical::LogicalColumn> = specs
                    .iter()
                    .enumerate()
                    .map(|(i, s)| crate::logical::LogicalColumn {
                        table_idx: Some(crate::logical::AGGREGATE_TABLE_IDX),
                        column_id: zyron_catalog::ColumnId(i as u16),
                        name: s.name.clone(),
                        type_id: s.return_type,
                        nullable: true,
                        ts_precision: None,
                    })
                    .collect();
                return Ok(PhysicalPlan::ColumnarMetadataAggregate {
                    table_id: tid,
                    specs,
                    schema,
                    cost: PlanCost {
                        io_cost: 1.0,
                        cpu_cost: 1.0,
                        row_count: 1.0,
                    },
                });
            }
        }

        let group_count = if group_by.is_empty() {
            1.0
        } else {
            child_cost.row_count.sqrt().max(1.0)
        };

        let cost = self
            .cost_model
            .cost_hash_aggregate(&child_cost, group_count);

        // Detect a time_bucket_gapfill(width, ts) grouping key: it groups
        // exactly like time_bucket, then a GapFill node densifies the result.
        let gapfill = group_by
            .iter()
            .enumerate()
            .find_map(|(i, g)| gapfill_width(g).map(|w| (i, w)));

        // Use HashAggregate by default (better for random group distributions)
        let agg = PhysicalPlan::HashAggregate {
            group_by,
            aggregates,
            child: Box::new(child_plan),
            cost: PlanCost {
                io_cost: 0.0,
                cpu_cost: cost.cpu_cost - child_cost.cpu_cost,
                row_count: group_count,
            },
        };

        match gapfill {
            Some((bucket_col, width)) => {
                let agg_cost = *agg.cost();
                Ok(PhysicalPlan::GapFill {
                    bucket_col,
                    width,
                    child: Box::new(agg),
                    cost: PlanCost {
                        io_cost: 0.0,
                        cpu_cost: agg_cost.cpu_cost,
                        // Dense output row count is data-derived at execution;
                        // estimate at least the grouped row count.
                        row_count: agg_cost.row_count,
                    },
                })
            }
            None => Ok(agg),
        }
    }
}

/// A constant integer value of a bound expression (unwrapping Nested), or None.
fn const_int(e: &BoundExpr) -> Option<i128> {
    match e {
        BoundExpr::Literal {
            value: zyron_parser::ast::LiteralValue::Integer(v),
            ..
        } => Some(*v as i128),
        BoundExpr::Nested(inner) => const_int(inner),
        _ => None,
    }
}

/// If `e` is `time_bucket_gapfill(width, ts)` with a constant integer width,
/// returns that width (in the timestamp column's storage unit). The grouping
/// itself is computed by the time_bucket_gapfill scalar (identical to
/// time_bucket); a GapFill node then densifies the grouped result.
fn gapfill_width(e: &BoundExpr) -> Option<i128> {
    match e {
        BoundExpr::Function { name, args, .. }
            if name.eq_ignore_ascii_case("time_bucket_gapfill") && !args.is_empty() =>
        {
            const_int(&args[0])
        }
        BoundExpr::Nested(inner) => gapfill_width(inner),
        _ => None,
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

/// Extracts a spatial predicate (st_dwithin / st_intersects / st_contains)
/// from a conjunction tree. Returns the spatial expression and any
/// remaining non-spatial predicate so the caller can wrap a Filter on top.
fn extract_spatial_predicate(predicate: &BoundExpr) -> Option<(&BoundExpr, Option<&BoundExpr>)> {
    match predicate {
        BoundExpr::Function { name, .. }
            if matches!(
                name.as_str(),
                "st_dwithin" | "st_intersects" | "st_contains"
            ) =>
        {
            Some((predicate, None))
        }
        BoundExpr::BinaryOp {
            left,
            op: BinaryOperator::And,
            right,
            ..
        } => {
            if let Some((sp, _)) = extract_spatial_predicate(left) {
                return Some((sp, Some(right)));
            }
            if let Some((sp, _)) = extract_spatial_predicate(right) {
                return Some((sp, Some(left)));
            }
            None
        }
        _ => None,
    }
}

/// Reads a literal `f64` out of a `BoundExpr::Literal`, transparently
/// handling `UnaryOp::Minus` wrappers the binder emits for negative number
/// literals (the parser represents `-74.0` as `- 74.0`).
fn extract_f64_literal(expr: &BoundExpr) -> Option<f64> {
    use zyron_parser::ast::{LiteralValue, UnaryOperator};
    match expr {
        BoundExpr::Literal { value, .. } => match value {
            LiteralValue::Float(f) => Some(*f),
            LiteralValue::Integer(i) => Some(*i as f64),
            _ => None,
        },
        BoundExpr::UnaryOp {
            op: UnaryOperator::Minus,
            expr: inner,
            ..
        } => extract_f64_literal(inner).map(|v| -v),
        BoundExpr::Nested(inner) => extract_f64_literal(inner),
        _ => None,
    }
}

/// Builds a SpatialScanKind from a recognized spatial function call.
/// Returns None when the call shape doesn't match the supported patterns.
fn build_spatial_scan_kind(call: &BoundExpr) -> Option<SpatialScanKind> {
    if let BoundExpr::Function { name, args, .. } = call {
        match name.as_str() {
            "st_dwithin" if args.len() == 3 => {
                // Signature: st_dwithin(geom_col, point, radius_meters). The
                // radius has to be a literal and the point has to be one the
                // plan can build without reading a row
                let radius = extract_f64_literal(&args[2])?;
                let qp = extract_point_from_expr(&args[1])?;
                Some(SpatialScanKind::DWithin {
                    query_point: qp,
                    radius_meters: radius,
                })
            }
            "st_intersects" | "st_contains" if args.len() == 2 => {
                // The bounding box of the second argument is the query
                // envelope, taken from whatever constant geometry it denotes
                let env = extract_envelope_from_expr(&args[1])?;
                Some(SpatialScanKind::Range {
                    mbr_min: env.0,
                    mbr_max: env.1,
                })
            }
            _ => None,
        }
    } else {
        None
    }
}

/// The coordinates of a constant point argument, whether it is written as a
/// constructor call or as a geometry the plan can build without a row.
fn extract_point_from_expr(expr: &BoundExpr) -> Option<Vec<f64>> {
    match expr {
        BoundExpr::Nested(inner) => return extract_point_from_expr(inner),
        BoundExpr::Function { name, args, .. } => {
            if (name == "st_make_point" || name == "st_makepoint") && args.len() >= 2 {
                let x = extract_f64_literal(&args[0])?;
                let y = extract_f64_literal(&args[1])?;
                return Some(vec![x, y]);
            }
        }
        _ => {}
    }
    match extract_constant_geometry(expr)?.kind {
        zyron_types::geospatial::GeometryKind::Point(p) => Some(vec![p.x, p.y]),
        _ => None,
    }
}

/// Extracts the query point from an `ST_Distance(col, point)` ordering
/// expression. The column argument is the indexed geometry and the other is a
/// literal point. Returns the point when one argument is a column and the other
/// resolves to a point literal.
fn extract_st_distance_point(expr: &BoundExpr) -> Option<Vec<f64>> {
    let inner = match expr {
        BoundExpr::Nested(e) => e.as_ref(),
        other => other,
    };
    let BoundExpr::Function { name, args, .. } = inner else {
        return None;
    };
    if name != "st_distance" || args.len() != 2 {
        return None;
    }
    // The point is whichever argument is not the indexed column.
    if extract_column_id_from_expr(&args[0]).is_some() {
        extract_point_from_expr(&args[1])
    } else if extract_column_id_from_expr(&args[1]).is_some() {
        extract_point_from_expr(&args[0])
    } else {
        None
    }
}

/// The bounding box of a constant geometry argument.
///
/// The index answers a bounding box, so any argument whose bounds are known
/// without reading a row can drive it: the envelope constructors, and any
/// geometry built from constants, including a bare WKT string. A geometry
/// the plan cannot evaluate here returns None and the scan filters row by
/// row, which is slower and returns the same rows.
fn extract_envelope_from_expr(expr: &BoundExpr) -> Option<(Vec<f64>, Vec<f64>)> {
    match expr {
        BoundExpr::Nested(inner) => return extract_envelope_from_expr(inner),
        BoundExpr::Function { name, args, .. } => {
            // st_makeenvelope(min_x, min_y, max_x, max_y), which names its
            // bounds directly rather than through a geometry
            if (name == "st_make_envelope" || name == "st_makeenvelope") && args.len() == 4 {
                let min_x = extract_f64_literal(&args[0])?;
                let min_y = extract_f64_literal(&args[1])?;
                let max_x = extract_f64_literal(&args[2])?;
                let max_y = extract_f64_literal(&args[3])?;
                return Some((
                    vec![min_x.min(max_x), min_y.min(max_y)],
                    vec![min_x.max(max_x), min_y.max(max_y)],
                ));
            }
        }
        _ => {}
    }
    // A point is a degenerate box, which is exactly what an intersects or
    // contains test against one asks the index for
    if let Some(point) = extract_point_from_expr(expr) {
        return Some((point.clone(), point));
    }
    let geometry = extract_constant_geometry(expr)?;
    let mbr = zyron_types::spatial_index::mbr_from_geometry(&geometry, 2);
    // An empty geometry has an inverted box, which would prune everything
    if mbr.mins[0] > mbr.maxs[0] || mbr.mins[1] > mbr.maxs[1] {
        return None;
    }
    Some((
        vec![mbr.mins[0], mbr.mins[1]],
        vec![mbr.maxs[0], mbr.maxs[1]],
    ))
}

/// Builds the geometry a constant expression denotes, without reading a row.
///
/// Covers a bare WKT string, the text and WKB constructors, and the
/// spatial-reference wrappers that pass their geometry through unchanged,
/// since none of those move the bounding box.
fn extract_constant_geometry(expr: &BoundExpr) -> Option<zyron_types::geospatial::Geometry> {
    match expr {
        BoundExpr::Nested(inner) => extract_constant_geometry(inner),
        BoundExpr::Literal {
            value: zyron_parser::ast::LiteralValue::String(text),
            ..
        } => zyron_types::geospatial::st_geom_from_text(text).ok(),
        BoundExpr::Cast { expr: inner, .. } => extract_constant_geometry(inner),
        BoundExpr::Function { name, args, .. } => match name.as_str() {
            "st_geomfromtext" | "st_geom_from_text" | "st_geometryfromtext" | "st_geogfromtext"
                if !args.is_empty() =>
            {
                extract_constant_geometry(&args[0])
            }
            // SRID only labels the coordinate system, it does not move the
            // coordinates, so the box is the inner geometry's
            "st_setsrid" | "st_set_srid" if !args.is_empty() => {
                extract_constant_geometry(&args[0])
            }
            _ => None,
        },
        _ => None,
    }
}

/// Rewrites a bare `match_against` call in boolean position into a
/// comparison against zero.
///
/// The call returns a relevance value, so `WHERE MATCH(...) AGAINST(...)`
/// hands a filter a number where it needs a truth value. Walking only the
/// boolean connectives keeps a call already inside a comparison, or used as a
/// projected score, exactly as written.
fn compare_bare_match_to_zero(predicate: BoundExpr) -> BoundExpr {
    match predicate {
        BoundExpr::Function {
            ref name,
            ref return_type,
            ..
        } if name == "match_against" => {
            let return_type = *return_type;
            BoundExpr::BinaryOp {
                left: Box::new(predicate),
                op: BinaryOperator::Gt,
                right: Box::new(BoundExpr::Literal {
                    value: zyron_parser::ast::LiteralValue::Float(0.0),
                    type_id: return_type,
                }),
                type_id: zyron_common::TypeId::Boolean,
            }
        }
        BoundExpr::BinaryOp {
            left,
            op,
            right,
            type_id,
        } if matches!(op, BinaryOperator::And | BinaryOperator::Or) => BoundExpr::BinaryOp {
            left: Box::new(compare_bare_match_to_zero(*left)),
            op,
            right: Box::new(compare_bare_match_to_zero(*right)),
            type_id,
        },
        BoundExpr::UnaryOp { op, expr, type_id }
            if matches!(op, zyron_parser::ast::UnaryOperator::Not) =>
        {
            BoundExpr::UnaryOp {
                op,
                expr: Box::new(compare_bare_match_to_zero(*expr)),
                type_id,
            }
        }
        BoundExpr::Nested(inner) => BoundExpr::Nested(Box::new(compare_bare_match_to_zero(*inner))),
        other => other,
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

/// Extracts the literal query vector from a `vector_distance_*` call. The call
/// has two arguments, one the indexed column and the other an array constructor
/// of numeric literals. Returns the f32 vector when the array side is present.
fn extract_query_vector(call: &BoundExpr) -> Option<Vec<f32>> {
    let BoundExpr::Function { args, .. } = call else {
        return None;
    };
    if args.len() != 2 {
        return None;
    }
    // The query side is whichever argument is not the column reference.
    let query_arg = if extract_column_id_from_expr(&args[0]).is_some() {
        &args[1]
    } else {
        &args[0]
    };
    array_literal_to_f32(query_arg)
}

/// Reads an array constructor of numeric literals into an f32 vector.
fn array_literal_to_f32(expr: &BoundExpr) -> Option<Vec<f32>> {
    let inner = match expr {
        BoundExpr::Nested(e) => e.as_ref(),
        other => other,
    };
    let BoundExpr::Function { name, args, .. } = inner else {
        return None;
    };
    if name != "array" || args.is_empty() {
        return None;
    }
    let mut out = Vec::with_capacity(args.len());
    for a in args {
        out.push(extract_f64_literal(a)? as f32);
    }
    Some(out)
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

/// Carries a LIMIT's row cap down to a foreign scan, so a capped query does
/// not drag a whole remote table across the network to discard most of it.
///
/// It descends only through nodes that neither drop nor reorder rows, which
/// is Project alone. A Filter below the cap would return fewer rows than
/// asked for; a Sort needs every row before it can say which N come first;
/// a join or an aggregate changes the row count outright. The scan also has
/// to have kept nothing to filter locally, for the same reason a Filter
/// blocks it.
///
/// `cap` is offset plus limit, because rows the offset skips still have to
/// arrive. The enclosing Limit applies unchanged either way, so this only
/// ever removes work.
fn push_limit_to_foreign_scan(plan: &mut PhysicalPlan, cap: usize) {
    match plan {
        PhysicalPlan::ForeignScan {
            residual, request, ..
        } => {
            if residual.is_none() {
                request.limit = Some(match request.limit {
                    Some(existing) => existing.min(cap),
                    None => cap,
                });
            }
        }
        PhysicalPlan::Project { child, .. } => push_limit_to_foreign_scan(child, cap),
        _ => {}
    }
}

fn is_column_ref(expr: &BoundExpr) -> bool {
    matches!(expr, BoundExpr::ColumnRef(_))
}

/// Returns true when an expression tree contains a subquery node. Such a
/// predicate is evaluated by a Filter operator rather than fused into a scan,
/// so the scan never sees an outer reference it cannot resolve.
fn predicate_has_subquery(expr: &BoundExpr) -> bool {
    match expr {
        BoundExpr::Subquery { .. } | BoundExpr::Exists { .. } | BoundExpr::InSubquery { .. } => {
            true
        }
        BoundExpr::BinaryOp { left, right, .. } => {
            predicate_has_subquery(left) || predicate_has_subquery(right)
        }
        BoundExpr::UnaryOp { expr, .. }
        | BoundExpr::IsNull { expr, .. }
        | BoundExpr::Cast { expr, .. }
        | BoundExpr::Nested(expr)
        | BoundExpr::TemporalRef { inner: expr, .. } => predicate_has_subquery(expr),
        BoundExpr::Between {
            expr, low, high, ..
        } => {
            predicate_has_subquery(expr)
                || predicate_has_subquery(low)
                || predicate_has_subquery(high)
        }
        BoundExpr::InList { expr, list, .. } => {
            predicate_has_subquery(expr) || list.iter().any(predicate_has_subquery)
        }
        BoundExpr::Like { expr, pattern, .. } | BoundExpr::ILike { expr, pattern, .. } => {
            predicate_has_subquery(expr) || predicate_has_subquery(pattern)
        }
        BoundExpr::Function { args, .. } | BoundExpr::AggregateFunction { args, .. } => {
            args.iter().any(predicate_has_subquery)
        }
        BoundExpr::Case {
            operand,
            conditions,
            else_result,
            ..
        } => {
            operand.as_deref().is_some_and(predicate_has_subquery)
                || conditions.iter().any(|w| {
                    predicate_has_subquery(&w.condition) || predicate_has_subquery(&w.result)
                })
                || else_result.as_deref().is_some_and(predicate_has_subquery)
        }
        BoundExpr::WindowFunction {
            function,
            partition_by,
            order_by,
            ..
        } => {
            predicate_has_subquery(function)
                || partition_by.iter().any(predicate_has_subquery)
                || order_by.iter().any(|o| predicate_has_subquery(&o.expr))
        }
        BoundExpr::ColumnRef(_) | BoundExpr::Literal { .. } | BoundExpr::Parameter { .. } => false,
    }
}

/// Walks a BoundExpr tree and replaces WindowFunction nodes with ColumnRefs
/// pointing to positions in an auxiliary window-output column list.
/// Collects the WindowFunction expressions into `collected`.
fn rewrite_window_refs(
    expr: &BoundExpr,
    collected: &mut Vec<BoundExpr>,
    names: &mut Vec<String>,
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
            // Window output columns are appended after the input columns and
            // addressed by (WINDOW_TABLE_IDX, window index). The sentinel
            // table_idx cannot collide with an input column's (table_idx,
            // column_id), which the old positional column_id did once projection
            // pushdown trimmed the input. Must match Window::output_schema.
            BE::ColumnRef(ColumnRef {
                table_idx: crate::logical::WINDOW_TABLE_IDX,
                column_id: ColumnId(idx as u16),
                type_id: *type_id,
                nullable: true,
                // Window-output precision finalized in B5.
                ts_precision: None,
            })
        }
        BE::ColumnRef(_) | BE::Literal { .. } | BE::Parameter { .. } => expr.clone(),
        BE::BinaryOp {
            left,
            op,
            right,
            type_id,
        } => BE::BinaryOp {
            left: Box::new(rewrite_window_refs(left, collected, names, None)),
            op: op.clone(),
            right: Box::new(rewrite_window_refs(right, collected, names, None)),
            type_id: *type_id,
        },
        BE::UnaryOp {
            op,
            expr: inner,
            type_id,
        } => BE::UnaryOp {
            op: op.clone(),
            expr: Box::new(rewrite_window_refs(inner, collected, names, None)),
            type_id: *type_id,
        },
        BE::IsNull {
            expr: inner,
            negated,
        } => BE::IsNull {
            expr: Box::new(rewrite_window_refs(inner, collected, names, None)),
            negated: *negated,
        },
        BE::InList {
            expr: inner,
            list,
            negated,
        } => BE::InList {
            expr: Box::new(rewrite_window_refs(inner, collected, names, None)),
            list: list
                .iter()
                .map(|e| rewrite_window_refs(e, collected, names, None))
                .collect(),
            negated: *negated,
        },
        BE::Between {
            expr: inner,
            low,
            high,
            negated,
        } => BE::Between {
            expr: Box::new(rewrite_window_refs(inner, collected, names, None)),
            low: Box::new(rewrite_window_refs(low, collected, names, None)),
            high: Box::new(rewrite_window_refs(high, collected, names, None)),
            negated: *negated,
        },
        BE::Like {
            expr: inner,
            pattern,
            negated,
        } => BE::Like {
            expr: Box::new(rewrite_window_refs(inner, collected, names, None)),
            pattern: Box::new(rewrite_window_refs(pattern, collected, names, None)),
            negated: *negated,
        },
        BE::ILike {
            expr: inner,
            pattern,
            negated,
        } => BE::ILike {
            expr: Box::new(rewrite_window_refs(inner, collected, names, None)),
            pattern: Box::new(rewrite_window_refs(pattern, collected, names, None)),
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
                .map(|a| rewrite_window_refs(a, collected, names, None))
                .collect(),
            return_type: *return_type,
            distinct: *distinct,
        },
        BE::AggregateFunction {
            name,
            args,
            distinct,
            return_type,
            uda,
        } => BE::AggregateFunction {
            name: name.clone(),
            args: args
                .iter()
                .map(|a| rewrite_window_refs(a, collected, names, None))
                .collect(),
            distinct: *distinct,
            return_type: *return_type,
            uda: uda.clone(),
        },
        BE::Cast {
            expr: inner,
            target_type,
        } => BE::Cast {
            expr: Box::new(rewrite_window_refs(inner, collected, names, None)),
            target_type: *target_type,
        },
        BE::Case {
            operand,
            conditions,
            else_result,
            type_id,
        } => BE::Case {
            operand: operand
                .as_ref()
                .map(|o| Box::new(rewrite_window_refs(o, collected, names, None))),
            conditions: conditions
                .iter()
                .map(|w| BoundWhen {
                    condition: rewrite_window_refs(&w.condition, collected, names, None),
                    result: rewrite_window_refs(&w.result, collected, names, None),
                })
                .collect(),
            else_result: else_result
                .as_ref()
                .map(|e| Box::new(rewrite_window_refs(e, collected, names, None))),
            type_id: *type_id,
        },
        BE::Nested(inner) => {
            BE::Nested(Box::new(rewrite_window_refs(inner, collected, names, None)))
        }
        // Subqueries don't participate in window rewriting at this level.
        BE::Subquery { .. } | BE::Exists { .. } | BE::InSubquery { .. } => expr.clone(),
        BE::TemporalRef { inner, temporal } => BE::TemporalRef {
            inner: Box::new(rewrite_window_refs(inner, collected, names, None)),
            temporal: temporal.clone(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binder::ColumnRef;
    use zyron_catalog::ColumnId;

    fn geom_literal(wkt: &str) -> BoundExpr {
        BoundExpr::Literal {
            value: zyron_parser::ast::LiteralValue::String(wkt.to_string()),
            type_id: TypeId::Geometry,
        }
    }

    fn call(name: &str, args: Vec<BoundExpr>) -> BoundExpr {
        BoundExpr::Function {
            name: name.to_string(),
            args,
            return_type: TypeId::Geometry,
            distinct: false,
        }
    }

    fn float(v: f64) -> BoundExpr {
        BoundExpr::Literal {
            value: zyron_parser::ast::LiteralValue::Float(v),
            type_id: TypeId::Float64,
        }
    }

    /// The index answers a bounding box, so every constant way of writing a
    /// geometry has to reach it. A shape that does not route still returns
    /// the right rows through the row filter, so what this pins is that the
    /// ordinary spellings do not silently fall off the index.
    #[test]
    fn test_a_constant_geometry_yields_its_bounding_box_however_it_is_written() {
        let expected = (vec![0.0, 0.0], vec![10.0, 20.0]);
        let forms = [
            call(
                "st_makeenvelope",
                vec![float(0.0), float(0.0), float(10.0), float(20.0)],
            ),
            // Reversed corners name the same box
            call(
                "st_make_envelope",
                vec![float(10.0), float(20.0), float(0.0), float(0.0)],
            ),
            geom_literal("POLYGON((0 0, 10 0, 10 20, 0 20, 0 0))"),
            call(
                "st_geomfromtext",
                vec![geom_literal("POLYGON((0 0, 10 0, 10 20, 0 20, 0 0))")],
            ),
            call(
                "st_setsrid",
                vec![
                    call(
                        "st_geom_from_text",
                        vec![geom_literal("LINESTRING(0 0, 10 20)")],
                    ),
                    float(4326.0),
                ],
            ),
            BoundExpr::Nested(Box::new(geom_literal("LINESTRING(0 20, 10 0)"))),
        ];
        for form in &forms {
            assert_eq!(
                extract_envelope_from_expr(form),
                Some(expected.clone()),
                "form {form:?}"
            );
        }

        // A point is a degenerate box, and reaches the index as one
        let point = vec![3.0, 4.0];
        for form in [
            call("st_makepoint", vec![float(3.0), float(4.0)]),
            geom_literal("POINT(3 4)"),
        ] {
            assert_eq!(extract_point_from_expr(&form), Some(point.clone()));
            assert_eq!(
                extract_envelope_from_expr(&form),
                Some((point.clone(), point.clone()))
            );
        }

        // A geometry the plan cannot build yields nothing, so the scan
        // filters row by row rather than pruning against a wrong box
        let column = BoundExpr::ColumnRef(ColumnRef {
            table_idx: 0,
            column_id: ColumnId(0),
            type_id: TypeId::Geometry,
            nullable: false,
            ts_precision: None,
        });
        assert_eq!(extract_envelope_from_expr(&column), None);
        assert_eq!(extract_envelope_from_expr(&geom_literal("not wkt")), None);
        assert_eq!(extract_point_from_expr(&geom_literal("POLYGON((0 0, 1 0, 1 1, 0 0))")), None);
    }

    #[test]
    fn test_extract_equi_keys() {
        let left_col = BoundExpr::ColumnRef(ColumnRef {
            table_idx: 0,
            column_id: ColumnId(0),
            type_id: TypeId::Int64,
            nullable: false,
            ts_precision: None,
        });
        let right_col = BoundExpr::ColumnRef(ColumnRef {
            table_idx: 1,
            column_id: ColumnId(0),
            type_id: TypeId::Int64,
            nullable: false,
            ts_precision: None,
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
            ts_precision: None,
        });
        let right_col = BoundExpr::ColumnRef(ColumnRef {
            table_idx: 1,
            column_id: ColumnId(0),
            type_id: TypeId::Int64,
            nullable: false,
            ts_precision: None,
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
                ts_precision: None,
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

//! Converts bound AST into logical plan trees.
//!
//! The builder takes a BoundStatement and produces a LogicalPlan tree
//! following standard relational algebra construction:
//! FROM -> Filter -> Aggregate -> Having -> Sort -> Project -> Distinct -> Limit
//! Sort sits beneath Project so ORDER BY keys resolve against the input schema
//! (a projection emits positional columns a ColumnRef cannot address).

use crate::binder::*;
use crate::logical::*;
use std::sync::Arc;
use zyron_catalog::ColumnId;
use zyron_common::{Result, TypeId, ZyronError};
use zyron_parser::ast::LiteralValue;

/// Converts a bound statement into a logical plan tree.
pub fn build_logical_plan(bound: &BoundStatement) -> Result<LogicalPlan> {
    match bound {
        BoundStatement::Select(select) => build_select_plan(select),
        BoundStatement::Insert(insert) => build_insert_plan(insert),
        BoundStatement::Update(update) => build_update_plan(update),
        BoundStatement::Delete(delete) => build_delete_plan(delete),
        BoundStatement::CreateStreamingJob(_)
        | BoundStatement::DropStreamingJob { .. }
        | BoundStatement::AlterStreamingJob { .. } => Err(ZyronError::PlanError(
            "streaming jobs are dispatched directly from wire, not through physical planner"
                .to_string(),
        )),
        BoundStatement::CreateExternalSource(_)
        | BoundStatement::CreateExternalSink(_)
        | BoundStatement::DropExternalSource { .. }
        | BoundStatement::DropExternalSink { .. }
        | BoundStatement::AlterExternalSource(_)
        | BoundStatement::AlterExternalSink(_) => Err(ZyronError::PlanError(
            "external source and sink DDL is dispatched directly from wire, not through physical planner"
                .to_string(),
        )),
        BoundStatement::CreatePublication(_)
        | BoundStatement::AlterPublication(_)
        | BoundStatement::DropPublication { .. }
        | BoundStatement::TagPublication { .. }
        | BoundStatement::UntagPublication { .. }
        | BoundStatement::CreateEndpoint(_)
        | BoundStatement::CreateStreamingEndpoint(_)
        | BoundStatement::AlterEndpoint(_)
        | BoundStatement::DropEndpoint { .. }
        | BoundStatement::AlterSecurityMap(_)
        | BoundStatement::DropSecurityMap(_) => Err(ZyronError::PlanError(
            "Zyron-to-Zyron DDL is dispatched directly from wire, not through physical planner"
                .to_string(),
        )),
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
                fractional_digits: None,
            }],
        }
    } else {
        let mut from_plan: Option<LogicalPlan> = None;
        for item in &select.from {
            from_plan = Some(match from_plan {
                // First item: no left to correlate to. A lateral here would have
                // failed to bind (empty outer scope), so build it plainly.
                None => build_from_item(item)?,
                // A comma-separated LATERAL subquery cross-joins to the
                // accumulated left with per-left-row execution.
                Some(left) => {
                    if let BoundFromItem::Subquery {
                        query,
                        table_idx,
                        lateral: true,
                    } = item
                    {
                        LogicalPlan::LateralJoin {
                            left: Arc::new(left),
                            subquery: crate::logical::LateralSubquery(query.clone()),
                            subquery_table_idx: *table_idx,
                            join_type: zyron_parser::ast::JoinType::Cross,
                            condition: None,
                        }
                    } else {
                        LogicalPlan::Join {
                            left: Arc::new(left),
                            right: Arc::new(build_from_item(item)?),
                            join_type: zyron_parser::ast::JoinType::Cross,
                            condition: JoinCondition::Cross,
                        }
                    }
                }
            });
        }
        from_plan.unwrap()
    };

    // 2. WHERE -> Filter
    if let Some(predicate) = &select.where_clause {
        plan = LogicalPlan::Filter {
            predicate: predicate.clone(),
            child: Arc::new(plan),
        };
    }

    // 3. GROUP BY + aggregates -> Aggregate
    //
    // Aggregates can appear in the projection list, the HAVING predicate, and
    // the ORDER BY keys. All three feed the same Aggregate node so the executor
    // computes each one once; references in every position are later rewritten
    // to aggregate output columns.
    let (has_aggregates, aggregates) = extract_aggregates(
        &select.projections,
        select.having.as_ref(),
        &select.order_by,
    );
    let aggregate_pushed = !select.group_by.is_empty() || has_aggregates;
    if aggregate_pushed {
        plan = LogicalPlan::Aggregate {
            group_by: select.group_by.clone(),
            aggregates: aggregates.clone(),
            child: Arc::new(plan),
        };
    }

    // 4. HAVING -> Filter on top of Aggregate. The predicate references the
    // aggregate results and group-by keys; rewrite them to aggregate output
    // columns so the filter reads computed values instead of raw aggregates.
    if let Some(having) = &select.having {
        let mut predicate = having.clone();
        if aggregate_pushed {
            rewrite_post_aggregate(&mut predicate, &select.group_by, &aggregates);
        }
        plan = LogicalPlan::Filter {
            predicate,
            child: Arc::new(plan),
        };
    }

    // 5. ORDER BY -> Sort, placed beneath the projection.
    //
    // A projection emits positional output columns carrying table_idx None,
    // which a ColumnRef cannot address, so a Sort above the projection could
    // not resolve its keys. Sorting below the projection evaluates keys against
    // the input schema, letting ORDER BY reference any input column whether or
    // not it is selected. The projection above and the row-order-preserving
    // Distinct above that carry the sorted order through to the result. In an
    // aggregate query the keys reference group results and aggregates, rewritten
    // to aggregate output columns just like the projection and HAVING.
    if !select.order_by.is_empty() {
        let mut order_by = select.order_by.clone();
        if aggregate_pushed {
            for ob in order_by.iter_mut() {
                rewrite_post_aggregate(&mut ob.expr, &select.group_by, &aggregates);
            }
        }
        plan = LogicalPlan::Sort {
            order_by,
            child: Arc::new(plan),
        };
    }

    // FOR UPDATE/SHARE -> LockRows, above Sort so the locked set is the
    // final row set, below Project because projection drops row locators.
    // The binder rejected multi-table, DISTINCT, GROUP BY and set-op shapes,
    // aggregation hiding in the projection list is only discoverable here
    if let Some(lock) = &select.row_lock {
        if aggregate_pushed {
            return Err(ZyronError::PlanError(
                "FOR UPDATE/SHARE is not allowed with aggregate functions".to_string(),
            ));
        }
        let cap = extract_u64_literal(&select.limit)
            .map(|l| l + extract_u64_literal(&select.offset).unwrap_or(0));
        plan = LogicalPlan::LockRows {
            table_id: lock.table_id,
            mode: lock.mode,
            wait: lock.wait,
            cap,
            child: Arc::new(plan),
        };
    }

    // 6. SELECT -> Project
    //
    // When an aggregate is in scope, the projection list still references
    // the original `AggregateFunction` and group-by expressions. Those
    // cannot be evaluated by the projection operator: the aggregate has
    // already collapsed the rows. Rewrite each occurrence into a column
    // reference into the aggregate node's output schema.
    let (mut expressions, mut aliases) = build_projection_list(&select.projections);
    // Fill implicit output names from the bound output schema so a projection
    // exposes real column names (e.g. `id`) rather than positional `col0`.
    // The two align whenever the projection is all expressions (no wildcard,
    // which is the only case that builds a Project node).
    if aliases.len() == select.output_schema.len() {
        for (alias, col) in aliases.iter_mut().zip(select.output_schema.iter()) {
            if alias.is_none() {
                *alias = Some(col.name.clone());
            }
        }
    }
    if aggregate_pushed {
        for expr in expressions.iter_mut() {
            rewrite_post_aggregate(expr, &select.group_by, &aggregates);
        }
    }
    if !expressions.is_empty() {
        plan = LogicalPlan::Project {
            expressions,
            aliases,
            child: Arc::new(plan),
            output_table_idx: None,
        };
    }

    // 7. DISTINCT -> Distinct
    if select.distinct {
        plan = LogicalPlan::Distinct {
            child: Arc::new(plan),
        };
    }

    // 8. LIMIT/OFFSET -> Limit
    let limit_val = extract_u64_literal(&select.limit);
    let offset_val = extract_u64_literal(&select.offset);
    if limit_val.is_some() || offset_val.is_some() {
        plan = LogicalPlan::Limit {
            limit: limit_val,
            offset: offset_val,
            child: Arc::new(plan),
        };
    }

    // 9. Set operations
    for set_op in &select.set_ops {
        let right_plan = build_select_plan(&set_op.right)?;
        plan = LogicalPlan::SetOp {
            op: set_op.op,
            all: set_op.all,
            left: Arc::new(plan),
            right: Arc::new(right_plan),
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
            as_of,
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
                    fractional_digits: c.fractional_digits,
                })
                .collect();
            let as_of_target = match as_of {
                None => None,
                Some(crate::binder::BoundAsOfTarget::Timestamp(ts)) => {
                    Some(crate::logical::AsOfTarget::Timestamp(*ts))
                }
                Some(crate::binder::BoundAsOfTarget::Version(v)) => {
                    Some(crate::logical::AsOfTarget::Version(*v))
                }
                Some(crate::binder::BoundAsOfTarget::Branch(name)) => {
                    Some(crate::logical::AsOfTarget::Branch(name.clone()))
                }
            };
            Ok(LogicalPlan::Scan {
                table_id: *table_id,
                table_idx: *table_idx,
                columns,
                alias: entry.name.clone(),
                encoding_hints: None,
                as_of: as_of_target,
            })
        }
        BoundFromItem::Join {
            left,
            join_type,
            right,
            condition,
        } => {
            // A LATERAL subquery on the right side correlates to the left, so it
            // becomes a LateralJoin (per-left-row execution) rather than a plain
            // join over two independently planned inputs.
            if let BoundFromItem::Subquery {
                query,
                table_idx,
                lateral: true,
            } = right.as_ref()
            {
                let left_plan = build_from_item(left)?;
                let cond = match condition {
                    BoundJoinCondition::On(expr) => Some(expr.clone()),
                    _ => None,
                };
                return Ok(LogicalPlan::LateralJoin {
                    left: Arc::new(left_plan),
                    subquery: crate::logical::LateralSubquery(query.clone()),
                    subquery_table_idx: *table_idx,
                    join_type: *join_type,
                    condition: cond,
                });
            }
            let left_plan = build_from_item(left)?;
            let right_plan = build_from_item(right)?;
            // A subquery in an INNER join ON predicate cannot be evaluated by
            // the join operator. An inner join with an ON predicate is exactly a
            // cross join filtered by that predicate, so lower it to Cross + Filter;
            // the Filter then folds an uncorrelated subquery or runs a correlated
            // one per row. Outer joins are not equivalent to this rewrite, so a
            // subquery in their ON predicate is left to surface as an error.
            if matches!(join_type, zyron_parser::ast::JoinType::Inner) {
                if let BoundJoinCondition::On(expr) = condition {
                    if crate::binder::expr_contains_subquery(expr) {
                        let cross = LogicalPlan::Join {
                            left: Arc::new(left_plan),
                            right: Arc::new(right_plan),
                            join_type: zyron_parser::ast::JoinType::Cross,
                            condition: JoinCondition::Cross,
                        };
                        return Ok(LogicalPlan::Filter {
                            predicate: expr.clone(),
                            child: Arc::new(cross),
                        });
                    }
                }
            }
            let join_condition = match condition {
                BoundJoinCondition::On(expr) => JoinCondition::On(expr.clone()),
                BoundJoinCondition::Using(cols) => JoinCondition::Using(cols.clone()),
                BoundJoinCondition::Natural => JoinCondition::Natural,
                BoundJoinCondition::None => JoinCondition::Cross,
            };
            Ok(LogicalPlan::Join {
                left: Arc::new(left_plan),
                right: Arc::new(right_plan),
                join_type: *join_type,
                condition: join_condition,
            })
        }
        BoundFromItem::Subquery {
            query, table_idx, ..
        } => {
            let inner = build_select_plan(query)?;
            Ok(relabel_derived(inner, *table_idx))
        }
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
        BoundFromItem::AnalyticsFunction {
            function_name,
            params,
            positional,
            output_columns,
            ..
        } => Ok(LogicalPlan::AnalyticsTableFunction {
            function_name: function_name.clone(),
            named_args: params.clone(),
            positional_args: positional.clone(),
            output_columns: output_columns.clone(),
        }),
    }
}

/// Relabels a derived table's output columns under `table_idx` so the
/// enclosing query addresses them by `(table_idx, ordinal)`. A query topped by
/// a projection is stamped in place; any other shape is wrapped in a
/// passthrough projection of its output columns carrying the table index.
fn relabel_derived(inner: LogicalPlan, table_idx: usize) -> LogicalPlan {
    match inner {
        LogicalPlan::Project {
            expressions,
            aliases,
            child,
            ..
        } => LogicalPlan::Project {
            expressions,
            aliases,
            child,
            output_table_idx: Some(table_idx),
        },
        other => {
            let schema = other.output_schema();
            let expressions: Vec<BoundExpr> = schema
                .iter()
                .map(|c| {
                    BoundExpr::ColumnRef(ColumnRef {
                        // Passthrough columns belong to the derived table being
                        // relabeled, so an unlabeled child column resolves under
                        // this table_idx rather than a hardcoded table 0.
                        table_idx: c.table_idx.unwrap_or(table_idx),
                        column_id: c.column_id,
                        type_id: c.type_id,
                        nullable: c.nullable,
                        fractional_digits: c.fractional_digits,
                    })
                })
                .collect();
            let aliases: Vec<Option<String>> =
                schema.iter().map(|c| Some(c.name.clone())).collect();
            LogicalPlan::Project {
                expressions,
                aliases,
                child: Arc::new(other),
                output_table_idx: Some(table_idx),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Aggregate extraction
// ---------------------------------------------------------------------------

/// Walks projection items to find aggregate functions.
/// Returns (has_any_aggregates, list_of_aggregate_exprs).
fn extract_aggregates(
    projections: &[BoundSelectItem],
    having: Option<&BoundExpr>,
    order_by: &[BoundOrderBy],
) -> (bool, Vec<AggregateExpr>) {
    let mut aggregates = Vec::new();
    let mut has_agg = false;

    for item in projections {
        if let BoundSelectItem::Expr(expr, _) = item {
            collect_aggregates_from_expr(expr, &mut aggregates, &mut has_agg);
        }
    }
    if let Some(expr) = having {
        collect_aggregates_from_expr(expr, &mut aggregates, &mut has_agg);
    }
    for ob in order_by {
        collect_aggregates_from_expr(&ob.expr, &mut aggregates, &mut has_agg);
    }

    dedup_aggregates(&mut aggregates);
    (has_agg, aggregates)
}

/// Equality for an aggregate's argument list, used to match an aggregate against
/// the extracted list during dedup and post-aggregate rewrite. The fast path is
/// structural equality, which covers every subquery-free argument. When that
/// fails only because an argument contains a subquery (BoundExpr compares
/// subqueries as always-unequal, a conservative choice for the optimizer's
/// change-detection), it falls back to a Debug-representation comparison: the
/// derived Debug reflects every field, including the subquery plan, so it stays
/// correct as the AST evolves. The compared expressions are clones of the same
/// bound source, so an identical subquery formats identically. The fallback runs
/// only for subquery-bearing aggregates, which are rare.
fn agg_args_equal(a: &[BoundExpr], b: &[BoundExpr]) -> bool {
    if a == b {
        return true;
    }
    let has_subquery = a
        .iter()
        .chain(b.iter())
        .any(crate::binder::expr_contains_subquery);
    has_subquery && format!("{a:?}") == format!("{b:?}")
}

/// Removes repeated aggregate expressions so an aggregate referenced in both
/// the projection list and HAVING is computed once. Identity matches the key
/// `rewrite_post_aggregate` uses: function name, args, and DISTINCT flag.
fn dedup_aggregates(aggregates: &mut Vec<AggregateExpr>) {
    let mut seen: Vec<AggregateExpr> = Vec::new();
    aggregates.retain(|a| {
        let dup = seen.iter().any(|s| {
            s.function_name == a.function_name
                && agg_args_equal(&s.args, &a.args)
                && s.distinct == a.distinct
        });
        if dup {
            false
        } else {
            seen.push(a.clone());
            true
        }
    });
}

/// Walks a projection expression that sits above an `Aggregate` plan node and
/// rewrites every `AggregateFunction` and group-by sub-expression into a
/// `ColumnRef` that points at the aggregate's output schema.
///
/// Aggregate output column layout: `group_by[0..n]` first, then
/// `aggregates[0..m]` at indices `n..(n+m)`. The `table_idx` on each
/// generated `ColumnRef` is `AGGREGATE_TABLE_IDX`, which matches the
/// synthetic table_idx that `HashAggregate` and `SortAggregate` set on
/// their output schemas.
fn rewrite_post_aggregate(
    expr: &mut BoundExpr,
    group_by: &[BoundExpr],
    aggregates: &[AggregateExpr],
) {
    // Group-by exprs at the top level of the projection collapse to a
    // ColumnRef into the i-th group output column.
    if let Some(i) = group_by.iter().position(|g| g == expr) {
        let g = &group_by[i];
        *expr = BoundExpr::ColumnRef(crate::binder::ColumnRef {
            table_idx: AGGREGATE_TABLE_IDX,
            column_id: ColumnId(i as u16),
            type_id: g.type_id(),
            nullable: g.nullable(),
            fractional_digits: g.fractional_digits(),
        });
        return;
    }

    match expr {
        BoundExpr::AggregateFunction {
            name,
            args,
            distinct,
            return_type,
            uda: _,
        } => {
            let needle = AggregateExpr {
                function_name: name.clone(),
                args: args.clone(),
                distinct: *distinct,
                return_type: *return_type,
                uda: None,
            };
            let Some(agg_idx) = aggregates.iter().position(|a| {
                a.function_name == needle.function_name
                    && agg_args_equal(&a.args, &needle.args)
                    && a.distinct == needle.distinct
            }) else {
                // Defensive: every aggregate in the projection was collected into
                // the list by extract_aggregates, so a match always exists
                // (agg_args_equal handles subquery-bearing args). If one somehow
                // is not found, leave the node in place so the executor returns a
                // clear error rather than the planner panicking.
                return;
            };
            let column_idx = group_by.len() + agg_idx;
            // A decimal aggregate keeps its argument's scale, so a HAVING
            // or ORDER BY reading the output column compares the value
            // rather than the raw scaled integer
            let fractional_digits = if *return_type == zyron_common::TypeId::Decimal {
                args.first().and_then(|a| a.fractional_digits())
            } else {
                None
            };
            *expr = BoundExpr::ColumnRef(crate::binder::ColumnRef {
                table_idx: AGGREGATE_TABLE_IDX,
                column_id: ColumnId(column_idx as u16),
                type_id: *return_type,
                nullable: true,
                fractional_digits,
            });
        }
        BoundExpr::BinaryOp { left, right, .. } => {
            rewrite_post_aggregate(left, group_by, aggregates);
            rewrite_post_aggregate(right, group_by, aggregates);
        }
        BoundExpr::UnaryOp { expr: inner, .. } => {
            rewrite_post_aggregate(inner, group_by, aggregates);
        }
        BoundExpr::Function { args, .. } => {
            for arg in args.iter_mut() {
                rewrite_post_aggregate(arg, group_by, aggregates);
            }
        }
        BoundExpr::Nested(inner) => {
            rewrite_post_aggregate(inner, group_by, aggregates);
        }
        BoundExpr::Case {
            operand,
            conditions,
            else_result,
            ..
        } => {
            if let Some(op) = operand.as_mut() {
                rewrite_post_aggregate(op, group_by, aggregates);
            }
            for wc in conditions.iter_mut() {
                rewrite_post_aggregate(&mut wc.condition, group_by, aggregates);
                rewrite_post_aggregate(&mut wc.result, group_by, aggregates);
            }
            if let Some(e) = else_result.as_mut() {
                rewrite_post_aggregate(e, group_by, aggregates);
            }
        }
        BoundExpr::Cast { expr: inner, .. } => {
            rewrite_post_aggregate(inner, group_by, aggregates);
        }
        BoundExpr::IsNull { expr: inner, .. } => {
            rewrite_post_aggregate(inner, group_by, aggregates);
        }
        BoundExpr::InList {
            expr: inner, list, ..
        } => {
            rewrite_post_aggregate(inner, group_by, aggregates);
            for item in list.iter_mut() {
                rewrite_post_aggregate(item, group_by, aggregates);
            }
        }
        BoundExpr::Between {
            expr: inner,
            low,
            high,
            ..
        } => {
            rewrite_post_aggregate(inner, group_by, aggregates);
            rewrite_post_aggregate(low, group_by, aggregates);
            rewrite_post_aggregate(high, group_by, aggregates);
        }
        BoundExpr::Like {
            expr: inner,
            pattern,
            ..
        }
        | BoundExpr::ILike {
            expr: inner,
            pattern,
            ..
        } => {
            rewrite_post_aggregate(inner, group_by, aggregates);
            rewrite_post_aggregate(pattern, group_by, aggregates);
        }
        // A subquery in HAVING may reference a group-by key of the enclosing
        // query. Rewrite those references to the aggregate's output column so
        // the correlated filter resolves them against the aggregate's result
        // (the group value) rather than the pre-aggregation base column, which
        // the HAVING filter's input no longer carries. Only group-key
        // references are rewritten inside the subquery; its own aggregates and
        // columns are left intact.
        BoundExpr::Subquery { plan, .. } | BoundExpr::Exists { plan, .. } => {
            rewrite_group_keys_in_select(plan, group_by);
        }
        BoundExpr::InSubquery {
            expr: inner, plan, ..
        } => {
            rewrite_post_aggregate(inner, group_by, aggregates);
            rewrite_group_keys_in_select(plan, group_by);
        }
        BoundExpr::WindowFunction {
            function,
            partition_by,
            order_by,
            ..
        } => {
            // Rewrite aggregates nested inside the window's arguments and the
            // partition/order expressions, but leave the windowed function node
            // itself for the window operator.
            match function.as_mut() {
                BoundExpr::Function { args, .. } | BoundExpr::AggregateFunction { args, .. } => {
                    for arg in args.iter_mut() {
                        rewrite_post_aggregate(arg, group_by, aggregates);
                    }
                }
                _ => {}
            }
            for p in partition_by.iter_mut() {
                rewrite_post_aggregate(p, group_by, aggregates);
            }
            for o in order_by.iter_mut() {
                rewrite_post_aggregate(&mut o.expr, group_by, aggregates);
            }
        }
        BoundExpr::TemporalRef { inner, .. } => {
            rewrite_post_aggregate(inner, group_by, aggregates);
        }
        _ => {}
    }
}

/// Rewrites references that match an enclosing group-by expression into the
/// aggregate's output column, recursing through all sub-expressions including
/// aggregate arguments and nested subqueries. Unlike `rewrite_post_aggregate`
/// it never rewrites an aggregate node itself, so a subquery's own aggregates
/// (which may share a name with an enclosing aggregate) are preserved. Used to
/// resolve a correlated group-key reference inside a HAVING subquery.
fn rewrite_group_keys(expr: &mut BoundExpr, group_by: &[BoundExpr]) {
    if let Some(i) = group_by.iter().position(|g| g == expr) {
        let g = &group_by[i];
        *expr = BoundExpr::ColumnRef(crate::binder::ColumnRef {
            table_idx: AGGREGATE_TABLE_IDX,
            column_id: ColumnId(i as u16),
            type_id: g.type_id(),
            nullable: g.nullable(),
            fractional_digits: g.fractional_digits(),
        });
        return;
    }
    match expr {
        BoundExpr::BinaryOp { left, right, .. } => {
            rewrite_group_keys(left, group_by);
            rewrite_group_keys(right, group_by);
        }
        BoundExpr::UnaryOp { expr: inner, .. }
        | BoundExpr::IsNull { expr: inner, .. }
        | BoundExpr::Cast { expr: inner, .. }
        | BoundExpr::Nested(inner)
        | BoundExpr::TemporalRef { inner, .. } => rewrite_group_keys(inner, group_by),
        BoundExpr::Between {
            expr: inner,
            low,
            high,
            ..
        } => {
            rewrite_group_keys(inner, group_by);
            rewrite_group_keys(low, group_by);
            rewrite_group_keys(high, group_by);
        }
        BoundExpr::InList {
            expr: inner, list, ..
        } => {
            rewrite_group_keys(inner, group_by);
            for item in list.iter_mut() {
                rewrite_group_keys(item, group_by);
            }
        }
        BoundExpr::Like {
            expr: inner,
            pattern,
            ..
        }
        | BoundExpr::ILike {
            expr: inner,
            pattern,
            ..
        } => {
            rewrite_group_keys(inner, group_by);
            rewrite_group_keys(pattern, group_by);
        }
        BoundExpr::Function { args, .. } | BoundExpr::AggregateFunction { args, .. } => {
            for arg in args.iter_mut() {
                rewrite_group_keys(arg, group_by);
            }
        }
        BoundExpr::Case {
            operand,
            conditions,
            else_result,
            ..
        } => {
            if let Some(op) = operand.as_mut() {
                rewrite_group_keys(op, group_by);
            }
            for wc in conditions.iter_mut() {
                rewrite_group_keys(&mut wc.condition, group_by);
                rewrite_group_keys(&mut wc.result, group_by);
            }
            if let Some(e) = else_result.as_mut() {
                rewrite_group_keys(e, group_by);
            }
        }
        BoundExpr::WindowFunction {
            function,
            partition_by,
            order_by,
            ..
        } => {
            rewrite_group_keys(function, group_by);
            for e in partition_by.iter_mut() {
                rewrite_group_keys(e, group_by);
            }
            for o in order_by.iter_mut() {
                rewrite_group_keys(&mut o.expr, group_by);
            }
        }
        BoundExpr::Subquery { plan, .. } | BoundExpr::Exists { plan, .. } => {
            rewrite_group_keys_in_select(plan, group_by);
        }
        BoundExpr::InSubquery {
            expr: inner, plan, ..
        } => {
            rewrite_group_keys(inner, group_by);
            rewrite_group_keys_in_select(plan, group_by);
        }
        BoundExpr::ColumnRef(_) | BoundExpr::Literal { .. } | BoundExpr::Parameter { .. } => {}
    }
}

/// Applies `rewrite_group_keys` to the expression positions of a subquery plan.
/// The subquery's own GROUP BY list is left untouched so its grouping stands;
/// only references matching an enclosing group-by expression are rewritten.
fn rewrite_group_keys_in_select(plan: &mut BoundSelect, group_by: &[BoundExpr]) {
    for item in &mut plan.projections {
        if let BoundSelectItem::Expr(e, _) = item {
            rewrite_group_keys(e, group_by);
        }
    }
    if let Some(w) = &mut plan.where_clause {
        rewrite_group_keys(w, group_by);
    }
    if let Some(h) = &mut plan.having {
        rewrite_group_keys(h, group_by);
    }
    for o in &mut plan.order_by {
        rewrite_group_keys(&mut o.expr, group_by);
    }
    if let Some(l) = &mut plan.limit {
        rewrite_group_keys(l, group_by);
    }
    if let Some(o) = &mut plan.offset {
        rewrite_group_keys(o, group_by);
    }
    for item in &mut plan.from {
        rewrite_group_keys_in_from(item, group_by);
    }
    for sop in &mut plan.set_ops {
        rewrite_group_keys_in_select(&mut sop.right, group_by);
    }
}

fn rewrite_group_keys_in_from(item: &mut BoundFromItem, group_by: &[BoundExpr]) {
    match item {
        BoundFromItem::Join {
            left,
            right,
            condition,
            ..
        } => {
            rewrite_group_keys_in_from(left, group_by);
            rewrite_group_keys_in_from(right, group_by);
            if let BoundJoinCondition::On(e) = condition {
                rewrite_group_keys(e, group_by);
            }
        }
        BoundFromItem::Subquery { query, .. } => rewrite_group_keys_in_select(query, group_by),
        _ => {}
    }
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
            uda,
        } => {
            *has_agg = true;
            out.push(AggregateExpr {
                function_name: name.clone(),
                args: args.clone(),
                distinct: *distinct,
                return_type: *return_type,
                uda: uda.clone(),
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
        BoundExpr::IsNull { expr, .. } => {
            collect_aggregates_from_expr(expr, out, has_agg);
        }
        BoundExpr::InList { expr, list, .. } => {
            collect_aggregates_from_expr(expr, out, has_agg);
            for item in list {
                collect_aggregates_from_expr(item, out, has_agg);
            }
        }
        BoundExpr::Between {
            expr, low, high, ..
        } => {
            collect_aggregates_from_expr(expr, out, has_agg);
            collect_aggregates_from_expr(low, out, has_agg);
            collect_aggregates_from_expr(high, out, has_agg);
        }
        BoundExpr::Like { expr, pattern, .. } | BoundExpr::ILike { expr, pattern, .. } => {
            collect_aggregates_from_expr(expr, out, has_agg);
            collect_aggregates_from_expr(pattern, out, has_agg);
        }
        BoundExpr::WindowFunction {
            function,
            partition_by,
            order_by,
            ..
        } => {
            // The window's own function (e.g. sum(x) OVER ...) is computed by
            // the window operator, so it is not collected as a grouping
            // aggregate. Only aggregates nested inside its arguments and inside
            // the partition/order expressions are grouping aggregates.
            for arg in window_function_args(function) {
                collect_aggregates_from_expr(arg, out, has_agg);
            }
            for p in partition_by {
                collect_aggregates_from_expr(p, out, has_agg);
            }
            for o in order_by {
                collect_aggregates_from_expr(&o.expr, out, has_agg);
            }
        }
        BoundExpr::TemporalRef { inner, .. } => {
            collect_aggregates_from_expr(inner, out, has_agg);
        }
        _ => {}
    }
}

/// Returns the argument expressions of a window's inner function so an
/// aggregate nested in an argument is reachable, while the function node itself
/// (the windowed computation) is not treated as a grouping aggregate.
fn window_function_args(function: &BoundExpr) -> &[BoundExpr] {
    match function {
        BoundExpr::Function { args, .. } | BoundExpr::AggregateFunction { args, .. } => args,
        _ => &[],
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
            // The VALUES rows are in target-column order, so the schema must be
            // too. Look each target column up by id (positions are not column
            // ids) so a reordered or partial column list types correctly.
            let mut schema: Vec<LogicalColumn> = Vec::with_capacity(insert.target_columns.len());
            for col_id in &insert.target_columns {
                let c = insert
                    .table_entry
                    .columns
                    .iter()
                    .find(|c| c.id == *col_id)
                    .ok_or_else(|| {
                        zyron_common::ZyronError::PlanError(format!(
                            "insert target column id {} not found in table",
                            col_id.0
                        ))
                    })?;
                schema.push(LogicalColumn {
                    table_idx: None,
                    column_id: c.id,
                    name: c.name.clone(),
                    type_id: c.type_id,
                    nullable: c.nullable,
                    fractional_digits: c.fractional_digits,
                });
            }
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
        column_defaults: insert.column_defaults.clone(),
        check_constraints: insert.check_constraints.clone(),
        expectations: insert.expectations.clone(),
        source: Arc::new(source),
    })
}

fn build_update_plan(update: &BoundUpdate) -> Result<LogicalPlan> {
    // Scan the target table. The columns must carry the same `table_idx` as
    // the Scan node so column references in WHERE and SET resolve through
    // the executor's `(table_idx, column_id)` lookup.
    const TABLE_IDX: usize = 0;
    let columns: Vec<LogicalColumn> = update
        .table_entry
        .columns
        .iter()
        .map(|c| LogicalColumn {
            table_idx: Some(TABLE_IDX),
            column_id: c.id,
            name: c.name.clone(),
            type_id: c.type_id,
            nullable: c.nullable,
            fractional_digits: c.fractional_digits,
        })
        .collect();

    let mut plan = LogicalPlan::Scan {
        table_id: update.table_id,
        table_idx: TABLE_IDX,
        columns,
        alias: update.table_entry.name.clone(),
        encoding_hints: None,
        as_of: None,
    };

    // Apply WHERE filter
    if let Some(predicate) = &update.where_clause {
        plan = LogicalPlan::Filter {
            predicate: predicate.clone(),
            child: Arc::new(plan),
        };
    }

    Ok(LogicalPlan::Update {
        table_id: update.table_id,
        assignments: update.assignments.clone(),
        check_constraints: update.check_constraints.clone(),
        child: Arc::new(plan),
    })
}

fn build_delete_plan(delete: &BoundDelete) -> Result<LogicalPlan> {
    // Same column->scan table_idx alignment as build_update_plan.
    const TABLE_IDX: usize = 0;
    let columns: Vec<LogicalColumn> = delete
        .table_entry
        .columns
        .iter()
        .map(|c| LogicalColumn {
            table_idx: Some(TABLE_IDX),
            column_id: c.id,
            name: c.name.clone(),
            type_id: c.type_id,
            nullable: c.nullable,
            fractional_digits: c.fractional_digits,
        })
        .collect();

    let mut plan = LogicalPlan::Scan {
        table_id: delete.table_id,
        table_idx: TABLE_IDX,
        columns,
        alias: delete.table_entry.name.clone(),
        encoding_hints: None,
        as_of: None,
    };

    if let Some(predicate) = &delete.where_clause {
        plan = LogicalPlan::Filter {
            predicate: predicate.clone(),
            child: Arc::new(plan),
        };
    }

    Ok(LogicalPlan::Delete {
        table_id: delete.table_id,
        child: Arc::new(plan),
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
                uda: None,
            },
            Some("cnt".to_string()),
        )];
        let (has_agg, aggs) = extract_aggregates(&projections, None, &[]);
        assert!(has_agg);
        assert_eq!(aggs.len(), 1);
        assert_eq!(aggs[0].function_name, "count");
    }

    fn count_star() -> BoundExpr {
        BoundExpr::AggregateFunction {
            name: "count".to_string(),
            args: vec![BoundExpr::Literal {
                value: LiteralValue::Integer(1),
                type_id: TypeId::Int64,
            }],
            distinct: false,
            return_type: TypeId::Int64,
            uda: None,
        }
    }

    // An aggregate that appears only in HAVING must still be collected so the
    // Aggregate node computes it.
    #[test]
    fn test_extract_aggregates_finds_having_only_count() {
        let projections: Vec<BoundSelectItem> = vec![];
        let having = BoundExpr::BinaryOp {
            left: Box::new(count_star()),
            op: zyron_parser::ast::BinaryOperator::Gt,
            right: Box::new(BoundExpr::Literal {
                value: LiteralValue::Integer(1),
                type_id: TypeId::Int64,
            }),
            type_id: TypeId::Boolean,
        };
        let (has_agg, aggs) = extract_aggregates(&projections, Some(&having), &[]);
        assert!(has_agg, "HAVING aggregate must be detected");
        assert_eq!(aggs.len(), 1);
        assert_eq!(aggs[0].function_name, "count");
    }

    // The same aggregate in both projection and HAVING is computed once.
    #[test]
    fn test_extract_aggregates_dedups_shared_count() {
        let projections = vec![BoundSelectItem::Expr(count_star(), Some("c".to_string()))];
        let having = BoundExpr::BinaryOp {
            left: Box::new(count_star()),
            op: zyron_parser::ast::BinaryOperator::Gt,
            right: Box::new(BoundExpr::Literal {
                value: LiteralValue::Integer(1),
                type_id: TypeId::Int64,
            }),
            type_id: TypeId::Boolean,
        };
        let (_has_agg, aggs) = extract_aggregates(&projections, Some(&having), &[]);
        assert_eq!(aggs.len(), 1, "shared aggregate is deduped");
    }

    // A HAVING-only aggregate is rewritten to a column reference into the
    // aggregate output, never left as a raw AggregateFunction the filter would
    // try to evaluate.
    #[test]
    fn test_rewrite_having_aggregate_to_column_ref() {
        let aggregates = vec![AggregateExpr {
            function_name: "count".to_string(),
            args: vec![BoundExpr::Literal {
                value: LiteralValue::Integer(1),
                type_id: TypeId::Int64,
            }],
            distinct: false,
            return_type: TypeId::Int64,
            uda: None,
        }];
        let group_by = vec![BoundExpr::ColumnRef(crate::binder::ColumnRef {
            table_idx: 0,
            column_id: ColumnId(0),
            type_id: TypeId::Int64,
            nullable: false,
            fractional_digits: None,
        })];
        let mut having = BoundExpr::BinaryOp {
            left: Box::new(count_star()),
            op: zyron_parser::ast::BinaryOperator::Gt,
            right: Box::new(BoundExpr::Literal {
                value: LiteralValue::Integer(1),
                type_id: TypeId::Int64,
            }),
            type_id: TypeId::Boolean,
        };
        rewrite_post_aggregate(&mut having, &group_by, &aggregates);
        // count is aggregate index 0; with one group key the output column is 1.
        match having {
            BoundExpr::BinaryOp { left, .. } => match *left {
                BoundExpr::ColumnRef(cr) => {
                    assert_eq!(cr.table_idx, AGGREGATE_TABLE_IDX);
                    assert_eq!(cr.column_id, ColumnId(1));
                }
                other => panic!("aggregate not rewritten: {other:?}"),
            },
            other => panic!("unexpected shape: {other:?}"),
        }
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
        let (has_agg, aggs) = extract_aggregates(&projections, None, &[]);
        assert!(!has_agg);
        assert!(aggs.is_empty());
    }
}

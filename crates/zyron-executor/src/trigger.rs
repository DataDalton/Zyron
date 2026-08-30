//! Trigger firing for DML operators.
//!
//! On a matching INSERT/UPDATE/DELETE event a table's triggers run a stored
//! procedure: for a row-level trigger the procedure runs once per affected row
//! with the row's columns bound as positional parameters ($1..$N); a
//! statement-level trigger runs once with no parameters. The action runs in a
//! nested execution context that shares the firing statement's transaction,
//! snapshot, and index caches, so trigger effects commit and roll back with the
//! triggering statement. A depth guard bounds re-triggering.

use std::sync::Arc;

use zyron_catalog::{ColumnEntry, TableId, TriggerEntry};
use zyron_common::{Result, ZyronError};

use crate::batch::DataBatch;
use crate::column::ScalarValue;
use crate::context::ExecutionContext;

/// Maximum trigger nesting depth. A trigger whose action re-fires triggers past
/// this depth aborts the statement rather than recursing without bound.
const MAX_TRIGGER_DEPTH: usize = 16;

/// Fires the triggers defined on `table_id` that match `timing` and `event`.
/// `batch` holds the affected rows in table-column order: the NEW image for
/// INSERT/UPDATE, the OLD image for DELETE.
pub async fn fire_row_triggers(
    ctx: &Arc<ExecutionContext>,
    table_id: TableId,
    timing: u8,
    event: u8,
    batch: &DataBatch,
    columns: &[ColumnEntry],
) -> Result<()> {
    // A trigger fired on the leader and its row effects are in the changeset
    // this apply is replaying. Firing it again here would run its side effects
    // a second time and write rows nobody agreed on
    if ctx.replication_apply {
        return Ok(());
    }
    let triggers = ctx.catalog.triggers_for_table(table_id);
    if triggers.is_empty() {
        return Ok(());
    }
    if ctx.trigger_depth >= MAX_TRIGGER_DEPTH {
        return Err(ZyronError::ExecutionError(format!(
            "trigger recursion exceeded the maximum depth of {MAX_TRIGGER_DEPTH}"
        )));
    }

    for trig in &triggers {
        if !trig.enabled || trig.timing != timing || (trig.events & event) == 0 {
            continue;
        }
        // The definition stored a canonical schema.name at CREATE TRIGGER,
        // so the body fires the exact procedure it was bound to no matter
        // which session or namespace triggers it
        let proc = ctx
            .catalog
            .resolve_procedure_canonical(&trig.execute_function)
            .ok_or_else(|| {
                ZyronError::ExecutionError(format!(
                    "trigger '{}' references undefined procedure '{}'",
                    trig.name, trig.execute_function
                ))
            })?;
        let body_stmts = zyron_parser::parse(&proc.body_sql).map_err(|e| {
            ZyronError::ExecutionError(format!(
                "trigger '{}' procedure body parse error: {e}",
                trig.name
            ))
        })?;

        let body_plans = plan_trigger_body(ctx, &body_stmts).await?;
        if trig.for_each == TriggerEntry::FOR_EACH_STATEMENT {
            run_trigger_plans(ctx, &body_plans, &[]).await?;
        } else {
            for row in 0..batch.num_rows {
                let params: Vec<ScalarValue> = columns
                    .iter()
                    .enumerate()
                    .map(|(c, _)| {
                        let col = &batch.columns[c];
                        if col.is_null(row) {
                            ScalarValue::Null
                        } else {
                            col.data.get_scalar(row)
                        }
                    })
                    .collect();
                run_trigger_plans(ctx, &body_plans, &params).await?;
            }
        }
    }
    Ok(())
}

/// Fires the INSTEAD OF triggers a view defines for `event`. `batch` holds
/// one source row per affected view row; `param_map` routes each trigger
/// parameter position to a batch column, None binding NULL. For INSERT the
/// parameters are the NEW image in view column order, for DELETE the OLD
/// image, for UPDATE the OLD image followed by the NEW image.
pub async fn fire_instead_of_triggers(
    ctx: &Arc<ExecutionContext>,
    view_id: u32,
    event: u8,
    batch: &DataBatch,
    param_map: &[Option<usize>],
) -> Result<()> {
    // On a follower the leader's trigger effects arrive as row changes in the
    // replicated changeset; re-firing here would write them twice
    if ctx.replication_apply {
        return Ok(());
    }
    if ctx.trigger_depth >= MAX_TRIGGER_DEPTH {
        return Err(ZyronError::ExecutionError(format!(
            "trigger recursion exceeded the maximum depth of {MAX_TRIGGER_DEPTH}"
        )));
    }
    let triggers: Vec<_> = ctx
        .catalog
        .triggers_for_table(TableId(view_id))
        .into_iter()
        .filter(|t| {
            t.enabled && t.timing == TriggerEntry::TIMING_INSTEAD_OF && (t.events & event) != 0
        })
        .collect();
    if triggers.is_empty() {
        // The binder verified the trigger, so reaching execution without one
        // means it was dropped or disabled in between
        return Err(ZyronError::ExecutionError(format!(
            "view write reached execution but view id {view_id} no longer has an enabled INSTEAD OF trigger for the event"
        )));
    }

    for trig in &triggers {
        // The definition stored a canonical schema.name at CREATE TRIGGER,
        // so the body fires the exact procedure it was bound to no matter
        // which session or namespace triggers it
        let proc = ctx
            .catalog
            .resolve_procedure_canonical(&trig.execute_function)
            .ok_or_else(|| {
                ZyronError::ExecutionError(format!(
                    "trigger '{}' references undefined procedure '{}'",
                    trig.name, trig.execute_function
                ))
            })?;
        let body_stmts = zyron_parser::parse(&proc.body_sql).map_err(|e| {
            ZyronError::ExecutionError(format!(
                "trigger '{}' procedure body parse error: {e}",
                trig.name
            ))
        })?;

        let body_plans = plan_trigger_body(ctx, &body_stmts).await?;
        for row in 0..batch.num_rows {
            let mut params: Vec<ScalarValue> = Vec::with_capacity(param_map.len());
            for slot in param_map {
                match slot {
                    Some(c) => {
                        let col = batch.columns.get(*c).ok_or_else(|| {
                            ZyronError::ExecutionError(format!(
                                "view trigger parameter maps to source column {c} but the source produced {} columns",
                                batch.columns.len()
                            ))
                        })?;
                        params.push(if col.is_null(row) {
                            ScalarValue::Null
                        } else {
                            col.data.get_scalar(row)
                        });
                    }
                    None => params.push(ScalarValue::Null),
                }
            }
            run_trigger_plans(ctx, &body_plans, &params).await?;
        }
    }
    Ok(())
}

/// Plans a trigger body's statements once per firing. The row values bind as
/// $1..$N parameters at execution time, so one plan serves every affected
/// row; re-planning per row would repeat the parse-bind-optimize work N times
/// for identical plans. Bodies are DML and queries (DDL never reaches the
/// planner), so nothing a body statement executes can invalidate a sibling's
/// plan.
async fn plan_trigger_body(
    ctx: &Arc<ExecutionContext>,
    stmts: &[zyron_parser::Statement],
) -> Result<Vec<zyron_planner::physical::PhysicalPlan>> {
    let mut plans = Vec::with_capacity(stmts.len());
    for stmt in stmts {
        // The body is planned against the same peer facts the firing
        // statement was planned against. Binding is async, so this takes the
        // snapshot pointer rather than a guard: a lock held across the bind
        // would block every peer declaration behind it, and the pointer
        // copies nothing
        let peerFacts = ctx.peers.as_ref().map(|p| Arc::clone(&p.read()));
        // A stored body means the same tables no matter which session fires
        // it: user tables must be schema-qualified, and the system path
        // serves zyron_sys reads only. Inheriting the caller's search path
        // would let the same body resolve to different tables per caller
        let plan = zyron_planner::plan(
            &ctx.catalog,
            ctx.planning_database,
            zyron_catalog::default_search_path(),
            stmt.clone(),
            peerFacts.as_deref(),
        )
        .await?;
        plans.push(plan);
    }
    Ok(plans)
}

/// Runs pre-planned trigger body statements in a nested context that shares
/// the firing transaction (same txn_id and snapshot) and index caches, with
/// the row values bound as parameters and the trigger depth incremented.
async fn run_trigger_plans(
    ctx: &Arc<ExecutionContext>,
    plans: &[zyron_planner::physical::PhysicalPlan],
    params: &[ScalarValue],
) -> Result<()> {
    for plan in plans {
        // The child context carries everything the firing statement's context
        // holds, replication capture, undo log, and CDC hook included. A
        // trigger body's writes must land in the same changeset and undo log
        // as the statement that fired it, or they would neither replicate to
        // the group nor reverse on rollback to a savepoint
        let mut nested = ctx.child_with_params(params.to_vec());
        nested.trigger_depth = ctx.trigger_depth + 1;
        let nested = Arc::new(nested);
        let nested_writes = Arc::clone(&nested);

        // Run the action on a fresh task rather than nested inline. Each trigger
        // level otherwise stacks a full execute() poll frame on the previous
        // one; spawning lets the runtime poll the child from its own loop so a
        // chain of triggers cannot overflow the stack (the depth guard bounds
        // the logical recursion). The child shares the txn via the Arc context.
        let plan = plan.clone();
        let handle = tokio::spawn(async move { crate::execute(plan, &nested).await });
        let joined = handle.await;

        // A body that appended WAL must mark the firing context, or the wire
        // layer would see a statement whose own operator wrote nothing and
        // commit the transaction as read-only, skipping the durable commit
        // record and the group proposal. For an INSTEAD OF trigger the body
        // holds the statement's only writes. Propagated before the error
        // check so a partial write is never missed
        if nested_writes.wrote_wal() {
            ctx.mark_wrote_wal();
        }
        joined.map_err(|e| ZyronError::ExecutionError(format!("trigger task failed: {e}")))??;
    }
    Ok(())
}

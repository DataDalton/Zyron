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
        let proc = ctx
            .catalog
            .find_procedure_by_name(&trig.execute_function)
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

        if trig.for_each == TriggerEntry::FOR_EACH_STATEMENT {
            run_trigger_body(ctx, &body_stmts, &[]).await?;
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
                run_trigger_body(ctx, &body_stmts, &params).await?;
            }
        }
    }
    Ok(())
}

/// Runs a trigger procedure's body statements in a nested context that shares
/// the firing transaction (same txn_id and snapshot) and index caches, with the
/// row values bound as parameters and the trigger depth incremented.
async fn run_trigger_body(
    ctx: &Arc<ExecutionContext>,
    stmts: &[zyron_parser::Statement],
    params: &[ScalarValue],
) -> Result<()> {
    for stmt in stmts {
        let plan = zyron_planner::plan(
            &ctx.catalog,
            zyron_catalog::DatabaseId(1),
            vec!["public".to_string()],
            stmt.clone(),
        )
        .await?;

        let mut nested = ExecutionContext::new(
            Arc::clone(&ctx.catalog),
            Arc::clone(&ctx.wal),
            Arc::clone(&ctx.buffer_pool),
            Arc::clone(&ctx.disk_manager),
            ctx.txn_id,
            ctx.snapshot.clone(),
        );
        nested.heap_files = ctx.heap_files.clone();
        nested.btree_indexes = ctx.btree_indexes.clone();
        nested.intent_locks = ctx.intent_locks.clone();
        nested.fts_manager = ctx.fts_manager.clone();
        nested.vector_manager = ctx.vector_manager.clone();
        nested.spatial_manager = ctx.spatial_manager.clone();
        nested.graph_manager = ctx.graph_manager.clone();
        nested.params = params.to_vec();
        nested.trigger_depth = ctx.trigger_depth + 1;
        let nested = Arc::new(nested);

        // Run the action on a fresh task rather than nested inline. Each trigger
        // level otherwise stacks a full execute() poll frame on the previous
        // one; spawning lets the runtime poll the child from its own loop so a
        // chain of triggers cannot overflow the stack (the depth guard bounds
        // the logical recursion). The child shares the txn via the Arc context.
        let handle = tokio::spawn(async move { crate::execute(plan, &nested).await });
        handle
            .await
            .map_err(|e| ZyronError::ExecutionError(format!("trigger task failed: {e}")))??;
    }
    Ok(())
}

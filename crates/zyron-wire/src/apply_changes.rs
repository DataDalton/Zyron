//! APPLY CHANGES, resolved into the plans one pass over the target needs.
//!
//! The statement settles a target from a change set. The winning change per
//! key is a query the planner already knows how to run, a window over the
//! source ordered by the sequence, so it is written as SQL and planned like
//! any other. The target is read once through a planned scan, so the
//! reader's row security is in it, and the writes go through the same
//! operators an INSERT, UPDATE or DELETE statement uses, so constraints,
//! indexes, triggers, the change feed and the replication changeset see the
//! apply as ordinary row writes. The executor's own apply pass joins the two
//! in memory, which is what keeps the cost proportional to the change set
//! plus the target rather than their product.
//!
//! The winner per key is decided by the sequence, so a source delivering out
//! of order settles where an ordered one does. That is also what makes the
//! statement idempotent. Applying one captured range twice computes the same
//! winners and writes the same rows

use std::sync::Arc;

use zyron_catalog::{ColumnId, TableEntry};
use zyron_cdc::apply::{
    SCD2_CURRENT_COLUMN, SCD2_END_COLUMN, SCD2_START_COLUMN, ScdType, TrackHistory,
};
use zyron_common::ZyronError;
use zyron_executor::apply_changes::{
    ApplyCounts, ApplyJob, HistoryColumns, InsertShape, UpdateShape,
};
use zyron_parser::ast::{ApplyChangesStatement, ApplySource, ScdKind, TrackHistoryClause};
use zyron_planner::physical::PhysicalPlan;

use crate::connection::ServerState;
use crate::ddl_dispatch::DdlResult;
use crate::messages::ProtocolError;
use crate::session::Session;

/// The alias the change relation takes inside the generated SQL
const WINNERS: &str = "__zyron_changes";
/// The column the row numbering lands in
const RANK_COLUMN: &str = "__zyron_rank";
/// The column the sequence value lands in
const SEQUENCE_COLUMN: &str = "__zyron_seq";
/// The column the delete decision lands in
const DELETE_COLUMN: &str = "__zyron_is_delete";
/// The column the truncate decision lands in
const TRUNCATE_COLUMN: &str = "__zyron_is_truncate";

/// Writes an identifier the way a generated statement must quote it
fn ident(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

/// Writes a possibly qualified name, quoting each part
fn qualified(name: &str) -> String {
    name.split('.').map(ident).collect::<Vec<_>>().join(".")
}

/// The relation the change set is read from, as SQL.
///
/// A subquery is written without the alias it was parsed with, since the
/// generated statements alias the change set themselves and a relation
/// cannot carry two
fn source_sql(source: &ApplySource) -> Result<String, ProtocolError> {
    let unwritable = |e: zyron_parser::UnparseError| {
        ProtocolError::Database(ZyronError::PlanError(format!(
            "the change source could not be written back as SQL, {e}"
        )))
    };
    match source {
        ApplySource::Named(name) => Ok(qualified(name)),
        ApplySource::Relation(table) => match table.as_ref() {
            zyron_parser::ast::TableRef::Subquery { query, .. } => Ok(format!(
                "({})",
                zyron_parser::unparse::select_to_sql(query).map_err(unwritable)?
            )),
            other => zyron_parser::table_ref_to_sql(other).map_err(unwritable),
        },
    }
}

/// What orders two changes to one key
fn sequence_sql(stmt: &ApplyChangesStatement) -> String {
    match &stmt.sequence_by {
        Some(expr) => zyron_parser::expr_to_sql(expr),
        // A source that names no sequence is ordered by when the change was
        // committed, which is the order it was recorded in
        None => ident("_commit_version"),
    }
}

/// The predicate that turns a change into a delete
fn delete_sql(stmt: &ApplyChangesStatement) -> String {
    match &stmt.delete_when {
        Some(expr) => zyron_parser::expr_to_sql(expr),
        None => format!("{WINNERS}.{} = 'delete'", ident("_change_type")),
    }
}

/// The predicate that turns a change into a target truncate
fn truncate_sql(stmt: &ApplyChangesStatement) -> Option<String> {
    stmt.truncate_when.as_ref().map(zyron_parser::expr_to_sql)
}

/// The columns the apply writes into the target.
///
/// The target's own columns that the source also carries, less the metadata
/// ones, less what EXCEPT COLUMNS named, and less the reserved history
/// columns, which the apply itself maintains
fn applied_columns(
    stmt: &ApplyChangesStatement,
    target: &TableEntry,
    source_columns: &[String],
) -> Vec<String> {
    target
        .live_columns()
        .map(|c| c.name.clone())
        .filter(|name| !name.starts_with('_'))
        .filter(|name| {
            source_columns
                .iter()
                .any(|source| source.eq_ignore_ascii_case(name))
        })
        .filter(|name| {
            !stmt
                .except_columns
                .iter()
                .any(|dropped| dropped.eq_ignore_ascii_case(name))
        })
        .filter(|name| {
            ![SCD2_START_COLUMN, SCD2_END_COLUMN, SCD2_CURRENT_COLUMN]
                .iter()
                .any(|reserved| reserved.eq_ignore_ascii_case(name))
        })
        .collect()
}

/// The ranked change relation, every change of the source, numbered within
/// its key by the sequence, so the winner is the row numbered one.
///
/// A preimage never wins, because what a target should hold is what the row
/// became rather than what it was
fn ranked_sql(
    stmt: &ApplyChangesStatement,
    source_columns: &[String],
) -> Result<String, ProtocolError> {
    let source = source_sql(&stmt.source)?;
    let keys = stmt
        .keys
        .iter()
        .map(|k| ident(k))
        .collect::<Vec<_>>()
        .join(", ");
    let sequence = sequence_sql(stmt);
    let columns = source_columns
        .iter()
        .map(|c| format!("{WINNERS}.{}", ident(c)))
        .collect::<Vec<_>>()
        .join(", ");
    Ok(format!(
        "(SELECT {columns}, ROW_NUMBER() OVER (PARTITION BY {keys} ORDER BY {sequence} DESC, \
         {ordinal} DESC) AS {rank} FROM {source} AS {WINNERS} WHERE {change_type} <> \
         'update_preimage')",
        ordinal = ident("_change_ordinal"),
        rank = ident(RANK_COLUMN),
        change_type = ident("_change_type"),
    ))
}

/// The changes the apply reads, as one relation, the applied columns, the
/// sequence value, the delete and truncate decisions and each change's rank
/// within its key, one being the newest. Type 1 reads the winner alone,
/// since the last change per key is all it lands. Type 2 reads every change
/// of a key, because each one is a version of the row. The executor converts
/// each applied column to the target column's type as it takes the rows in
fn winners_sql(
    stmt: &ApplyChangesStatement,
    target: &TableEntry,
    applied: &[String],
    source_columns: &[String],
) -> Result<String, ProtocolError> {
    let ranked = ranked_sql(stmt, source_columns)?;
    let mut select: Vec<String> = Vec::with_capacity(applied.len() + 4);
    for column in applied {
        let Some(own) = target
            .live_columns()
            .find(|c| c.name.eq_ignore_ascii_case(column))
        else {
            continue;
        };
        select.push(format!("{WINNERS}.{name}", name = ident(&own.name)));
    }
    select.push(format!(
        "({}) AS {}",
        sequence_sql(stmt),
        ident(SEQUENCE_COLUMN)
    ));
    select.push(format!(
        "COALESCE(({}), FALSE) AS {}",
        delete_sql(stmt),
        ident(DELETE_COLUMN)
    ));
    if let Some(truncate) = truncate_sql(stmt) {
        select.push(format!(
            "COALESCE(({truncate}), FALSE) AS {}",
            ident(TRUNCATE_COLUMN)
        ));
    }
    select.push(format!("{WINNERS}.{}", ident(RANK_COLUMN)));
    let kept = match scd_of(stmt) {
        ScdType::Type1 => format!(" WHERE {} = 1", ident(RANK_COLUMN)),
        ScdType::Type2 => String::new(),
    };
    Ok(format!(
        "SELECT {} FROM {ranked} AS {WINNERS}{kept}",
        select.join(", ")
    ))
}

/// Which columns open a new version of a key under type 2
fn tracked_columns(stmt: &ApplyChangesStatement, applied: &[String]) -> Vec<String> {
    let history = match &stmt.track_history {
        None => TrackHistory::All,
        Some(TrackHistoryClause::On(names)) => TrackHistory::On(names.clone()),
        Some(TrackHistoryClause::Except(names)) => TrackHistory::Except(names.clone()),
    };
    applied
        .iter()
        .filter(|column| !stmt.keys.iter().any(|key| key.eq_ignore_ascii_case(column)))
        .filter(|column| history.versions_on(column))
        .cloned()
        .collect()
}

/// The history shape the statement asked for
fn scd_of(stmt: &ApplyChangesStatement) -> ScdType {
    match stmt.scd {
        ScdKind::Type1 => ScdType::Type1,
        ScdKind::Type2 => ScdType::Type2,
    }
}

/// Refuses a target that cannot carry what the apply writes
fn check_target(
    stmt: &ApplyChangesStatement,
    target: &TableEntry,
    applied: &[String],
) -> Result<(), ProtocolError> {
    for key in &stmt.keys {
        if !target
            .live_columns()
            .any(|c| c.name.eq_ignore_ascii_case(key))
        {
            return Err(ProtocolError::Database(ZyronError::PlanError(format!(
                "APPLY CHANGES names key column '{key}', which target '{}' does not have",
                target.name
            ))));
        }
        if !applied.iter().any(|c| c.eq_ignore_ascii_case(key)) {
            return Err(ProtocolError::Database(ZyronError::PlanError(format!(
                "APPLY CHANGES names key column '{key}', which the change source does not carry"
            ))));
        }
    }
    if scd_of(stmt) == ScdType::Type2 {
        for (reserved, expected) in [
            (SCD2_START_COLUMN, zyron_common::TypeId::Int64),
            (SCD2_END_COLUMN, zyron_common::TypeId::Int64),
            (SCD2_CURRENT_COLUMN, zyron_common::TypeId::Boolean),
        ] {
            let Some(column) = target
                .live_columns()
                .find(|c| c.name.eq_ignore_ascii_case(reserved))
            else {
                return Err(ProtocolError::Database(ZyronError::PlanError(format!(
                    "APPLY CHANGES STORED AS SCD TYPE 2 writes column '{reserved}', which \
                     target '{}' does not have. Add it before applying history to this target",
                    target.name
                ))));
            };
            let compatible = match expected {
                zyron_common::TypeId::Boolean => column.type_id == zyron_common::TypeId::Boolean,
                _ => matches!(
                    column.type_id,
                    zyron_common::TypeId::Int64
                        | zyron_common::TypeId::Timestamp
                        | zyron_common::TypeId::TimestampTz
                ),
            };
            if !compatible {
                return Err(ProtocolError::Database(ZyronError::PlanError(format!(
                    "APPLY CHANGES STORED AS SCD TYPE 2 writes column '{reserved}' as \
                     {expected:?}, and target '{}' already holds it as {:?}",
                    target.name, column.type_id
                ))));
            }
        }
    }
    if applied.is_empty() {
        return Err(ProtocolError::Database(ZyronError::PlanError(format!(
            "APPLY CHANGES writes no column into target '{}'. EXCEPT COLUMNS left nothing to \
             apply",
            target.name
        ))));
    }
    Ok(())
}

/// Refuses a source column whose type the target column cannot take.
///
/// Each applied column is converted to the target's type as the winners are
/// read, and a conversion the engine would refuse row by row is refused
/// here by name instead
fn check_types(
    target: &TableEntry,
    applied: &[String],
    source: &[zyron_planner::logical::LogicalColumn],
) -> Result<(), ProtocolError> {
    for name in applied {
        let Some(own) = target
            .live_columns()
            .find(|c| c.name.eq_ignore_ascii_case(name))
        else {
            continue;
        };
        let Some(from) = source.iter().find(|c| c.name.eq_ignore_ascii_case(name)) else {
            continue;
        };
        if !converts(from.type_id, own.type_id) {
            return Err(ProtocolError::Database(ZyronError::PlanError(format!(
                "APPLY CHANGES column '{}' is {} in the change source and {} in target '{}', \
                 and one does not convert to the other",
                own.name, from.type_id, own.type_id, target.name
            ))));
        }
    }
    Ok(())
}

/// Whether a value of one type is written into a column of another without
/// changing what it means, the same type, one number into another, one
/// string into another, or one instant into another
fn converts(from: zyron_common::TypeId, to: zyron_common::TypeId) -> bool {
    use zyron_common::TypeId as T;
    let numeric = |t: T| {
        matches!(
            t,
            T::Int8
                | T::Int16
                | T::Int32
                | T::Int64
                | T::Int128
                | T::UInt8
                | T::UInt16
                | T::UInt32
                | T::UInt64
                | T::UInt128
                | T::Float32
                | T::Float64
                | T::Decimal
        )
    };
    let text = |t: T| matches!(t, T::Char | T::Varchar | T::Text);
    let instant = |t: T| matches!(t, T::Timestamp | T::TimestampTz | T::Date);
    from == to
        || from == T::Null
        || (numeric(from) && numeric(to))
        || (text(from) && text(to))
        || (instant(from) && instant(to))
}

/// Plans one statement under the session's row security
async fn plan_for(
    server: &Arc<ServerState>,
    session: &Option<Session>,
    sql: &str,
) -> Result<PhysicalPlan, ProtocolError> {
    let stmt = zyron_parser::parse(sql)
        .map_err(ProtocolError::Database)?
        .into_iter()
        .next()
        .ok_or_else(|| {
            ProtocolError::Database(ZyronError::Internal(
                "an APPLY CHANGES plan was written as no statement".to_string(),
            ))
        })?;
    let (db_id, search_path) = match session.as_ref() {
        Some(s) => (s.database_id, s.search_path.clone()),
        None => (
            zyron_catalog::DatabaseId(1),
            zyron_catalog::default_search_path(),
        ),
    };
    let row_security: Option<Arc<dyn zyron_planner::RowSecurityProvider>> = match (
        &server.security_manager,
        session.as_ref().and_then(|s| s.security_context.as_ref()),
    ) {
        (Some(sm), Some(sc)) => Some(Arc::new(crate::row_security::SmRowSecurityProvider::new(
            Arc::clone(sm),
            sc,
        ))),
        _ => None,
    };
    let peers = server.peer_facts();
    zyron_planner::plan_for_session(
        &server.catalog,
        db_id,
        search_path,
        stmt,
        row_security,
        Some(&peers),
        session.as_ref().and_then(|s| s.temp_tables.clone()),
    )
    .await
    .map_err(ProtocolError::Database)
}

/// The scan under a planned SELECT, which is what a write reads its rows
/// through. A projection that only names the scan's columns in order is
/// stepped over, because the scan already produces them
fn scan_of(plan: PhysicalPlan) -> Result<PhysicalPlan, ProtocolError> {
    match plan {
        PhysicalPlan::Project {
            expressions, child, ..
        } => {
            let schema = child.output_schema();
            let identity = expressions.len() == schema.len()
                && expressions.iter().zip(schema.iter()).all(|(expr, column)| {
                    matches!(expr, zyron_planner::binder::BoundExpr::ColumnRef(cr)
                        if cr.column_id == column.column_id
                            && Some(cr.table_idx) == column.table_idx)
                });
            if identity {
                Ok(*child)
            } else {
                Err(ProtocolError::Database(ZyronError::Internal(
                    "the APPLY CHANGES target scan projected something other than its columns"
                        .to_string(),
                )))
            }
        }
        other => Ok(other),
    }
}

/// Everything an insert into the target carries, from the plan of one
fn insert_shape(plan: PhysicalPlan) -> Result<InsertShape, ProtocolError> {
    match plan {
        PhysicalPlan::Insert {
            target_columns,
            column_defaults,
            check_constraints,
            expectations,
            generated_columns,
            ..
        } => Ok(InsertShape {
            target_columns,
            column_defaults,
            check_constraints,
            expectations,
            generated_columns,
        }),
        _ => Err(ProtocolError::Database(ZyronError::Internal(
            "the APPLY CHANGES insert did not plan as an insert".to_string(),
        ))),
    }
}

/// Everything an update of the target carries, from the plan of one
fn update_shape(plan: PhysicalPlan) -> Result<UpdateShape, ProtocolError> {
    match plan {
        PhysicalPlan::Update {
            check_constraints,
            generated_columns,
            ..
        } => Ok(UpdateShape {
            check_constraints,
            generated_columns,
        }),
        _ => Err(ProtocolError::Database(ZyronError::Internal(
            "the APPLY CHANGES update did not plan as an update".to_string(),
        ))),
    }
}

/// Resolves one APPLY CHANGES into the job the executor runs
pub async fn resolve(
    stmt: &ApplyChangesStatement,
    server: &Arc<ServerState>,
    session: &Option<Session>,
    target: &TableEntry,
) -> Result<ApplyJob, ProtocolError> {
    if stmt.keys.is_empty() {
        return Err(ProtocolError::Database(ZyronError::PlanError(
            "APPLY CHANGES needs KEYS (<column>, ...) to say what identifies a row".to_string(),
        )));
    }
    if target.lake.is_lake() {
        return Err(ProtocolError::Database(ZyronError::PlanError(format!(
            "APPLY CHANGES writes into a heap table, and target '{}' is a lake table. Apply \
             into a heap table and load the lake table from it",
            target.name
        ))));
    }
    // The target's own schema names it in every statement planned here,
    // because a bare name resolves through the session's search path and
    // could land on a table of the same name in another schema
    let target_schema = server
        .catalog
        .get_schema_by_id(target.schema_id)
        .map_err(ProtocolError::Database)?;
    let target_sql = format!("{}.{}", ident(&target_schema.name), ident(&target.name));

    // The source's own columns decide what the apply can write
    let source = plan_for(
        server,
        session,
        &format!("SELECT * FROM {} AS {WINNERS}", source_sql(&stmt.source)?),
    )
    .await?;
    let source_schema = source.output_schema();
    let source_columns: Vec<String> = source_schema.iter().map(|c| c.name.clone()).collect();
    let applied = applied_columns(stmt, target, &source_columns);
    check_target(stmt, target, &applied)?;
    check_types(target, &applied, &source_schema)?;

    let winners = plan_for(
        server,
        session,
        &winners_sql(stmt, target, &applied, &source_columns)?,
    )
    .await?;
    let winner_schema = winners.output_schema();
    let position = |name: &str| -> Result<usize, ProtocolError> {
        winner_schema
            .iter()
            .position(|c| c.name.eq_ignore_ascii_case(name))
            .ok_or_else(|| {
                ProtocolError::Database(ZyronError::Internal(format!(
                    "the APPLY CHANGES winners relation lost column '{name}'"
                )))
            })
    };
    let column_id = |name: &str| -> Result<ColumnId, ProtocolError> {
        target
            .live_columns()
            .find(|c| c.name.eq_ignore_ascii_case(name))
            .map(|c| c.id)
            .ok_or_else(|| {
                ProtocolError::Database(ZyronError::PlanError(format!(
                    "APPLY CHANGES names column '{name}', which target '{}' does not have",
                    target.name
                )))
            })
    };
    let mut applied_ids = Vec::with_capacity(applied.len());
    for name in &applied {
        applied_ids.push((column_id(name)?, position(name)?));
    }
    let keys = stmt
        .keys
        .iter()
        .map(|k| column_id(k))
        .collect::<Result<Vec<_>, _>>()?;

    let history = match scd_of(stmt) {
        ScdType::Type1 => None,
        ScdType::Type2 => {
            let tracked = tracked_columns(stmt, &applied)
                .iter()
                .map(|c| column_id(c))
                .collect::<Result<Vec<_>, _>>()?;
            Some((
                HistoryColumns {
                    start_at: column_id(SCD2_START_COLUMN)?,
                    end_at: column_id(SCD2_END_COLUMN)?,
                    is_current: column_id(SCD2_CURRENT_COLUMN)?,
                },
                tracked,
            ))
        }
    };

    // The target's rows, read the way the reader is allowed to read them
    let target_scan =
        scan_of(plan_for(server, session, &format!("SELECT * FROM {target_sql}")).await?)?;

    // The insert and the update the apply performs, planned for what they
    // carry: defaults, constraints, expectations and generated columns
    let mut insert_columns: Vec<String> = applied.clone();
    if history.is_some() {
        insert_columns.extend([
            SCD2_START_COLUMN.to_string(),
            SCD2_END_COLUMN.to_string(),
            SCD2_CURRENT_COLUMN.to_string(),
        ]);
    }
    let column_list = insert_columns
        .iter()
        .map(|c| ident(c))
        .collect::<Vec<_>>()
        .join(", ");
    let insert = insert_shape(
        plan_for(
            server,
            session,
            &format!(
                "INSERT INTO {target_sql} ({column_list}) SELECT {column_list} FROM {target_sql} \
                 WHERE FALSE"
            ),
        )
        .await?,
    )?;
    let assigned: Option<String> = match &history {
        Some(_) => Some(SCD2_CURRENT_COLUMN.to_string()),
        None => applied
            .iter()
            .find(|c| !stmt.keys.iter().any(|k| k.eq_ignore_ascii_case(c)))
            .cloned(),
    };
    let update = match assigned {
        Some(column) => update_shape(
            plan_for(
                server,
                session,
                &format!(
                    "UPDATE {target_sql} SET {column} = {column} WHERE FALSE",
                    column = ident(&column)
                ),
            )
            .await?,
        )?,
        None => UpdateShape {
            check_constraints: Vec::new(),
            generated_columns: Vec::new(),
        },
    };

    Ok(ApplyJob {
        table_id: target.id,
        winners,
        target: target_scan,
        insert,
        update,
        applied: applied_ids,
        keys,
        ignore_null_updates: stmt.ignore_null_updates,
        sequence_at: position(SEQUENCE_COLUMN)?,
        is_delete_at: position(DELETE_COLUMN)?,
        is_truncate_at: match stmt.truncate_when {
            Some(_) => Some(position(TRUNCATE_COLUMN)?),
            None => None,
        },
        rank_at: position(RANK_COLUMN)?,
        history,
    })
}

/// Runs one APPLY CHANGES.
///
/// The whole apply runs in one transaction, so the target settles whole or
/// not at all, and a change stream the source read advances in that same
/// commit
pub async fn handle_apply_changes(
    stmt: &ApplyChangesStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let counts = run_apply_changes(stmt, server, session).await?;
    Ok(DdlResult::Tag(format!(
        "APPLY CHANGES {}",
        counts.upserted + counts.deleted
    )))
}

/// Runs one APPLY CHANGES and records the run, answering with what it did.
///
/// The statement on its own and a pipeline stage carrying it both come
/// through here, so `zyron_sys.cdc.apply_runs` lists either the same way
pub async fn run_apply_changes(
    stmt: &ApplyChangesStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<ApplyCounts, ProtocolError> {
    let started = std::time::Instant::now();
    let started_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as i64;
    let (schema_id, target_name) =
        crate::ddl_dispatch::resolve_qualified_name(&stmt.target, server, session)?;
    let target = server
        .catalog
        .get_table(schema_id, &target_name)
        .map_err(ProtocolError::Database)?;

    let source_label = match &stmt.source {
        ApplySource::Named(name) => name.clone(),
        ApplySource::Relation(_) => "relation".to_string(),
    };

    let outcome = match resolve(stmt, server, session, &target).await {
        Ok(job) => crate::ddl_dispatch::run_apply_job(server, session, job).await,
        Err(e) => Err(e),
    };

    let counts = outcome.as_ref().copied().unwrap_or(ApplyCounts::default());
    let run = crate::change_stream_dispatch::ApplyRun {
        target: stmt.target.clone(),
        source: source_label,
        rows_upserted: counts.upserted,
        rows_deleted: counts.deleted,
        rows_versioned: counts.versioned,
        truncated: counts.truncated,
        duration_micros: started.elapsed().as_micros() as u64,
        started_at,
        error: outcome
            .as_ref()
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default(),
    };
    tracing::info!(
        target: "zyron::audit",
        event = "ApplyChangesRun",
        target = %run.target,
        source = %run.source,
        rows_upserted = run.rows_upserted,
        rows_deleted = run.rows_deleted,
        rows_versioned = run.rows_versioned,
        failed = !run.error.is_empty(),
        actor_role = crate::ddl_dispatch::actor_role_id(session),
    );
    crate::change_stream_dispatch::apply_runs().record(run);

    outcome
}

#[cfg(test)]
mod tests {
    use super::*;

    fn statement(sql: &str) -> ApplyChangesStatement {
        match zyron_parser::parse(sql)
            .expect("parses")
            .into_iter()
            .next()
            .expect("one statement")
        {
            zyron_parser::Statement::ApplyChanges(apply) => *apply,
            other => panic!("expected APPLY CHANGES, got {other:?}"),
        }
    }

    #[test]
    fn test_an_identifier_is_quoted_and_its_quotes_doubled() {
        assert_eq!(ident("id"), "\"id\"");
        assert_eq!(ident("we\"ird"), "\"we\"\"ird\"");
        assert_eq!(qualified("s.t"), "\"s\".\"t\"");
    }

    #[test]
    fn test_the_ranked_relation_numbers_by_the_sequence() {
        let stmt =
            statement("APPLY CHANGES INTO silver FROM s KEYS (id) SEQUENCE BY _commit_version");
        let columns = ["id".to_string(), "seq".to_string()];
        let sql = ranked_sql(&stmt, &columns).expect("writes");
        assert!(
            sql.contains("ROW_NUMBER() OVER (PARTITION BY \"id\""),
            "{sql}"
        );
        assert!(sql.contains("<> 'update_preimage'"), "{sql}");
        assert!(sql.contains("\"_change_ordinal\" DESC"), "{sql}");
    }

    #[test]
    fn test_a_source_with_no_sequence_orders_by_the_commit_version() {
        let stmt = statement("APPLY CHANGES INTO silver FROM s KEYS (id)");
        assert_eq!(sequence_sql(&stmt), "\"_commit_version\"");
    }

    #[test]
    fn test_a_delete_predicate_replaces_the_change_kind_test() {
        let plain = statement("APPLY CHANGES INTO silver FROM s KEYS (id)");
        assert!(delete_sql(&plain).contains("'delete'"));
        let soft =
            statement("APPLY CHANGES INTO silver FROM s KEYS (id) APPLY AS DELETE WHEN op = 'D'");
        assert!(delete_sql(&soft).contains("op"), "{}", delete_sql(&soft));
    }

    #[test]
    fn test_track_history_on_and_except_pick_the_same_columns() {
        let applied = vec!["id".to_string(), "tier".to_string(), "visits".to_string()];
        let on = statement(
            "APPLY CHANGES INTO silver FROM s KEYS (id) STORED AS SCD TYPE 2 \
             TRACK HISTORY ON (tier)",
        );
        let except = statement(
            "APPLY CHANGES INTO silver FROM s KEYS (id) STORED AS SCD TYPE 2 \
             TRACK HISTORY EXCEPT (visits)",
        );
        assert_eq!(
            tracked_columns(&on, &applied),
            tracked_columns(&except, &applied)
        );
        assert_eq!(tracked_columns(&on, &applied), vec!["tier".to_string()]);
    }
}

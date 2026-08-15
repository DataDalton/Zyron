//! DDL and utility statement dispatch.
//!
//! Intercepts DDL, DCL, and utility statements before they reach the
//! planner/executor pipeline. These statements operate on the catalog,
//! auth, CDC, pipeline, streaming, and versioning subsystems directly.

use std::sync::Arc;

use zyron_common::ZyronError;

use crate::connection::ServerState;
use crate::messages::ProtocolError;
use crate::session::Session;

/// Result from a DDL dispatch. Contains the command tag and optionally
/// result rows for DDL statements that return data.
#[derive(Debug)]
pub enum DdlResult {
    /// Command completed with a tag (e.g., "CREATE TABLE").
    Tag(String),
    /// Command completed with result rows (for DDL queries like SHOW-style results).
    Rows {
        tag: String,
        columns: Vec<(String, i32)>,
        rows: Vec<Vec<String>>,
    },
}

/// Attempts to handle a DDL or utility statement directly.
/// Returns `Some(Ok(result))` if the statement was handled,
/// `Some(Err(e))` if handling failed, or `None` if the statement should
/// fall through to the planner/executor path.
///
/// This is a plain function returning a boxed future, and each arm boxes its
/// own, rather than the whole thing being one `async fn`. That is load
/// bearing. As an `async fn` the match is a single state machine, and a debug
/// build gives every arm's awaited handler future its own slot in the poll
/// frame instead of sharing one, so a statement paid for all hundred and
/// thirteen arms whichever one it took. That was 58 KB of stack per call, on
/// a path every connection runs before it can do anything else. Boxing per
/// arm constructs only the handler actually selected, at the cost of one
/// allocation per statement, which is nothing next to the work a DDL
/// statement then does
pub fn try_handle_ddl_utility<'a>(
    stmt: &'a zyron_parser::Statement,
    server: &'a Arc<ServerState>,
    session: &'a mut Option<Session>,
    txn: &'a mut Option<zyron_storage::txn::Transaction>,
    active_branch: &'a mut Option<String>,
    raw_sql: &'a str,
) -> std::pin::Pin<
    Box<dyn std::future::Future<Output = Option<Result<DdlResult, ProtocolError>>> + Send + 'a>,
> {
    use zyron_parser::Statement;

    match stmt {
        // DML statements fall through to planner
        Statement::Select(_)
        | Statement::Insert(_)
        | Statement::Update(_)
        | Statement::Delete(_) => Box::pin(async move { None }),

        // MERGE requires INSERT/UPDATE/DELETE combined execution, routed through planner
        Statement::Merge(s) => {
            Box::pin(async move { Some(handle_merge(s, server, session).await) })
        }

        // Standalone VALUES query handled as DDL result with rows
        Statement::ValuesQuery(s) => {
            Box::pin(async move { Some(handle_values_query(s, server, session).await) })
        }

        // Transaction control handled by try_handle_transaction_control
        Statement::Begin(_) | Statement::Commit(_) | Statement::Rollback(_) => {
            Box::pin(async move { None })
        }

        // SHOW CLUSTERING FOR t reads table state rather than session
        // state, so it is the one SHOW form that belongs here
        Statement::Show(s) if s.target.is_some() => {
            Box::pin(async move { Some(handle_show_clustering(s, server, session).await) })
        }

        // -- Node mesh --
        Statement::AlterTableFollow(s) => {
            Box::pin(async move { Some(handle_alter_table_follow(s, server, session).await) })
        }
        Statement::CreatePeer(s) => {
            Box::pin(async move { Some(handle_create_peer(s, server).await) })
        }
        Statement::DropPeer(s) => Box::pin(async move { Some(handle_drop_peer(s, server).await) }),
        Statement::CreateForeignTable(s) => {
            Box::pin(async move { Some(handle_create_foreign_table(s, server, session).await) })
        }
        Statement::DropForeignTable(s) => {
            Box::pin(async move { Some(handle_drop_foreign_table(s, server, session).await) })
        }

        // Session commands handled by try_handle_session_command
        Statement::SetVariable(_)
        | Statement::Show(_)
        | Statement::AlterSystemSet(_)
        | Statement::Checkpoint(_)
        | Statement::Vacuum(_)
        | Statement::Analyze(_) => Box::pin(async move { None }),

        // EXPLAIN handled by handle_explain_statement
        Statement::Explain(_) => Box::pin(async move { None }),

        // -- Core DDL --
        Statement::CreateTable(s) => {
            Box::pin(async move { Some(handle_create_table(s, server, session).await) })
        }
        Statement::DropTable(s) => {
            Box::pin(async move { Some(handle_drop_table(s, server, session).await) })
        }
        Statement::AlterTable(s) => {
            Box::pin(async move { Some(handle_alter_table(s, server, session).await) })
        }
        Statement::Truncate(s) => {
            Box::pin(async move { Some(handle_truncate(s, server, session).await) })
        }
        Statement::CreateIndex(s) => {
            Box::pin(async move { Some(handle_create_index(s, server, session).await) })
        }
        Statement::DropIndex(s) => {
            Box::pin(async move { Some(handle_drop_index(s, server, session).await) })
        }
        Statement::AlterIndex(s) => {
            Box::pin(async move { Some(handle_alter_index(s, server, session).await) })
        }
        Statement::CreateSchema(s) => {
            Box::pin(async move { Some(handle_create_schema(s, server, session).await) })
        }
        Statement::DropSchema(s) => {
            Box::pin(async move { Some(handle_drop_schema(s, server, session).await) })
        }
        Statement::CreateSequence(s) => {
            Box::pin(async move { Some(handle_create_sequence(s, server, session).await) })
        }
        Statement::DropSequence(s) => {
            Box::pin(async move { Some(handle_drop_sequence(s, server, session).await) })
        }
        Statement::AlterSequence(s) => {
            Box::pin(async move { Some(handle_alter_sequence(s, server, session).await) })
        }
        Statement::CreateView(s) => {
            Box::pin(async move { Some(handle_create_view(s, server, session, raw_sql).await) })
        }
        Statement::DropView(s) => {
            Box::pin(async move { Some(handle_drop_view(s, server, session).await) })
        }
        Statement::AlterView(s) => {
            Box::pin(async move { Some(handle_alter_view(s, server, session).await) })
        }
        Statement::AlterTableTtl(s) => Box::pin(async move {
            Some(crate::lifecycle_dispatch::handle_alter_table_ttl(s, server, session).await)
        }),
        Statement::AlterTableOptions(s) => Box::pin(async move {
            Some(crate::lifecycle_dispatch::handle_alter_table_options(s, server, session).await)
        }),
        Statement::AlterTableSetUsing(s) => {
            Box::pin(async move { Some(handle_set_using(s, server, session).await) })
        }
        Statement::AlterTableClusterBy(s) => {
            Box::pin(async move { Some(handle_cluster_by(s, server, session).await) })
        }
        Statement::AlterTableClusteringSchedule(s) => {
            Box::pin(async move { Some(handle_clustering_schedule(s, server, session).await) })
        }
        Statement::LegalHold(s) => Box::pin(async move {
            Some(crate::lifecycle_dispatch::handle_legal_hold(s, server, session).await)
        }),
        Statement::ForgetUser(s) => Box::pin(async move {
            Some(crate::lifecycle_dispatch::handle_forget_user(s, server, session).await)
        }),
        Statement::ExportUser(s) => Box::pin(async move {
            Some(crate::lifecycle_dispatch::handle_export_user(s, server, session).await)
        }),
        Statement::AlterTableMove(s) => Box::pin(async move {
            Some(crate::lifecycle_dispatch::handle_alter_table_move(s, server, session).await)
        }),
        Statement::AlterColumnClassification(s) => Box::pin(async move {
            Some(
                crate::lifecycle_dispatch::handle_alter_column_classification(s, server, session)
                    .await,
            )
        }),
        Statement::RestoreSoftDelete(s) => Box::pin(async move {
            Some(crate::lifecycle_dispatch::handle_restore_soft_delete(s, server, session).await)
        }),
        Statement::RunRetentionJob(s) => Box::pin(async move {
            Some(crate::lifecycle_dispatch::handle_run_retention_job(s, server, session).await)
        }),
        Statement::UndropTable(s) => Box::pin(async move {
            Some(crate::lifecycle_dispatch::handle_undrop_table(s, server, session).await)
        }),
        // OPTIMIZE TABLE and REINDEX are handled by the session command layer
        // (connection.rs handle_optimize / handle_reindex) before this dispatch
        // runs, so these never reach here; route to None for consistency.
        Statement::OptimizeTable(_) | Statement::Reindex(_) => Box::pin(async move { None }),
        Statement::CommentOn(s) => {
            Box::pin(async move { Some(handle_comment_on(s, server, session).await) })
        }

        // -- Materialized Views --
        Statement::CreateMaterializedView(s) => Box::pin(async move {
            Some(handle_create_materialized_view(s, server, session, raw_sql).await)
        }),
        Statement::DropMaterializedView(s) => {
            Box::pin(async move { Some(handle_drop_materialized_view(s, server, session).await) })
        }
        Statement::RefreshMaterializedView(s) => {
            Box::pin(
                async move { Some(handle_refresh_materialized_view(s, server, session).await) },
            )
        }

        // -- Search Indexes --
        Statement::CreateFulltextIndex(s) => {
            Box::pin(async move { Some(handle_create_fulltext_index(s, server, session).await) })
        }
        Statement::CreateVectorIndex(s) => {
            Box::pin(async move { Some(handle_create_vector_index(s, server, session).await) })
        }
        Statement::CreateSpatialIndex(s) => {
            Box::pin(async move { Some(handle_create_spatial_index(s, server, session).await) })
        }

        // -- Auth/Roles --
        Statement::CreateUser(s) => {
            Box::pin(async move { Some(handle_create_user(s, server, session).await) })
        }
        Statement::AlterUser(s) => {
            Box::pin(async move { Some(handle_alter_user(s, server, session).await) })
        }
        Statement::DropUser(s) => {
            Box::pin(async move { Some(handle_drop_user(s, server, session).await) })
        }
        Statement::CreateRole(s) => {
            Box::pin(async move { Some(handle_create_role(s, server).await) })
        }
        Statement::AlterRole(s) => {
            Box::pin(async move { Some(handle_alter_role(s, server, session).await) })
        }
        Statement::DropRole(s) => Box::pin(async move { Some(handle_drop_role(s, server).await) }),
        Statement::Grant(s) => {
            Box::pin(async move { Some(handle_grant(s, server, session).await) })
        }
        Statement::Revoke(s) => {
            Box::pin(async move { Some(handle_revoke(s, server, session).await) })
        }

        // -- CDC --
        Statement::CreateReplicationSlot(s) => {
            Box::pin(async move { Some(handle_create_replication_slot(s, server, session).await) })
        }
        Statement::DropReplicationSlot(s) => {
            Box::pin(async move { Some(handle_drop_replication_slot(s, server, session).await) })
        }
        Statement::CreateCdcStream(s) => {
            Box::pin(async move { Some(handle_create_cdc_stream(s, server, session).await) })
        }
        Statement::DropCdcStream(s) => {
            Box::pin(async move { Some(handle_drop_cdc_stream(s, server, session).await) })
        }
        Statement::CreateCdcIngest(s) => {
            Box::pin(async move { Some(handle_create_cdc_ingest(s, server, session).await) })
        }
        Statement::DropCdcIngest(s) => {
            Box::pin(async move { Some(handle_drop_cdc_ingest(s, server, session).await) })
        }

        // -- Streaming jobs --
        Statement::CreateStreamingJob(_)
        | Statement::DropStreamingJob(_)
        | Statement::AlterStreamingJob(_) => Box::pin(async move {
            Some(dispatch_streaming_statement(stmt.clone(), server, session, raw_sql).await)
        }),

        // -- Versioning --
        Statement::CreateBranch(s) => {
            Box::pin(async move { Some(handle_create_branch(s, server, session).await) })
        }
        Statement::MergeBranch(s) => {
            Box::pin(async move { Some(handle_merge_branch(s, server, session).await) })
        }
        Statement::DropBranch(s) => {
            Box::pin(async move { Some(handle_drop_branch(s, server, session).await) })
        }
        Statement::UseBranch(s) => {
            Box::pin(async move { Some(handle_use_branch(s, server, active_branch).await) })
        }
        Statement::CreateVersion(s) => {
            Box::pin(async move { Some(handle_create_version(s, server, session).await) })
        }
        Statement::DropVersion(s) => {
            Box::pin(async move { Some(handle_drop_version(s, server, session).await) })
        }

        // -- Pipeline --
        Statement::CreatePipeline(s) => {
            Box::pin(async move { Some(handle_create_pipeline(s, server, session, raw_sql).await) })
        }
        Statement::RunPipeline(s) => {
            Box::pin(async move { Some(handle_run_pipeline(s, server, session).await) })
        }
        Statement::DropPipeline(s) => {
            Box::pin(async move { Some(handle_drop_pipeline(s, server, session).await) })
        }

        // -- Feature store and ML --
        Statement::CreateFeatureGroup(s) => {
            Box::pin(async move { Some(handle_create_feature_group(s, server).await) })
        }
        Statement::DropFeatureGroup(s) => {
            Box::pin(async move { Some(handle_drop_feature_group(s, server).await) })
        }
        Statement::CreateModel(s) => {
            Box::pin(
                async move { Some(handle_create_model(s, server, session, txn, raw_sql).await) },
            )
        }
        Statement::DropModel(s) => {
            Box::pin(async move { Some(handle_drop_model(s, server).await) })
        }

        // -- Scheduling --
        Statement::CreateSchedule(s) => {
            Box::pin(async move { Some(handle_create_schedule(s, server, session, raw_sql).await) })
        }
        Statement::DropSchedule(s) => {
            Box::pin(async move { Some(handle_drop_schedule(s, server, session).await) })
        }
        Statement::PauseSchedule(s) => {
            Box::pin(async move { Some(handle_pause_schedule(s, server, session).await) })
        }
        Statement::ResumeSchedule(s) => {
            Box::pin(async move { Some(handle_resume_schedule(s, server, session).await) })
        }

        // -- Functions/Aggregates --
        Statement::CreateFunction(s) => {
            Box::pin(async move { Some(handle_create_function(s, server, session).await) })
        }
        Statement::DropFunction(s) => {
            Box::pin(async move { Some(handle_drop_function(s, server, session).await) })
        }
        Statement::CreateAggregate(s) => {
            Box::pin(async move { Some(handle_create_aggregate(s, server, session).await) })
        }
        Statement::DropAggregate(s) => {
            Box::pin(async move { Some(handle_drop_aggregate(s, server, session).await) })
        }

        // -- Procedures --
        Statement::CreateProcedure(s) => {
            Box::pin(async move { Some(handle_create_procedure(s, server, session).await) })
        }
        Statement::DropProcedure(s) => {
            Box::pin(async move { Some(handle_drop_procedure(s, server, session).await) })
        }
        Statement::Call(s) => Box::pin(async move { Some(handle_call(s, server, session).await) }),

        // -- Triggers --
        Statement::CreateTrigger(s) => {
            Box::pin(async move { Some(handle_create_trigger(s, server, session).await) })
        }
        Statement::DropTrigger(s) => {
            Box::pin(async move { Some(handle_drop_trigger(s, server, session).await) })
        }

        // -- Event Handlers --
        Statement::CreateEventHandler(s) => {
            Box::pin(async move { Some(handle_create_event_handler(s, server, session).await) })
        }
        Statement::DropEventHandler(s) => {
            Box::pin(async move { Some(handle_drop_event_handler(s, server, session).await) })
        }

        // -- Expectations/Features --
        Statement::AddExpectation(s) => {
            Box::pin(async move { Some(handle_add_expectation(s, server, session).await) })
        }
        Statement::DropExpectation(s) => {
            Box::pin(async move { Some(handle_drop_expectation(s, server, session).await) })
        }
        Statement::EnableFeature(s) => {
            Box::pin(async move { Some(handle_enable_feature(s, server, session).await) })
        }
        Statement::DisableFeature(s) => {
            Box::pin(async move { Some(handle_disable_feature(s, server, session).await) })
        }

        // -- Transaction extensions --
        Statement::Savepoint(s) => Box::pin(async move { Some(handle_savepoint(s, txn, server)) }),
        Statement::ReleaseSavepoint(s) => {
            Box::pin(async move { Some(handle_release_savepoint(s, txn)) })
        }

        // -- Prepared statements: handled by caller (needs statements map) --
        Statement::Prepare(_) | Statement::Execute(_) | Statement::Deallocate(_) => {
            Box::pin(async move { None })
        }

        // -- Cursors: handled by caller (needs cursors map) --
        Statement::DeclareCursor(_) | Statement::FetchCursor(_) | Statement::CloseCursor(_) => {
            Box::pin(async move { None })
        }

        // -- Pub/Sub: handled by caller (needs notification_receivers) --
        Statement::Listen(_) | Statement::Notify(_) => Box::pin(async move { None }),

        // -- COPY: handled by caller (needs wire protocol interaction) --
        Statement::Copy(_) => Box::pin(async move { None }),

        // -- Archive --
        Statement::ArchiveTable(s) => {
            Box::pin(async move { Some(handle_archive_table(s, server, session).await) })
        }
        Statement::RestoreTable(s) => {
            Box::pin(async move { Some(handle_restore_table(s, server, session).await) })
        }

        // -- Utility --
        Statement::DoBlock(s) => {
            Box::pin(async move { Some(handle_do_block(s, server, session).await) })
        }

        // -- Graph schema --
        Statement::CreateGraphSchema(stmt) => {
            Box::pin(async move { Some(handle_create_graph_schema(stmt, server, session).await) })
        }
        Statement::DropGraphSchema(stmt) => {
            Box::pin(async move { Some(handle_drop_graph_schema(stmt, server, session).await) })
        }

        // -- External sources and sinks --
        Statement::CreateExternalSource(_)
        | Statement::CreateExternalSink(_)
        | Statement::DropExternalSource(_)
        | Statement::DropExternalSink(_)
        | Statement::AlterExternalSource(_)
        | Statement::AlterExternalSink(_) => Box::pin(async move {
            Some(dispatch_external_statement(stmt.clone(), server, session).await)
        }),

        // -- Zyron-to-Zyron data plane --
        Statement::CreatePublication(_)
        | Statement::AlterPublication(_)
        | Statement::CreateEndpoint(_)
        | Statement::CreateStreamingEndpoint(_)
        | Statement::AlterEndpoint(_)
        | Statement::AlterSecurityMap(_)
        | Statement::DropSecurityMap(_) => {
            Box::pin(
                async move { Some(dispatch_z2z_statement(stmt.clone(), server, session).await) },
            )
        }
        Statement::TagPublication(s) => {
            Box::pin(async move { Some(handle_tag_publication(s, server, session).await) })
        }
        Statement::UntagPublication(s) => {
            Box::pin(async move { Some(handle_untag_publication(s, server, session).await) })
        }
        Statement::DropPublication(s) => {
            Box::pin(async move { Some(handle_drop_publication(s, server, session).await) })
        }
        Statement::DropEndpoint(s) => {
            Box::pin(async move { Some(handle_drop_endpoint(s, server, session).await) })
        }
        Statement::CreateAbacPolicy(s) => {
            Box::pin(async move { Some(handle_create_abac_policy(s, server, session).await) })
        }
    }
}

/// Creates an ABAC policy on a table or publication. Resolves the target to its
/// catalog id, checks ManagePolicy privilege, then stores the policy (the
/// verbatim predicate text is the row filter). Table policies are enforced at
/// query bind time via the row-security provider; publication policies are
/// enforced on the subscriber change stream.
async fn handle_create_abac_policy(
    stmt: &zyron_parser::ast::CreateAbacPolicyStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    use zyron_parser::ast::AbacPolicyTarget;

    let (_db_id, schema_id) = get_session_schema(session, server, None)?;

    let (object_id, target, object_type) = match stmt.target {
        AbacPolicyTarget::Table => {
            let table = server
                .catalog
                .get_table(schema_id, &stmt.target_name)
                .map_err(ProtocolError::Database)?;
            (
                table.id.0,
                zyron_auth::AbacTarget::Table,
                zyron_auth::ObjectType::Table,
            )
        }
        AbacPolicyTarget::Publication => {
            let publication = server
                .catalog
                .get_publication(schema_id, &stmt.target_name)
                .ok_or_else(|| {
                    ProtocolError::Database(ZyronError::Internal(format!(
                        "publication '{}' does not exist",
                        stmt.target_name
                    )))
                })?;
            (
                publication.id.0,
                zyron_auth::AbacTarget::Publication,
                zyron_auth::ObjectType::Publication,
            )
        }
    };

    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::ManagePolicy,
        object_type,
        object_id,
    )?;

    // A policy can only be stored and enforced when a security manager exists.
    // Fail closed rather than report success on a no-op, so the statement never
    // implies an access control that is not actually applied.
    let sm = server.security_manager.as_ref().ok_or_else(|| {
        ProtocolError::Database(ZyronError::Internal(
            "security is not enabled; cannot create ABAC policy".to_string(),
        ))
    })?;

    let policy = zyron_auth::AbacPolicy {
        id: 0,
        name: stmt.name.clone(),
        table_id: object_id,
        target,
        predicate: stmt.predicate_sql.clone(),
        enabled: true,
        permissive: true,
        roles: Vec::new(),
    };

    sm.create_abac_policy(policy)
        .await
        .map_err(ProtocolError::Database)?;

    Ok(DdlResult::Tag("CREATE ABAC POLICY".to_string()))
}

// ---------------------------------------------------------------------------
// DDL privilege checking
// ---------------------------------------------------------------------------

/// Checks whether the session has the required privilege for a DDL operation.
/// If no security manager is configured, the check is skipped (open access).
/// The object_id is the catalog ID of the target object (schema ID, table ID, etc.).
pub(crate) fn check_ddl_privilege(
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
    privilege: zyron_auth::PrivilegeType,
    object_type: zyron_auth::ObjectType,
    object_id: u32,
) -> Result<(), ProtocolError> {
    let sm = match server.security_manager.as_ref() {
        Some(sm) => sm,
        None => return Ok(()),
    };

    let session = session
        .as_mut()
        .ok_or(ProtocolError::Malformed("no active session".into()))?;

    let ctx = match session.security_context.as_mut() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    if ctx.has_privilege(
        &sm.privilege_store,
        privilege,
        object_type,
        object_id,
        None,
        now,
    ) {
        Ok(())
    } else {
        Err(ProtocolError::Database(ZyronError::PermissionDenied(
            format!(
                "permission denied: {:?} on {:?} {}",
                privilege, object_type, object_id
            ),
        )))
    }
}

// ---------------------------------------------------------------------------
// Core DDL handlers
// ---------------------------------------------------------------------------

/// Applies an ALTER TABLE operation. Metadata-only operations (RENAME TABLE,
/// RENAME COLUMN, SET/DROP DEFAULT, DROP NOT NULL, ADD/DROP CONSTRAINT) mutate
/// the catalog entry and re-persist it. Operations that change the physical
/// tuple layout (ADD/DROP COLUMN, ALTER TYPE) or require scanning existing data
/// (SET NOT NULL) return an explicit error until the table-rewrite path lands;
/// they never report success without doing the work.
async fn handle_alter_table(
    stmt: &zyron_parser::ast::AlterTableStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    use zyron_catalog::schema::ConstraintType;
    use zyron_parser::ast::AlterTableOperation as Op;

    let (_, schema_id) = get_session_schema(session, server, None)?;

    let table = server
        .catalog
        .get_table(schema_id, &stmt.name)
        .map_err(ProtocolError::Database)?;

    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Table,
        table.id.0,
    )?;

    let mut entry = (*table).clone();
    // Index work follows the catalog update, so a failure to write the
    // constraint leaves no tree behind it
    let mut provision_indexes_after = false;
    let mut drop_index_after: Option<String> = None;

    match &stmt.operation {
        Op::RenameTable { new_name } => {
            if server.catalog.get_table(schema_id, new_name).is_ok() {
                return Err(ProtocolError::Database(ZyronError::TableAlreadyExists(
                    new_name.clone(),
                )));
            }
            entry.name = new_name.clone();
        }
        Op::RenameColumn { old_name, new_name } => {
            if entry.columns.iter().any(|c| c.name == *new_name) {
                return Err(ProtocolError::Database(ZyronError::Internal(format!(
                    "column \"{new_name}\" already exists"
                ))));
            }
            let col = entry
                .columns
                .iter_mut()
                .find(|c| c.name == *old_name)
                .ok_or_else(|| {
                    ProtocolError::Database(ZyronError::ColumnNotFound(old_name.clone()))
                })?;
            col.name = new_name.clone();
        }
        Op::AlterColumnSetDefault { column, default } => {
            let col = alter_find_column_mut(&mut entry, column)?;
            col.default_expr = Some(zyron_parser::expr_to_sql(default));
        }
        Op::AlterColumnDropDefault { column } => {
            let col = alter_find_column_mut(&mut entry, column)?;
            col.default_expr = None;
        }
        Op::AlterColumnDropNotNull { column } => {
            let col_id = {
                let col = alter_find_column_mut(&mut entry, column)?;
                col.nullable = true;
                col.id
            };
            entry.constraints.retain(|c| {
                !(c.constraint_type == ConstraintType::NotNull && c.columns == [col_id])
            });
        }
        Op::AddConstraint(tc) => {
            let ce = build_constraint_entry(&stmt.name, tc, &entry.columns, server, schema_id)?;
            if entry.constraints.iter().any(|c| c.name == ce.name) {
                return Err(ProtocolError::Database(ZyronError::Internal(format!(
                    "constraint \"{}\" already exists",
                    ce.name
                ))));
            }
            // Reject the constraint if any current row already violates it.
            validate_constraint_against_existing(&stmt.name, tc, server, schema_id).await?;
            entry.constraints.push(ce);
            // Validating the existing rows settles the past. Ongoing
            // enforcement needs the index, which is provisioned once the
            // catalog carries the constraint
            provision_indexes_after = true;
        }
        Op::DropConstraint { name, if_exists } => {
            let before = entry.constraints.len();
            entry.constraints.retain(|c| c.name != *name);
            if entry.constraints.len() == before && !*if_exists {
                return Err(ProtocolError::Database(ZyronError::Internal(format!(
                    "constraint \"{name}\" does not exist"
                ))));
            }
            drop_index_after = Some(name.clone());
        }
        Op::AlterColumnSetNotNull { column } => {
            // SET NOT NULL does not change the tuple layout, so no heap rewrite
            // is needed: validate that no existing row holds a NULL in the
            // column, then flip the catalog flag. Rejecting with the live count
            // is correct (and far cheaper than a rewrite).
            if !entry.columns.iter().any(|c| c.name == *column) {
                return Err(ProtocolError::Database(ZyronError::ColumnNotFound(
                    column.clone(),
                )));
            }
            let null_count = count_query(
                server,
                &format!(
                    "SELECT \"{}\" FROM \"{}\" WHERE \"{}\" IS NULL",
                    column, stmt.name, column
                ),
            )
            .await?;
            if null_count > 0 {
                return Err(ProtocolError::Database(ZyronError::Internal(format!(
                    "column \"{column}\" contains {null_count} NULL value(s); cannot SET NOT NULL"
                ))));
            }
            let col = alter_find_column_mut(&mut entry, column)?;
            col.nullable = false;
        }
        Op::AddColumn(_) | Op::DropColumn { .. } | Op::AlterColumnSetType { .. } => {
            // Column-shape changes rewrite the heap: the tuple decoder walks the
            // full column list and drops any tuple narrower than the schema, so
            // existing rows must be re-encoded under the new layout. The rewrite
            // builds a fresh heap in side files and swaps the catalog, leaving
            // the old heap intact until it commits.
            return rewrite_table_columns(&stmt.operation, &stmt.name, server, schema_id, &table)
                .await;
        }
    }

    let table_id = entry.id;
    server
        .catalog
        .update_table(entry)
        .await
        .map_err(ProtocolError::Database)?;

    if provision_indexes_after {
        provision_constraint_indexes(server, schema_id, &stmt.name).await?;
    }
    if let Some(name) = drop_index_after {
        drop_constraint_index(server, table_id, &name).await;
    }

    Ok(DdlResult::Tag("ALTER TABLE".to_string()))
}

/// Ensures a table's companion quarantine table exists and returns its id.
/// The quarantine table mirrors the source columns and appends `_expectation`
/// (the violated expectation name) and `_quarantined_at` (epoch microseconds).
/// It carries no indexes, constraints, or expectations of its own.
async fn ensure_quarantine_table(
    server: &Arc<ServerState>,
    schema_id: zyron_catalog::SchemaId,
    table: &zyron_catalog::schema::TableEntry,
) -> Result<u32, ProtocolError> {
    let q_name = format!("{}_quarantine", table.name);
    if let Ok(existing) = server.catalog.get_table(schema_id, &q_name) {
        return Ok(existing.id.0);
    }
    let mut cols: Vec<(String, zyron_common::TypeId, bool, Option<u8>)> = table
        .columns
        .iter()
        .map(|c| (c.name.clone(), c.type_id, true, c.fractional_digits))
        .collect();
    cols.push((
        "_expectation".to_string(),
        zyron_common::TypeId::Varchar,
        true,
        None,
    ));
    cols.push((
        "_quarantined_at".to_string(),
        zyron_common::TypeId::Int64,
        true,
        None,
    ));
    let qid = server
        .catalog
        .create_table_from_columns(schema_id, &q_name, &cols)
        .await
        .map_err(ProtocolError::Database)?;
    Ok(qid.0)
}

/// ALTER TABLE name ADD EXPECTATION name EXPECT expr ON VIOLATION action.
/// Persists the predicate (as SQL text) and its action on the table entry.
/// A Quarantine expectation also creates the companion quarantine table.
async fn handle_add_expectation(
    stmt: &zyron_parser::ast::AddExpectationStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    use zyron_catalog::schema::{ExpectationAction, ExpectationEntry};
    use zyron_parser::ast::ViolationAction;

    let (_, schema_id) = get_session_schema(session, server, None)?;
    let table = server
        .catalog
        .get_table(schema_id, &stmt.table)
        .map_err(ProtocolError::Database)?;
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Table,
        table.id.0,
    )?;

    // A data-quality expectation is evaluated per row on the insert path, so its
    // predicate must reference only the inserted row's columns. Reject a subquery
    // at definition time rather than letting it persist and fail every insert.
    if zyron_parser::ast::expr_contains_subquery(&stmt.expr) {
        return Err(ProtocolError::Database(ZyronError::InvalidParameter {
            name: "expectation predicate".to_string(),
            value: format!(
                "expectation \"{}\" may not contain a subquery; expectations are row-local checks evaluated per row",
                stmt.name
            ),
        }));
    }

    let mut entry = (*table).clone();
    if entry.expectations.iter().any(|e| e.name == stmt.name) {
        return Err(ProtocolError::Database(ZyronError::Internal(format!(
            "expectation \"{}\" already exists",
            stmt.name
        ))));
    }

    let action = match stmt.on_violation {
        ViolationAction::Fail => ExpectationAction::Fail,
        ViolationAction::Warn => ExpectationAction::Warn,
        ViolationAction::Drop => ExpectationAction::Drop,
        ViolationAction::Quarantine => ExpectationAction::Quarantine,
    };
    let predicate_sql = zyron_parser::expr_to_sql(&stmt.expr);

    let quarantine_table_id = if action == ExpectationAction::Quarantine {
        Some(ensure_quarantine_table(server, schema_id, &entry).await?)
    } else {
        None
    };

    entry.expectations.push(ExpectationEntry {
        name: stmt.name.clone(),
        predicate_sql,
        on_violation: action,
        quarantine_table_id,
    });

    server
        .catalog
        .update_table(entry)
        .await
        .map_err(ProtocolError::Database)?;

    Ok(DdlResult::Tag("ALTER TABLE".to_string()))
}

/// ALTER TABLE name DROP EXPECTATION name. Removes the expectation from the
/// table entry. The companion quarantine table, if any, is left intact.
async fn handle_drop_expectation(
    stmt: &zyron_parser::ast::DropExpectationStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (_, schema_id) = get_session_schema(session, server, None)?;
    let table = server
        .catalog
        .get_table(schema_id, &stmt.table)
        .map_err(ProtocolError::Database)?;
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Table,
        table.id.0,
    )?;

    let mut entry = (*table).clone();
    let before = entry.expectations.len();
    entry.expectations.retain(|e| e.name != stmt.name);
    if entry.expectations.len() == before {
        return Err(ProtocolError::Database(ZyronError::Internal(format!(
            "expectation \"{}\" does not exist",
            stmt.name
        ))));
    }

    server
        .catalog
        .update_table(entry)
        .await
        .map_err(ProtocolError::Database)?;

    Ok(DdlResult::Tag("ALTER TABLE".to_string()))
}

// ---------------------------------------------------------------------------
// ALTER TABLE column-rewrite engine (ADD/DROP COLUMN, ALTER TYPE)
// ---------------------------------------------------------------------------

/// Definition captured for an index that survives a column rewrite, so it can
/// be recreated empty and repopulated from the new heap. Tuple ids change in a
/// rewrite, so every index is rebuilt rather than patched.
struct IndexRebuild {
    name: String,
    col_names: Vec<String>,
    index_type: zyron_catalog::IndexType,
    unique: bool,
    parameters: Option<Vec<u8>>,
    // (dimensions, HNSW config) read live from the vector manager before drop.
    vector_config: Option<(u16, zyron_search::vector::HnswConfig)>,
}

/// A source operator that replays in-memory batches into the InsertOperator
/// during a table rewrite. The reshaped rows carry the new schema layout.
struct BatchSourceOperator {
    batches: std::vec::IntoIter<zyron_executor::batch::DataBatch>,
}

impl BatchSourceOperator {
    fn new(batches: Vec<zyron_executor::batch::DataBatch>) -> Self {
        Self {
            batches: batches.into_iter(),
        }
    }
}

impl zyron_executor::operator::Operator for BatchSourceOperator {
    fn next(&mut self) -> zyron_executor::operator::OperatorResult<'_> {
        Box::pin(async move {
            Ok(self
                .batches
                .next()
                .map(zyron_executor::operator::ExecutionBatch::new))
        })
    }
}

/// Rewrites a table's heap to apply ADD COLUMN, DROP COLUMN, or ALTER COLUMN
/// TYPE. Reads all rows under the old schema, reshapes them, builds a new heap
/// in freshly allocated files, rebuilds every index, and swaps the catalog.
/// The old heap files are retained until the new heap is populated so a failure
/// mid-rewrite leaves the original data recoverable.
async fn rewrite_table_columns(
    op: &zyron_parser::ast::AlterTableOperation,
    table_name: &str,
    server: &Arc<ServerState>,
    schema_id: zyron_catalog::ids::SchemaId,
    old_table: &zyron_catalog::schema::TableEntry,
) -> Result<DdlResult, ProtocolError> {
    use zyron_catalog::ids::ColumnId;
    use zyron_catalog::schema::{ColumnEntry, ConstraintType};
    use zyron_executor::column::ScalarValue;
    use zyron_parser::ast::AlterTableOperation as Op;

    let table_id = old_table.id;

    // Read every current row under the old schema. INCLUDING DELETED keeps
    // soft-deleted tombstones so the rewrite does not silently drop them.
    // SELECT * returns columns in ordinal order, so each batch column lines up
    // with old_table.columns.
    let mut new_batches = select_query_batches(
        server,
        &format!("SELECT * FROM \"{table_name}\" INCLUDING DELETED"),
    )
    .await?;

    // Build the new column list and reshape the batches together so the batch
    // column order always matches the catalog column order the encoder uses.
    let mut new_columns = old_table.columns.clone();
    let mut dropped_col_id: Option<ColumnId> = None;

    match op {
        Op::AddColumn(def) => {
            if new_columns.iter().any(|c| c.name == def.name) {
                return Err(ProtocolError::Database(ZyronError::Internal(format!(
                    "column \"{}\" already exists",
                    def.name
                ))));
            }
            let nullable = def.nullable.unwrap_or(true);
            let type_id = def.data_type.to_type_id();
            let fractional_digits = def.data_type.timestamp_precision();

            // Resolve the backfill value once. A volatile default (now()) is
            // evaluated a single time for the rewrite, matching the semantics
            // of a one-shot column add.
            let fill = match &def.default {
                Some(expr) => eval_default_scalar(server, expr, type_id).await?,
                None => {
                    if !nullable {
                        return Err(ProtocolError::Database(ZyronError::Internal(format!(
                            "column \"{}\" is NOT NULL but has no DEFAULT; cannot backfill existing rows",
                            def.name
                        ))));
                    }
                    ScalarValue::Null
                }
            };

            for b in new_batches.iter_mut() {
                let col = build_constant_column(type_id, fractional_digits, &fill, b.num_rows);
                b.columns.push(col);
            }

            let next_id = new_columns
                .iter()
                .map(|c| c.id.0)
                .max()
                .map(|m| m + 1)
                .unwrap_or(0);
            new_columns.push(ColumnEntry {
                id: ColumnId(next_id),
                table_id,
                name: def.name.clone(),
                type_id,
                ordinal: new_columns.len() as u16,
                nullable,
                default_expr: def.default.as_ref().map(zyron_parser::expr_to_sql),
                max_length: alter_extract_max_length(&def.data_type),
                fractional_digits,
                tz_offset_secs: None,
                element_type: None,
            });
        }
        Op::DropColumn { name, if_exists } => {
            let Some(pos) = new_columns.iter().position(|c| c.name == *name) else {
                if *if_exists {
                    return Ok(DdlResult::Tag("ALTER TABLE".to_string()));
                }
                return Err(ProtocolError::Database(ZyronError::ColumnNotFound(
                    name.clone(),
                )));
            };
            if new_columns.len() == 1 {
                return Err(ProtocolError::Database(ZyronError::Internal(
                    "cannot drop the only column of a table".to_string(),
                )));
            }
            let col_id = new_columns[pos].id;

            // A column in a primary key or a foreign key cannot be dropped.
            for c in &old_table.constraints {
                if (c.constraint_type == ConstraintType::PrimaryKey
                    || c.constraint_type == ConstraintType::ForeignKey)
                    && c.columns.contains(&col_id)
                {
                    return Err(ProtocolError::Database(ZyronError::Internal(format!(
                        "cannot drop column \"{name}\": it participates in a {:?} constraint",
                        c.constraint_type
                    ))));
                }
            }
            // A column referenced by another table's foreign key cannot be dropped.
            for t in server.catalog.list_tables(schema_id) {
                if t.id == table_id {
                    continue;
                }
                for c in &t.constraints {
                    if c.ref_table_id == Some(table_id) && c.ref_columns.contains(&col_id) {
                        return Err(ProtocolError::Database(ZyronError::Internal(format!(
                            "cannot drop column \"{name}\": referenced by a foreign key on table \"{}\"",
                            t.name
                        ))));
                    }
                }
            }

            for b in new_batches.iter_mut() {
                b.columns.remove(pos);
            }
            new_columns.remove(pos);
            dropped_col_id = Some(col_id);
        }
        Op::AlterColumnSetType { column, data_type } => {
            let Some(pos) = new_columns.iter().position(|c| c.name == *column) else {
                return Err(ProtocolError::Database(ZyronError::ColumnNotFound(
                    column.clone(),
                )));
            };
            let target = data_type.to_type_id();
            for b in new_batches.iter_mut() {
                let casted = zyron_executor::compute::cast_column(&b.columns[pos], target)
                    .map_err(ProtocolError::Database)?;
                b.columns[pos] = casted;
            }
            let col = &mut new_columns[pos];
            col.type_id = target;
            col.max_length = alter_extract_max_length(data_type);
            col.fractional_digits = data_type.timestamp_precision();
        }
        _ => {
            return Err(ProtocolError::Database(ZyronError::Internal(
                "rewrite_table_columns invoked for a non-column-shape operation".to_string(),
            )));
        }
    }

    // Renumber ordinals to match the new physical position.
    for (i, c) in new_columns.iter_mut().enumerate() {
        c.ordinal = i as u16;
    }

    // Capture every index definition (reading vector configs from the live
    // manager) before anything is dropped. Indexes on the dropped column do
    // not survive; the rest are rebuilt from the new heap.
    let old_indexes = server.catalog.get_indexes_for_table(table_id);
    let mut survivors: Vec<IndexRebuild> = Vec::new();
    for idx in &old_indexes {
        let mut col_names = Vec::with_capacity(idx.columns.len());
        let mut references_dropped = false;
        for ic in &idx.columns {
            if Some(ic.column_id) == dropped_col_id {
                references_dropped = true;
            }
            if let Some(c) = old_table.columns.iter().find(|c| c.id == ic.column_id) {
                col_names.push(c.name.clone());
            }
        }
        if references_dropped {
            continue;
        }
        let vector_config = if idx.index_type == zyron_catalog::IndexType::Vector {
            server
                .vector_manager
                .as_ref()
                .and_then(|m| m.get_index(idx.id.0))
                .map(|vi| (vi.dimension_count(), vi.hnsw_config()))
        } else {
            None
        };
        survivors.push(IndexRebuild {
            name: idx.name.clone(),
            col_names,
            index_type: idx.index_type,
            unique: idx.unique,
            parameters: idx.parameters.clone(),
            vector_config,
        });
    }

    // Allocate fresh heap and FSM files. The old files stay intact until the
    // new heap is fully built and the rewrite commits.
    let (new_heap_id, new_fsm_id) = server.catalog.alloc_heap_files();
    let old_heap_id = old_table.heap_file_id;
    let old_fsm_id = old_table.fsm_file_id;

    // Swap the catalog to the new schema and new files. Constraints that
    // reference the dropped column are pruned.
    let mut new_entry = old_table.clone();
    new_entry.columns = new_columns;
    new_entry.heap_file_id = new_heap_id;
    new_entry.fsm_file_id = new_fsm_id;
    if let Some(did) = dropped_col_id {
        new_entry.constraints.retain(|c| !c.columns.contains(&did));
    }
    server
        .catalog
        .update_table(new_entry)
        .await
        .map_err(ProtocolError::Database)?;

    // Drop every old index from the catalog and its manager.
    for idx in &old_indexes {
        let _ = server.catalog.drop_index(table_id, &idx.name).await;
        match idx.index_type {
            zyron_catalog::IndexType::BTree => {
                let _ = server.btree_indexes.remove_async(&idx.id.0).await;
            }
            zyron_catalog::IndexType::Fulltext => {
                if let Some(m) = &server.fts_manager {
                    let _ = m.drop_index(idx.id.0);
                }
            }
            zyron_catalog::IndexType::Vector => {
                if let Some(m) = &server.vector_manager {
                    let _ = m.drop_index(idx.id.0);
                }
            }
            zyron_catalog::IndexType::Spatial => {
                if let Some(m) = &server.spatial_manager {
                    m.drop_index(idx.id.0);
                }
            }
        }
    }

    // Recreate the surviving indexes empty, resolving column ids against the
    // updated table. The InsertOperator repopulates them as it writes the new
    // heap.
    let updated = server
        .catalog
        .get_table(schema_id, table_name)
        .map_err(ProtocolError::Database)?;
    let checkpoint_dir = server.data_dir.join("indexes");
    let _ = std::fs::create_dir_all(&checkpoint_dir);
    for s in &survivors {
        match s.index_type {
            zyron_catalog::IndexType::BTree => {
                let new_id = server
                    .catalog
                    .create_index(
                        table_id,
                        schema_id,
                        &s.name,
                        &s.col_names,
                        s.unique,
                        zyron_catalog::IndexType::BTree,
                    )
                    .await
                    .map_err(ProtocolError::Database)?;
                let entry = server
                    .catalog
                    .get_indexes_for_table(table_id)
                    .into_iter()
                    .find(|e| e.id == new_id)
                    .ok_or_else(|| {
                        ProtocolError::Database(ZyronError::Internal(
                            "recreated B-tree index missing from catalog".to_string(),
                        ))
                    })?;
                let btree =
                    zyron_storage::BTreeIndex::create(entry.index_file_id, checkpoint_dir.clone())
                        .await
                        .map_err(ProtocolError::Database)?;
                let _ = server
                    .btree_indexes
                    .insert_async(new_id.0, Arc::new(btree))
                    .await;
            }
            zyron_catalog::IndexType::Fulltext => {
                let new_id = server
                    .catalog
                    .create_index(
                        table_id,
                        schema_id,
                        &s.name,
                        &s.col_names,
                        false,
                        zyron_catalog::IndexType::Fulltext,
                    )
                    .await
                    .map_err(ProtocolError::Database)?;
                if let Some(m) = &server.fts_manager {
                    let col_ids: Vec<u16> = s
                        .col_names
                        .iter()
                        .filter_map(|n| {
                            updated
                                .columns
                                .iter()
                                .find(|c| c.name == *n)
                                .map(|c| c.id.0)
                        })
                        .collect();
                    m.create_index(new_id.0, table_id.0, col_ids)
                        .map_err(ProtocolError::Database)?;
                }
            }
            zyron_catalog::IndexType::Vector => {
                let new_id = server
                    .catalog
                    .create_index(
                        table_id,
                        schema_id,
                        &s.name,
                        &s.col_names,
                        false,
                        zyron_catalog::IndexType::Vector,
                    )
                    .await
                    .map_err(ProtocolError::Database)?;
                if let (Some(m), Some((dims, cfg)), Some(col0)) = (
                    &server.vector_manager,
                    &s.vector_config,
                    s.col_names.first(),
                ) {
                    let col_id = updated
                        .columns
                        .iter()
                        .find(|c| c.name == *col0)
                        .map(|c| c.id.0)
                        .unwrap_or(0);
                    m.create_index(new_id.0, table_id.0, col_id, *dims, cfg.clone())
                        .map_err(ProtocolError::Database)?;
                }
            }
            zyron_catalog::IndexType::Spatial => {
                let new_id = server
                    .catalog
                    .create_index_with_params(
                        table_id,
                        schema_id,
                        &s.name,
                        &s.col_names,
                        false,
                        zyron_catalog::IndexType::Spatial,
                        s.parameters.clone(),
                    )
                    .await
                    .map_err(ProtocolError::Database)?;
                if let Some(m) = &server.spatial_manager {
                    let (dims, srid) = decode_spatial_params(&s.parameters);
                    m.create_index(new_id.0, dims, srid);
                }
            }
        }
    }

    // Populate the new heap and the recreated indexes by replaying the reshaped
    // rows through the standard insert pipeline.
    run_rebuild_insert(server, table_id, new_batches).await?;

    // The rewrite committed. Reclaim the old heap files and drop the stale
    // cached handle so later reads open the new files.
    let _ = server.heap_files.remove_async(&old_heap_id).await;
    if let Err(e) = server.disk_manager.delete_file(old_heap_id).await {
        eprintln!("ALTER TABLE: failed to remove old heap file {old_heap_id}: {e}");
    }
    if let Err(e) = server.disk_manager.delete_file(old_fsm_id).await {
        eprintln!("ALTER TABLE: failed to remove old FSM file {old_fsm_id}: {e}");
    }

    Ok(DdlResult::Tag("ALTER TABLE".to_string()))
}

/// Extracts the max_length parameter from a sized data type, mirroring the
/// catalog's create-table conversion so rewritten columns keep their width.
fn alter_extract_max_length(dt: &zyron_parser::ast::DataType) -> Option<usize> {
    // Same slot CREATE TABLE fills, so a column added by ALTER carries the
    // same declared bound. Reading it here from a local copy of the rule is
    // what let DECIMAL(p,s) reach this path with its precision dropped, and
    // an unbounded decimal stores a value the declaration says cannot exist
    dt.declared_max_length()
}

/// Decodes a spatial index parameter blob into (dims, srid). Layout is
/// [u8 dims][u32 srid little-endian]; missing or short blobs default to 2D
/// WGS-84.
fn decode_spatial_params(params: &Option<Vec<u8>>) -> (u8, u32) {
    match params {
        Some(p) if p.len() >= 5 => {
            let dims = p[0];
            let srid = u32::from_le_bytes([p[1], p[2], p[3], p[4]]);
            (dims, srid)
        }
        _ => (2, 4326),
    }
}

/// Builds a column of `n` rows all holding `fill`. Used to backfill an added
/// column. TIMESTAMP(p>6) values are scaled from i64 microseconds to i128
/// picoseconds to match the physical buffer.
fn build_constant_column(
    logical_type: zyron_common::TypeId,
    fractional_digits: Option<u8>,
    fill: &zyron_executor::column::ScalarValue,
    n: usize,
) -> zyron_executor::column::Column {
    use zyron_common::TypeId;
    use zyron_executor::batch::ColumnBuilder;
    use zyron_executor::column::ScalarValue;

    let phys = TypeId::timestamp_physical_type_id(logical_type, fractional_digits);
    let mut builder = if phys != logical_type {
        ColumnBuilder::new_ts(logical_type, phys, fractional_digits, n)
    } else {
        ColumnBuilder::new(logical_type, n)
    };
    let value = if phys == TypeId::Int128
        && matches!(logical_type, TypeId::Timestamp | TypeId::TimestampTz)
    {
        match fill {
            ScalarValue::Int64(us) => ScalarValue::Int128(*us as i128 * 1_000_000),
            other => other.clone(),
        }
    } else {
        fill.clone()
    };
    for _ in 0..n {
        builder.push(&value);
    }
    builder.finish()
}

/// Evaluates a column default expression once to a scalar of the target type.
/// Runs `SELECT <default>` so literals and non-correlated functions (now(),
/// gen_random_uuid()) resolve through the normal expression pipeline.
async fn eval_default_scalar(
    server: &Arc<ServerState>,
    expr: &zyron_parser::ast::Expr,
    target: zyron_common::TypeId,
) -> Result<zyron_executor::column::ScalarValue, ProtocolError> {
    use zyron_executor::column::ScalarValue;

    let sql = format!(
        "SELECT {} AS v",
        crate::lifecycle_dispatch::expr_to_sql(expr)
    );
    let batches = select_query_batches(server, &sql).await?;
    let scalar = batches
        .iter()
        .find(|b| b.num_rows > 0 && !b.columns.is_empty())
        .map(|b| b.columns[0].get_scalar(0))
        .unwrap_or(ScalarValue::Null);
    if scalar.is_null() {
        return Ok(ScalarValue::Null);
    }
    zyron_executor::compute::cast_scalar(&scalar, target).map_err(ProtocolError::Database)
}

/// Runs a read-only SELECT and returns its result batches. Uses a throwaway
/// ReadCommitted transaction that is aborted afterward.
async fn select_query_batches(
    server: &Arc<ServerState>,
    sql: &str,
) -> Result<Vec<zyron_executor::batch::DataBatch>, ProtocolError> {
    use zyron_executor::context::ExecutionContext;

    let stmt = zyron_parser::parse(sql)
        .map_err(ProtocolError::Database)?
        .into_iter()
        .next()
        .ok_or_else(|| ProtocolError::Database(ZyronError::Internal("empty sql".into())))?;
    let plan = zyron_planner::plan(
        &server.catalog,
        zyron_catalog::DatabaseId(1),
        vec!["public".to_string()],
        stmt,
        Some(&server.peer_facts()),
    )
    .await
    .map_err(ProtocolError::Database)?;

    let mut txn = server
        .txn_manager
        .begin(zyron_storage::txn::IsolationLevel::ReadCommitted)
        .map_err(ProtocolError::Database)?;
    let snapshot = txn.snapshot.clone();
    let txn_id = u32::try_from(txn.txn_id)
        .map_err(|_| ProtocolError::Database(ZyronError::Internal("txn id overflow".into())))?;
    let mut ctx = ExecutionContext::new(
        server.catalog.clone(),
        server.wal.clone(),
        server.buffer_pool.clone(),
        server.disk_manager.clone(),
        txn_id,
        snapshot,
    );
    ctx.heap_files = Some(Arc::clone(&server.heap_files));
    ctx.btree_indexes = Some(Arc::clone(&server.btree_indexes));
    ctx.foreign_reader = server.foreign_reader.clone();
    ctx.peers = Some(Arc::clone(&server.peers));
    let ctx = Arc::new(ctx);
    let result = zyron_executor::execute(plan, &ctx).await;
    let _ = server.txn_manager.abort(&mut txn);
    result.map_err(ProtocolError::Database)
}

/// Plans and executes a write statement (INSERT/DELETE/UPDATE) under a write
/// transaction, wiring index managers so the write maintains every index, then
/// commits. Aborts on failure. Used to populate and refresh materialized view
/// backing tables.
async fn execute_write_stmt(
    server: &Arc<ServerState>,
    db_id: zyron_catalog::DatabaseId,
    search_path: Vec<String>,
    stmt: zyron_parser::Statement,
) -> Result<(), ProtocolError> {
    use zyron_executor::context::ExecutionContext;

    let plan = zyron_planner::plan(
        &server.catalog,
        db_id,
        search_path,
        stmt,
        Some(&server.peer_facts()),
    )
    .await
    .map_err(ProtocolError::Database)?;

    let mut txn = server
        .txn_manager
        .begin(zyron_storage::txn::IsolationLevel::ReadCommitted)
        .map_err(ProtocolError::Database)?;
    let snapshot = txn.snapshot.clone();
    let txn_id = u32::try_from(txn.txn_id)
        .map_err(|_| ProtocolError::Database(ZyronError::Internal("txn id overflow".into())))?;
    let mut ctx = ExecutionContext::new(
        server.catalog.clone(),
        server.wal.clone(),
        server.buffer_pool.clone(),
        server.disk_manager.clone(),
        txn_id,
        snapshot,
    );
    ctx.heap_files = Some(Arc::clone(&server.heap_files));
    ctx.btree_indexes = Some(Arc::clone(&server.btree_indexes));
    ctx.foreign_reader = server.foreign_reader.clone();
    ctx.peers = Some(Arc::clone(&server.peers));
    ctx.intent_locks = Some(Arc::clone(server.txn_manager.intent_locks()));
    ctx.row_locks = Some(Arc::clone(server.txn_manager.lock_table()));
    ctx.doc_registry = Some(Arc::clone(&server.doc_registry));
    if let Some(m) = &server.fts_manager {
        ctx.set_fts_manager(Arc::clone(m));
    }
    if let Some(m) = &server.vector_manager {
        ctx.set_vector_manager(Arc::clone(m));
    }
    if let Some(m) = &server.spatial_manager {
        ctx.set_spatial_manager(Arc::clone(m));
    }
    let ctx = Arc::new(ctx);

    match zyron_executor::execute(plan, &ctx).await {
        Ok(_) => {
            server
                .txn_manager
                .commit(&mut txn)
                .await
                .map_err(ProtocolError::Database)?;
            Ok(())
        }
        Err(e) => {
            let _ = server.txn_manager.abort(&mut txn);
            Err(ProtocolError::Database(e))
        }
    }
}

/// Replays reshaped batches through the InsertOperator under a write
/// transaction, populating the new heap and every recreated index, then
/// commits. Aborts on any failure.
async fn run_rebuild_insert(
    server: &Arc<ServerState>,
    table_id: zyron_catalog::TableId,
    batches: Vec<zyron_executor::batch::DataBatch>,
) -> Result<u64, ProtocolError> {
    use zyron_executor::context::ExecutionContext;
    use zyron_executor::operator::Operator;
    use zyron_executor::operator::modify::InsertOperator;

    let expected: u64 = batches.iter().map(|b| b.num_rows as u64).sum();

    let mut txn = server
        .txn_manager
        .begin(zyron_storage::txn::IsolationLevel::ReadCommitted)
        .map_err(ProtocolError::Database)?;
    let snapshot = txn.snapshot.clone();
    let txn_id = u32::try_from(txn.txn_id)
        .map_err(|_| ProtocolError::Database(ZyronError::Internal("txn id overflow".into())))?;

    let mut ctx = ExecutionContext::new(
        server.catalog.clone(),
        server.wal.clone(),
        server.buffer_pool.clone(),
        server.disk_manager.clone(),
        txn_id,
        snapshot,
    );
    // heap_files left unset so get_heap_file opens a fresh HeapFile over the
    // newly allocated empty files. Index managers are wired so the insert
    // pipeline rebuilds FTS, vector, spatial, and B-tree indexes in one pass.
    ctx.btree_indexes = Some(Arc::clone(&server.btree_indexes));
    ctx.foreign_reader = server.foreign_reader.clone();
    ctx.peers = Some(Arc::clone(&server.peers));
    ctx.doc_registry = Some(Arc::clone(&server.doc_registry));
    if let Some(m) = &server.fts_manager {
        ctx.set_fts_manager(Arc::clone(m));
    }
    if let Some(m) = &server.vector_manager {
        ctx.set_vector_manager(Arc::clone(m));
    }
    if let Some(m) = &server.spatial_manager {
        ctx.set_spatial_manager(Arc::clone(m));
    }
    let ctx = Arc::new(ctx);

    let source = Box::new(BatchSourceOperator::new(batches));
    // The rebuild batches are already in full new-table-column order, so the
    // insert maps one-to-one (identity reshape).
    let full_targets: Vec<zyron_catalog::ColumnId> = server
        .catalog
        .get_table_by_id(table_id)
        .map_err(ProtocolError::Database)?
        .columns
        .iter()
        .map(|c| c.id)
        .collect();
    let mut op = InsertOperator::new(
        source,
        Arc::clone(&ctx),
        table_id,
        full_targets,
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );

    let mut run_result: Result<(), ZyronError> = Ok(());
    loop {
        match op.next().await {
            Ok(Some(_)) => {}
            Ok(None) => break,
            Err(e) => {
                run_result = Err(e);
                break;
            }
        }
    }

    match run_result {
        Ok(()) => {
            server
                .txn_manager
                .commit(&mut txn)
                .await
                .map_err(ProtocolError::Database)?;
            Ok(expected)
        }
        Err(e) => {
            let _ = server.txn_manager.abort(&mut txn);
            Err(ProtocolError::Database(e))
        }
    }
}

/// Runs a read-only SELECT and returns the number of rows it produced. Uses a
/// throwaway ReadCommitted transaction that is aborted afterward.
async fn count_query(server: &Arc<ServerState>, sql: &str) -> Result<u64, ProtocolError> {
    use zyron_executor::context::ExecutionContext;

    let stmt = zyron_parser::parse(sql)
        .map_err(ProtocolError::Database)?
        .into_iter()
        .next()
        .ok_or_else(|| ProtocolError::Database(ZyronError::Internal("empty sql".into())))?;
    let plan = zyron_planner::plan(
        &server.catalog,
        zyron_catalog::DatabaseId(1),
        vec!["public".to_string()],
        stmt,
        Some(&server.peer_facts()),
    )
    .await
    .map_err(ProtocolError::Database)?;

    let mut txn = server
        .txn_manager
        .begin(zyron_storage::txn::IsolationLevel::ReadCommitted)
        .map_err(ProtocolError::Database)?;
    let snapshot = txn.snapshot.clone();
    let txn_id = u32::try_from(txn.txn_id)
        .map_err(|_| ProtocolError::Database(ZyronError::Internal("txn id overflow".into())))?;
    let ctx = Arc::new(ExecutionContext::new(
        server.catalog.clone(),
        server.wal.clone(),
        server.buffer_pool.clone(),
        server.disk_manager.clone(),
        txn_id,
        snapshot,
    ));
    let result = zyron_executor::execute(plan, &ctx).await;
    let _ = server.txn_manager.abort(&mut txn);
    let batches = result.map_err(ProtocolError::Database)?;
    Ok(batches.iter().map(|b| b.num_rows as u64).sum())
}

/// Finds a mutable column entry by name, erroring if absent.
fn alter_find_column_mut<'a>(
    entry: &'a mut zyron_catalog::schema::TableEntry,
    name: &str,
) -> Result<&'a mut zyron_catalog::schema::ColumnEntry, ProtocolError> {
    entry
        .columns
        .iter_mut()
        .find(|c| c.name == name)
        .ok_or_else(|| ProtocolError::Database(ZyronError::ColumnNotFound(name.to_string())))
}

/// Converts a parser TableConstraint to a catalog ConstraintEntry, resolving
/// local column names to ids and, for foreign keys, resolving the referenced
/// table and columns to their catalog ids.
fn build_constraint_entry(
    table_name: &str,
    tc: &zyron_parser::ast::TableConstraint,
    columns: &[zyron_catalog::schema::ColumnEntry],
    server: &Arc<ServerState>,
    schema_id: zyron_catalog::ids::SchemaId,
) -> Result<zyron_catalog::schema::ConstraintEntry, ProtocolError> {
    use zyron_catalog::ids::ColumnId;
    use zyron_catalog::schema::{ConstraintEntry, ConstraintType};
    use zyron_parser::ast::TableConstraintKind as TC;

    let resolve = |names: &[String]| -> Result<Vec<ColumnId>, ProtocolError> {
        let mut ids = Vec::with_capacity(names.len());
        for n in names {
            let c = columns
                .iter()
                .find(|c| c.name == *n)
                .ok_or_else(|| ProtocolError::Database(ZyronError::ColumnNotFound(n.clone())))?;
            ids.push(c.id);
        }
        Ok(ids)
    };

    Ok(match &tc.kind {
        TC::PrimaryKey(cols) => ConstraintEntry {
            name: tc
                .name
                .clone()
                .unwrap_or_else(|| format!("pk_{table_name}_{}", cols.join("_"))),
            constraint_type: ConstraintType::PrimaryKey,
            columns: resolve(cols)?,
            ref_table_id: None,
            ref_columns: vec![],
            check_expr: None,
            on_delete: zyron_catalog::ReferentialAction::NoAction,
            on_update: zyron_catalog::ReferentialAction::NoAction,
            enforced: true,
            on_violation: zyron_catalog::schema::ConstraintViolationAction::Fail,
            quarantine_table_id: None,
        },
        TC::Unique(cols) => ConstraintEntry {
            name: tc
                .name
                .clone()
                .unwrap_or_else(|| format!("uq_{table_name}_{}", cols.join("_"))),
            constraint_type: ConstraintType::Unique,
            columns: resolve(cols)?,
            ref_table_id: None,
            ref_columns: vec![],
            check_expr: None,
            on_delete: zyron_catalog::ReferentialAction::NoAction,
            on_update: zyron_catalog::ReferentialAction::NoAction,
            enforced: true,
            on_violation: zyron_catalog::schema::ConstraintViolationAction::Fail,
            quarantine_table_id: None,
        },
        TC::Check(expr) => ConstraintEntry {
            name: tc
                .name
                .clone()
                .unwrap_or_else(|| format!("ck_{table_name}")),
            constraint_type: ConstraintType::Check,
            columns: vec![],
            ref_table_id: None,
            ref_columns: vec![],
            // Store as re-parseable SQL so INSERT/UPDATE can bind and enforce it.
            check_expr: Some(zyron_parser::expr_to_sql(expr)),
            on_delete: zyron_catalog::ReferentialAction::NoAction,
            on_update: zyron_catalog::ReferentialAction::NoAction,
            enforced: true,
            on_violation: zyron_catalog::schema::ConstraintViolationAction::Fail,
            quarantine_table_id: None,
        },
        TC::ForeignKey {
            columns: cols,
            ref_table,
            ref_columns,
            on_delete,
            on_update,
        } => {
            let ref_tbl = server
                .catalog
                .get_table(schema_id, ref_table)
                .map_err(ProtocolError::Database)?;
            let ref_col_ids = {
                let mut ids = Vec::with_capacity(ref_columns.len());
                for n in ref_columns {
                    let c = ref_tbl
                        .columns
                        .iter()
                        .find(|c| c.name == *n)
                        .ok_or_else(|| {
                            ProtocolError::Database(ZyronError::ColumnNotFound(format!(
                                "{ref_table}.{n}"
                            )))
                        })?;
                    ids.push(c.id);
                }
                ids
            };
            ConstraintEntry {
                name: tc
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("fk_{table_name}_{}", cols.join("_"))),
                constraint_type: ConstraintType::ForeignKey,
                columns: resolve(cols)?,
                ref_table_id: Some(ref_tbl.id),
                ref_columns: ref_col_ids,
                check_expr: None,
                on_delete: map_parser_ref_action(*on_delete),
                on_update: map_parser_ref_action(*on_update),
                enforced: true,
                on_violation: zyron_catalog::schema::ConstraintViolationAction::Fail,
                quarantine_table_id: None,
            }
        }
    })
}

/// Scans the existing rows of a table and rejects an ADD CONSTRAINT when any
/// row already violates it. Mirrors standard SQL: the constraint is only
/// accepted if the current data satisfies it.
///
/// CHECK rejects rows where the predicate evaluates FALSE (NULL/unknown passes).
/// FOREIGN KEY rejects rows whose key columns are all non-NULL yet match no
/// parent row, found with a LEFT anti-join (no correlated subquery). UNIQUE
/// rejects duplicate groups over non-NULL key tuples. PRIMARY KEY additionally
/// rejects any NULL in a key column.
async fn validate_constraint_against_existing(
    table_name: &str,
    tc: &zyron_parser::ast::TableConstraint,
    server: &Arc<ServerState>,
    schema_id: zyron_catalog::ids::SchemaId,
) -> Result<(), ProtocolError> {
    use zyron_catalog::schema::ConstraintType;
    use zyron_parser::ast::TableConstraintKind as TC;

    let quote =
        |cols: &[String]| -> Vec<String> { cols.iter().map(|c| format!("\"{c}\"")).collect() };

    match &tc.kind {
        TC::Check(expr) => {
            let pred = zyron_parser::expr_to_sql(expr);
            let violating = count_query(
                server,
                &format!("SELECT 1 FROM \"{table_name}\" WHERE NOT ({pred})"),
            )
            .await?;
            if violating > 0 {
                return Err(ProtocolError::Database(ZyronError::CheckViolation(
                    format!(
                        "{violating} existing row(s) in \"{table_name}\" violate CHECK ({pred})"
                    ),
                )));
            }
        }
        TC::ForeignKey {
            columns,
            ref_table,
            ref_columns,
            ..
        } => {
            // Resolve the referenced columns, defaulting to the parent primary
            // key when the FK omits an explicit column list.
            let ref_tbl = server
                .catalog
                .get_table(schema_id, ref_table)
                .map_err(ProtocolError::Database)?;
            let effective_ref: Vec<String> = if ref_columns.is_empty() {
                let pk = ref_tbl
                    .constraints
                    .iter()
                    .find(|c| c.constraint_type == ConstraintType::PrimaryKey)
                    .ok_or_else(|| {
                        ProtocolError::Database(ZyronError::Internal(format!(
                            "referenced table \"{ref_table}\" has no primary key for the foreign key to target"
                        )))
                    })?;
                let mut names = Vec::with_capacity(pk.columns.len());
                for id in &pk.columns {
                    let name = ref_tbl
                        .columns
                        .iter()
                        .find(|c| c.id == *id)
                        .map(|c| c.name.clone())
                        .ok_or_else(|| {
                            ProtocolError::Database(ZyronError::Internal(
                                "primary key references an unknown column".into(),
                            ))
                        })?;
                    names.push(name);
                }
                names
            } else {
                ref_columns.clone()
            };

            if columns.len() != effective_ref.len() {
                return Err(ProtocolError::Database(ZyronError::Internal(format!(
                    "foreign key column count ({}) does not match referenced column count ({})",
                    columns.len(),
                    effective_ref.len()
                ))));
            }

            let child = quote(columns);
            let parent = quote(&effective_ref);
            let on = child
                .iter()
                .zip(parent.iter())
                .map(|(c, p)| format!("c.{c} = p.{p}"))
                .collect::<Vec<_>>()
                .join(" AND ");
            let all_non_null = child
                .iter()
                .map(|c| format!("c.{c} IS NOT NULL"))
                .collect::<Vec<_>>()
                .join(" AND ");
            let orphans = count_query(
                server,
                &format!(
                    "SELECT 1 FROM \"{table_name}\" c LEFT JOIN \"{ref_table}\" p ON {on} \
                     WHERE ({all_non_null}) AND p.{} IS NULL",
                    parent[0]
                ),
            )
            .await?;
            if orphans > 0 {
                return Err(ProtocolError::Database(ZyronError::Internal(format!(
                    "{orphans} existing row(s) in \"{table_name}\" reference a missing row in \"{ref_table}\""
                ))));
            }
        }
        TC::Unique(cols) | TC::PrimaryKey(cols) => {
            let is_pk = matches!(&tc.kind, TC::PrimaryKey(_));
            let qcols = quote(cols);

            if is_pk {
                let any_null = qcols
                    .iter()
                    .map(|c| format!("{c} IS NULL"))
                    .collect::<Vec<_>>()
                    .join(" OR ");
                let nulls = count_query(
                    server,
                    &format!("SELECT 1 FROM \"{table_name}\" WHERE {any_null}"),
                )
                .await?;
                if nulls > 0 {
                    return Err(ProtocolError::Database(ZyronError::Internal(format!(
                        "{nulls} existing row(s) in \"{table_name}\" contain NULL in a PRIMARY KEY column"
                    ))));
                }
            }

            // Multiple NULLs are distinct under UNIQUE, so exclude any row with a
            // NULL key component from the duplicate scan. A key has duplicates
            // when the non-null row count exceeds the distinct key count.
            let not_null = qcols
                .iter()
                .map(|c| format!("{c} IS NOT NULL"))
                .collect::<Vec<_>>()
                .join(" AND ");
            let key_list = qcols.join(", ");
            let total = count_query(
                server,
                &format!("SELECT {key_list} FROM \"{table_name}\" WHERE {not_null}"),
            )
            .await?;
            let distinct = count_query(
                server,
                &format!("SELECT DISTINCT {key_list} FROM \"{table_name}\" WHERE {not_null}"),
            )
            .await?;
            if total > distinct {
                let kind = if is_pk { "PRIMARY KEY" } else { "UNIQUE" };
                return Err(ProtocolError::Database(ZyronError::Internal(format!(
                    "{} duplicate row(s) in \"{table_name}\" violate {kind} ({})",
                    total - distinct,
                    cols.join(", ")
                ))));
            }
        }
    }

    Ok(())
}

/// Maps a parser referential action to its catalog representation.
fn map_parser_ref_action(
    a: zyron_parser::ast::ReferentialAction,
) -> zyron_catalog::ReferentialAction {
    use zyron_catalog::ReferentialAction as C;
    use zyron_parser::ast::ReferentialAction as P;
    match a {
        P::NoAction => C::NoAction,
        P::Restrict => C::Restrict,
        P::Cascade => C::Cascade,
        P::SetNull => C::SetNull,
        P::SetDefault => C::SetDefault,
    }
}

async fn handle_create_table(
    stmt: &zyron_parser::ast::CreateTableStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (_, schema_id) = get_session_schema(session, server, None)?;

    // Check CREATE privilege on the target schema
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Schema,
        schema_id.0,
    )?;

    // Storage format, resolved before the catalog entry exists so a format
    // this node does not run refuses the statement whole
    let lake_format = resolve_create_table_format(server, stmt)?;

    match server
        .catalog
        .create_table(schema_id, &stmt.name, &stmt.columns, &stmt.constraints)
        .await
    {
        Ok(_) => {
            apply_create_table_retention(server, schema_id, &stmt.name, &stmt.options).await?;
            // A constraint declared ON VIOLATION QUARANTINE needs its
            // companion table to exist before the first row is rejected
            apply_constraint_quarantine(server, schema_id, &stmt.name).await?;
            if lake_format {
                if let Err(e) = apply_create_table_lake(server, schema_id, stmt).await {
                    // A lake table without its log is unusable, undo the
                    // catalog entry rather than leave a half-created table
                    let _ = server.catalog.drop_table(schema_id, &stmt.name).await;
                    return Err(e);
                }
            }
            // A declared PRIMARY KEY or UNIQUE needs the index that enforces
            // it, or the constraint is recorded and never checked
            if let Err(e) = provision_constraint_indexes(server, schema_id, &stmt.name).await {
                let _ = server.catalog.drop_table(schema_id, &stmt.name).await;
                return Err(e);
            }
            fire_event(
                server,
                zyron_pipeline::event_handler::EventType::TableCreated,
                &stmt.name,
                &[("table".to_string(), stmt.name.clone())],
            )
            .await;
            Ok(DdlResult::Tag("CREATE TABLE".to_string()))
        }
        Err(ZyronError::TableAlreadyExists(_)) if stmt.if_not_exists => {
            Ok(DdlResult::Tag("CREATE TABLE".to_string()))
        }
        Err(e) => Err(ProtocolError::Database(e)),
    }
}

/// Resolves the storage format a CREATE TABLE lands in and refuses a format
/// this node does not run. The USING clause names the format explicitly,
/// otherwise the deployment mode's default applies, so on a single-format
/// node nobody types USING at all. Returns true for ZyronLake.
fn resolve_create_table_format(
    server: &Arc<ServerState>,
    stmt: &zyron_parser::ast::CreateTableStatement,
) -> Result<bool, ProtocolError> {
    use zyron_parser::ast::TableFormat;

    let mode = server.deployment_mode;
    let wants_lake = match stmt.using {
        Some(TableFormat::ZyronLake) => {
            if !mode.allows_lake() {
                return Err(refuse_format(mode, "ZYRONLAKE", "heap", "lake"));
            }
            true
        }
        Some(TableFormat::Heap) => {
            if !mode.allows_heap() {
                return Err(refuse_format(mode, "HEAP", "ZyronLake", "db"));
            }
            false
        }
        None => mode.defaults_to_lake(),
    };

    // CLUSTER BY drives the lake writer's file layout. Accepting it on a heap
    // table would report success for a clause that changes nothing
    if !wants_lake && stmt.cluster_by.is_some() {
        return Err(ProtocolError::Database(ZyronError::ConfigError(format!(
            "CREATE TABLE \"{}\" CLUSTER BY applies to ZyronLake tables, this statement creates a heap table. \
             Add USING ZYRONLAKE, or drop the CLUSTER BY clause",
            stmt.name
        ))));
    }

    Ok(wants_lake)
}

/// Builds the refusal for a CREATE TABLE naming a format this node does not
/// run, naming the mode that refused it and the two modes that accept it.
fn refuse_format(
    mode: zyron_common::DeploymentMode,
    requested: &str,
    stored: &str,
    single_format_mode: &str,
) -> ProtocolError {
    ProtocolError::Database(ZyronError::ConfigError(format!(
        "CREATE TABLE ... USING {} is refused, this node runs deployment mode \"{}\" and stores {} tables only. \
         Set storage.deployment_mode = \"unified\" to run both formats here, or \"{}\" to make {} the default",
        requested,
        mode.as_str(),
        stored,
        single_format_mode,
        requested
    )))
}

/// Provisions the companion quarantine table when any constraint is declared
/// ON VIOLATION QUARANTINE, and records its id on those constraints.
///
/// One table serves every quarantining constraint, and the same table the
/// expectation path uses, so a row rejected by either lands in one place.
async fn apply_constraint_quarantine(
    server: &Arc<ServerState>,
    schema_id: zyron_catalog::SchemaId,
    table_name: &str,
) -> Result<(), ProtocolError> {
    use zyron_catalog::schema::ConstraintViolationAction;

    let table = server
        .catalog
        .get_table(schema_id, table_name)
        .map_err(ProtocolError::Database)?;
    if !table
        .constraints
        .iter()
        .any(|c| c.on_violation == ConstraintViolationAction::Quarantine)
    {
        return Ok(());
    }
    let quarantine_id = ensure_quarantine_table(server, schema_id, &table).await?;
    let mut entry = (*table).clone();
    for constraint in &mut entry.constraints {
        if constraint.on_violation == ConstraintViolationAction::Quarantine {
            constraint.quarantine_table_id = Some(quarantine_id);
        }
    }
    server
        .catalog
        .update_table(entry)
        .await
        .map_err(ProtocolError::Database)?;
    Ok(())
}

/// Creates the unique B+tree index that backs each enforced PRIMARY KEY and
/// UNIQUE constraint on a heap table.
///
/// Heap uniqueness is enforced by probing unique indexes, so a declared
/// constraint with no index behind it is a constraint nothing checks. The
/// index carries the constraint's own name, which is what lets DROP
/// CONSTRAINT take it away again.
///
/// A lake table needs none of this. `enforce_lake_unique` reads the declared
/// constraints directly and prunes on the manifest's per-file bounds, so an
/// index there would be a second mechanism answering the same question.
///
/// An existing index over exactly the constraint's columns already answers
/// for it and is left alone, so declaring a constraint over an indexed key
/// does not build a second copy of the same tree.
async fn provision_constraint_indexes(
    server: &Arc<ServerState>,
    schema_id: zyron_catalog::SchemaId,
    table_name: &str,
) -> Result<(), ProtocolError> {
    use zyron_catalog::schema::ConstraintType;

    let table = server
        .catalog
        .get_table(schema_id, table_name)
        .map_err(ProtocolError::Database)?;
    if table.lake.is_lake() {
        return Ok(());
    }

    let existing = server.catalog.get_indexes_for_table(table.id);
    for constraint in &table.constraints {
        if !matches!(
            constraint.constraint_type,
            ConstraintType::PrimaryKey | ConstraintType::Unique
        ) {
            continue;
        }
        // NOT ENFORCED keeps the declaration for the planner and asks the
        // write path to skip it, so it gets no index either
        if !constraint.enforced || constraint.columns.is_empty() {
            continue;
        }
        let covered = existing.iter().any(|idx| {
            idx.unique
                && idx.index_type == zyron_catalog::IndexType::BTree
                && idx.columns.len() == constraint.columns.len()
                && idx
                    .columns
                    .iter()
                    .zip(constraint.columns.iter())
                    .all(|(ic, cc)| ic.column_id == *cc)
        });
        if covered {
            continue;
        }
        let mut column_names = Vec::with_capacity(constraint.columns.len());
        for col_id in &constraint.columns {
            let Some(col) = table.columns.iter().find(|c| c.id == *col_id) else {
                return Err(ProtocolError::Database(ZyronError::ColumnNotFound(
                    format!(
                        "constraint \"{}\" names a column table \"{}\" does not have",
                        constraint.name, table_name
                    ),
                )));
            };
            column_names.push(col.name.clone());
        }
        create_backing_btree(server, schema_id, &table, &constraint.name, &column_names).await?;
    }
    Ok(())
}

/// Creates a unique B+tree index and fills it from the rows the table already
/// holds, so a constraint added to a populated table starts enforcing against
/// every existing row rather than only later ones.
async fn create_backing_btree(
    server: &Arc<ServerState>,
    schema_id: zyron_catalog::SchemaId,
    table: &Arc<zyron_catalog::schema::TableEntry>,
    index_name: &str,
    column_names: &[String],
) -> Result<(), ProtocolError> {
    let index_id = server
        .catalog
        .create_index(
            table.id,
            schema_id,
            index_name,
            column_names,
            true,
            zyron_catalog::IndexType::BTree,
        )
        .await
        .map_err(ProtocolError::Database)?;

    let entry = server
        .catalog
        .get_indexes_for_table(table.id)
        .into_iter()
        .find(|e| e.id == index_id)
        .ok_or_else(|| {
            ProtocolError::Database(ZyronError::Internal(format!(
                "index backing constraint \"{index_name}\" not found in catalog after creation"
            )))
        })?;

    let checkpoint_dir = server.data_dir.join("indexes");
    let _ = std::fs::create_dir_all(&checkpoint_dir);
    let btree = Arc::new(
        zyron_storage::BTreeIndex::create(entry.index_file_id, checkpoint_dir)
            .await
            .map_err(ProtocolError::Database)?,
    );

    let key_columns: Vec<zyron_catalog::ColumnId> =
        entry.columns.iter().map(|c| c.column_id).collect();
    let rows = crate::index_build::collect_live_rows(server, table)
        .await
        .map_err(ProtocolError::Database)?;
    crate::index_build::fill_btree_from_live_rows(table, &rows, &key_columns, &btree);

    let _ = server.btree_indexes.insert_async(index_id.0, btree).await;
    Ok(())
}

/// Drops the index backing a constraint, if the constraint had one. Called
/// when DROP CONSTRAINT removes the declaration, so the tree does not outlive
/// the rule it enforces.
async fn drop_constraint_index(
    server: &Arc<ServerState>,
    table_id: zyron_catalog::TableId,
    constraint_name: &str,
) {
    let Some(entry) = server
        .catalog
        .get_indexes_for_table(table_id)
        .into_iter()
        .find(|e| e.name == constraint_name)
    else {
        return;
    };
    if server
        .catalog
        .drop_index(table_id, constraint_name)
        .await
        .is_ok()
    {
        let _ = server.btree_indexes.remove_async(&entry.id.0).await;
    }
}

/// `ALTER TABLE t SET USING ZYRONLAKE | HEAP [WITH (drop_history = true)]`.
///
/// Converts in place, atomically from a reader's view: every row moves into
/// the destination and the destination commits before the catalog flips, so
/// the flip is the one moment the table changes format. A crash before it
/// leaves work startup reclaims; a crash after it leaves a source nobody
/// reads, reclaimed the same way.
///
/// Never automatic. The operator asks for it, because the two formats trade
/// commit rate against scan throughput and only the operator knows which the
/// table needs.
async fn handle_set_using(
    stmt: &zyron_parser::ast::AlterTableSetUsingStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    use zyron_catalog::schema::LakeConfig;
    use zyron_parser::ast::{TableFormat, TableOptionValue};

    let (_, schema_id) = get_session_schema(session, server, None)?;
    let table = server
        .catalog
        .get_table(schema_id, &stmt.table)
        .map_err(ProtocolError::Database)?;
    // Converting rewrites every row of the table, so it takes the same
    // privilege creating one does
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Table,
        table.id.0,
    )?;
    let to_lake = matches!(stmt.format, TableFormat::ZyronLake);
    if to_lake == table.lake.is_lake() {
        return Err(ProtocolError::Database(ZyronError::ConfigError(format!(
            "table \"{}\" is already stored as {}",
            stmt.table,
            if to_lake { "ZYRONLAKE" } else { "HEAP" }
        ))));
    }
    if to_lake && !server.deployment_mode.allows_lake() {
        return Err(ProtocolError::Database(ZyronError::ConfigError(format!(
            "this node runs deployment mode \"{}\" and stores heap tables only",
            server.deployment_mode.as_str()
        ))));
    }
    if !to_lake && !server.deployment_mode.allows_heap() {
        return Err(ProtocolError::Database(ZyronError::ConfigError(format!(
            "this node runs deployment mode \"{}\" and stores ZyronLake tables only",
            server.deployment_mode.as_str()
        ))));
    }
    let drop_history = stmt.options.iter().any(|o| {
        o.key.eq_ignore_ascii_case("drop_history")
            && matches!(&o.value, TableOptionValue::Boolean(true))
    });

    let (ctx, conversion_txn) = conversion_context(server)?;
    let paths = zyron_lake::LakePaths::new(server.disk_manager.data_dir(), table.id.0);
    let mut entry = (*table).clone();

    if to_lake {
        // Read both stores: a table with folded segments keeps rows the heap
        // walk alone cannot see
        let rows = zyron_executor::table_convert::read_heap_rows(&ctx, &table)
            .await
            .map_err(ProtocolError::Database)?;
        let lake_columns: Vec<zyron_lake::LakeColumn> = table
            .columns
            .iter()
            .map(|c| zyron_lake::LakeColumn {
                id: c.id.0 as u32,
                name: c.name.clone(),
                type_id: c.type_id,
                nullable: c.nullable,
                fractional_digits: c.fractional_digits,
                tz_offset_secs: c.tz_offset_secs,
                max_length: c.max_length.map(|n| n as u32),
                default_expr: c.default_expr.clone(),
            })
            .collect();
        let lake_schema =
            zyron_lake::LakeSchema::new(1, lake_columns).map_err(ProtocolError::Database)?;
        let now = conversion_timestamp();
        // Written before the flip, so a crash here leaves an orphan root
        let log = zyron_lake::load_lake_from_rows(
            &paths,
            &lake_schema,
            None,
            &std::collections::BTreeMap::new(),
            table.id.0 as u64,
            &rows,
            now,
        )
        .map_err(ProtocolError::Database)?;

        entry.lake = LakeConfig::lake();
        if let Err(e) = server.catalog.update_table(entry).await {
            // The flip failed, so the root is unreachable work
            let _ = zyron_lake::reclaim_orphan_root(&paths);
            return Err(ProtocolError::Database(e));
        }
        zyron_lake::TransactionLog::register_shared(Arc::new(log));
        // The rows now live in the lake, so the heap copy is dead weight.
        // Leaving it would resurrect every row on a conversion back
        reclaim_heap_storage(server, &table).await?;
        Ok(DdlResult::Tag("ALTER TABLE".to_string()))
    } else {
        let log = zyron_lake::TransactionLog::lookup_shared(&paths).ok_or_else(|| {
            ProtocolError::Database(ZyronError::ConfigError(format!(
                "this node does not run the lake tier, so it cannot convert \"{}\"",
                stmt.table
            )))
        })?;
        let rows = zyron_lake::read_all_rows(&paths, &log).map_err(ProtocolError::Database)?;
        let batches = zyron_executor::table_convert::cells_to_batches(
            &table,
            &rows,
            zyron_executor::batch::BATCH_SIZE,
        )
        .map_err(ProtocolError::Database)?;

        // The heap file is created lazily, so this is where a lake table
        // first gets one
        let heap = ctx
            .get_heap_file(table.id)
            .await
            .map_err(ProtocolError::Database)?;
        let mut writer_txn = conversion_txn;
        for batch in &batches {
            let tuples = zyron_executor::batch::batch_to_tuples(batch, &table.columns, ctx.txn_id);
            let mut records: Vec<(u32, &[u8])> = Vec::with_capacity(tuples.len());
            for tuple in &tuples {
                records.push((ctx.txn_id, tuple.data()));
            }
            server
                .wal
                .log_insert_batch_last_lsn(&records)
                .map_err(ProtocolError::Database)?;
            heap.insert_batch(&tuples)
                .await
                .map_err(ProtocolError::Database)?;
        }

        // The rows are only visible once their transaction commits, so this
        // lands before the catalog says the table is a heap table
        server
            .txn_manager
            .commit(&mut writer_txn)
            .await
            .map_err(ProtocolError::Database)?;

        entry.lake = if drop_history {
            LakeConfig::default()
        } else {
            // The history outlives the format, so the root stays and the
            // catalog says so, which is what keeps startup from reclaiming it
            LakeConfig::heap_retaining_history()
        };
        server
            .catalog
            .update_table(entry)
            .await
            .map_err(ProtocolError::Database)?;
        if drop_history {
            zyron_lake::TransactionLog::remove_shared(&paths);
            let _ = zyron_lake::reclaim_orphan_root(&paths);
        }
        Ok(DdlResult::Tag("ALTER TABLE".to_string()))
    }
}

/// Epoch microseconds for a conversion commit.
fn conversion_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
}

/// An execution context for reading and writing a table's rows during a
/// conversion. Runs under its own transaction, which is what gives the heap
/// read a snapshot.
fn conversion_context(
    server: &Arc<ServerState>,
) -> Result<
    (
        Arc<zyron_executor::context::ExecutionContext>,
        zyron_storage::txn::Transaction,
    ),
    ProtocolError,
> {
    let txn = server
        .txn_manager
        .begin(zyron_storage::txn::IsolationLevel::SnapshotIsolation)
        .map_err(ProtocolError::Database)?;
    let snapshot = txn.snapshot.clone();
    let txn_id = txn.txn_id as u32;
    let mut ctx = zyron_executor::context::ExecutionContext::new(
        Arc::clone(&server.catalog),
        Arc::clone(&server.wal),
        Arc::clone(&server.buffer_pool),
        Arc::clone(&server.disk_manager),
        txn_id,
        snapshot,
    );
    ctx.heap_files = Some(Arc::clone(&server.heap_files));
    ctx.btree_indexes = Some(Arc::clone(&server.btree_indexes));
    ctx.foreign_reader = server.foreign_reader.clone();
    ctx.peers = Some(Arc::clone(&server.peers));
    Ok((Arc::new(ctx), txn))
}

/// Truncates a converted table's heap and frees every cached page of it.
///
/// A converted table's rows live in its lake files now. The heap copy is not
/// just wasted space: a conversion back would read it and duplicate every
/// row, so it is removed as part of the conversion rather than left for a
/// vacuum to notice.
async fn reclaim_heap_storage(
    server: &Arc<ServerState>,
    table: &zyron_catalog::schema::TableEntry,
) -> Result<(), ProtocolError> {
    let heap_pages = server
        .disk_manager
        .num_pages(table.heap_file_id)
        .await
        .map_err(ProtocolError::Database)?;
    let fsm_pages = server
        .disk_manager
        .num_pages(table.fsm_file_id)
        .await
        .map_err(ProtocolError::Database)?;
    server
        .disk_manager
        .truncate_file(table.heap_file_id)
        .await
        .map_err(ProtocolError::Database)?;
    server
        .disk_manager
        .truncate_file(table.fsm_file_id)
        .await
        .map_err(ProtocolError::Database)?;
    // A cached handle or frame would keep serving rows the table no longer
    // stores here
    let _ = server.heap_files.remove_async(&table.heap_file_id).await;
    for page in 0..heap_pages {
        server
            .buffer_pool
            .delete_page(zyron_common::PageId::new(table.heap_file_id, page));
    }
    for page in 0..fsm_pages {
        server
            .buffer_pool
            .delete_page(zyron_common::PageId::new(table.fsm_file_id, page));
    }
    Ok(())
}

/// Materializes a USING ZYRONLAKE table: writes version one of its
/// transaction log carrying the schema, cluster spec and WITH options as
/// lake properties, then flips the catalog format flag. The table's
/// allocated heap file ids are never opened, storage files are created
/// lazily on first write and a lake table never writes one
async fn apply_create_table_lake(
    server: &Arc<ServerState>,
    schema_id: zyron_catalog::SchemaId,
    stmt: &zyron_parser::ast::CreateTableStatement,
) -> Result<(), ProtocolError> {
    use zyron_parser::ast::{ClusterMode, TableOptionValue};

    let table = server
        .catalog
        .get_table(schema_id, &stmt.name)
        .map_err(ProtocolError::Database)?;
    let mut entry = (*table).clone();

    let lake_columns: Vec<zyron_lake::LakeColumn> = entry
        .columns
        .iter()
        .map(|c| zyron_lake::LakeColumn {
            id: c.id.0 as u32,
            name: c.name.clone(),
            type_id: c.type_id,
            nullable: c.nullable,
            fractional_digits: c.fractional_digits,
            tz_offset_secs: c.tz_offset_secs,
            max_length: c.max_length.map(|n| n as u32),
            default_expr: c.default_expr.clone(),
        })
        .collect();
    let lake_schema =
        zyron_lake::LakeSchema::new(1, lake_columns).map_err(ProtocolError::Database)?;

    let cluster_spec = match &stmt.cluster_by {
        Some(clause) if !clause.keys.is_empty() => {
            let mut keys = Vec::with_capacity(clause.keys.len());
            for key in &clause.keys {
                let col = lake_schema.column_by_name(&key.column).ok_or_else(|| {
                    ProtocolError::Database(ZyronError::ParseError(format!(
                        "CLUSTER BY column \"{}\" is not a column of {}",
                        key.column, stmt.name
                    )))
                })?;
                let strategy = match &key.strategy {
                    Some(name) => parse_cluster_strategy(name)?,
                    None => zyron_lake::ClusterStrategy::RangePartition,
                };
                keys.push(zyron_lake::ClusterKey {
                    column_id: col.id,
                    strategy,
                    param: 0,
                });
            }
            Some(zyron_lake::ClusterSpec { spec_id: 1, keys })
        }
        _ => None,
    };

    let mut properties = std::collections::BTreeMap::new();
    for opt in &stmt.options {
        let value = match &opt.value {
            TableOptionValue::String(s) | TableOptionValue::Identifier(s) => s.clone(),
            TableOptionValue::Integer(i) => i.to_string(),
            TableOptionValue::Boolean(b) => b.to_string(),
            TableOptionValue::StringList(items) => items.join(","),
        };
        properties.insert(opt.key.to_ascii_lowercase(), value);
    }
    if let Some(clause) = &stmt.cluster_by {
        properties.insert(
            zyron_lake::CLUSTERING_MODE_PROPERTY.to_string(),
            clause.mode.as_str().to_ascii_lowercase(),
        );
        // Under Hybrid the declared keys are anchors measurement may not
        // reorder or drop. Force pins the whole spec and Auto pins
        // nothing, so neither needs a separate anchor list
        if clause.mode == ClusterMode::Hybrid {
            let anchors: Vec<&str> = clause.keys.iter().map(|k| k.column.as_str()).collect();
            properties.insert(
                zyron_lake::CLUSTERING_ANCHORS_PROPERTY.to_string(),
                anchors.join(","),
            );
        }
    }

    let timestamp_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0);
    let attempt = zyron_lake::CommitAttempt {
        operation: zyron_lake::OperationKind::SchemaChange,
        db_txn_id: 0,
        commit_lsn: 0,
        timestamp_us,
        read_predicate: None,
        audit: None,
    };
    // The storage root, the same one the scan path derives from
    let paths = zyron_lake::LakePaths::new(server.disk_manager.data_dir(), entry.id.0);
    let log = zyron_lake::TransactionLog::create(
        paths,
        attempt,
        &lake_schema,
        cluster_spec.as_ref(),
        &properties,
    )
    .map_err(ProtocolError::Database)?;
    // The empty table's statistics, so its first plan estimates zero rows
    // rather than falling to the planner's no-statistics defaults
    if let Ok(manifest) = log.latest_manifest() {
        zyron_executor::lake_stats::publish_manifest_stats(&server.catalog, &entry, &manifest);
    }
    zyron_lake::TransactionLog::register_shared(Arc::new(log));

    entry.lake = zyron_catalog::schema::LakeConfig::lake();
    server
        .catalog
        .update_table(entry)
        .await
        .map_err(ProtocolError::Database)?;
    Ok(())
}

/// Resolves a strategy name from `CLUSTER BY (col USING <strategy>)`
fn parse_cluster_strategy(name: &str) -> Result<zyron_lake::ClusterStrategy, ProtocolError> {
    zyron_lake::ClusterStrategy::from_name(name).ok_or_else(|| {
        ProtocolError::Database(ZyronError::ParseError(format!(
            "unknown clustering strategy \"{}\", expected BitInterleave, SpaceFilling, RangePartition or AntiCluster",
            name
        )))
    })
}

/// Opens the transaction log of a lake table named by clustering DDL.
///
/// Clustering governs both storage tiers, so a heap table is not refused
/// here: its policy lives in the catalog and the fold tier reads it. This
/// is only the lake half
fn lake_log_for_clustering(
    table_name: &str,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<std::sync::Arc<zyron_lake::TransactionLog>, ProtocolError> {
    let (_, schema_id) = get_session_schema(session, server, None)?;
    let entry = server
        .catalog
        .get_table(schema_id, table_name)
        .map_err(ProtocolError::Database)?;
    let paths = zyron_lake::LakePaths::new(server.disk_manager.data_dir(), entry.id.0);
    zyron_lake::TransactionLog::lookup_shared(&paths).ok_or_else(|| {
        ProtocolError::Database(ZyronError::ConfigError(format!(
            "this node does not run the lake tier, so it cannot cluster \"{}\"",
            table_name
        )))
    })
}

/// Resolves the cluster keys a statement declares against a table's
/// columns, refusing anything that cannot mean a layout
fn resolve_cluster_keys(
    clause: &zyron_parser::ast::ClusterByClause,
    table_name: &str,
    columns: &[zyron_catalog::schema::ColumnEntry],
) -> Result<(Vec<zyron_lake::ClusterKey>, Vec<String>), ProtocolError> {
    let mut keys = Vec::with_capacity(clause.keys.len());
    let mut names = Vec::with_capacity(clause.keys.len());
    let mut seen = std::collections::BTreeSet::new();
    for key in &clause.keys {
        let column = columns
            .iter()
            .find(|c| c.name.eq_ignore_ascii_case(&key.column))
            .ok_or_else(|| {
                ProtocolError::Database(ZyronError::ParseError(format!(
                    "CLUSTER BY column \"{}\" is not a column of {}",
                    key.column, table_name
                )))
            })?;
        let strategy = match &key.strategy {
            Some(name) => parse_cluster_strategy(name)?,
            None => zyron_lake::ClusterStrategy::RangePartition,
        };
        // A key declared twice orders rows by a dimension the first
        // occurrence already fixed
        if !seen.insert(column.id.0 as u32) {
            return Err(ProtocolError::Database(ZyronError::ParseError(format!(
                "CLUSTER BY names a column twice in {}",
                table_name
            ))));
        }
        keys.push(zyron_lake::ClusterKey {
            column_id: column.id.0 as u32,
            strategy,
            param: 0,
        });
        names.push(column.name.clone());
    }
    Ok((keys, names))
}

/// Persists a heap table's clustering policy. The fold tier reads it when
/// it lays out the next segment, so the change reaches the files the next
/// time rows fold rather than by rewriting what is already there
async fn set_heap_cluster_policy(
    server: &Arc<ServerState>,
    entry: &zyron_catalog::schema::TableEntry,
    keys: Option<&[zyron_lake::ClusterKey]>,
    mode: Option<zyron_common::ClusterMode>,
    schedule: Option<zyron_common::ClusteringSchedule>,
) -> Result<(), ProtocolError> {
    let mut updated = entry.clone();
    if let Some(keys) = keys {
        updated.cluster.set_keys(keys);
    }
    if let Some(mode) = mode {
        updated.cluster.mode = mode.to_u8();
    }
    if let Some(schedule) = schedule {
        updated.cluster.schedule = schedule.to_u8();
    }
    server
        .catalog
        .update_table(updated)
        .await
        .map_err(ProtocolError::Database)?;
    // The layout a scan costs changed with no catalog schema change, so
    // cached plans against the old one have to go
    server.catalog.bump_schema_version();
    Ok(())
}

/// The commit attempt clustering DDL uses. Clustering changes stand alone
/// rather than joining the session transaction: the layout is a physical
/// property of the table, not a row a rollback can take back
fn clustering_commit_attempt() -> zyron_lake::CommitAttempt<'static> {
    let timestamp_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0);
    zyron_lake::CommitAttempt {
        operation: zyron_lake::OperationKind::SetProperty,
        db_txn_id: 0,
        commit_lsn: 0,
        timestamp_us,
        read_predicate: None,
        audit: None,
    }
}

/// `ALTER TABLE t CLUSTER BY (...) | AUTO | (...) AUTO`.
///
/// Declaring keys commits a new cluster spec, which is what later appends
/// lay their rows out by and what a clustering pass rewrites existing
/// files to. `AUTO` with no keys changes the policy only: it hands the
/// choice to measurement without throwing away the layout the table
/// already has, because dropping a working layout to wait for evidence
/// would be a regression nobody asked for
async fn handle_cluster_by(
    stmt: &zyron_parser::ast::AlterTableClusterByStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    use zyron_parser::ast::ClusterMode;

    let (_, schema_id) = get_session_schema(session, server, None)?;
    let table = server
        .catalog
        .get_table(schema_id, &stmt.table)
        .map_err(ProtocolError::Database)?;
    let (keys, names) = resolve_cluster_keys(&stmt.clause, &stmt.table, &table.columns)?;

    let mode = stmt.clause.mode;
    if mode == ClusterMode::Hybrid && keys.is_empty() {
        return Err(ProtocolError::Database(ZyronError::ParseError(format!(
            "CLUSTER BY (...) AUTO on {} anchors the listed keys, so it needs at least one",
            stmt.table
        ))));
    }

    // A heap table's policy is catalog state, and the fold tier applies
    // it to the next segment it writes
    if !table.lake.is_lake() {
        set_heap_cluster_policy(server, &table, Some(&keys), Some(mode), None).await?;
        return Ok(DdlResult::Tag("ALTER TABLE".to_string()));
    }

    let log = lake_log_for_clustering(&stmt.table, server, session)?;
    let base = log.latest_manifest().map_err(ProtocolError::Database)?;
    let mode_name = mode.as_str().to_ascii_lowercase();
    let anchors = match mode {
        ClusterMode::Hybrid => names.join(","),
        _ => String::new(),
    };
    let spec_id = base.cluster_spec.spec_id.saturating_add(1);
    let declared = !keys.is_empty();
    let spec = zyron_lake::ClusterSpec { spec_id, keys };

    log.commit(clustering_commit_attempt(), |_| {
        let mut entries = vec![zyron_lake::LogEntry::SetProperty {
            key: zyron_lake::CLUSTERING_MODE_PROPERTY.to_string(),
            value: mode_name.clone(),
        }];
        entries.push(zyron_lake::LogEntry::SetProperty {
            key: zyron_lake::CLUSTERING_ANCHORS_PROPERTY.to_string(),
            value: anchors.clone(),
        });
        // AUTO with no keys sets the policy and leaves the layout to the
        // clustering planner, so no spec is committed here
        if declared {
            entries.push(zyron_lake::LogEntry::SetClusterSpec(spec.clone()));
        }
        Ok(entries)
    })
    .map_err(ProtocolError::Database)?;

    // A layout change invalidates cached plans that costed the old one
    server.catalog.bump_schema_version();
    Ok(DdlResult::Tag("ALTER TABLE".to_string()))
}

/// `SHOW CLUSTERING FOR t`: what the layout is, what the workload
/// measured, and what measurement would choose.
///
/// Reporting runs under every mode including Force, because an operator
/// who pinned a layout is exactly the person who should be told that
/// measurement disagrees. It changes nothing: the keys it reports as the
/// measured choice are a proposal, and only `ALTER TABLE ... CLUSTER BY`
/// or an accepted maintenance pass ever moves a file.
///
/// It reports the fit the workload actually measured and does not report
/// a predicted fit for the proposal. Predicting one means writing the
/// candidate files and scoring their statistics, which is what a
/// clustering pass does; a number invented here without that work would
/// be a guess dressed as a measurement.
async fn handle_show_clustering(
    stmt: &zyron_parser::ast::ShowStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let Some(table_name) = stmt.target.as_deref() else {
        return Err(ProtocolError::Database(ZyronError::ParseError(
            "SHOW CLUSTERING needs a table, as in SHOW CLUSTERING FOR t".into(),
        )));
    };
    if !stmt.name.eq_ignore_ascii_case("clustering") {
        return Err(ProtocolError::Database(ZyronError::ParseError(format!(
            "SHOW {} FOR <table> is not a statement",
            stmt.name
        ))));
    }
    let (_, schema_id) = get_session_schema(session, server, None)?;
    let table = server
        .catalog
        .get_table(schema_id, table_name)
        .map_err(ProtocolError::Database)?;
    if !table.lake.is_lake() {
        return Ok(show_heap_clustering(table_name, &table));
    }
    let log = lake_log_for_clustering(table_name, server, session)?;
    let manifest = log.latest_manifest().map_err(ProtocolError::Database)?;
    let table_id = log.paths().table_id().ok_or_else(|| {
        ProtocolError::Database(ZyronError::Internal(
            "lake root does not name its table".into(),
        ))
    })?;
    let now = zyron_lake::current_epoch();
    let observer = zyron_lake::observer();
    let evidence = zyron_lake::evidence_from_manifest(&manifest, observer, table_id, now);
    let anchors = manifest.clustering_anchors();

    // Byte-weighted mean of the per-column measured skip rates, weighted
    // by how much of the workload touched each column
    let mut fit_weight = 0f64;
    let mut fit_total = 0f64;
    let mut observed_columns = 0usize;
    for column in &evidence {
        let weight = column.total_weight();
        if weight <= 0.0 {
            continue;
        }
        observed_columns += 1;
        if let Some(rate) =
            zyron_lake::measured_skip_rate(observer, table_id, column.column_id, now)
        {
            fit_weight += weight;
            fit_total += weight * rate;
        }
    }
    let measured_fit = if fit_weight > 0.0 {
        Some(fit_total / fit_weight)
    } else {
        None
    };

    let name_of = |column_id: u32| -> String {
        manifest
            .schema
            .column_by_id(column_id)
            .map(|c| c.name.clone())
            .unwrap_or_else(|| format!("column {}", column_id))
    };
    let render_keys = |keys: &[zyron_lake::ClusterKey]| -> String {
        if keys.is_empty() {
            return "(none)".to_string();
        }
        keys.iter()
            .map(|k| format!("{} USING {}", name_of(k.column_id), k.strategy.as_str()))
            .collect::<Vec<_>>()
            .join(", ")
    };

    let proposal = zyron_lake::propose(&evidence, &anchors, 4);
    let mut rows: Vec<Vec<String>> = vec![
        vec!["mode".into(), manifest.clustering_mode().as_str().into()],
        vec![
            "schedule".into(),
            manifest.clustering_schedule().as_str().into(),
        ],
        vec!["keys".into(), render_keys(&manifest.cluster_spec.keys)],
        vec![
            "anchors".into(),
            if anchors.is_empty() {
                "(none)".to_string()
            } else {
                anchors
                    .iter()
                    .map(|id| name_of(*id))
                    .collect::<Vec<_>>()
                    .join(", ")
            },
        ],
        vec!["spec_id".into(), manifest.cluster_spec.spec_id.to_string()],
        vec!["files".into(), manifest.entries.len().to_string()],
        vec![
            "bytes".into(),
            manifest
                .entries
                .iter()
                .map(|e| e.size_bytes)
                .sum::<u64>()
                .to_string(),
        ],
        vec!["observed_columns".into(), observed_columns.to_string()],
        vec![
            "measured_fit".into(),
            match measured_fit {
                Some(fit) => format!("{:.3}", fit),
                None => "(no scans observed)".to_string(),
            },
        ],
        vec!["measurement_would_choose".into(), render_keys(&proposal)],
    ];

    // Warnings, so the operator learns what the numbers imply without
    // having to derive it
    let mut warnings: Vec<String> = Vec::new();
    if observed_columns == 0 {
        warnings.push("no scans have been observed, so measurement has nothing to judge on".into());
    } else if proposal != manifest.cluster_spec.keys {
        warnings.push(match manifest.clustering_mode() {
            zyron_lake::ClusterMode::Force => {
                "measurement would choose different keys, and the pinned choice is kept".into()
            }
            _ => "measurement would choose different keys, a maintenance pass will \
                  propose them"
                .to_string(),
        });
    }
    let dropped = observer.stats().dropped;
    if dropped > 0 {
        warnings.push(format!(
            "{} observations were dropped by a full counter neighbourhood",
            dropped
        ));
    }
    rows.push(vec![
        "warnings".into(),
        if warnings.is_empty() {
            "(none)".to_string()
        } else {
            warnings.join("; ")
        },
    ]);

    Ok(DdlResult::Rows {
        tag: "SHOW".to_string(),
        columns: vec![
            ("property".to_string(), crate::types::PG_TEXT_OID),
            ("value".to_string(), crate::types::PG_TEXT_OID),
        ],
        rows,
    })
}

/// `CREATE PEER <name> ADDRESS '<host:port>' [MODE <mode>]`.
///
/// Declares a node this one may talk to. Peering is stated rather than
/// discovered, so a node never joins a mesh because it happened to see
/// traffic from one.
///
/// The mode is what the operator believes the peer stores. It is recorded
/// as a belief, not a fact: the peer states its own mode when first
/// reached, and that is the authority. Recording it here lets the planner
/// choose pushdown before the first connection rather than after.
async fn handle_create_peer(
    stmt: &zyron_parser::ast::CreatePeerStatement,
    server: &Arc<ServerState>,
) -> Result<DdlResult, ProtocolError> {
    if stmt.address.trim().is_empty() {
        return Err(ProtocolError::Database(ZyronError::ParseError(format!(
            "CREATE PEER {} needs an address to reach it at",
            stmt.name
        ))));
    }
    let mode = match &stmt.mode {
        Some(name) => Some(zyron_common::DeploymentMode::parse(name).ok_or_else(|| {
            ProtocolError::Database(ZyronError::ParseError(format!(
                "unknown peer mode \"{}\", expected db, lake or unified",
                name
            )))
        })?),
        None => None,
    };
    // A node cannot peer with itself. The mesh would then have a cycle of
    // length one and every freshness answer would be about its own data
    if stmt.name == server.node_identity.name {
        return Err(ProtocolError::Database(ZyronError::ConfigError(format!(
            "\"{}\" is this node's own name, a node does not peer with itself",
            stmt.name
        ))));
    }

    let data_dir = server.disk_manager.data_dir().to_path_buf();
    let mut guard = server.peers.write();
    let registry = Arc::make_mut(&mut guard);
    if registry.get(&stmt.name).is_some() {
        if stmt.if_not_exists {
            return Ok(DdlResult::Tag("CREATE PEER".to_string()));
        }
        return Err(ProtocolError::Database(ZyronError::ConfigError(format!(
            "peer \"{}\" already exists, drop it before declaring a new address",
            stmt.name
        ))));
    }
    registry.add(zyron_common::PeerEntry::declared(
        stmt.name.clone(),
        stmt.address.clone(),
        mode,
        zyron_common::peer_timestamp_us(),
    ));
    registry
        .persist(&data_dir)
        .map_err(ProtocolError::Database)?;
    drop(guard);

    // Ask the peer who it is, off this statement's path. A peer that is
    // down would otherwise hold the DDL open for the probe timeout, and a
    // declaration that stalls on an unreachable node is the failure mode
    // the mesh exists to avoid. The statement returns now and the facts
    // fill in when the peer answers
    spawn_peer_contact(server, stmt.name.clone(), stmt.address.clone());
    Ok(DdlResult::Tag("CREATE PEER".to_string()))
}

/// `CREATE FOREIGN TABLE t (cols) SERVER <peer> [TABLE <remote>]`.
///
/// Registers the shape of a table that lives elsewhere. Nothing local is
/// allocated: no heap file, no free-space map, no lake root. Every read of
/// it is a read of the peer, and the catalog entry exists so a plan can be
/// built against it without reaching anything.
///
/// The peer has to be declared first. A foreign table naming a node this
/// one was never told about would be discovery by another name, and peering
/// is stated on purpose.
async fn handle_create_foreign_table(
    stmt: &zyron_parser::ast::CreateForeignTableStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (_, schema_id) = get_session_schema(session, server, None)?;
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Schema,
        schema_id.0,
    )?;

    if server.peers.read().get(&stmt.server).is_none() {
        return Err(ProtocolError::Database(ZyronError::ConfigError(format!(
            "no peer named \"{}\". Declare it with CREATE PEER first, because a node \
             is peered on purpose rather than by being named",
            stmt.server
        ))));
    }
    // An unnamed remote is the local name. Recording it resolved keeps the
    // request builder free of a fallback that would have to re-derive it
    let remote = stmt.remote_table.as_deref().unwrap_or(&stmt.name);

    match server
        .catalog
        .create_foreign_table(schema_id, &stmt.name, &stmt.columns, &stmt.server, remote)
        .await
    {
        Ok(_) => {
            fire_event(
                server,
                zyron_pipeline::event_handler::EventType::TableCreated,
                &stmt.name,
                &[("table".to_string(), stmt.name.clone())],
            )
            .await;
            Ok(DdlResult::Tag("CREATE FOREIGN TABLE".to_string()))
        }
        Err(ZyronError::TableAlreadyExists(_)) if stmt.if_not_exists => {
            Ok(DdlResult::Tag("CREATE FOREIGN TABLE".to_string()))
        }
        Err(e) => Err(ProtocolError::Database(e)),
    }
}

/// `DROP FOREIGN TABLE [IF EXISTS] t`.
///
/// Removes the local declaration and nothing else. The rows are the peer's
/// and stay where they are, which is why this frees no files and touches no
/// lake root.
///
/// A local table named here is refused rather than dropped: the two
/// statements mean different things, and one that silently did the other
/// would delete data on the strength of a typo.
async fn handle_drop_foreign_table(
    stmt: &zyron_parser::ast::DropForeignTableStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (_, schema_id) = get_session_schema(session, server, None)?;
    let table = match server.catalog.get_table(schema_id, &stmt.name) {
        Ok(t) => t,
        Err(_) if stmt.if_exists => return Ok(DdlResult::Tag("DROP FOREIGN TABLE".to_string())),
        Err(e) => return Err(ProtocolError::Database(e)),
    };
    if !table.foreign.is_foreign() {
        return Err(ProtocolError::Database(ZyronError::ConfigError(format!(
            "\"{}\" is a local table. DROP FOREIGN TABLE removes a declaration of \
             someone else's table, use DROP TABLE to remove this one",
            stmt.name
        ))));
    }
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Table,
        table.id.0,
    )?;
    server
        .catalog
        .drop_table(schema_id, &stmt.name)
        .await
        .map_err(ProtocolError::Database)?;
    Ok(DdlResult::Tag("DROP FOREIGN TABLE".to_string()))
}

/// `ALTER TABLE t FOLLOW <peer>.<table>` and `ALTER TABLE t UNFOLLOW`.
///
/// A follower replays a leader's log instead of accepting writes, so this
/// is what turns an ordinary lake table into a replica. The peer has to be
/// declared first: following a node this one was never told about would be
/// discovery by another name, and peering is stated on purpose.
///
/// UNFOLLOW leaves the table as its own authority holding everything it
/// applied. It does not roll anything back, because the rows are real and
/// the operator asked to stop following, not to forget.
async fn handle_alter_table_follow(
    stmt: &zyron_parser::ast::AlterTableFollowStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (_, schema_id) = get_session_schema(session, server, None)?;
    let table = server
        .catalog
        .get_table(schema_id, &stmt.table)
        .map_err(ProtocolError::Database)?;
    if !table.lake.is_lake() {
        return Err(ProtocolError::Database(ZyronError::ConfigError(format!(
            "\"{}\" is a heap table. Following replays a lake table's log, which a \
             heap table does not have",
            stmt.table
        ))));
    }
    let mut updated = (*table).clone();
    match &stmt.leader {
        Some((peer, remote)) => {
            if server.peers.read().get(peer).is_none() {
                return Err(ProtocolError::Database(ZyronError::ConfigError(format!(
                    "no peer named \"{}\". Declare it with CREATE PEER first, because \
                     a node is peered on purpose rather than by being named",
                    peer
                ))));
            }
            // A table that already holds versions of its own would have two
            // sources of truth, and replay assumes one
            let paths = zyron_lake::LakePaths::new(server.disk_manager.data_dir(), updated.id.0);
            if let Some(log) = zyron_lake::TransactionLog::lookup_shared(&paths) {
                let manifest = log.latest_manifest().map_err(ProtocolError::Database)?;
                if !manifest.entries.is_empty() && zyron_lake::load_cursor(&paths).is_none() {
                    return Err(ProtocolError::Database(ZyronError::ConfigError(format!(
                        "\"{}\" already holds its own data. A follower replays a leader \
                         rather than merging with it, so follow an empty table",
                        stmt.table
                    ))));
                }
            }
            updated.lake.follow(peer, remote);
        }
        None => updated.lake.unfollow(),
    }
    server
        .catalog
        .update_table(updated)
        .await
        .map_err(ProtocolError::Database)?;
    Ok(DdlResult::Tag("ALTER TABLE".to_string()))
}

/// Starts a peer contact in the background.
///
/// Contact is never on a statement's path. A probe against a node that is
/// down takes as long as its timeout, and an operator declaring a peer
/// should wait on their own node rather than on someone else's
pub fn spawn_peer_contact(server: &Arc<ServerState>, name: String, address: String) {
    let server = Arc::clone(server);
    tokio::spawn(async move {
        contact_peer(&server, &name, &address).await;
    });
}

/// Reaches a peer and records what it said about itself.
///
/// What the peer reports replaces what was declared, because an operator
/// can be wrong about a node's mode and the node cannot. A failure records
/// why and keeps whatever was learned before: a peer that is unreachable
/// now is still the peer it was, and forgetting its id would make a
/// transient outage look like a different node.
pub async fn contact_peer(server: &Arc<ServerState>, name: &str, address: &str) {
    let user = server.node_identity.name.clone();
    let outcome = crate::peer_probe::probe_peer(address, &user, "zyron").await;
    let data_dir = server.disk_manager.data_dir().to_path_buf();
    let mut guard = server.peers.write();
    let registry = Arc::make_mut(&mut guard);
    let Some(peer) = registry.get_mut(name) else {
        return;
    };
    match outcome {
        Ok(facts) => {
            if facts.node_id == server.node_identity.node_id {
                peer.unreachable(format!(
                    "\"{}\" answered with this node's own id, so it is this node \
                     reached by another address rather than a peer",
                    address
                ));
            } else {
                peer.observed(facts.node_id, facts.mode, zyron_common::peer_timestamp_us());
            }
        }
        Err(e) => peer.unreachable(e.to_string()),
    }
    if let Err(e) = registry.persist(&data_dir) {
        tracing::warn!(peer = %name, error = %e, "recording peer contact failed");
    }
}

/// `DROP PEER [IF EXISTS] <name>`.
async fn handle_drop_peer(
    stmt: &zyron_parser::ast::DropPeerStatement,
    server: &Arc<ServerState>,
) -> Result<DdlResult, ProtocolError> {
    let data_dir = server.disk_manager.data_dir().to_path_buf();
    let mut guard = server.peers.write();
    let registry = Arc::make_mut(&mut guard);
    if !registry.remove(&stmt.name) {
        if stmt.if_exists {
            return Ok(DdlResult::Tag("DROP PEER".to_string()));
        }
        return Err(ProtocolError::Database(ZyronError::ConfigError(format!(
            "peer \"{}\" does not exist",
            stmt.name
        ))));
    }
    registry
        .persist(&data_dir)
        .map_err(ProtocolError::Database)?;
    Ok(DdlResult::Tag("DROP PEER".to_string()))
}

/// `SHOW CLUSTERING FOR t` on a heap table.
///
/// A heap table's layout is decided at fold time and its policy is catalog
/// state, so there is no manifest to measure against and no per-file
/// statistics to score. It reports the policy and the segments that have
/// reached it, which is what the operator can act on
fn show_heap_clustering(table_name: &str, table: &zyron_catalog::schema::TableEntry) -> DdlResult {
    let name_of = |column_id: u32| -> String {
        table
            .columns
            .iter()
            .find(|c| c.id.0 as u32 == column_id)
            .map(|c| c.name.clone())
            .unwrap_or_else(|| format!("column {}", column_id))
    };
    let keys = table.cluster.fold_keys();
    let rendered = if keys.is_empty() {
        "(none)".to_string()
    } else {
        keys.iter()
            .map(|k| format!("{} USING {}", name_of(k.column_id), k.strategy.as_str()))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let spec_id = table.cluster.spec_id;
    let total = table.columnar.segments.len();
    let current = table
        .columnar
        .segments
        .iter()
        .filter(|s| s.cluster_spec_id == spec_id)
        .count();
    let warnings = if keys.is_empty() {
        "no cluster keys declared, folds order by the primary key".to_string()
    } else if current < total {
        format!(
            "{} of {} segments predate the current spec, later folds carry them over",
            total - current,
            total
        )
    } else {
        "(none)".to_string()
    };
    DdlResult::Rows {
        tag: "SHOW".to_string(),
        columns: vec![
            ("property".to_string(), crate::types::PG_TEXT_OID),
            ("value".to_string(), crate::types::PG_TEXT_OID),
        ],
        rows: vec![
            vec!["format".into(), "HEAP".into()],
            vec!["table".into(), table_name.to_string()],
            vec!["mode".into(), table.cluster.mode().as_str().into()],
            vec!["schedule".into(), table.cluster.schedule().as_str().into()],
            vec!["keys".into(), rendered],
            vec!["spec_id".into(), spec_id.to_string()],
            vec!["segments".into(), total.to_string()],
            vec!["segments_at_spec".into(), current.to_string()],
            vec!["warnings".into(), warnings],
        ],
    }
}

/// `ALTER TABLE t SET CLUSTERING SCHEDULE = OnDemand | Incremental | Continuous`
async fn handle_clustering_schedule(
    stmt: &zyron_parser::ast::AlterTableClusteringScheduleStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (_, schema_id) = get_session_schema(session, server, None)?;
    let table = server
        .catalog
        .get_table(schema_id, &stmt.table)
        .map_err(ProtocolError::Database)?;
    if !table.lake.is_lake() {
        set_heap_cluster_policy(server, &table, None, None, Some(stmt.schedule)).await?;
        return Ok(DdlResult::Tag("ALTER TABLE".to_string()));
    }
    let log = lake_log_for_clustering(&stmt.table, server, session)?;
    let schedule = stmt.schedule.as_str().to_ascii_lowercase();
    log.commit(clustering_commit_attempt(), |_| {
        Ok(vec![zyron_lake::LogEntry::SetProperty {
            key: zyron_lake::CLUSTERING_SCHEDULE_PROPERTY.to_string(),
            value: schedule.clone(),
        }])
    })
    .map_err(ProtocolError::Database)?;
    Ok(DdlResult::Tag("ALTER TABLE".to_string()))
}

/// Applies a `time_travel_retention` WITH-option to a freshly created table.
/// Absent or default leaves the table at the aggressive default (0). A finite
/// or unlimited window is persisted and turns on commit-LSN tracking.
async fn apply_create_table_retention(
    server: &Arc<ServerState>,
    schema_id: zyron_catalog::SchemaId,
    name: &str,
    options: &[zyron_parser::ast::TableOption],
) -> Result<(), ProtocolError> {
    let mut secs: Option<u64> = None;
    for opt in options {
        if opt.key.eq_ignore_ascii_case("time_travel_retention")
            || opt.key.eq_ignore_ascii_case("time_travel_retention_period")
        {
            let v = match &opt.value {
                zyron_parser::ast::TableOptionValue::String(s)
                | zyron_parser::ast::TableOptionValue::Identifier(s) => s.clone(),
                zyron_parser::ast::TableOptionValue::Integer(i) => i.to_string(),
                other => {
                    return Err(ProtocolError::Database(ZyronError::ParseError(format!(
                        "invalid time_travel_retention value {other:?}"
                    ))));
                }
            };
            secs = Some(crate::lifecycle_dispatch::parse_time_travel_retention(&v)?);
        }
    }
    let Some(secs) = secs else {
        return Ok(());
    };
    if secs == 0 {
        return Ok(());
    }
    let table = server
        .catalog
        .get_table(schema_id, name)
        .map_err(ProtocolError::Database)?;
    let mut entry = (*table).clone();
    entry.time_travel_retention_secs = secs;
    server
        .catalog
        .update_table(entry)
        .await
        .map_err(ProtocolError::Database)?;
    server.txn_manager.status_map().enable_lsn_tracking();
    Ok(())
}

async fn handle_drop_table(
    stmt: &zyron_parser::ast::DropTableStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (_, schema_id) = get_session_schema(session, server, None)?;

    // Check DROP privilege on the table if it exists. If the table does not
    // exist and IF EXISTS is set, skip the privilege check entirely. The
    // columnar registry is captured here because the catalog entry is gone
    // once the drop commits
    let mut columnar_segments: Vec<zyron_catalog::schema::ColumnarSegmentEntry> = Vec::new();
    let mut columnar_table_id: u64 = 0;
    let mut lake_table_id: Option<u32> = None;
    // Table and index ids captured before the drop so their IO counters can be
    // discarded with them. A soft drop keeps both, because UNDROP restores the
    // table under the same id and its history is still its own
    let mut dropped_table_id: Option<u32> = None;
    let mut dropped_index_ids: Vec<u32> = Vec::new();
    if let Ok(table) = server.catalog.get_table(schema_id, &stmt.name) {
        check_ddl_privilege(
            server,
            session,
            zyron_auth::PrivilegeType::Create,
            zyron_auth::ObjectType::Table,
            table.id.0,
        )?;
        columnar_segments = table.columnar.segments.clone();
        columnar_table_id = table.id.0 as u64;
        lake_table_id = table.lake.is_lake().then_some(table.id.0);
        dropped_table_id = Some(table.id.0);
        dropped_index_ids = server
            .catalog
            .get_indexes_for_table(table.id)
            .iter()
            .map(|idx| idx.id.0)
            .collect();
    }

    match server.catalog.drop_table(schema_id, &stmt.name).await {
        Ok(outcome) => {
            if !outcome.soft_dropped {
                if let Some(id) = dropped_table_id {
                    server.table_io_stats.remove(id);
                }
                for idx_id in &dropped_index_ids {
                    server.index_io_stats.remove(*idx_id);
                }
            }
            // A hard drop removed the catalog entry, so reclaim the backing
            // heap and FSM files now. A soft drop keeps them for UNDROP, the
            // reaper reclaims them after the recycle window elapses. File id
            // zero is the reserved "no file" value a foreign table carries,
            // and there is nothing local to reclaim for one
            if !outcome.soft_dropped && outcome.heap_file_id != 0 {
                let _ = server.heap_files.remove_async(&outcome.heap_file_id).await;
                // A failed file delete after the catalog entry is gone leaks the
                // backing files. Surface it to the caller rather than swallowing
                // it so the leak is not silent.
                if let Err(e) = server.disk_manager.delete_file(outcome.heap_file_id).await {
                    tracing::error!(
                        target: "zyron::ddl",
                        heap_file_id = outcome.heap_file_id,
                        "DROP TABLE failed to remove heap file: {e}"
                    );
                    return Err(ProtocolError::Database(e));
                }
                if let Err(e) = server.disk_manager.delete_file(outcome.fsm_file_id).await {
                    tracing::error!(
                        target: "zyron::ddl",
                        fsm_file_id = outcome.fsm_file_id,
                        "DROP TABLE failed to remove FSM file: {e}"
                    );
                    return Err(ProtocolError::Database(e));
                }
                // Reclaim the lake tier: the shared log handle and the whole
                // table root, log, checkpoints and data files. The catalog
                // entry is already gone so nothing can re-register them. A
                // soft drop keeps everything for UNDROP
                if let Some(id) = lake_table_id {
                    let paths = zyron_lake::LakePaths::new(server.disk_manager.data_dir(), id);
                    zyron_lake::TransactionLog::remove_shared(&paths);
                    if let Err(e) = std::fs::remove_dir_all(paths.root()) {
                        if e.kind() != std::io::ErrorKind::NotFound {
                            tracing::error!(
                                target: "zyron::ddl",
                                table_id = id,
                                "DROP TABLE failed to remove the lake root: {e}"
                            );
                            return Err(ProtocolError::Database(e.into()));
                        }
                    }
                }
                // Reclaim the columnar tier: .zyr segments, RID sidecars and
                // the patch store. The catalog entry is already gone, so
                // recovery cannot re-register these files, no WAL record is
                // needed. A soft drop keeps them for UNDROP
                if !columnar_segments.is_empty() {
                    let columnar_dir = std::path::Path::new(&columnar_segments[0].path)
                        .parent()
                        .map(|d| d.to_path_buf());
                    let store = zyron_storage::columnar::ColumnarPatchManager::store_for_segment(
                        columnar_table_id,
                        std::path::Path::new(&columnar_segments[0].path),
                    )
                    .map_err(ProtocolError::Database)?;
                    let patch_path = columnar_dir
                        .as_ref()
                        .map(|d| d.join(format!("{}.zyrpatch", columnar_table_id)));
                    for seg in &columnar_segments {
                        let seg_path = std::path::Path::new(&seg.path);
                        if let Err(e) = std::fs::remove_file(seg_path) {
                            if e.kind() != std::io::ErrorKind::NotFound {
                                tracing::error!(
                                    target: "zyron::ddl",
                                    segment = %seg.path,
                                    "DROP TABLE failed to remove columnar segment: {e}"
                                );
                                return Err(ProtocolError::Database(ZyronError::IoError(format!(
                                    "DROP TABLE failed to remove columnar segment {}: {e}",
                                    seg.path
                                ))));
                            }
                        }
                        let rids = seg_path.with_extension("zyrrids");
                        if let Err(e) = std::fs::remove_file(&rids) {
                            if e.kind() != std::io::ErrorKind::NotFound {
                                tracing::warn!(
                                    target: "zyron::ddl",
                                    segment = %seg.path,
                                    "DROP TABLE failed to remove RID sidecar: {e}"
                                );
                            }
                        }
                        if let Some(pp) = &patch_path {
                            store
                                .drop_file(seg.file_id, pp)
                                .map_err(ProtocolError::Database)?;
                        }
                    }
                    if let (Some(dir), Some(pp)) = (&columnar_dir, &patch_path) {
                        let mgr = zyron_storage::columnar::ColumnarPatchManager::global(dir);
                        if let Err(e) = mgr.remove_store(columnar_table_id, pp) {
                            tracing::warn!(
                                target: "zyron::ddl",
                                table_id = columnar_table_id,
                                "DROP TABLE left an empty patch file behind: {e}"
                            );
                        }
                    }
                }
            }
            fire_event(
                server,
                zyron_pipeline::event_handler::EventType::TableDropped,
                &stmt.name,
                &[("table".to_string(), stmt.name.clone())],
            )
            .await;
            Ok(DdlResult::Tag("DROP TABLE".to_string()))
        }
        Err(ZyronError::TableNotFound(_)) if stmt.if_exists => {
            Ok(DdlResult::Tag("DROP TABLE".to_string()))
        }
        Err(e) => Err(ProtocolError::Database(e)),
    }
}

async fn handle_truncate(
    stmt: &zyron_parser::ast::TruncateStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (_, schema_id) = get_session_schema(session, server, None)?;

    // Verify table exists
    let table = server
        .catalog
        .get_table(schema_id, &stmt.table)
        .map_err(ProtocolError::Database)?;

    // Check TRUNCATE privilege on the table
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Truncate,
        zyron_auth::ObjectType::Table,
        table.id.0,
    )?;

    // Capture the page counts before truncation so every stale pool frame
    // can be dropped afterwards, a cached frame or heap handle would keep
    // serving truncated rows.
    let heap_pages = server
        .disk_manager
        .num_pages(table.heap_file_id)
        .await
        .map_err(ProtocolError::Database)?;
    let fsm_pages = server
        .disk_manager
        .num_pages(table.fsm_file_id)
        .await
        .map_err(ProtocolError::Database)?;

    // Truncate the heap data file and its FSM file to zero pages.
    // This removes all row data while preserving table metadata in the catalog.
    server
        .disk_manager
        .truncate_file(table.heap_file_id)
        .await
        .map_err(|e| ProtocolError::Database(e))?;

    server
        .disk_manager
        .truncate_file(table.fsm_file_id)
        .await
        .map_err(|e| ProtocolError::Database(e))?;

    // Drop the cached heap handle (its page counters are stale) and evict
    // both files' frames from the buffer pool so scans re-open the file at
    // zero pages instead of resurrecting rows from cache.
    let _ = server.heap_files.remove_async(&table.heap_file_id).await;
    for p in 0..heap_pages {
        server
            .buffer_pool
            .delete_page(zyron_common::PageId::new(table.heap_file_id, p));
    }
    for p in 0..fsm_pages {
        server
            .buffer_pool
            .delete_page(zyron_common::PageId::new(table.fsm_file_id, p));
    }

    // Clear every B+tree index on the table so it does not point at rows that
    // no longer exist. Each index is replaced with a fresh empty tree in the
    // registry and its on-disk checkpoint removed so recovery does not reload
    // stale entries.
    let checkpoint_dir = server.data_dir.join("indexes");
    for index in server.catalog.get_indexes_for_table(table.id) {
        if index.index_type != zyron_catalog::IndexType::BTree {
            continue;
        }
        let empty = zyron_storage::BTreeIndex::create(index.index_file_id, checkpoint_dir.clone())
            .await
            .map_err(ProtocolError::Database)?;
        let _ = server
            .btree_indexes
            .insert_async(index.id.0, Arc::new(empty))
            .await;
        let checkpoint_path = checkpoint_dir.join(format!("index_{}.zyridx", index.index_file_id));
        if checkpoint_path.exists() {
            if let Err(e) = std::fs::remove_file(&checkpoint_path) {
                tracing::error!(
                    target: "zyron::ddl",
                    index = %index.name,
                    "TRUNCATE failed to remove index checkpoint: {e}"
                );
            }
        }
    }

    // Reclaim the columnar tier: folded rows are truncated with the heap
    // rows, otherwise they resurrect on the next scan. Each segment removal
    // is WAL-logged with the whole-segment-died merge record pair so crash
    // recovery suppresses the fold record that would re-register the file
    if !table.columnar.segments.is_empty() {
        let store = zyron_storage::columnar::ColumnarPatchManager::store_for_segment(
            table.id.0 as u64,
            std::path::Path::new(&table.columnar.segments[0].path),
        )
        .map_err(ProtocolError::Database)?;
        let patch_path = std::path::Path::new(&table.columnar.segments[0].path)
            .parent()
            .map(|d| d.join(format!("{}.zyrpatch", table.id.0)));
        for seg in &table.columnar.segments {
            let mut bp = Vec::new();
            bp.extend_from_slice(&(table.id.0 as u64).to_le_bytes());
            bp.extend_from_slice(seg.path.as_bytes());
            server
                .wal
                .log_merge_begin(&bp)
                .map_err(ProtocolError::Database)?;
            let mut ep = Vec::new();
            ep.extend_from_slice(&(table.id.0 as u64).to_le_bytes());
            ep.extend_from_slice(&seg.file_id.to_le_bytes());
            server
                .wal
                .log_merge_end(&ep)
                .map_err(ProtocolError::Database)?;
        }
        server.wal.flush().map_err(ProtocolError::Database)?;
        let mut entry = (*table).clone();
        entry.columnar.segments.clear();
        server
            .catalog
            .update_table(entry)
            .await
            .map_err(ProtocolError::Database)?;
        for seg in &table.columnar.segments {
            let seg_path = std::path::Path::new(&seg.path);
            if let Err(e) = std::fs::remove_file(seg_path) {
                if e.kind() != std::io::ErrorKind::NotFound {
                    tracing::error!(
                        target: "zyron::ddl",
                        segment = %seg.path,
                        "TRUNCATE failed to remove columnar segment: {e}"
                    );
                    return Err(ProtocolError::Database(ZyronError::IoError(format!(
                        "TRUNCATE failed to remove columnar segment {}: {e}",
                        seg.path
                    ))));
                }
            }
            let rids = seg_path.with_extension("zyrrids");
            if let Err(e) = std::fs::remove_file(&rids) {
                if e.kind() != std::io::ErrorKind::NotFound {
                    tracing::warn!(
                        target: "zyron::ddl",
                        segment = %seg.path,
                        "TRUNCATE failed to remove RID sidecar: {e}"
                    );
                }
            }
            if let Some(pp) = &patch_path {
                store
                    .drop_file(seg.file_id, pp)
                    .map_err(ProtocolError::Database)?;
            }
        }
    }

    Ok(DdlResult::Tag("TRUNCATE TABLE".to_string()))
}

async fn handle_create_index(
    stmt: &zyron_parser::ast::CreateIndexStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (_, schema_id) = get_session_schema(session, server, None)?;

    let table = server
        .catalog
        .get_table(schema_id, &stmt.table)
        .map_err(ProtocolError::Database)?;

    // Check CREATE privilege on the schema for index creation
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Schema,
        schema_id.0,
    )?;

    // Each key column carries its declared sort direction. `asc: None` is
    // the unwritten default, which is ascending
    let mut key_columns: Vec<(String, bool)> = Vec::with_capacity(stmt.columns.len());
    for c in &stmt.columns {
        match &c.expr {
            zyron_parser::ast::Expr::Identifier(name) => {
                key_columns.push((name.clone(), c.asc == Some(false)));
            }
            other => {
                return Err(ProtocolError::Database(ZyronError::PlanError(format!(
                    "expression indexes are not supported, use column names (got: {:?})",
                    other
                ))));
            }
        }
    }
    // A key column's stored bytes run in one direction for the whole key, so
    // an index can be declared entirely ascending or entirely descending but
    // not both. A uniform declaration is served either way by walking the
    // index forward or backward, which is why both spellings are accepted.
    // Mixed directions would have to be flattened to one of them, and an
    // index that silently sorts differently from its own declaration is worse
    // than one the statement refuses to create
    if key_columns.iter().any(|(_, d)| *d) && key_columns.iter().any(|(_, d)| !*d) {
        return Err(ProtocolError::Database(ZyronError::PlanError(format!(
            "index '{}' mixes ASC and DESC key columns, which one index cannot store, declare every column in the same direction or build one index per ordering",
            stmt.name
        ))));
    }
    let column_names: Vec<String> = key_columns.iter().map(|(n, _)| n.clone()).collect();

    match server
        .catalog
        .create_btree_index(table.id, schema_id, &stmt.name, &key_columns, stmt.unique)
        .await
    {
        Ok(index_id) => {
            // A lake table's index is a lake artifact committed into its
            // own transaction log, not a B+tree over heap addresses. It is
            // versioned with the data, survives the rewrites clustering and
            // compaction perform, and is readable at a past version
            if table.lake.is_lake() {
                let result = crate::index_build::build_lake_index(
                    server,
                    &table,
                    &column_names,
                    stmt.unique,
                )
                .await;
                if let Err(e) = result {
                    // The catalog entry would otherwise describe an index
                    // the table does not have
                    let _ = server.catalog.drop_index(table.id, &stmt.name).await;
                    let _ = index_id;
                    return Err(ProtocolError::Database(e));
                }
                fire_event(
                    server,
                    zyron_pipeline::event_handler::EventType::IndexCreated,
                    &stmt.name,
                    &[
                        ("index".to_string(), stmt.name.clone()),
                        ("table".to_string(), stmt.table.clone()),
                    ],
                )
                .await;
                return Ok(DdlResult::Tag("CREATE INDEX".to_string()));
            }
            let checkpoint_dir = server.data_dir.join("indexes");
            let _ = std::fs::create_dir_all(&checkpoint_dir);
            let entry = server
                .catalog
                .get_indexes_for_table(table.id)
                .into_iter()
                .find(|e| e.id == index_id)
                .ok_or_else(|| {
                    ProtocolError::Database(ZyronError::Internal(format!(
                        "newly created index {} not found in catalog",
                        index_id.0
                    )))
                })?;
            let btree = Arc::new(
                zyron_storage::BTreeIndex::create(entry.index_file_id, checkpoint_dir)
                    .await
                    .map_err(ProtocolError::Database)?,
            );
            // Fill the tree from the rows the table already holds. Without
            // this the index is empty, and every query the planner routes
            // through it returns nothing for rows that predate it, which is a
            // wrong answer rather than a slow one
            let key_columns: Vec<zyron_catalog::ColumnId> =
                entry.columns.iter().map(|c| c.column_id).collect();
            if !key_columns.is_empty() {
                let rows = crate::index_build::collect_live_rows(server, &table)
                    .await
                    .map_err(ProtocolError::Database)?;
                let entries = crate::index_build::fill_btree_from_live_rows(
                    &table,
                    &rows,
                    &key_columns,
                    &btree,
                );
                if entries > 0 {
                    tracing::info!(
                        target: "zyron::ddl",
                        index = %stmt.name,
                        entries,
                        "CREATE INDEX populated from existing rows"
                    );
                }
            }
            let _ = server.btree_indexes.insert_async(index_id.0, btree).await;
            fire_event(
                server,
                zyron_pipeline::event_handler::EventType::IndexCreated,
                &stmt.name,
                &[
                    ("index".to_string(), stmt.name.clone()),
                    ("table".to_string(), stmt.table.clone()),
                ],
            )
            .await;
            Ok(DdlResult::Tag("CREATE INDEX".to_string()))
        }
        Err(ZyronError::IndexAlreadyExists(_)) => {
            // CreateIndexStatement does not have if_not_exists, treat as error
            Err(ProtocolError::Database(ZyronError::IndexAlreadyExists(
                stmt.name.clone(),
            )))
        }
        Err(e) => Err(ProtocolError::Database(e)),
    }
}

async fn handle_drop_index(
    stmt: &zyron_parser::ast::DropIndexStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (_, schema_id) = get_session_schema(session, server, None)?;

    // Find the table that owns this index by scanning all tables in the
    // schema. Index names are unique within a schema, so the first match
    // is the owning table.
    let tables = server.catalog.list_tables(schema_id);
    let mut found_table_id = None;
    for table in &tables {
        let indexes = server.catalog.get_indexes_for_table(table.id);
        if indexes.iter().any(|idx| idx.name == stmt.name) {
            found_table_id = Some(table.id);
            break;
        }
    }

    match found_table_id {
        Some(table_id) => {
            // Privilege: dedicated DropIndex on the table.
            check_ddl_privilege(
                server,
                session,
                zyron_auth::PrivilegeType::DropIndex,
                zyron_auth::ObjectType::Table,
                table_id.0,
            )?;

            // Identify index type before dropping so we can clean up the right manager.
            let indexes = server.catalog.get_indexes_for_table(table_id);
            let matched = indexes.iter().find(|idx| idx.name == stmt.name);
            let fts_index_id = matched
                .filter(|idx| idx.index_type == zyron_catalog::IndexType::Fulltext)
                .map(|idx| idx.id.0);
            let vec_index_id = matched
                .filter(|idx| idx.index_type == zyron_catalog::IndexType::Vector)
                .map(|idx| idx.id.0);
            let spatial_index_id = matched
                .filter(|idx| idx.index_type == zyron_catalog::IndexType::Spatial)
                .map(|idx| idx.id.0);
            let dropped_index_id = matched.map(|idx| idx.id.0);
            // A lake index lives in the table's own log, so dropping the
            // catalog entry alone would leave its files referenced forever
            let lake_columns: Option<Vec<String>> = matched
                .filter(|idx| idx.index_type == zyron_catalog::IndexType::BTree)
                .and_then(|idx| {
                    let table = server.catalog.get_table_by_id(table_id).ok()?;
                    if !table.lake.is_lake() {
                        return None;
                    }
                    Some(
                        idx.columns
                            .iter()
                            .filter_map(|c| {
                                table
                                    .columns
                                    .iter()
                                    .find(|tc| tc.id == c.column_id)
                                    .map(|tc| tc.name.clone())
                            })
                            .collect(),
                    )
                });

            match server.catalog.drop_index(table_id, &stmt.name).await {
                Ok(()) => {
                    if let Some(columns) = lake_columns
                        && let Ok(table) = server.catalog.get_table_by_id(table_id)
                    {
                        crate::index_build::drop_lake_index(server, &table, &columns)
                            .await
                            .map_err(ProtocolError::Database)?;
                    }
                    // Discard the index counters with the index, so a later
                    // index reusing the id does not inherit its scan history
                    if let Some(id) = dropped_index_id {
                        server.index_io_stats.remove(id);
                    }
                    if let (Some(id), Some(fts_mgr)) = (fts_index_id, &server.fts_manager) {
                        let _ = fts_mgr.drop_index(id);
                    }
                    if let (Some(id), Some(vec_mgr)) = (vec_index_id, &server.vector_manager) {
                        let _ = vec_mgr.drop_index(id);
                    }
                    if let (Some(id), Some(spatial_mgr)) =
                        (spatial_index_id, &server.spatial_manager)
                    {
                        spatial_mgr.drop_index(id);
                    }
                    fire_event(
                        server,
                        zyron_pipeline::event_handler::EventType::IndexDropped,
                        &stmt.name,
                        &[("index".to_string(), stmt.name.clone())],
                    )
                    .await;
                    Ok(DdlResult::Tag("DROP INDEX".to_string()))
                }
                Err(e) => Err(ProtocolError::Database(e)),
            }
        }
        None if stmt.if_exists => Ok(DdlResult::Tag("DROP INDEX".to_string())),
        None => Err(ProtocolError::Database(ZyronError::IndexNotFound(
            stmt.name.clone(),
        ))),
    }
}

/// ALTER INDEX name RENAME TO new_name. The index is located by name across the
/// schema's tables (index names are unique within a schema), then the catalog
/// renames it. The index id and backing file are unchanged, so no live index
/// handle is rebuilt.
async fn handle_alter_index(
    stmt: &zyron_parser::ast::AlterIndexStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    use zyron_parser::ast::AlterIndexOperation as Op;

    let (_, schema_id) = get_session_schema(session, server, None)?;

    let tables = server.catalog.list_tables(schema_id);
    let mut owning_table = None;
    for table in &tables {
        if server
            .catalog
            .get_indexes_for_table(table.id)
            .iter()
            .any(|idx| idx.name == stmt.name)
        {
            owning_table = Some(table.id);
            break;
        }
    }
    let table_id = owning_table
        .ok_or_else(|| ProtocolError::Database(ZyronError::IndexNotFound(stmt.name.clone())))?;

    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Table,
        table_id.0,
    )?;

    match &stmt.operation {
        Op::Rename { new_name } => {
            server
                .catalog
                .rename_index(table_id, &stmt.name, new_name)
                .await
                .map_err(ProtocolError::Database)?;
        }
    }
    Ok(DdlResult::Tag("ALTER INDEX".to_string()))
}

async fn handle_create_schema(
    stmt: &zyron_parser::ast::CreateSchemaStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let db_id = get_session_database(session)?;

    // Check CREATE privilege on the database for schema creation
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Database,
        db_id.0,
    )?;

    match server
        .catalog
        .create_schema(db_id, &stmt.name, "zyron")
        .await
    {
        Ok(_) => {
            fire_event(
                server,
                zyron_pipeline::event_handler::EventType::SchemaChanged,
                &stmt.name,
                &[
                    ("schema".to_string(), stmt.name.clone()),
                    ("operation".to_string(), "create".to_string()),
                ],
            )
            .await;
            Ok(DdlResult::Tag("CREATE SCHEMA".to_string()))
        }
        Err(ZyronError::SchemaAlreadyExists(_)) if stmt.if_not_exists => {
            Ok(DdlResult::Tag("CREATE SCHEMA".to_string()))
        }
        Err(e) => Err(ProtocolError::Database(e)),
    }
}

async fn handle_drop_schema(
    stmt: &zyron_parser::ast::DropSchemaStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let db_id = get_session_database(session)?;

    // Check CREATE privilege on the database (schema owners can drop their schemas)
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Database,
        db_id.0,
    )?;

    match server.catalog.drop_schema(db_id, &stmt.name).await {
        Ok(()) => {
            fire_event(
                server,
                zyron_pipeline::event_handler::EventType::SchemaChanged,
                &stmt.name,
                &[
                    ("schema".to_string(), stmt.name.clone()),
                    ("operation".to_string(), "drop".to_string()),
                ],
            )
            .await;
            Ok(DdlResult::Tag("DROP SCHEMA".to_string()))
        }
        Err(ZyronError::SchemaNotFound(_)) if stmt.if_exists => {
            Ok(DdlResult::Tag("DROP SCHEMA".to_string()))
        }
        Err(e) => Err(ProtocolError::Database(e)),
    }
}

// ---------------------------------------------------------------------------
// Sequence handlers
// ---------------------------------------------------------------------------

/// Resolves a possibly schema-qualified object name to (schema_id, bare name).
/// `schema.name` resolves the named schema; a bare name uses the session's
/// default schema.
fn resolve_qualified_name(
    name: &str,
    server: &Arc<ServerState>,
    session: &Option<Session>,
) -> Result<(zyron_catalog::SchemaId, String), ProtocolError> {
    if let Some((schema_part, obj_part)) = name.split_once('.') {
        let db_id = get_session_database(session)?;
        let schema = server
            .catalog
            .get_schema(db_id, schema_part)
            .map_err(ProtocolError::Database)?;
        Ok((schema.id, obj_part.to_string()))
    } else {
        let (_, schema_id) = get_session_schema(session, server, None)?;
        Ok((schema_id, name.to_string()))
    }
}

/// Default minimum, maximum, and start for a sequence given its increment.
/// Ascending sequences run from 1 to i64::MAX starting at 1; descending run
/// from i64::MIN to -1 starting at -1.
fn sequence_defaults(increment: i64) -> (i64, i64, i64) {
    if increment < 0 {
        (i64::MIN, -1, -1)
    } else {
        (1, i64::MAX, 1)
    }
}

async fn handle_create_sequence(
    stmt: &zyron_parser::ast::CreateSequenceStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (schema_id, name) = resolve_qualified_name(&stmt.name, server, session)?;

    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Schema,
        schema_id.0,
    )?;

    if server.catalog.get_sequence(schema_id, &name).is_some() {
        if stmt.if_not_exists {
            return Ok(DdlResult::Tag("CREATE SEQUENCE".to_string()));
        }
        return Err(ProtocolError::Database(ZyronError::Internal(format!(
            "sequence '{name}' already exists"
        ))));
    }

    let increment = stmt.increment.unwrap_or(1);
    if increment == 0 {
        return Err(ProtocolError::Database(ZyronError::Internal(
            "sequence increment must not be zero".to_string(),
        )));
    }
    let (def_min, def_max, def_start) = sequence_defaults(increment);
    let min_value = stmt.min_value.unwrap_or(def_min);
    let max_value = stmt.max_value.unwrap_or(def_max);
    if min_value > max_value {
        return Err(ProtocolError::Database(ZyronError::Internal(format!(
            "sequence minimum {min_value} exceeds maximum {max_value}"
        ))));
    }
    let start = stmt.start.unwrap_or(def_start);
    if start < min_value || start > max_value {
        return Err(ProtocolError::Database(ZyronError::Internal(format!(
            "sequence start {start} is out of range [{min_value}, {max_value}]"
        ))));
    }

    let entry = zyron_catalog::SequenceEntry {
        id: 0,
        schema_id,
        name: name.clone(),
        increment,
        min_value,
        max_value,
        start,
        cache: stmt.cache.unwrap_or(1).max(1),
        cycle: stmt.cycle,
        reserved: 0,
    };

    server
        .catalog
        .create_sequence(entry)
        .await
        .map_err(ProtocolError::Database)?;
    Ok(DdlResult::Tag("CREATE SEQUENCE".to_string()))
}

async fn handle_drop_sequence(
    stmt: &zyron_parser::ast::DropSequenceStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (schema_id, name) = resolve_qualified_name(&stmt.name, server, session)?;

    let seq = match server.catalog.get_sequence(schema_id, &name) {
        Some(s) => s,
        None => {
            if stmt.if_exists {
                return Ok(DdlResult::Tag("DROP SEQUENCE".to_string()));
            }
            return Err(ProtocolError::Database(ZyronError::Internal(format!(
                "sequence '{name}' not found"
            ))));
        }
    };

    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Sequence,
        seq.id,
    )?;

    server
        .catalog
        .drop_sequence(schema_id, &name)
        .await
        .map_err(ProtocolError::Database)?;
    Ok(DdlResult::Tag("DROP SEQUENCE".to_string()))
}

async fn handle_alter_sequence(
    stmt: &zyron_parser::ast::AlterSequenceStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (schema_id, name) = resolve_qualified_name(&stmt.name, server, session)?;

    let live = server
        .catalog
        .get_sequence(schema_id, &name)
        .ok_or_else(|| {
            ProtocolError::Database(ZyronError::Internal(format!("sequence '{name}' not found")))
        })?;

    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Sequence,
        live.id,
    )?;

    let increment = stmt.increment.unwrap_or(live.increment);
    if increment == 0 {
        return Err(ProtocolError::Database(ZyronError::Internal(
            "sequence increment must not be zero".to_string(),
        )));
    }
    // Option<Option<i64>>: outer Some means the clause was given, inner None
    // means NO MINVALUE/MAXVALUE which falls back to the increment-based bound.
    let (def_min, def_max, _) = sequence_defaults(increment);
    let min_value = match stmt.min_value {
        Some(inner) => inner.unwrap_or(def_min),
        None => live.min_value,
    };
    let max_value = match stmt.max_value {
        Some(inner) => inner.unwrap_or(def_max),
        None => live.max_value,
    };
    if min_value > max_value {
        return Err(ProtocolError::Database(ZyronError::Internal(format!(
            "sequence minimum {min_value} exceeds maximum {max_value}"
        ))));
    }
    let start = stmt.start.unwrap_or(live.start);
    let cache = stmt.cache.unwrap_or(live.cache).max(1);
    let cycle = stmt.cycle.unwrap_or(live.cycle);

    // RESTART repositions the sequence: Some(Some(n)) restarts at n,
    // Some(None) restarts at the start value. Without RESTART the durable
    // high-water is preserved so values continue where they left off.
    let reserved = match stmt.restart {
        Some(inner) => {
            let restart_at = inner.unwrap_or(start);
            restart_at.saturating_sub(increment)
        }
        None => live.current_reserved(),
    };

    let entry = zyron_catalog::SequenceEntry {
        id: live.id,
        schema_id,
        name: name.clone(),
        increment,
        min_value,
        max_value,
        start,
        cache,
        cycle,
        reserved,
    };

    server
        .catalog
        .alter_sequence(entry)
        .await
        .map_err(ProtocolError::Database)?;
    Ok(DdlResult::Tag("ALTER SEQUENCE".to_string()))
}

// ---------------------------------------------------------------------------
// View handlers
// ---------------------------------------------------------------------------

async fn handle_create_view(
    stmt: &zyron_parser::ast::CreateViewStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
    raw_sql: &str,
) -> Result<DdlResult, ProtocolError> {
    let (schema_id, name) = resolve_qualified_name(&stmt.name, server, session)?;

    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Schema,
        schema_id.0,
    )?;

    // Validate the view query binds and plans before persisting it, so a
    // broken definition is rejected at creation rather than at every later
    // reference.
    let db_id = get_session_database(session)?;
    let search_path = session
        .as_ref()
        .map(|s| s.search_path.clone())
        .unwrap_or_default();
    let select = zyron_parser::ast::Statement::Select(stmt.query.clone());
    zyron_planner::plan(
        &server.catalog,
        db_id,
        search_path,
        select,
        Some(&server.peer_facts()),
    )
    .await
    .map_err(ProtocolError::Database)?;

    let entry = zyron_catalog::ViewEntry {
        id: 0,
        schema_id,
        name: name.clone(),
        definition_sql: raw_sql.to_string(),
        column_aliases: stmt.columns.clone(),
    };

    server
        .catalog
        .create_view(entry, stmt.or_replace)
        .await
        .map_err(ProtocolError::Database)?;
    Ok(DdlResult::Tag("CREATE VIEW".to_string()))
}

async fn handle_drop_view(
    stmt: &zyron_parser::ast::DropViewStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (schema_id, name) = resolve_qualified_name(&stmt.name, server, session)?;

    if server.catalog.get_view(schema_id, &name).is_none() {
        if stmt.if_exists {
            return Ok(DdlResult::Tag("DROP VIEW".to_string()));
        }
        return Err(ProtocolError::Database(ZyronError::Internal(format!(
            "view '{name}' not found"
        ))));
    }

    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Schema,
        schema_id.0,
    )?;

    server
        .catalog
        .drop_view(schema_id, &name)
        .await
        .map_err(ProtocolError::Database)?;
    Ok(DdlResult::Tag("DROP VIEW".to_string()))
}

async fn handle_alter_view(
    stmt: &zyron_parser::ast::AlterViewStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (schema_id, name) = resolve_qualified_name(&stmt.name, server, session)?;

    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Schema,
        schema_id.0,
    )?;

    match &stmt.operation {
        zyron_parser::ast::AlterViewOperation::Rename { new_name } => {
            let bare_new = new_name.rsplit('.').next().unwrap_or(new_name);
            server
                .catalog
                .rename_view(schema_id, &name, bare_new)
                .await
                .map_err(ProtocolError::Database)?;
        }
    }
    Ok(DdlResult::Tag("ALTER VIEW".to_string()))
}

// ---------------------------------------------------------------------------
// Comment handler
// ---------------------------------------------------------------------------

async fn handle_comment_on(
    stmt: &zyron_parser::ast::CommentOnStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    use zyron_parser::ast::CommentObjectType;

    // Setting a comment mutates catalog metadata, so require the schema-level
    // create privilege like the other metadata DDL handlers.
    let (_, schema_id) = get_session_schema(session, server, None)?;
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Schema,
        schema_id.0,
    )?;

    // Stable discriminant for the object kind, persisted with the comment.
    let object_type: u8 = match stmt.object_type {
        CommentObjectType::Table => 0,
        CommentObjectType::Column => 1,
        CommentObjectType::Index => 2,
        CommentObjectType::Schema => 3,
        CommentObjectType::Sequence => 4,
        CommentObjectType::View => 5,
    };
    let column = stmt.column.clone().unwrap_or_default();
    server
        .catalog
        .set_comment(object_type, &stmt.name, &column, stmt.comment.clone())
        .await
        .map_err(ProtocolError::Database)?;
    Ok(DdlResult::Tag("COMMENT".to_string()))
}

// ---------------------------------------------------------------------------
// Function handlers
// ---------------------------------------------------------------------------

async fn handle_create_function(
    stmt: &zyron_parser::ast::CreateFunctionStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    use zyron_parser::ast::{FunctionLanguage, FunctionReturnType};

    let (schema_id, name) = resolve_qualified_name(&stmt.name, server, session)?;

    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Schema,
        schema_id.0,
    )?;

    if stmt.language != FunctionLanguage::Sql {
        return Err(ProtocolError::Database(ZyronError::Internal(
            "only SQL-language functions are supported".to_string(),
        )));
    }
    let return_type = match &stmt.return_type {
        FunctionReturnType::Scalar(dt) => dt.to_type_id(),
        _ => {
            return Err(ProtocolError::Database(ZyronError::Internal(
                "only scalar-returning functions are supported".to_string(),
            )));
        }
    };

    let param_names: Vec<String> = stmt.params.iter().map(|p| p.name.clone()).collect();
    let param_types: Vec<zyron_common::TypeId> = stmt
        .params
        .iter()
        .map(|p| p.data_type.to_type_id())
        .collect();

    let entry = zyron_catalog::FunctionEntry {
        id: 0,
        schema_id,
        name: name.clone(),
        param_names,
        param_types,
        return_type,
        body_sql: stmt.body.clone(),
    };

    server
        .catalog
        .create_function(entry, stmt.or_replace)
        .await
        .map_err(ProtocolError::Database)?;
    Ok(DdlResult::Tag("CREATE FUNCTION".to_string()))
}

async fn handle_drop_function(
    stmt: &zyron_parser::ast::DropFunctionStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (schema_id, name) = resolve_qualified_name(&stmt.name, server, session)?;

    let bare = name.rsplit('.').next().unwrap_or(&name);
    let exists = server
        .catalog
        .list_functions()
        .iter()
        .any(|f| f.name == bare);
    if !exists {
        if stmt.if_exists {
            return Ok(DdlResult::Tag("DROP FUNCTION".to_string()));
        }
        return Err(ProtocolError::Database(ZyronError::Internal(format!(
            "function '{name}' not found"
        ))));
    }

    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Schema,
        schema_id.0,
    )?;

    server
        .catalog
        .drop_function(&name)
        .await
        .map_err(ProtocolError::Database)?;
    Ok(DdlResult::Tag("DROP FUNCTION".to_string()))
}

/// CREATE AGGREGATE name(input_type) (SFUNC=..., STYPE=..., FINALFUNC=...,
/// INITCOND=...). The state and final functions must already exist as SQL
/// functions: SFUNC takes (state, input) and returns the next state, FINALFUNC
/// takes (state) and returns the aggregate result. Exactly one input argument
/// is supported. The aggregate is persisted and resolvable by the binder.
async fn handle_create_aggregate(
    stmt: &zyron_parser::ast::CreateAggregateStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (schema_id, name) = resolve_qualified_name(&stmt.name, server, session)?;

    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Schema,
        schema_id.0,
    )?;

    let input_types: Vec<zyron_common::TypeId> = stmt
        .params
        .iter()
        .map(|p| p.data_type.to_type_id())
        .collect();
    if input_types.len() != 1 {
        return Err(ProtocolError::Database(ZyronError::Internal(
            "a user-defined aggregate must take exactly one input argument".to_string(),
        )));
    }
    let state_type = stmt.stype.to_type_id();

    // The state function takes (state, input) and must already exist.
    let mut sfunc_arg_types = Vec::with_capacity(1 + input_types.len());
    sfunc_arg_types.push(state_type);
    sfunc_arg_types.extend_from_slice(&input_types);
    if server
        .catalog
        .find_function(&stmt.sfunc, &sfunc_arg_types)
        .is_none()
    {
        return Err(ProtocolError::Database(ZyronError::Internal(format!(
            "state function '{}' taking (state, input) was not found; create it first",
            stmt.sfunc
        ))));
    }

    // The final function, when given, takes (state) and yields the result type.
    let return_type = match &stmt.finalfunc {
        Some(ff) => {
            let entry = server
                .catalog
                .find_function(ff, &[state_type])
                .ok_or_else(|| {
                    ProtocolError::Database(ZyronError::Internal(format!(
                        "final function '{ff}' taking (state) was not found; create it first"
                    )))
                })?;
            entry.return_type
        }
        None => state_type,
    };

    let entry = zyron_catalog::AggregateEntry {
        id: 0,
        schema_id,
        name: name.clone(),
        input_types,
        state_type,
        return_type,
        sfunc_name: stmt.sfunc.clone(),
        finalfunc_name: stmt.finalfunc.clone(),
        combinefunc_name: stmt.combinefunc.clone(),
        initcond: stmt.initcond.clone(),
    };

    server
        .catalog
        .create_aggregate(entry, false)
        .await
        .map_err(ProtocolError::Database)?;
    Ok(DdlResult::Tag("CREATE AGGREGATE".to_string()))
}

async fn handle_drop_aggregate(
    stmt: &zyron_parser::ast::DropAggregateStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (schema_id, name) = resolve_qualified_name(&stmt.name, server, session)?;

    let bare = name.rsplit('.').next().unwrap_or(&name);
    let exists = server
        .catalog
        .list_aggregates()
        .iter()
        .any(|a| a.name == bare);
    if !exists {
        if stmt.if_exists {
            return Ok(DdlResult::Tag("DROP AGGREGATE".to_string()));
        }
        return Err(ProtocolError::Database(ZyronError::Internal(format!(
            "aggregate '{name}' not found"
        ))));
    }

    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Schema,
        schema_id.0,
    )?;

    server
        .catalog
        .drop_aggregate(&name)
        .await
        .map_err(ProtocolError::Database)?;
    Ok(DdlResult::Tag("DROP AGGREGATE".to_string()))
}

// ---------------------------------------------------------------------------
// Stored procedure handlers
// ---------------------------------------------------------------------------

/// CREATE PROCEDURE name(params) LANGUAGE SQL AS $$ body $$. The body is one or
/// more SQL statements referencing parameters positionally as $1, $2, ... It is
/// parsed at creation time so a malformed body is rejected up front, then
/// persisted. Only SQL-language procedures are supported.
async fn handle_create_procedure(
    stmt: &zyron_parser::ast::CreateProcedureStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    use zyron_parser::ast::ProcedureLanguage;

    let (schema_id, name) = resolve_qualified_name(&stmt.name, server, session)?;

    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Schema,
        schema_id.0,
    )?;

    if stmt.language != ProcedureLanguage::Sql {
        return Err(ProtocolError::Database(ZyronError::Internal(
            "only SQL-language procedures are supported".to_string(),
        )));
    }

    // Reject a malformed body at creation time rather than on first CALL.
    zyron_parser::parse(&stmt.body).map_err(|e| {
        ProtocolError::Database(ZyronError::Internal(format!(
            "procedure '{name}' body parse error: {e}"
        )))
    })?;

    let param_names: Vec<String> = stmt.params.iter().map(|p| p.name.clone()).collect();
    let param_types: Vec<zyron_common::TypeId> = stmt
        .params
        .iter()
        .map(|p| p.data_type.to_type_id())
        .collect();

    let entry = zyron_catalog::ProcedureEntry {
        id: 0,
        schema_id,
        name: name.clone(),
        param_names,
        param_types,
        body_sql: stmt.body.clone(),
    };

    server
        .catalog
        .create_procedure(entry, stmt.or_replace)
        .await
        .map_err(ProtocolError::Database)?;
    Ok(DdlResult::Tag("CREATE PROCEDURE".to_string()))
}

async fn handle_drop_procedure(
    stmt: &zyron_parser::ast::DropProcedureStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (schema_id, name) = resolve_qualified_name(&stmt.name, server, session)?;

    let bare = name.rsplit('.').next().unwrap_or(&name);
    let exists = server
        .catalog
        .list_procedures()
        .iter()
        .any(|p| p.name == bare);
    if !exists {
        if stmt.if_exists {
            return Ok(DdlResult::Tag("DROP PROCEDURE".to_string()));
        }
        return Err(ProtocolError::Database(ZyronError::Internal(format!(
            "procedure '{name}' not found"
        ))));
    }

    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Schema,
        schema_id.0,
    )?;

    server
        .catalog
        .drop_procedure(&name)
        .await
        .map_err(ProtocolError::Database)?;
    Ok(DdlResult::Tag("DROP PROCEDURE".to_string()))
}

/// CALL name(args). Resolves the procedure, evaluates the arguments to values,
/// then runs the body statements in one transaction with the arguments bound as
/// positional parameters ($1, $2, ...). The whole body is atomic: any statement
/// error aborts the transaction.
/// MERGE desugars into UPDATE, DELETE and INSERT with correlated
/// subqueries and runs them atomically in one transaction. The desugar
/// enforces the guardrails that make sequential execution match single
/// snapshot semantics, see zyron_parser::merge_desugar
async fn handle_merge(
    stmt: &zyron_parser::ast::MergeStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let statements =
        zyron_parser::merge_desugar::desugar_merge(stmt).map_err(ProtocolError::Database)?;
    if statements.is_empty() {
        return Ok(DdlResult::Tag("MERGE 0".to_string()));
    }
    let (db_id, search_path) = session_db_and_search_path(session);
    execute_call_body(server, statements, Vec::new(), db_id, search_path).await?;
    Ok(DdlResult::Tag("MERGE".to_string()))
}

/// Runs a built-in ZyronLake maintenance procedure, or returns None when the
/// name is not one.
///
/// `zyronlake_validate(table)` reports what is wrong with a table's on-disk
/// state, `zyronlake_repair(table [, drop_missing_files])` clears what it can,
/// and `zyronlake_cleanup_orphans(table [, retain_from_version])` reclaims data
/// files no retained version and no branch can reach.
async fn handle_lake_procedure(
    name: &str,
    stmt: &zyron_parser::ast::CallStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<Option<DdlResult>, ProtocolError> {
    let lowered = name.to_ascii_lowercase();
    if !matches!(
        lowered.as_str(),
        "zyronlake_validate" | "zyronlake_repair" | "zyronlake_cleanup_orphans"
    ) {
        return Ok(None);
    }
    if stmt.args.is_empty() {
        return Err(ProtocolError::Database(ZyronError::ParseError(format!(
            "{}() needs the table name as its first argument",
            lowered
        ))));
    }
    let table_name = call_arg_text(&stmt.args[0]).ok_or_else(|| {
        ProtocolError::Database(ZyronError::ParseError(format!(
            "{}() needs a literal table name",
            lowered
        )))
    })?;

    let (_, schema_id) = get_session_schema(session, server, None)?;
    let entry = server
        .catalog
        .get_table(schema_id, &table_name)
        .map_err(ProtocolError::Database)?;
    if !entry.lake.is_lake() {
        return Err(ProtocolError::Database(ZyronError::ConfigError(format!(
            "{}() applies to ZyronLake tables, \"{}\" is a heap table",
            lowered, table_name
        ))));
    }
    let paths = zyron_lake::LakePaths::new(server.disk_manager.data_dir(), entry.id.0);
    let log = zyron_lake::TransactionLog::lookup_shared(&paths).ok_or_else(|| {
        ProtocolError::Database(ZyronError::ConfigError(format!(
            "this node does not run the lake tier, so it cannot maintain \"{}\"",
            table_name
        )))
    })?;

    let rows: Vec<Vec<String>> = match lowered.as_str() {
        "zyronlake_validate" => {
            let report = zyron_lake::validate(&log, true).map_err(ProtocolError::Database)?;
            let mut rows: Vec<Vec<String>> = report
                .problems
                .iter()
                .map(|p| vec!["problem".to_string(), format!("{:?}", p)])
                .collect();
            for (metric, value) in [
                ("head_version", report.head_version.to_string()),
                ("versions_checked", report.versions_checked.to_string()),
                (
                    "checkpoints_checked",
                    report.checkpoints_checked.to_string(),
                ),
                ("files_checked", report.files_checked.to_string()),
                ("healthy", report.is_healthy().to_string()),
            ] {
                rows.push(vec![metric.to_string(), value]);
            }
            rows
        }
        "zyronlake_repair" => {
            let drop_missing = stmt
                .args
                .get(1)
                .and_then(call_arg_text)
                .map(|v| v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            let attempt = lake_maintenance_attempt(zyron_lake::OperationKind::Vacuum);
            let report = zyron_lake::repair(
                &log,
                zyron_lake::RepairOptions {
                    remove_missing_files: drop_missing,
                },
                attempt,
            )
            .map_err(ProtocolError::Database)?;
            [
                (
                    "checkpoints_removed",
                    report.checkpoints_removed.len().to_string(),
                ),
                (
                    "versions_removed",
                    report.versions_removed.len().to_string(),
                ),
                ("files_removed", report.files_removed.len().to_string()),
                (
                    "committed_version",
                    report.version.map(|v| v.to_string()).unwrap_or_default(),
                ),
                ("unrepaired", report.unrepaired.len().to_string()),
            ]
            .into_iter()
            .map(|(metric, value)| vec![metric.to_string(), value])
            .collect()
        }
        _ => {
            let retain = stmt
                .args
                .get(1)
                .and_then(call_arg_text)
                .and_then(|v| v.trim().parse::<u64>().ok())
                .unwrap_or_else(|| log.latest_version());
            let report =
                zyron_lake::cleanup_orphans(&log, retain).map_err(ProtocolError::Database)?;
            [
                ("files_removed", report.removed.len().to_string()),
                ("bytes_reclaimed", report.bytes_reclaimed.to_string()),
                ("staged_files", report.staged_files.to_string()),
                ("retained_from", retain.to_string()),
            ]
            .into_iter()
            .map(|(metric, value)| vec![metric.to_string(), value])
            .collect()
        }
    };

    Ok(Some(DdlResult::Rows {
        tag: format!("CALL {}", rows.len()),
        columns: vec![
            ("metric".to_string(), crate::types::PG_TEXT_OID),
            ("value".to_string(), crate::types::PG_TEXT_OID),
        ],
        rows,
    }))
}

/// The literal text of one CALL argument, None when it is not a literal.
fn call_arg_text(expr: &zyron_parser::Expr) -> Option<String> {
    match expr {
        zyron_parser::Expr::Literal(zyron_parser::LiteralValue::String(s)) => Some(s.clone()),
        zyron_parser::Expr::Literal(zyron_parser::LiteralValue::Integer(n)) => Some(n.to_string()),
        zyron_parser::Expr::Literal(zyron_parser::LiteralValue::Boolean(b)) => Some(b.to_string()),
        zyron_parser::Expr::Identifier(name) => Some(name.clone()),
        zyron_parser::Expr::Nested(inner) => call_arg_text(inner),
        _ => None,
    }
}

/// A standalone maintenance commit: no enclosing database transaction, so it
/// publishes as soon as it lands.
fn lake_maintenance_attempt(
    operation: zyron_lake::OperationKind,
) -> zyron_lake::CommitAttempt<'static> {
    zyron_lake::CommitAttempt {
        operation,
        db_txn_id: 0,
        commit_lsn: 0,
        timestamp_us: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_micros() as i64)
            .unwrap_or(0),
        read_predicate: None,
        audit: None,
    }
}

async fn handle_call(
    stmt: &zyron_parser::ast::CallStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (_schema_id, name) = resolve_qualified_name(&stmt.name, server, session)?;

    // Table-format maintenance procedures are built in: they operate on a
    // table's log rather than running a SQL body, so they resolve before the
    // catalog lookup and a user cannot shadow one
    if let Some(result) = handle_lake_procedure(&name, stmt, server, session).await? {
        return Ok(result);
    }

    let proc = server
        .catalog
        .find_procedure_by_name(&name)
        .ok_or_else(|| {
            ProtocolError::Database(ZyronError::Internal(format!(
                "procedure '{name}' not found"
            )))
        })?;

    if stmt.args.len() != proc.param_names.len() {
        return Err(ProtocolError::Database(ZyronError::Internal(format!(
            "procedure '{}' expects {} argument(s), got {}",
            name,
            proc.param_names.len(),
            stmt.args.len()
        ))));
    }

    let (db_id, search_path) = session_db_and_search_path(session);
    let params = eval_call_args(server, &stmt.args, db_id, &search_path).await?;

    let body_stmts = zyron_parser::parse(&proc.body_sql).map_err(|e| {
        ProtocolError::Database(ZyronError::Internal(format!(
            "procedure '{name}' body parse error: {e}"
        )))
    })?;

    execute_call_body(server, body_stmts, params, db_id, search_path).await?;
    Ok(DdlResult::Tag("CALL".to_string()))
}

/// Evaluates CALL argument expressions to scalar values by running them as a
/// one-row VALUES query. Returns the values in argument order, which become the
/// procedure body's positional parameters.
async fn eval_call_args(
    server: &Arc<ServerState>,
    args: &[zyron_parser::ast::Expr],
    db_id: zyron_catalog::DatabaseId,
    search_path: &[String],
) -> Result<Vec<zyron_executor::column::ScalarValue>, ProtocolError> {
    use zyron_executor::context::ExecutionContext;

    if args.is_empty() {
        return Ok(Vec::new());
    }

    let list = args
        .iter()
        .map(zyron_parser::expr_to_sql)
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!("SELECT {list}");
    let stmt = zyron_parser::parse(&sql)
        .map_err(|e| {
            ProtocolError::Database(ZyronError::Internal(format!(
                "cannot evaluate CALL arguments: {e}"
            )))
        })?
        .into_iter()
        .next()
        .ok_or_else(|| {
            ProtocolError::Database(ZyronError::Internal(
                "cannot evaluate CALL arguments".to_string(),
            ))
        })?;

    let plan = zyron_planner::plan(
        &server.catalog,
        db_id,
        search_path.to_vec(),
        stmt,
        Some(&server.peer_facts()),
    )
    .await
    .map_err(ProtocolError::Database)?;

    let mut txn = server
        .txn_manager
        .begin(zyron_storage::txn::IsolationLevel::ReadCommitted)
        .map_err(ProtocolError::Database)?;
    let snapshot = txn.snapshot.clone();
    let txn_id = u32::try_from(txn.txn_id)
        .map_err(|_| ProtocolError::Database(ZyronError::Internal("txn id overflow".into())))?;
    let mut ctx = ExecutionContext::new(
        server.catalog.clone(),
        server.wal.clone(),
        server.buffer_pool.clone(),
        server.disk_manager.clone(),
        txn_id,
        snapshot,
    );
    ctx.heap_files = Some(Arc::clone(&server.heap_files));
    ctx.btree_indexes = Some(Arc::clone(&server.btree_indexes));
    ctx.foreign_reader = server.foreign_reader.clone();
    ctx.peers = Some(Arc::clone(&server.peers));
    ctx.intent_locks = Some(Arc::clone(server.txn_manager.intent_locks()));
    ctx.row_locks = Some(Arc::clone(server.txn_manager.lock_table()));
    ctx.doc_registry = Some(Arc::clone(&server.doc_registry));
    let ctx = Arc::new(ctx);

    let result = zyron_executor::execute(plan, &ctx).await;
    let _ = server.txn_manager.abort(&mut txn);
    let batches = result.map_err(ProtocolError::Database)?;

    for b in &batches {
        if b.num_rows > 0 {
            return Ok(b
                .columns
                .iter()
                .map(|c| {
                    if c.is_null(0) {
                        zyron_executor::column::ScalarValue::Null
                    } else {
                        c.data.get_scalar(0)
                    }
                })
                .collect());
        }
    }
    Err(ProtocolError::Database(ZyronError::Internal(
        "CALL arguments produced no value".to_string(),
    )))
}

/// Executes an anonymous DO block: parses its body and runs the statements
/// atomically in one transaction (like a procedure body, no parameters). Only
/// the SQL language is supported; DML and SELECT statements are allowed.
async fn handle_do_block(
    stmt: &zyron_parser::ast::DoBlockStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    if let Some(lang) = &stmt.language {
        if !lang.eq_ignore_ascii_case("sql") {
            return Err(ProtocolError::Database(ZyronError::Internal(format!(
                "DO block language '{lang}' is not supported; only SQL"
            ))));
        }
    }

    let statements = zyron_parser::parse(&stmt.body).map_err(|e| {
        ProtocolError::Database(ZyronError::Internal(format!("DO block parse error: {e}")))
    })?;
    if statements.is_empty() {
        return Ok(DdlResult::Tag("DO".to_string()));
    }

    let (db_id, search_path) = session_db_and_search_path(session);
    execute_call_body(server, statements, Vec::new(), db_id, search_path).await?;
    tracing::info!(
        target: "zyron::audit",
        event = "DoBlockExecuted",
        actor_role = actor_role_id(session),
    );
    Ok(DdlResult::Tag("DO".to_string()))
}

/// Runs a procedure body's statements in one transaction with `params` bound as
/// positional parameters. Commits when every statement succeeds, aborts on the
/// first error so the body is atomic.
async fn execute_call_body(
    server: &Arc<ServerState>,
    statements: Vec<zyron_parser::Statement>,
    params: Vec<zyron_executor::column::ScalarValue>,
    db_id: zyron_catalog::DatabaseId,
    search_path: Vec<String>,
) -> Result<(), ProtocolError> {
    use zyron_executor::context::ExecutionContext;

    let mut txn = server
        .txn_manager
        .begin(zyron_storage::txn::IsolationLevel::ReadCommitted)
        .map_err(ProtocolError::Database)?;
    let txn_id = u32::try_from(txn.txn_id)
        .map_err(|_| ProtocolError::Database(ZyronError::Internal("txn id overflow".into())))?;

    for stmt in statements {
        let plan = match zyron_planner::plan(
            &server.catalog,
            db_id,
            search_path.clone(),
            stmt,
            Some(&server.peer_facts()),
        )
        .await
        {
            Ok(p) => p,
            Err(e) => {
                let _ = server.txn_manager.abort(&mut txn);
                return Err(ProtocolError::Database(e));
            }
        };

        let snapshot = txn.snapshot.clone();
        let mut ctx = ExecutionContext::new(
            server.catalog.clone(),
            server.wal.clone(),
            server.buffer_pool.clone(),
            server.disk_manager.clone(),
            txn_id,
            snapshot,
        );
        ctx.heap_files = Some(Arc::clone(&server.heap_files));
        ctx.btree_indexes = Some(Arc::clone(&server.btree_indexes));
        ctx.foreign_reader = server.foreign_reader.clone();
        ctx.peers = Some(Arc::clone(&server.peers));
        ctx.intent_locks = Some(Arc::clone(server.txn_manager.intent_locks()));
        ctx.row_locks = Some(Arc::clone(server.txn_manager.lock_table()));
        ctx.doc_registry = Some(Arc::clone(&server.doc_registry));
        if let Some(m) = &server.fts_manager {
            ctx.set_fts_manager(Arc::clone(m));
        }
        if let Some(m) = &server.vector_manager {
            ctx.set_vector_manager(Arc::clone(m));
        }
        if let Some(m) = &server.spatial_manager {
            ctx.set_spatial_manager(Arc::clone(m));
        }
        ctx.params = params.clone();
        let ctx = Arc::new(ctx);

        if let Err(e) = zyron_executor::execute(plan, &ctx).await {
            let _ = server.txn_manager.abort(&mut txn);
            return Err(ProtocolError::Database(e));
        }
    }

    server
        .txn_manager
        .commit(&mut txn)
        .await
        .map_err(ProtocolError::Database)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Trigger handlers
// ---------------------------------------------------------------------------

/// CREATE TRIGGER name {BEFORE|AFTER} {INSERT|UPDATE|DELETE [OR ...]} ON table
/// FOR EACH {ROW|STATEMENT} EXECUTE FUNCTION proc. The function must be a stored
/// procedure; on a matching event the executor runs its body once per row (for
/// EACH ROW) with the affected row's columns bound as $1..$N, or once per
/// statement (FOR EACH STATEMENT), in the firing statement's transaction.
async fn handle_create_trigger(
    stmt: &zyron_parser::ast::CreateTriggerStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    use zyron_catalog::TriggerEntry;
    use zyron_parser::ast::{TriggerEvent, TriggerGranularity, TriggerTiming};

    let (_, schema_id) = get_session_schema(session, server, None)?;
    let table = server
        .catalog
        .get_table(schema_id, &stmt.table)
        .map_err(ProtocolError::Database)?;

    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Table,
        table.id.0,
    )?;

    let reject = |msg: &str| {
        Err(ProtocolError::Database(ZyronError::Internal(
            msg.to_string(),
        )))
    };
    let timing = match stmt.timing {
        TriggerTiming::Before => TriggerEntry::TIMING_BEFORE,
        TriggerTiming::After => TriggerEntry::TIMING_AFTER,
        TriggerTiming::InsteadOf => {
            return reject("INSTEAD OF triggers are not supported");
        }
    };
    if stmt.when_condition.is_some() {
        return reject("trigger WHEN conditions are not supported");
    }
    if stmt.referencing.is_some() {
        return reject("trigger REFERENCING (transition tables) is not supported");
    }
    if !stmt.args.is_empty() {
        return reject(
            "trigger function arguments are not supported; the procedure receives the row columns as $1..$N",
        );
    }
    let mut events: u8 = 0;
    for ev in &stmt.events {
        events |= match ev {
            TriggerEvent::Insert => TriggerEntry::EVENT_INSERT,
            TriggerEvent::Update => TriggerEntry::EVENT_UPDATE,
            TriggerEvent::Delete => TriggerEntry::EVENT_DELETE,
            TriggerEvent::Truncate => {
                return reject("TRUNCATE triggers are not supported");
            }
        };
    }
    if events == 0 {
        return reject("trigger must specify at least one of INSERT, UPDATE, DELETE");
    }
    let for_each = match stmt.for_each {
        TriggerGranularity::Row => TriggerEntry::FOR_EACH_ROW,
        TriggerGranularity::Statement => TriggerEntry::FOR_EACH_STATEMENT,
    };

    // The action must be an existing stored procedure.
    if server
        .catalog
        .find_procedure_by_name(&stmt.execute_function)
        .is_none()
    {
        return Err(ProtocolError::Database(ZyronError::Internal(format!(
            "trigger function '{}' must be an existing procedure; create it first",
            stmt.execute_function
        ))));
    }

    let entry = TriggerEntry {
        id: 0,
        schema_id,
        table_id: table.id.0,
        name: stmt.name.clone(),
        timing,
        events,
        for_each,
        execute_function: stmt.execute_function.clone(),
        enabled: stmt.enabled,
    };
    server
        .catalog
        .create_trigger(entry)
        .await
        .map_err(ProtocolError::Database)?;
    Ok(DdlResult::Tag("CREATE TRIGGER".to_string()))
}

async fn handle_drop_trigger(
    stmt: &zyron_parser::ast::DropTriggerStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (_, schema_id) = get_session_schema(session, server, None)?;
    let table = server
        .catalog
        .get_table(schema_id, &stmt.table)
        .map_err(ProtocolError::Database)?;

    if server.catalog.find_trigger(table.id, &stmt.name).is_none() {
        if stmt.if_exists {
            return Ok(DdlResult::Tag("DROP TRIGGER".to_string()));
        }
        return Err(ProtocolError::Database(ZyronError::Internal(format!(
            "trigger '{}' not found on table '{}'",
            stmt.name, stmt.table
        ))));
    }

    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Table,
        table.id.0,
    )?;

    server
        .catalog
        .drop_trigger(table.id, &stmt.name)
        .await
        .map_err(ProtocolError::Database)?;
    Ok(DdlResult::Tag("DROP TRIGGER".to_string()))
}

// ---------------------------------------------------------------------------
// Schedule handlers + background execution
// ---------------------------------------------------------------------------

/// Current time in epoch microseconds.
fn schedule_now_micros() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
}

/// Converts a parsed schedule interval to whole seconds. Rejects non-positive
/// durations so a schedule cannot fire in a tight loop.
fn interval_to_secs(d: &zyron_parser::ast::TtlDuration) -> Result<u64, ProtocolError> {
    use zyron_parser::ast::TtlUnit;
    if d.value <= 0 {
        return Err(ProtocolError::Database(ZyronError::Internal(
            "schedule interval must be positive".to_string(),
        )));
    }
    let unit_secs: i64 = match d.unit {
        TtlUnit::Seconds => 1,
        TtlUnit::Minutes => 60,
        TtlUnit::Hours => 3600,
        TtlUnit::Days => 86400,
    };
    Ok((d.value * unit_secs) as u64)
}

/// Extracts the schedule body (the statement after DO) from the original SQL.
/// Scans for the standalone `DO` keyword after the interval keyword so a `DO`
/// inside an identifier or the body itself is not mistaken for the separator.
fn extract_schedule_body(sql: &str) -> Option<String> {
    let lower = sql.to_ascii_lowercase();
    let lb = lower.as_bytes();
    let start = lower.find("every").or_else(|| lower.find("cron"))?;
    let mut i = start;
    while let Some(rel) = lower[i..].find("do") {
        let pos = i + rel;
        let after_idx = pos + 2;
        let before = if pos == 0 { b' ' } else { lb[pos - 1] };
        let after = if after_idx >= lb.len() {
            b' '
        } else {
            lb[after_idx]
        };
        let is_boundary = |c: u8| !(c.is_ascii_alphanumeric() || c == b'_');
        if is_boundary(before) && is_boundary(after) {
            return Some(sql[after_idx..].trim().to_string());
        }
        i = pos + 2;
    }
    None
}

/// Civil (year, month, day) from days since the Unix epoch (proleptic
/// Gregorian, UTC). Howard Hinnant's days-to-civil algorithm.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719468;
    let era = (if z >= 0 { z } else { z - 146096 }) / 146097;
    let doe = (z - era * 146097) as i64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m: i64 = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let year = if m <= 2 { y + 1 } else { y };
    (year, m as u32, d)
}

/// Breaks an epoch second into the cron match fields (minute, hour, day,
/// month, weekday) in UTC. weekday is 0=Sunday .. 6=Saturday.
fn epoch_to_cron_fields(secs: i64) -> (u8, u8, u8, u8, u8) {
    let days = secs.div_euclid(86400);
    let sod = secs.rem_euclid(86400);
    let (_year, month, day) = civil_from_days(days);
    let hour = (sod / 3600) as u8;
    let minute = ((sod % 3600) / 60) as u8;
    let weekday = (((days % 7) + 4).rem_euclid(7)) as u8;
    (minute, hour, day as u8, month as u8, weekday)
}

/// Finds the next epoch-micros instant strictly after `after_micros` that the
/// cron expression matches, scanning minute by minute up to a year ahead.
fn next_cron_fire(cron: &zyron_pipeline::schedule::CronSchedule, after_micros: i64) -> Option<i64> {
    let after_sec = after_micros.div_euclid(1_000_000);
    let mut t = (after_sec / 60 + 1) * 60;
    let limit = t + 366 * 86400;
    while t <= limit {
        let (minute, hour, day, month, weekday) = epoch_to_cron_fields(t);
        if cron.matches(minute, hour, day, month, weekday) {
            return Some(t * 1_000_000);
        }
        t += 60;
    }
    None
}

/// Computes the next fire time (epoch micros) for a schedule relative to `now`.
fn compute_next_run(entry: &zyron_catalog::ScheduleEntry, now_micros: i64) -> Option<i64> {
    if let Some(secs) = entry.interval_secs {
        // Advance from the prior scheduled fire time, not the sweep time, so the
        // cadence does not drift by the sweep latency each period. When the
        // schedule has fallen behind (missed sweeps), step forward in whole
        // intervals until the next fire is in the future, firing once per period
        // rather than bursting one fire per elapsed interval.
        let step = (secs as i64) * 1_000_000;
        if step <= 0 {
            return Some(now_micros);
        }
        let base = entry.next_run.unwrap_or(now_micros);
        let mut next = base + step;
        if next <= now_micros {
            let behind = now_micros - base;
            let periods = behind / step + 1;
            next = base + periods * step;
        }
        Some(next)
    } else if let Some(expr) = &entry.cron_expr {
        zyron_pipeline::schedule::CronSchedule::parse(expr)
            .ok()
            .and_then(|cron| next_cron_fire(&cron, now_micros))
    } else {
        None
    }
}

async fn handle_create_schedule(
    stmt: &zyron_parser::ast::CreateScheduleStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
    sql: &str,
) -> Result<DdlResult, ProtocolError> {
    use zyron_parser::ast::ScheduleInterval;

    let (schema_id, name) = resolve_qualified_name(&stmt.name, server, session)?;

    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Schema,
        schema_id.0,
    )?;

    let (cron_expr, interval_secs) = match &stmt.interval {
        ScheduleInterval::Every(d) => (None, Some(interval_to_secs(d)?)),
        ScheduleInterval::Cron(expr) => {
            // Validate the cron expression up front.
            zyron_pipeline::schedule::CronSchedule::parse(expr).map_err(|e| {
                ProtocolError::Database(ZyronError::Internal(format!(
                    "invalid cron expression: {e}"
                )))
            })?;
            (Some(expr.clone()), None)
        }
    };

    // The body is parsed as a Statement by the grammar; recover its source text
    // and validate it re-parses to exactly one statement before persisting.
    let body_sql = extract_schedule_body(sql).ok_or_else(|| {
        ProtocolError::Database(ZyronError::Internal(
            "could not extract schedule body after DO".to_string(),
        ))
    })?;
    let parsed = zyron_parser::parse(&body_sql).map_err(|e| {
        ProtocolError::Database(ZyronError::Internal(format!(
            "schedule '{name}' body parse error: {e}"
        )))
    })?;
    if parsed.len() != 1 {
        return Err(ProtocolError::Database(ZyronError::Internal(
            "schedule body must be exactly one statement".to_string(),
        )));
    }

    let now = schedule_now_micros();
    let entry = zyron_catalog::ScheduleEntry {
        id: 0,
        schema_id,
        name: name.clone(),
        cron_expr,
        interval_secs,
        body_sql,
        paused: false,
        last_run: None,
        next_run: None,
    };
    let next_run = compute_next_run(&entry, now);
    let entry = zyron_catalog::ScheduleEntry { next_run, ..entry };

    server
        .catalog
        .create_schedule(entry, false)
        .await
        .map_err(ProtocolError::Database)?;
    Ok(DdlResult::Tag("CREATE SCHEDULE".to_string()))
}

async fn handle_drop_schedule(
    stmt: &zyron_parser::ast::DropScheduleStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (_schema_id, name) = resolve_qualified_name(&stmt.name, server, session)?;

    if server.catalog.get_schedule_by_name(&name).is_none() {
        if stmt.if_exists {
            return Ok(DdlResult::Tag("DROP SCHEDULE".to_string()));
        }
        return Err(ProtocolError::Database(ZyronError::Internal(format!(
            "schedule '{name}' not found"
        ))));
    }

    server
        .catalog
        .drop_schedule(&name)
        .await
        .map_err(ProtocolError::Database)?;
    Ok(DdlResult::Tag("DROP SCHEDULE".to_string()))
}

async fn handle_pause_schedule(
    stmt: &zyron_parser::ast::PauseScheduleStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (_schema_id, name) = resolve_qualified_name(&stmt.name, server, session)?;
    let entry = server.catalog.get_schedule_by_name(&name).ok_or_else(|| {
        ProtocolError::Database(ZyronError::Internal(format!("schedule '{name}' not found")))
    })?;
    let mut updated = (*entry).clone();
    updated.paused = true;
    server
        .catalog
        .update_schedule(updated)
        .await
        .map_err(ProtocolError::Database)?;
    Ok(DdlResult::Tag("PAUSE SCHEDULE".to_string()))
}

async fn handle_resume_schedule(
    stmt: &zyron_parser::ast::ResumeScheduleStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (_schema_id, name) = resolve_qualified_name(&stmt.name, server, session)?;
    let entry = server.catalog.get_schedule_by_name(&name).ok_or_else(|| {
        ProtocolError::Database(ZyronError::Internal(format!("schedule '{name}' not found")))
    })?;
    let mut updated = (*entry).clone();
    updated.paused = false;
    // Recompute the next fire from now so a long pause does not produce a burst.
    updated.next_run = compute_next_run(&updated, schedule_now_micros());
    server
        .catalog
        .update_schedule(updated)
        .await
        .map_err(ProtocolError::Database)?;
    Ok(DdlResult::Tag("RESUME SCHEDULE".to_string()))
}

/// Outcome of one schedule sweep, returned for observability and testing.
#[derive(Debug, Clone, Copy, Default)]
pub struct ScheduleRunReport {
    pub executed: usize,
    pub failed: usize,
}

/// Plans and executes a schedule body in its own transaction. Mirrors the
/// background DML execution model: a fresh ExecutionContext over the shared
/// buffer pool and disk manager, with no client session.
async fn execute_schedule_body(
    catalog: &Arc<zyron_catalog::Catalog>,
    txn_manager: &Arc<zyron_storage::txn::TransactionManager>,
    wal: &Arc<zyron_wal::WalWriter>,
    buffer_pool: &Arc<zyron_buffer::BufferPool>,
    disk_manager: &Arc<zyron_storage::DiskManager>,
    body_sql: &str,
    schema_id: zyron_catalog::SchemaId,
) -> Result<(), ZyronError> {
    use zyron_executor::context::ExecutionContext;

    let stmt = zyron_parser::parse(body_sql)
        .map_err(|e| ZyronError::Internal(format!("schedule body parse: {e}")))?
        .into_iter()
        .next()
        .ok_or_else(|| ZyronError::Internal("schedule body is empty".to_string()))?;

    // Plan in the schedule's own database and schema rather than a fixed default
    // so unqualified names in the body resolve to the namespace it was created
    // in.
    let schema = catalog.get_schema_by_id(schema_id)?;
    let plan = zyron_planner::plan(
        catalog,
        schema.database_id,
        vec![schema.name.clone()],
        stmt,
        None,
    )
    .await?;

    let mut txn = txn_manager.begin(zyron_storage::txn::IsolationLevel::ReadCommitted)?;
    let snapshot = txn.snapshot.clone();
    let txn_id =
        u32::try_from(txn.txn_id).map_err(|_| ZyronError::Internal("txn id overflow".into()))?;
    let ctx = Arc::new(ExecutionContext::new(
        Arc::clone(catalog),
        Arc::clone(wal),
        Arc::clone(buffer_pool),
        Arc::clone(disk_manager),
        txn_id,
        snapshot,
    ));

    match zyron_executor::execute(plan, &ctx).await {
        Ok(_) => {
            txn_manager.commit(&mut txn).await?;
            Ok(())
        }
        Err(e) => {
            let _ = txn_manager.abort(&mut txn);
            Err(e)
        }
    }
}

/// Executes every active schedule whose next_run has elapsed, then advances its
/// next_run (and last_run on success). next_run is advanced even on failure so a
/// failing schedule retries on its next period rather than every sweep. Called
/// by the background schedule worker on each tick; reusable from tests with a
/// controlled `now_micros`.
pub async fn run_due_schedules(
    catalog: &Arc<zyron_catalog::Catalog>,
    txn_manager: &Arc<zyron_storage::txn::TransactionManager>,
    wal: &Arc<zyron_wal::WalWriter>,
    buffer_pool: &Arc<zyron_buffer::BufferPool>,
    disk_manager: &Arc<zyron_storage::DiskManager>,
    now_micros: i64,
) -> ScheduleRunReport {
    let mut report = ScheduleRunReport::default();
    let due: Vec<Arc<zyron_catalog::ScheduleEntry>> = catalog
        .list_schedules()
        .into_iter()
        .filter(|s| !s.paused && s.next_run.map(|n| n <= now_micros).unwrap_or(false))
        .collect();

    for sched in due {
        let result = execute_schedule_body(
            catalog,
            txn_manager,
            wal,
            buffer_pool,
            disk_manager,
            &sched.body_sql,
            sched.schema_id,
        )
        .await;

        let mut updated = (*sched).clone();
        updated.next_run = compute_next_run(&sched, now_micros);
        match result {
            Ok(()) => {
                updated.last_run = Some(now_micros);
                report.executed += 1;
            }
            Err(e) => {
                eprintln!("schedule '{}' execution failed: {e}", sched.name);
                report.failed += 1;
            }
        }
        if let Err(e) = catalog.update_schedule(updated).await {
            eprintln!("schedule '{}' state persist failed: {e}", sched.name);
        }
    }
    report
}

// ---------------------------------------------------------------------------
// Materialized view handlers
// ---------------------------------------------------------------------------

/// Builds an INSERT INTO <table> <query> statement that feeds the query result
/// into a materialized view's backing table.
fn build_mv_insert(
    table: String,
    query: Box<zyron_parser::ast::SelectStatement>,
) -> zyron_parser::Statement {
    zyron_parser::Statement::Insert(Box::new(zyron_parser::ast::InsertStatement {
        table,
        columns: Vec::new(),
        source: zyron_parser::ast::InsertSource::Query(query),
        on_conflict: None,
        returning: None,
    }))
}

/// Reads the highest watermark value already loaded into `target` as MAX(key).
/// Returns None when the target is empty (every source row is new) so the
/// incremental insert appends without a lower bound.
async fn read_max_watermark(
    server: &Arc<ServerState>,
    db_id: zyron_catalog::DatabaseId,
    search_path: &[String],
    target: &str,
    key: &str,
) -> Result<Option<zyron_parser::ast::Expr>, ProtocolError> {
    let sql = format!("SELECT MAX(\"{key}\") AS m FROM \"{target}\"");
    let stmt = zyron_parser::parse(&sql)
        .map_err(ProtocolError::Database)?
        .into_iter()
        .next()
        .ok_or_else(|| {
            ProtocolError::Database(ZyronError::Internal("empty watermark query".to_string()))
        })?;
    let (_, batches) = run_pipeline_read(server, db_id, search_path.to_vec(), stmt).await?;
    for b in &batches {
        if b.num_rows > 0 {
            if let Some(col) = b.columns.first() {
                let scalar = col.get_scalar(0);
                if matches!(scalar, zyron_executor::column::ScalarValue::Null) {
                    return Ok(None);
                }
                return Ok(scalar_to_literal(&scalar).map(zyron_parser::ast::Expr::Literal));
            }
        }
    }
    Ok(None)
}

/// Builds the incremental insert: appends the stage output restricted to rows
/// whose watermark column exceeds the highest value already loaded. With no
/// watermark (empty target) it inserts the whole output.
fn build_incremental_insert(
    target: String,
    effective: Box<zyron_parser::ast::SelectStatement>,
    key: &str,
    watermark: Option<zyron_parser::ast::Expr>,
) -> zyron_parser::Statement {
    use zyron_parser::ast::{BinaryOperator, Expr, SelectItem, SelectStatement, TableRef};

    let mut select: SelectStatement = empty_select();
    select.projections = vec![SelectItem::Wildcard];
    select.from = vec![TableRef::Subquery {
        query: effective,
        alias: "_src".to_string(),
    }];
    if let Some(lit) = watermark {
        select.where_clause = Some(Box::new(Expr::BinaryOp {
            left: Box::new(Expr::Identifier(key.to_string())),
            op: BinaryOperator::Gt,
            right: Box::new(lit),
        }));
    }
    build_mv_insert(target, Box::new(select))
}

async fn handle_create_materialized_view(
    stmt: &zyron_parser::ast::CreateMaterializedViewStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
    raw_sql: &str,
) -> Result<DdlResult, ProtocolError> {
    let (schema_id, name) = resolve_qualified_name(&stmt.name, server, session)?;

    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Schema,
        schema_id.0,
    )?;

    if server.catalog.get_mview(schema_id, &name).is_some()
        || server.catalog.get_table(schema_id, &name).is_ok()
    {
        if stmt.if_not_exists {
            return Ok(DdlResult::Tag("CREATE MATERIALIZED VIEW".to_string()));
        }
        return Err(ProtocolError::Database(ZyronError::Internal(format!(
            "relation '{name}' already exists"
        ))));
    }

    // Plan the query to resolve its output schema, then build the backing
    // table with those column types before populating it.
    let select = zyron_parser::ast::Statement::Select(stmt.query.clone());
    let plan = zyron_planner::plan(
        &server.catalog,
        get_session_database(session)?,
        session
            .as_ref()
            .map(|s| s.search_path.clone())
            .unwrap_or_default(),
        select,
        Some(&server.peer_facts()),
    )
    .await
    .map_err(ProtocolError::Database)?;

    let columns: Vec<(String, zyron_common::TypeId, bool, Option<u8>)> = plan
        .output_schema()
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let col_name = if c.name.is_empty() {
                format!("col{i}")
            } else {
                c.name.clone()
            };
            (col_name, c.type_id, c.nullable, c.fractional_digits)
        })
        .collect();
    if columns.is_empty() {
        return Err(ProtocolError::Database(ZyronError::Internal(
            "materialized view query produces no columns".to_string(),
        )));
    }

    let backing_table_id = server
        .catalog
        .create_table_from_columns(schema_id, &name, &columns)
        .await
        .map_err(ProtocolError::Database)?;

    // Populate the backing table from the query. Plan against the session's
    // database and search path so the backing table resolves in its own schema.
    let db_id = get_session_database(session)?;
    let search_path = session
        .as_ref()
        .map(|s| s.search_path.clone())
        .unwrap_or_default();
    let insert = build_mv_insert(name.clone(), stmt.query.clone());
    if let Err(e) = execute_write_stmt(server, db_id, search_path, insert).await {
        // Roll back the backing table so a failed populate leaves no orphan.
        let _ = server.catalog.drop_table(schema_id, &name).await;
        return Err(e);
    }

    let entry = zyron_catalog::MaterializedViewEntry {
        id: 0,
        schema_id,
        name: name.clone(),
        definition_sql: raw_sql.to_string(),
        backing_table_id: backing_table_id.0,
    };
    server
        .catalog
        .create_mview(entry)
        .await
        .map_err(ProtocolError::Database)?;
    Ok(DdlResult::Tag("CREATE MATERIALIZED VIEW".to_string()))
}

async fn handle_refresh_materialized_view(
    stmt: &zyron_parser::ast::RefreshMaterializedViewStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (schema_id, name) = resolve_qualified_name(&stmt.name, server, session)?;

    let mview = server.catalog.get_mview(schema_id, &name).ok_or_else(|| {
        ProtocolError::Database(ZyronError::Internal(format!(
            "materialized view '{name}' not found"
        )))
    })?;

    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Schema,
        schema_id.0,
    )?;

    // Re-parse the stored definition to recover the query.
    let parsed = zyron_parser::parse(&mview.definition_sql).map_err(ProtocolError::Database)?;
    let query = match parsed.into_iter().next() {
        Some(zyron_parser::ast::Statement::CreateMaterializedView(cv)) => cv.query,
        _ => {
            return Err(ProtocolError::Database(ZyronError::Internal(format!(
                "materialized view '{name}' definition is not a CREATE MATERIALIZED VIEW"
            ))));
        }
    };

    // Clear the backing table, then repopulate from the query. Plan against the
    // session's database and search path so the backing table resolves.
    let db_id = get_session_database(session)?;
    let search_path = session
        .as_ref()
        .map(|s| s.search_path.clone())
        .unwrap_or_default();
    let delete = zyron_parser::Statement::Delete(Box::new(zyron_parser::ast::DeleteStatement {
        table: name.clone(),
        where_clause: None,
        returning: None,
        hard: true,
    }));
    execute_write_stmt(server, db_id, search_path.clone(), delete).await?;
    let insert = build_mv_insert(name.clone(), query);
    execute_write_stmt(server, db_id, search_path, insert).await?;

    Ok(DdlResult::Tag("REFRESH MATERIALIZED VIEW".to_string()))
}

async fn handle_drop_materialized_view(
    stmt: &zyron_parser::ast::DropMaterializedViewStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (schema_id, name) = resolve_qualified_name(&stmt.name, server, session)?;

    if server.catalog.get_mview(schema_id, &name).is_none() {
        if stmt.if_exists {
            return Ok(DdlResult::Tag("DROP MATERIALIZED VIEW".to_string()));
        }
        return Err(ProtocolError::Database(ZyronError::Internal(format!(
            "materialized view '{name}' not found"
        ))));
    }

    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Schema,
        schema_id.0,
    )?;

    server
        .catalog
        .drop_mview(schema_id, &name)
        .await
        .map_err(ProtocolError::Database)?;

    // Drop the backing table through the standard path so its heap and FSM
    // files are reclaimed.
    let drop_table = zyron_parser::ast::DropTableStatement {
        name: name.clone(),
        if_exists: true,
    };
    handle_drop_table(&drop_table, server, session).await?;
    Ok(DdlResult::Tag("DROP MATERIALIZED VIEW".to_string()))
}

// ---------------------------------------------------------------------------
// Pipeline handlers
// ---------------------------------------------------------------------------

/// Builds the manager-side stage configs from the parsed AST stages. Only the
/// source/target/mode fields drive DAG validation; the transform SQL is
/// recovered from the stored definition on RUN, so it is not duplicated here.
fn pipeline_stage_configs(
    stages: &[zyron_parser::ast::PipelineStage],
) -> Vec<zyron_pipeline::pipeline::PipelineStageConfig> {
    use zyron_pipeline::pipeline::RefreshMode;
    stages
        .iter()
        .map(|s| zyron_pipeline::pipeline::PipelineStageConfig {
            name: s.name.clone(),
            source: s.source.clone(),
            target: s.target.clone(),
            refresh_mode: RefreshMode::from_mode_str(s.mode.as_deref()),
            transform_sql: None,
            quality_checks: Vec::new(),
        })
        .collect()
}

/// Re-parses a stored pipeline definition back to its statement.
fn parse_pipeline_definition(
    definition_sql: &str,
    name: &str,
) -> Result<zyron_parser::ast::CreatePipelineStatement, ProtocolError> {
    let parsed = zyron_parser::parse(definition_sql).map_err(ProtocolError::Database)?;
    match parsed.into_iter().next() {
        Some(zyron_parser::Statement::CreatePipeline(p)) => Ok(*p),
        _ => Err(ProtocolError::Database(ZyronError::Internal(format!(
            "pipeline '{name}' definition is not a CREATE PIPELINE statement"
        )))),
    }
}

/// A SelectStatement with every field at its empty default.
fn empty_select() -> zyron_parser::ast::SelectStatement {
    use zyron_parser::ast::{SelectStatement, SoftDeleteSelectMode};
    SelectStatement {
        with: None,
        distinct: false,
        distinct_on: Vec::new(),
        projections: Vec::new(),
        from: Vec::new(),
        where_clause: None,
        group_by: Vec::new(),
        group_by_sets: None,
        having: None,
        qualify: None,
        set_ops: Vec::new(),
        order_by: Vec::new(),
        limit: None,
        offset: None,
        fetch: None,
        for_clause: None,
        soft_delete_mode: SoftDeleteSelectMode::Default,
    }
}

/// The select a stage feeds into its target: its TRANSFORM if present, else a
/// full scan of the stage source.
fn stage_effective_select(
    stage: &zyron_parser::ast::PipelineStage,
) -> Box<zyron_parser::ast::SelectStatement> {
    use zyron_parser::ast::{SelectItem, TableRef};
    if let Some(transform) = &stage.transform {
        return transform.clone();
    }
    let mut select = empty_select();
    select.projections = vec![SelectItem::Wildcard];
    select.from = vec![TableRef::Table {
        name: stage.source.clone(),
        alias: None,
        as_of: None,
    }];
    Box::new(select)
}

/// Wraps a predicate so a violation matches: the row is selected when the
/// expectation does not evaluate to TRUE (false or null).
fn expectation_violation_predicate(expr: &zyron_parser::ast::Expr) -> zyron_parser::ast::Expr {
    use zyron_parser::ast::{BinaryOperator, Expr, UnaryOperator};
    let not_true = Expr::UnaryOp {
        op: UnaryOperator::Not,
        expr: Box::new(Expr::Nested(Box::new(expr.clone()))),
    };
    let is_null = Expr::IsNull {
        expr: Box::new(Expr::Nested(Box::new(expr.clone()))),
        negated: false,
    };
    Expr::BinaryOp {
        left: Box::new(not_true),
        op: BinaryOperator::Or,
        right: Box::new(is_null),
    }
}

/// Primary-key column names of a table, in key order. Empty when the table has
/// no primary key.
fn primary_key_columns(table: &zyron_catalog::schema::TableEntry) -> Vec<String> {
    use zyron_catalog::schema::ConstraintType;
    for constraint in &table.constraints {
        if constraint.constraint_type == ConstraintType::PrimaryKey {
            return constraint
                .columns
                .iter()
                .filter_map(|cid| {
                    table
                        .columns
                        .iter()
                        .find(|c| c.id == *cid)
                        .map(|c| c.name.clone())
                })
                .collect();
        }
    }
    Vec::new()
}

/// Plans a select and returns its output column descriptors for building a
/// target table. Does not execute.
async fn plan_select_columns(
    server: &Arc<ServerState>,
    db_id: zyron_catalog::DatabaseId,
    search_path: Vec<String>,
    select: Box<zyron_parser::ast::SelectStatement>,
) -> Result<Vec<(String, zyron_common::TypeId, bool, Option<u8>)>, ProtocolError> {
    let plan = zyron_planner::plan(
        &server.catalog,
        db_id,
        search_path,
        zyron_parser::ast::Statement::Select(select),
        Some(&server.peer_facts()),
    )
    .await
    .map_err(ProtocolError::Database)?;
    Ok(plan
        .output_schema()
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let name = if c.name.is_empty() {
                format!("col{i}")
            } else {
                c.name.clone()
            };
            (name, c.type_id, c.nullable, c.fractional_digits)
        })
        .collect())
}

/// Builds the execution context used for pipeline statements. Mirrors
/// `execute_write_stmt` but plans against the caller's database and search path
/// so stage tables resolve in the pipeline's own schema.
async fn pipeline_context(
    server: &Arc<ServerState>,
) -> Result<
    (
        zyron_storage::txn::Transaction,
        Arc<zyron_executor::context::ExecutionContext>,
    ),
    ProtocolError,
> {
    use zyron_executor::context::ExecutionContext;
    let txn = server
        .txn_manager
        .begin(zyron_storage::txn::IsolationLevel::ReadCommitted)
        .map_err(ProtocolError::Database)?;
    let snapshot = txn.snapshot.clone();
    let txn_id = u32::try_from(txn.txn_id)
        .map_err(|_| ProtocolError::Database(ZyronError::Internal("txn id overflow".into())))?;
    let mut ctx = ExecutionContext::new(
        server.catalog.clone(),
        server.wal.clone(),
        server.buffer_pool.clone(),
        server.disk_manager.clone(),
        txn_id,
        snapshot,
    );
    ctx.heap_files = Some(Arc::clone(&server.heap_files));
    ctx.btree_indexes = Some(Arc::clone(&server.btree_indexes));
    ctx.foreign_reader = server.foreign_reader.clone();
    ctx.peers = Some(Arc::clone(&server.peers));
    ctx.intent_locks = Some(Arc::clone(server.txn_manager.intent_locks()));
    ctx.row_locks = Some(Arc::clone(server.txn_manager.lock_table()));
    ctx.doc_registry = Some(Arc::clone(&server.doc_registry));
    if let Some(m) = &server.fts_manager {
        ctx.set_fts_manager(Arc::clone(m));
    }
    if let Some(m) = &server.vector_manager {
        ctx.set_vector_manager(Arc::clone(m));
    }
    if let Some(m) = &server.spatial_manager {
        ctx.set_spatial_manager(Arc::clone(m));
    }
    Ok((txn, Arc::new(ctx)))
}

/// Runs one read statement (a SELECT) in its own transaction, returning the
/// output column descriptors and the result batches.
async fn run_pipeline_read(
    server: &Arc<ServerState>,
    db_id: zyron_catalog::DatabaseId,
    search_path: Vec<String>,
    stmt: zyron_parser::Statement,
) -> Result<
    (
        Vec<(String, zyron_common::TypeId)>,
        Vec<zyron_executor::batch::DataBatch>,
    ),
    ProtocolError,
> {
    let plan = zyron_planner::plan(
        &server.catalog,
        db_id,
        search_path,
        stmt,
        Some(&server.peer_facts()),
    )
    .await
    .map_err(ProtocolError::Database)?;
    let schema: Vec<(String, zyron_common::TypeId)> = plan
        .output_schema()
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let name = if c.name.is_empty() {
                format!("col{i}")
            } else {
                c.name.clone()
            };
            (name, c.type_id)
        })
        .collect();
    let (mut txn, ctx) = pipeline_context(server).await?;
    match zyron_executor::execute(plan, &ctx).await {
        Ok(batches) => {
            server
                .txn_manager
                .commit(&mut txn)
                .await
                .map_err(ProtocolError::Database)?;
            Ok((schema, batches))
        }
        Err(e) => {
            let _ = server.txn_manager.abort(&mut txn);
            Err(ProtocolError::Database(e))
        }
    }
}

/// Runs a sequence of write statements in a single transaction so a stage's
/// clear-and-load (or merge delete-and-insert) commits atomically. Returns the
/// batches from the final statement, whose count column reports rows written.
async fn run_pipeline_write_txn(
    server: &Arc<ServerState>,
    db_id: zyron_catalog::DatabaseId,
    search_path: Vec<String>,
    stmts: Vec<zyron_parser::Statement>,
) -> Result<Vec<zyron_executor::batch::DataBatch>, ProtocolError> {
    let mut plans = Vec::with_capacity(stmts.len());
    for stmt in stmts {
        plans.push(
            zyron_planner::plan(
                &server.catalog,
                db_id,
                search_path.clone(),
                stmt,
                Some(&server.peer_facts()),
            )
            .await
            .map_err(ProtocolError::Database)?,
        );
    }
    let (mut txn, ctx) = pipeline_context(server).await?;
    let mut last = Vec::new();
    for plan in plans {
        match zyron_executor::execute(plan, &ctx).await {
            Ok(batches) => last = batches,
            Err(e) => {
                let _ = server.txn_manager.abort(&mut txn);
                return Err(ProtocolError::Database(e));
            }
        }
    }
    server
        .txn_manager
        .commit(&mut txn)
        .await
        .map_err(ProtocolError::Database)?;
    Ok(last)
}

/// Converts a scalar key value to a SQL literal for an IN list. Returns None
/// for types that cannot be represented as a literal (binary, UUID, interval),
/// so MERGE can reject them rather than silently skip keys.
fn scalar_to_literal(
    scalar: &zyron_executor::column::ScalarValue,
) -> Option<zyron_parser::ast::LiteralValue> {
    use zyron_executor::column::ScalarValue;
    use zyron_parser::ast::LiteralValue;
    match scalar {
        ScalarValue::Boolean(b) => Some(LiteralValue::Boolean(*b)),
        ScalarValue::Int8(v) => Some(LiteralValue::Integer(*v as i64)),
        ScalarValue::Int16(v) => Some(LiteralValue::Integer(*v as i64)),
        ScalarValue::Int32(v) => Some(LiteralValue::Integer(*v as i64)),
        ScalarValue::Int64(v) => Some(LiteralValue::Integer(*v)),
        ScalarValue::UInt8(v) => Some(LiteralValue::Integer(*v as i64)),
        ScalarValue::UInt16(v) => Some(LiteralValue::Integer(*v as i64)),
        ScalarValue::UInt32(v) => Some(LiteralValue::Integer(*v as i64)),
        ScalarValue::Int128(v) => i64::try_from(*v).ok().map(LiteralValue::Integer),
        ScalarValue::UInt64(v) => i64::try_from(*v).ok().map(LiteralValue::Integer),
        ScalarValue::Float32(v) => Some(LiteralValue::Float(*v as f64)),
        ScalarValue::Float64(v) => Some(LiteralValue::Float(*v)),
        ScalarValue::Utf8(s) => Some(LiteralValue::String(s.clone())),
        _ => None,
    }
}

/// Reads the affected-row count from a modify statement's result batch.
fn affected_count(batches: &[zyron_executor::batch::DataBatch]) -> u64 {
    use zyron_executor::column::ScalarValue;
    if let Some(batch) = batches.first() {
        if let Some(col) = batch.columns.first() {
            if let ScalarValue::Int64(n) = col.get_scalar(0) {
                return n.max(0) as u64;
            }
        }
    }
    0
}

/// Registers (or replaces) a pipeline definition in the in-memory manager so
/// runtime introspection of the live DAG stays consistent with the catalog.
fn register_pipeline_in_manager(server: &Arc<ServerState>, entry: &zyron_catalog::PipelineEntry) {
    let Some(manager) = &server.pipeline_manager else {
        return;
    };
    let Ok(def) = parse_pipeline_definition(&entry.definition_sql, &entry.name) else {
        return;
    };
    let pipeline = zyron_pipeline::pipeline::Pipeline {
        id: zyron_pipeline::ids::PipelineId(entry.id),
        name: entry.name.clone(),
        stages: pipeline_stage_configs(&def.stages),
        enabled: entry.enabled,
        created_at: entry.created_at,
        sla: None,
    };
    let _ = manager.drop_pipeline(&entry.name);
    let _ = manager.create_pipeline(pipeline);
}

async fn handle_create_pipeline(
    stmt: &zyron_parser::ast::CreatePipelineStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
    raw_sql: &str,
) -> Result<DdlResult, ProtocolError> {
    let (schema_id, name) = resolve_qualified_name(&stmt.name, server, session)?;

    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Schema,
        schema_id.0,
    )?;

    if server.catalog.get_pipeline_by_name(&name).is_some() {
        return Err(ProtocolError::Database(ZyronError::Internal(format!(
            "pipeline '{name}' already exists"
        ))));
    }

    if stmt.stages.is_empty() {
        return Err(ProtocolError::Database(ZyronError::Internal(format!(
            "pipeline '{name}' has no stages"
        ))));
    }

    // Reject duplicate stage names and validate the stage graph is acyclic
    // before persisting.
    let mut seen = std::collections::HashSet::new();
    for stage in &stmt.stages {
        if !seen.insert(stage.name.as_str()) {
            return Err(ProtocolError::Database(ZyronError::Internal(format!(
                "pipeline '{name}' has duplicate stage '{}'",
                stage.name
            ))));
        }
    }
    let configs = pipeline_stage_configs(&stmt.stages);
    zyron_pipeline::pipeline::validate_dag(&configs).map_err(ProtocolError::Database)?;

    let entry = zyron_catalog::PipelineEntry {
        id: 0,
        schema_id,
        name: name.clone(),
        definition_sql: raw_sql.to_string(),
        enabled: true,
        created_at: now_micros(),
        last_run: None,
        last_success: None,
        rows_processed: 0,
        status_code: zyron_catalog::PipelineEntry::STATUS_IDLE,
        status_msg: None,
    };
    let id = server
        .catalog
        .create_pipeline(entry, false)
        .await
        .map_err(ProtocolError::Database)?;

    if let Some(reg) = server.catalog.get_pipeline_by_name(&name) {
        let _ = id;
        register_pipeline_in_manager(server, &reg);
    }
    Ok(DdlResult::Tag("CREATE PIPELINE".to_string()))
}

async fn handle_run_pipeline(
    stmt: &zyron_parser::ast::RunPipelineStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    use zyron_parser::ast::Expr;

    let (schema_id, name) = resolve_qualified_name(&stmt.name, server, session)?;

    let entry = server.catalog.get_pipeline_by_name(&name).ok_or_else(|| {
        ProtocolError::Database(ZyronError::Internal(format!("pipeline '{name}' not found")))
    })?;

    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Schema,
        schema_id.0,
    )?;

    let db_id = get_session_database(session)?;
    let search_path = session
        .as_ref()
        .map(|s| s.search_path.clone())
        .unwrap_or_default();

    let definition = parse_pipeline_definition(&entry.definition_sql, &name)?;

    // Resolve the stages to run: a single named stage, or all stages in
    // topological order.
    let order: Vec<usize> = if let Some(stage_name) = &stmt.stage {
        let idx = definition
            .stages
            .iter()
            .position(|s| &s.name == stage_name)
            .ok_or_else(|| {
                ProtocolError::Database(ZyronError::Internal(format!(
                    "pipeline '{name}' has no stage '{stage_name}'"
                )))
            })?;
        vec![idx]
    } else {
        let configs = pipeline_stage_configs(&definition.stages);
        zyron_pipeline::pipeline::validate_dag(&configs).map_err(ProtocolError::Database)?
    };

    // Preview mode: return the rows the targeted stage would produce, without
    // writing. The targeted stage is the named one, or the terminal stage in
    // execution order.
    if let Some(limit) = stmt.preview_limit {
        let stage_idx = *order.last().ok_or_else(|| {
            ProtocolError::Database(ZyronError::Internal(format!(
                "pipeline '{name}' has no stages to preview"
            )))
        })?;
        let mut select = stage_effective_select(&definition.stages[stage_idx]);
        select.limit = Some(Box::new(Expr::Literal(
            zyron_parser::ast::LiteralValue::Integer(limit as i64),
        )));
        let (schema, batches) = run_pipeline_read(
            server,
            db_id,
            search_path.clone(),
            zyron_parser::ast::Statement::Select(select),
        )
        .await?;

        let columns: Vec<(String, i32)> = schema
            .iter()
            .map(|(n, _)| (n.clone(), crate::types::PG_TEXT_OID))
            .collect();
        let mut rows = Vec::new();
        for batch in &batches {
            for row in 0..batch.num_rows {
                let mut values = Vec::with_capacity(batch.columns.len());
                for col in &batch.columns {
                    let scalar = col.get_scalar(row);
                    let mut buf = bytes::BytesMut::with_capacity(32);
                    if crate::types::scalar_write_text(&scalar, &mut buf) {
                        values.push(String::from_utf8_lossy(&buf).into_owned());
                    } else {
                        values.push(String::new());
                    }
                }
                rows.push(values);
            }
        }
        return Ok(DdlResult::Rows {
            tag: format!("RUN PIPELINE {}", rows.len()),
            columns,
            rows,
        });
    }

    // Execute each stage in order, persisting the run outcome.
    let mut total_rows: u64 = 0;
    let run_result = run_pipeline_stages(
        server,
        db_id,
        &search_path,
        schema_id,
        &definition,
        &order,
        &mut total_rows,
    )
    .await;

    let now = now_micros();
    let mut updated = (*entry).clone();
    updated.last_run = Some(now);
    match &run_result {
        Ok(()) => {
            updated.last_success = Some(now);
            updated.rows_processed = total_rows;
            updated.status_code = zyron_catalog::PipelineEntry::STATUS_COMPLETED;
            updated.status_msg = None;
        }
        Err(e) => {
            updated.status_code = zyron_catalog::PipelineEntry::STATUS_FAILED;
            updated.status_msg = Some(format!("{e:?}"));
        }
    }
    server
        .catalog
        .update_pipeline(updated)
        .await
        .map_err(ProtocolError::Database)?;

    run_result?;
    fire_event(
        server,
        zyron_pipeline::event_handler::EventType::PipelineCompleted,
        &name,
        &[
            ("pipeline".to_string(), name.clone()),
            ("rows".to_string(), total_rows.to_string()),
        ],
    )
    .await;
    Ok(DdlResult::Tag("RUN PIPELINE".to_string()))
}

/// Runs the selected stages in order: ensure the target exists, enforce
/// expectations against the stage output, then load per the refresh mode.
#[allow(clippy::too_many_arguments)]
async fn run_pipeline_stages(
    server: &Arc<ServerState>,
    db_id: zyron_catalog::DatabaseId,
    search_path: &[String],
    schema_id: zyron_catalog::SchemaId,
    definition: &zyron_parser::ast::CreatePipelineStatement,
    order: &[usize],
    total_rows: &mut u64,
) -> Result<(), ProtocolError> {
    use zyron_parser::ast::{DeleteStatement, Expr, SelectItem, Statement, TableRef};
    use zyron_pipeline::pipeline::RefreshMode;

    for &idx in order {
        let stage = &definition.stages[idx];
        if stage.target.is_empty() {
            return Err(ProtocolError::Database(ZyronError::Internal(format!(
                "pipeline stage '{}' has no target",
                stage.name
            ))));
        }
        let effective = stage_effective_select(stage);
        let mode = RefreshMode::from_mode_str(stage.mode.as_deref());

        // Create the target from the stage output schema if it does not exist.
        let target_entry = server.catalog.get_table(schema_id, &stage.target);
        if target_entry.is_err() {
            let cols =
                plan_select_columns(server, db_id, search_path.to_vec(), effective.clone()).await?;
            if cols.is_empty() {
                return Err(ProtocolError::Database(ZyronError::Internal(format!(
                    "pipeline stage '{}' produces no columns",
                    stage.name
                ))));
            }
            server
                .catalog
                .create_table_from_columns(schema_id, &stage.target, &cols)
                .await
                .map_err(ProtocolError::Database)?;
        }

        // Enforce expectations against the stage output before loading.
        for (i, expectation) in stage.expectations.iter().enumerate() {
            let mut check = empty_select();
            check.projections = vec![SelectItem::Wildcard];
            check.from = vec![TableRef::Subquery {
                query: effective.clone(),
                alias: "_stage".to_string(),
            }];
            check.where_clause = Some(Box::new(expectation_violation_predicate(&expectation.expr)));
            check.limit = Some(Box::new(Expr::Literal(
                zyron_parser::ast::LiteralValue::Integer(1),
            )));
            let (_schema, batches) = run_pipeline_read(
                server,
                db_id,
                search_path.to_vec(),
                Statement::Select(Box::new(check)),
            )
            .await?;
            let violated = batches.iter().any(|b| b.num_rows > 0);
            if violated {
                return Err(ProtocolError::Database(ZyronError::Internal(format!(
                    "pipeline stage '{}' violated expectation #{}",
                    stage.name,
                    i + 1
                ))));
            }
        }

        // Build the load statements for the refresh mode.
        let mut stmts: Vec<Statement> = Vec::new();
        match mode {
            RefreshMode::Full => {
                stmts.push(Statement::Delete(Box::new(DeleteStatement {
                    table: stage.target.clone(),
                    where_clause: None,
                    returning: None,
                    hard: true,
                })));
                stmts.push(build_mv_insert(stage.target.clone(), effective.clone()));
            }
            RefreshMode::AppendOnly => {
                stmts.push(build_mv_insert(stage.target.clone(), effective.clone()));
            }
            RefreshMode::Incremental => {
                // Append only source rows past the highest watermark already
                // loaded, using the target's single-column primary key as the
                // monotonic watermark. Aliasing this to AppendOnly re-appends
                // the whole source each run and duplicates rows.
                let table = server
                    .catalog
                    .get_table(schema_id, &stage.target)
                    .map_err(ProtocolError::Database)?;
                let pk = primary_key_columns(&table);
                if pk.len() != 1 {
                    return Err(ProtocolError::Database(ZyronError::Internal(format!(
                        "pipeline stage '{}' uses MODE INCREMENTAL but target '{}' needs a \
                         single-column primary key to act as the watermark",
                        stage.name, stage.target
                    ))));
                }
                let key = pk.into_iter().next().unwrap_or_default();

                // Read the current high watermark MAX(pk) from the target.
                let watermark =
                    read_max_watermark(server, db_id, search_path, &stage.target, &key).await?;

                let insert = build_incremental_insert(
                    stage.target.clone(),
                    effective.clone(),
                    &key,
                    watermark,
                );
                stmts.push(insert);
            }
            RefreshMode::Merge => {
                let table = server
                    .catalog
                    .get_table(schema_id, &stage.target)
                    .map_err(ProtocolError::Database)?;
                let pk = primary_key_columns(&table);
                if pk.is_empty() {
                    return Err(ProtocolError::Database(ZyronError::Internal(format!(
                        "pipeline stage '{}' uses MODE MERGE but target '{}' has no primary key",
                        stage.name, stage.target
                    ))));
                }
                if pk.len() > 1 {
                    return Err(ProtocolError::Database(ZyronError::Internal(format!(
                        "pipeline stage '{}' uses MODE MERGE but target '{}' has a composite primary key; \
                         use a single-column key or MODE FULL",
                        stage.name, stage.target
                    ))));
                }
                // Upsert keyed on the primary key: read the new key values from
                // the stage output, delete the target rows carrying those keys,
                // then insert the new output. The keys are inlined as a literal
                // IN list because the executor does not run a subquery inside a
                // DELETE predicate.
                let key = pk.into_iter().next().unwrap_or_default();
                let (schema, batches) = run_pipeline_read(
                    server,
                    db_id,
                    search_path.to_vec(),
                    Statement::Select(effective.clone()),
                )
                .await?;
                let key_idx = schema.iter().position(|(n, _)| n == &key).ok_or_else(|| {
                    ProtocolError::Database(ZyronError::Internal(format!(
                        "pipeline stage '{}' MODE MERGE: transform output has no key column '{}'",
                        stage.name, key
                    )))
                })?;
                let mut key_lits: Vec<Expr> = Vec::new();
                let mut seen = std::collections::HashSet::new();
                for b in &batches {
                    if let Some(col) = b.columns.get(key_idx) {
                        for r in 0..b.num_rows {
                            let scalar = col.get_scalar(r);
                            if matches!(scalar, zyron_executor::column::ScalarValue::Null) {
                                continue;
                            }
                            let lit = scalar_to_literal(&scalar).ok_or_else(|| {
                                ProtocolError::Database(ZyronError::Internal(format!(
                                    "pipeline stage '{}' MODE MERGE: key column '{}' has an \
                                     unsupported type for merge; use MODE FULL",
                                    stage.name, key
                                )))
                            })?;
                            if seen.insert(format!("{lit:?}")) {
                                key_lits.push(Expr::Literal(lit));
                            }
                        }
                    }
                }
                if !key_lits.is_empty() {
                    let in_list = Expr::InList {
                        expr: Box::new(Expr::Identifier(key)),
                        list: key_lits,
                        negated: false,
                    };
                    stmts.push(Statement::Delete(Box::new(DeleteStatement {
                        table: stage.target.clone(),
                        where_clause: Some(Box::new(in_list)),
                        returning: None,
                        hard: true,
                    })));
                }
                stmts.push(build_mv_insert(stage.target.clone(), effective.clone()));
            }
        }

        let batches = run_pipeline_write_txn(server, db_id, search_path.to_vec(), stmts).await?;
        *total_rows = total_rows.saturating_add(affected_count(&batches));
    }
    Ok(())
}

async fn handle_drop_pipeline(
    stmt: &zyron_parser::ast::DropPipelineStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (schema_id, name) = resolve_qualified_name(&stmt.name, server, session)?;

    if server.catalog.get_pipeline_by_name(&name).is_none() {
        if stmt.if_exists {
            return Ok(DdlResult::Tag("DROP PIPELINE".to_string()));
        }
        return Err(ProtocolError::Database(ZyronError::Internal(format!(
            "pipeline '{name}' not found"
        ))));
    }

    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Schema,
        schema_id.0,
    )?;

    server
        .catalog
        .drop_pipeline(&name)
        .await
        .map_err(ProtocolError::Database)?;
    if let Some(manager) = &server.pipeline_manager {
        let _ = manager.drop_pipeline(&name);
    }
    Ok(DdlResult::Tag("DROP PIPELINE".to_string()))
}

// ---------------------------------------------------------------------------
// Branch handlers
// ---------------------------------------------------------------------------

/// Microseconds since the Unix epoch.
fn now_micros() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
}

fn branch_manager(
    server: &Arc<ServerState>,
) -> Result<&Arc<zyron_versioning::BranchManager>, ProtocolError> {
    server.branch_manager.as_ref().ok_or_else(|| {
        ProtocolError::Database(ZyronError::Internal(
            "branch manager is not configured on this server".into(),
        ))
    })
}

/// Requires the schema-level create privilege before a branch lifecycle
/// operation, matching the other catalog-mutating DDL handlers so a branch
/// cannot be created, dropped, or merged without authorization.
fn check_branch_privilege(
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<(), ProtocolError> {
    let (_, schema_id) = get_session_schema(session, server, None)?;
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Schema,
        schema_id.0,
    )
}

async fn handle_create_branch(
    stmt: &zyron_parser::ast::CreateBranchStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    check_branch_privilege(server, session)?;
    if let Some(table_name) = &stmt.on_table {
        return create_lake_branch(stmt, table_name, server, session).await;
    }
    let mgr = branch_manager(server)?;
    let parent = match &stmt.from_branch {
        Some(name) => Some(
            mgr.get_branch_by_name(name)
                .map_err(ProtocolError::Database)?
                .id,
        ),
        None => None,
    };
    // The fork point: an explicit AT VERSION integer, else the current WAL LSN.
    let base_version = match &stmt.at_version {
        Some(zyron_parser::ast::Expr::Literal(zyron_parser::ast::LiteralValue::Integer(v)))
            if *v >= 0 =>
        {
            zyron_versioning::VersionId(*v as u64)
        }
        Some(_) => {
            return Err(ProtocolError::Database(ZyronError::Internal(
                "CREATE BRANCH AT VERSION requires a non-negative integer literal".into(),
            )));
        }
        None => zyron_versioning::VersionId(server.wal.flushed_lsn().0),
    };
    mgr.create_branch(&stmt.name, parent, base_version, "", now_micros())
        .map_err(ProtocolError::Database)?;
    mgr.persist().map_err(ProtocolError::Database)?;
    // A database-wide branch covers lake tables as well, so every lake table
    // forks here rather than at its first branch write. Forking now is what
    // makes the fork point the branch's creation: a table forked later would
    // carry main's writes made in between. One marker file per table, no
    // data read or copied
    fork_lake_tables(server, &stmt.name)?;
    Ok(DdlResult::Tag("CREATE BRANCH".to_string()))
}

/// Forks every lake table this node holds onto a database-wide branch.
///
/// A table whose log this node does not run is skipped rather than failing
/// the statement, matching the branch DDL that names one table: a node that
/// does not hold the lake tier has nothing to fork.
fn fork_lake_tables(server: &Arc<ServerState>, branch: &str) -> Result<(), ProtocolError> {
    let created_us = now_micros();
    for table in server.catalog.list_all_tables() {
        if !table.lake.is_lake() {
            continue;
        }
        let paths = zyron_lake::LakePaths::new(server.disk_manager.data_dir(), table.id.0);
        let Some(log) = zyron_lake::TransactionLog::lookup_shared(&paths) else {
            continue;
        };
        match zyron_lake::create_branch(&log, branch, None, created_us) {
            Ok(_) | Err(ZyronError::BranchAlreadyExists(_)) => {}
            Err(e) => return Err(ProtocolError::Database(e)),
        }
    }
    Ok(())
}

/// Drops a database-wide branch's head on every lake table this node holds.
fn drop_lake_branch_heads(server: &Arc<ServerState>, branch: &str) -> Result<(), ProtocolError> {
    for table in server.catalog.list_all_tables() {
        if !table.lake.is_lake() {
            continue;
        }
        let paths = zyron_lake::LakePaths::new(server.disk_manager.data_dir(), table.id.0);
        match zyron_lake::drop_branch(&paths, branch) {
            // A table the branch never forked has no head to reclaim
            Ok(()) | Err(ZyronError::BranchNotFound(_)) => {}
            Err(e) => return Err(ProtocolError::Database(e)),
        }
    }
    Ok(())
}

/// CREATE VERSION <name> ON <table> [AS OF VERSION <n>]: registers a named,
/// immutable version tag pointing at a specific version of the table. With an
/// explicit AS OF VERSION it tags that version number; otherwise it tags the
/// current WAL position. Time-travel queries resolve `VERSION AS OF '<name>'`
/// to the tagged version.
async fn handle_create_version(
    stmt: &zyron_parser::ast::CreateVersionStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (schema_id, version_name) = resolve_qualified_name(&stmt.name, server, session)?;
    let table = server
        .catalog
        .get_table(schema_id, &stmt.table)
        .map_err(ProtocolError::Database)?;
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Table,
        table.id.0,
    )?;

    let version_id = match &stmt.at_version {
        Some(zyron_parser::ast::Expr::Literal(zyron_parser::ast::LiteralValue::Integer(v)))
            if *v >= 0 =>
        {
            *v as u64
        }
        Some(_) => {
            return Err(ProtocolError::Database(ZyronError::Internal(
                "CREATE VERSION AS OF VERSION requires a non-negative integer literal".into(),
            )));
        }
        // No explicit version: tag the current WAL position, matching how
        // CREATE BRANCH defines its base version.
        None => server.wal.flushed_lsn().0,
    };

    let entry = zyron_catalog::VersionTagEntry {
        id: 0,
        schema_id,
        name: version_name,
        table_id: table.id.0,
        version_id,
    };
    server
        .catalog
        .create_version_tag(entry)
        .await
        .map_err(ProtocolError::Database)?;
    // A retained version now exists, so date transactions by commit LSN from
    // here forward. Transactions that committed before this point are dated at
    // the dawn of recorded history, which is at or before this tag, so the tag
    // sees the full current state. Lower the retention floor so vacuum keeps
    // every tuple still visible at this version. Idempotent.
    let status_map = server.txn_manager.status_map();
    status_map.enable_lsn_tracking();
    status_map.retain_version(version_id);
    Ok(DdlResult::Tag("CREATE VERSION".to_string()))
}

async fn handle_drop_version(
    stmt: &zyron_parser::ast::DropVersionStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (_schema_id, version_name) = resolve_qualified_name(&stmt.name, server, session)?;
    let Some(tag) = server.catalog.get_version_tag_by_name(&version_name) else {
        if stmt.if_exists {
            return Ok(DdlResult::Tag("DROP VERSION".to_string()));
        }
        return Err(ProtocolError::Database(ZyronError::Internal(format!(
            "version '{version_name}' not found"
        ))));
    };
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Table,
        tag.table_id,
    )?;
    server
        .catalog
        .drop_version_tag(&version_name)
        .await
        .map_err(ProtocolError::Database)?;
    // Raise the retention floor to the oldest remaining tag (or u64::MAX when
    // none remain), so vacuum can resume reclaiming history below the dropped
    // tag. The vacuum also recomputes this from the catalog each cycle, so this
    // is the immediate effect rather than the source of truth.
    let new_floor = server
        .catalog
        .list_version_tags()
        .iter()
        .map(|t| t.version_id)
        .min()
        .unwrap_or(u64::MAX);
    server
        .txn_manager
        .status_map()
        .set_version_retention_floor(new_floor);
    Ok(DdlResult::Tag("DROP VERSION".to_string()))
}

async fn handle_use_branch(
    stmt: &zyron_parser::ast::UseBranchStatement,
    server: &Arc<ServerState>,
    active_branch: &mut Option<String>,
) -> Result<DdlResult, ProtocolError> {
    // Validate the branch exists before binding the session to it. A branch
    // created with ON <table> lives on that lake table's log alone and has no
    // database-wide entry, so both places are consulted. Binding to one of
    // those isolates the lake tables that carry it, and a heap write refuses
    // rather than landing on the main line
    let heap = branch_manager(server)
        .ok()
        .and_then(|mgr| mgr.get_branch_by_name(&stmt.name).ok())
        .is_some();
    if !heap && !lake_branch_exists(server, &stmt.name) {
        return Err(ProtocolError::Database(ZyronError::BranchNotFound(
            stmt.name.clone(),
        )));
    }
    *active_branch = Some(stmt.name.clone());
    Ok(DdlResult::Tag("USE BRANCH".to_string()))
}

/// True when any lake table this node holds carries a branch by this name.
fn lake_branch_exists(server: &Arc<ServerState>, branch: &str) -> bool {
    server.catalog.list_all_tables().iter().any(|table| {
        if !table.lake.is_lake() {
            return false;
        }
        let paths = zyron_lake::LakePaths::new(server.disk_manager.data_dir(), table.id.0);
        zyron_lake::branch_info(&paths, branch).is_ok()
    })
}

async fn handle_drop_branch(
    stmt: &zyron_parser::ast::DropBranchStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    check_branch_privilege(server, session)?;
    if let Some(table_name) = &stmt.on_table {
        return drop_lake_branch(stmt, table_name, server, session).await;
    }
    let mgr = branch_manager(server)?;
    match mgr.get_branch_by_name(&stmt.name) {
        Ok(entry) => {
            // Discard the branch's columnar overlays so its patch state does
            // not linger in the stores after the branch is gone
            for table in server.catalog.list_all_tables() {
                if table.columnar.segments.is_empty() {
                    continue;
                }
                let store = zyron_storage::columnar::ColumnarPatchManager::store_for_segment(
                    table.id.0 as u64,
                    std::path::Path::new(&table.columnar.segments[0].path),
                )
                .map_err(ProtocolError::Database)?;
                if store.branch_overlay_rows(entry.id.0).is_empty() {
                    continue;
                }
                store
                    .clear_branch_logged(&server.wal, table.id.0 as u64, entry.id.0)
                    .map_err(ProtocolError::Database)?;
            }
            // The lake heads the branch forked go with it. Data files it
            // added and main never merged become unreferenced, so the
            // table's orphan cleanup reclaims them
            drop_lake_branch_heads(server, &stmt.name)?;
            mgr.delete_branch(entry.id)
                .map_err(ProtocolError::Database)?;
            mgr.persist().map_err(ProtocolError::Database)?;
            Ok(DdlResult::Tag("DROP BRANCH".to_string()))
        }
        Err(_) if stmt.if_exists => Ok(DdlResult::Tag("DROP BRANCH".to_string())),
        Err(e) => Err(ProtocolError::Database(e)),
    }
}

/// Opens the transaction log of a lake table named by branch DDL, refusing a
/// heap table and a node that does not run the lake tier.
fn lake_log_for_branch_ddl(
    table_name: &str,
    what: &str,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<std::sync::Arc<zyron_lake::TransactionLog>, ProtocolError> {
    let (_, schema_id) = get_session_schema(session, server, None)?;
    let entry = server
        .catalog
        .get_table(schema_id, table_name)
        .map_err(ProtocolError::Database)?;
    if !entry.lake.is_lake() {
        return Err(ProtocolError::Database(ZyronError::ConfigError(format!(
            "{} names \"{}\", which is a heap table. Database-wide branches take no ON clause",
            what, table_name
        ))));
    }
    let paths = zyron_lake::LakePaths::new(server.disk_manager.data_dir(), entry.id.0);
    zyron_lake::TransactionLog::lookup_shared(&paths).ok_or_else(|| {
        ProtocolError::Database(ZyronError::ConfigError(format!(
            "this node does not run the lake tier, so it cannot branch \"{}\"",
            table_name
        )))
    })
}

/// `CREATE BRANCH <name> ON <table> [FROM VERSION <n>]`: an alternate head
/// over one lake table, which writes a marker and copies no data.
async fn create_lake_branch(
    stmt: &zyron_parser::ast::CreateBranchStatement,
    table_name: &str,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    if stmt.from_branch.is_some() {
        return Err(ProtocolError::Database(ZyronError::ParseError(
            "CREATE BRANCH ... ON <table> forks a version, use FROM VERSION <n>".into(),
        )));
    }
    let from_version = match &stmt.at_version {
        None => None,
        Some(zyron_parser::ast::Expr::Literal(zyron_parser::ast::LiteralValue::Integer(v)))
            if *v > 0 =>
        {
            Some(*v as u64)
        }
        Some(_) => {
            return Err(ProtocolError::Database(ZyronError::ParseError(
                "CREATE BRANCH ... FROM VERSION requires a positive integer literal".into(),
            )));
        }
    };
    let log = lake_log_for_branch_ddl(table_name, "CREATE BRANCH", server, session)?;
    let created_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0);
    zyron_lake::create_branch(&log, &stmt.name, from_version, created_us)
        .map_err(ProtocolError::Database)?;
    Ok(DdlResult::Tag("CREATE BRANCH".to_string()))
}

/// `MERGE BRANCH <name> INTO main FOR TABLE <table>`: replays the branch's
/// file set onto the table's main log, reporting a conflict rather than
/// resolving one.
async fn merge_lake_branch(
    stmt: &zyron_parser::ast::MergeBranchStatement,
    table_name: &str,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    if !stmt.into_target.eq_ignore_ascii_case("main") {
        return Err(ProtocolError::Database(ZyronError::ParseError(format!(
            "a lake branch merges into main, not \"{}\"",
            stmt.into_target
        ))));
    }
    let log = lake_log_for_branch_ddl(table_name, "MERGE BRANCH", server, session)?;
    let attempt = lake_maintenance_attempt(zyron_lake::OperationKind::Merge);
    let outcome =
        zyron_lake::merge_branch(&log, &stmt.source, attempt).map_err(ProtocolError::Database)?;
    Ok(DdlResult::Rows {
        tag: "MERGE BRANCH".to_string(),
        columns: vec![
            ("metric".to_string(), crate::types::PG_TEXT_OID),
            ("value".to_string(), crate::types::PG_TEXT_OID),
        ],
        rows: vec![
            vec![
                "merged_version".to_string(),
                outcome.version.map(|v| v.to_string()).unwrap_or_default(),
            ],
            vec!["files_added".to_string(), outcome.files_added.to_string()],
            vec![
                "files_removed".to_string(),
                outcome.files_removed.to_string(),
            ],
            vec![
                "predicates_added".to_string(),
                outcome.predicates_added.to_string(),
            ],
        ],
    })
}

/// `DROP BRANCH <name> ON <table>`: removes the branch's versions and marker.
/// Data files it added and main never merged become unreferenced, so the
/// table's orphan cleanup reclaims them.
async fn drop_lake_branch(
    stmt: &zyron_parser::ast::DropBranchStatement,
    table_name: &str,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let log = lake_log_for_branch_ddl(table_name, "DROP BRANCH", server, session)?;
    match zyron_lake::drop_branch(log.paths(), &stmt.name) {
        Ok(()) => Ok(DdlResult::Tag("DROP BRANCH".to_string())),
        Err(ZyronError::BranchNotFound(_)) if stmt.if_exists => {
            Ok(DdlResult::Tag("DROP BRANCH".to_string()))
        }
        Err(e) => Err(ProtocolError::Database(e)),
    }
}

async fn handle_merge_branch(
    stmt: &zyron_parser::ast::MergeBranchStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    check_branch_privilege(server, session)?;
    if let Some(table_name) = &stmt.for_table {
        return merge_lake_branch(stmt, table_name, server, session).await;
    }
    let mgr = branch_manager(server)?;
    let source = mgr
        .get_branch_by_name(&stmt.source)
        .map_err(ProtocolError::Database)?
        .id;

    // Merging into the main line materializes the branch overlay onto the base
    // heap and consumes the branch. Merging into another branch keeps the
    // page-overlay merge with conflict detection.
    if stmt.into_target.eq_ignore_ascii_case("main") {
        // Box the merge future so its large state (an execution context plus
        // per-table operator trees) is heap allocated rather than inlined into
        // the dispatch state machine, which would overflow the worker stack.
        return Box::pin(merge_branch_into_main(server, source, &stmt.source)).await;
    }

    let target = mgr
        .get_branch_by_name(&stmt.into_target)
        .map_err(ProtocolError::Database)?
        .id;
    let result_version = zyron_versioning::VersionId(server.wal.flushed_lsn().0);
    let result = mgr
        .merge_branch(source, target, result_version)
        .map_err(ProtocolError::Database)?;
    if !result.conflicts.is_empty() {
        return Err(ProtocolError::Database(ZyronError::Internal(format!(
            "MERGE BRANCH {} INTO {} has {} page conflict(s); resolve before merging",
            stmt.source,
            stmt.into_target,
            result.conflicts.len()
        ))));
    }
    Ok(DdlResult::Tag(format!(
        "MERGE BRANCH {}",
        result.merged_pages
    )))
}

/// Applies a branch's overlay to the main line: branch row tombstones are
/// applied to the main heap and branch-inserted rows are replayed through the
/// standard insert path so every main index stays consistent. The branch is
/// then consumed so re-reading it cannot double-apply the now-merged changes.
async fn merge_branch_into_main(
    server: &Arc<ServerState>,
    source: zyron_versioning::BranchId,
    source_name: &str,
) -> Result<DdlResult, ProtocolError> {
    use zyron_executor::context::ExecutionContext;

    let mgr = branch_manager(server)?;

    let mut txn = server
        .txn_manager
        .begin(zyron_storage::txn::IsolationLevel::ReadCommitted)
        .map_err(ProtocolError::Database)?;
    let snapshot = txn.snapshot.clone();
    let txn_id = u32::try_from(txn.txn_id)
        .map_err(|_| ProtocolError::Database(ZyronError::Internal("txn id overflow".into())))?;
    let mut ctx = ExecutionContext::new(
        server.catalog.clone(),
        server.wal.clone(),
        server.buffer_pool.clone(),
        server.disk_manager.clone(),
        txn_id,
        snapshot,
    );
    ctx.heap_files = Some(Arc::clone(&server.heap_files));
    ctx.btree_indexes = Some(Arc::clone(&server.btree_indexes));
    ctx.foreign_reader = server.foreign_reader.clone();
    ctx.peers = Some(Arc::clone(&server.peers));
    ctx.doc_registry = Some(Arc::clone(&server.doc_registry));
    if let Some(m) = &server.fts_manager {
        ctx.set_fts_manager(Arc::clone(m));
    }
    if let Some(m) = &server.vector_manager {
        ctx.set_vector_manager(Arc::clone(m));
    }
    if let Some(m) = &server.spatial_manager {
        ctx.set_spatial_manager(Arc::clone(m));
    }
    // No branch_catalog / active_branch_id: the merge writes land on main.
    let ctx = Arc::new(ctx);

    let mut total_inserted = 0u64;
    let mut total_deleted = 0u64;
    for table in server.catalog.list_all_tables() {
        let heap_file_id = table.heap_file_id;
        let cow = mgr.cow_overrides(source, heap_file_id);
        let append_pages = mgr.append_pages(source, heap_file_id);
        if cow.is_empty() && append_pages == 0 {
            continue;
        }
        let append_file_id = mgr
            .branch_files_lookup(source, heap_file_id)
            .map(|f| f.append_file_id)
            .unwrap_or(0);
        let stats = Box::pin(
            zyron_executor::operator::branch_write::merge_branch_table_into_main(
                &ctx,
                table.id,
                &cow,
                append_file_id,
                append_pages,
            ),
        )
        .await
        .map_err(ProtocolError::Database)?;
        total_inserted += stats.inserted;
        total_deleted += stats.deleted;
    }

    // Columnar side: materialize the branch's patch overlay onto the main
    // line. Branch entries the main row does not already hold (matched by
    // transaction id, patches also by column) append as main line writes,
    // then the branch overlay clears. This is the branch wins union the
    // heap page materialization above applies to heap rows.
    for table in server.catalog.list_all_tables() {
        if table.columnar.segments.is_empty() {
            continue;
        }
        let store = zyron_storage::columnar::ColumnarPatchManager::store_for_segment(
            table.id.0 as u64,
            std::path::Path::new(&table.columnar.segments[0].path),
        )
        .map_err(ProtocolError::Database)?;
        let rows = store.branch_overlay_rows(source.0);
        if rows.is_empty() {
            continue;
        }
        for (file_id, rid, row) in &rows {
            let main_row = store.row_overlay(*file_id, *rid);
            for sxid in &row.supersedes {
                let already = main_row
                    .as_ref()
                    .is_some_and(|m| m.supersedes.contains(sxid));
                if already {
                    continue;
                }
                store
                    .supersede_logged(&server.wal, table.id.0 as u64, 0, *file_id, *rid, *sxid)
                    .map_err(ProtocolError::Database)?;
                total_deleted += 1;
            }
            for (col, chain) in &row.patches {
                for p in chain {
                    let already = main_row.as_ref().is_some_and(|m| {
                        m.patches
                            .get(col)
                            .is_some_and(|c| c.iter().any(|mp| mp.patch_xid == p.patch_xid))
                    });
                    if already {
                        continue;
                    }
                    store
                        .patch_logged(
                            &server.wal,
                            table.id.0 as u64,
                            0,
                            *file_id,
                            *rid,
                            *col,
                            p.patch_xid,
                            &p.value,
                        )
                        .map_err(ProtocolError::Database)?;
                    total_inserted += 1;
                }
            }
        }
        store
            .clear_branch_logged(&server.wal, table.id.0 as u64, source.0)
            .map_err(ProtocolError::Database)?;
    }

    server
        .txn_manager
        .commit(&mut txn)
        .await
        .map_err(ProtocolError::Database)?;

    // Consume the branch: its changes now live in main, so the overlay must go.
    mgr.delete_branch(source).map_err(ProtocolError::Database)?;
    mgr.persist().map_err(ProtocolError::Database)?;

    let _ = source_name;
    Ok(DdlResult::Tag(format!(
        "MERGE BRANCH {} {}",
        total_inserted, total_deleted
    )))
}

// ---------------------------------------------------------------------------
// Auth/Role handlers
// ---------------------------------------------------------------------------

/// Seconds since the Unix epoch.
fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Parses a VALID UNTIL timestamp literal to whole seconds since the epoch.
fn parse_valid_until(s: &str) -> Result<u64, ProtocolError> {
    let micros = zyron_common::parse_timestamp_micros(s).map_err(ProtocolError::Database)?;
    if micros < 0 {
        return Err(ProtocolError::Database(ZyronError::Internal(format!(
            "VALID UNTIL timestamp \"{s}\" is before the Unix epoch"
        ))));
    }
    Ok((micros / 1_000_000) as u64)
}

/// Hashes a plaintext password with Balloon hashing for storage. Uses the
/// configured cost parameters when set, otherwise the built-in defaults.
fn hash_password(password: &str, server: &Arc<ServerState>) -> Result<String, ProtocolError> {
    let credential = match &server.balloon_params {
        Some(params) => {
            zyron_auth::PasswordCredential::from_plaintext_with_params(password, params)
        }
        None => zyron_auth::PasswordCredential::from_plaintext(password),
    };
    credential
        .map(|c| c.as_stored().to_string())
        .map_err(ProtocolError::Database)
}

/// Derives the three stored credentials from one plaintext password: the
/// Balloon PHC hash (cleartext/password verify), the SCRAM-SHA-256 secret
/// (SCRAM verify), and md5(password + username) (MD5 verify). Sets all three
/// on the user so any configured auth method can validate the same password.
fn set_user_password(
    user: &mut zyron_auth::User,
    password: &str,
    server: &Arc<ServerState>,
) -> Result<(), ProtocolError> {
    user.password_hash = Some(hash_password(password, server)?);
    user.scram_secret = Some(zyron_auth::scram_sha256_secret(password));
    user.md5_credential = Some(zyron_auth::md5_password_credential(&user.name, password));
    Ok(())
}

async fn handle_create_user(
    stmt: &zyron_parser::ast::CreateUserStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    use zyron_parser::ast::UserOption;

    let sm = require_security_manager(server)?;
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::ManageRoles,
        zyron_auth::ObjectType::System,
        0,
    )?;

    let mut superuser = false;
    let mut can_login = true;
    let mut valid_until = None;
    for opt in &stmt.options {
        match opt {
            UserOption::Superuser(b) => superuser = *b,
            UserOption::Login(b) => can_login = *b,
            UserOption::ValidUntil(s) => valid_until = Some(parse_valid_until(s)?),
        }
    }

    let now = now_secs();

    let mut user = zyron_auth::User {
        id: zyron_auth::UserId(0),
        name: stmt.name.clone(),
        password_hash: None,
        scram_secret: None,
        md5_credential: None,
        api_key_prefix: None,
        api_key_hash: None,
        totp_secret: None,
        connection_limit: -1,
        valid_until,
        locked: false,
        locked_at: None,
        locked_reason: None,
        created_at: now,
        superuser,
        can_login,
    };
    if let Some(pw) = &stmt.password {
        set_user_password(&mut user, pw, server)?;
    }
    sm.create_user(&user)
        .await
        .map_err(ProtocolError::Database)?;

    // A login resolves its privileges through a role of the same name, so a
    // user needs a companion role for the session to obtain a security context.
    if sm.lookup_role(&stmt.name).is_none() {
        let role = zyron_auth::Role {
            id: zyron_auth::RoleId(0),
            name: stmt.name.clone(),
            clearance: zyron_auth::ClassificationLevel::Public,
            created_at: now,
        };
        sm.create_role(&role)
            .await
            .map_err(ProtocolError::Database)?;
    }

    Ok(DdlResult::Tag("CREATE USER".to_string()))
}

async fn handle_alter_user(
    stmt: &zyron_parser::ast::AlterUserStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    use zyron_parser::ast::{AlterUserOperation as Op, UserOption};

    let sm = require_security_manager(server)?;
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::ManageRoles,
        zyron_auth::ObjectType::System,
        0,
    )?;

    let user_missing = || {
        ProtocolError::Database(ZyronError::Internal(format!(
            "user \"{}\" does not exist",
            stmt.name
        )))
    };

    match &stmt.operation {
        Op::SetPassword(pw) => {
            let mut user = sm.lookup_user(&stmt.name).ok_or_else(user_missing)?;
            set_user_password(&mut user, pw, server)?;
            sm.update_user(&user)
                .await
                .map_err(ProtocolError::Database)?;
        }
        Op::Rename { new_name } => {
            sm.rename_user(&stmt.name, new_name)
                .await
                .map_err(ProtocolError::Database)?;
            // Keep the companion role's name in sync so login still resolves.
            sm.rename_role(&stmt.name, new_name)
                .await
                .map_err(ProtocolError::Database)?;
        }
        Op::SetOptions(opts) => {
            let mut user = sm.lookup_user(&stmt.name).ok_or_else(user_missing)?;
            for opt in opts {
                match opt {
                    UserOption::Superuser(b) => user.superuser = *b,
                    UserOption::Login(b) => user.can_login = *b,
                    UserOption::ValidUntil(s) => user.valid_until = Some(parse_valid_until(s)?),
                }
            }
            sm.update_user(&user)
                .await
                .map_err(ProtocolError::Database)?;
        }
    }

    Ok(DdlResult::Tag("ALTER USER".to_string()))
}

async fn handle_alter_role(
    stmt: &zyron_parser::ast::AlterRoleStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    use zyron_parser::ast::{AlterUserOperation as Op, UserOption};

    let sm = require_security_manager(server)?;
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::ManageRoles,
        zyron_auth::ObjectType::System,
        0,
    )?;

    match &stmt.operation {
        Op::Rename { new_name } => {
            let renamed = sm
                .rename_role(&stmt.name, new_name)
                .await
                .map_err(ProtocolError::Database)?;
            if !renamed {
                return Err(ProtocolError::Database(ZyronError::RoleNotFound(
                    stmt.name.clone(),
                )));
            }
            // Rename the companion user account, if one exists.
            if sm.lookup_user(&stmt.name).is_some() {
                sm.rename_user(&stmt.name, new_name)
                    .await
                    .map_err(ProtocolError::Database)?;
            }
        }
        // Password and account options live on the user record. Apply them to
        // the role's companion user, which must exist.
        Op::SetPassword(pw) => {
            let mut user = sm.lookup_user(&stmt.name).ok_or_else(|| {
                ProtocolError::Database(ZyronError::Internal(format!(
                    "role \"{}\" has no login account to set a password on",
                    stmt.name
                )))
            })?;
            set_user_password(&mut user, pw, server)?;
            sm.update_user(&user)
                .await
                .map_err(ProtocolError::Database)?;
        }
        Op::SetOptions(opts) => {
            let mut user = sm.lookup_user(&stmt.name).ok_or_else(|| {
                ProtocolError::Database(ZyronError::Internal(format!(
                    "role \"{}\" has no login account to alter",
                    stmt.name
                )))
            })?;
            for opt in opts {
                match opt {
                    UserOption::Superuser(b) => user.superuser = *b,
                    UserOption::Login(b) => user.can_login = *b,
                    UserOption::ValidUntil(s) => user.valid_until = Some(parse_valid_until(s)?),
                }
            }
            sm.update_user(&user)
                .await
                .map_err(ProtocolError::Database)?;
        }
    }

    Ok(DdlResult::Tag("ALTER ROLE".to_string()))
}

async fn handle_drop_user(
    stmt: &zyron_parser::ast::DropUserStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let sm = require_security_manager(server)?;
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::ManageRoles,
        zyron_auth::ObjectType::System,
        0,
    )?;

    let dropped_user = sm
        .drop_user(&stmt.name)
        .await
        .map_err(ProtocolError::Database)?;

    // Remove the companion role created alongside the user.
    let dropped_role = match sm.lookup_role(&stmt.name) {
        Some(r) => {
            sm.drop_role(r.id).await.map_err(ProtocolError::Database)?;
            true
        }
        None => false,
    };

    if !dropped_user && !dropped_role && !stmt.if_exists {
        return Err(ProtocolError::Database(ZyronError::RoleNotFound(
            stmt.name.clone(),
        )));
    }
    Ok(DdlResult::Tag("DROP USER".to_string()))
}

async fn handle_create_role(
    stmt: &zyron_parser::ast::CreateRoleStatement,
    server: &Arc<ServerState>,
) -> Result<DdlResult, ProtocolError> {
    let sm = require_security_manager(server)?;

    let role = zyron_auth::Role {
        id: zyron_auth::RoleId(0),
        name: stmt.name.clone(),
        clearance: zyron_auth::ClassificationLevel::Public,
        created_at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
    };

    sm.create_role(&role)
        .await
        .map_err(ProtocolError::Database)?;
    Ok(DdlResult::Tag("CREATE ROLE".to_string()))
}

async fn handle_drop_role(
    stmt: &zyron_parser::ast::DropRoleStatement,
    server: &Arc<ServerState>,
) -> Result<DdlResult, ProtocolError> {
    let sm = require_security_manager(server)?;

    match sm.lookup_role(&stmt.name) {
        Some(r) => {
            sm.drop_role(r.id).await.map_err(ProtocolError::Database)?;
            Ok(DdlResult::Tag("DROP ROLE".to_string()))
        }
        None if stmt.if_exists => Ok(DdlResult::Tag("DROP ROLE".to_string())),
        None => Err(ProtocolError::Database(ZyronError::RoleNotFound(
            stmt.name.clone(),
        ))),
    }
}

// ---------------------------------------------------------------------------
// CDC stream DDL
// ---------------------------------------------------------------------------

/// Reads a table option as a string, accepting any scalar value form.
fn cdc_opt_str(options: &[zyron_parser::ast::TableOption], key: &str) -> Option<String> {
    use zyron_parser::ast::TableOptionValue as V;
    options
        .iter()
        .find(|o| o.key.eq_ignore_ascii_case(key))
        .map(|o| match &o.value {
            V::String(s) => s.clone(),
            V::Identifier(s) => s.clone(),
            V::Integer(n) => n.to_string(),
            V::Boolean(b) => b.to_string(),
            V::StringList(l) => l.join(","),
        })
}

/// Reads a comma-or-list table option as a vector of trimmed strings.
fn cdc_opt_list(options: &[zyron_parser::ast::TableOption], key: &str) -> Vec<String> {
    use zyron_parser::ast::TableOptionValue as V;
    match options.iter().find(|o| o.key.eq_ignore_ascii_case(key)) {
        Some(o) => match &o.value {
            V::StringList(l) => l.clone(),
            V::String(s) | V::Identifier(s) => s
                .split(',')
                .map(|p| p.trim().to_string())
                .filter(|p| !p.is_empty())
                .collect(),
            _ => Vec::new(),
        },
        None => Vec::new(),
    }
}

fn cdc_required(
    options: &[zyron_parser::ast::TableOption],
    key: &str,
) -> Result<String, ProtocolError> {
    cdc_opt_str(options, key).ok_or_else(|| {
        ProtocolError::Database(ZyronError::CdcStreamError(format!(
            "CDC stream sink requires option \"{key}\""
        )))
    })
}

async fn handle_create_cdc_stream(
    stmt: &zyron_parser::ast::CreateCdcStreamStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    use zyron_cdc::cdc_stream::{CdcOutputStream, CdcSinkConfig, OutputFormat, StreamRetryPolicy};
    use zyron_cdc::decoder::DecoderPlugin;

    let (_, schema_id) = get_session_schema(session, server, None)?;
    let mgr = server.cdc_stream_manager.as_ref().ok_or_else(|| {
        ProtocolError::Database(ZyronError::CdcStreamError(
            "CDC streaming is not enabled on this server".into(),
        ))
    })?;
    let slot_mgr = server.slot_manager.as_ref().ok_or_else(|| {
        ProtocolError::Database(ZyronError::CdcStreamError(
            "replication slots are not enabled on this server".into(),
        ))
    })?;

    let table = server
        .catalog
        .get_table(schema_id, &stmt.table_name)
        .map_err(ProtocolError::Database)?;

    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Table,
        table.id.0,
    )?;

    let opts = &stmt.options;
    let sink = match stmt.sink_type.to_lowercase().as_str() {
        "kafka" => CdcSinkConfig::Kafka {
            brokers: cdc_required(opts, "brokers")?,
            topic: cdc_required(opts, "topic")?,
            key_columns: cdc_opt_list(opts, "key_columns"),
        },
        "s3" => CdcSinkConfig::S3 {
            bucket: cdc_required(opts, "bucket")?,
            prefix: cdc_opt_str(opts, "prefix").unwrap_or_default(),
            region: cdc_opt_str(opts, "region").unwrap_or_else(|| "us-east-1".to_string()),
            format: match cdc_opt_str(opts, "format")
                .unwrap_or_default()
                .to_lowercase()
                .as_str()
            {
                "parquet" => OutputFormat::Parquet,
                "avro" => OutputFormat::Avro,
                _ => OutputFormat::Json,
            },
            partition_by: cdc_opt_str(opts, "partition_by"),
        },
        "webhook" => CdcSinkConfig::Webhook {
            url: cdc_required(opts, "url")?,
            headers: Vec::new(),
            batch_size: cdc_opt_str(opts, "batch_size")
                .and_then(|s| s.parse().ok())
                .unwrap_or(1000),
        },
        other => {
            return Err(ProtocolError::Database(ZyronError::CdcStreamError(
                format!("unknown CDC sink type \"{other}\"; expected kafka, s3, or webhook"),
            )));
        }
    };

    let decoder_plugin =
        DecoderPlugin::from_str(&cdc_opt_str(opts, "decoder").unwrap_or_else(|| "debezium".into()))
            .map_err(ProtocolError::Database)?;
    let batch_size = cdc_opt_str(opts, "batch_size")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000usize);
    let batch_interval_ms = cdc_opt_str(opts, "batch_interval_ms")
        .and_then(|s| s.parse().ok())
        .unwrap_or(100u64);
    let include_columns = {
        let cols = cdc_opt_list(opts, "columns");
        if cols.is_empty() { None } else { Some(cols) }
    };

    // A dedicated replication slot tracks delivery progress for the stream.
    // Create it checked, then pin WAL retention from the current head so no
    // change between creation and the consumer's first advance is reclaimed.
    let slot_name = format!("{}_slot", stmt.name);
    slot_mgr
        .create_slot(&slot_name, decoder_plugin, Some(vec![table.id.0]))
        .map_err(ProtocolError::Database)?;
    let start = server.wal.next_lsn();
    if let Err(e) = slot_mgr.advance_slot(&slot_name, start) {
        let _ = slot_mgr.drop_slot(&slot_name);
        return Err(ProtocolError::Database(e));
    }

    let stream = CdcOutputStream {
        name: stmt.name.clone(),
        table_id: table.id.0,
        slot_name: slot_name.clone(),
        sink,
        decoder_plugin,
        filter: cdc_opt_str(opts, "filter"),
        include_columns,
        batch_size,
        batch_interval_ms,
        active: true,
        retry_policy: StreamRetryPolicy::default(),
    };
    if let Err(e) = mgr.create_stream(stream) {
        // Roll back the slot so a failed stream registration leaves no orphan.
        let _ = slot_mgr.drop_slot(&slot_name);
        return Err(ProtocolError::Database(e));
    }

    Ok(DdlResult::Tag("CREATE CDC STREAM".to_string()))
}

async fn handle_drop_cdc_stream(
    stmt: &zyron_parser::ast::DropCdcStreamStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let mgr = server.cdc_stream_manager.as_ref().ok_or_else(|| {
        ProtocolError::Database(ZyronError::CdcStreamError(
            "CDC streaming is not enabled on this server".into(),
        ))
    })?;
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::System,
        0,
    )?;

    mgr.drop_stream(&stmt.name)
        .map_err(ProtocolError::Database)?;
    if let Some(slot_mgr) = server.slot_manager.as_ref() {
        let _ = slot_mgr.drop_slot(&format!("{}_slot", stmt.name));
    }
    Ok(DdlResult::Tag("DROP CDC STREAM".to_string()))
}

// ---------------------------------------------------------------------------
// CDC ingest handlers (inbound)
// ---------------------------------------------------------------------------

async fn handle_create_cdc_ingest(
    stmt: &zyron_parser::ast::CreateCdcIngestStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    use zyron_cdc::cdc_ingest::{CdcIngestConfig, CdcIngestSource, OnConflict};
    use zyron_cdc::cdc_stream::OutputFormat;
    use zyron_cdc::decoder::DecoderPlugin;

    let (_, schema_id) = get_session_schema(session, server, None)?;
    let mgr = server.cdc_ingest_manager.as_ref().ok_or_else(|| {
        ProtocolError::Database(ZyronError::CdcIngestError(
            "CDC ingestion is not enabled on this server".into(),
        ))
    })?;

    let table = server
        .catalog
        .get_table(schema_id, &stmt.target_table)
        .map_err(ProtocolError::Database)?;

    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Table,
        table.id.0,
    )?;

    let opts = &stmt.options;
    let source = match stmt.source_type.to_lowercase().as_str() {
        "kafka" => CdcIngestSource::Kafka {
            brokers: cdc_required(opts, "brokers")
                .or_else(|_| cdc_required(opts, "bootstrap_servers"))?,
            topic: cdc_required(opts, "topic")?,
            group_id: cdc_opt_str(opts, "group_id").unwrap_or_else(|| "zyron_ingest".to_string()),
            start_offset: cdc_opt_str(opts, "start_offset"),
        },
        "s3" => CdcIngestSource::S3 {
            bucket: cdc_required(opts, "bucket")?,
            prefix: cdc_opt_str(opts, "prefix").unwrap_or_default(),
            region: cdc_opt_str(opts, "region").unwrap_or_else(|| "us-east-1".to_string()),
            // Ingest reads newline-delimited record files; the record wire
            // format is selected by the decoder option, not a container format.
            format: OutputFormat::Json,
        },
        other => {
            return Err(ProtocolError::Database(ZyronError::CdcIngestError(
                format!("unknown CDC ingest source \"{other}\"; expected kafka or s3"),
            )));
        }
    };

    // Primary key columns drive UPSERT and DELETE keying. Take an explicit
    // option, else the target's declared primary key.
    let mut pk_cols = cdc_opt_list(opts, "primary_key");
    if pk_cols.is_empty() {
        pk_cols = cdc_opt_list(opts, "key_columns");
    }
    if pk_cols.is_empty() {
        pk_cols = primary_key_columns(&table);
    }

    let on_conflict = match cdc_opt_str(opts, "on_conflict")
        .unwrap_or_default()
        .to_lowercase()
        .as_str()
    {
        "skip" | "ignore" => OnConflict::Skip,
        "error" | "fail" => OnConflict::Error,
        _ => OnConflict::Update,
    };

    // Optional dead-letter table for records that fail to decode or apply.
    let dead_letter_table_id =
        match cdc_opt_str(opts, "dead_letter").or_else(|| cdc_opt_str(opts, "dlq")) {
            Some(name) => {
                let dlq = server.catalog.get_table(schema_id, &name).map_err(|_| {
                    ProtocolError::Database(ZyronError::CdcIngestError(format!(
                        "dead letter table \"{name}\" does not exist"
                    )))
                })?;
                Some(dlq.id.0)
            }
            None => None,
        };

    let decoder =
        DecoderPlugin::from_str(&cdc_opt_str(opts, "decoder").unwrap_or_else(|| "debezium".into()))
            .map_err(ProtocolError::Database)?;

    let batch_size = cdc_opt_str(opts, "batch_size")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000usize);

    // Avro writer schema JSON for the Avro decoder, supplied via SCHEMA '...'
    // or WITH (avro_schema = '...').
    let avro_writer_schema = cdc_opt_str(opts, "avro_schema");

    let config = CdcIngestConfig {
        name: stmt.name.clone(),
        source,
        target_table_id: table.id.0,
        primary_key_columns: pk_cols,
        on_conflict,
        dead_letter_table_id,
        decoder,
        avro_writer_schema,
        batch_size,
        active: true,
    };
    mgr.create_ingest(config).map_err(ProtocolError::Database)?;

    Ok(DdlResult::Tag("CREATE CDC INGEST".to_string()))
}

async fn handle_drop_cdc_ingest(
    stmt: &zyron_parser::ast::DropCdcIngestStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let mgr = server.cdc_ingest_manager.as_ref().ok_or_else(|| {
        ProtocolError::Database(ZyronError::CdcIngestError(
            "CDC ingestion is not enabled on this server".into(),
        ))
    })?;
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::System,
        0,
    )?;

    mgr.drop_ingest(&stmt.name)
        .map_err(ProtocolError::Database)?;
    Ok(DdlResult::Tag("DROP CDC INGEST".to_string()))
}

// ---------------------------------------------------------------------------
// Replication slot handlers
// ---------------------------------------------------------------------------

async fn handle_create_replication_slot(
    stmt: &zyron_parser::ast::CreateReplicationSlotStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    use zyron_cdc::decoder::DecoderPlugin;

    let slot_mgr = server.slot_manager.as_ref().ok_or_else(|| {
        ProtocolError::Database(ZyronError::CdcStreamError(
            "replication slots are not enabled on this server".into(),
        ))
    })?;

    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::System,
        0,
    )?;

    let plugin = DecoderPlugin::from_str(&stmt.plugin).map_err(ProtocolError::Database)?;

    // Resolve the optional table filter to table ids in the session schema.
    let table_filter = if stmt.table_filter.is_empty() {
        None
    } else {
        let (_, schema_id) = get_session_schema(session, server, None)?;
        let mut ids = Vec::with_capacity(stmt.table_filter.len());
        for name in &stmt.table_filter {
            let table = server
                .catalog
                .get_table(schema_id, name)
                .map_err(ProtocolError::Database)?;
            ids.push(table.id.0);
        }
        Some(ids)
    };

    slot_mgr
        .create_slot(&stmt.name, plugin, table_filter)
        .map_err(ProtocolError::Database)?;

    // Pin WAL retention from the current head so no change between creation and
    // the consumer's first advance is reclaimed.
    let start = server.wal.next_lsn();
    if let Err(e) = slot_mgr.advance_slot(&stmt.name, start) {
        // Roll back the slot so a failed pin leaves no half-created slot.
        let _ = slot_mgr.drop_slot(&stmt.name);
        return Err(ProtocolError::Database(e));
    }
    slot_mgr.flush_if_dirty().map_err(ProtocolError::Database)?;

    Ok(DdlResult::Tag("CREATE_REPLICATION_SLOT".to_string()))
}

async fn handle_drop_replication_slot(
    stmt: &zyron_parser::ast::DropReplicationSlotStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let slot_mgr = server.slot_manager.as_ref().ok_or_else(|| {
        ProtocolError::Database(ZyronError::CdcStreamError(
            "replication slots are not enabled on this server".into(),
        ))
    })?;

    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::System,
        0,
    )?;

    slot_mgr
        .drop_slot(&stmt.name)
        .map_err(ProtocolError::Database)?;
    Ok(DdlResult::Tag("DROP_REPLICATION_SLOT".to_string()))
}

// ---------------------------------------------------------------------------
// Event handler handlers
// ---------------------------------------------------------------------------

/// Builds the in-memory dispatcher handler from a persisted catalog entry.
fn event_handler_from_entry(
    entry: &zyron_catalog::EventHandlerEntry,
) -> zyron_pipeline::event_handler::EventHandler {
    use zyron_pipeline::event_handler::{EventHandler, EventType};
    use zyron_pipeline::ids::EventHandlerId;
    EventHandler {
        id: EventHandlerId(entry.id),
        name: entry.name.clone(),
        eventType: EventType::from_label(&entry.event_type),
        condition: entry.condition_sql.clone(),
        functionName: entry.execute_function.clone(),
        enabled: entry.enabled,
    }
}

async fn handle_create_event_handler(
    stmt: &zyron_parser::ast::CreateEventHandlerStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (schema_id, name) = resolve_qualified_name(&stmt.name, server, session)?;

    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Schema,
        schema_id.0,
    )?;

    if server.catalog.get_event_handler_by_name(&name).is_some() {
        return Err(ProtocolError::Database(ZyronError::Internal(format!(
            "event handler '{name}' already exists"
        ))));
    }

    // The handler runs a stored procedure; require it to exist up front.
    if server
        .catalog
        .find_procedure_by_name(&stmt.execute_function)
        .is_none()
    {
        return Err(ProtocolError::Database(ZyronError::Internal(format!(
            "event handler '{name}' references unknown procedure '{}'",
            stmt.execute_function
        ))));
    }

    let condition_sql = stmt
        .condition
        .as_ref()
        .map(|e| zyron_parser::expr_to_sql(e));

    let entry = zyron_catalog::EventHandlerEntry {
        id: 0,
        schema_id,
        name: name.clone(),
        event_type: stmt.event_type.clone(),
        condition_sql,
        execute_function: stmt.execute_function.clone(),
        enabled: true,
    };
    let id = server
        .catalog
        .create_event_handler(entry, false)
        .await
        .map_err(ProtocolError::Database)?;

    if let Some(dispatcher) = &server.event_dispatcher {
        if let Some(reg) = server.catalog.get_event_handler_by_name(&name) {
            let _ = id;
            let _ = dispatcher.register(event_handler_from_entry(&reg));
        }
    }

    Ok(DdlResult::Tag("CREATE EVENT HANDLER".to_string()))
}

async fn handle_drop_event_handler(
    stmt: &zyron_parser::ast::DropEventHandlerStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (_schema_id, name) = resolve_qualified_name(&stmt.name, server, session)?;

    if server.catalog.get_event_handler_by_name(&name).is_none() {
        if stmt.if_exists {
            return Ok(DdlResult::Tag("DROP EVENT HANDLER".to_string()));
        }
        return Err(ProtocolError::Database(ZyronError::Internal(format!(
            "event handler '{name}' not found"
        ))));
    }

    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Schema,
        0,
    )?;

    server
        .catalog
        .drop_event_handler(&name)
        .await
        .map_err(ProtocolError::Database)?;
    if let Some(dispatcher) = &server.event_dispatcher {
        let _ = dispatcher.dropHandler(&name);
    }
    Ok(DdlResult::Tag("DROP EVENT HANDLER".to_string()))
}

/// Evaluates a constant boolean predicate (an event handler condition). Returns
/// true only when the expression evaluates to a TRUE scalar. Any error or
/// non-boolean result is treated as not satisfied.
async fn eval_event_condition(server: &Arc<ServerState>, condition_sql: &str) -> bool {
    use zyron_executor::column::ScalarValue;
    use zyron_executor::context::ExecutionContext;

    let sql = format!("SELECT ({condition_sql})");
    let Ok(parsed) = zyron_parser::parse(&sql) else {
        return false;
    };
    let Some(stmt) = parsed.into_iter().next() else {
        return false;
    };
    let Ok(plan) = zyron_planner::plan(
        &server.catalog,
        zyron_catalog::DatabaseId(1),
        vec!["public".to_string()],
        stmt,
        Some(&server.peer_facts()),
    )
    .await
    else {
        return false;
    };
    let Ok(mut txn) = server
        .txn_manager
        .begin(zyron_storage::txn::IsolationLevel::ReadCommitted)
    else {
        return false;
    };
    let snapshot = txn.snapshot.clone();
    let Ok(txn_id) = u32::try_from(txn.txn_id) else {
        let _ = server.txn_manager.abort(&mut txn);
        return false;
    };
    let ctx = Arc::new(ExecutionContext::new(
        server.catalog.clone(),
        server.wal.clone(),
        server.buffer_pool.clone(),
        server.disk_manager.clone(),
        txn_id,
        snapshot,
    ));
    let result = zyron_executor::execute(plan, &ctx).await;
    let _ = server.txn_manager.abort(&mut txn);
    match result {
        Ok(batches) => batches.iter().any(|b| {
            b.num_rows > 0
                && b.columns
                    .first()
                    .map(|c| matches!(c.get_scalar(0), ScalarValue::Boolean(true)))
                    .unwrap_or(false)
        }),
        Err(_) => false,
    }
}

/// Fires all enabled handlers registered for an event by running each handler's
/// stored procedure with the event payload as a single JSON text parameter
/// (procedures may also declare zero parameters). Best effort: a missing
/// dispatcher, an unsatisfied condition, or a handler failure is logged and
/// does not affect the operation that raised the event.
pub async fn fire_event(
    server: &Arc<ServerState>,
    event_type: zyron_pipeline::event_handler::EventType,
    source: &str,
    details: &[(String, String)],
) {
    use zyron_executor::column::ScalarValue;

    let Some(dispatcher) = &server.event_dispatcher else {
        return;
    };
    let handlers = dispatcher.handlersFor(&event_type);
    if handlers.is_empty() {
        return;
    }

    // The payload the handler procedure receives as its single JSON parameter.
    let mut detail_map = serde_json::Map::new();
    for (k, v) in details {
        detail_map.insert(k.clone(), serde_json::Value::String(v.clone()));
    }
    let payload = serde_json::json!({
        "event": event_type.to_string(),
        "source": source,
        "details": serde_json::Value::Object(detail_map),
    })
    .to_string();

    for handler in handlers {
        if !handler.enabled {
            continue;
        }
        if let Some(cond) = &handler.condition {
            if !eval_event_condition(server, cond).await {
                continue;
            }
        }
        let Some(proc) = server.catalog.find_procedure_by_name(&handler.functionName) else {
            tracing::warn!(
                target: "zyron::events",
                handler = %handler.name,
                "event handler procedure '{}' not found", handler.functionName
            );
            continue;
        };
        let params = match proc.param_names.len() {
            0 => Vec::new(),
            1 => vec![ScalarValue::Utf8(payload.clone())],
            n => {
                tracing::warn!(
                    target: "zyron::events",
                    handler = %handler.name,
                    "event handler procedure '{}' takes {n} params; expected 0 or 1",
                    handler.functionName
                );
                continue;
            }
        };
        let body_stmts = match zyron_parser::parse(&proc.body_sql) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(target: "zyron::events", handler = %handler.name, "procedure body parse failed: {e}");
                continue;
            }
        };
        let (db_id, search_path) = session_db_and_search_path(&None);
        if let Err(e) = execute_call_body(server, body_stmts, params, db_id, search_path).await {
            tracing::warn!(target: "zyron::events", handler = %handler.name, "event handler execution failed: {e:?}");
        }
    }
}

// ---------------------------------------------------------------------------
// Archive / Restore (lifecycle data movement)
// ---------------------------------------------------------------------------

/// Serializes result batches to newline-delimited JSON objects keyed by column
/// name, text-encoding each value. A null cell becomes JSON null so restore can
/// omit it. This format round-trips through restore via build_ingest_insert.
fn serialize_rows_to_json(
    schema: &[(String, zyron_common::TypeId)],
    batches: &[zyron_executor::batch::DataBatch],
) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    for b in batches {
        for r in 0..b.num_rows {
            let mut obj = serde_json::Map::new();
            for (i, col) in b.columns.iter().enumerate() {
                let name = schema
                    .get(i)
                    .map(|(n, _)| n.clone())
                    .unwrap_or_else(|| format!("col{i}"));
                let scalar = col.get_scalar(r);
                let mut buf = bytes::BytesMut::with_capacity(32);
                let value = if crate::types::scalar_write_text(&scalar, &mut buf) {
                    serde_json::Value::String(String::from_utf8_lossy(&buf).into_owned())
                } else {
                    serde_json::Value::Null
                };
                obj.insert(name, value);
            }
            if let Ok(bytes) = serde_json::to_vec(&serde_json::Value::Object(obj)) {
                out.push(bytes);
            }
        }
    }
    out
}

async fn handle_archive_table(
    stmt: &zyron_parser::ast::ArchiveTableStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    use zyron_parser::ast::Expr;
    let (schema_id, name) = resolve_qualified_name(&stmt.table, server, session)?;
    let table = server
        .catalog
        .get_table(schema_id, &name)
        .map_err(ProtocolError::Database)?;
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Table,
        table.id.0,
    )?;

    let db_id = get_session_database(session)?;
    let search_path = session
        .as_ref()
        .map(|s| s.search_path.clone())
        .unwrap_or_default();

    // Read the rows to archive.
    let where_sql = stmt
        .where_clause
        .as_ref()
        .map(|e| format!(" WHERE {}", zyron_parser::expr_to_sql(e)))
        .unwrap_or_default();
    let select_sql = format!("SELECT * FROM \"{name}\"{where_sql}");
    let select_stmt = zyron_parser::parse(&select_sql)
        .map_err(ProtocolError::Database)?
        .into_iter()
        .next()
        .ok_or_else(|| {
            ProtocolError::Database(ZyronError::Internal("empty archive query".to_string()))
        })?;
    let (schema, batches) =
        run_pipeline_read(server, db_id, search_path.clone(), select_stmt).await?;
    let rows = serialize_rows_to_json(&schema, &batches);
    let count = rows.len();

    if stmt.dry_run {
        return Ok(DdlResult::Tag(format!(
            "ARCHIVE TABLE (dry run: {count} row(s) would be archived)"
        )));
    }

    if count > 0 {
        // Delete the exact rows that were archived, keyed on the primary key
        // read from the same batches. Re-running the original predicate would
        // also delete rows inserted between the read and the delete that were
        // never archived, losing them. Keying on the archived primary keys
        // bounds the delete to exactly the archived set.
        let pk = primary_key_columns(&table);
        if pk.is_empty() {
            return Err(ProtocolError::Database(ZyronError::Internal(format!(
                "ARCHIVE TABLE requires a primary key on '{name}' to delete exactly the archived \
                 rows; the table has none"
            ))));
        }

        // Map each primary-key column to its position in the read output.
        let mut pk_idx: Vec<usize> = Vec::with_capacity(pk.len());
        for col in &pk {
            let idx = schema.iter().position(|(n, _)| n == col).ok_or_else(|| {
                ProtocolError::Database(ZyronError::Internal(format!(
                    "ARCHIVE TABLE: primary key column '{col}' missing from the read output"
                )))
            })?;
            pk_idx.push(idx);
        }

        // Build a predicate matching the archived rows by primary key. A single
        // column uses an IN list; a composite key uses an OR of per-row AND
        // equalities.
        let where_clause = if pk.len() == 1 {
            let key = pk[0].clone();
            let key_pos = pk_idx[0];
            let mut key_lits: Vec<Expr> = Vec::new();
            let mut seen = std::collections::HashSet::new();
            for b in &batches {
                if let Some(col) = b.columns.get(key_pos) {
                    for r in 0..b.num_rows {
                        let scalar = col.get_scalar(r);
                        if matches!(scalar, zyron_executor::column::ScalarValue::Null) {
                            continue;
                        }
                        let lit = scalar_to_literal(&scalar).ok_or_else(|| {
                            ProtocolError::Database(ZyronError::Internal(format!(
                                "ARCHIVE TABLE: primary key column '{key}' has an unsupported type"
                            )))
                        })?;
                        if seen.insert(format!("{lit:?}")) {
                            key_lits.push(Expr::Literal(lit));
                        }
                    }
                }
            }
            Some(Box::new(Expr::InList {
                expr: Box::new(Expr::Identifier(key)),
                list: key_lits,
                negated: false,
            }))
        } else {
            let mut row_preds: Vec<Expr> = Vec::new();
            for b in &batches {
                for r in 0..b.num_rows {
                    let mut conj: Option<Expr> = None;
                    for (col_name, &pos) in pk.iter().zip(pk_idx.iter()) {
                        let scalar = b
                            .columns
                            .get(pos)
                            .map(|c| c.get_scalar(r))
                            .unwrap_or(zyron_executor::column::ScalarValue::Null);
                        let lit = scalar_to_literal(&scalar).ok_or_else(|| {
                            ProtocolError::Database(ZyronError::Internal(format!(
                                "ARCHIVE TABLE: primary key column '{col_name}' has an \
                                 unsupported type"
                            )))
                        })?;
                        let eq = Expr::BinaryOp {
                            left: Box::new(Expr::Identifier(col_name.clone())),
                            op: zyron_parser::ast::BinaryOperator::Eq,
                            right: Box::new(Expr::Literal(lit)),
                        };
                        conj = Some(match conj {
                            None => eq,
                            Some(c) => Expr::BinaryOp {
                                left: Box::new(c),
                                op: zyron_parser::ast::BinaryOperator::And,
                                right: Box::new(eq),
                            },
                        });
                    }
                    if let Some(c) = conj {
                        row_preds.push(c);
                    }
                }
            }
            row_preds
                .into_iter()
                .reduce(|a, b| Expr::BinaryOp {
                    left: Box::new(a),
                    op: zyron_parser::ast::BinaryOperator::Or,
                    right: Box::new(b),
                })
                .map(Box::new)
        };

        // Write the archive to durable object storage first, then remove the
        // archived rows. A failed delete leaves the rows present (the archive is
        // a superset), never loses data.
        zyron_lifecycle::archive::archive_rows(&stmt.destination, &rows)
            .await
            .map_err(ProtocolError::Database)?;

        if let Some(where_clause) = where_clause {
            let delete =
                zyron_parser::Statement::Delete(Box::new(zyron_parser::ast::DeleteStatement {
                    table: name.clone(),
                    where_clause: Some(where_clause),
                    returning: None,
                    hard: true,
                }));
            run_pipeline_write_txn(server, db_id, search_path, vec![delete]).await?;
        }
    }

    Ok(DdlResult::Tag(format!("ARCHIVE TABLE {count}")))
}

async fn handle_restore_table(
    stmt: &zyron_parser::ast::RestoreTableStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    if stmt.at_version.is_some() || stmt.at_timestamp.is_some() {
        return Err(ProtocolError::Database(ZyronError::Internal(
            "point-in-time RESTORE from an archive is not supported; the archive is a flat \
             snapshot. Use AS OF time-travel queries on a versioned table instead."
                .to_string(),
        )));
    }

    // Restore into INTO target, else the named table.
    let target_name = stmt
        .into_table
        .clone()
        .filter(|s| !s.is_empty())
        .or_else(|| Some(stmt.table.clone()).filter(|s| !s.is_empty()))
        .ok_or_else(|| {
            ProtocolError::Database(ZyronError::Internal(
                "RESTORE FROM requires a target table (INTO <table>)".to_string(),
            ))
        })?;

    let (schema_id, target) = resolve_qualified_name(&target_name, server, session)?;
    let table = server
        .catalog
        .get_table(schema_id, &target)
        .map_err(ProtocolError::Database)?;
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Table,
        table.id.0,
    )?;

    let (raw_rows, _result) = zyron_lifecycle::archive::restore_from(&stmt.source)
        .await
        .map_err(ProtocolError::Database)?;

    let db_id = get_session_database(session)?;
    let search_path = session
        .as_ref()
        .map(|s| s.search_path.clone())
        .unwrap_or_default();

    let mut stmts: Vec<zyron_parser::Statement> = Vec::with_capacity(raw_rows.len());
    for line in &raw_rows {
        let value: serde_json::Value = serde_json::from_slice(line).map_err(|e| {
            ProtocolError::Database(ZyronError::Internal(format!(
                "archive record is not valid JSON: {e}"
            )))
        })?;
        let Some(obj) = value.as_object() else {
            continue;
        };
        // Non-null fields become typed INSERT columns; null fields are omitted
        // so they take the column default or NULL.
        let pairs: Vec<(String, String)> = obj
            .iter()
            .filter_map(|(k, v)| match v {
                serde_json::Value::Null => None,
                serde_json::Value::String(s) => Some((k.clone(), s.clone())),
                other => Some((k.clone(), other.to_string())),
            })
            .collect();
        if let Some(insert) = build_ingest_insert(&target, &table, &pairs) {
            stmts.push(insert);
        }
    }

    let restored = stmts.len();
    if !stmts.is_empty() {
        run_pipeline_write_txn(server, db_id, search_path, stmts).await?;
    }

    Ok(DdlResult::Tag(format!("RESTORE TABLE {restored}")))
}

// ---------------------------------------------------------------------------
// Table feature toggles (ALTER TABLE ENABLE/DISABLE <feature>)
// ---------------------------------------------------------------------------

/// Default change-data-feed retention when a table enables CDF without an
/// explicit retention configured.
const DEFAULT_CDF_RETENTION_DAYS: u32 = 7;

/// Toggles a per-table feature flag, performing any companion setup. Supports
/// soft_delete (a pure metadata flag) and change_data_feed (the flag plus
/// registering or removing the table's change feed). Other features return a
/// clear error directing to their dedicated DDL.
async fn set_table_feature(
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
    table_name: &str,
    feature: &str,
    enable: bool,
) -> Result<DdlResult, ProtocolError> {
    let (_, schema_id) = get_session_schema(session, server, None)?;
    let table = server
        .catalog
        .get_table(schema_id, table_name)
        .map_err(ProtocolError::Database)?;
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Table,
        table.id.0,
    )?;

    let mut entry = (*table).clone();
    match feature.to_ascii_lowercase().as_str() {
        "soft_delete" | "soft_deletes" | "soft_delete_enabled" => {
            entry.lifecycle.soft_delete_enabled = enable;
            server
                .catalog
                .update_table(entry)
                .await
                .map_err(ProtocolError::Database)?;
        }
        "cdf" | "cdc" | "change_data_feed" | "change_feed" => {
            // CDF capture needs a registered feed, so the registry must exist.
            let registry = server.cdc_registry.as_ref().cloned().ok_or_else(|| {
                ProtocolError::Database(ZyronError::Internal(
                    "change data feed requires CDC to be enabled on this server".into(),
                ))
            })?;
            entry.cdf_enabled = enable;
            if enable && entry.cdf_retention_days == 0 {
                entry.cdf_retention_days = DEFAULT_CDF_RETENTION_DAYS;
            }
            let table_id = entry.id.0;
            let retention = entry.cdf_retention_days;
            server
                .catalog
                .update_table(entry)
                .await
                .map_err(ProtocolError::Database)?;
            if enable {
                registry
                    .enable_for_table(table_id, retention)
                    .map_err(ProtocolError::Database)?;
            } else {
                registry
                    .disable_for_table(table_id, false)
                    .map_err(ProtocolError::Database)?;
            }
        }
        other => {
            return Err(ProtocolError::Database(ZyronError::Internal(format!(
                "unknown or unsupported table feature '{other}'; supported features are \
                 soft_delete and change_data_feed"
            ))));
        }
    }
    Ok(DdlResult::Tag("ALTER TABLE".to_string()))
}

async fn handle_enable_feature(
    stmt: &zyron_parser::ast::EnableFeatureStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    set_table_feature(server, session, &stmt.table, &stmt.feature, true).await
}

async fn handle_disable_feature(
    stmt: &zyron_parser::ast::DisableFeatureStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    set_table_feature(server, session, &stmt.table, &stmt.feature, false).await
}

/// Outcome of applying a batch of inbound CDC records.
#[derive(Debug, Clone, Copy, Default)]
pub struct IngestApplyReport {
    pub applied: usize,
    pub skipped: usize,
    pub failed: usize,
    /// Number of leading records the source offset may advance past. A record
    /// is committed when it is applied, skipped, or durably routed to the dead
    /// letter table. Counting stops at the first record that could not be
    /// committed (a failure with no dead letter table, or a failed dead letter
    /// write) so the offset never moves past data that was lost.
    pub committed: usize,
    /// Records that failed to apply and could not be written to the dead letter
    /// table (no dead letter configured, or the dead letter write itself
    /// failed).
    pub dead_letter_errors: usize,
    /// Records that failed to apply and were durably written to the dead letter
    /// table. Accumulated into the persisted per-job dead-letter total.
    pub dead_lettered: usize,
}

/// Builds a SQL literal for an inbound string value coerced to the target
/// column type. Empty strings become NULL for non-text columns (the JSON
/// decoders render SQL null as an empty string). Integer and float columns
/// parse to typed literals; unparseable values fall back to a string literal so
/// the binder reports the type error rather than silently corrupting data.
fn ingest_value_literal(value: &str, type_id: zyron_common::TypeId) -> zyron_parser::ast::Expr {
    use zyron_common::TypeId;
    use zyron_parser::ast::{Expr, LiteralValue};
    if value.is_empty() && !type_id.is_string() {
        return Expr::Literal(LiteralValue::Null);
    }
    match type_id {
        TypeId::Boolean => Expr::Literal(LiteralValue::Boolean(matches!(
            value.to_ascii_lowercase().as_str(),
            "true" | "t" | "1" | "yes"
        ))),
        t if t.is_integer() => value
            .parse::<i64>()
            .map(|n| Expr::Literal(LiteralValue::Integer(n)))
            .unwrap_or_else(|_| Expr::Literal(LiteralValue::String(value.to_string()))),
        t if t.is_floating_point() => value
            .parse::<f64>()
            .map(|f| Expr::Literal(LiteralValue::Float(f)))
            .unwrap_or_else(|_| Expr::Literal(LiteralValue::String(value.to_string()))),
        _ => Expr::Literal(LiteralValue::String(value.to_string())),
    }
}

/// The database id and single-schema search path that resolves a table living
/// in `schema_id`. Inbound ingestion runs without a client session, so the
/// search path is derived from the target's own schema.
fn ingest_search_path(
    server: &Arc<ServerState>,
    schema_id: zyron_catalog::SchemaId,
) -> (zyron_catalog::DatabaseId, Vec<String>) {
    let name = server
        .catalog
        .get_schema_by_id(schema_id)
        .map(|s| s.name.clone())
        .unwrap_or_else(|_| "public".to_string());
    (zyron_catalog::DatabaseId(1), vec![name])
}

/// Selects the (column, value) pairs that key a row: the configured primary key
/// columns, or every provided value when no key is declared (full-row match).
fn ingest_key_pairs(pk_cols: &[String], values: &[(String, String)]) -> Vec<(String, String)> {
    if pk_cols.is_empty() {
        return values.to_vec();
    }
    pk_cols
        .iter()
        .filter_map(|k| {
            values
                .iter()
                .find(|(n, _)| n.eq_ignore_ascii_case(k))
                .map(|(n, v)| (n.clone(), v.clone()))
        })
        .collect()
}

/// Builds `INSERT INTO target (cols) VALUES (...)` from a change's column
/// values, keeping only columns that exist on the target. Returns None when no
/// value maps to a real column.
fn build_ingest_insert(
    target: &str,
    table: &zyron_catalog::schema::TableEntry,
    values: &[(String, String)],
) -> Option<zyron_parser::Statement> {
    use zyron_parser::ast::{InsertSource, InsertStatement};
    let mut columns = Vec::new();
    let mut row = Vec::new();
    for (name, val) in values {
        if let Some(c) = table
            .columns
            .iter()
            .find(|c| c.name.eq_ignore_ascii_case(name))
        {
            columns.push(c.name.clone());
            row.push(ingest_value_literal(val, c.type_id));
        }
    }
    if columns.is_empty() {
        return None;
    }
    Some(zyron_parser::Statement::Insert(Box::new(InsertStatement {
        table: target.to_string(),
        columns,
        source: InsertSource::Values(vec![row]),
        on_conflict: None,
        returning: None,
    })))
}

/// Builds an equality predicate over the given key pairs (AND of `col = value`).
fn build_ingest_key_predicate(
    table: &zyron_catalog::schema::TableEntry,
    keys: &[(String, String)],
) -> Option<zyron_parser::ast::Expr> {
    use zyron_parser::ast::{BinaryOperator, Expr};
    let mut pred: Option<Expr> = None;
    for (name, val) in keys {
        let type_id = table
            .columns
            .iter()
            .find(|c| c.name.eq_ignore_ascii_case(name))
            .map(|c| c.type_id)?;
        let eq = Expr::BinaryOp {
            left: Box::new(Expr::Identifier(name.clone())),
            op: BinaryOperator::Eq,
            right: Box::new(ingest_value_literal(val, type_id)),
        };
        pred = Some(match pred {
            Some(p) => Expr::BinaryOp {
                left: Box::new(p),
                op: BinaryOperator::And,
                right: Box::new(eq),
            },
            None => eq,
        });
    }
    pred
}

/// Builds `DELETE FROM target WHERE <key predicate>` for a change's key pairs.
fn build_ingest_delete(
    target: &str,
    table: &zyron_catalog::schema::TableEntry,
    keys: &[(String, String)],
) -> Option<zyron_parser::Statement> {
    use zyron_parser::ast::DeleteStatement;
    if keys.is_empty() {
        return None;
    }
    let pred = build_ingest_key_predicate(table, keys)?;
    Some(zyron_parser::Statement::Delete(Box::new(DeleteStatement {
        table: target.to_string(),
        where_clause: Some(Box::new(pred)),
        returning: None,
        hard: true,
    })))
}

/// Returns whether a row carrying the given keys already exists in the target.
async fn ingest_key_exists(
    server: &Arc<ServerState>,
    db_id: zyron_catalog::DatabaseId,
    search_path: &[String],
    target: &str,
    table: &zyron_catalog::schema::TableEntry,
    keys: &[(String, String)],
) -> bool {
    use zyron_parser::ast::{Expr, LiteralValue, SelectItem, Statement, TableRef};
    let Some(pred) = build_ingest_key_predicate(table, keys) else {
        return false;
    };
    let mut select = empty_select();
    select.projections = vec![SelectItem::Expr(
        Expr::Literal(LiteralValue::Integer(1)),
        None,
    )];
    select.from = vec![TableRef::Table {
        name: target.to_string(),
        alias: None,
        as_of: None,
    }];
    select.where_clause = Some(Box::new(pred));
    select.limit = Some(Box::new(Expr::Literal(LiteralValue::Integer(1))));
    match run_pipeline_read(
        server,
        db_id,
        search_path.to_vec(),
        Statement::Select(Box::new(select)),
    )
    .await
    {
        Ok((_, batches)) => batches.iter().any(|b| b.num_rows > 0),
        Err(_) => false,
    }
}

/// Inserts a record that failed to decode or apply into the configured dead
/// letter table, filling the columns it recognizes by name (payload, error,
/// ingest) or falling back to the first column for the raw payload. Returns
/// true only when the record was durably written to the dead letter table so
/// the caller can decide whether the source offset may advance past it.
async fn route_to_dead_letter(
    server: &Arc<ServerState>,
    dead_letter_table_id: u32,
    raw: &[u8],
    error: &str,
    ingest_name: &str,
) -> bool {
    let Ok(dlq) = server
        .catalog
        .get_table_by_id(zyron_catalog::TableId(dead_letter_table_id))
    else {
        return false;
    };
    let payload = String::from_utf8_lossy(raw).into_owned();
    let mut columns = Vec::new();
    let mut row = Vec::new();
    for c in &dlq.columns {
        let mapped = match c.name.to_ascii_lowercase().as_str() {
            "payload" | "raw" | "data" | "record" | "message" => Some(payload.clone()),
            "error" | "error_message" | "err" => Some(error.to_string()),
            "ingest" | "ingest_name" | "source" => Some(ingest_name.to_string()),
            _ => None,
        };
        if let Some(v) = mapped {
            columns.push(c.name.clone());
            row.push(ingest_value_literal(&v, c.type_id));
        }
    }
    if columns.is_empty() {
        if let Some(c) = dlq.columns.first() {
            columns.push(c.name.clone());
            row.push(ingest_value_literal(&payload, c.type_id));
        }
    }
    if columns.is_empty() {
        return false;
    }
    let stmt = zyron_parser::Statement::Insert(Box::new(zyron_parser::ast::InsertStatement {
        table: dlq.name.clone(),
        columns,
        source: zyron_parser::ast::InsertSource::Values(vec![row]),
        on_conflict: None,
        returning: None,
    }));
    let (db_id, search_path) = ingest_search_path(server, dlq.schema_id);
    match run_pipeline_write_txn(server, db_id, search_path, vec![stmt]).await {
        Ok(_) => true,
        Err(e) => {
            tracing::error!(
                target: "zyron::cdc",
                ingest = %ingest_name,
                dead_letter_table_id,
                "dead letter write failed: {e:?}"
            );
            false
        }
    }
}

/// Decodes and applies a batch of inbound CDC records to an ingest job's target
/// table. Inserts and update post-images upsert per the conflict strategy,
/// deletes and update pre-images remove by key, truncates clear the target, and
/// schema-change markers are no-ops. Records that fail to decode or apply are
/// routed to the dead letter table when configured. Each record commits on its
/// own so one bad record never rolls back the rest of the batch.
pub async fn apply_ingest_records(
    server: &Arc<ServerState>,
    config: &zyron_cdc::cdc_ingest::CdcIngestConfig,
    raw_records: &[Vec<u8>],
) -> IngestApplyReport {
    use zyron_cdc::ChangeType;
    use zyron_cdc::decoder::create_decoder_with_schema;
    use zyron_parser::Statement;
    use zyron_parser::ast::DeleteStatement;

    let mut report = IngestApplyReport::default();
    // Build the decoder with the configured Avro writer schema. A schema that
    // fails to parse is a configuration error affecting every record, so each
    // is routed to the dead letter table and the offset holds at the first that
    // cannot be durably routed.
    let decoder =
        match create_decoder_with_schema(config.decoder, config.avro_writer_schema.as_deref()) {
            Ok(d) => d,
            Err(e) => {
                let reason = format!("decoder construction failed: {e}");
                let mut blocked = false;
                for raw in raw_records {
                    report.failed += 1;
                    let durable = match config.dead_letter_table_id {
                        Some(dlq) => {
                            route_to_dead_letter(server, dlq, raw, &reason, &config.name).await
                        }
                        None => false,
                    };
                    if durable {
                        report.dead_lettered += 1;
                        if !blocked {
                            report.committed += 1;
                        }
                    } else {
                        report.dead_letter_errors += 1;
                        blocked = true;
                    }
                }
                return report;
            }
        };

    let table = match server
        .catalog
        .get_table_by_id(zyron_catalog::TableId(config.target_table_id))
    {
        Ok(t) => t,
        Err(_) => {
            // The target is gone: every record fails. Advance only past records
            // that land durably in the dead letter table; hold the offset at the
            // first that does not so nothing is lost.
            let mut blocked = false;
            for raw in raw_records {
                report.failed += 1;
                let durable = match config.dead_letter_table_id {
                    Some(dlq) => {
                        route_to_dead_letter(
                            server,
                            dlq,
                            raw,
                            "ingest target table not found",
                            &config.name,
                        )
                        .await
                    }
                    None => false,
                };
                if durable {
                    report.dead_lettered += 1;
                    if !blocked {
                        report.committed += 1;
                    }
                } else {
                    report.dead_letter_errors += 1;
                    blocked = true;
                }
            }
            return report;
        }
    };
    let target = table.name.clone();
    let (db_id, search_path) = ingest_search_path(server, table.schema_id);

    // Once a record cannot be committed (failed with no durable dead letter
    // landing) the source offset must not advance past it, so every later
    // record in the batch is also held back even if it would apply cleanly.
    // Holding the offset preserves source ordering on the next sweep.
    let mut blocked = false;

    for raw in raw_records {
        // A record is "lost" when it failed and could not be dead-lettered. It
        // blocks the offset and increments the dead-letter-error count.
        macro_rules! fail_record {
            ($msg:expr) => {{
                report.failed += 1;
                let durable = match config.dead_letter_table_id {
                    Some(dlq) => route_to_dead_letter(server, dlq, raw, $msg, &config.name).await,
                    None => false,
                };
                if durable {
                    report.dead_lettered += 1;
                    if !blocked {
                        report.committed += 1;
                    }
                } else {
                    report.dead_letter_errors += 1;
                    blocked = true;
                }
                continue;
            }};
        }

        let change = match decoder.deserialize(raw) {
            Ok(c) => c,
            Err(e) => fail_record!(&e.to_string()),
        };

        // Build the statements this record applies.
        let stmts: Vec<Statement> = match change.operation {
            ChangeType::Insert | ChangeType::UpdatePostimage => {
                let Some(new_values) = change.new_values.as_ref() else {
                    fail_record!("change carries no row image");
                };
                let Some(insert) = build_ingest_insert(&target, &table, new_values) else {
                    fail_record!("no column matched the target");
                };

                match config.on_conflict {
                    // With no declared key there is nothing to conflict on, so
                    // every record appends regardless of the conflict mode.
                    _ if config.primary_key_columns.is_empty() => vec![insert],
                    zyron_cdc::cdc_ingest::OnConflict::Skip => {
                        let keys = ingest_key_pairs(&config.primary_key_columns, new_values);
                        if !keys.is_empty()
                            && ingest_key_exists(
                                server,
                                db_id,
                                &search_path,
                                &target,
                                &table,
                                &keys,
                            )
                            .await
                        {
                            report.skipped += 1;
                            if !blocked {
                                report.committed += 1;
                            }
                            continue;
                        }
                        vec![insert]
                    }
                    zyron_cdc::cdc_ingest::OnConflict::Update => {
                        // Upsert: delete the old key (post-image carries the
                        // pre-image so primary-key changes are handled), then
                        // insert the new row.
                        let del_source = change
                            .old_values
                            .as_ref()
                            .filter(|v| !v.is_empty())
                            .unwrap_or(new_values);
                        let keys = ingest_key_pairs(&config.primary_key_columns, del_source);
                        match build_ingest_delete(&target, &table, &keys) {
                            Some(delete) => vec![delete, insert],
                            None => vec![insert],
                        }
                    }
                    zyron_cdc::cdc_ingest::OnConflict::Error => {
                        // Reject a duplicate key rather than silently appending
                        // a second row on a target without a unique index.
                        let keys = ingest_key_pairs(&config.primary_key_columns, new_values);
                        if !keys.is_empty()
                            && ingest_key_exists(
                                server,
                                db_id,
                                &search_path,
                                &target,
                                &table,
                                &keys,
                            )
                            .await
                        {
                            fail_record!("duplicate key on ON CONFLICT ERROR ingest");
                        }
                        vec![insert]
                    }
                }
            }
            ChangeType::Delete | ChangeType::UpdatePreimage => {
                let source = change.old_values.as_ref().or(change.new_values.as_ref());
                let Some(source) = source else {
                    fail_record!("delete carries no key image");
                };
                let keys = ingest_key_pairs(&config.primary_key_columns, source);
                match build_ingest_delete(&target, &table, &keys) {
                    Some(delete) => vec![delete],
                    None => {
                        fail_record!("delete has no usable key");
                    }
                }
            }
            ChangeType::Truncate => {
                vec![Statement::Delete(Box::new(DeleteStatement {
                    table: target.clone(),
                    where_clause: None,
                    returning: None,
                    hard: true,
                }))]
            }
            ChangeType::SchemaChange => {
                // Schema markers carry no row action.
                report.applied += 1;
                if !blocked {
                    report.committed += 1;
                }
                continue;
            }
        };

        match run_pipeline_write_txn(server, db_id, search_path.clone(), stmts).await {
            Ok(_) => {
                report.applied += 1;
                if !blocked {
                    report.committed += 1;
                }
            }
            Err(e) => fail_record!(&format!("{e:?}")),
        }
    }

    report
}

/// Outcome of one ingest sweep across all active jobs.
#[derive(Debug, Clone, Copy, Default)]
pub struct IngestRunReport {
    pub applied: usize,
    pub skipped: usize,
    pub failed: usize,
}

/// Polls every active CDC ingest job for new source records, applies them to
/// the target, and advances the job's checkpoint. Kafka jobs fetch from the
/// stored offset (or the configured start when uncheckpointed); S3 jobs read
/// objects sorted after the last processed key. Called by the background ingest
/// worker on each tick. Network or apply failures on one job are logged and the
/// sweep moves to the next job.
pub async fn run_due_ingests(server: &Arc<ServerState>) -> IngestRunReport {
    use zyron_cdc::cdc_ingest::{CdcIngestSource, IngestCheckpoint};
    use zyron_cdc::source_io::{KafkaStart, kafka_consume, kafka_start_offset, s3_consume_objects};

    let mut report = IngestRunReport::default();
    let Some(mgr) = server.cdc_ingest_manager.as_ref() else {
        return report;
    };

    for config in mgr.list_ingests() {
        if !config.active {
            continue;
        }
        let cp = mgr.get_checkpoint(&config.name);
        let prior_applied = cp.as_ref().map(|c| c.records_applied).unwrap_or(0);
        let prior_failed = cp.as_ref().map(|c| c.records_failed).unwrap_or(0);
        let prior_dead_lettered = cp.as_ref().map(|c| c.dead_letter_count).unwrap_or(0);

        match &config.source {
            CdcIngestSource::Kafka {
                brokers,
                topic,
                start_offset,
                ..
            } => {
                let offset = match cp
                    .as_ref()
                    .and_then(|c| c.last_source_offset.parse::<i64>().ok())
                {
                    Some(o) => o,
                    None => {
                        let start = KafkaStart::from_option(start_offset.as_deref());
                        match kafka_start_offset(brokers, topic, start) {
                            Ok(o) => o,
                            Err(e) => {
                                tracing::warn!(target: "zyron::cdc", ingest = %config.name, "ingest start offset failed: {e}");
                                continue;
                            }
                        }
                    }
                };
                // Budget roughly 4 KiB per record so a batch fetch stays bounded.
                let max_bytes =
                    (config.batch_size.saturating_mul(4096)).min(i32::MAX as usize) as i32;
                let (records, next) = match kafka_consume(brokers, topic, offset, max_bytes) {
                    Ok(r) => r,
                    Err(e) => {
                        tracing::warn!(target: "zyron::cdc", ingest = %config.name, "ingest fetch failed: {e}");
                        continue;
                    }
                };
                if records.is_empty() {
                    continue;
                }
                let r = apply_ingest_records(server, &config, &records).await;
                report.applied += r.applied;
                report.skipped += r.skipped;
                report.failed += r.failed;
                // Advance only past records that were committed (applied,
                // skipped, or durably dead-lettered). When every record
                // committed, jump to the fetch's next offset; otherwise stop at
                // the first uncommitted record so the next sweep retries it.
                let committed_offset = if r.committed == records.len() {
                    next
                } else {
                    offset + r.committed as i64
                };
                if r.dead_letter_errors > 0 {
                    tracing::warn!(
                        target: "zyron::cdc",
                        ingest = %config.name,
                        dead_letter_errors = r.dead_letter_errors,
                        "ingest held offset at uncommitted record"
                    );
                }
                if let Err(e) = mgr.update_checkpoint(IngestCheckpoint {
                    name: config.name.clone(),
                    last_source_offset: committed_offset.to_string(),
                    records_applied: prior_applied + r.applied as u64,
                    records_failed: prior_failed + r.failed as u64,
                    dead_letter_count: prior_dead_lettered + r.dead_lettered as u64,
                }) {
                    tracing::error!(target: "zyron::cdc", ingest = %config.name, "ingest checkpoint write failed: {e}");
                }
            }
            CdcIngestSource::S3 {
                bucket,
                prefix,
                region,
                ..
            } => {
                let after = cp
                    .as_ref()
                    .map(|c| c.last_source_offset.clone())
                    .unwrap_or_default();
                let objects = match s3_consume_objects(
                    bucket,
                    region,
                    prefix,
                    &after,
                    config.batch_size.min(i32::MAX as usize) as u32,
                ) {
                    Ok(o) => o,
                    Err(e) => {
                        tracing::warn!(target: "zyron::cdc", ingest = %config.name, "ingest list/get failed: {e}");
                        continue;
                    }
                };
                if objects.is_empty() {
                    continue;
                }
                let mut applied = 0u64;
                let mut failed = 0u64;
                let mut dead_lettered = 0u64;
                // Checkpoint after each object so a mid-batch failure resumes
                // from the last fully processed key. Stop advancing the key past
                // the first object that holds an uncommitted record so its data
                // is retried on the next sweep rather than skipped.
                for (key, records) in objects {
                    let r = apply_ingest_records(server, &config, &records).await;
                    report.applied += r.applied;
                    report.skipped += r.skipped;
                    report.failed += r.failed;
                    applied += r.applied as u64;
                    failed += r.failed as u64;
                    dead_lettered += r.dead_lettered as u64;
                    let fully_committed = r.committed == records.len();
                    if !fully_committed {
                        tracing::warn!(
                            target: "zyron::cdc",
                            ingest = %config.name,
                            object = %key,
                            dead_letter_errors = r.dead_letter_errors,
                            "ingest held S3 key at uncommitted record"
                        );
                        break;
                    }
                    if let Err(e) = mgr.update_checkpoint(IngestCheckpoint {
                        name: config.name.clone(),
                        last_source_offset: key,
                        records_applied: prior_applied + applied,
                        records_failed: prior_failed + failed,
                        dead_letter_count: prior_dead_lettered + dead_lettered,
                    }) {
                        tracing::error!(target: "zyron::cdc", ingest = %config.name, "ingest checkpoint write failed: {e}");
                        break;
                    }
                }
            }
        }
    }
    report
}

// ---------------------------------------------------------------------------
// GRANT/REVOKE handlers
// ---------------------------------------------------------------------------

/// Maps a parser Privilege variant to the corresponding auth PrivilegeType.
/// ALL expands to Select, Insert, Update, Delete plus the four index DDL
/// privileges so a table owner with ALL can manage indexes on the table.
fn map_privilege(p: zyron_parser::ast::Privilege) -> Vec<zyron_auth::PrivilegeType> {
    match p {
        zyron_parser::ast::Privilege::Select => vec![zyron_auth::PrivilegeType::Select],
        zyron_parser::ast::Privilege::Insert => vec![zyron_auth::PrivilegeType::Insert],
        zyron_parser::ast::Privilege::Update => vec![zyron_auth::PrivilegeType::Update],
        zyron_parser::ast::Privilege::Delete => vec![zyron_auth::PrivilegeType::Delete],
        zyron_parser::ast::Privilege::CreateIndex => {
            vec![zyron_auth::PrivilegeType::CreateIndex]
        }
        zyron_parser::ast::Privilege::DropIndex => vec![zyron_auth::PrivilegeType::DropIndex],
        zyron_parser::ast::Privilege::Reindex => vec![zyron_auth::PrivilegeType::Reindex],
        zyron_parser::ast::Privilege::AlterIndex => vec![zyron_auth::PrivilegeType::AlterIndex],
        zyron_parser::ast::Privilege::Subscribe => vec![zyron_auth::PrivilegeType::Subscribe],
        zyron_parser::ast::Privilege::Invoke => vec![zyron_auth::PrivilegeType::InvokeEndpoint],
        zyron_parser::ast::Privilege::All => vec![
            zyron_auth::PrivilegeType::Select,
            zyron_auth::PrivilegeType::Insert,
            zyron_auth::PrivilegeType::Update,
            zyron_auth::PrivilegeType::Delete,
            zyron_auth::PrivilegeType::CreateIndex,
            zyron_auth::PrivilegeType::DropIndex,
            zyron_auth::PrivilegeType::Reindex,
            zyron_auth::PrivilegeType::AlterIndex,
        ],
    }
}

async fn handle_grant(
    stmt: &zyron_parser::ast::GrantStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let sm = require_security_manager(server)?;
    // Only roles holding ManagePrivileges may grant privileges. Without this
    // any authenticated role could self-grant superuser.
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::ManagePrivileges,
        zyron_auth::ObjectType::System,
        0,
    )?;

    // Resolve the grantee role by name
    let grantee = sm
        .lookup_role(&stmt.to)
        .ok_or_else(|| ProtocolError::Database(ZyronError::RoleNotFound(stmt.to.clone())))?;

    // Resolve the target table to get its catalog ID
    let (_, schema_id) = get_session_schema(session, server, None)?;
    let table = server
        .catalog
        .get_table(schema_id, &stmt.on_table)
        .map_err(ProtocolError::Database)?;

    // The grantor is the session's current role, recorded so the privilege
    // graph attributes the grant to the actor rather than to role 0.
    let granted_by = zyron_auth::RoleId(actor_role_id(session));

    // Grant each privilege on the table
    for priv_ast in &stmt.privileges {
        let priv_types = map_privilege(*priv_ast);
        for pt in priv_types {
            let entry = zyron_auth::GrantEntry {
                grantee: grantee.id,
                privilege: pt,
                object_type: zyron_auth::ObjectType::Table,
                object_id: table.id.0,
                columns: None,
                state: zyron_auth::PrivilegeState::Grant,
                with_grant_option: stmt.with_grant_option,
                granted_by,
                valid_from: None,
                valid_until: None,
                time_window: None,
                object_pattern: None,
                no_inherit: false,
                mask_function: None,
            };
            sm.privilege_store
                .grant(entry)
                .map_err(ProtocolError::Database)?;
        }
    }

    tracing::info!(
        target: "zyron::audit",
        event = "PrivilegeGranted",
        grantee = %stmt.to,
        object = %stmt.on_table,
        actor_role = granted_by.0,
    );
    Ok(DdlResult::Tag("GRANT".to_string()))
}

async fn handle_revoke(
    stmt: &zyron_parser::ast::RevokeStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let sm = require_security_manager(server)?;
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::ManagePrivileges,
        zyron_auth::ObjectType::System,
        0,
    )?;

    // Resolve the grantee role by name
    let grantee = sm
        .lookup_role(&stmt.from)
        .ok_or_else(|| ProtocolError::Database(ZyronError::RoleNotFound(stmt.from.clone())))?;

    // Resolve the target table to get its catalog ID
    let (_, schema_id) = get_session_schema(session, server, None)?;
    let table = server
        .catalog
        .get_table(schema_id, &stmt.on_table)
        .map_err(ProtocolError::Database)?;

    // Revoke each privilege on the table
    for priv_ast in &stmt.privileges {
        let priv_types = map_privilege(*priv_ast);
        for pt in priv_types {
            sm.privilege_store
                .revoke(grantee.id, pt, zyron_auth::ObjectType::Table, table.id.0);
        }
    }

    tracing::info!(
        target: "zyron::audit",
        event = "PrivilegeRevoked",
        grantee = %stmt.from,
        object = %stmt.on_table,
        actor_role = actor_role_id(session),
    );
    Ok(DdlResult::Tag("REVOKE".to_string()))
}

// ---------------------------------------------------------------------------
// Transaction extension handlers
// ---------------------------------------------------------------------------

fn handle_savepoint(
    stmt: &zyron_parser::ast::SavepointStatement,
    txn: &mut Option<zyron_storage::txn::Transaction>,
    server: &Arc<ServerState>,
) -> Result<DdlResult, ProtocolError> {
    let txn = txn.as_mut().ok_or_else(|| {
        ProtocolError::Database(ZyronError::TransactionAborted(
            "SAVEPOINT can only be used in a transaction".to_string(),
        ))
    })?;
    // Capture the row and intent lock counts held now so ROLLBACK TO this
    // savepoint releases exactly the locks acquired after it. Both tables are
    // keyed by txn id and track per-txn acquisition order.
    let row_lock_count = server.txn_manager.lock_table().current_count(txn.txn_id());
    let intent_lock_count = server
        .txn_manager
        .intent_locks()
        .current_count(txn.txn_id());
    txn.savepoint(stmt.name.clone(), row_lock_count, intent_lock_count);
    Ok(DdlResult::Tag("SAVEPOINT".to_string()))
}

fn handle_release_savepoint(
    stmt: &zyron_parser::ast::ReleaseSavepointStatement,
    txn: &mut Option<zyron_storage::txn::Transaction>,
) -> Result<DdlResult, ProtocolError> {
    let txn = txn.as_mut().ok_or_else(|| {
        ProtocolError::Database(ZyronError::TransactionAborted(
            "RELEASE SAVEPOINT can only be used in a transaction".to_string(),
        ))
    })?;
    if txn.release_savepoint(&stmt.name) {
        Ok(DdlResult::Tag("RELEASE".to_string()))
    } else {
        Err(ProtocolError::Database(ZyronError::TransactionAborted(
            format!("savepoint \"{}\" does not exist", stmt.name),
        )))
    }
}

// ---------------------------------------------------------------------------
// VALUES query handler
// ---------------------------------------------------------------------------

async fn handle_values_query(
    stmt: &zyron_parser::ast::ValuesQueryStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    if stmt.rows.is_empty() {
        return Ok(DdlResult::Tag("SELECT 0".to_string()));
    }

    let num_cols = stmt.rows[0].len();
    for (i, row) in stmt.rows.iter().enumerate() {
        if row.len() != num_cols {
            return Err(ProtocolError::Database(ZyronError::PlanError(format!(
                "VALUES row {} has {} columns, expected {}",
                i + 1,
                row.len(),
                num_cols
            ))));
        }
    }

    // Evaluate the rows by running them through the planner and executor so
    // every expression form (arithmetic, function calls, casts) resolves to its
    // computed value. Each row becomes a SELECT of its expressions, joined with
    // UNION ALL so a single plan computes the whole result.
    let selects: Vec<String> = stmt
        .rows
        .iter()
        .map(|row| {
            let cells: Vec<String> = row
                .iter()
                .enumerate()
                .map(|(i, e)| format!("{} AS column{}", zyron_parser::expr_to_sql(e), i + 1))
                .collect();
            format!("SELECT {}", cells.join(", "))
        })
        .collect();
    let sql = selects.join(" UNION ALL ");

    let db_id = get_session_database(session)?;
    let search_path = session
        .as_ref()
        .map(|s| s.search_path.clone())
        .unwrap_or_default();
    let select_stmt = zyron_parser::parse(&sql)
        .map_err(ProtocolError::Database)?
        .into_iter()
        .next()
        .ok_or_else(|| {
            ProtocolError::Database(ZyronError::Internal("empty VALUES query".to_string()))
        })?;
    let (schema, batches) = run_pipeline_read(server, db_id, search_path, select_stmt).await?;

    let columns: Vec<(String, i32)> = (0..num_cols)
        .map(|i| (format!("column{}", i + 1), crate::types::PG_TEXT_OID))
        .collect();

    let mut rows = Vec::new();
    for b in &batches {
        for r in 0..b.num_rows {
            let mut row_values = Vec::with_capacity(schema.len());
            for col in &b.columns {
                let scalar = col.get_scalar(r);
                let mut buf = bytes::BytesMut::with_capacity(32);
                let value = if crate::types::scalar_write_text(&scalar, &mut buf) {
                    String::from_utf8_lossy(&buf).into_owned()
                } else {
                    String::new()
                };
                row_values.push(value);
            }
            rows.push(row_values);
        }
    }

    Ok(DdlResult::Rows {
        tag: format!("SELECT {}", rows.len()),
        columns,
        rows,
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Gets the database ID and the active schema ID from the session.
///
/// Zyron has no default user schema, so this returns an error when the
/// session's search_path is empty. Callers must either qualify DDL
/// identifiers or have the client set a search_path first.
pub(crate) fn get_session_schema(
    session: &Option<Session>,
    server: &Arc<ServerState>,
    _override_schema: Option<&str>,
) -> Result<(zyron_catalog::DatabaseId, zyron_catalog::SchemaId), ProtocolError> {
    let session = session
        .as_ref()
        .ok_or(ProtocolError::Malformed("no active session".into()))?;
    let db_id = session.database_id;
    let schema_name = session
        .search_path
        .first()
        .map(|s| s.as_str())
        .ok_or_else(|| {
            ProtocolError::Malformed(
                "no target schema: session search_path is empty. \
             Qualify the object as `schema.name` or run `SET search_path = your_schema` first."
                    .into(),
            )
        })?;

    let schema = server
        .catalog
        .get_schema(db_id, schema_name)
        .map_err(ProtocolError::Database)?;

    Ok((db_id, schema.id))
}

/// Returns the session's database id and search_path for executing procedure,
/// DO, and trigger bodies in the caller's namespace. Falls back to the default
/// database and the public schema when there is no session (background
/// dispatch).
fn session_db_and_search_path(
    session: &Option<Session>,
) -> (zyron_catalog::DatabaseId, Vec<String>) {
    match session.as_ref() {
        Some(s) => {
            let path = if s.search_path.is_empty() {
                vec!["public".to_string()]
            } else {
                s.search_path.clone()
            };
            (s.database_id, path)
        }
        None => (zyron_catalog::DatabaseId(1), vec!["public".to_string()]),
    }
}

/// Returns the database ID bound to the active session.
///
/// Used by DDL handlers that operate at the database scope (CREATE SCHEMA,
/// DROP SCHEMA, streaming-job dispatch, etc.) and therefore do not need a
/// target schema in the session's search_path.
fn get_session_database(
    session: &Option<Session>,
) -> Result<zyron_catalog::DatabaseId, ProtocolError> {
    let session = session
        .as_ref()
        .ok_or(ProtocolError::Malformed("no active session".into()))?;
    Ok(session.database_id)
}

async fn handle_create_fulltext_index(
    stmt: &zyron_parser::ast::CreateFulltextIndexStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    // Resolve the schema from the session search_path.
    let (_, schema_id) = get_session_schema(session, server, None)?;

    let table = server
        .catalog
        .get_table(schema_id, &stmt.table)
        .map_err(ProtocolError::Database)?;

    // Privilege check: require CREATE on the table (index is table-scoped)
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Table,
        table.id.0,
    )?;

    // Register index in the catalog with IndexType::Fulltext
    let index_id = server
        .catalog
        .create_index(
            table.id,
            schema_id,
            &stmt.name,
            &stmt.columns,
            false,
            zyron_catalog::IndexType::Fulltext,
        )
        .await
        .map_err(ProtocolError::Database)?;

    // Create live FTS index via the FTS manager if available.
    // On failure, roll back the catalog entry to prevent orphaned metadata.
    if let Some(ref fts_mgr) = server.fts_manager {
        let col_ids: Vec<u16> = stmt
            .columns
            .iter()
            .filter_map(|name| {
                table
                    .columns
                    .iter()
                    .find(|c| c.name == *name)
                    .map(|c| c.id.0)
            })
            .collect();
        if let Err(e) = fts_mgr.create_index(index_id.0, table.id.0, col_ids) {
            let _ = server.catalog.drop_index(table.id, &stmt.name).await;
            return Err(ProtocolError::Database(e));
        }
    }

    Ok(DdlResult::Tag("CREATE INDEX".to_string()))
}

async fn handle_create_vector_index(
    stmt: &zyron_parser::ast::CreateVectorIndexStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (_, schema_id) = get_session_schema(session, server, None)?;

    let table = server
        .catalog
        .get_table(schema_id, &stmt.table)
        .map_err(ProtocolError::Database)?;

    // Privilege check: require CREATE on the table
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Table,
        table.id.0,
    )?;

    // Parse distance metric and index parameters from options. A malformed
    // numeric option is rejected rather than silently substituting a default,
    // so a typo never builds an index with the wrong graph parameters.
    let mut distance_metric = "cosine".to_string();
    let mut m: u16 = 16;
    let mut ef_construction: u16 = 200;
    let mut dim_option: Option<u16> = None;
    for opt in &stmt.options {
        let key = opt.key.to_lowercase();
        let val_str = match &opt.value {
            zyron_parser::ast::TableOptionValue::String(s) => s.to_lowercase(),
            zyron_parser::ast::TableOptionValue::Identifier(s) => s.to_lowercase(),
            zyron_parser::ast::TableOptionValue::Integer(n) => n.to_string(),
            zyron_parser::ast::TableOptionValue::Boolean(b) => b.to_string(),
            zyron_parser::ast::TableOptionValue::StringList(_) => String::new(),
        };
        match key.as_str() {
            "distance_metric" => distance_metric = val_str,
            "m" => {
                m = val_str.parse().map_err(|_| {
                    ProtocolError::Database(ZyronError::ExecutionError(format!(
                        "vector index option 'm' must be an integer, got '{val_str}'"
                    )))
                })?;
            }
            "ef_construction" => {
                ef_construction = val_str.parse().map_err(|_| {
                    ProtocolError::Database(ZyronError::ExecutionError(format!(
                        "vector index option 'ef_construction' must be an integer, got '{val_str}'"
                    )))
                })?;
            }
            "dimensions" | "dim" | "dimension" => {
                dim_option = Some(val_str.parse().map_err(|_| {
                    ProtocolError::Database(ZyronError::ExecutionError(format!(
                        "vector index option 'dimensions' must be an integer, got '{val_str}'"
                    )))
                })?);
            }
            _ => {}
        }
    }

    // Find the column to determine dimensions.
    let col = table
        .columns
        .iter()
        .find(|c| c.name == stmt.column)
        .ok_or_else(|| {
            ProtocolError::Database(ZyronError::ExecutionError(format!(
                "column '{}' not found in table '{}'",
                stmt.column, stmt.table
            )))
        })?;

    // Dimension must be explicit: a DIMENSIONS option, or the column's declared
    // length. Defaulting to a fixed value would build an index that mismatches
    // the stored vectors, so an absent dimension is an error.
    let dimensions = dim_option
        .or_else(|| col.max_length.map(|l| l as u16))
        .filter(|d| *d > 0)
        .ok_or_else(|| {
            ProtocolError::Database(ZyronError::ExecutionError(format!(
                "vector index on '{}' requires an explicit dimension; declare the column length \
                 or pass WITH (dimensions = N)",
                stmt.column
            )))
        })?;

    // Register in catalog with IndexType::Vector
    let index_id = server
        .catalog
        .create_index(
            table.id,
            schema_id,
            &stmt.name,
            &[stmt.column.clone()],
            false,
            zyron_catalog::IndexType::Vector,
        )
        .await
        .map_err(ProtocolError::Database)?;

    // Create live vector index via the vector manager if available
    if let Some(ref vec_mgr) = server.vector_manager {
        let metric = match distance_metric.as_str() {
            "euclidean" | "l2" => zyron_search::vector::DistanceMetric::Euclidean,
            "dot_product" | "dot" => zyron_search::vector::DistanceMetric::DotProduct,
            "manhattan" | "l1" => zyron_search::vector::DistanceMetric::Manhattan,
            _ => zyron_search::vector::DistanceMetric::Cosine,
        };
        let config = zyron_search::vector::HnswConfig {
            m,
            efConstruction: ef_construction,
            efSearch: 64,
            metric,
        };
        if let Err(e) = vec_mgr.create_index(index_id.0, table.id.0, col.id.0, dimensions, config) {
            let _ = server.catalog.drop_index(table.id, &stmt.name).await;
            return Err(ProtocolError::Database(e));
        }
    }

    Ok(DdlResult::Tag("CREATE INDEX".to_string()))
}

async fn handle_create_spatial_index(
    stmt: &zyron_parser::ast::CreateSpatialIndexStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (_, schema_id) = get_session_schema(session, server, None)?;

    let table = server
        .catalog
        .get_table(schema_id, &stmt.table)
        .map_err(ProtocolError::Database)?;

    // IF NOT EXISTS: short-circuit if an index of this name already exists.
    if stmt.if_not_exists
        && server
            .catalog
            .get_indexes_for_table(table.id)
            .iter()
            .any(|idx| idx.name == stmt.name)
    {
        return Ok(DdlResult::Tag("CREATE SPATIAL INDEX".to_string()));
    }

    // Privilege check: dedicated CreateIndex privilege on the table.
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::CreateIndex,
        zyron_auth::ObjectType::Table,
        table.id.0,
    )?;

    // Parse spatial-specific tuning options.
    let mut dims: u8 = 2;
    let mut srid: u32 = 4326;
    for opt in &stmt.options {
        let key = opt.key.to_lowercase();
        let val_str = match &opt.value {
            zyron_parser::ast::TableOptionValue::String(s) => s.clone(),
            zyron_parser::ast::TableOptionValue::Identifier(s) => s.clone(),
            zyron_parser::ast::TableOptionValue::Integer(n) => n.to_string(),
            zyron_parser::ast::TableOptionValue::Boolean(b) => b.to_string(),
            zyron_parser::ast::TableOptionValue::StringList(_) => String::new(),
        };
        match key.as_str() {
            "dims" | "dimensions" => {
                let parsed: u8 = val_str.parse().unwrap_or(2);
                if parsed < 1 || parsed > 4 {
                    return Err(ProtocolError::Database(ZyronError::ExecutionError(
                        format!("spatial index dims must be 1..=4, got {}", parsed),
                    )));
                }
                dims = parsed;
            }
            "srid" => {
                srid = val_str.parse().unwrap_or(4326);
            }
            _ => {}
        }
    }

    // Verify the indexed column exists.
    let _col = table
        .columns
        .iter()
        .find(|c| c.name == stmt.column)
        .ok_or_else(|| {
            ProtocolError::Database(ZyronError::ExecutionError(format!(
                "column '{}' not found in table '{}'",
                stmt.column, stmt.table
            )))
        })?;

    // Encode dims and srid into the index parameters blob so startup
    // recovery can reconstruct the live R-tree without re-parsing CREATE.
    // Layout: [u8 dims][u32 srid little-endian].
    let mut params = Vec::with_capacity(5);
    params.push(dims);
    params.extend_from_slice(&srid.to_le_bytes());

    // Register in catalog.
    let index_id = server
        .catalog
        .create_index_with_params(
            table.id,
            schema_id,
            &stmt.name,
            &[stmt.column.clone()],
            false,
            zyron_catalog::IndexType::Spatial,
            Some(params),
        )
        .await
        .map_err(ProtocolError::Database)?;

    // Create the live R-tree if a spatial manager is configured.
    if let Some(ref spatial_mgr) = server.spatial_manager {
        spatial_mgr.create_index(index_id.0, dims, srid);
    }

    Ok(DdlResult::Tag("CREATE SPATIAL INDEX".to_string()))
}

// ---------------------------------------------------------------------------
// Graph schema DDL
// ---------------------------------------------------------------------------

/// Drops backing tables created for a graph schema, reclaiming their files.
/// Best-effort cleanup used to roll back a partial CREATE GRAPH SCHEMA.
async fn rollback_graph_tables(
    server: &Arc<ServerState>,
    schema_id: zyron_catalog::SchemaId,
    names: &[String],
) {
    for name in names {
        if let Ok(outcome) = server.catalog.drop_table(schema_id, name).await {
            if !outcome.soft_dropped {
                let _ = server.heap_files.remove_async(&outcome.heap_file_id).await;
                let _ = server.disk_manager.delete_file(outcome.heap_file_id).await;
                let _ = server.disk_manager.delete_file(outcome.fsm_file_id).await;
            }
        }
    }
}

async fn handle_create_graph_schema(
    stmt: &zyron_parser::ast::CreateGraphSchemaStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    use zyron_parser::ast::{ColumnDef, DataType, GraphSchemaElement};

    let graph_mgr = server.graph_manager.clone().ok_or_else(|| {
        ProtocolError::Database(ZyronError::GraphSchemaNotFound(
            "graph manager not configured".to_string(),
        ))
    })?;

    if graph_mgr.get_schema(&stmt.name).is_some() {
        if stmt.if_not_exists {
            return Ok(DdlResult::Tag("CREATE GRAPH SCHEMA".to_string()));
        }
        return Err(ProtocolError::Database(ZyronError::GraphQueryError(
            format!("graph schema '{}' already exists", stmt.name),
        )));
    }

    let (_, schema_id) = get_session_schema(session, server, None)?;
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Schema,
        schema_id.0,
    )?;

    // A node/edge identity column: a non-null BigInt holding a node id.
    let id_column = |name: &str| ColumnDef {
        name: name.to_string(),
        data_type: DataType::BigInt,
        nullable: Some(false),
        default: None,
        constraints: Vec::new(),
    };
    let to_props = |properties: &[ColumnDef]| -> Vec<zyron_search::graph::PropertyDef> {
        properties
            .iter()
            .map(|col| zyron_search::graph::PropertyDef {
                name: col.name.clone(),
                type_id: col.data_type.to_type_id(),
                nullable: col.nullable.unwrap_or(true),
            })
            .collect()
    };

    let schema_oid = server.catalog.next_oid();
    let mut schema = zyron_search::graph::GraphSchema::new(stmt.name.clone(), schema_oid);
    // Backing tables created so far, dropped if a later step fails.
    let mut created: Vec<String> = Vec::new();

    // First pass: a backing table per node label (node_id + properties), so
    // edge labels below can resolve their endpoint labels.
    for elem in &stmt.elements {
        if let GraphSchemaElement::Node { label, properties } = elem {
            let table_name = format!("{}_{}", stmt.name, label);
            let mut columns = Vec::with_capacity(properties.len() + 1);
            columns.push(id_column("node_id"));
            columns.extend(properties.iter().cloned());
            let table_id = match server
                .catalog
                .create_table(schema_id, &table_name, &columns, &[])
                .await
            {
                Ok(id) => id,
                Err(e) => {
                    rollback_graph_tables(server, schema_id, &created).await;
                    return Err(ProtocolError::Database(e));
                }
            };
            created.push(table_name);
            schema.add_node_label(label.clone(), to_props(properties), table_id.0);
        }
    }

    // Second pass: a backing table per edge label (from_node, to_node +
    // properties) with resolved endpoint label ids.
    for elem in &stmt.elements {
        if let GraphSchemaElement::Edge {
            label,
            from_label,
            to_label,
            properties,
        } = elem
        {
            let resolve = |name: &str| schema.get_node_label(name).map(|nl| nl.label_id);
            let (Some(from_id), Some(to_id)) = (resolve(from_label), resolve(to_label)) else {
                rollback_graph_tables(server, schema_id, &created).await;
                return Err(ProtocolError::Database(ZyronError::GraphQueryError(
                    format!("edge label '{}' references an undefined node label", label),
                )));
            };
            let table_name = format!("{}_{}", stmt.name, label);
            let mut columns = Vec::with_capacity(properties.len() + 2);
            columns.push(id_column("from_node"));
            columns.push(id_column("to_node"));
            columns.extend(properties.iter().cloned());
            let table_id = match server
                .catalog
                .create_table(schema_id, &table_name, &columns, &[])
                .await
            {
                Ok(id) => id,
                Err(e) => {
                    rollback_graph_tables(server, schema_id, &created).await;
                    return Err(ProtocolError::Database(e));
                }
            };
            created.push(table_name);
            if let Err(e) = schema.add_edge_label(
                label.clone(),
                from_id,
                to_id,
                to_props(properties),
                table_id.0,
                true,
            ) {
                rollback_graph_tables(server, schema_id, &created).await;
                return Err(ProtocolError::Database(e));
            }
        }
    }

    if let Err(e) = graph_mgr.create_schema(schema) {
        rollback_graph_tables(server, schema_id, &created).await;
        return Err(ProtocolError::Database(e));
    }

    // Persist to disk immediately so the schema survives restarts. If the
    // write fails, roll back both the in-memory schema and its backing tables.
    let graph_dir = server.data_dir.join("graph");
    if let Err(e) = graph_mgr.save_all(&graph_dir) {
        let _ = graph_mgr.drop_schema(&stmt.name);
        rollback_graph_tables(server, schema_id, &created).await;
        return Err(ProtocolError::Database(e));
    }

    Ok(DdlResult::Tag("CREATE GRAPH SCHEMA".to_string()))
}

async fn handle_drop_graph_schema(
    stmt: &zyron_parser::ast::DropGraphSchemaStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let graph_mgr = server.graph_manager.clone().ok_or_else(|| {
        ProtocolError::Database(ZyronError::GraphSchemaNotFound(
            "graph manager not configured".to_string(),
        ))
    })?;

    // Resolve the session schema before the irreversible drop so a missing
    // search_path fails the statement instead of leaving the backing tables
    // orphaned after the schema is already gone.
    let (_, schema_id) = get_session_schema(session, server, None)?;
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Schema,
        schema_id.0,
    )?;

    // Collect the backing-table names from the schema before dropping it.
    let backing: Vec<String> = match graph_mgr.get_schema(&stmt.name) {
        Some(schema) => schema
            .node_labels
            .iter()
            .map(|n| format!("{}_{}", stmt.name, n.name))
            .chain(
                schema
                    .edge_labels
                    .iter()
                    .map(|e| format!("{}_{}", stmt.name, e.name)),
            )
            .collect(),
        None => Vec::new(),
    };

    match graph_mgr.drop_schema(&stmt.name) {
        Ok(()) => {
            let graph_dir = server.data_dir.join("graph");
            graph_mgr
                .save_all(&graph_dir)
                .map_err(ProtocolError::Database)?;
            // Drop the backing tables and reclaim their files.
            rollback_graph_tables(server, schema_id, &backing).await;
            Ok(DdlResult::Tag("DROP GRAPH SCHEMA".to_string()))
        }
        Err(_) if stmt.if_exists => Ok(DdlResult::Tag("DROP GRAPH SCHEMA".to_string())),
        Err(e) => Err(ProtocolError::Database(e)),
    }
}

/// Returns a reference to the SecurityManager or an error.
fn require_security_manager(
    server: &Arc<ServerState>,
) -> Result<&zyron_auth::SecurityManager, ProtocolError> {
    server.security_manager.as_deref().ok_or_else(|| {
        ProtocolError::Database(ZyronError::AuthenticationFailed(
            "security manager not configured".to_string(),
        ))
    })
}

// ---------------------------------------------------------------------------
// Streaming job dispatch
// ---------------------------------------------------------------------------

/// Binds a streaming-job statement via the planner binder and dispatches it to
/// the matching create/drop/alter handler. Non-streaming statements are
/// rejected with an internal error because this entry point is only invoked
/// for the three streaming job variants.
async fn dispatch_streaming_statement(
    stmt: zyron_parser::Statement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
    raw_sql: &str,
) -> Result<DdlResult, ProtocolError> {
    let db_id = get_session_database(session)?;
    let search_path = session
        .as_ref()
        .map(|s| s.search_path.clone())
        .unwrap_or_default();

    let resolver = server.catalog.resolver(db_id, search_path);
    let mut binder = zyron_planner::Binder::new(resolver, &server.catalog);
    let bound = binder.bind(stmt).await.map_err(ProtocolError::Database)?;

    match bound {
        zyron_planner::BoundStatement::CreateStreamingJob(bsj) => {
            handle_create_streaming_job(bsj, server, session, raw_sql).await
        }
        zyron_planner::BoundStatement::DropStreamingJob { name, if_exists } => {
            handle_drop_streaming_job(&name, if_exists, server, session).await
        }
        zyron_planner::BoundStatement::AlterStreamingJob { name, action } => {
            handle_alter_streaming_job(&name, action, server, session).await
        }
        _ => Err(ProtocolError::Database(ZyronError::PlanError(
            "expected streaming job statement".to_string(),
        ))),
    }
}

/// Lowers a single BoundExpr into the runner's ExprSpec. Only literals, column
/// references, a fixed set of binary ops, and unary NOT are supported by the
/// streaming evaluator. Anything else returns a PlanError so the creator sees
/// a precise reason at CREATE time rather than a runner failure at Paused time.
pub fn lower_expr(
    e: &zyron_planner::binder::BoundExpr,
    source_columns: &[zyron_catalog::schema::ColumnEntry],
) -> Result<zyron_streaming::job_runner::ExprSpec, ProtocolError> {
    use zyron_parser::ast::{BinaryOperator, LiteralValue, UnaryOperator};
    use zyron_planner::binder::BoundExpr;
    use zyron_streaming::job_runner::{BinaryOpKind, ExprSpec};

    match e {
        BoundExpr::ColumnRef(cr) => {
            let ordinal = source_columns
                .iter()
                .find(|c| c.id == cr.column_id)
                .map(|c| c.ordinal)
                .ok_or_else(|| {
                    ProtocolError::Database(ZyronError::PlanError(format!(
                        "streaming job references unknown source column id {:?}",
                        cr.column_id
                    )))
                })?;
            Ok(ExprSpec::ColumnRef { ordinal })
        }
        BoundExpr::Literal { value, .. } => match value {
            LiteralValue::Integer(n) => Ok(ExprSpec::LiteralI64(*n)),
            LiteralValue::Float(f) => Ok(ExprSpec::LiteralF64(*f)),
            LiteralValue::String(s) => Ok(ExprSpec::LiteralString(s.clone())),
            LiteralValue::Boolean(b) => Ok(ExprSpec::LiteralBool(*b)),
            _ => Err(ProtocolError::Database(ZyronError::PlanError(
                "streaming job expression must be a column reference, literal, unary NOT, or a binary op over those".to_string(),
            ))),
        },
        BoundExpr::BinaryOp { left, op, right, .. } => {
            let op_kind = match op {
                BinaryOperator::Eq => BinaryOpKind::Eq,
                BinaryOperator::Neq => BinaryOpKind::NotEq,
                BinaryOperator::Lt => BinaryOpKind::Lt,
                BinaryOperator::LtEq => BinaryOpKind::LtEq,
                BinaryOperator::Gt => BinaryOpKind::Gt,
                BinaryOperator::GtEq => BinaryOpKind::GtEq,
                BinaryOperator::And => BinaryOpKind::And,
                BinaryOperator::Or => BinaryOpKind::Or,
                BinaryOperator::Plus => BinaryOpKind::Add,
                BinaryOperator::Minus => BinaryOpKind::Sub,
                BinaryOperator::Multiply => BinaryOpKind::Mul,
                BinaryOperator::Divide => BinaryOpKind::Div,
                _ => {
                    return Err(ProtocolError::Database(ZyronError::PlanError(
                        "streaming job expression must be a column reference, literal, unary NOT, or a binary op over those".to_string(),
                    )));
                }
            };
            Ok(ExprSpec::BinaryOp {
                op: op_kind,
                left: Box::new(lower_expr(left, source_columns)?),
                right: Box::new(lower_expr(right, source_columns)?),
            })
        }
        BoundExpr::UnaryOp { op: UnaryOperator::Not, expr, .. } => {
            Ok(ExprSpec::Not(Box::new(lower_expr(expr, source_columns)?)))
        }
        BoundExpr::Nested(inner) => lower_expr(inner, source_columns),
        _ => Err(ProtocolError::Database(ZyronError::PlanError(
            "streaming job expression must be a column reference, literal, unary NOT, or a binary op over those".to_string(),
        ))),
    }
}

/// Lowers a bound streaming-job plan into the runner StreamingJobSpec. Returns
/// ZyronError so callers outside the wire crate (for example the startup
/// recovery path in zyron-server) can use it without pulling ProtocolError in.
pub fn lower_bsj_to_spec(
    bsj: &zyron_planner::BoundStreamingJob,
) -> zyron_common::Result<zyron_streaming::job_runner::StreamingJobSpec> {
    // Handles every topology. ZyronTable endpoints carry a real table id,
    // external endpoints carry TableId(0) because the runner does not use
    // a table id on external-facing directions. Source and target column
    // lists come from the bound plan in both cases.
    let src_cols = bsj.source_columns();
    let tgt_cols = bsj.target_columns();
    let src_table_id = bsj.source_table_id().unwrap_or(zyron_catalog::TableId(0));
    let tgt_table_id = bsj.target_table_id().unwrap_or(zyron_catalog::TableId(0));
    let mut projections = Vec::with_capacity(bsj.projections.len());
    for proj in &bsj.projections {
        let lowered = lower_expr(proj, src_cols).map_err(|e| match e {
            ProtocolError::Database(err) => err,
            other => ZyronError::PlanError(format!("streaming job lowering failed: {other}")),
        })?;
        projections.push(lowered);
    }
    let predicate = match &bsj.predicate {
        Some(p) => {
            let lowered = lower_expr(p, src_cols).map_err(|e| match e {
                ProtocolError::Database(err) => err,
                other => ZyronError::PlanError(format!("streaming job lowering failed: {other}")),
            })?;
            Some(lowered)
        }
        None => None,
    };
    let source_types = src_cols.iter().map(|c| c.type_id).collect();
    let target_types = tgt_cols.iter().map(|c| c.type_id).collect();
    // Resolve PK ColumnIds to target column ordinals so the runner's upsert
    // sink can index into decoded rows without another catalog lookup. Empty
    // when the write mode is Append or the target is external.
    let target_pk_ordinals: Vec<u16> = bsj
        .target_pk_columns
        .iter()
        .filter_map(|col_id| tgt_cols.iter().find(|c| c.id == *col_id).map(|c| c.ordinal))
        .collect();
    let aggregate = build_aggregate_spec(bsj, src_cols)?;
    let join = build_join_spec(bsj)?;
    Ok(zyron_streaming::job_runner::StreamingJobSpec {
        source_table_id: src_table_id.0,
        target_table_id: tgt_table_id.0,
        write_mode: bsj.write_mode,
        projections,
        predicate,
        source_types,
        target_types,
        target_pk_ordinals,
        aggregate,
        join,
    })
}

/// Lowers the bound join spec into the runner JoinSpec. Returns Ok(None)
/// when the bound job has no join section (pure filter+project or
/// aggregating topologies). Interval joins require both sides to carry
/// column types, which are read from the bound plan.
fn build_join_spec(
    bsj: &zyron_planner::BoundStreamingJob,
) -> zyron_common::Result<Option<zyron_streaming::job_runner::JoinSpec>> {
    use zyron_planner::binder::BoundStreamingJoinSpec;
    use zyron_streaming::job_runner::{IntervalJoinConfig, JoinSpec, TemporalJoinConfig};
    let Some(join) = &bsj.join else {
        return Ok(None);
    };
    let src_table_id = bsj.source_table_id().unwrap_or(zyron_catalog::TableId(0));
    let left_types: Vec<_> = bsj.source_columns().iter().map(|c| c.type_id).collect();
    match join {
        BoundStreamingJoinSpec::Interval {
            right_source,
            left_key_ordinals,
            right_key_ordinals,
            left_event_time_ordinal,
            right_event_time_ordinal,
            within_us,
            combined_columns,
            join_type,
            ..
        } => {
            let (right_table_id, right_types) = match right_source {
                zyron_planner::binder::BoundStreamingSource::ZyronTable {
                    table_id,
                    columns,
                    ..
                } => (
                    table_id.0,
                    columns.iter().map(|c| c.type_id).collect::<Vec<_>>(),
                ),
                _ => {
                    return Err(zyron_common::ZyronError::PlanError(
                        "interval JOIN requires a Zyron table on the right side".to_string(),
                    ));
                }
            };
            let output_types: Vec<_> = combined_columns.iter().map(|c| c.type_id).collect();
            Ok(Some(JoinSpec::Interval(IntervalJoinConfig {
                left_source_table_id: src_table_id.0,
                right_source_table_id: right_table_id,
                left_types,
                right_types,
                output_types,
                left_key_ordinals: left_key_ordinals.clone(),
                right_key_ordinals: right_key_ordinals.clone(),
                left_event_time_ordinal: *left_event_time_ordinal,
                right_event_time_ordinal: *right_event_time_ordinal,
                within_us: *within_us,
                watermark: zyron_streaming::watermark::WatermarkStrategy::Punctual,
                join_kind: map_bound_join_kind(*join_type),
            })))
        }
        BoundStreamingJoinSpec::Temporal {
            right_table_id,
            right_pk_ordinals,
            left_key_ordinals,
            left_event_time_ordinal,
            combined_columns,
            join_type,
            ..
        } => {
            let right_types: Vec<_> = combined_columns
                .iter()
                .skip(left_types.len())
                .map(|c| c.type_id)
                .collect();
            let output_types: Vec<_> = combined_columns.iter().map(|c| c.type_id).collect();
            Ok(Some(JoinSpec::Temporal(TemporalJoinConfig {
                left_source_table_id: src_table_id.0,
                right_table_id: right_table_id.0,
                left_types,
                right_types,
                output_types,
                left_key_ordinals: left_key_ordinals.clone(),
                right_pk_ordinals: right_pk_ordinals.clone(),
                left_event_time_ordinal: *left_event_time_ordinal,
                join_kind: map_bound_join_kind(*join_type),
            })))
        }
    }
}

/// Maps the planner-level BoundStreamingJoinType to the streaming-crate
/// StreamingJoinKind so the runner has an identical enum without a planner
/// dependency.
fn map_bound_join_kind(
    t: zyron_planner::binder::BoundStreamingJoinType,
) -> zyron_streaming::job_runner::StreamingJoinKind {
    use zyron_planner::binder::BoundStreamingJoinType as B;
    use zyron_streaming::job_runner::StreamingJoinKind as K;
    match t {
        B::Inner => K::Inner,
        B::Left => K::Left,
        B::Right => K::Right,
        B::Full => K::Full,
    }
}

/// Lowers the bound aggregate spec into the runner shape. Returns Ok(None)
/// when the bound job has no aggregate section.
fn build_aggregate_spec(
    bsj: &zyron_planner::BoundStreamingJob,
    src_cols: &[zyron_catalog::ColumnEntry],
) -> zyron_common::Result<Option<zyron_streaming::job_runner::AggregateSpec>> {
    let Some(agg) = &bsj.aggregate else {
        return Ok(None);
    };
    use zyron_streaming::job_runner::{AggWindowType, AggregateItem, AggregateSpec};
    let event_time_ordinal = find_column_ordinal(src_cols, agg.event_time_column_id)?;
    let mut group_by_ordinals = Vec::with_capacity(agg.group_by_column_ids.len());
    for col_id in &agg.group_by_column_ids {
        group_by_ordinals.push(find_column_ordinal(src_cols, *col_id)?);
    }
    let mut aggregations = Vec::with_capacity(agg.aggregations.len());
    for item in &agg.aggregations {
        let input_ordinal = match item.input_column_id {
            Some(cid) => Some(find_column_ordinal(src_cols, cid)?),
            None => None,
        };
        let input_type = match item.input_column_id {
            Some(cid) => src_cols
                .iter()
                .find(|c| c.id == cid)
                .map(|c| c.type_id)
                .unwrap_or(zyron_common::TypeId::Null),
            None => zyron_common::TypeId::Null,
        };
        aggregations.push(AggregateItem {
            function: item.function.clone(),
            input_ordinal,
            input_type,
        });
    }
    let window_type = match agg.window_type {
        zyron_planner::binder::BoundStreamingWindowType::Tumbling { size_ms } => {
            AggWindowType::Tumbling { size_ms }
        }
        zyron_planner::binder::BoundStreamingWindowType::Hopping { size_ms, slide_ms } => {
            AggWindowType::Hopping { size_ms, slide_ms }
        }
        zyron_planner::binder::BoundStreamingWindowType::Session { gap_ms } => {
            AggWindowType::Session { gap_ms }
        }
    };
    let event_time_scale = match agg.event_time_scale {
        zyron_planner::binder::BoundEventTimeScale::Microseconds => {
            zyron_streaming::job_runner::EventTimeScale::Microseconds
        }
        zyron_planner::binder::BoundEventTimeScale::Milliseconds => {
            zyron_streaming::job_runner::EventTimeScale::Milliseconds
        }
        zyron_planner::binder::BoundEventTimeScale::Seconds => {
            zyron_streaming::job_runner::EventTimeScale::Seconds
        }
    };
    let watermark = match agg.watermark {
        zyron_planner::binder::BoundWatermark::BoundedOutOfOrderness {
            allowed_lateness_us,
        } => zyron_streaming::watermark::WatermarkStrategy::BoundedOutOfOrderness {
            allowed_lateness_us,
        },
        zyron_planner::binder::BoundWatermark::Punctual => {
            zyron_streaming::watermark::WatermarkStrategy::Punctual
        }
    };
    let late_data_policy = match agg.late_data_policy {
        zyron_planner::binder::BoundLateDataPolicy::Drop => {
            zyron_streaming::late_data::LateDataPolicy::Drop
        }
        zyron_planner::binder::BoundLateDataPolicy::ReopenWindow => {
            zyron_streaming::late_data::LateDataPolicy::ReopenWindow
        }
        zyron_planner::binder::BoundLateDataPolicy::SideOutput => {
            zyron_streaming::late_data::LateDataPolicy::SideOutput
        }
        zyron_planner::binder::BoundLateDataPolicy::Update => {
            zyron_streaming::late_data::LateDataPolicy::Update
        }
    };
    Ok(Some(AggregateSpec {
        window_type,
        event_time_ordinal,
        event_time_scale,
        group_by_ordinals,
        aggregations,
        watermark,
        late_data_policy,
    }))
}

fn find_column_ordinal(
    cols: &[zyron_catalog::ColumnEntry],
    id: zyron_catalog::ColumnId,
) -> zyron_common::Result<u16> {
    cols.iter()
        .find(|c| c.id == id)
        .map(|c| c.ordinal)
        .ok_or_else(|| {
            zyron_common::ZyronError::PlanError(format!(
                "streaming aggregate references unknown column id {:?}",
                id
            ))
        })
}

async fn handle_create_streaming_job(
    bsj: zyron_planner::BoundStreamingJob,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
    raw_sql: &str,
) -> Result<DdlResult, ProtocolError> {
    use zyron_planner::binder::{BoundStreamingSink, BoundStreamingSource};

    // Classify the endpoint topology so the appropriate spawn path runs.
    // Every topology (Zyron-to-Zyron, external-involved on either side)
    // shares the same pre-flight: catalog insert, snapshot capture, and
    // SecurityContext rehydration.
    let src_schema_id = bsj.source_schema_id();
    let tgt_schema_id = bsj.target_schema_id();
    let src_columns: Vec<_> = bsj.source_columns().to_vec();
    let tgt_columns: Vec<_> = bsj.target_columns().to_vec();
    let src_table_id = bsj.source_table_id().unwrap_or(zyron_catalog::TableId(0));
    let tgt_table_id = bsj.target_table_id().unwrap_or(zyron_catalog::TableId(0));

    // Privilege checks. CREATE on the relevant schema is required in all
    // shapes. For each endpoint, run the check that matches its kind:
    // Zyron tables need SELECT on the source table and INSERT on the target
    // table. Named external endpoints need USAGE on the catalog object.
    // Inline endpoints carry no catalog-level object, the CREATE STREAMING
    // JOB privilege at the schema suffices.
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::CreateStreamingJob,
        zyron_auth::ObjectType::Schema,
        src_schema_id.0,
    )?;
    match &bsj.source {
        BoundStreamingSource::ZyronTable { table_id, .. } => {
            check_ddl_privilege(
                server,
                session,
                zyron_auth::PrivilegeType::Select,
                zyron_auth::ObjectType::Table,
                table_id.0,
            )?;
        }
        BoundStreamingSource::ExternalNamed { source_id, .. } => {
            check_ddl_privilege(
                server,
                session,
                zyron_auth::PrivilegeType::Usage,
                zyron_auth::ObjectType::ExternalSource,
                source_id.0,
            )?;
        }
        BoundStreamingSource::ExternalInline { .. } => {}
    }
    match &bsj.target {
        BoundStreamingSink::ZyronTable { table_id, .. } => {
            check_ddl_privilege(
                server,
                session,
                zyron_auth::PrivilegeType::Insert,
                zyron_auth::ObjectType::Table,
                table_id.0,
            )?;
        }
        BoundStreamingSink::ExternalNamed { sink_id, .. } => {
            check_ddl_privilege(
                server,
                session,
                zyron_auth::PrivilegeType::Usage,
                zyron_auth::ObjectType::ExternalSink,
                sink_id.0,
            )?;
        }
        BoundStreamingSink::ExternalInline { .. } => {}
    }

    // Idempotent check on existing job.
    if server
        .catalog
        .get_streaming_job(src_schema_id, &bsj.name)
        .is_some()
    {
        if bsj.if_not_exists {
            return Ok(DdlResult::Tag("CREATE STREAMING JOB".to_string()));
        }
        return Err(ProtocolError::Database(ZyronError::Internal(format!(
            "streaming job '{}' already exists",
            bsj.name
        ))));
    }

    // Capture security context snapshot. If auth is not configured, store an
    // empty blob so the catalog record is still valid.
    let snap_bytes = session
        .as_ref()
        .and_then(|s| s.security_context.as_ref())
        .map(|ctx| zyron_auth::SecurityContextSnapshot::from_context(ctx).to_bytes())
        .unwrap_or_default();

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let entry = zyron_catalog::StreamingJobEntry {
        id: zyron_catalog::StreamingJobId(0),
        name: bsj.name.clone(),
        source_table_id: src_table_id,
        target_table_id: tgt_table_id,
        source_schema_id: src_schema_id,
        target_schema_id: tgt_schema_id,
        // Store the original CREATE STREAMING JOB SQL text so the startup
        // recovery path can re-parse and re-bind the job after a restart.
        select_sql: raw_sql.to_string(),
        write_mode: bsj.write_mode,
        status: zyron_catalog::schema::StreamingJobStatus::Active,
        creator_snapshot_bytes: snap_bytes,
        created_at: now,
        last_error: None,
    };

    let id = server
        .catalog
        .create_streaming_job(entry.clone())
        .await
        .map_err(ProtocolError::Database)?;

    // Lower BoundExpr projections and predicate into ExprSpec.
    let mut projections = Vec::with_capacity(bsj.projections.len());
    for proj in &bsj.projections {
        projections.push(lower_expr(proj, &src_columns)?);
    }
    let predicate = match &bsj.predicate {
        Some(p) => Some(lower_expr(p, &src_columns)?),
        None => None,
    };

    let source_types: Vec<_> = src_columns.iter().map(|c| c.type_id).collect();
    let target_types: Vec<_> = tgt_columns.iter().map(|c| c.type_id).collect();
    // Resolve PK ColumnIds to target column ordinals for the upsert sink.
    let target_pk_ordinals: Vec<u16> = bsj
        .target_pk_columns
        .iter()
        .filter_map(|col_id| {
            tgt_columns
                .iter()
                .find(|c| c.id == *col_id)
                .map(|c| c.ordinal)
        })
        .collect();
    let aggregate = build_aggregate_spec(&bsj, &src_columns).map_err(ProtocolError::Database)?;
    let join = build_join_spec(&bsj).map_err(ProtocolError::Database)?;
    let spec = zyron_streaming::job_runner::StreamingJobSpec {
        source_table_id: src_table_id.0,
        target_table_id: tgt_table_id.0,
        write_mode: bsj.write_mode,
        projections,
        predicate,
        source_types,
        target_types,
        target_pk_ordinals,
        aggregate,
        join,
    };

    // Required runtime dependencies for spawning the runner. The manager is
    // resolved inside spawn_bound_streaming_job so recovery and wire paths
    // share the same error surface.
    let cdc_registry = server.cdc_registry.as_ref().cloned().ok_or_else(|| {
        ProtocolError::Database(ZyronError::StreamingError(
            "CDC registry not configured".to_string(),
        ))
    })?;
    let security_manager = server.security_manager.as_ref().cloned().ok_or_else(|| {
        ProtocolError::Database(ZyronError::AuthenticationFailed(
            "security manager not configured".to_string(),
        ))
    })?;

    // Reconstruct a SecurityContext for the runner thread. Use the session
    // context when available, otherwise rehydrate from the snapshot bytes.
    let security_ctx = {
        let session_ref = session
            .as_ref()
            .ok_or(ProtocolError::Malformed("no active session".into()))?;
        let ctx_ref = session_ref.security_context.as_ref().ok_or_else(|| {
            ProtocolError::Database(ZyronError::AuthenticationFailed(
                "session has no security context".to_string(),
            ))
        })?;
        let snap = zyron_auth::SecurityContextSnapshot::from_context(ctx_ref);
        let limits = security_manager
            .query_limits
            .get_limits(&ctx_ref.effective_roles);
        snap.into_context(limits)
    };

    // Reload the entry so spawn calls see the catalog-assigned id.
    let stored_entry = server.catalog.get_streaming_job_by_id(id).ok_or_else(|| {
        ProtocolError::Database(ZyronError::Internal(
            "streaming job missing from catalog after create".to_string(),
        ))
    })?;

    // Dispatch to the matching spawn path based on endpoint topology. Shared
    // with the server-side recovery path so both entry points exercise the
    // same code.
    spawn_bound_streaming_job(
        &bsj,
        &stored_entry,
        spec,
        security_ctx,
        security_manager,
        cdc_registry,
        server,
    )?;

    Ok(DdlResult::Tag("CREATE STREAMING JOB".to_string()))
}

// ---------------------------------------------------------------------------
// Shared streaming-job dispatch
// ---------------------------------------------------------------------------

/// Spawns the appropriate runner for a bound streaming job. Used by both the
/// wire handler (CREATE STREAMING JOB path) and the server-side startup
/// recovery path so every topology is dispatched through a single match.
///
/// Callers are responsible for persisting the catalog entry and reconstructing
/// the creator SecurityContext before invoking this function. The runner
/// registers itself with the manager under the StreamingJobId of stored_entry.
#[allow(clippy::too_many_arguments)]
pub fn spawn_bound_streaming_job(
    bsj: &zyron_planner::BoundStreamingJob,
    stored_entry: &zyron_catalog::StreamingJobEntry,
    spec: zyron_streaming::job_runner::StreamingJobSpec,
    security_ctx: zyron_auth::SecurityContext,
    security_manager: Arc<zyron_auth::SecurityManager>,
    cdc_registry: Arc<zyron_cdc::CdfRegistry>,
    server: &Arc<ServerState>,
) -> Result<(), ProtocolError> {
    use zyron_planner::binder::{BoundStreamingSink, BoundStreamingSource};

    let src_columns: Vec<_> = bsj.source_columns().to_vec();
    let tgt_columns: Vec<_> = bsj.target_columns().to_vec();
    let src_table_id = bsj.source_table_id().unwrap_or(zyron_catalog::TableId(0));
    let tgt_table_id = bsj.target_table_id().unwrap_or(zyron_catalog::TableId(0));

    let manager = server.stream_job_manager.as_ref().ok_or_else(|| {
        ProtocolError::Database(ZyronError::StreamingError(
            "streaming job manager not configured".to_string(),
        ))
    })?;

    match (&bsj.source, &bsj.target) {
        (BoundStreamingSource::ZyronTable { .. }, BoundStreamingSink::ZyronTable { .. }) => {
            let target_entry = server
                .catalog
                .get_table_by_id(tgt_table_id)
                .map_err(ProtocolError::Database)?;
            let heap = zyron_storage::HeapFile::new(
                Arc::clone(&server.disk_manager),
                Arc::clone(&server.buffer_pool),
                zyron_storage::HeapFileConfig {
                    heap_file_id: target_entry.heap_file_id,
                    fsm_file_id: target_entry.fsm_file_id,
                },
            )
            .map_err(ProtocolError::Database)?;
            let heap_arc = Arc::new(heap);
            manager
                .lock()
                .spawn_zyron_table_job(
                    stored_entry.clone(),
                    spec,
                    security_ctx,
                    Arc::clone(&server.catalog),
                    heap_arc,
                    cdc_registry,
                    Arc::clone(&server.txn_manager),
                    Arc::clone(&server.wal),
                    security_manager,
                )
                .map_err(ProtocolError::Database)?;
            let _ = src_table_id;
            let _ = src_columns;
        }

        // Remote Zyron source -> Zyron table sink. Dispatched through the
        // ZyronSourceAdapter path so the runner pulls via the PG wire
        // client rather than OpenDAL.
        (src_variant, BoundStreamingSink::ZyronTable { .. })
            if source_is_zyron_backend(src_variant, server) =>
        {
            let (zyron_source_client, start_lsn) =
                build_zyron_source_client(src_variant, &src_columns, server)?;
            let target_entry = server
                .catalog
                .get_table_by_id(tgt_table_id)
                .map_err(ProtocolError::Database)?;
            let heap = zyron_storage::HeapFile::new(
                Arc::clone(&server.disk_manager),
                Arc::clone(&server.buffer_pool),
                zyron_storage::HeapFileConfig {
                    heap_file_id: target_entry.heap_file_id,
                    fsm_file_id: target_entry.fsm_file_id,
                },
            )
            .map_err(ProtocolError::Database)?;
            let heap_arc = Arc::new(heap);
            let ctx_arc = Arc::new(parking_lot::Mutex::new(security_ctx));
            let sink = match bsj.write_mode {
                zyron_catalog::schema::CatalogStreamingWriteMode::Upsert => {
                    let upsert = zyron_streaming::ZyronUpsertSink::new(
                        tgt_table_id.0,
                        spec.target_pk_ordinals.clone(),
                        spec.target_types.clone(),
                        Arc::clone(&server.catalog),
                        heap_arc,
                        Arc::clone(&server.txn_manager),
                        Arc::clone(&server.wal),
                        Arc::clone(&ctx_arc),
                        Arc::clone(&security_manager),
                    )
                    .map_err(ProtocolError::Database)?;
                    zyron_streaming::job_runner::RunnerSink::Upsert(upsert)
                }
                zyron_catalog::schema::CatalogStreamingWriteMode::Append => {
                    zyron_streaming::job_runner::RunnerSink::Append(
                        zyron_streaming::sink_connector::ZyronRowSink::new(
                            tgt_table_id.0,
                            bsj.write_mode,
                            Arc::clone(&server.catalog),
                            heap_arc,
                            Arc::clone(&server.txn_manager),
                            ctx_arc,
                            Arc::clone(&security_manager),
                        ),
                    )
                }
            };
            let adapter: Arc<dyn zyron_streaming::source_connector::ZyronSourceAdapter> =
                Arc::new(zyron_source_client);
            manager
                .lock()
                .spawn_remote_source_to_zyron_job(
                    stored_entry.clone(),
                    spec,
                    adapter,
                    sink,
                    Arc::clone(&server.catalog),
                    start_lsn,
                )
                .map_err(ProtocolError::Database)?;
            let _ = cdc_registry;
        }

        // Zyron table source -> remote Zyron sink. Dispatched through the
        // ZyronSinkAdapter path so the runner pushes via the PG wire client
        // rather than OpenDAL.
        (BoundStreamingSource::ZyronTable { .. }, tgt_variant)
            if sink_is_zyron_backend(tgt_variant, server) =>
        {
            let zyron_sink_client =
                build_zyron_sink_client(tgt_variant, &tgt_columns, bsj.write_mode, server)?;
            let source = zyron_streaming::source_connector::ZyronTableSource::new(
                src_table_id.0,
                Arc::clone(&cdc_registry),
            )
            .map_err(ProtocolError::Database)?;
            let adapter: Arc<dyn zyron_streaming::sink_connector::ZyronSinkAdapter> =
                Arc::new(zyron_sink_client);
            let sink = zyron_streaming::job_runner::RunnerSink::Remote(adapter);
            manager
                .lock()
                .spawn_zyron_source_to_runner_sink_job(
                    stored_entry.clone(),
                    spec,
                    source,
                    sink,
                    Arc::clone(&server.catalog),
                )
                .map_err(ProtocolError::Database)?;
            let _ = security_ctx;
            let _ = security_manager;
        }

        // External source -> Zyron table sink
        (src_variant, BoundStreamingSink::ZyronTable { .. }) => {
            let (external_source, mode, schedule_cron) =
                build_external_source(src_variant, &src_columns, server)?;
            let target_entry = server
                .catalog
                .get_table_by_id(tgt_table_id)
                .map_err(ProtocolError::Database)?;
            let heap = zyron_storage::HeapFile::new(
                Arc::clone(&server.disk_manager),
                Arc::clone(&server.buffer_pool),
                zyron_storage::HeapFileConfig {
                    heap_file_id: target_entry.heap_file_id,
                    fsm_file_id: target_entry.fsm_file_id,
                },
            )
            .map_err(ProtocolError::Database)?;
            let heap_arc = Arc::new(heap);
            let ctx_arc = Arc::new(parking_lot::Mutex::new(security_ctx));
            let sink = match bsj.write_mode {
                zyron_catalog::schema::CatalogStreamingWriteMode::Upsert => {
                    let upsert = zyron_streaming::ZyronUpsertSink::new(
                        tgt_table_id.0,
                        spec.target_pk_ordinals.clone(),
                        spec.target_types.clone(),
                        Arc::clone(&server.catalog),
                        heap_arc,
                        Arc::clone(&server.txn_manager),
                        Arc::clone(&server.wal),
                        Arc::clone(&ctx_arc),
                        Arc::clone(&security_manager),
                    )
                    .map_err(ProtocolError::Database)?;
                    zyron_streaming::job_runner::RunnerSink::Upsert(upsert)
                }
                zyron_catalog::schema::CatalogStreamingWriteMode::Append => {
                    zyron_streaming::job_runner::RunnerSink::Append(
                        zyron_streaming::sink_connector::ZyronRowSink::new(
                            tgt_table_id.0,
                            bsj.write_mode,
                            Arc::clone(&server.catalog),
                            heap_arc,
                            Arc::clone(&server.txn_manager),
                            ctx_arc,
                            Arc::clone(&security_manager),
                        ),
                    )
                }
            };
            manager
                .lock()
                .spawn_external_to_zyron_job(
                    stored_entry.clone(),
                    spec,
                    Arc::new(external_source),
                    sink,
                    mode,
                    schedule_cron,
                    Arc::clone(&server.catalog),
                )
                .map_err(ProtocolError::Database)?;
            let _ = cdc_registry;
        }

        // Zyron table source -> external sink
        (BoundStreamingSource::ZyronTable { .. }, tgt_variant) => {
            let external_sink = build_external_sink(tgt_variant, &tgt_columns, server)?;
            let source = zyron_streaming::source_connector::ZyronTableSource::new(
                src_table_id.0,
                Arc::clone(&cdc_registry),
            )
            .map_err(ProtocolError::Database)?;
            manager
                .lock()
                .spawn_zyron_to_external_job(
                    stored_entry.clone(),
                    spec,
                    source,
                    Arc::new(external_sink),
                    Arc::clone(&server.catalog),
                )
                .map_err(ProtocolError::Database)?;
            let _ = security_ctx;
            let _ = security_manager;
        }

        // External source -> external sink
        (src_variant, tgt_variant) => {
            let (external_source, mode, schedule_cron) =
                build_external_source(src_variant, &src_columns, server)?;
            let external_sink = build_external_sink(tgt_variant, &tgt_columns, server)?;
            manager
                .lock()
                .spawn_external_to_external_job(
                    stored_entry.clone(),
                    spec,
                    Arc::new(external_source),
                    Arc::new(external_sink),
                    mode,
                    schedule_cron,
                    Arc::clone(&server.catalog),
                )
                .map_err(ProtocolError::Database)?;
            let _ = cdc_registry;
            let _ = security_ctx;
            let _ = security_manager;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Audit tracing helpers for external endpoints
// ---------------------------------------------------------------------------

/// Returns the effective role id from the session's security context. Zero
/// when no context is attached, used as the audit actor identifier.
fn actor_role_id(session: &Option<Session>) -> u32 {
    session
        .as_ref()
        .and_then(|s| s.security_context.as_ref())
        .map(|ctx| ctx.current_role.0)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// External endpoint construction helpers
// ---------------------------------------------------------------------------

/// Converts ColumnEntry slices into streaming-layer ColumnSpec entries.
fn columns_to_specs(
    cols: &[zyron_catalog::ColumnEntry],
) -> Vec<zyron_streaming::format::ColumnSpec> {
    cols.iter()
        .map(|c| {
            zyron_streaming::format::ColumnSpec::with_precision(
                c.name.clone(),
                c.type_id,
                c.fractional_digits,
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Zyron-backend detection and client construction
// ---------------------------------------------------------------------------

/// Returns true when the bound source resolves to an ExternalBackend::Zyron
/// endpoint, either through a named catalog entry or an inline definition.
fn source_is_zyron_backend(
    src: &zyron_planner::binder::BoundStreamingSource,
    server: &Arc<ServerState>,
) -> bool {
    use zyron_planner::binder::BoundStreamingSource;
    match src {
        BoundStreamingSource::ExternalNamed { source_id, .. } => server
            .catalog
            .get_external_source_by_id(*source_id)
            .map(|e| matches!(e.backend, zyron_catalog::ExternalBackend::Zyron))
            .unwrap_or(false),
        BoundStreamingSource::ExternalInline { backend, .. } => {
            matches!(backend, zyron_parser::ast::ExternalBackendKind::Zyron)
        }
        BoundStreamingSource::ZyronTable { .. } => false,
    }
}

/// Returns true when the bound sink resolves to an ExternalBackend::Zyron
/// endpoint.
fn sink_is_zyron_backend(
    tgt: &zyron_planner::binder::BoundStreamingSink,
    server: &Arc<ServerState>,
) -> bool {
    use zyron_planner::binder::BoundStreamingSink;
    match tgt {
        BoundStreamingSink::ExternalNamed { sink_id, .. } => server
            .catalog
            .get_external_sink_by_id(*sink_id)
            .map(|e| matches!(e.backend, zyron_catalog::ExternalBackend::Zyron))
            .unwrap_or(false),
        BoundStreamingSink::ExternalInline { backend, .. } => {
            matches!(backend, zyron_parser::ast::ExternalBackendKind::Zyron)
        }
        BoundStreamingSink::ZyronTable { .. } => false,
    }
}

/// Parses the zyron://... URI on an external endpoint, constructs a PG-wire
/// ConnectionPool keyed on its hosts plus unsealed credentials, and returns
/// the pool, target schema, target table, and resolved options map.
fn build_zyron_pool_from_endpoint(
    uri: &str,
    options: &[(String, String)],
    creds: &std::collections::HashMap<String, String>,
) -> Result<
    (
        Arc<crate::pool::ConnectionPool>,
        String,
        String,
        std::collections::HashMap<String, String>,
    ),
    ProtocolError,
> {
    let parsed = crate::uri::parse_zyron_uri(uri).map_err(|e| {
        ProtocolError::Database(ZyronError::StreamingError(format!(
            "invalid zyron:// uri: {e}"
        )))
    })?;
    let first_host = parsed.hosts.first().ok_or_else(|| {
        ProtocolError::Database(ZyronError::StreamingError(
            "zyron:// uri has no hosts".to_string(),
        ))
    })?;
    let password = creds
        .get("password")
        .cloned()
        .or_else(|| parsed.password.clone());
    let mut cfg = crate::pool::PoolConfig::simple(
        &first_host.host,
        first_host.port,
        &parsed.user,
        password.as_deref(),
        &parsed.database,
    );
    // Merge remaining hosts from the URI beyond the first one.
    for h in parsed.hosts.iter().skip(1) {
        cfg.hosts.push(crate::pool::HostEntry {
            host: h.host.clone(),
            port: h.port,
            role: crate::pool::HostRole::Unknown,
            health: crate::pool::AtomicHealth::new(),
        });
    }
    let pool = Arc::new(crate::pool::ConnectionPool::new(cfg));
    let (schema, table) = match &parsed.target {
        crate::uri::ZyronUriTarget::Table { schema, table } => (schema.clone(), table.clone()),
        crate::uri::ZyronUriTarget::Publication { name } => (String::new(), name.clone()),
        crate::uri::ZyronUriTarget::Database => (String::new(), String::new()),
    };
    let opt_map: std::collections::HashMap<String, String> = options.iter().cloned().collect();
    Ok((pool, schema, table, opt_map))
}

/// Builds a ZyronSinkClient from a BoundStreamingSink whose backend is Zyron.
fn build_zyron_sink_client(
    tgt: &zyron_planner::binder::BoundStreamingSink,
    tgt_columns: &[zyron_catalog::ColumnEntry],
    write_mode: zyron_catalog::schema::CatalogStreamingWriteMode,
    server: &Arc<ServerState>,
) -> Result<crate::zyron_sink::ZyronSinkClient, ProtocolError> {
    use zyron_planner::binder::BoundStreamingSink;
    let (uri, options, creds) = match tgt {
        BoundStreamingSink::ExternalNamed { sink_id, .. } => {
            let entry = server
                .catalog
                .get_external_sink_by_id(*sink_id)
                .ok_or_else(|| {
                    ProtocolError::Database(ZyronError::Internal(format!(
                        "external sink id {} not found",
                        sink_id.0
                    )))
                })?;
            let unsealed = unseal_entry_credentials(
                entry.credential_key_id,
                entry.credential_ciphertext.as_deref(),
                server,
            )?;
            (entry.uri.clone(), entry.options.clone(), unsealed)
        }
        BoundStreamingSink::ExternalInline { uri, options, .. } => (
            uri.clone(),
            options.clone(),
            std::collections::HashMap::new(),
        ),
        BoundStreamingSink::ZyronTable { .. } => {
            return Err(ProtocolError::Database(ZyronError::Internal(
                "build_zyron_sink_client called with ZyronTable variant".to_string(),
            )));
        }
    };

    let (pool, target_schema, target_table, opt_map) =
        build_zyron_pool_from_endpoint(&uri, &options, &creds)?;

    let pk_columns: Vec<String> = opt_map
        .get("pk_columns")
        .map(|s| {
            s.split(',')
                .map(|c| c.trim().to_string())
                .filter(|c| !c.is_empty())
                .collect()
        })
        .unwrap_or_default();
    let idempotency_key_columns: Vec<String> = opt_map
        .get("idempotency_keys")
        .map(|s| {
            s.split(',')
                .map(|c| c.trim().to_string())
                .filter(|c| !c.is_empty())
                .collect()
        })
        .unwrap_or_default();
    let copy_threshold_rows = opt_map
        .get("copy_threshold_rows")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000usize);
    let batch_size = opt_map
        .get("batch_size")
        .and_then(|s| s.parse().ok())
        .unwrap_or(256usize);
    let flush_ms = opt_map
        .get("flush_interval_ms")
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(500);

    let target_types: Vec<zyron_common::TypeId> = tgt_columns.iter().map(|c| c.type_id).collect();
    let target_column_names: Vec<String> = tgt_columns.iter().map(|c| c.name.clone()).collect();

    let cb = Arc::new(zyron_streaming::retry::CircuitBreaker::new(
        0.5,
        4,
        std::time::Duration::from_secs(5),
    ));
    let retry_config = zyron_streaming::retry::RetryConfig::default();

    let cfg = crate::zyron_sink::ZyronSinkConfig {
        pool,
        target_schema,
        target_table,
        write_mode,
        pk_columns,
        target_types,
        target_column_names,
        copy_threshold_rows,
        batch_size,
        flush_interval: std::time::Duration::from_millis(flush_ms),
        dlq: None,
        circuit_breaker: cb,
        retry_config,
        idempotency_key_columns,
    };
    Ok(crate::zyron_sink::ZyronSinkClient::new(cfg))
}

/// Builds a ZyronSourceClient from a BoundStreamingSource whose backend is
/// Zyron. Returns the client plus the LSN the runner should resume from.
fn build_zyron_source_client(
    src: &zyron_planner::binder::BoundStreamingSource,
    _src_columns: &[zyron_catalog::ColumnEntry],
    server: &Arc<ServerState>,
) -> Result<(crate::zyron_source::ZyronSourceClient, u64), ProtocolError> {
    use zyron_planner::binder::BoundStreamingSource;
    let (uri, options, creds) = match src {
        BoundStreamingSource::ExternalNamed { source_id, .. } => {
            let entry = server
                .catalog
                .get_external_source_by_id(*source_id)
                .ok_or_else(|| {
                    ProtocolError::Database(ZyronError::Internal(format!(
                        "external source id {} not found",
                        source_id.0
                    )))
                })?;
            let unsealed = unseal_entry_credentials(
                entry.credential_key_id,
                entry.credential_ciphertext.as_deref(),
                server,
            )?;
            (entry.uri.clone(), entry.options.clone(), unsealed)
        }
        BoundStreamingSource::ExternalInline { uri, options, .. } => (
            uri.clone(),
            options.clone(),
            std::collections::HashMap::new(),
        ),
        BoundStreamingSource::ZyronTable { .. } => {
            return Err(ProtocolError::Database(ZyronError::Internal(
                "build_zyron_source_client called with ZyronTable variant".to_string(),
            )));
        }
    };

    let (pool, _schema, publication_from_uri, opt_map) =
        build_zyron_pool_from_endpoint(&uri, &options, &creds)?;

    let publication = opt_map
        .get("publication")
        .cloned()
        .filter(|s| !s.is_empty())
        .unwrap_or(publication_from_uri);
    let consumer_id = opt_map
        .get("consumer_id")
        .cloned()
        .unwrap_or_else(|| format!("zyron-consumer-{}", std::process::id()));
    let batch_size = opt_map
        .get("batch_size")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(256);
    let poll_ms = opt_map
        .get("poll_interval_ms")
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(200);
    let start_lsn = opt_map
        .get("start_lsn")
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);
    let checkpoint_interval = opt_map
        .get("checkpoint_interval_batches")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(4);

    // Parallel snapshot knobs. One worker keeps the previous behaviour, which
    // is what a subscription that says nothing gets
    let snapshot_workers = opt_map
        .get("snapshot_workers")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1)
        .clamp(1, 32);
    let snapshot_chunk_strategy = match opt_map
        .get("snapshot_chunk_strategy")
        .map(|s| s.to_ascii_lowercase())
        .as_deref()
    {
        Some("row_count") | Some("rowcount") => {
            crate::zyron_source::SnapshotChunkStrategy::RowCount
        }
        _ => crate::zyron_source::SnapshotChunkStrategy::PkRange,
    };

    let cfg = crate::zyron_source::ZyronSourceConfig {
        pool,
        publication,
        consumer_id,
        mode: crate::zyron_source::ZyronSourceMode::Pull {
            poll_interval: std::time::Duration::from_millis(poll_ms),
            batch_size,
        },
        schema_pin: None,
        on_schema_change: crate::zyron_source::OnSchemaChange::Refresh,
        checkpoint_interval_batches: checkpoint_interval,
        subscription_id: 0,
        catalog: Some(Arc::clone(&server.catalog)),
        snapshot_workers,
        snapshot_chunk_strategy,
    };
    Ok((crate::zyron_source::ZyronSourceClient::new(cfg), start_lsn))
}

/// Opens an ExternalTableSource from either a named catalog entry or an
/// inline definition. Unseals credentials through the server key store when
/// the named entry carries them. Inline variants carry no credentials, the
/// source runs unauthenticated against whichever backend it points at.
fn build_external_source(
    src: &zyron_planner::binder::BoundStreamingSource,
    src_columns: &[zyron_catalog::ColumnEntry],
    server: &Arc<ServerState>,
) -> Result<
    (
        zyron_streaming::external_source::ExternalTableSource,
        zyron_catalog::ExternalMode,
        Option<String>,
    ),
    ProtocolError,
> {
    use zyron_planner::binder::BoundStreamingSource;
    match src {
        BoundStreamingSource::ExternalNamed { source_id, .. } => {
            let entry = server
                .catalog
                .get_external_source_by_id(*source_id)
                .ok_or_else(|| {
                    ProtocolError::Database(ZyronError::Internal(format!(
                        "external source id {} not found in catalog",
                        source_id.0
                    )))
                })?;
            let creds = unseal_entry_credentials(
                entry.credential_key_id,
                entry.credential_ciphertext.as_deref(),
                server,
            )?;
            let column_schema = columns_to_specs(src_columns);
            let source = zyron_streaming::external_source::ExternalTableSource::new(
                &entry,
                creds,
                column_schema,
            )
            .map_err(ProtocolError::Database)?;
            Ok((source, entry.mode, entry.schedule_cron.clone()))
        }
        BoundStreamingSource::ExternalInline {
            backend,
            uri,
            format,
            options,
            mode,
            ..
        } => {
            let (cmode, cron) = parser_mode_to_catalog(mode);
            // Build a transient entry so ExternalTableSource::new can reuse
            // its catalog-entry constructor. This entry is never persisted.
            let entry = zyron_catalog::ExternalSourceEntry {
                id: zyron_catalog::ExternalSourceId(0),
                schema_id: zyron_catalog::SchemaId(0),
                name: String::new(),
                backend: parser_backend_to_catalog(backend.clone()),
                uri: uri.clone(),
                format: parser_format_to_catalog(format.clone()),
                mode: cmode,
                schedule_cron: cron.clone(),
                options: options.clone(),
                columns: Vec::new(),
                credential_key_id: None,
                credential_ciphertext: None,
                classification: zyron_catalog::CatalogClassification::Internal,
                tags: Vec::new(),
                owner_role_id: 0,
                created_at: 0,
            };
            let column_schema = columns_to_specs(src_columns);
            let source = zyron_streaming::external_source::ExternalTableSource::new(
                &entry,
                std::collections::HashMap::new(),
                column_schema,
            )
            .map_err(ProtocolError::Database)?;
            Ok((source, cmode, cron))
        }
        BoundStreamingSource::ZyronTable { .. } => {
            Err(ProtocolError::Database(ZyronError::Internal(
                "build_external_source called with ZyronTable variant".to_string(),
            )))
        }
    }
}

/// Opens an ExternalRowSink from either a named catalog entry or an inline
/// definition.
fn build_external_sink(
    tgt: &zyron_planner::binder::BoundStreamingSink,
    tgt_columns: &[zyron_catalog::ColumnEntry],
    server: &Arc<ServerState>,
) -> Result<zyron_streaming::external_sink::ExternalRowSink, ProtocolError> {
    use zyron_planner::binder::BoundStreamingSink;
    match tgt {
        BoundStreamingSink::ExternalNamed { sink_id, .. } => {
            let entry = server
                .catalog
                .get_external_sink_by_id(*sink_id)
                .ok_or_else(|| {
                    ProtocolError::Database(ZyronError::Internal(format!(
                        "external sink id {} not found in catalog",
                        sink_id.0
                    )))
                })?;
            let creds = unseal_entry_credentials(
                entry.credential_key_id,
                entry.credential_ciphertext.as_deref(),
                server,
            )?;
            let column_schema = columns_to_specs(tgt_columns);
            zyron_streaming::external_sink::ExternalRowSink::new(&entry, creds, column_schema)
                .map_err(ProtocolError::Database)
        }
        BoundStreamingSink::ExternalInline {
            backend,
            uri,
            format,
            options,
            ..
        } => {
            let entry = zyron_catalog::ExternalSinkEntry {
                id: zyron_catalog::ExternalSinkId(0),
                schema_id: zyron_catalog::SchemaId(0),
                name: String::new(),
                backend: parser_backend_to_catalog(backend.clone()),
                uri: uri.clone(),
                format: parser_format_to_catalog(format.clone()),
                options: options.clone(),
                columns: Vec::new(),
                credential_key_id: None,
                credential_ciphertext: None,
                classification: zyron_catalog::CatalogClassification::Internal,
                tags: Vec::new(),
                owner_role_id: 0,
                created_at: 0,
            };
            let column_schema = columns_to_specs(tgt_columns);
            zyron_streaming::external_sink::ExternalRowSink::new(
                &entry,
                std::collections::HashMap::new(),
                column_schema,
            )
            .map_err(ProtocolError::Database)
        }
        BoundStreamingSink::ZyronTable { .. } => Err(ProtocolError::Database(
            ZyronError::Internal("build_external_sink called with ZyronTable variant".to_string()),
        )),
    }
}

/// Unseals a credential blob stored on an external source/sink entry. An
/// entry without credentials returns an empty map.
fn unseal_entry_credentials(
    key_id: Option<u32>,
    ciphertext: Option<&[u8]>,
    server: &Arc<ServerState>,
) -> Result<std::collections::HashMap<String, String>, ProtocolError> {
    match (key_id, ciphertext) {
        (Some(kid), Some(ct)) => {
            let sealed = zyron_auth::SealedCredentials {
                key_id: kid,
                ciphertext: ct.to_vec(),
            };
            let opened = zyron_auth::open_credentials(&sealed, server.key_store.as_ref())
                .map_err(ProtocolError::Database)?;
            tracing::info!(
                target: "zyron::audit",
                event = "ExternalCredentialRead",
                key_id = kid,
            );
            Ok(opened)
        }
        _ => Ok(std::collections::HashMap::new()),
    }
}

async fn handle_drop_streaming_job(
    name: &str,
    if_exists: bool,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (_, schema_id) = get_session_schema(session, server, None)?;

    let job = match server.catalog.get_streaming_job(schema_id, name) {
        Some(j) => j,
        None => {
            if if_exists {
                return Ok(DdlResult::Tag("DROP STREAMING JOB".to_string()));
            }
            return Err(ProtocolError::Database(ZyronError::Internal(format!(
                "streaming job '{}' not found",
                name
            ))));
        }
    };

    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::DropStreamingJob,
        zyron_auth::ObjectType::StreamingJob,
        job.id.0,
    )?;

    if let Some(mgr) = &server.stream_job_manager {
        // Ignore missing-handle errors, a restart path leaves no live thread.
        let _ = mgr.lock().stop_job(job.id);
    }

    server
        .catalog
        .drop_streaming_job(schema_id, name)
        .await
        .map_err(ProtocolError::Database)?;

    Ok(DdlResult::Tag("DROP STREAMING JOB".to_string()))
}

async fn handle_alter_streaming_job(
    name: &str,
    action: zyron_parser::ast::AlterStreamingJobAction,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (_, schema_id) = get_session_schema(session, server, None)?;

    let job = server
        .catalog
        .get_streaming_job(schema_id, name)
        .ok_or_else(|| {
            ProtocolError::Database(ZyronError::Internal(format!(
                "streaming job '{}' not found",
                name
            )))
        })?;

    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::AlterStreamingJob,
        zyron_auth::ObjectType::StreamingJob,
        job.id.0,
    )?;

    let new_status = match action {
        zyron_parser::ast::AlterStreamingJobAction::Pause => {
            zyron_catalog::schema::StreamingJobStatus::Paused
        }
        zyron_parser::ast::AlterStreamingJobAction::Resume => {
            zyron_catalog::schema::StreamingJobStatus::Active
        }
    };

    server
        .catalog
        .update_streaming_job_status(job.id, new_status, None)
        .await
        .map_err(ProtocolError::Database)?;

    Ok(DdlResult::Tag("ALTER STREAMING JOB".to_string()))
}

// ---------------------------------------------------------------------------
// External source and sink DDL dispatch
// ---------------------------------------------------------------------------

/// Binds an external source/sink DDL statement through the planner binder and
/// dispatches it to the matching handler. Keeps the privilege and catalog
/// work inside this crate so the planner stays pure.
async fn dispatch_external_statement(
    stmt: zyron_parser::Statement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let db_id = get_session_database(session)?;
    let search_path = session
        .as_ref()
        .map(|s| s.search_path.clone())
        .unwrap_or_default();

    let resolver = server.catalog.resolver(db_id, search_path);
    let mut binder = zyron_planner::Binder::new(resolver, &server.catalog);
    let bound = binder.bind(stmt).await.map_err(ProtocolError::Database)?;

    match bound {
        zyron_planner::BoundStatement::CreateExternalSource(b) => {
            handle_create_external_source(*b, server, session).await
        }
        zyron_planner::BoundStatement::CreateExternalSink(b) => {
            handle_create_external_sink(*b, server, session).await
        }
        zyron_planner::BoundStatement::DropExternalSource {
            name,
            schema_id,
            if_exists,
        } => handle_drop_external_source(&name, schema_id, if_exists, server, session).await,
        zyron_planner::BoundStatement::DropExternalSink {
            name,
            schema_id,
            if_exists,
        } => handle_drop_external_sink(&name, schema_id, if_exists, server, session).await,
        zyron_planner::BoundStatement::AlterExternalSource(b) => {
            handle_alter_external_source(*b, server, session).await
        }
        zyron_planner::BoundStatement::AlterExternalSink(b) => {
            handle_alter_external_sink(*b, server, session).await
        }
        _ => Err(ProtocolError::Database(ZyronError::PlanError(
            "expected external source/sink DDL statement".to_string(),
        ))),
    }
}

// ---------------------------------------------------------------------------
// Parser-to-catalog enum mappers
// ---------------------------------------------------------------------------

fn parser_backend_to_catalog(
    b: zyron_parser::ast::ExternalBackendKind,
) -> zyron_catalog::ExternalBackend {
    use zyron_parser::ast::ExternalBackendKind;
    match b {
        ExternalBackendKind::File => zyron_catalog::ExternalBackend::File,
        ExternalBackendKind::S3 => zyron_catalog::ExternalBackend::S3,
        ExternalBackendKind::Gcs => zyron_catalog::ExternalBackend::Gcs,
        ExternalBackendKind::Azure => zyron_catalog::ExternalBackend::Azure,
        ExternalBackendKind::Http => zyron_catalog::ExternalBackend::Http,
        ExternalBackendKind::Zyron => zyron_catalog::ExternalBackend::Zyron,
    }
}

fn parser_format_to_catalog(
    f: zyron_parser::ast::ExternalFormatKind,
) -> zyron_catalog::ExternalFormat {
    use zyron_parser::ast::ExternalFormatKind;
    match f {
        ExternalFormatKind::Json => zyron_catalog::ExternalFormat::Json,
        ExternalFormatKind::JsonLines => zyron_catalog::ExternalFormat::JsonLines,
        ExternalFormatKind::Csv => zyron_catalog::ExternalFormat::Csv,
        ExternalFormatKind::Parquet => zyron_catalog::ExternalFormat::Parquet,
        ExternalFormatKind::ArrowIpc => zyron_catalog::ExternalFormat::ArrowIpc,
        ExternalFormatKind::Avro => zyron_catalog::ExternalFormat::Avro,
    }
}

/// Translates a parser ExternalModeSpec into a catalog ExternalMode plus the
/// trigger string. Scheduled mode returns cron or every as a single string
/// with a prefix, the runner parses it back out of the entry.
fn parser_mode_to_catalog(
    m: &zyron_parser::ast::ExternalModeSpec,
) -> (zyron_catalog::ExternalMode, Option<String>) {
    use zyron_parser::ast::ExternalModeSpec;
    match m {
        ExternalModeSpec::OneShot => (zyron_catalog::ExternalMode::OneShot, None),
        ExternalModeSpec::Watch => (zyron_catalog::ExternalMode::Watch, None),
        ExternalModeSpec::Scheduled { cron, every } => {
            let s = cron
                .clone()
                .or_else(|| every.clone())
                .filter(|s| !s.is_empty());
            (zyron_catalog::ExternalMode::Scheduled, s)
        }
    }
}

// ---------------------------------------------------------------------------
// CREATE EXTERNAL SOURCE / SINK
// ---------------------------------------------------------------------------

async fn handle_create_external_source(
    bound: zyron_planner::binder::BoundCreateExternalSource,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::CreateExternalSource,
        zyron_auth::ObjectType::Schema,
        bound.schema_id.0,
    )?;
    if !bound.credentials.is_empty() {
        check_ddl_privilege(
            server,
            session,
            zyron_auth::PrivilegeType::ManageExternalCredentials,
            zyron_auth::ObjectType::Schema,
            bound.schema_id.0,
        )?;
    }

    // Duplicate-name check. if_not_exists short-circuits to a success tag.
    if server
        .catalog
        .get_external_source(bound.schema_id, &bound.name)
        .is_some()
    {
        if bound.if_not_exists {
            return Ok(DdlResult::Tag("CREATE EXTERNAL SOURCE".to_string()));
        }
        return Err(ProtocolError::Database(ZyronError::Internal(format!(
            "external source '{}' already exists",
            bound.name
        ))));
    }

    let creds_map: std::collections::HashMap<String, String> =
        bound.credentials.iter().cloned().collect();
    let sealed = if creds_map.is_empty() {
        None
    } else {
        Some(
            zyron_auth::seal_credentials(&creds_map, server.key_store.as_ref())
                .map_err(ProtocolError::Database)?,
        )
    };

    let (mode, schedule_cron) = parser_mode_to_catalog(&bound.mode);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let owner_role_id = session
        .as_ref()
        .and_then(|s| s.security_context.as_ref())
        .map(|ctx| ctx.current_role.0)
        .unwrap_or(0);

    // Resolve the persisted column layout. Explicit COLUMNS (...) wins,
    // otherwise infer from the first matching file when the format carries
    // its own schema, otherwise reject because the layout cannot be known.
    let format_kind = parser_format_to_catalog(bound.format.clone());
    let columns: Vec<(String, zyron_common::TypeId)> = if !bound.columns.is_empty() {
        bound.columns.clone()
    } else {
        match format_kind {
            zyron_catalog::ExternalFormat::Parquet
            | zyron_catalog::ExternalFormat::ArrowIpc
            | zyron_catalog::ExternalFormat::Avro => {
                // Build a transient entry for the inference call so OpenDAL
                // wiring picks up the same backend, URI, and options as the
                // final persisted entry.
                let probe_entry = zyron_catalog::ExternalSourceEntry {
                    id: zyron_catalog::ExternalSourceId(0),
                    schema_id: bound.schema_id,
                    name: bound.name.clone(),
                    backend: parser_backend_to_catalog(bound.backend.clone()),
                    uri: bound.uri.clone(),
                    format: format_kind,
                    mode,
                    schedule_cron: schedule_cron.clone(),
                    options: bound.options.clone(),
                    columns: Vec::new(),
                    credential_key_id: None,
                    credential_ciphertext: None,
                    classification: zyron_catalog::CatalogClassification::Internal,
                    tags: Vec::new(),
                    owner_role_id,
                    created_at: now,
                };
                let specs = zyron_streaming::external_source::infer_schema_from_first_file(
                    &probe_entry,
                    creds_map.clone(),
                )
                .await
                .map_err(ProtocolError::Database)?;
                specs.into_iter().map(|c| (c.name, c.type_id)).collect()
            }
            zyron_catalog::ExternalFormat::Json
            | zyron_catalog::ExternalFormat::JsonLines
            | zyron_catalog::ExternalFormat::Csv => {
                return Err(ProtocolError::Database(ZyronError::PlanError(format!(
                    "external source format {:?} requires a COLUMNS clause, schema inference is only available for Parquet, Arrow IPC, and Avro",
                    format_kind
                ))));
            }
        }
    };

    let entry = zyron_catalog::ExternalSourceEntry {
        id: zyron_catalog::ExternalSourceId(0),
        schema_id: bound.schema_id,
        name: bound.name.clone(),
        backend: parser_backend_to_catalog(bound.backend),
        uri: bound.uri,
        format: format_kind,
        mode,
        schedule_cron,
        options: bound.options,
        columns,
        credential_key_id: sealed.as_ref().map(|s| s.key_id),
        credential_ciphertext: sealed.map(|s| s.ciphertext),
        classification: zyron_catalog::CatalogClassification::Internal,
        tags: Vec::new(),
        owner_role_id,
        created_at: now,
    };

    let has_creds = entry.credential_key_id.is_some();
    server
        .catalog
        .create_external_source(entry)
        .await
        .map_err(ProtocolError::Database)?;

    tracing::info!(
        target: "zyron::audit",
        event = "ExternalSourceCreated",
        object = %bound.name,
        schema_id = bound.schema_id.0,
        actor_role = actor_role_id(session),
        has_credentials = has_creds,
    );

    Ok(DdlResult::Tag("CREATE EXTERNAL SOURCE".to_string()))
}

async fn handle_create_external_sink(
    bound: zyron_planner::binder::BoundCreateExternalSink,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::CreateExternalSink,
        zyron_auth::ObjectType::Schema,
        bound.schema_id.0,
    )?;
    if !bound.credentials.is_empty() {
        check_ddl_privilege(
            server,
            session,
            zyron_auth::PrivilegeType::ManageExternalCredentials,
            zyron_auth::ObjectType::Schema,
            bound.schema_id.0,
        )?;
    }

    if server
        .catalog
        .get_external_sink(bound.schema_id, &bound.name)
        .is_some()
    {
        if bound.if_not_exists {
            return Ok(DdlResult::Tag("CREATE EXTERNAL SINK".to_string()));
        }
        return Err(ProtocolError::Database(ZyronError::Internal(format!(
            "external sink '{}' already exists",
            bound.name
        ))));
    }

    let creds_map: std::collections::HashMap<String, String> =
        bound.credentials.iter().cloned().collect();
    let sealed = if creds_map.is_empty() {
        None
    } else {
        Some(
            zyron_auth::seal_credentials(&creds_map, server.key_store.as_ref())
                .map_err(ProtocolError::Database)?,
        )
    };

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let owner_role_id = session
        .as_ref()
        .and_then(|s| s.security_context.as_ref())
        .map(|ctx| ctx.current_role.0)
        .unwrap_or(0);

    // Sinks cannot infer a schema, they produce rows whose shape is decided
    // by the streaming job binder. An explicit COLUMNS clause is carried
    // through when present so the persisted entry documents the layout.
    let entry = zyron_catalog::ExternalSinkEntry {
        id: zyron_catalog::ExternalSinkId(0),
        schema_id: bound.schema_id,
        name: bound.name.clone(),
        backend: parser_backend_to_catalog(bound.backend),
        uri: bound.uri,
        format: parser_format_to_catalog(bound.format),
        options: bound.options,
        columns: bound.columns.clone(),
        credential_key_id: sealed.as_ref().map(|s| s.key_id),
        credential_ciphertext: sealed.map(|s| s.ciphertext),
        classification: zyron_catalog::CatalogClassification::Internal,
        tags: Vec::new(),
        owner_role_id,
        created_at: now,
    };

    let has_creds = entry.credential_key_id.is_some();
    server
        .catalog
        .create_external_sink(entry)
        .await
        .map_err(ProtocolError::Database)?;

    tracing::info!(
        target: "zyron::audit",
        event = "ExternalSinkCreated",
        object = %bound.name,
        schema_id = bound.schema_id.0,
        actor_role = actor_role_id(session),
        has_credentials = has_creds,
    );

    Ok(DdlResult::Tag("CREATE EXTERNAL SINK".to_string()))
}

// ---------------------------------------------------------------------------
// DROP EXTERNAL SOURCE / SINK
// ---------------------------------------------------------------------------

/// Checks whether any persisted streaming job references the named external
/// source or sink. Re-parses each job's stored SQL and scans the bound form.
/// A parse or bind failure for a stored job is treated as non-blocking,
/// a malformed entry cannot reliably be shown to reference this object.
async fn external_endpoint_in_use(
    endpoint_name: &str,
    is_source: bool,
    server: &Arc<ServerState>,
) -> Option<String> {
    let jobs = server.catalog.list_streaming_jobs();
    for job in jobs {
        let statements = match zyron_parser::parse(&job.select_sql) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let stmt = match statements
            .into_iter()
            .find(|s| matches!(s, zyron_parser::Statement::CreateStreamingJob(_)))
        {
            Some(s) => s,
            None => continue,
        };
        // Recovery path: stored SQL is expected to be fully schema-qualified.
        let resolver = server
            .catalog
            .resolver(zyron_catalog::SYSTEM_DATABASE_ID, Vec::new());
        let mut binder = zyron_planner::Binder::new(resolver, &server.catalog);
        let bound = match binder.bind(stmt).await {
            Ok(b) => b,
            Err(_) => continue,
        };
        let bsj = match bound {
            zyron_planner::BoundStatement::CreateStreamingJob(b) => b,
            _ => continue,
        };
        if is_source {
            if let zyron_planner::binder::BoundStreamingSource::ExternalNamed {
                source_id, ..
            } = &bsj.source
            {
                if let Some(entry) = server.catalog.get_external_source_by_id(*source_id) {
                    if entry.name == endpoint_name {
                        return Some(job.name.clone());
                    }
                }
            }
        } else if let zyron_planner::binder::BoundStreamingSink::ExternalNamed { sink_id, .. } =
            &bsj.target
        {
            if let Some(entry) = server.catalog.get_external_sink_by_id(*sink_id) {
                if entry.name == endpoint_name {
                    return Some(job.name.clone());
                }
            }
        }
    }
    None
}

async fn handle_drop_external_source(
    name: &str,
    schema_id: zyron_catalog::SchemaId,
    if_exists: bool,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let entry = match server.catalog.get_external_source(schema_id, name) {
        Some(e) => e,
        None => {
            if if_exists {
                return Ok(DdlResult::Tag("DROP EXTERNAL SOURCE".to_string()));
            }
            return Err(ProtocolError::Database(ZyronError::Internal(format!(
                "external source '{}' not found",
                name
            ))));
        }
    };

    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::DropExternalSource,
        zyron_auth::ObjectType::ExternalSource,
        entry.id.0,
    )?;

    if let Some(job_name) = external_endpoint_in_use(name, true, server).await {
        return Err(ProtocolError::Database(ZyronError::Internal(format!(
            "cannot drop external source '{}': used by streaming job '{}'",
            name, job_name
        ))));
    }

    server
        .catalog
        .drop_external_source(schema_id, name)
        .await
        .map_err(ProtocolError::Database)?;

    tracing::info!(
        target: "zyron::audit",
        event = "ExternalSourceDropped",
        object = %name,
        schema_id = schema_id.0,
        actor_role = actor_role_id(session),
    );

    Ok(DdlResult::Tag("DROP EXTERNAL SOURCE".to_string()))
}

async fn handle_drop_external_sink(
    name: &str,
    schema_id: zyron_catalog::SchemaId,
    if_exists: bool,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let entry = match server.catalog.get_external_sink(schema_id, name) {
        Some(e) => e,
        None => {
            if if_exists {
                return Ok(DdlResult::Tag("DROP EXTERNAL SINK".to_string()));
            }
            return Err(ProtocolError::Database(ZyronError::Internal(format!(
                "external sink '{}' not found",
                name
            ))));
        }
    };

    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::DropExternalSink,
        zyron_auth::ObjectType::ExternalSink,
        entry.id.0,
    )?;

    if let Some(job_name) = external_endpoint_in_use(name, false, server).await {
        return Err(ProtocolError::Database(ZyronError::Internal(format!(
            "cannot drop external sink '{}': used by streaming job '{}'",
            name, job_name
        ))));
    }

    server
        .catalog
        .drop_external_sink(schema_id, name)
        .await
        .map_err(ProtocolError::Database)?;

    tracing::info!(
        target: "zyron::audit",
        event = "ExternalSinkDropped",
        object = %name,
        schema_id = schema_id.0,
        actor_role = actor_role_id(session),
    );

    Ok(DdlResult::Tag("DROP EXTERNAL SINK".to_string()))
}

// ---------------------------------------------------------------------------
// ALTER EXTERNAL SOURCE / SINK
// ---------------------------------------------------------------------------

async fn handle_alter_external_source(
    bound: zyron_planner::binder::BoundAlterExternalSource,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    use zyron_parser::ast::AlterExternalSourceAction;

    let entry = server
        .catalog
        .get_external_source(bound.schema_id, &bound.name)
        .ok_or_else(|| {
            ProtocolError::Database(ZyronError::Internal(format!(
                "external source '{}' not found",
                bound.name
            )))
        })?;

    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::AlterExternalSource,
        zyron_auth::ObjectType::ExternalSource,
        entry.id.0,
    )?;
    if matches!(bound.action, AlterExternalSourceAction::SetCredentials(_)) {
        check_ddl_privilege(
            server,
            session,
            zyron_auth::PrivilegeType::ManageExternalCredentials,
            zyron_auth::ObjectType::ExternalSource,
            entry.id.0,
        )?;
    }

    let action_kind_str: &'static str = match &bound.action {
        AlterExternalSourceAction::SetOptions(_) => "SetOptions",
        AlterExternalSourceAction::SetCredentials(_) => "SetCredentials",
        AlterExternalSourceAction::SetCredentialProvider(_) => "SetCredentialProvider",
        AlterExternalSourceAction::SetMode(_) => "SetMode",
        AlterExternalSourceAction::SetColumns(_) => "SetColumns",
        AlterExternalSourceAction::Rename(_) => "Rename",
        AlterExternalSourceAction::RefreshSchema => "RefreshSchema",
        AlterExternalSourceAction::ResetLsn(_) => "ResetLsn",
        AlterExternalSourceAction::Pause => "Pause",
        AlterExternalSourceAction::Resume => "Resume",
    };
    let mut updated = (*entry).clone();
    match bound.action {
        AlterExternalSourceAction::SetOptions(new_opts) => {
            // Replace any option keys present in new_opts, keep existing
            // keys not overridden so SET OPTIONS behaves as a merge.
            let mut map: std::collections::HashMap<String, String> =
                updated.options.into_iter().collect();
            for (k, v) in new_opts {
                map.insert(k, v);
            }
            updated.options = map.into_iter().collect();
        }
        AlterExternalSourceAction::SetCredentials(new_creds) => {
            let creds_map: std::collections::HashMap<String, String> =
                new_creds.into_iter().collect();
            if creds_map.is_empty() {
                updated.credential_key_id = None;
                updated.credential_ciphertext = None;
            } else {
                let sealed = zyron_auth::seal_credentials(&creds_map, server.key_store.as_ref())
                    .map_err(ProtocolError::Database)?;
                updated.credential_key_id = Some(sealed.key_id);
                updated.credential_ciphertext = Some(sealed.ciphertext);
            }
        }
        AlterExternalSourceAction::SetMode(mode_spec) => {
            let (mode, cron) = parser_mode_to_catalog(&mode_spec);
            updated.mode = mode;
            updated.schedule_cron = cron;
        }
        AlterExternalSourceAction::SetColumns(new_cols) => {
            // Replace the persisted column layout wholesale. Used after a
            // source file's schema changes, or to override a prior inference.
            updated.columns = new_cols
                .into_iter()
                .map(|(n, dt)| (n, (&dt).to_type_id()))
                .collect();
        }
        AlterExternalSourceAction::Rename(new_name) => {
            updated.name = new_name;
        }
        AlterExternalSourceAction::SetCredentialProvider(_)
        | AlterExternalSourceAction::RefreshSchema
        | AlterExternalSourceAction::ResetLsn(_)
        | AlterExternalSourceAction::Pause
        | AlterExternalSourceAction::Resume => {
            return Err(ProtocolError::Database(ZyronError::Internal(
                "ALTER EXTERNAL SOURCE action pending later phase wiring".to_string(),
            )));
        }
    }

    server
        .catalog
        .update_external_source(updated)
        .await
        .map_err(ProtocolError::Database)?;

    tracing::info!(
        target: "zyron::audit",
        event = "ExternalSourceAltered",
        object = %bound.name,
        schema_id = bound.schema_id.0,
        actor_role = actor_role_id(session),
        action = action_kind_str,
    );

    Ok(DdlResult::Tag("ALTER EXTERNAL SOURCE".to_string()))
}

async fn handle_alter_external_sink(
    bound: zyron_planner::binder::BoundAlterExternalSink,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    use zyron_parser::ast::AlterExternalSinkAction;

    let entry = server
        .catalog
        .get_external_sink(bound.schema_id, &bound.name)
        .ok_or_else(|| {
            ProtocolError::Database(ZyronError::Internal(format!(
                "external sink '{}' not found",
                bound.name
            )))
        })?;

    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::AlterExternalSink,
        zyron_auth::ObjectType::ExternalSink,
        entry.id.0,
    )?;
    if matches!(bound.action, AlterExternalSinkAction::SetCredentials(_)) {
        check_ddl_privilege(
            server,
            session,
            zyron_auth::PrivilegeType::ManageExternalCredentials,
            zyron_auth::ObjectType::ExternalSink,
            entry.id.0,
        )?;
    }

    let action_kind_str: &'static str = match &bound.action {
        AlterExternalSinkAction::SetOptions(_) => "SetOptions",
        AlterExternalSinkAction::SetCredentials(_) => "SetCredentials",
        AlterExternalSinkAction::Rename(_) => "Rename",
    };
    let mut updated = (*entry).clone();
    match bound.action {
        AlterExternalSinkAction::SetOptions(new_opts) => {
            let mut map: std::collections::HashMap<String, String> =
                updated.options.into_iter().collect();
            for (k, v) in new_opts {
                map.insert(k, v);
            }
            updated.options = map.into_iter().collect();
        }
        AlterExternalSinkAction::SetCredentials(new_creds) => {
            let creds_map: std::collections::HashMap<String, String> =
                new_creds.into_iter().collect();
            if creds_map.is_empty() {
                updated.credential_key_id = None;
                updated.credential_ciphertext = None;
            } else {
                let sealed = zyron_auth::seal_credentials(&creds_map, server.key_store.as_ref())
                    .map_err(ProtocolError::Database)?;
                updated.credential_key_id = Some(sealed.key_id);
                updated.credential_ciphertext = Some(sealed.ciphertext);
            }
        }
        AlterExternalSinkAction::Rename(new_name) => {
            updated.name = new_name;
        }
    }

    server
        .catalog
        .update_external_sink(updated)
        .await
        .map_err(ProtocolError::Database)?;

    tracing::info!(
        target: "zyron::audit",
        event = "ExternalSinkAltered",
        object = %bound.name,
        schema_id = bound.schema_id.0,
        actor_role = actor_role_id(session),
        action = action_kind_str,
    );

    Ok(DdlResult::Tag("ALTER EXTERNAL SINK".to_string()))
}

// ---------------------------------------------------------------------------
// Zyron-to-Zyron DDL: publications, endpoints, security map
// ---------------------------------------------------------------------------

/// Binds a Zyron-to-Zyron DDL statement through the planner binder and
/// dispatches to the matching handler. Covers publications, endpoints, and
/// security maps. DROP variants that do not need re-binding are handled
/// directly from the parser statement by their own dispatch arm.
async fn dispatch_z2z_statement(
    stmt: zyron_parser::Statement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let db_id = get_session_database(session)?;
    let search_path = session
        .as_ref()
        .map(|s| s.search_path.clone())
        .unwrap_or_default();

    let resolver = server.catalog.resolver(db_id, search_path);
    let mut binder = zyron_planner::Binder::new(resolver, &server.catalog);
    let bound = binder.bind(stmt).await.map_err(ProtocolError::Database)?;

    match bound {
        zyron_planner::BoundStatement::CreatePublication(b) => {
            handle_create_publication(*b, server, session).await
        }
        zyron_planner::BoundStatement::AlterPublication(b) => {
            handle_alter_publication(*b, server, session).await
        }
        zyron_planner::BoundStatement::CreateEndpoint(b) => {
            handle_create_endpoint(*b, server, session).await
        }
        zyron_planner::BoundStatement::CreateStreamingEndpoint(b) => {
            handle_create_streaming_endpoint(*b, server, session).await
        }
        zyron_planner::BoundStatement::AlterEndpoint(b) => {
            handle_alter_endpoint(*b, server, session).await
        }
        zyron_planner::BoundStatement::AlterSecurityMap(b) => {
            handle_alter_security_map(*b, server, session).await
        }
        zyron_planner::BoundStatement::DropSecurityMap(b) => {
            handle_drop_security_map(*b, server, session).await
        }
        _ => Err(ProtocolError::Database(ZyronError::PlanError(
            "expected Zyron-to-Zyron DDL statement".to_string(),
        ))),
    }
}

fn unix_now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn map_auth_security_map_kind(k: zyron_catalog::SecurityMapKind) -> zyron_auth::SecurityMapKind {
    match k {
        zyron_catalog::SecurityMapKind::K8sSa => zyron_auth::SecurityMapKind::K8sSa,
        zyron_catalog::SecurityMapKind::Jwt => zyron_auth::SecurityMapKind::Jwt,
        zyron_catalog::SecurityMapKind::MtlsSubject => zyron_auth::SecurityMapKind::MtlsSubject,
        zyron_catalog::SecurityMapKind::MtlsFingerprint => {
            zyron_auth::SecurityMapKind::MtlsFingerprint
        }
    }
}

async fn handle_create_publication(
    bound: zyron_planner::binder::BoundCreatePublication,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::CreatePublication,
        zyron_auth::ObjectType::Schema,
        bound.schema_id.0,
    )?;

    if server
        .catalog
        .get_publication(bound.schema_id, &bound.name)
        .is_some()
    {
        if bound.if_not_exists {
            return Ok(DdlResult::Tag("CREATE PUBLICATION".to_string()));
        }
        return Err(ProtocolError::Database(ZyronError::Internal(format!(
            "publication '{}' already exists",
            bound.name
        ))));
    }

    let owner_role_id = actor_role_id(session);
    let now = unix_now_secs();
    // Render the publication-level WHERE predicate to SQL against the first
    // member table's schema so the streaming read path re-parses and enforces
    // it. ColumnRefs resolve by column_id, table-specific, so a publication
    // predicate spanning a single column set binds against each member table.
    let where_predicate = match bound.where_predicate.as_ref() {
        None => None,
        Some(expr) => {
            let cols = bound
                .tables
                .first()
                .and_then(|t| server.catalog.get_table_by_id(t.table_id).ok())
                .map(|t| t.columns.clone())
                .unwrap_or_default();
            Some(
                zyron_planner::bound_predicate_sql::bound_predicate_to_sql(expr, &cols)
                    .ok_or_else(|| {
                        ProtocolError::Database(ZyronError::Internal(
                            "publication WHERE predicate is not a supported row filter".to_string(),
                        ))
                    })?,
            )
        }
    };
    // Column projection restricts the published columns. A publication-level
    // list is the union of the per-table column selections.
    let columns_projection: Vec<String> = {
        let mut names: Vec<String> = Vec::new();
        for tbl in &bound.tables {
            if let Ok(t) = server.catalog.get_table_by_id(tbl.table_id) {
                for cid in &tbl.columns {
                    if let Some(c) = t.columns.iter().find(|e| e.id == *cid) {
                        if !names.contains(&c.name) {
                            names.push(c.name.clone());
                        }
                    }
                }
            }
        }
        names
    };

    let entry = zyron_catalog::PublicationEntry {
        id: zyron_catalog::PublicationId(0),
        schema_id: bound.schema_id,
        name: bound.name.clone(),
        change_feed: bound.change_feed,
        row_format: bound.row_format,
        retention_days: bound.retention_days,
        retain_until_advance: bound.retain_until_subscribers_advance,
        max_rows_per_sec: if bound.max_rows_per_sec == 0 {
            None
        } else {
            Some(bound.max_rows_per_sec)
        },
        max_bytes_per_sec: if bound.max_bytes_per_sec == 0 {
            None
        } else {
            Some(bound.max_bytes_per_sec)
        },
        max_concurrent_subscribers: if bound.max_concurrent_subscribers == 0 {
            None
        } else {
            Some(bound.max_concurrent_subscribers)
        },
        classification: bound.classification,
        allow_initial_snapshot: bound.allow_initial_snapshot,
        where_predicate,
        columns_projection,
        rls_using_predicate: None,
        tags: Vec::new(),
        schema_fingerprint: bound.schema_fingerprint,
        owner_role_id,
        created_at: now,
    };

    let classification = entry.classification;
    let pub_id = {
        let mut temp = entry.clone();
        temp.id = zyron_catalog::PublicationId(0);
        // Insert publication first so add_publication_table can reference it.
        let mut e = temp;
        // Assign fresh id via catalog.update_publication style path: we mimic
        // external source flow by re-using the catalog's create path.
        e.id = zyron_catalog::PublicationId(0);
        // Catalog does not expose create_publication, emulate via update after
        // writing the DDL log. The project persists publications through
        // update_publication which acts as upsert.
        server
            .catalog
            .update_publication(e.clone())
            .await
            .map_err(ProtocolError::Database)?;
        e.id
    };

    for tbl in &bound.tables {
        let tbl_cols = server
            .catalog
            .get_table_by_id(tbl.table_id)
            .map(|t| t.columns.clone())
            .unwrap_or_default();
        let where_predicate = match tbl.where_predicate.as_ref() {
            None => None,
            Some(expr) => Some(
                zyron_planner::bound_predicate_sql::bound_predicate_to_sql(expr, &tbl_cols)
                    .ok_or_else(|| {
                        ProtocolError::Database(ZyronError::Internal(
                            "publication table WHERE predicate is not a supported row filter"
                                .to_string(),
                        ))
                    })?,
            ),
        };
        // Persist column names (not numeric ids) so the read path can prune by
        // name without re-resolving the catalog.
        let columns: Vec<String> = tbl
            .columns
            .iter()
            .filter_map(|c| tbl_cols.iter().find(|e| e.id == *c).map(|e| e.name.clone()))
            .collect();
        let tentry = zyron_catalog::PublicationTableEntry {
            id: 0,
            publication_id: pub_id,
            table_id: tbl.table_id,
            where_predicate,
            columns,
            created_at: now,
        };
        server
            .catalog
            .add_publication_table(tentry)
            .await
            .map_err(ProtocolError::Database)?;
    }

    tracing::info!(
        target: "zyron::audit",
        event = "PublicationCreated",
        name = %bound.name,
        schema_id = bound.schema_id.0,
        actor_role = owner_role_id,
        classification = ?classification,
    );

    Ok(DdlResult::Tag("CREATE PUBLICATION".to_string()))
}

async fn handle_alter_publication(
    bound: zyron_planner::binder::BoundAlterPublication,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    use zyron_planner::binder::BoundAlterPublicationAction;

    let current = server
        .catalog
        .get_publication(bound.schema_id, &bound.name)
        .ok_or_else(|| {
            ProtocolError::Database(ZyronError::Internal(format!(
                "publication '{}' not found",
                bound.name
            )))
        })?;

    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::AlterPublication,
        zyron_auth::ObjectType::Publication,
        current.id.0,
    )?;

    let now = unix_now_secs();
    let action_tag: &'static str = match &bound.action {
        BoundAlterPublicationAction::AddTable(_) => "AddTable",
        BoundAlterPublicationAction::DropTable(_) => "DropTable",
        BoundAlterPublicationAction::SetOptions(_) => "SetOptions",
        BoundAlterPublicationAction::SetWhere(_) => "SetWhere",
        BoundAlterPublicationAction::Rename(_) => "Rename",
    };

    match bound.action {
        BoundAlterPublicationAction::AddTable(t) => {
            let tbl_cols = server
                .catalog
                .get_table_by_id(t.table_id)
                .map(|e| e.columns.clone())
                .unwrap_or_default();
            let where_predicate = match t.where_predicate.as_ref() {
                None => None,
                Some(expr) => Some(
                    zyron_planner::bound_predicate_sql::bound_predicate_to_sql(expr, &tbl_cols)
                        .ok_or_else(|| {
                            ProtocolError::Database(ZyronError::Internal(
                                "publication table WHERE predicate is not a supported row filter"
                                    .to_string(),
                            ))
                        })?,
                ),
            };
            let columns: Vec<String> = t
                .columns
                .iter()
                .filter_map(|c| tbl_cols.iter().find(|e| e.id == *c).map(|e| e.name.clone()))
                .collect();
            let tentry = zyron_catalog::PublicationTableEntry {
                id: 0,
                publication_id: current.id,
                table_id: t.table_id,
                where_predicate,
                columns,
                created_at: now,
            };
            server
                .catalog
                .add_publication_table(tentry)
                .await
                .map_err(ProtocolError::Database)?;
        }
        BoundAlterPublicationAction::DropTable(tid) => {
            server
                .catalog
                .remove_publication_table(current.id, tid)
                .await
                .map_err(ProtocolError::Database)?;
        }
        BoundAlterPublicationAction::SetOptions(updates) => {
            let mut updated = (*current).clone();
            if let Some(v) = updates.retention_days {
                updated.retention_days = v;
            }
            if let Some(v) = updates.retain_until_subscribers_advance {
                updated.retain_until_advance = v;
            }
            if let Some(v) = updates.max_rows_per_sec {
                updated.max_rows_per_sec = if v == 0 { None } else { Some(v) };
            }
            if let Some(v) = updates.max_bytes_per_sec {
                updated.max_bytes_per_sec = if v == 0 { None } else { Some(v) };
            }
            if let Some(v) = updates.max_concurrent_subscribers {
                updated.max_concurrent_subscribers = if v == 0 { None } else { Some(v) };
            }
            if let Some(v) = updates.classification {
                updated.classification = v;
            }
            if let Some(v) = updates.allow_initial_snapshot {
                updated.allow_initial_snapshot = v;
            }
            if let Some(v) = updates.change_feed {
                updated.change_feed = v;
            }
            if let Some(v) = updates.row_format {
                updated.row_format = v;
            }
            server
                .catalog
                .update_publication(updated)
                .await
                .map_err(ProtocolError::Database)?;
        }
        BoundAlterPublicationAction::SetWhere(expr) => {
            // Render against the first member table's schema, matching the
            // CREATE PUBLICATION publication-level predicate storage.
            let cols = server
                .catalog
                .get_publication_tables(current.id)
                .first()
                .and_then(|pt| server.catalog.get_table_by_id(pt.table_id).ok())
                .map(|t| t.columns.clone())
                .unwrap_or_default();
            let rendered = zyron_planner::bound_predicate_sql::bound_predicate_to_sql(&expr, &cols)
                .ok_or_else(|| {
                    ProtocolError::Database(ZyronError::Internal(
                        "publication WHERE predicate is not a supported row filter".to_string(),
                    ))
                })?;
            let mut updated = (*current).clone();
            updated.where_predicate = Some(rendered);
            server
                .catalog
                .update_publication(updated)
                .await
                .map_err(ProtocolError::Database)?;
        }
        BoundAlterPublicationAction::Rename(new_name) => {
            let mut updated = (*current).clone();
            updated.name = new_name;
            server
                .catalog
                .update_publication(updated)
                .await
                .map_err(ProtocolError::Database)?;
        }
    }

    tracing::info!(
        target: "zyron::audit",
        event = "PublicationAltered",
        name = %bound.name,
        schema_id = bound.schema_id.0,
        actor_role = actor_role_id(session),
        action = action_tag,
    );

    Ok(DdlResult::Tag("ALTER PUBLICATION".to_string()))
}

async fn handle_drop_publication(
    stmt: &zyron_parser::ast::DropPublicationStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (_db_id, schema_id) = get_session_schema(session, server, None)?;
    let current = match server.catalog.get_publication(schema_id, &stmt.name) {
        Some(p) => p,
        None => {
            if stmt.if_exists {
                return Ok(DdlResult::Tag("DROP PUBLICATION".to_string()));
            }
            return Err(ProtocolError::Database(ZyronError::Internal(format!(
                "publication '{}' not found",
                stmt.name
            ))));
        }
    };
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::DropPublication,
        zyron_auth::ObjectType::Publication,
        current.id.0,
    )?;

    if !stmt.cascade {
        let subs = server.catalog.list_publication_subscribers(current.id);
        if !subs.is_empty() {
            return Err(ProtocolError::Database(ZyronError::Internal(format!(
                "publication '{}' has {} active subscribers, use CASCADE to force drop",
                stmt.name,
                subs.len()
            ))));
        }
    }

    server
        .catalog
        .drop_publication(schema_id, &stmt.name)
        .await
        .map_err(ProtocolError::Database)?;

    tracing::info!(
        target: "zyron::audit",
        event = "PublicationDropped",
        name = %stmt.name,
        schema_id = schema_id.0,
        actor_role = actor_role_id(session),
        cascade = stmt.cascade,
    );
    Ok(DdlResult::Tag("DROP PUBLICATION".to_string()))
}

async fn handle_tag_publication(
    stmt: &zyron_parser::ast::TagPublicationStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (_db_id, schema_id) = get_session_schema(session, server, None)?;
    let current = server
        .catalog
        .get_publication(schema_id, &stmt.name)
        .ok_or_else(|| {
            ProtocolError::Database(ZyronError::Internal(format!(
                "publication '{}' not found",
                stmt.name
            )))
        })?;
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::AlterPublication,
        zyron_auth::ObjectType::Publication,
        current.id.0,
    )?;

    let mut updated = (*current).clone();
    for t in &stmt.tags {
        if !updated.tags.iter().any(|x| x == t) {
            updated.tags.push(t.clone());
        }
    }
    server
        .catalog
        .update_publication(updated)
        .await
        .map_err(ProtocolError::Database)?;

    tracing::info!(
        target: "zyron::audit",
        event = "PublicationTagged",
        name = %stmt.name,
        actor_role = actor_role_id(session),
        tags = ?stmt.tags,
    );
    Ok(DdlResult::Tag("TAG PUBLICATION".to_string()))
}

async fn handle_untag_publication(
    stmt: &zyron_parser::ast::UntagPublicationStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (_db_id, schema_id) = get_session_schema(session, server, None)?;
    let current = server
        .catalog
        .get_publication(schema_id, &stmt.name)
        .ok_or_else(|| {
            ProtocolError::Database(ZyronError::Internal(format!(
                "publication '{}' not found",
                stmt.name
            )))
        })?;
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::AlterPublication,
        zyron_auth::ObjectType::Publication,
        current.id.0,
    )?;
    let mut updated = (*current).clone();
    updated.tags.retain(|t| t != &stmt.tag);
    server
        .catalog
        .update_publication(updated)
        .await
        .map_err(ProtocolError::Database)?;

    tracing::info!(
        target: "zyron::audit",
        event = "PublicationUntagged",
        name = %stmt.name,
        actor_role = actor_role_id(session),
        tag = %stmt.tag,
    );
    Ok(DdlResult::Tag("UNTAG PUBLICATION".to_string()))
}

fn methods_planner_to_catalog(
    methods: &[zyron_catalog::HttpMethod],
) -> Vec<zyron_catalog::HttpMethod> {
    methods.to_vec()
}

async fn handle_create_endpoint(
    bound: zyron_planner::binder::BoundCreateEndpoint,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Schema,
        bound.schema_id.0,
    )?;

    if server.catalog.get_endpoint_by_path(&bound.path).is_some() {
        if bound.if_not_exists {
            return Ok(DdlResult::Tag("CREATE ENDPOINT".to_string()));
        }
        return Err(ProtocolError::Database(ZyronError::Internal(format!(
            "endpoint path '{}' already in use",
            bound.path
        ))));
    }

    let now = unix_now_secs();
    let owner_role_id = actor_role_id(session);
    let entry = zyron_catalog::EndpointEntry {
        id: zyron_catalog::EndpointId(0),
        schema_id: bound.schema_id,
        name: bound.name.clone(),
        kind: zyron_catalog::EndpointKind::Rest,
        path: bound.path.clone(),
        methods: methods_planner_to_catalog(&bound.methods),
        sql_body: bound.sql.clone(),
        backed_publication_id: None,
        auth_mode: bound.auth,
        required_scopes: bound.required_scopes,
        output_format: Some(bound.output_format),
        cors_origins: bound.cors_origins,
        rate_limit: bound.rate_limit,
        cache_seconds: Some(bound.cache_seconds),
        timeout_seconds: Some(bound.timeout_seconds),
        max_request_body_kb: Some(bound.max_body_bytes / 1024),
        message_format: None,
        heartbeat_seconds: None,
        backpressure: None,
        max_connections: None,
        enabled: true,
        owner_role_id,
        created_at: now,
    };

    let created_id = server
        .catalog
        .create_endpoint(entry)
        .await
        .map_err(ProtocolError::Database)?;

    // Push the newly persisted entry into the live gateway router so HTTP
    // requests start resolving immediately. A registration failure means the
    // endpoint is not actually live, so roll back the catalog row and surface
    // the error rather than reporting a success the client cannot use.
    if let Some(ref registrar) = server.endpoint_registrar {
        if let Some(new_entry) = server.catalog.get_endpoint_by_id(created_id) {
            if let Err(e) = registrar.register(&new_entry).await {
                tracing::error!(
                    target: "zyron::gateway",
                    name = %bound.name,
                    path = %bound.path,
                    error = %e,
                    "endpoint router registration failed after catalog create"
                );
                let _ = server
                    .catalog
                    .drop_endpoint(bound.schema_id, &bound.name)
                    .await;
                return Err(ProtocolError::Database(e));
            }
        }
    }

    tracing::info!(
        target: "zyron::audit",
        event = "EndpointCreated",
        name = %bound.name,
        path = %bound.path,
        schema_id = bound.schema_id.0,
        actor_role = owner_role_id,
    );
    Ok(DdlResult::Tag("CREATE ENDPOINT".to_string()))
}

async fn handle_create_streaming_endpoint(
    bound: zyron_planner::binder::BoundCreateStreamingEndpoint,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Schema,
        bound.schema_id.0,
    )?;
    if server.catalog.get_endpoint_by_path(&bound.path).is_some() {
        if bound.if_not_exists {
            return Ok(DdlResult::Tag("CREATE STREAMING ENDPOINT".to_string()));
        }
        return Err(ProtocolError::Database(ZyronError::Internal(format!(
            "endpoint path '{}' already in use",
            bound.path
        ))));
    }

    use zyron_parser::ast::StreamingEndpointProtocol;
    let kind = match bound.protocol {
        StreamingEndpointProtocol::Websocket => zyron_catalog::EndpointKind::WebSocket,
        StreamingEndpointProtocol::Sse => zyron_catalog::EndpointKind::Sse,
    };

    let now = unix_now_secs();
    let owner_role_id = actor_role_id(session);
    let entry = zyron_catalog::EndpointEntry {
        id: zyron_catalog::EndpointId(0),
        schema_id: bound.schema_id,
        name: bound.name.clone(),
        kind,
        path: bound.path.clone(),
        methods: vec![zyron_catalog::HttpMethod::Get],
        sql_body: String::new(),
        backed_publication_id: Some(bound.backing_publication_id),
        auth_mode: bound.auth,
        required_scopes: bound.required_scopes,
        output_format: None,
        cors_origins: Vec::new(),
        rate_limit: None,
        cache_seconds: None,
        timeout_seconds: None,
        max_request_body_kb: None,
        message_format: Some(bound.message_format),
        heartbeat_seconds: Some(bound.heartbeat_seconds),
        backpressure: Some(bound.backpressure),
        max_connections: Some(bound.max_connections),
        enabled: true,
        owner_role_id,
        created_at: now,
    };

    let created_id = server
        .catalog
        .create_endpoint(entry)
        .await
        .map_err(ProtocolError::Database)?;

    if let Some(ref registrar) = server.endpoint_registrar {
        if let Some(new_entry) = server.catalog.get_endpoint_by_id(created_id) {
            if let Err(e) = registrar.register(&new_entry).await {
                tracing::error!(
                    target: "zyron::gateway",
                    name = %bound.name,
                    path = %bound.path,
                    error = %e,
                    "streaming endpoint router registration failed after catalog create"
                );
                let _ = server
                    .catalog
                    .drop_endpoint(bound.schema_id, &bound.name)
                    .await;
                return Err(ProtocolError::Database(e));
            }
        }
    }

    tracing::info!(
        target: "zyron::audit",
        event = "StreamingEndpointCreated",
        name = %bound.name,
        path = %bound.path,
        actor_role = owner_role_id,
    );
    Ok(DdlResult::Tag("CREATE STREAMING ENDPOINT".to_string()))
}

async fn handle_alter_endpoint(
    bound: zyron_planner::binder::BoundAlterEndpoint,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    use zyron_planner::binder::BoundAlterEndpointAction;

    let current = server
        .catalog
        .get_endpoint(bound.schema_id, &bound.name)
        .ok_or_else(|| {
            ProtocolError::Database(ZyronError::Internal(format!(
                "endpoint '{}' not found",
                bound.name
            )))
        })?;
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Endpoint,
        current.id.0,
    )?;

    let action_tag: &'static str = match &bound.action {
        BoundAlterEndpointAction::Enable => "Enable",
        BoundAlterEndpointAction::Disable => "Disable",
        BoundAlterEndpointAction::SetOptions(_) => "SetOptions",
    };

    match bound.action {
        BoundAlterEndpointAction::Enable => {
            server
                .catalog
                .set_endpoint_enabled(current.id, true)
                .await
                .map_err(ProtocolError::Database)?;
            if let Some(ref registrar) = server.endpoint_registrar {
                if let Some(refreshed) = server.catalog.get_endpoint_by_id(current.id) {
                    if let Err(e) = registrar.set_enabled(&refreshed, true).await {
                        tracing::warn!(
                            target: "zyron::gateway",
                            name = %bound.name,
                            error = %e,
                            "endpoint enable router sync failed"
                        );
                    }
                }
            }
        }
        BoundAlterEndpointAction::Disable => {
            server
                .catalog
                .set_endpoint_enabled(current.id, false)
                .await
                .map_err(ProtocolError::Database)?;
            if let Some(ref registrar) = server.endpoint_registrar {
                if let Err(e) = registrar.set_enabled(&current, false).await {
                    tracing::warn!(
                        target: "zyron::gateway",
                        name = %bound.name,
                        error = %e,
                        "endpoint disable router sync failed"
                    );
                }
            }
        }
        BoundAlterEndpointAction::SetOptions(updates) => {
            let mut updated = (*current).clone();
            if let Some(v) = updates.cache_seconds {
                updated.cache_seconds = Some(v);
            }
            if let Some(v) = updates.timeout_seconds {
                updated.timeout_seconds = Some(v);
            }
            if let Some(v) = updates.max_body_bytes {
                updated.max_request_body_kb = Some(v / 1024);
            }
            if let Some(v) = updates.heartbeat_seconds {
                updated.heartbeat_seconds = Some(v);
            }
            if let Some(v) = updates.max_connections {
                updated.max_connections = Some(v);
            }
            server
                .catalog
                .update_endpoint(updated)
                .await
                .map_err(ProtocolError::Database)?;
            // Unregister and re-register so the compiled route picks up the
            // new options. The fresh read from the catalog ensures we route
            // against the post-update state.
            if let Some(ref registrar) = server.endpoint_registrar {
                let _ = registrar.unregister(current.id).await;
                if let Some(refreshed) = server.catalog.get_endpoint_by_id(current.id) {
                    if refreshed.enabled {
                        if let Err(e) = registrar.register(&refreshed).await {
                            tracing::warn!(
                                target: "zyron::gateway",
                                name = %bound.name,
                                error = %e,
                                "endpoint re-register after SetOptions failed"
                            );
                        }
                    }
                }
            }
        }
    }

    tracing::info!(
        target: "zyron::audit",
        event = "EndpointAltered",
        name = %bound.name,
        actor_role = actor_role_id(session),
        action = action_tag,
    );
    Ok(DdlResult::Tag("ALTER ENDPOINT".to_string()))
}

async fn handle_drop_endpoint(
    stmt: &zyron_parser::ast::DropEndpointStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (_db_id, schema_id) = get_session_schema(session, server, None)?;
    let current = match server.catalog.get_endpoint(schema_id, &stmt.name) {
        Some(e) => e,
        None => {
            if stmt.if_exists {
                return Ok(DdlResult::Tag("DROP ENDPOINT".to_string()));
            }
            return Err(ProtocolError::Database(ZyronError::Internal(format!(
                "endpoint '{}' not found",
                stmt.name
            ))));
        }
    };
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Endpoint,
        current.id.0,
    )?;

    server
        .catalog
        .drop_endpoint(schema_id, &stmt.name)
        .await
        .map_err(ProtocolError::Database)?;

    if let Some(ref registrar) = server.endpoint_registrar {
        if let Err(e) = registrar.unregister(current.id).await {
            tracing::warn!(
                target: "zyron::gateway",
                name = %stmt.name,
                error = %e,
                "endpoint router unregister failed"
            );
        }
    }

    tracing::info!(
        target: "zyron::audit",
        event = "EndpointDropped",
        name = %stmt.name,
        actor_role = actor_role_id(session),
    );
    Ok(DdlResult::Tag("DROP ENDPOINT".to_string()))
}

async fn handle_alter_security_map(
    bound: zyron_planner::binder::BoundAlterSecurityMap,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::ManageAuthRules,
        zyron_auth::ObjectType::System,
        0,
    )?;

    let sm = match server.security_manager.as_ref() {
        Some(sm) => sm,
        None => {
            return Err(ProtocolError::Database(ZyronError::Internal(
                "security manager not configured".to_string(),
            )));
        }
    };

    let role_id = sm
        .lookup_role(&bound.role_name)
        .map(|r| r.id.0)
        .ok_or_else(|| {
            ProtocolError::Database(ZyronError::RoleNotFound(bound.role_name.clone()))
        })?;

    let entry = zyron_catalog::SecurityMapEntry {
        id: zyron_catalog::SecurityMapId(0),
        kind: bound.kind,
        key: bound.identity_key.clone(),
        role_id,
        created_at: unix_now_secs(),
    };
    server
        .catalog
        .create_security_map(entry)
        .await
        .map_err(ProtocolError::Database)?;

    let auth_kind = map_auth_security_map_kind(bound.kind);
    let auth_entry = zyron_auth::SecurityMapEntry {
        kind: auth_kind,
        key: bound.identity_key.clone(),
        role: zyron_auth::RoleId(role_id),
    };
    let mut snap = sm.security_map.snapshot();
    snap.push(auth_entry);
    sm.security_map.load(snap);

    tracing::info!(
        target: "zyron::audit",
        event = "SecurityMapAltered",
        kind = ?bound.kind,
        key = %bound.identity_key,
        role = %bound.role_name,
        actor_role = actor_role_id(session),
    );
    Ok(DdlResult::Tag("ALTER SECURITY MAP".to_string()))
}

async fn handle_drop_security_map(
    bound: zyron_planner::binder::BoundDropSecurityMap,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::ManageAuthRules,
        zyron_auth::ObjectType::System,
        0,
    )?;

    // Find the catalog entry and drop it.
    for entry in server.catalog.list_security_maps() {
        if entry.kind == bound.kind && entry.key == bound.identity_key {
            server
                .catalog
                .drop_security_map(entry.id)
                .await
                .map_err(ProtocolError::Database)?;
            break;
        }
    }

    if let Some(sm) = server.security_manager.as_ref() {
        let auth_kind = map_auth_security_map_kind(bound.kind);
        sm.security_map.unmap(auth_kind, &bound.identity_key);
    }

    tracing::info!(
        target: "zyron::audit",
        event = "SecurityMapDropped",
        kind = ?bound.kind,
        key = %bound.identity_key,
        actor_role = actor_role_id(session),
    );
    Ok(DdlResult::Tag("DROP SECURITY MAP".to_string()))
}

// ---------------------------------------------------------------------------
// Feature store and ML model handlers
//
// Materialization executor implementation lives in
// `crate::feature_materialization_executor` and is registered with the
// server's background worker via `install_materialization_executor`.
// The DDL handlers below are sync entry points called from the wire path
// ---------------------------------------------------------------------------

/// Render an expression AST back to a SQL string for storage and lineage
fn renderExpr(expr: &zyron_parser::ast::Expr) -> String {
    // Render the transform expression back to SQL so the stored feature
    // definition re-parses, rather than a debug form that does not round-trip.
    zyron_parser::expr_to_sql(expr)
}

/// Convert an AST DataType to a canonical type label
fn renderDataType(dt: &Option<zyron_parser::ast::DataType>) -> String {
    match dt {
        Some(t) => format!("{:?}", t).to_uppercase(),
        None => "FLOAT64".to_string(),
    }
}

/// Parses a human-readable interval like "1 hour", "30 minutes", "10s"
fn parseIntervalSeconds(spec: &str) -> Option<u64> {
    let s = spec.trim().to_ascii_lowercase();
    let mut split = s.split_whitespace();
    let n: f64 = split.next()?.parse().ok().or_else(|| {
        // Maybe "1h" / "30m" combined form
        let mut digits = String::new();
        let mut rest = String::new();
        let mut seen_unit = false;
        for c in s.chars() {
            if !seen_unit && (c.is_ascii_digit() || c == '.') {
                digits.push(c);
            } else {
                seen_unit = true;
                rest.push(c);
            }
        }
        let v: f64 = digits.parse().ok()?;
        let mult = unitToSeconds(rest.trim())?;
        Some(v * mult as f64)
    })?;
    let unit = split.next().unwrap_or("seconds");
    let mult = unitToSeconds(unit)?;
    Some((n * mult as f64) as u64)
}

fn unitToSeconds(unit: &str) -> Option<u64> {
    match unit.trim_end_matches('s') {
        "second" | "sec" | "s" | "" => Some(1),
        "minute" | "min" | "m" => Some(60),
        "hour" | "hr" | "h" => Some(3600),
        "day" | "d" => Some(86_400),
        "week" | "w" => Some(604_800),
        _ => None,
    }
}

async fn handle_create_feature_group(
    stmt: &zyron_parser::ast::CreateFeatureGroupStatement,
    server: &Arc<ServerState>,
) -> Result<DdlResult, ProtocolError> {
    use zyron_analytics::featureLineage::{LineageEntry, extractTablesAndColumns};
    use zyron_analytics::featureStore::{FeatureDefinition, FeatureGroup};

    let nowMs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0);

    if stmt.if_not_exists && server.feature_store.group(&stmt.name).is_some() {
        return Ok(DdlResult::Tag("CREATE FEATURE GROUP".to_string()));
    }

    let mut group = FeatureGroup::new(stmt.name.clone(), stmt.entity_key.clone());
    if let Some(ref iv) = stmt.refresh_interval {
        if let Some(secs) = parseIntervalSeconds(iv) {
            group.refreshSeconds = secs;
        } else {
            return Err(ProtocolError::Database(ZyronError::ParseError(format!(
                "invalid REFRESH EVERY interval '{}'",
                iv
            ))));
        }
    }
    for opt in &stmt.options {
        match opt.key.as_str() {
            "max_staleness_seconds" => {
                if let Some(v) = optionAsU64(&opt.value) {
                    group.maxStalenessSeconds = v;
                }
            }
            "retention_days" => {
                if let Some(v) = optionAsU64(&opt.value) {
                    group.retentionDays = v;
                }
            }
            _ => {}
        }
    }
    if let Some(ref sq) = stmt.source_query {
        group.sourceQuery = format!("{:?}", sq);
    }
    let backingTableName = format!("_feature_{}", stmt.name);
    group.backingTable = Some(backingTableName.clone());

    let mut lineageGuard = server.feature_lineage.write();
    for fdef in &stmt.features {
        let transformText = renderExpr(&fdef.transform_expr);
        let mut def = FeatureDefinition::new(
            fdef.name.clone(),
            renderDataType(&fdef.data_type),
            transformText.clone(),
        );
        def.createdAtMs = nowMs;
        group.addFeature(def);

        let (tables, cols) = extractTablesAndColumns(&transformText);
        let qualifiedName = format!("{}.{}", stmt.name, fdef.name);
        let mut entry = LineageEntry::new(qualifiedName.clone());
        entry.sourceTables = tables;
        entry.sourceColumns = cols;
        entry.transformChain = vec![transformText];
        entry.lastComputedMs = 0;
        lineageGuard.register(qualifiedName, entry);
    }
    drop(lineageGuard);

    server
        .feature_store
        .registerFeatureGroup(group)
        .map_err(ProtocolError::Database)?;

    snapshotFeatureStore(server).map_err(ProtocolError::Database)?;

    tracing::info!(
        target: "zyron::audit",
        event = "FeatureGroupCreated",
        name = %stmt.name,
        entity_key = %stmt.entity_key,
        feature_count = stmt.features.len(),
    );
    Ok(DdlResult::Tag("CREATE FEATURE GROUP".to_string()))
}

async fn handle_drop_feature_group(
    stmt: &zyron_parser::ast::DropFeatureGroupStatement,
    server: &Arc<ServerState>,
) -> Result<DdlResult, ProtocolError> {
    let existed = server.feature_store.group(&stmt.name).is_some();
    if !existed && !stmt.if_exists {
        return Err(ProtocolError::Database(ZyronError::ExecutionError(
            format!("feature group '{}' not found", stmt.name),
        )));
    }
    if existed {
        server
            .feature_store
            .dropFeatureGroup(&stmt.name)
            .map_err(ProtocolError::Database)?;
        snapshotFeatureStore(server).map_err(ProtocolError::Database)?;
    }
    Ok(DdlResult::Tag("DROP FEATURE GROUP".to_string()))
}

async fn handle_create_model(
    stmt: &zyron_parser::ast::CreateModelStatement,
    server: &Arc<ServerState>,
    _session: &mut Option<Session>,
    _txn: &mut Option<zyron_storage::txn::Transaction>,
    raw_sql: &str,
) -> Result<DdlResult, ProtocolError> {
    use zyron_analytics::ml::{
        Hyperparameters, ModelConfig, ModelType, TrainingData, decisionTree, gradientBoosting,
        kmeans, knn, linearRegression, logisticRegression, randomForest,
    };

    let nowMs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0);

    if stmt.if_not_exists && server.model_cache.get(&stmt.name).is_some() {
        return Ok(DdlResult::Tag("CREATE MODEL".to_string()));
    }

    let modelType = ModelType::fromStr(&stmt.model_type).ok_or_else(|| {
        ProtocolError::Database(ZyronError::ParseError(format!(
            "unknown model type '{}'",
            stmt.model_type
        )))
    })?;

    let mut hyperparameters = Hyperparameters::new();
    for opt in &stmt.options {
        match &opt.value {
            zyron_parser::ast::TableOptionValue::String(s) => {
                hyperparameters.setStr(&opt.key, s);
            }
            zyron_parser::ast::TableOptionValue::Integer(n) => {
                hyperparameters.setF64(&opt.key, *n as f64);
            }
            zyron_parser::ast::TableOptionValue::Boolean(b) => {
                hyperparameters.setF64(&opt.key, if *b { 1.0 } else { 0.0 });
            }
            zyron_parser::ast::TableOptionValue::Identifier(s) => {
                hyperparameters.setStr(&opt.key, s);
            }
            zyron_parser::ast::TableOptionValue::StringList(_) => {}
        }
    }

    let trainingQuery = stmt.training_query.as_ref().ok_or_else(|| {
        ProtocolError::Database(ZyronError::ParseError(
            "CREATE MODEL requires USING (query) clause".to_string(),
        ))
    })?;

    // Materialize the training query via the executor
    let (xs, ys, rowCount) = collectTrainingRowsFromQuery(
        server,
        trainingQuery,
        &stmt.features,
        stmt.target.as_deref(),
        raw_sql,
    )
    .await
    .map_err(ProtocolError::Database)?;

    if rowCount == 0 {
        return Err(ProtocolError::Database(ZyronError::ExecutionError(
            "training query returned no rows".to_string(),
        )));
    }
    let p = stmt.features.len();
    let data = TrainingData::new(&xs, &ys, rowCount, p);

    let mut config = ModelConfig::new(modelType, stmt.features.clone());
    config.targetColumn = stmt.target.clone();
    config.hyperparameters = hyperparameters;

    let mut trained = match modelType {
        ModelType::LinearRegression => linearRegression::train(&config, &data),
        ModelType::LogisticRegression => logisticRegression::train(&config, &data),
        ModelType::DecisionTreeRegression | ModelType::DecisionTreeClassification => {
            decisionTree::train(&config, &data)
        }
        ModelType::RandomForestRegression | ModelType::RandomForestClassification => {
            randomForest::train(&config, &data)
        }
        ModelType::GradientBoostingRegression | ModelType::GradientBoostingClassification => {
            gradientBoosting::train(&config, &data)
        }
        ModelType::KMeans => kmeans::train(&config, &data),
        ModelType::KnnRegression | ModelType::KnnClassification => knn::train(&config, &data),
    }
    .map_err(ProtocolError::Database)?;

    trained.modelId = stmt.name.clone();
    trained.createdAtMs = nowMs;

    persistModel(server, &stmt.name, &trained).map_err(ProtocolError::Database)?;
    server.model_cache.install(stmt.name.clone(), trained);

    tracing::info!(
        target: "zyron::audit",
        event = "ModelCreated",
        name = %stmt.name,
        model_type = %stmt.model_type,
        training_rows = rowCount,
    );

    Ok(DdlResult::Tag("CREATE MODEL".to_string()))
}

async fn handle_drop_model(
    stmt: &zyron_parser::ast::DropModelStatement,
    server: &Arc<ServerState>,
) -> Result<DdlResult, ProtocolError> {
    if server.model_cache.get(&stmt.name).is_none() && !stmt.if_exists {
        return Err(ProtocolError::Database(ZyronError::ExecutionError(
            format!("model '{}' not found", stmt.name),
        )));
    }
    server.model_cache.invalidate(&stmt.name);
    // Surface an on-disk delete failure rather than swallowing it, otherwise a
    // stale model file survives the drop and reloads on restart.
    removeModelFile(server, &stmt.name).map_err(ProtocolError::Database)?;
    Ok(DdlResult::Tag("DROP MODEL".to_string()))
}

fn optionAsU64(v: &zyron_parser::ast::TableOptionValue) -> Option<u64> {
    match v {
        zyron_parser::ast::TableOptionValue::Integer(n) => Some(*n as u64),
        zyron_parser::ast::TableOptionValue::String(s) => s.parse().ok(),
        _ => None,
    }
}

/// Runs the training SELECT and materializes feature columns and target
async fn collectTrainingRowsFromQuery(
    server: &Arc<ServerState>,
    query: &zyron_parser::ast::SelectStatement,
    featureColumns: &[String],
    targetColumn: Option<&str>,
    _raw_sql: &str,
) -> Result<(Vec<f64>, Vec<f64>, usize), ZyronError> {
    use zyron_executor::column::ScalarValue;
    use zyron_executor::context::ExecutionContext;

    let stmt = zyron_parser::Statement::Select(Box::new(query.clone()));
    let database_id = zyron_catalog::DatabaseId(1);
    let search_path: Vec<String> = vec!["public".to_string()];
    let plan = zyron_planner::plan(
        &server.catalog,
        database_id,
        search_path,
        stmt,
        Some(&server.peer_facts()),
    )
    .await?;

    let schema = plan.output_schema();
    let mut featureIndices: Vec<usize> = Vec::with_capacity(featureColumns.len());
    for f in featureColumns {
        let idx = schema
            .iter()
            .position(|c| c.name.eq_ignore_ascii_case(f))
            .ok_or_else(|| ZyronError::ColumnNotFound(f.clone()))?;
        featureIndices.push(idx);
    }
    let targetIdx = match targetColumn {
        Some(t) => Some(
            schema
                .iter()
                .position(|c| c.name.eq_ignore_ascii_case(t))
                .ok_or_else(|| ZyronError::ColumnNotFound(t.to_string()))?,
        ),
        None => None,
    };

    let mut read_txn = server
        .txn_manager
        .begin(zyron_storage::txn::IsolationLevel::ReadCommitted)?;
    let snapshot = read_txn.snapshot.clone();
    let txn_id_u32 = u32::try_from(read_txn.txn_id)
        .map_err(|_| ZyronError::ExecutionError("txn_id overflow".into()))?;
    let ctx = Arc::new(ExecutionContext::new(
        server.catalog.clone(),
        server.wal.clone(),
        server.buffer_pool.clone(),
        server.disk_manager.clone(),
        txn_id_u32,
        snapshot,
    ));
    let result = zyron_executor::execute(plan, &ctx).await;
    let _ = server.txn_manager.abort(&mut read_txn);
    let batches = result?;

    let mut xs: Vec<f64> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();
    let mut n: usize = 0;
    for batch in &batches {
        let rows = batch.num_rows;
        for r in 0..rows {
            for &fi in &featureIndices {
                xs.push(scalarToF64(&batch.column(fi).get_scalar(r)));
            }
            if let Some(ti) = targetIdx {
                ys.push(scalarToF64(&batch.column(ti).get_scalar(r)));
            } else {
                ys.push(0.0);
            }
            n += 1;
        }
    }

    fn scalarToF64(v: &ScalarValue) -> f64 {
        match v {
            ScalarValue::Null => f64::NAN,
            ScalarValue::Boolean(b) => {
                if *b {
                    1.0
                } else {
                    0.0
                }
            }
            ScalarValue::Int8(x) => *x as f64,
            ScalarValue::Int16(x) => *x as f64,
            ScalarValue::Int32(x) => *x as f64,
            ScalarValue::Int64(x) => *x as f64,
            ScalarValue::Int128(x) => *x as f64,
            ScalarValue::UInt8(x) => *x as f64,
            ScalarValue::UInt16(x) => *x as f64,
            ScalarValue::UInt32(x) => *x as f64,
            ScalarValue::UInt64(x) => *x as f64,
            ScalarValue::Float32(f) => *f as f64,
            ScalarValue::Float64(f) => *f,
            ScalarValue::Utf8(_) | ScalarValue::Binary(_) | ScalarValue::FixedBinary16(_) => 0.0,
            ScalarValue::Interval(_) => 0.0,
        }
    }

    Ok((xs, ys, n))
}

/// Where the feature store snapshot lives.
///
/// Named once because a writer and a reader that spelled this out separately
/// would still agree today and stop agreeing the first time either moved.
pub fn featureStoreSnapshotPath(server: &Arc<ServerState>) -> std::path::PathBuf {
    server.data_dir.join("feature_store.json")
}

/// Where trained model files live, named once for the same reason.
pub fn modelsDir(server: &Arc<ServerState>) -> std::path::PathBuf {
    server.data_dir.join("models")
}

/// Debounced async snapshot writer. DDL handlers mark the store dirty
/// and the worker flushes after a quiet period, coalescing bursts of
/// CREATE/DROP into one write to disk. The worker holds only a Weak
/// reference to ServerState so the server can drop cleanly; the worker
/// thread itself self-terminates when its sentinel `shutdown` flag is
/// set, joining synchronously through the public shutdown method
pub struct SnapshotFlusher {
    dirty: std::sync::atomic::AtomicBool,
    shutdown: std::sync::atomic::AtomicBool,
    waker: std::sync::OnceLock<std::thread::Thread>,
    path: std::path::PathBuf,
    server: std::sync::Weak<ServerState>,
    handle: std::sync::Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl SnapshotFlusher {
    /// Signal the worker to exit and join it. Idempotent; multiple calls
    /// return immediately after the first
    pub fn shutdown(&self) {
        self.shutdown
            .store(true, std::sync::atomic::Ordering::Release);
        if let Some(t) = self.waker.get() {
            t.unpark();
        }
        if let Ok(mut guard) = self.handle.lock() {
            if let Some(h) = guard.take() {
                let _ = h.join();
            }
        }
    }
}

impl Drop for SnapshotFlusher {
    fn drop(&mut self) {
        self.shutdown();
    }
}

static SNAPSHOT_FLUSHER: std::sync::OnceLock<Arc<SnapshotFlusher>> = std::sync::OnceLock::new();

fn ensureSnapshotFlusher(server: &Arc<ServerState>) -> Arc<SnapshotFlusher> {
    SNAPSHOT_FLUSHER
        .get_or_init(|| {
            let f = Arc::new(SnapshotFlusher {
                dirty: std::sync::atomic::AtomicBool::new(false),
                shutdown: std::sync::atomic::AtomicBool::new(false),
                waker: std::sync::OnceLock::new(),
                path: featureStoreSnapshotPath(server),
                server: Arc::downgrade(server),
                handle: std::sync::Mutex::new(None),
            });
            let worker = Arc::clone(&f);
            let handle = std::thread::Builder::new()
                .name("zyron-feature-snapshot".into())
                .spawn(move || {
                    let _ = worker.waker.set(std::thread::current());
                    loop {
                        std::thread::park();
                        if worker.shutdown.load(std::sync::atomic::Ordering::Acquire) {
                            return;
                        }
                        // 200ms debounce window. Re-check shutdown at the
                        // end so a shutdown signal during sleep also exits
                        std::thread::sleep(std::time::Duration::from_millis(200));
                        if worker.shutdown.load(std::sync::atomic::Ordering::Acquire) {
                            return;
                        }
                        if worker
                            .dirty
                            .swap(false, std::sync::atomic::Ordering::AcqRel)
                        {
                            if let Some(srv) = worker.server.upgrade() {
                                let _ = writeFeatureSnapshotSync(&worker.path, &srv);
                            }
                        }
                    }
                })
                .expect("failed to spawn snapshot flusher thread");
            if let Ok(mut guard) = f.handle.lock() {
                *guard = Some(handle);
            }
            f
        })
        .clone()
}

/// Public accessor so server shutdown can join the worker cleanly
pub fn snapshotFlusher() -> Option<Arc<SnapshotFlusher>> {
    SNAPSHOT_FLUSHER.get().cloned()
}

fn writeFeatureSnapshotSync(
    path: &std::path::Path,
    server: &Arc<ServerState>,
) -> std::result::Result<(), ZyronError> {
    let groups = server.feature_store.groups();
    let groupsVec: Vec<zyron_analytics::FeatureGroup> =
        groups.iter().map(|g| (**g).clone()).collect();
    let text = serde_json::to_string_pretty(&groupsVec)
        .map_err(|e| ZyronError::ExecutionError(format!("feature store snapshot encode: {}", e)))?;
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    std::fs::write(path, text)
        .map_err(|e| ZyronError::ExecutionError(format!("feature store snapshot write: {}", e)))?;
    Ok(())
}

fn snapshotFeatureStore(server: &Arc<ServerState>) -> Result<(), ZyronError> {
    let flusher = ensureSnapshotFlusher(server);
    flusher
        .dirty
        .store(true, std::sync::atomic::Ordering::Release);
    if let Some(t) = flusher.waker.get() {
        t.unpark();
    }
    Ok(())
}

fn persistModel(
    server: &Arc<ServerState>,
    name: &str,
    model: &zyron_analytics::TrainedModel,
) -> Result<(), ZyronError> {
    let dir = modelsDir(server);
    std::fs::create_dir_all(&dir)
        .map_err(|e| ZyronError::ExecutionError(format!("models dir: {}", e)))?;
    let path = dir.join(format!("{}.json", name));
    let text = serde_json::to_string(model)
        .map_err(|e| ZyronError::ExecutionError(format!("model serialize: {}", e)))?;
    std::fs::write(&path, text)
        .map_err(|e| ZyronError::ExecutionError(format!("model write: {}", e)))?;
    Ok(())
}

fn removeModelFile(server: &Arc<ServerState>, name: &str) -> Result<(), ZyronError> {
    let path = modelsDir(server).join(format!("{}.json", name));
    if path.exists() {
        std::fs::remove_file(&path)
            .map_err(|e| ZyronError::ExecutionError(format!("model remove: {}", e)))?;
    }
    Ok(())
}

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
pub async fn try_handle_ddl_utility(
    stmt: &zyron_parser::Statement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
    txn: &mut Option<zyron_storage::txn::Transaction>,
    active_branch: &mut Option<String>,
    raw_sql: &str,
) -> Option<Result<DdlResult, ProtocolError>> {
    use zyron_parser::Statement;

    match stmt {
        // DML statements fall through to planner
        Statement::Select(_)
        | Statement::Insert(_)
        | Statement::Update(_)
        | Statement::Delete(_) => None,

        // MERGE requires INSERT/UPDATE/DELETE combined execution, routed through planner
        Statement::Merge(_) => Some(Err(ProtocolError::Database(
            ZyronError::PlanError("MERGE statement execution requires planner support for combined INSERT/UPDATE/DELETE operations".to_string()),
        ))),

        // Standalone VALUES query handled as DDL result with rows
        Statement::ValuesQuery(s) => Some(handle_values_query(s)),

        // Transaction control handled by try_handle_transaction_control
        Statement::Begin(_) | Statement::Commit(_) | Statement::Rollback(_) => None,

        // Session commands handled by try_handle_session_command
        Statement::SetVariable(_)
        | Statement::Show(_)
        | Statement::AlterSystemSet(_)
        | Statement::Checkpoint(_)
        | Statement::Vacuum(_)
        | Statement::Analyze(_) => None,

        // EXPLAIN handled by handle_explain_statement
        Statement::Explain(_) => None,

        // -- Core DDL --
        Statement::CreateTable(s) => Some(handle_create_table(s, server, session).await),
        Statement::DropTable(s) => Some(handle_drop_table(s, server, session).await),
        Statement::AlterTable(_) => Some(Ok(DdlResult::Tag("ALTER TABLE".to_string()))),
        Statement::Truncate(s) => Some(handle_truncate(s, server, session).await),
        Statement::CreateIndex(s) => Some(handle_create_index(s, server, session).await),
        Statement::DropIndex(s) => Some(handle_drop_index(s, server, session).await),
        Statement::AlterIndex(_) => Some(Ok(DdlResult::Tag("ALTER INDEX".to_string()))),
        Statement::CreateSchema(s) => Some(handle_create_schema(s, server, session).await),
        Statement::DropSchema(s) => Some(handle_drop_schema(s, server, session).await),
        Statement::CreateSequence(_) => Some(Ok(DdlResult::Tag("CREATE SEQUENCE".to_string()))),
        Statement::DropSequence(_) => Some(Ok(DdlResult::Tag("DROP SEQUENCE".to_string()))),
        Statement::AlterSequence(_) => Some(Ok(DdlResult::Tag("ALTER SEQUENCE".to_string()))),
        Statement::CreateView(_) => Some(Ok(DdlResult::Tag("CREATE VIEW".to_string()))),
        Statement::DropView(_) => Some(Ok(DdlResult::Tag("DROP VIEW".to_string()))),
        Statement::AlterView(_) => Some(Ok(DdlResult::Tag("ALTER VIEW".to_string()))),
        Statement::AlterTableTtl(_) => Some(Ok(DdlResult::Tag("ALTER TABLE".to_string()))),
        Statement::AlterTableOptions(_) => Some(Ok(DdlResult::Tag("ALTER TABLE".to_string()))),
        Statement::OptimizeTable(_) => Some(Ok(DdlResult::Tag("OPTIMIZE".to_string()))),
        Statement::Reindex(_) => Some(Ok(DdlResult::Tag("REINDEX".to_string()))),
        Statement::CommentOn(_) => Some(Ok(DdlResult::Tag("COMMENT".to_string()))),

        // -- Materialized Views --
        Statement::CreateMaterializedView(_) => {
            Some(Ok(DdlResult::Tag("CREATE MATERIALIZED VIEW".to_string())))
        }
        Statement::DropMaterializedView(_) => {
            Some(Ok(DdlResult::Tag("DROP MATERIALIZED VIEW".to_string())))
        }
        Statement::RefreshMaterializedView(_) => {
            Some(Ok(DdlResult::Tag("REFRESH MATERIALIZED VIEW".to_string())))
        }

        // -- Search Indexes --
        Statement::CreateFulltextIndex(s) => {
            Some(handle_create_fulltext_index(s, server, session).await)
        }
        Statement::CreateVectorIndex(s) => {
            Some(handle_create_vector_index(s, server, session).await)
        }
        Statement::CreateSpatialIndex(s) => {
            Some(handle_create_spatial_index(s, server, session).await)
        }

        // -- Auth/Roles --
        Statement::CreateUser(s) => Some(handle_create_user(s, server).await),
        Statement::AlterUser(_) => Some(Ok(DdlResult::Tag("ALTER USER".to_string()))),
        Statement::DropUser(s) => Some(handle_drop_user(s, server).await),
        Statement::CreateRole(s) => Some(handle_create_role(s, server).await),
        Statement::AlterRole(_) => Some(Ok(DdlResult::Tag("ALTER ROLE".to_string()))),
        Statement::DropRole(s) => Some(handle_drop_role(s, server).await),
        Statement::Grant(s) => Some(handle_grant(s, server, session).await),
        Statement::Revoke(s) => Some(handle_revoke(s, server, session).await),

        // -- CDC --
        Statement::CreateReplicationSlot(_) => {
            Some(Ok(DdlResult::Tag("CREATE_REPLICATION_SLOT".to_string())))
        }
        Statement::DropReplicationSlot(_) => {
            Some(Ok(DdlResult::Tag("DROP_REPLICATION_SLOT".to_string())))
        }
        Statement::CreatePublication(_) => {
            Some(Ok(DdlResult::Tag("CREATE PUBLICATION".to_string())))
        }
        Statement::AlterPublication(_) => {
            Some(Ok(DdlResult::Tag("ALTER PUBLICATION".to_string())))
        }
        Statement::DropPublication(_) => {
            Some(Ok(DdlResult::Tag("DROP PUBLICATION".to_string())))
        }
        Statement::CreateCdcStream(_) => {
            Some(Ok(DdlResult::Tag("CREATE CDC STREAM".to_string())))
        }
        Statement::DropCdcStream(_) => {
            Some(Ok(DdlResult::Tag("DROP CDC STREAM".to_string())))
        }
        Statement::CreateCdcIngest(_) => {
            Some(Ok(DdlResult::Tag("CREATE CDC INGEST".to_string())))
        }
        Statement::DropCdcIngest(_) => {
            Some(Ok(DdlResult::Tag("DROP CDC INGEST".to_string())))
        }

        // -- Streaming jobs --
        Statement::CreateStreamingJob(_)
        | Statement::DropStreamingJob(_)
        | Statement::AlterStreamingJob(_) => {
            Some(dispatch_streaming_statement(stmt.clone(), server, session, raw_sql).await)
        }

        // -- Versioning --
        Statement::CreateBranch(_) => Some(Ok(DdlResult::Tag("CREATE BRANCH".to_string()))),
        Statement::MergeBranch(_) => Some(Ok(DdlResult::Tag("MERGE BRANCH".to_string()))),
        Statement::DropBranch(_) => Some(Ok(DdlResult::Tag("DROP BRANCH".to_string()))),
        Statement::UseBranch(s) => {
            *active_branch = Some(s.name.clone());
            Some(Ok(DdlResult::Tag("USE BRANCH".to_string())))
        }
        Statement::CreateVersion(_) => Some(Ok(DdlResult::Tag("CREATE VERSION".to_string()))),

        // -- Pipeline --
        Statement::CreatePipeline(_) => {
            Some(Ok(DdlResult::Tag("CREATE PIPELINE".to_string())))
        }
        Statement::RunPipeline(_) => Some(Ok(DdlResult::Tag("RUN PIPELINE".to_string()))),
        Statement::DropPipeline(_) => Some(Ok(DdlResult::Tag("DROP PIPELINE".to_string()))),

        // -- Scheduling --
        Statement::CreateSchedule(_) => {
            Some(Ok(DdlResult::Tag("CREATE SCHEDULE".to_string())))
        }
        Statement::DropSchedule(_) => Some(Ok(DdlResult::Tag("DROP SCHEDULE".to_string()))),
        Statement::PauseSchedule(_) => Some(Ok(DdlResult::Tag("PAUSE SCHEDULE".to_string()))),
        Statement::ResumeSchedule(_) => {
            Some(Ok(DdlResult::Tag("RESUME SCHEDULE".to_string())))
        }

        // -- Functions/Aggregates --
        Statement::CreateFunction(_) => {
            Some(Ok(DdlResult::Tag("CREATE FUNCTION".to_string())))
        }
        Statement::DropFunction(_) => Some(Ok(DdlResult::Tag("DROP FUNCTION".to_string()))),
        Statement::CreateAggregate(_) => {
            Some(Ok(DdlResult::Tag("CREATE AGGREGATE".to_string())))
        }
        Statement::DropAggregate(_) => {
            Some(Ok(DdlResult::Tag("DROP AGGREGATE".to_string())))
        }

        // -- Procedures --
        Statement::CreateProcedure(_) => {
            Some(Ok(DdlResult::Tag("CREATE PROCEDURE".to_string())))
        }
        Statement::DropProcedure(_) => Some(Ok(DdlResult::Tag("DROP PROCEDURE".to_string()))),
        Statement::Call(_) => Some(Ok(DdlResult::Tag("CALL".to_string()))),

        // -- Triggers --
        Statement::CreateTrigger(_) => {
            Some(Ok(DdlResult::Tag("CREATE TRIGGER".to_string())))
        }
        Statement::DropTrigger(_) => Some(Ok(DdlResult::Tag("DROP TRIGGER".to_string()))),

        // -- Event Handlers --
        Statement::CreateEventHandler(_) => {
            Some(Ok(DdlResult::Tag("CREATE EVENT HANDLER".to_string())))
        }
        Statement::DropEventHandler(_) => {
            Some(Ok(DdlResult::Tag("DROP EVENT HANDLER".to_string())))
        }

        // -- Expectations/Features --
        Statement::AddExpectation(_) => Some(Ok(DdlResult::Tag("ALTER TABLE".to_string()))),
        Statement::DropExpectation(_) => Some(Ok(DdlResult::Tag("ALTER TABLE".to_string()))),
        Statement::EnableFeature(_) => Some(Ok(DdlResult::Tag("ALTER TABLE".to_string()))),
        Statement::DisableFeature(_) => Some(Ok(DdlResult::Tag("ALTER TABLE".to_string()))),

        // -- Transaction extensions --
        Statement::Savepoint(s) => Some(handle_savepoint(s, txn)),
        Statement::ReleaseSavepoint(s) => Some(handle_release_savepoint(s, txn)),

        // -- Prepared statements: handled by caller (needs statements map) --
        Statement::Prepare(_) | Statement::Execute(_) | Statement::Deallocate(_) => None,

        // -- Cursors: handled by caller (needs cursors map) --
        Statement::DeclareCursor(_) | Statement::FetchCursor(_) | Statement::CloseCursor(_) => {
            None
        }

        // -- Pub/Sub: handled by caller (needs notification_receivers) --
        Statement::Listen(_) | Statement::Notify(_) => None,

        // -- COPY: handled by caller (needs wire protocol interaction) --
        Statement::Copy(_) => None,

        // -- Archive --
        Statement::ArchiveTable(_) => {
            Some(Ok(DdlResult::Tag("ARCHIVE TABLE".to_string())))
        }
        Statement::RestoreTable(_) => {
            Some(Ok(DdlResult::Tag("RESTORE TABLE".to_string())))
        }

        // -- Utility --
        Statement::DoBlock(_) => Some(Ok(DdlResult::Tag("DO".to_string()))),

        // -- Graph schema --
        Statement::CreateGraphSchema(stmt) => {
            Some(handle_create_graph_schema(stmt, server))
        }
        Statement::DropGraphSchema(stmt) => {
            Some(handle_drop_graph_schema(stmt, server))
        }

        // -- External sources and sinks --
        Statement::CreateExternalSource(_)
        | Statement::CreateExternalSink(_)
        | Statement::DropExternalSource(_)
        | Statement::DropExternalSink(_)
        | Statement::AlterExternalSource(_)
        | Statement::AlterExternalSink(_) => {
            Some(dispatch_external_statement(stmt.clone(), server, session).await)
        }
    }
}

// ---------------------------------------------------------------------------
// DDL privilege checking
// ---------------------------------------------------------------------------

/// Checks whether the session has the required privilege for a DDL operation.
/// If no security manager is configured, the check is skipped (open access).
/// The object_id is the catalog ID of the target object (schema ID, table ID, etc.).
fn check_ddl_privilege(
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

    match server
        .catalog
        .create_table(schema_id, &stmt.name, &stmt.columns, &stmt.constraints)
        .await
    {
        Ok(_) => Ok(DdlResult::Tag("CREATE TABLE".to_string())),
        Err(ZyronError::TableAlreadyExists(_)) if stmt.if_not_exists => {
            Ok(DdlResult::Tag("CREATE TABLE".to_string()))
        }
        Err(e) => Err(ProtocolError::Database(e)),
    }
}

async fn handle_drop_table(
    stmt: &zyron_parser::ast::DropTableStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (_, schema_id) = get_session_schema(session, server, None)?;

    // Check DROP privilege on the table if it exists. If the table does not
    // exist and IF EXISTS is set, skip the privilege check entirely.
    if let Ok(table) = server.catalog.get_table(schema_id, &stmt.name) {
        check_ddl_privilege(
            server,
            session,
            zyron_auth::PrivilegeType::Create,
            zyron_auth::ObjectType::Table,
            table.id.0,
        )?;
    }

    match server.catalog.drop_table(schema_id, &stmt.name).await {
        Ok(()) => Ok(DdlResult::Tag("DROP TABLE".to_string())),
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

    let mut column_names: Vec<String> = Vec::with_capacity(stmt.columns.len());
    for c in &stmt.columns {
        match &c.expr {
            zyron_parser::ast::Expr::Identifier(name) => {
                column_names.push(name.clone());
            }
            other => {
                return Err(ProtocolError::Database(ZyronError::PlanError(format!(
                    "expression indexes are not supported, use column names (got: {:?})",
                    other
                ))));
            }
        }
    }

    match server
        .catalog
        .create_index(
            table.id,
            schema_id,
            &stmt.name,
            &column_names,
            stmt.unique,
            zyron_catalog::IndexType::BTree,
        )
        .await
    {
        Ok(_) => Ok(DdlResult::Tag("CREATE INDEX".to_string())),
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

            match server.catalog.drop_index(table_id, &stmt.name).await {
                Ok(()) => {
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

async fn handle_create_schema(
    stmt: &zyron_parser::ast::CreateSchemaStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (db_id, _) = get_session_schema(session, server, None)?;

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
        Ok(_) => Ok(DdlResult::Tag("CREATE SCHEMA".to_string())),
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
    let (db_id, _) = get_session_schema(session, server, None)?;

    // Check CREATE privilege on the database (schema owners can drop their schemas)
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Database,
        db_id.0,
    )?;

    match server.catalog.drop_schema(db_id, &stmt.name).await {
        Ok(()) => Ok(DdlResult::Tag("DROP SCHEMA".to_string())),
        Err(ZyronError::SchemaNotFound(_)) if stmt.if_exists => {
            Ok(DdlResult::Tag("DROP SCHEMA".to_string()))
        }
        Err(e) => Err(ProtocolError::Database(e)),
    }
}

// ---------------------------------------------------------------------------
// Auth/Role handlers
// ---------------------------------------------------------------------------

async fn handle_create_user(
    stmt: &zyron_parser::ast::CreateUserStatement,
    server: &Arc<ServerState>,
) -> Result<DdlResult, ProtocolError> {
    let sm = require_security_manager(server)?;

    let role = zyron_auth::Role {
        id: zyron_auth::RoleId(0), // assigned by create_role
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
    Ok(DdlResult::Tag("CREATE USER".to_string()))
}

async fn handle_drop_user(
    stmt: &zyron_parser::ast::DropUserStatement,
    server: &Arc<ServerState>,
) -> Result<DdlResult, ProtocolError> {
    let sm = require_security_manager(server)?;

    match sm.lookup_role(&stmt.name) {
        Some(r) => {
            sm.drop_role(r.id).await.map_err(ProtocolError::Database)?;
            Ok(DdlResult::Tag("DROP USER".to_string()))
        }
        None if stmt.if_exists => Ok(DdlResult::Tag("DROP USER".to_string())),
        None => Err(ProtocolError::Database(ZyronError::RoleNotFound(
            stmt.name.clone(),
        ))),
    }
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
                with_grant_option: false,
                granted_by: zyron_auth::RoleId(0),
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

    Ok(DdlResult::Tag("GRANT".to_string()))
}

async fn handle_revoke(
    stmt: &zyron_parser::ast::RevokeStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let sm = require_security_manager(server)?;

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

    Ok(DdlResult::Tag("REVOKE".to_string()))
}

// ---------------------------------------------------------------------------
// Transaction extension handlers
// ---------------------------------------------------------------------------

fn handle_savepoint(
    stmt: &zyron_parser::ast::SavepointStatement,
    txn: &mut Option<zyron_storage::txn::Transaction>,
) -> Result<DdlResult, ProtocolError> {
    let txn = txn.as_mut().ok_or_else(|| {
        ProtocolError::Database(ZyronError::TransactionAborted(
            "SAVEPOINT can only be used in a transaction".to_string(),
        ))
    })?;
    txn.savepoint(stmt.name.clone(), 0);
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

fn handle_values_query(
    stmt: &zyron_parser::ast::ValuesQueryStatement,
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

    let columns: Vec<(String, i32)> = (0..num_cols)
        .map(|i| (format!("column{}", i + 1), crate::types::PG_TEXT_OID))
        .collect();

    let mut rows = Vec::with_capacity(stmt.rows.len());
    for row in &stmt.rows {
        let mut row_values = Vec::with_capacity(row.len());
        for expr in row {
            let value = eval_literal_expr(expr);
            row_values.push(value);
        }
        rows.push(row_values);
    }

    Ok(DdlResult::Rows {
        tag: format!("SELECT {}", rows.len()),
        columns,
        rows,
    })
}

/// Evaluates a simple literal expression to its string representation.
/// Handles integer, float, string, boolean, and null literals.
/// Complex expressions evaluate to their debug representation.
fn eval_literal_expr(expr: &zyron_parser::ast::Expr) -> String {
    use zyron_parser::ast::{Expr, LiteralValue};
    match expr {
        Expr::Literal(LiteralValue::Integer(n)) => n.to_string(),
        Expr::Literal(LiteralValue::Float(f)) => f.to_string(),
        Expr::Literal(LiteralValue::String(s)) => s.clone(),
        Expr::Literal(LiteralValue::Boolean(b)) => b.to_string(),
        Expr::Literal(LiteralValue::Null) => "".to_string(),
        Expr::UnaryOp { op, expr } => {
            let inner = eval_literal_expr(expr);
            format!("{:?}{}", op, inner)
        }
        other => format!("{:?}", other),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Gets the database ID and default schema ID from the active session.
fn get_session_schema(
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
        .unwrap_or("public");

    let schema = server
        .catalog
        .get_schema(db_id, schema_name)
        .map_err(ProtocolError::Database)?;

    Ok((db_id, schema.id))
}

async fn handle_create_fulltext_index(
    stmt: &zyron_parser::ast::CreateFulltextIndexStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    // Resolve table from the default schema
    let schema_id = zyron_catalog::SchemaId(1); // default public schema

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
    let schema_id = zyron_catalog::SchemaId(1); // default public schema

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

    // Parse distance metric and index parameters from options
    let mut distance_metric = "cosine".to_string();
    let mut m: u16 = 16;
    let mut ef_construction: u16 = 200;
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
                m = val_str.parse().unwrap_or(16);
            }
            "ef_construction" => {
                ef_construction = val_str.parse().unwrap_or(200);
            }
            _ => {}
        }
    }

    // Find the column to determine dimensions
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

    let dimensions = col.max_length.unwrap_or(128) as u16;

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
    let schema_id = zyron_catalog::SchemaId(1); // default public schema

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

fn handle_create_graph_schema(
    stmt: &zyron_parser::ast::CreateGraphSchemaStatement,
    server: &Arc<ServerState>,
) -> Result<DdlResult, ProtocolError> {
    let graph_mgr = server.graph_manager.as_ref().ok_or_else(|| {
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

    let schema_oid = server.catalog.next_oid();
    let mut schema = zyron_search::graph::GraphSchema::new(stmt.name.clone(), schema_oid);

    // First pass: register all node labels so edge labels can reference them.
    for elem in &stmt.elements {
        if let zyron_parser::ast::GraphSchemaElement::Node { label, properties } = elem {
            let props: Vec<zyron_search::graph::PropertyDef> = properties
                .iter()
                .map(|col| zyron_search::graph::PropertyDef {
                    name: col.name.clone(),
                    type_id: col.data_type.to_type_id(),
                    nullable: col.nullable.unwrap_or(true),
                })
                .collect();
            schema.add_node_label(label.clone(), props, 0);
        }
    }

    // Second pass: register edge labels with resolved node label IDs.
    for elem in &stmt.elements {
        if let zyron_parser::ast::GraphSchemaElement::Edge {
            label,
            from_label,
            to_label,
            properties,
        } = elem
        {
            let from_id = schema
                .get_node_label(from_label)
                .map(|nl| nl.label_id)
                .ok_or_else(|| {
                    ProtocolError::Database(ZyronError::GraphQueryError(format!(
                        "source node label '{}' not found in schema",
                        from_label
                    )))
                })?;
            let to_id = schema
                .get_node_label(to_label)
                .map(|nl| nl.label_id)
                .ok_or_else(|| {
                    ProtocolError::Database(ZyronError::GraphQueryError(format!(
                        "target node label '{}' not found in schema",
                        to_label
                    )))
                })?;
            let props: Vec<zyron_search::graph::PropertyDef> = properties
                .iter()
                .map(|col| zyron_search::graph::PropertyDef {
                    name: col.name.clone(),
                    type_id: col.data_type.to_type_id(),
                    nullable: col.nullable.unwrap_or(true),
                })
                .collect();
            schema
                .add_edge_label(label.clone(), from_id, to_id, props, 0, true)
                .map_err(ProtocolError::Database)?;
        }
    }

    graph_mgr
        .create_schema(schema)
        .map_err(ProtocolError::Database)?;

    // Persist to disk immediately so the schema survives restarts. If the
    // write fails, roll back the in-memory create so the catalog state
    // matches what's on disk.
    let graph_dir = server.data_dir.join("graph");
    if let Err(e) = graph_mgr.save_all(&graph_dir) {
        let _ = graph_mgr.drop_schema(&stmt.name);
        return Err(ProtocolError::Database(e));
    }

    Ok(DdlResult::Tag("CREATE GRAPH SCHEMA".to_string()))
}

fn handle_drop_graph_schema(
    stmt: &zyron_parser::ast::DropGraphSchemaStatement,
    server: &Arc<ServerState>,
) -> Result<DdlResult, ProtocolError> {
    let graph_mgr = server.graph_manager.as_ref().ok_or_else(|| {
        ProtocolError::Database(ZyronError::GraphSchemaNotFound(
            "graph manager not configured".to_string(),
        ))
    })?;

    match graph_mgr.drop_schema(&stmt.name) {
        Ok(()) => {
            let graph_dir = server.data_dir.join("graph");
            graph_mgr
                .save_all(&graph_dir)
                .map_err(ProtocolError::Database)?;
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
    let (db_id, _) = get_session_schema(session, server, None)?;
    let search_path = session
        .as_ref()
        .map(|s| s.search_path.clone())
        .unwrap_or_else(|| vec!["public".to_string()]);

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
                    security_manager,
                )
                .map_err(ProtocolError::Database)?;
            let _ = src_table_id;
            let _ = src_columns;
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
        .map(|c| zyron_streaming::format::ColumnSpec {
            name: c.name.clone(),
            type_id: c.type_id,
        })
        .collect()
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
    let (db_id, _) = get_session_schema(session, server, None)?;
    let search_path = session
        .as_ref()
        .map(|s| s.search_path.clone())
        .unwrap_or_else(|| vec!["public".to_string()]);

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
        let resolver = server.catalog.resolver(
            zyron_catalog::SYSTEM_DATABASE_ID,
            vec!["public".to_string()],
        );
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
        AlterExternalSourceAction::SetMode(_) => "SetMode",
        AlterExternalSourceAction::SetColumns(_) => "SetColumns",
        AlterExternalSourceAction::Rename(_) => "Rename",
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

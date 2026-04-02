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
        Statement::CreateVectorIndex(_) => {
            Some(Ok(DdlResult::Tag("CREATE INDEX".to_string())))
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
            // Check CREATE privilege on the schema (index owner must have schema-level rights)
            check_ddl_privilege(
                server,
                session,
                zyron_auth::PrivilegeType::Create,
                zyron_auth::ObjectType::Schema,
                schema_id.0,
            )?;

            // Check if this is an FTS index so we can clean up the FTS manager.
            let fts_index_id = server
                .catalog
                .get_indexes_for_table(table_id)
                .iter()
                .find(|idx| {
                    idx.name == stmt.name && idx.index_type == zyron_catalog::IndexType::Fulltext
                })
                .map(|idx| idx.id.0);

            match server.catalog.drop_index(table_id, &stmt.name).await {
                Ok(()) => {
                    // Remove from FTS manager if it was a fulltext index.
                    if let (Some(id), Some(fts_mgr)) = (fts_index_id, &server.fts_manager) {
                        let _ = fts_mgr.drop_index(id);
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
/// ALL expands to Select, Insert, Update, Delete.
fn map_privilege(p: zyron_parser::ast::Privilege) -> Vec<zyron_auth::PrivilegeType> {
    match p {
        zyron_parser::ast::Privilege::Select => vec![zyron_auth::PrivilegeType::Select],
        zyron_parser::ast::Privilege::Insert => vec![zyron_auth::PrivilegeType::Insert],
        zyron_parser::ast::Privilege::Update => vec![zyron_auth::PrivilegeType::Update],
        zyron_parser::ast::Privilege::Delete => vec![zyron_auth::PrivilegeType::Delete],
        zyron_parser::ast::Privilege::All => vec![
            zyron_auth::PrivilegeType::Select,
            zyron_auth::PrivilegeType::Insert,
            zyron_auth::PrivilegeType::Update,
            zyron_auth::PrivilegeType::Delete,
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

    // Privilege check: require CREATE on schema (same as regular index creation)
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Schema,
        schema_id.0,
    )?;

    let table = server
        .catalog
        .get_table(schema_id, &stmt.table)
        .map_err(ProtocolError::Database)?;

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

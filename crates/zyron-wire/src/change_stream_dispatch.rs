//! Change stream DDL, the APPLY CHANGES statement, and the run log behind
//! `zyron_sys.cdc.apply_runs`.
//!
//! Every handler here resolves its name through `resolve_qualified_name`, so a
//! schema-qualified stream is looked up in the schema it names rather than
//! under a literal dotted name, checks the privilege the action needs, and
//! then writes through the catalog so the change reaches the rest of a
//! consensus group as an ordinary catalog statement

use std::sync::Arc;

use parking_lot::Mutex;
use zyron_catalog::{
    ChangeStreamEntry, ChangeStreamMode, ChangeStreamOrigin, ChangeStreamSource, ColumnId,
    SchemaId, TableEntry,
};
use zyron_cdc::change_stream::view::{ViewShape, plan_view_stream};
use zyron_cdc::change_stream::{self, ChangeStreamRuntime, ResetTarget};
use zyron_common::ZyronError;
use zyron_parser::ast::{
    AlterChangeStreamAction, AlterChangeStreamStatement, ChangeStreamStart, ChangeStreamTarget,
    CreateChangeStreamStatement, DropChangeStreamStatement, Expr, LiteralValue,
    ShowChangeStreamsStatement,
};

use crate::connection::ServerState;
use crate::ddl_dispatch::DdlResult;
use crate::messages::ProtocolError;
use crate::session::Session;

// ---------------------------------------------------------------------------
// ApplyRunLog
// ---------------------------------------------------------------------------

/// One APPLY CHANGES run, as `zyron_sys.cdc.apply_runs` reports it
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApplyRun {
    pub target: String,
    pub source: String,
    pub rows_upserted: u64,
    pub rows_deleted: u64,
    pub rows_versioned: u64,
    pub truncated: bool,
    pub duration_micros: u64,
    pub started_at: i64,
    /// Empty when the run reached its target
    pub error: String,
}

/// Runs of APPLY CHANGES, newest last, bounded so a busy workflow does not
/// grow the log without limit.
///
/// Node-local operator telemetry rather than catalog state, so a member reports
/// the runs it carried out, and a replicated apply carries its rows rather
/// than a copy of the leader's counters
pub struct ApplyRunLog {
    runs: Mutex<Vec<ApplyRun>>,
    capacity: usize,
}

impl Default for ApplyRunLog {
    fn default() -> Self {
        Self::new(APPLY_RUN_LOG_CAPACITY)
    }
}

/// Runs the log holds before the oldest is dropped
const APPLY_RUN_LOG_CAPACITY: usize = 256;

impl ApplyRunLog {
    pub fn new(capacity: usize) -> Self {
        Self {
            runs: Mutex::new(Vec::new()),
            capacity: capacity.max(1),
        }
    }

    pub fn record(&self, run: ApplyRun) {
        let mut runs = self.runs.lock();
        if runs.len() == self.capacity {
            runs.remove(0);
        }
        runs.push(run);
    }

    pub fn list(&self) -> Vec<ApplyRun> {
        self.runs.lock().clone()
    }

    /// Runs that ended with an error, which is what `cdc_apply_failed` fires
    /// on
    pub fn failures(&self) -> Vec<ApplyRun> {
        self.runs
            .lock()
            .iter()
            .filter(|run| !run.error.is_empty())
            .cloned()
            .collect()
    }
}

/// The node's apply run log.
///
/// One per process, the same shape the lake's compaction history takes. The
/// runs belong to the node that carried them out rather than to any one
/// connection, and nothing replicates them
pub fn apply_runs() -> &'static ApplyRunLog {
    static LOG: std::sync::OnceLock<ApplyRunLog> = std::sync::OnceLock::new();
    LOG.get_or_init(ApplyRunLog::default)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// The change stream runtime for this node's feeds.
///
/// Built from the feed registry rather than held beside it, because the
/// runtime keeps no state of its own. A stream's definition lives in the
/// catalog and its changes live in the feeds, so a second durable copy of
/// either is a copy that can disagree
pub fn runtime_of(server: &Arc<ServerState>) -> Result<Arc<ChangeStreamRuntime>, ProtocolError> {
    let feeds = server.cdc_registry.as_ref().cloned().ok_or_else(|| {
        ProtocolError::Database(ZyronError::CdcStreamError(
            "change streams require CDC to be enabled on this server".into(),
        ))
    })?;
    Ok(Arc::new(ChangeStreamRuntime::new(feeds)))
}

fn now_micros() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as i64
}

/// The branch a session is on, by id, None on the table itself
pub fn active_branch_id(server: &Arc<ServerState>, active_branch: &Option<String>) -> Option<u64> {
    let name = active_branch.as_deref()?;
    server
        .branch_manager
        .as_ref()?
        .get_branch_by_name(name)
        .ok()
        .map(|entry| entry.id.0)
}

/// Reads an integer out of a literal the statement carried
fn integer_of(expr: &Expr, what: &str) -> Result<i64, ProtocolError> {
    match expr {
        Expr::Literal(LiteralValue::Integer(n)) => Ok(*n),
        other => Err(ProtocolError::Database(ZyronError::ParseError(format!(
            "{what} takes a whole number, found {other:?}"
        )))),
    }
}

/// Reads a timestamp literal, in microseconds, out of the statement
fn timestamp_of(expr: &Expr, what: &str) -> Result<i64, ProtocolError> {
    match expr {
        Expr::Literal(LiteralValue::Integer(n)) => Ok(*n),
        Expr::Literal(LiteralValue::String(text)) => {
            zyron_common::interval::parse_timestamp_micros(text).map_err(|e| {
                ProtocolError::Database(ZyronError::ParseError(format!(
                    "{what} did not read as a timestamp, {e}"
                )))
            })
        }
        other => Err(ProtocolError::Database(ZyronError::ParseError(format!(
            "{what} takes a timestamp, found {other:?}"
        )))),
    }
}

/// Resolves the column names a stream names to their catalog ids
fn column_ids(
    table: &TableEntry,
    names: &[String],
) -> Result<Option<Vec<ColumnId>>, ProtocolError> {
    if names.is_empty() {
        return Ok(None);
    }
    let mut ids = Vec::with_capacity(names.len());
    for name in names {
        let column = table
            .live_columns()
            .find(|c| c.name.eq_ignore_ascii_case(name))
            .ok_or_else(|| {
                ProtocolError::Database(ZyronError::PlanError(format!(
                    "column '{name}' is not a column of table '{}'",
                    table.name
                )))
            })?;
        ids.push(column.id);
    }
    Ok(Some(ids))
}

/// Resolves the column names a stream over several tables was created
/// with against the columns the stream yields, None when it names none
fn union_column_ids(
    union: &[zyron_planner::logical::LogicalColumn],
    names: &[String],
) -> Result<Option<Vec<ColumnId>>, ProtocolError> {
    if names.is_empty() {
        return Ok(None);
    }
    let mut ids = Vec::with_capacity(names.len());
    for name in names {
        let column = union
            .iter()
            .find(|c| c.name.eq_ignore_ascii_case(name))
            .ok_or_else(|| {
                ProtocolError::Database(ZyronError::PlanError(format!(
                    "column '{name}' is not a column of any table the stream reads"
                )))
            })?;
        ids.push(column.column_id);
    }
    Ok(Some(ids))
}

/// Reads what a view's stored query holds, as far as a change stream cares.
///
/// The definition is parsed rather than re-planned, because what decides the
/// answer is which constructs the query is written with
fn view_shape(
    server: &Arc<ServerState>,
    session: &Option<Session>,
    definition_sql: &str,
) -> Result<ViewShape, ProtocolError> {
    let parsed = zyron_parser::parse(definition_sql).map_err(|e| {
        ProtocolError::Database(ZyronError::PlanError(format!(
            "the view definition did not parse, {e}"
        )))
    })?;
    let Some(zyron_parser::Statement::CreateView(create)) = parsed.into_iter().next() else {
        return Err(ProtocolError::Database(ZyronError::PlanError(
            "the view definition is not a CREATE VIEW".into(),
        )));
    };
    let query = &create.query;

    let mut shape = ViewShape {
        has_set_operation: !query.set_ops.is_empty(),
        has_aggregate: !query.group_by.is_empty() || query.having.is_some(),
        has_distinct: query.distinct || !query.distinct_on.is_empty(),
        has_limit: query.limit.is_some() || query.offset.is_some(),
        ..ViewShape::default()
    };
    if query.from.len() > 1 {
        shape.has_join = true;
    }
    for item in &query.from {
        match item {
            zyron_parser::TableRef::Table { name, .. } => {
                let (schema_id, table_name) =
                    crate::ddl_dispatch::resolve_qualified_name(name, server, session)?;
                match server.catalog.get_table(schema_id, &table_name) {
                    Ok(table) => shape.base_tables.push(table.id.0),
                    Err(e) => return Err(ProtocolError::Database(e)),
                }
            }
            zyron_parser::TableRef::Join(_) => shape.has_join = true,
            zyron_parser::TableRef::Subquery { .. } | zyron_parser::TableRef::Lateral { .. } => {
                shape.has_subquery = true
            }
            _ => shape.has_subquery = true,
        }
    }
    // A projection that is anything but a bare column list carries an
    // expression a change set cannot recompute per row without the rest of
    // the table, which reads as a subquery for this purpose
    let mut projection = Vec::new();
    for item in &query.projections {
        match item {
            zyron_parser::ast::SelectItem::Wildcard => {}
            zyron_parser::ast::SelectItem::Expr(Expr::Identifier(name), _) => {
                projection.push(name.clone());
            }
            zyron_parser::ast::SelectItem::Expr(Expr::QualifiedIdentifier { column, .. }, _) => {
                projection.push(column.clone());
            }
            _ => shape.has_subquery = true,
        }
    }
    if let Some(base) = shape.base_tables.first().copied() {
        if let Some(table) = server
            .catalog
            .get_table_by_id(zyron_catalog::TableId(base))
            .ok()
        {
            let mut ids = Vec::new();
            for name in &projection {
                if let Some(column) = table
                    .live_columns()
                    .find(|c| c.name.eq_ignore_ascii_case(name))
                {
                    ids.push(column.id.0);
                }
            }
            shape.projection = ids;
        }
    }
    shape.predicate = query
        .where_clause
        .as_ref()
        .map(|expr| zyron_parser::expr_to_sql(expr));
    Ok(shape)
}

// ---------------------------------------------------------------------------
// Position advances
// ---------------------------------------------------------------------------

/// Writes the position advances a transaction's statements recorded into that
/// transaction's own log chain.
///
/// The record goes under the consumer's transaction id and chains from its
/// last write, so the commit record that makes the consumer's rows durable is
/// the same one that makes the advance durable. A rollback leaves both behind
/// and the position stands where it did.
///
/// `actor` is the role the consumer ran under, zero for a path with no
/// session, such as a background workflow run or a delivery pass.
///
/// Answers with the entries to install once the commit lands
pub fn log_stream_advances(
    server: &Arc<ServerState>,
    txn: &mut zyron_storage::txn::Transaction,
    advances: &[zyron_executor::context::PendingStreamAdvance],
    actor: u32,
) -> Result<Vec<ChangeStreamEntry>, ZyronError> {
    if advances.is_empty() {
        return Ok(Vec::new());
    }
    let at = now_micros();
    let mut updated = Vec::with_capacity(advances.len());
    for advance in advances {
        let Some(entry) = server.catalog.get_change_stream_by_id(advance.stream_id) else {
            continue;
        };
        let mut next = ChangeStreamEntry::clone(&entry);
        for (table_id, version, consumed) in &advance.positions {
            next.set_position(*table_id, *version, *consumed);
        }
        next.last_advanced_at = at;
        next.last_advanced_by = actor;
        // The first consume of a stream created with SHOW INITIAL ROWS is
        // the one that yields the existing rows, and once it commits the
        // feed is what the stream reads from
        next.initial_rows_pending = false;
        let lsn = server
            .catalog
            .log_change_stream_advance(txn.txn_id, txn.last_lsn(), &next)?;
        txn.set_last_lsn(lsn);
        txn.mark_wrote_data();
        updated.push(next);
    }
    Ok(updated)
}

/// Installs the advances a committed transaction made and releases the
/// positions it held.
///
/// The durable record was written under that transaction, so this only brings
/// the in-memory entry and its catalog row into agreement with what the log
/// already says. The positions are released after the entries are in place,
/// whatever the outcome, so a consumer that was waiting on one reads the
/// position this transaction left and no position outlives its holder
pub async fn install_stream_advances(
    server: &Arc<ServerState>,
    txn_id: u64,
    entries: Vec<ChangeStreamEntry>,
) -> Result<(), ZyronError> {
    let mut outcome = Ok(());
    for entry in entries {
        if let Err(e) = server.catalog.apply_change_stream_advance(entry).await {
            outcome = Err(e);
            break;
        }
    }
    server.txn_manager.stream_positions().unlock_all(txn_id);
    outcome
}

/// Releases the positions a transaction held without installing anything,
/// for a commit that failed after its advances were logged
pub fn release_stream_positions(server: &Arc<ServerState>, txn_id: u64) {
    server.txn_manager.stream_positions().unlock_all(txn_id);
}

/// Re-addresses a replicated advance in this member's own feed versions.
///
/// The consumed count is the same number on every member, and each member's
/// version index turns it back into the version that names that place here
pub fn localize(server: &Arc<ServerState>, entry: &mut ChangeStreamEntry) {
    let Some(feeds) = server.cdc_registry.as_ref().cloned() else {
        return;
    };
    let runtime = ChangeStreamRuntime::new(feeds);
    change_stream::localize_positions(&runtime, entry);
}

// ---------------------------------------------------------------------------
// CREATE CHANGE STREAM
// ---------------------------------------------------------------------------

pub async fn handle_create_change_stream(
    stmt: &CreateChangeStreamStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
    active_branch: &Option<String>,
) -> Result<DdlResult, ProtocolError> {
    let runtime = runtime_of(server)?;
    let (schema_id, name) =
        crate::ddl_dispatch::resolve_qualified_name(&stmt.name, server, session)?;

    if server.catalog.get_change_stream(schema_id, &name).is_some() {
        if stmt.if_not_exists {
            return Ok(DdlResult::Tag("CREATE CHANGE STREAM".to_string()));
        }
        return Err(ProtocolError::Database(ZyronError::Internal(format!(
            "change stream '{}' already exists",
            stmt.name
        ))));
    }

    // The source, plus the predicate and projection a view contributes,
    // and the columns a stream over several tables yields, which is where
    // COLUMNS on such a stream names its columns from
    let (source, mut predicate, mut projection, union) = match &stmt.target {
        ChangeStreamTarget::Table(table_name) => {
            let table = resolve_source_table(server, session, table_name)?;
            (
                ChangeStreamSource::Table(table.id.0),
                None,
                Vec::new(),
                None,
            )
        }
        ChangeStreamTarget::Tables(names) => {
            let mut tables = Vec::with_capacity(names.len());
            for table_name in names {
                tables.push(resolve_source_table(server, session, table_name)?);
            }
            tables.sort_by_key(|table| table.id.0);
            tables.dedup_by_key(|table| table.id.0);
            // The stream yields every column any source has, so a name two
            // sources hold at different types is refused now rather than
            // at the first read
            let union = zyron_planner::change_scan::multi_table_columns(&tables, 0)
                .map_err(ProtocolError::Database)?;
            let ids = tables.iter().map(|table| table.id.0).collect();
            (
                ChangeStreamSource::Tables(ids),
                None,
                Vec::new(),
                Some(union),
            )
        }
        ChangeStreamTarget::View(view_name) => {
            let (view_schema, bare) =
                crate::ddl_dispatch::resolve_qualified_name(view_name, server, session)?;
            let view = server.catalog.get_view(view_schema, &bare).ok_or_else(|| {
                ProtocolError::Database(ZyronError::Internal(format!(
                    "view '{view_name}' does not exist"
                )))
            })?;
            let shape = view_shape(server, session, &view.definition_sql)?;
            let plan = plan_view_stream(view_name, &shape).map_err(ProtocolError::Database)?;
            // The base table is what the stream reads, so SELECT on it is
            // what a reader has to hold. A view does not widen that
            let base = server
                .catalog
                .get_table_by_id(zyron_catalog::TableId(plan.base_table))
                .map_err(ProtocolError::Database)?;
            crate::ddl_dispatch::check_ddl_privilege(
                server,
                session,
                zyron_auth::PrivilegeType::Select,
                zyron_auth::ObjectType::Table,
                base.id.0,
            )?;
            (
                ChangeStreamSource::View {
                    view_id: view.id,
                    base_table: plan.base_table,
                },
                plan.predicate,
                plan.projection.into_iter().map(ColumnId).collect(),
                None,
            )
        }
    };

    // A WHERE written on the stream narrows further, and a view's own
    // predicate is kept alongside it
    if let Some(expr) = &stmt.predicate {
        let written = zyron_parser::expr_to_sql(expr);
        predicate = Some(match predicate {
            Some(existing) => format!("({existing}) AND ({written})"),
            None => written,
        });
    }

    // COLUMNS on the stream narrows the view's projection rather than
    // widening it. A stream over several tables names its columns out of
    // the union they yield, each addressed by the id it has on the first
    // table that holds it, which is how the scan addresses them
    let named = match &union {
        Some(union) => union_column_ids(union, &stmt.columns)?,
        None => {
            let source_table = server
                .catalog
                .get_table_by_id(zyron_catalog::TableId(source.table_ids()[0]))
                .map_err(ProtocolError::Database)?;
            column_ids(&source_table, &stmt.columns)?
        }
    };
    if let Some(named) = named {
        projection = match projection.is_empty() {
            true => named,
            false => named
                .into_iter()
                .filter(|id| projection.contains(id))
                .collect(),
        };
    }

    let origin = match &stmt.start {
        ChangeStreamStart::Now => ChangeStreamOrigin::Now,
        ChangeStreamStart::Version(expr) => {
            ChangeStreamOrigin::Version(integer_of(expr, "AT VERSION")? as u64)
        }
        ChangeStreamStart::Timestamp(expr) => {
            ChangeStreamOrigin::Timestamp(timestamp_of(expr, "AT TIMESTAMP")?)
        }
        ChangeStreamStart::InitialRows => ChangeStreamOrigin::InitialRows,
    };

    // A stream created on a branch reads the branch's feed, which is opened
    // now at the table's current version when the branch has not written
    // the table yet, so the stream's first read starts where the branch did.
    // A lake table's changes are derived from the head the branch keeps on
    // it, forked now on the same terms
    let branch = active_branch_id(server, active_branch);
    if let Some(branch) = branch {
        for table_id in source.table_ids() {
            let lake = server
                .catalog
                .get_table_by_id(zyron_catalog::TableId(table_id))
                .map(|table| table.lake.is_lake())
                .unwrap_or(false);
            if lake {
                crate::change_feed_bridge::open_lake_branch_source(server, table_id, branch)
                    .map_err(ProtocolError::Database)?;
            } else {
                runtime
                    .feeds()
                    .open_branch_feed(table_id, branch)
                    .map_err(ProtocolError::Database)?;
            }
        }
    }
    let position = change_stream::initial_position(&runtime, &source, origin, branch)
        .map_err(ProtocolError::Database)?;

    let mut entry = ChangeStreamEntry {
        id: 0,
        catalog_id: crate::ddl_dispatch::get_session_database(session)?,
        schema_id,
        name: name.clone(),
        source,
        position: Vec::new(),
        created_at: now_micros(),
        created_from: origin,
        mode: if stmt.append_only {
            ChangeStreamMode::AppendOnly
        } else {
            ChangeStreamMode::Standard
        },
        predicate,
        columns: if projection.is_empty() {
            None
        } else {
            Some(projection)
        },
        owner_id: crate::ddl_dispatch::actor_role_id(session),
        last_advanced_at: 0,
        last_advanced_by: 0,
        stale: false,
        stale_reason: String::new(),
        needs_attention: false,
        attention_reason: String::new(),
        initial_rows_pending: origin == ChangeStreamOrigin::InitialRows,
        branch,
    };
    for (table_id, version, consumed) in position {
        entry.set_position(table_id, version, consumed);
    }

    // The node's own cap on streams per table, the catalog's when nothing
    // configures one
    let cap = match crate::lifecycle_dispatch::cdc_setting(server, "cdc.change_streams_per_table") {
        0 => zyron_catalog::Catalog::CHANGE_STREAMS_PER_TABLE_CAP,
        configured => configured as usize,
    };
    server
        .catalog
        .create_change_stream_under(entry, cap)
        .await
        .map_err(ProtocolError::Database)?;

    tracing::info!(
        target: "zyron::audit",
        event = "ChangeStreamCreated",
        stream = %stmt.name,
        actor_role = crate::ddl_dispatch::actor_role_id(session),
    );
    Ok(DdlResult::Tag("CREATE CHANGE STREAM".to_string()))
}

/// Creates the change stream an outbound CDC stream consumes when it names
/// none of its own, one over the table, positioned at its current version,
/// in the table's schema.
///
/// The position is the one the outbound stream's delivery moves, so what
/// `zyron_sys.cdc.change_streams` shows for it is where delivery has got to
pub async fn create_implicit_stream(
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
    name: &str,
    table: &TableEntry,
) -> Result<(), ProtocolError> {
    create_implicit_stream_at(
        server,
        crate::ddl_dispatch::get_session_database(session)?,
        crate::ddl_dispatch::actor_role_id(session),
        name,
        table,
        ChangeStreamOrigin::Now,
    )
    .await
    .map_err(ProtocolError::Database)
}

/// Creates the change stream an outbound CDC stream consumes, owned by
/// `owner_id` and positioned by `origin`.
///
/// The origin is the table's current version for a stream created with the
/// outbound stream, and the version an outbound stream's earlier delivery
/// record had reached when that record is carried over into a stream
pub async fn create_implicit_stream_at(
    server: &Arc<ServerState>,
    catalog_id: zyron_catalog::DatabaseId,
    owner_id: u32,
    name: &str,
    table: &TableEntry,
    origin: ChangeStreamOrigin,
) -> Result<(), ZyronError> {
    if !table.cdf_enabled {
        return Err(ZyronError::CdcStreamError(format!(
            "table '{}' has no change data feed, so an outbound stream over it would deliver \
             nothing. Enable it with ALTER TABLE {} SET (change_data_feed = true)",
            table.name, table.name
        )));
    }
    let runtime = runtime_of(server).map_err(|e| match e {
        ProtocolError::Database(e) => e,
        other => ZyronError::CdcStreamError(other.to_string()),
    })?;
    if server
        .catalog
        .get_change_stream(table.schema_id, name)
        .is_some()
    {
        return Err(ZyronError::CdcStreamError(format!(
            "change stream '{name}' already exists, so an outbound stream of that name would \
             share its position. Name the outbound stream differently or consume the stream \
             with FROM CHANGE STREAM"
        )));
    }
    let source = ChangeStreamSource::Table(table.id.0);
    let position = change_stream::initial_position(&runtime, &source, origin, None)?;
    let mut entry = ChangeStreamEntry {
        id: 0,
        catalog_id,
        schema_id: table.schema_id,
        name: name.to_string(),
        source,
        position: Vec::new(),
        created_at: now_micros(),
        created_from: origin,
        mode: ChangeStreamMode::Standard,
        predicate: None,
        columns: None,
        owner_id,
        last_advanced_at: 0,
        last_advanced_by: 0,
        stale: false,
        stale_reason: String::new(),
        needs_attention: false,
        attention_reason: String::new(),
        initial_rows_pending: false,
        branch: None,
    };
    for (table_id, version, consumed) in position {
        entry.set_position(table_id, version, consumed);
    }
    server.catalog.create_change_stream(entry).await?;
    tracing::info!(
        target: "zyron::audit",
        event = "ChangeStreamCreated",
        stream = %name,
        actor_role = owner_id,
    );
    Ok(())
}

/// Drops the change stream an outbound CDC stream created for itself
pub async fn drop_implicit_stream(
    server: &Arc<ServerState>,
    name: &str,
) -> Result<(), ProtocolError> {
    let Some(entry) = server
        .catalog
        .list_change_streams()
        .into_iter()
        .find(|entry| entry.name == name)
    else {
        return Ok(());
    };
    server
        .catalog
        .drop_change_stream(entry.schema_id, name)
        .await
        .map_err(ProtocolError::Database)?;
    crate::ddl_dispatch::forget_object_privileges(
        server,
        zyron_auth::ObjectType::ChangeStream,
        entry.id,
    )
    .await?;
    tracing::info!(
        target: "zyron::audit",
        event = "ChangeStreamDropped",
        stream = %name,
    );
    Ok(())
}

/// Resolves a source table and checks the reader holds SELECT on it.
///
/// A stream never widens what its owner could read, so the check is the same
/// one a plain SELECT of the table goes through
fn resolve_source_table(
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
    name: &str,
) -> Result<Arc<TableEntry>, ProtocolError> {
    let (schema_id, table_name) =
        crate::ddl_dispatch::resolve_qualified_name(name, server, session)?;
    let table = server
        .catalog
        .get_table(schema_id, &table_name)
        .map_err(ProtocolError::Database)?;
    crate::ddl_dispatch::check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Select,
        zyron_auth::ObjectType::Table,
        table.id.0,
    )?;
    if !table.cdf_enabled {
        return Err(ProtocolError::Database(ZyronError::CdcStreamError(
            format!(
                "table '{name}' has no change data feed, so a change stream over it would follow \
             nothing. Enable it with ALTER TABLE {name} SET (change_data_feed = true)"
            ),
        )));
    }
    Ok(table)
}

// ---------------------------------------------------------------------------
// ALTER CHANGE STREAM
// ---------------------------------------------------------------------------

pub async fn handle_alter_change_stream(
    stmt: &AlterChangeStreamStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let runtime = runtime_of(server)?;
    let (schema_id, name) =
        crate::ddl_dispatch::resolve_qualified_name(&stmt.name, server, session)?;
    let entry = server
        .catalog
        .get_change_stream(schema_id, &name)
        .ok_or_else(|| {
            ProtocolError::Database(ZyronError::Internal(format!(
                "change stream '{}' does not exist",
                stmt.name
            )))
        })?;
    crate::ddl_dispatch::check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::ManageChangeStream,
        zyron_auth::ObjectType::ChangeStream,
        entry.id,
    )?;

    // The stream's row is written under its position lock, so a consume in
    // flight finishes and installs before this writes rather than over it.
    // On a member of a group the connection holds the position from before
    // the statement was agreed, see `resolve_reset_for_group`, because on
    // that member this runs on the applier's turn and a consume holding
    // the position waits on that same applier to commit. Alone, the
    // position is taken here
    let locks = server.txn_manager.stream_positions();
    let owner = reset_lock_owner(entry.id);
    let held_here = server.replication.is_none();
    if held_here {
        locks
            .lock_wait(owner, entry.id as u64)
            .await
            .map_err(ProtocolError::Database)?;
    }
    let outcome = alter_change_stream_locked(stmt, server, session, &runtime, &entry).await;
    if held_here {
        locks.unlock_all(owner);
    }
    outcome
}

/// The owner an ALTER CHANGE STREAM holds a stream's position under, apart
/// from every transaction's own id, so one statement per stream at a time
/// and no transaction's release can let go of it
fn reset_lock_owner(stream_id: u32) -> u64 {
    (1u64 << 63) | stream_id as u64
}

/// The body of ALTER CHANGE STREAM, run with the stream's position held
async fn alter_change_stream_locked(
    stmt: &AlterChangeStreamStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
    runtime: &Arc<ChangeStreamRuntime>,
    entry: &Arc<ChangeStreamEntry>,
) -> Result<DdlResult, ProtocolError> {
    let actor = crate::ddl_dispatch::actor_role_id(session);
    let at = now_micros();
    let updated = match &stmt.action {
        AlterChangeStreamAction::Reset => {
            change_stream::reset_to(runtime, entry, ResetTarget::Earliest, actor, at)
                .map_err(ProtocolError::Database)?
        }
        AlterChangeStreamAction::ResetToLatest => {
            change_stream::reset_to(&runtime, &entry, ResetTarget::Latest, actor, at)
                .map_err(ProtocolError::Database)?
        }
        AlterChangeStreamAction::ResetToVersion(expr) => change_stream::reset_to(
            &runtime,
            &entry,
            ResetTarget::Version(integer_of(expr, "RESET TO VERSION")? as u64),
            actor,
            at,
        )
        .map_err(ProtocolError::Database)?,
        AlterChangeStreamAction::ResetToTimestamp(expr) => change_stream::reset_to(
            &runtime,
            &entry,
            ResetTarget::Timestamp(timestamp_of(expr, "RESET TO TIMESTAMP")?),
            actor,
            at,
        )
        .map_err(ProtocolError::Database)?,
        AlterChangeStreamAction::ResetToPosition(expr) => {
            // A count names the same place on every member, so this is the
            // form a reset takes once it reaches a group, one count per
            // source in the order the stream lists them or one count for
            // every source. Each member turns it back into the version that
            // addresses it here
            let counts: Vec<u64> = match expr {
                Expr::ArrayConstructor(items) => items
                    .iter()
                    .map(|item| integer_of(item, "RESET TO POSITION").map(|n| n.max(0) as u64))
                    .collect::<Result<_, _>>()?,
                other => vec![integer_of(other, "RESET TO POSITION")?.max(0) as u64],
            };
            let mut next = ChangeStreamEntry::clone(&entry);
            if counts.is_empty() || (counts.len() != 1 && counts.len() != next.position.len()) {
                return Err(ProtocolError::Database(ZyronError::ParseError(format!(
                    "RESET TO POSITION names {} counts, and change stream '{}' reads {} sources",
                    counts.len(),
                    stmt.name,
                    next.position.len()
                ))));
            }
            for (at, slot) in next.position.iter_mut().enumerate() {
                slot.consumed = if counts.len() == 1 {
                    counts[0]
                } else {
                    counts[at]
                };
            }
            change_stream::localize_positions(&runtime, &mut next);
            next.stale = false;
            next.stale_reason.clear();
            next.last_advanced_at = at;
            next.last_advanced_by = actor;
            next.initial_rows_pending = false;
            next
        }
        AlterChangeStreamAction::SetAllColumns => {
            let mut next = ChangeStreamEntry::clone(&entry);
            next.columns = None;
            next.needs_attention = false;
            next.attention_reason.clear();
            next
        }
        AlterChangeStreamAction::SetColumns(names) => {
            let table = server
                .catalog
                .get_table_by_id(zyron_catalog::TableId(entry.source.table_ids()[0]))
                .map_err(ProtocolError::Database)?;
            let mut next = ChangeStreamEntry::clone(&entry);
            next.columns = column_ids(&table, names)?;
            // A list that no longer names a dropped column is a list the
            // stream can yield through again
            next.needs_attention = false;
            next.attention_reason.clear();
            next
        }
    };

    server
        .catalog
        .update_change_stream(updated)
        .await
        .map_err(ProtocolError::Database)?;

    tracing::info!(
        target: "zyron::audit",
        event = "ChangeStreamReset",
        stream = %stmt.name,
        actor_role = actor,
    );
    Ok(DdlResult::Tag("ALTER CHANGE STREAM".to_string()))
}

/// Rewrites a reset into the count form the group receives.
///
/// The version a person types addresses a place in this member's own feed,
/// and the same number would address a different place on another member.
/// Resolving it here and sending the count is what makes every member land on
/// the same change
pub fn resolve_reset_for_group(
    stmt: &AlterChangeStreamStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<Option<AlterChangeStreamStatement>, ZyronError> {
    let (schema_id, name) =
        crate::ddl_dispatch::resolve_qualified_name(&stmt.name, server, session)
            .map_err(|e| ZyronError::Internal(e.to_string()))?;
    let Some(entry) = server.catalog.get_change_stream(schema_id, &name) else {
        return Ok(None);
    };
    // The stream's position is held from here until the statement has run
    // on this member, see `handle_alter_change_stream`, so a consume that
    // commits meanwhile cannot install over what the statement writes. A
    // consume already holding it is not waited for, since its commit waits
    // on the applier this statement is about to occupy
    let owner = reset_lock_owner(entry.id);
    if server
        .txn_manager
        .stream_positions()
        .try_lock(owner, entry.id as u64)
        .is_err()
    {
        return Err(ZyronError::CdcStreamError(format!(
            "change stream '{}' is being consumed, retry once that consume has committed",
            stmt.name
        )));
    }
    if let Some(session) = session.as_mut() {
        session.reset_lock_owner = Some(owner);
    }
    let needs_resolving = matches!(
        stmt.action,
        AlterChangeStreamAction::Reset
            | AlterChangeStreamAction::ResetToLatest
            | AlterChangeStreamAction::ResetToVersion(_)
            | AlterChangeStreamAction::ResetToTimestamp(_)
    );
    if !needs_resolving {
        return Ok(None);
    }
    let Some(feeds) = server.cdc_registry.as_ref().cloned() else {
        return Ok(None);
    };
    let runtime = ChangeStreamRuntime::new(feeds);
    let target = match &stmt.action {
        AlterChangeStreamAction::Reset => ResetTarget::Earliest,
        AlterChangeStreamAction::ResetToLatest => ResetTarget::Latest,
        AlterChangeStreamAction::ResetToVersion(expr) => match expr {
            Expr::Literal(LiteralValue::Integer(n)) => ResetTarget::Version(*n as u64),
            _ => return Ok(None),
        },
        AlterChangeStreamAction::ResetToTimestamp(expr) => match expr {
            Expr::Literal(LiteralValue::Integer(n)) => ResetTarget::Timestamp(*n),
            Expr::Literal(LiteralValue::String(text)) => ResetTarget::Timestamp(
                zyron_common::interval::parse_timestamp_micros(text)
                    .map_err(|e| ZyronError::ParseError(e.to_string()))?,
            ),
            _ => return Ok(None),
        },
        _ => return Ok(None),
    };
    let resolved = change_stream::reset_to(&runtime, &entry, target, 0, 0)?;
    // The stream's own schema names it, so every member resolves the same
    // stream whatever search path it runs the statement under
    let schema = server.catalog.get_schema_by_id(schema_id)?;
    // One count per source in the order the stream lists them, which is
    // the same order on every member. A stream over one table sends the
    // count bare
    let mut counts: Vec<Expr> = resolved
        .position
        .iter()
        .map(|slot| Expr::Literal(LiteralValue::Integer(slot.consumed as i64)))
        .collect();
    let position = if counts.len() == 1 {
        counts.remove(0)
    } else {
        Expr::ArrayConstructor(counts)
    };
    Ok(Some(AlterChangeStreamStatement {
        name: format!("{}.{}", schema.name, name),
        action: AlterChangeStreamAction::ResetToPosition(position),
    }))
}

// ---------------------------------------------------------------------------
// DROP CHANGE STREAM
// ---------------------------------------------------------------------------

pub async fn handle_drop_change_stream(
    stmt: &DropChangeStreamStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (schema_id, name) =
        crate::ddl_dispatch::resolve_qualified_name(&stmt.name, server, session)?;
    let Some(entry) = server.catalog.get_change_stream(schema_id, &name) else {
        if stmt.if_exists {
            return Ok(DdlResult::Tag("DROP CHANGE STREAM".to_string()));
        }
        return Err(ProtocolError::Database(ZyronError::Internal(format!(
            "change stream '{}' does not exist",
            stmt.name
        ))));
    };
    crate::ddl_dispatch::check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::ManageChangeStream,
        zyron_auth::ObjectType::ChangeStream,
        entry.id,
    )?;

    server
        .catalog
        .drop_change_stream(schema_id, &name)
        .await
        .map_err(ProtocolError::Database)?;
    // The grants on a dropped stream go with it, so an id issued again never
    // arrives carrying what the last holder was allowed
    crate::ddl_dispatch::forget_object_privileges(
        server,
        zyron_auth::ObjectType::ChangeStream,
        entry.id,
    )
    .await?;

    tracing::info!(
        target: "zyron::audit",
        event = "ChangeStreamDropped",
        stream = %stmt.name,
        actor_role = crate::ddl_dispatch::actor_role_id(session),
    );
    Ok(DdlResult::Tag("DROP CHANGE STREAM".to_string()))
}

// ---------------------------------------------------------------------------
// SHOW CHANGE STREAMS
// ---------------------------------------------------------------------------

pub async fn handle_show_change_streams(
    stmt: &ShowChangeStreamsStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let runtime = runtime_of(server)?;
    let now = now_micros();

    let entries: Vec<Arc<ChangeStreamEntry>> = match (&stmt.name, &stmt.on_table) {
        (Some(name), _) => {
            let (schema_id, bare) =
                crate::ddl_dispatch::resolve_qualified_name(name, server, session)?;
            let entry = server
                .catalog
                .get_change_stream(schema_id, &bare)
                .ok_or_else(|| {
                    ProtocolError::Database(ZyronError::Internal(format!(
                        "change stream '{name}' does not exist"
                    )))
                })?;
            vec![entry]
        }
        (None, Some(table_name)) => {
            let (schema_id, bare) =
                crate::ddl_dispatch::resolve_qualified_name(table_name, server, session)?;
            let table = server
                .catalog
                .get_table(schema_id, &bare)
                .map_err(ProtocolError::Database)?;
            server.catalog.change_streams_on_table(table.id.0)
        }
        (None, None) => server.catalog.list_change_streams(),
    };

    let columns = vec![
        ("stream".to_string(), crate::types::PG_TEXT_OID),
        ("sources".to_string(), crate::types::PG_TEXT_OID),
        ("position".to_string(), crate::types::PG_TEXT_OID),
        ("mode".to_string(), crate::types::PG_TEXT_OID),
        ("stale".to_string(), crate::types::PG_TEXT_OID),
        ("needs_attention".to_string(), crate::types::PG_TEXT_OID),
        ("pending_rows".to_string(), crate::types::PG_INT8_OID),
        ("lag_seconds".to_string(), crate::types::PG_INT8_OID),
    ];
    let rows = entries
        .iter()
        .map(|entry| {
            let status = runtime.status(entry, now);
            vec![
                entry.name.clone(),
                render_sources(entry),
                render_position(entry),
                entry.mode.name().to_string(),
                entry.stale.to_string(),
                entry.needs_attention.to_string(),
                status.pending_rows.to_string(),
                status.lag_seconds.to_string(),
            ]
        })
        .collect();

    Ok(DdlResult::Rows {
        tag: "SHOW CHANGE STREAMS".to_string(),
        columns,
        rows,
    })
}

/// The source tables a stream reads, as the views render them
pub fn render_sources(entry: &ChangeStreamEntry) -> String {
    entry
        .source
        .table_ids()
        .iter()
        .map(|id| id.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

/// The stream's position, one `table:version` pair per source
pub fn render_position(entry: &ChangeStreamEntry) -> String {
    entry
        .position
        .iter()
        .map(|slot| format!("{}:{}", slot.table_id, slot.version))
        .collect::<Vec<_>>()
        .join(",")
}

/// The consumed record counts, one per source. This is the number that agrees
/// across a consensus group, so it is what a conformance probe compares
pub fn render_consumed(entry: &ChangeStreamEntry) -> String {
    entry
        .position
        .iter()
        .map(|slot| format!("{}:{}", slot.table_id, slot.consumed))
        .collect::<Vec<_>>()
        .join(",")
}

/// Marks every stream on a table stale for one reason, answering with the
/// entries that changed so the caller makes them durable.
///
/// Turning a feed off does not drop the streams that read it. They keep their
/// positions, and an operator sees what turning it off broke
pub fn mark_streams_stale(
    server: &Arc<ServerState>,
    table_id: u32,
    reason: &str,
) -> Vec<ChangeStreamEntry> {
    server
        .catalog
        .change_streams_on_table(table_id)
        .into_iter()
        .filter(|entry| !entry.stale || entry.stale_reason != reason)
        .map(|entry| {
            let mut updated = ChangeStreamEntry::clone(&entry);
            updated.stale = true;
            updated.stale_reason = reason.to_string();
            updated
        })
        .collect()
}

/// Marks every stream on a table as needing attention for one reason
pub fn mark_streams_needing_attention(
    server: &Arc<ServerState>,
    table_id: u32,
    reason: &str,
) -> Vec<ChangeStreamEntry> {
    server
        .catalog
        .change_streams_on_table(table_id)
        .into_iter()
        .filter(|entry| !entry.needs_attention || entry.attention_reason != reason)
        .map(|entry| {
            let mut updated = ChangeStreamEntry::clone(&entry);
            updated.needs_attention = true;
            updated.attention_reason = reason.to_string();
            updated
        })
        .collect()
}

/// Re-checks every stream on a table against the columns it names.
///
/// A stream naming a column the table no longer has keeps its position and
/// stops yielding, which is what `needs_attention` records, and it resumes
/// once an ALTER drops that column from its list
pub fn recheck_stream_columns(
    server: &Arc<ServerState>,
    table: &TableEntry,
) -> Vec<ChangeStreamEntry> {
    let mut changed = Vec::new();
    for entry in server.catalog.change_streams_on_table(table.id.0) {
        let Some(columns) = &entry.columns else {
            continue;
        };
        let missing = zyron_cdc::schema_evolution::missing_stream_column(table, columns);
        match missing {
            Some(column) => {
                let reason = format!("column '{column}' was dropped from the source");
                if entry.needs_attention && entry.attention_reason == reason {
                    continue;
                }
                let mut updated = ChangeStreamEntry::clone(&entry);
                updated.needs_attention = true;
                updated.attention_reason = reason;
                changed.push(updated);
            }
            None => {
                if !entry.needs_attention {
                    continue;
                }
                let mut updated = ChangeStreamEntry::clone(&entry);
                updated.needs_attention = false;
                updated.attention_reason.clear();
                changed.push(updated);
            }
        }
    }
    changed
}

/// Writes a batch of stream entry changes through the catalog
pub async fn persist_stream_changes(
    server: &Arc<ServerState>,
    changed: Vec<ChangeStreamEntry>,
) -> Result<(), ProtocolError> {
    for entry in changed {
        server
            .catalog
            .update_change_stream(entry)
            .await
            .map_err(ProtocolError::Database)?;
    }
    Ok(())
}

/// The schema a change stream's own name resolves in, for the paths that
/// carry a stream id rather than a name
pub fn schema_of(server: &Arc<ServerState>, stream_id: u32) -> Option<SchemaId> {
    server
        .catalog
        .get_change_stream_by_id(stream_id)
        .map(|entry| entry.schema_id)
}

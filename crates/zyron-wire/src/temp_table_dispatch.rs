//! Creating, dropping and reclaiming temporary tables.
//!
//! A temporary table is a heap table whose files live under
//! `<data_dir>/tmp/<session_key>/` and whose definition lives in the session
//! that created it. Nothing here writes the catalog, the write-ahead log, or
//! the consensus log, which is what makes the whole feature node-local by
//! construction rather than by a rule someone has to remember.

use std::sync::Arc;

use zyron_catalog::{SessionTempTables, TempTable, TempTableRegistry};
use zyron_common::{Result, ZyronError};
use zyron_parser::ast::{CreateTableStatement, OnCommitAction};
use zyron_storage::{HeapFile, HeapFileConfig};

use crate::connection::ServerState;
use crate::session::Session;

/// The session's temporary namespace, created on first use and pointed at
/// this node's data directory.
pub fn session_namespace(
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<Arc<SessionTempTables>> {
    let Some(session) = session.as_mut() else {
        return Err(ZyronError::ConfigError(
            "a temporary table belongs to a session, and this statement is running without one"
                .to_string(),
        ));
    };
    if let Some(existing) = &session.temp_tables {
        return Ok(Arc::clone(existing));
    }
    let registry = server.catalog.temp_tables();
    registry.set_data_dir(&server.data_dir);
    if let Some(budget) = server.max_query_memory {
        registry.set_default_max_bytes(
            (budget as f64 * zyron_catalog::DEFAULT_TEMP_TABLE_BYTES_FRACTION) as u64,
        );
    }
    let namespace = registry.session(session.session_key, session.process_id);
    // One directory per session, made once here rather than per file placed
    // in it, which is a filesystem call per table otherwise
    std::fs::create_dir_all(namespace.directory()).map_err(|e| {
        ZyronError::IoError(format!(
            "creating the session's temporary table directory {}: {e}",
            namespace.directory().display()
        ))
    })?;
    session.temp_tables = Some(Arc::clone(&namespace));
    session.temp_table_guard = Some(TempTableGuard::new(Arc::clone(server), session.session_key));
    Ok(namespace)
}

/// Creates one temporary table.
///
/// The entry is built by the catalog so the column and constraint conversion
/// is the same one a permanent table goes through, and is then registered in
/// the session rather than stored, logged or replicated.
pub async fn create(
    stmt: &CreateTableStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<String> {
    refuse_unsupported_clauses(stmt)?;
    let namespace = session_namespace(server, session)?;
    let existing = namespace.contains(&stmt.name);
    if existing {
        if stmt.if_not_exists {
            return Ok("CREATE TABLE".to_string());
        }
        if !stmt.or_replace {
            return Err(ZyronError::TableAlreadyExists(stmt.name.clone()));
        }
    }
    namespace.check_limits()?;

    // A temporary table has no schema of its own, so its entry carries the
    // session's first search-path schema for the column ids the converter
    // stamps. Nothing resolves it by that schema: only the session's own
    // namespace ever finds it
    let schema_id = default_schema(server, session)?;
    let entry = server.catalog.build_temp_table_entry(
        schema_id,
        &stmt.name,
        &stmt.columns,
        &stmt.constraints,
    )?;
    let entry = Arc::new(entry);

    // The files go in the session's own directory, registered before the
    // heap is opened because the path is read when a file is first opened
    let directory = namespace.directory().to_path_buf();
    server
        .disk_manager
        .place_file_in(entry.heap_file_id, &directory);
    server
        .disk_manager
        .place_file_in(entry.fsm_file_id, &directory);
    // A failure between placing the files and opening them leaves the two
    // placements behind, and the ids descend without reuse, so nothing would
    // ever claim them again
    let opened = async {
        let heap = HeapFile::new(
            Arc::clone(&server.disk_manager),
            Arc::clone(&server.buffer_pool),
            HeapFileConfig {
                heap_file_id: entry.heap_file_id,
                fsm_file_id: entry.fsm_file_id,
            },
        )?;
        heap.init_cache().await?;
        Ok::<_, ZyronError>(heap)
    }
    .await;
    let heap = match opened {
        Ok(heap) => heap,
        Err(e) => {
            server
                .disk_manager
                .forget_file_placement(entry.heap_file_id);
            server.disk_manager.forget_file_placement(entry.fsm_file_id);
            return Err(e);
        }
    };
    let _ = server
        .heap_files
        .insert_async(entry.heap_file_id, Arc::new(heap))
        .await;

    // A replacement drops what it replaced only once the new table's files
    // exist, so a failure part way leaves the old one in place
    if let Some(replaced) = namespace.insert(Arc::clone(&entry), on_commit_of(stmt)) {
        // The replaced table's id stops resolving before its files go, so
        // nothing reaches an entry whose heap has been unlinked
        server.catalog.temp_tables().forget(replaced.entry.id);
        reclaim(server, &replaced).await;
    }
    server.catalog.temp_tables().register(entry);
    Ok("CREATE TABLE".to_string())
}

/// Builds a B+tree index on a temporary table.
///
/// There is no publish, wait and flip here, and that is not a shortcut: the
/// online sequence exists so a build catches rows other transactions write
/// while it scans, and a temporary table has exactly one writer, the session
/// running this statement. The tree is built from what is there and is ready
/// when the scan ends.
pub async fn create_index(
    stmt: &zyron_parser::ast::CreateIndexStatement,
    key_columns: &[(String, bool)],
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<Option<String>> {
    let Some(namespace) = session
        .as_ref()
        .and_then(|s| s.temp_tables.as_ref())
        .map(Arc::clone)
    else {
        return Ok(None);
    };
    if stmt.table.contains('.') {
        return Ok(None);
    }
    let Some(table) = namespace.get(&stmt.table) else {
        return Ok(None);
    };
    let issuer = session
        .as_ref()
        .map(|s| s.user.clone())
        .unwrap_or_else(|| "unknown".to_string());

    let entry = server.catalog.create_temp_btree_index(
        table.entry.id,
        table.entry.schema_id,
        &stmt.name,
        key_columns,
        stmt.unique,
    )?;

    // The index's file and its checkpoint sit beside the table's, so one
    // directory holds everything the session owns and one removal clears it
    let directory = namespace.directory().to_path_buf();
    server
        .disk_manager
        .place_file_in(entry.index_file_id, &directory);
    // A failed create leaves the placement behind and leaves the entry the
    // catalog already published pointing at a file that does not exist, so
    // both are taken back before the error goes up
    let created = zyron_storage::BTreeIndex::create(entry.index_file_id, directory.clone()).await;
    let btree = match created {
        Ok(btree) => Arc::new(btree),
        Err(e) => {
            server
                .disk_manager
                .forget_file_placement(entry.index_file_id);
            server.catalog.forget_temp_index(entry.id);
            return Err(e);
        }
    };
    let _ = server
        .btree_indexes
        .insert_async(entry.id.0, Arc::clone(&btree))
        .await;

    let progress = server.ddl_progress.begin(
        &table.entry.name,
        &stmt.name,
        crate::ddl_progress::DdlOperation::CreateIndex,
        &issuer,
    );
    let key_ids: Vec<zyron_catalog::ColumnId> = entry.columns.iter().map(|c| c.column_id).collect();
    let outcome = crate::index_build::scan_and_load(
        server,
        &table.entry,
        entry.id.0,
        &key_ids,
        stmt.unique,
        &btree,
        progress.progress(),
        crate::index_build::DEFAULT_BUILD_BATCH_ROWS,
    )
    .await;

    match outcome {
        Ok(Ok(_)) => Ok(Some("CREATE INDEX".to_string())),
        Ok(Err(duplicate)) => {
            abandon_index(server, table.entry.id, entry.id.0, entry.index_file_id).await;
            Err(ZyronError::ExecutionError(format!(
                "CREATE UNIQUE INDEX \"{}\" found the same key on two live rows of the temporary table \"{}\", at {:?} and at {:?}. The index was not created",
                stmt.name, table.entry.name, duplicate.first, duplicate.second
            )))
        }
        Err(e) => {
            abandon_index(server, table.entry.id, entry.id.0, entry.index_file_id).await;
            Err(e)
        }
    }
}

/// Removes everything a failed temporary index build put in place.
async fn abandon_index(
    server: &Arc<ServerState>,
    table_id: zyron_catalog::TableId,
    index_id: u32,
    index_file_id: u32,
) {
    server.catalog.forget_temp_indexes(table_id);
    let _ = server.btree_indexes.remove_async(&index_id).await;
    let _ = server.disk_manager.delete_file(index_file_id).await;
    server.disk_manager.forget_file_placement(index_file_id);
}

/// Drops one temporary table, removing its files.
pub async fn drop_table(
    name: &str,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<bool> {
    let Some(namespace) = session
        .as_ref()
        .and_then(|s| s.temp_tables.as_ref())
        .map(Arc::clone)
    else {
        return Ok(false);
    };
    let Some(table) = namespace.remove(name) else {
        return Ok(false);
    };
    server.catalog.temp_tables().forget(table.entry.id);
    reclaim(server, &table).await;
    Ok(true)
}

/// Empties a temporary table's rows without dropping it, which is what
/// `ON COMMIT DELETE ROWS` asks for at every commit.
///
/// The files are cut to zero pages and every frame they had in the buffer
/// pool is evicted, because a cached frame or a stale heap handle would keep
/// serving rows the truncate removed.
pub async fn truncate(server: &Arc<ServerState>, table: &TempTable) -> Result<()> {
    let heap_file_id = table.entry.heap_file_id;
    let fsm_file_id = table.entry.fsm_file_id;
    let heap_pages = server.disk_manager.num_pages(heap_file_id).await?;
    let fsm_pages = server.disk_manager.num_pages(fsm_file_id).await?;
    server.disk_manager.truncate_file(heap_file_id).await?;
    server.disk_manager.truncate_file(fsm_file_id).await?;
    let _ = server.heap_files.remove_async(&heap_file_id).await;
    for page in 0..heap_pages {
        server
            .buffer_pool
            .delete_page(zyron_common::PageId::new(heap_file_id, page));
    }
    for page in 0..fsm_pages {
        server
            .buffer_pool
            .delete_page(zyron_common::PageId::new(fsm_file_id, page));
    }
    table.record_statistics(0, 0);
    Ok(())
}

/// Acts on a session's temporary tables at commit: empties the ones declared
/// ON COMMIT DELETE ROWS and drops the ones declared ON COMMIT DROP.
pub async fn on_commit(server: &Arc<ServerState>, session: &Option<Session>) {
    let Some(namespace) = session
        .as_ref()
        .and_then(|s| s.temp_tables.as_ref())
        .map(Arc::clone)
    else {
        return;
    };
    let (to_empty, to_drop) = namespace.on_commit();
    for table in to_empty {
        if let Err(e) = truncate(server, &table).await {
            tracing::warn!(
                table = %table.entry.name,
                error = %e,
                "emptying a temporary table at commit failed"
            );
        }
    }
    for table in to_drop {
        server.catalog.temp_tables().forget(table.entry.id);
        reclaim(server, &table).await;
    }
}

/// Drops every temporary table a session held, for a connection that has
/// ended or an HTTP session that has expired.
pub async fn end_session(server: &Arc<ServerState>, session_key: zyron_catalog::SessionKey) {
    let held = server.catalog.temp_tables().end_session(session_key);
    discard_held(server, held).await;
}

/// Releases the file handles of tables already taken out of the registry and
/// removes the session directory holding them.
///
/// Separate from the registry removal so a session ending outside an async
/// context can free its registry entries, and with them its share of
/// `temp_table_max_bytes` and `temp_table_max_count`, without waiting for the
/// filesystem.
async fn discard_held(server: &Arc<ServerState>, held: Vec<Arc<TempTable>>) {
    let mut directory = None;
    for table in &held {
        directory = Some(table.directory.clone());
        // The whole directory goes below, so each file's handle is released
        // rather than each file being flushed and unlinked on its own. A
        // session holding a gigabyte would otherwise pay one fsync and one
        // unlink per file for bytes the directory removal takes anyway
        release(server, table).await;
    }
    // The session's directory goes with it, so a session that created and
    // dropped tables leaves nothing behind either
    if let Some(directory) = directory
        && let Err(e) = tokio::fs::remove_dir_all(&directory).await
        && e.kind() != std::io::ErrorKind::NotFound
    {
        tracing::warn!(
            directory = %directory.display(),
            error = %e,
            "removing a session's temporary table directory failed"
        );
    }
}

/// Releases a session's temporary tables when the session itself is dropped.
///
/// A temporary table lives exactly as long as the session that created it, so
/// the release hangs off the session's lifetime rather than off the teardown
/// path of whatever opened it. A wire connection closing and an HTTP session
/// expiring both drop their `Session`, and both free the tables through this
/// without a call of their own.
pub struct TempTableGuard {
    server: Arc<ServerState>,
    session_key: zyron_catalog::SessionKey,
}

impl TempTableGuard {
    /// Binds the guard to the session that owns the namespace.
    fn new(server: Arc<ServerState>, session_key: zyron_catalog::SessionKey) -> Self {
        Self {
            server,
            session_key,
        }
    }
}

impl Drop for TempTableGuard {
    fn drop(&mut self) {
        // The registry entries go first and synchronously, so the session's
        // share of temp_table_max_bytes and temp_table_max_count is free the
        // moment the session is gone even where no runtime remains to remove
        // the files
        let held = self
            .server
            .catalog
            .temp_tables()
            .end_session(self.session_key);
        if held.is_empty() {
            return;
        }
        // Removing the files needs the async filesystem, which a drop cannot
        // await, so it runs as a task of its own. A node with no runtime left
        // is shutting down, and the next start clears the whole tree
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            return;
        };
        let server = Arc::clone(&self.server);
        handle.spawn(async move {
            discard_held(&server, held).await;
        });
    }
}

/// Collects statistics for the session's temporary tables that were written
/// since they were last collected, so the planner has cardinalities for them.
///
/// Called before a statement is planned, which is what makes collection
/// happen on the first read after a write rather than on every read.
pub async fn refresh_statistics(server: &Arc<ServerState>, session: &Option<Session>) {
    let Some(namespace) = session
        .as_ref()
        .and_then(|s| s.temp_tables.as_ref())
        .map(Arc::clone)
    else {
        return;
    };
    // A statement that follows no write asks one atomic question and stops,
    // rather than locking the namespace and walking it
    if !namespace.any_dirty() {
        return;
    }
    for table in namespace.dirty_tables() {
        match measure(server, &table).await {
            Ok((rows, bytes)) => {
                namespace.record_statistics(&table, rows, bytes);
                // The planner reads cardinalities out of the catalog's
                // statistics, so a temporary table's go in the same place a
                // permanent table's do
                server.catalog.put_stats(
                    table.entry.id,
                    zyron_catalog::TableStats {
                        table_id: table.entry.id,
                        row_count: rows,
                        page_count: bytes.div_ceil(zyron_common::PAGE_SIZE as u64) as u32,
                        avg_row_size: if rows == 0 {
                            0
                        } else {
                            (bytes / rows).min(u32::MAX as u64) as u32
                        },
                        last_analyzed: 0,
                    },
                    Vec::new(),
                );
            }
            Err(e) => {
                tracing::warn!(
                    table = %table.entry.name,
                    error = %e,
                    "collecting statistics for a temporary table failed"
                );
            }
        }
    }
}

/// A temporary table's live rows and the bytes its own files hold.
///
/// Neither figure costs a pass over the data. The row count is carried
/// forward by each write, because the session that owns the table is its only
/// writer, so a scan is owed only after a statement whose tag does not
/// separate rows added from rows removed. The byte total reads each file's
/// page count out of the disk manager, which is an atomic load rather than a
/// directory walk and a stat per file.
async fn measure(server: &Arc<ServerState>, table: &TempTable) -> Result<(u64, u64)> {
    let rows = if table.needs_row_scan() {
        count_rows(server, table).await?
    } else {
        table.rows()
    };
    Ok((rows, table_bytes(server, table)))
}

/// Counts a temporary table's live rows by scanning it.
///
/// Reached only to recover a count a write could not carry forward.
async fn count_rows(server: &Arc<ServerState>, table: &TempTable) -> Result<u64> {
    let heap = open_heap(server, table).await?;
    let guard = heap.scan()?;
    let mut rows = 0u64;
    guard.for_each(|_tid, view| {
        if !view.is_deleted() {
            rows += 1;
        }
    });
    Ok(rows)
}

/// Bytes the files of one temporary table hold, its indexes included.
///
/// Scoped to the table rather than to the session's directory, which holds
/// every table the session owns. Measuring the directory per table would
/// report each table as costing the whole session and would count the session
/// once per table against `temp_table_max_bytes`.
fn table_bytes(server: &Arc<ServerState>, table: &TempTable) -> u64 {
    let page = zyron_common::PAGE_SIZE as u64;
    let mut total = 0u64;
    let mut add = |file_id: u32| {
        total = total.saturating_add(
            server
                .disk_manager
                .pages_if_open(file_id)
                .unwrap_or(0)
                .saturating_mul(page),
        );
    };
    add(table.entry.heap_file_id);
    add(table.entry.fsm_file_id);
    for index in server.catalog.get_indexes_for_table(table.entry.id) {
        add(index.index_file_id);
    }
    total
}

/// The heap file behind a temporary table, opened if it is not already.
async fn open_heap(server: &Arc<ServerState>, table: &TempTable) -> Result<Arc<HeapFile>> {
    if let Some(hit) = server.heap_files.get_async(&table.entry.heap_file_id).await {
        return Ok(Arc::clone(hit.get()));
    }
    let heap = HeapFile::new(
        Arc::clone(&server.disk_manager),
        Arc::clone(&server.buffer_pool),
        HeapFileConfig {
            heap_file_id: table.entry.heap_file_id,
            fsm_file_id: table.entry.fsm_file_id,
        },
    )?;
    heap.init_cache().await?;
    let arc = Arc::new(heap);
    let _ = server
        .heap_files
        .insert_async(table.entry.heap_file_id, Arc::clone(&arc))
        .await;
    Ok(arc)
}

/// Forgets one temporary table's open handles and cached state without
/// unlinking anything, for a caller that is removing the whole directory.
async fn release(server: &Arc<ServerState>, table: &TempTable) {
    let _ = server
        .heap_files
        .remove_async(&table.entry.heap_file_id)
        .await;
    for index_file_id in server.catalog.forget_temp_indexes(table.entry.id) {
        server.disk_manager.discard_file(index_file_id).await;
    }
    server
        .disk_manager
        .discard_file(table.entry.heap_file_id)
        .await;
    server
        .disk_manager
        .discard_file(table.entry.fsm_file_id)
        .await;
    server.catalog.remove_stats(table.entry.id);
}

/// Removes one temporary table's files and forgets its open handles.
///
/// Failures are reported rather than returned: this runs while a session is
/// ending or a commit is landing, where there is nothing left to fail back
/// to, and a file left behind is cleared by the next node start.
async fn reclaim(server: &Arc<ServerState>, table: &TempTable) {
    let _ = server
        .heap_files
        .remove_async(&table.entry.heap_file_id)
        .await;
    // Every index on it goes too, and its file with it
    let index_files = server.catalog.forget_temp_indexes(table.entry.id);
    for index_file_id in &index_files {
        if let Err(e) = server.disk_manager.delete_file(*index_file_id).await {
            tracing::warn!(
                file_id = index_file_id,
                error = %e,
                "removing a temporary index's file failed"
            );
        }
        server.disk_manager.forget_file_placement(*index_file_id);
    }
    for file_id in [table.entry.heap_file_id, table.entry.fsm_file_id] {
        if let Err(e) = server.disk_manager.delete_file(file_id).await {
            tracing::warn!(
                file_id,
                error = %e,
                "removing a temporary table's file failed"
            );
        }
        server.disk_manager.forget_file_placement(file_id);
    }
    server.catalog.remove_stats(table.entry.id);
}

/// The schema a temporary table's column ids are stamped against.
fn default_schema(
    server: &Arc<ServerState>,
    session: &Option<Session>,
) -> Result<zyron_catalog::SchemaId> {
    let Some(session) = session else {
        return Err(ZyronError::ConfigError(
            "a temporary table belongs to a session".to_string(),
        ));
    };
    for name in &session.search_path {
        if let Ok(schema) = server.catalog.get_schema(session.database_id, name) {
            return Ok(schema.id);
        }
    }
    Err(ZyronError::ConfigError(
        "no schema on the session's search path exists, so there is nothing to create against"
            .to_string(),
    ))
}

/// What a commit does to the table this statement creates.
fn on_commit_of(stmt: &CreateTableStatement) -> OnCommitAction {
    stmt.on_commit.unwrap_or(OnCommitAction::PreserveRows)
}

/// Refuses the clauses a temporary table has no meaning for, naming the
/// clause rather than accepting it and doing nothing with it.
fn refuse_unsupported_clauses(stmt: &CreateTableStatement) -> Result<()> {
    use zyron_parser::ast::TableFormat;
    if matches!(stmt.using, Some(TableFormat::ZyronLake)) {
        return Err(ZyronError::ConfigError(
            "a temporary table is a heap table on one node; USING ZYRONLAKE writes a transaction log the cluster shares, which a session-local table has no place in".to_string(),
        ));
    }
    if stmt.cluster_by.is_some() {
        return Err(ZyronError::ConfigError(
            "CLUSTER BY lays out a lake table's files, and a temporary table is a heap table"
                .to_string(),
        ));
    }
    if stmt.clone_of.is_some() {
        return Err(ZyronError::ConfigError(
            "CLONE OF shares a lake table's data files, and a temporary table is a heap table"
                .to_string(),
        ));
    }
    if stmt.ttl.is_some() {
        return Err(ZyronError::ConfigError(
            "a TTL ages rows out on a schedule the node keeps, and a temporary table does not outlive the session that created it".to_string(),
        ));
    }
    Ok(())
}

/// The bare table name a statement writes, when it writes one.
///
/// Only a bare name can be a temporary table, so a qualified one is not
/// reported here at all: it always addresses a permanent table.
pub fn write_target(stmt: &zyron_parser::Statement) -> Option<(&str, WriteKind)> {
    use zyron_parser::Statement as S;
    let (name, kind) = match stmt {
        S::Insert(s) => (&s.table, WriteKind::Added),
        S::Update(s) => (&s.table, WriteKind::Unchanged),
        S::Delete(s) => (&s.table, WriteKind::Removed),
        S::Truncate(s) => (&s.table, WriteKind::Emptied),
        S::Merge(s) => (&s.target, WriteKind::Mixed),
        _ => return None,
    };
    if name.contains('.') {
        return None;
    }
    Some((name.as_str(), kind))
}

/// The rows a write reported, read out of the batch the operator produced.
///
/// A DML operator hands back a one-row, one-column Int64 batch whose cell
/// holds the total, so counting the batch's rows would answer one however
/// many rows the statement touched.
pub fn affected_rows(batches: &[zyron_executor::batch::DataBatch]) -> u64 {
    let mut total: i64 = 0;
    for batch in batches {
        let Some(column) = batch.columns.first() else {
            continue;
        };
        match &column.data {
            zyron_executor::column::ColumnData::Int64(values) => {
                if let Some(v) = values.first() {
                    total = total.saturating_add(*v);
                }
            }
            // A shape the operator was not expected to produce still counts
            // as the rows it handed back rather than as nothing
            _ => total = total.saturating_add(batch.num_rows as i64),
        }
    }
    total.max(0) as u64
}

/// What a write does to a table's live row count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteKind {
    /// Every affected row is a row the table did not hold
    Added,
    /// Every affected row is a row the table no longer holds
    Removed,
    /// Rows changed in place, so the count is what it was
    Unchanged,
    /// The table holds nothing afterwards
    Emptied,
    /// Rows were both added and removed and the tag reports one total, so
    /// the count is recovered by a scan rather than carried forward
    Mixed,
}

/// Records that a temporary table's rows changed, so the next read of it
/// collects statistics first.
pub fn mark_written(
    session: &Option<Session>,
    table: &str,
    kind: WriteKind,
    batches: &[zyron_executor::batch::DataBatch],
) {
    if let Some(namespace) = session.as_ref().and_then(|s| s.temp_tables.as_ref()) {
        let affected = affected_rows(batches).min(i64::MAX as u64) as i64;
        match kind {
            WriteKind::Added => namespace.record_write(table, affected, true),
            WriteKind::Removed => namespace.record_write(table, -affected, true),
            WriteKind::Unchanged => namespace.record_write(table, 0, true),
            WriteKind::Emptied => namespace.record_truncate(table),
            WriteKind::Mixed => namespace.record_write(table, 0, false),
        }
    }
}

/// Refuses every statement that would tie something durable to a temporary
/// table, or run one where it has no meaning.
///
/// One place rather than a check inside each handler, because the rule is one
/// rule: a temporary table is this session's alone and is gone when the
/// session ends, so nothing that outlives the session may name it and no
/// operation over stored versions applies to it.
pub fn refuse_statement(
    stmt: &zyron_parser::Statement,
    session: &Option<Session>,
    active_branch: &Option<String>,
) -> Result<()> {
    use zyron_parser::Statement as S;
    use zyron_parser::ast::{AbacPolicyTarget, CommentObjectType, GrantObject};

    // A branch is a version of stored data that other sessions read. A
    // temporary table has no versions and no other reader, so it has no
    // place inside one
    if let S::CreateTable(s) = stmt
        && s.temporary
        && active_branch.is_some()
    {
        return Err(ZyronError::ConfigError(format!(
            "CREATE TEMPORARY TABLE \"{}\" is refused inside a branch: a branch is a version of stored data that other sessions read, and a temporary table has no versions and no other reader",
            s.name
        )));
    }

    let Some(namespace) = session.as_ref().and_then(|s| s.temp_tables.as_ref()) else {
        return Ok(());
    };
    let holds = |name: &str| !name.contains('.') && namespace.contains(name);

    match stmt {
        S::CreateView(s) => refuse_query("a view", &s.name, &s.query, &holds),
        S::CreateMaterializedView(s) => {
            refuse_query("a materialized view", &s.name, &s.query, &holds)
        }
        S::CreateTrigger(s) if holds(&s.table) => {
            Err(reference_refusal("a trigger", &s.name, &s.table))
        }
        S::CreateAbacPolicy(s) if s.target == AbacPolicyTarget::Table && holds(&s.target_name) => {
            Err(reference_refusal(
                "a row security policy",
                &s.name,
                &s.target_name,
            ))
        }
        S::Grant(s) => match &s.object {
            GrantObject::Table(name) if holds(name) => Err(grant_refusal(name)),
            _ if holds(&s.on_table) => Err(grant_refusal(&s.on_table)),
            _ => Ok(()),
        },
        S::Revoke(s) if holds(&s.on_table) => Err(grant_refusal(&s.on_table)),
        S::CommentOn(s)
            if matches!(
                s.object_type,
                CommentObjectType::Table | CommentObjectType::Column
            ) && holds(comment_table(&s.name)) =>
        {
            Err(ZyronError::ConfigError(format!(
                "COMMENT ON is refused for the temporary table \"{}\": a comment is catalog metadata that outlives the session, and the table it would describe does not",
                comment_table(&s.name)
            )))
        }
        S::ArchiveTable(s) if holds(&s.table) => Err(refuse_operation("ARCHIVE", &s.table)),
        S::RestoreTable(s) if holds(&s.table) => Err(refuse_operation("RESTORE", &s.table)),
        // A permanent table's foreign key would point at rows only one
        // session can see, and at nothing at all once that session ends
        S::CreateTable(s) if !s.temporary => refuse_foreign_keys(&s.name, &s.constraints, &holds),
        S::AlterTable(s) => refuse_alter_foreign_key(s, &holds),
        _ => Ok(()),
    }
}

/// The table a comment names, dropping the column part of `table.column`.
fn comment_table(name: &str) -> &str {
    name.split('.').next().unwrap_or(name)
}

/// Refuses a definition whose query reads a temporary table.
fn refuse_query(
    kind: &str,
    object: &str,
    query: &zyron_parser::ast::SelectStatement,
    holds: &dyn Fn(&str) -> bool,
) -> Result<()> {
    let mut found = None;
    for item in &query.from {
        collect_bare_table_names(item, &mut |name| {
            if found.is_none() && holds(name) {
                found = Some(name.to_string());
            }
        });
    }
    match found {
        Some(table) => Err(reference_refusal(kind, object, &table)),
        None => Ok(()),
    }
}

/// Every bare relation name a FROM item reads.
fn collect_bare_table_names(item: &zyron_parser::ast::TableRef, f: &mut dyn FnMut(&str)) {
    use zyron_parser::ast::TableRef;
    match item {
        TableRef::Table { name, .. } => f(name),
        TableRef::Join(join) => {
            collect_bare_table_names(&join.left, f);
            collect_bare_table_names(&join.right, f);
        }
        TableRef::Subquery { query, .. } => {
            for inner in &query.from {
                collect_bare_table_names(inner, f);
            }
        }
        TableRef::Lateral { subquery } => collect_bare_table_names(subquery, f),
        TableRef::Pivot(p) => collect_bare_table_names(&p.input, f),
        TableRef::Unpivot(u) => collect_bare_table_names(&u.input, f),
        TableRef::TableFunction(_)
        | TableRef::ExternalInline(_)
        | TableRef::Unnest(_)
        | TableRef::Flatten(_) => {}
    }
}

/// Refuses a declared foreign key that points at a temporary table.
fn refuse_foreign_keys(
    object: &str,
    constraints: &[zyron_parser::ast::TableConstraint],
    holds: &dyn Fn(&str) -> bool,
) -> Result<()> {
    use zyron_parser::ast::TableConstraintKind;
    for constraint in constraints {
        if let TableConstraintKind::ForeignKey { ref_table, .. } = &constraint.kind
            && holds(ref_table)
        {
            return Err(reference_refusal("a foreign key on", object, ref_table));
        }
    }
    Ok(())
}

/// Refuses a foreign key added by ALTER TABLE that points at a temporary
/// table, and any ALTER against a temporary table that changes stored state
/// the session cannot own.
fn refuse_alter_foreign_key(
    stmt: &zyron_parser::ast::AlterTableStatement,
    holds: &dyn Fn(&str) -> bool,
) -> Result<()> {
    use zyron_parser::ast::{AlterTableOperation, TableConstraintKind};
    if let AlterTableOperation::AddConstraint(constraint) = &stmt.operation
        && let TableConstraintKind::ForeignKey { ref_table, .. } = &constraint.kind
        && holds(ref_table)
    {
        return Err(reference_refusal("a foreign key on", &stmt.name, ref_table));
    }
    Ok(())
}

/// The refusal a durable definition naming a temporary table produces.
fn reference_refusal(kind: &str, object: &str, table: &str) -> ZyronError {
    ZyronError::ConfigError(format!(
        "{kind} \"{object}\" would reference the temporary table \"{table}\", which only this session can see and which is dropped when the session ends; a permanent definition cannot depend on session state"
    ))
}

/// The refusal a grant on a temporary table produces.
fn grant_refusal(table: &str) -> ZyronError {
    ZyronError::ConfigError(format!(
        "a grant on the temporary table \"{table}\" is refused: a grant is recorded against a catalog object and outlives the session, and no other principal can reach a temporary table to be granted anything on it"
    ))
}

/// Refuses an object whose definition would depend on a temporary table.
///
/// A permanent definition outlives the session, so it must not name
/// something only that session can see: the object would be unreadable to
/// every other session and unresolvable after the creating one ends.
pub fn refuse_reference(
    kind: &str,
    object: &str,
    table: &str,
    session: &Option<Session>,
) -> Result<()> {
    let holds = session
        .as_ref()
        .and_then(|s| s.temp_tables.as_ref())
        .is_some_and(|t| t.contains(table));
    if !holds {
        return Ok(());
    }
    Err(ZyronError::ConfigError(format!(
        "{kind} \"{object}\" would reference the temporary table \"{table}\", which only this session can see and which is dropped when the session ends; a permanent definition cannot depend on session state"
    )))
}

/// Refuses a statement that a temporary table has no form of.
pub fn refuse_operation(operation: &str, table: &str) -> ZyronError {
    ZyronError::ConfigError(format!(
        "{operation} does not apply to the temporary table \"{table}\": it lives on this node only, is dropped when the session ends, and is never written to the catalog or the consensus log"
    ))
}

/// Clears the temporary directory a crash left behind.
///
/// Runs before the node accepts connections: the files under it belong to
/// sessions that no longer exist, and nothing will ever read them again.
pub fn clear_on_start(data_dir: &std::path::Path) -> Result<()> {
    TempTableRegistry::clear_directory_on_start(data_dir)
}

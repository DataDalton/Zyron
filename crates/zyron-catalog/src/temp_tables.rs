//! Temporary tables: a per-session namespace of tables that live on one node
//! and never reach the catalog or the consensus log.
//!
//! A temporary table is a heap table like any other, so everything that reads
//! a heap reads one of these. What differs is where its definition lives:
//! here, in the session that created it, rather than in the cluster catalog.
//! Nothing about one is persisted, replicated, or visible to another session.
//!
//! Two consequences follow from that and are enforced here.
//!
//! Ids come from a node-local allocator counting down from the top of the id
//! space. The catalog's own allocator is a function of the applied consensus
//! log, so two members number the same object identically without any id
//! travelling; taking an id from it for a table only one node has would move
//! that counter on one member and desynchronize every later object's id.
//!
//! Resolution searches a session's own namespace before the catalog, so a
//! temporary table shadows a permanent one of the same bare name for that
//! session and for no other. A qualified name never reaches here at all,
//! which is what makes the permanent table always reachable.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

use parking_lot::RwLock;
use zyron_common::{Result, ZyronError};
use zyron_parser::ast::OnCommitAction;

use crate::ids::TableId;
use crate::schema::TableEntry;

/// The first id a temporary table takes. Ids descend from here, and the
/// catalog's own allocator ascends from `USER_OID_START`, so the two spaces
/// meet only after four billion objects and the floor below refuses to cross.
pub const TEMP_OID_START: u32 = u32::MAX;

/// The lowest id a temporary object may take. Reaching it means the node has
/// created roughly a billion temporary objects since it started, and handing
/// out a lower one would eventually collide with a catalog id.
pub const TEMP_OID_FLOOR: u32 = u32::MAX - 1_000_000_000;

/// How many temporary tables one session may hold before creation is refused.
pub const DEFAULT_TEMP_TABLE_MAX_COUNT: u32 = 256;

/// The share of a node's memory budget one session's temporary tables may
/// occupy before creation is refused.
pub const DEFAULT_TEMP_TABLE_BYTES_FRACTION: f64 = 0.10;

/// Counts the sessions this node has opened, whatever transport opened them.
static NEXT_SESSION_KEY: AtomicU64 = AtomicU64::new(1);

/// What names a session's temporary namespace.
///
/// Distinct from the backend process id, which the wire protocol sizes at
/// 32 bits and reuses when its counter wraps. Reuse is harmless for a cancel
/// key and is not harmless here: the namespace a key names holds a session's
/// tables and their directory, so two sessions sharing a key would share
/// both. A key is taken once per session and never again on this node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SessionKey(u64);

impl SessionKey {
    /// Takes the next key. Every session takes one when it is built, so no
    /// caller has to know this exists to get a namespace of its own.
    pub fn next() -> Self {
        Self(NEXT_SESSION_KEY.fetch_add(1, Ordering::Relaxed))
    }

    /// The key as it names the session's directory and appears in
    /// `zyron_sys.stat.temp_tables`.
    pub fn get(&self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for SessionKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// What a session's temporary tables cost and how many there are, as
/// `zyron_sys.stat.temp_tables` reports them.
#[derive(Debug, Clone)]
pub struct TempTableInfo {
    /// Names the session's directory under `<data_dir>/tmp/`
    pub session_key: SessionKey,
    /// The backend process id, so a row here joins the one for the same
    /// session in `zyron_sys.stat.sessions`
    pub session_id: i32,
    pub name: String,
    pub table_id: TableId,
    /// Bytes the table's heap and index files hold, as of the last collection
    pub bytes: u64,
    /// Live rows, as of the last collection
    pub rows: u64,
    pub on_commit: OnCommitAction,
    /// True when rows changed since statistics were last collected, so the
    /// figures above are behind the table
    pub stale: bool,
}

/// One temporary table.
pub struct TempTable {
    pub entry: Arc<TableEntry>,
    pub on_commit: OnCommitAction,
    /// The directory holding this table's heap, free space map and index
    /// files
    pub directory: std::path::PathBuf,
    /// Set by every write, cleared when statistics are collected, which is
    /// what makes collection happen on the first read after a write
    dirty: AtomicBool,
    /// Live rows, carried forward by each write rather than counted by a
    /// scan. The session that owns the table is its only writer, so a delta
    /// applied per statement is exact and a read costs no pass over the heap
    rows: AtomicU64,
    /// Set when a statement changed the row count by an amount its tag does
    /// not separate into rows added and rows removed, which is MERGE. The
    /// count is recovered by one scan on the next read
    rows_unknown: AtomicBool,
    bytes: AtomicU64,
}

impl TempTable {
    /// Records that rows changed, so the next read collects statistics.
    pub fn mark_written(&self) {
        self.dirty.store(true, Ordering::Relaxed);
    }

    /// True when rows changed since statistics were last collected.
    pub fn needs_statistics(&self) -> bool {
        self.dirty.load(Ordering::Relaxed)
    }

    /// Carries the live row count forward by what a statement added or
    /// removed.
    ///
    /// Saturating, so a delta that disagrees with the count leaves a floor of
    /// zero rather than wrapping into a cardinality the planner would trust.
    pub fn add_rows(&self, delta: i64) {
        if delta == 0 {
            return;
        }
        let _ = self
            .rows
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |rows| {
                Some(if delta >= 0 {
                    rows.saturating_add(delta as u64)
                } else {
                    rows.saturating_sub(delta.unsigned_abs())
                })
            });
    }

    /// Sets the live row count outright, for a statement that empties the
    /// table and for the scan that recovers a count.
    pub fn set_rows(&self, rows: u64) {
        self.rows.store(rows, Ordering::Relaxed);
        self.rows_unknown.store(false, Ordering::Relaxed);
    }

    /// Records that the row count can no longer be carried forward, so the
    /// next collection counts the rows instead.
    pub fn require_row_scan(&self) {
        self.rows_unknown.store(true, Ordering::Relaxed);
    }

    /// True when the carried count is not trustworthy and a scan is owed.
    pub fn needs_row_scan(&self) -> bool {
        self.rows_unknown.load(Ordering::Relaxed)
    }

    /// Records what a collection measured.
    pub fn record_statistics(&self, rows: u64, bytes: u64) {
        self.rows.store(rows, Ordering::Relaxed);
        self.bytes.store(bytes, Ordering::Relaxed);
        self.rows_unknown.store(false, Ordering::Relaxed);
        self.dirty.store(false, Ordering::Relaxed);
    }

    pub fn rows(&self) -> u64 {
        self.rows.load(Ordering::Relaxed)
    }

    pub fn bytes(&self) -> u64 {
        self.bytes.load(Ordering::Relaxed)
    }
}

/// One session's temporary tables, keyed by their bare names.
pub struct SessionTempTables {
    session_key: SessionKey,
    session_id: i32,
    directory: std::path::PathBuf,
    /// Lowercased name to table, because a bare name resolves case
    /// insensitively the way every other relation name does
    tables: RwLock<HashMap<String, Arc<TempTable>>>,
    /// How many tables owe a statistics collection. Read before every
    /// statement, so the common case of nothing written since the last one
    /// costs a single load rather than a lock and a walk of the namespace
    dirty: AtomicU32,
    max_count: AtomicU32,
    max_bytes: AtomicU64,
}

impl SessionTempTables {
    fn new(
        session_key: SessionKey,
        session_id: i32,
        directory: std::path::PathBuf,
        max_bytes: u64,
    ) -> Self {
        Self {
            session_key,
            session_id,
            directory,
            tables: RwLock::new(HashMap::new()),
            dirty: AtomicU32::new(0),
            max_count: AtomicU32::new(DEFAULT_TEMP_TABLE_MAX_COUNT),
            max_bytes: AtomicU64::new(max_bytes),
        }
    }

    /// The key naming this namespace and its directory.
    pub fn session_key(&self) -> SessionKey {
        self.session_key
    }

    /// The backend process id of the session holding this namespace.
    pub fn session_id(&self) -> i32 {
        self.session_id
    }

    /// The directory this session's temporary files live in.
    pub fn directory(&self) -> &std::path::Path {
        &self.directory
    }

    /// Raises or lowers this session's limits, from the cascading config.
    pub fn set_limits(&self, max_bytes: u64, max_count: u32) {
        self.max_bytes.store(max_bytes, Ordering::Relaxed);
        self.max_count.store(max_count, Ordering::Relaxed);
    }

    pub fn max_bytes(&self) -> u64 {
        self.max_bytes.load(Ordering::Relaxed)
    }

    pub fn max_count(&self) -> u32 {
        self.max_count.load(Ordering::Relaxed)
    }

    /// The table a bare name resolves to in this session, or None.
    pub fn resolve(&self, name: &str) -> Option<Arc<TableEntry>> {
        self.with_entry(name, |t| Arc::clone(&t.entry))
    }

    /// The temporary table a bare name resolves to, with its lifecycle state.
    pub fn get(&self, name: &str) -> Option<Arc<TempTable>> {
        self.with_entry(name, Arc::clone)
    }

    /// True when this session holds a temporary table of this bare name.
    pub fn contains(&self, name: &str) -> bool {
        self.with_entry(name, |_| ()).is_some()
    }

    /// Looks a bare name up in the namespace and reads what it found.
    ///
    /// The map is keyed by the lowercased name, and a name that is already
    /// lowercase probes it borrowed. Only a name carrying an uppercase byte
    /// pays the fold, which keeps the resolver's per-reference cost at a
    /// hash of bytes already in hand rather than a heap allocation.
    fn with_entry<R>(&self, name: &str, read: impl FnOnce(&Arc<TempTable>) -> R) -> Option<R> {
        let tables = self.tables.read();
        if let Some(hit) = tables.get(name) {
            return Some(read(hit));
        }
        if name.bytes().any(|b| b.is_ascii_uppercase()) {
            return tables.get(&name.to_ascii_lowercase()).map(read);
        }
        None
    }

    pub fn is_empty(&self) -> bool {
        self.tables.read().is_empty()
    }

    pub fn len(&self) -> usize {
        self.tables.read().len()
    }

    /// Refuses a further table when a limit is already reached, naming the
    /// limit that refused it.
    pub fn check_limits(&self) -> Result<()> {
        // One acquisition for both figures, so the count and the byte total
        // describe the same namespace rather than two moments of it
        let (count, held) = {
            let tables = self.tables.read();
            (
                tables.len() as u32,
                tables.values().map(|t| t.bytes()).sum::<u64>(),
            )
        };
        let max_count = self.max_count();
        if count >= max_count {
            return Err(ZyronError::ConfigError(format!(
                "this session already holds {count} temporary tables, which is the temp_table_max_count limit of {max_count}; drop one before creating another or raise temp_table_max_count"
            )));
        }
        let max_bytes = self.max_bytes();
        if max_bytes > 0 && held >= max_bytes {
            return Err(ZyronError::ConfigError(format!(
                "this session's temporary tables hold {held} bytes, which is the temp_table_max_bytes limit of {max_bytes}; drop one before creating another or raise temp_table_max_bytes"
            )));
        }
        Ok(())
    }

    /// True when at least one table owes a statistics collection.
    ///
    /// Every statement asks this before planning, so it is one atomic load
    /// and no lock for a session whose tables have not changed.
    pub fn any_dirty(&self) -> bool {
        self.dirty.load(Ordering::Relaxed) != 0
    }

    /// The tables owing a collection, gathered under one read lock.
    ///
    /// Only reached when `any_dirty` says there is work, so the allocation
    /// is paid by a statement that follows a write rather than by every
    /// statement.
    ///
    /// The count is resynchronized to what the walk actually found. Dropping
    /// a table that owed a collection leaves the counter above the truth, and
    /// this is what brings it back down, so no removal path has to adjust it.
    pub fn dirty_tables(&self) -> Vec<Arc<TempTable>> {
        let out: Vec<Arc<TempTable>> = self
            .tables
            .read()
            .values()
            .filter(|t| t.needs_statistics())
            .map(Arc::clone)
            .collect();
        self.dirty.store(out.len() as u32, Ordering::Relaxed);
        out
    }

    /// Records what a write changed in one table, by name.
    ///
    /// `delta` is the rows the statement added, negative for rows it removed.
    /// `exact` is false for a statement whose tag does not separate the two,
    /// which leaves the count to be recovered by a scan.
    pub fn record_write(&self, name: &str, delta: i64, exact: bool) {
        let Some(table) = self.get(name) else {
            return;
        };
        if exact {
            table.add_rows(delta);
        } else {
            table.require_row_scan();
        }
        if !table.needs_statistics() {
            table.mark_written();
            self.dirty.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Empties a table's row count, for a statement that removes every row.
    pub fn record_truncate(&self, name: &str) {
        let Some(table) = self.get(name) else {
            return;
        };
        table.set_rows(0);
        if !table.needs_statistics() {
            table.mark_written();
            self.dirty.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Records a collection against one table and clears what it owed.
    pub fn record_statistics(&self, table: &TempTable, rows: u64, bytes: u64) {
        if table.needs_statistics() {
            let _ = self
                .dirty
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |d| {
                    Some(d.saturating_sub(1))
                });
        }
        table.record_statistics(rows, bytes);
    }

    /// Adds a table to this session's namespace, replacing one of the same
    /// name. Returns the table it replaced, whose files the caller unlinks.
    pub fn insert(
        &self,
        entry: Arc<TableEntry>,
        on_commit: OnCommitAction,
    ) -> Option<Arc<TempTable>> {
        let table = Arc::new(TempTable {
            entry: Arc::clone(&entry),
            on_commit,
            directory: self.directory.clone(),
            dirty: AtomicBool::new(true),
            rows: AtomicU64::new(0),
            rows_unknown: AtomicBool::new(false),
            bytes: AtomicU64::new(0),
        });
        // A new table owes its first collection, so it counts against the
        // dirty total the same way a written one does
        self.dirty.fetch_add(1, Ordering::Relaxed);
        self.tables
            .write()
            .insert(entry.name.to_ascii_lowercase(), table)
    }

    /// Removes a table from this session's namespace.
    pub fn remove(&self, name: &str) -> Option<Arc<TempTable>> {
        self.tables.write().remove(&name.to_ascii_lowercase())
    }

    /// Every table this session holds.
    pub fn list(&self) -> Vec<Arc<TempTable>> {
        let mut out: Vec<Arc<TempTable>> = self.tables.read().values().map(Arc::clone).collect();
        out.sort_by(|a, b| a.entry.name.cmp(&b.entry.name));
        out
    }

    /// Empties the namespace and hands back what it held, for a session that
    /// is ending.
    pub fn take_all(&self) -> Vec<Arc<TempTable>> {
        let mut held = self.tables.write();
        let mut out: Vec<Arc<TempTable>> = held.values().map(Arc::clone).collect();
        held.clear();
        out.sort_by(|a, b| a.entry.name.cmp(&b.entry.name));
        out
    }

    /// The tables a commit acts on: those to empty and those to drop.
    ///
    /// A table declared ON COMMIT DROP is removed from the namespace here, so
    /// the caller unlinks its files and nothing resolves to it afterwards.
    pub fn on_commit(&self) -> (Vec<Arc<TempTable>>, Vec<Arc<TempTable>>) {
        let mut held = self.tables.write();
        let mut truncate = Vec::new();
        let mut dropped = Vec::new();
        held.retain(|_, table| match table.on_commit {
            OnCommitAction::PreserveRows => true,
            OnCommitAction::DeleteRows => {
                truncate.push(Arc::clone(table));
                true
            }
            OnCommitAction::Drop => {
                dropped.push(Arc::clone(table));
                false
            }
        });
        truncate.sort_by(|a, b| a.entry.name.cmp(&b.entry.name));
        dropped.sort_by(|a, b| a.entry.name.cmp(&b.entry.name));
        (truncate, dropped)
    }
}

/// Every temporary table on this node, so an id resolves and an operator can
/// see what sessions are holding.
pub struct TempTableRegistry {
    /// The directory every session's own directory sits under
    root: RwLock<std::path::PathBuf>,
    sessions: RwLock<HashMap<SessionKey, Arc<SessionTempTables>>>,
    /// Table id to table, across every session, so `get_table_by_id`
    /// resolves without knowing which session owns it. A session only ever
    /// learns the id of its own table, because only its own namespace is
    /// searched by name
    by_id: RwLock<HashMap<u32, Arc<TableEntry>>>,
    next_oid: AtomicU32,
    /// What one session's tables may hold, from the node's memory budget
    default_max_bytes: AtomicU64,
}

impl Default for TempTableRegistry {
    fn default() -> Self {
        Self::new(std::path::PathBuf::new(), 0)
    }
}

impl TempTableRegistry {
    /// Creates the registry over `<data_dir>/tmp`.
    pub fn new(data_dir: std::path::PathBuf, node_memory_bytes: u64) -> Self {
        let default_max_bytes =
            (node_memory_bytes as f64 * DEFAULT_TEMP_TABLE_BYTES_FRACTION) as u64;
        Self {
            root: RwLock::new(temp_root(&data_dir)),
            sessions: RwLock::new(HashMap::new()),
            by_id: RwLock::new(HashMap::new()),
            next_oid: AtomicU32::new(TEMP_OID_START),
            default_max_bytes: AtomicU64::new(default_max_bytes),
        }
    }

    /// The directory every session's temporary files sit under.
    pub fn root(&self) -> std::path::PathBuf {
        self.root.read().clone()
    }

    /// Points the registry at a data directory, for a node whose directory is
    /// settled after the registry is built.
    pub fn set_data_dir(&self, data_dir: &std::path::Path) {
        let root = temp_root(data_dir);
        // The node's data directory is fixed at startup, so every session
        // after the first sets the value it already holds. Reading first
        // keeps concurrent first-creates off a node-wide write lock
        if *self.root.read() == root {
            return;
        }
        *self.root.write() = root;
    }

    /// Sets what one session's tables may hold by default.
    pub fn set_default_max_bytes(&self, bytes: u64) {
        self.default_max_bytes.store(bytes, Ordering::Relaxed);
    }

    /// Allocates the next node-local id for a temporary object.
    pub fn next_oid(&self) -> Result<u32> {
        let id = self.next_oid.fetch_sub(1, Ordering::Relaxed);
        if id <= TEMP_OID_FLOOR {
            return Err(ZyronError::Internal(
                "this node has exhausted the temporary object id space; restart it to reclaim the range".to_string(),
            ));
        }
        Ok(id)
    }

    /// The session's namespace, created on first use.
    ///
    /// A key is taken once per session, so an entry already under one belongs
    /// to the same session asking a second time and is handed back. The
    /// process id travels alongside for the operator view and never selects
    /// the namespace.
    pub fn session(&self, session_key: SessionKey, session_id: i32) -> Arc<SessionTempTables> {
        if let Some(existing) = self.sessions.read().get(&session_key) {
            return Arc::clone(existing);
        }
        let mut sessions = self.sessions.write();
        // Another caller may have created it while the read lock was released
        if let Some(existing) = sessions.get(&session_key) {
            return Arc::clone(existing);
        }
        let directory = self.root.read().join(session_key.to_string());
        let created = Arc::new(SessionTempTables::new(
            session_key,
            session_id,
            directory,
            self.default_max_bytes.load(Ordering::Relaxed),
        ));
        sessions.insert(session_key, Arc::clone(&created));
        created
    }

    /// The session's namespace when it has one, without creating it.
    pub fn existing_session(&self, session_key: SessionKey) -> Option<Arc<SessionTempTables>> {
        self.sessions.read().get(&session_key).map(Arc::clone)
    }

    /// Records a table so its id resolves.
    pub fn register(&self, entry: Arc<TableEntry>) {
        self.by_id.write().insert(entry.id.0, entry);
    }

    /// Forgets a table's id.
    pub fn forget(&self, table_id: TableId) {
        self.by_id.write().remove(&table_id.0);
    }

    /// The temporary table an id names, or None when the id is not one.
    pub fn by_id(&self, table_id: TableId) -> Option<Arc<TableEntry>> {
        // An id below the floor cannot be a temporary one, so the common
        // case costs a comparison rather than a lock
        if table_id.0 <= TEMP_OID_FLOOR {
            return None;
        }
        self.by_id.read().get(&table_id.0).map(Arc::clone)
    }

    /// Removes a session's namespace and hands back what it held, for a
    /// connection that has ended.
    pub fn end_session(&self, session_key: SessionKey) -> Vec<Arc<TempTable>> {
        let Some(session) = self.sessions.write().remove(&session_key) else {
            return Vec::new();
        };
        let held = session.take_all();
        let mut by_id = self.by_id.write();
        for table in &held {
            by_id.remove(&table.entry.id.0);
        }
        held
    }

    /// What every session is holding, for `zyron_sys.stat.temp_tables`.
    pub fn report(&self) -> Vec<TempTableInfo> {
        let sessions: Vec<Arc<SessionTempTables>> =
            self.sessions.read().values().map(Arc::clone).collect();
        let mut out = Vec::new();
        for session in sessions {
            for table in session.list() {
                out.push(TempTableInfo {
                    session_key: session.session_key(),
                    session_id: session.session_id(),
                    name: table.entry.name.clone(),
                    table_id: table.entry.id,
                    bytes: table.bytes(),
                    rows: table.rows(),
                    on_commit: table.on_commit,
                    stale: table.needs_statistics(),
                });
            }
        }
        out.sort_by(|a, b| (a.session_key, &a.name).cmp(&(b.session_key, &b.name)));
        out
    }

    /// True when any session holds a temporary table, which is what pins a
    /// session to the node it is running on.
    pub fn any_held(&self) -> bool {
        self.sessions.read().values().any(|s| !s.is_empty())
    }

    /// Removes the whole temporary directory, for a node that is starting.
    ///
    /// A crash leaves a session's files behind with no session to own them,
    /// and nothing else will ever read them, so a start clears the tree
    /// before it accepts a connection rather than leaving it to accumulate.
    pub fn clear_directory_on_start(data_dir: &std::path::Path) -> Result<()> {
        let root = temp_root(data_dir);
        match std::fs::remove_dir_all(&root) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(ZyronError::IoError(format!(
                "clearing the temporary table directory {}: {e}",
                root.display()
            ))),
        }
    }
}

/// The directory a node's temporary tables live under.
pub fn temp_root(data_dir: &std::path::Path) -> std::path::PathBuf {
    data_dir.join("tmp")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: u32, name: &str) -> Arc<TableEntry> {
        Arc::new(TableEntry {
            id: TableId(id),
            schema_id: crate::ids::SchemaId(1),
            name: name.to_string(),
            // The real allocator hands out descending ids, so the file ids
            // sit below the table id rather than above it
            heap_file_id: id.wrapping_sub(1),
            fsm_file_id: id.wrapping_sub(2),
            columns: Vec::new(),
            constraints: Vec::new(),
            created_at: 0,
            versioning_enabled: false,
            scd_type: None,
            system_versioned: false,
            history_table_id: None,
            cdf_enabled: false,
            cdf_retention_days: 0,
            lifecycle: Default::default(),
            columnar: Default::default(),
            dropped_at: None,
            expectations: Vec::new(),
            time_travel_retention_secs: 0,
            lake: Default::default(),
            cluster: Default::default(),
            foreign: Default::default(),
            schema_epoch: 0,
            schema_epochs: Vec::new(),
            pre_stamp_columns: Vec::new(),
            cdf: Default::default(),
        })
    }

    #[test]
    fn ids_descend_from_the_top_and_never_reach_the_catalog_range() {
        let registry = TempTableRegistry::new(std::path::PathBuf::from("."), 0);
        let first = registry.next_oid().expect("allocates");
        let second = registry.next_oid().expect("allocates");
        assert_eq!(first, TEMP_OID_START);
        assert_eq!(second, TEMP_OID_START - 1);
        assert!(
            second > crate::ids::USER_OID_START,
            "a temporary id must never fall into the catalog's own range"
        );
    }

    #[test]
    fn a_name_resolves_only_inside_the_session_that_created_it() {
        let registry = TempTableRegistry::new(std::path::PathBuf::from("."), 1024);
        let one = registry.session(SessionKey::next(), 1);
        let two = registry.session(SessionKey::next(), 2);
        one.insert(
            entry(TEMP_OID_START, "scratch"),
            OnCommitAction::PreserveRows,
        );
        assert!(one.resolve("scratch").is_some());
        assert!(
            two.resolve("scratch").is_none(),
            "another session sees nothing of it"
        );
        // The same bare name is free in the other session
        two.insert(
            entry(TEMP_OID_START - 1, "scratch"),
            OnCommitAction::PreserveRows,
        );
        assert_ne!(
            one.resolve("scratch").map(|e| e.id),
            two.resolve("scratch").map(|e| e.id)
        );
    }

    #[test]
    fn a_name_resolves_case_insensitively() {
        let registry = TempTableRegistry::new(std::path::PathBuf::from("."), 1024);
        let session = registry.session(SessionKey::next(), 1);
        session.insert(
            entry(TEMP_OID_START, "Scratch"),
            OnCommitAction::PreserveRows,
        );
        assert!(session.resolve("scratch").is_some());
        assert!(session.resolve("SCRATCH").is_some());
    }

    #[test]
    fn an_id_resolves_across_sessions_and_a_catalog_id_never_does() {
        let registry = TempTableRegistry::new(std::path::PathBuf::from("."), 1024);
        let session = registry.session(SessionKey::next(), 7);
        let e = entry(TEMP_OID_START, "scratch");
        session.insert(Arc::clone(&e), OnCommitAction::PreserveRows);
        registry.register(e);
        assert!(registry.by_id(TableId(TEMP_OID_START)).is_some());
        assert!(
            registry.by_id(TableId(10_001)).is_none(),
            "a catalog id is answered by the catalog, never here"
        );
    }

    #[test]
    fn a_commit_empties_delete_rows_and_drops_drop() {
        let registry = TempTableRegistry::new(std::path::PathBuf::from("."), 1024);
        let session = registry.session(SessionKey::next(), 1);
        session.insert(entry(TEMP_OID_START, "keep"), OnCommitAction::PreserveRows);
        session.insert(
            entry(TEMP_OID_START - 2, "empty"),
            OnCommitAction::DeleteRows,
        );
        session.insert(entry(TEMP_OID_START - 4, "gone"), OnCommitAction::Drop);

        let (truncate, dropped) = session.on_commit();
        assert_eq!(truncate.len(), 1);
        assert_eq!(truncate[0].entry.name, "empty");
        assert_eq!(dropped.len(), 1);
        assert_eq!(dropped[0].entry.name, "gone");
        assert!(session.resolve("gone").is_none(), "the drop takes effect");
        assert!(session.resolve("empty").is_some(), "the definition stays");
        assert!(session.resolve("keep").is_some());
    }

    #[test]
    fn a_count_limit_refuses_a_further_table_naming_itself() {
        let registry = TempTableRegistry::new(std::path::PathBuf::from("."), 1024);
        let session = registry.session(SessionKey::next(), 1);
        session.set_limits(u64::MAX, 1);
        session.insert(entry(TEMP_OID_START, "one"), OnCommitAction::PreserveRows);
        let error = session.check_limits().expect_err("refused");
        let text = error.to_string();
        assert!(
            text.contains("temp_table_max_count"),
            "the refusal names the limit, got {text}"
        );
    }

    #[test]
    fn a_byte_limit_refuses_a_further_table_naming_itself() {
        let registry = TempTableRegistry::new(std::path::PathBuf::from("."), 1024);
        let session = registry.session(SessionKey::next(), 1);
        session.set_limits(100, 256);
        session.insert(entry(TEMP_OID_START, "one"), OnCommitAction::PreserveRows);
        session.get("one").expect("held").record_statistics(10, 500);
        let error = session.check_limits().expect_err("refused");
        let text = error.to_string();
        assert!(
            text.contains("temp_table_max_bytes"),
            "the refusal names the limit, got {text}"
        );
    }

    #[test]
    fn statistics_are_stale_after_a_write_and_current_after_a_collection() {
        let registry = TempTableRegistry::new(std::path::PathBuf::from("."), 1024);
        let session = registry.session(SessionKey::next(), 1);
        session.insert(entry(TEMP_OID_START, "one"), OnCommitAction::PreserveRows);
        let table = session.get("one").expect("held");
        assert!(
            table.needs_statistics(),
            "a table with no statistics needs them collected"
        );
        table.record_statistics(42, 8192);
        assert!(!table.needs_statistics());
        assert_eq!(table.rows(), 42);
        assert_eq!(table.bytes(), 8192);
        table.mark_written();
        assert!(table.needs_statistics(), "a write makes them stale again");
    }

    #[test]
    fn ending_a_session_hands_back_everything_and_forgets_the_ids() {
        let registry = TempTableRegistry::new(std::path::PathBuf::from("."), 1024);
        let key = SessionKey::next();
        let session = registry.session(key, 3);
        let e = entry(TEMP_OID_START, "scratch");
        session.insert(Arc::clone(&e), OnCommitAction::PreserveRows);
        registry.register(e);
        assert!(registry.any_held());

        let held = registry.end_session(key);
        assert_eq!(held.len(), 1);
        assert!(registry.by_id(TableId(TEMP_OID_START)).is_none());
        assert!(!registry.any_held());
        assert!(registry.existing_session(key).is_none());
    }

    #[test]
    fn each_session_gets_its_own_directory_under_the_node_tree() {
        let registry = TempTableRegistry::new(std::path::PathBuf::from("/data"), 1024);
        let first = SessionKey::next();
        let second = SessionKey::next();
        let one = registry.session(first, 11);
        let two = registry.session(second, 12);
        assert_ne!(first, second, "each session takes a key of its own");
        assert!(one.directory().ends_with(first.to_string()));
        assert!(two.directory().ends_with(second.to_string()));
        assert_eq!(one.directory().parent(), two.directory().parent());
        assert_eq!(registry.root(), temp_root(std::path::Path::new("/data")));
    }
}

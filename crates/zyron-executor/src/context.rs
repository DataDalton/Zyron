//! Execution context providing access to storage, catalog, and transaction state.
//!
//! Each query execution receives an ExecutionContext that holds references to
//! shared infrastructure (buffer pool, WAL, catalog) along with per-query
//! state (transaction ID, MVCC snapshot, batch size). Also provides query
//! cancellation via an atomic flag and optional per-operator metrics
//! collection for EXPLAIN ANALYZE.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use zyron_buffer::BufferPool;
use zyron_catalog::{Catalog, IndexId, TableEntry, TableId, TableIndexSnapshot};
use zyron_common::{Result, ZyronError};
use zyron_storage::{BTreeIndex, DiskManager, HeapFile, HeapFileConfig, Snapshot, TupleId};
use zyron_wal::WalWriter;

use crate::batch::BATCH_SIZE;
use crate::column::ScalarValue;

/// Hook for Change Data Capture. Implemented by zyron-cdc, called by DML operators.
pub trait CdcHook: Send + Sync {
    /// Called after rows are inserted.
    fn on_insert(
        &self,
        table_id: u32,
        tuples: &[&[u8]],
        version: u64,
        timestamp: i64,
        txn_id: u64,
        is_last_in_txn: bool,
    ) -> zyron_common::Result<()>;

    /// Called after rows are deleted. old_data contains pre-delete tuple bytes.
    fn on_delete(
        &self,
        table_id: u32,
        old_data: &[&[u8]],
        version: u64,
        timestamp: i64,
        txn_id: u64,
        is_last_in_txn: bool,
    ) -> zyron_common::Result<()>;

    /// Called after rows are updated. old_data/new_data contain pre/post tuple bytes.
    fn on_update(
        &self,
        table_id: u32,
        old_data: &[&[u8]],
        new_data: &[&[u8]],
        version: u64,
        timestamp: i64,
        txn_id: u64,
        is_last_in_txn: bool,
    ) -> zyron_common::Result<()>;
}

/// Hook for BEFORE triggers. Called before DML mutations to allow
/// trigger logic to inspect, modify, or cancel the operation.
pub trait DmlHook: Send + Sync {
    /// Called before rows are inserted. Returns false to cancel the insert.
    fn before_insert(
        &self,
        table_id: u32,
        tuples: &[&[u8]],
        txn_id: u64,
    ) -> zyron_common::Result<bool>;

    /// Called before rows are deleted. Returns false to cancel the delete.
    fn before_delete(
        &self,
        table_id: u32,
        old_data: &[&[u8]],
        txn_id: u64,
    ) -> zyron_common::Result<bool>;

    /// Called before rows are updated. Returns false to cancel the update.
    fn before_update(
        &self,
        table_id: u32,
        old_data: &[&[u8]],
        new_data: &[&[u8]],
        txn_id: u64,
    ) -> zyron_common::Result<bool>;
}

/// Byte budget one query's materializing operators share. Sorts, hash
/// joins, aggregations, window buffers, set operation stores, and the
/// final result collection all reserve against it, so a query that would
/// exhaust server memory fails itself with a clear error instead.
#[derive(Debug)]
pub struct QueryMemoryBudget {
    limit: std::sync::atomic::AtomicU64,
    used: std::sync::atomic::AtomicU64,
}

impl QueryMemoryBudget {
    pub fn new(limit_bytes: u64) -> Arc<Self> {
        Arc::new(Self {
            limit: std::sync::atomic::AtomicU64::new(limit_bytes),
            used: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Reserves bytes against the budget, failing the query loudly once
    /// the limit would be exceeded. A reservation lasts for the query
    /// unless the operator that made it gives it back with `release`.
    pub fn reserve(&self, bytes: u64) -> Result<()> {
        let limit = self.limit.load(Ordering::Relaxed);
        let prev = self.used.fetch_add(bytes, Ordering::Relaxed);
        if prev.saturating_add(bytes) > limit {
            self.used.fetch_sub(bytes, Ordering::Relaxed);
            return Err(ZyronError::ExecutionError(format!(
                "query exceeds its memory budget of {} bytes, raise query.max_memory_bytes or narrow the query",
                limit
            )));
        }
        Ok(())
    }

    /// Gives reserved bytes back, for an operator that held rows and then
    /// discarded them. The caller releases no more than it reserved.
    ///
    /// A sort under a limit cuts its buffer back to the limit as it reads,
    /// and the rows it drops are no longer the query's to hold. Every other
    /// reservation stands for the query's lifetime
    pub fn release(&self, bytes: u64) {
        self.used.fetch_sub(bytes, Ordering::Relaxed);
    }

    /// Bytes reserved so far.
    pub fn used(&self) -> u64 {
        self.used.load(Ordering::Relaxed)
    }

    /// What this query may hold at once.
    ///
    /// Read by the materializing operators to decide when to spill, which is
    /// the same number that used to decide when to fail.
    pub fn limit(&self) -> u64 {
        self.limit.load(Ordering::Relaxed)
    }
}

/// Gives the node back what this context charged it.
///
/// A query that errors, is cancelled, or panics releases the same way one that
/// finished does. Without this the node would believe it was full of work that
/// ended hours ago and would refuse everything.
impl Drop for ExecutionContext {
    fn drop(&mut self) {
        let held = self.node_memory_held.swap(0, Ordering::Relaxed);
        if held > 0 {
            zyron_pressure::pressure_control::PressureController::global()
                .memory()
                .release(held);
        }
    }
}

/// Per-query execution context with access to storage and transaction state.
pub struct ExecutionContext {
    pub catalog: Arc<Catalog>,
    pub wal: Arc<WalWriter>,
    pub buffer_pool: Arc<BufferPool>,
    pub disk_manager: Arc<DiskManager>,
    pub batch_size: usize,
    pub txn_id: u64,
    /// Transaction id lake commits run under, when it differs from
    /// `txn_id`. Set for a `BEGIN ZYRONLAKE TRANSACTION`, whose lake writes
    /// commit through a cross-table intent rather than the database commit
    /// record, so their pending versions are keyed by the intent instead.
    pub lake_txn_id: Option<u64>,
    pub snapshot: Snapshot,
    /// When set, operators bail with a cancellation error at their next
    /// batch boundary. Shared into child contexts, so cancelling a statement
    /// also stops its trigger bodies, correlated subqueries, and LATERAL
    /// inner plans.
    cancelled: Arc<AtomicBool>,
    /// Wall-clock instant past which the statement is treated as timed out.
    /// check_cancelled reports cancelled once Instant::now passes this. None
    /// disables the deadline. Set from the session statement_timeout by wire.
    deadline: Option<std::time::Instant>,
    /// Upper bound on rows the top-level execute loop materializes. None
    /// disables the cap. Set from the session max_result_rows by wire so a
    /// runaway query is bounded before its full result set lands in memory.
    pub max_result_rows: Option<u64>,
    /// Byte budget shared by every materializing operator of this query.
    /// None disables the budget. Set from query.max_memory_bytes by wire so
    /// one sort, hash join, aggregation, or result set cannot take the
    /// whole server down instead of failing its own query.
    pub memory_budget: Option<Arc<QueryMemoryBudget>>,
    /// Where materializing operators put what does not fit. None means they
    /// have nowhere to go and fail at the budget the way they used to
    pub spill: Option<Arc<crate::spill::SpillDirectory>>,
    /// Bytes charged to the node gauge by this context, released when it drops
    node_memory_held: AtomicU64,
    /// Set by DML operators when they append a WAL data record. The server
    /// reads it after execution to decide whether the transaction must commit
    /// durably; a transaction that wrote nothing commits without a WAL commit
    /// record or a flush wait.
    wrote_wal: AtomicBool,
    /// Paths this statement reads out of variant columns, collected from the
    /// plan before the operator tree is built.
    ///
    /// A columnar scan takes the ones naming its own table and reads those
    /// promoted paths out of the segment columns holding them, leaving the
    /// promoted paths nothing asked for on disk. Set per plan, so a nested
    /// plan replaces it with its own and a scan that already captured its
    /// share is unaffected
    variant_paths: std::sync::RwLock<Arc<[zyron_planner::physical::variant_paths::VariantPath]>>,
    /// When true, operators collect per-operator metrics (rows, timing).
    pub analyze: bool,
    /// Optional CDC hook invoked by DML operators after mutations.
    pub cdc_hook: Option<Arc<dyn CdcHook>>,
    /// Where this transaction's effects accumulate for the consensus group.
    /// None on a node that leads no group, which is what makes replication an
    /// addition to the write path rather than a second write path
    pub replication: Option<Arc<crate::replication::TxnChangeset>>,
    /// True while committed changes from another node are being put back.
    ///
    /// The leader already ran the constraints, the foreign keys, the checks
    /// and the triggers, and re-running them here would fire side effects a
    /// second time and could reject a row the group has already agreed on. A
    /// `CHECK` over a volatile function is the clearest case: it can answer
    /// differently here and there is no answer but the leader's.
    ///
    /// Index maintenance, the WAL and the change feed all stay on: those
    /// describe what happened rather than deciding it
    pub replication_apply: bool,
    /// Optional DML hook invoked by DML operators before mutations (BEFORE triggers).
    pub dml_hook: Option<Arc<dyn DmlHook>>,
    /// Bound parameter values ($1, $2, ...) for prepared statements.
    pub params: Vec<ScalarValue>,
    /// Database the firing statement was planned against. A nested plan
    /// built during execution, a trigger body above all, plans against this
    /// database with the system default search path: stored bodies must
    /// qualify user tables, so a body resolves the same objects for every
    /// caller. Wire sets it from the session.
    pub planning_database: zyron_catalog::DatabaseId,
    /// Per-session security context for privilege checks. None when the auth
    /// system is not configured or for internal queries that bypass auth.
    /// Held behind an Arc so a nested execution (a correlated subquery or a
    /// LATERAL inner plan) shares the same clearance and masking policy as the
    /// enclosing query rather than running unsecured.
    pub security_context: Option<Arc<zyron_auth::SecurityContext>>,
    /// Live B+ tree index instances keyed by IndexId. Registered by the
    /// server layer so the index scan operator can perform actual tree lookups.
    ///
    /// Shared with every child context rather than copied into it. A
    /// correlated subquery builds one child per outer row, and an owned map
    /// made each of those rebuild the table and copy every entry
    indexes: Arc<HashMap<IndexId, Arc<BTreeIndex>>>,
    /// Live full-text search index instances keyed by IndexId. Registered by
    /// the server layer after creating or loading fulltext indexes. Shared
    /// with child contexts for the same reason as `indexes`
    fts_indexes: Arc<HashMap<IndexId, Arc<zyron_search::InvertedIndex>>>,
    /// FTS manager reference for DML index maintenance. DML operators use this
    /// to look up which FTS indexes exist for a table and update them.
    pub fts_manager: Option<Arc<zyron_search::FtsManager>>,
    /// Key store for column level encryption, present when the server
    /// configured one. Encrypted columns encrypt on write and decrypt on
    /// scan through it
    pub key_store: Option<Arc<dyn zyron_auth::KeyStore>>,
    /// Content addressed media store, present on a running server. Media
    /// columns externalize oversized payloads into it on write and read
    /// them back at scan
    pub media_store: Option<Arc<zyron_media::store::MediaStore>>,
    /// Security manager reference for search privilege checks at query time.
    /// Operators use this to verify FulltextSearch, VectorSearch, GraphTraverse,
    /// and GraphAlgorithm privileges before executing search operations.
    pub security_manager: Option<Arc<zyron_auth::SecurityManager>>,
    /// Vector index manager reference for DML index maintenance and query-time
    /// index lookup. DML operators use this to maintain vector indexes on
    /// INSERT/UPDATE/DELETE. Scan operators use it to find vector indexes.
    pub vector_manager: Option<Arc<zyron_search::vector::VectorIndexManager>>,
    /// Graph manager reference for graph algorithm execution. Graph scan
    /// operators use this to look up graph schemas and build CSR representations.
    pub graph_manager: Option<Arc<zyron_search::graph::GraphManager>>,
    /// Spatial (R-tree) index manager. Spatial scan operators look up
    /// indexes by id; DML operators use it to maintain indexes on
    /// INSERT/UPDATE/DELETE of indexed geometry columns.
    pub spatial_manager: Option<Arc<zyron_types::spatial_index::SpatialIndexManager>>,
    /// Server-wide HeapFile cache keyed by TableId. Each `HeapFile` carries
    /// its own free-space hint cache, so reusing one instance across queries
    /// is what lets sequential single-row INSERTs land on the same hot page
    /// instead of allocating a fresh one per call
    pub heap_files: Option<Arc<scc::HashMap<u32, Arc<HeapFile>>>>,
    /// Server-wide live B+Tree index cache keyed by index_id. IndexScan
    /// operators look up here via get_index, DML operators maintain entries
    /// here on insert/update/delete
    pub btree_indexes: Option<Arc<scc::HashMap<u32, Arc<BTreeIndex>>>>,
    /// Reads tables that live on another node. Injected rather than built
    /// here because reaching a peer means the wire protocol, the client
    /// pool and the peer registry, all of which live above this crate.
    /// None on a node that holds no client, where a foreign scan reports
    /// that plainly instead of returning no rows
    pub foreign_reader: Option<Arc<dyn crate::operator::foreign_scan::ForeignReader>>,
    /// This node's view of the mesh, needed when a plan is built here
    /// rather than above: a subquery, a correlated inner plan or a trigger
    /// body re-plans, and a foreign scan inside one has to be costed
    /// against the same peer facts the outer plan used
    pub peers: Option<Arc<parking_lot::RwLock<Arc<zyron_common::PeerRegistry>>>>,
    /// Branch override resolver. Set when a session has a branch active or a
    /// query reads `IN BRANCH`. Heap reads route page ids through this so a
    /// branch sees its copy-on-write pages. None means the main line.
    pub branch_catalog: Option<Arc<dyn zyron_common::BranchCatalog>>,
    /// Active branch id for this execution (from USE BRANCH). A per-query
    /// `IN BRANCH name` resolves its own id at the scan that carries it.
    pub active_branch_id: Option<u64>,
    /// The same branch by name. A heap branch is addressed by the id above,
    /// which is what routes copy-on-write pages, while a lake branch is an
    /// alternate log head addressed by name. Both come from one USE BRANCH,
    /// so the session carries both and each store reads the one it uses.
    /// Shared rather than owned so a child context built per outer row of a
    /// correlated subquery carries the name without reallocating it
    pub active_branch_name: Option<Arc<str>>,
    /// Shared intent-lock table for key-level conflict detection. When present,
    /// unique-index inserts take a key lock on the indexed value so concurrent
    /// transactions inserting the same value serialize (first locker wins, the
    /// loser gets a conflict). None disables key locking (single-threaded paths).
    pub intent_locks: Option<Arc<zyron_storage::IntentLockTable>>,
    /// Shared row-level lock table. SELECT FOR UPDATE/SHARE locks its result
    /// rows through this, and DML takes exclusive row locks before writing so
    /// a held FOR UPDATE lock actually blocks a concurrent write. Keys on
    /// RowLocator, so heap and columnar resident rows lock uniformly. None
    /// disables row locking (single-threaded internal paths).
    pub row_locks: Option<Arc<zyron_storage::LockTable>>,
    /// Shared per-table document identity for search indexes. DML allocates
    /// a dense ordinal DocId per indexed row and resolves a row's DocId for
    /// index deletes; search scans map result DocIds back to row locators.
    /// Keys on RowLocator, so folded rows keep their documents. None when
    /// no search index maintenance can occur.
    pub doc_registry: Option<Arc<zyron_common::DocRegistry>>,
    /// Per-table IO and tuple counters. Scan operators resolve their table's
    /// entry once when they are built and record per batch; DML operators
    /// record the rows they write. The stat views read the registry back.
    /// None for internal queries that run outside a server.
    pub table_io_stats: Option<Arc<zyron_common::TableIOStatsRegistry>>,
    /// Per-index scan counters, recorded by the index scan operators alongside
    /// the table counters above. None for internal queries.
    pub index_io_stats: Option<Arc<zyron_common::IndexIOStatsRegistry>>,
    /// Per-session sequence state for currval and lastval. Shared across the
    /// session's queries so currval('s') reads the value the session's last
    /// nextval('s') produced. None for internal queries with no session.
    pub session_sequences: Option<Arc<crate::sequence::SessionSeqState>>,
    /// Number of triggers currently on the firing stack. A trigger action runs
    /// in a nested context with this incremented; firing stops past a fixed
    /// depth so a trigger that re-triggers itself cannot recurse without bound.
    pub trigger_depth: usize,
    /// Shared undo log of the owning transaction. DML operators record one
    /// reverse-op per write here, but only while the transaction has an open
    /// savepoint, so a transaction with no savepoint records nothing. ROLLBACK
    /// TO SAVEPOINT reverses these entries. None for executions outside a
    /// savepoint-capable transaction.
    pub undo_log: Option<Arc<zyron_storage::TxnUndoLog>>,
    /// True when the enclosing transaction was started READ ONLY. Write
    /// operators reject before touching the heap, so no execution path (direct
    /// DML, a prepared write run through the extended protocol, or a write
    /// inside CALL, DO, or a trigger) can mutate data in a read-only
    /// transaction. Inherited by child contexts so nested execution stays
    /// read-only.
    pub read_only: bool,
}

impl ExecutionContext {
    /// Creates a new execution context for a query within the given transaction.
    pub fn new(
        catalog: Arc<Catalog>,
        wal: Arc<WalWriter>,
        buffer_pool: Arc<BufferPool>,
        disk_manager: Arc<DiskManager>,
        txn_id: u64,
        snapshot: Snapshot,
    ) -> Self {
        Self {
            catalog,
            wal,
            buffer_pool,
            disk_manager,
            batch_size: BATCH_SIZE,
            txn_id,
            lake_txn_id: None,
            snapshot,
            cancelled: Arc::new(AtomicBool::new(false)),
            deadline: None,
            max_result_rows: None,
            memory_budget: None,
            spill: None,
            node_memory_held: AtomicU64::new(0),
            wrote_wal: AtomicBool::new(false),
            variant_paths: std::sync::RwLock::new(Arc::from(Vec::new())),
            analyze: false,
            cdc_hook: None,
            replication: None,
            replication_apply: false,
            dml_hook: None,
            params: Vec::new(),
            planning_database: zyron_catalog::DatabaseId(1),
            security_context: None,
            indexes: Arc::new(HashMap::new()),
            fts_indexes: Arc::new(HashMap::new()),
            fts_manager: None,
            key_store: None,
            media_store: None,
            security_manager: None,
            vector_manager: None,
            graph_manager: None,
            spatial_manager: None,
            heap_files: None,
            btree_indexes: None,
            foreign_reader: None,
            peers: None,
            branch_catalog: None,
            active_branch_id: None,
            active_branch_name: None,
            intent_locks: None,
            row_locks: None,
            doc_registry: None,
            table_io_stats: None,
            index_io_stats: None,
            session_sequences: None,
            trigger_depth: 0,
            undo_log: None,
            read_only: false,
        }
    }

    /// Records the variant paths the plan about to be built reads, replacing
    /// whatever a previous plan on this context recorded
    pub fn set_variant_paths(
        &self,
        paths: Vec<zyron_planner::physical::variant_paths::VariantPath>,
    ) {
        if let Ok(mut slot) = self.variant_paths.write() {
            *slot = Arc::from(paths);
        }
    }

    /// The variant paths recorded for the plan being built. A scan captures
    /// its share at construction, so a nested plan recording its own later
    /// cannot change what an already-built scan reads
    pub fn variant_paths(&self) -> Arc<[zyron_planner::physical::variant_paths::VariantPath]> {
        match self.variant_paths.read() {
            Ok(slot) => Arc::clone(&slot),
            Err(_) => Arc::from(Vec::new()),
        }
    }

    /// Builds a child context for executing a nested plan (a correlated
    /// subquery's per-row evaluation or a LATERAL inner plan) that shares this
    /// context's transaction, snapshot, storage caches, index managers, and
    /// security context but carries its own parameter set. Cancellation is
    /// SHARED so a cancelled statement stops its nested work too; wrote_wal
    /// starts fresh and callers that run writes through a child propagate it
    /// back explicitly.
    pub fn child_with_params(&self, params: Vec<ScalarValue>) -> Self {
        Self {
            catalog: Arc::clone(&self.catalog),
            wal: Arc::clone(&self.wal),
            buffer_pool: Arc::clone(&self.buffer_pool),
            disk_manager: Arc::clone(&self.disk_manager),
            batch_size: self.batch_size,
            txn_id: self.txn_id,
            lake_txn_id: self.lake_txn_id,
            snapshot: self.snapshot.clone(),
            cancelled: Arc::clone(&self.cancelled),
            deadline: self.deadline,
            max_result_rows: self.max_result_rows,
            memory_budget: self.memory_budget.clone(),
            spill: self.spill.clone(),
            // A child context charges and releases its own bytes. Inheriting
            // the parent's total would release it twice
            node_memory_held: AtomicU64::new(0),
            wrote_wal: AtomicBool::new(false),
            variant_paths: std::sync::RwLock::new(self.variant_paths()),
            analyze: false,
            cdc_hook: self.cdc_hook.clone(),
            replication: self.replication.clone(),
            replication_apply: self.replication_apply,
            dml_hook: self.dml_hook.clone(),
            params,
            planning_database: self.planning_database,
            security_context: self.security_context.clone(),
            indexes: Arc::clone(&self.indexes),
            fts_indexes: Arc::clone(&self.fts_indexes),
            fts_manager: self.fts_manager.clone(),
            key_store: self.key_store.clone(),
            media_store: self.media_store.clone(),
            security_manager: self.security_manager.clone(),
            vector_manager: self.vector_manager.clone(),
            graph_manager: self.graph_manager.clone(),
            spatial_manager: self.spatial_manager.clone(),
            heap_files: self.heap_files.clone(),
            btree_indexes: self.btree_indexes.clone(),
            foreign_reader: self.foreign_reader.clone(),
            peers: self.peers.clone(),
            branch_catalog: self.branch_catalog.clone(),
            active_branch_id: self.active_branch_id,
            active_branch_name: self.active_branch_name.clone(),
            intent_locks: self.intent_locks.clone(),
            row_locks: self.row_locks.clone(),
            doc_registry: self.doc_registry.clone(),
            table_io_stats: self.table_io_stats.clone(),
            index_io_stats: self.index_io_stats.clone(),
            session_sequences: self.session_sequences.clone(),
            trigger_depth: self.trigger_depth,
            undo_log: self.undo_log.clone(),
            read_only: self.read_only,
        }
    }

    /// Resolves a heap page through the active branch's override chain. Returns
    /// `page_id` unchanged when no branch is active or the branch has not
    /// modified the page. `branch_id` is the scan's effective branch.
    #[inline]
    pub fn resolve_branch_page(
        &self,
        branch_id: Option<u64>,
        page_id: zyron_common::PageId,
    ) -> zyron_common::PageId {
        match (branch_id, &self.branch_catalog) {
            (Some(bid), Some(cat)) => cat.resolve_page_for(bid, page_id),
            _ => page_id,
        }
    }

    /// Resolves the IO counters for a table, creating the entry on first use.
    ///
    /// Operators call this once while they are being built and hold the Arc for
    /// their lifetime, so the registry hash lookup never lands on a batch path.
    /// Returns None when no registry is installed, which is what an internal
    /// query running outside a server sees.
    pub fn table_io_stats_for(&self, table_id: u32) -> Option<Arc<zyron_common::TableIOStats>> {
        self.table_io_stats
            .as_ref()
            .map(|registry| registry.get_or_create(table_id))
    }

    /// Resolves the IO counters for an index, creating the entry on first use.
    /// Held for the operator's lifetime like the table counters above.
    pub fn index_io_stats_for(&self, index_id: u32) -> Option<Arc<zyron_common::IndexIOStats>> {
        self.index_io_stats
            .as_ref()
            .map(|registry| registry.get_or_create(index_id))
    }

    /// Signals all operators using this context to stop execution.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    /// Sets the wall-clock deadline past which the statement times out.
    /// Operators observe it through check_cancelled at batch boundaries.
    pub fn set_deadline(&mut self, deadline: std::time::Instant) {
        self.deadline = Some(deadline);
    }

    /// Reserves bytes against the node and against this query's budget.
    ///
    /// Two counters answering two questions. The node gauge asks whether this
    /// machine is about to run out, so it is charged even when the session has
    /// no per-query limit, and it is given back when the query ends. The query
    /// budget asks whether this one query has asked for more than it may, so
    /// it never gives anything back.
    ///
    /// The node is charged first because it is the one that can be refunded.
    /// Charging the query budget first and then failing on the node would
    /// leave the query permanently charged for memory it never received, and
    /// a later reservation inside its own limit would be refused.
    #[inline]
    pub fn reserve_memory(&self, bytes: u64) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        let gauge = zyron_pressure::pressure_control::PressureController::global().memory();
        if !gauge.try_reserve(bytes) {
            return Err(ZyronError::MemoryAllocationFailed { bytes });
        }
        if let Some(budget) = &self.memory_budget {
            if let Err(e) = budget.reserve(bytes) {
                gauge.release(bytes);
                return Err(e);
            }
        }
        self.node_memory_held.fetch_add(bytes, Ordering::Relaxed);
        Ok(())
    }

    /// How much a materializing operator may hold before it spills.
    ///
    /// The query's own budget, so an operator spills at exactly the point it
    /// used to fail. Without a budget there is nothing to exceed and nothing
    /// spills, which is the unbudgeted behaviour that was always there.
    pub fn spill_threshold_bytes(&self) -> u64 {
        match &self.memory_budget {
            Some(budget) => zyron_pressure::pressure_control::PressureController::global()
                .spill_threshold(budget.limit()),
            None => 0,
        }
    }

    /// Bytes this context is currently holding against the node gauge.
    pub fn node_memory_held(&self) -> u64 {
        self.node_memory_held.load(Ordering::Relaxed)
    }

    /// Returns true if this query has been cancelled.
    #[inline]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }

    /// The wall-clock deadline for this statement, if the session set one.
    ///
    /// A lake commit takes this so its retry loop stops waiting when the
    /// statement is over time. It governs waiting only: the commit reads it
    /// between attempts, where nothing is staged and no version exists, and
    /// never during one
    #[inline]
    pub fn deadline(&self) -> Option<std::time::Instant> {
        self.deadline
    }

    /// Returns true once the statement deadline has elapsed. Always false
    /// when no deadline is set.
    #[inline]
    pub fn deadline_exceeded(&self) -> bool {
        match self.deadline {
            Some(d) => std::time::Instant::now() >= d,
            None => false,
        }
    }

    /// Records that a WAL data record was appended during this execution.
    /// DML operators call this when they log inserts, updates, or deletes.
    #[inline]
    /// The transaction id a lake commit runs under: the cross-table intent
    /// when one is open, otherwise the database transaction.
    pub fn lake_txn_id(&self) -> u64 {
        self.lake_txn_id.unwrap_or(self.txn_id as u64)
    }

    pub fn mark_wrote_wal(&self) {
        self.wrote_wal.store(true, Ordering::Relaxed);
    }

    /// Rejects a write in a read-only transaction. Write operators call this
    /// before any heap mutation so a read-only transaction cannot write through
    /// any path. op is the SQL verb for the error message.
    #[inline]
    pub fn ensure_writable(&self, op: &str) -> Result<()> {
        if self.read_only {
            return Err(ZyronError::ExecutionError(format!(
                "cannot execute {op} in a read-only transaction"
            )));
        }
        Ok(())
    }

    /// Refuses a heap or columnar write when the session names a branch the
    /// heap does not carry.
    ///
    /// A lake branch can exist on one table alone, so a session can be bound
    /// to a branch with no database-wide entry. The lake side writes that
    /// branch's head; the heap side has no overlay to write and would land
    /// on the main line, which is the isolation the session asked for being
    /// silently dropped.
    #[inline]
    pub fn ensure_heap_branch_resolved(&self, op: &str, table_name: &str) -> Result<()> {
        if self.active_branch_id.is_none() {
            if let Some(branch) = &self.active_branch_name {
                return Err(ZyronError::ExecutionError(format!(
                    "{op} on \"{}\" while the session is on branch \"{}\", which exists on lake \
                     tables only. Create the branch database-wide to write heap tables on it",
                    table_name, branch
                )));
            }
        }
        Ok(())
    }

    /// True when the owning transaction has an open savepoint, so DML operators
    /// must record reverse-ops for their writes. False on the common path, where
    /// no undo recording happens. A single relaxed atomic load when an undo log
    /// is present.
    #[inline]
    pub fn recording_undo(&self) -> bool {
        self.undo_log
            .as_ref()
            .is_some_and(|log| log.has_active_savepoint())
    }

    /// Records a ReverseInsert undo entry for a tuple this transaction inserted,
    /// so ROLLBACK TO SAVEPOINT self-deletes it. No-op unless a savepoint is
    /// open. `heap_file_id`/`fsm_file_id` address the heap that holds the tuple.
    #[inline]
    pub fn record_insert_undo(&self, heap_file_id: u32, fsm_file_id: u32, tid: TupleId) {
        if let Some(log) = &self.undo_log {
            if log.has_active_savepoint() {
                log.record(zyron_storage::UndoEntry::ReverseInsert {
                    heap_file_id,
                    fsm_file_id,
                    tid,
                });
            }
        }
    }

    /// Records a ReverseDelete undo entry for a pre-existing tuple this
    /// transaction deleted (stamped xmax), so ROLLBACK TO SAVEPOINT clears its
    /// xmax and restores it. No-op unless a savepoint is open.
    #[inline]
    pub fn record_delete_undo(&self, heap_file_id: u32, fsm_file_id: u32, tid: TupleId) {
        if let Some(log) = &self.undo_log {
            if log.has_active_savepoint() {
                log.record(zyron_storage::UndoEntry::ReverseDelete {
                    heap_file_id,
                    fsm_file_id,
                    tid,
                });
            }
        }
    }

    /// Records that this transaction superseded a columnar-resident row, so
    /// ROLLBACK TO SAVEPOINT revokes the supersede and the row reappears.
    /// No-op unless a savepoint is open.
    #[inline]
    pub fn record_columnar_supersede_undo(
        &self,
        table_id: u32,
        branch: u64,
        file_id: u64,
        sys_rowid: u64,
    ) {
        if let Some(log) = &self.undo_log {
            if log.has_active_savepoint() {
                log.record(zyron_storage::UndoEntry::ColumnarSupersede {
                    table_id,
                    branch,
                    file_id,
                    sys_rowid,
                });
            }
        }
    }

    /// Records that this transaction patched one column of a
    /// columnar-resident row, so ROLLBACK TO SAVEPOINT revokes the patch and
    /// the prior value is visible again. No-op unless a savepoint is open.
    #[inline]
    pub fn record_columnar_patch_undo(
        &self,
        table_id: u32,
        branch: u64,
        file_id: u64,
        sys_rowid: u64,
        column_id: u32,
    ) {
        if let Some(log) = &self.undo_log {
            if log.has_active_savepoint() {
                log.record(zyron_storage::UndoEntry::ColumnarPatch {
                    table_id,
                    branch,
                    file_id,
                    sys_rowid,
                    column_id,
                });
            }
        }
    }

    /// Returns true if a WAL data record was appended during this execution.
    #[inline]
    pub fn wrote_wal(&self) -> bool {
        self.wrote_wal.load(Ordering::Relaxed)
    }

    /// Checks cancellation and the statement deadline, returning an error when
    /// either trips. Operators call this at batch boundaries for cooperative
    /// cancellation. A tripped deadline reports a statement timeout so the
    /// client sees the limit rather than a generic cancellation.
    #[inline]
    pub fn check_cancelled(&self) -> Result<()> {
        if self.is_cancelled() {
            Err(ZyronError::Internal("Query cancelled".into()))
        } else if self.deadline_exceeded() {
            Err(ZyronError::Internal("statement timeout".into()))
        } else {
            Ok(())
        }
    }

    /// Returns the per-table cached `HeapFile` if the server-wide cache is
    /// installed, otherwise constructs a fresh one. The cached instance keeps
    /// hint_slots warm across queries so sequential single-row INSERTs land
    /// on the same hot page instead of allocating a fresh one per call.
    /// Cached_heap_pages and cached_fsm_pages are seeded from disk via
    /// init_cache the first time a table is touched.
    pub async fn get_heap_file(&self, table_id: TableId) -> Result<Arc<HeapFile>> {
        let entry = self.catalog.get_table_by_id(table_id)?;
        if let Some(cache) = &self.heap_files {
            if let Some(hit) = cache.get_async(&entry.heap_file_id).await {
                return Ok(Arc::clone(hit.get()));
            }
            let hf = HeapFile::new(
                self.disk_manager.clone(),
                self.buffer_pool.clone(),
                HeapFileConfig {
                    heap_file_id: entry.heap_file_id,
                    fsm_file_id: entry.fsm_file_id,
                },
            )?;
            hf.init_cache().await?;
            let arc = Arc::new(hf);
            // Race tolerated, the loser's instance is dropped, ensuing
            // calls converge on the winner. Init cost is one disk stat per
            // file id, which is cheap relative to losing a race once
            match cache
                .insert_async(entry.heap_file_id, Arc::clone(&arc))
                .await
            {
                Ok(()) => Ok(arc),
                Err(_) => {
                    let hit = cache
                        .get_async(&entry.heap_file_id)
                        .await
                        .expect("racer just inserted");
                    Ok(Arc::clone(hit.get()))
                }
            }
        } else {
            let hf = HeapFile::new(
                self.disk_manager.clone(),
                self.buffer_pool.clone(),
                HeapFileConfig {
                    heap_file_id: entry.heap_file_id,
                    fsm_file_id: entry.fsm_file_id,
                },
            )?;
            hf.init_cache().await?;
            Ok(Arc::new(hf))
        }
    }

    /// Returns a `HeapFile` bound to a branch's append overlay files, building
    /// and caching it in the shared heap file cache keyed by the append file id.
    /// Branch append file ids are disjoint from table heap file ids, so the same
    /// cache holds both without collision.
    pub async fn branch_append_heap(
        &self,
        append_file_id: u32,
        append_fsm_file_id: u32,
    ) -> Result<Arc<HeapFile>> {
        let build = || -> Result<HeapFile> {
            HeapFile::new(
                self.disk_manager.clone(),
                self.buffer_pool.clone(),
                HeapFileConfig {
                    heap_file_id: append_file_id,
                    fsm_file_id: append_fsm_file_id,
                },
            )
        };
        if let Some(cache) = &self.heap_files {
            if let Some(hit) = cache.get_async(&append_file_id).await {
                return Ok(Arc::clone(hit.get()));
            }
            let hf = build()?;
            hf.init_cache().await?;
            let arc = Arc::new(hf);
            match cache.insert_async(append_file_id, Arc::clone(&arc)).await {
                Ok(()) => Ok(arc),
                Err(_) => {
                    let hit = cache
                        .get_async(&append_file_id)
                        .await
                        .expect("racer just inserted");
                    Ok(Arc::clone(hit.get()))
                }
            }
        } else {
            let hf = build()?;
            hf.init_cache().await?;
            Ok(Arc::new(hf))
        }
    }

    /// Returns the catalog TableEntry for the given table ID.
    pub fn get_table_entry(&self, table_id: TableId) -> Result<Arc<TableEntry>> {
        self.catalog.get_table_by_id(table_id)
    }

    /// Registers a live B+ tree index instance for use by index scan operators.
    /// Called by the server layer after creating or loading an index.
    pub fn register_index(&mut self, index_id: IndexId, btree: Arc<BTreeIndex>) {
        Arc::make_mut(&mut self.indexes).insert(index_id, btree);
    }

    /// Returns the B+ tree index instance for the given IndexId. Consults
    /// the server-wide btree_indexes registry first (lock-free scc lookup),
    /// then falls back to the per-context map, which is what a context
    /// built without a server registry carries
    pub fn get_index(&self, index_id: IndexId) -> Option<Arc<BTreeIndex>> {
        if let Some(server) = &self.btree_indexes {
            if let Some(hit) = server.read_sync(&index_id.0, |_, v| Arc::clone(v)) {
                return Some(hit);
            }
        }
        self.indexes.get(&index_id).cloned()
    }

    /// Registers a live full-text search index instance for use by FTS scan operators.
    pub fn register_fts_index(&mut self, index_id: IndexId, fts: Arc<zyron_search::InvertedIndex>) {
        Arc::make_mut(&mut self.fts_indexes).insert(index_id, fts);
    }

    /// Returns the FTS index instance for the given IndexId.
    /// Checks local cache first, then falls through to the FTS manager.
    pub fn get_fts_index(&self, index_id: IndexId) -> Option<Arc<zyron_search::InvertedIndex>> {
        if let Some(idx) = self.fts_indexes.get(&index_id) {
            return Some(idx.clone());
        }
        if let Some(ref mgr) = self.fts_manager {
            return mgr.get_index(index_id.0);
        }
        None
    }

    /// Sets the FTS manager reference. Scan operators look up indexes
    /// through the manager on demand. DML operators use fts_indexes_for_table().
    pub fn set_fts_manager(&mut self, mgr: Arc<zyron_search::FtsManager>) {
        self.fts_manager = Some(mgr);
    }

    /// Sets the key store column encryption resolves keys through
    pub fn set_key_store(&mut self, store: Arc<dyn zyron_auth::KeyStore>) {
        self.key_store = Some(store);
    }

    /// Sets the media store media columns externalize through
    pub fn set_media_store(&mut self, store: Arc<zyron_media::store::MediaStore>) {
        self.media_store = Some(store);
    }

    /// Resolves the analyzer for an FTS index. Falls back to the simple
    /// pipeline when no manager is attached, matching how unconfigured
    /// indexes were always analyzed
    pub fn fts_analyzer(&self, index_id: u32) -> Arc<dyn zyron_search::Analyzer> {
        match self.fts_manager.as_ref() {
            Some(mgr) => mgr.analyzer_for_index(index_id),
            None => Arc::new(zyron_search::SimpleAnalyzer),
        }
    }

    /// Reports whether an FTS index already stores phonetic codes as terms
    pub fn fts_phonetic_indexed(&self, index_id: u32) -> bool {
        self.fts_manager
            .as_ref()
            .is_some_and(|mgr| mgr.index_phonetic_indexed(index_id))
    }

    /// Sets the security manager for search privilege checks at query time.
    pub fn set_security_manager(&mut self, mgr: Arc<zyron_auth::SecurityManager>) {
        self.security_manager = Some(mgr);
    }

    /// Checks whether the current session has the given search privilege on an object.
    /// When security is not configured (no SecurityManager or no SecurityContext),
    /// access is allowed by default. Uses the PrivilegeStore directly to avoid
    /// needing mutable access to the SecurityContext cache.
    pub fn check_search_privilege(
        &self,
        privilege: zyron_auth::PrivilegeType,
        object_id: u32,
    ) -> Result<()> {
        let sm = match self.security_manager.as_ref() {
            Some(sm) => sm,
            None => return Ok(()),
        };
        let ctx = match self.security_context.as_ref() {
            Some(ctx) => ctx,
            None => return Ok(()),
        };
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let decision = sm.privilege_store.check_privilege(
            &ctx.effective_roles,
            privilege,
            zyron_auth::ObjectType::Table,
            object_id,
            None,
            now,
        );
        if decision == zyron_auth::PrivilegeDecision::Allow
            || decision == zyron_auth::PrivilegeDecision::Unset
        {
            Ok(())
        } else {
            Err(ZyronError::PermissionDenied(format!(
                "permission denied: {:?} on table {}",
                privilege, object_id
            )))
        }
    }

    /// Returns the lock-free, pre-partitioned index snapshot for a table.
    /// DML hot paths consult this once per statement instead of paying the
    /// per-batch cost of four separate catalog `RwLock` reads and Vec
    /// allocations. Tables with no indexes share a single static empty
    /// snapshot.
    #[inline]
    pub fn index_snapshot_for_table(&self, table_id: u32) -> Arc<TableIndexSnapshot> {
        self.catalog.index_snapshot(TableId(table_id))
    }

    /// Returns all live FTS indexes for the given table. Used by DML operators
    /// to maintain FTS indexes on INSERT/UPDATE/DELETE.
    pub fn fts_indexes_for_table(
        &self,
        table_id: u32,
    ) -> Vec<(IndexId, Arc<zyron_search::InvertedIndex>)> {
        let Some(mgr) = self.fts_manager.as_ref() else {
            return Vec::new();
        };
        let snap = self.index_snapshot_for_table(table_id);
        if snap.fts.is_empty() {
            return Vec::new();
        }
        snap.fts
            .iter()
            .filter_map(|id| mgr.get_index(id.0).map(|idx| (*id, idx)))
            .collect()
    }

    /// Sets the vector index manager for DML maintenance and query-time lookups.
    pub fn set_vector_manager(&mut self, mgr: Arc<zyron_search::vector::VectorIndexManager>) {
        self.vector_manager = Some(mgr);
    }

    /// Returns the vector index with the given ID from the vector manager.
    pub fn get_vector_index(
        &self,
        index_id: u32,
    ) -> Option<Arc<zyron_search::vector::VectorIndex>> {
        self.vector_manager
            .as_ref()
            .and_then(|mgr| mgr.get_index(index_id))
    }

    /// Returns all vector index IDs for the given table. Used by DML operators
    /// to maintain vector indexes on INSERT/UPDATE/DELETE.
    pub fn vector_indexes_for_table(&self, table_id: u32) -> Vec<u32> {
        if self.vector_manager.is_none() {
            return Vec::new();
        }
        let snap = self.index_snapshot_for_table(table_id);
        snap.vector.iter().map(|id| id.0).collect()
    }

    /// Sets the graph manager for algorithm execution.
    pub fn set_graph_manager(&mut self, mgr: Arc<zyron_search::graph::GraphManager>) {
        self.graph_manager = Some(mgr);
    }

    /// Sets the spatial index manager for R-tree-backed scan operators.
    pub fn set_spatial_manager(
        &mut self,
        mgr: Arc<zyron_types::spatial_index::SpatialIndexManager>,
    ) {
        self.spatial_manager = Some(mgr);
    }

    /// Returns all (index_id, indexed column_id) pairs for spatial indexes
    /// on the given table. Lock-free read from the catalog index snapshot.
    pub fn spatial_indexes_for_table(&self, table_id: u32) -> Vec<(u32, zyron_catalog::ColumnId)> {
        let snap = self.index_snapshot_for_table(table_id);
        snap.spatial.iter().map(|(id, col)| (id.0, *col)).collect()
    }

    /// Returns (index_id, leading key column_id) for B+Tree indexes on the
    /// table. Lock-free read from the catalog index snapshot. Index selection
    /// matches on the leading column, so that is what this reports.
    pub fn btree_indexes_for_table(&self, table_id: u32) -> Vec<(u32, zyron_catalog::ColumnId)> {
        let snap = self.index_snapshot_for_table(table_id);
        snap.btree
            .iter()
            .map(|spec| (spec.id.0, spec.leading()))
            .collect()
    }
}

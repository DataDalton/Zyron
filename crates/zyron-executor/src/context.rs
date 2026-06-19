//! Execution context providing access to storage, catalog, and transaction state.
//!
//! Each query execution receives an ExecutionContext that holds references to
//! shared infrastructure (buffer pool, WAL, catalog) along with per-query
//! state (transaction ID, MVCC snapshot, batch size). Also provides query
//! cancellation via an atomic flag and optional per-operator metrics
//! collection for EXPLAIN ANALYZE.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use zyron_buffer::BufferPool;
use zyron_catalog::{Catalog, IndexId, TableEntry, TableId, TableIndexSnapshot};
use zyron_common::{Result, ZyronError};
use zyron_storage::{BTreeIndex, DiskManager, HeapFile, HeapFileConfig, Snapshot};
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
        txn_id: u32,
        is_last_in_txn: bool,
    ) -> zyron_common::Result<()>;

    /// Called after rows are deleted. old_data contains pre-delete tuple bytes.
    fn on_delete(
        &self,
        table_id: u32,
        old_data: &[&[u8]],
        version: u64,
        timestamp: i64,
        txn_id: u32,
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
        txn_id: u32,
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
        txn_id: u32,
    ) -> zyron_common::Result<bool>;

    /// Called before rows are deleted. Returns false to cancel the delete.
    fn before_delete(
        &self,
        table_id: u32,
        old_data: &[&[u8]],
        txn_id: u32,
    ) -> zyron_common::Result<bool>;

    /// Called before rows are updated. Returns false to cancel the update.
    fn before_update(
        &self,
        table_id: u32,
        old_data: &[&[u8]],
        new_data: &[&[u8]],
        txn_id: u32,
    ) -> zyron_common::Result<bool>;
}

/// Per-query execution context with access to storage and transaction state.
pub struct ExecutionContext {
    pub catalog: Arc<Catalog>,
    pub wal: Arc<WalWriter>,
    pub buffer_pool: Arc<BufferPool>,
    pub disk_manager: Arc<DiskManager>,
    pub batch_size: usize,
    pub txn_id: u32,
    pub snapshot: Snapshot,
    /// When set to true, operators check this flag and bail with a cancellation error.
    cancelled: AtomicBool,
    /// Set by DML operators when they append a WAL data record. The server
    /// reads it after execution to decide whether the transaction must commit
    /// durably; a transaction that wrote nothing commits without a WAL commit
    /// record or a flush wait.
    wrote_wal: AtomicBool,
    /// When true, operators collect per-operator metrics (rows, timing).
    pub analyze: bool,
    /// Optional CDC hook invoked by DML operators after mutations.
    pub cdc_hook: Option<Arc<dyn CdcHook>>,
    /// Optional DML hook invoked by DML operators before mutations (BEFORE triggers).
    pub dml_hook: Option<Arc<dyn DmlHook>>,
    /// Bound parameter values ($1, $2, ...) for prepared statements.
    pub params: Vec<ScalarValue>,
    /// Per-session security context for privilege checks. None when the auth
    /// system is not configured or for internal queries that bypass auth.
    /// Held behind an Arc so a nested execution (a correlated subquery or a
    /// LATERAL inner plan) shares the same clearance and masking policy as the
    /// enclosing query rather than running unsecured.
    pub security_context: Option<Arc<zyron_auth::SecurityContext>>,
    /// Live B+ tree index instances keyed by IndexId. Registered by the
    /// server layer so the index scan operator can perform actual tree lookups.
    indexes: HashMap<IndexId, Arc<BTreeIndex>>,
    /// Live full-text search index instances keyed by IndexId. Registered by
    /// the server layer after creating or loading fulltext indexes.
    fts_indexes: HashMap<IndexId, Arc<zyron_search::InvertedIndex>>,
    /// FTS manager reference for DML index maintenance. DML operators use this
    /// to look up which FTS indexes exist for a table and update them.
    pub fts_manager: Option<Arc<zyron_search::FtsManager>>,
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
    /// Branch override resolver. Set when a session has a branch active or a
    /// query reads `IN BRANCH`. Heap reads route page ids through this so a
    /// branch sees its copy-on-write pages. None means the main line.
    pub branch_catalog: Option<Arc<dyn zyron_common::BranchCatalog>>,
    /// Active branch id for this execution (from USE BRANCH). A per-query
    /// `IN BRANCH name` resolves its own id at the scan that carries it.
    pub active_branch_id: Option<u64>,
    /// Shared intent-lock table for key-level conflict detection. When present,
    /// unique-index inserts take a key lock on the indexed value so concurrent
    /// transactions inserting the same value serialize (first locker wins, the
    /// loser gets a conflict). None disables key locking (single-threaded paths).
    pub intent_locks: Option<Arc<zyron_storage::IntentLockTable>>,
    /// Per-session sequence state for currval and lastval. Shared across the
    /// session's queries so currval('s') reads the value the session's last
    /// nextval('s') produced. None for internal queries with no session.
    pub session_sequences: Option<Arc<crate::sequence::SessionSeqState>>,
    /// Number of triggers currently on the firing stack. A trigger action runs
    /// in a nested context with this incremented; firing stops past a fixed
    /// depth so a trigger that re-triggers itself cannot recurse without bound.
    pub trigger_depth: usize,
}

impl ExecutionContext {
    /// Creates a new execution context for a query within the given transaction.
    pub fn new(
        catalog: Arc<Catalog>,
        wal: Arc<WalWriter>,
        buffer_pool: Arc<BufferPool>,
        disk_manager: Arc<DiskManager>,
        txn_id: u32,
        snapshot: Snapshot,
    ) -> Self {
        Self {
            catalog,
            wal,
            buffer_pool,
            disk_manager,
            batch_size: BATCH_SIZE,
            txn_id,
            snapshot,
            cancelled: AtomicBool::new(false),
            wrote_wal: AtomicBool::new(false),
            analyze: false,
            cdc_hook: None,
            dml_hook: None,
            params: Vec::new(),
            security_context: None,
            indexes: HashMap::new(),
            fts_indexes: HashMap::new(),
            fts_manager: None,
            security_manager: None,
            vector_manager: None,
            graph_manager: None,
            spatial_manager: None,
            heap_files: None,
            btree_indexes: None,
            branch_catalog: None,
            active_branch_id: None,
            intent_locks: None,
            session_sequences: None,
            trigger_depth: 0,
        }
    }

    /// Builds a child context for executing a nested plan (a correlated
    /// subquery's per-row evaluation or a LATERAL inner plan) that shares this
    /// context's transaction, snapshot, storage caches, index managers, and
    /// security context but carries its own parameter set. Cancellation and
    /// wrote_wal start fresh because the child is read-only and short lived.
    pub fn child_with_params(&self, params: Vec<ScalarValue>) -> Self {
        Self {
            catalog: Arc::clone(&self.catalog),
            wal: Arc::clone(&self.wal),
            buffer_pool: Arc::clone(&self.buffer_pool),
            disk_manager: Arc::clone(&self.disk_manager),
            batch_size: self.batch_size,
            txn_id: self.txn_id,
            snapshot: self.snapshot.clone(),
            cancelled: AtomicBool::new(false),
            wrote_wal: AtomicBool::new(false),
            analyze: false,
            cdc_hook: self.cdc_hook.clone(),
            dml_hook: self.dml_hook.clone(),
            params,
            security_context: self.security_context.clone(),
            indexes: self.indexes.clone(),
            fts_indexes: self.fts_indexes.clone(),
            fts_manager: self.fts_manager.clone(),
            security_manager: self.security_manager.clone(),
            vector_manager: self.vector_manager.clone(),
            graph_manager: self.graph_manager.clone(),
            spatial_manager: self.spatial_manager.clone(),
            heap_files: self.heap_files.clone(),
            btree_indexes: self.btree_indexes.clone(),
            branch_catalog: self.branch_catalog.clone(),
            active_branch_id: self.active_branch_id,
            intent_locks: self.intent_locks.clone(),
            session_sequences: self.session_sequences.clone(),
            trigger_depth: self.trigger_depth,
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

    /// Signals all operators using this context to stop execution.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    /// Returns true if this query has been cancelled.
    #[inline]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }

    /// Records that a WAL data record was appended during this execution.
    /// DML operators call this when they log inserts, updates, or deletes.
    #[inline]
    pub fn mark_wrote_wal(&self) {
        self.wrote_wal.store(true, Ordering::Relaxed);
    }

    /// Returns true if a WAL data record was appended during this execution.
    #[inline]
    pub fn wrote_wal(&self) -> bool {
        self.wrote_wal.load(Ordering::Relaxed)
    }

    /// Checks cancellation and returns an error if cancelled.
    /// Operators call this at batch boundaries for cooperative cancellation.
    #[inline]
    pub fn check_cancelled(&self) -> Result<()> {
        if self.is_cancelled() {
            Err(ZyronError::Internal("Query cancelled".into()))
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
        self.indexes.insert(index_id, btree);
    }

    /// Returns the B+ tree index instance for the given IndexId. Consults
    /// the server-wide btree_indexes registry first (lock-free scc lookup),
    /// then falls back to the per-context map for legacy registrations
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
        self.fts_indexes.insert(index_id, fts);
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

    /// Returns (index_id, indexed column_id) for B+Tree indexes on the
    /// table. Lock-free read from the catalog index snapshot.
    pub fn btree_indexes_for_table(&self, table_id: u32) -> Vec<(u32, zyron_catalog::ColumnId)> {
        let snap = self.index_snapshot_for_table(table_id);
        snap.btree.iter().map(|(id, col, _)| (id.0, *col)).collect()
    }
}

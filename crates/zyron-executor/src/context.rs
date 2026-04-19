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
use zyron_catalog::{Catalog, IndexId, TableEntry, TableId};
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
    pub security_context: Option<zyron_auth::SecurityContext>,
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

    /// Constructs a HeapFile handle for the given table.
    pub fn get_heap_file(&self, table_id: TableId) -> Result<HeapFile> {
        let entry = self.catalog.get_table_by_id(table_id)?;
        HeapFile::new(
            self.disk_manager.clone(),
            self.buffer_pool.clone(),
            HeapFileConfig {
                heap_file_id: entry.heap_file_id,
                fsm_file_id: entry.fsm_file_id,
            },
        )
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

    /// Returns the B+ tree index instance for the given IndexId, if registered.
    pub fn get_index(&self, index_id: IndexId) -> Option<Arc<BTreeIndex>> {
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

    /// Returns all live FTS indexes for the given table. Used by DML operators
    /// to maintain FTS indexes on INSERT/UPDATE/DELETE.
    pub fn fts_indexes_for_table(
        &self,
        table_id: u32,
    ) -> Vec<(IndexId, Arc<zyron_search::InvertedIndex>)> {
        match &self.fts_manager {
            Some(mgr) => mgr
                .indexes_for_table(table_id)
                .into_iter()
                .filter_map(|id| mgr.get_index(id).map(|idx| (IndexId(id), idx)))
                .collect(),
            None => Vec::new(),
        }
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
        match &self.vector_manager {
            Some(mgr) => mgr.indexes_for_table(table_id),
            None => Vec::new(),
        }
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
    /// on the given table. Used by DML operators to maintain spatial indexes
    /// on INSERT/UPDATE/DELETE. Reads catalog directly because the spatial
    /// manager only stores trees keyed by index_id and doesn't carry the
    /// column mapping.
    pub fn spatial_indexes_for_table(&self, table_id: u32) -> Vec<(u32, zyron_catalog::ColumnId)> {
        let table_id = zyron_catalog::TableId(table_id);
        self.catalog
            .get_indexes_for_table(table_id)
            .iter()
            .filter(|idx| idx.index_type == zyron_catalog::IndexType::Spatial)
            .filter_map(|idx| idx.columns.first().map(|c| (idx.id.0, c.column_id)))
            .collect()
    }
}

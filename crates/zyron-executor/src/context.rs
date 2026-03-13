//! Execution context providing access to storage, catalog, and transaction state.
//!
//! Each query execution receives an ExecutionContext that holds references to
//! shared infrastructure (buffer pool, WAL, catalog) along with per-query
//! state (transaction ID, MVCC snapshot, batch size). Also provides query
//! cancellation via an atomic flag and optional per-operator metrics
//! collection for EXPLAIN ANALYZE.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use zyron_buffer::BufferPool;
use zyron_catalog::{Catalog, TableEntry, TableId};
use zyron_common::{Result, ZyronError};
use zyron_storage::{DiskManager, HeapFile, HeapFileConfig, Snapshot};
use zyron_wal::WalWriter;

use crate::batch::BATCH_SIZE;

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
}

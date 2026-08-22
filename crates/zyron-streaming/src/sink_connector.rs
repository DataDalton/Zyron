//! Sink connectors for writing streaming output.
//!
//! ZyronRowSink inserts raw CDC row bytes into a target table through the
//! transaction and storage layers with a privilege check. ZyronSinkAdapter is
//! the trait object the runner uses to dispatch rows to a remote Zyron
//! instance over the wire protocol.

use std::sync::Arc;

use zyron_common::{Result, ZyronError};

use crate::source_connector::CdfChange;

// ---------------------------------------------------------------------------
// ZyronRowSink
// ---------------------------------------------------------------------------

/// Sink that inserts raw CDC row bytes into a Zyron target table through
/// the transaction manager and heap file. Runs an Insert privilege check
/// against the captured SecurityContext before opening a transaction.
///
/// write_batch operates on CdfChange rows, one row_data payload per change.
pub struct ZyronRowSink {
    target_table_id: u32,
    write_mode: zyron_catalog::schema::CatalogStreamingWriteMode,
    catalog: Arc<zyron_catalog::Catalog>,
    heap: Arc<zyron_storage::HeapFile>,
    txn_manager: Arc<zyron_storage::txn::TransactionManager>,
    security_ctx: Arc<parking_lot::Mutex<zyron_auth::SecurityContext>>,
    security_manager: Arc<zyron_auth::SecurityManager>,
    // The same per-table write counters DML maintains, so stat views count
    // streamed rows and the background workers' activity gates see them
    io_stats: Arc<zyron_common::TableIOStatsRegistry>,
}

impl ZyronRowSink {
    pub fn new(
        target_table_id: u32,
        write_mode: zyron_catalog::schema::CatalogStreamingWriteMode,
        catalog: Arc<zyron_catalog::Catalog>,
        heap: Arc<zyron_storage::HeapFile>,
        txn_manager: Arc<zyron_storage::txn::TransactionManager>,
        security_ctx: Arc<parking_lot::Mutex<zyron_auth::SecurityContext>>,
        security_manager: Arc<zyron_auth::SecurityManager>,
        io_stats: Arc<zyron_common::TableIOStatsRegistry>,
    ) -> Self {
        Self {
            target_table_id,
            write_mode,
            catalog,
            heap,
            txn_manager,
            security_ctx,
            security_manager,
            io_stats,
        }
    }

    /// Returns the target table id configured for this sink.
    pub fn target_table_id(&self) -> u32 {
        self.target_table_id
    }

    /// Inserts each CdfChange as a new heap tuple inside a single transaction.
    /// Empty input is a no-op. This sink handles Append write mode only;
    /// UPSERT is dispatched to ZyronUpsertSink by the runner. The privilege
    /// check runs once per batch, outside the transaction, so an unauthorized
    /// sink fails fast without touching the WAL.
    pub fn write_batch(&self, records: Vec<CdfChange>) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }

        if self.write_mode == zyron_catalog::schema::CatalogStreamingWriteMode::Upsert {
            return Err(ZyronError::StreamingError(
                "ZyronRowSink received Upsert mode, use ZyronUpsertSink".to_string(),
            ));
        }

        // Verify the creator still has INSERT on the target table.
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        {
            let mut ctx = self.security_ctx.lock();
            let allowed = ctx.has_privilege(
                &self.security_manager.privilege_store,
                zyron_auth::privilege::PrivilegeType::Insert,
                zyron_auth::privilege::ObjectType::Table,
                self.target_table_id,
                None,
                now,
            );
            if !allowed {
                return Err(ZyronError::PermissionDenied(format!(
                    "streaming job sink lacks INSERT on table {}",
                    self.target_table_id
                )));
            }
        }

        // Look up the target table to verify it still exists at insert time.
        let _target = self
            .catalog
            .get_table_by_id(zyron_catalog::TableId(self.target_table_id))?;

        // Begin a transaction, build tuples, insert, commit. Any error aborts.
        let mut txn = self
            .txn_manager
            .begin(zyron_storage::txn::IsolationLevel::SnapshotIsolation)?;
        let txn_id_u32 = match u32::try_from(txn.txn_id) {
            Ok(v) => v,
            Err(_) => {
                let _ = self.txn_manager.abort(&mut txn);
                return Err(ZyronError::Internal(
                    "txn_id exceeds u32::MAX in streaming sink".to_string(),
                ));
            }
        };

        let tuples: Vec<zyron_storage::Tuple> = records
            .iter()
            .map(|c| zyron_storage::Tuple::new(c.row_data.clone(), txn_id_u32))
            .collect();

        // The heap insert is async. Block on a small local runtime since the
        // job runner thread sits outside the main tokio runtime.
        let rt = match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(r) => r,
            Err(e) => {
                let _ = self.txn_manager.abort(&mut txn);
                return Err(ZyronError::Internal(format!(
                    "failed to build tokio runtime for sink insert: {e}"
                )));
            }
        };

        let insert_result = rt.block_on(async { self.heap.insert_batch(&tuples).await });
        match insert_result {
            Ok(_) => {
                self.txn_manager.commit_blocking(&mut txn)?;
                self.io_stats
                    .get_or_create(self.target_table_id)
                    .record_inserts(records.len() as u64);
                Ok(())
            }
            Err(e) => {
                let _ = self.txn_manager.abort(&mut txn);
                Err(e)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ZyronSinkAdapter trait
// ---------------------------------------------------------------------------

/// Adapter trait for sinks that deliver rows from a streaming job to a remote
/// Zyron instance over the PG wire protocol. The concrete implementation lives
/// in the zyron-wire crate as ZyronSinkClient. The trait object form lets the
/// runner in this crate dispatch to the remote sink without taking a build
/// dependency on zyron-wire, preserving the streaming to wire direction of
/// the dep graph.
#[async_trait::async_trait]
pub trait ZyronSinkAdapter: Send + Sync {
    /// Writes a batch of change records to the remote Zyron instance.
    async fn write_batch(&self, records: Vec<CdfChange>) -> Result<()>;

    /// Flushes any pending rows held in the adapter's buffers.
    async fn flush(&self) -> Result<()>;

    /// Shuts the adapter down, draining any remaining rows before return.
    async fn shutdown(&self) -> Result<()>;
}

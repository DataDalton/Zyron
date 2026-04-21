//! Sink connectors for writing streaming output.
//!
//! Provides the SinkConnector trait for writing StreamRecord micro-batches
//! to external systems. Includes an S3 stub, a functional InMemorySink,
//! a counting ZyronTableSink (for trait-level pipelines), and a
//! ZyronRowSink that inserts raw CDC row bytes through the transaction
//! and storage layers with a privilege check.

use std::sync::Arc;

use zyron_common::{Result, ZyronError};

use crate::record::{ChangeFlag, StreamRecord};
use crate::source_connector::CdfChange;

// ---------------------------------------------------------------------------
// SinkConnector trait
// ---------------------------------------------------------------------------

/// Trait for streaming sink connectors.
/// Sinks consume StreamRecord micro-batches and write them to external systems.
pub trait SinkConnector: Send {
    /// Writes a batch of records to the sink.
    fn write_batch(&mut self, records: &[StreamRecord]) -> Result<()>;

    /// Commits the current transaction (for exactly-once sinks).
    fn commit(&mut self) -> Result<()>;

    /// Rolls back the current transaction.
    fn rollback(&mut self) -> Result<()>;

    /// Closes the sink and flushes any buffered data.
    fn close(&mut self) -> Result<()>;
}

// ---------------------------------------------------------------------------
// SinkConfig
// ---------------------------------------------------------------------------

/// Configuration for different sink types.
#[derive(Debug, Clone)]
pub enum SinkConfig {
    S3 {
        bucket: String,
        prefix: String,
        format: String,
    },
    ZyronTable {
        table_id: u32,
        write_mode: WriteMode,
    },
    InMemory,
}

/// Write mode for the ZyronTableSink.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteMode {
    /// Append all records as new rows.
    Append,
    /// Insert or update based on key.
    Upsert,
}

// ---------------------------------------------------------------------------
// StreamS3Sink (stub)
// ---------------------------------------------------------------------------

/// S3 sink connector stub. Fields are placeholders for the real S3 client
/// wiring; the stub impl only counts records.
#[allow(dead_code)]
pub struct StreamS3Sink {
    bucket: String,
    prefix: String,
    format: String,
    records_written: u64,
}

impl StreamS3Sink {
    pub fn new(bucket: String, prefix: String, format: String) -> Self {
        Self {
            bucket,
            prefix,
            format,
            records_written: 0,
        }
    }

    pub fn records_written(&self) -> u64 {
        self.records_written
    }
}

impl SinkConnector for StreamS3Sink {
    fn write_batch(&mut self, records: &[StreamRecord]) -> Result<()> {
        for record in records {
            self.records_written += record.num_rows() as u64;
        }
        Ok(())
    }

    fn commit(&mut self) -> Result<()> {
        Ok(())
    }

    fn rollback(&mut self) -> Result<()> {
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ZyronTableSink
// ---------------------------------------------------------------------------

/// Sink that writes StreamRecord data into a ZyronDB table.
/// Supports append and upsert write modes. Converts ChangeFlags into
/// the appropriate table operations (insert, update, delete).
pub struct ZyronTableSink {
    table_id: u32,
    write_mode: WriteMode,
    /// Buffered records awaiting commit.
    buffer: Vec<StreamRecord>,
    /// Total rows written.
    rows_written: u64,
    /// Total rows deleted.
    rows_deleted: u64,
}

impl ZyronTableSink {
    pub fn new(table_id: u32, write_mode: WriteMode) -> Self {
        Self {
            table_id,
            write_mode,
            buffer: Vec::new(),
            rows_written: 0,
            rows_deleted: 0,
        }
    }

    pub fn rows_written(&self) -> u64 {
        self.rows_written
    }

    pub fn rows_deleted(&self) -> u64 {
        self.rows_deleted
    }

    pub fn table_id(&self) -> u32 {
        self.table_id
    }
}

impl SinkConnector for ZyronTableSink {
    fn write_batch(&mut self, records: &[StreamRecord]) -> Result<()> {
        for record in records {
            let num_rows = record.num_rows();
            for i in 0..num_rows {
                match record.change_flags[i] {
                    ChangeFlag::Insert | ChangeFlag::UpdateAfter => {
                        self.rows_written += 1;
                    }
                    ChangeFlag::Delete | ChangeFlag::UpdateBefore => {
                        self.rows_deleted += 1;
                    }
                }
            }
            self.buffer.push(record.clone());
        }
        Ok(())
    }

    fn commit(&mut self) -> Result<()> {
        self.buffer.clear();
        Ok(())
    }

    fn rollback(&mut self) -> Result<()> {
        // Rollback: discard buffered records and undo counters.
        for record in &self.buffer {
            let num_rows = record.num_rows();
            for i in 0..num_rows {
                match record.change_flags[i] {
                    ChangeFlag::Insert | ChangeFlag::UpdateAfter => {
                        self.rows_written = self.rows_written.saturating_sub(1);
                    }
                    ChangeFlag::Delete | ChangeFlag::UpdateBefore => {
                        self.rows_deleted = self.rows_deleted.saturating_sub(1);
                    }
                }
            }
        }
        self.buffer.clear();
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        self.buffer.clear();
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// InMemorySink
// ---------------------------------------------------------------------------

/// In-memory sink for testing. Collects all output records into a
/// shared Vec behind a Mutex.
pub struct InMemorySink {
    output: Arc<parking_lot::Mutex<Vec<StreamRecord>>>,
}

impl InMemorySink {
    pub fn new() -> Self {
        Self {
            output: Arc::new(parking_lot::Mutex::new(Vec::new())),
        }
    }

    /// Returns a handle to the output buffer.
    pub fn output(&self) -> Arc<parking_lot::Mutex<Vec<StreamRecord>>> {
        Arc::clone(&self.output)
    }

    /// Returns total number of records stored.
    pub fn record_count(&self) -> usize {
        self.output.lock().len()
    }

    /// Returns total number of rows across all records.
    pub fn row_count(&self) -> usize {
        self.output.lock().iter().map(|r| r.num_rows()).sum()
    }
}

impl Default for InMemorySink {
    fn default() -> Self {
        Self::new()
    }
}

impl SinkConnector for InMemorySink {
    fn write_batch(&mut self, records: &[StreamRecord]) -> Result<()> {
        let mut output = self.output.lock();
        for record in records {
            output.push(record.clone());
        }
        Ok(())
    }

    fn commit(&mut self) -> Result<()> {
        Ok(())
    }

    fn rollback(&mut self) -> Result<()> {
        // Remove the last batch of records (simple rollback).
        let mut output = self.output.lock();
        output.pop();
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ZyronRowSink
// ---------------------------------------------------------------------------

/// Sink that inserts raw CDC row bytes into a ZyronDB target table through
/// the transaction manager and heap file. Runs an Insert privilege check
/// against the captured SecurityContext before opening a transaction.
///
/// write_batch signature differs from the SinkConnector trait because this
/// sink operates on CdfChange rows (one row_data payload per change),
/// not columnar StreamRecord batches.
pub struct ZyronRowSink {
    target_table_id: u32,
    write_mode: zyron_catalog::schema::CatalogStreamingWriteMode,
    catalog: Arc<zyron_catalog::Catalog>,
    heap: Arc<zyron_storage::HeapFile>,
    txn_manager: Arc<zyron_storage::txn::TransactionManager>,
    security_ctx: Arc<parking_lot::Mutex<zyron_auth::SecurityContext>>,
    security_manager: Arc<zyron_auth::SecurityManager>,
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
    ) -> Self {
        Self {
            target_table_id,
            write_mode,
            catalog,
            heap,
            txn_manager,
            security_ctx,
            security_manager,
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
                self.txn_manager.commit(&mut txn)?;
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::{StreamBatch, StreamColumn, StreamColumnData};
    use crate::record::ChangeFlag;

    fn make_test_record(n: usize) -> StreamRecord {
        let col = StreamColumn::from_data(StreamColumnData::Int64((0..n as i64).collect()));
        let batch = StreamBatch::new(vec![col]);
        let times: Vec<i64> = (0..n as i64).map(|i| i * 1000).collect();
        StreamRecord::new(batch, times, vec![ChangeFlag::Insert; n])
    }

    fn make_change_record() -> StreamRecord {
        let col = StreamColumn::from_data(StreamColumnData::Int64(vec![1, 2, 3]));
        let batch = StreamBatch::new(vec![col]);
        StreamRecord::new(
            batch,
            vec![1000, 2000, 3000],
            vec![
                ChangeFlag::Insert,
                ChangeFlag::Delete,
                ChangeFlag::UpdateAfter,
            ],
        )
    }

    #[test]
    fn test_in_memory_sink() {
        let mut sink = InMemorySink::new();
        let output = sink.output();

        let record = make_test_record(5);
        sink.write_batch(&[record]).expect("write should succeed");

        assert_eq!(sink.record_count(), 1);
        assert_eq!(sink.row_count(), 5);

        let records = output.lock();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].num_rows(), 5);
    }

    #[test]
    fn test_zyron_table_sink_append() {
        let mut sink = ZyronTableSink::new(100, WriteMode::Append);
        let record = make_test_record(3);
        sink.write_batch(&[record]).expect("write should succeed");

        assert_eq!(sink.rows_written(), 3);
        assert_eq!(sink.rows_deleted(), 0);

        sink.commit().expect("commit should succeed");
    }

    #[test]
    fn test_zyron_table_sink_with_changes() {
        let mut sink = ZyronTableSink::new(100, WriteMode::Upsert);
        let record = make_change_record();
        sink.write_batch(&[record]).expect("write should succeed");

        // Insert + UpdateAfter = 2 writes, Delete = 1 delete.
        assert_eq!(sink.rows_written(), 2);
        assert_eq!(sink.rows_deleted(), 1);
    }

    #[test]
    fn test_zyron_table_sink_rollback() {
        let mut sink = ZyronTableSink::new(100, WriteMode::Append);
        let record = make_test_record(5);
        sink.write_batch(&[record]).expect("write should succeed");
        assert_eq!(sink.rows_written(), 5);

        sink.rollback().expect("rollback should succeed");
        assert_eq!(sink.rows_written(), 0);
    }

    #[test]
    fn test_s3_sink_stub() {
        let mut sink = StreamS3Sink::new("my-bucket".into(), "prefix/".into(), "parquet".into());
        let record = make_test_record(3);
        sink.write_batch(&[record]).expect("write should succeed");
        assert_eq!(sink.records_written(), 3);
    }
}

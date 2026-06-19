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
// StreamS3Sink
// ---------------------------------------------------------------------------

/// Converts a column scalar to a JSON value. i128 is rendered as a string so it
/// survives JSON's f64-backed number range; binary is hex-encoded.
fn scalar_to_json(s: crate::column::ScalarValue) -> serde_json::Value {
    use crate::column::ScalarValue as S;
    use serde_json::Value;
    match s {
        S::Null => Value::Null,
        S::Boolean(b) => Value::Bool(b),
        S::Int8(v) => Value::from(v),
        S::Int16(v) => Value::from(v),
        S::Int32(v) => Value::from(v),
        S::Int64(v) => Value::from(v),
        S::Int128(v) => Value::String(v.to_string()),
        S::UInt8(v) => Value::from(v),
        S::UInt16(v) => Value::from(v),
        S::UInt32(v) => Value::from(v),
        S::UInt64(v) => Value::from(v),
        S::Float32(v) => serde_json::Number::from_f64(v as f64)
            .map(Value::Number)
            .unwrap_or(Value::Null),
        S::Float64(v) => serde_json::Number::from_f64(v)
            .map(Value::Number)
            .unwrap_or(Value::Null),
        S::Utf8(v) => Value::String(v),
        S::Binary(v) => Value::String(hex::encode(v)),
    }
}

fn s3_extension(format: &str) -> &'static str {
    match format.to_lowercase().as_str() {
        "jsonl" | "ndjson" => "jsonl",
        _ => "json",
    }
}

/// S3 sink connector. Buffers rows as JSON objects and, on commit, uploads the
/// buffer as a single object via the opendal S3 backend. Credentials resolve
/// from the standard AWS environment, region from AWS_REGION (default
/// us-east-1). Only JSON output is produced.
pub struct StreamS3Sink {
    bucket: String,
    prefix: String,
    format: String,
    buffer: Vec<serde_json::Value>,
    records_written: u64,
    seq: u64,
}

impl StreamS3Sink {
    pub fn new(bucket: String, prefix: String, format: String) -> Self {
        Self {
            bucket,
            prefix,
            format,
            buffer: Vec::new(),
            records_written: 0,
            seq: 0,
        }
    }

    pub fn records_written(&self) -> u64 {
        self.records_written
    }

    /// Serializes the buffered rows and uploads them as one S3 object.
    fn upload(&mut self) -> Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        let fmt = self.format.to_lowercase();
        let body: Vec<u8> = match fmt.as_str() {
            "jsonl" | "ndjson" => {
                let mut buf = Vec::new();
                for v in &self.buffer {
                    buf.extend_from_slice(serde_json::to_vec(v).unwrap_or_default().as_slice());
                    buf.push(b'\n');
                }
                buf
            }
            "json" => serde_json::to_vec(&self.buffer)
                .map_err(|e| ZyronError::StreamingError(format!("S3 JSON encode failed: {e}")))?,
            other => {
                return Err(ZyronError::StreamingError(format!(
                    "S3 sink output format \"{other}\" is not supported; use json or jsonl"
                )));
            }
        };

        let region = std::env::var("AWS_REGION")
            .or_else(|_| std::env::var("AWS_DEFAULT_REGION"))
            .unwrap_or_else(|_| "us-east-1".to_string());
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let ext = s3_extension(&self.format);
        let key = if self.prefix.is_empty() || self.prefix.ends_with('/') {
            format!("{}part-{:020}-{:010}.{}", self.prefix, ts, self.seq, ext)
        } else {
            format!("{}/part-{:020}-{:010}.{}", self.prefix, ts, self.seq, ext)
        };
        self.seq += 1;

        let bucket = self.bucket.clone();
        zyron_cdc::sink_io::block_on_io(async move {
            use opendal::Operator;
            use opendal::services::S3;
            let builder = S3::default().bucket(&bucket).region(&region);
            let op = Operator::new(builder)
                .map_err(|e| ZyronError::StreamingError(format!("S3 operator build failed: {e}")))?
                .finish();
            op.write(&key, body).await.map_err(|e| {
                ZyronError::StreamingError(format!("S3 write to {key} failed: {e}"))
            })?;
            Ok::<(), ZyronError>(())
        })?;

        self.buffer.clear();
        Ok(())
    }
}

impl SinkConnector for StreamS3Sink {
    fn write_batch(&mut self, records: &[StreamRecord]) -> Result<()> {
        for record in records {
            let num_cols = record.batch.num_columns();
            let num_rows = record.num_rows();
            for row in 0..num_rows {
                let mut obj = serde_json::Map::with_capacity(num_cols);
                for c in 0..num_cols {
                    let scalar = record.batch.column(c).get_scalar(row);
                    obj.insert(format!("c{c}"), scalar_to_json(scalar));
                }
                self.buffer.push(serde_json::Value::Object(obj));
            }
            self.records_written += num_rows as u64;
        }
        Ok(())
    }

    fn commit(&mut self) -> Result<()> {
        self.upload()
    }

    fn rollback(&mut self) -> Result<()> {
        self.buffer.clear();
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        self.upload()
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
                self.txn_manager.commit_blocking(&mut txn)?;
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
    fn test_s3_sink_buffers_rows() {
        // write_batch buffers rows without contacting S3; the upload happens on
        // commit/close. This exercises the buffering and counter path.
        let mut sink = StreamS3Sink::new("my-bucket".into(), "prefix/".into(), "json".into());
        let record = make_test_record(3);
        sink.write_batch(&[record]).expect("write should succeed");
        assert_eq!(sink.records_written(), 3);
        sink.rollback().expect("rollback clears the buffer");
    }
}

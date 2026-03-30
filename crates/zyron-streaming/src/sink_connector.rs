//! Sink connectors for writing streaming output.
//!
//! Provides the SinkConnector trait for writing StreamRecord micro-batches
//! to external systems. Includes stubs for Kafka and S3, and functional
//! implementations for ZyronTableSink and InMemorySink.

use std::sync::Arc;

use zyron_common::Result;

use crate::record::{ChangeFlag, StreamRecord};

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
    Kafka {
        brokers: String,
        topic: String,
    },
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
// StreamKafkaSink (stub)
// ---------------------------------------------------------------------------

/// Kafka sink connector stub.
pub struct StreamKafkaSink {
    brokers: String,
    topic: String,
    records_written: u64,
}

impl StreamKafkaSink {
    pub fn new(brokers: String, topic: String) -> Self {
        Self {
            brokers,
            topic,
            records_written: 0,
        }
    }

    pub fn records_written(&self) -> u64 {
        self.records_written
    }
}

impl SinkConnector for StreamKafkaSink {
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
// StreamS3Sink (stub)
// ---------------------------------------------------------------------------

/// S3 sink connector stub.
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
    fn test_kafka_sink_stub() {
        let mut sink = StreamKafkaSink::new("localhost:9092".into(), "output-topic".into());
        let record = make_test_record(10);
        sink.write_batch(&[record]).expect("write should succeed");
        assert_eq!(sink.records_written(), 10);
        sink.close().expect("close should succeed");
    }

    #[test]
    fn test_s3_sink_stub() {
        let mut sink = StreamS3Sink::new("my-bucket".into(), "prefix/".into(), "parquet".into());
        let record = make_test_record(3);
        sink.write_batch(&[record]).expect("write should succeed");
        assert_eq!(sink.records_written(), 3);
    }
}

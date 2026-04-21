// -----------------------------------------------------------------------------
// ExternalRowSink
// -----------------------------------------------------------------------------
//
// Writes rows to an external endpoint in one of the supported formats. Rows
// are buffered until a row count or approximate byte trigger is met, then a
// single object is written under the configured prefix with a monotonically
// increasing sequence number. Masking is applied upstream through the wire
// layer's lowering pass that rewrites projections into ExprSpec entries.
// This sink does not apply masking on its own.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use opendal::Operator;
use parking_lot::Mutex as PlMutex;

use crate::external_source::map_format;
use crate::format::{ColumnSpec, FormatKind, writer_for};
use crate::row_codec::StreamValue;
use zyron_catalog::schema::ExternalSinkEntry;
use zyron_common::{Result, ZyronError};

// -----------------------------------------------------------------------------
// Public type
// -----------------------------------------------------------------------------

pub struct ExternalRowSink {
    operator: Operator,
    uri_prefix: String,
    format: FormatKind,
    column_schema: Vec<ColumnSpec>,
    buffer: PlMutex<Vec<Vec<StreamValue>>>,
    size_trigger_rows: usize,
    size_trigger_bytes: usize,
    approx_buffer_bytes: PlMutex<usize>,
    seq: AtomicU64,
}

impl ExternalRowSink {
    /// Builds a sink from the catalog entry plus already-unsealed credentials.
    pub fn new(
        entry: &ExternalSinkEntry,
        credentials: HashMap<String, String>,
        column_schema: Vec<ColumnSpec>,
    ) -> Result<Self> {
        let (operator, prefix, _glob) = crate::external_source::build_operator_and_key_for_sink(
            entry.backend,
            &entry.uri,
            &entry.options,
            &credentials,
        )?;
        let opt_map: HashMap<&str, &str> = entry
            .options
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        let size_trigger_rows = opt_map
            .get("rows_per_file")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(10_000);
        let size_trigger_bytes = opt_map
            .get("bytes_per_file")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(5 * 1024 * 1024);

        Ok(Self {
            operator,
            uri_prefix: prefix,
            format: map_format(entry.format),
            column_schema,
            buffer: PlMutex::new(Vec::new()),
            size_trigger_rows,
            size_trigger_bytes,
            approx_buffer_bytes: PlMutex::new(0),
            seq: AtomicU64::new(0),
        })
    }

    /// Constructs the sink and discards it. No remote calls are made, only the
    /// OpenDAL operator and credential wiring are exercised. Used at server
    /// startup to detect misconfigured external sinks without forcing a remote
    /// round-trip.
    pub fn probe(entry: &ExternalSinkEntry, credentials: HashMap<String, String>) -> Result<()> {
        let _ = ExternalRowSink::new(entry, credentials, Vec::new())?;
        Ok(())
    }

    /// Appends a batch of rows to the internal buffer and flushes if any of
    /// the configured triggers are met.
    pub async fn write_batch(&self, rows: Vec<Vec<StreamValue>>) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        let added_bytes = approximate_bytes(&rows);
        {
            let mut buf = self.buffer.lock();
            buf.extend(rows);
            *self.approx_buffer_bytes.lock() += added_bytes;
        }

        let should_flush = {
            let buf = self.buffer.lock();
            let bytes = *self.approx_buffer_bytes.lock();
            buf.len() >= self.size_trigger_rows || bytes >= self.size_trigger_bytes
        };
        if should_flush {
            self.flush().await?;
        }
        Ok(())
    }

    /// Flushes the current buffer as a single object, regardless of trigger.
    pub async fn flush(&self) -> Result<()> {
        let drained: Vec<Vec<StreamValue>> = {
            let mut buf = self.buffer.lock();
            let out = std::mem::take(&mut *buf);
            *self.approx_buffer_bytes.lock() = 0;
            out
        };
        if drained.is_empty() {
            return Ok(());
        }

        let bytes = writer_for(self.format).write_rows(&drained, &self.column_schema)?;

        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let ext = extension_for(self.format);
        let key = if self.uri_prefix.is_empty() || self.uri_prefix.ends_with('/') {
            format!("{}part-{:020}-{:010}.{}", self.uri_prefix, ts, seq, ext)
        } else {
            format!("{}/part-{:020}-{:010}.{}", self.uri_prefix, ts, seq, ext)
        };

        self.operator
            .write(&key, bytes)
            .await
            .map_err(|e| ZyronError::StreamingError(format!("external write failed: {e}")))?;
        Ok(())
    }
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

fn extension_for(kind: FormatKind) -> &'static str {
    match kind {
        FormatKind::Json => "json",
        FormatKind::JsonLines => "jsonl",
        FormatKind::Csv => "csv",
        FormatKind::Parquet => "parquet",
        FormatKind::ArrowIpc => "arrow",
        FormatKind::Avro => "avro",
    }
}

// Rough lower-bound size estimate. Avoids serializing twice.
fn approximate_bytes(rows: &[Vec<StreamValue>]) -> usize {
    let mut total = 0usize;
    for row in rows {
        for v in row {
            total += match v {
                StreamValue::Null => 1,
                StreamValue::Bool(_) => 1,
                StreamValue::I64(_) => 8,
                StreamValue::I128(_) => 16,
                StreamValue::F64(_) => 8,
                StreamValue::Utf8(s) => s.len() + 2,
                StreamValue::Binary(b) => b.len() + 2,
            };
        }
    }
    total
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{FormatKind, reader_for};
    use opendal::services::Memory;
    use zyron_common::TypeId;

    fn schema() -> Vec<ColumnSpec> {
        vec![
            ColumnSpec {
                name: "id".into(),
                type_id: TypeId::Int64,
            },
            ColumnSpec {
                name: "name".into(),
                type_id: TypeId::Varchar,
            },
        ]
    }

    fn make_row(i: i64) -> Vec<StreamValue> {
        vec![StreamValue::I64(i), StreamValue::Utf8(format!("r{i}"))]
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn external_sink_jsonl_buffering() {
        let op = Operator::new(Memory::default()).unwrap().finish();
        let sink = ExternalRowSink {
            operator: op.clone(),
            uri_prefix: "out/".into(),
            format: FormatKind::JsonLines,
            column_schema: schema(),
            buffer: PlMutex::new(Vec::new()),
            size_trigger_rows: 50,
            size_trigger_bytes: usize::MAX,
            approx_buffer_bytes: PlMutex::new(0),
            seq: AtomicU64::new(0),
        };

        // Write 100 rows, 10 at a time, trigger at 50.
        for chunk in 0..10 {
            let rows: Vec<_> = (0..10).map(|i| make_row(chunk * 10 + i)).collect();
            sink.write_batch(rows).await.unwrap();
        }
        // Force any trailing buffer to flush.
        sink.flush().await.unwrap();

        // List all objects under out/ and verify two-or-more parts, total 100.
        let entries: Vec<opendal::Entry> = op.list("out/").await.unwrap();
        let mut total = 0usize;
        let mut object_count = 0usize;
        for e in entries {
            let path: String = e.path().to_string();
            if path.ends_with('/') {
                continue;
            }
            object_count += 1;
            let buf: opendal::Buffer = op.read(&path).await.unwrap();
            let bytes: Vec<u8> = buf.to_vec();
            let rows = reader_for(FormatKind::JsonLines)
                .read_rows(&bytes, &schema())
                .unwrap();
            total += rows.len();
        }
        assert_eq!(object_count, 2, "expected two part files at 50-row trigger");
        assert_eq!(total, 100);
    }
}

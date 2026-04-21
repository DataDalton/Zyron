// -----------------------------------------------------------------------------
// External-to-external COPY executor
// -----------------------------------------------------------------------------
//
// Implements the one-shot bulk transfer between two external endpoints for the
// COPY <source-external> TO <sink-external> form. This runs inline in the
// session, returns a CopyResult, and does not participate in any Zyron
// transaction because no Zyron table is read or written. Semantics are
// at-least-once: on mid-transfer failure the sink may contain a partial
// prefix of the rows that were streamed from the source before the error.

use std::collections::HashMap;
use std::time::Instant;

use zyron_catalog::schema::{
    CatalogClassification, ExternalBackend, ExternalFormat, ExternalMode, ExternalSinkEntry,
    ExternalSourceEntry,
};
use zyron_catalog::{ExternalSinkId, ExternalSourceId, SchemaId};
use zyron_common::{Result, ZyronError};

use crate::external_sink::ExternalRowSink;
use crate::external_source::ExternalTableSource;
use crate::format::ColumnSpec;

// -----------------------------------------------------------------------------
// Public types
// -----------------------------------------------------------------------------

/// Spec for one side of a COPY transfer. Used by the caller to describe both
/// the source and the sink endpoint. The caller is responsible for carrying
/// the column schema, whether that came from a named catalog entry, a target
/// table, or schema inference.
#[derive(Debug, Clone)]
pub struct CopyEndpoint {
    pub backend: ExternalBackend,
    pub uri: String,
    pub format: ExternalFormat,
    pub options: Vec<(String, String)>,
    pub credentials: HashMap<String, String>,
    pub columns: Vec<ColumnSpec>,
}

/// Row-count and byte-count summary returned by an external-to-external
/// transfer.
#[derive(Debug, Clone, Copy, Default)]
pub struct CopyResult {
    pub rows_read: u64,
    pub rows_written: u64,
    pub batches: u64,
    pub elapsed_ms: u64,
}

// -----------------------------------------------------------------------------
// Entry point
// -----------------------------------------------------------------------------

/// Streams rows from `source` into `sink` in batches of `batch_rows`. Returns
/// when the source is exhausted. The caller owns classification checks and
/// privilege enforcement. The sink is flushed before return so every accepted
/// row is durable in the remote store.
pub async fn run_external_to_external(
    source: CopyEndpoint,
    sink: CopyEndpoint,
    batch_rows: usize,
) -> Result<CopyResult> {
    if batch_rows == 0 {
        return Err(ZyronError::Internal(
            "external-to-external COPY batch size must be greater than zero".to_string(),
        ));
    }
    if source.columns.is_empty() {
        return Err(ZyronError::Internal(
            "external-to-external COPY requires a non-empty source column schema".to_string(),
        ));
    }
    if sink.columns.is_empty() {
        return Err(ZyronError::Internal(
            "external-to-external COPY requires a non-empty sink column schema".to_string(),
        ));
    }
    if source.columns.len() != sink.columns.len() {
        return Err(ZyronError::Internal(format!(
            "external-to-external COPY column count mismatch: source has {}, sink has {}",
            source.columns.len(),
            sink.columns.len()
        )));
    }

    let source_entry = source_entry_from_endpoint(&source);
    let reader = ExternalTableSource::new(&source_entry, source.credentials, source.columns)?;

    let sink_entry = sink_entry_from_endpoint(&sink);
    let writer = ExternalRowSink::new(&sink_entry, sink.credentials, sink.columns)?;

    let started = Instant::now();
    let mut result = CopyResult::default();

    loop {
        let batch = reader.read_batch(batch_rows)?;
        if batch.is_empty() {
            if reader.exhausted() {
                break;
            }
            // A non-exhausted source returning zero rows means the polling
            // source had no new objects. For OneShot inference mode used by
            // COPY this should not occur, so treat it as completion.
            break;
        }
        let batch_len = batch.len() as u64;
        result.rows_read += batch_len;
        writer.write_batch(batch).await?;
        result.rows_written += batch_len;
        result.batches += 1;
    }

    writer.flush().await?;
    result.elapsed_ms = started.elapsed().as_millis() as u64;
    Ok(result)
}

// -----------------------------------------------------------------------------
// Catalog-entry adapters
// -----------------------------------------------------------------------------

// Builds a transient ExternalSourceEntry so ExternalTableSource::new can be
// used without a persisted catalog row. Ingest mode is forced to OneShot
// because COPY does not run as a background job.
fn source_entry_from_endpoint(ep: &CopyEndpoint) -> ExternalSourceEntry {
    ExternalSourceEntry {
        id: ExternalSourceId(0),
        schema_id: SchemaId(0),
        name: String::new(),
        backend: ep.backend,
        uri: ep.uri.clone(),
        format: ep.format,
        mode: ExternalMode::OneShot,
        schedule_cron: None,
        options: ep.options.clone(),
        columns: Vec::new(),
        credential_key_id: None,
        credential_ciphertext: None,
        classification: CatalogClassification::Internal,
        tags: Vec::new(),
        owner_role_id: 0,
        created_at: 0,
    }
}

// Builds a transient ExternalSinkEntry so ExternalRowSink::new can be used
// without a persisted catalog row.
fn sink_entry_from_endpoint(ep: &CopyEndpoint) -> ExternalSinkEntry {
    ExternalSinkEntry {
        id: ExternalSinkId(0),
        schema_id: SchemaId(0),
        name: String::new(),
        backend: ep.backend,
        uri: ep.uri.clone(),
        format: ep.format,
        options: ep.options.clone(),
        columns: Vec::new(),
        credential_key_id: None,
        credential_ciphertext: None,
        classification: CatalogClassification::Internal,
        tags: Vec::new(),
        owner_role_id: 0,
        created_at: 0,
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{FormatKind, writer_for};
    use crate::row_codec::StreamValue;
    use zyron_common::TypeId;

    fn sample_columns() -> Vec<ColumnSpec> {
        vec![
            ColumnSpec {
                name: "id".to_string(),
                type_id: TypeId::Int64,
            },
            ColumnSpec {
                name: "label".to_string(),
                type_id: TypeId::Varchar,
            },
        ]
    }

    fn sample_rows() -> Vec<Vec<StreamValue>> {
        vec![
            vec![StreamValue::I64(1), StreamValue::Utf8("alpha".to_string())],
            vec![StreamValue::I64(2), StreamValue::Utf8("beta".to_string())],
            vec![StreamValue::I64(3), StreamValue::Utf8("gamma".to_string())],
        ]
    }

    // Writes `rows` to a temporary file in the requested format and returns
    // the absolute path. The path lives for the duration of the test process.
    fn materialise_source(format: FormatKind, rows: &[Vec<StreamValue>]) -> String {
        let columns = sample_columns();
        let bytes = writer_for(format).write_rows(rows, &columns).unwrap();
        let dir = std::env::temp_dir();
        let name = format!(
            "zyron_copy_src_{}_{}.bin",
            std::process::id(),
            uuid_like(),
        );
        let path = dir.join(name);
        std::fs::write(&path, &bytes).unwrap();
        path.to_string_lossy().into_owned()
    }

    fn reserve_sink_dir() -> String {
        let dir = std::env::temp_dir().join(format!(
            "zyron_copy_sink_{}_{}",
            std::process::id(),
            uuid_like(),
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir.to_string_lossy().into_owned()
    }

    // Monotonic counter derived from a process-local atomic. Avoids a uuid
    // dependency for tests.
    fn uuid_like() -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let next = COUNTER.fetch_add(1, Ordering::Relaxed);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        nanos.wrapping_add(next)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn runs_end_to_end_jsonl_to_jsonl() {
        // Share one working directory for both sides. The source file is
        // written at `src/<name>.jsonl` and the sink writes under `sink/`.
        let rows = sample_rows();
        let columns = sample_columns();
        let bytes = writer_for(FormatKind::JsonLines)
            .write_rows(&rows, &columns)
            .unwrap();

        let root = std::env::temp_dir().join(format!(
            "zyron_copy_root_{}_{}",
            std::process::id(),
            uuid_like(),
        ));
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::create_dir_all(root.join("sink")).unwrap();
        let src_rel = "src/data.jsonl";
        std::fs::write(root.join(src_rel), &bytes).unwrap();

        let root_str = root.to_string_lossy().replace('\\', "/");

        let source = CopyEndpoint {
            backend: ExternalBackend::File,
            uri: src_rel.to_string(),
            format: ExternalFormat::JsonLines,
            options: vec![("root".to_string(), root_str.clone())],
            credentials: HashMap::new(),
            columns: sample_columns(),
        };
        let sink = CopyEndpoint {
            backend: ExternalBackend::File,
            uri: "sink/".to_string(),
            format: ExternalFormat::JsonLines,
            options: vec![("root".to_string(), root_str.clone())],
            credentials: HashMap::new(),
            columns: sample_columns(),
        };
        let result = run_external_to_external(source, sink, 1024).await.unwrap();
        assert_eq!(result.rows_read, 3);
        assert_eq!(result.rows_written, 3);

        // The sink directory must now contain at least one file with the
        // expected number of lines.
        let sink_dir = root.join("sink");
        let mut total_lines = 0usize;
        for entry in std::fs::read_dir(&sink_dir).unwrap() {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_file() {
                let text = std::fs::read_to_string(entry.path()).unwrap();
                total_lines += text.lines().filter(|l| !l.trim().is_empty()).count();
            }
        }
        assert_eq!(total_lines, 3);
    }

    #[tokio::test]
    async fn rejects_column_count_mismatch() {
        let source = CopyEndpoint {
            backend: ExternalBackend::File,
            uri: "/tmp/does-not-exist".to_string(),
            format: ExternalFormat::JsonLines,
            options: Vec::new(),
            credentials: HashMap::new(),
            columns: sample_columns(),
        };
        let mut skewed = sample_columns();
        skewed.pop();
        let sink = CopyEndpoint {
            backend: ExternalBackend::File,
            uri: "/tmp/does-not-exist-2".to_string(),
            format: ExternalFormat::JsonLines,
            options: Vec::new(),
            credentials: HashMap::new(),
            columns: skewed,
        };
        let err = run_external_to_external(source, sink, 1024).await.unwrap_err();
        match err {
            ZyronError::Internal(msg) => assert!(msg.contains("column count mismatch")),
            other => panic!("unexpected error: {other:?}"),
        }
    }
}

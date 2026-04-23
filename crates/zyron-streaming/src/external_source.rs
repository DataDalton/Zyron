// -----------------------------------------------------------------------------
// ExternalTableSource
// -----------------------------------------------------------------------------
//
// Reads rows from an external endpoint (local FS, S3, GCS, Azure blob, HTTP)
// in one of the supported formats, yielding Vec<StreamValue> per row. Used by
// streaming jobs when the source is an EXTERNAL SOURCE catalog object. The
// runtime receives already-unsealed credentials. Credentials must not be
// logged or placed into error messages.

use std::collections::{HashMap, HashSet, VecDeque};

use globset::{Glob, GlobMatcher};
use opendal::{Operator, services};
use parking_lot::Mutex as PlMutex;

use crate::format::{ColumnSpec, FormatKind, reader_for};
use crate::row_codec::StreamValue;
use zyron_catalog::schema::{ExternalBackend, ExternalFormat, ExternalMode, ExternalSourceEntry};
use zyron_common::{Result, ZyronError};

// -----------------------------------------------------------------------------
// Public type
// -----------------------------------------------------------------------------

/// Source endpoint that reads rows from an external location. For glob URIs
/// the queue of matching objects is populated lazily on the first read. For
/// Watch mode, each call re-lists the prefix and processes new objects.
pub struct ExternalTableSource {
    operator: Operator,
    prefix: String,
    glob: Option<GlobMatcher>,
    format: FormatKind,
    column_schema: Vec<ColumnSpec>,
    mode: ExternalMode,
    file_queue: PlMutex<VecDeque<String>>,
    completed: PlMutex<HashSet<String>>,
    // Holds leftover rows for a partially drained file across calls.
    leftover: PlMutex<Vec<Vec<StreamValue>>>,
    // Becomes true when one-shot mode finishes draining the last object.
    exhausted: PlMutex<bool>,
    // For OneShot: true once the initial listing has been performed.
    listed_once: PlMutex<bool>,
}

impl ExternalTableSource {
    /// Constructs a source. The credential map is already unsealed. The
    /// column schema must match the target table the rows will be written
    /// into so the format layer can coerce cleanly.
    pub fn new(
        entry: &ExternalSourceEntry,
        credentials: HashMap<String, String>,
        column_schema: Vec<ColumnSpec>,
    ) -> Result<Self> {
        let (operator, prefix, glob) =
            build_operator_and_key(entry.backend, &entry.uri, &entry.options, &credentials)?;
        let format = map_format(entry.format);
        Ok(Self {
            operator,
            prefix,
            glob,
            format,
            column_schema,
            mode: entry.mode,
            file_queue: PlMutex::new(VecDeque::new()),
            completed: PlMutex::new(HashSet::new()),
            leftover: PlMutex::new(Vec::new()),
            exhausted: PlMutex::new(false),
            listed_once: PlMutex::new(false),
        })
    }

    /// Returns true once every matching object has been drained in OneShot
    /// mode. Watch mode never returns true from this method.
    pub fn exhausted(&self) -> bool {
        *self.exhausted.lock()
    }

    /// Reads up to max_rows rows from the source. For columnar formats the
    /// entire object is loaded and decoded in one call, leftover rows beyond
    /// max_rows stay buffered for the next call. For watch mode, each call
    /// re-lists the prefix to pick up new objects.
    pub fn read_batch(&self, max_rows: usize) -> Result<Vec<Vec<StreamValue>>> {
        if max_rows == 0 {
            return Ok(Vec::new());
        }

        // Serve from leftover buffer first.
        {
            let mut left = self.leftover.lock();
            if !left.is_empty() {
                let take = max_rows.min(left.len());
                let out: Vec<Vec<StreamValue>> = left.drain(0..take).collect();
                return Ok(out);
            }
        }

        // Make sure the queue has something to pop, or mark exhausted.
        self.refresh_queue()?;

        let next_key = {
            let mut q = self.file_queue.lock();
            q.pop_front()
        };
        let key = match next_key {
            Some(k) => k,
            None => return Ok(Vec::new()),
        };

        let rt = tokio::runtime::Handle::try_current();
        let bytes: Vec<u8> = match rt {
            Ok(handle) => {
                // Read via block_in_place if available, else use a short
                // blocking path with the operator's synchronous helper.
                let op = self.operator.clone();
                let k = key.clone();
                tokio::task::block_in_place(|| handle.block_on(async move { op.read(&k).await }))
                    .map_err(|e| ZyronError::StreamingError(format!("external read failed: {e}")))?
                    .to_vec()
            }
            Err(_) => {
                let op = self.operator.clone();
                let k = key.clone();
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|e| {
                        ZyronError::StreamingError(format!("failed to build runtime: {e}"))
                    })?;
                rt.block_on(async move { op.read(&k).await })
                    .map_err(|e| ZyronError::StreamingError(format!("external read failed: {e}")))?
                    .to_vec()
            }
        };

        let mut rows = reader_for(self.format).read_rows(&bytes, &self.column_schema)?;

        {
            let mut done = self.completed.lock();
            done.insert(key);
        }

        // Update exhausted flag for OneShot when queue is empty and no
        // leftover is staged after this batch.
        let returned = if rows.len() > max_rows {
            let leftover: Vec<_> = rows.split_off(max_rows);
            *self.leftover.lock() = leftover;
            rows
        } else {
            rows
        };

        self.maybe_mark_exhausted();
        Ok(returned)
    }

    // Refreshes the file queue based on the mode. OneShot only lists once.
    // Watch re-lists every call and diffs against the completed set.
    fn refresh_queue(&self) -> Result<()> {
        let should_list = match self.mode {
            ExternalMode::OneShot => !*self.listed_once.lock(),
            ExternalMode::Scheduled => !*self.listed_once.lock(),
            ExternalMode::Watch => true,
        };
        if !should_list {
            return Ok(());
        }

        // List via operator.
        let op = self.operator.clone();
        let prefix = self.prefix.clone();
        let rt = tokio::runtime::Handle::try_current();
        let entries: Vec<opendal::Entry> = match rt {
            Ok(handle) => tokio::task::block_in_place(|| {
                handle.block_on(async move { op.list(&prefix).await })
            })
            .map_err(|e| ZyronError::StreamingError(format!("external list failed: {e}")))?,
            Err(_) => {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|e| {
                        ZyronError::StreamingError(format!("failed to build runtime: {e}"))
                    })?;
                rt.block_on(async move { op.list(&prefix).await })
                    .map_err(|e| ZyronError::StreamingError(format!("external list failed: {e}")))?
            }
        };

        // Filter against the glob if present, skip directories, skip already
        // completed keys (Watch mode deduplication).
        let mut keys: Vec<String> = Vec::new();
        let completed = self.completed.lock();
        for e in entries {
            let path = e.path().to_string();
            if path.ends_with('/') {
                continue;
            }
            if let Some(g) = &self.glob {
                if !g.is_match(&path) {
                    continue;
                }
            }
            if completed.contains(&path) {
                continue;
            }
            keys.push(path);
        }
        drop(completed);
        keys.sort();
        let mut q = self.file_queue.lock();
        q.extend(keys);
        *self.listed_once.lock() = true;
        Ok(())
    }

    fn maybe_mark_exhausted(&self) {
        if !matches!(self.mode, ExternalMode::OneShot) {
            return;
        }
        let q_empty = self.file_queue.lock().is_empty();
        let l_empty = self.leftover.lock().is_empty();
        let listed = *self.listed_once.lock();
        if listed && q_empty && l_empty {
            *self.exhausted.lock() = true;
        }
    }
}

// -----------------------------------------------------------------------------
// Operator construction
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// Probe helpers
// -----------------------------------------------------------------------------

/// Verifies that an external source entry can be opened. Constructs the
/// ExternalTableSource and discards it. No remote calls are made, only the
/// OpenDAL operator and credential wiring are exercised. The column schema is
/// inferred from the entry options or left empty for probes that do not have
/// a target table available.
pub fn probe_external_source(
    entry: &ExternalSourceEntry,
    credentials: HashMap<String, String>,
) -> Result<()> {
    let _ = ExternalTableSource::new(entry, credentials, Vec::new())?;
    Ok(())
}

/// Opens the first file matching the URI pattern and returns the inferred
/// column list. Errors if the format does not support inference (JSON,
/// JSONL, and CSV have no embedded schema).
pub async fn infer_schema_from_first_file(
    entry: &ExternalSourceEntry,
    credentials: HashMap<String, String>,
) -> Result<Vec<ColumnSpec>> {
    let format = map_format(entry.format);
    if matches!(
        format,
        FormatKind::Json | FormatKind::JsonLines | FormatKind::Csv
    ) {
        return Err(ZyronError::PlanError(
            "schema inference requires a self-describing format (Parquet, Arrow IPC, or Avro)"
                .to_string(),
        ));
    }
    let (operator, prefix, glob) =
        build_operator_and_key(entry.backend, &entry.uri, &entry.options, &credentials)?;
    let entries = operator
        .list(&prefix)
        .await
        .map_err(|e| ZyronError::StreamingError(format!("external list failed: {e}")))?;
    let mut keys: Vec<String> = Vec::new();
    for e in entries {
        let path = e.path().to_string();
        if path.ends_with('/') {
            continue;
        }
        if let Some(g) = &glob {
            if !g.is_match(&path) {
                continue;
            }
        }
        keys.push(path);
    }
    keys.sort();
    let key = keys.into_iter().next().ok_or_else(|| {
        ZyronError::PlanError(
            "schema inference found no files matching the external source URI".to_string(),
        )
    })?;
    let bytes = operator
        .read(&key)
        .await
        .map_err(|e| ZyronError::StreamingError(format!("external read failed: {e}")))?
        .to_vec();
    crate::format::infer_schema(format, &bytes)
}

/// Shared sink-side wrapper so external_sink can reuse the same parser and
/// operator factory without cloning the code.
pub(crate) fn build_operator_and_key_for_sink(
    backend: ExternalBackend,
    uri: &str,
    options: &[(String, String)],
    credentials: &HashMap<String, String>,
) -> Result<(Operator, String, Option<GlobMatcher>)> {
    build_operator_and_key(backend, uri, options, credentials)
}

/// Parses the URI and builds an OpenDAL operator plus the prefix and
/// optional glob matcher. For s3/gcs/azblob URIs the bucket or container is
/// extracted from the URI and the remainder is used as the object prefix.
fn build_operator_and_key(
    backend: ExternalBackend,
    uri: &str,
    options: &[(String, String)],
    credentials: &HashMap<String, String>,
) -> Result<(Operator, String, Option<GlobMatcher>)> {
    let opt_map: HashMap<&str, &str> = options
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();

    match backend {
        ExternalBackend::File => {
            // file:///path or just /path. Use root configurable via options.
            let path = strip_scheme(uri, "file://").unwrap_or(uri).to_string();
            let root = opt_map.get("root").copied().unwrap_or("/").to_string();
            let builder = services::Fs::default().root(&root);
            let op = Operator::new(builder)
                .map_err(|e| ZyronError::StreamingError(format!("fs operator build: {e}")))?
                .finish();
            let (prefix, glob) = split_prefix_and_glob(&path);
            Ok((op, prefix, glob))
        }
        ExternalBackend::S3 => {
            let rest = strip_scheme(uri, "s3://")
                .ok_or_else(|| ZyronError::StreamingError("s3 uri missing scheme".to_string()))?;
            let (bucket, key_pat) = split_bucket_key(rest)?;
            let mut b = services::S3::default().bucket(bucket);
            if let Some(region) = opt_map.get("region").copied() {
                b = b.region(region);
            }
            if let Some(endpoint) = opt_map.get("endpoint").copied() {
                b = b.endpoint(endpoint);
            }
            if let Some(ak) = credentials.get("access_key_id") {
                b = b.access_key_id(ak);
            }
            if let Some(sk) = credentials.get("secret_access_key") {
                b = b.secret_access_key(sk);
            }
            if let Some(tok) = credentials.get("session_token") {
                b = b.session_token(tok);
            }
            let op = Operator::new(b)
                .map_err(|e| ZyronError::StreamingError(format!("s3 operator build: {e}")))?
                .finish();
            let (prefix, glob) = split_prefix_and_glob(key_pat);
            Ok((op, prefix, glob))
        }
        ExternalBackend::Gcs => {
            let rest = strip_scheme(uri, "gs://")
                .or_else(|| strip_scheme(uri, "gcs://"))
                .ok_or_else(|| ZyronError::StreamingError("gcs uri missing scheme".to_string()))?;
            let (bucket, key_pat) = split_bucket_key(rest)?;
            let mut b = services::Gcs::default().bucket(bucket);
            if let Some(sa) = credentials.get("service_account") {
                b = b.credential(sa);
            }
            let op = Operator::new(b)
                .map_err(|e| ZyronError::StreamingError(format!("gcs operator build: {e}")))?
                .finish();
            let (prefix, glob) = split_prefix_and_glob(key_pat);
            Ok((op, prefix, glob))
        }
        ExternalBackend::Azure => {
            let rest = strip_scheme(uri, "azblob://")
                .or_else(|| strip_scheme(uri, "az://"))
                .ok_or_else(|| {
                    ZyronError::StreamingError("azblob uri missing scheme".to_string())
                })?;
            let (container, key_pat) = split_bucket_key(rest)?;
            let mut b = services::Azblob::default().container(container);
            if let Some(account) = opt_map.get("account_name").copied() {
                b = b.account_name(account);
            }
            if let Some(endpoint) = opt_map.get("endpoint").copied() {
                b = b.endpoint(endpoint);
            }
            if let Some(key) = credentials.get("account_key") {
                b = b.account_key(key);
            }
            let op = Operator::new(b)
                .map_err(|e| ZyronError::StreamingError(format!("azblob operator build: {e}")))?
                .finish();
            let (prefix, glob) = split_prefix_and_glob(key_pat);
            Ok((op, prefix, glob))
        }
        ExternalBackend::Http => {
            let endpoint = opt_map
                .get("endpoint")
                .copied()
                .or_else(|| {
                    strip_scheme(uri, "http://")
                        .map(|_| uri)
                        .or_else(|| strip_scheme(uri, "https://").map(|_| uri))
                })
                .ok_or_else(|| {
                    ZyronError::StreamingError("http source needs endpoint".to_string())
                })?;
            let b = services::Http::default().endpoint(endpoint);
            let op = Operator::new(b)
                .map_err(|e| ZyronError::StreamingError(format!("http operator build: {e}")))?
                .finish();
            // For http, path portion after host serves as key.
            let path = opt_map.get("path").copied().unwrap_or("/").to_string();
            let (prefix, glob) = split_prefix_and_glob(&path);
            Ok((op, prefix, glob))
        }
        ExternalBackend::Zyron => Err(ZyronError::StreamingError(
            "zyron:// backend is handled by the Zyron source runtime, not OpenDAL".to_string(),
        )),
    }
}

fn strip_scheme<'a>(uri: &'a str, scheme: &str) -> Option<&'a str> {
    uri.strip_prefix(scheme)
}

/// Splits `bucket/rest/of/key` into `("bucket", "rest/of/key")`.
fn split_bucket_key(rest: &str) -> Result<(&str, &str)> {
    let trimmed = rest.trim_start_matches('/');
    match trimmed.find('/') {
        Some(idx) => Ok((&trimmed[..idx], &trimmed[idx + 1..])),
        None => Ok((trimmed, "")),
    }
}

/// Splits a key pattern into a list prefix (up to the first wildcard) and an
/// optional glob matcher applied to full keys returned by list.
fn split_prefix_and_glob(pattern: &str) -> (String, Option<GlobMatcher>) {
    let has_wild = pattern.contains('*') || pattern.contains('?') || pattern.contains('[');
    if !has_wild {
        return (pattern.to_string(), None);
    }
    let prefix_end = pattern
        .find(|c: char| c == '*' || c == '?' || c == '[')
        .unwrap_or(pattern.len());
    let prefix: String = pattern[..prefix_end]
        .rsplit_once('/')
        .map(|(p, _)| format!("{p}/"))
        .unwrap_or_else(|| String::new());
    let glob = Glob::new(pattern).ok().map(|g| g.compile_matcher());
    (prefix, glob)
}

// -----------------------------------------------------------------------------
// Format mapping
// -----------------------------------------------------------------------------

pub(crate) fn map_format(kind: ExternalFormat) -> FormatKind {
    match kind {
        ExternalFormat::Json => FormatKind::Json,
        ExternalFormat::JsonLines => FormatKind::JsonLines,
        ExternalFormat::Csv => FormatKind::Csv,
        ExternalFormat::Parquet => FormatKind::Parquet,
        ExternalFormat::ArrowIpc => FormatKind::ArrowIpc,
        ExternalFormat::Avro => FormatKind::Avro,
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{FormatKind, writer_for};
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

    fn rows() -> Vec<Vec<StreamValue>> {
        vec![
            vec![StreamValue::I64(1), StreamValue::Utf8("a".into())],
            vec![StreamValue::I64(2), StreamValue::Utf8("b".into())],
            vec![StreamValue::I64(3), StreamValue::Utf8("c".into())],
        ]
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn external_source_jsonl_oneshot() {
        // Write 3 rows of JSONL into an in-memory operator.
        let op = Operator::new(Memory::default()).unwrap().finish();
        let bytes = writer_for(FormatKind::JsonLines)
            .write_rows(&rows(), &schema())
            .unwrap();
        let buf: opendal::Buffer = opendal::Buffer::from(bytes);
        op.write("data/file.jsonl", buf).await.unwrap();

        // Hand-build a source that points at the same in-memory operator by
        // constructing the struct directly. Use a File entry so the builder
        // would not be hit, and inject the operator through a small helper.
        let source = ExternalTableSource {
            operator: op.clone(),
            prefix: "data/".into(),
            glob: Some(Glob::new("data/*.jsonl").unwrap().compile_matcher()),
            format: FormatKind::JsonLines,
            column_schema: schema(),
            mode: ExternalMode::OneShot,
            file_queue: PlMutex::new(VecDeque::new()),
            completed: PlMutex::new(HashSet::new()),
            leftover: PlMutex::new(Vec::new()),
            exhausted: PlMutex::new(false),
            listed_once: PlMutex::new(false),
        };
        let batch = source.read_batch(10).unwrap();
        assert_eq!(batch.len(), 3);
        assert!(source.exhausted());
    }
}

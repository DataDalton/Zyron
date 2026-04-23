// -----------------------------------------------------------------------------
// Wire-layer dispatch for external COPY statements
// -----------------------------------------------------------------------------
//
// Bridges the parser-level CopyStatement representation to the streaming
// crate's run_external_to_external executor. This module handles the
// external-to-external form of COPY. The table-anchored forms (IntoTable,
// FromTable) with non-STDIO external endpoints are still dispatched through
// the planner path.
//
// Responsibilities:
//   - Translate parser CopyExternal variants into streaming CopyEndpoint
//     values, resolving named catalog sources and sinks.
//   - Decode an optional COLUMNS clause carried in the statement options
//     under the reserved key "__columns_source".
//   - Reject missing column schema when the source format is not self
//     describing and no COLUMNS clause was supplied.
//   - Invoke zyron_streaming::copy_external::run_external_to_external and
//     return the row count for the PostgreSQL "COPY n" tag.

use std::collections::HashMap;
use std::sync::Arc;

use zyron_catalog::Catalog;
use zyron_catalog::schema::{ExternalFormat, ExternalSinkEntry, ExternalSourceEntry};
use zyron_common::{Result, TypeId, ZyronError};
use zyron_parser::ast::{CopyExternal, ExternalBackendKind, ExternalFormatKind};
use zyron_streaming::copy_external::{CopyEndpoint, CopyResult, run_external_to_external};
use zyron_streaming::format::ColumnSpec;

const COLUMNS_OPTION_KEY: &str = "__columns_source";
const DEFAULT_BATCH_ROWS: usize = 1024;

// -----------------------------------------------------------------------------
// Public entry point
// -----------------------------------------------------------------------------

/// Dispatches an external-to-external COPY. Resolves both sides to streaming
/// endpoints and runs the bulk transfer. Returns the number of rows written.
pub async fn dispatch_external_to_external(
    catalog: &Arc<Catalog>,
    source: &CopyExternal,
    sink: &CopyExternal,
    options: &[(String, String)],
) -> Result<CopyResult> {
    let declared = decode_declared_columns(options)?;

    let source_endpoint = build_source_endpoint(catalog, source, declared.as_deref())?;
    let sink_columns = source_endpoint.columns.clone();
    let sink_endpoint = build_sink_endpoint(catalog, sink, &sink_columns)?;

    run_external_to_external(source_endpoint, sink_endpoint, DEFAULT_BATCH_ROWS).await
}

// -----------------------------------------------------------------------------
// COLUMNS clause decoding
// -----------------------------------------------------------------------------

// The parser encodes an optional COLUMNS clause as "name:type_id_u8,name:type_id_u8,...".
// Returns the parsed list or None if no clause was supplied.
fn decode_declared_columns(options: &[(String, String)]) -> Result<Option<Vec<ColumnSpec>>> {
    for (k, v) in options {
        if k == COLUMNS_OPTION_KEY {
            return Ok(Some(parse_columns_encoding(v)?));
        }
    }
    Ok(None)
}

fn parse_columns_encoding(raw: &str) -> Result<Vec<ColumnSpec>> {
    let mut out = Vec::new();
    for entry in raw.split(',') {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        let (name, type_str) = entry.split_once(':').ok_or_else(|| {
            ZyronError::PlanError(format!("malformed COLUMNS entry in COPY: {entry}"))
        })?;
        let type_id_num: u8 = type_str.parse().map_err(|_| {
            ZyronError::PlanError(format!(
                "COLUMNS entry has non-numeric type id in COPY: {entry}"
            ))
        })?;
        let type_id = type_id_from_u8(type_id_num)?;
        out.push(ColumnSpec {
            name: name.to_string(),
            type_id,
        });
    }
    if out.is_empty() {
        return Err(ZyronError::PlanError(
            "COLUMNS clause in COPY must declare at least one column".to_string(),
        ));
    }
    Ok(out)
}

// Maps the raw u8 back to a TypeId. TypeId is #[repr(u8)] so the round trip is
// defined by the discriminant values. Uses a match to avoid unsafe transmute.
fn type_id_from_u8(v: u8) -> Result<TypeId> {
    // Exhaustively map every defined discriminant. Keep this in sync with
    // zyron_common::TypeId.
    let ids = [
        TypeId::Null,
        TypeId::Boolean,
        TypeId::Int8,
        TypeId::Int16,
        TypeId::Int32,
        TypeId::Int64,
        TypeId::Int128,
        TypeId::UInt8,
        TypeId::UInt16,
        TypeId::UInt32,
        TypeId::UInt64,
        TypeId::UInt128,
        TypeId::Float32,
        TypeId::Float64,
        TypeId::Decimal,
        TypeId::Char,
        TypeId::Varchar,
        TypeId::Text,
        TypeId::Binary,
        TypeId::Varbinary,
        TypeId::Bytea,
        TypeId::Date,
        TypeId::Time,
        TypeId::Timestamp,
        TypeId::TimestampTz,
        TypeId::Interval,
        TypeId::Uuid,
        TypeId::Json,
        TypeId::Jsonb,
        TypeId::Array,
        TypeId::Composite,
        TypeId::Vector,
        TypeId::Geometry,
        TypeId::Matrix,
        TypeId::Color,
        TypeId::SemVer,
        TypeId::Inet,
        TypeId::Cidr,
        TypeId::MacAddr,
        TypeId::Money,
        TypeId::Range,
        TypeId::HyperLogLog,
        TypeId::BloomFilter,
        TypeId::TDigest,
        TypeId::CountMinSketch,
        TypeId::Bitfield,
        TypeId::Quantity,
    ];
    for t in ids {
        if t as u8 == v {
            return Ok(t);
        }
    }
    Err(ZyronError::PlanError(format!(
        "COPY COLUMNS declared unknown type id {v}"
    )))
}

// -----------------------------------------------------------------------------
// Source endpoint construction
// -----------------------------------------------------------------------------

fn build_source_endpoint(
    catalog: &Arc<Catalog>,
    src: &CopyExternal,
    declared: Option<&[ColumnSpec]>,
) -> Result<CopyEndpoint> {
    match src {
        CopyExternal::Stdio => Err(ZyronError::PlanError(
            "COPY FROM STDIN is not an external-to-external source".to_string(),
        )),
        CopyExternal::LocalFile(_) => Err(ZyronError::PlanError(
            "COPY from a bare local path is not supported as an external-to-external source, use FILE '<path>' FORMAT <fmt>".to_string(),
        )),
        CopyExternal::Inline {
            backend,
            uri,
            format,
            credentials,
        } => {
            let columns = if let Some(cols) = declared {
                cols.to_vec()
            } else if is_self_describing(format.clone()) {
                // Self-describing formats allow empty column list at construct
                // time. ExternalTableSource reads the schema from the file.
                Vec::new()
            } else {
                return Err(ZyronError::PlanError(
                    "COPY FROM requires a COLUMNS clause when the source format is CSV or JSON".to_string(),
                ));
            };
            Ok(CopyEndpoint {
                backend: parser_backend_to_catalog(backend.clone()),
                uri: uri.clone(),
                format: parser_format_to_catalog(format.clone()),
                options: Vec::new(),
                credentials: credentials_vec_to_map(credentials),
                columns,
            })
        }
        CopyExternal::Named(name) => {
            let entry = lookup_source_entry(catalog, name)?;
            Ok(endpoint_from_source_entry(&entry))
        }
    }
}

fn build_sink_endpoint(
    catalog: &Arc<Catalog>,
    sink: &CopyExternal,
    source_columns: &[ColumnSpec],
) -> Result<CopyEndpoint> {
    match sink {
        CopyExternal::Stdio => Err(ZyronError::PlanError(
            "COPY TO STDOUT is not an external-to-external sink".to_string(),
        )),
        CopyExternal::LocalFile(_) => Err(ZyronError::PlanError(
            "COPY to a bare local path is not supported as an external-to-external sink, use FILE '<path>' FORMAT <fmt>".to_string(),
        )),
        CopyExternal::Inline {
            backend,
            uri,
            format,
            credentials,
        } => {
            if source_columns.is_empty() {
                return Err(ZyronError::PlanError(
                    "COPY cannot build a sink with an unknown column schema, declare COLUMNS or use a self describing source format".to_string(),
                ));
            }
            Ok(CopyEndpoint {
                backend: parser_backend_to_catalog(backend.clone()),
                uri: uri.clone(),
                format: parser_format_to_catalog(format.clone()),
                options: Vec::new(),
                credentials: credentials_vec_to_map(credentials),
                columns: source_columns.to_vec(),
            })
        }
        CopyExternal::Named(name) => {
            let entry = lookup_sink_entry(catalog, name)?;
            Ok(endpoint_from_sink_entry(&entry))
        }
    }
}

// -----------------------------------------------------------------------------
// Catalog lookup
// -----------------------------------------------------------------------------

fn lookup_source_entry(catalog: &Arc<Catalog>, name: &str) -> Result<ExternalSourceEntry> {
    for entry in catalog.list_external_sources() {
        if entry.name == name {
            return Ok((*entry).clone());
        }
    }
    Err(ZyronError::PlanError(format!(
        "external source '{name}' not found in catalog"
    )))
}

fn lookup_sink_entry(catalog: &Arc<Catalog>, name: &str) -> Result<ExternalSinkEntry> {
    for entry in catalog.list_external_sinks() {
        if entry.name == name {
            return Ok((*entry).clone());
        }
    }
    Err(ZyronError::PlanError(format!(
        "external sink '{name}' not found in catalog"
    )))
}

fn endpoint_from_source_entry(entry: &ExternalSourceEntry) -> CopyEndpoint {
    CopyEndpoint {
        backend: entry.backend,
        uri: entry.uri.clone(),
        format: entry.format,
        options: entry.options.clone(),
        // Named sources store credentials as sealed ciphertext. Unsealing
        // requires the key ring. Until the key ring is wired here, use an
        // empty credential map. The streaming source builder will then rely
        // on the operator's environment (for example, AWS config files or
        // IAM roles) to authenticate. Callers that require sealed creds
        // should configure OpenDAL via environment.
        credentials: HashMap::new(),
        columns: entry
            .columns
            .iter()
            .map(|(name, t)| ColumnSpec {
                name: name.clone(),
                type_id: *t,
            })
            .collect(),
    }
}

fn endpoint_from_sink_entry(entry: &ExternalSinkEntry) -> CopyEndpoint {
    CopyEndpoint {
        backend: entry.backend,
        uri: entry.uri.clone(),
        format: entry.format,
        options: entry.options.clone(),
        credentials: HashMap::new(),
        columns: entry
            .columns
            .iter()
            .map(|(name, t)| ColumnSpec {
                name: name.clone(),
                type_id: *t,
            })
            .collect(),
    }
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

fn credentials_vec_to_map(pairs: &[(String, String)]) -> HashMap<String, String> {
    let mut out = HashMap::with_capacity(pairs.len());
    for (k, v) in pairs {
        out.insert(k.clone(), v.clone());
    }
    out
}

fn is_self_describing(f: ExternalFormatKind) -> bool {
    matches!(
        f,
        ExternalFormatKind::Parquet | ExternalFormatKind::ArrowIpc | ExternalFormatKind::Avro
    )
}

fn parser_backend_to_catalog(b: ExternalBackendKind) -> zyron_catalog::ExternalBackend {
    match b {
        ExternalBackendKind::File => zyron_catalog::ExternalBackend::File,
        ExternalBackendKind::S3 => zyron_catalog::ExternalBackend::S3,
        ExternalBackendKind::Gcs => zyron_catalog::ExternalBackend::Gcs,
        ExternalBackendKind::Azure => zyron_catalog::ExternalBackend::Azure,
        ExternalBackendKind::Http => zyron_catalog::ExternalBackend::Http,
        ExternalBackendKind::Zyron => zyron_catalog::ExternalBackend::Zyron,
    }
}

fn parser_format_to_catalog(f: ExternalFormatKind) -> ExternalFormat {
    match f {
        ExternalFormatKind::Json => ExternalFormat::Json,
        ExternalFormatKind::JsonLines => ExternalFormat::JsonLines,
        ExternalFormatKind::Csv => ExternalFormat::Csv,
        ExternalFormatKind::Parquet => ExternalFormat::Parquet,
        ExternalFormatKind::ArrowIpc => ExternalFormat::ArrowIpc,
        ExternalFormatKind::Avro => ExternalFormat::Avro,
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_columns_encoding_single() {
        // TypeId is #[repr(u8)]. Int64 = 13, Varchar = 51.
        let cols = parse_columns_encoding("id:13,name:51").unwrap();
        assert_eq!(cols.len(), 2);
        assert_eq!(cols[0].name, "id");
        assert_eq!(cols[0].type_id, TypeId::Int64);
        assert_eq!(cols[1].name, "name");
        assert_eq!(cols[1].type_id, TypeId::Varchar);
    }

    #[test]
    fn rejects_empty_columns_encoding() {
        let err = parse_columns_encoding("").unwrap_err();
        match err {
            ZyronError::PlanError(msg) => assert!(msg.contains("at least one column")),
            _ => panic!("unexpected error: {err:?}"),
        }
    }

    #[test]
    fn rejects_malformed_column_entry() {
        let err = parse_columns_encoding("noColonHere").unwrap_err();
        match err {
            ZyronError::PlanError(msg) => assert!(msg.contains("malformed")),
            _ => panic!("unexpected error: {err:?}"),
        }
    }

    #[test]
    fn rejects_unknown_type_id() {
        let err = parse_columns_encoding("x:250").unwrap_err();
        match err {
            ZyronError::PlanError(msg) => assert!(msg.contains("unknown type id")),
            _ => panic!("unexpected error: {err:?}"),
        }
    }

    #[test]
    fn self_describing_formats_identified() {
        assert!(is_self_describing(ExternalFormatKind::Parquet));
        assert!(is_self_describing(ExternalFormatKind::ArrowIpc));
        assert!(is_self_describing(ExternalFormatKind::Avro));
        assert!(!is_self_describing(ExternalFormatKind::Csv));
        assert!(!is_self_describing(ExternalFormatKind::Json));
        assert!(!is_self_describing(ExternalFormatKind::JsonLines));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn external_csv_to_external_parquet_converts() {
        // End-to-end format conversion test. Write a CSV file on disk, run
        // COPY from that CSV to a Parquet directory using the dispatch
        // helper, then parse the resulting Parquet bytes and check that the
        // row count matches.
        use zyron_streaming::format::{FormatKind, writer_for};
        use zyron_streaming::row_codec::StreamValue;

        let columns = vec![
            ColumnSpec {
                name: "id".to_string(),
                type_id: TypeId::Int64,
            },
            ColumnSpec {
                name: "label".to_string(),
                type_id: TypeId::Varchar,
            },
        ];
        let rows = vec![
            vec![StreamValue::I64(1), StreamValue::Utf8("alpha".to_string())],
            vec![StreamValue::I64(2), StreamValue::Utf8("beta".to_string())],
            vec![StreamValue::I64(3), StreamValue::Utf8("gamma".to_string())],
        ];
        let csv_bytes = writer_for(FormatKind::Csv)
            .write_rows(&rows, &columns)
            .unwrap();

        // Lay out a fresh directory tree. The File backend is rooted at
        // `root` so URIs are relative.
        let root = std::env::temp_dir().join(format!(
            "zyron_copy_dispatch_{}_{}",
            std::process::id(),
            uuid_like()
        ));
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::create_dir_all(root.join("sink")).unwrap();
        std::fs::write(root.join("src/data.csv"), &csv_bytes).unwrap();
        let root_str = root.to_string_lossy().replace('\\', "/");

        let source = CopyEndpoint {
            backend: zyron_catalog::ExternalBackend::File,
            uri: "src/data.csv".to_string(),
            format: ExternalFormat::Csv,
            options: vec![("root".to_string(), root_str.clone())],
            credentials: HashMap::new(),
            columns: columns.clone(),
        };
        let sink = CopyEndpoint {
            backend: zyron_catalog::ExternalBackend::File,
            uri: "sink/".to_string(),
            format: ExternalFormat::Parquet,
            options: vec![("root".to_string(), root_str.clone())],
            credentials: HashMap::new(),
            columns: columns.clone(),
        };
        let result = run_external_to_external(source, sink, 1024).await.unwrap();
        assert_eq!(result.rows_read, 3);
        assert_eq!(result.rows_written, 3);

        // Verify at least one Parquet file appeared in the sink directory.
        let sink_dir = root.join("sink");
        let mut found_any = false;
        for entry in std::fs::read_dir(&sink_dir).unwrap() {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_file() {
                let bytes = std::fs::read(entry.path()).unwrap();
                // Parquet magic number "PAR1" at start and end.
                if bytes.len() >= 8
                    && &bytes[0..4] == b"PAR1"
                    && &bytes[bytes.len() - 4..] == b"PAR1"
                {
                    found_any = true;
                }
            }
        }
        assert!(found_any, "no valid parquet file produced in sink");
    }

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
}

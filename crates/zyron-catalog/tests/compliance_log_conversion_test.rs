//! The compliance audit log as a verified table.
//!
//! The log used to carry a CRC32 chain in two fields of every row. CRC32 is
//! a checksum rather than a one-way function, so anyone who could rewrite an
//! entry could recompute every hash after it, and the chain stated only that
//! nobody had corrupted a row by accident. The log is a table now, immutable
//! and verified, and what states that it is whole is the commit chain over
//! it.
//!
//! These tests cover the conversion: that the table is registered with both
//! protections, that its rows keep every field they had except the two
//! hashes, that the migration step is registered against the table with a
//! description saying what the old chain was worth, and that nothing in the
//! codebase computes a chain of its own any more.
//!
//! Run: cargo test -p zyron-catalog --test compliance_log_conversion_test -- --nocapture

use std::sync::Arc;

use zyron_catalog::schema::ComplianceLogEntry;
use zyron_catalog::system_catalog::{COMPLIANCE_SCHEMA, SYSTEM_CATALOG_NAME, SystemCatalog};
use zyron_catalog::{Catalog, CatalogCache, HeapCatalogStorage};
use zyron_common::format::catalog_evolution::CatalogSchemaRegistry;
use zyron_storage::DiskManager;
use zyron_wal::WalWriter;

/// The table the log's rows are held under
const LOG_TABLE: &str = "zyron_sys.compliance.log";

async fn open_catalog(tmp: &tempfile::TempDir) -> Arc<Catalog> {
    let (data_dir, wal_dir) = zyron_bench_harness::create_dirs(tmp.path()).expect("dirs");
    let wal = Arc::new(WalWriter::new(zyron_bench_harness::wal_config(&wal_dir)).expect("wal"));
    let disk = Arc::new(
        DiskManager::new(zyron_bench_harness::disk_config(&data_dir))
            .await
            .expect("disk"),
    );
    let pool = Arc::new(zyron_buffer::BufferPool::new(
        zyron_bench_harness::buffer_pool_config(),
    ));
    zyron_bench_harness::install_evict_writer(&pool, &disk, Some(&wal));
    let storage =
        Arc::new(HeapCatalogStorage::new(Arc::clone(&disk), Arc::clone(&pool)).expect("storage"));
    let cache = Arc::new(CatalogCache::new(256, 64));
    let catalog = Arc::new(
        Catalog::new(storage, cache, Arc::clone(&wal))
            .await
            .expect("catalog"),
    );
    SystemCatalog::init(&catalog)
        .await
        .expect("the system catalog is registered");
    catalog
}

fn log_entry(catalog: &Catalog) -> Arc<zyron_catalog::TableEntry> {
    let database = catalog
        .get_database(SYSTEM_CATALOG_NAME)
        .expect("the system catalog exists");
    let schema = catalog
        .get_schema(database.id, COMPLIANCE_SCHEMA)
        .expect("the compliance schema exists");
    catalog
        .compliance_log_entry(schema.id)
        .expect("the compliance log is registered as a table")
}

/// The log is a table of the system catalog, and it carries both
/// protections: immutable, so no path rewrites or removes an event, and
/// verified, so the commit that appends one extends a chain
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_compliance_log_is_an_immutable_verified_table() {
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let catalog = open_catalog(&tmp).await;
    let table = log_entry(&catalog);

    assert!(
        table.lifecycle.immutable,
        "the log refuses UPDATE and DELETE"
    );
    assert!(table.lifecycle.verified, "the log carries a commit chain");
    assert_eq!(
        table.lifecycle.chain_algorithm,
        zyron_lifecycle::verify::DEFAULT_CHAIN_ALGORITHM,
        "the chain links with the scheme a chain is started with"
    );
    assert_ne!(
        table.heap_file_id, 0,
        "the table names the heap its rows are in"
    );
}

/// The rows keep every field they carried except the two the chain now
/// holds, so an auditor reading an entry reads the same event as before
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_rows_keep_every_column_except_the_two_hash_fields() {
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let catalog = open_catalog(&tmp).await;
    let table = log_entry(&catalog);

    let names: Vec<&str> = table.columns.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(
        names,
        vec![
            "event_id",
            "event_type",
            "subject",
            "table_id",
            "ts",
            "detail",
            "record_version",
        ]
    );
    assert!(
        !names.contains(&"prev_hash") && !names.contains(&"entry_hash"),
        "the two hash fields are gone, the chain holds what they held"
    );

    // The encoding carries exactly those fields and round-trips
    let entry = ComplianceLogEntry {
        event_id: 4,
        event_type: zyron_lifecycle::compliance::event::LEGAL_HOLD,
        subject: "h1".to_string(),
        table_id: 9,
        ts: 5_000,
        detail: "create hold".to_string(),
        record_version: zyron_lifecycle::format::AUDIT_RECORD_VERSION_BYTE,
    };
    assert_eq!(
        ComplianceLogEntry::from_bytes(&entry.to_bytes()).expect("decodes"),
        entry
    );
}

/// Every event appended reads back, under an id of its own, in the order it
/// happened
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn appended_events_read_back_in_order() {
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let catalog = open_catalog(&tmp).await;
    for n in 1..=5u64 {
        catalog
            .append_compliance_log(ComplianceLogEntry {
                event_id: 0,
                event_type: zyron_lifecycle::compliance::event::TTL,
                subject: format!("s{n}"),
                table_id: n as u32,
                ts: 1_000 + n as i64,
                detail: format!("event {n}"),
                record_version: zyron_lifecycle::format::AUDIT_RECORD_VERSION_BYTE,
            })
            .await
            .expect("appends");
    }
    let log = catalog.load_compliance_log().await.expect("loads");
    assert_eq!(log.len(), 5);
    for (index, entry) in log.iter().enumerate() {
        assert_eq!(entry.event_id, index as u64 + 1);
        assert_eq!(entry.subject, format!("s{}", index + 1));
        assert_eq!(entry.detail, format!("event {}", index + 1));
    }
}

/// The conversion is a registered catalog schema step against the log's own
/// table, and its description records what the chain it removed was worth
#[test]
fn the_conversion_is_a_registered_step_that_says_what_the_old_chain_was() {
    let registry = CatalogSchemaRegistry::load().expect("the registry loads");
    let table = registry
        .table(LOG_TABLE)
        .expect("the log is a registered catalog table");
    assert_eq!(
        table.registration.current_schema_version,
        zyron_catalog::catalog_schema::COMPLIANCE_LOG_SCHEMA_VERSION
    );
    assert_eq!(table.steps.len(), 1, "one step, from the shape before it");
    let step = table.steps[0];
    assert!(
        !step.reversible,
        "the hashes cannot be put back, because the entries written after the conversion \
         never had them"
    );
    let description = step.description.to_lowercase();
    assert!(description.contains("crc32"), "{description}");
    assert!(
        description.contains("checksum rather than evidence"),
        "the step records what the chain it removed was worth: {description}"
    );
    assert!(
        step.migration_function_ref
            .contains("drop_compliance_log_hashes")
    );
}

/// Nothing in the workspace computes a chain of its own.
///
/// The phase exists to leave exactly one chain implementation, so the two
/// fields, the function that hashed them and the walk that verified them are
/// gone, and nothing calls the checksum they used for chaining
#[test]
fn exactly_one_chain_implementation_exists_in_the_workspace() {
    let root = workspace_root();
    let mut offenders: Vec<String> = Vec::new();
    let mut chain_definitions = 0usize;

    // The names the deleted surface went by. This file is the one place
    // they are written down, so it names itself out of the search
    const DELETED: &[&str] = &[
        "verify_compliance_chain",
        "ComplianceLogEntry::compute_hash",
        "AuditChain",
        "audit_chain",
    ];

    for path in rust_sources(&root.join("crates")) {
        let text = std::fs::read_to_string(&path).unwrap_or_default();
        let shown = path
            .strip_prefix(&root)
            .unwrap_or(&path)
            .display()
            .to_string();
        // This file is where the names being searched for are written
        // down, so it is not searched
        if shown.ends_with("compliance_log_conversion_test.rs") {
            continue;
        }
        // The one place a chain entry's link is computed. That file is the
        // chain, so its own fields are what the count below asserts rather
        // than what the search below forbids
        if text.contains("fn compute_entry_hash") {
            chain_definitions += 1;
            continue;
        }
        for gone in DELETED {
            if text.contains(gone) {
                offenders.push(format!("{shown} still names {gone}"));
            }
        }
        for line in text.lines() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("//") || trimmed.starts_with("///") {
                continue;
            }
            // A compliance entry no longer carries a hash of its own, so
            // nothing sets one when it builds a row
            if trimmed.starts_with("prev_hash:") || trimmed.starts_with("entry_hash:") {
                offenders.push(format!("{shown} builds a row with a hash field: {trimmed}"));
            }
            // The checksum the old chain linked with. Still the right tool
            // for a page or a payload, never for linking one record to the
            // one before it
            if trimmed.contains("hash32(")
                && (trimmed.contains("prev") || trimmed.contains("chain"))
            {
                offenders.push(format!("{shown} chains with a checksum: {trimmed}"));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "the deleted chain survives in {} place(s):\n  {}",
        offenders.len(),
        offenders.join("\n  ")
    );
    assert_eq!(
        chain_definitions, 1,
        "exactly one chain implementation computes a link, found {chain_definitions}"
    );
}

/// The workspace root, from this crate's manifest directory
fn workspace_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|crates| crates.parent())
        .map(|root| root.to_path_buf())
        .expect("the crate sits two levels under the workspace root")
}

/// Every `.rs` file under a directory
fn rust_sources(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            out.extend(rust_sources(&path));
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
    out
}

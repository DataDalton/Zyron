//! Deployment-mode gating of the lake tier at startup.
//!
//! A `db` node must open no lake transaction log and register no shared
//! state, which is the whole reason to run that mode: the lake tier costs it
//! neither startup IO nor resident memory. A `lake` or `unified` node must
//! reconcile every lake table's log before any query runs.
//!
//! Run: cargo test -p zyron-server --test lake_deployment_mode_test

use std::sync::Arc;

use zyron_buffer::{BufferPool, BufferPoolConfig};
use zyron_catalog::storage::HeapCatalogStorage;
use zyron_catalog::{Catalog, CatalogCache, SYSTEM_DATABASE_ID};
use zyron_common::DeploymentMode;
use zyron_lake::{AllCommitted, LakePaths, LakeSchema, TransactionLog};
use zyron_parser::ast::{ColumnDef, DataType};
use zyron_server::lake_recovery::recover_lake_logs;
use zyron_storage::{DiskManager, DiskManagerConfig};
use zyron_wal::{WalWriter, WalWriterConfig};

struct Harness {
    catalog: Arc<Catalog>,
    data_dir: std::path::PathBuf,
    _tmp: tempfile::TempDir,
}

/// Builds a catalog holding one heap table and one lake table whose log
/// exists on disk at version one, exactly the shape recovery walks.
async fn harness_with_one_lake_table() -> (Harness, LakePaths) {
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let data_dir = tmp.path().join("data");
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir_all(&data_dir).unwrap();
    std::fs::create_dir_all(&wal_dir).unwrap();

    let wal = Arc::new(WalWriter::new(zyron_bench_harness::wal_config(wal_dir)).expect("wal"));
    let disk = Arc::new(
        DiskManager::new(zyron_bench_harness::disk_config(data_dir.clone()))
            .await
            .expect("disk"),
    );
    let pool = Arc::new(BufferPool::new(zyron_bench_harness::buffer_pool_config()));
    let storage =
        Arc::new(HeapCatalogStorage::new(Arc::clone(&disk), Arc::clone(&pool)).expect("storage"));
    let catalog = Arc::new(
        Catalog::new(
            storage,
            Arc::new(CatalogCache::new(64, 32)),
            Arc::clone(&wal),
        )
        .await
        .expect("catalog"),
    );
    let schema_id = catalog
        .create_schema(SYSTEM_DATABASE_ID, "public", "test_user")
        .await
        .expect("schema");

    let columns = vec![ColumnDef {
        name: "id".into(),
        data_type: DataType::BigInt,
        nullable: Some(true),
        default: None,
        constraints: vec![],
    }];
    catalog
        .create_table(schema_id, "heap_only", &columns, &[])
        .await
        .expect("heap table");
    catalog
        .create_table(schema_id, "lake_one", &columns, &[])
        .await
        .expect("lake table");

    let table = catalog.get_table(schema_id, "lake_one").expect("entry");
    let mut entry = (*table).clone();
    let lake_columns = entry
        .columns
        .iter()
        .map(|c| zyron_lake::LakeColumn {
            id: c.id.0 as u32,
            name: c.name.clone(),
            type_id: c.type_id,
            nullable: c.nullable,
            fractional_digits: c.fractional_digits,
            tz_offset_secs: c.tz_offset_secs,
            max_length: c.max_length.map(|n| n as u32),
            default_expr: c.default_expr.clone(),
        })
        .collect();
    let lake_schema = LakeSchema::new(1, lake_columns).expect("lake schema");
    let paths = LakePaths::new(&data_dir, entry.id.0);
    let log = TransactionLog::create(
        paths.clone(),
        zyron_lake::CommitAttempt {
            operation: zyron_lake::OperationKind::SchemaChange,
            db_txn_id: 0,
            commit_lsn: 0,
            timestamp_us: 1,
            read_predicate: None,
            audit: None,
        },
        &lake_schema,
        None,
        &std::collections::BTreeMap::new(),
    )
    .expect("create lake log");
    drop(log);

    entry.lake = zyron_catalog::schema::LakeConfig::lake();
    catalog.update_table(entry).await.expect("flip lake flag");

    (
        Harness {
            catalog,
            data_dir,
            _tmp: tmp,
        },
        paths,
    )
}

#[tokio::test]
async fn test_db_node_starts_no_lake_workers_or_caches() {
    let (h, paths) = harness_with_one_lake_table().await;

    let report = recover_lake_logs(DeploymentMode::Db, &h.catalog, &h.data_dir, &AllCommitted);

    assert_eq!(report.recovered, 0, "a db node opens no lake log");
    assert_eq!(report.failed, 0);
    assert_eq!(
        report.skipped, 1,
        "the lake table it left closed must be reported, not hidden"
    );
    assert!(
        TransactionLog::lookup_shared(&paths).is_none(),
        "a db node must hold no lake state after startup"
    );
}

#[tokio::test]
async fn test_lake_and_unified_nodes_reconcile_every_lake_log_at_startup() {
    for mode in [DeploymentMode::Lake, DeploymentMode::Unified] {
        let (h, paths) = harness_with_one_lake_table().await;

        let report = recover_lake_logs(mode, &h.catalog, &h.data_dir, &AllCommitted);

        assert_eq!(report.recovered, 1, "{mode:?} must open the lake log");
        assert_eq!(report.failed, 0, "{mode:?}");
        assert_eq!(report.skipped, 0, "{mode:?} leaves no lake table closed");
        let shared = TransactionLog::lookup_shared(&paths)
            .unwrap_or_else(|| panic!("{mode:?} must register the shared head"));
        assert_eq!(shared.latest_version(), 1);
        // Heap tables are never routed through lake recovery
        TransactionLog::remove_shared(&paths);
    }
}

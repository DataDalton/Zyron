//! Recycle-bin DROP/UNDROP behavior at the catalog layer.
//!
//! Tables configured with a recycle window are soft-dropped: hidden from
//! lookups but retained for UNDROP. Tables without a window hard-drop. The
//! reaper finalizes a soft-dropped table once its window elapses.

use std::sync::Arc;
use tempfile::tempdir;

use zyron_buffer::{BufferPool, BufferPoolConfig};
use zyron_catalog::storage::{CatalogStorage, HeapCatalogStorage};
use zyron_catalog::*;
use zyron_parser::ast::{ColumnConstraint, ColumnDef, DataType};
use zyron_wal::{WalWriter, WalWriterConfig};

async fn setup(dir: &std::path::Path) -> (Catalog, SchemaId) {
    let data_dir = dir.join("data");
    let wal_dir = dir.join("wal");
    std::fs::create_dir_all(&data_dir).unwrap();
    std::fs::create_dir_all(&wal_dir).unwrap();

    let disk = Arc::new(
        zyron_storage::DiskManager::new(zyron_bench_harness::disk_config(data_dir))
            .await
            .unwrap(),
    );
    let pool = Arc::new(BufferPool::new(zyron_bench_harness::buffer_pool_config()));
    let wal = Arc::new(WalWriter::new(zyron_bench_harness::wal_config(wal_dir)).unwrap());
    let storage = HeapCatalogStorage::new(Arc::clone(&disk), Arc::clone(&pool)).unwrap();
    storage.init_cache().await.unwrap();
    let storage: Arc<dyn CatalogStorage> = Arc::new(storage);
    let cache = Arc::new(CatalogCache::new(1024, 256));
    let catalog = Catalog::new(storage, cache, Arc::clone(&wal))
        .await
        .unwrap();
    let schema_id = catalog
        .create_schema(SYSTEM_DATABASE_ID, "recycle_test", "system")
        .await
        .unwrap();
    (catalog, schema_id)
}

fn one_col() -> Vec<ColumnDef> {
    vec![ColumnDef {
        name: "id".to_string(),
        data_type: DataType::BigInt,
        nullable: Some(false),
        default: None,
        constraints: vec![ColumnConstraint::PrimaryKey],
    }]
}

/// Sets a recycle window on an existing table by editing its lifecycle config.
async fn set_recycle_window(catalog: &Catalog, schema_id: SchemaId, name: &str, secs: i64) {
    let table = catalog.get_table(schema_id, name).unwrap();
    let mut entry = (*table).clone();
    entry.lifecycle.recycle_window_seconds = secs;
    catalog.update_table(entry).await.unwrap();
}

#[tokio::test]
async fn soft_drop_hides_table_and_undrop_restores_it() {
    let dir = tempdir().unwrap();
    let (catalog, schema_id) = setup(dir.path()).await;

    catalog
        .create_table(schema_id, "orders", &one_col(), &[])
        .await
        .unwrap();
    set_recycle_window(&catalog, schema_id, "orders", 3600).await;

    let outcome = catalog.drop_table(schema_id, "orders").await.unwrap();
    assert!(outcome.soft_dropped, "table with a window soft-drops");

    // Hidden from every live lookup path.
    assert!(catalog.get_table(schema_id, "orders").is_err());
    assert!(
        catalog
            .list_tables(schema_id)
            .iter()
            .all(|t| t.name != "orders")
    );
    // Present in the recycle bin.
    assert_eq!(catalog.list_dropped_tables().len(), 1);

    // UNDROP brings it back, queryable again.
    catalog.undrop_table(schema_id, "orders").await.unwrap();
    assert!(catalog.get_table(schema_id, "orders").is_ok());
    assert_eq!(catalog.list_dropped_tables().len(), 0);
}

#[tokio::test]
async fn drop_without_window_removes_table_immediately() {
    let dir = tempdir().unwrap();
    let (catalog, schema_id) = setup(dir.path()).await;

    catalog
        .create_table(schema_id, "temp", &one_col(), &[])
        .await
        .unwrap();

    let outcome = catalog.drop_table(schema_id, "temp").await.unwrap();
    assert!(!outcome.soft_dropped, "no window means a hard drop");
    assert!(catalog.get_table(schema_id, "temp").is_err());
    assert_eq!(catalog.list_dropped_tables().len(), 0);
    // Not recoverable: there is nothing in the recycle bin to restore.
    assert!(catalog.undrop_table(schema_id, "temp").await.is_err());
}

#[tokio::test]
async fn finalize_purges_recycled_table() {
    let dir = tempdir().unwrap();
    let (catalog, schema_id) = setup(dir.path()).await;

    catalog
        .create_table(schema_id, "events", &one_col(), &[])
        .await
        .unwrap();
    set_recycle_window(&catalog, schema_id, "events", 3600).await;
    let id = catalog.get_table(schema_id, "events").unwrap().id;

    catalog.drop_table(schema_id, "events").await.unwrap();
    assert_eq!(catalog.list_dropped_tables().len(), 1);

    let purged = catalog.finalize_dropped_table(id).await.unwrap();
    assert!(purged.is_some(), "soft-dropped table is finalized");
    assert_eq!(catalog.list_dropped_tables().len(), 0);

    // A second finalize is a no-op: nothing left to purge.
    assert!(catalog.finalize_dropped_table(id).await.unwrap().is_none());
    // The name is free again for a fresh table.
    catalog
        .create_table(schema_id, "events", &one_col(), &[])
        .await
        .unwrap();
    assert!(catalog.get_table(schema_id, "events").is_ok());
}

#[tokio::test]
async fn undrop_blocked_when_live_table_reuses_name() {
    let dir = tempdir().unwrap();
    let (catalog, schema_id) = setup(dir.path()).await;

    catalog
        .create_table(schema_id, "ledger", &one_col(), &[])
        .await
        .unwrap();
    set_recycle_window(&catalog, schema_id, "ledger", 3600).await;
    catalog.drop_table(schema_id, "ledger").await.unwrap();

    // A new live table grabs the recycled name.
    catalog
        .create_table(schema_id, "ledger", &one_col(), &[])
        .await
        .unwrap();

    // UNDROP cannot shadow the live table.
    assert!(catalog.undrop_table(schema_id, "ledger").await.is_err());
    assert!(catalog.get_table(schema_id, "ledger").is_ok());
}

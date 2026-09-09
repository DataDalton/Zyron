//! An identifier this catalog has handed out is never handed out again.
//!
//! The id allocator recovers by scanning the rows the catalog holds and
//! taking the highest id it finds. That number is the highest id still in
//! use, which is not the highest ever allocated: dropping the newest object
//! removes the only evidence its id was taken, and a counter rebuilt from the
//! rows alone hands that id to the next object created.
//!
//! Reuse is not cosmetic. Grants, classifications, tags and masking rules are
//! all recorded against an object by kind and id, so a new object numbered
//! like a dropped one inherits what was written about the dropped one. On a
//! consensus group it is worse: every member allocates independently and they
//! agree because they apply the same log from the same position, so a member
//! that restarted after a drop numbers the next object differently from a
//! member that did not, and the two disagree about what the same object is
//! called.

use std::sync::Arc;
use tempfile::tempdir;

use zyron_buffer::BufferPool;
use zyron_catalog::storage::{CatalogStorage, HeapCatalogStorage};
use zyron_catalog::*;
use zyron_parser::ast::{ColumnConstraint, ColumnDef, DataType};
use zyron_wal::WalWriter;

/// Opens a catalog over a directory, the way a node does as it starts.
///
/// Called twice against the same directory to stand in for a restart: the
/// second catalog loads from what the first wrote and rebuilds its allocator
/// from that, which is the moment an id can be handed out twice
async fn open(dir: &std::path::Path) -> Catalog {
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
    Catalog::new(storage, cache, Arc::clone(&wal))
        .await
        .unwrap()
}

fn one_col() -> Vec<ColumnDef> {
    vec![ColumnDef {
        name: "id".to_string(),
        data_type: DataType::BigInt,
        nullable: Some(false),
        default: None,
        constraints: vec![ColumnConstraint::PrimaryKey],
        generated: None,
        encrypted: None,
        collation: None,
        media_format: None,
        media_storage: None,
        user_type_id: None,
    }]
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_dropped_table_does_not_lend_its_id_to_the_next_table() {
    let dir = tempdir().unwrap();

    let dropped_id;
    let schema_name = "reuse_test";
    {
        let catalog = open(dir.path()).await;
        let schema = catalog
            .create_schema(SYSTEM_DATABASE_ID, schema_name, "system")
            .await
            .unwrap();
        // The newest object, so its id is the highest the rows carry and the
        // one a scan would stop at once it is gone
        dropped_id = catalog
            .create_table(schema, "goes_away", &one_col(), &[])
            .await
            .unwrap();
        catalog.drop_table(schema, "goes_away").await.unwrap();
    }

    // A restart, which is when the allocator is rebuilt
    let catalog = open(dir.path()).await;
    let schema = catalog
        .get_schema(SYSTEM_DATABASE_ID, schema_name)
        .expect("the schema survived the restart")
        .id;
    let fresh_id = catalog
        .create_table(schema, "comes_after", &one_col(), &[])
        .await
        .unwrap();

    assert_ne!(
        fresh_id, dropped_id,
        "a table created after a restart was given the id a dropped table held, so \
         everything recorded against id {} now applies to a table nobody granted it on",
        dropped_id.0
    );
    assert!(
        fresh_id.0 > dropped_id.0,
        "ids go forwards: the dropped table held {} and the next table was given {}",
        dropped_id.0,
        fresh_id.0
    );
}

/// The same for a heap file id, which decides which files on disk a table
/// reads and writes. Two tables given the same pair would read each other's
/// pages
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_dropped_table_does_not_lend_its_heap_files_to_the_next_table() {
    let dir = tempdir().unwrap();

    let dropped_files;
    let schema_name = "reuse_files";
    {
        let catalog = open(dir.path()).await;
        let schema = catalog
            .create_schema(SYSTEM_DATABASE_ID, schema_name, "system")
            .await
            .unwrap();
        let id = catalog
            .create_table(schema, "goes_away", &one_col(), &[])
            .await
            .unwrap();
        let entry = catalog.get_table_by_id(id).expect("the table exists");
        dropped_files = (entry.heap_file_id, entry.fsm_file_id);
        catalog.drop_table(schema, "goes_away").await.unwrap();
    }

    let catalog = open(dir.path()).await;
    let schema = catalog
        .get_schema(SYSTEM_DATABASE_ID, schema_name)
        .expect("the schema survived the restart")
        .id;
    let fresh = catalog
        .create_table(schema, "comes_after", &one_col(), &[])
        .await
        .unwrap();
    let entry = catalog.get_table_by_id(fresh).expect("the table exists");

    assert_ne!(
        (entry.heap_file_id, entry.fsm_file_id),
        dropped_files,
        "a table created after a restart was given the heap and free-space files a \
         dropped table held"
    );
}

/// Restarting without dropping anything leaves the counter where it was, so
/// the record is a floor rather than something that pushes ids upward on its
/// own
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_restart_on_its_own_does_not_move_the_counter() {
    let dir = tempdir().unwrap();
    let schema_name = "reuse_stable";

    let first_id;
    {
        let catalog = open(dir.path()).await;
        let schema = catalog
            .create_schema(SYSTEM_DATABASE_ID, schema_name, "system")
            .await
            .unwrap();
        first_id = catalog
            .create_table(schema, "kept", &one_col(), &[])
            .await
            .unwrap();
    }

    let catalog = open(dir.path()).await;
    let schema = catalog
        .get_schema(SYSTEM_DATABASE_ID, schema_name)
        .expect("the schema survived the restart")
        .id;
    let second_id = catalog
        .create_table(schema, "next", &one_col(), &[])
        .await
        .unwrap();

    assert!(
        second_id.0 > first_id.0,
        "the second table takes an id above the first"
    );
    // Nothing was dropped, so the scan already knew the highest id and the
    // record adds nothing. A gap here would mean the counter is being pushed
    // by the record rather than held up by it
    assert_eq!(
        second_id.0,
        first_id.0 + 1,
        "an unbroken run of creates numbers them consecutively, so two members \
         allocating from the same log reach the same ids"
    );
}

/// The counters row survives on its own, without a catalog around it.
///
/// Kept separate from the cases above so a failure says whether the record is
/// not being written or the allocator is not reading it
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_counters_row_is_read_back_after_a_reopen() {
    let dir = tempdir().unwrap();
    let data_dir = dir.path().join("data");
    std::fs::create_dir_all(&data_dir).unwrap();

    let written = CatalogCounters {
        next_oid: 12_345,
        next_heap_file: 678,
        next_index_file: 91_011,
    };

    {
        let disk = Arc::new(
            zyron_storage::DiskManager::new(zyron_bench_harness::disk_config(data_dir.clone()))
                .await
                .unwrap(),
        );
        let pool = Arc::new(BufferPool::new(zyron_bench_harness::buffer_pool_config()));
        let storage = HeapCatalogStorage::new(Arc::clone(&disk), Arc::clone(&pool)).unwrap();
        storage.init_cache().await.unwrap();
        storage.store_counters(written).await.unwrap();
        assert_eq!(
            storage.load_counters().await.unwrap(),
            Some(written),
            "the row does not read back through the handle that wrote it"
        );
        storage.flush_all_dirty().await.unwrap();
    }

    let disk = Arc::new(
        zyron_storage::DiskManager::new(zyron_bench_harness::disk_config(data_dir))
            .await
            .unwrap(),
    );
    let pool = Arc::new(BufferPool::new(zyron_bench_harness::buffer_pool_config()));
    let storage = HeapCatalogStorage::new(disk, pool).unwrap();
    storage.init_cache().await.unwrap();
    assert_eq!(
        storage.load_counters().await.unwrap(),
        Some(written),
        "the row did not survive a reopen"
    );
}

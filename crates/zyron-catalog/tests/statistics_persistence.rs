//! Statistics survive a restart.
//!
//! ANALYZE writes one file per table beside the heap files. A restart reads
//! them back before connections are accepted, so the first plan after a
//! reboot costs against real statistics instead of the no-statistics
//! default. A file whose table is gone is deleted rather than loaded

use std::sync::Arc;
use tempfile::tempdir;

use zyron_buffer::BufferPool;
use zyron_catalog::statistics::{STATISTICS_DIR, read_statistics_file, statistics_path};
use zyron_catalog::storage::{CatalogStorage, HeapCatalogStorage};
use zyron_catalog::*;
use zyron_parser::ast::{ColumnDef, DataType};
use zyron_wal::WalWriter;

/// Opens a catalog over an existing data directory, so a test can close one
/// and reopen another the way a restart does
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
    Catalog::new(storage, cache, wal).await.unwrap()
}

fn one_col() -> Vec<ColumnDef> {
    vec![ColumnDef {
        name: "value".to_string(),
        data_type: DataType::BigInt,
        nullable: Some(true),
        default: None,
        constraints: vec![],
        generated: None,
        encrypted: None,
        collation: None,
        media_format: None,
        media_storage: None,
        user_type_id: None,
    }]
}

fn sample_stats(table_id: TableId, row_count: u64) -> (TableStats, Vec<ColumnStats>) {
    (
        TableStats {
            table_id,
            row_count,
            page_count: 9,
            avg_row_size: 24,
            last_analyzed: 1_700_000_000,
        },
        vec![ColumnStats {
            table_id,
            column_id: zyron_catalog::ids::ColumnId(0),
            null_fraction: 0.25,
            distinct_count: row_count / 2,
            avg_width: 8,
            histogram: None,
            most_common_values: vec![b"7".to_vec()],
            most_common_freqs: vec![0.5],
        }],
    )
}

/// The stats directory the catalog derives from its data directory
fn stats_dir(dir: &std::path::Path) -> std::path::PathBuf {
    dir.join("data").join(STATISTICS_DIR)
}

#[tokio::test]
async fn test_persist_writes_a_file_and_updates_memory() {
    let dir = tempdir().unwrap();
    let catalog = open(dir.path()).await;
    let schema_id = catalog
        .create_schema(SYSTEM_DATABASE_ID, "zyron_test", "system")
        .await
        .unwrap();
    let table_id = catalog
        .create_table(schema_id, "t", &one_col(), &[])
        .await
        .unwrap();

    let (table_stats, column_stats) = sample_stats(table_id, 4_000);
    catalog
        .persist_stats(table_id, table_stats, column_stats)
        .await
        .unwrap();

    // Visible in memory straight away, without a read of the file
    let held = catalog.get_stats(table_id).expect("stats are in memory");
    assert_eq!(held.0.row_count, 4_000);

    // And on disk, parsed back through the file format
    let (from_disk, cols) = read_statistics_file(&stats_dir(dir.path()), table_id)
        .unwrap()
        .expect("file was written");
    assert_eq!(from_disk.row_count, 4_000);
    assert_eq!(cols.len(), 1);
    assert_eq!(cols[0].most_common_values, vec![b"7".to_vec()]);
}

#[tokio::test]
async fn test_statistics_survive_a_restart() {
    let dir = tempdir().unwrap();
    let table_id;
    {
        let catalog = open(dir.path()).await;
        let schema_id = catalog
            .create_schema(SYSTEM_DATABASE_ID, "zyron_test", "system")
            .await
            .unwrap();
        table_id = catalog
            .create_table(schema_id, "t", &one_col(), &[])
            .await
            .unwrap();
        let (table_stats, column_stats) = sample_stats(table_id, 12_345);
        catalog
            .persist_stats(table_id, table_stats, column_stats)
            .await
            .unwrap();
    }

    let reopened = open(dir.path()).await;
    reopened.load().await.unwrap();

    // Nothing is in memory until the load runs
    assert!(reopened.get_stats(table_id).is_none());
    let restored = reopened.load_persisted_stats().await.unwrap();
    assert_eq!(restored, 1);

    let held = reopened.get_stats(table_id).expect("restored");
    assert_eq!(held.0.row_count, 12_345);
    assert_eq!(held.0.page_count, 9);
    assert_eq!(held.1[0].distinct_count, 12_345 / 2);
    assert!((held.1[0].null_fraction - 0.25).abs() < f64::EPSILON);
}

#[tokio::test]
async fn test_dropping_a_table_removes_its_statistics_file() {
    let dir = tempdir().unwrap();
    let catalog = open(dir.path()).await;
    let schema_id = catalog
        .create_schema(SYSTEM_DATABASE_ID, "zyron_test", "system")
        .await
        .unwrap();
    let table_id = catalog
        .create_table(schema_id, "t", &one_col(), &[])
        .await
        .unwrap();
    let (table_stats, column_stats) = sample_stats(table_id, 100);
    catalog
        .persist_stats(table_id, table_stats, column_stats)
        .await
        .unwrap();

    let path = statistics_path(&stats_dir(dir.path()), table_id);
    assert!(path.exists(), "the file should exist before the drop");

    catalog.drop_table(schema_id, "t").await.unwrap();

    assert!(!path.exists(), "the drop should have removed the file");
    assert!(catalog.get_stats(table_id).is_none());
}

/// A file left behind by a drop that crashed before its unlink is deleted at
/// the next load rather than restored, so a table id reused later never
/// inherits the old table's numbers
#[tokio::test]
async fn test_a_file_for_a_table_that_is_gone_is_swept() {
    let dir = tempdir().unwrap();
    let catalog = open(dir.path()).await;
    let schema_id = catalog
        .create_schema(SYSTEM_DATABASE_ID, "zyron_test", "system")
        .await
        .unwrap();
    let live_id = catalog
        .create_table(schema_id, "live", &one_col(), &[])
        .await
        .unwrap();
    let (table_stats, column_stats) = sample_stats(live_id, 500);
    catalog
        .persist_stats(live_id, table_stats, column_stats)
        .await
        .unwrap();

    // A file for a table id that was never created
    let ghost_id = TableId(live_id.0 + 9_999);
    let (ghost_stats, ghost_cols) = sample_stats(ghost_id, 77);
    zyron_catalog::statistics::write_statistics_file(
        &stats_dir(dir.path()),
        ghost_id,
        &ghost_stats,
        &ghost_cols,
    )
    .unwrap();
    let ghost_path = statistics_path(&stats_dir(dir.path()), ghost_id);
    assert!(ghost_path.exists());

    let restored = catalog.load_persisted_stats().await.unwrap();
    assert_eq!(restored, 1, "only the live table should be restored");
    assert!(catalog.get_stats(live_id).is_some());
    assert!(catalog.get_stats(ghost_id).is_none());
    assert!(!ghost_path.exists(), "the stale file should be deleted");
}

/// Statistics are an optimization. A damaged file costs the plans for its
/// own table and must not stop the others being restored
#[tokio::test]
async fn test_a_corrupt_file_does_not_block_the_others() {
    let dir = tempdir().unwrap();
    let catalog = open(dir.path()).await;
    let schema_id = catalog
        .create_schema(SYSTEM_DATABASE_ID, "zyron_test", "system")
        .await
        .unwrap();
    let good_id = catalog
        .create_table(schema_id, "good", &one_col(), &[])
        .await
        .unwrap();
    let bad_id = catalog
        .create_table(schema_id, "bad", &one_col(), &[])
        .await
        .unwrap();

    for (id, rows) in [(good_id, 10u64), (bad_id, 20u64)] {
        let (table_stats, column_stats) = sample_stats(id, rows);
        catalog
            .persist_stats(id, table_stats, column_stats)
            .await
            .unwrap();
    }

    // Flip a body byte so the footer checksum no longer matches
    let bad_path = statistics_path(&stats_dir(dir.path()), bad_id);
    let mut bytes = std::fs::read(&bad_path).unwrap();
    let last = bytes.len() - 8;
    bytes[last] ^= 0xFF;
    std::fs::write(&bad_path, &bytes).unwrap();

    let fresh = open(dir.path()).await;
    fresh.load().await.unwrap();
    let restored = fresh.load_persisted_stats().await.unwrap();

    assert_eq!(restored, 1, "the good table is still restored");
    assert_eq!(fresh.get_stats(good_id).expect("good").0.row_count, 10);
    assert!(
        fresh.get_stats(bad_id).is_none(),
        "the damaged file is skipped, not trusted"
    );
}

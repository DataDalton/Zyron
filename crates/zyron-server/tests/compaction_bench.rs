//! Fold-path throughput benchmark.
//!
//! `columnar_bench`'s "Compaction throughput" metric calls
//! `run_compaction_cycle` with a pre-built input and never executes the
//! server fold path (`CompactionWorker::run_cycle` -> `compact_table`:
//! heap scan, arena materialization, .zyr encode/write, sidecar, WAL
//! commit, registry, heap-slot zeroing). This benchmark drives that real
//! path against a real heap so fold-path changes have a trustworthy
//! before/after number.

use std::sync::Arc;
use std::time::Instant;

use zyron_bench_harness::*;
use zyron_buffer::{BufferPool, BufferPoolConfig};
use zyron_catalog::storage::{CatalogStorage, HeapCatalogStorage};
use zyron_catalog::{Catalog, CatalogCache};
use zyron_parser::ast::{ColumnDef, DataType};
use zyron_server::background::compaction::{CompactionWorker, CompactionWorkerConfig};
use zyron_storage::txn::TransactionManager;
use zyron_storage::{DiskManager, DiskManagerConfig, HeapFile, HeapFileConfig, Tuple};
use zyron_wal::{WalWriter, WalWriterConfig};

static BENCHMARK_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn col(name: &str, ty: DataType) -> ColumnDef {
    ColumnDef {
        name: name.to_string(),
        data_type: ty,
        nullable: Some(true),
        default: None,
        constraints: vec![],
    }
}

/// NSM row of (k:i64, name:text, v:i64): null bitmap then fixed/varlen.
fn encode_row(k: i64, name: &str, v: i64) -> Vec<u8> {
    let mut d = Vec::new();
    d.push(0u8);
    d.extend_from_slice(&k.to_le_bytes());
    d.extend_from_slice(&(name.len() as u32).to_le_bytes());
    d.extend_from_slice(name.as_bytes());
    d.extend_from_slice(&v.to_le_bytes());
    d
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_fold_path_throughput() {
    zyron_bench_harness::init("columnar");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    const ROW_COUNT: usize = 100_000;
    const RUNS: usize = 5;
    const TARGET_ROWS_SEC: f64 = 1_000_000.0;

    tprintln!("\n=== Fold-path throughput (CompactionWorker::run_cycle) ===");
    tprintln!("Rows per run: {}, runs: {}", ROW_COUNT, RUNS);

    let util_before = take_util_snapshot();
    let mut rows_per_sec: Vec<f64> = Vec::with_capacity(RUNS);

    for run in 0..RUNS {
        let tmp = tempfile::tempdir().expect("tmp");
        let data_dir = tmp.path().join("data");
        let wal_dir = tmp.path().join("wal");
        std::fs::create_dir_all(&data_dir).unwrap();
        std::fs::create_dir_all(&wal_dir).unwrap();
        let columnar_dir = data_dir.join("columnar");

        let disk = Arc::new(
            DiskManager::new(zyron_bench_harness::disk_config(data_dir.clone()))
                .await
                .unwrap(),
        );
        let pool = Arc::new(BufferPool::new(zyron_bench_harness::buffer_pool_config()));
        let wal =
            Arc::new(WalWriter::new(zyron_bench_harness::wal_config(wal_dir.clone())).unwrap());

        let storage = HeapCatalogStorage::new(Arc::clone(&disk), Arc::clone(&pool)).unwrap();
        storage.init_cache().await.unwrap();
        let storage: Arc<dyn CatalogStorage> = Arc::new(storage);
        let cache = Arc::new(CatalogCache::new(1024, 256));
        let catalog = Catalog::new(Arc::clone(&storage), cache, Arc::clone(&wal))
            .await
            .unwrap();
        let db = catalog.create_database("db", "admin").await.unwrap();
        let schema = catalog.create_schema(db, "app", "admin").await.unwrap();
        let cols = vec![
            col("k", DataType::BigInt),
            col("name", DataType::Text),
            col("v", DataType::BigInt),
        ];
        let table_id = catalog
            .create_table(schema, "metrics", &cols, &[])
            .await
            .unwrap();
        let txn = Arc::new(TransactionManager::with_start_txn_id(Arc::clone(&wal), 100));

        let te = catalog.get_table_by_id(table_id).unwrap();
        let heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: te.heap_file_id,
                fsm_file_id: te.fsm_file_id,
            },
        )
        .unwrap();
        let tuples: Vec<Tuple> = (0..ROW_COUNT as i64)
            .map(|i| Tuple::new(encode_row(i, "row", i * 100), 1))
            .collect();
        heap.insert_batch(&tuples).await.unwrap();
        heap.flush().await.unwrap();

        // The worker the server runs. Only the trigger moves, because a
        // fold has to happen at all for there to be anything to measure
        let cfg = CompactionWorkerConfig {
            min_rows: 0,
            columnar_dir: columnar_dir.clone(),
            ..CompactionWorkerConfig::default()
        };

        let (rows, segs) = {
            let c = &catalog;
            let t = &txn;
            let d = &disk;
            let p = &pool;
            let w = &wal;
            let cf = &cfg;
            let start = Instant::now();
            let res = tokio::task::block_in_place(|| {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap();
                CompactionWorker::run_cycle(&rt, c, t, d, p, w, cf, None, None, None)
            });
            let elapsed = start.elapsed();
            let rps = res.0 as f64 / elapsed.as_secs_f64();
            rows_per_sec.push(rps);
            res
        };
        assert_eq!(rows, ROW_COUNT as u64, "all rows folded");
        assert_eq!(segs, 1, "one segment written");
        tprintln!(
            "  run {}: {} rows/sec",
            run + 1,
            format_with_commas(rows_per_sec[run])
        );
    }

    validate_metric(
        "Fold Path",
        "Fold-path throughput (rows/sec)",
        rows_per_sec,
        TARGET_ROWS_SEC,
        true,
    );
    let util_after = take_util_snapshot();
    record_test_util("Fold Path", util_before, util_after);
}

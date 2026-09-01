//! Fold-path throughput benchmark.
//!
//! `columnar_bench`'s "Compaction throughput" metric calls
//! `run_compaction_cycle` with a pre-built input and never executes the
//! server fold path (`CompactionWorker::run_cycle` -> `compact_table`:
//! heap scan, arena materialization, .zyr encode/write, sidecar, WAL
//! commit, registry, heap-slot zeroing). This benchmark drives that real
//! path against a real heap so fold-path changes have a trustworthy
//! before/after number.
//!
//! Writes the `fold_path` suite, not `columnar`. A suite name is also the
//! output file name, and a run id is per process, so two binaries claiming
//! one name write rival files instead of a combined one and whichever ran
//! last is the only one a reader finds. `columnar_bench` in zyron-storage
//! owns `columnar`. This cannot join it, because `CompactionWorker` lives in
//! zyron-server and storage cannot depend on the server.

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
        generated: None,
        encrypted: None,
        collation: None,
        media_format: None,
        media_storage: None,
        user_type_id: None,
    }
}

/// NSM row of (k:i64, name:text, v:i64): null bitmap then fixed/varlen.
/// NSM-encodes one row of (k:i64, v:text) the way the heap and the fold read
/// it: null bitmap, the fixed column, then the variable length one
fn encode_variant_row(k: i64, doc: &str) -> Vec<u8> {
    let mut d = Vec::with_capacity(13 + doc.len());
    d.push(0u8);
    d.extend_from_slice(&k.to_le_bytes());
    d.extend_from_slice(&(doc.len() as u32).to_le_bytes());
    d.extend_from_slice(doc.as_bytes());
    d
}

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
    zyron_bench_harness::init("fold_path");
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
                CompactionWorker::run_cycle(&rt, c, t, d, p, w, cf, None, None, None, None)
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

/// A promoted VARIANT path, read out of the column the fold materialized it
/// into rather than by walking every document for it.
///
/// This is the measurement the search suite states a 100 ns target for and
/// cannot take: shredding needs a fold, and the fold lives here. The same
/// read is timed twice over the same folded rows, once with the path promoted
/// and once without, so the number is the distance between the two paths on
/// one machine rather than two runs compared across time.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_shredded_variant_read() {
    zyron_bench_harness::init("fold_path");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let row_count: usize = if measuring() { 200_000 } else { 20_000 };
    const RUNS: usize = 5;
    /// The shredded read reaches a decoded column, so it is bounded by the
    /// decode rather than by a parse
    const TARGET_SHREDDED_NS: f64 = 100.0;
    /// Walking the document for the path is the unshredded cost
    const TARGET_UNSHREDDED_NS: f64 = 5_000.0;

    tprintln!("\n=== Shredded VARIANT read (folded segment) ===");
    tprintln!("Rows: {row_count}, runs: {RUNS}");

    let mut shredded_ns: Vec<f64> = Vec::with_capacity(RUNS);
    let mut walked_ns: Vec<f64> = Vec::with_capacity(RUNS);
    let util_before = take_util_snapshot();

    for run in 0..RUNS {
        // `promoted` decides whether the fold materializes the path, so the
        // two arms differ in one thing only
        for promoted in [false, true] {
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
            zyron_bench_harness::install_evict_writer(&pool, &disk, None);
            let wal =
                Arc::new(WalWriter::new(zyron_bench_harness::wal_config(wal_dir.clone())).unwrap());

            let storage = HeapCatalogStorage::new(Arc::clone(&disk), Arc::clone(&pool)).unwrap();
            storage.init_cache().await.unwrap();
            let storage: Arc<dyn CatalogStorage> = Arc::new(storage);
            let cache = Arc::new(CatalogCache::new(1024, 256));
            let catalog = Arc::new(
                Catalog::new(Arc::clone(&storage), cache, Arc::clone(&wal))
                    .await
                    .unwrap(),
            );
            let db = catalog.create_database("db", "admin").await.unwrap();
            let schema = catalog.create_schema(db, "app", "admin").await.unwrap();
            let cols = vec![col("k", DataType::BigInt), col("v", DataType::Variant)];
            let table_id = catalog
                .create_table(schema, "events", &cols, &[])
                .await
                .unwrap();
            let txn = Arc::new(TransactionManager::with_start_txn_id(Arc::clone(&wal), 100));

            let te = catalog.get_table_by_id(table_id).unwrap();
            let v_col = te.columns.iter().find(|c| c.name == "v").unwrap().id.0;
            let heap = HeapFile::new(
                Arc::clone(&disk),
                Arc::clone(&pool),
                HeapFileConfig {
                    heap_file_id: te.heap_file_id,
                    fsm_file_id: te.fsm_file_id,
                },
            )
            .unwrap();

            // Documents wide enough that walking them costs what a real one
            // costs, with the read path buried past several other fields
            let tuples: Vec<Tuple> = (0..row_count as i64)
                .map(|i| {
                    let doc = format!(
                        "{{\"kind\":\"k_{}\",\"src\":\"ingest\",\"seq\":{i},\
                         \"tags\":\"a,b,c\",\"meta\":{{\"depth\":{},\"name\":\"n_{i}\"}},\
                         \"user\":{{\"id\":{}}}}}",
                        i % 40,
                        i % 97,
                        i % 1000
                    );
                    Tuple::new(encode_variant_row(i, &doc), 1)
                })
                .collect();
            heap.insert_batch(&tuples).await.unwrap();
            heap.flush().await.unwrap();

            if promoted {
                zyron_executor::variant_shred::mark_shredded(table_id.0, v_col, "user.id");
            } else {
                zyron_executor::variant_shred::clear_column(table_id.0, v_col);
            }

            let cfg = CompactionWorkerConfig {
                min_rows: 0,
                columnar_dir: columnar_dir.clone(),
                ..CompactionWorkerConfig::default()
            };
            let (folded, segs) = {
                let c = &catalog;
                let t = &txn;
                let d = &disk;
                let p = &pool;
                let w = &wal;
                let cf = &cfg;
                tokio::task::block_in_place(|| {
                    let rt = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .unwrap();
                    CompactionWorker::run_cycle(&rt, c, t, d, p, w, cf, None, None, None, None)
                })
            };
            assert_eq!(folded, row_count as u64, "every row folded");
            assert!(segs >= 1, "a segment was written");

            let te = catalog.get_table_by_id(table_id).unwrap();
            let stored: usize = te.columnar.segments.iter().map(|s| s.shredded.len()).sum();
            assert_eq!(
                stored > 0,
                promoted,
                "the fold materialized the path when and only when it was promoted"
            );

            let per_row = read_path_ns(&catalog, &wal, &pool, &disk, table_id, v_col, row_count);
            if promoted {
                shredded_ns.push(per_row);
            } else {
                walked_ns.push(per_row);
            }
            zyron_executor::variant_shred::clear_column(table_id.0, v_col);
        }
        tprintln!(
            "  run {}: shredded {:.0} ns/row, document walk {:.0} ns/row",
            run + 1,
            shredded_ns[run],
            walked_ns[run]
        );
    }

    validate_metric_with_unit(
        "VARIANT Shredded Read",
        "Promoted path read from its own column, per row",
        "ns",
        shredded_ns.clone(),
        TARGET_SHREDDED_NS,
        false,
    );
    validate_metric_with_unit(
        "VARIANT Walked Read",
        "Same path read by walking the document, per row",
        "ns",
        walked_ns.clone(),
        TARGET_UNSHREDDED_NS,
        false,
    );
    let shredded_avg = shredded_ns.iter().sum::<f64>() / shredded_ns.len() as f64;
    let walked_avg = walked_ns.iter().sum::<f64>() / walked_ns.len() as f64;
    tprintln!(
        "  shredding is {:.1}x the document walk on this machine",
        walked_avg / shredded_avg.max(f64::MIN_POSITIVE)
    );

    let util_after = take_util_snapshot();
    record_test_util("VARIANT Shredded Read", util_before, util_after);
}

/// Times one full pass of `variant_extract` over a folded table, in
/// nanoseconds per row. The scan is the same one a query runs, so the number
/// carries the decode the read actually pays.
fn read_path_ns(
    catalog: &Arc<Catalog>,
    wal: &Arc<WalWriter>,
    pool: &Arc<BufferPool>,
    disk: &Arc<DiskManager>,
    table_id: zyron_catalog::TableId,
    v_col: u16,
    row_count: usize,
) -> f64 {
    use zyron_executor::operator::Operator;
    use zyron_planner::binder::{BoundExpr, ColumnRef};
    use zyron_planner::logical::LogicalColumn;

    let te = catalog.get_table_by_id(table_id).unwrap();
    let columns: Vec<LogicalColumn> = te
        .columns
        .iter()
        .filter(|c| c.name == "v")
        .map(|c| LogicalColumn {
            table_idx: Some(0),
            column_id: c.id,
            name: c.name.clone(),
            type_id: c.type_id,
            nullable: c.nullable,
            fractional_digits: c.fractional_digits,
        })
        .collect();
    let extract = BoundExpr::Function {
        name: "variant_extract".to_string(),
        args: vec![
            BoundExpr::ColumnRef(ColumnRef {
                table_idx: 0,
                column_id: zyron_catalog::ColumnId(v_col),
                type_id: zyron_common::TypeId::Variant,
                nullable: true,
                fractional_digits: None,
            }),
            BoundExpr::Literal {
                value: zyron_parser::ast::LiteralValue::String("user.id".to_string()),
                type_id: zyron_common::TypeId::Text,
            },
        ],
        return_type: zyron_common::TypeId::Text,
        distinct: false,
    };

    let ctx = Arc::new(zyron_executor::ExecutionContext::new(
        Arc::clone(catalog),
        Arc::clone(wal),
        Arc::clone(pool),
        Arc::clone(disk),
        200,
        zyron_storage::txn::Snapshot::new(
            200,
            vec![],
            Arc::new(zyron_storage::TxnStatusMap::all_committed()),
        ),
    ));
    ctx.set_variant_paths(vec![zyron_planner::physical::variant_paths::VariantPath {
        table_idx: 0,
        column_id: v_col,
        path: "user.id".to_string(),
    }]);

    let mut op = zyron_executor::operator::column_scan::ColumnScanOperator::new(
        ctx.clone(),
        table_id,
        columns.clone(),
        None,
    )
    .unwrap();

    let start = Instant::now();
    let mut seen = 0usize;
    // The scan is driven on this thread rather than on a nested runtime,
    // which the caller is already inside
    tokio::task::block_in_place(|| {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            while let Some(eb) = op.next().await.unwrap() {
                let batch = eb.batch;
                let values =
                    zyron_executor::expr::evaluate(&extract, &batch, &columns, &[]).unwrap();
                // Touched so the evaluation cannot be optimized away
                seen += values.len();
            }
        });
    });
    let elapsed = start.elapsed();
    assert_eq!(seen, row_count, "the read pass lost rows");
    elapsed.as_secs_f64() * 1e9 / row_count as f64
}

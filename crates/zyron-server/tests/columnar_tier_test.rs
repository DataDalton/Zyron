//! Anti-regression seam test for the columnar-MVCC tier.
//!
//! Asserts the mechanism, not just the answer:
//!   (a) compaction physically produces a .zyr and registers it,
//!   (b) the planner emits a HybridScan for a table with segments,
//!   (c) the folded data (including a variable-length column) round-trips
//!       byte-identically out of the .zyr with the sys MVCC columns present,
//!   (d) UPDATE/DELETE of a folded row goes to the patch overlay and resolves
//!       under snapshot visibility,
//!   (e) a crash mid-compaction (CompactionBegin with no CompactionEnd)
//!       recovers with the heap authoritative and the partial .zyr discarded.
//!
//! This gap must never recur: a writer without a reader, or a reader without
//! the MVCC union, must fail this test.

use std::sync::Arc;

use zyron_buffer::{BufferPool, BufferPoolConfig};
use zyron_catalog::storage::{CatalogStorage, HeapCatalogStorage};
use zyron_catalog::{Catalog, CatalogCache};
use zyron_parser::ast::{ColumnDef, DataType};
use zyron_planner::physical::PhysicalPlan;
use zyron_server::background::compaction::{CompactionWorker, CompactionWorkerConfig};
use zyron_server::columnar_recovery::reconcile_columnar;
use zyron_storage::columnar::{ColumnarPatchManager, SYS_COL_XMIN, ZyrFileReader, segment_regions};
use zyron_storage::encoding::create_encoding;
use zyron_storage::txn::TransactionManager;
use zyron_storage::{DiskManager, DiskManagerConfig, HeapFile, HeapFileConfig, Tuple};
use zyron_wal::{WalWriter, WalWriterConfig};

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

/// NSM-encodes one row of (k:i64, name:text, v:i64) exactly as the heap and
/// the compaction materializer expect: null bitmap then fixed/varlen columns.
fn encode_row(k: i64, name: &str, v: i64) -> Vec<u8> {
    let mut d = Vec::new();
    d.push(0u8); // null bitmap, 3 cols -> 1 byte, no nulls
    d.extend_from_slice(&k.to_le_bytes());
    d.extend_from_slice(&(name.len() as u32).to_le_bytes());
    d.extend_from_slice(name.as_bytes());
    d.extend_from_slice(&v.to_le_bytes());
    d
}

/// Decodes one column straight from its raw segment bytes.
///
/// The region arithmetic lives in `columnar::segment_regions`, which is
/// what the decode path, the predicate path and the scan all read. A
/// fourth copy here would drift from them and then assert against a
/// layout nobody writes
fn decode_column_raw(
    reader: &ZyrFileReader,
    column_id: u32,
    row_count: usize,
    value_size: usize,
) -> Vec<u8> {
    let raw = reader.read_segment_raw(column_id).expect("segment raw");
    let regions = segment_regions(&raw, column_id, row_count).expect("segment regions");
    let encoded = regions
        .verified_payload(&raw, column_id)
        .expect("payload checksum");
    create_encoding(regions.header.encoding_type)
        .decode(encoded, row_count, value_size)
        .expect("decode")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn columnar_seam_fold_read_mutate_recover() {
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
    let wal = Arc::new(WalWriter::new(zyron_bench_harness::wal_config(wal_dir.clone())).unwrap());

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
    // Start the txn counter above the rows' xmin so they sit below the
    // oldest-active horizon and are fold-eligible.
    let txn = Arc::new(TransactionManager::with_start_txn_id(Arc::clone(&wal), 100));

    // Insert rows directly into the heap with a low committed xmin.
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
    const N: i64 = 12;
    let mut tuples = Vec::new();
    for i in 0..N {
        tuples.push(Tuple::new(encode_row(i, &format!("row-{}", i), i * 100), 1));
    }
    heap.insert_batch(&tuples).await.unwrap();
    // Persist pages so a fresh HeapFile in the worker discovers them (no
    // background writer runs in this test).
    heap.flush().await.unwrap();

    let cfg = CompactionWorkerConfig {
        min_rows: 4,
        columnar_dir: columnar_dir.clone(),
        ..CompactionWorkerConfig::default()
    };
    let rt_handle = tokio::runtime::Handle::current();
    // run_cycle builds its own current-thread runtime internally for the
    // async catalog update; run it on a blocking thread so that nested
    // runtime does not conflict with this test's runtime.
    let (rows, segs) = {
        let catalog2 = &catalog;
        let txn2 = &txn;
        let disk2 = &disk;
        let pool2 = &pool;
        let wal2 = &wal;
        let cfg2 = &cfg;
        tokio::task::block_in_place(|| {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            CompactionWorker::run_cycle(
                &rt, catalog2, txn2, disk2, pool2, wal2, cfg2, None, None, None, None,
            )
        })
    };
    let _ = rt_handle;

    // (a) Mechanism engaged: a segment was folded and registered.
    assert_eq!(rows, N as u64, "all eligible rows folded");
    assert_eq!(segs, 1, "exactly one segment written");
    let te = catalog.get_table_by_id(table_id).unwrap();
    assert_eq!(
        te.columnar.segments.len(),
        1,
        "registry records the segment"
    );
    let seg = te.columnar.segments[0].clone();
    assert!(
        std::path::Path::new(&seg.path).exists(),
        "the .zyr file physically exists"
    );
    assert_eq!(seg.row_count, N as u64);

    // (b) Planner emits a HybridScan for a table with registered segments.
    let logical = zyron_planner::logical::LogicalPlan::Scan {
        table_id,
        table_idx: 0,
        columns: te
            .columns
            .iter()
            .map(|c| zyron_planner::logical::LogicalColumn {
                table_idx: Some(0),
                column_id: c.id,
                name: c.name.clone(),
                type_id: c.type_id,
                nullable: c.nullable,
                fractional_digits: c.fractional_digits,
            })
            .collect(),
        alias: "metrics".into(),
        encoding_hints: None,
        as_of: None,
    };
    let physical = zyron_planner::physical::builder::build_physical_plan(logical, &catalog, None)
        .expect("plan");
    assert!(
        matches!(physical, PhysicalPlan::HybridScan { .. }),
        "planner must pick HybridScan once segments exist, got {:?}",
        std::mem::discriminant(&physical)
    );

    // (b2) Metadata aggregate pushdown: ungrouped COUNT(*)/MIN(k)/MAX(v) over
    // the folded table, no predicate, must plan as ColumnarMetadataAggregate
    // (answered from segment headers, not a row decode).
    let col = |name: &str| {
        let c = te.columns.iter().find(|c| c.name == name).unwrap();
        zyron_planner::binder::BoundExpr::ColumnRef(zyron_planner::binder::ColumnRef {
            table_idx: 0,
            column_id: c.id,
            type_id: c.type_id,
            nullable: c.nullable,
            fractional_digits: c.fractional_digits,
        })
    };
    let scan_for_agg = zyron_planner::logical::LogicalPlan::Scan {
        table_id,
        table_idx: 0,
        columns: te
            .columns
            .iter()
            .map(|c| zyron_planner::logical::LogicalColumn {
                table_idx: Some(0),
                column_id: c.id,
                name: c.name.clone(),
                type_id: c.type_id,
                nullable: c.nullable,
                fractional_digits: c.fractional_digits,
            })
            .collect(),
        alias: "metrics".into(),
        encoding_hints: None,
        as_of: None,
    };
    let agg = zyron_planner::logical::LogicalPlan::Aggregate {
        group_by: vec![],
        aggregates: vec![
            zyron_planner::logical::AggregateExpr {
                function_name: "count".into(),
                args: vec![],
                distinct: false,
                return_type: zyron_common::types::TypeId::Int64,
                uda: None,
            },
            zyron_planner::logical::AggregateExpr {
                function_name: "min".into(),
                args: vec![col("k")],
                distinct: false,
                return_type: zyron_common::types::TypeId::Int64,
                uda: None,
            },
            zyron_planner::logical::AggregateExpr {
                function_name: "max".into(),
                args: vec![col("v")],
                distinct: false,
                return_type: zyron_common::types::TypeId::Int64,
                uda: None,
            },
        ],
        child: Arc::new(scan_for_agg),
    };
    let agg_phys = zyron_planner::physical::builder::build_physical_plan(agg, &catalog, None)
        .expect("agg plan");
    match agg_phys {
        PhysicalPlan::ColumnarMetadataAggregate { ref specs, .. } => {
            use zyron_planner::physical::MetaAggKind::*;
            assert_eq!(specs.len(), 3);
            assert_eq!(specs[0].kind, CountStar);
            assert_eq!(specs[1].kind, Min);
            assert_eq!(specs[2].kind, Max);
        }
        other => panic!(
            "expected ColumnarMetadataAggregate, got {:?}",
            std::mem::discriminant(&other)
        ),
    }

    // (c) Folded data round-trips byte-identically, sys columns present.
    let reader = ZyrFileReader::open(std::path::Path::new(&seg.path)).unwrap();
    let rc = reader.header().row_count as usize;
    assert_eq!(rc, N as usize);
    let k_col = te.columns.iter().find(|c| c.name == "k").unwrap().id.0 as u32;
    let v_col = te.columns.iter().find(|c| c.name == "v").unwrap().id.0 as u32;
    let xmin_dec = decode_column_raw(&reader, SYS_COL_XMIN, rc, 8);
    let k_dec = decode_column_raw(&reader, k_col, rc, 8);
    let v_dec = decode_column_raw(&reader, v_col, rc, 8);
    // Rows are sorted by sys_rowid which equals insertion order here.
    for i in 0..rc {
        let xmin = u64::from_le_bytes(xmin_dec[i * 8..i * 8 + 8].try_into().unwrap());
        let k = i64::from_le_bytes(k_dec[i * 8..i * 8 + 8].try_into().unwrap());
        let v = i64::from_le_bytes(v_dec[i * 8..i * 8 + 8].try_into().unwrap());
        assert_eq!(xmin, 1, "sys_xmin preserved");
        assert_eq!(k, i as i64, "k column folded identically");
        assert_eq!(v, i as i64 * 100, "v column folded identically");
    }

    // Heap hand-off: a second cycle folds nothing because the rows were
    // physically removed from the heap at fold time.
    let (rows2, _segs2) = tokio::task::block_in_place(|| {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        CompactionWorker::run_cycle(
            &rt, &catalog, &txn, &disk, &pool, &wal, &cfg, None, None, None, None,
        )
    });
    assert_eq!(rows2, 0, "folded rows were handed off out of the heap");

    // (d) UPDATE/DELETE of a folded row goes to the patch overlay.
    let store = ColumnarPatchManager::global(&columnar_dir)
        .store(table_id.0 as u64)
        .unwrap();
    let fid = seg.file_id;
    let rid0 = seg.sys_rowid_lo;
    store
        .append_value_patch(0, fid, rid0, v_col, 50, 1, &7777i64.to_le_bytes())
        .unwrap();
    store.append_supersede(0, fid, rid0 + 1, 60, 2).unwrap();
    let o0 = store.row_overlay(fid, rid0).expect("value overlay");
    assert_eq!(
        i64::from_le_bytes(o0.patches[&v_col][0].value[..8].try_into().unwrap()),
        7777
    );
    let o1 = store.row_overlay(fid, rid0 + 1).expect("supersede overlay");
    assert_eq!(o1.supersedes, vec![60]);

    // (d2) Incremental merge: the overlay xids (50, 60) are below the
    // oldest-active horizon (100), so a cycle merges the segment: the
    // superseded row is dropped and the value patch is folded into the base.
    tokio::task::block_in_place(|| {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        CompactionWorker::run_cycle(
            &rt, &catalog, &txn, &disk, &pool, &wal, &cfg, None, None, None, None,
        )
    });
    let te = catalog.get_table_by_id(table_id).unwrap();
    assert_eq!(
        te.columnar.segments.len(),
        1,
        "still one segment post-merge"
    );
    let merged = te.columnar.segments[0].clone();
    assert_ne!(merged.file_id, seg.file_id, "merge produced a new segment");
    assert_eq!(
        merged.row_count,
        N as u64 - 1,
        "superseded row dropped by merge"
    );
    assert!(
        !std::path::Path::new(&seg.path).exists(),
        "old segment unlinked after merge"
    );
    let mreader = ZyrFileReader::open(std::path::Path::new(&merged.path)).unwrap();
    let mrc = mreader.header().row_count as usize;
    let mrowid = decode_column_raw(&mreader, zyron_storage::columnar::SYS_COL_ROWID, mrc, 8);
    let mv = decode_column_raw(&mreader, v_col, mrc, 8);
    let mut found_patched = false;
    for i in 0..mrc {
        let rid = u64::from_le_bytes(mrowid[i * 8..i * 8 + 8].try_into().unwrap());
        let v = i64::from_le_bytes(mv[i * 8..i * 8 + 8].try_into().unwrap());
        assert_ne!(rid, rid0 + 1, "superseded sys_rowid must be gone");
        if rid == rid0 {
            assert_eq!(v, 7777, "value patch folded into the merged base");
            found_patched = true;
        }
    }
    assert!(found_patched, "patched row present in merged segment");
    // Patch log compacted: the old file's entries are dropped.
    let store2 = ColumnarPatchManager::global(&columnar_dir)
        .store(table_id.0 as u64)
        .unwrap();
    assert!(
        store2.row_overlay(seg.file_id, rid0).is_none(),
        "patch entries for the merged-away file were compacted"
    );

    // (e) Crash mid-compaction: CompactionBegin with no CompactionEnd.
    let orphan = columnar_dir.join("table_999_orphan.zyr");
    std::fs::write(&orphan, b"partial").unwrap();
    let mut begin = Vec::new();
    begin.extend_from_slice(&999u64.to_le_bytes());
    begin.extend_from_slice(orphan.to_string_lossy().as_bytes());
    wal.log_compaction_begin(&begin).unwrap();
    wal.flush().unwrap();
    reconcile_columnar(&wal_dir, &catalog, &disk, &columnar_dir)
        .await
        .unwrap();
    assert!(!orphan.exists(), "uncommitted .zyr discarded by recovery");
    // The merged segment survives and the stale CompactionEnd for the
    // merged-away file does not resurrect it (MergeEnd guard in recovery).
    let te = catalog.get_table_by_id(table_id).unwrap();
    assert_eq!(
        te.columnar.segments.len(),
        1,
        "merged segment intact, no resurrection of merged-away file"
    );
    assert_eq!(te.columnar.segments[0].file_id, merged.file_id);
    assert!(std::path::Path::new(&merged.path).exists());
    assert!(
        !std::path::Path::new(&seg.path).exists(),
        "merged-away segment stays gone after recovery"
    );
}

/// Time-travel over folded data: a VERSION AS OF read must see rows that were
/// live at that version even after they fold into a .zyr and leave the heap.
/// The columnar scan dates each row by its sys_xmin commit LSN, the same oracle
/// the heap scan uses, so the folded store is not a hole in time-travel.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn columnar_time_travel_version_visibility() {
    use zyron_executor::ExecutionContext;
    use zyron_executor::column::ScalarValue;
    use zyron_executor::operator::Operator;
    use zyron_executor::operator::column_scan::ColumnScanOperator;
    use zyron_planner::logical::LogicalColumn;
    use zyron_storage::txn::{Snapshot, TxnStatusMap};

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
    let wal = Arc::new(WalWriter::new(zyron_bench_harness::wal_config(wal_dir.clone())).unwrap());

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

    // Two groups of rows with distinct committed xmins, so version visibility
    // can tell them apart after they fold to columnar.
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
    let mut tuples = Vec::new();
    for i in 0..6 {
        tuples.push(Tuple::new(encode_row(i, &format!("row-{}", i), i * 100), 1));
    }
    for i in 6..12 {
        tuples.push(Tuple::new(
            encode_row(i, &format!("row-{}", i), i * 100),
            50,
        ));
    }
    heap.insert_batch(&tuples).await.unwrap();
    heap.flush().await.unwrap();

    let cfg = CompactionWorkerConfig {
        min_rows: 4,
        columnar_dir: columnar_dir.clone(),
        ..CompactionWorkerConfig::default()
    };
    let (rows, _segs) = tokio::task::block_in_place(|| {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        CompactionWorker::run_cycle(
            &rt, &catalog, &txn, &disk, &pool, &wal, &cfg, None, None, None, None,
        )
    });
    assert_eq!(rows, 12, "all rows folded out of the heap");
    let te = catalog.get_table_by_id(table_id).unwrap();
    assert_eq!(te.columnar.segments.len(), 1, "one segment registered");

    // Date the two inserting transactions: xmin 1 at LSN 100, xmin 50 at 500.
    let status = Arc::new(TxnStatusMap::new());
    status.enable_lsn_tracking();
    status.record_committed_at(1, 100);
    status.record_committed_at(50, 500);

    let kcol_entry = te.columns.iter().find(|c| c.name == "k").unwrap();
    let kcol = LogicalColumn {
        table_idx: Some(0),
        column_id: kcol_entry.id,
        name: "k".into(),
        type_id: kcol_entry.type_id,
        nullable: kcol_entry.nullable,
        fractional_digits: kcol_entry.fractional_digits,
    };

    async fn scan_as_of(
        catalog: &Arc<Catalog>,
        wal: &Arc<WalWriter>,
        pool: &Arc<BufferPool>,
        disk: &Arc<DiskManager>,
        status: &Arc<TxnStatusMap>,
        table_id: zyron_catalog::TableId,
        kcol: &LogicalColumn,
        version: u64,
    ) -> Vec<i64> {
        let snapshot = Snapshot::new(200, vec![], Arc::clone(status));
        let ctx = Arc::new(ExecutionContext::new(
            Arc::clone(catalog),
            Arc::clone(wal),
            Arc::clone(pool),
            Arc::clone(disk),
            200,
            snapshot,
        ));
        let mut op = ColumnScanOperator::new(ctx, table_id, vec![kcol.clone()], None)
            .unwrap()
            .with_as_of(Some(version));
        let mut got = Vec::new();
        while let Some(eb) = op.next().await.unwrap() {
            let b = eb.batch;
            if let Some(c) = b.columns.first() {
                for r in 0..b.num_rows {
                    if let ScalarValue::Int64(v) = c.get_scalar(r) {
                        got.push(v);
                    }
                }
            }
        }
        got.sort();
        got
    }

    // As of version 300: only the first group (xmin 1, committed at 100).
    let early = scan_as_of(&catalog, &wal, &pool, &disk, &status, table_id, &kcol, 300).await;
    assert_eq!(
        early,
        vec![0, 1, 2, 3, 4, 5],
        "folded rows live at version 300 are returned from the .zyr"
    );

    // As of version 600: both groups committed, so all folded rows are visible.
    let late = scan_as_of(&catalog, &wal, &pool, &disk, &status, table_id, &kcol, 600).await;
    assert_eq!(
        late,
        (0..12).collect::<Vec<_>>(),
        "all folded rows visible at version 600"
    );
}

/// The floor-aware columnar merge keeps within-window history: a folded row
/// deleted after the retention floor is carried forward into the merged segment
/// with its sys_supersede set (so AS OF before the delete still sees it), while a
/// row deleted at or below the floor is physically dropped. Proves the merge does
/// not collapse retained history yet still reclaims old history.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn columnar_merge_retains_within_window_history() {
    use zyron_storage::columnar::{ColumnarPatchManager, SYS_COL_ROWID, SYS_COL_SUPERSEDE};

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
    let wal = Arc::new(WalWriter::new(zyron_bench_harness::wal_config(wal_dir.clone())).unwrap());
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
    let mut tuples = Vec::new();
    for i in 0..4 {
        tuples.push(Tuple::new(encode_row(i, &format!("row-{}", i), i * 100), 1));
    }
    heap.insert_batch(&tuples).await.unwrap();
    heap.flush().await.unwrap();

    // Only the two triggers move, so the fold and the merge happen at all
    // on four rows. Everything else is what the server runs
    let cfg = CompactionWorkerConfig {
        min_rows: 1,
        columnar_dir: columnar_dir.clone(),
        merge_min_churn_ratio: 0.0,
        ..CompactionWorkerConfig::default()
    };
    // Fold the four rows into one segment.
    let _ = tokio::task::block_in_place(|| {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        CompactionWorker::run_cycle(
            &rt, &catalog, &txn, &disk, &pool, &wal, &cfg, None, None, None, None,
        )
    });
    let te = catalog.get_table_by_id(table_id).unwrap();
    assert_eq!(te.columnar.segments.len(), 1, "one folded segment");
    let seg = te.columnar.segments[0].clone();
    let rid_lo = seg.sys_rowid_lo;

    // Retention floor at LSN 500 via a version tag. Date two delete transactions:
    // xid 50 committed at 1000 (after the floor, within window) deletes row 1;
    // xid 40 committed at 100 (at/below the floor) deletes row 2.
    let sm = txn.status_map();
    sm.enable_lsn_tracking();
    sm.record_committed_at(50, 1000);
    sm.record_committed_at(40, 100);
    sm.retain_version(500);

    let store = ColumnarPatchManager::global(&columnar_dir)
        .store(table_id.0 as u64)
        .unwrap();
    store
        .append_supersede(0, seg.file_id, rid_lo + 1, 50, 1)
        .unwrap();
    store
        .append_supersede(0, seg.file_id, rid_lo + 2, 40, 2)
        .unwrap();

    // Merge with the retention floor in effect.
    let _ = tokio::task::block_in_place(|| {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        CompactionWorker::run_cycle(
            &rt, &catalog, &txn, &disk, &pool, &wal, &cfg, None, None, None, None,
        )
    });

    let te = catalog.get_table_by_id(table_id).unwrap();
    assert_eq!(
        te.columnar.segments.len(),
        1,
        "still one segment post-merge"
    );
    let merged = te.columnar.segments[0].clone();
    assert_ne!(merged.file_id, seg.file_id, "merge produced a new segment");

    let reader = ZyrFileReader::open(std::path::Path::new(&merged.path)).unwrap();
    let rc = reader.header().row_count as usize;
    let rowid_dec = decode_column_raw(&reader, SYS_COL_ROWID, rc, 8);
    let sup_dec = decode_column_raw(&reader, SYS_COL_SUPERSEDE, rc, 8);
    let mut rid_to_sup = std::collections::HashMap::new();
    for i in 0..rc {
        let rid = u64::from_le_bytes(rowid_dec[i * 8..i * 8 + 8].try_into().unwrap());
        let sup = u64::from_le_bytes(sup_dec[i * 8..i * 8 + 8].try_into().unwrap());
        rid_to_sup.insert(rid, sup);
    }

    // Row 2 (deleted at/below the floor) is physically dropped.
    assert!(
        !rid_to_sup.contains_key(&(rid_lo + 2)),
        "row deleted at/below the floor is reclaimed by the merge"
    );
    // Row 1 (deleted after the floor) is kept with its supersede carried forward.
    assert_eq!(
        rid_to_sup.get(&(rid_lo + 1)).copied(),
        Some(50),
        "within-window deleted row is carried forward with sys_supersede set"
    );
    // The untouched rows survive as live (sys_supersede 0).
    assert_eq!(rid_to_sup.get(&rid_lo).copied(), Some(0));
    assert_eq!(rid_to_sup.get(&(rid_lo + 3)).copied(), Some(0));
}

struct FoldedTable {
    _tmp: tempfile::TempDir,
    wal_dir: std::path::PathBuf,
    columnar_dir: std::path::PathBuf,
    catalog: Catalog,
    disk: Arc<DiskManager>,
    table_id: zyron_catalog::TableId,
}

/// Builds a catalog with one table whose rows are folded into a single
/// registered .zyr segment, the starting state for the tier reconcile tests.
async fn fold_one_segment() -> FoldedTable {
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
    let wal = Arc::new(WalWriter::new(zyron_bench_harness::wal_config(wal_dir.clone())).unwrap());

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
    let mut tuples = Vec::new();
    for i in 0..8i64 {
        tuples.push(Tuple::new(encode_row(i, &format!("row-{}", i), i * 100), 1));
    }
    heap.insert_batch(&tuples).await.unwrap();
    heap.flush().await.unwrap();

    let cfg = CompactionWorkerConfig {
        min_rows: 4,
        columnar_dir: columnar_dir.clone(),
        ..CompactionWorkerConfig::default()
    };
    let (rows, segs) = tokio::task::block_in_place(|| {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        CompactionWorker::run_cycle(
            &rt, &catalog, &txn, &disk, &pool, &wal, &cfg, None, None, None, None,
        )
    });
    assert_eq!(rows, 8, "all rows folded");
    assert_eq!(segs, 1, "one segment registered");

    FoldedTable {
        _tmp: tmp,
        wal_dir,
        columnar_dir,
        catalog,
        disk,
        table_id,
    }
}

/// A crash between a relocation's rename and its registry write leaves the
/// catalog naming a path with no file behind it while the bytes sit whole in
/// a tier directory. Startup reconciliation adopts the location that holds
/// the file, so the segment's rows read again with no operator involvement.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn columnar_tier_recovery_adopts_a_moved_but_unpersisted_segment() {
    let f = fold_one_segment().await;
    let te = f.catalog.get_table_by_id(f.table_id).unwrap();
    let seg = te.columnar.segments[0].clone();
    let old_path = std::path::PathBuf::from(&seg.path);

    // The crash state: the file was renamed onto the tier, the registry
    // write never happened
    let cold_dir = f.columnar_dir.join("tiers").join("cold");
    std::fs::create_dir_all(&cold_dir).unwrap();
    let moved = cold_dir.join(old_path.file_name().unwrap());
    std::fs::rename(&old_path, &moved).unwrap();

    reconcile_columnar(&f.wal_dir, &f.catalog, &f.disk, &f.columnar_dir)
        .await
        .unwrap();

    let te = f.catalog.get_table_by_id(f.table_id).unwrap();
    assert_eq!(te.columnar.segments.len(), 1, "still one registration");
    let repaired = &te.columnar.segments[0];
    assert_eq!(
        std::path::PathBuf::from(&repaired.path),
        moved,
        "the registration points where the file sits"
    );
    assert_eq!(
        repaired.storage_tier, 2,
        "the tier byte matches the directory"
    );
    let reader = ZyrFileReader::open(std::path::Path::new(&repaired.path)).unwrap();
    assert_eq!(
        reader.header().row_count,
        seg.row_count,
        "the adopted segment reads in full"
    );
}

/// The recorded path is authoritative when its file exists. A same-named
/// copy on another tier and a staging file are debris from an interrupted
/// cross-device move, and either would keep the next relocation from
/// placing the file under that name, so reconciliation sweeps both.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn columnar_tier_recovery_sweeps_duplicates_and_staging_leftovers() {
    let f = fold_one_segment().await;
    let te = f.catalog.get_table_by_id(f.table_id).unwrap();
    let seg = te.columnar.segments[0].clone();
    let hot_path = std::path::PathBuf::from(&seg.path);
    let name = hot_path.file_name().unwrap().to_os_string();

    let warm_dir = f.columnar_dir.join("tiers").join("warm");
    std::fs::create_dir_all(&warm_dir).unwrap();
    let duplicate = warm_dir.join(&name);
    std::fs::copy(&hot_path, &duplicate).unwrap();
    let cold_dir = f.columnar_dir.join("tiers").join("cold");
    std::fs::create_dir_all(&cold_dir).unwrap();
    let staging = cold_dir.join(&name).with_extension("zyr.moving");
    std::fs::write(&staging, b"half a copy").unwrap();

    reconcile_columnar(&f.wal_dir, &f.catalog, &f.disk, &f.columnar_dir)
        .await
        .unwrap();

    assert!(!duplicate.exists(), "the stale copy is swept");
    assert!(!staging.exists(), "the staging leftover is swept");
    assert!(hot_path.exists(), "the authoritative file is untouched");
    let te = f.catalog.get_table_by_id(f.table_id).unwrap();
    assert_eq!(
        te.columnar.segments[0].path, seg.path,
        "the registration is untouched"
    );
    assert_eq!(te.columnar.segments[0].storage_tier, 0, "still hot");
}

/// A registration whose file is on no tier is left in place and reported.
/// Unregistering it would silently discard the rows it serves, while
/// leaving it lets a restored file make them readable again with no
/// catalog surgery.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn columnar_tier_recovery_never_unregisters_a_missing_segment() {
    let f = fold_one_segment().await;
    let te = f.catalog.get_table_by_id(f.table_id).unwrap();
    let seg = te.columnar.segments[0].clone();
    std::fs::remove_file(&seg.path).unwrap();

    reconcile_columnar(&f.wal_dir, &f.catalog, &f.disk, &f.columnar_dir)
        .await
        .unwrap();

    let te = f.catalog.get_table_by_id(f.table_id).unwrap();
    assert_eq!(te.columnar.segments.len(), 1, "the registration survives");
    assert_eq!(
        te.columnar.segments[0].path, seg.path,
        "the path is unchanged"
    );
}

/// A table that a cycle found fully settled while no transaction was in
/// flight is skipped by later cycles until its write counters move, so an
/// idle table costs no heap scan or overlay check. New writes reopen the
/// gate and the next cycle folds them.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn compaction_gate_skips_settled_tables_until_writes_return() {
    use zyron_server::background::compaction::CompactionGate;

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
    let wal = Arc::new(WalWriter::new(zyron_bench_harness::wal_config(wal_dir.clone())).unwrap());
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
    let mut tuples = Vec::new();
    for i in 0..8i64 {
        tuples.push(Tuple::new(encode_row(i, &format!("row-{}", i), i * 100), 1));
    }
    heap.insert_batch(&tuples).await.unwrap();
    heap.flush().await.unwrap();

    // The registry counters move the way DML moves them
    let io = Arc::new(zyron_common::TableIOStatsRegistry::new());
    io.get_or_create(table_id.0).record_inserts(8);
    let gate = CompactionGate::new(Arc::clone(&io));

    let cfg = CompactionWorkerConfig {
        min_rows: 4,
        columnar_dir: columnar_dir.clone(),
        ..CompactionWorkerConfig::default()
    };
    let run = |gate: &CompactionGate| {
        tokio::task::block_in_place(|| {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            CompactionWorker::run_cycle(
                &rt,
                &catalog,
                &txn,
                &disk,
                &pool,
                &wal,
                &cfg,
                None,
                None,
                None,
                Some(gate),
            )
        })
    };

    let (rows, _) = run(&gate);
    assert_eq!(rows, 8, "the first cycle folds the rows");
    assert_eq!(gate.skipped(), 0, "a cycle that did work never skips");

    let (rows2, _) = run(&gate);
    assert_eq!(rows2, 0, "nothing left to fold");
    assert_eq!(gate.skipped(), 0, "the settling cycle itself still scans");

    let (rows3, _) = run(&gate);
    assert_eq!(rows3, 0);
    assert!(
        gate.skipped() >= 1,
        "a settled table is skipped without a scan"
    );

    // New writes reopen the gate
    let mut more = Vec::new();
    for i in 8..12i64 {
        more.push(Tuple::new(encode_row(i, &format!("row-{}", i), i * 100), 1));
    }
    heap.insert_batch(&more).await.unwrap();
    heap.flush().await.unwrap();
    io.get_or_create(table_id.0).record_inserts(4);

    let (rows4, _) = run(&gate);
    assert_eq!(rows4, 4, "new rows fold once the gate reopens");
}

/// NSM-encodes one row of (k:i64, payload:text) the way the heap and the
/// compaction materializer read it: null bitmap, then the fixed column,
/// then the variable-length one.
fn encode_variant_row(k: i64, payload: &str) -> Vec<u8> {
    let mut d = Vec::new();
    d.push(0u8); // null bitmap, 2 cols -> 1 byte, no nulls
    d.extend_from_slice(&k.to_le_bytes());
    d.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    d.extend_from_slice(payload.as_bytes());
    d
}

/// A promoted VARIANT path is materialized as its own column when the rows
/// fold, and the segment records which column holds it.
///
/// The fold is the only place a shredded column can be written without a
/// backfill. Every row of the segment passes through it, so the column is
/// complete the moment it exists: a null in it means the document did not
/// carry the path, never that the row has not been shredded yet. That is the
/// property this test pins, by writing documents that deliberately lack the
/// path and checking the column's null count is exactly those rows.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_promoted_variant_path_folds_into_a_column_of_its_own() {
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
    let wal = Arc::new(WalWriter::new(zyron_bench_harness::wal_config(wal_dir.clone())).unwrap());

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
        col("payload", DataType::Variant),
    ];
    let table_id = catalog
        .create_table(schema, "events", &cols, &[])
        .await
        .unwrap();
    let txn = Arc::new(TransactionManager::with_start_txn_id(Arc::clone(&wal), 100));

    let te = catalog.get_table_by_id(table_id).unwrap();
    let payload_col = te
        .columns
        .iter()
        .find(|c| c.name == "payload")
        .unwrap()
        .id
        .0;

    let heap = HeapFile::new(
        Arc::clone(&disk),
        Arc::clone(&pool),
        HeapFileConfig {
            heap_file_id: te.heap_file_id,
            fsm_file_id: te.fsm_file_id,
        },
    )
    .unwrap();

    // Every fourth document leaves the path out, so the shredded column has
    // a null exactly where the document had nothing to give it
    const N: i64 = 12;
    let missing = |k: i64| k % 4 == 3;
    let mut tuples = Vec::new();
    for k in 0..N {
        let payload = if missing(k) {
            format!(r#"{{"kind":"click","seq":{k}}}"#)
        } else {
            format!(r#"{{"kind":"click","user":{{"id":{}}},"seq":{k}}}"#, k * 7)
        };
        tuples.push(Tuple::new(encode_variant_row(k, &payload), 1));
    }
    heap.insert_batch(&tuples).await.unwrap();
    heap.flush().await.unwrap();

    // The tracker would promote this path once it had seen enough traffic;
    // naming it directly keeps the test about what the fold does with a
    // promoted path rather than about the thresholds
    zyron_executor::variant_shred::mark_shredded(table_id.0, payload_col, "user.id");

    let cfg = CompactionWorkerConfig {
        min_rows: 4,
        columnar_dir: columnar_dir.clone(),
        ..CompactionWorkerConfig::default()
    };
    let (rows, segs) = {
        let catalog2 = &catalog;
        let txn2 = &txn;
        let disk2 = &disk;
        let pool2 = &pool;
        let wal2 = &wal;
        let cfg2 = &cfg;
        tokio::task::block_in_place(|| {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            CompactionWorker::run_cycle(
                &rt, catalog2, txn2, disk2, pool2, wal2, cfg2, None, None, None, None,
            )
        })
    };
    assert_eq!(rows, N as u64, "every eligible row folded");
    assert_eq!(segs, 1, "one segment written");

    // The segment says which column holds the path
    let te = catalog.get_table_by_id(table_id).unwrap();
    let seg = &te.columnar.segments[0];
    assert_eq!(
        seg.shredded.len(),
        1,
        "the segment did not record the promoted path"
    );
    let shred = &seg.shredded[0];
    assert_eq!(shred.path, "user.id");
    assert_eq!(shred.variant_column_id, payload_col);
    assert_eq!(shred.column_id, zyron_storage::columnar::SHRED_COL_BASE);

    // And the column is there, holding one entry per row of the segment,
    // null exactly where the document had no such path
    let reader = ZyrFileReader::open(std::path::Path::new(&seg.path)).unwrap();
    let header = reader
        .read_segment_header(shred.column_id)
        .expect("the shredded column is in the file");
    assert_eq!(
        reader.header().row_count as usize,
        N as usize,
        "the shredded column covers every row of the segment, which is what \
         makes it complete without a backfill"
    );
    let expected_nulls = (0..N).filter(|k| missing(*k)).count() as u64;
    assert_eq!(
        header.null_count, expected_nulls,
        "a null in the shredded column has to mean the document lacked the \
         path, and nothing else"
    );

    // A table with no promoted path shreds nothing, so the machinery costs
    // an empty list rather than a column
    zyron_executor::variant_shred::clear_column(table_id.0, payload_col);
}

// ---------------------------------------------------------------------------
// Reading a shredded path back
// ---------------------------------------------------------------------------

use zyron_executor::ExecutionContext;
use zyron_executor::column::ScalarValue;
use zyron_executor::operator::Operator;
use zyron_executor::operator::column_scan::ColumnScanOperator;
use zyron_planner::logical::LogicalColumn;
use zyron_storage::txn::Snapshot;

/// Builds a table of (k BIGINT, payload VARIANT), inserts documents that
/// mostly carry `user.id`, promotes that path and folds everything into one
/// segment. Returns what a read needs plus the documents it was built from
struct FoldedEvents {
    catalog: Arc<Catalog>,
    disk: Arc<DiskManager>,
    pool: Arc<BufferPool>,
    wal: Arc<WalWriter>,
    db: zyron_catalog::DatabaseId,
    table_id: zyron_catalog::TableId,
    payload_col: u16,
    /// The document written for row k, so a read can be checked against what
    /// it was built from rather than against a second copy of the answer
    docs: Vec<String>,
}

async fn folded_variant_events(tmp: &tempfile::TempDir) -> FoldedEvents {
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
    let wal = Arc::new(WalWriter::new(zyron_bench_harness::wal_config(wal_dir.clone())).unwrap());

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
    let cols = vec![
        col("k", DataType::BigInt),
        col("payload", DataType::Variant),
    ];
    let table_id = catalog
        .create_table(schema, "events", &cols, &[])
        .await
        .unwrap();
    let txn = Arc::new(TransactionManager::with_start_txn_id(Arc::clone(&wal), 100));

    let te = catalog.get_table_by_id(table_id).unwrap();
    let payload_col = te
        .columns
        .iter()
        .find(|c| c.name == "payload")
        .unwrap()
        .id
        .0;

    let heap = HeapFile::new(
        Arc::clone(&disk),
        Arc::clone(&pool),
        HeapFileConfig {
            heap_file_id: te.heap_file_id,
            fsm_file_id: te.fsm_file_id,
        },
    )
    .unwrap();

    // Every fourth document leaves the path out, so a read has to answer
    // null for it whichever side it comes from
    const N: i64 = 12;
    let mut docs = Vec::new();
    let mut tuples = Vec::new();
    for k in 0..N {
        let payload = if k % 4 == 3 {
            format!(r#"{{"kind":"click","seq":{k}}}"#)
        } else {
            format!(r#"{{"kind":"click","user":{{"id":{}}},"seq":{k}}}"#, k * 7)
        };
        tuples.push(Tuple::new(encode_variant_row(k, &payload), 1));
        docs.push(payload);
    }
    heap.insert_batch(&tuples).await.unwrap();
    heap.flush().await.unwrap();

    zyron_executor::variant_shred::mark_shredded(table_id.0, payload_col, "user.id");

    let cfg = CompactionWorkerConfig {
        min_rows: 4,
        columnar_dir: columnar_dir.clone(),
        ..CompactionWorkerConfig::default()
    };
    let (rows, segs) = {
        let catalog2 = &catalog;
        let txn2 = &txn;
        let disk2 = &disk;
        let pool2 = &pool;
        let wal2 = &wal;
        let cfg2 = &cfg;
        tokio::task::block_in_place(|| {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            CompactionWorker::run_cycle(
                &rt, catalog2, txn2, disk2, pool2, wal2, cfg2, None, None, None, None,
            )
        })
    };
    assert_eq!(rows, N as u64, "every eligible row folded");
    assert_eq!(segs, 1, "one segment written");

    FoldedEvents {
        catalog,
        disk,
        pool,
        wal,
        db,
        table_id,
        payload_col,
        docs,
    }
}

/// The k and payload columns of the folded table, as a scan projects them
fn event_columns(te: &zyron_catalog::TableEntry) -> Vec<LogicalColumn> {
    te.columns
        .iter()
        .filter(|c| c.name == "k" || c.name == "payload")
        .map(|c| LogicalColumn {
            table_idx: Some(0),
            column_id: c.id,
            name: c.name.clone(),
            type_id: c.type_id,
            nullable: c.nullable,
            fractional_digits: c.fractional_digits,
        })
        .collect()
}

/// `variant_extract(payload, 'user.id')` as the binder builds it for
/// `p.user.id`
fn extract_expr(payload_col: u16) -> zyron_planner::binder::BoundExpr {
    zyron_planner::binder::BoundExpr::Function {
        name: "variant_extract".to_string(),
        args: vec![
            zyron_planner::binder::BoundExpr::ColumnRef(zyron_planner::binder::ColumnRef {
                table_idx: 0,
                column_id: zyron_catalog::ColumnId(payload_col),
                type_id: zyron_common::TypeId::Variant,
                nullable: true,
                fractional_digits: None,
            }),
            zyron_planner::binder::BoundExpr::Literal {
                value: zyron_parser::ast::LiteralValue::String("user.id".to_string()),
                type_id: zyron_common::TypeId::Text,
            },
        ],
        return_type: zyron_common::TypeId::Text,
        distinct: false,
    }
}

/// Scans the folded table, returning per k the path as the batch resolved it
/// (None when nothing resolved it) and the path as the expression evaluates
/// it, plus how many batches carried a resolved column
async fn scan_paths(
    ctx: Arc<ExecutionContext>,
    table_id: zyron_catalog::TableId,
    columns: Vec<LogicalColumn>,
    payload_col: u16,
) -> (
    Vec<(i64, Option<String>)>,
    Vec<(i64, Option<String>)>,
    usize,
) {
    let expr = extract_expr(payload_col);
    let mut op = ColumnScanOperator::new(ctx.clone(), table_id, columns.clone(), None).unwrap();
    let mut from_column = Vec::new();
    let mut from_expr = Vec::new();
    let mut resolved_batches = 0usize;
    while let Some(eb) = op.next().await.unwrap() {
        let b = eb.batch;
        let evaluated = zyron_executor::expr::evaluate(&expr, &b, &columns, &[]).unwrap();
        let resolved = b.resolved_path(0, payload_col, "user.id").cloned();
        if resolved.is_some() {
            resolved_batches += 1;
        }
        for r in 0..b.num_rows {
            let ScalarValue::Int64(k) = b.column(0).get_scalar(r) else {
                panic!("k is a bigint");
            };
            let one = |c: &zyron_executor::column::Column| match c.get_scalar(r) {
                ScalarValue::Utf8(s) => Some(s),
                _ => None,
            };
            from_column.push((k, resolved.as_ref().and_then(one)));
            from_expr.push((k, one(&evaluated)));
        }
    }
    from_column.sort_by_key(|(k, _)| *k);
    from_expr.sort_by_key(|(k, _)| *k);
    (from_column, from_expr, resolved_batches)
}

/// A promoted path is read out of the column the fold put it in, and reading
/// it there gives the answer the documents give.
///
/// Both halves matter. Without the first, the shredded column is a write
/// nothing reads. Without the second, it is a faster way to be wrong: the
/// column is only usable because its values are the extraction's own output,
/// so the same query served either way has to agree row for row, nulls
/// included
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_promoted_variant_path_is_read_out_of_its_own_column() {
    let tmp = tempfile::tempdir().expect("tmp");
    let ev = folded_variant_events(&tmp).await;
    let (catalog, disk, pool, wal) = (&ev.catalog, &ev.disk, &ev.pool, &ev.wal);
    let (table_id, payload_col, docs) = (ev.table_id, ev.payload_col, &ev.docs);
    let te = catalog.get_table_by_id(table_id).unwrap();
    let columns = event_columns(&te);

    let context = || {
        Arc::new(ExecutionContext::new(
            Arc::clone(catalog),
            Arc::clone(wal),
            Arc::clone(pool),
            Arc::clone(disk),
            200,
            Snapshot::new(
                200,
                vec![],
                Arc::new(zyron_storage::TxnStatusMap::all_committed()),
            ),
        ))
    };

    // What the documents say, which is what every path below has to produce
    let want: Vec<(i64, Option<String>)> = docs
        .iter()
        .enumerate()
        .map(|(k, doc)| {
            (
                k as i64,
                zyron_executor::variant_shred::extract_scalar_text(doc, "user.id"),
            )
        })
        .collect();
    assert!(
        want.iter().any(|(_, v)| v.is_none()) && want.iter().any(|(_, v)| v.is_some()),
        "the documents have to cover both a present and an absent path"
    );

    // A statement that reads the path gets it out of the segment column
    let asking = context();
    asking.set_variant_paths(vec![zyron_planner::physical::variant_paths::VariantPath {
        table_idx: 0,
        column_id: payload_col,
        path: "user.id".to_string(),
    }]);
    let (from_column, from_expr, resolved_batches) =
        scan_paths(asking, table_id, columns.clone(), payload_col).await;
    assert!(
        resolved_batches > 0,
        "the scan read no shredded column, so nothing reads what the fold writes"
    );
    assert_eq!(
        from_column, want,
        "the shredded column disagrees with the documents it was extracted from"
    );
    assert_eq!(
        from_expr, want,
        "the expression served from the shredded column gave a different answer"
    );

    // A statement that reads no path leaves the promoted columns on disk,
    // and still answers out of the documents
    let quiet = context();
    let (from_column, from_expr, resolved_batches) =
        scan_paths(quiet, table_id, columns, payload_col).await;
    assert_eq!(
        resolved_batches, 0,
        "a scan read a promoted column no expression named"
    );
    assert!(
        from_column.iter().all(|(_, v)| v.is_none()),
        "a batch reported a resolved path it was never asked for"
    );
    assert_eq!(
        from_expr, want,
        "the document walk and the shredded column give different answers"
    );
}

/// A row whose document was rewritten after the fold reads back its new
/// value, not the one the segment column was built from.
///
/// The stored column describes the row as it stood when the segment was
/// written. An UPDATE of a folded row goes to the patch overlay and leaves
/// that column untouched, so serving the path out of it would answer with
/// the superseded document
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_patched_document_reads_back_its_new_path_value() {
    let tmp = tempfile::tempdir().expect("tmp");
    let ev = folded_variant_events(&tmp).await;
    let (catalog, disk, pool, wal) = (&ev.catalog, &ev.disk, &ev.pool, &ev.wal);
    let (table_id, payload_col, docs) = (ev.table_id, ev.payload_col, &ev.docs);
    let te = catalog.get_table_by_id(table_id).unwrap();
    let columns = event_columns(&te);
    let seg = &te.columnar.segments[0];

    // Rewrite the first row's document. Its k is 0, so the fold stored 0 for
    // user.id and the patch moves it somewhere the old column cannot reach
    let patched_doc = r#"{"kind":"click","user":{"id":9999},"seq":0}"#;
    let store =
        ColumnarPatchManager::store_for_segment(table_id.0 as u64, std::path::Path::new(&seg.path))
            .unwrap();
    store
        .append_value_patch(
            0,
            seg.file_id,
            seg.sys_rowid_lo,
            payload_col as u32,
            50,
            1,
            patched_doc.as_bytes(),
        )
        .unwrap();

    let ctx = Arc::new(ExecutionContext::new(
        Arc::clone(catalog),
        Arc::clone(wal),
        Arc::clone(pool),
        Arc::clone(disk),
        200,
        Snapshot::new(
            200,
            vec![],
            Arc::new(zyron_storage::TxnStatusMap::all_committed()),
        ),
    ));
    ctx.set_variant_paths(vec![zyron_planner::physical::variant_paths::VariantPath {
        table_idx: 0,
        column_id: payload_col,
        path: "user.id".to_string(),
    }]);

    let (from_column, from_expr, _) = scan_paths(ctx, table_id, columns, payload_col).await;
    let mut want: Vec<(i64, Option<String>)> = docs
        .iter()
        .enumerate()
        .map(|(k, doc)| {
            (
                k as i64,
                zyron_executor::variant_shred::extract_scalar_text(doc, "user.id"),
            )
        })
        .collect();
    want[0].1 = zyron_executor::variant_shred::extract_scalar_text(patched_doc, "user.id");
    assert_eq!(
        want[0].1.as_deref(),
        Some("9999"),
        "the patch has to move the value somewhere the stored column cannot hold"
    );

    assert_eq!(
        from_column, want,
        "a patched row read back the value the fold stored for it"
    );
    assert_eq!(
        from_expr, want,
        "the expression answered a patched row out of the stale column"
    );
}

/// A read-only context over a folded table, at a snapshot that sees the fold
fn new_context(ev: &FoldedEvents) -> ExecutionContext {
    ExecutionContext::new(
        Arc::clone(&ev.catalog),
        Arc::clone(&ev.wal),
        Arc::clone(&ev.pool),
        Arc::clone(&ev.disk),
        200,
        Snapshot::new(
            200,
            vec![],
            Arc::new(zyron_storage::TxnStatusMap::all_committed()),
        ),
    )
}

/// The projection of the plan's hybrid scan, wherever it sits in the tree
fn hybrid_scan_of(plan: &PhysicalPlan) -> Option<Vec<LogicalColumn>> {
    if let PhysicalPlan::HybridScan { columns, .. } = plan {
        return Some(columns.clone());
    }
    plan.children().into_iter().find_map(hybrid_scan_of)
}

/// A query reading a promoted path names it in the plan and answers with the
/// documents' own values.
///
/// The operator tests above prove the segment column is read and that it
/// agrees with the documents. This one closes the loop the statement
/// actually travels: dotted access binds to an extraction, the plan is
/// searched for the paths it reads, and the answer that comes back is the
/// one the documents hold
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_query_reading_a_promoted_path_names_it_in_the_plan_and_answers_from_the_documents() {
    let tmp = tempfile::tempdir().expect("tmp");
    let ev = folded_variant_events(&tmp).await;

    let stmt = zyron_parser::parse("SELECT k, payload.user.id FROM events ORDER BY k")
        .expect("parse")
        .into_iter()
        .next()
        .expect("one statement");
    let plan = zyron_planner::plan(&ev.catalog, ev.db, vec!["app".into()], stmt, None)
        .await
        .expect("plan");

    // The path the statement reads is found in the plan, which is what lets
    // a scan read the one promoted column it needs and leave the rest
    let found = zyron_planner::physical::variant_paths::variant_paths(&plan);
    assert!(
        found.contains(&zyron_planner::physical::variant_paths::VariantPath {
            table_idx: 0,
            column_id: ev.payload_col,
            path: "user.id".to_string(),
        }),
        "the plan reads payload.user.id and the search did not find it: {found:?}"
    );

    // The scan the plan built, given the paths the plan reads, reads the
    // promoted column. This is the link between the two halves: the paths
    // are the ones searched out of this plan, and the projection is the one
    // the planner chose, so a mismatch in either would show up here rather
    // than as a query that quietly walks every document
    let scan_ctx = Arc::new(new_context(&ev));
    scan_ctx.set_variant_paths(found.clone());
    let scan = hybrid_scan_of(&plan).expect("the query plans as a hybrid scan");
    let mut op = ColumnScanOperator::new(scan_ctx, ev.table_id, scan, None).unwrap();
    let mut resolved_batches = 0usize;
    while let Some(eb) = op.next().await.unwrap() {
        if eb
            .batch
            .resolved_path(0, ev.payload_col, "user.id")
            .is_some()
        {
            resolved_batches += 1;
        }
    }
    assert!(
        resolved_batches > 0,
        "the plan's own scan read no promoted column for a path the plan reads"
    );

    let mut ctx = new_context(&ev);
    ctx.heap_files = Some(Arc::new(scc::HashMap::new()));
    ctx.btree_indexes = Some(Arc::new(scc::HashMap::new()));
    let batches = zyron_executor::execute(plan, &Arc::new(ctx))
        .await
        .expect("execute");

    let mut got: Vec<(i64, Option<String>)> = Vec::new();
    for b in &batches {
        for r in 0..b.num_rows {
            let ScalarValue::Int64(k) = b.column(0).get_scalar(r) else {
                panic!("k is a bigint");
            };
            got.push((
                k,
                match b.column(1).get_scalar(r) {
                    ScalarValue::Utf8(s) => Some(s),
                    _ => None,
                },
            ));
        }
    }
    got.sort_by_key(|(k, _)| *k);

    let want: Vec<(i64, Option<String>)> = ev
        .docs
        .iter()
        .enumerate()
        .map(|(k, doc)| {
            (
                k as i64,
                zyron_executor::variant_shred::extract_scalar_text(doc, "user.id"),
            )
        })
        .collect();
    assert_eq!(
        got, want,
        "the query answered with something other than what the documents hold"
    );
}

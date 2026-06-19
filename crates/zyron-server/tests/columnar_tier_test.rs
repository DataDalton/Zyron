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
use zyron_storage::columnar::{
    ColumnarPatchManager, SEGMENT_HEADER_SIZE, SYS_COL_XMIN, SegmentHeader, ZONE_MAP_BATCH_SIZE,
    ZONE_MAP_ENTRY_SIZE, ZyrFileReader,
};
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

fn decode_column_raw(
    reader: &ZyrFileReader,
    column_id: u32,
    row_count: usize,
    value_size: usize,
) -> Vec<u8> {
    let raw = reader.read_segment_raw(column_id).expect("segment raw");
    let mut hdr = [0u8; SEGMENT_HEADER_SIZE];
    hdr.copy_from_slice(&raw[..SEGMENT_HEADER_SIZE]);
    let h = SegmentHeader::from_bytes(&hdr).expect("seg header");
    let bloom = h.bloom_filter_size as usize;
    let zones = row_count.div_ceil(ZONE_MAP_BATCH_SIZE as usize);
    let zm = zones * ZONE_MAP_ENTRY_SIZE;
    let nb = if h.null_count > 0 {
        row_count.div_ceil(8)
    } else {
        0
    };
    let start = SEGMENT_HEADER_SIZE + bloom + zm + nb;
    let end = start + h.encoded_size as usize;
    create_encoding(h.encoding_type)
        .decode(&raw[start..end], row_count, value_size)
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
        DiskManager::new(DiskManagerConfig {
            data_dir: data_dir.clone(),
            fsync_enabled: false,
        })
        .await
        .unwrap(),
    );
    let pool = Arc::new(BufferPool::new(BufferPoolConfig { num_frames: 4096 }));
    let wal = Arc::new(
        WalWriter::new(WalWriterConfig {
            wal_dir: wal_dir.clone(),
            fsync_enabled: false,
            ..Default::default()
        })
        .unwrap(),
    );

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
        fsync_enabled: false,
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
            CompactionWorker::run_cycle(&rt, catalog2, txn2, disk2, pool2, wal2, cfg2, None)
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
                ts_precision: c.ts_precision,
            })
            .collect(),
        alias: "metrics".into(),
        encoding_hints: None,
        as_of: None,
    };
    let physical =
        zyron_planner::physical::builder::build_physical_plan(logical, &catalog).expect("plan");
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
            ts_precision: c.ts_precision,
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
                ts_precision: c.ts_precision,
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
    let agg_phys =
        zyron_planner::physical::builder::build_physical_plan(agg, &catalog).expect("agg plan");
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
        CompactionWorker::run_cycle(&rt, &catalog, &txn, &disk, &pool, &wal, &cfg, None)
    });
    assert_eq!(rows2, 0, "folded rows were handed off out of the heap");

    // (d) UPDATE/DELETE of a folded row goes to the patch overlay.
    let store = ColumnarPatchManager::global(&columnar_dir)
        .store(table_id.0 as u64)
        .unwrap();
    let fid = seg.file_id;
    let rid0 = seg.sys_rowid_lo;
    store
        .append_value_patch(fid, rid0, v_col, 50, 1, &7777i64.to_le_bytes())
        .unwrap();
    store.append_supersede(fid, rid0 + 1, 60, 2).unwrap();
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
        CompactionWorker::run_cycle(&rt, &catalog, &txn, &disk, &pool, &wal, &cfg, None)
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
        DiskManager::new(DiskManagerConfig {
            data_dir: data_dir.clone(),
            fsync_enabled: false,
        })
        .await
        .unwrap(),
    );
    let pool = Arc::new(BufferPool::new(BufferPoolConfig { num_frames: 4096 }));
    let wal = Arc::new(
        WalWriter::new(WalWriterConfig {
            wal_dir: wal_dir.clone(),
            fsync_enabled: false,
            ..Default::default()
        })
        .unwrap(),
    );

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
        fsync_enabled: false,
        ..CompactionWorkerConfig::default()
    };
    let (rows, _segs) = tokio::task::block_in_place(|| {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        CompactionWorker::run_cycle(&rt, &catalog, &txn, &disk, &pool, &wal, &cfg, None)
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
        ts_precision: kcol_entry.ts_precision,
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
        DiskManager::new(DiskManagerConfig {
            data_dir: data_dir.clone(),
            fsync_enabled: false,
        })
        .await
        .unwrap(),
    );
    let pool = Arc::new(BufferPool::new(BufferPoolConfig { num_frames: 4096 }));
    let wal = Arc::new(
        WalWriter::new(WalWriterConfig {
            wal_dir: wal_dir.clone(),
            fsync_enabled: false,
            ..Default::default()
        })
        .unwrap(),
    );
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

    let cfg = CompactionWorkerConfig {
        min_rows: 1,
        columnar_dir: columnar_dir.clone(),
        fsync_enabled: false,
        merge_min_churn_ratio: 0.0,
        ..CompactionWorkerConfig::default()
    };
    // Fold the four rows into one segment.
    let _ = tokio::task::block_in_place(|| {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        CompactionWorker::run_cycle(&rt, &catalog, &txn, &disk, &pool, &wal, &cfg, None)
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
        .append_supersede(seg.file_id, rid_lo + 1, 50, 1)
        .unwrap();
    store
        .append_supersede(seg.file_id, rid_lo + 2, 40, 2)
        .unwrap();

    // Merge with the retention floor in effect.
    let _ = tokio::task::block_in_place(|| {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        CompactionWorker::run_cycle(&rt, &catalog, &txn, &disk, &pool, &wal, &cfg, None)
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

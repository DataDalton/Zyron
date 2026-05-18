//! F1: temporal precision correctness through the columnar fold/patch/merge
//! pipeline.
//!
//! A `TIMESTAMP(9)` / `TIMESTAMPTZ(12)` column is physically a 16-byte i128
//! picosecond value, and `HLC` is a 16-byte packed instant. This test proves
//! those 16-byte values survive byte-identically through:
//!   (a) the heap -> .zyr fold (physical width derived from ts_precision),
//!   (b) a value patch + supersede in the .zyrpatch overlay,
//!   (c) the incremental merge that folds the patch into a new base segment,
//! including the `MAX_TIMESTAMP_PS` open-interval sentinel (a live versioned
//! row's sys_end) and a pre-1970 negative picosecond value (signed i128 round
//! trip through the encoder). A regression that assumes 8 bytes / derives the
//! width from TypeId alone, or that truncates ps->us, fails this test.

use std::sync::Arc;

use zyron_buffer::{BufferPool, BufferPoolConfig};
use zyron_catalog::storage::{CatalogStorage, HeapCatalogStorage};
use zyron_catalog::{Catalog, CatalogCache};
use zyron_parser::ast::{ColumnDef, DataType};
use zyron_server::background::compaction::{CompactionWorker, CompactionWorkerConfig};
use zyron_storage::columnar::{
    ColumnarPatchManager, SEGMENT_HEADER_SIZE, SYS_COL_ROWID, SYS_COL_SUPERSEDE, SYS_COL_XMIN,
    SegmentHeader, ZONE_MAP_BATCH_SIZE, ZONE_MAP_ENTRY_SIZE, ZyrFileReader,
};
use zyron_storage::encoding::{EncodingType, create_encoding};
use zyron_storage::txn::TransactionManager;
use zyron_storage::{DiskManager, DiskManagerConfig, HeapFile, HeapFileConfig, Tuple};
use zyron_versioning::MAX_TIMESTAMP_PS;
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

/// NSM-encodes one row of (id:i64, t9:i128, t12:i128, h:i128). All four
/// columns are fixed width, so the layout is: null bitmap (1 byte, 4 cols,
/// no nulls) then each value inline at its physical width.
fn encode_row(id: i64, t9: i128, t12: i128, h: i128) -> Vec<u8> {
    let mut d = Vec::with_capacity(1 + 8 + 16 * 3);
    d.push(0u8);
    d.extend_from_slice(&id.to_le_bytes());
    d.extend_from_slice(&t9.to_le_bytes());
    d.extend_from_slice(&t12.to_le_bytes());
    d.extend_from_slice(&h.to_le_bytes());
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

fn rd_i128(buf: &[u8], i: usize) -> i128 {
    i128::from_le_bytes(buf[i * 16..i * 16 + 16].try_into().unwrap())
}

fn seg_header(reader: &ZyrFileReader, column_id: u32) -> SegmentHeader {
    let raw = reader.read_segment_raw(column_id).expect("segment raw");
    let mut hdr = [0u8; SEGMENT_HEADER_SIZE];
    hdr.copy_from_slice(&raw[..SEGMENT_HEADER_SIZE]);
    SegmentHeader::from_bytes(&hdr).expect("seg header")
}

/// NSM-encodes (id:i64, ts9:i128, sysend:i128, status:text). status is
/// variable-length: u32 length prefix then bytes.
fn encode_row_mix(id: i64, ts9: i128, sysend: i128, status: &str) -> Vec<u8> {
    let mut d = Vec::new();
    d.push(0u8); // 4 cols, no nulls
    d.extend_from_slice(&id.to_le_bytes());
    d.extend_from_slice(&ts9.to_le_bytes());
    d.extend_from_slice(&sysend.to_le_bytes());
    d.extend_from_slice(&(status.len() as u32).to_le_bytes());
    d.extend_from_slice(status.as_bytes());
    d
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn temporal_ps_hlc_survives_fold_patch_merge() {
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
    // t9 = TIMESTAMP(9) ps, t12 = TIMESTAMPTZ(12) ps, h = HLC. All 16-byte
    // i128 physically; only id is 8-byte. Precision drives the fold width.
    let cols = vec![
        col("id", DataType::BigInt),
        col("t9", DataType::Timestamp(Some(9))),
        col("t12", DataType::TimestampTz(Some(12))),
        col("h", DataType::Hlc),
    ];
    let table_id = catalog
        .create_table(schema, "events", &cols, &[])
        .await
        .unwrap();
    let txn = Arc::new(TransactionManager::with_start_txn_id(Arc::clone(&wal), 100));

    let te = catalog.get_table_by_id(table_id).unwrap();
    // ts_precision must be persisted on the catalog column or the fold path
    // can't derive the 16-byte physical width.
    let t9c = te.columns.iter().find(|c| c.name == "t9").unwrap();
    assert_eq!(
        t9c.ts_precision,
        Some(9),
        "TIMESTAMP(9) precision persisted"
    );
    assert_eq!(
        te.columns
            .iter()
            .find(|c| c.name == "t12")
            .unwrap()
            .ts_precision,
        Some(12)
    );

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
    // Expected per-row values. Row 0 carries a pre-1970 negative picosecond
    // value (signed i128 round trip). Row 1 carries the open-interval
    // sentinel in t12 (a live system-versioned row's sys_end).
    let t9_of = |i: i64| -> i128 {
        if i == 0 {
            -123_456_789_012_345i128
        } else {
            // 2026-ish in ps: us value * 1_000_000.
            1_775_000_000_000_000i128 * 1_000_000 + i as i128 * 1_000_000
        }
    };
    let t12_of = |i: i64| -> i128 {
        if i == 1 {
            MAX_TIMESTAMP_PS
        } else {
            1_775_000_000_000_000i128 * 1_000_000 + i as i128 * 1_000_000_007
        }
    };
    let h_of = |i: i64| -> i128 {
        // Packed HLC: high 64 bits physical, low 64 bits logical counter.
        ((1_775_000_000_000_000i128 + i as i128) << 64) | (i as i128 & 0xFFFF)
    };

    let mut tuples = Vec::new();
    for i in 0..N {
        tuples.push(Tuple::new(encode_row(i, t9_of(i), t12_of(i), h_of(i)), 1));
    }
    heap.insert_batch(&tuples).await.unwrap();
    heap.flush().await.unwrap();

    let cfg = CompactionWorkerConfig {
        min_rows: 4,
        columnar_dir: columnar_dir.clone(),
        fsync_enabled: false,
        ..CompactionWorkerConfig::default()
    };
    let run = |c: &Catalog, t: &Arc<TransactionManager>| -> (u64, u64) {
        let (cat, tx, dk, pl, wl, cf) = (c, t, &disk, &pool, &wal, &cfg);
        tokio::task::block_in_place(|| {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            CompactionWorker::run_cycle(&rt, cat, tx, dk, pl, wl, cf, None)
        })
    };

    let (rows, segs) = run(&catalog, &txn);
    assert_eq!(rows, N as u64, "all rows folded");
    assert_eq!(segs, 1, "one segment written");

    let te = catalog.get_table_by_id(table_id).unwrap();
    let seg = te.columnar.segments[0].clone();
    let id_col = te.columns.iter().find(|c| c.name == "id").unwrap().id.0 as u32;
    let t9_col = te.columns.iter().find(|c| c.name == "t9").unwrap().id.0 as u32;
    let t12_col = te.columns.iter().find(|c| c.name == "t12").unwrap().id.0 as u32;
    let h_col = te.columns.iter().find(|c| c.name == "h").unwrap().id.0 as u32;

    // (a) Fold round trip: every 16-byte i128 value byte-identical, incl. the
    // negative ps and the MAX_TIMESTAMP_PS sentinel.
    {
        let reader = ZyrFileReader::open(std::path::Path::new(&seg.path)).unwrap();
        let rc = reader.header().row_count as usize;
        assert_eq!(rc, N as usize);
        let idd = decode_column_raw(&reader, id_col, rc, 8);
        let t9d = decode_column_raw(&reader, t9_col, rc, 16);
        let t12d = decode_column_raw(&reader, t12_col, rc, 16);
        let hd = decode_column_raw(&reader, h_col, rc, 16);
        for r in 0..rc {
            let id = i64::from_le_bytes(idd[r * 8..r * 8 + 8].try_into().unwrap());
            assert_eq!(rd_i128(&t9d, r), t9_of(id), "t9 ps folded identically");
            assert_eq!(rd_i128(&t12d, r), t12_of(id), "t12 ps folded identically");
            assert_eq!(rd_i128(&hd, r), h_of(id), "HLC folded identically");
        }
        // Explicit sentinel + negative assertions independent of row order.
        let row_of = |want: i64| {
            (0..rc)
                .find(|&r| i64::from_le_bytes(idd[r * 8..r * 8 + 8].try_into().unwrap()) == want)
                .unwrap()
        };
        assert_eq!(
            rd_i128(&t12d, row_of(1)),
            MAX_TIMESTAMP_PS,
            "open-interval ps sentinel survived fold (not truncated to us)"
        );
        assert_eq!(
            rd_i128(&t9d, row_of(0)),
            -123_456_789_012_345i128,
            "pre-1970 negative ps survived fold"
        );
    }

    // Heap hand-off: nothing left to fold.
    assert_eq!(run(&catalog, &txn).0, 0, "rows handed off out of the heap");

    // (b) Patch overlay: rewrite row id=0's t9, supersede row id=1.
    let store = ColumnarPatchManager::global(&columnar_dir)
        .store(table_id.0 as u64)
        .unwrap();
    let fid = seg.file_id;
    let rid0 = seg.sys_rowid_lo; // rowid order == insertion order here
    let new_t9: i128 = 1_775_000_000_000_000i128 * 1_000_000 + 999;
    store
        .append_value_patch(fid, rid0, t9_col, 50, 1, &new_t9.to_le_bytes())
        .unwrap();
    store.append_supersede(fid, rid0 + 1, 60, 2).unwrap();
    let o0 = store.row_overlay(fid, rid0).expect("value overlay");
    assert_eq!(
        i128::from_le_bytes(o0.patches[&t9_col][0].value[..16].try_into().unwrap()),
        new_t9,
        "16-byte i128 patch value stored width-exact"
    );

    // (c) Incremental merge folds the patch into a new base; the superseded
    // row is dropped; ps width and sentinel are preserved on the rebuild.
    run(&catalog, &txn);
    let te = catalog.get_table_by_id(table_id).unwrap();
    let merged = te.columnar.segments[0].clone();
    assert_ne!(merged.file_id, seg.file_id, "merge produced a new segment");
    assert_eq!(merged.row_count, N as u64 - 1, "superseded row dropped");

    let mreader = ZyrFileReader::open(std::path::Path::new(&merged.path)).unwrap();
    let mrc = mreader.header().row_count as usize;
    let mrowid = decode_column_raw(&mreader, SYS_COL_ROWID, mrc, 8);
    let m_id = decode_column_raw(&mreader, id_col, mrc, 8);
    let m_t9 = decode_column_raw(&mreader, t9_col, mrc, 16);
    let m_t12 = decode_column_raw(&mreader, t12_col, mrc, 16);
    let m_h = decode_column_raw(&mreader, h_col, mrc, 16);
    let mut saw_patched = false;
    for r in 0..mrc {
        let rid = u64::from_le_bytes(mrowid[r * 8..r * 8 + 8].try_into().unwrap());
        assert_ne!(rid, rid0 + 1, "superseded sys_rowid gone after merge");
        let id = i64::from_le_bytes(m_id[r * 8..r * 8 + 8].try_into().unwrap());
        if rid == rid0 {
            assert_eq!(
                rd_i128(&m_t9, r),
                new_t9,
                "i128 ps patch folded into merged base at 16-byte width"
            );
            saw_patched = true;
        } else {
            assert_eq!(rd_i128(&m_t9, r), t9_of(id), "unpatched t9 preserved");
        }
        assert_eq!(
            rd_i128(&m_t12, r),
            t12_of(id),
            "t12 ps preserved through merge"
        );
        assert_eq!(rd_i128(&m_h, r), h_of(id), "HLC preserved through merge");
    }
    assert!(saw_patched, "patched row present in merged segment");
}

/// F2: the encoder selection that runs at fold time picks a dense encoding
/// for each real folded column, not a generic fallback. Asserts the chosen
/// `encoding_type` from the on-disk segment headers and that the encoded
/// payload is materially smaller than the raw column for the cases that must
/// compact (sentinel-heavy, low-cardinality, monotone).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn encoder_selection_is_dense_on_folded_columns() {
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
    let pool = Arc::new(BufferPool::new(BufferPoolConfig { num_frames: 8192 }));
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
        col("id", DataType::BigInt),
        col("ts9", DataType::Timestamp(Some(9))),
        col("sysend", DataType::TimestampTz(Some(9))),
        col("status", DataType::Text),
    ];
    let table_id = catalog
        .create_table(schema, "obs", &cols, &[])
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

    const N: i64 = 4096;
    let base_ps = 1_775_000_000_000_000i128 * 1_000_000;
    let statuses = ["active", "inactive", "suspended"];
    let mut tuples = Vec::new();
    for i in 0..N {
        // ts9: a regular 1-second-step picosecond series (monotone).
        let ts9 = base_ps + i as i128 * 1_000_000_000_000;
        // sysend: the open-interval sentinel for every live row, a real
        // expiry on two rows (the system-versioning sys_end shape).
        let sysend = if i == 7 || i == 4000 {
            base_ps + i as i128 * 1_000_000
        } else {
            MAX_TIMESTAMP_PS
        };
        // status: 3 distinct values over 4096 rows (enum/category shape).
        let status = statuses[(i as usize) % 3];
        tuples.push(Tuple::new(encode_row_mix(i, ts9, sysend, status), 1));
    }
    heap.insert_batch(&tuples).await.unwrap();
    heap.flush().await.unwrap();

    let cfg = CompactionWorkerConfig {
        min_rows: 4,
        columnar_dir: columnar_dir.clone(),
        fsync_enabled: false,
        ..CompactionWorkerConfig::default()
    };
    let (rows, segs) = {
        let (cat, tx, dk, pl, wl, cf) = (&catalog, &txn, &disk, &pool, &wal, &cfg);
        tokio::task::block_in_place(|| {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            CompactionWorker::run_cycle(&rt, cat, tx, dk, pl, wl, cf, None)
        })
    };
    assert_eq!(rows, N as u64);
    assert_eq!(segs, 1);

    let te = catalog.get_table_by_id(table_id).unwrap();
    let seg = te.columnar.segments[0].clone();
    let cid = |name: &str| te.columns.iter().find(|c| c.name == name).unwrap().id.0 as u32;
    let reader = ZyrFileReader::open(std::path::Path::new(&seg.path)).unwrap();

    // Monotone i128 ps series: FoR+delta (or const-step) under FastLanes,
    // far below the 16 bytes/row raw width.
    let ts9h = seg_header(&reader, cid("ts9"));
    assert_eq!(
        ts9h.encoding_type,
        EncodingType::FastLanes,
        "regular ps series must select FastLanes, got {:?}",
        ts9h.encoding_type
    );
    assert!(
        ts9h.encoded_size < ts9h.raw_size / 4,
        "ps series must compact hard: {} vs {}",
        ts9h.encoded_size,
        ts9h.raw_size
    );

    // sys_end shape: one sentinel for ~every row collapses to a
    // Constant/Dictionary/RLE form, never stored verbatim.
    let seh = seg_header(&reader, cid("sysend"));
    assert!(
        matches!(
            seh.encoding_type,
            EncodingType::Constant | EncodingType::Dictionary | EncodingType::Rle
        ),
        "sentinel-heavy sys_end must compact, got {:?}",
        seh.encoding_type
    );
    assert!(seh.encoded_size < seh.raw_size / 8);

    // Low-cardinality variable-length text -> the variable-length dictionary.
    let sth = seg_header(&reader, cid("status"));
    assert_eq!(
        sth.encoding_type,
        EncodingType::Dictionary,
        "low-card text must select the variable-length dictionary, got {:?}",
        sth.encoding_type
    );
    assert!(sth.encoded_size < sth.raw_size / 4);

    // System columns: xmin is one committed value, supersede is all zero ->
    // both collapse to Constant (zero per-row cost).
    let xh = seg_header(&reader, SYS_COL_XMIN);
    assert_eq!(xh.encoding_type, EncodingType::Constant, "uniform xmin");
    let suh = seg_header(&reader, SYS_COL_SUPERSEDE);
    assert_eq!(
        suh.encoding_type,
        EncodingType::Constant,
        "all-zero supersede"
    );

    // sys_rowid is a dense monotone identity -> FastLanes constant-step.
    let rh = seg_header(&reader, SYS_COL_ROWID);
    assert_eq!(
        rh.encoding_type,
        EncodingType::FastLanes,
        "monotone sys_rowid must select FastLanes, got {:?}",
        rh.encoding_type
    );
    assert!(rh.encoded_size < rh.raw_size / 4);
}

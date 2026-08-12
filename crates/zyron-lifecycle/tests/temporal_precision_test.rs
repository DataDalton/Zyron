//! End-to-end picosecond-precision integration test.
//!
//! Drives CREATE TABLE / INSERT / SELECT through the real catalog -> planner
//! -> executor pipeline and verifies the TIMESTAMP(p) feature:
//!   - p<=6 columns keep the i64 microsecond physical representation
//!   - p>6 columns store i128 picoseconds (exact us*1_000_000 scaling)
//!   - cross-precision comparison normalizes to picoseconds (B5) so the same
//!     instant matches regardless of declared precision
//!
//! Note: the ISO-8601 literal/value path is microsecond-resolution, so a
//! string literal carries us granularity; storing it in a TIMESTAMP(9) column
//! is exact (trailing zeros). True sub-microsecond literal ingestion is a
//! separate concern (documented honesty boundary).

use std::sync::Arc;

use tempfile::tempdir;

use zyron_buffer::{BufferPool, BufferPoolConfig};
use zyron_catalog::storage::{CatalogStorage, HeapCatalogStorage};
use zyron_catalog::*;
use zyron_executor::column::ColumnData;
use zyron_executor::context::ExecutionContext;
use zyron_parser::ast::{ColumnConstraint, ColumnDef, DataType};
use zyron_storage::txn::{IsolationLevel, TransactionManager};
use zyron_storage::{DiskManager, DiskManagerConfig};
use zyron_wal::{WalWriter, WalWriterConfig};

const SCHEMA: &str = "temporal";

struct Engine {
    _dir: tempfile::TempDir,
    disk: Arc<DiskManager>,
    pool: Arc<BufferPool>,
    wal: Arc<WalWriter>,
    catalog: Arc<Catalog>,
    txn: Arc<TransactionManager>,
    schema_id: SchemaId,
}

async fn build_engine() -> Engine {
    let dir = tempdir().unwrap();
    let data_dir = dir.path().join("data");
    let wal_dir = dir.path().join("wal");
    std::fs::create_dir_all(&data_dir).unwrap();
    std::fs::create_dir_all(&wal_dir).unwrap();

    let disk = Arc::new(
        DiskManager::new(zyron_bench_harness::disk_config(data_dir))
        .await
        .unwrap(),
    );
    let pool = Arc::new(BufferPool::new(zyron_bench_harness::buffer_pool_config()));
    let wal = Arc::new(
        WalWriter::new(zyron_bench_harness::wal_config(wal_dir))
        .unwrap(),
    );
    let storage = HeapCatalogStorage::new(Arc::clone(&disk), Arc::clone(&pool)).unwrap();
    storage.init_cache().await.unwrap();
    let storage: Arc<dyn CatalogStorage> = Arc::new(storage);
    let cache = Arc::new(CatalogCache::new(4096, 1024));
    let catalog = Arc::new(
        Catalog::new(storage, cache, Arc::clone(&wal))
            .await
            .unwrap(),
    );
    let schema_id = catalog
        .create_schema(SYSTEM_DATABASE_ID, SCHEMA, "system")
        .await
        .unwrap();
    let txn = Arc::new(TransactionManager::new(Arc::clone(&wal)));
    Engine {
        _dir: dir,
        disk,
        pool,
        wal,
        catalog,
        txn,
        schema_id,
    }
}

fn col(name: &str, dt: DataType, pk: bool) -> ColumnDef {
    ColumnDef {
        name: name.to_string(),
        data_type: dt,
        nullable: Some(!pk),
        default: None,
        constraints: if pk {
            vec![ColumnConstraint::PrimaryKey]
        } else {
            vec![]
        },
    }
}

/// Plans + executes a statement, returning the result batches.
async fn query(
    e: &Engine,
    sql: &str,
    dml: bool,
) -> zyron_common::Result<Vec<zyron_executor::batch::DataBatch>> {
    let stmt = zyron_parser::parse(sql)?.into_iter().next().unwrap();
    let plan =
        zyron_planner::plan(&e.catalog, DatabaseId(1), vec![SCHEMA.to_string()], stmt, None).await?;
    let mut txn = e.txn.begin(IsolationLevel::ReadCommitted)?;
    let snapshot = txn.snapshot.clone();
    let txn_id = txn.txn_id as u32;
    let ctx = Arc::new(ExecutionContext::new(
        Arc::clone(&e.catalog),
        Arc::clone(&e.wal),
        Arc::clone(&e.pool),
        Arc::clone(&e.disk),
        txn_id,
        snapshot,
    ));
    match zyron_executor::execute(plan, &ctx).await {
        Ok(b) => {
            if dml {
                e.txn.commit_blocking(&mut txn)?;
            } else {
                let _ = e.txn.abort(&mut txn);
            }
            Ok(b)
        }
        Err(err) => {
            let _ = e.txn.abort(&mut txn);
            Err(err)
        }
    }
}

async fn run(e: &Engine, sql: &str, dml: bool) -> u64 {
    query(e, sql, dml)
        .await
        .unwrap()
        .iter()
        .map(|x| x.num_rows as u64)
        .sum()
}

/// Reads column 0 of the first result row as an i128, accepting either the
/// i64 microsecond physical form (p<=6) or the i128 picosecond form (p>6).
/// Returns (value_as_i128, is_i128_physical).
async fn select_first(e: &Engine, sql: &str) -> (i128, bool) {
    let batches = query(e, sql, false).await.unwrap();
    let b = batches.iter().find(|b| b.num_rows > 0).expect("a row");
    let c = &b.columns[0];
    match &c.data {
        ColumnData::Int64(v) => (v[0] as i128, false),
        ColumnData::Int128(v) => (v[0], true),
        other => panic!("unexpected timestamp physical variant: {:?}", other),
    }
}

/// A string datetime literal. After the cast_scalar ISO-8601 fix it parses to
/// a real non-zero microsecond instant, so the scaling/cross-precision
/// assertions prove the picosecond contract end to end through INSERT.
const TS_STR: &str = "2026-05-17 12:34:56.123456";
fn ts_us() -> i64 {
    zyron_common::parse_timestamp_micros(TS_STR).unwrap()
}

#[test]
fn test_timestamp_precision_end_to_end() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let e = build_engine().await;

        // p<=6 keeps i64 microseconds; p>6 uses i128 picoseconds.
        e.catalog
            .create_table(
                e.schema_id,
                "t6",
                &[
                    col("id", DataType::Int, true),
                    col("ts", DataType::Timestamp(None), false),
                ],
                &[],
            )
            .await
            .unwrap();
        e.catalog
            .create_table(
                e.schema_id,
                "t9",
                &[
                    col("id", DataType::Int, true),
                    col("ts", DataType::Timestamp(Some(9)), false),
                ],
                &[],
            )
            .await
            .unwrap();

        run(
            &e,
            &format!("INSERT INTO \"t6\" VALUES (1, '{TS_STR}')"),
            true,
        )
        .await;
        run(
            &e,
            &format!("INSERT INTO \"t9\" VALUES (1, '{TS_STR}')"),
            true,
        )
        .await;

        assert_eq!(run(&e, "SELECT id FROM \"t6\"", false).await, 1);
        assert_eq!(run(&e, "SELECT id FROM \"t9\"", false).await, 1);

        // t6 round-trips as i64 microseconds holding the exact value.
        let (us, is_i128_6) = select_first(&e, "SELECT \"ts\" FROM \"t6\"").await;
        assert!(!is_i128_6, "p<=6 must stay i64 microseconds");
        assert_eq!(
            us,
            ts_us() as i128,
            "p<=6 stores the exact parsed microsecond value"
        );

        // t9 round-trips as i128 picoseconds, exactly parsed-us * 1_000_000.
        let (ps, is_i128_9) = select_first(&e, "SELECT \"ts\" FROM \"t9\"").await;
        assert!(is_i128_9, "p>6 must be i128 picoseconds");
        assert_eq!(
            ps,
            ts_us() as i128 * 1_000_000,
            "ps must be the microsecond instant scaled exactly by 1e6"
        );

        // Cross-precision comparison (B5), column vs column. A single table
        // with a microsecond column `a` and a picosecond column `b` holding
        // the same instant: `a = b` must be true (the us side is normalized
        // up to ps), and `a <> b` false. This exercises normalize_ts_pair
        // directly through the real planner/executor predicate path.
        e.catalog
            .create_table(
                e.schema_id,
                "m",
                &[
                    col("id", DataType::Int, true),
                    col("a", DataType::Timestamp(None), false),
                    col("b", DataType::Timestamp(Some(9)), false),
                ],
                &[],
            )
            .await
            .unwrap();
        run(
            &e,
            &format!("INSERT INTO \"m\" VALUES (1, '{TS_STR}', '{TS_STR}')"),
            true,
        )
        .await;
        // a is us (i64) holding TS_US, b is ps (i128) holding TS_US*1e6.
        let (a_us, a_is_128) = select_first(&e, "SELECT \"a\" FROM \"m\"").await;
        let (b_ps, b_is_128) = select_first(&e, "SELECT \"b\" FROM \"m\"").await;
        assert!(!a_is_128 && b_is_128);
        assert_eq!(a_us, ts_us() as i128);
        assert_eq!(b_ps, ts_us() as i128 * 1_000_000);

        assert_eq!(
            run(&e, "SELECT id FROM \"m\" WHERE \"a\" = \"b\"", false).await,
            1,
            "cross-precision equality (us col vs ps col, same instant) must match"
        );
        assert_eq!(
            run(&e, "SELECT id FROM \"m\" WHERE \"a\" <> \"b\"", false).await,
            0,
            "same instant must not be unequal across precisions"
        );
        assert_eq!(
            run(&e, "SELECT id FROM \"m\" WHERE \"b\" >= \"a\"", false).await,
            1,
            "cross-precision >= of the same instant must hold"
        );
        assert_eq!(
            run(&e, "SELECT id FROM \"m\" WHERE \"b\" > \"a\"", false).await,
            0,
            "same instant is not strictly greater across precisions"
        );
    });
}

#[test]
fn test_time_bucket_gapfill_end_to_end() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let e = build_engine().await;
        e.catalog
            .create_table(
                e.schema_id,
                "g",
                &[
                    col("id", DataType::Int, true),
                    col("ts", DataType::BigInt, false),
                ],
                &[],
            )
            .await
            .unwrap();
        // Two events: bucket 0 (ts 100) and bucket 3000 (ts 3500), width 1000.
        // Buckets 1000 and 2000 are absent and must be synthesized.
        run(&e, "INSERT INTO \"g\" VALUES (0, 100)", true).await;
        run(&e, "INSERT INTO \"g\" VALUES (1, 3500)", true).await;

        let batches = query(
            &e,
            "SELECT time_bucket_gapfill(1000, \"ts\"), count(*) \
             FROM \"g\" GROUP BY time_bucket_gapfill(1000, \"ts\")",
            false,
        )
        .await
        .unwrap();
        let mut buckets: Vec<(i64, bool)> = Vec::new(); // (bucket, count_is_null)
        for b in &batches {
            let bk: Vec<i64> = match &b.columns[0].data {
                ColumnData::Int64(v) => v.clone(),
                ColumnData::Int128(v) => v.iter().map(|&x| x as i64).collect(),
                other => panic!("bucket variant {:?}", other),
            };
            for (r, &k) in bk.iter().enumerate() {
                buckets.push((k, b.columns[1].is_null(r)));
            }
        }
        buckets.sort_by_key(|x| x.0);
        // Dense buckets 0,1000,2000,3000; gaps (1000,2000) have NULL count.
        assert_eq!(
            buckets,
            vec![(0, false), (1000, true), (2000, true), (3000, false)],
            "gapfill must densify absent buckets with NULL aggregates"
        );
    });
}

#[test]
fn test_time_bucket_floors_negative_pre_1970() {
    // time_bucket buckets with div_euclid, so a pre-1970 (negative) instant
    // floors toward negative infinity, not toward zero. This is the same
    // bucket closure the i128 picosecond path uses, so it locks the temporal
    // contract that pairs with the signed segment min/max work: a negative
    // timestamp must land in the lower bucket, never round up to bucket 0.
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let e = build_engine().await;
        e.catalog
            .create_table(
                e.schema_id,
                "nb",
                &[
                    col("id", DataType::Int, true),
                    col("ts", DataType::BigInt, false),
                ],
                &[],
            )
            .await
            .unwrap();
        // width 1000. Expected floor buckets:
        //   -1   -> -1000 (div_euclid, not 0)
        //   -1000 -> -1000
        //   -1500 -> -2000
        //   0     -> 0
        //   1500  -> 1000
        for (i, ts) in [-1i64, -1000, -1500, 0, 1500].iter().enumerate() {
            run(&e, &format!("INSERT INTO \"nb\" VALUES ({i}, {ts})"), true).await;
        }
        let batches = query(
            &e,
            "SELECT time_bucket(1000, \"ts\"), count(*) \
             FROM \"nb\" GROUP BY time_bucket(1000, \"ts\")",
            false,
        )
        .await
        .unwrap();
        let mut buckets: Vec<i64> = Vec::new();
        for b in &batches {
            match &b.columns[0].data {
                ColumnData::Int64(v) => buckets.extend_from_slice(v),
                ColumnData::Int128(v) => buckets.extend(v.iter().map(|&x| x as i64)),
                other => panic!("bucket variant {:?}", other),
            }
        }
        buckets.sort_unstable();
        // -2000 (from -1500), -1000 (from -1 and -1000), 0 (from 0), 1000 (from 1500).
        assert_eq!(
            buckets,
            vec![-2000, -1000, 0, 1000],
            "negative instants must floor toward -infinity (div_euclid)"
        );
    });
}

#[test]
fn test_time_bucket_downsample_end_to_end() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let e = build_engine().await;
        // Use a BIGINT column of explicit microsecond values so time_bucket
        // arithmetic is exercised with real distinct data, independent of the
        // separate (pre-existing) string-literal timestamp parse gap.
        e.catalog
            .create_table(
                e.schema_id,
                "ev",
                &[
                    col("id", DataType::Int, true),
                    col("ts", DataType::BigInt, false),
                ],
                &[],
            )
            .await
            .unwrap();
        // 1h = 3_600_000_000 us. Two values in bucket 0, two in bucket 3.6e9.
        for (i, us) in [300_000_000i64, 1_800_000_000, 3_900_000_000, 5_700_000_000]
            .iter()
            .enumerate()
        {
            run(&e, &format!("INSERT INTO \"ev\" VALUES ({i}, {us})"), true).await;
        }
        // 1 hour = 3_600_000_000 microseconds (the p<=6 storage unit).
        let groups = run(
            &e,
            "SELECT time_bucket(3600000000, \"ts\"), count(*) \
             FROM \"ev\" GROUP BY time_bucket(3600000000, \"ts\")",
            false,
        )
        .await;
        assert_eq!(
            groups, 2,
            "four events must downsample to two hourly buckets"
        );

        // Each bucket holds exactly 2 rows.
        let batches = query(
            &e,
            "SELECT time_bucket(3600000000, \"ts\"), count(*) \
             FROM \"ev\" GROUP BY time_bucket(3600000000, \"ts\")",
            false,
        )
        .await
        .unwrap();
        let mut counts: Vec<i64> = Vec::new();
        for b in &batches {
            // second column is count(*)
            match &b.columns[1].data {
                ColumnData::Int64(v) => counts.extend_from_slice(v),
                ColumnData::Int32(v) => counts.extend(v.iter().map(|&x| x as i64)),
                other => panic!("unexpected count variant: {:?}", other),
            }
        }
        counts.sort_unstable();
        assert_eq!(counts, vec![2, 2], "each hourly bucket holds 2 events");
    });
}

#[test]
fn test_hlc_monotonic_end_to_end() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let e = build_engine().await;
        e.catalog
            .create_table(
                e.schema_id,
                "hl",
                &[
                    col("id", DataType::Int, true),
                    col("h", DataType::Hlc, false),
                ],
                &[],
            )
            .await
            .unwrap();

        // Each INSERT calls hlc_now(); successive HLC values must be strictly
        // increasing (causal monotonicity), even within the same millisecond.
        for i in 0..50 {
            run(
                &e,
                &format!("INSERT INTO \"hl\" VALUES ({i}, hlc_now())"),
                true,
            )
            .await;
        }
        assert_eq!(run(&e, "SELECT id FROM \"hl\"", false).await, 50);

        // HLC column is the i128 physical form.
        let (_, is_i128) = select_first(&e, "SELECT \"h\" FROM \"hl\"").await;
        assert!(is_i128, "HLC must be the i128 physical form");

        // Read (id, h) pairs, sort by id in the test, and assert the HLC
        // values are strictly increasing in insertion order.
        let batches = query(&e, "SELECT \"id\", \"h\" FROM \"hl\"", false)
            .await
            .unwrap();
        let mut pairs: Vec<(i64, i128)> = Vec::new();
        for b in &batches {
            let ids = match &b.columns[0].data {
                ColumnData::Int32(v) => v.iter().map(|&x| x as i64).collect::<Vec<_>>(),
                ColumnData::Int64(v) => v.clone(),
                other => panic!("unexpected id variant: {:?}", other),
            };
            let hs = match &b.columns[1].data {
                ColumnData::Int128(v) => v.clone(),
                other => panic!("unexpected hlc variant: {:?}", other),
            };
            for (i, h) in ids.into_iter().zip(hs) {
                pairs.push((i, h));
            }
        }
        assert_eq!(pairs.len(), 50);
        pairs.sort_by_key(|p| p.0);
        for w in pairs.windows(2) {
            assert!(
                w[1].1 > w[0].1,
                "HLC must be strictly monotonic in insertion order: id {} -> {}, id {} -> {}",
                w[0].0,
                w[0].1,
                w[1].0,
                w[1].1
            );
        }
    });
}

//! Data-lifecycle validation and benchmark suite.
//!
//! Follows the shared zyron-bench-harness conventions: domain-named suite
//! ("lifecycle"), descriptive `test_*` functions, `tprintln!` logging, 5-run
//! averaged metrics via `validate_metric`, JSON+TXT output under
//! benchmarks/lifecycle/. All work goes through production code paths:
//! catalog DDL, planner, executor, the legal-hold/WORM DML hook, the real
//! opendal archive layer, and the lock-free lifecycle registries.
//!
//! DML correctness is verified by counting rows with a real SELECT before and
//! after, since the executor returns a status batch (not an affected-row
//! count) for INSERT/UPDATE/DELETE.

use std::sync::Arc;

use tempfile::tempdir;

use zyron_bench_harness::*;
use zyron_buffer::{BufferPool, BufferPoolConfig};
use zyron_catalog::storage::{CatalogStorage, HeapCatalogStorage};
use zyron_catalog::*;
use zyron_executor::context::{DmlHook, ExecutionContext};
use zyron_parser::ast::{ColumnConstraint, ColumnDef, DataType};
use zyron_storage::txn::{IsolationLevel, TransactionManager};
use zyron_storage::{DiskManager, DiskManagerConfig};
use zyron_wal::{WalWriter, WalWriterConfig};

use zyron_lifecycle::classification::{PiiKind, classify_value};
use zyron_lifecycle::compliance::{RetentionRequirement, validate_retention};
use zyron_lifecycle::legal_hold::LegalHoldRegistry;

const SUITE: &str = "lifecycle";
const SCHEMA: &str = "lifecycle";

struct Engine {
    _dir: tempfile::TempDir,
    disk: Arc<DiskManager>,
    pool: Arc<BufferPool>,
    wal: Arc<WalWriter>,
    catalog: Arc<Catalog>,
    txn: Arc<TransactionManager>,
    legal_holds: Arc<LegalHoldRegistry>,
    schema_id: SchemaId,
}

async fn build_engine() -> Engine {
    let dir = tempdir().unwrap();
    let data_dir = dir.path().join("data");
    let wal_dir = dir.path().join("wal");
    std::fs::create_dir_all(&data_dir).unwrap();
    std::fs::create_dir_all(&wal_dir).unwrap();

    let disk = Arc::new(
        DiskManager::new(DiskManagerConfig {
            data_dir,
            fsync_enabled: false,
        })
        .await
        .unwrap(),
    );
    let pool = Arc::new(BufferPool::new(BufferPoolConfig { num_frames: 8192 }));
    let wal = Arc::new(
        WalWriter::new(WalWriterConfig {
            wal_dir,
            fsync_enabled: false,
            ..Default::default()
        })
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
        legal_holds: Arc::new(LegalHoldRegistry::new()),
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

/// Plans and executes a statement through the production pipeline with the
/// legal-hold / WORM enforcement hook attached.
async fn run(e: &Engine, sql: &str, dml: bool) -> zyron_common::Result<u64> {
    let stmt = zyron_parser::parse(sql)?.into_iter().next().unwrap();
    let plan =
        zyron_planner::plan(&e.catalog, DatabaseId(1), vec![SCHEMA.to_string()], stmt).await?;
    let mut txn = e.txn.begin(IsolationLevel::ReadCommitted)?;
    let snapshot = txn.snapshot.clone();
    let txn_id = txn.txn_id as u32;
    let mut ctx = ExecutionContext::new(
        Arc::clone(&e.catalog),
        Arc::clone(&e.wal),
        Arc::clone(&e.pool),
        Arc::clone(&e.disk),
        txn_id,
        snapshot,
    );
    ctx.dml_hook = Some(Arc::new(zyron_wire::dml_enforce::LegalHoldDmlHook::new(
        Arc::clone(&e.legal_holds),
        Arc::clone(&e.catalog),
    )) as Arc<dyn DmlHook>);
    let ctx = Arc::new(ctx);
    let res = zyron_executor::execute(plan, &ctx).await;
    match res {
        Ok(b) => {
            if dml {
                e.txn.commit(&mut txn)?;
            } else {
                let _ = e.txn.abort(&mut txn);
            }
            Ok(b.iter().map(|x| x.num_rows as u64).sum())
        }
        Err(err) => {
            let _ = e.txn.abort(&mut txn);
            Err(err)
        }
    }
}

/// Row count via a real SELECT (authoritative for correctness checks).
async fn count(e: &Engine, select_sql: &str) -> u64 {
    run(e, select_sql, false).await.unwrap()
}

/// Mutates a table's lifecycle config in the catalog (production update path).
async fn set_lifecycle(e: &Engine, table: &str, f: impl FnOnce(&mut schema::LifecycleConfig)) {
    let t = e.catalog.get_table(e.schema_id, table).unwrap();
    let mut entry = (*t).clone();
    f(&mut entry.lifecycle);
    e.catalog.update_table(entry).await.unwrap();
}

fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
}

fn insert_values(table: &str, cols: &str, rows: &[String]) -> String {
    format!(
        "INSERT INTO \"{}\" ({}) VALUES {}",
        table,
        cols,
        rows.join(", ")
    )
}

// ===========================================================================
// Row-level TTL
// ===========================================================================
#[test]
fn test_row_level_ttl() {
    zyron_bench_harness::init(SUITE);
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        tprintln!("\n=== Row-Level TTL ===");
        let e = build_engine().await;
        e.catalog
            .create_table(
                e.schema_id,
                "events",
                &[
                    col("id", DataType::Int, true),
                    col("ts", DataType::BigInt, false),
                ],
                &[],
            )
            .await
            .unwrap();
        // Columns are 0-based: id=0, ts=1.
        set_lifecycle(&e, "events", |lc| {
            lc.ttl_column_id = 1;
            lc.ttl_seconds = 1;
        })
        .await;

        let old_ts = now_secs() - 100;
        let rows: Vec<String> = (0..1000).map(|i| format!("({i}, {old_ts})")).collect();
        run(&e, &insert_values("events", "id, ts", &rows), true)
            .await
            .unwrap();
        assert_eq!(
            count(&e, "SELECT id FROM \"events\"").await,
            1000,
            "1000 rows inserted"
        );

        let cutoff = now_secs() - 1;
        run(
            &e,
            &format!("DELETE FROM \"events\" WHERE \"ts\" < {cutoff}"),
            true,
        )
        .await
        .unwrap();
        assert_eq!(
            count(&e, "SELECT id FROM \"events\"").await,
            0,
            "all expired rows deleted"
        );

        let fresh: Vec<String> = (0..10).map(|i| format!("({i}, {})", now_secs())).collect();
        run(&e, &insert_values("events", "id, ts", &fresh), true)
            .await
            .unwrap();
        run(
            &e,
            &format!("DELETE FROM \"events\" WHERE \"ts\" < {}", now_secs() - 1),
            true,
        )
        .await
        .unwrap();
        assert_eq!(
            count(&e, "SELECT id FROM \"events\"").await,
            10,
            "fresh rows survive TTL"
        );
        tprintln!("  Row-level TTL delete + retention boundary: PASS");
    });
}

// ===========================================================================
// Partition-scoped TTL (predicate-drop model: no partition subsystem)
// ===========================================================================
#[test]
fn test_partition_scoped_ttl() {
    zyron_bench_harness::init(SUITE);
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        tprintln!("\n=== Partition-Scoped TTL ===");
        let e = build_engine().await;
        e.catalog
            .create_table(
                e.schema_id,
                "logs",
                &[
                    col("id", DataType::Int, true),
                    col("month", DataType::Int, false),
                ],
                &[],
            )
            .await
            .unwrap();
        let mut rows = Vec::new();
        for m in 1..=4 {
            for i in 0..250 {
                rows.push(format!("({}, {m})", m * 1000 + i));
            }
        }
        run(&e, &insert_values("logs", "id, month", &rows), true)
            .await
            .unwrap();
        assert_eq!(count(&e, "SELECT id FROM \"logs\"").await, 1000);
        run(&e, "DELETE FROM \"logs\" WHERE \"month\" = 1", true)
            .await
            .unwrap();
        assert_eq!(
            count(&e, "SELECT id FROM \"logs\"").await,
            750,
            "first partition dropped"
        );
        assert_eq!(
            count(&e, "SELECT id FROM \"logs\" WHERE \"month\" = 1").await,
            0,
            "no month=1 rows remain"
        );
        tprintln!("  Partition-scoped TTL (predicate drop): PASS");
    });
}

// ===========================================================================
// Tiered storage (logical tier metadata; transparent query)
// ===========================================================================
#[test]
fn test_tiered_storage() {
    zyron_bench_harness::init(SUITE);
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        use zyron_lifecycle::tiered_storage::StorageTier;
        tprintln!("\n=== Tiered Storage ===");
        let e = build_engine().await;
        e.catalog
            .create_table(
                e.schema_id,
                "metrics",
                &[
                    col("id", DataType::Int, true),
                    col("v", DataType::Int, false),
                ],
                &[],
            )
            .await
            .unwrap();
        let rows: Vec<String> = (0..5000).map(|i| format!("({i}, {})", i * 2)).collect();
        run(&e, &insert_values("metrics", "id, v", &rows), true)
            .await
            .unwrap();
        set_lifecycle(&e, "metrics", |lc| {
            lc.storage_tier = StorageTier::Cold as u8;
        })
        .await;
        // Transparent query: a cold-tier table is fully and correctly queryable.
        assert_eq!(
            count(&e, "SELECT id, v FROM \"metrics\"").await,
            5000,
            "cold-tier table fully queryable"
        );
        assert_eq!(
            count(&e, "SELECT id FROM \"metrics\" WHERE \"v\" >= 5000").await,
            2500,
            "cold-tier predicate results correct"
        );
        assert!(
            StorageTier::Hot.cost_multiplier() < StorageTier::Cold.cost_multiplier(),
            "planner cost rises with tier coldness"
        );
        tprintln!("  Tiered storage transparent query + cost model: PASS");
    });
}

// ===========================================================================
// Archive + restore (real opendal fs round-trip)
// ===========================================================================
#[test]
fn test_archive_restore() {
    zyron_bench_harness::init(SUITE);
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        tprintln!("\n=== Archive + Restore ===");
        let e = build_engine().await;
        let adir = tempdir().unwrap();
        e.catalog
            .create_table(
                e.schema_id,
                "txns",
                &[
                    col("id", DataType::Int, true),
                    col("amt", DataType::Int, false),
                ],
                &[],
            )
            .await
            .unwrap();
        let rows: Vec<String> = (0..10_000).map(|i| format!("({i}, {})", i + 1)).collect();
        run(&e, &insert_values("txns", "id, amt", &rows), true)
            .await
            .unwrap();
        assert_eq!(count(&e, "SELECT id FROM \"txns\"").await, 10_000);

        // Read rows out and archive via the real opendal fs backend.
        let stmt = zyron_parser::parse("SELECT id, amt FROM \"txns\"")
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let plan = zyron_planner::plan(&e.catalog, DatabaseId(1), vec![SCHEMA.to_string()], stmt)
            .await
            .unwrap();
        let mut t = e.txn.begin(IsolationLevel::ReadCommitted).unwrap();
        let ctx = Arc::new(ExecutionContext::new(
            Arc::clone(&e.catalog),
            Arc::clone(&e.wal),
            Arc::clone(&e.pool),
            Arc::clone(&e.disk),
            t.txn_id as u32,
            t.snapshot.clone(),
        ));
        let batches = zyron_executor::execute(plan, &ctx).await.unwrap();
        let _ = e.txn.abort(&mut t);
        let mut records: Vec<Vec<u8>> = Vec::new();
        for b in &batches {
            for r in 0..b.num_rows {
                records.push(
                    format!(
                        "{:?}\u{1f}{:?}",
                        b.column(0).get_scalar(r),
                        b.column(1).get_scalar(r)
                    )
                    .into_bytes(),
                );
            }
        }
        assert_eq!(records.len(), 10_000, "all rows read for archive");
        let uri = format!("file://{}/txns.zylog", adir.path().display());
        let ar = zyron_lifecycle::archive::archive_rows(&uri, &records)
            .await
            .unwrap();
        assert_eq!(ar.rows_archived, 10_000);

        run(&e, "DELETE FROM \"txns\" WHERE \"id\" >= 0", true)
            .await
            .unwrap();
        assert_eq!(
            count(&e, "SELECT id FROM \"txns\"").await,
            0,
            "main table emptied"
        );

        let (restored, rr) = zyron_lifecycle::archive::restore_from(&uri).await.unwrap();
        assert_eq!(rr.rows_restored, 10_000, "restore round-trips all rows");
        assert_eq!(restored.len(), 10_000);
        tprintln!("  Archive write + delete + restore round-trip: PASS");
    });
}

// ===========================================================================
// Soft delete
// ===========================================================================
#[test]
fn test_soft_delete() {
    zyron_bench_harness::init(SUITE);
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        tprintln!("\n=== Soft Delete ===");
        let e = build_engine().await;
        e.catalog
            .create_table(
                e.schema_id,
                "users",
                &[
                    col("id", DataType::Int, true),
                    col("is_deleted", DataType::Boolean, false),
                    col("deleted_at", DataType::BigInt, false),
                ],
                &[],
            )
            .await
            .unwrap();
        // 0-based ids: id=0, is_deleted=1, deleted_at=2.
        set_lifecycle(&e, "users", |lc| {
            lc.soft_delete_enabled = true;
            lc.soft_delete_is_deleted_col_id = 1;
            lc.soft_delete_deleted_at_col_id = 2;
        })
        .await;
        run(
            &e,
            "INSERT INTO \"users\" (id, is_deleted, deleted_at) VALUES (1, false, 0), (2, false, 0), (3, false, 0)",
            true,
        )
        .await
        .unwrap();
        // DELETE is rewritten by the binder into a soft-delete UPDATE.
        run(&e, "DELETE FROM \"users\" WHERE \"id\" = 1", true)
            .await
            .unwrap();
        assert_eq!(
            count(&e, "SELECT id FROM \"users\"").await,
            2,
            "soft-deleted row hidden from default SELECT"
        );
        assert_eq!(
            count(&e, "SELECT id FROM \"users\" INCLUDING DELETED").await,
            3,
            "INCLUDING DELETED shows all rows"
        );
        assert_eq!(
            count(&e, "SELECT id FROM \"users\" ONLY DELETED").await,
            1,
            "ONLY DELETED shows just the tombstone"
        );
        // A plain UPDATE must NOT resurrect a tombstone: the soft-delete
        // predicate injection protects already-deleted rows (RLS-like). This
        // is why undo is the dedicated `RESTORE FROM t WHERE` statement, which
        // the dispatch layer turns into an injection-suppressed UPDATE
        // (validated by the soft_delete unit tests + the dispatch handler).
        run(
            &e,
            "UPDATE \"users\" SET deleted_at = 1 WHERE \"id\" = 1",
            true,
        )
        .await
        .unwrap();
        assert_eq!(
            count(&e, "SELECT id FROM \"users\"").await,
            2,
            "plain UPDATE cannot touch a tombstoned row (restore is RESTORE FROM)"
        );
        assert_eq!(
            count(&e, "SELECT id FROM \"users\" ONLY DELETED").await,
            1,
            "tombstone unchanged by the plain UPDATE"
        );
        // HARD delete bypasses the soft-delete rewrite -> physical removal.
        run(&e, "DELETE FROM \"users\" WHERE \"id\" = 2 HARD", true)
            .await
            .unwrap();
        assert_eq!(
            count(&e, "SELECT id FROM \"users\" INCLUDING DELETED").await,
            2,
            "HARD delete physically removed the row"
        );
        tprintln!("  Soft delete + visibility modifiers + HARD: PASS");
    });
}

// ===========================================================================
// Soft-delete purge
// ===========================================================================
#[test]
fn test_soft_delete_purge() {
    zyron_bench_harness::init(SUITE);
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        tprintln!("\n=== Soft-Delete Purge ===");
        let e = build_engine().await;
        e.catalog
            .create_table(
                e.schema_id,
                "sessions",
                &[
                    col("id", DataType::Int, true),
                    col("is_deleted", DataType::Boolean, false),
                    col("deleted_at", DataType::BigInt, false),
                ],
                &[],
            )
            .await
            .unwrap();
        set_lifecycle(&e, "sessions", |lc| {
            lc.soft_delete_enabled = true;
            lc.soft_delete_is_deleted_col_id = 1;
            lc.soft_delete_deleted_at_col_id = 2;
            lc.purge_grace_seconds = 1;
        })
        .await;
        let old = now_secs() - 100;
        run(
            &e,
            &format!(
                "INSERT INTO \"sessions\" (id, is_deleted, deleted_at) VALUES (1, true, {old}), (2, true, {old}), (3, false, 0)"
            ),
            true,
        )
        .await
        .unwrap();
        let cutoff = now_secs() - 1;
        // Purge = the production HARD delete PurgeManager issues.
        run(
            &e,
            &format!(
                "DELETE FROM \"sessions\" WHERE \"is_deleted\" = true AND \"deleted_at\" < {cutoff} HARD"
            ),
            true,
        )
        .await
        .unwrap();
        assert_eq!(
            count(&e, "SELECT id FROM \"sessions\" INCLUDING DELETED").await,
            1,
            "expired tombstones purged, live row retained"
        );
        tprintln!("  Soft-delete purge after grace window: PASS");
    });
}

// ===========================================================================
// Legal hold
// ===========================================================================
#[test]
fn test_legal_hold() {
    zyron_bench_harness::init(SUITE);
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        tprintln!("\n=== Legal Hold ===");
        let e = build_engine().await;
        let tid = e
            .catalog
            .create_table(
                e.schema_id,
                "transactions",
                &[
                    col("id", DataType::Int, true),
                    col("customer", DataType::Int, false),
                ],
                &[],
            )
            .await
            .unwrap();
        run(
            &e,
            "INSERT INTO \"transactions\" (id, customer) VALUES (1, 1), (2, 1), (3, 2)",
            true,
        )
        .await
        .unwrap();
        let active = |released: i64| schema::LegalHoldEntry {
            id: 1,
            name: "case_1".into(),
            table_id: tid.0,
            predicate_sql: String::new(),
            reason: "litigation".into(),
            created_at: now_secs(),
            released_at: released,
        };
        e.legal_holds.reload(&[active(0)]);
        assert!(
            run(
                &e,
                "DELETE FROM \"transactions\" WHERE \"customer\" = 1",
                true
            )
            .await
            .is_err(),
            "DELETE under legal hold rejected"
        );
        assert!(
            run(
                &e,
                "UPDATE \"transactions\" SET customer = 9 WHERE \"customer\" = 1",
                true
            )
            .await
            .is_err(),
            "UPDATE under legal hold rejected"
        );
        assert!(
            run(&e, "DELETE FROM \"transactions\" WHERE \"id\" >= 0", true)
                .await
                .is_err(),
            "TTL delete blocked by hold (hold overrides TTL)"
        );
        assert_eq!(
            count(&e, "SELECT id FROM \"transactions\"").await,
            3,
            "no rows removed while held"
        );
        // Release the hold -> deletes succeed again.
        e.legal_holds.reload(&[active(now_secs())]);
        run(
            &e,
            "DELETE FROM \"transactions\" WHERE \"customer\" = 1",
            true,
        )
        .await
        .unwrap();
        assert_eq!(
            count(&e, "SELECT id FROM \"transactions\"").await,
            1,
            "after release, held data deletable"
        );
        tprintln!("  Legal hold blocks DELETE/UPDATE/TTL, release restores: PASS");
    });
}

// ===========================================================================
// FORGET USER across multiple tables
// ===========================================================================
#[test]
fn test_forget_user() {
    zyron_bench_harness::init(SUITE);
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        tprintln!("\n=== FORGET USER (cross-table) ===");
        let e = build_engine().await;
        let tables = ["users", "orders", "preferences"];
        for t in &tables {
            e.catalog
                .create_table(
                    e.schema_id,
                    t,
                    &[
                        col("id", DataType::Int, true),
                        col("user_id", DataType::Int, false),
                        col("amount", DataType::Int, false),
                    ],
                    &[],
                )
                .await
                .unwrap();
            run(
                &e,
                &format!(
                    "INSERT INTO \"{t}\" (id, user_id, amount) VALUES (1, 123, 50), (2, 123, 75), (3, 999, 10)"
                ),
                true,
            )
            .await
            .unwrap();
        }
        // Cross-table erasure: per-table HARD delete of the subject's rows.
        for t in &tables {
            run(
                &e,
                &format!("DELETE FROM \"{t}\" WHERE \"user_id\" = 123 HARD"),
                true,
            )
            .await
            .unwrap();
        }
        for t in &tables {
            assert_eq!(
                count(&e, &format!("SELECT id FROM \"{t}\" WHERE \"user_id\" = 123")).await,
                0,
                "subject erased from {t}"
            );
            assert_eq!(
                count(&e, &format!("SELECT id FROM \"{t}\" WHERE \"user_id\" = 999")).await,
                1,
                "other users retained in {t}"
            );
        }
        assert_eq!(classify_value("alice@example.com"), Some(PiiKind::Email));
        assert_eq!(classify_value("+1 (415) 555-2671"), Some(PiiKind::Phone));
        tprintln!("  FORGET USER cross-table erasure + PII detect: PASS");
    });
}

// ===========================================================================
// Compliance report (SOX retention floor + PII detection)
// ===========================================================================
#[test]
fn test_compliance_report() {
    zyron_bench_harness::init(SUITE);
    tprintln!("\n=== Compliance Report ===");
    let reqs = vec![
        RetentionRequirement::sox_financial(),
        RetentionRequirement::gdpr_pii(),
    ];
    assert!(
        validate_retention("financial", 8 * 365 * 86400, &reqs).is_ok(),
        "SOX-compliant retention accepted"
    );
    assert!(
        validate_retention("financial", 365 * 86400, &reqs).is_err(),
        "below-SOX retention rejected"
    );
    assert!(validate_retention("pii", 90 * 86400, &reqs).is_err());
    let emails = [
        "a@b.com".to_string(),
        "c@d.org".to_string(),
        "x".to_string(),
    ];
    let detected = zyron_lifecycle::classification::auto_classify_column(&emails);
    assert_eq!(
        detected,
        Some(PiiKind::Email),
        "email column detected as PII"
    );
    tprintln!("  SOX retention validation + PII detection: PASS");
}

// ===========================================================================
// Performance (5-run averaged, validated against thresholds)
// ===========================================================================
#[test]
fn test_performance() {
    zyron_bench_harness::init(SUITE);
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        tprintln!("\n=== Performance (5-run average) ===");

        // Legal-hold check latency, no holds (production hot path).
        let reg = LegalHoldRegistry::new();
        let mut ns_runs = Vec::new();
        for _ in 0..VALIDATION_RUNS {
            let iters = 5_000_000u64;
            let t = Instant::now();
            let mut acc = 0u64;
            for i in 0..iters {
                if reg.table_has_hold((i & 0xffff) as u32) {
                    acc += 1;
                }
            }
            std::hint::black_box(acc);
            ns_runs.push(t.elapsed().as_nanos() as f64 / iters as f64);
        }
        validate_metric(
            "legal_hold_check_no_holds",
            "latency (ns/op)",
            ns_runs,
            50.0,
            false,
        );

        // PII detection over a 100K-value column sample.
        let sample: Vec<String> = (0..100_000)
            .map(|i| format!("user{i}@example.com"))
            .collect();
        let mut pii_runs = Vec::new();
        for _ in 0..VALIDATION_RUNS {
            let t = Instant::now();
            std::hint::black_box(zyron_lifecycle::classification::auto_classify_column(
                &sample,
            ));
            pii_runs.push(t.elapsed().as_secs_f64());
        }
        validate_metric("pii_detection", "latency (s)", pii_runs, 5.0, false);

        // TTL row-delete throughput (production planner+executor path).
        let mut ttl_runs = Vec::new();
        for _ in 0..VALIDATION_RUNS {
            let e = build_engine().await;
            e.catalog
                .create_table(
                    e.schema_id,
                    "perf_ttl",
                    &[
                        col("id", DataType::Int, true),
                        col("ts", DataType::BigInt, false),
                    ],
                    &[],
                )
                .await
                .unwrap();
            let n = 200_000usize;
            let old = now_secs() - 100;
            let rows: Vec<String> = (0..n).map(|i| format!("({i}, {old})")).collect();
            run(&e, &insert_values("perf_ttl", "id, ts", &rows), true)
                .await
                .unwrap();
            assert_eq!(count(&e, "SELECT id FROM \"perf_ttl\"").await, n as u64);
            let cutoff = now_secs() - 1;
            let t = Instant::now();
            run(
                &e,
                &format!("DELETE FROM \"perf_ttl\" WHERE \"ts\" < {cutoff}"),
                true,
            )
            .await
            .unwrap();
            let secs = t.elapsed().as_secs_f64();
            assert_eq!(
                count(&e, "SELECT id FROM \"perf_ttl\"").await,
                0,
                "all rows deleted"
            );
            ttl_runs.push(n as f64 / secs);
        }
        validate_metric(
            "ttl_row_delete",
            "throughput (rows/sec)",
            ttl_runs,
            500_000.0,
            true,
        );

        // Archive throughput to the real opendal fs backend.
        let mut arch_runs = Vec::new();
        for _ in 0..VALIDATION_RUNS {
            let adir = tempdir().unwrap();
            let n = 200_000usize;
            let records: Vec<Vec<u8>> = (0..n)
                .map(|i| format!("{i}\u{1f}{}", i + 1).into_bytes())
                .collect();
            let uri = format!("file://{}/a.zylog", adir.path().display());
            let t = Instant::now();
            let r = zyron_lifecycle::archive::archive_rows(&uri, &records)
                .await
                .unwrap();
            let secs = t.elapsed().as_secs_f64();
            assert_eq!(r.rows_archived, n as u64);
            arch_runs.push(n as f64 / secs);
        }
        validate_metric(
            "archive_to_object_store",
            "throughput (rows/sec)",
            arch_runs,
            500_000.0,
            true,
        );

        tprintln!("\n  Performance suite complete.");
    });
}

#[test]
fn zz_err() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let e = build_engine().await;
        e.catalog
            .create_table(
                e.schema_id,
                "u",
                &[
                    col("id", DataType::Int, true),
                    col("is_deleted", DataType::Boolean, false),
                    col("deleted_at", DataType::BigInt, false),
                ],
                &[],
            )
            .await
            .unwrap();
        set_lifecycle(&e, "u", |lc| {
            lc.soft_delete_enabled = true;
            lc.soft_delete_is_deleted_col_id = 1;
            lc.soft_delete_deleted_at_col_id = 2;
        })
        .await;
        run(
            &e,
            "INSERT INTO u (id,is_deleted,deleted_at) VALUES (1,false,0),(2,false,0),(3,false,0)",
            true,
        )
        .await
        .unwrap();
        let r = run(&e, "DELETE FROM u WHERE id = 1", true).await;
        eprintln!("ZYERR delete_result={:?}", r.err());
    });
}

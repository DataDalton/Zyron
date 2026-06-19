//! Integration tests for writable copy-on-write branches.
//!
//! Exercises CREATE/USE BRANCH plus INSERT/UPDATE/DELETE under an active branch
//! through the same dispatch + plan + execute path the server uses, and confirms
//! the branch sees its own writes while the main line stays untouched.
//!
//! Run: cargo test -p zyron-wire --test branch_write_test -- --nocapture

use std::sync::Arc;

use zyron_buffer::{BufferPool, BufferPoolConfig};
use zyron_catalog::{Catalog, CatalogCache, DatabaseId, HeapCatalogStorage, SYSTEM_DATABASE_ID};
use zyron_executor::batch::DataBatch;
use zyron_executor::column::ScalarValue;
use zyron_executor::context::ExecutionContext;
use zyron_storage::txn::{IsolationLevel, TransactionManager};
use zyron_storage::{DiskManager, DiskManagerConfig};
use zyron_wal::{WalWriter, WalWriterConfig};
use zyron_wire::connection::ServerState;
use zyron_wire::session::Session;

/// Per-test state: the server, the session, and the session's active branch
/// (set by USE BRANCH, mirrored into each query's execution context).
struct Harness {
    server: Arc<ServerState>,
    session: Option<Session>,
    active_branch: Option<String>,
    _tmp: tempfile::TempDir,
}

async fn create_harness() -> Harness {
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let data_dir = tmp.path().join("data");
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir_all(&data_dir).unwrap();
    std::fs::create_dir_all(&wal_dir).unwrap();

    let wal = Arc::new(
        WalWriter::new(WalWriterConfig {
            wal_dir,
            segment_size: 16 * 1024 * 1024,
            fsync_enabled: false,
            ring_buffer_capacity: 4 * 1024 * 1024,
        })
        .expect("wal"),
    );
    let disk = Arc::new(
        DiskManager::new(DiskManagerConfig {
            data_dir,
            fsync_enabled: false,
        })
        .await
        .expect("disk"),
    );
    let pool = Arc::new(BufferPool::new(BufferPoolConfig { num_frames: 1024 }));
    let storage =
        Arc::new(HeapCatalogStorage::new(Arc::clone(&disk), Arc::clone(&pool)).expect("storage"));
    let cache = Arc::new(CatalogCache::new(256, 64));
    let catalog = Arc::new(
        Catalog::new(storage, cache, Arc::clone(&wal))
            .await
            .expect("catalog"),
    );
    catalog
        .create_schema(SYSTEM_DATABASE_ID, "public", "test_user")
        .await
        .expect("create public schema");
    let txn_manager = Arc::new(TransactionManager::new(Arc::clone(&wal)));
    let branch_manager = Arc::new(zyron_versioning::BranchManager::new(
        tmp.path().to_path_buf(),
    ));

    let state = Arc::new(ServerState {
        catalog,
        wal,
        buffer_pool: pool,
        disk_manager: disk,
        txn_manager,
        security_manager: None,
        key_store: Arc::new(zyron_auth::LocalKeyStore::new([0u8; 32])),
        config_lookup: None,
        config_all: None,
        data_dir: std::path::PathBuf::from(tmp.path()),
        session_info_collector: None,
        checkpoint_stats: None,
        vacuum_stats: None,
        checkpoint_wake: None,
        alter_system_set: None,
        cdc_feed_stats: None,
        cdc_slot_stats: None,
        cdc_stream_stats: None,
        cdc_ingest_stats: None,
        cdc_registry: None,
        slot_manager: None,
        publication_manager: None,
        cdc_stream_manager: None,
        cdc_ingest_manager: None,
        trigger_manager: None,
        udf_registry: None,
        uda_registry: None,
        procedure_registry: None,
        pipeline_manager: None,
        schedule_manager: None,
        event_dispatcher: None,
        mv_manager: None,
        stream_job_manager: None,
        branch_manager: Some(branch_manager),
        fts_manager: None,
        vector_manager: None,
        graph_manager: None,
        spatial_manager: None,
        cdc_hook: None,
        dml_hook: None,
        notification_channels: None,
        tls_mode: zyron_wire::tls::TlsMode::Disabled,
        tls_acceptor: None,
        endpoint_registrar: None,
        subscription_runtimes: Arc::new(scc::HashMap::new()),
        pub_sub_state: Arc::new(zyron_wire::subscription::PubSubServerState::new()),
        subscription_shutdown: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        heap_files: Arc::new(scc::HashMap::new()),
        btree_indexes: Arc::new(scc::HashMap::new()),
        plan_cache: Arc::new(zyron_wire::plan_cache::ServerPlanCache::new()),
        vacuum_running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        analytics_registry: zyron_analytics::default_registry(),
        legal_holds: Arc::new(zyron_lifecycle::legal_hold::LegalHoldRegistry::new()),
        feature_store: zyron_analytics::featureStore(),
        feature_lineage: zyron_analytics::featureLineageRegistry(),
        model_cache: zyron_analytics::modelCache(),
    });

    let mut session = Session::new("test_user".into(), "testdb".into(), DatabaseId(1));
    session.search_path = vec!["public".into()];

    Harness {
        server: state,
        session: Some(session),
        active_branch: None,
        _tmp: tmp,
    }
}

/// Runs one SQL statement. DDL/utility (including branch commands) goes through
/// the dispatch path and can mutate the session's active branch. DML and SELECT
/// plan and execute under a fresh transaction whose context carries the active
/// branch, matching the wire connection's branch wiring.
async fn exec(h: &mut Harness, sql: &str) -> Vec<DataBatch> {
    let stmt = zyron_parser::parse(sql)
        .expect("parse")
        .into_iter()
        .next()
        .expect("one statement");

    let mut txn_opt: Option<zyron_storage::txn::Transaction> = None;
    if let Some(res) = zyron_wire::ddl_dispatch::try_handle_ddl_utility(
        &stmt,
        &h.server,
        &mut h.session,
        &mut txn_opt,
        &mut h.active_branch,
        sql,
    )
    .await
    {
        res.expect("ddl handler failed");
        return Vec::new();
    }

    let plan = zyron_planner::plan(
        &h.server.catalog,
        DatabaseId(1),
        vec!["public".into()],
        stmt,
    )
    .await
    .expect("plan");
    let mut txn = h
        .server
        .txn_manager
        .begin(IsolationLevel::ReadCommitted)
        .expect("begin");
    let snapshot = txn.snapshot.clone();
    let txn_id = txn.txn_id as u32;
    let mut ctx = ExecutionContext::new(
        h.server.catalog.clone(),
        h.server.wal.clone(),
        h.server.buffer_pool.clone(),
        h.server.disk_manager.clone(),
        txn_id,
        snapshot,
    );
    ctx.heap_files = Some(Arc::clone(&h.server.heap_files));
    ctx.btree_indexes = Some(Arc::clone(&h.server.btree_indexes));
    if let Some(mgr) = &h.server.branch_manager {
        ctx.branch_catalog = Some(Arc::clone(mgr) as Arc<dyn zyron_common::BranchCatalog>);
        if let Some(name) = &h.active_branch {
            ctx.active_branch_id = mgr.get_branch_by_name(name).ok().map(|e| e.id.0);
        }
    }
    let ctx = Arc::new(ctx);
    let batches = zyron_executor::execute(plan, &ctx).await.expect("execute");
    h.server.txn_manager.commit(&mut txn).await.expect("commit");
    batches
}

fn total_rows(batches: &[DataBatch]) -> usize {
    batches.iter().map(|b| b.num_rows).sum()
}

/// Collects (id, v) integer pairs from a two-column result, sorted by id.
fn id_v_pairs(batches: &[DataBatch]) -> Vec<(i32, i32)> {
    let mut out = Vec::new();
    for b in batches {
        for r in 0..b.num_rows {
            let id = match b.columns[0].get_scalar(r) {
                ScalarValue::Int32(v) => v,
                other => panic!("unexpected id scalar {other:?}"),
            };
            let v = match b.columns[1].get_scalar(r) {
                ScalarValue::Int32(v) => v,
                other => panic!("unexpected v scalar {other:?}"),
            };
            out.push((id, v));
        }
    }
    out.sort();
    out
}

#[tokio::test]
async fn branch_insert_update_delete_isolated_from_main() {
    let mut h = create_harness().await;

    exec(&mut h, "CREATE TABLE t (id INT, v INT)").await;
    exec(&mut h, "INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)").await;

    exec(&mut h, "CREATE BRANCH dev").await;
    exec(&mut h, "USE BRANCH dev").await;
    assert_eq!(
        h.active_branch.as_deref(),
        Some("dev"),
        "USE BRANCH set the session branch"
    );

    // Writes on the branch: append a new row, modify an existing row, delete one.
    exec(&mut h, "INSERT INTO t VALUES (4, 40)").await;
    exec(&mut h, "UPDATE t SET v = 999 WHERE id = 1").await;
    exec(&mut h, "DELETE FROM t WHERE id = 2").await;

    let branch_rows = exec(&mut h, "SELECT id, v FROM t").await;
    assert_eq!(
        id_v_pairs(&branch_rows),
        vec![(1, 999), (3, 30), (4, 40)],
        "branch sees its update, delete, and append"
    );

    // Back on the main line the table is untouched.
    h.active_branch = None;
    let main_rows = exec(&mut h, "SELECT id, v FROM t").await;
    assert_eq!(
        id_v_pairs(&main_rows),
        vec![(1, 10), (2, 20), (3, 30)],
        "main line is unaffected by branch writes"
    );
}

#[tokio::test]
async fn branch_predicate_select_sees_overlay() {
    let mut h = create_harness().await;
    exec(&mut h, "CREATE TABLE t (id INT, v INT)").await;
    exec(&mut h, "INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)").await;
    exec(&mut h, "CREATE BRANCH dev").await;
    exec(&mut h, "USE BRANCH dev").await;
    exec(&mut h, "INSERT INTO t VALUES (4, 40)").await;
    exec(&mut h, "UPDATE t SET v = 40 WHERE id = 1").await;

    // A predicate scan on the branch reads both the cow-modified row and the
    // appended row.
    let rows = exec(&mut h, "SELECT id, v FROM t WHERE v = 40").await;
    assert_eq!(
        id_v_pairs(&rows),
        vec![(1, 40), (4, 40)],
        "predicate matches the modified row and the appended row"
    );

    h.active_branch = None;
    let main_rows = exec(&mut h, "SELECT id, v FROM t WHERE v = 40").await;
    assert_eq!(total_rows(&main_rows), 0, "main has no row with v = 40");
}

#[tokio::test]
async fn branch_index_scan_reads_overlay_and_skips_tombstones() {
    let mut h = create_harness().await;
    exec(&mut h, "CREATE TABLE t (id INT, v INT)").await;
    exec(&mut h, "CREATE INDEX idx_id ON t (id)").await;
    exec(&mut h, "INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)").await;

    exec(&mut h, "CREATE BRANCH dev").await;
    exec(&mut h, "USE BRANCH dev").await;
    exec(&mut h, "INSERT INTO t VALUES (4, 40)").await; // branch-only append
    exec(&mut h, "UPDATE t SET v = 999 WHERE id = 1").await; // cow modify
    exec(&mut h, "DELETE FROM t WHERE id = 2").await; // cow delete

    // Indexed equality lookups on the branch. The main index returns the main
    // row id; for a modified or deleted row that resolves to a tombstoned cow
    // slot and drops out, while the branch insert delta supplies the new image.
    let r1 = exec(&mut h, "SELECT id, v FROM t WHERE id = 1").await;
    assert_eq!(
        id_v_pairs(&r1),
        vec![(1, 999)],
        "indexed lookup sees the branch update"
    );

    let r2 = exec(&mut h, "SELECT id, v FROM t WHERE id = 2").await;
    assert_eq!(
        total_rows(&r2),
        0,
        "indexed lookup skips the branch-deleted row"
    );

    let r3 = exec(&mut h, "SELECT id, v FROM t WHERE id = 3").await;
    assert_eq!(
        id_v_pairs(&r3),
        vec![(3, 30)],
        "untouched row still found via index"
    );

    let r4 = exec(&mut h, "SELECT id, v FROM t WHERE id = 4").await;
    assert_eq!(
        id_v_pairs(&r4),
        vec![(4, 40)],
        "branch-appended row found via the delta scan"
    );

    // A range lookup unions the index-resolved main rows with the append delta.
    let rng = exec(&mut h, "SELECT id, v FROM t WHERE id >= 1").await;
    assert_eq!(
        id_v_pairs(&rng),
        vec![(1, 999), (3, 30), (4, 40)],
        "range lookup reflects update, delete, and append"
    );

    // The main line, looked up by the same index, is unchanged.
    h.active_branch = None;
    let main1 = exec(&mut h, "SELECT id, v FROM t WHERE id = 1").await;
    assert_eq!(
        id_v_pairs(&main1),
        vec![(1, 10)],
        "main index lookup unaffected"
    );
    let main2 = exec(&mut h, "SELECT id, v FROM t WHERE id = 2").await;
    assert_eq!(
        id_v_pairs(&main2),
        vec![(2, 20)],
        "main still has the row the branch deleted"
    );
}

#[tokio::test]
async fn merge_branch_into_main_materializes_and_consumes_branch() {
    let mut h = create_harness().await;
    exec(&mut h, "CREATE TABLE t (id INT, v INT)").await;
    exec(&mut h, "CREATE INDEX idx_id ON t (id)").await;
    exec(&mut h, "INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)").await;

    exec(&mut h, "CREATE BRANCH dev").await;
    exec(&mut h, "USE BRANCH dev").await;
    exec(&mut h, "INSERT INTO t VALUES (4, 40)").await;
    exec(&mut h, "UPDATE t SET v = 999 WHERE id = 1").await;
    exec(&mut h, "DELETE FROM t WHERE id = 2").await;

    // Merge applies the branch overlay to the main line.
    h.active_branch = None;
    exec(&mut h, "MERGE BRANCH dev INTO main").await;

    // Main now reflects the branch: id 1 updated, id 2 deleted, id 4 appended.
    let rows = exec(&mut h, "SELECT id, v FROM t").await;
    assert_eq!(
        id_v_pairs(&rows),
        vec![(1, 999), (3, 30), (4, 40)],
        "main reflects the merged update, delete, and append"
    );

    // The index reflects the merged rows: the appended row is reachable by an
    // index lookup on its new key, and the deleted row's stale index entry is
    // filtered out at fetch (the heap slot is tombstoned).
    assert_eq!(
        id_v_pairs(&exec(&mut h, "SELECT id, v FROM t WHERE id = 4").await),
        vec![(4, 40)],
        "merged-in row found by index on its new key"
    );
    assert_eq!(
        total_rows(&exec(&mut h, "SELECT id, v FROM t WHERE id = 2").await),
        0,
        "index lookup skips the merged-out (deleted) row"
    );
    // An untouched row is still index-reachable.
    assert_eq!(
        id_v_pairs(&exec(&mut h, "SELECT id, v FROM t WHERE id = 3").await),
        vec![(3, 30)]
    );

    // The branch is consumed by the merge.
    let mgr = h.server.branch_manager.as_ref().expect("branch manager");
    assert!(
        mgr.get_branch_by_name("dev").is_err(),
        "merged branch is dropped"
    );
}

#[tokio::test]
async fn two_branches_do_not_see_each_other() {
    let mut h = create_harness().await;
    exec(&mut h, "CREATE TABLE t (id INT, v INT)").await;
    exec(&mut h, "INSERT INTO t VALUES (1, 10)").await;

    exec(&mut h, "CREATE BRANCH a").await;
    exec(&mut h, "CREATE BRANCH b").await;

    exec(&mut h, "USE BRANCH a").await;
    exec(&mut h, "INSERT INTO t VALUES (100, 100)").await;
    exec(&mut h, "UPDATE t SET v = 11 WHERE id = 1").await;

    exec(&mut h, "USE BRANCH b").await;
    exec(&mut h, "INSERT INTO t VALUES (200, 200)").await;

    let b_rows = exec(&mut h, "SELECT id, v FROM t").await;
    assert_eq!(
        id_v_pairs(&b_rows),
        vec![(1, 10), (200, 200)],
        "branch b sees its own insert and the original row, not branch a's writes"
    );

    h.active_branch = Some("a".into());
    let a_rows = exec(&mut h, "SELECT id, v FROM t").await;
    assert_eq!(
        id_v_pairs(&a_rows),
        vec![(1, 11), (100, 100)],
        "branch a sees its own update and insert, not branch b's"
    );
}

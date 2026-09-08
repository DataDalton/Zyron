//! Integration tests for materialized views.
//!
//! Exercises CREATE/REFRESH/DROP MATERIALIZED VIEW through the DDL dispatch
//! path. A materialized view stores the query result in a backing table, so a
//! SELECT against it reads the snapshot taken at create/refresh time, not the
//! live base table.
//!
//! Run: cargo test -p zyron-wire --test matview_test -- --nocapture

use std::sync::Arc;

use zyron_buffer::{BufferPool, BufferPoolConfig};
use zyron_catalog::{
    Catalog, CatalogCache, DatabaseId, HeapCatalogStorage, SYSTEM_DATABASE_ID, SchemaId,
};
use zyron_executor::batch::DataBatch;
use zyron_executor::column::ScalarValue;
use zyron_executor::context::ExecutionContext;
use zyron_storage::txn::{IsolationLevel, TransactionManager};
use zyron_storage::{DiskManager, DiskManagerConfig};
use zyron_wal::{WalWriter, WalWriterConfig};
use zyron_wire::connection::ServerState;
use zyron_wire::session::Session;

async fn create_test_server() -> (Arc<ServerState>, SchemaId, tempfile::TempDir) {
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let data_dir = tmp.path().join("data");
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir_all(&data_dir).unwrap();
    std::fs::create_dir_all(&wal_dir).unwrap();

    let wal = Arc::new(WalWriter::new(zyron_bench_harness::wal_config(wal_dir)).expect("wal"));
    let disk = Arc::new(
        DiskManager::new(zyron_bench_harness::disk_config(data_dir))
            .await
            .expect("disk"),
    );
    let pool = Arc::new(BufferPool::new(zyron_bench_harness::buffer_pool_config()));
    let storage =
        Arc::new(HeapCatalogStorage::new(Arc::clone(&disk), Arc::clone(&pool)).expect("storage"));
    let cache = Arc::new(CatalogCache::new(256, 64));
    let catalog = Arc::new(
        Catalog::new(storage, cache, Arc::clone(&wal))
            .await
            .expect("catalog"),
    );
    let public_schema = catalog
        .create_schema(SYSTEM_DATABASE_ID, "zyron_test", "test_user")
        .await
        .expect("create zyron_test schema");
    let txn_manager = Arc::new(TransactionManager::new(Arc::clone(&wal)));

    let state = Arc::new(ServerState {
        raft: None,
        replication: None,
        node_capabilities: None,
        catalog,
        ddl_progress: std::sync::Arc::new(zyron_wire::ddl_progress::DdlProgressRegistry::new()),
        shadow_targets: std::sync::Arc::new(scc::HashMap::new()),
        wal,
        buffer_pool: pool,
        disk_manager: disk,
        txn_manager,
        doc_registry: std::sync::Arc::new(zyron_common::DocRegistry::new()),
        table_io_stats: std::sync::Arc::new(zyron_common::TableIOStatsRegistry::new()),
        index_io_stats: std::sync::Arc::new(zyron_common::IndexIOStatsRegistry::new()),
        columnar_maintenance: None,
        security_manager: None,
        key_store: Arc::new(zyron_auth::LocalKeyStore::new([0u8; 32])),
        media_store: Arc::new(
            zyron_media::store::MediaStore::open(tmp.path().join("data"))
                .expect("media store opens in the test data dir"),
        ),
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
        branch_manager: None,
        fts_manager: None,
        vector_manager: None,
        graph_manager: Some(Arc::new(zyron_search::graph::GraphManager::new())),
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
        cancel_registry: Default::default(),
        heap_files: Arc::new(scc::HashMap::new()),
        btree_indexes: Arc::new(scc::HashMap::new()),
        plan_cache: Arc::new(zyron_wire::plan_cache::ServerPlanCache::new()),
        vacuum_running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        analytics_registry: zyron_analytics::default_registry(),
        legal_holds: Arc::new(zyron_lifecycle::legal_hold::LegalHoldRegistry::new()),
        dlq_registry: Arc::new(zyron_streaming::dlq::DlqRegistry::new()),
        feature_store: zyron_analytics::featureStore(),
        feature_lineage: zyron_analytics::featureLineageRegistry(),
        model_cache: zyron_analytics::modelCache(),
        default_isolation: zyron_storage::IsolationLevel::ReadCommitted,
        deployment_mode: zyron_common::DeploymentMode::Unified,
        node_identity: Default::default(),
        foreign_reader: None,
        peers: Default::default(),
        statement_timeout: None,
        max_result_rows: None,
        max_query_memory: None,
        spill_directory: None,
        balloon_params: None,
        default_auth_method: zyron_auth::auth_rules::AuthMethod::Trust,
        password_encryption: "balloon-sha-256".into(),
        admission: Arc::new(zyron_common::Admission::new()),
        query_metrics: Arc::new(zyron_common::QueryMetrics::new()),
        upgrade_control: None,
    });
    (state, public_schema, tmp)
}

fn new_session() -> Option<Session> {
    let mut s = Session::new("test_user".into(), "testdb".into(), DatabaseId(1));
    s.search_path = vec!["zyron_test".into()];
    Some(s)
}

async fn try_exec(
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
    sql: &str,
) -> std::result::Result<Vec<DataBatch>, String> {
    let stmt = zyron_parser::parse(sql)
        .map_err(|e| e.to_string())?
        .into_iter()
        .next()
        .expect("one statement");

    let mut txn_opt: Option<zyron_storage::txn::Transaction> = None;
    let mut active_branch: Option<String> = None;
    if let Some(res) = zyron_wire::ddl_dispatch::try_handle_ddl_utility(
        &stmt,
        server,
        session,
        &mut txn_opt,
        &mut active_branch,
        sql,
    )
    .await
    {
        return res.map(|_| Vec::new()).map_err(|e| format!("{e:?}"));
    }

    let plan = zyron_planner::plan(
        &server.catalog,
        DatabaseId(1),
        vec!["zyron_test".into()],
        stmt,
        None,
    )
    .await
    .map_err(|e| e.to_string())?;
    let mut txn = server
        .txn_manager
        .begin(IsolationLevel::ReadCommitted)
        .expect("begin");
    let snapshot = txn.snapshot.clone();
    let txn_id = txn.txn_id;
    let mut ctx = ExecutionContext::new(
        server.catalog.clone(),
        server.wal.clone(),
        server.buffer_pool.clone(),
        server.disk_manager.clone(),
        txn_id,
        snapshot,
    );
    ctx.heap_files = Some(Arc::clone(&server.heap_files));
    ctx.btree_indexes = Some(Arc::clone(&server.btree_indexes));
    ctx.intent_locks = Some(Arc::clone(server.txn_manager.intent_locks()));
    let ctx = Arc::new(ctx);
    match zyron_executor::execute(plan, &ctx).await {
        Ok(batches) => {
            server.txn_manager.commit(&mut txn).await.expect("commit");
            Ok(batches)
        }
        Err(e) => Err(e.to_string()),
    }
}

async fn exec(
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
    sql: &str,
) -> Vec<DataBatch> {
    try_exec(server, session, sql)
        .await
        .unwrap_or_else(|e| panic!("statement failed: {sql}\n{e}"))
}

fn col_i64(batches: &[DataBatch], idx: usize) -> Vec<i64> {
    let mut out = Vec::new();
    for b in batches {
        if let Some(col) = b.columns.get(idx) {
            for r in 0..b.num_rows {
                match col.data.get_scalar(r) {
                    ScalarValue::Int64(v) => out.push(v),
                    ScalarValue::Int32(v) => out.push(v as i64),
                    other => panic!("expected integer, got {other:?}"),
                }
            }
        }
    }
    out
}

fn row_count(batches: &[DataBatch]) -> usize {
    batches.iter().map(|b| b.num_rows).sum()
}

async fn seed(server: &Arc<ServerState>, session: &mut Option<Session>) {
    exec(server, session, "CREATE TABLE t (id INT, v INT)").await;
    exec(
        server,
        session,
        "INSERT INTO t (id, v) VALUES (1, 10), (2, 20), (3, 30)",
    )
    .await;
}

#[tokio::test]
async fn create_and_select_materializes_rows() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    exec(
        &server,
        &mut session,
        "CREATE MATERIALIZED VIEW mv AS SELECT id, v FROM t",
    )
    .await;
    let mut ids = col_i64(&exec(&server, &mut session, "SELECT id FROM mv").await, 0);
    ids.sort();
    assert_eq!(ids, vec![1, 2, 3]);
}

#[tokio::test]
async fn materialized_view_is_a_snapshot_until_refresh() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    exec(
        &server,
        &mut session,
        "CREATE MATERIALIZED VIEW mv AS SELECT id, v FROM t",
    )
    .await;
    // Mutate the base table after creating the view.
    exec(
        &server,
        &mut session,
        "INSERT INTO t (id, v) VALUES (4, 40)",
    )
    .await;
    // The view still reflects the original 3-row snapshot.
    assert_eq!(
        row_count(&exec(&server, &mut session, "SELECT id FROM mv").await),
        3
    );
    // After refresh it reflects all 4 rows.
    exec(&server, &mut session, "REFRESH MATERIALIZED VIEW mv").await;
    assert_eq!(
        row_count(&exec(&server, &mut session, "SELECT id FROM mv").await),
        4
    );
}

#[tokio::test]
async fn materialized_view_with_filter() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    exec(
        &server,
        &mut session,
        "CREATE MATERIALIZED VIEW mv AS SELECT id, v FROM t WHERE v >= 20",
    )
    .await;
    let mut ids = col_i64(&exec(&server, &mut session, "SELECT id FROM mv").await, 0);
    ids.sort();
    assert_eq!(ids, vec![2, 3]);
}

#[tokio::test]
async fn refresh_reflects_deletes() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    exec(
        &server,
        &mut session,
        "CREATE MATERIALIZED VIEW mv AS SELECT id, v FROM t",
    )
    .await;
    exec(&server, &mut session, "DELETE FROM t WHERE id = 1").await;
    exec(&server, &mut session, "REFRESH MATERIALIZED VIEW mv").await;
    let mut ids = col_i64(&exec(&server, &mut session, "SELECT id FROM mv").await, 0);
    ids.sort();
    assert_eq!(ids, vec![2, 3]);
}

#[tokio::test]
async fn drop_materialized_view_removes_it() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    exec(
        &server,
        &mut session,
        "CREATE MATERIALIZED VIEW mv AS SELECT id FROM t",
    )
    .await;
    exec(&server, &mut session, "DROP MATERIALIZED VIEW mv").await;
    assert!(
        try_exec(&server, &mut session, "SELECT id FROM mv")
            .await
            .is_err()
    );
}

#[tokio::test]
async fn duplicate_create_errors_without_if_not_exists() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    exec(
        &server,
        &mut session,
        "CREATE MATERIALIZED VIEW mv AS SELECT id FROM t",
    )
    .await;
    assert!(
        try_exec(
            &server,
            &mut session,
            "CREATE MATERIALIZED VIEW mv AS SELECT id FROM t"
        )
        .await
        .is_err()
    );
}

#[tokio::test]
async fn drop_missing_if_exists_is_noop() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    assert!(
        try_exec(
            &server,
            &mut session,
            "DROP MATERIALIZED VIEW IF EXISTS nope"
        )
        .await
        .is_ok()
    );
    assert!(
        try_exec(&server, &mut session, "DROP MATERIALIZED VIEW nope")
            .await
            .is_err()
    );
}

#[tokio::test]
async fn refresh_missing_view_errors() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    assert!(
        try_exec(&server, &mut session, "REFRESH MATERIALIZED VIEW nope")
            .await
            .is_err()
    );
}

// ---------------------------------------------------------------------------
// REFRESH MATERIALIZED VIEW CONCURRENTLY
// ---------------------------------------------------------------------------

#[tokio::test]
async fn refresh_concurrently_applies_the_new_contents() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    exec(
        &server,
        &mut session,
        "CREATE MATERIALIZED VIEW mv AS SELECT id, v FROM t",
    )
    .await;
    exec(
        &server,
        &mut session,
        "CREATE UNIQUE INDEX mv_id ON mv (id)",
    )
    .await;

    exec(&server, &mut session, "UPDATE t SET v = v + 1000").await;
    exec(
        &server,
        &mut session,
        "REFRESH MATERIALIZED VIEW CONCURRENTLY mv",
    )
    .await;

    let mut vs = col_i64(&exec(&server, &mut session, "SELECT v FROM mv").await, 0);
    vs.sort_unstable();
    assert_eq!(vs, vec![1010, 1020, 1030]);
}

#[tokio::test]
async fn refresh_concurrently_tolerates_overlapping_unique_rows() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    exec(
        &server,
        &mut session,
        "CREATE MATERIALIZED VIEW mv AS SELECT id, v FROM t",
    )
    .await;
    exec(
        &server,
        &mut session,
        "CREATE UNIQUE INDEX mv_id ON mv (id)",
    )
    .await;

    // The refresh re-inserts the same unique keys the delete removed inside
    // one transaction, so the uniqueness probe must see the transaction's own
    // deletes or every refresh of unchanged data would conflict.
    exec(
        &server,
        &mut session,
        "REFRESH MATERIALIZED VIEW CONCURRENTLY mv",
    )
    .await;
    exec(
        &server,
        &mut session,
        "REFRESH MATERIALIZED VIEW CONCURRENTLY mv",
    )
    .await;

    let mut ids = col_i64(&exec(&server, &mut session, "SELECT id FROM mv").await, 0);
    ids.sort_unstable();
    assert_eq!(ids, vec![1, 2, 3]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn readers_never_see_a_partial_state_during_concurrent_refresh() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    exec(
        &server,
        &mut session,
        "CREATE MATERIALIZED VIEW mv AS SELECT id, v FROM t",
    )
    .await;
    exec(
        &server,
        &mut session,
        "CREATE UNIQUE INDEX mv_id ON mv (id)",
    )
    .await;

    // Change the source so pre-refresh and post-refresh contents are
    // distinguishable: the old set is {10,20,30}, the new {1010,1020,1030,4}.
    exec(&server, &mut session, "UPDATE t SET v = v + 1000").await;
    exec(&server, &mut session, "INSERT INTO t (id, v) VALUES (4, 4)").await;
    let old_set: Vec<i64> = vec![10, 20, 30];
    let new_set: Vec<i64> = vec![4, 1010, 1020, 1030];

    // One task refreshes while this task reads in a loop. Every read must
    // return exactly the old contents or exactly the new contents.
    let refresher = {
        let server = Arc::clone(&server);
        tokio::spawn(async move {
            let mut session = new_session();
            exec(
                &server,
                &mut session,
                "REFRESH MATERIALIZED VIEW CONCURRENTLY mv",
            )
            .await;
        })
    };

    let mut saw_new = false;
    while !refresher.is_finished() {
        let mut vs = col_i64(&exec(&server, &mut session, "SELECT v FROM mv").await, 0);
        vs.sort_unstable();
        // This has failed once under whole-suite load and never reproduced
        // in isolation, so the message carries what the shape of the torn
        // read was. Which side it came from says where to look: rows only
        // from the old set means the delete became visible before the
        // insert, a mix means one statement saw two commit states, and
        // anything else means the reader saw a row from neither
        if vs != old_set && vs != new_set {
            let from_old: Vec<i64> = vs.iter().copied().filter(|v| old_set.contains(v)).collect();
            let from_new: Vec<i64> = vs.iter().copied().filter(|v| new_set.contains(v)).collect();
            let foreign: Vec<i64> = vs
                .iter()
                .copied()
                .filter(|v| !old_set.contains(v) && !new_set.contains(v))
                .collect();
            panic!(
                "a reader saw a partial refresh state: {vs:?}\n  \
                 rows: {} (old set has {}, new set has {})\n  \
                 from old: {from_old:?}\n  from new: {from_new:?}\n  \
                 from neither: {foreign:?}",
                vs.len(),
                old_set.len(),
                new_set.len()
            );
        }
        if vs == new_set {
            saw_new = true;
        }
        tokio::task::yield_now().await;
    }
    refresher.await.expect("refresh task");

    // After the refresh the new contents are the only visible state.
    let mut vs = col_i64(&exec(&server, &mut session, "SELECT v FROM mv").await, 0);
    vs.sort_unstable();
    assert_eq!(vs, new_set);
    let _ = saw_new;
}

/// Reproduction harness for the tear that
/// `readers_never_see_a_partial_state_during_concurrent_refresh` catches
/// about one run in seventy-five, and only under whole-suite load.
///
/// That test refreshes once with one reader. This drives many refresh
/// cycles with several concurrent readers so the window is entered far more
/// often per run, and reports the shape of any tear. Ignored by default
/// because it is a hunting tool: it either finds a real defect or costs
/// wall clock, and neither belongs in a normal suite run.
///
/// Run it with:
///   cargo test -p zyron-wire --test matview_test tear_hunt -- --ignored --nocapture
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn tear_hunt_concurrent_refresh() {
    const CYCLES: usize = 40;
    const READERS: usize = 4;

    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    exec(
        &server,
        &mut session,
        "CREATE MATERIALIZED VIEW mv AS SELECT id, v FROM t",
    )
    .await;
    // The index is not implicated: the empty read reproduces at the same
    // rate with it dropped. It is kept so this mirrors the test it hunts for
    exec(
        &server,
        &mut session,
        "CREATE UNIQUE INDEX mv_id ON mv (id)",
    )
    .await;

    // Two states the view alternates between. Every read must land on one
    // of them exactly, whichever refresh is in flight
    let low: Vec<i64> = vec![10, 20, 30];
    let high: Vec<i64> = vec![1010, 1020, 1030];

    for cycle in 0..CYCLES {
        let going_high = cycle % 2 == 0;
        let delta = if going_high { "+ 1000" } else { "- 1000" };
        exec(
            &server,
            &mut session,
            &format!("UPDATE t SET v = v {delta}"),
        )
        .await;

        let refresher = {
            let server = Arc::clone(&server);
            tokio::spawn(async move {
                let mut session = new_session();
                exec(
                    &server,
                    &mut session,
                    "REFRESH MATERIALIZED VIEW CONCURRENTLY mv",
                )
                .await;
            })
        };

        let mut readers = Vec::with_capacity(READERS);
        for _ in 0..READERS {
            let server = Arc::clone(&server);
            let low = low.clone();
            let high = high.clone();
            readers.push(tokio::spawn(async move {
                let mut session = new_session();
                for _ in 0..60 {
                    let mut vs = col_i64(&exec(&server, &mut session, "SELECT v FROM mv").await, 0);
                    vs.sort_unstable();
                    if vs != low && vs != high {
                        // Immediately read again. If the rows are back, the
                        // state was never wrong and one scan failed to see
                        // them, which points at the scan. If it is still
                        // empty, the refresh really did leave it that way
                        let mut again =
                            col_i64(&exec(&server, &mut session, "SELECT v FROM mv").await, 0);
                        again.sort_unstable();
                        // The base table too. If `t` also reads empty then
                        // the refresh read nothing to copy, and the fault is
                        // a committed table scanning as empty rather than
                        // anything specific to the view
                        let mut base =
                            col_i64(&exec(&server, &mut session, "SELECT v FROM t").await, 0);
                        base.sort_unstable();
                        let from_low: Vec<i64> =
                            vs.iter().copied().filter(|v| low.contains(v)).collect();
                        let from_high: Vec<i64> =
                            vs.iter().copied().filter(|v| high.contains(v)).collect();
                        let foreign: Vec<i64> = vs
                            .iter()
                            .copied()
                            .filter(|v| !low.contains(v) && !high.contains(v))
                            .collect();
                        panic!(
                            "cycle {cycle}: a reader saw a partial refresh state: {vs:?}\n  \
                             rows: {} (each state has 3)\n  \
                             from low: {from_low:?}\n  from high: {from_high:?}\n  \
                             from neither: {foreign:?}\n  \
                             immediate re-read of mv: {again:?}\n  \
                             base table t reads: {base:?}",
                            vs.len()
                        );
                    }
                    tokio::task::yield_now().await;
                }
            }));
        }

        refresher.await.expect("refresh task");
        for reader in readers {
            reader.await.expect("reader task");
        }
    }
}

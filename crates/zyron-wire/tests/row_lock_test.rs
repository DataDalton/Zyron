//! Integration tests for SELECT ... FOR UPDATE/SHARE row locking.
//!
//! FOR UPDATE takes exclusive row locks on the returned rows, FOR SHARE
//! takes shared locks. DML takes exclusive row locks before writing, so a
//! held FOR UPDATE lock blocks a concurrent UPDATE/DELETE of the same row
//! and two plain writers of one row conflict deterministically. NOWAIT
//! errors on contention, SKIP LOCKED filters contended rows, and the lock
//! cap keeps FOR UPDATE SKIP LOCKED LIMIT n locking exactly n rows. Locks
//! release at commit and abort.
//!
//! Run: cargo test -p zyron-wire --test row_lock_test -- --nocapture

use std::sync::Arc;

use zyron_buffer::{BufferPool, BufferPoolConfig};
use zyron_catalog::{
    Catalog, CatalogCache, DatabaseId, HeapCatalogStorage, SYSTEM_DATABASE_ID, SchemaId,
};
use zyron_executor::batch::DataBatch;
use zyron_executor::column::ScalarValue;
use zyron_executor::context::ExecutionContext;
use zyron_storage::txn::{IsolationLevel, Transaction, TransactionManager};
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

    let wal = Arc::new(
        WalWriter::new(zyron_bench_harness::wal_config(wal_dir))
        .expect("wal"),
    );
    let disk = Arc::new(
        DiskManager::new(zyron_bench_harness::disk_config(data_dir.clone()))
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
        .create_schema(SYSTEM_DATABASE_ID, "public", "test_user")
        .await
        .expect("create public schema");
    let txn_manager = Arc::new(TransactionManager::new(Arc::clone(&wal)));
    let cdc_registry = Arc::new(zyron_cdc::CdfRegistry::new(data_dir.clone()));

    let state = Arc::new(ServerState {
        catalog,
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
        cdc_registry: Some(cdc_registry),
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
        heap_files: Arc::new(scc::HashMap::new()),
        btree_indexes: Arc::new(scc::HashMap::new()),
        plan_cache: Arc::new(zyron_wire::plan_cache::ServerPlanCache::new()),
        vacuum_running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        analytics_registry: zyron_analytics::default_registry(),
        legal_holds: Arc::new(zyron_lifecycle::legal_hold::LegalHoldRegistry::new()),
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
        balloon_params: None,
        default_auth_method: zyron_auth::auth_rules::AuthMethod::Trust,
    });
    (state, public_schema, tmp)
}

fn new_session() -> Option<Session> {
    let mut s = Session::new("test_user".into(), "testdb".into(), DatabaseId(1));
    s.search_path = vec!["public".into()];
    Some(s)
}

fn build_ctx(server: &Arc<ServerState>, txn: &Transaction) -> Arc<ExecutionContext> {
    let mut ctx = ExecutionContext::new(
        server.catalog.clone(),
        server.wal.clone(),
        server.buffer_pool.clone(),
        server.disk_manager.clone(),
        txn.txn_id() as u32,
        txn.snapshot.clone(),
    );
    ctx.heap_files = Some(Arc::clone(&server.heap_files));
    ctx.btree_indexes = Some(Arc::clone(&server.btree_indexes));
    ctx.intent_locks = Some(Arc::clone(server.txn_manager.intent_locks()));
    ctx.row_locks = Some(Arc::clone(server.txn_manager.lock_table()));
    ctx.undo_log = Some(txn.undo_log());
    ctx.read_only = txn.read_only();
    Arc::new(ctx)
}

async fn exec_autocommit(server: &Arc<ServerState>, session: &mut Option<Session>, sql: &str) {
    let stmt = zyron_parser::parse(sql)
        .expect("parse")
        .into_iter()
        .next()
        .expect("one statement");

    let mut txn_opt: Option<Transaction> = None;
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
        res.expect("ddl utility");
        return;
    }

    let plan = zyron_planner::plan(
        &server.catalog,
        DatabaseId(1),
        vec!["public".into()],
        stmt,
        None,
    )
    .await
    .expect("plan");
    let mut txn = server
        .txn_manager
        .begin(IsolationLevel::ReadCommitted)
        .expect("begin");
    let ctx = build_ctx(server, &txn);
    zyron_executor::execute(plan, &ctx).await.expect("execute");
    server.txn_manager.commit(&mut txn).await.expect("commit");
}

/// Plans a statement, surfacing binder/planner rejections as the Err.
async fn try_plan(
    server: &Arc<ServerState>,
    sql: &str,
) -> zyron_common::Result<zyron_planner::PhysicalPlan> {
    let stmt = zyron_parser::parse(sql)
        .expect("parse")
        .into_iter()
        .next()
        .expect("one statement");
    zyron_planner::plan(
        &server.catalog,
        DatabaseId(1),
        vec!["public".into()],
        stmt,
        None,
    )
    .await
}

async fn try_exec_in_txn(
    server: &Arc<ServerState>,
    txn: &Transaction,
    sql: &str,
) -> zyron_common::Result<Vec<DataBatch>> {
    let plan = try_plan(server, sql).await?;
    let ctx = build_ctx(server, txn);
    zyron_executor::execute(plan, &ctx).await
}

async fn exec_in_txn(server: &Arc<ServerState>, txn: &Transaction, sql: &str) -> Vec<DataBatch> {
    try_exec_in_txn(server, txn, sql).await.expect("execute")
}

async fn begin(server: &Arc<ServerState>) -> Transaction {
    server
        .txn_manager
        .begin(IsolationLevel::ReadCommitted)
        .expect("begin")
}

fn sorted_ids(batches: &[DataBatch]) -> Vec<i64> {
    let mut out = Vec::new();
    for b in batches {
        if let Some(col) = b.columns.first() {
            for r in 0..b.num_rows {
                match col.get_scalar(r) {
                    ScalarValue::Int64(v) => out.push(v),
                    ScalarValue::Int32(v) => out.push(v as i64),
                    other => panic!("expected integer, got {other:?}"),
                }
            }
        }
    }
    out.sort();
    out
}

fn assert_conflict(result: zyron_common::Result<Vec<DataBatch>>, expect_in_reason: &str) {
    match result {
        Err(zyron_common::ZyronError::TransactionConflict { reason, .. }) => {
            assert!(
                reason.contains(expect_in_reason),
                "conflict reason {reason:?} does not mention {expect_in_reason:?}"
            );
        }
        Err(other) => panic!("expected TransactionConflict, got {other:?}"),
        Ok(_) => panic!("expected TransactionConflict, statement succeeded"),
    }
}

async fn seed(server: &Arc<ServerState>, session: &mut Option<Session>) {
    exec_autocommit(server, session, "CREATE TABLE jobs (id INT, state INT)").await;
    exec_autocommit(
        server,
        session,
        "INSERT INTO jobs (id, state) VALUES (1, 0), (2, 0), (3, 0)",
    )
    .await;
}

// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn for_update_blocks_concurrent_update_until_commit() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;

    let mut txn1 = begin(&server).await;
    let rows = exec_in_txn(
        &server,
        &txn1,
        "SELECT id FROM jobs WHERE id = 1 FOR UPDATE",
    )
    .await;
    assert_eq!(sorted_ids(&rows), vec![1]);

    // a concurrent write of the locked row conflicts immediately
    let mut txn2 = begin(&server).await;
    assert_conflict(
        try_exec_in_txn(&server, &txn2, "UPDATE jobs SET state = 9 WHERE id = 1").await,
        "locked by txn",
    );
    // an unlocked row writes fine in the same transaction
    exec_in_txn(&server, &txn2, "UPDATE jobs SET state = 5 WHERE id = 2").await;
    server.txn_manager.abort(&mut txn2).expect("abort");

    // commit releases the lock, a fresh writer succeeds
    server.txn_manager.commit(&mut txn1).await.expect("commit");
    let mut txn3 = begin(&server).await;
    exec_in_txn(&server, &txn3, "UPDATE jobs SET state = 9 WHERE id = 1").await;
    server.txn_manager.commit(&mut txn3).await.expect("commit");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn two_plain_writers_of_one_row_conflict_deterministically() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;

    let mut txn1 = begin(&server).await;
    exec_in_txn(&server, &txn1, "UPDATE jobs SET state = 1 WHERE id = 1").await;

    let mut txn2 = begin(&server).await;
    assert_conflict(
        try_exec_in_txn(&server, &txn2, "UPDATE jobs SET state = 2 WHERE id = 1").await,
        "locked by txn",
    );
    assert_conflict(
        try_exec_in_txn(&server, &txn2, "DELETE FROM jobs WHERE id = 1").await,
        "locked by txn",
    );
    server.txn_manager.abort(&mut txn2).expect("abort");
    server.txn_manager.commit(&mut txn1).await.expect("commit");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn nowait_errors_and_skip_locked_filters() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;

    let mut txn1 = begin(&server).await;
    exec_in_txn(
        &server,
        &txn1,
        "SELECT id FROM jobs WHERE id = 1 FOR UPDATE",
    )
    .await;

    let mut txn2 = begin(&server).await;
    assert_conflict(
        try_exec_in_txn(&server, &txn2, "SELECT id FROM jobs FOR UPDATE NOWAIT").await,
        "NOWAIT",
    );
    let rows = exec_in_txn(&server, &txn2, "SELECT id FROM jobs FOR UPDATE SKIP LOCKED").await;
    assert_eq!(
        sorted_ids(&rows),
        vec![2, 3],
        "SKIP LOCKED returns every row txn1 does not hold"
    );
    server.txn_manager.abort(&mut txn2).expect("abort");
    server.txn_manager.commit(&mut txn1).await.expect("commit");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn skip_locked_limit_locks_exactly_the_returned_rows() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;

    // the work queue pattern: each worker claims one job
    let mut worker1 = begin(&server).await;
    let claimed1 = exec_in_txn(
        &server,
        &worker1,
        "SELECT id FROM jobs ORDER BY id LIMIT 1 FOR UPDATE SKIP LOCKED",
    )
    .await;
    assert_eq!(sorted_ids(&claimed1), vec![1]);

    let mut worker2 = begin(&server).await;
    let claimed2 = exec_in_txn(
        &server,
        &worker2,
        "SELECT id FROM jobs ORDER BY id LIMIT 1 FOR UPDATE SKIP LOCKED",
    )
    .await;
    assert_eq!(
        sorted_ids(&claimed2),
        vec![2],
        "the cap keeps worker1 from having locked the whole batch"
    );

    server
        .txn_manager
        .commit(&mut worker1)
        .await
        .expect("commit");
    server
        .txn_manager
        .commit(&mut worker2)
        .await
        .expect("commit");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn for_share_allows_shared_readers_and_blocks_writers() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;

    let mut txn1 = begin(&server).await;
    exec_in_txn(&server, &txn1, "SELECT id FROM jobs WHERE id = 1 FOR SHARE").await;

    // a second shared locker is compatible
    let mut txn2 = begin(&server).await;
    let rows = exec_in_txn(&server, &txn2, "SELECT id FROM jobs WHERE id = 1 FOR SHARE").await;
    assert_eq!(sorted_ids(&rows), vec![1]);

    // an exclusive writer is blocked while any shared holder remains
    let mut txn3 = begin(&server).await;
    assert_conflict(
        try_exec_in_txn(&server, &txn3, "UPDATE jobs SET state = 9 WHERE id = 1").await,
        "locked by txn",
    );
    server.txn_manager.abort(&mut txn3).expect("abort");
    server.txn_manager.commit(&mut txn2).await.expect("commit");
    server.txn_manager.commit(&mut txn1).await.expect("commit");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn abort_releases_row_locks() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;

    let mut txn1 = begin(&server).await;
    exec_in_txn(
        &server,
        &txn1,
        "SELECT id FROM jobs WHERE id = 3 FOR UPDATE",
    )
    .await;
    server.txn_manager.abort(&mut txn1).expect("abort");

    let mut txn2 = begin(&server).await;
    exec_in_txn(&server, &txn2, "UPDATE jobs SET state = 7 WHERE id = 3").await;
    server.txn_manager.commit(&mut txn2).await.expect("commit");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn invalid_for_update_shapes_are_rejected_at_plan_time() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    exec_autocommit(&server, &mut session, "CREATE TABLE other (id INT)").await;

    for (sql, needle) in [
        ("SELECT count(*) FROM jobs FOR UPDATE", "aggregate"),
        ("SELECT id FROM jobs GROUP BY id FOR UPDATE", "GROUP BY"),
        ("SELECT DISTINCT id FROM jobs FOR UPDATE", "DISTINCT"),
        (
            "SELECT j.id FROM jobs j, other o FOR UPDATE",
            "single base table",
        ),
        ("SELECT id FROM jobs FOR UPDATE OF other", "does not name"),
    ] {
        match try_plan(&server, sql).await {
            Err(e) => {
                let msg = format!("{e}");
                assert!(
                    msg.contains(needle),
                    "error for {sql:?} was {msg:?}, expected it to mention {needle:?}"
                );
            }
            Ok(_) => panic!("{sql:?} planned successfully, expected a rejection"),
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn for_update_wait_parks_until_holder_aborts() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;

    let mut txn1 = begin(&server).await;
    exec_in_txn(
        &server,
        &txn1,
        "SELECT id FROM jobs WHERE id = 1 FOR UPDATE",
    )
    .await;

    // the waiter parks, then acquires after the holder rolls back. The
    // holder committed nothing, so no conflict is raised
    let waiter = {
        let server = Arc::clone(&server);
        tokio::spawn(async move {
            let mut txn2 = begin(&server).await;
            let rows = try_exec_in_txn(
                &server,
                &txn2,
                "SELECT id FROM jobs WHERE id = 1 FOR UPDATE",
            )
            .await;
            server.txn_manager.commit(&mut txn2).await.expect("commit");
            rows
        })
    };
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    server.txn_manager.abort(&mut txn1).expect("abort");

    let rows = waiter.await.expect("join").expect("waiter succeeds");
    assert_eq!(sorted_ids(&rows), vec![1]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn for_update_wait_conflicts_when_holder_committed_a_change() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;

    let mut txn1 = begin(&server).await;
    exec_in_txn(&server, &txn1, "UPDATE jobs SET state = 1 WHERE id = 1").await;

    // the waiter parks behind the writer, the writer commits, the waiter
    // must not return the stale pre-commit row image under its lock
    let waiter = {
        let server = Arc::clone(&server);
        tokio::spawn(async move {
            let mut txn2 = begin(&server).await;
            let rows = try_exec_in_txn(
                &server,
                &txn2,
                "SELECT id FROM jobs WHERE id = 1 FOR UPDATE",
            )
            .await;
            server.txn_manager.abort(&mut txn2).expect("abort");
            rows
        })
    };
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    server.txn_manager.commit(&mut txn1).await.expect("commit");

    assert_conflict(waiter.await.expect("join"), "concurrently committed");
}

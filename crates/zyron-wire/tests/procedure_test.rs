//! Integration tests for stored procedures and CALL.
//!
//! Defines SQL procedures whose body references parameters positionally as $1,
//! $2, then exercises CALL: single-statement inserts, multi-statement bodies,
//! an update, argument-count validation, and DROP PROCEDURE. Verifies the body
//! runs with the call arguments bound as parameters.
//!
//! Run: cargo test -p zyron-wire --test procedure_test -- --nocapture

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
    let public_schema = catalog
        .create_schema(SYSTEM_DATABASE_ID, "public", "test_user")
        .await
        .expect("create public schema");
    let txn_manager = Arc::new(TransactionManager::new(Arc::clone(&wal)));

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
    });
    (state, public_schema, tmp)
}

fn new_session() -> Option<Session> {
    let mut s = Session::new("test_user".into(), "testdb".into(), DatabaseId(1));
    s.search_path = vec!["public".into()];
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

    let plan = zyron_planner::plan(&server.catalog, DatabaseId(1), vec!["public".into()], stmt)
        .await
        .map_err(|e| e.to_string())?;
    let mut txn = server
        .txn_manager
        .begin(IsolationLevel::ReadCommitted)
        .expect("begin");
    let snapshot = txn.snapshot.clone();
    let txn_id = txn.txn_id as u32;
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

/// Collects column `idx` across all batches as i64, sorted ascending.
fn sorted_col(batches: &[DataBatch], idx: usize) -> Vec<i64> {
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
    out.sort_unstable();
    out
}

async fn seed(server: &Arc<ServerState>, session: &mut Option<Session>) {
    exec(server, session, "CREATE TABLE t (g INT, x INT)").await;
}

#[tokio::test]
async fn call_runs_single_statement_body_with_params() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    exec(
        &server,
        &mut session,
        "CREATE PROCEDURE add_row(grp INT, val INT) AS 'INSERT INTO t (g, x) VALUES ($1, $2)' LANGUAGE SQL",
    )
    .await;
    exec(&server, &mut session, "CALL add_row(1, 100)").await;
    exec(&server, &mut session, "CALL add_row(1, 200)").await;
    let xs = sorted_col(
        &exec(&server, &mut session, "SELECT x FROM t WHERE g = 1").await,
        0,
    );
    assert_eq!(xs, vec![100, 200]);
}

#[tokio::test]
async fn call_runs_multi_statement_body() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    exec(
        &server,
        &mut session,
        "CREATE PROCEDURE add_pair(a INT, b INT) AS 'INSERT INTO t (g, x) VALUES ($1, $2); INSERT INTO t (g, x) VALUES ($2, $1)' LANGUAGE SQL",
    )
    .await;
    exec(&server, &mut session, "CALL add_pair(3, 4)").await;
    // Two rows inserted: (3,4) and (4,3).
    let gs = sorted_col(&exec(&server, &mut session, "SELECT g FROM t").await, 0);
    assert_eq!(gs, vec![3, 4]);
    let xs = sorted_col(&exec(&server, &mut session, "SELECT x FROM t").await, 0);
    assert_eq!(xs, vec![3, 4]);
}

#[tokio::test]
async fn call_runs_update_body_with_param() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    exec(
        &server,
        &mut session,
        "INSERT INTO t (g, x) VALUES (1, 10), (2, 20), (3, 30)",
    )
    .await;
    exec(
        &server,
        &mut session,
        "CREATE PROCEDURE set_all(val INT) AS 'UPDATE t SET x = $1' LANGUAGE SQL",
    )
    .await;
    exec(&server, &mut session, "CALL set_all(5)").await;
    let xs = sorted_col(&exec(&server, &mut session, "SELECT x FROM t").await, 0);
    assert_eq!(xs, vec![5, 5, 5]);
}

#[tokio::test]
async fn call_with_wrong_argument_count_errors() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    exec(
        &server,
        &mut session,
        "CREATE PROCEDURE add_row(grp INT, val INT) AS 'INSERT INTO t (g, x) VALUES ($1, $2)' LANGUAGE SQL",
    )
    .await;
    let err = try_exec(&server, &mut session, "CALL add_row(1)")
        .await
        .expect_err("wrong arg count should error");
    assert!(
        err.to_lowercase().contains("argument") || err.contains("add_row"),
        "unexpected error: {err}"
    );
}

#[tokio::test]
async fn drop_procedure_then_call_errors() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    exec(
        &server,
        &mut session,
        "CREATE PROCEDURE add_row(grp INT, val INT) AS 'INSERT INTO t (g, x) VALUES ($1, $2)' LANGUAGE SQL",
    )
    .await;
    exec(&server, &mut session, "CALL add_row(1, 100)").await;
    exec(&server, &mut session, "DROP PROCEDURE add_row").await;
    let err = try_exec(&server, &mut session, "CALL add_row(1, 100)")
        .await
        .expect_err("dropped procedure should not resolve");
    assert!(
        err.to_lowercase().contains("not found") || err.contains("add_row"),
        "unexpected error: {err}"
    );
}

#[tokio::test]
async fn create_procedure_rejects_malformed_body() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    let err = try_exec(
        &server,
        &mut session,
        "CREATE PROCEDURE bad() AS 'INSRT INTO t VALUES (1)' LANGUAGE SQL",
    )
    .await
    .expect_err("malformed body should be rejected at creation");
    assert!(
        err.to_lowercase().contains("parse") || err.to_lowercase().contains("body"),
        "unexpected error: {err}"
    );
}

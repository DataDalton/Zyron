//! Integration tests for data-quality expectations.
//!
//! Exercises ALTER TABLE ADD/DROP EXPECTATION through the DDL dispatch path and
//! the four violation actions (FAIL, WARN, DROP, QUARANTINE) through the plan +
//! execute INSERT pipeline. Verifies that surviving rows land in the table and
//! quarantined rows land in the companion quarantine table.
//!
//! Run: cargo test -p zyron-wire --test expectation_test -- --nocapture

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
        default_isolation: zyron_storage::IsolationLevel::ReadCommitted,
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

/// Collects an integer column (by position) across all batches as i64.
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

async fn ids_in(server: &Arc<ServerState>, session: &mut Option<Session>, table: &str) -> Vec<i64> {
    let mut ids = col_i64(
        &exec(server, session, &format!("SELECT id FROM {table}")).await,
        0,
    );
    ids.sort_unstable();
    ids
}

#[tokio::test]
async fn fail_action_rejects_violating_row() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE t (id INT, amount INT)").await;
    exec(
        &server,
        &mut session,
        "ALTER TABLE t ADD EXPECTATION pos EXPECT amount > 0 ON VIOLATION FAIL",
    )
    .await;
    // A passing row is accepted.
    exec(
        &server,
        &mut session,
        "INSERT INTO t (id, amount) VALUES (1, 10)",
    )
    .await;
    // A violating row aborts the statement.
    let err = try_exec(
        &server,
        &mut session,
        "INSERT INTO t (id, amount) VALUES (2, -5)",
    )
    .await
    .expect_err("violating insert should fail");
    assert!(
        err.contains("pos") || err.to_lowercase().contains("violat"),
        "unexpected error: {err}"
    );
    assert_eq!(ids_in(&server, &mut session, "t").await, vec![1]);
}

#[tokio::test]
async fn warn_action_keeps_violating_row() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE t (id INT, amount INT)").await;
    exec(
        &server,
        &mut session,
        "ALTER TABLE t ADD EXPECTATION pos EXPECT amount > 0 ON VIOLATION WARN",
    )
    .await;
    exec(
        &server,
        &mut session,
        "INSERT INTO t (id, amount) VALUES (1, 10), (2, -5)",
    )
    .await;
    // Both rows are kept; the violation is only counted.
    assert_eq!(ids_in(&server, &mut session, "t").await, vec![1, 2]);
}

#[tokio::test]
async fn drop_action_filters_violating_rows() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE t (id INT, amount INT)").await;
    exec(
        &server,
        &mut session,
        "ALTER TABLE t ADD EXPECTATION pos EXPECT amount > 0 ON VIOLATION DROP",
    )
    .await;
    exec(
        &server,
        &mut session,
        "INSERT INTO t (id, amount) VALUES (1, 10), (2, -5), (3, 20)",
    )
    .await;
    // Only passing rows survive.
    assert_eq!(ids_in(&server, &mut session, "t").await, vec![1, 3]);
}

#[tokio::test]
async fn quarantine_action_routes_violating_rows() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE t (id INT, amount INT)").await;
    exec(
        &server,
        &mut session,
        "ALTER TABLE t ADD EXPECTATION pos EXPECT amount > 0 ON VIOLATION QUARANTINE",
    )
    .await;
    exec(
        &server,
        &mut session,
        "INSERT INTO t (id, amount) VALUES (1, 10), (2, -5), (3, 20)",
    )
    .await;
    // Passing rows land in the table; the violating row lands in quarantine.
    assert_eq!(ids_in(&server, &mut session, "t").await, vec![1, 3]);
    assert_eq!(ids_in(&server, &mut session, "t_quarantine").await, vec![2]);
}

#[tokio::test]
async fn drop_expectation_stops_enforcement() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE t (id INT, amount INT)").await;
    exec(
        &server,
        &mut session,
        "ALTER TABLE t ADD EXPECTATION pos EXPECT amount > 0 ON VIOLATION FAIL",
    )
    .await;
    // While the expectation exists, a violating row is rejected.
    try_exec(
        &server,
        &mut session,
        "INSERT INTO t (id, amount) VALUES (2, -5)",
    )
    .await
    .expect_err("violating insert should fail while expectation is present");
    // After dropping the expectation, the same row is accepted.
    exec(&server, &mut session, "ALTER TABLE t DROP EXPECTATION pos").await;
    exec(
        &server,
        &mut session,
        "INSERT INTO t (id, amount) VALUES (2, -5)",
    )
    .await;
    assert_eq!(ids_in(&server, &mut session, "t").await, vec![2]);
}

#[tokio::test]
async fn duplicate_expectation_name_is_rejected() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE t (id INT, amount INT)").await;
    exec(
        &server,
        &mut session,
        "ALTER TABLE t ADD EXPECTATION pos EXPECT amount > 0 ON VIOLATION FAIL",
    )
    .await;
    let err = try_exec(
        &server,
        &mut session,
        "ALTER TABLE t ADD EXPECTATION pos EXPECT amount > 1 ON VIOLATION FAIL",
    )
    .await
    .expect_err("duplicate name should be rejected");
    assert!(err.contains("pos"), "unexpected error: {err}");
}

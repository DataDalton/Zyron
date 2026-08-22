//! Integration tests for DO blocks and ALTER TABLE ENABLE/DISABLE features.
//!
//! DO blocks run an anonymous SQL body atomically. Feature toggles flip a
//! per-table flag (soft_delete) and, for change_data_feed, register or remove
//! the table's change feed.
//!
//! Run: cargo test -p zyron-wire --test do_feature_test -- --nocapture

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
        vec!["public".into()],
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

fn col_i64(batches: &[DataBatch], idx: usize) -> Vec<i64> {
    let mut out = Vec::new();
    for b in batches {
        if let Some(col) = b.columns.get(idx) {
            for r in 0..b.num_rows {
                match col.get_scalar(r) {
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

fn table_id(server: &Arc<ServerState>, schema: SchemaId, name: &str) -> u32 {
    server.catalog.get_table(schema, name).expect("table").id.0
}

// -- DO blocks --------------------------------------------------------------

#[tokio::test]
async fn do_block_runs_body() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE t (id INT)").await;
    exec(
        &server,
        &mut session,
        "DO 'INSERT INTO t (id) VALUES (1); INSERT INTO t (id) VALUES (2)'",
    )
    .await;
    let mut ids = col_i64(&exec(&server, &mut session, "SELECT id FROM t").await, 0);
    ids.sort();
    assert_eq!(ids, vec![1, 2]);
}

#[tokio::test]
async fn do_block_is_atomic() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE t (id INT)").await;
    // The second statement targets a missing table, so the whole block aborts.
    let result = try_exec(
        &server,
        &mut session,
        "DO 'INSERT INTO t (id) VALUES (1); INSERT INTO missing (id) VALUES (2)'",
    )
    .await;
    assert!(result.is_err());
    assert_eq!(
        row_count(&exec(&server, &mut session, "SELECT id FROM t").await),
        0
    );
}

#[tokio::test]
async fn do_block_rejects_non_sql_language() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    assert!(
        try_exec(&server, &mut session, "DO 'BEGIN END' LANGUAGE plpgsql")
            .await
            .is_err()
    );
}

// -- Feature toggles --------------------------------------------------------

#[tokio::test]
async fn enable_disable_soft_delete() {
    let (server, schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE t (id INT)").await;

    exec(&server, &mut session, "ALTER TABLE t ENABLE soft_delete").await;
    assert!(
        server
            .catalog
            .get_table(schema, "t")
            .unwrap()
            .lifecycle
            .soft_delete_enabled
    );

    exec(&server, &mut session, "ALTER TABLE t DISABLE soft_delete").await;
    assert!(
        !server
            .catalog
            .get_table(schema, "t")
            .unwrap()
            .lifecycle
            .soft_delete_enabled
    );
}

#[tokio::test]
async fn enable_disable_change_data_feed() {
    let (server, schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE t (id INT)").await;
    let tid = table_id(&server, schema, "t");

    exec(
        &server,
        &mut session,
        "ALTER TABLE t ENABLE change_data_feed",
    )
    .await;
    assert!(server.catalog.get_table(schema, "t").unwrap().cdf_enabled);
    // The feed is registered so changes are captured.
    assert!(
        server
            .cdc_registry
            .as_ref()
            .unwrap()
            .get_feed(tid)
            .is_some()
    );

    exec(
        &server,
        &mut session,
        "ALTER TABLE t DISABLE change_data_feed",
    )
    .await;
    assert!(!server.catalog.get_table(schema, "t").unwrap().cdf_enabled);
    assert!(
        server
            .cdc_registry
            .as_ref()
            .unwrap()
            .get_feed(tid)
            .is_none()
    );
}

#[tokio::test]
async fn enable_unknown_feature_errors() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE t (id INT)").await;
    assert!(
        try_exec(&server, &mut session, "ALTER TABLE t ENABLE warp_drive")
            .await
            .is_err()
    );
}

#[tokio::test]
async fn enable_feature_unknown_table_errors() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    assert!(
        try_exec(&server, &mut session, "ALTER TABLE nope ENABLE soft_delete")
            .await
            .is_err()
    );
}

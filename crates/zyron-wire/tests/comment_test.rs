//! Integration tests for COMMENT ON.
//!
//! Exercises COMMENT ON {TABLE|COLUMN|SCHEMA|SEQUENCE|VIEW} through the DDL
//! dispatch path and verifies the comment is persisted in the catalog and
//! removable with IS NULL.
//!
//! Run: cargo test -p zyron-wire --test comment_test -- --nocapture

use std::sync::Arc;

use zyron_buffer::{BufferPool, BufferPoolConfig};
use zyron_catalog::{
    Catalog, CatalogCache, DatabaseId, HeapCatalogStorage, SYSTEM_DATABASE_ID, SchemaId,
};
use zyron_executor::batch::DataBatch;
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

async fn exec(server: &Arc<ServerState>, session: &mut Option<Session>, sql: &str) {
    let stmt = zyron_parser::parse(sql)
        .expect("parse")
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
        res.unwrap_or_else(|e| panic!("ddl failed: {sql}\n{e:?}"));
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
    let _: Vec<DataBatch> = zyron_executor::execute(plan, &ctx).await.expect("execute");
    server.txn_manager.commit(&mut txn).await.expect("commit");
}

const TABLE: u8 = 0;
const COLUMN: u8 = 1;
const SCHEMA: u8 = 3;

#[tokio::test]
async fn comment_on_table_persists() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE t (id INT)").await;
    exec(&server, &mut session, "COMMENT ON TABLE t IS 'the t table'").await;
    assert_eq!(
        server.catalog.get_comment(TABLE, "t", ""),
        Some("the t table".to_string())
    );
}

#[tokio::test]
async fn comment_overwrite_replaces_text() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE t (id INT)").await;
    exec(&server, &mut session, "COMMENT ON TABLE t IS 'first'").await;
    exec(&server, &mut session, "COMMENT ON TABLE t IS 'second'").await;
    assert_eq!(
        server.catalog.get_comment(TABLE, "t", ""),
        Some("second".to_string())
    );
    // One comment, not two, for the same object.
    assert_eq!(server.catalog.list_comments().len(), 1);
}

#[tokio::test]
async fn comment_is_null_removes() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE t (id INT)").await;
    exec(&server, &mut session, "COMMENT ON TABLE t IS 'doc'").await;
    exec(&server, &mut session, "COMMENT ON TABLE t IS NULL").await;
    assert_eq!(server.catalog.get_comment(TABLE, "t", ""), None);
    assert_eq!(server.catalog.list_comments().len(), 0);
}

#[tokio::test]
async fn comment_on_column_persists() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE t (id INT, v INT)").await;
    exec(
        &server,
        &mut session,
        "COMMENT ON COLUMN t.v IS 'the value'",
    )
    .await;
    assert_eq!(
        server.catalog.get_comment(COLUMN, "t", "v"),
        Some("the value".to_string())
    );
    // A table comment and a column comment are distinct keys.
    assert_eq!(server.catalog.get_comment(TABLE, "t", ""), None);
}

#[tokio::test]
async fn comment_on_schema_persists() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(
        &server,
        &mut session,
        "COMMENT ON SCHEMA public IS 'default schema'",
    )
    .await;
    assert_eq!(
        server.catalog.get_comment(SCHEMA, "public", ""),
        Some("default schema".to_string())
    );
}

#[tokio::test]
async fn distinct_objects_keep_separate_comments() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE a (id INT)").await;
    exec(&server, &mut session, "CREATE TABLE b (id INT)").await;
    exec(&server, &mut session, "COMMENT ON TABLE a IS 'table a'").await;
    exec(&server, &mut session, "COMMENT ON TABLE b IS 'table b'").await;
    assert_eq!(
        server.catalog.get_comment(TABLE, "a", ""),
        Some("table a".to_string())
    );
    assert_eq!(
        server.catalog.get_comment(TABLE, "b", ""),
        Some("table b".to_string())
    );
    assert_eq!(server.catalog.list_comments().len(), 2);
}

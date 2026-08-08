//! Integration tests for replication slots.
//!
//! Exercises CREATE/DROP REPLICATION SLOT through the DDL dispatch path. A slot
//! records a consumer's WAL position and pins WAL retention from the current
//! head at creation time.
//!
//! Run: cargo test -p zyron-wire --test slot_test -- --nocapture

use std::sync::Arc;

use zyron_buffer::{BufferPool, BufferPoolConfig};
use zyron_catalog::{
    Catalog, CatalogCache, DatabaseId, HeapCatalogStorage, SYSTEM_DATABASE_ID, SchemaId,
};
use zyron_cdc::decoder::DecoderPlugin;
use zyron_cdc::replication_slot::{SlotLagConfig, SlotManager};
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
            data_dir: data_dir.clone(),
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
    let slot_mgr =
        Arc::new(SlotManager::open(&data_dir, SlotLagConfig::default()).expect("slot mgr"));

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
        slot_manager: Some(slot_mgr),
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

fn table_id(server: &Arc<ServerState>, schema: SchemaId, name: &str) -> u32 {
    server.catalog.get_table(schema, name).expect("table").id.0
}

#[tokio::test]
async fn create_and_drop_slot() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(
        &server,
        &mut session,
        "CREATE REPLICATION SLOT s1 PLUGIN 'zyron_cdc'",
    )
    .await;
    let mgr = server.slot_manager.as_ref().unwrap();
    let slot = mgr.get_slot("s1").expect("slot exists");
    assert_eq!(slot.plugin, DecoderPlugin::ZyronCdc);
    assert!(slot.active);
    assert!(slot.table_filter.is_none());

    exec(&server, &mut session, "DROP REPLICATION SLOT s1").await;
    assert!(mgr.get_slot("s1").is_err());
}

#[tokio::test]
async fn create_slot_pins_wal_from_head() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    let pin = server.wal.next_lsn().0;
    exec(
        &server,
        &mut session,
        "CREATE REPLICATION SLOT s1 PLUGIN 'zyron_cdc'",
    )
    .await;
    let mgr = server.slot_manager.as_ref().unwrap();
    let slot = mgr.get_slot("s1").unwrap();
    // The slot pins retention at (or after) the head captured before creation,
    // and confirmed/restart move together.
    assert!(slot.restart_lsn >= pin);
    assert_eq!(slot.restart_lsn, slot.confirmed_lsn);
}

#[tokio::test]
async fn create_slot_with_table_filter() {
    let (server, schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE orders (id INT)").await;
    exec(&server, &mut session, "CREATE TABLE items (id INT)").await;
    exec(
        &server,
        &mut session,
        "CREATE REPLICATION SLOT s1 PLUGIN 'debezium' FOR TABLE orders, items",
    )
    .await;
    let mgr = server.slot_manager.as_ref().unwrap();
    let slot = mgr.get_slot("s1").unwrap();
    let mut filter = slot.table_filter.expect("filter set");
    filter.sort();
    let mut expected = vec![
        table_id(&server, schema, "orders"),
        table_id(&server, schema, "items"),
    ];
    expected.sort();
    assert_eq!(filter, expected);
}

#[tokio::test]
async fn create_slot_unknown_filter_table_errors() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    assert!(
        try_exec(
            &server,
            &mut session,
            "CREATE REPLICATION SLOT s1 PLUGIN 'zyron_cdc' FOR TABLE nope"
        )
        .await
        .is_err()
    );
    // The failed filter resolution leaves no slot behind.
    assert!(
        server
            .slot_manager
            .as_ref()
            .unwrap()
            .get_slot("s1")
            .is_err()
    );
}

#[tokio::test]
async fn create_slot_invalid_plugin_errors() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    assert!(
        try_exec(
            &server,
            &mut session,
            "CREATE REPLICATION SLOT s1 PLUGIN 'not_a_plugin'"
        )
        .await
        .is_err()
    );
}

#[tokio::test]
async fn duplicate_slot_errors() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(
        &server,
        &mut session,
        "CREATE REPLICATION SLOT s1 PLUGIN 'zyron_cdc'",
    )
    .await;
    assert!(
        try_exec(
            &server,
            &mut session,
            "CREATE REPLICATION SLOT s1 PLUGIN 'zyron_cdc'"
        )
        .await
        .is_err()
    );
}

#[tokio::test]
async fn drop_unknown_slot_errors() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    assert!(
        try_exec(&server, &mut session, "DROP REPLICATION SLOT nope")
            .await
            .is_err()
    );
}

#[tokio::test]
async fn min_restart_lsn_reflects_created_slot() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    // Write some WAL so the head advances past zero.
    exec(&server, &mut session, "CREATE TABLE t (id INT)").await;
    exec(&server, &mut session, "INSERT INTO t (id) VALUES (1)").await;
    exec(
        &server,
        &mut session,
        "CREATE REPLICATION SLOT s1 PLUGIN 'zyron_cdc'",
    )
    .await;
    let mgr = server.slot_manager.as_ref().unwrap();
    // The slot now pins retention, so the minimum restart LSN is set.
    assert!(mgr.min_restart_lsn().is_some());
}

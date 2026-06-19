//! Integration tests for CREATE VERSION (named version tags).
//!
//! CREATE VERSION registers a named, immutable tag pointing at a table version.
//! Time-travel queries resolve `VERSION AS OF '<name>'` to the tagged version
//! through the binder. Tables without version headers fall back to snapshot
//! visibility, so a named-version query returns current rows; the test asserts
//! the name resolves (no error) and that an unknown name errors.
//!
//! Run: cargo test -p zyron-wire --test version_test -- --nocapture

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

async fn seed(server: &Arc<ServerState>, session: &mut Option<Session>) {
    exec(server, session, "CREATE TABLE t (id INT)").await;
    exec(server, session, "INSERT INTO t (id) VALUES (1), (2), (3)").await;
}

#[tokio::test]
async fn create_version_persists_tag() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    exec(&server, &mut session, "CREATE VERSION v1 ON t").await;
    let tag = server
        .catalog
        .get_version_tag_by_name("v1")
        .expect("tag exists");
    assert_eq!(tag.name, "v1");
    // Tagged at the current WAL position (non-zero after the seed writes).
    assert!(tag.version_id > 0);
}

#[tokio::test]
async fn create_version_explicit_version_number() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    exec(
        &server,
        &mut session,
        "CREATE VERSION v5 ON t AS OF VERSION 5",
    )
    .await;
    let tag = server
        .catalog
        .get_version_tag_by_name("v5")
        .expect("tag exists");
    assert_eq!(tag.version_id, 5);
}

#[tokio::test]
async fn duplicate_version_errors() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    exec(&server, &mut session, "CREATE VERSION v1 ON t").await;
    assert!(
        try_exec(&server, &mut session, "CREATE VERSION v1 ON t")
            .await
            .is_err()
    );
}

#[tokio::test]
async fn create_version_unknown_table_errors() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    assert!(
        try_exec(&server, &mut session, "CREATE VERSION v1 ON nope")
            .await
            .is_err()
    );
}

#[tokio::test]
async fn named_version_resolves_in_time_travel_query() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    exec(&server, &mut session, "CREATE VERSION v1 ON t").await;
    // The named version resolves through the binder; the query executes.
    let mut ids = col_i64(
        &exec(&server, &mut session, "SELECT id FROM t VERSION AS OF 'v1'").await,
        0,
    );
    ids.sort();
    assert_eq!(ids, vec![1, 2, 3]);
}

#[tokio::test]
async fn unknown_named_version_errors_in_query() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    assert!(
        try_exec(
            &server,
            &mut session,
            "SELECT id FROM t VERSION AS OF 'ghost'"
        )
        .await
        .is_err()
    );
}

#[tokio::test]
async fn integer_version_still_works() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    // A raw integer version still binds (no named-tag lookup).
    assert!(
        try_exec(
            &server,
            &mut session,
            "SELECT id FROM t VERSION AS OF 999999"
        )
        .await
        .is_ok()
    );
}

#[tokio::test]
async fn time_travel_update_returns_old_value() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    // Tag the state before the update. The tag dates the version by commit LSN.
    exec(&server, &mut session, "CREATE VERSION v1 ON t").await;
    exec(&server, &mut session, "UPDATE t SET id = 99 WHERE id = 1").await;

    // Current state reflects the update.
    let mut now = col_i64(&exec(&server, &mut session, "SELECT id FROM t").await, 0);
    now.sort();
    assert_eq!(now, vec![2, 3, 99]);

    // The tagged version still sees the pre-update row.
    let mut past = col_i64(
        &exec(&server, &mut session, "SELECT id FROM t VERSION AS OF 'v1'").await,
        0,
    );
    past.sort();
    assert_eq!(past, vec![1, 2, 3]);
}

#[tokio::test]
async fn time_travel_delete_row_still_visible_at_version() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    exec(&server, &mut session, "CREATE VERSION v1 ON t").await;
    exec(&server, &mut session, "DELETE FROM t WHERE id = 2").await;

    // Current state no longer has the deleted row.
    let mut now = col_i64(&exec(&server, &mut session, "SELECT id FROM t").await, 0);
    now.sort();
    assert_eq!(now, vec![1, 3]);

    // The tagged version still sees the deleted row: vacuum kept it because its
    // deleter committed above the retention floor.
    let mut past = col_i64(
        &exec(&server, &mut session, "SELECT id FROM t VERSION AS OF 'v1'").await,
        0,
    );
    past.sort();
    assert_eq!(past, vec![1, 2, 3]);
}

#[tokio::test]
async fn two_versions_see_their_respective_states() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    exec(&server, &mut session, "CREATE VERSION v1 ON t").await;
    exec(&server, &mut session, "UPDATE t SET id = 99 WHERE id = 1").await;
    exec(&server, &mut session, "CREATE VERSION v2 ON t").await;

    let mut v1 = col_i64(
        &exec(&server, &mut session, "SELECT id FROM t VERSION AS OF 'v1'").await,
        0,
    );
    v1.sort();
    assert_eq!(v1, vec![1, 2, 3]);

    let mut v2 = col_i64(
        &exec(&server, &mut session, "SELECT id FROM t VERSION AS OF 'v2'").await,
        0,
    );
    v2.sort();
    assert_eq!(v2, vec![2, 3, 99]);
}

#[tokio::test]
async fn drop_version_removes_tag_and_raises_floor() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    exec(&server, &mut session, "CREATE VERSION v1 ON t").await;
    let floor_with_tag = server.txn_manager.status_map().version_retention_floor();
    assert!(
        floor_with_tag != u64::MAX,
        "a tag pins the retention floor below the dawn-of-everything sentinel"
    );

    exec(&server, &mut session, "DROP VERSION v1").await;
    assert!(
        server.catalog.get_version_tag_by_name("v1").is_none(),
        "the tag is gone after DROP VERSION"
    );
    assert_eq!(
        server.txn_manager.status_map().version_retention_floor(),
        u64::MAX,
        "dropping the last tag releases the retention floor"
    );
    // The name no longer resolves in a time-travel query.
    assert!(
        try_exec(&server, &mut session, "SELECT id FROM t VERSION AS OF 'v1'")
            .await
            .is_err()
    );
}

#[tokio::test]
async fn drop_version_floor_falls_back_to_oldest_remaining_tag() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    // Two tags at explicit versions; the floor tracks the oldest.
    exec(
        &server,
        &mut session,
        "CREATE VERSION v_old ON t AS OF VERSION 10",
    )
    .await;
    exec(
        &server,
        &mut session,
        "CREATE VERSION v_new ON t AS OF VERSION 90",
    )
    .await;
    assert_eq!(
        server.txn_manager.status_map().version_retention_floor(),
        10
    );

    // Dropping the newer tag leaves the older floor in place.
    exec(&server, &mut session, "DROP VERSION v_new").await;
    assert_eq!(
        server.txn_manager.status_map().version_retention_floor(),
        10
    );

    // Dropping the older tag releases the floor entirely.
    exec(&server, &mut session, "DROP VERSION v_old").await;
    assert_eq!(
        server.txn_manager.status_map().version_retention_floor(),
        u64::MAX
    );
}

#[tokio::test]
async fn drop_unknown_version_errors_unless_if_exists() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    assert!(
        try_exec(&server, &mut session, "DROP VERSION ghost")
            .await
            .is_err()
    );
    // IF EXISTS makes the drop a no-op success.
    assert!(
        try_exec(&server, &mut session, "DROP VERSION IF EXISTS ghost")
            .await
            .is_ok()
    );
}

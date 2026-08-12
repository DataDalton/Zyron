//! MERGE integration tests.
//!
//! MERGE desugars into UPDATE, DELETE and INSERT with correlated
//! subqueries executed atomically in one transaction, see
//! zyron_parser::merge_desugar for the rewrite and its guardrails.
//!
//! Run: cargo test -p zyron-wire --test merge_test -- --nocapture

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
        WalWriter::new(zyron_bench_harness::wal_config(wal_dir))
        .expect("wal"),
    );
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
    exec(server, session, "CREATE TABLE tgt (k INT, v INT)").await;
    exec(server, session, "CREATE TABLE src (k INT, v INT)").await;
    exec(
        server,
        session,
        "INSERT INTO tgt (k, v) VALUES (1, 10), (2, 20), (3, 30)",
    )
    .await;
    exec(
        server,
        session,
        "INSERT INTO src (k, v) VALUES (2, 200), (3, 300), (4, 400)",
    )
    .await;
}

#[tokio::test]
async fn merge_upsert_updates_matched_and_inserts_unmatched() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    exec(
        &server,
        &mut session,
        "MERGE INTO tgt USING src ON tgt.k = src.k          WHEN MATCHED THEN UPDATE SET v = src.v          WHEN NOT MATCHED THEN INSERT (k, v) VALUES (src.k, src.v)",
    )
    .await;
    let ks = sorted_col(&exec(&server, &mut session, "SELECT k FROM tgt").await, 0);
    assert_eq!(ks, vec![1, 2, 3, 4]);
    let vs = sorted_col(&exec(&server, &mut session, "SELECT v FROM tgt").await, 0);
    assert_eq!(vs, vec![10, 200, 300, 400]);
}

#[tokio::test]
async fn merge_matched_delete_removes_matched_rows() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    exec(
        &server,
        &mut session,
        "MERGE INTO tgt USING src ON tgt.k = src.k WHEN MATCHED THEN DELETE",
    )
    .await;
    let ks = sorted_col(&exec(&server, &mut session, "SELECT k FROM tgt").await, 0);
    assert_eq!(ks, vec![1]);
}

#[tokio::test]
async fn merge_clause_condition_narrows_the_action() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    exec(
        &server,
        &mut session,
        "MERGE INTO tgt USING src ON tgt.k = src.k          WHEN MATCHED AND src.v > 250 THEN UPDATE SET v = src.v",
    )
    .await;
    let vs = sorted_col(&exec(&server, &mut session, "SELECT v FROM tgt").await, 0);
    assert_eq!(vs, vec![10, 20, 300]);
}

#[tokio::test]
async fn merge_update_only_and_insert_only_forms() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    exec(
        &server,
        &mut session,
        "MERGE INTO tgt USING src ON tgt.k = src.k          WHEN NOT MATCHED THEN INSERT (k, v) VALUES (src.k, src.v)",
    )
    .await;
    let ks = sorted_col(&exec(&server, &mut session, "SELECT k FROM tgt").await, 0);
    assert_eq!(ks, vec![1, 2, 3, 4]);
    let vs = sorted_col(&exec(&server, &mut session, "SELECT v FROM tgt").await, 0);
    assert_eq!(vs, vec![10, 20, 30, 400]);
}

#[tokio::test]
async fn merge_rejects_assigning_on_columns() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    let err = try_exec(
        &server,
        &mut session,
        "MERGE INTO tgt USING src ON tgt.k = src.k WHEN MATCHED THEN UPDATE SET k = src.v",
    )
    .await
    .expect_err("assigning an ON column must fail");
    assert!(err.contains("ON"), "unexpected error: {err}");
}

#[tokio::test]
async fn merge_rejects_delete_with_insert_combo() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    let err = try_exec(
        &server,
        &mut session,
        "MERGE INTO tgt USING src ON tgt.k = src.k          WHEN MATCHED THEN DELETE          WHEN NOT MATCHED THEN INSERT (k, v) VALUES (src.k, src.v)",
    )
    .await
    .expect_err("delete plus insert combination must fail");
    assert!(err.contains("snapshot"), "unexpected error: {err}");
}

#[tokio::test]
async fn merge_unmatched_source_only_touches_matching_targets() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    exec(
        &server,
        &mut session,
        "MERGE INTO tgt USING src ON tgt.k = src.k WHEN MATCHED THEN UPDATE SET v = 0",
    )
    .await;
    let vs = sorted_col(&exec(&server, &mut session, "SELECT v FROM tgt").await, 0);
    assert_eq!(vs, vec![0, 0, 10]);
}

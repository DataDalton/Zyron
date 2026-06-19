//! Integration tests for declarative pipelines.
//!
//! Exercises CREATE/RUN/DROP PIPELINE through the DDL dispatch path. A pipeline
//! is a set of stages, each transforming a source into a target table; RUN
//! executes the stages in topological order, creating targets on demand,
//! enforcing expectations, and loading per the stage refresh mode.
//!
//! Run: cargo test -p zyron-wire --test pipeline_test -- --nocapture

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
use zyron_wire::ddl_dispatch::DdlResult;
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
        pipeline_manager: Some(Arc::new(zyron_pipeline::pipeline::PipelineManager::new())),
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

/// Runs a DDL/utility statement and returns its DdlResult so callers can inspect
/// preview rows. Errors are returned as strings.
async fn run_ddl(
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
    sql: &str,
) -> std::result::Result<DdlResult, String> {
    let stmt = zyron_parser::parse(sql)
        .map_err(|e| e.to_string())?
        .into_iter()
        .next()
        .expect("one statement");
    let mut txn_opt: Option<zyron_storage::txn::Transaction> = None;
    let mut active_branch: Option<String> = None;
    match zyron_wire::ddl_dispatch::try_handle_ddl_utility(
        &stmt,
        server,
        session,
        &mut txn_opt,
        &mut active_branch,
        sql,
    )
    .await
    {
        Some(res) => res.map_err(|e| format!("{e:?}")),
        None => Err("statement did not dispatch as DDL/utility".to_string()),
    }
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

fn row_count(batches: &[DataBatch]) -> usize {
    batches.iter().map(|b| b.num_rows).sum()
}

async fn seed_src(server: &Arc<ServerState>, session: &mut Option<Session>) {
    exec(server, session, "CREATE TABLE src (id INT, v INT)").await;
    exec(
        server,
        session,
        "INSERT INTO src (id, v) VALUES (1, 10), (2, 20), (3, 30)",
    )
    .await;
}

#[tokio::test]
async fn create_run_full_creates_target_and_loads() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed_src(&server, &mut session).await;
    exec(
        &server,
        &mut session,
        "CREATE PIPELINE p AS (STAGE s (SOURCE src, TARGET dst, MODE full, \
         TRANSFORM AS (SELECT id, v FROM src WHERE v >= 20)))",
    )
    .await;
    // Target does not exist before the run.
    assert!(
        try_exec(&server, &mut session, "SELECT id FROM dst")
            .await
            .is_err()
    );
    exec(&server, &mut session, "RUN PIPELINE p").await;
    let mut ids = col_i64(&exec(&server, &mut session, "SELECT id FROM dst").await, 0);
    ids.sort();
    assert_eq!(ids, vec![2, 3]);
}

#[tokio::test]
async fn full_refresh_rebuilds_not_appends() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed_src(&server, &mut session).await;
    exec(
        &server,
        &mut session,
        "CREATE PIPELINE p AS (STAGE s (SOURCE src, TARGET dst, MODE full, \
         TRANSFORM AS (SELECT id, v FROM src)))",
    )
    .await;
    exec(&server, &mut session, "RUN PIPELINE p").await;
    exec(&server, &mut session, "RUN PIPELINE p").await;
    // Full mode clears then loads, so the row count matches the source, not double.
    assert_eq!(
        row_count(&exec(&server, &mut session, "SELECT id FROM dst").await),
        3
    );
}

#[tokio::test]
async fn append_mode_accumulates() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed_src(&server, &mut session).await;
    exec(
        &server,
        &mut session,
        "CREATE PIPELINE p AS (STAGE s (SOURCE src, TARGET dst, MODE append, \
         TRANSFORM AS (SELECT id, v FROM src)))",
    )
    .await;
    exec(&server, &mut session, "RUN PIPELINE p").await;
    exec(&server, &mut session, "RUN PIPELINE p").await;
    // Append mode adds the source rows on each run.
    assert_eq!(
        row_count(&exec(&server, &mut session, "SELECT id FROM dst").await),
        6
    );
}

#[tokio::test]
async fn multi_stage_runs_in_topological_order() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE raw (id INT, v INT)").await;
    exec(
        &server,
        &mut session,
        "INSERT INTO raw (id, v) VALUES (1, 5), (2, 25), (3, 35)",
    )
    .await;
    // silver depends on bronze's target; the run must order bronze before silver.
    exec(
        &server,
        &mut session,
        "CREATE PIPELINE etl AS (\
           STAGE silver (SOURCE bronze_t, TARGET silver_t, MODE full, \
             TRANSFORM AS (SELECT id, v FROM bronze_t WHERE v >= 25)), \
           STAGE bronze (SOURCE raw, TARGET bronze_t, MODE full, \
             TRANSFORM AS (SELECT id, v FROM raw)))",
    )
    .await;
    exec(&server, &mut session, "RUN PIPELINE etl").await;
    assert_eq!(
        row_count(&exec(&server, &mut session, "SELECT id FROM bronze_t").await),
        3
    );
    let mut ids = col_i64(
        &exec(&server, &mut session, "SELECT id FROM silver_t").await,
        0,
    );
    ids.sort();
    assert_eq!(ids, vec![2, 3]);
}

#[tokio::test]
async fn expectation_violation_fails_run() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE src (id INT, v INT)").await;
    exec(
        &server,
        &mut session,
        "INSERT INTO src (id, v) VALUES (1, 10), (2, -5)",
    )
    .await;
    exec(
        &server,
        &mut session,
        "CREATE PIPELINE p AS (STAGE s (SOURCE src, TARGET dst, MODE full, \
         TRANSFORM AS (SELECT id, v FROM src), EXPECT v > 0))",
    )
    .await;
    // The negative row violates EXPECT v > 0, so the run fails.
    assert!(
        try_exec(&server, &mut session, "RUN PIPELINE p")
            .await
            .is_err()
    );
    // The expectation is checked before loading, so the target stays empty.
    assert_eq!(
        row_count(&exec(&server, &mut session, "SELECT id FROM dst").await),
        0
    );
}

#[tokio::test]
async fn expectation_pass_loads() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed_src(&server, &mut session).await;
    exec(
        &server,
        &mut session,
        "CREATE PIPELINE p AS (STAGE s (SOURCE src, TARGET dst, MODE full, \
         TRANSFORM AS (SELECT id, v FROM src), EXPECT v > 0))",
    )
    .await;
    exec(&server, &mut session, "RUN PIPELINE p").await;
    assert_eq!(
        row_count(&exec(&server, &mut session, "SELECT id FROM dst").await),
        3
    );
}

#[tokio::test]
async fn preview_returns_rows_without_writing() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed_src(&server, &mut session).await;
    exec(
        &server,
        &mut session,
        "CREATE PIPELINE p AS (STAGE s (SOURCE src, TARGET dst, MODE full, \
         TRANSFORM AS (SELECT id, v FROM src)))",
    )
    .await;
    let result = run_ddl(&server, &mut session, "RUN PIPELINE p PREVIEW LIMIT 2")
        .await
        .expect("preview");
    match result {
        DdlResult::Rows { rows, .. } => assert_eq!(rows.len(), 2),
        other => panic!("expected preview rows, got {other:?}"),
    }
    // Preview does not create or load the target.
    assert!(
        try_exec(&server, &mut session, "SELECT id FROM dst")
            .await
            .is_err()
    );
}

#[tokio::test]
async fn run_single_named_stage() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE raw (id INT, v INT)").await;
    exec(
        &server,
        &mut session,
        "INSERT INTO raw (id, v) VALUES (1, 5), (2, 25)",
    )
    .await;
    exec(
        &server,
        &mut session,
        "CREATE PIPELINE etl AS (\
           STAGE bronze (SOURCE raw, TARGET bronze_t, MODE full, \
             TRANSFORM AS (SELECT id, v FROM raw)), \
           STAGE silver (SOURCE bronze_t, TARGET silver_t, MODE full, \
             TRANSFORM AS (SELECT id, v FROM bronze_t)))",
    )
    .await;
    exec(&server, &mut session, "RUN PIPELINE etl STAGE bronze").await;
    // Only the named stage ran: bronze_t is populated, silver_t never created.
    assert_eq!(
        row_count(&exec(&server, &mut session, "SELECT id FROM bronze_t").await),
        2
    );
    assert!(
        try_exec(&server, &mut session, "SELECT id FROM silver_t")
            .await
            .is_err()
    );
}

#[tokio::test]
async fn merge_mode_upserts_by_primary_key() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE src (id INT, v INT)").await;
    exec(
        &server,
        &mut session,
        "INSERT INTO src (id, v) VALUES (1, 10), (2, 20)",
    )
    .await;
    // Pre-create the target with a primary key so MERGE has a key to upsert on.
    exec(
        &server,
        &mut session,
        "CREATE TABLE dst (id INT PRIMARY KEY, v INT)",
    )
    .await;
    exec(
        &server,
        &mut session,
        "CREATE PIPELINE p AS (STAGE s (SOURCE src, TARGET dst, MODE merge, \
         TRANSFORM AS (SELECT id, v FROM src)))",
    )
    .await;
    exec(&server, &mut session, "RUN PIPELINE p").await;
    assert_eq!(
        row_count(&exec(&server, &mut session, "SELECT id FROM dst").await),
        2
    );

    // Change id=1's value and add id=3, then re-run: id=1 updates, no duplicate.
    exec(&server, &mut session, "UPDATE src SET v = 99 WHERE id = 1").await;
    exec(
        &server,
        &mut session,
        "INSERT INTO src (id, v) VALUES (3, 30)",
    )
    .await;
    exec(&server, &mut session, "RUN PIPELINE p").await;
    let mut ids = col_i64(&exec(&server, &mut session, "SELECT id FROM dst").await, 0);
    ids.sort();
    assert_eq!(ids, vec![1, 2, 3]);
    let v1 = col_i64(
        &exec(&server, &mut session, "SELECT v FROM dst WHERE id = 1").await,
        0,
    );
    assert_eq!(v1, vec![99]);
}

#[tokio::test]
async fn merge_without_primary_key_errors() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed_src(&server, &mut session).await;
    // Target auto-created from the transform has no primary key.
    exec(
        &server,
        &mut session,
        "CREATE PIPELINE p AS (STAGE s (SOURCE src, TARGET dst, MODE merge, \
         TRANSFORM AS (SELECT id, v FROM src)))",
    )
    .await;
    assert!(
        try_exec(&server, &mut session, "RUN PIPELINE p")
            .await
            .is_err()
    );
}

#[tokio::test]
async fn cyclic_pipeline_rejected_at_create() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    // a -> b -> c -> a forms a cycle in the stage graph.
    let result = try_exec(
        &server,
        &mut session,
        "CREATE PIPELINE c AS (\
           STAGE a (SOURCE t_c, TARGET t_a), \
           STAGE b (SOURCE t_a, TARGET t_b), \
           STAGE c (SOURCE t_b, TARGET t_c))",
    )
    .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn duplicate_create_errors() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed_src(&server, &mut session).await;
    exec(
        &server,
        &mut session,
        "CREATE PIPELINE p AS (STAGE s (SOURCE src, TARGET dst, MODE full, \
         TRANSFORM AS (SELECT id, v FROM src)))",
    )
    .await;
    assert!(
        try_exec(
            &server,
            &mut session,
            "CREATE PIPELINE p AS (STAGE s2 (SOURCE src, TARGET dst2, MODE full, \
         TRANSFORM AS (SELECT id, v FROM src)))"
        )
        .await
        .is_err()
    );
}

#[tokio::test]
async fn drop_removes_pipeline() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed_src(&server, &mut session).await;
    exec(
        &server,
        &mut session,
        "CREATE PIPELINE p AS (STAGE s (SOURCE src, TARGET dst, MODE full, \
         TRANSFORM AS (SELECT id, v FROM src)))",
    )
    .await;
    assert!(server.catalog.get_pipeline_by_name("p").is_some());
    exec(&server, &mut session, "DROP PIPELINE p").await;
    assert!(server.catalog.get_pipeline_by_name("p").is_none());
    // A run after drop fails because the pipeline no longer exists.
    assert!(
        try_exec(&server, &mut session, "RUN PIPELINE p")
            .await
            .is_err()
    );
}

#[tokio::test]
async fn drop_missing_if_exists_is_noop() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    assert!(
        try_exec(&server, &mut session, "DROP PIPELINE IF EXISTS nope")
            .await
            .is_ok()
    );
    assert!(
        try_exec(&server, &mut session, "DROP PIPELINE nope")
            .await
            .is_err()
    );
}

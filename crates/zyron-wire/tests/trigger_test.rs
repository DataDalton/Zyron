//! Integration tests for triggers.
//!
//! Defines triggers that fire a stored procedure on INSERT/UPDATE/DELETE and
//! verifies: AFTER INSERT fires once per row with the row's columns bound as
//! $1..$N, AFTER DELETE sees the OLD row, statement-level fires once, the
//! recursion guard stops a self-triggering loop, DROP TRIGGER stops firing, and
//! a trigger referencing a missing procedure is rejected at creation.
//!
//! Run: cargo test -p zyron-wire --test trigger_test -- --nocapture

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
        Err(e) => {
            let _ = server.txn_manager.abort(&mut txn);
            Err(e.to_string())
        }
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

/// Creates table t(id,v), an audit sink, and a procedure that logs $1 into it.
async fn base_schema(server: &Arc<ServerState>, session: &mut Option<Session>) {
    exec(server, session, "CREATE TABLE t (id INT, v INT)").await;
    exec(server, session, "CREATE TABLE audit (tid INT)").await;
    exec(
        server,
        session,
        "CREATE PROCEDURE log_row() AS 'INSERT INTO audit (tid) VALUES ($1)' LANGUAGE SQL",
    )
    .await;
}

#[tokio::test]
async fn after_insert_fires_per_row() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    base_schema(&server, &mut session).await;
    exec(
        &server,
        &mut session,
        "CREATE TRIGGER trg AFTER INSERT ON t FOR EACH ROW EXECUTE FUNCTION log_row",
    )
    .await;

    exec(
        &server,
        &mut session,
        "INSERT INTO t (id, v) VALUES (1, 10), (2, 20), (3, 30)",
    )
    .await;
    // The trigger fired once per row, logging each id ($1).
    let logged = sorted_col(
        &exec(&server, &mut session, "SELECT tid FROM audit").await,
        0,
    );
    assert_eq!(logged, vec![1, 2, 3]);
}

#[tokio::test]
async fn after_delete_sees_old_row() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    base_schema(&server, &mut session).await;
    exec(
        &server,
        &mut session,
        "INSERT INTO t (id, v) VALUES (1, 10), (2, 20)",
    )
    .await;
    exec(
        &server,
        &mut session,
        "CREATE TRIGGER trg AFTER DELETE ON t FOR EACH ROW EXECUTE FUNCTION log_row",
    )
    .await;
    exec(&server, &mut session, "DELETE FROM t WHERE id = 1").await;
    // The deleted row's id is logged from the OLD image.
    let logged = sorted_col(
        &exec(&server, &mut session, "SELECT tid FROM audit").await,
        0,
    );
    assert_eq!(logged, vec![1]);
}

#[tokio::test]
async fn statement_level_fires_once() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE t (id INT, v INT)").await;
    exec(&server, &mut session, "CREATE TABLE audit (tid INT)").await;
    // Statement-level triggers run with no row params; the body uses a constant.
    exec(
        &server,
        &mut session,
        "CREATE PROCEDURE log_stmt() AS 'INSERT INTO audit (tid) VALUES (99)' LANGUAGE SQL",
    )
    .await;
    exec(
        &server,
        &mut session,
        "CREATE TRIGGER trg AFTER INSERT ON t FOR EACH STATEMENT EXECUTE FUNCTION log_stmt",
    )
    .await;
    exec(
        &server,
        &mut session,
        "INSERT INTO t (id, v) VALUES (1, 10), (2, 20), (3, 30)",
    )
    .await;
    // Fired once for the whole statement, not per row.
    let logged = sorted_col(
        &exec(&server, &mut session, "SELECT tid FROM audit").await,
        0,
    );
    assert_eq!(logged, vec![99]);
}

#[tokio::test]
async fn recursion_guard_stops_self_trigger() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE t (id INT, v INT)").await;
    // A procedure that inserts back into t, re-firing the trigger.
    exec(
        &server,
        &mut session,
        "CREATE PROCEDURE recurse() AS 'INSERT INTO t (id, v) VALUES ($1, $2)' LANGUAGE SQL",
    )
    .await;
    exec(
        &server,
        &mut session,
        "CREATE TRIGGER trg AFTER INSERT ON t FOR EACH ROW EXECUTE FUNCTION recurse",
    )
    .await;
    let err = try_exec(
        &server,
        &mut session,
        "INSERT INTO t (id, v) VALUES (1, 10)",
    )
    .await
    .expect_err("self-triggering insert must hit the recursion guard");
    assert!(
        err.to_lowercase().contains("recursion") || err.to_lowercase().contains("depth"),
        "unexpected: {err}"
    );
}

#[tokio::test]
async fn drop_trigger_stops_firing() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    base_schema(&server, &mut session).await;
    exec(
        &server,
        &mut session,
        "CREATE TRIGGER trg AFTER INSERT ON t FOR EACH ROW EXECUTE FUNCTION log_row",
    )
    .await;
    exec(&server, &mut session, "DROP TRIGGER trg ON t").await;
    exec(
        &server,
        &mut session,
        "INSERT INTO t (id, v) VALUES (1, 10)",
    )
    .await;
    // No trigger fired, so the audit sink is empty.
    let logged = sorted_col(
        &exec(&server, &mut session, "SELECT tid FROM audit").await,
        0,
    );
    assert!(logged.is_empty());
}

#[tokio::test]
async fn create_trigger_requires_existing_procedure() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE t (id INT, v INT)").await;
    let err = try_exec(
        &server,
        &mut session,
        "CREATE TRIGGER trg AFTER INSERT ON t FOR EACH ROW EXECUTE FUNCTION nope",
    )
    .await
    .expect_err("missing trigger procedure should be rejected");
    assert!(
        err.to_lowercase().contains("procedure") || err.to_lowercase().contains("nope"),
        "unexpected: {err}"
    );
}

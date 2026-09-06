//! Integration tests for triggers.
//!
//! Defines triggers that fire a stored procedure on INSERT/UPDATE/DELETE and
//! verifies: AFTER INSERT fires once per row with the row's columns bound as
//! $1..$N, AFTER DELETE sees the OLD row, statement-level fires once, the
//! recursion guard stops a self-triggering loop, DROP TRIGGER stops firing, and
//! a trigger referencing a missing procedure is rejected at creation. The
//! INSTEAD OF section covers view writes: the trigger body runs instead of a
//! direct write, the underlying table stays untouched, UPDATE binds OLD then
//! NEW, and the creation rules (view target, FOR EACH ROW, one trigger per
//! event) are enforced.
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
        .create_schema(SYSTEM_DATABASE_ID, "zyron_test", "test_user")
        .await
        .expect("create zyron_test schema");
    let txn_manager = Arc::new(TransactionManager::new(Arc::clone(&wal)));

    let state = Arc::new(ServerState {
        raft: None,
        replication: None,
        node_capabilities: None,
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
        media_store: Arc::new(
            zyron_media::store::MediaStore::open(tmp.path().join("data"))
                .expect("media store opens in the test data dir"),
        ),
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
        cancel_registry: Default::default(),
        heap_files: Arc::new(scc::HashMap::new()),
        btree_indexes: Arc::new(scc::HashMap::new()),
        plan_cache: Arc::new(zyron_wire::plan_cache::ServerPlanCache::new()),
        vacuum_running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        analytics_registry: zyron_analytics::default_registry(),
        legal_holds: Arc::new(zyron_lifecycle::legal_hold::LegalHoldRegistry::new()),
        dlq_registry: Arc::new(zyron_streaming::dlq::DlqRegistry::new()),
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
        max_query_memory: None,
        spill_directory: None,
        balloon_params: None,
        default_auth_method: zyron_auth::auth_rules::AuthMethod::Trust,
        password_encryption: "balloon-sha-256".into(),
        admission: Arc::new(zyron_common::Admission::new()),
        query_metrics: Arc::new(zyron_common::QueryMetrics::new()),
        upgrade_control: None,
    });
    (state, public_schema, tmp)
}

fn new_session() -> Option<Session> {
    let mut s = Session::new("test_user".into(), "testdb".into(), DatabaseId(1));
    s.search_path = vec!["zyron_test".into()];
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
        vec!["zyron_test".into()],
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
    let db_txn_id = txn.txn_id;
    let txn_id = txn.txn_id;
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
            // Mirrors the wire layer: a lake version becomes visible only
            // after the durable commit, so a suite that skips this reads a
            // lake table as permanently empty
            let _ = zyron_lake::publish_txn(server.disk_manager.data_dir(), db_txn_id);
            Ok(batches)
        }
        Err(e) => {
            let _ = server.txn_manager.abort(&mut txn);
            let _ = zyron_lake::abandon_txn(server.disk_manager.data_dir(), db_txn_id);
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
        "CREATE PROCEDURE log_row() AS 'INSERT INTO zyron_test.audit (tid) VALUES ($1)' LANGUAGE SQL",
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
        "CREATE PROCEDURE log_stmt() AS 'INSERT INTO zyron_test.audit (tid) VALUES (99)' LANGUAGE SQL",
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
        "CREATE PROCEDURE recurse() AS 'INSERT INTO zyron_test.t (id, v) VALUES ($1, $2)' LANGUAGE SQL",
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

#[tokio::test]
async fn triggers_fire_on_a_lake_table_the_same_as_on_a_heap_one() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(
        &server,
        &mut session,
        "CREATE TABLE lt (id INT, v INT) USING ZYRONLAKE",
    )
    .await;
    exec(&server, &mut session, "CREATE TABLE audit (tid INT)").await;
    exec(
        &server,
        &mut session,
        "CREATE PROCEDURE log_row() AS 'INSERT INTO zyron_test.audit (tid) VALUES ($1)' LANGUAGE SQL",
    )
    .await;
    exec(
        &server,
        &mut session,
        "CREATE TRIGGER lt_trg AFTER INSERT ON lt FOR EACH ROW EXECUTE FUNCTION log_row",
    )
    .await;

    exec(
        &server,
        &mut session,
        "INSERT INTO lt (id, v) VALUES (1, 10), (2, 20), (3, 30)",
    )
    .await;

    // Where a row is stored does not decide whether its triggers run
    let logged = sorted_col(
        &exec(&server, &mut session, "SELECT tid FROM audit").await,
        0,
    );
    assert_eq!(logged, vec![1, 2, 3]);
    assert_eq!(
        sorted_col(&exec(&server, &mut session, "SELECT id FROM lt").await, 0),
        vec![1, 2, 3],
        "the rows themselves must still land"
    );
}

#[tokio::test]
async fn an_update_trigger_fires_on_a_lake_table() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(
        &server,
        &mut session,
        "CREATE TABLE lu (id INT, v INT) USING ZYRONLAKE",
    )
    .await;
    exec(&server, &mut session, "CREATE TABLE audit (tid INT)").await;
    exec(
        &server,
        &mut session,
        "CREATE PROCEDURE log_row() AS 'INSERT INTO zyron_test.audit (tid) VALUES ($1)' LANGUAGE SQL",
    )
    .await;
    exec(
        &server,
        &mut session,
        "INSERT INTO lu (id, v) VALUES (1, 10), (2, 20)",
    )
    .await;
    exec(
        &server,
        &mut session,
        "CREATE TRIGGER lu_trg AFTER UPDATE ON lu FOR EACH ROW EXECUTE FUNCTION log_row",
    )
    .await;

    exec(&server, &mut session, "UPDATE lu SET v = 99 WHERE id = 2").await;

    let logged = sorted_col(
        &exec(&server, &mut session, "SELECT tid FROM audit").await,
        0,
    );
    assert_eq!(logged, vec![2], "the updated row did not fire its trigger");
}

// ---------------------------------------------------------------------------
// INSTEAD OF triggers on views
// ---------------------------------------------------------------------------

/// A base table, a view over it, a log table, and one procedure per event.
/// The procedures write to the log rather than the base table, so a test can
/// prove the trigger body ran and the view's underlying table was never
/// written directly.
async fn view_schema(server: &Arc<ServerState>, session: &mut Option<Session>) {
    exec(server, session, "CREATE TABLE base (id INT, val INT)").await;
    exec(
        server,
        session,
        "CREATE TABLE vlog (op INT, rid INT, rval INT)",
    )
    .await;
    exec(
        server,
        session,
        "INSERT INTO base (id, val) VALUES (1, 10), (2, 20)",
    )
    .await;
    exec(
        server,
        session,
        "CREATE VIEW bv AS SELECT id, val FROM base",
    )
    .await;
    exec(
        server,
        session,
        "CREATE PROCEDURE bv_ins() AS 'INSERT INTO zyron_test.vlog (op, rid, rval) VALUES (1, $1, $2)' LANGUAGE SQL",
    )
    .await;
    exec(
        server,
        session,
        "CREATE PROCEDURE bv_upd() AS 'INSERT INTO zyron_test.vlog (op, rid, rval) VALUES (2, $1, $4)' LANGUAGE SQL",
    )
    .await;
    exec(
        server,
        session,
        "CREATE PROCEDURE bv_del() AS 'INSERT INTO zyron_test.vlog (op, rid, rval) VALUES (3, $1, $2)' LANGUAGE SQL",
    )
    .await;
}

#[tokio::test]
async fn instead_of_insert_runs_trigger_body_not_direct_insert() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    view_schema(&server, &mut session).await;
    exec(
        &server,
        &mut session,
        "CREATE TRIGGER bv_t_ins INSTEAD OF INSERT ON bv FOR EACH ROW EXECUTE FUNCTION bv_ins",
    )
    .await;

    let counts = sorted_col(
        &exec(
            &server,
            &mut session,
            "INSERT INTO bv VALUES (7, 70), (8, 80)",
        )
        .await,
        0,
    );
    assert_eq!(counts, vec![2], "two view rows were written");

    // The trigger body logged each NEW row.
    let logged = sorted_col(
        &exec(&server, &mut session, "SELECT rid FROM vlog WHERE op = 1").await,
        0,
    );
    assert_eq!(logged, vec![7, 8]);

    // The base table was not written directly.
    let base_ids = sorted_col(&exec(&server, &mut session, "SELECT id FROM base").await, 0);
    assert_eq!(base_ids, vec![1, 2], "the base table must be untouched");
}

#[tokio::test]
async fn instead_of_insert_column_list_pads_null() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    view_schema(&server, &mut session).await;
    exec(
        &server,
        &mut session,
        "CREATE TRIGGER bv_t_ins INSTEAD OF INSERT ON bv FOR EACH ROW EXECUTE FUNCTION bv_ins",
    )
    .await;

    exec(&server, &mut session, "INSERT INTO bv (id) VALUES (9)").await;
    let null_count = sorted_col(
        &exec(
            &server,
            &mut session,
            "SELECT COUNT(*) FROM vlog WHERE rid = 9 AND rval IS NULL",
        )
        .await,
        0,
    );
    assert_eq!(null_count, vec![1], "the omitted column binds NULL");
}

#[tokio::test]
async fn instead_of_update_passes_old_then_new() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    view_schema(&server, &mut session).await;
    exec(
        &server,
        &mut session,
        "CREATE TRIGGER bv_t_upd INSTEAD OF UPDATE ON bv FOR EACH ROW EXECUTE FUNCTION bv_upd",
    )
    .await;

    exec(&server, &mut session, "UPDATE bv SET val = 99 WHERE id = 1").await;

    // The body logged OLD id ($1) and NEW val ($4).
    let old_id = sorted_col(
        &exec(&server, &mut session, "SELECT rid FROM vlog WHERE op = 2").await,
        0,
    );
    assert_eq!(old_id, vec![1]);
    let new_val = sorted_col(
        &exec(&server, &mut session, "SELECT rval FROM vlog WHERE op = 2").await,
        0,
    );
    assert_eq!(new_val, vec![99]);

    // The base row was not updated directly.
    let base_val = sorted_col(
        &exec(&server, &mut session, "SELECT val FROM base WHERE id = 1").await,
        0,
    );
    assert_eq!(base_val, vec![10], "the base table must be untouched");
}

#[tokio::test]
async fn instead_of_delete_passes_old_row() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    view_schema(&server, &mut session).await;
    exec(
        &server,
        &mut session,
        "CREATE TRIGGER bv_t_del INSTEAD OF DELETE ON bv FOR EACH ROW EXECUTE FUNCTION bv_del",
    )
    .await;

    exec(&server, &mut session, "DELETE FROM bv WHERE id = 2").await;

    let logged = sorted_col(
        &exec(&server, &mut session, "SELECT rval FROM vlog WHERE op = 3").await,
        0,
    );
    assert_eq!(logged, vec![20], "the body saw the OLD row");

    // The base row survives, the trigger body chose not to delete it.
    let base_ids = sorted_col(&exec(&server, &mut session, "SELECT id FROM base").await, 0);
    assert_eq!(base_ids, vec![1, 2]);
}

#[tokio::test]
async fn instead_of_insert_from_select_maps_columns() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    view_schema(&server, &mut session).await;
    exec(
        &server,
        &mut session,
        "CREATE TRIGGER bv_t_ins INSTEAD OF INSERT ON bv FOR EACH ROW EXECUTE FUNCTION bv_ins",
    )
    .await;

    exec(
        &server,
        &mut session,
        "INSERT INTO bv SELECT id, val FROM base",
    )
    .await;
    let logged = sorted_col(
        &exec(&server, &mut session, "SELECT rid FROM vlog WHERE op = 1").await,
        0,
    );
    assert_eq!(logged, vec![1, 2]);
}

#[tokio::test]
async fn view_write_without_trigger_names_the_missing_form() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    view_schema(&server, &mut session).await;

    let err = try_exec(&server, &mut session, "INSERT INTO bv VALUES (5, 50)")
        .await
        .expect_err("a view without an INSTEAD OF trigger is not writable");
    assert!(
        err.contains("INSTEAD OF INSERT"),
        "error names the missing trigger form: {err}"
    );
    assert!(
        !err.contains("not supported"),
        "no blanket unsupported message: {err}"
    );
}

#[tokio::test]
async fn instead_of_requires_row_granularity() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    view_schema(&server, &mut session).await;

    let err = try_exec(
        &server,
        &mut session,
        "CREATE TRIGGER bv_t INSTEAD OF INSERT ON bv FOR EACH STATEMENT EXECUTE FUNCTION bv_ins",
    )
    .await
    .expect_err("statement granularity is rejected");
    assert!(err.contains("FOR EACH ROW"), "{err}");
}

#[tokio::test]
async fn instead_of_requires_a_view_target() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    view_schema(&server, &mut session).await;

    let err = try_exec(
        &server,
        &mut session,
        "CREATE TRIGGER b_t INSTEAD OF INSERT ON base FOR EACH ROW EXECUTE FUNCTION bv_ins",
    )
    .await
    .expect_err("a table target is rejected");
    assert!(err.contains("require a view"), "{err}");
}

#[tokio::test]
async fn before_trigger_on_a_view_points_to_instead_of() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    view_schema(&server, &mut session).await;

    let err = try_exec(
        &server,
        &mut session,
        "CREATE TRIGGER bv_t BEFORE INSERT ON bv FOR EACH ROW EXECUTE FUNCTION bv_ins",
    )
    .await
    .expect_err("BEFORE on a view is rejected");
    assert!(err.contains("INSTEAD OF"), "{err}");
}

#[tokio::test]
async fn second_instead_of_trigger_for_same_event_is_rejected() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    view_schema(&server, &mut session).await;
    exec(
        &server,
        &mut session,
        "CREATE TRIGGER bv_t1 INSTEAD OF INSERT ON bv FOR EACH ROW EXECUTE FUNCTION bv_ins",
    )
    .await;

    let err = try_exec(
        &server,
        &mut session,
        "CREATE TRIGGER bv_t2 INSTEAD OF INSERT ON bv FOR EACH ROW EXECUTE FUNCTION bv_ins",
    )
    .await
    .expect_err("one INSTEAD OF trigger per event");
    assert!(err.contains("already has"), "{err}");
}

#[tokio::test]
async fn drop_instead_of_trigger_stops_view_writes() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    view_schema(&server, &mut session).await;
    exec(
        &server,
        &mut session,
        "CREATE TRIGGER bv_t_ins INSTEAD OF INSERT ON bv FOR EACH ROW EXECUTE FUNCTION bv_ins",
    )
    .await;
    exec(&server, &mut session, "INSERT INTO bv VALUES (7, 70)").await;

    exec(&server, &mut session, "DROP TRIGGER bv_t_ins ON bv").await;
    let err = try_exec(&server, &mut session, "INSERT INTO bv VALUES (8, 80)")
        .await
        .expect_err("the dropped trigger no longer routes the write");
    assert!(err.contains("INSTEAD OF INSERT"), "{err}");
}

#[tokio::test]
async fn instead_of_write_marks_the_firing_context_as_writing() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    view_schema(&server, &mut session).await;
    exec(
        &server,
        &mut session,
        "CREATE TRIGGER bv_t_ins INSTEAD OF INSERT ON bv FOR EACH ROW EXECUTE FUNCTION bv_ins",
    )
    .await;

    // Execute the view write directly so the firing statement's context is
    // observable. Its own operator writes nothing, only the trigger body
    // does, and the wire layer decides durable-commit vs read-only-commit
    // (and whether to propose to the replication group) from this flag.
    let stmt = zyron_parser::parse("INSERT INTO bv VALUES (7, 70)")
        .expect("parse")
        .into_iter()
        .next()
        .expect("one statement");
    let plan = zyron_planner::plan(
        &server.catalog,
        DatabaseId(1),
        vec!["zyron_test".into()],
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
    let txn_id = txn.txn_id;
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
    zyron_executor::execute(plan, &ctx)
        .await
        .expect("view write executes");

    assert!(
        ctx.wrote_wal(),
        "a trigger body's WAL writes must mark the firing statement's context"
    );
    server.txn_manager.commit(&mut txn).await.expect("commit");
}

#[tokio::test]
async fn trigger_fires_its_canonical_procedure_despite_a_decoy() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    base_schema(&server, &mut session).await;
    exec(
        &server,
        &mut session,
        "CREATE TRIGGER trg AFTER INSERT ON t FOR EACH ROW EXECUTE FUNCTION zyron_test.log_row",
    )
    .await;

    // A same-named procedure appears in another schema before the trigger
    // ever fires. The definition stored a canonical reference, so the decoy
    // must never run.
    exec(&server, &mut session, "CREATE SCHEMA decoy").await;
    if let Some(s) = session.as_mut() {
        s.search_path = vec!["decoy".into()];
    }
    exec(
        &server,
        &mut session,
        "CREATE PROCEDURE log_row() AS 'INSERT INTO zyron_test.audit (tid) VALUES (999)' LANGUAGE SQL",
    )
    .await;
    if let Some(s) = session.as_mut() {
        s.search_path = vec!["zyron_test".into()];
    }

    exec(
        &server,
        &mut session,
        "INSERT INTO t (id, v) VALUES (7, 70)",
    )
    .await;
    let logged = sorted_col(
        &exec(&server, &mut session, "SELECT tid FROM audit").await,
        0,
    );
    assert_eq!(
        logged,
        vec![7],
        "the canonical procedure fired, not the decoy"
    );
}

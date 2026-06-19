//! Integration tests for user-defined aggregates.
//!
//! Defines aggregates whose state and final functions are SQL scalar UDFs,
//! then exercises them through the plan + execute pipeline: a global aggregate,
//! one with a final function, a grouped aggregate, and DROP AGGREGATE. Verifies
//! the executor folds each group through the bound state function.
//!
//! Run: cargo test -p zyron-wire --test aggregate_test -- --nocapture

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

/// Reads the first column of the first row as i64.
fn scalar_i64(batches: &[DataBatch]) -> i64 {
    for b in batches {
        if b.num_rows > 0 {
            return match b.columns[0].data.get_scalar(0) {
                ScalarValue::Int64(v) => v,
                ScalarValue::Int32(v) => v as i64,
                other => panic!("expected integer, got {other:?}"),
            };
        }
    }
    panic!("no rows");
}

/// Collects column `idx` across all batches as i64.
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

async fn seed_table(server: &Arc<ServerState>, session: &mut Option<Session>) {
    exec(server, session, "CREATE TABLE t (g INT, x INT)").await;
    exec(
        server,
        session,
        "INSERT INTO t (g, x) VALUES (1, 10), (1, 20), (2, 5), (2, 7), (2, 3)",
    )
    .await;
}

async fn define_sum_sfunc(server: &Arc<ServerState>, session: &mut Option<Session>) {
    exec(
        server,
        session,
        "CREATE FUNCTION agg_add(acc INT, val INT) RETURNS INT AS 'acc + val 'LANGUAGE SQL",
    )
    .await;
}

#[tokio::test]
async fn global_uda_matches_sum() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed_table(&server, &mut session).await;
    define_sum_sfunc(&server, &mut session).await;
    exec(
        &server,
        &mut session,
        "CREATE AGGREGATE mysum(val INT) (SFUNC = agg_add, STYPE = INT, INITCOND = '0')",
    )
    .await;

    let total = scalar_i64(&exec(&server, &mut session, "SELECT mysum(x) FROM t").await);
    // 10 + 20 + 5 + 7 + 3 = 45
    assert_eq!(total, 45);
}

#[tokio::test]
async fn uda_with_final_function() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed_table(&server, &mut session).await;
    define_sum_sfunc(&server, &mut session).await;
    exec(
        &server,
        &mut session,
        "CREATE FUNCTION agg_double(acc INT) RETURNS INT AS 'acc * 2 'LANGUAGE SQL",
    )
    .await;
    exec(
        &server,
        &mut session,
        "CREATE AGGREGATE sumdbl(val INT) (SFUNC = agg_add, STYPE = INT, FINALFUNC = agg_double, INITCOND = '0')",
    )
    .await;

    let doubled = scalar_i64(&exec(&server, &mut session, "SELECT sumdbl(x) FROM t").await);
    // 2 * 45 = 90
    assert_eq!(doubled, 90);
}

#[tokio::test]
async fn grouped_uda_folds_per_group() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed_table(&server, &mut session).await;
    define_sum_sfunc(&server, &mut session).await;
    exec(
        &server,
        &mut session,
        "CREATE AGGREGATE mysum(val INT) (SFUNC = agg_add, STYPE = INT, INITCOND = '0')",
    )
    .await;

    let rows = exec(
        &server,
        &mut session,
        "SELECT g, mysum(x) FROM t GROUP BY g ORDER BY g",
    )
    .await;
    let groups = col_i64(&rows, 0);
    let sums = col_i64(&rows, 1);
    assert_eq!(groups, vec![1, 2]);
    // group 1: 10 + 20 = 30; group 2: 5 + 7 + 3 = 15
    assert_eq!(sums, vec![30, 15]);
}

#[tokio::test]
async fn drop_aggregate_makes_it_unresolvable() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed_table(&server, &mut session).await;
    define_sum_sfunc(&server, &mut session).await;
    exec(
        &server,
        &mut session,
        "CREATE AGGREGATE mysum(val INT) (SFUNC = agg_add, STYPE = INT, INITCOND = '0')",
    )
    .await;
    // It resolves before the drop.
    assert_eq!(
        scalar_i64(&exec(&server, &mut session, "SELECT mysum(x) FROM t").await),
        45
    );
    exec(&server, &mut session, "DROP AGGREGATE mysum").await;
    // After the drop the name no longer resolves to an aggregate.
    let err = try_exec(&server, &mut session, "SELECT mysum(x) FROM t")
        .await
        .expect_err("dropped aggregate should not resolve");
    assert!(
        err.to_lowercase().contains("mysum")
            || err.to_lowercase().contains("aggregate")
            || err.to_lowercase().contains("function"),
        "unexpected error: {err}"
    );
}

#[tokio::test]
async fn create_aggregate_requires_existing_sfunc() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed_table(&server, &mut session).await;
    // No state function defined; CREATE AGGREGATE must reject it.
    let err = try_exec(
        &server,
        &mut session,
        "CREATE AGGREGATE mysum(val INT) (SFUNC = nope, STYPE = INT, INITCOND = '0')",
    )
    .await
    .expect_err("missing sfunc should be rejected");
    assert!(
        err.to_lowercase().contains("state function") || err.to_lowercase().contains("nope"),
        "unexpected error: {err}"
    );
}

//! Integration tests for LATERAL joins.
//!
//! A LATERAL subquery may reference columns from the preceding FROM items, so
//! it is executed once per left row. These tests exercise CROSS JOIN LATERAL,
//! comma LATERAL, LEFT JOIN LATERAL (NULL extension), and a LATERAL subquery
//! that aggregates per outer row.
//!
//! Run: cargo test -p zyron-wire --test lateral_test -- --nocapture

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

/// Reads column `idx` across all batches as i64, mapping NULL to i64::MIN.
fn col_i64(batches: &[DataBatch], idx: usize) -> Vec<i64> {
    let mut out = Vec::new();
    for b in batches {
        if let Some(col) = b.columns.get(idx) {
            for r in 0..b.num_rows {
                match col.get_scalar(r) {
                    ScalarValue::Int64(v) => out.push(v),
                    ScalarValue::Int32(v) => out.push(v as i64),
                    ScalarValue::Null => out.push(i64::MIN),
                    other => panic!("expected integer, got {other:?}"),
                }
            }
        }
    }
    out
}

/// Returns (col0, col1) pairs across all batches, sorted ascending.
fn sorted_pairs(batches: &[DataBatch]) -> Vec<(i64, i64)> {
    let a = col_i64(batches, 0);
    let b = col_i64(batches, 1);
    let mut pairs: Vec<(i64, i64)> = a.into_iter().zip(b).collect();
    pairs.sort_unstable();
    pairs
}

async fn seed(server: &Arc<ServerState>, session: &mut Option<Session>) {
    exec(server, session, "CREATE TABLE t1 (id INT, g INT, x INT)").await;
    exec(server, session, "CREATE TABLE t2 (fk INT, g INT, y INT)").await;
    exec(
        server,
        session,
        "INSERT INTO t1 (id, g, x) VALUES (1, 10, 100), (2, 10, 200), (3, 20, 5), (4, 30, 9)",
    )
    .await;
    exec(
        server,
        session,
        "INSERT INTO t2 (fk, g, y) VALUES (1, 10, 50), (1, 10, 60), (2, 20, 70), (3, 20, 5)",
    )
    .await;
}

#[tokio::test]
async fn cross_join_lateral_emits_per_match() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    // One output row per matching t2 row; id=4 has no match and is dropped.
    let rows = exec(
        &server,
        &mut session,
        "SELECT t1.id, sub.y FROM t1 CROSS JOIN LATERAL (SELECT y FROM t2 WHERE t2.fk = t1.id) sub",
    )
    .await;
    assert_eq!(sorted_pairs(&rows), vec![(1, 50), (1, 60), (2, 70), (3, 5)]);
}

#[tokio::test]
async fn comma_lateral_emits_per_match() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    let rows = exec(
        &server,
        &mut session,
        "SELECT t1.id, sub.y FROM t1, LATERAL (SELECT y FROM t2 WHERE t2.fk = t1.id) sub",
    )
    .await;
    assert_eq!(sorted_pairs(&rows), vec![(1, 50), (1, 60), (2, 70), (3, 5)]);
}

#[tokio::test]
async fn left_join_lateral_null_extends() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    // id=4 has no matching t2 row, so a LEFT JOIN LATERAL keeps it with NULL y
    // (mapped to i64::MIN by the reader).
    let rows = exec(
        &server,
        &mut session,
        "SELECT t1.id, sub.y FROM t1 LEFT JOIN LATERAL (SELECT y FROM t2 WHERE t2.fk = t1.id) sub ON true",
    )
    .await;
    assert_eq!(
        sorted_pairs(&rows),
        vec![(1, 50), (1, 60), (2, 70), (3, 5), (4, i64::MIN)]
    );
}

#[tokio::test]
async fn lateral_aggregate_per_outer_row() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    // The aggregate subquery returns one row per outer row, so every t1 row
    // appears with its match count, including id=4 with zero.
    let rows = exec(
        &server,
        &mut session,
        "SELECT t1.id, sub.c FROM t1 CROSS JOIN LATERAL (SELECT count(*) AS c FROM t2 WHERE t2.fk = t1.id) sub",
    )
    .await;
    assert_eq!(sorted_pairs(&rows), vec![(1, 2), (2, 1), (3, 1), (4, 0)]);
}

#[tokio::test]
async fn lateral_table_function_with_outer_ref_is_rejected() {
    let (server, _s, _t) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    // A table function consumes a whole graph or table, so a LATERAL one that
    // references an outer column has no valid per-row execution and must be
    // rejected with a clear error rather than produce a wrong result or panic.
    let r = try_exec(
        &server,
        &mut session,
        "SELECT t1.id FROM t1, LATERAL pagerank(t1.id) AS p",
    )
    .await;
    assert!(r.is_err(), "correlated LATERAL table function must error");
}

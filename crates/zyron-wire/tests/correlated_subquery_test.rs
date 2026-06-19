//! Integration tests for correlated subqueries.
//!
//! A correlated subquery references columns from the enclosing query, so it
//! runs once per outer row. These tests exercise correlated scalar subqueries
//! in WHERE and in the projection list, correlated EXISTS and NOT EXISTS,
//! correlated IN, and confirm uncorrelated subqueries still fold once.
//!
//! Run: cargo test -p zyron-wire --test correlated_subquery_test -- --nocapture

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

/// Reads column `idx` across all batches as i64.
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

/// Reads column `idx` sorted ascending.
fn sorted_col(batches: &[DataBatch], idx: usize) -> Vec<i64> {
    let mut v = col_i64(batches, idx);
    v.sort_unstable();
    v
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
async fn correlated_scalar_subquery_in_where() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    // For each t1 row, compare x to max(y) of the matching t2 rows. Only id=3
    // (x=5, max y where fk=3 is 5) matches.
    let rows = exec(
        &server,
        &mut session,
        "SELECT id FROM t1 WHERE x = (SELECT max(y) FROM t2 WHERE t2.fk = t1.id)",
    )
    .await;
    assert_eq!(sorted_col(&rows, 0), vec![3]);
}

#[tokio::test]
async fn correlated_exists_in_where() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    // t2 has fk values 1,2,3 but not 4.
    let rows = exec(
        &server,
        &mut session,
        "SELECT id FROM t1 WHERE EXISTS (SELECT 1 FROM t2 WHERE t2.fk = t1.id)",
    )
    .await;
    assert_eq!(sorted_col(&rows, 0), vec![1, 2, 3]);
}

#[tokio::test]
async fn correlated_not_exists_in_where() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    let rows = exec(
        &server,
        &mut session,
        "SELECT id FROM t1 WHERE NOT EXISTS (SELECT 1 FROM t2 WHERE t2.fk = t1.id)",
    )
    .await;
    assert_eq!(sorted_col(&rows, 0), vec![4]);
}

#[tokio::test]
async fn correlated_in_subquery_in_where() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    // id IN (fk of t2 rows whose y >= this row's x). Only id=3 (x=5) qualifies:
    // all t2 rows have y>=5 so fk set is {1,2,3} and 3 is a member.
    let rows = exec(
        &server,
        &mut session,
        "SELECT id FROM t1 WHERE id IN (SELECT fk FROM t2 WHERE t2.y >= t1.x)",
    )
    .await;
    assert_eq!(sorted_col(&rows, 0), vec![3]);
}

#[tokio::test]
async fn correlated_scalar_subquery_in_projection() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    // Per-row count of matching t2 rows: id1->2, id2->1, id3->1, id4->0.
    let rows = exec(
        &server,
        &mut session,
        "SELECT id, (SELECT count(*) FROM t2 WHERE t2.fk = t1.id) AS c FROM t1",
    )
    .await;
    let ids = col_i64(&rows, 0);
    let counts = col_i64(&rows, 1);
    let mut pairs: Vec<(i64, i64)> = ids.into_iter().zip(counts).collect();
    pairs.sort_unstable();
    assert_eq!(pairs, vec![(1, 2), (2, 1), (3, 1), (4, 0)]);
}

#[tokio::test]
async fn uncorrelated_subquery_in_inner_join_on() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    // t2 rows with y>40 have fk in {1,1,2}; joined to t1 by id=fk gives ids
    // 1 (twice) and 2.
    let rows = exec(
        &server,
        &mut session,
        "SELECT t1.id FROM t1 JOIN t2 ON t1.id = t2.fk AND t2.y > (SELECT 40)",
    )
    .await;
    assert_eq!(sorted_col(&rows, 0), vec![1, 1, 2]);
}

#[tokio::test]
async fn correlated_subquery_in_inner_join_on() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    // Keep only the join row whose t2.y equals the max y of t2 rows matching the
    // outer t1.id: id1->60, id2->70, id3->5.
    let rows = exec(
        &server,
        &mut session,
        "SELECT t1.id, t2.y FROM t1 JOIN t2 ON t1.id = t2.fk AND t2.y = (SELECT max(y) FROM t2 t3 WHERE t3.fk = t1.id)",
    )
    .await;
    let ids = col_i64(&rows, 0);
    let ys = col_i64(&rows, 1);
    let mut pairs: Vec<(i64, i64)> = ids.into_iter().zip(ys).collect();
    pairs.sort_unstable();
    assert_eq!(pairs, vec![(1, 60), (2, 70), (3, 5)]);
}

#[tokio::test]
async fn correlated_subquery_in_having_group_key() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    // Per group g, keep groups whose t2 row count exceeds the count of t1 rows
    // with the same g. g=10: 2 vs 2 (excluded); g=20: 2 vs 1 (kept).
    let rows = exec(
        &server,
        &mut session,
        "SELECT t2.g, count(*) FROM t2 GROUP BY t2.g HAVING count(*) > (SELECT count(*) FROM t1 WHERE t1.g = t2.g)",
    )
    .await;
    let gs = col_i64(&rows, 0);
    let cs = col_i64(&rows, 1);
    let mut pairs: Vec<(i64, i64)> = gs.into_iter().zip(cs).collect();
    pairs.sort_unstable();
    assert_eq!(pairs, vec![(20, 2)]);
}

#[tokio::test]
async fn uncorrelated_subquery_in_left_join_on() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    // LEFT JOIN keeps every t1 row; matches need id=fk and y>40. id3 (only y=5)
    // and id4 (no t2) get NULL (i64::MIN).
    let rows = exec(
        &server,
        &mut session,
        "SELECT t1.id, t2.y FROM t1 LEFT JOIN t2 ON t1.id = t2.fk AND t2.y > (SELECT 40)",
    )
    .await;
    let ids = col_i64(&rows, 0);
    let ys = col_i64(&rows, 1);
    let mut pairs: Vec<(i64, i64)> = ids.into_iter().zip(ys).collect();
    pairs.sort_unstable();
    assert_eq!(
        pairs,
        vec![(1, 50), (1, 60), (2, 70), (3, i64::MIN), (4, i64::MIN)]
    );
}

#[tokio::test]
async fn correlated_subquery_in_left_join_on() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    // Match the t2 row whose y is the max for the outer t1.id; t1 rows with no
    // match are NULL-extended (id4).
    let rows = exec(
        &server,
        &mut session,
        "SELECT t1.id, t2.y FROM t1 LEFT JOIN t2 ON t1.id = t2.fk AND t2.y = (SELECT max(y) FROM t2 t3 WHERE t3.fk = t1.id)",
    )
    .await;
    let ids = col_i64(&rows, 0);
    let ys = col_i64(&rows, 1);
    let mut pairs: Vec<(i64, i64)> = ids.into_iter().zip(ys).collect();
    pairs.sort_unstable();
    assert_eq!(pairs, vec![(1, 60), (2, 70), (3, 5), (4, i64::MIN)]);
}

#[tokio::test]
async fn uncorrelated_in_subquery_still_works() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    seed(&server, &mut session).await;
    // No correlation: folded once. fk set is {1,2,3}.
    let rows = exec(
        &server,
        &mut session,
        "SELECT id FROM t1 WHERE id IN (SELECT fk FROM t2)",
    )
    .await;
    assert_eq!(sorted_col(&rows, 0), vec![1, 2, 3]);
}

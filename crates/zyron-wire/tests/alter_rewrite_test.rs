//! Integration tests for the ALTER TABLE column-rewrite engine.
//!
//! Exercises ADD COLUMN, DROP COLUMN, and ALTER COLUMN TYPE through the same
//! DDL dispatch path the server uses, then reads rows back to confirm the heap
//! and indexes were rebuilt under the new schema.
//!
//! Run: cargo test -p zyron-wire --test alter_rewrite_test -- --nocapture

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

/// Creates a full ServerState backed by temp directories. The default database
/// (id 1) is created by catalog bootstrap; this adds a `public` user schema and
/// returns its id, matching the production layout the rewrite helpers assume.
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

/// Runs one SQL statement. DDL goes through the dispatch path; DML and SELECT
/// go through plan + execute under a committed transaction. Returns result
/// batches (empty for DDL).
async fn exec(
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
    sql: &str,
) -> Vec<DataBatch> {
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
        res.expect("ddl handler failed");
        return Vec::new();
    }

    let plan = zyron_planner::plan(&server.catalog, DatabaseId(1), vec!["public".into()], stmt)
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
    if let Some(gm) = &server.graph_manager {
        ctx.set_graph_manager(Arc::clone(gm));
    }
    let ctx = Arc::new(ctx);
    let batches = zyron_executor::execute(plan, &ctx).await.expect("execute");
    server.txn_manager.commit(&mut txn).await.expect("commit");
    batches
}

/// Like `exec` but returns the execution result instead of panicking, so a
/// test can assert that a statement fails (e.g. a foreign-key violation).
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

/// Reads every scalar of one column across all batches, in row order.
fn column_values(batches: &[DataBatch], col_idx: usize) -> Vec<ScalarValue> {
    let mut out = Vec::new();
    for b in batches {
        let col = &b.columns[col_idx];
        for r in 0..b.num_rows {
            out.push(col.get_scalar(r));
        }
    }
    out
}

fn total_rows(batches: &[DataBatch]) -> usize {
    batches.iter().map(|b| b.num_rows).sum()
}

#[tokio::test]
async fn add_column_nullable_backfills_null() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec(&server, &mut session, "CREATE TABLE t (id INT, name TEXT)").await;
    exec(
        &server,
        &mut session,
        "INSERT INTO t VALUES (1, 'a'), (2, 'b')",
    )
    .await;

    exec(&server, &mut session, "ALTER TABLE t ADD COLUMN score INT").await;

    let rows = exec(&server, &mut session, "SELECT id, name, score FROM t").await;
    assert_eq!(total_rows(&rows), 2, "both rows survive the rewrite");
    let scores = column_values(&rows, 2);
    assert!(
        scores.iter().all(|s| s.is_null()),
        "added column is NULL for old rows: {scores:?}"
    );
    let ids = column_values(&rows, 0);
    assert_eq!(ids, vec![ScalarValue::Int32(1), ScalarValue::Int32(2)]);
}

#[tokio::test]
async fn add_column_with_default_backfills_value() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec(&server, &mut session, "CREATE TABLE t (id INT)").await;
    exec(&server, &mut session, "INSERT INTO t VALUES (1), (2), (3)").await;

    exec(
        &server,
        &mut session,
        "ALTER TABLE t ADD COLUMN status INT DEFAULT 7",
    )
    .await;

    let rows = exec(&server, &mut session, "SELECT id, status FROM t").await;
    assert_eq!(total_rows(&rows), 3);
    let status = column_values(&rows, 1);
    assert_eq!(
        status,
        vec![
            ScalarValue::Int32(7),
            ScalarValue::Int32(7),
            ScalarValue::Int32(7)
        ],
        "default backfilled into existing rows"
    );

    // New inserts under the new schema still work.
    exec(&server, &mut session, "INSERT INTO t VALUES (4, 9)").await;
    let rows = exec(&server, &mut session, "SELECT id, status FROM t").await;
    assert_eq!(total_rows(&rows), 4);
}

#[tokio::test]
async fn drop_column_removes_it_and_keeps_others() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec(
        &server,
        &mut session,
        "CREATE TABLE t (id INT, name TEXT, age INT)",
    )
    .await;
    exec(
        &server,
        &mut session,
        "INSERT INTO t VALUES (1, 'a', 30), (2, 'b', 40)",
    )
    .await;

    exec(&server, &mut session, "ALTER TABLE t DROP COLUMN name").await;

    let rows = exec(&server, &mut session, "SELECT id, age FROM t").await;
    assert_eq!(total_rows(&rows), 2, "rows survive the drop");
    let ids = column_values(&rows, 0);
    assert_eq!(ids, vec![ScalarValue::Int32(1), ScalarValue::Int32(2)]);
    let ages = column_values(&rows, 1);
    assert_eq!(ages, vec![ScalarValue::Int32(30), ScalarValue::Int32(40)]);

    // The dropped column is gone from the catalog: selecting it must fail.
    let stmt = zyron_parser::parse("SELECT name FROM t")
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
    let planned =
        zyron_planner::plan(&server.catalog, DatabaseId(1), vec!["public".into()], stmt).await;
    assert!(planned.is_err(), "dropped column should no longer resolve");
}

#[tokio::test]
async fn drop_column_index_is_removed_other_index_survives() {
    let (server, schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec(
        &server,
        &mut session,
        "CREATE TABLE t (id INT, name TEXT, age INT)",
    )
    .await;
    exec(&server, &mut session, "CREATE INDEX idx_name ON t (name)").await;
    exec(&server, &mut session, "CREATE INDEX idx_age ON t (age)").await;
    exec(
        &server,
        &mut session,
        "INSERT INTO t VALUES (1, 'a', 30), (2, 'b', 40)",
    )
    .await;

    let table = server.catalog.get_table(schema_id, "t").unwrap();
    exec(&server, &mut session, "ALTER TABLE t DROP COLUMN name").await;

    let table_after = server.catalog.get_table(schema_id, "t").unwrap();
    let names: Vec<String> = server
        .catalog
        .get_indexes_for_table(table_after.id)
        .iter()
        .map(|i| i.name.clone())
        .collect();
    assert!(
        !names.contains(&"idx_name".to_string()),
        "index on dropped column removed: {names:?}"
    );
    assert!(
        names.contains(&"idx_age".to_string()),
        "index on surviving column kept: {names:?}"
    );

    // The surviving index points at the new heap, so an indexed lookup returns
    // the rebuilt rows.
    let rows = exec(
        &server,
        &mut session,
        "SELECT id, age FROM t WHERE age = 40",
    )
    .await;
    assert_eq!(total_rows(&rows), 1);
    assert_eq!(column_values(&rows, 0), vec![ScalarValue::Int32(2)]);
    let _ = table; // table id is stable across the rewrite
    assert_eq!(
        table.id, table_after.id,
        "table id preserved across rewrite"
    );
}

#[tokio::test]
async fn alter_type_widens_values() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec(&server, &mut session, "CREATE TABLE t (id INT, amount INT)").await;
    exec(
        &server,
        &mut session,
        "INSERT INTO t VALUES (1, 100), (2, 200)",
    )
    .await;

    exec(
        &server,
        &mut session,
        "ALTER TABLE t ALTER COLUMN amount TYPE BIGINT",
    )
    .await;

    let rows = exec(&server, &mut session, "SELECT id, amount FROM t").await;
    assert_eq!(total_rows(&rows), 2);
    let amounts = column_values(&rows, 1);
    assert_eq!(
        amounts,
        vec![ScalarValue::Int64(100), ScalarValue::Int64(200)],
        "values widened to i64"
    );
}

#[tokio::test]
async fn add_not_null_without_default_is_rejected() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec(&server, &mut session, "CREATE TABLE t (id INT)").await;
    exec(&server, &mut session, "INSERT INTO t VALUES (1)").await;

    let stmt = zyron_parser::parse("ALTER TABLE t ADD COLUMN req INT NOT NULL")
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
    let mut txn_opt: Option<zyron_storage::txn::Transaction> = None;
    let mut active_branch: Option<String> = None;
    let res = zyron_wire::ddl_dispatch::try_handle_ddl_utility(
        &stmt,
        &server,
        &mut session,
        &mut txn_opt,
        &mut active_branch,
        "ALTER TABLE t ADD COLUMN req INT NOT NULL",
    )
    .await
    .expect("dispatch returns a result");
    assert!(
        res.is_err(),
        "ADD COLUMN NOT NULL without a default must be rejected"
    );
}

// ---------------------------------------------------------------------------
// B+tree index maintenance on UPDATE and DELETE
// ---------------------------------------------------------------------------

#[tokio::test]
async fn update_nonkey_column_keeps_row_index_reachable() {
    let (server, _s, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE t (id INT, v INT)").await;
    exec(&server, &mut session, "CREATE INDEX idx_id ON t (id)").await;
    exec(
        &server,
        &mut session,
        "INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)",
    )
    .await;

    // Update a non-indexed column; the indexed key (id) is unchanged. The row
    // must remain reachable by an index lookup on that key.
    exec(&server, &mut session, "UPDATE t SET v = 999 WHERE id = 2").await;

    let rows = exec(&server, &mut session, "SELECT id, v FROM t WHERE id = 2").await;
    assert_eq!(total_rows(&rows), 1, "updated row still reachable by index");
    assert_eq!(column_values(&rows, 1), vec![ScalarValue::Int32(999)]);
}

#[tokio::test]
async fn update_key_column_moves_index_entry() {
    let (server, _s, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE t (id INT, v INT)").await;
    exec(&server, &mut session, "CREATE INDEX idx_id ON t (id)").await;
    exec(
        &server,
        &mut session,
        "INSERT INTO t VALUES (1, 10), (2, 20)",
    )
    .await;

    // Change the indexed value: the old key must stop matching and the new key
    // must start matching.
    exec(&server, &mut session, "UPDATE t SET id = 99 WHERE id = 2").await;

    assert_eq!(
        total_rows(&exec(&server, &mut session, "SELECT id, v FROM t WHERE id = 2").await),
        0,
        "old index key no longer matches"
    );
    let moved = exec(&server, &mut session, "SELECT id, v FROM t WHERE id = 99").await;
    assert_eq!(total_rows(&moved), 1, "new index key matches");
    assert_eq!(column_values(&moved, 1), vec![ScalarValue::Int32(20)]);
}

#[tokio::test]
async fn delete_removes_index_entry_no_phantom_after_reuse() {
    let (server, _s, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE t (id INT, v INT)").await;
    exec(&server, &mut session, "CREATE INDEX idx_id ON t (id)").await;
    exec(
        &server,
        &mut session,
        "INSERT INTO t VALUES (1, 10), (2, 20)",
    )
    .await;

    exec(&server, &mut session, "DELETE FROM t WHERE id = 2").await;
    assert_eq!(
        total_rows(&exec(&server, &mut session, "SELECT id, v FROM t WHERE id = 2").await),
        0,
        "deleted row's index entry is gone"
    );

    // A new row may reuse the freed heap slot. The stale entry for the old key
    // must not surface it: looking up the old key finds nothing, the new key
    // finds the new row.
    exec(&server, &mut session, "INSERT INTO t VALUES (5, 555)").await;
    assert_eq!(
        total_rows(&exec(&server, &mut session, "SELECT id, v FROM t WHERE id = 2").await),
        0,
        "old key does not surface the slot-reusing row"
    );
    let five = exec(&server, &mut session, "SELECT id, v FROM t WHERE id = 5").await;
    assert_eq!(total_rows(&five), 1);
    assert_eq!(column_values(&five, 1), vec![ScalarValue::Int32(555)]);
}

/// Collects (id, v) Int32 pairs from a two-column result, sorted by id.
fn id_v_pairs(batches: &[DataBatch]) -> Vec<(i32, i32)> {
    let mut out = Vec::new();
    for b in batches {
        for r in 0..b.num_rows {
            let id = match b.columns[0].get_scalar(r) {
                ScalarValue::Int32(v) => v,
                other => panic!("unexpected id scalar {other:?}"),
            };
            let v = match b.columns[1].get_scalar(r) {
                ScalarValue::Int32(v) => v,
                other => panic!("unexpected v scalar {other:?}"),
            };
            out.push((id, v));
        }
    }
    out.sort();
    out
}

/// Collects the integer values of one column across all batches, sorted.
fn sorted_ints(batches: &[DataBatch], col: usize) -> Vec<i32> {
    let mut v: Vec<i32> = column_values(batches, col)
        .into_iter()
        .filter_map(|s| match s {
            ScalarValue::Int32(x) => Some(x),
            _ => None,
        })
        .collect();
    v.sort();
    v
}

#[tokio::test]
async fn non_unique_index_returns_all_duplicate_value_rows() {
    let (server, _s, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE t (id INT, cat INT)").await;
    exec(&server, &mut session, "CREATE INDEX idx_cat ON t (cat)").await;
    exec(
        &server,
        &mut session,
        "INSERT INTO t VALUES (1, 10), (2, 10), (3, 20), (4, 10)",
    )
    .await;

    // Three rows share cat = 10; a non-unique index must return all of them.
    let rows = exec(
        &server,
        &mut session,
        "SELECT id, cat FROM t WHERE cat = 10",
    )
    .await;
    assert_eq!(
        sorted_ints(&rows, 0),
        vec![1, 2, 4],
        "all rows with the duplicate value"
    );

    let twenty = exec(
        &server,
        &mut session,
        "SELECT id, cat FROM t WHERE cat = 20",
    )
    .await;
    assert_eq!(sorted_ints(&twenty, 0), vec![3]);

    // Range scan spanning duplicate values returns every match.
    let ge = exec(
        &server,
        &mut session,
        "SELECT id, cat FROM t WHERE cat >= 10",
    )
    .await;
    assert_eq!(sorted_ints(&ge, 0), vec![1, 2, 3, 4]);
}

#[tokio::test]
async fn non_unique_index_delete_one_duplicate_keeps_others() {
    let (server, _s, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE t (id INT, cat INT)").await;
    exec(&server, &mut session, "CREATE INDEX idx_cat ON t (cat)").await;
    exec(
        &server,
        &mut session,
        "INSERT INTO t VALUES (1, 10), (2, 10), (3, 10)",
    )
    .await;

    exec(&server, &mut session, "DELETE FROM t WHERE id = 2").await;

    // Deleting one row sharing the value must leave the others reachable: the
    // composite (value, tid) key removes only the deleted row's entry.
    let rows = exec(
        &server,
        &mut session,
        "SELECT id, cat FROM t WHERE cat = 10",
    )
    .await;
    assert_eq!(
        sorted_ints(&rows, 0),
        vec![1, 3],
        "siblings of the deleted row remain indexed"
    );
}

#[tokio::test]
async fn non_unique_index_update_one_duplicate_repoints() {
    let (server, _s, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE t (id INT, cat INT)").await;
    exec(&server, &mut session, "CREATE INDEX idx_cat ON t (cat)").await;
    exec(
        &server,
        &mut session,
        "INSERT INTO t VALUES (1, 10), (2, 10), (3, 10)",
    )
    .await;

    exec(&server, &mut session, "UPDATE t SET cat = 99 WHERE id = 2").await;

    assert_eq!(
        sorted_ints(
            &exec(&server, &mut session, "SELECT id FROM t WHERE cat = 10").await,
            0
        ),
        vec![1, 3],
        "the moved row leaves its old value, siblings stay"
    );
    assert_eq!(
        sorted_ints(
            &exec(&server, &mut session, "SELECT id FROM t WHERE cat = 99").await,
            0
        ),
        vec![2],
        "the moved row is reachable by its new value"
    );
}

#[tokio::test]
async fn unique_index_rejects_duplicate_insert() {
    let (server, _s, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE u (id INT, email INT)").await;
    exec(
        &server,
        &mut session,
        "CREATE UNIQUE INDEX idx_email ON u (email)",
    )
    .await;
    exec(&server, &mut session, "INSERT INTO u VALUES (1, 100)").await;

    // A second row with the same unique value is rejected.
    let err = try_exec(&server, &mut session, "INSERT INTO u VALUES (2, 100)")
        .await
        .expect_err("duplicate unique value must be rejected");
    assert!(
        err.to_lowercase().contains("unique"),
        "error mentions unique: {err}"
    );

    // The rejected row is not present; a distinct value is accepted.
    assert_eq!(
        total_rows(&exec(&server, &mut session, "SELECT id FROM u").await),
        1
    );
    try_exec(&server, &mut session, "INSERT INTO u VALUES (3, 200)")
        .await
        .expect("distinct unique value is accepted");
    assert_eq!(
        total_rows(&exec(&server, &mut session, "SELECT id FROM u").await),
        2
    );
}

#[tokio::test]
async fn unique_index_rejects_intra_batch_duplicate() {
    let (server, _s, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE u (id INT, email INT)").await;
    exec(
        &server,
        &mut session,
        "CREATE UNIQUE INDEX idx_email ON u (email)",
    )
    .await;

    // Two rows with the same unique value in one statement: the statement fails
    // atomically and nothing is inserted.
    let err = try_exec(
        &server,
        &mut session,
        "INSERT INTO u VALUES (1, 100), (2, 100)",
    )
    .await
    .expect_err("intra-batch duplicate must be rejected");
    assert!(
        err.to_lowercase().contains("unique"),
        "error mentions unique: {err}"
    );
    assert_eq!(
        total_rows(&exec(&server, &mut session, "SELECT id FROM u").await),
        0
    );
}

#[tokio::test]
async fn unique_index_rejects_update_into_existing_value() {
    let (server, _s, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE u (id INT, email INT)").await;
    exec(
        &server,
        &mut session,
        "CREATE UNIQUE INDEX idx_email ON u (email)",
    )
    .await;
    exec(
        &server,
        &mut session,
        "INSERT INTO u VALUES (1, 100), (2, 200)",
    )
    .await;

    let err = try_exec(
        &server,
        &mut session,
        "UPDATE u SET email = 100 WHERE id = 2",
    )
    .await
    .expect_err("update colliding with an existing unique value must be rejected");
    assert!(
        err.to_lowercase().contains("unique"),
        "error mentions unique: {err}"
    );

    // A non-colliding update of the unique-indexed row's other column succeeds
    // and the row stays reachable by its (unchanged) unique value.
    exec(&server, &mut session, "UPDATE u SET id = 9 WHERE id = 2").await;
    let r = exec(
        &server,
        &mut session,
        "SELECT id, email FROM u WHERE email = 200",
    )
    .await;
    assert_eq!(sorted_ints(&r, 0), vec![9]);
}

/// Runs a DML statement inside an already-open transaction without committing,
/// returning the execution result. Used to overlap two transactions.
async fn exec_in_txn(
    server: &Arc<ServerState>,
    txn: &zyron_storage::txn::Transaction,
    sql: &str,
) -> std::result::Result<(), String> {
    let stmt = zyron_parser::parse(sql)
        .map_err(|e| e.to_string())?
        .into_iter()
        .next()
        .expect("one statement");
    let plan = zyron_planner::plan(&server.catalog, DatabaseId(1), vec!["public".into()], stmt)
        .await
        .map_err(|e| e.to_string())?;
    let mut ctx = ExecutionContext::new(
        server.catalog.clone(),
        server.wal.clone(),
        server.buffer_pool.clone(),
        server.disk_manager.clone(),
        txn.txn_id as u32,
        txn.snapshot.clone(),
    );
    ctx.heap_files = Some(Arc::clone(&server.heap_files));
    ctx.btree_indexes = Some(Arc::clone(&server.btree_indexes));
    ctx.intent_locks = Some(Arc::clone(server.txn_manager.intent_locks()));
    let ctx = Arc::new(ctx);
    zyron_executor::execute(plan, &ctx)
        .await
        .map(|_| ())
        .map_err(|e| e.to_string())
}

#[tokio::test]
async fn unique_index_concurrent_insert_conflicts() {
    let (server, _s, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE u (id INT, email INT)").await;
    exec(
        &server,
        &mut session,
        "CREATE UNIQUE INDEX idx_email ON u (email)",
    )
    .await;

    use zyron_storage::txn::IsolationLevel;
    // Two overlapping transactions both insert the same unique value.
    let mut a = server
        .txn_manager
        .begin(IsolationLevel::ReadCommitted)
        .unwrap();
    let mut b = server
        .txn_manager
        .begin(IsolationLevel::ReadCommitted)
        .unwrap();

    exec_in_txn(&server, &a, "INSERT INTO u VALUES (1, 100)")
        .await
        .expect("first concurrent insert succeeds");
    // While A is still open, B inserting the same value must be rejected.
    let err = exec_in_txn(&server, &b, "INSERT INTO u VALUES (2, 100)")
        .await
        .expect_err("concurrent duplicate must be rejected");
    assert!(
        err.to_lowercase().contains("unique") || err.to_lowercase().contains("conflict"),
        "rejection mentions unique or conflict: {err}"
    );

    server.txn_manager.commit(&mut a).await.expect("commit A");
    let _ = server.txn_manager.abort(&mut b);

    // After A commits, exactly its row is present and a later duplicate still fails.
    assert_eq!(
        total_rows(&exec(&server, &mut session, "SELECT id FROM u").await),
        1
    );
    let err2 = try_exec(&server, &mut session, "INSERT INTO u VALUES (3, 100)")
        .await
        .expect_err("post-commit duplicate rejected");
    assert!(
        err2.to_lowercase().contains("unique"),
        "post-commit dup mentions unique: {err2}"
    );
}

// ---------------------------------------------------------------------------
// MVCC delete/update rollback atomicity
// ---------------------------------------------------------------------------

#[tokio::test]
async fn aborted_delete_keeps_row() {
    let (server, _s, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE t (id INT, v INT)").await;
    exec(
        &server,
        &mut session,
        "INSERT INTO t VALUES (1, 10), (2, 20)",
    )
    .await;

    use zyron_storage::txn::IsolationLevel;
    let mut d = server
        .txn_manager
        .begin(IsolationLevel::ReadCommitted)
        .unwrap();
    exec_in_txn(&server, &d, "DELETE FROM t WHERE id = 1")
        .await
        .expect("delete executes");
    // Roll back: the engine does no physical undo, so correctness depends on the
    // delete being an xmax stamp the commit-status map reports as aborted.
    let _ = server.txn_manager.abort(&mut d);

    let rows = exec(&server, &mut session, "SELECT id, v FROM t").await;
    assert_eq!(
        id_v_pairs(&rows),
        vec![(1, 10), (2, 20)],
        "rolled-back delete leaves both rows visible"
    );
}

#[tokio::test]
async fn committed_delete_hides_row() {
    let (server, _s, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE t (id INT, v INT)").await;
    exec(
        &server,
        &mut session,
        "INSERT INTO t VALUES (1, 10), (2, 20)",
    )
    .await;

    exec(&server, &mut session, "DELETE FROM t WHERE id = 1").await; // committed

    let rows = exec(&server, &mut session, "SELECT id, v FROM t").await;
    assert_eq!(
        id_v_pairs(&rows),
        vec![(2, 20)],
        "committed delete hides the row"
    );
}

#[tokio::test]
async fn aborted_update_keeps_old_value() {
    let (server, _s, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE t (id INT, v INT)").await;
    exec(&server, &mut session, "INSERT INTO t VALUES (1, 10)").await;

    use zyron_storage::txn::IsolationLevel;
    let mut u = server
        .txn_manager
        .begin(IsolationLevel::ReadCommitted)
        .unwrap();
    exec_in_txn(&server, &u, "UPDATE t SET v = 999 WHERE id = 1")
        .await
        .expect("update executes");
    let _ = server.txn_manager.abort(&mut u);

    // The old image stays visible (its xmax = aborted txn) and the new image
    // (xmin = aborted txn) is invisible.
    let rows = exec(&server, &mut session, "SELECT id, v FROM t").await;
    assert_eq!(
        id_v_pairs(&rows),
        vec![(1, 10)],
        "rolled-back update keeps the original value and drops the new image"
    );
}

#[tokio::test]
async fn aborted_delete_then_reinsert_value_unique_ok() {
    // A rolled-back delete leaves the row live, so re-inserting its unique value
    // in another transaction is correctly rejected (the row still holds it).
    let (server, _s, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE u (id INT, email INT)").await;
    exec(
        &server,
        &mut session,
        "CREATE UNIQUE INDEX idx_email ON u (email)",
    )
    .await;
    exec(&server, &mut session, "INSERT INTO u VALUES (1, 100)").await;

    use zyron_storage::txn::IsolationLevel;
    let mut d = server
        .txn_manager
        .begin(IsolationLevel::ReadCommitted)
        .unwrap();
    exec_in_txn(&server, &d, "DELETE FROM u WHERE id = 1")
        .await
        .expect("delete");
    let _ = server.txn_manager.abort(&mut d);

    // Row 1 is still live after the rolled-back delete, so email=100 conflicts.
    let err = try_exec(&server, &mut session, "INSERT INTO u VALUES (2, 100)")
        .await
        .expect_err("value still held by the un-deleted row");
    assert!(
        err.to_lowercase().contains("unique"),
        "mentions unique: {err}"
    );
}

#[tokio::test]
async fn committed_delete_then_reinsert_value_unique_ok() {
    // After a committed delete, the value is free and can be re-inserted.
    let (server, _s, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE u (id INT, email INT)").await;
    exec(
        &server,
        &mut session,
        "CREATE UNIQUE INDEX idx_email ON u (email)",
    )
    .await;
    exec(&server, &mut session, "INSERT INTO u VALUES (1, 100)").await;
    exec(&server, &mut session, "DELETE FROM u WHERE id = 1").await; // committed

    try_exec(&server, &mut session, "INSERT INTO u VALUES (2, 100)")
        .await
        .expect("value freed by committed delete can be reused");
    let rows = exec(&server, &mut session, "SELECT id, email FROM u").await;
    assert_eq!(sorted_ints(&rows, 0), vec![2]);
}

// ---------------------------------------------------------------------------
// Foreign key enforcement
// ---------------------------------------------------------------------------

async fn setup_parent_child(
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
    fk_clause: &str,
) {
    exec(
        server,
        session,
        "CREATE TABLE parent (id INT PRIMARY KEY, label TEXT)",
    )
    .await;
    exec(
        server,
        session,
        &format!("CREATE TABLE child (cid INT, pid INT REFERENCES parent(id){fk_clause})"),
    )
    .await;
    exec(
        server,
        session,
        "INSERT INTO parent VALUES (1, 'a'), (2, 'b')",
    )
    .await;
}

#[tokio::test]
async fn fk_insert_rejects_missing_parent() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    setup_parent_child(&server, &mut session, "").await;

    let err = try_exec(&server, &mut session, "INSERT INTO child VALUES (10, 99)")
        .await
        .expect_err("insert with no matching parent must fail");
    assert!(
        err.contains("foreign key"),
        "error mentions foreign key: {err}"
    );
}

#[tokio::test]
async fn fk_insert_accepts_existing_parent_and_null() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    setup_parent_child(&server, &mut session, "").await;

    // Existing parent key is accepted.
    try_exec(&server, &mut session, "INSERT INTO child VALUES (10, 1)")
        .await
        .expect("insert with matching parent succeeds");
    // MATCH SIMPLE: a null foreign key references nothing and is allowed.
    try_exec(&server, &mut session, "INSERT INTO child VALUES (11, NULL)")
        .await
        .expect("null foreign key is allowed");

    let rows = exec(&server, &mut session, "SELECT cid FROM child").await;
    assert_eq!(total_rows(&rows), 2);
}

#[tokio::test]
async fn fk_delete_no_action_blocks_when_children_exist() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    setup_parent_child(&server, &mut session, "").await;
    exec(&server, &mut session, "INSERT INTO child VALUES (10, 1)").await;

    let err = try_exec(&server, &mut session, "DELETE FROM parent WHERE id = 1")
        .await
        .expect_err("delete of a referenced parent must fail under NO ACTION");
    assert!(
        err.contains("foreign key"),
        "error mentions foreign key: {err}"
    );

    // Deleting an unreferenced parent still works.
    try_exec(&server, &mut session, "DELETE FROM parent WHERE id = 2")
        .await
        .expect("unreferenced parent deletes");
}

#[tokio::test]
async fn fk_delete_cascade_removes_children() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    setup_parent_child(&server, &mut session, " ON DELETE CASCADE").await;
    exec(
        &server,
        &mut session,
        "INSERT INTO child VALUES (10, 1), (11, 1), (12, 2)",
    )
    .await;

    exec(&server, &mut session, "DELETE FROM parent WHERE id = 1").await;

    let rows = exec(&server, &mut session, "SELECT cid, pid FROM child").await;
    assert_eq!(
        total_rows(&rows),
        1,
        "two children of parent 1 cascade-deleted"
    );
    assert_eq!(column_values(&rows, 0), vec![ScalarValue::Int32(12)]);
}

#[tokio::test]
async fn fk_delete_set_null_clears_child_key() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    setup_parent_child(&server, &mut session, " ON DELETE SET NULL").await;
    exec(
        &server,
        &mut session,
        "INSERT INTO child VALUES (10, 1), (11, 2)",
    )
    .await;

    exec(&server, &mut session, "DELETE FROM parent WHERE id = 1").await;

    let rows = exec(&server, &mut session, "SELECT cid, pid FROM child").await;
    assert_eq!(total_rows(&rows), 2, "children are kept, not deleted");
    let cids = column_values(&rows, 0);
    let pids = column_values(&rows, 1);
    // Child 10 referenced the deleted parent, so its key is now null; child 11 keeps 2.
    for (cid, pid) in cids.iter().zip(&pids) {
        match cid {
            ScalarValue::Int32(10) => {
                assert!(pid.is_null(), "orphaned child key set to null: {pid:?}")
            }
            ScalarValue::Int32(11) => assert_eq!(*pid, ScalarValue::Int32(2)),
            other => panic!("unexpected child cid {other:?}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Uncorrelated subquery execution
// ---------------------------------------------------------------------------

async fn setup_nums(server: &Arc<ServerState>, session: &mut Option<Session>) {
    exec(server, session, "CREATE TABLE nums (id INT, v INT)").await;
    exec(
        server,
        session,
        "INSERT INTO nums VALUES (1, 10), (2, 20), (3, 30)",
    )
    .await;
}

fn ints(batches: &[DataBatch], col: usize) -> Vec<i32> {
    column_values(batches, col)
        .into_iter()
        .filter_map(|s| match s {
            ScalarValue::Int32(v) => Some(v),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn subquery_scalar_in_where() {
    let (server, _s, _tmp) = create_test_server().await;
    let mut session = new_session();
    setup_nums(&server, &mut session).await;

    let rows = exec(
        &server,
        &mut session,
        "SELECT id FROM nums WHERE v = (SELECT MAX(v) FROM nums)",
    )
    .await;
    assert_eq!(ints(&rows, 0), vec![3], "only the max-valued row matches");
}

#[tokio::test]
async fn subquery_in_list_from_select() {
    let (server, _s, _tmp) = create_test_server().await;
    let mut session = new_session();
    setup_nums(&server, &mut session).await;

    let rows = exec(
        &server,
        &mut session,
        "SELECT id FROM nums WHERE v IN (SELECT v FROM nums WHERE v >= 20)",
    )
    .await;
    let mut got = ints(&rows, 0);
    got.sort();
    assert_eq!(got, vec![2, 3]);
}

#[tokio::test]
async fn subquery_exists_and_not_exists() {
    let (server, _s, _tmp) = create_test_server().await;
    let mut session = new_session();
    setup_nums(&server, &mut session).await;

    let yes = exec(
        &server,
        &mut session,
        "SELECT id FROM nums WHERE EXISTS (SELECT 1 FROM nums WHERE v > 25)",
    )
    .await;
    assert_eq!(total_rows(&yes), 3, "EXISTS is true so all rows pass");

    let no = exec(
        &server,
        &mut session,
        "SELECT id FROM nums WHERE EXISTS (SELECT 1 FROM nums WHERE v > 1000)",
    )
    .await;
    assert_eq!(total_rows(&no), 0, "EXISTS is false so no rows pass");

    let not_ex = exec(
        &server,
        &mut session,
        "SELECT id FROM nums WHERE NOT EXISTS (SELECT 1 FROM nums WHERE v > 1000)",
    )
    .await;
    assert_eq!(
        total_rows(&not_ex),
        3,
        "NOT EXISTS is true so all rows pass"
    );
}

#[tokio::test]
async fn subquery_scalar_in_projection() {
    let (server, _s, _tmp) = create_test_server().await;
    let mut session = new_session();
    setup_nums(&server, &mut session).await;

    let rows = exec(
        &server,
        &mut session,
        "SELECT id, (SELECT COUNT(*) FROM nums) FROM nums",
    )
    .await;
    assert_eq!(total_rows(&rows), 3);
    let counts = column_values(&rows, 1);
    assert!(
        counts
            .iter()
            .all(|c| matches!(c, ScalarValue::Int64(3) | ScalarValue::Int32(3))),
        "scalar count subquery is 3 for every row: {counts:?}"
    );
}

// A column list that omits or reorders columns reshapes to full table order:
// omitted columns become NULL, reordered ones land in the right column.
#[tokio::test]
async fn partial_and_reordered_column_insert() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec(
        &server,
        &mut session,
        "CREATE TABLE t (a INT, b TEXT, c INT)",
    )
    .await;
    // Omit c.
    exec(
        &server,
        &mut session,
        "INSERT INTO t (a, b) VALUES (1, 'x')",
    )
    .await;
    // Reorder b before a; omit c.
    exec(
        &server,
        &mut session,
        "INSERT INTO t (b, a) VALUES ('y', 2)",
    )
    .await;
    // Provide only c; a and b omitted.
    exec(&server, &mut session, "INSERT INTO t (c) VALUES (9)").await;

    let rows = exec(&server, &mut session, "SELECT a, b, c FROM t").await;
    assert_eq!(total_rows(&rows), 3);

    let a_vals = column_values(&rows, 0);
    let b_vals = column_values(&rows, 1);
    let c_vals = column_values(&rows, 2);

    // a: 1, 2, and one NULL (from the c-only row).
    assert_eq!(
        a_vals
            .iter()
            .filter(|v| matches!(v, ScalarValue::Int32(1)))
            .count(),
        1
    );
    assert_eq!(
        a_vals
            .iter()
            .filter(|v| matches!(v, ScalarValue::Int32(2)))
            .count(),
        1
    );
    assert_eq!(
        a_vals
            .iter()
            .filter(|v| matches!(v, ScalarValue::Null))
            .count(),
        1
    );
    // b reordered into place: 'x', 'y', and one NULL.
    assert_eq!(
        b_vals
            .iter()
            .filter(|v| matches!(v, ScalarValue::Utf8(s) if s == "x"))
            .count(),
        1
    );
    assert_eq!(
        b_vals
            .iter()
            .filter(|v| matches!(v, ScalarValue::Utf8(s) if s == "y"))
            .count(),
        1
    );
    assert_eq!(
        b_vals
            .iter()
            .filter(|v| matches!(v, ScalarValue::Null))
            .count(),
        1
    );
    // c: 9 once, NULL for the two rows that omitted it.
    assert_eq!(
        c_vals
            .iter()
            .filter(|v| matches!(v, ScalarValue::Int32(9)))
            .count(),
        1
    );
    assert_eq!(
        c_vals
            .iter()
            .filter(|v| matches!(v, ScalarValue::Null))
            .count(),
        2
    );
}

// CHECK constraints are enforced on INSERT and UPDATE; passing rows go through,
// violating rows are rejected with no effect.
#[tokio::test]
async fn check_constraint_enforced_on_insert_and_update() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec(
        &server,
        &mut session,
        "CREATE TABLE accounts (id INT, balance INT CHECK (balance >= 0))",
    )
    .await;

    // A satisfying insert succeeds.
    exec(
        &server,
        &mut session,
        "INSERT INTO accounts VALUES (1, 100)",
    )
    .await;
    // A violating insert is rejected.
    let bad_insert = try_exec(&server, &mut session, "INSERT INTO accounts VALUES (2, -5)").await;
    assert!(
        bad_insert.is_err(),
        "negative balance must violate the CHECK"
    );

    // The rejected row left no trace.
    let rows = exec(&server, &mut session, "SELECT id FROM accounts").await;
    assert_eq!(total_rows(&rows), 1);

    // A satisfying update succeeds; a violating update is rejected.
    exec(
        &server,
        &mut session,
        "UPDATE accounts SET balance = 50 WHERE id = 1",
    )
    .await;
    let bad_update = try_exec(
        &server,
        &mut session,
        "UPDATE accounts SET balance = -1 WHERE id = 1",
    )
    .await;
    assert!(
        bad_update.is_err(),
        "update to negative balance must violate the CHECK"
    );

    // The balance is still the last valid value.
    let balances = exec(
        &server,
        &mut session,
        "SELECT balance FROM accounts WHERE id = 1",
    )
    .await;
    let vals = column_values(&balances, 0);
    assert!(vals.iter().any(|v| matches!(v, ScalarValue::Int32(50))));
}

// A cascaded ON DELETE SET NULL that would violate the child's CHECK must abort
// the delete rather than write a violating row.
#[tokio::test]
async fn fk_cascade_set_null_respects_check() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(
        &server,
        &mut session,
        "CREATE TABLE parent (id INT PRIMARY KEY)",
    )
    .await;
    exec(
        &server,
        &mut session,
        "CREATE TABLE child (cid INT, pid INT REFERENCES parent(id) ON DELETE SET NULL, CHECK (pid IS NOT NULL))",
    )
    .await;
    exec(&server, &mut session, "INSERT INTO parent VALUES (1)").await;
    exec(&server, &mut session, "INSERT INTO child VALUES (10, 1)").await;

    // SET NULL on pid would break CHECK (pid IS NOT NULL); the cascade aborts.
    let res = try_exec(&server, &mut session, "DELETE FROM parent WHERE id = 1").await;
    assert!(
        res.is_err(),
        "cascade SET NULL violating a CHECK must abort the delete"
    );

    // Nothing changed: parent and child rows remain.
    let p = exec(&server, &mut session, "SELECT id FROM parent").await;
    assert_eq!(total_rows(&p), 1, "parent delete rolled back");
}

// An omitted column with a DEFAULT is filled with that default, not NULL.
#[tokio::test]
async fn omitted_column_uses_default() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec(
        &server,
        &mut session,
        "CREATE TABLE t (id INT, status TEXT DEFAULT 'active', score INT DEFAULT 7)",
    )
    .await;
    // Omit status and score; both carry defaults.
    exec(&server, &mut session, "INSERT INTO t (id) VALUES (1)").await;
    // Provide status, omit score.
    exec(
        &server,
        &mut session,
        "INSERT INTO t (id, status) VALUES (2, 'paused')",
    )
    .await;

    let rows = exec(&server, &mut session, "SELECT id, status, score FROM t").await;
    assert_eq!(total_rows(&rows), 2);
    let status = column_values(&rows, 1);
    let score = column_values(&rows, 2);
    // Defaults applied where omitted; provided value kept where given.
    assert_eq!(
        status
            .iter()
            .filter(|v| matches!(v, ScalarValue::Utf8(s) if s == "active"))
            .count(),
        1
    );
    assert_eq!(
        status
            .iter()
            .filter(|v| matches!(v, ScalarValue::Utf8(s) if s == "paused"))
            .count(),
        1
    );
    assert_eq!(
        status
            .iter()
            .filter(|v| matches!(v, ScalarValue::Null))
            .count(),
        0
    );
    // score defaulted to 7 for both rows (omitted in both).
    assert_eq!(
        score
            .iter()
            .filter(|v| matches!(v, ScalarValue::Int32(7)))
            .count(),
        2
    );
}

// CREATE GRAPH SCHEMA materializes backing tables; a graph algorithm query
// loads edges from those heaps on demand (no pre-cached CSR) and runs.
#[tokio::test]
async fn graph_create_insert_and_algorithm_end_to_end() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec(
        &server,
        &mut session,
        "CREATE GRAPH SCHEMA social (NODE Person (name TEXT), EDGE Knows FROM Person TO Person (since INT))",
    )
    .await;

    // The edge backing table is a real, insertable catalog table: a directed
    // 3-cycle 1 -> 2 -> 3 -> 1.
    exec(
        &server,
        &mut session,
        "INSERT INTO social_Knows (from_node, to_node) VALUES (1, 2)",
    )
    .await;
    exec(
        &server,
        &mut session,
        "INSERT INTO social_Knows (from_node, to_node) VALUES (2, 3)",
    )
    .await;
    exec(
        &server,
        &mut session,
        "INSERT INTO social_Knows (from_node, to_node) VALUES (3, 1)",
    )
    .await;

    // Cold query: no CSR is cached, so the operator loads edges from the heap.
    let rows = exec(
        &server,
        &mut session,
        "SELECT node_id, component FROM connected_components('social')",
    )
    .await;
    assert_eq!(total_rows(&rows), 3, "every node appears once");

    // A cycle is one connected component: all nodes share a component id.
    let components = column_values(&rows, 1);
    let mut distinct = std::collections::HashSet::new();
    for c in &components {
        match c {
            ScalarValue::Int64(v) => {
                distinct.insert(*v);
            }
            other => panic!("component id should be Int64, got {other:?}"),
        }
    }
    assert_eq!(distinct.len(), 1, "a directed cycle is a single component");
}

// After an edge insert invalidates the cached CSR, a re-run reflects the new
// edge: a fourth disconnected node-pair becomes its own component.
#[tokio::test]
async fn graph_csr_rebuilds_after_edge_insert() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec(
        &server,
        &mut session,
        "CREATE GRAPH SCHEMA net (NODE N (v INT), EDGE E FROM N TO N (w INT))",
    )
    .await;
    exec(
        &server,
        &mut session,
        "INSERT INTO net_E (from_node, to_node) VALUES (1, 2)",
    )
    .await;

    let first = exec(
        &server,
        &mut session,
        "SELECT node_id, component FROM connected_components('net')",
    )
    .await;
    assert_eq!(total_rows(&first), 2, "two connected nodes");

    // A disconnected edge 10 -> 11 must show up after invalidation.
    exec(
        &server,
        &mut session,
        "INSERT INTO net_E (from_node, to_node) VALUES (10, 11)",
    )
    .await;
    let second = exec(
        &server,
        &mut session,
        "SELECT node_id, component FROM connected_components('net')",
    )
    .await;
    assert_eq!(total_rows(&second), 4, "CSR rebuilt with the new edge");
    let components = column_values(&second, 1);
    let mut distinct = std::collections::HashSet::new();
    for c in &components {
        if let ScalarValue::Int64(v) = c {
            distinct.insert(*v);
        }
    }
    assert_eq!(
        distinct.len(),
        2,
        "two disjoint components after the insert"
    );
}

// ---------------------------------------------------------------------------
// ALTER TABLE ADD CONSTRAINT validates existing rows
// ---------------------------------------------------------------------------

// ADD CHECK is rejected when a current row already violates the predicate, and
// accepted once the offending row is removed.
#[tokio::test]
async fn add_check_validates_existing_rows() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec(&server, &mut session, "CREATE TABLE t (id INT, qty INT)").await;
    exec(
        &server,
        &mut session,
        "INSERT INTO t VALUES (1, 5), (2, -3)",
    )
    .await;

    let err = try_exec(
        &server,
        &mut session,
        "ALTER TABLE t ADD CONSTRAINT ck_qty CHECK (qty >= 0)",
    )
    .await
    .expect_err("ADD CHECK must reject the negative row");
    assert!(err.contains("violate CHECK"), "unexpected error: {err}");

    exec(&server, &mut session, "DELETE FROM t WHERE qty < 0").await;
    try_exec(
        &server,
        &mut session,
        "ALTER TABLE t ADD CONSTRAINT ck_qty CHECK (qty >= 0)",
    )
    .await
    .expect("ADD CHECK succeeds after the violating row is gone");

    // The accepted constraint now blocks a new violating insert.
    let after = try_exec(&server, &mut session, "INSERT INTO t VALUES (3, -1)").await;
    assert!(after.is_err(), "the enforced CHECK rejects later inserts");
}

// CHECK treats NULL as unknown (passes), so a NULL row does not block the ADD.
#[tokio::test]
async fn add_check_passes_null_rows() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec(&server, &mut session, "CREATE TABLE t (id INT, qty INT)").await;
    exec(
        &server,
        &mut session,
        "INSERT INTO t VALUES (1, 5), (2, NULL)",
    )
    .await;

    try_exec(
        &server,
        &mut session,
        "ALTER TABLE t ADD CONSTRAINT ck_qty CHECK (qty >= 0)",
    )
    .await
    .expect("NULL qty is unknown, not a violation");
}

// ADD FOREIGN KEY is rejected when a child row points at a missing parent, and
// accepted once every non-null key has a matching parent. A NULL key passes.
#[tokio::test]
async fn add_foreign_key_validates_existing_rows() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec(
        &server,
        &mut session,
        "CREATE TABLE parent (pid INT, PRIMARY KEY (pid))",
    )
    .await;
    exec(&server, &mut session, "INSERT INTO parent VALUES (1), (2)").await;
    exec(
        &server,
        &mut session,
        "CREATE TABLE child (cid INT, pid INT)",
    )
    .await;
    exec(
        &server,
        &mut session,
        "INSERT INTO child VALUES (10, 1), (11, 99)",
    )
    .await;

    let err = try_exec(
        &server,
        &mut session,
        "ALTER TABLE child ADD CONSTRAINT fk_pid FOREIGN KEY (pid) REFERENCES parent (pid)",
    )
    .await
    .expect_err("orphan row 99 must block the FK");
    assert!(
        err.contains("reference a missing row"),
        "unexpected error: {err}"
    );

    // Fix the orphan to point at an existing parent, plus add a NULL row that
    // must be allowed under MATCH SIMPLE.
    exec(
        &server,
        &mut session,
        "UPDATE child SET pid = 2 WHERE cid = 11",
    )
    .await;
    exec(&server, &mut session, "INSERT INTO child VALUES (12, NULL)").await;
    try_exec(
        &server,
        &mut session,
        "ALTER TABLE child ADD CONSTRAINT fk_pid FOREIGN KEY (pid) REFERENCES parent (pid)",
    )
    .await
    .expect("FK accepted once orphan is fixed; NULL key passes");
}

// ADD UNIQUE is rejected on duplicate non-null values but ignores multiple NULLs.
#[tokio::test]
async fn add_unique_validates_existing_rows() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec(&server, &mut session, "CREATE TABLE t (id INT, email TEXT)").await;
    exec(
        &server,
        &mut session,
        "INSERT INTO t VALUES (1, 'a@x'), (2, 'a@x')",
    )
    .await;

    let err = try_exec(
        &server,
        &mut session,
        "ALTER TABLE t ADD CONSTRAINT uq_email UNIQUE (email)",
    )
    .await
    .expect_err("duplicate email must block UNIQUE");
    assert!(err.contains("violate UNIQUE"), "unexpected error: {err}");

    exec(
        &server,
        &mut session,
        "UPDATE t SET email = 'b@x' WHERE id = 2",
    )
    .await;
    // Two NULL emails are distinct and must not block UNIQUE.
    exec(
        &server,
        &mut session,
        "INSERT INTO t VALUES (3, NULL), (4, NULL)",
    )
    .await;
    try_exec(
        &server,
        &mut session,
        "ALTER TABLE t ADD CONSTRAINT uq_email UNIQUE (email)",
    )
    .await
    .expect("UNIQUE accepted once duplicates are gone; NULLs are distinct");
}

// ADD PRIMARY KEY rejects both duplicates and NULLs in the key column.
#[tokio::test]
async fn add_primary_key_validates_existing_rows() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec(&server, &mut session, "CREATE TABLE t (id INT, name TEXT)").await;
    exec(
        &server,
        &mut session,
        "INSERT INTO t VALUES (1, 'a'), (1, 'b')",
    )
    .await;

    let dup_err = try_exec(
        &server,
        &mut session,
        "ALTER TABLE t ADD CONSTRAINT pk_id PRIMARY KEY (id)",
    )
    .await
    .expect_err("duplicate id must block PRIMARY KEY");
    assert!(
        dup_err.contains("violate PRIMARY KEY"),
        "unexpected error: {dup_err}"
    );

    exec(
        &server,
        &mut session,
        "UPDATE t SET id = 2 WHERE name = 'b'",
    )
    .await;
    exec(&server, &mut session, "INSERT INTO t VALUES (NULL, 'c')").await;
    let null_err = try_exec(
        &server,
        &mut session,
        "ALTER TABLE t ADD CONSTRAINT pk_id PRIMARY KEY (id)",
    )
    .await
    .expect_err("NULL id must block PRIMARY KEY");
    assert!(
        null_err.contains("NULL in a PRIMARY KEY"),
        "unexpected error: {null_err}"
    );

    exec(&server, &mut session, "DELETE FROM t WHERE id IS NULL").await;
    try_exec(
        &server,
        &mut session,
        "ALTER TABLE t ADD CONSTRAINT pk_id PRIMARY KEY (id)",
    )
    .await
    .expect("PRIMARY KEY accepted once unique and non-null");
}

// End-to-end proof that an aggregate appearing only in HAVING (not the SELECT
// list) is computed by the aggregate operator instead of erroring.
#[tokio::test]
async fn having_only_aggregate_executes() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec(&server, &mut session, "CREATE TABLE t (k INT, v INT)").await;
    exec(
        &server,
        &mut session,
        "INSERT INTO t VALUES (1, 10), (1, 11), (2, 20), (3, 30), (3, 31), (3, 32)",
    )
    .await;

    // Keys 1 and 3 have more than one row; key 2 has exactly one.
    let rows = exec(
        &server,
        &mut session,
        "SELECT k FROM t GROUP BY k HAVING COUNT(*) > 1",
    )
    .await;
    let mut keys: Vec<i64> = column_values(&rows, 0)
        .into_iter()
        .filter_map(|s| match s {
            ScalarValue::Int32(v) => Some(v as i64),
            ScalarValue::Int64(v) => Some(v),
            _ => None,
        })
        .collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        vec![1, 3],
        "HAVING COUNT(*) > 1 keeps only duplicated keys"
    );
}

// Reads one Int32/Int64 column across all batches as i64 in row order.
fn int_col(batches: &[DataBatch], col_idx: usize) -> Vec<i64> {
    column_values(batches, col_idx)
        .into_iter()
        .filter_map(|s| match s {
            ScalarValue::Int32(v) => Some(v as i64),
            ScalarValue::Int64(v) => Some(v),
            _ => None,
        })
        .collect()
}

// ORDER BY resolves against the input schema beneath the projection, so it
// works whether the key is selected, unselected, or a computed expression.
#[tokio::test]
async fn order_by_with_projection_sorts_rows() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE t (a INT, b INT)").await;
    exec(
        &server,
        &mut session,
        "INSERT INTO t VALUES (3, 1), (1, 2), (2, 3)",
    )
    .await;

    // ORDER BY a selected column.
    let r = exec(&server, &mut session, "SELECT a FROM t ORDER BY a").await;
    assert_eq!(int_col(&r, 0), vec![1, 2, 3]);

    // Descending.
    let r = exec(&server, &mut session, "SELECT a FROM t ORDER BY a DESC").await;
    assert_eq!(int_col(&r, 0), vec![3, 2, 1]);

    // ORDER BY a column not in the projection: b ascending is 1,2,3 which maps
    // to a = 3,1,2.
    let r = exec(&server, &mut session, "SELECT a FROM t ORDER BY b").await;
    assert_eq!(int_col(&r, 0), vec![3, 1, 2]);

    // ORDER BY a computed expression over unselected columns.
    let r = exec(&server, &mut session, "SELECT a FROM t ORDER BY a + b DESC").await;
    // a+b = 4,3,5 for rows (3,1),(1,2),(2,3); desc -> (2,3)=5,(3,1)=4,(1,2)=3 -> a = 2,3,1.
    assert_eq!(int_col(&r, 0), vec![2, 3, 1]);
}

// Column OP column arithmetic preserves the integer type instead of writing
// zeros. INT is Int32, which is neither the Int64 nor Float64 fast path.
#[tokio::test]
async fn column_column_arithmetic_int() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE t (a INT, b INT)").await;
    exec(
        &server,
        &mut session,
        "INSERT INTO t VALUES (3, 1), (1, 2), (2, 3)",
    )
    .await;

    let r = exec(&server, &mut session, "SELECT a + b AS s FROM t").await;
    assert_eq!(int_col(&r, 0), vec![4, 3, 5]);

    let r = exec(&server, &mut session, "SELECT b - a AS s FROM t").await;
    assert_eq!(int_col(&r, 0), vec![-2, 1, 1]);

    let r = exec(&server, &mut session, "SELECT a * b AS s FROM t").await;
    assert_eq!(int_col(&r, 0), vec![3, 2, 6]);

    // Column OP literal still works (it used the Int32 fast path values too).
    let r = exec(&server, &mut session, "SELECT a + 10 AS s FROM t").await;
    assert_eq!(int_col(&r, 0), vec![13, 11, 12]);
}

// ORDER BY over a group key and over an aggregate, including an aggregate that
// is not in the SELECT list.
#[tokio::test]
async fn order_by_aggregate_and_group_key() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE t (k INT)").await;
    // k=1 x1, k=2 x3, k=3 x2.
    exec(
        &server,
        &mut session,
        "INSERT INTO t VALUES (1), (2), (2), (2), (3), (3)",
    )
    .await;

    // ORDER BY group key.
    let r = exec(
        &server,
        &mut session,
        "SELECT k FROM t GROUP BY k ORDER BY k DESC",
    )
    .await;
    assert_eq!(int_col(&r, 0), vec![3, 2, 1]);

    // ORDER BY an aggregate present in the SELECT list.
    let r = exec(
        &server,
        &mut session,
        "SELECT k, COUNT(*) FROM t GROUP BY k ORDER BY COUNT(*) DESC",
    )
    .await;
    assert_eq!(
        int_col(&r, 0),
        vec![2, 3, 1],
        "keys ordered by descending count"
    );
    assert_eq!(int_col(&r, 1), vec![3, 2, 1], "the counts themselves");

    // ORDER BY an aggregate that is NOT selected.
    let r = exec(
        &server,
        &mut session,
        "SELECT k FROM t GROUP BY k ORDER BY COUNT(*) ASC",
    )
    .await;
    assert_eq!(
        int_col(&r, 0),
        vec![1, 3, 2],
        "keys ordered by ascending count"
    );
}

// DISTINCT then ORDER BY: the row-order-preserving distinct keeps the sorted
// order produced beneath it.
#[tokio::test]
async fn order_by_with_distinct() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE t (a INT)").await;
    exec(
        &server,
        &mut session,
        "INSERT INTO t VALUES (3), (1), (2), (3), (1)",
    )
    .await;

    let r = exec(&server, &mut session, "SELECT DISTINCT a FROM t ORDER BY a").await;
    assert_eq!(
        int_col(&r, 0),
        vec![1, 2, 3],
        "distinct values in sorted order"
    );
}

//! Integration tests for ROLLBACK TO SAVEPOINT (partial transaction rollback).
//!
//! A savepoint marks a point in an open transaction. ROLLBACK TO SAVEPOINT
//! reverses only the writes made after the savepoint and keeps the transaction
//! open. The engine records a per-transaction undo log of reverse-ops while a
//! savepoint is open (reverse-insert self-deletes an inserted row, reverse-delete
//! restores a deleted row); rollback replays them at the heap-tuple level. These
//! tests drive a single open transaction across statements and exercise the same
//! undo-record + replay path the wire ROLLBACK TO SAVEPOINT handler uses.
//!
//! Run: cargo test -p zyron-wire --test savepoint_test -- --nocapture

use std::sync::Arc;

use zyron_buffer::{BufferPool, BufferPoolConfig};
use zyron_catalog::{
    Catalog, CatalogCache, DatabaseId, HeapCatalogStorage, SYSTEM_DATABASE_ID, SchemaId,
};
use zyron_executor::batch::DataBatch;
use zyron_executor::column::ScalarValue;
use zyron_executor::context::ExecutionContext;
use zyron_storage::txn::{IsolationLevel, Transaction, TransactionManager};
use zyron_storage::{DiskManager, DiskManagerConfig, HeapFile, HeapFileConfig, UndoEntry};
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
        DiskManager::new(zyron_bench_harness::disk_config(data_dir.clone()))
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
    let cdc_registry = Arc::new(zyron_cdc::CdfRegistry::new(data_dir.clone()));

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
        cdc_registry: Some(cdc_registry),
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

/// Runs one statement in its own auto-committed transaction. Used for DDL and
/// for reads that must observe committed state from a second transaction.
async fn exec_autocommit(server: &Arc<ServerState>, session: &mut Option<Session>, sql: &str) {
    let stmt = zyron_parser::parse(sql)
        .expect("parse")
        .into_iter()
        .next()
        .expect("one statement");

    let mut txn_opt: Option<Transaction> = None;
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
        res.expect("ddl utility");
        return;
    }

    let plan = zyron_planner::plan(
        &server.catalog,
        DatabaseId(1),
        vec!["public".into()],
        stmt,
        None,
    )
    .await
    .expect("plan");
    let mut txn = server
        .txn_manager
        .begin(IsolationLevel::ReadCommitted)
        .expect("begin");
    let ctx = build_ctx(server, &txn);
    zyron_executor::execute(plan, &ctx).await.expect("execute");
    server.txn_manager.commit(&mut txn).await.expect("commit");
}

/// Builds an execution context bound to `txn`, with the transaction's undo log
/// attached so DML operators record reverse-ops while a savepoint is open.
fn build_ctx(server: &Arc<ServerState>, txn: &Transaction) -> Arc<ExecutionContext> {
    let mut ctx = ExecutionContext::new(
        server.catalog.clone(),
        server.wal.clone(),
        server.buffer_pool.clone(),
        server.disk_manager.clone(),
        txn.txn_id() as u32,
        txn.snapshot.clone(),
    );
    ctx.heap_files = Some(Arc::clone(&server.heap_files));
    ctx.btree_indexes = Some(Arc::clone(&server.btree_indexes));
    ctx.intent_locks = Some(Arc::clone(server.txn_manager.intent_locks()));
    ctx.row_locks = Some(Arc::clone(server.txn_manager.lock_table()));
    ctx.undo_log = Some(txn.undo_log());
    ctx.read_only = txn.read_only();
    Arc::new(ctx)
}

/// Runs a statement in a transaction and returns the execute Result instead of
/// panicking, so a rejection can be asserted.
async fn try_exec_in_txn(
    server: &Arc<ServerState>,
    txn: &Transaction,
    sql: &str,
) -> zyron_common::Result<Vec<DataBatch>> {
    let stmt = zyron_parser::parse(sql)
        .expect("parse")
        .into_iter()
        .next()
        .expect("one statement");
    let plan = zyron_planner::plan(
        &server.catalog,
        DatabaseId(1),
        vec!["public".into()],
        stmt,
        None,
    )
    .await
    .expect("plan");
    let ctx = build_ctx(server, txn);
    zyron_executor::execute(plan, &ctx).await
}

/// Runs a DML or SELECT statement inside the given open transaction. The
/// transaction is NOT committed, so it stays open for the next statement.
async fn exec_in_txn(server: &Arc<ServerState>, txn: &Transaction, sql: &str) -> Vec<DataBatch> {
    let stmt = zyron_parser::parse(sql)
        .expect("parse")
        .into_iter()
        .next()
        .expect("one statement");
    let plan = zyron_planner::plan(
        &server.catalog,
        DatabaseId(1),
        vec!["public".into()],
        stmt,
        None,
    )
    .await
    .expect("plan");
    let ctx = build_ctx(server, txn);
    zyron_executor::execute(plan, &ctx).await.expect("execute")
}

/// Replays the undo log for a ROLLBACK TO SAVEPOINT exactly as the wire handler
/// does: reverse each recorded write at the heap-tuple level, then release locks
/// acquired after the savepoint. The transaction stays open.
async fn rollback_to_savepoint(server: &Arc<ServerState>, txn: &mut Transaction, name: &str) {
    let txn_id = txn.txn_id();
    let xmax = txn_id as u32;
    let rb = txn
        .rollback_to_savepoint(name)
        .unwrap_or_else(|| panic!("savepoint {name} not found"));
    for entry in &rb.undo {
        let (heap_file_id, fsm_file_id, tid, is_insert) = match entry {
            UndoEntry::ReverseInsert {
                heap_file_id,
                fsm_file_id,
                tid,
            } => (*heap_file_id, *fsm_file_id, *tid, true),
            UndoEntry::ReverseDelete {
                heap_file_id,
                fsm_file_id,
                tid,
            } => (*heap_file_id, *fsm_file_id, *tid, false),
            // this suite drives heap tables only, columnar undo is covered by
            // columnar_dml_parity_test in zyron-server
            UndoEntry::ColumnarSupersede { .. } | UndoEntry::ColumnarPatch { .. } => {
                panic!("heap only suite recorded a columnar undo entry")
            }
        };
        let heap = HeapFile::new(
            Arc::clone(&server.disk_manager),
            Arc::clone(&server.buffer_pool),
            HeapFileConfig {
                heap_file_id,
                fsm_file_id,
            },
        )
        .expect("heap");
        if is_insert {
            heap.set_xmax(tid, xmax).await.expect("set xmax");
        } else {
            heap.clear_xmax(tid).await.expect("clear xmax");
        }
    }
    server
        .txn_manager
        .lock_table()
        .unlock_after(txn_id, rb.row_lock_count);
    server
        .txn_manager
        .intent_locks()
        .unlock_after(txn_id, rb.intent_lock_count);
}

/// Takes a savepoint on the open transaction, capturing the current lock counts
/// just like the wire SAVEPOINT handler.
fn savepoint(server: &Arc<ServerState>, txn: &mut Transaction, name: &str) {
    let row = server.txn_manager.lock_table().current_count(txn.txn_id());
    let intent = server
        .txn_manager
        .intent_locks()
        .current_count(txn.txn_id());
    txn.savepoint(name.into(), row, intent);
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

fn sorted_ids(batches: &[DataBatch]) -> Vec<i64> {
    let mut v = col_i64(batches, 0);
    v.sort();
    v
}

async fn begin(server: &Arc<ServerState>) -> Transaction {
    server
        .txn_manager
        .begin(IsolationLevel::ReadCommitted)
        .expect("begin")
}

// ---------------------------------------------------------------------------

#[tokio::test]
async fn rollback_to_savepoint_undoes_insert() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_autocommit(&server, &mut session, "CREATE TABLE t (id INT)").await;
    exec_autocommit(&server, &mut session, "INSERT INTO t (id) VALUES (1)").await;

    let mut txn = begin(&server).await;
    exec_in_txn(&server, &txn, "INSERT INTO t (id) VALUES (2)").await;
    savepoint(&server, &mut txn, "sp1");
    exec_in_txn(&server, &txn, "INSERT INTO t (id) VALUES (3)").await;

    // Before rollback the transaction sees 1, 2, 3.
    assert_eq!(
        sorted_ids(&exec_in_txn(&server, &txn, "SELECT id FROM t").await),
        vec![1, 2, 3]
    );

    rollback_to_savepoint(&server, &mut txn, "sp1").await;

    // After rollback the post-savepoint insert (3) is gone; 1 and 2 remain.
    assert_eq!(
        sorted_ids(&exec_in_txn(&server, &txn, "SELECT id FROM t").await),
        vec![1, 2]
    );

    // The transaction is still open and can write then commit.
    exec_in_txn(&server, &txn, "INSERT INTO t (id) VALUES (4)").await;
    server.txn_manager.commit(&mut txn).await.expect("commit");

    let mut reader = begin(&server).await;
    assert_eq!(
        sorted_ids(&exec_in_txn(&server, &reader, "SELECT id FROM t").await),
        vec![1, 2, 4]
    );
    server
        .txn_manager
        .commit(&mut reader)
        .await
        .expect("commit");
}

#[tokio::test]
async fn rollback_to_savepoint_restores_delete() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_autocommit(&server, &mut session, "CREATE TABLE t (id INT)").await;
    exec_autocommit(&server, &mut session, "INSERT INTO t (id) VALUES (1)").await;
    exec_autocommit(&server, &mut session, "INSERT INTO t (id) VALUES (2)").await;

    let mut txn = begin(&server).await;
    savepoint(&server, &mut txn, "sp1");
    exec_in_txn(&server, &txn, "DELETE FROM t WHERE id = 1").await;
    assert_eq!(
        sorted_ids(&exec_in_txn(&server, &txn, "SELECT id FROM t").await),
        vec![2]
    );

    rollback_to_savepoint(&server, &mut txn, "sp1").await;

    // The deleted row is restored.
    assert_eq!(
        sorted_ids(&exec_in_txn(&server, &txn, "SELECT id FROM t").await),
        vec![1, 2]
    );
    server.txn_manager.commit(&mut txn).await.expect("commit");
}

#[tokio::test]
async fn rollback_to_savepoint_restores_update() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_autocommit(&server, &mut session, "CREATE TABLE t (id INT, v INT)").await;
    exec_autocommit(
        &server,
        &mut session,
        "INSERT INTO t (id, v) VALUES (1, 10)",
    )
    .await;

    let mut txn = begin(&server).await;
    savepoint(&server, &mut txn, "sp1");
    exec_in_txn(&server, &txn, "UPDATE t SET v = 99 WHERE id = 1").await;
    assert_eq!(
        col_i64(
            &exec_in_txn(&server, &txn, "SELECT v FROM t WHERE id = 1").await,
            0
        ),
        vec![99]
    );

    rollback_to_savepoint(&server, &mut txn, "sp1").await;

    // The old value is restored.
    assert_eq!(
        col_i64(
            &exec_in_txn(&server, &txn, "SELECT v FROM t WHERE id = 1").await,
            0
        ),
        vec![10]
    );
    // Exactly one row remains (the reverse-insert hid the new image, the
    // reverse-delete restored the old image).
    assert_eq!(
        sorted_ids(&exec_in_txn(&server, &txn, "SELECT id FROM t").await),
        vec![1]
    );
    server.txn_manager.commit(&mut txn).await.expect("commit");
}

#[tokio::test]
async fn nested_rollback_to_outer_undoes_inner() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_autocommit(&server, &mut session, "CREATE TABLE t (id INT)").await;

    let mut txn = begin(&server).await;
    savepoint(&server, &mut txn, "a");
    exec_in_txn(&server, &txn, "INSERT INTO t (id) VALUES (1)").await;
    savepoint(&server, &mut txn, "b");
    exec_in_txn(&server, &txn, "INSERT INTO t (id) VALUES (2)").await;

    // Rolling back to a undoes writes made after a, including those after b.
    rollback_to_savepoint(&server, &mut txn, "a").await;
    assert_eq!(
        sorted_ids(&exec_in_txn(&server, &txn, "SELECT id FROM t").await),
        Vec::<i64>::new()
    );
    server.txn_manager.commit(&mut txn).await.expect("commit");
}

#[tokio::test]
async fn release_then_rollback_outer_still_undoes_inner_writes() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_autocommit(&server, &mut session, "CREATE TABLE t (id INT)").await;

    let mut txn = begin(&server).await;
    savepoint(&server, &mut txn, "a");
    exec_in_txn(&server, &txn, "INSERT INTO t (id) VALUES (1)").await;
    savepoint(&server, &mut txn, "b");
    exec_in_txn(&server, &txn, "INSERT INTO t (id) VALUES (2)").await;

    // Releasing b keeps its undo entries in the log so an outer rollback to a
    // still reverses the b-era write.
    assert!(txn.release_savepoint("b"));
    rollback_to_savepoint(&server, &mut txn, "a").await;
    assert_eq!(
        sorted_ids(&exec_in_txn(&server, &txn, "SELECT id FROM t").await),
        Vec::<i64>::new()
    );
    server.txn_manager.commit(&mut txn).await.expect("commit");
}

#[tokio::test]
async fn writes_after_rollback_commit_only_kept_state() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_autocommit(&server, &mut session, "CREATE TABLE t (id INT)").await;

    let mut txn = begin(&server).await;
    exec_in_txn(&server, &txn, "INSERT INTO t (id) VALUES (1)").await;
    savepoint(&server, &mut txn, "sp1");
    exec_in_txn(&server, &txn, "INSERT INTO t (id) VALUES (2)").await;
    rollback_to_savepoint(&server, &mut txn, "sp1").await;
    exec_in_txn(&server, &txn, "INSERT INTO t (id) VALUES (3)").await;
    server.txn_manager.commit(&mut txn).await.expect("commit");

    // Committed state reflects 1 (pre-savepoint) and 3 (post-rollback), not the
    // rolled-back 2.
    let mut reader = begin(&server).await;
    assert_eq!(
        sorted_ids(&exec_in_txn(&server, &reader, "SELECT id FROM t").await),
        vec![1, 3]
    );
    server
        .txn_manager
        .commit(&mut reader)
        .await
        .expect("commit");
}

#[tokio::test]
async fn second_transaction_never_observes_rolled_back_rows() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_autocommit(&server, &mut session, "CREATE TABLE t (id INT)").await;

    let mut txn = begin(&server).await;
    savepoint(&server, &mut txn, "sp1");
    exec_in_txn(&server, &txn, "INSERT INTO t (id) VALUES (7)").await;
    rollback_to_savepoint(&server, &mut txn, "sp1").await;

    // A concurrent committed-read transaction never saw 7 (it was never
    // committed and is now self-deleted).
    let mut reader = begin(&server).await;
    assert_eq!(
        sorted_ids(&exec_in_txn(&server, &reader, "SELECT id FROM t").await),
        Vec::<i64>::new()
    );
    server
        .txn_manager
        .commit(&mut reader)
        .await
        .expect("commit");

    server.txn_manager.commit(&mut txn).await.expect("commit");
    let mut reader2 = begin(&server).await;
    assert_eq!(
        sorted_ids(&exec_in_txn(&server, &reader2, "SELECT id FROM t").await),
        Vec::<i64>::new()
    );
    server
        .txn_manager
        .commit(&mut reader2)
        .await
        .expect("commit");
}

#[tokio::test]
async fn locks_after_savepoint_are_released_on_rollback() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_autocommit(&server, &mut session, "CREATE TABLE t (id INT)").await;
    exec_autocommit(&server, &mut session, "INSERT INTO t (id) VALUES (1)").await;

    let mut txn = begin(&server).await;
    // Acquire a row lock before the savepoint by deleting a pre-existing row.
    let lt = server.txn_manager.lock_table();
    let table = server.catalog.get_table(_schema, "t").expect("table");
    let pre_rid = zyron_storage::TupleId::new(zyron_common::page::PageId::new(0, 0), 0);
    lt.lock_row(
        txn.txn_id(),
        table.id.0,
        pre_rid.locator(),
        zyron_storage::LockMode::Exclusive,
    )
    .expect("pre lock");
    let before = lt.current_count(txn.txn_id());

    savepoint(&server, &mut txn, "sp1");

    // Acquire two more row locks after the savepoint.
    lt.lock_row(
        txn.txn_id(),
        table.id.0,
        zyron_storage::TupleId::new(zyron_common::page::PageId::new(0, 0), 5).locator(),
        zyron_storage::LockMode::Exclusive,
    )
    .expect("lock 5");
    lt.lock_row(
        txn.txn_id(),
        table.id.0,
        zyron_storage::TupleId::new(zyron_common::page::PageId::new(0, 0), 6).locator(),
        zyron_storage::LockMode::Exclusive,
    )
    .expect("lock 6");
    assert_eq!(lt.current_count(txn.txn_id()), before + 2);

    rollback_to_savepoint(&server, &mut txn, "sp1").await;

    // The two post-savepoint locks are released; the pre-savepoint lock stays.
    assert_eq!(lt.current_count(txn.txn_id()), before);
    server.txn_manager.commit(&mut txn).await.expect("commit");
}

// A read-only transaction allows reads but rejects writes at the write operator,
// the universal enforcement point, so a write cannot reach the heap through any
// execution path.
#[tokio::test]
async fn read_only_transaction_rejects_writes_at_write_operator() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_autocommit(&server, &mut session, "CREATE TABLE t (id INT)").await;
    exec_autocommit(&server, &mut session, "INSERT INTO t (id) VALUES (1)").await;

    let mut txn = begin(&server).await;
    txn.set_read_only(true);

    // A read is allowed in a read-only transaction.
    assert_eq!(
        sorted_ids(
            &try_exec_in_txn(&server, &txn, "SELECT id FROM t")
                .await
                .expect("read allowed")
        ),
        vec![1]
    );

    // A write is rejected through the read-only execution context.
    let err = try_exec_in_txn(&server, &txn, "INSERT INTO t (id) VALUES (2)")
        .await
        .expect_err("read-only INSERT must be rejected");
    assert!(
        err.to_string().contains("read-only"),
        "unexpected error: {err}"
    );

    server.txn_manager.abort(&mut txn).expect("abort");
}

//! Database-wide MERGE BRANCH must carry lake-table branch writes to main.
//!
//! The per-table form MERGE BRANCH b INTO main FOR TABLE t replays a lake
//! branch's file set onto the table's main log. The database-wide form
//! merged heap pages and columnar overlays, then deleted the branch, so a
//! lake table's branch writes were silently lost.
//!
//! Run: cargo test -p zyron-wire --test merge_branch_lake_test

use std::sync::Arc;

use zyron_buffer::BufferPool;
use zyron_catalog::{Catalog, CatalogCache, DatabaseId, HeapCatalogStorage, SYSTEM_DATABASE_ID};
use zyron_executor::batch::DataBatch;
use zyron_executor::context::ExecutionContext;
use zyron_storage::DiskManager;
use zyron_storage::txn::{IsolationLevel, TransactionManager};
use zyron_wal::WalWriter;
use zyron_wire::connection::ServerState;
use zyron_wire::session::Session;

/// Per-test state carrying the session's active branch, mirrored into each
/// query's execution context the way the wire connection does.
struct Harness {
    server: Arc<ServerState>,
    session: Option<Session>,
    active_branch: Option<String>,
    _tmp: tempfile::TempDir,
}

async fn create_harness() -> Harness {
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let (data_dir, wal_dir) = zyron_bench_harness::create_dirs(tmp.path()).expect("dirs");

    let wal = Arc::new(WalWriter::new(zyron_bench_harness::wal_config(&wal_dir)).expect("wal"));
    let disk = Arc::new(
        DiskManager::new(zyron_bench_harness::disk_config(&data_dir))
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
    catalog
        .create_schema(SYSTEM_DATABASE_ID, "public", "test_user")
        .await
        .expect("create public schema");
    let txn_manager = Arc::new(TransactionManager::new(Arc::clone(&wal)));
    let branch_manager = Arc::new(zyron_versioning::BranchManager::new(
        tmp.path().to_path_buf(),
    ));

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
        branch_manager: Some(branch_manager),
        fts_manager: None,
        vector_manager: None,
        graph_manager: None,
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

    let mut session = Session::new("test_user".into(), "testdb".into(), DatabaseId(1));
    session.search_path = vec!["public".into()];

    Harness {
        server: state,
        session: Some(session),
        active_branch: None,
        _tmp: tmp,
    }
}

/// Runs one statement through DDL dispatch or plan + execute, with the
/// active branch mirrored into the context for both the heap and lake
/// sides, and lake versions published after the durable commit.
async fn exec(h: &mut Harness, sql: &str) -> Vec<DataBatch> {
    let stmt = zyron_parser::parse(sql)
        .expect("parse")
        .into_iter()
        .next()
        .expect("one statement");

    let mut txn_opt: Option<zyron_storage::txn::Transaction> = None;
    if let Some(res) = zyron_wire::ddl_dispatch::try_handle_ddl_utility(
        &stmt,
        &h.server,
        &mut h.session,
        &mut txn_opt,
        &mut h.active_branch,
        sql,
    )
    .await
    {
        res.expect("ddl handler failed");
        return Vec::new();
    }

    let plan = zyron_planner::plan(
        &h.server.catalog,
        DatabaseId(1),
        vec!["public".into()],
        stmt,
        None,
    )
    .await
    .expect("plan");
    let mut txn = h
        .server
        .txn_manager
        .begin(IsolationLevel::ReadCommitted)
        .expect("begin");
    let snapshot = txn.snapshot.clone();
    let txn_id = txn.txn_id;
    let mut ctx = ExecutionContext::new(
        h.server.catalog.clone(),
        h.server.wal.clone(),
        h.server.buffer_pool.clone(),
        h.server.disk_manager.clone(),
        txn_id as u32,
        snapshot,
    );
    ctx.heap_files = Some(Arc::clone(&h.server.heap_files));
    ctx.btree_indexes = Some(Arc::clone(&h.server.btree_indexes));
    if let Some(mgr) = &h.server.branch_manager {
        ctx.branch_catalog = Some(Arc::clone(mgr) as Arc<dyn zyron_common::BranchCatalog>);
        if let Some(name) = &h.active_branch {
            ctx.active_branch_id = mgr.get_branch_by_name(name).ok().map(|e| e.id.0);
        }
    }
    ctx.active_branch_name = h.active_branch.clone();
    let ctx = Arc::new(ctx);
    let batches = zyron_executor::execute(plan, &ctx).await.expect("execute");
    h.server.txn_manager.commit(&mut txn).await.expect("commit");
    let logs =
        zyron_lake::publish_txn(h.server.disk_manager.data_dir(), txn_id).expect("publish");
    zyron_wire::connection::refresh_lake_stats(&h.server, &logs);
    batches
}

fn total_rows(batches: &[DataBatch]) -> usize {
    batches.iter().map(|b| b.num_rows).sum()
}

/// The FK probe against a lake parent resolves through the session's
/// branch, so a parent row the branch inserted satisfies the reference.
#[tokio::test]
async fn lake_parent_fk_probe_reads_the_branch_head() {
    let mut h = create_harness().await;
    exec(
        &mut h,
        "CREATE TABLE p (k BIGINT NOT NULL) USING ZYRONLAKE",
    )
    .await;
    exec(
        &mut h,
        "CREATE TABLE c (id INT NOT NULL, k BIGINT, FOREIGN KEY (k) REFERENCES p(k))",
    )
    .await;
    exec(&mut h, "INSERT INTO p VALUES (1)").await;

    exec(&mut h, "CREATE BRANCH dev").await;
    exec(&mut h, "USE BRANCH dev").await;

    // A parent key only the branch holds still satisfies the reference
    exec(&mut h, "INSERT INTO p VALUES (7)").await;
    exec(&mut h, "INSERT INTO c VALUES (10, 7)").await;
    let rows = exec(&mut h, "SELECT id FROM c").await;
    assert_eq!(total_rows(&rows), 1, "the branch-parented child landed");
}

/// A lake table's branch writes survive the database-wide merge into main.
#[tokio::test]
async fn database_wide_merge_carries_lake_branch_writes() {
    let mut h = create_harness().await;
    exec(
        &mut h,
        "CREATE TABLE lk (id BIGINT NOT NULL, v BIGINT) USING ZYRONLAKE",
    )
    .await;
    exec(&mut h, "INSERT INTO lk VALUES (1, 10)").await;

    exec(&mut h, "CREATE BRANCH dev").await;
    exec(&mut h, "USE BRANCH dev").await;
    exec(&mut h, "INSERT INTO lk VALUES (2, 20)").await;

    // The branch sees both rows, main still sees one
    let rows = exec(&mut h, "SELECT id FROM lk").await;
    assert_eq!(total_rows(&rows), 2, "branch sees its own write");
    h.active_branch = None;
    let rows = exec(&mut h, "SELECT id FROM lk").await;
    assert_eq!(total_rows(&rows), 1, "main is untouched before the merge");

    exec(&mut h, "MERGE BRANCH dev INTO main").await;

    let rows = exec(&mut h, "SELECT id FROM lk").await;
    assert_eq!(
        total_rows(&rows),
        2,
        "the branch's lake write reaches main through the database-wide merge"
    );
}

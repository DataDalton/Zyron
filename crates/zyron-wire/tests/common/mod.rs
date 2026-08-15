//! Server, session and statement helpers shared by the wire integration
//! suites.
//!
//! One test binary per file means each one would otherwise carry its own
//! copy of the `ServerState` construction below. That literal names every
//! subsystem the server holds, so two copies drift the moment a field is
//! added, and a suite built on a stale copy tests a server nobody runs.

#![allow(dead_code)]

use std::sync::Arc;

use zyron_buffer::BufferPool;
use zyron_catalog::{
    Catalog, CatalogCache, DatabaseId, HeapCatalogStorage, SYSTEM_DATABASE_ID, SchemaId,
};
use zyron_executor::column::ScalarValue;
use zyron_storage::DiskManager;
use zyron_storage::txn::TransactionManager;
use zyron_wal::WalWriter;
use zyron_wire::connection::ServerState;
use zyron_wire::session::Session;

pub async fn create_test_server() -> (Arc<ServerState>, SchemaId, tempfile::TempDir) {
    create_test_server_in_mode(zyron_common::DeploymentMode::Unified).await
}

pub async fn create_test_server_in_mode(
    mode: zyron_common::DeploymentMode,
) -> (Arc<ServerState>, SchemaId, tempfile::TempDir) {
    create_test_server_configured(mode, None).await
}

/// A server whose buffer pool is smaller than the one it ships with.
///
/// For the one class of test whose subject IS the pool size: what the engine
/// does when a page it needs is not resident. Reaching that state with the
/// shipped 8192 frames would take a table larger than any test should build.
/// Nothing else may pass a frame count, per the rule that a test runs the
/// engine that ships unless the value under test is the value being varied
pub async fn create_test_server_with_pool_frames(
    frames: usize,
) -> (Arc<ServerState>, SchemaId, tempfile::TempDir) {
    create_test_server_configured(zyron_common::DeploymentMode::Unified, Some(frames)).await
}

async fn create_test_server_configured(
    mode: zyron_common::DeploymentMode,
    pool_frames: Option<usize>,
) -> (Arc<ServerState>, SchemaId, tempfile::TempDir) {
    // The engine the server ships, so what is measured here is what runs.
    // Only the directories differ, because a run needs its own
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let (data_dir, wal_dir) = zyron_bench_harness::create_dirs(tmp.path()).expect("dirs");

    let wal = Arc::new(WalWriter::new(zyron_bench_harness::wal_config(&wal_dir)).expect("wal"));
    let disk = Arc::new(
        DiskManager::new(zyron_bench_harness::disk_config(&data_dir))
            .await
            .expect("disk"),
    );
    let pool_config = match pool_frames {
        None => zyron_bench_harness::buffer_pool_config(),
        Some(num_frames) => zyron_buffer::BufferPoolConfig { num_frames },
    };
    let pool = Arc::new(BufferPool::new(pool_config));
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
        // The search managers the server builds, so a suite can exercise a
        // fulltext, vector or spatial index rather than only the two formats'
        // heap paths. Each gets its own directory under the run's temp root,
        // which is the only thing that differs from what ships
        fts_manager: Some(Arc::new(zyron_search::FtsManager::with_data_dir(
            data_dir.join("fts"),
        ))),
        vector_manager: Some(Arc::new(
            zyron_search::vector::VectorIndexManager::with_data_dir(data_dir.join("vector")),
        )),
        graph_manager: Some(Arc::new(zyron_search::graph::GraphManager::new())),
        spatial_manager: Some(Arc::new(
            zyron_types::spatial_index::SpatialIndexManager::new(),
        )),
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
        deployment_mode: mode,
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

pub fn new_session() -> Option<Session> {
    let mut s = Session::new("test_user".into(), "testdb".into(), DatabaseId(1));
    s.search_path = vec!["public".into()];
    Some(s)
}

pub async fn exec_ddl(
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
    sql: &str,
) -> Result<(), String> {
    let stmt = zyron_parser::parse(sql)
        .expect("parse")
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
        Some(Ok(_)) => Ok(()),
        Some(Err(e)) => Err(format!("{e:?}")),
        None => Err(format!("statement was not handled as DDL: {sql}")),
    }
}

pub async fn exec_dml(server: &Arc<ServerState>, sql: &str) {
    exec_dml_result(server, sql).await.expect("execute");
}

/// Runs several DML statements inside one transaction, for a test whose
/// subject is what a statement sees of its own transaction's earlier writes.
/// The first failure aborts and is returned.
pub async fn exec_dml_script(
    server: &Arc<ServerState>,
    statements: &[&str],
) -> Result<(), zyron_common::ZyronError> {
    let mut txn = server
        .txn_manager
        .begin(zyron_storage::txn::IsolationLevel::ReadCommitted)
        .expect("begin");
    let txn_id = txn.txn_id;
    for sql in statements {
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
        // A fresh snapshot per statement, which is what the wire layer gives
        // each statement of a read-committed transaction
        let snapshot = server.txn_manager.refresh_snapshot(&txn);
        let mut ctx = zyron_executor::context::ExecutionContext::new(
            server.catalog.clone(),
            server.wal.clone(),
            server.buffer_pool.clone(),
            server.disk_manager.clone(),
            txn_id as u32,
            snapshot,
        );
        ctx.heap_files = Some(Arc::clone(&server.heap_files));
        ctx.btree_indexes = Some(Arc::clone(&server.btree_indexes));
        ctx.doc_registry = Some(Arc::clone(&server.doc_registry));
        let ctx = Arc::new(ctx);
        if let Err(e) = zyron_executor::execute(plan, &ctx).await {
            let _ = zyron_lake::abandon_txn(server.disk_manager.data_dir(), txn_id);
            let _ = server.txn_manager.abort(&mut txn);
            return Err(e);
        }
    }
    server.txn_manager.commit(&mut txn).await.expect("commit");
    let logs = zyron_lake::publish_txn(server.disk_manager.data_dir(), txn_id).expect("publish");
    zyron_wire::connection::refresh_lake_stats(server, &logs);
    Ok(())
}

/// Runs one DML statement and returns what it produced, for a test whose
/// subject is the statement being refused
pub async fn exec_dml_result(
    server: &Arc<ServerState>,
    sql: &str,
) -> Result<(), zyron_common::ZyronError> {
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
    let mut txn = server
        .txn_manager
        .begin(zyron_storage::txn::IsolationLevel::ReadCommitted)
        .expect("begin");
    let snapshot = txn.snapshot.clone();
    let txn_id = txn.txn_id;
    let mut ctx = zyron_executor::context::ExecutionContext::new(
        server.catalog.clone(),
        server.wal.clone(),
        server.buffer_pool.clone(),
        server.disk_manager.clone(),
        txn_id as u32,
        snapshot,
    );
    ctx.heap_files = Some(Arc::clone(&server.heap_files));
    ctx.btree_indexes = Some(Arc::clone(&server.btree_indexes));
    ctx.table_io_stats = Some(Arc::clone(&server.table_io_stats));
    ctx.index_io_stats = Some(Arc::clone(&server.index_io_stats));
    ctx.doc_registry = Some(Arc::clone(&server.doc_registry));
    if let Some(mgr) = &server.fts_manager {
        ctx.set_fts_manager(Arc::clone(mgr));
    }
    if let Some(mgr) = &server.vector_manager {
        ctx.set_vector_manager(Arc::clone(mgr));
    }
    if let Some(mgr) = &server.spatial_manager {
        ctx.set_spatial_manager(Arc::clone(mgr));
    }
    let ctx = Arc::new(ctx);
    let outcome = zyron_executor::execute(plan, &ctx).await.map(|_| ());
    if outcome.is_err() {
        // A refused statement leaves its lake versions pending, and
        // abandoning them keeps the next statement's base clean
        let _ = zyron_lake::abandon_txn(server.disk_manager.data_dir(), txn_id);
        let _ = server.txn_manager.abort(&mut txn);
        return outcome;
    }
    server.txn_manager.commit(&mut txn).await.expect("commit");
    // Mirrors the wire layer, lake versions publish after the durable commit
    // and the manifest's statistics reach the planner with them
    let logs = zyron_lake::publish_txn(server.disk_manager.data_dir(), txn_id).expect("publish");
    zyron_wire::connection::refresh_lake_stats(server, &logs);
    outcome
}

pub async fn query_rows(server: &Arc<ServerState>, sql: &str) -> usize {
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
    let mut txn = server
        .txn_manager
        .begin(zyron_storage::txn::IsolationLevel::ReadCommitted)
        .expect("begin");
    let snapshot = txn.snapshot.clone();
    let txn_id = txn.txn_id as u32;
    let mut ctx = zyron_executor::context::ExecutionContext::new(
        server.catalog.clone(),
        server.wal.clone(),
        server.buffer_pool.clone(),
        server.disk_manager.clone(),
        txn_id,
        snapshot,
    );
    ctx.heap_files = Some(Arc::clone(&server.heap_files));
    ctx.btree_indexes = Some(Arc::clone(&server.btree_indexes));
    ctx.table_io_stats = Some(Arc::clone(&server.table_io_stats));
    ctx.index_io_stats = Some(Arc::clone(&server.index_io_stats));
    ctx.doc_registry = Some(Arc::clone(&server.doc_registry));
    if let Some(mgr) = &server.fts_manager {
        ctx.set_fts_manager(Arc::clone(mgr));
    }
    if let Some(mgr) = &server.vector_manager {
        ctx.set_vector_manager(Arc::clone(mgr));
    }
    if let Some(mgr) = &server.spatial_manager {
        ctx.set_spatial_manager(Arc::clone(mgr));
    }
    let ctx = Arc::new(ctx);
    let batches = zyron_executor::execute(plan, &ctx).await.expect("execute");
    server.txn_manager.commit(&mut txn).await.expect("commit");
    batches.iter().map(|b| b.num_rows).sum()
}

/// The message a query fails with, for the cases where the refusal is the
/// behavior under test. Panics when the query succeeds, so a test cannot pass
/// by silently getting an answer where it expected an error.
pub async fn query_error(server: &Arc<ServerState>, sql: &str) -> String {
    let stmt = zyron_parser::parse(sql)
        .expect("parse")
        .into_iter()
        .next()
        .expect("one statement");
    let plan = match zyron_planner::plan(
        &server.catalog,
        DatabaseId(1),
        vec!["public".into()],
        stmt,
        None,
    )
    .await
    {
        Ok(plan) => plan,
        Err(e) => return e.to_string(),
    };
    let mut txn = server
        .txn_manager
        .begin(zyron_storage::txn::IsolationLevel::ReadCommitted)
        .expect("begin");
    let snapshot = txn.snapshot.clone();
    let txn_id = txn.txn_id as u32;
    let mut ctx = zyron_executor::context::ExecutionContext::new(
        server.catalog.clone(),
        server.wal.clone(),
        server.buffer_pool.clone(),
        server.disk_manager.clone(),
        txn_id,
        snapshot,
    );
    ctx.heap_files = Some(Arc::clone(&server.heap_files));
    ctx.btree_indexes = Some(Arc::clone(&server.btree_indexes));
    ctx.doc_registry = Some(Arc::clone(&server.doc_registry));
    let ctx = Arc::new(ctx);
    let result = zyron_executor::execute(plan, &ctx).await;
    let _ = server.txn_manager.commit(&mut txn).await;
    match result {
        Ok(batches) => panic!(
            "expected {sql} to fail, got {} rows",
            batches.iter().map(|b| b.num_rows).sum::<usize>()
        ),
        Err(e) => e.to_string(),
    }
}

/// Runs a query and returns whatever it produced, for a suite that reports
/// which of a set of queries the engine refused rather than stopping at the
/// first one.
pub async fn query_result(
    server: &Arc<ServerState>,
    sql: &str,
) -> Result<Vec<Vec<ScalarValue>>, String> {
    let stmt = zyron_parser::parse(sql)
        .map_err(|e| format!("parse: {e}"))?
        .into_iter()
        .next()
        .ok_or_else(|| "parse produced no statement".to_string())?;
    let plan = zyron_planner::plan(
        &server.catalog,
        DatabaseId(1),
        vec!["public".into()],
        stmt,
        None,
    )
    .await
    .map_err(|e| format!("plan: {e}"))?;
    let mut txn = server
        .txn_manager
        .begin(zyron_storage::txn::IsolationLevel::ReadCommitted)
        .expect("begin");
    let snapshot = txn.snapshot.clone();
    let txn_id = txn.txn_id as u32;
    let mut ctx = zyron_executor::context::ExecutionContext::new(
        server.catalog.clone(),
        server.wal.clone(),
        server.buffer_pool.clone(),
        server.disk_manager.clone(),
        txn_id,
        snapshot,
    );
    ctx.heap_files = Some(Arc::clone(&server.heap_files));
    ctx.btree_indexes = Some(Arc::clone(&server.btree_indexes));
    ctx.doc_registry = Some(Arc::clone(&server.doc_registry));
    let ctx = Arc::new(ctx);
    let result = zyron_executor::execute(plan, &ctx).await;
    let _ = server.txn_manager.commit(&mut txn).await;
    match result {
        Ok(batches) => Ok(batches
            .iter()
            .flat_map(|b| {
                (0..b.num_rows)
                    .map(|r| b.columns.iter().map(|c| c.get_scalar(r)).collect())
                    .collect::<Vec<Vec<ScalarValue>>>()
            })
            .collect()),
        Err(e) => Err(format!("execute: {e}")),
    }
}

/// Every row a query returned, as scalars, so two runs can be compared
/// value by value rather than only by how many rows each produced.
///
/// A row count matching is not the answers matching, and a timing taken
/// against a different answer measures a cheaper wrong thing
pub async fn query_values(server: &Arc<ServerState>, sql: &str) -> Vec<Vec<ScalarValue>> {
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
    let mut txn = server
        .txn_manager
        .begin(zyron_storage::txn::IsolationLevel::ReadCommitted)
        .expect("begin");
    let snapshot = txn.snapshot.clone();
    let txn_id = txn.txn_id as u32;
    let mut ctx = zyron_executor::context::ExecutionContext::new(
        server.catalog.clone(),
        server.wal.clone(),
        server.buffer_pool.clone(),
        server.disk_manager.clone(),
        txn_id,
        snapshot,
    );
    ctx.heap_files = Some(Arc::clone(&server.heap_files));
    ctx.btree_indexes = Some(Arc::clone(&server.btree_indexes));
    ctx.table_io_stats = Some(Arc::clone(&server.table_io_stats));
    ctx.index_io_stats = Some(Arc::clone(&server.index_io_stats));
    ctx.doc_registry = Some(Arc::clone(&server.doc_registry));
    if let Some(mgr) = &server.fts_manager {
        ctx.set_fts_manager(Arc::clone(mgr));
    }
    if let Some(mgr) = &server.vector_manager {
        ctx.set_vector_manager(Arc::clone(mgr));
    }
    if let Some(mgr) = &server.spatial_manager {
        ctx.set_spatial_manager(Arc::clone(mgr));
    }
    let ctx = Arc::new(ctx);
    let batches = zyron_executor::execute(plan, &ctx).await.expect("execute");
    server.txn_manager.commit(&mut txn).await.expect("commit");

    let mut rows = Vec::new();
    for batch in &batches {
        for r in 0..batch.num_rows {
            rows.push(batch.columns.iter().map(|c| c.get_scalar(r)).collect());
        }
    }
    rows
}

/// Runs one statement with the session bound to a branch, the way a
/// connection does after USE BRANCH, and returns its rows.
///
/// Branch DML has no per-statement qualifier, so the session binding is the
/// only way to reach it and a test that skipped it would exercise main
pub async fn run_on_branch(
    server: &Arc<ServerState>,
    sql: &str,
    branch: &str,
) -> Result<Vec<Vec<ScalarValue>>, zyron_common::ZyronError> {
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
    .await?;
    let mut txn = server
        .txn_manager
        .begin(zyron_storage::txn::IsolationLevel::ReadCommitted)
        .expect("begin");
    let snapshot = txn.snapshot.clone();
    let txn_id = txn.txn_id;
    let mut ctx = zyron_executor::context::ExecutionContext::new(
        server.catalog.clone(),
        server.wal.clone(),
        server.buffer_pool.clone(),
        server.disk_manager.clone(),
        txn_id as u32,
        snapshot,
    );
    ctx.heap_files = Some(Arc::clone(&server.heap_files));
    ctx.btree_indexes = Some(Arc::clone(&server.btree_indexes));
    ctx.table_io_stats = Some(Arc::clone(&server.table_io_stats));
    ctx.index_io_stats = Some(Arc::clone(&server.index_io_stats));
    ctx.doc_registry = Some(Arc::clone(&server.doc_registry));
    ctx.active_branch_name = Some(branch.to_string());
    if let Some(mgr) = &server.fts_manager {
        ctx.set_fts_manager(Arc::clone(mgr));
    }
    if let Some(mgr) = &server.vector_manager {
        ctx.set_vector_manager(Arc::clone(mgr));
    }
    if let Some(mgr) = &server.spatial_manager {
        ctx.set_spatial_manager(Arc::clone(mgr));
    }
    let ctx = Arc::new(ctx);
    match zyron_executor::execute(plan, &ctx).await {
        Ok(batches) => {
            server.txn_manager.commit(&mut txn).await.expect("commit");
            let logs =
                zyron_lake::publish_txn(server.disk_manager.data_dir(), txn_id).expect("publish");
            zyron_wire::connection::refresh_lake_stats(server, &logs);
            let mut rows = Vec::new();
            for batch in &batches {
                for r in 0..batch.num_rows {
                    rows.push(batch.columns.iter().map(|c| c.get_scalar(r)).collect());
                }
            }
            Ok(rows)
        }
        Err(e) => {
            let _ = zyron_lake::abandon_txn(server.disk_manager.data_dir(), txn_id);
            let _ = server.txn_manager.abort(&mut txn);
            Err(e)
        }
    }
}

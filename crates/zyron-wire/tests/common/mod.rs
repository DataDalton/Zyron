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
    let (server, schema, _, tmp) =
        create_test_server_configured(mode, None, false, false, false).await;
    (server, schema, tmp)
}

/// A server with the security manager installed, for a test whose subject is
/// a privilege gate. Nothing is granted, the test grants exactly what its
/// scenario names.
pub async fn create_test_server_with_security() -> (
    Arc<ServerState>,
    SchemaId,
    Arc<zyron_auth::SecurityManager>,
    tempfile::TempDir,
) {
    let (server, schema, sm, tmp) = create_test_server_configured(
        zyron_common::DeploymentMode::Unified,
        None,
        true,
        false,
        false,
    )
    .await;
    (server, schema, sm.expect("security manager built"), tmp)
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
    let (server, schema, _, tmp) = create_test_server_configured(
        zyron_common::DeploymentMode::Unified,
        Some(frames),
        false,
        false,
        false,
    )
    .await;
    (server, schema, tmp)
}

/// A server with the branch manager installed, for a suite whose subject is a
/// write under an active branch. A branch routes heap pages through a
/// copy-on-write overlay, so nothing reaches that path without this.
pub async fn create_test_server_with_branches() -> (Arc<ServerState>, SchemaId, tempfile::TempDir) {
    let (server, schema, _, tmp) = create_test_server_configured(
        zyron_common::DeploymentMode::Unified,
        None,
        false,
        true,
        false,
    )
    .await;
    (server, schema, tmp)
}

/// A server with both the branch manager and the change feeds, for a suite
/// whose subject is what a branch records and reads of a table's changes
pub async fn create_test_server_with_cdc_and_branches()
-> (Arc<ServerState>, SchemaId, tempfile::TempDir) {
    let (server, schema, _, tmp) = create_test_server_configured(
        zyron_common::DeploymentMode::Unified,
        None,
        false,
        true,
        true,
    )
    .await;
    (server, schema, tmp)
}

/// A server whose change data feeds are open, for a suite whose subject is a
/// change stream or `table_changes`.
///
/// The feed registry, the reader the executor asks for changes through and
/// the facts the planner resolves a change window with are all installed, so
/// what a test runs is the engine a server with CDC enabled runs
pub async fn create_test_server_with_cdc() -> (Arc<ServerState>, SchemaId, tempfile::TempDir) {
    let (server, schema, _, tmp) = create_test_server_configured(
        zyron_common::DeploymentMode::Unified,
        None,
        false,
        false,
        true,
    )
    .await;
    (server, schema, tmp)
}

/// A server with both the security manager and the change feeds, for a suite
/// whose subject is what one principal may read of another's changes
pub async fn create_test_server_with_cdc_and_security() -> (
    Arc<ServerState>,
    SchemaId,
    Arc<zyron_auth::SecurityManager>,
    tempfile::TempDir,
) {
    let (server, schema, sm, tmp) = create_test_server_configured(
        zyron_common::DeploymentMode::Unified,
        None,
        true,
        false,
        true,
    )
    .await;
    (server, schema, sm.expect("security manager built"), tmp)
}

async fn create_test_server_configured(
    mode: zyron_common::DeploymentMode,
    pool_frames: Option<usize>,
    with_security: bool,
    with_branches: bool,
    with_cdc: bool,
) -> (
    Arc<ServerState>,
    SchemaId,
    Option<Arc<zyron_auth::SecurityManager>>,
    tempfile::TempDir,
) {
    // The engine the server ships, so what is measured here is what runs.
    // Only the directories differ, because a run needs its own
    let tmp = tempfile::TempDir::new().expect("temp dir");
    build_test_server(
        tmp,
        mode,
        pool_frames,
        with_security,
        with_branches,
        with_cdc,
        true,
    )
    .await
}

/// Opens a server with change feeds again over the directories an earlier
/// one wrote, the way a restart does. The write-ahead log is recovered, the
/// catalog replays what it had not flushed, every transaction that committed
/// reads as committed, and every feed a table records into is open again.
///
/// The earlier server must have been dropped, so this is the only writer
/// over the log
pub async fn reopen_test_server_with_cdc(
    tmp: tempfile::TempDir,
) -> (Arc<ServerState>, SchemaId, tempfile::TempDir) {
    let (server, schema, _, tmp) = build_test_server(
        tmp,
        zyron_common::DeploymentMode::Unified,
        None,
        false,
        false,
        true,
        false,
    )
    .await;
    (server, schema, tmp)
}

async fn build_test_server(
    tmp: tempfile::TempDir,
    mode: zyron_common::DeploymentMode,
    pool_frames: Option<usize>,
    with_security: bool,
    with_branches: bool,
    with_cdc: bool,
    fresh: bool,
) -> (
    Arc<ServerState>,
    SchemaId,
    Option<Arc<zyron_auth::SecurityManager>>,
    tempfile::TempDir,
) {
    let (data_dir, wal_dir) = zyron_bench_harness::create_dirs(tmp.path()).expect("dirs");

    // A reopen recovers the log first, the way the server does, so the
    // transactions that committed before the stop read as committed and the
    // ones that had not read as aborted
    let recovered = if fresh {
        None
    } else {
        Some(
            zyron_wal::RecoveryManager::new(&wal_dir)
                .expect("the log opens")
                .recover()
                .expect("the log recovers"),
        )
    };
    let start_txn_id = recovered
        .as_ref()
        .map(|r| {
            let max_committed = r
                .committed_txns
                .iter()
                .map(|&(txn_id, _)| txn_id)
                .max()
                .unwrap_or(0);
            let max_undo = r.undo_txns.iter().copied().max().unwrap_or(0);
            max_committed.max(max_undo) + 1
        })
        .unwrap_or(1);

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
    // The hook the server installs at startup. A pool without one cannot
    // evict a dirty page, so a suite that skipped it would run against an
    // engine that stalls where the product writes
    zyron_bench_harness::install_evict_writer(&pool, &disk, Some(&wal));
    let storage =
        Arc::new(HeapCatalogStorage::new(Arc::clone(&disk), Arc::clone(&pool)).expect("storage"));
    let cache = Arc::new(CatalogCache::new(256, 64));
    let catalog = Arc::new(
        Catalog::new(storage, cache, Arc::clone(&wal))
            .await
            .expect("catalog"),
    );
    // The server registers the zyron_sys catalog before it accepts a
    // connection, so a harness that skipped it would run tests against an
    // engine that does not exist
    zyron_catalog::SystemCatalog::init(&catalog)
        .await
        .expect("register the zyron_sys catalog");
    let public_schema = if fresh {
        catalog
            .create_schema(SYSTEM_DATABASE_ID, "zyron_test", "test_user")
            .await
            .expect("create zyron_test schema")
    } else {
        catalog
            .get_schema(SYSTEM_DATABASE_ID, "zyron_test")
            .expect("the test schema survives a reopen")
            .id
    };
    let txn_manager = Arc::new(TransactionManager::with_start_txn_id(
        Arc::clone(&wal),
        start_txn_id,
    ));
    if let Some(recovered) = recovered.as_ref() {
        let status_map = txn_manager.status_map();
        status_map.load(&data_dir).expect("the commit log loads");
        for &(txn_id, commit_lsn) in &recovered.committed_txns {
            status_map.record_committed_at(txn_id, commit_lsn);
        }
        for &txn_id in &recovered.undo_txns {
            status_map.record_aborted(txn_id);
        }
    }

    // The server installs a presign secret at startup, so PRESIGNED_URL works
    // against a test server the same way it works against what ships
    zyron_executor::media_runtime::install_presign_secret([7u8; 32]);

    // The branches of the database, which a change read inside a branch
    // names a lake table's head by
    let branch_manager = with_branches.then(|| {
        Arc::new(zyron_versioning::BranchManager::new(
            tmp.path().to_path_buf(),
        ))
    });
    // The change feed registry the server opens at startup, plus the hook
    // that captures into it, so a test through this harness records changes
    // exactly as a running server does
    let cdc_registry =
        with_cdc.then(|| Arc::new(zyron_cdc::CdfRegistry::new(data_dir.to_path_buf())));
    // Every table recording changes records again, the way the server
    // reopens its feeds before accepting a connection. A lake table's source
    // is its log, which a reopen inside one process still holds in the
    // shared registry, so its changes are followed from the same instance
    // the server would register after opening the logs at startup
    if let Some(registry) = cdc_registry.as_ref() {
        // The appends the log holds go back into the segment files before
        // any feed opens, and every feed records its appends in the log,
        // the way the server wires its feeds
        if let Some(recovered) = recovered.as_ref() {
            zyron_cdc::restore_logged_frames(&data_dir, &recovered.feed_records)
                .expect("the logged appends lay back into their segments");
        }
        registry.attach_wal(&wal);
        let feeds = Arc::clone(registry);
        wal.add_retention_hook(Arc::new(move || feeds.retained_lsn()));
        zyron_wire::change_feed_bridge::reopen_feeds(&catalog, registry).expect("the feeds reopen");
        zyron_wire::change_feed_bridge::reopen_lake_sources(
            &catalog,
            registry,
            branch_manager.as_deref(),
        )
        .expect("the lake sources register");
    }
    // The commit chains this node holds, opened the way the server opens
    // them, so a verified table in a suite chains its commits exactly as
    // one on a running server does
    let chain_registry = Arc::new(
        zyron_lifecycle::verify::ChainRegistry::open(data_dir.to_path_buf())
            .expect("the commit chains open"),
    );
    zyron_wire::verify_dispatch::register_chained_tables(&chain_registry, &catalog);
    if let Some(recovered) = recovered.as_ref() {
        zyron_wire::verify_dispatch::restore_chain_entries(
            &chain_registry,
            &recovered.chain_records,
        )
        .expect("the logged chain entries go back into their chains");
    }
    // The registry the WORM and legal hold hook reads. One instance, shared
    // with the hook, so a hold placed through a statement is the one the
    // hook consults
    let legal_holds = Arc::new(zyron_lifecycle::legal_hold::LegalHoldRegistry::new());
    let cdc_hook: Option<Arc<dyn zyron_executor::context::CdcHook>> =
        cdc_registry.as_ref().map(|registry| {
            Arc::new(
                zyron_wire::dml_hooks::CdcHookBridge::new(Arc::clone(registry))
                    .with_catalog(Arc::clone(&catalog)),
            ) as Arc<dyn zyron_executor::context::CdcHook>
        });
    // The outbound stream and replication slot managers the server opens
    // beside the feeds, so a suite can create an outbound stream and read
    // where its position lives
    let cdc_stream_manager = with_cdc
        .then(|| {
            zyron_cdc::CdcStreamManager::new(&data_dir)
                .ok()
                .map(Arc::new)
        })
        .flatten();
    let slot_manager = with_cdc
        .then(|| {
            zyron_cdc::SlotManager::open(&data_dir, zyron_cdc::SlotLagConfig::default())
                .ok()
                .map(Arc::new)
        })
        .flatten();
    let cdc_stream_stats: Option<Arc<dyn Fn() -> Vec<(String, u32, bool, String)> + Send + Sync>> =
        cdc_stream_manager.as_ref().map(|mgr| {
            let mgr = Arc::clone(mgr);
            Arc::new(move || -> Vec<(String, u32, bool, String)> {
                mgr.list_streams()
                    .into_iter()
                    .map(|s| (s.name, s.table_id, s.active, s.change_stream))
                    .collect()
            }) as Arc<dyn Fn() -> Vec<(String, u32, bool, String)> + Send + Sync>
        });
    let cdc_slot_stats: Option<
        Arc<dyn Fn() -> Vec<(String, String, u64, u64, bool, u64)> + Send + Sync>,
    > = slot_manager.as_ref().map(|mgr| {
        let mgr = Arc::clone(mgr);
        Arc::new(move || -> Vec<(String, String, u64, u64, bool, u64)> {
            mgr.list_slots()
                .into_iter()
                .map(|s| {
                    (
                        s.name,
                        format!("{:?}", s.plugin),
                        s.confirmed_lsn,
                        s.restart_lsn,
                        s.active,
                        0,
                    )
                })
                .collect()
        }) as Arc<dyn Fn() -> Vec<(String, String, u64, u64, bool, u64)> + Send + Sync>
    });
    // The planner resolves a change window against the feeds, so a suite that
    // reads changes needs the same facts a server installs
    if let Some(registry) = cdc_registry.as_ref() {
        zyron_planner::install_change_feed_facts_for(
            data_dir.to_path_buf(),
            Arc::new(zyron_wire::change_feed_bridge::ChangeFeedBridge::new(
                Arc::clone(registry),
                Arc::clone(&catalog),
                data_dir.to_path_buf(),
                branch_manager.clone(),
            )),
        );
    }

    let security_manager = if with_security {
        let heap_auth_storage =
            zyron_auth::HeapAuthStorage::new(Arc::clone(&disk), Arc::clone(&pool))
                .expect("auth storage");
        heap_auth_storage.attach_wal(&wal);
        let auth_storage: Arc<dyn zyron_auth::storage::AuthStorage> = Arc::new(heap_auth_storage);
        Some(Arc::new(
            zyron_auth::SecurityManager::new(auth_storage)
                .await
                .expect("security manager"),
        ))
    } else {
        None
    };

    let dml_hook: Arc<dyn zyron_executor::context::DmlHook> =
        Arc::new(zyron_wire::dml_enforce::LegalHoldDmlHook::new(
            Arc::clone(&legal_holds),
            Arc::clone(&catalog),
        ));
    let state = Arc::new(ServerState {
        raft: None,
        replication: None,
        node_capabilities: None,
        catalog,
        ddl_progress: std::sync::Arc::new(zyron_wire::ddl_progress::DdlProgressRegistry::new()),
        shadow_targets: std::sync::Arc::new(scc::HashMap::new()),
        wal,
        buffer_pool: pool,
        disk_manager: disk,
        txn_manager,
        doc_registry: std::sync::Arc::new(zyron_common::DocRegistry::new()),
        table_io_stats: std::sync::Arc::new(zyron_common::TableIOStatsRegistry::new()),
        index_io_stats: std::sync::Arc::new(zyron_common::IndexIOStatsRegistry::new()),
        columnar_maintenance: None,
        security_manager: security_manager.clone(),
        key_store: Arc::new(zyron_auth::LocalKeyStore::new([0u8; 32])),
        media_store: Arc::new(
            zyron_media::store::MediaStore::open(std::path::PathBuf::from(tmp.path()))
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
        cdc_slot_stats,
        cdc_stream_stats,
        cdc_ingest_stats: None,
        cdc_registry: cdc_registry.clone(),
        slot_manager,
        publication_manager: None,
        cdc_stream_manager,
        cdc_ingest_manager: None,
        udf_registry: None,
        uda_registry: None,
        procedure_registry: None,
        pipeline_manager: None,
        schedule_manager: None,
        event_dispatcher: None,
        mv_manager: None,
        stream_job_manager: None,
        branch_manager,
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
        cdc_hook: cdc_hook.clone(),
        dml_hook: Some(dml_hook),
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
        legal_holds: Arc::clone(&legal_holds),
        dlq_registry: Arc::new(zyron_streaming::dlq::DlqRegistry::new()),
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
        max_query_memory: None,
        spill_directory: None,
        balloon_params: None,
        default_auth_method: zyron_auth::auth_rules::AuthMethod::Trust,
        password_encryption: "balloon-sha-256".into(),
        admission: Arc::new(zyron_common::Admission::new()),
        query_metrics: Arc::new(zyron_common::QueryMetrics::new()),
        upgrade_control: None,
        chain_registry: Some(Arc::clone(&chain_registry)),
    });
    // The compliance log's rows from before its chain began are covered
    // the way the server covers them at start
    zyron_wire::verify_dispatch::cover_compliance_log(&state)
        .await
        .expect("the compliance log's genesis is written");
    (state, public_schema, security_manager, tmp)
}

/// The catalog id of a table in the test schema
pub fn table_id_of(server: &Arc<ServerState>, name: &str) -> u32 {
    let schema = server
        .catalog
        .get_schema(DatabaseId(1), "zyron_test")
        .expect("the test schema");
    server
        .catalog
        .get_table(schema.id, name)
        .unwrap_or_else(|_| panic!("table {name} exists"))
        .id
        .0
}

/// The plan a statement renders as, for a test whose subject is what the
/// planner settled on
pub async fn explain_text(server: &Arc<ServerState>, sql: &str) -> String {
    let stmt = zyron_parser::parse(sql)
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
    let node = zyron_planner::explain::ExplainNode::from_physical_plan(&plan);
    format!("{node:?}")
}

/// Installs the change feed reader and the position locks on a context.
///
/// The advances a stream read records go into the context's own list, which
/// `end_statement` writes when the statement commits. That is the same
/// sequence a connection follows, so a rolled back read through the harness
/// leaves the position where it was
pub fn install_change_reads(
    server: &Arc<ServerState>,
    ctx: &mut zyron_executor::context::ExecutionContext,
) -> Arc<zyron_lifecycle::verify::PendingChainWrites> {
    let advances = Arc::new(parking_lot::Mutex::new(Vec::new()));
    zyron_wire::change_feed_bridge::install_change_reads(server, ctx, &advances);
    // The rows a write puts into a verified table are hashed here, the way
    // a connection hashes them, so a commit through the harness extends the
    // chain exactly as one through the wire does
    let chain_writes = Arc::new(zyron_lifecycle::verify::PendingChainWrites::new());
    zyron_wire::verify_dispatch::install_chain_writes(server, ctx, &chain_writes);
    // The server installs the capture hook on every context, so a write
    // through the harness records its changes the way one through the wire
    // does. Without it a suite would read a feed nothing ever wrote to
    if let Some(hook) = server.cdc_hook.as_ref() {
        ctx.cdc_hook = Some(Arc::clone(hook));
    }
    if let Some(hook) = server.dml_hook.as_ref() {
        ctx.dml_hook = Some(Arc::clone(hook));
    }
    chain_writes
}

/// Writes the advances a statement's reads recorded, the way a connection's
/// commit does
pub async fn commit_stream_advances(
    server: &Arc<ServerState>,
    ctx: &Arc<zyron_executor::context::ExecutionContext>,
    txn: &mut zyron_storage::txn::Transaction,
) -> Result<(), zyron_common::ZyronError> {
    let held = std::mem::take(&mut *ctx.pending_stream_advances.lock());
    if held.is_empty() {
        return Ok(());
    }
    let advanced = zyron_wire::change_stream_dispatch::log_stream_advances(server, txn, &held, 0)?;
    zyron_wire::change_stream_dispatch::install_stream_advances(server, txn.txn_id, advanced).await
}

pub fn new_session() -> Option<Session> {
    let mut s = Session::new("test_user".into(), "testdb".into(), DatabaseId(1));
    s.search_path = vec!["zyron_test".into()];
    Some(s)
}

pub async fn exec_ddl(
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
    sql: &str,
) -> Result<(), String> {
    exec_ddl_on(server, session, sql, None).await
}

/// Runs a DDL statement with the session on a branch, the way a connection
/// after USE BRANCH runs one
pub async fn exec_ddl_on_branch(
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
    sql: &str,
    branch: &str,
) -> Result<(), String> {
    exec_ddl_on(server, session, sql, Some(branch.to_string())).await
}

async fn exec_ddl_on(
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
    sql: &str,
    mut active_branch: Option<String>,
) -> Result<(), String> {
    let stmt = zyron_parser::parse(sql)
        .expect("parse")
        .into_iter()
        .next()
        .expect("one statement");
    let mut txn_opt: Option<zyron_storage::txn::Transaction> = None;
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
    // One list for the whole script, so a stream read by any statement in
    // it advances once when the script commits, and one accumulator, so
    // every statement's rows in a verified table reach the one entry the
    // commit links
    let advances = Arc::new(parking_lot::Mutex::new(Vec::new()));
    let chain_writes = Arc::new(zyron_lifecycle::verify::PendingChainWrites::new());
    for sql in statements {
        let stmt = zyron_parser::parse(sql)
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
        // A fresh snapshot per statement, which is what the wire layer gives
        // each statement of a read-committed transaction
        let snapshot = server.txn_manager.refresh_snapshot(&txn);
        let mut ctx = zyron_executor::context::ExecutionContext::new(
            server.catalog.clone(),
            server.wal.clone(),
            server.buffer_pool.clone(),
            server.disk_manager.clone(),
            txn_id,
            snapshot,
        );
        // The server sets this on every DML context, so the harness does
        // too. Without it a write takes no row lock, and every test using
        // this helper exercises an engine that cannot detect two writers
        // of the same row
        ctx.spill = server.spill_directory.clone();
        ctx.memory_budget = server
            .max_query_memory
            .map(zyron_executor::QueryMemoryBudget::new);
        ctx.row_locks = Some(Arc::clone(server.txn_manager.lock_table()));
        // The server sets both of these on every query context. Without them the
        // harness runs an engine with no memory budget and nowhere to spill, so
        // no test through it can reach either path
        ctx.spill = server.spill_directory.clone();
        ctx.memory_budget = server
            .max_query_memory
            .map(zyron_executor::QueryMemoryBudget::new);
        ctx.spill = server.spill_directory.clone();
        ctx.memory_budget = server
            .max_query_memory
            .map(zyron_executor::QueryMemoryBudget::new);
        ctx.row_locks = Some(Arc::clone(server.txn_manager.lock_table()));
        // The server sets both of these on every query context. Without them the
        // harness runs an engine with no memory budget and nowhere to spill, so
        // no test through it can reach either path
        ctx.spill = server.spill_directory.clone();
        ctx.memory_budget = server
            .max_query_memory
            .map(zyron_executor::QueryMemoryBudget::new);
        // The server hands every context the media store, so media columns
        // externalize on write and inflate on scan through the harness too
        ctx.set_media_store(Arc::clone(&server.media_store));
        ctx.heap_files = Some(Arc::clone(&server.heap_files));
        ctx.btree_indexes = Some(Arc::clone(&server.btree_indexes));
        // The server installs its key store on every context, so ENCRYPTED
        // columns encrypt on write and decrypt on scan through the harness too
        ctx.set_key_store(Arc::clone(&server.key_store));
        ctx.doc_registry = Some(Arc::clone(&server.doc_registry));
        // The server installs these on every context, so a change read
        // through the harness resolves its feeds and takes its position lock
        // the way one through the wire does
        zyron_wire::change_feed_bridge::install_change_reads(server, &mut ctx, &advances);
        zyron_wire::verify_dispatch::install_chain_writes(server, &mut ctx, &chain_writes);
        if let Some(hook) = server.cdc_hook.as_ref() {
            ctx.cdc_hook = Some(Arc::clone(hook));
        }
        if let Some(hook) = server.dml_hook.as_ref() {
            ctx.dml_hook = Some(Arc::clone(hook));
        }
        let ctx = Arc::new(ctx);
        if let Err(e) = zyron_executor::execute(plan, &ctx).await {
            let _ = zyron_lake::abandon_txn(server.disk_manager.data_dir(), txn_id);
            let _ = server.txn_manager.abort(&mut txn);
            chain_writes.clear();
            return Err(e);
        }
    }
    let held = std::mem::take(&mut *advances.lock());
    let advanced =
        zyron_wire::change_stream_dispatch::log_stream_advances(server, &mut txn, &held, 0)?;
    // The chain entry goes into this transaction's own log chain, ahead of
    // its commit record, the way a connection's commit writes one. A script
    // the chain cannot cover is rolled back here the way a connection's
    // commit rolls it back
    let chained =
        match zyron_wire::verify_dispatch::log_commit_chains(server, &mut txn, &chain_writes) {
            Ok(chained) => chained,
            Err(e) => {
                let _ = zyron_lake::abandon_txn(server.disk_manager.data_dir(), txn_id);
                let _ = server.txn_manager.abort(&mut txn);
                return Err(e);
            }
        };
    server.txn_manager.commit(&mut txn).await.expect("commit");
    zyron_wire::verify_dispatch::publish_commit_chains(server, &chained);
    let logs = zyron_lake::publish_txn(server.disk_manager.data_dir(), txn_id).expect("publish");
    zyron_wire::connection::refresh_lake_stats(server, &logs);
    zyron_wire::change_stream_dispatch::install_stream_advances(server, txn_id, advanced).await?;
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
        vec!["zyron_test".into()],
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
        txn_id,
        snapshot,
    );
    ctx.row_locks = Some(Arc::clone(server.txn_manager.lock_table()));
    // The server sets both of these on every query context. Without them the
    // harness runs an engine with no memory budget and nowhere to spill, so
    // no test through it can reach either path
    ctx.spill = server.spill_directory.clone();
    ctx.memory_budget = server
        .max_query_memory
        .map(zyron_executor::QueryMemoryBudget::new);
    // The server hands every context the media store, so media columns
    // externalize on write and inflate on scan through the harness too
    ctx.set_media_store(Arc::clone(&server.media_store));
    ctx.heap_files = Some(Arc::clone(&server.heap_files));
    ctx.btree_indexes = Some(Arc::clone(&server.btree_indexes));
    // The server installs its key store on every context, so ENCRYPTED
    // columns encrypt on write and decrypt on scan through the harness too
    ctx.set_key_store(Arc::clone(&server.key_store));
    ctx.table_io_stats = Some(Arc::clone(&server.table_io_stats));
    ctx.index_io_stats = Some(Arc::clone(&server.index_io_stats));
    ctx.doc_registry = Some(Arc::clone(&server.doc_registry));
    // The server installs these on every context, so a change read through
    // the harness resolves its feeds and takes its position lock the way one
    // through the wire does
    let chain_writes = install_change_reads(server, &mut ctx);
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
        chain_writes.clear();
        return outcome;
    }
    let held = std::mem::take(&mut *ctx.pending_stream_advances.lock());
    let advanced =
        zyron_wire::change_stream_dispatch::log_stream_advances(server, &mut txn, &held, 0)?;
    // The chain entry goes into this transaction's own log chain, ahead of
    // its commit record, the way a connection's commit writes one
    let chained = zyron_wire::verify_dispatch::log_commit_chains(server, &mut txn, &chain_writes)?;
    server.txn_manager.commit(&mut txn).await.expect("commit");
    zyron_wire::verify_dispatch::publish_commit_chains(server, &chained);
    // Mirrors the wire layer, lake versions publish after the durable commit
    // and the manifest's statistics reach the planner with them
    let logs = zyron_lake::publish_txn(server.disk_manager.data_dir(), txn_id).expect("publish");
    zyron_wire::connection::refresh_lake_stats(server, &logs);
    zyron_wire::change_stream_dispatch::install_stream_advances(server, txn_id, advanced).await?;
    outcome
}

/// Ends a statement's implicit transaction the way a connection does.
///
/// A transaction that appended a WAL record commits durably and then
/// publishes the lake versions written under it. One that only read has
/// nothing to make durable, so it releases its slot with no commit record
/// and no flush wait, which keeps a read-only statement's time the engine's
/// rather than the device's
pub async fn end_statement(
    server: &Arc<ServerState>,
    ctx: &Arc<zyron_executor::context::ExecutionContext>,
    txn: &mut zyron_storage::txn::Transaction,
) -> Result<(), zyron_common::ZyronError> {
    // A change stream the statement read moves in this commit, the way it
    // does on a connection, so a rolled back or failed read leaves it alone
    let held = std::mem::take(&mut *ctx.pending_stream_advances.lock());
    let advanced = zyron_wire::change_stream_dispatch::log_stream_advances(server, txn, &held, 0)?;
    let txn_id = txn.txn_id;
    if !txn.wrote_data() {
        server.txn_manager.commit_read_only(txn)?;
        zyron_wire::change_stream_dispatch::install_stream_advances(server, txn_id, advanced)
            .await?;
        return Ok(());
    }
    server.txn_manager.commit(txn).await?;
    let logs = zyron_lake::publish_txn(server.disk_manager.data_dir(), txn_id)?;
    zyron_wire::connection::refresh_lake_stats(server, &logs);
    zyron_wire::change_stream_dispatch::install_stream_advances(server, txn_id, advanced).await?;
    Ok(())
}

pub async fn query_rows(server: &Arc<ServerState>, sql: &str) -> usize {
    query_batches(server, sql)
        .await
        .iter()
        .map(|b| b.num_rows)
        .sum()
}

/// Runs a query through the harness and hands back every batch it
/// produced, so a caller measuring the query stops its clock before the
/// batches are dropped
pub async fn query_batches(
    server: &Arc<ServerState>,
    sql: &str,
) -> Vec<zyron_executor::batch::DataBatch> {
    let stmt = zyron_parser::parse(sql)
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
        .begin(zyron_storage::txn::IsolationLevel::ReadCommitted)
        .expect("begin");
    let snapshot = txn.snapshot.clone();
    let txn_id = txn.txn_id;
    let mut ctx = zyron_executor::context::ExecutionContext::new(
        server.catalog.clone(),
        server.wal.clone(),
        server.buffer_pool.clone(),
        server.disk_manager.clone(),
        txn_id,
        snapshot,
    );
    ctx.row_locks = Some(Arc::clone(server.txn_manager.lock_table()));
    // The server sets both of these on every query context. Without them the
    // harness runs an engine with no memory budget and nowhere to spill, so
    // no test through it can reach either path
    ctx.spill = server.spill_directory.clone();
    ctx.memory_budget = server
        .max_query_memory
        .map(zyron_executor::QueryMemoryBudget::new);
    // The server hands every context the media store, so media columns
    // externalize on write and inflate on scan through the harness too
    ctx.set_media_store(Arc::clone(&server.media_store));
    ctx.heap_files = Some(Arc::clone(&server.heap_files));
    ctx.btree_indexes = Some(Arc::clone(&server.btree_indexes));
    // The server installs its key store on every context, so ENCRYPTED
    // columns encrypt on write and decrypt on scan through the harness too
    ctx.set_key_store(Arc::clone(&server.key_store));
    ctx.table_io_stats = Some(Arc::clone(&server.table_io_stats));
    ctx.index_io_stats = Some(Arc::clone(&server.index_io_stats));
    ctx.doc_registry = Some(Arc::clone(&server.doc_registry));
    // The server installs these on every context, so a change read through
    // the harness resolves its feeds and takes its position lock the way one
    // through the wire does
    let _chain_writes = install_change_reads(server, &mut ctx);
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
    end_statement(server, &ctx, &mut txn).await.expect("commit");
    batches
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
        vec!["zyron_test".into()],
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
    let txn_id = txn.txn_id;
    let mut ctx = zyron_executor::context::ExecutionContext::new(
        server.catalog.clone(),
        server.wal.clone(),
        server.buffer_pool.clone(),
        server.disk_manager.clone(),
        txn_id,
        snapshot,
    );
    ctx.row_locks = Some(Arc::clone(server.txn_manager.lock_table()));
    // The server sets both of these on every query context. Without them the
    // harness runs an engine with no memory budget and nowhere to spill, so
    // no test through it can reach either path
    ctx.spill = server.spill_directory.clone();
    ctx.memory_budget = server
        .max_query_memory
        .map(zyron_executor::QueryMemoryBudget::new);
    // The server hands every context the media store, so media columns
    // externalize on write and inflate on scan through the harness too
    ctx.set_media_store(Arc::clone(&server.media_store));
    ctx.heap_files = Some(Arc::clone(&server.heap_files));
    ctx.btree_indexes = Some(Arc::clone(&server.btree_indexes));
    // The server installs its key store on every context, so ENCRYPTED
    // columns encrypt on write and decrypt on scan through the harness too
    ctx.set_key_store(Arc::clone(&server.key_store));
    ctx.doc_registry = Some(Arc::clone(&server.doc_registry));
    // The server installs these on every context, so a change read through
    // the harness resolves its feeds and takes its position lock the way one
    // through the wire does
    let _chain_writes = install_change_reads(server, &mut ctx);
    let ctx = Arc::new(ctx);
    let result = zyron_executor::execute(plan, &ctx).await;
    let _ = end_statement(server, &ctx, &mut txn).await;
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
        vec!["zyron_test".into()],
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
    let txn_id = txn.txn_id;
    let mut ctx = zyron_executor::context::ExecutionContext::new(
        server.catalog.clone(),
        server.wal.clone(),
        server.buffer_pool.clone(),
        server.disk_manager.clone(),
        txn_id,
        snapshot,
    );
    ctx.row_locks = Some(Arc::clone(server.txn_manager.lock_table()));
    // The server sets both of these on every query context. Without them the
    // harness runs an engine with no memory budget and nowhere to spill, so
    // no test through it can reach either path
    ctx.spill = server.spill_directory.clone();
    ctx.memory_budget = server
        .max_query_memory
        .map(zyron_executor::QueryMemoryBudget::new);
    // The server hands every context the media store, so media columns
    // externalize on write and inflate on scan through the harness too
    ctx.set_media_store(Arc::clone(&server.media_store));
    ctx.heap_files = Some(Arc::clone(&server.heap_files));
    ctx.btree_indexes = Some(Arc::clone(&server.btree_indexes));
    // The server installs its key store on every context, so ENCRYPTED
    // columns encrypt on write and decrypt on scan through the harness too
    ctx.set_key_store(Arc::clone(&server.key_store));
    ctx.doc_registry = Some(Arc::clone(&server.doc_registry));
    // The server installs these on every context, so a change read through
    // the harness resolves its feeds and takes its position lock the way one
    // through the wire does
    let _chain_writes = install_change_reads(server, &mut ctx);
    let ctx = Arc::new(ctx);
    let result = zyron_executor::execute(plan, &ctx).await;
    let _ = end_statement(server, &ctx, &mut txn).await;
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
    // The statement phases a connection records, so a profiled run through
    // the harness decomposes the same way one through the wire does.
    // Compiled in by --features profile, gated at runtime by ZYRON_PROFILE
    use zyron_common::profile::{Phase, scope};
    let stmt = {
        let _s = scope(Phase::WireRecvParse);
        zyron_parser::parse(sql)
            .expect("parse")
            .into_iter()
            .next()
            .expect("one statement")
    };
    let plan = {
        let _s = scope(Phase::WirePlan);
        zyron_planner::plan(
            &server.catalog,
            DatabaseId(1),
            vec!["zyron_test".into()],
            stmt,
            None,
        )
        .await
        .expect("plan")
    };
    let setup = scope(Phase::WireExecSetup);
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
        txn_id,
        snapshot,
    );
    ctx.row_locks = Some(Arc::clone(server.txn_manager.lock_table()));
    // The server sets both of these on every query context. Without them the
    // harness runs an engine with no memory budget and nowhere to spill, so
    // no test through it can reach either path
    ctx.spill = server.spill_directory.clone();
    ctx.memory_budget = server
        .max_query_memory
        .map(zyron_executor::QueryMemoryBudget::new);
    // The server hands every context the media store, so media columns
    // externalize on write and inflate on scan through the harness too
    ctx.set_media_store(Arc::clone(&server.media_store));
    ctx.heap_files = Some(Arc::clone(&server.heap_files));
    ctx.btree_indexes = Some(Arc::clone(&server.btree_indexes));
    // The server installs its key store on every context, so ENCRYPTED
    // columns encrypt on write and decrypt on scan through the harness too
    ctx.set_key_store(Arc::clone(&server.key_store));
    ctx.table_io_stats = Some(Arc::clone(&server.table_io_stats));
    ctx.index_io_stats = Some(Arc::clone(&server.index_io_stats));
    ctx.doc_registry = Some(Arc::clone(&server.doc_registry));
    // The server installs these on every context, so a change read through
    // the harness resolves its feeds and takes its position lock the way one
    // through the wire does
    let _chain_writes = install_change_reads(server, &mut ctx);
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
    drop(setup);
    let batches = {
        let _s = scope(Phase::WireExecute);
        zyron_executor::execute(plan, &ctx).await.expect("execute")
    };
    {
        let _s = scope(Phase::WireAutoCommit);
        end_statement(server, &ctx, &mut txn).await.expect("commit");
    }

    let _s = scope(Phase::WireSend);
    let mut rows = Vec::new();
    for batch in &batches {
        for r in 0..batch.num_rows {
            rows.push(batch.columns.iter().map(|c| c.get_scalar(r)).collect());
        }
    }
    rows
}

/// Fallible counterpart of query_values, for tests asserting that a query
/// errors loudly instead of returning an empty or wrong result.
#[allow(dead_code)]
pub async fn try_query_values(
    server: &Arc<ServerState>,
    sql: &str,
) -> Result<Vec<Vec<ScalarValue>>, zyron_common::ZyronError> {
    let stmt = zyron_parser::parse(sql)
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
        txn_id,
        snapshot,
    );
    ctx.row_locks = Some(Arc::clone(server.txn_manager.lock_table()));
    // The server sets both of these on every query context. Without them the
    // harness runs an engine with no memory budget and nowhere to spill, so
    // no test through it can reach either path
    ctx.spill = server.spill_directory.clone();
    ctx.memory_budget = server
        .max_query_memory
        .map(zyron_executor::QueryMemoryBudget::new);
    // The server hands every context the media store, so media columns
    // externalize on write and inflate on scan through the harness too
    ctx.set_media_store(Arc::clone(&server.media_store));
    ctx.heap_files = Some(Arc::clone(&server.heap_files));
    ctx.btree_indexes = Some(Arc::clone(&server.btree_indexes));
    // The server installs its key store on every context, so ENCRYPTED
    // columns encrypt on write and decrypt on scan through the harness too
    ctx.set_key_store(Arc::clone(&server.key_store));
    let ctx = Arc::new(ctx);
    let result = zyron_executor::execute(plan, &ctx).await;
    match result {
        Ok(batches) => {
            end_statement(server, &ctx, &mut txn).await.expect("commit");
            let mut rows = Vec::new();
            for batch in &batches {
                for r in 0..batch.num_rows {
                    rows.push(batch.columns.iter().map(|c| c.get_scalar(r)).collect());
                }
            }
            Ok(rows)
        }
        Err(e) => {
            let _ = server.txn_manager.abort(&mut txn);
            Err(e)
        }
    }
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
        vec!["zyron_test".into()],
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
        txn_id,
        snapshot,
    );
    ctx.row_locks = Some(Arc::clone(server.txn_manager.lock_table()));
    // The server sets both of these on every query context. Without them the
    // harness runs an engine with no memory budget and nowhere to spill, so
    // no test through it can reach either path
    ctx.spill = server.spill_directory.clone();
    ctx.memory_budget = server
        .max_query_memory
        .map(zyron_executor::QueryMemoryBudget::new);
    // The server hands every context the media store, so media columns
    // externalize on write and inflate on scan through the harness too
    ctx.set_media_store(Arc::clone(&server.media_store));
    ctx.heap_files = Some(Arc::clone(&server.heap_files));
    ctx.btree_indexes = Some(Arc::clone(&server.btree_indexes));
    // The server installs its key store on every context, so ENCRYPTED
    // columns encrypt on write and decrypt on scan through the harness too
    ctx.set_key_store(Arc::clone(&server.key_store));
    ctx.table_io_stats = Some(Arc::clone(&server.table_io_stats));
    ctx.index_io_stats = Some(Arc::clone(&server.index_io_stats));
    ctx.doc_registry = Some(Arc::clone(&server.doc_registry));
    // The server installs these on every context, so a change read through
    // the harness resolves its feeds and takes its position lock the way one
    // through the wire does
    let _chain_writes = install_change_reads(server, &mut ctx);
    ctx.active_branch_name = Some(std::sync::Arc::from(branch));
    // The heap routes copy-on-write pages by branch id and a write on the
    // branch is recorded in the branch's own feed, both the way a connection
    // after USE BRANCH does it
    if let Some(mgr) = &server.branch_manager {
        ctx.branch_catalog = Some(Arc::clone(mgr) as Arc<dyn zyron_common::BranchCatalog>);
        ctx.active_branch_id = mgr.get_branch_by_name(branch).ok().map(|e| e.id.0);
    }
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
            let held = std::mem::take(&mut *ctx.pending_stream_advances.lock());
            let advanced = zyron_wire::change_stream_dispatch::log_stream_advances(
                server, &mut txn, &held, 0,
            )?;
            server.txn_manager.commit(&mut txn).await.expect("commit");
            let logs =
                zyron_lake::publish_txn(server.disk_manager.data_dir(), txn_id).expect("publish");
            zyron_wire::connection::refresh_lake_stats(server, &logs);
            zyron_wire::change_stream_dispatch::install_stream_advances(server, txn_id, advanced)
                .await?;
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

/// What one statement produced, read back from the protocol stream
pub struct WireStatementOutcome {
    /// NoticeResponse message texts, in arrival order
    pub notices: Vec<String>,
    /// ErrorResponse message texts, in arrival order
    pub errors: Vec<String>,
    /// CommandComplete tags, in arrival order
    pub tags: Vec<String>,
}

/// Runs statements over a real TCP connection against the production
/// connection loop, one Query message per statement.
///
/// This is the path a client takes: startup handshake under trust auth,
/// simple-query dispatch, and a clean terminate. A test that has to prove
/// a statement's grammar reaches its work goes through here rather than
/// through a helper that starts past the parse
pub async fn wire_query(
    server: &Arc<ServerState>,
    statements: &[&str],
) -> Vec<WireStatementOutcome> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr");
    let state = Arc::clone(server);

    let server_side = async move {
        let (stream, _) = listener.accept().await.expect("accept");
        let mut conn = zyron_wire::connection::Connection::new(stream, state, None);
        if let Err(e) = conn.run().await {
            eprintln!("wire_query server side ended with {e:?}");
        }
    };

    let client_side = async move {
        let mut client = tokio::net::TcpStream::connect(addr).await.expect("connect");

        // One protocol message off the stream: a type byte, a length that
        // includes itself, and the payload
        async fn read_message(client: &mut tokio::net::TcpStream) -> (u8, Vec<u8>) {
            let mut head = [0u8; 5];
            client.read_exact(&mut head).await.expect("message head");
            let len = i32::from_be_bytes([head[1], head[2], head[3], head[4]]) as usize;
            let mut payload = vec![0u8; len - 4];
            client
                .read_exact(&mut payload)
                .await
                .expect("message payload");
            (head[0], payload)
        }

        // The M field of a NoticeResponse or ErrorResponse payload, which
        // is the human message among the coded fields
        fn message_field(payload: &[u8]) -> Option<String> {
            let mut i = 0;
            while i < payload.len() && payload[i] != 0 {
                let code = payload[i];
                i += 1;
                let start = i;
                while i < payload.len() && payload[i] != 0 {
                    i += 1;
                }
                if code == b'M' {
                    return Some(String::from_utf8_lossy(&payload[start..i]).into_owned());
                }
                i += 1;
            }
            None
        }

        // Startup at protocol 3.0 against the bootstrap database, the
        // one the rest of the harness plans in. Trust auth answers
        // AuthenticationOk with no credential round trip
        let mut params = Vec::new();
        params.extend_from_slice(&196608i32.to_be_bytes());
        params.extend_from_slice(b"user\0test_user\0database\0zyron\0\0");
        let mut startup = Vec::new();
        startup.extend_from_slice(&((params.len() + 4) as i32).to_be_bytes());
        startup.extend_from_slice(&params);
        client.write_all(&startup).await.expect("startup");
        client.flush().await.expect("flush");
        loop {
            let (kind, _) = read_message(&mut client).await;
            if kind == b'Z' {
                break;
            }
        }

        let mut outcomes = Vec::with_capacity(statements.len());
        for sql in statements {
            let mut payload = sql.as_bytes().to_vec();
            payload.push(0);
            let mut msg = vec![b'Q'];
            msg.extend_from_slice(&((payload.len() + 4) as i32).to_be_bytes());
            msg.extend_from_slice(&payload);
            client.write_all(&msg).await.expect("query");
            client.flush().await.expect("flush");

            let mut outcome = WireStatementOutcome {
                notices: Vec::new(),
                errors: Vec::new(),
                tags: Vec::new(),
            };
            loop {
                let (kind, payload) = read_message(&mut client).await;
                match kind {
                    b'N' => outcome.notices.extend(message_field(&payload)),
                    b'E' => outcome.errors.extend(message_field(&payload)),
                    b'C' => {
                        let end = payload
                            .iter()
                            .position(|&b| b == 0)
                            .unwrap_or(payload.len());
                        outcome
                            .tags
                            .push(String::from_utf8_lossy(&payload[..end]).into_owned());
                    }
                    b'Z' => break,
                    _ => {}
                }
            }
            outcomes.push(outcome);
        }

        client
            .write_all(&[b'X', 0, 0, 0, 4])
            .await
            .expect("terminate");
        let _ = client.shutdown().await;
        outcomes
    };

    let (_, outcomes) = tokio::join!(server_side, client_side);
    outcomes
}

// =============================================================================
// EXPLAIN ANALYZE
// =============================================================================

/// Plans and executes one statement the way EXPLAIN ANALYZE does, then
/// merges the executor's counters into the plan tree.
///
/// A suite that wants to know what a scan actually did has to go through
/// this rather than through `query_values`, because the counters only exist
/// when the executor is told to wrap its operators in metrics collectors
pub async fn analyze(
    server: &Arc<ServerState>,
    sql: &str,
) -> (zyron_planner::ExplainNode, zyron_planner::NodeMetrics) {
    let stmt = zyron_parser::parse(sql)
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
    let mut tree = zyron_planner::ExplainNode::from_physical_plan(&plan);

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
        txn_id,
        snapshot,
    );
    ctx.row_locks = Some(Arc::clone(server.txn_manager.lock_table()));
    // The server sets both of these on every query context. Without them the
    // harness runs an engine with no memory budget and nowhere to spill, so
    // no test through it can reach either path
    ctx.spill = server.spill_directory.clone();
    ctx.memory_budget = server
        .max_query_memory
        .map(zyron_executor::QueryMemoryBudget::new);
    // The server hands every context the media store, so media columns
    // externalize on write and inflate on scan through the harness too
    ctx.set_media_store(Arc::clone(&server.media_store));
    ctx.heap_files = Some(Arc::clone(&server.heap_files));
    ctx.btree_indexes = Some(Arc::clone(&server.btree_indexes));
    // The server installs its key store on every context, so ENCRYPTED
    // columns encrypt on write and decrypt on scan through the harness too
    ctx.set_key_store(Arc::clone(&server.key_store));
    // The only thing that makes the executor wrap its operators in metrics
    // collectors
    ctx.analyze = true;
    let ctx = Arc::new(ctx);
    let (_batches, metrics) = zyron_executor::execute_analyze(plan, &ctx)
        .await
        .expect("analyze");
    end_statement(server, &ctx, &mut txn).await.expect("commit");

    let metrics = metrics.expect("analyze mode produces metrics");
    let node_metrics = node_metrics_of(&metrics);
    assert!(
        tree.merge_metrics(&node_metrics) > 0,
        "the plan and the executor must agree on the operator names"
    );
    (tree, node_metrics)
}

/// One statement's plan as EXPLAIN renders it, without running it.
///
/// What a reader sees before committing to a query, which is where a
/// verdict the planner reached rather than measured belongs
pub async fn render_plan(server: &Arc<ServerState>, sql: &str) -> String {
    let stmt = zyron_parser::parse(sql)
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
    zyron_planner::ExplainNode::from_physical_plan(&plan)
        .render(&zyron_planner::ExplainOptions::default())
}

/// The analyzed plan as the text a person reads
pub async fn render_analyzed(server: &Arc<ServerState>, sql: &str) -> String {
    let (tree, _) = analyze(server, sql).await;
    tree.render(&zyron_planner::ExplainOptions {
        analyze: true,
        ..Default::default()
    })
}

/// The measured counters of the LakeScan in one statement's plan
pub async fn analyze_lake_scan(
    server: &Arc<ServerState>,
    sql: &str,
) -> zyron_planner::ActualMetrics {
    let (tree, _) = analyze(server, sql).await;
    find_named(&tree, "LakeScan")
        .expect("a LakeScan node")
        .actual_metrics
        .clone()
        .expect("the scan reports what it measured")
}

/// The first node in the tree with this operator name
pub fn find_named<'a>(
    node: &'a zyron_planner::ExplainNode,
    name: &str,
) -> Option<&'a zyron_planner::ExplainNode> {
    if node.operator_name == name {
        return Some(node);
    }
    node.children.iter().find_map(|c| find_named(c, name))
}

fn node_metrics_of(
    metrics: &zyron_executor::operator::OperatorMetrics,
) -> zyron_planner::NodeMetrics {
    use std::sync::atomic::Ordering;
    let mut aux = [0u64; zyron_planner::ACTUAL_AUX_SLOTS];
    for (slot, value) in aux.iter_mut().enumerate() {
        *value = metrics.aux(slot);
    }
    zyron_planner::NodeMetrics {
        name: metrics.name.clone(),
        rows: metrics.rows_produced.load(Ordering::Relaxed),
        elapsed_ns: metrics.elapsed_ns.load(Ordering::Relaxed),
        batches: metrics.batches.load(Ordering::Relaxed),
        aux,
        children: metrics
            .children
            .iter()
            .map(|c| node_metrics_of(c))
            .collect(),
    }
}

/// An explicit transaction through the harness, with the stream advances its
/// statements record written at commit and dropped at rollback, the way a
/// connection's are
pub struct Txn {
    server: Arc<ServerState>,
    txn: Option<zyron_storage::txn::Transaction>,
    advances: Arc<parking_lot::Mutex<Vec<zyron_executor::context::PendingStreamAdvance>>>,
}

impl Txn {
    pub fn begin(server: &Arc<ServerState>) -> Self {
        let txn = server
            .txn_manager
            .begin(zyron_storage::txn::IsolationLevel::ReadCommitted)
            .expect("begin");
        Self {
            server: Arc::clone(server),
            txn: Some(txn),
            advances: Arc::new(parking_lot::Mutex::new(Vec::new())),
        }
    }

    fn context(&self) -> Arc<zyron_executor::context::ExecutionContext> {
        let txn = self.txn.as_ref().expect("an open transaction");
        let snapshot = self.server.txn_manager.refresh_snapshot(txn);
        let mut ctx = zyron_executor::context::ExecutionContext::new(
            self.server.catalog.clone(),
            self.server.wal.clone(),
            self.server.buffer_pool.clone(),
            self.server.disk_manager.clone(),
            txn.txn_id,
            snapshot,
        );
        ctx.row_locks = Some(Arc::clone(self.server.txn_manager.lock_table()));
        ctx.intent_locks = Some(Arc::clone(self.server.txn_manager.intent_locks()));
        ctx.heap_files = Some(Arc::clone(&self.server.heap_files));
        ctx.btree_indexes = Some(Arc::clone(&self.server.btree_indexes));
        ctx.doc_registry = Some(Arc::clone(&self.server.doc_registry));
        ctx.set_media_store(Arc::clone(&self.server.media_store));
        ctx.set_key_store(Arc::clone(&self.server.key_store));
        zyron_wire::change_feed_bridge::install_change_reads(
            &self.server,
            &mut ctx,
            &self.advances,
        );
        if let Some(hook) = self.server.cdc_hook.as_ref() {
            ctx.cdc_hook = Some(Arc::clone(hook));
        }
        Arc::new(ctx)
    }

    /// Runs one statement and returns its rows
    pub async fn run(
        &mut self,
        sql: &str,
    ) -> Result<Vec<Vec<ScalarValue>>, zyron_common::ZyronError> {
        let stmt = zyron_parser::parse(sql)
            .expect("parse")
            .into_iter()
            .next()
            .expect("one statement");
        let plan = zyron_planner::plan(
            &self.server.catalog,
            zyron_catalog::DatabaseId(1),
            vec!["zyron_test".into()],
            stmt,
            None,
        )
        .await?;
        let ctx = self.context();
        let batches = zyron_executor::execute(plan, &ctx).await?;
        Ok(batches
            .iter()
            .flat_map(|b| {
                (0..b.num_rows)
                    .map(|r| b.columns.iter().map(|c| c.get_scalar(r)).collect())
                    .collect::<Vec<Vec<ScalarValue>>>()
            })
            .collect())
    }

    /// Commits the way a connection does. The commit record lands, then
    /// the lake versions written under the transaction publish
    pub async fn commit(self) {
        let server = Arc::clone(&self.server);
        let txn_id = self.commit_holding_lake().await;
        let logs = zyron_lake::publish_txn(server.disk_manager.data_dir(), txn_id)
            .expect("publish the lake versions");
        zyron_wire::connection::refresh_lake_stats(&server, &logs);
    }

    /// Writes the commit record and stops there, leaving every lake
    /// version the transaction wrote pending, which is the instant between
    /// a connection's two commit steps. The caller publishes them through
    /// `zyron_lake::publish_txn` with the returned transaction id
    pub async fn commit_holding_lake(mut self) -> u64 {
        let mut txn = self.txn.take().expect("an open transaction");
        let held = std::mem::take(&mut *self.advances.lock());
        let advanced = zyron_wire::change_stream_dispatch::log_stream_advances(
            &self.server,
            &mut txn,
            &held,
            0,
        )
        .expect("log the advances");
        if txn.wrote_data() {
            self.server
                .txn_manager
                .commit(&mut txn)
                .await
                .expect("commit");
        } else {
            self.server
                .txn_manager
                .commit_read_only(&mut txn)
                .expect("commit");
        }
        zyron_wire::change_stream_dispatch::install_stream_advances(
            &self.server,
            txn.txn_id,
            advanced,
        )
        .await
        .expect("install the advances");
        txn.txn_id
    }

    /// The transaction's id, for a caller that acts on it from outside
    pub fn txn_id(&self) -> u64 {
        self.txn.as_ref().expect("an open transaction").txn_id
    }

    pub fn rollback(mut self) {
        let mut txn = self.txn.take().expect("an open transaction");
        self.advances.lock().clear();
        let _ = zyron_lake::abandon_txn(self.server.disk_manager.data_dir(), txn.txn_id);
        self.server.txn_manager.abort(&mut txn).expect("abort");
    }
}

impl Drop for Txn {
    fn drop(&mut self) {
        // A transaction dropped without a decision is a killed session. It
        // aborts, its lake versions are discarded, and its advances go
        // with it
        if let Some(mut txn) = self.txn.take() {
            let _ = zyron_lake::abandon_txn(self.server.disk_manager.data_dir(), txn.txn_id);
            let _ = self.server.txn_manager.abort(&mut txn);
        }
    }
}

/// Where a table's rows live, which a scenario over change data runs
/// against both of, so what holds for a heap table is shown to hold for a
/// lake table by the same assertions
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Storage {
    Heap,
    Lake,
}

impl Storage {
    pub const BOTH: [Storage; 2] = [Storage::Heap, Storage::Lake];

    /// A CREATE TABLE statement for this storage, the lake form named by
    /// its USING clause
    pub fn create(self, sql: &str) -> String {
        match self {
            Storage::Heap => sql.to_string(),
            Storage::Lake => format!("{sql} USING ZYRONLAKE"),
        }
    }
}

impl std::fmt::Display for Storage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Storage::Heap => "heap",
            Storage::Lake => "lake",
        })
    }
}

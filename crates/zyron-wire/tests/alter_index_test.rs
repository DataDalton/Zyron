//! Integration tests for ALTER INDEX RENAME.
//!
//! Creates an index, renames it through the DDL dispatch path, and verifies the
//! catalog reflects the new name, the old name no longer resolves, and rename
//! collisions and missing indexes are rejected.
//!
//! Run: cargo test -p zyron-wire --test alter_index_test -- --nocapture

use std::sync::Arc;

use zyron_buffer::{BufferPool, BufferPoolConfig};
use zyron_catalog::{
    Catalog, CatalogCache, DatabaseId, HeapCatalogStorage, SYSTEM_DATABASE_ID, SchemaId,
};
use zyron_executor::batch::DataBatch;
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

async fn exec(server: &Arc<ServerState>, session: &mut Option<Session>, sql: &str) {
    try_exec(server, session, sql)
        .await
        .unwrap_or_else(|e| panic!("statement failed: {sql}\n{e}"));
}

fn index_names(server: &Arc<ServerState>, schema: SchemaId) -> Vec<String> {
    let table = server.catalog.get_table(schema, "t").expect("table t");
    let mut names: Vec<String> = server
        .catalog
        .get_indexes_for_table(table.id)
        .iter()
        .map(|i| i.name.clone())
        .collect();
    names.sort();
    names
}

#[tokio::test]
async fn rename_changes_index_name() {
    let (server, schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE t (id INT, v INT)").await;
    exec(&server, &mut session, "CREATE INDEX idx_v ON t (v)").await;
    assert_eq!(index_names(&server, schema), vec!["idx_v".to_string()]);

    exec(&server, &mut session, "ALTER INDEX idx_v RENAME TO idx_v2").await;
    assert_eq!(index_names(&server, schema), vec!["idx_v2".to_string()]);

    // The old name no longer resolves; the new one drops cleanly.
    let err = try_exec(&server, &mut session, "DROP INDEX idx_v")
        .await
        .expect_err("old index name should not resolve");
    assert!(
        err.to_lowercase().contains("idx_v") || err.to_lowercase().contains("not"),
        "unexpected: {err}"
    );
    exec(&server, &mut session, "DROP INDEX idx_v2").await;
    assert!(index_names(&server, schema).is_empty());
}

#[tokio::test]
async fn rename_to_existing_name_errors() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE t (id INT, v INT)").await;
    exec(&server, &mut session, "CREATE INDEX idx_a ON t (id)").await;
    exec(&server, &mut session, "CREATE INDEX idx_b ON t (v)").await;
    let err = try_exec(&server, &mut session, "ALTER INDEX idx_a RENAME TO idx_b")
        .await
        .expect_err("rename onto an existing index name should fail");
    assert!(
        err.to_lowercase().contains("idx_b") || err.to_lowercase().contains("exist"),
        "unexpected: {err}"
    );
}

#[tokio::test]
async fn rename_missing_index_errors() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE t (id INT, v INT)").await;
    let err = try_exec(&server, &mut session, "ALTER INDEX nope RENAME TO other")
        .await
        .expect_err("renaming a missing index should fail");
    assert!(
        err.to_lowercase().contains("nope") || err.to_lowercase().contains("not"),
        "unexpected: {err}"
    );
}

/// Counts the live entries in a B+tree index by full range scan.
fn index_entry_count(server: &Arc<ServerState>, index_key: u32) -> usize {
    server
        .btree_indexes
        .read_sync(&index_key, |_, tree| tree.range_scan_sync(None, None).len())
        .unwrap_or(0)
}

/// Runs one vacuum pass over a table, mirroring the background worker: reclaims
/// committed-deleted tuples below the oldest active transaction and deletes
/// their B+tree index entries.
async fn run_vacuum(server: &Arc<ServerState>, table: &Arc<zyron_catalog::TableEntry>) {
    run_vacuum_with_floor(server, table, u64::MAX).await;
}

/// Runs one vacuum pass with an explicit retention floor: a committed-delete
/// tuple is reclaimed only when its deleter committed at or below the floor.
/// `u64::MAX` reclaims everything (the default), 0 keeps every committed delete
/// (unlimited retention).
async fn run_vacuum_with_floor(
    server: &Arc<ServerState>,
    table: &Arc<zyron_catalog::TableEntry>,
    retention_floor: u64,
) {
    use zyron_storage::HeapPage;

    let active = server.txn_manager.active_txn_ids();
    let oldest_active = if active.is_empty() {
        server.txn_manager.next_txn_id()
    } else {
        active[0]
    };
    let status_map = server.txn_manager.status_map();
    // Use the shared heap-file handle so the scan sees the in-memory pages the
    // inserts wrote (the background writer does not run in this test).
    let heap = server
        .heap_files
        .read_sync(&table.heap_file_id, |_, v| Arc::clone(v))
        .expect("heap file registered");
    let index_snap = server.catalog.index_snapshot(table.id);
    let btree = &index_snap.btree;

    let is_dead = |xmin: u32, x: u32| {
        status_map.is_aborted(xmin as u64)
            || (x != 0
                && status_map.is_committed(x as u64)
                && (x as u64) < oldest_active
                && status_map.is_reclaimable_below(x as u64, retention_floor))
    };
    let is_aborted = |xid: u32| status_map.is_aborted(xid as u64);

    let scan = heap.scan().expect("scan");
    let page_ids = scan.page_ids().to_vec();
    drop(scan);

    for page_id in page_ids {
        let Some(frame) = server.buffer_pool.fetch_page(page_id) else {
            continue;
        };
        let mut dead: Vec<(u16, Vec<u8>)> = Vec::new();
        let modified = {
            let mut guard = frame.write_data();
            let data: &mut [u8] = &mut guard[..];
            if HeapPage::heap_header_from_slice(data).slot_count == 0 {
                false
            } else {
                HeapPage::vacuum_in_slice_collect(data, &is_dead, &is_aborted, &mut dead).1
            }
        };
        server.buffer_pool.unpin_page(page_id, modified);
        if !dead.is_empty() {
            zyron_executor::operator::modify::vacuum_index_cleanup(
                table.as_ref(),
                page_id,
                &dead,
                btree,
                &server.btree_indexes,
            );
        }
    }
}

#[tokio::test]
async fn vacuum_removes_dead_index_entries() {
    let (server, schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE t (id INT, v INT)").await;
    exec(&server, &mut session, "CREATE INDEX idx_v ON t (v)").await;
    for i in 1..=6 {
        exec(
            &server,
            &mut session,
            &format!("INSERT INTO t (id, v) VALUES ({i}, {})", i * 10),
        )
        .await;
    }

    let table = server.catalog.get_table(schema, "t").expect("table t");
    let idx = server
        .catalog
        .get_indexes_for_table(table.id)
        .into_iter()
        .next()
        .expect("index");
    let idx_key = idx.id.0;

    // Every inserted row is indexed.
    assert_eq!(
        index_entry_count(&server, idx_key),
        6,
        "inserts populate the index"
    );

    // An MVCC delete keeps the dead row's index entry: a live snapshot could
    // still read the old version, and an index scan rechecks on fetch.
    exec(&server, &mut session, "DELETE FROM t WHERE v = 30").await;
    assert_eq!(
        index_entry_count(&server, idx_key),
        6,
        "index entry is kept right after an MVCC delete"
    );

    // Vacuum reclaims the dead heap tuple and removes its now-unreachable entry.
    run_vacuum(&server, &table).await;
    assert_eq!(
        index_entry_count(&server, idx_key),
        5,
        "vacuum removes the dead row's index entry"
    );

    // No live data was lost: five rows remain and are queryable.
    let live_rows: usize = try_exec(&server, &mut session, "SELECT id FROM t")
        .await
        .unwrap()
        .iter()
        .map(|b| b.num_rows)
        .sum();
    assert_eq!(live_rows, 5, "the five live rows survive vacuum");
}

#[tokio::test]
async fn alter_table_sets_time_travel_retention() {
    let (server, schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE t (id INT)").await;

    // Default: unset (aggressive auto-vacuum).
    let table = server.catalog.get_table(schema, "t").unwrap();
    assert_eq!(table.time_travel_retention_secs, 0);

    // A duration window is stored in seconds and turns on commit-LSN tracking.
    exec(
        &server,
        &mut session,
        "ALTER TABLE t SET (time_travel_retention = '30 days')",
    )
    .await;
    let table = server.catalog.get_table(schema, "t").unwrap();
    assert_eq!(table.time_travel_retention_secs, 30 * 86400);
    assert!(server.txn_manager.status_map().lsn_tracking_enabled());

    // 'unlimited' keeps history forever (sentinel u64::MAX).
    exec(
        &server,
        &mut session,
        "ALTER TABLE t SET (time_travel_retention = 'unlimited')",
    )
    .await;
    assert_eq!(
        server
            .catalog
            .get_table(schema, "t")
            .unwrap()
            .time_travel_retention_secs,
        u64::MAX
    );

    // 'default' returns to the aggressive default.
    exec(
        &server,
        &mut session,
        "ALTER TABLE t SET (time_travel_retention = 'default')",
    )
    .await;
    assert_eq!(
        server
            .catalog
            .get_table(schema, "t")
            .unwrap()
            .time_travel_retention_secs,
        0
    );

    // A garbage duration is rejected, not silently treated as the default.
    assert!(
        try_exec(
            &server,
            &mut session,
            "ALTER TABLE t SET (time_travel_retention = 'soon')",
        )
        .await
        .is_err()
    );
}

#[tokio::test]
async fn retention_floor_gates_vacuum_reclamation() {
    let (server, schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    // Date transactions so a delete's commit LSN is known and comparable.
    server.txn_manager.status_map().enable_lsn_tracking();

    // Two tables that receive identical writes. They differ only in the floor
    // the vacuum is run with, isolating retention as the cause.
    for name in ["t_default", "t_keep"] {
        exec(
            &server,
            &mut session,
            &format!("CREATE TABLE {name} (id INT, v INT)"),
        )
        .await;
        exec(
            &server,
            &mut session,
            &format!("CREATE INDEX idx_{name} ON {name} (v)"),
        )
        .await;
        for i in 1..=4 {
            exec(
                &server,
                &mut session,
                &format!("INSERT INTO {name} (id, v) VALUES ({i}, {})", i * 10),
            )
            .await;
        }
        exec(
            &server,
            &mut session,
            &format!("DELETE FROM {name} WHERE v = 20"),
        )
        .await;
    }

    let t_default = server.catalog.get_table(schema, "t_default").unwrap();
    let t_keep = server.catalog.get_table(schema, "t_keep").unwrap();
    let key = |t: &Arc<zyron_catalog::TableEntry>| {
        server
            .catalog
            .get_indexes_for_table(t.id)
            .into_iter()
            .next()
            .unwrap()
            .id
            .0
    };

    // Default policy -> floor u64::MAX -> the committed delete is reclaimed and
    // its index entry removed.
    run_vacuum_with_floor(&server, &t_default, u64::MAX).await;
    assert_eq!(
        index_entry_count(&server, key(&t_default)),
        3,
        "default retention reclaims the deleted row's index entry"
    );

    // Unlimited policy -> floor 0 -> no committed delete is reclaimable, so the
    // deleted row's version and its index entry are kept for time travel.
    run_vacuum_with_floor(&server, &t_keep, 0).await;
    assert_eq!(
        index_entry_count(&server, key(&t_keep)),
        4,
        "unlimited retention keeps the deleted row's history"
    );
}

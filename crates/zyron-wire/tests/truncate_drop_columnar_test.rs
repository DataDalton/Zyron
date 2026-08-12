//! TRUNCATE and DROP TABLE must reclaim the columnar tier with the heap.
//!
//! A segment is manufactured directly through the storage writer and
//! registered in the catalog, standing in for a background fold, then the
//! DDL runs through the dispatch path and the on-disk and in-memory
//! columnar state is checked. Without the reclaim, truncated folded rows
//! resurrect on the next scan and dropped tables leak .zyr and patch files.
//!
//! Run: cargo test -p zyron-wire --test truncate_drop_columnar_test

use std::sync::Arc;

use zyron_buffer::{BufferPool, BufferPoolConfig};
use zyron_catalog::schema::ColumnarSegmentEntry;
use zyron_catalog::{
    Catalog, CatalogCache, DatabaseId, HeapCatalogStorage, SYSTEM_DATABASE_ID, SchemaId,
};
use zyron_common::TypeId;
use zyron_executor::batch::DataBatch;
use zyron_executor::context::ExecutionContext;
use zyron_storage::columnar::{
    BloomPolicy, ColumnDescriptor, ColumnarPatchManager, CompactionConfig, SYS_COL_ROWID,
    SYS_COL_SUPERSEDE, SYS_COL_XMIN, encode_and_write,
};
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
        WalWriter::new(zyron_bench_harness::wal_config(wal_dir))
        .expect("wal"),
    );
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
        .create_schema(SYSTEM_DATABASE_ID, "public", "test_user")
        .await
        .expect("create public schema");
    // txn counter above the manufactured segment xmin so its rows are
    // visible to every snapshot
    let txn_manager = Arc::new(TransactionManager::with_start_txn_id(Arc::clone(&wal), 100));

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

async fn exec(server: &Arc<ServerState>, session: &mut Option<Session>, sql: &str) {
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
        res.unwrap_or_else(|e| panic!("ddl failed: {sql}\n{e:?}"));
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
    let _: Vec<DataBatch> = zyron_executor::execute(plan, &ctx).await.expect("execute");
    server.txn_manager.commit(&mut txn).await.expect("commit");
}

async fn query_rows(server: &Arc<ServerState>, sql: &str) -> usize {
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
    let batches = zyron_executor::execute(plan, &ctx).await.expect("execute");
    server.txn_manager.commit(&mut txn).await.expect("commit");
    batches.iter().map(|b| b.num_rows).sum()
}

/// Writes a .zyr segment holding the given (k, v) BigInt rows with the three
/// system columns (rowids 0.., xmin 1, supersede 0), registers it in the
/// catalog as file_id 1, and returns the segment path
async fn manufacture_segment(
    server: &Arc<ServerState>,
    schema: SchemaId,
    table_name: &str,
    rows: &[(i64, i64)],
) -> std::path::PathBuf {
    let te = server
        .catalog
        .get_table(schema, table_name)
        .expect("table entry");
    let columnar_dir = server.data_dir.join("columnar");

    let k_cells: Vec<Vec<u8>> = rows.iter().map(|(k, _)| k.to_le_bytes().to_vec()).collect();
    let v_cells: Vec<Vec<u8>> = rows.iter().map(|(_, v)| v.to_le_bytes().to_vec()).collect();
    let rowid_cells: Vec<Vec<u8>> = (0..rows.len() as u64)
        .map(|r| r.to_le_bytes().to_vec())
        .collect();
    let xmin_cells: Vec<Vec<u8>> = rows.iter().map(|_| 1u64.to_le_bytes().to_vec()).collect();
    let super_cells: Vec<Vec<u8>> = rows.iter().map(|_| 0u64.to_le_bytes().to_vec()).collect();
    let all = [k_cells, v_cells, rowid_cells, xmin_cells, super_cells];

    let mut columns: Vec<_> = te.columns.clone();
    columns.sort_by_key(|c| c.ordinal);
    let mut descriptors: Vec<ColumnDescriptor> = columns
        .iter()
        .map(|c| ColumnDescriptor {
            column_id: c.id.0 as u32,
            type_id: c.type_id,
            value_size: 8,
            is_primary_key: false,
            bloom_policy: BloomPolicy::Auto,
        })
        .collect();
    descriptors.push(ColumnDescriptor {
        column_id: SYS_COL_ROWID,
        type_id: TypeId::UInt64,
        value_size: 8,
        is_primary_key: true,
        bloom_policy: BloomPolicy::Auto,
    });
    descriptors.push(ColumnDescriptor {
        column_id: SYS_COL_XMIN,
        type_id: TypeId::UInt64,
        value_size: 8,
        is_primary_key: false,
        bloom_policy: BloomPolicy::Auto,
    });
    descriptors.push(ColumnDescriptor {
        column_id: SYS_COL_SUPERSEDE,
        type_id: TypeId::UInt64,
        value_size: 8,
        is_primary_key: false,
        bloom_policy: BloomPolicy::Auto,
    });

    let cfg = zyron_bench_harness::compaction_config(&columnar_dir);
    let result = encode_and_write(
        &cfg,
        &descriptors,
        rows.len(),
        |ci| all[ci].iter().map(|c| Some(c.as_slice())).collect(),
        zyron_storage::columnar::FileOrdering::Ascending {
            column_id: descriptors[columns.len()].column_id,
        },
        te.id.0 as u64,
        1,
        1,
    )
    .expect("segment write");

    let mut entry = (*te).clone();
    entry.columnar.segments.push(ColumnarSegmentEntry {
        file_id: 1,
        path: result.file_path.to_string_lossy().into_owned(),
        row_count: rows.len() as u64,
        sys_rowid_lo: 0,
        sys_rowid_hi: rows.len() as u64 - 1,
        sys_xmin_lo: 1,
        sys_xmin_hi: 1,
        cluster_spec_id: 0,
    });
    entry.columnar.next_rowid = rows.len() as u64;
    entry.columnar.next_file_id = 2;
    server
        .catalog
        .update_table(entry)
        .await
        .expect("register segment");
    result.file_path
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn truncate_clears_folded_rows_and_reclaims_segments() {
    let (server, schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec(&server, &mut session, "CREATE TABLE t (k BIGINT, v BIGINT)").await;
    exec(
        &server,
        &mut session,
        "INSERT INTO t (k, v) VALUES (1, 10), (2, 20)",
    )
    .await;
    let seg_path = manufacture_segment(&server, schema, "t", &[(100, 1000), (101, 1010)]).await;

    // both tiers are visible before the truncate
    let total = query_rows(&server, "SELECT k FROM t").await;
    assert_eq!(
        total, 4,
        "two heap rows and two folded rows before TRUNCATE"
    );

    // an overlay entry proves the patch store is cleared too, the
    // uncommitted supersede leaves the row visible
    let te = server.catalog.get_table(schema, "t").expect("entry");
    let store = ColumnarPatchManager::store_for_segment(
        te.id.0 as u64,
        std::path::Path::new(&te.columnar.segments[0].path),
    )
    .expect("store");
    store
        .append_supersede(0, 1, 0, 9999, 1)
        .expect("overlay entry");
    assert!(!store.file_overlay(1).is_empty(), "overlay present");

    exec(&server, &mut session, "TRUNCATE TABLE t").await;

    let te = server.catalog.get_table(schema, "t").expect("entry");
    assert!(
        te.columnar.segments.is_empty(),
        "columnar registry cleared by TRUNCATE"
    );
    assert!(!seg_path.exists(), "segment file removed by TRUNCATE");
    assert!(
        store.file_overlay(1).is_empty(),
        "patch overlay cleared by TRUNCATE"
    );
    let total = query_rows(&server, "SELECT k FROM t").await;
    assert_eq!(total, 0, "no rows resurrect after TRUNCATE");

    // the table stays usable
    exec(&server, &mut session, "INSERT INTO t (k, v) VALUES (7, 70)").await;
    let total = query_rows(&server, "SELECT k FROM t").await;
    assert_eq!(total, 1, "fresh insert visible after TRUNCATE");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn drop_table_reclaims_segments_and_patch_store() {
    let (server, schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec(&server, &mut session, "CREATE TABLE d (k BIGINT, v BIGINT)").await;
    exec(&server, &mut session, "INSERT INTO d (k, v) VALUES (1, 10)").await;
    let seg_path = manufacture_segment(&server, schema, "d", &[(100, 1000)]).await;

    let te = server.catalog.get_table(schema, "d").expect("entry");
    let table_id = te.id.0 as u64;
    let columnar_dir = server.data_dir.join("columnar");
    let patch_path = columnar_dir.join(format!("{}.zyrpatch", table_id));
    {
        let store = ColumnarPatchManager::store_for_segment(
            table_id,
            std::path::Path::new(&te.columnar.segments[0].path),
        )
        .expect("store");
        store
            .append_supersede(0, 1, 0, 9999, 1)
            .expect("overlay entry");
    }
    assert!(patch_path.exists(), "patch file exists before DROP");

    exec(&server, &mut session, "DROP TABLE d").await;

    assert!(
        server.catalog.get_table(schema, "d").is_err(),
        "catalog entry gone"
    );
    assert!(!seg_path.exists(), "segment file removed by DROP TABLE");
    assert!(
        !patch_path.exists(),
        "patch store file removed by DROP TABLE"
    );
    let rids = seg_path.with_extension("zyrrids");
    assert!(!rids.exists(), "no RID sidecar left behind");
}

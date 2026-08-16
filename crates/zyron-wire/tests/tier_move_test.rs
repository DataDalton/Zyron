//! Tiered storage relocation: `ALTER TABLE ... MOVE ... TO TIER`.
//!
//! A tier is a directory the operator points at storage of a given speed.
//! Relocation moves a whole `.zyr` segment file into `<columnar>/tiers/<name>/`
//! and records where it went, so nothing about the file changes and a read of
//! a relocated segment is the same positioned read as before.
//!
//! Two things have to hold for that to be a move rather than data loss.
//! Rows in a relocated segment must still read back, and the table's patch
//! log must still be reachable from the segment's new location: the log lives
//! beside the columnar root, and resolving it from the segment's parent
//! directory would give a table two logs the moment one of its segments moved.
//!
//! Segments are manufactured through the storage writer and registered, the
//! same stand-in for a background fold that the truncate/drop test uses.
//!
//! Run: cargo test -p zyron-wire --test tier_move_test

use std::sync::Arc;

use zyron_buffer::BufferPool;
use zyron_catalog::schema::ColumnarSegmentEntry;
use zyron_catalog::{
    Catalog, CatalogCache, DatabaseId, HeapCatalogStorage, SYSTEM_DATABASE_ID, SchemaId,
};
use zyron_common::TypeId;
use zyron_executor::batch::DataBatch;
use zyron_executor::context::ExecutionContext;
use zyron_storage::DiskManager;
use zyron_storage::columnar::{
    BloomPolicy, ColumnDescriptor, SYS_COL_ROWID, SYS_COL_SUPERSEDE, SYS_COL_XMIN, encode_and_write,
};
use zyron_storage::txn::{IsolationLevel, TransactionManager};
use zyron_wal::WalWriter;
use zyron_wire::connection::ServerState;
use zyron_wire::session::Session;

async fn create_test_server() -> (Arc<ServerState>, SchemaId, tempfile::TempDir) {
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
    let public_schema = catalog
        .create_schema(SYSTEM_DATABASE_ID, "public", "test_user")
        .await
        .expect("create public schema");
    // Transaction counter above the manufactured segment xmin so its rows are
    // visible to every snapshot the test takes
    let txn_manager = Arc::new(TransactionManager::with_start_txn_id(Arc::clone(&wal), 100));

    let state = Arc::new(ServerState {
        catalog,
        wal,
        buffer_pool: pool,
        disk_manager: disk,
        txn_manager,
        doc_registry: Arc::new(zyron_common::DocRegistry::new()),
        table_io_stats: Arc::new(zyron_common::TableIOStatsRegistry::new()),
        index_io_stats: Arc::new(zyron_common::IndexIOStatsRegistry::new()),
        columnar_maintenance: None,
        security_manager: None,
        key_store: Arc::new(zyron_auth::LocalKeyStore::new([0u8; 32])),
        config_lookup: None,
        config_all: None,
        data_dir: data_dir.clone(),
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
    (state, public_schema, tmp)
}

fn new_session() -> Option<Session> {
    let mut s = Session::new("test_user".into(), "testdb".into(), DatabaseId(1));
    s.search_path = vec!["public".into()];
    Some(s)
}

/// Runs a statement, taking the DDL/utility path when it claims the statement
/// and the plan/execute path otherwise. Returns the command tag for DDL.
async fn exec(server: &Arc<ServerState>, session: &mut Option<Session>, sql: &str) -> String {
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
        let out = res.unwrap_or_else(|e| panic!("ddl failed: {sql}\n{e:?}"));
        return format!("{out:?}");
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
    String::new()
}

/// Every (k, v) pair a query returns, ordered by the query.
async fn query_pairs(server: &Arc<ServerState>, sql: &str) -> Vec<(i64, i64)> {
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

    let mut out = Vec::new();
    for b in &batches {
        for r in 0..b.num_rows {
            let k = match b.column(0).get_scalar(r) {
                zyron_executor::column::ScalarValue::Int64(v) => v,
                other => panic!("expected k as a 64 bit integer, got {other:?}"),
            };
            let v = match b.column(1).get_scalar(r) {
                zyron_executor::column::ScalarValue::Int64(v) => v,
                other => panic!("expected v as a 64 bit integer, got {other:?}"),
            };
            out.push((k, v));
        }
    }
    out
}

/// Writes a .zyr segment holding the given (k, v) rows with the three system
/// columns (rowids from `base_rowid`, xmin 1, supersede 0) and registers it.
async fn manufacture_segment(
    server: &Arc<ServerState>,
    schema: SchemaId,
    table_name: &str,
    file_id: u64,
    base_rowid: u64,
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
        .map(|r| (base_rowid + r).to_le_bytes().to_vec())
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
    for (column_id, is_primary_key) in [
        (SYS_COL_ROWID, true),
        (SYS_COL_XMIN, false),
        (SYS_COL_SUPERSEDE, false),
    ] {
        descriptors.push(ColumnDescriptor {
            column_id,
            type_id: TypeId::UInt64,
            value_size: 8,
            is_primary_key,
            bloom_policy: BloomPolicy::Auto,
        });
    }

    let cfg = zyron_bench_harness::compaction_config(&columnar_dir);
    let result = encode_and_write(
        &cfg,
        &descriptors,
        rows.len(),
        |ci| all[ci].iter().map(|c| Some(c.as_slice())).collect(),
        zyron_storage::columnar::FileOrdering::Ascending {
            column_id: SYS_COL_ROWID,
        },
        te.id.0 as u64,
        1,
        1,
    )
    .expect("segment write");

    let mut entry = (*te).clone();
    entry.columnar.segments.push(ColumnarSegmentEntry {
        file_id,
        path: result.file_path.to_string_lossy().into_owned(),
        row_count: rows.len() as u64,
        sys_rowid_lo: base_rowid,
        sys_rowid_hi: base_rowid + rows.len() as u64 - 1,
        sys_xmin_lo: 1,
        sys_xmin_hi: 1,
        cluster_spec_id: 0,
        storage_tier: 0,
    });
    entry.columnar.next_rowid = base_rowid + rows.len() as u64;
    entry.columnar.next_file_id = file_id + 1;
    server
        .catalog
        .update_table(entry)
        .await
        .expect("register segment");
    result.file_path
}

/// The recorded (file_id, path, tier) of every registered segment.
fn segment_state(
    server: &Arc<ServerState>,
    schema: SchemaId,
    table: &str,
) -> Vec<(u64, std::path::PathBuf, u8)> {
    server
        .catalog
        .get_table(schema, table)
        .expect("table")
        .columnar
        .segments
        .iter()
        .map(|s| (s.file_id, std::path::PathBuf::from(&s.path), s.storage_tier))
        .collect()
}

/// Two segments, one entirely below k = 100 and one entirely above.
async fn two_segment_table(
    server: &Arc<ServerState>,
    schema: SchemaId,
    session: &mut Option<Session>,
) {
    exec(server, session, "CREATE TABLE t (k BIGINT, v BIGINT)").await;
    manufacture_segment(server, schema, "t", 1, 0, &[(1, 10), (2, 20), (3, 30)]).await;
    manufacture_segment(server, schema, "t", 2, 3, &[(500, 5000), (600, 6000)]).await;
}

/// The covered segment moves into the tier directory, the uncovered one does
/// not, and every row still reads back with the value it had.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_a_fully_covered_segment_relocates_and_still_reads() {
    let (server, schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    two_segment_table(&server, schema, &mut session).await;

    let before = query_pairs(&server, "SELECT k, v FROM t ORDER BY k").await;
    assert_eq!(
        before,
        vec![(1, 10), (2, 20), (3, 30), (500, 5000), (600, 6000)]
    );

    exec(
        &server,
        &mut session,
        "ALTER TABLE t MOVE WHERE k < 100 TO TIER 'cold'",
    )
    .await;

    let state = segment_state(&server, schema, "t");
    let moved: Vec<_> = state.iter().filter(|(_, _, tier)| *tier == 2).collect();
    assert_eq!(moved.len(), 1, "only the covered segment moves: {state:?}");
    assert_eq!(moved[0].0, 1, "the segment holding k < 100");
    assert!(
        moved[0].1.exists(),
        "the file is where the catalog says it is"
    );
    assert!(
        moved[0]
            .1
            .starts_with(server.data_dir.join("columnar").join("tiers").join("cold")),
        "relocated into the tier directory: {:?}",
        moved[0].1
    );
    for (file_id, path, tier) in &state {
        assert!(path.exists(), "segment {file_id} vanished");
        if *file_id == 2 {
            assert_eq!(*tier, 0, "the uncovered segment stays hot");
        }
    }

    assert_eq!(
        query_pairs(&server, "SELECT k, v FROM t ORDER BY k").await,
        before,
        "a relocated segment reads back exactly what it held"
    );
}

/// A predicate that only part of a segment satisfies moves nothing: the unit
/// that relocates is the whole file, and moving it would drag rows the
/// operator did not name onto a colder tier.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_a_partly_covered_segment_is_left_alone() {
    let (server, schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    two_segment_table(&server, schema, &mut session).await;

    exec(
        &server,
        &mut session,
        "ALTER TABLE t MOVE WHERE k < 2 TO TIER 'cold'",
    )
    .await;

    for (file_id, _, tier) in segment_state(&server, schema, "t") {
        assert_eq!(tier, 0, "segment {file_id} moved on a partial match");
    }
}

/// DRY RUN reports what would move and moves nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_dry_run_reports_without_moving() {
    let (server, schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    two_segment_table(&server, schema, &mut session).await;
    let before = segment_state(&server, schema, "t");

    let tag = exec(
        &server,
        &mut session,
        "ALTER TABLE t MOVE WHERE k < 100 TO TIER 'cold' DRY RUN",
    )
    .await;
    assert!(
        tag.contains("DRY RUN 1"),
        "one segment would move, tag was {tag}"
    );
    assert_eq!(
        segment_state(&server, schema, "t"),
        before,
        "DRY RUN changed the registry"
    );
}

/// Moving back to hot returns the file to the columnar root, and a repeated
/// move reports zero because the segment is already where it was asked to go.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_a_relocated_segment_moves_back_and_a_repeat_is_a_no_op() {
    let (server, schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    two_segment_table(&server, schema, &mut session).await;

    exec(
        &server,
        &mut session,
        "ALTER TABLE t MOVE WHERE k < 100 TO TIER 'cold'",
    )
    .await;
    let repeat = exec(
        &server,
        &mut session,
        "ALTER TABLE t MOVE WHERE k < 100 TO TIER 'cold'",
    )
    .await;
    assert!(
        repeat.contains("MOVE 0"),
        "a segment already on the tier is not moved again, tag was {repeat}"
    );

    exec(
        &server,
        &mut session,
        "ALTER TABLE t MOVE WHERE k < 100 TO TIER 'hot'",
    )
    .await;
    let columnar_dir = server.data_dir.join("columnar");
    for (file_id, path, tier) in segment_state(&server, schema, "t") {
        assert_eq!(tier, 0, "segment {file_id} is hot again");
        assert_eq!(
            path.parent(),
            Some(columnar_dir.as_path()),
            "segment {file_id} returned to the columnar root"
        );
        assert!(path.exists());
    }
    assert_eq!(
        query_pairs(&server, "SELECT k, v FROM t ORDER BY k").await,
        vec![(1, 10), (2, 20), (3, 30), (500, 5000), (600, 6000)]
    );
}

/// The patch log lives beside the columnar root, so a row in a relocated
/// segment must still take an UPDATE and read back the new value. Resolving
/// the log from the segment's parent directory instead would give the table
/// a second, empty log once any segment moved, and the update would be
/// written where no scan looks for it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_a_row_in_a_relocated_segment_still_takes_an_update() {
    let (server, schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    two_segment_table(&server, schema, &mut session).await;

    exec(
        &server,
        &mut session,
        "ALTER TABLE t MOVE WHERE k < 100 TO TIER 'cold'",
    )
    .await;
    exec(&server, &mut session, "UPDATE t SET v = 999 WHERE k = 2").await;

    assert_eq!(
        query_pairs(&server, "SELECT k, v FROM t ORDER BY k").await,
        vec![(1, 10), (2, 999), (3, 30), (500, 5000), (600, 6000)],
        "the patch resolved against the relocated segment"
    );

    exec(&server, &mut session, "DELETE FROM t WHERE k = 3").await;
    assert_eq!(
        query_pairs(&server, "SELECT k, v FROM t ORDER BY k").await,
        vec![(1, 10), (2, 999), (500, 5000), (600, 6000)],
        "the supersede resolved against the relocated segment"
    );
}

/// `MOVE PARTITION 'col=value'` desugars to the equality it stands for, so a
/// text value is compared as a string rather than read as an identifier.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_the_partition_form_moves_by_equality() {
    let (server, schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE t (k BIGINT, v BIGINT)").await;
    manufacture_segment(&server, schema, "t", 1, 0, &[(7, 10), (7, 20)]).await;
    manufacture_segment(&server, schema, "t", 2, 2, &[(8, 30)]).await;

    exec(
        &server,
        &mut session,
        "ALTER TABLE t MOVE PARTITION 'k=7' TO TIER 'warm'",
    )
    .await;

    let state = segment_state(&server, schema, "t");
    let warm: Vec<_> = state.iter().filter(|(_, _, tier)| *tier == 1).collect();
    assert_eq!(warm.len(), 1, "only the k = 7 segment moves: {state:?}");
    assert_eq!(warm[0].0, 1);
    assert_eq!(
        query_pairs(&server, "SELECT k, v FROM t ORDER BY k, v").await,
        vec![(7, 10), (7, 20), (8, 30)]
    );
}

/// `cold_after` relocates segments whose rows have all aged past the
/// threshold when the retention job runs, and leaves younger ones hot.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_cold_after_relocates_aged_segments_on_a_retention_run() {
    let (server, schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE t (k BIGINT, v BIGINT)").await;

    // k stands in for the row's age in microseconds. One segment is well
    // past a one-hour threshold, the other is in the future
    let now_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0);
    let old = now_us - 86_400 * 1_000_000;
    let fresh = now_us + 86_400 * 1_000_000;
    manufacture_segment(&server, schema, "t", 1, 0, &[(old, 1), (old + 1, 2)]).await;
    manufacture_segment(&server, schema, "t", 2, 2, &[(fresh, 3)]).await;

    // A TTL far enough out that the retention pass deletes nothing, so what
    // the assertions see is the tiering alone
    exec(
        &server,
        &mut session,
        "ALTER TABLE t SET TTL 36500 DAYS ON k",
    )
    .await;
    exec(
        &server,
        &mut session,
        "ALTER TABLE t SET (cold_after = '1 hour')",
    )
    .await;

    exec(&server, &mut session, "RUN RETENTION JOB ON t").await;

    let state = segment_state(&server, schema, "t");
    let cold: Vec<_> = state.iter().filter(|(_, _, tier)| *tier == 2).collect();
    assert_eq!(cold.len(), 1, "only the aged segment relocates: {state:?}");
    assert_eq!(cold[0].0, 1);
    assert_eq!(
        query_pairs(&server, "SELECT k, v FROM t ORDER BY v").await,
        vec![(old, 1), (old + 1, 2), (fresh, 3)],
        "an aged, relocated segment still reads"
    );
}

/// A TTL declared on a table's first column expires the rows it governs.
///
/// Column ids start at zero and zero was also the marker for "no policy", so
/// a TTL on the first column read back as no TTL at all: the retention job
/// skipped the table and the rows it was meant to expire lived forever. The
/// statement reported success the whole time.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_a_ttl_on_the_first_column_expires_rows() {
    let (server, schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(
        &server,
        &mut session,
        "CREATE TABLE t (ts BIGINT, v BIGINT)",
    )
    .await;

    let now_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0);
    let old = now_us - 30 * 86_400 * 1_000_000;
    let fresh = now_us;
    exec(
        &server,
        &mut session,
        &format!("INSERT INTO t VALUES ({old}, 1), ({fresh}, 2)"),
    )
    .await;

    // ts is the first column, so its ColumnId is zero
    assert_eq!(
        server
            .catalog
            .get_table(schema, "t")
            .expect("table")
            .columns[0]
            .id
            .0,
        0
    );
    exec(&server, &mut session, "ALTER TABLE t SET TTL 1 DAYS ON ts").await;
    exec(&server, &mut session, "RUN RETENTION JOB ON t").await;

    assert_eq!(
        query_pairs(&server, "SELECT ts, v FROM t").await,
        vec![(fresh, 2)],
        "the aged row expired and the fresh one did not"
    );
}

/// `purge_after_soft_delete` sets the grace window the retention worker
/// reads before physically purging a soft-deleted row. Writing it to the
/// archive threshold instead left the grace at zero, so a table declaring
/// the option purged on the next pass rather than holding for the window.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_purge_after_soft_delete_sets_the_purge_grace() {
    let (server, schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE t (k BIGINT, v BIGINT)").await;
    exec(
        &server,
        &mut session,
        "ALTER TABLE t SET (purge_after_soft_delete = '30 days')",
    )
    .await;

    let lc = server
        .catalog
        .get_table(schema, "t")
        .expect("table")
        .lifecycle
        .clone();
    assert_eq!(lc.purge_grace_seconds, 30 * 86_400);
    assert_eq!(
        lc.archive_after_seconds, 0,
        "the purge window is not an archive threshold"
    );
}

/// A move that fails part way keeps the registry true for what already
/// moved. The first segment's file sits at its new path when the second
/// segment's rename fails, so dropping the registry edits would leave the
/// catalog naming a path with no file behind it and every read of that
/// segment failing until a restart.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_a_failed_move_keeps_the_registry_true_for_what_already_moved() {
    let (server, schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    two_segment_table(&server, schema, &mut session).await;
    let before = segment_state(&server, schema, "t");

    // Both segments are covered by the predicate. A directory planted under
    // the second segment's exact target name makes its rename fail after
    // the first segment has already moved
    let cold_dir = server.data_dir.join("columnar").join("tiers").join("cold");
    let blocked = cold_dir.join(before[1].1.file_name().expect("file name"));
    std::fs::create_dir_all(&blocked).expect("blocking directory");

    let sql = "ALTER TABLE t MOVE WHERE k < 1000 TO TIER 'cold'";
    let stmt = zyron_parser::parse(sql)
        .expect("parse")
        .into_iter()
        .next()
        .expect("one statement");
    let mut txn_opt: Option<zyron_storage::txn::Transaction> = None;
    let mut active_branch: Option<String> = None;
    let res = zyron_wire::ddl_dispatch::try_handle_ddl_utility(
        &stmt,
        &server,
        &mut session,
        &mut txn_opt,
        &mut active_branch,
        sql,
    )
    .await
    .expect("the move statement is claimed by the DDL path");
    res.expect_err("the blocked rename must surface as an error");

    let state = segment_state(&server, schema, "t");
    assert_eq!(
        state[0].2, 2,
        "the completed move is recorded despite the failure: {state:?}"
    );
    assert!(
        state[0].1.starts_with(&cold_dir),
        "the registry names the tier path: {:?}",
        state[0].1
    );
    assert!(state[0].1.exists(), "the file is where the registry says");
    assert_eq!(state[1].2, 0, "the blocked segment stays hot");
    assert_eq!(
        state[1].1, before[1].1,
        "the blocked segment keeps its path"
    );
    assert!(state[1].1.exists());
    assert!(
        !blocked.with_extension("zyr.moving").exists(),
        "the failed fallback removed its staging file"
    );

    assert_eq!(
        query_pairs(&server, "SELECT k, v FROM t ORDER BY k").await,
        vec![(1, 10), (2, 20), (3, 30), (500, 5000), (600, 6000)],
        "every row reads back through the registry after the failure"
    );
}

/// An unknown tier name is refused rather than silently creating a directory
/// nobody configured.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_an_unknown_tier_is_refused() {
    let (server, schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    two_segment_table(&server, schema, &mut session).await;

    let stmt = zyron_parser::parse("ALTER TABLE t MOVE WHERE k < 100 TO TIER 'frozen'")
        .expect("parse")
        .into_iter()
        .next()
        .expect("one statement");
    let mut txn_opt: Option<zyron_storage::txn::Transaction> = None;
    let mut active_branch: Option<String> = None;
    let res = zyron_wire::ddl_dispatch::try_handle_ddl_utility(
        &stmt,
        &server,
        &mut session,
        &mut txn_opt,
        &mut active_branch,
        "ALTER TABLE t MOVE WHERE k < 100 TO TIER 'frozen'",
    )
    .await
    .expect("the move statement is claimed by the DDL path");
    let err = res.expect_err("an unknown tier must be refused");
    assert!(
        format!("{err:?}").contains("frozen"),
        "the refusal names the tier: {err:?}"
    );
    for (_, _, tier) in segment_state(&server, schema, "t") {
        assert_eq!(tier, 0);
    }
}

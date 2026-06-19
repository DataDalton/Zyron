//! Integration tests for inbound CDC ingestion.
//!
//! Exercises CREATE/DROP CDC INGEST through the DDL dispatch path and the
//! record applier (apply_ingest_records) directly: decode -> INSERT/UPSERT/
//! DELETE -> dead letter routing. The live Kafka/S3 source fetch is not
//! exercised here (it needs external services); the applier is fed decoded
//! record bytes built with the native decoder.
//!
//! Run: cargo test -p zyron-wire --test cdc_ingest_test -- --nocapture

use std::sync::Arc;

use zyron_buffer::{BufferPool, BufferPoolConfig};
use zyron_catalog::{
    Catalog, CatalogCache, DatabaseId, HeapCatalogStorage, SYSTEM_DATABASE_ID, SchemaId,
};
use zyron_cdc::ChangeType;
use zyron_cdc::cdc_ingest::{CdcIngestConfig, CdcIngestManager, CdcIngestSource, OnConflict};
use zyron_cdc::decoder::{DecodedChange, DecoderPlugin, LogicalDecoder, ZyronCdcDecoder};
use zyron_executor::batch::DataBatch;
use zyron_executor::column::ScalarValue;
use zyron_executor::context::ExecutionContext;
use zyron_storage::txn::{IsolationLevel, TransactionManager};
use zyron_storage::{DiskManager, DiskManagerConfig};
use zyron_wal::{WalWriter, WalWriterConfig};
use zyron_wire::connection::ServerState;
use zyron_wire::ddl_dispatch::apply_ingest_records;
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
            data_dir: data_dir.clone(),
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
    let ingest_mgr = Arc::new(CdcIngestManager::new(&data_dir).expect("ingest mgr"));

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
        cdc_ingest_manager: Some(ingest_mgr),
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

fn col_str(batches: &[DataBatch], idx: usize) -> Vec<String> {
    let mut out = Vec::new();
    for b in batches {
        if let Some(col) = b.columns.get(idx) {
            for r in 0..b.num_rows {
                match col.get_scalar(r) {
                    ScalarValue::Utf8(s) => out.push(s),
                    ScalarValue::Null => out.push(String::new()),
                    other => panic!("expected string, got {other:?}"),
                }
            }
        }
    }
    out
}

fn row_count(batches: &[DataBatch]) -> usize {
    batches.iter().map(|b| b.num_rows).sum()
}

fn table_id(server: &Arc<ServerState>, schema: SchemaId, name: &str) -> u32 {
    server.catalog.get_table(schema, name).expect("table").id.0
}

/// Serializes a change event to native-decoder bytes for the applier.
fn record(op: ChangeType, new: Option<&[(&str, &str)]>, old: Option<&[(&str, &str)]>) -> Vec<u8> {
    let to_pairs = |v: &[(&str, &str)]| {
        v.iter()
            .map(|(k, val)| (k.to_string(), val.to_string()))
            .collect::<Vec<_>>()
    };
    let change = DecodedChange {
        table_name: "t".into(),
        table_id: 0,
        operation: op,
        old_values: old.map(to_pairs),
        new_values: new.map(to_pairs),
        commit_lsn: 0,
        commit_timestamp: 0,
        txn_id: 0,
        is_last_in_txn: true,
        schema_version: 0,
    };
    ZyronCdcDecoder.serialize(&change).unwrap().to_vec()
}

fn config(target: u32, on_conflict: OnConflict, dead_letter: Option<u32>) -> CdcIngestConfig {
    CdcIngestConfig {
        name: "ing".into(),
        source: CdcIngestSource::Kafka {
            brokers: "localhost:9092".into(),
            topic: "events".into(),
            group_id: "g".into(),
            start_offset: None,
        },
        target_table_id: target,
        primary_key_columns: vec!["id".into()],
        on_conflict,
        dead_letter_table_id: dead_letter,
        decoder: DecoderPlugin::ZyronCdc,
        batch_size: 100,
        active: true,
    }
}

#[tokio::test]
async fn create_and_drop_ingest_persists_config() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(
        &server,
        &mut session,
        "CREATE TABLE t (id INT PRIMARY KEY, name VARCHAR)",
    )
    .await;
    exec(
        &server,
        &mut session,
        "CREATE CDC INGEST i FROM kafka INTO t WITH (brokers = 'h:9092', topic = 'evt', primary_key = 'id')",
    )
    .await;
    let mgr = server.cdc_ingest_manager.as_ref().unwrap();
    assert!(mgr.get_ingest("i").is_ok());
    assert_eq!(
        mgr.get_ingest("i").unwrap().target_table_id,
        table_id(&server, _schema, "t")
    );

    exec(&server, &mut session, "DROP CDC INGEST i").await;
    assert!(mgr.get_ingest("i").is_err());
}

#[tokio::test]
async fn create_ingest_missing_required_option_errors() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(&server, &mut session, "CREATE TABLE t (id INT PRIMARY KEY)").await;
    // Kafka source requires brokers and topic.
    assert!(
        try_exec(
            &server,
            &mut session,
            "CREATE CDC INGEST i FROM kafka INTO t WITH (topic = 'evt')"
        )
        .await
        .is_err()
    );
}

#[tokio::test]
async fn apply_inserts_rows() {
    let (server, schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(
        &server,
        &mut session,
        "CREATE TABLE t (id INT PRIMARY KEY, name VARCHAR)",
    )
    .await;
    let tid = table_id(&server, schema, "t");
    let cfg = config(tid, OnConflict::Error, None);

    let records = vec![
        record(
            ChangeType::Insert,
            Some(&[("id", "1"), ("name", "alice")]),
            None,
        ),
        record(
            ChangeType::Insert,
            Some(&[("id", "2"), ("name", "bob")]),
            None,
        ),
    ];
    let report = apply_ingest_records(&server, &cfg, &records).await;
    assert_eq!(report.applied, 2);
    assert_eq!(report.failed, 0);

    let mut ids = col_i64(&exec(&server, &mut session, "SELECT id FROM t").await, 0);
    ids.sort();
    assert_eq!(ids, vec![1, 2]);
}

#[tokio::test]
async fn apply_upserts_on_conflict_update() {
    let (server, schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(
        &server,
        &mut session,
        "CREATE TABLE t (id INT PRIMARY KEY, name VARCHAR)",
    )
    .await;
    let tid = table_id(&server, schema, "t");
    let cfg = config(tid, OnConflict::Update, None);

    apply_ingest_records(
        &server,
        &cfg,
        &[record(
            ChangeType::Insert,
            Some(&[("id", "1"), ("name", "alice")]),
            None,
        )],
    )
    .await;
    // Update post-image carries the new value plus the old key image.
    let report = apply_ingest_records(
        &server,
        &cfg,
        &[record(
            ChangeType::UpdatePostimage,
            Some(&[("id", "1"), ("name", "alice2")]),
            Some(&[("id", "1"), ("name", "alice")]),
        )],
    )
    .await;
    assert_eq!(report.applied, 1);

    assert_eq!(
        row_count(&exec(&server, &mut session, "SELECT id FROM t").await),
        1
    );
    let names = col_str(&exec(&server, &mut session, "SELECT name FROM t").await, 0);
    assert_eq!(names, vec!["alice2".to_string()]);
}

#[tokio::test]
async fn apply_skips_existing_on_conflict_skip() {
    let (server, schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(
        &server,
        &mut session,
        "CREATE TABLE t (id INT PRIMARY KEY, name VARCHAR)",
    )
    .await;
    let tid = table_id(&server, schema, "t");

    apply_ingest_records(
        &server,
        &config(tid, OnConflict::Error, None),
        &[record(
            ChangeType::Insert,
            Some(&[("id", "1"), ("name", "alice")]),
            None,
        )],
    )
    .await;
    // Skip leaves the existing row untouched.
    let report = apply_ingest_records(
        &server,
        &config(tid, OnConflict::Skip, None),
        &[record(
            ChangeType::Insert,
            Some(&[("id", "1"), ("name", "zzz")]),
            None,
        )],
    )
    .await;
    assert_eq!(report.skipped, 1);
    assert_eq!(report.applied, 0);

    let names = col_str(&exec(&server, &mut session, "SELECT name FROM t").await, 0);
    assert_eq!(names, vec!["alice".to_string()]);
}

#[tokio::test]
async fn apply_deletes_by_key() {
    let (server, schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(
        &server,
        &mut session,
        "CREATE TABLE t (id INT PRIMARY KEY, name VARCHAR)",
    )
    .await;
    let tid = table_id(&server, schema, "t");
    let cfg = config(tid, OnConflict::Update, None);

    apply_ingest_records(
        &server,
        &cfg,
        &[
            record(
                ChangeType::Insert,
                Some(&[("id", "1"), ("name", "a")]),
                None,
            ),
            record(
                ChangeType::Insert,
                Some(&[("id", "2"), ("name", "b")]),
                None,
            ),
        ],
    )
    .await;
    let report = apply_ingest_records(
        &server,
        &cfg,
        &[record(
            ChangeType::Delete,
            None,
            Some(&[("id", "1"), ("name", "a")]),
        )],
    )
    .await;
    assert_eq!(report.applied, 1);

    let ids = col_i64(&exec(&server, &mut session, "SELECT id FROM t").await, 0);
    assert_eq!(ids, vec![2]);
}

#[tokio::test]
async fn undecodable_record_routes_to_dead_letter() {
    let (server, schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(
        &server,
        &mut session,
        "CREATE TABLE t (id INT PRIMARY KEY, name VARCHAR)",
    )
    .await;
    exec(
        &server,
        &mut session,
        "CREATE TABLE dlq (payload TEXT, error TEXT)",
    )
    .await;
    let tid = table_id(&server, schema, "t");
    let dlq_id = table_id(&server, schema, "dlq");
    let cfg = config(tid, OnConflict::Error, Some(dlq_id));

    // A claimed table-name length far past the buffer fails to decode.
    let garbage = vec![0xFFu8, 0xFF];
    let report = apply_ingest_records(&server, &cfg, &[garbage]).await;
    assert_eq!(report.failed, 1);
    assert_eq!(report.applied, 0);

    // The dead letter table captured the failed payload.
    assert_eq!(
        row_count(&exec(&server, &mut session, "SELECT payload FROM dlq").await),
        1
    );
    // The target stayed empty.
    assert_eq!(
        row_count(&exec(&server, &mut session, "SELECT id FROM t").await),
        0
    );
}

#[tokio::test]
async fn apply_without_primary_key_inserts() {
    let (server, schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec(
        &server,
        &mut session,
        "CREATE TABLE t (id INT, name VARCHAR)",
    )
    .await;
    let tid = table_id(&server, schema, "t");
    // No declared key: each insert just appends.
    let mut cfg = config(tid, OnConflict::Update, None);
    cfg.primary_key_columns = Vec::new();

    let report = apply_ingest_records(
        &server,
        &cfg,
        &[
            record(
                ChangeType::Insert,
                Some(&[("id", "1"), ("name", "a")]),
                None,
            ),
            record(
                ChangeType::Insert,
                Some(&[("id", "1"), ("name", "a")]),
                None,
            ),
        ],
    )
    .await;
    assert_eq!(report.applied, 2);
    assert_eq!(
        row_count(&exec(&server, &mut session, "SELECT id FROM t").await),
        2
    );
}

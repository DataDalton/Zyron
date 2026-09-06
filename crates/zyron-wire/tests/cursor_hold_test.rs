//! Integration tests for WITH HOLD cursor enforcement at COMMIT.
//!
//! Drives a real Connection over TCP with the simple query protocol and
//! verifies: a WITH HOLD cursor materializes at COMMIT and stays fetchable in
//! a later transaction, a WITHOUT HOLD cursor closes at COMMIT, a held cursor
//! keeps its snapshot (rows inserted after COMMIT never appear), fetch
//! position survives COMMIT, and ROLLBACK closes a cursor the transaction
//! declared.
//!
//! Run: cargo test -p zyron-wire --test cursor_hold_test -- --nocapture

use std::sync::Arc;

use zyron_buffer::BufferPool;
use zyron_catalog::{Catalog, CatalogCache, HeapCatalogStorage, SchemaId};
use zyron_storage::DiskManager;
use zyron_storage::txn::TransactionManager;
use zyron_wal::WalWriter;
use zyron_wire::connection::ServerState;

async fn create_test_server() -> (Arc<ServerState>, SchemaId, tempfile::TempDir) {
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let data_dir = tmp.path().join("data");
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir_all(&data_dir).unwrap();
    std::fs::create_dir_all(&wal_dir).unwrap();

    let wal = Arc::new(WalWriter::new(zyron_bench_harness::wal_config(wal_dir)).expect("wal"));
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
    // The wire startup resolves the requested database against the catalog,
    // so the zyron_test schema lives under testdb; each connection then runs
    // SET search_path = zyron_test because the default path targets zyron_sys.
    let testdb_id = catalog
        .create_database("testdb", "test_user")
        .await
        .expect("create testdb");
    let public_schema = catalog
        .create_schema(testdb_id, "zyron_test", "test_user")
        .await
        .expect("create zyron_test schema");
    zyron_catalog::SystemCatalog::init(&catalog)
        .await
        .expect("init zyron_sys");
    let txn_manager = Arc::new(TransactionManager::new(Arc::clone(&wal)));

    let state = Arc::new(ServerState {
        raft: None,
        replication: None,
        node_capabilities: None,
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
        media_store: Arc::new(
            zyron_media::store::MediaStore::open(tmp.path().join("data"))
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
        cancel_registry: Default::default(),
        heap_files: Arc::new(scc::HashMap::new()),
        btree_indexes: Arc::new(scc::HashMap::new()),
        plan_cache: Arc::new(zyron_wire::plan_cache::ServerPlanCache::new()),
        vacuum_running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        analytics_registry: zyron_analytics::default_registry(),
        legal_holds: Arc::new(zyron_lifecycle::legal_hold::LegalHoldRegistry::new()),
        dlq_registry: Arc::new(zyron_streaming::dlq::DlqRegistry::new()),
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
        max_query_memory: None,
        spill_directory: None,
        balloon_params: None,
        default_auth_method: zyron_auth::auth_rules::AuthMethod::Trust,
        password_encryption: "balloon-sha-256".into(),
        admission: Arc::new(zyron_common::Admission::new()),
        query_metrics: Arc::new(zyron_common::QueryMetrics::new()),
        upgrade_control: None,
    });
    (state, public_schema, tmp)
}

// ---------------------------------------------------------------------------
// Raw protocol client
// ---------------------------------------------------------------------------

/// Starts an accept loop serving connections from the test server state and
/// returns the listener address.
async fn spawn_server(state: Arc<ServerState>) -> std::net::SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("local addr");
    tokio::spawn(async move {
        while let Ok((stream, peer)) = listener.accept().await {
            let st = Arc::clone(&state);
            let handle = tokio::spawn(async move {
                let mut conn =
                    zyron_wire::connection::Connection::new(stream, st, Some(peer.to_string()));
                if let Err(e) = conn.run().await {
                    eprintln!("server connection error: {e}");
                }
            });
            tokio::spawn(async move {
                if let Err(e) = handle.await {
                    eprintln!("server connection task panicked: {e}");
                }
            });
        }
    });
    addr
}

fn build_startup_bytes(user: &str, database: &str) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&196608i32.to_be_bytes());
    payload.extend_from_slice(b"user\0");
    payload.extend_from_slice(user.as_bytes());
    payload.push(0);
    payload.extend_from_slice(b"database\0");
    payload.extend_from_slice(database.as_bytes());
    payload.push(0);
    payload.push(0);
    let len = (payload.len() + 4) as i32;
    let mut msg = Vec::new();
    msg.extend_from_slice(&len.to_be_bytes());
    msg.extend_from_slice(&payload);
    msg
}

fn build_query_bytes(sql: &str) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(sql.as_bytes());
    payload.push(0);
    let len = (payload.len() + 4) as i32;
    let mut msg = Vec::new();
    msg.push(b'Q');
    msg.extend_from_slice(&len.to_be_bytes());
    msg.extend_from_slice(&payload);
    msg
}

async fn read_backend_message(stream: &mut tokio::net::TcpStream) -> (u8, Vec<u8>) {
    use tokio::io::AsyncReadExt;
    let mut type_buf = [0u8; 1];
    stream.read_exact(&mut type_buf).await.expect("read type");
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).await.expect("read len");
    let len = i32::from_be_bytes(len_buf) as usize;
    assert!(len >= 4, "invalid message length");
    let mut payload = vec![0u8; len - 4];
    if !payload.is_empty() {
        stream.read_exact(&mut payload).await.expect("read payload");
    }
    (type_buf[0], payload)
}

/// Text values of one DataRow payload; None is a NULL field.
fn parse_data_row(payload: &[u8]) -> Vec<Option<String>> {
    let mut off = 0usize;
    let count = i16::from_be_bytes([payload[off], payload[off + 1]]) as usize;
    off += 2;
    let mut fields = Vec::with_capacity(count);
    for _ in 0..count {
        let len = i32::from_be_bytes([
            payload[off],
            payload[off + 1],
            payload[off + 2],
            payload[off + 3],
        ]);
        off += 4;
        if len < 0 {
            fields.push(None);
        } else {
            let end = off + len as usize;
            fields.push(Some(
                String::from_utf8_lossy(&payload[off..end]).into_owned(),
            ));
            off = end;
        }
    }
    fields
}

/// Opens a client connection and completes the startup handshake.
async fn connect(addr: std::net::SocketAddr) -> tokio::net::TcpStream {
    let mut stream = tokio::net::TcpStream::connect(addr).await.expect("connect");
    {
        use tokio::io::AsyncWriteExt;
        stream
            .write_all(&build_startup_bytes("test_user", "testdb"))
            .await
            .expect("startup");
        stream.flush().await.expect("flush");
    }
    loop {
        let (t, _) = read_backend_message(&mut stream).await;
        if t == b'Z' {
            break;
        }
    }
    // The default search path targets the zyron_sys schemas; the tests work
    // in testdb's zyron_test schema.
    sql_ok(&mut stream, "SET search_path = zyron_test").await;
    stream
}

/// Sends one simple query and collects the data rows and any error until
/// ReadyForQuery.
async fn sql(
    stream: &mut tokio::net::TcpStream,
    query: &str,
) -> (Vec<Vec<Option<String>>>, Option<String>) {
    use tokio::io::AsyncWriteExt;
    stream
        .write_all(&build_query_bytes(query))
        .await
        .expect("write query");
    stream.flush().await.expect("flush");
    let mut rows = Vec::new();
    let mut err = None;
    loop {
        let (t, payload) = read_backend_message(stream).await;
        match t {
            b'D' => rows.push(parse_data_row(&payload)),
            b'E' => err = Some(String::from_utf8_lossy(&payload).into_owned()),
            b'Z' => break,
            _ => {}
        }
    }
    (rows, err)
}

/// Runs a query that must succeed and returns its rows.
async fn sql_ok(stream: &mut tokio::net::TcpStream, query: &str) -> Vec<Vec<Option<String>>> {
    let (rows, err) = sql(stream, query).await;
    assert!(err.is_none(), "statement failed: {query}\n{err:?}");
    rows
}

/// Collects the first column of each row as i64.
fn first_col_i64(rows: &[Vec<Option<String>>]) -> Vec<i64> {
    rows.iter()
        .map(|r| {
            r[0].as_deref()
                .expect("non-null value")
                .parse::<i64>()
                .expect("integer value")
        })
        .collect()
}

async fn seed(stream: &mut tokio::net::TcpStream) {
    sql_ok(stream, "CREATE TABLE ch (id INT)").await;
    sql_ok(stream, "INSERT INTO ch (id) VALUES (1), (2), (3)").await;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn with_hold_cursor_survives_commit() {
    let (state, _schema, _tmp) = create_test_server().await;
    let addr = spawn_server(state).await;
    let mut c = connect(addr).await;
    seed(&mut c).await;

    sql_ok(&mut c, "BEGIN").await;
    sql_ok(
        &mut c,
        "DECLARE hc CURSOR WITH HOLD FOR SELECT id FROM ch ORDER BY id",
    )
    .await;
    sql_ok(&mut c, "COMMIT").await;

    // The held cursor is fetchable inside a new transaction.
    sql_ok(&mut c, "BEGIN").await;
    let rows = sql_ok(&mut c, "FETCH ALL FROM hc").await;
    assert_eq!(first_col_i64(&rows), vec![1, 2, 3]);
    sql_ok(&mut c, "COMMIT").await;
}

#[tokio::test]
async fn without_hold_cursor_closes_at_commit() {
    let (state, _schema, _tmp) = create_test_server().await;
    let addr = spawn_server(state).await;
    let mut c = connect(addr).await;
    seed(&mut c).await;

    sql_ok(&mut c, "BEGIN").await;
    sql_ok(&mut c, "DECLARE nc CURSOR FOR SELECT id FROM ch").await;
    // Fetchable inside its transaction.
    let rows = sql_ok(&mut c, "FETCH FORWARD 1 FROM nc").await;
    assert_eq!(rows.len(), 1);
    sql_ok(&mut c, "COMMIT").await;

    let (_, err) = sql(&mut c, "FETCH ALL FROM nc").await;
    let err = err.expect("a WITHOUT HOLD cursor closes at COMMIT");
    assert!(err.contains("does not exist"), "{err}");
}

#[tokio::test]
async fn held_cursor_keeps_its_snapshot() {
    let (state, _schema, _tmp) = create_test_server().await;
    let addr = spawn_server(state).await;
    let mut c = connect(addr).await;
    seed(&mut c).await;

    // Never fetched inside the transaction, so COMMIT materializes it.
    sql_ok(&mut c, "BEGIN").await;
    sql_ok(
        &mut c,
        "DECLARE hc CURSOR WITH HOLD FOR SELECT id FROM ch ORDER BY id",
    )
    .await;
    sql_ok(&mut c, "COMMIT").await;

    // A row committed after the COMMIT must not appear in the held set.
    sql_ok(&mut c, "INSERT INTO ch (id) VALUES (99)").await;
    let rows = sql_ok(&mut c, "FETCH ALL FROM hc").await;
    assert_eq!(
        first_col_i64(&rows),
        vec![1, 2, 3],
        "the held cursor exposes its COMMIT-time snapshot"
    );
}

#[tokio::test]
async fn held_cursor_fetch_position_survives_commit() {
    let (state, _schema, _tmp) = create_test_server().await;
    let addr = spawn_server(state).await;
    let mut c = connect(addr).await;
    seed(&mut c).await;

    sql_ok(&mut c, "BEGIN").await;
    sql_ok(
        &mut c,
        "DECLARE hc CURSOR WITH HOLD FOR SELECT id FROM ch ORDER BY id",
    )
    .await;
    let rows = sql_ok(&mut c, "FETCH FORWARD 1 FROM hc").await;
    assert_eq!(first_col_i64(&rows), vec![1]);
    sql_ok(&mut c, "COMMIT").await;

    // The next fetch continues where the transaction left off.
    let rows = sql_ok(&mut c, "FETCH FORWARD 1 FROM hc").await;
    assert_eq!(first_col_i64(&rows), vec![2]);
}

#[tokio::test]
async fn rollback_closes_a_cursor_the_transaction_declared() {
    let (state, _schema, _tmp) = create_test_server().await;
    let addr = spawn_server(state).await;
    let mut c = connect(addr).await;
    seed(&mut c).await;

    sql_ok(&mut c, "BEGIN").await;
    sql_ok(&mut c, "DECLARE hc CURSOR WITH HOLD FOR SELECT id FROM ch").await;
    sql_ok(&mut c, "ROLLBACK").await;

    let (_, err) = sql(&mut c, "FETCH ALL FROM hc").await;
    let err = err.expect("a held cursor dies with its aborted transaction");
    assert!(err.contains("does not exist"), "{err}");
}

#[tokio::test]
async fn held_cursor_already_fetched_survives_commit_with_rows() {
    let (state, _schema, _tmp) = create_test_server().await;
    let addr = spawn_server(state).await;
    let mut c = connect(addr).await;
    seed(&mut c).await;

    sql_ok(&mut c, "BEGIN").await;
    sql_ok(
        &mut c,
        "DECLARE hc CURSOR WITH HOLD FOR SELECT id FROM ch ORDER BY id",
    )
    .await;
    // Materializes during the transaction via the first fetch.
    let rows = sql_ok(&mut c, "FETCH ALL FROM hc").await;
    assert_eq!(first_col_i64(&rows), vec![1, 2, 3]);
    sql_ok(&mut c, "COMMIT").await;

    // Still open afterwards; the position is at the end, so a fetch returns
    // no further rows rather than an error.
    let rows = sql_ok(&mut c, "FETCH ALL FROM hc").await;
    assert!(rows.is_empty());
}

#[tokio::test]
async fn fetch_directions_follow_postgres_semantics() {
    let (state, _schema, _tmp) = create_test_server().await;
    let addr = spawn_server(state).await;
    let mut c = connect(addr).await;
    seed(&mut c).await;
    sql_ok(&mut c, "INSERT INTO ch (id) VALUES (4), (5)").await;

    sql_ok(&mut c, "BEGIN").await;
    sql_ok(
        &mut c,
        "DECLARE dc CURSOR FOR SELECT id FROM ch ORDER BY id",
    )
    .await;

    // Forward from before the first row lands on the last row returned.
    let rows = sql_ok(&mut c, "FETCH FORWARD 2 FROM dc").await;
    assert_eq!(first_col_i64(&rows), vec![1, 2]);

    // FIRST jumps to row one no matter where the cursor stands.
    let rows = sql_ok(&mut c, "FETCH FIRST FROM dc").await;
    assert_eq!(first_col_i64(&rows), vec![1]);

    // LAST jumps to the final row.
    let rows = sql_ok(&mut c, "FETCH LAST FROM dc").await;
    assert_eq!(first_col_i64(&rows), vec![5]);

    // PRIOR steps back one row.
    let rows = sql_ok(&mut c, "FETCH PRIOR FROM dc").await;
    assert_eq!(first_col_i64(&rows), vec![4]);

    // ABSOLUTE addresses a row by position, not a count of rows.
    let rows = sql_ok(&mut c, "FETCH ABSOLUTE 3 FROM dc").await;
    assert_eq!(first_col_i64(&rows), vec![3]);

    // A negative RELATIVE moves backward and returns that one row.
    let rows = sql_ok(&mut c, "FETCH RELATIVE -1 FROM dc").await;
    assert_eq!(first_col_i64(&rows), vec![2]);

    // BACKWARD past the beginning returns what exists, newest first, and
    // parks the cursor before the first row.
    let rows = sql_ok(&mut c, "FETCH BACKWARD 2 FROM dc").await;
    assert_eq!(first_col_i64(&rows), vec![1]);

    // ALL from before the first row is the whole set.
    let rows = sql_ok(&mut c, "FETCH ALL FROM dc").await;
    assert_eq!(first_col_i64(&rows), vec![1, 2, 3, 4, 5]);

    // NEXT past the end returns nothing and stands past the last row, so
    // PRIOR comes back with the last row.
    let rows = sql_ok(&mut c, "FETCH NEXT FROM dc").await;
    assert!(rows.is_empty());
    let rows = sql_ok(&mut c, "FETCH PRIOR FROM dc").await;
    assert_eq!(first_col_i64(&rows), vec![5]);

    // BACKWARD ALL walks everything before the current row in reverse.
    let rows = sql_ok(&mut c, "FETCH BACKWARD ALL FROM dc").await;
    assert_eq!(first_col_i64(&rows), vec![4, 3, 2, 1]);

    // ABSOLUTE -1 addresses the last row from the end.
    let rows = sql_ok(&mut c, "FETCH ABSOLUTE -1 FROM dc").await;
    assert_eq!(first_col_i64(&rows), vec![5]);

    sql_ok(&mut c, "COMMIT").await;
}

#[tokio::test]
async fn without_hold_cursor_requires_a_transaction_block() {
    let (state, _schema, _tmp) = create_test_server().await;
    let addr = spawn_server(state).await;
    let mut c = connect(addr).await;
    seed(&mut c).await;

    // In autocommit there is no transaction for the cursor to live in, so
    // the declaration itself is refused instead of leaving a cursor
    // floating across statements.
    let (_, err) = sql(&mut c, "DECLARE ac CURSOR FOR SELECT id FROM ch").await;
    let err = err.expect("WITHOUT HOLD outside a transaction block is refused");
    assert!(err.contains("transaction block"), "{err}");

    // The refused declaration left nothing behind.
    let (_, err) = sql(&mut c, "FETCH ALL FROM ac").await;
    assert!(err.expect("no cursor exists").contains("does not exist"));

    // WITH HOLD is the form that outlives a transaction, valid anywhere.
    sql_ok(
        &mut c,
        "DECLARE ac CURSOR WITH HOLD FOR SELECT id FROM ch ORDER BY id",
    )
    .await;
    let rows = sql_ok(&mut c, "FETCH ALL FROM ac").await;
    assert_eq!(first_col_i64(&rows), vec![1, 2, 3]);

    // Inside a block the plain form works as before.
    sql_ok(&mut c, "BEGIN").await;
    sql_ok(&mut c, "DECLARE bc CURSOR FOR SELECT id FROM ch").await;
    let rows = sql_ok(&mut c, "FETCH FORWARD 1 FROM bc").await;
    assert_eq!(rows.len(), 1);
    sql_ok(&mut c, "COMMIT").await;
}

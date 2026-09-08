//! Query cancellation over the wire.
//!
//! The server hands every connection a BackendKeyData secret at startup; a
//! CancelRequest on a fresh connection proving that secret must flip the
//! running statement's cancel flag, abort it with a clear error, and leave
//! the connection usable. Verifies the whole path: registry, startup arm,
//! cooperative operator checks.
//!
//! Run: cargo test -p zyron-wire --test cancel_test -- --nocapture

use std::sync::Arc;

use zyron_buffer::BufferPool;
use zyron_catalog::{Catalog, CatalogCache, HeapCatalogStorage, SchemaId};
use zyron_executor::context::ExecutionContext;
use zyron_storage::DiskManager;
use zyron_storage::txn::IsolationLevel;
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
// Cancel-specific client pieces
// ---------------------------------------------------------------------------

/// Connects, completes startup, and captures the BackendKeyData the server
/// issues, which a cancel connection must echo to prove itself.
async fn connect_with_key(addr: std::net::SocketAddr) -> (tokio::net::TcpStream, i32, i32) {
    let mut stream = tokio::net::TcpStream::connect(addr).await.expect("connect");
    {
        use tokio::io::AsyncWriteExt;
        stream
            .write_all(&build_startup_bytes("test_user", "testdb"))
            .await
            .expect("startup");
        stream.flush().await.expect("flush");
    }
    let mut key = None;
    loop {
        let (t, payload) = read_backend_message(&mut stream).await;
        if t == b'K' {
            let pid = i32::from_be_bytes([payload[0], payload[1], payload[2], payload[3]]);
            let secret = i32::from_be_bytes([payload[4], payload[5], payload[6], payload[7]]);
            key = Some((pid, secret));
        }
        if t == b'Z' {
            break;
        }
    }
    let (pid, secret) = key.expect("server sends BackendKeyData at startup");
    sql_ok(&mut stream, "SET search_path = zyron_test").await;
    (stream, pid, secret)
}

/// The 16-byte CancelRequest packet: length, magic code, pid, secret.
fn build_cancel_bytes(process_id: i32, secret_key: i32) -> Vec<u8> {
    let mut msg = Vec::with_capacity(16);
    msg.extend_from_slice(&16i32.to_be_bytes());
    msg.extend_from_slice(&80877102i32.to_be_bytes());
    msg.extend_from_slice(&process_id.to_be_bytes());
    msg.extend_from_slice(&secret_key.to_be_bytes());
    msg
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancel_request_aborts_the_running_statement() {
    use tokio::io::AsyncWriteExt;

    let (state, _schema, _tmp) = create_test_server().await;
    let addr = spawn_server(Arc::clone(&state)).await;
    let (mut c, pid, secret) = connect_with_key(addr).await;

    sql_ok(&mut c, "CREATE TABLE big (id INT)").await;
    let values: Vec<String> = (0..800).map(|i| format!("({i})")).collect();
    sql_ok(
        &mut c,
        &format!("INSERT INTO big (id) VALUES {}", values.join(", ")),
    )
    .await;

    // A bare count over a cross join answers from row-count metadata
    // instantly, so the predicate references columns to force the real
    // half-billion-pair join, a debug-build eternity without a cancel.
    c.write_all(&build_query_bytes(
        "SELECT count(*) FROM big a, big b, big c WHERE a.id + b.id + c.id >= 0",
    ))
    .await
    .expect("send query");
    c.flush().await.expect("flush");

    // Decompose the path before sending the packet: the statement must be
    // registered under this connection's key with a live context. Planning
    // takes what it takes, so poll rather than guess a delay.
    let mut live = None;
    for _ in 0..100 {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let found = state
            .cancel_registry
            .read_sync(&pid, |_, (sec, w)| (*sec, w.clone()));
        if let Some((reg_secret, weak_ctx)) = found {
            assert_eq!(
                reg_secret, secret,
                "registry holds the BackendKeyData secret"
            );
            if let Some(ctx) = weak_ctx.upgrade() {
                live = Some(ctx);
                break;
            }
        }
    }
    let live_ctx = live.expect("the running statement never registered a live context");

    // The cancel arrives on its own connection, keyed by the first
    // connection's BackendKeyData. The server closes it without a reply.
    let mut canceler = tokio::net::TcpStream::connect(addr).await.expect("connect");
    canceler
        .write_all(&build_cancel_bytes(pid, secret))
        .await
        .expect("send cancel");
    canceler.flush().await.expect("flush");
    drop(canceler);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    assert!(
        live_ctx.is_cancelled(),
        "the CancelRequest flipped the running statement's cancel flag"
    );
    drop(live_ctx);

    // The running statement must come back as an error, not hang and not
    // return a result.
    let outcome = tokio::time::timeout(std::time::Duration::from_secs(60), async {
        let mut err = None;
        loop {
            let (t, payload) = read_backend_message(&mut c).await;
            match t {
                b'E' => err = Some(String::from_utf8_lossy(&payload).into_owned()),
                b'Z' => break,
                _ => {}
            }
        }
        err
    })
    .await
    .expect("the cancel must interrupt the statement, not let it run for hours");
    let err = outcome.expect("a cancelled statement reports an error");
    assert!(err.contains("cancelled"), "{err}");

    // The connection survives its cancelled statement.
    let rows = sql_ok(&mut c, "SELECT count(*) FROM big").await;
    assert_eq!(first_col_i64(&rows), vec![800]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancel_with_wrong_secret_is_ignored() {
    use tokio::io::AsyncWriteExt;

    let (state, _schema, _tmp) = create_test_server().await;
    let addr = spawn_server(state).await;
    let (mut c, pid, secret) = connect_with_key(addr).await;
    sql_ok(&mut c, "CREATE TABLE t (id INT)").await;
    sql_ok(&mut c, "INSERT INTO t (id) VALUES (1)").await;

    // A wrong secret proves nothing: the cancel connection is dropped
    // silently and the session it aimed at is untouched.
    let mut canceler = tokio::net::TcpStream::connect(addr).await.expect("connect");
    canceler
        .write_all(&build_cancel_bytes(pid, secret.wrapping_add(1)))
        .await
        .expect("send cancel");
    canceler.flush().await.expect("flush");
    drop(canceler);

    let rows = sql_ok(&mut c, "SELECT id FROM t").await;
    assert_eq!(first_col_i64(&rows), vec![1]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn executor_polls_cancel_inside_a_cross_join() {
    let (state, _schema, _tmp) = create_test_server().await;
    let addr = spawn_server(Arc::clone(&state)).await;
    let (mut c, _pid, _secret) = connect_with_key(addr).await;
    sql_ok(&mut c, "CREATE TABLE big (id INT)").await;
    let values: Vec<String> = (0..800).map(|i| format!("({i})")).collect();
    sql_ok(
        &mut c,
        &format!("INSERT INTO big (id) VALUES {}", values.join(", ")),
    )
    .await;

    // The same runaway join, executed in process so the cancel flag is
    // flipped directly: this isolates the operator polling from the wire
    // path that delivers a CancelRequest.
    let stmt = zyron_parser::parse(
        "SELECT count(*) FROM big a, big b, big c WHERE a.id + b.id + c.id >= 0",
    )
    .expect("parse")
    .into_iter()
    .next()
    .expect("one statement");
    let db_id = state.catalog.get_database("testdb").expect("testdb").id;
    let plan = zyron_planner::plan(&state.catalog, db_id, vec!["zyron_test".into()], stmt, None)
        .await
        .expect("plan");
    let mut txn = state
        .txn_manager
        .begin(IsolationLevel::ReadCommitted)
        .expect("begin");
    let mut ctx = ExecutionContext::new(
        state.catalog.clone(),
        state.wal.clone(),
        state.buffer_pool.clone(),
        state.disk_manager.clone(),
        txn.txn_id,
        txn.snapshot.clone(),
    );
    ctx.heap_files = Some(Arc::clone(&state.heap_files));
    ctx.btree_indexes = Some(Arc::clone(&state.btree_indexes));
    let ctx = Arc::new(ctx);
    let exec_ctx = Arc::clone(&ctx);
    let handle = tokio::spawn(async move { zyron_executor::execute(plan, &exec_ctx).await });

    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    ctx.cancel();
    let joined = tokio::time::timeout(std::time::Duration::from_secs(30), handle)
        .await
        .expect("the executor must observe the cancel inside the join")
        .expect("task join");
    let err = joined.expect_err("a cancelled statement errors");
    assert!(err.to_string().contains("cancelled"), "{err}");
    let _ = state.txn_manager.abort(&mut txn);
}

#[tokio::test]
async fn scc_upsert_replaces_the_stored_weak() {
    let reg: Arc<scc::HashMap<i32, (i32, std::sync::Weak<String>)>> = Default::default();
    let a1 = Arc::new("one".to_string());
    reg.upsert_sync(1, (7, Arc::downgrade(&a1)));
    drop(a1);
    let a2 = Arc::new("two".to_string());
    reg.upsert_sync(1, (7, Arc::downgrade(&a2)));
    let live = reg.read_sync(&1, |_, (_, w)| w.upgrade().is_some());
    assert_eq!(live, Some(true), "upsert_sync must replace the stored weak");
    drop(a2);
}

// ---------------------------------------------------------------------------
// CANCEL BACKEND, the SQL-level kill surface
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancel_backend_statement_aborts_another_connections_statement() {
    use tokio::io::AsyncWriteExt;

    let (state, _schema, _tmp) = create_test_server().await;
    let addr = spawn_server(Arc::clone(&state)).await;
    let (mut victim, pid, _secret) = connect_with_key(addr).await;

    sql_ok(&mut victim, "CREATE TABLE big (id INT)").await;
    let values: Vec<String> = (0..800).map(|i| format!("({i})")).collect();
    sql_ok(
        &mut victim,
        &format!("INSERT INTO big (id) VALUES {}", values.join(", ")),
    )
    .await;

    // The same runaway join the CancelRequest test uses: the predicate
    // forces the real half-billion-pair product.
    victim
        .write_all(&build_query_bytes(
            "SELECT count(*) FROM big a, big b, big c WHERE a.id + b.id + c.id >= 0",
        ))
        .await
        .expect("send query");
    victim.flush().await.expect("flush");

    // Wait until the statement registers a live context under the pid.
    let mut registered = false;
    for _ in 0..100 {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let live = state
            .cancel_registry
            .read_sync(&pid, |_, (_, w)| w.upgrade().is_some());
        if live == Some(true) {
            registered = true;
            break;
        }
    }
    assert!(
        registered,
        "the running statement never registered a live context"
    );

    // The kill arrives as ordinary SQL on a second authenticated session,
    // no BackendKeyData secret involved.
    let (mut killer, _kpid, _ksecret) = connect_with_key(addr).await;
    sql_ok(&mut killer, &format!("CANCEL BACKEND {pid}")).await;

    // The victim's statement comes back as a cancellation error, and the
    // connection survives it.
    let outcome = tokio::time::timeout(std::time::Duration::from_secs(60), async {
        let mut err = None;
        loop {
            let (t, payload) = read_backend_message(&mut victim).await;
            match t {
                b'E' => err = Some(String::from_utf8_lossy(&payload).into_owned()),
                b'Z' => break,
                _ => {}
            }
        }
        err
    })
    .await
    .expect("CANCEL BACKEND must interrupt the statement");
    let err = outcome.expect("a cancelled statement reports an error");
    assert!(err.contains("cancelled"), "{err}");

    let rows = sql_ok(&mut victim, "SELECT count(*) FROM big").await;
    assert_eq!(first_col_i64(&rows), vec![800]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancel_backend_of_an_idle_connection_is_a_no_op() {
    let (state, _schema, _tmp) = create_test_server().await;
    let addr = spawn_server(Arc::clone(&state)).await;
    let (mut idle, pid, _secret) = connect_with_key(addr).await;
    sql_ok(&mut idle, "CREATE TABLE ct (id INT)").await;
    sql_ok(&mut idle, "INSERT INTO ct (id) VALUES (1)").await;

    // The idle connection's registry entry holds a dead weak from its last
    // finished statement; canceling it succeeds and touches nothing.
    let (mut killer, _kpid, _ksecret) = connect_with_key(addr).await;
    sql_ok(&mut killer, &format!("CANCEL BACKEND {pid}")).await;

    let rows = sql_ok(&mut idle, "SELECT id FROM ct").await;
    assert_eq!(first_col_i64(&rows), vec![1]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancel_backend_of_an_unknown_pid_errors() {
    let (state, _schema, _tmp) = create_test_server().await;
    let addr = spawn_server(Arc::clone(&state)).await;
    let (mut c, _pid, _secret) = connect_with_key(addr).await;

    let (_rows, err) = sql(&mut c, "CANCEL BACKEND 999999999").await;
    let err = err.expect("an unknown backend pid is an error");
    assert!(err.contains("not found"), "{err}");
}

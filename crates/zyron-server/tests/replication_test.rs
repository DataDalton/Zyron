//! Ordinary writes reaching the rest of a consensus group.
//!
//! The suite next to this one, `raft_test`, is about consensus: elections,
//! partitions, snapshots, the log. This one is about what consensus is
//! carrying. Every test runs a real engine per node, so what a follower ends
//! up holding is what its own heap, indexes and catalog say it holds, read
//! back through the same SQL a client would use.
//!
//! Run: `cargo test -p zyron-server --test replication_test --release --
//! --nocapture --test-threads=1`

use std::sync::Arc;
use std::time::{Duration, Instant};

use zyron_bench_harness::{
    VALIDATION_RUNS, format_with_commas, record_metric, tprintln, validate_metric_with_unit,
};
use zyron_buffer::BufferPool;
use zyron_catalog::{Catalog, CatalogCache, HeapCatalogStorage, SYSTEM_DATABASE_ID, SchemaId};
use zyron_common::{Result, ZyronError};
use zyron_executor::batch::DataBatch;
use zyron_executor::context::ExecutionContext;
use zyron_server::config::{ClusterPeerSection, ClusterSection};
use zyron_server::raft::{ClusterHandle, start_cluster};
use zyron_server::replication::DdlRunner;
use zyron_storage::DiskManager;
use zyron_storage::txn::{IsolationLevel, TransactionManager};
use zyron_wal::WalWriter;
use zyron_wire::connection::{ReplicationRouter, ServerState};

/// One measurement at a time. Two groups driving load on the same machine
/// measure each other rather than themselves
static BENCHMARK_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

// ---------------------------------------------------------------------------
// One node, with a whole engine behind it
// ---------------------------------------------------------------------------

struct Node {
    name: String,
    catalog: Arc<Catalog>,
    wal: Arc<WalWriter>,
    buffer_pool: Arc<BufferPool>,
    disk: Arc<DiskManager>,
    txn_manager: Arc<TransactionManager>,
    cluster: ClusterHandle,
    /// Held so the applier's weak reference to it stays alive
    _server: Arc<ServerState>,
    schema: SchemaId,
    _tmp: tempfile::TempDir,
}

impl Node {
    /// Builds the engine the server ships, then joins it to the group the
    /// section describes
    async fn start(name: &str, peers: &[(String, String)], listen: &str) -> Node {
        let tmp = tempfile::TempDir::new().expect("temp dir");
        let (data_dir, wal_dir) = zyron_bench_harness::create_dirs(tmp.path()).expect("dirs");

        let wal = Arc::new(WalWriter::new(zyron_bench_harness::wal_config(&wal_dir)).expect("wal"));
        let disk = Arc::new(
            DiskManager::new(zyron_bench_harness::disk_config(&data_dir))
                .await
                .expect("disk"),
        );
        let buffer_pool = Arc::new(BufferPool::new(zyron_bench_harness::buffer_pool_config()));
        let storage = Arc::new(
            HeapCatalogStorage::new(Arc::clone(&disk), Arc::clone(&buffer_pool)).expect("storage"),
        );
        let cache = Arc::new(CatalogCache::new(256, 64));
        let catalog = Arc::new(
            Catalog::new(storage, cache, Arc::clone(&wal))
                .await
                .expect("catalog"),
        );
        zyron_catalog::SystemCatalog::init(&catalog)
            .await
            .expect("register the zyron_sys catalog");
        let schema = catalog
            .create_schema(SYSTEM_DATABASE_ID, "public", "test_user")
            .await
            .expect("create public schema");
        let txn_manager = Arc::new(TransactionManager::new(Arc::clone(&wal)));

        let section = ClusterSection {
            enabled: true,
            node_name: name.to_string(),
            listen: listen.to_string(),
            peers: peers
                .iter()
                .map(|(n, a)| ClusterPeerSection {
                    name: n.clone(),
                    address: a.clone(),
                })
                .collect(),
            ..ClusterSection::default()
        };
        let cluster = start_cluster(
            &section,
            &data_dir,
            Arc::clone(&catalog),
            Arc::clone(&wal),
            Arc::clone(&buffer_pool),
            Arc::clone(&disk),
            Arc::clone(&txn_manager),
        )
        .await
        .expect("join the group");

        // The applier carries out a schema change through the dispatcher,
        // which needs the whole server around it, exactly as it does in
        // production
        let server = build_server_state(
            Arc::clone(&catalog),
            Arc::clone(&wal),
            Arc::clone(&buffer_pool),
            Arc::clone(&disk),
            Arc::clone(&txn_manager),
            &data_dir,
            &cluster,
        );
        cluster
            .replication
            .machine
            .attach_ddl_runner(zyron_server::replication::DispatchedDdl::new(&server));

        Node {
            name: name.to_string(),
            catalog,
            wal,
            buffer_pool,
            disk,
            txn_manager,
            cluster,
            _server: server,
            schema,
            _tmp: tmp,
        }
    }

    fn is_leader(&self) -> bool {
        self.cluster.node.is_leader()
    }

    fn router(&self) -> &Arc<zyron_server::replication::ReplicationHandle> {
        &self.cluster.replication
    }

    /// Runs a schema change the way a connection would: agree it with the
    /// group, wait for this node's turn, run it, report back
    async fn ddl(&self, sql: &str) -> Result<()> {
        let context = zyron_executor::replication::StatementContext {
            user: "test_user".to_string(),
            database: "zyron".to_string(),
            search_path: vec!["public".to_string()],
        };
        let done = self.router().begin_statement(sql, &context).await?;
        let outcome = zyron_server::replication::DispatchedDdl::new(&self._server)
            .run(sql, &context)
            .await;
        let _ = done.send(match &outcome {
            Ok(()) => Ok(()),
            Err(e) => Err(ZyronError::Internal(e.to_string())),
        });
        outcome
    }

    /// Runs statements in a transaction of this node's, then commits through
    /// the group
    async fn write(&self, statements: &[&str]) -> Result<()> {
        let mut txn = self.txn_manager.begin(IsolationLevel::ReadCommitted)?;
        let changeset = self.router().changeset(txn.txn_id());
        let outcome = async {
            for sql in statements {
                self.execute(sql, txn.txn_id(), txn.snapshot.clone(), Some(&changeset))
                    .await?;
            }
            Ok::<(), ZyronError>(())
        }
        .await;
        if let Err(e) = outcome {
            let _ = self.txn_manager.abort(&mut txn);
            return Err(e);
        }
        ReplicationRouter::capture_lake(self.router().as_ref(), txn.txn_id(), &changeset)?;
        let _ = ReplicationRouter::commit(self.router().as_ref(), txn, changeset).await?;
        Ok(())
    }

    /// Reads rows back the way a client would, so what is checked is what the
    /// node's own storage and catalog say rather than what a test remembers
    async fn query(&self, sql: &str) -> Result<Vec<DataBatch>> {
        let mut txn = self.txn_manager.begin(IsolationLevel::ReadCommitted)?;
        let batches = self
            .execute(sql, txn.txn_id(), txn.snapshot.clone(), None)
            .await;
        let _ = self.txn_manager.commit_read_only(&mut txn);
        batches
    }

    async fn execute(
        &self,
        sql: &str,
        txn_id: u64,
        snapshot: zyron_storage::Snapshot,
        changeset: Option<&Arc<zyron_executor::replication::TxnChangeset>>,
    ) -> Result<Vec<DataBatch>> {
        let statements = zyron_parser::parse(sql)?;
        let mut last = Vec::new();
        for statement in statements {
            let plan = zyron_planner::plan(
                &self.catalog,
                zyron_catalog::DatabaseId(1),
                vec!["public".to_string()],
                statement,
                None,
            )
            .await?;
            // The server's own context, registries included. A bare one
            // builds a fresh `HeapFile` per statement, whose insertion shards
            // are empty, so every single row insert would take a page of its
            // own and the measurement would be of that rather than of the
            // engine
            let mut ctx = self._server.statement_context(txn_id, snapshot.clone());
            ctx.replication = changeset.cloned();
            last = zyron_executor::execute(plan, &Arc::new(ctx)).await?;
        }
        Ok(last)
    }

    /// Rows of one integer column, sorted, so two nodes compare by contents
    async fn ints(&self, sql: &str) -> Vec<i64> {
        let batches = self.query(sql).await.expect("query");
        let mut out = Vec::new();
        for batch in &batches {
            let Some(column) = batch.columns.first() else {
                continue;
            };
            for row in 0..batch.num_rows {
                match column.data.get_scalar(row) {
                    zyron_executor::column::ScalarValue::Int64(v) => out.push(v),
                    zyron_executor::column::ScalarValue::Int32(v) => out.push(i64::from(v)),
                    zyron_executor::column::ScalarValue::Null => {}
                    other => panic!("a test read a column it cannot compare: {other:?}"),
                }
            }
        }
        out.sort_unstable();
        out
    }

    async fn count(&self, table: &str) -> i64 {
        let rows = self.ints(&format!("SELECT COUNT(*) FROM {table}")).await;
        rows.first().copied().unwrap_or(0)
    }

    /// Serves the wire protocol off a listening socket the way the process
    /// does, so a test can connect to this node exactly as a driver would
    async fn serve_wire(&self) -> std::net::SocketAddr {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind wire listener");
        let addr = listener.local_addr().expect("wire address");
        let server = Arc::clone(&self._server);
        tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    return;
                };
                tokio::spawn(zyron_wire::handle_connection(stream, Arc::clone(&server)));
            }
        });
        addr
    }

    /// Runs a statement the way a connection dispatches the ones the
    /// dispatcher owns: MERGE, CALL and DO all go through it rather than the
    /// planner, in their own transaction
    async fn dispatch(&self, sql: &str) -> Result<()> {
        let statements = zyron_parser::parse(sql)?;
        for statement in statements {
            let mut session = Some(zyron_wire::session::Session::new(
                "test_user".to_string(),
                "zyron".to_string(),
                zyron_catalog::DatabaseId(1),
            ));
            if let Some(session) = session.as_mut() {
                session.search_path = vec!["public".to_string()];
            }
            let mut txn = None;
            let mut branch = None;
            let handled = zyron_wire::ddl_dispatch::try_handle_ddl_utility(
                &statement,
                &self._server,
                &mut session,
                &mut txn,
                &mut branch,
                sql,
            )
            .await;
            match handled {
                Some(Ok(_)) => {}
                Some(Err(e)) => return Err(ZyronError::Internal(e.to_string())),
                None => {
                    return Err(ZyronError::Internal(format!(
                        "the dispatcher did not handle: {sql}"
                    )));
                }
            }
            if let Some(mut txn) = txn {
                if txn.wrote_data() {
                    self.txn_manager.commit(&mut txn).await?;
                } else {
                    self.txn_manager.commit_read_only(&mut txn)?;
                }
            }
        }
        Ok(())
    }
}

/// The server a node runs, built the way the process builds it.
///
/// Every field the real one carries, so the dispatcher a replayed schema
/// change goes through is the same dispatcher a client's statement goes
/// through rather than a reduced copy of it
fn build_server_state(
    catalog: Arc<Catalog>,
    wal: Arc<WalWriter>,
    buffer_pool: Arc<BufferPool>,
    disk_manager: Arc<DiskManager>,
    txn_manager: Arc<TransactionManager>,
    data_dir: &std::path::Path,
    cluster: &ClusterHandle,
) -> Arc<ServerState> {
    Arc::new(ServerState {
        raft: Some(Arc::clone(&cluster.node)),
        replication: Some(
            Arc::clone(&cluster.replication) as Arc<dyn zyron_wire::connection::ReplicationRouter>
        ),
        node_capabilities: None,
        catalog,
        wal,
        buffer_pool,
        disk_manager,
        txn_manager,
        doc_registry: Arc::new(zyron_common::DocRegistry::new()),
        table_io_stats: Arc::new(zyron_common::TableIOStatsRegistry::new()),
        index_io_stats: Arc::new(zyron_common::IndexIOStatsRegistry::new()),
        columnar_maintenance: None,
        security_manager: None,
        key_store: Arc::new(zyron_auth::LocalKeyStore::new([0u8; 32])),
        config_lookup: None,
        config_all: None,
        data_dir: data_dir.to_path_buf(),
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
        dlq_registry: Arc::new(zyron_streaming::dlq::DlqRegistry::new()),
        feature_store: zyron_analytics::featureStore(),
        feature_lineage: zyron_analytics::featureLineageRegistry(),
        model_cache: zyron_analytics::modelCache(),
        default_isolation: zyron_storage::IsolationLevel::ReadCommitted,
        peers: Default::default(),
        foreign_reader: None,
        deployment_mode: zyron_common::DeploymentMode::Unified,
        node_identity: Default::default(),
        statement_timeout: None,
        max_result_rows: None,
        max_query_memory: None,
        spill_directory: None,
        balloon_params: None,
        default_auth_method: zyron_auth::auth_rules::AuthMethod::Trust,
        password_encryption: "balloon-sha-256".into(),
    })
}

// ---------------------------------------------------------------------------
// A client on the wire
// ---------------------------------------------------------------------------

/// A client speaking the protocol off a socket, the way a driver does.
///
/// The harness everywhere else drives the executor directly, which is exactly
/// why it never caught a connection-level path that skipped replication. This
/// client exists so the explicit transaction block and the extended protocol
/// are exercised through the same bytes a real driver sends
struct WireClient {
    stream: tokio::net::TcpStream,
}

impl WireClient {
    /// Connects and completes the startup handshake under trust auth
    async fn connect(addr: std::net::SocketAddr) -> WireClient {
        use tokio::io::AsyncWriteExt;
        let mut stream = tokio::net::TcpStream::connect(addr).await.expect("connect");
        let mut params = Vec::new();
        params.extend_from_slice(&196608i32.to_be_bytes());
        params.extend_from_slice(b"user test_user database zyron  ");
        let mut startup = Vec::new();
        startup.extend_from_slice(&((params.len() + 4) as i32).to_be_bytes());
        startup.extend_from_slice(&params);
        stream.write_all(&startup).await.expect("startup");
        stream.flush().await.expect("flush startup");
        let mut client = WireClient { stream };
        client.drain_to_ready().await;
        client
    }

    /// One protocol message off the stream: a type byte, a length that
    /// includes itself, and the payload
    async fn read_message(&mut self) -> (u8, Vec<u8>) {
        use tokio::io::AsyncReadExt;
        let mut head = [0u8; 5];
        self.stream
            .read_exact(&mut head)
            .await
            .expect("message head");
        let len = i32::from_be_bytes([head[1], head[2], head[3], head[4]]) as usize;
        let mut payload = vec![0u8; len - 4];
        self.stream
            .read_exact(&mut payload)
            .await
            .expect("message payload");
        (head[0], payload)
    }

    /// Reads until ReadyForQuery, collecting command tags and error texts
    async fn drain_to_ready(&mut self) -> (Vec<String>, Vec<String>) {
        let mut tags = Vec::new();
        let mut errors = Vec::new();
        loop {
            let (kind, payload) = self.read_message().await;
            match kind {
                b'C' => {
                    let end = payload
                        .iter()
                        .position(|&b| b == 0)
                        .unwrap_or(payload.len());
                    tags.push(String::from_utf8_lossy(&payload[..end]).into_owned());
                }
                b'E' => errors.extend(error_text(&payload)),
                b'Z' => return (tags, errors),
                _ => {}
            }
        }
    }

    /// Sends one simple Query message and reads its whole answer
    async fn query(&mut self, sql: &str) -> (Vec<String>, Vec<String>) {
        use tokio::io::AsyncWriteExt;
        let mut msg = vec![b'Q'];
        msg.extend_from_slice(&((sql.len() + 1 + 4) as i32).to_be_bytes());
        msg.extend_from_slice(sql.as_bytes());
        msg.push(0);
        self.stream.write_all(&msg).await.expect("query");
        self.stream.flush().await.expect("flush query");
        self.drain_to_ready().await
    }

    /// Runs one statement through the extended protocol, the shape a driver
    /// prepared statement takes: Parse, Bind, Execute, Sync in one batch,
    /// with the parameters declared BIGINT and sent as text
    async fn prepared(&mut self, sql: &str, params: &[i64]) -> (Vec<String>, Vec<String>) {
        use tokio::io::AsyncWriteExt;
        let mut out = Vec::new();

        let mut body = Vec::new();
        body.push(0);
        body.extend_from_slice(sql.as_bytes());
        body.push(0);
        body.extend_from_slice(&(params.len() as i16).to_be_bytes());
        for _ in params {
            body.extend_from_slice(&20i32.to_be_bytes());
        }
        push_message(&mut out, b'P', &body);

        let mut body = Vec::new();
        body.push(0);
        body.push(0);
        body.extend_from_slice(&0i16.to_be_bytes());
        body.extend_from_slice(&(params.len() as i16).to_be_bytes());
        for value in params {
            let text = value.to_string();
            body.extend_from_slice(&(text.len() as i32).to_be_bytes());
            body.extend_from_slice(text.as_bytes());
        }
        body.extend_from_slice(&0i16.to_be_bytes());
        push_message(&mut out, b'B', &body);

        let mut body = Vec::new();
        body.push(0);
        body.extend_from_slice(&0i32.to_be_bytes());
        push_message(&mut out, b'E', &body);
        push_message(&mut out, b'S', &[]);

        self.stream
            .write_all(&out)
            .await
            .expect("extended protocol batch");
        self.stream.flush().await.expect("flush extended");
        self.drain_to_ready().await
    }

    async fn terminate(mut self) {
        use tokio::io::AsyncWriteExt;
        let _ = self.stream.write_all(&[b'X', 0, 0, 0, 4]).await;
        let _ = self.stream.shutdown().await;
    }
}

fn push_message(out: &mut Vec<u8>, kind: u8, body: &[u8]) {
    out.push(kind);
    out.extend_from_slice(&((body.len() + 4) as i32).to_be_bytes());
    out.extend_from_slice(body);
}

/// The M field of an ErrorResponse payload, which is the human message among
/// the coded fields
fn error_text(payload: &[u8]) -> Option<String> {
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

// ---------------------------------------------------------------------------
// A group of them
// ---------------------------------------------------------------------------

struct Group {
    /// Behind an `Arc` so a load generator can hand the same node to many
    /// concurrent tasks, which is the only way to ask what the group commits
    /// per second rather than what one client does
    nodes: Vec<Arc<Node>>,
}

impl Group {
    async fn start(count: usize) -> Group {
        // An apply that fails stops a node, and the reason only exists in the
        // log, so a run that is diagnosing one needs it on
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::WARN)
            .with_test_writer()
            .try_init();
        // Bound to port zero first so the addresses are real and free, then
        // handed over: a fixed port would collide with whatever else the
        // machine is running
        let mut listeners = Vec::new();
        let mut peers = Vec::new();
        for i in 0..count {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .expect("bind");
            let address = listener.local_addr().expect("addr").to_string();
            peers.push((format!("node-{}", i + 1), address));
            listeners.push(listener);
        }
        drop(listeners);

        let mut nodes = Vec::new();
        for (i, (name, address)) in peers.iter().enumerate() {
            let _ = i;
            nodes.push(Arc::new(Node::start(name, &peers, address).await));
        }
        Group { nodes }
    }

    async fn leader(&self, within: Duration) -> usize {
        let deadline = Instant::now() + within;
        while Instant::now() < deadline {
            if let Some(at) = self.nodes.iter().position(|n| n.is_leader()) {
                // A leader that has not committed an entry of its own term
                // cannot serve reads yet, so wait for the no-op it appends
                if self.nodes[at].cluster.node.commit_index() > 0 {
                    return at;
                }
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        panic!("no leader inside {within:?}");
    }

    /// Waits for every node to have applied everything the leader has
    async fn settle(&self, leader: usize, within: Duration) {
        let target = self.nodes[leader].cluster.node.last_log_index();
        let deadline = Instant::now() + within;
        loop {
            let behind: Vec<&str> = self
                .nodes
                .iter()
                .filter(|n| n.cluster.node.last_applied() < target)
                .map(|n| n.name.as_str())
                .collect();
            if behind.is_empty() {
                return;
            }
            if Instant::now() >= deadline {
                let where_they_are: Vec<String> = self
                    .nodes
                    .iter()
                    .map(|n| {
                        format!(
                            "{} applied {} of {} committed, {} staged",
                            n.name,
                            n.cluster.node.last_applied(),
                            n.cluster.node.commit_index(),
                            n.cluster.replication.machine.staged()
                        )
                    })
                    .collect();
                panic!("{behind:?} never reached index {target}: {where_they_are:?}");
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    async fn shutdown(self) {
        for node in &self.nodes {
            node.cluster.shutdown().await;
        }
    }

    /// Waits for every node to have applied `target`, and reports how long it
    /// took. Panics rather than returning, because a group that never settles
    /// has nothing further worth measuring
    async fn catch_up(&self, target: u64, within: Duration) -> Duration {
        let at = Instant::now();
        loop {
            if self
                .nodes
                .iter()
                .all(|n| n.cluster.node.last_applied() >= target)
            {
                return at.elapsed();
            }
            if at.elapsed() >= within {
                let where_they_are: Vec<String> = self
                    .nodes
                    .iter()
                    .map(|n| format!("{} at {}", n.name, n.cluster.node.last_applied()))
                    .collect();
                panic!("the group never reached {target}: {where_they_are:?}");
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    }

    /// How far the furthest behind node is from what the leader has committed
    fn worst_lag(&self, leader: usize) -> u64 {
        let committed = self.nodes[leader].cluster.node.commit_index();
        self.nodes
            .iter()
            .map(|n| committed.saturating_sub(n.cluster.node.last_applied()))
            .max()
            .unwrap_or(0)
    }
}

/// Drives `total` single row transactions with `concurrency` of them
/// outstanding, and returns how long each one took.
///
/// One transaction per row on purpose. Batching would measure the batch and
/// the question here is what one write costs when it has to be agreed by a
/// group before it is answered
async fn drive_writes(
    node: &Arc<Node>,
    table: &str,
    from: i64,
    total: i64,
    concurrency: usize,
) -> Vec<Duration> {
    let next = Arc::new(std::sync::atomic::AtomicI64::new(from));
    let end = from + total;
    let mut tasks = Vec::with_capacity(concurrency);
    for _ in 0..concurrency {
        let node = Arc::clone(node);
        let next = Arc::clone(&next);
        let table = table.to_string();
        tasks.push(tokio::spawn(async move {
            let mut taken = Vec::new();
            loop {
                let id = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if id >= end {
                    return taken;
                }
                let sql = format!("INSERT INTO {table} (id, v) VALUES ({id}, {id})");
                let at = Instant::now();
                node.write(&[sql.as_str()]).await.expect("write");
                taken.push(at.elapsed());
            }
        }));
    }
    let mut latencies = Vec::with_capacity(total as usize);
    for task in tasks {
        latencies.extend(task.await.expect("load task"));
    }
    latencies
}

/// Mean lag over the second half of a run against the first half.
///
/// The shape that matters, not the size. A follower that keeps up holds a
/// queue of about the client's concurrency for as long as the load runs, so
/// the two halves match. One that cannot keep up accumulates, and the ratio
/// climbs with the length of the run
fn lag_ratio(samples: &[u64]) -> f64 {
    if samples.len() < 4 {
        return 1.0;
    }
    let half = samples.len() / 2;
    let mean = |slice: &[u64]| slice.iter().sum::<u64>() as f64 / slice.len() as f64;
    let first = mean(&samples[..half]);
    let second = mean(&samples[half..]);
    if first <= 0.0 {
        // Nothing ever queued in the first half, so there is no growth to
        // measure and the second half is judged on its own against the same
        // rule the caller applies
        return if second <= 1.0 { 1.0 } else { second };
    }
    second / first
}

/// The value at a percentile of a sorted list.
fn percentile(sorted: &[Duration], q: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let at = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[at]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// A schema change reaches every node, and the object ids it allocates match
/// without any of them being replicated
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_schema_change_reaches_every_node_with_the_same_ids() {
    zyron_bench_harness::init("replication");
    tprintln!("\n=== Schema Change ===");

    let group = Group::start(3).await;
    let leader = group.leader(Duration::from_secs(5)).await;
    group.nodes[leader]
        .ddl("CREATE TABLE t (id BIGINT PRIMARY KEY, v BIGINT)")
        .await
        .expect("create table");
    group.settle(leader, Duration::from_secs(10)).await;

    let mut ids = Vec::new();
    for node in &group.nodes {
        let entry = node
            .catalog
            .get_table(node.schema, "t")
            .unwrap_or_else(|e| panic!("{} has no table t: {e}", node.name));
        ids.push(entry.id.0);
        assert_eq!(
            entry.columns.len(),
            2,
            "{} sees a different shape",
            node.name
        );
    }
    assert!(
        ids.windows(2).all(|w| w[0] == w[1]),
        "the same table got different ids: {ids:?}"
    );
    tprintln!("  every node holds table t as id {}", ids[0]);
    group.shutdown().await;
}

/// Rows written on the leader are on every follower, read back through each
/// one's own storage
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rows_written_on_the_leader_are_on_every_follower() {
    zyron_bench_harness::init("replication");
    tprintln!("\n=== Insert, Update, Delete ===");

    let group = Group::start(3).await;
    let leader = group.leader(Duration::from_secs(5)).await;
    group.nodes[leader]
        .ddl("CREATE TABLE t (id BIGINT PRIMARY KEY, v BIGINT)")
        .await
        .expect("create table");
    group.settle(leader, Duration::from_secs(10)).await;

    group.nodes[leader]
        .write(&[
            "INSERT INTO t (id, v) VALUES (1, 10), (2, 20), (3, 30)",
            "INSERT INTO t (id, v) VALUES (4, 40)",
        ])
        .await
        .expect("insert");
    group.settle(leader, Duration::from_secs(10)).await;
    for node in &group.nodes {
        assert_eq!(
            node.ints("SELECT id FROM t").await,
            vec![1, 2, 3, 4],
            "{} does not hold the inserted rows",
            node.name
        );
        assert_eq!(node.ints("SELECT v FROM t").await, vec![10, 20, 30, 40]);
    }
    tprintln!("  four rows on all three nodes");

    group.nodes[leader]
        .write(&["UPDATE t SET v = 99 WHERE id = 2"])
        .await
        .expect("update");
    group.settle(leader, Duration::from_secs(10)).await;
    for node in &group.nodes {
        assert_eq!(
            node.ints("SELECT v FROM t").await,
            vec![10, 30, 40, 99],
            "{} did not take the update",
            node.name
        );
    }
    tprintln!("  the update reached all three");

    group.nodes[leader]
        .write(&["DELETE FROM t WHERE id = 3"])
        .await
        .expect("delete");
    group.settle(leader, Duration::from_secs(10)).await;
    for node in &group.nodes {
        assert_eq!(
            node.ints("SELECT id FROM t").await,
            vec![1, 2, 4],
            "{} did not take the delete",
            node.name
        );
    }
    tprintln!("  the delete reached all three");
    group.shutdown().await;
}

/// A table with no unique index still replicates, matching rows by their whole
/// image, and duplicates delete one at a time
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_table_with_no_key_replicates_by_whole_row() {
    zyron_bench_harness::init("replication");
    tprintln!("\n=== No Primary Key ===");

    let group = Group::start(3).await;
    let leader = group.leader(Duration::from_secs(5)).await;
    group.nodes[leader]
        .ddl("CREATE TABLE nokey (v BIGINT)")
        .await
        .expect("create table");
    group.settle(leader, Duration::from_secs(10)).await;

    // Three rows that are byte for byte the same, and one that is not
    group.nodes[leader]
        .write(&["INSERT INTO nokey (v) VALUES (7), (7), (7), (8)"])
        .await
        .expect("insert");
    group.settle(leader, Duration::from_secs(10)).await;
    for node in &group.nodes {
        assert_eq!(
            node.ints("SELECT v FROM nokey").await,
            vec![7, 7, 7, 8],
            "{} does not hold the duplicates",
            node.name
        );
    }

    // Deleting the identical rows must delete all three and no more, which is
    // what the count carried beside each image is for
    group.nodes[leader]
        .write(&["DELETE FROM nokey WHERE v = 7"])
        .await
        .expect("delete the duplicates");
    group.settle(leader, Duration::from_secs(10)).await;
    for node in &group.nodes {
        assert_eq!(
            node.ints("SELECT v FROM nokey").await,
            vec![8],
            "{} deleted a different number of duplicates",
            node.name
        );
    }

    // And an update over duplicates changes exactly as many as it did there
    group.nodes[leader]
        .write(&["INSERT INTO nokey (v) VALUES (9), (9)"])
        .await
        .expect("insert duplicates");
    group.settle(leader, Duration::from_secs(10)).await;
    group.nodes[leader]
        .write(&["UPDATE nokey SET v = 11 WHERE v = 9"])
        .await
        .expect("update duplicates");
    group.settle(leader, Duration::from_secs(10)).await;
    for node in &group.nodes {
        assert_eq!(
            node.ints("SELECT v FROM nokey").await,
            vec![8, 11, 11],
            "{} updated a different number of duplicates",
            node.name
        );
    }
    tprintln!("  duplicates delete and update in the right numbers everywhere");
    group.shutdown().await;
}

/// A transaction larger than one entry replicates while it runs and still
/// commits as one
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_large_transaction_streams_and_commits_once() {
    zyron_bench_harness::init("replication");
    tprintln!("\n=== Streaming a Large Transaction ===");

    let group = Group::start(3).await;
    let leader = group.leader(Duration::from_secs(5)).await;
    group.nodes[leader]
        .ddl("CREATE TABLE big (id BIGINT PRIMARY KEY, pad VARCHAR(512))")
        .await
        .expect("create table");
    group.settle(leader, Duration::from_secs(10)).await;

    // Enough rows that the transaction crosses the chunk size several times
    let pad = "x".repeat(400);
    let mut statements = Vec::new();
    for chunk in 0..40 {
        let values: Vec<String> = (0..100)
            .map(|i| format!("({}, '{pad}')", chunk * 100 + i))
            .collect();
        statements.push(format!(
            "INSERT INTO big (id, pad) VALUES {}",
            values.join(", ")
        ));
    }
    let refs: Vec<&str> = statements.iter().map(|s| s.as_str()).collect();
    let at = Instant::now();
    group.nodes[leader].write(&refs).await.expect("bulk insert");
    tprintln!("  4000 rows through the group in {:.3?}", at.elapsed());

    group.settle(leader, Duration::from_secs(60)).await;
    for node in &group.nodes {
        assert_eq!(
            node.count("big").await,
            4000,
            "{} is missing rows from the bulk load",
            node.name
        );
    }

    // The entry count says it did not arrive as one entry
    let entries = group.nodes[leader].cluster.node.last_log_index();
    tprintln!("  the group's log holds {entries} entries");
    assert!(
        entries > 3,
        "a transaction past the chunk size should have streamed"
    );

    // Nothing lingers once it commits, on any node. The leader in particular
    // must not have replayed its own streamed chunks into a shadow staged
    // transaction, which would sit open holding the version horizon and
    // duplicate rows until the next leadership change
    for node in &group.nodes {
        let machine = &node.cluster.replication.machine;
        assert_eq!(
            machine.staged(),
            0,
            "{} still holds a staged transaction after the commit",
            node.name
        );
        assert_eq!(
            machine.open_spanning(),
            0,
            "{} still counts the transaction as open, so it pins the log",
            node.name
        );
    }
    group.shutdown().await;
}

/// MERGE, CALL and DO run through the dispatcher in transactions of their
/// own rather than through the planner path, and the rows they write have to
/// reach the group the same as any other write.
///
/// Each of the three used to commit locally with nothing captured, so a
/// leader answered the client that the write succeeded while no follower
/// ever heard of it
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn dispatcher_writes_reach_every_node() {
    zyron_bench_harness::init("replication");
    tprintln!("\n=== MERGE, CALL and DO Reach Every Node ===");

    let group = Group::start(3).await;
    let leader = group.leader(Duration::from_secs(5)).await;
    group.nodes[leader]
        .ddl("CREATE TABLE d (k BIGINT PRIMARY KEY, v BIGINT)")
        .await
        .expect("create target");
    group.nodes[leader]
        .ddl("CREATE TABLE src (k BIGINT PRIMARY KEY, v BIGINT)")
        .await
        .expect("create source");
    group.nodes[leader]
        .ddl(
            "CREATE PROCEDURE add_d(k INT, v INT) AS 'INSERT INTO d (k, v) VALUES ($1, $2)' \
LANGUAGE SQL",
        )
        .await
        .expect("create procedure");
    group.settle(leader, Duration::from_secs(10)).await;

    group.nodes[leader]
        .write(&[
            "INSERT INTO d (k, v) VALUES (1, 10), (2, 20)",
            "INSERT INTO src (k, v) VALUES (2, 200), (3, 300)",
        ])
        .await
        .expect("seed rows");

    // A MERGE updates row 2 and inserts row 3
    group.nodes[leader]
        .dispatch(
            "MERGE INTO d USING src ON d.k = src.k \
WHEN MATCHED THEN UPDATE SET v = src.v \
WHEN NOT MATCHED THEN INSERT (k, v) VALUES (src.k, src.v)",
        )
        .await
        .expect("merge");
    // A procedure body inserts row 4
    group.nodes[leader]
        .dispatch("CALL add_d(4, 400)")
        .await
        .expect("call");
    // A DO block inserts row 5
    group.nodes[leader]
        .dispatch("DO 'INSERT INTO d (k, v) VALUES (5, 500)'")
        .await
        .expect("do block");

    group.settle(leader, Duration::from_secs(30)).await;
    for node in &group.nodes {
        assert_eq!(
            node.ints("SELECT v FROM d").await,
            vec![10, 200, 300, 400, 500],
            "{} does not hold what the dispatcher wrote",
            node.name
        );
    }
    tprintln!("  merge, procedure and do block rows are on all three nodes");

    // A follower refuses the same work and names the leader, exactly as it
    // does for a plain write
    let follower = (leader + 1) % group.nodes.len();
    let refused = group.nodes[follower].dispatch("CALL add_d(6, 600)").await;
    assert!(
        refused.is_err(),
        "a follower ran a procedure body without leading"
    );

    // The records those commits wrote are pinned in the write-ahead log until
    // a raft snapshot covers their entries, because a restart reads them to
    // avoid committing the same transactions twice
    assert!(
        group.nodes[leader]
            .cluster
            .replication
            .machine
            .wal_retention_pin(0)
            .is_some(),
        "agreed commit records are not pinning WAL retention"
    );
    group.settle(leader, Duration::from_secs(10)).await;
    for node in &group.nodes {
        assert_eq!(
            node.count("d").await,
            5,
            "{} holds rows from a refused follower write",
            node.name
        );
    }
    group.shutdown().await;
}

/// The paths a real driver exercises, over a real socket: an explicit BEGIN
/// block, a savepoint rolled back inside it, and prepared statements through
/// the extended protocol.
///
/// Every one of these has committed on one node alone at some point: BEGIN
/// created the transaction without its changeset, a rollback to a savepoint
/// shipped the discarded rows, and the portal path attached no changeset at
/// all. The harness everywhere else drives the executor directly, which is
/// why none of that was caught, so this test speaks the wire
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_wire_client_transaction_replicates() {
    zyron_bench_harness::init("replication");
    tprintln!(
        "
=== A Client on the Wire ==="
    );

    let group = Group::start(3).await;
    let leader = group.leader(Duration::from_secs(5)).await;
    group.nodes[leader]
        .ddl("CREATE TABLE w (k BIGINT PRIMARY KEY, v BIGINT)")
        .await
        .expect("create table");
    group.settle(leader, Duration::from_secs(10)).await;

    let addr = group.nodes[leader].serve_wire().await;
    let mut client = WireClient::connect(addr).await;

    // An explicit transaction with a savepoint rolled back inside it. Rows 1,
    // 2 and 4 survive, row 3 was unwritten and must reach nobody
    for sql in [
        "BEGIN",
        "INSERT INTO w (k, v) VALUES (1, 10)",
        "INSERT INTO w (k, v) VALUES (2, 20)",
        "SAVEPOINT s1",
        "INSERT INTO w (k, v) VALUES (3, 30)",
        "ROLLBACK TO SAVEPOINT s1",
        "INSERT INTO w (k, v) VALUES (4, 40)",
        "COMMIT",
    ] {
        let (_, errors) = client.query(sql).await;
        assert!(errors.is_empty(), "{sql} failed over the wire: {errors:?}");
    }

    // Two prepared inserts through Parse, Bind, Execute, Sync
    let (tags, errors) = client
        .prepared("INSERT INTO w (k, v) VALUES ($1, $2)", &[5, 50])
        .await;
    assert!(errors.is_empty(), "prepared insert failed: {errors:?}");
    assert!(
        tags.iter().any(|t| t.starts_with("INSERT")),
        "no insert tag came back: {tags:?}"
    );
    let (_, errors) = client
        .prepared("INSERT INTO w (k, v) VALUES ($1, $2)", &[6, 60])
        .await;
    assert!(
        errors.is_empty(),
        "second prepared insert failed: {errors:?}"
    );
    client.terminate().await;

    group.settle(leader, Duration::from_secs(30)).await;
    for node in &group.nodes {
        assert_eq!(
            node.ints("SELECT v FROM w").await,
            vec![10, 20, 40, 50, 60],
            "{} does not hold what the wire client wrote",
            node.name
        );
    }
    tprintln!("  the BEGIN block, the savepoint cut and the portal writes are on all three nodes");

    // The same client shape against a follower is refused rather than kept
    // to itself
    let follower = (leader + 1) % group.nodes.len();
    let follower_addr = group.nodes[follower].serve_wire().await;
    let mut client = WireClient::connect(follower_addr).await;
    let mut refused = false;
    for sql in ["BEGIN", "INSERT INTO w (k, v) VALUES (7, 70)", "COMMIT"] {
        let (_, errors) = client.query(sql).await;
        refused |= !errors.is_empty();
    }
    client.terminate().await;
    assert!(refused, "a follower accepted a write over the wire");
    group.settle(leader, Duration::from_secs(10)).await;
    for node in &group.nodes {
        assert_eq!(
            node.count("w").await,
            5,
            "{} holds a row from a refused follower write",
            node.name
        );
    }
    group.shutdown().await;
}

/// A follower refuses a write and names the leader, so a client redirects
/// rather than splitting the group
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_follower_refuses_a_write_and_names_the_leader() {
    zyron_bench_harness::init("replication");
    tprintln!("\n=== A Follower Does Not Write ===");

    let group = Group::start(3).await;
    let leader = group.leader(Duration::from_secs(5)).await;
    group.nodes[leader]
        .ddl("CREATE TABLE t (id BIGINT PRIMARY KEY)")
        .await
        .expect("create table");
    group.settle(leader, Duration::from_secs(10)).await;

    let follower = (leader + 1) % group.nodes.len();
    let refused = group.nodes[follower]
        .write(&["INSERT INTO t (id) VALUES (1)"])
        .await
        .expect_err("a follower must not decide a write for the group");
    match refused {
        ZyronError::NotLeader { leader: named } => {
            assert!(named.is_some(), "the refusal should name a leader to go to");
        }
        other => panic!("expected a redirect, got {other:?}"),
    }
    tprintln!("  the follower redirected instead of writing");
    group.shutdown().await;
}

/// Every node's changes come out in the group's order, so the state a node
/// passes through is a state every node passes through
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_sequence_of_writes_lands_in_the_same_order_everywhere() {
    zyron_bench_harness::init("replication");
    tprintln!("\n=== Commit Order ===");

    let group = Group::start(3).await;
    let leader = group.leader(Duration::from_secs(5)).await;
    group.nodes[leader]
        .ddl("CREATE TABLE seq (id BIGINT PRIMARY KEY, v BIGINT)")
        .await
        .expect("create table");
    group.settle(leader, Duration::from_secs(10)).await;

    for round in 0..50i64 {
        group.nodes[leader]
            .write(&[format!("INSERT INTO seq (id, v) VALUES ({round}, {round})").as_str()])
            .await
            .expect("insert");
    }
    group.settle(leader, Duration::from_secs(30)).await;

    let expected: Vec<i64> = (0..50).collect();
    for node in &group.nodes {
        assert_eq!(
            node.ints("SELECT id FROM seq").await,
            expected,
            "{} holds a different set",
            node.name
        );
    }
    tprintln!("  fifty writes, identical on all three nodes");
    group.shutdown().await;
}

// ---------------------------------------------------------------------------
// What it costs
// ---------------------------------------------------------------------------

/// Writes outstanding at once while throughput is measured. Enough for the
/// leader to batch proposals and for the appliers to be given a queue
const THROUGHPUT_CONCURRENCY: usize = 64;
/// Writes outstanding while latency is measured. A p99 taken at saturation is
/// the client's queue depth rather than the cost of a write
const LATENCY_CONCURRENCY: usize = 8;

const THROUGHPUT_WRITES: i64 = 20_000;
const LATENCY_WRITES: i64 = 2_000;

/// What a replicated write costs, and whether a follower keeps up with one.
///
/// The second half is the point. A follower's apply skips parsing, planning,
/// expression evaluation, defaults, constraints and triggers, so the question
/// is whether one thread doing the cheap half keeps pace with a leader doing
/// the expensive half on every core it has. If it does, applying entries in
/// parallel would buy nothing; if it does not, the lag grows without bound and
/// a follower is never a current replica
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn replicated_write_throughput_and_follower_lag() {
    zyron_bench_harness::init("replication");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Replicated Write Throughput and Follower Lag ===");

    let mut leader_rates = Vec::with_capacity(VALIDATION_RUNS);
    let mut follower_rates = Vec::with_capacity(VALIDATION_RUNS);
    let mut keep_up = Vec::with_capacity(VALIDATION_RUNS);
    let mut lags = Vec::with_capacity(VALIDATION_RUNS);
    let mut lag_growth = Vec::with_capacity(VALIDATION_RUNS);
    let mut p50s = Vec::with_capacity(VALIDATION_RUNS);
    let mut p99s = Vec::with_capacity(VALIDATION_RUNS);
    let mut catch_ups = Vec::with_capacity(VALIDATION_RUNS);

    for run in 0..VALIDATION_RUNS {
        tprintln!("--- Run {}/{} ---", run + 1, VALIDATION_RUNS);
        let group = Group::start(3).await;
        let leader = group.leader(Duration::from_secs(5)).await;
        group.nodes[leader]
            .ddl("CREATE TABLE bench (id BIGINT PRIMARY KEY, v BIGINT)")
            .await
            .expect("create table");
        group.settle(leader, Duration::from_secs(10)).await;

        // What one write costs with the group well short of saturation
        let mut latencies = drive_writes(
            &group.nodes[leader],
            "bench",
            0,
            LATENCY_WRITES,
            LATENCY_CONCURRENCY,
        )
        .await;
        latencies.sort();
        let p50 = percentile(&latencies, 0.50);
        let p99 = percentile(&latencies, 0.99);
        tprintln!(
            "  {} writes at {} outstanding: p50 {:.3?}, p99 {:.3?}",
            format_with_commas(LATENCY_WRITES as f64),
            LATENCY_CONCURRENCY,
            p50,
            p99
        );
        p50s.push(p50.as_secs_f64());
        p99s.push(p99.as_secs_f64());
        group
            .catch_up(
                group.nodes[leader].cluster.node.last_log_index(),
                Duration::from_secs(60),
            )
            .await;

        // Where each node had got to before the load, so what it applies
        // during the load can be counted rather than inferred
        let baseline: Vec<u64> = group
            .nodes
            .iter()
            .map(|n| n.cluster.node.last_applied())
            .collect();

        // The lag is sampled all the way through and split in half. A follower
        // that cannot keep up shows a lag that climbs with every write, so the
        // second half against the first is the question asked directly
        let sampler_group: Vec<Arc<Node>> = group.nodes.iter().map(Arc::clone).collect();
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let sampler_stop = Arc::clone(&stop);
        let sampler = tokio::spawn(async move {
            let mut samples: Vec<u64> = Vec::new();
            while !sampler_stop.load(std::sync::atomic::Ordering::Relaxed) {
                let committed = sampler_group
                    .iter()
                    .map(|n| n.cluster.node.commit_index())
                    .max()
                    .unwrap_or(0);
                let worst = sampler_group
                    .iter()
                    .map(|n| committed.saturating_sub(n.cluster.node.last_applied()))
                    .max()
                    .unwrap_or(0);
                samples.push(worst);
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
            samples
        });

        let at = Instant::now();
        drive_writes(
            &group.nodes[leader],
            "bench",
            LATENCY_WRITES,
            THROUGHPUT_WRITES,
            THROUGHPUT_CONCURRENCY,
        )
        .await;
        let elapsed = at.elapsed();
        let leader_rate = THROUGHPUT_WRITES as f64 / elapsed.as_secs_f64();

        // How far behind the appliers were the moment the load stopped, and
        // how long the group took to close it with nothing else running
        let target = group.nodes[leader].cluster.node.last_log_index();
        let behind = group.worst_lag(leader);
        // Read the moment the load stops, so what follows measures the applier
        // under load rather than under load plus the wakeup that delivers the
        // last handful of entries once there is nothing left to batch them with
        let at_end: Vec<u64> = group
            .nodes
            .iter()
            .map(|n| n.cluster.node.last_applied())
            .collect();
        let catch_up = group.catch_up(target, Duration::from_secs(120)).await;
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        let samples = sampler.await.expect("sampler");
        let worst_during = samples.iter().copied().max().unwrap_or(0);
        let growth = lag_ratio(&samples);

        // Entries the slowest node applied over the same wall clock the leader
        // was measured on. Timing the handful left at the end instead would
        // measure the wakeup that delivered them, not the work
        let slowest = at_end
            .iter()
            .zip(&baseline)
            .map(|(now, from)| now.saturating_sub(*from))
            .min()
            .unwrap_or(0);
        let follower_rate = slowest as f64 / elapsed.as_secs_f64();

        tprintln!(
            "  {} writes at {} outstanding in {:.3?}, {} txn/sec",
            format_with_commas(THROUGHPUT_WRITES as f64),
            THROUGHPUT_CONCURRENCY,
            elapsed,
            format_with_commas(leader_rate)
        );
        tprintln!(
            "  worst follower lag {} entries while running, {} left at the end, closed in {:.3?}",
            format_with_commas(worst_during as f64),
            format_with_commas(behind as f64),
            catch_up
        );
        tprintln!("  lag in the second half was {:.2}x the first half", growth);
        tprintln!(
            "  slowest node applied {} entries in {:.3?}, {} entries/sec against a leader \
committing {} txn/sec",
            format_with_commas(slowest as f64),
            elapsed,
            format_with_commas(follower_rate),
            format_with_commas(leader_rate)
        );

        leader_rates.push(leader_rate);
        follower_rates.push(follower_rate);
        keep_up.push(follower_rate / leader_rate);
        lags.push(worst_during as f64);
        lag_growth.push(growth);
        catch_ups.push(catch_up.as_secs_f64());

        // And the group agrees on every row of it
        let expected = LATENCY_WRITES + THROUGHPUT_WRITES;
        for node in &group.nodes {
            assert_eq!(
                node.count("bench").await,
                expected,
                "{} does not hold every row the group committed",
                node.name
            );
        }
        group.shutdown().await;
    }

    record_metric(
        "Replication",
        "Replicated write throughput, three nodes",
        " txn/sec",
        leader_rates,
    );
    record_metric("Replication", "Write latency p50, unsaturated", "s", p50s);
    record_metric("Replication", "Write latency p99, unsaturated", "s", p99s);
    record_metric(
        "Replication",
        "Follower apply rate",
        " entries/sec",
        follower_rates,
    );
    record_metric("Replication", "Worst follower lag", " entries", lags);
    record_metric("Replication", "Time to close the lag", "s", catch_ups);

    // A follower that cannot keep up shows a lag that climbs with every write.
    // One that keeps up shows a queue of roughly the client's concurrency,
    // whatever the load runs for
    let r = validate_metric_with_unit(
        "Replication",
        "Follower lag, second half against first",
        "x",
        lag_growth,
        2.0,
        false,
    );
    assert!(
        r.passed,
        "follower lag grows through the load, so the applier is behind the leader"
    );

    // The one figure that decides whether applying in parallel would buy
    // anything. A follower that applies at least as fast as the leader commits
    // never falls behind however long the load runs, and the lag above is a
    // queue rather than a deficit
    let r = validate_metric_with_unit(
        "Replication",
        "Follower apply rate against leader commit rate",
        "x",
        keep_up,
        0.99,
        true,
    );
    assert!(
        r.passed,
        "a follower applies more slowly than the leader commits, so its lag grows without bound"
    );
}

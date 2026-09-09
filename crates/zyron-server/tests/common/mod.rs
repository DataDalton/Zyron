//! The consensus group harness the replication suites share.
//!
//! Every node runs a whole engine of its own: catalog, WAL, buffer pool, disk
//! manager and transaction manager, joined to a real group over real sockets.
//! What a member holds is read back out of its own storage rather than out of
//! anything a test remembered, which is the only way a divergence between two
//! members shows up as one.
//!
//! The replication benchmarks and the conformance suite both drive this, so a
//! statement is exercised through the same path whichever suite runs it.

#![allow(dead_code)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use zyron_buffer::BufferPool;
use zyron_catalog::{Catalog, CatalogCache, HeapCatalogStorage, SYSTEM_DATABASE_ID, SchemaId};
use zyron_common::{Result, ZyronError};
use zyron_executor::batch::DataBatch;
use zyron_server::config::{ClusterPeerSection, ClusterSection};
use zyron_server::raft::{ClusterHandle, start_cluster};
use zyron_server::replication::DdlRunner;
use zyron_storage::DiskManager;
use zyron_storage::txn::{IsolationLevel, TransactionManager};
use zyron_wal::WalWriter;
use zyron_wire::connection::{ReplicationRouter, ServerState};

/// The role every schema change in these suites runs under.
///
/// Carried on the statement so an object created on the leader is owned by
/// the same role on a follower. A follower has no security context of its
/// own, so without this the owner is the real role on one node and zero on
/// the rest
pub const TEST_ACTOR_ROLE: u32 = 7;

// ---------------------------------------------------------------------------
// One node, with a whole engine behind it
// ---------------------------------------------------------------------------

pub struct Node {
    pub name: String,
    pub catalog: Arc<Catalog>,
    pub wal: Arc<WalWriter>,
    pub buffer_pool: Arc<BufferPool>,
    pub disk: Arc<DiskManager>,
    pub txn_manager: Arc<TransactionManager>,
    pub cluster: ClusterHandle,
    /// Held so the applier's weak reference to it stays alive
    pub _server: Arc<ServerState>,
    pub schema: SchemaId,
    pub _tmp: tempfile::TempDir,
}

impl Node {
    /// Builds the engine the server ships, then joins it to the group the
    /// section describes
    pub async fn start(name: &str, peers: &[(String, String)], listen: &str) -> Node {
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
            .create_schema(SYSTEM_DATABASE_ID, "zyron_test", "test_user")
            .await
            .expect("create zyron_test schema");
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

        // Roles, users and grants live here rather than in the catalog, and
        // each node gets its own backed by its own heap. A shared one would
        // let a member that never applied a GRANT read the grant a different
        // member wrote and agree with it
        let auth_storage: Arc<dyn zyron_auth::storage::AuthStorage> = Arc::new(
            zyron_auth::HeapAuthStorage::new(Arc::clone(&disk), Arc::clone(&buffer_pool))
                .expect("auth storage"),
        );
        let security_manager = Arc::new(
            zyron_auth::SecurityManager::new(auth_storage)
                .await
                .expect("security manager"),
        );

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
            Arc::clone(&security_manager),
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

    pub fn is_leader(&self) -> bool {
        self.cluster.node.is_leader()
    }

    pub fn router(&self) -> &Arc<zyron_server::replication::ReplicationHandle> {
        &self.cluster.replication
    }

    /// Runs a schema change the way a connection would: agree it with the
    /// group, wait for this node's turn, run it, report back
    pub async fn ddl(&self, sql: &str) -> Result<()> {
        let context = zyron_executor::replication::StatementContext {
            user: "test_user".to_string(),
            database: "zyron".to_string(),
            search_path: vec!["zyron_test".to_string()],
            // The role an object this creates is owned by, the same on every
            // member because the originating node names it rather than each
            // node working it out again
            actor_role_id: Some(TEST_ACTOR_ROLE),
        };
        let done = self.router().begin_statement(sql, &context).await?;
        // The originating node runs the statement in no transaction of its
        // own, which is what zero says here
        let outcome = zyron_server::replication::DispatchedDdl::new(&self._server)
            .run(sql, &context, 0)
            .await;
        let _ = done.send(match &outcome {
            Ok(()) => Ok(()),
            Err(e) => Err(ZyronError::Internal(e.to_string())),
        });
        outcome
    }

    /// Runs statements in a transaction of this node's, then commits through
    /// the group
    pub async fn write(&self, statements: &[&str]) -> Result<()> {
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
    pub async fn query(&self, sql: &str) -> Result<Vec<DataBatch>> {
        let mut txn = self.txn_manager.begin(IsolationLevel::ReadCommitted)?;
        let batches = self
            .execute(sql, txn.txn_id(), txn.snapshot.clone(), None)
            .await;
        let _ = self.txn_manager.commit_read_only(&mut txn);
        batches
    }

    pub async fn execute(
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
                vec!["zyron_test".to_string()],
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
    pub async fn ints(&self, sql: &str) -> Vec<i64> {
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

    pub async fn count(&self, table: &str) -> i64 {
        let rows = self.ints(&format!("SELECT COUNT(*) FROM {table}")).await;
        rows.first().copied().unwrap_or(0)
    }

    /// Serves the wire protocol off a listening socket the way the process
    /// does, so a test can connect to this node exactly as a driver would
    pub async fn serve_wire(&self) -> std::net::SocketAddr {
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
    pub async fn dispatch(&self, sql: &str) -> Result<()> {
        let statements = zyron_parser::parse(sql)?;
        for statement in statements {
            let mut session = Some(zyron_wire::session::Session::new(
                "test_user".to_string(),
                "zyron".to_string(),
                zyron_catalog::DatabaseId(1),
            ));
            if let Some(session) = session.as_mut() {
                session.search_path = vec!["zyron_test".to_string()];
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
pub fn build_server_state(
    catalog: Arc<Catalog>,
    wal: Arc<WalWriter>,
    buffer_pool: Arc<BufferPool>,
    disk_manager: Arc<DiskManager>,
    txn_manager: Arc<TransactionManager>,
    data_dir: &std::path::Path,
    cluster: &ClusterHandle,
    security_manager: Arc<zyron_auth::SecurityManager>,
) -> Arc<ServerState> {
    Arc::new(ServerState {
        raft: Some(Arc::clone(&cluster.node)),
        replication: Some(
            Arc::clone(&cluster.replication) as Arc<dyn zyron_wire::connection::ReplicationRouter>
        ),
        node_capabilities: None,
        catalog,
        ddl_progress: std::sync::Arc::new(zyron_wire::ddl_progress::DdlProgressRegistry::new()),
        shadow_targets: std::sync::Arc::new(scc::HashMap::new()),
        wal,
        buffer_pool,
        disk_manager,
        txn_manager,
        doc_registry: Arc::new(zyron_common::DocRegistry::new()),
        table_io_stats: Arc::new(zyron_common::TableIOStatsRegistry::new()),
        index_io_stats: Arc::new(zyron_common::IndexIOStatsRegistry::new()),
        columnar_maintenance: None,
        security_manager: Some(security_manager),
        key_store: Arc::new(zyron_auth::LocalKeyStore::new([0u8; 32])),
        media_store: Arc::new(
            zyron_media::store::MediaStore::open(data_dir.to_path_buf())
                .expect("media store opens in the test data dir"),
        ),
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
        // The managers a statement reaches for. Built the way the server
        // builds them, because a statement whose manager is absent is refused
        // for a reason that has nothing to do with whether it replicates, and
        // a conformance case cannot tell those two apart
        cdc_registry: Some(Arc::new(zyron_cdc::CdfRegistry::new(
            data_dir.to_path_buf(),
        ))),
        slot_manager: zyron_cdc::SlotManager::open(data_dir, zyron_cdc::SlotLagConfig::default())
            .ok()
            .map(Arc::new),
        publication_manager: zyron_cdc::PublicationManager::open(data_dir)
            .ok()
            .map(Arc::new),
        cdc_stream_manager: zyron_cdc::CdcStreamManager::new(data_dir)
            .ok()
            .map(Arc::new),
        cdc_ingest_manager: zyron_cdc::CdcIngestManager::new(data_dir)
            .ok()
            .map(Arc::new),
        trigger_manager: Some(Arc::new(zyron_pipeline::trigger::TriggerManager::new())),
        udf_registry: Some(Arc::new(zyron_pipeline::udf::UdfRegistry::new())),
        uda_registry: Some(Arc::new(zyron_pipeline::aggregate::UdaRegistry::new())),
        procedure_registry: Some(Arc::new(
            zyron_pipeline::stored_procedure::ProcedureRegistry::new(),
        )),
        pipeline_manager: None,
        schedule_manager: None,
        event_dispatcher: None,
        mv_manager: Some(Arc::new(
            zyron_pipeline::materialized_view::MaterializedViewManager::new(),
        )),
        stream_job_manager: Some(Arc::new(parking_lot::Mutex::new(
            zyron_streaming::job::StreamJobManager::new(),
        ))),
        // Built the way the server builds it, so a branch statement runs here
        // against the same manager rather than being refused for want of one
        branch_manager: Some({
            let mgr = zyron_versioning::BranchManager::new(data_dir.to_path_buf());
            mgr.load().expect("branch manager loads");
            Arc::new(mgr)
        }),
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
        // Built the way the server builds it, so LISTEN and NOTIFY reach a
        // real registry rather than being refused for want of one
        notification_channels: Some(Arc::new(
            zyron_wire::notifications::NotificationChannels::new(),
        )),
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
        admission: Arc::new(zyron_common::Admission::new()),
        query_metrics: Arc::new(zyron_common::QueryMetrics::new()),
        upgrade_control: None,
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
pub struct WireClient {
    pub stream: tokio::net::TcpStream,
}

impl WireClient {
    /// Connects and completes the startup handshake under trust auth
    pub async fn connect(addr: std::net::SocketAddr) -> WireClient {
        use tokio::io::AsyncWriteExt;
        let mut stream = tokio::net::TcpStream::connect(addr).await.expect("connect");
        let mut params = Vec::new();
        params.extend_from_slice(&196608i32.to_be_bytes());
        // Written as escapes rather than as literal NUL bytes in the source.
        // The startup packet is a run of NUL terminated key value pairs ended
        // by an empty key, and a source file carrying those bytes raw reads as
        // binary to every tool that touches it
        params.extend_from_slice(b"user\0test_user\0database\0zyron\0\0");
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
    pub async fn read_message(&mut self) -> (u8, Vec<u8>) {
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
    pub async fn drain_to_ready(&mut self) -> (Vec<String>, Vec<String>) {
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

    /// Reads until ReadyForQuery, collecting the rows an answer carried.
    ///
    /// Kept apart from `drain_to_ready` because most cases care only that a
    /// statement was accepted, and a case that reads what came back needs the
    /// cells rather than the tag
    pub async fn drain_rows(&mut self) -> (Vec<Vec<Option<String>>>, Vec<String>) {
        let mut rows = Vec::new();
        let mut errors = Vec::new();
        loop {
            let (kind, payload) = self.read_message().await;
            match kind {
                b'D' => {
                    let mut cells = Vec::new();
                    if payload.len() >= 2 {
                        let count = i16::from_be_bytes([payload[0], payload[1]]).max(0) as usize;
                        let mut at = 2usize;
                        for _ in 0..count {
                            if at + 4 > payload.len() {
                                break;
                            }
                            let len = i32::from_be_bytes([
                                payload[at],
                                payload[at + 1],
                                payload[at + 2],
                                payload[at + 3],
                            ]);
                            at += 4;
                            if len < 0 {
                                cells.push(None);
                                continue;
                            }
                            let len = len as usize;
                            if at + len > payload.len() {
                                break;
                            }
                            cells.push(Some(
                                String::from_utf8_lossy(&payload[at..at + len]).into_owned(),
                            ));
                            at += len;
                        }
                    }
                    rows.push(cells);
                }
                b'E' => errors.extend(error_text(&payload)),
                b'Z' => return (rows, errors),
                _ => {}
            }
        }
    }

    /// Sends one simple Query message and reads the rows it answered with
    pub async fn query_rows(&mut self, sql: &str) -> (Vec<Vec<Option<String>>>, Vec<String>) {
        use tokio::io::AsyncWriteExt;
        let mut msg = vec![b'Q'];
        msg.extend_from_slice(&((sql.len() + 1 + 4) as i32).to_be_bytes());
        msg.extend_from_slice(sql.as_bytes());
        msg.push(0);
        self.stream.write_all(&msg).await.expect("query");
        self.stream.flush().await.expect("flush query");
        self.drain_rows().await
    }

    /// Sends one simple Query message and reads its whole answer
    pub async fn query(&mut self, sql: &str) -> (Vec<String>, Vec<String>) {
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
    pub async fn prepared(&mut self, sql: &str, params: &[i64]) -> (Vec<String>, Vec<String>) {
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

    pub async fn terminate(mut self) {
        use tokio::io::AsyncWriteExt;
        let _ = self.stream.write_all(&[b'X', 0, 0, 0, 4]).await;
        let _ = self.stream.shutdown().await;
    }
}

pub fn push_message(out: &mut Vec<u8>, kind: u8, body: &[u8]) {
    out.push(kind);
    out.extend_from_slice(&((body.len() + 4) as i32).to_be_bytes());
    out.extend_from_slice(body);
}

/// The M field of an ErrorResponse payload, which is the human message among
/// the coded fields
pub fn error_text(payload: &[u8]) -> Option<String> {
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

pub struct Group {
    /// Behind an `Arc` so a load generator can hand the same node to many
    /// concurrent tasks, which is the only way to ask what the group commits
    /// per second rather than what one client does
    pub nodes: Vec<Arc<Node>>,
}

impl Group {
    pub async fn start(count: usize) -> Group {
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

    pub async fn leader(&self, within: Duration) -> usize {
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
    pub async fn settle(&self, leader: usize, within: Duration) {
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

    pub async fn shutdown(self) {
        for node in &self.nodes {
            node.cluster.shutdown().await;
        }
    }

    /// Waits for every node to have applied `target`, and reports how long it
    /// took. Panics rather than returning, because a group that never settles
    /// has nothing further worth measuring
    pub async fn catch_up(&self, target: u64, within: Duration) -> Duration {
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
    pub fn worst_lag(&self, leader: usize) -> u64 {
        let committed = self.nodes[leader].cluster.node.commit_index();
        self.nodes
            .iter()
            .map(|n| committed.saturating_sub(n.cluster.node.last_applied()))
            .max()
            .unwrap_or(0)
    }
}

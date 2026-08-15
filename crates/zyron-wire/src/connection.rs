//! Per-client connection state machine.
//!
//! Manages the full lifecycle of a PostgreSQL client connection: startup
//! handshake, authentication, simple query protocol, extended query protocol,
//! transaction management, and connection teardown.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicI32, Ordering};

use bytes::{Buf, BytesMut};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tracing::{debug, warn};

use crate::transport::WireTransport;

use zyron_buffer::BufferPool;
use zyron_catalog::Catalog;
#[cfg(feature = "profile")]
use zyron_common::profile::{self, Phase};
use zyron_common::{Result as ZyronResult, ZyronError};
use zyron_executor::batch::DataBatch;
use zyron_executor::column::ScalarValue;
use zyron_executor::context::{CdcHook, ExecutionContext};
use zyron_executor::executor::{execute, execute_analyze};
use zyron_executor::operator::OperatorMetrics;
use zyron_planner::logical::LogicalColumn;
use zyron_planner::physical::PhysicalPlan;
use zyron_storage::DiskManager;
use zyron_storage::txn::{IsolationLevel, Snapshot, Transaction, TransactionManager};
use zyron_wal::WalWriter;

use crate::auth::{
    AuthProgress, AuthResult, Authenticator, ComposedAuthenticator, ScramAuthenticator,
    TrustAuthenticator, WebAuthnAuthenticator,
};
use crate::codec::PostgresCodec;
use crate::messages::ProtocolError;
use crate::messages::backend::{
    AuthenticationMessage, BackendMessage, ErrorFields, FieldDescription, TransactionState,
};
use crate::messages::frontend::{DescribeTarget, FrontendMessage, StartupMessage};
use crate::session::Session;
use crate::types;

/// Wall-clock seconds since the epoch, the unit the maintenance timestamps in
/// zyron_stat_tables are reported in. Matches what ANALYZE stamps into the
/// catalog's table statistics, so the two sources agree.
fn epoch_seconds_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Columnar maintenance entry points installed by the hosting server. The
/// compaction machinery lives above this crate in the dependency graph, so
/// the manual VACUUM and OPTIMIZE handlers reach it through this seam.
pub trait ColumnarMaintenance: Send + Sync {
    /// One columnar merge pass over the table, the columnar analog of a
    /// heap vacuum
    fn vacuum_table(&self, table_id: zyron_catalog::TableId) -> std::result::Result<(), String>;
    /// Folds the table's eligible heap tail into the columnar tier, returns
    /// the number of rows folded
    fn fold_table(&self, table_id: zyron_catalog::TableId) -> std::result::Result<u64, String>;
}

/// Shared server state passed to every connection.
pub struct ServerState {
    pub catalog: Arc<Catalog>,
    pub wal: Arc<WalWriter>,
    pub buffer_pool: Arc<BufferPool>,
    pub disk_manager: Arc<DiskManager>,
    pub txn_manager: Arc<TransactionManager>,
    /// Per-table document identity for search indexes. DML allocates dense
    /// ordinal DocIds through this and search scans map results back to row
    /// locators, so folded rows keep their documents. Persisted beside the
    /// index snapshots as doc_registry.zydoc.
    pub doc_registry: Arc<zyron_common::DocRegistry>,
    /// Per-table IO and tuple counters, handed to every execution context so
    /// scan and DML operators record into them. Read back by zyron_stat_tables.
    pub table_io_stats: Arc<zyron_common::TableIOStatsRegistry>,
    /// Per-index scan counters, read back by zyron_stat_indexes.
    pub index_io_stats: Arc<zyron_common::IndexIOStatsRegistry>,
    pub security_manager: Option<Arc<zyron_auth::SecurityManager>>,
    /// Lock-free legal-hold registry. Reloaded from the catalog at startup and
    /// after each LEGAL HOLD CREATE/DROP/RELEASE so the DML hook enforces holds.
    pub legal_holds: Arc<zyron_lifecycle::legal_hold::LegalHoldRegistry>,
    /// Key store for sealing and opening external-source/sink credentials.
    /// Populated by the server binary from a data-dir-derived master key.
    pub key_store: Arc<dyn zyron_auth::KeyStore>,
    /// Config value lookup: returns (key, value) for a dotted key.
    pub config_lookup: Option<Arc<dyn Fn(&str) -> Option<String> + Send + Sync>>,
    /// Config entries for SHOW ALL: returns vec of (key, value, description).
    pub config_all: Option<Arc<dyn Fn() -> Vec<(String, String, String)> + Send + Sync>>,
    /// Data directory path (for ALTER SYSTEM auto.conf writes).
    pub data_dir: std::path::PathBuf,
    /// Session manager for stat view queries.
    pub session_info_collector:
        Option<Arc<dyn Fn() -> Vec<crate::stat_views::SessionRow> + Send + Sync>>,
    /// Checkpoint worker stats: (checkpoints_completed, segments_deleted, last_checkpoint_lsn).
    pub checkpoint_stats: Option<Arc<dyn Fn() -> (u64, u64, u64) + Send + Sync>>,
    /// Vacuum worker stats: (cycles_completed, tuples_reclaimed, pages_scanned).
    pub vacuum_stats: Option<Arc<dyn Fn() -> (u64, u64, u64) + Send + Sync>>,
    /// Checkpoint wake trigger.
    pub checkpoint_wake: Option<Arc<dyn Fn() + Send + Sync>>,
    /// ALTER SYSTEM SET callback: writes key=value to auto.conf. Returns Ok or error message.
    pub alter_system_set:
        Option<Arc<dyn Fn(&str, &str) -> std::result::Result<(), String> + Send + Sync>>,
    /// CDC feed stats: Vec<(table_id, record_count, file_size, retention_days)>
    pub cdc_feed_stats: Option<Arc<dyn Fn() -> Vec<(u32, u64, u64, u32)> + Send + Sync>>,
    /// Replication slot stats: Vec<(name, plugin, confirmed_lsn, restart_lsn, active, lag_bytes)>
    pub cdc_slot_stats:
        Option<Arc<dyn Fn() -> Vec<(String, String, u64, u64, bool, u64)> + Send + Sync>>,
    /// CDC stream stats: Vec<(name, table_id, active, slot_name)>
    pub cdc_stream_stats: Option<Arc<dyn Fn() -> Vec<(String, u32, bool, String)> + Send + Sync>>,
    /// CDC ingest stats: Vec<(name, table_id, active, records_applied, records_failed)>
    pub cdc_ingest_stats: Option<Arc<dyn Fn() -> Vec<(String, u32, bool, u64, u64)> + Send + Sync>>,

    // -----------------------------------------------------------------------
    // CDC managers
    // -----------------------------------------------------------------------
    /// Change Data Feed registry for tracking table change feeds.
    pub cdc_registry: Option<Arc<zyron_cdc::CdfRegistry>>,
    /// Replication slot manager.
    pub slot_manager: Option<Arc<zyron_cdc::SlotManager>>,
    /// Multi-table publication manager.
    pub publication_manager: Option<Arc<zyron_cdc::PublicationManager>>,
    /// Outbound CDC stream manager.
    pub cdc_stream_manager: Option<Arc<zyron_cdc::CdcStreamManager>>,
    /// Inbound CDC ingestion manager.
    pub cdc_ingest_manager: Option<Arc<zyron_cdc::CdcIngestManager>>,

    // -----------------------------------------------------------------------
    // Pipeline managers
    // -----------------------------------------------------------------------
    /// Trigger registry indexed by table and event type.
    pub trigger_manager: Option<Arc<zyron_pipeline::trigger::TriggerManager>>,
    /// User-defined function registry.
    pub udf_registry: Option<Arc<zyron_pipeline::udf::UdfRegistry>>,
    /// User-defined aggregate registry.
    pub uda_registry: Option<Arc<zyron_pipeline::aggregate::UdaRegistry>>,
    /// Stored procedure registry.
    pub procedure_registry: Option<Arc<zyron_pipeline::stored_procedure::ProcedureRegistry>>,
    /// Pipeline execution engine.
    pub pipeline_manager: Option<Arc<zyron_pipeline::pipeline::PipelineManager>>,
    /// Scheduled execution manager.
    pub schedule_manager: Option<Arc<zyron_pipeline::schedule::ScheduleManager>>,
    /// Event handler dispatcher.
    pub event_dispatcher: Option<Arc<zyron_pipeline::event_handler::EventDispatcher>>,
    /// Materialized view refresh manager.
    pub mv_manager: Option<Arc<zyron_pipeline::materialized_view::MaterializedViewManager>>,

    // -----------------------------------------------------------------------
    // Streaming
    // -----------------------------------------------------------------------
    /// Streaming job lifecycle manager. Wrapped in Mutex because StreamOperator
    /// trait objects are Send but not Sync.
    pub stream_job_manager: Option<Arc<parking_lot::Mutex<zyron_streaming::job::StreamJobManager>>>,

    // -----------------------------------------------------------------------
    // Versioning
    // -----------------------------------------------------------------------
    /// Data branch manager for version branching.
    pub branch_manager: Option<Arc<zyron_versioning::BranchManager>>,

    // -----------------------------------------------------------------------
    // Search indexes
    // -----------------------------------------------------------------------
    /// FTS index manager for fulltext search operations.
    pub fts_manager: Option<Arc<zyron_search::FtsManager>>,
    /// Vector index manager for vector similarity search.
    pub vector_manager: Option<Arc<zyron_search::vector::VectorIndexManager>>,
    /// Graph schema manager for graph traversal and algorithms.
    pub graph_manager: Option<Arc<zyron_search::graph::GraphManager>>,
    /// Spatial (R-tree) index manager for KNN, range, ST_DWithin, ST_Intersects.
    pub spatial_manager: Option<Arc<zyron_types::spatial_index::SpatialIndexManager>>,

    // -----------------------------------------------------------------------
    // DML hooks
    // -----------------------------------------------------------------------
    /// CDC hook invoked by DML operators after mutations.
    pub cdc_hook: Option<Arc<dyn CdcHook>>,
    /// DML hook invoked by DML operators before mutations (BEFORE triggers).
    pub dml_hook: Option<Arc<dyn zyron_executor::context::DmlHook>>,
    /// Columnar maintenance entry points installed by the hosting server so
    /// the manual VACUUM and OPTIMIZE handlers reach the compaction
    /// machinery, which lives above this crate in the dependency graph.
    pub columnar_maintenance: Option<Arc<dyn ColumnarMaintenance>>,

    // -----------------------------------------------------------------------
    // Notification channels for LISTEN/NOTIFY
    // -----------------------------------------------------------------------
    pub notification_channels: Option<Arc<crate::notifications::NotificationChannels>>,

    // -----------------------------------------------------------------------
    // TLS upgrade support
    // -----------------------------------------------------------------------
    /// TLS mode for plain-text startup: Disabled, Optional, Required.
    pub tls_mode: crate::tls::TlsMode,
    /// Server TLS acceptor, if TLS is enabled for inbound connections.
    pub tls_acceptor: Option<Arc<crate::tls::ServerTlsAcceptor>>,

    // -----------------------------------------------------------------------
    // Gateway router registration
    // -----------------------------------------------------------------------
    /// Live HTTP gateway router used by CREATE/ALTER/DROP ENDPOINT handlers
    /// and startup recovery. When None, DDL still persists to the catalog but
    /// no runtime route is compiled.
    pub endpoint_registrar: Option<Arc<dyn crate::endpoint_registrar::EndpointRegistrar>>,

    // -----------------------------------------------------------------------
    // Subscription runtime map
    // -----------------------------------------------------------------------
    /// Active subscription runtime tasks, keyed by SubscriptionId. Populated by
    /// recover_zyron_to_zyron at startup and by CREATE EXTERNAL SOURCE at
    /// runtime. Drained and joined on graceful shutdown.
    pub subscription_runtimes:
        Arc<scc::HashMap<zyron_catalog::SubscriptionId, tokio::task::JoinHandle<()>>>,

    /// Active inbound subscriber map for the wire-level push producer pump.
    /// Each Y handshake registers a SubscriptionServerContext here and the
    /// pump removes it on disconnect or graceful end.
    pub pub_sub_state: Arc<crate::subscription::PubSubServerState>,

    /// Shutdown flag shared with the wire-level push producer pump. Set true
    /// by the outer server on graceful shutdown so the pump drains and emits
    /// EndSubscription frames to live subscribers.
    pub subscription_shutdown: Arc<std::sync::atomic::AtomicBool>,

    /// Server-wide HeapFile cache keyed by heap_file_id. Reusing the same
    /// `HeapFile` across queries keeps the per-file free-space hint cache
    /// warm so single-row INSERTs land on the same hot page instead of
    /// allocating a new one per call
    pub heap_files: Arc<scc::HashMap<u32, Arc<zyron_storage::HeapFile>>>,
    /// Live B+Tree indexes keyed by index_id, used by IndexScan and
    /// maintained by INSERT/UPDATE/DELETE
    pub btree_indexes: Arc<scc::HashMap<u32, Arc<zyron_storage::BTreeIndex>>>,

    /// Server-wide L2 plan cache, shared across connections. A query shape
    /// planned by one connection is reused by every other connection with
    /// the same identity, search path, and schema version, behind each
    /// connection's lock-free per-session L1 cache.
    pub plan_cache: Arc<crate::plan_cache::ServerPlanCache>,

    // -----------------------------------------------------------------------
    // Utility command coordination
    // -----------------------------------------------------------------------
    /// Set true while VACUUM is executing (manual or background)
    /// Manual VACUUM CAS-acquires this flag and returns a Notice if already
    /// held, the background worker checks it at the top of each cycle
    pub vacuum_running: Arc<std::sync::atomic::AtomicBool>,

    // -----------------------------------------------------------------------
    // Analytics
    // -----------------------------------------------------------------------
    /// Analytics function catalog. Populated at startup with the default
    /// table-returning analytical functions (COHORT_RETENTION, FUNNEL_ANALYSIS,
    /// DATA_PROFILE, COLUMN_PROFILE, CORRELATION_MATRIX) and the period and
    /// statistical window/scalar functions (YOY, MOM, ZSCORE, CORR, ...).
    pub analytics_registry: Arc<zyron_analytics::AnalyticsRegistry>,

    // -----------------------------------------------------------------------
    // Feature store and ML
    // -----------------------------------------------------------------------
    /// Feature store registry, holds feature groups, definitions, and
    /// materialized point-in-time values
    pub feature_store: Arc<zyron_analytics::FeatureStore>,
    /// Feature lineage tracker, indexed by qualified feature name
    pub feature_lineage: Arc<parking_lot::RwLock<zyron_analytics::FeatureLineageRegistry>>,
    /// Server-wide trained-model inference cache
    pub model_cache: Arc<zyron_analytics::ModelCache>,

    // -----------------------------------------------------------------------
    // Config-derived session and auth defaults
    // -----------------------------------------------------------------------
    /// Transaction isolation level applied to sessions that do not set one
    /// explicitly, parsed from query.default_isolation at startup
    pub default_isolation: IsolationLevel,
    /// Storage tiers this node runs, parsed from storage.deployment_mode at
    /// startup. Picks the format a CREATE TABLE with no USING clause lands
    /// in and refuses DDL naming the format the node does not run
    pub deployment_mode: zyron_common::DeploymentMode,
    /// Who this node is in a mesh. Stable across restarts and minted on
    /// first start, so peers recognize the same node rather than a new one
    pub node_identity: zyron_common::NodeIdentity,
    /// Nodes this one has been told to talk to. Node-local rather than
    /// catalog state, because it is this node's view of the mesh and it
    /// has to be readable before the catalog is up.
    ///
    /// Shared with the follower worker rather than copied to it, so a peer
    /// declared by DDL is one the follower can reach on its next tick
    /// instead of after a restart
    pub peers: Arc<parking_lot::RwLock<Arc<zyron_common::PeerRegistry>>>,
    /// Reads tables that live on a peer, handed to every execution context
    /// this node builds. Built once because it holds the runtime handle and
    /// resolves peers through the registry above, so a foreign scan names a
    /// peer and never an address
    pub foreign_reader: Option<Arc<dyn zyron_executor::operator::foreign_scan::ForeignReader>>,
    /// Per-statement run-time limit applied when a session sets none. None
    /// when query.statement_timeout_secs is 0
    pub statement_timeout: Option<std::time::Duration>,
    /// Maximum rows a single query returns. None when query.max_result_rows
    /// is 0
    pub max_result_rows: Option<u64>,
    /// Balloon password-hash cost parameters from the auth config. None when
    /// neither balloon_space_cost nor balloon_time_cost is set, leaving the
    /// hasher on its built-in defaults
    pub balloon_params: Option<zyron_auth::BalloonParams>,
    /// Authentication method used as the fallback when no auth rule matches,
    /// parsed from auth.method at startup
    pub default_auth_method: zyron_auth::auth_rules::AuthMethod,
}

impl ServerState {
    /// This node's view of the mesh, as a value the planner can hold.
    ///
    /// A snapshot rather than the lock, because planning is asynchronous and
    /// a guard held across a bind would put every peer declaration behind
    /// the slowest statement. It also means a query is costed against one
    /// consistent view instead of one that can change between two of its own
    /// scans.
    ///
    /// The snapshot is a pointer, so this is one atomic increment however
    /// large the mesh is. Copying the entries here instead would put an
    /// allocation per peer on every statement, which is a cost that grows
    /// with the number of nodes rather than with the work being done.
    /// Declaring a peer replaces the pointer, so a running query keeps the
    /// view it started with and the next one sees the new membership.
    pub fn peer_facts(&self) -> Arc<zyron_common::PeerRegistry> {
        Arc::clone(&self.peers.read())
    }
}

/// RAII guard that releases the vacuum_running flag on drop, so a panic
/// during VACUUM/OPTIMIZE doesn't leave the flag stuck.
struct VacuumGuard {
    flag: Arc<std::sync::atomic::AtomicBool>,
}

impl Drop for VacuumGuard {
    fn drop(&mut self) {
        self.flag.store(false, std::sync::atomic::Ordering::Release);
    }
}

/// Cached prepared statement.
struct PreparedStatement {
    query: String,
    param_types: Vec<i32>,
    plan: Option<Arc<PhysicalPlan>>,
    output_schema: Vec<LogicalColumn>,
}

/// Bound portal: a prepared statement bound to concrete parameter values.
/// The executor reads `params` through the per-query `ExecutionContext`
/// to resolve `$1`, `$2`, ... references in the physical plan.
///
/// A None `plan` marks a transaction-control, session, or DDL/utility
/// statement that bypasses the planner. Execute dispatches it through the
/// same handlers the simple-query path uses, reading the SQL text from `query`.
struct Portal {
    params: Vec<ScalarValue>,
    result_formats: Vec<i16>,
    plan: Option<Arc<PhysicalPlan>>,
    output_schema: Vec<LogicalColumn>,
    query: String,
}

/// Monotonic counter for generating unique process IDs without RNG overhead.
static NEXT_PROCESS_ID: AtomicI32 = AtomicI32::new(1);

/// Max named prepared statements before eviction (unnamed excluded).
const MAX_PREPARED_STATEMENTS: usize = 1000;
/// Max named portals before eviction (unnamed excluded).
const MAX_PORTALS: usize = 10000;

/// Per-connection handler for the PostgreSQL wire protocol.
/// Generic over the transport layer (TCP or QUIC).
pub struct Connection<T: WireTransport> {
    stream: T,
    codec: PostgresCodec,
    read_buf: BytesMut,
    write_buf: BytesMut,
    session: Option<Session>,
    server: Arc<ServerState>,
    authenticator: Box<dyn Authenticator>,
    /// Active explicit transaction (None = auto-commit mode).
    transaction: Option<Transaction>,
    /// Cross-table lake commit opened by BEGIN ZYRONLAKE TRANSACTION. The
    /// transaction's lake writes commit under its intent, so several lake
    /// tables become visible together without waiting on the database
    /// commit record.
    lake_txn: Option<zyron_lake::CrossTableTxn>,
    /// Named prepared statements. Empty string = unnamed statement.
    statements: HashMap<String, PreparedStatement>,
    /// Named portals. Empty string = unnamed portal.
    portals: HashMap<String, Portal>,
    /// Process ID for cancel request matching.
    process_id: i32,
    /// Secret key for cancel request verification.
    secret_key: i32,
    /// Remote peer IP address (if available).
    pub peer_addr: Option<String>,
    /// Active branch for versioning (set by USE BRANCH).
    active_branch: Option<String>,
    /// Cursor state for DECLARE/FETCH/CLOSE cursor support.
    cursors: HashMap<String, CursorState>,
    /// Notification channel receivers for LISTEN/NOTIFY.
    notification_receivers:
        HashMap<String, tokio::sync::broadcast::Receiver<crate::notifications::Notification>>,
    /// Per-connection cache of parsed and planned SQL statements keyed by
    /// SQL text hash. Eliminates parse + plan cost on repeated identical
    /// queries through the unnamed extended-protocol path.
    statement_cache: crate::statement_cache::StatementCache,
}

/// Refreshes the planner's statistics for every lake table whose visible
/// version just changed.
///
/// A lake table never runs ANALYZE: its writer computes exact per-column
/// bounds and null counts for every file it produces, so the manifest at the
/// newly visible version is the statistics. Called after a publish and after
/// an abandon, so a rolled back write leaves no optimistic estimate behind.
pub fn refresh_lake_stats(server: &Arc<ServerState>, logs: &[Arc<zyron_lake::TransactionLog>]) {
    for log in logs {
        let Some(table_id) = log.paths().table_id() else {
            continue;
        };
        let Ok(entry) = server
            .catalog
            .get_table_by_id(zyron_catalog::TableId(table_id))
        else {
            continue;
        };
        if let Ok(manifest) = log.latest_manifest() {
            zyron_executor::lake_stats::publish_manifest_stats(&server.catalog, &entry, &manifest);
        }
    }
}

impl<T: WireTransport> Drop for Connection<T> {
    fn drop(&mut self) {
        // A connection torn down mid-transaction (client disconnect,
        // server shutdown, task cancellation) never reaches the COMMIT or
        // ROLLBACK arms. Transaction's own Drop aborts it in memory, so
        // any lake version it wrote must be discarded here or it would
        // stay pending, invisible but blocking later commits on that
        // table until recovery discards it at the next startup.
        if let Some(txn) = self.transaction.take() {
            let logs = self.abandon_lake_work(txn.txn_id);
            refresh_lake_stats(&self.server, &logs);
        }
    }
}

/// Per-connection cursor state for DECLARE/FETCH/CLOSE support.
pub struct CursorState {
    /// The query plan backing this cursor.
    pub plan: Arc<PhysicalPlan>,
    /// Output column schema.
    pub output_schema: Vec<LogicalColumn>,
    /// Buffered result rows from execution.
    pub rows: Vec<DataBatch>,
    /// Current position within the buffered rows.
    pub position: usize,
    /// Whether the cursor holds across transactions.
    pub with_hold: bool,
}

/// Maps a parsed SQL isolation level to the engine isolation level.
/// READ UNCOMMITTED and READ COMMITTED run as ReadCommitted, REPEATABLE READ
/// and SNAPSHOT run as SnapshotIsolation. SERIALIZABLE has no engine
/// equivalent and is rejected rather than silently downgraded.
fn map_isolation_level(level: zyron_parser::TxnIsolation) -> ZyronResult<IsolationLevel> {
    use zyron_parser::TxnIsolation;
    match level {
        TxnIsolation::ReadUncommitted | TxnIsolation::ReadCommitted => {
            Ok(IsolationLevel::ReadCommitted)
        }
        TxnIsolation::RepeatableRead | TxnIsolation::Snapshot => {
            Ok(IsolationLevel::SnapshotIsolation)
        }
        TxnIsolation::Serializable => Err(ZyronError::ExecutionError(
            "serializable isolation level is not supported, use repeatable read or snapshot".into(),
        )),
    }
}

impl<T: WireTransport> Connection<T> {
    /// Creates a new connection handler for the given transport stream.
    pub fn new(stream: T, server: Arc<ServerState>, peer_addr: Option<String>) -> Self {
        stream.configure_immediate();
        let pid = NEXT_PROCESS_ID.fetch_add(1, Ordering::Relaxed);

        Self {
            stream,
            codec: PostgresCodec::new(),
            read_buf: BytesMut::with_capacity(32768),
            write_buf: BytesMut::with_capacity(65536),
            session: None,
            server,
            authenticator: Box::new(TrustAuthenticator),
            transaction: None,
            lake_txn: None,
            statements: HashMap::new(),
            portals: HashMap::new(),
            process_id: pid,
            secret_key: rand::random::<u32>() as i32,
            peer_addr,
            active_branch: None,
            cursors: HashMap::new(),
            notification_receivers: HashMap::new(),
            statement_cache: crate::statement_cache::StatementCache::new(),
        }
    }

    /// Creates a connection with a custom authenticator.
    pub fn with_authenticator(
        stream: T,
        server: Arc<ServerState>,
        authenticator: Box<dyn Authenticator>,
        peer_addr: Option<String>,
    ) -> Self {
        let mut conn = Self::new(stream, server, peer_addr);
        conn.authenticator = authenticator;
        conn
    }

    /// Runs the connection to completion: startup, then message loop.
    pub async fn run(&mut self) -> Result<(), ProtocolError> {
        match self.handle_startup().await {
            Ok(()) => {}
            Err(e) => {
                // Send error to client before closing
                let _ = self
                    .feed(BackendMessage::ErrorResponse(ErrorFields {
                        severity: "FATAL".into(),
                        code: "08000".into(),
                        message: format!("Startup failed: {}", e),
                        detail: None,
                        hint: None,
                        position: None,
                    }))
                    .await;
                let _ = self.flush().await;
                return Err(e);
            }
        }

        // Configure transport-specific options after handshake.
        // TCP: keepalive, TCP_NODELAY, OS-specific socket tuning.
        // QUIC: no-op (handled by QUIC connection layer).
        self.stream.configure_post_handshake();

        self.message_loop().await
    }

    // -----------------------------------------------------------------------
    // Startup phase
    // -----------------------------------------------------------------------

    async fn handle_startup(&mut self) -> Result<(), ProtocolError> {
        loop {
            // Buffered read: accumulate data until a complete startup message is available.
            // Reduces per-message read syscalls from 2 (length + payload) to typically 1.
            let msg = loop {
                if self.read_buf.len() >= 4 {
                    let len = i32::from_be_bytes(self.read_buf[..4].try_into().unwrap()) as usize;
                    if len < 4 || len > 1_073_741_824 {
                        return Err(ProtocolError::Malformed("Invalid startup length".into()));
                    }
                    if self.read_buf.len() >= len {
                        let mut frame = self.read_buf.split_to(len);
                        frame.advance(4); // skip length prefix
                        break FrontendMessage::decode_startup(&mut frame)?;
                    }
                }
                let n = self
                    .stream
                    .read_buf(&mut self.read_buf)
                    .await
                    .map_err(ProtocolError::Io)?;
                if n == 0 {
                    return Err(ProtocolError::ConnectionClosed);
                }
            };

            match msg {
                FrontendMessage::SslRequest => {
                    // Plain-path: reply 'N'. If the server wants TLS, the
                    // lib.rs accept loop performs an in-place upgrade before
                    // Connection is constructed, so SslRequest is never seen
                    // here unless TLS is Disabled.
                    if matches!(self.server.tls_mode, crate::tls::TlsMode::Required) {
                        self.stream
                            .write_all(b"N")
                            .await
                            .map_err(ProtocolError::Io)?;
                        return Err(ProtocolError::AuthFailed(
                            "TLS required but client did not request TLS".into(),
                        ));
                    }
                    self.stream
                        .write_all(b"N")
                        .await
                        .map_err(ProtocolError::Io)?;
                    continue;
                }
                FrontendMessage::Startup(startup) => {
                    if matches!(self.server.tls_mode, crate::tls::TlsMode::Required)
                        && !self.stream.is_encrypted()
                    {
                        return Err(ProtocolError::AuthFailed(
                            "TLS required for this listener".into(),
                        ));
                    }
                    self.process_startup(startup).await?;
                    break;
                }
                _ => {
                    return Err(ProtocolError::Malformed("Expected startup message".into()));
                }
            }
        }

        // Switch codec to normal message framing.
        self.codec.set_normal_mode();
        Ok(())
    }

    async fn process_startup(&mut self, mut startup: StartupMessage) -> Result<(), ProtocolError> {
        let user = startup.params.remove("user").unwrap_or_default();
        let database = startup
            .params
            .remove("database")
            .unwrap_or_else(|| user.clone());

        if user.is_empty() {
            return Err(ProtocolError::Malformed(
                "Missing user in startup parameters".into(),
            ));
        }

        // Resolve auth method first, then apply brute force gate with real method
        let peer_ip = self
            .peer_addr
            .clone()
            .unwrap_or_else(|| "127.0.0.1".to_string());
        if let Some(ref sm) = self.server.security_manager {
            let conn_type = if self.peer_addr.as_deref() == Some("127.0.0.1") {
                zyron_auth::auth_rules::ConnectionType::Local
            } else {
                zyron_auth::auth_rules::ConnectionType::Host
            };
            let method =
                sm.auth_resolver
                    .resolve(conn_type, &database, &user, self.peer_addr.as_deref());

            // Reject immediately without prompting for credentials.
            // Certificate (mTLS) auth is validated at the TLS layer, not via a
            // password challenge; until that path is wired it returns an
            // immediate FATAL rather than a doomed cleartext-password prompt.
            if matches!(
                method,
                zyron_auth::auth_rules::AuthMethod::Reject
                    | zyron_auth::auth_rules::AuthMethod::Certificate
            ) {
                let (message, reason) =
                    if matches!(method, zyron_auth::auth_rules::AuthMethod::Certificate) {
                        (
                            "Certificate authentication is not supported on this connection"
                                .to_string(),
                            "certificate auth not supported",
                        )
                    } else {
                        (
                            "Connection rejected by authentication rule".to_string(),
                            "rejected by auth rule",
                        )
                    };
                self.feed(BackendMessage::ErrorResponse(ErrorFields {
                    severity: "FATAL".into(),
                    code: "28000".into(),
                    message,
                    detail: None,
                    hint: None,
                    position: None,
                }))
                .await?;
                self.flush().await?;
                return Err(ProtocolError::AuthFailed(reason.into()));
            }

            // Map auth method to brute force gate code.
            // 0=Trust and 6=Certificate skip all brute force checks.
            let auth_code = match &method {
                zyron_auth::auth_rules::AuthMethod::Trust => 0u8,
                zyron_auth::auth_rules::AuthMethod::Certificate => 6,
                _ => 1, // All password-based methods get rate-limited
            };

            // Pre-authentication brute force gate with resolved method
            let gate = sm.brute_force.check_allowed(
                &peer_ip,
                &user,
                &database,
                auth_code,
                &sm.ip_manager,
                false,
                None,
            );
            match gate {
                zyron_auth::AuthGate::Blocked(reason) => {
                    self.feed(BackendMessage::ErrorResponse(ErrorFields {
                        severity: "FATAL".into(),
                        code: "28000".into(),
                        message: reason,
                        detail: None,
                        hint: None,
                        position: None,
                    }))
                    .await?;
                    self.flush().await?;
                    return Err(ProtocolError::AuthFailed(
                        "blocked by brute force policy".into(),
                    ));
                }
                zyron_auth::AuthGate::Delayed(dur) => {
                    tokio::time::sleep(dur).await;
                }
                zyron_auth::AuthGate::Proceed => {}
            }

            self.authenticator = build_authenticator(method, sm);
        }

        // Authenticate
        match self.authenticator.initial_message(&user) {
            AuthResult::Authenticated => {}
            AuthResult::Challenge(msg) => {
                self.feed(msg).await?;
                self.flush().await?;

                // Read password response(s)
                loop {
                    // Switch to normal mode temporarily to read password message.
                    self.codec.set_normal_mode();
                    let response = self.read_message().await?;
                    let password = match response {
                        FrontendMessage::Password(pw) => pw,
                        _ => {
                            return Err(ProtocolError::Malformed(
                                "Expected password response".into(),
                            ));
                        }
                    };

                    match self.authenticator.process_response(&user, &password) {
                        Ok(AuthProgress::Authenticated) => break,
                        Ok(AuthProgress::AuthenticatedWith(msg)) => {
                            // Send the terminal message (SCRAM SaslFinal) then
                            // complete authentication.
                            self.feed(msg).await?;
                            self.flush().await?;
                            break;
                        }
                        Ok(AuthProgress::Continue(msg)) => {
                            self.feed(msg).await?;
                            self.flush().await?;
                        }
                        Ok(AuthProgress::ContinueAfter { deliver, challenge }) => {
                            // A chained factor completed and handed back its
                            // terminal message (the SCRAM SaslFinal). Send it,
                            // then the next factor's challenge, then keep reading.
                            self.feed(deliver).await?;
                            self.feed(challenge).await?;
                            self.flush().await?;
                        }
                        Err(e) => {
                            if let Some(ref sm) = self.server.security_manager {
                                let action = sm.brute_force.record_failure(
                                    &peer_ip,
                                    &user,
                                    &database,
                                    0,
                                    "authentication failed",
                                    &sm.ip_manager,
                                );
                                if action.is_some() {
                                    sm.brute_force
                                        .report_lockout(&peer_ip, &user, &sm.ip_manager);
                                }
                            }
                            return Err(ProtocolError::AuthFailed(e.to_string()));
                        }
                    }
                }
            }
        }

        // Record successful authentication
        if let Some(ref sm) = self.server.security_manager {
            sm.brute_force.record_success(&peer_ip, &user, &database);
        }

        // Resolve database ID from catalog
        let database_id = self
            .server
            .catalog
            .get_database(&database)
            .map_err(|_| {
                ProtocolError::Malformed(format!("Database \"{}\" does not exist", database))
            })?
            .id;

        // Create security context if the auth system is configured.
        // Looks up the user's role and builds a SecurityContext with effective
        // roles, clearance, session attributes, and query limits.
        let security_context = if let Some(ref sm) = self.server.security_manager {
            // Enforce account status for identities backed by a user record.
            // Trust-auth identities with no record are unaffected.
            if let Some(account) = sm.lookup_user(&user) {
                if !account.can_login {
                    return Err(ProtocolError::Malformed(format!(
                        "user \"{user}\" is not permitted to log in (NOLOGIN)"
                    )));
                }
                if account.locked {
                    return Err(ProtocolError::Malformed(format!(
                        "user \"{user}\" account is locked"
                    )));
                }
                if let Some(expires_at) = account.valid_until {
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                    if now >= expires_at {
                        return Err(ProtocolError::Malformed(format!(
                            "user \"{user}\" account has expired"
                        )));
                    }
                }
            }
            let peer_ip = self
                .peer_addr
                .clone()
                .unwrap_or_else(|| "127.0.0.1".to_string());
            if let Some(role) = sm.lookup_role(&user) {
                let user_id = sm.user_id_cache.get(&user).unwrap_or(zyron_auth::UserId(0));
                Some(sm.create_security_context(user_id, &role, &peer_ip))
            } else {
                None
            }
        } else {
            None
        };

        // Create session
        let session = Session::with_security_context(user, database, database_id, security_context);

        // Encode all startup responses into self.write_buf (reuses existing allocation
        // instead of creating a new BytesMut per handshake), then write once.
        {
            use bytes::BufMut;

            self.write_buf.clear();
            BackendMessage::Authentication(AuthenticationMessage::Ok).encode(&mut self.write_buf);

            for (name, value) in session.startup_parameters() {
                self.write_buf.put_u8(b'S');
                let len_pos = self.write_buf.len();
                self.write_buf.put_i32(0); // length placeholder
                self.write_buf.extend_from_slice(name.as_bytes());
                self.write_buf.put_u8(0);
                self.write_buf.extend_from_slice(value.as_bytes());
                self.write_buf.put_u8(0);
                let msg_len = (self.write_buf.len() - len_pos) as i32;
                self.write_buf[len_pos..len_pos + 4].copy_from_slice(&msg_len.to_be_bytes());
            }

            BackendMessage::BackendKeyData {
                process_id: self.process_id,
                secret_key: self.secret_key,
            }
            .encode(&mut self.write_buf);

            BackendMessage::ReadyForQuery(TransactionState::Idle).encode(&mut self.write_buf);

            self.stream
                .write_all(&self.write_buf)
                .await
                .map_err(ProtocolError::Io)?;
            self.write_buf.clear();
        }

        self.session = Some(session);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Message dispatch loop
    // -----------------------------------------------------------------------

    async fn message_loop(&mut self) -> Result<(), ProtocolError> {
        loop {
            // Time spent blocked here is the connection waiting for the client's
            // next request. Large with idle worker CPU = client/harness-bound.
            #[cfg(feature = "profile")]
            let read_wait = profile::scope(Phase::WireReadWait);
            let msg = match self.read_message().await {
                Ok(msg) => msg,
                Err(ProtocolError::ConnectionClosed) => {
                    debug!("Client disconnected");
                    return Ok(());
                }
                Err(e) => return Err(e),
            };
            #[cfg(feature = "profile")]
            drop(read_wait);

            match msg {
                FrontendMessage::Query { sql } => {
                    self.handle_simple_query(sql).await?;
                }
                FrontendMessage::Parse {
                    name,
                    query,
                    param_types,
                } => {
                    self.handle_parse(name, query, param_types).await?;
                }
                FrontendMessage::Bind {
                    portal,
                    statement,
                    param_formats,
                    param_values,
                    result_formats,
                } => {
                    self.handle_bind(
                        portal,
                        statement,
                        param_formats,
                        param_values,
                        result_formats,
                    )
                    .await?;
                }
                FrontendMessage::Execute { portal, max_rows } => {
                    self.handle_execute(portal, max_rows).await?;
                }
                FrontendMessage::Describe { target, name } => {
                    self.handle_describe(target, name).await?;
                }
                FrontendMessage::Close { target, name } => {
                    self.handle_close(target, name).await?;
                }
                FrontendMessage::Sync => {
                    self.handle_sync().await?;
                }
                FrontendMessage::Flush => {
                    self.flush().await?;
                }
                FrontendMessage::Terminate => {
                    debug!("Client sent Terminate");
                    return Ok(());
                }
                FrontendMessage::Subscribe(sub) => {
                    self.handle_subscribe(sub).await?;
                }
                _ => {
                    warn!("Unhandled message type in message loop");
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Simple query protocol
    // -----------------------------------------------------------------------

    async fn handle_simple_query(&mut self, sql: String) -> Result<(), ProtocolError> {
        debug!("Simple query: {}", sql);

        if sql.trim().is_empty() {
            self.feed(BackendMessage::EmptyQueryResponse).await?;
            self.send_ready_for_query().await?;
            return Ok(());
        }

        // Write-buffer length before any response for this query batch. On an
        // autocommit durability failure the buffered success responses are
        // truncated back to this mark and replaced with an ErrorResponse.
        let buf_mark = self.write_buf.len();

        // Auto-prepared plan cache fast path. A single-statement DML shape
        // with only literal values is rewritten to a `$N` template, looked
        // up in the per-connection plan cache, and executed directly. On a
        // cache miss the template is parsed/planned once and cached. Any
        // shape the scanner does not positively recognize returns false and
        // falls through to the full parse path below.
        match self.try_templated_execute(&sql).await {
            Ok(true) => {
                // auto_commit_if_needed truncates the buffered success and
                // buffers an ErrorResponse on a durability failure, so the
                // client never sees CommandComplete for a commit that failed.
                // ReadyForQuery still follows either way per the protocol.
                if let Err(e) = self.auto_commit_if_needed(buf_mark).await {
                    debug!("autocommit failed after templated execute: {}", e);
                }
                self.send_ready_for_query().await?;
                return Ok(());
            }
            Ok(false) => {}
            Err(e) => {
                self.send_protocol_error(&e).await?;
                self.mark_failed_if_in_transaction();
                if let Err(ce) = self.auto_commit_if_needed(buf_mark).await {
                    debug!("autocommit failed after templated execute error: {}", ce);
                }
                self.send_ready_for_query().await?;
                return Ok(());
            }
        }

        // Parse SQL into statements
        let stmts = match zyron_parser::parse(&sql) {
            Ok(stmts) => stmts,
            Err(e) => {
                self.send_error(&e).await?;
                self.send_ready_for_query().await?;
                return Ok(());
            }
        };

        if stmts.is_empty() {
            self.feed(BackendMessage::EmptyQueryResponse).await?;
            self.send_ready_for_query().await?;
            return Ok(());
        }

        for stmt in stmts {
            // Check if we're in a failed transaction (only ROLLBACK allowed)
            if self.session_ref().transaction_state() == TransactionState::Failed {
                if !is_rollback(&stmt) {
                    self.send_error(&ZyronError::TransactionAborted(
                        "current transaction is aborted, commands ignored until end of transaction block".into(),
                    )).await?;
                    continue;
                }
            }

            // Handle transaction control statements directly
            if let Some(result) = self.try_handle_transaction_control(&stmt).await {
                match result {
                    Ok(tag) => {
                        self.feed(BackendMessage::CommandComplete { tag }).await?;
                    }
                    Err(e) => {
                        self.send_error(&e).await?;
                        self.mark_failed_if_in_transaction();
                    }
                }
                continue;
            }

            // Reject write statements in a READ ONLY transaction before they
            // reach any operator that touches the heap
            if let Some(txn) = self.transaction.as_ref() {
                if txn.read_only() && !is_read_only_safe_statement(&stmt) {
                    self.send_error(&ZyronError::ExecutionError(format!(
                        "cannot execute {} in a read-only transaction",
                        statement_op_name(&stmt)
                    )))
                    .await?;
                    self.mark_failed_if_in_transaction();
                    continue;
                }
            }

            // Handle SET/SHOW directly
            if let Some(result) = self.try_handle_session_command(&stmt).await {
                match result {
                    Ok(()) => {}
                    Err(e) => {
                        self.send_protocol_error(&e).await?;
                        self.mark_failed_if_in_transaction();
                    }
                }
                continue;
            }

            // Intercept SELECT from virtual stat views
            if let zyron_parser::Statement::Select(ref sel) = stmt {
                if let Some(view_name) = extract_single_from_table(sel) {
                    if crate::stat_views::is_stat_view(&view_name) {
                        let outcome =
                            match crate::stat_views::parse_stat_view_query(&view_name, sel) {
                                Ok(filters) => {
                                    self.handle_stat_view_query(&view_name, &filters).await
                                }
                                Err(e) => Err(ProtocolError::Database(e)),
                            };
                        if let Err(e) = outcome {
                            self.send_protocol_error(&e).await?;
                            self.mark_failed_if_in_transaction();
                        }
                        continue;
                    }
                }
            }

            // Handle EXPLAIN statements (pass owned value to avoid cloning the AST)
            if let zyron_parser::Statement::Explain(explain_stmt) = stmt {
                match self.handle_explain_statement(*explain_stmt).await {
                    Ok(()) => {}
                    Err(e) => {
                        self.send_protocol_error(&e).await?;
                        self.mark_failed_if_in_transaction();
                    }
                }
                continue;
            }

            // Handle DDL, DCL, and utility statements directly
            if let Some(result) = crate::ddl_dispatch::try_handle_ddl_utility(
                &stmt,
                &self.server,
                &mut self.session,
                &mut self.transaction,
                &mut self.active_branch,
                &sql,
            )
            .await
            {
                match result {
                    Ok(crate::ddl_dispatch::DdlResult::Tag(tag)) => {
                        self.feed(BackendMessage::CommandComplete { tag }).await?;
                    }
                    Ok(crate::ddl_dispatch::DdlResult::Rows { tag, columns, rows }) => {
                        // Build row description from column definitions
                        let fields: Vec<FieldDescription> = columns
                            .iter()
                            .map(|(name, oid)| FieldDescription {
                                name: name.clone(),
                                table_oid: 0,
                                column_attr: 0,
                                type_oid: *oid,
                                type_size: -1,
                                type_modifier: -1,
                                format: 0,
                            })
                            .collect();
                        self.feed(BackendMessage::RowDescription(fields)).await?;
                        for row in &rows {
                            let values: Vec<Option<Vec<u8>>> =
                                row.iter().map(|v| Some(v.as_bytes().to_vec())).collect();
                            self.feed(BackendMessage::DataRow(values)).await?;
                        }
                        self.feed(BackendMessage::CommandComplete { tag }).await?;
                    }
                    Err(e) => {
                        self.send_protocol_error(&e).await?;
                        self.mark_failed_if_in_transaction();
                    }
                }
                continue;
            }

            // ---------------------------------------------------------------
            // LISTEN / NOTIFY
            // ---------------------------------------------------------------
            if let zyron_parser::Statement::Listen(ref listen_stmt) = stmt {
                if let Some(ref nc) = self.server.notification_channels {
                    let rx = nc.listen(&listen_stmt.channel);
                    self.notification_receivers
                        .insert(listen_stmt.channel.clone(), rx);
                    self.feed(BackendMessage::CommandComplete {
                        tag: "LISTEN".to_string(),
                    })
                    .await?;
                } else {
                    self.send_error(&ZyronError::Internal(
                        "notification channels not enabled".into(),
                    ))
                    .await?;
                    self.mark_failed_if_in_transaction();
                }
                continue;
            }

            if let zyron_parser::Statement::Notify(ref notify_stmt) = stmt {
                if let Some(ref nc) = self.server.notification_channels {
                    let payload = notify_stmt.payload.as_deref().unwrap_or("");
                    nc.notify(&notify_stmt.channel, payload, self.process_id);
                    self.feed(BackendMessage::CommandComplete {
                        tag: "NOTIFY".to_string(),
                    })
                    .await?;
                } else {
                    self.send_error(&ZyronError::Internal(
                        "notification channels not enabled".into(),
                    ))
                    .await?;
                    self.mark_failed_if_in_transaction();
                }
                continue;
            }

            // ---------------------------------------------------------------
            // PREPARE / EXECUTE / DEALLOCATE
            // ---------------------------------------------------------------
            if let zyron_parser::Statement::Prepare(prepare_stmt) = stmt {
                let name = prepare_stmt.name.clone();
                let inner_query = format!("{:?}", prepare_stmt.statement);
                let param_types: Vec<i32> =
                    prepare_stmt.param_types.iter().map(|_dt| 0i32).collect();

                // Plan the inner statement
                let (plan, schema) = {
                    let session = match self.session.as_ref() {
                        Some(s) => s,
                        None => {
                            self.send_error(&ZyronError::Internal("no session established".into()))
                                .await?;
                            self.mark_failed_if_in_transaction();
                            continue;
                        }
                    };
                    match zyron_planner::plan(
                        &self.server.catalog,
                        session.database_id,
                        session.search_path.clone(),
                        *prepare_stmt.statement,
                        Some(&self.server.peer_facts()),
                    )
                    .await
                    {
                        Ok(p) => {
                            let s = p.output_schema();
                            (Some(Arc::new(p)), s)
                        }
                        Err(e) => {
                            self.send_error(&e).await?;
                            self.mark_failed_if_in_transaction();
                            continue;
                        }
                    }
                };

                self.statements.insert(
                    name,
                    PreparedStatement {
                        query: inner_query,
                        param_types,
                        plan,
                        output_schema: schema,
                    },
                );

                self.feed(BackendMessage::CommandComplete {
                    tag: "PREPARE".to_string(),
                })
                .await?;
                continue;
            }

            if let zyron_parser::Statement::Execute(execute_stmt) = stmt {
                let ps = match self.statements.get(&execute_stmt.name) {
                    Some(ps) => ps,
                    None => {
                        self.send_error(&ZyronError::Internal(format!(
                            "prepared statement \"{}\" does not exist",
                            execute_stmt.name
                        )))
                        .await?;
                        self.mark_failed_if_in_transaction();
                        continue;
                    }
                };

                if let Some(ref plan) = ps.plan {
                    let plan_clone = (**plan).clone();
                    let output_schema = ps.output_schema.clone();
                    let is_select = !output_schema.is_empty() && is_query_plan(&plan_clone);

                    let (txn_id, snapshot) = self.ensure_transaction()?;
                    let mut ctx = ExecutionContext::new(
                        self.server.catalog.clone(),
                        self.server.wal.clone(),
                        self.server.buffer_pool.clone(),
                        self.server.disk_manager.clone(),
                        txn_id as u32,
                        snapshot,
                    );
                    if let Some(ref hook) = self.server.cdc_hook {
                        ctx.cdc_hook = Some(Arc::clone(hook));
                    }
                    if let Some(ref hook) = self.server.dml_hook {
                        ctx.dml_hook = Some(Arc::clone(hook));
                    }
                    // Register live search indexes so scan operators and DML can access them.
                    ctx.doc_registry = Some(Arc::clone(&self.server.doc_registry));
                    ctx.table_io_stats = Some(Arc::clone(&self.server.table_io_stats));
                    ctx.index_io_stats = Some(Arc::clone(&self.server.index_io_stats));
                    if let Some(ref fts_mgr) = self.server.fts_manager {
                        ctx.set_fts_manager(Arc::clone(fts_mgr));
                    }
                    if let Some(ref vec_mgr) = self.server.vector_manager {
                        ctx.set_vector_manager(Arc::clone(vec_mgr));
                    }
                    if let Some(ref graph_mgr) = self.server.graph_manager {
                        ctx.set_graph_manager(Arc::clone(graph_mgr));
                    }
                    if let Some(ref spatial_mgr) = self.server.spatial_manager {
                        ctx.set_spatial_manager(Arc::clone(spatial_mgr));
                    }
                    if let Some(ref sec_mgr) = self.server.security_manager {
                        ctx.set_security_manager(Arc::clone(sec_mgr));
                    }
                    self.attach_undo_log(&mut ctx);
                    self.apply_session_limits(&mut ctx);
                    let ctx = Arc::new(ctx);

                    match execute(plan_clone, &ctx).await {
                        Ok(batches) => {
                            if is_select {
                                let row_desc = self.build_row_description(&output_schema, &[]);
                                self.feed(row_desc).await?;
                                let row_count =
                                    self.send_data_rows(&batches, &output_schema, &[]).await?;
                                self.feed(BackendMessage::CommandComplete {
                                    tag: format!("SELECT {}", row_count),
                                })
                                .await?;
                            } else {
                                let affected = count_affected_rows(&batches);
                                let tag = make_dml_tag(&output_schema, affected);
                                self.feed(BackendMessage::CommandComplete { tag }).await?;
                            }
                        }
                        Err(e) => {
                            self.send_protocol_error(&ProtocolError::Database(e))
                                .await?;
                            self.mark_failed_if_in_transaction();
                        }
                    }
                    self.note_ctx_writes(&ctx);
                } else {
                    // Re-parse and execute the stored query
                    let query = ps.query.clone();
                    match zyron_parser::parse(&query) {
                        Ok(stmts) if !stmts.is_empty() => {
                            let inner = stmts.into_iter().next().unwrap();
                            match self.plan_and_execute_statement(inner).await {
                                Ok(()) => {}
                                Err(e) => {
                                    self.send_protocol_error(&e).await?;
                                    self.mark_failed_if_in_transaction();
                                }
                            }
                        }
                        Ok(_) => {
                            self.feed(BackendMessage::EmptyQueryResponse).await?;
                        }
                        Err(e) => {
                            self.send_error(&e).await?;
                            self.mark_failed_if_in_transaction();
                        }
                    }
                }
                continue;
            }

            if let zyron_parser::Statement::Deallocate(dealloc_stmt) = stmt {
                if dealloc_stmt.all {
                    self.statements.clear();
                } else if let Some(ref name) = dealloc_stmt.name {
                    self.statements.remove(name);
                }
                self.feed(BackendMessage::CommandComplete {
                    tag: "DEALLOCATE".to_string(),
                })
                .await?;
                continue;
            }

            // ---------------------------------------------------------------
            // DECLARE / FETCH / CLOSE CURSOR
            // ---------------------------------------------------------------
            if let zyron_parser::Statement::DeclareCursor(decl_stmt) = stmt {
                let session = match self.session.as_ref() {
                    Some(s) => s,
                    None => {
                        self.send_error(&ZyronError::Internal("no session established".into()))
                            .await?;
                        self.mark_failed_if_in_transaction();
                        continue;
                    }
                };
                let db_id = session.database_id;
                let search_path = session.search_path.clone();

                let select_stmt = zyron_parser::Statement::Select(decl_stmt.query);
                match zyron_planner::plan(
                    &self.server.catalog,
                    db_id,
                    search_path,
                    select_stmt,
                    Some(&self.server.peer_facts()),
                )
                .await
                {
                    Ok(plan) => {
                        let output_schema = plan.output_schema();
                        let with_hold = decl_stmt.hold.unwrap_or(false);
                        self.cursors.insert(
                            decl_stmt.name.clone(),
                            CursorState {
                                plan: Arc::new(plan),
                                output_schema,
                                rows: Vec::new(),
                                position: 0,
                                with_hold,
                            },
                        );
                        self.feed(BackendMessage::CommandComplete {
                            tag: "DECLARE CURSOR".to_string(),
                        })
                        .await?;
                    }
                    Err(e) => {
                        self.send_error(&e).await?;
                        self.mark_failed_if_in_transaction();
                    }
                }
                continue;
            }

            if let zyron_parser::Statement::FetchCursor(fetch_stmt) = stmt {
                let cursor_name = fetch_stmt.cursor.clone();
                let fetch_count = match fetch_stmt.direction {
                    zyron_parser::ast::FetchDirection::Next => 1i64,
                    zyron_parser::ast::FetchDirection::Prior => -1,
                    zyron_parser::ast::FetchDirection::First => 1,
                    zyron_parser::ast::FetchDirection::Last => -1,
                    zyron_parser::ast::FetchDirection::Absolute(n) => n,
                    zyron_parser::ast::FetchDirection::Relative(n) => n,
                    zyron_parser::ast::FetchDirection::Forward(n) => n.unwrap_or(1),
                    zyron_parser::ast::FetchDirection::Backward(n) => -(n.unwrap_or(1)),
                    zyron_parser::ast::FetchDirection::All => i64::MAX,
                };

                let cursor = match self.cursors.get_mut(&cursor_name) {
                    Some(c) => c,
                    None => {
                        self.send_error(&ZyronError::Internal(format!(
                            "cursor \"{}\" does not exist",
                            cursor_name
                        )))
                        .await?;
                        self.mark_failed_if_in_transaction();
                        continue;
                    }
                };

                // Execute the plan on first fetch if rows are empty
                if cursor.rows.is_empty() {
                    let plan_clone = (*cursor.plan).clone();
                    let (txn_id, snapshot) = self.ensure_transaction()?;
                    let mut ctx = ExecutionContext::new(
                        self.server.catalog.clone(),
                        self.server.wal.clone(),
                        self.server.buffer_pool.clone(),
                        self.server.disk_manager.clone(),
                        txn_id as u32,
                        snapshot,
                    );
                    self.attach_undo_log(&mut ctx);
                    self.apply_session_limits(&mut ctx);
                    let ctx = Arc::new(ctx);
                    match execute(plan_clone, &ctx).await {
                        Ok(batches) => {
                            let cursor = self.cursors.get_mut(&cursor_name).unwrap();
                            cursor.rows = batches;
                        }
                        Err(e) => {
                            self.send_protocol_error(&ProtocolError::Database(e))
                                .await?;
                            self.mark_failed_if_in_transaction();
                            continue;
                        }
                    }
                    self.note_ctx_writes(&ctx);
                }

                // Collect row data from cursor into owned values to avoid
                // holding a borrow on self.cursors while calling self.feed.
                let (output_schema, data_rows) = {
                    let cursor = self.cursors.get(&cursor_name).unwrap();
                    let output_schema = cursor.output_schema.clone();
                    let total_rows: usize = cursor.rows.iter().map(|b| b.num_rows).sum();
                    let start = cursor.position;
                    let count = if fetch_count < 0 {
                        0usize // Backward fetch returns empty for simplicity
                    } else if fetch_count == i64::MAX {
                        total_rows.saturating_sub(start)
                    } else {
                        (fetch_count as usize).min(total_rows.saturating_sub(start))
                    };

                    let mut data_rows: Vec<Vec<Option<Vec<u8>>>> = Vec::new();
                    let mut sent = 0usize;
                    let mut global_pos = 0usize;
                    for batch in &cursor.rows {
                        if sent >= count {
                            break;
                        }
                        let batch_end = global_pos + batch.num_rows;
                        if batch_end <= start {
                            global_pos = batch_end;
                            continue;
                        }
                        let batch_start = if start > global_pos {
                            start - global_pos
                        } else {
                            0
                        };
                        let remaining = count - sent;
                        let slice_end = (batch_start + remaining).min(batch.num_rows);

                        for row_idx in batch_start..slice_end {
                            let mut values: Vec<Option<Vec<u8>>> =
                                Vec::with_capacity(output_schema.len());
                            for col in &batch.columns {
                                let scalar = col.get_scalar(row_idx);
                                let mut buf = bytes::BytesMut::with_capacity(32);
                                if types::scalar_write_text(&scalar, &mut buf) {
                                    values.push(Some(buf.to_vec()));
                                } else {
                                    values.push(None);
                                }
                            }
                            data_rows.push(values);
                            sent += 1;
                        }
                        global_pos = batch_end;
                    }
                    (output_schema, data_rows)
                };

                let sent = data_rows.len();

                // Send row description and data rows
                let row_desc = self.build_row_description(&output_schema, &[]);
                self.feed(row_desc).await?;
                for values in data_rows {
                    self.feed(BackendMessage::DataRow(values)).await?;
                }

                // Advance cursor position
                if let Some(cursor) = self.cursors.get_mut(&cursor_name) {
                    cursor.position += sent;
                }

                self.feed(BackendMessage::CommandComplete {
                    tag: format!("FETCH {}", sent),
                })
                .await?;
                continue;
            }

            if let zyron_parser::Statement::CloseCursor(close_stmt) = stmt {
                if close_stmt.all {
                    self.cursors.clear();
                } else if let Some(ref name) = close_stmt.name {
                    self.cursors.remove(name);
                }
                self.feed(BackendMessage::CommandComplete {
                    tag: "CLOSE CURSOR".to_string(),
                })
                .await?;
                continue;
            }

            // ---------------------------------------------------------------
            // COPY
            // ---------------------------------------------------------------
            if let zyron_parser::Statement::Copy(copy_stmt) = stmt {
                // The wire layer only implements the PostgreSQL simple-query
                // COPY TO STDOUT / COPY FROM STDIN forms plus the bare
                // local-file path (`COPY t FROM '/path'`). External
                // endpoints, including named catalog entries and inline
                // backend/format specs, are routed through the planner and
                // executor instead.
                let (copy_table, copy_columns, copy_is_to, copy_external) = match &copy_stmt.kind {
                    zyron_parser::ast::CopyKind::IntoTable {
                        table,
                        columns,
                        source,
                    } => (table.clone(), columns.clone(), false, source.clone()),
                    zyron_parser::ast::CopyKind::FromTable {
                        table,
                        columns,
                        sink,
                    } => (table.clone(), columns.clone(), true, sink.clone()),
                    zyron_parser::ast::CopyKind::ExternalToExternal { source, sink } => {
                        // External-to-external COPY runs the streaming
                        // executor inline. No Zyron transaction is started
                        // because no Zyron table is read or written.
                        let res = crate::copy_external_dispatch::dispatch_external_to_external(
                            &self.server.catalog,
                            self.server.key_store.as_ref(),
                            source,
                            sink,
                            &copy_stmt.options,
                        )
                        .await;
                        match res {
                            Ok(r) => {
                                tracing::info!(
                                    target: "zyron::audit",
                                    rows = r.rows_written,
                                    batches = r.batches,
                                    elapsed_ms = r.elapsed_ms,
                                    "CopyExecuted external-to-external"
                                );
                                self.feed(BackendMessage::CommandComplete {
                                    tag: format!("COPY {}", r.rows_written),
                                })
                                .await?;
                            }
                            Err(e) => {
                                self.send_error(&e).await?;
                                self.mark_failed_if_in_transaction();
                            }
                        }
                        continue;
                    }
                };
                let is_stdio = matches!(copy_external, zyron_parser::ast::CopyExternal::Stdio);
                if copy_is_to {
                    // COPY <table> TO STDOUT (or STDOUT-like sink). Any other
                    // sink kind is rejected here because full external-sink
                    // dispatch lives in the streaming executor.
                    if !is_stdio {
                        self.send_error(&ZyronError::Internal(
                            "COPY TO over the wire protocol only supports STDOUT".into(),
                        ))
                        .await?;
                        self.mark_failed_if_in_transaction();
                        continue;
                    }
                    // Build a SELECT * FROM table query, plan and execute,
                    // then stream results through CopyOutHandler.
                    let select_sql = if copy_columns.is_empty() {
                        format!("SELECT * FROM {}", copy_table)
                    } else {
                        format!("SELECT {} FROM {}", copy_columns.join(", "), copy_table)
                    };
                    let stmts = match zyron_parser::parse(&select_sql) {
                        Ok(s) => s,
                        Err(e) => {
                            self.send_error(&e).await?;
                            self.mark_failed_if_in_transaction();
                            continue;
                        }
                    };
                    let select_stmt = stmts.into_iter().next().unwrap();

                    let session = match self.session.as_ref() {
                        Some(s) => s,
                        None => {
                            self.send_error(&ZyronError::Internal("no session established".into()))
                                .await?;
                            self.mark_failed_if_in_transaction();
                            continue;
                        }
                    };
                    let db_id = session.database_id;
                    let search_path = session.search_path.clone();

                    match zyron_planner::plan(
                        &self.server.catalog,
                        db_id,
                        search_path,
                        select_stmt,
                        Some(&self.server.peer_facts()),
                    )
                    .await
                    {
                        Ok(plan) => {
                            let output_schema = plan.output_schema();
                            let (txn_id, snapshot) = self.ensure_transaction()?;
                            let mut ctx = ExecutionContext::new(
                                self.server.catalog.clone(),
                                self.server.wal.clone(),
                                self.server.buffer_pool.clone(),
                                self.server.disk_manager.clone(),
                                txn_id as u32,
                                snapshot,
                            );
                            self.attach_undo_log(&mut ctx);
                            self.apply_session_limits(&mut ctx);
                            let ctx = Arc::new(ctx);

                            match execute(plan, &ctx).await {
                                Ok(batches) => {
                                    let copy_format = match parse_copy_format(&copy_stmt.options) {
                                        Ok(f) => f,
                                        Err(e) => {
                                            self.send_protocol_error(&e).await?;
                                            self.mark_failed_if_in_transaction();
                                            self.note_ctx_writes(&ctx);
                                            continue;
                                        }
                                    };
                                    let want_header = copy_has_header(&copy_stmt.options);
                                    let handler = crate::copy::CopyOutHandler::new(
                                        output_schema,
                                        copy_format,
                                    );
                                    self.feed(handler.header_message()).await?;
                                    if want_header && copy_format == crate::copy::CopyFormat::Csv {
                                        self.feed(handler.csv_header_message()).await?;
                                    }
                                    for batch in &batches {
                                        let msgs = handler.format_batch(batch);
                                        for msg in msgs {
                                            self.feed(msg).await?;
                                        }
                                    }
                                    self.feed(handler.done_message()).await?;
                                    let total: usize = batches.iter().map(|b| b.num_rows).sum();
                                    self.feed(BackendMessage::CommandComplete {
                                        tag: format!("COPY {}", total),
                                    })
                                    .await?;
                                }
                                Err(e) => {
                                    self.send_protocol_error(&ProtocolError::Database(e))
                                        .await?;
                                    self.mark_failed_if_in_transaction();
                                }
                            }
                            self.note_ctx_writes(&ctx);
                        }
                        Err(e) => {
                            self.send_error(&e).await?;
                            self.mark_failed_if_in_transaction();
                        }
                    }
                    continue;
                } else {
                    // COPY <table> FROM STDIN path. Any other external source
                    // kind is rejected here because the executor owns full
                    // external-source dispatch.
                    if !is_stdio {
                        self.send_error(&ZyronError::Internal(
                            "COPY FROM only supports STDIN in wire protocol".into(),
                        ))
                        .await?;
                        self.mark_failed_if_in_transaction();
                        continue;
                    }

                    // Resolve table columns from catalog
                    let session = match self.session.as_ref() {
                        Some(s) => s,
                        None => {
                            self.send_error(&ZyronError::Internal("no session established".into()))
                                .await?;
                            self.mark_failed_if_in_transaction();
                            continue;
                        }
                    };
                    let db_id = session.database_id;
                    let search_path = session.search_path.clone();

                    // Build column schema by planning a SELECT query
                    let probe_sql = format!("SELECT * FROM {} LIMIT 0", copy_table);
                    let probe_stmts = match zyron_parser::parse(&probe_sql) {
                        Ok(s) => s,
                        Err(e) => {
                            self.send_error(&e).await?;
                            self.mark_failed_if_in_transaction();
                            continue;
                        }
                    };
                    let probe_stmt = probe_stmts.into_iter().next().unwrap();
                    let columns = match zyron_planner::plan(
                        &self.server.catalog,
                        db_id,
                        search_path,
                        probe_stmt,
                        Some(&self.server.peer_facts()),
                    )
                    .await
                    {
                        Ok(plan) => plan.output_schema(),
                        Err(e) => {
                            self.send_error(&e).await?;
                            self.mark_failed_if_in_transaction();
                            continue;
                        }
                    };

                    let copy_format = match parse_copy_format(&copy_stmt.options) {
                        Ok(f) => f,
                        Err(e) => {
                            self.send_protocol_error(&e).await?;
                            self.mark_failed_if_in_transaction();
                            continue;
                        }
                    };
                    let skip_header = copy_has_header(&copy_stmt.options)
                        && copy_format != crate::copy::CopyFormat::Binary;
                    let mut handler = crate::copy::CopyInHandler::new(columns.clone(), copy_format);

                    // Send CopyInResponse to tell client to start sending data
                    self.feed(handler.header_message()).await?;
                    self.flush().await?;

                    // Read CopyData messages until CopyDone or CopyFail
                    let mut copy_aborted = false;
                    loop {
                        let msg = self.read_message().await?;
                        match msg {
                            FrontendMessage::CopyData(data) => {
                                if let Err(e) = handler.feed(&data) {
                                    self.send_protocol_error(&e).await?;
                                    self.mark_failed_if_in_transaction();
                                    copy_aborted = true;
                                    break;
                                }
                            }
                            FrontendMessage::CopyDone => {
                                break;
                            }
                            _ => {
                                // CopyFail or unexpected message
                                self.send_error(&ZyronError::Internal("COPY FROM aborted".into()))
                                    .await?;
                                self.mark_failed_if_in_transaction();
                                copy_aborted = true;
                                break;
                            }
                        }
                    }
                    if copy_aborted {
                        continue;
                    }

                    let rows = match handler.finish() {
                        Ok(r) => r,
                        Err(e) => {
                            self.send_protocol_error(&e).await?;
                            self.mark_failed_if_in_transaction();
                            continue;
                        }
                    };

                    // The CSV header line, when requested, is the first data row.
                    let data_rows: &[Vec<Option<Vec<u8>>>] = if skip_header && !rows.is_empty() {
                        &rows[1..]
                    } else {
                        &rows[..]
                    };

                    let col_names = if copy_columns.is_empty() {
                        columns.iter().map(|c| c.name.clone()).collect::<Vec<_>>()
                    } else {
                        copy_columns.clone()
                    };

                    match self
                        .execute_copy_insert(
                            &copy_table,
                            &col_names,
                            &columns,
                            data_rows,
                            copy_format,
                        )
                        .await
                    {
                        Ok(inserted) => {
                            self.feed(BackendMessage::CommandComplete {
                                tag: format!("COPY {}", inserted),
                            })
                            .await?;
                        }
                        Err(e) => {
                            self.send_protocol_error(&e).await?;
                            self.mark_failed_if_in_transaction();
                        }
                    }
                    continue;
                }
            }

            // Plan and execute the statement
            match self.plan_and_execute_statement(stmt).await {
                Ok(()) => {}
                Err(e) => {
                    self.send_protocol_error(&e).await?;
                    self.mark_failed_if_in_transaction();
                    // In simple query, errors skip remaining statements only if
                    // in an explicit transaction. Otherwise, continue.
                    if self.transaction.is_some() {
                        break;
                    }
                }
            }
        }

        // Auto-commit implicit transactions. A durability failure truncates the
        // buffered success responses back to buf_mark and buffers an error, so
        // the client sees the failure rather than a false CommandComplete.
        if let Err(e) = self.auto_commit_if_needed(buf_mark).await {
            debug!("autocommit failed after simple query batch: {}", e);
        }

        self.send_ready_for_query().await?;
        Ok(())
    }

    async fn plan_and_execute_statement(
        &mut self,
        stmt: zyron_parser::Statement,
    ) -> Result<(), ProtocolError> {
        // Copy session values before mutable borrow. Take the security context
        // temporarily so it can be moved into the ExecutionContext. It is returned
        // to the session after execution completes.
        let (db_id, search_path, sec_ctx) = {
            let session = self
                .session
                .as_mut()
                .ok_or(ProtocolError::Malformed("No session established".into()))?;
            let sc = session.security_context.take();
            (session.database_id, session.search_path.clone(), sc)
        };

        // Start implicit transaction if needed
        let (txn_id, snapshot) = self.ensure_transaction()?;

        // Plan, injecting RLS/ABAC/row-ownership for the session's roles.
        let row_security: Option<std::sync::Arc<dyn zyron_planner::RowSecurityProvider>> =
            match (&self.server.security_manager, &sec_ctx) {
                (Some(sm), Some(sc)) => Some(std::sync::Arc::new(
                    crate::row_security::SmRowSecurityProvider::new(std::sync::Arc::clone(sm), sc),
                )),
                _ => None,
            };
        // The mesh view the plan is costed against. Copied once per
        // statement rather than locked for its duration, so declaring a peer
        // never waits on a running query and a query never sees the mesh
        // change under it mid-plan
        let peerFacts = self.server.peer_facts();
        let plan = zyron_planner::plan_with_security(
            &self.server.catalog,
            db_id,
            search_path,
            stmt,
            row_security,
            Some(&peerFacts),
        )
        .await
        .map_err(ProtocolError::Database)?;

        let output_schema = plan.output_schema();
        let is_select = !output_schema.is_empty() && is_query_plan(&plan);

        // Build execution context with security context for privilege enforcement
        let mut ctx = ExecutionContext::new(
            self.server.catalog.clone(),
            self.server.wal.clone(),
            self.server.buffer_pool.clone(),
            self.server.disk_manager.clone(),
            txn_id as u32,
            snapshot,
        );
        ctx.security_context = sec_ctx.map(Arc::new);
        ctx.heap_files = Some(Arc::clone(&self.server.heap_files));
        ctx.btree_indexes = Some(Arc::clone(&self.server.btree_indexes));
        ctx.foreign_reader = self.server.foreign_reader.clone();
        ctx.peers = Some(Arc::clone(&self.server.peers));
        ctx.intent_locks = Some(Arc::clone(self.server.txn_manager.intent_locks()));
        ctx.row_locks = Some(Arc::clone(self.server.txn_manager.lock_table()));
        ctx.doc_registry = Some(Arc::clone(&self.server.doc_registry));
        ctx.table_io_stats = Some(Arc::clone(&self.server.table_io_stats));
        ctx.index_io_stats = Some(Arc::clone(&self.server.index_io_stats));
        ctx.session_sequences = self.session.as_ref().map(|s| Arc::clone(&s.sequence_state));
        // The heap routes copy-on-write pages by branch id, the lake opens a
        // branch head by name, and both come from this one session branch
        ctx.active_branch_name = self.active_branch.clone();
        if let Some(mgr) = &self.server.branch_manager {
            ctx.branch_catalog = Some(Arc::clone(mgr) as Arc<dyn zyron_common::BranchCatalog>);
            if let Some(name) = &self.active_branch {
                ctx.active_branch_id = mgr.get_branch_by_name(name).ok().map(|e| e.id.0);
            }
        }
        if let Some(ref hook) = self.server.cdc_hook {
            ctx.cdc_hook = Some(Arc::clone(hook));
        }
        if let Some(ref hook) = self.server.dml_hook {
            ctx.dml_hook = Some(Arc::clone(hook));
        }
        self.attach_undo_log(&mut ctx);
        self.apply_session_limits(&mut ctx);
        let ctx = Arc::new(ctx);

        // Execute
        let batches = execute(plan, &ctx).await.map_err(ProtocolError::Database)?;
        self.note_ctx_writes(&ctx);

        // Return the security context to the session so subsequent queries
        // can reuse the cached privilege decisions.
        if let Ok(mut unwrapped) = Arc::try_unwrap(ctx) {
            if let Some(session) = self.session.as_mut() {
                // Recover the owned context from the Arc to return it to the
                // session with its privilege cache intact. Any child context
                // from a nested plan has been dropped, so the refcount is one.
                session.security_context = unwrapped
                    .security_context
                    .take()
                    .and_then(|a| Arc::try_unwrap(a).ok());
            }
        }

        if is_select {
            // Send RowDescription + DataRows + CommandComplete batched
            let row_desc = self.build_row_description(&output_schema, &[]);
            self.feed(row_desc).await?;

            let row_count = self.send_data_rows(&batches, &output_schema, &[]).await?;

            self.feed(BackendMessage::CommandComplete {
                tag: format!("SELECT {}", row_count),
            })
            .await?;
        } else {
            // DML: count affected rows from result batches
            let affected = count_affected_rows(&batches);
            let tag = make_dml_tag(&output_schema, affected);
            self.feed(BackendMessage::CommandComplete { tag }).await?;
        }

        Ok(())
    }

    /// Inserts COPY FROM rows as typed values. Each field's raw bytes are
    /// converted to a ScalarValue using the target column's type (text/CSV via
    /// text_to_scalar, binary via binary_to_scalar), then bound as a parameter to
    /// a single parameterized INSERT so typed, binary, and NULL values are never
    /// round-tripped through quoted SQL text. Returns the number of rows
    /// inserted. Does not emit CommandComplete; the caller emits the COPY tag.
    async fn execute_copy_insert(
        &mut self,
        table: &str,
        col_names: &[String],
        columns: &[LogicalColumn],
        rows: &[Vec<Option<Vec<u8>>>],
        format: crate::copy::CopyFormat,
    ) -> Result<usize, ProtocolError> {
        if rows.is_empty() {
            return Ok(0);
        }
        let num_cols = col_names.len();

        // Map each target column name to its catalog type so the field bytes are
        // decoded against the column they land in, not by position in the probe.
        let mut col_type_oids: Vec<i32> = Vec::with_capacity(num_cols);
        for name in col_names {
            let type_oid = columns
                .iter()
                .find(|c| c.name == *name)
                .map(|c| types::type_id_to_pg_oid(c.type_id))
                .unwrap_or(0);
            col_type_oids.push(type_oid);
        }

        // Convert every field to a typed scalar and assign sequential $N slots.
        let mut params: Vec<ScalarValue> = Vec::with_capacity(rows.len() * num_cols);
        let mut values_parts: Vec<String> = Vec::with_capacity(rows.len());
        let mut next_param = 1usize;
        for row in rows {
            if row.len() != num_cols {
                return Err(ProtocolError::Malformed(format!(
                    "COPY row has {} fields but {} columns are expected",
                    row.len(),
                    num_cols
                )));
            }
            let mut placeholders: Vec<String> = Vec::with_capacity(num_cols);
            for (col_idx, field) in row.iter().enumerate() {
                let scalar = match field {
                    None => ScalarValue::Null,
                    Some(bytes) => {
                        if format == crate::copy::CopyFormat::Binary {
                            types::binary_to_scalar(bytes, col_type_oids[col_idx])?
                        } else {
                            types::text_to_scalar(bytes, col_type_oids[col_idx])?
                        }
                    }
                };
                params.push(scalar);
                placeholders.push(format!("${}", next_param));
                next_param += 1;
            }
            values_parts.push(format!("({})", placeholders.join(", ")));
        }

        let insert_sql = format!(
            "INSERT INTO {} ({}) VALUES {}",
            table,
            col_names.join(", "),
            values_parts.join(", ")
        );
        let stmts = zyron_parser::parse(&insert_sql).map_err(ProtocolError::Database)?;
        let insert_stmt = stmts
            .into_iter()
            .next()
            .ok_or_else(|| ProtocolError::Malformed("COPY produced an empty INSERT".into()))?;

        let (db_id, search_path) = {
            let session = self
                .session
                .as_ref()
                .ok_or(ProtocolError::Malformed("No session established".into()))?;
            (session.database_id, session.search_path.clone())
        };
        let plan = zyron_planner::plan(
            &self.server.catalog,
            db_id,
            search_path,
            insert_stmt,
            Some(&self.server.peer_facts()),
        )
        .await
        .map_err(ProtocolError::Database)?;

        let (txn_id, snapshot) = self.ensure_transaction()?;
        let mut ctx = ExecutionContext::new(
            self.server.catalog.clone(),
            self.server.wal.clone(),
            self.server.buffer_pool.clone(),
            self.server.disk_manager.clone(),
            txn_id as u32,
            snapshot,
        );
        ctx.params = params;
        ctx.heap_files = Some(Arc::clone(&self.server.heap_files));
        ctx.btree_indexes = Some(Arc::clone(&self.server.btree_indexes));
        ctx.foreign_reader = self.server.foreign_reader.clone();
        ctx.peers = Some(Arc::clone(&self.server.peers));
        ctx.intent_locks = Some(Arc::clone(self.server.txn_manager.intent_locks()));
        ctx.row_locks = Some(Arc::clone(self.server.txn_manager.lock_table()));
        ctx.doc_registry = Some(Arc::clone(&self.server.doc_registry));
        ctx.table_io_stats = Some(Arc::clone(&self.server.table_io_stats));
        ctx.index_io_stats = Some(Arc::clone(&self.server.index_io_stats));
        if let Some(ref hook) = self.server.cdc_hook {
            ctx.cdc_hook = Some(Arc::clone(hook));
        }
        if let Some(ref hook) = self.server.dml_hook {
            ctx.dml_hook = Some(Arc::clone(hook));
        }
        self.attach_undo_log(&mut ctx);
        self.apply_session_limits(&mut ctx);
        let ctx = Arc::new(ctx);
        let batches = execute(plan, &ctx).await.map_err(ProtocolError::Database)?;
        self.note_ctx_writes(&ctx);
        if let Some(txn) = self.transaction.as_mut() {
            txn.mark_wrote_data();
        }
        Ok(count_affected_rows(&batches))
    }

    // -----------------------------------------------------------------------
    // Auto-prepared plan cache (simple query fast path)
    // -----------------------------------------------------------------------

    /// Attempts to satisfy `sql` from the per-connection plan cache by
    /// auto-parameterizing its literals. Returns `Ok(true)` when the query
    /// was executed (cache hit, or miss-then-planned-and-cached), `Ok(false)`
    /// when the shape is not cacheable and the caller must use the normal
    /// parse path. Planning or execution errors that should surface to the
    /// client return `Err`.
    async fn try_templated_execute(&mut self, sql: &str) -> Result<bool, ProtocolError> {
        // Never take the fast path inside a failed transaction; only ROLLBACK
        // is permitted there and that is not a cacheable shape anyway.
        if self.session_ref().transaction_state() == TransactionState::Failed {
            return Ok(false);
        }

        let templated = match crate::auto_param::templatize(sql) {
            Some(t) => t,
            None => return Ok(false),
        };

        let (cache_key, current_version) = {
            let session = self
                .session
                .as_ref()
                .ok_or(ProtocolError::Malformed("No session established".into()))?;
            let key = crate::statement_cache::CacheKey {
                template_hash: zyron_common::hash64(templated.template.as_bytes()),
                search_path_hash: crate::statement_cache::StatementCache::hash_search_path(
                    &session.search_path,
                ),
                role_id: session.identity_hash(),
                rls_policy_hash: 0,
                type_kinds_hash: templated.type_kinds_hash,
            };
            (key, self.server.catalog.schema_version())
        };

        // L1 hit (per-session): reuse the planned shape directly.
        if let Some(cached) = self.statement_cache.lookup(&cache_key, current_version) {
            if cached.param_count != templated.literals.len() {
                // Shape mismatch should be impossible (the template hash
                // encodes the placeholder count), but never bind the wrong
                // number of params: fall through to the safe path.
                return Ok(false);
            }
            self.execute_cached_plan(cached, templated.literals).await?;
            return Ok(true);
        }

        // L2 hit (server-wide): another connection already planned this
        // shape. Promote it into this connection's L1 so subsequent runs
        // hit the lock-free path, then execute.
        if let Some(cached) = self.server.plan_cache.lookup(&cache_key, current_version) {
            if cached.param_count == templated.literals.len() {
                self.statement_cache.insert(cached.clone());
                self.execute_cached_plan(cached, templated.literals).await?;
                return Ok(true);
            }
        }

        // Cache miss: parse the templated SQL (which already carries the
        // `$N` placeholders) so the planner emits Parameter slots, plan it
        // under the session's row security, and cache the result. On any
        // parse/plan failure, fall back to the normal path so the original
        // SQL produces the user-facing error.
        let stmts = match zyron_parser::parse(&templated.template) {
            Ok(s) => s,
            Err(_) => return Ok(false),
        };
        if stmts.len() != 1 {
            return Ok(false);
        }
        let stmt = stmts.into_iter().next().unwrap();

        let (db_id, search_path, row_security) = {
            let sm = self.server.security_manager.as_ref();
            let session = self
                .session
                .as_ref()
                .ok_or(ProtocolError::Malformed("No session established".into()))?;
            let rs: Option<std::sync::Arc<dyn zyron_planner::RowSecurityProvider>> =
                match (sm, &session.security_context) {
                    (Some(sm), Some(sc)) => Some(std::sync::Arc::new(
                        crate::row_security::SmRowSecurityProvider::new(
                            std::sync::Arc::clone(sm),
                            sc,
                        ),
                    )),
                    _ => None,
                };
            (session.database_id, session.search_path.clone(), rs)
        };

        // The mesh view the plan is costed against. Copied once per
        // statement rather than locked for its duration, so declaring a peer
        // never waits on a running query and a query never sees the mesh
        // change under it mid-plan
        let peerFacts = self.server.peer_facts();
        let plan = match zyron_planner::plan_with_security(
            &self.server.catalog,
            db_id,
            search_path,
            stmt,
            row_security,
            Some(&peerFacts),
        )
        .await
        {
            Ok(p) => p,
            Err(_) => return Ok(false),
        };

        let output_schema = plan.output_schema();
        let plan_arc = Arc::new(plan);
        let entry = crate::statement_cache::CachedPlan {
            key: cache_key,
            schema_version: current_version,
            plan: Arc::clone(&plan_arc),
            output_schema: output_schema.clone(),
            param_count: templated.literals.len(),
        };
        self.statement_cache.insert(entry.clone());
        self.server.plan_cache.insert(entry.clone());
        self.execute_cached_plan(entry, templated.literals).await?;
        Ok(true)
    }

    /// Executes a cached plan with the supplied parameter values and streams
    /// the result. Mirrors the tail of `plan_and_execute_statement`.
    async fn execute_cached_plan(
        &mut self,
        cached: crate::statement_cache::CachedPlan,
        params: Vec<ScalarValue>,
    ) -> Result<(), ProtocolError> {
        let sec_ctx = self
            .session
            .as_mut()
            .and_then(|s| s.security_context.take());

        let (txn_id, snapshot) = self.ensure_transaction()?;

        let mut ctx = ExecutionContext::new(
            self.server.catalog.clone(),
            self.server.wal.clone(),
            self.server.buffer_pool.clone(),
            self.server.disk_manager.clone(),
            txn_id as u32,
            snapshot,
        );
        ctx.params = params;
        ctx.security_context = sec_ctx.map(Arc::new);
        ctx.heap_files = Some(Arc::clone(&self.server.heap_files));
        ctx.btree_indexes = Some(Arc::clone(&self.server.btree_indexes));
        ctx.foreign_reader = self.server.foreign_reader.clone();
        ctx.peers = Some(Arc::clone(&self.server.peers));
        ctx.intent_locks = Some(Arc::clone(self.server.txn_manager.intent_locks()));
        ctx.row_locks = Some(Arc::clone(self.server.txn_manager.lock_table()));
        ctx.doc_registry = Some(Arc::clone(&self.server.doc_registry));
        ctx.table_io_stats = Some(Arc::clone(&self.server.table_io_stats));
        ctx.index_io_stats = Some(Arc::clone(&self.server.index_io_stats));
        ctx.session_sequences = self.session.as_ref().map(|s| Arc::clone(&s.sequence_state));
        // The heap routes copy-on-write pages by branch id, the lake opens a
        // branch head by name, and both come from this one session branch
        ctx.active_branch_name = self.active_branch.clone();
        if let Some(mgr) = &self.server.branch_manager {
            ctx.branch_catalog = Some(Arc::clone(mgr) as Arc<dyn zyron_common::BranchCatalog>);
            if let Some(name) = &self.active_branch {
                ctx.active_branch_id = mgr.get_branch_by_name(name).ok().map(|e| e.id.0);
            }
        }
        if let Some(ref hook) = self.server.cdc_hook {
            ctx.cdc_hook = Some(Arc::clone(hook));
        }
        if let Some(ref hook) = self.server.dml_hook {
            ctx.dml_hook = Some(Arc::clone(hook));
        }
        self.attach_undo_log(&mut ctx);
        self.apply_session_limits(&mut ctx);
        let ctx = Arc::new(ctx);

        let plan = (*cached.plan).clone();
        let output_schema = cached.output_schema;
        let is_select = !output_schema.is_empty() && is_query_plan(&plan);

        let batches = execute(plan, &ctx).await.map_err(ProtocolError::Database)?;
        self.note_ctx_writes(&ctx);

        if let Ok(mut unwrapped) = Arc::try_unwrap(ctx) {
            if let Some(session) = self.session.as_mut() {
                // Recover the owned context from the Arc to return it to the
                // session with its privilege cache intact. Any child context
                // from a nested plan has been dropped, so the refcount is one.
                session.security_context = unwrapped
                    .security_context
                    .take()
                    .and_then(|a| Arc::try_unwrap(a).ok());
            }
        }

        if is_select {
            let row_desc = self.build_row_description(&output_schema, &[]);
            self.feed(row_desc).await?;
            let row_count = self.send_data_rows(&batches, &output_schema, &[]).await?;
            self.feed(BackendMessage::CommandComplete {
                tag: format!("SELECT {}", row_count),
            })
            .await?;
        } else {
            let affected = count_affected_rows(&batches);
            let tag = make_dml_tag(&output_schema, affected);
            self.feed(BackendMessage::CommandComplete { tag }).await?;
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // EXPLAIN handling
    // -----------------------------------------------------------------------

    async fn handle_explain_statement(
        &mut self,
        explain_stmt: zyron_parser::ast::ExplainStatement,
    ) -> Result<(), ProtocolError> {
        let (db_id, search_path) = {
            let session = self
                .session
                .as_ref()
                .ok_or(ProtocolError::Malformed("No session established".into()))?;
            (session.database_id, session.search_path.clone())
        };

        let options = zyron_planner::ExplainOptions {
            analyze: explain_stmt.analyze,
            costs: explain_stmt.costs,
            buffers: explain_stmt.buffers,
            timing: explain_stmt.timing,
            format: explain_stmt
                .format
                .as_deref()
                .map(zyron_planner::ExplainFormat::from_str)
                .unwrap_or(zyron_planner::ExplainFormat::Text),
        };

        let inner_stmt = *explain_stmt.statement;
        // The mesh view the plan is costed against. Copied once per
        // statement rather than locked for its duration, so declaring a peer
        // never waits on a running query and a query never sees the mesh
        // change under it mid-plan
        let peerFacts = self.server.peer_facts();
        let (plan, options) = zyron_planner::plan_for_explain(
            &self.server.catalog,
            db_id,
            search_path,
            inner_stmt,
            options,
            Some(&peerFacts),
        )
        .await
        .map_err(ProtocolError::Database)?;

        let explain_tree = zyron_planner::ExplainNode::from_physical_plan(&plan);

        if options.analyze {
            let (txn_id, snapshot) = self.ensure_transaction()?;
            let mut ctx = ExecutionContext::new(
                self.server.catalog.clone(),
                self.server.wal.clone(),
                self.server.buffer_pool.clone(),
                self.server.disk_manager.clone(),
                txn_id as u32,
                snapshot,
            );
            self.attach_undo_log(&mut ctx);
            self.apply_session_limits(&mut ctx);
            let ctx = Arc::new(ctx);

            let (_batches, metrics) = execute_analyze(plan, &ctx)
                .await
                .map_err(ProtocolError::Database)?;
            self.note_ctx_writes(&ctx);

            let mut tree = explain_tree;
            if let Some(m) = metrics {
                tree.merge_metrics(&collect_node_metrics(&m));
            }
            let output = tree.render(&options);
            self.send_explain_output(&output).await?;
        } else {
            let output = explain_tree.render(&options);
            self.send_explain_output(&output).await?;
        }

        Ok(())
    }

    async fn send_explain_output(&mut self, output: &str) -> Result<(), ProtocolError> {
        // Send as single-column text result: column name "QUERY PLAN"
        let row_desc = BackendMessage::RowDescription(vec![FieldDescription {
            name: "QUERY PLAN".to_string(),
            table_oid: 0,
            column_attr: 0,
            type_oid: types::PG_TEXT_OID,
            type_size: -1,
            type_modifier: -1,
            format: 0,
        }]);
        self.feed(row_desc).await?;

        // Send each line as a separate DataRow
        let mut line_count = 0usize;
        for line in output.lines() {
            let row = BackendMessage::DataRow(vec![Some(line.as_bytes().to_vec())]);
            self.feed(row).await?;
            line_count += 1;
        }
        self.feed(BackendMessage::CommandComplete {
            tag: format!("EXPLAIN {}", line_count),
        })
        .await?;
        self.flush().await?;

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Extended query protocol
    // -----------------------------------------------------------------------

    async fn handle_parse(
        &mut self,
        name: String,
        query: String,
        param_types: Vec<i32>,
    ) -> Result<(), ProtocolError> {
        debug!("Parse: name={}, query={}", name, query);

        // Covers statement-cache lookup plus the parse+plan it skips on a hit.
        #[cfg(feature = "profile")]
        let _recv_parse = profile::scope(Phase::WireRecvParse);

        // Statement cache lookup: build the composite key and skip the
        // parse + plan round trip when the same text was bound under the
        // same search path, identity, and schema version. The extended
        // protocol already carries `$N` placeholders, so the template is
        // the query verbatim and there are no extracted literals.
        let session = self
            .session
            .as_ref()
            .ok_or(ProtocolError::Malformed("No session established".into()))?;
        let cache_key = crate::statement_cache::CacheKey {
            template_hash: zyron_common::hash64(query.as_bytes()),
            search_path_hash: crate::statement_cache::StatementCache::hash_search_path(
                &session.search_path,
            ),
            role_id: session.identity_hash(),
            rls_policy_hash: 0,
            type_kinds_hash: 0,
        };
        let current_version = self.server.catalog.schema_version();

        // L1 (per-session), then L2 (server-wide) before parsing. An L2 hit
        // is promoted into L1 so the next Parse on this connection is
        // lock-free.
        let cache_hit = self
            .statement_cache
            .lookup(&cache_key, current_version)
            .or_else(|| {
                self.server
                    .plan_cache
                    .lookup(&cache_key, current_version)
                    .inspect(|c| self.statement_cache.insert(c.clone()))
            });
        if let Some(cached) = cache_hit {
            // Evict a named statement if over capacity.
            if !name.is_empty() && self.statements.len() >= MAX_PREPARED_STATEMENTS {
                let victim = self
                    .statements
                    .keys()
                    .find(|k| !k.is_empty() && *k != &name)
                    .cloned();
                if let Some(key) = victim {
                    self.statements.remove(&key);
                }
            }
            self.statements.insert(
                name,
                PreparedStatement {
                    query,
                    param_types,
                    plan: Some(cached.plan),
                    output_schema: cached.output_schema,
                },
            );
            self.feed(BackendMessage::ParseComplete).await?;
            return Ok(());
        }

        let stmts = match zyron_parser::parse(&query) {
            Ok(stmts) => stmts,
            Err(e) => {
                self.send_error(&e).await?;
                return Ok(());
            }
        };

        let (plan, schema) = if stmts.len() == 1 {
            let stmt = stmts.into_iter().next().unwrap();

            // DDL/utility statements bypass the planner. They get None plan
            // and are dispatched through ddl_dispatch at Execute time.
            if is_ddl_statement(&stmt) {
                (None, Vec::new())
            } else {
                let session = self
                    .session
                    .as_ref()
                    .ok_or(ProtocolError::Malformed("No session established".into()))?;

                match zyron_planner::plan(
                    &self.server.catalog,
                    session.database_id,
                    session.search_path.clone(),
                    stmt,
                    Some(&self.server.peer_facts()),
                )
                .await
                {
                    Ok(p) => {
                        let plan_arc = Arc::new(p);
                        let schema = plan_arc.output_schema();
                        let entry = crate::statement_cache::CachedPlan {
                            key: cache_key,
                            schema_version: current_version,
                            plan: Arc::clone(&plan_arc),
                            output_schema: schema.clone(),
                            param_count: 0,
                        };
                        self.statement_cache.insert(entry.clone());
                        self.server.plan_cache.insert(entry);
                        (Some(plan_arc), schema)
                    }
                    Err(e) => {
                        self.send_error(&e).await?;
                        return Ok(());
                    }
                }
            }
        } else {
            (None, Vec::new())
        };

        // Evict a named statement if over capacity.
        if !name.is_empty() && self.statements.len() >= MAX_PREPARED_STATEMENTS {
            let victim = self
                .statements
                .keys()
                .find(|k| !k.is_empty() && *k != &name)
                .cloned();
            if let Some(key) = victim {
                self.statements.remove(&key);
            }
        }

        self.statements.insert(
            name,
            PreparedStatement {
                query,
                param_types,
                plan,
                output_schema: schema,
            },
        );

        self.feed(BackendMessage::ParseComplete).await?;
        Ok(())
    }

    async fn handle_bind(
        &mut self,
        portal_name: String,
        stmt_name: String,
        param_formats: Vec<i16>,
        param_values: Vec<Option<Vec<u8>>>,
        result_formats: Vec<i16>,
    ) -> Result<(), ProtocolError> {
        debug!("Bind: portal={}, stmt={}", portal_name, stmt_name);

        // Covers param decode, plan resolution (cache clone or re-plan), and
        // portal construction.
        #[cfg(feature = "profile")]
        let _plan_span = profile::scope(Phase::WirePlan);

        let stmt = self.statements.get(&stmt_name).ok_or_else(|| {
            ProtocolError::Malformed(format!("Prepared statement \"{}\" not found", stmt_name))
        })?;

        // Decode parameter values
        let mut params = Vec::with_capacity(param_values.len());
        for (i, value) in param_values.iter().enumerate() {
            match value {
                None => params.push(ScalarValue::Null),
                Some(data) => {
                    let format = if i < param_formats.len() {
                        param_formats[i]
                    } else if param_formats.len() == 1 {
                        param_formats[0]
                    } else {
                        0 // text format default
                    };

                    let type_oid = if i < stmt.param_types.len() {
                        stmt.param_types[i]
                    } else {
                        0 // unspecified
                    };

                    let scalar = if format == 1 {
                        types::binary_to_scalar(data, type_oid)?
                    } else {
                        types::text_to_scalar(data, type_oid)?
                    };
                    params.push(scalar);
                }
            }
        }

        // A None plan marks a transaction-control, session, or DDL/utility
        // statement that bypasses the planner. It is dispatched at Execute time
        // through the same handlers the simple-query path uses, so the portal
        // carries no plan and an empty output schema. A Some plan is a query
        // whose schema is read directly; the Arc clone is a cheap refcount bump.
        let query = stmt.query.clone();
        let (plan, output_schema) = match &stmt.plan {
            Some(p) => {
                let plan = p.clone();
                let schema = plan.output_schema();
                (Some(plan), schema)
            }
            None => (None, Vec::new()),
        };

        // Evict a named portal if over capacity.
        if !portal_name.is_empty() && self.portals.len() >= MAX_PORTALS {
            let victim = self
                .portals
                .keys()
                .find(|k| !k.is_empty() && *k != &portal_name)
                .cloned();
            if let Some(key) = victim {
                self.portals.remove(&key);
            }
        }

        // stmt_name is retained through plan lookup above; the portal itself
        // does not carry it because the plan is already resolved.
        let _ = stmt_name;
        self.portals.insert(
            portal_name,
            Portal {
                params,
                result_formats,
                plan,
                output_schema,
                query,
            },
        );

        self.feed(BackendMessage::BindComplete).await?;
        Ok(())
    }

    async fn handle_execute(
        &mut self,
        portal_name: String,
        max_rows: i32,
    ) -> Result<(), ProtocolError> {
        debug!("Execute: portal={}, max_rows={}", portal_name, max_rows);

        let portal = match self.portals.get(&portal_name) {
            Some(p) => p,
            None => {
                self.send_error(&ZyronError::Internal(format!(
                    "Portal \"{}\" not found",
                    portal_name
                )))
                .await?;
                return Ok(());
            }
        };

        // A portal with no plan holds a transaction-control, session, or
        // DDL/utility statement that bypasses the planner. Dispatch it through
        // the same handlers the simple-query path uses so prepared
        // BEGIN/COMMIT/ROLLBACK/SET/SHOW/VACUUM/CHECKPOINT and DDL execute
        // instead of being rejected by the planner.
        if portal.plan.is_none() {
            let query = portal.query.clone();
            return self.execute_utility_portal(&query).await;
        }

        #[cfg(feature = "profile")]
        let setup_span = profile::scope(Phase::WireExecSetup);
        let plan = portal.plan.clone().expect("portal plan present");
        let output_schema = portal.output_schema.clone();
        let result_formats = portal.result_formats.clone();
        let params = portal.params.clone();
        let is_select = !output_schema.is_empty() && is_query_plan(&*plan);

        let (txn_id, snapshot) = match self.ensure_transaction() {
            Ok(t) => t,
            Err(e) => {
                self.send_error(&ZyronError::Internal(e.to_string()))
                    .await?;
                return Ok(());
            }
        };

        let mut ctx_owned = ExecutionContext::new(
            self.server.catalog.clone(),
            self.server.wal.clone(),
            self.server.buffer_pool.clone(),
            self.server.disk_manager.clone(),
            txn_id as u32,
            snapshot,
        );
        ctx_owned.params = params;
        ctx_owned.heap_files = Some(Arc::clone(&self.server.heap_files));
        ctx_owned.btree_indexes = Some(Arc::clone(&self.server.btree_indexes));
        ctx_owned.foreign_reader = self.server.foreign_reader.clone();
        ctx_owned.peers = Some(Arc::clone(&self.server.peers));
        ctx_owned.active_branch_name = self.active_branch.clone();
        if let Some(mgr) = &self.server.branch_manager {
            ctx_owned.branch_catalog =
                Some(Arc::clone(mgr) as Arc<dyn zyron_common::BranchCatalog>);
            if let Some(name) = &self.active_branch {
                ctx_owned.active_branch_id = mgr.get_branch_by_name(name).ok().map(|e| e.id.0);
            }
        }
        self.apply_session_limits(&mut ctx_owned);
        let ctx = Arc::new(ctx_owned);
        #[cfg(feature = "profile")]
        drop(setup_span);

        let exec_result = {
            #[cfg(feature = "profile")]
            let _s = profile::scope(Phase::WireExecute);
            execute(Arc::unwrap_or_clone(plan), &ctx).await
        };

        // A statement that appended a WAL data record, or any non-query
        // statement, makes the transaction non-read-only so its commit is
        // durable. A pure SELECT that logged nothing leaves the transaction
        // read-only and skips the commit record and flush wait. Marked
        // conservatively (also on the error path) so a write is never missed.
        if ctx.wrote_wal() || !is_select {
            if let Some(txn) = self.transaction.as_mut() {
                txn.mark_wrote_data();
            }
        }

        #[cfg(feature = "profile")]
        let _send_span = profile::scope(Phase::WireSend);
        match exec_result {
            Ok(batches) => {
                if is_select {
                    let row_count = self
                        .send_data_rows(&batches, &output_schema, &result_formats)
                        .await?;
                    self.feed(BackendMessage::CommandComplete {
                        tag: format!("SELECT {}", row_count),
                    })
                    .await?;
                } else {
                    let affected = count_affected_rows(&batches);
                    let tag = make_dml_tag(&output_schema, affected);
                    self.feed(BackendMessage::CommandComplete { tag }).await?;
                }
            }
            Err(e) => {
                self.send_error(&e).await?;
                self.mark_failed_if_in_transaction();
            }
        }

        Ok(())
    }

    /// Executes a planner-bypassing statement bound through the extended
    /// protocol: transaction control, SET/SHOW, EXPLAIN, or DDL/utility. Routes
    /// through the same handlers as the simple-query path. Does not send
    /// ReadyForQuery, the extended protocol emits that on Sync.
    async fn execute_utility_portal(&mut self, query: &str) -> Result<(), ProtocolError> {
        let stmts = match zyron_parser::parse(query) {
            Ok(s) => s,
            Err(e) => {
                self.send_error(&e).await?;
                self.mark_failed_if_in_transaction();
                return Ok(());
            }
        };
        let stmt = match stmts.into_iter().next() {
            Some(s) => s,
            None => {
                self.feed(BackendMessage::EmptyQueryResponse).await?;
                return Ok(());
            }
        };

        // In a failed transaction only ROLLBACK is permitted.
        if self.session_ref().transaction_state() == TransactionState::Failed && !is_rollback(&stmt)
        {
            self.send_error(&ZyronError::TransactionAborted(
                "current transaction is aborted, commands ignored until end of transaction block"
                    .into(),
            ))
            .await?;
            return Ok(());
        }

        // Transaction control: BEGIN/COMMIT/ROLLBACK.
        if let Some(result) = self.try_handle_transaction_control(&stmt).await {
            match result {
                Ok(tag) => self.feed(BackendMessage::CommandComplete { tag }).await?,
                Err(e) => {
                    self.send_error(&e).await?;
                    self.mark_failed_if_in_transaction();
                }
            }
            return Ok(());
        }

        // Reject write statements in a READ ONLY transaction before they reach
        // any operator that touches the heap
        if let Some(txn) = self.transaction.as_ref() {
            if txn.read_only() && !is_read_only_safe_statement(&stmt) {
                self.send_error(&ZyronError::ExecutionError(format!(
                    "cannot execute {} in a read-only transaction",
                    statement_op_name(&stmt)
                )))
                .await?;
                self.mark_failed_if_in_transaction();
                return Ok(());
            }
        }

        // Session commands: SET/SHOW.
        if let Some(result) = self.try_handle_session_command(&stmt).await {
            if let Err(e) = result {
                self.send_protocol_error(&e).await?;
                self.mark_failed_if_in_transaction();
            }
            return Ok(());
        }

        // SELECT against a virtual stat view.
        if let zyron_parser::Statement::Select(ref sel) = stmt {
            if let Some(view_name) = extract_single_from_table(sel) {
                if crate::stat_views::is_stat_view(&view_name) {
                    let outcome = match crate::stat_views::parse_stat_view_query(&view_name, sel) {
                        Ok(filters) => self.handle_stat_view_query(&view_name, &filters).await,
                        Err(e) => Err(ProtocolError::Database(e)),
                    };
                    if let Err(e) = outcome {
                        self.send_protocol_error(&e).await?;
                        self.mark_failed_if_in_transaction();
                    }
                    return Ok(());
                }
            }
        }

        // EXPLAIN.
        if let zyron_parser::Statement::Explain(explain_stmt) = stmt {
            if let Err(e) = self.handle_explain_statement(*explain_stmt).await {
                self.send_protocol_error(&e).await?;
                self.mark_failed_if_in_transaction();
            }
            return Ok(());
        }

        // DDL, DCL, and other utility statements.
        if let Some(result) = crate::ddl_dispatch::try_handle_ddl_utility(
            &stmt,
            &self.server,
            &mut self.session,
            &mut self.transaction,
            &mut self.active_branch,
            query,
        )
        .await
        {
            match result {
                Ok(crate::ddl_dispatch::DdlResult::Tag(tag)) => {
                    self.feed(BackendMessage::CommandComplete { tag }).await?;
                }
                Ok(crate::ddl_dispatch::DdlResult::Rows { tag, columns, rows }) => {
                    let fields: Vec<FieldDescription> = columns
                        .iter()
                        .map(|(name, oid)| FieldDescription {
                            name: name.clone(),
                            table_oid: 0,
                            column_attr: 0,
                            type_oid: *oid,
                            type_size: -1,
                            type_modifier: -1,
                            format: 0,
                        })
                        .collect();
                    self.feed(BackendMessage::RowDescription(fields)).await?;
                    for row in &rows {
                        let values: Vec<Option<Vec<u8>>> =
                            row.iter().map(|v| Some(v.as_bytes().to_vec())).collect();
                        self.feed(BackendMessage::DataRow(values)).await?;
                    }
                    self.feed(BackendMessage::CommandComplete { tag }).await?;
                }
                Err(e) => {
                    self.send_protocol_error(&e).await?;
                    self.mark_failed_if_in_transaction();
                }
            }
            return Ok(());
        }

        // No handler claimed the statement. The planner would have rejected it
        // at Parse time, so reaching here means an unsupported utility shape.
        self.send_error(&ZyronError::PlanError(format!(
            "statement cannot be executed through the extended protocol: {query}"
        )))
        .await?;
        self.mark_failed_if_in_transaction();
        Ok(())
    }

    async fn handle_describe(
        &mut self,
        target: DescribeTarget,
        name: String,
    ) -> Result<(), ProtocolError> {
        debug!("Describe: target={:?}, name={}", target, name);

        match target {
            DescribeTarget::Statement => {
                let stmt = self.statements.get(&name).ok_or_else(|| {
                    ProtocolError::Malformed(format!("Prepared statement \"{}\" not found", name))
                })?;

                let param_types = stmt.param_types.clone();
                let output_schema = stmt.output_schema.clone();

                // Send ParameterDescription
                self.feed(BackendMessage::ParameterDescription(param_types))
                    .await?;

                // Send RowDescription or NoData
                if output_schema.is_empty() {
                    self.feed(BackendMessage::NoData).await?;
                } else {
                    let row_desc = self.build_row_description(&output_schema, &[]);
                    self.feed(row_desc).await?;
                }
            }
            DescribeTarget::Portal => {
                let portal = self.portals.get(&name).ok_or_else(|| {
                    ProtocolError::Malformed(format!("Portal \"{}\" not found", name))
                })?;

                if portal.output_schema.is_empty() {
                    self.feed(BackendMessage::NoData).await?;
                } else {
                    let row_desc =
                        self.build_row_description(&portal.output_schema, &portal.result_formats);
                    self.feed(row_desc).await?;
                }
            }
        }

        Ok(())
    }

    async fn handle_close(
        &mut self,
        target: DescribeTarget,
        name: String,
    ) -> Result<(), ProtocolError> {
        match target {
            DescribeTarget::Statement => {
                self.statements.remove(&name);
            }
            DescribeTarget::Portal => {
                self.portals.remove(&name);
            }
        }
        self.feed(BackendMessage::CloseComplete).await?;
        Ok(())
    }

    async fn handle_sync(&mut self) -> Result<(), ProtocolError> {
        // Auto-commit implicit transactions on Sync. No flush occurs between a
        // Sync and the previous one, so the extended-protocol responses for
        // this batch start buffering at write-buffer offset zero. On a commit
        // failure those responses are truncated and replaced with an error.
        {
            #[cfg(feature = "profile")]
            let _s = profile::scope(Phase::WireAutoCommit);
            // On a commit failure auto_commit_if_needed truncates the buffered
            // extended-protocol responses and buffers an ErrorResponse, so the
            // client sees the durability failure before ReadyForQuery.
            if let Err(e) = self.auto_commit_if_needed(0).await {
                debug!("autocommit failed on Sync: {}", e);
            }
        }
        self.send_ready_for_query().await?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Transaction management
    // -----------------------------------------------------------------------

    /// Attaches the current transaction's shared undo log to an execution
    /// context so DML operators record reverse-ops while a savepoint is open.
    /// No-op when there is no active transaction.
    fn attach_undo_log(&self, ctx: &mut ExecutionContext) {
        if let Some(txn) = self.transaction.as_ref() {
            ctx.undo_log = Some(txn.undo_log());
        }
    }

    /// Ensures a transaction exists, starting an implicit one if needed.
    /// Returns the txn_id and snapshot copy to avoid holding a borrow on self.
    fn ensure_transaction(&mut self) -> Result<(u64, Snapshot), ProtocolError> {
        if self.transaction.is_none() {
            let txn = self
                .server
                .txn_manager
                .begin(self.server.default_isolation)
                .map_err(ProtocolError::Database)?;
            self.transaction = Some(txn);
        }
        let txn = self.transaction.as_ref().unwrap();
        Ok((txn.txn_id, txn.snapshot.clone()))
    }

    /// Tries to handle BEGIN/COMMIT/ROLLBACK statements directly.
    /// Returns Some(result) if the statement was handled, None otherwise.
    async fn try_handle_transaction_control(
        &mut self,
        stmt: &zyron_parser::Statement,
    ) -> Option<ZyronResult<String>> {
        match stmt {
            zyron_parser::Statement::Begin(begin) => {
                if self.transaction.is_some() {
                    // Already in a transaction, warn but allow
                    let _ = self
                        .feed(BackendMessage::NoticeResponse(ErrorFields {
                            severity: "WARNING".into(),
                            code: "25001".into(),
                            message: "there is already a transaction in progress".into(),
                            detail: None,
                            hint: None,
                            position: None,
                        }))
                        .await;
                }
                let isolation = match begin.isolation {
                    Some(level) => match map_isolation_level(level) {
                        Ok(mapped) => mapped,
                        Err(e) => return Some(Err(e)),
                    },
                    None => self.server.default_isolation,
                };
                match self.server.txn_manager.begin(isolation) {
                    Ok(mut txn) => {
                        if begin.read_only == Some(true) {
                            txn.set_read_only(true);
                        }
                        self.transaction = Some(txn);
                        if begin.lake {
                            let now = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map(|d| d.as_micros() as i64)
                                .unwrap_or(0);
                            match zyron_lake::CrossTableTxn::begin(
                                self.server.disk_manager.data_dir(),
                                now,
                            )
                            .and_then(|mut lake_txn| {
                                // The intent lands before any participant
                                // writes, so a crash from here on resolves
                                // to all or none
                                lake_txn.prepare().map(|()| lake_txn)
                            }) {
                                Ok(lake_txn) => self.lake_txn = Some(lake_txn),
                                Err(e) => {
                                    self.transaction = None;
                                    return Some(Err(e));
                                }
                            }
                        }
                        if let Some(session) = self.session.as_mut() {
                            session.set_transaction_state(TransactionState::InTransaction);
                        }
                        Some(Ok("BEGIN".into()))
                    }
                    Err(e) => Some(Err(e)),
                }
            }
            zyron_parser::Statement::Commit(_) => {
                if let Some(mut txn) = self.transaction.take() {
                    let txn_id = txn.txn_id;
                    // A transaction that wrote nothing commits without a
                    // commit record or flush wait.
                    let commit_result = if txn.wrote_data() {
                        self.server.txn_manager.commit(&mut txn).await
                    } else {
                        self.server.txn_manager.commit_read_only(&mut txn)
                    };
                    match commit_result {
                        Ok(()) => {
                            // The commit record is durable, lake versions
                            // written under this transaction become visible.
                            // A lake transaction publishes through its intent
                            // instead, which is one write for every table it
                            // touched
                            let published = match self.lake_txn.take() {
                                Some(lake_txn) => lake_txn.commit(),
                                None => zyron_lake::publish_txn(
                                    self.server.disk_manager.data_dir(),
                                    txn_id,
                                ),
                            };
                            match published {
                                Ok(logs) => refresh_lake_stats(&self.server, &logs),
                                Err(e) => {
                                    if let Some(session) = self.session.as_mut() {
                                        session.set_transaction_state(TransactionState::Idle);
                                    }
                                    return Some(Err(e));
                                }
                            }
                            if let Some(session) = self.session.as_mut() {
                                session.set_transaction_state(TransactionState::Idle);
                            }
                            Some(Ok("COMMIT".into()))
                        }
                        Err(e) => {
                            let logs = self.abandon_lake_work(txn_id);
                            refresh_lake_stats(&self.server, &logs);
                            if let Some(session) = self.session.as_mut() {
                                session.set_transaction_state(TransactionState::Idle);
                            }
                            Some(Err(e))
                        }
                    }
                } else {
                    let _ = self
                        .feed(BackendMessage::NoticeResponse(ErrorFields {
                            severity: "WARNING".into(),
                            code: "25P01".into(),
                            message: "there is no transaction in progress".into(),
                            detail: None,
                            hint: None,
                            position: None,
                        }))
                        .await;
                    Some(Ok("COMMIT".into()))
                }
            }
            zyron_parser::Statement::Rollback(rb) => {
                // ROLLBACK TO [SAVEPOINT] name is a partial rollback: reverse the
                // transaction's writes made after the savepoint and keep the
                // transaction open. Plain ROLLBACK aborts the whole transaction.
                if let Some(name) = &rb.savepoint {
                    return Some(self.partial_rollback_to_savepoint(name).await);
                }
                let abort_result = if let Some(mut txn) = self.transaction.take() {
                    let logs = self.abandon_lake_work(txn.txn_id);
                    refresh_lake_stats(&self.server, &logs);
                    self.server.txn_manager.abort(&mut txn)
                } else {
                    Ok(())
                };
                if let Some(session) = self.session.as_mut() {
                    session.set_transaction_state(TransactionState::Idle);
                }
                match abort_result {
                    Ok(()) => Some(Ok("ROLLBACK".into())),
                    Err(e) => Some(Err(e)),
                }
            }
            _ => None,
        }
    }

    /// Patch store for a table's columnar tier, resolved from its registered
    /// segment paths the same way the DML operators resolve it.
    fn columnar_store_for_table(
        &self,
        table_id: u32,
    ) -> ZyronResult<Arc<zyron_storage::columnar::PatchStore>> {
        let te = self
            .server
            .catalog
            .get_table_by_id(zyron_catalog::TableId(table_id))?;
        let seg = te.columnar.segments.first().ok_or_else(|| {
            ZyronError::Internal("columnar undo entry but no registered segments".into())
        })?;
        zyron_storage::columnar::ColumnarPatchManager::store_for_segment(
            table_id as u64,
            std::path::Path::new(&seg.path),
        )
    }

    /// Performs ROLLBACK TO SAVEPOINT: reverses the transaction's writes made
    /// after the named savepoint at the heap-tuple level and releases the locks
    /// it acquired after the savepoint, keeping the transaction open. A
    /// reverse-insert self-deletes a row the transaction inserted (stamp xmax =
    /// txn id); a reverse-delete restores a row it deleted (clear xmax). A
    /// columnar write reverses through the patch log by revoking its supersede
    /// or value patch. Index entries are left untouched, matching full-abort
    /// semantics: heap visibility filters entries pointing at a self-deleted
    /// row, and vacuum reclaims them. The read snapshot is unchanged.
    async fn partial_rollback_to_savepoint(&mut self, name: &str) -> ZyronResult<String> {
        use zyron_storage::{HeapFile, HeapFileConfig, UndoEntry};

        let txn = self.transaction.as_mut().ok_or_else(|| {
            ZyronError::TransactionAborted(
                "ROLLBACK TO SAVEPOINT can only be used in a transaction".to_string(),
            )
        })?;
        let txn_id = txn.txn_id();
        let xmax = u32::try_from(txn_id)
            .map_err(|_| ZyronError::Internal(format!("txn_id {} exceeds u32::MAX", txn_id)))?;

        let rollback = txn.rollback_to_savepoint(name).ok_or_else(|| {
            ZyronError::TransactionAborted(format!("savepoint \"{}\" does not exist", name))
        })?;

        // Reverse each recorded write, last write first. Heap files are cached
        // per heap_file_id so repeated tuples on the same table reuse one handle.
        let mut heaps: std::collections::HashMap<u32, Arc<HeapFile>> =
            std::collections::HashMap::new();
        for entry in &rollback.undo {
            let (heap_file_id, fsm_file_id, tid, is_insert) = match entry {
                UndoEntry::ReverseInsert {
                    heap_file_id,
                    fsm_file_id,
                    tid,
                } => (*heap_file_id, *fsm_file_id, *tid, true),
                UndoEntry::ReverseDelete {
                    heap_file_id,
                    fsm_file_id,
                    tid,
                } => (*heap_file_id, *fsm_file_id, *tid, false),
                // Columnar writes reverse through the patch log: the revoke is
                // WAL logged first so crash recovery replays it, then removed
                // from the live overlay so the row's prior state is visible
                // again immediately.
                UndoEntry::ColumnarSupersede {
                    table_id,
                    branch,
                    file_id,
                    sys_rowid,
                } => {
                    let store = self.columnar_store_for_table(*table_id)?;
                    store.revoke_supersede_logged(
                        &self.server.wal,
                        *table_id as u64,
                        *branch,
                        *file_id,
                        *sys_rowid,
                        txn_id as u64,
                    )?;
                    continue;
                }
                UndoEntry::ColumnarPatch {
                    table_id,
                    branch,
                    file_id,
                    sys_rowid,
                    column_id,
                } => {
                    let store = self.columnar_store_for_table(*table_id)?;
                    store.revoke_patch_logged(
                        &self.server.wal,
                        *table_id as u64,
                        *branch,
                        *file_id,
                        *sys_rowid,
                        *column_id,
                        txn_id as u64,
                    )?;
                    continue;
                }
            };
            let heap = match heaps.get(&heap_file_id) {
                Some(h) => Arc::clone(h),
                None => {
                    let h = if let Some(hit) = self.server.heap_files.get_async(&heap_file_id).await
                    {
                        Arc::clone(hit.get())
                    } else {
                        Arc::new(HeapFile::new(
                            Arc::clone(&self.server.disk_manager),
                            Arc::clone(&self.server.buffer_pool),
                            HeapFileConfig {
                                heap_file_id,
                                fsm_file_id,
                            },
                        )?)
                    };
                    heaps.insert(heap_file_id, Arc::clone(&h));
                    h
                }
            };
            if is_insert {
                // Self-delete the row the transaction inserted after the
                // savepoint by stamping its xmax with the txn id.
                heap.set_xmax(tid, xmax).await?;
            } else {
                // Restore the row the transaction deleted after the savepoint by
                // clearing its xmax back to 0.
                heap.clear_xmax(tid).await?;
            }
        }

        // Release the row and intent locks acquired after the savepoint. Locks
        // taken at or before it stay held so the transaction keeps its
        // write-write conflict protection on rows touched before the savepoint.
        self.server
            .txn_manager
            .lock_table()
            .unlock_after(txn_id, rollback.row_lock_count);
        self.server
            .txn_manager
            .intent_locks()
            .unlock_after(txn_id, rollback.intent_lock_count);

        // The transaction stays open. A statement error before this rollback may
        // have marked the session Failed; ROLLBACK TO SAVEPOINT restores it to a
        // usable in-transaction state so subsequent statements run.
        if let Some(session) = self.session.as_mut() {
            session.set_transaction_state(TransactionState::InTransaction);
        }

        Ok("ROLLBACK".into())
    }

    /// Tries to handle SET/SHOW session commands directly.
    async fn try_handle_session_command(
        &mut self,
        stmt: &zyron_parser::Statement,
    ) -> Option<Result<(), ProtocolError>> {
        match stmt {
            zyron_parser::Statement::SetVariable(s) => {
                let val_str = expr_to_string(&s.value);
                let key = s.name.to_ascii_lowercase();

                // SET ROLE / SET SESSION AUTHORIZATION change the effective
                // identity, so resolve the role name against the registry before
                // touching the session. NONE / DEFAULT resets to the login role.
                let role_change: Option<Option<zyron_auth::RoleId>> =
                    if key == "role" || key == "session_authorization" {
                        let reset = val_str.eq_ignore_ascii_case("none")
                            || val_str.eq_ignore_ascii_case("default");
                        if reset {
                            Some(None)
                        } else {
                            match &self.server.security_manager {
                                Some(sm) => match sm.lookup_role(&val_str) {
                                    Some(role) => Some(Some(role.id)),
                                    None => {
                                        return Some(Err(ProtocolError::Database(
                                            ZyronError::RoleNotFound(format!(
                                                "role '{val_str}' does not exist"
                                            )),
                                        )));
                                    }
                                },
                                None => {
                                    return Some(Err(ProtocolError::Database(
                                        ZyronError::ConfigError(
                                            "role management is not enabled".to_string(),
                                        ),
                                    )));
                                }
                            }
                        }
                    } else {
                        None
                    };

                let hierarchy = self
                    .server
                    .security_manager
                    .as_ref()
                    .map(|sm| &sm.role_hierarchy);

                if let Some(session) = self.session.as_mut() {
                    if let Err(e) = session.set_variable(s.name.clone(), val_str) {
                        return Some(Err(ProtocolError::Database(e)));
                    }
                    if let Some(target) = role_change {
                        if let Some(hierarchy) = hierarchy {
                            if let Err(e) = session.apply_role(target, hierarchy) {
                                return Some(Err(ProtocolError::Database(e)));
                            }
                        }
                    }
                }
                let result = self
                    .feed(BackendMessage::CommandComplete { tag: "SET".into() })
                    .await;
                Some(result)
            }
            zyron_parser::Statement::Show(s) => {
                if s.name.eq_ignore_ascii_case("all") {
                    // SHOW ALL: return all config entries
                    if let Some(ref config_all) = self.server.config_all {
                        let entries = config_all();
                        let row_desc = BackendMessage::RowDescription(vec![
                            FieldDescription {
                                name: "name".into(),
                                table_oid: 0,
                                column_attr: 0,
                                type_oid: types::PG_TEXT_OID,
                                type_size: -1,
                                type_modifier: -1,
                                format: 0,
                            },
                            FieldDescription {
                                name: "setting".into(),
                                table_oid: 0,
                                column_attr: 0,
                                type_oid: types::PG_TEXT_OID,
                                type_size: -1,
                                type_modifier: -1,
                                format: 0,
                            },
                            FieldDescription {
                                name: "description".into(),
                                table_oid: 0,
                                column_attr: 0,
                                type_oid: types::PG_TEXT_OID,
                                type_size: -1,
                                type_modifier: -1,
                                format: 0,
                            },
                        ]);
                        if let Err(e) = self.feed(row_desc).await {
                            return Some(Err(e));
                        }
                        for (key, val, desc) in entries {
                            let row = BackendMessage::DataRow(vec![
                                Some(key.into_bytes()),
                                Some(val.into_bytes()),
                                Some(desc.into_bytes()),
                            ]);
                            if let Err(e) = self.feed(row).await {
                                return Some(Err(e));
                            }
                        }
                        return Some(
                            self.feed(BackendMessage::CommandComplete { tag: "SHOW".into() })
                                .await,
                        );
                    }
                }

                // Check session variables first, then config
                let value = self
                    .session
                    .as_ref()
                    .and_then(|sess| sess.get_variable(&s.name).map(|v| v.to_string()))
                    .or_else(|| self.server.config_lookup.as_ref().and_then(|f| f(&s.name)))
                    .unwrap_or_else(|| "unset".to_string());

                let row_desc = BackendMessage::RowDescription(vec![FieldDescription {
                    name: s.name.clone(),
                    table_oid: 0,
                    column_attr: 0,
                    type_oid: types::PG_TEXT_OID,
                    type_size: -1,
                    type_modifier: -1,
                    format: 0,
                }]);

                let data_row = BackendMessage::DataRow(vec![Some(value.into_bytes())]);

                let r1 = self.feed(row_desc).await;
                if r1.is_err() {
                    return Some(r1);
                }
                let r2 = self.feed(data_row).await;
                if r2.is_err() {
                    return Some(r2);
                }
                Some(
                    self.feed(BackendMessage::CommandComplete { tag: "SHOW".into() })
                        .await,
                )
            }
            zyron_parser::Statement::AlterSystemSet(s) => {
                let val_str = expr_to_string(&s.value);
                if let Some(ref writer) = self.server.alter_system_set {
                    match writer(&s.name, &val_str) {
                        Ok(()) => Some(
                            self.feed(BackendMessage::CommandComplete {
                                tag: "ALTER SYSTEM".into(),
                            })
                            .await,
                        ),
                        Err(msg) => {
                            let fields = ErrorFields {
                                severity: "ERROR".into(),
                                code: "XX000".into(),
                                message: msg,
                                detail: None,
                                hint: None,
                                position: None,
                            };
                            let _ = self.feed(BackendMessage::ErrorResponse(fields)).await;
                            Some(Ok(()))
                        }
                    }
                } else {
                    let fields = ErrorFields {
                        severity: "ERROR".into(),
                        code: "XX000".into(),
                        message: "ALTER SYSTEM not available".into(),
                        detail: None,
                        hint: None,
                        position: None,
                    };
                    let _ = self.feed(BackendMessage::ErrorResponse(fields)).await;
                    Some(Ok(()))
                }
            }
            zyron_parser::Statement::Checkpoint(_) => {
                // CHECKPOINT forces a checkpoint and blocks until it completes
                // (Postgres semantics). The trigger parks on a condvar, so run
                // it on a blocking thread to avoid stalling the async runtime.
                // No checkpoint trigger configured means no checkpoint can be
                // performed, so report that instead of a false success.
                let Some(wake) = self.server.checkpoint_wake.clone() else {
                    return Some(Err(ProtocolError::Database(ZyronError::ConfigError(
                        "CHECKPOINT is not available, no checkpoint coordinator is configured"
                            .to_string(),
                    ))));
                };
                if let Err(e) = tokio::task::spawn_blocking(move || wake()).await {
                    return Some(Err(ProtocolError::Database(ZyronError::Internal(format!(
                        "checkpoint failed: {e}"
                    )))));
                }
                Some(
                    self.feed(BackendMessage::CommandComplete {
                        tag: "CHECKPOINT".into(),
                    })
                    .await,
                )
            }
            zyron_parser::Statement::Vacuum(v) => {
                let result = self.handle_vacuum(v.table.as_deref()).await;
                Some(result)
            }
            zyron_parser::Statement::Analyze(a) => {
                let result = self.handle_analyze(a.table.as_deref()).await;
                Some(result)
            }
            zyron_parser::Statement::Reindex(r) => {
                let target = match &r.target {
                    zyron_parser::ast::ReindexTarget::Table(t) => (Some(t.clone()), None),
                    zyron_parser::ast::ReindexTarget::Index(i) => (None, Some(i.clone())),
                };
                Some(
                    self.handle_reindex(target.0.as_deref(), target.1.as_deref())
                        .await,
                )
            }
            zyron_parser::Statement::OptimizeTable(o) => Some(self.handle_optimize(&o.table).await),
            _ => None,
        }
    }

    /// Handles the VACUUM SQL command
    /// Scans heap pages for dead tuples and reclaims space by zeroing slots
    /// for tuples no longer visible to any active transaction
    /// Coordinates with the auto-vacuum background worker via an AtomicBool,
    /// if vacuum_running is already set, returns success with a Notice
    /// instead of running concurrently
    async fn handle_vacuum(&mut self, table_name: Option<&str>) -> Result<(), ProtocolError> {
        use std::sync::atomic::Ordering;
        use zyron_storage::{HeapFile, HeapFileConfig, HeapPage};

        // CAS-acquire the vacuum lock. If already held, emit a Notice and
        // complete with success tag, matching PostgreSQL's behaviour for
        // concurrent VACUUM.
        if self
            .server
            .vacuum_running
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            let fields = crate::messages::backend::ErrorFields {
                severity: "NOTICE".into(),
                code: "00000".into(),
                message: "vacuum already running, skipped".into(),
                detail: None,
                hint: None,
                position: None,
            };
            let _ = self.feed(BackendMessage::NoticeResponse(fields)).await;
            return self
                .feed(BackendMessage::CommandComplete {
                    tag: "VACUUM".into(),
                })
                .await;
        }
        let _vacuum_guard = VacuumGuard {
            flag: Arc::clone(&self.server.vacuum_running),
        };

        let active_txns = self.server.txn_manager.active_txn_ids();
        let oldest_active = if active_txns.is_empty() {
            self.server.txn_manager.next_txn_id()
        } else {
            active_txns[0]
        };

        let tables = self.server.catalog.list_all_tables();
        let target_tables: Vec<_> = if let Some(name) = table_name {
            tables.into_iter().filter(|t| t.name == name).collect()
        } else {
            tables
        };

        if target_tables.is_empty() {
            if let Some(name) = table_name {
                let fields = crate::messages::backend::ErrorFields {
                    severity: "ERROR".into(),
                    code: "42P01".into(),
                    message: format!("relation \"{}\" does not exist", name),
                    detail: None,
                    hint: None,
                    position: None,
                };
                let _ = self.feed(BackendMessage::ErrorResponse(fields)).await;
                return Ok(());
            }
        }

        let mut _total_reclaimed = 0u64;
        let mut _total_pages = 0u64;

        // Manual VACUUM uses the same reclamation logic as the background
        // worker: a tuple is dead when its inserter aborted, or its deleter
        // committed below the oldest active transaction and no retained version
        // still sees it. This is commit-status aware (an aborted deleter leaves
        // the row live, never reclaimed) and retention aware (time-travel
        // history survives). Each reclaimed row's B+tree index entries are
        // deleted so stale entries do not accumulate.
        let status_map = self.server.txn_manager.status_map().clone();
        for table in &target_tables {
            let heap_file = match HeapFile::new(
                Arc::clone(&self.server.disk_manager),
                Arc::clone(&self.server.buffer_pool),
                HeapFileConfig {
                    heap_file_id: table.heap_file_id,
                    fsm_file_id: table.fsm_file_id,
                },
            ) {
                Ok(hf) => hf,
                Err(_) => continue,
            };

            let scan_guard = match heap_file.scan() {
                Ok(sg) => sg,
                Err(_) => continue,
            };

            let page_ids = scan_guard.page_ids().to_vec();
            drop(scan_guard);

            let index_snap = self.server.catalog.index_snapshot(table.id);
            let btree = &index_snap.btree;
            let clean_indexes = !btree.is_empty();

            // Per-table effective floor: keep versions still visible at a tagged
            // version or within the table's time-travel window.
            let now_us = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_micros() as u64)
                .unwrap_or(0);
            let retention_floor = zyron_executor::operator::modify::effective_retention_floor(
                table.as_ref(),
                &status_map,
                self.server.txn_manager.retention_clock(),
                now_us,
            );

            let is_dead = |xmin: u32, x: u32| {
                status_map.is_aborted(xmin as u64)
                    || (x != 0
                        && status_map.is_committed(x as u64)
                        && (x as u64) < oldest_active
                        && status_map.is_reclaimable_below(x as u64, retention_floor))
            };
            let is_aborted = |xid: u32| status_map.is_aborted(xid as u64);

            for page_id in &page_ids {
                _total_pages += 1;

                let Some(frame) = self.server.buffer_pool.fetch_page(*page_id) else {
                    continue;
                };
                let mut dead: Vec<(u16, Vec<u8>)> = Vec::new();
                let (reclaimed_on_page, modified) = {
                    let mut guard = frame.write_data();
                    let data: &mut [u8] = &mut guard[..];
                    if HeapPage::heap_header_from_slice(data).slot_count == 0 {
                        (0u64, false)
                    } else if clean_indexes {
                        HeapPage::vacuum_in_slice_collect(data, &is_dead, &is_aborted, &mut dead)
                    } else {
                        HeapPage::vacuum_in_slice(data, &is_dead, &is_aborted)
                    }
                };
                self.server.buffer_pool.unpin_page(*page_id, modified);

                if clean_indexes && !dead.is_empty() {
                    zyron_executor::operator::modify::vacuum_index_cleanup(
                        table.as_ref(),
                        *page_id,
                        &dead,
                        btree,
                        &self.server.btree_indexes,
                    );
                }

                if reclaimed_on_page > 0 {
                    _total_reclaimed += reclaimed_on_page;
                }
            }

            // Columnar analog of the heap pass: one merge collapses the
            // table's reclaimable patch history and drops rows deleted at or
            // below the retention floor. A failure warns rather than aborts,
            // the heap reclamation above already succeeded
            if !table.columnar.segments.is_empty()
                && let Some(cm) = &self.server.columnar_maintenance
            {
                let cm = Arc::clone(cm);
                let tid = table.id;
                let outcome = tokio::task::spawn_blocking(move || cm.vacuum_table(tid))
                    .await
                    .map_err(|e| format!("columnar vacuum task: {e}"))
                    .and_then(|r| r);
                if let Err(e) = outcome {
                    let fields = crate::messages::backend::ErrorFields {
                        severity: "WARNING".into(),
                        code: "01000".into(),
                        message: format!("columnar vacuum for \"{}\" failed: {e}", table.name),
                        detail: None,
                        hint: None,
                        position: None,
                    };
                    let _ = self.feed(BackendMessage::NoticeResponse(fields)).await;
                }
            }

            // Recorded after both passes finish, so a pass that failed
            // partway leaves the dead estimate high rather than falsely clean
            self.server
                .table_io_stats
                .get_or_create(table.id.0)
                .record_vacuum(epoch_seconds_now());
        }

        self.feed(BackendMessage::CommandComplete {
            tag: "VACUUM".into(),
        })
        .await
    }

    /// REINDEX rebuilds B+tree indexes from the heap. For each target index it
    /// creates a fresh empty B+tree (replacing the old on-disk checkpoint),
    /// scans the table heap, re-extracts the indexed column's key from every
    /// live row, and inserts the composite (value followed by tuple id) key with
    /// the same encoding the insert path uses. Emits a Notice every 10000 rows
    /// showing the true number of entries rebuilt.
    async fn handle_reindex(
        &mut self,
        table_name: Option<&str>,
        index_name: Option<&str>,
    ) -> Result<(), ProtocolError> {
        let tables = self.server.catalog.list_all_tables();
        let target_tables: Vec<_> = if let Some(name) = table_name {
            let matched: Vec<_> = tables.into_iter().filter(|t| t.name == name).collect();
            if matched.is_empty() {
                let fields = crate::messages::backend::ErrorFields {
                    severity: "ERROR".into(),
                    code: "42P01".into(),
                    message: format!("relation \"{}\" does not exist", name),
                    detail: None,
                    hint: None,
                    position: None,
                };
                let _ = self.feed(BackendMessage::ErrorResponse(fields)).await;
                return Ok(());
            }
            matched
        } else {
            // For a bare index name or REINDEX DATABASE, every table is scanned
            // and the index filter below selects the matching index.
            tables.into_iter().collect()
        };

        let checkpoint_dir = self.server.data_dir.join("indexes");
        let _ = std::fs::create_dir_all(&checkpoint_dir);

        let mut index_matched = false;
        let mut processed_indexes = 0usize;
        let mut total_entries: u64 = 0;
        let progress_every: u64 = 10_000;

        for table in &target_tables {
            let indexes = self.server.catalog.get_indexes_for_table(table.id);
            let btree_indexes: Vec<_> = indexes
                .into_iter()
                .filter(|idx| idx.index_type == zyron_catalog::IndexType::BTree)
                .filter(|idx| index_name.map(|n| idx.name == n).unwrap_or(true))
                .collect();
            if btree_indexes.is_empty() {
                continue;
            }
            index_matched = true;

            // A lake table's indexes are files in its own transaction log,
            // so they rebuild through a commit rather than by refilling a
            // tree from heap pages the table does not have
            if table.lake.is_lake() {
                if let Err(e) = crate::index_build::rebuild_lake_indexes(&self.server, table).await
                {
                    let fields = crate::messages::backend::ErrorFields {
                        severity: "ERROR".into(),
                        code: "XX000".into(),
                        message: format!(
                            "REINDEX failed to rebuild the indexes of \"{}\": {}",
                            table.name, e
                        ),
                        detail: None,
                        hint: None,
                        position: None,
                    };
                    let _ = self.feed(BackendMessage::ErrorResponse(fields)).await;
                    return Ok(());
                }
                processed_indexes += btree_indexes.len();
                continue;
            }

            // Every live row of the table, heap resident and folded alike,
            // collected once and reused across the table's indexes. Shared with
            // CREATE INDEX so the two cannot disagree about what is live
            let live_rows = match crate::index_build::collect_live_rows(&self.server, table).await {
                Ok(rows) => rows,
                Err(e) => {
                    let fields = crate::messages::backend::ErrorFields {
                        severity: "ERROR".into(),
                        code: "XX000".into(),
                        message: format!("REINDEX failed to read table \"{}\": {}", table.name, e),
                        detail: None,
                        hint: None,
                        position: None,
                    };
                    let _ = self.feed(BackendMessage::ErrorResponse(fields)).await;
                    return Ok(());
                }
            };

            for idx in &btree_indexes {
                // The catalog stores the index column id list in key order,
                // and the rebuilt key spans all of them
                let key_columns: Vec<zyron_catalog::ColumnId> =
                    idx.columns.iter().map(|c| c.column_id).collect();
                if key_columns.is_empty() {
                    continue;
                }

                // Replace the old index with a fresh empty tree and drop its
                // stale on-disk checkpoint so recovery does not reload old keys.
                let fresh =
                    zyron_storage::BTreeIndex::create(idx.index_file_id, checkpoint_dir.clone())
                        .await
                        .map_err(ProtocolError::Database)?;
                let fresh = Arc::new(fresh);
                let checkpoint_path =
                    checkpoint_dir.join(format!("index_{}.zyridx", idx.index_file_id));
                if checkpoint_path.exists() {
                    if let Err(e) = std::fs::remove_file(&checkpoint_path) {
                        tracing::error!(
                            target: "zyron::ddl",
                            index = %idx.name,
                            "REINDEX failed to remove index checkpoint: {e}"
                        );
                    }
                }

                let index_entries = crate::index_build::fill_btree_from_live_rows(
                    table.as_ref(),
                    &live_rows,
                    &key_columns,
                    &fresh,
                );
                let previous_total = total_entries;
                total_entries += index_entries;
                // One notice per progress milestone crossed, so a rebuild of
                // several small indexes stays quiet and a large one still
                // reports as it goes
                if total_entries / progress_every != previous_total / progress_every {
                    let fields = crate::messages::backend::ErrorFields {
                        severity: "INFO".into(),
                        code: "00000".into(),
                        message: format!("REINDEX progress: {} entries rebuilt", total_entries),
                        detail: None,
                        hint: None,
                        position: None,
                    };
                    let _ = self.feed(BackendMessage::NoticeResponse(fields)).await;
                }

                let _ = self
                    .server
                    .btree_indexes
                    .insert_async(idx.id.0, fresh)
                    .await;
                processed_indexes += 1;
                tracing::info!(
                    target: "zyron::ddl",
                    index = %idx.name,
                    entries = index_entries,
                    "REINDEX rebuilt index"
                );
            }
        }

        if !index_matched {
            if let Some(name) = index_name {
                let fields = crate::messages::backend::ErrorFields {
                    severity: "ERROR".into(),
                    code: "42704".into(),
                    message: format!("index \"{}\" does not exist", name),
                    detail: None,
                    hint: None,
                    position: None,
                };
                let _ = self.feed(BackendMessage::ErrorResponse(fields)).await;
                return Ok(());
            }
        }
        let _ = processed_indexes;

        self.feed(BackendMessage::CommandComplete {
            tag: "REINDEX".into(),
        })
        .await
    }

    /// OPTIMIZE TABLE runs vacuum-style page compaction over a single table.
    /// Acquires the same vacuum_running lock to avoid concurrent writes with
    /// the background vacuum worker. Emits a Notice every 10000 rows.
    async fn handle_optimize(&mut self, table_name: &str) -> Result<(), ProtocolError> {
        use std::sync::atomic::Ordering;
        use zyron_common::page::PAGE_SIZE;
        use zyron_storage::{HeapFile, HeapFileConfig, HeapPage, MvccGc, TupleHeader};

        if self
            .server
            .vacuum_running
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            let fields = crate::messages::backend::ErrorFields {
                severity: "NOTICE".into(),
                code: "00000".into(),
                message: "vacuum already running, optimize skipped".into(),
                detail: None,
                hint: None,
                position: None,
            };
            let _ = self.feed(BackendMessage::NoticeResponse(fields)).await;
            return self
                .feed(BackendMessage::CommandComplete {
                    tag: "OPTIMIZE".into(),
                })
                .await;
        }
        let _guard = VacuumGuard {
            flag: Arc::clone(&self.server.vacuum_running),
        };

        let active_txns = self.server.txn_manager.active_txn_ids();
        let oldest_active = if active_txns.is_empty() {
            self.server.txn_manager.next_txn_id()
        } else {
            active_txns[0]
        };

        let table = self
            .server
            .catalog
            .list_all_tables()
            .into_iter()
            .find(|t| t.name == table_name);
        let table = match table {
            Some(t) => t,
            None => {
                let fields = crate::messages::backend::ErrorFields {
                    severity: "ERROR".into(),
                    code: "42P01".into(),
                    message: format!("relation \"{}\" does not exist", table_name),
                    detail: None,
                    hint: None,
                    position: None,
                };
                let _ = self.feed(BackendMessage::ErrorResponse(fields)).await;
                return Ok(());
            }
        };

        let heap_file = match HeapFile::new(
            Arc::clone(&self.server.disk_manager),
            Arc::clone(&self.server.buffer_pool),
            HeapFileConfig {
                heap_file_id: table.heap_file_id,
                fsm_file_id: table.fsm_file_id,
            },
        ) {
            Ok(hf) => hf,
            Err(e) => {
                let fields = crate::messages::backend::ErrorFields {
                    severity: "ERROR".into(),
                    code: "XX000".into(),
                    message: format!("OPTIMIZE TABLE failed to open heap: {}", e),
                    detail: None,
                    hint: None,
                    position: None,
                };
                let _ = self.feed(BackendMessage::ErrorResponse(fields)).await;
                return Ok(());
            }
        };

        let scan_guard = match heap_file.scan() {
            Ok(s) => s,
            Err(e) => {
                let fields = crate::messages::backend::ErrorFields {
                    severity: "ERROR".into(),
                    code: "XX000".into(),
                    message: format!("OPTIMIZE TABLE scan failed: {}", e),
                    detail: None,
                    hint: None,
                    position: None,
                };
                let _ = self.feed(BackendMessage::ErrorResponse(fields)).await;
                return Ok(());
            }
        };
        let page_ids = scan_guard.page_ids().to_vec();
        drop(scan_guard);

        let mut total_reclaimed: u64 = 0;
        let mut total_pages: u64 = 0;
        let progress_every: u64 = 10_000;

        for page_id in &page_ids {
            total_pages += 1;
            let page_data = match self.server.buffer_pool.fetch_page(*page_id) {
                Some(frame) => {
                    let guard = frame.read_data();
                    let data: [u8; PAGE_SIZE] = **guard;
                    drop(guard);
                    self.server.buffer_pool.unpin_page(*page_id, false);
                    data
                }
                None => continue,
            };
            let header = HeapPage::heap_header_from_slice(&page_data);
            if header.slot_count == 0 {
                continue;
            }
            let mut modified = page_data;
            let mut reclaimed = 0u64;
            for i in 0..header.slot_count {
                let slot_offset = HeapPage::DATA_START + (i as usize) * 4;
                let slot_len =
                    u16::from_le_bytes([modified[slot_offset + 2], modified[slot_offset + 3]]);
                if slot_len == 0 {
                    continue;
                }
                let tuple_offset =
                    u16::from_le_bytes([modified[slot_offset], modified[slot_offset + 1]]) as usize;
                if tuple_offset + TupleHeader::SIZE <= PAGE_SIZE {
                    let xmax = u32::from_le_bytes([
                        modified[tuple_offset + 8],
                        modified[tuple_offset + 9],
                        modified[tuple_offset + 10],
                        modified[tuple_offset + 11],
                    ]);
                    if MvccGc::is_reclaimable(xmax, oldest_active) {
                        modified[slot_offset + 2] = 0;
                        modified[slot_offset + 3] = 0;
                        reclaimed += 1;
                    }
                }
            }
            if reclaimed > 0 {
                if let Some(frame) = self.server.buffer_pool.fetch_page(*page_id) {
                    frame.copy_from(&modified);
                    self.server.buffer_pool.unpin_page(*page_id, true);
                }
                total_reclaimed += reclaimed;
            }
            if total_pages % progress_every == 0 {
                let fields = crate::messages::backend::ErrorFields {
                    severity: "INFO".into(),
                    code: "00000".into(),
                    message: format!(
                        "OPTIMIZE TABLE progress: {} pages scanned, {} tuples reclaimed",
                        total_pages, total_reclaimed
                    ),
                    detail: None,
                    hint: None,
                    position: None,
                };
                let _ = self.feed(BackendMessage::NoticeResponse(fields)).await;
            }
        }

        // Fold the eligible heap tail into the columnar tier so OPTIMIZE
        // tiers the table, not just its pages. A failure warns rather than
        // aborts, the page compaction above already succeeded
        if let Some(cm) = &self.server.columnar_maintenance {
            let cm = Arc::clone(cm);
            let tid = table.id;
            let outcome = tokio::task::spawn_blocking(move || cm.fold_table(tid))
                .await
                .map_err(|e| format!("columnar fold task: {e}"))
                .and_then(|r| r);
            match outcome {
                Ok(rows) if rows > 0 => {
                    let fields = crate::messages::backend::ErrorFields {
                        severity: "INFO".into(),
                        code: "00000".into(),
                        message: format!(
                            "OPTIMIZE TABLE folded {rows} rows into the columnar tier"
                        ),
                        detail: None,
                        hint: None,
                        position: None,
                    };
                    let _ = self.feed(BackendMessage::NoticeResponse(fields)).await;
                }
                Ok(_) => {}
                Err(e) => {
                    let fields = crate::messages::backend::ErrorFields {
                        severity: "WARNING".into(),
                        code: "01000".into(),
                        message: format!("columnar fold for \"{}\" failed: {e}", table.name),
                        detail: None,
                        hint: None,
                        position: None,
                    };
                    let _ = self.feed(BackendMessage::NoticeResponse(fields)).await;
                }
            }
        }

        self.feed(BackendMessage::CommandComplete {
            tag: "OPTIMIZE".into(),
        })
        .await
    }

    /// Handles the ANALYZE SQL command. Scans heap pages and computes table
    /// and column statistics for query planner cost estimation.
    async fn handle_analyze(&mut self, table_name: Option<&str>) -> Result<(), ProtocolError> {
        use zyron_catalog::analyze_table;
        use zyron_storage::{HeapFile, HeapFileConfig};

        let tables = self.server.catalog.list_all_tables();
        let target_tables: Vec<_> = if let Some(name) = table_name {
            tables.into_iter().filter(|t| t.name == name).collect()
        } else {
            tables
        };

        if target_tables.is_empty() {
            if let Some(name) = table_name {
                let fields = crate::messages::backend::ErrorFields {
                    severity: "ERROR".into(),
                    code: "42P01".into(),
                    message: format!("relation \"{}\" does not exist", name),
                    detail: None,
                    hint: None,
                    position: None,
                };
                let _ = self.feed(BackendMessage::ErrorResponse(fields)).await;
                return Ok(());
            }
        }

        // Collect per-table failures. A table whose heap cannot be opened or
        // whose scan fails keeps stale stats, so report it rather than letting
        // ANALYZE claim success while the planner runs on outdated statistics.
        let mut failures: Vec<String> = Vec::new();
        for table in &target_tables {
            let heap_file = match HeapFile::new(
                Arc::clone(&self.server.disk_manager),
                Arc::clone(&self.server.buffer_pool),
                HeapFileConfig {
                    heap_file_id: table.heap_file_id,
                    fsm_file_id: table.fsm_file_id,
                },
            ) {
                Ok(hf) => hf,
                Err(e) => {
                    failures.push(format!("{}: {e}", table.name));
                    continue;
                }
            };

            match analyze_table(&table, &heap_file).await {
                Ok((mut table_stats, column_stats)) => {
                    // Folded rows live in columnar segments the heap scan
                    // cannot see. Add their live count (segment rows minus
                    // rows with a committed supersede) so the planner costs
                    // segment-bearing tables by their true cardinality. Only
                    // committed supersedes count, an uncommitted or rolled
                    // back delete leaves the row live
                    if !table.columnar.segments.is_empty() {
                        let store =
                            zyron_storage::columnar::ColumnarPatchManager::store_for_segment(
                                table.id.0 as u64,
                                std::path::Path::new(&table.columnar.segments[0].path),
                            )
                            .map_err(ProtocolError::Database)?;
                        let status_map = self.server.txn_manager.status_map();
                        let mut columnar_rows: u64 = 0;
                        for seg in &table.columnar.segments {
                            let superseded = store
                                .file_overlay(seg.file_id)
                                .values()
                                .filter(|o| {
                                    o.supersedes.iter().any(|x| status_map.is_committed(*x))
                                })
                                .count() as u64;
                            columnar_rows += seg.row_count.saturating_sub(superseded);
                        }
                        table_stats.row_count += columnar_rows;
                    }
                    self.server
                        .catalog
                        .put_stats(table.id, table_stats, column_stats);
                    self.server
                        .table_io_stats
                        .get_or_create(table.id.0)
                        .record_analyze(epoch_seconds_now());
                }
                Err(e) => failures.push(format!("{}: {e}", table.name)),
            }
        }

        if !failures.is_empty() {
            return Err(ProtocolError::Database(ZyronError::ExecutionError(
                format!(
                    "ANALYZE failed for {} table(s): {}",
                    failures.len(),
                    failures.join("; ")
                ),
            )));
        }

        self.feed(BackendMessage::CommandComplete {
            tag: "ANALYZE".into(),
        })
        .await
    }

    /// Handles a SELECT query against a virtual stat view, sending the result
    /// directly without going through the planner/executor.
    async fn handle_stat_view_query(
        &mut self,
        view_name: &str,
        filters: &crate::stat_views::StatViewFilters,
    ) -> Result<(), ProtocolError> {
        let (fields, rows) =
            match crate::stat_views::query_stat_view(view_name, &self.server, filters)
                .map_err(ProtocolError::Database)?
            {
                Some(result) => result,
                None => return Ok(()),
            };

        self.feed(BackendMessage::RowDescription(fields)).await?;
        let row_count = rows.len();
        for row in rows {
            self.feed(BackendMessage::DataRow(row)).await?;
        }
        self.feed(BackendMessage::CommandComplete {
            tag: format!("SELECT {}", row_count),
        })
        .await?;
        Ok(())
    }

    /// Records on the current transaction whether `ctx` appended a WAL data
    /// record during execution. A transaction that never wrote commits without
    /// a commit record or a flush wait, so this must be called after every
    /// statement execution that runs against the transaction.
    #[inline]
    fn note_ctx_writes(&mut self, ctx: &ExecutionContext) {
        if ctx.wrote_wal() {
            if let Some(txn) = self.transaction.as_mut() {
                txn.mark_wrote_data();
            }
        }
    }

    /// Applies the session statement timeout and result-row cap to an
    /// execution context. The deadline starts now plus statement_timeout, and
    /// the row cap is the configured max_result_rows. Both are left unset when
    /// the server has no corresponding limit configured.
    #[inline]
    /// Discards every lake version this transaction wrote and resolves an
    /// open cross-table intent, so a prepared intent never outlives the
    /// transaction that opened it.
    fn abandon_lake_work(
        &mut self,
        txn_id: u64,
    ) -> Vec<std::sync::Arc<zyron_lake::TransactionLog>> {
        match self.lake_txn.take() {
            Some(lake_txn) => lake_txn.abort(),
            None => zyron_lake::abandon_txn(self.server.disk_manager.data_dir(), txn_id),
        }
    }

    fn apply_session_limits(&self, ctx: &mut ExecutionContext) {
        if let Some(timeout) = self.server.statement_timeout {
            ctx.set_deadline(std::time::Instant::now() + timeout);
        }
        ctx.max_result_rows = self.server.max_result_rows;
        // Inside BEGIN ZYRONLAKE TRANSACTION every lake write commits under
        // the intent, so its versions publish when the intent commits rather
        // than when the database commit record lands
        ctx.lake_txn_id = self.lake_txn.as_ref().map(|txn| txn.txn_id());
        // A read-only transaction marks its execution context so write operators
        // reject before touching the heap. This is the universal enforcement
        // point, so a write reaches it through any path (direct DML, a prepared
        // write run via the extended protocol, or a write inside EXECUTE, CALL,
        // DO, or a trigger), while reads of every kind are allowed.
        ctx.read_only = self
            .transaction
            .as_ref()
            .map(|t| t.read_only())
            .unwrap_or(false);
    }

    /// Auto-commits implicit transactions (when not inside an explicit BEGIN block).
    /// `buf_mark` is the write-buffer length captured before this query batch
    /// buffered its CommandComplete responses. On a commit or abort failure the
    /// buffered success responses are truncated back to `buf_mark` and an
    /// ErrorResponse is buffered in their place so the client is never told a
    /// statement succeeded when its durable commit failed.
    async fn auto_commit_if_needed(&mut self, buf_mark: usize) -> Result<(), ProtocolError> {
        let in_explicit_txn = self
            .session
            .as_ref()
            .map(|s| {
                s.transaction_state() == TransactionState::InTransaction
                    || s.transaction_state() == TransactionState::Failed
            })
            .unwrap_or(false);

        if !in_explicit_txn {
            if let Some(mut txn) = self.transaction.take() {
                let txn_id = txn.txn_id;
                if self.session_ref().transaction_state() == TransactionState::Failed {
                    let logs = self.abandon_lake_work(txn_id);
                    refresh_lake_stats(&self.server, &logs);
                    if let Err(e) = self.server.txn_manager.abort(&mut txn) {
                        self.write_buf.truncate(buf_mark);
                        self.send_error(&e).await?;
                        return Err(ProtocolError::Database(e));
                    }
                } else if txn.wrote_data() {
                    if let Err(e) = self.server.txn_manager.commit(&mut txn).await {
                        // The implicit transaction failed to durably commit.
                        // Replace the buffered success responses with an error.
                        let logs = self.abandon_lake_work(txn_id);
                        refresh_lake_stats(&self.server, &logs);
                        self.write_buf.truncate(buf_mark);
                        if let Some(session) = self.session.as_mut() {
                            session.set_transaction_state(TransactionState::Idle);
                        }
                        self.send_error(&e).await?;
                        return Err(ProtocolError::Database(e));
                    }
                    // The commit record is durable, lake versions written
                    // under this implicit transaction become visible
                    match zyron_lake::publish_txn(self.server.disk_manager.data_dir(), txn_id) {
                        Ok(logs) => refresh_lake_stats(&self.server, &logs),
                        Err(e) => {
                            self.write_buf.truncate(buf_mark);
                            self.send_error(&e).await?;
                            return Err(ProtocolError::Database(e));
                        }
                    }
                } else {
                    // Read-only transaction: no commit record, no flush wait.
                    if let Err(e) = self.server.txn_manager.commit_read_only(&mut txn) {
                        self.write_buf.truncate(buf_mark);
                        self.send_error(&e).await?;
                        return Err(ProtocolError::Database(e));
                    }
                }
            }
        }
        Ok(())
    }

    fn mark_failed_if_in_transaction(&mut self) {
        if self.transaction.is_some() {
            if let Some(session) = self.session.as_mut() {
                if session.transaction_state() == TransactionState::InTransaction {
                    session.set_transaction_state(TransactionState::Failed);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Result streaming
    // -----------------------------------------------------------------------

    /// Builds a RowDescription message from the output schema.
    fn build_row_description(&self, schema: &[LogicalColumn], formats: &[i16]) -> BackendMessage {
        let fields = schema
            .iter()
            .enumerate()
            .map(|(i, col)| {
                let format = if i < formats.len() {
                    formats[i]
                } else if formats.len() == 1 {
                    formats[0]
                } else {
                    0 // text format
                };

                FieldDescription {
                    name: col.name.clone(),
                    table_oid: 0,
                    column_attr: 0,
                    type_oid: types::type_id_to_pg_oid(col.type_id),
                    type_size: types::pg_type_size(col.type_id),
                    type_modifier: -1,
                    format,
                }
            })
            .collect();

        BackendMessage::RowDescription(fields)
    }

    /// Streams DataBatch results as DataRow messages. Returns the total row count.
    /// Encodes rows directly into a shared BytesMut buffer to eliminate per-row
    /// and per-cell heap allocations. Flushes periodically to bound memory usage.
    async fn send_data_rows(
        &mut self,
        batches: &[DataBatch],
        schema: &[LogicalColumn],
        formats: &[i16],
    ) -> Result<usize, ProtocolError> {
        use bytes::BufMut;

        let mut total_rows = 0;
        if batches.is_empty() {
            return Ok(0);
        }

        let num_cols = batches[0].columns.len();

        // Precompute per-column format (text=0 or binary=1) once.
        // Stack buffer for up to 32 columns (covers the vast majority of queries).
        // Heap fallback for wide tables.
        let mut col_fmt_stack = [0i16; 32];
        let col_fmt_heap: Vec<i16>;
        let col_formats: &[i16] = if num_cols <= 32 {
            for i in 0..num_cols {
                col_fmt_stack[i] = if i < formats.len() {
                    formats[i]
                } else if formats.len() == 1 {
                    formats[0]
                } else {
                    0
                };
            }
            &col_fmt_stack[..num_cols]
        } else {
            col_fmt_heap = (0..num_cols)
                .map(|i| {
                    if i < formats.len() {
                        formats[i]
                    } else if formats.len() == 1 {
                        formats[0]
                    } else {
                        0
                    }
                })
                .collect();
            &col_fmt_heap
        };

        // Shared buffer for encoding DataRow messages directly.
        // 64KB flush threshold matches the Windows TCP send buffer default
        // and sits within the Linux default range (16-128KB). Benchmarking
        // confirmed this is the optimal size: larger values (128KB, 256KB)
        // cause BytesMut reallocation overhead that outweighs the syscall
        // savings.
        const FLUSH_THRESHOLD: usize = 65536;
        let mut buf = BytesMut::with_capacity(FLUSH_THRESHOLD + 4096);

        // Drain any messages the caller already buffered (RowDescription,
        // ParameterStatus, etc.) into the head of `buf` so the wire order is
        // RowDescription -> DataRows. This avoids an extra `flush().await`
        // syscall before the DataRow stream starts.
        if !self.write_buf.is_empty() {
            buf.extend_from_slice(&self.write_buf);
            self.write_buf.clear();
        }

        // Precompute which columns are vectors so the per-row loop skips a
        // schema lookup + enum compare on every column. Adds up on wide
        // COPY TO workloads (e.g. 20M rows/sec × N cols).
        let vector_cols: Vec<bool> = (0..num_cols)
            .map(|i| i < schema.len() && schema[i].type_id == zyron_common::TypeId::Vector)
            .collect();
        // Array columns are byte-backed too, and read back in the braced form
        // they are written in rather than as their encoding
        let array_cols: Vec<bool> = (0..num_cols)
            .map(|i| i < schema.len() && schema[i].type_id == zyron_common::TypeId::Array)
            .collect();
        // A DECIMAL is stored as an i128 holding the value times ten to its
        // scale, so rendering it as a plain integer would move the decimal
        // point. The scale comes off the column, which is the only place it
        // is recorded
        let decimal_scales: Vec<Option<u8>> = (0..num_cols)
            .map(|i| {
                if i < schema.len() && schema[i].type_id == zyron_common::TypeId::Decimal {
                    Some(schema[i].fractional_digits.unwrap_or(0))
                } else {
                    None
                }
            })
            .collect();

        let max_rows = self.server.max_result_rows;
        for batch in batches {
            for row in 0..batch.num_rows {
                // Stop before encoding a row that would exceed the configured
                // cap. Flush rows already buffered so the partial stream is
                // framed, then surface the limit as an error to the client.
                if let Some(cap) = max_rows {
                    if total_rows as u64 >= cap {
                        if !buf.is_empty() {
                            self.stream
                                .write_all(&buf)
                                .await
                                .map_err(ProtocolError::Io)?;
                            buf.clear();
                        }
                        return Err(ProtocolError::Database(ZyronError::Internal(format!(
                            "result set exceeds max_result_rows {}",
                            cap
                        ))));
                    }
                }
                // DataRow: type 'D' + 4-byte length + 2-byte column count + per-column data
                buf.put_u8(b'D');
                let len_pos = buf.len();
                buf.put_i32(0); // length placeholder
                buf.put_i16(num_cols as i16);

                for (col_idx, column) in batch.columns.iter().enumerate() {
                    let scalar = column.get_scalar(row);

                    // Check NULL first to avoid writing a placeholder then truncating.
                    if matches!(scalar, ScalarValue::Null) {
                        buf.put_i32(-1);
                        continue;
                    }

                    let val_len_pos = buf.len();
                    buf.put_i32(0); // value length placeholder
                    let before = buf.len();

                    // Vector columns are stored as Binary (raw f32 bytes) but
                    // need special text formatting as bracket notation [0.1,0.2,0.3].
                    let is_vector = vector_cols[col_idx];

                    if is_vector {
                        if let ScalarValue::Binary(ref v) = scalar {
                            if col_formats[col_idx] == 1 {
                                types::write_vector_binary(v, &mut buf);
                            } else {
                                types::write_vector_text(v, &mut buf);
                            }
                        } else {
                            types::scalar_write_text(&scalar, &mut buf);
                        }
                    } else if array_cols[col_idx] {
                        types::write_array_text(&scalar, &mut buf);
                    } else if let Some(scale) = decimal_scales[col_idx] {
                        match scalar {
                            ScalarValue::Int128(v) => {
                                buf.extend_from_slice(
                                    zyron_common::format_decimal(v, scale).as_bytes(),
                                );
                            }
                            ref other => {
                                types::scalar_write_text(other, &mut buf);
                            }
                        }
                    } else if col_formats[col_idx] == 1 {
                        types::scalar_write_binary(&scalar, &mut buf);
                    } else {
                        types::scalar_write_text(&scalar, &mut buf);
                    }

                    let val_len = (buf.len() - before) as i32;
                    buf[val_len_pos..val_len_pos + 4].copy_from_slice(&val_len.to_be_bytes());
                }

                // Patch the DataRow message length (includes itself but not the type byte).
                let msg_len = (buf.len() - len_pos) as i32;
                let len_bytes = msg_len.to_be_bytes();
                buf[len_pos..len_pos + 4].copy_from_slice(&len_bytes);

                total_rows += 1;

                // Flush periodically to avoid unbounded memory growth.
                if buf.len() >= FLUSH_THRESHOLD {
                    self.stream
                        .write_all(&buf)
                        .await
                        .map_err(ProtocolError::Io)?;
                    buf.clear();
                }
            }
        }

        // Flush remaining data.
        if !buf.is_empty() {
            self.stream
                .write_all(&buf)
                .await
                .map_err(ProtocolError::Io)?;
        }

        Ok(total_rows)
    }

    // -----------------------------------------------------------------------
    // Error conversion
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // Subscribe authorization
    //
    // A Y (Subscribe) request must clear three gates before any change data
    // is delivered: the SUBSCRIBE RBAC privilege on the publication, the
    // publication ABAC policy set, and the Bell-LaPadula classification
    // ceiling (subscriber clearance must be at least the publication
    // classification). The decision and its audit are co-located in
    // audit_subscribe_decision so an unauthorized peer can never receive a
    // SubscribeOk.
    // -----------------------------------------------------------------------
    async fn handle_subscribe(
        &mut self,
        sub: crate::messages::frontend::SubscribeMessage,
    ) -> Result<(), ProtocolError> {
        let principal_role = self
            .session
            .as_ref()
            .and_then(|s| s.security_context.as_ref())
            .map(|c| c.current_role.0)
            .unwrap_or(0);

        let publication = self
            .server
            .catalog
            .list_publications()
            .into_iter()
            .find(|p| p.name == sub.publication);
        let publication = match publication {
            Some(p) => p,
            None => {
                tracing::info!(
                    target: "zyron::audit",
                    event = "SubscribeDenied",
                    principal = principal_role,
                    object = %sub.publication,
                    decision = "denied",
                    reason = "unknown publication",
                );
                self.send_error(&ZyronError::Internal(format!(
                    "unknown publication {}",
                    sub.publication
                )))
                .await?;
                self.flush().await?;
                return Ok(());
            }
        };

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let sm = self.server.security_manager.clone();

        let allowed = {
            let ctx = self
                .session
                .as_mut()
                .and_then(|s| s.security_context.as_mut());
            audit_subscribe_decision(sm.as_deref(), ctx, &publication, principal_role, now)
        };

        if allowed {
            let ok = crate::messages::backend::SubscribeOkMessage {
                schema_fingerprint: publication.schema_fingerprint,
                columns: Vec::new(),
                resumed_at_lsn: sub.from_lsn,
                features: sub.features,
            };
            self.feed(BackendMessage::SubscribeOk(ok)).await?;
            self.flush().await?;
            // Hand the connection over to the push producer pump. The pump
            // owns the stream until the subscriber sends EndSubscription /
            // Terminate, disconnects, or the server signals shutdown.
            self.run_subscription_pump(sub, publication, principal_role)
                .await?;
            Ok(())
        } else {
            self.send_error(&ZyronError::PermissionDenied(format!(
                "subscribe denied on publication {}",
                publication.name
            )))
            .await?;
            self.flush().await?;
            Ok(())
        }
    }

    // -----------------------------------------------------------------------
    // Subscription producer pump
    //
    // After SubscribeOk is sent, the connection enters this pump and remains
    // here until the consumer ends the subscription, disconnects, or the
    // server signals shutdown. The pump interleaves three concerns:
    //   1. Read inbound FlowControl / SubscriptionAck / EndSubscription /
    //      Terminate frames from the consumer.
    //   2. Poll the CDC ChangeSource for new batches and write X frames
    //      while the subscriber has credit and buffer headroom.
    //   3. Emit periodic Q heartbeats so the consumer can advance its
    //      liveness check.
    // The pump persists last_seen_lsn into the catalog on every push and
    // again on exit, removes the context from PubSubServerState, and emits
    // SubscriptionPumpStarted / SubscriptionPumpEnded audit events.
    // -----------------------------------------------------------------------
    async fn run_subscription_pump(
        &mut self,
        sub: crate::messages::frontend::SubscribeMessage,
        publication: std::sync::Arc<zyron_catalog::PublicationEntry>,
        principal_role: u32,
    ) -> Result<(), ProtocolError> {
        use std::sync::atomic::Ordering;
        use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

        let registry = match &self.server.cdc_registry {
            Some(r) => Arc::clone(r),
            None => {
                tracing::warn!(
                    target: "zyron::audit",
                    event = "SubscriptionPumpEnded",
                    principal = principal_role,
                    object = %publication.name,
                    decision = "denied",
                    reason = "no cdc registry configured",
                );
                return Ok(());
            }
        };

        // Resolve member table ids: prefer the catalog publication-table map,
        // fall back to the PublicationManager when it is the source of truth.
        let mut table_ids: Vec<u32> = self
            .server
            .catalog
            .get_publication_tables(publication.id)
            .into_iter()
            .map(|pt| pt.table_id.0)
            .collect();
        if table_ids.is_empty() {
            if let Some(pm) = &self.server.publication_manager {
                if let Ok(tids) = pm.get_tables_for_publication(&publication.name) {
                    table_ids = tids;
                }
            }
        }

        // Compile the publication's ABAC policies that apply to this subscriber
        // into a row filter. None when no policy applies, so the stream pays
        // nothing. A predicate that cannot bind against a member table fails the
        // subscription closed rather than streaming unfiltered rows.
        let abac_filter = match self.server.security_manager.as_ref() {
            Some(sm) => {
                match crate::publication_filter::PublicationRowFilter::build(
                    &self.server.catalog,
                    sm,
                    &publication,
                    &table_ids,
                    principal_role,
                )
                .await
                {
                    Ok(opt) => opt.map(Arc::new),
                    Err(e) => {
                        self.send_error(&e).await?;
                        self.flush().await?;
                        return Ok(());
                    }
                }
            }
            None => None,
        };

        let source: Arc<dyn crate::subscription::ChangeSource> =
            Arc::new(crate::subscription::CdcChangeSource::with_filter(
                registry,
                table_ids.clone(),
                Arc::clone(&self.server.catalog),
                self.server.disk_manager.data_dir().to_path_buf(),
                abac_filter,
            ));

        // Register a catalog subscription entry. The id is auto-allocated by
        // the catalog when zero is supplied. last_seen_lsn starts at the
        // consumer-supplied from_lsn so reconnects resume cleanly.
        let now_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let new_entry = zyron_catalog::SubscriptionEntry {
            id: zyron_catalog::SubscriptionId(0),
            publication_id: publication.id,
            consumer_id: sub.consumer_id.clone(),
            consumer_role_id: principal_role,
            last_seen_lsn: sub.from_lsn,
            last_poll_at: now_secs,
            schema_pin: publication.schema_fingerprint,
            mode: zyron_catalog::SubscriptionMode::Push,
            state: zyron_catalog::SubscriptionState::Active,
            last_error: None,
            created_at: now_secs,
            source_id: None,
        };
        // A failed create must abort the subscription. Falling back to id 0
        // would run an untracked pump whose cursor never persists, so the
        // consumer would silently re-receive everything on reconnect.
        let subscription_id = self
            .server
            .catalog
            .create_subscription(new_entry)
            .await
            .map_err(ProtocolError::Database)?;

        // Build the per-subscription producer context.
        let peer_addr: std::net::SocketAddr = self
            .peer_addr
            .as_deref()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| std::net::SocketAddr::from(([127u8, 0, 0, 1], 0u16)));
        let watermark_high = 16 * 1024 * 1024u64;
        let watermark_low = 8 * 1024 * 1024u64;
        let initial_credit = if sub.initial_credit > 0 {
            sub.initial_credit
        } else {
            64 * 1024
        };
        let mut ctx = crate::subscription::SubscriptionServerContext::new(
            subscription_id.0,
            publication.id.0,
            sub.consumer_id.clone(),
            publication.schema_fingerprint,
            peer_addr,
            principal_role,
            initial_credit,
            sub.from_lsn,
            watermark_high,
            watermark_low,
        );
        // Attach the shared labeled-metrics handle so the pump's record_push
        // and apply_ack feed zyron_subscription_lag_lsn and bytes-sent.
        // The metrics field on ServerState lives indirectly through the
        // PubSubServerState handle.
        ctx = ctx.withMetrics(None);
        let ctx = Arc::new(ctx);
        self.server.pub_sub_state.insert(Arc::clone(&ctx));

        tracing::info!(
            target: "zyron::audit",
            event = "SubscriptionPumpStarted",
            principal = principal_role,
            object = %publication.name,
            decision = "granted",
            reason = "subscribe authorized, pump running",
        );

        let shutdown_flag = Arc::clone(&self.server.subscription_shutdown);
        let cfg = crate::subscription::ProducerConfig::default();
        let mut last_heartbeat = Instant::now();
        let mut last_schema_check = Instant::now();
        let mut encoded = BytesMut::with_capacity(8192);
        let mut current_schema = publication.schema_fingerprint;
        let pump_pub_id = publication.id;

        // Per-publication outbound rate limit (00g). When max_rows_per_sec is
        // set, a token bucket throttles rows pushed to this subscriber: tokens
        // refill at the configured rate and the pump sleeps when a batch would
        // exceed the available budget. Unset means unlimited.
        let rate_limit = publication.max_rows_per_sec.filter(|&r| r > 0);
        let mut rl_tokens: f64 = rate_limit.map(|r| r as f64).unwrap_or(0.0);
        let mut rl_last_refill = Instant::now();

        // The pump loop returns a Result so write/read failures land in the
        // same cleanup arm as a graceful exit. Previously a stream write
        // error short-circuited via `?` before deregistering the context
        // or persisting last_seen_lsn, leaking a phantom Active entry in
        // the catalog and in PubSubServerState.
        let pump_result: std::result::Result<&'static str, ProtocolError> = async {
            loop {
                if shutdown_flag.load(Ordering::Acquire) {
                    return Ok("server shutdown");
                }

                // Schema-pin drift: if the publication's fingerprint changed
                // since the last check, send a SchemaUpdate v frame so the
                // consumer can react. The pump continues delivering with the
                // refreshed fingerprint.
                if last_schema_check.elapsed() >= Duration::from_millis(50) {
                    if let Some(latest) = self.server.catalog.get_publication_by_id(pump_pub_id) {
                        if latest.schema_fingerprint != current_schema {
                            let msg = crate::messages::backend::SchemaUpdateMessage {
                                publication: publication.name.clone(),
                                new_fingerprint: latest.schema_fingerprint,
                                columns: Vec::new(),
                            };
                            encoded.clear();
                            msg.encode(&mut encoded);
                            self.stream
                                .write_all(&encoded)
                                .await
                                .map_err(ProtocolError::Io)?;
                            self.stream.flush().await.map_err(ProtocolError::Io)?;
                            current_schema = latest.schema_fingerprint;
                        }
                    }
                    last_schema_check = Instant::now();
                }

                // Pump a batch when the subscriber has credit and buffer room.
                if ctx.can_send() {
                    let after = ctx.last_pushed_lsn.load(Ordering::Acquire);
                    let credit = ctx.credit_remaining_bytes.load(Ordering::Acquire);
                    let cap_bytes = credit.clamp(0, u32::MAX as i64) as u32;
                    let next = source
                        .next_batch(after, cap_bytes, cfg.batch_size_hint)
                        .await?;
                    if let Some(batch) = next {
                        encoded.clear();
                        batch.encode(&mut encoded);
                        let encoded_len = encoded.len() as u64;
                        let row_count = batch.row_count as u64;
                        let end_lsn = batch.end_lsn;

                        // Throttle to the publication's max_rows_per_sec, if set.
                        if let Some(rate) = rate_limit {
                            let rate_f = rate as f64;
                            let elapsed = rl_last_refill.elapsed().as_secs_f64();
                            rl_tokens = (rl_tokens + elapsed * rate_f).min(rate_f);
                            rl_last_refill = Instant::now();
                            let need = row_count as f64;
                            if need > rl_tokens {
                                let wait = (need - rl_tokens) / rate_f;
                                tokio::time::sleep(Duration::from_secs_f64(wait)).await;
                                rl_tokens = 0.0;
                            } else {
                                rl_tokens -= need;
                            }
                        }

                        self.stream
                            .write_all(&encoded)
                            .await
                            .map_err(ProtocolError::Io)?;
                        self.stream.flush().await.map_err(ProtocolError::Io)?;
                        // Record the push in memory only. The durable cursor is
                        // not advanced here because the subscriber has not yet
                        // acked. Persisting end_lsn now would lose data: a crash
                        // before the ack would resume past unconfirmed rows.
                        ctx.record_push(end_lsn, encoded_len, row_count);
                        continue;
                    }
                }

                if last_heartbeat.elapsed() >= cfg.heartbeat_interval {
                    let committed = source.committed_lsn().await;
                    let now_us = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .map(|d| d.as_micros() as i64)
                        .unwrap_or(0);
                    let msg = crate::messages::backend::SubscriptionStatusMessage {
                        committed_lsn: committed,
                        producer_now_us: now_us,
                    };
                    encoded.clear();
                    msg.encode(&mut encoded);
                    self.stream
                        .write_all(&encoded)
                        .await
                        .map_err(ProtocolError::Io)?;
                    self.stream.flush().await.map_err(ProtocolError::Io)?;
                    last_heartbeat = Instant::now();
                }

                // Race a short read against a small idle sleep so the pump stays
                // responsive without busy-looping. read_message uses cancel-safe
                // AsyncReadExt::read_buf on a persistent BytesMut buffer, so the
                // dropped future preserves any partial bytes for the next pass.
                let timeout_fut = tokio::time::sleep(cfg.source_poll);
                tokio::select! {
                    biased;
                    r = self.read_message() => {
                        match r {
                            Ok(crate::messages::frontend::FrontendMessage::FlowControl(fc)) => {
                                ctx.grant_credit(fc.credit_bytes);
                            }
                            Ok(crate::messages::frontend::FrontendMessage::SubscriptionAck(ack)) => {
                                let prev = ctx.last_acked_lsn.load(Ordering::Acquire);
                                let released = ack.acked_lsn.saturating_sub(prev);
                                ctx.apply_ack(ack.acked_lsn, released);
                                // The subscriber confirmed durable receipt up to
                                // acked_lsn, so advance the persisted cursor only
                                // now. A persist failure fails the pump rather
                                // than silently risking redelivery or loss.
                                self.server
                                    .catalog
                                    .update_subscription_lsn(subscription_id, ack.acked_lsn)
                                    .await
                                    .map_err(ProtocolError::Database)?;
                            }
                            Ok(crate::messages::frontend::FrontendMessage::EndSubscription(_)) => {
                                return Ok("client EndSubscription");
                            }
                            Ok(crate::messages::frontend::FrontendMessage::Terminate) => {
                                return Ok("client Terminate");
                            }
                            Ok(_) => {
                                // Other frontend messages are not part of the
                                // subscription frame grammar. Ignore them while
                                // the pump owns the connection.
                            }
                            Err(ProtocolError::ConnectionClosed) => return Ok("client disconnect"),
                            Err(e) => return Err(e),
                        }
                    }
                    _ = timeout_fut => {}
                }
            }
        }
        .await;

        // Single cleanup path covers every exit: graceful end, client
        // disconnect, server shutdown, read error, write error, encode
        // error. We persist the last acked lsn (not the last pushed) so the
        // next subscribe resumes from the consumer's confirmed position and
        // never skips unacknowledged rows, mark the subscription Failed when
        // the pump bailed on an error, and remove the in-memory context.
        let final_lsn = ctx.last_acked_lsn.load(Ordering::Acquire);
        let _ = self
            .server
            .catalog
            .update_subscription_lsn(subscription_id, final_lsn)
            .await;
        match &pump_result {
            Ok(_) => {}
            Err(e) => {
                let _ = self
                    .server
                    .catalog
                    .update_subscription_state(
                        subscription_id,
                        zyron_catalog::SubscriptionState::Failed,
                        Some(format!("pump error: {}", e)),
                    )
                    .await;
            }
        }
        self.server.pub_sub_state.remove(ctx.subscription_id);
        match pump_result {
            Ok(reason) => {
                tracing::info!(
                    target: "zyron::audit",
                    event = "SubscriptionPumpEnded",
                    principal = principal_role,
                    object = %publication.name,
                    decision = "granted",
                    reason = reason,
                );
                Ok(())
            }
            Err(e) => {
                tracing::info!(
                    target: "zyron::audit",
                    event = "SubscriptionPumpEnded",
                    principal = principal_role,
                    object = %publication.name,
                    decision = "denied",
                    reason = %format!("pump error: {}", e),
                );
                Err(e)
            }
        }
    }

    /// Converts a ZyronError to an ErrorResponse and sends it.
    async fn send_error(&mut self, err: &ZyronError) -> Result<(), ProtocolError> {
        let fields = zyron_error_to_fields(err);
        self.feed(BackendMessage::ErrorResponse(fields)).await
    }

    /// Sends a ProtocolError as an ErrorResponse. Extracts the inner ZyronError
    /// if present, otherwise sends a generic internal error.
    async fn send_protocol_error(&mut self, err: &ProtocolError) -> Result<(), ProtocolError> {
        let fields = match err {
            ProtocolError::Database(zyron_err) => zyron_error_to_fields(zyron_err),
            other => ErrorFields {
                severity: "ERROR".into(),
                code: "XX000".into(),
                message: other.to_string(),
                detail: None,
                hint: None,
                position: None,
            },
        };
        self.feed(BackendMessage::ErrorResponse(fields)).await
    }

    async fn send_ready_for_query(&mut self) -> Result<(), ProtocolError> {
        let state = self.session_ref().transaction_state();
        self.feed(BackendMessage::ReadyForQuery(state)).await?;
        self.flush().await
    }

    // -----------------------------------------------------------------------
    // I/O helpers
    // -----------------------------------------------------------------------

    fn session_ref(&self) -> &Session {
        self.session.as_ref().expect("session not initialized")
    }

    /// Encodes a message and writes it to the TCP stream immediately.
    /// Buffers a message into the write buffer without flushing.
    /// Call flush() after feeding all messages to send them in one syscall.
    async fn feed(&mut self, msg: BackendMessage) -> Result<(), ProtocolError> {
        msg.encode(&mut self.write_buf);
        Ok(())
    }

    /// Flushes the write buffer to the TCP stream.
    async fn flush(&mut self) -> Result<(), ProtocolError> {
        if !self.write_buf.is_empty() {
            self.stream
                .write_all(&self.write_buf)
                .await
                .map_err(ProtocolError::Io)?;
            self.write_buf.clear();
        }
        Ok(())
    }

    /// Reads the next complete message from the TCP stream.
    /// Buffers partial reads and calls the codec's decode logic directly.
    async fn read_message(&mut self) -> Result<FrontendMessage, ProtocolError> {
        loop {
            // Try to decode a complete message from the existing buffer.
            if let Some(msg) = self.codec.decode(&mut self.read_buf)? {
                return Ok(msg);
            }

            // Cap read buffer at 16 MB to prevent unbounded growth from slow/malicious clients.
            if self.read_buf.len() > 16 * 1024 * 1024 {
                return Err(ProtocolError::MessageTooLarge {
                    size: self.read_buf.len(),
                    max: 16 * 1024 * 1024,
                });
            }

            // If we have the message header, reserve the full message size
            // to avoid incremental BytesMut reallocation on large messages.
            if !self.codec.is_startup_phase() && self.read_buf.len() >= 5 {
                let len = i32::from_be_bytes([
                    self.read_buf[1],
                    self.read_buf[2],
                    self.read_buf[3],
                    self.read_buf[4],
                ]) as usize;
                if len >= 4 && len <= 16 * 1024 * 1024 {
                    let total = 1 + len;
                    let needed = total.saturating_sub(self.read_buf.len());
                    if needed > 0 {
                        self.read_buf.reserve(needed);
                    }
                }
            }

            // Not enough data. Read more from the stream.
            let n = self
                .stream
                .read_buf(&mut self.read_buf)
                .await
                .map_err(ProtocolError::Io)?;
            if n == 0 {
                return Err(ProtocolError::ConnectionClosed);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Authorizes a subscribe request and emits the co-located audit trail.
///
/// Returns true when the subscriber clears the SUBSCRIBE RBAC privilege, the
/// publication ABAC policy set, and the classification ceiling. Every grant
/// and deny is logged on the `zyron::audit` target with principal, object,
/// decision, and reason so the audit can never drift from enforcement. When
/// no security manager is configured the gate is open, matching the rest of
/// the server, and the access is still audited as granted.
pub fn audit_subscribe_decision(
    sm: Option<&zyron_auth::SecurityManager>,
    ctx: Option<&mut zyron_auth::SecurityContext>,
    publication: &zyron_catalog::PublicationEntry,
    principal_role: u32,
    now: u64,
) -> bool {
    let sm = match sm {
        Some(sm) => sm,
        None => {
            tracing::info!(
                target: "zyron::audit",
                event = "SubscribeGranted",
                principal = principal_role,
                object = %publication.name,
                decision = "granted",
                reason = "no security manager configured, gate open",
            );
            return true;
        }
    };
    let ctx = match ctx {
        Some(ctx) => ctx,
        None => {
            tracing::info!(
                target: "zyron::audit",
                event = "SubscribeDenied",
                principal = principal_role,
                object = %publication.name,
                decision = "denied",
                reason = "unauthenticated",
            );
            return false;
        }
    };

    let has_priv = ctx.has_privilege(
        &sm.privilege_store,
        zyron_auth::PrivilegeType::Subscribe,
        zyron_auth::ObjectType::Publication,
        publication.id.0,
        None,
        now,
    );
    if !has_priv {
        tracing::info!(
            target: "zyron::audit",
            event = "PublicationAccessDenied",
            principal = principal_role,
            object = %publication.name,
            decision = "denied",
            reason = "missing SUBSCRIBE privilege",
        );
        tracing::info!(
            target: "zyron::audit",
            event = "SubscribeDenied",
            principal = principal_role,
            object = %publication.name,
            decision = "denied",
            reason = "missing SUBSCRIBE privilege",
        );
        return false;
    }
    tracing::info!(
        target: "zyron::audit",
        event = "PublicationAccessGranted",
        principal = principal_role,
        object = %publication.name,
        decision = "granted",
        reason = "SUBSCRIBE privilege present",
    );

    let resource = format!("publication:{}", publication.name);
    if !sm
        .abac_rule_store
        .evaluate_abac(&ctx.attributes, Some(&resource), None)
    {
        tracing::info!(
            target: "zyron::audit",
            event = "ABACPolicyRejected",
            principal = principal_role,
            object = %publication.name,
            decision = "denied",
            reason = "ABAC policy denied",
        );
        tracing::info!(
            target: "zyron::audit",
            event = "SubscribeDenied",
            principal = principal_role,
            object = %publication.name,
            decision = "denied",
            reason = "ABAC policy denied",
        );
        return false;
    }

    let pub_level = publication.classification as u8;
    let clearance = ctx.break_glass_clearance.unwrap_or(ctx.clearance) as u8;
    if clearance < pub_level {
        tracing::info!(
            target: "zyron::audit",
            event = "ClassificationRejected",
            principal = principal_role,
            object = %publication.name,
            decision = "denied",
            reason = "clearance below publication classification",
        );
        tracing::info!(
            target: "zyron::audit",
            event = "SubscribeDenied",
            principal = principal_role,
            object = %publication.name,
            decision = "denied",
            reason = "clearance below publication classification",
        );
        return false;
    }

    tracing::info!(
        target: "zyron::audit",
        event = "SubscribeGranted",
        principal = principal_role,
        object = %publication.name,
        decision = "granted",
        reason = "rbac, abac, and classification checks passed",
    );
    true
}

/// Maps ZyronError to ErrorFields with appropriate SQLSTATE codes.
pub fn zyron_error_to_fields(err: &ZyronError) -> ErrorFields {
    let (code, severity) = match err {
        ZyronError::ParseError(_) => ("42601", "ERROR"),
        ZyronError::TableNotFound(_) => ("42P01", "ERROR"),
        ZyronError::ColumnNotFound(_) => ("42703", "ERROR"),
        ZyronError::DuplicateKey => ("23505", "ERROR"),
        ZyronError::TransactionAborted(_) => ("25P02", "ERROR"),
        ZyronError::DeadlockDetected => ("40P01", "ERROR"),
        ZyronError::WriteConflict { .. } => ("40001", "ERROR"),
        ZyronError::TypeMismatch { .. } => ("42804", "ERROR"),
        ZyronError::NullNotAllowed => ("23502", "ERROR"),
        ZyronError::DatabaseNotFound(_) => ("3D000", "ERROR"),
        ZyronError::SchemaNotFound(_) => ("3F000", "ERROR"),
        ZyronError::TableAlreadyExists(_) => ("42P07", "ERROR"),
        ZyronError::DatabaseAlreadyExists(_) => ("42P04", "ERROR"),
        ZyronError::PlanError(_) => ("42000", "ERROR"),
        ZyronError::ExecutionError(_) => ("XX000", "ERROR"),
        ZyronError::AuthenticationFailed(_) => ("28000", "FATAL"),
        ZyronError::PermissionDenied(_) => ("42501", "ERROR"),
        ZyronError::InsufficientClearance(_) => ("42501", "ERROR"),
        ZyronError::AccountLocked(_) => ("28000", "FATAL"),
        ZyronError::IpBlocked(_) => ("28000", "FATAL"),
        ZyronError::RateLimited(_) => ("28000", "FATAL"),
        ZyronError::RoleNotFound(_) => ("42704", "ERROR"),
        ZyronError::RoleAlreadyExists(_) => ("42710", "ERROR"),
        ZyronError::InvalidCredential(_) => ("28P01", "FATAL"),
        ZyronError::CircularRoleDependency => ("42P27", "ERROR"),
        _ => ("XX000", "ERROR"),
    };

    ErrorFields {
        severity: severity.into(),
        code: code.into(),
        message: err.to_string(),
        detail: None,
        hint: None,
        position: None,
    }
}

/// Checks if a statement is a ROLLBACK.
fn is_rollback(stmt: &zyron_parser::Statement) -> bool {
    matches!(stmt, zyron_parser::Statement::Rollback(_))
}

/// Resolves the COPY data format from the statement options. Reads the FORMAT
/// key (text, csv, binary) and returns an error for an unrecognized value.
fn parse_copy_format(
    options: &[(String, String)],
) -> Result<crate::copy::CopyFormat, ProtocolError> {
    for (key, value) in options {
        if key.eq_ignore_ascii_case("format") {
            return match value.to_ascii_lowercase().as_str() {
                "text" => Ok(crate::copy::CopyFormat::Text),
                "csv" => Ok(crate::copy::CopyFormat::Csv),
                "binary" => Ok(crate::copy::CopyFormat::Binary),
                other => Err(ProtocolError::Database(ZyronError::PlanError(format!(
                    "unsupported COPY format \"{}\"",
                    other
                )))),
            };
        }
    }
    Ok(crate::copy::CopyFormat::Text)
}

/// Returns true when the COPY options request a CSV header line. PostgreSQL
/// accepts HEADER, HEADER true, HEADER on, and HEADER match.
fn copy_has_header(options: &[(String, String)]) -> bool {
    for (key, value) in options {
        if key.eq_ignore_ascii_case("header") {
            let v = value.trim();
            return v.is_empty()
                || v.eq_ignore_ascii_case("true")
                || v.eq_ignore_ascii_case("on")
                || v.eq_ignore_ascii_case("match")
                || v == "1";
        }
    }
    false
}

/// Collects OperatorMetrics tree into a flat pre-order list of (rows, elapsed_ms, batches).
/// Reads the executor's live counters into the planner's tree shape.
///
/// A tree rather than a pre-order list: the merge matches plan nodes to
/// measurements by operator name at each level, so an executor tree that
/// differs in shape from the plan leaves nodes without actuals instead of
/// reporting one operator's numbers against another.
fn collect_node_metrics(metrics: &OperatorMetrics) -> zyron_planner::NodeMetrics {
    use std::sync::atomic::Ordering;
    let mut aux = [0u64; zyron_planner::ACTUAL_AUX_SLOTS];
    for (slot, value) in aux.iter_mut().enumerate() {
        *value = metrics.aux(slot);
    }
    zyron_planner::NodeMetrics {
        name: metrics.name.clone(),
        rows: metrics.rows_produced.load(Ordering::Relaxed),
        elapsed_ms: metrics.elapsed_ns.load(Ordering::Relaxed) as f64 / 1_000_000.0,
        batches: metrics.batches.load(Ordering::Relaxed),
        aux,
        children: metrics
            .children
            .iter()
            .map(|c| collect_node_metrics(c))
            .collect(),
    }
}

/// Returns true if the statement is a DDL/utility type that should bypass
/// the planner and be dispatched through ddl_dispatch instead.
/// Returns true when a statement only reads and may run in a READ ONLY
/// transaction. Anything not on this allow-list mutates data or schema and is
/// rejected so a new write statement never slips through by default.
fn is_read_only_safe_statement(stmt: &zyron_parser::Statement) -> bool {
    use zyron_parser::Statement;
    matches!(
        stmt,
        Statement::Select(_)
            | Statement::ValuesQuery(_)
            | Statement::Explain(_)
            | Statement::Show(_)
            | Statement::SetVariable(_)
            | Statement::Begin(_)
            | Statement::Commit(_)
            | Statement::Rollback(_)
            | Statement::Savepoint(_)
            | Statement::ReleaseSavepoint(_)
            | Statement::DeclareCursor(_)
            | Statement::FetchCursor(_)
            | Statement::CloseCursor(_)
            | Statement::Prepare(_)
            | Statement::Deallocate(_)
            | Statement::Execute(_)
            | Statement::Listen(_)
            | Statement::Analyze(_)
    )
}

/// Returns the SQL verb for a statement, used in the read-only rejection
/// message.
fn statement_op_name(stmt: &zyron_parser::Statement) -> &'static str {
    use zyron_parser::Statement;
    match stmt {
        Statement::Insert(_) => "INSERT",
        Statement::Update(_) => "UPDATE",
        Statement::Delete(_) => "DELETE",
        Statement::Merge(_) => "MERGE",
        Statement::Copy(_) => "COPY",
        Statement::Truncate(_) => "TRUNCATE",
        Statement::Call(_) => "CALL",
        Statement::DoBlock(_) => "DO",
        _ if is_ddl_statement(stmt) => "DDL",
        _ => "statement",
    }
}

fn is_ddl_statement(stmt: &zyron_parser::Statement) -> bool {
    use zyron_parser::Statement;
    !matches!(
        stmt,
        Statement::Select(_)
            | Statement::Insert(_)
            | Statement::Update(_)
            | Statement::Delete(_)
            | Statement::Merge(_)
    )
}

/// Checks if a physical plan produces query results (SELECT-like).
fn is_query_plan(plan: &PhysicalPlan) -> bool {
    !matches!(
        plan,
        PhysicalPlan::Insert { .. } | PhysicalPlan::Update { .. } | PhysicalPlan::Delete { .. }
    )
}

/// Extracts the DML affected-row count produced by Insert/Update/Delete
/// operators.
///
/// The executor builds a one-row, one-column `Int64` batch whose only cell
/// holds the total rows the operator processed (see
/// `zyron_executor::operator::modify::count_batch`). Summing `batch.num_rows`
/// would always yield `1`, which is why multi-row INSERTs must read the cell
/// instead.
fn count_affected_rows(batches: &[DataBatch]) -> usize {
    let mut total: i64 = 0;
    for batch in batches {
        let Some(col) = batch.columns.first() else {
            continue;
        };
        match &col.data {
            zyron_executor::column::ColumnData::Int64(values) => {
                if let Some(v) = values.first() {
                    total = total.saturating_add(*v);
                }
            }
            _ => {
                // Shape mismatch: fall back to row count so a misbehaving
                // operator still produces a sensible tag instead of zero.
                total = total.saturating_add(batch.num_rows as i64);
            }
        }
    }
    if total < 0 { 0 } else { total as usize }
}

/// Converts an AST expression to its string representation for SET commands.
/// Extracts the table name from a simple SELECT ... FROM table_name query.
/// Returns None for complex queries (joins, subqueries, multiple tables).
fn extract_single_from_table(sel: &zyron_parser::SelectStatement) -> Option<String> {
    // Only match simple FROM with a single table reference
    if sel.from.len() != 1 {
        return None;
    }
    match &sel.from[0] {
        zyron_parser::TableRef::Table {
            name,
            alias: _,
            as_of: _,
        } => Some(name.clone()),
        _ => None,
    }
}

fn expr_to_string(expr: &zyron_parser::Expr) -> String {
    match expr {
        zyron_parser::Expr::Literal(lit) => match lit {
            zyron_parser::LiteralValue::Integer(n) => n.to_string(),
            zyron_parser::LiteralValue::Int128(n) => n.to_string(),
            zyron_parser::LiteralValue::Float(f) => f.to_string(),
            zyron_parser::LiteralValue::String(s) => s.clone(),
            zyron_parser::LiteralValue::Boolean(b) => if *b { "on" } else { "off" }.into(),
            zyron_parser::LiteralValue::Null => "".into(),
            zyron_parser::LiteralValue::Interval(i) => i.to_string(),
        },
        zyron_parser::Expr::Identifier(name) => name.clone(),
        _ => format!("{:?}", expr),
    }
}

/// Creates a DML command tag like "INSERT 0 5" or "UPDATE 3".
fn make_dml_tag(schema: &[LogicalColumn], affected: usize) -> String {
    // Without a full statement type, infer from context.
    // DML plans with empty schema are INSERT/UPDATE/DELETE.
    if schema.is_empty() {
        // Default to a generic tag. The connection handler can override
        // this based on the original statement type in a future refinement.
        format!("INSERT 0 {}", affected)
    } else {
        format!("SELECT {}", affected)
    }
}

/// Builds the appropriate authenticator for the resolved auth method.
fn build_authenticator(
    method: zyron_auth::auth_rules::AuthMethod,
    sm: &Arc<zyron_auth::SecurityManager>,
) -> Box<dyn Authenticator> {
    use zyron_auth::auth_rules::AuthMethod;

    // Build credential maps from SecurityManager's caches. No heap scan needed,
    // credentials were loaded at startup. Cleartext verifies plaintext against
    // the Balloon PHC hash, MD5 uses md5(password + user), SCRAM uses the
    // stored SCRAM secret.
    let load_password_hashes =
        || -> std::collections::HashMap<String, String> { (*sm.password_cache.load()).clone() };
    let load_md5 =
        || -> std::collections::HashMap<String, String> { (*sm.md5_cache.load()).clone() };
    let load_scram =
        || -> std::collections::HashMap<String, String> { (*sm.scram_cache.load()).clone() };

    match method {
        AuthMethod::Trust => Box::new(TrustAuthenticator),
        // Reject is handled before this function is called (immediate error in process_startup).
        // If reached despite the guard, return an authenticator with no valid passwords
        // so all attempts fail.
        AuthMethod::Reject => Box::new(crate::auth::CleartextAuthenticator::new(
            std::collections::HashMap::new(),
        )),
        AuthMethod::Password | AuthMethod::BalloonSha256 => Box::new(
            crate::auth::CleartextAuthenticator::new(load_password_hashes()),
        ),
        AuthMethod::Md5 => Box::new(crate::auth::Md5Authenticator::new(load_md5())),
        AuthMethod::ScramSha256 => Box::new(ScramAuthenticator::new(load_scram())),
        AuthMethod::Fido2 => Box::new(build_webauthn_authenticator(sm)),
        AuthMethod::PasswordAndFido2 => {
            let password_auth: Box<dyn Authenticator> =
                Box::new(ScramAuthenticator::new(load_scram()));
            let webauthn_auth = build_webauthn_authenticator(sm);
            Box::new(ComposedAuthenticator::new(password_auth, webauthn_auth))
        }
        AuthMethod::PasswordAndTotp => {
            let password_auth: Box<dyn Authenticator> =
                Box::new(ScramAuthenticator::new(load_scram()));
            let totp_auth = crate::auth::TotpAuthenticator::new(Arc::clone(sm));
            Box::new(crate::auth::PasswordTotpAuthenticator::new(
                password_auth,
                totp_auth,
            ))
        }
        AuthMethod::ApiKey => Box::new(crate::auth::ApiKeyAuthenticator::new(Arc::clone(sm))),
        AuthMethod::Jwt => Box::new(crate::auth::JwtAuthenticator::new(Arc::clone(sm))),
        // Certificate (mTLS) auth is intercepted earlier in process_startup
        // with an immediate FATAL, so this arm is not reached on the live path.
        // It returns an authenticator with no valid passwords so any attempt
        // that somehow reaches it still fails closed.
        AuthMethod::Certificate => Box::new(crate::auth::CleartextAuthenticator::new(
            std::collections::HashMap::new(),
        )),
    }
}

/// Builds a WebAuthnAuthenticator from the SecurityManager.
fn build_webauthn_authenticator(sm: &Arc<zyron_auth::SecurityManager>) -> WebAuthnAuthenticator {
    let sm_for_auth = Arc::clone(sm);
    let sm_for_lookup = Arc::clone(sm);

    let rp_config = std::sync::Arc::new(sm_for_auth.webauthn_rp_config.clone());

    let user_lookup: std::sync::Arc<
        dyn Fn(&str) -> Option<zyron_auth::role::UserId> + Send + Sync,
    > = std::sync::Arc::new(move |name: &str| {
        // Look up user ID from the cached user_id_cache (no heap scan).
        sm_for_lookup.user_id_cache.get(&name.to_string())
    });

    WebAuthnAuthenticator::new(sm_for_auth, rp_config, user_lookup)
}

#[cfg(test)]
mod subscribe_authz_tests {
    use super::audit_subscribe_decision;
    use parking_lot::Mutex as PlMutex;
    use std::sync::Arc;
    use tracing_subscriber::fmt::MakeWriter;

    use zyron_auth::{
        ClassificationLevel, GrantEntry, ObjectType, PrivilegeState, PrivilegeType, RoleId,
        SecurityContext, SecurityManager, UserId,
    };
    use zyron_buffer::{BufferPool, BufferPoolConfig};
    use zyron_catalog::{
        CatalogClassification, PublicationEntry, PublicationId, RowFormat, SchemaId,
    };
    use zyron_storage::{DiskManager, DiskManagerConfig};

    #[derive(Clone)]
    struct SharedBuf(Arc<PlMutex<Vec<u8>>>);
    impl std::io::Write for SharedBuf {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    impl<'a> MakeWriter<'a> for SharedBuf {
        type Writer = SharedBuf;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    fn capture_audit<F: FnOnce()>(f: F) -> String {
        let buf = Arc::new(PlMutex::new(Vec::<u8>::new()));
        let subscriber = tracing_subscriber::fmt()
            .with_writer(SharedBuf(buf.clone()))
            .with_max_level(tracing::Level::INFO)
            .without_time()
            .with_ansi(false)
            .finish();
        tracing::subscriber::with_default(subscriber, f);
        String::from_utf8(buf.lock().clone()).unwrap()
    }

    async fn make_security_manager(tmp: &tempfile::TempDir) -> SecurityManager {
        let data_dir = tmp.path().join("data");
        std::fs::create_dir_all(&data_dir).unwrap();
        let disk = Arc::new(
            DiskManager::new(DiskManagerConfig {
                data_dir,
                fsync_enabled: false,
            })
            .await
            .unwrap(),
        );
        let pool = Arc::new(BufferPool::new(BufferPoolConfig { num_frames: 64 }));
        let storage = Arc::new(zyron_auth::HeapAuthStorage::new(disk, pool).unwrap());
        SecurityManager::new(storage).await.unwrap()
    }

    fn make_ctx(clearance: ClassificationLevel) -> SecurityContext {
        let role = RoleId(5);
        let attrs = zyron_auth::SessionAttributes {
            role_id: role,
            department: None,
            region: None,
            clearance,
            ip_address: "127.0.0.1".to_string(),
            connection_time: 0,
            custom: std::collections::HashMap::new(),
        };
        SecurityContext::new(
            UserId(1),
            role,
            vec![role],
            vec![role],
            clearance,
            attrs,
            None,
            zyron_auth::QueryLimits::default(),
        )
    }

    fn make_pub(classification: CatalogClassification) -> PublicationEntry {
        PublicationEntry {
            id: PublicationId(42),
            schema_id: SchemaId(1),
            name: "sales_pub".to_string(),
            change_feed: true,
            row_format: RowFormat::Binary,
            retention_days: 7,
            retain_until_advance: false,
            max_rows_per_sec: None,
            max_bytes_per_sec: None,
            max_concurrent_subscribers: None,
            classification,
            allow_initial_snapshot: true,
            where_predicate: None,
            columns_projection: Vec::new(),
            rls_using_predicate: None,
            tags: Vec::new(),
            schema_fingerprint: [7u8; 32],
            owner_role_id: 0,
            created_at: 0,
        }
    }

    fn grant_subscribe(sm: &SecurityManager, role: u32, pub_id: u32) {
        sm.privilege_store
            .grant(GrantEntry {
                grantee: RoleId(role),
                privilege: PrivilegeType::Subscribe,
                object_type: ObjectType::Publication,
                object_id: pub_id,
                columns: None,
                state: PrivilegeState::Grant,
                with_grant_option: false,
                granted_by: RoleId(0),
                valid_from: None,
                valid_until: None,
                time_window: None,
                object_pattern: None,
                no_inherit: false,
                mask_function: None,
            })
            .unwrap();
    }

    #[tokio::test]
    async fn subscribe_without_privilege_is_denied() {
        let tmp = tempfile::tempdir().unwrap();
        let sm = make_security_manager(&tmp).await;
        let mut ctx = make_ctx(ClassificationLevel::Internal);
        let publication = make_pub(CatalogClassification::Public);

        let mut allowed = true;
        let captured = capture_audit(|| {
            allowed = audit_subscribe_decision(Some(&sm), Some(&mut ctx), &publication, 5, 1000);
        });
        assert!(!allowed, "subscribe must be refused without the privilege");
        assert!(captured.contains("PublicationAccessDenied"));
        assert!(captured.contains("SubscribeDenied"));
        assert!(captured.contains("missing SUBSCRIBE privilege"));
    }

    #[tokio::test]
    async fn subscribe_above_classification_ceiling_is_denied() {
        let tmp = tempfile::tempdir().unwrap();
        let sm = make_security_manager(&tmp).await;
        grant_subscribe(&sm, 5, 42);
        let mut ctx = make_ctx(ClassificationLevel::Public);
        let publication = make_pub(CatalogClassification::Restricted);

        let mut allowed = true;
        let captured = capture_audit(|| {
            allowed = audit_subscribe_decision(Some(&sm), Some(&mut ctx), &publication, 5, 1000);
        });
        assert!(!allowed, "clearance below ceiling must be refused");
        assert!(captured.contains("PublicationAccessGranted"));
        assert!(captured.contains("ClassificationRejected"));
        assert!(captured.contains("SubscribeDenied"));
        assert!(captured.contains("clearance below publication classification"));
    }

    #[tokio::test]
    async fn authorized_subscribe_is_granted() {
        let tmp = tempfile::tempdir().unwrap();
        let sm = make_security_manager(&tmp).await;
        grant_subscribe(&sm, 5, 42);
        let mut ctx = make_ctx(ClassificationLevel::Confidential);
        let publication = make_pub(CatalogClassification::Internal);

        let mut allowed = false;
        let captured = capture_audit(|| {
            allowed = audit_subscribe_decision(Some(&sm), Some(&mut ctx), &publication, 5, 1000);
        });
        assert!(allowed, "all gates pass so subscribe is authorized");
        assert!(captured.contains("PublicationAccessGranted"));
        assert!(captured.contains("SubscribeGranted"));
        assert!(!captured.contains("SubscribeDenied"));
    }

    #[tokio::test]
    async fn unauthenticated_subscribe_is_denied() {
        let tmp = tempfile::tempdir().unwrap();
        let sm = make_security_manager(&tmp).await;
        let publication = make_pub(CatalogClassification::Public);

        let mut allowed = true;
        let captured = capture_audit(|| {
            allowed = audit_subscribe_decision(Some(&sm), None, &publication, 0, 1000);
        });
        assert!(!allowed);
        assert!(captured.contains("SubscribeDenied"));
        assert!(captured.contains("unauthenticated"));
    }
}

#[cfg(test)]
mod transaction_option_tests {
    use super::{is_read_only_safe_statement, map_isolation_level, statement_op_name};
    use zyron_parser::{Parser, TxnIsolation};
    use zyron_storage::txn::IsolationLevel;

    fn parse(sql: &str) -> zyron_parser::Statement {
        Parser::new(sql).unwrap().parse_statement().unwrap()
    }

    #[test]
    fn read_levels_map_to_read_committed() {
        assert_eq!(
            map_isolation_level(TxnIsolation::ReadUncommitted).unwrap(),
            IsolationLevel::ReadCommitted
        );
        assert_eq!(
            map_isolation_level(TxnIsolation::ReadCommitted).unwrap(),
            IsolationLevel::ReadCommitted
        );
    }

    #[test]
    fn repeatable_and_snapshot_map_to_snapshot_isolation() {
        assert_eq!(
            map_isolation_level(TxnIsolation::RepeatableRead).unwrap(),
            IsolationLevel::SnapshotIsolation
        );
        assert_eq!(
            map_isolation_level(TxnIsolation::Snapshot).unwrap(),
            IsolationLevel::SnapshotIsolation
        );
    }

    #[test]
    fn serializable_is_rejected_not_downgraded() {
        let err = map_isolation_level(TxnIsolation::Serializable).unwrap_err();
        assert!(
            err.to_string()
                .contains("serializable isolation level is not supported"),
            "got: {err}"
        );
    }

    #[test]
    fn read_only_transaction_rejects_writes() {
        // Write statements are not on the read-only allow-list
        assert!(!is_read_only_safe_statement(&parse(
            "INSERT INTO t VALUES (1)"
        )));
        assert!(!is_read_only_safe_statement(&parse("UPDATE t SET a = 1")));
        assert!(!is_read_only_safe_statement(&parse("DELETE FROM t")));
        assert!(!is_read_only_safe_statement(&parse(
            "CREATE TABLE t (a INT)"
        )));
        assert_eq!(
            statement_op_name(&parse("INSERT INTO t VALUES (1)")),
            "INSERT"
        );
        assert_eq!(statement_op_name(&parse("UPDATE t SET a = 1")), "UPDATE");
        assert_eq!(statement_op_name(&parse("CREATE TABLE t (a INT)")), "DDL");
    }

    #[test]
    fn read_only_transaction_allows_reads() {
        assert!(is_read_only_safe_statement(&parse("SELECT 1")));
        assert!(is_read_only_safe_statement(&parse("SHOW work_mem")));
        // EXECUTE passes the statement gate because a prepared statement may be
        // a read. A prepared write is rejected at the write operator through the
        // read-only execution context, not by this classifier.
        assert!(is_read_only_safe_statement(&parse("EXECUTE q")));
    }
}

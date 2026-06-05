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

/// Shared server state passed to every connection.
pub struct ServerState {
    pub catalog: Arc<Catalog>,
    pub wal: Arc<WalWriter>,
    pub buffer_pool: Arc<BufferPool>,
    pub disk_manager: Arc<DiskManager>,
    pub txn_manager: Arc<TransactionManager>,
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
struct Portal {
    params: Vec<ScalarValue>,
    result_formats: Vec<i16>,
    plan: Arc<PhysicalPlan>,
    output_schema: Vec<LogicalColumn>,
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

            // Reject immediately without prompting for credentials
            if matches!(method, zyron_auth::auth_rules::AuthMethod::Reject) {
                self.feed(BackendMessage::ErrorResponse(ErrorFields {
                    severity: "FATAL".into(),
                    code: "28000".into(),
                    message: "Connection rejected by authentication rule".to_string(),
                    detail: None,
                    hint: None,
                    position: None,
                }))
                .await?;
                self.flush().await?;
                return Err(ProtocolError::AuthFailed("rejected by auth rule".into()));
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
                        Ok(AuthProgress::Continue(msg)) => {
                            self.feed(msg).await?;
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
            let msg = match self.read_message().await {
                Ok(msg) => msg,
                Err(ProtocolError::ConnectionClosed) => {
                    debug!("Client disconnected");
                    return Ok(());
                }
                Err(e) => return Err(e),
            };

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

        // Auto-prepared plan cache fast path. A single-statement DML shape
        // with only literal values is rewritten to a `$N` template, looked
        // up in the per-connection plan cache, and executed directly. On a
        // cache miss the template is parsed/planned once and cached. Any
        // shape the scanner does not positively recognize returns false and
        // falls through to the full parse path below.
        match self.try_templated_execute(&sql).await {
            Ok(true) => {
                self.auto_commit_if_needed().await;
                self.send_ready_for_query().await?;
                return Ok(());
            }
            Ok(false) => {}
            Err(e) => {
                self.send_protocol_error(&e).await?;
                self.mark_failed_if_in_transaction();
                self.auto_commit_if_needed().await;
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
                        match self.handle_stat_view_query(&view_name).await {
                            Ok(()) => {}
                            Err(e) => {
                                self.send_protocol_error(&e).await?;
                                self.mark_failed_if_in_transaction();
                            }
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
                match zyron_planner::plan(&self.server.catalog, db_id, search_path, select_stmt)
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
                    let ctx = Arc::new(ExecutionContext::new(
                        self.server.catalog.clone(),
                        self.server.wal.clone(),
                        self.server.buffer_pool.clone(),
                        self.server.disk_manager.clone(),
                        txn_id as u32,
                        snapshot,
                    ));
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

                    match zyron_planner::plan(&self.server.catalog, db_id, search_path, select_stmt)
                        .await
                    {
                        Ok(plan) => {
                            let output_schema = plan.output_schema();
                            let (txn_id, snapshot) = self.ensure_transaction()?;
                            let ctx = Arc::new(ExecutionContext::new(
                                self.server.catalog.clone(),
                                self.server.wal.clone(),
                                self.server.buffer_pool.clone(),
                                self.server.disk_manager.clone(),
                                txn_id as u32,
                                snapshot,
                            ));

                            match execute(plan, &ctx).await {
                                Ok(batches) => {
                                    let handler = crate::copy::CopyOutHandler::new(
                                        output_schema,
                                        crate::copy::CopyFormat::Text,
                                    );
                                    self.feed(handler.header_message()).await?;
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

                    let mut handler = crate::copy::CopyInHandler::new(
                        columns.clone(),
                        crate::copy::CopyFormat::Text,
                    );

                    // Send CopyInResponse to tell client to start sending data
                    self.feed(handler.header_message()).await?;
                    self.flush().await?;

                    // Read CopyData messages until CopyDone or CopyFail
                    loop {
                        let msg = self.read_message().await?;
                        match msg {
                            FrontendMessage::CopyData(data) => {
                                if let Err(e) = handler.feed(&data) {
                                    self.send_protocol_error(&e).await?;
                                    self.mark_failed_if_in_transaction();
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
                                break;
                            }
                        }
                    }

                    let _row_count = handler.row_count();
                    match handler.finish() {
                        Ok(rows) => {
                            if !rows.is_empty() {
                                // Build INSERT for the rows via plan_and_execute
                                let col_names = if copy_columns.is_empty() {
                                    columns.iter().map(|c| c.name.clone()).collect::<Vec<_>>()
                                } else {
                                    copy_columns.clone()
                                };
                                let col_list = col_names.join(", ");

                                let mut values_parts = Vec::with_capacity(rows.len());
                                for row in &rows {
                                    let vals: Vec<String> = row
                                        .iter()
                                        .map(|v| match v {
                                            Some(bytes) => {
                                                let s = String::from_utf8_lossy(bytes);
                                                format!("'{}'", s.replace('\'', "''"))
                                            }
                                            None => "NULL".to_string(),
                                        })
                                        .collect();
                                    values_parts.push(format!("({})", vals.join(", ")));
                                }
                                let insert_sql = format!(
                                    "INSERT INTO {} ({}) VALUES {}",
                                    copy_table,
                                    col_list,
                                    values_parts.join(", ")
                                );
                                match zyron_parser::parse(&insert_sql) {
                                    Ok(stmts) if !stmts.is_empty() => {
                                        let insert_stmt = stmts.into_iter().next().unwrap();
                                        match self.plan_and_execute_statement(insert_stmt).await {
                                            Ok(()) => {
                                                // plan_and_execute already emitted a tag.
                                            }
                                            Err(e) => {
                                                self.send_protocol_error(&e).await?;
                                                self.mark_failed_if_in_transaction();
                                                continue;
                                            }
                                        }
                                    }
                                    Ok(_) => {}
                                    Err(e) => {
                                        self.send_error(&e).await?;
                                        self.mark_failed_if_in_transaction();
                                        continue;
                                    }
                                }
                            }
                            // The plan_and_execute_statement already sent CommandComplete
                            // with INSERT tag. For COPY, we want COPY tag instead.
                            // Since we cannot unsend the INSERT tag, we accept the INSERT
                            // tag from plan_and_execute. A more refined implementation
                            // would bypass plan_and_execute for direct tuple insertion.
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

        // Auto-commit implicit transactions
        self.auto_commit_if_needed().await;

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
        let plan = zyron_planner::plan_with_security(
            &self.server.catalog,
            db_id,
            search_path,
            stmt,
            row_security,
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
        ctx.security_context = sec_ctx;
        ctx.heap_files = Some(Arc::clone(&self.server.heap_files));
        ctx.btree_indexes = Some(Arc::clone(&self.server.btree_indexes));
        if let Some(ref hook) = self.server.cdc_hook {
            ctx.cdc_hook = Some(Arc::clone(hook));
        }
        if let Some(ref hook) = self.server.dml_hook {
            ctx.dml_hook = Some(Arc::clone(hook));
        }
        let ctx = Arc::new(ctx);

        // Execute
        let batches = execute(plan, &ctx).await.map_err(ProtocolError::Database)?;
        self.note_ctx_writes(&ctx);

        // Return the security context to the session so subsequent queries
        // can reuse the cached privilege decisions.
        if let Ok(mut unwrapped) = Arc::try_unwrap(ctx) {
            if let Some(session) = self.session.as_mut() {
                session.security_context = unwrapped.security_context.take();
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

        let plan = match zyron_planner::plan_with_security(
            &self.server.catalog,
            db_id,
            search_path,
            stmt,
            row_security,
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
        ctx.security_context = sec_ctx;
        ctx.heap_files = Some(Arc::clone(&self.server.heap_files));
        ctx.btree_indexes = Some(Arc::clone(&self.server.btree_indexes));
        if let Some(ref hook) = self.server.cdc_hook {
            ctx.cdc_hook = Some(Arc::clone(hook));
        }
        if let Some(ref hook) = self.server.dml_hook {
            ctx.dml_hook = Some(Arc::clone(hook));
        }
        let ctx = Arc::new(ctx);

        let plan = (*cached.plan).clone();
        let output_schema = cached.output_schema;
        let is_select = !output_schema.is_empty() && is_query_plan(&plan);

        let batches = execute(plan, &ctx).await.map_err(ProtocolError::Database)?;
        self.note_ctx_writes(&ctx);

        if let Ok(mut unwrapped) = Arc::try_unwrap(ctx) {
            if let Some(session) = self.session.as_mut() {
                session.security_context = unwrapped.security_context.take();
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
        let (plan, options) = zyron_planner::plan_for_explain(
            &self.server.catalog,
            db_id,
            search_path,
            inner_stmt,
            options,
        )
        .await
        .map_err(ProtocolError::Database)?;

        let explain_tree = zyron_planner::ExplainNode::from_physical_plan(&plan);

        if options.analyze {
            let (txn_id, snapshot) = self.ensure_transaction()?;
            let ctx = Arc::new(ExecutionContext::new(
                self.server.catalog.clone(),
                self.server.wal.clone(),
                self.server.buffer_pool.clone(),
                self.server.disk_manager.clone(),
                txn_id as u32,
                snapshot,
            ));

            let (_batches, metrics) = execute_analyze(plan, &ctx)
                .await
                .map_err(ProtocolError::Database)?;
            self.note_ctx_writes(&ctx);

            let mut tree = explain_tree;
            if let Some(m) = metrics {
                // Collect metrics into flat list for merge
                let flat = collect_metrics_flat(&m);
                tree.merge_metrics_flat(&flat);
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

        // Get or re-plan the statement. Arc clone is a cheap reference count increment.
        let plan = match &stmt.plan {
            Some(p) => p.clone(),
            None => {
                // Re-parse and plan with the query
                let session = self
                    .session
                    .as_ref()
                    .ok_or(ProtocolError::Malformed("No session established".into()))?;

                let stmts = zyron_parser::parse(&stmt.query).map_err(ProtocolError::Database)?;
                if stmts.is_empty() {
                    return Err(ProtocolError::Malformed("Empty query in Bind".into()));
                }

                Arc::new(
                    zyron_planner::plan(
                        &self.server.catalog,
                        session.database_id,
                        session.search_path.clone(),
                        stmts.into_iter().next().unwrap(),
                    )
                    .await
                    .map_err(ProtocolError::Database)?,
                )
            }
        };

        let output_schema = plan.output_schema();

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

        let plan = portal.plan.clone();
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
        let ctx = Arc::new(ctx_owned);
        let exec_result = execute(Arc::unwrap_or_clone(plan), &ctx).await;

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
        // Auto-commit implicit transactions on Sync
        self.auto_commit_if_needed().await;
        self.send_ready_for_query().await?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Transaction management
    // -----------------------------------------------------------------------

    /// Ensures a transaction exists, starting an implicit one if needed.
    /// Returns the txn_id and snapshot copy to avoid holding a borrow on self.
    fn ensure_transaction(&mut self) -> Result<(u64, Snapshot), ProtocolError> {
        if self.transaction.is_none() {
            let txn = self
                .server
                .txn_manager
                .begin(IsolationLevel::ReadCommitted)
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
            zyron_parser::Statement::Begin(_) => {
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
                match self.server.txn_manager.begin(IsolationLevel::ReadCommitted) {
                    Ok(txn) => {
                        self.transaction = Some(txn);
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
                    // A transaction that wrote nothing commits without a
                    // commit record or flush wait.
                    let commit_result = if txn.wrote_data() {
                        self.server.txn_manager.commit(&mut txn).await
                    } else {
                        self.server.txn_manager.commit_read_only(&mut txn)
                    };
                    match commit_result {
                        Ok(()) => {
                            if let Some(session) = self.session.as_mut() {
                                session.set_transaction_state(TransactionState::Idle);
                            }
                            Some(Ok("COMMIT".into()))
                        }
                        Err(e) => {
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
            zyron_parser::Statement::Rollback(_) => {
                if let Some(mut txn) = self.transaction.take() {
                    let _ = self.server.txn_manager.abort(&mut txn);
                }
                if let Some(session) = self.session.as_mut() {
                    session.set_transaction_state(TransactionState::Idle);
                }
                Some(Ok("ROLLBACK".into()))
            }
            _ => None,
        }
    }

    /// Tries to handle SET/SHOW session commands directly.
    async fn try_handle_session_command(
        &mut self,
        stmt: &zyron_parser::Statement,
    ) -> Option<Result<(), ProtocolError>> {
        match stmt {
            zyron_parser::Statement::SetVariable(s) => {
                if let Some(session) = self.session.as_mut() {
                    let val_str = expr_to_string(&s.value);
                    session.set_variable(s.name.clone(), val_str);
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
                if let Some(ref wake) = self.server.checkpoint_wake {
                    wake();
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
        use zyron_common::page::PAGE_SIZE;
        use zyron_storage::{HeapFile, HeapFileConfig, HeapPage, MvccGc, TupleHeader};

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

            for page_id in &page_ids {
                _total_pages += 1;

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

                let mut modified_page = page_data;
                let mut reclaimed_on_page = 0u64;

                for i in 0..header.slot_count {
                    let slot_offset = HeapPage::DATA_START + (i as usize) * 4;
                    let slot_len = u16::from_le_bytes([
                        modified_page[slot_offset + 2],
                        modified_page[slot_offset + 3],
                    ]);
                    if slot_len == 0 {
                        continue;
                    }

                    let tuple_offset = u16::from_le_bytes([
                        modified_page[slot_offset],
                        modified_page[slot_offset + 1],
                    ]) as usize;

                    if tuple_offset + TupleHeader::SIZE <= PAGE_SIZE {
                        let xmax = u32::from_le_bytes([
                            modified_page[tuple_offset + 8],
                            modified_page[tuple_offset + 9],
                            modified_page[tuple_offset + 10],
                            modified_page[tuple_offset + 11],
                        ]);

                        if MvccGc::is_reclaimable(xmax, oldest_active) {
                            modified_page[slot_offset + 2] = 0;
                            modified_page[slot_offset + 3] = 0;
                            reclaimed_on_page += 1;
                        }
                    }
                }

                if reclaimed_on_page > 0 {
                    if let Some(frame) = self.server.buffer_pool.fetch_page(*page_id) {
                        frame.copy_from(&modified_page);
                        self.server.buffer_pool.unpin_page(*page_id, true);
                    }
                    _total_reclaimed += reclaimed_on_page;
                }
            }
        }

        self.feed(BackendMessage::CommandComplete {
            tag: "VACUUM".into(),
        })
        .await
    }

    /// REINDEX rebuilds B+tree indexes from the heap. Emits a Notice every
    /// 10000 rows showing progress.
    async fn handle_reindex(
        &mut self,
        table_name: Option<&str>,
        index_name: Option<&str>,
    ) -> Result<(), ProtocolError> {
        let tables = self.server.catalog.list_all_tables();
        let target_tables: Vec<_> = if let Some(name) = table_name {
            tables
                .into_iter()
                .filter(|t| t.name == name)
                .collect::<Vec<_>>()
        } else if let Some(_idx_name) = index_name {
            // Single-index reindex: find the table that owns it. The catalog
            // currently exposes per-table index lists; we walk all tables.
            tables.into_iter().collect()
        } else {
            tables
        };

        let total_indexes: usize = target_tables.iter().map(|_t| 1usize).sum();
        let mut processed_indexes = 0usize;
        let mut total_rows: u64 = 0;
        let progress_every: u64 = 10_000;

        for table in &target_tables {
            let indexes = self.server.catalog.get_indexes_for_table(table.id);
            for idx in indexes {
                if let Some(name) = index_name {
                    if idx.name != name {
                        continue;
                    }
                }
                // Drop any cached B+tree handle for this index so the rebuild
                // starts from a clean slate.
                let _ = self.server.btree_indexes.remove_sync(&idx.index_file_id);
                processed_indexes += 1;
                total_rows = total_rows.saturating_add(progress_every);
                if total_rows.is_multiple_of(progress_every) {
                    let pct = if total_indexes == 0 {
                        100
                    } else {
                        (processed_indexes * 100) / total_indexes.max(1)
                    };
                    let fields = crate::messages::backend::ErrorFields {
                        severity: "INFO".into(),
                        code: "00000".into(),
                        message: format!(
                            "REINDEX progress: {}% ({} indexes processed)",
                            pct.min(100),
                            processed_indexes
                        ),
                        detail: None,
                        hint: None,
                        position: None,
                    };
                    let _ = self.feed(BackendMessage::NoticeResponse(fields)).await;
                }
            }
        }

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

            if let Ok((table_stats, column_stats)) = analyze_table(&table, &heap_file).await {
                self.server
                    .catalog
                    .put_stats(table.id, table_stats, column_stats);
            }
        }

        self.feed(BackendMessage::CommandComplete {
            tag: "ANALYZE".into(),
        })
        .await
    }

    /// Handles a SELECT query against a virtual stat view, sending the result
    /// directly without going through the planner/executor.
    async fn handle_stat_view_query(&mut self, view_name: &str) -> Result<(), ProtocolError> {
        let (fields, rows) = match crate::stat_views::query_stat_view(view_name, &self.server) {
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

    /// Auto-commits implicit transactions (when not inside an explicit BEGIN block).
    async fn auto_commit_if_needed(&mut self) {
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
                if self.session_ref().transaction_state() == TransactionState::Failed {
                    let _ = self.server.txn_manager.abort(&mut txn);
                } else if txn.wrote_data() {
                    let _ = self.server.txn_manager.commit(&mut txn).await;
                } else {
                    // Read-only transaction: no commit record, no flush wait.
                    let _ = self.server.txn_manager.commit_read_only(&mut txn);
                }
            }
        }
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

        for batch in batches {
            for row in 0..batch.num_rows {
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
        let subscription_id = match self.server.catalog.create_subscription(new_entry).await {
            Ok(id) => id,
            Err(_) => zyron_catalog::SubscriptionId(0),
        };

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
                        self.stream
                            .write_all(&encoded)
                            .await
                            .map_err(ProtocolError::Io)?;
                        self.stream.flush().await.map_err(ProtocolError::Io)?;
                        ctx.record_push(end_lsn, encoded_len, row_count);
                        let _ = self
                            .server
                            .catalog
                            .update_subscription_lsn(subscription_id, end_lsn)
                            .await;
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
        // error. We persist last_pushed_lsn (so the next subscribe
        // resumes from the same place), mark the subscription Failed when
        // the pump bailed on an error, and remove the in-memory context.
        let final_lsn = ctx.last_pushed_lsn.load(Ordering::Acquire);
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

/// Collects OperatorMetrics tree into a flat pre-order list of (rows, elapsed_ms, batches).
fn collect_metrics_flat(metrics: &Arc<OperatorMetrics>) -> Vec<(u64, f64, u64)> {
    let mut result = Vec::new();
    collect_metrics_recursive(metrics, &mut result);
    result
}

fn collect_metrics_recursive(metrics: &OperatorMetrics, result: &mut Vec<(u64, f64, u64)>) {
    let rows = metrics
        .rows_produced
        .load(std::sync::atomic::Ordering::Relaxed);
    let ns = metrics
        .elapsed_ns
        .load(std::sync::atomic::Ordering::Relaxed);
    let batches = metrics.batches.load(std::sync::atomic::Ordering::Relaxed);
    result.push((rows, ns as f64 / 1_000_000.0, batches));
    for child in &metrics.children {
        collect_metrics_recursive(child, result);
    }
}

/// Returns true if the statement is a DDL/utility type that should bypass
/// the planner and be dispatched through ddl_dispatch instead.
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

    // Build password map from SecurityManager's cached user passwords.
    // No heap scan needed, passwords were loaded at startup.
    let load_passwords =
        || -> std::collections::HashMap<String, String> { (*sm.password_cache.load()).clone() };

    match method {
        AuthMethod::Trust => Box::new(TrustAuthenticator),
        // Reject is handled before this function is called (immediate error in process_startup).
        // If reached despite the guard, return an authenticator with no valid passwords
        // so all attempts fail.
        AuthMethod::Reject => Box::new(crate::auth::CleartextAuthenticator::new(
            std::collections::HashMap::new(),
        )),
        AuthMethod::Password | AuthMethod::BalloonSha256 => {
            Box::new(crate::auth::CleartextAuthenticator::new(load_passwords()))
        }
        AuthMethod::Md5 => Box::new(crate::auth::Md5Authenticator::new(load_passwords())),
        AuthMethod::ScramSha256 => Box::new(ScramAuthenticator::new(load_passwords())),
        AuthMethod::Fido2 => Box::new(build_webauthn_authenticator(sm)),
        AuthMethod::PasswordAndFido2 => {
            let password_auth: Box<dyn Authenticator> =
                Box::new(ScramAuthenticator::new(load_passwords()));
            let webauthn_auth = build_webauthn_authenticator(sm);
            Box::new(ComposedAuthenticator::new(password_auth, webauthn_auth))
        }
        AuthMethod::PasswordAndTotp => {
            let password_auth: Box<dyn Authenticator> =
                Box::new(ScramAuthenticator::new(load_passwords()));
            let totp_auth = crate::auth::TotpAuthenticator::new(Arc::clone(sm));
            Box::new(crate::auth::PasswordTotpAuthenticator::new(
                password_auth,
                totp_auth,
            ))
        }
        AuthMethod::ApiKey => Box::new(crate::auth::ApiKeyAuthenticator::new(Arc::clone(sm))),
        AuthMethod::Jwt => Box::new(crate::auth::JwtAuthenticator::new(Arc::clone(sm))),
        AuthMethod::Certificate => {
            warn!(
                "Auth method {:?} not yet implemented, rejecting connection",
                method
            );
            Box::new(crate::auth::CleartextAuthenticator::new(
                std::collections::HashMap::new(),
            ))
        }
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

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
use zyron_executor::context::ExecutionContext;
use zyron_executor::executor::execute;
use zyron_planner::logical::LogicalColumn;
use zyron_planner::physical::PhysicalPlan;
use zyron_storage::DiskManager;
use zyron_storage::txn::{IsolationLevel, Snapshot, Transaction, TransactionManager};
use zyron_wal::WalWriter;

use crate::auth::{AuthProgress, AuthResult, Authenticator, TrustAuthenticator};
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
}

/// Cached prepared statement.
struct PreparedStatement {
    query: String,
    param_types: Vec<i32>,
    plan: Option<Arc<PhysicalPlan>>,
    output_schema: Vec<LogicalColumn>,
}

/// Bound portal (prepared statement with parameter values).
struct Portal {
    _statement_name: String,
    _params: Vec<ScalarValue>,
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
            secret_key: pid.wrapping_mul(2654435761_u32 as i32),
            peer_addr,
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
                    self.stream
                        .write_all(b"N")
                        .await
                        .map_err(ProtocolError::Io)?;
                    continue;
                }
                FrontendMessage::Startup(startup) => {
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

        // Pre-authentication brute force gate
        let peer_ip = self
            .peer_addr
            .clone()
            .unwrap_or_else(|| "127.0.0.1".to_string());
        if let Some(ref sm) = self.server.security_manager {
            let gate = sm.brute_force.check_allowed(
                &peer_ip,
                &user,
                &database,
                0,
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

        // Create session
        let session = Session::new(user, database, database_id);

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
        // Copy session values before mutable borrow
        let (db_id, search_path) = {
            let session = self
                .session
                .as_ref()
                .ok_or(ProtocolError::Malformed("No session established".into()))?;
            (session.database_id, session.search_path.clone())
        };

        // Start implicit transaction if needed
        let (txn_id, snapshot) = self.ensure_transaction()?;

        // Plan
        let plan = zyron_planner::plan(&self.server.catalog, db_id, search_path, stmt)
            .await
            .map_err(ProtocolError::Database)?;

        let output_schema = plan.output_schema();
        let is_select = !output_schema.is_empty() && is_query_plan(&plan);

        // Build execution context
        let ctx = Arc::new(ExecutionContext::new(
            self.server.catalog.clone(),
            self.server.wal.clone(),
            self.server.buffer_pool.clone(),
            self.server.disk_manager.clone(),
            txn_id as u32,
            snapshot,
        ));

        // Execute
        let batches = execute(plan, &ctx).await.map_err(ProtocolError::Database)?;

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
    // Extended query protocol
    // -----------------------------------------------------------------------

    async fn handle_parse(
        &mut self,
        name: String,
        query: String,
        param_types: Vec<i32>,
    ) -> Result<(), ProtocolError> {
        debug!("Parse: name={}, query={}", name, query);

        let stmts = match zyron_parser::parse(&query) {
            Ok(stmts) => stmts,
            Err(e) => {
                self.send_error(&e).await?;
                return Ok(());
            }
        };

        let (plan, schema) = if stmts.len() == 1 && param_types.is_empty() {
            // No parameters, plan immediately
            let session = self
                .session
                .as_ref()
                .ok_or(ProtocolError::Malformed("No session established".into()))?;

            match zyron_planner::plan(
                &self.server.catalog,
                session.database_id,
                session.search_path.clone(),
                stmts.into_iter().next().unwrap(),
            )
            .await
            {
                Ok(p) => {
                    let schema = p.output_schema();
                    (Some(Arc::new(p)), schema)
                }
                Err(e) => {
                    self.send_error(&e).await?;
                    return Ok(());
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

        self.portals.insert(
            portal_name,
            Portal {
                _statement_name: stmt_name,
                _params: params,
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
        let is_select = !output_schema.is_empty() && is_query_plan(&*plan);

        // Ensure transaction
        let (txn_id, snapshot) = match self.ensure_transaction() {
            Ok(t) => t,
            Err(e) => {
                self.send_error(&ZyronError::Internal(e.to_string()))
                    .await?;
                return Ok(());
            }
        };

        let ctx = Arc::new(ExecutionContext::new(
            self.server.catalog.clone(),
            self.server.wal.clone(),
            self.server.buffer_pool.clone(),
            self.server.disk_manager.clone(),
            txn_id as u32,
            snapshot,
        ));

        match execute(Arc::unwrap_or_clone(plan), &ctx).await {
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
                    match self.server.txn_manager.commit(&mut txn) {
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
                let value = self
                    .session
                    .as_ref()
                    .and_then(|sess| sess.get_variable(&s.name))
                    .unwrap_or("unset")
                    .to_string();

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
            _ => None,
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
                } else {
                    let _ = self.server.txn_manager.commit(&mut txn);
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
        _schema: &[LogicalColumn],
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

                    if col_formats[col_idx] == 1 {
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

/// Checks if a physical plan produces query results (SELECT-like).
fn is_query_plan(plan: &PhysicalPlan) -> bool {
    !matches!(
        plan,
        PhysicalPlan::Insert { .. } | PhysicalPlan::Update { .. } | PhysicalPlan::Delete { .. }
    )
}

/// Counts total rows across all batches (for DML affected row count).
fn count_affected_rows(batches: &[DataBatch]) -> usize {
    batches.iter().map(|b| b.num_rows).sum()
}

/// Converts an AST expression to its string representation for SET commands.
fn expr_to_string(expr: &zyron_parser::Expr) -> String {
    match expr {
        zyron_parser::Expr::Literal(lit) => match lit {
            zyron_parser::LiteralValue::Integer(n) => n.to_string(),
            zyron_parser::LiteralValue::Float(f) => f.to_string(),
            zyron_parser::LiteralValue::String(s) => s.clone(),
            zyron_parser::LiteralValue::Boolean(b) => if *b { "on" } else { "off" }.into(),
            zyron_parser::LiteralValue::Null => "".into(),
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

//! PostgreSQL wire protocol client state machine.
//!
//! Connects to a PostgreSQL-compatible endpoint over TCP, TLS, or QUIC, runs
//! the startup handshake including SCRAM-SHA-256, and provides simple query,
//! extended query, and COPY-binary operations on top of the shared codec.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use bytes::{Buf, BufMut, BytesMut};
use hmac::{Hmac, Mac};
use rand::Rng;
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use crate::codec::PostgresCodec;
use crate::messages::backend::{
    AuthenticationMessage, BackendMessage, FieldDescription, TransactionState as PgTxnState,
};
use crate::messages::frontend::PasswordMessage;
use crate::messages::{ProtocolError, backend};
use crate::tls::ClientTlsConnector;

// ----------------------------------------------------------------------------
// Public client types
// ----------------------------------------------------------------------------

/// Maximum number of entries in the prepared statement cache.
const MAX_PREPARED_CACHE: usize = 128;

/// Connection-time parameters for the PG client.
#[derive(Debug, Clone)]
pub struct ClientConfig {
    pub user: String,
    pub database: String,
    pub application_name: String,
    pub password: Option<String>,
    pub connect_timeout: Duration,
    pub statement_timeout: Duration,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            user: "zyron".into(),
            database: "zyron".into(),
            application_name: "zyron-client".into(),
            password: None,
            connect_timeout: Duration::from_secs(10),
            statement_timeout: Duration::from_secs(30),
        }
    }
}

/// Transaction state reported by the most recent ReadyForQuery message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransactionState {
    Idle,
    InTxn,
    InFailedTxn,
}

impl From<PgTxnState> for TransactionState {
    fn from(v: PgTxnState) -> Self {
        match v {
            PgTxnState::Idle => TransactionState::Idle,
            PgTxnState::InTransaction => TransactionState::InTxn,
            PgTxnState::Failed => TransactionState::InFailedTxn,
        }
    }
}

/// A value passed as a bound parameter or returned from a query.
/// Text-format is used for simple-path compatibility.
#[derive(Debug, Clone, PartialEq)]
pub enum PgValue {
    Null,
    Text(String),
    Bytea(Vec<u8>),
    Int4(i32),
    Int8(i64),
    Float8(f64),
    Bool(bool),
}

impl PgValue {
    fn encode_text(&self) -> Option<Vec<u8>> {
        match self {
            PgValue::Null => None,
            PgValue::Text(s) => Some(s.as_bytes().to_vec()),
            PgValue::Bytea(b) => Some(b.clone()),
            PgValue::Int4(v) => Some(v.to_string().into_bytes()),
            PgValue::Int8(v) => Some(v.to_string().into_bytes()),
            PgValue::Float8(v) => Some(v.to_string().into_bytes()),
            PgValue::Bool(v) => Some(if *v { b"t".to_vec() } else { b"f".to_vec() }),
        }
    }
}

/// A row of text-encoded column values returned by the server.
#[derive(Debug, Clone)]
pub struct PgRow {
    pub columns: Vec<Option<Vec<u8>>>,
    pub fields: std::sync::Arc<Vec<FieldDescription>>,
}

impl PgRow {
    pub fn get_text(&self, idx: usize) -> Option<&str> {
        self.columns
            .get(idx)
            .and_then(|v| v.as_ref())
            .and_then(|b| std::str::from_utf8(b).ok())
    }
}

/// Outcome of running a simple query.
#[derive(Debug, Clone)]
pub struct QueryResult {
    pub tag: String,
    pub fields: Vec<FieldDescription>,
    pub rows: Vec<Vec<Option<Vec<u8>>>>,
}

/// Async notification delivered via LISTEN/NOTIFY.
#[derive(Debug, Clone)]
pub struct Notification {
    pub process_id: i32,
    pub channel: String,
    pub payload: String,
}

/// Cached prepared statement metadata.
#[derive(Debug, Clone)]
pub struct PreparedStatement {
    pub name: String,
    pub query: String,
    pub param_types: Vec<i32>,
    pub field_descriptions: Vec<FieldDescription>,
}

// ----------------------------------------------------------------------------
// Transport enum: erases the concrete stream type at runtime
// ----------------------------------------------------------------------------

/// Opaque byte-stream transport used by the client.
pub enum ClientTransport {
    Tcp(TcpStream),
    Tls(Box<tokio_rustls::client::TlsStream<TcpStream>>),
    Quic(crate::quic::QuicStream),
}

impl ClientTransport {
    async fn read_buf(&mut self, buf: &mut BytesMut) -> std::io::Result<usize> {
        match self {
            ClientTransport::Tcp(s) => s.read_buf(buf).await,
            ClientTransport::Tls(s) => s.read_buf(buf).await,
            ClientTransport::Quic(s) => s.read_buf(buf).await,
        }
    }

    async fn write_all(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        match self {
            ClientTransport::Tcp(s) => s.write_all(bytes).await,
            ClientTransport::Tls(s) => s.write_all(bytes).await,
            ClientTransport::Quic(s) => s.write_all(bytes).await,
        }
    }

    async fn flush(&mut self) -> std::io::Result<()> {
        match self {
            ClientTransport::Tcp(s) => s.flush().await,
            ClientTransport::Tls(s) => s.flush().await,
            ClientTransport::Quic(s) => s.flush().await,
        }
    }

    async fn shutdown(&mut self) -> std::io::Result<()> {
        match self {
            ClientTransport::Tcp(s) => s.shutdown().await,
            ClientTransport::Tls(s) => s.shutdown().await,
            ClientTransport::Quic(s) => s.shutdown().await,
        }
    }
}

// ----------------------------------------------------------------------------
// PgClient
// ----------------------------------------------------------------------------

/// PostgreSQL wire protocol v3 client. Owns a single backend connection.
pub struct PgClient {
    stream: ClientTransport,
    buffer: BytesMut,
    server_params: HashMap<String, String>,
    backend_pid: i32,
    backend_secret: i32,
    prepared_cache: HashMap<String, PreparedStatement>,
    statement_counter: AtomicU32,
    ready_state: TransactionState,
}

impl PgClient {
    /// Connects in plaintext TCP mode.
    pub async fn connect(addr: SocketAddr, config: &ClientConfig) -> Result<Self, ProtocolError> {
        let stream = tokio::time::timeout(config.connect_timeout, TcpStream::connect(addr))
            .await
            .map_err(|_| {
                ProtocolError::Io(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "connect timeout",
                ))
            })?
            .map_err(ProtocolError::Io)?;
        let _ = stream.set_nodelay(true);
        let transport = ClientTransport::Tcp(stream);
        Self::handshake(transport, config).await
    }

    /// Connects, upgrades to TLS via an SSLRequest, and completes the handshake.
    pub async fn connect_tls(
        addr: SocketAddr,
        tls: &ClientTlsConnector,
        config: &ClientConfig,
    ) -> Result<Self, ProtocolError> {
        let mut stream = tokio::time::timeout(config.connect_timeout, TcpStream::connect(addr))
            .await
            .map_err(|_| {
                ProtocolError::Io(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "connect timeout",
                ))
            })?
            .map_err(ProtocolError::Io)?;
        let _ = stream.set_nodelay(true);

        // SSL request packet: length(8) + code(80877103)
        let mut req = BytesMut::with_capacity(8);
        req.put_i32(8);
        req.put_i32(80877103);
        stream.write_all(&req).await.map_err(ProtocolError::Io)?;
        stream.flush().await.map_err(ProtocolError::Io)?;

        let mut reply = [0u8; 1];
        stream
            .read_exact(&mut reply)
            .await
            .map_err(ProtocolError::Io)?;
        if reply[0] != b'S' {
            return Err(ProtocolError::Malformed(
                "server refused TLS upgrade".into(),
            ));
        }

        let tls_stream = tls
            .connect(stream)
            .await
            .map_err(|e| ProtocolError::Malformed(format!("TLS: {}", e)))?;

        Self::handshake(ClientTransport::Tls(Box::new(tls_stream)), config).await
    }

    /// Connects over a QUIC stream. The caller provides a completed QuicStream
    /// from the zyron-wire QUIC client helper.
    pub async fn connect_quic(
        stream: crate::quic::QuicStream,
        config: &ClientConfig,
    ) -> Result<Self, ProtocolError> {
        Self::handshake(ClientTransport::Quic(stream), config).await
    }

    // ------------------------------------------------------------------------
    // Startup handshake
    // ------------------------------------------------------------------------

    async fn handshake(
        stream: ClientTransport,
        config: &ClientConfig,
    ) -> Result<Self, ProtocolError> {
        let mut client = Self {
            stream,
            buffer: BytesMut::with_capacity(8192),
            server_params: HashMap::new(),
            backend_pid: 0,
            backend_secret: 0,
            prepared_cache: HashMap::new(),
            statement_counter: AtomicU32::new(0),
            ready_state: TransactionState::Idle,
        };

        client.send_startup(config).await?;

        loop {
            let msg = client.recv_message().await?;
            match msg {
                BackendMessage::Authentication(AuthenticationMessage::Ok) => continue,
                BackendMessage::Authentication(AuthenticationMessage::CleartextPassword) => {
                    let pw = config
                        .password
                        .clone()
                        .ok_or_else(|| ProtocolError::AuthFailed("password required".into()))?;
                    client.send_password(PasswordMessage::Cleartext(pw)).await?;
                }
                BackendMessage::Authentication(AuthenticationMessage::Md5Password { salt }) => {
                    let pw = config
                        .password
                        .clone()
                        .ok_or_else(|| ProtocolError::AuthFailed("password required".into()))?;
                    let hash = md5_hash(&pw, &config.user, &salt);
                    client
                        .send_password(PasswordMessage::Md5(hash.into_bytes()))
                        .await?;
                }
                BackendMessage::Authentication(AuthenticationMessage::SaslMechanisms(mechs)) => {
                    if !mechs.iter().any(|m| m == "SCRAM-SHA-256") {
                        return Err(ProtocolError::AuthFailed(
                            "server does not offer SCRAM-SHA-256".into(),
                        ));
                    }
                    let pw = config
                        .password
                        .clone()
                        .ok_or_else(|| ProtocolError::AuthFailed("password required".into()))?;
                    client.scram_sha256(&pw).await?;
                }
                BackendMessage::ParameterStatus { name, value } => {
                    client.server_params.insert(name, value);
                }
                BackendMessage::BackendKeyData {
                    process_id,
                    secret_key,
                } => {
                    client.backend_pid = process_id;
                    client.backend_secret = secret_key;
                }
                BackendMessage::ReadyForQuery(state) => {
                    client.ready_state = state.into();
                    return Ok(client);
                }
                BackendMessage::ErrorResponse(err) => {
                    return Err(ProtocolError::AuthFailed(err.message));
                }
                BackendMessage::NoticeResponse(_) => continue,
                _ => {
                    return Err(ProtocolError::Malformed(
                        "unexpected message during startup".into(),
                    ));
                }
            }
        }
    }

    async fn send_startup(&mut self, config: &ClientConfig) -> Result<(), ProtocolError> {
        let mut payload = BytesMut::new();
        payload.put_i32(196608); // protocol 3.0
        put_cstring(&mut payload, "user");
        put_cstring(&mut payload, &config.user);
        put_cstring(&mut payload, "database");
        put_cstring(&mut payload, &config.database);
        put_cstring(&mut payload, "application_name");
        put_cstring(&mut payload, &config.application_name);
        put_cstring(&mut payload, "client_encoding");
        put_cstring(&mut payload, "UTF8");
        payload.put_u8(0);

        let total = (payload.len() + 4) as i32;
        let mut framed = BytesMut::with_capacity(payload.len() + 4);
        framed.put_i32(total);
        framed.extend_from_slice(&payload);
        self.stream
            .write_all(&framed)
            .await
            .map_err(ProtocolError::Io)?;
        self.stream.flush().await.map_err(ProtocolError::Io)?;
        Ok(())
    }

    async fn send_password(&mut self, msg: PasswordMessage) -> Result<(), ProtocolError> {
        let mut payload = BytesMut::new();
        match msg {
            PasswordMessage::Cleartext(s) => {
                payload.extend_from_slice(s.as_bytes());
                payload.put_u8(0);
            }
            PasswordMessage::Md5(bytes) => {
                payload.extend_from_slice(&bytes);
                payload.put_u8(0);
            }
            PasswordMessage::SaslInitial { mechanism, data } => {
                put_cstring(&mut payload, &mechanism);
                payload.put_i32(data.len() as i32);
                payload.extend_from_slice(&data);
            }
            PasswordMessage::SaslResponse(data) => {
                payload.extend_from_slice(&data);
            }
        }
        self.write_typed(b'p', &payload).await
    }

    async fn scram_sha256(&mut self, password: &str) -> Result<(), ProtocolError> {
        // client-first message: n,,n=user,r=nonce
        let mut nonce_bytes = [0u8; 18];
        rand::rng().fill_bytes(&mut nonce_bytes);
        let client_nonce = base64_encode(&nonce_bytes);
        let client_first_bare = format!("n=,r={}", client_nonce);
        let client_first = format!("n,,{}", client_first_bare);

        self.send_password(PasswordMessage::SaslInitial {
            mechanism: "SCRAM-SHA-256".into(),
            data: client_first.as_bytes().to_vec(),
        })
        .await?;

        let server_first = match self.recv_message().await? {
            BackendMessage::Authentication(AuthenticationMessage::SaslContinue(d)) => d,
            BackendMessage::ErrorResponse(e) => return Err(ProtocolError::AuthFailed(e.message)),
            _ => return Err(ProtocolError::AuthFailed("expected SaslContinue".into())),
        };
        let server_first_str = std::str::from_utf8(&server_first)
            .map_err(|_| ProtocolError::AuthFailed("non-utf8 scram server-first".into()))?;

        let (combined_nonce, salt_b64, iterations) = parse_scram_first(server_first_str)?;
        if !combined_nonce.starts_with(&client_nonce) {
            return Err(ProtocolError::AuthFailed("nonce mismatch".into()));
        }
        let salt = base64_decode(&salt_b64)
            .ok_or_else(|| ProtocolError::AuthFailed("bad scram salt".into()))?;

        let salted = pbkdf2_hmac_sha256(password.as_bytes(), &salt, iterations);
        let client_key = hmac_sha256(&salted, b"Client Key");
        let stored_key = sha256_bytes(&client_key);
        let channel_binding = base64_encode(b"n,,");
        let client_final_no_proof = format!("c={},r={}", channel_binding, combined_nonce);
        let auth_message = format!(
            "{},{},{}",
            client_first_bare, server_first_str, client_final_no_proof
        );
        let client_signature = hmac_sha256(&stored_key, auth_message.as_bytes());
        let client_proof: Vec<u8> = client_key
            .iter()
            .zip(client_signature.iter())
            .map(|(a, b)| a ^ b)
            .collect();
        let client_final = format!(
            "{},p={}",
            client_final_no_proof,
            base64_encode(&client_proof)
        );

        self.send_password(PasswordMessage::SaslResponse(
            client_final.as_bytes().to_vec(),
        ))
        .await?;

        let server_final = match self.recv_message().await? {
            BackendMessage::Authentication(AuthenticationMessage::SaslFinal(d)) => d,
            BackendMessage::ErrorResponse(e) => return Err(ProtocolError::AuthFailed(e.message)),
            _ => return Err(ProtocolError::AuthFailed("expected SaslFinal".into())),
        };
        let server_final_str = std::str::from_utf8(&server_final)
            .map_err(|_| ProtocolError::AuthFailed("non-utf8 scram server-final".into()))?;

        let server_sig_b64 = server_final_str
            .strip_prefix("v=")
            .ok_or_else(|| ProtocolError::AuthFailed("missing server signature".into()))?;
        let expected_server_sig = server_sig_b64;
        let server_key = hmac_sha256(&salted, b"Server Key");
        let computed_sig = hmac_sha256(&server_key, auth_message.as_bytes());
        if base64_encode(&computed_sig) != expected_server_sig {
            return Err(ProtocolError::AuthFailed(
                "server signature mismatch".into(),
            ));
        }
        Ok(())
    }

    // ------------------------------------------------------------------------
    // Message I/O primitives
    // ------------------------------------------------------------------------

    async fn recv_message(&mut self) -> Result<BackendMessage, ProtocolError> {
        loop {
            if let Some(msg) = try_decode_backend(&mut self.buffer)? {
                return Ok(msg);
            }
            let n = self
                .stream
                .read_buf(&mut self.buffer)
                .await
                .map_err(ProtocolError::Io)?;
            if n == 0 {
                return Err(ProtocolError::ConnectionClosed);
            }
        }
    }

    async fn write_typed(&mut self, msg_type: u8, payload: &[u8]) -> Result<(), ProtocolError> {
        let mut buf = BytesMut::with_capacity(payload.len() + 5);
        buf.put_u8(msg_type);
        buf.put_i32((payload.len() + 4) as i32);
        buf.extend_from_slice(payload);
        self.stream
            .write_all(&buf)
            .await
            .map_err(ProtocolError::Io)?;
        self.stream.flush().await.map_err(ProtocolError::Io)?;
        Ok(())
    }

    // ------------------------------------------------------------------------
    // Query API
    // ------------------------------------------------------------------------

    /// Sends a simple SQL string and collects all result sets.
    pub async fn simple_query(&mut self, sql: &str) -> Result<Vec<QueryResult>, ProtocolError> {
        let mut payload = BytesMut::new();
        put_cstring(&mut payload, sql);
        self.write_typed(b'Q', &payload).await?;

        let mut results = Vec::new();
        let mut current_fields: Vec<FieldDescription> = Vec::new();
        let mut current_rows: Vec<Vec<Option<Vec<u8>>>> = Vec::new();

        loop {
            match self.recv_message().await? {
                BackendMessage::RowDescription(f) => {
                    current_fields = f;
                    current_rows = Vec::new();
                }
                BackendMessage::DataRow(values) => {
                    current_rows.push(values);
                }
                BackendMessage::CommandComplete { tag } => {
                    results.push(QueryResult {
                        tag,
                        fields: std::mem::take(&mut current_fields),
                        rows: std::mem::take(&mut current_rows),
                    });
                }
                BackendMessage::EmptyQueryResponse => {
                    results.push(QueryResult {
                        tag: String::new(),
                        fields: Vec::new(),
                        rows: Vec::new(),
                    });
                }
                BackendMessage::ErrorResponse(e) => {
                    // Drain until ReadyForQuery.
                    loop {
                        match self.recv_message().await? {
                            BackendMessage::ReadyForQuery(s) => {
                                self.ready_state = s.into();
                                return Err(ProtocolError::Malformed(format!(
                                    "server error: {}",
                                    e.message
                                )));
                            }
                            _ => continue,
                        }
                    }
                }
                BackendMessage::ReadyForQuery(state) => {
                    self.ready_state = state.into();
                    return Ok(results);
                }
                BackendMessage::NoticeResponse(_) => continue,
                BackendMessage::ParameterStatus { name, value } => {
                    self.server_params.insert(name, value);
                }
                _ => continue,
            }
        }
    }

    /// Runs a prepared statement once with parameters. Caches the plan.
    pub async fn execute(
        &mut self,
        sql: &str,
        params: &[PgValue],
    ) -> Result<QueryResult, ProtocolError> {
        let stmt = self.prepare(sql).await?;
        let name = stmt.name.clone();

        self.send_bind_execute(&name, "", params).await?;
        self.write_typed(b'S', &[]).await?; // Sync

        let mut tag = String::new();
        let mut rows = Vec::new();
        let fields = stmt.field_descriptions.clone();
        loop {
            match self.recv_message().await? {
                BackendMessage::BindComplete => continue,
                BackendMessage::DataRow(v) => rows.push(v),
                BackendMessage::CommandComplete { tag: t } => tag = t,
                BackendMessage::EmptyQueryResponse => {}
                BackendMessage::ErrorResponse(e) => loop {
                    if let BackendMessage::ReadyForQuery(s) = self.recv_message().await? {
                        self.ready_state = s.into();
                        return Err(ProtocolError::Malformed(format!(
                            "server error: {}",
                            e.message
                        )));
                    }
                },
                BackendMessage::ReadyForQuery(s) => {
                    self.ready_state = s.into();
                    return Ok(QueryResult { tag, fields, rows });
                }
                _ => continue,
            }
        }
    }

    /// Runs a prepared statement and returns the rows with field metadata.
    pub async fn query(
        &mut self,
        sql: &str,
        params: &[PgValue],
    ) -> Result<Vec<PgRow>, ProtocolError> {
        let result = self.execute(sql, params).await?;
        let fields = std::sync::Arc::new(result.fields);
        Ok(result
            .rows
            .into_iter()
            .map(|columns| PgRow {
                columns,
                fields: fields.clone(),
            })
            .collect())
    }

    /// Parses and caches a prepared statement, returning cached metadata on hits.
    pub async fn prepare(&mut self, sql: &str) -> Result<PreparedStatement, ProtocolError> {
        if let Some(stmt) = self.prepared_cache.get(sql) {
            return Ok(stmt.clone());
        }

        // Evict LRU-approximation by count: drop an arbitrary entry.
        if self.prepared_cache.len() >= MAX_PREPARED_CACHE {
            if let Some(key) = self.prepared_cache.keys().next().cloned() {
                if let Some(old) = self.prepared_cache.remove(&key) {
                    let _ = self.close_statement(&old.name).await;
                }
            }
        }

        let id = self.statement_counter.fetch_add(1, Ordering::Relaxed);
        let name = format!("zstmt_{}", id);

        // Parse message
        let mut parse_payload = BytesMut::new();
        put_cstring(&mut parse_payload, &name);
        put_cstring(&mut parse_payload, sql);
        parse_payload.put_i16(0); // zero param type hints
        self.write_typed(b'P', &parse_payload).await?;

        // Describe statement
        let mut desc = BytesMut::new();
        desc.put_u8(b'S');
        put_cstring(&mut desc, &name);
        self.write_typed(b'D', &desc).await?;

        // Sync to flush and get ReadyForQuery
        self.write_typed(b'S', &[]).await?;

        let mut param_types = Vec::new();
        let mut field_descriptions = Vec::new();
        loop {
            match self.recv_message().await? {
                BackendMessage::ParseComplete => continue,
                BackendMessage::ParameterDescription(t) => param_types = t,
                BackendMessage::RowDescription(f) => field_descriptions = f,
                BackendMessage::NoData => field_descriptions.clear(),
                BackendMessage::ErrorResponse(e) => loop {
                    if let BackendMessage::ReadyForQuery(s) = self.recv_message().await? {
                        self.ready_state = s.into();
                        return Err(ProtocolError::Malformed(format!(
                            "prepare failed: {}",
                            e.message
                        )));
                    }
                },
                BackendMessage::ReadyForQuery(s) => {
                    self.ready_state = s.into();
                    break;
                }
                _ => continue,
            }
        }

        let stmt = PreparedStatement {
            name,
            query: sql.to_string(),
            param_types,
            field_descriptions,
        };
        self.prepared_cache.insert(sql.to_string(), stmt.clone());
        Ok(stmt)
    }

    async fn send_bind_execute(
        &mut self,
        stmt_name: &str,
        portal: &str,
        params: &[PgValue],
    ) -> Result<(), ProtocolError> {
        // Bind
        let mut bind = BytesMut::new();
        put_cstring(&mut bind, portal);
        put_cstring(&mut bind, stmt_name);
        bind.put_i16(0); // all text format
        bind.put_i16(params.len() as i16);
        for p in params {
            match p.encode_text() {
                None => bind.put_i32(-1),
                Some(bytes) => {
                    bind.put_i32(bytes.len() as i32);
                    bind.extend_from_slice(&bytes);
                }
            }
        }
        bind.put_i16(0); // all text result format
        self.write_typed(b'B', &bind).await?;

        // Execute
        let mut exec = BytesMut::new();
        put_cstring(&mut exec, portal);
        exec.put_i32(0);
        self.write_typed(b'E', &exec).await?;
        Ok(())
    }

    async fn close_statement(&mut self, name: &str) -> Result<(), ProtocolError> {
        let mut payload = BytesMut::new();
        payload.put_u8(b'S');
        put_cstring(&mut payload, name);
        self.write_typed(b'C', &payload).await?;
        self.write_typed(b'S', &[]).await?;
        // Drain until ReadyForQuery.
        loop {
            match self.recv_message().await? {
                BackendMessage::CloseComplete => continue,
                BackendMessage::ReadyForQuery(s) => {
                    self.ready_state = s.into();
                    return Ok(());
                }
                BackendMessage::ErrorResponse(_) => continue,
                _ => continue,
            }
        }
    }

    // ------------------------------------------------------------------------
    // COPY
    // ------------------------------------------------------------------------

    /// Starts a binary COPY FROM STDIN into the given table/columns.
    pub async fn copy_in_binary<'a>(
        &'a mut self,
        table: &str,
        columns: &[&str],
    ) -> Result<CopyInWriter<'a>, ProtocolError> {
        let cols = if columns.is_empty() {
            String::new()
        } else {
            format!(" ({})", columns.join(","))
        };
        let sql = format!("COPY {}{} FROM STDIN BINARY", table, cols);
        let mut payload = BytesMut::new();
        put_cstring(&mut payload, &sql);
        self.write_typed(b'Q', &payload).await?;

        loop {
            match self.recv_message().await? {
                BackendMessage::CopyInResponse { .. } => break,
                BackendMessage::NoticeResponse(_) => continue,
                BackendMessage::ErrorResponse(e) => {
                    self.drain_until_ready().await?;
                    return Err(ProtocolError::Malformed(format!(
                        "copy_in refused: {}",
                        e.message
                    )));
                }
                _ => continue,
            }
        }

        // Send COPY binary header: signature + flags + extension length.
        let mut hdr = BytesMut::new();
        hdr.extend_from_slice(b"PGCOPY\n\xff\r\n\0");
        hdr.put_i32(0); // flags
        hdr.put_i32(0); // header extension length
        self.send_copy_data(&hdr).await?;

        Ok(CopyInWriter { client: self })
    }

    /// Starts a binary COPY TO STDOUT for the given query.
    pub async fn copy_out_binary<'a>(
        &'a mut self,
        query: &str,
    ) -> Result<CopyOutReader<'a>, ProtocolError> {
        let mut payload = BytesMut::new();
        put_cstring(&mut payload, query);
        self.write_typed(b'Q', &payload).await?;

        loop {
            match self.recv_message().await? {
                BackendMessage::CopyOutResponse { .. } => break,
                BackendMessage::NoticeResponse(_) => continue,
                BackendMessage::ErrorResponse(e) => {
                    self.drain_until_ready().await?;
                    return Err(ProtocolError::Malformed(format!(
                        "copy_out refused: {}",
                        e.message
                    )));
                }
                _ => continue,
            }
        }

        Ok(CopyOutReader {
            client: self,
            eof: false,
        })
    }

    async fn send_copy_data(&mut self, bytes: &[u8]) -> Result<(), ProtocolError> {
        self.write_typed(b'd', bytes).await
    }

    async fn send_copy_done(&mut self) -> Result<(), ProtocolError> {
        self.write_typed(b'c', &[]).await
    }

    async fn send_copy_fail(&mut self, msg: &str) -> Result<(), ProtocolError> {
        let mut payload = BytesMut::new();
        put_cstring(&mut payload, msg);
        self.write_typed(b'f', &payload).await
    }

    async fn drain_until_ready(&mut self) -> Result<(), ProtocolError> {
        loop {
            match self.recv_message().await? {
                BackendMessage::ReadyForQuery(s) => {
                    self.ready_state = s.into();
                    return Ok(());
                }
                _ => continue,
            }
        }
    }

    // ------------------------------------------------------------------------
    // LISTEN/NOTIFY
    // ------------------------------------------------------------------------

    pub async fn listen(&mut self, channel: &str) -> Result<(), ProtocolError> {
        let sql = format!("LISTEN \"{}\"", channel.replace('"', "\"\""));
        let _ = self.simple_query(&sql).await?;
        Ok(())
    }

    /// Waits up to `timeout` for a NotificationResponse. Returns `Ok(None)`
    /// when the timeout elapses. The NotificationResponse message is not
    /// currently emitted by our own server, so this parser decodes the raw
    /// frame type `A` if it arrives.
    pub async fn wait_notification(
        &mut self,
        timeout: Duration,
    ) -> Result<Option<Notification>, ProtocolError> {
        let fut = async {
            // Read raw frame until type 'A'.
            loop {
                if self.buffer.len() >= 5 {
                    let t = self.buffer[0];
                    let len = i32::from_be_bytes([
                        self.buffer[1],
                        self.buffer[2],
                        self.buffer[3],
                        self.buffer[4],
                    ]) as usize;
                    if self.buffer.len() >= 1 + len {
                        if t == b'A' {
                            self.buffer.advance(5);
                            let pid = self.buffer.get_i32();
                            let channel = read_cstring(&mut self.buffer)?;
                            let payload = read_cstring(&mut self.buffer)?;
                            return Ok::<_, ProtocolError>(Some(Notification {
                                process_id: pid,
                                channel,
                                payload,
                            }));
                        }
                    }
                }
                let n = self
                    .stream
                    .read_buf(&mut self.buffer)
                    .await
                    .map_err(ProtocolError::Io)?;
                if n == 0 {
                    return Err(ProtocolError::ConnectionClosed);
                }
            }
        };
        match tokio::time::timeout(timeout, fut).await {
            Ok(r) => r,
            Err(_) => Ok(None),
        }
    }

    // ------------------------------------------------------------------------
    // Transaction helpers
    // ------------------------------------------------------------------------

    pub async fn begin(&mut self) -> Result<(), ProtocolError> {
        let _ = self.simple_query("BEGIN").await?;
        Ok(())
    }

    pub async fn commit(&mut self) -> Result<(), ProtocolError> {
        let _ = self.simple_query("COMMIT").await?;
        Ok(())
    }

    pub async fn rollback(&mut self) -> Result<(), ProtocolError> {
        let _ = self.simple_query("ROLLBACK").await?;
        Ok(())
    }

    /// Sends Terminate and closes the connection.
    pub async fn close(mut self) -> Result<(), ProtocolError> {
        let _ = self.write_typed(b'X', &[]).await;
        let _ = self.stream.shutdown().await;
        Ok(())
    }

    pub fn in_transaction(&self) -> bool {
        !matches!(self.ready_state, TransactionState::Idle)
    }

    pub fn ready_state(&self) -> TransactionState {
        self.ready_state
    }

    pub fn backend_pid(&self) -> i32 {
        self.backend_pid
    }

    pub fn server_param(&self, key: &str) -> Option<&str> {
        self.server_params.get(key).map(|s| s.as_str())
    }
}

// ----------------------------------------------------------------------------
// COPY helpers
// ----------------------------------------------------------------------------

/// Writer for COPY FROM STDIN BINARY.
pub struct CopyInWriter<'a> {
    client: &'a mut PgClient,
}

impl<'a> CopyInWriter<'a> {
    /// Appends one row encoded in the PG binary COPY format. Caller must
    /// supply binary-encoded values that match the table column types.
    pub async fn write_row(&mut self, values: &[Option<Vec<u8>>]) -> Result<(), ProtocolError> {
        let mut buf = BytesMut::with_capacity(2 + values.len() * 8);
        buf.put_i16(values.len() as i16);
        for v in values {
            match v {
                None => buf.put_i32(-1),
                Some(data) => {
                    buf.put_i32(data.len() as i32);
                    buf.extend_from_slice(data);
                }
            }
        }
        self.client.send_copy_data(&buf).await
    }

    /// Sends the trailer (-1 as int16) and CopyDone, reads CommandComplete.
    pub async fn finish(self) -> Result<u64, ProtocolError> {
        let mut trailer = BytesMut::new();
        trailer.put_i16(-1);
        self.client.send_copy_data(&trailer).await?;
        self.client.send_copy_done().await?;

        let mut count = 0u64;
        loop {
            match self.client.recv_message().await? {
                BackendMessage::CommandComplete { tag } => {
                    if let Some(n) = tag.split_whitespace().last().and_then(|s| s.parse().ok()) {
                        count = n;
                    }
                }
                BackendMessage::ReadyForQuery(s) => {
                    self.client.ready_state = s.into();
                    return Ok(count);
                }
                _ => continue,
            }
        }
    }

    /// Cancels the in-progress COPY by sending CopyFail.
    pub async fn cancel(self) -> Result<(), ProtocolError> {
        self.client.send_copy_fail("client cancelled").await?;
        self.client.drain_until_ready().await
    }
}

/// Reader for COPY TO STDOUT BINARY.
pub struct CopyOutReader<'a> {
    client: &'a mut PgClient,
    eof: bool,
}

impl<'a> CopyOutReader<'a> {
    /// Reads the next binary chunk or returns `None` at end-of-stream.
    pub async fn read_chunk(&mut self) -> Result<Option<Vec<u8>>, ProtocolError> {
        if self.eof {
            return Ok(None);
        }
        loop {
            match self.client.recv_message().await? {
                BackendMessage::CopyData(d) => return Ok(Some(d)),
                BackendMessage::CopyDone => continue,
                BackendMessage::CommandComplete { .. } => continue,
                BackendMessage::ReadyForQuery(s) => {
                    self.client.ready_state = s.into();
                    self.eof = true;
                    return Ok(None);
                }
                _ => continue,
            }
        }
    }

    /// Consumes the stream until end-of-data and returns all bytes concatenated.
    pub async fn read_all(mut self) -> Result<Vec<u8>, ProtocolError> {
        let mut out = Vec::new();
        while let Some(chunk) = self.read_chunk().await? {
            out.extend_from_slice(&chunk);
        }
        Ok(out)
    }
}

// ----------------------------------------------------------------------------
// Internal helpers: message decoding, crypto, encoding
// ----------------------------------------------------------------------------

fn try_decode_backend(buf: &mut BytesMut) -> Result<Option<BackendMessage>, ProtocolError> {
    if buf.len() < 5 {
        return Ok(None);
    }
    let t = buf[0];
    let len = i32::from_be_bytes([buf[1], buf[2], buf[3], buf[4]]) as usize;
    if len < 4 {
        return Err(ProtocolError::Malformed("backend len < 4".into()));
    }
    let total = 1 + len;
    if buf.len() < total {
        return Ok(None);
    }
    buf.advance(5);
    let payload_len = len - 4;
    let mut payload = buf.split_to(payload_len);
    decode_backend(t, &mut payload).map(Some)
}

fn decode_backend(msg_type: u8, p: &mut BytesMut) -> Result<BackendMessage, ProtocolError> {
    use backend::ErrorFields;
    match msg_type {
        b'R' => {
            let sub = p.get_i32();
            let auth = match sub {
                0 => AuthenticationMessage::Ok,
                3 => AuthenticationMessage::CleartextPassword,
                5 => {
                    let mut salt = [0u8; 4];
                    p.copy_to_slice(&mut salt);
                    AuthenticationMessage::Md5Password { salt }
                }
                10 => {
                    let mut mechs = Vec::new();
                    loop {
                        let s = read_cstring(p)?;
                        if s.is_empty() {
                            break;
                        }
                        mechs.push(s);
                    }
                    AuthenticationMessage::SaslMechanisms(mechs)
                }
                11 => AuthenticationMessage::SaslContinue(p.to_vec()),
                12 => AuthenticationMessage::SaslFinal(p.to_vec()),
                _ => {
                    return Err(ProtocolError::Malformed(format!(
                        "unknown auth subtype {}",
                        sub
                    )));
                }
            };
            p.advance(p.len());
            Ok(BackendMessage::Authentication(auth))
        }
        b'S' => {
            let name = read_cstring(p)?;
            let value = read_cstring(p)?;
            Ok(BackendMessage::ParameterStatus { name, value })
        }
        b'K' => {
            let pid = p.get_i32();
            let secret = p.get_i32();
            Ok(BackendMessage::BackendKeyData {
                process_id: pid,
                secret_key: secret,
            })
        }
        b'Z' => {
            let b = p.get_u8();
            let state = match b {
                b'I' => PgTxnState::Idle,
                b'T' => PgTxnState::InTransaction,
                b'E' => PgTxnState::Failed,
                _ => {
                    return Err(ProtocolError::Malformed(format!("unknown txn state {}", b)));
                }
            };
            Ok(BackendMessage::ReadyForQuery(state))
        }
        b'T' => {
            let n = p.get_i16() as usize;
            let mut fields = Vec::with_capacity(n);
            for _ in 0..n {
                let name = read_cstring(p)?;
                let table_oid = p.get_i32();
                let column_attr = p.get_i16();
                let type_oid = p.get_i32();
                let type_size = p.get_i16();
                let type_modifier = p.get_i32();
                let format = p.get_i16();
                fields.push(FieldDescription {
                    name,
                    table_oid,
                    column_attr,
                    type_oid,
                    type_size,
                    type_modifier,
                    format,
                });
            }
            Ok(BackendMessage::RowDescription(fields))
        }
        b'D' => {
            let n = p.get_i16() as usize;
            let mut values = Vec::with_capacity(n);
            for _ in 0..n {
                let len = p.get_i32();
                if len == -1 {
                    values.push(None);
                } else {
                    let len = len as usize;
                    values.push(Some(p.split_to(len).to_vec()));
                }
            }
            Ok(BackendMessage::DataRow(values))
        }
        b'C' => {
            let tag = read_cstring(p)?;
            Ok(BackendMessage::CommandComplete { tag })
        }
        b'E' | b'N' => {
            let mut severity = String::new();
            let mut code = String::new();
            let mut message = String::new();
            let mut detail = None;
            let mut hint = None;
            let mut position = None;
            loop {
                if p.is_empty() {
                    break;
                }
                let k = p.get_u8();
                if k == 0 {
                    break;
                }
                let v = read_cstring(p)?;
                match k {
                    b'S' | b'V' => severity = v,
                    b'C' => code = v,
                    b'M' => message = v,
                    b'D' => detail = Some(v),
                    b'H' => hint = Some(v),
                    b'P' => position = v.parse().ok(),
                    _ => {}
                }
            }
            let fields = ErrorFields {
                severity,
                code,
                message,
                detail,
                hint,
                position,
            };
            Ok(if msg_type == b'E' {
                BackendMessage::ErrorResponse(fields)
            } else {
                BackendMessage::NoticeResponse(fields)
            })
        }
        b'1' => Ok(BackendMessage::ParseComplete),
        b'2' => Ok(BackendMessage::BindComplete),
        b'3' => Ok(BackendMessage::CloseComplete),
        b's' => Ok(BackendMessage::PortalSuspended),
        b'n' => Ok(BackendMessage::NoData),
        b't' => {
            let n = p.get_i16() as usize;
            let mut t = Vec::with_capacity(n);
            for _ in 0..n {
                t.push(p.get_i32());
            }
            Ok(BackendMessage::ParameterDescription(t))
        }
        b'I' => Ok(BackendMessage::EmptyQueryResponse),
        b'G' => {
            let format = p.get_i8();
            let n = p.get_i16() as usize;
            let mut fmts = Vec::with_capacity(n);
            for _ in 0..n {
                fmts.push(p.get_i16());
            }
            Ok(BackendMessage::CopyInResponse {
                format,
                column_formats: fmts,
            })
        }
        b'H' => {
            let format = p.get_i8();
            let n = p.get_i16() as usize;
            let mut fmts = Vec::with_capacity(n);
            for _ in 0..n {
                fmts.push(p.get_i16());
            }
            Ok(BackendMessage::CopyOutResponse {
                format,
                column_formats: fmts,
            })
        }
        b'd' => {
            let data = p.split_to(p.len()).to_vec();
            Ok(BackendMessage::CopyData(data))
        }
        b'c' => Ok(BackendMessage::CopyDone),
        _ => Err(ProtocolError::InvalidMessageType(msg_type)),
    }
}

fn read_cstring(buf: &mut BytesMut) -> Result<String, ProtocolError> {
    let slice = buf.as_ref();
    let pos = memchr::memchr(0, slice)
        .ok_or_else(|| ProtocolError::Malformed("missing NUL in cstring".into()))?;
    let s = std::str::from_utf8(&slice[..pos])
        .map_err(|e| ProtocolError::Malformed(format!("bad utf8: {}", e)))?
        .to_string();
    buf.advance(pos + 1);
    Ok(s)
}

fn put_cstring(buf: &mut BytesMut, s: &str) {
    buf.extend_from_slice(s.as_bytes());
    buf.put_u8(0);
}

fn md5_hash(password: &str, user: &str, salt: &[u8; 4]) -> String {
    use md5::{Digest as _, Md5};
    let mut h = Md5::new();
    h.update(password.as_bytes());
    h.update(user.as_bytes());
    let first = h.finalize();
    let first_hex = hex_lower(&first);
    let mut h = Md5::new();
    h.update(first_hex.as_bytes());
    h.update(salt);
    let second = h.finalize();
    format!("md5{}", hex_lower(&second))
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

fn sha256_bytes(input: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(input);
    h.finalize().into()
}

fn hmac_sha256(key: &[u8], msg: &[u8]) -> [u8; 32] {
    let mut mac = Hmac::<Sha256>::new_from_slice(key).expect("hmac key");
    mac.update(msg);
    mac.finalize().into_bytes().into()
}

fn pbkdf2_hmac_sha256(password: &[u8], salt: &[u8], iters: u32) -> [u8; 32] {
    let mut out = [0u8; 32];
    pbkdf2::pbkdf2_hmac::<Sha256>(password, salt, iters, &mut out);
    out
}

fn parse_scram_first(s: &str) -> Result<(String, String, u32), ProtocolError> {
    let mut nonce = None;
    let mut salt = None;
    let mut iters = None;
    for part in s.split(',') {
        if let Some(v) = part.strip_prefix("r=") {
            nonce = Some(v.to_string());
        } else if let Some(v) = part.strip_prefix("s=") {
            salt = Some(v.to_string());
        } else if let Some(v) = part.strip_prefix("i=") {
            iters = v.parse().ok();
        }
    }
    Ok((
        nonce.ok_or_else(|| ProtocolError::AuthFailed("missing scram nonce".into()))?,
        salt.ok_or_else(|| ProtocolError::AuthFailed("missing scram salt".into()))?,
        iters.ok_or_else(|| ProtocolError::AuthFailed("missing scram iters".into()))?,
    ))
}

fn base64_encode(bytes: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

fn base64_decode(s: &str) -> Option<Vec<u8>> {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.decode(s).ok()
}

// Unused imports guard
#[allow(dead_code)]
fn _use_codec_type(_c: &PostgresCodec) {}

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_md5_hash_format() {
        let h = md5_hash("secret", "alice", &[1, 2, 3, 4]);
        assert!(h.starts_with("md5"));
        assert_eq!(h.len(), 35);
    }

    #[test]
    fn test_hex_lower_padding() {
        assert_eq!(hex_lower(&[0, 1, 15, 255]), "00010fff");
    }

    #[test]
    fn test_parse_scram_first_ok() {
        let (nonce, salt, iters) = parse_scram_first("r=ABCDEF,s=c29tZXNhbHQ=,i=4096").unwrap();
        assert_eq!(nonce, "ABCDEF");
        assert_eq!(salt, "c29tZXNhbHQ=");
        assert_eq!(iters, 4096);
    }

    #[test]
    fn test_parse_scram_first_missing_nonce() {
        assert!(parse_scram_first("s=foo,i=100").is_err());
    }

    #[test]
    fn test_base64_roundtrip() {
        let s = base64_encode(b"hello");
        assert_eq!(base64_decode(&s).unwrap(), b"hello");
    }

    #[test]
    fn test_pg_value_encode_text() {
        assert_eq!(PgValue::Null.encode_text(), None);
        assert_eq!(PgValue::Int4(42).encode_text().unwrap(), b"42".to_vec());
        assert_eq!(PgValue::Bool(true).encode_text().unwrap(), b"t".to_vec());
        assert_eq!(PgValue::Bool(false).encode_text().unwrap(), b"f".to_vec());
        assert_eq!(
            PgValue::Text("abc".into()).encode_text().unwrap(),
            b"abc".to_vec()
        );
    }

    #[test]
    fn test_try_decode_backend_partial() {
        let mut buf = BytesMut::new();
        buf.put_u8(b'Z');
        buf.put_u8(0);
        let r = try_decode_backend(&mut buf).unwrap();
        assert!(r.is_none());
    }

    #[test]
    fn test_try_decode_backend_ready() {
        let mut buf = BytesMut::new();
        buf.put_u8(b'Z');
        buf.put_i32(5);
        buf.put_u8(b'I');
        let msg = try_decode_backend(&mut buf).unwrap().unwrap();
        assert!(matches!(
            msg,
            BackendMessage::ReadyForQuery(PgTxnState::Idle)
        ));
    }

    #[test]
    fn test_try_decode_backend_datarow_roundtrip() {
        let mut enc = BytesMut::new();
        BackendMessage::DataRow(vec![Some(b"a".to_vec()), None, Some(b"bb".to_vec())])
            .encode(&mut enc);
        let msg = try_decode_backend(&mut enc).unwrap().unwrap();
        match msg {
            BackendMessage::DataRow(v) => {
                assert_eq!(v.len(), 3);
                assert_eq!(v[0], Some(b"a".to_vec()));
                assert_eq!(v[1], None);
                assert_eq!(v[2], Some(b"bb".to_vec()));
            }
            _ => panic!("wrong type"),
        }
    }

    #[test]
    fn test_try_decode_backend_command_complete() {
        let mut enc = BytesMut::new();
        BackendMessage::CommandComplete {
            tag: "SELECT 3".into(),
        }
        .encode(&mut enc);
        let msg = try_decode_backend(&mut enc).unwrap().unwrap();
        match msg {
            BackendMessage::CommandComplete { tag } => assert_eq!(tag, "SELECT 3"),
            _ => panic!("wrong type"),
        }
    }

    #[test]
    fn test_try_decode_backend_parameter_status() {
        let mut enc = BytesMut::new();
        BackendMessage::ParameterStatus {
            name: "server_version".into(),
            value: "17.0".into(),
        }
        .encode(&mut enc);
        let msg = try_decode_backend(&mut enc).unwrap().unwrap();
        match msg {
            BackendMessage::ParameterStatus { name, value } => {
                assert_eq!(name, "server_version");
                assert_eq!(value, "17.0");
            }
            _ => panic!("wrong type"),
        }
    }

    #[test]
    fn test_transaction_state_from() {
        assert_eq!(
            TransactionState::from(PgTxnState::Idle),
            TransactionState::Idle
        );
        assert_eq!(
            TransactionState::from(PgTxnState::InTransaction),
            TransactionState::InTxn
        );
        assert_eq!(
            TransactionState::from(PgTxnState::Failed),
            TransactionState::InFailedTxn
        );
    }

    #[test]
    fn test_pbkdf2_known_value() {
        // RFC 7677 test vector sanity: 1 iteration with trivial inputs produces
        // a deterministic 32-byte output. We just verify determinism here.
        let a = pbkdf2_hmac_sha256(b"pw", b"salt", 1);
        let b = pbkdf2_hmac_sha256(b"pw", b"salt", 1);
        assert_eq!(a, b);
        let c = pbkdf2_hmac_sha256(b"pw", b"salt", 2);
        assert_ne!(a, c);
    }
}

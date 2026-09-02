//! PostgreSQL wire protocol v3 implementation for Zyron.
//!
//! Provides a server that speaks the PostgreSQL wire protocol over TCP
//! and QUIC transports. Any PostgreSQL-compatible client (psql, psycopg2,
//! node-postgres, JDBC) can connect via TCP. QUIC transport provides
//! mandatory TLS 1.3, 0-RTT connection resumption, and connection migration.

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

pub mod admission;
pub mod auth;
pub mod auto_param;
pub mod codec;
pub mod connection;
pub mod copy;
pub mod copy_external_dispatch;
pub mod currency_rates;
pub mod ddl_dispatch;
pub mod dml_enforce;
pub mod endpoint_registrar;
pub mod foreign_reader;
pub mod format_dispatch;
pub mod index_build;
pub mod lake_changes;
pub mod lifecycle_dispatch;
pub mod messages;
pub mod notifications;
pub mod peer_probe;
pub mod pem;
pub mod pg_client;
pub mod plan_cache;
pub mod pool;
pub mod pressure_views;
pub mod publication_filter;
pub mod quic;
pub mod row_security;
pub mod search_resilience_ddl;
pub mod session;
pub mod statement_cache;
pub mod subscription;
pub mod system_cdc_views;
pub mod system_compliance_report;
pub mod system_core_views;
pub mod system_format_views;
pub mod system_retention_views;
pub mod system_streaming_views;
pub mod system_views;
pub mod tls;
pub mod transport;
pub mod types;
pub mod uri;
pub mod version_registry;
pub mod zyron_sink;
pub mod zyron_source;

use std::net::SocketAddr;
use std::sync::Arc;

use tokio::net::TcpListener;
use tracing::{debug, error, info, warn};

use zyron_common::ServerConfig;
use zyron_pressure::pressure_control::{ConnectionSlot, PressureController};

use crate::connection::{Connection, ServerState};

pub use crate::endpoint_registrar::EndpointRegistrar;
pub use crate::zyron_sink::{ZyronSinkClient, build_sink_client_from_entry};

/// Creates a TCP listener with SO_REUSEADDR for fast server restarts
/// (no waiting for TIME_WAIT sockets to expire) and SO_REUSEPORT on Linux
/// for kernel-level load balancing across accept loops.
///
/// When `addr` is IPv6 and `dual_stack` is true, IPV6_V6ONLY is disabled so
/// the listener also accepts IPv4 connections via IPv4-mapped IPv6 addresses.
/// Linux defaults V6ONLY off (matches dual_stack=true), Windows defaults it
/// on (so this option overrides explicitly via socket2)
///
/// Public so tests and benchmarks bind the way the server binds. A listener
/// created any other way inherits the platform's small default accept queue,
/// which changes what a connection benchmark measures.
pub fn create_tcp_listener(
    addr: SocketAddr,
    dual_stack: bool,
) -> std::io::Result<std::net::TcpListener> {
    let socket = socket2::Socket::new(
        socket2::Domain::for_address(addr),
        socket2::Type::STREAM,
        Some(socket2::Protocol::TCP),
    )?;
    socket.set_reuse_address(true)?;
    #[cfg(target_os = "linux")]
    socket.set_reuse_port(true)?;
    if addr.is_ipv6() {
        socket.set_only_v6(!dual_stack)?;
    }
    socket.bind(&addr.into())?;
    // The accept queue holds connections the kernel has completed but the
    // accept loop has not taken yet. Once it is full the kernel drops further
    // handshakes rather than refusing them, so those clients stall for a
    // retransmit timeout instead of failing fast. Asking for the maximum lets
    // the kernel apply its own limit, which an operator can tune, rather than
    // a depth chosen here that no deployment can raise
    socket.listen(i32::MAX)?;
    socket.set_nonblocking(true)?;
    Ok(std::net::TcpListener::from(socket))
}

/// Strips optional brackets from an IPv6 host literal, so callers can pass
/// either `[::]` or `::` interchangeably. Hostnames and IPv4 addresses pass
/// through unchanged
fn strip_ipv6_brackets(host: &str) -> &str {
    let trimmed = host.trim();
    trimmed
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .unwrap_or(trimmed)
}

/// Peeks at the first 8 bytes of a TCP stream and returns true if the client
/// sent an SSLRequest. Handles partial reads by looping until 8 bytes are
/// buffered or EOF.
async fn is_ssl_request(stream: &tokio::net::TcpStream) -> std::io::Result<bool> {
    let mut peek = [0u8; 8];
    loop {
        let n = stream.peek(&mut peek).await?;
        if n >= 8 {
            let len = i32::from_be_bytes([peek[0], peek[1], peek[2], peek[3]]);
            let code = i32::from_be_bytes([peek[4], peek[5], peek[6], peek[7]]);
            return Ok(len == 8 && code == 80877103);
        }
        if n == 0 {
            return Ok(false);
        }
        // tokio::net::TcpStream::peek is one-shot; try again.
        tokio::task::yield_now().await;
    }
}

/// Reads and discards the 8-byte SSLRequest packet from the stream, replies
/// with 'S', and performs the TLS handshake. Returns the TLS stream.
async fn upgrade_to_tls(
    mut stream: tokio::net::TcpStream,
    acceptor: &tls::ServerTlsAcceptor,
) -> std::io::Result<tokio_rustls::server::TlsStream<tokio::net::TcpStream>> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut discard = [0u8; 8];
    stream.read_exact(&mut discard).await?;
    stream.write_all(b"S").await?;
    stream.flush().await?;
    acceptor
        .accept(stream)
        .await
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
}

/// Starts the wire protocol server on the configured address.
///
/// Every accepted connection becomes one task on the runtime the caller is
/// already running, which is the work-stealing multi-thread runtime the server
/// binary builds and sizes from `server.worker_threads`.
///
/// There was a pool of dedicated threads here, each with a current-thread
/// runtime and a LocalSet, on the stated grounds that the planner produced
/// futures that were not Send. They are Send, and so is the whole of
/// `Connection::run`, so the pool bought nothing and cost two things: a query
/// could not move off the thread it started on, and any work an operator
/// spawned landed on that same single-threaded runtime, which turned
/// intra-query parallelism into interleaving on one core.
///
/// How many clients may connect is not configured. A connection costs memory,
/// so the ceiling is what memory affords and it rises with the hardware,
/// rather than being a fixed count no deployment could raise.
///
/// When QUIC is enabled, listens on both TCP and UDP simultaneously.
pub async fn start_server(
    config: &ServerConfig,
    server_state: Arc<ServerState>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Strip IPv6 brackets so `[::]` and `::` are equivalent in the config
    let host_clean = strip_ipv6_brackets(&config.host);
    let addr: SocketAddr = format!("{}:{}", host_clean, config.port).parse()?;
    let std_listener = create_tcp_listener(addr, config.dual_stack)?;
    let listener = TcpListener::from_std(std_listener)?;

    // Start QUIC listener if enabled.
    let mut quic_rx = None;
    if config.quic_enabled {
        if let (Some(cert_path), Some(key_path)) = (&config.tls_cert_path, &config.tls_key_path) {
            let quic_addr: SocketAddr =
                format!("{}:{}", config.host, config.quic_listen_port()).parse()?;
            match quic::setup_quic_listener(
                quic_addr,
                cert_path,
                key_path,
                config.quic_idle_timeout_secs,
                config.quic_zero_rtt,
            )
            .await
            {
                Ok(rx) => {
                    info!(
                        "HTTP/3 over QUIC serving on {} as the primary transport",
                        quic_addr
                    );
                    quic_rx = Some(rx);
                }
                Err(e) => {
                    error!(
                        "HTTP/3 listener failed to start: {}. Serving the TCP fallback only.",
                        e
                    );
                }
            }
        } else {
            info!(
                "HTTP/3 is the primary transport but needs tls_cert_path and tls_key_path.                  Serving the TCP fallback only until they are configured."
            );
        }
    }

    let connection_ceiling = PressureController::global().connections().ceiling();
    info!(
        "Zyron listening on {} (TCP), up to {} concurrent connections from this node's memory",
        addr, connection_ceiling
    );

    // Accept loop: select between TCP and QUIC connections.
    loop {
        tokio::select! {
            result = listener.accept() => {
                let (stream, peer_addr) = result?;
                let Some(slot) = ConnectionSlot::acquire() else {
                    warn!(
                        "refusing connection from {}: node memory affords no further connections",
                        peer_addr
                    );
                    drop(stream);
                    continue;
                };

                // TLS upgrade: if the client opens with an SSLRequest and the
                // server has a TLS acceptor, perform the handshake in the
                // accept loop before handing the connection to a task.
                match (server_state.tls_acceptor.as_ref(), server_state.tls_mode) {
                    (Some(acceptor), mode) if mode != tls::TlsMode::Disabled => {
                        let wants_tls = is_ssl_request(&stream).await.unwrap_or(false);
                        if wants_tls {
                            match upgrade_to_tls(stream, acceptor).await {
                                Ok(tls_stream) => {
                                    serve(tls_stream, peer_addr, server_state.clone(), slot, "TLS");
                                }
                                Err(e) => {
                                    error!("TLS handshake failed from {}: {}", peer_addr, e);
                                }
                            }
                        } else if mode == tls::TlsMode::Required {
                            // Plaintext attempts are rejected immediately.
                            error!("rejecting plaintext connection from {}", peer_addr);
                        } else {
                            serve(stream, peer_addr, server_state.clone(), slot, "TCP");
                        }
                    }
                    _ => serve(stream, peer_addr, server_state.clone(), slot, "TCP"),
                }
            }

            Some((quic_stream, peer_addr)) = async {
                match quic_rx.as_mut() {
                    Some(rx) => rx.recv().await,
                    None => std::future::pending().await,
                }
            } => {
                let Some(slot) = ConnectionSlot::acquire() else {
                    warn!(
                        "refusing QUIC connection from {}: node memory affords no further connections",
                        peer_addr
                    );
                    continue;
                };
                serve(quic_stream, peer_addr, server_state.clone(), slot, "QUIC");
            }
        }
    }
}

/// Puts one connection on the runtime and holds its memory slot until it ends.
///
/// The slot is moved into the task rather than released here, so a connection
/// that panics or is cancelled still gives its slot back through the guard's
/// Drop instead of leaking it out of the ceiling.
fn serve<T>(
    stream: T,
    peer_addr: SocketAddr,
    state: Arc<ServerState>,
    slot: ConnectionSlot,
    transport: &'static str,
) where
    T: crate::transport::WireTransport + Send + 'static,
{
    tokio::spawn(async move {
        debug!("{} connection from {}", transport, peer_addr);
        let mut conn = Connection::new(stream, state, Some(peer_addr.ip().to_string()));
        if let Err(e) = conn.run().await {
            error!("Connection error from {}: {}", peer_addr, e);
        }
        debug!("Connection closed: {}", peer_addr);
        drop(slot);
    });
}

/// Handles a single TCP connection. Useful for testing and embedding.
pub async fn handle_connection(stream: tokio::net::TcpStream, server_state: Arc<ServerState>) {
    let mut conn = Connection::new(stream, server_state, None);
    if let Err(e) = conn.run().await {
        error!("Connection error: {}", e);
    }
}

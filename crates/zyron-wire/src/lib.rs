//! PostgreSQL wire protocol v3 implementation for ZyronDB.
//!
//! Provides a server that speaks the PostgreSQL wire protocol over TCP
//! and QUIC transports. Any PostgreSQL-compatible client (psql, psycopg2,
//! node-postgres, JDBC) can connect via TCP. QUIC transport provides
//! mandatory TLS 1.3, 0-RTT connection resumption, and connection migration.

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

pub mod auth;
pub mod codec;
pub mod connection;
pub mod copy;
pub mod copy_external_dispatch;
pub mod ddl_dispatch;
pub mod endpoint_registrar;
pub mod messages;
pub mod notifications;
pub mod pem;
pub mod pg_client;
pub mod pool;
pub mod quic;
pub mod session;
pub mod stat_views;
pub mod subscription;
pub mod tls;
pub mod transport;
pub mod types;
pub mod uri;
pub mod zyron_sink;
pub mod zyron_source;

use std::net::SocketAddr;
use std::sync::Arc;

use tokio::net::TcpListener;
use tokio::sync::Semaphore;
use tokio::task::LocalSet;
use tracing::{debug, error, info};

use zyron_common::ServerConfig;

use crate::connection::{Connection, ServerState};

pub use crate::endpoint_registrar::EndpointRegistrar;
pub use crate::zyron_sink::{ZyronSinkClient, build_sink_client_from_entry};

/// Creates a TCP listener with SO_REUSEADDR for fast server restarts
/// (no waiting for TIME_WAIT sockets to expire) and SO_REUSEPORT on Linux
/// for kernel-level load balancing across accept loops.
fn create_tcp_listener(addr: SocketAddr) -> std::io::Result<std::net::TcpListener> {
    let socket = socket2::Socket::new(
        socket2::Domain::for_address(addr),
        socket2::Type::STREAM,
        Some(socket2::Protocol::TCP),
    )?;
    socket.set_reuse_address(true)?;
    #[cfg(target_os = "linux")]
    socket.set_reuse_port(true)?;
    socket.bind(&addr.into())?;
    socket.listen(1024)?;
    socket.set_nonblocking(true)?;
    Ok(std::net::TcpListener::from(socket))
}

/// Message sent from the accept loop to a worker thread.
enum ConnectionTask {
    Tcp {
        stream: tokio::net::TcpStream,
        peer_addr: SocketAddr,
        state: Arc<ServerState>,
        _permit: tokio::sync::OwnedSemaphorePermit,
    },
    Tls {
        stream: tokio_rustls::server::TlsStream<tokio::net::TcpStream>,
        peer_addr: SocketAddr,
        state: Arc<ServerState>,
        _permit: tokio::sync::OwnedSemaphorePermit,
    },
    Quic {
        stream: quic::QuicStream,
        peer_addr: SocketAddr,
        state: Arc<ServerState>,
        _permit: tokio::sync::OwnedSemaphorePermit,
    },
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
/// Pre-spawns a pool of worker threads, each with a persistent tokio runtime
/// and LocalSet (required for the planner's !Send futures). The accept loop
/// distributes connections round-robin via a crossbeam channel, so each
/// connection reuses an existing runtime instead of spawning a new thread.
///
/// When QUIC is enabled, listens on both TCP and UDP simultaneously.
/// Both transports feed into the same worker pool.
pub async fn start_server(
    config: &ServerConfig,
    server_state: Arc<ServerState>,
) -> Result<(), Box<dyn std::error::Error>> {
    let addr: SocketAddr = format!("{}:{}", config.host, config.port).parse()?;
    let std_listener = create_tcp_listener(addr)?;
    let listener = TcpListener::from_std(std_listener)?;
    let semaphore = Arc::new(Semaphore::new(config.max_connections as usize));

    let num_workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    let (sender, receiver) = crossbeam::channel::bounded::<ConnectionTask>(num_workers * 8);

    // Spawn persistent worker threads, each with its own runtime + LocalSet.
    // LocalSet is created once per thread and reused across connections to
    // avoid per-connection allocation overhead.
    for worker_id in 0..num_workers {
        let rx = receiver.clone();
        std::thread::Builder::new()
            .name(format!("zyron-worker-{}", worker_id))
            .spawn(move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("Failed to create worker runtime");
                let local = LocalSet::new();

                while let Ok(task) = rx.recv() {
                    match task {
                        ConnectionTask::Tcp {
                            stream,
                            peer_addr,
                            state,
                            _permit,
                        } => {
                            local.block_on(&rt, async {
                                debug!("TCP connection from {} on worker {}", peer_addr, worker_id);
                                let mut conn = Connection::new(
                                    stream,
                                    state,
                                    Some(peer_addr.ip().to_string()),
                                );
                                if let Err(e) = conn.run().await {
                                    error!("Connection error from {}: {}", peer_addr, e);
                                }
                                debug!("Connection closed: {}", peer_addr);
                                drop(_permit);
                            });
                        }
                        ConnectionTask::Tls {
                            stream,
                            peer_addr,
                            state,
                            _permit,
                        } => {
                            local.block_on(&rt, async {
                                debug!("TLS connection from {} on worker {}", peer_addr, worker_id);
                                let mut conn = Connection::new(
                                    stream,
                                    state,
                                    Some(peer_addr.ip().to_string()),
                                );
                                if let Err(e) = conn.run().await {
                                    error!("Connection error from {}: {}", peer_addr, e);
                                }
                                debug!("Connection closed: {}", peer_addr);
                                drop(_permit);
                            });
                        }
                        ConnectionTask::Quic {
                            stream,
                            peer_addr,
                            state,
                            _permit,
                        } => {
                            local.block_on(&rt, async {
                                debug!(
                                    "QUIC connection from {} on worker {}",
                                    peer_addr, worker_id
                                );
                                let mut conn = Connection::new(
                                    stream,
                                    state,
                                    Some(peer_addr.ip().to_string()),
                                );
                                if let Err(e) = conn.run().await {
                                    error!("Connection error from {}: {}", peer_addr, e);
                                }
                                debug!("Connection closed: {}", peer_addr);
                                drop(_permit);
                            });
                        }
                    }
                }
            })
            .expect("Failed to spawn worker thread");
    }

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
            )
            .await
            {
                Ok(rx) => {
                    info!("QUIC transport enabled on {}", quic_addr);
                    quic_rx = Some(rx);
                }
                Err(e) => {
                    error!("Failed to start QUIC listener: {}. TCP-only mode.", e);
                }
            }
        } else {
            error!("QUIC requires tls_cert_path and tls_key_path. TCP-only mode.");
        }
    }

    info!(
        "ZyronDB listening on {} (TCP) with {} workers",
        addr, num_workers
    );

    // Accept loop: select between TCP and QUIC connections.
    loop {
        tokio::select! {
            result = listener.accept() => {
                let (stream, peer_addr) = result?;
                let permit = semaphore.clone().acquire_owned().await?;

                // TLS upgrade: if the client opens with an SSLRequest and the
                // server has a TLS acceptor, perform the handshake in the
                // accept loop before handing off to a worker.
                let task = match (
                    server_state.tls_acceptor.as_ref(),
                    server_state.tls_mode,
                ) {
                    (Some(acceptor), mode)
                        if mode != tls::TlsMode::Disabled =>
                    {
                        let wants_tls = is_ssl_request(&stream).await.unwrap_or(false);
                        if wants_tls {
                            match upgrade_to_tls(stream, acceptor).await {
                                Ok(tls_stream) => ConnectionTask::Tls {
                                    stream: tls_stream,
                                    peer_addr,
                                    state: server_state.clone(),
                                    _permit: permit,
                                },
                                Err(e) => {
                                    error!(
                                        "TLS handshake failed from {}: {}",
                                        peer_addr, e
                                    );
                                    drop(permit);
                                    continue;
                                }
                            }
                        } else if mode == tls::TlsMode::Required {
                            // Plaintext attempts are rejected immediately.
                            error!("rejecting plaintext connection from {}", peer_addr);
                            drop(permit);
                            continue;
                        } else {
                            ConnectionTask::Tcp {
                                stream,
                                peer_addr,
                                state: server_state.clone(),
                                _permit: permit,
                            }
                        }
                    }
                    _ => ConnectionTask::Tcp {
                        stream,
                        peer_addr,
                        state: server_state.clone(),
                        _permit: permit,
                    },
                };

                if sender.send(task).is_err() {
                    error!("All worker threads have exited");
                    break;
                }
            }

            Some((quic_stream, peer_addr)) = async {
                match quic_rx.as_mut() {
                    Some(rx) => rx.recv().await,
                    None => std::future::pending().await,
                }
            } => {
                let permit = semaphore.clone().acquire_owned().await?;

                let task = ConnectionTask::Quic {
                    stream: quic_stream,
                    peer_addr,
                    state: server_state.clone(),
                    _permit: permit,
                };

                if sender.send(task).is_err() {
                    error!("All worker threads have exited");
                    break;
                }
            }
        }
    }

    Ok(())
}

/// Handles a single TCP connection. Useful for testing and embedding.
pub async fn handle_connection(stream: tokio::net::TcpStream, server_state: Arc<ServerState>) {
    let mut conn = Connection::new(stream, server_state, None);
    if let Err(e) = conn.run().await {
        error!("Connection error: {}", e);
    }
}

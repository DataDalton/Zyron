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
pub mod ddl_dispatch;
pub mod messages;
pub mod notifications;
pub mod quic;
pub mod session;
pub mod stat_views;
pub mod transport;
pub mod types;

use std::net::SocketAddr;
use std::sync::Arc;

use tokio::net::TcpListener;
use tokio::sync::Semaphore;
use tokio::task::LocalSet;
use tracing::{debug, error, info};

use zyron_common::ServerConfig;

use crate::connection::{Connection, ServerState};

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
    Quic {
        stream: quic::QuicStream,
        peer_addr: SocketAddr,
        state: Arc<ServerState>,
        _permit: tokio::sync::OwnedSemaphorePermit,
    },
}

/// Starts the wire protocol server on the configured address.
/// Pre-spawns a pool of worker threads, each with a persistent tokio runtime
/// and LocalSet (required for the planner's !Send futures). The accept loop
/// distributes connections round-robin via a crossbeam channel, eliminating
/// per-connection thread spawn and runtime creation overhead (~200us each).
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

                let task = ConnectionTask::Tcp {
                    stream,
                    peer_addr,
                    state: server_state.clone(),
                    _permit: permit,
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

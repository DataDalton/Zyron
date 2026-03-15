//! PostgreSQL wire protocol v3 implementation for ZyronDB.
//!
//! Provides a TCP server that speaks the PostgreSQL wire protocol,
//! enabling any PostgreSQL-compatible client (psql, psycopg2,
//! node-postgres, JDBC) to connect and execute queries against ZyronDB.

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

pub mod auth;
pub mod codec;
pub mod connection;
pub mod copy;
pub mod messages;
pub mod session;
pub mod types;

use std::net::SocketAddr;
use std::sync::Arc;

use tokio::net::TcpListener;
use tokio::sync::Semaphore;
use tokio::task::LocalSet;
use tracing::{error, info};

use zyron_common::ServerConfig;

use crate::connection::{Connection, ServerState};

/// Message sent from the accept loop to a worker thread.
struct ConnectionTask {
    stream: tokio::net::TcpStream,
    peer_addr: SocketAddr,
    state: Arc<ServerState>,
    _permit: tokio::sync::OwnedSemaphorePermit,
}

/// Starts the wire protocol server on the configured address.
/// Pre-spawns a pool of worker threads, each with a persistent tokio runtime
/// and LocalSet (required for the planner's !Send futures). The accept loop
/// distributes connections round-robin via a crossbeam channel, eliminating
/// per-connection thread spawn and runtime creation overhead (~200us each).
pub async fn start_server(
    config: &ServerConfig,
    server_state: Arc<ServerState>,
) -> Result<(), Box<dyn std::error::Error>> {
    let addr = format!("{}:{}", config.host, config.port);
    let listener = TcpListener::bind(&addr).await?;
    let semaphore = Arc::new(Semaphore::new(config.max_connections as usize));

    let num_workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    let (sender, receiver) = crossbeam::channel::unbounded::<ConnectionTask>();

    // Spawn persistent worker threads, each with its own runtime + LocalSet.
    for worker_id in 0..num_workers {
        let rx = receiver.clone();
        std::thread::Builder::new()
            .name(format!("zyron-worker-{}", worker_id))
            .spawn(move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("Failed to create worker runtime");

                while let Ok(task) = rx.recv() {
                    let local = LocalSet::new();
                    local.block_on(&rt, async move {
                        info!("Connection from {} on worker {}", task.peer_addr, worker_id);
                        let mut conn = Connection::new(task.stream, task.state);
                        if let Err(e) = conn.run().await {
                            error!("Connection error from {}: {}", task.peer_addr, e);
                        }
                        info!("Connection closed: {}", task.peer_addr);
                        drop(task._permit);
                    });
                }
            })
            .expect("Failed to spawn worker thread");
    }

    info!("ZyronDB listening on {} with {} workers", addr, num_workers);

    loop {
        let permit = semaphore.clone().acquire_owned().await?;
        let (stream, peer_addr) = listener.accept().await?;

        let task = ConnectionTask {
            stream,
            peer_addr,
            state: server_state.clone(),
            _permit: permit,
        };

        // If all workers are busy, this blocks the accept loop until one is free,
        // which is the desired backpressure behavior.
        if sender.send(task).is_err() {
            error!("All worker threads have exited");
            break;
        }
    }

    Ok(())
}

/// Handles a single connection. Useful for testing and embedding.
pub async fn handle_connection(stream: tokio::net::TcpStream, server_state: Arc<ServerState>) {
    let mut conn = Connection::new(stream, server_state);
    if let Err(e) = conn.run().await {
        error!("Connection error: {}", e);
    }
}

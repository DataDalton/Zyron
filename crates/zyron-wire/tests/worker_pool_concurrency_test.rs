//! Many clients at once must all be served, not served in turn.
//!
//! The server used to pre-spawn one thread per core, each with a
//! current-thread runtime, and drive a connection to completion on it. That
//! capped the server at one client per core: every client past the core count
//! was accepted, queued, and then never served until an earlier one
//! disconnected. For a database that is a hang, not slowness.
//!
//! Connections are now one task each on the work-stealing runtime, so this
//! opens well more clients than the machine has cores, holds every one of them
//! open on a barrier, and requires all of them to complete a query.
//!
//! Run: cargo test -p zyron-wire --test worker_pool_concurrency_test

mod common;

use std::sync::Arc;
use std::time::Duration;

use common::create_test_server;
use zyron_wire::pg_client::{ClientConfig, PgClient};

/// Takes an ephemeral port from the OS and releases it, so the server can
/// bind it without this test hard-coding one.
async fn reserve_port() -> u16 {
    let probe = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind probe");
    let port = probe.local_addr().expect("addr").port();
    drop(probe);
    port
}

async fn wait_until_listening(port: u16) {
    for _ in 0..200 {
        if tokio::net::TcpStream::connect(("127.0.0.1", port))
            .await
            .is_ok()
        {
            return;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    panic!("server never began listening on port {port}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn every_client_past_the_core_count_is_still_served() {
    let (state, _schema, _tmp) = create_test_server().await;
    let port = reserve_port().await;

    let mut config = zyron_common::config::ServerConfig::default();
    config.host = "127.0.0.1".to_string();
    config.port = port;
    config.quic_enabled = false;
    // The old pool sized itself from the core count, and the point of the test
    // is to stand well clear of it in client count
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let clients = cores * 3 + 8;

    // Nothing configures how many clients may connect. The ceiling comes from
    // the memory the node measured, and this asserts the machine running the
    // test actually affords the load being applied, so a failure below reads
    // as a serving failure rather than as a refusal
    let ceiling = zyron_pressure::pressure_control::PressureController::global()
        .connections()
        .ceiling();
    assert!(
        ceiling as usize >= clients,
        "node ceiling is {ceiling}, below the {clients} clients this test opens"
    );

    let leaked: &'static zyron_common::config::ServerConfig = Box::leak(Box::new(config));
    let server_state: Arc<_> = state;
    tokio::spawn(async move {
        let _ = zyron_wire::start_server(leaked, server_state).await;
    });
    wait_until_listening(port).await;

    // Every client holds its connection open until all of them have run a
    // query. A server that serves one connection per thread can never get the
    // whole set past this point, because the clients it has not reached are
    // waiting on the ones it will not release
    let all_served = Arc::new(tokio::sync::Barrier::new(clients));

    let mut handles = Vec::with_capacity(clients);
    for _ in 0..clients {
        let all_served = Arc::clone(&all_served);
        handles.push(tokio::spawn(async move {
            let cfg = ClientConfig {
                user: "zyron".to_string(),
                database: "zyron".to_string(),
                application_name: "worker-pool-test".to_string(),
                password: None,
                connect_timeout: Duration::from_secs(10),
                statement_timeout: Duration::from_secs(10),
            };
            let addr: std::net::SocketAddr = ([127, 0, 0, 1], port).into();
            let mut client = PgClient::connect(addr, &cfg)
                .await
                .map_err(|e| format!("handshake: {e}"))?;
            client
                .simple_query("SELECT 1")
                .await
                .map_err(|e| format!("query: {e}"))?;
            all_served.wait().await;
            Ok::<(), String>(())
        }));
    }

    let all = futures::future::join_all(handles);
    let results = tokio::time::timeout(Duration::from_secs(60), all)
        .await
        .unwrap_or_else(|_| {
            panic!(
                "{clients} clients against {cores} cores did not all complete: \
                 connections are being served one at a time"
            )
        });

    let mut failures = Vec::new();
    for (i, r) in results.into_iter().enumerate() {
        match r.expect("client task panicked") {
            Ok(()) => {}
            Err(e) => failures.push(format!("client {i}: {e}")),
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {clients} clients failed: {:?}",
        failures.len(),
        &failures[..failures.len().min(5)]
    );
}

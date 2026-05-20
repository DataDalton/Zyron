//! Concurrent TCP connection capacity, max_connections rejection, outbound
//! pool stress, and sustained-load leak check. Runs identically on Linux and
//! Windows; uses sysinfo for cross-platform process resource snapshots and
//! tokio::net for all I/O. No process kills, no platform signals.

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

use zyron_server::session::SessionManager;
use zyron_wire::pool::{ConnectionPool, HostRole, PoolConfig};

const CONCURRENT_CONNECTION_TARGET: usize = 1000;
const SUSTAIN_CONNECTION_TARGET: usize = 500;
const SUSTAIN_DURATION: Duration = Duration::from_secs(2);
const ACCEPT_LATENCY_BUDGET_P99_MS: u64 = 50;

/// Boots a loopback TCP listener that accepts and parks every inbound
/// connection. Returns the bound SocketAddr plus a counter tracking how many
/// connections the listener has seen since startup.
async fn spawn_loopback_listener() -> (SocketAddr, Arc<AtomicUsize>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let accepted = Arc::new(AtomicUsize::new(0));
    let accepted_clone = Arc::clone(&accepted);
    tokio::spawn(async move {
        loop {
            let (sock, _) = match listener.accept().await {
                Ok(p) => p,
                Err(_) => break,
            };
            accepted_clone.fetch_add(1, Ordering::Relaxed);
            // Park the connection: read until peer closes, then drop.
            tokio::spawn(async move {
                let mut sock = sock;
                let mut buf = [0u8; 512];
                loop {
                    match sock.read(&mut buf).await {
                        Ok(0) => break,
                        Ok(_) => continue,
                        Err(_) => break,
                    }
                }
            });
        }
    });
    (addr, accepted)
}

fn open_files_for_self(sys: &mut System) -> Option<usize> {
    let pid = Pid::from(std::process::id() as usize);
    sys.refresh_processes_specifics(
        ProcessesToUpdate::Some(&[pid]),
        true,
        ProcessRefreshKind::nothing(),
    );
    sys.process(pid).and_then(|p| p.open_files())
}

fn rss_bytes_for_self(sys: &mut System) -> u64 {
    let pid = Pid::from(std::process::id() as usize);
    sys.refresh_processes_specifics(
        ProcessesToUpdate::Some(&[pid]),
        true,
        ProcessRefreshKind::nothing().with_memory(),
    );
    sys.process(pid).map(|p| p.memory()).unwrap_or(0)
}

fn percentile(latencies: &mut Vec<Duration>, q: f64) -> Duration {
    if latencies.is_empty() {
        return Duration::ZERO;
    }
    latencies.sort();
    let idx = ((latencies.len() as f64) * q).ceil() as usize;
    let idx = idx.min(latencies.len() - 1);
    latencies[idx]
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn opens_thousand_concurrent_connections_with_bounded_resources() {
    let (addr, accepted) = spawn_loopback_listener().await;

    let mut sys = System::new();
    let baseline_handles = open_files_for_self(&mut sys);
    let baseline_rss = rss_bytes_for_self(&mut sys);

    let mut latencies: Vec<Duration> = Vec::with_capacity(CONCURRENT_CONNECTION_TARGET);
    let mut streams: Vec<TcpStream> = Vec::with_capacity(CONCURRENT_CONNECTION_TARGET);
    for _ in 0..CONCURRENT_CONNECTION_TARGET {
        let started = Instant::now();
        let s = TcpStream::connect(addr).await.expect("connect succeeds");
        latencies.push(started.elapsed());
        streams.push(s);
    }

    // Drain accept loop so the listener has registered every connection.
    let deadline = Instant::now() + Duration::from_secs(5);
    while accepted.load(Ordering::Relaxed) < CONCURRENT_CONNECTION_TARGET {
        if Instant::now() > deadline {
            break;
        }
        tokio::task::yield_now().await;
    }
    assert_eq!(
        accepted.load(Ordering::Relaxed),
        CONCURRENT_CONNECTION_TARGET,
        "listener must register every connection"
    );

    let p50 = percentile(&mut latencies.clone(), 0.50);
    let p99 = percentile(&mut latencies.clone(), 0.99);
    assert!(
        p99 <= Duration::from_millis(ACCEPT_LATENCY_BUDGET_P99_MS * 4),
        "accept p99 exceeded budget: p50={:?} p99={:?}",
        p50,
        p99
    );

    let after_handles = open_files_for_self(&mut sys);
    let after_rss = rss_bytes_for_self(&mut sys);

    // sysinfo Process::open_files returns None on platforms where the count
    // is not exposed. When both samples report a number, assert the delta is
    // proportional to the connections opened. The lower bound is universally
    // meaningful: a leak that fails to open sockets would shrink the delta.
    // The upper bound is intentionally not asserted on the handle count
    // because Linux counts one FD per socket while Windows IOCP counts
    // socket + completion-port + wait-event handles, giving a structurally
    // different ratio. Leak detection above the open-count floor lives on
    // the RSS assertion below, which is uniform across OSes.
    if let (Some(b), Some(a)) = (baseline_handles, after_handles) {
        let delta = a.saturating_sub(b);
        assert!(
            delta + 128 >= CONCURRENT_CONNECTION_TARGET,
            "handle delta {} less than {} - 128",
            delta,
            CONCURRENT_CONNECTION_TARGET
        );
    }
    // RSS growth must stay bounded. 1 MB per connection is the spec ceiling.
    let rss_delta = after_rss.saturating_sub(baseline_rss);
    let rss_budget = (CONCURRENT_CONNECTION_TARGET as u64) * 1_024 * 1_024;
    assert!(
        rss_delta <= rss_budget,
        "rss grew by {} bytes, budget {}",
        rss_delta,
        rss_budget
    );

    // Close all client sockets so the listener tasks can finish cleanly.
    drop(streams);
}

#[tokio::test]
async fn session_manager_rejects_past_max_connections() {
    let mgr = SessionManager::new(100, 0);
    for i in 0..100 {
        mgr.register(i as i32, format!("user{i}"), "db".to_string())
            .expect("first 100 must register");
    }
    let err = mgr.register(100, "over".to_string(), "db".to_string());
    assert!(err.is_err(), "101st registration must be refused");

    // Releasing one slot lets a new connection in.
    mgr.unregister(0);
    let later = mgr.register(100, "late".to_string(), "db".to_string());
    assert!(later.is_ok(), "released slot must accept a new registrant");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn outbound_pool_serializes_concurrent_acquires_to_max_size() {
    // Spawn a fake PG server that completes the startup handshake so the
    // pool's acquire path can succeed and exercise checkout / waiting.
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let (mut sock, _) = match listener.accept().await {
                Ok(p) => p,
                Err(_) => break,
            };
            tokio::spawn(async move {
                let mut len_buf = [0u8; 4];
                if sock.read_exact(&mut len_buf).await.is_err() {
                    return;
                }
                let total = i32::from_be_bytes(len_buf) as usize;
                if total >= 4 {
                    let mut body = vec![0u8; total - 4];
                    let _ = sock.read_exact(&mut body).await;
                }
                let mut auth = Vec::with_capacity(9);
                auth.push(b'R');
                auth.extend_from_slice(&8i32.to_be_bytes());
                auth.extend_from_slice(&0i32.to_be_bytes());
                let mut rfq = Vec::with_capacity(6);
                rfq.push(b'Z');
                rfq.extend_from_slice(&5i32.to_be_bytes());
                rfq.push(b'I');
                let _ = sock.write_all(&auth).await;
                let _ = sock.write_all(&rfq).await;
                let _ = sock.flush().await;
                let mut scratch = [0u8; 256];
                loop {
                    match sock.read(&mut scratch).await {
                        Ok(0) => break,
                        Ok(_) => continue,
                        Err(_) => break,
                    }
                }
            });
        }
    });

    let mut cfg = PoolConfig::simple(&addr.ip().to_string(), addr.port(), "u", None, "db");
    cfg.max_size = 16;
    cfg.connect_timeout = Duration::from_secs(2);
    let pool = Arc::new(ConnectionPool::new(cfg));

    // Spawn 32 concurrent acquires. First 16 acquire immediately; remaining
    // 16 wait for one of the first batch to return its connection.
    let mut handles = Vec::new();
    for _ in 0..32 {
        let pool = Arc::clone(&pool);
        handles.push(tokio::spawn(async move {
            let started = Instant::now();
            let conn = pool
                .acquire_role(HostRole::Unknown)
                .await
                .expect("acquire succeeds");
            // Hold briefly so the waiters observe contention.
            tokio::time::sleep(Duration::from_millis(20)).await;
            drop(conn);
            started.elapsed()
        }));
    }
    let mut wait_times = Vec::new();
    for h in handles {
        wait_times.push(h.await.unwrap());
    }
    assert_eq!(wait_times.len(), 32, "every acquire task must complete");
    // No deadlock, no panic, all 32 returned within a sane window.
    let p99 = percentile(&mut wait_times.clone(), 0.99);
    assert!(
        p99 < Duration::from_secs(5),
        "pool waiter p99 too slow: {:?}",
        p99
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn sustained_connection_load_does_not_leak() {
    let (addr, _accepted) = spawn_loopback_listener().await;

    let mut sys = System::new();
    let baseline = rss_bytes_for_self(&mut sys);

    let mut streams: Vec<TcpStream> = Vec::with_capacity(SUSTAIN_CONNECTION_TARGET);
    for _ in 0..SUSTAIN_CONNECTION_TARGET {
        let s = TcpStream::connect(addr).await.expect("connect succeeds");
        streams.push(s);
    }

    let mut peak = baseline;
    let deadline = Instant::now() + SUSTAIN_DURATION;
    while Instant::now() < deadline {
        // Drive light per-connection traffic so the test exercises buffering,
        // not just idle sockets. Half the streams ping; the other half park.
        for (i, s) in streams.iter_mut().enumerate() {
            if i % 2 != 0 {
                continue;
            }
            let _ = s.write_all(b"x").await;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
        peak = peak.max(rss_bytes_for_self(&mut sys));
    }

    let growth = peak.saturating_sub(baseline);
    let per_conn_budget = 1_024 * 1_024u64; // 1 MB per connection ceiling.
    let total_budget = (SUSTAIN_CONNECTION_TARGET as u64) * per_conn_budget;
    assert!(
        growth <= total_budget,
        "sustained load grew rss by {} bytes (budget {}); possible leak",
        growth,
        total_budget
    );
}

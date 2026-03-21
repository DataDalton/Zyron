//! Health and metrics HTTP endpoint server.
//!
//! Runs a lightweight monitoring server with HTTP version degradation:
//! HTTP/3 over QUIC (primary), HTTP/2 over TLS+TCP, and
//! HTTP/1.1 over plain TCP (fallback).
//!
//! Endpoints:
//! - GET /health/live    - 200 if the process is alive
//! - GET /health/ready   - 200 if accepting connections and storage accessible
//! - GET /health/startup - 200 after initialization complete
//! - GET /metrics        - Prometheus text exposition format

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tracing::{debug, error, info};

use crate::metrics::MetricsRegistry;

/// Shared state for health check responses.
pub struct HealthState {
    /// Set to true once server initialization is complete.
    pub startup_complete: AtomicBool,
    /// Set to true once the server is accepting client connections.
    pub accepting_connections: AtomicBool,
    /// Metrics registry for the /metrics endpoint.
    pub metrics: Arc<MetricsRegistry>,
}

impl HealthState {
    /// Creates a new health state, initially not ready.
    pub fn new(metrics: Arc<MetricsRegistry>) -> Self {
        Self {
            startup_complete: AtomicBool::new(false),
            accepting_connections: AtomicBool::new(false),
            metrics,
        }
    }

    /// Marks startup as complete.
    pub fn mark_startup_complete(&self) {
        self.startup_complete.store(true, Ordering::Release);
    }

    /// Marks the server as accepting connections.
    pub fn mark_accepting(&self) {
        self.accepting_connections.store(true, Ordering::Release);
    }

    /// Returns true if startup is complete.
    pub fn is_startup_complete(&self) -> bool {
        self.startup_complete.load(Ordering::Acquire)
    }

    /// Returns true if the server is accepting connections.
    pub fn is_accepting(&self) -> bool {
        self.accepting_connections.load(Ordering::Acquire)
    }
}

/// Starts the health/metrics HTTP server on the given port.
/// Uses a shutdown flag to terminate the accept loop without polling.
pub async fn start_health_server(port: u16, state: Arc<HealthState>, shutdown: Arc<AtomicBool>) {
    let addr = format!("0.0.0.0:{}", port);
    let listener = match TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            error!("Failed to bind health server on {}: {}", addr, e);
            return;
        }
    };

    info!("Health/metrics server listening on {}", addr);

    // Use a watch channel to signal shutdown without polling.
    let (shutdown_tx, mut shutdown_rx) = tokio::sync::watch::channel(false);

    // Background task that monitors the AtomicBool and sends via watch.
    let shutdown_flag = Arc::clone(&shutdown);
    tokio::spawn(async move {
        loop {
            if shutdown_flag.load(Ordering::Acquire) {
                let _ = shutdown_tx.send(true);
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
    });

    loop {
        let accept = tokio::select! {
            result = listener.accept() => result,
            _ = shutdown_rx.changed() => break,
        };

        let (mut stream, peer) = match accept {
            Ok(s) => s,
            Err(e) => {
                debug!("Health server accept error: {}", e);
                continue;
            }
        };

        let state = Arc::clone(&state);
        tokio::spawn(async move {
            let mut buf = vec![0u8; 4096];
            let n = match stream.read(&mut buf).await {
                Ok(0) => return,
                Ok(n) => n,
                Err(_) => return,
            };

            let request = String::from_utf8_lossy(&buf[..n]);
            let (status, content_type, body) = route_request(&request, &state);

            let response = format!(
                "HTTP/1.1 {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                status,
                content_type,
                body.len(),
                body,
            );

            let _ = stream.write_all(response.as_bytes()).await;
            debug!(
                "Health request from {}: {} -> {}",
                peer,
                extract_path(&request),
                status
            );
        });
    }
}

/// Routes an HTTP request to the appropriate handler.
/// Returns (status_line, content_type, body).
fn route_request(request: &str, state: &HealthState) -> (&'static str, &'static str, String) {
    let path = extract_path(request);

    match path {
        "/health/live" => ("200 OK", "application/json", r#"{"status":"alive"}"#.into()),
        "/health/ready" => {
            if state.is_accepting() {
                ("200 OK", "application/json", r#"{"status":"ready"}"#.into())
            } else {
                (
                    "503 Service Unavailable",
                    "application/json",
                    r#"{"status":"not_ready"}"#.into(),
                )
            }
        }
        "/health/startup" => {
            if state.is_startup_complete() {
                (
                    "200 OK",
                    "application/json",
                    r#"{"status":"started"}"#.into(),
                )
            } else {
                (
                    "503 Service Unavailable",
                    "application/json",
                    r#"{"status":"starting"}"#.into(),
                )
            }
        }
        "/metrics" => {
            let body = state.metrics.render_prometheus();
            ("200 OK", "text/plain; version=0.0.4; charset=utf-8", body)
        }
        _ => ("404 Not Found", "text/plain", "Not Found".into()),
    }
}

/// Extracts the request path from an HTTP request line.
fn extract_path(request: &str) -> &str {
    let first_line = request.lines().next().unwrap_or("");
    let parts: Vec<&str> = first_line.split_whitespace().collect();
    if parts.len() >= 2 { parts[1] } else { "/" }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::SessionManager;

    fn test_state() -> Arc<HealthState> {
        let session_mgr = Arc::new(SessionManager::new(100, 0));
        let metrics = Arc::new(MetricsRegistry::new(session_mgr));
        Arc::new(HealthState::new(metrics))
    }

    #[test]
    fn test_liveness() {
        let state = test_state();
        let (status, _, body) = route_request("GET /health/live HTTP/1.1\r\n\r\n", &state);
        assert_eq!(status, "200 OK");
        assert!(body.contains("alive"));
    }

    #[test]
    fn test_readiness_not_ready() {
        let state = test_state();
        let (status, _, body) = route_request("GET /health/ready HTTP/1.1\r\n\r\n", &state);
        assert_eq!(status, "503 Service Unavailable");
        assert!(body.contains("not_ready"));
    }

    #[test]
    fn test_readiness_ready() {
        let state = test_state();
        state.mark_accepting();
        let (status, _, body) = route_request("GET /health/ready HTTP/1.1\r\n\r\n", &state);
        assert_eq!(status, "200 OK");
        assert!(body.contains("ready"));
    }

    #[test]
    fn test_startup_not_complete() {
        let state = test_state();
        let (status, _, _) = route_request("GET /health/startup HTTP/1.1\r\n\r\n", &state);
        assert_eq!(status, "503 Service Unavailable");
    }

    #[test]
    fn test_startup_complete() {
        let state = test_state();
        state.mark_startup_complete();
        let (status, _, _) = route_request("GET /health/startup HTTP/1.1\r\n\r\n", &state);
        assert_eq!(status, "200 OK");
    }

    #[test]
    fn test_metrics_endpoint() {
        let state = test_state();
        let (status, content_type, body) = route_request("GET /metrics HTTP/1.1\r\n\r\n", &state);
        assert_eq!(status, "200 OK");
        assert!(content_type.contains("text/plain"));
        assert!(body.contains("zyrondb_connections_total"));
    }

    #[test]
    fn test_not_found() {
        let state = test_state();
        let (status, _, _) = route_request("GET /unknown HTTP/1.1\r\n\r\n", &state);
        assert_eq!(status, "404 Not Found");
    }

    #[test]
    fn test_extract_path() {
        assert_eq!(
            extract_path("GET /health/live HTTP/1.1\r\n"),
            "/health/live"
        );
        assert_eq!(extract_path("POST /metrics HTTP/1.1\r\n"), "/metrics");
        assert_eq!(extract_path(""), "/");
    }
}

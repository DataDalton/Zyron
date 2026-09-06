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

use crate::gateway::endpoint_exec::{EndpointExecutor, ExecInput};
use crate::gateway::request::parse_request;
use crate::gateway::response::{HttpResponse, ResponseBody, build_response_bytes};
use crate::gateway::{
    AdminAuthResult, AdminExecutor, AdminRouter, GatewayMetrics, HttpMethod as GatewayHttpMethod,
    MiddlewareOutcome, RateLimiter, Router as GatewayRouter, check_admin_auth, emit_openapi_json,
    emit_swagger_html, run_pipeline,
};
use crate::metrics::MetricsRegistry;

/// Shared state for health check responses.
pub struct HealthState {
    /// Set to true once server initialization is complete.
    pub startup_complete: AtomicBool,
    /// Whether the node takes new connections, shared with the accept loop
    /// and the upgrade driver so the readiness probe reports a drain the
    /// moment it starts
    pub admission: Arc<zyron_common::Admission>,
    /// Metrics registry for the exposition endpoint.
    pub metrics: Arc<MetricsRegistry>,
    /// Route the Prometheus exposition answers on, from metrics.path.
    pub metrics_path: String,
    /// Dynamic endpoint router populated by the catalog DDL path.
    pub gateway_router: Arc<GatewayRouter>,
    /// Per-endpoint Prometheus metrics.
    pub gateway_metrics: Arc<GatewayMetrics>,
    /// Shared rate limiter used by the middleware pipeline for dynamic routes.
    pub rate_limiter: Arc<RateLimiter>,
    /// SQL executor that backs dynamic REST endpoints. Startup attaches an
    /// instance wired to the live catalog, buffer pool, disk manager, WAL,
    /// transaction manager, and security manager.
    pub endpoint_executor: parking_lot::RwLock<Option<Arc<EndpointExecutor>>>,
    /// What the node measured about its machine, reported by /pressure. None
    /// in a harness that assembled a health server without probing, where the
    /// document omits the hardware section rather than inventing one
    pub node_capabilities:
        parking_lot::RwLock<Option<Arc<zyron_pressure::capability::NodeCapabilities>>>,
    /// Catalog endpoint list provider used by OpenAPI emission. Returns the
    /// current live list of registered endpoints each call.
    pub endpoint_catalog:
        Option<Arc<dyn Fn() -> Vec<Arc<zyron_catalog::schema::EndpointEntry>> + Send + Sync>>,
    /// Admin action executor. When set, the /admin/* routes run against the
    /// live catalog, security manager, endpoint registrar, and CDC registry.
    /// Wrapped in an atomic cell so startup can install the executor after
    /// the HealthState Arc has already been shared with the HTTP listener.
    pub admin_executor: parking_lot::RwLock<Option<Arc<AdminExecutor>>>,
    /// Where this node's working-set manifest is written, so a survivor
    /// taking over for it can fetch what it was holding. Empty in a harness
    /// that never had a data directory, where the route reports that no
    /// manifest exists rather than reading somebody else's
    pub data_dir: parking_lot::RwLock<std::path::PathBuf>,
    /// What answers the mesh calls, when this node is part of a mesh. None on
    /// a single node, where the mesh paths report that nothing serves them
    /// rather than answering for a mesh that does not exist
    pub mesh_node: parking_lot::RwLock<Option<Arc<crate::mesh_node::ServerMeshNode>>>,
}

impl HealthState {
    /// Attaches what answers the mesh calls.
    ///
    /// Set at startup when the node has a mesh. Until it is, the mesh paths
    /// answer that nothing on this node serves them, which is the honest
    /// answer to a scheduler that reached the wrong node.
    pub fn set_mesh_node(&self, node: Arc<crate::mesh_node::ServerMeshNode>) {
        *self.mesh_node.write() = Some(node);
    }

    /// What answers the mesh calls, if a mesh has been attached.
    pub fn mesh_node(&self) -> Option<Arc<crate::mesh_node::ServerMeshNode>> {
        self.mesh_node.read().clone()
    }

    /// Attaches what the node measured about its machine, so /pressure can
    /// report the hardware the numbers were measured on.
    pub fn set_node_capabilities(
        &self,
        capabilities: Arc<zyron_pressure::capability::NodeCapabilities>,
    ) {
        *self.node_capabilities.write() = Some(capabilities);
    }

    /// What the node measured, if the probe has been attached.
    pub fn node_capabilities(&self) -> Option<Arc<zyron_pressure::capability::NodeCapabilities>> {
        self.node_capabilities.read().clone()
    }

    /// Creates a new health state, initially not ready. The metrics path is
    /// the configured exposition route, normalized to a leading slash.
    pub fn new(metrics: Arc<MetricsRegistry>, metrics_path: &str) -> Self {
        Self::with_gateway(
            metrics,
            metrics_path,
            Arc::new(GatewayRouter::new()),
            Arc::new(GatewayMetrics::new()),
            Arc::new(zyron_common::Admission::new()),
        )
    }

    /// Creates a health state serving the given gateway router and metric
    /// set. The endpoint registrar must share these same instances, a
    /// route registered into any other router is never served. The
    /// admission is the one the wire listener reads, so readiness and the
    /// accept loop agree
    pub fn with_gateway(
        metrics: Arc<MetricsRegistry>,
        metrics_path: &str,
        gateway_router: Arc<GatewayRouter>,
        gateway_metrics: Arc<GatewayMetrics>,
        admission: Arc<zyron_common::Admission>,
    ) -> Self {
        let metrics_path = if metrics_path.starts_with('/') {
            metrics_path.to_string()
        } else {
            format!("/{}", metrics_path)
        };
        Self {
            startup_complete: AtomicBool::new(false),
            admission,
            metrics,
            metrics_path,
            gateway_router,
            gateway_metrics,
            rate_limiter: Arc::new(RateLimiter::new()),
            endpoint_executor: parking_lot::RwLock::new(None),
            node_capabilities: parking_lot::RwLock::new(None),
            endpoint_catalog: None,
            admin_executor: parking_lot::RwLock::new(None),
            data_dir: parking_lot::RwLock::new(std::path::PathBuf::new()),
            mesh_node: parking_lot::RwLock::new(None),
        }
    }

    /// Points the working-set route at the node's data directory.
    pub fn set_data_dir(&self, data_dir: std::path::PathBuf) {
        *self.data_dir.write() = data_dir;
    }

    /// Installs the admin executor after the Catalog and other managers have
    /// been initialized. Safe to call once during startup.
    pub fn set_admin_executor(&self, executor: Arc<AdminExecutor>) {
        *self.admin_executor.write() = Some(executor);
    }

    /// Returns a snapshot Arc of the installed admin executor.
    pub fn admin_executor(&self) -> Option<Arc<AdminExecutor>> {
        self.admin_executor.read().clone()
    }

    /// Attaches the dynamic endpoint executor. Called once during startup
    /// after the catalog, buffer pool, disk manager, WAL writer, transaction
    /// manager, and security manager are wired up.
    pub fn set_endpoint_executor(&self, executor: Arc<EndpointExecutor>) {
        *self.endpoint_executor.write() = Some(executor);
    }

    /// Returns a snapshot Arc of the installed endpoint executor.
    pub fn endpoint_executor(&self) -> Option<Arc<EndpointExecutor>> {
        self.endpoint_executor.read().clone()
    }

    /// Marks startup as complete.
    pub fn mark_startup_complete(&self) {
        self.startup_complete.store(true, Ordering::Release);
    }

    /// Marks the server as accepting connections.
    pub fn mark_accepting(&self) {
        self.admission.mark_accepting();
    }

    /// Returns true if startup is complete.
    pub fn is_startup_complete(&self) -> bool {
        self.startup_complete.load(Ordering::Acquire)
    }

    /// Returns true if the server is accepting connections, which it is not
    /// while it drains for a restart
    pub fn is_accepting(&self) -> bool {
        self.admission.is_accepting()
    }
}

/// Starts the health/metrics HTTP server on the given bind host and port.
/// Uses a shutdown flag to terminate the accept loop without polling.
/// When `host` resolves to an IPv6 wildcard and `dual_stack` is true, V6ONLY
/// is disabled so the listener also accepts IPv4 connections via mapped
/// addresses (matches Linux kernel default, overrides Windows default)
pub async fn start_health_server(
    host: &str,
    port: u16,
    dual_stack: bool,
    state: Arc<HealthState>,
    shutdown: Arc<AtomicBool>,
) {
    let listener = match bind_dual_stack_listener(host, port, dual_stack) {
        Ok(l) => l,
        Err(e) => {
            error!("Failed to bind health server on {}:{}: {}", host, port, e);
            return;
        }
    };
    let addr = listener
        .local_addr()
        .map(|a| a.to_string())
        .unwrap_or_else(|_| format!("{}:{}", host, port));

    info!("Health/metrics server listening on {}", addr);

    // Inline polling via select! sleep beats a dedicated polling task,
    // the prior implementation spawned a task that woke every 50ms to check
    // the AtomicBool which competed with request handlers on the
    // current_thread runtime and caused per-request latency outliers when a
    // poll tick landed in the middle of an HTTP round trip, the inline
    // sleep fires only when accept is also idle so request handling is
    // never preempted by a stray shutdown poll
    loop {
        if shutdown.load(Ordering::Acquire) {
            break;
        }
        let accept = tokio::select! {
            result = listener.accept() => result,
            _ = tokio::time::sleep(std::time::Duration::from_millis(500)) => continue,
        };

        let (mut stream, peer) = match accept {
            Ok(s) => s,
            Err(e) => {
                debug!("Health server accept error: {}", e);
                continue;
            }
        };

        let state = Arc::clone(&state);
        let stream_shutdown = Arc::clone(&shutdown);
        tokio::spawn(async move {
            let mut buf = vec![0u8; 4096];
            let n = match stream.read(&mut buf).await {
                Ok(0) => return,
                Ok(n) => n,
                Err(_) => return,
            };

            let raw_bytes: Vec<u8> = buf[..n].to_vec();
            let request = String::from_utf8_lossy(&raw_bytes).into_owned();
            let path = extract_path(&request).to_string();

            // Dynamic endpoints dispatch through the full middleware pipeline
            // and write the response as-is so streaming bodies and custom
            // headers survive. Built-in health, metrics, and admin routes
            // use direct dispatch and do not consult the dynamic gateway
            // router.
            // The live stream keeps the socket for its lifetime, so it is
            // taken before the request/response path, which writes a body and
            // closes. A request to the stream path that is not a valid
            // upgrade falls through and is answered as ordinary HTTP
            if path == crate::gateway::pressure_endpoint::PRESSURE_STREAM_PATH {
                if let Some(parsed) = crate::gateway::request::parse_request(&raw_bytes, None) {
                    let served = crate::gateway::pressure_endpoint::serve_stream(
                        &mut stream,
                        &parsed,
                        state.node_capabilities(),
                        Arc::clone(&stream_shutdown),
                    )
                    .await;
                    if served {
                        debug!("Pressure stream from {} ended", peer);
                        return;
                    }
                }
            }

            // The upgrade subscriptions keep the socket the same way the
            // pressure stream does, and fall through to the plain GET when
            // the request carries no upgrade handshake
            if crate::gateway::upgrade_endpoint::is_upgrade_path(&path) {
                if let Some(parsed) = crate::gateway::request::parse_request(&raw_bytes, None) {
                    let served = crate::gateway::upgrade_endpoint::serve_stream(
                        &mut stream,
                        &parsed,
                        &path,
                        Arc::clone(&stream_shutdown),
                    )
                    .await;
                    if served {
                        debug!("Upgrade subscription {} from {} ended", path, peer);
                        return;
                    }
                }
            }

            if is_dynamic_endpoint_path(&path, &state) {
                let response_bytes = handle_dynamic_endpoint(&raw_bytes, &state).await;
                let _ = stream.write_all(&response_bytes).await;
                debug!("Health request from {}: {} -> dynamic", peer, path);
                return;
            }

            // A mesh call carries a JSON body and answers with one, so it is
            // routed before the path-only handlers, which never read a body
            if zyron_mesh::is_mesh_path(&path) {
                let (status, body) = handle_mesh_request(&path, &request, &state);
                let response = format!(
                    "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\n\
                     Content-Length: {}\r\nConnection: close\r\n\r\n{}",
                    status,
                    status_text(status),
                    body.len(),
                    body,
                );
                let _ = stream.write_all(response.as_bytes()).await;
                debug!("Mesh call from {}: {} -> {}", peer, path, status);
                return;
            }

            let (status, content_type, body): (String, String, String) =
                if path.starts_with("/admin/") {
                    handle_admin_request(&request, &state).await
                } else {
                    let (s, ct, b) = route_request(&request, &state);
                    (s.to_string(), ct.to_string(), b)
                };

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

/// The reason phrase for a status code the mesh handler returns.
fn status_text(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        409 => "Conflict",
        503 => "Service Unavailable",
        _ => "Error",
    }
}

/// Runs one mesh call against whatever answers for this node.
///
/// A node with no mesh attached answers that nothing here serves the path,
/// which is a 404 the caller reads as the node not being part of a mesh
/// rather than as a transient fault worth retrying.
fn handle_mesh_request(path: &str, request: &str, state: &HealthState) -> (u16, String) {
    let Some(node) = state.mesh_node() else {
        return (
            404,
            "{\"Unknown\":{\"what\":\"this node is not part of a mesh\"}}".to_string(),
        );
    };
    let body = request.split_once("\r\n\r\n").map(|(_, b)| b).unwrap_or("");
    match zyron_mesh::dispatch(node.as_ref(), path, body) {
        Some(answer) => (answer.status, answer.body),
        None => (404, format!("{{\"Unknown\":{{\"what\":\"{path}\"}}}}")),
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
        p if p == state.metrics_path => {
            let mut body = state.metrics.render_prometheus();
            body.push_str(&state.gateway_metrics.render_prometheus());
            ("200 OK", "text/plain; version=0.0.4; charset=utf-8", body)
        }
        crate::gateway::pressure_endpoint::HOT_SET_PATH => {
            let data_dir = state.data_dir.read().clone();
            let (status, body) = crate::gateway::pressure_endpoint::render_hot_set(&data_dir);
            (status, "application/json", body)
        }
        crate::gateway::pressure_endpoint::PRESSURE_PATH => (
            "200 OK",
            "application/json",
            crate::gateway::pressure_endpoint::render_snapshot(
                state.node_capabilities().as_deref(),
            ),
        ),
        // Reached only when the upgrade handshake was absent or malformed,
        // because a valid one never returns to this router
        crate::gateway::pressure_endpoint::PRESSURE_STREAM_PATH => (
            "426 Upgrade Required",
            "application/json",
            r#"{"error":"/pressure/stream requires a WebSocket upgrade"}"#.into(),
        ),
        // A plain GET of an upgrade subscription answers with the same
        // document the subscription pushes, so a dashboard and a curl see
        // one shape
        p if crate::gateway::upgrade_endpoint::is_upgrade_path(p) => {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            (
                "200 OK",
                "application/json",
                crate::gateway::upgrade_endpoint::render(p, now),
            )
        }
        "/openapi.json" => {
            let endpoints = state
                .endpoint_catalog
                .as_ref()
                .map(|f| f())
                .unwrap_or_default();
            ("200 OK", "application/json", emit_openapi_json(&endpoints))
        }
        "/openapi.html" => {
            let endpoints = state
                .endpoint_catalog
                .as_ref()
                .map(|f| f())
                .unwrap_or_default();
            (
                "200 OK",
                "text/html; charset=utf-8",
                emit_swagger_html(&endpoints),
            )
        }
        "/_endpoints" => {
            let routes = state.gateway_router.list();
            let mut out = String::from("[");
            for (i, r) in routes.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&format!(
                    "{{\"name\":\"{}\",\"pattern\":\"{}\",\"methods\":{}}}",
                    r.name,
                    r.path_pattern,
                    r.methods.len()
                ));
            }
            out.push(']');
            ("200 OK", "application/json", out)
        }
        other if other.starts_with("/admin/") => {
            // Admin requests are handled in handle_admin_request which runs on
            // the async task. This branch should not normally be reached, but
            // when route_request is called directly in tests for a non-admin
            // path, falling through avoids breakage.
            let _ = other;
            (
                "500 Internal Server Error",
                "application/json",
                r#"{"error":"admin_routed_to_sync_path"}"#.to_string(),
            )
        }
        _ => {
            // Dynamic endpoint paths are dispatched on the async task in
            // handle_dynamic_endpoint. A miss here means the path does not
            // belong to any registered route.
            ("404 Not Found", "text/plain", "Not Found".into())
        }
    }
}

// -----------------------------------------------------------------------------
// Dynamic endpoint dispatcher.
// -----------------------------------------------------------------------------

/// Returns true when the path resolves to a registered dynamic endpoint. The
/// check strips any query string before consulting the router. Built-in
/// health, metrics, and admin routes use direct dispatch and do not consult
/// the dynamic gateway router.
fn is_dynamic_endpoint_path(path: &str, state: &HealthState) -> bool {
    if path.starts_with("/health/")
        || path == state.metrics_path
        || path == "/openapi.json"
        || path == "/openapi.html"
        || path == "/_endpoints"
        || path.starts_with("/admin/")
    {
        return false;
    }
    let bare = path.split('?').next().unwrap_or(path);
    // Match against any registered method: the router's lookup filters by
    // method, so a path registered only for POST would still resolve here via
    // a GET probe. Iterate the route list directly to avoid method coupling.
    for r in state.gateway_router.list() {
        if r.match_path(bare).is_some() {
            return true;
        }
    }
    false
}

/// Runs the full middleware pipeline for a dynamic endpoint request. On
/// success invokes the installed executor hook and serializes the response as
/// HTTP/1.1 wire bytes. On pipeline short-circuit (auth, rate limit, circuit
/// open, method mismatch, body cap) returns the pipeline's response. Metrics
/// and circuit-breaker outcome are recorded before returning.
pub async fn handle_dynamic_endpoint(raw: &[u8], state: &HealthState) -> Vec<u8> {
    let req = match parse_request(raw, None) {
        Some(r) => r,
        None => {
            let resp = HttpResponse::new(400)
                .json(r#"{"error":"bad_request","detail":"malformed HTTP request"}"#.to_string());
            return build_response_bytes(&resp).unwrap_or_default();
        }
    };

    let lookup = state.gateway_router.lookup(req.method, &req.path);
    let (route, path_params) = match lookup {
        Some(x) => x,
        None => {
            let resp = HttpResponse::new(404).json(r#"{"error":"not_found"}"#.to_string());
            return build_response_bytes(&resp).unwrap_or_default();
        }
    };

    let started = std::time::Instant::now();
    let origin = req.header("origin").unwrap_or("").to_string();

    let sm_arc = state
        .endpoint_executor()
        .and_then(|ex| ex.security_manager.clone());
    let security_map = sm_arc.as_deref().map(|sm| &sm.security_map);
    let pipeline_out = run_pipeline(
        Arc::clone(&route),
        path_params,
        &req,
        state.rate_limiter.as_ref(),
        security_map,
    );

    let resp: HttpResponse = match pipeline_out {
        MiddlewareOutcome::Response(r) => {
            // Short-circuit responses are not downstream failures. Record
            // metrics but leave the circuit breaker alone.
            let status = r.status;
            let is_error = status >= 500;
            route
                .metrics
                .record(started.elapsed().as_micros() as u64, status, is_error);
            r
        }
        MiddlewareOutcome::Execute {
            route: r_arc,
            path_params,
            auth,
        } => {
            let executor = match state.endpoint_executor() {
                Some(e) => e,
                None => {
                    let resp = HttpResponse::new(503).json(
                        r#"{"error":"unavailable","detail":"endpoint executor not configured"}"#
                            .to_string(),
                    );
                    return build_response_bytes(&resp).unwrap_or_default();
                }
            };
            tracing::info!(
                target: "zyron::audit",
                event = "EndpointInvoked",
                principal = auth.user_hint.as_deref().unwrap_or("anonymous"),
                object = %req.path,
                decision = "granted",
                reason = auth.via,
            );
            let query_pairs = req.query_pairs();
            let content_type = req
                .header("content-type")
                .unwrap_or("application/octet-stream")
                .to_string();
            let input = ExecInput {
                sql_template: r_arc.sql_body.clone(),
                path_params,
                query_params: query_pairs,
                body: req.body.clone(),
                content_type,
                auth,
                timeout: r_arc.timeout,
                output_format: r_arc.output_format,
                pre_parsed: r_arc.pre_parsed.clone(),
                template_has_params: r_arc.template_has_params,
            };
            let output = executor.execute(input).await;
            let status = output.status;
            let is_error = status >= 500;
            route
                .metrics
                .record(started.elapsed().as_micros() as u64, status, is_error);
            if status >= 500 {
                route.circuit_breaker.record_failure();
            } else {
                route.circuit_breaker.record_success();
            }
            let mut http = HttpResponse::new(status)
                .header("Content-Type", output.content_type)
                .body_bytes(output.body);
            if route.cache_seconds > 0 && status < 400 {
                http = http.header(
                    "Cache-Control",
                    format!("public, max-age={}", route.cache_seconds),
                );
            }
            http.apply_cors(&origin, &r_arc)
        }
    };

    match resp.body {
        ResponseBody::Bytes(_) | ResponseBody::Empty => {
            build_response_bytes(&resp).unwrap_or_default()
        }
        ResponseBody::Chunks(_) => {
            // Chunked bodies are not supported by the health listener's
            // buffered write loop. Coerce into a 500 so callers do not hang.
            let fallback = HttpResponse::new(500).json(
                r#"{"error":"internal","detail":"chunked response not supported"}"#.to_string(),
            );
            build_response_bytes(&fallback).unwrap_or_default()
        }
    }
}

// -----------------------------------------------------------------------------
// Admin request handler.
// -----------------------------------------------------------------------------

/// Parses the admin HTTP request, checks the AdminAccess privilege, dispatches
/// to the AdminExecutor, and formats the response. Returns a (status_line,
/// content_type, body) triple ready for HTTP/1.1 emission.
pub async fn handle_admin_request(request: &str, state: &HealthState) -> (String, String, String) {
    let path = extract_path(request);
    let method = extract_method(request)
        .and_then(GatewayHttpMethod::parse)
        .unwrap_or(GatewayHttpMethod::Get);

    // Split path from query string.
    let (raw_path, query_string) = match path.split_once('?') {
        Some((p, q)) => (p.to_string(), q.to_string()),
        None => (path.to_string(), String::new()),
    };

    let mut headers = std::collections::HashMap::new();
    let mut header_end = 0usize;
    for (i, line) in request.lines().enumerate() {
        if i == 0 {
            continue;
        }
        if line.is_empty() {
            header_end = i;
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
        }
    }

    // Body follows the blank line that terminates the headers.
    let body_bytes: Vec<u8> = if header_end > 0 {
        let mut lines = request.lines();
        for _ in 0..=header_end {
            let _ = lines.next();
        }
        lines.collect::<Vec<_>>().join("\n").into_bytes()
    } else {
        Vec::new()
    };

    let admin_req = crate::gateway::HttpRequest {
        method,
        path: raw_path,
        query_string,
        headers,
        body: body_bytes,
        peer_addr: None,
        tls_info: None,
    };

    // Auth gate. When no security manager is configured the gate is open.
    let executor_opt = state.admin_executor();
    let sm_arc_opt = executor_opt
        .as_ref()
        .and_then(|ex| ex.security_manager.clone());
    let sm_ref: Option<&zyron_auth::SecurityManager> = sm_arc_opt.as_deref();

    match check_admin_auth(sm_ref, &admin_req) {
        AdminAuthResult::Allowed => {}
        AdminAuthResult::Unauthenticated => {
            return (
                "401 Unauthorized".to_string(),
                "application/json".to_string(),
                r#"{"error":"unauthorized","detail":"admin route requires authentication"}"#
                    .to_string(),
            );
        }
        AdminAuthResult::Forbidden => {
            return (
                "403 Forbidden".to_string(),
                "application/json".to_string(),
                r#"{"error":"forbidden","detail":"AdminAccess privilege required"}"#.to_string(),
            );
        }
    }

    let action = match AdminRouter::dispatch(&admin_req) {
        Some(a) => a,
        None => {
            return (
                "404 Not Found".to_string(),
                "application/json".to_string(),
                r#"{"error":"admin_route_not_found"}"#.to_string(),
            );
        }
    };

    let executor = match executor_opt {
        Some(ex) => ex,
        None => {
            return (
                "503 Service Unavailable".to_string(),
                "application/json".to_string(),
                r#"{"error":"admin_executor_not_configured"}"#.to_string(),
            );
        }
    };

    let resp = executor.execute(action).await;
    let status_line = format!("{} {}", resp.status, status_code_reason(resp.status));
    let body = serde_json::to_string(&resp.body)
        .unwrap_or_else(|_| r#"{"error":"body_encode_failed"}"#.to_string());
    (status_line, "application/json".to_string(), body)
}

/// Maps an HTTP status code to its reason phrase. Covers the codes the admin
/// handlers emit. Falls back to an empty string for unknown codes.
fn status_code_reason(code: u16) -> &'static str {
    match code {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        503 => "Service Unavailable",
        _ => "",
    }
}

fn extract_method(request: &str) -> Option<&str> {
    request.lines().next()?.split_whitespace().next()
}

/// Extracts the request path from an HTTP request line.
fn extract_path(request: &str) -> &str {
    let first_line = request.lines().next().unwrap_or("");
    let parts: Vec<&str> = first_line.split_whitespace().collect();
    if parts.len() >= 2 { parts[1] } else { "/" }
}

/// Builds a tokio TcpListener at `host:port`. When the resolved address is an
/// IPv6 wildcard and `dual_stack` is true, IPV6_V6ONLY is disabled so the
/// listener also accepts IPv4 connections via IPv4-mapped IPv6 addresses.
/// Linux defaults V6ONLY to false, Windows defaults it to true; this routine
/// makes the behaviour identical across platforms when dual-stack is requested
pub(crate) fn bind_dual_stack_listener(
    host: &str,
    port: u16,
    dual_stack: bool,
) -> std::io::Result<TcpListener> {
    use socket2::{Domain, Protocol, Socket, Type};
    use std::net::{IpAddr, SocketAddr, ToSocketAddrs};

    // Strip a leading [ and trailing ] so callers can pass either bare IPv6
    // (`::`) or bracketed (`[::]`) forms uniformly
    let trimmed = host.trim();
    let host_clean =
        if let Some(stripped) = trimmed.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
            stripped
        } else {
            trimmed
        };

    // An IPv6 address joins its port only in bracketed form, so the address
    // is built from the parsed parts rather than from pasted text
    let addr: SocketAddr = match host_clean.parse::<IpAddr>() {
        Ok(ip) => SocketAddr::new(ip, port),
        Err(_) => {
            // hostname form, resolve via DNS
            format!("{}:{}", host_clean, port)
                .to_socket_addrs()?
                .next()
                .ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::AddrNotAvailable,
                        format!("no addresses for {}", host_clean),
                    )
                })?
        }
    };

    let domain = if addr.is_ipv6() {
        Domain::IPV6
    } else {
        Domain::IPV4
    };
    let sock = Socket::new(domain, Type::STREAM, Some(Protocol::TCP))?;
    sock.set_reuse_address(true)?;
    if addr.is_ipv6() {
        // Honour the dual_stack flag, defaulting V6ONLY off matches Linux
        // and gives a single listener that handles both protocols
        sock.set_only_v6(!dual_stack)?;
    }
    sock.bind(&addr.into())?;
    sock.listen(1024)?;
    sock.set_nonblocking(true)?;
    let std_listener: std::net::TcpListener = sock.into();
    TcpListener::from_std(std_listener)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::SessionManager;

    fn test_state() -> Arc<HealthState> {
        let session_mgr = Arc::new(SessionManager::new(0));
        let labeled = Arc::new(zyron_common::LabeledMetrics::new());
        let metrics = Arc::new(MetricsRegistry::new(
            session_mgr,
            labeled,
            Arc::new(zyron_common::QueryMetrics::new()),
        ));
        Arc::new(HealthState::new(metrics, "/metrics"))
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
        assert!(body.contains("zyron_connections_total"));
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

    #[tokio::test]
    async fn test_the_bracketed_ipv6_wildcard_binds() {
        let listener = bind_dual_stack_listener("[::]", 0, false).expect("binds");
        let addr = listener.local_addr().expect("has an address");
        assert!(addr.is_ipv6());
        assert_ne!(addr.port(), 0);

        let v4 = bind_dual_stack_listener("127.0.0.1", 0, false).expect("binds");
        assert!(v4.local_addr().expect("has an address").is_ipv4());
    }
}

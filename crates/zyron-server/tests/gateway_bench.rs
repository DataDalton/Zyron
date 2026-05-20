//! HTTP gateway benchmark suite.
//!
//! Covers router lookup, path parameter extraction, middleware pipeline cost,
//! OpenAPI emission, WebSocket framing, Server-Sent Events emission, and
//! endpoint parameter substitution.
//!
//! Run: cargo test -p zyron-server --test gateway_bench --release -- --nocapture --test-threads=1

use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::{Arc, Mutex};

use zyron_catalog::schema::{
    EndpointAuthMode, EndpointEntry, EndpointKind, EndpointOutputFormat,
    HttpMethod as CatalogMethod,
};
use zyron_catalog::{EndpointId, SchemaId};

use zyron_server::gateway::middleware::MiddlewareOutcome;
use zyron_server::gateway::websocket::{decode_frame, encode_frame};
use zyron_server::gateway::{
    CompiledRoute, HttpMethod, HttpRequest, RateLimiter, Router, WsOpcode, emit_openapi_json,
    run_pipeline, sse,
};

use zyron_bench_harness::*;

// ---------------------------------------------------------------------------
// Performance targets
// ---------------------------------------------------------------------------

// Router holds 100 routes, lookup is O(N) over the route vector today. The
// target reflects that expected cost at N=100 with mixed static/param
// segments on a modern x86 machine.
const ROUTER_LOOKUP_TARGET_OPS: f64 = 2_000_000.0;

// Path param extraction on a two-param route, ops/sec.
const PARAM_EXTRACTION_TARGET_OPS: f64 = 1_000_000.0;

// Middleware pipeline with auth (none path), scope check, and circuit check.
const MIDDLEWARE_PIPELINE_TARGET_OPS: f64 = 100_000.0;

// Full OpenAPI JSON emission for 50 endpoints, target ms per emission.
const OPENAPI_EMISSION_TARGET_MS: f64 = 10.0;

// WebSocket frame encode+decode for 100-byte text frames, ops/sec.
const WS_TEXT_ROUNDTRIP_TARGET_OPS: f64 = 10_000_000.0;
// 1 KB binary frames pay data-copy cost.
const WS_BINARY_ROUNDTRIP_TARGET_OPS: f64 = 5_000_000.0;

// SSE event emission throughput, events/sec.
const SSE_EMIT_TARGET_OPS: f64 = 1_000_000.0;

// Endpoint template param substitution, ops/sec for a three-param template.
const PARAM_SUBSTITUTION_TARGET_OPS: f64 = 500_000.0;

// ---------------------------------------------------------------------------
// Benchmark infrastructure
// ---------------------------------------------------------------------------

static BENCHMARK_LOCK: Mutex<()> = Mutex::new(());

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Builds a CompiledRoute with the given pattern and methods. Keeps auth open
/// so the pipeline bench measures the code path that real requests follow on
/// the common no-auth case.
fn make_route(pattern: &str, methods: Vec<HttpMethod>) -> CompiledRoute {
    CompiledRoute::compile(
        EndpointId(1),
        "ep".to_string(),
        pattern.to_string(),
        methods,
        EndpointAuthMode::None,
        Vec::new(),
        EndpointOutputFormat::Json,
        vec!["*".to_string()],
        0,
        30,
        65_536,
        "SELECT 1".to_string(),
    )
}

/// Builds an HttpRequest for the pipeline bench with no auth and no body.
fn make_request(method: HttpMethod, path: &str) -> HttpRequest {
    HttpRequest {
        method,
        path: path.to_string(),
        query_string: String::new(),
        headers: HashMap::new(),
        body: Vec::new(),
        peer_addr: Some(SocketAddr::new(
            IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
            50_000,
        )),
        tls_info: None,
    }
}

/// Builds a catalog endpoint entry used by the OpenAPI emission bench.
fn make_endpoint_entry(idx: usize) -> Arc<EndpointEntry> {
    Arc::new(EndpointEntry {
        id: EndpointId(idx as u32),
        schema_id: SchemaId(1),
        name: format!("ep_{}", idx),
        kind: EndpointKind::Rest,
        path: format!("/api/v1/resource_{}/:id/items/:sku", idx),
        methods: vec![CatalogMethod::Get, CatalogMethod::Post],
        sql_body: "SELECT * FROM resource WHERE id = $id AND sku = $sku".to_string(),
        backed_publication_id: None,
        auth_mode: EndpointAuthMode::Jwt,
        required_scopes: vec!["read".to_string()],
        output_format: Some(EndpointOutputFormat::Json),
        cors_origins: vec!["*".to_string()],
        rate_limit: None,
        cache_seconds: Some(30),
        timeout_seconds: Some(30),
        max_request_body_kb: Some(64),
        message_format: None,
        heartbeat_seconds: None,
        backpressure: None,
        max_connections: None,
        enabled: true,
        owner_role_id: 1,
        created_at: 0,
    })
}

// ---------------------------------------------------------------------------
// Local implementation of parameter substitution.
//
// The production substitute_params function lives inside endpoint_exec and
// is not exported. This copy follows the same scanner rules so the bench
// number tracks the runtime behavior. If the production logic diverges the
// bench should be updated to match.
// ---------------------------------------------------------------------------

fn bench_substitute_params(
    template: &str,
    path_params: &HashMap<String, String>,
    query_params: &[(String, String)],
) -> Result<String, String> {
    let mut combined: HashMap<String, String> = HashMap::new();
    for (k, v) in query_params {
        combined.insert(k.clone(), v.clone());
    }
    for (k, v) in path_params {
        combined.insert(k.clone(), v.clone());
    }
    let mut out = String::with_capacity(template.len());
    let bytes = template.as_bytes();
    let mut i = 0usize;
    let mut used: Vec<String> = Vec::new();
    while i < bytes.len() {
        let c = bytes[i];
        if c == b'$' {
            let mut j = i + 1;
            while j < bytes.len() {
                let cj = bytes[j];
                if cj == b'_' || cj.is_ascii_alphanumeric() {
                    j += 1;
                } else {
                    break;
                }
            }
            if j > i + 1 {
                let name = &template[i + 1..j];
                let value = combined
                    .get(name)
                    .ok_or_else(|| format!("missing value for parameter '{}'", name))?;
                used.push(name.to_string());
                if !value.is_empty()
                    && value.chars().all(|c| c.is_ascii_digit() || c == '-')
                    && value.parse::<i64>().is_ok()
                {
                    out.push_str(value);
                } else {
                    out.push('\'');
                    for ch in value.chars() {
                        if ch == '\'' {
                            out.push('\'');
                            out.push('\'');
                        } else {
                            out.push(ch);
                        }
                    }
                    out.push('\'');
                }
                i = j;
                continue;
            }
        }
        out.push(c as char);
        i += 1;
    }
    for k in combined.keys() {
        if !used.iter().any(|u| u == k) {
            return Err(format!("unexpected parameter '{}'", k));
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Test 1: Router lookup throughput
// ---------------------------------------------------------------------------

#[test]
fn test_gateway_router_lookup() {
    zyron_bench_harness::init("gateway");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Router Lookup Throughput ===");

    let router = Router::new();
    // 80 static routes, 20 routes with a single param.
    for i in 0..80 {
        router.insert(make_route(
            &format!("/api/s_{}/resource", i),
            vec![HttpMethod::Get],
        ));
    }
    for i in 0..20 {
        router.insert(make_route(
            &format!("/api/p_{}/:id", i),
            vec![HttpMethod::Get],
        ));
    }

    let lookup_paths: Vec<String> = (0..100)
        .map(|i| {
            if i < 80 {
                format!("/api/s_{}/resource", i)
            } else {
                format!("/api/p_{}/42", i - 80)
            }
        })
        .collect();
    // Add 50 misses so the measurement reflects mix behavior.
    let miss_paths: Vec<String> = (0..50).map(|i| format!("/api/unknown_{}/x", i)).collect();

    let iterations = 1_000_000usize;
    let mut results = Vec::with_capacity(VALIDATION_RUNS);

    for run in 0..VALIDATION_RUNS {
        tprintln!("--- Run {}/{} ---\n", run + 1, VALIDATION_RUNS);

        let start = Instant::now();
        let mut hits = 0usize;
        for i in 0..iterations {
            let path = if i % 3 == 0 {
                &miss_paths[i % miss_paths.len()]
            } else {
                &lookup_paths[i % lookup_paths.len()]
            };
            if router.lookup(HttpMethod::Get, path).is_some() {
                hits += 1;
            }
        }
        let elapsed = start.elapsed();
        let ops = iterations as f64 / elapsed.as_secs_f64();
        results.push(ops);
        tprintln!(
            "  {} lookups ({} hits) in {:.2?}, {} ops/sec\n",
            format_with_commas(iterations as f64),
            format_with_commas(hits as f64),
            elapsed,
            format_with_commas(ops),
        );
    }

    let r = validate_metric(
        "Router Lookup",
        "Router.lookup throughput (ops/sec)",
        results,
        ROUTER_LOOKUP_TARGET_OPS,
        true,
    );
    assert!(r.passed, "router lookup below target");
    assert!(!r.regression_detected, "router lookup regression");
}

// ---------------------------------------------------------------------------
// Test 2: Path parameter extraction
// ---------------------------------------------------------------------------

#[test]
fn test_gateway_path_param_extraction() {
    zyron_bench_harness::init("gateway");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Path Param Extraction Throughput ===");

    let route = make_route("/api/orders/:id/items/:sku", vec![HttpMethod::Get]);
    let paths: Vec<String> = (0..1024)
        .map(|i| format!("/api/orders/{}/items/sku-{}", i, i))
        .collect();

    let iterations = 1_000_000usize;
    let mut results = Vec::with_capacity(VALIDATION_RUNS);

    for run in 0..VALIDATION_RUNS {
        tprintln!("--- Run {}/{} ---\n", run + 1, VALIDATION_RUNS);

        let start = Instant::now();
        let mut matched = 0usize;
        for i in 0..iterations {
            let p = &paths[i % paths.len()];
            if let Some(caps) = route.match_path(p) {
                matched += 1;
                std::hint::black_box(&caps);
            }
        }
        let elapsed = start.elapsed();
        let ops = iterations as f64 / elapsed.as_secs_f64();
        results.push(ops);
        tprintln!(
            "  {} extracts ({} matched) in {:.2?}, {} ops/sec\n",
            format_with_commas(iterations as f64),
            format_with_commas(matched as f64),
            elapsed,
            format_with_commas(ops),
        );
    }

    let r = validate_metric(
        "Path Param Extraction",
        "CompiledRoute.match_path throughput (ops/sec)",
        results,
        PARAM_EXTRACTION_TARGET_OPS,
        true,
    );
    assert!(r.passed, "path param extraction below target");
    assert!(!r.regression_detected, "path param extraction regression");
}

// ---------------------------------------------------------------------------
// Test 3: Middleware pipeline cost
// ---------------------------------------------------------------------------

#[test]
fn test_gateway_middleware_pipeline() {
    zyron_bench_harness::init("gateway");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Middleware Pipeline Throughput ===");

    let route = Arc::new(make_route("/api/ep", vec![HttpMethod::Get]));
    let rate_limiter = RateLimiter::new();
    let req = make_request(HttpMethod::Get, "/api/ep");

    let iterations = 100_000usize;
    let mut results = Vec::with_capacity(VALIDATION_RUNS);

    for run in 0..VALIDATION_RUNS {
        tprintln!("--- Run {}/{} ---\n", run + 1, VALIDATION_RUNS);

        let start = Instant::now();
        let mut exec_count = 0usize;
        for _ in 0..iterations {
            let out = run_pipeline(
                Arc::clone(&route),
                HashMap::new(),
                &req,
                &rate_limiter,
                None,
            );
            if matches!(out, MiddlewareOutcome::Execute { .. }) {
                exec_count += 1;
            }
        }
        let elapsed = start.elapsed();
        let ops = iterations as f64 / elapsed.as_secs_f64();
        results.push(ops);
        assert_eq!(exec_count, iterations, "every request should pass pipeline");
        let p99_us = elapsed.as_nanos() as f64 / iterations as f64 / 1000.0;
        tprintln!(
            "  {} pipeline runs in {:.2?}, {} ops/sec, {:.2} us/iter\n",
            format_with_commas(iterations as f64),
            elapsed,
            format_with_commas(ops),
            p99_us,
        );
    }

    let r = validate_metric(
        "Middleware Pipeline",
        "run_pipeline throughput (ops/sec)",
        results,
        MIDDLEWARE_PIPELINE_TARGET_OPS,
        true,
    );
    assert!(r.passed, "middleware pipeline below target");
    assert!(!r.regression_detected, "middleware pipeline regression");
}

// ---------------------------------------------------------------------------
// Test 4: OpenAPI emission cost
// ---------------------------------------------------------------------------

#[test]
fn test_gateway_openapi_emission() {
    zyron_bench_harness::init("gateway");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== OpenAPI Emission Cost ===");

    let endpoints: Vec<Arc<EndpointEntry>> = (0..50).map(make_endpoint_entry).collect();

    let iterations = 1_000usize;
    let mut results_ms = Vec::with_capacity(VALIDATION_RUNS);

    for run in 0..VALIDATION_RUNS {
        tprintln!("--- Run {}/{} ---\n", run + 1, VALIDATION_RUNS);

        let start = Instant::now();
        for _ in 0..iterations {
            let json = emit_openapi_json(&endpoints);
            std::hint::black_box(&json);
        }
        let elapsed = start.elapsed();
        let per_emission_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
        results_ms.push(per_emission_ms);
        tprintln!(
            "  {} emissions (50 endpoints) in {:.2?}, {:.3} ms/emission\n",
            format_with_commas(iterations as f64),
            elapsed,
            per_emission_ms,
        );
    }

    let r = validate_metric(
        "OpenAPI Emission",
        "emit_openapi_json(50) per emission (ms)",
        results_ms,
        OPENAPI_EMISSION_TARGET_MS,
        false,
    );
    assert!(r.passed, "openapi emission above target");
    assert!(!r.regression_detected, "openapi emission regression");
}

// ---------------------------------------------------------------------------
// Test 5: WebSocket frame roundtrip
// ---------------------------------------------------------------------------

#[test]
fn test_gateway_websocket_frame_roundtrip() {
    zyron_bench_harness::init("gateway");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== WebSocket Frame Encode+Decode Throughput ===");

    let text_payload = vec![b'a'; 100];
    let binary_payload = vec![0xA5u8; 1024];
    let ping_payload = vec![0u8; 0];

    let iterations = 1_000_000usize;

    let mut text_results = Vec::with_capacity(VALIDATION_RUNS);
    let mut binary_results = Vec::with_capacity(VALIDATION_RUNS);
    let mut ping_results = Vec::with_capacity(VALIDATION_RUNS);

    for run in 0..VALIDATION_RUNS {
        tprintln!("--- Run {}/{} ---\n", run + 1, VALIDATION_RUNS);

        // Text 100 B roundtrip.
        let start = Instant::now();
        for _ in 0..iterations {
            let frame = encode_frame(WsOpcode::Text, &text_payload, true);
            let (_n, decoded) = decode_frame(&frame).expect("decode");
            std::hint::black_box(&decoded);
        }
        let elapsed = start.elapsed();
        let ops = iterations as f64 / elapsed.as_secs_f64();
        text_results.push(ops);
        tprintln!(
            "  Text 100B: {} roundtrips in {:.2?}, {} ops/sec\n",
            format_with_commas(iterations as f64),
            elapsed,
            format_with_commas(ops),
        );

        // Binary 1 KB roundtrip.
        let start = Instant::now();
        for _ in 0..iterations {
            let frame = encode_frame(WsOpcode::Binary, &binary_payload, true);
            let (_n, decoded) = decode_frame(&frame).expect("decode");
            std::hint::black_box(&decoded);
        }
        let elapsed = start.elapsed();
        let ops = iterations as f64 / elapsed.as_secs_f64();
        binary_results.push(ops);
        tprintln!(
            "  Binary 1KB: {} roundtrips in {:.2?}, {} ops/sec\n",
            format_with_commas(iterations as f64),
            elapsed,
            format_with_commas(ops),
        );

        // Ping + Pong pair roundtrip.
        let start = Instant::now();
        for _ in 0..iterations {
            let ping = encode_frame(WsOpcode::Ping, &ping_payload, true);
            let (_n, _d) = decode_frame(&ping).expect("decode ping");
            let pong = encode_frame(WsOpcode::Pong, &ping_payload, true);
            let (_n, _d) = decode_frame(&pong).expect("decode pong");
        }
        let elapsed = start.elapsed();
        let ops = (iterations * 2) as f64 / elapsed.as_secs_f64();
        ping_results.push(ops);
        tprintln!(
            "  Ping+Pong: {} frames in {:.2?}, {} ops/sec\n",
            format_with_commas((iterations * 2) as f64),
            elapsed,
            format_with_commas(ops),
        );
    }

    let r = validate_metric(
        "WebSocket Text",
        "100-byte text frame roundtrip (ops/sec)",
        text_results,
        WS_TEXT_ROUNDTRIP_TARGET_OPS,
        true,
    );
    assert!(r.passed, "ws text roundtrip below target");
    assert!(!r.regression_detected, "ws text regression");

    let r = validate_metric(
        "WebSocket Binary",
        "1 KB binary frame roundtrip (ops/sec)",
        binary_results,
        WS_BINARY_ROUNDTRIP_TARGET_OPS,
        true,
    );
    assert!(r.passed, "ws binary roundtrip below target");
    assert!(!r.regression_detected, "ws binary regression");

    let r = validate_metric(
        "WebSocket Ping",
        "Ping+Pong frame roundtrip (ops/sec)",
        ping_results,
        WS_TEXT_ROUNDTRIP_TARGET_OPS,
        true,
    );
    assert!(r.passed, "ws ping roundtrip below target");
    assert!(!r.regression_detected, "ws ping regression");
}

// ---------------------------------------------------------------------------
// Test 6: SSE event emission
// ---------------------------------------------------------------------------

#[test]
fn test_gateway_sse_event_emit() {
    zyron_bench_harness::init("gateway");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== SSE Event Emission Throughput ===");

    let data = "{\"id\":42,\"event\":\"update\",\"value\":\"example\"}";
    let iterations = 1_000_000usize;
    let mut results = Vec::with_capacity(VALIDATION_RUNS);

    for run in 0..VALIDATION_RUNS {
        tprintln!("--- Run {}/{} ---\n", run + 1, VALIDATION_RUNS);

        // Sink that buffers events into a contiguous Vec<u8> so the bench
        // includes the copy into a shared write buffer like a real stream.
        let mut sink: Vec<u8> = Vec::with_capacity(iterations * 80);
        let start = Instant::now();
        for i in 0..iterations {
            let id = i.to_string();
            let bytes = sse::encode_event(None, data, Some(&id));
            sink.extend_from_slice(&bytes);
        }
        let elapsed = start.elapsed();
        let ops = iterations as f64 / elapsed.as_secs_f64();
        results.push(ops);
        tprintln!(
            "  {} events, {} bytes written in {:.2?}, {} events/sec\n",
            format_with_commas(iterations as f64),
            format_with_commas(sink.len() as f64),
            elapsed,
            format_with_commas(ops),
        );
        std::hint::black_box(&sink);
    }

    let r = validate_metric(
        "SSE Emission",
        "encode_event+sink.extend (events/sec)",
        results,
        SSE_EMIT_TARGET_OPS,
        true,
    );
    assert!(r.passed, "sse emission below target");
    assert!(!r.regression_detected, "sse emission regression");
}

// ---------------------------------------------------------------------------
// Test 7: Endpoint param substitution
// ---------------------------------------------------------------------------

#[test]
fn test_gateway_endpoint_param_substitution() {
    zyron_bench_harness::init("gateway");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Endpoint Param Substitution Throughput ===");

    let template = "SELECT * FROM orders WHERE id = $id AND sku = $sku AND region = $region";
    let mut path_params = HashMap::new();
    path_params.insert("id".to_string(), "42".to_string());
    path_params.insert("sku".to_string(), "widget-001".to_string());
    let query_params = vec![("region".to_string(), "us-west-2".to_string())];

    // One warmup run to confirm correctness before timing.
    let out = bench_substitute_params(template, &path_params, &query_params).expect("sub");
    assert!(out.contains("42"));
    assert!(out.contains("'widget-001'"));

    let iterations = 1_000_000usize;
    let mut results = Vec::with_capacity(VALIDATION_RUNS);

    for run in 0..VALIDATION_RUNS {
        tprintln!("--- Run {}/{} ---\n", run + 1, VALIDATION_RUNS);

        let start = Instant::now();
        for _ in 0..iterations {
            let s = bench_substitute_params(template, &path_params, &query_params).expect("sub");
            std::hint::black_box(&s);
        }
        let elapsed = start.elapsed();
        let ops = iterations as f64 / elapsed.as_secs_f64();
        results.push(ops);
        tprintln!(
            "  {} substitutions in {:.2?}, {} ops/sec\n",
            format_with_commas(iterations as f64),
            elapsed,
            format_with_commas(ops),
        );
    }

    let r = validate_metric(
        "Param Substitution",
        "3-param template substitution (ops/sec)",
        results,
        PARAM_SUBSTITUTION_TARGET_OPS,
        true,
    );
    assert!(r.passed, "param substitution below target");
    assert!(!r.regression_detected, "param substitution regression");
}

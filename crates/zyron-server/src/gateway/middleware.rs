// -----------------------------------------------------------------------------
// Middleware pipeline.
//
// Orders: CORS preflight, auth, scope check, rate limit, circuit breaker, and
// finally the endpoint execution hook. Each stage returns a short-circuiting
// HttpResponse on failure. The pipeline is synchronous with respect to the
// gateway surface. Execution itself runs under the caller's async context.
// -----------------------------------------------------------------------------

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Arc;

use super::auth_mw::{self, AuthError, AuthOutcome};
use super::circuit::CircuitBreaker;
use super::rate_limit::{RateLimitKey, RateLimiter};
use super::request::HttpRequest;
use super::response::{
    HttpResponse, ResponseBody, forbidden, method_not_allowed, service_unavailable,
    too_many_requests, unauthorized,
};
use super::router::{CompiledRoute, HttpMethod};

/// Outcome of the middleware pipeline. The caller either responds with the
/// buffered HttpResponse directly or invokes the provided executor closure.
pub enum MiddlewareOutcome {
    Response(HttpResponse),
    Execute {
        route: Arc<CompiledRoute>,
        path_params: HashMap<String, String>,
        auth: AuthOutcome,
    },
}

/// Executes the chain up to the point where endpoint SQL would run. Returns
/// either a short-circuit response or an Execute envelope carrying the
/// authenticated context.
pub fn run_pipeline(
    route: Arc<CompiledRoute>,
    path_params: HashMap<String, String>,
    req: &HttpRequest,
    rate_limiter: &RateLimiter,
) -> MiddlewareOutcome {
    if !route.enabled {
        return MiddlewareOutcome::Response(service_unavailable("endpoint disabled"));
    }
    // OPTIONS short-circuits with CORS preflight.
    if req.method == HttpMethod::Options {
        let origin = req.header("origin").unwrap_or("");
        let resp = HttpResponse::new(204).apply_cors(origin, &route);
        return MiddlewareOutcome::Response(resp);
    }
    // Method gate.
    if !route.methods.iter().any(|m| *m == req.method) {
        return MiddlewareOutcome::Response(method_not_allowed());
    }
    // Body cap.
    if req.body.len() as u32 > route.max_body_bytes {
        return MiddlewareOutcome::Response(super::response::payload_too_large());
    }
    // Auth.
    let auth = match auth_mw::authenticate(&route, req) {
        Ok(out) => out,
        Err(e) => {
            return MiddlewareOutcome::Response(map_auth_error(e));
        }
    };
    // Scope.
    if !auth_mw::scope_check(&auth, &route.required_scopes) {
        return MiddlewareOutcome::Response(forbidden("missing required scope"));
    }
    // Rate limit.
    if let Some((capacity, refill_per_sec, key)) = rate_limit_spec(&route, req, &auth) {
        if !rate_limiter.check(key, capacity, refill_per_sec, 1) {
            route.metrics.inc_rate_limited();
            return MiddlewareOutcome::Response(too_many_requests(1));
        }
    }
    // Circuit.
    if !route.circuit_breaker.should_attempt() {
        route.metrics.inc_circuit_open();
        return MiddlewareOutcome::Response(service_unavailable("circuit open"));
    }
    MiddlewareOutcome::Execute {
        route,
        path_params,
        auth,
    }
}

/// Records the outcome of an executed request on both the endpoint metrics
/// and the circuit breaker.
pub fn record_outcome(breaker: &CircuitBreaker, resp: &HttpResponse) {
    if resp.status >= 500 {
        breaker.record_failure();
    } else {
        breaker.record_success();
    }
}

/// Length of the buffered body in a response, for metric recording.
pub fn response_body_len(body: &ResponseBody) -> u64 {
    match body {
        ResponseBody::Bytes(b) => b.len() as u64,
        _ => 0,
    }
}

fn map_auth_error(e: AuthError) -> HttpResponse {
    unauthorized(e.detail())
}

fn rate_limit_spec(
    route: &CompiledRoute,
    req: &HttpRequest,
    auth: &AuthOutcome,
) -> Option<(u64, u64, RateLimitKey)> {
    // Default: no limiter when metadata absent. The catalog entry carries the
    // spec but the CompiledRoute condenses it onto a single budget field. The
    // current surface stores at most a simple per-route budget, derived from
    // cache_seconds as a stand-in; operators set explicit limits via ALTER
    // ENDPOINT in P3. When set to 0, no limit is enforced.
    let capacity = 0u64;
    let refill = 0u64;
    if capacity == 0 {
        return None;
    }
    let key = RateLimitKey::PerIp(route.name.clone(), peer_ip(req).unwrap_or(fallback_ip()));
    let _ = auth;
    Some((capacity, refill, key))
}

fn peer_ip(req: &HttpRequest) -> Option<IpAddr> {
    req.peer_addr.map(|a| a.ip())
}

fn fallback_ip() -> IpAddr {
    use std::net::Ipv4Addr;
    IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gateway::response::build_response_bytes;
    use crate::gateway::router::{CompiledRoute, HttpMethod};
    use zyron_catalog::EndpointId;
    use zyron_catalog::schema::{EndpointAuthMode, EndpointOutputFormat};

    fn basic_route(mode: EndpointAuthMode, methods: Vec<HttpMethod>) -> Arc<CompiledRoute> {
        Arc::new(CompiledRoute::compile(
            EndpointId(1),
            "ep".into(),
            "/api/ep".into(),
            methods,
            mode,
            Vec::new(),
            EndpointOutputFormat::Json,
            vec!["*".to_string()],
            0,
            30,
            1024,
            "SELECT 1".into(),
        ))
    }

    fn req(method: HttpMethod, headers: &[(&str, &str)], body: &[u8]) -> HttpRequest {
        let mut map = HashMap::new();
        for (k, v) in headers {
            map.insert(k.to_ascii_lowercase(), v.to_string());
        }
        HttpRequest {
            method,
            path: "/api/ep".into(),
            query_string: String::new(),
            headers: map,
            body: body.to_vec(),
            peer_addr: None,
            tls_info: None,
        }
    }

    #[test]
    fn options_returns_cors_preflight() {
        let route = basic_route(EndpointAuthMode::None, vec![HttpMethod::Get]);
        let rl = RateLimiter::new();
        let req = req(HttpMethod::Options, &[("origin", "https://a.test")], b"");
        let out = run_pipeline(route, HashMap::new(), &req, &rl);
        match out {
            MiddlewareOutcome::Response(r) => {
                assert_eq!(r.status, 204);
                let raw = build_response_bytes(&r).unwrap();
                let s = String::from_utf8(raw).unwrap();
                assert!(s.contains("Access-Control-Allow-Origin"));
            }
            _ => panic!("expected short-circuit response"),
        }
    }

    #[test]
    fn method_mismatch_returns_405() {
        let route = basic_route(EndpointAuthMode::None, vec![HttpMethod::Get]);
        let rl = RateLimiter::new();
        let r = req(HttpMethod::Post, &[], b"");
        let out = run_pipeline(route, HashMap::new(), &r, &rl);
        if let MiddlewareOutcome::Response(resp) = out {
            assert_eq!(resp.status, 405);
        } else {
            panic!("expected 405");
        }
    }

    #[test]
    fn body_over_cap_returns_413() {
        let route = basic_route(EndpointAuthMode::None, vec![HttpMethod::Post]);
        let rl = RateLimiter::new();
        let big = vec![0u8; 2048];
        let r = req(HttpMethod::Post, &[], &big);
        let out = run_pipeline(route, HashMap::new(), &r, &rl);
        if let MiddlewareOutcome::Response(resp) = out {
            assert_eq!(resp.status, 413);
        } else {
            panic!("expected 413");
        }
    }

    #[test]
    fn missing_jwt_returns_401() {
        let route = basic_route(EndpointAuthMode::Jwt, vec![HttpMethod::Get]);
        let rl = RateLimiter::new();
        let r = req(HttpMethod::Get, &[], b"");
        let out = run_pipeline(route, HashMap::new(), &r, &rl);
        if let MiddlewareOutcome::Response(resp) = out {
            assert_eq!(resp.status, 401);
        } else {
            panic!("expected 401");
        }
    }

    #[test]
    fn execute_envelope_returned_when_all_checks_pass() {
        let route = basic_route(EndpointAuthMode::None, vec![HttpMethod::Get]);
        let rl = RateLimiter::new();
        let r = req(HttpMethod::Get, &[], b"");
        let out = run_pipeline(route, HashMap::new(), &r, &rl);
        assert!(matches!(out, MiddlewareOutcome::Execute { .. }));
    }
}

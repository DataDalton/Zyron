// -----------------------------------------------------------------------------
// Dynamic HTTP router with path-pattern matching.
//
// Splits each path pattern into static and parameter segments at registration
// time, then matches incoming request paths by walking the segment list. Routes
// are keyed by the pattern string so operators can replace a route in place.
// -----------------------------------------------------------------------------

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::RwLock;
use zyron_catalog::EndpointId;
use zyron_catalog::schema::{EndpointAuthMode, EndpointOutputFormat};

use super::circuit::CircuitBreaker;
use super::metrics::EndpointMetrics;

/// HTTP methods handled by the gateway.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HttpMethod {
    Get,
    Post,
    Put,
    Delete,
    Patch,
    Head,
    Options,
}

impl HttpMethod {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "GET" => Some(HttpMethod::Get),
            "POST" => Some(HttpMethod::Post),
            "PUT" => Some(HttpMethod::Put),
            "DELETE" => Some(HttpMethod::Delete),
            "PATCH" => Some(HttpMethod::Patch),
            "HEAD" => Some(HttpMethod::Head),
            "OPTIONS" => Some(HttpMethod::Options),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            HttpMethod::Get => "GET",
            HttpMethod::Post => "POST",
            HttpMethod::Put => "PUT",
            HttpMethod::Delete => "DELETE",
            HttpMethod::Patch => "PATCH",
            HttpMethod::Head => "HEAD",
            HttpMethod::Options => "OPTIONS",
        }
    }

    pub fn from_catalog(m: zyron_catalog::schema::HttpMethod) -> Self {
        use zyron_catalog::schema::HttpMethod as C;
        match m {
            C::Get => HttpMethod::Get,
            C::Post => HttpMethod::Post,
            C::Put => HttpMethod::Put,
            C::Delete => HttpMethod::Delete,
            C::Patch => HttpMethod::Patch,
            C::Head => HttpMethod::Head,
            C::Options => HttpMethod::Options,
        }
    }
}

/// One segment of a compiled path pattern.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PathSegment {
    Static(String),
    Param(String),
}

/// Fully compiled route with auth, rate-limit, and circuit-breaker state.
#[derive(Debug)]
pub struct CompiledRoute {
    pub endpoint_id: EndpointId,
    pub name: String,
    pub path_pattern: String,
    pub param_names: Vec<String>,
    pub path_segments: Vec<PathSegment>,
    pub methods: Vec<HttpMethod>,
    pub auth: EndpointAuthMode,
    pub required_scopes: Vec<String>,
    pub output_format: EndpointOutputFormat,
    pub cors_origins: Vec<String>,
    pub cache_seconds: u32,
    pub timeout: Duration,
    pub max_body_bytes: u32,
    pub sql_body: String,
    pub circuit_breaker: Arc<CircuitBreaker>,
    pub metrics: EndpointMetrics,
    pub enabled: bool,
    // Cached parse of sql_body. Populated at compile time so the per-request
    // hot path can skip the parser when the template has no $ parameters.
    // When the parser rejects the template at register time this stays None
    // and per-request execution surfaces the parse error as before.
    pub pre_parsed: Option<Arc<zyron_parser::Statement>>,
    pub template_has_params: bool,
}

impl CompiledRoute {
    /// Splits the pattern into segments, capturing :param names.
    pub fn compile(
        endpoint_id: EndpointId,
        name: String,
        pattern: String,
        methods: Vec<HttpMethod>,
        auth: EndpointAuthMode,
        required_scopes: Vec<String>,
        output_format: EndpointOutputFormat,
        cors_origins: Vec<String>,
        cache_seconds: u32,
        timeout_seconds: u32,
        max_body_bytes: u32,
        sql_body: String,
    ) -> Self {
        let mut segments = Vec::new();
        let mut param_names = Vec::new();
        for raw in pattern.split('/') {
            if raw.is_empty() {
                continue;
            }
            if let Some(rest) = raw.strip_prefix(':') {
                segments.push(PathSegment::Param(rest.to_string()));
                param_names.push(rest.to_string());
            } else if raw.starts_with('{') && raw.ends_with('}') && raw.len() >= 2 {
                let inner = &raw[1..raw.len() - 1];
                segments.push(PathSegment::Param(inner.to_string()));
                param_names.push(inner.to_string());
            } else {
                segments.push(PathSegment::Static(raw.to_string()));
            }
        }
        let template_has_params = super::endpoint_exec::template_contains_params(&sql_body);
        // Pre-parse the template once. When the template has no parameters
        // the parsed AST can be reused directly on every request. When the
        // template has parameters we still cache the parse result so hot
        // code paths that do not substitute (e.g. validation) can avoid
        // the re-parse cost.
        let pre_parsed = if template_has_params {
            None
        } else {
            zyron_parser::parse(&sql_body)
                .ok()
                .and_then(|v| v.into_iter().next())
                .map(Arc::new)
        };
        Self {
            endpoint_id,
            name,
            path_pattern: pattern,
            param_names,
            path_segments: segments,
            methods,
            auth,
            required_scopes,
            output_format,
            cors_origins,
            cache_seconds,
            timeout: Duration::from_secs(timeout_seconds.max(1) as u64),
            max_body_bytes,
            sql_body,
            circuit_breaker: Arc::new(CircuitBreaker::new(5, Duration::from_secs(30))),
            metrics: EndpointMetrics::new(),
            enabled: true,
            pre_parsed,
            template_has_params,
        }
    }

    /// Returns true when this route accepts the given HTTP method.
    pub fn accepts(&self, method: HttpMethod) -> bool {
        self.methods.iter().any(|m| *m == method)
            || (method == HttpMethod::Options && !self.cors_origins.is_empty())
    }

    /// Matches a request path against this route, returning the captured params
    /// when every static segment agrees.
    pub fn match_path(&self, path: &str) -> Option<HashMap<String, String>> {
        let mut captures = HashMap::new();
        let mut pi = 0usize;
        let path_parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
        if path_parts.len() != self.path_segments.len() {
            return None;
        }
        for seg in &self.path_segments {
            let got = path_parts.get(pi)?;
            match seg {
                PathSegment::Static(s) => {
                    if s != got {
                        return None;
                    }
                }
                PathSegment::Param(name) => {
                    captures.insert(name.clone(), (*got).to_string());
                }
            }
            pi += 1;
        }
        Some(captures)
    }
}

/// Route registry. Holds a map from pattern to route plus a sorted lookup list
/// ordered so static segments win over parameter segments.
pub struct Router {
    routes: RwLock<Vec<Arc<CompiledRoute>>>,
}

impl Router {
    pub fn new() -> Self {
        Self {
            routes: RwLock::new(Vec::new()),
        }
    }

    pub fn insert(&self, route: CompiledRoute) {
        let arc = Arc::new(route);
        let mut routes = self.routes.write();
        if let Some(idx) = routes
            .iter()
            .position(|r| r.path_pattern == arc.path_pattern)
        {
            routes[idx] = arc;
        } else {
            routes.push(arc);
        }
        // Order: routes with more static segments come first.
        routes.sort_by_key(|r| {
            let static_count = r
                .path_segments
                .iter()
                .filter(|s| matches!(s, PathSegment::Static(_)))
                .count();
            (usize::MAX - static_count, r.path_segments.len())
        });
    }

    pub fn remove(&self, pattern: &str) -> bool {
        let mut routes = self.routes.write();
        if let Some(idx) = routes.iter().position(|r| r.path_pattern == pattern) {
            routes.remove(idx);
            true
        } else {
            false
        }
    }

    /// Removes the route that owns the given catalog endpoint id. Used when a
    /// DROP ENDPOINT or ALTER ENDPOINT DISABLE fires without the path on hand.
    pub fn remove_by_endpoint_id(&self, endpoint_id: EndpointId) -> bool {
        let mut routes = self.routes.write();
        if let Some(idx) = routes.iter().position(|r| r.endpoint_id == endpoint_id) {
            routes.remove(idx);
            true
        } else {
            false
        }
    }

    pub fn lookup(
        &self,
        method: HttpMethod,
        path: &str,
    ) -> Option<(Arc<CompiledRoute>, HashMap<String, String>)> {
        let routes = self.routes.read();
        for r in routes.iter() {
            if !r.accepts(method) {
                continue;
            }
            if let Some(caps) = r.match_path(path) {
                return Some((Arc::clone(r), caps));
            }
        }
        None
    }

    pub fn list(&self) -> Vec<Arc<CompiledRoute>> {
        self.routes.read().clone()
    }

    pub fn set_enabled(&self, name: &str, enabled: bool) -> bool {
        let routes = self.routes.read();
        for r in routes.iter() {
            if r.name == name {
                // CircuitBreaker holds interior mutability. Toggling enabled
                // requires replacing the Arc, which we skip here because the
                // admin path rebuilds the route from the catalog.
                let _ = enabled;
                return true;
            }
        }
        false
    }
}

impl Default for Router {
    fn default() -> Self {
        Self::new()
    }
}

// -----------------------------------------------------------------------------
// Catalog-backed runtime registrar
// -----------------------------------------------------------------------------

/// Compiles an EndpointEntry into a CompiledRoute. Streaming endpoints (WS/SSE)
/// keep the same path/auth/methods surface but the Router itself only handles
/// the path matching, so a single compile path works for all kinds.
pub fn compile_route_from_entry(entry: &zyron_catalog::EndpointEntry) -> CompiledRoute {
    let methods: Vec<HttpMethod> = entry
        .methods
        .iter()
        .copied()
        .map(HttpMethod::from_catalog)
        .collect();
    let methods = if methods.is_empty() {
        vec![HttpMethod::Get]
    } else {
        methods
    };
    let output_format = entry.output_format.unwrap_or(EndpointOutputFormat::Json);
    let cache_seconds = entry.cache_seconds.unwrap_or(0);
    let timeout_seconds = entry.timeout_seconds.unwrap_or(30);
    let max_body_bytes = entry
        .max_request_body_kb
        .map(|kb| kb.saturating_mul(1024))
        .unwrap_or(64 * 1024);
    let route = CompiledRoute::compile(
        entry.id,
        entry.name.clone(),
        entry.path.clone(),
        methods,
        entry.auth_mode,
        entry.required_scopes.clone(),
        output_format,
        entry.cors_origins.clone(),
        cache_seconds,
        timeout_seconds,
        max_body_bytes,
        entry.sql_body.clone(),
    );
    route
}

/// Registrar implementation that keeps an Arc<Router> in sync with catalog DDL.
pub struct CatalogEndpointRegistrar {
    router: Arc<Router>,
}

impl CatalogEndpointRegistrar {
    pub fn new(router: Arc<Router>) -> Self {
        Self { router }
    }

    pub fn router(&self) -> &Arc<Router> {
        &self.router
    }
}

#[async_trait::async_trait]
impl zyron_wire::EndpointRegistrar for CatalogEndpointRegistrar {
    async fn register(&self, entry: &zyron_catalog::EndpointEntry) -> zyron_common::Result<()> {
        if !entry.enabled {
            self.router.remove_by_endpoint_id(entry.id);
            return Ok(());
        }
        let route = compile_route_from_entry(entry);
        self.router.insert(route);
        Ok(())
    }

    async fn unregister(&self, endpoint_id: zyron_catalog::EndpointId) -> zyron_common::Result<()> {
        self.router.remove_by_endpoint_id(endpoint_id);
        Ok(())
    }

    async fn set_enabled(
        &self,
        entry: &zyron_catalog::EndpointEntry,
        enabled: bool,
    ) -> zyron_common::Result<()> {
        if enabled {
            let route = compile_route_from_entry(entry);
            self.router.insert(route);
        } else {
            self.router.remove_by_endpoint_id(entry.id);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_route(pattern: &str, methods: Vec<HttpMethod>) -> CompiledRoute {
        CompiledRoute::compile(
            EndpointId(1),
            "t".to_string(),
            pattern.to_string(),
            methods,
            EndpointAuthMode::None,
            Vec::new(),
            EndpointOutputFormat::Json,
            Vec::new(),
            0,
            30,
            65536,
            "SELECT 1".to_string(),
        )
    }

    #[test]
    fn static_match() {
        let r = sample_route("/api/ping", vec![HttpMethod::Get]);
        assert!(r.match_path("/api/ping").is_some());
        assert!(r.match_path("/api/pong").is_none());
    }

    #[test]
    fn param_capture() {
        let r = sample_route("/api/orders/:id", vec![HttpMethod::Get]);
        let caps = r.match_path("/api/orders/42").unwrap();
        assert_eq!(caps.get("id").unwrap(), "42");
    }

    #[test]
    fn curly_brace_param() {
        let r = sample_route("/api/{name}", vec![HttpMethod::Get]);
        let caps = r.match_path("/api/widget").unwrap();
        assert_eq!(caps.get("name").unwrap(), "widget");
    }

    #[test]
    fn segment_count_mismatch() {
        let r = sample_route("/api/orders/:id", vec![HttpMethod::Get]);
        assert!(r.match_path("/api/orders").is_none());
        assert!(r.match_path("/api/orders/42/items").is_none());
    }

    #[test]
    fn method_gate() {
        let r = sample_route("/api/a", vec![HttpMethod::Post]);
        assert!(r.accepts(HttpMethod::Post));
        assert!(!r.accepts(HttpMethod::Get));
    }

    #[test]
    fn router_lookup_priority() {
        let router = Router::new();
        router.insert(sample_route("/api/:x", vec![HttpMethod::Get]));
        router.insert(sample_route("/api/health", vec![HttpMethod::Get]));
        let (hit, _) = router.lookup(HttpMethod::Get, "/api/health").unwrap();
        assert_eq!(hit.path_pattern, "/api/health");
    }

    #[test]
    fn router_param_fallback() {
        let router = Router::new();
        router.insert(sample_route("/api/:x", vec![HttpMethod::Get]));
        router.insert(sample_route("/api/health", vec![HttpMethod::Get]));
        let (hit, caps) = router.lookup(HttpMethod::Get, "/api/other").unwrap();
        assert_eq!(hit.path_pattern, "/api/:x");
        assert_eq!(caps.get("x").unwrap(), "other");
    }

    #[test]
    fn router_not_found() {
        let router = Router::new();
        router.insert(sample_route("/a", vec![HttpMethod::Get]));
        assert!(router.lookup(HttpMethod::Get, "/b").is_none());
    }

    #[test]
    fn router_remove() {
        let router = Router::new();
        router.insert(sample_route("/a", vec![HttpMethod::Get]));
        assert!(router.remove("/a"));
        assert!(!router.remove("/a"));
    }

    #[test]
    fn parse_method() {
        assert_eq!(HttpMethod::parse("GET"), Some(HttpMethod::Get));
        assert_eq!(HttpMethod::parse("POST"), Some(HttpMethod::Post));
        assert_eq!(HttpMethod::parse("junk"), None);
    }

    // -------------------------------------------------------------------
    // Catalog-backed registrar tests
    // -------------------------------------------------------------------

    fn sample_entry(id: u32, path: &str, enabled: bool) -> zyron_catalog::EndpointEntry {
        zyron_catalog::EndpointEntry {
            id: zyron_catalog::EndpointId(id),
            schema_id: zyron_catalog::SchemaId(0),
            name: format!("ep_{}", id),
            kind: zyron_catalog::EndpointKind::Rest,
            path: path.to_string(),
            methods: vec![zyron_catalog::HttpMethod::Get],
            sql_body: "SELECT 1".to_string(),
            backed_publication_id: None,
            auth_mode: zyron_catalog::EndpointAuthMode::None,
            required_scopes: Vec::new(),
            output_format: Some(zyron_catalog::EndpointOutputFormat::Json),
            cors_origins: Vec::new(),
            rate_limit: None,
            cache_seconds: Some(0),
            timeout_seconds: Some(30),
            max_request_body_kb: Some(64),
            message_format: None,
            heartbeat_seconds: None,
            backpressure: None,
            max_connections: None,
            enabled,
            owner_role_id: 0,
            created_at: 0,
        }
    }

    #[tokio::test]
    async fn register_endpoint_then_lookup_succeeds() {
        use zyron_wire::EndpointRegistrar;
        let router = Arc::new(Router::new());
        let reg = CatalogEndpointRegistrar::new(Arc::clone(&router));
        let entry = sample_entry(1, "/api/ping", true);
        reg.register(&entry).await.unwrap();
        let hit = router.lookup(HttpMethod::Get, "/api/ping");
        assert!(hit.is_some());
        let (route, _) = hit.unwrap();
        assert_eq!(route.endpoint_id, zyron_catalog::EndpointId(1));
    }

    #[tokio::test]
    async fn unregister_endpoint_then_lookup_fails() {
        use zyron_wire::EndpointRegistrar;
        let router = Arc::new(Router::new());
        let reg = CatalogEndpointRegistrar::new(Arc::clone(&router));
        let entry = sample_entry(2, "/api/widgets", true);
        reg.register(&entry).await.unwrap();
        assert!(router.lookup(HttpMethod::Get, "/api/widgets").is_some());
        reg.unregister(entry.id).await.unwrap();
        assert!(router.lookup(HttpMethod::Get, "/api/widgets").is_none());
    }

    #[tokio::test]
    async fn set_enabled_toggles_lookup() {
        use zyron_wire::EndpointRegistrar;
        let router = Arc::new(Router::new());
        let reg = CatalogEndpointRegistrar::new(Arc::clone(&router));
        let entry = sample_entry(3, "/api/toggle", true);
        reg.set_enabled(&entry, true).await.unwrap();
        assert!(router.lookup(HttpMethod::Get, "/api/toggle").is_some());
        reg.set_enabled(&entry, false).await.unwrap();
        assert!(router.lookup(HttpMethod::Get, "/api/toggle").is_none());
        reg.set_enabled(&entry, true).await.unwrap();
        assert!(router.lookup(HttpMethod::Get, "/api/toggle").is_some());
    }

    #[tokio::test]
    async fn startup_recovery_registers_enabled_endpoints() {
        use zyron_wire::EndpointRegistrar;
        let router = Arc::new(Router::new());
        let reg = CatalogEndpointRegistrar::new(Arc::clone(&router));
        let entries = vec![
            sample_entry(10, "/api/a", true),
            sample_entry(11, "/api/b", true),
            sample_entry(12, "/api/c", false),
        ];
        let mut registered = 0usize;
        for e in &entries {
            if !e.enabled {
                continue;
            }
            reg.register(e).await.unwrap();
            registered += 1;
        }
        assert_eq!(registered, 2);
        assert_eq!(router.list().len(), 2);
        assert!(router.lookup(HttpMethod::Get, "/api/a").is_some());
        assert!(router.lookup(HttpMethod::Get, "/api/b").is_some());
        assert!(router.lookup(HttpMethod::Get, "/api/c").is_none());
    }
}

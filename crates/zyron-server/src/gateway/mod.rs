// -----------------------------------------------------------------------------
// HTTP gateway module.
//
// Provides dynamic HTTP, WebSocket, and Server-Sent Events routing layered on
// top of the hand-rolled HTTP/1.1 listener in health.rs. Includes per-endpoint
// middleware (auth, rate limiting, CORS, circuit breaker), OpenAPI emission,
// and an admin API surface.
// -----------------------------------------------------------------------------

pub mod admin;
pub mod auth_mw;
pub mod circuit;
pub mod endpoint_exec;
pub mod metrics;
pub mod middleware;
pub mod openapi;
pub mod rate_limit;
pub mod request;
pub mod response;
pub mod router;
pub mod sse;
pub mod streaming_endpoint;
pub mod websocket;

pub use admin::{
    AdminAction, AdminAuthResult, AdminExecutor, AdminResponse, AdminRouter,
    PublicationStatusRegistry, ResetLsnTarget, check_admin_auth,
};
pub use circuit::{CircuitBreaker, CircuitState};
pub use metrics::{EndpointMetrics, GatewayMetrics};
pub use middleware::{MiddlewareOutcome, run_pipeline};
pub use openapi::{emit_openapi_json, emit_swagger_html};
pub use rate_limit::{RateLimitKey, RateLimiter};
pub use request::{HttpRequest, TlsInfo, parse_request};
pub use response::{HttpResponse, ResponseBody, build_response_bytes};
pub use router::{CatalogEndpointRegistrar, CompiledRoute, HttpMethod, PathSegment, Router};
pub use sse::SseStream;
pub use websocket::{WebSocketConnection, WsMessage, WsOpcode};

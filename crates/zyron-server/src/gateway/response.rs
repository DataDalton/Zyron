// -----------------------------------------------------------------------------
// HTTP response builder and format dispatcher.
//
// ResponseBody covers a fully-buffered byte slice or a streaming iterator.
// build_response_bytes emits a complete HTTP/1.1 response for the buffered
// case. Streaming responses are handed to the caller who writes chunks
// directly on the socket.
// -----------------------------------------------------------------------------

use std::fmt::Write;

use super::router::CompiledRoute;

pub type ChunkIter = Box<dyn Iterator<Item = Vec<u8>> + Send>;

/// Response body kinds handled by the gateway.
pub enum ResponseBody {
    Bytes(Vec<u8>),
    Chunks(ChunkIter),
    Empty,
}

impl std::fmt::Debug for ResponseBody {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResponseBody::Bytes(b) => write!(f, "Bytes({} bytes)", b.len()),
            ResponseBody::Chunks(_) => write!(f, "Chunks(..)"),
            ResponseBody::Empty => write!(f, "Empty"),
        }
    }
}

/// Complete HTTP response.
#[derive(Debug)]
pub struct HttpResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: ResponseBody,
}

impl HttpResponse {
    pub fn new(status: u16) -> Self {
        Self {
            status,
            headers: Vec::new(),
            body: ResponseBody::Empty,
        }
    }

    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    pub fn body_bytes(mut self, bytes: Vec<u8>) -> Self {
        self.body = ResponseBody::Bytes(bytes);
        self
    }

    pub fn json(mut self, json: String) -> Self {
        self.headers
            .push(("Content-Type".to_string(), "application/json".to_string()));
        self.body = ResponseBody::Bytes(json.into_bytes());
        self
    }

    pub fn plain(mut self, text: String) -> Self {
        self.headers.push((
            "Content-Type".to_string(),
            "text/plain; charset=utf-8".to_string(),
        ));
        self.body = ResponseBody::Bytes(text.into_bytes());
        self
    }

    pub fn apply_cors(mut self, origin: &str, route: &CompiledRoute) -> Self {
        if route.cors_origins.is_empty() {
            return self;
        }
        let allow = if route.cors_origins.iter().any(|o| o == "*") {
            "*".to_string()
        } else if route.cors_origins.iter().any(|o| o == origin) {
            origin.to_string()
        } else {
            return self;
        };
        self.headers
            .push(("Access-Control-Allow-Origin".to_string(), allow));
        self.headers.push((
            "Access-Control-Allow-Methods".to_string(),
            route
                .methods
                .iter()
                .map(|m| m.as_str())
                .collect::<Vec<_>>()
                .join(", "),
        ));
        self.headers.push((
            "Access-Control-Allow-Headers".to_string(),
            "Authorization, Content-Type, X-API-Key".to_string(),
        ));
        self
    }
}

/// Serializes an HttpResponse with buffered body into HTTP/1.1 wire bytes.
/// Streaming bodies return None so the caller handles them directly.
pub fn build_response_bytes(resp: &HttpResponse) -> Option<Vec<u8>> {
    let body_bytes: &[u8] = match &resp.body {
        ResponseBody::Bytes(b) => b.as_slice(),
        ResponseBody::Empty => &[],
        ResponseBody::Chunks(_) => return None,
    };
    let mut out = String::with_capacity(256 + body_bytes.len());
    let _ = write!(
        out,
        "HTTP/1.1 {} {}\r\n",
        resp.status,
        status_reason(resp.status)
    );
    let mut has_len = false;
    let mut has_type = false;
    for (k, v) in &resp.headers {
        if k.eq_ignore_ascii_case("content-length") {
            has_len = true;
        }
        if k.eq_ignore_ascii_case("content-type") {
            has_type = true;
        }
        let _ = write!(out, "{}: {}\r\n", k, v);
    }
    if !has_len {
        let _ = write!(out, "Content-Length: {}\r\n", body_bytes.len());
    }
    if !has_type && !body_bytes.is_empty() {
        let _ = write!(out, "Content-Type: application/octet-stream\r\n");
    }
    let _ = write!(out, "Connection: close\r\n\r\n");
    let mut bytes = out.into_bytes();
    bytes.extend_from_slice(body_bytes);
    Some(bytes)
}

/// Returns the canonical HTTP reason phrase for a status code.
pub fn status_reason(code: u16) -> &'static str {
    match code {
        100 => "Continue",
        101 => "Switching Protocols",
        200 => "OK",
        201 => "Created",
        204 => "No Content",
        301 => "Moved Permanently",
        302 => "Found",
        304 => "Not Modified",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        408 => "Request Timeout",
        413 => "Payload Too Large",
        415 => "Unsupported Media Type",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        _ => "OK",
    }
}

/// Canonical error responses reused by the middleware chain.
pub fn unauthorized(msg: &str) -> HttpResponse {
    HttpResponse::new(401).json(format!(r#"{{"error":"unauthorized","detail":{:?}}}"#, msg))
}

pub fn forbidden(msg: &str) -> HttpResponse {
    HttpResponse::new(403).json(format!(r#"{{"error":"forbidden","detail":{:?}}}"#, msg))
}

pub fn not_found() -> HttpResponse {
    HttpResponse::new(404).json(r#"{"error":"not_found"}"#.to_string())
}

pub fn method_not_allowed() -> HttpResponse {
    HttpResponse::new(405).json(r#"{"error":"method_not_allowed"}"#.to_string())
}

pub fn too_many_requests(retry_after: u64) -> HttpResponse {
    HttpResponse::new(429)
        .header("Retry-After", retry_after.to_string())
        .json(r#"{"error":"rate_limited"}"#.to_string())
}

pub fn service_unavailable(reason: &str) -> HttpResponse {
    HttpResponse::new(503).json(format!(
        r#"{{"error":"unavailable","detail":{:?}}}"#,
        reason
    ))
}

pub fn payload_too_large() -> HttpResponse {
    HttpResponse::new(413).json(r#"{"error":"payload_too_large"}"#.to_string())
}

pub fn internal_error(msg: &str) -> HttpResponse {
    HttpResponse::new(500).json(format!(r#"{{"error":"internal","detail":{:?}}}"#, msg))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_simple_json() {
        let resp = HttpResponse::new(200).json(r#"{"ok":true}"#.to_string());
        let raw = build_response_bytes(&resp).unwrap();
        let s = String::from_utf8(raw).unwrap();
        assert!(s.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(s.contains("Content-Length: 11\r\n"));
        assert!(s.contains("Content-Type: application/json\r\n"));
        assert!(s.ends_with(r#"{"ok":true}"#));
    }

    #[test]
    fn status_reasons() {
        assert_eq!(status_reason(404), "Not Found");
        assert_eq!(status_reason(429), "Too Many Requests");
        assert_eq!(status_reason(101), "Switching Protocols");
    }

    #[test]
    fn error_shortcuts() {
        let r = unauthorized("bad token");
        assert_eq!(r.status, 401);
        let r = too_many_requests(30);
        assert!(r.headers.iter().any(|(k, _)| k == "Retry-After"));
    }
}

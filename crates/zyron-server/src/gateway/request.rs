// -----------------------------------------------------------------------------
// HTTP/1.1 request parsing.
//
// Reads the request line, headers, and body directly from a byte buffer. No
// pre-built HTTP parser is used. Supports Content-Length bodies up to the
// per-endpoint max_body_bytes cap.
// -----------------------------------------------------------------------------

use std::collections::HashMap;
use std::net::SocketAddr;

use super::router::HttpMethod;

/// TLS peer information recorded when the connection was accepted.
#[derive(Debug, Clone)]
pub struct TlsInfo {
    pub peer_cert_der: Option<Vec<u8>>,
    pub peer_cert_fingerprint: Option<[u8; 32]>,
    pub negotiated_cipher: String,
}

/// Parsed HTTP/1.1 request. Bodies above max_body_bytes are rejected by the
/// caller before this struct is built.
#[derive(Debug, Clone)]
pub struct HttpRequest {
    pub method: HttpMethod,
    pub path: String,
    pub query_string: String,
    pub headers: HashMap<String, String>,
    pub body: Vec<u8>,
    pub peer_addr: Option<SocketAddr>,
    pub tls_info: Option<TlsInfo>,
}

impl HttpRequest {
    /// Returns the value of a header looked up case-insensitively.
    pub fn header(&self, name: &str) -> Option<&str> {
        let lower = name.to_ascii_lowercase();
        self.headers.get(&lower).map(|s| s.as_str())
    }

    /// Parses the query string into decoded (key, value) pairs.
    pub fn query_pairs(&self) -> Vec<(String, String)> {
        parse_form_urlencoded(&self.query_string)
    }

    /// Parses the body as application/x-www-form-urlencoded key=value pairs.
    pub fn body_form_pairs(&self) -> Vec<(String, String)> {
        if let Ok(body) = std::str::from_utf8(&self.body) {
            parse_form_urlencoded(body)
        } else {
            Vec::new()
        }
    }
}

/// Parses the raw request bytes. Returns None when the buffer is malformed or
/// incomplete.
pub fn parse_request(buf: &[u8], peer_addr: Option<SocketAddr>) -> Option<HttpRequest> {
    let header_end = find_double_crlf(buf)?;
    let head = std::str::from_utf8(&buf[..header_end]).ok()?;
    let mut lines = head.split("\r\n");
    let start = lines.next()?;
    let mut parts = start.split_whitespace();
    let method = HttpMethod::parse(parts.next()?)?;
    let full_target = parts.next()?;
    let (path, query) = match full_target.split_once('?') {
        Some((p, q)) => (p.to_string(), q.to_string()),
        None => (full_target.to_string(), String::new()),
    };
    let mut headers = HashMap::new();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        if let Some((k, v)) = line.split_once(':') {
            headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
        }
    }
    let content_length = headers
        .get("content-length")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);
    let body_start = header_end + 4;
    let body_end = body_start.checked_add(content_length)?;
    let body = if body_end <= buf.len() {
        buf[body_start..body_end].to_vec()
    } else {
        // Caller passes the entire buffered request. Treat truncated bodies as
        // a zero-length body rather than silently corrupting the read.
        Vec::new()
    };
    Some(HttpRequest {
        method,
        path,
        query_string: query,
        headers,
        body,
        peer_addr,
        tls_info: None,
    })
}

/// Searches for the end-of-headers marker \r\n\r\n.
pub fn find_double_crlf(buf: &[u8]) -> Option<usize> {
    let needle = b"\r\n\r\n";
    buf.windows(4).position(|w| w == needle)
}

/// Decodes form-urlencoded input into key-value string pairs.
pub fn parse_form_urlencoded(s: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for pair in s.split('&') {
        if pair.is_empty() {
            continue;
        }
        let (k, v) = match pair.split_once('=') {
            Some((k, v)) => (k, v),
            None => (pair, ""),
        };
        out.push((url_decode(k), url_decode(v)));
    }
    out
}

/// Minimal percent-decoder. Invalid escapes pass through as-is.
pub fn url_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hi = hex_nibble(bytes[i + 1]);
            let lo = hex_nibble(bytes[i + 2]);
            match (hi, lo) {
                (Some(h), Some(l)) => {
                    out.push((h << 4) | l);
                    i += 3;
                    continue;
                }
                _ => {}
            }
        }
        if bytes[i] == b'+' {
            out.push(b' ');
        } else {
            out.push(bytes[i]);
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(10 + b - b'a'),
        b'A'..=b'F' => Some(10 + b - b'A'),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_basic_get() {
        let raw = b"GET /api/ping HTTP/1.1\r\nHost: x\r\n\r\n";
        let req = parse_request(raw, None).unwrap();
        assert_eq!(req.method, HttpMethod::Get);
        assert_eq!(req.path, "/api/ping");
        assert_eq!(req.header("host"), Some("x"));
    }

    #[test]
    fn parse_with_query() {
        let raw = b"GET /search?q=rust&limit=5 HTTP/1.1\r\n\r\n";
        let req = parse_request(raw, None).unwrap();
        assert_eq!(req.path, "/search");
        assert_eq!(req.query_string, "q=rust&limit=5");
        let pairs = req.query_pairs();
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0], ("q".to_string(), "rust".to_string()));
    }

    #[test]
    fn parse_post_body() {
        let raw = b"POST /x HTTP/1.1\r\nContent-Length: 5\r\n\r\nhello";
        let req = parse_request(raw, None).unwrap();
        assert_eq!(req.body, b"hello");
    }

    #[test]
    fn form_urlencoded_roundtrip() {
        let pairs = parse_form_urlencoded("a=1&b=hello%20world&c=%2B");
        assert_eq!(pairs[0], ("a".to_string(), "1".to_string()));
        assert_eq!(pairs[1], ("b".to_string(), "hello world".to_string()));
        assert_eq!(pairs[2], ("c".to_string(), "+".to_string()));
    }

    #[test]
    fn url_decode_plus() {
        assert_eq!(url_decode("a+b"), "a b");
        assert_eq!(url_decode("a%2Bb"), "a+b");
    }

    #[test]
    fn bad_request_returns_none() {
        assert!(parse_request(b"not-http\r\n\r\n", None).is_none());
    }
}

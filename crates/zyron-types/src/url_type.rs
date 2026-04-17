//! URL parsing and operations.
//!
//! Parses URLs into components (scheme, host, port, path, query, fragment),
//! supports query parameter extraction, normalization, and relative URL resolution.

use zyron_common::{Result, ZyronError};

/// Parsed URL components.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UrlParts {
    pub scheme: String,
    pub user: Option<String>,
    pub password: Option<String>,
    pub host: String,
    pub port: Option<u16>,
    pub path: String,
    pub query: Option<String>,
    pub fragment: Option<String>,
}

/// Parses a URL into its components.
pub fn url_parse(text: &str) -> Result<UrlParts> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(ZyronError::ExecutionError("Empty URL".into()));
    }

    // Extract scheme
    let colon_pos = trimmed
        .find(':')
        .ok_or_else(|| ZyronError::ExecutionError("URL missing scheme".into()))?;
    let scheme = trimmed[..colon_pos].to_lowercase();

    // Validate scheme
    if !scheme
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '.' || c == '-')
    {
        return Err(ZyronError::ExecutionError(format!(
            "Invalid URL scheme: {}",
            scheme
        )));
    }
    if scheme.is_empty() || !scheme.chars().next().unwrap().is_ascii_alphabetic() {
        return Err(ZyronError::ExecutionError(
            "Scheme must start with a letter".into(),
        ));
    }

    let after_scheme = &trimmed[colon_pos + 1..];

    // Check for "//" authority indicator
    let (authority_and_rest, has_authority) = if let Some(rest) = after_scheme.strip_prefix("//") {
        (rest, true)
    } else {
        (after_scheme, false)
    };

    // Split off fragment first (# comes after query)
    let (before_fragment, fragment) = match authority_and_rest.find('#') {
        Some(idx) => (
            &authority_and_rest[..idx],
            Some(authority_and_rest[idx + 1..].to_string()),
        ),
        None => (authority_and_rest, None),
    };

    // Split off query (? comes before fragment)
    let (before_query, query) = match before_fragment.find('?') {
        Some(idx) => (
            &before_fragment[..idx],
            Some(before_fragment[idx + 1..].to_string()),
        ),
        None => (before_fragment, None),
    };

    if !has_authority {
        // Simple scheme:path URL (e.g., "mailto:user@example.com")
        return Ok(UrlParts {
            scheme,
            user: None,
            password: None,
            host: String::new(),
            port: None,
            path: before_query.to_string(),
            query,
            fragment,
        });
    }

    // Split authority from path
    let (authority, path) = match before_query.find('/') {
        Some(idx) => (&before_query[..idx], before_query[idx..].to_string()),
        None => (before_query, String::new()),
    };

    // Parse userinfo
    let (user, password, host_port) = if let Some(at_idx) = authority.rfind('@') {
        let userinfo = &authority[..at_idx];
        let host_part = &authority[at_idx + 1..];
        let (u, p) = match userinfo.find(':') {
            Some(idx) => (
                Some(userinfo[..idx].to_string()),
                Some(userinfo[idx + 1..].to_string()),
            ),
            None => (Some(userinfo.to_string()), None),
        };
        (u, p, host_part)
    } else {
        (None, None, authority)
    };

    // Parse host and port
    let (host, port) = parse_host_port(host_port)?;

    Ok(UrlParts {
        scheme,
        user,
        password,
        host,
        port,
        path,
        query,
        fragment,
    })
}

fn parse_host_port(text: &str) -> Result<(String, Option<u16>)> {
    // IPv6 in brackets: [::1]:8080
    if let Some(rest) = text.strip_prefix('[') {
        if let Some(close) = rest.find(']') {
            let ipv6 = rest[..close].to_string();
            let after = &rest[close + 1..];
            if let Some(port_str) = after.strip_prefix(':') {
                let port = port_str
                    .parse::<u16>()
                    .map_err(|e| ZyronError::ExecutionError(format!("Invalid port: {}", e)))?;
                return Ok((ipv6, Some(port)));
            }
            return Ok((ipv6, None));
        }
        return Err(ZyronError::ExecutionError("Unclosed IPv6 bracket".into()));
    }

    // Regular host:port
    match text.rfind(':') {
        Some(idx) => {
            let host = text[..idx].to_lowercase();
            let port_str = &text[idx + 1..];
            if port_str.is_empty() {
                Ok((host, None))
            } else {
                let port = port_str
                    .parse::<u16>()
                    .map_err(|e| ZyronError::ExecutionError(format!("Invalid port: {}", e)))?;
                Ok((host, Some(port)))
            }
        }
        None => Ok((text.to_lowercase(), None)),
    }
}

/// Extracts the scheme from a URL.
pub fn url_scheme(text: &str) -> Result<String> {
    url_parse(text).map(|u| u.scheme)
}

/// Extracts the host from a URL.
pub fn url_host(text: &str) -> Result<String> {
    url_parse(text).map(|u| u.host)
}

/// Extracts the port from a URL. Returns None if no port is specified.
pub fn url_port(text: &str) -> Result<Option<u16>> {
    url_parse(text).map(|u| u.port)
}

/// Extracts the path from a URL.
pub fn url_path(text: &str) -> Result<String> {
    url_parse(text).map(|u| u.path)
}

/// Extracts a specific query parameter value from a URL.
pub fn url_query_param(text: &str, key: &str) -> Result<Option<String>> {
    let parts = url_parse(text)?;
    let query = match parts.query {
        Some(q) => q,
        None => return Ok(None),
    };

    for pair in query.split('&') {
        if let Some(eq_idx) = pair.find('=') {
            let k = &pair[..eq_idx];
            if url_decode(k) == key {
                return Ok(Some(url_decode(&pair[eq_idx + 1..])));
            }
        } else if url_decode(pair) == key {
            return Ok(Some(String::new()));
        }
    }
    Ok(None)
}

/// Extracts all query parameters as a list of (key, value) pairs.
pub fn url_query_params(text: &str) -> Result<Vec<(String, String)>> {
    let parts = url_parse(text)?;
    let query = match parts.query {
        Some(q) => q,
        None => return Ok(Vec::new()),
    };

    let mut result = Vec::new();
    for pair in query.split('&') {
        if pair.is_empty() {
            continue;
        }
        if let Some(eq_idx) = pair.find('=') {
            result.push((url_decode(&pair[..eq_idx]), url_decode(&pair[eq_idx + 1..])));
        } else {
            result.push((url_decode(pair), String::new()));
        }
    }
    Ok(result)
}

/// Extracts the fragment from a URL.
pub fn url_fragment(text: &str) -> Result<Option<String>> {
    url_parse(text).map(|u| u.fragment)
}

/// Normalizes a URL: lowercase scheme and host, remove default ports,
/// normalize path (remove dot segments), sort query parameters.
pub fn url_normalize(text: &str) -> Result<String> {
    let parts = url_parse(text)?;

    let mut result = String::new();
    result.push_str(&parts.scheme);
    result.push(':');

    let has_authority = !parts.host.is_empty() || parts.user.is_some();
    if has_authority {
        result.push_str("//");

        if let Some(user) = &parts.user {
            result.push_str(user);
            if let Some(password) = &parts.password {
                result.push(':');
                result.push_str(password);
            }
            result.push('@');
        }

        // Host (lowercased already from parse)
        if parts.host.contains(':') && !parts.host.starts_with('[') {
            // IPv6 without brackets - add them
            result.push('[');
            result.push_str(&parts.host);
            result.push(']');
        } else {
            result.push_str(&parts.host);
        }

        // Remove default ports
        if let Some(port) = parts.port {
            let is_default = matches!(
                (parts.scheme.as_str(), port),
                ("http", 80) | ("https", 443) | ("ftp", 21) | ("ssh", 22)
            );
            if !is_default {
                result.push(':');
                result.push_str(&port.to_string());
            }
        }
    }

    // Normalize path: remove dot segments, ensure leading slash if authority present
    let normalized_path = normalize_path(&parts.path);
    if has_authority && !normalized_path.starts_with('/') && !normalized_path.is_empty() {
        result.push('/');
    }
    if has_authority && normalized_path.is_empty() {
        result.push('/');
    } else {
        result.push_str(&normalized_path);
    }

    // Sort query parameters for canonical form
    if let Some(q) = parts.query {
        let mut pairs: Vec<(String, String)> = q
            .split('&')
            .filter(|p| !p.is_empty())
            .map(|pair| match pair.find('=') {
                Some(i) => (pair[..i].to_string(), pair[i + 1..].to_string()),
                None => (pair.to_string(), String::new()),
            })
            .collect();
        pairs.sort_by(|a, b| a.0.cmp(&b.0));

        if !pairs.is_empty() {
            result.push('?');
            for (i, (k, v)) in pairs.iter().enumerate() {
                if i > 0 {
                    result.push('&');
                }
                result.push_str(k);
                if !v.is_empty() {
                    result.push('=');
                    result.push_str(v);
                }
            }
        }
    }

    if let Some(f) = parts.fragment {
        result.push('#');
        result.push_str(&f);
    }

    Ok(result)
}

fn normalize_path(path: &str) -> String {
    if path.is_empty() {
        return String::new();
    }

    let segments: Vec<&str> = path.split('/').collect();
    let mut stack: Vec<&str> = Vec::with_capacity(segments.len());
    let is_absolute = path.starts_with('/');

    for seg in &segments {
        match *seg {
            "" | "." => {}
            ".." => {
                if let Some(top) = stack.last() {
                    if *top != ".." {
                        stack.pop();
                        continue;
                    }
                }
                if !is_absolute {
                    stack.push("..");
                }
            }
            s => stack.push(s),
        }
    }

    let mut result = String::new();
    if is_absolute {
        result.push('/');
    }
    result.push_str(&stack.join("/"));
    if path.ends_with('/') && !result.ends_with('/') {
        result.push('/');
    }
    result
}

/// Returns the registered domain (eTLD+1) from a URL.
/// Uses a simple heuristic: the last two labels, or last three for known multi-level TLDs.
pub fn url_domain(text: &str) -> Result<String> {
    let parts = url_parse(text)?;
    if parts.host.is_empty() {
        return Err(ZyronError::ExecutionError("URL has no host".into()));
    }

    // If host is an IP address, return as-is
    if is_ip_address(&parts.host) {
        return Ok(parts.host);
    }

    let labels: Vec<&str> = parts.host.split('.').collect();
    if labels.len() < 2 {
        return Ok(parts.host);
    }

    // Known two-part TLDs
    let two_part_tlds = [
        "co.uk", "co.jp", "co.nz", "co.za", "co.kr", "co.in", "com.au", "com.br", "com.cn",
        "com.hk", "com.mx", "com.sg", "com.tw", "org.uk", "net.au", "edu.au", "gov.uk", "ac.uk",
        "ac.jp",
    ];
    let last_two = format!("{}.{}", labels[labels.len() - 2], labels[labels.len() - 1]);
    if two_part_tlds.contains(&last_two.as_str()) && labels.len() >= 3 {
        Ok(labels[labels.len() - 3..].join("."))
    } else {
        Ok(labels[labels.len() - 2..].join("."))
    }
}

/// Extracts the top-level domain (last label) from a URL host.
pub fn url_tld(text: &str) -> Result<String> {
    let parts = url_parse(text)?;
    if parts.host.is_empty() {
        return Err(ZyronError::ExecutionError("URL has no host".into()));
    }
    if is_ip_address(&parts.host) {
        return Err(ZyronError::ExecutionError("Host is an IP address".into()));
    }
    let labels: Vec<&str> = parts.host.split('.').collect();
    if labels.len() < 2 {
        return Err(ZyronError::ExecutionError("No TLD found".into()));
    }
    Ok(labels[labels.len() - 1].to_string())
}

fn is_ip_address(host: &str) -> bool {
    // IPv4
    let parts: Vec<&str> = host.split('.').collect();
    if parts.len() == 4 && parts.iter().all(|p| p.parse::<u8>().is_ok()) {
        return true;
    }
    // IPv6 (simplified check)
    host.contains(':') && host.chars().all(|c| c.is_ascii_hexdigit() || c == ':')
}

/// Returns true if the URL is absolute (has a scheme).
pub fn url_is_absolute(text: &str) -> bool {
    url_parse(text).is_ok()
}

/// Resolves a relative URL against a base URL per RFC 3986.
pub fn url_resolve(base: &str, relative: &str) -> Result<String> {
    // If relative is already absolute, return it normalized
    if url_is_absolute(relative) {
        return Ok(relative.to_string());
    }

    let base_parts = url_parse(base)?;

    // Fragment-only relative reference
    if let Some(fragment_ref) = relative.strip_prefix('#') {
        return Ok(format_url_with_fragment(&base_parts, Some(fragment_ref)));
    }

    // Query-only relative reference
    if let Some(query_ref) = relative.strip_prefix('?') {
        let (q, f) = match query_ref.find('#') {
            Some(idx) => (&query_ref[..idx], Some(&query_ref[idx + 1..])),
            None => (query_ref, None),
        };
        return Ok(format_url_with_query(&base_parts, q, f));
    }

    // Network-path reference (starts with //)
    if let Some(rest) = relative.strip_prefix("//") {
        return Ok(format!("{}://{}", base_parts.scheme, rest));
    }

    // Absolute path reference (starts with /)
    if relative.starts_with('/') {
        return Ok(format_url_with_path(&base_parts, relative));
    }

    // Relative path reference: merge with base path
    let base_dir = match base_parts.path.rfind('/') {
        Some(idx) => &base_parts.path[..=idx],
        None => "/",
    };
    let merged = format!("{}{}", base_dir, relative);
    let normalized = normalize_path(&merged);
    Ok(format_url_with_path(&base_parts, &normalized))
}

fn format_url_with_path(base: &UrlParts, path: &str) -> String {
    let (q, f) = match path.find('#') {
        Some(idx) => {
            let (p, rest) = path.split_at(idx);
            let frag = &rest[1..];
            let (pp, qq) = match p.find('?') {
                Some(qi) => (&p[..qi], Some(&p[qi + 1..])),
                None => (p, None),
            };
            return build_url(base, pp, qq, Some(frag));
        }
        None => match path.find('?') {
            Some(qi) => (Some(&path[qi + 1..]), None),
            None => (None, None),
        },
    };
    let clean_path = path
        .split('?')
        .next()
        .unwrap_or(path)
        .split('#')
        .next()
        .unwrap_or(path);
    build_url(base, clean_path, q, f)
}

fn format_url_with_query(base: &UrlParts, query: &str, fragment: Option<&str>) -> String {
    build_url(base, &base.path, Some(query), fragment)
}

fn format_url_with_fragment(base: &UrlParts, fragment: Option<&str>) -> String {
    build_url(base, &base.path, base.query.as_deref(), fragment)
}

fn build_url(base: &UrlParts, path: &str, query: Option<&str>, fragment: Option<&str>) -> String {
    let mut result = String::new();
    result.push_str(&base.scheme);
    result.push_str("://");
    if let Some(user) = &base.user {
        result.push_str(user);
        if let Some(pw) = &base.password {
            result.push(':');
            result.push_str(pw);
        }
        result.push('@');
    }
    result.push_str(&base.host);
    if let Some(port) = base.port {
        result.push(':');
        result.push_str(&port.to_string());
    }
    if !path.starts_with('/') && !path.is_empty() {
        result.push('/');
    }
    result.push_str(path);
    if let Some(q) = query {
        result.push('?');
        result.push_str(q);
    }
    if let Some(f) = fragment {
        result.push('#');
        result.push_str(f);
    }
    result
}

/// Simple URL percent-decoding (for query parameter values).
fn url_decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut result = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(hi), Some(lo)) = (hex_val(bytes[i + 1]), hex_val(bytes[i + 2])) {
                result.push((hi << 4) | lo);
                i += 3;
                continue;
            }
        }
        if bytes[i] == b'+' {
            result.push(b' ');
            i += 1;
            continue;
        }
        result.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&result).into_owned()
}

fn hex_val(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple() {
        let u = url_parse("https://example.com/path").unwrap();
        assert_eq!(u.scheme, "https");
        assert_eq!(u.host, "example.com");
        assert_eq!(u.path, "/path");
    }

    #[test]
    fn test_parse_with_port() {
        let u = url_parse("http://example.com:8080/").unwrap();
        assert_eq!(u.port, Some(8080));
    }

    #[test]
    fn test_parse_with_userinfo() {
        let u = url_parse("https://user:pass@example.com/").unwrap();
        assert_eq!(u.user, Some("user".to_string()));
        assert_eq!(u.password, Some("pass".to_string()));
    }

    #[test]
    fn test_parse_with_query() {
        let u = url_parse("https://example.com/path?a=1&b=2").unwrap();
        assert_eq!(u.query, Some("a=1&b=2".to_string()));
    }

    #[test]
    fn test_parse_with_fragment() {
        let u = url_parse("https://example.com/path#section").unwrap();
        assert_eq!(u.fragment, Some("section".to_string()));
    }

    #[test]
    fn test_parse_complete() {
        let u = url_parse("https://user:pass@example.com:8080/path?q=1#frag").unwrap();
        assert_eq!(u.scheme, "https");
        assert_eq!(u.user, Some("user".to_string()));
        assert_eq!(u.password, Some("pass".to_string()));
        assert_eq!(u.host, "example.com");
        assert_eq!(u.port, Some(8080));
        assert_eq!(u.path, "/path");
        assert_eq!(u.query, Some("q=1".to_string()));
        assert_eq!(u.fragment, Some("frag".to_string()));
    }

    #[test]
    fn test_parse_mailto() {
        let u = url_parse("mailto:user@example.com").unwrap();
        assert_eq!(u.scheme, "mailto");
        assert_eq!(u.path, "user@example.com");
        assert!(u.host.is_empty());
    }

    #[test]
    fn test_parse_invalid() {
        assert!(url_parse("").is_err());
        assert!(url_parse("not a url").is_err());
    }

    #[test]
    fn test_scheme() {
        assert_eq!(url_scheme("HTTP://example.com").unwrap(), "http");
    }

    #[test]
    fn test_host() {
        assert_eq!(url_host("https://Example.COM/").unwrap(), "example.com");
    }

    #[test]
    fn test_port_specified() {
        assert_eq!(url_port("http://example.com:8080/").unwrap(), Some(8080));
    }

    #[test]
    fn test_port_none() {
        assert_eq!(url_port("http://example.com/").unwrap(), None);
    }

    #[test]
    fn test_path() {
        assert_eq!(url_path("https://example.com/foo/bar").unwrap(), "/foo/bar");
    }

    #[test]
    fn test_query_param() {
        let val = url_query_param("https://example.com/?q=hello", "q").unwrap();
        assert_eq!(val, Some("hello".to_string()));
    }

    #[test]
    fn test_query_param_missing() {
        let val = url_query_param("https://example.com/?q=hello", "missing").unwrap();
        assert_eq!(val, None);
    }

    #[test]
    fn test_query_param_url_encoded() {
        let val = url_query_param("https://example.com/?name=hello%20world", "name").unwrap();
        assert_eq!(val, Some("hello world".to_string()));
    }

    #[test]
    fn test_query_params_all() {
        let params = url_query_params("https://example.com/?a=1&b=2&c=3").unwrap();
        assert_eq!(params.len(), 3);
        assert_eq!(params[0], ("a".to_string(), "1".to_string()));
    }

    #[test]
    fn test_fragment() {
        assert_eq!(
            url_fragment("https://example.com/#section").unwrap(),
            Some("section".to_string())
        );
    }

    #[test]
    fn test_normalize_default_port() {
        let n = url_normalize("http://example.com:80/path").unwrap();
        assert!(!n.contains(":80"));
    }

    #[test]
    fn test_normalize_lowercase() {
        let n = url_normalize("HTTPS://EXAMPLE.COM/Path").unwrap();
        assert!(n.starts_with("https://example.com"));
    }

    #[test]
    fn test_normalize_sort_params() {
        let n = url_normalize("https://example.com/?b=2&a=1").unwrap();
        // After normalize, a should come before b
        let q_start = n.find('?').unwrap();
        let a_pos = n[q_start..].find("a=").unwrap();
        let b_pos = n[q_start..].find("b=").unwrap();
        assert!(a_pos < b_pos);
    }

    #[test]
    fn test_normalize_dot_segments() {
        let n = url_normalize("http://example.com/a/./b/../c").unwrap();
        assert!(n.contains("/a/c"));
    }

    #[test]
    fn test_domain_basic() {
        assert_eq!(
            url_domain("https://www.example.com/").unwrap(),
            "example.com"
        );
        assert_eq!(url_domain("https://example.com/").unwrap(), "example.com");
    }

    #[test]
    fn test_domain_multipart_tld() {
        assert_eq!(
            url_domain("https://www.example.co.uk/").unwrap(),
            "example.co.uk"
        );
    }

    #[test]
    fn test_domain_ip() {
        assert_eq!(url_domain("http://192.168.1.1/").unwrap(), "192.168.1.1");
    }

    #[test]
    fn test_tld() {
        assert_eq!(url_tld("https://example.com").unwrap(), "com");
        assert_eq!(url_tld("https://example.co.uk").unwrap(), "uk");
    }

    #[test]
    fn test_is_absolute() {
        assert!(url_is_absolute("https://example.com/"));
        assert!(!url_is_absolute("/path/only"));
        assert!(!url_is_absolute("../relative"));
    }

    #[test]
    fn test_resolve_absolute() {
        let resolved = url_resolve("https://example.com/", "https://other.com/").unwrap();
        assert_eq!(resolved, "https://other.com/");
    }

    #[test]
    fn test_resolve_absolute_path() {
        let resolved = url_resolve("https://example.com/foo/bar", "/other").unwrap();
        assert!(resolved.contains("/other"));
        assert!(resolved.contains("example.com"));
    }

    #[test]
    fn test_resolve_relative() {
        let resolved = url_resolve("https://example.com/foo/bar", "baz").unwrap();
        assert!(resolved.contains("/foo/baz"));
    }

    #[test]
    fn test_resolve_fragment_only() {
        let resolved = url_resolve("https://example.com/page", "#section").unwrap();
        assert_eq!(resolved, "https://example.com/page#section");
    }

    #[test]
    fn test_ipv6_host() {
        let u = url_parse("http://[::1]:8080/").unwrap();
        assert_eq!(u.host, "::1");
        assert_eq!(u.port, Some(8080));
    }
}

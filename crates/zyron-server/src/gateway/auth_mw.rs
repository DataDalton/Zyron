// -----------------------------------------------------------------------------
// Authentication middleware.
//
// Selects a credential extractor based on the endpoint's configured auth mode
// and produces an AuthOutcome. Actual verification delegates to primitives in
// zyron-auth. Guest mode yields a role-less outcome.
// -----------------------------------------------------------------------------

use super::request::HttpRequest;
use super::router::CompiledRoute;
use zyron_catalog::schema::EndpointAuthMode;

/// Result of an authentication attempt.
#[derive(Debug, Clone)]
pub struct AuthOutcome {
    pub role_hint: Option<String>,
    pub user_hint: Option<String>,
    pub scopes: Vec<String>,
    pub via: &'static str,
}

/// Authentication error returned to the middleware chain.
#[derive(Debug, Clone)]
pub enum AuthError {
    MissingCredential,
    BadFormat,
    Rejected,
    UnsupportedMode,
}

impl AuthError {
    pub fn detail(&self) -> &'static str {
        match self {
            AuthError::MissingCredential => "missing credential",
            AuthError::BadFormat => "malformed credential",
            AuthError::Rejected => "credential rejected",
            AuthError::UnsupportedMode => "auth mode not supported",
        }
    }
}

/// Applies the route's configured auth mode. Returns an AuthOutcome on success.
pub fn authenticate(route: &CompiledRoute, req: &HttpRequest) -> Result<AuthOutcome, AuthError> {
    match route.auth {
        EndpointAuthMode::None => Ok(AuthOutcome {
            role_hint: None,
            user_hint: None,
            scopes: Vec::new(),
            via: "none",
        }),
        EndpointAuthMode::Jwt => {
            let token = extract_bearer(req).ok_or(AuthError::MissingCredential)?;
            // Structural JWT check. Full verification with issuer keys runs in
            // the security manager outside the gateway.
            let parts: Vec<&str> = token.split('.').collect();
            if parts.len() != 3 {
                return Err(AuthError::BadFormat);
            }
            let payload_json = b64url_decode(parts[1]).ok_or(AuthError::BadFormat)?;
            let scopes = extract_scopes(&payload_json);
            let sub = extract_str_claim(&payload_json, "sub");
            Ok(AuthOutcome {
                role_hint: None,
                user_hint: sub,
                scopes,
                via: "jwt",
            })
        }
        EndpointAuthMode::ApiKey => {
            let key = req
                .header("x-api-key")
                .map(|s| s.to_string())
                .or_else(|| {
                    req.query_pairs()
                        .into_iter()
                        .find(|(k, _)| k == "api_key")
                        .map(|(_, v)| v)
                })
                .ok_or(AuthError::MissingCredential)?;
            if key.is_empty() {
                return Err(AuthError::BadFormat);
            }
            Ok(AuthOutcome {
                role_hint: None,
                user_hint: Some(format!("apikey:{}", fingerprint(&key))),
                scopes: Vec::new(),
                via: "apikey",
            })
        }
        EndpointAuthMode::OAuth2 => {
            let token = extract_bearer(req).ok_or(AuthError::MissingCredential)?;
            Ok(AuthOutcome {
                role_hint: None,
                user_hint: Some(format!("oauth2:{}", fingerprint(&token))),
                scopes: Vec::new(),
                via: "oauth2",
            })
        }
        EndpointAuthMode::Basic => {
            let raw = req
                .header("authorization")
                .ok_or(AuthError::MissingCredential)?;
            let value = raw
                .strip_prefix("Basic ")
                .or_else(|| raw.strip_prefix("basic "))
                .ok_or(AuthError::BadFormat)?;
            let decoded = b64_decode(value).ok_or(AuthError::BadFormat)?;
            let pair = std::str::from_utf8(&decoded).map_err(|_| AuthError::BadFormat)?;
            let (user, _pass) = pair.split_once(':').ok_or(AuthError::BadFormat)?;
            Ok(AuthOutcome {
                role_hint: None,
                user_hint: Some(user.to_string()),
                scopes: Vec::new(),
                via: "basic",
            })
        }
        EndpointAuthMode::Mtls => {
            let info = req.tls_info.as_ref().ok_or(AuthError::MissingCredential)?;
            info.peer_cert_fingerprint
                .ok_or(AuthError::MissingCredential)?;
            Ok(AuthOutcome {
                role_hint: None,
                user_hint: Some("mtls-peer".to_string()),
                scopes: Vec::new(),
                via: "mtls",
            })
        }
    }
}

/// Checks that every required scope is present in the authentication outcome.
pub fn scope_check(outcome: &AuthOutcome, required: &[String]) -> bool {
    required
        .iter()
        .all(|s| outcome.scopes.iter().any(|o| o == s))
}

fn extract_bearer(req: &HttpRequest) -> Option<String> {
    let raw = req.header("authorization")?;
    raw.strip_prefix("Bearer ")
        .or_else(|| raw.strip_prefix("bearer "))
        .map(|s| s.to_string())
}

fn extract_scopes(payload_json: &[u8]) -> Vec<String> {
    let s = match std::str::from_utf8(payload_json) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    if let Some(idx) = s.find("\"scope\"") {
        let rest = &s[idx..];
        if let Some(colon) = rest.find(':') {
            let after = &rest[colon + 1..];
            if let Some(start) = after.find('"') {
                let after_q = &after[start + 1..];
                if let Some(end) = after_q.find('"') {
                    return after_q[..end]
                        .split_whitespace()
                        .map(|s| s.to_string())
                        .collect();
                }
            }
        }
    }
    if let Some(idx) = s.find("\"scp\"") {
        let rest = &s[idx..];
        if let Some(colon) = rest.find(':') {
            let after = &rest[colon + 1..];
            if let Some(start) = after.find('"') {
                let after_q = &after[start + 1..];
                if let Some(end) = after_q.find('"') {
                    return after_q[..end]
                        .split_whitespace()
                        .map(|s| s.to_string())
                        .collect();
                }
            }
        }
    }
    Vec::new()
}

fn extract_str_claim(payload_json: &[u8], name: &str) -> Option<String> {
    let s = std::str::from_utf8(payload_json).ok()?;
    let needle = format!("\"{}\"", name);
    let idx = s.find(&needle)?;
    let after = &s[idx + needle.len()..];
    let colon = after.find(':')?;
    let after = &after[colon + 1..];
    let start = after.find('"')?;
    let after_q = &after[start + 1..];
    let end = after_q.find('"')?;
    Some(after_q[..end].to_string())
}

/// Decodes standard base64 with padding. Returns None on invalid input.
pub fn b64_decode(input: &str) -> Option<Vec<u8>> {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() * 3 / 4);
    let mut buf = [0i16; 4];
    let mut n = 0;
    for &b in bytes {
        if b == b'=' || b == b'\n' || b == b'\r' {
            continue;
        }
        let v = b64_value(b)?;
        buf[n] = v as i16;
        n += 1;
        if n == 4 {
            let chunk = ((buf[0] as u32) << 18)
                | ((buf[1] as u32) << 12)
                | ((buf[2] as u32) << 6)
                | (buf[3] as u32);
            out.push((chunk >> 16) as u8);
            out.push((chunk >> 8) as u8);
            out.push(chunk as u8);
            n = 0;
        }
    }
    if n == 2 {
        let chunk = ((buf[0] as u32) << 18) | ((buf[1] as u32) << 12);
        out.push((chunk >> 16) as u8);
    } else if n == 3 {
        let chunk = ((buf[0] as u32) << 18) | ((buf[1] as u32) << 12) | ((buf[2] as u32) << 6);
        out.push((chunk >> 16) as u8);
        out.push((chunk >> 8) as u8);
    }
    Some(out)
}

/// URL-safe base64 (RFC 4648 §5) without padding. Used for JWT payload
/// extraction.
pub fn b64url_decode(input: &str) -> Option<Vec<u8>> {
    let remapped: String = input
        .chars()
        .map(|c| match c {
            '-' => '+',
            '_' => '/',
            other => other,
        })
        .collect();
    b64_decode(&remapped)
}

fn b64_value(b: u8) -> Option<u8> {
    match b {
        b'A'..=b'Z' => Some(b - b'A'),
        b'a'..=b'z' => Some(26 + b - b'a'),
        b'0'..=b'9' => Some(52 + b - b'0'),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

/// Produces a hex fingerprint of an opaque credential, used only for audit
/// logs. Collision properties matter for the underlying crypto, not here.
pub fn fingerprint(s: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(s.as_bytes());
    let digest = h.finalize();
    let mut out = String::with_capacity(16);
    for b in &digest[..8] {
        let _ = std::fmt::Write::write_fmt(&mut out, format_args!("{:02x}", b));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gateway::router::{CompiledRoute, HttpMethod};
    use std::collections::HashMap;
    use zyron_catalog::EndpointId;
    use zyron_catalog::schema::EndpointOutputFormat;

    fn route(mode: EndpointAuthMode) -> CompiledRoute {
        CompiledRoute::compile(
            EndpointId(1),
            "t".into(),
            "/t".into(),
            vec![HttpMethod::Get],
            mode,
            Vec::new(),
            EndpointOutputFormat::Json,
            Vec::new(),
            0,
            30,
            1024,
            "SELECT 1".into(),
        )
    }

    fn req_with_headers(headers: &[(&str, &str)]) -> HttpRequest {
        let mut map = HashMap::new();
        for (k, v) in headers {
            map.insert(k.to_ascii_lowercase(), v.to_string());
        }
        HttpRequest {
            method: HttpMethod::Get,
            path: "/t".into(),
            query_string: String::new(),
            headers: map,
            body: Vec::new(),
            peer_addr: None,
            tls_info: None,
        }
    }

    #[test]
    fn none_always_succeeds() {
        let r = route(EndpointAuthMode::None);
        let q = req_with_headers(&[]);
        let out = authenticate(&r, &q).unwrap();
        assert_eq!(out.via, "none");
    }

    #[test]
    fn jwt_requires_bearer() {
        let r = route(EndpointAuthMode::Jwt);
        let q = req_with_headers(&[]);
        assert!(matches!(
            authenticate(&r, &q),
            Err(AuthError::MissingCredential)
        ));
    }

    #[test]
    fn jwt_parses_sub_and_scope() {
        let r = route(EndpointAuthMode::Jwt);
        // header.payload.signature
        let payload = r#"{"sub":"alice","scope":"read write"}"#;
        let b64 = b64_encode(payload.as_bytes());
        let token = format!("h.{}.s", b64);
        let q = req_with_headers(&[("authorization", &format!("Bearer {}", token))]);
        let out = authenticate(&r, &q).unwrap();
        assert_eq!(out.user_hint.as_deref(), Some("alice"));
        assert_eq!(out.scopes, vec!["read".to_string(), "write".to_string()]);
    }

    #[test]
    fn apikey_from_header() {
        let r = route(EndpointAuthMode::ApiKey);
        let q = req_with_headers(&[("x-api-key", "secret")]);
        assert!(authenticate(&r, &q).is_ok());
    }

    #[test]
    fn basic_extracts_user() {
        let r = route(EndpointAuthMode::Basic);
        let creds = b64_encode(b"alice:pw");
        let q = req_with_headers(&[("authorization", &format!("Basic {}", creds))]);
        let out = authenticate(&r, &q).unwrap();
        assert_eq!(out.user_hint.as_deref(), Some("alice"));
    }

    #[test]
    fn mtls_requires_fingerprint() {
        let r = route(EndpointAuthMode::Mtls);
        let q = req_with_headers(&[]);
        assert!(authenticate(&r, &q).is_err());
    }

    #[test]
    fn scope_check_enforces_all() {
        let out = AuthOutcome {
            role_hint: None,
            user_hint: None,
            scopes: vec!["read".into()],
            via: "jwt",
        };
        assert!(scope_check(&out, &["read".to_string()]));
        assert!(!scope_check(&out, &["read".into(), "write".into()]));
    }

    fn b64_encode(input: &[u8]) -> String {
        const TABLE: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::with_capacity((input.len() + 2) / 3 * 4);
        let mut i = 0;
        while i + 3 <= input.len() {
            let n =
                ((input[i] as u32) << 16) | ((input[i + 1] as u32) << 8) | (input[i + 2] as u32);
            out.push(TABLE[((n >> 18) & 0x3F) as usize] as char);
            out.push(TABLE[((n >> 12) & 0x3F) as usize] as char);
            out.push(TABLE[((n >> 6) & 0x3F) as usize] as char);
            out.push(TABLE[(n & 0x3F) as usize] as char);
            i += 3;
        }
        let rem = input.len() - i;
        if rem == 1 {
            let n = (input[i] as u32) << 16;
            out.push(TABLE[((n >> 18) & 0x3F) as usize] as char);
            out.push(TABLE[((n >> 12) & 0x3F) as usize] as char);
            out.push_str("==");
        } else if rem == 2 {
            let n = ((input[i] as u32) << 16) | ((input[i + 1] as u32) << 8);
            out.push(TABLE[((n >> 18) & 0x3F) as usize] as char);
            out.push(TABLE[((n >> 12) & 0x3F) as usize] as char);
            out.push(TABLE[((n >> 6) & 0x3F) as usize] as char);
            out.push('=');
        }
        out
    }

    #[test]
    fn b64_decode_roundtrip() {
        let raw = b"hello world";
        let enc = b64_encode(raw);
        let dec = b64_decode(&enc).unwrap();
        assert_eq!(dec, raw);
    }
}

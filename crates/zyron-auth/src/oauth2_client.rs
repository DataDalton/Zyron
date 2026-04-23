//! OAuth 2.0 client-credentials grant flow.
//!
//! POSTs form-encoded credentials to a token endpoint and parses the JSON
//! response for `access_token`, `token_type`, and `expires_in`. Returns the
//! token material plus the advertised TTL so the credential cache can manage
//! refresh timing.

use std::collections::HashMap;
use std::time::Duration;

use async_trait::async_trait;
use zyron_common::{Result, ZyronError};

use crate::credential_provider::CredentialProvider;

// -----------------------------------------------------------------------------
// Provider
// -----------------------------------------------------------------------------

/// OAuth 2.0 client-credentials provider. `scope` and `audience` are optional
/// per RFC 6749 / RFC 8707.
pub struct OAuth2ClientCredentialsProvider {
    pub token_endpoint: String,
    pub client_id: String,
    pub client_secret: String,
    pub scope: Option<String>,
    pub audience: Option<String>,
    http_client: reqwest::Client,
}

impl OAuth2ClientCredentialsProvider {
    /// Creates a new provider. Builds a reqwest client with a 10 second total
    /// timeout so a stalled token endpoint cannot wedge the caller.
    pub fn new(
        token_endpoint: String,
        client_id: String,
        client_secret: String,
        scope: Option<String>,
        audience: Option<String>,
    ) -> Result<Self> {
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .map_err(|e| {
                ZyronError::InvalidCredential(format!("OAuth2 http client build failed: {}", e))
            })?;
        Ok(Self {
            token_endpoint,
            client_id,
            client_secret,
            scope,
            audience,
            http_client,
        })
    }

    /// Supplies a custom reqwest client. Primarily used in tests that need a
    /// client pointing at a mock server.
    pub fn with_http_client(mut self, client: reqwest::Client) -> Self {
        self.http_client = client;
        self
    }
}

// -----------------------------------------------------------------------------
// CredentialProvider impl
// -----------------------------------------------------------------------------

#[async_trait]
impl CredentialProvider for OAuth2ClientCredentialsProvider {
    async fn fetch(&self) -> Result<(HashMap<String, String>, Duration)> {
        let mut form: Vec<(&str, &str)> = vec![
            ("grant_type", "client_credentials"),
            ("client_id", self.client_id.as_str()),
            ("client_secret", self.client_secret.as_str()),
        ];
        if let Some(ref scope) = self.scope {
            form.push(("scope", scope.as_str()));
        }
        if let Some(ref aud) = self.audience {
            form.push(("audience", aud.as_str()));
        }

        let resp = self
            .http_client
            .post(&self.token_endpoint)
            .form(&form)
            .send()
            .await
            .map_err(|e| {
                ZyronError::AuthenticationFailed(format!("OAuth2 request failed: {}", e))
            })?;

        let status = resp.status();
        let body = resp.text().await.map_err(|e| {
            ZyronError::AuthenticationFailed(format!("OAuth2 body read failed: {}", e))
        })?;

        if !status.is_success() {
            return Err(ZyronError::AuthenticationFailed(format!(
                "OAuth2 token endpoint returned {}: {}",
                status.as_u16(),
                body
            )));
        }

        let (token, token_type, expires_in) = parse_token_response(&body)?;
        let mut out: HashMap<String, String> = HashMap::new();
        out.insert("token".to_string(), token);
        out.insert("token_type".to_string(), token_type);
        Ok((out, Duration::from_secs(expires_in)))
    }

    fn provider_kind(&self) -> &'static str {
        "oauth2_client_credentials"
    }
}

// -----------------------------------------------------------------------------
// Minimal JSON extraction for access_token/token_type/expires_in
// -----------------------------------------------------------------------------

/// Extracts the three fields from a JSON body without pulling in a full
/// mapping layer. Missing `token_type` defaults to `Bearer`. Missing
/// `expires_in` defaults to 3600 seconds (RFC 6749 fallback).
fn parse_token_response(body: &str) -> Result<(String, String, u64)> {
    let token = extract_json_string(body, "access_token").ok_or_else(|| {
        ZyronError::AuthenticationFailed("OAuth2 response missing access_token".to_string())
    })?;
    let token_type =
        extract_json_string(body, "token_type").unwrap_or_else(|| "Bearer".to_string());
    let expires_in = extract_json_number(body, "expires_in").unwrap_or(3600);
    if expires_in == 0 {
        return Err(ZyronError::AuthenticationFailed(
            "OAuth2 expires_in was zero".to_string(),
        ));
    }
    Ok((token, token_type, expires_in))
}

fn extract_json_string(body: &str, key: &str) -> Option<String> {
    let needle = format!("\"{}\"", key);
    let kpos = body.find(&needle)?;
    let after_key = &body[kpos + needle.len()..];
    let colon = after_key.find(':')?;
    let rest = &after_key[colon + 1..];
    let trimmed = rest.trim_start();
    if !trimmed.starts_with('"') {
        return None;
    }
    let val_start = trimmed.as_ptr() as usize - body.as_ptr() as usize + 1;
    let mut i = val_start;
    let bytes = body.as_bytes();
    let mut out = String::new();
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'\\' && i + 1 < bytes.len() {
            let nxt = bytes[i + 1];
            match nxt {
                b'"' => out.push('"'),
                b'\\' => out.push('\\'),
                b'/' => out.push('/'),
                b'n' => out.push('\n'),
                b'r' => out.push('\r'),
                b't' => out.push('\t'),
                _ => out.push(nxt as char),
            }
            i += 2;
        } else if b == b'"' {
            return Some(out);
        } else {
            out.push(b as char);
            i += 1;
        }
    }
    None
}

fn extract_json_number(body: &str, key: &str) -> Option<u64> {
    let needle = format!("\"{}\"", key);
    let kpos = body.find(&needle)?;
    let after_key = &body[kpos + needle.len()..];
    let colon = after_key.find(':')?;
    let rest = &after_key[colon + 1..];
    let trimmed = rest.trim_start();
    let end = trimmed
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(trimmed.len());
    if end == 0 {
        return None;
    }
    trimmed[..end].parse::<u64>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use httpmock::prelude::*;

    #[test]
    fn parse_response_fields() {
        let body = r#"{"access_token":"xyz123","token_type":"Bearer","expires_in":3600}"#;
        let (t, tt, e) = parse_token_response(body).expect("parse");
        assert_eq!(t, "xyz123");
        assert_eq!(tt, "Bearer");
        assert_eq!(e, 3600);
    }

    #[test]
    fn parse_defaults_on_missing() {
        let body = r#"{"access_token":"abc"}"#;
        let (t, tt, e) = parse_token_response(body).expect("parse");
        assert_eq!(t, "abc");
        assert_eq!(tt, "Bearer");
        assert_eq!(e, 3600);
    }

    #[test]
    fn parse_rejects_missing_token() {
        let body = r#"{"token_type":"Bearer","expires_in":60}"#;
        assert!(parse_token_response(body).is_err());
    }

    #[test]
    fn parse_rejects_zero_expiry() {
        let body = r#"{"access_token":"a","expires_in":0}"#;
        assert!(parse_token_response(body).is_err());
    }

    #[tokio::test]
    async fn mock_success() {
        let server = MockServer::start_async().await;
        let m = server.mock_async(|when, then| {
            when.method(POST).path("/token");
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"access_token":"tok1","token_type":"Bearer","expires_in":120}"#);
        }).await;

        let provider = OAuth2ClientCredentialsProvider::new(
            server.url("/token"),
            "cid".to_string(),
            "csec".to_string(),
            Some("read:publications".to_string()),
            Some("zyron".to_string()),
        )
        .expect("provider");
        let (creds, ttl) = provider.fetch().await.expect("fetch");
        assert_eq!(creds.get("token").map(|s| s.as_str()), Some("tok1"));
        assert_eq!(ttl, Duration::from_secs(120));
        m.assert_async().await;
    }

    #[tokio::test]
    async fn mock_401_returns_auth_error() {
        let server = MockServer::start_async().await;
        let _m = server.mock_async(|when, then| {
            when.method(POST).path("/token");
            then.status(401).body(r#"{"error":"invalid_client"}"#);
        }).await;
        let provider = OAuth2ClientCredentialsProvider::new(
            server.url("/token"),
            "cid".to_string(),
            "csec".to_string(),
            None,
            None,
        )
        .expect("provider");
        let r = provider.fetch().await;
        assert!(r.is_err());
    }

    #[tokio::test]
    async fn mock_500_returns_auth_error() {
        let server = MockServer::start_async().await;
        let _m = server.mock_async(|when, then| {
            when.method(POST).path("/token");
            then.status(500).body("boom");
        }).await;
        let provider = OAuth2ClientCredentialsProvider::new(
            server.url("/token"),
            "cid".to_string(),
            "csec".to_string(),
            None,
            None,
        )
        .expect("provider");
        assert!(provider.fetch().await.is_err());
    }

    #[tokio::test]
    async fn mock_bad_json_errors() {
        let server = MockServer::start_async().await;
        let _m = server.mock_async(|when, then| {
            when.method(POST).path("/token");
            then.status(200).body("not json");
        }).await;
        let provider = OAuth2ClientCredentialsProvider::new(
            server.url("/token"),
            "cid".to_string(),
            "csec".to_string(),
            None,
            None,
        )
        .expect("provider");
        assert!(provider.fetch().await.is_err());
    }
}

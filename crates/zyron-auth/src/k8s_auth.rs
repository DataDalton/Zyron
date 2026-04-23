//! Kubernetes service-account token provider plus TokenReview API client.
//!
//! `K8sTokenProvider` reads a projected-volume service-account token off disk
//! and returns it as the credential. `K8sTokenReviewer` calls the Kubernetes
//! TokenReview API on the producer side to validate a presented SA token and
//! strictly checks that the requested audience is present in the response.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use async_trait::async_trait;
use zyron_common::{Result, ZyronError};

use crate::credential_provider::CredentialProvider;

// -----------------------------------------------------------------------------
// Token provider (consumer side)
// -----------------------------------------------------------------------------

/// Reads a K8s projected-volume SA token from disk. The kubelet rotates this
/// file, so callers set `refresh_interval` to something smaller than the
/// projected token lifetime (default ~1 hour) so the cache refetches before
/// expiry.
pub struct K8sTokenProvider {
    pub token_path: PathBuf,
    pub audience: Option<String>,
    pub refresh_interval: Duration,
}

impl K8sTokenProvider {
    pub fn new(token_path: PathBuf, audience: Option<String>, refresh_interval: Duration) -> Self {
        Self {
            token_path,
            audience,
            refresh_interval,
        }
    }
}

#[async_trait]
impl CredentialProvider for K8sTokenProvider {
    async fn fetch(&self) -> Result<(HashMap<String, String>, Duration)> {
        let raw = tokio::fs::read(&self.token_path).await.map_err(|e| {
            ZyronError::AuthenticationFailed(format!(
                "failed to read K8s SA token at {}: {}",
                self.token_path.display(),
                e
            ))
        })?;
        let token = String::from_utf8(raw).map_err(|_| {
            ZyronError::AuthenticationFailed("K8s SA token is not UTF-8".to_string())
        })?;
        let trimmed = token.trim();
        if trimmed.is_empty() {
            return Err(ZyronError::AuthenticationFailed(
                "K8s SA token file was empty".to_string(),
            ));
        }
        let mut out = HashMap::new();
        out.insert("token".to_string(), trimmed.to_string());
        if let Some(ref aud) = self.audience {
            out.insert("audience".to_string(), aud.clone());
        }
        Ok((out, self.refresh_interval))
    }

    fn provider_kind(&self) -> &'static str {
        "k8s_sa_token"
    }
}

// -----------------------------------------------------------------------------
// TokenReview result
// -----------------------------------------------------------------------------

/// Decoded result of a successful TokenReview call.
#[derive(Debug, Clone)]
pub struct TokenReviewResult {
    pub authenticated: bool,
    pub username: String,
    pub serviceaccount_namespace: String,
    pub serviceaccount_name: String,
    pub audiences: Vec<String>,
}

// -----------------------------------------------------------------------------
// TokenReview REST client (producer side)
// -----------------------------------------------------------------------------

/// Calls the Kubernetes TokenReview API to validate a presented SA token.
///
/// `ca_cert` is the PEM-encoded CA used to verify the apiserver certificate.
/// `own_token` is the SA token this Zyron instance uses to authenticate itself
/// against the apiserver. If `ca_cert` is empty, the default WebPKI roots are
/// used.
pub struct K8sTokenReviewer {
    pub apiserver_url: String,
    pub ca_cert: Vec<u8>,
    pub own_token: String,
    http_client: reqwest::Client,
}

impl K8sTokenReviewer {
    /// Creates a reviewer. Builds a reqwest client with the provided CA
    /// certificate added to the trust store.
    pub fn new(apiserver_url: String, ca_cert: Vec<u8>, own_token: String) -> Result<Self> {
        let mut builder = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .use_rustls_tls();
        if !ca_cert.is_empty() {
            let cert = reqwest::Certificate::from_pem(&ca_cert).map_err(|e| {
                ZyronError::AuthenticationFailed(format!("invalid K8s apiserver CA: {}", e))
            })?;
            builder = builder.add_root_certificate(cert);
        }
        let http_client = builder.build().map_err(|e| {
            ZyronError::AuthenticationFailed(format!("K8s TokenReview client build failed: {}", e))
        })?;
        Ok(Self {
            apiserver_url,
            ca_cert,
            own_token,
            http_client,
        })
    }

    /// Test helper: build with a caller-supplied http client, bypassing TLS
    /// setup. Used against a plain `httpmock` server.
    pub fn with_http_client(
        apiserver_url: String,
        own_token: String,
        http_client: reqwest::Client,
    ) -> Self {
        Self {
            apiserver_url,
            ca_cert: Vec::new(),
            own_token,
            http_client,
        }
    }

    /// Submits a TokenReview and parses the response. Returns an error if the
    /// apiserver rejects the call or the token is not authenticated. If an
    /// audience is requested, the response audiences must contain it.
    pub async fn review_token(
        &self,
        token: &str,
        audience: Option<&str>,
    ) -> Result<TokenReviewResult> {
        let audiences_field = match audience {
            Some(a) => format!(",\"audiences\":[\"{}\"]", a.replace('"', "\\\"")),
            None => String::new(),
        };
        let body = format!(
            "{{\"apiVersion\":\"authentication.k8s.io/v1\",\"kind\":\"TokenReview\",\"spec\":{{\"token\":\"{}\"{}}}}}",
            token.replace('"', "\\\""),
            audiences_field
        );

        let url = format!(
            "{}/apis/authentication.k8s.io/v1/tokenreviews",
            self.apiserver_url.trim_end_matches('/')
        );
        let resp = self
            .http_client
            .post(url)
            .bearer_auth(&self.own_token)
            .header("Content-Type", "application/json")
            .body(body)
            .send()
            .await
            .map_err(|e| {
                ZyronError::AuthenticationFailed(format!("TokenReview request failed: {}", e))
            })?;

        let status = resp.status();
        let text = resp.text().await.map_err(|e| {
            ZyronError::AuthenticationFailed(format!("TokenReview body read failed: {}", e))
        })?;
        if !status.is_success() {
            return Err(ZyronError::AuthenticationFailed(format!(
                "TokenReview returned {}: {}",
                status.as_u16(),
                text
            )));
        }

        let result = parse_token_review(&text)?;
        if !result.authenticated {
            return Err(ZyronError::AuthenticationFailed(
                "TokenReview rejected token".to_string(),
            ));
        }
        if let Some(expected) = audience {
            if !result.audiences.iter().any(|a| a == expected) {
                return Err(ZyronError::AuthenticationFailed(format!(
                    "TokenReview missing required audience {}",
                    expected
                )));
            }
        }
        Ok(result)
    }
}

// -----------------------------------------------------------------------------
// TokenReview response parser
// -----------------------------------------------------------------------------

fn parse_token_review(body: &str) -> Result<TokenReviewResult> {
    let authenticated = find_bool(body, "authenticated").unwrap_or(false);
    let username = find_string(body, "username").unwrap_or_default();
    let audiences = find_string_array(body, "audiences");
    let (ns, name) = parse_sa_identity(&username);
    Ok(TokenReviewResult {
        authenticated,
        username,
        serviceaccount_namespace: ns,
        serviceaccount_name: name,
        audiences,
    })
}

fn find_bool(body: &str, key: &str) -> Option<bool> {
    let needle = format!("\"{}\"", key);
    let p = body.find(&needle)?;
    let rest = &body[p + needle.len()..];
    let colon = rest.find(':')?;
    let after = rest[colon + 1..].trim_start();
    if after.starts_with("true") {
        Some(true)
    } else if after.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

fn find_string(body: &str, key: &str) -> Option<String> {
    let needle = format!("\"{}\"", key);
    let p = body.find(&needle)?;
    let rest = &body[p + needle.len()..];
    let colon = rest.find(':')?;
    let after = rest[colon + 1..].trim_start();
    if !after.starts_with('"') {
        return None;
    }
    let inner = &after[1..];
    let end = inner.find('"')?;
    Some(inner[..end].to_string())
}

fn find_string_array(body: &str, key: &str) -> Vec<String> {
    let needle = format!("\"{}\"", key);
    let p = match body.find(&needle) {
        Some(x) => x,
        None => return Vec::new(),
    };
    let rest = &body[p + needle.len()..];
    let colon = match rest.find(':') {
        Some(x) => x,
        None => return Vec::new(),
    };
    let after = rest[colon + 1..].trim_start();
    if !after.starts_with('[') {
        return Vec::new();
    }
    let bracket_end = match after.find(']') {
        Some(x) => x,
        None => return Vec::new(),
    };
    let inner = &after[1..bracket_end];
    let mut out = Vec::new();
    let mut i = 0;
    let bytes = inner.as_bytes();
    while i < bytes.len() {
        if bytes[i] == b'"' {
            let start = i + 1;
            let mut j = start;
            while j < bytes.len() && bytes[j] != b'"' {
                j += 1;
            }
            if j > start {
                out.push(
                    std::str::from_utf8(&bytes[start..j])
                        .unwrap_or("")
                        .to_string(),
                );
            }
            i = j + 1;
        } else {
            i += 1;
        }
    }
    out
}

fn parse_sa_identity(username: &str) -> (String, String) {
    // Format: "system:serviceaccount:<namespace>:<name>"
    let parts: Vec<&str> = username.split(':').collect();
    if parts.len() == 4 && parts[0] == "system" && parts[1] == "serviceaccount" {
        (parts[2].to_string(), parts[3].to_string())
    } else {
        (String::new(), String::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use httpmock::prelude::*;
    use std::io::Write;

    #[test]
    fn parse_authenticated_response() {
        let body = r#"{
          "apiVersion":"authentication.k8s.io/v1",
          "kind":"TokenReview",
          "status":{
            "authenticated":true,
            "user":{"username":"system:serviceaccount:prod:zyron-reader"},
            "username":"system:serviceaccount:prod:zyron-reader",
            "audiences":["zyron","sts"]
          }
        }"#;
        let r = parse_token_review(body).expect("parse");
        assert!(r.authenticated);
        assert_eq!(r.serviceaccount_namespace, "prod");
        assert_eq!(r.serviceaccount_name, "zyron-reader");
        assert_eq!(r.audiences, vec!["zyron", "sts"]);
    }

    #[tokio::test]
    async fn mock_review_success() {
        let server = MockServer::start_async().await;
        let _m = server.mock_async(|when, then| {
            when.method(POST)
                .path("/apis/authentication.k8s.io/v1/tokenreviews");
            then.status(200).body(
                r#"{"status":{"authenticated":true,"username":"system:serviceaccount:ns1:sa1","audiences":["zyron"]}}"#,
            );
        }).await;

        let client = reqwest::Client::new();
        let rev =
            K8sTokenReviewer::with_http_client(server.base_url(), "own-tok".to_string(), client);
        let r = rev
            .review_token("some-token", Some("zyron"))
            .await
            .expect("review");
        assert!(r.authenticated);
        assert_eq!(r.serviceaccount_namespace, "ns1");
    }

    #[tokio::test]
    async fn mock_review_audience_mismatch() {
        let server = MockServer::start_async().await;
        let _m = server.mock_async(|when, then| {
            when.method(POST)
                .path("/apis/authentication.k8s.io/v1/tokenreviews");
            then.status(200).body(
                r#"{"status":{"authenticated":true,"username":"system:serviceaccount:ns1:sa1","audiences":["other"]}}"#,
            );
        }).await;

        let client = reqwest::Client::new();
        let rev =
            K8sTokenReviewer::with_http_client(server.base_url(), "own-tok".to_string(), client);
        assert!(rev.review_token("tok", Some("zyron")).await.is_err());
    }

    #[tokio::test]
    async fn mock_review_not_authenticated() {
        let server = MockServer::start_async().await;
        let _m = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/apis/authentication.k8s.io/v1/tokenreviews");
                then.status(200)
                    .body(r#"{"status":{"authenticated":false}}"#);
            })
            .await;

        let client = reqwest::Client::new();
        let rev =
            K8sTokenReviewer::with_http_client(server.base_url(), "own-tok".to_string(), client);
        assert!(rev.review_token("tok", None).await.is_err());
    }

    #[tokio::test]
    async fn token_provider_reads_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("token");
        let mut f = std::fs::File::create(&path).expect("create");
        f.write_all(b"eyJabc.payload.sig\n").expect("write");
        drop(f);

        let p = K8sTokenProvider::new(path, Some("zyron".to_string()), Duration::from_secs(300));
        let (creds, ttl) = p.fetch().await.expect("fetch");
        assert_eq!(
            creds.get("token").map(|s| s.as_str()),
            Some("eyJabc.payload.sig")
        );
        assert_eq!(creds.get("audience").map(|s| s.as_str()), Some("zyron"));
        assert_eq!(ttl, Duration::from_secs(300));
    }

    #[tokio::test]
    async fn token_provider_rejects_missing() {
        let p = K8sTokenProvider::new(
            PathBuf::from("/nonexistent/zyron/token"),
            None,
            Duration::from_secs(60),
        );
        assert!(p.fetch().await.is_err());
    }
}

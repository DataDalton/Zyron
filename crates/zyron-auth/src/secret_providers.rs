//! Secret-manager-backed credential providers for HashiCorp Vault, GCP Secret
//! Manager, and Azure Key Vault.
//!
//! Each provider implements `CredentialProvider` and fetches the latest secret
//! material on each call. TTL is provider-supplied (Vault leases) or
//! caller-supplied (GCP/Azure) so the credential cache can schedule refresh.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use parking_lot::RwLock;
use zyron_common::{Result, ZyronError};

use crate::credential_provider::CredentialProvider;

// -----------------------------------------------------------------------------
// Vault
// -----------------------------------------------------------------------------

/// Vault KV engine version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VaultEngine {
    Kv1,
    Kv2,
}

/// Vault authentication method. For AppRole and Kubernetes, a login call is
/// made to swap the material for a client token before reading the secret.
#[derive(Debug, Clone)]
pub enum VaultAuth {
    Token(String),
    AppRole { role_id: String, secret_id: String },
    Kubernetes { role: String, jwt_path: PathBuf },
}

#[derive(Debug, Clone)]
struct VaultToken {
    token: String,
    ttl: Duration,
}

/// Fetches a secret from HashiCorp Vault. For KV v2 the caller sets `path` to
/// the mount + data path, for example `secret/data/zyron/prod`. The provider
/// does not transform the path, it forwards it to the configured engine
/// prefix.
pub struct VaultProvider {
    pub vault_url: String,
    pub path: String,
    pub auth_method: VaultAuth,
    pub engine: VaultEngine,
    http_client: reqwest::Client,
    token_cache: RwLock<Option<VaultToken>>,
    default_ttl: Duration,
}

impl VaultProvider {
    pub fn new(
        vault_url: String,
        path: String,
        auth_method: VaultAuth,
        engine: VaultEngine,
        default_ttl: Duration,
    ) -> Result<Self> {
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .map_err(|e| {
                ZyronError::InvalidCredential(format!("Vault client build failed: {}", e))
            })?;
        Ok(Self {
            vault_url,
            path,
            auth_method,
            engine,
            http_client,
            token_cache: RwLock::new(None),
            default_ttl,
        })
    }

    pub fn with_http_client(mut self, client: reqwest::Client) -> Self {
        self.http_client = client;
        self
    }

    async fn acquire_token(&self) -> Result<VaultToken> {
        if let Some(ref t) = *self.token_cache.read() {
            return Ok(t.clone());
        }
        let token = match &self.auth_method {
            VaultAuth::Token(t) => VaultToken {
                token: t.clone(),
                ttl: self.default_ttl,
            },
            VaultAuth::AppRole { role_id, secret_id } => {
                let url = format!(
                    "{}/v1/auth/approle/login",
                    self.vault_url.trim_end_matches('/')
                );
                let body = format!(
                    "{{\"role_id\":\"{}\",\"secret_id\":\"{}\"}}",
                    role_id, secret_id
                );
                let resp = self
                    .http_client
                    .post(url)
                    .header("Content-Type", "application/json")
                    .body(body)
                    .send()
                    .await
                    .map_err(|e| {
                        ZyronError::AuthenticationFailed(format!(
                            "Vault AppRole login failed: {}",
                            e
                        ))
                    })?;
                parse_vault_login(resp).await?
            }
            VaultAuth::Kubernetes { role, jwt_path } => {
                let jwt = tokio::fs::read_to_string(jwt_path).await.map_err(|e| {
                    ZyronError::AuthenticationFailed(format!("Vault K8s JWT read failed: {}", e))
                })?;
                let jwt = jwt.trim().to_string();
                let url = format!(
                    "{}/v1/auth/kubernetes/login",
                    self.vault_url.trim_end_matches('/')
                );
                let body = format!("{{\"role\":\"{}\",\"jwt\":\"{}\"}}", role, jwt);
                let resp = self
                    .http_client
                    .post(url)
                    .header("Content-Type", "application/json")
                    .body(body)
                    .send()
                    .await
                    .map_err(|e| {
                        ZyronError::AuthenticationFailed(format!(
                            "Vault Kubernetes login failed: {}",
                            e
                        ))
                    })?;
                parse_vault_login(resp).await?
            }
        };
        *self.token_cache.write() = Some(token.clone());
        Ok(token)
    }

    fn secret_url(&self) -> String {
        format!(
            "{}/v1/{}",
            self.vault_url.trim_end_matches('/'),
            self.path.trim_start_matches('/')
        )
    }
}

async fn parse_vault_login(resp: reqwest::Response) -> Result<VaultToken> {
    let status = resp.status();
    let text = resp.text().await.map_err(|e| {
        ZyronError::AuthenticationFailed(format!("Vault login body read failed: {}", e))
    })?;
    if !status.is_success() {
        return Err(ZyronError::AuthenticationFailed(format!(
            "Vault login returned {}: {}",
            status.as_u16(),
            text
        )));
    }
    let token = extract_nested_string(&text, "auth", "client_token").ok_or_else(|| {
        ZyronError::AuthenticationFailed("Vault login missing client_token".to_string())
    })?;
    let ttl_secs = extract_nested_number(&text, "auth", "lease_duration").unwrap_or(3600);
    Ok(VaultToken {
        token,
        ttl: Duration::from_secs(ttl_secs.max(1)),
    })
}

#[async_trait]
impl CredentialProvider for VaultProvider {
    async fn fetch(&self) -> Result<(HashMap<String, String>, Duration)> {
        let token = self.acquire_token().await?;
        let resp = self
            .http_client
            .get(self.secret_url())
            .header("X-Vault-Token", &token.token)
            .send()
            .await
            .map_err(|e| ZyronError::AuthenticationFailed(format!("Vault read failed: {}", e)))?;
        let status = resp.status();
        let body = resp.text().await.map_err(|e| {
            ZyronError::AuthenticationFailed(format!("Vault body read failed: {}", e))
        })?;
        if !status.is_success() {
            return Err(ZyronError::AuthenticationFailed(format!(
                "Vault read returned {}: {}",
                status.as_u16(),
                body
            )));
        }
        let map = match self.engine {
            VaultEngine::Kv1 => extract_object_fields(&body, &["data"]),
            VaultEngine::Kv2 => extract_object_fields(&body, &["data", "data"]),
        };
        if map.is_empty() {
            return Err(ZyronError::AuthenticationFailed(
                "Vault response contained no secret data".to_string(),
            ));
        }
        Ok((map, self.default_ttl))
    }

    fn provider_kind(&self) -> &'static str {
        "vault"
    }
}

// -----------------------------------------------------------------------------
// GCP Secret Manager
// -----------------------------------------------------------------------------

/// Fetches a secret version from GCP Secret Manager. The caller provides an
/// access-token provider (typically a service-account flow) that yields a
/// Bearer token under key `token`.
pub struct GcpSecretManagerProvider {
    pub project: String,
    pub secret_name: String,
    pub version: String,
    pub access_token_provider: Arc<dyn CredentialProvider>,
    pub default_ttl: Duration,
    http_client: reqwest::Client,
    base_url: String,
}

impl GcpSecretManagerProvider {
    pub fn new(
        project: String,
        secret_name: String,
        version: String,
        access_token_provider: Arc<dyn CredentialProvider>,
        default_ttl: Duration,
    ) -> Result<Self> {
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .map_err(|e| {
                ZyronError::InvalidCredential(format!("GCP client build failed: {}", e))
            })?;
        Ok(Self {
            project,
            secret_name,
            version,
            access_token_provider,
            default_ttl,
            http_client,
            base_url: "https://secretmanager.googleapis.com".to_string(),
        })
    }

    pub fn with_base_url(mut self, base: String) -> Self {
        self.base_url = base;
        self
    }

    pub fn with_http_client(mut self, client: reqwest::Client) -> Self {
        self.http_client = client;
        self
    }
}

#[async_trait]
impl CredentialProvider for GcpSecretManagerProvider {
    async fn fetch(&self) -> Result<(HashMap<String, String>, Duration)> {
        let (tok, _) = self.access_token_provider.fetch().await?;
        let bearer = tok.get("token").ok_or_else(|| {
            ZyronError::AuthenticationFailed(
                "GCP access token provider returned no token field".to_string(),
            )
        })?;
        let url = format!(
            "{}/v1/projects/{}/secrets/{}/versions/{}:access",
            self.base_url.trim_end_matches('/'),
            self.project,
            self.secret_name,
            self.version
        );
        let resp = self
            .http_client
            .get(url)
            .bearer_auth(bearer)
            .send()
            .await
            .map_err(|e| {
                ZyronError::AuthenticationFailed(format!("GCP secret fetch failed: {}", e))
            })?;
        let status = resp.status();
        let body = resp.text().await.map_err(|e| {
            ZyronError::AuthenticationFailed(format!("GCP body read failed: {}", e))
        })?;
        if !status.is_success() {
            return Err(ZyronError::AuthenticationFailed(format!(
                "GCP secret fetch returned {}: {}",
                status.as_u16(),
                body
            )));
        }
        let data_b64 = extract_nested_string(&body, "payload", "data").ok_or_else(|| {
            ZyronError::AuthenticationFailed("GCP response missing payload.data".to_string())
        })?;
        let decoded = base64_standard_decode(&data_b64)?;
        let mut out = HashMap::new();
        out.insert(
            "secret".to_string(),
            String::from_utf8(decoded).map_err(|_| {
                ZyronError::AuthenticationFailed("GCP secret payload was not UTF-8".to_string())
            })?,
        );
        Ok((out, self.default_ttl))
    }

    fn provider_kind(&self) -> &'static str {
        "gcp_secret_manager"
    }
}

// -----------------------------------------------------------------------------
// Azure Key Vault
// -----------------------------------------------------------------------------

/// Fetches a secret from Azure Key Vault. `vault_url` is the full vault
/// endpoint, for example `https://my-vault.vault.azure.net`.
pub struct AzureKeyVaultProvider {
    pub vault_url: String,
    pub secret_name: String,
    pub access_token_provider: Arc<dyn CredentialProvider>,
    pub default_ttl: Duration,
    http_client: reqwest::Client,
    api_version: String,
}

impl AzureKeyVaultProvider {
    pub fn new(
        vault_url: String,
        secret_name: String,
        access_token_provider: Arc<dyn CredentialProvider>,
        default_ttl: Duration,
    ) -> Result<Self> {
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .map_err(|e| {
                ZyronError::InvalidCredential(format!("Azure client build failed: {}", e))
            })?;
        Ok(Self {
            vault_url,
            secret_name,
            access_token_provider,
            default_ttl,
            http_client,
            api_version: "7.4".to_string(),
        })
    }

    pub fn with_api_version(mut self, v: String) -> Self {
        self.api_version = v;
        self
    }

    pub fn with_http_client(mut self, client: reqwest::Client) -> Self {
        self.http_client = client;
        self
    }
}

#[async_trait]
impl CredentialProvider for AzureKeyVaultProvider {
    async fn fetch(&self) -> Result<(HashMap<String, String>, Duration)> {
        let (tok, _) = self.access_token_provider.fetch().await?;
        let bearer = tok.get("token").ok_or_else(|| {
            ZyronError::AuthenticationFailed(
                "Azure access token provider returned no token field".to_string(),
            )
        })?;
        let url = format!(
            "{}/secrets/{}?api-version={}",
            self.vault_url.trim_end_matches('/'),
            self.secret_name,
            self.api_version
        );
        let resp = self
            .http_client
            .get(url)
            .bearer_auth(bearer)
            .send()
            .await
            .map_err(|e| {
                ZyronError::AuthenticationFailed(format!("Azure secret fetch failed: {}", e))
            })?;
        let status = resp.status();
        let body = resp.text().await.map_err(|e| {
            ZyronError::AuthenticationFailed(format!("Azure body read failed: {}", e))
        })?;
        if !status.is_success() {
            return Err(ZyronError::AuthenticationFailed(format!(
                "Azure secret fetch returned {}: {}",
                status.as_u16(),
                body
            )));
        }
        let val = extract_top_string(&body, "value").ok_or_else(|| {
            ZyronError::AuthenticationFailed("Azure response missing value".to_string())
        })?;
        let mut out = HashMap::new();
        out.insert("secret".to_string(), val);
        Ok((out, self.default_ttl))
    }

    fn provider_kind(&self) -> &'static str {
        "azure_key_vault"
    }
}

// -----------------------------------------------------------------------------
// JSON helpers used across secret providers
// -----------------------------------------------------------------------------

fn extract_top_string(body: &str, key: &str) -> Option<String> {
    let needle = format!("\"{}\"", key);
    let p = body.find(&needle)?;
    let rest = &body[p + needle.len()..];
    let colon = rest.find(':')?;
    let after = rest[colon + 1..].trim_start();
    if !after.starts_with('"') {
        return None;
    }
    let inner = &after[1..];
    let mut out = String::new();
    let bytes = inner.as_bytes();
    let mut i = 0;
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

fn extract_nested_string(body: &str, outer: &str, inner: &str) -> Option<String> {
    // Find outer object, then search for inner key inside it. This is
    // tolerant of additional siblings but does not handle nested braces on
    // the same key name.
    let outer_needle = format!("\"{}\"", outer);
    let p = body.find(&outer_needle)?;
    let rest = &body[p + outer_needle.len()..];
    let colon = rest.find(':')?;
    let obj_start = rest[colon + 1..].trim_start();
    if !obj_start.starts_with('{') {
        return None;
    }
    let bytes = obj_start.as_bytes();
    let mut depth = 0i32;
    let mut end = 0usize;
    for (i, b) in bytes.iter().enumerate() {
        match b {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    end = i;
                    break;
                }
            }
            _ => {}
        }
    }
    if end == 0 {
        return None;
    }
    let slice = &obj_start[1..end];
    extract_top_string(slice, inner)
}

fn extract_nested_number(body: &str, outer: &str, inner: &str) -> Option<u64> {
    let outer_needle = format!("\"{}\"", outer);
    let p = body.find(&outer_needle)?;
    let rest = &body[p + outer_needle.len()..];
    let colon = rest.find(':')?;
    let obj_start = rest[colon + 1..].trim_start();
    if !obj_start.starts_with('{') {
        // Fallback: treat `outer` itself as a number-valued key.
        return None;
    }
    let bytes = obj_start.as_bytes();
    let mut depth = 0i32;
    let mut end = 0usize;
    for (i, b) in bytes.iter().enumerate() {
        match b {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    end = i;
                    break;
                }
            }
            _ => {}
        }
    }
    if end == 0 {
        return None;
    }
    let slice = &obj_start[1..end];
    let needle = format!("\"{}\"", inner);
    let p = slice.find(&needle)?;
    let rest = &slice[p + needle.len()..];
    let colon = rest.find(':')?;
    let after = rest[colon + 1..].trim_start();
    let end = after
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(after.len());
    if end == 0 {
        return None;
    }
    after[..end].parse::<u64>().ok()
}

pub(crate) fn extract_object_fields(body: &str, path: &[&str]) -> HashMap<String, String> {
    // Walk into the JSON object by successive "path" keys, return flat
    // string fields of the target object.
    let mut cursor = body;
    for key in path {
        let needle = format!("\"{}\"", key);
        let Some(p) = cursor.find(&needle) else {
            return HashMap::new();
        };
        let after = &cursor[p + needle.len()..];
        let Some(colon) = after.find(':') else {
            return HashMap::new();
        };
        let rest = after[colon + 1..].trim_start();
        if !rest.starts_with('{') {
            return HashMap::new();
        }
        cursor = rest;
    }
    // Now cursor starts with '{' and we want top-level string fields until
    // the matching close brace.
    let bytes = cursor.as_bytes();
    let mut depth = 0i32;
    let mut end = 0usize;
    for (i, b) in bytes.iter().enumerate() {
        match b {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    end = i;
                    break;
                }
            }
            _ => {}
        }
    }
    if end == 0 {
        return HashMap::new();
    }
    let inner = &cursor[1..end];
    parse_flat_string_fields(inner)
}

fn parse_flat_string_fields(inner: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();
    let bytes = inner.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // Skip whitespace and commas.
        while i < bytes.len() && (bytes[i].is_ascii_whitespace() || bytes[i] == b',') {
            i += 1;
        }
        if i >= bytes.len() || bytes[i] != b'"' {
            i += 1;
            continue;
        }
        // Read key.
        let key_start = i + 1;
        let mut j = key_start;
        while j < bytes.len() && bytes[j] != b'"' {
            if bytes[j] == b'\\' {
                j += 2;
            } else {
                j += 1;
            }
        }
        if j >= bytes.len() {
            break;
        }
        let key = std::str::from_utf8(&bytes[key_start..j])
            .unwrap_or("")
            .to_string();
        i = j + 1;
        // Skip to colon.
        while i < bytes.len() && bytes[i] != b':' {
            i += 1;
        }
        i += 1;
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        // Value may be string, number, bool, null, object, or array.
        if bytes[i] == b'"' {
            let vstart = i + 1;
            let mut k = vstart;
            let mut val = String::new();
            while k < bytes.len() && bytes[k] != b'"' {
                if bytes[k] == b'\\' && k + 1 < bytes.len() {
                    let nxt = bytes[k + 1];
                    match nxt {
                        b'"' => val.push('"'),
                        b'\\' => val.push('\\'),
                        b'/' => val.push('/'),
                        b'n' => val.push('\n'),
                        b'r' => val.push('\r'),
                        b't' => val.push('\t'),
                        _ => val.push(nxt as char),
                    }
                    k += 2;
                } else {
                    val.push(bytes[k] as char);
                    k += 1;
                }
            }
            if k > vstart || bytes.get(k) == Some(&b'"') {
                out.insert(key, val);
            }
            i = k + 1;
        } else if bytes[i] == b'{' || bytes[i] == b'[' {
            // Skip nested object/array.
            let opener = bytes[i];
            let closer = if opener == b'{' { b'}' } else { b']' };
            let mut depth = 1i32;
            i += 1;
            while i < bytes.len() && depth > 0 {
                if bytes[i] == opener {
                    depth += 1;
                } else if bytes[i] == closer {
                    depth -= 1;
                }
                i += 1;
            }
        } else {
            // Number/bool/null, read until comma or end.
            let vstart = i;
            while i < bytes.len() && bytes[i] != b',' && bytes[i] != b'}' {
                i += 1;
            }
            let v = std::str::from_utf8(&bytes[vstart..i]).unwrap_or("").trim();
            if !v.is_empty() {
                out.insert(key, v.to_string());
            }
        }
    }
    out
}

fn base64_standard_decode(s: &str) -> Result<Vec<u8>> {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD
        .decode(s.trim())
        .map_err(|e| ZyronError::AuthenticationFailed(format!("base64 decode failed: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::credential_provider::StaticCredentialProvider;
    use httpmock::prelude::*;

    fn access_token_provider() -> Arc<dyn CredentialProvider> {
        let mut m = HashMap::new();
        m.insert("token".to_string(), "bearer-xyz".to_string());
        Arc::new(StaticCredentialProvider::new(m, Duration::from_secs(120)))
    }

    #[tokio::test]
    async fn vault_kv2_read_with_token_auth() {
        let server = MockServer::start_async().await;
        let _m = server.mock_async(|when, then| {
            when.method(GET).path("/v1/secret/data/zyron/prod");
            then.status(200).body(
                r#"{"request_id":"r","data":{"data":{"password":"hunter2","username":"svc"}},"lease_duration":3600}"#,
            );
        }).await;

        let provider = VaultProvider::new(
            server.base_url(),
            "secret/data/zyron/prod".to_string(),
            VaultAuth::Token("root".to_string()),
            VaultEngine::Kv2,
            Duration::from_secs(600),
        )
        .expect("provider");
        let (creds, ttl) = provider.fetch().await.expect("fetch");
        assert_eq!(creds.get("password").map(|s| s.as_str()), Some("hunter2"));
        assert_eq!(creds.get("username").map(|s| s.as_str()), Some("svc"));
        assert_eq!(ttl, Duration::from_secs(600));
    }

    #[tokio::test]
    async fn vault_kv1_read() {
        let server = MockServer::start_async().await;
        let _m = server
            .mock_async(|when, then| {
                when.method(GET).path("/v1/kv/zyron");
                then.status(200)
                    .body(r#"{"data":{"password":"h","user":"u"}}"#);
            })
            .await;

        let provider = VaultProvider::new(
            server.base_url(),
            "kv/zyron".to_string(),
            VaultAuth::Token("root".to_string()),
            VaultEngine::Kv1,
            Duration::from_secs(60),
        )
        .expect("provider");
        let (creds, _) = provider.fetch().await.expect("fetch");
        assert_eq!(creds.get("password").map(|s| s.as_str()), Some("h"));
    }

    #[tokio::test]
    async fn vault_read_failure() {
        let server = MockServer::start_async().await;
        let _m = server
            .mock_async(|when, then| {
                when.method(GET).path("/v1/secret/data/x");
                then.status(403).body("denied");
            })
            .await;

        let provider = VaultProvider::new(
            server.base_url(),
            "secret/data/x".to_string(),
            VaultAuth::Token("root".to_string()),
            VaultEngine::Kv2,
            Duration::from_secs(60),
        )
        .expect("provider");
        assert!(provider.fetch().await.is_err());
    }

    #[tokio::test]
    async fn gcp_secret_read() {
        let server = MockServer::start_async().await;
        let _m = server
            .mock_async(|when, then| {
                when.method(GET)
                    .path("/v1/projects/p/secrets/n/versions/latest:access");
                then.status(200).body(
                r#"{"name":"projects/p/secrets/n/versions/1","payload":{"data":"aHVudGVyMg=="}}"#,
            );
            })
            .await;

        let provider = GcpSecretManagerProvider::new(
            "p".to_string(),
            "n".to_string(),
            "latest".to_string(),
            access_token_provider(),
            Duration::from_secs(600),
        )
        .expect("provider")
        .with_base_url(server.base_url());
        let (creds, ttl) = provider.fetch().await.expect("fetch");
        assert_eq!(creds.get("secret").map(|s| s.as_str()), Some("hunter2"));
        assert_eq!(ttl, Duration::from_secs(600));
    }

    #[tokio::test]
    async fn azure_secret_read() {
        let server = MockServer::start_async().await;
        let _m = server
            .mock_async(|when, then| {
                when.method(GET)
                    .path("/secrets/mysecret")
                    .query_param("api-version", "7.4");
                then.status(200).body(r#"{"value":"topsecret","id":"..."}"#);
            })
            .await;

        let provider = AzureKeyVaultProvider::new(
            server.base_url(),
            "mysecret".to_string(),
            access_token_provider(),
            Duration::from_secs(300),
        )
        .expect("provider");
        let (creds, ttl) = provider.fetch().await.expect("fetch");
        assert_eq!(creds.get("secret").map(|s| s.as_str()), Some("topsecret"));
        assert_eq!(ttl, Duration::from_secs(300));
    }

    #[test]
    fn parse_nested_fields_vault_kv2() {
        let body = r#"{"data":{"data":{"a":"1","b":"2"}}}"#;
        let map = extract_object_fields(body, &["data", "data"]);
        assert_eq!(map.get("a").map(|s| s.as_str()), Some("1"));
        assert_eq!(map.get("b").map(|s| s.as_str()), Some("2"));
    }
}

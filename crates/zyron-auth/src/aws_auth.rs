//! AWS credential providers backed by the AWS SDKs.
//!
//! Three providers cover the common Zyron patterns: STS AssumeRole (for cross
//! account trust), STS AssumeRoleWithWebIdentity (for EKS IRSA), and Secrets
//! Manager (for credential material stored in AWS). Each implements the
//! `CredentialProvider` trait so they plug into the credential cache like
//! Vault, GCP, and Azure.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use async_trait::async_trait;
use zyron_common::{Result, ZyronError};

use crate::credential_provider::CredentialProvider;

// -----------------------------------------------------------------------------
// AssumeRole
// -----------------------------------------------------------------------------

/// Calls STS AssumeRole using the ambient AWS config chain to authenticate the
/// caller.
pub struct AwsStsAssumeRoleProvider {
    pub role_arn: String,
    pub session_name: String,
    pub external_id: Option<String>,
    pub region: String,
    pub duration: Duration,
}

impl AwsStsAssumeRoleProvider {
    pub fn new(
        role_arn: String,
        session_name: String,
        external_id: Option<String>,
        region: String,
        duration: Duration,
    ) -> Self {
        Self {
            role_arn,
            session_name,
            external_id,
            region,
            duration,
        }
    }
}

#[async_trait]
impl CredentialProvider for AwsStsAssumeRoleProvider {
    async fn fetch(&self) -> Result<(HashMap<String, String>, Duration)> {
        let region = aws_config::Region::new(self.region.clone());
        let cfg = aws_config::defaults(aws_config::BehaviorVersion::latest())
            .region(region)
            .load()
            .await;
        let client = aws_sdk_sts::Client::new(&cfg);
        let mut builder = client
            .assume_role()
            .role_arn(self.role_arn.clone())
            .role_session_name(self.session_name.clone())
            .duration_seconds(self.duration.as_secs().min(43200) as i32);
        if let Some(ref eid) = self.external_id {
            builder = builder.external_id(eid.clone());
        }
        let out = builder.send().await.map_err(|e| {
            ZyronError::AuthenticationFailed(format!("STS AssumeRole failed: {}", e))
        })?;
        convert_sts_credentials(out.credentials(), self.duration)
    }

    fn provider_kind(&self) -> &'static str {
        "aws_sts_assume_role"
    }
}

// -----------------------------------------------------------------------------
// AssumeRoleWithWebIdentity (EKS IRSA path)
// -----------------------------------------------------------------------------

/// Calls STS AssumeRoleWithWebIdentity using a web identity token read from
/// disk. Typically used with Kubernetes service-account projected tokens for
/// the EKS IAM Roles for Service Accounts pattern.
pub struct AwsStsWebIdentityProvider {
    pub role_arn: String,
    pub session_name: String,
    pub web_identity_token_path: PathBuf,
    pub region: String,
    pub duration: Duration,
}

impl AwsStsWebIdentityProvider {
    pub fn new(
        role_arn: String,
        session_name: String,
        web_identity_token_path: PathBuf,
        region: String,
        duration: Duration,
    ) -> Self {
        Self {
            role_arn,
            session_name,
            web_identity_token_path,
            region,
            duration,
        }
    }
}

#[async_trait]
impl CredentialProvider for AwsStsWebIdentityProvider {
    async fn fetch(&self) -> Result<(HashMap<String, String>, Duration)> {
        let token = tokio::fs::read_to_string(&self.web_identity_token_path)
            .await
            .map_err(|e| {
                ZyronError::AuthenticationFailed(format!(
                    "web identity token read failed at {}: {}",
                    self.web_identity_token_path.display(),
                    e
                ))
            })?;
        let token = token.trim().to_string();
        let region = aws_config::Region::new(self.region.clone());
        let cfg = aws_config::defaults(aws_config::BehaviorVersion::latest())
            .region(region)
            .no_credentials()
            .load()
            .await;
        let client = aws_sdk_sts::Client::new(&cfg);
        let out = client
            .assume_role_with_web_identity()
            .role_arn(self.role_arn.clone())
            .role_session_name(self.session_name.clone())
            .web_identity_token(token)
            .duration_seconds(self.duration.as_secs().min(43200) as i32)
            .send()
            .await
            .map_err(|e| {
                ZyronError::AuthenticationFailed(format!(
                    "STS AssumeRoleWithWebIdentity failed: {}",
                    e
                ))
            })?;
        convert_sts_credentials(out.credentials(), self.duration)
    }

    fn provider_kind(&self) -> &'static str {
        "aws_sts_web_identity"
    }
}

// -----------------------------------------------------------------------------
// Secrets Manager
// -----------------------------------------------------------------------------

/// Fetches a secret from AWS Secrets Manager. The returned map has a single
/// `secret` field holding the `SecretString`, or a `secret_binary` field
/// holding a base64-encoded blob if the secret is binary.
pub struct AwsSecretsManagerProvider {
    pub region: String,
    pub secret_id: String,
    pub default_ttl: Duration,
}

impl AwsSecretsManagerProvider {
    pub fn new(region: String, secret_id: String, default_ttl: Duration) -> Self {
        Self {
            region,
            secret_id,
            default_ttl,
        }
    }
}

#[async_trait]
impl CredentialProvider for AwsSecretsManagerProvider {
    async fn fetch(&self) -> Result<(HashMap<String, String>, Duration)> {
        let region = aws_config::Region::new(self.region.clone());
        let cfg = aws_config::defaults(aws_config::BehaviorVersion::latest())
            .region(region)
            .load()
            .await;
        let client = aws_sdk_secretsmanager::Client::new(&cfg);
        let out = client
            .get_secret_value()
            .secret_id(self.secret_id.clone())
            .send()
            .await
            .map_err(|e| {
                ZyronError::AuthenticationFailed(format!(
                    "Secrets Manager GetSecretValue failed: {}",
                    e
                ))
            })?;
        let mut map = HashMap::new();
        if let Some(s) = out.secret_string() {
            map.insert("secret".to_string(), s.to_string());
        } else if let Some(blob) = out.secret_binary() {
            use base64::Engine;
            let b = base64::engine::general_purpose::STANDARD.encode(blob.as_ref());
            map.insert("secret_binary".to_string(), b);
        } else {
            return Err(ZyronError::AuthenticationFailed(
                "Secrets Manager returned no secret material".to_string(),
            ));
        }
        Ok((map, self.default_ttl))
    }

    fn provider_kind(&self) -> &'static str {
        "aws_secrets_manager"
    }
}

// -----------------------------------------------------------------------------
// Helper: convert STS response credentials to the common credential map
// -----------------------------------------------------------------------------

fn convert_sts_credentials(
    creds: Option<&aws_sdk_sts::types::Credentials>,
    fallback_ttl: Duration,
) -> Result<(HashMap<String, String>, Duration)> {
    let c = creds.ok_or_else(|| {
        ZyronError::AuthenticationFailed("STS response missing credentials".to_string())
    })?;
    let mut map = HashMap::new();
    map.insert("access_key_id".to_string(), c.access_key_id().to_string());
    map.insert(
        "secret_access_key".to_string(),
        c.secret_access_key().to_string(),
    );
    map.insert("session_token".to_string(), c.session_token().to_string());
    let ttl = {
        let secs = c.expiration().secs();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        let remaining = (secs - now).max(1);
        Duration::from_secs(remaining as u64)
    };
    let ttl = if ttl.is_zero() { fallback_ttl } else { ttl };
    Ok((map, ttl))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_kinds_are_stable() {
        let a = AwsStsAssumeRoleProvider::new(
            "arn:aws:iam::123:role/r".to_string(),
            "sess".to_string(),
            None,
            "us-east-1".to_string(),
            Duration::from_secs(900),
        );
        assert_eq!(a.provider_kind(), "aws_sts_assume_role");

        let b = AwsStsWebIdentityProvider::new(
            "arn:aws:iam::123:role/r".to_string(),
            "sess".to_string(),
            PathBuf::from("/var/run/secrets/token"),
            "us-east-1".to_string(),
            Duration::from_secs(900),
        );
        assert_eq!(b.provider_kind(), "aws_sts_web_identity");

        let c = AwsSecretsManagerProvider::new(
            "us-east-1".to_string(),
            "my/secret".to_string(),
            Duration::from_secs(600),
        );
        assert_eq!(c.provider_kind(), "aws_secrets_manager");
    }
}

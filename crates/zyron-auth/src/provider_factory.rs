//! Building a credential provider from a stored configuration.
//!
//! An external source names a provider and carries that provider's settings
//! sealed alongside it. This turns those two into a live provider, so the one
//! place that knows which option keys each provider needs is here rather than
//! spread across every caller that opens a source.
//!
//! Option keys are read case insensitively and an unknown key is refused
//! rather than ignored, because a misspelled `secret_id` that silently
//! reaches the provider as a missing value fails later as an authentication
//! error against the remote service, where nothing points back at the typo.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use zyron_common::{Result, ZyronError};

use crate::aws_auth::{AwsSecretsManagerProvider, AwsStsAssumeRoleProvider};
use crate::credential_provider::{CredentialProvider, StaticCredentialProvider};
use crate::k8s_auth::K8sTokenProvider;
use crate::oauth2_client::OAuth2ClientCredentialsProvider;
use crate::secret_providers::{
    AzureKeyVaultProvider, GcpSecretManagerProvider, VaultAuth, VaultEngine, VaultProvider,
};

/// How long a fetched credential is treated as good for when the provider
/// does not report a lease of its own. One hour matches the shortest default
/// lease the supported services issue
pub const DEFAULT_CREDENTIAL_TTL_SECS: u64 = 3_600;

/// The path a Kubernetes service account token is projected to by default
const DEFAULT_K8S_TOKEN_PATH: &str = "/var/run/secrets/kubernetes.io/serviceaccount/token";

/// The provider names the grammar spells, matched to what this builds.
///
/// The catalog stores the kind as a byte and renders it back through this
/// same set of names, so a name here, a name in the statement, and a name in
/// a system view are one set
pub const PROVIDER_NAMES: [&str; 7] = [
    "VAULT",
    "AWS_SECRETS_MANAGER",
    "GCP_SECRET_MANAGER",
    "AZURE_KEY_VAULT",
    "OAUTH2_CLIENT_CREDENTIALS",
    "AWS_IAM_ASSUME_ROLE",
    "K8S_SA_TOKEN",
];

/// One provider's configuration, read case insensitively with every key
/// accounted for
struct Options {
    provider: &'static str,
    values: HashMap<String, String>,
    /// Keys handed out so far, so what is left over can be reported
    taken: Vec<String>,
}

impl Options {
    fn new(provider: &'static str, values: &HashMap<String, String>) -> Self {
        Self {
            provider,
            values: values
                .iter()
                .map(|(k, v)| (k.to_ascii_lowercase(), v.clone()))
                .collect(),
            taken: Vec::new(),
        }
    }

    fn get(&mut self, key: &str) -> Option<String> {
        self.taken.push(key.to_string());
        self.values.get(key).map(|v| v.to_string())
    }

    fn required(&mut self, key: &str) -> Result<String> {
        let provider = self.provider;
        self.get(key).filter(|v| !v.is_empty()).ok_or_else(|| {
            ZyronError::InvalidCredential(format!(
                "credential provider {provider} needs `{key}`, which was not set"
            ))
        })
    }

    /// A duration option in seconds, falling back when it is absent and
    /// refusing a value that is not a number rather than silently defaulting
    fn secs_or(&mut self, key: &str, fallback: u64) -> Result<Duration> {
        let provider = self.provider;
        match self.get(key) {
            None => Ok(Duration::from_secs(fallback)),
            Some(text) if text.is_empty() => Ok(Duration::from_secs(fallback)),
            Some(text) => text.parse::<u64>().map(Duration::from_secs).map_err(|_| {
                ZyronError::InvalidCredential(format!(
                    "credential provider {provider} option `{key}` is `{text}`, \
                     which is not a whole number of seconds"
                ))
            }),
        }
    }

    /// Refuses any option the provider never asked for. Called once every
    /// option a provider understands has been read
    fn finish(self) -> Result<()> {
        let mut unknown: Vec<&str> = self
            .values
            .keys()
            .filter(|k| !self.taken.iter().any(|t| t == *k))
            .map(|k| k.as_str())
            .collect();
        if unknown.is_empty() {
            return Ok(());
        }
        unknown.sort_unstable();
        let mut understood = self.taken.clone();
        understood.sort_unstable();
        understood.dedup();
        Err(ZyronError::InvalidCredential(format!(
            "credential provider {} does not use {}. It reads {}",
            self.provider,
            unknown.join(", "),
            understood.join(", ")
        )))
    }
}

/// Builds a provider from its name and configuration.
///
/// The configuration is the map that was sealed with the source, so it
/// carries the provider's own secrets. Nothing here logs a value
pub fn build_credential_provider(
    provider_name: &str,
    config: &HashMap<String, String>,
) -> Result<Arc<dyn CredentialProvider>> {
    let canonical = PROVIDER_NAMES
        .iter()
        .copied()
        .find(|name| name.eq_ignore_ascii_case(provider_name))
        .ok_or_else(|| {
            ZyronError::InvalidCredential(format!(
                "`{provider_name}` is not a credential provider. The providers are {}",
                PROVIDER_NAMES.join(", ")
            ))
        })?;
    let mut opts = Options::new(canonical, config);

    let provider: Arc<dyn CredentialProvider> = match canonical {
        "VAULT" => {
            let url = opts.required("url")?;
            let path = opts.required("path")?;
            let engine = match opts.get("engine") {
                None => VaultEngine::Kv2,
                Some(named) if named.eq_ignore_ascii_case("kv2") || named.is_empty() => {
                    VaultEngine::Kv2
                }
                Some(named) if named.eq_ignore_ascii_case("kv1") => VaultEngine::Kv1,
                Some(named) => {
                    return Err(ZyronError::InvalidCredential(format!(
                        "credential provider VAULT option `engine` is `{named}`, use kv1 or kv2"
                    )));
                }
            };
            let ttl = opts.secs_or("ttl_seconds", DEFAULT_CREDENTIAL_TTL_SECS)?;
            // Exactly one login method, so a config carrying both a token and
            // an AppRole does not silently use whichever is checked first
            let token = opts.get("token").filter(|v| !v.is_empty());
            let role_id = opts.get("role_id").filter(|v| !v.is_empty());
            let secret_id = opts.get("secret_id").filter(|v| !v.is_empty());
            let role = opts.get("role").filter(|v| !v.is_empty());
            let jwt_path = opts.get("jwt_path").filter(|v| !v.is_empty());
            let auth = match (token, role_id, secret_id, role, jwt_path) {
                (Some(token), None, None, None, None) => VaultAuth::Token(token),
                (None, Some(role_id), Some(secret_id), None, None) => {
                    VaultAuth::AppRole { role_id, secret_id }
                }
                (None, None, None, Some(role), jwt_path) => VaultAuth::Kubernetes {
                    role,
                    jwt_path: PathBuf::from(
                        jwt_path.unwrap_or_else(|| DEFAULT_K8S_TOKEN_PATH.to_string()),
                    ),
                },
                _ => {
                    return Err(ZyronError::InvalidCredential(
                        "credential provider VAULT needs exactly one login method, either \
                         `token`, or `role_id` with `secret_id`, or `role` with an optional \
                         `jwt_path`"
                            .to_string(),
                    ));
                }
            };
            opts.finish()?;
            Arc::new(VaultProvider::new(url, path, auth, engine, ttl)?)
        }

        "AWS_SECRETS_MANAGER" => {
            let region = opts.required("region")?;
            let secret_id = opts.required("secret_id")?;
            let ttl = opts.secs_or("ttl_seconds", DEFAULT_CREDENTIAL_TTL_SECS)?;
            opts.finish()?;
            Arc::new(AwsSecretsManagerProvider::new(region, secret_id, ttl))
        }

        "GCP_SECRET_MANAGER" => {
            let project = opts.required("project")?;
            let secret = opts.required("secret")?;
            let version = opts
                .get("version")
                .filter(|v| !v.is_empty())
                .unwrap_or_else(|| "latest".to_string());
            let ttl = opts.secs_or("ttl_seconds", DEFAULT_CREDENTIAL_TTL_SECS)?;
            let token = access_token_provider(&mut opts, ttl)?;
            opts.finish()?;
            Arc::new(GcpSecretManagerProvider::new(
                project, secret, version, token, ttl,
            )?)
        }

        "AZURE_KEY_VAULT" => {
            let vault_url = opts.required("vault_url")?;
            let secret = opts.required("secret")?;
            let ttl = opts.secs_or("ttl_seconds", DEFAULT_CREDENTIAL_TTL_SECS)?;
            let token = access_token_provider(&mut opts, ttl)?;
            opts.finish()?;
            Arc::new(AzureKeyVaultProvider::new(vault_url, secret, token, ttl)?)
        }

        "OAUTH2_CLIENT_CREDENTIALS" => {
            let token_endpoint = opts.required("token_endpoint")?;
            let client_id = opts.required("client_id")?;
            let client_secret = opts.required("client_secret")?;
            let scope = opts.get("scope").filter(|v| !v.is_empty());
            let audience = opts.get("audience").filter(|v| !v.is_empty());
            opts.finish()?;
            Arc::new(OAuth2ClientCredentialsProvider::new(
                token_endpoint,
                client_id,
                client_secret,
                scope,
                audience,
            )?)
        }

        "AWS_IAM_ASSUME_ROLE" => {
            let role_arn = opts.required("role_arn")?;
            let region = opts.required("region")?;
            let session_name = opts
                .get("session_name")
                .filter(|v| !v.is_empty())
                .unwrap_or_else(|| "zyron".to_string());
            let external_id = opts.get("external_id").filter(|v| !v.is_empty());
            let duration = opts.secs_or("duration_seconds", DEFAULT_CREDENTIAL_TTL_SECS)?;
            opts.finish()?;
            Arc::new(AwsStsAssumeRoleProvider::new(
                role_arn,
                session_name,
                external_id,
                region,
                duration,
            ))
        }

        "K8S_SA_TOKEN" => {
            let token_path = opts
                .get("token_path")
                .filter(|v| !v.is_empty())
                .unwrap_or_else(|| DEFAULT_K8S_TOKEN_PATH.to_string());
            let audience = opts.get("audience").filter(|v| !v.is_empty());
            let refresh = opts.secs_or("refresh_seconds", DEFAULT_CREDENTIAL_TTL_SECS)?;
            opts.finish()?;
            Arc::new(K8sTokenProvider::new(
                PathBuf::from(token_path),
                audience,
                refresh,
            ))
        }

        // PROVIDER_NAMES and this match are the same list, and the lookup
        // above only yields a name from it
        other => {
            return Err(ZyronError::InvalidCredential(format!(
                "credential provider {other} is named but not built"
            )));
        }
    };
    Ok(provider)
}

/// The provider that supplies the bearer token GCP and Azure read their
/// secret with.
///
/// A configuration either carries a token directly, which suits a short-lived
/// token injected by the platform, or names an OAuth2 client the token is
/// minted from, which is what a long-running source needs so the token
/// refreshes without the source being altered
fn access_token_provider(opts: &mut Options, ttl: Duration) -> Result<Arc<dyn CredentialProvider>> {
    let direct = opts.get("access_token").filter(|v| !v.is_empty());
    let endpoint = opts.get("token_endpoint").filter(|v| !v.is_empty());
    let client_id = opts.get("client_id").filter(|v| !v.is_empty());
    let client_secret = opts.get("client_secret").filter(|v| !v.is_empty());
    let scope = opts.get("scope").filter(|v| !v.is_empty());
    let audience = opts.get("audience").filter(|v| !v.is_empty());

    match (direct, endpoint, client_id, client_secret) {
        (Some(token), None, None, None) => {
            let mut creds = HashMap::with_capacity(1);
            creds.insert("access_token".to_string(), token);
            Ok(Arc::new(StaticCredentialProvider::new(creds, ttl)))
        }
        (None, Some(endpoint), Some(client_id), Some(client_secret)) => {
            Ok(Arc::new(OAuth2ClientCredentialsProvider::new(
                endpoint,
                client_id,
                client_secret,
                scope,
                audience,
            )?))
        }
        _ => Err(ZyronError::InvalidCredential(format!(
            "credential provider {} needs an access token, either `access_token` or \
             `token_endpoint` with `client_id` and `client_secret`",
            opts.provider
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn every_named_provider_builds_from_a_minimal_config() {
        let cases: Vec<(&str, HashMap<String, String>)> = vec![
            (
                "VAULT",
                config(&[
                    ("url", "https://vault.internal:8200"),
                    ("path", "secret/data/etl"),
                    ("token", "s.token"),
                ]),
            ),
            (
                "AWS_SECRETS_MANAGER",
                config(&[("region", "us-west-2"), ("secret_id", "prod/etl")]),
            ),
            (
                "GCP_SECRET_MANAGER",
                config(&[
                    ("project", "analytics"),
                    ("secret", "etl"),
                    ("access_token", "ya29.token"),
                ]),
            ),
            (
                "AZURE_KEY_VAULT",
                config(&[
                    ("vault_url", "https://kv.vault.azure.net"),
                    ("secret", "etl"),
                    ("access_token", "eyJ.token"),
                ]),
            ),
            (
                "OAUTH2_CLIENT_CREDENTIALS",
                config(&[
                    ("token_endpoint", "https://idp/token"),
                    ("client_id", "zyron"),
                    ("client_secret", "shh"),
                ]),
            ),
            (
                "AWS_IAM_ASSUME_ROLE",
                config(&[
                    ("role_arn", "arn:aws:iam::1:role/etl"),
                    ("region", "eu-west-1"),
                ]),
            ),
            ("K8S_SA_TOKEN", config(&[])),
        ];
        assert_eq!(
            cases.len(),
            PROVIDER_NAMES.len(),
            "every provider name needs a case here"
        );
        for (name, cfg) in cases {
            let built = build_credential_provider(name, &cfg)
                .unwrap_or_else(|e| panic!("{name} did not build, {e}"));
            assert!(!built.provider_kind().is_empty());
            // The name is matched without regard to case
            build_credential_provider(&name.to_ascii_lowercase(), &cfg)
                .unwrap_or_else(|e| panic!("{name} did not build from a lowercase name, {e}"));
        }
    }

    #[test]
    fn a_missing_required_option_names_the_option() {
        let err = build_credential_provider("AWS_SECRETS_MANAGER", &config(&[("region", "us-1")]))
            .map(|_| ())
            .map(|_| ())
            .expect_err("refused");
        assert!(err.to_string().contains("secret_id"), "{err}");
    }

    /// A misspelled key reaching the provider as a missing value fails later
    /// against the remote service, where nothing points back at the typo
    #[test]
    fn an_option_the_provider_does_not_use_is_refused_with_what_it_does_read() {
        let err = build_credential_provider(
            "AWS_SECRETS_MANAGER",
            &config(&[
                ("region", "us-1"),
                ("secret_id", "prod/etl"),
                ("secrt_arn", "typo"),
            ]),
        )
        .map(|_| ())
        .expect_err("refused");
        let text = err.to_string();
        assert!(text.contains("secrt_arn"), "{text}");
        assert!(text.contains("secret_id"), "{text}");
    }

    #[test]
    fn vault_refuses_two_login_methods_at_once() {
        let err = build_credential_provider(
            "VAULT",
            &config(&[
                ("url", "https://vault:8200"),
                ("path", "secret/data/etl"),
                ("token", "s.token"),
                ("role_id", "r"),
                ("secret_id", "s"),
            ]),
        )
        .map(|_| ())
        .expect_err("refused");
        assert!(
            err.to_string().contains("exactly one login method"),
            "{err}"
        );
    }

    #[test]
    fn vault_kubernetes_login_defaults_the_projected_token_path() {
        build_credential_provider(
            "VAULT",
            &config(&[
                ("url", "https://vault:8200"),
                ("path", "secret/data/etl"),
                ("role", "etl"),
            ]),
        )
        .expect("builds with the default jwt path");
    }

    #[test]
    fn a_ttl_that_is_not_a_number_is_refused_rather_than_defaulted() {
        let err = build_credential_provider(
            "AWS_SECRETS_MANAGER",
            &config(&[
                ("region", "us-1"),
                ("secret_id", "prod/etl"),
                ("ttl_seconds", "an hour"),
            ]),
        )
        .map(|_| ())
        .expect_err("refused");
        assert!(err.to_string().contains("ttl_seconds"), "{err}");
    }

    #[test]
    fn an_unknown_provider_lists_the_ones_that_exist() {
        let err = build_credential_provider("HASHICORP", &config(&[]))
            .map(|_| ())
            .expect_err("refused");
        let text = err.to_string();
        assert!(text.contains("HASHICORP"), "{text}");
        assert!(text.contains("VAULT"), "{text}");
    }

    #[test]
    fn gcp_without_any_access_token_is_refused() {
        let err = build_credential_provider(
            "GCP_SECRET_MANAGER",
            &config(&[("project", "analytics"), ("secret", "etl")]),
        )
        .map(|_| ())
        .expect_err("refused");
        assert!(err.to_string().contains("access token"), "{err}");
    }

    #[test]
    fn gcp_mints_its_access_token_from_an_oauth_client() {
        build_credential_provider(
            "GCP_SECRET_MANAGER",
            &config(&[
                ("project", "analytics"),
                ("secret", "etl"),
                ("token_endpoint", "https://oauth2.googleapis.com/token"),
                ("client_id", "zyron"),
                ("client_secret", "shh"),
            ]),
        )
        .expect("builds with a minted token");
    }
}

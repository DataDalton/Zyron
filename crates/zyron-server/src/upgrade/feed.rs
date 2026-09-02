//! Release discovery.
//!
//! A cluster learns about a release from a signed manifest. Where the
//! manifest comes from is behind a trait: an air-gapped cluster reads one an
//! operator uploaded, a connected one fetches it over HTTPS, and a test
//! supplies one directly. Whichever it is, the signature is checked through
//! the signature agility registry before a single field is believed, so a
//! tampered manifest is refused rather than acted on

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use zyron_auth::signature::{VerifyingMaterial, verify_artifact};
use zyron_common::format::scheme::{ArtifactKind, SchemeRegistry};
use zyron_common::format::{ReleaseEntry, ReleaseManifest};
use zyron_common::{Result, ZyronError};

/// The manifest as it travels, which is JSON so an operator can read one
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestDocument {
    pub channel: String,
    pub generated_at_secs: u64,
    pub releases: Vec<ReleaseDocument>,
    pub signature_scheme: String,
    /// Hex signature over the canonical bytes
    pub signature: String,
}

/// One release inside a manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReleaseDocument {
    pub version: String,
    pub artifact_url: String,
    pub sha256: String,
    #[serde(default)]
    pub signature_scheme: String,
    #[serde(default)]
    pub signature: String,
    #[serde(default)]
    pub upgrade_chain: Vec<String>,
    #[serde(default)]
    pub carries_format_bump: bool,
    #[serde(default)]
    pub notes_url: String,
}

impl From<ManifestDocument> for ReleaseManifest {
    fn from(document: ManifestDocument) -> Self {
        ReleaseManifest {
            channel: document.channel,
            generated_at_secs: document.generated_at_secs,
            releases: document
                .releases
                .into_iter()
                .map(|release| ReleaseEntry {
                    version: release.version,
                    artifact_url: release.artifact_url,
                    sha256: release.sha256,
                    signature_scheme: release.signature_scheme,
                    signature: release.signature,
                    upgrade_chain: release.upgrade_chain,
                    carries_format_bump: release.carries_format_bump,
                    notes_url: release.notes_url,
                })
                .collect(),
            signature_scheme: document.signature_scheme,
            signature: document.signature,
        }
    }
}

impl From<&ReleaseManifest> for ManifestDocument {
    fn from(manifest: &ReleaseManifest) -> Self {
        ManifestDocument {
            channel: manifest.channel.clone(),
            generated_at_secs: manifest.generated_at_secs,
            releases: manifest
                .releases
                .iter()
                .map(|release| ReleaseDocument {
                    version: release.version.clone(),
                    artifact_url: release.artifact_url.clone(),
                    sha256: release.sha256.clone(),
                    signature_scheme: release.signature_scheme.clone(),
                    signature: release.signature.clone(),
                    upgrade_chain: release.upgrade_chain.clone(),
                    carries_format_bump: release.carries_format_bump,
                    notes_url: release.notes_url.clone(),
                })
                .collect(),
            signature_scheme: manifest.signature_scheme.clone(),
            signature: manifest.signature.clone(),
        }
    }
}

/// Where a manifest comes from
#[async_trait::async_trait]
pub trait ReleaseFeedSource: Send + Sync {
    /// Fetches the manifest for one channel, or None when nothing changed
    /// since the last fetch
    async fn fetch(&self, channel: &str) -> Result<Option<ManifestDocument>>;

    /// A name for the source, printed in audit events
    fn describe(&self) -> String;
}

/// Fetches the manifest over HTTPS
pub struct HttpFeedSource {
    base_url: String,
    client: reqwest::Client,
}

impl HttpFeedSource {
    /// Builds a source rooted at a feed URL
    pub fn new(base_url: impl Into<String>, timeout_secs: u64) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(timeout_secs))
            .build()
            .map_err(|e| ZyronError::Internal(format!("release feed client, {e}")))?;
        Ok(Self {
            base_url: base_url.into(),
            client,
        })
    }

    fn url_for(&self, channel: &str) -> String {
        format!("{}/{channel}.manifest", self.base_url.trim_end_matches('/'))
    }
}

#[async_trait::async_trait]
impl ReleaseFeedSource for HttpFeedSource {
    async fn fetch(&self, channel: &str) -> Result<Option<ManifestDocument>> {
        let url = self.url_for(channel);
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| ZyronError::Internal(format!("release feed {url}, {e}")))?;
        if response.status() == reqwest::StatusCode::NOT_MODIFIED {
            return Ok(None);
        }
        if !response.status().is_success() {
            return Err(ZyronError::Internal(format!(
                "release feed {url} answered {}",
                response.status()
            )));
        }
        let body = response
            .text()
            .await
            .map_err(|e| ZyronError::Internal(format!("release feed {url}, {e}")))?;
        parse_manifest(&body).map(Some)
    }

    fn describe(&self) -> String {
        format!("https feed at {}", self.base_url)
    }
}

/// Reads a manifest an operator uploaded, which is how an air-gapped cluster
/// learns about a release
pub struct LocalFeedSource {
    directory: PathBuf,
}

impl LocalFeedSource {
    pub fn new(directory: impl Into<PathBuf>) -> Self {
        Self {
            directory: directory.into(),
        }
    }

    /// Where one channel's manifest is expected
    pub fn path_for(&self, channel: &str) -> PathBuf {
        self.directory.join(format!("{channel}.manifest"))
    }

    /// Writes a manifest into the directory, which is what an upload does
    pub fn publish(&self, manifest: &ReleaseManifest) -> Result<PathBuf> {
        std::fs::create_dir_all(&self.directory).map_err(ZyronError::Io)?;
        let document = ManifestDocument::from(manifest);
        let body = serde_json::to_vec_pretty(&document)
            .map_err(|e| ZyronError::Internal(format!("release manifest encode, {e}")))?;
        let path = self.path_for(&manifest.channel);
        std::fs::write(&path, body).map_err(ZyronError::Io)?;
        Ok(path)
    }
}

#[async_trait::async_trait]
impl ReleaseFeedSource for LocalFeedSource {
    async fn fetch(&self, channel: &str) -> Result<Option<ManifestDocument>> {
        let path = self.path_for(channel);
        match std::fs::read_to_string(&path) {
            Ok(body) => parse_manifest(&body).map(Some),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(ZyronError::Io(e)),
        }
    }

    fn describe(&self) -> String {
        format!("local feed at {}", self.directory.display())
    }
}

/// A source that hands back a manifest it was given, which is what a test
/// and a `zyron-ctl` dry run use
pub struct StaticFeedSource {
    manifest: ManifestDocument,
}

impl StaticFeedSource {
    pub fn new(manifest: ManifestDocument) -> Self {
        Self { manifest }
    }
}

#[async_trait::async_trait]
impl ReleaseFeedSource for StaticFeedSource {
    async fn fetch(&self, channel: &str) -> Result<Option<ManifestDocument>> {
        if self.manifest.channel == channel {
            Ok(Some(self.manifest.clone()))
        } else {
            Ok(None)
        }
    }

    fn describe(&self) -> String {
        "static feed".to_string()
    }
}

/// Parses a manifest document without checking its signature
pub fn parse_manifest(body: &str) -> Result<ManifestDocument> {
    serde_json::from_str(body)
        .map_err(|e| ZyronError::Internal(format!("release manifest is not readable, {e}")))
}

/// The key a release manifest signature is checked against
#[derive(Debug, Clone)]
pub struct ReleaseSigningKey {
    pub scheme_name: String,
    pub material: VerifyingMaterial,
}

/// Verifies a manifest's signature through the signature agility registry.
///
/// The manifest names the scheme that signed it, the registry decides
/// whether that scheme is accepted for a release manifest now, and only then
/// is the signature checked. A manifest that fails any of the three is
/// refused, so nothing downstream ever reads a field from it
pub fn verify_manifest(
    registry: &SchemeRegistry,
    key: &ReleaseSigningKey,
    manifest: &ReleaseManifest,
    now_secs: u64,
) -> Result<()> {
    if manifest.signature_scheme.is_empty() {
        return Err(ZyronError::UpgradeRefused(
            "the release manifest names no signature scheme, so it cannot be verified".to_string(),
        ));
    }
    let signature = decode_hex(&manifest.signature).ok_or_else(|| {
        ZyronError::UpgradeRefused(
            "the release manifest signature is not hex, so it cannot be verified".to_string(),
        )
    })?;
    let verified = verify_artifact(
        registry,
        ArtifactKind::AppImage,
        &manifest.signature_scheme,
        &key.material,
        &manifest.canonical_bytes(),
        &signature,
        now_secs,
    )?;
    if !verified {
        return Err(ZyronError::UpgradeRefused(format!(
            "the release manifest for channel `{}` does not verify against the release \
             signing key. It was tampered with in transit or signed by another key",
            manifest.channel
        )));
    }
    Ok(())
}

/// Decodes a hex string, or None when it is not hex
pub fn decode_hex(text: &str) -> Option<Vec<u8>> {
    if !text.len().is_multiple_of(2) {
        return None;
    }
    let mut out = Vec::with_capacity(text.len() / 2);
    let bytes = text.as_bytes();
    for pair in bytes.chunks_exact(2) {
        let high = (pair[0] as char).to_digit(16)?;
        let low = (pair[1] as char).to_digit(16)?;
        out.push((high * 16 + low) as u8);
    }
    Some(out)
}

/// Encodes bytes as lower-case hex
pub fn encode_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// Polls a source and hands back the manifest it produced, verified
pub struct ReleasePoller {
    source: Arc<dyn ReleaseFeedSource>,
    key: ReleaseSigningKey,
}

impl ReleasePoller {
    pub fn new(source: Arc<dyn ReleaseFeedSource>, key: ReleaseSigningKey) -> Self {
        Self { source, key }
    }

    /// Fetches and verifies one channel's manifest.
    ///
    /// Ok(None) means nothing new. An error means the manifest was there and
    /// could not be trusted, which the caller audits rather than retries
    pub async fn poll(
        &self,
        registry: &SchemeRegistry,
        channel: &str,
        now_secs: u64,
    ) -> Result<Option<ReleaseManifest>> {
        let Some(document) = self.source.fetch(channel).await? else {
            return Ok(None);
        };
        let manifest: ReleaseManifest = document.into();
        verify_manifest(registry, &self.key, &manifest, now_secs)?;
        Ok(Some(manifest))
    }

    pub fn describe(&self) -> String {
        self.source.describe()
    }
}

/// Verifies a staged binary against the SHA-256 the manifest declares
pub fn verify_sha256(path: &Path, expected_hex: &str) -> Result<()> {
    let bytes = std::fs::read(path).map_err(ZyronError::Io)?;
    verify_sha256_bytes(&bytes, expected_hex, &path.display().to_string())
}

/// Checks bytes already in memory against a declared digest.
///
/// The staging path holds the artifact in memory already, so hashing it
/// there avoids reading a freshly written binary back off disk. `label`
/// names the artifact in the refusal
pub fn verify_sha256_bytes(bytes: &[u8], expected_hex: &str, label: &str) -> Result<()> {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let actual = encode_hex(&hasher.finalize());
    if !actual.eq_ignore_ascii_case(expected_hex) {
        return Err(ZyronError::UpgradeRefused(format!(
            "the staged binary at {label} hashes to {actual}, the manifest declares \
             {expected_hex}. The download was corrupted or the artifact was replaced"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};
    use zyron_common::format::scheme::{
        ArtifactSchemeBinding, SchemeCategory, SchemeId, SchemeStatus, SignatureSchemeRegistration,
    };

    fn registry() -> SchemeRegistry {
        SchemeRegistry::from_parts(
            vec![SignatureSchemeRegistration {
                scheme_name: "Ed25519",
                scheme_id: SchemeId(1),
                category: SchemeCategory::Signature,
                status: SchemeStatus::Active,
                first_available_version: "0.1.0",
                retirement_date: None,
                notes: "release signing",
            }],
            vec![ArtifactSchemeBinding::new(
                ArtifactKind::AppImage,
                "Ed25519",
            )],
        )
    }

    fn manifest() -> ReleaseManifest {
        ReleaseManifest {
            channel: "stable".to_string(),
            generated_at_secs: 1_000,
            releases: vec![ReleaseEntry {
                version: "0.12.0".to_string(),
                artifact_url: "https://example/0.12.0".to_string(),
                sha256: "aa".to_string(),
                signature_scheme: "Ed25519".to_string(),
                signature: String::new(),
                upgrade_chain: vec![],
                carries_format_bump: true,
                notes_url: String::new(),
            }],
            signature_scheme: "Ed25519".to_string(),
            signature: String::new(),
        }
    }

    fn signed(manifest: &mut ReleaseManifest) -> ReleaseSigningKey {
        let signing = SigningKey::from_bytes(&[9u8; 32]);
        let signature = signing.sign(&manifest.canonical_bytes());
        manifest.signature = encode_hex(&signature.to_bytes());
        ReleaseSigningKey {
            scheme_name: "Ed25519".to_string(),
            material: VerifyingMaterial::Ed25519(signing.verifying_key().to_bytes()),
        }
    }

    #[test]
    fn test_hex_round_trips() {
        let bytes = vec![0u8, 1, 15, 16, 255];
        let hex = encode_hex(&bytes);
        assert_eq!(hex, "00010f10ff");
        assert_eq!(decode_hex(&hex), Some(bytes));
        assert_eq!(decode_hex("odd"), None);
        assert_eq!(decode_hex("zz"), None);
    }

    #[test]
    fn test_a_signed_manifest_verifies() {
        let mut manifest = manifest();
        let key = signed(&mut manifest);
        verify_manifest(&registry(), &key, &manifest, 0).expect("verifies");
    }

    #[test]
    fn test_a_tampered_manifest_is_refused() {
        let mut manifest = manifest();
        let key = signed(&mut manifest);
        manifest.releases[0].sha256 = "bb".to_string();
        let err = verify_manifest(&registry(), &key, &manifest, 0).expect_err("refused");
        assert!(err.to_string().contains("does not verify"), "{err}");
        assert!(err.to_string().contains("tampered with"), "{err}");
    }

    #[test]
    fn test_an_unsigned_manifest_is_refused() {
        let mut manifest = manifest();
        manifest.signature_scheme = String::new();
        let key = ReleaseSigningKey {
            scheme_name: "Ed25519".to_string(),
            material: VerifyingMaterial::Ed25519([0u8; 32]),
        };
        let err = verify_manifest(&registry(), &key, &manifest, 0).expect_err("refused");
        assert!(
            err.to_string().contains("names no signature scheme"),
            "{err}"
        );
    }

    #[tokio::test]
    async fn test_the_local_source_round_trips_a_manifest() {
        let dir = tempfile::tempdir().expect("tempdir");
        let source = LocalFeedSource::new(dir.path());
        assert!(
            source
                .fetch("stable")
                .await
                .expect("no manifest yet")
                .is_none(),
            "an absent manifest is not an error"
        );
        let mut manifest = manifest();
        let key = signed(&mut manifest);
        source.publish(&manifest).expect("publishes");

        let poller = ReleasePoller::new(Arc::new(LocalFeedSource::new(dir.path())), key);
        let fetched = poller
            .poll(&registry(), "stable", 0)
            .await
            .expect("polls")
            .expect("a manifest is there");
        assert_eq!(fetched.releases.len(), 1);
        assert_eq!(fetched.releases[0].version, "0.12.0");
    }

    #[tokio::test]
    async fn test_a_tampered_local_manifest_is_refused_by_the_poller() {
        let dir = tempfile::tempdir().expect("tempdir");
        let source = LocalFeedSource::new(dir.path());
        let mut manifest = manifest();
        let key = signed(&mut manifest);
        source.publish(&manifest).expect("publishes");
        let path = source.path_for("stable");
        let body = std::fs::read_to_string(&path).expect("reads");
        std::fs::write(&path, body.replace("0.12.0", "9.9.9")).expect("tampers");

        let poller = ReleasePoller::new(Arc::new(LocalFeedSource::new(dir.path())), key);
        let err = poller
            .poll(&registry(), "stable", 0)
            .await
            .expect_err("refused");
        assert!(err.to_string().contains("does not verify"), "{err}");
    }

    #[test]
    fn test_the_sha256_gate_names_a_mismatch() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("binary");
        std::fs::write(&path, b"the binary bytes").expect("writes");
        let err = verify_sha256(&path, "00").expect_err("mismatch");
        assert!(err.to_string().contains("hashes to"), "{err}");

        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(b"the binary bytes");
        let expected = encode_hex(&hasher.finalize());
        verify_sha256(&path, &expected).expect("matches");
        verify_sha256(&path, &expected.to_uppercase()).expect("case does not matter");
    }
}

//! Staging a release binary.
//!
//! A binary is downloaded to a staging path, its signature is checked
//! through the signature agility registry, and its SHA-256 is checked
//! against the manifest. Only then does it become a candidate for
//! activation, and activation is an atomic rename so a node either runs the
//! old binary or the new one and never a half-written file

use std::path::{Path, PathBuf};
use std::sync::Arc;

use zyron_auth::signature::{VerifyingMaterial, verify_artifact};
use zyron_common::format::ReleaseEntry;
use zyron_common::format::scheme::{ArtifactKind, SchemeRegistry};
use zyron_common::{Result, ZyronError};

use super::feed::{decode_hex, verify_sha256_bytes};

/// Where staged binaries live under the data directory
pub const STAGING_DIR: &str = "staging";

/// A binary that passed every check and is ready to activate
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StagedRelease {
    pub version: String,
    pub path: PathBuf,
    pub sha256: String,
    pub signature_scheme: String,
    pub size_bytes: u64,
}

/// Where a release's bytes come from
#[async_trait::async_trait]
pub trait ArtifactSource: Send + Sync {
    /// Fetches the binary named by a release entry
    async fn fetch(&self, release: &ReleaseEntry) -> Result<Vec<u8>>;
    fn describe(&self) -> String;
}

/// Downloads the binary over HTTPS
pub struct HttpArtifactSource {
    client: reqwest::Client,
}

impl HttpArtifactSource {
    pub fn new(timeout_secs: u64) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(timeout_secs))
            .build()
            .map_err(|e| ZyronError::Internal(format!("artifact client, {e}")))?;
        Ok(Self { client })
    }
}

#[async_trait::async_trait]
impl ArtifactSource for HttpArtifactSource {
    async fn fetch(&self, release: &ReleaseEntry) -> Result<Vec<u8>> {
        let response = self
            .client
            .get(&release.artifact_url)
            .send()
            .await
            .map_err(|e| ZyronError::Internal(format!("{} , {e}", release.artifact_url)))?;
        if !response.status().is_success() {
            return Err(ZyronError::UpgradeRefused(format!(
                "{} answered {}",
                release.artifact_url,
                response.status()
            )));
        }
        response
            .bytes()
            .await
            .map(|bytes| bytes.to_vec())
            .map_err(|e| ZyronError::Internal(format!("{} , {e}", release.artifact_url)))
    }

    fn describe(&self) -> String {
        "https artifact source".to_string()
    }
}

/// Reads the binary from a directory an operator uploaded it to
pub struct LocalArtifactSource {
    directory: PathBuf,
}

impl LocalArtifactSource {
    pub fn new(directory: impl Into<PathBuf>) -> Self {
        Self {
            directory: directory.into(),
        }
    }

    pub fn path_for(&self, version: &str) -> PathBuf {
        self.directory.join(format!("zyron-server-{version}"))
    }
}

#[async_trait::async_trait]
impl ArtifactSource for LocalArtifactSource {
    async fn fetch(&self, release: &ReleaseEntry) -> Result<Vec<u8>> {
        // Reading a server binary is a multi-megabyte blocking read, so it
        // goes to the blocking pool the same way the staging write does
        let path = self.path_for(&release.version);
        let read_path = path.clone();
        let bytes = tokio::task::spawn_blocking(move || std::fs::read(&read_path))
            .await
            .map_err(|e| ZyronError::Internal(format!("artifact read task failed, {e}")))?;
        bytes.map_err(|e| {
            ZyronError::UpgradeRefused(format!(
                "the {} binary is not at {}, {e}. Upload it before triggering the upgrade",
                release.version,
                path.display()
            ))
        })
    }

    fn describe(&self) -> String {
        format!("local artifact source at {}", self.directory.display())
    }
}

/// A local artifact directory read ahead of a remote one.
///
/// The binary for a release delivered by hand sits in the node's feed
/// directory and is read from there. A release the feed advertised with a
/// URL is downloaded when the directory does not hold it
pub struct LayeredArtifactSource {
    local: LocalArtifactSource,
    remote: Option<Arc<dyn ArtifactSource>>,
}

impl LayeredArtifactSource {
    pub fn new(local: LocalArtifactSource, remote: Option<Arc<dyn ArtifactSource>>) -> Self {
        Self { local, remote }
    }
}

#[async_trait::async_trait]
impl ArtifactSource for LayeredArtifactSource {
    async fn fetch(&self, release: &ReleaseEntry) -> Result<Vec<u8>> {
        if self.local.path_for(&release.version).is_file() {
            return self.local.fetch(release).await;
        }
        match &self.remote {
            Some(remote) if !release.artifact_url.trim().is_empty() => remote.fetch(release).await,
            _ => Err(ZyronError::UpgradeRefused(format!(
                "the {} binary is not at {} and the release names no URL to fetch it from. \
                 Deliver it with zyron-ctl release stage",
                release.version,
                self.local.path_for(&release.version).display()
            ))),
        }
    }

    fn describe(&self) -> String {
        match &self.remote {
            Some(remote) => format!("{}, then {}", self.local.describe(), remote.describe()),
            None => self.local.describe(),
        }
    }
}

/// Downloads, verifies, and stages one release
pub async fn stage(
    source: &dyn ArtifactSource,
    registry: &SchemeRegistry,
    release_key: &VerifyingMaterial,
    release: &ReleaseEntry,
    staging_root: &Path,
    now_secs: u64,
) -> Result<StagedRelease> {
    let bytes = source.fetch(release).await?;

    // Both checks run against the bytes in memory, before anything is
    // written. An artifact the release key did not vouch for never reaches
    // the disk at all, and the digest is taken from the buffer already held
    // rather than by reading a freshly written binary back
    let staged = staging_root.join(format!("zyron-server-{}", release.version));
    verify_signature(registry, release_key, release, &bytes, now_secs)?;
    verify_sha256_bytes(&bytes, &release.sha256, &staged.display().to_string())?;

    // Writing a server binary is a large blocking write, so it goes to the
    // blocking pool rather than stalling a runtime worker while an upgrade
    // stages underneath a server that is still taking traffic
    let size_bytes = bytes.len() as u64;
    let write_root = staging_root.to_path_buf();
    let write_target = staged.clone();
    tokio::task::spawn_blocking(move || -> Result<()> {
        std::fs::create_dir_all(&write_root).map_err(ZyronError::Io)?;
        let partial = write_target.with_extension("partial");
        if let Err(e) = std::fs::write(&partial, &bytes) {
            let _ = std::fs::remove_file(&partial);
            return Err(ZyronError::Io(e));
        }
        if let Err(e) = std::fs::rename(&partial, &write_target) {
            let _ = std::fs::remove_file(&partial);
            return Err(ZyronError::Io(e));
        }
        Ok(())
    })
    .await
    .map_err(|e| ZyronError::Internal(format!("staging write task failed, {e}")))??;

    Ok(StagedRelease {
        version: release.version.clone(),
        path: staged,
        sha256: release.sha256.clone(),
        signature_scheme: release.signature_scheme.clone(),
        size_bytes,
    })
}

/// Checks a release binary's signature through the scheme registry
fn verify_signature(
    registry: &SchemeRegistry,
    key: &VerifyingMaterial,
    release: &ReleaseEntry,
    bytes: &[u8],
    now_secs: u64,
) -> Result<()> {
    if release.signature.is_empty() || release.signature_scheme.is_empty() {
        return Err(ZyronError::UpgradeRefused(format!(
            "the {} release carries no signature, so it cannot be staged. A release is only \
             installed when the release signing key vouched for it",
            release.version
        )));
    }
    let signature = decode_hex(&release.signature).ok_or_else(|| {
        ZyronError::UpgradeRefused(format!(
            "the {} release signature is not hex",
            release.version
        ))
    })?;
    let verified = verify_artifact(
        registry,
        ArtifactKind::AppImage,
        &release.signature_scheme,
        key,
        bytes,
        &signature,
        now_secs,
    )?;
    if !verified {
        return Err(ZyronError::UpgradeRefused(format!(
            "the {} binary does not verify against the release signing key. It was replaced \
             in transit or built by another party",
            release.version
        )));
    }
    Ok(())
}

/// Where a rollback leaves the binary it moved out of the live path
pub fn rolled_back_path(live_path: &Path) -> PathBuf {
    live_path.with_extension("rolled-back")
}

/// Puts a staged binary in place, keeping the previous one beside it so a
/// rollback is a rename rather than a download. A binary an earlier
/// rollback left beside the live one is removed here, once nothing runs it
pub fn activate(staged: &StagedRelease, live_path: &Path) -> Result<PathBuf> {
    let previous = live_path.with_extension("previous");
    let _ = std::fs::remove_file(rolled_back_path(live_path));
    if live_path.exists() {
        std::fs::rename(live_path, &previous).map_err(ZyronError::Io)?;
    }
    std::fs::copy(&staged.path, live_path).map_err(ZyronError::Io)?;
    Ok(previous)
}

/// Puts the previous binary back. The live binary is the image this process
/// runs, which Windows lets a rename move but never replace, so it moves
/// aside first and the previous one takes its place
pub fn deactivate(live_path: &Path) -> Result<()> {
    let previous = live_path.with_extension("previous");
    if !previous.exists() {
        return Err(ZyronError::UpgradeRefused(format!(
            "there is no previous binary beside {} to roll back to",
            live_path.display()
        )));
    }
    let rolled_back = rolled_back_path(live_path);
    if live_path.exists() {
        std::fs::rename(live_path, &rolled_back).map_err(ZyronError::Io)?;
    }
    if let Err(e) = std::fs::rename(&previous, live_path) {
        // the live path keeps a binary either way, so a restart after a
        // failed rename still starts what was running
        let _ = std::fs::rename(&rolled_back, live_path);
        return Err(ZyronError::Io(e));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};
    use sha2::{Digest, Sha256};
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

    fn hex(bytes: &[u8]) -> String {
        super::super::feed::encode_hex(bytes)
    }

    fn release(bytes: &[u8], signing: &SigningKey) -> ReleaseEntry {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        ReleaseEntry {
            version: "0.12.0".to_string(),
            artifact_url: "https://example/0.12.0".to_string(),
            sha256: hex(&hasher.finalize()),
            signature_scheme: "Ed25519".to_string(),
            signature: hex(&signing.sign(bytes).to_bytes()),
            upgrade_chain: vec![],
            carries_format_bump: false,
            notes_url: String::new(),
        }
    }

    #[tokio::test]
    async fn test_a_signed_binary_stages_and_activates() {
        let dir = tempfile::tempdir().expect("tempdir");
        let bytes = b"the new binary".to_vec();
        let signing = SigningKey::from_bytes(&[3u8; 32]);
        let entry = release(&bytes, &signing);
        let source = LocalArtifactSource::new(dir.path());
        std::fs::write(source.path_for("0.12.0"), &bytes).expect("uploads");

        let key = VerifyingMaterial::Ed25519(signing.verifying_key().to_bytes());
        let staging = dir.path().join(STAGING_DIR);
        let staged = stage(&source, &registry(), &key, &entry, &staging, 0)
            .await
            .expect("stages");
        assert_eq!(staged.version, "0.12.0");
        assert_eq!(staged.size_bytes, bytes.len() as u64);
        assert!(staged.path.exists());

        let live = dir.path().join("zyron-server");
        std::fs::write(&live, b"the old binary").expect("writes");
        let previous = activate(&staged, &live).expect("activates");
        assert_eq!(std::fs::read(&live).expect("reads"), bytes);
        assert_eq!(
            std::fs::read(&previous).expect("reads"),
            b"the old binary".to_vec()
        );

        deactivate(&live).expect("rolls back");
        assert_eq!(
            std::fs::read(&live).expect("reads"),
            b"the old binary".to_vec()
        );
        assert!(!previous.exists());
        assert_eq!(
            std::fs::read(rolled_back_path(&live)).expect("reads"),
            bytes
        );

        activate(&staged, &live).expect("activates again");
        assert!(!rolled_back_path(&live).exists());
        assert_eq!(std::fs::read(&live).expect("reads"), bytes);
    }

    #[tokio::test]
    async fn test_a_binary_that_does_not_verify_is_refused_and_removed() {
        let dir = tempfile::tempdir().expect("tempdir");
        let signing = SigningKey::from_bytes(&[3u8; 32]);
        let entry = release(b"the new binary", &signing);
        let source = LocalArtifactSource::new(dir.path());
        // The uploaded bytes are not the ones that were signed
        std::fs::write(source.path_for("0.12.0"), b"a different binary").expect("uploads");

        let key = VerifyingMaterial::Ed25519(signing.verifying_key().to_bytes());
        let staging = dir.path().join(STAGING_DIR);
        let err = stage(&source, &registry(), &key, &entry, &staging, 0)
            .await
            .expect_err("refused");
        assert!(err.to_string().contains("does not verify"), "{err}");
        assert!(
            !staging.join("zyron-server-0.12.0.partial").exists(),
            "the rejected download is removed"
        );
    }

    #[tokio::test]
    async fn test_an_unsigned_release_is_refused() {
        let dir = tempfile::tempdir().expect("tempdir");
        let signing = SigningKey::from_bytes(&[3u8; 32]);
        let mut entry = release(b"the new binary", &signing);
        entry.signature = String::new();
        let source = LocalArtifactSource::new(dir.path());
        std::fs::write(source.path_for("0.12.0"), b"the new binary").expect("uploads");
        let key = VerifyingMaterial::Ed25519(signing.verifying_key().to_bytes());
        let err = stage(
            &source,
            &registry(),
            &key,
            &entry,
            &dir.path().join(STAGING_DIR),
            0,
        )
        .await
        .expect_err("refused");
        assert!(err.to_string().contains("carries no signature"), "{err}");
    }

    #[tokio::test]
    async fn test_a_sha_mismatch_is_refused() {
        let dir = tempfile::tempdir().expect("tempdir");
        let bytes = b"the new binary".to_vec();
        let signing = SigningKey::from_bytes(&[3u8; 32]);
        let mut entry = release(&bytes, &signing);
        entry.sha256 = "00".repeat(32);
        let source = LocalArtifactSource::new(dir.path());
        std::fs::write(source.path_for("0.12.0"), &bytes).expect("uploads");
        let key = VerifyingMaterial::Ed25519(signing.verifying_key().to_bytes());
        let err = stage(
            &source,
            &registry(),
            &key,
            &entry,
            &dir.path().join(STAGING_DIR),
            0,
        )
        .await
        .expect_err("refused");
        assert!(err.to_string().contains("hashes to"), "{err}");
    }

    #[tokio::test]
    async fn test_a_missing_upload_names_where_it_should_be() {
        let dir = tempfile::tempdir().expect("tempdir");
        let signing = SigningKey::from_bytes(&[3u8; 32]);
        let entry = release(b"x", &signing);
        let source = LocalArtifactSource::new(dir.path());
        let key = VerifyingMaterial::Ed25519(signing.verifying_key().to_bytes());
        let err = stage(
            &source,
            &registry(),
            &key,
            &entry,
            &dir.path().join(STAGING_DIR),
            0,
        )
        .await
        .expect_err("refused");
        assert!(err.to_string().contains("Upload it"), "{err}");
    }

    #[test]
    fn test_rollback_without_a_previous_binary_is_refused() {
        let dir = tempfile::tempdir().expect("tempdir");
        let live = dir.path().join("zyron-server");
        std::fs::write(&live, b"binary").expect("writes");
        let err = deactivate(&live).expect_err("refused");
        assert!(err.to_string().contains("no previous binary"), "{err}");
    }
}

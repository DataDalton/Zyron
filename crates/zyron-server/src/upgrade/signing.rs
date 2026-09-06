//! Signing a release, the vendor's half of what a node verifies.
//!
//! A release is a binary and a manifest entry naming it. The binary is
//! signed as its raw bytes and the manifest as its canonical bytes, both
//! with the same Ed25519 key, and both signatures travel in the manifest
//! as hex. `zyron-ctl release keygen` draws the key and `zyron-ctl release
//! sign` produces the manifest, which is what the stager on every node
//! checks against the public half built into its binary

use std::path::Path;

use ed25519_dalek::{Signer, SigningKey};
use zyron_auth::signature::VerifyingMaterial;
use zyron_common::format::{ReleaseEntry, ReleaseManifest};
use zyron_common::{Result, ZyronError};

use super::feed::{decode_hex, encode_hex, sha256_hex};
use super::release_key::BUILT_IN_SCHEME;

/// The private half of a release key
pub struct ReleaseSigningSeed([u8; 32]);

impl std::fmt::Debug for ReleaseSigningSeed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ReleaseSigningSeed(..)")
    }
}

impl ReleaseSigningSeed {
    /// Draws a new key from the operating system's randomness
    pub fn generate() -> Self {
        Self(rand::random())
    }

    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Reads a seed written by `to_hex`
    pub fn from_hex(text: &str) -> Result<Self> {
        let bytes = decode_hex(text.trim()).ok_or_else(|| {
            ZyronError::UpgradeRefused("the release signing seed is not hex".to_string())
        })?;
        let seed: [u8; 32] = bytes.as_slice().try_into().map_err(|_| {
            ZyronError::UpgradeRefused(format!(
                "the release signing seed is {} bytes, an Ed25519 seed is 32",
                bytes.len()
            ))
        })?;
        Ok(Self(seed))
    }

    pub fn to_hex(&self) -> String {
        encode_hex(&self.0)
    }

    fn signing_key(&self) -> SigningKey {
        SigningKey::from_bytes(&self.0)
    }

    /// The public half as hex, what goes into `release-signing.pub` or
    /// `upgrade.release_signing_key`
    pub fn verifying_hex(&self) -> String {
        encode_hex(&self.signing_key().verifying_key().to_bytes())
    }

    pub fn verifying_material(&self) -> VerifyingMaterial {
        VerifyingMaterial::Ed25519(self.signing_key().verifying_key().to_bytes())
    }

    /// Signs bytes, returning the signature as hex
    pub fn sign_hex(&self, message: &[u8]) -> String {
        encode_hex(&self.signing_key().sign(message).to_bytes())
    }

    /// Reads a seed file, the one `write_to` wrote
    pub fn read_from(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|e| {
            ZyronError::UpgradeRefused(format!(
                "the release signing key at {} could not be read, {e}",
                path.display()
            ))
        })?;
        Self::from_hex(&text)
    }

    /// Writes the seed as hex, readable by its owner alone where the
    /// platform can say so, and refuses to replace a key that is there
    pub fn write_to(&self, path: &Path) -> Result<()> {
        if path.exists() {
            return Err(ZyronError::UpgradeRefused(format!(
                "{} already holds a key. Move it aside before drawing another, a key that is \
                 replaced by accident cannot sign the next release",
                path.display()
            )));
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(ZyronError::Io)?;
        }
        std::fs::write(path, format!("{}\n", self.to_hex())).map_err(ZyronError::Io)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
                .map_err(ZyronError::Io)?;
        }
        Ok(())
    }
}

/// What one release needs to be signed into a manifest
#[derive(Debug, Clone)]
pub struct ReleaseToSign<'a> {
    pub version: String,
    /// Where nodes fetch the binary. Empty for a release delivered by hand
    /// into a node's feed directory
    pub artifact_url: String,
    pub notes_url: String,
    /// Versions to pass through to reach this one, oldest first
    pub upgrade_chain: Vec<String>,
    pub carries_format_bump: bool,
    pub binary: &'a [u8],
}

/// Signs the binary into a release entry and puts it in the manifest,
/// replacing an entry of the same version, then signs the manifest
pub fn sign_release(
    seed: &ReleaseSigningSeed,
    manifest: &mut ReleaseManifest,
    release: ReleaseToSign<'_>,
    now_secs: u64,
) -> Result<ReleaseEntry> {
    if zyron_common::format::BinaryVersion::parse(&release.version).is_none() {
        return Err(ZyronError::UpgradeRefused(format!(
            "`{}` is not a major.minor.patch version",
            release.version
        )));
    }
    let entry = ReleaseEntry {
        version: release.version.clone(),
        artifact_url: release.artifact_url,
        sha256: sha256_hex(release.binary),
        signature_scheme: BUILT_IN_SCHEME.to_string(),
        signature: seed.sign_hex(release.binary),
        upgrade_chain: release.upgrade_chain,
        carries_format_bump: release.carries_format_bump,
        notes_url: release.notes_url,
    };
    manifest.releases.retain(|r| r.version != entry.version);
    manifest.releases.push(entry.clone());
    manifest.releases.sort_by(|a, b| {
        let av = zyron_common::format::BinaryVersion::parse(&a.version);
        let bv = zyron_common::format::BinaryVersion::parse(&b.version);
        av.cmp(&bv)
    });
    sign_manifest(seed, manifest, now_secs);
    Ok(entry)
}

/// Signs a manifest's canonical bytes, stamping when it was generated
pub fn sign_manifest(seed: &ReleaseSigningSeed, manifest: &mut ReleaseManifest, now_secs: u64) {
    manifest.generated_at_secs = now_secs;
    manifest.signature_scheme = BUILT_IN_SCHEME.to_string();
    manifest.signature = seed.sign_hex(&manifest.canonical_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::upgrade::feed::{ReleaseSigningKey, verify_manifest};
    use crate::upgrade::stager::{LocalArtifactSource, stage};

    #[test]
    fn test_a_signed_release_verifies_and_stages_against_the_public_half() {
        let seed = ReleaseSigningSeed::generate();
        let dir = tempfile::tempdir().expect("tempdir");
        let binary = b"a release binary".to_vec();
        std::fs::write(dir.path().join("zyron-server-0.13.0"), &binary).expect("writes");

        let mut manifest = ReleaseManifest {
            channel: "stable".to_string(),
            generated_at_secs: 0,
            releases: Vec::new(),
            signature_scheme: String::new(),
            signature: String::new(),
        };
        let entry = sign_release(
            &seed,
            &mut manifest,
            ReleaseToSign {
                version: "0.13.0".to_string(),
                artifact_url: String::new(),
                notes_url: String::new(),
                upgrade_chain: vec!["0.13.0".to_string()],
                carries_format_bump: false,
                binary: &binary,
            },
            1_000,
        )
        .expect("signs");
        assert_eq!(entry.sha256, sha256_hex(&binary));
        assert_eq!(manifest.generated_at_secs, 1_000);

        let substrate = zyron_common::format::substrate().expect("registers every scheme");
        let key = ReleaseSigningKey {
            scheme_name: BUILT_IN_SCHEME.to_string(),
            material: seed.verifying_material(),
        };
        verify_manifest(&substrate.schemes, &key, &manifest, 1_000).expect("the manifest verifies");

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");
        let staged = runtime
            .block_on(stage(
                &LocalArtifactSource::new(dir.path()),
                &substrate.schemes,
                &seed.verifying_material(),
                &entry,
                &dir.path().join("staging"),
                1_000,
            ))
            .expect("the binary verifies and stages");
        assert_eq!(staged.version, "0.13.0");

        // Another key does not vouch for it
        let other = ReleaseSigningSeed::generate();
        let wrong = ReleaseSigningKey {
            scheme_name: BUILT_IN_SCHEME.to_string(),
            material: other.verifying_material(),
        };
        assert!(verify_manifest(&substrate.schemes, &wrong, &manifest, 1_000).is_err());
    }

    #[test]
    fn test_signing_again_replaces_the_entry_and_keeps_versions_in_order() {
        let seed = ReleaseSigningSeed::generate();
        let mut manifest = ReleaseManifest {
            channel: "beta".to_string(),
            generated_at_secs: 0,
            releases: Vec::new(),
            signature_scheme: String::new(),
            signature: String::new(),
        };
        for (version, body) in [
            ("0.14.0", "later"),
            ("0.13.0", "earlier"),
            ("0.14.0", "again"),
        ] {
            sign_release(
                &seed,
                &mut manifest,
                ReleaseToSign {
                    version: version.to_string(),
                    artifact_url: String::new(),
                    notes_url: String::new(),
                    upgrade_chain: Vec::new(),
                    carries_format_bump: false,
                    binary: body.as_bytes(),
                },
                5,
            )
            .expect("signs");
        }
        let versions: Vec<&str> = manifest
            .releases
            .iter()
            .map(|r| r.version.as_str())
            .collect();
        assert_eq!(versions, vec!["0.13.0", "0.14.0"]);
        assert_eq!(manifest.releases[1].sha256, sha256_hex(b"again"));
        let err = sign_release(
            &seed,
            &mut manifest,
            ReleaseToSign {
                version: "latest".to_string(),
                artifact_url: String::new(),
                notes_url: String::new(),
                upgrade_chain: Vec::new(),
                carries_format_bump: false,
                binary: b"x",
            },
            6,
        )
        .expect_err("refused");
        assert!(err.to_string().contains("major.minor.patch"), "{err}");
    }

    #[test]
    fn test_a_seed_round_trips_through_its_file_and_is_not_overwritten() {
        let seed = ReleaseSigningSeed::generate();
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("keys").join("release.key");
        seed.write_to(&path).expect("writes");
        let back = ReleaseSigningSeed::read_from(&path).expect("reads");
        assert_eq!(back.verifying_hex(), seed.verifying_hex());
        assert_eq!(back.to_hex(), seed.to_hex());
        let err = ReleaseSigningSeed::generate()
            .write_to(&path)
            .expect_err("a key in place is kept");
        assert!(err.to_string().contains("already holds a key"), "{err}");
        assert!(ReleaseSigningSeed::from_hex("zz").is_err());
    }
}

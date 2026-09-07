//! Signature scheme bindings as cluster state.
//!
//! Which scheme signs a JWT is policy for the whole group, the same shape as
//! an upgrade pause, so it rides the consensus log as a key-value entry and
//! lands in `zyron.auto.conf` on every member. The registry itself is built
//! at startup from what the binary submits, so without this the answer to
//! `SET SIGNATURE SCHEME` lasted until the process ended and reached no other
//! member at all.
//!
//! One config key per artifact kind under `[crypto]`, keyed by the kind's
//! catalog name lowercased. The value is what `SchemeRegistry` renders and
//! reads, so the string in the file, the string in the log, and the string a
//! node applies are one string handled by one pair of functions.

use std::path::Path;

use zyron_common::format::BinaryVersion;
use zyron_common::format::scheme::{ALL_ARTIFACT_KINDS, ArtifactKind};
use zyron_common::{Result, ZyronError};

use crate::config::ZyronConfig;

/// The release whose apply loop takes a scheme binding from the log. A member
/// on an earlier release has no key for it and would write a setting it never
/// reads, so the leader holds the entry until the whole group runs this or
/// later
pub const INTRODUCED_IN: BinaryVersion = BinaryVersion::new(0, 13, 0);

/// The config section every binding persists under
pub const CONFIG_SECTION: &str = "crypto";

/// The config key one artifact kind's binding persists under
pub fn config_key(kind: ArtifactKind) -> String {
    format!(
        "{CONFIG_SECTION}.{}",
        kind.catalog_name().to_ascii_lowercase()
    )
}

/// The artifact kind a config key names, None for a key that is not one
pub fn artifact_kind_for_config_key(key: &str) -> Option<ArtifactKind> {
    let field = key.strip_prefix(CONFIG_SECTION)?.strip_prefix('.')?;
    ALL_ARTIFACT_KINDS
        .iter()
        .copied()
        .find(|kind| kind.catalog_name().eq_ignore_ascii_case(field))
}

/// Whether a config key is a scheme binding the cluster replicates
pub fn is_crypto_setting(key: &str) -> bool {
    artifact_kind_for_config_key(key).is_some()
}

/// Applies one binding to this node's registry, without touching the config
/// file.
///
/// This is what a node seeds itself with at boot, where the file is already
/// the source, and what the log apply path goes through before it writes the
/// file back
pub fn apply_to_registry(key: &str, value: &str) -> Result<String> {
    let kind = artifact_kind_for_config_key(key).ok_or_else(|| {
        ZyronError::Internal(format!("`{key}` is not a signature scheme binding"))
    })?;
    let substrate = zyron_common::format::substrate()?;
    substrate
        .schemes
        .apply_binding_setting(kind, value)
        .map_err(|e| ZyronError::Internal(format!("{key} = {value} was refused, {e}")))
}

/// Applies one replicated binding on this node and writes it back to the
/// config, so the next boot seeds the registry with the value that is in
/// force now rather than with the binary's default
pub fn apply(data_dir: &Path, key: &str, value: &str) -> Result<()> {
    let stored = apply_to_registry(key, value)?;
    ZyronConfig::write_auto_conf(data_dir, key, &stored)
}

/// Seeds the registry from the config at boot.
///
/// A binding naming a scheme this binary does not carry is reported and
/// skipped rather than stopping the boot, because the alternative is a node
/// that will not start after a downgrade and cannot be reached to fix the
/// setting. The kind keeps the binary's default, which still verifies every
/// artifact the group has signed, and the warning names what to put right
pub fn seed_from_config(config: &ZyronConfig) -> usize {
    let mut seeded = 0;
    for (field, value) in &config.crypto.artifact_schemes {
        let key = format!("{CONFIG_SECTION}.{field}");
        match apply_to_registry(&key, value) {
            Ok(_) => seeded += 1,
            Err(e) => tracing::warn!(
                "the stored signature scheme binding {key} = {value} was not applied, {e}. \
                 This artifact kind is signed with the binary's default until the binding is set \
                 again"
            ),
        }
    }
    seeded
}

/// Every binding this node holds, as the config keys and values they persist
/// under. This is what a node hands the group when it is asked what it is
/// running, and what a test compares two members with
pub fn current_settings() -> Result<Vec<(String, String)>> {
    let substrate = zyron_common::format::substrate()?;
    Ok(ALL_ARTIFACT_KINDS
        .iter()
        .copied()
        .filter(|kind| kind.is_signed())
        .filter_map(|kind| {
            substrate
                .schemes
                .binding_setting(kind)
                .map(|value| (config_key(kind), value))
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_config_key_names_exactly_one_artifact_kind() {
        for kind in ALL_ARTIFACT_KINDS.iter().copied() {
            let key = config_key(kind);
            assert_eq!(
                artifact_kind_for_config_key(&key),
                Some(kind),
                "{key} did not resolve back to the kind it was built from"
            );
            assert!(is_crypto_setting(&key), "{key} is not recognised");
        }
        assert!(!is_crypto_setting("upgrade.paused"));
        assert!(!is_crypto_setting("server.port"));
        assert!(!is_crypto_setting("crypto"));
        assert!(!is_crypto_setting("crypto.not_a_kind"));
    }

    /// The value that reaches the log is the value the next boot reads, so a
    /// rotation set on one member and a restart of another arrive at the same
    /// binding
    #[test]
    fn a_binding_survives_the_round_trip_through_the_config() {
        let dir = tempfile::tempdir().expect("tempdir");
        let key = config_key(ArtifactKind::Jwt);
        let substrate = zyron_common::format::substrate().expect("substrate");
        let before = substrate
            .schemes
            .binding_setting(ArtifactKind::Jwt)
            .expect("JWT is bound");

        apply(dir.path(), &key, "Ed25519").expect("applies");
        let conf = std::fs::read_to_string(dir.path().join("zyron.auto.conf")).expect("reads");
        assert!(conf.contains("[crypto]"), "{conf}");
        assert!(conf.contains("Ed25519"), "{conf}");

        let parsed: toml::Table = toml::from_str(&conf).expect("parses");
        let section = parsed["crypto"].as_table().expect("crypto section");
        let stored = section["jwt"].as_str().expect("jwt binding");
        assert_eq!(
            apply_to_registry(&key, stored).expect("seeds"),
            "Ed25519",
            "the stored value did not seed back to what was applied"
        );

        substrate
            .schemes
            .apply_binding_setting(ArtifactKind::Jwt, &before)
            .expect("restores");
    }

    /// A binding naming a scheme this binary has no verifier for is refused
    /// where it is set, rather than accepted and found unusable later
    #[test]
    fn a_binding_onto_an_unregistered_scheme_is_refused() {
        let key = config_key(ArtifactKind::Jwt);
        let err = apply_to_registry(&key, "NotAScheme").expect_err("refused");
        assert!(err.to_string().contains("NotAScheme"), "{err}");
    }
}

//! Auth format and signature scheme registrations.
//!
//! The secret store is rewritten in full every time a key is created,
//! rotated, or deleted, so it migrates lazily: the next key operation after
//! an upgrade moves it forward and nothing sweeps it in the meantime.
//!
//! The signature schemes registered here are the ones this binary can verify
//! with today, plus reserved slots for schemes whose verifier has not
//! shipped. A reserved slot cannot be set on an artifact kind and cannot
//! verify anything: it exists so the numeric tag is allocated for the life
//! of the product and a future release adds a verifier without a wire change

use zyron_common::format::FormatKind;
use zyron_common::format::registry::{DeprecationStatus, FormatRegistration, MigrationPolicy};
use zyron_common::format::scheme::{
    SchemeCategory, SchemeId, SchemeStatus, SignatureSchemeRegistration,
};
use zyron_common::format::version::{FormatVersion, VersionWindow};

const GATE: &str = "0.11.0";

/// Version the secret store file is written at
pub const SECRET_STORE_FORMAT_VERSION: FormatVersion = FormatVersion::V1;

inventory::submit! {
    FormatRegistration {
        kind: FormatKind::SecretStorePersistence,
        writer_current_version: SECRET_STORE_FORMAT_VERSION,
        reader_supported_versions: VersionWindow::single(SECRET_STORE_FORMAT_VERSION),
        migration_policy: MigrationPolicy::Lazy,
        migration_reversible: true,
        binary_version_gate: GATE,
        deprecation_status: DeprecationStatus::Active,
        retirement_date: None,
        downgrade_write_supported: false,
        notes: "wrapped key material, lazy on the next key operation",
    }
}

/// Version the principal key file is written at
pub const PRINCIPAL_KEY_STORE_FORMAT_VERSION: FormatVersion = FormatVersion::V1;

inventory::submit! {
    FormatRegistration {
        kind: FormatKind::PrincipalKeyStorePersistence,
        writer_current_version: PRINCIPAL_KEY_STORE_FORMAT_VERSION,
        reader_supported_versions: VersionWindow::single(PRINCIPAL_KEY_STORE_FORMAT_VERSION),
        migration_policy: MigrationPolicy::Lazy,
        migration_reversible: true,
        binary_version_gate: "0.13.0",
        deprecation_status: DeprecationStatus::Active,
        retirement_date: None,
        downgrade_write_supported: false,
        notes: "principal signing keys, rewritten in full on the next issue or rotation",
    }
}

// ---------------------------------------------------------------------------
// Signature schemes
// ---------------------------------------------------------------------------

inventory::submit! {
    SignatureSchemeRegistration {
        scheme_name: "Ed25519",
        scheme_id: SchemeId(1),
        category: SchemeCategory::Signature,
        status: SchemeStatus::Active,
        first_available_version: "0.1.0",
        retirement_date: None,
        notes: "Zyron's default signing scheme for its own artifacts",
    }
}

inventory::submit! {
    SignatureSchemeRegistration {
        scheme_name: "ES256",
        scheme_id: SchemeId(2),
        category: SchemeCategory::Signature,
        status: SchemeStatus::Active,
        first_available_version: "0.1.0",
        retirement_date: None,
        notes: "ECDSA over P-256 with SHA-256, accepted for WebAuthn interop",
    }
}

inventory::submit! {
    SignatureSchemeRegistration {
        scheme_name: "RS256",
        scheme_id: SchemeId(3),
        category: SchemeCategory::Signature,
        status: SchemeStatus::Active,
        first_available_version: "0.1.0",
        retirement_date: None,
        notes: "RSASSA-PKCS1-v1_5 with SHA-256, accepted for external IdP interop",
    }
}

inventory::submit! {
    SignatureSchemeRegistration {
        scheme_name: "ML-DSA-65",
        scheme_id: SchemeId(16),
        category: SchemeCategory::Signature,
        status: SchemeStatus::Reserved,
        first_available_version: "0.11.0",
        retirement_date: None,
        notes: "tag reserved for the lattice signature scheme, no verifier in this binary",
    }
}

inventory::submit! {
    SignatureSchemeRegistration {
        scheme_name: "SLH-DSA-SHA2-128s",
        scheme_id: SchemeId(17),
        category: SchemeCategory::Signature,
        status: SchemeStatus::Reserved,
        first_available_version: "0.11.0",
        retirement_date: None,
        notes: "tag reserved for the hash signature scheme, no verifier in this binary",
    }
}

inventory::submit! {
    SignatureSchemeRegistration {
        scheme_name: "Ed25519+ML-DSA-65",
        scheme_id: SchemeId(18),
        category: SchemeCategory::Signature,
        status: SchemeStatus::Reserved,
        first_available_version: "0.11.0",
        retirement_date: None,
        notes: "tag reserved for the hybrid of the two, no verifier in this binary",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_the_secret_store_registers_once() {
        let count = inventory::iter::<FormatRegistration>
            .into_iter()
            .filter(|r| r.kind == FormatKind::SecretStorePersistence)
            .count();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_scheme_ids_and_names_are_unique() {
        let mut ids = HashSet::new();
        let mut names = HashSet::new();
        for scheme in inventory::iter::<SignatureSchemeRegistration> {
            assert!(
                ids.insert(scheme.scheme_id),
                "scheme id {} is allocated twice",
                scheme.scheme_id
            );
            assert!(
                names.insert(scheme.scheme_name.to_ascii_lowercase()),
                "scheme name `{}` is registered twice",
                scheme.scheme_name
            );
        }
    }

    #[test]
    fn test_the_three_classical_schemes_are_active() {
        for name in ["Ed25519", "ES256", "RS256"] {
            let scheme = inventory::iter::<SignatureSchemeRegistration>
                .into_iter()
                .find(|s| s.scheme_name == name)
                .unwrap_or_else(|| panic!("{name} is registered"));
            assert_eq!(scheme.status, SchemeStatus::Active);
            assert!(scheme.scheme_id.as_artifact_byte().is_some());
        }
    }

    #[test]
    fn test_the_post_quantum_slots_are_reserved() {
        for name in ["ML-DSA-65", "SLH-DSA-SHA2-128s", "Ed25519+ML-DSA-65"] {
            let scheme = inventory::iter::<SignatureSchemeRegistration>
                .into_iter()
                .find(|s| s.scheme_name == name)
                .unwrap_or_else(|| panic!("{name} is registered"));
            assert_eq!(scheme.status, SchemeStatus::Reserved);
            assert!(!scheme.status.can_sign());
        }
    }
}

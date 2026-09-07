//! The deprecation lifecycle and signature agility.
//!
//! Covers validation items 10 through 14 and 22 through 24: the three
//! classical schemes registered with distinct ids and per-artifact defaults,
//! verification dispatching on the artifact's declared scheme, rotation with
//! an overlap accepting both, service principal key rotation with a scheme
//! change, a retired scheme's verifier surviving until its last artifact
//! expires, the warn to error to removed walk, the per-tenant warning rate
//! limit, and the generated migration guide

use ed25519_dalek::{Signer, SigningKey};
use zyron_auth::signature::{
    DEFAULT_ROTATION_OVERLAP_SECS, PrincipalKeyStore, VerifyingMaterial, tag_binary_signature,
    verify_artifact, verify_binary_artifact,
};
use zyron_common::format::deprecation::{
    BinaryVersion, DeprecatedItemKind, DeprecationRecord, DeprecationRegistry,
    DeprecationRegistryError, DeprecationStage, WarningLog, WarningRateLimiter,
};
use zyron_common::format::migration::MigrationBoard;
use zyron_common::format::scheme::{
    ALL_ARTIFACT_KINDS, ArtifactKind, ArtifactSchemeBinding, SchemeCategory, SchemeError, SchemeId,
    SchemeIdentifierEncoding, SchemeRegistry, SchemeStatus, SignatureSchemeRegistration,
    default_artifact_bindings,
};
use zyron_common::format::wire_version::WireVersionRegistry;
use zyron_common::format::{CatalogSchemaRegistry, FormatRegistry, FormatSubstrate};

fn substrate() -> &'static FormatSubstrate {
    zyron_common::format::substrate().expect("loads")
}

fn scheme(name: &'static str, id: u16, status: SchemeStatus) -> SignatureSchemeRegistration {
    SignatureSchemeRegistration {
        scheme_name: name,
        scheme_id: SchemeId(id),
        category: SchemeCategory::Signature,
        status,
        first_available_version: "0.11.0",
        retirement_date: None,
        notes: "test scheme",
    }
}

fn ed25519_pair(seed: u8) -> (SigningKey, VerifyingMaterial) {
    let signing = SigningKey::from_bytes(&[seed; 32]);
    let material = VerifyingMaterial::Ed25519(signing.verifying_key().to_bytes());
    (signing, material)
}

// ---------------------------------------------------------------------------
// Signature agility
// ---------------------------------------------------------------------------

/// Item 10. The three classical schemes are registered with unique ids, and
/// every signed artifact kind has a default scheme
#[test]
fn the_classical_schemes_are_registered_with_defaults_per_artifact_kind() {
    let registry = &substrate().schemes;
    for (name, id) in [("Ed25519", 1u16), ("ES256", 2), ("RS256", 3)] {
        let scheme = registry
            .by_name(name)
            .unwrap_or_else(|| panic!("{name} is registered"));
        assert_eq!(scheme.scheme_id, SchemeId(id));
        assert_eq!(scheme.status, SchemeStatus::Active);
        assert_eq!(scheme.category, SchemeCategory::Signature);
        assert_eq!(
            registry.by_id(SchemeId(id)).map(|s| s.scheme_name),
            Some(name)
        );
    }

    // The reserved slots hold their tags without being usable
    for (name, id) in [
        ("ML-DSA-65", 16u16),
        ("SLH-DSA-SHA2-128s", 17),
        ("Ed25519+ML-DSA-65", 18),
    ] {
        let scheme = registry
            .by_name(name)
            .unwrap_or_else(|| panic!("{name} holds a reserved slot"));
        assert_eq!(scheme.scheme_id, SchemeId(id));
        assert_eq!(scheme.status, SchemeStatus::Reserved);
        assert!(!scheme.status.can_sign());
    }

    // Every signed artifact kind is bound, and SAML goes out RS256
    for kind in ALL_ARTIFACT_KINDS {
        let binding = registry.binding(*kind);
        if kind.is_signed() {
            let binding = binding.unwrap_or_else(|| panic!("{kind} is bound"));
            let expected = match kind {
                ArtifactKind::SamlAssertion => "RS256",
                _ => "Ed25519",
            };
            assert_eq!(binding.current_scheme, expected, "{kind}");
            assert!(binding.deprecating_scheme.is_none());
        } else {
            assert!(binding.is_none(), "{kind} carries no signature");
            assert_eq!(
                kind.identifier_encoding(),
                SchemeIdentifierEncoding::OpaqueBearer
            );
        }
    }
}

/// Item 11. A JWT declaring Ed25519 dispatches to the Ed25519 verifier, one
/// declaring an unregistered scheme fails closed, and one declaring a
/// registered scheme the kind is not bound to also fails closed
#[test]
fn verification_dispatches_on_the_declared_scheme() {
    let registry = SchemeRegistry::from_parts(
        vec![
            scheme("Ed25519", 1, SchemeStatus::Active),
            scheme("ES256", 2, SchemeStatus::Active),
        ],
        default_artifact_bindings(),
    );
    let (signing, material) = ed25519_pair(5);
    let message = b"the jwt signing input";
    let signature = signing.sign(message).to_bytes().to_vec();

    assert!(
        verify_artifact(
            &registry,
            ArtifactKind::Jwt,
            "Ed25519",
            &material,
            message,
            &signature,
            0
        )
        .expect("verifies")
    );

    // A forged signature does not verify
    assert!(
        !verify_artifact(
            &registry,
            ArtifactKind::Jwt,
            "Ed25519",
            &material,
            b"a different message",
            &signature,
            0
        )
        .expect("runs")
    );

    // An unregistered scheme fails closed, naming it
    let err = verify_artifact(
        &registry,
        ArtifactKind::Jwt,
        "HS256",
        &material,
        message,
        &signature,
        0,
    )
    .expect_err("fails closed");
    assert!(err.to_string().contains("HS256"), "{err}");
    assert!(err.to_string().contains("not registered"), "{err}");

    // A registered scheme the kind is not bound to also fails closed
    let err = verify_artifact(
        &registry,
        ArtifactKind::Jwt,
        "ES256",
        &material,
        message,
        &signature,
        0,
    )
    .expect_err("fails closed");
    assert!(err.to_string().contains("not accepted for JWT"), "{err}");
}

/// Item 12. Rotating a kind onto a new scheme with an overlap accepts both
/// until the overlap ends, then refuses the outgoing one
#[test]
fn a_rotation_accepts_both_schemes_until_the_overlap_ends() {
    let registry = SchemeRegistry::from_parts(
        vec![
            scheme("Ed25519", 1, SchemeStatus::Active),
            // The reserved tag, pretend-registered as active so a rotation
            // onto it can be exercised before its verifier ships
            scheme("ML-DSA-65", 16, SchemeStatus::Active),
        ],
        default_artifact_bindings(),
    );
    let overlap_end = 24 * 3_600;
    let binding = registry
        .rotate_scheme(ArtifactKind::Jwt, "ML-DSA-65", overlap_end)
        .expect("rotates");
    assert_eq!(binding.current_scheme, "ML-DSA-65");
    assert_eq!(binding.deprecating_scheme.as_deref(), Some("Ed25519"));
    assert_eq!(binding.overlap_end_secs, Some(overlap_end));
    assert!(binding.rotation_in_progress(0));

    // Both are accepted during the overlap
    for now in [0, overlap_end - 1] {
        assert!(
            registry
                .resolve_for_verification(ArtifactKind::Jwt, "Ed25519", now)
                .is_ok(),
            "the outgoing scheme still verifies at {now}"
        );
        assert!(
            registry
                .resolve_for_verification(ArtifactKind::Jwt, "ML-DSA-65", now)
                .is_ok()
        );
    }

    // After the overlap the outgoing one is refused, naming when it ended
    let err = registry
        .resolve_for_verification(ArtifactKind::Jwt, "Ed25519", overlap_end)
        .expect_err("refused");
    assert!(matches!(err, SchemeError::NotAcceptedForArtifact { .. }));
    assert!(err.to_string().contains("overlap ended"), "{err}");
    assert!(
        registry
            .resolve_for_verification(ArtifactKind::Jwt, "ML-DSA-65", overlap_end)
            .is_ok(),
        "the new scheme keeps verifying"
    );
    assert!(
        !registry
            .binding(ArtifactKind::Jwt)
            .expect("bound")
            .rotation_in_progress(overlap_end)
    );
}

/// A rotation onto a reserved scheme is refused, because a scheme with no
/// verifier would strand every artifact it signed
#[test]
fn a_rotation_onto_a_reserved_scheme_is_refused() {
    let registry = &substrate().schemes;
    let err = registry
        .rotate_scheme(ArtifactKind::SessionToken, "ML-DSA-65", 1_000)
        .expect_err("refused");
    assert!(matches!(err, SchemeError::ReservedScheme { .. }));
    assert!(err.to_string().contains("no verifier"), "{err}");
}

/// Item 13. A service principal's key rotates onto a new scheme with an
/// overlap, the new key signs, and the old one still verifies until the
/// overlap ends
#[test]
fn a_service_principal_key_rotates_with_an_overlap() {
    let registry = SchemeRegistry::from_parts(
        vec![
            scheme("Ed25519", 1, SchemeStatus::Active),
            scheme("ML-DSA-65", 16, SchemeStatus::Active),
        ],
        default_artifact_bindings(),
    );
    let store = PrincipalKeyStore::new();
    let first = store.issue("sp1", "Ed25519", 0).expect("issues");

    let overlap = DEFAULT_ROTATION_OVERLAP_SECS;
    let outcome = store
        .rotate(&registry, "sp1", Some("Ed25519"), overlap, 100)
        .expect("rotates");
    assert_eq!(outcome.previous_scheme.as_deref(), Some("Ed25519"));
    assert_eq!(outcome.overlap_end_secs, 100 + overlap);

    let current = store.current("sp1").expect("has a key");
    assert_ne!(current.public_key, first.public_key, "a new key was issued");

    // Both keys verify during the overlap, only the new one after it
    let during = store.verifying("sp1", 200);
    assert_eq!(during.len(), 2);
    assert!(during.iter().any(|k| k.public_key == first.public_key));
    let after = store.verifying("sp1", 100 + overlap + 1);
    assert_eq!(after.len(), 1);
    assert_eq!(after[0].public_key, current.public_key);

    // The new key signs, and its signature verifies against its public half
    let signature = store.sign("sp1", b"a token body").expect("signs");
    let material = current.verifying_material().expect("key material");
    assert!(
        zyron_auth::signature::verify_with(&material, b"a token body", &signature)
            .expect("verifies")
    );

    // The sweep drops the outgoing key once its overlap has passed
    assert_eq!(store.sweep(150), 0);
    assert_eq!(store.sweep(100 + overlap + 1), 1);
    assert_eq!(store.verifying("sp1", 100 + overlap + 1).len(), 1);
}

/// Item 14. A retired scheme keeps verifying until the last artifact signed
/// with it expires, then its verifier is reported as removable and refuses
#[test]
fn a_retired_scheme_verifies_until_its_last_artifact_expires() {
    let registry = SchemeRegistry::from_parts(
        vec![
            scheme("Ed25519", 1, SchemeStatus::Active),
            scheme("ES256", 2, SchemeStatus::Retired),
        ],
        default_artifact_bindings(),
    );
    // ES256 signed the tokens already out there, then the kind rotated to
    // Ed25519 with an overlap that has not ended
    registry
        .set_scheme(ArtifactKind::SessionToken, "ES256")
        .expect("sets");
    registry
        .rotate_scheme(ArtifactKind::SessionToken, "Ed25519", u64::MAX)
        .expect("rotates");

    let last_expiry = 5_000;
    registry.note_artifact_expiry("ES256", last_expiry);
    assert_eq!(registry.last_valid_artifact_expiry("ES256"), last_expiry);

    // Before the expiry the verifier is kept and still answers
    assert!(
        registry
            .resolve_for_verification(ArtifactKind::SessionToken, "ES256", last_expiry - 1)
            .is_ok()
    );
    assert!(
        registry
            .verifiers_ready_for_removal(last_expiry - 1)
            .is_empty()
    );

    // At and after the expiry the sweep reports it removable and it refuses
    assert_eq!(
        registry.verifiers_ready_for_removal(last_expiry),
        vec!["ES256"]
    );
    let err = registry
        .resolve_for_verification(ArtifactKind::SessionToken, "ES256", last_expiry)
        .expect_err("refused");
    assert!(matches!(err, SchemeError::RetiredScheme { .. }));
    assert!(err.to_string().contains("5000"), "{err}");
}

/// A binary artifact carries its scheme in a leading id byte, which the
/// dispatch reads before any verifier runs
#[test]
fn a_binary_artifact_carries_its_scheme_in_a_leading_byte() {
    let registry = SchemeRegistry::from_parts(
        vec![scheme("Ed25519", 1, SchemeStatus::Active)],
        default_artifact_bindings(),
    );
    let (signing, material) = ed25519_pair(9);
    let message = b"federation invite body";
    let signature = signing.sign(message).to_bytes().to_vec();
    let artifact = tag_binary_signature(&registry, "Ed25519", &signature).expect("tags");
    assert_eq!(artifact[0], 1, "the leading byte is the scheme id");

    assert!(
        verify_binary_artifact(
            &registry,
            ArtifactKind::FederationInviteToken,
            &material,
            message,
            &artifact,
            0
        )
        .expect("verifies")
    );

    // An unknown tag fails closed, naming the id it found
    let mut unknown = artifact.clone();
    unknown[0] = 200;
    let err = verify_binary_artifact(
        &registry,
        ArtifactKind::FederationInviteToken,
        &material,
        message,
        &unknown,
        0,
    )
    .expect_err("fails closed");
    assert!(err.to_string().contains("200"), "{err}");
}

// ---------------------------------------------------------------------------
// Deprecation lifecycle
// ---------------------------------------------------------------------------

fn record() -> DeprecationRecord {
    DeprecationRecord {
        item_kind: DeprecatedItemKind::DdlKeyword,
        item_id: "CREATE PIPELINE",
        deprecated_since_version: "1.0.0",
        warn_until_version: "1.2.0",
        error_since_version: "1.3.0",
        removed_since_version: "1.4.0",
        replacement_ref: Some("CREATE WORKFLOW"),
        migration_guide_url: Some("/docs/sql/create-workflow.md"),
        no_guide_required: false,
        summary: "Pipelines became workflows",
        before_example: "CREATE PIPELINE p AS (STAGE s AS SELECT 1)",
        after_example: "CREATE WORKFLOW w AS (TASK s AS SELECT 1)",
        migration_snippet: "SELECT name FROM zyron_sys.core.pipelines",
        common_pitfall: "A stage name is not a task name, rename it explicitly",
    }
}

/// A substrate holding one deprecation, so emission can be exercised without
/// this release deprecating anything
fn substrate_with(records: Vec<DeprecationRecord>, limit: u32) -> FormatSubstrate {
    FormatSubstrate {
        formats: FormatRegistry::from_parts(&[], &[], &[]).expect("loads"),
        schemes: SchemeRegistry::from_parts(Vec::new(), Vec::new()),
        catalog_schemas: CatalogSchemaRegistry::from_parts(&[], &[]).expect("loads"),
        deprecations: DeprecationRegistry::from_records(records).expect("loads"),
        wire_versions: WireVersionRegistry::from_versions(Vec::new()),
        migrations: MigrationBoard::new(),
        warning_limiter: WarningRateLimiter::new(limit),
        warning_log: WarningLog::default(),
    }
}

/// Item 22. An item walks warn, error, then removed as the running version
/// advances, and its record stays in the registry forever
#[test]
fn a_deprecated_item_walks_warn_then_error_then_removed() {
    let record = record();
    let stages = [
        ("0.9.0", DeprecationStage::Supported),
        ("1.0.0", DeprecationStage::Warning),
        ("1.1.0", DeprecationStage::Warning),
        ("1.2.0", DeprecationStage::Warning),
        ("1.3.0", DeprecationStage::Erroring),
        ("1.4.0", DeprecationStage::Removed),
        ("9.9.9", DeprecationStage::Removed),
    ];
    for (version, expected) in stages {
        let running = BinaryVersion::parse(version).expect("parses");
        assert_eq!(record.stage(running), expected, "at {version}");
    }

    // The record survives past removal, so a support question has an answer
    let registry = DeprecationRegistry::from_records(vec![record]).expect("loads");
    let found = registry.find("create pipeline").expect("found");
    assert_eq!(found.removed_since_version, "1.4.0");
    assert_eq!(
        registry
            .at_stage(
                BinaryVersion::parse("9.9.9").expect("parses"),
                DeprecationStage::Removed
            )
            .len(),
        1
    );
}

/// Item 22, the guidance. Every stage past supported carries a message
/// naming the replacement and the guide
#[test]
fn the_guidance_names_the_replacement_and_the_guide() {
    let text = record().guidance();
    assert!(text.contains("CREATE PIPELINE"), "{text}");
    assert!(text.contains("CREATE WORKFLOW"), "{text}");
    assert!(text.contains("errors from 1.3.0"), "{text}");
    assert!(text.contains("removed in 1.4.0"), "{text}");
    assert!(text.contains("/docs/sql/create-workflow.md"), "{text}");
}

/// Item 23. A hundred uses inside the hour produce ten warnings, and the
/// eleventh use still works, silently
#[test]
fn the_warning_rate_limit_is_per_item_per_tenant_per_hour() {
    let mut warning = record();
    warning.deprecated_since_version = "0.0.1";
    warning.warn_until_version = "99.0.0";
    warning.error_since_version = "99.0.0";
    warning.removed_since_version = "99.1.0";
    let substrate = substrate_with(vec![warning], 10);

    let mut warned = 0;
    let mut permitted = 0;
    for _ in 0..100 {
        let outcome = substrate.check_deprecated_use("CREATE PIPELINE", "tenant-a", 0);
        if outcome.message().is_some() {
            warned += 1;
        }
        if outcome.permitted() {
            permitted += 1;
        }
    }
    assert_eq!(warned, 10, "ten warnings inside the hour");
    assert_eq!(permitted, 100, "every use still works");
    assert_eq!(substrate.warning_limiter.suppressed(), 90);

    // A second tenant has its own budget
    let outcome = substrate.check_deprecated_use("CREATE PIPELINE", "tenant-b", 0);
    assert!(outcome.message().is_some(), "tenant B has its own budget");

    // And the budget resets on the next hour
    let outcome = substrate.check_deprecated_use("CREATE PIPELINE", "tenant-a", 3_600);
    assert!(outcome.message().is_some(), "the budget resets hourly");

    // The report shows the item, the count, and both tenants
    let rows = substrate.warning_log.report(0);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].item_id, "CREATE PIPELINE");
    assert_eq!(rows[0].count, 12);
    assert_eq!(
        rows[0].tenants,
        vec!["tenant-a".to_string(), "tenant-b".to_string()]
    );
}

/// Item 22, the error window. A use past the warn window is refused with the
/// same guidance the warning carried
#[test]
fn a_use_past_the_warn_window_is_refused() {
    let mut erroring = record();
    erroring.deprecated_since_version = "0.0.1";
    erroring.warn_until_version = "0.0.2";
    erroring.error_since_version = "0.0.2";
    erroring.removed_since_version = "99.0.0";
    let substrate = substrate_with(vec![erroring], 10);
    let outcome = substrate.check_deprecated_use("CREATE PIPELINE", "t", 0);
    assert!(!outcome.permitted());
    assert!(
        outcome
            .message()
            .expect("carries guidance")
            .contains("CREATE WORKFLOW")
    );
}

/// A scan over statement text finds a deprecated keyword on a word boundary
/// and leaves an object whose name merely contains it alone
#[test]
fn a_scan_finds_a_deprecated_keyword_without_false_positives() {
    let mut warning = record();
    warning.deprecated_since_version = "0.0.1";
    warning.warn_until_version = "99.0.0";
    warning.error_since_version = "99.0.0";
    warning.removed_since_version = "99.1.0";
    let substrate = substrate_with(vec![warning], 100);

    let found = substrate.scan_sql_for_deprecations(
        "create pipeline nightly AS (STAGE s AS SELECT 1)",
        "t",
        0,
    );
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].0, "CREATE PIPELINE");

    let clean =
        substrate.scan_sql_for_deprecations("SELECT * FROM zyron_sys.stat.pipeline_runs", "t", 0);
    assert!(clean.is_empty(), "{clean:?}");
}

/// Item 24. The migration guide is generated from the record, so it carries
/// the lifecycle, the before and after examples, the script, and the
/// pitfalls, and cannot drift from the dates beside it
#[test]
fn the_migration_guide_is_generated_from_the_record() {
    let guide = record().migration_guide();
    assert_eq!(guide.item_id, "CREATE PIPELINE");
    assert_eq!(guide.item_kind, DeprecatedItemKind::DdlKeyword);
    assert_eq!(guide.url, Some("/docs/sql/create-workflow.md"));

    let body = &guide.body;
    assert!(body.contains("# CREATE PIPELINE"), "{body}");
    assert!(body.contains("Deprecated in 1.0.0"), "{body}");
    assert!(body.contains("Errors from 1.3.0"), "{body}");
    assert!(body.contains("Removed in 1.4.0"), "{body}");
    assert!(body.contains("Replaced by `CREATE WORKFLOW`"), "{body}");
    assert!(body.contains("## Before"), "{body}");
    assert!(
        body.contains("CREATE PIPELINE p AS (STAGE s AS SELECT 1)"),
        "{body}"
    );
    assert!(body.contains("## After"), "{body}");
    assert!(
        body.contains("CREATE WORKFLOW w AS (TASK s AS SELECT 1)"),
        "{body}"
    );
    assert!(body.contains("## Migration script"), "{body}");
    assert!(body.contains("zyron_sys.core.pipelines"), "{body}");
    assert!(body.contains("## Common pitfalls"), "{body}");
    assert!(body.contains("A stage name is not a task name"), "{body}");
}

/// A record with lifecycle versions out of order, or with no guide and no
/// exemption, is refused at load rather than shipped
#[test]
fn a_malformed_deprecation_record_is_refused_at_load() {
    let mut out_of_order = record();
    out_of_order.error_since_version = "1.0.0";
    out_of_order.warn_until_version = "1.2.0";
    assert!(matches!(
        DeprecationRegistry::from_records(vec![out_of_order]),
        Err(DeprecationRegistryError::OutOfOrderLifecycle { .. })
    ));

    let mut no_guide = record();
    no_guide.migration_guide_url = None;
    assert!(matches!(
        DeprecationRegistry::from_records(vec![no_guide]),
        Err(DeprecationRegistryError::MissingGuide { .. })
    ));

    let mut exempt = record();
    exempt.migration_guide_url = None;
    exempt.no_guide_required = true;
    DeprecationRegistry::from_records(vec![exempt]).expect("an exempt record loads");

    assert!(matches!(
        DeprecationRegistry::from_records(vec![record(), record()]),
        Err(DeprecationRegistryError::DuplicateItem { .. })
    ));
}

/// This release deprecates nothing, which is what makes the registry empty
/// and the release check's rewriter requirement vacuously satisfied
#[test]
fn this_release_deprecates_the_removed_contact_channel_kind() {
    let records = substrate().deprecations.records();
    assert_eq!(
        records.len(),
        1,
        "a deprecation added here needs a rewriter and a guide"
    );
    let record = &records[0];
    assert_eq!(record.item_id, "upgrade contact channel kind pagerduty");
    assert_eq!(record.item_kind, DeprecatedItemKind::ConfigKey);
    assert!(
        !record.item_kind.is_user_authored_sql(),
        "a config key is owed no SQL rewriter"
    );
    assert_eq!(
        record.stage(BinaryVersion::parse("0.14.0").expect("parses")),
        DeprecationStage::Removed
    );

    // The replacement the record names is a channel this binary builds
    assert!(
        zyron_server::upgrade::notification::ContactChannel::discord(
            "https://discord.com/api/webhooks/1/abc"
        )
        .is_ok()
    );

    let guides = substrate().deprecations.guides();
    assert_eq!(guides.len(), 1);
    assert!(
        guides[0].body.contains("notify_discord_webhook_url"),
        "{}",
        guides[0].body
    );
}

//! The format, signature, and upgrade DDL surface.
//!
//! Every word these statements introduce is matched as a soft keyword, so a
//! table or column already called `rotate`, `list`, `upgrade`, `scheme` or
//! `object` keeps working. The last test in this file is what holds that

use zyron_parser::ast::{
    AcknowledgeRewriteCategory, ListRegistryTarget, ShowUpgradeTarget, Statement,
    TriggerUpgradeAction,
};
use zyron_parser::parse;

fn one(sql: &str) -> Statement {
    let mut stmts = parse(sql).unwrap_or_else(|e| panic!("`{sql}` did not parse, {e}"));
    assert_eq!(
        stmts.len(),
        1,
        "`{sql}` produced {} statements",
        stmts.len()
    );
    stmts.remove(0)
}

#[test]
fn test_set_signature_scheme_bare_and_quoted() {
    for sql in [
        "SET SIGNATURE SCHEME Ed25519 FOR ARTIFACT KIND JWT",
        "SET SIGNATURE SCHEME 'Ed25519' FOR ARTIFACT KIND 'JWT'",
    ] {
        match one(sql) {
            Statement::SetSignatureScheme(s) => {
                // A bare name folds to lower case the way every other bare
                // identifier does, and both registries resolve case
                // insensitively, so the two spellings mean the same thing
                assert!(s.scheme.eq_ignore_ascii_case("Ed25519"), "{}", s.scheme);
                assert!(
                    s.artifact_kind.eq_ignore_ascii_case("JWT"),
                    "{}",
                    s.artifact_kind
                );
            }
            other => panic!("`{sql}` parsed as {other:?}"),
        }
    }
}

#[test]
fn test_rotate_signature_scheme_with_and_without_overlap() {
    match one("ROTATE SIGNATURE SCHEME JWT TO 'ML-DSA-65' OVERLAP '24h'") {
        Statement::RotateSignatureScheme(s) => {
            assert!(s.artifact_kind.eq_ignore_ascii_case("JWT"));
            assert_eq!(s.new_scheme, "ML-DSA-65");
            assert_eq!(s.overlap.as_deref(), Some("24h"));
        }
        other => panic!("parsed as {other:?}"),
    }
    match one("ROTATE SIGNATURE SCHEME JWT TO 'ML-DSA-65'") {
        Statement::RotateSignatureScheme(s) => assert!(s.overlap.is_none()),
        other => panic!("parsed as {other:?}"),
    }
}

#[test]
fn test_rotate_service_principal_key() {
    match one("ROTATE SERVICE PRINCIPAL KEY sp1 SCHEME 'ML-DSA-65' OVERLAP '24h'") {
        Statement::RotateServicePrincipalKey(s) => {
            assert_eq!(s.principal, "sp1");
            assert_eq!(s.new_scheme.as_deref(), Some("ML-DSA-65"));
            assert_eq!(s.overlap.as_deref(), Some("24h"));
        }
        other => panic!("parsed as {other:?}"),
    }
    match one("ROTATE SERVICE PRINCIPAL KEY sp1") {
        Statement::RotateServicePrincipalKey(s) => {
            assert_eq!(s.principal, "sp1");
            assert!(s.new_scheme.is_none());
            assert!(s.overlap.is_none());
        }
        other => panic!("parsed as {other:?}"),
    }
}

#[test]
fn test_list_statements() {
    let cases = [
        (
            "LIST SIGNATURE SCHEMES",
            ListRegistryTarget::SignatureSchemes,
        ),
        ("LIST ARTIFACT SCHEMES", ListRegistryTarget::ArtifactSchemes),
        ("LIST UPGRADE HISTORY", ListRegistryTarget::UpgradeHistory),
        ("LIST FORMAT REGISTRY", ListRegistryTarget::FormatRegistry),
        ("LIST DEPRECATIONS", ListRegistryTarget::Deprecations),
    ];
    for (sql, expected) in cases {
        match one(sql) {
            Statement::ListRegistry(s) => {
                assert_eq!(s.target, expected, "{sql}");
                assert!(s.limit.is_none(), "{sql}");
            }
            other => panic!("`{sql}` parsed as {other:?}"),
        }
    }
    match one("LIST UPGRADE HISTORY LIMIT 10") {
        Statement::ListRegistry(s) => {
            assert_eq!(s.target, ListRegistryTarget::UpgradeHistory);
            assert_eq!(s.limit, Some(10));
        }
        other => panic!("parsed as {other:?}"),
    }
}

#[test]
fn test_trigger_manual_upgrade_and_rollback() {
    match one("TRIGGER MANUAL UPGRADE TO '2.3.1'") {
        Statement::TriggerUpgrade(s) => {
            assert_eq!(s.action, TriggerUpgradeAction::UpgradeTo("2.3.1".into()));
        }
        other => panic!("parsed as {other:?}"),
    }
    match one("TRIGGER MANUAL ROLLBACK") {
        Statement::TriggerUpgrade(s) => assert_eq!(s.action, TriggerUpgradeAction::Rollback),
        other => panic!("parsed as {other:?}"),
    }
}

#[test]
fn test_acknowledge_upgrade_rewrites_names_a_category() {
    match one("ACKNOWLEDGE UPGRADE REWRITES AMBIGUOUS") {
        Statement::AcknowledgeUpgradeRewrites(s) => {
            assert_eq!(s.category, AcknowledgeRewriteCategory::Ambiguous);
        }
        other => panic!("parsed as {other:?}"),
    }
    match one("acknowledge upgrade rewrites unsafe") {
        Statement::AcknowledgeUpgradeRewrites(s) => {
            assert_eq!(s.category, AcknowledgeRewriteCategory::Unsafe);
        }
        other => panic!("parsed as {other:?}"),
    }
    // Safe rewrites apply on their own or are refused by policy, so there is
    // nothing for an operator to acknowledge about them
    let err = parse("ACKNOWLEDGE UPGRADE REWRITES SAFE").expect_err("safe is not a category");
    assert!(err.to_string().contains("AMBIGUOUS"), "{err}");
}

#[test]
fn test_show_upgrade_state_and_format_migrations() {
    match one("SHOW UPGRADE STATE") {
        Statement::ShowUpgrade(s) => assert_eq!(s.target, ShowUpgradeTarget::State),
        other => panic!("parsed as {other:?}"),
    }
    match one("SHOW FORMAT MIGRATIONS") {
        Statement::ShowUpgrade(s) => assert_eq!(
            s.target,
            ShowUpgradeTarget::FormatMigrations { format_kind: None }
        ),
        other => panic!("parsed as {other:?}"),
    }
    match one("SHOW FORMAT MIGRATIONS FOR FORMAT heap_page") {
        Statement::ShowUpgrade(s) => assert_eq!(
            s.target,
            ShowUpgradeTarget::FormatMigrations {
                format_kind: Some("heap_page".to_string())
            }
        ),
        other => panic!("parsed as {other:?}"),
    }
}

#[test]
fn test_explain_rewrite_for_object() {
    match one("EXPLAIN REWRITE FOR OBJECT sales.monthly_view") {
        Statement::ExplainRewrite(s) => assert_eq!(s.object_name, "sales.monthly_view"),
        other => panic!("parsed as {other:?}"),
    }
}

#[test]
fn test_alter_system_accepts_the_upgrade_settings() {
    for sql in [
        "ALTER SYSTEM SET upgrade_channel = 'beta'",
        "ALTER SYSTEM SET pinned_version = '2.3.1'",
        "ALTER SYSTEM SET auto_upgrade_enabled = false",
        "ALTER SYSTEM SET auto_upgrade_window = '02:00-04:00 UTC'",
        "ALTER SYSTEM SET auto_upgrade_paused = true",
    ] {
        match one(sql) {
            Statement::AlterSystemSet(_) => {}
            other => panic!("`{sql}` parsed as {other:?}"),
        }
    }
}

#[test]
fn test_set_user_object_rewrite_policy() {
    match one("SET USER_OBJECT_REWRITE_POLICY = 'notify_all'") {
        Statement::SetVariable(s) => {
            assert!(s.name.eq_ignore_ascii_case("user_object_rewrite_policy"));
        }
        other => panic!("parsed as {other:?}"),
    }
}

#[test]
fn test_malformed_statements_are_refused_with_guidance() {
    for sql in [
        "ROTATE NOTHING",
        "LIST NOTHING",
        "TRIGGER MANUAL NOTHING",
        "SET SIGNATURE SCHEME Ed25519",
    ] {
        assert!(parse(sql).is_err(), "`{sql}` should not parse");
    }
    let err = parse("ROTATE NOTHING").expect_err("refused").to_string();
    assert!(err.contains("ROTATE SIGNATURE SCHEME"), "{err}");
}

/// Every word this DDL introduces is soft, so a schema that already uses one
/// as a name keeps working. This is the test the house rule asks for.
///
/// `rollback` is not in the list because SQL:2016 already reserves it, which
/// is the one exception the rule allows
#[test]
fn test_new_words_still_work_as_identifiers() {
    for word in [
        "rotate",
        "list",
        "signature",
        "scheme",
        "schemes",
        "artifact",
        "kind",
        "overlap",
        "manual",
        "upgrade",
        "migrations",
        "rewrite",
        "object",
        "registry",
        "deprecations",
        "principal",
        "service",
        "state",
        "history",
        "acknowledge",
        "rewrites",
        "ambiguous",
    ] {
        let select = format!("SELECT {word} FROM t");
        assert!(parse(&select).is_ok(), "`{select}` should parse");
        let from = format!("SELECT a FROM {word}");
        assert!(parse(&from).is_ok(), "`{from}` should parse");
    }
}

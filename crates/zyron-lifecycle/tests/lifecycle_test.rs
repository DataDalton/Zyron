//! Integration tests for the data-lifecycle logic modules.

use zyron_lifecycle::classification::{
    ClassificationService, PiiKind, auto_classify_column, classify_value,
};
use zyron_lifecycle::compliance::{RetentionRequirement, validate_retention};
use zyron_lifecycle::legal_hold::{CompiledHold, HoldPredicateEvaluator, LegalHoldRegistry};
use zyron_lifecycle::scheduling::CleanupGovernor;
use zyron_lifecycle::tiered_storage::{StorageTier, TierCache, should_rehydrate};
use zyron_lifecycle::ttl::{TtlMode, is_expired};

#[test]
fn pii_detection_recognizes_common_shapes() {
    assert_eq!(classify_value("alice@example.com"), Some(PiiKind::Email));
    assert_eq!(classify_value("123-45-6789"), Some(PiiKind::Ssn));
    // Valid Luhn test card number.
    assert_eq!(
        classify_value("4111 1111 1111 1111"),
        Some(PiiKind::CreditCard)
    );
    assert_eq!(classify_value("+1 (415) 555-2671"), Some(PiiKind::Phone));
    assert_eq!(classify_value("just a sentence"), None);
    assert_eq!(classify_value(""), None);
}

#[test]
fn auto_classify_uses_majority() {
    let samples = vec![
        "a@b.com".to_string(),
        "c@d.org".to_string(),
        "not-an-email".to_string(),
    ];
    assert_eq!(auto_classify_column(&samples), Some(PiiKind::Email));
    let none: Vec<String> = vec!["x".into(), "y".into()];
    assert_eq!(auto_classify_column(&none), None);
}

#[test]
fn classification_level_parsing_is_case_insensitive() {
    use zyron_auth::ClassificationLevel;
    assert_eq!(
        ClassificationService::parse_level("Confidential").unwrap(),
        ClassificationLevel::Confidential
    );
    assert_eq!(
        ClassificationService::parse_level("RESTRICTED").unwrap(),
        ClassificationLevel::Restricted
    );
    assert!(ClassificationService::parse_level("bogus").is_err());
}

#[test]
fn ttl_interval_expiry() {
    let mode = TtlMode::Interval {
        column_id: 1,
        ttl_seconds: 90 * 86400,
        action: zyron_lifecycle::ttl::TtlAction::Delete,
    };
    let now = 10_000_000_000_000i64; // arbitrary micros
    let old = now - 91 * 86400 * 1_000_000;
    let fresh = now - 10 * 86400 * 1_000_000;
    assert!(is_expired(&mode, old, now));
    assert!(!is_expired(&mode, fresh, now));
}

#[test]
fn legal_hold_registry_whole_table_blocks() {
    struct NeverMatches;
    impl HoldPredicateEvaluator for NeverMatches {
        fn matches(&self, _t: u32, _p: &str, _row: &[u8]) -> zyron_common::Result<bool> {
            Ok(false)
        }
    }
    let reg = LegalHoldRegistry::new();
    assert!(!reg.table_has_hold(7));
    // Manually install a whole-table hold via reload from catalog entries.
    let entries = vec![zyron_catalog::schema::LegalHoldEntry {
        id: 1,
        name: "h".into(),
        table_id: 7,
        predicate_sql: String::new(), // whole table
        reason: "litigation".into(),
        created_at: 1,
        released_at: 0,
    }];
    reg.reload(&entries);
    assert!(reg.table_has_hold(7));
    assert!(reg.row_protected(7, b"row", &NeverMatches));
    let g = reg.generation();
    reg.reload(&[]);
    assert!(reg.generation() > g);
}

#[test]
fn legal_hold_compiled_hold_whole_table_flag() {
    let h = CompiledHold {
        name: "x".into(),
        table_id: 1,
        predicate_sql: "  ".into(),
    };
    assert!(h.whole_table());
}

#[test]
fn tier_cost_multiplier_increases_with_coldness() {
    assert!(StorageTier::Hot.cost_multiplier() < StorageTier::Warm.cost_multiplier());
    assert!(StorageTier::Warm.cost_multiplier() < StorageTier::Cold.cost_multiplier());
    assert!(StorageTier::Cold.cost_multiplier() < StorageTier::Archive.cost_multiplier());
    assert_eq!(StorageTier::parse("cold").unwrap(), StorageTier::Cold);
    assert!(StorageTier::parse("frozen").is_err());
}

#[test]
fn tier_cache_tracks_hits_and_rehydration() {
    let c = TierCache::new(8);
    c.put(1, 2, vec![9, 9, 9]);
    assert_eq!(c.get(1, 2).map(|b| b.len()), Some(3));
    assert!(c.hit_count(1, 2) >= 2);
    assert!(should_rehydrate(c.hit_count(1, 2), 1));
    c.invalidate(1, 2);
    assert!(c.get(1, 2).is_none());
}

#[test]
fn compliance_retention_floor_and_ceiling() {
    let reqs = vec![
        RetentionRequirement::sox_financial(),
        RetentionRequirement::gdpr_pii(),
    ];
    // Below SOX 7-year floor -> violation.
    assert!(validate_retention("financial", 86400, &reqs).is_err());
    // Above GDPR PII 30-day ceiling -> violation.
    assert!(validate_retention("pii", 90 * 86400, &reqs).is_err());
    // Within bounds -> ok.
    assert!(validate_retention("financial", 8 * 365 * 86400, &reqs).is_ok());
    assert!(validate_retention("pii", 10 * 86400, &reqs).is_ok());
    // Unknown category is unconstrained.
    assert!(validate_retention("other", 1, &reqs).is_ok());
}

#[test]
fn cleanup_governor_rate_limits_and_windows() {
    let g = CleanupGovernor::new(100, 100, 50);
    // Within default (no preferred hours) window, grants up to batch size.
    let granted = g.acquire(1000);
    assert!(granted > 0 && granted <= 50);
    // Restrict to an hour that is almost certainly not "now" for both values.
    g.set_preferred_hours(&[]);
    assert!(g.acquire(10) > 0);
}

#[test]
fn audit_chain_detects_tampering() {
    use zyron_catalog::schema::ComplianceLogEntry;
    use zyron_lifecycle::audit_chain::AuditChain;
    let chain = AuditChain::new();
    let e1 = chain.next_entry(0, "s1".into(), 1, 100, "ttl".into());
    let e2 = chain.next_entry(3, "s2".into(), 1, 200, "hold".into());
    let good = vec![e1.clone(), e2.clone()];
    let (n, intact) = AuditChain::verify(&good);
    assert!(intact && n == 2);
    // Tamper with the second entry's detail without recomputing the hash.
    let mut bad = good.clone();
    bad[1] = ComplianceLogEntry {
        detail: "tampered".into(),
        ..bad[1].clone()
    };
    let (_, intact2) = AuditChain::verify(&bad);
    assert!(!intact2);
}

#[test]
fn worm_write_lock_blocks_until_expiry() {
    use zyron_catalog::schema::{LifecycleConfig, TableEntry};
    use zyron_lifecycle::ttl::now_micros;
    let mut e = TableEntry {
        id: zyron_catalog::TableId(1),
        schema_id: zyron_catalog::SchemaId(1),
        name: "t".into(),
        heap_file_id: 200,
        fsm_file_id: 201,
        columns: vec![],
        constraints: vec![],
        created_at: 0,
        versioning_enabled: false,
        scd_type: None,
        system_versioned: false,
        history_table_id: None,
        cdf_enabled: false,
        cdf_retention_days: 0,
        lifecycle: LifecycleConfig::default(),
        columnar: Default::default(),
    };
    assert!(!zyron_lifecycle::worm::write_locked(&e));
    e.lifecycle.retention_lock_until = now_micros() + 3_600_000_000;
    assert!(zyron_lifecycle::worm::write_locked(&e));
    e.lifecycle.retention_lock_until = 0;
    e.lifecycle.immutable = true;
    assert!(zyron_lifecycle::worm::write_locked(&e));
}

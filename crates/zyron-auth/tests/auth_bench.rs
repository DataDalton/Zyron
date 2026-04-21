//! Auth Benchmark Suite
//!
//! Validates password hashing, API keys, JWTs, TOTP, privilege checks,
//! role hierarchy traversal, classification, data masking, time windows,
//! governance analytics, concurrent privilege access, break-glass,
//! admin privileges, SecurityContext, ABAC, tagging, row ownership,
//! session binding, auth rules, temporal/column-level privileges,
//! delegation chains, two-person approval, RLS policies, column masking
//! policies, AES-GCM encryption, security labels (MAC), webhook verification,
//! and crypto SQL functions.
//!
//! Run: cargo test -p zyron-auth --test auth_bench --release -- --nocapture

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

use zyron_auth::abac::{
    AbacEffect, AbacOperator, AbacPolicy, AbacRule, AbacRuleStore, AbacStore, AttributeCondition,
    SessionAttributes,
};
use zyron_auth::auth_rules::{AuthMethod, AuthResolver, AuthRule, ConnectionType};
use zyron_auth::balloon::{self, BalloonParams};
use zyron_auth::breakglass::BreakGlassManager;
use zyron_auth::classification::{ClassificationLevel, ClassificationStore, ColumnClassification};
use zyron_auth::column_security::{MaskingPolicy, MaskingPolicyStore};
use zyron_auth::context::SecurityContext;
use zyron_auth::credentials::{
    ApiKeyCredential, JwtAlgorithm, JwtClaims, JwtCredential, PasswordCredential, TotpCredential,
};
use zyron_auth::encryption::{
    EncryptionAlgorithm, KeyStore, LocalKeyStore, decrypt_value, encrypt_value,
};
use zyron_auth::governance::{
    DelegationEdge, GovernanceManager, PrivilegeAnalytics, TwoPersonOperation, TwoPersonRule,
};
use zyron_auth::masking::{self, MaskFunction};
use zyron_auth::privilege::{
    GrantEntry, ObjectType, PrivilegeDecision, PrivilegeState, PrivilegeStore, PrivilegeType,
};
use zyron_auth::rls::{PolicyType, RlsCommand, RlsPolicy, RlsPolicyStore};
use zyron_auth::role::{RoleHierarchy, RoleId, RoleMembership, UserId};
use zyron_auth::row_ownership::{RowOwnershipConfig, RowOwnershipStore};
use zyron_auth::security_label::{MandatoryAccessControl, SecurityLabel, SecurityLevel};
use zyron_auth::session_binding::{QueryLimitStore, QueryLimits, TimeWindow};
use zyron_auth::tagging::{ObjectTag, TagStore};
use zyron_auth::webhook;
use zyron_bench_harness::{check_performance, init, tprintln, validate_metric};

static BENCHMARK_LOCK: Mutex<()> = Mutex::new(());

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Creates a basic GrantEntry with minimal fields.
fn make_grant(grantee: u32, privilege: PrivilegeType, object_id: u32) -> GrantEntry {
    GrantEntry {
        grantee: RoleId(grantee),
        privilege,
        object_type: ObjectType::Table,
        object_id,
        columns: None,
        state: PrivilegeState::Grant,
        with_grant_option: false,
        granted_by: RoleId(0),
        valid_from: None,
        valid_until: None,
        time_window: None,
        object_pattern: None,
        no_inherit: false,
        mask_function: None,
    }
}

/// Creates a DENY GrantEntry.
fn make_deny(grantee: u32, privilege: PrivilegeType, object_id: u32) -> GrantEntry {
    GrantEntry {
        grantee: RoleId(grantee),
        privilege,
        object_type: ObjectType::Table,
        object_id,
        columns: None,
        state: PrivilegeState::Deny,
        with_grant_option: false,
        granted_by: RoleId(0),
        valid_from: None,
        valid_until: None,
        time_window: None,
        object_pattern: None,
        no_inherit: false,
        mask_function: None,
    }
}

// ---------------------------------------------------------------------------
// 1. Balloon Hash Timing
// ---------------------------------------------------------------------------

#[test]
fn test_balloon_hash_timing() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- Balloon Hash Timing ---");

    let params = BalloonParams::test();
    let password = "benchmark_password_2024";
    let salt = b"benchmark_salt16";

    let start = Instant::now();
    let hash = balloon::balloon_hash(password.as_bytes(), salt, &params);
    let elapsed = start.elapsed();

    tprintln!(
        "  Params: space_cost={}, time_cost={}, delta={}",
        params.space_cost,
        params.time_cost,
        params.delta
    );
    tprintln!("  Hash time: {:.3}ms", elapsed.as_secs_f64() * 1000.0);
    tprintln!("  Hash output: {} bytes", hash.len());

    assert_eq!(hash.len(), 32, "Balloon hash must produce 32 bytes");

    // Test params (1024 blocks) keep the AES-based compression step fast
    // enough for unit-test turnaround. Production params (64MB, 3 rounds)
    // are validated in test_password_authentication.
    let elapsed_us = elapsed.as_micros() as f64;
    tprintln!("  Elapsed: {:.0} us", elapsed_us);

    validate_metric(
        "balloon_hash_timing",
        "hash_time_us",
        vec![elapsed_us],
        10.0,
        true,
    );
    check_performance(
        "balloon_hash_timing",
        "memory_hard_minimum",
        elapsed_us,
        10.0,
        true,
    );

    // Verify determinism: same inputs produce same output.
    let hash2 = balloon::balloon_hash(password.as_bytes(), salt, &params);
    assert_eq!(hash, hash2, "Balloon hash must be deterministic");
    tprintln!("  Determinism: verified");
}

// ---------------------------------------------------------------------------
// 2. API Key Throughput
// ---------------------------------------------------------------------------

#[test]
fn test_api_key_throughput() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- API Key Generate + Verify Throughput ---");

    let count = 10_000;

    let start = Instant::now();
    let mut keys: Vec<(ApiKeyCredential, String)> = Vec::with_capacity(count);
    for _ in 0..count {
        keys.push(ApiKeyCredential::generate());
    }
    let gen_elapsed = start.elapsed();

    let gen_ops = count as f64 / gen_elapsed.as_secs_f64();
    tprintln!(
        "  Generate: {:.0} ops/sec ({:.1}ms for {} keys)",
        gen_ops,
        gen_elapsed.as_millis(),
        count
    );

    let start = Instant::now();
    let mut verified = 0u64;
    for (cred, full_key) in &keys {
        if cred.verify(full_key) {
            verified += 1;
        }
    }
    let verify_elapsed = start.elapsed();

    let verify_ops = count as f64 / verify_elapsed.as_secs_f64();
    tprintln!(
        "  Verify: {:.0} ops/sec ({:.1}ms for {} keys)",
        verify_ops,
        verify_elapsed.as_millis(),
        count
    );
    tprintln!("  Verified: {}/{}", verified, count);

    assert_eq!(
        verified, count as u64,
        "All API keys must verify successfully"
    );

    let combined_ops = (count * 2) as f64 / (gen_elapsed + verify_elapsed).as_secs_f64();
    tprintln!("  Combined: {:.0} ops/sec", combined_ops);

    validate_metric(
        "api_key_throughput",
        "combined_ops_sec",
        vec![combined_ops],
        100_000.0,
        true,
    );
    check_performance(
        "api_key_throughput",
        "combined_ops_sec",
        combined_ops,
        100_000.0,
        true,
    );
}

// ---------------------------------------------------------------------------
// 3. JWT Encode/Decode Throughput
// ---------------------------------------------------------------------------

#[test]
fn test_jwt_encode_decode_throughput() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- JWT Encode/Decode Throughput ---");

    let count = 10_000;
    let secret = vec![0xABu8; 32];
    let jwt =
        JwtCredential::new(secret, JwtAlgorithm::Hs256).expect("JWT credential creation failed");

    let claims = JwtClaims {
        sub: "user:1000".to_string(),
        iss: None,
        exp: u64::MAX,
        iat: 1700000000,
        roles: vec!["admin".to_string(), "reader".to_string()],
        custom: std::collections::HashMap::new(),
    };

    // Encode phase.
    let start = Instant::now();
    let mut tokens: Vec<String> = Vec::with_capacity(count);
    for _ in 0..count {
        let token = jwt.encode(&claims).expect("JWT encode failed");
        tokens.push(token);
    }
    let encode_elapsed = start.elapsed();

    let encode_ops = count as f64 / encode_elapsed.as_secs_f64();
    tprintln!(
        "  Encode: {:.0} ops/sec ({:.1}ms)",
        encode_ops,
        encode_elapsed.as_millis()
    );

    // Decode phase.
    let start = Instant::now();
    let mut decoded = 0u64;
    for token in &tokens {
        let result = jwt.decode(token).expect("JWT decode failed");
        if result.sub == "user:1000" {
            decoded += 1;
        }
    }
    let decode_elapsed = start.elapsed();

    let decode_ops = count as f64 / decode_elapsed.as_secs_f64();
    tprintln!(
        "  Decode: {:.0} ops/sec ({:.1}ms)",
        decode_ops,
        decode_elapsed.as_millis()
    );
    tprintln!("  Decoded: {}/{}", decoded, count);

    assert_eq!(
        decoded, count as u64,
        "All JWT tokens must decode successfully"
    );

    let combined_ops = (count * 2) as f64 / (encode_elapsed + decode_elapsed).as_secs_f64();
    tprintln!("  Combined: {:.0} ops/sec", combined_ops);

    validate_metric(
        "jwt_throughput",
        "combined_ops_sec",
        vec![combined_ops],
        50_000.0,
        true,
    );
    check_performance(
        "jwt_throughput",
        "combined_ops_sec",
        combined_ops,
        50_000.0,
        true,
    );
}

// ---------------------------------------------------------------------------
// 4. TOTP Throughput
// ---------------------------------------------------------------------------

#[test]
fn test_totp_throughput() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- TOTP Generate + Verify Throughput ---");

    let count = 100_000;
    let totp = TotpCredential::from_secret(vec![0xBB; 20]);
    let timestamp = 1700000000u64;

    // Generate phase.
    let start = Instant::now();
    let mut last_code = String::new();
    for i in 0..count {
        let ts = timestamp + (i as u64) * 30;
        last_code = totp.generate_code(ts);
    }
    let gen_elapsed = start.elapsed();

    let gen_ops = count as f64 / gen_elapsed.as_secs_f64();
    tprintln!(
        "  Generate: {:.0} ops/sec ({:.1}ms)",
        gen_ops,
        gen_elapsed.as_millis()
    );
    tprintln!("  Last code: {}", last_code);

    // Verify phase.
    let start = Instant::now();
    let mut verified = 0u64;
    for i in 0..count {
        let ts = timestamp + (i as u64) * 30;
        let code = totp.generate_code(ts);
        if totp.verify(&code, ts) {
            verified += 1;
        }
    }
    let verify_elapsed = start.elapsed();

    let verify_ops = count as f64 / verify_elapsed.as_secs_f64();
    tprintln!(
        "  Verify: {:.0} ops/sec ({:.1}ms)",
        verify_ops,
        verify_elapsed.as_millis()
    );
    tprintln!("  Verified: {}/{}", verified, count);

    assert_eq!(
        verified, count as u64,
        "All TOTP codes must verify successfully"
    );

    let combined_ops = (count * 2) as f64 / (gen_elapsed + verify_elapsed).as_secs_f64();
    tprintln!("  Combined: {:.0} ops/sec", combined_ops);

    validate_metric(
        "totp_throughput",
        "combined_ops_sec",
        vec![combined_ops],
        500_000.0,
        true,
    );
    check_performance(
        "totp_throughput",
        "combined_ops_sec",
        combined_ops,
        500_000.0,
        true,
    );
}

// ---------------------------------------------------------------------------
// 5. Privilege Check Throughput
// ---------------------------------------------------------------------------

#[test]
fn test_privilege_check_throughput() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- Privilege Check Throughput ---");

    let store = PrivilegeStore::new();
    let now = 1700000000u64;

    // Load 100 grants across different objects and roles.
    let mut entries = Vec::with_capacity(100);
    for i in 0..100u32 {
        entries.push(make_grant(i % 10, PrivilegeType::Select, i));
    }
    store.load(entries);

    let effective_roles = vec![RoleId(0), RoleId(1), RoleId(2)];
    let count = 1_000_000;

    let start = Instant::now();
    let mut allowed = 0u64;
    for i in 0..count {
        let object_id = (i % 100) as u32;
        let decision = store.check_privilege(
            &effective_roles,
            PrivilegeType::Select,
            ObjectType::Table,
            object_id,
            None,
            now,
        );
        if decision == PrivilegeDecision::Allow {
            allowed += 1;
        }
    }
    let elapsed = start.elapsed();

    let ops_per_sec = count as f64 / elapsed.as_secs_f64();
    tprintln!(
        "  Throughput: {:.0} ops/sec ({:.1}ms for {} checks)",
        ops_per_sec,
        elapsed.as_millis(),
        count
    );
    tprintln!("  Allowed: {}/{}", allowed, count);

    validate_metric(
        "privilege_check",
        "ops_sec",
        vec![ops_per_sec],
        1_000_000.0,
        true,
    );
    check_performance("privilege_check", "ops_sec", ops_per_sec, 1_000_000.0, true);
}

// ---------------------------------------------------------------------------
// 6. Privilege DENY Wins
// ---------------------------------------------------------------------------

#[test]
fn test_privilege_deny_wins() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- Privilege DENY Wins ---");

    let store = PrivilegeStore::new();
    let now = 1700000000u64;

    // Grant SELECT on table 1 to role 1, then DENY SELECT on the same table.
    let mut entries = Vec::new();
    entries.push(make_grant(1, PrivilegeType::Select, 1));
    entries.push(make_deny(1, PrivilegeType::Select, 1));

    // Grant INSERT on table 2 to role 1 (no deny).
    entries.push(make_grant(1, PrivilegeType::Insert, 2));

    store.load(entries);

    let effective_roles = vec![RoleId(1)];

    // Verify DENY wins for table 1 SELECT.
    let decision = store.check_privilege(
        &effective_roles,
        PrivilegeType::Select,
        ObjectType::Table,
        1,
        None,
        now,
    );
    assert_eq!(
        decision,
        PrivilegeDecision::Denied,
        "DENY must win over GRANT"
    );
    tprintln!("  Table 1 SELECT: {:?} (DENY wins)", decision);

    // Verify INSERT on table 2 is allowed.
    let decision = store.check_privilege(
        &effective_roles,
        PrivilegeType::Insert,
        ObjectType::Table,
        2,
        None,
        now,
    );
    assert_eq!(
        decision,
        PrivilegeDecision::Allow,
        "INSERT should be allowed"
    );
    tprintln!("  Table 2 INSERT: {:?}", decision);

    // Measure throughput of deny-wins checks.
    let count = 1_000_000;
    let start = Instant::now();
    let mut denied = 0u64;
    for _ in 0..count {
        let d = store.check_privilege(
            &effective_roles,
            PrivilegeType::Select,
            ObjectType::Table,
            1,
            None,
            now,
        );
        if d == PrivilegeDecision::Denied {
            denied += 1;
        }
    }
    let elapsed = start.elapsed();

    let ops_per_sec = count as f64 / elapsed.as_secs_f64();
    tprintln!(
        "  DENY check throughput: {:.0} ops/sec ({:.1}ms)",
        ops_per_sec,
        elapsed.as_millis()
    );
    tprintln!("  Denied: {}/{}", denied, count);

    assert_eq!(denied, count as u64, "All checks must return Denied");

    validate_metric(
        "privilege_deny_wins",
        "deny_ops_sec",
        vec![ops_per_sec],
        1_000_000.0,
        true,
    );
    check_performance(
        "privilege_deny_wins",
        "deny_ops_sec",
        ops_per_sec,
        1_000_000.0,
        true,
    );
}

// ---------------------------------------------------------------------------
// 7. Role Hierarchy Traversal
// ---------------------------------------------------------------------------

#[test]
fn test_role_hierarchy_traversal() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- Role Hierarchy Traversal ---");

    let hierarchy = RoleHierarchy::new();

    // Build a chain 10 levels deep: role 0 -> role 1 -> ... -> role 9.
    let mut memberships = Vec::new();
    for i in 0..10u32 {
        memberships.push(RoleMembership {
            member_id: RoleId(i),
            parent_id: RoleId(i + 1),
            admin_option: false,
            inherit: true,
            granted_by: RoleId(0),
        });
    }
    hierarchy.load(&memberships);

    // Verify correctness: role 0 should inherit from all 11 roles (0..=10).
    let effective = hierarchy.effective_roles(RoleId(0));
    tprintln!(
        "  Depth: 10 levels, effective roles for role 0: {}",
        effective.len()
    );
    assert_eq!(
        effective.len(),
        11,
        "Role 0 should see 11 effective roles (self + 10 ancestors)"
    );

    let count = 100_000;
    let start = Instant::now();
    for _ in 0..count {
        let _ = hierarchy.effective_roles(RoleId(0));
    }
    let elapsed = start.elapsed();

    let per_traversal_us = elapsed.as_secs_f64() * 1_000_000.0 / count as f64;
    let ops_per_sec = count as f64 / elapsed.as_secs_f64();
    tprintln!("  Per traversal: {:.2} us", per_traversal_us);
    tprintln!(
        "  Throughput: {:.0} traversals/sec ({:.1}ms total)",
        ops_per_sec,
        elapsed.as_millis()
    );

    // Lower is better for latency: target <10us per traversal.
    validate_metric(
        "role_hierarchy",
        "traversal_us",
        vec![per_traversal_us],
        10.0,
        false,
    );
    check_performance(
        "role_hierarchy",
        "traversal_us",
        per_traversal_us,
        10.0,
        false,
    );
}

// ---------------------------------------------------------------------------
// 8. Classification Check Throughput
// ---------------------------------------------------------------------------

#[test]
fn test_classification_check_throughput() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- Classification Check Throughput ---");

    let store = ClassificationStore::new();

    // Set up 100 classifications across 10 tables, 10 columns each.
    let mut entries = Vec::with_capacity(100);
    for table in 0..10u32 {
        for col in 0..10u16 {
            let level = match (table + col as u32) % 4 {
                0 => ClassificationLevel::Public,
                1 => ClassificationLevel::Internal,
                2 => ClassificationLevel::Confidential,
                _ => ClassificationLevel::Restricted,
            };
            entries.push(ColumnClassification {
                table_id: table,
                column_id: col,
                level,
            });
        }
    }
    store.load(entries);

    let count = 1_000_000;
    let start = Instant::now();
    let mut passed = 0u64;
    for i in 0..count {
        let table_id = (i % 10) as u32;
        let column_id = ((i / 10) % 10) as u16;
        if store.check_clearance(ClassificationLevel::Confidential, table_id, column_id) {
            passed += 1;
        }
    }
    let elapsed = start.elapsed();

    let ops_per_sec = count as f64 / elapsed.as_secs_f64();
    tprintln!(
        "  Throughput: {:.0} ops/sec ({:.1}ms for {} checks)",
        ops_per_sec,
        elapsed.as_millis(),
        count
    );
    tprintln!("  Passed: {}/{}", passed, count);

    validate_metric(
        "classification_check",
        "ops_sec",
        vec![ops_per_sec],
        10_000_000.0,
        true,
    );
    check_performance(
        "classification_check",
        "ops_sec",
        ops_per_sec,
        10_000_000.0,
        true,
    );
}

// ---------------------------------------------------------------------------
// 9. Data Mask Throughput
// ---------------------------------------------------------------------------

#[test]
fn test_data_mask_throughput() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- Data Mask Throughput (Email) ---");

    let count = 100_000;
    let email = "longusername@example.com";

    let mut buf = String::with_capacity(64);
    let start = Instant::now();
    for _ in 0..count {
        masking::apply_mask(email, &MaskFunction::Email, &mut buf);
    }
    let last_masked = buf.clone();
    let elapsed = start.elapsed();

    let ops_per_sec = count as f64 / elapsed.as_secs_f64();
    tprintln!("  Input: {}", email);
    tprintln!("  Masked: {}", last_masked);
    tprintln!(
        "  Throughput: {:.0} ops/sec ({:.1}ms for {} masks)",
        ops_per_sec,
        elapsed.as_millis(),
        count
    );

    assert_eq!(
        last_masked, "l***********@example.com",
        "Email mask output mismatch"
    );

    validate_metric(
        "data_mask",
        "email_ops_sec",
        vec![ops_per_sec],
        1_000_000.0,
        true,
    );
    check_performance("data_mask", "email_ops_sec", ops_per_sec, 1_000_000.0, true);
}

// ---------------------------------------------------------------------------
// 10. Time Window Evaluation
// ---------------------------------------------------------------------------

#[test]
fn test_time_window_evaluation() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- Time Window Evaluation ---");

    let window = TimeWindow::business_hours();
    // 2024-01-01 12:00 UTC is a Monday at noon.
    let monday_noon = 1704067200u64 + 12 * 3600;

    // Correctness check.
    assert!(
        window.is_active(monday_noon),
        "Monday noon should be within business hours"
    );
    tprintln!(
        "  Window: start_hour={}, end_hour={}, days=0b{:07b}",
        window.start_hour,
        window.end_hour,
        window.days
    );

    let count = 1_000_000;
    let start = Instant::now();
    let mut active = 0u64;
    for i in 0..count {
        // Vary timestamp across days and hours.
        let ts = monday_noon + (i as u64) * 3600;
        if window.is_active(ts) {
            active += 1;
        }
    }
    let elapsed = start.elapsed();

    let ops_per_sec = count as f64 / elapsed.as_secs_f64();
    tprintln!(
        "  Throughput: {:.0} ops/sec ({:.1}ms for {} evaluations)",
        ops_per_sec,
        elapsed.as_millis(),
        count
    );
    tprintln!("  Active: {}/{}", active, count);

    validate_metric(
        "time_window",
        "eval_ops_sec",
        vec![ops_per_sec],
        10_000_000.0,
        true,
    );
    check_performance(
        "time_window",
        "eval_ops_sec",
        ops_per_sec,
        10_000_000.0,
        true,
    );
}

// ---------------------------------------------------------------------------
// 11. Governance Usage Recording
// ---------------------------------------------------------------------------

#[test]
fn test_governance_usage_recording() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- Governance Usage Recording ---");

    let analytics = PrivilegeAnalytics::new();
    let count = 1_000_000;

    let start = Instant::now();
    for i in 0..count {
        let role_id = RoleId((i % 100) as u32);
        let object_id = (i % 1000) as u32;
        analytics.record_usage(role_id, PrivilegeType::Select, ObjectType::Table, object_id);
    }
    let elapsed = start.elapsed();

    let ops_per_sec = count as f64 / elapsed.as_secs_f64();
    tprintln!(
        "  Throughput: {:.0} ops/sec ({:.1}ms for {} records)",
        ops_per_sec,
        elapsed.as_millis(),
        count
    );

    // Spot check: role 0 should have entries.
    let report = analytics.usage_report(RoleId(0));
    tprintln!("  Role 0 distinct entries: {}", report.len());
    assert!(!report.is_empty(), "Role 0 should have usage records");

    validate_metric(
        "governance_usage",
        "record_ops_sec",
        vec![ops_per_sec],
        5_000_000.0,
        true,
    );
    check_performance(
        "governance_usage",
        "record_ops_sec",
        ops_per_sec,
        5_000_000.0,
        true,
    );
}

// ---------------------------------------------------------------------------
// 12. Concurrent Privilege Checks
// ---------------------------------------------------------------------------

#[test]
fn test_concurrent_privilege_checks() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- Concurrent Privilege Checks ---");

    let store = std::sync::Arc::new(PrivilegeStore::new());
    let now = 1700000000u64;

    // Load grants: 100 entries.
    let mut entries = Vec::with_capacity(100);
    for i in 0..100u32 {
        entries.push(make_grant(i % 10, PrivilegeType::Select, i));
    }
    store.load(entries);

    let thread_count = 8;
    let checks_per_thread = 500_000;

    // Single-threaded baseline. Snapshot the grants Rcu once (as a query
    // planner would at plan time), then do pure HashMap lookups per row.
    let effective_roles = vec![RoleId(0), RoleId(1), RoleId(2)];
    let grants_snap = store.grants_snapshot();
    let baseline_start = Instant::now();
    let mut baseline_allowed = 0u64;
    for i in 0..checks_per_thread {
        let object_id = (i % 100) as u32;
        let key = (ObjectType::Table as u8, object_id);
        let d = if let Some(entries) = grants_snap.get(&key) {
            PrivilegeStore::check_entries_static(
                entries,
                &effective_roles,
                PrivilegeType::Select,
                None,
                now,
            )
        } else {
            PrivilegeDecision::Unset
        };
        if d == PrivilegeDecision::Allow {
            baseline_allowed += 1;
        }
    }
    let baseline_elapsed = baseline_start.elapsed();
    let baseline_ops = checks_per_thread as f64 / baseline_elapsed.as_secs_f64();
    tprintln!(
        "  Single-thread: {:.0} ops/sec ({:.1}ms)",
        baseline_ops,
        baseline_elapsed.as_millis()
    );

    // Multi-threaded run. Each thread loads one Rcu snapshot (one atomic
    // refcount increment), then does pure HashMap lookups with zero
    // contention. This mirrors real query execution where the planner
    // snapshots once and the executor checks per-row.
    let start = Instant::now();
    let mut handles = Vec::with_capacity(thread_count);
    for _ in 0..thread_count {
        let store_clone = store.clone();
        let roles = effective_roles.clone();
        let handle = std::thread::spawn(move || {
            let snap = store_clone.grants_snapshot();
            let mut local_allowed = 0u64;
            for i in 0..checks_per_thread {
                let object_id = (i % 100) as u32;
                let key = (ObjectType::Table as u8, object_id);
                let d = if let Some(entries) = snap.get(&key) {
                    PrivilegeStore::check_entries_static(
                        entries,
                        &roles,
                        PrivilegeType::Select,
                        None,
                        now,
                    )
                } else {
                    PrivilegeDecision::Unset
                };
                if d == PrivilegeDecision::Allow {
                    local_allowed += 1;
                }
            }
            local_allowed
        });
        handles.push(handle);
    }

    let mut total_allowed = 0u64;
    for handle in handles {
        total_allowed += handle.join().expect("Thread panicked");
    }
    let elapsed = start.elapsed();

    let total_checks = (thread_count * checks_per_thread) as f64;
    let concurrent_ops = total_checks / elapsed.as_secs_f64();
    tprintln!(
        "  Multi-thread ({} threads): {:.0} ops/sec ({:.1}ms)",
        thread_count,
        concurrent_ops,
        elapsed.as_millis()
    );
    tprintln!("  Total allowed: {}", total_allowed);
    tprintln!("  Baseline allowed (single): {}", baseline_allowed);

    // Each thread should get the same ratio of allowed checks.
    let expected_per_thread = baseline_allowed;
    let expected_total = expected_per_thread * thread_count as u64;
    assert_eq!(
        total_allowed, expected_total,
        "Concurrent results must match single-threaded results"
    );

    // Check scaling: multi-threaded throughput should be at least 2x single-threaded.
    // Conservative target to account for scheduling overhead.
    let scaling_factor = concurrent_ops / baseline_ops;
    tprintln!("  Scaling factor: {:.2}x (target: >2.0x)", scaling_factor);

    validate_metric(
        "concurrent_privilege",
        "scaling_factor",
        vec![scaling_factor],
        2.0,
        true,
    );
    check_performance(
        "concurrent_privilege",
        "scaling_factor",
        scaling_factor,
        2.0,
        true,
    );
}

// ---------------------------------------------------------------------------
// Additional helpers
// ---------------------------------------------------------------------------

fn make_grant_on(
    grantee: u32,
    privilege: PrivilegeType,
    object_type: ObjectType,
    object_id: u32,
) -> GrantEntry {
    GrantEntry {
        grantee: RoleId(grantee),
        privilege,
        object_type,
        object_id,
        columns: None,
        state: PrivilegeState::Grant,
        with_grant_option: false,
        granted_by: RoleId(0),
        valid_from: None,
        valid_until: None,
        time_window: None,
        object_pattern: None,
        no_inherit: false,
        mask_function: None,
    }
}

fn make_column_grant(
    grantee: u32,
    privilege: PrivilegeType,
    object_id: u32,
    columns: Vec<u16>,
) -> GrantEntry {
    GrantEntry {
        grantee: RoleId(grantee),
        privilege,
        object_type: ObjectType::Table,
        object_id,
        columns: Some(columns),
        state: PrivilegeState::Grant,
        with_grant_option: false,
        granted_by: RoleId(0),
        valid_from: None,
        valid_until: None,
        time_window: None,
        object_pattern: None,
        no_inherit: false,
        mask_function: None,
    }
}

fn make_temporal_grant(
    grantee: u32,
    privilege: PrivilegeType,
    object_id: u32,
    valid_from: Option<u64>,
    valid_until: Option<u64>,
) -> GrantEntry {
    GrantEntry {
        grantee: RoleId(grantee),
        privilege,
        object_type: ObjectType::Table,
        object_id,
        columns: None,
        state: PrivilegeState::Grant,
        with_grant_option: false,
        granted_by: RoleId(0),
        valid_from,
        valid_until,
        time_window: None,
        object_pattern: None,
        no_inherit: false,
        mask_function: None,
    }
}

fn make_attributes(role_id: RoleId) -> SessionAttributes {
    SessionAttributes {
        role_id,
        department: Some("engineering".to_string()),
        region: Some("us-east".to_string()),
        clearance: ClassificationLevel::Internal,
        ip_address: "10.0.0.1".to_string(),
        connection_time: 1700000000,
        custom: HashMap::new(),
    }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ===========================================================================
// 13. Password Authentication Correctness
// ===========================================================================

#[test]
fn test_password_authentication() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- Password Authentication Correctness ---");

    let params = BalloonParams::test();

    // Hash and verify correct password.
    let cred = PasswordCredential::from_plaintext_with_params("correct_password", &params)
        .expect("hash should succeed");
    assert!(
        cred.verify("correct_password").expect("verify should work"),
        "Correct password must verify"
    );
    tprintln!("  Correct password: verified");

    // Wrong password fails.
    assert!(
        !cred.verify("wrong_password").expect("verify should work"),
        "Wrong password must not verify"
    );
    tprintln!("  Wrong password: rejected");

    // Stored hash roundtrip.
    let stored = cred.as_stored().to_string();
    let restored = PasswordCredential::from_stored(stored);
    assert!(
        restored
            .verify("correct_password")
            .expect("verify should work"),
    );
    tprintln!("  Stored hash roundtrip: verified");

    // Timing attack resistance: both correct and wrong should take similar time.
    let iterations = 10;
    let mut correct_times = Vec::with_capacity(iterations);
    let mut wrong_times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        let _ = cred.verify("correct_password");
        correct_times.push(start.elapsed().as_nanos() as f64);
        let start = Instant::now();
        let _ = cred.verify("wrong_password");
        wrong_times.push(start.elapsed().as_nanos() as f64);
    }
    let avg_correct: f64 = correct_times.iter().sum::<f64>() / iterations as f64;
    let avg_wrong: f64 = wrong_times.iter().sum::<f64>() / iterations as f64;
    let ratio = if avg_correct > avg_wrong {
        avg_correct / avg_wrong
    } else {
        avg_wrong / avg_correct
    };
    tprintln!(
        "  Timing: correct={:.0}ns, wrong={:.0}ns, ratio={:.2}",
        avg_correct,
        avg_wrong,
        ratio
    );
    assert!(
        ratio < 3.0,
        "Timing ratio {:.2} exceeds 3x tolerance",
        ratio
    );

    // Benchmark hash time with default production params (64MB, 3 rounds, delta 3).
    let bench_params = BalloonParams::default();
    let start = Instant::now();
    let prod_cred =
        PasswordCredential::from_plaintext_with_params("benchmark_password", &bench_params)
            .expect("hash");
    let hash_ms = start.elapsed().as_secs_f64() * 1000.0;
    tprintln!(
        "  Hash time: {:.1}ms (target: >=80ms, params: 64MB/3 rounds)",
        hash_ms
    );
    validate_metric("password_auth", "hash_time_ms", vec![hash_ms], 80.0, true);
    check_performance("password_auth", "hash_time_ms", hash_ms, 80.0, true);

    // Benchmark verify time.
    let start = Instant::now();
    let _ = prod_cred.verify("benchmark_password");
    let verify_ms = start.elapsed().as_secs_f64() * 1000.0;
    tprintln!("  Verify time: {:.1}ms (target: >=80ms)", verify_ms);
    validate_metric(
        "password_auth",
        "verify_time_ms",
        vec![verify_ms],
        80.0,
        true,
    );
    check_performance("password_auth", "verify_time_ms", verify_ms, 80.0, true);
}

// ===========================================================================
// 14. Challenge-Response Authentication
// ===========================================================================

#[test]
fn test_challenge_response_authentication() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- Challenge-Response Authentication ---");

    let password = "challenge_response_test_password";
    let params = BalloonParams::test();

    // Server hashes password with test params for correctness check.
    let start = Instant::now();
    let server_cred =
        PasswordCredential::from_plaintext_with_params(password, &params).expect("hash");
    let hash_elapsed = start.elapsed();

    // Client sends proof, server verifies.
    let start = Instant::now();
    let verified = server_cred.verify(password).expect("verify");
    let verify_elapsed = start.elapsed();
    assert!(verified, "Correct proof must authenticate");
    tprintln!("  Correct proof: authenticated");

    // Tampered proof.
    assert!(
        !server_cred.verify("tampered_proof").expect("verify"),
        "Tampered proof must fail"
    );
    tprintln!("  Tampered proof: rejected");

    let verify_ms = verify_elapsed.as_secs_f64() * 1000.0;
    tprintln!(
        "  Hash: {:.1}ms, Verify: {:.1}ms (target: verify <200ms)",
        hash_elapsed.as_secs_f64() * 1000.0,
        verify_ms
    );
    validate_metric(
        "challenge_response",
        "verify_latency_ms",
        vec![verify_ms],
        200.0,
        false,
    );
    check_performance(
        "challenge_response",
        "verify_latency_ms",
        verify_ms,
        200.0,
        false,
    );
}

// ===========================================================================
// 15. Role Hierarchy Correctness
// ===========================================================================

#[test]
fn test_role_hierarchy_correctness() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- Role Hierarchy Correctness ---");

    // admin(1) -> manager(2) -> analyst(3) -> viewer(4)
    let hierarchy = RoleHierarchy::new();
    hierarchy.load(&[
        RoleMembership {
            member_id: RoleId(4),
            parent_id: RoleId(3),
            admin_option: false,
            inherit: true,
            granted_by: RoleId(0),
        },
        RoleMembership {
            member_id: RoleId(3),
            parent_id: RoleId(2),
            admin_option: false,
            inherit: true,
            granted_by: RoleId(0),
        },
        RoleMembership {
            member_id: RoleId(2),
            parent_id: RoleId(1),
            admin_option: false,
            inherit: true,
            granted_by: RoleId(0),
        },
    ]);

    // Grant analyst to user_a(100).
    hierarchy
        .add_membership(RoleId(100), RoleId(3), true)
        .expect("add membership");
    let effective = hierarchy.effective_roles(RoleId(100));
    assert!(effective.contains(&RoleId(100)));
    assert!(effective.contains(&RoleId(3)), "should have analyst");
    assert!(effective.contains(&RoleId(2)), "should inherit manager");
    assert!(effective.contains(&RoleId(1)), "should inherit admin");
    assert!(!effective.contains(&RoleId(4)), "should NOT have viewer");
    tprintln!("  user_a with analyst: has analyst,manager,admin. Not viewer.");

    assert!(hierarchy.is_member_of(RoleId(100), RoleId(3)));
    assert!(hierarchy.is_member_of(RoleId(100), RoleId(1)));
    assert!(!hierarchy.is_member_of(RoleId(100), RoleId(4)));

    // Grant manager directly.
    hierarchy
        .add_membership(RoleId(100), RoleId(2), true)
        .expect("add manager");
    let effective2 = hierarchy.effective_roles(RoleId(100));
    assert!(effective2.contains(&RoleId(2)));
    assert!(effective2.contains(&RoleId(1)));
    tprintln!("  After granting manager: user_a has manager+admin");

    // Revoke direct manager.
    hierarchy.remove_membership(RoleId(100), RoleId(2));
    let effective3 = hierarchy.effective_roles(RoleId(100));
    assert!(effective3.contains(&RoleId(3)), "still has analyst");
    assert!(
        effective3.contains(&RoleId(2)),
        "still inherits manager through analyst"
    );
    tprintln!("  After revoking direct manager: still inherits via analyst");

    // Performance.
    let count = 100_000;
    let start = Instant::now();
    for _ in 0..count {
        let _ = hierarchy.effective_roles(RoleId(100));
    }
    let elapsed = start.elapsed();
    let per_ns = elapsed.as_nanos() as f64 / count as f64;
    tprintln!("  Traversal: {:.1}ns/op", per_ns);
    validate_metric(
        "role_hierarchy_correctness",
        "traversal_ns",
        vec![per_ns],
        200.0,
        false,
    );
    check_performance(
        "role_hierarchy_correctness",
        "traversal_ns",
        per_ns,
        200.0,
        false,
    );
}

// ===========================================================================
// 16. Privilege Grant/Revoke Correctness
// ===========================================================================

#[test]
fn test_privilege_grant_revoke() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- Privilege Grant/Revoke ---");

    let store = PrivilegeStore::new();
    let now = now_secs();
    let analyst = vec![RoleId(3)];
    let users_tbl = 100u32;

    // GRANT SELECT ON users TO analyst.
    store
        .grant(make_grant(3, PrivilegeType::Select, users_tbl))
        .expect("grant");
    let d = store.check_privilege(
        &analyst,
        PrivilegeType::Select,
        ObjectType::Table,
        users_tbl,
        None,
        now,
    );
    assert_eq!(d, PrivilegeDecision::Allow);
    tprintln!("  GRANT SELECT: allowed");

    // Cannot INSERT.
    let d = store.check_privilege(
        &analyst,
        PrivilegeType::Insert,
        ObjectType::Table,
        users_tbl,
        None,
        now,
    );
    assert_ne!(d, PrivilegeDecision::Allow);
    tprintln!("  INSERT without grant: denied");

    // GRANT INSERT.
    store
        .grant(make_grant(3, PrivilegeType::Insert, users_tbl))
        .expect("grant");
    let d = store.check_privilege(
        &analyst,
        PrivilegeType::Insert,
        ObjectType::Table,
        users_tbl,
        None,
        now,
    );
    assert_eq!(d, PrivilegeDecision::Allow);
    tprintln!("  GRANT INSERT: allowed");

    // REVOKE ALL.
    store.revoke(
        RoleId(3),
        PrivilegeType::Select,
        ObjectType::Table,
        users_tbl,
    );
    store.revoke(
        RoleId(3),
        PrivilegeType::Insert,
        ObjectType::Table,
        users_tbl,
    );
    let d = store.check_privilege(
        &analyst,
        PrivilegeType::Select,
        ObjectType::Table,
        users_tbl,
        None,
        now,
    );
    assert_ne!(d, PrivilegeDecision::Allow);
    let d = store.check_privilege(
        &analyst,
        PrivilegeType::Insert,
        ObjectType::Table,
        users_tbl,
        None,
        now,
    );
    assert_ne!(d, PrivilegeDecision::Allow);
    tprintln!("  REVOKE ALL: both denied");
}

// ===========================================================================
// 17. Privilege Inheritance Through Hierarchy
// ===========================================================================

#[test]
fn test_privilege_inheritance() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- Privilege Inheritance ---");

    let hierarchy = RoleHierarchy::new();
    hierarchy.load(&[RoleMembership {
        member_id: RoleId(3),
        parent_id: RoleId(2),
        admin_option: false,
        inherit: true,
        granted_by: RoleId(0),
    }]);

    let store = PrivilegeStore::new();
    let now = now_secs();
    let reports = 200u32;

    // GRANT SELECT ON reports TO manager(2).
    store
        .grant(make_grant(2, PrivilegeType::Select, reports))
        .expect("grant");

    // analyst(3) effective roles include manager(2).
    let effective = hierarchy.effective_roles(RoleId(3));
    assert!(effective.contains(&RoleId(2)));
    let d = store.check_privilege(
        &effective,
        PrivilegeType::Select,
        ObjectType::Table,
        reports,
        None,
        now,
    );
    assert_eq!(d, PrivilegeDecision::Allow);
    tprintln!("  analyst inherits SELECT on reports from manager: verified");

    // Without hierarchy, analyst alone has no access.
    let d = store.check_privilege(
        &[RoleId(3)],
        PrivilegeType::Select,
        ObjectType::Table,
        reports,
        None,
        now,
    );
    assert_ne!(d, PrivilegeDecision::Allow);
    tprintln!("  analyst alone (no hierarchy): denied");
}

// ===========================================================================
// 18. JWT Correctness
// ===========================================================================

#[test]
fn test_jwt_correctness() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- JWT Correctness ---");

    let secret = vec![0xABu8; 32];
    let jwt = JwtCredential::new(secret.clone(), JwtAlgorithm::Hs256).expect("create");
    let claims = JwtClaims {
        sub: "user:1000".to_string(),
        iss: None,
        exp: u64::MAX,
        iat: 1700000000,
        roles: vec!["admin".to_string()],
        custom: HashMap::new(),
    };

    // Valid token.
    let token = jwt.encode(&claims).expect("encode");
    let decoded = jwt.decode(&token).expect("decode");
    assert_eq!(decoded.sub, "user:1000");
    assert_eq!(decoded.roles, vec!["admin"]);
    tprintln!("  Valid token: verified");

    // Tampered signature.
    let mut tampered = token.clone();
    tampered.pop();
    tampered.push('X');
    assert!(jwt.decode(&tampered).is_err());
    tprintln!("  Tampered signature: rejected");

    // Wrong secret.
    let wrong = JwtCredential::new(vec![0xCDu8; 32], JwtAlgorithm::Hs256).expect("create");
    assert!(wrong.decode(&token).is_err());
    tprintln!("  Wrong secret: rejected");

    // Expired token (exp=0 treated as missing).
    let expired_claims = JwtClaims {
        sub: "user:expired".to_string(),
        iss: None,
        exp: 0,
        iat: 1700000000,
        roles: Vec::new(),
        custom: HashMap::new(),
    };
    let expired_token = jwt.encode(&expired_claims).expect("encode");
    assert!(jwt.decode(&expired_token).is_err());
    tprintln!("  Expired token: rejected");

    // Issuer mismatch.
    let jwt_iss = JwtCredential::new(secret.clone(), JwtAlgorithm::Hs256)
        .expect("create")
        .with_issuer("zyrondb".to_string());
    let wrong_iss_claims = JwtClaims {
        sub: "u".to_string(),
        iss: Some("wrong".to_string()),
        exp: u64::MAX,
        iat: 1700000000,
        roles: Vec::new(),
        custom: HashMap::new(),
    };
    let wrong_iss_token = jwt.encode(&wrong_iss_claims).expect("encode");
    assert!(jwt_iss.decode(&wrong_iss_token).is_err());
    tprintln!("  Issuer mismatch: rejected");

    // Decode unverified.
    let (header, uv_claims) = JwtCredential::decode_unverified(&token).expect("decode");
    assert_eq!(header.alg, "HS256");
    assert_eq!(uv_claims.sub, "user:1000");
    tprintln!("  Decode unverified: OK");

    // HS384 + HS512.
    let jwt384 = JwtCredential::new(vec![0xCDu8; 48], JwtAlgorithm::Hs384).expect("create");
    let t384 = jwt384.encode(&claims).expect("encode");
    assert_eq!(jwt384.decode(&t384).expect("decode").sub, "user:1000");
    let jwt512 = JwtCredential::new(vec![0xEFu8; 64], JwtAlgorithm::Hs512).expect("create");
    let t512 = jwt512.encode(&claims).expect("encode");
    assert_eq!(jwt512.decode(&t512).expect("decode").sub, "user:1000");
    tprintln!("  HS384 + HS512: verified");

    // Performance: JWT verify latency.
    let count = 100_000;
    let start = Instant::now();
    for _ in 0..count {
        let _ = jwt.decode(&token);
    }
    let elapsed = start.elapsed();
    let per_us = elapsed.as_secs_f64() * 1_000_000.0 / count as f64;
    tprintln!("  JWT verify: {:.2}us/op (target: <200us)", per_us);
    validate_metric(
        "jwt_correctness",
        "verify_latency_us",
        vec![per_us],
        200.0,
        false,
    );
    check_performance("jwt_correctness", "verify_latency_us", per_us, 200.0, false);
}

// ===========================================================================
// 19. API Key Correctness
// ===========================================================================

#[test]
fn test_api_key_correctness() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- API Key Correctness ---");

    let (cred, full_key) = ApiKeyCredential::generate();
    assert!(full_key.starts_with("zyron_"));
    tprintln!("  Generated: {}...", &full_key[..14]);

    assert!(cred.verify(&full_key), "Valid key must authenticate");
    tprintln!("  Valid key: authenticated");

    let mut modified = full_key.clone();
    modified.push('X');
    assert!(!cred.verify(&modified), "Modified key must fail");
    tprintln!("  Modified key: rejected");

    let wrong_prefix = full_key.replace("zyron_", "wrong_");
    assert!(!cred.verify(&wrong_prefix), "Wrong prefix must fail");
    tprintln!("  Wrong prefix: rejected");

    // Stored roundtrip.
    let restored = ApiKeyCredential::from_stored(cred.prefix().to_string(), *cred.key_hash());
    assert!(restored.verify(&full_key));
    tprintln!("  Stored roundtrip: verified");

    // Performance.
    let count = 1_000_000;
    let start = Instant::now();
    for _ in 0..count {
        let _ = cred.verify(&full_key);
    }
    let elapsed = start.elapsed();
    let per_ns = elapsed.as_nanos() as f64 / count as f64;
    tprintln!("  API key verify: {:.1}ns/op (target: <50us)", per_ns);
    validate_metric(
        "api_key_correctness",
        "lookup_ns",
        vec![per_ns],
        50_000.0,
        false,
    );
    check_performance("api_key_correctness", "lookup_ns", per_ns, 50_000.0, false);
}

// ===========================================================================
// 20. TOTP Correctness
// ===========================================================================

#[test]
fn test_totp_correctness() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- TOTP Correctness ---");

    let totp = TotpCredential::generate();
    assert!(!totp.secret_base32().is_empty());
    let ts = 1700000000u64;

    // Current code.
    let code = totp.generate_code(ts);
    assert_eq!(code.len(), 6);
    assert!(totp.verify(&code, ts), "Current code must verify");
    tprintln!("  Current code: verified");

    // Code from 30s ago (drift=1 period).
    let code_prev = totp.generate_code(ts - 30);
    assert!(
        totp.verify(&code_prev, ts),
        "Code from 30s ago must verify with drift=1"
    );
    tprintln!("  Code from 30s ago (drift=1): verified");

    // Code from 90s ago (drift=3 periods, beyond tolerance).
    let code_old = totp.generate_code(ts - 90);
    assert!(
        !totp.verify(&code_old, ts),
        "Code from 90s ago should NOT verify"
    );
    tprintln!("  Code from 90s ago (drift=3): rejected");

    // RFC 6238 test vector.
    let rfc = TotpCredential::from_secret(b"12345678901234567890".to_vec());
    assert_eq!(rfc.generate_code(59), "287082", "RFC 6238 test vector");
    tprintln!("  RFC 6238 vector: 287082");

    // Performance.
    let count = 1_000_000;
    let start = Instant::now();
    for i in 0..count {
        let t = ts + (i as u64) * 30;
        let c = totp.generate_code(t);
        let _ = totp.verify(&c, t);
    }
    let elapsed = start.elapsed();
    let per_ns = elapsed.as_nanos() as f64 / count as f64;
    tprintln!("  TOTP verify: {:.1}ns/op (target: <500us)", per_ns);
    validate_metric(
        "totp_correctness",
        "verify_ns",
        vec![per_ns],
        500_000.0,
        false,
    );
    check_performance("totp_correctness", "verify_ns", per_ns, 500_000.0, false);
}

// ===========================================================================
// 21. Concurrent Authentication
// ===========================================================================

#[test]
fn test_concurrent_authentication() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- Concurrent Authentication ---");

    let params = BalloonParams::test();
    let cred =
        PasswordCredential::from_plaintext_with_params("concurrent_pw", &params).expect("hash");
    let stored = cred.as_stored().to_string();

    let threads = 100;
    let per_thread = 10;

    let start = Instant::now();
    let handles: Vec<_> = (0..threads)
        .map(|_| {
            let h = stored.clone();
            std::thread::spawn(move || {
                let c = PasswordCredential::from_stored(h);
                let mut ok = 0u64;
                for _ in 0..per_thread {
                    if c.verify("concurrent_pw").unwrap_or(false) {
                        ok += 1;
                    }
                }
                ok
            })
        })
        .collect();

    let mut total = 0u64;
    for h in handles {
        total += h.join().expect("thread panicked");
    }
    let elapsed = start.elapsed();

    let expected = (threads * per_thread) as u64;
    assert_eq!(total, expected, "All concurrent auths must succeed");
    let auth_per_sec = expected as f64 / elapsed.as_secs_f64();
    tprintln!(
        "  {} threads x {} = {} auths, {:.0} auth/sec",
        threads,
        per_thread,
        total,
        auth_per_sec
    );
    validate_metric(
        "concurrent_auth",
        "auth_per_sec",
        vec![auth_per_sec],
        1000.0,
        true,
    );
    check_performance(
        "concurrent_auth",
        "auth_per_sec",
        auth_per_sec,
        1000.0,
        true,
    );

    // Also verify concurrent privilege checks (no deadlock).
    let store = std::sync::Arc::new(PrivilegeStore::new());
    let mut entries = Vec::with_capacity(100);
    for i in 0..100u32 {
        entries.push(make_grant(i % 10, PrivilegeType::Select, i));
    }
    store.load(entries);
    let now = now_secs();

    let priv_handles: Vec<_> = (0..threads)
        .map(|_| {
            let s = store.clone();
            std::thread::spawn(move || {
                let roles = vec![RoleId(0), RoleId(1), RoleId(2)];
                let mut ok = 0u64;
                for i in 0..1000 {
                    if s.check_privilege(
                        &roles,
                        PrivilegeType::Select,
                        ObjectType::Table,
                        (i % 100) as u32,
                        None,
                        now,
                    ) == PrivilegeDecision::Allow
                    {
                        ok += 1;
                    }
                }
                ok
            })
        })
        .collect();
    let mut total_priv = 0u64;
    for h in priv_handles {
        total_priv += h.join().expect("thread panicked");
    }
    tprintln!(
        "  Concurrent privilege checks: {} total, {} allowed (no deadlock)",
        threads * 1000,
        total_priv
    );
}

// ===========================================================================
// 22. Privilege Check Latency
// ===========================================================================

#[test]
fn test_privilege_check_latency() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- Privilege Check Latency ---");

    let store = PrivilegeStore::new();
    let now = now_secs();
    let mut entries = Vec::with_capacity(200);
    for t in 0..100u32 {
        entries.push(make_grant(t % 10, PrivilegeType::Select, t));
        entries.push(make_grant(t % 10, PrivilegeType::Insert, t));
    }
    store.load(entries);

    let roles = vec![RoleId(0), RoleId(1), RoleId(2)];
    let count = 1_000_000;

    let start = Instant::now();
    let mut allowed = 0u64;
    for i in 0..count {
        let t = (i % 100) as u32;
        let p = if i % 2 == 0 {
            PrivilegeType::Select
        } else {
            PrivilegeType::Insert
        };
        if store.check_privilege(&roles, p, ObjectType::Table, t, None, now)
            == PrivilegeDecision::Allow
        {
            allowed += 1;
        }
    }
    let elapsed = start.elapsed();
    let per_ns = elapsed.as_nanos() as f64 / count as f64;
    tprintln!(
        "  Latency: {:.1}ns/check, allowed: {}/{}",
        per_ns,
        allowed,
        count
    );
    validate_metric(
        "privilege_latency",
        "per_check_ns",
        vec![per_ns],
        80.0,
        false,
    );
    check_performance("privilege_latency", "per_check_ns", per_ns, 80.0, false);
}

// ===========================================================================
// 23. Admin Privilege Types
// ===========================================================================

#[test]
fn test_admin_privilege_types() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- Admin Privilege Types ---");

    let store = PrivilegeStore::new();
    let now = now_secs();
    let admin_privs = [
        PrivilegeType::ManageRoles,
        PrivilegeType::ManagePrivileges,
        PrivilegeType::ManagePolicy,
        PrivilegeType::ManageClassification,
        PrivilegeType::ManageTags,
        PrivilegeType::ManageMasking,
        PrivilegeType::ManageOwnership,
        PrivilegeType::ManageBreakGlass,
        PrivilegeType::ManageQueryLimits,
        PrivilegeType::ManageAuthRules,
    ];

    for p in &admin_privs {
        store
            .grant(make_grant_on(1, *p, ObjectType::Database, 1))
            .expect("grant");
    }

    // All granted to role 1.
    for p in &admin_privs {
        let d = store.check_privilege(&[RoleId(1)], *p, ObjectType::Database, 1, None, now);
        assert_eq!(d, PrivilegeDecision::Allow, "{:?} should be granted", p);
    }
    tprintln!("  All 10 admin types granted to role 1: verified");

    // None granted to role 2.
    for p in &admin_privs {
        let d = store.check_privilege(&[RoleId(2)], *p, ObjectType::Database, 1, None, now);
        assert_ne!(
            d,
            PrivilegeDecision::Allow,
            "role 2 should NOT have {:?}",
            p
        );
    }
    tprintln!("  Role 2 denied all admin types: verified");

    // Impersonate.
    store
        .grant(make_grant_on(
            1,
            PrivilegeType::Impersonate,
            ObjectType::Type,
            50,
        ))
        .expect("grant");
    let d = store.check_privilege(
        &[RoleId(1)],
        PrivilegeType::Impersonate,
        ObjectType::Type,
        50,
        None,
        now,
    );
    assert_eq!(d, PrivilegeDecision::Allow);
    tprintln!("  Impersonate privilege: verified");

    // All matches admin types.
    store
        .grant(make_grant(5, PrivilegeType::All, 500))
        .expect("grant");
    let d = store.check_privilege(
        &[RoleId(5)],
        PrivilegeType::ManageRoles,
        ObjectType::Table,
        500,
        None,
        now,
    );
    assert_eq!(d, PrivilegeDecision::Allow);
    tprintln!("  PrivilegeType::All matches admin types: verified");
}

// ===========================================================================
// 24. Temporal Privileges
// ===========================================================================

#[test]
fn test_temporal_privileges() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- Temporal Privileges ---");

    let store = PrivilegeStore::new();
    store
        .grant(make_temporal_grant(
            1,
            PrivilegeType::Select,
            100,
            Some(1000),
            Some(2000),
        ))
        .expect("grant");

    let d = store.check_privilege(
        &[RoleId(1)],
        PrivilegeType::Select,
        ObjectType::Table,
        100,
        None,
        1500,
    );
    assert_eq!(d, PrivilegeDecision::Allow);
    tprintln!("  Within window (t=1500): allowed");

    let d = store.check_privilege(
        &[RoleId(1)],
        PrivilegeType::Select,
        ObjectType::Table,
        100,
        None,
        500,
    );
    assert_ne!(d, PrivilegeDecision::Allow);
    tprintln!("  Before window (t=500): denied");

    let d = store.check_privilege(
        &[RoleId(1)],
        PrivilegeType::Select,
        ObjectType::Table,
        100,
        None,
        2500,
    );
    assert_ne!(d, PrivilegeDecision::Allow);
    tprintln!("  After window (t=2500): denied");
}

// ===========================================================================
// 25. Column-Level Privileges
// ===========================================================================

#[test]
fn test_column_level_privileges() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- Column-Level Privileges ---");

    let store = PrivilegeStore::new();
    let now = now_secs();
    store
        .grant(make_column_grant(1, PrivilegeType::Select, 100, vec![0, 1]))
        .expect("grant");

    let d = store.check_privilege(
        &[RoleId(1)],
        PrivilegeType::Select,
        ObjectType::Table,
        100,
        Some(&[0]),
        now,
    );
    assert_eq!(d, PrivilegeDecision::Allow);
    tprintln!("  Column 0: allowed");

    let d = store.check_privilege(
        &[RoleId(1)],
        PrivilegeType::Select,
        ObjectType::Table,
        100,
        Some(&[1]),
        now,
    );
    assert_eq!(d, PrivilegeDecision::Allow);
    tprintln!("  Column 1: allowed");

    let d = store.check_privilege(
        &[RoleId(1)],
        PrivilegeType::Select,
        ObjectType::Table,
        100,
        Some(&[2]),
        now,
    );
    assert_ne!(d, PrivilegeDecision::Allow);
    tprintln!("  Column 2 (not granted): denied");
}

// ===========================================================================
// 26. SecurityContext: Impersonation and Role Switching
// ===========================================================================

#[test]
fn test_security_context_impersonation() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- SecurityContext Impersonation ---");

    let store = PrivilegeStore::new();
    let hierarchy = RoleHierarchy::new();
    hierarchy.load(&[RoleMembership {
        member_id: RoleId(10),
        parent_id: RoleId(20),
        admin_option: false,
        inherit: true,
        granted_by: RoleId(0),
    }]);

    // Grant Impersonate on role 50 and SELECT on table 100 to role 1.
    store
        .grant(make_grant_on(
            1,
            PrivilegeType::Impersonate,
            ObjectType::Type,
            50,
        ))
        .expect("grant");
    store
        .grant(make_grant(1, PrivilegeType::Select, 100))
        .expect("grant");

    let effective = hierarchy.effective_roles(RoleId(1));
    let all = hierarchy.all_roles(RoleId(1));
    let mut ctx = SecurityContext::new(
        UserId(1),
        RoleId(1),
        effective,
        all,
        ClassificationLevel::Restricted,
        make_attributes(RoleId(1)),
        Some("10.0.0.1".to_string()),
        QueryLimits::default(),
    );

    // Privilege check through context.
    assert!(ctx.has_privilege(
        &store,
        PrivilegeType::Select,
        ObjectType::Table,
        100,
        None,
        now_secs()
    ));
    tprintln!("  Privilege check: verified");

    // EXECUTE AS role 50.
    ctx.execute_as(RoleId(50), &store, &hierarchy)
        .expect("execute_as");
    assert_eq!(ctx.current_role, RoleId(50));
    tprintln!("  Execute as role 50: succeeded");

    // REVERT.
    ctx.revert(&hierarchy).expect("revert");
    assert_eq!(ctx.current_role, RoleId(1));
    tprintln!("  Revert: restored");

    // Execute as without privilege.
    let empty_store = PrivilegeStore::new();
    assert!(
        ctx.execute_as(RoleId(99), &empty_store, &hierarchy)
            .is_err()
    );
    tprintln!("  Execute as without privilege: denied");

    // SET ROLE.
    let user_eff = hierarchy.effective_roles(RoleId(10));
    let user_all = hierarchy.all_roles(RoleId(10));
    let mut user_ctx = SecurityContext::new(
        UserId(10),
        RoleId(10),
        user_eff,
        user_all,
        ClassificationLevel::Internal,
        make_attributes(RoleId(10)),
        None,
        QueryLimits::default(),
    );
    user_ctx.set_role(RoleId(20), &hierarchy).expect("set_role");
    assert_eq!(user_ctx.current_role, RoleId(20));
    user_ctx.reset_role(&hierarchy);
    assert_eq!(user_ctx.current_role, RoleId(10));
    assert!(user_ctx.set_role(RoleId(999), &hierarchy).is_err());
    tprintln!("  SET ROLE / RESET ROLE: verified");

    // Attributes.
    assert_eq!(user_ctx.get_attribute("department"), Some("engineering"));
    user_ctx.set_attribute("project".to_string(), "zyrondb".to_string());
    assert_eq!(user_ctx.get_attribute("project"), Some("zyrondb"));
    tprintln!("  Session attributes: verified");
}

// ===========================================================================
// 27. Break-Glass Emergency Access
// ===========================================================================

#[test]
fn test_break_glass() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- Break-Glass ---");

    let mgr = BreakGlassManager::new(3600);

    // Activate with roles + privileges + clearance.
    mgr.activate(
        RoleId(100),
        vec![RoleId(50), RoleId(51)],
        vec![PrivilegeType::ManageRoles, PrivilegeType::ManagePrivileges],
        Some(ClassificationLevel::Restricted),
        "Incident #5678".to_string(),
        1800,
    )
    .expect("activate");

    let s = mgr.is_active(RoleId(100)).expect("should be active");
    assert_eq!(s.activated_roles, vec![RoleId(50), RoleId(51)]);
    assert_eq!(
        s.activated_privileges,
        vec![PrivilegeType::ManageRoles, PrivilegeType::ManagePrivileges]
    );
    assert_eq!(s.elevated_clearance, Some(ClassificationLevel::Restricted));
    assert!(s.expires_at - s.granted_at <= 1800);
    tprintln!("  Activated with roles, privs, clearance, capped duration: verified");

    mgr.deactivate(RoleId(100));
    assert!(mgr.is_active(RoleId(100)).is_none());
    assert!(mgr.audit_trail().last().map(|e| e.revoked).unwrap_or(false));
    tprintln!("  Deactivate + audit: verified");

    // Serialization roundtrip.
    let session = zyron_auth::breakglass::BreakGlassSession {
        role_id: RoleId(42),
        activated_roles: vec![RoleId(60)],
        activated_privileges: vec![PrivilegeType::ManageTags],
        elevated_clearance: Some(ClassificationLevel::Confidential),
        reason: "test roundtrip".to_string(),
        granted_at: 1700000000,
        expires_at: 1700003600,
        revoked: false,
    };
    let bytes = session.to_bytes();
    let restored = zyron_auth::breakglass::BreakGlassSession::from_bytes(&bytes).expect("decode");
    assert_eq!(restored.role_id, RoleId(42));
    assert_eq!(restored.activated_roles, vec![RoleId(60)]);
    assert_eq!(
        restored.activated_privileges,
        vec![PrivilegeType::ManageTags]
    );
    assert_eq!(
        restored.elevated_clearance,
        Some(ClassificationLevel::Confidential)
    );
    tprintln!("  Serialization roundtrip: verified");
}

// ===========================================================================
// 28. Classification and Clearance
// ===========================================================================

#[test]
fn test_classification_correctness() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- Classification Correctness ---");

    let store = ClassificationStore::new();
    store.set_classification(100, 0, ClassificationLevel::Public);
    store.set_classification(100, 1, ClassificationLevel::Internal);
    store.set_classification(100, 2, ClassificationLevel::Confidential);
    store.set_classification(100, 3, ClassificationLevel::Restricted);

    // Internal clearance.
    assert!(store.check_clearance(ClassificationLevel::Internal, 100, 0));
    assert!(store.check_clearance(ClassificationLevel::Internal, 100, 1));
    assert!(!store.check_clearance(ClassificationLevel::Internal, 100, 2));
    assert!(!store.check_clearance(ClassificationLevel::Internal, 100, 3));
    tprintln!("  Internal clearance: Public(Y) Internal(Y) Confidential(N) Restricted(N)");

    // Restricted clearance.
    for col in 0..4 {
        assert!(store.check_clearance(ClassificationLevel::Restricted, 100, col));
    }
    tprintln!("  Restricted clearance: all accessible");

    // Unclassified column.
    assert!(store.check_clearance(ClassificationLevel::Public, 999, 0));
    tprintln!("  Unclassified: accessible at any level");

    // Drop + re-check.
    store.drop_classification(100, 3);
    assert!(store.check_clearance(ClassificationLevel::Public, 100, 3));
    tprintln!("  Drop classification: now accessible");

    // Serialization.
    let e = ColumnClassification {
        table_id: 42,
        column_id: 7,
        level: ClassificationLevel::Confidential,
    };
    let restored = ColumnClassification::from_bytes(&e.to_bytes()).expect("decode");
    assert_eq!(restored.table_id, 42);
    assert_eq!(restored.level, ClassificationLevel::Confidential);
    tprintln!("  Serialization: verified");
}

// ===========================================================================
// 29. Data Masking All Types
// ===========================================================================

#[test]
fn test_data_masking_all_types() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- Data Masking All Types ---");

    let mut buf = String::with_capacity(64);

    masking::apply_mask("user@example.com", &MaskFunction::Email, &mut buf);
    assert!(buf.contains("@example.com"));
    tprintln!("  Email: user@example.com -> {}", buf);

    masking::apply_mask("555-123-4567", &MaskFunction::Phone, &mut buf);
    tprintln!("  Phone: 555-123-4567 -> {}", buf);

    masking::apply_mask("123-45-6789", &MaskFunction::Ssn, &mut buf);
    tprintln!("  SSN: 123-45-6789 -> {}", buf);

    masking::apply_mask("4111111111111111", &MaskFunction::CreditCard, &mut buf);
    tprintln!("  CreditCard: 4111111111111111 -> {}", buf);

    assert!(!masking::apply_mask(
        "anything",
        &MaskFunction::Null,
        &mut buf
    ));
    tprintln!("  Null: anything -> None (SQL NULL)");

    masking::apply_mask("sensitive", &MaskFunction::Redact, &mut buf);
    assert_eq!(buf, "[REDACTED]");
    tprintln!("  Redact: -> [REDACTED]");

    masking::apply_mask("data", &MaskFunction::Hash, &mut buf);
    let h1 = buf.clone();
    masking::apply_mask("data", &MaskFunction::Hash, &mut buf);
    assert_eq!(h1, buf, "Hash must be deterministic");
    tprintln!("  Hash: deterministic, {} chars", h1.len());

    masking::apply_mask("Hello World", &MaskFunction::Partial(3), &mut buf);
    tprintln!("  Partial(3): Hello World -> {}", buf);
}

// ===========================================================================
// 30. Object Tagging
// ===========================================================================

#[test]
fn test_object_tagging() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- Object Tagging ---");

    let store = TagStore::new();
    store
        .tag_object(ObjectTag {
            object_type: ObjectType::Table,
            object_id: 100,
            column_id: None,
            tag: "pii".to_string(),
        })
        .expect("tag");
    store
        .tag_object(ObjectTag {
            object_type: ObjectType::Table,
            object_id: 100,
            column_id: Some(5),
            tag: "email".to_string(),
        })
        .expect("tag");

    assert!(store.has_tag(ObjectType::Table, 100, "pii"));
    assert!(!store.has_tag(ObjectType::Table, 100, "financial"));
    tprintln!("  has_tag: pii=true, financial=false");

    let tags = store.tags_for_object(ObjectType::Table, 100);
    assert!(tags.contains(&"pii".to_string()));
    tprintln!("  tags_for_object: {:?}", tags);

    assert!(!store.objects_with_tag("pii").is_empty());
    tprintln!("  objects_with_tag(pii): found");

    assert!(store.untag_object(ObjectType::Table, 100, "pii"));
    assert!(!store.has_tag(ObjectType::Table, 100, "pii"));
    tprintln!("  untag: removed");

    let tag = ObjectTag {
        object_type: ObjectType::Table,
        object_id: 42,
        column_id: Some(3),
        tag: "sensitive".to_string(),
    };
    let restored = ObjectTag::from_bytes(&tag.to_bytes()).expect("decode");
    assert_eq!(restored.tag, "sensitive");
    tprintln!("  Serialization: verified");
}

// ===========================================================================
// 31. Row Ownership
// ===========================================================================

#[test]
fn test_row_ownership() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- Row Ownership ---");

    let store = RowOwnershipStore::new();
    assert!(!store.is_enabled(100));

    let config = RowOwnershipConfig {
        table_id: 100,
        enabled: true,
        owner_column: "created_by".to_string(),
        admin_roles: vec![RoleId(1), RoleId(2)],
    };
    store.enable(100, config);
    assert!(store.is_enabled(100));
    assert_eq!(
        store.get_config(100).expect("config").owner_column,
        "created_by"
    );
    assert!(store.is_admin(100, RoleId(1)));
    assert!(!store.is_admin(100, RoleId(99)));
    tprintln!("  Enable, config, admin check: verified");

    store.disable(100);
    assert!(!store.is_enabled(100));
    tprintln!("  Disable: verified");

    let cfg = RowOwnershipConfig {
        table_id: 200,
        enabled: true,
        owner_column: "user_id".to_string(),
        admin_roles: vec![RoleId(10)],
    };
    let restored = RowOwnershipConfig::from_bytes(&cfg.to_bytes()).expect("decode");
    assert_eq!(restored.owner_column, "user_id");
    tprintln!("  Serialization: verified");
}

// ===========================================================================
// 32. Session Binding: Query Limits and Time Windows
// ===========================================================================

#[test]
fn test_session_binding() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- Session Binding ---");

    let window = TimeWindow::business_hours();
    let monday_noon = 1704067200u64 + 12 * 3600;
    assert!(window.is_active(monday_noon));
    let saturday = 1704067200u64 + 5 * 86400;
    assert!(!window.is_active(saturday));
    tprintln!("  Business hours: Mon noon=active, Sat=inactive");

    let limit_store = QueryLimitStore::new();
    let limits = QueryLimits {
        role_id: RoleId(10),
        max_scan_rows: Some(100_000),
        max_result_rows: Some(10_000),
        max_execution_time_ms: Some(30_000),
        max_memory_bytes: Some(1_073_741_824),
        max_temp_bytes: Some(536_870_912),
        allow_full_scan: false,
    };
    limit_store.set_limits(limits);
    let r = limit_store.get_limits(&[RoleId(10)]);
    assert_eq!(r.max_scan_rows, Some(100_000));
    assert!(!r.allow_full_scan);
    let d = limit_store.get_limits(&[RoleId(999)]);
    assert!(d.allow_full_scan);
    tprintln!("  Query limits: set/get/default verified");

    let restored = TimeWindow::from_bytes(&window.to_bytes()).expect("decode");
    assert_eq!(restored.start_hour, window.start_hour);
    tprintln!("  TimeWindow serialization: verified");
}

// ===========================================================================
// 33. ABAC Policies
// ===========================================================================

#[test]
fn test_abac_policies() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- ABAC Policies ---");

    let store = AbacStore::new();
    store
        .add_policy(AbacPolicy {
            id: 1,
            name: "eng_access".to_string(),
            table_id: 100,
            predicate: "department = 'engineering'".to_string(),
            enabled: true,
            permissive: true,
            roles: vec![],
        })
        .expect("add");
    store
        .add_policy(AbacPolicy {
            id: 2,
            name: "region_filter".to_string(),
            table_id: 100,
            predicate: "region = 'us-east'".to_string(),
            enabled: true,
            permissive: false,
            roles: vec![RoleId(10)],
        })
        .expect("add");

    assert_eq!(store.policies_for_table(100).len(), 2);
    assert!(store.policies_for_table(200).is_empty());
    tprintln!("  Table 100: 2 policies, Table 200: 0");

    assert!(store.remove_policy(100, "eng_access"));
    assert_eq!(store.policies_for_table(100).len(), 1);
    tprintln!("  Remove policy: 1 remaining");

    let policy = AbacPolicy {
        id: 42,
        name: "test".to_string(),
        table_id: 300,
        predicate: "clearance >= 'confidential'".to_string(),
        enabled: true,
        permissive: true,
        roles: vec![RoleId(1), RoleId(2)],
    };
    let restored = AbacPolicy::from_bytes(&policy.to_bytes()).expect("decode");
    assert_eq!(restored.name, "test");
    assert_eq!(restored.roles, vec![RoleId(1), RoleId(2)]);
    tprintln!("  Serialization: verified");

    let mut attrs = make_attributes(RoleId(10));
    assert_eq!(attrs.get("department"), Some("engineering"));
    attrs.set("project".to_string(), "zyrondb".to_string());
    assert_eq!(attrs.get("project"), Some("zyrondb"));
    tprintln!("  Session attributes: verified");
}

// ===========================================================================
// 34. Auth Rules Resolution
// ===========================================================================

#[test]
fn test_auth_rules_resolution() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- Auth Rules Resolution ---");

    let resolver = AuthResolver::new(vec![
        AuthRule {
            priority: 1,
            connection_type: ConnectionType::HostSsl,
            database_pattern: "production".to_string(),
            user_pattern: "admin".to_string(),
            source_cidr: Some("10.0.0.0/8".to_string()),
            method: AuthMethod::BalloonSha256,
            options: HashMap::new(),
        },
        AuthRule {
            priority: 50,
            connection_type: ConnectionType::All,
            database_pattern: "*".to_string(),
            user_pattern: "*".to_string(),
            source_cidr: None,
            method: AuthMethod::ScramSha256,
            options: HashMap::new(),
        },
        AuthRule {
            priority: 5,
            connection_type: ConnectionType::Local,
            database_pattern: "dev".to_string(),
            user_pattern: "*".to_string(),
            source_cidr: None,
            method: AuthMethod::Trust,
            options: HashMap::new(),
        },
    ]);

    assert_eq!(
        resolver.resolve(
            ConnectionType::HostSsl,
            "production",
            "admin",
            Some("10.0.0.1")
        ),
        AuthMethod::BalloonSha256
    );
    tprintln!("  admin/prod/SSL/10.x: BalloonSha256");

    assert_eq!(
        resolver.resolve(
            ConnectionType::HostSsl,
            "production",
            "admin",
            Some("192.168.1.1")
        ),
        AuthMethod::ScramSha256
    );
    tprintln!("  admin/prod/SSL/192.x: ScramSha256 (CIDR mismatch)");

    assert_eq!(
        resolver.resolve(ConnectionType::Host, "mydb", "alice", None),
        AuthMethod::ScramSha256
    );
    tprintln!("  alice/mydb/Host: ScramSha256 (wildcard)");

    assert_eq!(
        resolver.resolve(ConnectionType::Local, "dev", "dev", None),
        AuthMethod::Trust
    );
    tprintln!("  dev/Local: Trust");
}

// ===========================================================================
// 35. Governance: Delegation, Cascade Revoke, Two-Person Approval
// ===========================================================================

#[test]
fn test_governance_full() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- Governance ---");

    let gov = GovernanceManager::new();

    // Analytics.
    gov.analytics
        .record_usage(RoleId(1), PrivilegeType::Select, ObjectType::Table, 100);
    gov.analytics
        .record_usage(RoleId(1), PrivilegeType::Select, ObjectType::Table, 100);
    let report = gov.analytics.usage_report(RoleId(1));
    assert!(report[0].use_count >= 2);
    tprintln!("  Analytics: 2 usages recorded");

    // Delegation: A(1) -> B(2) -> C(3).
    gov.delegation.record_grant(DelegationEdge {
        grantor: RoleId(1),
        grantee: RoleId(2),
        privilege: PrivilegeType::Select,
        object_type: ObjectType::Table,
        object_id: 100,
        granted_at: 1000,
    });
    gov.delegation.record_grant(DelegationEdge {
        grantor: RoleId(2),
        grantee: RoleId(3),
        privilege: PrivilegeType::Select,
        object_type: ObjectType::Table,
        object_id: 100,
        granted_at: 2000,
    });

    let revoked =
        gov.delegation
            .cascade_revoke(RoleId(1), PrivilegeType::Select, ObjectType::Table, 100);
    assert_eq!(revoked.len(), 2);
    tprintln!("  Cascade revoke from A: B+C revoked");

    let chain =
        gov.delegation
            .chain_for_grant(RoleId(3), PrivilegeType::Select, ObjectType::Table, 100);
    assert_eq!(chain.len(), 2);
    assert_eq!(chain[0].grantor, RoleId(2));
    assert_eq!(chain[1].grantor, RoleId(1));
    tprintln!("  Grant chain C->B->A: verified");

    // Two-person approval.
    gov.two_person.add_rule(TwoPersonRule {
        operation: TwoPersonOperation::DropDatabase,
        required_role: None,
        timeout_secs: 3600,
    });
    assert!(
        gov.two_person
            .requires_approval(TwoPersonOperation::DropDatabase)
    );
    assert!(
        !gov.two_person
            .requires_approval(TwoPersonOperation::DropTable)
    );

    let id = gov
        .two_person
        .request_approval(
            RoleId(1),
            TwoPersonOperation::DropDatabase,
            "decommission".to_string(),
        )
        .expect("request");
    assert!(
        gov.two_person.approve(id, RoleId(1)).is_err(),
        "Same person cannot approve"
    );
    gov.two_person.approve(id, RoleId(2)).expect("approve");
    tprintln!("  Two-person: same-person denied, different-person approved");

    let deny_id = gov
        .two_person
        .request_approval(
            RoleId(5),
            TwoPersonOperation::DropDatabase,
            "test".to_string(),
        )
        .expect("request");
    gov.two_person.deny(deny_id, RoleId(6)).expect("deny");
    tprintln!("  Two-person deny: removed");

    assert!(
        gov.two_person
            .request_approval(
                RoleId(1),
                TwoPersonOperation::DropTable,
                "no rule".to_string()
            )
            .is_err()
    );
    tprintln!("  No rule for DropTable: request denied");
}

// ===========================================================================
// 36. Serialization Roundtrips for All Entry Types
// ===========================================================================

#[test]
fn test_serialization_roundtrips() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- Serialization Roundtrips ---");

    // GrantEntry with all fields populated.
    let grant = GrantEntry {
        grantee: RoleId(42),
        privilege: PrivilegeType::ManageRoles,
        object_type: ObjectType::Database,
        object_id: 1,
        columns: Some(vec![0, 1, 2]),
        state: PrivilegeState::Grant,
        with_grant_option: true,
        granted_by: RoleId(1),
        valid_from: Some(1000),
        valid_until: Some(2000),
        time_window: Some(TimeWindow::business_hours()),
        object_pattern: Some("users_%".to_string()),
        no_inherit: true,
        mask_function: Some("email".to_string()),
    };
    let r = GrantEntry::from_bytes(&grant.to_bytes()).expect("decode");
    assert_eq!(r.grantee, RoleId(42));
    assert_eq!(r.privilege, PrivilegeType::ManageRoles);
    assert_eq!(r.columns, Some(vec![0, 1, 2]));
    assert!(r.with_grant_option);
    assert_eq!(r.valid_from, Some(1000));
    assert!(r.time_window.is_some());
    assert_eq!(r.object_pattern.as_deref(), Some("users_%"));
    assert!(r.no_inherit);
    assert_eq!(r.mask_function.as_deref(), Some("email"));
    tprintln!("  GrantEntry (all fields): verified");

    // RoleMembership.
    let m = RoleMembership {
        member_id: RoleId(100),
        parent_id: RoleId(200),
        admin_option: true,
        inherit: true,
        granted_by: RoleId(1),
    };
    let mr = RoleMembership::from_bytes(&m.to_bytes()).expect("decode");
    assert_eq!(mr.member_id, RoleId(100));
    assert!(mr.admin_option);
    tprintln!("  RoleMembership: verified");

    // AuthRule.
    let mut opts = HashMap::new();
    opts.insert("key".to_string(), "value".to_string());
    let rule = AuthRule {
        priority: 5,
        connection_type: ConnectionType::HostSsl,
        database_pattern: "prod".to_string(),
        user_pattern: "admin".to_string(),
        source_cidr: Some("10.0.0.0/8".to_string()),
        method: AuthMethod::BalloonSha256,
        options: opts,
    };
    let rr = AuthRule::from_bytes(&rule.to_bytes()).expect("decode");
    assert_eq!(rr.priority, 5);
    assert_eq!(rr.method, AuthMethod::BalloonSha256);
    tprintln!("  AuthRule: verified");

    // DelegationEdge with new admin privilege type.
    let edge = DelegationEdge {
        grantor: RoleId(10),
        grantee: RoleId(20),
        privilege: PrivilegeType::ManagePrivileges,
        object_type: ObjectType::Schema,
        object_id: 50,
        granted_at: 1700000000,
    };
    let er = DelegationEdge::from_bytes(&edge.to_bytes()).expect("decode");
    assert_eq!(er.privilege, PrivilegeType::ManagePrivileges);
    assert_eq!(er.object_type, ObjectType::Schema);
    tprintln!("  DelegationEdge (admin priv types): verified");
}

// ===========================================================================
// Row-Level Security
// ===========================================================================

#[test]
fn test_rls_user_isolation() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- RLS: User Isolation ---");

    let store = RlsPolicyStore::new();

    // Create policy: users can only see their own orders
    store
        .add_policy(RlsPolicy {
            id: 1,
            name: "own_orders".to_string(),
            table_id: 42,
            command: RlsCommand::Select,
            policy_type: PolicyType::Permissive,
            roles: Vec::new(),
            using_expr: Some("user_id = current_user_id()".to_string()),
            check_expr: None,
            enabled: true,
        })
        .expect("add policy");

    // User A (RoleId 10) queries
    let result_a = store.evaluate_rls(42, RlsCommand::Select, &[RoleId(10)], false);
    assert_eq!(result_a.using_predicates.len(), 1);
    assert_eq!(result_a.using_predicates[0], "user_id = current_user_id()");
    tprintln!(
        "  User A sees filter predicate: {}",
        result_a.using_predicates[0]
    );

    // User B (RoleId 20) queries, same predicate
    let result_b = store.evaluate_rls(42, RlsCommand::Select, &[RoleId(20)], false);
    assert_eq!(result_b.using_predicates.len(), 1);
    tprintln!(
        "  User B sees filter predicate: {}",
        result_b.using_predicates[0]
    );

    // Admin (table owner) bypasses RLS
    let result_admin = store.evaluate_rls(42, RlsCommand::Select, &[RoleId(1)], true);
    assert!(result_admin.using_predicates.is_empty());
    tprintln!("  Admin (table owner): bypasses RLS, no predicates injected");
}

#[test]
fn test_rls_predicate_injection_latency() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- RLS: Predicate Injection Latency ---");

    let store = RlsPolicyStore::new();

    // Add a realistic set of policies
    store
        .add_policy(RlsPolicy {
            id: 1,
            name: "own_rows".to_string(),
            table_id: 100,
            command: RlsCommand::All,
            policy_type: PolicyType::Permissive,
            roles: Vec::new(),
            using_expr: Some("user_id = current_user_id()".to_string()),
            check_expr: None,
            enabled: true,
        })
        .expect("add");
    store
        .add_policy(RlsPolicy {
            id: 2,
            name: "public_rows".to_string(),
            table_id: 100,
            command: RlsCommand::Select,
            policy_type: PolicyType::Permissive,
            roles: Vec::new(),
            using_expr: Some("is_public = true".to_string()),
            check_expr: None,
            enabled: true,
        })
        .expect("add");
    store
        .add_policy(RlsPolicy {
            id: 3,
            name: "region_restrict".to_string(),
            table_id: 100,
            command: RlsCommand::All,
            policy_type: PolicyType::Restrictive,
            roles: Vec::new(),
            using_expr: Some("region = current_setting('region')".to_string()),
            check_expr: None,
            enabled: true,
        })
        .expect("add");

    let roles = vec![RoleId(10)];
    let iterations = 1_000_000;

    // Warmup
    for _ in 0..10_000 {
        let _ = store.evaluate_rls(100, RlsCommand::Select, &roles, false);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let result = store.evaluate_rls(100, RlsCommand::Select, &roles, false);
        std::hint::black_box(&result);
    }
    let elapsed = start.elapsed();
    let latency_ns = elapsed.as_nanos() as f64 / iterations as f64;

    tprintln!(
        "  Latency: {:.0} ns/eval ({} iterations)",
        latency_ns,
        iterations
    );
    validate_metric(
        "rls_predicate_inject",
        "latency_ns",
        vec![latency_ns],
        1000.0,
        false,
    );
    check_performance(
        "rls_predicate_inject",
        "latency_ns",
        latency_ns,
        1000.0,
        false,
    );

    // Verify the result is correct (2 permissive OR'd + 1 restrictive = 2 predicates)
    let result = store.evaluate_rls(100, RlsCommand::Select, &roles, false);
    assert_eq!(result.using_predicates.len(), 2);
    tprintln!("  Predicates generated: {}", result.using_predicates.len());
}

#[test]
fn test_rls_command_filtering() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- RLS: Command Filtering ---");

    let store = RlsPolicyStore::new();

    // SELECT-only policy
    store
        .add_policy(RlsPolicy {
            id: 1,
            name: "select_filter".to_string(),
            table_id: 42,
            command: RlsCommand::Select,
            policy_type: PolicyType::Permissive,
            roles: Vec::new(),
            using_expr: Some("visible = true".to_string()),
            check_expr: None,
            enabled: true,
        })
        .expect("add");

    // INSERT write-check policy
    store
        .add_policy(RlsPolicy {
            id: 2,
            name: "insert_check".to_string(),
            table_id: 42,
            command: RlsCommand::Insert,
            policy_type: PolicyType::Permissive,
            roles: Vec::new(),
            using_expr: None,
            check_expr: Some("user_id = current_user_id()".to_string()),
            enabled: true,
        })
        .expect("add");

    let roles = vec![RoleId(10)];

    // SELECT applies the filter
    let select_result = store.evaluate_rls(42, RlsCommand::Select, &roles, false);
    assert_eq!(select_result.using_predicates.len(), 1);
    assert!(select_result.check_predicates.is_empty());
    tprintln!(
        "  SELECT: {} using, {} check",
        select_result.using_predicates.len(),
        select_result.check_predicates.len()
    );

    // INSERT applies the check
    let insert_result = store.evaluate_rls(42, RlsCommand::Insert, &roles, false);
    assert!(insert_result.using_predicates.is_empty());
    assert_eq!(insert_result.check_predicates.len(), 1);
    tprintln!(
        "  INSERT: {} using, {} check",
        insert_result.using_predicates.len(),
        insert_result.check_predicates.len()
    );

    // DELETE has no matching policy
    let delete_result = store.evaluate_rls(42, RlsCommand::Delete, &roles, false);
    assert!(delete_result.using_predicates.is_empty());
    assert!(delete_result.check_predicates.is_empty());
    tprintln!(
        "  DELETE: {} using, {} check (no matching policy)",
        delete_result.using_predicates.len(),
        delete_result.check_predicates.len()
    );
}

#[test]
fn test_rls_role_scoped_policies() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- RLS: Role-Scoped Policies ---");

    let store = RlsPolicyStore::new();

    // Policy applies only to analyst role (RoleId 50)
    store
        .add_policy(RlsPolicy {
            id: 1,
            name: "analyst_only".to_string(),
            table_id: 42,
            command: RlsCommand::Select,
            policy_type: PolicyType::Permissive,
            roles: vec![RoleId(50)],
            using_expr: Some("department = 'analytics'".to_string()),
            check_expr: None,
            enabled: true,
        })
        .expect("add");

    // Analyst sees the predicate
    let result = store.evaluate_rls(42, RlsCommand::Select, &[RoleId(50)], false);
    assert_eq!(result.using_predicates.len(), 1);
    tprintln!(
        "  Analyst (role 50): sees {} predicates",
        result.using_predicates.len()
    );

    // Non-analyst role sees nothing (no matching permissive policy)
    let result = store.evaluate_rls(42, RlsCommand::Select, &[RoleId(99)], false);
    assert!(result.using_predicates.is_empty());
    tprintln!(
        "  Other (role 99): sees {} predicates (no match)",
        result.using_predicates.len()
    );
}

#[test]
fn test_rls_serialization_roundtrip() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- RLS: Serialization Roundtrip ---");

    let policy = RlsPolicy {
        id: 42,
        name: "complex_policy".to_string(),
        table_id: 100,
        command: RlsCommand::Update,
        policy_type: PolicyType::Restrictive,
        roles: vec![RoleId(10), RoleId(20), RoleId(30)],
        using_expr: Some("region = 'us-east' AND active = true".to_string()),
        check_expr: Some("user_id = current_user_id()".to_string()),
        enabled: true,
    };

    let bytes = policy.to_bytes();
    let restored = RlsPolicy::from_bytes(&bytes).expect("decode");
    assert_eq!(restored.id, 42);
    assert_eq!(restored.name, "complex_policy");
    assert_eq!(restored.table_id, 100);
    assert_eq!(restored.command, RlsCommand::Update);
    assert_eq!(restored.policy_type, PolicyType::Restrictive);
    assert_eq!(restored.roles.len(), 3);
    assert_eq!(
        restored.using_expr.as_deref(),
        Some("region = 'us-east' AND active = true")
    );
    assert_eq!(
        restored.check_expr.as_deref(),
        Some("user_id = current_user_id()")
    );
    assert!(restored.enabled);
    tprintln!("  RlsPolicy roundtrip: verified (all fields)");
}

// ===========================================================================
// Column Masking Policies
// ===========================================================================

#[test]
fn test_column_masking_policy_ssn() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- Column Masking: SSN Policy ---");

    let store = MaskingPolicyStore::new();
    store
        .add_policy(MaskingPolicy {
            id: 1,
            name: "ssn_mask".to_string(),
            table_id: 100,
            column_id: 5,
            function: MaskFunction::Ssn,
            exempt_roles: vec![RoleId(99)], // Admin role exempt
            enabled: true,
        })
        .expect("add");

    // Analyst (not exempt) sees masked SSN
    let mut buf = String::with_capacity(64);
    assert!(store.apply_masking(100, 5, "123-45-6789", &[RoleId(10)], &mut buf));
    assert!(buf.ends_with("6789"), "Last 4 digits preserved: {}", buf);
    tprintln!("  Analyst: 123-45-6789 -> {}", buf);

    // Admin (exempt) sees original
    assert!(!store.apply_masking(100, 5, "123-45-6789", &[RoleId(99)], &mut buf));
    tprintln!("  Admin (exempt): sees original value");
}

#[test]
fn test_column_masking_throughput() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- Column Masking: Throughput ---");

    let store = MaskingPolicyStore::new();
    store
        .add_policy(MaskingPolicy {
            id: 1,
            name: "partial_mask".to_string(),
            table_id: 100,
            column_id: 3,
            function: MaskFunction::Ssn,
            exempt_roles: Vec::new(),
            enabled: true,
        })
        .expect("add");

    let roles = vec![RoleId(10)];
    let value = "123-45-6789";
    let iterations = 1_000_000;
    let mut buf = String::with_capacity(64);

    // Warmup
    for _ in 0..10_000 {
        store.apply_masking(100, 3, value, &roles, &mut buf);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        store.apply_masking(100, 3, value, &roles, &mut buf);
        std::hint::black_box(&buf);
    }
    let elapsed = start.elapsed();
    let ops_per_sec = iterations as f64 / elapsed.as_secs_f64();

    tprintln!("  Partial mask: {:.1}M vals/sec", ops_per_sec / 1_000_000.0);
    validate_metric(
        "column_mask_partial",
        "ops_per_sec",
        vec![ops_per_sec],
        30_000_000.0,
        true,
    );
    check_performance(
        "column_mask_partial",
        "ops_per_sec",
        ops_per_sec,
        30_000_000.0,
        true,
    );

    // Hash masking throughput
    let store2 = MaskingPolicyStore::new();
    store2
        .add_policy(MaskingPolicy {
            id: 2,
            name: "hash_mask".to_string(),
            table_id: 100,
            column_id: 4,
            function: MaskFunction::Hash,
            exempt_roles: Vec::new(),
            enabled: true,
        })
        .expect("add");

    let start = Instant::now();
    for _ in 0..iterations {
        store2.apply_masking(100, 4, value, &roles, &mut buf);
        std::hint::black_box(&buf);
    }
    let elapsed = start.elapsed();
    let hash_ops = iterations as f64 / elapsed.as_secs_f64();

    tprintln!("  Hash mask: {:.1}M vals/sec", hash_ops / 1_000_000.0);
    validate_metric(
        "column_mask_hash",
        "ops_per_sec",
        vec![hash_ops],
        15_000_000.0,
        true,
    );
    check_performance(
        "column_mask_hash",
        "ops_per_sec",
        hash_ops,
        15_000_000.0,
        true,
    );
}

#[test]
fn test_column_masking_all_types() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- Column Masking: All Mask Functions ---");

    let roles = vec![RoleId(10)];

    let test_cases: Vec<(&str, &str, MaskFunction)> = vec![
        ("email_mask", "john@example.com", MaskFunction::Email),
        ("phone_mask", "555-123-4567", MaskFunction::Phone),
        ("ssn_mask", "123-45-6789", MaskFunction::Ssn),
        ("cc_mask", "4111111111111111", MaskFunction::CreditCard),
        ("hash_mask", "sensitive_data", MaskFunction::Hash),
        ("redact_mask", "anything", MaskFunction::Redact),
        (
            "partial_mask",
            "1234567890",
            MaskFunction::PartialMask {
                show_first: 2,
                show_last: 2,
                mask_char: b'*',
            },
        ),
        (
            "noise_mask",
            "50000",
            MaskFunction::NoiseMask { factor: 0.1 },
        ),
        (
            "bucket_mask",
            "75000",
            MaskFunction::BucketMask {
                boundaries: vec![50000.0, 100000.0, 200000.0],
            },
        ),
    ];

    for (name, value, func) in &test_cases {
        let store = MaskingPolicyStore::new();
        store
            .add_policy(MaskingPolicy {
                id: 1,
                name: name.to_string(),
                table_id: 100,
                column_id: 1,
                function: func.clone(),
                exempt_roles: Vec::new(),
                enabled: true,
            })
            .expect("add");

        let mut buf = String::with_capacity(64);
        let applied = store.apply_masking(100, 1, value, &roles, &mut buf);
        if applied {
            tprintln!("  {}: {} -> {}", name, value, buf);
        } else {
            tprintln!("  {}: {} -> NULL", name, value);
        }
    }
}

#[test]
fn test_column_masking_policy_serialization() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- Column Masking: Policy Serialization ---");

    let policy = MaskingPolicy {
        id: 42,
        name: "ssn_mask".to_string(),
        table_id: 100,
        column_id: 5,
        function: MaskFunction::Ssn,
        exempt_roles: vec![RoleId(10), RoleId(20)],
        enabled: true,
    };
    let bytes = policy.to_bytes();
    let restored = MaskingPolicy::from_bytes(&bytes).expect("decode");
    assert_eq!(restored.id, 42);
    assert_eq!(restored.name, "ssn_mask");
    assert_eq!(restored.table_id, 100);
    assert_eq!(restored.column_id, 5);
    assert_eq!(restored.function, MaskFunction::Ssn);
    assert_eq!(restored.exempt_roles.len(), 2);
    assert!(restored.enabled);
    tprintln!("  MaskingPolicy roundtrip: verified");
}

// ===========================================================================
// ABAC Rule Evaluation
// ===========================================================================

#[test]
fn test_abac_time_based_access() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- ABAC: Time-Based Access ---");

    let store = AbacRuleStore::new();

    // Rule: analysts can only query during business hours (9-17)
    store
        .add_rule(AbacRule {
            id: 1,
            name: "business_hours_only".to_string(),
            conditions: vec![AttributeCondition {
                attribute_key: "hour".to_string(),
                operator: AbacOperator::In,
                value: "9,10,11,12,13,14,15,16,17".to_string(),
            }],
            effect: AbacEffect::Allow,
            resource_pattern: None,
            action: None,
            enabled: true,
            roles: vec![RoleId(50)],
            priority: 10,
        })
        .expect("add");

    // Business hours attributes
    let mut bh_attrs = SessionAttributes {
        role_id: RoleId(50),
        department: None,
        region: None,
        clearance: ClassificationLevel::Public,
        ip_address: "127.0.0.1".to_string(),
        connection_time: 0,
        custom: HashMap::new(),
    };
    bh_attrs.set("hour".to_string(), "14".to_string());

    let allowed = store.evaluate_abac(&bh_attrs, None, None);
    assert!(allowed);
    tprintln!("  Business hours (hour=14): allowed={}", allowed);

    // After hours attributes
    let mut ah_attrs = SessionAttributes {
        role_id: RoleId(50),
        department: None,
        region: None,
        clearance: ClassificationLevel::Public,
        ip_address: "127.0.0.1".to_string(),
        connection_time: 0,
        custom: HashMap::new(),
    };
    ah_attrs.set("hour".to_string(), "22".to_string());

    let allowed = store.evaluate_abac(&ah_attrs, None, None);
    // No matching allow rule at hour=22, but default is allow (ABAC is additive)
    tprintln!("  After hours (hour=22): allowed={}", allowed);
}

#[test]
fn test_abac_region_based_access() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- ABAC: Region-Based Access ---");

    let store = AbacRuleStore::new();

    // Rule: Deny non-EU users from accessing EU data
    store
        .add_rule(AbacRule {
            id: 1,
            name: "eu_data_restriction".to_string(),
            conditions: vec![AttributeCondition {
                attribute_key: "region".to_string(),
                operator: AbacOperator::NotEq,
                value: "eu".to_string(),
            }],
            effect: AbacEffect::Deny,
            resource_pattern: Some("eu_customers".to_string()),
            action: None,
            enabled: true,
            roles: Vec::new(),
            priority: 10,
        })
        .expect("add");

    // EU user queries EU data
    let eu_attrs = SessionAttributes {
        role_id: RoleId(10),
        department: None,
        region: Some("eu".to_string()),
        clearance: ClassificationLevel::Public,
        ip_address: "127.0.0.1".to_string(),
        connection_time: 0,
        custom: HashMap::new(),
    };
    let allowed = store.evaluate_abac(&eu_attrs, Some("eu_customers"), None);
    assert!(allowed);
    tprintln!("  EU user -> EU data: allowed={}", allowed);

    // US user queries EU data
    let us_attrs = SessionAttributes {
        role_id: RoleId(20),
        department: None,
        region: Some("us".to_string()),
        clearance: ClassificationLevel::Public,
        ip_address: "127.0.0.1".to_string(),
        connection_time: 0,
        custom: HashMap::new(),
    };
    let denied = store.evaluate_abac(&us_attrs, Some("eu_customers"), None);
    assert!(!denied);
    tprintln!("  US user -> EU data: allowed={}", denied);
}

#[test]
fn test_abac_evaluate_latency() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- ABAC: Rule Evaluation Latency ---");

    let store = AbacRuleStore::new();

    // Add 10 rules with various conditions
    for i in 0..10 {
        store
            .add_rule(AbacRule {
                id: i,
                name: format!("rule_{}", i),
                conditions: vec![
                    AttributeCondition {
                        attribute_key: "department".to_string(),
                        operator: AbacOperator::Eq,
                        value: format!("dept_{}", i),
                    },
                    AttributeCondition {
                        attribute_key: "region".to_string(),
                        operator: AbacOperator::Eq,
                        value: "us".to_string(),
                    },
                ],
                effect: if i == 5 {
                    AbacEffect::Deny
                } else {
                    AbacEffect::Allow
                },
                resource_pattern: None,
                action: None,
                enabled: true,
                roles: Vec::new(),
                priority: i as u16,
            })
            .expect("add");
    }

    let attrs = SessionAttributes {
        role_id: RoleId(10),
        department: Some("dept_3".to_string()),
        region: Some("us".to_string()),
        clearance: ClassificationLevel::Public,
        ip_address: "127.0.0.1".to_string(),
        connection_time: 0,
        custom: HashMap::new(),
    };

    let iterations = 1_000_000;

    // Warmup
    for _ in 0..10_000 {
        let _ = store.evaluate_abac(&attrs, None, None);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let result = store.evaluate_abac(&attrs, None, None);
        std::hint::black_box(&result);
    }
    let elapsed = start.elapsed();
    let latency_ns = elapsed.as_nanos() as f64 / iterations as f64;

    tprintln!(
        "  Latency: {:.0} ns/eval ({} iterations)",
        latency_ns,
        iterations
    );
    validate_metric(
        "abac_evaluate",
        "latency_ns",
        vec![latency_ns],
        2000.0,
        false,
    );
    check_performance("abac_evaluate", "latency_ns", latency_ns, 2000.0, false);
}

#[test]
fn test_abac_rule_serialization() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- ABAC: Rule Serialization ---");

    let rule = AbacRule {
        id: 42,
        name: "complex_rule".to_string(),
        conditions: vec![
            AttributeCondition {
                attribute_key: "department".to_string(),
                operator: AbacOperator::Eq,
                value: "engineering".to_string(),
            },
            AttributeCondition {
                attribute_key: "clearance".to_string(),
                operator: AbacOperator::In,
                value: "secret,top_secret".to_string(),
            },
        ],
        effect: AbacEffect::Allow,
        resource_pattern: Some("sensitive_table".to_string()),
        action: Some(0), // Select
        enabled: true,
        roles: vec![RoleId(10), RoleId(20)],
        priority: 100,
    };

    let bytes = rule.to_bytes();
    let restored = AbacRule::from_bytes(&bytes).expect("decode");
    assert_eq!(restored.id, 42);
    assert_eq!(restored.name, "complex_rule");
    assert_eq!(restored.conditions.len(), 2);
    assert_eq!(restored.effect, AbacEffect::Allow);
    assert_eq!(
        restored.resource_pattern.as_deref(),
        Some("sensitive_table")
    );
    assert_eq!(restored.action, Some(0));
    assert!(restored.enabled);
    assert_eq!(restored.roles.len(), 2);
    assert_eq!(restored.priority, 100);
    tprintln!("  AbacRule roundtrip: verified (all fields)");
}

// ===========================================================================
// Encryption Round-Trip
// ===========================================================================

#[test]
fn test_encryption_roundtrip() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- Encryption: AES-GCM Round-Trip ---");

    let key_128 = [0x42u8; 16];
    let key_256 = [0x42u8; 32];

    // AES-128-GCM round-trip
    let plaintext = b"Hello, ZyronDB! This is sensitive data.";
    let ciphertext =
        encrypt_value(plaintext, &key_128, EncryptionAlgorithm::Aes128Gcm, &[]).expect("encrypt");
    let decrypted =
        decrypt_value(&ciphertext, &key_128, EncryptionAlgorithm::Aes128Gcm, &[]).expect("decrypt");
    assert_eq!(decrypted, plaintext);
    tprintln!(
        "  AES-128-GCM: {} bytes -> {} bytes -> {} bytes",
        plaintext.len(),
        ciphertext.len(),
        decrypted.len()
    );

    // AES-256-GCM round-trip
    let ciphertext =
        encrypt_value(plaintext, &key_256, EncryptionAlgorithm::Aes256Gcm, &[]).expect("encrypt");
    let decrypted =
        decrypt_value(&ciphertext, &key_256, EncryptionAlgorithm::Aes256Gcm, &[]).expect("decrypt");
    assert_eq!(decrypted, plaintext);
    tprintln!("  AES-256-GCM: round-trip verified");
}

#[test]
fn test_encryption_nonce_uniqueness() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- Encryption: Nonce Uniqueness ---");

    let key = [0x42u8; 32];
    let plaintext = b"same plaintext for all";
    let count = 10_000;

    let mut nonces: Vec<[u8; 12]> = Vec::with_capacity(count);
    for _ in 0..count {
        let ct =
            encrypt_value(plaintext, &key, EncryptionAlgorithm::Aes256Gcm, &[]).expect("encrypt");
        let mut nonce = [0u8; 12];
        nonce.copy_from_slice(&ct[..12]);
        nonces.push(nonce);
    }

    // Verify all nonces are unique
    nonces.sort();
    let unique_count = nonces.windows(2).filter(|w| w[0] != w[1]).count() + 1;
    assert_eq!(unique_count, count, "All nonces must be unique");
    tprintln!("  {}/{} nonces unique: PASS", unique_count, count);
}

#[test]
fn test_encryption_aad_prevents_swap() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- Encryption: AAD Prevents Column Swap ---");

    let key = [0x42u8; 32];
    let plaintext = b"sensitive value";

    // Encrypt with AAD for column A
    let aad_a = b"table=1,col=3";
    let ct_a =
        encrypt_value(plaintext, &key, EncryptionAlgorithm::Aes256Gcm, aad_a).expect("encrypt");

    // Decrypt with correct AAD succeeds
    let result = decrypt_value(&ct_a, &key, EncryptionAlgorithm::Aes256Gcm, aad_a);
    assert!(result.is_ok());
    tprintln!("  Correct AAD: decryption succeeded");

    // Decrypt with wrong AAD fails (ciphertext swapped to different column)
    let aad_b = b"table=1,col=7";
    let result = decrypt_value(&ct_a, &key, EncryptionAlgorithm::Aes256Gcm, aad_b);
    assert!(result.is_err());
    tprintln!("  Wrong AAD: decryption rejected (tag mismatch)");
}

#[test]
fn test_encryption_tamper_detection() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- Encryption: Tamper Detection ---");

    let key = [0x42u8; 32];
    let plaintext = b"authenticated data";

    let mut ct =
        encrypt_value(plaintext, &key, EncryptionAlgorithm::Aes256Gcm, &[]).expect("encrypt");

    // Tamper with a ciphertext byte
    let mid = 12 + ct.len() / 2 - 8; // Middle of ciphertext section
    ct[mid] ^= 0xFF;

    let result = decrypt_value(&ct, &key, EncryptionAlgorithm::Aes256Gcm, &[]);
    assert!(result.is_err());
    tprintln!("  Tampered ciphertext: decryption rejected");
}

#[test]
fn test_encryption_throughput() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- Encryption: Throughput ---");

    let key = [0x42u8; 32];

    // Use 1KB blocks to measure throughput
    let block_size = 1024;
    let plaintext = vec![0xABu8; block_size];
    let iterations = 100_000;

    // Warmup
    for _ in 0..1_000 {
        let ct =
            encrypt_value(&plaintext, &key, EncryptionAlgorithm::Aes256Gcm, &[]).expect("encrypt");
        let _ = decrypt_value(&ct, &key, EncryptionAlgorithm::Aes256Gcm, &[]).expect("decrypt");
    }

    // Encrypt throughput
    let start = Instant::now();
    for _ in 0..iterations {
        let ct =
            encrypt_value(&plaintext, &key, EncryptionAlgorithm::Aes256Gcm, &[]).expect("encrypt");
        std::hint::black_box(&ct);
    }
    let elapsed = start.elapsed();
    let bytes_total = iterations as f64 * block_size as f64;
    let encrypt_gbps = bytes_total / elapsed.as_secs_f64() / 1_000_000_000.0;
    tprintln!(
        "  Encrypt: {:.2} GB/sec ({}x 1KB blocks)",
        encrypt_gbps,
        iterations
    );
    validate_metric(
        "authenticated_encrypt",
        "gb_per_sec",
        vec![encrypt_gbps],
        4.0,
        true,
    );
    check_performance(
        "authenticated_encrypt",
        "gb_per_sec",
        encrypt_gbps,
        4.0,
        true,
    );

    // Decrypt throughput
    let ct = encrypt_value(&plaintext, &key, EncryptionAlgorithm::Aes256Gcm, &[]).expect("encrypt");

    let start = Instant::now();
    for _ in 0..iterations {
        let pt = decrypt_value(&ct, &key, EncryptionAlgorithm::Aes256Gcm, &[]).expect("decrypt");
        std::hint::black_box(&pt);
    }
    let elapsed = start.elapsed();
    let decrypt_gbps = bytes_total / elapsed.as_secs_f64() / 1_000_000_000.0;
    tprintln!(
        "  Decrypt: {:.2} GB/sec ({}x 1KB blocks)",
        decrypt_gbps,
        iterations
    );
    validate_metric(
        "authenticated_decrypt",
        "gb_per_sec",
        vec![decrypt_gbps],
        4.0,
        true,
    );
    check_performance(
        "authenticated_decrypt",
        "gb_per_sec",
        decrypt_gbps,
        4.0,
        true,
    );
}

#[test]
fn test_encryption_key_store() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- Encryption: LocalKeyStore ---");

    let master_key = [0x42u8; 32];
    let ks = LocalKeyStore::new(master_key);

    // Create keys
    let id_128 = ks
        .create_key(EncryptionAlgorithm::Aes128Gcm)
        .expect("create 128");
    let id_256 = ks
        .create_key(EncryptionAlgorithm::Aes256Gcm)
        .expect("create 256");
    tprintln!("  Created key IDs: {}, {}", id_128, id_256);

    // Retrieve keys
    let key_128 = ks.get_key(id_128).expect("get 128");
    assert_eq!(key_128.len(), 16);
    let key_256 = ks.get_key(id_256).expect("get 256");
    assert_eq!(key_256.len(), 32);
    tprintln!(
        "  Retrieved: 128-bit ({} bytes), 256-bit ({} bytes)",
        key_128.len(),
        key_256.len()
    );

    // Encrypt/decrypt using store-managed key
    let plaintext = b"managed key encryption test";
    let ct =
        encrypt_value(plaintext, &key_256, EncryptionAlgorithm::Aes256Gcm, &[]).expect("encrypt");
    let pt = decrypt_value(&ct, &key_256, EncryptionAlgorithm::Aes256Gcm, &[]).expect("decrypt");
    assert_eq!(pt, plaintext);
    tprintln!("  Encrypt/decrypt with store-managed key: verified");

    // Rotate key
    let new_id = ks.rotate_key(id_256).expect("rotate");
    assert_ne!(new_id, id_256);
    let new_key = ks.get_key(new_id).expect("get new");
    assert_eq!(new_key.len(), 32);
    assert_ne!(new_key, key_256);
    tprintln!("  Key rotation: old={}, new={}", id_256, new_id);

    // Old key should be deleted after rotation
    assert!(ks.get_key(id_256).is_err());
    tprintln!("  Old key deleted: verified");
}

#[test]
fn test_encryption_column_config_serialization() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- Encryption: ColumnEncryption Serialization ---");

    use zyron_auth::encryption::ColumnEncryption;

    let config = ColumnEncryption {
        table_id: 42,
        column_id: 7,
        algorithm: EncryptionAlgorithm::Aes256Gcm,
        key_id: 123,
    };
    let bytes = config.to_bytes();
    let restored = ColumnEncryption::from_bytes(&bytes).expect("decode");
    assert_eq!(restored.table_id, 42);
    assert_eq!(restored.column_id, 7);
    assert_eq!(restored.algorithm, EncryptionAlgorithm::Aes256Gcm);
    assert_eq!(restored.key_id, 123);
    tprintln!("  ColumnEncryption roundtrip: verified");
}

// ===========================================================================
// Security Labels (Mandatory Access Control)
// ===========================================================================

#[test]
fn test_security_label_access_control() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- Security Labels: MAC Access Control ---");

    let mac = MandatoryAccessControl::new();

    // User with "secret:finance" label
    mac.set_subject_label(
        RoleId(10),
        SecurityLabel::new(SecurityLevel::Secret, vec!["finance".to_string()]),
    );

    // Row with "confidential:finance" label -> visible (user level >= row level, compartment match)
    mac.set_object_label(
        ObjectType::Table,
        1,
        SecurityLabel::new(SecurityLevel::Confidential, vec!["finance".to_string()]),
    );
    assert!(mac.check_access(RoleId(10), ObjectType::Table, 1));
    tprintln!("  secret:finance user -> confidential:finance row: VISIBLE");

    // Row with "top_secret:finance" label -> not visible (user level < row level)
    mac.set_object_label(
        ObjectType::Table,
        2,
        SecurityLabel::new(SecurityLevel::TopSecret, vec!["finance".to_string()]),
    );
    assert!(!mac.check_access(RoleId(10), ObjectType::Table, 2));
    tprintln!("  secret:finance user -> top_secret:finance row: HIDDEN");

    // Row with "confidential:hr" label -> not visible (user missing hr compartment)
    mac.set_object_label(
        ObjectType::Table,
        3,
        SecurityLabel::new(SecurityLevel::Confidential, vec!["hr".to_string()]),
    );
    assert!(!mac.check_access(RoleId(10), ObjectType::Table, 3));
    tprintln!("  secret:finance user -> confidential:hr row: HIDDEN (missing compartment)");
}

#[test]
fn test_security_label_check_latency() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- Security Labels: Check Latency ---");

    let mac = MandatoryAccessControl::new();
    mac.set_subject_label(
        RoleId(10),
        SecurityLabel::new(
            SecurityLevel::Secret,
            vec!["finance".to_string(), "hr".to_string(), "legal".to_string()],
        ),
    );
    mac.set_object_label(
        ObjectType::Table,
        42,
        SecurityLabel::new(
            SecurityLevel::Confidential,
            vec!["finance".to_string(), "hr".to_string()],
        ),
    );

    let iterations = 5_000_000;

    // Warmup
    for _ in 0..50_000 {
        let _ = mac.check_access(RoleId(10), ObjectType::Table, 42);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let result = mac.check_access(RoleId(10), ObjectType::Table, 42);
        std::hint::black_box(&result);
    }
    let elapsed = start.elapsed();
    let latency_ns = elapsed.as_nanos() as f64 / iterations as f64;

    tprintln!(
        "  Latency: {:.1} ns/check ({} iterations)",
        latency_ns,
        iterations
    );
    validate_metric(
        "security_label_check",
        "latency_ns",
        vec![latency_ns],
        100.0,
        false,
    );
    check_performance(
        "security_label_check",
        "latency_ns",
        latency_ns,
        100.0,
        false,
    );
}

#[test]
fn test_security_label_dominance() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- Security Labels: Dominance Rules ---");

    // Same level, same compartments = dominates
    let a = SecurityLabel::new(SecurityLevel::Secret, vec!["finance".to_string()]);
    let b = SecurityLabel::new(SecurityLevel::Secret, vec!["finance".to_string()]);
    assert!(a.dominates(&b));
    tprintln!("  secret:finance >= secret:finance: true");

    // Higher level, superset compartments = dominates
    let subject = SecurityLabel::new(
        SecurityLevel::TopSecret,
        vec!["finance".to_string(), "hr".to_string()],
    );
    let object = SecurityLabel::new(SecurityLevel::Secret, vec!["finance".to_string()]);
    assert!(subject.dominates(&object));
    tprintln!("  top_secret:finance,hr >= secret:finance: true");

    // Higher level but missing compartment = does not dominate
    let subject = SecurityLabel::new(SecurityLevel::TopSecret, vec!["finance".to_string()]);
    let object = SecurityLabel::new(
        SecurityLevel::Secret,
        vec!["finance".to_string(), "hr".to_string()],
    );
    assert!(!subject.dominates(&object));
    tprintln!("  top_secret:finance >= secret:finance,hr: false (missing hr)");

    // Unclassified with no compartments can see unclassified with no compartments
    let a = SecurityLabel::new(SecurityLevel::Unclassified, vec![]);
    let b = SecurityLabel::new(SecurityLevel::Unclassified, vec![]);
    assert!(a.dominates(&b));
    tprintln!("  unclassified >= unclassified: true");
}

#[test]
fn test_security_label_serialization() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- Security Labels: Serialization ---");

    use zyron_auth::security_label::{ObjectSecurityLabel, SubjectSecurityLabel};

    let label = SecurityLabel::new(
        SecurityLevel::TopSecret,
        vec![
            "alpha".to_string(),
            "bravo".to_string(),
            "charlie".to_string(),
        ],
    );
    let bytes = label.to_bytes();
    let (restored, consumed) = SecurityLabel::from_bytes(&bytes).expect("decode");
    assert_eq!(restored, label);
    assert_eq!(consumed, bytes.len());
    tprintln!(
        "  SecurityLabel roundtrip: verified ({} bytes)",
        bytes.len()
    );

    let osl = ObjectSecurityLabel {
        object_type: ObjectType::Table,
        object_id: 42,
        label: SecurityLabel::new(SecurityLevel::Secret, vec!["finance".to_string()]),
    };
    let bytes = osl.to_bytes();
    let restored = ObjectSecurityLabel::from_bytes(&bytes).expect("decode");
    assert_eq!(restored.object_type, ObjectType::Table);
    assert_eq!(restored.object_id, 42);
    tprintln!("  ObjectSecurityLabel roundtrip: verified");

    let ssl = SubjectSecurityLabel {
        role_id: RoleId(10),
        label: SecurityLabel::new(
            SecurityLevel::Secret,
            vec!["finance".to_string(), "hr".to_string()],
        ),
    };
    let bytes = ssl.to_bytes();
    let restored = SubjectSecurityLabel::from_bytes(&bytes).expect("decode");
    assert_eq!(restored.role_id, RoleId(10));
    tprintln!("  SubjectSecurityLabel roundtrip: verified");
}

// ===========================================================================
// Webhook Verification
// ===========================================================================

#[test]
fn test_webhook_stripe_verification() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- Webhook: Stripe Verification ---");

    let secret = "whsec_test_secret_12345";
    let payload = b"{\"type\":\"payment_intent.succeeded\",\"data\":{\"amount\":2000}}";

    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        .to_string();
    let signed = format!(
        "{}.{}",
        timestamp,
        std::str::from_utf8(payload).expect("utf8")
    );
    let sig = webhook::compute_hmac_sha256(signed.as_bytes(), secret.as_bytes()).expect("compute");
    let header = format!("t={},v1={}", timestamp, sig);

    // Correct payload verifies
    let result = webhook::verify_stripe_webhook(payload, &header, secret).expect("verify");
    assert!(result);
    tprintln!("  Correct payload: verified=true");

    // Tampered payload fails
    let tampered = b"{\"type\":\"payment_intent.succeeded\",\"data\":{\"amount\":99999}}";
    let result = webhook::verify_stripe_webhook(tampered, &header, secret).expect("verify");
    assert!(!result);
    tprintln!("  Tampered payload: verified=false");

    // Wrong secret fails
    let result = webhook::verify_stripe_webhook(payload, &header, "wrong_secret").expect("verify");
    assert!(!result);
    tprintln!("  Wrong secret: verified=false");
}

#[test]
fn test_webhook_github_verification() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- Webhook: GitHub Verification ---");

    let secret = "github_webhook_secret";
    let payload = b"{\"action\":\"push\",\"ref\":\"refs/heads/main\"}";

    let sig = webhook::compute_hmac_sha256(payload, secret.as_bytes()).expect("compute");
    let header = format!("sha256={}", sig);

    // Correct
    let result = webhook::verify_github_webhook(payload, &header, secret).expect("verify");
    assert!(result);
    tprintln!("  Correct payload: verified=true");

    // Tampered
    let tampered = b"{\"action\":\"push\",\"ref\":\"refs/heads/evil\"}";
    let result = webhook::verify_github_webhook(tampered, &header, secret).expect("verify");
    assert!(!result);
    tprintln!("  Tampered payload: verified=false");
}

#[test]
fn test_webhook_slack_verification() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- Webhook: Slack Verification ---");

    let secret = "slack_signing_secret";
    let payload = b"token=xoxb&team_id=T12345";
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        .to_string();

    let signed = format!(
        "v0:{}:{}",
        timestamp,
        std::str::from_utf8(payload).expect("utf8")
    );
    let sig = webhook::compute_hmac_sha256(signed.as_bytes(), secret.as_bytes()).expect("compute");
    let header = format!("v0={}", sig);

    // Correct
    let result =
        webhook::verify_slack_webhook(payload, &timestamp, &header, secret).expect("verify");
    assert!(result);
    tprintln!("  Correct payload: verified=true");

    // Wrong secret
    let result =
        webhook::verify_slack_webhook(payload, &timestamp, &header, "wrong").expect("verify");
    assert!(!result);
    tprintln!("  Wrong secret: verified=false");
}

#[test]
fn test_webhook_verify_latency() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- Webhook: Verification Latency ---");

    let secret = "benchmark_secret";
    let payload = b"{\"event\":\"test\",\"data\":{\"id\":12345,\"amount\":100}}";

    let sig = webhook::compute_hmac_sha256(payload, secret.as_bytes()).expect("compute");
    let header = format!("sha256={}", sig);

    let iterations = 500_000;

    // Warmup
    for _ in 0..5_000 {
        let _ = webhook::verify_github_webhook(payload, &header, secret);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let result = webhook::verify_github_webhook(payload, &header, secret).expect("verify");
        std::hint::black_box(&result);
    }
    let elapsed = start.elapsed();
    let latency_us = elapsed.as_nanos() as f64 / iterations as f64 / 1000.0;

    tprintln!(
        "  Latency: {:.2} us/verify ({} iterations)",
        latency_us,
        iterations
    );
    validate_metric(
        "webhook_signature_verify",
        "latency_us",
        vec![latency_us],
        20.0,
        false,
    );
    check_performance(
        "webhook_signature_verify",
        "latency_us",
        latency_us,
        20.0,
        false,
    );
}

// ===========================================================================
// Crypto SQL Functions
// ===========================================================================

#[test]
fn test_crypto_password_hash_verify() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- Crypto Functions: Password Hash/Verify ---");

    use zyron_auth::crypto_functions;

    let password = "S3cure_P@ssw0rd!";

    let hash = crypto_functions::password_hash(password).expect("hash");
    assert!(hash.starts_with("$balloon-aes$"));
    tprintln!("  Hash: {}...{}", &hash[..25], &hash[hash.len() - 10..]);

    let valid = crypto_functions::password_verify(password, &hash).expect("verify");
    assert!(valid);
    tprintln!("  Correct password: verified=true");

    let invalid = crypto_functions::password_verify("wrong_password", &hash).expect("verify");
    assert!(!invalid);
    tprintln!("  Wrong password: verified=false");
}

#[test]
fn test_crypto_key_generation() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("auth");
    tprintln!("--- Crypto Functions: Key Generation ---");

    use zyron_auth::crypto_functions;

    // 128-bit symmetric key
    let key128 = crypto_functions::generate_symmetric_key(128).expect("gen 128");
    assert_eq!(key128.len(), 16);
    tprintln!("  128-bit key: {} bytes", key128.len());

    // 256-bit symmetric key
    let key256 = crypto_functions::generate_symmetric_key(256).expect("gen 256");
    assert_eq!(key256.len(), 32);
    tprintln!("  256-bit key: {} bytes", key256.len());

    // Keys are unique
    let key256_2 = crypto_functions::generate_symmetric_key(256).expect("gen 256");
    assert_ne!(key256, key256_2);
    tprintln!("  Uniqueness: two 256-bit keys differ");

    // Invalid key size
    assert!(crypto_functions::generate_symmetric_key(64).is_err());
    tprintln!("  Invalid size (64 bits): rejected");

    // Signing keypair
    let (private_key, public_key) = crypto_functions::generate_signing_keypair().expect("gen kp");
    assert_eq!(private_key.len(), 32);
    assert_eq!(public_key.len(), 32);
    tprintln!(
        "  Signing keypair: {} byte private, {} byte public",
        private_key.len(),
        public_key.len()
    );
}

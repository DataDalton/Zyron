//! Auth Benchmark Suite
//!
//! Validates password hashing, API keys, JWTs, TOTP, privilege checks,
//! role hierarchy traversal, classification, data masking, time windows,
//! governance analytics, concurrent privilege access, break-glass,
//! admin privileges, SecurityContext, ABAC, tagging, row ownership,
//! session binding, auth rules, temporal/column-level privileges,
//! delegation chains, and two-person approval.
//!
//! Run: cargo test -p zyron-auth --test auth_bench --release -- --nocapture

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

use zyron_auth::abac::{AbacPolicy, AbacStore, SessionAttributes};
use zyron_auth::auth_rules::{AuthMethod, AuthResolver, AuthRule, ConnectionType};
use zyron_auth::balloon::{self, BalloonParams};
use zyron_auth::breakglass::BreakGlassManager;
use zyron_auth::classification::{ClassificationLevel, ClassificationStore, ColumnClassification};
use zyron_auth::context::SecurityContext;
use zyron_auth::credentials::{
    ApiKeyCredential, JwtAlgorithm, JwtClaims, JwtCredential, PasswordCredential, TotpCredential,
};
use zyron_auth::governance::{
    DelegationEdge, GovernanceManager, PrivilegeAnalytics, TwoPersonOperation, TwoPersonRule,
};
use zyron_auth::masking::{self, MaskFunction};
use zyron_auth::privilege::{
    GrantEntry, ObjectType, PrivilegeDecision, PrivilegeState, PrivilegeStore, PrivilegeType,
};
use zyron_auth::role::{RoleHierarchy, RoleId, RoleMembership, UserId};
use zyron_auth::row_ownership::{RowOwnershipConfig, RowOwnershipStore};
use zyron_auth::session_binding::{QueryLimitStore, QueryLimits, TimeWindow};
use zyron_auth::tagging::{ObjectTag, TagStore};
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

    // With AES-based compression, test params (1024 blocks) complete in ~100us.
    // Production params (64MB/3 rounds) are validated in test_password_authentication.
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

    let start = Instant::now();
    let mut last_masked = None;
    for _ in 0..count {
        last_masked = masking::apply_mask(email, &MaskFunction::Email);
    }
    let last_masked = last_masked.expect("email mask returns Some");
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

    // Single-threaded baseline.
    let effective_roles = vec![RoleId(0), RoleId(1), RoleId(2)];
    let baseline_start = Instant::now();
    let mut baseline_allowed = 0u64;
    for i in 0..checks_per_thread {
        let object_id = (i % 100) as u32;
        let d = store.check_privilege(
            &effective_roles,
            PrivilegeType::Select,
            ObjectType::Table,
            object_id,
            None,
            now,
        );
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

    // Multi-threaded run.
    let start = Instant::now();
    let mut handles = Vec::with_capacity(thread_count);
    for _ in 0..thread_count {
        let store_clone = store.clone();
        let roles = effective_roles.clone();
        let handle = std::thread::spawn(move || {
            let mut local_allowed = 0u64;
            for i in 0..checks_per_thread {
                let object_id = (i % 100) as u32;
                let d = store_clone.check_privilege(
                    &roles,
                    PrivilegeType::Select,
                    ObjectType::Table,
                    object_id,
                    None,
                    now,
                );
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

    let email = masking::apply_mask("user@example.com", &MaskFunction::Email).expect("email mask");
    assert!(email.contains("@example.com"));
    tprintln!("  Email: user@example.com -> {}", email);

    let phone = masking::apply_mask("555-123-4567", &MaskFunction::Phone).expect("phone mask");
    tprintln!("  Phone: 555-123-4567 -> {}", phone);

    let ssn = masking::apply_mask("123-45-6789", &MaskFunction::Ssn).expect("ssn mask");
    tprintln!("  SSN: 123-45-6789 -> {}", ssn);

    let cc = masking::apply_mask("4111111111111111", &MaskFunction::CreditCard).expect("cc mask");
    tprintln!("  CreditCard: 4111111111111111 -> {}", cc);

    let null_masked = masking::apply_mask("anything", &MaskFunction::Null);
    assert!(null_masked.is_none(), "Null mask returns None (SQL NULL)");
    tprintln!("  Null: anything -> None (SQL NULL)");

    assert_eq!(
        masking::apply_mask("sensitive", &MaskFunction::Redact),
        Some("[REDACTED]".to_string())
    );
    tprintln!("  Redact: -> [REDACTED]");

    let h1 = masking::apply_mask("data", &MaskFunction::Hash).expect("hash mask");
    let h2 = masking::apply_mask("data", &MaskFunction::Hash).expect("hash mask");
    assert_eq!(h1, h2, "Hash must be deterministic");
    tprintln!("  Hash: deterministic, {} chars", h1.len());

    let partial =
        masking::apply_mask("Hello World", &MaskFunction::Partial(3)).expect("partial mask");
    tprintln!("  Partial(3): Hello World -> {}", partial);
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

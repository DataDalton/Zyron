//! Phase 17 security additions: new privilege variants and two-person gating
//! for irreversible data-lifecycle operations.

use zyron_auth::governance::{TwoPersonManager, TwoPersonOperation, TwoPersonRule};
use zyron_auth::{PrivilegeType, RoleId};

#[test]
fn new_lifecycle_privileges_roundtrip() {
    for p in [
        PrivilegeType::ManageLegalHold,
        PrivilegeType::ManageDataLifecycle,
        PrivilegeType::ManageRetention,
        PrivilegeType::ManageErasure,
        PrivilegeType::ManageCryptoShred,
    ] {
        let code = p as u8;
        assert_eq!(PrivilegeType::from_u8(code).unwrap(), p);
    }
    // They are part of the concrete privilege set (so GRANT ALL covers them).
    let all = PrivilegeType::concrete_types();
    for p in [
        PrivilegeType::ManageLegalHold,
        PrivilegeType::ManageErasure,
        PrivilegeType::ManageCryptoShred,
    ] {
        assert!(all.contains(&p), "{p:?} missing from concrete_types");
    }
}

#[test]
fn two_person_gate_blocks_solo_then_allows_after_approval() {
    let mgr = TwoPersonManager::new();
    // No rule configured -> the op is not gated.
    assert!(!mgr.requires_approval(TwoPersonOperation::ForgetUser));

    mgr.add_rule(TwoPersonRule {
        operation: TwoPersonOperation::ForgetUser,
        required_role: None,
        timeout_secs: 3600,
    });
    assert!(mgr.requires_approval(TwoPersonOperation::ForgetUser));

    // Requester registers a pending approval (the solo attempt is blocked by
    // the dispatch layer which returns this id as an error).
    let id = mgr
        .request_approval(
            RoleId(10),
            TwoPersonOperation::ForgetUser,
            "FORGET USER 'u-1'".into(),
        )
        .expect("request approval");
    assert!(id >= 1);
    assert_eq!(mgr.pending_approvals().len(), 1);

    // A second authorized role approves.
    mgr.approve(id, RoleId(20)).expect("approve");
}

#[test]
fn two_person_release_legal_hold_and_retention_lock_are_distinct_ops() {
    let mgr = TwoPersonManager::new();
    mgr.add_rule(TwoPersonRule {
        operation: TwoPersonOperation::ReleaseLegalHold,
        required_role: None,
        timeout_secs: 60,
    });
    // Only the configured op is gated; others remain ungated.
    assert!(mgr.requires_approval(TwoPersonOperation::ReleaseLegalHold));
    assert!(!mgr.requires_approval(TwoPersonOperation::RetentionLock));
    assert!(!mgr.requires_approval(TwoPersonOperation::CryptoShred));
}

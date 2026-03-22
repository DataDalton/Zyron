//! Privilege analytics, delegation chains, and two-person integrity.
//!
//! PrivilegeAnalytics tracks which privileges are actually used.
//! DelegationTracker records grant chains for cascade revocation.
//! TwoPersonManager requires a second approver for destructive operations.

use crate::privilege::{ObjectType, PrivilegeType};
use crate::rcu::{Rcu, RcuMap};
use crate::role::RoleId;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use zyron_common::{Result, ZyronError};

/// Returns the current Unix timestamp in seconds.
fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ---------------------------------------------------------------------------
// Privilege Analytics
// ---------------------------------------------------------------------------

/// Tracks atomic counters for a single privilege usage entry.
pub struct PrivilegeUsageRecord {
    pub role_id: RoleId,
    pub privilege: PrivilegeType,
    pub object_type: ObjectType,
    pub object_id: u32,
    pub last_used: AtomicU64,
    pub use_count: AtomicU64,
}

/// Non-atomic snapshot of a PrivilegeUsageRecord for reporting.
#[derive(Debug, Clone)]
pub struct PrivilegeUsageSummary {
    pub role_id: RoleId,
    pub privilege: PrivilegeType,
    pub object_type: ObjectType,
    pub object_id: u32,
    pub last_used: u64,
    pub use_count: u64,
}

/// Tracks privilege usage with atomic counters.
/// The key is (role_id, privilege, object_type, object_id) packed as (u32, u8, u8, u32).
pub struct PrivilegeAnalytics {
    usage: scc::HashMap<(u32, u8, u8, u32), PrivilegeUsageRecord>,
    pub tracking_since: u64,
}

impl PrivilegeAnalytics {
    pub fn new() -> Self {
        Self {
            usage: scc::HashMap::new(),
            tracking_since: now_secs(),
        }
    }

    /// Records a privilege usage. Updates atomically if the entry exists, otherwise inserts.
    pub fn record_usage(
        &self,
        role_id: RoleId,
        privilege: PrivilegeType,
        object_type: ObjectType,
        object_id: u32,
    ) {
        let key = (role_id.0, privilege as u8, object_type as u8, object_id);
        let now = now_secs();

        let found = self.usage.read_sync(&key, |_k, v| {
            v.last_used.store(now, Ordering::Relaxed);
            v.use_count.fetch_add(1, Ordering::Relaxed);
        });
        if found.is_none() {
            let record = PrivilegeUsageRecord {
                role_id,
                privilege,
                object_type,
                object_id,
                last_used: AtomicU64::new(now),
                use_count: AtomicU64::new(1),
            };
            // Another thread may have inserted between read and insert.
            // insert_sync returns Err if key exists, which is fine.
            let _ = self.usage.insert_sync(key, record);
        }
    }

    /// Returns a usage report for the given role.
    pub fn usage_report(&self, role_id: RoleId) -> Vec<PrivilegeUsageSummary> {
        let mut results = Vec::new();
        self.usage.iter_sync(|_k, v| {
            if v.role_id == role_id {
                results.push(PrivilegeUsageSummary {
                    role_id: v.role_id,
                    privilege: v.privilege,
                    object_type: v.object_type,
                    object_id: v.object_id,
                    last_used: v.last_used.load(Ordering::Relaxed),
                    use_count: v.use_count.load(Ordering::Relaxed),
                });
            }
            true
        });
        results
    }
}

// ---------------------------------------------------------------------------
// Delegation Tracking
// ---------------------------------------------------------------------------

/// A single edge in the delegation graph: grantor granted privilege to grantee.
#[derive(Debug, Clone)]
pub struct DelegationEdge {
    pub grantor: RoleId,
    pub grantee: RoleId,
    pub privilege: PrivilegeType,
    pub object_type: ObjectType,
    pub object_id: u32,
    pub granted_at: u64,
}

impl DelegationEdge {
    /// Serializes to bytes.
    /// Layout: grantor(4) + grantee(4) + privilege(1) + object_type(1) + object_id(4) + granted_at(8) = 22 bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(22);
        buf.extend_from_slice(&self.grantor.0.to_le_bytes());
        buf.extend_from_slice(&self.grantee.0.to_le_bytes());
        buf.push(self.privilege as u8);
        buf.push(self.object_type as u8);
        buf.extend_from_slice(&self.object_id.to_le_bytes());
        buf.extend_from_slice(&self.granted_at.to_le_bytes());
        buf
    }

    /// Deserializes from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < 22 {
            return Err(ZyronError::DecodingFailed(
                "DelegationEdge data too short".to_string(),
            ));
        }
        let grantor = RoleId(u32::from_le_bytes([data[0], data[1], data[2], data[3]]));
        let grantee = RoleId(u32::from_le_bytes([data[4], data[5], data[6], data[7]]));
        let privilege = PrivilegeType::from_u8(data[8])?;
        let object_type = ObjectType::from_u8(data[9])?;
        let object_id = u32::from_le_bytes([data[10], data[11], data[12], data[13]]);
        let granted_at = u64::from_le_bytes([
            data[14], data[15], data[16], data[17], data[18], data[19], data[20], data[21],
        ]);
        Ok(Self {
            grantor,
            grantee,
            privilege,
            object_type,
            object_id,
            granted_at,
        })
    }
}

/// Tracks delegation chains for cascade revocation and lineage queries.
pub struct DelegationTracker {
    incoming: RcuMap<RoleId, Vec<DelegationEdge>>,
    outgoing: RcuMap<RoleId, Vec<DelegationEdge>>,
}

impl DelegationTracker {
    pub fn new() -> Self {
        Self {
            incoming: RcuMap::empty_map(),
            outgoing: RcuMap::empty_map(),
        }
    }

    /// Records a grant in both the incoming (grantee) and outgoing (grantor) maps.
    pub fn record_grant(&self, edge: DelegationEdge) {
        let grantee = edge.grantee;
        let grantor = edge.grantor;
        let edge_clone = edge.clone();

        // Add to incoming[grantee]
        self.incoming.update(|m| {
            m.entry(grantee).or_insert_with(Vec::new).push(edge);
        });

        // Add to outgoing[grantor]
        self.outgoing.update(|m| {
            m.entry(grantor).or_insert_with(Vec::new).push(edge_clone);
        });
    }

    /// BFS from grantor through outgoing edges matching the given privilege and object.
    /// Returns all downstream delegation edges that should be revoked.
    pub fn cascade_revoke(
        &self,
        grantor: RoleId,
        privilege: PrivilegeType,
        object_type: ObjectType,
        object_id: u32,
    ) -> Vec<DelegationEdge> {
        let mut result = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(grantor);

        let mut visited = std::collections::HashSet::new();
        visited.insert(grantor);

        let snap = self.outgoing.load();
        while let Some(current) = queue.pop_front() {
            if let Some(edges) = snap.get(&current) {
                for e in edges {
                    if e.privilege == privilege
                        && e.object_type == object_type
                        && e.object_id == object_id
                        && !visited.contains(&e.grantee)
                    {
                        visited.insert(e.grantee);
                        result.push(e.clone());
                        queue.push_back(e.grantee);
                    }
                }
            }
        }
        result
    }

    /// Walks incoming edges from grantee to find the full grant lineage.
    pub fn chain_for_grant(
        &self,
        grantee: RoleId,
        privilege: PrivilegeType,
        object_type: ObjectType,
        object_id: u32,
    ) -> Vec<DelegationEdge> {
        let mut chain = Vec::new();
        let mut current = grantee;
        let mut visited = std::collections::HashSet::new();
        visited.insert(current);

        let snap = self.incoming.load();
        loop {
            let mut found_edge = None;
            if let Some(edges) = snap.get(&current) {
                for e in edges {
                    if e.privilege == privilege
                        && e.object_type == object_type
                        && e.object_id == object_id
                        && !visited.contains(&e.grantor)
                    {
                        found_edge = Some(e.clone());
                    }
                }
            }
            match found_edge {
                Some(edge) => {
                    let next = edge.grantor;
                    visited.insert(next);
                    chain.push(edge);
                    current = next;
                }
                None => break,
            }
        }
        chain
    }

    /// Bulk-loads delegation edges.
    pub fn load(&self, edges: Vec<DelegationEdge>) {
        for edge in edges {
            self.record_grant(edge);
        }
    }
}

// ---------------------------------------------------------------------------
// Two-Person Integrity
// ---------------------------------------------------------------------------

/// Operations that require two-person approval.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TwoPersonOperation {
    DropDatabase,
    DropTable,
    AlterRoleSuperuser,
    GrantSuperuser,
    BreakGlassActivate,
    DropAllPrivileges,
}

/// A pending two-person approval request.
#[derive(Debug, Clone)]
pub struct PendingApproval {
    pub id: u64,
    pub operation: TwoPersonOperation,
    pub requester: RoleId,
    pub requested_at: u64,
    pub details: String,
    pub approved_by: Option<RoleId>,
}

/// Defines which operations require approval, by whom, and the timeout.
#[derive(Clone)]
pub struct TwoPersonRule {
    pub operation: TwoPersonOperation,
    pub required_role: Option<RoleId>,
    pub timeout_secs: u64,
}

/// Manages two-person approval workflows.
pub struct TwoPersonManager {
    rules: Rcu<Vec<TwoPersonRule>>,
    pending: scc::HashMap<u64, PendingApproval>,
    next_id: AtomicU64,
}

impl TwoPersonManager {
    pub fn new() -> Self {
        Self {
            rules: Rcu::new(Vec::new()),
            pending: scc::HashMap::new(),
            next_id: AtomicU64::new(1),
        }
    }

    /// Adds a rule defining which operations need approval.
    pub fn add_rule(&self, rule: TwoPersonRule) {
        self.rules.update(|v| v.push(rule));
    }

    /// Returns true if any rule matches the given operation.
    pub fn requires_approval(&self, operation: TwoPersonOperation) -> bool {
        let rules = self.rules.load();
        rules.iter().any(|r| r.operation == operation)
    }

    /// Creates a pending approval request and returns its ID.
    pub fn request_approval(
        &self,
        requester: RoleId,
        operation: TwoPersonOperation,
        details: String,
    ) -> Result<u64> {
        if !self.requires_approval(operation) {
            return Err(ZyronError::PermissionDenied(
                "No two-person rule for this operation".to_string(),
            ));
        }
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let approval = PendingApproval {
            id,
            operation,
            requester,
            requested_at: now_secs(),
            details,
            approved_by: None,
        };
        let _ = self.pending.insert_sync(id, approval);
        Ok(id)
    }

    /// Approves a pending request. The approver must be a different person than the requester.
    pub fn approve(&self, approval_id: u64, approver: RoleId) -> Result<()> {
        match self.pending.entry_sync(approval_id) {
            scc::hash_map::Entry::Occupied(mut occ) => {
                if occ.get().requester == approver {
                    return Err(ZyronError::PermissionDenied(
                        "Approver must be different from requester".to_string(),
                    ));
                }
                occ.get_mut().approved_by = Some(approver);
                Ok(())
            }
            scc::hash_map::Entry::Vacant(_) => Err(ZyronError::PermissionDenied(
                "Approval request not found".to_string(),
            )),
        }
    }

    /// Denies (removes) a pending approval request.
    pub fn deny(&self, approval_id: u64, _denier: RoleId) -> Result<()> {
        let removed = self.pending.remove_sync(&approval_id);
        if removed.is_none() {
            return Err(ZyronError::PermissionDenied(
                "Approval request not found".to_string(),
            ));
        }
        Ok(())
    }

    /// Returns all currently pending approvals.
    pub fn pending_approvals(&self) -> Vec<PendingApproval> {
        let mut results = Vec::new();
        self.pending.iter_sync(|_k, v| {
            results.push(v.clone());
            true
        });
        results
    }

    /// Removes timed-out approval requests based on rule timeouts.
    pub fn cleanup_expired(&self) {
        let now = now_secs();
        let rules = self.rules.load();
        let mut to_remove = Vec::new();
        self.pending.iter_sync(|_k, v| {
            for rule in rules.iter() {
                if rule.operation == v.operation && now > v.requested_at + rule.timeout_secs {
                    to_remove.push(v.id);
                    break;
                }
            }
            true
        });
        for id in to_remove {
            let _ = self.pending.remove_sync(&id);
        }
    }
}

// ---------------------------------------------------------------------------
// GovernanceManager
// ---------------------------------------------------------------------------

/// Top-level governance coordinator owning analytics, delegation, and two-person subsystems.
pub struct GovernanceManager {
    pub analytics: PrivilegeAnalytics,
    pub delegation: DelegationTracker,
    pub two_person: TwoPersonManager,
}

impl GovernanceManager {
    pub fn new() -> Self {
        Self {
            analytics: PrivilegeAnalytics::new(),
            delegation: DelegationTracker::new(),
            two_person: TwoPersonManager::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analytics_record_and_report() {
        let analytics = PrivilegeAnalytics::new();
        analytics.record_usage(RoleId(1), PrivilegeType::Select, ObjectType::Table, 100);
        analytics.record_usage(RoleId(1), PrivilegeType::Select, ObjectType::Table, 100);
        analytics.record_usage(RoleId(1), PrivilegeType::Insert, ObjectType::Table, 200);

        let report = analytics.usage_report(RoleId(1));
        assert_eq!(report.len(), 2);

        let select_entry = report.iter().find(|r| r.privilege == PrivilegeType::Select);
        assert!(select_entry.is_some());
        let s = select_entry.expect("select entry present");
        assert!(s.use_count >= 2);
    }

    #[test]
    fn test_analytics_empty_report() {
        let analytics = PrivilegeAnalytics::new();
        let report = analytics.usage_report(RoleId(999));
        assert!(report.is_empty());
    }

    #[test]
    fn test_delegation_record_and_cascade() {
        let tracker = DelegationTracker::new();
        // A grants to B, B grants to C
        tracker.record_grant(DelegationEdge {
            grantor: RoleId(1),
            grantee: RoleId(2),
            privilege: PrivilegeType::Select,
            object_type: ObjectType::Table,
            object_id: 100,
            granted_at: 1000,
        });
        tracker.record_grant(DelegationEdge {
            grantor: RoleId(2),
            grantee: RoleId(3),
            privilege: PrivilegeType::Select,
            object_type: ObjectType::Table,
            object_id: 100,
            granted_at: 2000,
        });

        let revoked =
            tracker.cascade_revoke(RoleId(1), PrivilegeType::Select, ObjectType::Table, 100);
        assert_eq!(revoked.len(), 2);
        let grantees: Vec<RoleId> = revoked.iter().map(|e| e.grantee).collect();
        assert!(grantees.contains(&RoleId(2)));
        assert!(grantees.contains(&RoleId(3)));
    }

    #[test]
    fn test_delegation_chain_for_grant() {
        let tracker = DelegationTracker::new();
        tracker.record_grant(DelegationEdge {
            grantor: RoleId(1),
            grantee: RoleId(2),
            privilege: PrivilegeType::Select,
            object_type: ObjectType::Table,
            object_id: 50,
            granted_at: 1000,
        });
        tracker.record_grant(DelegationEdge {
            grantor: RoleId(2),
            grantee: RoleId(3),
            privilege: PrivilegeType::Select,
            object_type: ObjectType::Table,
            object_id: 50,
            granted_at: 2000,
        });

        let chain =
            tracker.chain_for_grant(RoleId(3), PrivilegeType::Select, ObjectType::Table, 50);
        assert_eq!(chain.len(), 2);
        assert_eq!(chain[0].grantor, RoleId(2));
        assert_eq!(chain[1].grantor, RoleId(1));
    }

    #[test]
    fn test_delegation_edge_to_bytes_from_bytes() {
        let edge = DelegationEdge {
            grantor: RoleId(10),
            grantee: RoleId(20),
            privilege: PrivilegeType::Select,
            object_type: ObjectType::Table,
            object_id: 42,
            granted_at: 1700000000,
        };
        let bytes = edge.to_bytes();
        let restored = DelegationEdge::from_bytes(&bytes).expect("decode should succeed");
        assert_eq!(restored.grantor, edge.grantor);
        assert_eq!(restored.grantee, edge.grantee);
        assert_eq!(restored.privilege, edge.privilege);
        assert_eq!(restored.object_type, edge.object_type);
        assert_eq!(restored.object_id, edge.object_id);
        assert_eq!(restored.granted_at, edge.granted_at);
    }

    #[test]
    fn test_two_person_approve() {
        let mgr = TwoPersonManager::new();
        mgr.add_rule(TwoPersonRule {
            operation: TwoPersonOperation::DropDatabase,
            required_role: None,
            timeout_secs: 3600,
        });

        assert!(mgr.requires_approval(TwoPersonOperation::DropDatabase));
        assert!(!mgr.requires_approval(TwoPersonOperation::DropTable));

        let id = mgr
            .request_approval(
                RoleId(1),
                TwoPersonOperation::DropDatabase,
                "decommission".to_string(),
            )
            .expect("request should succeed");

        // Same person cannot approve
        assert!(mgr.approve(id, RoleId(1)).is_err());

        // Different person can approve
        mgr.approve(id, RoleId(2)).expect("approve should succeed");

        let pending = mgr.pending_approvals();
        let entry = pending
            .iter()
            .find(|p| p.id == id)
            .expect("approval present");
        assert_eq!(entry.approved_by, Some(RoleId(2)));
    }

    #[test]
    fn test_two_person_deny() {
        let mgr = TwoPersonManager::new();
        mgr.add_rule(TwoPersonRule {
            operation: TwoPersonOperation::DropTable,
            required_role: None,
            timeout_secs: 600,
        });

        let id = mgr
            .request_approval(
                RoleId(5),
                TwoPersonOperation::DropTable,
                "cleanup".to_string(),
            )
            .expect("request should succeed");

        mgr.deny(id, RoleId(6)).expect("deny should succeed");
        assert!(mgr.pending_approvals().is_empty());
    }

    #[test]
    fn test_two_person_no_rule() {
        let mgr = TwoPersonManager::new();
        let result = mgr.request_approval(
            RoleId(1),
            TwoPersonOperation::DropDatabase,
            "test".to_string(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_two_person_deny_nonexistent() {
        let mgr = TwoPersonManager::new();
        assert!(mgr.deny(999, RoleId(1)).is_err());
    }

    #[test]
    fn test_governance_manager_new() {
        let gov = GovernanceManager::new();
        gov.analytics
            .record_usage(RoleId(1), PrivilegeType::Select, ObjectType::Table, 1);
        let report = gov.analytics.usage_report(RoleId(1));
        assert_eq!(report.len(), 1);
    }
}

//! Row-level security (RLS) policies with arbitrary SQL predicates.
//!
//! Provides per-table, per-command security policies that inject SQL predicate
//! filters into queries. Supports permissive (OR'd together) and restrictive
//! (AND'd together) policy types following PostgreSQL semantics.
//! Distinct from row_ownership.rs which only supports owner-column filtering.

use crate::rcu::RcuMap;
use crate::role::RoleId;
use serde::{Deserialize, Serialize};
use zyron_common::{Result, ZyronError};

/// SQL command type that an RLS policy applies to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum RlsCommand {
    Select = 0,
    Insert = 1,
    Update = 2,
    Delete = 3,
    All = 255,
}

impl RlsCommand {
    /// Returns true if this command matches the given command.
    /// All matches everything, and a specific command matches itself.
    pub fn matches(&self, other: RlsCommand) -> bool {
        matches!(self, RlsCommand::All)
            || matches!(other, RlsCommand::All)
            || *self as u8 == other as u8
    }

    fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(RlsCommand::Select),
            1 => Ok(RlsCommand::Insert),
            2 => Ok(RlsCommand::Update),
            3 => Ok(RlsCommand::Delete),
            255 => Ok(RlsCommand::All),
            _ => Err(ZyronError::DecodingFailed(format!(
                "Unknown RlsCommand value: {}",
                v
            ))),
        }
    }
}

/// Whether a policy is permissive (OR'd) or restrictive (AND'd).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum PolicyType {
    Permissive = 0,
    Restrictive = 1,
}

/// An RLS policy binding a SQL predicate to a table for specific commands and roles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlsPolicy {
    pub id: u32,
    pub name: String,
    pub table_id: u32,
    pub command: RlsCommand,
    pub policy_type: PolicyType,
    /// Roles this policy applies to. Empty means all roles.
    pub roles: Vec<RoleId>,
    /// Filter predicate for SELECT/UPDATE/DELETE. Rows not matching are hidden.
    pub using_expr: Option<String>,
    /// Write validation predicate for INSERT/UPDATE. Rows not matching are rejected.
    pub check_expr: Option<String>,
    pub enabled: bool,
}

impl RlsPolicy {
    /// Serializes the policy to bytes.
    /// Layout: id(4) + table_id(4) + command(1) + policy_type(1) + enabled(1)
    ///       + roles_count(4) + roles(N*4)
    ///       + has_using(1) + [using_len(4) + using(N)]
    ///       + has_check(1) + [check_len(4) + check(N)]
    ///       + name_len(4) + name(N)
    pub fn to_bytes(&self) -> Vec<u8> {
        let name_bytes = self.name.as_bytes();
        let mut buf = Vec::with_capacity(64 + name_bytes.len());

        buf.extend_from_slice(&self.id.to_le_bytes());
        buf.extend_from_slice(&self.table_id.to_le_bytes());
        buf.push(self.command as u8);
        buf.push(self.policy_type as u8);
        buf.push(if self.enabled { 1 } else { 0 });

        buf.extend_from_slice(&(self.roles.len() as u32).to_le_bytes());
        for role in &self.roles {
            buf.extend_from_slice(&role.0.to_le_bytes());
        }

        match &self.using_expr {
            Some(expr) => {
                buf.push(1);
                let expr_bytes = expr.as_bytes();
                buf.extend_from_slice(&(expr_bytes.len() as u32).to_le_bytes());
                buf.extend_from_slice(expr_bytes);
            }
            None => buf.push(0),
        }

        match &self.check_expr {
            Some(expr) => {
                buf.push(1);
                let expr_bytes = expr.as_bytes();
                buf.extend_from_slice(&(expr_bytes.len() as u32).to_le_bytes());
                buf.extend_from_slice(expr_bytes);
            }
            None => buf.push(0),
        }

        buf.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(name_bytes);

        buf
    }

    /// Deserializes a policy from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < 15 {
            return Err(ZyronError::DecodingFailed(
                "RlsPolicy data too short".to_string(),
            ));
        }
        let mut pos = 0;

        let id = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
        pos += 4;
        let table_id = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
        pos += 4;
        let command = RlsCommand::from_u8(data[pos])?;
        pos += 1;
        let policy_type = if data[pos] == 0 {
            PolicyType::Permissive
        } else {
            PolicyType::Restrictive
        };
        pos += 1;
        let enabled = data[pos] != 0;
        pos += 1;

        if data.len() < pos + 4 {
            return Err(ZyronError::DecodingFailed(
                "RlsPolicy roles count truncated".to_string(),
            ));
        }
        let roles_count =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        let mut roles = Vec::with_capacity(roles_count);
        for _ in 0..roles_count {
            if data.len() < pos + 4 {
                return Err(ZyronError::DecodingFailed(
                    "RlsPolicy role data truncated".to_string(),
                ));
            }
            let r = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
            pos += 4;
            roles.push(RoleId(r));
        }

        // using_expr
        if data.len() < pos + 1 {
            return Err(ZyronError::DecodingFailed(
                "RlsPolicy using_expr flag missing".to_string(),
            ));
        }
        let using_expr = if data[pos] == 1 {
            pos += 1;
            if data.len() < pos + 4 {
                return Err(ZyronError::DecodingFailed(
                    "RlsPolicy using_expr length truncated".to_string(),
                ));
            }
            let len = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
                as usize;
            pos += 4;
            if data.len() < pos + len {
                return Err(ZyronError::DecodingFailed(
                    "RlsPolicy using_expr data truncated".to_string(),
                ));
            }
            let s = std::str::from_utf8(&data[pos..pos + len])
                .map_err(|_| ZyronError::DecodingFailed("Invalid UTF-8 in using_expr".to_string()))?
                .to_string();
            pos += len;
            Some(s)
        } else {
            pos += 1;
            None
        };

        // check_expr
        if data.len() < pos + 1 {
            return Err(ZyronError::DecodingFailed(
                "RlsPolicy check_expr flag missing".to_string(),
            ));
        }
        let check_expr = if data[pos] == 1 {
            pos += 1;
            if data.len() < pos + 4 {
                return Err(ZyronError::DecodingFailed(
                    "RlsPolicy check_expr length truncated".to_string(),
                ));
            }
            let len = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
                as usize;
            pos += 4;
            if data.len() < pos + len {
                return Err(ZyronError::DecodingFailed(
                    "RlsPolicy check_expr data truncated".to_string(),
                ));
            }
            let s = std::str::from_utf8(&data[pos..pos + len])
                .map_err(|_| ZyronError::DecodingFailed("Invalid UTF-8 in check_expr".to_string()))?
                .to_string();
            pos += len;
            Some(s)
        } else {
            pos += 1;
            None
        };

        // name
        if data.len() < pos + 4 {
            return Err(ZyronError::DecodingFailed(
                "RlsPolicy name length truncated".to_string(),
            ));
        }
        let name_len =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        if data.len() < pos + name_len {
            return Err(ZyronError::DecodingFailed(
                "RlsPolicy name data truncated".to_string(),
            ));
        }
        let name = std::str::from_utf8(&data[pos..pos + name_len])
            .map_err(|_| ZyronError::DecodingFailed("Invalid UTF-8 in policy name".to_string()))?
            .to_string();

        Ok(Self {
            id,
            name,
            table_id,
            command,
            policy_type,
            roles,
            using_expr,
            check_expr,
            enabled,
        })
    }
}

/// Holds the evaluated RLS predicates for a query.
pub struct RlsResult {
    /// Filter predicates to inject into WHERE clause.
    pub using_predicates: Vec<String>,
    /// Write validation predicates for INSERT/UPDATE.
    pub check_predicates: Vec<String>,
}

/// Stores RLS policies indexed by table_id.
pub struct RlsPolicyStore {
    policies: RcuMap<u32, Vec<RlsPolicy>>,
}

impl RlsPolicyStore {
    pub fn new() -> Self {
        Self {
            policies: RcuMap::empty_map(),
        }
    }

    /// Adds a policy. Rejects duplicate names within the same table.
    pub fn add_policy(&self, policy: RlsPolicy) -> Result<()> {
        let table_id = policy.table_id;
        let name = policy.name.clone();
        // Check for duplicates before mutating.
        let snap = self.policies.load();
        if let Some(v) = snap.get(&table_id) {
            if v.iter().any(|p| p.name == name) {
                return Err(ZyronError::PolicyAlreadyExists(name));
            }
        }
        self.policies.update(|m| {
            m.entry(table_id).or_insert_with(Vec::new).push(policy);
        });
        Ok(())
    }

    /// Removes a policy by table_id and name. Returns true if removed.
    pub fn remove_policy(&self, table_id: u32, name: &str) -> bool {
        let mut removed = false;
        self.policies.update(|m| {
            if let Some(v) = m.get_mut(&table_id) {
                let before = v.len();
                v.retain(|p| p.name != name);
                removed = v.len() < before;
            }
        });
        removed
    }

    /// Updates a policy by name within a table. Returns an error if not found.
    pub fn alter_policy(
        &self,
        table_id: u32,
        name: &str,
        new_using: Option<Option<String>>,
        new_check: Option<Option<String>>,
        new_roles: Option<Vec<RoleId>>,
    ) -> Result<()> {
        let snap = self.policies.load();
        if snap.get(&table_id).is_none() {
            return Err(ZyronError::PolicyNotFound(format!(
                "{} on table {}",
                name, table_id
            )));
        }
        let mut found = false;
        self.policies.update(|m| {
            if let Some(v) = m.get_mut(&table_id) {
                if let Some(policy) = v.iter_mut().find(|p| p.name == name) {
                    if let Some(u) = &new_using {
                        policy.using_expr = u.clone();
                    }
                    if let Some(c) = &new_check {
                        policy.check_expr = c.clone();
                    }
                    if let Some(r) = &new_roles {
                        policy.roles = r.clone();
                    }
                    found = true;
                }
            }
        });
        if !found {
            return Err(ZyronError::PolicyNotFound(format!(
                "{} on table {}",
                name, table_id
            )));
        }
        Ok(())
    }

    /// Returns all policies for a table.
    pub fn policies_for_table(&self, table_id: u32) -> Vec<RlsPolicy> {
        let snap = self.policies.load();
        snap.get(&table_id).cloned().unwrap_or_default()
    }

    /// Evaluates RLS policies for a given table, command, and set of role IDs.
    /// Table owners bypass RLS. Returns combined predicates for query injection.
    ///
    /// Permissive policies are OR'd: if any permissive policy matches, the row is visible.
    /// Restrictive policies are AND'd: all restrictive policies must pass.
    /// Final predicate = (permissive1 OR permissive2 ...) AND restrictive1 AND restrictive2 ...
    /// If no permissive policies exist for the command, no rows are visible (empty USING = deny all).
    pub fn evaluate_rls(
        &self,
        table_id: u32,
        command: RlsCommand,
        role_ids: &[RoleId],
        is_table_owner: bool,
    ) -> RlsResult {
        if is_table_owner {
            return RlsResult {
                using_predicates: Vec::new(),
                check_predicates: Vec::new(),
            };
        }

        let policies = self.policies_for_table(table_id);
        if policies.is_empty() {
            return RlsResult {
                using_predicates: Vec::new(),
                check_predicates: Vec::new(),
            };
        }

        let mut permissive_using: Vec<String> = Vec::new();
        let mut restrictive_using: Vec<String> = Vec::new();
        let mut check_predicates: Vec<String> = Vec::new();

        for policy in &policies {
            if !policy.enabled {
                continue;
            }
            if !policy.command.matches(command) {
                continue;
            }
            // Check if the policy applies to any of the user's roles.
            // Empty roles list means the policy applies to all roles.
            if !policy.roles.is_empty() && !policy.roles.iter().any(|pr| role_ids.contains(pr)) {
                continue;
            }

            if let Some(using) = &policy.using_expr {
                match policy.policy_type {
                    PolicyType::Permissive => permissive_using.push(using.clone()),
                    PolicyType::Restrictive => restrictive_using.push(using.clone()),
                }
            }

            if let Some(check) = &policy.check_expr {
                check_predicates.push(check.clone());
            }
        }

        // Combine predicates: (p1 OR p2 ...) AND r1 AND r2 ...
        let mut using_predicates = Vec::new();

        if !permissive_using.is_empty() {
            if permissive_using.len() == 1 {
                using_predicates.push(permissive_using.into_iter().next().unwrap_or_default());
            } else {
                let combined = permissive_using
                    .iter()
                    .map(|p| format!("({})", p))
                    .collect::<Vec<_>>()
                    .join(" OR ");
                using_predicates.push(format!("({})", combined));
            }
        }

        using_predicates.extend(restrictive_using);

        RlsResult {
            using_predicates,
            check_predicates,
        }
    }

    /// Bulk-loads policies from storage.
    pub fn load(&self, policies: Vec<RlsPolicy>) {
        for policy in policies {
            let _ = self.add_policy(policy);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_policy(id: u32, name: &str, table_id: u32) -> RlsPolicy {
        RlsPolicy {
            id,
            name: name.to_string(),
            table_id,
            command: RlsCommand::All,
            policy_type: PolicyType::Permissive,
            roles: Vec::new(),
            using_expr: Some("user_id = current_user_id()".to_string()),
            check_expr: None,
            enabled: true,
        }
    }

    // -- Serialization tests --

    #[test]
    fn test_rls_policy_roundtrip() {
        let policy = RlsPolicy {
            id: 1,
            name: "user_orders".to_string(),
            table_id: 42,
            command: RlsCommand::Select,
            policy_type: PolicyType::Permissive,
            roles: vec![RoleId(10), RoleId(20)],
            using_expr: Some("user_id = current_user_id()".to_string()),
            check_expr: Some("status != 'deleted'".to_string()),
            enabled: true,
        };
        let bytes = policy.to_bytes();
        let restored = RlsPolicy::from_bytes(&bytes).expect("decode");
        assert_eq!(restored.id, 1);
        assert_eq!(restored.name, "user_orders");
        assert_eq!(restored.table_id, 42);
        assert_eq!(restored.command, RlsCommand::Select);
        assert_eq!(restored.policy_type, PolicyType::Permissive);
        assert_eq!(restored.roles.len(), 2);
        assert_eq!(restored.roles[0], RoleId(10));
        assert_eq!(
            restored.using_expr.as_deref(),
            Some("user_id = current_user_id()")
        );
        assert_eq!(restored.check_expr.as_deref(), Some("status != 'deleted'"));
        assert!(restored.enabled);
    }

    #[test]
    fn test_rls_policy_roundtrip_no_exprs() {
        let policy = RlsPolicy {
            id: 2,
            name: "admin_all".to_string(),
            table_id: 42,
            command: RlsCommand::All,
            policy_type: PolicyType::Restrictive,
            roles: Vec::new(),
            using_expr: None,
            check_expr: None,
            enabled: false,
        };
        let bytes = policy.to_bytes();
        let restored = RlsPolicy::from_bytes(&bytes).expect("decode");
        assert_eq!(restored.id, 2);
        assert_eq!(restored.name, "admin_all");
        assert_eq!(restored.command, RlsCommand::All);
        assert_eq!(restored.policy_type, PolicyType::Restrictive);
        assert!(restored.roles.is_empty());
        assert!(restored.using_expr.is_none());
        assert!(restored.check_expr.is_none());
        assert!(!restored.enabled);
    }

    #[test]
    fn test_rls_policy_from_bytes_too_short() {
        assert!(RlsPolicy::from_bytes(&[0u8; 5]).is_err());
    }

    // -- RlsCommand tests --

    #[test]
    fn test_rls_command_matches() {
        assert!(RlsCommand::All.matches(RlsCommand::Select));
        assert!(RlsCommand::Select.matches(RlsCommand::All));
        assert!(RlsCommand::Select.matches(RlsCommand::Select));
        assert!(!RlsCommand::Select.matches(RlsCommand::Insert));
    }

    #[test]
    fn test_rls_command_from_u8() {
        assert_eq!(RlsCommand::from_u8(0).unwrap(), RlsCommand::Select);
        assert_eq!(RlsCommand::from_u8(255).unwrap(), RlsCommand::All);
        assert!(RlsCommand::from_u8(100).is_err());
    }

    // -- Store tests --

    #[test]
    fn test_store_add_and_get() {
        let store = RlsPolicyStore::new();
        store.add_policy(make_policy(1, "p1", 100)).expect("add");
        let policies = store.policies_for_table(100);
        assert_eq!(policies.len(), 1);
        assert_eq!(policies[0].name, "p1");
    }

    #[test]
    fn test_store_reject_duplicate_name() {
        let store = RlsPolicyStore::new();
        store.add_policy(make_policy(1, "p1", 100)).expect("add");
        assert!(store.add_policy(make_policy(2, "p1", 100)).is_err());
    }

    #[test]
    fn test_store_remove() {
        let store = RlsPolicyStore::new();
        store.add_policy(make_policy(1, "p1", 100)).expect("add");
        store.add_policy(make_policy(2, "p2", 100)).expect("add");
        assert!(store.remove_policy(100, "p1"));
        assert_eq!(store.policies_for_table(100).len(), 1);
    }

    #[test]
    fn test_store_remove_nonexistent() {
        let store = RlsPolicyStore::new();
        assert!(!store.remove_policy(100, "missing"));
    }

    #[test]
    fn test_store_alter() {
        let store = RlsPolicyStore::new();
        store.add_policy(make_policy(1, "p1", 100)).expect("add");
        store
            .alter_policy(
                100,
                "p1",
                Some(Some("region = 'us'".to_string())),
                None,
                None,
            )
            .expect("alter");
        let policies = store.policies_for_table(100);
        assert_eq!(policies[0].using_expr.as_deref(), Some("region = 'us'"));
    }

    #[test]
    fn test_store_alter_not_found() {
        let store = RlsPolicyStore::new();
        assert!(
            store
                .alter_policy(100, "missing", None, None, None)
                .is_err()
        );
    }

    #[test]
    fn test_store_empty_table() {
        let store = RlsPolicyStore::new();
        assert!(store.policies_for_table(999).is_empty());
    }

    #[test]
    fn test_store_load() {
        let store = RlsPolicyStore::new();
        let policies = vec![
            make_policy(1, "a", 100),
            make_policy(2, "b", 100),
            make_policy(3, "c", 200),
        ];
        store.load(policies);
        assert_eq!(store.policies_for_table(100).len(), 2);
        assert_eq!(store.policies_for_table(200).len(), 1);
    }

    // -- Evaluation tests --

    #[test]
    fn test_evaluate_owner_bypass() {
        let store = RlsPolicyStore::new();
        store.add_policy(make_policy(1, "p1", 100)).expect("add");
        let result = store.evaluate_rls(100, RlsCommand::Select, &[RoleId(1)], true);
        assert!(result.using_predicates.is_empty());
    }

    #[test]
    fn test_evaluate_no_policies() {
        let store = RlsPolicyStore::new();
        let result = store.evaluate_rls(100, RlsCommand::Select, &[RoleId(1)], false);
        assert!(result.using_predicates.is_empty());
    }

    #[test]
    fn test_evaluate_single_permissive() {
        let store = RlsPolicyStore::new();
        store.add_policy(make_policy(1, "p1", 100)).expect("add");
        let result = store.evaluate_rls(100, RlsCommand::Select, &[RoleId(1)], false);
        assert_eq!(result.using_predicates.len(), 1);
        assert_eq!(result.using_predicates[0], "user_id = current_user_id()");
    }

    #[test]
    fn test_evaluate_multiple_permissive_or() {
        let store = RlsPolicyStore::new();
        store
            .add_policy(RlsPolicy {
                id: 1,
                name: "p1".to_string(),
                table_id: 100,
                command: RlsCommand::All,
                policy_type: PolicyType::Permissive,
                roles: Vec::new(),
                using_expr: Some("user_id = 1".to_string()),
                check_expr: None,
                enabled: true,
            })
            .expect("add");
        store
            .add_policy(RlsPolicy {
                id: 2,
                name: "p2".to_string(),
                table_id: 100,
                command: RlsCommand::All,
                policy_type: PolicyType::Permissive,
                roles: Vec::new(),
                using_expr: Some("is_public = true".to_string()),
                check_expr: None,
                enabled: true,
            })
            .expect("add");
        let result = store.evaluate_rls(100, RlsCommand::Select, &[RoleId(1)], false);
        assert_eq!(result.using_predicates.len(), 1);
        assert!(result.using_predicates[0].contains("OR"));
    }

    #[test]
    fn test_evaluate_restrictive_and() {
        let store = RlsPolicyStore::new();
        store
            .add_policy(RlsPolicy {
                id: 1,
                name: "perm".to_string(),
                table_id: 100,
                command: RlsCommand::All,
                policy_type: PolicyType::Permissive,
                roles: Vec::new(),
                using_expr: Some("true".to_string()),
                check_expr: None,
                enabled: true,
            })
            .expect("add");
        store
            .add_policy(RlsPolicy {
                id: 2,
                name: "restrict".to_string(),
                table_id: 100,
                command: RlsCommand::All,
                policy_type: PolicyType::Restrictive,
                roles: Vec::new(),
                using_expr: Some("region = 'us'".to_string()),
                check_expr: None,
                enabled: true,
            })
            .expect("add");
        let result = store.evaluate_rls(100, RlsCommand::Select, &[RoleId(1)], false);
        // One permissive + one restrictive = 2 predicates
        assert_eq!(result.using_predicates.len(), 2);
    }

    #[test]
    fn test_evaluate_disabled_policy_skipped() {
        let store = RlsPolicyStore::new();
        store
            .add_policy(RlsPolicy {
                id: 1,
                name: "disabled".to_string(),
                table_id: 100,
                command: RlsCommand::All,
                policy_type: PolicyType::Permissive,
                roles: Vec::new(),
                using_expr: Some("false".to_string()),
                check_expr: None,
                enabled: false,
            })
            .expect("add");
        let result = store.evaluate_rls(100, RlsCommand::Select, &[RoleId(1)], false);
        assert!(result.using_predicates.is_empty());
    }

    #[test]
    fn test_evaluate_command_filtering() {
        let store = RlsPolicyStore::new();
        store
            .add_policy(RlsPolicy {
                id: 1,
                name: "select_only".to_string(),
                table_id: 100,
                command: RlsCommand::Select,
                policy_type: PolicyType::Permissive,
                roles: Vec::new(),
                using_expr: Some("visible = true".to_string()),
                check_expr: None,
                enabled: true,
            })
            .expect("add");
        // Should apply for Select
        let result = store.evaluate_rls(100, RlsCommand::Select, &[RoleId(1)], false);
        assert_eq!(result.using_predicates.len(), 1);
        // Should not apply for Delete
        let result = store.evaluate_rls(100, RlsCommand::Delete, &[RoleId(1)], false);
        assert!(result.using_predicates.is_empty());
    }

    #[test]
    fn test_evaluate_role_filtering() {
        let store = RlsPolicyStore::new();
        store
            .add_policy(RlsPolicy {
                id: 1,
                name: "admin_only".to_string(),
                table_id: 100,
                command: RlsCommand::All,
                policy_type: PolicyType::Permissive,
                roles: vec![RoleId(99)],
                using_expr: Some("true".to_string()),
                check_expr: None,
                enabled: true,
            })
            .expect("add");
        // Non-matching role
        let result = store.evaluate_rls(100, RlsCommand::Select, &[RoleId(1)], false);
        assert!(result.using_predicates.is_empty());
        // Matching role
        let result = store.evaluate_rls(100, RlsCommand::Select, &[RoleId(99)], false);
        assert_eq!(result.using_predicates.len(), 1);
    }

    #[test]
    fn test_evaluate_check_expr() {
        let store = RlsPolicyStore::new();
        store
            .add_policy(RlsPolicy {
                id: 1,
                name: "insert_check".to_string(),
                table_id: 100,
                command: RlsCommand::Insert,
                policy_type: PolicyType::Permissive,
                roles: Vec::new(),
                using_expr: None,
                check_expr: Some("user_id = current_user_id()".to_string()),
                enabled: true,
            })
            .expect("add");
        let result = store.evaluate_rls(100, RlsCommand::Insert, &[RoleId(1)], false);
        assert_eq!(result.check_predicates.len(), 1);
    }

    #[test]
    fn test_evaluate_mixed_permissive_restrictive() {
        let store = RlsPolicyStore::new();
        // Two permissive (OR'd)
        store
            .add_policy(RlsPolicy {
                id: 1,
                name: "own".to_string(),
                table_id: 100,
                command: RlsCommand::All,
                policy_type: PolicyType::Permissive,
                roles: Vec::new(),
                using_expr: Some("owner = current_user_id()".to_string()),
                check_expr: None,
                enabled: true,
            })
            .expect("add");
        store
            .add_policy(RlsPolicy {
                id: 2,
                name: "public".to_string(),
                table_id: 100,
                command: RlsCommand::All,
                policy_type: PolicyType::Permissive,
                roles: Vec::new(),
                using_expr: Some("is_public = true".to_string()),
                check_expr: None,
                enabled: true,
            })
            .expect("add");
        // Two restrictive (AND'd)
        store
            .add_policy(RlsPolicy {
                id: 3,
                name: "region".to_string(),
                table_id: 100,
                command: RlsCommand::All,
                policy_type: PolicyType::Restrictive,
                roles: Vec::new(),
                using_expr: Some("region = 'us'".to_string()),
                check_expr: None,
                enabled: true,
            })
            .expect("add");
        store
            .add_policy(RlsPolicy {
                id: 4,
                name: "active".to_string(),
                table_id: 100,
                command: RlsCommand::All,
                policy_type: PolicyType::Restrictive,
                roles: Vec::new(),
                using_expr: Some("active = true".to_string()),
                check_expr: None,
                enabled: true,
            })
            .expect("add");

        let result = store.evaluate_rls(100, RlsCommand::Select, &[RoleId(1)], false);
        // 1 combined permissive OR + 2 restrictive = 3
        assert_eq!(result.using_predicates.len(), 3);
        assert!(result.using_predicates[0].contains("OR"));
    }
}

//! Attribute-based access control (ABAC).
//!
//! Provides row-level security policies that reference session attributes
//! (department, region, clearance, IP, custom key-value pairs) to generate
//! WHERE-clause predicates for table access.

use crate::classification::ClassificationLevel;
use crate::role::RoleId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use zyron_common::{Result, ZyronError};

/// Attributes attached to the current session, used for ABAC policy evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionAttributes {
    pub role_id: RoleId,
    pub department: Option<String>,
    pub region: Option<String>,
    pub clearance: ClassificationLevel,
    pub ip_address: String,
    pub connection_time: u64,
    pub custom: HashMap<String, String>,
}

impl SessionAttributes {
    /// Looks up an attribute by key. Checks known fields first, then custom map.
    pub fn get(&self, key: &str) -> Option<&str> {
        match key {
            "department" => self.department.as_deref(),
            "region" => self.region.as_deref(),
            "ip_address" => Some(self.ip_address.as_str()),
            _ => self.custom.get(key).map(|s| s.as_str()),
        }
    }

    /// Sets a value in the custom attributes map.
    pub fn set(&mut self, key: String, value: String) {
        self.custom.insert(key, value);
    }
}

/// An ABAC policy attached to a table. The predicate string is appended as a
/// WHERE clause filter when the policy is active.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbacPolicy {
    pub id: u32,
    pub name: String,
    pub table_id: u32,
    /// SQL predicate expression injected into queries.
    pub predicate: String,
    pub enabled: bool,
    /// If true, rows matching the predicate are visible (permissive).
    /// If false, rows matching the predicate are hidden (restrictive).
    pub permissive: bool,
    /// Roles this policy applies to. Empty means all roles.
    pub roles: Vec<RoleId>,
}

impl AbacPolicy {
    /// Serializes the policy to bytes.
    /// Layout: id(4) + table_id(4) + enabled(1) + permissive(1) + roles_count(4) + roles(N*4)
    ///       + name_len(4) + name(N) + predicate_len(4) + predicate(N)
    pub fn to_bytes(&self) -> Vec<u8> {
        let name_bytes = self.name.as_bytes();
        let pred_bytes = self.predicate.as_bytes();
        let total = 4
            + 4
            + 1
            + 1
            + 4
            + (self.roles.len() * 4)
            + 4
            + name_bytes.len()
            + 4
            + pred_bytes.len();
        let mut buf = Vec::with_capacity(total);

        buf.extend_from_slice(&self.id.to_le_bytes());
        buf.extend_from_slice(&self.table_id.to_le_bytes());
        buf.push(if self.enabled { 1 } else { 0 });
        buf.push(if self.permissive { 1 } else { 0 });

        buf.extend_from_slice(&(self.roles.len() as u32).to_le_bytes());
        for role in &self.roles {
            buf.extend_from_slice(&role.0.to_le_bytes());
        }

        buf.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(name_bytes);

        buf.extend_from_slice(&(pred_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(pred_bytes);

        buf
    }

    /// Deserializes a policy from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < 18 {
            return Err(ZyronError::DecodingFailed(
                "AbacPolicy data too short".to_string(),
            ));
        }
        let mut pos = 0;

        let id = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
        pos += 4;
        let table_id = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
        pos += 4;
        let enabled = data[pos] != 0;
        pos += 1;
        let permissive = data[pos] != 0;
        pos += 1;

        let roles_count =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        if data.len() < pos + roles_count * 4 + 8 {
            return Err(ZyronError::DecodingFailed(
                "AbacPolicy roles data truncated".to_string(),
            ));
        }
        let mut roles = Vec::with_capacity(roles_count);
        for _ in 0..roles_count {
            let r = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
            pos += 4;
            roles.push(RoleId(r));
        }

        if data.len() < pos + 4 {
            return Err(ZyronError::DecodingFailed(
                "AbacPolicy name length missing".to_string(),
            ));
        }
        let name_len =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        if data.len() < pos + name_len {
            return Err(ZyronError::DecodingFailed(
                "AbacPolicy name truncated".to_string(),
            ));
        }
        let name = std::str::from_utf8(&data[pos..pos + name_len])
            .map_err(|_| ZyronError::DecodingFailed("Invalid UTF-8 in policy name".to_string()))?
            .to_string();
        pos += name_len;

        if data.len() < pos + 4 {
            return Err(ZyronError::DecodingFailed(
                "AbacPolicy predicate length missing".to_string(),
            ));
        }
        let pred_len =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        if data.len() < pos + pred_len {
            return Err(ZyronError::DecodingFailed(
                "AbacPolicy predicate truncated".to_string(),
            ));
        }
        let predicate = std::str::from_utf8(&data[pos..pos + pred_len])
            .map_err(|_| {
                ZyronError::DecodingFailed("Invalid UTF-8 in policy predicate".to_string())
            })?
            .to_string();

        Ok(Self {
            id,
            name,
            table_id,
            predicate,
            enabled,
            permissive,
            roles,
        })
    }
}

/// Stores ABAC policies indexed by table_id.
pub struct AbacStore {
    policies: scc::HashMap<u32, Vec<AbacPolicy>>,
}

impl AbacStore {
    pub fn new() -> Self {
        Self {
            policies: scc::HashMap::new(),
        }
    }

    /// Returns all policies for the given table.
    pub fn policies_for_table(&self, table_id: u32) -> Vec<AbacPolicy> {
        self.policies
            .read_sync(&table_id, |_k, v| v.clone())
            .unwrap_or_default()
    }

    /// Adds a policy to the store, grouped by table_id.
    pub fn add_policy(&self, policy: AbacPolicy) -> Result<()> {
        let table_id = policy.table_id;
        match self.policies.entry_sync(table_id) {
            scc::hash_map::Entry::Occupied(mut occ) => {
                let v = occ.get_mut();
                if !v.iter().any(|p| p.name == policy.name) {
                    v.push(policy);
                }
            }
            scc::hash_map::Entry::Vacant(vac) => {
                vac.insert_entry(vec![policy]);
            }
        }
        Ok(())
    }

    /// Removes a policy by table_id and name. Returns true if removed.
    pub fn remove_policy(&self, table_id: u32, name: &str) -> bool {
        let mut removed = false;
        if let scc::hash_map::Entry::Occupied(mut occ) = self.policies.entry_sync(table_id) {
            let v = occ.get_mut();
            let before = v.len();
            v.retain(|p| p.name != name);
            removed = v.len() < before;
        }
        removed
    }

    /// Bulk-loads policies, grouping by table_id.
    pub fn load(&self, policies: Vec<AbacPolicy>) {
        for policy in policies {
            let _ = self.add_policy(policy);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_policy(id: u32, name: &str, table_id: u32) -> AbacPolicy {
        AbacPolicy {
            id,
            name: name.to_string(),
            table_id,
            predicate: "department = current_department()".to_string(),
            enabled: true,
            permissive: true,
            roles: vec![RoleId(1), RoleId(2)],
        }
    }

    #[test]
    fn test_abac_store_add_and_get() {
        let store = AbacStore::new();
        let policy = make_policy(1, "dept_filter", 100);
        store.add_policy(policy).expect("add should succeed");

        let policies = store.policies_for_table(100);
        assert_eq!(policies.len(), 1);
        assert_eq!(policies[0].name, "dept_filter");
    }

    #[test]
    fn test_abac_store_empty_table() {
        let store = AbacStore::new();
        let policies = store.policies_for_table(999);
        assert!(policies.is_empty());
    }

    #[test]
    fn test_abac_store_remove_policy() {
        let store = AbacStore::new();
        store
            .add_policy(make_policy(1, "p1", 100))
            .expect("add should succeed");
        store
            .add_policy(make_policy(2, "p2", 100))
            .expect("add should succeed");

        assert!(store.remove_policy(100, "p1"));
        let policies = store.policies_for_table(100);
        assert_eq!(policies.len(), 1);
        assert_eq!(policies[0].name, "p2");
    }

    #[test]
    fn test_abac_store_remove_nonexistent() {
        let store = AbacStore::new();
        assert!(!store.remove_policy(100, "missing"));
    }

    #[test]
    fn test_abac_store_load() {
        let store = AbacStore::new();
        let policies = vec![
            make_policy(1, "a", 100),
            make_policy(2, "b", 100),
            make_policy(3, "c", 200),
        ];
        store.load(policies);
        assert_eq!(store.policies_for_table(100).len(), 2);
        assert_eq!(store.policies_for_table(200).len(), 1);
    }

    #[test]
    fn test_abac_policy_to_bytes_from_bytes() {
        let policy = make_policy(42, "region_policy", 300);
        let bytes = policy.to_bytes();
        let restored = AbacPolicy::from_bytes(&bytes).expect("decode should succeed");
        assert_eq!(restored.id, 42);
        assert_eq!(restored.name, "region_policy");
        assert_eq!(restored.table_id, 300);
        assert_eq!(restored.predicate, policy.predicate);
        assert!(restored.enabled);
        assert!(restored.permissive);
        assert_eq!(restored.roles.len(), 2);
        assert_eq!(restored.roles[0], RoleId(1));
    }

    #[test]
    fn test_abac_policy_from_bytes_too_short() {
        assert!(AbacPolicy::from_bytes(&[0u8; 5]).is_err());
    }

    #[test]
    fn test_session_attributes_get_known_fields() {
        let attrs = SessionAttributes {
            role_id: RoleId(1),
            department: Some("engineering".to_string()),
            region: Some("us-east".to_string()),
            clearance: ClassificationLevel::Internal,
            ip_address: "192.168.1.1".to_string(),
            connection_time: 1000,
            custom: HashMap::new(),
        };
        assert_eq!(attrs.get("department"), Some("engineering"));
        assert_eq!(attrs.get("region"), Some("us-east"));
        assert_eq!(attrs.get("ip_address"), Some("192.168.1.1"));
        assert_eq!(attrs.get("nonexistent"), None);
    }

    #[test]
    fn test_session_attributes_get_custom() {
        let mut attrs = SessionAttributes {
            role_id: RoleId(1),
            department: None,
            region: None,
            clearance: ClassificationLevel::Public,
            ip_address: "10.0.0.1".to_string(),
            connection_time: 0,
            custom: HashMap::new(),
        };
        attrs.set("team".to_string(), "backend".to_string());
        assert_eq!(attrs.get("team"), Some("backend"));
    }

    #[test]
    fn test_session_attributes_department_none() {
        let attrs = SessionAttributes {
            role_id: RoleId(1),
            department: None,
            region: None,
            clearance: ClassificationLevel::Public,
            ip_address: "10.0.0.1".to_string(),
            connection_time: 0,
            custom: HashMap::new(),
        };
        assert_eq!(attrs.get("department"), None);
    }
}

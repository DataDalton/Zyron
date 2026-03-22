//! Column-level masking policies with role exemptions.
//!
//! Wraps MaskFunction (from masking.rs) with a named policy that specifies
//! which column to mask and which roles are exempt from masking.
//! Distinct from MaskingRule which is a lower-level binding without exemptions.

use crate::masking::MaskFunction;
use crate::rcu::RcuMap;
use crate::role::RoleId;
use serde::{Deserialize, Serialize};
use zyron_common::{Result, ZyronError};

/// A named masking policy binding a MaskFunction to a column with exempt roles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaskingPolicy {
    pub id: u32,
    pub name: String,
    pub table_id: u32,
    pub column_id: u16,
    pub function: MaskFunction,
    /// Roles that see the original unmasked value.
    pub exempt_roles: Vec<RoleId>,
    pub enabled: bool,
}

impl MaskingPolicy {
    /// Serializes the policy to bytes.
    /// Layout: id(4) + table_id(4) + column_id(2) + enabled(1)
    ///       + exempt_count(4) + exempt_roles(N*4)
    ///       + function_bytes
    ///       + name_len(4) + name(N)
    pub fn to_bytes(&self) -> Vec<u8> {
        let name_bytes = self.name.as_bytes();
        let func_bytes = self.function.to_bytes();
        let mut buf = Vec::with_capacity(
            4 + 4
                + 2
                + 1
                + 4
                + self.exempt_roles.len() * 4
                + func_bytes.len()
                + 4
                + name_bytes.len(),
        );

        buf.extend_from_slice(&self.id.to_le_bytes());
        buf.extend_from_slice(&self.table_id.to_le_bytes());
        buf.extend_from_slice(&self.column_id.to_le_bytes());
        buf.push(if self.enabled { 1 } else { 0 });

        buf.extend_from_slice(&(self.exempt_roles.len() as u32).to_le_bytes());
        for role in &self.exempt_roles {
            buf.extend_from_slice(&role.0.to_le_bytes());
        }

        buf.extend_from_slice(&(func_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(&func_bytes);

        buf.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(name_bytes);

        buf
    }

    /// Deserializes a masking policy from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < 15 {
            return Err(ZyronError::DecodingFailed(
                "MaskingPolicy data too short".to_string(),
            ));
        }
        let mut pos = 0;

        let id = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
        pos += 4;
        let table_id = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
        pos += 4;
        let column_id = u16::from_le_bytes([data[pos], data[pos + 1]]);
        pos += 2;
        let enabled = data[pos] != 0;
        pos += 1;

        if data.len() < pos + 4 {
            return Err(ZyronError::DecodingFailed(
                "MaskingPolicy exempt count truncated".to_string(),
            ));
        }
        let exempt_count =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        let mut exempt_roles = Vec::with_capacity(exempt_count);
        for _ in 0..exempt_count {
            if data.len() < pos + 4 {
                return Err(ZyronError::DecodingFailed(
                    "MaskingPolicy exempt role truncated".to_string(),
                ));
            }
            let r = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
            pos += 4;
            exempt_roles.push(RoleId(r));
        }

        // function bytes
        if data.len() < pos + 4 {
            return Err(ZyronError::DecodingFailed(
                "MaskingPolicy function length truncated".to_string(),
            ));
        }
        let func_len =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        if data.len() < pos + func_len {
            return Err(ZyronError::DecodingFailed(
                "MaskingPolicy function data truncated".to_string(),
            ));
        }
        let (function, _) = MaskFunction::from_bytes(&data[pos..pos + func_len])?;
        pos += func_len;

        // name
        if data.len() < pos + 4 {
            return Err(ZyronError::DecodingFailed(
                "MaskingPolicy name length truncated".to_string(),
            ));
        }
        let name_len =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        if data.len() < pos + name_len {
            return Err(ZyronError::DecodingFailed(
                "MaskingPolicy name data truncated".to_string(),
            ));
        }
        let name = std::str::from_utf8(&data[pos..pos + name_len])
            .map_err(|_| ZyronError::DecodingFailed("Invalid UTF-8 in policy name".to_string()))?
            .to_string();

        Ok(Self {
            id,
            name,
            table_id,
            column_id,
            function,
            exempt_roles,
            enabled,
        })
    }
}

/// Stores masking policies indexed by (table_id, column_id).
/// Uses Rcu for lock-free reads. Policies are loaded at startup and rarely modified.
pub struct MaskingPolicyStore {
    policies: RcuMap<(u32, u16), Vec<MaskingPolicy>>,
}

impl MaskingPolicyStore {
    pub fn new() -> Self {
        Self {
            policies: RcuMap::empty_map(),
        }
    }

    /// Adds a masking policy. Rejects duplicate names on the same column.
    pub fn add_policy(&self, policy: MaskingPolicy) -> Result<()> {
        let key = (policy.table_id, policy.column_id);
        let name = policy.name.clone();
        // Check for duplicates before mutating.
        let snap = self.policies.load();
        if let Some(v) = snap.get(&key) {
            if v.iter().any(|p| p.name == name) {
                return Err(ZyronError::PolicyAlreadyExists(name));
            }
        }
        self.policies.update(|m| {
            m.entry(key).or_insert_with(Vec::new).push(policy);
        });
        Ok(())
    }

    /// Removes a masking policy by table_id, column_id, and name.
    pub fn remove_policy(&self, table_id: u32, column_id: u16, name: &str) -> bool {
        let key = (table_id, column_id);
        let mut removed = false;
        self.policies.update(|m| {
            if let Some(v) = m.get_mut(&key) {
                let before = v.len();
                v.retain(|p| p.name != name);
                removed = v.len() < before;
            }
        });
        removed
    }

    /// Returns all policies for a column.
    pub fn policies_for_column(&self, table_id: u32, column_id: u16) -> Vec<MaskingPolicy> {
        let snap = self.policies.load();
        snap.get(&(table_id, column_id))
            .cloned()
            .unwrap_or_default()
    }

    /// Applies the first matching masking policy to a value. Writes the
    /// masked result into buf. Returns true if masking was applied, false
    /// if no policy matched, the role is exempt, or the mask is Null.
    pub fn apply_masking(
        &self,
        table_id: u32,
        column_id: u16,
        value: &str,
        role_ids: &[RoleId],
        buf: &mut String,
    ) -> bool {
        let snap = self.policies.load();
        let policies = match snap.get(&(table_id, column_id)) {
            Some(v) => v,
            None => return false,
        };
        for policy in policies {
            if !policy.enabled {
                continue;
            }
            if !policy.exempt_roles.is_empty()
                && policy.exempt_roles.iter().any(|er| role_ids.contains(er))
            {
                return false;
            }
            return crate::masking::apply_mask(value, &policy.function, buf);
        }
        false
    }

    /// Bulk-loads policies from storage.
    pub fn load(&self, policies: Vec<MaskingPolicy>) {
        for policy in policies {
            let _ = self.add_policy(policy);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_policy(id: u32, name: &str, table_id: u32, column_id: u16) -> MaskingPolicy {
        MaskingPolicy {
            id,
            name: name.to_string(),
            table_id,
            column_id,
            function: MaskFunction::Redact,
            exempt_roles: vec![RoleId(99)],
            enabled: true,
        }
    }

    // -- Serialization tests --

    #[test]
    fn test_masking_policy_roundtrip() {
        let policy = MaskingPolicy {
            id: 1,
            name: "ssn_mask".to_string(),
            table_id: 42,
            column_id: 5,
            function: MaskFunction::Ssn,
            exempt_roles: vec![RoleId(10), RoleId(20)],
            enabled: true,
        };
        let bytes = policy.to_bytes();
        let restored = MaskingPolicy::from_bytes(&bytes).expect("decode");
        assert_eq!(restored.id, 1);
        assert_eq!(restored.name, "ssn_mask");
        assert_eq!(restored.table_id, 42);
        assert_eq!(restored.column_id, 5);
        assert_eq!(restored.function, MaskFunction::Ssn);
        assert_eq!(restored.exempt_roles.len(), 2);
        assert!(restored.enabled);
    }

    #[test]
    fn test_masking_policy_roundtrip_no_exempt() {
        let policy = MaskingPolicy {
            id: 2,
            name: "hash_col".to_string(),
            table_id: 10,
            column_id: 3,
            function: MaskFunction::Hash,
            exempt_roles: Vec::new(),
            enabled: false,
        };
        let bytes = policy.to_bytes();
        let restored = MaskingPolicy::from_bytes(&bytes).expect("decode");
        assert_eq!(restored.name, "hash_col");
        assert!(restored.exempt_roles.is_empty());
        assert!(!restored.enabled);
    }

    #[test]
    fn test_masking_policy_from_bytes_too_short() {
        assert!(MaskingPolicy::from_bytes(&[0u8; 5]).is_err());
    }

    // -- Store tests --

    #[test]
    fn test_store_add_and_get() {
        let store = MaskingPolicyStore::new();
        store.add_policy(make_policy(1, "p1", 100, 5)).expect("add");
        let policies = store.policies_for_column(100, 5);
        assert_eq!(policies.len(), 1);
        assert_eq!(policies[0].name, "p1");
    }

    #[test]
    fn test_store_reject_duplicate() {
        let store = MaskingPolicyStore::new();
        store.add_policy(make_policy(1, "p1", 100, 5)).expect("add");
        assert!(store.add_policy(make_policy(2, "p1", 100, 5)).is_err());
    }

    #[test]
    fn test_store_different_columns() {
        let store = MaskingPolicyStore::new();
        store.add_policy(make_policy(1, "p1", 100, 5)).expect("add");
        store.add_policy(make_policy(2, "p1", 100, 6)).expect("add");
        assert_eq!(store.policies_for_column(100, 5).len(), 1);
        assert_eq!(store.policies_for_column(100, 6).len(), 1);
    }

    #[test]
    fn test_store_remove() {
        let store = MaskingPolicyStore::new();
        store.add_policy(make_policy(1, "p1", 100, 5)).expect("add");
        assert!(store.remove_policy(100, 5, "p1"));
        assert!(store.policies_for_column(100, 5).is_empty());
    }

    #[test]
    fn test_store_remove_nonexistent() {
        let store = MaskingPolicyStore::new();
        assert!(!store.remove_policy(100, 5, "missing"));
    }

    // -- apply_masking tests --

    #[test]
    fn test_apply_masking_exempt_role() {
        let store = MaskingPolicyStore::new();
        store.add_policy(make_policy(1, "p1", 100, 5)).expect("add");
        let mut buf = String::new();
        assert!(!store.apply_masking(100, 5, "secret", &[RoleId(99)], &mut buf));
    }

    #[test]
    fn test_apply_masking_non_exempt_role() {
        let store = MaskingPolicyStore::new();
        store.add_policy(make_policy(1, "p1", 100, 5)).expect("add");
        let mut buf = String::new();
        assert!(store.apply_masking(100, 5, "secret", &[RoleId(1)], &mut buf));
        assert_eq!(buf, "[REDACTED]");
    }

    #[test]
    fn test_apply_masking_disabled_policy() {
        let store = MaskingPolicyStore::new();
        let mut policy = make_policy(1, "p1", 100, 5);
        policy.enabled = false;
        store.add_policy(policy).expect("add");
        let mut buf = String::new();
        assert!(!store.apply_masking(100, 5, "secret", &[RoleId(1)], &mut buf));
    }

    #[test]
    fn test_apply_masking_no_policies() {
        let store = MaskingPolicyStore::new();
        let mut buf = String::new();
        assert!(!store.apply_masking(100, 5, "secret", &[RoleId(1)], &mut buf));
    }

    #[test]
    fn test_apply_masking_with_email_function() {
        let store = MaskingPolicyStore::new();
        store
            .add_policy(MaskingPolicy {
                id: 1,
                name: "email_mask".to_string(),
                table_id: 100,
                column_id: 3,
                function: MaskFunction::Email,
                exempt_roles: Vec::new(),
                enabled: true,
            })
            .expect("add");
        let mut buf = String::new();
        assert!(store.apply_masking(100, 3, "john@example.com", &[RoleId(1)], &mut buf));
        assert_eq!(buf, "j***@example.com");
    }

    #[test]
    fn test_store_load() {
        let store = MaskingPolicyStore::new();
        let policies = vec![
            make_policy(1, "a", 100, 5),
            make_policy(2, "b", 100, 5),
            make_policy(3, "c", 100, 6),
        ];
        store.load(policies);
        assert_eq!(store.policies_for_column(100, 5).len(), 2);
        assert_eq!(store.policies_for_column(100, 6).len(), 1);
    }
}

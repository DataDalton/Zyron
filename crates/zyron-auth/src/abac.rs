//! Attribute-based access control (ABAC).
//!
//! Provides row-level security policies that reference session attributes
//! (department, region, clearance, IP, custom key-value pairs) to generate
//! WHERE-clause predicates for table access.

use crate::classification::ClassificationLevel;
use crate::rcu::{Rcu, RcuMap};
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
    policies: RcuMap<u32, Vec<AbacPolicy>>,
}

impl AbacStore {
    pub fn new() -> Self {
        Self {
            policies: RcuMap::empty_map(),
        }
    }

    /// Returns all policies for the given table.
    pub fn policies_for_table(&self, table_id: u32) -> Vec<AbacPolicy> {
        let snap = self.policies.load();
        snap.get(&table_id).cloned().unwrap_or_default()
    }

    /// Adds a policy to the store, grouped by table_id.
    pub fn add_policy(&self, policy: AbacPolicy) -> Result<()> {
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

    /// Bulk-loads policies, grouping by table_id.
    pub fn load(&self, policies: Vec<AbacPolicy>) {
        for policy in policies {
            let _ = self.add_policy(policy);
        }
    }
}

// ---------------------------------------------------------------------------
// ABAC Rule evaluation (resource/action gate checks)
// ---------------------------------------------------------------------------

/// Comparison operator for attribute conditions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum AbacOperator {
    Eq = 0,
    NotEq = 1,
    In = 2,
    Contains = 3,
    Matches = 4,
    Gt = 5,
    Lt = 6,
}

impl AbacOperator {
    fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(AbacOperator::Eq),
            1 => Ok(AbacOperator::NotEq),
            2 => Ok(AbacOperator::In),
            3 => Ok(AbacOperator::Contains),
            4 => Ok(AbacOperator::Matches),
            5 => Ok(AbacOperator::Gt),
            6 => Ok(AbacOperator::Lt),
            _ => Err(ZyronError::DecodingFailed(format!(
                "Unknown AbacOperator value: {}",
                v
            ))),
        }
    }
}

/// Whether a rule allows or denies access.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum AbacEffect {
    Allow = 0,
    Deny = 1,
}

/// A single condition comparing a session attribute to a value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributeCondition {
    pub attribute_key: String,
    /// The operator to apply.
    pub operator: AbacOperator,
    /// The value to compare against. For the In operator, comma-separated values.
    pub value: String,
}

impl AttributeCondition {
    /// Serializes to bytes.
    /// Layout: key_len(4) + key(N) + operator(1) + value_len(4) + value(N)
    pub fn to_bytes(&self) -> Vec<u8> {
        let key_bytes = self.attribute_key.as_bytes();
        let val_bytes = self.value.as_bytes();
        let mut buf = Vec::with_capacity(4 + key_bytes.len() + 1 + 4 + val_bytes.len());
        buf.extend_from_slice(&(key_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(key_bytes);
        buf.push(self.operator as u8);
        buf.extend_from_slice(&(val_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(val_bytes);
        buf
    }

    /// Deserializes from bytes. Returns the condition and bytes consumed.
    pub fn from_bytes(data: &[u8]) -> Result<(Self, usize)> {
        if data.len() < 9 {
            return Err(ZyronError::DecodingFailed(
                "AttributeCondition data too short".to_string(),
            ));
        }
        let mut pos = 0;
        let key_len =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        if data.len() < pos + key_len + 1 + 4 {
            return Err(ZyronError::DecodingFailed(
                "AttributeCondition key truncated".to_string(),
            ));
        }
        let attribute_key = std::str::from_utf8(&data[pos..pos + key_len])
            .map_err(|_| ZyronError::DecodingFailed("Invalid UTF-8 in attribute key".to_string()))?
            .to_string();
        pos += key_len;

        let operator = AbacOperator::from_u8(data[pos])?;
        pos += 1;

        let val_len =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        if data.len() < pos + val_len {
            return Err(ZyronError::DecodingFailed(
                "AttributeCondition value truncated".to_string(),
            ));
        }
        let value = std::str::from_utf8(&data[pos..pos + val_len])
            .map_err(|_| {
                ZyronError::DecodingFailed("Invalid UTF-8 in condition value".to_string())
            })?
            .to_string();
        pos += val_len;

        Ok((
            Self {
                attribute_key,
                operator,
                value,
            },
            pos,
        ))
    }
}

/// An ABAC rule for resource/action gate checks (evaluated once per query).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbacRule {
    pub id: u32,
    pub name: String,
    pub conditions: Vec<AttributeCondition>,
    pub effect: AbacEffect,
    /// Table name pattern to match. None = all resources.
    pub resource_pattern: Option<String>,
    /// Command to match. None = all actions.
    pub action: Option<u8>,
    pub enabled: bool,
    /// Roles this rule applies to. Empty = all roles.
    pub roles: Vec<RoleId>,
    /// Higher priority rules are evaluated first.
    pub priority: u16,
}

impl AbacRule {
    /// Serializes to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let name_bytes = self.name.as_bytes();
        let mut buf = Vec::with_capacity(128);

        buf.extend_from_slice(&self.id.to_le_bytes());
        buf.push(self.effect as u8);
        buf.push(if self.enabled { 1 } else { 0 });
        buf.extend_from_slice(&self.priority.to_le_bytes());

        // conditions
        buf.extend_from_slice(&(self.conditions.len() as u32).to_le_bytes());
        for cond in &self.conditions {
            let cond_bytes = cond.to_bytes();
            buf.extend_from_slice(&cond_bytes);
        }

        // resource_pattern
        match &self.resource_pattern {
            Some(pat) => {
                buf.push(1);
                let pat_bytes = pat.as_bytes();
                buf.extend_from_slice(&(pat_bytes.len() as u32).to_le_bytes());
                buf.extend_from_slice(pat_bytes);
            }
            None => buf.push(0),
        }

        // action
        match self.action {
            Some(a) => {
                buf.push(1);
                buf.push(a);
            }
            None => buf.push(0),
        }

        // roles
        buf.extend_from_slice(&(self.roles.len() as u32).to_le_bytes());
        for role in &self.roles {
            buf.extend_from_slice(&role.0.to_le_bytes());
        }

        // name
        buf.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(name_bytes);

        buf
    }

    /// Deserializes from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < 12 {
            return Err(ZyronError::DecodingFailed(
                "AbacRule data too short".to_string(),
            ));
        }
        let mut pos = 0;

        let id = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
        pos += 4;
        let effect = if data[pos] == 0 {
            AbacEffect::Allow
        } else {
            AbacEffect::Deny
        };
        pos += 1;
        let enabled = data[pos] != 0;
        pos += 1;
        let priority = u16::from_le_bytes([data[pos], data[pos + 1]]);
        pos += 2;

        // conditions
        if data.len() < pos + 4 {
            return Err(ZyronError::DecodingFailed(
                "AbacRule conditions count truncated".to_string(),
            ));
        }
        let cond_count =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        let mut conditions = Vec::with_capacity(cond_count);
        for _ in 0..cond_count {
            let (cond, consumed) = AttributeCondition::from_bytes(&data[pos..])?;
            pos += consumed;
            conditions.push(cond);
        }

        // resource_pattern
        if data.len() < pos + 1 {
            return Err(ZyronError::DecodingFailed(
                "AbacRule resource_pattern flag missing".to_string(),
            ));
        }
        let resource_pattern = if data[pos] == 1 {
            pos += 1;
            if data.len() < pos + 4 {
                return Err(ZyronError::DecodingFailed(
                    "AbacRule resource_pattern length truncated".to_string(),
                ));
            }
            let len = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
                as usize;
            pos += 4;
            if data.len() < pos + len {
                return Err(ZyronError::DecodingFailed(
                    "AbacRule resource_pattern data truncated".to_string(),
                ));
            }
            let s = std::str::from_utf8(&data[pos..pos + len])
                .map_err(|_| {
                    ZyronError::DecodingFailed("Invalid UTF-8 in resource_pattern".to_string())
                })?
                .to_string();
            pos += len;
            Some(s)
        } else {
            pos += 1;
            None
        };

        // action
        if data.len() < pos + 1 {
            return Err(ZyronError::DecodingFailed(
                "AbacRule action flag missing".to_string(),
            ));
        }
        let action = if data[pos] == 1 {
            pos += 1;
            if data.len() < pos + 1 {
                return Err(ZyronError::DecodingFailed(
                    "AbacRule action value missing".to_string(),
                ));
            }
            let a = data[pos];
            pos += 1;
            Some(a)
        } else {
            pos += 1;
            None
        };

        // roles
        if data.len() < pos + 4 {
            return Err(ZyronError::DecodingFailed(
                "AbacRule roles count truncated".to_string(),
            ));
        }
        let roles_count =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        let mut roles = Vec::with_capacity(roles_count);
        for _ in 0..roles_count {
            if data.len() < pos + 4 {
                return Err(ZyronError::DecodingFailed(
                    "AbacRule role data truncated".to_string(),
                ));
            }
            let r = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
            pos += 4;
            roles.push(RoleId(r));
        }

        // name
        if data.len() < pos + 4 {
            return Err(ZyronError::DecodingFailed(
                "AbacRule name length truncated".to_string(),
            ));
        }
        let name_len =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        if data.len() < pos + name_len {
            return Err(ZyronError::DecodingFailed(
                "AbacRule name data truncated".to_string(),
            ));
        }
        let name = std::str::from_utf8(&data[pos..pos + name_len])
            .map_err(|_| ZyronError::DecodingFailed("Invalid UTF-8 in rule name".to_string()))?
            .to_string();

        Ok(Self {
            id,
            name,
            conditions,
            effect,
            resource_pattern,
            action,
            enabled,
            roles,
            priority,
        })
    }
}

/// Evaluates a single condition against session attributes.
pub fn evaluate_condition(attributes: &SessionAttributes, condition: &AttributeCondition) -> bool {
    let attr_value = match condition.attribute_key.as_str() {
        "connection_time" => {
            // Compare numerically for time-based conditions
            let attr_num = attributes.connection_time;
            let cond_num: u64 = condition.value.parse().unwrap_or(0);
            return match condition.operator {
                AbacOperator::Eq => attr_num == cond_num,
                AbacOperator::NotEq => attr_num != cond_num,
                AbacOperator::Gt => attr_num > cond_num,
                AbacOperator::Lt => attr_num < cond_num,
                _ => false,
            };
        }
        "clearance" => {
            let level = attributes.clearance as u8;
            let cond_level: u8 = condition.value.parse().unwrap_or(0);
            return match condition.operator {
                AbacOperator::Eq => level == cond_level,
                AbacOperator::NotEq => level != cond_level,
                AbacOperator::Gt => level > cond_level,
                AbacOperator::Lt => level < cond_level,
                _ => false,
            };
        }
        key => match attributes.get(key) {
            Some(v) => v,
            None => return false,
        },
    };

    match condition.operator {
        AbacOperator::Eq => attr_value == condition.value,
        AbacOperator::NotEq => attr_value != condition.value,
        AbacOperator::In => condition.value.split(',').any(|v| v.trim() == attr_value),
        AbacOperator::Contains => attr_value.contains(&condition.value),
        AbacOperator::Matches => attr_value == condition.value, // Simple match (regex would need a dep)
        AbacOperator::Gt => {
            // Try numeric comparison first, fall back to lexicographic
            match (attr_value.parse::<f64>(), condition.value.parse::<f64>()) {
                (Ok(a), Ok(b)) => a > b,
                _ => attr_value > condition.value.as_str(),
            }
        }
        AbacOperator::Lt => match (attr_value.parse::<f64>(), condition.value.parse::<f64>()) {
            (Ok(a), Ok(b)) => a < b,
            _ => attr_value < condition.value.as_str(),
        },
    }
}

/// Stores ABAC rules indexed by rule id. Maintains a pre-sorted cache
/// for evaluation to avoid cloning and sorting on every call.
pub struct AbacRuleStore {
    rules: scc::HashMap<u32, AbacRule>,
    /// Rules pre-sorted by priority descending. Rebuilt on insert/remove.
    sorted_rules: Rcu<Vec<AbacRule>>,
}

impl AbacRuleStore {
    pub fn new() -> Self {
        Self {
            rules: scc::HashMap::new(),
            sorted_rules: Rcu::new(Vec::new()),
        }
    }

    /// Adds a rule. Rejects duplicate IDs.
    pub fn add_rule(&self, rule: AbacRule) -> Result<()> {
        let id = rule.id;
        let rule_clone = rule.clone();
        match self.rules.insert_sync(id, rule) {
            Ok(_) => {
                self.sorted_rules.update(|sorted| {
                    sorted.push(rule_clone);
                    sorted.sort_by(|a, b| b.priority.cmp(&a.priority));
                });
                Ok(())
            }
            Err(_) => Err(ZyronError::PolicyAlreadyExists(format!(
                "ABAC rule id {}",
                id
            ))),
        }
    }

    /// Removes a rule by ID.
    pub fn remove_rule(&self, id: u32) -> bool {
        if self.rules.remove_sync(&id).is_some() {
            self.sorted_rules.update(|sorted| {
                sorted.retain(|r| r.id != id);
            });
            true
        } else {
            false
        }
    }

    /// Evaluates all applicable rules for the given attributes, resource, and action.
    /// Deny-wins: any matching Deny rule blocks access.
    /// Returns true if access is allowed, false if denied.
    /// Uses the pre-sorted cache to avoid cloning and sorting per call.
    pub fn evaluate_abac(
        &self,
        attributes: &SessionAttributes,
        resource: Option<&str>,
        action: Option<u8>,
    ) -> bool {
        let sorted = self.sorted_rules.load();
        let mut has_allow = false;

        for rule in sorted.iter() {
            if !rule.enabled {
                continue;
            }

            // Check role match
            if !rule.roles.is_empty() && !rule.roles.iter().any(|r| *r == attributes.role_id) {
                continue;
            }

            // Check resource match
            if let Some(pattern) = &rule.resource_pattern {
                match resource {
                    Some(res) => {
                        if pattern != res {
                            continue;
                        }
                    }
                    None => continue,
                }
            }

            // Check action match
            if let Some(rule_action) = rule.action {
                match action {
                    Some(a) => {
                        if rule_action != a && rule_action != 255 {
                            continue;
                        }
                    }
                    None => continue,
                }
            }

            // Evaluate all conditions (all must be true for the rule to apply)
            let all_match = rule
                .conditions
                .iter()
                .all(|c| evaluate_condition(attributes, c));

            if !all_match {
                continue;
            }

            match rule.effect {
                AbacEffect::Deny => return false,
                AbacEffect::Allow => has_allow = true,
            }
        }

        // If no rules matched at all, default allow (ABAC is additive to privilege checks)
        if !has_allow {
            return true;
        }

        true
    }

    /// Bulk-loads rules from storage.
    pub fn load(&self, rules: Vec<AbacRule>) {
        for rule in rules {
            let _ = self.add_rule(rule);
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

    // -- AbacRule / AttributeCondition tests --

    fn make_attrs() -> SessionAttributes {
        SessionAttributes {
            role_id: RoleId(1),
            department: Some("engineering".to_string()),
            region: Some("us-east".to_string()),
            clearance: ClassificationLevel::Internal,
            ip_address: "10.0.0.1".to_string(),
            connection_time: 1000,
            custom: HashMap::new(),
        }
    }

    #[test]
    fn test_condition_eq() {
        let attrs = make_attrs();
        let cond = AttributeCondition {
            attribute_key: "department".to_string(),
            operator: AbacOperator::Eq,
            value: "engineering".to_string(),
        };
        assert!(evaluate_condition(&attrs, &cond));
    }

    #[test]
    fn test_condition_not_eq() {
        let attrs = make_attrs();
        let cond = AttributeCondition {
            attribute_key: "department".to_string(),
            operator: AbacOperator::NotEq,
            value: "sales".to_string(),
        };
        assert!(evaluate_condition(&attrs, &cond));
    }

    #[test]
    fn test_condition_in() {
        let attrs = make_attrs();
        let cond = AttributeCondition {
            attribute_key: "region".to_string(),
            operator: AbacOperator::In,
            value: "us-east, eu-west, ap-south".to_string(),
        };
        assert!(evaluate_condition(&attrs, &cond));
    }

    #[test]
    fn test_condition_in_not_matching() {
        let attrs = make_attrs();
        let cond = AttributeCondition {
            attribute_key: "region".to_string(),
            operator: AbacOperator::In,
            value: "eu-west, ap-south".to_string(),
        };
        assert!(!evaluate_condition(&attrs, &cond));
    }

    #[test]
    fn test_condition_time_gt() {
        let attrs = make_attrs();
        let cond = AttributeCondition {
            attribute_key: "connection_time".to_string(),
            operator: AbacOperator::Gt,
            value: "500".to_string(),
        };
        assert!(evaluate_condition(&attrs, &cond));
    }

    #[test]
    fn test_condition_time_lt() {
        let attrs = make_attrs();
        let cond = AttributeCondition {
            attribute_key: "connection_time".to_string(),
            operator: AbacOperator::Lt,
            value: "2000".to_string(),
        };
        assert!(evaluate_condition(&attrs, &cond));
    }

    #[test]
    fn test_condition_missing_attribute() {
        let attrs = make_attrs();
        let cond = AttributeCondition {
            attribute_key: "nonexistent".to_string(),
            operator: AbacOperator::Eq,
            value: "anything".to_string(),
        };
        assert!(!evaluate_condition(&attrs, &cond));
    }

    #[test]
    fn test_abac_rule_roundtrip() {
        let rule = AbacRule {
            id: 1,
            name: "business_hours".to_string(),
            conditions: vec![
                AttributeCondition {
                    attribute_key: "connection_time".to_string(),
                    operator: AbacOperator::Gt,
                    value: "32400".to_string(),
                },
                AttributeCondition {
                    attribute_key: "region".to_string(),
                    operator: AbacOperator::Eq,
                    value: "us-east".to_string(),
                },
            ],
            effect: AbacEffect::Allow,
            resource_pattern: Some("sensitive_data".to_string()),
            action: Some(0), // Select
            enabled: true,
            roles: vec![RoleId(10)],
            priority: 100,
        };
        let bytes = rule.to_bytes();
        let restored = AbacRule::from_bytes(&bytes).expect("decode");
        assert_eq!(restored.id, 1);
        assert_eq!(restored.name, "business_hours");
        assert_eq!(restored.conditions.len(), 2);
        assert_eq!(restored.effect, AbacEffect::Allow);
        assert_eq!(restored.resource_pattern.as_deref(), Some("sensitive_data"));
        assert_eq!(restored.action, Some(0));
        assert!(restored.enabled);
        assert_eq!(restored.priority, 100);
    }

    #[test]
    fn test_abac_rule_roundtrip_minimal() {
        let rule = AbacRule {
            id: 2,
            name: "deny_all".to_string(),
            conditions: Vec::new(),
            effect: AbacEffect::Deny,
            resource_pattern: None,
            action: None,
            enabled: true,
            roles: Vec::new(),
            priority: 0,
        };
        let bytes = rule.to_bytes();
        let restored = AbacRule::from_bytes(&bytes).expect("decode");
        assert_eq!(restored.id, 2);
        assert_eq!(restored.name, "deny_all");
        assert!(restored.conditions.is_empty());
        assert_eq!(restored.effect, AbacEffect::Deny);
        assert!(restored.resource_pattern.is_none());
    }

    #[test]
    fn test_abac_rule_store_add_and_remove() {
        let store = AbacRuleStore::new();
        let rule = AbacRule {
            id: 1,
            name: "test".to_string(),
            conditions: Vec::new(),
            effect: AbacEffect::Allow,
            resource_pattern: None,
            action: None,
            enabled: true,
            roles: Vec::new(),
            priority: 0,
        };
        store.add_rule(rule).expect("add");
        assert!(store.remove_rule(1));
        assert!(!store.remove_rule(1));
    }

    #[test]
    fn test_abac_rule_store_deny_wins() {
        let store = AbacRuleStore::new();
        store
            .add_rule(AbacRule {
                id: 1,
                name: "allow_all".to_string(),
                conditions: Vec::new(),
                effect: AbacEffect::Allow,
                resource_pattern: None,
                action: None,
                enabled: true,
                roles: Vec::new(),
                priority: 10,
            })
            .expect("add");
        store
            .add_rule(AbacRule {
                id: 2,
                name: "deny_all".to_string(),
                conditions: Vec::new(),
                effect: AbacEffect::Deny,
                resource_pattern: None,
                action: None,
                enabled: true,
                roles: Vec::new(),
                priority: 20,
            })
            .expect("add");
        let attrs = make_attrs();
        assert!(!store.evaluate_abac(&attrs, None, None));
    }

    #[test]
    fn test_abac_rule_store_no_rules_allows() {
        let store = AbacRuleStore::new();
        let attrs = make_attrs();
        assert!(store.evaluate_abac(&attrs, None, None));
    }

    #[test]
    fn test_abac_rule_store_disabled_skipped() {
        let store = AbacRuleStore::new();
        store
            .add_rule(AbacRule {
                id: 1,
                name: "deny_disabled".to_string(),
                conditions: Vec::new(),
                effect: AbacEffect::Deny,
                resource_pattern: None,
                action: None,
                enabled: false,
                roles: Vec::new(),
                priority: 100,
            })
            .expect("add");
        let attrs = make_attrs();
        assert!(store.evaluate_abac(&attrs, None, None));
    }

    #[test]
    fn test_abac_rule_store_condition_must_match() {
        let store = AbacRuleStore::new();
        store
            .add_rule(AbacRule {
                id: 1,
                name: "deny_sales".to_string(),
                conditions: vec![AttributeCondition {
                    attribute_key: "department".to_string(),
                    operator: AbacOperator::Eq,
                    value: "sales".to_string(),
                }],
                effect: AbacEffect::Deny,
                resource_pattern: None,
                action: None,
                enabled: true,
                roles: Vec::new(),
                priority: 100,
            })
            .expect("add");
        let attrs = make_attrs(); // department = "engineering"
        // Condition does not match, so deny rule does not fire
        assert!(store.evaluate_abac(&attrs, None, None));
    }

    #[test]
    fn test_abac_rule_store_resource_pattern() {
        let store = AbacRuleStore::new();
        store
            .add_rule(AbacRule {
                id: 1,
                name: "deny_sensitive".to_string(),
                conditions: Vec::new(),
                effect: AbacEffect::Deny,
                resource_pattern: Some("sensitive_table".to_string()),
                action: None,
                enabled: true,
                roles: Vec::new(),
                priority: 100,
            })
            .expect("add");
        let attrs = make_attrs();
        // Different resource, should not match
        assert!(store.evaluate_abac(&attrs, Some("public_table"), None));
        // Matching resource, should deny
        assert!(!store.evaluate_abac(&attrs, Some("sensitive_table"), None));
    }
}

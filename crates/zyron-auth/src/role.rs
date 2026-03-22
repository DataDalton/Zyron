//! Role management with hierarchy traversal and cycle detection.
//!
//! Roles form a directed acyclic graph (DAG) where edges represent membership.
//! Each edge carries an "inherit" flag that controls whether the member
//! automatically receives the parent's privileges. The RoleHierarchy struct
//! provides thread-safe traversal with a depth limit to prevent runaway
//! recursion in the presence of bugs.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

use serde::{Deserialize, Serialize};
use zyron_catalog::encoding::{
    read_bool, read_string, read_u8, read_u32, read_u64, write_bool, write_string, write_u8,
    write_u32, write_u64,
};
use zyron_common::{Result, ZyronError};

use crate::classification::ClassificationLevel;
use crate::rcu::RcuMap;

/// Unique identifier for a role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RoleId(pub u32);

impl fmt::Display for RoleId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "role:{}", self.0)
    }
}

/// Unique identifier for a user. Shares the same ID space as RoleId.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct UserId(pub u32);

impl fmt::Display for UserId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "user:{}", self.0)
    }
}

/// A database role with a clearance level for privilege grouping.
#[derive(Clone)]
pub struct Role {
    pub id: RoleId,
    pub name: String,
    pub clearance: ClassificationLevel,
    pub created_at: u64,
}

impl Role {
    /// Serializes the role to a binary byte vector for storage.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(32);
        write_u32(&mut buf, self.id.0);
        write_string(&mut buf, &self.name);
        write_u8(&mut buf, self.clearance as u8);
        write_u64(&mut buf, self.created_at);
        buf
    }

    /// Deserializes a role from a binary byte slice.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let mut off = 0;
        let id = RoleId(read_u32(data, &mut off)?);
        let name = read_string(data, &mut off)?;
        let clearance_val = read_u8(data, &mut off)?;
        let clearance = ClassificationLevel::from_u8(clearance_val)?;
        let created_at = read_u64(data, &mut off)?;

        Ok(Self {
            id,
            name,
            clearance,
            created_at,
        })
    }
}

/// Records a membership edge: member_id is a member of parent_id.
#[derive(Debug, Clone)]
pub struct RoleMembership {
    pub member_id: RoleId,
    pub parent_id: RoleId,
    pub admin_option: bool,
    pub inherit: bool,
    pub granted_by: RoleId,
}

impl RoleMembership {
    /// Serializes the membership to a binary byte vector.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(20);
        write_u32(&mut buf, self.member_id.0);
        write_u32(&mut buf, self.parent_id.0);
        write_bool(&mut buf, self.admin_option);
        write_bool(&mut buf, self.inherit);
        write_u32(&mut buf, self.granted_by.0);
        buf
    }

    /// Deserializes a membership from a binary byte slice.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let mut off = 0;
        let member_id = RoleId(read_u32(data, &mut off)?);
        let parent_id = RoleId(read_u32(data, &mut off)?);
        let admin_option = read_bool(data, &mut off)?;
        let inherit = read_bool(data, &mut off)?;
        let granted_by = RoleId(read_u32(data, &mut off)?);
        Ok(Self {
            member_id,
            parent_id,
            admin_option,
            inherit,
            granted_by,
        })
    }
}

/// Maximum depth for role hierarchy traversal to prevent infinite loops.
const MAX_ROLE_DEPTH: usize = 64;

/// Thread-safe role hierarchy stored as an adjacency list.
/// Key = member role, Value = list of (parent role, inherit flag).
pub struct RoleHierarchy {
    adjacency: RcuMap<RoleId, Vec<(RoleId, bool)>>,
}

impl RoleHierarchy {
    /// Creates an empty role hierarchy.
    pub fn new() -> Self {
        Self {
            adjacency: RcuMap::empty_map(),
        }
    }

    /// Bulk loads the adjacency list from a slice of membership entries.
    /// Replaces all existing data.
    pub fn load(&self, memberships: &[RoleMembership]) {
        let mut adj = HashMap::new();
        for m in memberships {
            adj.entry(m.member_id)
                .or_insert_with(Vec::new)
                .push((m.parent_id, m.inherit));
        }
        self.adjacency.store(adj);
    }

    /// Adds a membership edge after checking for cycles.
    /// Returns an error if adding this edge would create a cycle.
    pub fn add_membership(&self, member: RoleId, parent: RoleId, inherit: bool) -> Result<()> {
        if member == parent {
            return Err(ZyronError::CircularRoleDependency);
        }

        let snap = self.adjacency.load();
        if would_create_cycle(&snap, member, parent) {
            return Err(ZyronError::CircularRoleDependency);
        }

        self.adjacency.update(|m| {
            m.entry(member)
                .or_insert_with(Vec::new)
                .push((parent, inherit));
        });
        Ok(())
    }

    /// Removes a membership edge between member and parent.
    pub fn remove_membership(&self, member: RoleId, parent: RoleId) {
        self.adjacency.update(|m| {
            if let Some(parents) = m.get_mut(&member) {
                parents.retain(|(p, _)| *p != parent);
                if parents.is_empty() {
                    m.remove(&member);
                }
            }
        });
    }

    /// Returns all roles that the given role inherits from, including itself.
    /// Only follows edges where inherit=true. BFS with depth limit.
    pub fn effective_roles(&self, role_id: RoleId) -> Vec<RoleId> {
        let adj = self.adjacency.load();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut result = Vec::new();

        visited.insert(role_id);
        result.push(role_id);
        queue.push_back((role_id, 0usize));

        while let Some((current, depth)) = queue.pop_front() {
            if depth >= MAX_ROLE_DEPTH {
                break;
            }
            if let Some(parents) = adj.get(&current) {
                for &(parent, inherit) in parents {
                    if inherit && visited.insert(parent) {
                        result.push(parent);
                        queue.push_back((parent, depth + 1));
                    }
                }
            }
        }
        result
    }

    /// Returns all roles reachable from the given role, including non-inherited ones.
    /// Includes the role itself. BFS with depth limit.
    pub fn all_roles(&self, role_id: RoleId) -> Vec<RoleId> {
        let adj = self.adjacency.load();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut result = Vec::new();

        visited.insert(role_id);
        result.push(role_id);
        queue.push_back((role_id, 0usize));

        while let Some((current, depth)) = queue.pop_front() {
            if depth >= MAX_ROLE_DEPTH {
                break;
            }
            if let Some(parents) = adj.get(&current) {
                for &(parent, _) in parents {
                    if visited.insert(parent) {
                        result.push(parent);
                        queue.push_back((parent, depth + 1));
                    }
                }
            }
        }
        result
    }

    /// Returns true if role_id is a (possibly transitive) member of group_id.
    /// Uses DFS with depth limit.
    pub fn is_member_of(&self, role_id: RoleId, group_id: RoleId) -> bool {
        if role_id == group_id {
            return true;
        }
        let adj = self.adjacency.load();
        let mut visited = HashSet::new();
        let mut stack = Vec::new();
        stack.push((role_id, 0usize));
        visited.insert(role_id);

        while let Some((current, depth)) = stack.pop() {
            if depth >= MAX_ROLE_DEPTH {
                continue;
            }
            if let Some(parents) = adj.get(&current) {
                for &(parent, _) in parents {
                    if parent == group_id {
                        return true;
                    }
                    if visited.insert(parent) {
                        stack.push((parent, depth + 1));
                    }
                }
            }
        }
        false
    }
}

/// DFS from parent to check if member is reachable, which would create a cycle.
fn would_create_cycle(
    adj: &HashMap<RoleId, Vec<(RoleId, bool)>>,
    member: RoleId,
    parent: RoleId,
) -> bool {
    let mut visited = HashSet::new();
    let mut stack = vec![parent];
    visited.insert(parent);

    while let Some(current) = stack.pop() {
        if current == member {
            return true;
        }
        if let Some(parents) = adj.get(&current) {
            for &(p, _) in parents {
                if visited.insert(p) {
                    stack.push(p);
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_role(id: u32, name: &str) -> Role {
        Role {
            id: RoleId(id),
            name: name.to_string(),
            clearance: ClassificationLevel::Public,
            created_at: 1700000000,
        }
    }

    #[test]
    fn test_role_id_display() {
        assert_eq!(RoleId(42).to_string(), "role:42");
    }

    #[test]
    fn test_user_id_display() {
        assert_eq!(UserId(7).to_string(), "user:7");
    }

    #[test]
    fn test_role_to_bytes_from_bytes_roundtrip() {
        let role = make_role(1, "alice");
        let bytes = role.to_bytes();
        let restored = Role::from_bytes(&bytes).expect("from_bytes failed");
        assert_eq!(restored.id, RoleId(1));
        assert_eq!(restored.name, "alice");
        assert_eq!(restored.clearance, ClassificationLevel::Public);
        assert_eq!(restored.created_at, 1700000000);
    }

    #[test]
    fn test_role_to_bytes_from_bytes_restricted() {
        let role = Role {
            id: RoleId(99),
            name: "superadmin".to_string(),
            clearance: ClassificationLevel::Restricted,
            created_at: 1700000000,
        };
        let bytes = role.to_bytes();
        let restored = Role::from_bytes(&bytes).expect("from_bytes failed");
        assert_eq!(restored.id, RoleId(99));
        assert_eq!(restored.name, "superadmin");
        assert_eq!(restored.clearance, ClassificationLevel::Restricted);
        assert_eq!(restored.created_at, 1700000000);
    }

    #[test]
    fn test_role_membership_to_bytes_from_bytes() {
        let m = RoleMembership {
            member_id: RoleId(10),
            parent_id: RoleId(20),
            admin_option: true,
            inherit: false,
            granted_by: RoleId(1),
        };
        let bytes = m.to_bytes();
        let restored = RoleMembership::from_bytes(&bytes).expect("from_bytes failed");
        assert_eq!(restored.member_id, RoleId(10));
        assert_eq!(restored.parent_id, RoleId(20));
        assert!(restored.admin_option);
        assert!(!restored.inherit);
        assert_eq!(restored.granted_by, RoleId(1));
    }

    #[test]
    fn test_hierarchy_load() {
        let h = RoleHierarchy::new();
        let memberships = vec![
            RoleMembership {
                member_id: RoleId(1),
                parent_id: RoleId(2),
                admin_option: false,
                inherit: true,
                granted_by: RoleId(0),
            },
            RoleMembership {
                member_id: RoleId(1),
                parent_id: RoleId(3),
                admin_option: false,
                inherit: false,
                granted_by: RoleId(0),
            },
        ];
        h.load(&memberships);
        let eff = h.effective_roles(RoleId(1));
        // Should include self (1) and inherited parent (2), but not non-inherited (3).
        assert!(eff.contains(&RoleId(1)));
        assert!(eff.contains(&RoleId(2)));
        assert!(!eff.contains(&RoleId(3)));
    }

    #[test]
    fn test_hierarchy_all_roles_includes_non_inherited() {
        let h = RoleHierarchy::new();
        let memberships = vec![
            RoleMembership {
                member_id: RoleId(1),
                parent_id: RoleId(2),
                admin_option: false,
                inherit: true,
                granted_by: RoleId(0),
            },
            RoleMembership {
                member_id: RoleId(1),
                parent_id: RoleId(3),
                admin_option: false,
                inherit: false,
                granted_by: RoleId(0),
            },
        ];
        h.load(&memberships);
        let all = h.all_roles(RoleId(1));
        assert!(all.contains(&RoleId(1)));
        assert!(all.contains(&RoleId(2)));
        assert!(all.contains(&RoleId(3)));
    }

    #[test]
    fn test_hierarchy_transitive() {
        // 1 -> 2 -> 3 (all inherited)
        let h = RoleHierarchy::new();
        h.add_membership(RoleId(1), RoleId(2), true)
            .expect("add failed");
        h.add_membership(RoleId(2), RoleId(3), true)
            .expect("add failed");

        let eff = h.effective_roles(RoleId(1));
        assert!(eff.contains(&RoleId(1)));
        assert!(eff.contains(&RoleId(2)));
        assert!(eff.contains(&RoleId(3)));
    }

    #[test]
    fn test_hierarchy_is_member_of() {
        let h = RoleHierarchy::new();
        h.add_membership(RoleId(1), RoleId(2), true)
            .expect("add failed");
        h.add_membership(RoleId(2), RoleId(3), true)
            .expect("add failed");

        assert!(h.is_member_of(RoleId(1), RoleId(3)));
        assert!(h.is_member_of(RoleId(1), RoleId(2)));
        assert!(h.is_member_of(RoleId(1), RoleId(1)));
        assert!(!h.is_member_of(RoleId(3), RoleId(1)));
    }

    #[test]
    fn test_hierarchy_cycle_detection_direct() {
        let h = RoleHierarchy::new();
        h.add_membership(RoleId(1), RoleId(2), true)
            .expect("add failed");
        let result = h.add_membership(RoleId(2), RoleId(1), true);
        assert!(result.is_err());
    }

    #[test]
    fn test_hierarchy_cycle_detection_transitive() {
        let h = RoleHierarchy::new();
        h.add_membership(RoleId(1), RoleId(2), true)
            .expect("add failed");
        h.add_membership(RoleId(2), RoleId(3), true)
            .expect("add failed");
        let result = h.add_membership(RoleId(3), RoleId(1), true);
        assert!(result.is_err());
    }

    #[test]
    fn test_hierarchy_self_membership_rejected() {
        let h = RoleHierarchy::new();
        let result = h.add_membership(RoleId(5), RoleId(5), true);
        assert!(result.is_err());
    }

    #[test]
    fn test_hierarchy_remove_membership() {
        let h = RoleHierarchy::new();
        h.add_membership(RoleId(1), RoleId(2), true)
            .expect("add failed");
        h.add_membership(RoleId(1), RoleId(3), true)
            .expect("add failed");

        h.remove_membership(RoleId(1), RoleId(2));

        let eff = h.effective_roles(RoleId(1));
        assert!(!eff.contains(&RoleId(2)));
        assert!(eff.contains(&RoleId(3)));
    }

    #[test]
    fn test_hierarchy_remove_nonexistent() {
        let h = RoleHierarchy::new();
        // Removing a non-existent edge should not panic.
        h.remove_membership(RoleId(1), RoleId(2));
    }

    #[test]
    fn test_hierarchy_diamond() {
        // Diamond: 1->2, 1->3, 2->4, 3->4
        let h = RoleHierarchy::new();
        h.add_membership(RoleId(1), RoleId(2), true)
            .expect("add failed");
        h.add_membership(RoleId(1), RoleId(3), true)
            .expect("add failed");
        h.add_membership(RoleId(2), RoleId(4), true)
            .expect("add failed");
        h.add_membership(RoleId(3), RoleId(4), true)
            .expect("add failed");

        let eff = h.effective_roles(RoleId(1));
        assert_eq!(eff.len(), 4); // 1, 2, 3, 4 each counted once
        assert!(eff.contains(&RoleId(4)));
    }

    #[test]
    fn test_hierarchy_no_parents() {
        let h = RoleHierarchy::new();
        let eff = h.effective_roles(RoleId(42));
        assert_eq!(eff, vec![RoleId(42)]);
    }

    #[test]
    fn test_hierarchy_non_inherited_not_in_effective() {
        let h = RoleHierarchy::new();
        h.add_membership(RoleId(1), RoleId(2), false)
            .expect("add failed");
        let eff = h.effective_roles(RoleId(1));
        // Only self, since the edge is not inherited.
        assert_eq!(eff, vec![RoleId(1)]);

        // But all_roles includes it.
        let all = h.all_roles(RoleId(1));
        assert!(all.contains(&RoleId(2)));
    }

    #[test]
    fn test_hierarchy_is_member_of_ignores_inherit() {
        // is_member_of follows all edges regardless of inherit flag.
        let h = RoleHierarchy::new();
        h.add_membership(RoleId(1), RoleId(2), false)
            .expect("add failed");
        assert!(h.is_member_of(RoleId(1), RoleId(2)));
    }

    #[test]
    fn test_role_from_bytes_truncated() {
        let data = vec![0u8; 2];
        assert!(Role::from_bytes(&data).is_err());
    }

    #[test]
    fn test_membership_from_bytes_truncated() {
        let data = vec![0u8; 3];
        assert!(RoleMembership::from_bytes(&data).is_err());
    }
}

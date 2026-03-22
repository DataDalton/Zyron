//! SecurityContext with privilege cache and impersonation.
//!
//! A SecurityContext is created per session and tracks the current role,
//! effective roles, clearance level, session attributes, and a privilege
//! decision cache. It supports SET ROLE, EXECUTE AS impersonation with
//! a stack for nested impersonation, and break-glass override tracking.

use crate::abac::SessionAttributes;
use crate::classification::ClassificationLevel;
use crate::classification::ClassificationStore;
use crate::privilege::{ObjectType, PrivilegeDecision, PrivilegeStore, PrivilegeType};
use crate::role::{RoleHierarchy, RoleId, UserId};
use crate::session_binding::QueryLimits;
use std::collections::HashMap;

/// Cached privilege decisions keyed by (privilege, object_type, object_id).
/// The generation counter tracks when the cache was last validated.
struct PrivilegeCache {
    entries: HashMap<(u8, u8, u32), PrivilegeDecision>,
    generation: u64,
}

impl PrivilegeCache {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
            generation: 0,
        }
    }

    fn get(
        &self,
        priv_type: u8,
        obj_type: u8,
        obj_id: u32,
        current_gen: u64,
    ) -> Option<PrivilegeDecision> {
        if self.generation != current_gen {
            return None;
        }
        self.entries.get(&(priv_type, obj_type, obj_id)).copied()
    }

    fn insert(
        &mut self,
        priv_type: u8,
        obj_type: u8,
        obj_id: u32,
        decision: PrivilegeDecision,
        current_gen: u64,
    ) {
        if self.generation != current_gen {
            self.entries.clear();
            self.generation = current_gen;
        }
        self.entries.insert((priv_type, obj_type, obj_id), decision);
    }

    fn invalidate(&mut self) {
        self.entries.clear();
        self.generation = self.generation.wrapping_add(1);
    }
}

/// Per-session security state tracking the active role, privileges, and attributes.
pub struct SecurityContext {
    pub user_id: UserId,
    pub session_role: RoleId,
    pub current_role: RoleId,
    pub effective_roles: Vec<RoleId>,
    pub all_roles: Vec<RoleId>,
    pub clearance: ClassificationLevel,
    pub attributes: SessionAttributes,
    pub bound_ip: Option<String>,
    pub query_limits: QueryLimits,
    pub break_glass: Option<RoleId>,
    pub break_glass_clearance: Option<ClassificationLevel>,
    impersonation_stack: Vec<RoleId>,
    cache: PrivilegeCache,
}

impl SecurityContext {
    /// Creates a new SecurityContext for the given user and role.
    pub fn new(
        user_id: UserId,
        role_id: RoleId,
        effective_roles: Vec<RoleId>,
        all_roles: Vec<RoleId>,
        clearance: ClassificationLevel,
        attributes: SessionAttributes,
        bound_ip: Option<String>,
        query_limits: QueryLimits,
    ) -> Self {
        Self {
            user_id,
            session_role: role_id,
            current_role: role_id,
            effective_roles,
            all_roles,
            clearance,
            attributes,
            bound_ip,
            query_limits,
            break_glass: None,
            break_glass_clearance: None,
            impersonation_stack: Vec::new(),
            cache: PrivilegeCache::new(),
        }
    }

    /// Checks whether the current role has the given privilege on the specified object.
    /// Results are cached per generation.
    pub fn has_privilege(
        &mut self,
        store: &PrivilegeStore,
        privilege: PrivilegeType,
        object_type: ObjectType,
        object_id: u32,
        columns: Option<&[u16]>,
        now: u64,
    ) -> bool {
        let priv_u8 = privilege as u8;
        let obj_u8 = object_type as u8;
        let store_gen = store.generation();

        // Check cache for column-less lookups.
        if columns.is_none() {
            if let Some(decision) = self.cache.get(priv_u8, obj_u8, object_id, store_gen) {
                return decision == PrivilegeDecision::Allow;
            }
        }

        let decision = store.check_privilege(
            &self.effective_roles,
            privilege,
            object_type,
            object_id,
            columns,
            now,
        );

        if columns.is_none() {
            self.cache
                .insert(priv_u8, obj_u8, object_id, decision, store_gen);
        }

        decision == PrivilegeDecision::Allow
    }

    /// Checks whether the role clearance is sufficient for the given column.
    pub fn check_clearance(
        &self,
        store: &ClassificationStore,
        table_id: u32,
        column_id: u16,
    ) -> bool {
        store.check_clearance(self.clearance, table_id, column_id)
    }

    /// Switches the current role to the target. The target must be in all_roles.
    /// Updates effective_roles to match the new current role.
    pub fn set_role(
        &mut self,
        target: RoleId,
        hierarchy: &RoleHierarchy,
    ) -> zyron_common::Result<()> {
        if !self.all_roles.contains(&target) {
            return Err(zyron_common::ZyronError::PermissionDenied(format!(
                "Role {} is not in the set of allowed roles",
                target
            )));
        }
        self.current_role = target;
        self.effective_roles = hierarchy.effective_roles(target);
        self.cache.invalidate();
        Ok(())
    }

    /// Resets the current role back to the session role.
    pub fn reset_role(&mut self, hierarchy: &RoleHierarchy) {
        self.current_role = self.session_role;
        self.effective_roles = hierarchy.effective_roles(self.session_role);
        self.cache.invalidate();
    }

    /// Impersonates a target role if the current role has Impersonate privilege on it.
    /// Pushes the current role onto the impersonation stack.
    pub fn execute_as(
        &mut self,
        target: RoleId,
        store: &PrivilegeStore,
        hierarchy: &RoleHierarchy,
    ) -> zyron_common::Result<()> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let decision = store.check_privilege(
            &self.effective_roles,
            PrivilegeType::Impersonate,
            ObjectType::Type,
            target.0,
            None,
            now,
        );
        if decision != PrivilegeDecision::Allow {
            return Err(zyron_common::ZyronError::PermissionDenied(format!(
                "No Impersonate privilege on {}",
                target
            )));
        }
        self.impersonation_stack.push(self.current_role);
        self.current_role = target;
        self.effective_roles = hierarchy.effective_roles(target);
        self.cache.invalidate();
        Ok(())
    }

    /// Reverts the most recent impersonation, restoring the previous role.
    pub fn revert(&mut self, hierarchy: &RoleHierarchy) -> zyron_common::Result<()> {
        let previous = self.impersonation_stack.pop().ok_or_else(|| {
            zyron_common::ZyronError::PermissionDenied("No impersonation to revert".to_string())
        })?;
        self.current_role = previous;
        self.effective_roles = hierarchy.effective_roles(previous);
        self.cache.invalidate();
        Ok(())
    }

    /// Clears the privilege cache, forcing re-resolution on next check.
    pub fn invalidate_cache(&mut self) {
        self.cache.invalidate();
    }

    /// Returns the user ID for this session.
    pub fn current_user_id(&self) -> UserId {
        self.user_id
    }

    /// Returns the effective roles for the current role.
    pub fn current_roles(&self) -> &[RoleId] {
        &self.effective_roles
    }

    /// Returns the effective clearance level, considering break-glass elevation.
    pub fn effective_clearance(&self) -> ClassificationLevel {
        match self.break_glass_clearance {
            Some(elevated) if elevated > self.clearance => elevated,
            _ => self.clearance,
        }
    }

    /// Looks up a session attribute by key.
    pub fn get_attribute(&self, key: &str) -> Option<&str> {
        self.attributes.get(key)
    }

    /// Sets a custom session attribute.
    pub fn set_attribute(&mut self, key: String, value: String) {
        self.attributes.set(key, value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::privilege::{GrantEntry, PrivilegeState};
    use std::collections::HashMap as StdHashMap;

    fn make_attributes(role_id: RoleId) -> SessionAttributes {
        SessionAttributes {
            role_id,
            department: Some("engineering".to_string()),
            region: None,
            clearance: ClassificationLevel::Internal,
            ip_address: "10.0.0.1".to_string(),
            connection_time: 1000,
            custom: StdHashMap::new(),
        }
    }

    fn make_admin_context() -> (SecurityContext, PrivilegeStore) {
        let role = RoleId(1);
        let store = PrivilegeStore::new();
        // Grant All privileges on all object types so admin has full access through normal resolution.
        let grant = GrantEntry {
            grantee: role,
            privilege: PrivilegeType::All,
            object_type: ObjectType::Table,
            object_id: 0,
            columns: None,
            state: PrivilegeState::Grant,
            with_grant_option: true,
            granted_by: role,
            valid_from: None,
            valid_until: None,
            time_window: None,
            object_pattern: Some("%".to_string()),
            no_inherit: false,
            mask_function: None,
        };
        store.grant(grant).expect("grant should succeed");
        let ctx = SecurityContext::new(
            UserId(1),
            role,
            vec![role],
            vec![role],
            ClassificationLevel::Restricted,
            make_attributes(role),
            Some("10.0.0.1".to_string()),
            QueryLimits::default(),
        );
        (ctx, store)
    }

    fn make_normal_context() -> SecurityContext {
        let role = RoleId(10);
        let parent = RoleId(20);
        SecurityContext::new(
            UserId(10),
            role,
            vec![role, parent],
            vec![role, parent, RoleId(30)],
            ClassificationLevel::Internal,
            make_attributes(role),
            Some("10.0.0.1".to_string()),
            QueryLimits::default(),
        )
    }

    #[test]
    fn test_admin_with_all_privileges() {
        let (mut ctx, store) = make_admin_context();
        // Admin with All privilege granted via pattern should get access through normal resolution.
        // Grant a specific object entry for object_id 100 so the non-pattern check works.
        let grant = GrantEntry {
            grantee: RoleId(1),
            privilege: PrivilegeType::All,
            object_type: ObjectType::Table,
            object_id: 100,
            columns: None,
            state: PrivilegeState::Grant,
            with_grant_option: true,
            granted_by: RoleId(1),
            valid_from: None,
            valid_until: None,
            time_window: None,
            object_pattern: None,
            no_inherit: false,
            mask_function: None,
        };
        store.grant(grant).expect("grant should succeed");
        assert!(ctx.has_privilege(
            &store,
            PrivilegeType::Select,
            ObjectType::Table,
            100,
            None,
            1000,
        ));
    }

    #[test]
    fn test_normal_user_no_grant_denied() {
        let mut ctx = make_normal_context();
        let store = PrivilegeStore::new();
        // No grants in store, normal user should be denied.
        assert!(!ctx.has_privilege(
            &store,
            PrivilegeType::Select,
            ObjectType::Table,
            100,
            None,
            1000,
        ));
    }

    #[test]
    fn test_cache_invalidation() {
        let mut ctx = make_normal_context();
        let store = PrivilegeStore::new();
        // First call populates cache.
        let result1 = ctx.has_privilege(
            &store,
            PrivilegeType::Select,
            ObjectType::Table,
            100,
            None,
            1000,
        );
        assert!(!result1);

        // Invalidate cache.
        ctx.invalidate_cache();
        // After invalidation, the cache miss forces re-resolution.
        let result2 = ctx.has_privilege(
            &store,
            PrivilegeType::Select,
            ObjectType::Table,
            100,
            None,
            1000,
        );
        assert!(!result2);
    }

    #[test]
    fn test_set_role_valid() {
        let mut ctx = make_normal_context();
        let hierarchy = RoleHierarchy::new();
        // RoleId(20) is in all_roles.
        ctx.set_role(RoleId(20), &hierarchy)
            .expect("set_role should succeed");
        assert_eq!(ctx.current_role, RoleId(20));
    }

    #[test]
    fn test_set_role_invalid() {
        let mut ctx = make_normal_context();
        let hierarchy = RoleHierarchy::new();
        // RoleId(999) is not in all_roles.
        let result = ctx.set_role(RoleId(999), &hierarchy);
        assert!(result.is_err());
    }

    #[test]
    fn test_reset_role() {
        let mut ctx = make_normal_context();
        let hierarchy = RoleHierarchy::new();
        ctx.set_role(RoleId(20), &hierarchy)
            .expect("set_role should succeed");
        ctx.reset_role(&hierarchy);
        assert_eq!(ctx.current_role, ctx.session_role);
    }

    #[test]
    fn test_execute_as_with_impersonate_privilege() {
        let role = RoleId(1);
        let store = PrivilegeStore::new();
        let grant = GrantEntry {
            grantee: role,
            privilege: PrivilegeType::Impersonate,
            object_type: ObjectType::Type,
            object_id: 50,
            columns: None,
            state: PrivilegeState::Grant,
            with_grant_option: false,
            granted_by: role,
            valid_from: None,
            valid_until: None,
            time_window: None,
            object_pattern: None,
            no_inherit: false,
            mask_function: None,
        };
        store.grant(grant).expect("grant should succeed");
        let mut ctx = SecurityContext::new(
            UserId(1),
            role,
            vec![role],
            vec![role],
            ClassificationLevel::Restricted,
            make_attributes(role),
            Some("10.0.0.1".to_string()),
            QueryLimits::default(),
        );
        let hierarchy = RoleHierarchy::new();
        ctx.execute_as(RoleId(50), &store, &hierarchy)
            .expect("execute_as should succeed");
        assert_eq!(ctx.current_role, RoleId(50));
    }

    #[test]
    fn test_execute_as_normal_no_privilege() {
        let mut ctx = make_normal_context();
        let store = PrivilegeStore::new();
        let hierarchy = RoleHierarchy::new();
        // Normal user without Impersonate privilege should be denied.
        let result = ctx.execute_as(RoleId(50), &store, &hierarchy);
        assert!(result.is_err());
    }

    #[test]
    fn test_revert_with_stack() {
        let role = RoleId(1);
        let store = PrivilegeStore::new();
        let grant = GrantEntry {
            grantee: role,
            privilege: PrivilegeType::Impersonate,
            object_type: ObjectType::Type,
            object_id: 50,
            columns: None,
            state: PrivilegeState::Grant,
            with_grant_option: false,
            granted_by: role,
            valid_from: None,
            valid_until: None,
            time_window: None,
            object_pattern: None,
            no_inherit: false,
            mask_function: None,
        };
        store.grant(grant).expect("grant should succeed");
        let mut ctx = SecurityContext::new(
            UserId(1),
            role,
            vec![role],
            vec![role],
            ClassificationLevel::Restricted,
            make_attributes(role),
            Some("10.0.0.1".to_string()),
            QueryLimits::default(),
        );
        let hierarchy = RoleHierarchy::new();
        let original = ctx.current_role;
        ctx.execute_as(RoleId(50), &store, &hierarchy)
            .expect("execute_as should succeed");
        assert_eq!(ctx.current_role, RoleId(50));
        ctx.revert(&hierarchy).expect("revert should succeed");
        assert_eq!(ctx.current_role, original);
    }

    #[test]
    fn test_revert_empty_stack() {
        let mut ctx = make_normal_context();
        let hierarchy = RoleHierarchy::new();
        let result = ctx.revert(&hierarchy);
        assert!(result.is_err());
    }

    #[test]
    fn test_clearance_check() {
        let ctx = make_normal_context();
        let store = ClassificationStore::new();
        store.set_classification(100, 0, ClassificationLevel::Internal);
        store.set_classification(100, 1, ClassificationLevel::Restricted);

        // Internal clearance can access Internal column.
        assert!(ctx.check_clearance(&store, 100, 0));
        // Internal clearance cannot access Restricted column.
        assert!(!ctx.check_clearance(&store, 100, 1));
    }

    #[test]
    fn test_clearance_check_no_label() {
        let ctx = make_normal_context();
        let store = ClassificationStore::new();
        // No label set on column, any clearance passes.
        assert!(ctx.check_clearance(&store, 999, 0));
    }

    #[test]
    fn test_current_user_id() {
        let ctx = make_normal_context();
        assert_eq!(ctx.current_user_id(), UserId(10));
    }

    #[test]
    fn test_current_roles() {
        let ctx = make_normal_context();
        assert_eq!(ctx.current_roles().len(), 2);
        assert!(ctx.current_roles().contains(&RoleId(10)));
        assert!(ctx.current_roles().contains(&RoleId(20)));
    }

    #[test]
    fn test_get_set_attribute() {
        let mut ctx = make_normal_context();
        assert_eq!(ctx.get_attribute("department"), Some("engineering"));
        assert_eq!(ctx.get_attribute("project"), None);

        ctx.set_attribute("project".to_string(), "zyrondb".to_string());
        assert_eq!(ctx.get_attribute("project"), Some("zyrondb"));
    }
}

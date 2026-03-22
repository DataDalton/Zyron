//! Authentication, authorization, and access control for ZyronDB.
//!
//! Provides a comprehensive security subsystem with three-state privileges
//! (GRANT/DENY/unset), temporal access control, data classification labels,
//! data masking, object tagging, row ownership, break-glass emergency access,
//! session binding, ABAC policies, and privilege governance.

pub mod abac;
pub mod auth_rules;
pub mod balloon;
pub mod breakglass;
pub mod brute_force;
pub mod classification;
pub mod context;
pub mod credentials;
pub mod governance;
pub mod heap_storage;
pub mod ip_management;
pub mod masking;
pub mod privilege;
pub mod role;
pub mod row_ownership;
pub mod session_binding;
pub mod storage;
pub mod tagging;
pub mod user;

pub use abac::{AbacPolicy, AbacStore, SessionAttributes};
pub use auth_rules::{AuthMethod, AuthResolver, AuthRule, ConnectionType};
pub use balloon::BalloonParams;
pub use breakglass::{BreakGlassManager, BreakGlassSession};
pub use brute_force::{
    AttemptEntry, AuthGate, BruteForceManager, BruteForcePolicy, BruteForcePolicyBinding,
    LockAction,
};
pub use classification::{ClassificationLevel, ClassificationStore, ColumnClassification};
pub use context::SecurityContext;
pub use credentials::{
    ApiKeyCredential, JwtAlgorithm, JwtClaims, JwtCredential, PasswordCredential, TotpCredential,
};
pub use governance::{
    DelegationEdge, DelegationTracker, GovernanceManager, PendingApproval, PrivilegeAnalytics,
    TwoPersonManager, TwoPersonOperation,
};
pub use heap_storage::HeapAuthStorage;
pub use ip_management::{IpBlockEntry, IpBlockSource, IpManager, TrustedIpEntry};
pub use masking::{MaskFunction, MaskingRule};
pub use privilege::{
    GrantEntry, ObjectType, PrivilegeDecision, PrivilegeState, PrivilegeStore, PrivilegeType,
};
pub use role::{Role, RoleHierarchy, RoleId, RoleMembership, UserId};
pub use row_ownership::{RowOwnershipConfig, RowOwnershipStore};
pub use session_binding::{QueryLimitStore, QueryLimits, SessionBinding, TimeWindow};
pub use tagging::{ObjectTag, TagStore};
pub use user::{User, UserRoleMembership};

use std::sync::Arc;

use zyron_common::Result;

/// Top-level coordinator that owns all auth subsystems.
pub struct SecurityManager {
    pub role_hierarchy: RoleHierarchy,
    pub privilege_store: PrivilegeStore,
    pub classification_store: ClassificationStore,
    pub tag_store: TagStore,
    pub abac_store: AbacStore,
    pub row_ownership_store: RowOwnershipStore,
    pub break_glass: BreakGlassManager,
    pub query_limits: QueryLimitStore,
    pub governance: GovernanceManager,
    pub auth_resolver: AuthResolver,
    pub ip_manager: IpManager,
    pub brute_force: BruteForceManager,
    pub auth_storage: Arc<dyn storage::AuthStorage>,
}

impl SecurityManager {
    /// Creates a new SecurityManager and loads all data from storage.
    pub async fn new(auth_storage: Arc<dyn storage::AuthStorage>) -> Result<Self> {
        let role_hierarchy = RoleHierarchy::new();
        let privilege_store = PrivilegeStore::new();
        let classification_store = ClassificationStore::new();
        let tag_store = TagStore::new();
        let abac_store = AbacStore::new();
        let row_ownership_store = RowOwnershipStore::new();
        let break_glass = BreakGlassManager::new(3600);
        let query_limits = QueryLimitStore::new();
        let governance = GovernanceManager::new();
        let auth_resolver = AuthResolver::new(Vec::new());
        let ip_manager = IpManager::new();
        let brute_force = BruteForceManager::new();

        let mut manager = Self {
            role_hierarchy,
            privilege_store,
            classification_store,
            tag_store,
            abac_store,
            row_ownership_store,
            break_glass,
            query_limits,
            governance,
            auth_resolver,
            ip_manager,
            brute_force,
            auth_storage,
        };

        manager.load_from_storage().await?;
        Ok(manager)
    }

    /// Loads all auth data from persistent storage into in-memory stores.
    async fn load_from_storage(&mut self) -> Result<()> {
        let memberships = self.auth_storage.load_memberships().await?;
        self.role_hierarchy.load(&memberships);

        let grants = self.auth_storage.load_grants().await?;
        self.privilege_store.load(grants);

        let classifications = self.auth_storage.load_classifications().await?;
        self.classification_store.load(classifications);

        let tags = self.auth_storage.load_tags().await?;
        self.tag_store.load(tags);

        let policies = self.auth_storage.load_abac_policies().await?;
        self.abac_store.load(policies);

        let configs = self.auth_storage.load_row_ownership_configs().await?;
        self.row_ownership_store.load(configs);

        let edges = self.auth_storage.load_delegation_edges().await?;
        self.governance.delegation.load(edges);

        let rules = self.auth_storage.load_auth_rules().await?;
        self.auth_resolver = AuthResolver::new(rules);

        let ip_blocks = self.auth_storage.load_ip_blocks().await?;
        self.ip_manager.load_blocks(ip_blocks);
        let trusted_ips = self.auth_storage.load_trusted_ips().await?;
        self.ip_manager.load_trusted(trusted_ips);
        let bf_policies = self.auth_storage.load_brute_force_policies().await?;
        self.brute_force.load_policy_bindings(bf_policies);

        Ok(())
    }

    /// Creates a SecurityContext for an authenticated user with a given role.
    pub fn create_security_context(
        &self,
        user_id: UserId,
        role: &Role,
        ip: &str,
    ) -> SecurityContext {
        let effective_roles = self.role_hierarchy.effective_roles(role.id);
        let all_roles = self.role_hierarchy.all_roles(role.id);
        let query_limits = self.query_limits.get_limits(&effective_roles);

        let attributes = SessionAttributes {
            role_id: role.id,
            department: None,
            region: None,
            clearance: role.clearance,
            ip_address: ip.to_string(),
            connection_time: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            custom: std::collections::HashMap::new(),
        };

        SecurityContext::new(
            user_id,
            role.id,
            effective_roles,
            all_roles,
            role.clearance,
            attributes,
            Some(ip.to_string()),
            query_limits,
        )
    }

    /// Looks up a role by name from storage.
    pub async fn lookup_role(&self, name: &str) -> Result<Option<Role>> {
        let roles = self.auth_storage.load_roles().await?;
        Ok(roles.into_iter().find(|r| r.name == name))
    }

    /// Creates a new role and persists it.
    pub async fn create_role(&self, role: &Role) -> Result<()> {
        self.auth_storage.store_role(role).await?;
        Ok(())
    }

    /// Drops a role and all associated memberships and grants.
    pub async fn drop_role(&self, id: RoleId) -> Result<()> {
        self.auth_storage.delete_role(id).await?;
        Ok(())
    }

    /// Grants a role membership.
    pub async fn grant_role(
        &self,
        member: RoleId,
        parent: RoleId,
        admin_option: bool,
        inherit: bool,
        granted_by: RoleId,
    ) -> Result<()> {
        self.role_hierarchy
            .add_membership(member, parent, inherit)?;
        let membership = RoleMembership {
            member_id: member,
            parent_id: parent,
            admin_option,
            inherit,
            granted_by,
        };
        self.auth_storage.store_membership(&membership).await?;
        Ok(())
    }

    /// Revokes a role membership.
    pub async fn revoke_role(&self, member: RoleId, parent: RoleId) -> Result<()> {
        self.role_hierarchy.remove_membership(member, parent);
        self.auth_storage.delete_membership(member, parent).await?;
        Ok(())
    }

    /// Grants a privilege (stores and updates in-memory).
    pub async fn grant_privilege(&self, entry: GrantEntry) -> Result<()> {
        self.auth_storage.store_grant(&entry).await?;
        let edge = DelegationEdge {
            grantor: entry.granted_by,
            grantee: entry.grantee,
            privilege: entry.privilege,
            object_type: entry.object_type,
            object_id: entry.object_id,
            granted_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };
        self.governance.delegation.record_grant(edge);
        self.privilege_store.grant(entry)?;
        Ok(())
    }

    /// Denies a privilege (stores and updates in-memory).
    pub async fn deny_privilege(&self, mut entry: GrantEntry) -> Result<()> {
        entry.state = PrivilegeState::Deny;
        self.auth_storage.store_grant(&entry).await?;
        self.privilege_store.deny(entry)?;
        Ok(())
    }

    /// Revokes a privilege with optional cascade through delegation chain.
    pub async fn revoke_privilege(
        &self,
        grantee: RoleId,
        privilege: PrivilegeType,
        object_type: ObjectType,
        object_id: u32,
        cascade: bool,
    ) -> Result<()> {
        if cascade {
            let downstream = self.governance.delegation.cascade_revoke(
                grantee,
                privilege,
                object_type,
                object_id,
            );
            for edge in &downstream {
                self.auth_storage
                    .delete_grant(
                        edge.grantee,
                        edge.privilege,
                        edge.object_type,
                        edge.object_id,
                    )
                    .await?;
                self.privilege_store.revoke(
                    edge.grantee,
                    edge.privilege,
                    edge.object_type,
                    edge.object_id,
                );
            }
        }
        self.auth_storage
            .delete_grant(grantee, privilege, object_type, object_id)
            .await?;
        self.privilege_store
            .revoke(grantee, privilege, object_type, object_id);
        Ok(())
    }

    /// Sets a classification level on a column.
    pub async fn set_classification(
        &self,
        table_id: u32,
        column_id: u16,
        level: ClassificationLevel,
    ) -> Result<()> {
        let entry = ColumnClassification {
            table_id,
            column_id,
            level,
        };
        self.auth_storage.store_classification(&entry).await?;
        self.classification_store
            .set_classification(table_id, column_id, level);
        Ok(())
    }

    /// Tags an object.
    pub async fn tag_object(&self, tag: ObjectTag) -> Result<()> {
        self.auth_storage.store_tag(&tag).await?;
        self.tag_store.tag_object(tag)?;
        Ok(())
    }

    /// Sets a masking rule on a column.
    pub async fn set_masking_rule(&self, rule: MaskingRule) -> Result<()> {
        self.auth_storage.store_masking_rule(&rule).await?;
        Ok(())
    }

    /// Enables row ownership on a table.
    pub async fn enable_row_ownership(
        &self,
        table_id: u32,
        config: RowOwnershipConfig,
    ) -> Result<()> {
        self.auth_storage
            .store_row_ownership_config(&config)
            .await?;
        self.row_ownership_store.enable(table_id, config);
        Ok(())
    }

    /// Creates an ABAC policy.
    pub async fn create_abac_policy(&self, policy: AbacPolicy) -> Result<()> {
        self.auth_storage.store_abac_policy(&policy).await?;
        self.abac_store.add_policy(policy)?;
        Ok(())
    }

    /// Activates break-glass emergency access. Rejects locked accounts.
    pub fn activate_break_glass(
        &self,
        role_id: RoleId,
        activated_roles: Vec<RoleId>,
        activated_privileges: Vec<PrivilegeType>,
        elevated_clearance: Option<ClassificationLevel>,
        reason: String,
        duration_secs: u64,
        user_locked: bool,
    ) -> Result<()> {
        if user_locked {
            return Err(zyron_common::ZyronError::AccountLocked(
                "Cannot activate break-glass while account is locked".to_string(),
            ));
        }
        self.break_glass.activate(
            role_id,
            activated_roles,
            activated_privileges,
            elevated_clearance,
            reason,
            duration_secs,
        )
    }

    /// Requests a two-person approval.
    pub fn request_approval(
        &self,
        requester: RoleId,
        operation: TwoPersonOperation,
        details: String,
    ) -> Result<u64> {
        self.governance
            .two_person
            .request_approval(requester, operation, details)
    }

    /// Approves a pending two-person request.
    pub fn approve_request(&self, approval_id: u64, approver: RoleId) -> Result<()> {
        self.governance.two_person.approve(approval_id, approver)
    }
}

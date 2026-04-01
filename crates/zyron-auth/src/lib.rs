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
pub mod cbor;
pub mod classification;
pub mod column_security;
pub mod context;
pub mod credentials;
pub mod crypto_functions;
pub mod encryption;
pub mod governance;
pub mod heap_storage;
pub mod ip_management;
pub mod masking;
pub mod privilege;
pub mod rcu;
pub mod rls;
pub mod role;
pub mod row_ownership;
pub mod security_label;
pub mod session_binding;
pub mod storage;
pub mod tagging;
pub mod user;
pub mod webauthn;
pub mod webauthn_store;
pub mod webhook;

pub use abac::{
    AbacEffect, AbacOperator, AbacPolicy, AbacRule, AbacRuleStore, AbacStore, AttributeCondition,
    SessionAttributes,
};
pub use auth_rules::{AuthMethod, AuthResolver, AuthRule, ConnectionType};
pub use balloon::BalloonParams;
pub use breakglass::{BreakGlassManager, BreakGlassSession};
pub use brute_force::{
    AttemptEntry, AuthGate, BruteForceManager, BruteForcePolicy, BruteForcePolicyBinding,
    LockAction,
};
pub use classification::{ClassificationLevel, ClassificationStore, ColumnClassification};
pub use column_security::{MaskingPolicy, MaskingPolicyStore};
pub use context::SecurityContext;
pub use credentials::{
    ApiKeyCredential, JwtAlgorithm, JwtClaims, JwtCredential, PasswordCredential, TotpCredential,
};
pub use encryption::{
    ColumnEncryption, EncryptionAlgorithm, EncryptionStore, KeyStore, LocalKeyStore,
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
pub use rls::{PolicyType, RlsCommand, RlsPolicy, RlsPolicyStore, RlsResult};
pub use role::{Role, RoleHierarchy, RoleId, RoleMembership, UserId};
pub use row_ownership::{RowOwnershipConfig, RowOwnershipStore};
pub use security_label::{
    MandatoryAccessControl, ObjectSecurityLabel, SecurityLabel, SecurityLevel, SubjectSecurityLabel,
};
pub use session_binding::{QueryLimitStore, QueryLimits, SessionBinding, TimeWindow};
pub use tagging::{ObjectTag, TagStore};
pub use user::{User, UserRoleMembership};
pub use webauthn::{
    CoseAlgorithm, CosePublicKey, CredentialTransport, RelyingPartyConfig, WebAuthnCredential,
};
pub use webauthn_store::WebAuthnCredentialStore;

use std::collections::HashMap;
use std::sync::Arc;

use crate::rcu::RcuMap;
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
    pub rls_store: RlsPolicyStore,
    pub masking_policy_store: MaskingPolicyStore,
    pub abac_rule_store: AbacRuleStore,
    pub encryption_store: EncryptionStore,
    pub mac: MandatoryAccessControl,
    pub webauthn_store: WebAuthnCredentialStore,
    /// Cached password hashes keyed by username. Loaded at startup and updated
    /// on user create/alter. Avoids heap scans per connection.
    pub password_cache: RcuMap<String, String>,
    /// Cached user ID lookup by name.
    pub user_id_cache: RcuMap<String, UserId>,
    /// Cached role lookup by name. Populated at startup, updated on create/drop.
    pub role_cache: RcuMap<String, Role>,
    /// Cached TOTP secrets keyed by username for TOTP authentication.
    pub totp_secret_cache: RcuMap<String, Vec<u8>>,
    /// Cached API key credentials keyed by username: (prefix, hash).
    pub api_key_cache: RcuMap<String, (String, Vec<u8>)>,
    /// Server-wide JWT signing secret for JWT authentication.
    pub jwt_secret: Option<Vec<u8>>,
    /// JWT signing algorithm (default HS256).
    pub jwt_algorithm: JwtAlgorithm,
    /// Expected JWT issuer claim. If set, tokens must match.
    pub jwt_issuer: Option<String>,
    /// WebAuthn relying party configuration for FIDO2 authentication.
    pub webauthn_rp_config: RelyingPartyConfig,
    pub auth_storage: Arc<dyn storage::AuthStorage>,
    /// Monotonic counter for generating unique role IDs. Initialized from
    /// the max existing role ID at startup to survive restarts.
    next_role_id: std::sync::atomic::AtomicU32,
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
        let rls_store = RlsPolicyStore::new();
        let masking_policy_store = MaskingPolicyStore::new();
        let abac_rule_store = AbacRuleStore::new();
        let encryption_store = EncryptionStore::new();
        let mac = MandatoryAccessControl::new();
        let webauthn_store = WebAuthnCredentialStore::new();
        let password_cache: RcuMap<String, String> = RcuMap::empty_map();
        let user_id_cache: RcuMap<String, UserId> = RcuMap::empty_map();
        let role_cache: RcuMap<String, Role> = RcuMap::empty_map();
        let totp_secret_cache: RcuMap<String, Vec<u8>> = RcuMap::empty_map();
        let api_key_cache: RcuMap<String, (String, Vec<u8>)> = RcuMap::empty_map();
        let webauthn_rp_config = RelyingPartyConfig::default();

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
            rls_store,
            masking_policy_store,
            abac_rule_store,
            encryption_store,
            mac,
            webauthn_store,
            password_cache,
            user_id_cache,
            role_cache,
            totp_secret_cache,
            api_key_cache,
            jwt_secret: None,
            jwt_algorithm: JwtAlgorithm::Hs256,
            jwt_issuer: None,
            webauthn_rp_config,
            auth_storage,
            next_role_id: std::sync::atomic::AtomicU32::new(1),
        };

        manager.load_from_storage().await?;

        // Initialize the role ID counter from the max existing role ID
        let max_id = manager
            .role_cache
            .load()
            .values()
            .map(|r| r.id.0)
            .max()
            .unwrap_or(0);
        manager
            .next_role_id
            .store(max_id + 1, std::sync::atomic::Ordering::Relaxed);

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

        let rls_policies = self.auth_storage.load_rls_policies().await?;
        self.rls_store.load(rls_policies);

        let masking_policies = self.auth_storage.load_masking_policies().await?;
        self.masking_policy_store.load(masking_policies);

        let abac_rules = self.auth_storage.load_abac_rules().await?;
        self.abac_rule_store.load(abac_rules);

        let col_encryptions = self.auth_storage.load_column_encryptions().await?;
        self.encryption_store.load(col_encryptions);

        let object_labels = self.auth_storage.load_object_security_labels().await?;
        self.mac.load_object_labels(object_labels);

        let subject_labels = self.auth_storage.load_subject_security_labels().await?;
        self.mac.load_subject_labels(subject_labels);

        let webauthn_creds = self.auth_storage.load_webauthn_credentials().await?;
        self.webauthn_store.load(webauthn_creds);

        // Cache user passwords, IDs, TOTP secrets, and API keys for fast auth lookups
        let users = self.auth_storage.load_users().await?;
        let mut id_map = HashMap::with_capacity(users.len());
        let mut pw_map = HashMap::with_capacity(users.len());
        let mut totp_map = HashMap::new();
        let mut api_key_map = HashMap::new();
        for user in &users {
            id_map.insert(user.name.clone(), user.id);
            if let Some(ref hash) = user.password_hash {
                pw_map.insert(user.name.clone(), hash.clone());
            }
            if let Some(ref secret) = user.totp_secret {
                totp_map.insert(user.name.clone(), secret.clone());
            }
            if let (Some(prefix), Some(hash)) = (&user.api_key_prefix, &user.api_key_hash) {
                api_key_map.insert(user.name.clone(), (prefix.clone(), hash.clone()));
            }
        }
        self.user_id_cache.store(id_map);
        self.password_cache.store(pw_map);
        self.totp_secret_cache.store(totp_map);
        self.api_key_cache.store(api_key_map);

        // Cache roles by name for O(1) lookup
        let roles = self.auth_storage.load_roles().await?;
        let mut role_map = HashMap::with_capacity(roles.len());
        for role in roles {
            role_map.insert(role.name.clone(), role);
        }
        self.role_cache.store(role_map);

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
    pub fn lookup_role(&self, name: &str) -> Option<Role> {
        self.role_cache.get(&name.to_string())
    }

    /// Allocates a unique monotonic role ID.
    pub fn allocate_role_id(&self) -> RoleId {
        RoleId(
            self.next_role_id
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        )
    }

    /// Creates a new role, persists it, and updates the in-memory cache.
    /// If role.id is RoleId(0), a new unique ID is allocated automatically.
    pub async fn create_role(&self, role: &Role) -> Result<()> {
        let mut role = role.clone();
        if role.id.0 == 0 {
            role.id = self.allocate_role_id();
        }
        self.auth_storage.store_role(&role).await?;
        self.role_cache.insert(role.name.clone(), role);
        Ok(())
    }

    /// Drops a role, removes it from storage and the in-memory cache.
    pub async fn drop_role(&self, id: RoleId) -> Result<()> {
        // Find the role name by scanning the cache snapshot
        let snap = self.role_cache.load();
        let name = snap
            .iter()
            .find_map(|(k, v)| if v.id == id { Some(k.clone()) } else { None });
        self.auth_storage.delete_role(id).await?;
        if let Some(name) = name {
            self.role_cache.remove(&name);
        }
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
        // Update in-memory masking policy store so the rule takes effect immediately
        let policy = column_security::MaskingPolicy {
            id: 0,
            name: format!("rule_{}_{}", rule.table_id, rule.column_id),
            table_id: rule.table_id,
            column_id: rule.column_id,
            function: rule.function.clone(),
            exempt_roles: rule.role_id.map_or_else(Vec::new, |r| vec![r]),
            enabled: true,
        };
        let _ = self.masking_policy_store.add_policy(policy);
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

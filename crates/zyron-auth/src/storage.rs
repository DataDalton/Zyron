//! Storage trait for persisting auth data.
//!
//! AuthStorage defines the async interface for loading and storing
//! roles, memberships, grants, classifications, tags, policies,
//! row ownership configs, delegation edges, masking rules, and auth rules.

use async_trait::async_trait;
use zyron_common::Result;

use crate::abac::{AbacPolicy, AbacRule};
use crate::auth_rules::AuthRule;
use crate::classification::ColumnClassification;
use crate::column_security::MaskingPolicy;
use crate::encryption::ColumnEncryption;
use crate::governance::DelegationEdge;
use crate::masking::MaskingRule;
use crate::privilege::{GrantEntry, ObjectType, PrivilegeType};
use crate::rls::RlsPolicy;
use crate::role::{Role, RoleId, RoleMembership};
use crate::row_ownership::RowOwnershipConfig;
use crate::security_label::{ObjectSecurityLabel, SubjectSecurityLabel};
use crate::tagging::ObjectTag;
use crate::webauthn::WebAuthnCredential;

/// Async persistence interface for all auth subsystem data.
#[async_trait]
pub trait AuthStorage: Send + Sync {
    async fn load_roles(&self) -> Result<Vec<Role>>;
    async fn store_role(&self, role: &Role) -> Result<()>;
    async fn delete_role(&self, id: RoleId) -> Result<()>;

    async fn load_memberships(&self) -> Result<Vec<RoleMembership>>;
    async fn store_membership(&self, membership: &RoleMembership) -> Result<()>;
    async fn delete_membership(&self, member: RoleId, parent: RoleId) -> Result<()>;

    async fn load_grants(&self) -> Result<Vec<GrantEntry>>;
    async fn store_grant(&self, entry: &GrantEntry) -> Result<()>;
    async fn delete_grant(
        &self,
        grantee: RoleId,
        privilege: PrivilegeType,
        object_type: ObjectType,
        object_id: u32,
    ) -> Result<()>;

    async fn load_classifications(&self) -> Result<Vec<ColumnClassification>>;
    async fn store_classification(&self, entry: &ColumnClassification) -> Result<()>;

    async fn load_tags(&self) -> Result<Vec<ObjectTag>>;
    async fn store_tag(&self, tag: &ObjectTag) -> Result<()>;

    async fn load_abac_policies(&self) -> Result<Vec<AbacPolicy>>;
    async fn store_abac_policy(&self, policy: &AbacPolicy) -> Result<()>;

    async fn load_row_ownership_configs(&self) -> Result<Vec<RowOwnershipConfig>>;
    async fn store_row_ownership_config(&self, config: &RowOwnershipConfig) -> Result<()>;

    async fn load_delegation_edges(&self) -> Result<Vec<DelegationEdge>>;

    async fn store_masking_rule(&self, rule: &MaskingRule) -> Result<()>;

    async fn load_auth_rules(&self) -> Result<Vec<AuthRule>>;

    // User storage
    async fn load_users(&self) -> Result<Vec<crate::user::User>>;
    async fn store_user(&self, user: &crate::user::User) -> Result<()>;
    async fn delete_user(&self, id: crate::role::UserId) -> Result<()>;

    // User-role membership
    async fn load_user_memberships(&self) -> Result<Vec<crate::user::UserRoleMembership>>;
    async fn store_user_membership(&self, m: &crate::user::UserRoleMembership) -> Result<()>;
    async fn delete_user_membership(
        &self,
        user_id: crate::role::UserId,
        role_id: crate::role::RoleId,
    ) -> Result<()>;

    // IP management
    async fn load_ip_blocks(&self) -> Result<Vec<crate::ip_management::IpBlockEntry>>;
    async fn store_ip_block(&self, entry: &crate::ip_management::IpBlockEntry) -> Result<()>;
    async fn delete_ip_block(&self, ip: &str) -> Result<()>;
    async fn load_trusted_ips(&self) -> Result<Vec<crate::ip_management::TrustedIpEntry>>;
    async fn store_trusted_ip(&self, entry: &crate::ip_management::TrustedIpEntry) -> Result<()>;
    async fn delete_trusted_ip(&self, ip: &str) -> Result<()>;

    // Brute force
    async fn store_attempt(&self, entry: &crate::brute_force::AttemptEntry) -> Result<()>;
    async fn load_recent_attempts(
        &self,
        limit: usize,
    ) -> Result<Vec<crate::brute_force::AttemptEntry>>;
    async fn load_brute_force_policies(
        &self,
    ) -> Result<Vec<crate::brute_force::BruteForcePolicyBinding>>;
    async fn store_brute_force_policy(
        &self,
        binding: &crate::brute_force::BruteForcePolicyBinding,
    ) -> Result<()>;
    async fn delete_brute_force_policy(
        &self,
        role_id: Option<crate::role::RoleId>,
        database: Option<&str>,
    ) -> Result<()>;

    // RLS policies
    async fn load_rls_policies(&self) -> Result<Vec<RlsPolicy>>;
    async fn store_rls_policy(&self, policy: &RlsPolicy) -> Result<()>;
    async fn delete_rls_policy(&self, table_id: u32, name: &str) -> Result<()>;

    // Masking policies (column_security)
    async fn load_masking_policies(&self) -> Result<Vec<MaskingPolicy>>;
    async fn store_masking_policy(&self, policy: &MaskingPolicy) -> Result<()>;
    async fn delete_masking_policy(&self, table_id: u32, column_id: u16, name: &str) -> Result<()>;

    // ABAC rules
    async fn load_abac_rules(&self) -> Result<Vec<AbacRule>>;
    async fn store_abac_rule(&self, rule: &AbacRule) -> Result<()>;
    async fn delete_abac_rule(&self, id: u32) -> Result<()>;

    // Column encryption
    async fn load_column_encryptions(&self) -> Result<Vec<ColumnEncryption>>;
    async fn store_column_encryption(&self, config: &ColumnEncryption) -> Result<()>;
    async fn delete_column_encryption(&self, table_id: u32, column_id: u16) -> Result<()>;

    // Security labels
    async fn load_object_security_labels(&self) -> Result<Vec<ObjectSecurityLabel>>;
    async fn store_object_security_label(&self, label: &ObjectSecurityLabel) -> Result<()>;
    async fn load_subject_security_labels(&self) -> Result<Vec<SubjectSecurityLabel>>;
    async fn store_subject_security_label(&self, label: &SubjectSecurityLabel) -> Result<()>;

    // WebAuthn credentials
    async fn load_webauthn_credentials(&self) -> Result<Vec<WebAuthnCredential>>;
    async fn store_webauthn_credential(&self, cred: &WebAuthnCredential) -> Result<()>;
    async fn delete_webauthn_credential(&self, credential_id: &[u8]) -> Result<()>;
    async fn update_webauthn_sign_count(&self, credential_id: &[u8], new_count: u32) -> Result<()>;
}

//! In-memory placeholder implementation of AuthStorage.
//!
//! Returns empty collections for all load operations and no-ops for all store
//! operations. This allows SecurityManager to initialize without a persistent
//! catalog-backed storage layer.

use async_trait::async_trait;
use zyron_common::Result;

use zyron_auth::abac::AbacPolicy;
use zyron_auth::auth_rules::AuthRule;
use zyron_auth::brute_force::{AttemptEntry, BruteForcePolicyBinding};
use zyron_auth::classification::ColumnClassification;
use zyron_auth::governance::DelegationEdge;
use zyron_auth::ip_management::{IpBlockEntry, TrustedIpEntry};
use zyron_auth::masking::MaskingRule;
use zyron_auth::privilege::{GrantEntry, ObjectType, PrivilegeType};
use zyron_auth::role::{Role, RoleId, RoleMembership, UserId};
use zyron_auth::row_ownership::RowOwnershipConfig;
use zyron_auth::storage::AuthStorage;
use zyron_auth::tagging::ObjectTag;
use zyron_auth::user::{User, UserRoleMembership};

/// In-memory auth storage that returns empty results for all loads
/// and accepts (but discards) all stores. Placeholder until the
/// catalog-backed AuthStorage implementation is built.
pub struct InMemoryAuthStorage;

impl InMemoryAuthStorage {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl AuthStorage for InMemoryAuthStorage {
    async fn load_roles(&self) -> Result<Vec<Role>> {
        Ok(Vec::new())
    }
    async fn store_role(&self, _role: &Role) -> Result<()> {
        Ok(())
    }
    async fn delete_role(&self, _id: RoleId) -> Result<()> {
        Ok(())
    }

    async fn load_memberships(&self) -> Result<Vec<RoleMembership>> {
        Ok(Vec::new())
    }
    async fn store_membership(&self, _membership: &RoleMembership) -> Result<()> {
        Ok(())
    }
    async fn delete_membership(&self, _member: RoleId, _parent: RoleId) -> Result<()> {
        Ok(())
    }

    async fn load_grants(&self) -> Result<Vec<GrantEntry>> {
        Ok(Vec::new())
    }
    async fn store_grant(&self, _entry: &GrantEntry) -> Result<()> {
        Ok(())
    }
    async fn delete_grant(
        &self,
        _grantee: RoleId,
        _privilege: PrivilegeType,
        _object_type: ObjectType,
        _object_id: u32,
    ) -> Result<()> {
        Ok(())
    }

    async fn load_classifications(&self) -> Result<Vec<ColumnClassification>> {
        Ok(Vec::new())
    }
    async fn store_classification(&self, _entry: &ColumnClassification) -> Result<()> {
        Ok(())
    }

    async fn load_tags(&self) -> Result<Vec<ObjectTag>> {
        Ok(Vec::new())
    }
    async fn store_tag(&self, _tag: &ObjectTag) -> Result<()> {
        Ok(())
    }

    async fn load_abac_policies(&self) -> Result<Vec<AbacPolicy>> {
        Ok(Vec::new())
    }
    async fn store_abac_policy(&self, _policy: &AbacPolicy) -> Result<()> {
        Ok(())
    }

    async fn load_row_ownership_configs(&self) -> Result<Vec<RowOwnershipConfig>> {
        Ok(Vec::new())
    }
    async fn store_row_ownership_config(&self, _config: &RowOwnershipConfig) -> Result<()> {
        Ok(())
    }

    async fn load_delegation_edges(&self) -> Result<Vec<DelegationEdge>> {
        Ok(Vec::new())
    }

    async fn store_masking_rule(&self, _rule: &MaskingRule) -> Result<()> {
        Ok(())
    }

    async fn load_auth_rules(&self) -> Result<Vec<AuthRule>> {
        Ok(Vec::new())
    }

    async fn load_users(&self) -> Result<Vec<User>> {
        Ok(Vec::new())
    }
    async fn store_user(&self, _user: &User) -> Result<()> {
        Ok(())
    }
    async fn delete_user(&self, _id: UserId) -> Result<()> {
        Ok(())
    }

    async fn load_user_memberships(&self) -> Result<Vec<UserRoleMembership>> {
        Ok(Vec::new())
    }
    async fn store_user_membership(&self, _m: &UserRoleMembership) -> Result<()> {
        Ok(())
    }
    async fn delete_user_membership(&self, _user_id: UserId, _role_id: RoleId) -> Result<()> {
        Ok(())
    }

    async fn load_ip_blocks(&self) -> Result<Vec<IpBlockEntry>> {
        Ok(Vec::new())
    }
    async fn store_ip_block(&self, _entry: &IpBlockEntry) -> Result<()> {
        Ok(())
    }
    async fn delete_ip_block(&self, _ip: &str) -> Result<()> {
        Ok(())
    }
    async fn load_trusted_ips(&self) -> Result<Vec<TrustedIpEntry>> {
        Ok(Vec::new())
    }
    async fn store_trusted_ip(&self, _entry: &TrustedIpEntry) -> Result<()> {
        Ok(())
    }
    async fn delete_trusted_ip(&self, _ip: &str) -> Result<()> {
        Ok(())
    }

    async fn store_attempt(&self, _entry: &AttemptEntry) -> Result<()> {
        Ok(())
    }
    async fn load_recent_attempts(&self, _limit: usize) -> Result<Vec<AttemptEntry>> {
        Ok(Vec::new())
    }
    async fn load_brute_force_policies(&self) -> Result<Vec<BruteForcePolicyBinding>> {
        Ok(Vec::new())
    }
    async fn store_brute_force_policy(&self, _binding: &BruteForcePolicyBinding) -> Result<()> {
        Ok(())
    }
    async fn delete_brute_force_policy(
        &self,
        _role_id: Option<RoleId>,
        _database: Option<&str>,
    ) -> Result<()> {
        Ok(())
    }
}

//! Heap-backed AuthStorage implementation.
//!
//! Persists all auth data to disk through the buffer pool using dedicated
//! HeapFile instances for each system table. Follows the same pattern as
//! HeapCatalogStorage in zyron-catalog.

use std::sync::Arc;

use async_trait::async_trait;

use zyron_buffer::BufferPool;
use zyron_common::Result;
use zyron_storage::{DiskManager, HeapFile, HeapFileConfig, Tuple};

use crate::abac::{AbacPolicy, AbacRule};
use crate::auth_rules::AuthRule;
use crate::brute_force::{AttemptEntry, BruteForcePolicyBinding};
use crate::classification::ColumnClassification;
use crate::column_security::MaskingPolicy;
use crate::encryption::ColumnEncryption;
use crate::governance::DelegationEdge;
use crate::ip_management::{IpBlockEntry, TrustedIpEntry};
use crate::masking::MaskingRule;
use crate::privilege::{GrantEntry, ObjectType, PrivilegeType};
use crate::rls::RlsPolicy;
use crate::role::{Role, RoleId, RoleMembership, UserId};
use crate::row_ownership::RowOwnershipConfig;
use crate::security_label::{ObjectSecurityLabel, SubjectSecurityLabel};
use crate::storage::AuthStorage;
use crate::tagging::ObjectTag;
use crate::user::{User, UserRoleMembership};
use crate::webauthn::WebAuthnCredential;

// System table file ID assignments (reserved range 110-159 for auth).
const ROLES_HEAP_FILE_ID: u32 = 110;
const ROLES_FSM_FILE_ID: u32 = 111;
const MEMBERSHIPS_HEAP_FILE_ID: u32 = 112;
const MEMBERSHIPS_FSM_FILE_ID: u32 = 113;
const PRIVILEGES_HEAP_FILE_ID: u32 = 114;
const PRIVILEGES_FSM_FILE_ID: u32 = 115;
const AUTH_RULES_HEAP_FILE_ID: u32 = 116;
const AUTH_RULES_FSM_FILE_ID: u32 = 117;
const CLASSIFICATIONS_HEAP_FILE_ID: u32 = 118;
const CLASSIFICATIONS_FSM_FILE_ID: u32 = 119;
const ABAC_POLICIES_HEAP_FILE_ID: u32 = 120;
const ABAC_POLICIES_FSM_FILE_ID: u32 = 121;
const OBJECT_TAGS_HEAP_FILE_ID: u32 = 122;
const OBJECT_TAGS_FSM_FILE_ID: u32 = 123;
const MASKING_RULES_HEAP_FILE_ID: u32 = 124;
const MASKING_RULES_FSM_FILE_ID: u32 = 125;
const ROW_OWNERSHIP_HEAP_FILE_ID: u32 = 126;
const ROW_OWNERSHIP_FSM_FILE_ID: u32 = 127;
const DELEGATION_EDGES_HEAP_FILE_ID: u32 = 130;
const DELEGATION_EDGES_FSM_FILE_ID: u32 = 131;
const IP_BLOCKS_HEAP_FILE_ID: u32 = 134;
const IP_BLOCKS_FSM_FILE_ID: u32 = 135;
const AUTH_ATTEMPTS_HEAP_FILE_ID: u32 = 136;
const AUTH_ATTEMPTS_FSM_FILE_ID: u32 = 137;
const BRUTE_FORCE_POLICIES_HEAP_FILE_ID: u32 = 138;
const BRUTE_FORCE_POLICIES_FSM_FILE_ID: u32 = 139;
const TRUSTED_IPS_HEAP_FILE_ID: u32 = 140;
const TRUSTED_IPS_FSM_FILE_ID: u32 = 141;
const USERS_HEAP_FILE_ID: u32 = 142;
const USERS_FSM_FILE_ID: u32 = 143;
const USER_MEMBERSHIPS_HEAP_FILE_ID: u32 = 144;
const USER_MEMBERSHIPS_FSM_FILE_ID: u32 = 145;
const RLS_POLICIES_HEAP_FILE_ID: u32 = 146;
const RLS_POLICIES_FSM_FILE_ID: u32 = 147;
const MASKING_POLICIES_HEAP_FILE_ID: u32 = 148;
const MASKING_POLICIES_FSM_FILE_ID: u32 = 149;
const ABAC_RULES_HEAP_FILE_ID: u32 = 150;
const ABAC_RULES_FSM_FILE_ID: u32 = 151;
const COLUMN_ENCRYPTION_HEAP_FILE_ID: u32 = 152;
const COLUMN_ENCRYPTION_FSM_FILE_ID: u32 = 153;
const SECURITY_LABELS_HEAP_FILE_ID: u32 = 154;
const SECURITY_LABELS_FSM_FILE_ID: u32 = 155;
const SUBJECT_LABELS_HEAP_FILE_ID: u32 = 156;
const SUBJECT_LABELS_FSM_FILE_ID: u32 = 157;
const WEBAUTHN_CREDENTIALS_HEAP_FILE_ID: u32 = 158;
const WEBAUTHN_CREDENTIALS_FSM_FILE_ID: u32 = 159;

/// Auth storage backed by heap files (self-hosting).
/// Each auth entity type has its own heap file (system table).
pub struct HeapAuthStorage {
    roles_heap: HeapFile,
    memberships_heap: HeapFile,
    privileges_heap: HeapFile,
    auth_rules_heap: HeapFile,
    classifications_heap: HeapFile,
    abac_policies_heap: HeapFile,
    object_tags_heap: HeapFile,
    masking_rules_heap: HeapFile,
    row_ownership_heap: HeapFile,
    delegation_edges_heap: HeapFile,
    users_heap: HeapFile,
    user_memberships_heap: HeapFile,
    ip_blocks_heap: HeapFile,
    trusted_ips_heap: HeapFile,
    auth_attempts_heap: HeapFile,
    brute_force_policies_heap: HeapFile,
    rls_policies_heap: HeapFile,
    masking_policies_heap: HeapFile,
    abac_rules_heap: HeapFile,
    column_encryption_heap: HeapFile,
    security_labels_heap: HeapFile,
    subject_labels_heap: HeapFile,
    webauthn_credentials_heap: HeapFile,
}

impl HeapAuthStorage {
    /// Creates a new HeapAuthStorage with system table heap files.
    pub fn new(disk: Arc<DiskManager>, pool: Arc<BufferPool>) -> Result<Self> {
        let roles_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: ROLES_HEAP_FILE_ID,
                fsm_file_id: ROLES_FSM_FILE_ID,
            },
        )?;
        let memberships_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: MEMBERSHIPS_HEAP_FILE_ID,
                fsm_file_id: MEMBERSHIPS_FSM_FILE_ID,
            },
        )?;
        let privileges_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: PRIVILEGES_HEAP_FILE_ID,
                fsm_file_id: PRIVILEGES_FSM_FILE_ID,
            },
        )?;
        let auth_rules_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: AUTH_RULES_HEAP_FILE_ID,
                fsm_file_id: AUTH_RULES_FSM_FILE_ID,
            },
        )?;
        let classifications_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: CLASSIFICATIONS_HEAP_FILE_ID,
                fsm_file_id: CLASSIFICATIONS_FSM_FILE_ID,
            },
        )?;
        let abac_policies_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: ABAC_POLICIES_HEAP_FILE_ID,
                fsm_file_id: ABAC_POLICIES_FSM_FILE_ID,
            },
        )?;
        let object_tags_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: OBJECT_TAGS_HEAP_FILE_ID,
                fsm_file_id: OBJECT_TAGS_FSM_FILE_ID,
            },
        )?;
        let masking_rules_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: MASKING_RULES_HEAP_FILE_ID,
                fsm_file_id: MASKING_RULES_FSM_FILE_ID,
            },
        )?;
        let row_ownership_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: ROW_OWNERSHIP_HEAP_FILE_ID,
                fsm_file_id: ROW_OWNERSHIP_FSM_FILE_ID,
            },
        )?;
        let delegation_edges_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: DELEGATION_EDGES_HEAP_FILE_ID,
                fsm_file_id: DELEGATION_EDGES_FSM_FILE_ID,
            },
        )?;
        let users_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: USERS_HEAP_FILE_ID,
                fsm_file_id: USERS_FSM_FILE_ID,
            },
        )?;
        let user_memberships_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: USER_MEMBERSHIPS_HEAP_FILE_ID,
                fsm_file_id: USER_MEMBERSHIPS_FSM_FILE_ID,
            },
        )?;
        let ip_blocks_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: IP_BLOCKS_HEAP_FILE_ID,
                fsm_file_id: IP_BLOCKS_FSM_FILE_ID,
            },
        )?;
        let trusted_ips_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: TRUSTED_IPS_HEAP_FILE_ID,
                fsm_file_id: TRUSTED_IPS_FSM_FILE_ID,
            },
        )?;
        let auth_attempts_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: AUTH_ATTEMPTS_HEAP_FILE_ID,
                fsm_file_id: AUTH_ATTEMPTS_FSM_FILE_ID,
            },
        )?;
        let brute_force_policies_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: BRUTE_FORCE_POLICIES_HEAP_FILE_ID,
                fsm_file_id: BRUTE_FORCE_POLICIES_FSM_FILE_ID,
            },
        )?;

        let rls_policies_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: RLS_POLICIES_HEAP_FILE_ID,
                fsm_file_id: RLS_POLICIES_FSM_FILE_ID,
            },
        )?;
        let masking_policies_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: MASKING_POLICIES_HEAP_FILE_ID,
                fsm_file_id: MASKING_POLICIES_FSM_FILE_ID,
            },
        )?;
        let abac_rules_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: ABAC_RULES_HEAP_FILE_ID,
                fsm_file_id: ABAC_RULES_FSM_FILE_ID,
            },
        )?;
        let column_encryption_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: COLUMN_ENCRYPTION_HEAP_FILE_ID,
                fsm_file_id: COLUMN_ENCRYPTION_FSM_FILE_ID,
            },
        )?;
        let security_labels_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: SECURITY_LABELS_HEAP_FILE_ID,
                fsm_file_id: SECURITY_LABELS_FSM_FILE_ID,
            },
        )?;
        let subject_labels_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: SUBJECT_LABELS_HEAP_FILE_ID,
                fsm_file_id: SUBJECT_LABELS_FSM_FILE_ID,
            },
        )?;

        Ok(Self {
            roles_heap,
            memberships_heap,
            privileges_heap,
            auth_rules_heap,
            classifications_heap,
            abac_policies_heap,
            object_tags_heap,
            masking_rules_heap,
            row_ownership_heap,
            delegation_edges_heap,
            users_heap,
            user_memberships_heap,
            ip_blocks_heap,
            trusted_ips_heap,
            auth_attempts_heap,
            brute_force_policies_heap,
            rls_policies_heap,
            masking_policies_heap,
            abac_rules_heap,
            column_encryption_heap,
            security_labels_heap,
            subject_labels_heap,
            webauthn_credentials_heap: HeapFile::new(
                Arc::clone(&disk),
                Arc::clone(&pool),
                HeapFileConfig {
                    heap_file_id: WEBAUTHN_CREDENTIALS_HEAP_FILE_ID,
                    fsm_file_id: WEBAUTHN_CREDENTIALS_FSM_FILE_ID,
                },
            )?,
        })
    }

    /// Initializes page count caches for all auth system table heap files.
    pub async fn init_cache(&self) -> Result<()> {
        tokio::try_join!(
            self.roles_heap.init_cache(),
            self.memberships_heap.init_cache(),
            self.privileges_heap.init_cache(),
            self.auth_rules_heap.init_cache(),
            self.classifications_heap.init_cache(),
            self.abac_policies_heap.init_cache(),
            self.object_tags_heap.init_cache(),
            self.masking_rules_heap.init_cache(),
            self.row_ownership_heap.init_cache(),
            self.delegation_edges_heap.init_cache(),
            self.users_heap.init_cache(),
            self.user_memberships_heap.init_cache(),
            self.ip_blocks_heap.init_cache(),
            self.trusted_ips_heap.init_cache(),
            self.auth_attempts_heap.init_cache(),
            self.brute_force_policies_heap.init_cache(),
        )?;
        tokio::try_join!(
            self.rls_policies_heap.init_cache(),
            self.masking_policies_heap.init_cache(),
            self.abac_rules_heap.init_cache(),
            self.column_encryption_heap.init_cache(),
            self.security_labels_heap.init_cache(),
            self.subject_labels_heap.init_cache(),
        )?;
        self.webauthn_credentials_heap.init_cache().await?;
        Ok(())
    }
}

#[async_trait]
impl AuthStorage for HeapAuthStorage {
    // -----------------------------------------------------------------------
    // Roles
    // -----------------------------------------------------------------------

    async fn load_roles(&self) -> Result<Vec<Role>> {
        let mut entries = Vec::new();
        let guard = self.roles_heap.scan()?;
        guard.for_each(|_tid, view| {
            if let Ok(entry) = Role::from_bytes(view.data) {
                entries.push(entry);
            }
        });
        Ok(entries)
    }

    async fn store_role(&self, role: &Role) -> Result<()> {
        let bytes = role.to_bytes();
        let tuple = Tuple::new(bytes, 0);
        let _ = self.roles_heap.insert_batch(&[tuple]).await?;
        Ok(())
    }

    async fn delete_role(&self, id: RoleId) -> Result<()> {
        let mut target = None;
        let guard = self.roles_heap.scan()?;
        guard.for_each(|tid, view| {
            if let Ok(entry) = Role::from_bytes(view.data) {
                if entry.id == id {
                    target = Some(tid);
                }
            }
        });
        if let Some(tid) = target {
            self.roles_heap.delete(tid).await?;
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Role memberships
    // -----------------------------------------------------------------------

    async fn load_memberships(&self) -> Result<Vec<RoleMembership>> {
        let mut entries = Vec::new();
        let guard = self.memberships_heap.scan()?;
        guard.for_each(|_tid, view| {
            if let Ok(entry) = RoleMembership::from_bytes(view.data) {
                entries.push(entry);
            }
        });
        Ok(entries)
    }

    async fn store_membership(&self, membership: &RoleMembership) -> Result<()> {
        let bytes = membership.to_bytes();
        let tuple = Tuple::new(bytes, 0);
        let _ = self.memberships_heap.insert_batch(&[tuple]).await?;
        Ok(())
    }

    async fn delete_membership(&self, member: RoleId, parent: RoleId) -> Result<()> {
        let mut target = None;
        let guard = self.memberships_heap.scan()?;
        guard.for_each(|tid, view| {
            if let Ok(entry) = RoleMembership::from_bytes(view.data) {
                if entry.member_id == member && entry.parent_id == parent {
                    target = Some(tid);
                }
            }
        });
        if let Some(tid) = target {
            self.memberships_heap.delete(tid).await?;
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Grants (privileges)
    // -----------------------------------------------------------------------

    async fn load_grants(&self) -> Result<Vec<GrantEntry>> {
        let mut entries = Vec::new();
        let guard = self.privileges_heap.scan()?;
        guard.for_each(|_tid, view| {
            if let Ok(entry) = GrantEntry::from_bytes(view.data) {
                entries.push(entry);
            }
        });
        Ok(entries)
    }

    async fn store_grant(&self, entry: &GrantEntry) -> Result<()> {
        let bytes = entry.to_bytes();
        let tuple = Tuple::new(bytes, 0);
        let _ = self.privileges_heap.insert_batch(&[tuple]).await?;
        Ok(())
    }

    async fn delete_grant(
        &self,
        grantee: RoleId,
        privilege: PrivilegeType,
        object_type: ObjectType,
        object_id: u32,
    ) -> Result<()> {
        let mut target = None;
        let guard = self.privileges_heap.scan()?;
        guard.for_each(|tid, view| {
            if let Ok(entry) = GrantEntry::from_bytes(view.data) {
                if entry.grantee == grantee
                    && entry.privilege == privilege
                    && entry.object_type == object_type
                    && entry.object_id == object_id
                {
                    target = Some(tid);
                }
            }
        });
        if let Some(tid) = target {
            self.privileges_heap.delete(tid).await?;
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Classifications
    // -----------------------------------------------------------------------

    async fn load_classifications(&self) -> Result<Vec<ColumnClassification>> {
        let mut entries = Vec::new();
        let guard = self.classifications_heap.scan()?;
        guard.for_each(|_tid, view| {
            if let Ok(entry) = ColumnClassification::from_bytes(view.data) {
                entries.push(entry);
            }
        });
        Ok(entries)
    }

    async fn store_classification(&self, entry: &ColumnClassification) -> Result<()> {
        let bytes = entry.to_bytes();
        let tuple = Tuple::new(bytes.to_vec(), 0);
        let _ = self.classifications_heap.insert_batch(&[tuple]).await?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Tags
    // -----------------------------------------------------------------------

    async fn load_tags(&self) -> Result<Vec<ObjectTag>> {
        let mut entries = Vec::new();
        let guard = self.object_tags_heap.scan()?;
        guard.for_each(|_tid, view| {
            if let Ok(entry) = ObjectTag::from_bytes(view.data) {
                entries.push(entry);
            }
        });
        Ok(entries)
    }

    async fn store_tag(&self, tag: &ObjectTag) -> Result<()> {
        let bytes = tag.to_bytes();
        let tuple = Tuple::new(bytes, 0);
        let _ = self.object_tags_heap.insert_batch(&[tuple]).await?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // ABAC policies
    // -----------------------------------------------------------------------

    async fn load_abac_policies(&self) -> Result<Vec<AbacPolicy>> {
        let mut entries = Vec::new();
        let guard = self.abac_policies_heap.scan()?;
        guard.for_each(|_tid, view| {
            if let Ok(entry) = AbacPolicy::from_bytes(view.data) {
                entries.push(entry);
            }
        });
        Ok(entries)
    }

    async fn store_abac_policy(&self, policy: &AbacPolicy) -> Result<()> {
        let bytes = policy.to_bytes();
        let tuple = Tuple::new(bytes, 0);
        let _ = self.abac_policies_heap.insert_batch(&[tuple]).await?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Row ownership configs
    // -----------------------------------------------------------------------

    async fn load_row_ownership_configs(&self) -> Result<Vec<RowOwnershipConfig>> {
        let mut entries = Vec::new();
        let guard = self.row_ownership_heap.scan()?;
        guard.for_each(|_tid, view| {
            if let Ok(entry) = RowOwnershipConfig::from_bytes(view.data) {
                entries.push(entry);
            }
        });
        Ok(entries)
    }

    async fn store_row_ownership_config(&self, config: &RowOwnershipConfig) -> Result<()> {
        let bytes = config.to_bytes();
        let tuple = Tuple::new(bytes, 0);
        let _ = self.row_ownership_heap.insert_batch(&[tuple]).await?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Delegation edges
    // -----------------------------------------------------------------------

    async fn load_delegation_edges(&self) -> Result<Vec<DelegationEdge>> {
        let mut entries = Vec::new();
        let guard = self.delegation_edges_heap.scan()?;
        guard.for_each(|_tid, view| {
            if let Ok(entry) = DelegationEdge::from_bytes(view.data) {
                entries.push(entry);
            }
        });
        Ok(entries)
    }

    // -----------------------------------------------------------------------
    // Masking rules
    // -----------------------------------------------------------------------

    async fn store_masking_rule(&self, rule: &MaskingRule) -> Result<()> {
        let bytes = rule.to_bytes();
        let tuple = Tuple::new(bytes, 0);
        let _ = self.masking_rules_heap.insert_batch(&[tuple]).await?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Auth rules
    // -----------------------------------------------------------------------

    async fn load_auth_rules(&self) -> Result<Vec<AuthRule>> {
        let mut entries = Vec::new();
        let guard = self.auth_rules_heap.scan()?;
        guard.for_each(|_tid, view| {
            if let Ok(entry) = AuthRule::from_bytes(view.data) {
                entries.push(entry);
            }
        });
        Ok(entries)
    }

    // -----------------------------------------------------------------------
    // Users
    // -----------------------------------------------------------------------

    async fn load_users(&self) -> Result<Vec<User>> {
        let mut entries = Vec::new();
        let guard = self.users_heap.scan()?;
        guard.for_each(|_tid, view| {
            if let Ok(entry) = User::from_bytes(view.data) {
                entries.push(entry);
            }
        });
        Ok(entries)
    }

    async fn store_user(&self, user: &User) -> Result<()> {
        let bytes = user.to_bytes();
        let tuple = Tuple::new(bytes, 0);
        let _ = self.users_heap.insert_batch(&[tuple]).await?;
        Ok(())
    }

    async fn delete_user(&self, id: UserId) -> Result<()> {
        let mut target = None;
        let guard = self.users_heap.scan()?;
        guard.for_each(|tid, view| {
            if let Ok(entry) = User::from_bytes(view.data) {
                if entry.id == id {
                    target = Some(tid);
                }
            }
        });
        if let Some(tid) = target {
            self.users_heap.delete(tid).await?;
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // User-role memberships
    // -----------------------------------------------------------------------

    async fn load_user_memberships(&self) -> Result<Vec<UserRoleMembership>> {
        let mut entries = Vec::new();
        let guard = self.user_memberships_heap.scan()?;
        guard.for_each(|_tid, view| {
            if let Ok(entry) = UserRoleMembership::from_bytes(view.data) {
                entries.push(entry);
            }
        });
        Ok(entries)
    }

    async fn store_user_membership(&self, m: &UserRoleMembership) -> Result<()> {
        let bytes = m.to_bytes();
        let tuple = Tuple::new(bytes, 0);
        let _ = self.user_memberships_heap.insert_batch(&[tuple]).await?;
        Ok(())
    }

    async fn delete_user_membership(&self, user_id: UserId, role_id: RoleId) -> Result<()> {
        let mut target = None;
        let guard = self.user_memberships_heap.scan()?;
        guard.for_each(|tid, view| {
            if let Ok(entry) = UserRoleMembership::from_bytes(view.data) {
                if entry.user_id == user_id && entry.role_id == role_id {
                    target = Some(tid);
                }
            }
        });
        if let Some(tid) = target {
            self.user_memberships_heap.delete(tid).await?;
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // IP blocks
    // -----------------------------------------------------------------------

    async fn load_ip_blocks(&self) -> Result<Vec<IpBlockEntry>> {
        let mut entries = Vec::new();
        let guard = self.ip_blocks_heap.scan()?;
        guard.for_each(|_tid, view| {
            if let Ok(entry) = IpBlockEntry::from_bytes(view.data) {
                entries.push(entry);
            }
        });
        Ok(entries)
    }

    async fn store_ip_block(&self, entry: &IpBlockEntry) -> Result<()> {
        let bytes = entry.to_bytes();
        let tuple = Tuple::new(bytes, 0);
        let _ = self.ip_blocks_heap.insert_batch(&[tuple]).await?;
        Ok(())
    }

    async fn delete_ip_block(&self, ip: &str) -> Result<()> {
        let mut target = None;
        let guard = self.ip_blocks_heap.scan()?;
        guard.for_each(|tid, view| {
            if let Ok(entry) = IpBlockEntry::from_bytes(view.data) {
                if entry.ip_or_cidr == ip {
                    target = Some(tid);
                }
            }
        });
        if let Some(tid) = target {
            self.ip_blocks_heap.delete(tid).await?;
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Trusted IPs
    // -----------------------------------------------------------------------

    async fn load_trusted_ips(&self) -> Result<Vec<TrustedIpEntry>> {
        let mut entries = Vec::new();
        let guard = self.trusted_ips_heap.scan()?;
        guard.for_each(|_tid, view| {
            if let Ok(entry) = TrustedIpEntry::from_bytes(view.data) {
                entries.push(entry);
            }
        });
        Ok(entries)
    }

    async fn store_trusted_ip(&self, entry: &TrustedIpEntry) -> Result<()> {
        let bytes = entry.to_bytes();
        let tuple = Tuple::new(bytes, 0);
        let _ = self.trusted_ips_heap.insert_batch(&[tuple]).await?;
        Ok(())
    }

    async fn delete_trusted_ip(&self, ip: &str) -> Result<()> {
        let mut target = None;
        let guard = self.trusted_ips_heap.scan()?;
        guard.for_each(|tid, view| {
            if let Ok(entry) = TrustedIpEntry::from_bytes(view.data) {
                if entry.ip_or_cidr == ip {
                    target = Some(tid);
                }
            }
        });
        if let Some(tid) = target {
            self.trusted_ips_heap.delete(tid).await?;
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Auth attempts (brute force audit trail)
    // -----------------------------------------------------------------------

    async fn store_attempt(&self, entry: &AttemptEntry) -> Result<()> {
        let bytes = entry.to_bytes();
        let tuple = Tuple::new(bytes, 0);
        let _ = self.auth_attempts_heap.insert_batch(&[tuple]).await?;
        Ok(())
    }

    async fn load_recent_attempts(&self, limit: usize) -> Result<Vec<AttemptEntry>> {
        let mut entries = Vec::new();
        let guard = self.auth_attempts_heap.scan()?;
        guard.for_each(|_tid, view| {
            if let Ok(entry) = AttemptEntry::from_bytes(view.data) {
                entries.push(entry);
            }
        });
        // Sort by timestamp descending, then truncate to limit
        entries.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        entries.truncate(limit);
        Ok(entries)
    }

    // -----------------------------------------------------------------------
    // Brute force policy bindings
    // -----------------------------------------------------------------------

    async fn load_brute_force_policies(&self) -> Result<Vec<BruteForcePolicyBinding>> {
        let mut entries = Vec::new();
        let guard = self.brute_force_policies_heap.scan()?;
        guard.for_each(|_tid, view| {
            if let Ok(entry) = BruteForcePolicyBinding::from_bytes(view.data) {
                entries.push(entry);
            }
        });
        Ok(entries)
    }

    async fn store_brute_force_policy(&self, binding: &BruteForcePolicyBinding) -> Result<()> {
        let bytes = binding.to_bytes();
        let tuple = Tuple::new(bytes, 0);
        let _ = self
            .brute_force_policies_heap
            .insert_batch(&[tuple])
            .await?;
        Ok(())
    }

    async fn delete_brute_force_policy(
        &self,
        role_id: Option<RoleId>,
        database: Option<&str>,
    ) -> Result<()> {
        let mut target = None;
        let guard = self.brute_force_policies_heap.scan()?;
        guard.for_each(|tid, view| {
            if let Ok(entry) = BruteForcePolicyBinding::from_bytes(view.data) {
                if entry.role_id == role_id && entry.database_pattern.as_deref() == database {
                    target = Some(tid);
                }
            }
        });
        if let Some(tid) = target {
            self.brute_force_policies_heap.delete(tid).await?;
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // RLS Policies
    // -----------------------------------------------------------------------

    async fn load_rls_policies(&self) -> Result<Vec<RlsPolicy>> {
        let mut entries = Vec::new();
        let guard = self.rls_policies_heap.scan()?;
        guard.for_each(|_tid, view| {
            if let Ok(entry) = RlsPolicy::from_bytes(view.data) {
                entries.push(entry);
            }
        });
        Ok(entries)
    }

    async fn store_rls_policy(&self, policy: &RlsPolicy) -> Result<()> {
        let bytes = policy.to_bytes();
        let tuple = Tuple::new(bytes, 0);
        let _ = self.rls_policies_heap.insert_batch(&[tuple]).await?;
        Ok(())
    }

    async fn delete_rls_policy(&self, table_id: u32, name: &str) -> Result<()> {
        let mut target = None;
        let guard = self.rls_policies_heap.scan()?;
        guard.for_each(|tid, view| {
            if let Ok(entry) = RlsPolicy::from_bytes(view.data) {
                if entry.table_id == table_id && entry.name == name {
                    target = Some(tid);
                }
            }
        });
        if let Some(tid) = target {
            self.rls_policies_heap.delete(tid).await?;
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Masking Policies
    // -----------------------------------------------------------------------

    async fn load_masking_policies(&self) -> Result<Vec<MaskingPolicy>> {
        let mut entries = Vec::new();
        let guard = self.masking_policies_heap.scan()?;
        guard.for_each(|_tid, view| {
            if let Ok(entry) = MaskingPolicy::from_bytes(view.data) {
                entries.push(entry);
            }
        });
        Ok(entries)
    }

    async fn store_masking_policy(&self, policy: &MaskingPolicy) -> Result<()> {
        let bytes = policy.to_bytes();
        let tuple = Tuple::new(bytes, 0);
        let _ = self.masking_policies_heap.insert_batch(&[tuple]).await?;
        Ok(())
    }

    async fn delete_masking_policy(&self, table_id: u32, column_id: u16, name: &str) -> Result<()> {
        let mut target = None;
        let guard = self.masking_policies_heap.scan()?;
        guard.for_each(|tid, view| {
            if let Ok(entry) = MaskingPolicy::from_bytes(view.data) {
                if entry.table_id == table_id && entry.column_id == column_id && entry.name == name
                {
                    target = Some(tid);
                }
            }
        });
        if let Some(tid) = target {
            self.masking_policies_heap.delete(tid).await?;
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // ABAC Rules
    // -----------------------------------------------------------------------

    async fn load_abac_rules(&self) -> Result<Vec<AbacRule>> {
        let mut entries = Vec::new();
        let guard = self.abac_rules_heap.scan()?;
        guard.for_each(|_tid, view| {
            if let Ok(entry) = AbacRule::from_bytes(view.data) {
                entries.push(entry);
            }
        });
        Ok(entries)
    }

    async fn store_abac_rule(&self, rule: &AbacRule) -> Result<()> {
        let bytes = rule.to_bytes();
        let tuple = Tuple::new(bytes, 0);
        let _ = self.abac_rules_heap.insert_batch(&[tuple]).await?;
        Ok(())
    }

    async fn delete_abac_rule(&self, id: u32) -> Result<()> {
        let mut target = None;
        let guard = self.abac_rules_heap.scan()?;
        guard.for_each(|tid, view| {
            if let Ok(entry) = AbacRule::from_bytes(view.data) {
                if entry.id == id {
                    target = Some(tid);
                }
            }
        });
        if let Some(tid) = target {
            self.abac_rules_heap.delete(tid).await?;
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Column Encryption
    // -----------------------------------------------------------------------

    async fn load_column_encryptions(&self) -> Result<Vec<ColumnEncryption>> {
        let mut entries = Vec::new();
        let guard = self.column_encryption_heap.scan()?;
        guard.for_each(|_tid, view| {
            if let Ok(entry) = ColumnEncryption::from_bytes(view.data) {
                entries.push(entry);
            }
        });
        Ok(entries)
    }

    async fn store_column_encryption(&self, config: &ColumnEncryption) -> Result<()> {
        let bytes = config.to_bytes();
        let tuple = Tuple::new(bytes, 0);
        let _ = self.column_encryption_heap.insert_batch(&[tuple]).await?;
        Ok(())
    }

    async fn delete_column_encryption(&self, table_id: u32, column_id: u16) -> Result<()> {
        let mut target = None;
        let guard = self.column_encryption_heap.scan()?;
        guard.for_each(|tid, view| {
            if let Ok(entry) = ColumnEncryption::from_bytes(view.data) {
                if entry.table_id == table_id && entry.column_id == column_id {
                    target = Some(tid);
                }
            }
        });
        if let Some(tid) = target {
            self.column_encryption_heap.delete(tid).await?;
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Security Labels
    // -----------------------------------------------------------------------

    async fn load_object_security_labels(&self) -> Result<Vec<ObjectSecurityLabel>> {
        let mut entries = Vec::new();
        let guard = self.security_labels_heap.scan()?;
        guard.for_each(|_tid, view| {
            if let Ok(entry) = ObjectSecurityLabel::from_bytes(view.data) {
                entries.push(entry);
            }
        });
        Ok(entries)
    }

    async fn store_object_security_label(&self, label: &ObjectSecurityLabel) -> Result<()> {
        let bytes = label.to_bytes();
        let tuple = Tuple::new(bytes, 0);
        let _ = self.security_labels_heap.insert_batch(&[tuple]).await?;
        Ok(())
    }

    async fn load_subject_security_labels(&self) -> Result<Vec<SubjectSecurityLabel>> {
        let mut entries = Vec::new();
        let guard = self.subject_labels_heap.scan()?;
        guard.for_each(|_tid, view| {
            if let Ok(entry) = SubjectSecurityLabel::from_bytes(view.data) {
                entries.push(entry);
            }
        });
        Ok(entries)
    }

    async fn store_subject_security_label(&self, label: &SubjectSecurityLabel) -> Result<()> {
        let bytes = label.to_bytes();
        let tuple = Tuple::new(bytes, 0);
        let _ = self.subject_labels_heap.insert_batch(&[tuple]).await?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // WebAuthn Credentials
    // -----------------------------------------------------------------------

    async fn load_webauthn_credentials(&self) -> Result<Vec<WebAuthnCredential>> {
        let mut entries = Vec::new();
        let guard = self.webauthn_credentials_heap.scan()?;
        guard.for_each(|_tid, view| {
            if let Ok(entry) = WebAuthnCredential::from_bytes(view.data) {
                entries.push(entry);
            }
        });
        Ok(entries)
    }

    async fn store_webauthn_credential(&self, cred: &WebAuthnCredential) -> Result<()> {
        let bytes = cred.to_bytes();
        let tuple = Tuple::new(bytes, 0);
        let _ = self
            .webauthn_credentials_heap
            .insert_batch(&[tuple])
            .await?;
        Ok(())
    }

    async fn delete_webauthn_credential(&self, credential_id: &[u8]) -> Result<()> {
        let mut target = None;
        let guard = self.webauthn_credentials_heap.scan()?;
        guard.for_each(|tid, view| {
            if let Ok(entry) = WebAuthnCredential::from_bytes(view.data) {
                if entry.credential_id == credential_id {
                    target = Some(tid);
                }
            }
        });
        if let Some(tid) = target {
            self.webauthn_credentials_heap.delete(tid).await?;
        }
        Ok(())
    }

    async fn update_webauthn_sign_count(&self, credential_id: &[u8], new_count: u32) -> Result<()> {
        // Delete the old entry and re-insert with updated sign count.
        // This is the simplest approach with heap storage (no in-place update).
        let mut found = None;
        let guard = self.webauthn_credentials_heap.scan()?;
        guard.for_each(|tid, view| {
            if let Ok(entry) = WebAuthnCredential::from_bytes(view.data) {
                if entry.credential_id == credential_id {
                    found = Some((tid, entry));
                }
            }
        });
        drop(guard);

        if let Some((tid, mut cred)) = found {
            self.webauthn_credentials_heap.delete(tid).await?;
            cred.sign_count = new_count;
            cred.last_used_at = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            let bytes = cred.to_bytes();
            let tuple = Tuple::new(bytes, 0);
            let _ = self
                .webauthn_credentials_heap
                .insert_batch(&[tuple])
                .await?;
        }
        Ok(())
    }
}

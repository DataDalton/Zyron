//! Data lifecycle management for ZyronDB (Phase 17).
//!
//! Retention/TTL, tiered storage, archival, soft delete, legal hold,
//! GDPR erasure, data classification, compliance auditing, recycle bin,
//! crypto-shredding, and a tamper-evident audit chain.
//!
//! Hot-path reads (legal hold lookup, tier cache) are lock-free via RcuMap
//! and atomics. Background workers own their own threads and may use local
//! locks. No unwrap on any path; all fallible work returns Result.

pub mod archive;
pub mod audit_chain;
pub mod classification;
pub mod compliance;
pub mod cryptoshred;
pub mod dryrun;
pub mod dsar;
pub mod erasure;
pub mod legal_hold;
pub mod presets;
pub mod recyclebin;
pub mod residency;
pub mod scheduling;
pub mod soft_delete;
pub mod tiered_storage;
pub mod ttl;
pub mod worm;

use std::sync::Arc;

use audit_chain::AuditChain;
use classification::ClassificationService;
use cryptoshred::KeyVault;
use legal_hold::LegalHoldRegistry;
use scheduling::CleanupGovernor;
use tiered_storage::TierCache;

/// Shared lifecycle state held by the server and handed to hooks, the binder
/// bridge, and background workers.
pub struct LifecycleState {
    pub legal_holds: Arc<LegalHoldRegistry>,
    pub classification: Arc<ClassificationService>,
    pub tier_cache: Arc<TierCache>,
    pub cleanup_governor: Arc<CleanupGovernor>,
    pub key_vault: Arc<KeyVault>,
    pub audit_chain: Arc<AuditChain>,
}

impl LifecycleState {
    pub fn new() -> Self {
        Self {
            legal_holds: Arc::new(LegalHoldRegistry::new()),
            classification: Arc::new(ClassificationService::new()),
            tier_cache: Arc::new(TierCache::new(4096)),
            cleanup_governor: Arc::new(CleanupGovernor::default()),
            key_vault: Arc::new(KeyVault::new()),
            audit_chain: Arc::new(AuditChain::new()),
        }
    }
}

impl Default for LifecycleState {
    fn default() -> Self {
        Self::new()
    }
}

//! Crypto-shredding key vault. Per-subject/tenant data keys; destroying a key
//! makes all data encrypted under it permanently unreadable, giving instant
//! compliant erasure of tiered/archived data without rewriting remote objects.
//!
//! Key bytes are random 32-byte values; XChaCha-style stream encryption is
//! delegated to zyron-auth's encryption layer at the call site. This module
//! owns the key lifecycle (create / lookup / destroy) lock-free.

use std::sync::atomic::{AtomicU64, Ordering};

use zyron_auth::rcu::RcuMap;
use zyron_common::{Result, ZyronError};

/// A data key. `destroyed` keys are retained as tombstones so their id is
/// never reused and audits can prove destruction.
#[derive(Clone)]
pub struct DataKey {
    pub id: u64,
    pub material: Option<[u8; 32]>,
}

impl DataKey {
    pub fn is_destroyed(&self) -> bool {
        self.material.is_none()
    }
}

/// Lock-free vault of per-subject data keys.
pub struct KeyVault {
    keys: RcuMap<u64, DataKey>,
    next_id: AtomicU64,
}

impl KeyVault {
    pub fn new() -> Self {
        Self {
            keys: RcuMap::empty_map(),
            next_id: AtomicU64::new(1),
        }
    }

    /// Creates a new data key from a 32-byte seed and returns its id.
    pub fn create_key(&self, seed: [u8; 32]) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::AcqRel);
        self.keys.insert(
            id,
            DataKey {
                id,
                material: Some(seed),
            },
        );
        id
    }

    /// Returns the key material, or an error if the key was destroyed.
    pub fn key_material(&self, id: u64) -> Result<[u8; 32]> {
        match self.keys.get(&id) {
            Some(k) => k
                .material
                .ok_or_else(|| ZyronError::Internal(format!("data key {id} was crypto-shredded"))),
            None => Err(ZyronError::Internal(format!("data key {id} not found"))),
        }
    }

    /// Destroys a key (crypto-shred). Idempotent. Leaves a tombstone.
    pub fn destroy_key(&self, id: u64) -> bool {
        match self.keys.get(&id) {
            Some(mut k) => {
                if k.is_destroyed() {
                    return true;
                }
                k.material = None;
                self.keys.insert(id, k);
                true
            }
            None => false,
        }
    }
}

impl Default for KeyVault {
    fn default() -> Self {
        Self::new()
    }
}

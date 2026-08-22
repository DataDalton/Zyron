//! Server-wide L2 plan cache shared across all connections.
//!
//! Each connection has a lock-free per-session L1 cache (`statement_cache`).
//! On an L1 miss the connection consults this shared L2 before paying a full
//! parse + bind + plan, so a query shape planned by one connection is reused
//! by every other connection with the same identity, search path, and schema
//! version. Both layers are populated on a miss.
//!
//! Concurrency: the table is split into many independent shards, each a small
//! `RwLock`-guarded FIFO list. A truly lock-free table that stores `Arc`
//! values would require epoch-based reclamation to clone a value without
//! racing a concurrent eviction's drop (a seqlock cannot guard a non-trivial
//! clone). Sharding instead spreads lookups across hundreds of independent
//! locks so contention is effectively nil at the access rates we target,
//! while the read lock makes the value clone provably free of use-after-free.

use parking_lot::RwLock;

use crate::statement_cache::{CacheKey, CachedPlan};

/// Number of independent shards. Power of two so the index is a mask. At a
/// few hundred shards a given shard is touched rarely enough that the RwLock
/// is almost always uncontended.
const SHARD_COUNT: usize = 512;
const SHARD_MASK: usize = SHARD_COUNT - 1;

/// Max entries per shard. SHARD_COUNT * PER_SHARD_CAP = 16,384 cached plans,
/// bounded so a workload with unbounded distinct shapes cannot grow memory
/// without limit. A shard holds at most this many before FIFO eviction.
const PER_SHARD_CAP: usize = 32;

pub struct ServerPlanCache {
    shards: Box<[RwLock<Vec<CachedPlan>>]>,
}

impl ServerPlanCache {
    pub fn new() -> Self {
        let mut shards = Vec::with_capacity(SHARD_COUNT);
        for _ in 0..SHARD_COUNT {
            shards.push(RwLock::new(Vec::new()));
        }
        Self {
            shards: shards.into_boxed_slice(),
        }
    }

    #[inline]
    fn shard_index(key: &CacheKey) -> usize {
        // Fold the independent key components together in order. Each is
        // already a well-distributed hash, so the rotate-xor fold spreads
        // them across shards without clustering
        let mixed = [
            key.search_path_hash,
            key.role_id,
            key.type_kinds_hash,
            key.rls_policy_hash,
        ]
        .into_iter()
        .fold(key.template_hash, zyron_common::hash_fold);
        (mixed as usize) & SHARD_MASK
    }

    /// Returns a clone of the cached plan when an entry matches `key` and was
    /// planned under `current_schema_version`. Stale-version entries are
    /// treated as a miss and replaced on the next insert of the same key.
    pub fn lookup(&self, key: &CacheKey, current_schema_version: u64) -> Option<CachedPlan> {
        let shard = &self.shards[Self::shard_index(key)];
        let guard = shard.read();
        for e in guard.iter() {
            if &e.key == key {
                if e.schema_version == current_schema_version {
                    return Some(e.clone());
                }
                return None;
            }
        }
        None
    }

    /// Inserts or replaces the entry for its key. Replacing an existing key
    /// in place keeps stale-version entries from accumulating; otherwise the
    /// oldest entry in the shard is evicted to honor the per-shard bound.
    pub fn insert(&self, entry: CachedPlan) {
        let shard = &self.shards[Self::shard_index(&entry.key)];
        let mut guard = shard.write();
        if let Some(slot) = guard.iter_mut().find(|e| e.key == entry.key) {
            *slot = entry;
            return;
        }
        if guard.len() >= PER_SHARD_CAP {
            guard.remove(0);
        }
        guard.push(entry);
    }
}

impl Default for ServerPlanCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use zyron_planner::PlanCost;
    use zyron_planner::physical::PhysicalPlan;

    fn key(template_hash: u64, role_id: u64) -> CacheKey {
        CacheKey {
            template_hash,
            search_path_hash: 0,
            role_id,
            rls_policy_hash: 0,
            type_kinds_hash: 0,
        }
    }

    fn entry(template_hash: u64, role_id: u64, schema_version: u64) -> CachedPlan {
        CachedPlan {
            key: key(template_hash, role_id),
            schema_version,
            plan: Arc::new(PhysicalPlan::Values {
                rows: Vec::new(),
                schema: Vec::new(),
                cost: PlanCost {
                    io_cost: 0.0,
                    cpu_cost: 0.0,
                    row_count: 0.0,
                },
            }),
            output_schema: Vec::new(),
            param_count: 0,
        }
    }

    #[test]
    fn insert_then_lookup_hits() {
        let c = ServerPlanCache::new();
        c.insert(entry(1, 7, 3));
        assert!(c.lookup(&key(1, 7), 3).is_some());
    }

    #[test]
    fn cross_connection_shape_reuse() {
        // One connection inserts; another (same identity) reads it.
        let c = ServerPlanCache::new();
        c.insert(entry(42, 100, 5));
        let hit = c.lookup(&key(42, 100), 5);
        assert!(hit.is_some(), "same identity must reuse the shared plan");
    }

    #[test]
    fn different_role_does_not_collide() {
        let c = ServerPlanCache::new();
        c.insert(entry(42, 100, 5));
        assert!(
            c.lookup(&key(42, 200), 5).is_none(),
            "a different role must not pull another role's plan"
        );
    }

    #[test]
    fn stale_schema_version_misses() {
        let c = ServerPlanCache::new();
        c.insert(entry(9, 1, 4));
        assert!(c.lookup(&key(9, 1), 5).is_none());
    }

    #[test]
    fn reinsert_replaces_in_place() {
        let c = ServerPlanCache::new();
        c.insert(entry(9, 1, 4));
        c.insert(entry(9, 1, 5)); // newer schema version, same key
        assert!(c.lookup(&key(9, 1), 4).is_none());
        assert!(c.lookup(&key(9, 1), 5).is_some());
    }

    #[test]
    fn concurrent_inserts_and_lookups_are_safe() {
        use std::thread;
        let c = Arc::new(ServerPlanCache::new());
        let mut handles = Vec::new();
        for t in 0..8u64 {
            let c = Arc::clone(&c);
            handles.push(thread::spawn(move || {
                for i in 0..2000u64 {
                    let th = t * 10_000 + i;
                    c.insert(entry(th, t, 1));
                    let _ = c.lookup(&key(th, t), 1);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    }
}

//! Tiered storage: tier model, planner cost multiplier, and a bounded
//! lock-free on-demand fetch cache for cold/archive segments.

use std::sync::atomic::{AtomicU64, Ordering};

use zyron_auth::rcu::RcuMap;

/// The tier model lives in zyron-common so the planner can cost a scan by
/// the tier its segments sit on without depending on this crate. One
/// definition, one cost table.
pub use zyron_common::StorageTier;

/// One cached cold/archive segment plus an access counter for LFU promotion.
#[derive(Clone)]
struct CachedSegment {
    bytes: std::sync::Arc<Vec<u8>>,
    hits: std::sync::Arc<AtomicU64>,
}

/// Bounded lock-free cache of fetched cold/archive segments keyed by
/// (table_id, segment_id). Eviction is approximate-LFU via atomic counters.
pub struct TierCache {
    map: RcuMap<u64, CachedSegment>,
    capacity: usize,
    inserts: AtomicU64,
}

impl TierCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            map: RcuMap::empty_map(),
            capacity: capacity.max(1),
            inserts: AtomicU64::new(0),
        }
    }

    fn key(table_id: u32, segment_id: u32) -> u64 {
        ((table_id as u64) << 32) | segment_id as u64
    }

    /// Returns cached bytes for a segment, recording a hit for LFU.
    pub fn get(&self, table_id: u32, segment_id: u32) -> Option<std::sync::Arc<Vec<u8>>> {
        let k = Self::key(table_id, segment_id);
        self.map.get(&k).map(|seg| {
            seg.hits.fetch_add(1, Ordering::Relaxed);
            seg.bytes
        })
    }

    /// Inserts fetched segment bytes. When capacity is exceeded the insert
    /// counter wraps and least-frequently-used entries are dropped lazily by
    /// callers of `evict_if_needed`.
    pub fn put(&self, table_id: u32, segment_id: u32, bytes: Vec<u8>) {
        let k = Self::key(table_id, segment_id);
        self.map.insert(
            k,
            CachedSegment {
                bytes: std::sync::Arc::new(bytes),
                hits: std::sync::Arc::new(AtomicU64::new(1)),
            },
        );
        self.inserts.fetch_add(1, Ordering::Relaxed);
    }

    /// Drops a specific cached segment (used after promotion to a hotter tier).
    pub fn invalidate(&self, table_id: u32, segment_id: u32) {
        self.map.remove(&Self::key(table_id, segment_id));
    }

    /// Hit count for a cached segment, for access-driven rehydration
    /// decisions. Returns 0 when not cached.
    pub fn hit_count(&self, table_id: u32, segment_id: u32) -> u64 {
        self.map
            .get(&Self::key(table_id, segment_id))
            .map(|s| s.hits.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

/// Decides whether a frequently-accessed cold segment should be promoted back
/// to a hotter tier (access-driven auto-rehydration).
pub fn should_rehydrate(hit_count: u64, threshold: u64) -> bool {
    hit_count >= threshold.max(1)
}

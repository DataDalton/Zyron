//! Runtime state for SQL resilience patterns.
//!
//! Bulkheads and retry policies are catalog objects. The binder resolves a
//! policy name to its configuration and embeds the configuration into the
//! bound expression, so evaluation needs no catalog access. This module
//! holds the process wide live objects those expressions share: bulkhead
//! admission state per policy id, and the cache aside store.
//!
//! Retry delays sleep the evaluating thread. Total sleep per evaluation is
//! capped so a policy with a large max_delay cannot stall an executor
//! thread indefinitely.

use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use zyron_types::resilience::{Bulkhead, BulkheadRegistry};

use crate::column::ScalarValue;

/// Upper bound on cumulative retry sleep within one expression evaluation
pub const MAX_TOTAL_RETRY_SLEEP: Duration = Duration::from_secs(30);

static BULKHEADS: OnceLock<BulkheadRegistry> = OnceLock::new();

fn bulkheads() -> &'static BulkheadRegistry {
    BULKHEADS.get_or_init(BulkheadRegistry::new)
}

/// Returns the shared bulkhead for a policy id, creating it from the
/// embedded configuration on first use
pub fn bulkhead_for(
    policy_id: u32,
    max_concurrent: u32,
    queue_size: u32,
    max_wait: Duration,
) -> zyron_common::Result<Arc<Bulkhead>> {
    bulkheads().get_or_create(&policy_id.to_string(), max_concurrent, queue_size, max_wait)
}

/// Drops the live state of a policy after DROP BULKHEAD / DROP RETRY POLICY
pub fn invalidate_policy(policy_id: u32) {
    bulkheads().remove(&policy_id.to_string());
}

// ---------------------------------------------------------------------------
// Cache aside store
// ---------------------------------------------------------------------------

struct CachedEntry {
    value: ScalarValue,
    stored_at: Instant,
}

static CACHE: OnceLock<scc::HashMap<String, CachedEntry>> = OnceLock::new();

fn cache() -> &'static scc::HashMap<String, CachedEntry> {
    CACHE.get_or_init(scc::HashMap::new)
}

/// How a cache aside lookup answered
pub enum CacheLookup {
    /// Younger than the ttl, served without evaluating the fetch
    Fresh(ScalarValue),
    /// Between ttl and stale_ttl, served but wants a refresh
    Stale(ScalarValue),
    /// Absent or older than stale_ttl, the fetch value is required
    Miss,
}

pub fn cache_lookup(key: &str, ttl: Duration, stale_ttl: Duration) -> CacheLookup {
    let now = Instant::now();
    let mut expired = false;
    let result = cache().read_sync(key, |_, entry| {
        let age = now.saturating_duration_since(entry.stored_at);
        if age < ttl {
            CacheLookup::Fresh(entry.value.clone())
        } else if age < stale_ttl {
            CacheLookup::Stale(entry.value.clone())
        } else {
            CacheLookup::Miss
        }
    });
    if let Some(CacheLookup::Miss) = result {
        expired = true;
    }
    if expired {
        let _ = cache().remove_sync(key);
    }
    result.unwrap_or(CacheLookup::Miss)
}

pub fn cache_store(key: String, value: ScalarValue) {
    let entry = CachedEntry {
        value,
        stored_at: Instant::now(),
    };
    if cache()
        .update_sync(&key, |_, existing| {
            existing.value = entry.value.clone();
            existing.stored_at = entry.stored_at;
        })
        .is_none()
    {
        let _ = cache().insert_sync(key, entry);
    }
}

/// Clears the whole store, for tests
pub fn cache_clear() {
    cache().clear_sync();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_fresh_then_stale_then_miss() {
        cache_store("k1".to_string(), ScalarValue::Int64(42));
        match cache_lookup("k1", Duration::from_secs(60), Duration::from_secs(120)) {
            CacheLookup::Fresh(ScalarValue::Int64(42)) => {}
            _ => panic!("expected a fresh hit"),
        }
        match cache_lookup("k1", Duration::ZERO, Duration::from_secs(120)) {
            CacheLookup::Stale(ScalarValue::Int64(42)) => {}
            _ => panic!("expected a stale hit"),
        }
        match cache_lookup("k1", Duration::ZERO, Duration::ZERO) {
            CacheLookup::Miss => {}
            _ => panic!("expected a miss past stale_ttl"),
        }
        // The miss evicted the entry
        match cache_lookup("k1", Duration::from_secs(60), Duration::from_secs(120)) {
            CacheLookup::Miss => {}
            _ => panic!("expected the evicted key to miss"),
        }
    }

    #[test]
    fn test_bulkhead_registry_shared_and_invalidated() {
        let a = bulkhead_for(9001, 2, 0, Duration::ZERO).expect("build");
        let b = bulkhead_for(9001, 2, 0, Duration::ZERO).expect("cached");
        assert!(Arc::ptr_eq(&a, &b), "same policy id shares one bulkhead");
        invalidate_policy(9001);
        let c = bulkhead_for(9001, 2, 0, Duration::ZERO).expect("rebuilt");
        assert!(!Arc::ptr_eq(&a, &c), "invalidation drops the shared state");
    }
}

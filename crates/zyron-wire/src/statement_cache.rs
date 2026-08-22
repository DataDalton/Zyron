//! Per-connection cache of planned SQL statements keyed by a normalized
//! query template plus session identity.
//!
//! The same query *shape* is executed over and over by real workloads:
//! an application issues `INSERT INTO t VALUES (...)` per row, a bulk load
//! batches the same multi-row INSERT, an OLTP loop runs the same
//! `SELECT ... WHERE id = ?` millions of times. Parsing + binding +
//! planning that shape on every execution dominates per-op latency.
//!
//! This cache stores the planned `PhysicalPlan` keyed by a normalized
//! template (literals replaced with `$N` placeholders) so a repeated shape
//! with different literal values still hits. The extracted literals are
//! bound as parameters at execute time, so the cached plan is reused
//! verbatim. Invalidation compares the catalog `schema_version` recorded
//! at plan time, which DDL bumps atomically.

use std::collections::VecDeque;
use std::sync::Arc;

use zyron_planner::logical::LogicalColumn;
use zyron_planner::physical::PhysicalPlan;

/// Default capacity, bounded so a misbehaving client cannot grow the cache
/// without limit. Hit rate drops sharply past this size for typical OLTP
/// working sets, so the bound is not tight in practice.
const DEFAULT_CAPACITY: usize = 64;

/// Composite cache key. Two queries share a cached plan only when every
/// component matches, which guarantees the cached plan was bound under the
/// same name resolution, the same row/column security, and the same literal
/// type inference as the incoming query.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct CacheKey {
    /// Hash of the normalized template (literals replaced with `$N`,
    /// comments and redundant whitespace canonicalized).
    pub template_hash: u64,
    /// Hash of the search path the plan was bound under. A different search
    /// path resolves table names differently, so plans must not be shared.
    pub search_path_hash: u64,
    /// Effective role/user id of the session. RLS and column-security
    /// predicates differ per role, so a plan bound for one role must never
    /// be served to another.
    pub role_id: u64,
    /// Hash of the row-level-security policy state visible to this session.
    /// ALTER POLICY can change predicates without a schema_version bump in
    /// some paths, so this is tracked explicitly.
    pub rls_policy_hash: u64,
    /// Hash of the ordered literal `TypeId` sequence extracted from the
    /// query. `(1, 1.0)` and `(1, 2)` must not share a plan because the
    /// second literal is float in one and integer in the other, which can
    /// drive different planner coercions.
    pub type_kinds_hash: u64,
}

/// One cache entry: a planned statement plus the metadata needed to
/// validate and reuse it.
#[derive(Clone)]
pub struct CachedPlan {
    /// The composite lookup key.
    pub key: CacheKey,
    /// Snapshot of `Catalog::schema_version()` at plan time. Stale entries
    /// are discarded on lookup.
    pub schema_version: u64,
    /// The planned physical operator tree.
    pub plan: Arc<PhysicalPlan>,
    /// Output schema for RowDescription.
    pub output_schema: Vec<LogicalColumn>,
    /// Number of `$N` parameters the plan expects. The caller must bind
    /// exactly this many extracted literals before executing.
    pub param_count: usize,
}

/// FIFO-evicted plan cache. FIFO over LRU keeps lookups branchless: a hit
/// does not write back an access timestamp. For short-lived OLTP working
/// sets the difference between FIFO and LRU is noise, and the saved write
/// makes the lookup cheaper.
pub struct StatementCache {
    capacity: usize,
    entries: VecDeque<CachedPlan>,
}

impl StatementCache {
    pub fn new() -> Self {
        Self {
            capacity: DEFAULT_CAPACITY,
            entries: VecDeque::with_capacity(DEFAULT_CAPACITY),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            entries: VecDeque::with_capacity(capacity.max(1)),
        }
    }

    /// Computes the search-path hash. Stable order is required so the same
    /// list always produces the same hash.
    pub fn hash_search_path(search_path: &[String]) -> u64 {
        let mut state: u64 = 0xcbf2_9ce4_8422_2325;
        for entry in search_path {
            state = zyron_common::hash_fold(state, zyron_common::hash64(entry.as_bytes()));
        }
        state
    }

    /// Looks up a plan. Returns `Some` only when the cached entry's key
    /// matches AND its schema_version matches `current_schema_version`.
    /// Stale entries are removed in place.
    pub fn lookup(&mut self, key: &CacheKey, current_schema_version: u64) -> Option<CachedPlan> {
        let mut found_idx: Option<usize> = None;
        for (i, e) in self.entries.iter().enumerate() {
            if &e.key == key {
                found_idx = Some(i);
                break;
            }
        }
        let idx = found_idx?;
        if self.entries[idx].schema_version != current_schema_version {
            self.entries.remove(idx);
            return None;
        }
        Some(self.entries[idx].clone())
    }

    /// Inserts a freshly planned entry. Evicts the oldest entry to stay
    /// within the configured capacity.
    pub fn insert(&mut self, entry: CachedPlan) {
        if self.entries.len() >= self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back(entry);
    }

    /// Drops every cached entry. Used on `SET ROLE` or any event that
    /// invalidates the entire per-session cache wholesale.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for StatementCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyron_planner::PlanCost;
    use zyron_planner::physical::PhysicalPlan;

    fn key(template_hash: u64) -> CacheKey {
        CacheKey {
            template_hash,
            search_path_hash: 0,
            role_id: 0,
            rls_policy_hash: 0,
            type_kinds_hash: 0,
        }
    }

    fn make_entry(template_hash: u64, schema_version: u64) -> CachedPlan {
        CachedPlan {
            key: key(template_hash),
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
    fn lookup_returns_inserted_entry() {
        let mut c = StatementCache::new();
        c.insert(make_entry(7, 1));
        let hit = c.lookup(&key(7), 1).expect("should hit");
        assert_eq!(hit.key.template_hash, 7);
    }

    #[test]
    fn stale_version_evicts_entry() {
        let mut c = StatementCache::new();
        c.insert(make_entry(11, 4));
        assert!(c.lookup(&key(11), 5).is_none());
        assert_eq!(c.len(), 0);
    }

    #[test]
    fn capacity_bound_enforced() {
        let mut c = StatementCache::with_capacity(2);
        c.insert(make_entry(1, 1));
        c.insert(make_entry(2, 1));
        c.insert(make_entry(3, 1));
        assert_eq!(c.len(), 2);
        assert!(c.lookup(&key(1), 1).is_none(), "oldest should be evicted");
        assert!(c.lookup(&key(2), 1).is_some());
        assert!(c.lookup(&key(3), 1).is_some());
    }

    #[test]
    fn different_search_path_does_not_collide() {
        let mut c = StatementCache::new();
        let mut e = make_entry(42, 1);
        e.key.search_path_hash = 100;
        c.insert(e);
        let mut probe = key(42);
        probe.search_path_hash = 200;
        assert!(c.lookup(&probe, 1).is_none());
        probe.search_path_hash = 100;
        assert!(c.lookup(&probe, 1).is_some());
    }

    #[test]
    fn different_role_does_not_collide() {
        let mut c = StatementCache::new();
        let mut e = make_entry(42, 1);
        e.key.role_id = 7;
        c.insert(e);
        let mut probe = key(42);
        probe.role_id = 9;
        assert!(c.lookup(&probe, 1).is_none(), "role must not collide");
        probe.role_id = 7;
        assert!(c.lookup(&probe, 1).is_some());
    }

    #[test]
    fn different_type_kinds_does_not_collide() {
        let mut c = StatementCache::new();
        let mut e = make_entry(42, 1);
        e.key.type_kinds_hash = 0xAAAA;
        c.insert(e);
        let mut probe = key(42);
        probe.type_kinds_hash = 0xBBBB;
        assert!(c.lookup(&probe, 1).is_none(), "type kinds must not collide");
        probe.type_kinds_hash = 0xAAAA;
        assert!(c.lookup(&probe, 1).is_some());
    }
}

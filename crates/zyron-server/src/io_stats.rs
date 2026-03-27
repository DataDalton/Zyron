// Lock-free per-table and per-index IO statistics counters.
//
// Each table and index gets its own stats struct with atomic counters.
// Registries provide concurrent access to stats by ID using scc::HashMap.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

// Per-table IO and tuple activity counters.
pub struct TableIOStats {
    /// Number of sequential scans initiated on this table.
    pub seq_scan: AtomicU64,
    /// Number of tuples read by sequential scans.
    pub seq_tup_read: AtomicU64,
    /// Number of index scans initiated on this table.
    pub idx_scan: AtomicU64,
    /// Number of tuples fetched by index scans.
    pub idx_tup_fetch: AtomicU64,
    /// Number of tuples inserted.
    pub n_tup_ins: AtomicU64,
    /// Number of tuples updated.
    pub n_tup_upd: AtomicU64,
    /// Number of tuples deleted.
    pub n_tup_del: AtomicU64,
    /// Estimated number of dead tuples.
    pub n_dead_tup: AtomicU64,
    /// Last vacuum timestamp in epoch seconds.
    pub last_vacuum: AtomicU64,
    /// Last analyze timestamp in epoch seconds.
    pub last_analyze: AtomicU64,
}

impl TableIOStats {
    /// Creates a new TableIOStats with all counters set to zero.
    pub fn new() -> Self {
        Self {
            seq_scan: AtomicU64::new(0),
            seq_tup_read: AtomicU64::new(0),
            idx_scan: AtomicU64::new(0),
            idx_tup_fetch: AtomicU64::new(0),
            n_tup_ins: AtomicU64::new(0),
            n_tup_upd: AtomicU64::new(0),
            n_tup_del: AtomicU64::new(0),
            n_dead_tup: AtomicU64::new(0),
            last_vacuum: AtomicU64::new(0),
            last_analyze: AtomicU64::new(0),
        }
    }

    /// Increments the sequential scan counter by 1.
    pub fn increment_seq_scan(&self) {
        self.seq_scan.fetch_add(1, Ordering::Relaxed);
    }

    /// Adds the given count to the sequential tuples read counter.
    pub fn increment_seq_tup_read(&self, count: u64) {
        self.seq_tup_read.fetch_add(count, Ordering::Relaxed);
    }

    /// Increments the index scan counter by 1.
    pub fn increment_idx_scan(&self) {
        self.idx_scan.fetch_add(1, Ordering::Relaxed);
    }

    /// Adds the given count to the index tuples fetched counter.
    pub fn increment_idx_tup_fetch(&self, count: u64) {
        self.idx_tup_fetch.fetch_add(count, Ordering::Relaxed);
    }

    /// Increments the tuples inserted counter by 1.
    pub fn increment_n_tup_ins(&self) {
        self.n_tup_ins.fetch_add(1, Ordering::Relaxed);
    }

    /// Increments the tuples updated counter by 1.
    pub fn increment_n_tup_upd(&self) {
        self.n_tup_upd.fetch_add(1, Ordering::Relaxed);
    }

    /// Increments the tuples deleted counter by 1.
    pub fn increment_n_tup_del(&self) {
        self.n_tup_del.fetch_add(1, Ordering::Relaxed);
    }

    /// Adds the given count to the dead tuples counter.
    pub fn increment_n_dead_tup(&self, count: u64) {
        self.n_dead_tup.fetch_add(count, Ordering::Relaxed);
    }

    /// Stores the last vacuum epoch timestamp.
    pub fn set_last_vacuum(&self, epochSecs: u64) {
        self.last_vacuum.store(epochSecs, Ordering::Relaxed);
    }

    /// Stores the last analyze epoch timestamp.
    pub fn set_last_analyze(&self, epochSecs: u64) {
        self.last_analyze.store(epochSecs, Ordering::Relaxed);
    }
}

/// Concurrent registry mapping table IDs to their IO stats.
pub struct TableIOStatsRegistry {
    stats: scc::HashMap<u32, Arc<TableIOStats>>,
}

impl TableIOStatsRegistry {
    /// Creates a new empty registry.
    pub fn new() -> Self {
        Self {
            stats: scc::HashMap::new(),
        }
    }

    /// Returns the stats for the given table ID, creating a new entry if absent.
    pub fn get_or_create(&self, tableId: u32) -> Arc<TableIOStats> {
        match self.stats.entry_sync(tableId) {
            scc::hash_map::Entry::Occupied(entry) => Arc::clone(entry.get()),
            scc::hash_map::Entry::Vacant(entry) => {
                let tableStats = Arc::new(TableIOStats::new());
                let cloned = Arc::clone(&tableStats);
                entry.insert_entry(tableStats);
                cloned
            }
        }
    }

    /// Iterates over all entries, calling f(table_id, stats) for each one.
    pub fn for_each<F: FnMut(u32, &TableIOStats)>(&self, mut f: F) {
        self.stats.iter_sync(|k, v| {
            f(*k, v.as_ref());
            true
        });
    }
}

// Per-index IO counters.
pub struct IndexIOStats {
    /// Number of index scans initiated on this index.
    pub idx_scan: AtomicU64,
    /// Number of index tuples read during scans.
    pub idx_tup_read: AtomicU64,
    /// Number of heap tuples fetched via this index.
    pub idx_tup_fetch: AtomicU64,
}

impl IndexIOStats {
    /// Creates a new IndexIOStats with all counters set to zero.
    pub fn new() -> Self {
        Self {
            idx_scan: AtomicU64::new(0),
            idx_tup_read: AtomicU64::new(0),
            idx_tup_fetch: AtomicU64::new(0),
        }
    }

    /// Increments the index scan counter by 1.
    pub fn increment_idx_scan(&self) {
        self.idx_scan.fetch_add(1, Ordering::Relaxed);
    }

    /// Adds the given count to the index tuples read counter.
    pub fn increment_idx_tup_read(&self, count: u64) {
        self.idx_tup_read.fetch_add(count, Ordering::Relaxed);
    }

    /// Adds the given count to the index tuples fetched counter.
    pub fn increment_idx_tup_fetch(&self, count: u64) {
        self.idx_tup_fetch.fetch_add(count, Ordering::Relaxed);
    }
}

/// Concurrent registry mapping index IDs to their IO stats.
pub struct IndexIOStatsRegistry {
    stats: scc::HashMap<u32, Arc<IndexIOStats>>,
}

impl IndexIOStatsRegistry {
    /// Creates a new empty registry.
    pub fn new() -> Self {
        Self {
            stats: scc::HashMap::new(),
        }
    }

    /// Returns the stats for the given index ID, creating a new entry if absent.
    pub fn get_or_create(&self, indexId: u32) -> Arc<IndexIOStats> {
        match self.stats.entry_sync(indexId) {
            scc::hash_map::Entry::Occupied(entry) => Arc::clone(entry.get()),
            scc::hash_map::Entry::Vacant(entry) => {
                let indexStats = Arc::new(IndexIOStats::new());
                let cloned = Arc::clone(&indexStats);
                entry.insert_entry(indexStats);
                cloned
            }
        }
    }

    /// Iterates over all entries, calling f(index_id, stats) for each one.
    pub fn for_each<F: FnMut(u32, &IndexIOStats)>(&self, mut f: F) {
        self.stats.iter_sync(|k, v| {
            f(*k, v.as_ref());
            true
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_table_stats_increment() {
        let stats = TableIOStats::new();

        stats.increment_seq_scan();
        stats.increment_seq_scan();
        stats.increment_seq_tup_read(100);
        stats.increment_idx_scan();
        stats.increment_idx_tup_fetch(50);
        stats.increment_n_tup_ins();
        stats.increment_n_tup_ins();
        stats.increment_n_tup_ins();
        stats.increment_n_tup_upd();
        stats.increment_n_tup_del();
        stats.increment_n_dead_tup(5);
        stats.set_last_vacuum(1700000000);
        stats.set_last_analyze(1700000100);

        assert_eq!(stats.seq_scan.load(Ordering::Relaxed), 2);
        assert_eq!(stats.seq_tup_read.load(Ordering::Relaxed), 100);
        assert_eq!(stats.idx_scan.load(Ordering::Relaxed), 1);
        assert_eq!(stats.idx_tup_fetch.load(Ordering::Relaxed), 50);
        assert_eq!(stats.n_tup_ins.load(Ordering::Relaxed), 3);
        assert_eq!(stats.n_tup_upd.load(Ordering::Relaxed), 1);
        assert_eq!(stats.n_tup_del.load(Ordering::Relaxed), 1);
        assert_eq!(stats.n_dead_tup.load(Ordering::Relaxed), 5);
        assert_eq!(stats.last_vacuum.load(Ordering::Relaxed), 1700000000);
        assert_eq!(stats.last_analyze.load(Ordering::Relaxed), 1700000100);
    }

    #[test]
    fn test_registry_get_or_create() {
        let registry = TableIOStatsRegistry::new();

        let statsA = registry.get_or_create(42);
        let statsB = registry.get_or_create(42);

        // Both should point to the same allocation.
        assert!(Arc::ptr_eq(&statsA, &statsB));

        // Different table ID returns a different allocation.
        let statsC = registry.get_or_create(99);
        assert!(!Arc::ptr_eq(&statsA, &statsC));
    }

    #[test]
    fn test_registry_for_each() {
        let registry = TableIOStatsRegistry::new();

        let stats1 = registry.get_or_create(1);
        stats1.increment_n_tup_ins();
        stats1.increment_n_tup_ins();

        let stats2 = registry.get_or_create(2);
        stats2.increment_n_tup_ins();
        stats2.increment_n_tup_ins();
        stats2.increment_n_tup_ins();

        let mut totalInserts: u64 = 0;
        let mut tableCount: u32 = 0;

        registry.for_each(|_tableId, tableStats| {
            totalInserts += tableStats.n_tup_ins.load(Ordering::Relaxed);
            tableCount += 1;
        });

        assert_eq!(tableCount, 2);
        assert_eq!(totalInserts, 5);
    }

    #[test]
    fn test_index_stats_increment() {
        let stats = IndexIOStats::new();

        stats.increment_idx_scan();
        stats.increment_idx_scan();
        stats.increment_idx_scan();
        stats.increment_idx_tup_read(200);
        stats.increment_idx_tup_fetch(150);

        assert_eq!(stats.idx_scan.load(Ordering::Relaxed), 3);
        assert_eq!(stats.idx_tup_read.load(Ordering::Relaxed), 200);
        assert_eq!(stats.idx_tup_fetch.load(Ordering::Relaxed), 150);
    }
}

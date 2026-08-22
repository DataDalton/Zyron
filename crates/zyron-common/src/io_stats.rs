#![allow(non_snake_case)]
//! Lock-free per-table and per-index IO statistics counters.
//!
//! Lives here rather than in the server crate because the executor is what
//! observes the activity: scan operators count the rows and bytes they fetch,
//! DML operators count the rows they write, and both sit below the server in
//! the dependency graph. The server owns the registries and hands them to the
//! executor through the execution context, and the stat views read them back.
//!
//! Counters are updated once per batch, never once per row, so a scan pays two
//! relaxed atomic adds per thousand rows rather than two per row.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Per-table IO and tuple activity counters.
///
/// All counts are since process start. Nothing here is persisted; a restart
/// resets every counter, which matches what the numbers mean (activity this
/// server has served).
pub struct TableIOStats {
    /// Sequential passes over this table's storage.
    ///
    /// One pass per store actually read, not one per query: a table with both
    /// a heap tail and folded columnar segments is read by two passes with two
    /// separate costs, and a hybrid scan counts both. A parallel scan counts
    /// one however many workers divide it, because the workers split one pass.
    pub seq_scan: AtomicU64,
    /// Rows returned by sequential scans, counted after visibility filtering
    /// and before any predicate the scan applies on top.
    pub seq_tup_read: AtomicU64,
    /// Index scans initiated against this table, summed over its indexes.
    pub idx_scan: AtomicU64,
    /// Rows fetched through an index scan on this table.
    pub idx_tup_fetch: AtomicU64,
    /// Rows inserted.
    pub n_tup_ins: AtomicU64,
    /// Rows updated. Each update also adds one to n_dead_tup, because the
    /// superseded version stays until vacuum reclaims it.
    pub n_tup_upd: AtomicU64,
    /// Rows deleted.
    pub n_tup_del: AtomicU64,
    /// Rows superseded or deleted and not yet reclaimed. Reset to zero when a
    /// vacuum pass completes on the table.
    pub n_dead_tup: AtomicU64,
    /// Bytes of table data fetched from the storage layer to answer queries.
    ///
    /// Counted whether the bytes came off disk or out of a cache, because the
    /// quantity being measured is the volume the storage format made the query
    /// touch. That is what file pruning, zone maps and column projection
    /// reduce, and it is the one read-side number a heap table and a lake table
    /// can be compared on directly. A heap scan counts whole pages, since a
    /// page read yields the whole row; a lake scan counts only the bytes of the
    /// column segments it actually read out of surviving files.
    pub bytes_read: AtomicU64,
    /// Last vacuum completion, epoch seconds. Zero means never this run.
    pub last_vacuum: AtomicU64,
    /// Last analyze completion, epoch seconds. Zero means never this run.
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
            bytes_read: AtomicU64::new(0),
            last_vacuum: AtomicU64::new(0),
            last_analyze: AtomicU64::new(0),
        }
    }

    /// Records one sequential scan being initiated.
    #[inline]
    pub fn record_seq_scan(&self) {
        self.seq_scan.fetch_add(1, Ordering::Relaxed);
    }

    /// Records one batch produced by a sequential scan: the rows it carries
    /// and the bytes of table data read to produce it. Called once per batch,
    /// so both adds are amortized over the whole batch. A call with nothing to
    /// report costs one predictable branch.
    #[inline]
    pub fn record_seq_batch(&self, tuples: u64, bytes: u64) {
        if tuples != 0 {
            self.seq_tup_read.fetch_add(tuples, Ordering::Relaxed);
        }
        if bytes != 0 {
            self.bytes_read.fetch_add(bytes, Ordering::Relaxed);
        }
    }

    /// Records one index scan being initiated against this table.
    #[inline]
    pub fn record_idx_scan(&self) {
        self.idx_scan.fetch_add(1, Ordering::Relaxed);
    }

    /// Records one batch produced by an index scan: the rows fetched and the
    /// bytes of table data read to fetch them.
    #[inline]
    pub fn record_idx_batch(&self, tuples: u64, bytes: u64) {
        if tuples != 0 {
            self.idx_tup_fetch.fetch_add(tuples, Ordering::Relaxed);
        }
        if bytes != 0 {
            self.bytes_read.fetch_add(bytes, Ordering::Relaxed);
        }
    }

    /// Records bytes read outside a scan batch, such as a point lookup that
    /// resolves a single row locator.
    #[inline]
    pub fn record_bytes_read(&self, bytes: u64) {
        if bytes != 0 {
            self.bytes_read.fetch_add(bytes, Ordering::Relaxed);
        }
    }

    /// Records rows inserted by one DML batch.
    #[inline]
    pub fn record_inserts(&self, rows: u64) {
        if rows != 0 {
            self.n_tup_ins.fetch_add(rows, Ordering::Relaxed);
        }
    }

    /// Records rows updated by one DML batch. Each updated row leaves a dead
    /// version behind until vacuum reclaims it, so the dead count moves with it.
    #[inline]
    pub fn record_updates(&self, rows: u64) {
        if rows != 0 {
            self.n_tup_upd.fetch_add(rows, Ordering::Relaxed);
            self.n_dead_tup.fetch_add(rows, Ordering::Relaxed);
        }
    }

    /// Records rows deleted by one DML batch. Each deleted row leaves a dead
    /// version behind until vacuum reclaims it.
    #[inline]
    pub fn record_deletes(&self, rows: u64) {
        if rows != 0 {
            self.n_tup_del.fetch_add(rows, Ordering::Relaxed);
            self.n_dead_tup.fetch_add(rows, Ordering::Relaxed);
        }
    }

    /// Records a completed vacuum pass, clearing the dead-row estimate. Called
    /// after the pass commits, so a crash mid-pass leaves the estimate high
    /// rather than falsely clean.
    /// Monotonic count of row writes against this table, for a background
    /// worker's has-anything-changed gate. Inserts, updates and deletes all
    /// count, the gate needs change detection rather than a breakdown
    pub fn write_activity(&self) -> u64 {
        self.n_tup_ins.load(Ordering::Relaxed)
            + self.n_tup_upd.load(Ordering::Relaxed)
            + self.n_tup_del.load(Ordering::Relaxed)
    }

    pub fn record_vacuum(&self, epochSecs: u64) {
        self.n_dead_tup.store(0, Ordering::Relaxed);
        self.last_vacuum.store(epochSecs, Ordering::Relaxed);
    }

    /// Records a completed analyze pass.
    pub fn record_analyze(&self, epochSecs: u64) {
        self.last_analyze.store(epochSecs, Ordering::Relaxed);
    }

    /// Rows inserted less rows deleted, floored at zero. Used as the live-row
    /// estimate for a table that has never been analyzed.
    pub fn observed_live_rows(&self) -> u64 {
        self.n_tup_ins
            .load(Ordering::Relaxed)
            .saturating_sub(self.n_tup_del.load(Ordering::Relaxed))
    }
}

impl Default for TableIOStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Concurrent registry mapping table ids to their IO stats.
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

    /// Returns the stats for the given table id, creating the entry if absent.
    ///
    /// Operators resolve this once when they are constructed and hold the Arc,
    /// so the hash lookup never appears on a batch path.
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

    /// Returns the stats for the given table id without creating an entry.
    pub fn get(&self, tableId: u32) -> Option<Arc<TableIOStats>> {
        self.stats.read_sync(&tableId, |_, v| Arc::clone(v))
    }

    /// Drops the entry for a table. Called when the table is dropped so a
    /// later table reusing the id does not inherit its counters.
    pub fn remove(&self, tableId: u32) {
        self.stats.remove_sync(&tableId);
    }

    /// Iterates over all entries, calling f(table_id, stats) for each one.
    pub fn for_each<F: FnMut(u32, &TableIOStats)>(&self, mut f: F) {
        self.stats.iter_sync(|k, v| {
            f(*k, v.as_ref());
            true
        });
    }
}

impl Default for TableIOStatsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-index IO counters.
pub struct IndexIOStats {
    /// Scans initiated on this index.
    pub idx_scan: AtomicU64,
    /// Index entries examined during scans, before the heap fetch.
    pub idx_tup_read: AtomicU64,
    /// Table rows fetched through this index.
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

    /// Records one scan being initiated on this index.
    #[inline]
    pub fn record_scan(&self) {
        self.idx_scan.fetch_add(1, Ordering::Relaxed);
    }

    /// Records one batch of index work: entries examined and rows fetched.
    #[inline]
    pub fn record_batch(&self, entriesRead: u64, rowsFetched: u64) {
        if entriesRead != 0 {
            self.idx_tup_read.fetch_add(entriesRead, Ordering::Relaxed);
        }
        if rowsFetched != 0 {
            self.idx_tup_fetch.fetch_add(rowsFetched, Ordering::Relaxed);
        }
    }
}

impl Default for IndexIOStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Concurrent registry mapping index ids to their IO stats.
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

    /// Returns the stats for the given index id, creating the entry if absent.
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

    /// Returns the stats for the given index id without creating an entry.
    pub fn get(&self, indexId: u32) -> Option<Arc<IndexIOStats>> {
        self.stats.read_sync(&indexId, |_, v| Arc::clone(v))
    }

    /// Drops the entry for an index. Called when the index is dropped so a
    /// later index reusing the id does not inherit its counters.
    pub fn remove(&self, indexId: u32) {
        self.stats.remove_sync(&indexId);
    }

    /// Iterates over all entries, calling f(index_id, stats) for each one.
    pub fn for_each<F: FnMut(u32, &IndexIOStats)>(&self, mut f: F) {
        self.stats.iter_sync(|k, v| {
            f(*k, v.as_ref());
            true
        });
    }
}

impl Default for IndexIOStatsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_table_counters_accumulate_per_batch() {
        let stats = TableIOStats::new();

        stats.record_seq_scan();
        stats.record_seq_scan();
        stats.record_seq_batch(100, 16384);
        stats.record_seq_batch(40, 16384);
        stats.record_idx_scan();
        stats.record_idx_batch(50, 8192);
        stats.record_inserts(3);
        stats.record_updates(1);
        stats.record_deletes(1);

        assert_eq!(stats.seq_scan.load(Ordering::Relaxed), 2);
        assert_eq!(stats.seq_tup_read.load(Ordering::Relaxed), 140);
        assert_eq!(stats.idx_scan.load(Ordering::Relaxed), 1);
        assert_eq!(stats.idx_tup_fetch.load(Ordering::Relaxed), 50);
        assert_eq!(stats.n_tup_ins.load(Ordering::Relaxed), 3);
        assert_eq!(stats.n_tup_upd.load(Ordering::Relaxed), 1);
        assert_eq!(stats.n_tup_del.load(Ordering::Relaxed), 1);
        // Sequential and index reads accumulate into the same byte counter.
        assert_eq!(stats.bytes_read.load(Ordering::Relaxed), 16384 * 2 + 8192);
    }

    #[test]
    fn test_update_and_delete_both_leave_dead_rows() {
        let stats = TableIOStats::new();

        stats.record_updates(4);
        stats.record_deletes(3);
        assert_eq!(stats.n_dead_tup.load(Ordering::Relaxed), 7);

        stats.record_vacuum(1700000000);
        assert_eq!(stats.n_dead_tup.load(Ordering::Relaxed), 0);
        assert_eq!(stats.last_vacuum.load(Ordering::Relaxed), 1700000000);
    }

    #[test]
    fn test_observed_live_rows_floors_at_zero() {
        let stats = TableIOStats::new();

        stats.record_inserts(10);
        stats.record_deletes(4);
        assert_eq!(stats.observed_live_rows(), 6);

        // A delete of rows this process never saw inserted must not underflow.
        stats.record_deletes(100);
        assert_eq!(stats.observed_live_rows(), 0);
    }

    #[test]
    fn test_zero_valued_records_leave_counters_untouched() {
        let stats = TableIOStats::new();

        stats.record_seq_batch(0, 0);
        stats.record_idx_batch(0, 0);
        stats.record_inserts(0);
        stats.record_updates(0);
        stats.record_deletes(0);

        assert_eq!(stats.seq_tup_read.load(Ordering::Relaxed), 0);
        assert_eq!(stats.bytes_read.load(Ordering::Relaxed), 0);
        assert_eq!(stats.n_dead_tup.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_registry_returns_one_allocation_per_table() {
        let registry = TableIOStatsRegistry::new();

        let statsA = registry.get_or_create(42);
        let statsB = registry.get_or_create(42);
        assert!(Arc::ptr_eq(&statsA, &statsB));

        let statsC = registry.get_or_create(99);
        assert!(!Arc::ptr_eq(&statsA, &statsC));
    }

    #[test]
    fn test_registry_get_does_not_create() {
        let registry = TableIOStatsRegistry::new();

        assert!(registry.get(7).is_none());
        registry.get_or_create(7);
        assert!(registry.get(7).is_some());
    }

    #[test]
    fn test_registry_remove_drops_the_counters() {
        let registry = TableIOStatsRegistry::new();

        registry.get_or_create(5).record_inserts(9);
        registry.remove(5);

        assert_eq!(
            registry.get_or_create(5).n_tup_ins.load(Ordering::Relaxed),
            0
        );
    }

    #[test]
    fn test_registry_for_each_visits_every_table() {
        let registry = TableIOStatsRegistry::new();

        registry.get_or_create(1).record_inserts(2);
        registry.get_or_create(2).record_inserts(3);

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
    fn test_index_counters_accumulate() {
        let stats = IndexIOStats::new();

        stats.record_scan();
        stats.record_scan();
        stats.record_scan();
        stats.record_batch(200, 150);

        assert_eq!(stats.idx_scan.load(Ordering::Relaxed), 3);
        assert_eq!(stats.idx_tup_read.load(Ordering::Relaxed), 200);
        assert_eq!(stats.idx_tup_fetch.load(Ordering::Relaxed), 150);
    }

    #[test]
    fn test_index_registry_returns_one_allocation_per_index() {
        let registry = IndexIOStatsRegistry::new();

        let statsA = registry.get_or_create(3);
        let statsB = registry.get_or_create(3);
        assert!(Arc::ptr_eq(&statsA, &statsB));

        registry.remove(3);
        assert!(registry.get(3).is_none());
    }
}

//! MVCC garbage collector for dead tuple reclamation.
//!
//! Scans heap pages to identify tuples that are no longer visible to any
//! active transaction (xmax != 0 and xmax < oldest active txn). Reclaimable
//! tuples have their slot lengths zeroed, freeing space for new inserts.
//! Runs as a background thread per the lock-free-first policy (Mutex
//! acceptable for single-owner background threads).

/// Statistics from a GC sweep.
#[derive(Debug, Clone, Default)]
pub struct GcStats {
    /// Number of heap pages scanned.
    pub pages_scanned: u64,
    /// Number of dead tuples reclaimed.
    pub tuples_reclaimed: u64,
    /// Total bytes freed by reclaiming dead tuples.
    pub bytes_freed: u64,
}

impl GcStats {
    /// Returns true if any tuples were reclaimed.
    pub fn has_reclaimed(&self) -> bool {
        self.tuples_reclaimed > 0
    }
}

impl std::fmt::Display for GcStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GC: scanned {} pages, reclaimed {} tuples, freed {} bytes",
            self.pages_scanned, self.tuples_reclaimed, self.bytes_freed
        )
    }
}

/// MVCC garbage collector configuration.
#[derive(Debug, Clone)]
pub struct GcConfig {
    /// Minimum ratio of dead tuples to total tuples before triggering GC on a page.
    /// Default: 0.2 (20%).
    pub dead_ratio_threshold: f64,
}

impl Default for GcConfig {
    fn default() -> Self {
        Self {
            dead_ratio_threshold: 0.2,
        }
    }
}

impl GcConfig {
    /// Returns true if GC should process this page based on the dead tuple ratio.
    pub fn should_gc(&self, total_tuples: u64, dead_tuples: u64) -> bool {
        if total_tuples == 0 {
            return false;
        }
        (dead_tuples as f64 / total_tuples as f64) >= self.dead_ratio_threshold
    }
}

/// MVCC garbage collector.
///
/// Identifies and reclaims dead tuples that are no longer visible to any
/// active transaction. The oldest active transaction ID determines the
/// reclamation boundary: any tuple with xmax < oldest_active is safe to reclaim.
///
/// This struct provides the core GC logic and statistics. Wiring of the
/// actual reclamation pass through TransactionManager, DiskManager, and
/// BufferPool is done by callers that drive the GC loop.
pub struct MvccGc {
    config: GcConfig,
}

impl MvccGc {
    /// Creates a new MVCC garbage collector with default configuration.
    pub fn new() -> Self {
        Self {
            config: GcConfig::default(),
        }
    }

    /// Creates a new MVCC garbage collector with the given configuration.
    pub fn with_config(config: GcConfig) -> Self {
        Self { config }
    }

    /// Returns the oldest active transaction ID from a sorted list of active txn_ids.
    /// Returns None if no transactions are active (all tuples are reclaimable).
    /// The input slice is expected to be sorted (as produced by Snapshot::active_txn_ids).
    pub fn oldest_active_txn(active_txn_ids: &[u64]) -> Option<u64> {
        active_txn_ids.first().copied()
    }

    /// Determines if a tuple is reclaimable given the oldest active transaction.
    ///
    /// A tuple is reclaimable when:
    /// - xmax != 0 (tuple has been deleted/updated)
    /// - xmax < oldest_active (the deleting transaction committed and is no longer
    ///   visible to any active transaction)
    #[inline]
    pub fn is_reclaimable(xmax: u32, oldest_active: u64) -> bool {
        xmax != 0 && (xmax as u64) < oldest_active
    }

    /// Determines if a tuple is reclaimable when no transactions are active.
    /// All deleted tuples are reclaimable.
    #[inline]
    pub fn is_reclaimable_no_active(xmax: u32) -> bool {
        xmax != 0
    }

    /// Returns true if a page should be GC'd based on its dead tuple ratio.
    pub fn should_gc_page(&self, total_tuples: u64, dead_tuples: u64) -> bool {
        self.config.should_gc(total_tuples, dead_tuples)
    }

    /// Returns a reference to the GC configuration.
    pub fn config(&self) -> &GcConfig {
        &self.config
    }
}

impl Default for MvccGc {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_oldest_active_txn() {
        // Input must be sorted (as produced by Snapshot::active_txn_ids)
        assert_eq!(MvccGc::oldest_active_txn(&[3, 5, 8, 10]), Some(3));
        assert_eq!(MvccGc::oldest_active_txn(&[100]), Some(100));
        assert_eq!(MvccGc::oldest_active_txn(&[]), None);
    }

    #[test]
    fn test_is_reclaimable() {
        // xmax=0: not deleted, never reclaimable
        assert!(!MvccGc::is_reclaimable(0, 100));

        // xmax=5 < oldest_active=10: reclaimable
        assert!(MvccGc::is_reclaimable(5, 10));

        // xmax=10 == oldest_active=10: NOT reclaimable (txn 10 might still see it)
        assert!(!MvccGc::is_reclaimable(10, 10));

        // xmax=15 > oldest_active=10: NOT reclaimable
        assert!(!MvccGc::is_reclaimable(15, 10));
    }

    #[test]
    fn test_is_reclaimable_no_active() {
        // No active transactions means all deleted tuples are reclaimable
        assert!(MvccGc::is_reclaimable_no_active(1));
        assert!(MvccGc::is_reclaimable_no_active(u32::MAX));
        assert!(!MvccGc::is_reclaimable_no_active(0)); // Not deleted
    }

    #[test]
    fn test_should_gc_page() {
        let gc = MvccGc::new();

        // 0 total: never GC
        assert!(!gc.should_gc_page(0, 0));

        // 20% dead: at threshold, should GC
        assert!(gc.should_gc_page(100, 20));

        // 25% dead: above threshold
        assert!(gc.should_gc_page(100, 25));

        // 10% dead: below threshold
        assert!(!gc.should_gc_page(100, 10));

        // 19% dead: below threshold
        assert!(!gc.should_gc_page(100, 19));
    }

    #[test]
    fn test_custom_gc_config() {
        let config = GcConfig {
            dead_ratio_threshold: 0.5,
        };
        let gc = MvccGc::with_config(config);

        // 50% threshold
        assert!(!gc.should_gc_page(100, 49));
        assert!(gc.should_gc_page(100, 50));
        assert!(gc.should_gc_page(100, 51));
    }

    #[test]
    fn test_gc_stats_display() {
        let stats = GcStats {
            pages_scanned: 42,
            tuples_reclaimed: 100,
            bytes_freed: 4096,
        };
        let display = stats.to_string();
        assert!(display.contains("42 pages"));
        assert!(display.contains("100 tuples"));
        assert!(display.contains("4096 bytes"));
    }

    #[test]
    fn test_gc_stats_has_reclaimed() {
        let empty = GcStats::default();
        assert!(!empty.has_reclaimed());

        let with_work = GcStats {
            pages_scanned: 10,
            tuples_reclaimed: 5,
            bytes_freed: 1024,
        };
        assert!(with_work.has_reclaimed());
    }
}

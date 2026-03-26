//! Mutation tracking and column correlation detection for auto-analyze.
//!
//! MutationTracker monitors insert, update, and delete counts per table
//! to trigger automatic re-analysis when 10% of rows have changed.
//! Column correlation detection identifies correlated columns for
//! multi-column cardinality adjustments.

use std::sync::atomic::{AtomicU64, Ordering};
use zyron_catalog::ColumnId;

// ---------------------------------------------------------------------------
// Auto-analyze threshold
// ---------------------------------------------------------------------------

/// Fraction of rows that must change before auto-analyze triggers.
const AUTO_ANALYZE_THRESHOLD: f64 = 0.10;

// ---------------------------------------------------------------------------
// Mutation tracker
// ---------------------------------------------------------------------------

/// Tracks row mutation counts per table for auto-analyze decisions.
/// All counters use relaxed atomic ordering for lock-free updates
/// since exact precision is not required for triggering analysis.
pub struct MutationTracker {
    inserts: AtomicU64,
    updates: AtomicU64,
    deletes: AtomicU64,
    last_analyzed_row_count: AtomicU64,
}

impl MutationTracker {
    pub fn new(initial_row_count: u64) -> Self {
        Self {
            inserts: AtomicU64::new(0),
            updates: AtomicU64::new(0),
            deletes: AtomicU64::new(0),
            last_analyzed_row_count: AtomicU64::new(initial_row_count),
        }
    }

    /// Records inserts to the tracked table.
    pub fn record_insert(&self, count: u64) {
        self.inserts.fetch_add(count, Ordering::Relaxed);
    }

    /// Records updates to the tracked table.
    pub fn record_update(&self, count: u64) {
        self.updates.fetch_add(count, Ordering::Relaxed);
    }

    /// Records deletes from the tracked table.
    pub fn record_delete(&self, count: u64) {
        self.deletes.fetch_add(count, Ordering::Relaxed);
    }

    /// Returns true if enough mutations have accumulated to justify re-analysis.
    /// Triggers when total mutations exceed 10% of the last analyzed row count.
    pub fn should_analyze(&self) -> bool {
        let total_mutations = self.inserts.load(Ordering::Relaxed)
            + self.updates.load(Ordering::Relaxed)
            + self.deletes.load(Ordering::Relaxed);
        let last_count = self.last_analyzed_row_count.load(Ordering::Relaxed);

        if last_count == 0 {
            return total_mutations > 0;
        }

        (total_mutations as f64 / last_count as f64) > AUTO_ANALYZE_THRESHOLD
    }

    /// Resets mutation counters after analysis completes.
    pub fn reset(&self, current_row_count: u64) {
        self.inserts.store(0, Ordering::Relaxed);
        self.updates.store(0, Ordering::Relaxed);
        self.deletes.store(0, Ordering::Relaxed);
        self.last_analyzed_row_count
            .store(current_row_count, Ordering::Relaxed);
    }

    /// Returns the total number of mutations since last reset.
    pub fn total_mutations(&self) -> u64 {
        self.inserts.load(Ordering::Relaxed)
            + self.updates.load(Ordering::Relaxed)
            + self.deletes.load(Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// Column correlation detection
// ---------------------------------------------------------------------------

/// A detected correlation between two columns.
#[derive(Debug, Clone)]
pub struct ColumnCorrelation {
    pub column_a: ColumnId,
    pub column_b: ColumnId,
    /// Pearson correlation coefficient in range [-1.0, 1.0].
    /// Values near +/-1.0 indicate strong correlation.
    pub coefficient: f64,
}

/// Detects correlated column pairs from sampled data.
/// Uses ordinal encoding (byte-comparison rank) to compute Pearson correlation.
/// Returns pairs with |correlation| > 0.7 (strong correlation).
///
/// `samples` is indexed as samples[column_index][row_index].
/// `column_ids` maps column_index to ColumnId.
pub fn detect_correlations(
    samples: &[Vec<Vec<u8>>],
    column_ids: &[ColumnId],
) -> Vec<ColumnCorrelation> {
    if samples.len() < 2 || column_ids.len() < 2 {
        return Vec::new();
    }

    let n_cols = samples.len().min(column_ids.len());
    let n_rows = samples.iter().map(|col| col.len()).min().unwrap_or(0);
    if n_rows < 10 {
        return Vec::new();
    }

    // Convert columns to ordinal ranks for Pearson computation
    let ranks: Vec<Vec<f64>> = samples[..n_cols]
        .iter()
        .map(|col| ordinal_ranks(&col[..n_rows]))
        .collect();

    let mut correlations = Vec::new();

    for i in 0..n_cols {
        for j in (i + 1)..n_cols {
            let coeff = pearson_correlation(&ranks[i], &ranks[j]);
            if coeff.abs() > 0.7 {
                correlations.push(ColumnCorrelation {
                    column_a: column_ids[i],
                    column_b: column_ids[j],
                    coefficient: coeff,
                });
            }
        }
    }

    correlations
}

/// Converts byte-encoded values into ordinal ranks (0-based position in sorted order).
fn ordinal_ranks(values: &[Vec<u8>]) -> Vec<f64> {
    let n = values.len();
    let mut indexed: Vec<(usize, &Vec<u8>)> = values.iter().enumerate().collect();
    indexed.sort_by(|a, b| a.1.cmp(b.1));

    let mut ranks = vec![0.0; n];
    for (rank, (original_idx, _)) in indexed.iter().enumerate() {
        ranks[*original_idx] = rank as f64;
    }
    ranks
}

/// Computes Pearson correlation coefficient between two rank vectors.
fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n == 0 {
        return 0.0;
    }

    let n_f = n as f64;
    let mean_x: f64 = x[..n].iter().sum::<f64>() / n_f;
    let mean_y: f64 = y[..n].iter().sum::<f64>() / n_f;

    let mut cov = 0.0_f64;
    let mut var_x = 0.0_f64;
    let mut var_y = 0.0_f64;

    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom < 1e-15 {
        return 0.0;
    }

    cov / denom
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mutation_tracker_new_table() {
        let tracker = MutationTracker::new(0);
        // Empty table should trigger analyze on any mutation
        assert!(!tracker.should_analyze());
        tracker.record_insert(1);
        assert!(tracker.should_analyze());
    }

    #[test]
    fn test_mutation_tracker_threshold() {
        let tracker = MutationTracker::new(1000);
        // 50 mutations on 1000 rows = 5%, below threshold
        tracker.record_insert(50);
        assert!(!tracker.should_analyze());
        // 60 more = 110 total = 11%, above threshold
        tracker.record_update(60);
        assert!(tracker.should_analyze());
    }

    #[test]
    fn test_mutation_tracker_reset() {
        let tracker = MutationTracker::new(1000);
        tracker.record_insert(200);
        assert!(tracker.should_analyze());
        tracker.reset(1200);
        assert!(!tracker.should_analyze());
        assert_eq!(tracker.total_mutations(), 0);
    }

    #[test]
    fn test_mutation_tracker_mixed_ops() {
        let tracker = MutationTracker::new(1000);
        tracker.record_insert(30);
        tracker.record_update(30);
        tracker.record_delete(41);
        // Total = 101, 10.1% > 10% threshold
        assert!(tracker.should_analyze());
    }

    #[test]
    fn test_detect_correlations_perfect() {
        // Columns with identical ordering should correlate perfectly
        let col_a: Vec<Vec<u8>> = (0u64..100).map(|i| i.to_be_bytes().to_vec()).collect();
        let col_b: Vec<Vec<u8>> = (0u64..100)
            .map(|i| (i * 2).to_be_bytes().to_vec())
            .collect();
        let samples = vec![col_a, col_b];
        let ids = vec![ColumnId(0), ColumnId(1)];

        let correlations = detect_correlations(&samples, &ids);
        assert_eq!(correlations.len(), 1);
        assert!(correlations[0].coefficient > 0.99);
    }

    #[test]
    fn test_detect_correlations_uncorrelated() {
        // Random-looking columns should not correlate
        let col_a: Vec<Vec<u8>> = (0u64..100).map(|i| i.to_be_bytes().to_vec()).collect();
        let col_b: Vec<Vec<u8>> = (0u64..100)
            .map(|i| ((i * 37 + 13) % 100).to_be_bytes().to_vec())
            .collect();
        let samples = vec![col_a, col_b];
        let ids = vec![ColumnId(0), ColumnId(1)];

        let correlations = detect_correlations(&samples, &ids);
        // Should not detect strong correlation
        for c in &correlations {
            assert!(c.coefficient.abs() <= 0.7);
        }
    }

    #[test]
    fn test_detect_correlations_too_few_rows() {
        let col_a = vec![vec![1u8], vec![2u8]];
        let col_b = vec![vec![3u8], vec![4u8]];
        let samples = vec![col_a, col_b];
        let ids = vec![ColumnId(0), ColumnId(1)];

        let correlations = detect_correlations(&samples, &ids);
        assert!(correlations.is_empty());
    }

    #[test]
    fn test_pearson_correlation_identical() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let r = pearson_correlation(&x, &x);
        assert!((r - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_pearson_correlation_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - (-1.0)).abs() < 0.001);
    }
}

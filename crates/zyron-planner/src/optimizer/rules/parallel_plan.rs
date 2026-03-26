//! Parallel plan cost estimation and worker count computation.
//!
//! This module provides helper functions used by the physical plan builder
//! to decide when and how to parallelize scan and join operators.
//! It is not an OptimizationRule (operates on physical plans, not logical).

/// Minimum estimated row count to consider parallel execution.
const PARALLEL_ROW_THRESHOLD: f64 = 100_000.0;

/// Minimum number of pages per worker partition.
const MIN_PAGES_PER_WORKER: u32 = 64;

/// Returns true if the estimated row count justifies parallel execution.
pub fn should_parallelize(row_count: f64) -> bool {
    row_count > PARALLEL_ROW_THRESHOLD
}

/// Computes the optimal number of parallel workers based on table page count.
/// Uses min(available_cores / 2, page_count / MIN_PAGES_PER_WORKER),
/// clamped to [1, 16].
pub fn compute_worker_count(page_count: u32) -> usize {
    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let max_by_cores = available / 2;
    let max_by_pages = (page_count / MIN_PAGES_PER_WORKER) as usize;
    max_by_cores.min(max_by_pages).clamp(1, 16)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_parallelize_small() {
        assert!(!should_parallelize(50_000.0));
    }

    #[test]
    fn test_should_parallelize_large() {
        assert!(should_parallelize(200_000.0));
    }

    #[test]
    fn test_should_parallelize_threshold() {
        assert!(!should_parallelize(100_000.0));
        assert!(should_parallelize(100_001.0));
    }

    #[test]
    fn test_worker_count_minimum() {
        // Very small table: should get at least 1 worker
        assert_eq!(compute_worker_count(10), 1);
    }

    #[test]
    fn test_worker_count_scales() {
        // 1000 pages / 64 = 15 workers max by pages
        let workers = compute_worker_count(1000);
        assert!(workers >= 1);
        assert!(workers <= 16);
    }

    #[test]
    fn test_worker_count_capped() {
        // Even with huge page count, should not exceed 16
        let workers = compute_worker_count(100_000);
        assert!(workers <= 16);
    }
}

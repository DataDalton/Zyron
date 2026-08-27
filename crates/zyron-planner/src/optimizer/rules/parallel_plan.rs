//! Parallel plan worker count.
//!
//! Used by the physical plan builder to decide when and how to parallelize
//! scan and join operators. Not an OptimizationRule (operates on physical
//! plans, not logical).
//!
//! The worker count is not a property of the machine. It is a property of the
//! machine and everything else running on it: a plan built while the node is
//! idle can use the whole box, and the same plan built while fifty other
//! queries hold the parallel budget cannot. Both answers come from the same
//! account in `zyron_pressure::pressure::ParallelCapacity`, which the executor
//! spends against, so what the planner promises and what the executor can
//! deliver are the same number.

use zyron_pressure::pressure::ParallelCapacity;

/// Minimum estimated row count to consider parallel execution.
const PARALLEL_ROW_THRESHOLD: f64 = 100_000.0;

/// Minimum number of pages per worker partition. Below this the split costs
/// more in setup than the extra worker returns.
const MIN_PAGES_PER_WORKER: u32 = 64;

/// Returns true if the estimated row count justifies parallel execution.
pub fn should_parallelize(row_count: f64) -> bool {
    row_count > PARALLEL_ROW_THRESHOLD
}

/// Workers to plan for, given how much data there is to divide and how much of
/// the machine is currently free.
///
/// The page count bounds what the data can usefully be split into. The
/// capacity account bounds what the node can afford, and it is read at
/// plan-final time rather than being a constant, so the same query answers
/// differently under load.
pub fn compute_worker_count(page_count: u32) -> usize {
    compute_worker_count_against(page_count, ParallelCapacity::global())
}

/// The decision against a given capacity account, so a test can supply one of
/// a known size rather than depending on the machine it runs on.
pub fn compute_worker_count_against(page_count: u32, capacity: &ParallelCapacity) -> usize {
    let by_pages = (page_count / MIN_PAGES_PER_WORKER) as usize;
    if by_pages <= 1 {
        return 1;
    }
    capacity.advise(by_pages)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_row_counts_stay_serial() {
        assert!(!should_parallelize(50_000.0));
        assert!(should_parallelize(200_000.0));
        assert!(!should_parallelize(PARALLEL_ROW_THRESHOLD));
        assert!(should_parallelize(PARALLEL_ROW_THRESHOLD + 1.0));
    }

    #[test]
    fn a_table_too_small_to_divide_gets_one_worker() {
        let capacity = ParallelCapacity::with_total(32);
        assert_eq!(compute_worker_count_against(10, &capacity), 1);
        assert_eq!(compute_worker_count_against(0, &capacity), 1);
        // Exactly one worker's worth of pages is still one worker
        assert_eq!(
            compute_worker_count_against(MIN_PAGES_PER_WORKER, &capacity),
            1
        );
    }

    #[test]
    fn the_page_count_bounds_the_split() {
        let capacity = ParallelCapacity::with_total(64);
        // Four workers' worth of pages asks for four, not for the machine
        assert_eq!(
            compute_worker_count_against(MIN_PAGES_PER_WORKER * 4, &capacity),
            4
        );
    }

    #[test]
    fn capacity_bounds_the_split_when_the_data_would_allow_more() {
        let capacity = ParallelCapacity::with_total(8);
        // A thousand workers' worth of pages cannot exceed the account
        assert_eq!(
            compute_worker_count_against(MIN_PAGES_PER_WORKER * 1000, &capacity),
            8
        );
    }

    /// The behaviour the old core-count constant could not express: a plan
    /// built while the node is busy asks for less than the same plan built
    /// while it is idle.
    #[test]
    fn a_busy_node_plans_a_narrower_split() {
        let capacity = ParallelCapacity::with_total(16);
        let pages = MIN_PAGES_PER_WORKER * 64;
        assert_eq!(compute_worker_count_against(pages, &capacity), 16);

        let taken = capacity.try_take(12);
        assert_eq!(taken, 12);
        assert_eq!(compute_worker_count_against(pages, &capacity), 4);

        capacity.give_back(taken);
        assert_eq!(compute_worker_count_against(pages, &capacity), 16);
    }

    #[test]
    fn a_saturated_node_still_plans_one_worker_not_zero() {
        let capacity = ParallelCapacity::with_total(4);
        let taken = capacity.try_take(4);
        assert_eq!(taken, 4);
        assert_eq!(
            compute_worker_count_against(MIN_PAGES_PER_WORKER * 100, &capacity),
            1
        );
        capacity.give_back(taken);
    }

    #[test]
    fn the_controller_scale_narrows_the_split() {
        let capacity = ParallelCapacity::with_total(16);
        let pages = MIN_PAGES_PER_WORKER * 64;
        assert_eq!(compute_worker_count_against(pages, &capacity), 16);
        capacity.set_scale_pct(25);
        assert_eq!(compute_worker_count_against(pages, &capacity), 4);
        capacity.set_scale_pct(100);
    }

    #[test]
    fn the_process_account_answers_within_its_own_bounds() {
        let workers = compute_worker_count(MIN_PAGES_PER_WORKER * 4096);
        let capacity = ParallelCapacity::global();
        assert!(workers >= 1);
        assert!(workers <= capacity.total() as usize);
    }
}

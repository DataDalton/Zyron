//! GDPR right-to-erasure. Traverses owner/PII columns across the catalog and,
//! crucially, also scrubs system-versioned history and CDF so erased data does
//! not survive in time-travel. A legal hold on any matched live OR historical
//! row aborts the whole operation. The actual per-table mutation is delegated
//! to an injected executor so this crate stays decoupled from zyron-executor.

use zyron_common::{Result, ZyronError};

/// What to do with matched data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErasureAction {
    Delete,
    Anonymize,
    /// Destroy the per-subject crypto key (instant logical erasure).
    CryptoShred,
}

/// One table to scrub: live rows + its history table + CDF stream.
#[derive(Debug, Clone)]
pub struct ErasureTarget {
    pub table_id: u32,
    pub table_name: String,
    /// SQL predicate selecting the subject's rows in this table.
    pub predicate_sql: String,
    pub history_table_id: Option<u32>,
    pub cdf_enabled: bool,
}

#[derive(Debug, Clone, Default)]
pub struct ErasureResult {
    pub tables_affected: u64,
    pub live_rows: u64,
    pub history_rows: u64,
    pub cdf_rows: u64,
}

/// Performs the storage mutation for one target inside the caller's
/// transaction. Implemented by the dispatch/executor bridge.
pub trait ErasureExecutor {
    /// Returns true if any row matched by `predicate_sql` on `table_id`
    /// (live OR history) is under an active legal hold.
    fn any_under_hold(&self, target: &ErasureTarget) -> Result<bool>;

    /// Scrubs the subject's rows for one target. Returns (live, history, cdf)
    /// row counts touched.
    fn scrub_target(
        &self,
        target: &ErasureTarget,
        action: ErasureAction,
        dry_run: bool,
    ) -> Result<(u64, u64, u64)>;
}

/// Runs erasure across all targets in one logical operation. If any target
/// has held rows the whole erasure is rejected before any mutation (legal
/// hold supersedes erasure).
pub fn forget_user(
    targets: &[ErasureTarget],
    action: ErasureAction,
    dry_run: bool,
    exec: &dyn ErasureExecutor,
) -> Result<ErasureResult> {
    // Phase 1: hold check across every target (live + history).
    for t in targets {
        if exec.any_under_hold(t)? {
            return Err(ZyronError::LegalHoldViolation(format!(
                "erasure blocked: rows in '{}' are under an active legal hold",
                t.table_name
            )));
        }
    }
    // Phase 2: scrub. Caller runs this inside a single transaction.
    let mut result = ErasureResult::default();
    for t in targets {
        let (live, hist, cdf) = exec.scrub_target(t, action, dry_run)?;
        result.tables_affected += 1;
        result.live_rows += live;
        result.history_rows += hist;
        result.cdf_rows += cdf;
    }
    Ok(result)
}

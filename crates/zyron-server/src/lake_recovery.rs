//! Startup recovery for ZyronLake transaction logs.
//!
//! Runs once per boot, after WAL recovery has rebuilt the commit-status map
//! and before any worker or query touches a lake table. Opening a log
//! discards version files whose enclosing database transaction never
//! committed, together with every version built after one, then the shared
//! registry is primed so scans and commits share the reconciled head.
//!
//! The deployment mode gates the whole pass. A `db` node opens no log and
//! caches nothing, which is the point of running that mode: the lake tier
//! costs it neither startup IO nor resident state.

use std::path::Path;
use std::sync::Arc;

use zyron_catalog::Catalog;
use zyron_common::DeploymentMode;
use zyron_lake::{CommitStatus, IntentAware, LakePaths, TransactionLog};

/// What the startup pass did, reported by the caller and asserted by tests.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct LakeRecoveryReport {
    /// Logs opened, reconciled and registered for query and commit
    pub recovered: usize,
    /// Tables whose log failed to open, unusable until repaired
    pub failed: usize,
    /// Lake tables left closed because this node does not run the lake tier
    pub skipped: usize,
    /// Cross-table transactions whose commit point had landed
    pub intents_committed: usize,
    /// Cross-table transactions that never reached their commit point, so
    /// every participant's version is discarded
    pub intents_discarded: usize,
    /// Lake roots a conversion wrote and never flipped the catalog for,
    /// removed because no reachable manifest names their files
    pub orphan_roots_reclaimed: usize,
}

/// Opens and registers the transaction log of every lake table in the
/// catalog. A table whose log fails to open is loudly logged and the rest of
/// the server still comes up, because one unreadable table must not take the
/// node with it.
///
/// On a node that does not run the lake tier the pass opens nothing and
/// reports the lake tables it left closed, so a mode changed under a
/// populated data directory is visible in the startup log rather than as a
/// query failure later.
pub fn recover_lake_logs(
    mode: DeploymentMode,
    catalog: &Catalog,
    data_dir: &Path,
    status: &dyn CommitStatus,
) -> LakeRecoveryReport {
    let mut report = LakeRecoveryReport::default();

    if !mode.runs_lake_tier() {
        let closed: Vec<String> = catalog
            .list_all_tables()
            .into_iter()
            .filter(|t| t.lake.is_lake())
            .map(|t| t.name.clone())
            .collect();
        report.skipped = closed.len();
        if !closed.is_empty() {
            tracing::error!(
                mode = mode.as_str(),
                tables = %closed.join(", "),
                "lake tables exist but this node does not run the lake tier, their logs stay closed, \
                 set storage.deployment_mode = \"unified\" to run both formats here"
            );
        }
        return report;
    }

    // Cross-table commit intents answer for their own transactions, and the
    // answer has to be in place before any participant's log opens. The
    // intent files stay on disk until every log has opened, because removing
    // one early would let a later log read "absent means committed" and
    // resurrect versions this pass is discarding
    let intents = match zyron_lake::recover_intents(data_dir) {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(error = %e, "lake commit intents could not be read");
            zyron_lake::IntentRecovery::default()
        }
    };
    report.intents_committed = intents.committed.len();
    report.intents_discarded = intents.discarded.len();
    if !intents.discarded.is_empty() {
        tracing::warn!(
            count = intents.discarded.len(),
            "discarding cross-table transactions that never reached their commit point"
        );
    }
    let status = &IntentAware::new(status, data_dir);

    for table in catalog.list_all_tables() {
        if !table.lake.is_lake() {
            // A lake root under a table the catalog calls heap is a
            // conversion that died before its flip, unless the table kept
            // its history deliberately. Reclaiming it restores the table to
            // the state the flip would have left it in had it never started
            if !table.lake.owns_lake_root() {
                let paths = LakePaths::new(data_dir, table.id.0);
                match zyron_lake::reclaim_orphan_root(&paths) {
                    Ok(true) => {
                        tracing::warn!(
                            table = %table.name,
                            "reclaimed a lake root a conversion left before its catalog flip"
                        );
                        report.orphan_roots_reclaimed += 1;
                    }
                    Ok(false) => {}
                    Err(e) => tracing::error!(
                        table = %table.name,
                        error = %e,
                        "could not reclaim an orphaned lake root"
                    ),
                }
            }
            continue;
        }
        let paths = LakePaths::new(data_dir, table.id.0);
        match TransactionLog::open(paths, status) {
            Ok(log) => {
                tracing::info!(
                    table = %table.name,
                    version = log.latest_version(),
                    "lake table log recovered"
                );
                // A lake table never runs ANALYZE, so its planner statistics
                // come from the recovered manifest before the first query
                if let Ok(manifest) = log.latest_manifest() {
                    zyron_executor::lake_stats::publish_manifest_stats(catalog, &table, &manifest);
                }
                TransactionLog::register_shared(Arc::new(log));
                report.recovered += 1;
            }
            Err(e) => {
                tracing::error!(
                    table = %table.name,
                    error = %e,
                    "lake table log failed to open, the table is unavailable"
                );
                report.failed += 1;
            }
        }
    }

    // Every log has read the intents, so the files have done their job
    if let Err(e) = zyron_lake::clear_recovered_intents(data_dir, &intents) {
        tracing::error!(error = %e, "lake commit intents could not be cleared");
    }

    report
}

// -----------------------------------------------------------------------------
// Adaptive Clustering maintenance.
//
// Runs clustering passes for lake tables whose schedule asks for them. The
// decisions all live in zyron-lake: this worker chooses which table to look
// at, hands the pass what it needs, and records what came back.
//
// Two things it deliberately does not do. It never overrides an operator:
// under Force the declared spec is applied and the gate's verdict is
// reported rather than enforced, and under Auto and Hybrid the gate is
// binding so a pass can only improve a layout or leave it alone. And it
// never blocks a query: passes stage their candidates outside the active
// file set and touch it only through one metadata commit.
// -----------------------------------------------------------------------------

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use tracing::{debug, info, warn};

use zyron_catalog::Catalog;
use zyron_common::{ClusterDecision, ClusterMode, DeploymentMode, LabeledMetrics};
use zyron_lake::{
    ClusterPassOptions, ClusterSpec, Decision, GateConfig, LakePaths, ResumeOptions, TransactionLog,
};

/// Seconds between passes. Clustering is a background rewrite, so it runs
/// on the scale of minutes rather than seconds
pub const DEFAULT_INTERVAL_SECS: u64 = 300;

/// Most keys a proposal may carry. Past four the tail keys refine rows a
/// file already holds together and stop buying pruning
const MAX_PROPOSED_KEYS: usize = 4;

#[derive(Debug, Clone)]
pub struct LakeClusteringConfig {
    pub interval_secs: u64,
    /// Input files one pass rewrites, which is what bounds its memory
    pub max_inputs: usize,
    pub target_rows_per_file: u64,
    pub gate: GateConfig,
    pub data_dir: PathBuf,
}

impl LakeClusteringConfig {
    pub fn new(data_dir: PathBuf) -> Self {
        Self {
            interval_secs: DEFAULT_INTERVAL_SECS,
            max_inputs: zyron_lake::DEFAULT_MAX_INPUTS,
            target_rows_per_file: zyron_lake::DEFAULT_ROWS_PER_FILE,
            gate: GateConfig::default(),
            data_dir,
        }
    }
}

/// What the worker has done since startup.
#[derive(Debug, Default)]
pub struct LakeClusteringStats {
    pub passes: AtomicU64,
    pub accepted: AtomicU64,
    pub refused: AtomicU64,
    pub files_rewritten: AtomicU64,
    pub bytes_written: AtomicU64,
    pub resumed: AtomicU64,
}

pub struct LakeClusteringWorker {
    shutdown: Arc<AtomicBool>,
    stats: Arc<LakeClusteringStats>,
    handle: Option<tokio::task::JoinHandle<()>>,
}

impl LakeClusteringWorker {
    /// Starts the worker, or returns None on a node that does not run the
    /// lake tier. A db node must not pay for a thread it can never use
    pub fn start(
        mode: DeploymentMode,
        catalog: Arc<Catalog>,
        metrics: Option<Arc<LabeledMetrics>>,
        config: LakeClusteringConfig,
    ) -> Option<Self> {
        if !mode.runs_lake_tier() {
            return None;
        }
        let shutdown = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(LakeClusteringStats::default());
        let handle = tokio::spawn(clustering_loop(
            Arc::clone(&shutdown),
            Arc::clone(&stats),
            catalog,
            metrics,
            config,
        ));
        info!("lake clustering worker started");
        Some(Self {
            shutdown,
            stats,
            handle: Some(handle),
        })
    }

    pub fn stats(&self) -> &Arc<LakeClusteringStats> {
        &self.stats
    }

    pub fn shutdown(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        if let Some(handle) = self.handle.take() {
            handle.abort();
        }
    }
}

async fn clustering_loop(
    shutdown: Arc<AtomicBool>,
    stats: Arc<LakeClusteringStats>,
    catalog: Arc<Catalog>,
    metrics: Option<Arc<LabeledMetrics>>,
    config: LakeClusteringConfig,
) {
    // Passes a crash left half done are finished or unwound before any new
    // one starts, so a resumed pass never races a fresh one over the same
    // staging directory
    for (name, log) in lake_tables(&catalog, &config.data_dir) {
        match zyron_lake::resume_cluster_passes(
            &log,
            pass_attempt(),
            &[],
            &ResumeOptions::default(),
        ) {
            Ok(outcomes) => {
                for outcome in outcomes {
                    stats.resumed.fetch_add(1, Ordering::Relaxed);
                    info!(
                        table = %name,
                        pass_id = outcome.pass_id,
                        version = ?outcome.version,
                        "resumed an interrupted clustering pass"
                    );
                }
            }
            Err(e) => warn!(table = %name, error = %e, "clustering resume failed"),
        }
    }

    let mut ticker = tokio::time::interval(Duration::from_secs(config.interval_secs.max(30)));
    let pass_counter = AtomicU64::new(1);
    loop {
        ticker.tick().await;
        if shutdown.load(Ordering::Acquire) {
            break;
        }
        for (name, log) in lake_tables(&catalog, &config.data_dir) {
            if shutdown.load(Ordering::Acquire) {
                break;
            }
            let pass_id = pass_counter.fetch_add(1, Ordering::Relaxed);
            if let Err(e) = run_one_table(&name, &log, pass_id, &stats, metrics.as_deref(), &config)
            {
                warn!(table = %name, error = %e, "clustering pass failed");
            }
        }
    }
}

/// Considers one table and runs a pass when there is something to do.
fn run_one_table(
    name: &str,
    log: &TransactionLog,
    pass_id: u64,
    stats: &LakeClusteringStats,
    metrics: Option<&LabeledMetrics>,
    config: &LakeClusteringConfig,
) -> Result<(), zyron_common::ZyronError> {
    let manifest = log.latest_manifest()?;
    let schedule = manifest.clustering_schedule();
    if let Some(metrics) = metrics {
        metrics.clusteringPolicySet(name, manifest.clustering_mode(), schedule);
    }
    // OnDemand means exactly that: only OPTIMIZE starts a pass
    if !schedule.runs_in_background() {
        return Ok(());
    }
    let Some(table_id) = log.paths().table_id() else {
        return Ok(());
    };

    // Index runs first, because it is cheap to decide and a fragmented
    // index costs every probe until it is collapsed. Each write commit
    // appends its own index file over whatever keys it touched, so their
    // ranges overlap and a probe stops being able to prune to one file.
    // The check reads the manifest and opens nothing
    match zyron_lake::operations::compact_indexes_if_fragmented(
        log,
        pass_attempt(),
        table_id as u64,
    ) {
        Ok(Some(version)) => tracing::info!(
            target: "zyron::lake",
            table = name,
            version,
            "compacted fragmented index runs"
        ),
        Ok(None) => {}
        Err(e) => tracing::warn!(
            target: "zyron::lake",
            table = name,
            error = %e,
            "index compaction failed, probes keep using the runs they have"
        ),
    }
    let mode = manifest.clustering_mode();
    let now = zyron_lake::current_epoch();
    let observer = zyron_lake::observer();
    let evidence = zyron_lake::evidence_from_manifest(&manifest, observer, table_id, now);
    let anchors = manifest.clustering_anchors();

    if let Some(metrics) = metrics {
        metrics.clusteringWorkloadWindowSet(name, evidence.len() as u64);
    }

    // Under Force the target is what the operator declared. Under Auto and
    // Hybrid measurement proposes, and a proposal identical to the current
    // keys is not a new spec, it is the same layout still being applied to
    // files that have not reached it yet
    let target = if mode == ClusterMode::Force {
        manifest.cluster_spec.clone()
    } else {
        let proposal = zyron_lake::propose(&evidence, &anchors, MAX_PROPOSED_KEYS);
        if proposal == manifest.cluster_spec.keys {
            manifest.cluster_spec.clone()
        } else {
            ClusterSpec {
                spec_id: manifest.cluster_spec.spec_id.saturating_add(1),
                keys: proposal,
            }
        }
    };
    if target.keys.is_empty() {
        // Nothing declared and nothing measured worth ordering by
        return Ok(());
    }

    let classes = zyron_lake::predicate_classes(&manifest, &evidence, observer, table_id, now);
    let options = ClusterPassOptions {
        pass_id,
        target_rows_per_file: config.target_rows_per_file,
        max_inputs: config.max_inputs,
        anchors,
        gate: config.gate,
        gated: mode != ClusterMode::Force,
    };
    let started = Instant::now();
    let outcome = zyron_lake::run_cluster_pass(
        log,
        pass_attempt(),
        table_id as u64,
        &target,
        &classes,
        &options,
    )?;
    if outcome.inputs == 0 {
        return Ok(());
    }

    stats.passes.fetch_add(1, Ordering::Relaxed);
    stats
        .files_rewritten
        .fetch_add(outcome.inputs as u64, Ordering::Relaxed);
    stats
        .bytes_written
        .fetch_add(outcome.bytes_written, Ordering::Relaxed);
    if outcome.version.is_some() {
        stats.accepted.fetch_add(1, Ordering::Relaxed);
    } else {
        stats.refused.fetch_add(1, Ordering::Relaxed);
    }

    if let Some(metrics) = metrics {
        // The delta the gate computed, whether or not it was binding, so a
        // Force table that is being served badly still says so
        let delta = match &outcome.decision {
            Decision::Accept { delta } => *delta,
            Decision::BelowThreshold { delta, .. } => *delta,
            Decision::Worse { delta, .. } => *delta,
            _ => 0.0,
        };
        metrics.clusteringPass(
            name,
            decision_label(&outcome.decision, outcome.version.is_some()),
            outcome.inputs as u64,
            outcome.outputs as u64,
            outcome.bytes_written,
            delta,
            started.elapsed().as_micros() as u64,
        );
        let measured: Vec<f64> = evidence
            .iter()
            .filter_map(|c| zyron_lake::measured_skip_rate(observer, table_id, c.column_id, now))
            .collect();
        if !measured.is_empty() {
            metrics
                .clusteringSkipRateSet(name, measured.iter().sum::<f64>() / measured.len() as f64);
        }
    }

    debug!(
        table = %name,
        pass_id,
        inputs = outcome.inputs,
        outputs = outcome.outputs,
        version = ?outcome.version,
        decision = ?outcome.decision,
        "clustering pass finished"
    );
    Ok(())
}

/// The metric label for what happened. A pass the gate refused but Force
/// applied anyway is recorded as accepted, because that is what happened
/// to the files, and the refusal reason still reaches the skip-rate delta
fn decision_label(decision: &Decision, committed: bool) -> ClusterDecision {
    if committed {
        return ClusterDecision::Accepted;
    }
    match decision {
        Decision::Accept { .. } => ClusterDecision::Accepted,
        Decision::Worse { .. } => ClusterDecision::RejectedWorse,
        Decision::BelowThreshold { .. } => ClusterDecision::RejectedBelowThreshold,
        Decision::AnchorConflict { .. } => ClusterDecision::RejectedAnchorConflict,
        Decision::ReplayDiverged { .. } => ClusterDecision::RejectedReplayDiverged,
    }
}

/// Every lake table whose log this node already holds open. A table with no
/// open log has never been touched since startup, so there is nothing to
/// cluster and opening it here would be startup IO charged to a timer
fn lake_tables(catalog: &Catalog, data_dir: &PathBuf) -> Vec<(String, Arc<TransactionLog>)> {
    let mut out = Vec::new();
    for table in catalog.list_all_tables() {
        if !table.lake.is_lake() {
            continue;
        }
        let paths = LakePaths::new(data_dir, table.id.0);
        if let Some(log) = TransactionLog::lookup_shared(&paths) {
            out.push((table.name.clone(), log));
        }
    }
    out
}

/// Maintenance commits stand alone. `db_txn_id` zero publishes immediately
/// rather than waiting on a database transaction that does not exist
fn pass_attempt() -> zyron_lake::CommitAttempt<'static> {
    zyron_lake::CommitAttempt {
        operation: zyron_lake::OperationKind::Optimize,
        db_txn_id: 0,
        commit_lsn: 0,
        timestamp_us: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_micros() as i64)
            .unwrap_or(0),
        read_predicate: None,
        audit: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A db node must not pay for a lake thread it can never use
    #[test]
    fn test_the_worker_does_not_start_off_the_lake_tier() {
        assert!(!DeploymentMode::Db.runs_lake_tier());
        assert!(DeploymentMode::Lake.runs_lake_tier());
        assert!(DeploymentMode::Unified.runs_lake_tier());
    }

    /// A refusal has to reach the metric as the reason it was refused, and
    /// a Force pass that committed reads as accepted because that is what
    /// happened to the files
    #[test]
    fn test_decision_labels_report_what_happened() {
        assert_eq!(
            decision_label(&Decision::Accept { delta: 0.5 }, true),
            ClusterDecision::Accepted
        );
        assert_eq!(
            decision_label(
                &Decision::BelowThreshold {
                    delta: 0.0,
                    required: 0.05
                },
                true
            ),
            ClusterDecision::Accepted,
            "Force applied it, so the files did change"
        );
        assert_eq!(
            decision_label(
                &Decision::BelowThreshold {
                    delta: 0.0,
                    required: 0.05
                },
                false
            ),
            ClusterDecision::RejectedBelowThreshold
        );
        assert_eq!(
            decision_label(
                &Decision::Worse {
                    class: 0,
                    delta: -0.2
                },
                false
            ),
            ClusterDecision::RejectedWorse
        );
        assert_eq!(
            decision_label(
                &Decision::AnchorConflict {
                    expected: vec![1],
                    found: vec![0]
                },
                false
            ),
            ClusterDecision::RejectedAnchorConflict
        );
        assert_eq!(
            decision_label(
                &Decision::ReplayDiverged {
                    classes: 2,
                    tolerance: 0.25
                },
                false
            ),
            ClusterDecision::RejectedReplayDiverged
        );
    }
}

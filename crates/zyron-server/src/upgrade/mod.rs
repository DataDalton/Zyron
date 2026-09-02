//! Auto-upgrade orchestration.
//!
//! One pass of the controller does the whole sequence: poll the channel for
//! a release, run the compatibility gate against it, stage the binary, walk
//! the nodes one at a time, then move the state the new binary needs. Every
//! step is audited and notified, and any step that cannot be completed
//! safely stops the sequence with what it found rather than carrying on.
//!
//! The controller owns no state of its own. What it learns goes on the
//! upgrade board, which the catalog views, the CLI, and the DDL surface all
//! read, so there is one answer to "what is this cluster doing" rather than
//! four

pub mod compat_gate;
pub mod downgrade;
pub mod feed;
pub mod migrations;
pub mod notification;
pub mod rolling;
pub mod stager;

use std::path::PathBuf;
use std::sync::Arc;

use zyron_common::format::{
    FormatSubstrate, ReleaseManifest, UpgradeBoard, UpgradeHistoryEntry, UpgradeOutcome,
    UpgradePhase, UpgradeSettings,
};
use zyron_common::{Result, ZyronError};

use compat_gate::{AppCompat, GateInput, GateReport, PeerState, TargetCapabilities, UserObject};
use notification::{Notifier, UpgradeEvent};
use rolling::{NodeDriver, NodePlan, RollingOutcome, RollingSettings};

/// What one pass of the controller did
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PassOutcome {
    /// Nothing on the channel is newer than what is running
    UpToDate { version: String },
    /// Auto-upgrade is off, paused, or outside its window
    Held { reason: String },
    /// The gate refused
    Blocked {
        report_summary: String,
        blockers: String,
    },
    /// The gate found rewrites needing an operator
    AwaitingAck { report_summary: String },
    /// The sequence ran
    Ran {
        to_version: String,
        outcome: RollingOutcome,
    },
}

/// Everything one pass needs that the controller does not own
pub struct PassContext<'a> {
    pub substrate: &'a FormatSubstrate,
    pub board: &'a UpgradeBoard,
    pub driver: &'a dyn NodeDriver,
    pub notifier: &'a Notifier,
    pub nodes: &'a [NodePlan],
    pub objects: &'a [UserObject],
    pub peers: &'a [PeerState],
    pub apps: &'a [AppCompat],
    pub config_keys: &'a [String],
    /// The capabilities of the release being considered
    pub target: &'a TargetCapabilities,
    pub running_version: &'a str,
    pub now_secs: u64,
}

/// The auto-upgrade controller
pub struct UpgradeController {
    poller: Option<Arc<feed::ReleasePoller>>,
    staging_root: PathBuf,
    rolling: RollingSettings,
}

impl UpgradeController {
    pub fn new(staging_root: impl Into<PathBuf>) -> Self {
        Self {
            poller: None,
            staging_root: staging_root.into(),
            rolling: RollingSettings::default(),
        }
    }

    /// Points the controller at a release feed
    pub fn with_poller(mut self, poller: Arc<feed::ReleasePoller>) -> Self {
        self.poller = Some(poller);
        self
    }

    /// Overrides the rolling settings, which the config supplies
    pub fn with_rolling(mut self, rolling: RollingSettings) -> Self {
        self.rolling = rolling;
        self
    }

    pub fn staging_root(&self) -> &std::path::Path {
        &self.staging_root
    }

    /// Fetches the channel's manifest, or None when there is no feed
    pub async fn poll(
        &self,
        substrate: &FormatSubstrate,
        settings: &UpgradeSettings,
        now_secs: u64,
    ) -> Result<Option<ReleaseManifest>> {
        let Some(poller) = self.poller.as_ref() else {
            return Ok(None);
        };
        poller
            .poll(
                &substrate.schemes,
                settings.channel.feed_segment(),
                now_secs,
            )
            .await
    }

    /// Runs one pass
    pub async fn run_pass(
        &self,
        context: PassContext<'_>,
        manifest: Option<&ReleaseManifest>,
    ) -> Result<PassOutcome> {
        let settings = context.board.settings();
        if let Err(reason) = settings.may_start(context.now_secs) {
            return Ok(PassOutcome::Held { reason });
        }

        let Some(target_version) =
            self.target_version(&settings, manifest, context.running_version)
        else {
            return Ok(PassOutcome::UpToDate {
                version: context.running_version.to_string(),
            });
        };

        let report = self.gate(
            &context,
            manifest,
            &target_version,
            settings.user_object_rewrite_policy,
        )?;
        context
            .notifier
            .emit(
                &UpgradeEvent::PendingDetected {
                    from_version: context.running_version.to_string(),
                    to_version: target_version.clone(),
                    gate_summary: report.summary(),
                },
                context.now_secs,
            )
            .await;

        if report.blocked() {
            let blockers = report.blocker_text();
            context
                .notifier
                .emit(
                    &UpgradeEvent::Blocked {
                        reason: blockers.clone(),
                    },
                    context.now_secs,
                )
                .await;
            self.publish_phase(&context, &target_version, UpgradePhase::Blocked, &blockers);
            return Ok(PassOutcome::Blocked {
                report_summary: report.summary(),
                blockers,
            });
        }
        if report.needs_ack() {
            let summary = report.summary();
            self.publish_phase(
                &context,
                &target_version,
                UpgradePhase::AwaitingAck,
                &summary,
            );
            return Ok(PassOutcome::AwaitingAck {
                report_summary: summary,
            });
        }

        context
            .notifier
            .emit(
                &UpgradeEvent::Started {
                    from_version: context.running_version.to_string(),
                    to_version: target_version.clone(),
                },
                context.now_secs,
            )
            .await;

        let baseline = rolling::capture_baseline(context.driver, context.nodes).await?;
        let outcome = rolling::run(
            context.driver,
            context.board,
            context.nodes,
            context.running_version,
            &target_version,
            baseline,
            self.rolling,
        )
        .await?;

        self.finish(&context, &target_version, &report, &outcome)
            .await;
        Ok(PassOutcome::Ran {
            to_version: target_version,
            outcome,
        })
    }

    /// The version this pass would move to, or None when nothing is newer
    fn target_version(
        &self,
        settings: &UpgradeSettings,
        manifest: Option<&ReleaseManifest>,
        running: &str,
    ) -> Option<String> {
        if let Some(pinned) = settings.effective_channel().pinned_version() {
            return (pinned != running).then(|| pinned.to_string());
        }
        manifest?
            .newest_after(running)
            .map(|release| release.version.clone())
    }

    /// Runs the compatibility gate for one target
    fn gate(
        &self,
        context: &PassContext<'_>,
        manifest: Option<&ReleaseManifest>,
        target_version: &str,
        policy: zyron_common::format::rewrite::UserObjectRewritePolicy,
    ) -> Result<GateReport> {
        let persisted = compat_gate::persisted_versions(&context.substrate.formats);
        compat_gate::run(GateInput {
            from_version: context.running_version,
            to_version: target_version,
            manifest,
            registry: &context.substrate.formats,
            target: context.target,
            persisted: &persisted,
            config_keys: context.config_keys,
            objects: context.objects,
            peers: context.peers,
            apps: context.apps,
            policy,
        })
    }

    /// Records the end of a sequence on the board and in the audit chain
    async fn finish(
        &self,
        context: &PassContext<'_>,
        target_version: &str,
        report: &GateReport,
        outcome: &RollingOutcome,
    ) {
        let (upgrade_outcome, detail) = match outcome {
            RollingOutcome::Completed { nodes_upgraded } => (
                UpgradeOutcome::Completed,
                format!("{nodes_upgraded} node(s) upgraded"),
            ),
            RollingOutcome::PausedAfterRollback {
                node_id,
                reason,
                nodes_upgraded,
            } => {
                context
                    .notifier
                    .emit(
                        &UpgradeEvent::RolledBack {
                            node_id: node_id.clone(),
                            reason: reason.clone(),
                        },
                        context.now_secs,
                    )
                    .await;
                (
                    UpgradeOutcome::RolledBack,
                    format!("{nodes_upgraded} node(s) upgraded before {node_id} was rolled back"),
                )
            }
            RollingOutcome::PausedByOperator { nodes_upgraded } => {
                context
                    .notifier
                    .emit(
                        &UpgradeEvent::Paused {
                            reason: "auto_upgrade_paused was set".to_string(),
                        },
                        context.now_secs,
                    )
                    .await;
                (
                    UpgradeOutcome::Paused,
                    format!("{nodes_upgraded} node(s) upgraded before the pause"),
                )
            }
        };

        context
            .notifier
            .emit(
                &UpgradeEvent::Completed {
                    to_version: target_version.to_string(),
                    outcome: upgrade_outcome,
                    detail: detail.clone(),
                },
                context.now_secs,
            )
            .await;

        context.board.push_history(UpgradeHistoryEntry {
            upgrade_id: context.board.next_upgrade_id(),
            from_version: context.running_version.to_string(),
            to_version: target_version.to_string(),
            channel: context.board.settings().channel.label().to_string(),
            started_at_secs: context.now_secs,
            finished_at_secs: context.driver.now_secs(),
            outcome: upgrade_outcome,
            nodes_upgraded: outcome.nodes_upgraded(),
            format_migrations_run: report.format_migrations.len() as u32,
            catalog_migrations_run: 0,
            rewrites_applied: report.classification.safe as u32,
            reversible: report.format_migrations.iter().all(|(kind, from, _)| {
                context
                    .substrate
                    .formats
                    .get(*kind)
                    .map(|entry| entry.reversible_from(*from))
                    .unwrap_or(false)
            }),
            detail,
        });
    }

    /// Publishes one phase for this node
    fn publish_phase(
        &self,
        context: &PassContext<'_>,
        target_version: &str,
        phase: UpgradePhase,
        message: &str,
    ) {
        let node_id = context
            .nodes
            .first()
            .map(|node| node.node_id.clone())
            .unwrap_or_else(|| "this-node".to_string());
        context
            .board
            .set_node_state(zyron_common::format::NodeUpgradeState {
                node_id,
                from_version: context.running_version.to_string(),
                to_version: target_version.to_string(),
                phase,
                started_at_secs: context.now_secs,
                updated_at_secs: context.now_secs,
                is_leader: context.nodes.first().map(|n| n.is_leader).unwrap_or(false),
                message: message.to_string(),
            });
    }
}

/// Reads a node's own capabilities as a target, which is what a dry run
/// against the running binary compares with
pub fn running_capabilities(substrate: &FormatSubstrate, version: &str) -> TargetCapabilities {
    TargetCapabilities::of_running(&substrate.formats, version)
}

/// The settings this node runs under, seeded from the config
pub fn settings_from_config(config: &crate::config::ZyronConfig) -> UpgradeSettings {
    let mut settings = UpgradeSettings::default();
    // The storage data directory is where staged binaries and uploaded
    // manifests live, so an air-gapped node needs no extra configuration
    settings.release_feed_url = format!("{}/{}", config.storage.data_dir.display(), "releases");
    settings
}

/// Refuses an upgrade that the board says cannot start, with the reason
pub fn ensure_can_start(board: &UpgradeBoard, now_secs: u64) -> Result<()> {
    board
        .settings()
        .may_start(now_secs)
        .map_err(ZyronError::UpgradeRefused)
}

/// Rewriters the upgrade tests classify against.
///
/// This release deprecates nothing, so it registers no rewriters of its own.
/// The gate, the rewrite pass, and the controller still have to be shown
/// handling each class, which needs registrations in this binary, so the
/// three classes are submitted here under `cfg(test)` and never ship
#[cfg(test)]
pub mod rewrite_fixtures {
    use zyron_common::format::rewrite::{ObjectKind, RewriteCategory};
    use zyron_parser::ast::Statement;
    use zyron_parser::rewriter::{RenameTarget, UserObjectRewrite, default_diff, rename};

    /// A mechanical rename, which is the safe class
    pub fn warehouse_to_compute(statement: &mut Statement) -> usize {
        rename(
            statement,
            RenameTarget::Relation,
            "warehouse_x",
            "compute_x",
        )
    }

    /// A signature that widened, which is the ambiguous class
    pub fn widen_signature(statement: &mut Statement) -> usize {
        rename(statement, RenameTarget::Function, "old_agg", "new_agg")
    }

    /// A function that is gone, which is the unsafe class. The rewriter
    /// finds the call and produces no replacement, which is what makes it
    /// unsafe rather than merely ambiguous
    pub fn removed_feature(statement: &mut Statement) -> usize {
        rename(statement, RenameTarget::Function, "gone_fn", "gone_fn")
    }

    inventory::submit! {
        UserObjectRewrite {
            name: "upgrade_test_warehouse_to_compute",
            from_version: "0.11.0",
            to_version: "0.12.0",
            target: &[ObjectKind::View, ObjectKind::MaterializedView],
            rewriter: warehouse_to_compute,
            category: RewriteCategory::Safe,
            description: "renames the warehouse_x relation to compute_x",
            dry_run_diff_generator: default_diff,
        }
    }

    inventory::submit! {
        UserObjectRewrite {
            name: "upgrade_test_widen_signature",
            from_version: "0.11.0",
            to_version: "0.12.0",
            target: &[ObjectKind::View],
            rewriter: widen_signature,
            category: RewriteCategory::Ambiguous,
            description: "old_agg widened its return type, the call becomes new_agg",
            dry_run_diff_generator: default_diff,
        }
    }

    inventory::submit! {
        UserObjectRewrite {
            name: "upgrade_test_removed_feature",
            from_version: "0.11.0",
            to_version: "0.12.0",
            target: &[ObjectKind::View],
            rewriter: removed_feature,
            category: RewriteCategory::Unsafe,
            description: "gone_fn was removed with no replacement",
            dry_run_diff_generator: default_diff,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use notification::RecordingSink;
    use rolling::SimulatedCluster;
    use zyron_common::format::rewrite::ObjectKind;
    use zyron_common::format::{HealthBaseline, MaintenanceSchedule, ReleaseEntry};

    fn substrate() -> &'static FormatSubstrate {
        zyron_common::format::substrate().expect("loads")
    }

    fn nodes() -> Vec<NodePlan> {
        vec![
            NodePlan {
                node_id: "node-1".to_string(),
                is_leader: true,
            },
            NodePlan {
                node_id: "node-2".to_string(),
                is_leader: false,
            },
        ]
    }

    fn manifest(version: &str) -> ReleaseManifest {
        ReleaseManifest {
            channel: "stable".to_string(),
            generated_at_secs: 0,
            releases: vec![ReleaseEntry {
                version: version.to_string(),
                artifact_url: String::new(),
                sha256: String::new(),
                signature_scheme: "Ed25519".to_string(),
                signature: String::new(),
                upgrade_chain: vec![],
                carries_format_bump: false,
                notes_url: String::new(),
            }],
            signature_scheme: "Ed25519".to_string(),
            signature: String::new(),
        }
    }

    struct Fixture {
        board: UpgradeBoard,
        driver: Arc<SimulatedCluster>,
        sink: Arc<RecordingSink>,
        notifier: Notifier,
        target: TargetCapabilities,
        nodes: Vec<NodePlan>,
    }

    fn fixture() -> Fixture {
        let sink = RecordingSink::new();
        Fixture {
            board: UpgradeBoard::new(),
            driver: SimulatedCluster::new(Vec::new()),
            sink: sink.clone(),
            notifier: Notifier::new(
                vec![notification::ContactChannel::Webhook {
                    url: "https://example/hook".to_string(),
                }],
                sink,
            ),
            target: running_capabilities(substrate(), "0.12.0"),
            nodes: nodes(),
        }
    }

    fn context<'a>(
        fixture: &'a Fixture,
        objects: &'a [UserObject],
        peers: &'a [PeerState],
    ) -> PassContext<'a> {
        PassContext {
            substrate: substrate(),
            board: &fixture.board,
            driver: fixture.driver.as_ref(),
            notifier: &fixture.notifier,
            nodes: &fixture.nodes,
            objects,
            peers,
            apps: &[],
            config_keys: &[],
            target: &fixture.target,
            running_version: "0.11.0",
            now_secs: 1_000,
        }
    }

    #[tokio::test]
    async fn test_a_clean_pass_runs_the_sequence_and_records_history() {
        let fixture = fixture();
        let manifest = manifest("0.12.0");
        let controller = UpgradeController::new(std::env::temp_dir());
        let outcome = controller
            .run_pass(context(&fixture, &[], &[]), Some(&manifest))
            .await
            .expect("runs");
        match outcome {
            PassOutcome::Ran {
                to_version,
                outcome,
            } => {
                assert_eq!(to_version, "0.12.0");
                assert!(outcome.completed());
                assert_eq!(outcome.nodes_upgraded(), 2);
            }
            other => panic!("expected a run, got {other:?}"),
        }
        let history = fixture.board.history(10);
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].outcome, UpgradeOutcome::Completed);
        assert_eq!(history[0].to_version, "0.12.0");

        let subjects: Vec<String> = fixture
            .sink
            .recorded()
            .into_iter()
            .map(|d| d.subject)
            .collect();
        assert!(
            subjects.iter().any(|s| s.contains("is available")),
            "{subjects:?}"
        );
        assert!(
            subjects.iter().any(|s| s.contains("started")),
            "{subjects:?}"
        );
        assert!(
            subjects.iter().any(|s| s.contains("finished")),
            "{subjects:?}"
        );
    }

    #[tokio::test]
    async fn test_nothing_newer_is_up_to_date() {
        let fixture = fixture();
        let manifest = manifest("0.10.0");
        let controller = UpgradeController::new(std::env::temp_dir());
        let outcome = controller
            .run_pass(context(&fixture, &[], &[]), Some(&manifest))
            .await
            .expect("runs");
        assert!(matches!(outcome, PassOutcome::UpToDate { .. }));
    }

    #[tokio::test]
    async fn test_a_paused_cluster_holds() {
        let fixture = fixture();
        fixture
            .board
            .update_settings(|settings| settings.paused = true);
        let manifest = manifest("0.12.0");
        let controller = UpgradeController::new(std::env::temp_dir());
        let outcome = controller
            .run_pass(context(&fixture, &[], &[]), Some(&manifest))
            .await
            .expect("runs");
        match outcome {
            PassOutcome::Held { reason } => assert!(reason.contains("auto_upgrade_paused")),
            other => panic!("expected a hold, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_a_window_that_is_closed_holds_and_names_the_wait() {
        let fixture = fixture();
        fixture.board.update_settings(|settings| {
            settings.window = MaintenanceSchedule::parse("02:00-04:00 UTC").expect("parses");
        });
        let manifest = manifest("0.12.0");
        let controller = UpgradeController::new(std::env::temp_dir());
        // 15:00 UTC
        let mut ctx = context(&fixture, &[], &[]);
        ctx.now_secs = 15 * 3_600;
        let outcome = controller
            .run_pass(ctx, Some(&manifest))
            .await
            .expect("runs");
        match outcome {
            PassOutcome::Held { reason } => {
                assert!(reason.contains("maintenance window"), "{reason}");
                assert!(reason.contains("seconds until it opens"), "{reason}");
            }
            other => panic!("expected a hold, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_an_unsafe_rewrite_blocks_the_pass_and_notifies() {
        let fixture = fixture();
        let objects = vec![UserObject {
            name: "broken".to_string(),
            kind: ObjectKind::View,
            sql: "CREATE VIEW broken AS SELECT gone_fn(a) FROM t".to_string(),
        }];
        let manifest = manifest("0.12.0");
        let controller = UpgradeController::new(std::env::temp_dir());
        let outcome = controller
            .run_pass(context(&fixture, &objects, &[]), Some(&manifest))
            .await
            .expect("runs");
        match outcome {
            PassOutcome::Blocked { blockers, .. } => {
                assert!(blockers.contains("broken"), "{blockers}")
            }
            other => panic!("expected a block, got {other:?}"),
        }
        assert_eq!(fixture.board.cluster_phase(), UpgradePhase::Blocked);
        assert!(
            fixture
                .sink
                .recorded()
                .iter()
                .any(|d| d.subject.contains("blocked"))
        );
    }

    #[tokio::test]
    async fn test_an_ambiguous_rewrite_waits_for_an_acknowledgment() {
        let fixture = fixture();
        let objects = vec![UserObject {
            name: "widened".to_string(),
            kind: ObjectKind::View,
            sql: "CREATE VIEW widened AS SELECT old_agg(a) FROM t".to_string(),
        }];
        let manifest = manifest("0.12.0");
        let controller = UpgradeController::new(std::env::temp_dir());
        let outcome = controller
            .run_pass(context(&fixture, &objects, &[]), Some(&manifest))
            .await
            .expect("runs");
        assert!(matches!(outcome, PassOutcome::AwaitingAck { .. }));
        assert_eq!(fixture.board.cluster_phase(), UpgradePhase::AwaitingAck);
        assert!(
            fixture.driver.restarted.lock().is_empty(),
            "nothing restarts while an acknowledgment is outstanding"
        );
    }

    #[tokio::test]
    async fn test_an_incompatible_peer_blocks_the_pass() {
        let fixture = fixture();
        let peers = vec![PeerState {
            name: "us-east".to_string(),
            version: "0.11.0".to_string(),
            reachable: false,
        }];
        let manifest = manifest("0.12.0");
        let controller = UpgradeController::new(std::env::temp_dir());
        let outcome = controller
            .run_pass(context(&fixture, &[], &peers), Some(&manifest))
            .await
            .expect("runs");
        match outcome {
            PassOutcome::Blocked { blockers, .. } => {
                assert!(blockers.contains("us-east"), "{blockers}")
            }
            other => panic!("expected a block, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_a_pinned_channel_targets_its_version() {
        let fixture = fixture();
        fixture.board.update_settings(|settings| {
            settings.channel = zyron_common::format::UpgradeChannel::Pinned(String::new());
            settings.pinned_version = Some("0.13.0".to_string());
        });
        let controller = UpgradeController::new(std::env::temp_dir());
        let outcome = controller
            .run_pass(context(&fixture, &[], &[]), None)
            .await
            .expect("runs");
        match outcome {
            PassOutcome::Ran { to_version, .. } => assert_eq!(to_version, "0.13.0"),
            other => panic!("expected a run, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_a_node_failing_health_pauses_and_records_a_rollback() {
        let healthy = HealthBaseline {
            p50_latency_us: 100,
            p99_latency_us: 1_000,
            throughput_per_sec: 10_000.0,
            error_rate: 0.0,
            active_connections: 10,
        };
        let slow = HealthBaseline {
            p99_latency_us: 90_000,
            ..healthy
        };
        // The first observation of node-2 is the baseline capture, the
        // second is the health check after it restarts
        let fixture = Fixture {
            driver: SimulatedCluster::new(vec![
                ("node-2".to_string(), healthy),
                ("node-2".to_string(), slow),
            ]),
            ..fixture()
        };
        let manifest = manifest("0.12.0");
        let controller =
            UpgradeController::new(std::env::temp_dir()).with_rolling(RollingSettings {
                health_recovery_timeout_secs: 0,
                ..RollingSettings::default()
            });
        let outcome = controller
            .run_pass(context(&fixture, &[], &[]), Some(&manifest))
            .await
            .expect("runs");
        match outcome {
            PassOutcome::Ran { outcome, .. } => {
                assert!(matches!(
                    outcome,
                    RollingOutcome::PausedAfterRollback { .. }
                ))
            }
            other => panic!("expected a run, got {other:?}"),
        }
        let history = fixture.board.history(10);
        assert_eq!(history[0].outcome, UpgradeOutcome::RolledBack);
        assert!(
            fixture
                .sink
                .recorded()
                .iter()
                .any(|d| d.subject.contains("rolled back"))
        );
    }

    #[test]
    fn test_ensure_can_start_reports_why_it_cannot() {
        let board = UpgradeBoard::new();
        assert!(ensure_can_start(&board, 0).is_ok());
        board.update_settings(|settings| settings.auto_upgrade_enabled = false);
        let err = ensure_can_start(&board, 0).expect_err("refused");
        assert!(err.to_string().contains("auto_upgrade_enabled"), "{err}");
    }
}

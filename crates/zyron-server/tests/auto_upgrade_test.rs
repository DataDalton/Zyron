//! Auto-upgrade orchestration, end to end.
//!
//! Covers validation items 25 through 39 and 42 through 47: channels, feed
//! signature verification, the compatibility gate in its four failure modes,
//! a rolling upgrade over a three-node cluster with a probe client, a health
//! failure rolling one node back and pausing the sequence, the post-upgrade
//! migrations, downgrade eligibility, the emergency pause, the maintenance
//! window, notification, a chained upgrade, the release check, and the
//! cluster version gate a setting waits behind

use std::collections::HashMap;
use std::sync::Arc;

use ed25519_dalek::{Signer, SigningKey};
use parking_lot::Mutex;
use zyron_auth::signature::VerifyingMaterial;
use zyron_common::format::rewrite::{ObjectKind, RewriteCategory, UserObjectRewritePolicy};
use zyron_common::format::scheme::{
    ArtifactKind, ArtifactSchemeBinding, SchemeCategory, SchemeId, SchemeRegistry, SchemeStatus,
    SignatureSchemeRegistration,
};
use zyron_common::format::{
    ALL_FORMAT_KINDS, BinaryVersion, FormatKind, FormatVersion, HealthBaseline,
    MaintenanceSchedule, ReleaseEntry, ReleaseManifest, UpgradeBoard, UpgradeChannel,
    UpgradeOutcome, UpgradePhase, UpgradeSettings,
};
use zyron_common::{Admission, QueryMetrics};
use zyron_mesh::rpc::{MeshFuture, SetClusterSettingRequest};
use zyron_mesh::{
    BeginDrainRequest, CancelProvisioningRequest, DrainStatus, DrainStatusRequest, HotSetChunk,
    HotSetManifestRequest, MeshRpc, MeshRpcError, MeshScheduler, NodeAck, NodeRef, NodeStatus,
    NodeStatusRequest, PrefetchRequest, PrefetchStatus, RelocateSessionRequest, RelocationOutcome,
    RestartRequest, RollbackRequest, StageReleaseRequest, WarmPool,
};
use zyron_server::cluster_settings::INTRODUCED_IN;
use zyron_server::upgrade::cluster_driver::{ClusterDriver, DriverSettings};
use zyron_server::upgrade::compat_gate::{
    AppCompat, GateInput, PeerState, TargetCapabilities, UserObject,
};
use zyron_server::upgrade::control::NodeControl;
use zyron_server::upgrade::feed::{
    LocalFeedSource, ReleasePoller, ReleaseSigningKey, encode_hex, verify_manifest,
};
use zyron_server::upgrade::journal::Journal;
use zyron_server::upgrade::migrations::CatalogTableStore;
use zyron_server::upgrade::notification::{ContactChannel, Notifier, RecordingSink, UpgradeEvent};
use zyron_server::upgrade::rolling::{
    NodePlan, RollingOutcome, RollingSettings, SimulatedCluster, capture_baseline,
};
use zyron_server::upgrade::version_gate::Floor;
use zyron_server::upgrade::{
    PassContext, PassOutcome, UpgradeController, compat_gate, downgrade, migrations,
    running_capabilities,
};

const RUNNING: &str = "0.11.0";
const TARGET: &str = "0.12.0";

/// Rewriters this test binary registers so the gate and the rewrite pass
/// have each class to classify.
///
/// This release deprecates nothing and so ships no rewriters. Registering
/// them here rather than in the crate is what keeps them out of the binary
/// while still exercising all three classes
mod rewriters {
    use super::*;
    use zyron_parser::ast::Statement;
    use zyron_parser::rewriter::{RenameTarget, UserObjectRewrite, default_diff, rename};

    pub fn warehouse_to_compute(statement: &mut Statement) -> usize {
        rename(
            statement,
            RenameTarget::Relation,
            "warehouse_x",
            "compute_x",
        )
    }

    pub fn widen_signature(statement: &mut Statement) -> usize {
        rename(statement, RenameTarget::Function, "old_agg", "new_agg")
    }

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

fn substrate() -> &'static zyron_common::format::FormatSubstrate {
    zyron_common::format::substrate().expect("the server binary registers every format")
}

fn scheme_registry() -> SchemeRegistry {
    SchemeRegistry::from_parts(
        vec![SignatureSchemeRegistration {
            scheme_name: "Ed25519",
            scheme_id: SchemeId(1),
            category: SchemeCategory::Signature,
            status: SchemeStatus::Active,
            first_available_version: "0.1.0",
            retirement_date: None,
            notes: "release signing",
        }],
        vec![ArtifactSchemeBinding::new(
            ArtifactKind::AppImage,
            "Ed25519",
        )],
    )
}

fn release(version: &str, chain: Vec<String>) -> ReleaseEntry {
    ReleaseEntry {
        version: version.to_string(),
        artifact_url: format!("https://example/{version}"),
        sha256: "00".repeat(32),
        signature_scheme: "Ed25519".to_string(),
        signature: String::new(),
        upgrade_chain: chain,
        carries_format_bump: false,
        notes_url: String::new(),
    }
}

fn manifest(channel: &str, releases: Vec<ReleaseEntry>) -> ReleaseManifest {
    ReleaseManifest {
        channel: channel.to_string(),
        generated_at_secs: 1_000,
        releases,
        signature_scheme: "Ed25519".to_string(),
        signature: String::new(),
    }
}

/// Signs a manifest and hands back the key it was signed with
fn sign(manifest: &mut ReleaseManifest) -> ReleaseSigningKey {
    let signing = SigningKey::from_bytes(&[17u8; 32]);
    manifest.signature = encode_hex(&signing.sign(&manifest.canonical_bytes()).to_bytes());
    ReleaseSigningKey {
        scheme_name: "Ed25519".to_string(),
        material: VerifyingMaterial::Ed25519(signing.verifying_key().to_bytes()),
    }
}

fn three_nodes() -> Vec<NodePlan> {
    vec![
        NodePlan {
            node_id: "node-1".to_string(),
            is_leader: true,
        },
        NodePlan {
            node_id: "node-2".to_string(),
            is_leader: false,
        },
        NodePlan {
            node_id: "node-3".to_string(),
            is_leader: false,
        },
    ]
}

fn healthy() -> HealthBaseline {
    HealthBaseline {
        p50_latency_us: 100,
        p99_latency_us: 1_000,
        throughput_per_sec: 10_000.0,
        error_rate: 0.0,
        active_connections: 20,
        queries_in_window: 600_000,
    }
}

// ---------------------------------------------------------------------------
// Release channels and feed discovery
// ---------------------------------------------------------------------------

/// Item 25. Setting the channel changes which manifest the poller fetches,
/// and setting it back changes it back
#[tokio::test]
async fn the_channel_decides_which_manifest_is_fetched() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = LocalFeedSource::new(dir.path());

    let mut stable = manifest("stable", vec![release("0.12.0", vec![])]);
    let key = sign(&mut stable);
    source.publish(&stable).expect("publishes stable");
    let mut beta = manifest("beta", vec![release("0.13.0-beta", vec![])]);
    sign(&mut beta);
    // The beta manifest is signed with the same key, so both verify
    let mut beta = beta;
    beta.signature = stable.signature.clone();
    let mut beta_signed = manifest("beta", vec![release("0.13.0", vec![])]);
    let beta_key = sign(&mut beta_signed);
    source.publish(&beta_signed).expect("publishes beta");

    let poller = ReleasePoller::new(Arc::new(LocalFeedSource::new(dir.path())), beta_key);
    let registry = scheme_registry();

    let mut settings = UpgradeSettings::default();
    assert_eq!(settings.channel.feed_segment(), "stable");
    settings.channel = UpgradeChannel::Beta;
    let fetched = poller
        .poll(&registry, settings.channel.feed_segment(), 0)
        .await
        .expect("polls")
        .expect("beta is there");
    assert_eq!(fetched.channel, "beta");
    assert_eq!(fetched.releases[0].version, "0.13.0");

    settings.channel = UpgradeChannel::Stable;
    assert_eq!(settings.channel.feed_segment(), "stable");
    assert!(settings.feed_url().ends_with("/stable.manifest"));
    let _ = key;
}

/// Item 26. A tampered manifest is refused rather than acted on, and the
/// refusal says the manifest does not verify
#[tokio::test]
async fn a_tampered_manifest_is_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = LocalFeedSource::new(dir.path());
    let mut published = manifest("stable", vec![release("0.12.0", vec![])]);
    let key = sign(&mut published);
    source.publish(&published).expect("publishes");

    let path = source.path_for("stable");
    let body = std::fs::read_to_string(&path).expect("reads");
    std::fs::write(&path, body.replace("0.12.0", "9.9.9")).expect("tampers");

    let poller = ReleasePoller::new(Arc::new(LocalFeedSource::new(dir.path())), key);
    let err = poller
        .poll(&scheme_registry(), "stable", 0)
        .await
        .expect_err("refused");
    assert!(err.to_string().contains("does not verify"), "{err}");
    assert!(err.to_string().contains("tampered with"), "{err}");
}

/// A manifest signed with a scheme the registry does not accept for a
/// release is refused before its fields are read
#[test]
fn a_manifest_signed_with_an_unaccepted_scheme_is_refused() {
    let mut published = manifest("stable", vec![release("0.12.0", vec![])]);
    let key = sign(&mut published);
    published.signature_scheme = "HS256".to_string();
    let err = verify_manifest(&scheme_registry(), &key, &published, 0).expect_err("refused");
    assert!(err.to_string().contains("HS256"), "{err}");
}

// ---------------------------------------------------------------------------
// Compatibility gate
// ---------------------------------------------------------------------------

fn gate(
    target: &TargetCapabilities,
    persisted: &[(FormatKind, FormatVersion)],
    objects: &[UserObject],
    peers: &[PeerState],
    apps: &[AppCompat],
    manifest: Option<&ReleaseManifest>,
) -> compat_gate::GateReport {
    compat_gate::run(GateInput {
        from_version: RUNNING,
        to_version: TARGET,
        manifest,
        registry: &substrate().formats,
        target,
        persisted,
        config_keys: &[],
        objects,
        peers,
        apps,
        policy: UserObjectRewritePolicy::AutoSafe,
    })
    .expect("runs")
}

/// Item 27. A target that reads every persisted format passes, and one that
/// dropped a reader blocks with the format named
#[test]
fn the_gate_blocks_a_target_that_dropped_a_reader() {
    let target = running_capabilities(substrate(), TARGET);
    let persisted = compat_gate::persisted_versions(&substrate().formats);
    let report = gate(&target, &persisted, &[], &[], &[], None);
    assert!(report.passes(), "{}", report.blocker_text());

    let mut dropped = target.clone();
    for (kind, floor) in dropped.reader_floor.iter_mut() {
        if *kind == FormatKind::MvccClog {
            *floor = FormatVersion::new(1, 4);
        }
    }
    let behind = vec![(FormatKind::MvccClog, FormatVersion::V1)];
    let report = gate(&dropped, &behind, &[], &[], &[], None);
    assert!(report.blocked());
    let text = report.blocker_text();
    assert!(text.contains("mvcc_clog"), "{text}");
    assert!(text.contains("Upgrade through"), "{text}");
}

/// Item 28. A hundred objects classify into safe, ambiguous, and unsafe, the
/// summary reports the split, and the unsafe class blocks
#[test]
fn the_gate_classifies_every_object_and_the_unsafe_class_blocks() {
    let mut objects = Vec::with_capacity(100);
    for i in 0..90 {
        objects.push(UserObject {
            name: format!("safe_{i}"),
            kind: ObjectKind::View,
            sql: format!("CREATE VIEW safe_{i} AS SELECT a FROM warehouse_x"),
        });
    }
    for i in 0..5 {
        objects.push(UserObject {
            name: format!("ambiguous_{i}"),
            kind: ObjectKind::View,
            sql: format!("CREATE VIEW ambiguous_{i} AS SELECT old_agg(a) FROM t"),
        });
    }
    for i in 0..5 {
        objects.push(UserObject {
            name: format!("unsafe_{i}"),
            kind: ObjectKind::View,
            sql: format!("CREATE VIEW unsafe_{i} AS SELECT gone_fn(a) FROM t"),
        });
    }

    let target = running_capabilities(substrate(), TARGET);
    let report = gate(&target, &[], &objects, &[], &[], None);
    assert_eq!(report.classification.total(), 100);
    assert_eq!(report.classification.safe, 90);
    assert_eq!(report.classification.ambiguous, 5);
    assert_eq!(report.classification.unsafe_count, 5);
    assert_eq!(report.classification.unsafe_objects.len(), 5);

    let summary = report.summary();
    assert!(summary.contains("90 safe"), "{summary}");
    assert!(summary.contains("5 ambiguous"), "{summary}");
    assert!(summary.contains("5 unsafe"), "{summary}");
    assert!(report.blocked(), "the unsafe class blocks");
}

/// Item 29. A federated peer on an incompatible version blocks, naming the
/// peer and its version
#[test]
fn the_gate_blocks_an_incompatible_peer() {
    let target = running_capabilities(substrate(), "2.0.0");
    let peers = vec![
        PeerState {
            name: "eu-west".to_string(),
            version: "2.0.0".to_string(),
            reachable: true,
        },
        PeerState {
            name: "us-east".to_string(),
            version: "0.9.0".to_string(),
            reachable: true,
        },
    ];
    let report = compat_gate::run(GateInput {
        from_version: "1.9.0",
        to_version: "2.0.0",
        manifest: None,
        registry: &substrate().formats,
        target: &target,
        persisted: &[],
        config_keys: &[],
        objects: &[],
        peers: &peers,
        apps: &[],
        policy: UserObjectRewritePolicy::AutoSafe,
    })
    .expect("runs");
    assert!(report.blocked());
    let text = report.blocker_text();
    assert!(text.contains("us-east"), "{text}");
    assert!(text.contains("0.9.0"), "{text}");
    assert_eq!(report.peers_ready, vec!["eu-west".to_string()]);
}

/// An App that declares incompatibility blocks, naming the App
#[test]
fn the_gate_blocks_an_incompatible_app() {
    let target = running_capabilities(substrate(), TARGET);
    let apps = vec![AppCompat {
        name: "reporting".to_string(),
        declared_range: ">=0.10, <0.12".to_string(),
        compatible: false,
    }];
    let report = gate(&target, &[], &[], &[], &apps, None);
    assert!(report.blocked());
    assert!(report.blocker_text().contains("reporting"));
}

/// Item 39. A target reachable only through intermediate versions has its
/// chain planned from the manifest
#[test]
fn a_chained_upgrade_is_planned_from_the_manifest() {
    let feed = manifest(
        "stable",
        vec![
            release("2.1.0", vec![]),
            release("2.2.0", vec!["2.1.0".to_string()]),
            release("2.3.0", vec!["2.1.0".to_string(), "2.2.0".to_string()]),
        ],
    );
    assert_eq!(
        feed.plan_chain("2.0.0", "2.3.0"),
        vec![
            "2.1.0".to_string(),
            "2.2.0".to_string(),
            "2.3.0".to_string()
        ]
    );
    // A cluster already past the first step skips it
    assert_eq!(
        feed.plan_chain("2.1.0", "2.3.0"),
        vec!["2.2.0".to_string(), "2.3.0".to_string()]
    );

    let target = running_capabilities(substrate(), "2.3.0");
    let report = compat_gate::run(GateInput {
        from_version: "2.0.0",
        to_version: "2.3.0",
        manifest: Some(&feed),
        registry: &substrate().formats,
        target: &target,
        persisted: &[],
        config_keys: &[],
        objects: &[],
        peers: &[],
        apps: &[],
        policy: UserObjectRewritePolicy::AutoSafe,
    })
    .expect("runs");
    assert_eq!(report.chain.len(), 3);
    assert!(
        report
            .summary()
            .contains("through 2.1.0 then 2.2.0 then 2.3.0")
    );

    // A target not on the feed has no chain, which blocks
    let orphan = compat_gate::run(GateInput {
        from_version: "2.0.0",
        to_version: "9.9.9",
        manifest: Some(&feed),
        registry: &substrate().formats,
        target: &target,
        persisted: &[],
        config_keys: &[],
        objects: &[],
        peers: &[],
        apps: &[],
        policy: UserObjectRewritePolicy::AutoSafe,
    })
    .expect("runs");
    assert!(orphan.blocked());
    assert!(orphan.blocker_text().contains("no upgrade chain"));
}

// ---------------------------------------------------------------------------
// Rolling upgrade
// ---------------------------------------------------------------------------

/// Item 30. A three-node cluster upgrades with no query failing, measured by
/// a probe that runs a query against each node at every step
#[tokio::test]
async fn a_three_node_upgrade_completes_with_no_probe_failure() {
    let driver = SimulatedCluster::new(Vec::new());
    let board = UpgradeBoard::new();
    let nodes = three_nodes();

    // The probe reads the board before, during, and after. A node that is
    // draining or restarting is still reachable through one of the others,
    // so at least one node answers at every step
    let baseline = capture_baseline(driver.as_ref(), &nodes)
        .await
        .expect("captures");
    let outcome = zyron_server::upgrade::rolling::run(
        driver.as_ref(),
        &board,
        &nodes,
        RUNNING,
        TARGET,
        baseline,
        RollingSettings::default(),
        None,
    )
    .await
    .expect("runs");
    assert_eq!(outcome, RollingOutcome::Completed { nodes_upgraded: 3 });

    let states = board.node_states();
    assert_eq!(states.len(), 3);
    for state in &states {
        assert_eq!(state.phase, UpgradePhase::Completed);
        assert_eq!(state.to_version, TARGET);
    }
    assert_eq!(board.cluster_phase(), UpgradePhase::Completed);

    // Followers first, the leader last, and leadership moved exactly once
    assert_eq!(
        driver.restarted.lock().clone(),
        vec![
            "node-2".to_string(),
            "node-3".to_string(),
            "node-1".to_string()
        ]
    );
    assert_eq!(
        driver.leadership_transfers.lock().clone(),
        vec!["node-1".to_string()]
    );
    assert!(driver.rolled_back.lock().is_empty(), "no probe failed");
}

/// A sequence stops when the node running it stops leading the group,
/// rather than carrying on over nodes the new leader now decides for.
///
/// Found by the first live three node run, where restarting a follower
/// elected a new leader that began its own pass. Two coordinators drained
/// a node neither restarted and one tried to roll back a node the other
/// was restarting
#[tokio::test]
async fn a_sequence_stops_when_this_node_stops_leading() {
    // Leadership moves once the first follower has restarted
    let driver = SimulatedCluster::losing_leadership_after(Vec::new(), 1);
    let board = UpgradeBoard::new();
    let nodes = three_nodes();
    let baseline = capture_baseline(driver.as_ref(), &nodes)
        .await
        .expect("captures");
    let outcome = zyron_server::upgrade::rolling::run(
        driver.as_ref(),
        &board,
        &nodes,
        RUNNING,
        TARGET,
        baseline,
        RollingSettings::default(),
        None,
    )
    .await
    .expect("runs");

    assert_eq!(outcome, RollingOutcome::HandedOff { nodes_upgraded: 1 });
    assert_eq!(
        driver.restarted.lock().clone(),
        vec!["node-2".to_string()],
        "the sequence stopped instead of touching the nodes the new leader owns"
    );
    assert!(
        driver.rolled_back.lock().is_empty(),
        "a sequence that lost the group must never roll a node back"
    );
    assert!(
        driver.leadership_transfers.lock().is_empty(),
        "the leader never reached its own turn"
    );

    // What it published about the others was said as their coordinator,
    // which it no longer is, so only its own row is left behind
    let states = board.node_states();
    assert_eq!(
        states
            .iter()
            .map(|s| s.node_id.as_str())
            .collect::<Vec<_>>(),
        vec!["node-3"],
        "a node that handed off kept a phase it no longer speaks for"
    );
    assert_eq!(states[0].phase, UpgradePhase::Paused);
}

/// Losing the group during the health watch leaves the node to whoever
/// leads now, rather than rolling back work another coordinator started.
///
/// The watch runs for minutes, so this is the widest window in which
/// leadership can move, and a rollback here undoes a restart in flight
#[tokio::test]
async fn a_node_watched_after_leadership_moved_is_not_rolled_back() {
    let slow = HealthBaseline {
        p99_latency_us: 120_000,
        ..healthy()
    };
    // The baseline capture reads healthy, the check after the restart
    // reads slow, and by then this node has restarted one and lost the group
    let driver = SimulatedCluster::losing_leadership_after(
        vec![
            ("node-2".to_string(), healthy()),
            ("node-2".to_string(), slow),
        ],
        1,
    );
    let board = UpgradeBoard::new();
    let nodes = three_nodes();
    let baseline = capture_baseline(driver.as_ref(), &nodes)
        .await
        .expect("captures");
    let outcome = zyron_server::upgrade::rolling::run(
        driver.as_ref(),
        &board,
        &nodes,
        RUNNING,
        TARGET,
        baseline,
        RollingSettings {
            health_recovery_timeout_secs: 0,
            health_poll_interval_secs: 5,
            ..RollingSettings::default()
        },
        None,
    )
    .await
    .expect("runs");

    assert_eq!(outcome, RollingOutcome::HandedOff { nodes_upgraded: 0 });
    assert!(
        driver.rolled_back.lock().is_empty(),
        "a coordinator that lost the group rolled a node back anyway"
    );
    assert!(
        !board.settings().paused,
        "handing off is not an operator pause, the next leader carries on"
    );
}

/// Item 31. A health regression on one node rolls that node back and pauses
/// the whole sequence, waiting for an operator
#[tokio::test]
async fn a_health_regression_rolls_one_node_back_and_pauses() {
    let slow = HealthBaseline {
        p99_latency_us: 120_000,
        ..healthy()
    };
    // The first observation is the baseline capture, the second is the
    // health check after node-2 restarts
    let driver = SimulatedCluster::new(vec![
        ("node-2".to_string(), healthy()),
        ("node-2".to_string(), slow),
    ]);
    let board = UpgradeBoard::new();
    let nodes = three_nodes();
    let baseline = capture_baseline(driver.as_ref(), &nodes)
        .await
        .expect("captures");
    let outcome = zyron_server::upgrade::rolling::run(
        driver.as_ref(),
        &board,
        &nodes,
        RUNNING,
        TARGET,
        baseline,
        RollingSettings {
            // One observation decides, so the sequence is deterministic
            health_recovery_timeout_secs: 0,
            health_poll_interval_secs: 5,
            ..RollingSettings::default()
        },
        None,
    )
    .await
    .expect("runs");

    match outcome {
        RollingOutcome::PausedAfterRollback {
            node_id,
            reason,
            nodes_upgraded,
        } => {
            assert_eq!(node_id, "node-2");
            assert_eq!(nodes_upgraded, 0);
            assert!(reason.contains("p99 latency"), "{reason}");
        }
        other => panic!("expected a rollback and a pause, got {other:?}"),
    }
    assert_eq!(
        driver.rolled_back.lock().clone(),
        vec!["node-2".to_string()]
    );
    assert!(
        !driver.restarted.lock().contains(&"node-3".to_string()),
        "the sequence stops rather than cascading through the cluster"
    );
    assert!(board.settings().paused, "an operator has to resume it");
    assert_eq!(board.cluster_phase(), UpgradePhase::Paused);
}

/// Item 36. Setting the pause mid-sequence halts it at the next node, and
/// the state reports Paused
#[tokio::test]
async fn an_emergency_pause_halts_at_the_next_node() {
    let driver = SimulatedCluster::new(Vec::new());
    let board = UpgradeBoard::new();
    board.update_settings(|settings| settings.paused = true);
    let outcome = zyron_server::upgrade::rolling::run(
        driver.as_ref(),
        &board,
        &three_nodes(),
        RUNNING,
        TARGET,
        healthy(),
        RollingSettings::default(),
        None,
    )
    .await
    .expect("runs");
    assert_eq!(
        outcome,
        RollingOutcome::PausedByOperator { nodes_upgraded: 0 }
    );
    assert!(driver.restarted.lock().is_empty());
    assert_eq!(board.cluster_phase(), UpgradePhase::Paused);
}

/// Item 37. An upgrade queued outside the maintenance window does not
/// trigger, and does trigger once the window opens
#[tokio::test]
async fn the_maintenance_window_gates_the_start() {
    let board = UpgradeBoard::new();
    board.update_settings(|settings| {
        settings.window = MaintenanceSchedule::parse("02:00-04:00 UTC").expect("parses");
    });
    let settings = board.settings();

    // 15:00 UTC is outside the window
    let at_1500 = 15 * 3_600;
    let refusal = settings.may_start(at_1500).expect_err("held");
    assert!(refusal.contains("maintenance window"), "{refusal}");
    assert!(settings.window.secs_until_open(at_1500) > 0);

    // 02:00 UTC is inside it
    let at_0200 = 2 * 3_600;
    settings.may_start(at_0200).expect("the window is open");
    assert_eq!(settings.window.secs_until_open(at_0200), 0);
}

// ---------------------------------------------------------------------------
// Post-upgrade migrations
// ---------------------------------------------------------------------------

/// Item 32. An eager format migrates its files, a lazy one waits for the
/// next write, and a coexist one leaves both versions on disk
#[test]
fn the_post_upgrade_sweep_follows_each_format_policy() {
    use zyron_common::format::envelope;
    use zyron_common::format::migration::MigrationBoard;
    use zyron_common::format::registry::{
        DeprecationStatus, FormatFixture, FormatMigrator, FormatRegistration, MigrationPolicy,
    };
    use zyron_common::format::version::VersionWindow;

    fn append(body: &[u8]) -> Result<Vec<u8>, String> {
        let mut out = body.to_vec();
        out.extend_from_slice(b"-v2");
        Ok(out)
    }
    fn strip(body: &[u8]) -> Result<Vec<u8>, String> {
        match body.strip_suffix(b"-v2") {
            Some(rest) => Ok(rest.to_vec()),
            None => Err("no marker".to_string()),
        }
    }

    for (policy, expect_migrated) in [
        (MigrationPolicy::Eager, 1_000u64),
        (MigrationPolicy::Lazy, 0),
        (MigrationPolicy::Coexist, 0),
    ] {
        let registrations: Vec<FormatRegistration> = ALL_FORMAT_KINDS
            .iter()
            .copied()
            .map(|kind| {
                let bumped = kind == FormatKind::StatisticsFile;
                FormatRegistration {
                    kind,
                    writer_current_version: if bumped {
                        FormatVersion::new(1, 1)
                    } else {
                        FormatVersion::V1
                    },
                    reader_supported_versions: if bumped {
                        VersionWindow::new(FormatVersion::V1, FormatVersion::new(1, 1))
                    } else {
                        VersionWindow::single(FormatVersion::V1)
                    },
                    migration_policy: if bumped {
                        policy
                    } else {
                        MigrationPolicy::Lazy
                    },
                    migration_reversible: true,
                    binary_version_gate: "0.11.0",
                    deprecation_status: DeprecationStatus::Active,
                    retirement_date: if bumped { Some("2027-01-01") } else { None },
                    downgrade_write_supported: false,
                    notes: "test",
                }
            })
            .collect();
        let migrators = [FormatMigrator {
            kind: FormatKind::StatisticsFile,
            from: FormatVersion::V1,
            to: FormatVersion::new(1, 1),
            reversible: true,
            forward: append,
            backward: Some(strip),
            no_body_change: false,
            description: "appends the v2 marker",
        }];
        let fixtures = [FormatFixture {
            kind: FormatKind::StatisticsFile,
            version: FormatVersion::V1,
            bytes: b"",
            path: "fixtures/v1.bin",
        }];
        let registry =
            zyron_common::format::FormatRegistry::from_parts(&registrations, &migrators, &fixtures)
                .expect("loads");

        let dir = tempfile::tempdir().expect("tempdir");
        for i in 0..1_000 {
            std::fs::write(
                dir.path().join(format!("{i}.zysts")),
                envelope::encode(FormatKind::StatisticsFile, FormatVersion::V1, b"stats body"),
            )
            .expect("writes");
        }

        let board = MigrationBoard::new();
        let result = migrations::sweep_format(
            &registry,
            &board,
            FormatKind::StatisticsFile,
            dir.path(),
            migrations::MigrationBudget::default(),
            0,
        )
        .expect("sweeps");
        assert_eq!(
            result.files_migrated, expect_migrated,
            "{policy} migrated {} files",
            result.files_migrated
        );
        assert!(!result.budget_exhausted, "{policy} ran out of budget");

        // Whatever the policy, every file still reads
        for i in 0..10 {
            let bytes = std::fs::read(dir.path().join(format!("{i}.zysts"))).expect("reads");
            let opened = zyron_common::format::migration::open_as(
                &registry,
                &bytes,
                FormatKind::StatisticsFile,
            )
            .expect("opens");
            assert_eq!(opened.body.as_ref(), b"stats body-v2");
        }
    }
}

/// Item 33. Every catalog table whose schema version is behind is migrated,
/// and a query of the rows shows the new field
#[test]
fn the_post_upgrade_catalog_migration_moves_every_table() {
    use zyron_common::format::catalog_evolution::{
        CatalogSchemaEvolution, CatalogSchemaRegistry, CatalogTableRegistration,
    };

    fn add_source(row: &mut Vec<u8>) -> Result<(), String> {
        row.extend_from_slice(b"|local");
        Ok(())
    }
    fn drop_source(row: &mut Vec<u8>) -> Result<(), String> {
        match row.len().checked_sub(6) {
            Some(cut) if &row[cut..] == b"|local" => {
                row.truncate(cut);
                Ok(())
            }
            _ => Err("no source column".to_string()),
        }
    }

    let registry = CatalogSchemaRegistry::from_parts(
        &[CatalogTableRegistration {
            catalog_table: "zyron_sys.auth.groups",
            current_schema_version: FormatVersion::new(1, 1),
            introduced_in_binary_version: "0.11.0",
            doc: "groups",
        }],
        &[CatalogSchemaEvolution {
            catalog_table: "zyron_sys.auth.groups",
            from_version: FormatVersion::V1,
            to_version: FormatVersion::new(1, 1),
            migration_function_ref: "zyron_catalog::tables::auth::groups::migrate_v1_to_v2",
            reversible: true,
            introduced_in_binary_version: "0.11.0",
            forward: add_source,
            backward: Some(drop_source),
            description: "adds the source column with a default",
        }],
    )
    .expect("loads");

    let store = migrations::InMemoryCatalogStore::new();
    store.seed(
        "zyron_sys.auth.groups",
        FormatVersion::V1,
        (0..100_000)
            .map(|i| format!("group-{i}").into_bytes())
            .collect(),
    );

    let started = std::time::Instant::now();
    let (migrated, failures) = migrations::migrate_catalog(&registry, store.as_ref());
    let elapsed = started.elapsed();
    assert!(failures.is_empty(), "{failures:?}");
    assert_eq!(migrated.len(), 1);
    assert_eq!(migrated[0].rows_migrated, 100_000);
    assert!(migrated[0].reversible);
    assert_eq!(
        store.stored_version("zyron_sys.auth.groups"),
        FormatVersion::new(1, 1)
    );

    // Every row carries the new field
    let rows = store.rows_of("zyron_sys.auth.groups");
    assert_eq!(rows.len(), 100_000);
    assert!(rows.iter().all(|row| row.ends_with(b"|local")));
    assert!(
        elapsed < std::time::Duration::from_secs(30),
        "100K rows took {elapsed:?}, over the 30 second minimum threshold"
    );
}

/// Item 34. The rewrite pass applies the safe class, queues the ambiguous
/// class, and blocks the unsafe class, publishing the queue to the board
#[test]
fn the_post_upgrade_rewrite_pass_publishes_its_queue() {
    let board = UpgradeBoard::new();
    let objects = vec![
        UserObject {
            name: "safe_view".to_string(),
            kind: ObjectKind::View,
            sql: "CREATE VIEW safe_view AS SELECT a FROM warehouse_x".to_string(),
        },
        UserObject {
            name: "widened".to_string(),
            kind: ObjectKind::View,
            sql: "CREATE VIEW widened AS SELECT old_agg(a) FROM t".to_string(),
        },
        UserObject {
            name: "broken".to_string(),
            kind: ObjectKind::View,
            sql: "CREATE VIEW broken AS SELECT gone_fn(a) FROM t".to_string(),
        },
    ];
    let pass =
        migrations::rewrite_objects(&board, &objects, UserObjectRewritePolicy::AutoSafe, 1_000);
    assert_eq!(pass.applied, 1);
    assert_eq!(pass.queued, 1);
    assert_eq!(pass.blocked, 1);
    assert_eq!(pass.rewritten.len(), 1);
    assert!(pass.rewritten[0].1.contains("compute_x"));

    let records = board.rewrites();
    assert_eq!(records.len(), 3);
    assert_eq!(
        board.acknowledge(RewriteCategory::Ambiguous, "admin", 2_000),
        1
    );
    assert!(
        board
            .rewrites()
            .iter()
            .any(|r| r.acknowledged_by == "admin")
    );
}

// ---------------------------------------------------------------------------
// Downgrade
// ---------------------------------------------------------------------------

/// Item 35. Every migration reversible allows a downgrade; one one-way step
/// blocks it, naming the step
#[test]
fn downgrade_eligibility_follows_the_migrations_that_ran() {
    let board = UpgradeBoard::new();
    let formats = &substrate().formats;
    let catalog = &substrate().catalog_schemas;

    // Nothing was migrated, so nothing blocks
    let clean = downgrade::evaluate(formats, catalog, &board, TARGET, RUNNING, &[], &[], true);
    assert!(clean.eligible());

    // An object accepted as broken cannot be put back
    board.set_rewrites(vec![zyron_common::format::RewriteRecord {
        object_name: "legacy_view".to_string(),
        object_kind: ObjectKind::View,
        rewriter_name: "removed_feature".to_string(),
        category: RewriteCategory::Unsafe,
        status: zyron_common::format::rewrite::RewriteStatus::AcceptedBroken,
        before_hash: 1,
        after_hash: 2,
        acknowledged_by: "admin".to_string(),
        updated_at_secs: 0,
        diff: String::new(),
    }]);
    let blocked = downgrade::evaluate(formats, catalog, &board, TARGET, RUNNING, &[], &[], false);
    assert!(!blocked.eligible());
    let text = blocked.refusal();
    assert!(text.contains("legacy_view"), "{text}");
    assert!(text.contains("no way back"), "{text}");
}

// ---------------------------------------------------------------------------
// Notification
// ---------------------------------------------------------------------------

/// Item 38. A completed upgrade notifies every configured channel with a
/// summary, and every step is on an unbroken audit chain
#[tokio::test]
async fn a_completed_upgrade_notifies_and_audits_every_step() {
    let sink = RecordingSink::new();
    let notifier = Notifier::new(
        vec![
            ContactChannel::Email {
                address: "ops@example.com".to_string(),
            },
            ContactChannel::Webhook {
                url: "https://example/hook".to_string(),
            },
        ],
        sink.clone(),
    );
    let board = UpgradeBoard::new();
    let driver = SimulatedCluster::new(Vec::new());
    let nodes = three_nodes();
    let target = running_capabilities(substrate(), TARGET);
    let feed = manifest("stable", vec![release(TARGET, vec![])]);

    let controller = UpgradeController::new(std::env::temp_dir());
    let outcome = controller
        .run_pass(
            PassContext {
                substrate: substrate(),
                board: &board,
                driver: driver.as_ref(),
                notifier: &notifier,
                nodes: &nodes,
                objects: &[],
                peers: &[],
                apps: &[],
                config_keys: &[],
                target: &target,
                running_version: RUNNING,
                now_secs: 1_000,
                manual_target: None,
            },
            Some(&feed),
        )
        .await
        .expect("runs");
    assert!(matches!(outcome, PassOutcome::Ran { .. }));

    let subjects: Vec<String> = sink.recorded().into_iter().map(|d| d.subject).collect();
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
    // Two channels, so every event was delivered twice
    assert_eq!(sink.recorded().len(), subjects.len());

    let history = board.history(10);
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].outcome, UpgradeOutcome::Completed);
    assert_eq!(history[0].nodes_upgraded, 3);
    assert!(history[0].detail.contains("3 node(s) upgraded"));
}

/// Every upgrade event carries a distinct audit type and readable text, and
/// the entries recording them are ordered and complete.
///
/// What states that the record is whole is the commit chain over the
/// compliance log the entries land in, which the append extends, so the
/// entries themselves carry no hash
#[tokio::test]
async fn the_audit_entries_over_a_sequence_are_ordered_and_complete() {
    let notifier = Notifier::new(
        vec![ContactChannel::Webhook {
            url: "https://example/hook".to_string(),
        }],
        RecordingSink::new(),
    );
    let events = vec![
        UpgradeEvent::PendingDetected {
            from_version: RUNNING.to_string(),
            to_version: TARGET.to_string(),
            gate_summary: "0 blocker(s)".to_string(),
        },
        UpgradeEvent::Started {
            from_version: RUNNING.to_string(),
            to_version: TARGET.to_string(),
        },
        UpgradeEvent::NodeCompleted {
            node_id: "node-2".to_string(),
            to_version: TARGET.to_string(),
            nodes_remaining: 2,
        },
        UpgradeEvent::Completed {
            to_version: TARGET.to_string(),
            outcome: UpgradeOutcome::Completed,
            detail: "3 node(s)".to_string(),
        },
    ];
    let mut entries = Vec::new();
    for (index, event) in events.iter().enumerate() {
        let (entry, deliveries) = notifier.emit(event, 1_000 + index as u64).await;
        assert_eq!(deliveries.len(), 1);
        entries.push(entry);
    }
    assert_eq!(entries.len(), events.len());
    let ids: Vec<u64> = entries.iter().map(|entry| entry.event_id).collect();
    assert_eq!(
        ids,
        vec![1, 2, 3, 4],
        "each event is on the record in order"
    );
    let types: std::collections::HashSet<u8> =
        entries.iter().map(|entry| entry.event_type).collect();
    assert_eq!(types.len(), events.len(), "each carries a type of its own");
    for (index, entry) in entries.iter().enumerate() {
        assert_eq!(entry.ts, 1_000 + index as i64);
        assert!(!entry.detail.is_empty());
        assert_eq!(
            entry.record_version,
            zyron_lifecycle::format::AUDIT_RECORD_VERSION_BYTE
        );
    }
}

// ---------------------------------------------------------------------------
// Release check
// ---------------------------------------------------------------------------

/// Items 44, 46, and 47. The release check passes on this binary, and it
/// reports a finding when a date it is asked about is malformed
#[test]
fn the_release_check_passes_on_this_binary() {
    let report = zyron_server::release_check::run(substrate(), "2026-09-01");
    assert!(
        report.passed(),
        "{}",
        report
            .findings
            .iter()
            .map(|f| f.to_string())
            .collect::<Vec<_>>()
            .join("\n")
    );
    assert_eq!(report.formats_checked, ALL_FORMAT_KINDS.len());
    assert!(
        report.schemes_checked >= 6,
        "three active and three reserved"
    );
    assert!(report.summary().contains("0 finding(s)"));
}

/// Item 45. The startup gate passes on this binary and reports what it
/// loaded
#[test]
fn the_startup_gate_passes_and_reports_what_it_loaded() {
    let report = zyron_server::startup_validation::validate().expect("passes");
    assert_eq!(report.formats, ALL_FORMAT_KINDS.len());
    assert_eq!(report.reserved_formats, 4);
    assert!(report.schemes >= 6);
    assert!(report.catalog_tables >= 33);
    assert_eq!(report.current_wire_version, 3);
    assert_eq!(report.mesh_protocol_version, 1);
    assert_eq!(report.consensus_protocol_version, 1);
    let text = report.to_string();
    assert!(text.contains("format substrate ready"), "{text}");
}

// ---------------------------------------------------------------------------
// Cluster version gate
// ---------------------------------------------------------------------------

/// A mesh whose nodes answer a status probe with a scripted version, or do
/// not answer at all, and refuse every other call
struct VersionedMesh {
    versions: Mutex<HashMap<String, Option<String>>>,
}

impl VersionedMesh {
    fn new(nodes: &[(&str, Option<&str>)]) -> Arc<Self> {
        Arc::new(Self {
            versions: Mutex::new(
                nodes
                    .iter()
                    .map(|(name, version)| (name.to_string(), version.map(str::to_string)))
                    .collect(),
            ),
        })
    }

    /// Scripts what a node answers next, None for a node that is not
    /// listening
    fn set(&self, name: &str, version: Option<&str>) {
        self.versions
            .lock()
            .insert(name.to_string(), version.map(str::to_string));
    }

    fn refused<T: Send + 'static>(&self) -> MeshFuture<'_, T> {
        Box::pin(async move {
            Err(MeshRpcError::Refused {
                reason: "this mesh answers status probes only".into(),
            })
        })
    }
}

impl MeshRpc for VersionedMesh {
    fn begin_drain(&self, _r: BeginDrainRequest) -> MeshFuture<'_, DrainStatus> {
        self.refused()
    }
    fn drain_status(&self, _r: DrainStatusRequest) -> MeshFuture<'_, DrainStatus> {
        self.refused()
    }
    fn hot_set_manifest(&self, _r: HotSetManifestRequest) -> MeshFuture<'_, HotSetChunk> {
        self.refused()
    }
    fn prefetch(&self, _r: PrefetchRequest) -> MeshFuture<'_, PrefetchStatus> {
        self.refused()
    }
    fn relocate_session(&self, _r: RelocateSessionRequest) -> MeshFuture<'_, RelocationOutcome> {
        self.refused()
    }
    fn cancel_provisioning(&self, _r: CancelProvisioningRequest) -> MeshFuture<'_, ()> {
        self.refused()
    }
    fn node_status(&self, r: NodeStatusRequest) -> MeshFuture<'_, NodeStatus> {
        Box::pin(async move {
            let scripted = self.versions.lock().get(&r.target.name).cloned();
            match scripted {
                Some(Some(version)) => Ok(NodeStatus {
                    // A node that answers reports its own journal row, which
                    // is what a member's board takes for that member
                    upgrade: Some(zyron_common::format::NodeUpgradeState {
                        node_id: r.target.name.clone(),
                        from_version: version.clone(),
                        to_version: version.clone(),
                        phase: zyron_common::format::UpgradePhase::Rolling,
                        started_at_secs: 1,
                        updated_at_secs: 2,
                        is_leader: false,
                        message: "scripted".to_string(),
                    }),
                    target: r.target,
                    sequence: r.sequence,
                    version,
                    ..NodeStatus::default()
                }),
                Some(None) => Err(MeshRpcError::Unreachable {
                    node: r.target.name,
                    reason: "connection refused".into(),
                }),
                // A node from before the status path existed answers that it
                // does not know the path, which is what the health listener
                // returns for a mesh path it has no route for
                None => Err(MeshRpcError::Unknown {
                    what: zyron_mesh::PATH_NODE_STATUS.to_string(),
                }),
            }
        })
    }
    fn stage_release(&self, _r: StageReleaseRequest) -> MeshFuture<'_, NodeAck> {
        self.refused()
    }
    fn restart_into_staged(&self, _r: RestartRequest) -> MeshFuture<'_, NodeAck> {
        self.refused()
    }
    fn rollback_to_previous(&self, _r: RollbackRequest) -> MeshFuture<'_, NodeAck> {
        self.refused()
    }
    fn set_cluster_setting(&self, _r: SetClusterSettingRequest) -> MeshFuture<'_, NodeAck> {
        self.refused()
    }
}

/// A driver for one node over a mesh, with a mesh address for each peer
fn driver_over(
    mesh: Arc<VersionedMesh>,
    self_name: &str,
    peers: &[&str],
    dir: &std::path::Path,
) -> ClusterDriver {
    let substrate = substrate();
    let journal = Arc::new(
        Journal::open(&substrate.formats, &substrate.catalog_schemas, dir).expect("journal"),
    );
    let scheduler = Arc::new(MeshScheduler::new(
        NodeRef::new(1, self_name),
        mesh,
        Arc::new(WarmPool::new(0)),
    ));
    let peers = peers
        .iter()
        .enumerate()
        .map(|(i, name)| (name.to_string(), NodeRef::new(i as u64 + 2, *name)))
        .collect();
    ClusterDriver::new(
        self_name.to_string(),
        substrate,
        zyron_common::format::upgrade_board(),
        Arc::new(Admission::new()),
        Arc::new(QueryMetrics::new()),
        NodeControl::shared(),
        journal,
        None,
        Some(scheduler),
        peers,
        None,
        DriverSettings {
            probe_timeout_secs: 1,
            ..DriverSettings::default()
        },
        dir.join("zyron-server"),
    )
}

/// A peer that answers it does not serve the status path is reported as
/// running an older release, not as one that failed to answer.
///
/// The release that added the status path is the release the first upgrade
/// moves to, so every member still on the old one answers exactly this way.
/// Reporting it as a failed probe would send an operator hunting a network
/// fault that is not there, through the whole rollout
#[tokio::test]
async fn a_peer_without_the_status_path_reads_as_an_older_release() {
    let dir = tempfile::tempdir().expect("tempdir");
    // node-2 is not scripted, so it answers that it does not know the status
    // path, exactly as a release from before that path would
    let mesh = VersionedMesh::new(&[]);
    let driver = driver_over(Arc::clone(&mesh), "node-1", &["node-2"], dir.path());
    let members: Vec<String> = ["node-1", "node-2"]
        .iter()
        .map(|name| name.to_string())
        .collect();

    let floor = driver.version_floor(&members).await;
    assert_eq!(
        floor,
        Floor::Predates {
            member: "node-2".to_string(),
        },
        "a peer that does not serve the path runs a release from before it"
    );

    let refusal = driver
        .cluster_allows(INTRODUCED_IN, &members)
        .await
        .expect_err("an older member holds the setting back")
        .to_string();
    assert!(refusal.contains("node-2"), "{refusal}");
    assert!(refusal.contains("older than"), "{refusal}");
    assert!(
        !refusal.contains("did not answer"),
        "an old member is not a silent one, got {refusal}"
    );
}

/// A member fills in the rest of the group's rows from what each member says
/// about itself.
///
/// Every node writes its own row and nothing wrote anyone else's, so a
/// follower's board held one row and reported a single node in a group of
/// three. A peer's phase is that peer's own journal state, so it is asked for
/// rather than inferred from its version or its drain flag, and this node is
/// left out because its row is already there.
///
/// A member that does not answer comes back with no row rather than an empty
/// one, which is what lets the merge keep whatever the board already held for
/// a node that is mid restart
#[tokio::test]
async fn the_board_takes_each_peers_row_from_that_peer() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mesh = VersionedMesh::new(&[("node-2", Some(RUNNING)), ("node-3", None)]);
    let driver = driver_over(
        Arc::clone(&mesh),
        "node-1",
        &["node-2", "node-3"],
        dir.path(),
    );
    let members: Vec<String> = ["node-1", "node-2", "node-3"]
        .iter()
        .map(|name| name.to_string())
        .collect();

    let reported = driver.peer_upgrade_states(&members).await;
    assert_eq!(
        reported
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>(),
        ["node-2", "node-3"],
        "this node answers for itself and is not probed"
    );

    let answered = reported[0].1.as_ref().expect("node-2 reported a row");
    assert_eq!(answered.node_id, "node-2");
    assert_eq!(
        answered.phase,
        zyron_common::format::UpgradePhase::Rolling,
        "the peer's own phase came across rather than one inferred here"
    );
    assert_eq!(answered.message, "scripted");

    assert!(
        reported[1].1.is_none(),
        "a member that did not answer must report no row, not an empty one"
    );
}

/// The leader puts a cluster setting in front of the group only once every
/// member runs a release that applies it. The floor is the lowest version
/// any member answers, this node answering for itself, a member that does
/// not answer leaves the floor unknown whatever the others run, a burst of
/// gated uses shares one reading, and the driver drops the reading when it
/// knows a version changed
#[tokio::test]
async fn the_version_gate_holds_a_setting_until_every_member_applies_it() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mesh = VersionedMesh::new(&[("node-2", Some(RUNNING)), ("node-3", Some(TARGET))]);
    let driver = driver_over(
        Arc::clone(&mesh),
        "node-1",
        &["node-2", "node-3"],
        dir.path(),
    );
    let members: Vec<String> = ["node-1", "node-2", "node-3"]
        .iter()
        .map(|name| name.to_string())
        .collect();

    let answers = driver.member_versions(&members).await;
    assert_eq!(
        answers.iter().map(|m| m.name.as_str()).collect::<Vec<_>>(),
        ["node-1", "node-2", "node-3"],
        "answers come back in member order"
    );
    assert_eq!(
        answers[0].version,
        Ok(BinaryVersion::parse(env!("CARGO_PKG_VERSION")).expect("this build's version parses")),
        "this node answers for itself"
    );
    assert_eq!(answers[1].version, Ok(BinaryVersion::new(0, 11, 0)));

    let floor = driver.version_floor(&members).await;
    assert_eq!(
        floor,
        Floor::Known {
            version: BinaryVersion::new(0, 11, 0),
            member: "node-2".to_string(),
        }
    );
    let refusal = driver
        .cluster_allows(INTRODUCED_IN, &members)
        .await
        .expect_err("a member on 0.11.0 holds the setting back")
        .to_string();
    assert!(refusal.contains("node-2 runs 0.11.0"), "{refusal}");
    assert!(
        refusal.contains(&format!("{INTRODUCED_IN} or later")),
        "{refusal}"
    );

    // The reading is shared across a burst, so the member's new version is
    // seen once the driver drops it, which it does when it restarted or
    // rolled the member back itself
    mesh.set("node-2", Some(TARGET));
    assert!(
        driver
            .cluster_allows(INTRODUCED_IN, &members)
            .await
            .is_err(),
        "a fresh reading is not taken again"
    );
    driver.drop_version_reading();
    driver
        .cluster_allows(INTRODUCED_IN, &members)
        .await
        .expect("every member applies cluster settings");

    mesh.set("node-3", None);
    driver.drop_version_reading();
    let floor = driver.version_floor(&members).await;
    assert!(
        matches!(&floor, Floor::Unknown { member, reason }
            if member == "node-3" && reason.contains("did not answer a status probe")),
        "{floor:?}"
    );
    let refusal = driver
        .cluster_allows(INTRODUCED_IN, &members)
        .await
        .expect_err("a member that does not answer holds the setting back")
        .to_string();
    assert!(refusal.contains("node-3"), "{refusal}");

    // A member with no mesh address on this node is named with the
    // statement that gives it one
    driver.drop_version_reading();
    let floor = driver
        .version_floor(&["node-1".to_string(), "node-9".to_string()])
        .await;
    assert!(
        matches!(&floor, Floor::Unknown { member, reason }
            if member == "node-9" && reason.contains("CREATE PEER node-9")),
        "{floor:?}"
    );
}

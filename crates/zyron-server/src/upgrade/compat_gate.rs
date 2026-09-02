//! The compatibility gate.
//!
//! Before a node stages anything, the gate asks four questions and answers
//! them from evidence rather than from a version number. Can the target read
//! every format version this cluster has on disk. Does every config key the
//! cluster sets still exist, or does a migrator handle it. What would the
//! target do to the user-authored objects here. Are the federated peers and
//! the deployed Apps ready for it.
//!
//! The gate never guesses. Anything it cannot verify is reported as a
//! blocker naming what it could not verify, because an upgrade that proceeds
//! on an unanswered question is the one that strands a cluster

use zyron_common::format::rewrite::{
    RewriteClassification, RewriteDisposition, UserObjectRewritePolicy,
};
use zyron_common::format::{
    ALL_FORMAT_KINDS, BinaryVersion, FormatKind, FormatRegistry, FormatVersion, ReleaseManifest,
};
use zyron_common::{Result, ZyronError};
use zyron_parser::rewriter::{self, ProposedRewrite};

/// One reason the gate refused
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Blocker {
    /// The target binary has no reader for a format version on disk here
    FormatUnreadable {
        kind: FormatKind,
        persisted: FormatVersion,
        target_oldest: FormatVersion,
    },
    /// A config key this cluster sets is gone in the target and no migrator
    /// handles it
    ConfigKeyRemoved { key: String },
    /// A user-authored object cannot be rewritten safely
    UnsafeRewrite { object_name: String, reason: String },
    /// A federated peer is on a version the target cannot talk to
    PeerIncompatible { peer: String, peer_version: String },
    /// An App deployed here declares it does not work with the target
    AppIncompatible { app: String, declared: String },
    /// The target is not on the feed, so the chain to it cannot be planned
    NoChain { from: String, to: String },
}

impl std::fmt::Display for Blocker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Blocker::FormatUnreadable {
                kind,
                persisted,
                target_oldest,
            } => write!(
                f,
                "the target cannot read {kind} files at version {persisted}, its oldest \
                 reader is {target_oldest}. Upgrade through a release that still reads \
                 {persisted} first"
            ),
            Blocker::ConfigKeyRemoved { key } => write!(
                f,
                "config key `{key}` is removed in the target and no migrator handles it"
            ),
            Blocker::UnsafeRewrite {
                object_name,
                reason,
            } => write!(
                f,
                "`{object_name}` cannot be rewritten automatically, {reason}. Rewrite it or \
                 accept that it will break before upgrading"
            ),
            Blocker::PeerIncompatible { peer, peer_version } => write!(
                f,
                "federated peer `{peer}` is on {peer_version}, which the target cannot talk \
                 to. Upgrade the peer first, then retry"
            ),
            Blocker::AppIncompatible { app, declared } => write!(
                f,
                "App `{app}` declares compatibility with {declared}, which does not include \
                 the target"
            ),
            Blocker::NoChain { from, to } => write!(
                f,
                "no upgrade chain from {from} to {to} is on the release feed"
            ),
        }
    }
}

/// What the gate found
#[derive(Debug, Clone, Default)]
pub struct GateReport {
    pub from_version: String,
    pub to_version: String,
    /// The versions to pass through, target last. One entry means direct
    pub chain: Vec<String>,
    pub blockers: Vec<Blocker>,
    pub classification: RewriteClassification,
    /// Every rewrite the gate proposed, whatever its class
    pub proposals: Vec<ProposedRewrite>,
    /// Formats the target will migrate, and from which version
    pub format_migrations: Vec<(FormatKind, FormatVersion, FormatVersion)>,
    /// Peers the gate checked and found ready
    pub peers_ready: Vec<String>,
    pub policy: UserObjectRewritePolicy,
}

impl GateReport {
    /// Whether the upgrade may proceed without an operator
    pub fn passes(&self) -> bool {
        self.blockers.is_empty() && !self.needs_ack()
    }

    /// Whether an operator has to acknowledge something first
    pub fn needs_ack(&self) -> bool {
        self.classification.needs_ack(self.policy)
    }

    /// Whether the upgrade is refused outright
    pub fn blocked(&self) -> bool {
        !self.blockers.is_empty() || self.classification.blocks_upgrade(self.policy)
    }

    /// The line the CLI and the notification both print
    pub fn summary(&self) -> String {
        let chain = if self.chain.len() > 1 {
            format!(" through {}", self.chain.join(" then "))
        } else {
            String::new()
        };
        format!(
            "{} to {}{chain}, {}, {} blocker(s), {} format migration(s), {} peer(s) ready",
            self.from_version,
            self.to_version,
            self.classification.summary(),
            self.blockers.len(),
            self.format_migrations.len(),
            self.peers_ready.len()
        )
    }

    /// Every blocker as text, which is what an error message carries
    pub fn blocker_text(&self) -> String {
        self.blockers
            .iter()
            .map(|b| b.to_string())
            .collect::<Vec<_>>()
            .join("; ")
    }
}

/// What the target binary says it can read.
///
/// A real upgrade reads this out of the staged binary's own registry. A dry
/// run against a candidate reads it the same way, and a test supplies one
/// directly, so all three go through one shape
#[derive(Debug, Clone, Default)]
pub struct TargetCapabilities {
    pub version: String,
    /// Per format, the oldest version the target still reads
    pub reader_floor: Vec<(FormatKind, FormatVersion)>,
    /// Config keys the target no longer accepts and does not migrate
    pub removed_config_keys: Vec<String>,
}

impl TargetCapabilities {
    /// The capabilities of the running binary, which is what a same-version
    /// dry run compares against
    pub fn of_running(registry: &FormatRegistry, version: &str) -> Self {
        Self {
            version: version.to_string(),
            reader_floor: registry
                .entries()
                .map(|entry| {
                    (
                        entry.registration.kind,
                        entry.registration.reader_supported_versions.oldest,
                    )
                })
                .collect(),
            removed_config_keys: Vec::new(),
        }
    }

    /// The oldest version the target reads for a format, or None when the
    /// target does not carry that format at all
    pub fn floor_for(&self, kind: FormatKind) -> Option<FormatVersion> {
        self.reader_floor
            .iter()
            .find(|(k, _)| *k == kind)
            .map(|(_, v)| *v)
    }
}

/// One user-authored object the gate classifies
#[derive(Debug, Clone)]
pub struct UserObject {
    pub name: String,
    pub kind: zyron_common::format::rewrite::ObjectKind,
    pub sql: String,
}

/// One federated peer the gate checks
#[derive(Debug, Clone)]
pub struct PeerState {
    pub name: String,
    pub version: String,
    /// Whether the peer answered at all. A peer that did not is a blocker,
    /// never a pass
    pub reachable: bool,
}

/// Asks every federated peer what version it runs, giving up on one that
/// does not answer inside the coordination timeout.
///
/// A peer that times out comes back unreachable rather than absent, so the
/// gate blocks on it instead of quietly proceeding without its answer. That
/// is the whole point of the timeout: an unanswered peer is a peer that
/// would be stranded
pub async fn gather_peers<F, Fut>(peers: &[String], timeout_secs: u64, probe: F) -> Vec<PeerState>
where
    F: Fn(String) -> Fut,
    Fut: std::future::Future<Output = Result<String>>,
{
    let mut out = Vec::with_capacity(peers.len());
    let budget = std::time::Duration::from_secs(timeout_secs);
    for name in peers {
        let answered = tokio::time::timeout(budget, probe(name.clone())).await;
        out.push(match answered {
            Ok(Ok(version)) => PeerState {
                name: name.clone(),
                version,
                reachable: true,
            },
            Ok(Err(e)) => PeerState {
                name: name.clone(),
                version: e.to_string(),
                reachable: false,
            },
            Err(_) => PeerState {
                name: name.clone(),
                version: format!("no answer in {timeout_secs}s"),
                reachable: false,
            },
        });
    }
    out
}

/// One App deployed on the cluster
#[derive(Debug, Clone)]
pub struct AppCompat {
    pub name: String,
    /// The version range the App declares it works with
    pub declared_range: String,
    pub compatible: bool,
}

/// What the gate is asked to check
pub struct GateInput<'a> {
    pub from_version: &'a str,
    pub to_version: &'a str,
    pub manifest: Option<&'a ReleaseManifest>,
    pub registry: &'a FormatRegistry,
    pub target: &'a TargetCapabilities,
    /// The format versions actually on disk here, per kind
    pub persisted: &'a [(FormatKind, FormatVersion)],
    pub config_keys: &'a [String],
    pub objects: &'a [UserObject],
    pub peers: &'a [PeerState],
    pub apps: &'a [AppCompat],
    pub policy: UserObjectRewritePolicy,
}

/// Runs the gate
pub fn run(input: GateInput<'_>) -> Result<GateReport> {
    if BinaryVersion::parse(input.to_version).is_none() {
        return Err(ZyronError::UpgradeRefused(format!(
            "`{}` is not a major.minor.patch version",
            input.to_version
        )));
    }
    let mut report = GateReport {
        from_version: input.from_version.to_string(),
        to_version: input.to_version.to_string(),
        policy: input.policy,
        ..GateReport::default()
    };

    report.chain = match input.manifest {
        Some(manifest) => {
            let chain = manifest.plan_chain(input.from_version, input.to_version);
            if chain.is_empty() {
                report.blockers.push(Blocker::NoChain {
                    from: input.from_version.to_string(),
                    to: input.to_version.to_string(),
                });
            }
            chain
        }
        None => vec![input.to_version.to_string()],
    };

    check_formats(&input, &mut report);
    check_config(&input, &mut report);
    classify_objects(&input, &mut report)?;
    check_peers(&input, &mut report);
    check_apps(&input, &mut report);

    Ok(report)
}

/// Every persisted format version has to be inside the target's window
fn check_formats(input: &GateInput<'_>, report: &mut GateReport) {
    for (kind, persisted) in input.persisted {
        let Some(floor) = input.target.floor_for(*kind) else {
            // The target does not carry this format at all, which is the
            // same problem as an unreadable version and reads the same way
            report.blockers.push(Blocker::FormatUnreadable {
                kind: *kind,
                persisted: *persisted,
                target_oldest: FormatVersion::V1,
            });
            continue;
        };
        if *persisted < floor {
            report.blockers.push(Blocker::FormatUnreadable {
                kind: *kind,
                persisted: *persisted,
                target_oldest: floor,
            });
            continue;
        }
        if let Some(entry) = input.registry.get(*kind) {
            let current = entry.registration.writer_current_version;
            if *persisted < current {
                report.format_migrations.push((*kind, *persisted, current));
            }
        }
    }
}

/// A config key the target dropped is a blocker unless it migrates
fn check_config(input: &GateInput<'_>, report: &mut GateReport) {
    for key in input.config_keys {
        if input
            .target
            .removed_config_keys
            .iter()
            .any(|removed| removed.eq_ignore_ascii_case(key))
        {
            report
                .blockers
                .push(Blocker::ConfigKeyRemoved { key: key.clone() });
        }
    }
}

/// Every user-authored object is put through every registered rewriter
fn classify_objects(input: &GateInput<'_>, report: &mut GateReport) -> Result<()> {
    for object in input.objects {
        let proposals = match rewriter::dry_run(&object.name, object.kind, &object.sql) {
            Ok(proposals) => proposals,
            Err(e) => {
                // An object the target cannot parse is exactly the case the
                // gate exists for, and it blocks rather than being skipped
                report.blockers.push(Blocker::UnsafeRewrite {
                    object_name: object.name.clone(),
                    reason: e.to_string(),
                });
                continue;
            }
        };
        if proposals.is_empty() {
            report.classification.record_unaffected();
            continue;
        }
        for proposal in proposals {
            report
                .classification
                .record(&proposal.object_name, proposal.category);
            if proposal.disposition(input.policy) == RewriteDisposition::Block {
                report.blockers.push(Blocker::UnsafeRewrite {
                    object_name: proposal.object_name.clone(),
                    reason: proposal.description.to_string(),
                });
            }
            report.proposals.push(proposal);
        }
    }
    Ok(())
}

/// A peer that is unreachable or on an incompatible version blocks
fn check_peers(input: &GateInput<'_>, report: &mut GateReport) {
    let Some(target) = BinaryVersion::parse(input.to_version) else {
        return;
    };
    for peer in input.peers {
        if !peer.reachable {
            report.blockers.push(Blocker::PeerIncompatible {
                peer: peer.name.clone(),
                peer_version: "unreachable".to_string(),
            });
            continue;
        }
        match BinaryVersion::parse(&peer.version) {
            // A peer within one major of the target can talk to it. A peer a
            // whole major behind cannot, and upgrading past it would strand
            // it
            Some(peer_version) if peer_version.major + 1 >= target.major => {
                report.peers_ready.push(peer.name.clone());
            }
            _ => report.blockers.push(Blocker::PeerIncompatible {
                peer: peer.name.clone(),
                peer_version: peer.version.clone(),
            }),
        }
    }
}

/// An App that declares it does not work with the target blocks
fn check_apps(input: &GateInput<'_>, report: &mut GateReport) {
    for app in input.apps {
        if !app.compatible {
            report.blockers.push(Blocker::AppIncompatible {
                app: app.name.clone(),
                declared: app.declared_range.clone(),
            });
        }
    }
}

/// The format versions on disk, read from the registry's writer versions.
///
/// A node that has been running one binary has every format at its writer
/// version. A node that was upgraded has older files until their policy
/// moves them, which the migration board reports and this reads
pub fn persisted_versions(registry: &FormatRegistry) -> Vec<(FormatKind, FormatVersion)> {
    ALL_FORMAT_KINDS
        .iter()
        .filter_map(|kind| {
            registry
                .get(*kind)
                .map(|entry| (*kind, entry.registration.reader_supported_versions.oldest))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyron_common::format::registry::{DeprecationStatus, FormatRegistration, MigrationPolicy};
    use zyron_common::format::rewrite::ObjectKind;
    use zyron_common::format::version::VersionWindow;

    fn registry() -> FormatRegistry {
        let registrations: Vec<FormatRegistration> = ALL_FORMAT_KINDS
            .iter()
            .copied()
            .map(|kind| FormatRegistration {
                kind,
                writer_current_version: FormatVersion::V1,
                reader_supported_versions: VersionWindow::single(FormatVersion::V1),
                migration_policy: MigrationPolicy::Lazy,
                migration_reversible: true,
                binary_version_gate: "0.11.0",
                deprecation_status: DeprecationStatus::Active,
                retirement_date: None,
                downgrade_write_supported: false,
                notes: "test",
            })
            .collect();
        FormatRegistry::from_parts(&registrations, &[], &[]).expect("loads")
    }

    fn input<'a>(
        registry: &'a FormatRegistry,
        target: &'a TargetCapabilities,
        persisted: &'a [(FormatKind, FormatVersion)],
        objects: &'a [UserObject],
        peers: &'a [PeerState],
        apps: &'a [AppCompat],
        config_keys: &'a [String],
    ) -> GateInput<'a> {
        GateInput {
            from_version: "0.11.0",
            to_version: "0.12.0",
            manifest: None,
            registry,
            target,
            persisted,
            config_keys,
            objects,
            peers,
            apps,
            policy: UserObjectRewritePolicy::AutoSafe,
        }
    }

    #[test]
    fn test_a_clean_cluster_passes() {
        let registry = registry();
        let target = TargetCapabilities::of_running(&registry, "0.12.0");
        let persisted = persisted_versions(&registry);
        let report = run(input(&registry, &target, &persisted, &[], &[], &[], &[])).expect("runs");
        assert!(report.passes(), "{}", report.blocker_text());
        assert!(!report.blocked());
        assert!(report.summary().contains("0 blocker(s)"));
    }

    #[test]
    fn test_a_target_that_dropped_a_reader_blocks_naming_the_format() {
        let registry = registry();
        let mut target = TargetCapabilities::of_running(&registry, "0.12.0");
        for (kind, floor) in target.reader_floor.iter_mut() {
            if *kind == FormatKind::HeapPage {
                *floor = FormatVersion::new(1, 4);
            }
        }
        let persisted = vec![(FormatKind::HeapPage, FormatVersion::V1)];
        let report = run(input(&registry, &target, &persisted, &[], &[], &[], &[])).expect("runs");
        assert!(report.blocked());
        let text = report.blocker_text();
        assert!(text.contains("heap_page"), "{text}");
        assert!(text.contains("1.0"), "{text}");
        assert!(text.contains("Upgrade through"), "{text}");
    }

    #[test]
    fn test_a_removed_config_key_blocks() {
        let registry = registry();
        let mut target = TargetCapabilities::of_running(&registry, "0.12.0");
        target.removed_config_keys.push("wal.sync_mode".to_string());
        let keys = vec!["wal.sync_mode".to_string(), "server.port".to_string()];
        let report = run(input(&registry, &target, &[], &[], &[], &[], &keys)).expect("runs");
        assert!(report.blocked());
        assert!(report.blocker_text().contains("wal.sync_mode"));
    }

    #[test]
    fn test_objects_are_classified_and_unsafe_ones_block() {
        let registry = registry();
        let target = TargetCapabilities::of_running(&registry, "0.12.0");
        let objects = vec![
            UserObject {
                name: "safe_view".to_string(),
                kind: ObjectKind::View,
                sql: "CREATE VIEW safe_view AS SELECT a FROM warehouse_x".to_string(),
            },
            UserObject {
                name: "unaffected".to_string(),
                kind: ObjectKind::View,
                sql: "CREATE VIEW unaffected AS SELECT a FROM untouched".to_string(),
            },
            UserObject {
                name: "broken".to_string(),
                kind: ObjectKind::View,
                sql: "CREATE VIEW broken AS SELECT gone_fn(a) FROM t".to_string(),
            },
        ];
        let report = run(input(&registry, &target, &[], &objects, &[], &[], &[])).expect("runs");
        assert_eq!(report.classification.safe, 1);
        assert_eq!(report.classification.unaffected, 1);
        assert_eq!(report.classification.unsafe_count, 1);
        assert!(report.blocked());
        assert!(report.blocker_text().contains("broken"));
    }

    #[test]
    fn test_an_object_that_does_not_parse_blocks() {
        let registry = registry();
        let target = TargetCapabilities::of_running(&registry, "0.12.0");
        let objects = vec![UserObject {
            name: "bad".to_string(),
            kind: ObjectKind::View,
            sql: "NOT SQL".to_string(),
        }];
        let report = run(input(&registry, &target, &[], &objects, &[], &[], &[])).expect("runs");
        assert!(report.blocked());
        assert!(report.blocker_text().contains("bad"));
    }

    #[test]
    fn test_an_incompatible_peer_blocks_naming_it() {
        let registry = registry();
        let target = TargetCapabilities::of_running(&registry, "0.12.0");
        let peers = vec![
            PeerState {
                name: "eu-west".to_string(),
                version: "0.12.0".to_string(),
                reachable: true,
            },
            PeerState {
                name: "us-east".to_string(),
                version: "0.11.0".to_string(),
                reachable: false,
            },
        ];
        let report = run(input(&registry, &target, &[], &[], &peers, &[], &[])).expect("runs");
        assert!(report.blocked());
        assert!(
            report.blocker_text().contains("us-east"),
            "{}",
            report.blocker_text()
        );
        assert!(report.blocker_text().contains("unreachable"));
        assert_eq!(report.peers_ready, vec!["eu-west".to_string()]);
    }

    #[tokio::test]
    async fn test_a_peer_that_does_not_answer_in_time_comes_back_unreachable() {
        let peers = vec!["fast".to_string(), "slow".to_string(), "broken".to_string()];
        let gathered = gather_peers(&peers, 1, |name| async move {
            match name.as_str() {
                "fast" => Ok("0.12.0".to_string()),
                "broken" => Err(ZyronError::Internal("connection refused".to_string())),
                _ => {
                    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                    Ok("0.12.0".to_string())
                }
            }
        })
        .await;

        assert_eq!(gathered.len(), 3);
        assert!(gathered[0].reachable);
        assert_eq!(gathered[0].version, "0.12.0");
        assert!(!gathered[1].reachable, "the slow peer timed out");
        assert!(gathered[1].version.contains("no answer in 1s"));
        assert!(!gathered[2].reachable, "the broken peer errored");
        assert!(gathered[2].version.contains("connection refused"));

        // A peer that did not answer blocks rather than being skipped
        let registry = registry();
        let target = TargetCapabilities::of_running(&registry, "0.12.0");
        let report = run(input(&registry, &target, &[], &[], &gathered, &[], &[])).expect("runs");
        assert!(report.blocked());
        assert!(
            report.blocker_text().contains("slow"),
            "{}",
            report.blocker_text()
        );
        assert_eq!(report.peers_ready, vec!["fast".to_string()]);
    }

    #[test]
    fn test_an_incompatible_app_blocks_naming_it() {
        let registry = registry();
        let target = TargetCapabilities::of_running(&registry, "0.12.0");
        let apps = vec![AppCompat {
            name: "reporting".to_string(),
            declared_range: ">=0.10, <0.12".to_string(),
            compatible: false,
        }];
        let report = run(input(&registry, &target, &[], &[], &[], &apps, &[])).expect("runs");
        assert!(report.blocked());
        assert!(report.blocker_text().contains("reporting"));
    }

    #[test]
    fn test_a_bad_target_version_is_refused_before_anything_runs() {
        let registry = registry();
        let target = TargetCapabilities::of_running(&registry, "latest");
        let mut gate = input(&registry, &target, &[], &[], &[], &[], &[]);
        gate.to_version = "latest";
        let err = run(gate).expect_err("refused");
        assert!(err.to_string().contains("major.minor.patch"), "{err}");
    }

    #[test]
    fn test_an_ambiguous_rewrite_needs_an_acknowledgment_but_does_not_block() {
        let registry = registry();
        let target = TargetCapabilities::of_running(&registry, "0.12.0");
        let objects = vec![UserObject {
            name: "widened".to_string(),
            kind: ObjectKind::View,
            sql: "CREATE VIEW widened AS SELECT old_agg(a) FROM t".to_string(),
        }];
        let report = run(input(&registry, &target, &[], &objects, &[], &[], &[])).expect("runs");
        assert_eq!(report.classification.ambiguous, 1);
        assert!(report.needs_ack());
        assert!(!report.passes());
        assert!(!report.blocked());
    }
}

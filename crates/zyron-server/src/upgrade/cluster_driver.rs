//! The node driver over a real cluster.
//!
//! The rolling sequence asks a driver to stage, drain, restart, observe, and
//! roll back nodes by name. Here a name is either this node, which is acted
//! on directly, or a peer, which is reached over the mesh. Followers go
//! first and the coordinator is the leader, so every remote node has
//! restarted and been judged healthy before this node's own turn comes.
//!
//! ## This node's own restart
//!
//! A process cannot watch itself come back. Restarting this node activates
//! the staged binary, writes what the next process needs to judge itself
//! into the upgrade journal, arms the restart, and then parks: the run loop
//! drains the node and exits, the new binary starts, and the service in it
//! reads the journal and finishes the story, rolling itself back to the
//! previous binary if it does not reach the baseline.
//!
//! ## The version floor
//!
//! The driver is also where the group's version floor is read. Every member
//! is asked what it runs, this node answering for itself, the lowest answer
//! is the floor, and anything a release adds to what one member puts in
//! front of the others is used only once the floor reaches that release.
//! The reading lives in [`VersionGate`] and is dropped whenever this driver
//! restarts or rolls back a member

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use zyron_auth::signature::VerifyingMaterial;
use zyron_common::format::{
    BinaryVersion, FormatSubstrate, HealthBaseline, ReleaseEntry, UpgradeBoard,
};
use zyron_common::{Admission, QueryMetrics, Result, ZyronError};
use zyron_mesh::{
    MeshScheduler, NodeRef, NodeStatusRequest, RestartRequest, RollbackRequest, StageReleaseRequest,
};
use zyron_raft::RaftNode;

use super::control::{NodeControl, RestartIntent};
use super::journal::{Journal, PendingRestart, RestartKind};
use super::rolling::{NodeDriver, NodePlan, SequencePlan};
use super::stager::{self, ArtifactSource, StagedRelease};
use super::version_gate::{Floor, MemberVersion, VersionGate};

/// Timings the driver works to
#[derive(Debug, Clone, Copy)]
pub struct DriverSettings {
    /// Seconds a node has to finish its in-flight work
    pub drain_timeout_secs: u64,
    /// Seconds a node has to fetch and verify a release
    pub stage_timeout_secs: u64,
    /// Seconds a leadership transfer may take
    pub transfer_timeout_secs: u64,
    /// Seconds a status probe waits for an answer
    pub probe_timeout_secs: u64,
}

impl Default for DriverSettings {
    fn default() -> Self {
        Self {
            drain_timeout_secs: 300,
            stage_timeout_secs: 600,
            transfer_timeout_secs: 30,
            probe_timeout_secs: 10,
        }
    }
}

/// What a running sequence carries, taken when this node's own restart is
/// written to the journal
#[derive(Debug, Clone)]
pub struct SequenceContext {
    pub upgrade_id: u64,
    pub from_version: String,
    pub to_version: String,
    pub channel: String,
    pub started_at_secs: u64,
    pub baseline: HealthBaseline,
    pub nodes_total: u32,
    pub format_migrations: Vec<(String, u32, u32)>,
    pub reversible: bool,
    pub snapshot_path: Option<String>,
    pub actor: String,
}

/// What the service settles before a pass and the sequence context takes
#[derive(Debug, Clone, Default)]
struct PendingSequence {
    snapshot_path: Option<String>,
    actor: String,
}

/// Where this node's releases come from and how they are checked
pub struct ReleaseAccess {
    pub artifacts: Arc<dyn ArtifactSource>,
    pub release_key: VerifyingMaterial,
    pub staging_root: PathBuf,
}

/// The driver
pub struct ClusterDriver {
    self_name: String,
    substrate: &'static FormatSubstrate,
    board: &'static UpgradeBoard,
    admission: Arc<Admission>,
    query_metrics: Arc<QueryMetrics>,
    control: Arc<NodeControl>,
    journal: Arc<Journal>,
    raft: Option<Arc<RaftNode>>,
    /// The mesh, when this node has peers. Remote nodes are reached through
    /// it, and a name it does not know is one this driver cannot act on
    scheduler: Option<Arc<MeshScheduler>>,
    /// Mesh addresses by node name
    peers: HashMap<String, NodeRef>,
    releases: Option<ReleaseAccess>,
    settings: DriverSettings,
    /// The version each remote node is expected to answer with after its
    /// restart, so an answer from the old process is not taken as recovery
    expected: parking_lot::Mutex<HashMap<String, String>>,
    /// The sequence in flight, from `begin_sequence` to this node's restart
    sequence: parking_lot::Mutex<Option<SequenceContext>>,
    /// What the next sequence starts with
    pending: parking_lot::Mutex<PendingSequence>,
    /// Remote nodes restarted so far in the sequence
    restarted: parking_lot::Mutex<Vec<String>>,
    /// The last reading of the group's version floor
    version_gate: VersionGate,
    /// The path this process runs from, which activation replaces
    live_path: PathBuf,
}

impl ClusterDriver {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        self_name: String,
        substrate: &'static FormatSubstrate,
        board: &'static UpgradeBoard,
        admission: Arc<Admission>,
        query_metrics: Arc<QueryMetrics>,
        control: Arc<NodeControl>,
        journal: Arc<Journal>,
        raft: Option<Arc<RaftNode>>,
        scheduler: Option<Arc<MeshScheduler>>,
        peers: HashMap<String, NodeRef>,
        releases: Option<ReleaseAccess>,
        settings: DriverSettings,
        live_path: PathBuf,
    ) -> Self {
        Self {
            self_name,
            substrate,
            board,
            admission,
            query_metrics,
            control,
            journal,
            raft,
            scheduler,
            peers,
            releases,
            settings,
            expected: parking_lot::Mutex::new(HashMap::new()),
            sequence: parking_lot::Mutex::new(None),
            pending: parking_lot::Mutex::new(PendingSequence::default()),
            restarted: parking_lot::Mutex::new(Vec::new()),
            version_gate: VersionGate::new(),
            live_path,
        }
    }

    pub fn self_name(&self) -> &str {
        &self.self_name
    }

    pub fn live_path(&self) -> &std::path::Path {
        &self.live_path
    }

    /// What the service knows before a pass that the sequence context will
    /// need: the snapshot it took and who asked
    pub fn set_pending(&self, snapshot_path: Option<String>, actor: &str) {
        *self.pending.lock() = PendingSequence {
            snapshot_path,
            actor: actor.to_string(),
        };
    }

    /// The sequence in flight, if one has begun
    pub fn sequence(&self) -> Option<SequenceContext> {
        self.sequence.lock().clone()
    }

    /// Whether a node name is this node
    fn is_self(&self, node_id: &str) -> bool {
        node_id == self.self_name
    }

    /// Hands one cluster setting to the leader, which appends it to the
    /// replicated log on its next pass
    pub async fn forward_cluster_setting(
        &self,
        leader: &str,
        key: &str,
        value: &str,
    ) -> Result<()> {
        let (scheduler, node) = self.peer(leader)?;
        let ack = scheduler
            .rpc()
            .set_cluster_setting(zyron_mesh::rpc::SetClusterSettingRequest {
                target: node,
                sequence: scheduler.next_call_sequence(),
                key: key.to_string(),
                value: value.to_string(),
            })
            .await
            .map_err(|e| {
                ZyronError::UpgradeRefused(format!("{leader} could not be handed {key}, {e}"))
            })?;
        if !ack.accepted {
            return Err(ZyronError::UpgradeRefused(format!(
                "{leader} refused {key}, {}",
                ack.detail
            )));
        }
        Ok(())
    }

    fn peer(&self, node_id: &str) -> Result<(Arc<MeshScheduler>, NodeRef)> {
        let scheduler = self.scheduler.clone().ok_or_else(|| {
            ZyronError::UpgradeRefused(format!(
                "{node_id} is another node and this node has no mesh to reach it through. Add \
                 its address with CREATE PEER {node_id} ADDRESS '<host:port>'"
            ))
        })?;
        let node = self.peers.get(node_id).cloned().ok_or_else(|| {
            ZyronError::UpgradeRefused(format!(
                "{node_id} has no mesh address on this node. Add it with CREATE PEER {node_id} \
                 ADDRESS '<host:port>'"
            ))
        })?;
        Ok((scheduler, node))
    }

    fn releases(&self) -> Result<&ReleaseAccess> {
        self.releases.as_ref().ok_or_else(|| {
            ZyronError::UpgradeRefused(
                "upgrade.release_signing_key is not configured, so no release can be verified \
                 and none is staged"
                    .to_string(),
            )
        })
    }

    /// Whether the staged binary for a release is already on disk and still
    /// matches its digest
    fn already_staged(&self, release: &ReleaseEntry) -> Option<StagedRelease> {
        let access = self.releases.as_ref()?;
        let path = access
            .staging_root
            .join(format!("zyron-server-{}", release.version));
        if !path.is_file() {
            return None;
        }
        super::feed::verify_sha256(&path, &release.sha256).ok()?;
        let size_bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        Some(StagedRelease {
            version: release.version.clone(),
            path,
            sha256: release.sha256.clone(),
            signature_scheme: release.signature_scheme.clone(),
            size_bytes,
        })
    }

    /// Stages a release on this node, or reports the one already staged
    pub async fn stage_here(&self, release: &ReleaseEntry) -> Result<StagedRelease> {
        if let Some(staged) = self.already_staged(release) {
            self.control.set_staged(Some(staged.clone()));
            return Ok(staged);
        }
        let access = self.releases()?;
        let staged = stager::stage(
            access.artifacts.as_ref(),
            &self.substrate.schemes,
            &access.release_key,
            release,
            &access.staging_root,
            wall_clock_secs(),
        )
        .await?;
        self.control.set_staged(Some(staged.clone()));
        Ok(staged)
    }

    /// Drains this node: stops taking new work and waits for what is in
    /// flight. On timeout the drain is abandoned so the node serves again
    pub async fn drain_here(&self) -> Result<()> {
        self.admission.begin_drain();
        let deadline =
            std::time::Instant::now() + Duration::from_secs(self.settings.drain_timeout_secs);
        loop {
            if self.admission.is_quiescent() {
                return Ok(());
            }
            if std::time::Instant::now() >= deadline {
                let counts = self.admission.in_flight();
                self.admission.end_drain();
                return Err(ZyronError::UpgradeRefused(format!(
                    "{} did not finish its work inside {}s, {} queries and {} transactions were \
                     still open, so it is serving again",
                    self.self_name,
                    self.settings.drain_timeout_secs,
                    counts.queries,
                    counts.transactions
                )));
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    }

    /// Activates the staged binary, journals the restart, and arms it.
    /// Returns the journal record it wrote
    pub fn arm_upgrade_restart(
        &self,
        staged: &StagedRelease,
        context: &SequenceContext,
        coordinated: bool,
    ) -> Result<PendingRestart> {
        let previous = stager::activate(staged, &self.live_path)?;
        let record = PendingRestart {
            kind: RestartKind::Upgrade,
            upgrade_id: context.upgrade_id,
            node_id: self.self_name.clone(),
            from_version: context.from_version.clone(),
            to_version: context.to_version.clone(),
            channel: context.channel.clone(),
            started_at_secs: context.started_at_secs,
            requested_at_secs: wall_clock_secs(),
            baseline: context.baseline,
            nodes_upgraded_before: self.restarted.lock().len() as u32,
            nodes_total: context.nodes_total,
            format_migrations: context.format_migrations.clone(),
            reversible: context.reversible,
            snapshot_path: context.snapshot_path.clone(),
            live_path: self.live_path.display().to_string(),
            rolled_back_reason: None,
            pause_on_return: false,
            actor: context.actor.clone(),
            coordinated,
        };
        let board_snapshot = self.board.snapshot();
        if let Err(e) = self.journal.update(|journal| {
            journal.board = board_snapshot;
            journal.restart = Some(record.clone());
        }) {
            // The live path already holds the new binary. Put the old one
            // back so a failed journal write does not leave a restart the
            // next process cannot account for
            let _ = std::fs::remove_file(&self.live_path);
            let _ = std::fs::rename(&previous, &self.live_path);
            return Err(e);
        }
        if !self.control.arm_restart(RestartIntent::Upgrade {
            version: context.to_version.clone(),
        }) {
            return Err(ZyronError::UpgradeRefused(
                "a restart is already armed on this node".to_string(),
            ));
        }
        Ok(record)
    }

    /// Puts the previous binary back, journals the restart, and arms it
    pub fn arm_rollback_restart(
        &self,
        from_version: &str,
        to_version: &str,
        reason: &str,
        coordinated: bool,
        pause_on_return: bool,
        actor: &str,
    ) -> Result<()> {
        stager::deactivate(&self.live_path)?;
        let record = PendingRestart {
            kind: RestartKind::Rollback,
            upgrade_id: self.board.next_upgrade_id(),
            node_id: self.self_name.clone(),
            from_version: from_version.to_string(),
            to_version: to_version.to_string(),
            channel: self.board.settings().channel.label().to_string(),
            started_at_secs: wall_clock_secs(),
            requested_at_secs: wall_clock_secs(),
            baseline: HealthBaseline::default(),
            nodes_upgraded_before: 0,
            nodes_total: 1,
            format_migrations: Vec::new(),
            reversible: true,
            snapshot_path: None,
            live_path: self.live_path.display().to_string(),
            rolled_back_reason: Some(reason.to_string()),
            pause_on_return,
            actor: actor.to_string(),
            coordinated,
        };
        let board_snapshot = self.board.snapshot();
        self.journal.update(|journal| {
            journal.board = board_snapshot;
            journal.restart = Some(record);
        })?;
        if !self.control.arm_restart(RestartIntent::Rollback {
            version: to_version.to_string(),
        }) {
            return Err(ZyronError::UpgradeRefused(
                "a restart is already armed on this node".to_string(),
            ));
        }
        Ok(())
    }

    /// This node's health as the connections have measured it
    pub fn observe_here(&self) -> HealthBaseline {
        self.query_metrics
            .sample(wall_clock_secs(), self.admission.in_flight().sessions)
    }

    /// Asks a peer for its status, once
    async fn probe(&self, node_id: &str) -> Result<zyron_mesh::NodeStatus> {
        let (scheduler, node) = self.peer(node_id)?;
        let request = NodeStatusRequest {
            target: node,
            sequence: scheduler.next_call_sequence(),
        };
        let rpc = scheduler.rpc();
        let probe = rpc.node_status(request);
        match tokio::time::timeout(Duration::from_secs(self.settings.probe_timeout_secs), probe)
            .await
        {
            Ok(Ok(status)) => Ok(status),
            Ok(Err(e)) => Err(ZyronError::UpgradeRefused(format!(
                "{node_id} did not answer a status probe, {e}"
            ))),
            Err(_) => Err(ZyronError::UpgradeRefused(format!(
                "{node_id} did not answer a status probe inside {}s",
                self.settings.probe_timeout_secs
            ))),
        }
    }

    /// The version a peer runs, for the gate's peer check and the plan
    pub async fn peer_version(&self, node_id: &str) -> Result<String> {
        Ok(self.probe(node_id).await?.version)
    }

    /// What every member of the group runs, this node answering for itself
    /// and every other member asked over the mesh at the same time, so a
    /// round costs one probe's latency rather than one per member. The
    /// answers come back in the order the members were named
    pub async fn member_versions(&self, members: &[String]) -> Vec<MemberVersion> {
        let running = BinaryVersion::parse(env!("CARGO_PKG_VERSION"));
        let timeout = Duration::from_secs(self.settings.probe_timeout_secs);
        let probes = members.iter().map(|name| async move {
            if self.is_self(name) {
                return MemberVersion {
                    name: name.clone(),
                    version: running.ok_or_else(|| {
                        format!(
                            "this binary reports `{}`, which is not major.minor.patch",
                            env!("CARGO_PKG_VERSION")
                        )
                    }),
                };
            }
            let (scheduler, node) = match self.peer(name) {
                Ok(peer) => peer,
                Err(e) => {
                    return MemberVersion {
                        name: name.clone(),
                        version: Err(e.to_string()),
                    };
                }
            };
            let request = NodeStatusRequest {
                target: node,
                sequence: scheduler.next_call_sequence(),
            };
            let rpc = scheduler.rpc();
            let version = match tokio::time::timeout(timeout, rpc.node_status(request)).await {
                Ok(Ok(status)) => BinaryVersion::parse(&status.version).ok_or_else(|| {
                    format!(
                        "it reports `{}`, which is not major.minor.patch",
                        status.version
                    )
                }),
                Ok(Err(e)) => Err(format!("it did not answer a status probe, {e}")),
                Err(_) => Err(format!(
                    "it did not answer a status probe inside {}s",
                    timeout.as_secs()
                )),
            };
            MemberVersion {
                name: name.clone(),
                version,
            }
        });
        futures::future::join_all(probes).await
    }

    /// The lowest version any member runs, from the last reading while it
    /// is fresh and otherwise from a round of probes taken now
    pub async fn version_floor(&self, members: &[String]) -> Floor {
        if let Some(floor) = self.version_gate.fresh(Instant::now()) {
            return floor;
        }
        let floor = Floor::over(&self.member_versions(members).await);
        self.version_gate.record(floor.clone(), Instant::now());
        floor
    }

    /// Whether something introduced at a release may be put in front of the
    /// group. The refusal names the member that holds it back
    pub async fn cluster_allows(
        &self,
        introduced: BinaryVersion,
        members: &[String],
    ) -> Result<()> {
        self.version_floor(members)
            .await
            .allows(introduced)
            .map_err(ZyronError::UpgradeRefused)
    }

    /// Drops the last reading of the floor, for a moment a member's version
    /// is known to change
    pub fn drop_version_reading(&self) {
        self.version_gate.invalidate();
    }
}

fn wall_clock_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// A node that cannot be observed reads as fully failing, so the health
/// watch keeps polling until it answers or the recovery window closes
fn unreachable() -> HealthBaseline {
    HealthBaseline {
        p50_latency_us: 0,
        p99_latency_us: 0,
        throughput_per_sec: 0.0,
        error_rate: 1.0,
        active_connections: 0,
        queries_in_window: 0,
    }
}

#[async_trait::async_trait]
impl NodeDriver for ClusterDriver {
    async fn stage(&self, node_id: &str, release: &ReleaseEntry) -> Result<()> {
        if self.is_self(node_id) {
            self.stage_here(release).await?;
            return Ok(());
        }
        let (scheduler, node) = self.peer(node_id)?;
        let ack = scheduler
            .rpc()
            .stage_release(StageReleaseRequest {
                target: node.clone(),
                sequence: scheduler.next_call_sequence(),
                version: release.version.clone(),
            })
            .await
            .map_err(|e| {
                ZyronError::UpgradeRefused(format!("{node_id} could not be asked to stage, {e}"))
            })?;
        if !ack.accepted {
            return Err(ZyronError::UpgradeRefused(format!(
                "{node_id} refused to stage {}, {}",
                release.version, ack.detail
            )));
        }
        // The node fetches after answering. Its status says when the
        // release is in place, or its next refusal says why it is not
        let deadline =
            std::time::Instant::now() + Duration::from_secs(self.settings.stage_timeout_secs);
        loop {
            match self.probe(node_id).await {
                Ok(status) if status.staged_version == release.version => return Ok(()),
                Ok(_) => {}
                Err(e) => {
                    tracing::debug!(node = node_id, error = %e, "waiting for the node to stage")
                }
            }
            if std::time::Instant::now() >= deadline {
                let detail = scheduler
                    .rpc()
                    .stage_release(StageReleaseRequest {
                        target: node.clone(),
                        sequence: scheduler.next_call_sequence(),
                        version: release.version.clone(),
                    })
                    .await
                    .map(|ack| ack.detail)
                    .unwrap_or_default();
                return Err(ZyronError::UpgradeRefused(format!(
                    "{node_id} did not stage {} inside {}s. {detail}",
                    release.version, self.settings.stage_timeout_secs
                )));
            }
            tokio::time::sleep(Duration::from_secs(2)).await;
        }
    }

    async fn begin_sequence(
        &self,
        from_version: &str,
        to_version: &str,
        baseline: &HealthBaseline,
        nodes: &[NodePlan],
        plan: &SequencePlan,
    ) -> Result<()> {
        let settings = self.board.settings();
        let pending = self.pending.lock().clone();
        *self.sequence.lock() = Some(SequenceContext {
            upgrade_id: self.board.next_upgrade_id(),
            from_version: from_version.to_string(),
            to_version: to_version.to_string(),
            channel: settings.channel.label().to_string(),
            started_at_secs: wall_clock_secs(),
            baseline: *baseline,
            nodes_total: nodes.len() as u32,
            format_migrations: plan
                .format_migrations
                .iter()
                .map(|(kind, from, to)| {
                    (kind.catalog_name().to_string(), from.as_u32(), to.as_u32())
                })
                .collect(),
            reversible: plan.reversible,
            snapshot_path: pending.snapshot_path,
            actor: pending.actor,
        });
        self.restarted.lock().clear();
        self.expected.lock().clear();
        Ok(())
    }

    async fn drain(&self, node_id: &str) -> Result<()> {
        if self.is_self(node_id) {
            return self.drain_here().await;
        }
        let (scheduler, node) = self.peer(node_id)?;
        let (status, _) = scheduler
            .drain_for_restart(
                &node,
                &[],
                Duration::from_secs(self.settings.drain_timeout_secs),
            )
            .await
            .map_err(|e| {
                ZyronError::UpgradeRefused(format!("{node_id} could not be drained, {e}"))
            })?;
        if !status.drained {
            return Err(ZyronError::UpgradeRefused(format!(
                "{node_id} did not finish its work inside {}s, {} queries and {} transactions \
                 were still open",
                self.settings.drain_timeout_secs,
                status.queries_in_flight,
                status.transactions_open
            )));
        }
        Ok(())
    }

    async fn restart(&self, node_id: &str, to_version: &str) -> Result<()> {
        if self.is_self(node_id) {
            let staged = self.control.staged().ok_or_else(|| {
                ZyronError::UpgradeRefused(format!(
                    "no release is staged on {node_id}, so it cannot restart into {to_version}"
                ))
            })?;
            if staged.version != to_version {
                return Err(ZyronError::UpgradeRefused(format!(
                    "{node_id} has {} staged, not {to_version}",
                    staged.version
                )));
            }
            let context = self.sequence.lock().clone().ok_or_else(|| {
                ZyronError::UpgradeRefused(
                    "no sequence has begun on this node, so its restart has nothing to record"
                        .to_string(),
                )
            })?;
            self.arm_upgrade_restart(&staged, &context, false)?;
            tracing::info!(
                to_version,
                "this node is restarting on the new binary, the next process judges the outcome"
            );
            // The run loop takes it from here. This task ends with the
            // process
            std::future::pending::<()>().await;
            return Ok(());
        }
        let (scheduler, node) = self.peer(node_id)?;
        let ack = scheduler
            .rpc()
            .restart_into_staged(RestartRequest {
                target: node,
                sequence: scheduler.next_call_sequence(),
                version: to_version.to_string(),
            })
            .await
            .map_err(|e| {
                ZyronError::UpgradeRefused(format!("{node_id} could not be asked to restart, {e}"))
            })?;
        if !ack.accepted {
            return Err(ZyronError::UpgradeRefused(format!(
                "{node_id} refused to restart into {to_version}, {}",
                ack.detail
            )));
        }
        self.expected
            .lock()
            .insert(node_id.to_string(), to_version.to_string());
        self.restarted.lock().push(node_id.to_string());
        self.version_gate.invalidate();
        Ok(())
    }

    async fn observe(&self, node_id: &str) -> Result<HealthBaseline> {
        if self.is_self(node_id) {
            return Ok(self.observe_here());
        }
        let expected = self.expected.lock().get(node_id).cloned();
        match self.probe(node_id).await {
            Ok(status) => {
                if let Some(expected) = expected {
                    if status.version != expected {
                        tracing::debug!(
                            node = node_id,
                            running = %status.version,
                            expected = %expected,
                            "the node has not come back on the expected binary yet"
                        );
                        return Ok(unreachable());
                    }
                }
                Ok(HealthBaseline {
                    p50_latency_us: status.p50_latency_us,
                    p99_latency_us: status.p99_latency_us,
                    throughput_per_sec: status.throughput_milli_per_sec as f64 / 1_000.0,
                    error_rate: status.error_rate_ppm as f64 / 1_000_000.0,
                    active_connections: status.sessions_attached as u64,
                    queries_in_window: status.queries_in_window,
                })
            }
            // A node mid-restart cannot answer. Before any restart in this
            // sequence an unanswered probe is a fault the baseline capture
            // has to report, not paper over
            Err(e) if expected.is_some() => {
                tracing::debug!(node = node_id, error = %e, "the node is not answering yet");
                Ok(unreachable())
            }
            Err(e) => Err(e),
        }
    }

    async fn rollback(&self, node_id: &str, to_version: &str) -> Result<()> {
        if self.is_self(node_id) {
            let running = env!("CARGO_PKG_VERSION");
            self.arm_rollback_restart(
                running,
                to_version,
                "rolled back by the rolling sequence",
                false,
                true,
                "",
            )?;
            std::future::pending::<()>().await;
            return Ok(());
        }
        let (scheduler, node) = self.peer(node_id)?;
        let ack = scheduler
            .rpc()
            .rollback_to_previous(RollbackRequest {
                target: node,
                sequence: scheduler.next_call_sequence(),
            })
            .await
            .map_err(|e| {
                ZyronError::UpgradeRefused(format!(
                    "{node_id} could not be asked to roll back, {e}"
                ))
            })?;
        if !ack.accepted {
            return Err(ZyronError::UpgradeRefused(format!(
                "{node_id} refused to roll back, {}",
                ack.detail
            )));
        }
        self.expected
            .lock()
            .insert(node_id.to_string(), to_version.to_string());
        self.version_gate.invalidate();
        Ok(())
    }

    async fn transfer_leadership(&self, from_node: &str) -> Result<()> {
        if !self.is_self(from_node) {
            return Err(ZyronError::UpgradeRefused(format!(
                "leadership is handed off by the node that holds it, and {from_node} is not this \
                 node"
            )));
        }
        let Some(raft) = self.raft.as_ref() else {
            return Ok(());
        };
        match raft
            .transfer_leadership(Duration::from_secs(self.settings.transfer_timeout_secs))
            .await?
        {
            Some(leader) => {
                tracing::info!(leader, "leadership handed off before this node restarts");
                Ok(())
            }
            None => Ok(()),
        }
    }

    async fn wait(&self, secs: u64) {
        tokio::time::sleep(Duration::from_secs(secs)).await;
    }

    fn now_secs(&self) -> u64 {
        wall_clock_secs()
    }
}

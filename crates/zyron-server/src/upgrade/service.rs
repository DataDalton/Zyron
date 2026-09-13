//! The upgrade service, the task that runs upgrades on a node.
//!
//! One instance per process. It seeds the board from the journal and the
//! config, finishes whatever restart the previous process left in the
//! journal, and then loops: carrying out what the DDL surface and the mesh
//! left for it, and, on the node that leads the cluster or stands alone,
//! polling the release feed and running the controller's pass when a
//! release is newer than what runs.
//!
//! ## Who coordinates
//!
//! The leader of the consensus group, or the only node when there is no
//! group. Every other node stages releases and restarts when asked, and
//! serves its status.
//!
//! Leadership can move part way. Restarting a node is what elects a new
//! leader, and the node that takes over is not always one this sequence has
//! already upgraded. So a sequence stops the moment the node running it
//! stops leading, rather than acting on nodes another coordinator now
//! decides for, and the node that leads next runs its own pass over what is
//! left. Two coordinators driving one cluster drain a node neither of them
//! restarts and roll back a node the other is restarting

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use zyron_catalog::Catalog;
use zyron_common::format::registry::MigrationPolicy;
use zyron_common::format::rewrite::{RewriteCategory, RewriteStatus};
use zyron_common::format::{
    BinaryVersion, FormatSubstrate, HealthBaseline, HealthThreshold, NodeUpgradeState,
    UpgradeBoard, UpgradeHistoryEntry, UpgradeOutcome, UpgradePhase,
};
use zyron_common::{Admission, QueryMetrics, Result, ZyronError};
use zyron_mesh::NodeRef;
use zyron_raft::{ClusterConfig, RaftCommand, RaftNode};
use zyron_wire::format_dispatch::UpgradeControl;

use super::capabilities;
use super::catalog_store::HeapCatalogTableStore;
use super::cluster_driver::{ClusterDriver, DriverSettings, ReleaseAccess, SequenceContext};
use super::control::{CoordinatedRestart, ManualRequest, NodeControl};
use super::feed::{
    HttpFeedSource, LayeredFeedSource, LocalFeedSource, ReleaseFeedSource, ReleasePoller,
    ReleaseSigningKey,
};
use super::journal::{CompletedUpgrade, Journal, PendingRestart, RestartKind};
use super::migrations::{self, MigrationBudget};
use super::notification::{ContactChannel, HttpNotificationSink, Notifier, UpgradeEvent};
use super::objects;
use super::rolling::{NodeDriver, NodePlan, RollingSettings};
use super::stager::{
    ArtifactSource, HttpArtifactSource, LayeredArtifactSource, LocalArtifactSource, STAGING_DIR,
};
use super::{PassContext, PassOutcome, UpgradeController, downgrade};
use crate::config::ZyronConfig;

/// Seconds a staged binary has to answer what it can read
const CAPABILITIES_TIMEOUT: Duration = Duration::from_secs(60);

/// Seconds a feed fetch or an artifact download may take
const FEED_TIMEOUT_SECS: u64 = 30;
const ARTIFACT_TIMEOUT_SECS: u64 = 600;

/// Seconds a notification post may take
const NOTIFY_TIMEOUT_SECS: u64 = 30;

/// The contact channels the operator configured, in the order they are
/// notified.
///
/// A configured address outside the Discord webhook shape is refused here,
/// so a configured alerting path is live or the node says why it is not
pub fn contact_channels(config: &ZyronConfig) -> Result<Vec<ContactChannel>> {
    let mut channels = Vec::new();
    if !config.upgrade.notify_webhook_url.trim().is_empty() {
        channels.push(ContactChannel::Webhook {
            url: config.upgrade.notify_webhook_url.trim().to_string(),
        });
    }
    if !config.upgrade.notify_slack_webhook_url.trim().is_empty() {
        channels.push(ContactChannel::Slack {
            webhook_url: config.upgrade.notify_slack_webhook_url.trim().to_string(),
        });
    }
    let discord = config.upgrade.notify_discord_webhook_url.trim();
    if !discord.is_empty() {
        channels.push(ContactChannel::discord(discord).map_err(|reason| {
            ZyronError::Internal(format!("upgrade.notify_discord_webhook_url, {reason}"))
        })?);
    }
    Ok(channels)
}

/// A notifier over the operator's contact channels, delivering over HTTPS
pub fn build_notifier(config: &ZyronConfig) -> Result<Notifier> {
    let channels = contact_channels(config)?;
    let sink = HttpNotificationSink::new(NOTIFY_TIMEOUT_SECS).map_err(ZyronError::Internal)?;
    Ok(Notifier::new(channels, Arc::new(sink)))
}

/// What the server hands the service at boot
pub struct ServiceParts {
    pub config: ZyronConfig,
    pub node_name: String,
    pub catalog: Arc<Catalog>,
    pub raft: Option<Arc<RaftNode>>,
    pub admission: Arc<Admission>,
    pub query_metrics: Arc<QueryMetrics>,
    pub control: Arc<NodeControl>,
    /// Mesh addresses by node name, from the peer registry
    pub peers: Vec<(String, NodeRef)>,
    /// Forces a checkpoint and waits for it, for the pre-upgrade snapshot
    pub checkpoint: Option<Arc<dyn Fn() + Send + Sync>>,
    /// The LSN of the last checkpoint
    pub checkpoint_lsn: Arc<dyn Fn() -> u64 + Send + Sync>,
    pub shutdown: Arc<AtomicBool>,
    /// Raised once every member of the group runs a binary that reads the
    /// actor role off a replicated schema change. None on a node in no group,
    /// where nothing is replicated and there is nothing to hold back
    pub group_carries_actor_role: Option<Arc<AtomicBool>>,
    /// Raised once every member of the group runs a binary that applies a
    /// change stream advance. None on a node in no group
    pub group_carries_stream_advance: Option<Arc<AtomicBool>>,
    /// Raised once every member of the group runs a binary that reads the
    /// row images a feed table's delete or update carries. None on a node
    /// in no group
    pub group_carries_feed_images: Option<Arc<AtomicBool>>,
    /// Raised once every member of the group runs a binary that writes the
    /// files a lake commit carries. None on a node in no group
    pub group_carries_lake_files: Option<Arc<AtomicBool>>,
    /// Raised once every member of the group runs a binary that records a
    /// schedule's run. None on a node in no group
    pub group_carries_schedule_runs: Option<Arc<AtomicBool>>,
}

/// What the post-upgrade migrations did on this node
#[derive(Debug, Clone, Default)]
pub struct MigrationSummary {
    pub formats_run: u32,
    pub catalog_run: u32,
    pub rewrites_written: u32,
    pub migrated_formats: Vec<(String, u32)>,
    pub migrated_tables: Vec<(String, u32)>,
    pub reversible: bool,
    pub detail: Vec<String>,
}

/// The service
pub struct UpgradeService {
    config: ZyronConfig,
    data_dir: PathBuf,
    wal_dir: PathBuf,
    node_name: String,
    substrate: &'static FormatSubstrate,
    board: &'static UpgradeBoard,
    journal: Arc<Journal>,
    control: Arc<NodeControl>,
    driver: Arc<ClusterDriver>,
    controller: UpgradeController,
    notifier: Notifier,
    catalog: Arc<Catalog>,
    raft: Option<Arc<RaftNode>>,
    admission: Arc<Admission>,
    /// Every member of the cluster by name, this node included
    cluster_members: Vec<String>,
    checkpoint: Option<Arc<dyn Fn() + Send + Sync>>,
    checkpoint_lsn: Arc<dyn Fn() -> u64 + Send + Sync>,
    shutdown: Arc<AtomicBool>,
    /// Raised once every member reads the actor role off a schema change, so
    /// the replication path knows it may carry one
    group_carries_actor_role: Option<Arc<AtomicBool>>,
    /// Raised once every member applies a change stream advance, so the
    /// replication path knows a consume may commit
    group_carries_stream_advance: Option<Arc<AtomicBool>>,
    /// Raised once every member reads the row images a feed table's delete
    /// or update carries, so the replication path knows such a write may
    /// commit
    group_carries_feed_images: Option<Arc<AtomicBool>>,
    /// Raised once every member writes the files a lake commit carries, so
    /// the replication path knows a lake write may commit
    group_carries_lake_files: Option<Arc<AtomicBool>>,
    /// Raised once every member records a schedule's run, so the schedule
    /// worker knows a schedule may run
    group_carries_schedule_runs: Option<Arc<AtomicBool>>,
    last_poll: parking_lot::Mutex<Option<Instant>>,
    /// When the board last took the other members' rows, so the probe runs on
    /// its own interval rather than on every pass of the one-second loop
    last_board_reconcile: parking_lot::Mutex<Option<Instant>>,
    /// The last reason a cluster setting could not reach the log, so the
    /// retry each second does not repeat it
    setting_warning: parking_lot::Mutex<Option<String>>,
}

impl UpgradeService {
    /// Builds the service: opens the journal, seeds the board, restores a
    /// catalog table left mid-replacement, and wires the feed, the driver,
    /// and the notifier from the config
    pub async fn boot(parts: ServiceParts) -> Result<Arc<Self>> {
        let substrate = zyron_common::format::substrate()?;
        let board = zyron_common::format::upgrade_board();
        let data_dir = parts.config.storage.data_dir.clone();
        let wal_dir = parts.config.wal_dir();

        let journal = Arc::new(Journal::open(
            &substrate.formats,
            &substrate.catalog_schemas,
            &data_dir,
        )?);
        journal.restore_board(board);
        let settings = parts.config.upgrade.to_settings(&data_dir);
        substrate
            .warning_limiter
            .set_limit(settings.deprecation_warning_rate_limit_per_hour);
        board.update_settings(|slot| *slot = settings.clone());

        if let Some(table) =
            HeapCatalogTableStore::restore_preimage(&parts.catalog, &journal).await?
        {
            tracing::warn!(
                table,
                "a catalog table was left mid-replacement by the last process, its rows were put back"
            );
        }

        // The feed is the local releases directory first, which is where a
        // release delivered by hand lands, and the remote feed behind it.
        // The key is the vendor's built into this binary unless the config
        // names another, so a node checks releases from its first start
        let staging_root = data_dir.join(STAGING_DIR);
        let local_dir = parts.config.upgrade.local_feed_dir(&data_dir);
        let remote_url = parts.config.upgrade.remote_feed_url();
        let material = parts.config.upgrade.release_key_material()?;
        let scheme_name = parts.config.upgrade.release_signing_scheme_name();
        let remote_feed: Option<Arc<dyn ReleaseFeedSource>> = match &remote_url {
            Some(url) => Some(Arc::new(HttpFeedSource::new(
                url.as_str(),
                FEED_TIMEOUT_SECS,
            )?)),
            None => None,
        };
        let remote_artifacts: Option<Arc<dyn ArtifactSource>> = match &remote_url {
            Some(_) => Some(Arc::new(HttpArtifactSource::new(ARTIFACT_TIMEOUT_SECS)?)),
            None => None,
        };
        let source: Arc<dyn ReleaseFeedSource> = Arc::new(LayeredFeedSource::new(
            LocalFeedSource::new(local_dir.clone()),
            remote_feed,
        ));
        let artifacts: Arc<dyn ArtifactSource> = Arc::new(LayeredArtifactSource::new(
            LocalArtifactSource::new(local_dir),
            remote_artifacts,
        ));
        tracing::info!(
            feed = %source.describe(),
            channel = settings.channel.label(),
            key = if parts.config.upgrade.release_signing_key.trim().is_empty() {
                "built in"
            } else {
                "configured"
            },
            "release polling is on"
        );
        let poller = Some(Arc::new(ReleasePoller::new(
            source,
            ReleaseSigningKey {
                scheme_name,
                material: material.clone(),
            },
        )));
        let releases = Some(ReleaseAccess {
            artifacts,
            release_key: material,
            staging_root: staging_root.clone(),
        });

        let live_path = std::env::current_exe().map_err(|e| {
            ZyronError::Internal(format!("the running binary's path is not known, {e}"))
        })?;
        let scheduler = zyron_mesh::MeshActuator::scheduler().cloned();
        let peers: HashMap<String, NodeRef> = parts.peers.into_iter().collect();
        let driver = Arc::new(ClusterDriver::new(
            parts.node_name.clone(),
            substrate,
            board,
            Arc::clone(&parts.admission),
            Arc::clone(&parts.query_metrics),
            Arc::clone(&parts.control),
            Arc::clone(&journal),
            parts.raft.clone(),
            scheduler,
            peers,
            releases,
            DriverSettings {
                drain_timeout_secs: parts.config.upgrade.drain_timeout_secs,
                stage_timeout_secs: parts.config.upgrade.stage_timeout_secs,
                ..DriverSettings::default()
            },
            live_path,
        ));

        let rolling = RollingSettings {
            health_recovery_timeout_secs: parts.config.upgrade.health_recovery_timeout_secs,
            health_poll_interval_secs: parts.config.upgrade.health_poll_interval_secs,
            threshold: HealthThreshold {
                latency_multiplier: parts.config.upgrade.health_latency_multiplier,
                throughput_floor: parts.config.upgrade.health_throughput_floor,
                error_rate_ceiling: parts.config.upgrade.health_error_rate_ceiling,
            },
            rollback_on_health_fail: parts.config.upgrade.rollback_on_health_fail,
        };
        let mut controller = UpgradeController::new(staging_root).with_rolling(rolling);
        if let Some(poller) = poller {
            controller = controller.with_poller(poller);
        }

        let notifier = build_notifier(&parts.config)?;

        let cluster_members = if parts.config.cluster.enabled {
            parts
                .config
                .cluster
                .peers
                .iter()
                .map(|peer| peer.name.clone())
                .collect()
        } else {
            vec![parts.node_name.clone()]
        };

        Ok(Arc::new(Self {
            last_board_reconcile: parking_lot::Mutex::new(None),
            config: parts.config,
            data_dir,
            wal_dir,
            node_name: parts.node_name,
            substrate,
            board,
            journal,
            control: parts.control,
            driver,
            controller,
            notifier,
            catalog: parts.catalog,
            raft: parts.raft,
            admission: parts.admission,
            cluster_members,
            checkpoint: parts.checkpoint,
            checkpoint_lsn: parts.checkpoint_lsn,
            shutdown: parts.shutdown,
            group_carries_actor_role: parts.group_carries_actor_role,
            group_carries_stream_advance: parts.group_carries_stream_advance,
            group_carries_feed_images: parts.group_carries_feed_images,
            group_carries_lake_files: parts.group_carries_lake_files,
            group_carries_schedule_runs: parts.group_carries_schedule_runs,
            last_poll: parking_lot::Mutex::new(None),
            setting_warning: parking_lot::Mutex::new(None),
        }))
    }

    pub fn journal(&self) -> &Arc<Journal> {
        &self.journal
    }

    pub fn control(&self) -> &Arc<NodeControl> {
        &self.control
    }

    fn running_version() -> &'static str {
        env!("CARGO_PKG_VERSION")
    }

    /// Whether this node drives the cluster's upgrades
    pub fn is_coordinator(&self) -> bool {
        match self.raft.as_ref() {
            Some(raft) => raft.is_leader(),
            None => true,
        }
    }

    /// The name of the node that does, for a refusal to name it
    fn coordinator_name(&self) -> String {
        let Some(raft) = self.raft.as_ref() else {
            return self.node_name.clone();
        };
        let leader = raft.leader_id();
        self.cluster_members
            .iter()
            .find(|name| Some(zyron_raft::node_id_for_name(name)) == leader)
            .cloned()
            .unwrap_or_else(|| "the leader, which has not been elected yet".to_string())
    }

    fn persist_board(&self) {
        if let Err(e) = self.journal.sync_board(self.board) {
            tracing::error!(error = %e, "the upgrade journal could not be written");
        }
    }

    /// Records a pause the way an operator's `ALTER SYSTEM` would, so the
    /// next boot reads it from the config
    /// Pauses automatic upgrades, here at once and on every node through
    /// the replicated log, which the loop carries the request to
    fn persist_pause(&self) {
        self.board
            .update_settings(|settings| settings.paused = true);
        if let Err(e) = ZyronConfig::write_auto_conf(&self.data_dir, "upgrade.paused", "true") {
            tracing::error!(error = %e, "upgrade.paused could not be written to zyron.auto.conf");
        }
        self.control
            .request_cluster_setting("upgrade.paused", "true");
        self.persist_board();
    }

    /// Writes one upgrade setting to the replicated log. The leader appends
    /// it once every member of the group runs a binary that applies it, any
    /// other node hands it to the leader, and a node with no group applied
    /// it locally already, which is all there is to do
    async fn carry_cluster_setting(&self, key: &str, value: &str) -> Result<()> {
        let Some(raft) = self.raft.as_ref() else {
            return Ok(());
        };
        if raft.is_leader() {
            // A member on a release before cluster settings refuses the
            // entry as corruption and stops applying, so the entry waits
            // for the group's version floor rather than stalling a member.
            // The members are the group's live membership, not the config's
            // peer list, so a member the config does not name holds the
            // entry rather than being passed over
            let members = group_member_names(
                &raft.cluster_config(),
                &self.node_name,
                &self.cluster_members,
            )?;
            self.driver
                .cluster_allows(crate::cluster_settings::INTRODUCED_IN, &members)
                .await?;
            raft.propose(RaftCommand::Put {
                key: key.as_bytes().to_vec(),
                value: value.as_bytes().to_vec(),
            })
            .await?;
            return Ok(());
        }
        let leader = self.coordinator_name();
        self.driver
            .forward_cluster_setting(&leader, key, value)
            .await
    }

    /// Reads whether every member of the group runs a binary that takes the
    /// actor role off a replicated schema change, and lets the replication
    /// path know.
    ///
    /// Only the leader proposes a schema change, so only the leader has to
    /// know. The flag is lowered again the moment this node stops leading or
    /// a member's version cannot be read, because a member that joined on an
    /// older release must not be sent an operation it refuses. The reading is
    /// the one the version gate already caches, so this costs a map lookup on
    /// most passes rather than a round of mesh calls
    async fn refresh_actor_role_carriage(&self) {
        self.refresh_carriage(
            self.group_carries_actor_role.as_ref(),
            crate::replication::ACTOR_ROLE_INTRODUCED_IN,
        )
        .await;
    }

    /// Reads whether every member of the group applies a change stream
    /// advance, and lets the replication path know.
    ///
    /// Same shape as the actor role reading and lowered for the same
    /// reasons. While it is down a transactional consume on this node is
    /// refused, because an advance has no shorter form to fall back to
    async fn refresh_stream_advance_carriage(&self) {
        self.refresh_carriage(
            self.group_carries_stream_advance.as_ref(),
            crate::replication::STREAM_ADVANCE_INTRODUCED_IN,
        )
        .await;
    }

    /// Reads whether every member of the group reads the row images a feed
    /// table's delete or update carries, and lets the replication path
    /// know.
    ///
    /// Same shape as the two readings above. While it is down a delete or
    /// update on a table with a change data feed is refused on this node,
    /// because a key sent in place of the row would leave every member's
    /// feed short of it
    async fn refresh_feed_images_carriage(&self) {
        self.refresh_carriage(
            self.group_carries_feed_images.as_ref(),
            crate::replication::FEED_IMAGES_INTRODUCED_IN,
        )
        .await;
    }

    /// Reads whether every member of the group writes the files a lake
    /// commit carries. While it is down a lake write is refused on this
    /// node, because a version sent without its files would name bytes a
    /// member does not hold
    async fn refresh_lake_files_carriage(&self) {
        self.refresh_carriage(
            self.group_carries_lake_files.as_ref(),
            crate::replication::LAKE_FILES_INTRODUCED_IN,
        )
        .await;
    }

    /// Reads whether every member of the group records a schedule's run.
    /// While it is down no schedule runs on this node, because a run
    /// recorded here alone would be run again by the next leader
    async fn refresh_schedule_runs_carriage(&self) {
        self.refresh_carriage(
            self.group_carries_schedule_runs.as_ref(),
            crate::replication::SCHEDULE_RUNS_INTRODUCED_IN,
        )
        .await;
    }

    /// Raises `flag` when this node leads and the group's version floor
    /// stands at `introduced_in` or later, and lowers it otherwise, which is
    /// when this node stops leading, when a member joined on an older
    /// release, and when half the members or more cannot be asked what they
    /// run. A member that is away while the rest answer does not lower it,
    /// because it commits nothing until it is back. The reading is the one
    /// the version gate already caches, so this costs a map lookup on most
    /// passes rather than a round of mesh calls
    async fn refresh_carriage(
        &self,
        flag: Option<&Arc<AtomicBool>>,
        introduced_in: zyron_common::format::BinaryVersion,
    ) {
        let Some(flag) = flag else {
            return;
        };
        let Some(raft) = self.raft.as_ref() else {
            return;
        };
        if !raft.is_leader() {
            flag.store(false, Ordering::Relaxed);
            return;
        }
        let members = match group_member_names(
            &raft.cluster_config(),
            &self.node_name,
            &self.cluster_members,
        ) {
            Ok(members) => members,
            Err(_) => {
                flag.store(false, Ordering::Relaxed);
                return;
            }
        };
        let carried = self
            .driver
            .cluster_allows(introduced_in, &members)
            .await
            .is_ok();
        flag.store(carried, Ordering::Relaxed);
    }

    /// Carries every waiting cluster setting to the log, keeping the ones
    /// that could not go yet for the next pass
    async fn carry_cluster_settings(&self) {
        let requests = self.control.take_cluster_settings();
        if requests.is_empty() {
            return;
        }
        let mut waiting = Vec::new();
        for (key, value) in requests {
            if let Err(e) = self.carry_cluster_setting(&key, &value).await {
                let message = format!("{key} is not in the replicated log yet, {e}");
                if self.setting_warning.lock().as_deref() != Some(message.as_str()) {
                    tracing::warn!("{message}");
                    *self.setting_warning.lock() = Some(message);
                }
                waiting.push((key, value));
            }
        }
        if waiting.is_empty() {
            *self.setting_warning.lock() = None;
        }
        self.control.requeue_cluster_settings(waiting);
    }

    fn publish_self(&self, to_version: &str, phase: UpgradePhase, message: &str) {
        self.publish_node(Self::running_version(), to_version, phase, message);
    }

    /// Publishes one phase for this node with the version it started from,
    /// which the process after a restart takes from the journal because the
    /// version it runs is the one it moved to
    fn publish_node(
        &self,
        from_version: &str,
        to_version: &str,
        phase: UpgradePhase,
        message: &str,
    ) {
        let now = now_secs();
        let started = self
            .board
            .node_states()
            .into_iter()
            .find(|s| s.node_id == self.node_name)
            .map(|s| s.started_at_secs)
            .unwrap_or(now);
        self.board.set_node_state(NodeUpgradeState {
            node_id: self.node_name.clone(),
            from_version: from_version.to_string(),
            to_version: to_version.to_string(),
            phase,
            started_at_secs: started,
            updated_at_secs: now,
            is_leader: self.is_coordinator(),
            message: message.to_string(),
        });
    }

    // -----------------------------------------------------------------------
    // The loop
    // -----------------------------------------------------------------------

    /// Runs until shutdown or until a restart is armed
    pub async fn run(self: Arc<Self>) {
        let mut was_paused = self.board.settings().paused;
        let mut was_coordinator = false;
        loop {
            if self.shutdown.load(Ordering::Acquire) || self.control.restart_intent().is_some() {
                return;
            }
            // clearing the pause polls at once rather than at the interval,
            // which is hours, so resuming is seen to do something
            let paused = self.board.settings().paused;
            if was_paused && !paused {
                *self.last_poll.lock() = None;
            }
            was_paused = paused;
            // Taking the group means picking up whatever the node that led
            // before left unfinished, and the interval between polls is
            // hours, so a node that has just become the coordinator looks
            // at once rather than waiting out an interval it has already
            // spent as a follower
            let coordinates = self.is_coordinator();
            if coordinates && !was_coordinator {
                *self.last_poll.lock() = None;
            }
            was_coordinator = coordinates;
            self.resume_after_abandoned_drain();
            if let Some(request) = self.control.take_coordinated() {
                self.perform_coordinated(request).await;
                continue;
            }
            for version in self.control.take_stage_requests() {
                self.stage_requested(&version).await;
            }
            self.carry_cluster_settings().await;
            self.refresh_actor_role_carriage().await;
            self.refresh_stream_advance_carriage().await;
            self.refresh_feed_images_carriage().await;
            self.refresh_lake_files_carriage().await;
            self.refresh_schedule_runs_carriage().await;
            self.reconcile_board().await;
            if self
                .board
                .rewrites()
                .iter()
                .any(|r| r.status == RewriteStatus::Acknowledged)
            {
                let applied =
                    objects::apply_acknowledged(&self.catalog, self.board, now_secs()).await;
                tracing::info!(
                    written = applied.written,
                    not_written = applied.not_written,
                    "acknowledged rewrites were applied"
                );
                self.persist_board();
            }
            if let Some(actor) = self.control.take_manual_rollback() {
                self.perform_rollback(&actor).await;
                continue;
            }
            if let Some(request) = self.control.take_manual_upgrade() {
                self.run_pass(Some(request)).await;
            } else if self.is_coordinator() && self.poll_due() {
                self.run_pass(None).await;
            }
            self.control.wait_wake(Duration::from_secs(1)).await;
        }
    }

    /// Fills in the rest of the group's rows on this node's board.
    ///
    /// Every node writes its own row and nothing wrote anyone else's, so a
    /// follower's board held one row and `SHOW UPGRADE STATE` reported a
    /// single node in a group of five. Each member's row is asked for rather
    /// than inferred, because a node's phase is its own journal state and no
    /// other node can speak for it.
    ///
    /// Runs on every member, not only the coordinator: a follower is exactly
    /// where the board was incomplete. A member that does not answer keeps
    /// whatever row is already there, which is what stops a restarting node
    /// from erasing the coordinator's record of why it is down
    async fn reconcile_board(&self) {
        let interval =
            Duration::from_secs(crate::background::upgrade_board_reconcile::DEFAULT_INTERVAL_SECS);
        if let Some(at) = *self.last_board_reconcile.lock()
            && at.elapsed() < interval
        {
            return;
        }
        let Some(raft) = self.raft.as_ref() else {
            // A node in no group is the whole group, and its own row is
            // already on the board
            return;
        };
        let Ok(members) = group_member_names(
            &raft.cluster_config(),
            &self.node_name,
            &self.cluster_members,
        ) else {
            return;
        };
        *self.last_board_reconcile.lock() = Some(Instant::now());

        let reported = self.driver.peer_upgrade_states(&members).await;
        if reported.is_empty() {
            return;
        }
        let held: Vec<String> = self
            .board
            .node_states()
            .into_iter()
            .map(|state| state.node_id)
            .collect();
        let decided =
            crate::background::upgrade_board_reconcile::plan(&reported, &held, now_secs());
        if !decided.changed() {
            return;
        }
        for row in decided.write {
            self.board.set_node_state(row);
        }
        self.persist_board();
    }

    /// Serves again after a drain whose caller never came back.
    ///
    /// A node drains on a peer's word and leaves the drain on the same
    /// peer's word, so a coordinator that stopped between the two would
    /// hold this node out of service for the life of the process. The
    /// deadline the caller named bounds that, and a restart already armed
    /// is left alone because the drain is about to end with the process
    fn resume_after_abandoned_drain(&self) {
        if !self.admission.is_draining() || self.control.restart_intent().is_some() {
            return;
        }
        if !self.control.peer_drain_expired(Instant::now()) {
            return;
        }
        self.control.clear_peer_drain();
        self.admission.end_drain();
        tracing::warn!(
            "the node that asked this one to drain stopped before it was restarted, so it is \
             serving again"
        );
    }

    fn poll_due(&self) -> bool {
        let interval = self.board.settings().release_feed_poll_interval_secs.max(1);
        self.last_poll
            .lock()
            .map(|at| at.elapsed() >= Duration::from_secs(interval))
            .unwrap_or(true)
    }

    // -----------------------------------------------------------------------
    // A pass
    // -----------------------------------------------------------------------

    /// Polls the feed and runs the controller's pass when there is a target
    async fn run_pass(&self, manual: Option<ManualRequest>) {
        *self.last_poll.lock() = Some(Instant::now());
        let now = now_secs();
        let settings = self.board.settings();
        let running = Self::running_version();

        let manifest = match self.controller.poll(self.substrate, &settings, now).await {
            Ok(manifest) => manifest,
            Err(e) => {
                tracing::warn!(error = %e, "the release feed could not be read");
                if let Some(request) = &manual {
                    self.publish_self(
                        &request.version,
                        UpgradePhase::Failed,
                        &format!("the release feed could not be read, {e}"),
                    );
                    self.persist_board();
                }
                return;
            }
        };

        let target = match &manual {
            Some(request) => request.version.clone(),
            None => match self
                .controller
                .target_version(&settings, manifest.as_ref(), running)
            {
                Some(target) => target,
                None => return,
            },
        };
        if target == running {
            if let Some(request) = &manual {
                self.publish_self(
                    &target,
                    UpgradePhase::Completed,
                    &format!("{} is already running, asked by {}", target, request.actor),
                );
                self.persist_board();
            }
            return;
        }
        match &manual {
            Some(_) if settings.paused => {
                self.publish_self(&target, UpgradePhase::Paused, "auto_upgrade_paused is true");
                self.persist_board();
                return;
            }
            Some(_) => {}
            None => {
                if let Err(reason) = settings.may_start(now) {
                    tracing::debug!(target, reason, "a release is newer and the pass is held");
                    return;
                }
            }
        }

        // This node stages first, because the gate reads the target's
        // capabilities out of the staged binary
        let Some(release) = manifest.as_ref().and_then(|m| m.release(&target)).cloned() else {
            let reason = format!(
                "no release {target} is on the {} channel, so there is nothing to stage",
                settings.channel.label()
            );
            self.publish_self(&target, UpgradePhase::Blocked, &reason);
            self.persist_board();
            return;
        };
        self.publish_self(
            &target,
            UpgradePhase::Staging,
            "fetching and verifying the release",
        );
        let staged = match self.driver.stage_here(&release).await {
            Ok(staged) => staged,
            Err(e) => {
                self.blocked(&target, &e.to_string()).await;
                return;
            }
        };
        let running_keys = capabilities::running_config_keys();
        let target_caps = match capabilities::from_staged(
            &staged.path,
            &running_keys,
            CAPABILITIES_TIMEOUT,
        )
        .await
        {
            Ok(caps) => caps,
            Err(e) => {
                self.blocked(&target, &e.to_string()).await;
                return;
            }
        };
        if target_caps.version != target {
            self.blocked(
                &target,
                &format!(
                    "the staged binary reports version {} rather than {target}",
                    target_caps.version
                ),
            )
            .await;
            return;
        }

        let nodes = match self.plan(running).await {
            Ok(nodes) => nodes,
            Err(e) => {
                self.blocked(&target, &e.to_string()).await;
                return;
            }
        };

        let snapshot_path =
            if settings.pre_upgrade_backup_snapshot && crosses_major(running, &target) {
                match self.take_snapshot(&target).await {
                    Ok(path) => Some(path),
                    Err(e) => {
                        self.blocked(&target, &format!("the pre-upgrade snapshot failed, {e}"))
                            .await;
                        return;
                    }
                }
            } else {
                None
            };
        self.driver.set_pending(
            snapshot_path,
            manual.as_ref().map(|r| r.actor.as_str()).unwrap_or(""),
        );

        let user_objects = objects::collect_user_objects(&self.catalog);
        let manual_target = manual.as_ref().map(|r| r.version.as_str());
        let outcome = self
            .controller
            .run_pass(
                PassContext {
                    substrate: self.substrate,
                    board: self.board,
                    driver: self.driver.as_ref(),
                    notifier: &self.notifier,
                    nodes: &nodes,
                    objects: &user_objects,
                    // Federation and Apps are not built, so there are no
                    // peers of other clusters and no Apps to check
                    peers: &[],
                    apps: &[],
                    config_keys: &running_keys,
                    target: &target_caps,
                    running_version: running,
                    now_secs: now,
                    manual_target,
                },
                manifest.as_ref(),
            )
            .await;
        match outcome {
            Ok(PassOutcome::Ran { outcome, .. }) => {
                if let super::rolling::RollingOutcome::PausedAfterRollback { .. } = outcome {
                    self.persist_pause();
                }
                self.persist_board();
            }
            Ok(PassOutcome::Held { reason }) => {
                tracing::info!(target, reason, "the pass was held");
                self.persist_board();
            }
            Ok(PassOutcome::Blocked { blockers, .. }) => {
                tracing::warn!(
                    target,
                    blockers,
                    "the compatibility gate refused the upgrade"
                );
                self.persist_board();
            }
            Ok(PassOutcome::AwaitingAck { report_summary }) => {
                tracing::info!(
                    target,
                    report_summary,
                    "the upgrade waits for ACKNOWLEDGE UPGRADE REWRITES"
                );
                self.persist_board();
            }
            Ok(PassOutcome::UpToDate { .. }) => {}
            Err(e) => {
                self.publish_self(&target, UpgradePhase::Failed, &e.to_string());
                tracing::error!(target, error = %e, "the upgrade pass stopped");
                self.notifier
                    .emit(
                        &UpgradeEvent::Blocked {
                            reason: e.to_string(),
                        },
                        now_secs(),
                    )
                    .await;
                self.persist_board();
            }
        }
    }

    async fn blocked(&self, target: &str, reason: &str) {
        tracing::warn!(target, reason, "the upgrade cannot proceed");
        self.publish_self(target, UpgradePhase::Blocked, reason);
        self.notifier
            .emit(
                &UpgradeEvent::Blocked {
                    reason: reason.to_string(),
                },
                now_secs(),
            )
            .await;
        self.persist_board();
    }

    /// The nodes a sequence walks: every cluster member, this one included,
    /// each proven to run the same version as this node
    async fn plan(&self, running: &str) -> Result<Vec<NodePlan>> {
        let leader = self.raft.as_ref().and_then(|raft| raft.leader_id());
        // Every member is asked at the same time. One after another costs a
        // round trip per member, so a group whose members sit in different
        // regions waited once per node to learn what the whole group runs
        let answers = self.driver.member_versions(&self.cluster_members).await;
        let expected = BinaryVersion::parse(running);
        let mut nodes = Vec::with_capacity(answers.len());
        for member in answers {
            let is_leader = if member.name == self.node_name {
                self.is_coordinator()
            } else {
                let version = member.version.map_err(|reason| {
                    ZyronError::UpgradeRefused(format!(
                        "cluster member {} could not be asked what it runs, {reason}",
                        member.name
                    ))
                })?;
                if expected != Some(version) {
                    return Err(ZyronError::UpgradeRefused(format!(
                        "cluster member {} runs {version} while this node runs {running}. \
                         Every member starts a sequence from the same version",
                        member.name
                    )));
                }
                Some(zyron_raft::node_id_for_name(&member.name)) == leader
            };
            nodes.push(NodePlan {
                node_id: member.name,
                is_leader,
            });
        }
        Ok(nodes)
    }

    /// Takes a physical backup after a checkpoint, for a rollback that a
    /// one-way migration would otherwise rule out
    async fn take_snapshot(&self, target: &str) -> Result<String> {
        if let Some(checkpoint) = self.checkpoint.clone() {
            tokio::task::spawn_blocking(move || checkpoint())
                .await
                .map_err(|e| ZyronError::Internal(format!("checkpoint task failed, {e}")))?;
        }
        let lsn = (self.checkpoint_lsn)();
        let dest = self
            .data_dir
            .join("backups")
            .join(format!("pre-upgrade-{target}-{}", now_secs()));
        let data_dir = self.data_dir.clone();
        let wal_dir = self.wal_dir.clone();
        let dest_for_task = dest.clone();
        tokio::task::spawn_blocking(move || {
            crate::backup::BackupManager::backup(&data_dir, &wal_dir, &dest_for_task, lsn)
        })
        .await
        .map_err(|e| ZyronError::Internal(format!("backup task failed, {e}")))??;
        tracing::info!(path = %dest.display(), "pre-upgrade snapshot written");
        Ok(dest.display().to_string())
    }

    // -----------------------------------------------------------------------
    // What other nodes and operators ask for
    // -----------------------------------------------------------------------

    /// Stages a release another node asked for
    async fn stage_requested(&self, version: &str) {
        let now = now_secs();
        let settings = self.board.settings();
        let release = match self.controller.poll(self.substrate, &settings, now).await {
            Ok(Some(manifest)) => manifest.release(version).cloned(),
            Ok(None) => None,
            Err(e) => {
                self.control.record_stage_failure(
                    version,
                    format!("the release feed could not be read, {e}"),
                );
                return;
            }
        };
        let Some(release) = release else {
            self.control.record_stage_failure(
                version,
                format!(
                    "no release {version} is on the {} channel this node reads",
                    settings.channel.label()
                ),
            );
            return;
        };
        self.publish_self(
            version,
            UpgradePhase::Staging,
            "fetching and verifying the release",
        );
        match self.driver.stage_here(&release).await {
            Ok(staged) => {
                tracing::info!(version, path = %staged.path.display(), "release staged");
                self.publish_self(
                    version,
                    UpgradePhase::Staging,
                    "staged, waiting for the coordinator",
                );
            }
            Err(e) => {
                tracing::warn!(version, error = %e, "the release could not be staged");
                self.control.record_stage_failure(version, e.to_string());
                self.publish_self(version, UpgradePhase::Blocked, &e.to_string());
            }
        }
        self.persist_board();
    }

    /// Restarts as a coordinator asked
    async fn perform_coordinated(&self, request: CoordinatedRestart) {
        let running = Self::running_version();
        match request {
            CoordinatedRestart::IntoStaged { version } => {
                let Some(staged) = self.control.staged().filter(|s| s.version == version) else {
                    tracing::warn!(
                        version,
                        "asked to restart into a release that is not staged"
                    );
                    return;
                };
                self.publish_self(
                    &version,
                    UpgradePhase::Rolling,
                    "draining at the coordinator's request",
                );
                if let Err(e) = self.driver.drain_here().await {
                    tracing::warn!(error = %e, "the coordinated restart was abandoned");
                    self.publish_self(&version, UpgradePhase::Failed, &e.to_string());
                    self.persist_board();
                    return;
                }
                let context = SequenceContext {
                    upgrade_id: self.board.next_upgrade_id(),
                    from_version: running.to_string(),
                    to_version: version.clone(),
                    channel: self.board.settings().channel.label().to_string(),
                    started_at_secs: now_secs(),
                    baseline: HealthBaseline::default(),
                    nodes_total: self.cluster_members.len() as u32,
                    format_migrations: Vec::new(),
                    reversible: true,
                    snapshot_path: None,
                    actor: String::new(),
                };
                self.publish_self(
                    &version,
                    UpgradePhase::Rolling,
                    "restarting on the new binary",
                );
                if let Err(e) = self
                    .driver
                    .arm_upgrade_restart(&staged, &context, true)
                    .await
                {
                    self.admission.end_drain();
                    tracing::error!(error = %e, "the new binary could not be activated");
                    self.publish_self(&version, UpgradePhase::Failed, &e.to_string());
                    self.persist_board();
                }
            }
            CoordinatedRestart::ToPrevious => {
                let to_version = self
                    .journal
                    .read()
                    .last_completed
                    .map(|c| c.from_version)
                    .unwrap_or_else(|| "previous".to_string());
                self.publish_self(
                    &to_version,
                    UpgradePhase::RollingBack,
                    "draining at the coordinator's request",
                );
                if let Err(e) = self.driver.drain_here().await {
                    tracing::warn!(error = %e, "the coordinated rollback was abandoned");
                    self.publish_self(&to_version, UpgradePhase::Failed, &e.to_string());
                    self.persist_board();
                    return;
                }
                if let Err(e) = self.driver.arm_rollback_restart(
                    running,
                    &to_version,
                    "rolled back at the coordinator's request",
                    true,
                    false,
                    "",
                ) {
                    self.admission.end_drain();
                    tracing::error!(error = %e, "the previous binary could not be put back");
                    self.publish_self(&to_version, UpgradePhase::Failed, &e.to_string());
                    self.persist_board();
                }
            }
        }
    }

    /// Rolls the cluster back to the version before the last completed
    /// upgrade, remote nodes first and this node last
    async fn perform_rollback(&self, actor: &str) {
        let Some(last) = self.journal.read().last_completed else {
            tracing::warn!("a rollback was asked for and there is no completed upgrade to undo");
            return;
        };
        let to_version = last.from_version.clone();
        let running = Self::running_version();
        for name in &self.cluster_members {
            if name == &self.node_name {
                continue;
            }
            self.board.set_node_state(NodeUpgradeState {
                node_id: name.clone(),
                from_version: running.to_string(),
                to_version: to_version.clone(),
                phase: UpgradePhase::RollingBack,
                started_at_secs: now_secs(),
                updated_at_secs: now_secs(),
                is_leader: false,
                message: format!("rolling back at the request of {actor}"),
            });
            if let Err(e) = self.driver.rollback(name, &to_version).await {
                tracing::error!(node = name, error = %e, "the rollback stopped at this node");
                self.board.set_node_state(NodeUpgradeState {
                    node_id: name.clone(),
                    from_version: running.to_string(),
                    to_version: to_version.clone(),
                    phase: UpgradePhase::Failed,
                    started_at_secs: now_secs(),
                    updated_at_secs: now_secs(),
                    is_leader: false,
                    message: e.to_string(),
                });
                self.persist_board();
                return;
            }
            let deadline = Instant::now()
                + Duration::from_secs(self.config.upgrade.health_recovery_timeout_secs);
            loop {
                match self.driver.peer_version(name).await {
                    Ok(version) if version == to_version => break,
                    _ if Instant::now() >= deadline => {
                        tracing::error!(node = name, "the node did not come back on {to_version}");
                        self.persist_board();
                        return;
                    }
                    _ => tokio::time::sleep(Duration::from_secs(2)).await,
                }
            }
            self.board.set_node_state(NodeUpgradeState {
                node_id: name.clone(),
                from_version: running.to_string(),
                to_version: to_version.clone(),
                phase: UpgradePhase::Completed,
                started_at_secs: now_secs(),
                updated_at_secs: now_secs(),
                is_leader: false,
                message: format!("back on {to_version}"),
            });
        }
        self.publish_self(&to_version, UpgradePhase::RollingBack, "draining");
        if let Err(e) = self.driver.drain_here().await {
            self.publish_self(&to_version, UpgradePhase::Failed, &e.to_string());
            self.persist_board();
            return;
        }
        self.publish_self(
            &to_version,
            UpgradePhase::RollingBack,
            "restarting on the previous binary",
        );
        // the cluster comes back paused, otherwise the next poll would find
        // the same release on the feed and apply it again. The pause goes to
        // the log now, while this node still leads and before its restart
        // ends the loop that would otherwise carry it
        self.persist_pause();
        self.carry_cluster_settings().await;
        if let Err(e) = self.driver.arm_rollback_restart(
            running,
            &to_version,
            &format!(
                "manual rollback by {actor}, automatic upgrades paused until \
                 auto_upgrade_paused is cleared"
            ),
            false,
            true,
            actor,
        ) {
            self.admission.end_drain();
            self.publish_self(&to_version, UpgradePhase::Failed, &e.to_string());
        }
        self.persist_board();
    }

    // -----------------------------------------------------------------------
    // After a restart
    // -----------------------------------------------------------------------

    /// Finishes the restart the previous process journaled, if any.
    ///
    /// Run once the node is serving. A node that upgraded itself watches
    /// its own health against the baseline in the journal and rolls back
    /// when it does not recover. A node another coordinator restarted
    /// records the outcome and moves on. Either way the migrations the new
    /// binary needs run once the node is judged healthy
    pub async fn finish_restart(&self) {
        let Some(record) = self.journal.read().restart else {
            return;
        };
        let running = Self::running_version();
        match record.kind {
            RestartKind::Rollback => self.finish_rollback(&record, running).await,
            RestartKind::Upgrade if running == record.to_version => {
                self.finish_upgrade(&record).await
            }
            RestartKind::Upgrade => {
                let detail = format!(
                    "the restart into {} came up as {running}, the binary in place is not the \
                     one activated",
                    record.to_version
                );
                tracing::error!("{detail}");
                self.board.push_history(UpgradeHistoryEntry {
                    upgrade_id: record.upgrade_id,
                    from_version: record.from_version.clone(),
                    to_version: record.to_version.clone(),
                    channel: record.channel.clone(),
                    started_at_secs: record.started_at_secs,
                    finished_at_secs: now_secs(),
                    outcome: UpgradeOutcome::Failed,
                    nodes_upgraded: record.nodes_upgraded_before,
                    format_migrations_run: 0,
                    catalog_migrations_run: 0,
                    rewrites_applied: 0,
                    reversible: true,
                    detail: detail.clone(),
                });
                self.publish_node(
                    &record.from_version,
                    &record.to_version,
                    UpgradePhase::Failed,
                    &detail,
                );
                self.clear_restart();
            }
        }
    }

    async fn finish_rollback(&self, record: &PendingRestart, running: &str) {
        let reason = record
            .rolled_back_reason
            .clone()
            .unwrap_or_else(|| "rolled back".to_string());
        let detail = if running == record.to_version {
            format!("back on {running}, {reason}")
        } else {
            format!(
                "the rollback to {} came up as {running}, {reason}",
                record.to_version
            )
        };
        tracing::warn!("{detail}");
        self.board.push_history(UpgradeHistoryEntry {
            upgrade_id: record.upgrade_id,
            from_version: record.from_version.clone(),
            to_version: record.to_version.clone(),
            channel: record.channel.clone(),
            started_at_secs: record.started_at_secs,
            finished_at_secs: now_secs(),
            outcome: UpgradeOutcome::RolledBack,
            nodes_upgraded: 0,
            format_migrations_run: 0,
            catalog_migrations_run: 0,
            rewrites_applied: 0,
            reversible: true,
            detail: detail.clone(),
        });
        let phase = if record.pause_on_return {
            UpgradePhase::Paused
        } else {
            UpgradePhase::Completed
        };
        self.board.set_node_state(NodeUpgradeState {
            node_id: self.node_name.clone(),
            from_version: record.from_version.clone(),
            to_version: record.to_version.clone(),
            phase,
            started_at_secs: record.started_at_secs,
            updated_at_secs: now_secs(),
            is_leader: self.is_coordinator(),
            message: detail.clone(),
        });
        if !record.coordinated {
            self.notifier
                .emit(
                    &UpgradeEvent::RolledBack {
                        node_id: self.node_name.clone(),
                        reason: reason.clone(),
                    },
                    now_secs(),
                )
                .await;
            self.notifier
                .emit(
                    &UpgradeEvent::Completed {
                        to_version: record.to_version.clone(),
                        outcome: UpgradeOutcome::RolledBack,
                        detail: detail.clone(),
                    },
                    now_secs(),
                )
                .await;
        }
        if record.pause_on_return {
            self.persist_pause();
        }
        if let Err(e) = self.journal.update(|journal| {
            journal.restart = None;
            journal.last_completed = None;
        }) {
            tracing::error!(error = %e, "the upgrade journal could not be written");
        }
        self.persist_board();
    }

    async fn finish_upgrade(&self, record: &PendingRestart) {
        let settings = self.board.settings();
        if record.coordinated {
            tracing::info!(
                to_version = %record.to_version,
                "restarted on the new binary at the coordinator's request"
            );
        } else {
            self.publish_node(
                &record.from_version,
                &record.to_version,
                UpgradePhase::Rolling,
                "back on the new binary, watching health against the baseline",
            );
            let rolling = RollingSettings {
                health_recovery_timeout_secs: settings.health_recovery_timeout_secs,
                health_poll_interval_secs: self.config.upgrade.health_poll_interval_secs,
                threshold: HealthThreshold {
                    latency_multiplier: self.config.upgrade.health_latency_multiplier,
                    throughput_floor: self.config.upgrade.health_throughput_floor,
                    error_rate_ceiling: self.config.upgrade.health_error_rate_ceiling,
                },
                rollback_on_health_fail: settings.rollback_on_health_fail,
            };
            let verdict = self.watch_own_health(&record.baseline, rolling).await;
            if !verdict.is_healthy() {
                let reason = verdict.reason();
                tracing::error!(reason, "this node did not recover on the new binary");
                if rolling.rollback_on_health_fail {
                    self.publish_node(
                        &record.from_version,
                        &record.to_version,
                        UpgradePhase::RollingBack,
                        &format!("unhealthy, {reason}"),
                    );
                    if let Err(e) = self.driver.drain_here().await {
                        tracing::warn!(error = %e, "draining before the rollback did not finish");
                    }
                    match self.driver.arm_rollback_restart(
                        &record.to_version,
                        &record.from_version,
                        &format!("unhealthy on {}, {reason}", record.to_version),
                        false,
                        true,
                        "",
                    ) {
                        Ok(()) => {
                            self.persist_board();
                            return;
                        }
                        Err(e) => {
                            self.admission.end_drain();
                            tracing::error!(error = %e, "the previous binary could not be put back");
                        }
                    }
                }
                let detail = format!(
                    "unhealthy on {}, {reason}, left in place and paused",
                    record.to_version
                );
                self.board.push_history(UpgradeHistoryEntry {
                    upgrade_id: record.upgrade_id,
                    from_version: record.from_version.clone(),
                    to_version: record.to_version.clone(),
                    channel: record.channel.clone(),
                    started_at_secs: record.started_at_secs,
                    finished_at_secs: now_secs(),
                    outcome: UpgradeOutcome::Paused,
                    nodes_upgraded: record.nodes_upgraded_before + 1,
                    format_migrations_run: 0,
                    catalog_migrations_run: 0,
                    rewrites_applied: 0,
                    reversible: true,
                    detail: detail.clone(),
                });
                self.publish_node(
                    &record.from_version,
                    &record.to_version,
                    UpgradePhase::Paused,
                    &detail,
                );
                self.notifier
                    .emit(
                        &UpgradeEvent::Paused {
                            reason: detail.clone(),
                        },
                        now_secs(),
                    )
                    .await;
                self.persist_pause();
                self.clear_restart();
                return;
            }
        }

        self.publish_node(
            &record.from_version,
            &record.to_version,
            UpgradePhase::Migrating,
            "moving formats, catalog rows, and user objects forward",
        );
        self.persist_board();
        let summary = self.run_post_upgrade_migrations(&settings).await;
        let detail = if summary.detail.is_empty() {
            format!(
                "{} node(s) upgraded, {} format(s), {} catalog table(s), {} rewrite(s) written",
                record.nodes_upgraded_before + 1,
                summary.formats_run,
                summary.catalog_run,
                summary.rewrites_written
            )
        } else {
            format!(
                "{} node(s) upgraded, {} format(s), {} catalog table(s), {} rewrite(s) written. {}",
                record.nodes_upgraded_before + 1,
                summary.formats_run,
                summary.catalog_run,
                summary.rewrites_written,
                summary.detail.join(". ")
            )
        };
        let accepted_broken = self
            .board
            .rewrites()
            .iter()
            .any(|r| r.status == RewriteStatus::AcceptedBroken);
        self.board.push_history(UpgradeHistoryEntry {
            upgrade_id: record.upgrade_id,
            from_version: record.from_version.clone(),
            to_version: record.to_version.clone(),
            channel: record.channel.clone(),
            started_at_secs: record.started_at_secs,
            finished_at_secs: now_secs(),
            outcome: UpgradeOutcome::Completed,
            nodes_upgraded: record.nodes_upgraded_before + 1,
            format_migrations_run: summary.formats_run,
            catalog_migrations_run: summary.catalog_run,
            rewrites_applied: summary.rewrites_written,
            // what this node actually moved decides reversibility, the gate's
            // forecast counted every format whose window could have moved
            reversible: summary.reversible && !accepted_broken,
            detail: detail.clone(),
        });
        self.publish_node(
            &record.from_version,
            &record.to_version,
            UpgradePhase::Completed,
            "healthy on the new binary",
        );
        if !record.coordinated {
            self.notifier
                .emit(
                    &UpgradeEvent::NodeCompleted {
                        node_id: self.node_name.clone(),
                        to_version: record.to_version.clone(),
                        nodes_remaining: 0,
                    },
                    now_secs(),
                )
                .await;
            self.notifier
                .emit(
                    &UpgradeEvent::Completed {
                        to_version: record.to_version.clone(),
                        outcome: UpgradeOutcome::Completed,
                        detail: detail.clone(),
                    },
                    now_secs(),
                )
                .await;
        }
        let completed = CompletedUpgrade {
            upgrade_id: record.upgrade_id,
            from_version: record.from_version.clone(),
            to_version: record.to_version.clone(),
            finished_at_secs: now_secs(),
            migrated_formats: summary.migrated_formats.clone(),
            migrated_tables: summary.migrated_tables.clone(),
            snapshot_path: record.snapshot_path.clone(),
        };
        let board_snapshot = self.board.snapshot();
        if let Err(e) = self.journal.update(|journal| {
            journal.board = board_snapshot;
            journal.restart = None;
            journal.last_completed = Some(completed);
        }) {
            tracing::error!(error = %e, "the upgrade journal could not be written");
        }
        tracing::info!(to_version = %record.to_version, "{detail}");
    }

    fn clear_restart(&self) {
        let board_snapshot = self.board.snapshot();
        if let Err(e) = self.journal.update(|journal| {
            journal.board = board_snapshot;
            journal.restart = None;
        }) {
            tracing::error!(error = %e, "the upgrade journal could not be written");
        }
    }

    /// Watches this node against the baseline until it is healthy or the
    /// recovery window closes
    async fn watch_own_health(
        &self,
        baseline: &HealthBaseline,
        settings: RollingSettings,
    ) -> zyron_common::format::HealthVerdict {
        let deadline = Instant::now() + Duration::from_secs(settings.health_recovery_timeout_secs);
        loop {
            let observed = self.driver.observe_here();
            let verdict = baseline.judge(&observed, settings.threshold);
            if verdict.is_healthy() || Instant::now() >= deadline {
                return verdict;
            }
            if self.shutdown.load(Ordering::Acquire) {
                return verdict;
            }
            tokio::time::sleep(Duration::from_secs(
                settings.health_poll_interval_secs.max(1),
            ))
            .await;
        }
    }

    /// Moves this node's persisted state forward: eager format sweeps,
    /// catalog rows, and the rewrites the policy applies on its own
    async fn run_post_upgrade_migrations(
        &self,
        settings: &zyron_common::format::UpgradeSettings,
    ) -> MigrationSummary {
        let mut summary = MigrationSummary {
            reversible: true,
            ..MigrationSummary::default()
        };
        let budget = MigrationBudget {
            time_secs: settings.format_migration_budget_time_secs,
            disk_multiple: settings.format_migration_budget_disk_multiple,
            memory_fraction: settings.format_migration_budget_memory_fraction,
            disk_free_bytes: free_disk_bytes(&self.data_dir),
            node_memory_bytes: node_memory_bytes(),
        };

        // Format sweeps, one per eager format, on a blocking thread because
        // each rewrites files whole
        let eager: Vec<zyron_common::format::FormatKind> = self
            .substrate
            .formats
            .entries()
            .filter(|entry| entry.registration.migration_policy == MigrationPolicy::Eager)
            .map(|entry| entry.registration.kind)
            .collect();
        let substrate = self.substrate;
        let data_dir = self.data_dir.clone();
        let sweeps = tokio::task::spawn_blocking(move || {
            eager
                .into_iter()
                .map(|kind| {
                    migrations::sweep_format(
                        &substrate.formats,
                        &substrate.migrations,
                        kind,
                        &data_dir,
                        budget,
                        now_secs(),
                    )
                })
                .collect::<Vec<_>>()
        })
        .await;
        match sweeps {
            Ok(results) => {
                for result in results {
                    match result {
                        Ok(sweep) if sweep.files_migrated > 0 => {
                            summary.formats_run += 1;
                            let from = self
                                .substrate
                                .formats
                                .get(sweep.kind)
                                .map(|entry| entry.registration.reader_supported_versions.oldest)
                                .unwrap_or(zyron_common::format::FormatVersion::V1);
                            if let Some(entry) = self.substrate.formats.get(sweep.kind) {
                                if !entry.reversible_from(from) {
                                    summary.reversible = false;
                                }
                            }
                            summary
                                .migrated_formats
                                .push((sweep.kind.catalog_name().to_string(), from.as_u32()));
                            if sweep.budget_exhausted {
                                summary.detail.push(format!(
                                    "the {} sweep stopped at its budget with {} file(s) migrated",
                                    sweep.kind, sweep.files_migrated
                                ));
                            }
                            if sweep.failures > 0 {
                                summary.detail.push(format!(
                                    "{} {} file(s) could not be migrated",
                                    sweep.failures, sweep.kind
                                ));
                            }
                        }
                        Ok(_) => {}
                        Err(e) => summary.detail.push(format!("a format sweep failed, {e}")),
                    }
                }
            }
            Err(e) => summary
                .detail
                .push(format!("the format sweeps did not run, {e}")),
        }

        // Catalog rows, table by table on a blocking thread
        let store =
            HeapCatalogTableStore::new(Arc::clone(&self.catalog), Arc::clone(&self.journal));
        let catalog_outcome = tokio::task::spawn_blocking(move || {
            migrations::migrate_catalog(&substrate.catalog_schemas, &store)
        })
        .await;
        match catalog_outcome {
            Ok((migrated, failures)) => {
                for table in migrated {
                    summary.catalog_run += 1;
                    if !table.reversible {
                        summary.reversible = false;
                    }
                    summary
                        .migrated_tables
                        .push((table.catalog_table, table.from_version.as_u32()));
                }
                for failure in failures {
                    summary.detail.push(failure);
                }
            }
            Err(e) => summary
                .detail
                .push(format!("the catalog migration did not run, {e}")),
        }

        // User objects
        let user_objects = objects::collect_user_objects(&self.catalog);
        let pass = migrations::rewrite_objects(
            self.board,
            &user_objects,
            settings.user_object_rewrite_policy,
            now_secs(),
        );
        let written = objects::write_rewritten(
            &self.catalog,
            self.board,
            &pass.rewritten,
            &user_objects,
            now_secs(),
        )
        .await;
        summary.rewrites_written = written.written;
        if written.not_written > 0 {
            summary.detail.push(format!(
                "{} rewrite(s) were computed but not written, the queue holds the diffs",
                written.not_written
            ));
        }
        if pass.queued > 0 {
            summary.detail.push(format!(
                "{} rewrite(s) wait for ACKNOWLEDGE UPGRADE REWRITES",
                pass.queued
            ));
        }
        summary
    }
}

impl UpgradeControl for UpgradeService {
    fn request_upgrade(&self, version: &str, actor: &str, now_secs: u64) -> Result<()> {
        if !self.is_coordinator() {
            return Err(ZyronError::UpgradeRefused(format!(
                "upgrades are driven by {}, run this there",
                self.coordinator_name()
            )));
        }
        self.control
            .request_manual_upgrade(ManualRequest {
                version: version.to_string(),
                actor: actor.to_string(),
            })
            .map_err(ZyronError::UpgradeRefused)?;
        self.board.set_node_state(NodeUpgradeState {
            node_id: self.node_name.clone(),
            from_version: Self::running_version().to_string(),
            to_version: version.to_string(),
            phase: UpgradePhase::Detected,
            started_at_secs: now_secs,
            updated_at_secs: now_secs,
            is_leader: true,
            message: format!("manual upgrade to {version} requested by {actor}"),
        });
        self.persist_board();
        Ok(())
    }

    fn request_rollback(&self, actor: &str, _now_secs: u64) -> Result<()> {
        if !self.is_coordinator() {
            return Err(ZyronError::UpgradeRefused(format!(
                "rollbacks are driven by {}, run this there",
                self.coordinator_name()
            )));
        }
        let journal = self.journal.read();
        let Some(last) = journal.last_completed.as_ref() else {
            return Err(ZyronError::UpgradeRefused(
                "there is no completed upgrade to roll back. `LIST UPGRADE HISTORY` reports \
                 what this node has been through"
                    .to_string(),
            ));
        };
        let migrated_formats: Vec<_> = last
            .migrated_formats
            .iter()
            .filter_map(|(name, from)| {
                zyron_common::format::FormatKind::from_catalog_name(name)
                    .map(|kind| (kind, zyron_common::format::FormatVersion::from_u32(*from)))
            })
            .collect();
        let migrated_tables: Vec<_> = last
            .migrated_tables
            .iter()
            .map(|(name, from)| {
                (
                    name.clone(),
                    zyron_common::format::FormatVersion::from_u32(*from),
                )
            })
            .collect();
        let snapshot_available = last
            .snapshot_path
            .as_ref()
            .map(|path| Path::new(path).is_dir())
            .unwrap_or(false);
        let report = downgrade::evaluate(
            &self.substrate.formats,
            &self.substrate.catalog_schemas,
            self.board,
            &last.to_version,
            &last.from_version,
            &migrated_formats,
            &migrated_tables,
            snapshot_available,
        );
        if !report.eligible() {
            return Err(ZyronError::UpgradeRefused(report.refusal()));
        }
        let previous =
            Path::new(&self.driver.live_path().display().to_string()).with_extension("previous");
        if !previous.exists() {
            return Err(ZyronError::UpgradeRefused(format!(
                "there is no previous binary beside {} to roll back to",
                self.driver.live_path().display()
            )));
        }
        drop(journal);
        self.control
            .request_manual_rollback(actor)
            .map_err(ZyronError::UpgradeRefused)?;
        Ok(())
    }

    fn acknowledge_rewrites(
        &self,
        category: RewriteCategory,
        actor: &str,
        now_secs: u64,
    ) -> Result<usize> {
        let moved = match category {
            // Everything waiting on a person that is not a break: ambiguous
            // rewrites, and safe ones under a policy that asks for every
            // class
            RewriteCategory::Ambiguous | RewriteCategory::Safe => {
                self.board
                    .acknowledge(RewriteCategory::Safe, actor, now_secs)
                    + self
                        .board
                        .acknowledge(RewriteCategory::Ambiguous, actor, now_secs)
            }
            RewriteCategory::Unsafe => {
                self.board
                    .accept_broken(RewriteCategory::Unsafe, actor, now_secs)
            }
        };
        self.persist_board();
        // The service applies what was acknowledged on its next pass, which
        // it is woken for
        self.control.wake();
        Ok(moved)
    }
}

/// The members of the consensus group by name, from the live membership
/// rather than the config's peer list, so a member the config does not
/// name still counts. This node names itself, every other member is
/// matched to a configured peer by the id its name hashes to, and a
/// member no configured peer names is refused by id and address, because
/// what it runs cannot be asked and the floor is unknown without it
fn group_member_names(
    config: &ClusterConfig,
    self_name: &str,
    configured: &[String],
) -> Result<Vec<String>> {
    let self_id = zyron_raft::node_id_for_name(self_name);
    let mut names = Vec::with_capacity(config.nodes.len());
    for node in &config.nodes {
        if node.node_id == self_id {
            names.push(self_name.to_string());
            continue;
        }
        match configured
            .iter()
            .find(|name| zyron_raft::node_id_for_name(name) == node.node_id)
        {
            Some(name) => names.push(name.clone()),
            None => {
                return Err(ZyronError::UpgradeRefused(format!(
                    "node {} at {} is a member of the consensus group and none of this node's \
                     [cluster] peers has that name, so what it runs cannot be asked",
                    node.node_id, node.address
                )));
            }
        }
    }
    Ok(names)
}

/// Whether the target is a new major version
fn crosses_major(running: &str, target: &str) -> bool {
    match (BinaryVersion::parse(running), BinaryVersion::parse(target)) {
        (Some(running), Some(target)) => target.major > running.major,
        _ => false,
    }
}

/// Bytes free on the volume holding the data directory, zero when it
/// cannot be measured
fn free_disk_bytes(data_dir: &Path) -> u64 {
    let canonical = data_dir
        .canonicalize()
        .unwrap_or_else(|_| data_dir.to_path_buf());
    // Windows answers a canonical path in its verbatim form, which no mount
    // point is written in, so the prefix comes off before the comparison
    let plain = canonical.to_string_lossy();
    let target = PathBuf::from(plain.strip_prefix(r"\\?\").unwrap_or(&plain));
    let disks = sysinfo::Disks::new_with_refreshed_list();
    disks
        .iter()
        .filter(|disk| target.starts_with(disk.mount_point()))
        .max_by_key(|disk| disk.mount_point().as_os_str().len())
        .map(|disk| disk.available_space())
        .unwrap_or(0)
}

/// Bytes of memory on the node, zero when it cannot be measured
fn node_memory_bytes() -> u64 {
    let mut system = sysinfo::System::new();
    system.refresh_memory();
    system.total_memory()
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_a_major_bump_is_what_takes_a_snapshot() {
        assert!(crosses_major("0.12.0", "1.0.0"));
        assert!(!crosses_major("0.12.0", "0.13.0"));
        assert!(!crosses_major("1.2.0", "1.3.0"));
        assert!(!crosses_major("nonsense", "1.0.0"));
    }

    #[test]
    fn test_the_node_measures_itself() {
        // Both probes answer for the machine the tests run on, and a
        // measurement of zero would silently turn the budgets off
        assert!(node_memory_bytes() > 0);
        assert!(free_disk_bytes(&std::env::temp_dir()) > 0);
    }

    /// The gate asks the group's live members. A configured peer that is
    /// not in the group is not asked, this node names itself whether or
    /// not the config lists it, and a member no configured peer names
    /// holds the gate by id and address
    #[test]
    fn test_group_members_come_from_the_live_membership() {
        use zyron_raft::{NodeConfig, node_id_for_name};
        let configured: Vec<String> = ["node-1", "node-2", "node-9"]
            .iter()
            .map(|n| n.to_string())
            .collect();
        let config = ClusterConfig::of_voters([
            (node_id_for_name("node-1"), "a:1".to_string()),
            (node_id_for_name("node-2"), "b:1".to_string()),
        ]);

        let mut names = group_member_names(&config, "node-1", &configured).expect("named");
        names.sort();
        assert_eq!(names, vec!["node-1".to_string(), "node-2".to_string()]);

        let mut names =
            group_member_names(&config, "node-2", &["node-1".to_string()]).expect("named");
        names.sort();
        assert_eq!(names, vec!["node-1".to_string(), "node-2".to_string()]);

        let with_stranger = config.with_node(NodeConfig::voter(7, "c:1".to_string()));
        let err = group_member_names(&with_stranger, "node-1", &configured)
            .expect_err("a member with no name holds the gate")
            .to_string();
        assert!(err.contains("node 7 at c:1"), "{err}");
        assert!(err.contains("[cluster] peers"), "{err}");
    }
}

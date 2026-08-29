//! The running node: the protocol from [`crate::consensus`], driven.
//!
//! ## Four loops and nothing else
//!
//! - The **ticker** owns time. It is the only thing that calls
//!   [`RaftConsensus::tick`], so elections and step-downs happen in one place.
//! - The **replicator** owns outbound traffic. It asks consensus what each
//!   peer is owed and dispatches it without waiting, which is what makes
//!   replication a pipeline. Replies come back on their own tasks.
//! - The **applier** owns the state machine. It is the only thing that calls
//!   `apply`, so entries reach the machine exactly once and in order, and it
//!   is the only place a snapshot can be captured at an exact index.
//! - The **durability watcher** turns the log writer's fsync into a commit.
//!   On a leader that is the node's own vote in its own quorum.
//!
//! Everything else is a request handler or a caller waiting on a watermark.
//!
//! ## The lock is held for microseconds, never across an await
//!
//! Consensus is one mutex. Every use of it is take, mutate, drop, and the
//! result is then acted on outside. A lock held across a network call would
//! turn one slow peer into a stalled group, and a lock held across an fsync
//! would put the disk in front of every heartbeat. The two things that do wait
//! on the disk, an AppendEntries reply and a proposal, wait on a published
//! watermark rather than on the lock.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use tokio::sync::watch;
use tokio::task::JoinHandle;
use zyron_common::error::{Result, ZyronError};

use crate::NodeId;
use crate::config::RaftConfig;
use crate::consensus::{
    ConsensusMetrics, PeerWork, RaftConsensus, ReadIndexOutcome, SnapshotDecision,
};
use crate::election::{RequestVoteReply, RequestVoteRequest};
use crate::log::{LogPager, LogWriterHandle, RaftCommand, RaftLogEntry, SliceItem};
use crate::machine::StateMachine;
use crate::membership::ClusterConfig;
use crate::replication::{
    AppendEntriesReply, AppendEntriesRequest, ReadIndexReply, ReadIndexRequest,
};
use crate::snapshot::{InstallSnapshotReply, InstallSnapshotRequest, SnapshotMeta, SnapshotStore};
use crate::state::{RaftRole, RaftState};
use crate::transport::{
    RaftHandlerFuture, RaftRequestHandler, RaftRpcError, RaftTransport, TransportStats,
};

/// How close a learner has to be before it is promoted to a voter.
///
/// Promotion makes it count in every quorum immediately, so a learner still
/// far behind would stall commits. One batch of slack is enough that the
/// promotion is not chasing a moving target on a busy leader
const PROMOTION_TOLERANCE: u64 = 1024;

/// What a node needs to start.
pub struct RaftNodeOptions {
    pub id: NodeId,
    /// Where the log, the term, the vote, and the snapshots live
    pub dir: PathBuf,
    pub config: RaftConfig,
    /// The group this node starts as part of. Ignored in favour of the
    /// configuration in a snapshot, and then in favour of whatever the log
    /// says, so a restart never rewinds membership
    pub bootstrap: ClusterConfig,
    pub machine: Arc<dyn StateMachine>,
    pub transport: Arc<dyn RaftTransport>,
}

/// Everything an operator reads about one node.
#[derive(Debug, Clone)]
pub struct RaftMetrics {
    pub id: NodeId,
    pub role: RaftRole,
    pub term: u64,
    pub leader_id: Option<NodeId>,
    pub commit_index: u64,
    pub last_applied: u64,
    pub last_log_index: u64,
    pub first_log_index: u64,
    pub persisted_index: u64,
    pub snapshot_index: u64,
    pub voters: usize,
    pub learners: usize,
    pub consensus: ConsensusMetrics,
    pub transport: TransportStats,
    pub snapshots_created: u64,
    pub snapshot_bytes_sent: u64,
    pub snapshot_bytes_received: u64,
    pub proposals: u64,
}

struct RaftInner {
    id: NodeId,
    config: RaftConfig,
    core: parking_lot::Mutex<RaftConsensus>,
    machine: Arc<dyn StateMachine>,
    transport: Arc<dyn RaftTransport>,
    snapshots: parking_lot::Mutex<SnapshotStore>,
    log_writer: LogWriterHandle,
    dir: PathBuf,

    /// Bumped when there is outbound work, so the replicator does not have to
    /// poll at proposal rate.
    ///
    /// A generation counter rather than a notify, because a notify delivered
    /// while the replicator is between iterations is lost, and the work would
    /// then wait for the fallback timer. A watch receiver that has not seen
    /// the current value returns from `changed` at once
    work_tx: watch::Sender<u64>,
    commit_tx: watch::Sender<u64>,
    applied_tx: watch::Sender<u64>,
    round_tx: watch::Sender<u64>,
    role_tx: watch::Sender<(RaftRole, u64, Option<NodeId>)>,

    shutdown: Arc<tokio::sync::Notify>,
    running: AtomicBool,
    tasks: parking_lot::Mutex<Vec<JoinHandle<()>>>,

    snapshotting: AtomicBool,
    snapshots_created: AtomicU64,
    snapshot_bytes_sent: AtomicU64,
    snapshot_bytes_received: AtomicU64,
    proposals: AtomicU64,
    directory_version: AtomicU64,
}

/// A running consensus node.
#[derive(Clone)]
pub struct RaftNode {
    inner: Arc<RaftInner>,
}

impl RaftNode {
    /// Opens the durable state, replays what survives, and starts the loops.
    ///
    /// A node with a snapshot restores the state machine from it before
    /// anything else, because the log it holds starts after that point and
    /// replaying it into an empty machine would produce a hole
    pub async fn start(options: RaftNodeOptions) -> Result<Self> {
        let RaftNodeOptions {
            id,
            dir,
            config,
            bootstrap,
            machine,
            transport,
        } = options;
        config.validate()?;
        std::fs::create_dir_all(&dir)
            .map_err(|e| ZyronError::IoError(format!("create raft directory: {e}")))?;

        let snapshots = SnapshotStore::open(&dir)?;
        let base = match snapshots.current() {
            Some(snapshot) => snapshot.meta.config.clone(),
            None => bootstrap,
        };
        base.validate()?;

        let state = RaftState::open(&dir, config.resident_log_bytes)?;
        let now = Instant::now();
        let mut core = RaftConsensus::new(id, config.clone(), state, base, now)?;

        if let Some(snapshot) = snapshots.current() {
            let index = snapshot.meta.last_included_index;
            if machine.applied_index() < index {
                let data = snapshot.data.clone();
                let m = Arc::clone(&machine);
                tokio::task::spawn_blocking(move || m.restore(&data, index))
                    .await
                    .map_err(|e| {
                        ZyronError::Internal(format!("snapshot restore task failed: {e}"))
                    })??;
            }
            if index > core.commit_index() {
                core.state.volatile.commit_index = index;
            }
            core.set_last_applied(index);
        }
        // A machine whose state outlives the process may already be ahead of
        // the snapshot, and re-applying entries it has seen would double them
        let machine_applied = machine.applied_index();
        if machine_applied > core.last_applied() {
            core.set_last_applied(machine_applied);
        }

        for node in &core.cluster_config().nodes {
            if node.node_id != id {
                transport.set_address(node.node_id, &node.address);
            }
        }

        let log_writer = core.state.log().writer_handle();
        let commit = core.commit_index();
        let applied = core.last_applied();
        let role = (core.role(), core.term(), core.leader_id());

        let (commit_tx, _) = watch::channel(commit);
        let (applied_tx, _) = watch::channel(applied);
        let (round_tx, _) = watch::channel(0u64);
        let (work_tx, _) = watch::channel(0u64);
        let (role_tx, _) = watch::channel(role);

        let inner = Arc::new(RaftInner {
            id,
            config,
            core: parking_lot::Mutex::new(core),
            machine,
            transport,
            snapshots: parking_lot::Mutex::new(snapshots),
            log_writer,
            dir,
            work_tx,
            commit_tx,
            applied_tx,
            round_tx,
            role_tx,
            shutdown: Arc::new(tokio::sync::Notify::new()),
            running: AtomicBool::new(true),
            tasks: parking_lot::Mutex::new(Vec::new()),
            snapshotting: AtomicBool::new(false),
            snapshots_created: AtomicU64::new(0),
            snapshot_bytes_sent: AtomicU64::new(0),
            snapshot_bytes_received: AtomicU64::new(0),
            proposals: AtomicU64::new(0),
            directory_version: AtomicU64::new(u64::MAX),
        });

        let node = Self { inner };
        node.spawn_loops();
        Ok(node)
    }

    fn spawn_loops(&self) {
        let mut tasks = self.inner.tasks.lock();
        for task in [
            tokio::spawn(ticker(Arc::clone(&self.inner))),
            tokio::spawn(replicator(Arc::clone(&self.inner))),
            tokio::spawn(applier(Arc::clone(&self.inner))),
            tokio::spawn(durability_watcher(Arc::clone(&self.inner))),
        ] {
            tasks.push(task);
        }
    }

    /// Stops the loops and closes the transport.
    ///
    /// The log writer thread is joined when the consensus state drops, so a
    /// node that has shut down has everything it acknowledged on disk
    pub async fn shutdown(&self) {
        if !self.inner.running.swap(false, Ordering::AcqRel) {
            return;
        }
        self.inner.shutdown.notify_waiters();
        let tasks = std::mem::take(&mut *self.inner.tasks.lock());
        for task in tasks {
            task.abort();
            let _ = task.await;
        }
    }

    /// A handle the transport server answers requests through
    pub fn handler(&self) -> Arc<dyn RaftRequestHandler> {
        Arc::new(self.clone())
    }

    // -----------------------------------------------------------------------
    // What this node is
    // -----------------------------------------------------------------------

    pub fn id(&self) -> NodeId {
        self.inner.id
    }

    pub fn role(&self) -> RaftRole {
        self.inner.core.lock().role()
    }

    pub fn term(&self) -> u64 {
        self.inner.core.lock().term()
    }

    pub fn leader_id(&self) -> Option<NodeId> {
        self.inner.core.lock().leader_id()
    }

    pub fn is_leader(&self) -> bool {
        self.inner.core.lock().is_leader()
    }

    pub fn commit_index(&self) -> u64 {
        self.inner.core.lock().commit_index()
    }

    pub fn last_applied(&self) -> u64 {
        self.inner.core.lock().last_applied()
    }

    pub fn last_log_index(&self) -> u64 {
        self.inner.core.lock().last_log_index()
    }

    pub fn cluster_config(&self) -> ClusterConfig {
        self.inner.core.lock().cluster_config()
    }

    /// The term of one log entry, or None when it is past the end or behind a
    /// snapshot.
    ///
    /// Read by an operator comparing two replicas: two nodes agreeing on the
    /// term at every index agree on the log, because Raft only ever writes one
    /// entry per index per term
    pub fn term_at(&self, index: u64) -> Option<u64> {
        self.inner.core.lock().state.log().term_at(index)
    }

    /// A value over the terms in a range, so two replicas can be compared
    /// without shipping their logs
    pub fn log_signature(&self, from: u64, to: u64) -> u64 {
        let core = self.inner.core.lock();
        let log = core.state.log();
        let mut acc = 0u64;
        for index in from..=to {
            let term = log.term_at(index).unwrap_or(u64::MAX);
            acc = acc
                .rotate_left(7)
                .wrapping_add(index.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ term);
        }
        acc
    }

    pub fn machine(&self) -> &Arc<dyn StateMachine> {
        &self.inner.machine
    }

    pub fn config(&self) -> &RaftConfig {
        &self.inner.config
    }

    pub fn snapshot_meta(&self) -> Option<SnapshotMeta> {
        self.inner
            .snapshots
            .lock()
            .current()
            .map(|s| s.meta.clone())
    }

    /// The entry the log is compacted to, which is the point a restart of
    /// this node replays from at the earliest. Read by the engine's WAL
    /// retention so nothing a restart still needs is reclaimed underneath it
    pub fn snapshot_point(&self) -> u64 {
        self.inner.core.lock().state.log().prev_index()
    }

    pub fn metrics(&self) -> RaftMetrics {
        let core = self.inner.core.lock();
        let config = core.cluster_config();
        RaftMetrics {
            id: self.inner.id,
            role: core.role(),
            term: core.term(),
            leader_id: core.leader_id(),
            commit_index: core.commit_index(),
            last_applied: core.last_applied(),
            last_log_index: core.last_log_index(),
            first_log_index: core.state.log().first_index(),
            persisted_index: core.state.log().persisted_index(),
            snapshot_index: core.state.log().prev_index(),
            voters: config.voter_count(),
            learners: config.nodes.len() - config.voter_count(),
            consensus: core.metrics.clone(),
            transport: self.inner.transport.stats(),
            snapshots_created: self.inner.snapshots_created.load(Ordering::Relaxed),
            snapshot_bytes_sent: self.inner.snapshot_bytes_sent.load(Ordering::Relaxed),
            snapshot_bytes_received: self.inner.snapshot_bytes_received.load(Ordering::Relaxed),
            proposals: self.inner.proposals.load(Ordering::Relaxed),
        }
    }

    /// Waits until this node knows who leads
    pub async fn wait_for_leader(&self, timeout: Duration) -> Result<NodeId> {
        let mut rx = self.inner.role_tx.subscribe();
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(leader) = self.leader_id() {
                return Ok(leader);
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                let core = self.inner.core.lock();
                return Err(ZyronError::ElectionTimeout {
                    term: core.term(),
                    elapsed_ms: timeout.as_millis() as u64,
                });
            }
            if tokio::time::timeout(left.min(Duration::from_millis(20)), rx.changed())
                .await
                .is_err()
            {
                continue;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Writes
    // -----------------------------------------------------------------------

    /// Replicates one command and returns once it is committed
    pub async fn propose(&self, command: RaftCommand) -> Result<u64> {
        self.propose_batch(vec![command]).await
    }

    /// Appends a run of commands and returns where the first one landed,
    /// without waiting for any of them to commit.
    ///
    /// A transaction that outgrows one entry proposes while it is still
    /// running, and a caller that waited on each chunk would turn a bulk load
    /// into one round trip per chunk. The caller waits once, on the entry that
    /// completes the transaction
    pub fn propose_batch_detached(&self, commands: Vec<RaftCommand>) -> Result<u64> {
        if commands.is_empty() {
            return Err(ZyronError::Internal(
                "an empty proposal has no index to report".into(),
            ));
        }
        let count = commands.len() as u64;
        let (last, _term) = {
            let mut core = self.inner.core.lock();
            core.propose_many(commands)?
        };
        self.inner.proposals.fetch_add(count, Ordering::Relaxed);
        wake_replicator(&self.inner);
        Ok(last + 1 - count)
    }

    /// Replicates a run of commands under one append and one fsync.
    ///
    /// Returns the index of the last one, which is committed when this
    /// returns, and so is everything before it
    pub async fn propose_batch(&self, commands: Vec<RaftCommand>) -> Result<u64> {
        if commands.is_empty() {
            return Ok(self.commit_index());
        }
        let count = commands.len() as u64;
        let (index, term) = {
            let mut core = self.inner.core.lock();
            core.propose_many(commands)?
        };
        self.inner.proposals.fetch_add(count, Ordering::Relaxed);
        wake_replicator(&self.inner);
        self.wait_committed(index, term).await?;
        Ok(index)
    }

    /// Waits for a specific entry to commit, or for the log to prove it never
    /// will.
    ///
    /// ## Why this does not touch the consensus lock while it waits
    ///
    /// Every outstanding write waits here, and a busy leader has hundreds of
    /// them. Taking the lock on each wakeup to read the commit index put the
    /// whole waiting set through one mutex on every commit advance, which on a
    /// five hundred way client is a million acquisitions a second. The ticker
    /// and the replication loop are behind the same lock, so the leader ended
    /// up unable to send the heartbeats that prove it still holds the group,
    /// and stood itself down mid-run.
    ///
    /// So the wait is against the published watermark, which costs a shared
    /// read and no contention. The lock is taken once, at the end, and only to
    /// answer the question the watermark cannot: the term at the index. An
    /// entry this node proposed can be replaced by a new leader's, and a
    /// caller told its write committed when a different entry now sits there
    /// would be told a falsehood
    async fn wait_committed(&self, index: u64, term: u64) -> Result<()> {
        let mut rx = self.inner.commit_tx.subscribe();
        let deadline = Instant::now() + self.inner.config.propose_timeout;
        loop {
            if *rx.borrow_and_update() >= index {
                let core = self.inner.core.lock();
                if core.commit_index() >= index {
                    return match core.state.log().term_at(index) {
                        // Compacted, which only happens after it applied
                        None => Ok(()),
                        Some(t) if t == term => Ok(()),
                        Some(_) => Err(ZyronError::NotLeader {
                            leader: core.leader_id(),
                        }),
                    };
                }
            }
            if let Some(err) = self.inner.log_writer.failure() {
                return Err(ZyronError::WalWriteFailed(err));
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Err(ZyronError::ConsensusTimeout {
                    operation: format!("commit of index {index}"),
                    elapsed_ms: self.inner.config.propose_timeout.as_millis() as u64,
                });
            }
            // A commit advance ends the wait early. The bound is what makes
            // the overwrite check happen at all, and it is deliberately not on
            // the wakeup path: a proposal being overwritten means this node
            // lost the group, which is rare enough to notice on a timer
            if tokio::time::timeout(left.min(Duration::from_millis(20)), rx.changed())
                .await
                .is_err()
            {
                let core = self.inner.core.lock();
                if let Some(t) = core.state.log().term_at(index) {
                    if t != term {
                        return Err(ZyronError::NotLeader {
                            leader: core.leader_id(),
                        });
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Reads
    // -----------------------------------------------------------------------

    /// The index this node must have applied before a local read is
    /// linearizable.
    ///
    /// On a leader this is its own commit index, confirmed by a heartbeat
    /// round or by a lease that a heartbeat round established recently enough.
    /// On a follower it is whatever the leader answers
    pub async fn read_index(&self) -> Result<u64> {
        let deadline = Instant::now() + self.inner.config.propose_timeout;
        loop {
            let outcome = {
                let mut core = self.inner.core.lock();
                core.begin_read_index(Instant::now())
            };
            match outcome {
                ReadIndexOutcome::Ready(index) => return Ok(index),
                ReadIndexOutcome::Pending { round, index } => {
                    wake_replicator(&self.inner);
                    self.wait_round(round, deadline).await?;
                    return Ok(index);
                }
                ReadIndexOutcome::NotLeader(Some(leader)) => {
                    return self.read_index_from(leader).await;
                }
                ReadIndexOutcome::NotLeader(None) | ReadIndexOutcome::NotReady => {
                    if Instant::now() >= deadline {
                        let core = self.inner.core.lock();
                        return Err(ZyronError::ElectionTimeout {
                            term: core.term(),
                            elapsed_ms: self.inner.config.propose_timeout.as_millis() as u64,
                        });
                    }
                    tokio::time::sleep(self.inner.config.tick_interval).await;
                }
            }
        }
    }

    async fn read_index_from(&self, leader: NodeId) -> Result<u64> {
        let term = self.term();
        let reply = self
            .inner
            .transport
            .send_read_index(
                leader,
                ReadIndexRequest {
                    term,
                    from: self.inner.id,
                },
            )
            .await
            .map_err(ZyronError::from)?;
        if !reply.success {
            return Err(ZyronError::NotLeader {
                leader: reply.leader_id,
            });
        }
        Ok(reply.read_index)
    }

    /// Blocks until the state machine has applied everything up to `index`
    pub async fn wait_applied(&self, index: u64) -> Result<()> {
        let mut rx = self.inner.applied_tx.subscribe();
        let deadline = Instant::now() + self.inner.config.propose_timeout;
        loop {
            if self.last_applied() >= index {
                return Ok(());
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Err(ZyronError::ConsensusTimeout {
                    operation: format!("apply of index {index}"),
                    elapsed_ms: self.inner.config.propose_timeout.as_millis() as u64,
                });
            }
            let _ = tokio::time::timeout(left.min(Duration::from_millis(20)), rx.changed()).await;
        }
    }

    /// Establishes the read index and waits for it, after which a local read
    /// of the state machine returns everything committed before the call
    pub async fn linearizable_read_index(&self) -> Result<u64> {
        let index = self.read_index().await?;
        self.wait_applied(index).await?;
        Ok(index)
    }

    async fn wait_round(&self, round: u64, deadline: Instant) -> Result<()> {
        let mut rx = self.inner.round_tx.subscribe();
        loop {
            {
                let core = self.inner.core.lock();
                if !core.is_leader() {
                    return Err(ZyronError::NotLeader {
                        leader: core.leader_id(),
                    });
                }
                if core.quorum_round() >= round {
                    return Ok(());
                }
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Err(ZyronError::ConsensusTimeout {
                    operation: format!("read index round {round}"),
                    elapsed_ms: self.inner.config.propose_timeout.as_millis() as u64,
                });
            }
            let _ = tokio::time::timeout(left.min(Duration::from_millis(5)), rx.changed()).await;
        }
    }

    // -----------------------------------------------------------------------
    // Membership
    // -----------------------------------------------------------------------

    /// Brings a node in: first as a learner, then as a voter once it has
    /// caught up.
    ///
    /// The two steps are separate because they carry different risk. The
    /// learner step cannot hurt the group whatever state the new node is in.
    /// The promotion changes what a majority is, so it waits for the node to
    /// be close enough that the group does not stall on it
    pub async fn add_node(&self, node_id: NodeId, address: &str) -> Result<()> {
        if !self.is_leader() {
            return Err(self.inner.core.lock().not_leader());
        }
        let command = {
            let core = self.inner.core.lock();
            core.add_node_command(node_id, address)?
        };
        self.inner.transport.set_address(node_id, address);
        self.propose(command).await?;
        wake_replicator(&self.inner);

        // Wait for the learner to be close, then promote through a joint
        // configuration
        let deadline = Instant::now() + self.inner.config.propose_timeout;
        loop {
            let caught_up = {
                let core = self.inner.core.lock();
                if !core.is_leader() {
                    return Err(core.not_leader());
                }
                core.learner_is_caught_up(node_id, PROMOTION_TOLERANCE)
            };
            if caught_up {
                break;
            }
            if Instant::now() >= deadline {
                return Err(ZyronError::ConsensusTimeout {
                    operation: format!("learner {node_id} catching up"),
                    elapsed_ms: self.inner.config.propose_timeout.as_millis() as u64,
                });
            }
            tokio::time::sleep(self.inner.config.tick_interval).await;
        }

        let joint = {
            let core = self.inner.core.lock();
            core.promote_command(node_id)?
        };
        self.propose(joint).await?;
        let final_config = {
            let core = self.inner.core.lock();
            core.leave_joint_command()?
        };
        self.propose(final_config).await?;
        Ok(())
    }

    /// Takes a node out.
    ///
    /// A learner leaves in one entry. A voter leaves through a joint
    /// configuration, and if the node leaving is this one, it gives up the
    /// group once the change commits rather than leading a group it is not in
    pub async fn remove_node(&self, node_id: NodeId) -> Result<()> {
        if !self.is_leader() {
            return Err(self.inner.core.lock().not_leader());
        }
        let command = {
            let core = self.inner.core.lock();
            core.remove_node_command(node_id)?
        };
        let was_joint =
            command.is_config_change() && matches!(command, RaftCommand::JointConfig { .. });
        self.propose(command).await?;
        if was_joint {
            let final_config = {
                let core = self.inner.core.lock();
                core.leave_joint_command()?
            };
            self.propose(final_config).await?;
        }
        self.inner.transport.forget(node_id);
        if node_id == self.inner.id {
            let mut core = self.inner.core.lock();
            core.step_down(Instant::now());
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Snapshots
    // -----------------------------------------------------------------------

    /// Captures the state machine at its current applied index and installs it
    /// as this node's snapshot, then discards the log it covers.
    ///
    /// The capture is taken under the machine's own lock and the bytes are
    /// written afterwards on a blocking thread, so proposals and replication
    /// carry on for the whole of the write
    pub async fn create_snapshot(&self) -> Result<SnapshotMeta> {
        if self.inner.snapshotting.swap(true, Ordering::AcqRel) {
            return Err(ZyronError::Internal(
                "a snapshot is already being written on this node".into(),
            ));
        }
        let result = self.create_snapshot_inner().await;
        self.inner.snapshotting.store(false, Ordering::Release);
        result
    }

    async fn create_snapshot_inner(&self) -> Result<SnapshotMeta> {
        let mut source = self.inner.machine.begin_checkpoint()?;
        let applied = source.last_applied();
        // A transaction that spans entries has to be replayable from its
        // first one after a restart, so the machine's floor caps compaction
        // the same way the slowest member does
        let machine_floor = self.inner.machine.retain_floor();
        // Compaction stops at whichever is lower, what this node has applied
        // or what the slowest member has confirmed holding. A member that is
        // merely behind catches up from the log, which is cheap; one that has
        // been left behind by the retention bound needs a copy of somebody's
        // data, which is not. Keeping the log for the slower of the two turns
        // the second case into the rare one it should be
        let (index, term, config, retained, left_behind) = {
            let core = self.inner.core.lock();
            let held = core.group_match_index();
            let bound = self.inner.config.snapshot_threshold_bytes;
            // Past the retention bound the log is compacted whatever anyone
            // still needs. A member that cannot be left waited for forever is
            // better told plainly that it needs a copy of somebody's data than
            // left pinning a log that grows without limit
            let over_bound = bound != 0 && core.state.log().stored_bytes() >= bound;
            let index = if over_bound {
                applied
            } else {
                applied.min(held.max(core.state.log().prev_index()))
            };
            if applied == 0 {
                return Err(ZyronError::Internal(
                    "nothing has been applied, so there is nothing to snapshot".into(),
                ));
            }
            let index = index.min(machine_floor).max(core.state.log().prev_index());
            if index <= core.state.log().prev_index() {
                return Err(ZyronError::Internal(
                    "the log is pinned by a transaction still being replayed, so there is nothing to compact yet".into(),
                ));
            }
            let Some(term) = core.state.log().term_at(index) else {
                return Err(ZyronError::Internal(format!(
                    "no log entry at {index} to anchor a snapshot"
                )));
            };
            (
                index,
                term,
                core.snapshot_config(),
                applied - index,
                over_bound && held < index,
            )
        };
        if retained > 0 {
            tracing::debug!(
                node = self.inner.id,
                applied,
                index,
                retained,
                "log entries kept for a member that has not caught up"
            );
        }
        if left_behind {
            tracing::warn!(
                node = self.inner.id,
                index,
                "a member is further behind than the log is kept for, so it will have to be started again from a copy of a current member's data"
            );
        }

        let staged = self.inner.dir.join(format!("snapshot-{index:020}.staged"));
        let write_path = staged.clone();
        let bytes = tokio::task::spawn_blocking(move || source.write_to(&write_path))
            .await
            .map_err(|e| ZyronError::Internal(format!("snapshot write task failed: {e}")))??;

        let meta = SnapshotMeta {
            last_included_index: index,
            last_included_term: term,
            config,
            size_bytes: bytes,
        };
        let published = {
            let mut store = self.inner.snapshots.lock();
            store.publish(meta, &staged)?
        };
        {
            let mut core = self.inner.core.lock();
            core.compact_log(index, term)?;
        }
        self.inner.snapshots_created.fetch_add(1, Ordering::Relaxed);
        tracing::info!(
            node = self.inner.id,
            index,
            term,
            bytes = published.meta.size_bytes,
            "snapshot taken and log compacted"
        );
        Ok(published.meta)
    }
}

// ---------------------------------------------------------------------------
// The loops
// ---------------------------------------------------------------------------

async fn ticker(inner: Arc<RaftInner>) {
    let mut interval = tokio::time::interval(inner.config.tick_interval);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            biased;
            _ = inner.shutdown.notified() => break,
            _ = interval.tick() => {}
        }
        if !inner.running.load(Ordering::Acquire) {
            break;
        }
        let now = Instant::now();
        let (outcome, version, role) = {
            let mut core = inner.core.lock();
            let outcome = core.tick(now);
            (
                outcome,
                core.membership_version(),
                (core.role(), core.term(), core.leader_id()),
            )
        };
        publish_role(&inner, role);
        sync_directory(&inner, version);

        for (peer, request) in outcome.vote_requests {
            tokio::spawn(send_vote(Arc::clone(&inner), peer, request));
        }
        if outcome.replicate || outcome.became_leader {
            wake_replicator(&inner);
        }
        if outcome.became_leader {
            publish_commit(&inner);
        }
    }
}

/// Points the transport at wherever the current configuration says nodes are
fn sync_directory(inner: &Arc<RaftInner>, version: u64) {
    if inner.directory_version.load(Ordering::Relaxed) == version {
        return;
    }
    let config = inner.core.lock().cluster_config();
    for node in &config.nodes {
        if node.node_id != inner.id {
            inner.transport.set_address(node.node_id, &node.address);
        }
    }
    inner.directory_version.store(version, Ordering::Relaxed);
}

fn publish_role(inner: &Arc<RaftInner>, role: (RaftRole, u64, Option<NodeId>)) {
    if *inner.role_tx.borrow() != role {
        let _ = inner.role_tx.send_replace(role);
    }
}

/// Wakes the replication loop.
///
/// A bump rather than a signal, so a wake that lands while the loop is busy is
/// still seen the moment it comes back round
fn wake_replicator(inner: &RaftInner) {
    inner
        .work_tx
        .send_modify(|generation| *generation = generation.wrapping_add(1));
}

fn publish_commit(inner: &Arc<RaftInner>) {
    let commit = inner.core.lock().commit_index();
    if *inner.commit_tx.borrow() != commit {
        inner.commit_tx.send_replace(commit);
        // A follower learns what is committed only from the leader, so an
        // advance here is a message owed to every one of them
        wake_replicator(inner);
    }
}

fn publish_round(inner: &Arc<RaftInner>) {
    let round = inner.core.lock().quorum_round();
    if *inner.round_tx.borrow() != round {
        inner.round_tx.send_replace(round);
    }
}

/// Asks one peer for its vote and records the answer.
///
/// Boxed rather than a plain `async fn` because a won pre-vote produces the
/// real vote requests, and this is what sends those too. A recursive async fn
/// has no nameable type for the compiler to prove `Send` about
fn send_vote(
    inner: Arc<RaftInner>,
    peer: NodeId,
    request: RequestVoteRequest,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>> {
    Box::pin(send_vote_inner(inner, peer, request))
}

async fn send_vote_inner(inner: Arc<RaftInner>, peer: NodeId, request: RequestVoteRequest) {
    let pre_vote = request.pre_vote;
    let term = request.term;
    match inner.transport.send_request_vote(peer, request).await {
        Ok(reply) => {
            let follow_up = {
                let mut core = inner.core.lock();
                core.handle_request_vote_reply(&reply, Instant::now())
            };
            match follow_up {
                Ok(requests) => {
                    for (peer, request) in requests {
                        tokio::spawn(send_vote(Arc::clone(&inner), peer, request));
                    }
                }
                Err(e) => {
                    tracing::warn!(node = inner.id, error = %e, "could not record a vote reply");
                }
            }
            let role = {
                let core = inner.core.lock();
                (core.role(), core.term(), core.leader_id())
            };
            publish_role(&inner, role);
            publish_commit(&inner);
            wake_replicator(&inner);
        }
        Err(e) => {
            tracing::debug!(
                node = inner.id,
                peer,
                term,
                pre_vote,
                error = %e,
                "vote request produced no answer"
            );
        }
    }
}

async fn replicator(inner: Arc<RaftInner>) {
    let mut work = inner.work_tx.subscribe();
    loop {
        // Marked seen before the pass, so a bump that arrives during it is
        // still waiting when the pass ends
        work.borrow_and_update();
        let peers = {
            let core = inner.core.lock();
            if core.is_leader() {
                core.peers()
            } else {
                Vec::new()
            }
        };
        for peer in peers {
            loop {
                let work = {
                    let mut core = inner.core.lock();
                    core.build_peer_work(peer, Instant::now())
                };
                match work {
                    Ok(PeerWork::Idle) => break,
                    Ok(PeerWork::Append(request)) => {
                        let inner = Arc::clone(&inner);
                        tokio::spawn(async move {
                            send_append(inner, peer, request).await;
                        });
                    }
                    Ok(PeerWork::AppendPaged {
                        request,
                        plan,
                        pager,
                    }) => {
                        let inner = Arc::clone(&inner);
                        tokio::spawn(async move {
                            send_paged_append(inner, peer, request, plan, pager).await;
                        });
                    }
                    Ok(PeerWork::Snapshot) => {
                        let inner = Arc::clone(&inner);
                        tokio::spawn(async move {
                            send_snapshot(inner, peer).await;
                        });
                        break;
                    }
                    Err(e) => {
                        tracing::warn!(node = inner.id, peer, error = %e, "could not build replication work");
                        break;
                    }
                }
            }
        }
        tokio::select! {
            biased;
            _ = inner.shutdown.notified() => break,
            _ = work.changed() => {}
            // A heartbeat is due even when nothing has been proposed, and the
            // tick bound keeps that inside the heartbeat interval
            _ = tokio::time::sleep(inner.config.tick_interval) => {}
        }
        if !inner.running.load(Ordering::Acquire) {
            break;
        }
    }
}

/// Reads the entries a batch is missing, then sends it.
///
/// The reads run on a blocking thread because they are file reads, and the
/// batch is abandoned rather than sent short if any of them fails: a follower
/// that received a gap would take it as the leader's log and truncate its own.
/// Abandoning leaves `next_index` optimistically advanced, which the next
/// refused append rewinds, and a follower that keeps missing this way ends up
/// taking a snapshot instead
async fn send_paged_append(
    inner: Arc<RaftInner>,
    peer: NodeId,
    mut request: AppendEntriesRequest,
    plan: Vec<SliceItem>,
    pager: Arc<LogPager>,
) {
    let read = tokio::task::spawn_blocking(move || {
        let mut entries = Vec::with_capacity(plan.len());
        for item in plan {
            match item {
                SliceItem::Resident(entry) => entries.push(entry),
                SliceItem::Paged(record) => {
                    entries.push(pager.read(
                        record.index,
                        record.term,
                        record.raw_offset(),
                        record.len(),
                    )?);
                }
            }
        }
        Ok::<_, ZyronError>(entries)
    })
    .await;

    let entries = match read {
        Ok(Ok(entries)) => entries,
        Ok(Err(e)) => {
            tracing::warn!(node = inner.id, peer, error = %e, "could not read log entries back for replication");
            let mut core = inner.core.lock();
            core.handle_peer_unreachable(peer);
            return;
        }
        Err(e) => {
            tracing::warn!(node = inner.id, peer, error = %e, "the log read back task did not finish");
            let mut core = inner.core.lock();
            core.handle_peer_unreachable(peer);
            return;
        }
    };
    request.entries = entries;
    send_append(inner, peer, request).await;
}

async fn send_append(inner: Arc<RaftInner>, peer: NodeId, request: AppendEntriesRequest) {
    let had_entries = !request.entries.is_empty();
    match inner.transport.send_append_entries(peer, request).await {
        Ok(reply) => {
            {
                let mut core = inner.core.lock();
                if let Err(e) = core.handle_append_reply(&reply, Instant::now()) {
                    tracing::warn!(node = inner.id, peer, error = %e, "could not record an append reply");
                }
            }
            publish_commit(&inner);
            publish_round(&inner);
            let role = {
                let core = inner.core.lock();
                (core.role(), core.term(), core.leader_id())
            };
            publish_role(&inner, role);
            if had_entries || !reply.success {
                wake_replicator(&inner);
            }
        }
        Err(e) => {
            if !matches!(e, RaftRpcError::Shutdown) {
                tracing::debug!(node = inner.id, peer, error = %e, "append produced no answer");
            }
            let mut core = inner.core.lock();
            core.handle_peer_unreachable(peer);
        }
    }
}

/// Streams the current snapshot to one follower, one chunk at a time.
///
/// The reply is the flow control: the next chunk is read only once the
/// previous one has been acknowledged, so a slow receiver slows the sender
/// rather than filling its memory
async fn send_snapshot(inner: Arc<RaftInner>, peer: NodeId) {
    let mut delivered = None;
    let outcome = stream_snapshot(&inner, peer, &mut delivered).await;
    if let Err(e) = outcome {
        tracing::warn!(node = inner.id, peer, error = %e, "snapshot transfer did not finish");
    }
    {
        let mut core = inner.core.lock();
        core.finish_snapshot_transfer(peer, delivered);
    }
    wake_replicator(&inner);
}

async fn stream_snapshot(
    inner: &Arc<RaftInner>,
    peer: NodeId,
    delivered: &mut Option<u64>,
) -> Result<()> {
    let (meta, term, leader_id) = {
        let core = inner.core.lock();
        let store = inner.snapshots.lock();
        let Some(snapshot) = store.current() else {
            return Err(ZyronError::Internal(
                "a follower needs a snapshot and this node has none".into(),
            ));
        };
        (snapshot.meta.clone(), core.term(), core.leader_id())
    };
    if leader_id != Some(inner.id) {
        return Err(ZyronError::NotLeader { leader: leader_id });
    }

    let chunk_bytes = inner.config.snapshot_chunk_bytes;
    let mut offset = 0u64;
    loop {
        let (data, done) = {
            let store = inner.snapshots.lock();
            store.read_chunk(offset, chunk_bytes)?
        };
        let sent = data.len() as u64;
        let request = InstallSnapshotRequest {
            term,
            leader_id: inner.id,
            last_included_index: meta.last_included_index,
            last_included_term: meta.last_included_term,
            config: meta.config.clone(),
            offset,
            data,
            done,
        };
        let reply = inner
            .transport
            .send_install_snapshot(peer, request)
            .await
            .map_err(ZyronError::from)?;
        inner.snapshot_bytes_sent.fetch_add(sent, Ordering::Relaxed);
        if reply.term > term {
            let mut core = inner.core.lock();
            core.observe_peer_term(reply.term, Instant::now())?;
            return Ok(());
        }
        if !reply.success {
            return Err(ZyronError::Internal(format!(
                "node {peer} refused a snapshot chunk at offset {offset}"
            )));
        }
        offset = reply.bytes_received;
        if done {
            *delivered = Some(meta.last_included_index);
            return Ok(());
        }
    }
}

async fn applier(inner: Arc<RaftInner>) {
    let mut rx = inner.commit_tx.subscribe();
    loop {
        let applied_any = apply_ready(&inner).await;
        maybe_snapshot(&inner);
        if applied_any {
            continue;
        }
        tokio::select! {
            biased;
            _ = inner.shutdown.notified() => break,
            _ = rx.changed() => {}
            _ = tokio::time::sleep(inner.config.tick_interval) => {}
        }
        if !inner.running.load(Ordering::Acquire) {
            break;
        }
    }
}

/// Hands whatever is committed and unapplied to the state machine.
///
/// The batch is planned behind the lock and resolved outside it. An entry the
/// residency cap has evicted is read back from the log file, because a
/// follower further behind on apply than the cap keeps in memory would
/// otherwise wait forever for entries that are on its own disk. The read runs
/// on a blocking thread, and consensus never waits on it
async fn apply_ready(inner: &Arc<RaftInner>) -> bool {
    let (plan, pager) = {
        let mut core = inner.core.lock();
        let from = core.last_applied() + 1;
        let to = core.commit_index();
        if from > to {
            (Vec::new(), None)
        } else {
            // Bounded to the commit index by the entry count: the slice is
            // gapless and starts at `from`, so capping the count is capping
            // the index, and nothing uncommitted can be handed to the machine
            let max_entries = inner.config.apply_batch.min((to - from + 1) as usize);
            let plan =
                core.state
                    .log()
                    .plan_slice(from, max_entries, inner.config.resident_log_bytes);
            let paged = plan
                .iter()
                .filter(|item| matches!(item, SliceItem::Paged(_)))
                .count();
            let pager = if paged > 0 {
                core.metrics.entries_paged_for_apply += paged as u64;
                Some(core.state.log().pager())
            } else {
                None
            };
            (plan, pager)
        }
    };
    if plan.is_empty() {
        return false;
    }
    let batch: Vec<Arc<RaftLogEntry>> = match pager {
        Some(pager) => {
            let read = tokio::task::spawn_blocking(move || {
                let mut out = Vec::with_capacity(plan.len());
                for item in plan {
                    match item {
                        SliceItem::Resident(entry) => out.push(entry),
                        SliceItem::Paged(record) => out.push(pager.read(
                            record.index,
                            record.term,
                            record.raw_offset(),
                            record.len(),
                        )?),
                    }
                }
                Ok::<_, ZyronError>(out)
            })
            .await;
            match read {
                Ok(Ok(batch)) => batch,
                Ok(Err(e)) => {
                    tracing::error!(node = inner.id, error = %e, "committed entries could not be read back for apply");
                    return false;
                }
                Err(e) => {
                    tracing::error!(node = inner.id, error = %e, "the apply read back task did not finish");
                    return false;
                }
            }
        }
        None => plan
            .into_iter()
            .map(|item| match item {
                SliceItem::Resident(entry) => entry,
                SliceItem::Paged(_) => unreachable!("counted above"),
            })
            .collect(),
    };
    let last = match inner.machine.apply_batch(&batch).await {
        Ok(last) => last,
        Err(e) => {
            tracing::error!(
                node = inner.id,
                from = batch[0].index,
                error = %e,
                "state machine refused a committed entry, apply is stopped"
            );
            return false;
        }
    };
    if last == 0 {
        return false;
    }
    {
        let mut core = inner.core.lock();
        core.set_last_applied(last);
    }
    if *inner.applied_tx.borrow() != last {
        inner.applied_tx.send_replace(last);
    }
    true
}

/// Takes a snapshot once the log has grown past either threshold.
///
/// Entries and bytes both count. A log of keys hits the entry count first and
/// a log of transactions hits the byte budget first, and a group that only
/// watched the count would carry gigabytes of committed log it no longer
/// needs.
///
/// Started rather than awaited, so the apply loop is not blocked behind the
/// write. The flag makes sure only one runs
fn maybe_snapshot(inner: &Arc<RaftInner>) {
    let threshold = inner.config.snapshot_threshold;
    let threshold_bytes = inner.config.snapshot_threshold_bytes;
    if threshold == 0 && threshold_bytes == 0 {
        return;
    }
    let (applied, snapshot_point, stored) = {
        let core = inner.core.lock();
        (
            core.last_applied(),
            core.state.log().prev_index(),
            core.state.log().stored_bytes(),
        )
    };
    let by_count = threshold != 0 && applied >= snapshot_point + threshold;
    let by_bytes = threshold_bytes != 0 && stored >= threshold_bytes;
    if !by_count && !by_bytes {
        return;
    }
    // A snapshot covers what has been applied, so one triggered by size with
    // nothing applied past the last would compact nothing and run forever
    if applied <= snapshot_point {
        return;
    }
    // A transaction still being streamed pins the log at its first entry.
    // Waiting quietly is right: the pin is released when the transaction
    // commits or is abandoned, and a pass now would compact nothing
    if inner.machine.retain_floor() <= snapshot_point {
        return;
    }
    // Under the retention bound the log is kept for the slowest member, and a
    // pass that would stop where the last one did is one that does nothing
    if !by_bytes {
        let held = {
            let core = inner.core.lock();
            core.group_match_index()
        };
        if held <= snapshot_point {
            return;
        }
    }
    if inner.snapshotting.swap(true, Ordering::AcqRel) {
        return;
    }
    let inner = Arc::clone(inner);
    tokio::spawn(async move {
        let node = RaftNode {
            inner: Arc::clone(&inner),
        };
        if let Err(e) = node.create_snapshot_inner().await {
            tracing::warn!(node = inner.id, error = %e, "scheduled snapshot did not complete");
        }
        inner.snapshotting.store(false, Ordering::Release);
    });
}

/// Turns the log writer's fsync into a commit on the leader.
///
/// The leader's own durable index is its contribution to every quorum, so
/// without this a group whose followers are idle would never advance its
/// commit index between heartbeats
async fn durability_watcher(inner: Arc<RaftInner>) {
    let mut rx = inner.log_writer.subscribe();
    let mut published = 0u64;
    loop {
        let commit = {
            let mut core = inner.core.lock();
            core.advance_commit();
            core.commit_index()
        };
        publish_commit(&inner);
        // A commit this node's own fsync produced is one no follower has been
        // told about: nothing was replied to and nothing was sent. Waiting for
        // the next heartbeat to carry it leaves every follower up to a
        // heartbeat behind on a group that has just gone quiet, which is
        // exactly when a read is most likely to arrive
        if commit > published {
            published = commit;
            wake_replicator(&inner);
        }
        tokio::select! {
            biased;
            _ = inner.shutdown.notified() => break,
            _ = rx.changed() => {}
            _ = tokio::time::sleep(inner.config.tick_interval) => {}
        }
        if !inner.running.load(Ordering::Acquire) {
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// Serving peers
// ---------------------------------------------------------------------------

impl RaftRequestHandler for RaftNode {
    fn on_request_vote(&self, req: RequestVoteRequest) -> RaftHandlerFuture<'_, RequestVoteReply> {
        let inner = Arc::clone(&self.inner);
        let outcome = {
            let mut core = inner.core.lock();
            core.handle_request_vote(&req, Instant::now())
        };
        Box::pin(async move {
            let reply = outcome?;
            // A granted vote is a promise, and the term and vote it rests on
            // are already on disk because the consensus core fsyncs them
            // before it returns
            let role = {
                let core = inner.core.lock();
                (core.role(), core.term(), core.leader_id())
            };
            publish_role(&inner, role);
            Ok(reply)
        })
    }

    fn on_append_entries(
        &self,
        req: AppendEntriesRequest,
    ) -> RaftHandlerFuture<'_, AppendEntriesReply> {
        let inner = Arc::clone(&self.inner);
        // The entries are taken into the log here, in the order the frames
        // arrived. Only the wait for the fsync is deferred into the future
        let outcome = {
            let mut core = inner.core.lock();
            core.handle_append_entries(&req, Instant::now())
        };
        Box::pin(async move {
            let reply = outcome?;
            if reply.success {
                wait_persisted(&inner, reply.match_index).await?;
            }
            publish_commit(&inner);
            let role = {
                let core = inner.core.lock();
                (core.role(), core.term(), core.leader_id())
            };
            publish_role(&inner, role);
            Ok(reply)
        })
    }

    fn on_install_snapshot(
        &self,
        req: InstallSnapshotRequest,
    ) -> RaftHandlerFuture<'_, InstallSnapshotReply> {
        let inner = Arc::clone(&self.inner);
        let decision = {
            let mut core = inner.core.lock();
            core.check_install_snapshot(
                req.term,
                req.leader_id,
                req.last_included_index,
                Instant::now(),
            )
        };
        Box::pin(async move { install_snapshot(inner, req, decision?).await })
    }

    fn on_read_index(&self, req: ReadIndexRequest) -> RaftHandlerFuture<'_, ReadIndexReply> {
        let inner = Arc::clone(&self.inner);
        let outcome = {
            let mut core = inner.core.lock();
            core.handle_read_index(&req, Instant::now())
        };
        let timeout = inner.config.propose_timeout;
        Box::pin(async move {
            let (outcome, reply) = outcome?;
            if let ReadIndexOutcome::Pending { round, .. } = outcome {
                // The index is only safe to hand out once a majority has
                // echoed the round, so the answer waits rather than being sent
                // on a leadership this node has not re-confirmed
                wake_replicator(&inner);
                let node = RaftNode {
                    inner: Arc::clone(&inner),
                };
                node.wait_round(round, Instant::now() + timeout).await?;
            }
            Ok(reply)
        })
    }
}

async fn wait_persisted(inner: &Arc<RaftInner>, index: u64) -> Result<()> {
    let mut rx = inner.log_writer.subscribe();
    loop {
        if inner.log_writer.persisted_index() >= index {
            return Ok(());
        }
        if let Some(err) = inner.log_writer.failure() {
            return Err(ZyronError::WalWriteFailed(err));
        }
        if rx.changed().await.is_err() {
            return Err(ZyronError::WalWriteFailed(
                "the raft log writer stopped before the entry was durable".into(),
            ));
        }
    }
}

async fn install_snapshot(
    inner: Arc<RaftInner>,
    req: InstallSnapshotRequest,
    decision: SnapshotDecision,
) -> Result<InstallSnapshotReply> {
    let term = inner.core.lock().term();
    let mut reply = InstallSnapshotReply {
        term,
        success: false,
        bytes_received: 0,
        follower_id: inner.id,
    };
    match decision {
        SnapshotDecision::Reject(term) => {
            reply.term = term;
            return Ok(reply);
        }
        SnapshotDecision::AlreadyCovered => {
            // Nothing to write, and reporting the whole size ends the transfer
            reply.success = true;
            reply.bytes_received = u64::MAX;
            return Ok(reply);
        }
        SnapshotDecision::Accept => {}
    }

    let written = {
        let store = inner.snapshots.lock();
        store.write_chunk(
            req.last_included_index,
            req.offset,
            &req.data,
            // The file only has to survive a crash once it is about to become
            // this node's snapshot
            req.done,
        )?
    };
    inner
        .snapshot_bytes_received
        .fetch_add(req.data.len() as u64, Ordering::Relaxed);
    reply.success = true;
    reply.bytes_received = written;

    if !req.done {
        return Ok(reply);
    }

    let (staged, meta) = {
        let mut store = inner.snapshots.lock();
        let staged = store.incoming_path(req.last_included_index);
        let meta = SnapshotMeta {
            last_included_index: req.last_included_index,
            last_included_term: req.last_included_term,
            config: req.config.clone(),
            size_bytes: written,
        };
        let published = store.publish(meta, &staged)?;
        (published.data.clone(), published.meta.clone())
    };

    let machine = Arc::clone(&inner.machine);
    let index = meta.last_included_index;
    tokio::task::spawn_blocking(move || machine.restore(&staged, index))
        .await
        .map_err(|e| ZyronError::Internal(format!("snapshot restore task failed: {e}")))??;

    {
        let mut core = inner.core.lock();
        core.adopt_snapshot(
            meta.last_included_index,
            meta.last_included_term,
            meta.config.clone(),
        )?;
    }
    publish_commit(&inner);
    if *inner.applied_tx.borrow() < index {
        inner.applied_tx.send_replace(index);
    }
    tracing::info!(
        node = inner.id,
        index,
        term = meta.last_included_term,
        bytes = meta.size_bytes,
        "installed a snapshot from the leader"
    );
    Ok(reply)
}

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

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use tokio::sync::{oneshot, watch};
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
    /// Flushes the log writer has made to stable storage
    pub log_fsyncs: u64,
    /// The longest one of them took, in microseconds
    pub log_fsync_max_us: u64,
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
    /// Proposals waiting for the commit index to reach their entry.
    ///
    /// Taken after the consensus lock, never before it, wherever both are
    /// held. A registration takes it alone
    commit_waiters: parking_lot::Mutex<IndexWaiters>,
    /// Reads waiting for the applied index to reach their read index
    applied_waiters: parking_lot::Mutex<IndexWaiters>,

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

/// Tasks waiting for a watermark, the commit index or the applied index, to
/// reach an index of theirs.
///
/// Keyed by index, so an advance wakes the waiters it satisfies and nobody
/// else. A shared watch wakes every waiter on every advance, and under
/// hundreds of outstanding proposals that is a herd of tasks re-arming
/// timers and taking the consensus lock to learn that nothing changed for
/// them.
///
/// Waiters are taken out under a lock and answered after it is released.
/// Answering one schedules a task, and a commit advance under load answers
/// a hundred of them, which is longer than the consensus lock may be held
#[derive(Default)]
struct IndexWaiters {
    by_index: BTreeMap<u64, Vec<IndexWaiter>>,
    next_ticket: u64,
}

struct IndexWaiter {
    /// Tells a waiter that gave up apart from the others at its index
    ticket: u64,
    /// The term the waiter's entry was proposed under, zero when the entry's
    /// identity does not matter to it
    term: u64,
    tx: oneshot::Sender<Result<()>>,
}

/// A waiter taken out of the registry with the verdict it is owed
type Answer = (oneshot::Sender<Result<()>>, Result<()>);

/// Delivers verdicts, after every lock they were decided under is released
fn answer(answers: Vec<Answer>) {
    for (tx, verdict) in answers {
        let _ = tx.send(verdict);
    }
}

impl IndexWaiters {
    fn register(&mut self, index: u64, term: u64) -> (u64, oneshot::Receiver<Result<()>>) {
        let (tx, rx) = oneshot::channel();
        self.next_ticket += 1;
        let ticket = self.next_ticket;
        self.by_index
            .entry(index)
            .or_default()
            .push(IndexWaiter { ticket, term, tx });
        (ticket, rx)
    }

    /// Drops a waiter that stopped waiting, so a proposal that timed out
    /// does not sit here until its index commits
    fn forget(&mut self, index: u64, ticket: u64) {
        if let Some(waiters) = self.by_index.get_mut(&index) {
            waiters.retain(|waiter| waiter.ticket != ticket);
            if waiters.is_empty() {
                self.by_index.remove(&index);
            }
        }
    }

    fn first_index(&self) -> Option<u64> {
        self.by_index.keys().next().copied()
    }

    /// Takes every waiter at or below the mark out, with the verdict for its
    /// index and the term it proposed under
    fn take_up_to(&mut self, mark: u64, verdict: impl Fn(u64, u64) -> Result<()>) -> Vec<Answer> {
        match self.first_index() {
            Some(first) if first <= mark => {}
            _ => return Vec::new(),
        }
        let later = self.by_index.split_off(&(mark + 1));
        let ready = std::mem::replace(&mut self.by_index, later);
        let mut answers = Vec::new();
        for (index, waiters) in ready {
            for waiter in waiters {
                answers.push((waiter.tx, verdict(index, waiter.term)));
            }
        }
        answers
    }

    /// Keeps the waiters the predicate still stands behind and takes the
    /// rest out with the failure
    fn take_where(
        &mut self,
        lose: impl Fn(u64, u64) -> bool,
        err: impl Fn() -> ZyronError,
    ) -> Vec<Answer> {
        let mut answers = Vec::new();
        let mut drained = Vec::new();
        for (index, waiters) in self.by_index.iter_mut() {
            let mut kept = Vec::with_capacity(waiters.len());
            for waiter in waiters.drain(..) {
                if lose(*index, waiter.term) {
                    answers.push((waiter.tx, Err(err())));
                } else {
                    kept.push(waiter);
                }
            }
            *waiters = kept;
            if waiters.is_empty() {
                drained.push(*index);
            }
        }
        for index in drained {
            self.by_index.remove(&index);
        }
        answers
    }

    fn fail_all(&mut self, err: impl Fn() -> ZyronError) {
        for (_, waiters) in std::mem::take(&mut self.by_index) {
            for waiter in waiters {
                let _ = waiter.tx.send(Err(err()));
            }
        }
    }
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
            commit_waiters: parking_lot::Mutex::new(IndexWaiters::default()),
            applied_waiters: parking_lot::Mutex::new(IndexWaiters::default()),
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
        let stopped = || ZyronError::Internal("the node stopped".into());
        self.inner.commit_waiters.lock().fail_all(stopped);
        self.inner.applied_waiters.lock().fail_all(stopped);
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
            log_fsyncs: self.inner.log_writer.fsyncs(),
            log_fsync_max_us: self.inner.log_writer.fsync_max_us(),
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

    /// Hands the group to the best placed follower and waits for it to lead.
    ///
    /// The follower with the most confirmed log is let catch up, then told to
    /// campaign at once. This node steps down when it sees the higher term.
    /// Returns the new leader, or None when this node does not lead or has
    /// no voting peer to hand the group to
    pub async fn transfer_leadership(&self, timeout: Duration) -> Result<Option<NodeId>> {
        let target = {
            let core = self.inner.core.lock();
            if !core.is_leader() {
                return Ok(None);
            }
            core.transfer_target()
        };
        let Some(target) = target else {
            return Ok(None);
        };
        let deadline = Instant::now() + timeout;

        // The target campaigns in a term that must hold everything this
        // leader holds, so it is brought level before it is asked
        wake_replicator(&self.inner);
        while !self.inner.core.lock().peer_caught_up(target) {
            if Instant::now() >= deadline {
                break;
            }
            tokio::time::sleep(self.inner.config.tick_interval).await;
        }

        let request = {
            let core = self.inner.core.lock();
            crate::election::TimeoutNowRequest {
                term: core.term(),
                leader_id: self.inner.id,
            }
        };
        let reply = self
            .inner
            .transport
            .send_timeout_now(target, request)
            .await
            .map_err(ZyronError::from)?;
        if !reply.started {
            return Err(ZyronError::Internal(format!(
                "node {target} declined to take the group at term {}",
                reply.term
            )));
        }

        let mut rx = self.inner.role_tx.subscribe();
        loop {
            if !self.is_leader() {
                if let Some(leader) = self.leader_id() {
                    return Ok(Some(leader));
                }
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Err(ZyronError::ElectionTimeout {
                    term: self.term(),
                    elapsed_ms: timeout.as_millis() as u64,
                });
            }
            let _ = tokio::time::timeout(left.min(Duration::from_millis(20)), rx.changed()).await;
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
        let (last, _term) = self.append_proposal(commands)?;
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
        let (index, term) = self.append_proposal(commands)?;
        self.wait_committed(index, term).await?;
        Ok(index)
    }

    /// Appends a proposal to the log and wakes whoever it gives work to.
    ///
    /// The replicator is woken only when a follower can take the entries
    /// now. Under load every follower has its share in flight, and a wake
    /// per proposal then costs the replicator a pass through the consensus
    /// lock to find that out, at the rate proposals arrive. The reply that
    /// frees a slot wakes it instead
    fn append_proposal(&self, commands: Vec<RaftCommand>) -> Result<(u64, u64)> {
        let count = commands.len() as u64;
        let (index, term, committed, room) = {
            let mut core = self.inner.core.lock();
            let before = core.commit_index();
            let (index, term) = core.propose_many(commands)?;
            (
                index,
                term,
                core.commit_index() > before,
                core.replication_has_room(Instant::now()),
            )
        };
        self.inner.proposals.fetch_add(count, Ordering::Relaxed);
        if committed {
            // The append found this node's own fsync ahead of the durability
            // watcher and moved the commit index, so the proposals it reached
            // are answered here rather than left to the watcher's next pass
            publish_commit(&self.inner);
        }
        if room {
            wake_replicator(&self.inner);
        }
        Ok((index, term))
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
        if let Some(err) = self.inner.log_writer.failure() {
            return Err(ZyronError::WalWriteFailed(err));
        }
        let (ticket, rx) = self.inner.commit_waiters.lock().register(index, term);
        // A commit that landed between the append and the registration is
        // answered here, the same way a later advance answers it. The
        // published watermark says whether one did, and it is read instead
        // of the consensus lock because every proposal passes through here
        // and that lock is the one the replies are waiting for. An entry a
        // truncation replaces is answered from the truncation, and a log
        // writer that fails answers everyone from the durability watcher
        if *self.inner.commit_tx.borrow() >= index {
            resolve_commit_waiters(&self.inner);
        }
        match tokio::time::timeout(self.inner.config.propose_timeout, rx).await {
            Ok(Ok(outcome)) => outcome,
            Ok(Err(_)) => Err(ZyronError::Internal(
                "the node stopped before the entry committed".into(),
            )),
            Err(_) => {
                self.inner.commit_waiters.lock().forget(index, ticket);
                Err(ZyronError::ConsensusTimeout {
                    operation: format!("commit of index {index}"),
                    elapsed_ms: self.inner.config.propose_timeout.as_millis() as u64,
                })
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
        if *self.inner.applied_tx.borrow() >= index {
            return Ok(());
        }
        let (ticket, rx) = self.inner.applied_waiters.lock().register(index, 0);
        resolve_applied_waiters(&self.inner);
        match tokio::time::timeout(self.inner.config.propose_timeout, rx).await {
            Ok(Ok(outcome)) => outcome,
            Ok(Err(_)) => Err(ZyronError::Internal(
                "the node stopped before the entry applied".into(),
            )),
            Err(_) => {
                self.inner.applied_waiters.lock().forget(index, ticket);
                Err(ZyronError::ConsensusTimeout {
                    operation: format!("apply of index {index}"),
                    elapsed_ms: self.inner.config.propose_timeout.as_millis() as u64,
                })
            }
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
        let index = self.propose(final_config).await?;
        // The entry commits on the old voters' say so, and the new voter
        // learns it is one only when it holds that entry. Returning before
        // then hands the caller a membership the node itself does not report
        self.wait_peer_holds(node_id, index).await;
        Ok(())
    }

    /// Waits, inside the rpc timeout, for a peer to confirm it holds an
    /// entry. A peer that does not answer in time is left to catch up on its
    /// own, the entry is committed either way
    async fn wait_peer_holds(&self, peer: NodeId, index: u64) {
        let deadline = Instant::now() + self.inner.config.rpc_timeout;
        loop {
            let held = self
                .inner
                .core
                .lock()
                .peer_match_index(peer)
                .unwrap_or(u64::MAX);
            if held >= index || Instant::now() >= deadline {
                return;
            }
            tokio::time::sleep(self.inner.config.tick_interval).await;
        }
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
        // Capturing the state is the machine's own copy or lock of whatever
        // it holds, and for a large machine that is real work, so it runs
        // beside the runtime rather than on a worker the ticker needs
        let machine = Arc::clone(&self.inner.machine);
        let mut source = tokio::task::spawn_blocking(move || machine.begin_checkpoint())
            .await
            .map_err(|e| ZyronError::Internal(format!("checkpoint capture task failed: {e}")))??;
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
    // A tick that lands long after the one before means the runtime held
    // this task up, and replies that arrived meanwhile are still queued.
    // The gap is logged because a leader judges its quorum on this loop and
    // a held-up loop is the one way a reachable quorum reads as absent
    let held_up_after = inner.config.election_timeout_min / 3;
    let mut last_tick = Instant::now();
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
        let gap = now.duration_since(last_tick);
        last_tick = now;
        if gap > held_up_after {
            tracing::info!(
                node = inner.id,
                gap_ms = gap.as_millis() as u64,
                "the tick loop was held up"
            );
        }
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

/// Publishes a commit index read under the consensus lock.
///
/// Monotone, so a value that was read before another and published after
/// it never rolls the advance back. The applier waits on this, and on a
/// follower it is published before the entries are on this node's own disk:
/// what the leader says is committed holds on a majority already, and
/// waiting for the local flush would put one more fsync in front of every
/// follower's apply
fn publish_commit_at(inner: &RaftInner, commit: u64) {
    let advanced = inner.commit_tx.send_if_modified(|current| {
        if commit > *current {
            *current = commit;
            true
        } else {
            false
        }
    });
    if advanced {
        // A follower learns what is committed only from the leader, so an
        // advance here is a message owed to every one of them
        wake_replicator(inner);
    }
}

/// Publishes the commit index and answers the proposals it reached, for a
/// caller that does not hold the consensus lock
fn publish_commit(inner: &Arc<RaftInner>) {
    let (commit, answers) = {
        let core = inner.core.lock();
        (core.commit_index(), take_committed_waiters(inner, &core))
    };
    publish_commit_at(inner, commit);
    answer(answers);
}

/// Answers the proposals whose entries the commit index has reached
fn resolve_commit_waiters(inner: &RaftInner) {
    let answers = {
        let core = inner.core.lock();
        take_committed_waiters(inner, &core)
    };
    answer(answers);
}

/// Takes out the proposals the commit index has reached, with their
/// verdicts.
///
/// Runs under the consensus lock because a verdict is the term at the
/// index, which is the one thing the watermark cannot say. An entry this
/// node proposed can be replaced by a new leader's, and a caller told its
/// write committed when a different entry now sits there would be told a
/// falsehood
fn take_committed_waiters(inner: &RaftInner, core: &RaftConsensus) -> Vec<Answer> {
    let mut waiters = inner.commit_waiters.lock();
    let commit = core.commit_index();
    match waiters.first_index() {
        Some(first) if first <= commit => {}
        _ => return Vec::new(),
    }
    let leader = core.leader_id();
    let log = core.state.log();
    waiters.take_up_to(commit, |index, term| match log.term_at(index) {
        // Compacted, which only happens after it applied
        None => Ok(()),
        Some(t) if t == term => Ok(()),
        Some(_) => Err(ZyronError::NotLeader { leader }),
    })
}

/// Takes out the proposals whose entries a truncation replaced or removed,
/// with the failure they are owed
fn take_overwritten_waiters(inner: &RaftInner, core: &RaftConsensus) -> Vec<Answer> {
    let mut waiters = inner.commit_waiters.lock();
    if waiters.first_index().is_none() {
        return Vec::new();
    }
    let leader = core.leader_id();
    let log = core.state.log();
    waiters.take_where(
        |index, term| !matches!(log.term_at(index), Some(t) if t == term),
        || ZyronError::NotLeader { leader },
    )
}

/// Answers the reads whose index the applier has reached
fn resolve_applied_waiters(inner: &RaftInner) {
    let answers = {
        let mut waiters = inner.applied_waiters.lock();
        let applied = *inner.applied_tx.borrow();
        waiters.take_up_to(applied, |_, _| Ok(()))
    };
    answer(answers);
}

fn publish_round_at(inner: &RaftInner, round: u64) {
    inner.round_tx.send_if_modified(|current| {
        if round > *current {
            *current = round;
            true
        } else {
            false
        }
    });
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
        let peers = inner.core.lock().leader_peers().to_vec();
        for peer in peers {
            loop {
                let work = {
                    let mut core = inner.core.lock();
                    core.build_peer_work(peer, Instant::now())
                };
                match work {
                    Ok(PeerWork::Idle) => break,
                    Ok(PeerWork::Append(request)) => dispatch_append(&inner, peer, request),
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
/// Abandoning rewinds the guess to what the follower confirmed, and a
/// follower that keeps missing this way ends up taking a snapshot instead.
/// The follower is held for the length of the read, so nothing built after
/// this batch is placed ahead of it
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
        Ok(Ok(entries)) => Some(entries),
        Ok(Err(e)) => {
            tracing::warn!(node = inner.id, peer, error = %e, "could not read log entries back for replication");
            None
        }
        Err(e) => {
            tracing::warn!(node = inner.id, peer, error = %e, "the log read back task did not finish");
            None
        }
    };
    match entries {
        Some(entries) => {
            request.entries = entries;
            dispatch_append(&inner, peer, request);
        }
        None => inner
            .core
            .lock()
            .handle_peer_unreachable(peer, Instant::now()),
    }
    inner.core.lock().finish_page_in(peer);
    wake_replicator(&inner);
}

/// Hands one append to the transport in the order it was built, and settles
/// its reply on a task of its own.
///
/// The hand-off runs on the replicator because the transport places a call's
/// frame at call time and the follower reads frames in the order they were
/// placed. A task per message that made the call itself would run in
/// whatever order the runtime chose, and a batch reaching the follower ahead
/// of the one before it is refused, which resets the pipeline and sends
/// everything after it again
fn dispatch_append(inner: &Arc<RaftInner>, peer: NodeId, request: AppendEntriesRequest) {
    let had_entries = !request.entries.is_empty();
    let call = inner.transport.send_append_entries(peer, request);
    let inner = Arc::clone(inner);
    tokio::spawn(async move {
        let outcome = call.await;
        settle_append(&inner, peer, had_entries, outcome);
    });
}

/// Takes one append's answer into the leader's view
fn settle_append(
    inner: &Arc<RaftInner>,
    peer: NodeId,
    had_entries: bool,
    outcome: std::result::Result<AppendEntriesReply, RaftRpcError>,
) {
    match outcome {
        Ok(reply) => {
            tracing::trace!(
                node = inner.id,
                peer,
                success = reply.success,
                match_index = reply.match_index,
                "append answered"
            );
            // Everything the reply changes is read under the one lock that
            // recorded it, and everything that has to be told is told after
            // the lock is released. Under load this runs for every message
            // to every follower, so a second acquisition here is one the
            // proposals queue behind
            let (owes_commit, commit, round, role, answers) = {
                let mut core = inner.core.lock();
                if let Err(e) = core.handle_append_reply(&reply, Instant::now()) {
                    tracing::warn!(node = inner.id, peer, error = %e, "could not record an append reply");
                }
                (
                    core.peer_owes_commit(peer),
                    core.commit_index(),
                    core.quorum_round(),
                    (core.role(), core.term(), core.leader_id()),
                    take_committed_waiters(&inner, &core),
                )
            };
            publish_commit_at(&inner, commit);
            publish_round_at(&inner, round);
            publish_role(&inner, role);
            answer(answers);
            // A reply carrying no entries still owes a wake when it drained the
            // last message to a follower the commit index has moved past.
            // The commit publish wakes only on the advance itself, which
            // happened while this message was still in flight
            if had_entries || !reply.success || owes_commit {
                wake_replicator(inner);
            }
        }
        Err(e) => {
            if !matches!(e, RaftRpcError::Shutdown) {
                tracing::debug!(node = inner.id, peer, error = %e, "append produced no answer");
            }
            let mut core = inner.core.lock();
            core.handle_peer_unreachable(peer, Instant::now());
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
        // A chunk is a file read, so it runs beside the runtime. A worker
        // held for a megabyte read a thousand times over is a worker the
        // ticker and the reply handlers do not have
        let reader = Arc::clone(inner);
        let (data, done) = tokio::task::spawn_blocking(move || {
            let mut store = reader.snapshots.lock();
            store.read_chunk(offset, chunk_bytes)
        })
        .await
        .map_err(|e| ZyronError::Internal(format!("snapshot read task failed: {e}")))??;
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
        inner.core.lock().note_peer_contact(peer, Instant::now());
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
        resolve_applied_waiters(inner);
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
    loop {
        let (commit, answers) = {
            let mut core = inner.core.lock();
            core.advance_commit();
            (core.commit_index(), take_committed_waiters(&inner, &core))
        };
        // A commit this node's own fsync produced is one no follower has been
        // told about: nothing was replied to and nothing was sent. The
        // publish wakes the replicator on the advance, so it goes out now
        // rather than on the next heartbeat, which would leave every follower
        // up to a heartbeat behind on a group that has just gone quiet
        publish_commit_at(&inner, commit);
        answer(answers);
        if let Some(err) = inner.log_writer.failure() {
            inner
                .commit_waiters
                .lock()
                .fail_all(|| ZyronError::WalWriteFailed(err.clone()));
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
        let (outcome, commit, role, overwritten, committed) = {
            let mut core = inner.core.lock();
            let before = core.metrics.log_truncations;
            let outcome = core.handle_append_entries(&req, Instant::now());
            let overwritten = if core.metrics.log_truncations != before {
                take_overwritten_waiters(&inner, &core)
            } else {
                Vec::new()
            };
            (
                outcome,
                core.commit_index(),
                (core.role(), core.term(), core.leader_id()),
                overwritten,
                take_committed_waiters(&inner, &core),
            )
        };
        // The commit index the leader sent is published now, ahead of this
        // node's own flush of the entries, so the applier is not held behind
        // the disk for entries a majority already holds
        publish_commit_at(&inner, commit);
        publish_role(&inner, role);
        answer(overwritten);
        answer(committed);
        Box::pin(async move {
            let reply = outcome?;
            if reply.success {
                wait_persisted(&inner, reply.match_index).await?;
            }
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

    fn on_timeout_now(
        &self,
        req: crate::election::TimeoutNowRequest,
    ) -> RaftHandlerFuture<'_, crate::election::TimeoutNowReply> {
        let inner = Arc::clone(&self.inner);
        let outcome = {
            let mut core = inner.core.lock();
            core.handle_timeout_now(&req, Instant::now())
        };
        Box::pin(async move {
            let (reply, requests) = outcome?;
            for (peer, request) in requests {
                tokio::spawn(send_vote(Arc::clone(&inner), peer, request));
            }
            let role = {
                let core = inner.core.lock();
                (core.role(), core.term(), core.leader_id())
            };
            publish_role(&inner, role);
            Ok(reply)
        })
    }

    fn on_read_index(&self, req: ReadIndexRequest) -> RaftHandlerFuture<'_, ReadIndexReply> {
        let inner = Arc::clone(&self.inner);
        // Both reads under one lock, so the index the answer promises and what
        // the asking follower has been told are the same instant
        let (result, owes_commit) = {
            let mut core = inner.core.lock();
            let result = core.handle_read_index(&req, Instant::now());
            (result, core.peer_owes_commit(req.from))
        };
        let timeout = inner.config.propose_timeout;
        Box::pin(async move {
            let (outcome, reply) = result?;
            if let ReadIndexOutcome::Pending { round, .. } = outcome {
                // The index is only safe to hand out once a majority has
                // echoed the round, so the answer waits rather than being sent
                // on a leadership this node has not re-confirmed
                wake_replicator(&inner);
                let node = RaftNode {
                    inner: Arc::clone(&inner),
                };
                node.wait_round(round, Instant::now() + timeout).await?;
            } else if owes_commit {
                // The follower asked because it is about to read locally, and
                // the commit index has moved past what it was last told. Only
                // a new message carries that advance, so send one now instead
                // of leaving the read to wait on the heartbeat timer
                wake_replicator(&inner);
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
    mut req: InstallSnapshotRequest,
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

    // A chunk is a file write, so it runs beside the runtime rather than on
    // a worker the election timer and the appends from the leader need
    let received = req.data.len() as u64;
    let data = std::mem::take(&mut req.data);
    let writer = Arc::clone(&inner);
    let (index, offset, done) = (req.last_included_index, req.offset, req.done);
    let written = tokio::task::spawn_blocking(move || {
        let store = writer.snapshots.lock();
        // The file only has to survive a crash once it is about to become
        // this node's snapshot
        store.write_chunk(index, offset, &data, done)
    })
    .await
    .map_err(|e| ZyronError::Internal(format!("snapshot write task failed: {e}")))??;
    inner
        .snapshot_bytes_received
        .fetch_add(received, Ordering::Relaxed);
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

    let (commit, committed, overwritten) = {
        let mut core = inner.core.lock();
        core.adopt_snapshot(
            meta.last_included_index,
            meta.last_included_term,
            meta.config.clone(),
        )?;
        (
            core.commit_index(),
            take_committed_waiters(&inner, &core),
            // The log restarts past the snapshot, so a proposal waiting on
            // an entry beyond it has nothing left to wait for
            take_overwritten_waiters(&inner, &core),
        )
    };
    publish_commit_at(&inner, commit);
    answer(committed);
    answer(overwritten);
    if *inner.applied_tx.borrow() < index {
        inner.applied_tx.send_replace(index);
        resolve_applied_waiters(&inner);
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

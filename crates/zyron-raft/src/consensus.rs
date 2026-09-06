//! The protocol itself, with no clock of its own and no sockets.
//!
//! Everything here is synchronous and takes `now` as an argument. That is
//! deliberate: consensus is a state machine over messages and time, and making
//! it a state machine in the code as well means it can be driven from a test
//! with an exact sequence of events, and driven in production by
//! [`crate::node`] with a real runtime. Nothing here blocks, allocates a task,
//! or waits on a peer.
//!
//! The one thing it does do is write. Raising a term and granting a vote both
//! fsync before returning, because both are promises to another node, and a
//! promise this node could forget across a restart is how two leaders appear
//! in one term. Those writes are rare: a healthy group makes none of them for
//! as long as its leader lives.
//!
//! ## Where a configuration takes effect
//!
//! When it is appended, not when it is committed. A node that waited for the
//! commit would be counting a quorum under the old configuration while the
//! new one was already deciding, which is the exact overlap joint consensus
//! exists to close. So [`Self::membership`] is derived from the log, and a
//! truncation that removes a configuration entry rebuilds it from the entries
//! that remain.

use std::sync::Arc;
use std::time::Instant;

use zyron_common::error::{Result, ZyronError};

use crate::NodeId;
use crate::config::RaftConfig;
use crate::election::{ElectionTimer, RequestVoteReply, RequestVoteRequest, log_is_up_to_date};
use crate::log::{LogPager, RaftCommand, RaftLogEntry, SliceItem};
use crate::membership::{ClusterConfig, Membership, NodeConfig};
use crate::replication::{
    AppendEntriesReply, AppendEntriesRequest, ReadIndexReply, ReadIndexRequest,
    next_index_after_reject,
};
use crate::state::{LeaderState, RaftRole, RaftState};

/// What one tick produced.
#[derive(Debug, Default)]
pub struct TickOutcome {
    /// Vote requests to fan out, one per voting peer, sent in parallel
    pub vote_requests: Vec<(NodeId, RequestVoteRequest)>,
    /// This node gave up the group during the tick
    pub stepped_down: bool,
    /// This node took the group during the tick, which happens inside a tick
    /// only in a group whose sole voter is this node
    pub became_leader: bool,
    /// Followers are due a message, so the replication loop should run
    pub replicate: bool,
}

/// What the leader has for one follower right now.
pub enum PeerWork {
    /// Nothing to send, and no heartbeat due
    Idle,
    /// Entries or a heartbeat
    Append(AppendEntriesRequest),
    /// Entries where some are no longer in memory.
    ///
    /// The plan is resolved into records by the replication task, off the
    /// consensus lock, because a page-in is a disk read and the lock it would
    /// otherwise hold is the one the heartbeat needs. The follower is held
    /// until [`RaftConsensus::finish_page_in`] says the batch has been handed
    /// to the transport, so nothing built after it is placed ahead of it
    AppendPaged {
        request: AppendEntriesRequest,
        plan: Vec<SliceItem>,
        pager: Arc<LogPager>,
    },
    /// The entries this follower needs are behind the snapshot point, so it
    /// gets the snapshot instead
    Snapshot,
}

/// The answer to a linearizable read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadIndexOutcome {
    /// The leader is inside its lease, so this index is already confirmed
    Ready(u64),
    /// A heartbeat round has been opened. The read may be answered once a
    /// majority has echoed `round`
    Pending { round: u64, index: u64 },
    /// This node does not lead, and here is who it thinks does
    NotLeader(Option<NodeId>),
    /// This node leads but has not yet committed an entry from its own term,
    /// so it does not know which inherited entries are committed
    NotReady,
}

/// Whether a snapshot chunk is worth writing down.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SnapshotDecision {
    /// Write it, the sender is a current leader
    Accept,
    /// Refuse, and tell the sender this term
    Reject(u64),
    /// Already covered by what this node holds, so the transfer is pointless
    AlreadyCovered,
}

/// Counters an operator reads to tell a healthy group from a busy one.
#[derive(Debug, Default, Clone)]
pub struct ConsensusMetrics {
    pub pre_votes_started: u64,
    pub elections_started: u64,
    pub elections_won: u64,
    pub step_downs: u64,
    pub appends_built: u64,
    /// Messages built with no entries because the heartbeat interval passed
    pub heartbeats_built: u64,
    /// Messages built with no entries to carry a commit index a follower
    /// with nothing in flight had not been told
    pub commit_carriers_built: u64,
    pub appends_rejected: u64,
    pub entries_appended: u64,
    pub entries_replicated: u64,
    /// Entries a replication batch had to read back from the log file because
    /// the residency cap had dropped them
    pub entries_paged_in: u64,
    /// Entries the apply loop had to read back from the log file, which
    /// happens when the applier is further behind than the residency cap
    /// keeps in memory
    pub entries_paged_for_apply: u64,
    pub log_truncations: u64,
    pub snapshots_installed: u64,
    pub read_index_leases_used: u64,
    pub read_index_rounds: u64,
}

/// One node's consensus state and every transition it can make.
pub struct RaftConsensus {
    pub id: NodeId,
    pub config: RaftConfig,
    pub state: RaftState,
    /// Derived from the log, so it is whatever the entries this node holds say
    membership: Membership,
    /// The configuration in force at the entry before the log starts, which is
    /// either the bootstrap configuration or the one a snapshot carried
    base_membership: Membership,
    timer: ElectionTimer,
    votes: Vec<NodeId>,
    pre_votes: Vec<NodeId>,
    pre_voting: bool,
    pre_vote_term: u64,
    /// When a leader last said anything to this node, which gates pre-vote
    last_leader_contact: Instant,
    /// Bumped whenever the effective configuration changes, so the driver can
    /// resync the transport directory without cloning the configuration on
    /// every tick
    membership_version: u64,
    pub metrics: ConsensusMetrics,
}

impl RaftConsensus {
    /// Builds the node's consensus state over an already opened [`RaftState`].
    ///
    /// `base` is the configuration in force before the first entry the log
    /// still holds. The effective configuration is then derived by replaying
    /// the configuration entries that survive in the log
    pub fn new(
        id: NodeId,
        config: RaftConfig,
        state: RaftState,
        base: ClusterConfig,
        now: Instant,
    ) -> Result<Self> {
        config.validate()?;
        base.validate()?;
        let timer = ElectionTimer::new(
            id,
            config.election_timeout_min,
            config.election_timeout_max,
            now,
        );
        let base_membership = Membership::Simple(base);
        let mut this = Self {
            id,
            config,
            state,
            membership: base_membership.clone(),
            base_membership,
            timer,
            votes: Vec::new(),
            pre_votes: Vec::new(),
            pre_voting: false,
            pre_vote_term: 0,
            last_leader_contact: now,
            membership_version: 0,
            metrics: ConsensusMetrics::default(),
        };
        this.recompute_membership();
        Ok(this)
    }

    // -----------------------------------------------------------------------
    // What this node currently is
    // -----------------------------------------------------------------------

    #[inline]
    pub fn role(&self) -> RaftRole {
        self.state.role
    }

    #[inline]
    pub fn term(&self) -> u64 {
        self.state.current_term()
    }

    #[inline]
    pub fn leader_id(&self) -> Option<NodeId> {
        self.state.leader_id
    }

    #[inline]
    pub fn is_leader(&self) -> bool {
        self.state.role == RaftRole::Leader
    }

    #[inline]
    pub fn commit_index(&self) -> u64 {
        self.state.volatile.commit_index
    }

    #[inline]
    pub fn last_applied(&self) -> u64 {
        self.state.volatile.last_applied
    }

    /// Recorded by the apply loop once entries have reached the state machine
    pub fn set_last_applied(&mut self, index: u64) {
        if index > self.state.volatile.last_applied {
            self.state.volatile.last_applied = index;
        }
    }

    #[inline]
    pub fn last_log_index(&self) -> u64 {
        self.state.log().last_index()
    }

    #[inline]
    pub fn last_log_term(&self) -> u64 {
        self.state.log().last_term()
    }

    pub fn membership(&self) -> &Membership {
        &self.membership
    }

    /// Changes whenever the effective configuration does
    #[inline]
    pub fn membership_version(&self) -> u64 {
        self.membership_version
    }

    /// The configuration to report to an operator
    pub fn cluster_config(&self) -> ClusterConfig {
        self.membership.effective().clone()
    }

    pub fn peers(&self) -> Vec<NodeId> {
        self.membership.peers(self.id)
    }

    /// The followers this leader replicates to, in the order its progress
    /// arrays hold them. Empty on a node that does not lead
    pub fn leader_peers(&self) -> &[NodeId] {
        self.state
            .leader
            .as_ref()
            .map(|leader| leader.peers.as_slice())
            .unwrap_or(&[])
    }

    /// Whether some follower can take entries now.
    ///
    /// A proposal wakes the replicator on this answer. Under load every
    /// follower has its share in flight and the answer is no, and the reply
    /// that frees a slot wakes the replicator instead, so the wake and the
    /// pass through the lock it costs are not paid at the proposal rate. A
    /// follower being probed has room for one message, and one that could
    /// not be reached has none until its retry time
    pub fn replication_has_room(&self, now: Instant) -> bool {
        let Some(leader) = self.state.leader.as_ref() else {
            return false;
        };
        let log_last = self.state.log().last_index();
        (0..leader.peers.len()).any(|i| {
            if leader.snapshot_in_flight[i] || leader.page_in_flight[i] {
                return false;
            }
            if leader.probing[i]
                && (leader.inflight[i] > 0 || leader.retry_at[i].is_some_and(|at| now < at))
            {
                return false;
            }
            let pending = entries_pending(leader.next_index[i], log_last);
            pending > 0 && send_window_open(&self.config, leader.inflight[i], pending)
        })
    }

    pub fn election_deadline(&self) -> Instant {
        self.timer.deadline()
    }

    /// Where a follower would send a write, for the error a client gets back
    pub fn not_leader(&self) -> ZyronError {
        ZyronError::NotLeader {
            leader: self.state.leader_id,
        }
    }

    // -----------------------------------------------------------------------
    // Time
    // -----------------------------------------------------------------------

    /// Advances every deadline this node holds.
    ///
    /// A follower that has not heard from a leader campaigns, a leader that
    /// has not heard from a majority steps down, and everything else is the
    /// replication loop's business
    pub fn tick(&mut self, now: Instant) -> TickOutcome {
        let mut outcome = TickOutcome::default();
        match self.state.role {
            RaftRole::Leader => {
                if self.config.check_quorum && !self.has_recent_quorum(now) {
                    // A leader that cannot reach a majority is a leader in
                    // name only. Standing down promptly is what stops a
                    // minority partition serving reads from a stale lease
                    tracing::info!(
                        node = self.id,
                        term = self.term(),
                        "stepping down, no majority reachable inside the election timeout"
                    );
                    self.become_follower(self.term(), None, now);
                    outcome.stepped_down = true;
                } else {
                    outcome.replicate = true;
                }
            }
            RaftRole::Follower | RaftRole::Candidate => {
                if self.timer.expired(now) && self.membership.is_voter(self.id) {
                    // A candidate whose own election timed out goes back to
                    // asking rather than raising its term again. Repeated
                    // split votes would otherwise inflate the term of every
                    // node in the group for no gain
                    if self.config.pre_vote {
                        outcome.vote_requests = self.start_pre_vote(now);
                    } else {
                        match self.become_candidate(now, false) {
                            Ok(requests) => outcome.vote_requests = requests,
                            Err(e) => {
                                tracing::error!(node = self.id, error = %e, "could not start an election");
                                self.timer.reset(now);
                            }
                        }
                    }
                    if self.is_leader() {
                        outcome.became_leader = true;
                        outcome.replicate = true;
                    }
                } else if self.timer.expired(now) {
                    // A learner never campaigns, so its timer is restarted
                    // rather than left expired and re-examined every tick
                    self.timer.reset(now);
                }
            }
        }
        outcome
    }

    fn has_recent_quorum(&self, now: Instant) -> bool {
        let Some(leader) = self.state.leader.as_ref() else {
            return false;
        };
        let window = self.config.election_timeout_min;
        self.membership.has_quorum(|id| {
            if id == self.id {
                return true;
            }
            match leader.pos(id) {
                Some(i) => now.duration_since(leader.last_contact[i]) < window,
                None => false,
            }
        })
    }

    // -----------------------------------------------------------------------
    // Elections
    // -----------------------------------------------------------------------

    /// Asks the group whether an election would succeed, without raising the
    /// term
    fn start_pre_vote(&mut self, now: Instant) -> Vec<(NodeId, RequestVoteRequest)> {
        self.metrics.pre_votes_started += 1;
        self.pre_voting = true;
        self.pre_vote_term = self.term() + 1;
        self.pre_votes.clear();
        self.pre_votes.push(self.id);
        self.timer.reset(now);
        self.state.leader_id = None;
        // Not campaigning yet, so this node is a follower until a majority
        // says an election would succeed
        self.state.role = RaftRole::Follower;
        self.state.leader = None;

        if self.membership.has_quorum(|id| id == self.id) {
            // A group whose only voter is this node has nobody to ask
            match self.become_candidate(now, false) {
                Ok(requests) => return requests,
                Err(e) => {
                    tracing::error!(node = self.id, error = %e, "could not start an election");
                    return Vec::new();
                }
            }
        }

        let request = RequestVoteRequest {
            term: self.pre_vote_term,
            candidate_id: self.id,
            last_log_index: self.last_log_index(),
            last_log_term: self.last_log_term(),
            pre_vote: true,
            transfer: false,
        };
        self.voting_peers()
            .into_iter()
            .map(|peer| (peer, request.clone()))
            .collect()
    }

    /// Raises the term, votes for itself, and asks for the rest. `transfer`
    /// marks an election the leader asked for, which voters grant even while
    /// they still hear that leader
    fn become_candidate(
        &mut self,
        now: Instant,
        transfer: bool,
    ) -> Result<Vec<(NodeId, RequestVoteRequest)>> {
        let term = self.term() + 1;
        self.state.persistent.start_term_voting_for(term, self.id)?;
        self.state.role = RaftRole::Candidate;
        self.state.leader = None;
        self.state.leader_id = None;
        self.pre_voting = false;
        self.votes.clear();
        self.votes.push(self.id);
        self.timer.reset(now);
        self.metrics.elections_started += 1;
        tracing::info!(node = self.id, term, "campaigning");

        if self.membership.has_quorum(|id| self.votes.contains(&id)) {
            self.become_leader(now)?;
            return Ok(Vec::new());
        }

        let request = RequestVoteRequest {
            term,
            candidate_id: self.id,
            last_log_index: self.last_log_index(),
            last_log_term: self.last_log_term(),
            pre_vote: false,
            transfer,
        };
        Ok(self
            .voting_peers()
            .into_iter()
            .map(|peer| (peer, request.clone()))
            .collect())
    }

    fn voting_peers(&self) -> Vec<NodeId> {
        self.membership
            .peers(self.id)
            .into_iter()
            .filter(|id| self.membership.is_voter(*id))
            .collect()
    }

    fn become_leader(&mut self, now: Instant) -> Result<()> {
        self.state.role = RaftRole::Leader;
        self.state.leader_id = Some(self.id);
        self.pre_voting = false;
        self.metrics.elections_won += 1;
        let mut leader = LeaderState::new(self.peers(), self.last_log_index(), now);

        // The no-op is appended before anything else this term. Until it
        // commits the leader cannot say which of the entries it inherited are
        // committed, so it cannot answer a linearizable read and cannot
        // advance the commit index past its predecessor's work
        let index = self.last_log_index() + 1;
        let term = self.term();
        leader.noop_index = index;
        self.state.leader = Some(leader);
        self.state
            .log_mut()
            .append(RaftLogEntry::new(term, index, RaftCommand::Noop))?;
        self.state.log_mut().flush_pending()?;
        self.metrics.entries_appended += 1;
        tracing::info!(node = self.id, term, noop = index, "took the group");
        self.advance_commit();
        Ok(())
    }

    fn become_follower(&mut self, term: u64, leader: Option<NodeId>, now: Instant) {
        if self.state.role == RaftRole::Leader {
            self.metrics.step_downs += 1;
        }
        self.state.role = RaftRole::Follower;
        self.state.leader = None;
        self.state.leader_id = leader;
        self.pre_voting = false;
        self.votes.clear();
        self.pre_votes.clear();
        if leader.is_some() {
            self.last_leader_contact = now;
        }
        self.timer.reset(now);
        let _ = term;
    }

    /// Steps down when a message carries a term above this node's.
    ///
    /// Returns whether the term moved, because a caller that was mid-election
    /// has to stop counting votes for a term it no longer holds
    fn observe_term(&mut self, term: u64, leader: Option<NodeId>, now: Instant) -> Result<bool> {
        if term <= self.term() {
            return Ok(false);
        }
        self.state.persistent.advance_term(term)?;
        self.become_follower(term, leader, now);
        Ok(true)
    }

    /// The voter best placed to take the group, for a leadership transfer.
    ///
    /// The follower with the most confirmed log needs the least catching up,
    /// so the handover is shortest through it. None when this node does not
    /// lead or has no voting peer
    pub fn transfer_target(&self) -> Option<NodeId> {
        if self.state.role != RaftRole::Leader {
            return None;
        }
        let leader = self.state.leader.as_ref()?;
        let mut best: Option<(NodeId, u64)> = None;
        for (i, peer) in leader.peers.iter().enumerate() {
            if !self.membership.is_voter(*peer) {
                continue;
            }
            let matched = leader.match_index[i];
            match best {
                Some((_, held)) if held >= matched => {}
                _ => best = Some((*peer, matched)),
            }
        }
        best.map(|(peer, _)| peer)
    }

    /// Whether a follower has confirmed everything this leader holds
    pub fn peer_caught_up(&self, peer: NodeId) -> bool {
        let Some(leader) = self.state.leader.as_ref() else {
            return false;
        };
        match leader.pos(peer) {
            Some(i) => leader.match_index[i] >= self.last_log_index(),
            None => false,
        }
    }

    /// What a peer has confirmed holding, None when this node does not lead
    /// or does not replicate to that peer
    pub fn peer_match_index(&self, peer: NodeId) -> Option<u64> {
        let leader = self.state.leader.as_ref()?;
        leader.pos(peer).map(|i| leader.match_index[i])
    }

    /// Takes a leader's request that this node campaign at once.
    ///
    /// The pre-vote is skipped. A leader asking to be replaced is the one
    /// node whose permission the pre-vote would otherwise seek. Returns the
    /// vote requests to fan out, none when the request was stale, this node
    /// already leads, or it cannot vote
    pub fn handle_timeout_now(
        &mut self,
        req: &crate::election::TimeoutNowRequest,
        now: Instant,
    ) -> Result<(
        crate::election::TimeoutNowReply,
        Vec<(NodeId, RequestVoteRequest)>,
    )> {
        let declined = |term: u64| {
            (
                crate::election::TimeoutNowReply {
                    term,
                    started: false,
                },
                Vec::new(),
            )
        };
        if req.term < self.term()
            || self.state.role == RaftRole::Leader
            || !self.membership.is_voter(self.id)
        {
            return Ok(declined(self.term()));
        }
        self.observe_term(req.term, Some(req.leader_id), now)?;
        let requests = self.become_candidate(now, true)?;
        tracing::info!(
            node = self.id,
            from = req.leader_id,
            term = self.term(),
            "campaigning at the leader's request"
        );
        Ok((
            crate::election::TimeoutNowReply {
                term: self.term(),
                started: true,
            },
            requests,
        ))
    }

    /// Answers a vote request, real or pre-vote.
    ///
    /// A granted real vote is on disk before this returns, because the reply
    /// is a promise not to vote for anyone else in this term
    pub fn handle_request_vote(
        &mut self,
        req: &RequestVoteRequest,
        now: Instant,
    ) -> Result<RequestVoteReply> {
        let mut reply = RequestVoteReply {
            term: self.term(),
            vote_granted: false,
            pre_vote: req.pre_vote,
            voter_id: self.id,
        };

        let up_to_date = log_is_up_to_date(
            req.last_log_term,
            req.last_log_index,
            self.last_log_term(),
            self.last_log_index(),
        );

        if req.pre_vote {
            // A node that can still hear a leader refuses to encourage anyone
            // to depose it. This is the whole of the disruption prevention:
            // the candidate never learns it could win, so it never raises its
            // term
            let heard_recently = self.state.role != RaftRole::Leader
                && now.duration_since(self.last_leader_contact) < self.config.election_timeout_min
                && self.state.leader_id.is_some();
            let leader_here = self.state.role == RaftRole::Leader;
            reply.vote_granted =
                req.term > self.term() && up_to_date && !heard_recently && !leader_here;
            return Ok(reply);
        }

        // A real vote from a candidate the pre-vote never sanctioned, one
        // that skipped it or won it while this node's leader was briefly
        // quiet, gets the same answer while the leader can still be heard,
        // and the term stays where it is so the candidate cannot depose a
        // working leader from outside the group. A node dropped from the
        // configuration that never learned it was dropped is the candidate
        // this refuses. The leader's own hand-off is the one exception
        if !req.transfer {
            let follower_hears_leader = self.state.role != RaftRole::Leader
                && self.state.leader_id.is_some()
                && now.duration_since(self.last_leader_contact) < self.config.election_timeout_min;
            let leader_with_quorum =
                self.state.role == RaftRole::Leader && self.has_recent_quorum(now);
            if follower_hears_leader || leader_with_quorum {
                return Ok(reply);
            }
        }

        if req.term < self.term() {
            return Ok(reply);
        }
        self.observe_term(req.term, None, now)?;
        reply.term = self.term();

        let free_to_vote = match self.state.voted_for() {
            None => true,
            Some(who) => who == req.candidate_id,
        };
        if free_to_vote && up_to_date {
            self.state.persistent.record_vote(req.candidate_id)?;
            self.timer.reset(now);
            reply.vote_granted = true;
        }
        Ok(reply)
    }

    /// Counts one vote answer, and promotes a won pre-vote into a real
    /// election
    pub fn handle_request_vote_reply(
        &mut self,
        reply: &RequestVoteReply,
        now: Instant,
    ) -> Result<Vec<(NodeId, RequestVoteRequest)>> {
        if reply.pre_vote {
            if !self.pre_voting {
                return Ok(Vec::new());
            }
            if !reply.vote_granted {
                // A refusal from a node ahead of this one is how a node that
                // was partitioned learns the real term without an election
                if self.observe_term(reply.term, None, now)? {
                    self.pre_voting = false;
                }
                return Ok(Vec::new());
            }
            if !self.pre_votes.contains(&reply.voter_id) {
                self.pre_votes.push(reply.voter_id);
            }
            if self
                .membership
                .has_quorum(|id| self.pre_votes.contains(&id))
            {
                return self.become_candidate(now, false);
            }
            return Ok(Vec::new());
        }

        if self.observe_term(reply.term, None, now)? {
            return Ok(Vec::new());
        }
        if self.state.role != RaftRole::Candidate || reply.term != self.term() {
            return Ok(Vec::new());
        }
        if reply.vote_granted && !self.votes.contains(&reply.voter_id) {
            self.votes.push(reply.voter_id);
        }
        if self.membership.has_quorum(|id| self.votes.contains(&id)) {
            self.become_leader(now)?;
        }
        Ok(Vec::new())
    }

    // -----------------------------------------------------------------------
    // Replication, follower side
    // -----------------------------------------------------------------------

    /// Runs the consistency check and takes whatever survives it.
    ///
    /// The reply is not safe to send until the log has reached
    /// `reply.match_index` on disk. The caller owns that wait, because it is
    /// the only place that can do it without holding the consensus lock across
    /// an fsync
    pub fn handle_append_entries(
        &mut self,
        req: &AppendEntriesRequest,
        now: Instant,
    ) -> Result<AppendEntriesReply> {
        let mut reply = AppendEntriesReply {
            term: self.term(),
            success: false,
            match_index: 0,
            hint_index: 0,
            read_round: req.read_round,
            follower_id: self.id,
        };

        if req.term < self.term() {
            reply.hint_index = self.last_log_index();
            return Ok(reply);
        }

        self.observe_term(req.term, Some(req.leader_id), now)?;
        reply.term = self.term();

        // A leader at this term exists, so whatever this node was doing stops
        if self.state.role != RaftRole::Follower {
            self.become_follower(req.term, Some(req.leader_id), now);
        }
        self.state.leader_id = Some(req.leader_id);
        self.last_leader_contact = now;
        self.timer.reset(now);

        let log_prev = self.state.log().prev_index();
        if req.prev_log_index < log_prev {
            // This node is past the leader's guess because a snapshot covers
            // it. Point the leader at the snapshot boundary, which is the
            // first position this node can still check
            reply.hint_index = log_prev;
            self.metrics.appends_rejected += 1;
            return Ok(reply);
        }
        match self.state.log().term_at(req.prev_log_index) {
            Some(term) if term == req.prev_log_term => {}
            Some(term) => {
                // A different entry sits here. Point past the whole of the
                // conflicting term rather than one entry back, because every
                // entry of that term is equally wrong
                reply.hint_index = self.first_index_of_term(req.prev_log_index, term) - 1;
                self.metrics.appends_rejected += 1;
                return Ok(reply);
            }
            None => {
                reply.hint_index = self.last_log_index();
                self.metrics.appends_rejected += 1;
                return Ok(reply);
            }
        }

        let mut index = req.prev_log_index;
        let mut membership_dirty = false;
        for entry in &req.entries {
            index += 1;
            match self.state.log().term_at(index) {
                // Already held, byte for byte, so nothing to do
                Some(term) if term == entry.term => continue,
                Some(_) => {
                    self.state.log_mut().truncate_after(index - 1)?;
                    self.metrics.log_truncations += 1;
                    membership_dirty = true;
                }
                None => {}
            }
            if entry.command.is_config_change() {
                membership_dirty = true;
            }
            self.state
                .log_mut()
                .append_shared(std::sync::Arc::clone(entry))?;
            self.metrics.entries_appended += 1;
        }
        self.state.log_mut().flush_pending()?;
        if membership_dirty {
            self.recompute_membership();
        }

        let last_new = req.prev_log_index + req.entries.len() as u64;
        if req.leader_commit > self.state.volatile.commit_index {
            self.state.volatile.commit_index = req.leader_commit.min(last_new);
        }
        tracing::trace!(
            node = self.id,
            from = req.leader_id,
            prev = req.prev_log_index,
            entries = req.entries.len(),
            leader_commit = req.leader_commit,
            commit = self.state.volatile.commit_index,
            "append taken"
        );
        reply.success = true;
        reply.match_index = last_new;
        Ok(reply)
    }

    /// Walks back to the first entry of the term sitting at `index`, so a
    /// rejection can skip the whole run in one message
    fn first_index_of_term(&self, index: u64, term: u64) -> u64 {
        let log = self.state.log();
        let floor = log.first_index();
        let mut at = index;
        while at > floor {
            match log.term_at(at - 1) {
                Some(t) if t == term => at -= 1,
                _ => break,
            }
        }
        at
    }

    // -----------------------------------------------------------------------
    // Replication, leader side
    // -----------------------------------------------------------------------

    /// Builds whatever this follower is owed, and books it as in flight.
    ///
    /// `next_index` moves forward on send rather than on reply, which is what
    /// makes the next call produce the following batch instead of the same
    /// one. A reply that refuses the batch resets the guess, so the optimism
    /// costs a round trip in the rare case and buys a pipeline in the common
    /// one
    pub fn build_peer_work(&mut self, peer: NodeId, now: Instant) -> Result<PeerWork> {
        if self.state.role != RaftRole::Leader {
            return Ok(PeerWork::Idle);
        }
        let (max_entries, max_bytes, heartbeat) = (
            self.config.max_batch_entries,
            self.config.max_batch_bytes,
            self.config.heartbeat_interval,
        );
        let log_prev = self.state.log().prev_index();
        let log_last = self.state.log().last_index();
        let commit = self.state.volatile.commit_index;
        let term = self.state.current_term();
        let id = self.id;

        let Some(leader) = self.state.leader.as_mut() else {
            return Ok(PeerWork::Idle);
        };
        let Some(i) = leader.pos(peer) else {
            return Ok(PeerWork::Idle);
        };
        if leader.snapshot_in_flight[i] || leader.page_in_flight[i] {
            return Ok(PeerWork::Idle);
        }
        let next = leader.next_index[i];
        if next <= log_prev {
            leader.snapshot_in_flight[i] = true;
            return Ok(PeerWork::Snapshot);
        }
        // A follower that could not be reached is left alone until its retry
        // time, heartbeats included, so a follower that is down costs one
        // attempt per interval rather than one per proposal
        if leader.probing[i] && leader.retry_at[i].is_some_and(|at| now < at) {
            return Ok(PeerWork::Idle);
        }
        let heartbeat_due = now >= leader.next_heartbeat[i];
        let pending = entries_pending(next, log_last);
        // A commit index is due when the follower holds entries it has not
        // been told are committed. What it can act on is bounded by what it
        // has confirmed holding, because a message with no entries attaches
        // at that point and the follower commits no further than it. So the
        // comparison is against that bound, and a message still in flight
        // does not cover an advance that landed after it was built. Entries
        // waiting to go carry the commit index with them, so it costs a
        // message of its own only when there are none
        let actionable_commit = commit.min(leader.match_index[i]);
        let commit_due = pending == 0 && actionable_commit > leader.sent_commit[i];
        // Until the follower has confirmed a position, entries go one
        // message at a time. Heartbeats still go on their timer, so a probe
        // that is slow to be answered does not silence the leader
        let probe_open = !leader.probing[i] || leader.inflight[i] == 0;
        let can_send_entries = pending > 0
            && probe_open
            && send_window_open(&self.config, leader.inflight[i], pending);
        if !heartbeat_due && !commit_due && !can_send_entries {
            return Ok(PeerWork::Idle);
        }

        let read_round = leader.heartbeat_round;
        let send_entries = can_send_entries;
        // A message carrying entries attaches at the optimistic guess, which
        // is where the batch before it ends. A message carrying none attaches
        // at what the follower has actually confirmed, because the guess may
        // be ahead of anything the follower holds and a heartbeat that is
        // refused for that reason would rewind the whole pipeline
        let prev_log_index = if send_entries {
            next - 1
        } else {
            leader.match_index[i].max(log_prev)
        };
        leader.inflight[i] += 1;
        leader.next_heartbeat[i] = now + heartbeat;

        let plan = if send_entries {
            let plan = self.state.log().plan_slice(next, max_entries, max_bytes);
            if let Some(leader) = self.state.leader.as_mut() {
                leader.next_index[i] = next + plan.len() as u64;
            }
            plan
        } else {
            Vec::new()
        };
        // What this message lets the follower commit, the leader's commit
        // index capped at the last entry the message leaves it holding. A
        // commit past that is still owed and a later message carries it
        if let Some(leader) = self.state.leader.as_mut() {
            leader.sent_commit[i] = commit.min(prev_log_index + plan.len() as u64);
        }
        let Some(prev_log_term) = self.state.log().term_at(prev_log_index) else {
            // The entry the check needs was compacted between the two reads
            if let Some(leader) = self.state.leader.as_mut() {
                leader.inflight[i] = leader.inflight[i].saturating_sub(1);
                leader.snapshot_in_flight[i] = true;
                leader.sent_commit[i] = 0;
            }
            return Ok(PeerWork::Snapshot);
        };

        self.metrics.appends_built += 1;
        self.metrics.entries_replicated += plan.len() as u64;
        if !send_entries {
            if commit_due {
                self.metrics.commit_carriers_built += 1;
            } else {
                self.metrics.heartbeats_built += 1;
            }
        }
        tracing::trace!(
            node = id,
            peer,
            prev_log_index,
            entries = plan.len(),
            commit,
            heartbeat_due,
            commit_due,
            "append built"
        );

        // A batch entirely in memory becomes the request here. One that
        // crosses the residency boundary is handed out as a plan, so the reads
        // it needs happen on the replication task rather than under this lock
        let paged = plan.iter().any(|item| matches!(item, SliceItem::Paged(_)));
        if !paged {
            let entries = plan
                .into_iter()
                .map(|item| match item {
                    SliceItem::Resident(entry) => entry,
                    SliceItem::Paged(_) => unreachable!("checked above"),
                })
                .collect();
            return Ok(PeerWork::Append(AppendEntriesRequest {
                term,
                leader_id: id,
                prev_log_index,
                prev_log_term,
                entries,
                leader_commit: commit,
                read_round,
            }));
        }
        self.metrics.entries_paged_in += plan
            .iter()
            .filter(|item| matches!(item, SliceItem::Paged(_)))
            .count() as u64;
        if let Some(leader) = self.state.leader.as_mut() {
            leader.page_in_flight[i] = true;
        }
        Ok(PeerWork::AppendPaged {
            request: AppendEntriesRequest {
                term,
                leader_id: id,
                prev_log_index,
                prev_log_term,
                entries: Vec::new(),
                leader_commit: commit,
                read_round,
            },
            plan,
            pager: self.state.log().pager(),
        })
    }

    /// The lowest index every member of the group has confirmed holding.
    ///
    /// Compacting past this would leave a member asking for entries that no
    /// longer exist, and the only way back for it would be a whole copy of
    /// somebody's data. So the log is kept for as long as the slowest member
    /// still needs it, up to whatever retention the operator allows.
    ///
    /// A node that leads nothing reports its own applied index, because it has
    /// no view of anyone else and its own log is all it can reason about
    pub fn group_match_index(&self) -> u64 {
        let Some(leader) = self.state.leader.as_ref() else {
            return self.state.volatile.last_applied;
        };
        let mut lowest = self.state.log().last_index();
        for i in 0..leader.peers.len() {
            lowest = lowest.min(leader.match_index[i]);
        }
        lowest
    }

    /// Takes a follower's answer into the leader's view
    pub fn handle_append_reply(&mut self, reply: &AppendEntriesReply, now: Instant) -> Result<()> {
        if self.observe_term(reply.term, None, now)? {
            return Ok(());
        }
        if self.state.role != RaftRole::Leader || reply.term != self.term() {
            return Ok(());
        }
        let Some(leader) = self.state.leader.as_mut() else {
            return Ok(());
        };
        let Some(i) = leader.pos(reply.follower_id) else {
            return Ok(());
        };
        leader.inflight[i] = leader.inflight[i].saturating_sub(1);
        leader.last_contact[i] = now;
        // Any reply proves the follower can be reached
        leader.retry_at[i] = None;

        if reply.success {
            // A confirmed position is what opens the pipeline
            leader.probing[i] = false;
            if reply.match_index > leader.match_index[i] {
                leader.match_index[i] = reply.match_index;
            }
            if leader.next_index[i] < leader.match_index[i] + 1 {
                leader.next_index[i] = leader.match_index[i] + 1;
            }
            if reply.read_round > leader.acked_round[i] {
                leader.acked_round[i] = reply.read_round;
            }
            self.refresh_quorum_round(now);
            self.advance_commit();
        } else {
            leader.next_index[i] = next_index_after_reject(reply.hint_index, leader.match_index[i]);
            // Whatever else was in flight to this follower will be refused for
            // the same reason, so the guess is rebuilt from here rather than
            // stepped once per stale reply, and sent one message at a time
            // until the follower confirms it
            leader.inflight[i] = 0;
            leader.probing[i] = true;
            // A refused message delivered no commit index either
            leader.sent_commit[i] = 0;
        }
        Ok(())
    }

    /// Whether `peer` holds committed entries it has not been told are
    /// committed.
    ///
    /// A message in flight is built with the commit index of the moment it
    /// was built and attaches at what the follower had confirmed then, so an
    /// advance that lands after it, or entries the follower confirms after
    /// it, are not covered by it. A follower that has not been told cannot
    /// apply, and no further message is due until the heartbeat timer, so a
    /// linearizable read against it would wait out the heartbeat interval
    /// for no reason. The answer is taken from what the follower has
    /// confirmed holding rather than from the pipeline being empty, because
    /// a message built while the pipeline was briefly empty and attached at
    /// an older confirmation delivers less than the commit index it carries
    pub fn peer_owes_commit(&self, peer: NodeId) -> bool {
        if self.state.role != RaftRole::Leader {
            return false;
        }
        let Some(leader) = self.state.leader.as_ref() else {
            return false;
        };
        let Some(i) = leader.pos(peer) else {
            return false;
        };
        self.commit_index().min(leader.match_index[i]) > leader.sent_commit[i]
    }

    /// Steps down if a peer's reply carried a later term.
    ///
    /// Used by the snapshot sender, which is the one reply path that is not an
    /// AppendEntries and still learns about a newer leader
    pub fn observe_peer_term(&mut self, term: u64, now: Instant) -> Result<bool> {
        self.observe_term(term, None, now)
    }

    /// Gives up the group without waiting for a timeout.
    ///
    /// Used when this node is the one being removed from the configuration:
    /// carrying on as leader of a group it is no longer in would keep a node
    /// outside the membership deciding what commits
    pub fn step_down(&mut self, now: Instant) {
        if self.state.role == RaftRole::Leader {
            self.become_follower(self.term(), None, now);
        }
    }

    /// Marks a call to a peer as having produced nothing, so the pipeline slot
    /// is returned, the guess falls back to what the peer confirmed, and the
    /// peer is left alone for a heartbeat interval before it is tried again
    pub fn handle_peer_unreachable(&mut self, peer: NodeId, now: Instant) {
        let heartbeat = self.config.heartbeat_interval;
        let Some(leader) = self.state.leader.as_mut() else {
            return;
        };
        if let Some(i) = leader.pos(peer) {
            leader.reset_progress(i);
            leader.retry_at[i] = Some(now + heartbeat);
        }
    }

    /// Clears the snapshot flag once a transfer ends, either way
    /// Records that a peer answered, whatever it answered. A follower taking
    /// a snapshot chunk by chunk answers nothing else for the length of the
    /// transfer, and without this it would read as unreachable to the quorum
    /// check the moment it became a voter
    pub fn note_peer_contact(&mut self, peer: NodeId, now: Instant) {
        let Some(leader) = self.state.leader.as_mut() else {
            return;
        };
        if let Some(i) = leader.pos(peer) {
            leader.last_contact[i] = now;
        }
    }

    /// Releases a follower held for a batch read back from the log file,
    /// once that batch has been handed to the transport or given up on
    pub fn finish_page_in(&mut self, peer: NodeId) {
        let Some(leader) = self.state.leader.as_mut() else {
            return;
        };
        if let Some(i) = leader.pos(peer) {
            leader.page_in_flight[i] = false;
        }
    }

    pub fn finish_snapshot_transfer(&mut self, peer: NodeId, delivered_through: Option<u64>) {
        let Some(leader) = self.state.leader.as_mut() else {
            return;
        };
        let Some(i) = leader.pos(peer) else {
            return;
        };
        leader.snapshot_in_flight[i] = false;
        if let Some(index) = delivered_through {
            if index > leader.match_index[i] {
                leader.match_index[i] = index;
            }
            leader.next_index[i] = leader.match_index[i] + 1;
        }
        leader.inflight[i] = 0;
        // The snapshot said nothing about the log past it, so the first
        // append after it is a probe
        leader.probing[i] = true;
    }

    /// Recomputes how far a majority has reached, and moves the commit index
    /// there when the entry is from this term.
    ///
    /// The current-term condition is not an optimization. An entry from an
    /// earlier term can sit on a majority and still not be committed, because
    /// a later leader with a shorter log could overwrite it. Only once an
    /// entry from this leader's own term is on a majority is everything before
    /// it pinned as well
    pub fn advance_commit(&mut self) {
        if self.state.role != RaftRole::Leader {
            return;
        }
        let self_id = self.id;
        // The leader's own contribution is what is on its disk, not what is in
        // its memory. Counting an unsynced append would let a majority form
        // out of two followers and a leader that could still lose the entry
        let self_match = self.state.log().persisted_index();
        let candidate = {
            let Some(leader) = self.state.leader.as_ref() else {
                return;
            };
            self.membership.quorum_match_index(|id| {
                if id == self_id {
                    self_match
                } else {
                    leader.match_of(id)
                }
            })
        };
        if candidate <= self.state.volatile.commit_index {
            return;
        }
        if self.state.log().term_at(candidate) != Some(self.state.current_term()) {
            return;
        }
        self.state.volatile.commit_index = candidate;
    }

    fn refresh_quorum_round(&mut self, now: Instant) {
        let self_id = self.id;
        let round = {
            let Some(leader) = self.state.leader.as_ref() else {
                return;
            };
            let current = leader.heartbeat_round;
            self.membership.quorum_match_index(|id| {
                if id == self_id {
                    current
                } else {
                    match leader.pos(id) {
                        Some(i) => leader.acked_round[i],
                        None => 0,
                    }
                }
            })
        };
        if let Some(leader) = self.state.leader.as_mut() {
            if round > leader.quorum_round {
                leader.quorum_round = round;
                leader.quorum_round_at = now;
            }
        }
    }

    /// The highest round a majority has echoed, for the read path to wait on
    pub fn quorum_round(&self) -> u64 {
        self.state
            .leader
            .as_ref()
            .map(|l| l.quorum_round)
            .unwrap_or(0)
    }

    // -----------------------------------------------------------------------
    // Proposals
    // -----------------------------------------------------------------------

    /// Appends one command, returning where it landed
    pub fn propose(&mut self, command: RaftCommand) -> Result<(u64, u64)> {
        self.propose_many(vec![command])
    }

    /// Appends a run of commands under one flush, which is what turns a burst
    /// of client writes into a single fsync
    pub fn propose_many(&mut self, commands: Vec<RaftCommand>) -> Result<(u64, u64)> {
        if self.state.role != RaftRole::Leader {
            return Err(self.not_leader());
        }
        if commands.is_empty() {
            return Ok((self.last_log_index(), self.term()));
        }
        let term = self.term();
        let mut membership_dirty = false;
        for command in commands {
            if command.is_config_change() {
                self.check_config_change_allowed(&command)?;
                membership_dirty = true;
            }
            let index = self.last_log_index() + 1;
            self.state
                .log_mut()
                .append(RaftLogEntry::new(term, index, command))?;
            self.metrics.entries_appended += 1;
            if membership_dirty {
                // Applied at once, because a later entry in this same batch
                // must be checked against the configuration this one creates
                self.recompute_membership();
                self.reconcile_leader_peers();
            }
        }
        self.state.log_mut().flush_pending()?;
        self.advance_commit();
        Ok((self.last_log_index(), term))
    }

    /// Refuses a configuration change that would overlap another one.
    ///
    /// Two voter changes in flight at once can produce a pair of
    /// configurations with no quorum in common, which is the split brain joint
    /// consensus exists to prevent. One at a time, and only once the previous
    /// one has committed
    fn check_config_change_allowed(&self, command: &RaftCommand) -> Result<()> {
        // The list is ascending, so the newest configuration entry is the
        // last one and it alone decides whether a change is still uncommitted
        let pending = self
            .state
            .log()
            .config_change_indexes()
            .last()
            .is_some_and(|&i| i > self.state.volatile.commit_index);
        if pending {
            return Err(ZyronError::Internal(
                "a configuration change is already uncommitted, only one may be in flight".into(),
            ));
        }
        match command {
            RaftCommand::JointConfig { .. } => {
                if self.membership.is_joint() {
                    return Err(ZyronError::Internal(
                        "the group is already in a joint configuration".into(),
                    ));
                }
            }
            RaftCommand::FinalConfig { .. } => {
                if !self.membership.is_joint() {
                    return Err(ZyronError::Internal(
                        "the group is not in a joint configuration to leave".into(),
                    ));
                }
            }
            RaftCommand::AddNode { .. } | RaftCommand::RemoveNode { .. } => {
                if self.membership.is_joint() {
                    return Err(ZyronError::Internal(
                        "membership cannot change while a joint configuration is in force".into(),
                    ));
                }
            }
            _ => {}
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Membership
    // -----------------------------------------------------------------------

    /// Rebuilds the effective configuration from the base plus every
    /// configuration entry still in the log
    fn recompute_membership(&mut self) {
        let mut m = self.base_membership.clone();
        let log = self.state.log();
        // Only the configuration entries matter, and the log tracks where
        // they are, so a truncation on a large retained log does not walk it
        for &index in log.config_change_indexes() {
            if let Some(entry) = log.entry(index) {
                apply_config_command(&mut m, &entry.command);
            }
        }
        if m != self.membership {
            self.membership = m;
            self.membership_version += 1;
        }
        self.reconcile_leader_peers();
    }

    fn reconcile_leader_peers(&mut self) {
        let peers = self.membership.peers(self.id);
        let last = self.state.log().last_index();
        let now = Instant::now();
        if let Some(leader) = self.state.leader.as_mut() {
            leader.reconcile(peers, last, now);
        }
    }

    /// The entry that brings a node in as a learner
    pub fn add_node_command(&self, node_id: NodeId, address: &str) -> Result<RaftCommand> {
        if self.membership.contains(node_id) {
            return Err(ZyronError::Internal(format!(
                "node {node_id} is already in the group"
            )));
        }
        Ok(RaftCommand::AddNode {
            node_id,
            address: address.to_string(),
        })
    }

    /// The joint entry that promotes a caught-up learner to a voter
    pub fn promote_command(&self, node_id: NodeId) -> Result<RaftCommand> {
        let current = self.membership.effective();
        let Some(node) = current.get(node_id) else {
            return Err(ZyronError::Internal(format!(
                "node {node_id} is not in the group"
            )));
        };
        if node.is_voter {
            return Err(ZyronError::Internal(format!(
                "node {node_id} already votes"
            )));
        }
        let new = current.promoted(node_id);
        new.validate()?;
        Ok(RaftCommand::JointConfig {
            old: current.clone(),
            new,
        })
    }

    /// The entry that starts removing a node.
    ///
    /// A learner leaves in one entry because it is in no quorum. A voter
    /// leaves through a joint configuration, because removing it changes what
    /// a majority is
    pub fn remove_node_command(&self, node_id: NodeId) -> Result<RaftCommand> {
        let current = self.membership.effective();
        let Some(node) = current.get(node_id) else {
            return Err(ZyronError::Internal(format!(
                "node {node_id} is not in the group"
            )));
        };
        if !node.is_voter {
            return Ok(RaftCommand::RemoveNode { node_id });
        }
        let new = current.without_node(node_id);
        new.validate()?;
        Ok(RaftCommand::JointConfig {
            old: current.clone(),
            new,
        })
    }

    /// The entry that leaves a joint configuration for the incoming one alone
    pub fn leave_joint_command(&self) -> Result<RaftCommand> {
        match &self.membership {
            Membership::Joint { new, .. } => Ok(RaftCommand::FinalConfig {
                config: new.clone(),
            }),
            Membership::Simple(_) => Err(ZyronError::Internal(
                "the group is not in a joint configuration".into(),
            )),
        }
    }

    /// Whether a learner is close enough to the leader to be promoted without
    /// stalling the group.
    ///
    /// Promotion makes the node count toward every quorum from that moment. A
    /// learner still thousands of entries behind would hold up commits until
    /// it caught up, so the leader waits
    pub fn learner_is_caught_up(&self, node_id: NodeId, tolerance: u64) -> bool {
        let Some(leader) = self.state.leader.as_ref() else {
            return false;
        };
        let last = self.state.log().last_index();
        leader.match_of(node_id) + tolerance >= last
    }

    // -----------------------------------------------------------------------
    // Snapshots
    // -----------------------------------------------------------------------

    /// Whether an incoming snapshot chunk should be written
    pub fn check_install_snapshot(
        &mut self,
        term: u64,
        leader_id: NodeId,
        last_included_index: u64,
        now: Instant,
    ) -> Result<SnapshotDecision> {
        if term < self.term() {
            return Ok(SnapshotDecision::Reject(self.term()));
        }
        self.observe_term(term, Some(leader_id), now)?;
        if self.state.role != RaftRole::Follower {
            self.become_follower(term, Some(leader_id), now);
        }
        self.state.leader_id = Some(leader_id);
        self.last_leader_contact = now;
        self.timer.reset(now);

        if last_included_index <= self.state.volatile.commit_index {
            return Ok(SnapshotDecision::AlreadyCovered);
        }
        Ok(SnapshotDecision::Accept)
    }

    /// Adopts a fully received snapshot: the log restarts past it and the
    /// configuration it carried becomes the base
    pub fn adopt_snapshot(
        &mut self,
        last_included_index: u64,
        last_included_term: u64,
        config: ClusterConfig,
    ) -> Result<()> {
        self.state
            .log_mut()
            .reset_to(last_included_index, last_included_term)?;
        if last_included_index > self.state.volatile.commit_index {
            self.state.volatile.commit_index = last_included_index;
        }
        if last_included_index > self.state.volatile.last_applied {
            self.state.volatile.last_applied = last_included_index;
        }
        self.base_membership = Membership::Simple(config);
        self.recompute_membership();
        self.metrics.snapshots_installed += 1;
        Ok(())
    }

    /// Discards the log prefix a snapshot has made redundant, carrying the
    /// configuration at that point into the base
    pub fn compact_log(&mut self, index: u64, term: u64) -> Result<()> {
        let log_first = self.state.log().first_index();
        if index < log_first {
            return Ok(());
        }
        let mut base = self.base_membership.clone();
        let upper = index.min(self.state.log().last_index());
        for &i in self.state.log().config_change_indexes() {
            if i < log_first || i > upper {
                continue;
            }
            if let Some(entry) = self.state.log().entry(i) {
                apply_config_command(&mut base, &entry.command);
            }
        }
        self.base_membership = base;
        self.state.log_mut().compact_to(index, term)
    }

    /// The configuration to ship with a snapshot
    pub fn snapshot_config(&self) -> ClusterConfig {
        self.membership.effective().clone()
    }

    // -----------------------------------------------------------------------
    // Linearizable reads
    // -----------------------------------------------------------------------

    /// Decides what a read must wait for.
    ///
    /// Inside the lease this costs nothing: the leader confirmed a majority
    /// recently enough that no other node could have been elected since. Past
    /// the lease it opens a heartbeat round, and the read waits for a majority
    /// to echo it
    pub fn begin_read_index(&mut self, now: Instant) -> ReadIndexOutcome {
        if self.state.role != RaftRole::Leader {
            return ReadIndexOutcome::NotLeader(self.state.leader_id);
        }
        let noop = self
            .state
            .leader
            .as_ref()
            .map(|l| l.noop_index)
            .unwrap_or(u64::MAX);
        if self.state.volatile.commit_index < noop {
            return ReadIndexOutcome::NotReady;
        }
        let index = self.state.volatile.commit_index;
        let lease = self.config.leader_lease;
        if let Some(leader) = self.state.leader.as_ref() {
            if leader.quorum_round > 0 && now.duration_since(leader.quorum_round_at) < lease {
                self.metrics.read_index_leases_used += 1;
                return ReadIndexOutcome::Ready(index);
            }
        }
        let round = {
            let Some(leader) = self.state.leader.as_mut() else {
                return ReadIndexOutcome::NotLeader(None);
            };
            leader.heartbeat_round += 1;
            // Every follower is due a heartbeat now, so the round goes out
            // without waiting for the next interval
            for slot in leader.next_heartbeat.iter_mut() {
                *slot = now;
            }
            leader.heartbeat_round
        };
        self.metrics.read_index_rounds += 1;
        // A group whose only voter is this node satisfies the round at once
        self.refresh_quorum_round(now);
        if let Some(leader) = self.state.leader.as_ref() {
            if leader.quorum_round >= round {
                return ReadIndexOutcome::Ready(index);
            }
        }
        ReadIndexOutcome::Pending { round, index }
    }

    /// Answers a follower asking where it must have applied to
    pub fn handle_read_index(
        &mut self,
        req: &ReadIndexRequest,
        now: Instant,
    ) -> Result<(ReadIndexOutcome, ReadIndexReply)> {
        self.observe_term(req.term, None, now)?;
        let outcome = self.begin_read_index(now);
        let reply = match &outcome {
            ReadIndexOutcome::Ready(index) => ReadIndexReply {
                term: self.term(),
                success: true,
                read_index: *index,
                leader_id: Some(self.id),
            },
            ReadIndexOutcome::Pending { index, .. } => ReadIndexReply {
                term: self.term(),
                success: true,
                read_index: *index,
                leader_id: Some(self.id),
            },
            ReadIndexOutcome::NotLeader(who) => ReadIndexReply {
                term: self.term(),
                success: false,
                read_index: 0,
                leader_id: *who,
            },
            ReadIndexOutcome::NotReady => ReadIndexReply {
                term: self.term(),
                success: false,
                read_index: 0,
                leader_id: Some(self.id),
            },
        };
        Ok((outcome, reply))
    }
}

/// How many entries a follower whose next index is `next` has yet to be sent
#[inline]
fn entries_pending(next: u64, log_last: u64) -> u64 {
    if next <= log_last {
        log_last + 1 - next
    } else {
        0
    }
}

/// Whether a message carrying `pending` entries may go out to a follower
/// with `inflight` messages unanswered.
///
/// A full batch goes out up to the pipeline depth, because what bounds it is
/// bandwidth. A partial one goes out only inside the smaller window for
/// partial batches. Past that window the entries wait for a reply to free a
/// slot or for enough of them to fill a batch, so under load the pipeline
/// carries batches rather than a message per proposal
#[inline]
fn send_window_open(config: &RaftConfig, inflight: u32, pending: u64) -> bool {
    let depth = config.max_inflight_appends as u32;
    let partial = (config.max_inflight_partial_appends as u32).min(depth);
    inflight < depth && (inflight < partial || pending >= config.max_batch_entries as u64)
}

/// Folds one configuration entry into a membership
fn apply_config_command(membership: &mut Membership, command: &RaftCommand) {
    match command {
        RaftCommand::AddNode { node_id, address } => {
            let next = membership
                .effective()
                .with_node(NodeConfig::learner(*node_id, address.clone()));
            *membership = Membership::Simple(next);
        }
        RaftCommand::RemoveNode { node_id } => {
            let next = membership.effective().without_node(*node_id);
            *membership = Membership::Simple(next);
        }
        RaftCommand::JointConfig { old, new } => {
            *membership = Membership::Joint {
                old: old.clone(),
                new: new.clone(),
            };
        }
        RaftCommand::FinalConfig { config } => {
            *membership = Membership::Simple(config.clone());
        }
        RaftCommand::Noop
        | RaftCommand::Put { .. }
        | RaftCommand::Delete { .. }
        | RaftCommand::Data { .. }
        | RaftCommand::Snapshot { .. } => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;

    fn config_of(ids: &[NodeId]) -> ClusterConfig {
        ClusterConfig::of_voters(ids.iter().map(|id| (*id, format!("node{id}:1"))))
    }

    struct Node {
        core: RaftConsensus,
        _dir: tempfile::TempDir,
    }

    fn node(id: NodeId, ids: &[NodeId], now: Instant) -> Node {
        let dir = tempfile::tempdir().expect("tempdir");
        let state = RaftState::open(dir.path(), 64 * 1024 * 1024).expect("state");
        let mut cfg = RaftConfig::default();
        cfg.pre_vote = true;
        let core = RaftConsensus::new(id, cfg, state, config_of(ids), now).expect("consensus");
        Node { core, _dir: dir }
    }

    fn wait_durable(core: &RaftConsensus) {
        for _ in 0..2000 {
            if core.state.log().is_durable() {
                return;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        panic!("log never became durable");
    }

    /// Runs the pre-vote and the real election against two willing peers
    fn elect(core: &mut RaftConsensus, peers: &[NodeId], now: Instant) {
        let requests = core.tick(now + Duration::from_millis(400)).vote_requests;
        assert!(!requests.is_empty(), "no campaign started");
        let now = now + Duration::from_millis(400);
        if requests[0].1.pre_vote {
            for peer in peers {
                core.handle_request_vote_reply(
                    &RequestVoteReply {
                        term: requests[0].1.term,
                        vote_granted: true,
                        pre_vote: true,
                        voter_id: *peer,
                    },
                    now,
                )
                .expect("pre vote reply");
            }
        }
        let term = core.term();
        for peer in peers {
            core.handle_request_vote_reply(
                &RequestVoteReply {
                    term,
                    vote_granted: true,
                    pre_vote: false,
                    voter_id: *peer,
                },
                now,
            )
            .expect("vote reply");
        }
    }

    #[test]
    fn a_leader_hands_the_group_to_its_most_current_voter() {
        let now = Instant::now();
        let mut leader = node(1, &[1, 2, 3], now);
        elect(&mut leader.core, &[2, 3], now);
        wait_durable(&leader.core);
        assert_eq!(leader.core.role(), RaftRole::Leader);
        let term = leader.core.term();
        let last = leader.core.last_log_index();

        // Follower 3 has confirmed everything, follower 2 nothing yet
        leader
            .core
            .handle_append_reply(
                &AppendEntriesReply {
                    term,
                    success: true,
                    match_index: last,
                    hint_index: 0,
                    read_round: 0,
                    follower_id: 3,
                },
                now,
            )
            .expect("append reply");
        assert_eq!(leader.core.transfer_target(), Some(3));
        assert!(leader.core.peer_caught_up(3));
        assert!(!leader.core.peer_caught_up(2));

        // A follower told to campaign raises its term and asks for real votes
        let mut follower = node(3, &[1, 2, 3], now);
        let (reply, requests) = follower
            .core
            .handle_timeout_now(
                &crate::election::TimeoutNowRequest { term, leader_id: 1 },
                now,
            )
            .expect("timeout now");
        assert!(reply.started);
        assert_eq!(follower.core.role(), RaftRole::Candidate);
        assert_eq!(follower.core.term(), term + 1);
        assert_eq!(reply.term, term + 1);
        assert_eq!(requests.len(), 2, "one real vote request per voting peer");
        assert!(
            requests
                .iter()
                .all(|(_, r)| !r.pre_vote && r.term == term + 1)
        );

        // A stale request is declined, and so is one at a leader
        let (stale, none) = follower
            .core
            .handle_timeout_now(
                &crate::election::TimeoutNowRequest {
                    term: 0,
                    leader_id: 1,
                },
                now,
            )
            .expect("stale timeout now");
        assert!(!stale.started);
        assert!(none.is_empty());
        let (at_leader, _) = leader
            .core
            .handle_timeout_now(
                &crate::election::TimeoutNowRequest { term, leader_id: 2 },
                now,
            )
            .expect("timeout now at a leader");
        assert!(!at_leader.started);
        assert_eq!(leader.core.role(), RaftRole::Leader);
    }

    #[test]
    fn a_lone_voter_takes_the_group_without_asking() {
        let now = Instant::now();
        let mut n = node(1, &[1], now);
        let out = n.core.tick(now + Duration::from_millis(400));
        assert!(out.became_leader);
        assert_eq!(n.core.role(), RaftRole::Leader);
        assert_eq!(n.core.term(), 1);
        // The no-op is there and commits as soon as it is durable
        wait_durable(&n.core);
        n.core.advance_commit();
        assert_eq!(n.core.commit_index(), 1);
    }

    #[test]
    fn pre_vote_runs_before_the_term_moves() {
        let now = Instant::now();
        let mut n = node(1, &[1, 2, 3], now);
        let out = n.core.tick(now + Duration::from_millis(400));
        assert_eq!(out.vote_requests.len(), 2);
        assert!(out.vote_requests[0].1.pre_vote);
        assert_eq!(out.vote_requests[0].1.term, 1);
        // The candidate's own term has not moved
        assert_eq!(n.core.term(), 0);
        assert_eq!(n.core.role(), RaftRole::Follower);
    }

    #[test]
    fn a_refused_pre_vote_from_a_later_term_catches_the_node_up() {
        let now = Instant::now();
        let mut n = node(1, &[1, 2, 3], now);
        n.core.tick(now + Duration::from_millis(400));
        n.core
            .handle_request_vote_reply(
                &RequestVoteReply {
                    term: 40,
                    vote_granted: false,
                    pre_vote: true,
                    voter_id: 2,
                },
                now,
            )
            .expect("reply");
        assert_eq!(n.core.term(), 40);
        assert_eq!(n.core.role(), RaftRole::Follower);
    }

    #[test]
    fn a_voter_refuses_a_pre_vote_while_it_can_hear_a_leader() {
        let now = Instant::now();
        let mut n = node(2, &[1, 2, 3], now);
        n.core
            .handle_append_entries(
                &AppendEntriesRequest {
                    term: 5,
                    leader_id: 1,
                    prev_log_index: 0,
                    prev_log_term: 0,
                    entries: Vec::new(),
                    leader_commit: 0,
                    read_round: 1,
                },
                now,
            )
            .expect("heartbeat");
        let reply = n
            .core
            .handle_request_vote(
                &RequestVoteRequest {
                    term: 6,
                    candidate_id: 3,
                    last_log_index: 0,
                    last_log_term: 0,
                    pre_vote: true,
                    transfer: false,
                },
                now + Duration::from_millis(10),
            )
            .expect("pre vote");
        assert!(!reply.vote_granted);
        // And the voter's term did not move for a pre-vote
        assert_eq!(n.core.term(), 5);
    }

    #[test]
    fn a_voter_refuses_a_real_vote_while_it_can_hear_a_leader_unless_the_leader_asked() {
        let now = Instant::now();
        let mut n = node(2, &[1, 2, 3], now);
        n.core
            .handle_append_entries(
                &AppendEntriesRequest {
                    term: 5,
                    leader_id: 1,
                    prev_log_index: 0,
                    prev_log_term: 0,
                    entries: Vec::new(),
                    leader_commit: 0,
                    read_round: 1,
                },
                now,
            )
            .expect("heartbeat");
        let mut req = RequestVoteRequest {
            term: 6,
            candidate_id: 3,
            last_log_index: 0,
            last_log_term: 0,
            pre_vote: false,
            transfer: false,
        };
        let reply = n
            .core
            .handle_request_vote(&req, now + Duration::from_millis(10))
            .expect("vote");
        assert!(!reply.vote_granted);
        // The term did not move either, so a candidate outside the group
        // cannot depose a leader this node still hears
        assert_eq!(n.core.term(), 5);

        // The leader's own hand-off is granted at once
        req.transfer = true;
        let reply = n
            .core
            .handle_request_vote(&req, now + Duration::from_millis(10))
            .expect("vote");
        assert!(reply.vote_granted);
        assert_eq!(n.core.term(), 6);
    }

    #[test]
    fn a_voter_grants_a_pre_vote_once_the_leader_goes_quiet() {
        let now = Instant::now();
        let mut n = node(2, &[1, 2, 3], now);
        n.core
            .handle_append_entries(
                &AppendEntriesRequest {
                    term: 5,
                    leader_id: 1,
                    prev_log_index: 0,
                    prev_log_term: 0,
                    entries: Vec::new(),
                    leader_commit: 0,
                    read_round: 1,
                },
                now,
            )
            .expect("heartbeat");
        let reply = n
            .core
            .handle_request_vote(
                &RequestVoteRequest {
                    term: 6,
                    candidate_id: 3,
                    last_log_index: 0,
                    last_log_term: 0,
                    pre_vote: true,
                    transfer: false,
                },
                now + Duration::from_millis(500),
            )
            .expect("pre vote");
        assert!(reply.vote_granted);
    }

    #[test]
    fn a_vote_is_refused_to_a_candidate_with_a_shorter_log() {
        let now = Instant::now();
        let mut n = node(2, &[1, 2, 3], now);
        // Give the voter entries from term 4
        let entries: Vec<Arc<RaftLogEntry>> = (1..=5)
            .map(|i| Arc::new(RaftLogEntry::new(4, i, RaftCommand::Noop)))
            .collect();
        n.core
            .handle_append_entries(
                &AppendEntriesRequest {
                    term: 4,
                    leader_id: 1,
                    prev_log_index: 0,
                    prev_log_term: 0,
                    entries,
                    leader_commit: 0,
                    read_round: 0,
                },
                now,
            )
            .expect("append");

        let reply = n
            .core
            .handle_request_vote(
                &RequestVoteRequest {
                    term: 5,
                    candidate_id: 3,
                    last_log_index: 2,
                    last_log_term: 4,
                    pre_vote: false,
                    transfer: false,
                },
                now + Duration::from_millis(500),
            )
            .expect("vote");
        assert!(!reply.vote_granted);
        // The term still moved, because the request carried a later one
        assert_eq!(n.core.term(), 5);
    }

    #[test]
    fn one_vote_per_term() {
        let now = Instant::now();
        let mut n = node(2, &[1, 2, 3], now);
        let req = |candidate| RequestVoteRequest {
            term: 7,
            candidate_id: candidate,
            last_log_index: 0,
            last_log_term: 0,
            pre_vote: false,
            transfer: false,
        };
        assert!(
            n.core
                .handle_request_vote(&req(1), now)
                .expect("vote")
                .vote_granted
        );
        assert!(
            !n.core
                .handle_request_vote(&req(3), now)
                .expect("vote")
                .vote_granted
        );
        // Asking again from the same candidate is idempotent
        assert!(
            n.core
                .handle_request_vote(&req(1), now)
                .expect("vote")
                .vote_granted
        );
    }

    #[test]
    fn a_leader_commits_only_once_a_majority_is_durable() {
        let now = Instant::now();
        let mut n = node(1, &[1, 2, 3], now);
        elect(&mut n.core, &[2, 3], now);
        assert!(n.core.is_leader());
        let now = now + Duration::from_millis(400);

        let (index, _) = n
            .core
            .propose(RaftCommand::Put {
                key: b"k".to_vec(),
                value: b"v".to_vec(),
            })
            .expect("propose");
        wait_durable(&n.core);
        n.core.advance_commit();
        // The leader alone is not a majority of three
        assert!(n.core.commit_index() < index);

        n.core
            .handle_append_reply(
                &AppendEntriesReply {
                    term: n.core.term(),
                    success: true,
                    match_index: index,
                    hint_index: 0,
                    read_round: 0,
                    follower_id: 2,
                },
                now,
            )
            .expect("reply");
        assert_eq!(n.core.commit_index(), index);
    }

    #[test]
    fn an_entry_from_an_earlier_term_is_not_committed_by_count_alone() {
        let now = Instant::now();
        let mut n = node(1, &[1, 2, 3], now);
        // Two entries arrive from an old leader
        let entries: Vec<Arc<RaftLogEntry>> = (1..=2)
            .map(|i| Arc::new(RaftLogEntry::new(1, i, RaftCommand::Noop)))
            .collect();
        n.core
            .handle_append_entries(
                &AppendEntriesRequest {
                    term: 1,
                    leader_id: 9,
                    prev_log_index: 0,
                    prev_log_term: 0,
                    entries,
                    leader_commit: 0,
                    read_round: 0,
                },
                now,
            )
            .expect("append");
        elect(&mut n.core, &[2, 3], now + Duration::from_millis(500));
        assert!(n.core.is_leader());
        wait_durable(&n.core);

        // A follower confirms the old entries but not the new term's no-op
        n.core
            .handle_append_reply(
                &AppendEntriesReply {
                    term: n.core.term(),
                    success: true,
                    match_index: 2,
                    hint_index: 0,
                    read_round: 0,
                    follower_id: 2,
                },
                now + Duration::from_millis(900),
            )
            .expect("reply");
        assert_eq!(n.core.commit_index(), 0);

        // Once the no-op is on a majority, everything before it commits too
        n.core
            .handle_append_reply(
                &AppendEntriesReply {
                    term: n.core.term(),
                    success: true,
                    match_index: 3,
                    hint_index: 0,
                    read_round: 0,
                    follower_id: 2,
                },
                now + Duration::from_millis(900),
            )
            .expect("reply");
        assert_eq!(n.core.commit_index(), 3);
    }

    #[test]
    fn a_conflicting_suffix_is_replaced_not_appended() {
        let now = Instant::now();
        let mut n = node(2, &[1, 2, 3], now);
        let old: Vec<Arc<RaftLogEntry>> = (1..=5)
            .map(|i| Arc::new(RaftLogEntry::new(2, i, RaftCommand::Noop)))
            .collect();
        n.core
            .handle_append_entries(
                &AppendEntriesRequest {
                    term: 2,
                    leader_id: 1,
                    prev_log_index: 0,
                    prev_log_term: 0,
                    entries: old,
                    leader_commit: 0,
                    read_round: 0,
                },
                now,
            )
            .expect("append");
        assert_eq!(n.core.last_log_index(), 5);

        // A new leader overwrites from index 3
        let new: Vec<Arc<RaftLogEntry>> = (3..=4)
            .map(|i| {
                Arc::new(RaftLogEntry::new(
                    3,
                    i,
                    RaftCommand::Put {
                        key: b"n".to_vec(),
                        value: b"v".to_vec(),
                    },
                ))
            })
            .collect();
        let reply = n
            .core
            .handle_append_entries(
                &AppendEntriesRequest {
                    term: 3,
                    leader_id: 4,
                    prev_log_index: 2,
                    prev_log_term: 2,
                    entries: new,
                    leader_commit: 0,
                    read_round: 0,
                },
                now,
            )
            .expect("append");
        assert!(reply.success);
        assert_eq!(reply.match_index, 4);
        assert_eq!(n.core.last_log_index(), 4);
        assert_eq!(n.core.state.log().term_at(3), Some(3));
    }

    #[test]
    fn a_stale_duplicate_does_not_truncate_committed_work() {
        let now = Instant::now();
        let mut n = node(2, &[1, 2, 3], now);
        let entries: Vec<Arc<RaftLogEntry>> = (1..=5)
            .map(|i| Arc::new(RaftLogEntry::new(2, i, RaftCommand::Noop)))
            .collect();
        n.core
            .handle_append_entries(
                &AppendEntriesRequest {
                    term: 2,
                    leader_id: 1,
                    prev_log_index: 0,
                    prev_log_term: 0,
                    entries: entries.clone(),
                    leader_commit: 5,
                    read_round: 0,
                },
                now,
            )
            .expect("append");
        assert_eq!(n.core.last_log_index(), 5);

        // The same message arrives again, carrying only the first two
        let reply = n
            .core
            .handle_append_entries(
                &AppendEntriesRequest {
                    term: 2,
                    leader_id: 1,
                    prev_log_index: 0,
                    prev_log_term: 0,
                    entries: entries[..2].to_vec(),
                    leader_commit: 5,
                    read_round: 0,
                },
                now,
            )
            .expect("append");
        assert!(reply.success);
        assert_eq!(n.core.last_log_index(), 5);
    }

    #[test]
    fn a_rejection_points_past_the_whole_conflicting_term() {
        let now = Instant::now();
        let mut n = node(2, &[1, 2, 3], now);
        // Indexes 1 to 3 in term 1, then 4 to 8 in term 2
        let mut entries: Vec<Arc<RaftLogEntry>> = (1..=3)
            .map(|i| Arc::new(RaftLogEntry::new(1, i, RaftCommand::Noop)))
            .collect();
        entries.extend((4..=8).map(|i| Arc::new(RaftLogEntry::new(2, i, RaftCommand::Noop))));
        n.core
            .handle_append_entries(
                &AppendEntriesRequest {
                    term: 2,
                    leader_id: 1,
                    prev_log_index: 0,
                    prev_log_term: 0,
                    entries,
                    leader_commit: 0,
                    read_round: 0,
                },
                now,
            )
            .expect("append");

        // A leader from term 5 believes index 8 is from term 5
        let reply = n
            .core
            .handle_append_entries(
                &AppendEntriesRequest {
                    term: 5,
                    leader_id: 4,
                    prev_log_index: 8,
                    prev_log_term: 5,
                    entries: Vec::new(),
                    leader_commit: 0,
                    read_round: 0,
                },
                now,
            )
            .expect("append");
        assert!(!reply.success);
        // The whole of term 2 is skipped in one message
        assert_eq!(reply.hint_index, 3);
        assert_eq!(next_index_after_reject(reply.hint_index, 0), 4);
    }

    #[test]
    fn adding_a_learner_does_not_move_the_quorum() {
        let now = Instant::now();
        let mut n = node(1, &[1, 2, 3], now);
        elect(&mut n.core, &[2, 3], now);
        assert_eq!(n.core.membership().effective().voter_count(), 3);
        let cmd = n.core.add_node_command(4, "node4:1").expect("command");
        n.core.propose(cmd).expect("propose");
        assert_eq!(n.core.membership().effective().voter_count(), 3);
        assert!(n.core.membership().contains(4));
        assert!(!n.core.membership().is_voter(4));
        assert_eq!(n.core.peers(), vec![2, 3, 4]);
    }

    #[test]
    fn promotion_goes_through_a_joint_configuration() {
        let now = Instant::now();
        let mut n = node(1, &[1, 2, 3], now);
        elect(&mut n.core, &[2, 3], now);
        let cmd = n.core.add_node_command(4, "node4:1").expect("command");
        let (index, _) = n.core.propose(cmd).expect("propose");
        n.core.state.volatile.commit_index = index;

        let cmd = n.core.promote_command(4).expect("promote");
        let (index, _) = n.core.propose(cmd).expect("propose");
        assert!(n.core.membership().is_joint());
        // Both majorities are needed while the change is in force
        assert!(!n.core.membership().has_quorum(|id| id == 1 || id == 2));
        n.core.state.volatile.commit_index = index;

        let cmd = n.core.leave_joint_command().expect("leave");
        n.core.propose(cmd).expect("propose");
        assert!(!n.core.membership().is_joint());
        assert_eq!(n.core.membership().effective().voter_count(), 4);
    }

    #[test]
    fn two_configuration_changes_cannot_overlap() {
        let now = Instant::now();
        let mut n = node(1, &[1, 2, 3], now);
        elect(&mut n.core, &[2, 3], now);
        let cmd = n.core.add_node_command(4, "node4:1").expect("command");
        n.core.propose(cmd).expect("propose");
        let cmd = n.core.add_node_command(5, "node5:1").expect("command");
        assert!(n.core.propose(cmd).is_err());
    }

    #[test]
    fn a_truncation_rolls_the_configuration_back() {
        let now = Instant::now();
        let mut n = node(2, &[1, 2, 3], now);
        let entries = vec![
            Arc::new(RaftLogEntry::new(
                2,
                1,
                RaftCommand::AddNode {
                    node_id: 9,
                    address: "node9:1".into(),
                },
            )),
            Arc::new(RaftLogEntry::new(2, 2, RaftCommand::Noop)),
        ];
        n.core
            .handle_append_entries(
                &AppendEntriesRequest {
                    term: 2,
                    leader_id: 1,
                    prev_log_index: 0,
                    prev_log_term: 0,
                    entries,
                    leader_commit: 0,
                    read_round: 0,
                },
                now,
            )
            .expect("append");
        assert!(n.core.membership().contains(9));

        // A new leader overwrites index 1, which removes the AddNode
        n.core
            .handle_append_entries(
                &AppendEntriesRequest {
                    term: 3,
                    leader_id: 4,
                    prev_log_index: 0,
                    prev_log_term: 0,
                    entries: vec![Arc::new(RaftLogEntry::new(3, 1, RaftCommand::Noop))],
                    leader_commit: 0,
                    read_round: 0,
                },
                now,
            )
            .expect("append");
        assert!(!n.core.membership().contains(9));
    }

    #[test]
    fn a_leader_that_cannot_reach_a_majority_stands_down() {
        let now = Instant::now();
        let mut n = node(1, &[1, 2, 3], now);
        elect(&mut n.core, &[2, 3], now);
        assert!(n.core.is_leader());
        // Nothing answers for longer than the election timeout
        let out = n.core.tick(now + Duration::from_millis(2000));
        assert!(out.stepped_down);
        assert_eq!(n.core.role(), RaftRole::Follower);
    }

    #[test]
    fn a_read_needs_the_no_op_committed_first() {
        let now = Instant::now();
        let mut n = node(1, &[1, 2, 3], now);
        elect(&mut n.core, &[2, 3], now);
        assert_eq!(n.core.begin_read_index(now), ReadIndexOutcome::NotReady);
    }

    #[test]
    fn a_read_opens_a_round_and_then_rides_the_lease() {
        let now = Instant::now();
        let mut n = node(1, &[1, 2, 3], now);
        elect(&mut n.core, &[2, 3], now);
        wait_durable(&n.core);
        let noop = n.core.last_log_index();
        n.core
            .handle_append_reply(
                &AppendEntriesReply {
                    term: n.core.term(),
                    success: true,
                    match_index: noop,
                    hint_index: 0,
                    read_round: 0,
                    follower_id: 2,
                },
                now,
            )
            .expect("reply");
        assert_eq!(n.core.commit_index(), noop);

        let outcome = n.core.begin_read_index(now);
        let ReadIndexOutcome::Pending { round, index } = outcome else {
            panic!("expected a round to open, got {outcome:?}");
        };
        assert_eq!(index, noop);

        n.core
            .handle_append_reply(
                &AppendEntriesReply {
                    term: n.core.term(),
                    success: true,
                    match_index: noop,
                    hint_index: 0,
                    read_round: round,
                    follower_id: 2,
                },
                now,
            )
            .expect("reply");
        assert!(n.core.quorum_round() >= round);

        // The next read inside the lease costs nothing
        assert_eq!(
            n.core.begin_read_index(now + Duration::from_millis(10)),
            ReadIndexOutcome::Ready(noop)
        );
        assert_eq!(n.core.metrics.read_index_leases_used, 1);
    }

    /// A commit index is owed the moment a follower has confirmed entries it
    /// has not been told are committed, whatever is in flight to it. A
    /// message attaches at what the follower had confirmed when it was
    /// built, so one built before that confirmation lets the follower commit
    /// no further than that and leaves the rest owed. Nothing else is due to
    /// the follower until its heartbeat timer, so a linearizable read
    /// against it would wait out the whole heartbeat interval otherwise
    #[test]
    fn a_follower_is_owed_the_commit_index_for_what_it_has_confirmed() {
        let now = Instant::now();
        let mut n = node(1, &[1, 2, 3], now);
        elect(&mut n.core, &[2, 3], now);
        wait_durable(&n.core);
        let noop = n.core.last_log_index();
        let reply = |core: &mut RaftConsensus, follower: NodeId, at: Instant| {
            core.handle_append_reply(
                &AppendEntriesReply {
                    term: core.term(),
                    success: true,
                    match_index: noop,
                    hint_index: 0,
                    read_round: 0,
                    follower_id: follower,
                },
                at,
            )
            .expect("reply");
        };

        // The no-op goes out to both followers, built while commit is still 0
        let PeerWork::Append(first) = n.core.build_peer_work(2, now).expect("work") else {
            panic!("expected an append carrying the no-op");
        };
        assert!(!first.entries.is_empty());
        let PeerWork::Append(_) = n.core.build_peer_work(3, now).expect("work") else {
            panic!("expected an append carrying the no-op");
        };

        // Follower 3 confirms first and the no-op commits. Follower 2 has
        // confirmed nothing yet, so there is nothing it could act on, and a
        // heartbeat built for it now attaches at index zero
        reply(&mut n.core, 3, now);
        assert_eq!(n.core.commit_index(), noop);
        assert!(
            !n.core.peer_owes_commit(2),
            "nothing confirmed, nothing actionable"
        );
        let heartbeat_at = now + n.core.config.heartbeat_interval + Duration::from_millis(1);
        let PeerWork::Append(early) = n.core.build_peer_work(2, heartbeat_at).expect("work") else {
            panic!("expected a heartbeat");
        };
        assert!(early.entries.is_empty());
        assert_eq!(early.prev_log_index, 0);
        assert_eq!(early.leader_commit, noop);

        // Follower 2 confirms the no-op. The heartbeat that went out let it
        // commit nothing past index zero, so the commit index is owed now,
        // with that heartbeat still unanswered
        reply(&mut n.core, 2, heartbeat_at);
        assert!(
            n.core.peer_owes_commit(2),
            "confirmed the entry, never told it committed"
        );
        let PeerWork::Append(carrier) = n.core.build_peer_work(2, heartbeat_at).expect("work")
        else {
            panic!("expected a commit carrier");
        };
        assert!(carrier.entries.is_empty());
        assert_eq!(carrier.prev_log_index, noop);
        assert_eq!(carrier.leader_commit, noop);

        // The carrier settles the debt and nothing more is due
        assert!(!n.core.peer_owes_commit(2));
        assert!(matches!(
            n.core.build_peer_work(2, heartbeat_at).expect("work"),
            PeerWork::Idle
        ));
    }

    #[test]
    fn a_follower_answers_a_read_with_a_redirect() {
        let now = Instant::now();
        let mut n = node(2, &[1, 2, 3], now);
        n.core
            .handle_append_entries(
                &AppendEntriesRequest {
                    term: 3,
                    leader_id: 1,
                    prev_log_index: 0,
                    prev_log_term: 0,
                    entries: Vec::new(),
                    leader_commit: 0,
                    read_round: 0,
                },
                now,
            )
            .expect("heartbeat");
        let (_, reply) = n
            .core
            .handle_read_index(&ReadIndexRequest { term: 3, from: 5 }, now)
            .expect("read index");
        assert!(!reply.success);
        assert_eq!(reply.leader_id, Some(1));
    }

    #[test]
    fn a_snapshot_resets_the_log_and_the_configuration() {
        let now = Instant::now();
        let mut n = node(2, &[1, 2, 3], now);
        let decision = n
            .core
            .check_install_snapshot(4, 1, 900, now)
            .expect("check");
        assert_eq!(decision, SnapshotDecision::Accept);
        n.core
            .adopt_snapshot(900, 3, config_of(&[1, 2, 3, 7]))
            .expect("adopt");
        assert_eq!(n.core.last_log_index(), 900);
        assert_eq!(n.core.last_log_term(), 3);
        assert_eq!(n.core.commit_index(), 900);
        assert_eq!(n.core.last_applied(), 900);
        assert!(n.core.membership().contains(7));

        // A follower ahead of the leader's guess points it at the boundary
        let reply = n
            .core
            .handle_append_entries(
                &AppendEntriesRequest {
                    term: 4,
                    leader_id: 1,
                    prev_log_index: 10,
                    prev_log_term: 2,
                    entries: Vec::new(),
                    leader_commit: 0,
                    read_round: 0,
                },
                now,
            )
            .expect("append");
        assert!(!reply.success);
        assert_eq!(reply.hint_index, 900);
    }

    #[test]
    fn compaction_carries_the_configuration_into_the_base() {
        let now = Instant::now();
        let mut n = node(1, &[1, 2, 3], now);
        elect(&mut n.core, &[2, 3], now);
        let cmd = n.core.add_node_command(4, "node4:1").expect("command");
        n.core.propose(cmd).expect("propose");
        for _ in 0..10 {
            n.core.propose(RaftCommand::Noop).expect("propose");
        }
        wait_durable(&n.core);
        let at = n.core.last_log_index();
        let term = n.core.term();
        n.core.compact_log(at, term).expect("compact");
        assert_eq!(n.core.state.log().first_index(), at + 1);
        // The learner added in a compacted entry is still in the group
        assert!(n.core.membership().contains(4));
    }

    /// Under load a pipeline that sends every proposal the moment a slot is
    /// free fills with messages of a few entries each. Past the window for
    /// partial batches a follower is sent a full batch or nothing until a
    /// reply frees a slot, and a commit index the follower is owed waits for
    /// the entries that are about to carry it rather than costing a message
    #[test]
    fn partial_batches_stop_at_their_window_and_full_ones_do_not() {
        let now = Instant::now();
        let mut n = node(1, &[1, 2, 3], now);
        n.core.config.max_batch_entries = 10;
        n.core.config.max_inflight_appends = 16;
        n.core.config.max_inflight_partial_appends = 2;
        elect(&mut n.core, &[2, 3], now);
        let noop = n.core.last_log_index();
        let build = |core: &mut RaftConsensus, peer: NodeId| match core
            .build_peer_work(peer, now)
            .expect("work")
        {
            PeerWork::Append(request) => Some(request),
            PeerWork::Idle => None,
            PeerWork::AppendPaged { .. } => panic!("no page-in expected"),
            PeerWork::Snapshot => panic!("no snapshot expected"),
        };
        let reply = |core: &mut RaftConsensus, follower: NodeId, match_index: u64| {
            core.handle_append_reply(
                &AppendEntriesReply {
                    term: core.term(),
                    success: true,
                    match_index,
                    hint_index: 0,
                    read_round: 0,
                    follower_id: follower,
                },
                now,
            )
            .expect("reply");
        };

        // The no-op is the first message, a probe that both followers
        // answer, which opens the pipeline to them. Follower 3 is kept in
        // step so the room check answers for 2
        let first = build(&mut n.core, 2).expect("the no-op goes out");
        assert_eq!(first.entries.len(), 1);
        build(&mut n.core, 3).expect("the no-op goes to the other follower");
        wait_durable(&n.core);
        reply(&mut n.core, 2, noop);
        reply(&mut n.core, 3, noop);
        assert_eq!(n.core.commit_index(), noop);

        // Two partial batches fill the window
        for _ in 0..3 {
            n.core.propose(RaftCommand::Noop).expect("propose");
        }
        let second = build(&mut n.core, 2).expect("a partial batch is inside the window");
        assert_eq!(second.entries.len(), 3);
        build(&mut n.core, 3).expect("the same batch goes to the other follower");
        for _ in 0..3 {
            n.core.propose(RaftCommand::Noop).expect("propose");
        }
        let third = build(&mut n.core, 2).expect("a second partial batch is inside the window");
        assert_eq!(third.entries.len(), 3);
        build(&mut n.core, 3).expect("and to the other follower");

        // A third partial batch waits, and does not wake the replicator
        for _ in 0..3 {
            n.core.propose(RaftCommand::Noop).expect("propose");
        }
        assert!(build(&mut n.core, 2).is_none());
        assert!(build(&mut n.core, 3).is_none());
        assert!(!n.core.replication_has_room(now));

        // A full batch goes out past the window
        for _ in 0..7 {
            n.core.propose(RaftCommand::Noop).expect("propose");
        }
        assert!(n.core.replication_has_room(now));
        let full = build(&mut n.core, 2).expect("a full batch goes out");
        assert_eq!(full.entries.len(), 10);
        build(&mut n.core, 3).expect("and to the other follower");
        assert!(!n.core.replication_has_room(now));

        // Two more are pending behind three in flight. The commit index
        // moves, follower 2 is owed it, and still nothing goes out on its
        // own because the entries about to go carry it
        for _ in 0..2 {
            n.core.propose(RaftCommand::Noop).expect("propose");
        }
        wait_durable(&n.core);
        reply(&mut n.core, 3, noop + 16);
        reply(&mut n.core, 2, noop + 3);
        assert_eq!(n.core.commit_index(), noop + 16);
        assert!(n.core.peer_owes_commit(2));
        assert!(build(&mut n.core, 2).is_none());
        assert!(!n.core.replication_has_room(now));

        // That reply left two in flight, which is still the window. The
        // next one frees a slot, the replicator is worth waking, and the
        // pending entries go out with the commit index
        reply(&mut n.core, 2, noop + 6);
        assert!(n.core.replication_has_room(now));
        let after = build(&mut n.core, 2).expect("a freed slot sends what is pending");
        assert_eq!(after.entries.len(), 2);
        assert_eq!(after.leader_commit, noop + 16);
        assert!(!n.core.peer_owes_commit(2));
    }

    /// A follower whose position is unconfirmed is sent entries one message
    /// at a time. A wrong guess then costs one refusal, and a follower that
    /// cannot be reached costs one batch per heartbeat interval rather than
    /// one per proposal, which for a follower that is down and behind by a
    /// gigabyte is the difference between a leader that carries on and one
    /// that rebuilds a megabyte on every write
    #[test]
    fn an_unconfirmed_follower_is_probed_one_message_at_a_time() {
        let now = Instant::now();
        let mut n = node(1, &[1, 2, 3], now);
        n.core.config.max_batch_entries = 2;
        n.core.config.max_inflight_appends = 4;
        n.core.config.max_inflight_partial_appends = 4;
        elect(&mut n.core, &[2, 3], now);
        let noop = n.core.last_log_index();
        for _ in 0..11 {
            n.core.propose(RaftCommand::Noop).expect("propose");
        }
        let build = |core: &mut RaftConsensus, at: Instant| match core
            .build_peer_work(2, at)
            .expect("work")
        {
            PeerWork::Append(request) => Some(request),
            PeerWork::Idle => None,
            PeerWork::AppendPaged { .. } => panic!("no page-in expected"),
            PeerWork::Snapshot => panic!("no snapshot expected"),
        };
        let answer = |core: &mut RaftConsensus, success: bool, match_index: u64, hint: u64| {
            core.handle_append_reply(
                &AppendEntriesReply {
                    term: core.term(),
                    success,
                    match_index,
                    hint_index: hint,
                    read_round: 0,
                    follower_id: 2,
                },
                now,
            )
            .expect("reply");
        };

        // After the election one batch goes out and nothing follows it
        // until the follower has answered. Follower 3 gets its probe too, so
        // the room check answers for 2 alone
        let probe = build(&mut n.core, now).expect("the probe goes out");
        assert_eq!(probe.entries.len(), 2);
        assert!(matches!(
            n.core.build_peer_work(3, now).expect("work"),
            PeerWork::Append(_)
        ));
        assert!(build(&mut n.core, now).is_none());
        assert!(!n.core.replication_has_room(now));
        answer(&mut n.core, true, noop + 1, 0);

        // Confirmed, the pipeline fills to its depth
        assert!(n.core.replication_has_room(now));
        assert!(build(&mut n.core, now).is_some());
        assert!(build(&mut n.core, now).is_some());
        assert!(build(&mut n.core, now).is_some());
        assert!(build(&mut n.core, now).is_some());
        assert!(
            build(&mut n.core, now).is_none(),
            "the pipeline is four deep"
        );

        // A call that produced no answer puts the follower back to one
        // message at a time, and not before a heartbeat interval has passed
        n.core.handle_peer_unreachable(2, now);
        assert!(build(&mut n.core, now).is_none());
        assert!(!n.core.replication_has_room(now));
        let retry = now + n.core.config.heartbeat_interval;
        assert!(n.core.replication_has_room(retry));
        let again = build(&mut n.core, retry).expect("one attempt at the retry time");
        assert_eq!(
            again.prev_log_index,
            noop + 1,
            "rebuilt from what was confirmed"
        );
        assert!(build(&mut n.core, retry).is_none());

        // A refusal keeps the follower probing from the hint it gave
        answer(&mut n.core, false, 0, noop + 1);
        let from_hint = build(&mut n.core, retry).expect("one attempt from the hint");
        assert_eq!(from_hint.prev_log_index, noop + 1);
        assert!(build(&mut n.core, retry).is_none());

        // And a confirmation opens the pipeline again
        answer(&mut n.core, true, noop + 3, 0);
        assert!(build(&mut n.core, retry).is_some());
        assert!(build(&mut n.core, retry).is_some());
    }

    /// A group of two needs both members for a majority. The leader's own
    /// durable index is one of two, and counting it as a majority would
    /// commit entries no follower holds, which a leader failure then loses
    #[test]
    fn a_group_of_two_commits_on_both_members_and_not_on_the_leader_alone() {
        let now = Instant::now();
        let mut n = node(1, &[1, 2], now);
        elect(&mut n.core, &[2], now);
        assert!(n.core.is_leader());
        wait_durable(&n.core);
        let noop = n.core.last_log_index();
        n.core.advance_commit();
        assert_eq!(
            n.core.commit_index(),
            0,
            "the leader alone is not a majority of two"
        );

        n.core
            .handle_append_reply(
                &AppendEntriesReply {
                    term: n.core.term(),
                    success: true,
                    match_index: noop,
                    hint_index: 0,
                    read_round: 0,
                    follower_id: 2,
                },
                now,
            )
            .expect("reply");
        assert_eq!(n.core.commit_index(), noop);
    }

    #[test]
    fn peer_work_pipelines_up_to_the_inflight_bound() {
        let now = Instant::now();
        let mut n = node(1, &[1, 2, 3], now);
        n.core.config.max_batch_entries = 4;
        n.core.config.max_inflight_appends = 3;
        elect(&mut n.core, &[2, 3], now);
        let noop = n.core.last_log_index();
        for _ in 0..100 {
            n.core.propose(RaftCommand::Noop).expect("propose");
        }
        // The first batch is the probe, and its answer opens the pipeline
        let PeerWork::Append(probe) = n.core.build_peer_work(2, now).expect("work") else {
            panic!("expected the probe");
        };
        assert_eq!(probe.entries.len(), 4);
        n.core
            .handle_append_reply(
                &AppendEntriesReply {
                    term: n.core.term(),
                    success: true,
                    match_index: noop + 3,
                    hint_index: 0,
                    read_round: 0,
                    follower_id: 2,
                },
                now,
            )
            .expect("reply");
        let mut built = 0;
        loop {
            match n.core.build_peer_work(2, now).expect("work") {
                PeerWork::Append(req) => {
                    assert!(req.entries.len() <= 4);
                    built += 1;
                }
                PeerWork::Idle => break,
                PeerWork::AppendPaged { .. } => panic!("no page-in expected"),
                PeerWork::Snapshot => panic!("no snapshot expected"),
            }
            if built > 10 {
                break;
            }
        }
        // The pipeline fills to its bound and no further
        assert_eq!(built, 3);
    }

    #[test]
    fn a_follower_behind_the_snapshot_point_gets_a_snapshot() {
        let now = Instant::now();
        let mut n = node(1, &[1, 2, 3], now);
        elect(&mut n.core, &[2, 3], now);
        for _ in 0..50 {
            n.core.propose(RaftCommand::Noop).expect("propose");
        }
        wait_durable(&n.core);
        let at = n.core.last_log_index();
        let term = n.core.term();
        n.core.compact_log(at, term).expect("compact");
        assert!(matches!(
            n.core.build_peer_work(2, now).expect("work"),
            PeerWork::Snapshot
        ));
    }
}

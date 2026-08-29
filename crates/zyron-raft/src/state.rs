//! What a node knows, split by what it may lose on a power cut.
//!
//! ## The split is the whole safety argument
//!
//! [`PersistentState`] holds the three things a node must never forget: the
//! term it has reached, the vote it gave in that term, and the log. Forgetting
//! a vote lets the same node vote twice in one term, and two votes from one
//! node is exactly how two leaders appear in a single term. So the term and
//! the vote are written and fsynced before any reply that depends on them
//! leaves the node.
//!
//! [`VolatileState`] holds what is cheaper to rediscover than to persist. The
//! commit index is re-derived by the leader from what followers report, and
//! the applied index is re-derived from the state machine's own checkpoint. A
//! node that restarts with both at zero is correct, only slow.
//!
//! [`LeaderState`] exists only while this node leads, and is rebuilt from
//! scratch each time it wins an election, because everything in it is a belief
//! about other nodes that a new term invalidates.

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use zyron_common::checksum::hash32;
use zyron_common::error::{Result, ZyronError};

use crate::NodeId;
use crate::log::RaftLog;

/// Where the term and the vote live under the raft directory
pub const STATE_FILE: &str = "raft.state";
const STATE_TMP_FILE: &str = "raft.state.tmp";
const STATE_MAGIC: [u8; 8] = *b"ZYRAFTST";
const STATE_VERSION: u32 = 1;
const STATE_LEN: usize = 40;

/// What a node is doing right now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RaftRole {
    /// Answers leaders and candidates, serves reads, proposes nothing
    Follower,
    /// Has raised its term and is collecting votes
    Candidate,
    /// Holds the group and is the only node that appends
    Leader,
}

impl RaftRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            RaftRole::Follower => "follower",
            RaftRole::Candidate => "candidate",
            RaftRole::Leader => "leader",
        }
    }
}

impl std::fmt::Display for RaftRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The state that outlives the process.
pub struct PersistentState {
    current_term: u64,
    voted_for: Option<NodeId>,
    /// The replicated log, whose own durability is published separately
    /// because it advances thousands of times more often than the term does
    pub log: RaftLog,
    path: PathBuf,
    tmp_path: PathBuf,
    /// How many times the term or vote has been written, so an operator can
    /// see an election storm as a rate rather than as a symptom
    writes: u64,
}

impl PersistentState {
    /// Opens the term, the vote, and the log under one directory.
    ///
    /// A state file that fails its checksum stops the node. The alternative,
    /// starting at term zero with no vote, is a node that will happily vote
    /// again in a term it has already voted in
    pub fn open(dir: &Path, resident_log_bytes: usize) -> Result<Self> {
        std::fs::create_dir_all(dir)
            .map_err(|e| ZyronError::IoError(format!("create raft directory: {e}")))?;
        let path = dir.join(STATE_FILE);
        let tmp_path = dir.join(STATE_TMP_FILE);
        let _ = std::fs::remove_file(&tmp_path);

        let (current_term, voted_for) = if path.exists() {
            let mut buf = Vec::with_capacity(STATE_LEN);
            File::open(&path)
                .map_err(|e| ZyronError::IoError(format!("open raft state: {e}")))?
                .read_to_end(&mut buf)
                .map_err(|e| ZyronError::IoError(format!("read raft state: {e}")))?;
            if buf.len() != STATE_LEN {
                return Err(ZyronError::RecoveryFailed(format!(
                    "raft state file is {} bytes, expected {STATE_LEN}",
                    buf.len()
                )));
            }
            if buf[0..8] != STATE_MAGIC {
                return Err(ZyronError::RecoveryFailed(
                    "raft state file magic does not match".into(),
                ));
            }
            let version = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);
            if version != STATE_VERSION {
                return Err(ZyronError::RecoveryFailed(format!(
                    "raft state version {version} is not {STATE_VERSION}"
                )));
            }
            let stored = u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]);
            let computed = hash32(&buf[16..STATE_LEN]);
            if stored != computed {
                return Err(ZyronError::RecoveryFailed(format!(
                    "raft state checksum {stored:#010x} does not match computed {computed:#010x}"
                )));
            }
            let term = u64::from_le_bytes([
                buf[16], buf[17], buf[18], buf[19], buf[20], buf[21], buf[22], buf[23],
            ]);
            let vote_value = u64::from_le_bytes([
                buf[24], buf[25], buf[26], buf[27], buf[28], buf[29], buf[30], buf[31],
            ]);
            let voted = if buf[32] == 1 { Some(vote_value) } else { None };
            (term, voted)
        } else {
            (0, None)
        };

        let log = RaftLog::open(dir, resident_log_bytes)?;
        Ok(Self {
            current_term,
            voted_for,
            log,
            path,
            tmp_path,
            writes: 0,
        })
    }

    #[inline]
    pub fn current_term(&self) -> u64 {
        self.current_term
    }

    #[inline]
    pub fn voted_for(&self) -> Option<NodeId> {
        self.voted_for
    }

    #[inline]
    pub fn writes(&self) -> u64 {
        self.writes
    }

    /// Raises the term and clears the vote, then makes both durable.
    ///
    /// The two move together because a term is the scope of a vote: carrying a
    /// vote forward into a new term would let this node vote for two different
    /// candidates in the same term across a restart
    pub fn advance_term(&mut self, term: u64) -> Result<()> {
        if term <= self.current_term {
            return Ok(());
        }
        self.current_term = term;
        self.voted_for = None;
        self.persist()
    }

    /// Records a vote in the current term
    pub fn record_vote(&mut self, candidate: NodeId) -> Result<()> {
        if self.voted_for == Some(candidate) {
            return Ok(());
        }
        self.voted_for = Some(candidate);
        self.persist()
    }

    /// Raises the term and votes for this node in one write, which is what a
    /// candidate does when it starts a real election
    pub fn start_term_voting_for(&mut self, term: u64, candidate: NodeId) -> Result<()> {
        self.current_term = term;
        self.voted_for = Some(candidate);
        self.persist()
    }

    fn persist(&mut self) -> Result<()> {
        let mut buf = [0u8; STATE_LEN];
        buf[0..8].copy_from_slice(&STATE_MAGIC);
        buf[8..12].copy_from_slice(&STATE_VERSION.to_le_bytes());
        buf[16..24].copy_from_slice(&self.current_term.to_le_bytes());
        buf[24..32].copy_from_slice(&self.voted_for.unwrap_or(0).to_le_bytes());
        buf[32] = u8::from(self.voted_for.is_some());
        let checksum = hash32(&buf[16..STATE_LEN]);
        buf[12..16].copy_from_slice(&checksum.to_le_bytes());

        // Written to a sibling and renamed so a crash mid-write leaves the
        // previous term and vote rather than a half of each
        let mut tmp = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&self.tmp_path)
            .map_err(|e| ZyronError::IoError(format!("open raft state temp: {e}")))?;
        tmp.write_all(&buf)
            .map_err(|e| ZyronError::IoError(format!("write raft state: {e}")))?;
        tmp.sync_all()
            .map_err(|e| ZyronError::IoError(format!("sync raft state: {e}")))?;
        drop(tmp);
        std::fs::rename(&self.tmp_path, &self.path)
            .map_err(|e| ZyronError::IoError(format!("rename raft state: {e}")))?;
        self.writes += 1;
        Ok(())
    }
}

/// What a node rebuilds rather than persists.
#[derive(Debug, Clone, Copy, Default)]
pub struct VolatileState {
    /// Highest index known to be replicated on a majority
    pub commit_index: u64,
    /// Highest index handed to the state machine
    pub last_applied: u64,
}

/// What a leader believes about each of its followers.
///
/// Parallel arrays rather than a map: a leader walks all of them on every
/// heartbeat and on every commit recomputation, and the peer count is small
/// and fixed between configuration changes
pub struct LeaderState {
    /// The peers these arrays describe, in the order they are indexed
    pub peers: Vec<NodeId>,
    /// The next index to send to each follower, guessed high and corrected
    /// down by rejections
    pub next_index: Vec<u64>,
    /// The highest index each follower has confirmed, never guessed
    pub match_index: Vec<u64>,
    /// AppendEntries outstanding to each follower, which is what bounds the
    /// pipeline depth
    pub inflight: Vec<u32>,
    /// Whether a snapshot transfer is running to each follower, so a second
    /// one is not started underneath it
    pub snapshot_in_flight: Vec<bool>,
    /// The highest read round each follower has echoed
    pub acked_round: Vec<u64>,
    /// The commit index last sent to each follower.
    ///
    /// A follower learns what is committed only from the leader, so a commit
    /// that advances with nothing left to replicate still owes every follower
    /// a message. Without this the follower would not apply until the next
    /// heartbeat, and a linearizable read against it would wait out that whole
    /// interval for no reason
    pub sent_commit: Vec<u64>,
    /// When each follower last answered anything, read by the quorum check
    pub last_contact: Vec<Instant>,
    /// When the next heartbeat is due to each follower
    pub next_heartbeat: Vec<Instant>,
    /// Round counter stamped into heartbeats so a read can prove it was
    /// answered by a leader that still held the group
    pub heartbeat_round: u64,
    /// The highest round a majority has echoed
    pub quorum_round: u64,
    /// When that happened, which is where the read lease starts
    pub quorum_round_at: Instant,
    /// Index of the no-op this leader appended on election. Reads and commits
    /// wait for it, because until it commits the leader cannot tell which
    /// inherited entries are committed
    pub noop_index: u64,
}

impl LeaderState {
    /// Builds the leader's view of a fresh term.
    ///
    /// `next_index` starts one past the leader's own last entry, which is the
    /// optimistic guess Raft prescribes: usually right, and cheap to correct
    /// when it is not
    pub fn new(peers: Vec<NodeId>, last_index: u64, now: Instant) -> Self {
        let n = peers.len();
        Self {
            peers,
            next_index: vec![last_index + 1; n],
            match_index: vec![0; n],
            inflight: vec![0; n],
            snapshot_in_flight: vec![false; n],
            acked_round: vec![0; n],
            sent_commit: vec![0; n],
            last_contact: vec![now; n],
            next_heartbeat: vec![now; n],
            heartbeat_round: 0,
            quorum_round: 0,
            quorum_round_at: now,
            noop_index: 0,
        }
    }

    #[inline]
    pub fn pos(&self, peer: NodeId) -> Option<usize> {
        self.peers.iter().position(|p| *p == peer)
    }

    /// Brings the arrays in line with a configuration change, keeping what is
    /// known about peers that stayed
    pub fn reconcile(&mut self, peers: Vec<NodeId>, last_index: u64, now: Instant) {
        if peers == self.peers {
            return;
        }
        let mut next_index = Vec::with_capacity(peers.len());
        let mut match_index = Vec::with_capacity(peers.len());
        let mut inflight = Vec::with_capacity(peers.len());
        let mut snapshot_in_flight = Vec::with_capacity(peers.len());
        let mut acked_round = Vec::with_capacity(peers.len());
        let mut sent_commit = Vec::with_capacity(peers.len());
        let mut last_contact = Vec::with_capacity(peers.len());
        let mut next_heartbeat = Vec::with_capacity(peers.len());
        for peer in &peers {
            match self.pos(*peer) {
                Some(i) => {
                    next_index.push(self.next_index[i]);
                    match_index.push(self.match_index[i]);
                    inflight.push(self.inflight[i]);
                    snapshot_in_flight.push(self.snapshot_in_flight[i]);
                    acked_round.push(self.acked_round[i]);
                    sent_commit.push(self.sent_commit[i]);
                    last_contact.push(self.last_contact[i]);
                    next_heartbeat.push(self.next_heartbeat[i]);
                }
                None => {
                    next_index.push(last_index + 1);
                    match_index.push(0);
                    inflight.push(0);
                    snapshot_in_flight.push(false);
                    acked_round.push(0);
                    sent_commit.push(0);
                    last_contact.push(now);
                    next_heartbeat.push(now);
                }
            }
        }
        self.peers = peers;
        self.next_index = next_index;
        self.match_index = match_index;
        self.inflight = inflight;
        self.snapshot_in_flight = snapshot_in_flight;
        self.acked_round = acked_round;
        self.sent_commit = sent_commit;
        self.last_contact = last_contact;
        self.next_heartbeat = next_heartbeat;
    }

    /// What a peer is known to hold, or zero for a peer that is not in the
    /// leader's arrays yet
    #[inline]
    pub fn match_of(&self, peer: NodeId) -> u64 {
        self.pos(peer).map(|i| self.match_index[i]).unwrap_or(0)
    }

    /// Forgets what was outstanding to a peer after a failed call, so the next
    /// batch is rebuilt from what the peer actually confirmed
    pub fn reset_progress(&mut self, i: usize) {
        self.next_index[i] = self.match_index[i] + 1;
        self.inflight[i] = 0;
        // The message that would have carried the commit index did not land,
        // so the follower is owed it again
        self.sent_commit[i] = 0;
    }
}

/// Everything one node holds about its own consensus state.
///
/// The term, the vote, and the log are inside [`PersistentState`], and the
/// commit and applied indexes inside [`VolatileState`], because that split is
/// what a reader has to understand first
pub struct RaftState {
    pub persistent: PersistentState,
    pub volatile: VolatileState,
    pub role: RaftRole,
    /// The leader this node last accepted, which is what a client redirect
    /// names. Cleared while an election is running
    pub leader_id: Option<NodeId>,
    /// Present only while this node is the leader
    pub leader: Option<LeaderState>,
}

impl RaftState {
    pub fn open(dir: &Path, resident_log_bytes: usize) -> Result<Self> {
        Ok(Self {
            persistent: PersistentState::open(dir, resident_log_bytes)?,
            volatile: VolatileState::default(),
            role: RaftRole::Follower,
            leader_id: None,
            leader: None,
        })
    }

    #[inline]
    pub fn current_term(&self) -> u64 {
        self.persistent.current_term()
    }

    #[inline]
    pub fn voted_for(&self) -> Option<NodeId> {
        self.persistent.voted_for()
    }

    #[inline]
    pub fn log(&self) -> &RaftLog {
        &self.persistent.log
    }

    #[inline]
    pub fn log_mut(&mut self) -> &mut RaftLog {
        &mut self.persistent.log
    }

    #[inline]
    pub fn commit_index(&self) -> u64 {
        self.volatile.commit_index
    }

    #[inline]
    pub fn last_applied(&self) -> u64 {
        self.volatile.last_applied
    }

    #[inline]
    pub fn is_leader(&self) -> bool {
        self.role == RaftRole::Leader
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn term_and_vote_survive_a_reopen() {
        let dir = tempfile::tempdir().expect("tempdir");
        {
            let mut s = PersistentState::open(dir.path(), 64 * 1024 * 1024).expect("open");
            assert_eq!(s.current_term(), 0);
            assert_eq!(s.voted_for(), None);
            s.start_term_voting_for(4, 77).expect("vote");
        }
        let s = PersistentState::open(dir.path(), 64 * 1024 * 1024).expect("reopen");
        assert_eq!(s.current_term(), 4);
        assert_eq!(s.voted_for(), Some(77));
    }

    #[test]
    fn advancing_the_term_clears_the_vote() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut s = PersistentState::open(dir.path(), 64 * 1024 * 1024).expect("open");
        s.start_term_voting_for(2, 5).expect("vote");
        s.advance_term(3).expect("advance");
        assert_eq!(s.current_term(), 3);
        assert_eq!(s.voted_for(), None);
    }

    #[test]
    fn a_corrupt_state_file_stops_the_node() {
        let dir = tempfile::tempdir().expect("tempdir");
        {
            let mut s = PersistentState::open(dir.path(), 64 * 1024 * 1024).expect("open");
            s.start_term_voting_for(9, 1).expect("vote");
        }
        let path = dir.path().join(STATE_FILE);
        let mut bytes = std::fs::read(&path).expect("read");
        bytes[20] ^= 0xFF;
        std::fs::write(&path, &bytes).expect("write");
        assert!(PersistentState::open(dir.path(), 64 * 1024 * 1024).is_err());
    }

    #[test]
    fn reconcile_keeps_what_is_known_about_peers_that_stayed() {
        let now = Instant::now();
        let mut ls = LeaderState::new(vec![2, 3], 10, now);
        ls.match_index[0] = 7;
        ls.next_index[0] = 8;
        ls.reconcile(vec![2, 4], 20, now);
        assert_eq!(ls.peers, vec![2, 4]);
        assert_eq!(ls.match_index[0], 7);
        assert_eq!(ls.next_index[0], 8);
        // The new peer starts from the optimistic guess
        assert_eq!(ls.match_index[1], 0);
        assert_eq!(ls.next_index[1], 21);
    }
}

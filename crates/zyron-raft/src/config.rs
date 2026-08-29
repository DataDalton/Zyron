//! The numbers that decide how fast a group reacts, and what they trade.
//!
//! Every value here is a latency budget in disguise. The election timeout
//! bounds how long a group is leaderless after a leader dies, and it must stay
//! several heartbeats above the round trip to a majority or a healthy leader
//! gets deposed by its own network. The batch bounds decide how much work one
//! fsync covers. The snapshot threshold decides how much memory the log holds
//! and how long a new node takes to catch up.
//!
//! Defaults are sized for a group inside one region. [`RaftConfig::multi_region`]
//! is the same shape stretched for links where a round trip is tens of
//! milliseconds rather than tenths.

use std::time::Duration;

use zyron_common::error::{Result, ZyronError};

/// Timers and bounds for one consensus group.
#[derive(Debug, Clone)]
pub struct RaftConfig {
    /// Shortest a follower waits without hearing from a leader before it
    /// campaigns
    pub election_timeout_min: Duration,
    /// Longest it waits. The gap between the two is what stops every follower
    /// campaigning at the same instant and splitting the vote
    pub election_timeout_max: Duration,
    /// How often a leader contacts each follower when it has nothing to send.
    /// Well under `election_timeout_min` so a healthy leader is never doubted
    pub heartbeat_interval: Duration,
    /// How often the driver looks at the clock. Bounds the error on every
    /// deadline above
    pub tick_interval: Duration,
    /// How long a call waits for an answer before the peer is treated as
    /// unreachable for that call
    pub rpc_timeout: Duration,
    /// Most entries one AppendEntries carries
    pub max_batch_entries: usize,
    /// Most bytes one AppendEntries carries
    pub max_batch_bytes: usize,
    /// How many AppendEntries may be outstanding to one follower at once.
    /// This is what makes replication a pipeline rather than a ping pong
    pub max_inflight_appends: usize,
    /// How many entries past the snapshot the log may grow before another
    /// snapshot is taken
    pub snapshot_threshold: u64,
    /// How many bytes past the snapshot the log may grow before another
    /// snapshot is taken.
    ///
    /// The entry count alone is the right trigger for a log of keys and the
    /// wrong one for a log of transactions: ten thousand entries carrying a
    /// megabyte each is ten gigabytes of log that no count would notice
    pub snapshot_threshold_bytes: u64,
    /// How many bytes of decoded commands the log holds in memory.
    ///
    /// Past this the oldest are dropped and read back from the file when a
    /// follower asks for them, so a leader carrying a large log does not carry
    /// it twice
    pub resident_log_bytes: usize,
    /// How much of a snapshot one InstallSnapshot message carries
    pub snapshot_chunk_bytes: usize,
    /// Whether a candidate asks whether it could win before it raises its term
    pub pre_vote: bool,
    /// Whether a leader that stops hearing from a majority steps down
    pub check_quorum: bool,
    /// How long a leader may answer a linearizable read from its last
    /// confirmed heartbeat round instead of running a new one
    pub leader_lease: Duration,
    /// Most entries one apply pass hands to the state machine before it looks
    /// at the clock again
    pub apply_batch: usize,
    /// How long a caller waits for its proposal to commit
    pub propose_timeout: Duration,
}

impl Default for RaftConfig {
    fn default() -> Self {
        let election_timeout_min = Duration::from_millis(150);
        Self {
            election_timeout_min,
            election_timeout_max: Duration::from_millis(300),
            heartbeat_interval: Duration::from_millis(30),
            tick_interval: Duration::from_millis(5),
            rpc_timeout: Duration::from_millis(1000),
            max_batch_entries: 1024,
            max_batch_bytes: 1024 * 1024,
            max_inflight_appends: 16,
            snapshot_threshold: 10_000,
            snapshot_threshold_bytes: 512 * 1024 * 1024,
            resident_log_bytes: 64 * 1024 * 1024,
            snapshot_chunk_bytes: 1024 * 1024,
            pre_vote: true,
            check_quorum: true,
            leader_lease: election_timeout_min / 2,
            apply_batch: 4096,
            propose_timeout: Duration::from_secs(10),
        }
    }
}

impl RaftConfig {
    /// Timers for a group whose members are in different regions.
    ///
    /// A cross-region round trip is tens of milliseconds, so a 150ms election
    /// timeout would have followers campaigning against a leader that is
    /// simply far away. Everything that waits on the network is stretched;
    /// nothing that waits on the local disk is
    pub fn multi_region() -> Self {
        let election_timeout_min = Duration::from_millis(500);
        Self {
            election_timeout_min,
            election_timeout_max: Duration::from_millis(1500),
            heartbeat_interval: Duration::from_millis(150),
            tick_interval: Duration::from_millis(20),
            rpc_timeout: Duration::from_millis(5000),
            leader_lease: election_timeout_min / 2,
            ..Self::default()
        }
    }

    /// Refuses a configuration whose timers fight each other.
    ///
    /// A heartbeat interval at or above the election timeout means the leader
    /// is deposed on a schedule, and a group configured that way never makes
    /// progress. Better to refuse at startup than to look like a network fault
    /// forever
    pub fn validate(&self) -> Result<()> {
        if self.election_timeout_min.is_zero() {
            return Err(ZyronError::Internal(
                "raft election_timeout_min must be above zero".into(),
            ));
        }
        if self.election_timeout_max < self.election_timeout_min {
            return Err(ZyronError::Internal(
                "raft election_timeout_max is below election_timeout_min".into(),
            ));
        }
        if self.heartbeat_interval * 2 >= self.election_timeout_min {
            return Err(ZyronError::Internal(format!(
                "raft heartbeat_interval {:?} leaves no room under election_timeout_min {:?}, a healthy leader would be deposed",
                self.heartbeat_interval, self.election_timeout_min
            )));
        }
        if self.tick_interval.is_zero() || self.tick_interval > self.heartbeat_interval {
            return Err(ZyronError::Internal(format!(
                "raft tick_interval {:?} cannot resolve a heartbeat_interval of {:?}",
                self.tick_interval, self.heartbeat_interval
            )));
        }
        if self.max_batch_entries == 0 || self.max_batch_bytes == 0 {
            return Err(ZyronError::Internal(
                "raft replication batch bounds must be above zero".into(),
            ));
        }
        if self.max_inflight_appends == 0 {
            return Err(ZyronError::Internal(
                "raft max_inflight_appends must be at least one".into(),
            ));
        }
        if self.snapshot_chunk_bytes == 0 {
            return Err(ZyronError::Internal(
                "raft snapshot_chunk_bytes must be above zero".into(),
            ));
        }
        if self.leader_lease >= self.election_timeout_min {
            return Err(ZyronError::Internal(
                "raft leader_lease must stay under election_timeout_min, or a deposed leader could still serve reads".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_coherent() {
        RaftConfig::default().validate().expect("defaults");
        RaftConfig::multi_region().validate().expect("multi region");
    }

    #[test]
    fn a_heartbeat_that_cannot_beat_the_election_timeout_is_refused() {
        let mut c = RaftConfig::default();
        c.heartbeat_interval = Duration::from_millis(200);
        assert!(c.validate().is_err());
    }

    #[test]
    fn a_lease_longer_than_the_election_timeout_is_refused() {
        let mut c = RaftConfig::default();
        c.leader_lease = Duration::from_millis(400);
        assert!(c.validate().is_err());
    }
}

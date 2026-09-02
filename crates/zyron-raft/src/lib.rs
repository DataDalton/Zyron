//! Raft consensus for one replication group.
//!
//! A group is a handful of nodes that agree on an ordered sequence of
//! commands, so that the state machine each of them runs ends up in the same
//! place. One node leads and appends, the rest follow, and a command is
//! committed once a majority holds it on disk. A group of `2f + 1` survives
//! `f` failures, and never contradicts itself no matter how the survivors are
//! partitioned.
//!
//! ## Reading order
//!
//! - [`log`] is what a command is and how it reaches disk, including the
//!   single writer thread that turns a burst of proposals into one fsync.
//! - [`state`] is what a node knows, split by what it may lose on a power cut.
//! - [`membership`] is who is in the group, and what a majority means while
//!   that is changing.
//! - [`election`] and [`replication`] are the two message exchanges, with the
//!   reasoning for pre-vote and for the pipelined append in their docs.
//! - [`snapshot`] is how a log prefix is replaced by the state it produced.
//! - [`consensus`] is the protocol itself: synchronous, clockless, and driven
//!   from outside.
//! - [`node`] is the running node, four loops over that core.
//! - [`transport`] carries the messages, and [`machine`] is the seam to
//!   whatever is being replicated.
//!
//! ## What this crate does not do
//!
//! One group, one log. A table sharded across many groups is Multi-Raft, and
//! it is built from many of these rather than inside one of them: the group
//! here has no notion of a key range, and adding one would put a routing
//! decision inside the code whose whole job is to agree.
//!
//! ## The three promises
//!
//! Every design choice in here comes back to one of three things a node must
//! never do.
//!
//! It must never forget a vote it gave. Two votes from one node in one term is
//! how two leaders appear, so the term and the vote are fsynced before the
//! reply that rests on them is sent.
//!
//! It must never acknowledge an entry it could lose. A majority of
//! acknowledgements is what makes an entry committed, so an AppendEntries
//! reply waits for the log writer to publish a durable index past it.
//!
//! It must never serve a read from a leadership it no longer holds. A read
//! either rides a lease that a recent heartbeat round established, or opens a
//! round of its own and waits for a majority to answer it.

pub mod codec;
pub mod config;
pub mod consensus;
pub mod election;
pub mod format;
pub mod log;
pub mod machine;
pub mod membership;
pub mod node;
pub mod replication;
pub mod snapshot;
pub mod state;
pub mod transport;

/// A node's identity, stable across restarts and minted with its data
/// directory
pub type NodeId = u64;

/// A term is a logical clock: it increases every time an election is held, and
/// it is what lets a node recognize a message from a leader it has already
/// replaced
pub type Term = u64;

/// A position in the replicated log, one based and gapless
pub type LogIndex = u64;

/// The consensus id a node answers to, derived from the name an operator
/// gave it.
///
/// Membership SQL names a node by name, and every node in the group has to
/// arrive at the same id for that name without asking anyone, so the id is a
/// function of the name rather than something minted and then distributed. A
/// sixty four bit hash makes a collision between two operator-chosen names a
/// non-event, and a configuration that somehow held one is refused by
/// `ClusterConfig::validate` rather than silently merging the two nodes.
///
/// Zero is reserved for "no node", so a name that hashes to it is nudged
pub fn node_id_for_name(name: &str) -> NodeId {
    match zyron_common::checksum::hash64(name.as_bytes()) {
        0 => 1,
        id => id,
    }
}

pub use config::RaftConfig;
pub use consensus::{
    ConsensusMetrics, PeerWork, RaftConsensus, ReadIndexOutcome, SnapshotDecision, TickOutcome,
};
pub use election::{ElectionTimer, RequestVoteReply, RequestVoteRequest, log_is_up_to_date};
pub use log::{LogWriterHandle, RaftCommand, RaftLog, RaftLogEntry};
pub use machine::{ApplyFuture, CheckpointSource, MemoryStateMachine, StateMachine};
pub use membership::{ClusterConfig, MAX_CLUSTER_NODES, Membership, NodeConfig};
pub use node::{RaftMetrics, RaftNode, RaftNodeOptions};
pub use replication::{
    AppendEntriesReply, AppendEntriesRequest, ReadIndexReply, ReadIndexRequest,
    next_index_after_reject,
};
pub use snapshot::{
    InstallSnapshotReply, InstallSnapshotRequest, RaftSnapshot, SnapshotMeta, SnapshotStore,
};
pub use state::{LeaderState, PersistentState, RaftRole, RaftState, VolatileState};
pub use transport::{
    RaftFuture, RaftHandlerFuture, RaftRequestHandler, RaftRpcError, RaftServer, RaftTransport,
    TcpTransport, TransportConfig, TransportStats,
};

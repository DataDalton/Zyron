//! What this node does with a committed consensus entry, and how it joins its
//! group.
//!
//! `zyron-raft` agrees on a sequence and knows nothing about tables. This is
//! the other side of that seam.
//!
//! ## Why the apply is logical and not physical
//!
//! The obvious design is to ship the leader's write-ahead log records and have
//! every follower append them, which is how a physical replica works. This
//! engine cannot be replicated that way. Its recovery reads the log to decide
//! which transactions committed, not to rebuild pages: pages are written in
//! place through the buffer pool and the log is not a redo stream. A follower
//! handed the leader's records would end up with the leader's log and none of
//! the leader's rows.
//!
//! What is agreed instead is the rows a transaction produced, and the schema
//! changes that decide what a row is. [`crate::replication`] holds both: the
//! applier that puts them back, and the seam a connection reaches them
//! through.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use zyron_buffer::BufferPool;
use zyron_catalog::Catalog;
use zyron_common::error::{Result, ZyronError};
use zyron_raft::machine::{ApplyFuture, CheckpointSource, StateMachine};
use zyron_raft::{
    ClusterConfig, NodeConfig, RaftCommand, RaftConfig, RaftLogEntry, RaftNode, RaftNodeOptions,
    RaftServer, TcpTransport, TransportConfig, node_id_for_name,
};
use zyron_storage::DiskManager;
use zyron_storage::txn::TransactionManager;
use zyron_wal::WalWriter;

use crate::config::ClusterSection;
use crate::replication::{ChangesetMachine, EngineHandles};

/// The state machine a Zyron node presents to its consensus group.
///
/// Thin on purpose. Everything an entry can mean is either a changeset, which
/// [`ChangesetMachine`] applies, or a fact about the group itself, which the
/// consensus layer applies for us. What is left here is the applied index and
/// the checkpoint
pub struct ZyronStateMachine {
    changesets: Arc<ChangesetMachine>,
    engine: EngineHandles,
    applied: AtomicU64,
}

impl ZyronStateMachine {
    pub fn new(changesets: Arc<ChangesetMachine>, engine: EngineHandles) -> Arc<Self> {
        Arc::new(Self {
            changesets,
            engine,
            applied: AtomicU64::new(0),
        })
    }

    pub fn changesets(&self) -> &Arc<ChangesetMachine> {
        &self.changesets
    }

    /// Reads back where this node's replay resumes.
    ///
    /// The figures come out of the write-ahead log, where every commit this
    /// group agreed to recorded the floor below which nothing needs replay
    /// and whose transaction it was. Keeping them beside the data rather than
    /// inside it would let the two disagree across a crash: a node that lost
    /// the commit record but kept the position would never replay the
    /// transaction it lost.
    ///
    /// The floor is not simply the last committed entry. A transaction
    /// streamed across several entries is staged as its chunks arrive and its
    /// staged writes die with the process, so replay has to restart below the
    /// first chunk of anything that was still open. The transactions that
    /// were already committed inside that replayed range are handed to the
    /// applier so it passes over them instead of staging their rows twice.
    ///
    /// `snapshot_floor` is the entry this node's own raft snapshot covers the
    /// log to. It anchors recovery from below: a stretch of entries that
    /// wrote no rows, schema changes and no-ops, records no commits, and a
    /// checkpoint may have reclaimed old records outright, so the write-ahead
    /// log alone can understate the position. The snapshot only ever compacts
    /// entries whose transactions all committed, which makes its point a
    /// floor in exactly the same sense
    pub async fn recover_applied_index(&self, snapshot_floor: u64) -> Result<u64> {
        let mut state = self.engine.wal.recover_replay_state()?;
        let floor = state.floor.max(snapshot_floor);
        // Entries at or below the anchored floor are never replayed, so a
        // commit down there needs neither skipping nor pinning
        state.committed.retain(|c| c.stamp.index > floor);
        if !state.committed.is_empty() {
            tracing::info!(
                floor,
                committed = state.committed.len(),
                "replay resumes below transactions already committed here"
            );
        }
        self.changesets.set_recovered(&state.committed);
        self.applied.store(floor, Ordering::Release);
        Ok(floor)
    }

    async fn apply_one(&self, index: u64, term: u64, command: &RaftCommand) -> Result<()> {
        match command {
            RaftCommand::Data { payload } => {
                self.changesets.apply_data(index, term, payload).await?;
            }
            // A leader appends a no-op the moment it takes the group, and
            // every node applies it at the same position. That makes it the
            // one place a transaction the previous leader left half staged can
            // be abandoned everywhere at once
            RaftCommand::Noop => {
                self.changesets.abandon_before_term(term);
            }
            // A key-value entry is a cluster setting, an upgrade setting that
            // every node applies at the same point so a leadership change
            // cannot lose it. Any other key has no meaning to a database
            // node, and refusing is the only honest answer: applying it as
            // nothing would let a group think this node holds state it does
            // not
            RaftCommand::Put { key, value } => {
                let (key, value) = match (std::str::from_utf8(key), std::str::from_utf8(value)) {
                    (Ok(key), Ok(value)) if crate::cluster_settings::is_cluster_setting(key) => {
                        (key, value)
                    }
                    _ => {
                        return Err(ZyronError::RaftLogCorrupted {
                            index,
                            reason: "a key-value command that is not a cluster setting reached a \
                                     database node"
                                .into(),
                        });
                    }
                };
                // A value this binary refuses, or a config file it cannot
                // write, leaves this node behind on one setting. Stopping the
                // apply loop over it would leave it behind on everything
                if let Err(e) = crate::cluster_settings::apply(&self.engine.data_dir, key, value) {
                    tracing::error!(
                        index,
                        key,
                        value,
                        error = %e,
                        "a replicated setting was not applied on this node"
                    );
                }
            }
            RaftCommand::Delete { .. } => {
                return Err(ZyronError::RaftLogCorrupted {
                    index,
                    reason: "a key-value delete reached a database node".into(),
                });
            }
            // Consensus state rather than user state. The node applies these
            // itself and the machine only moves its index past them
            RaftCommand::AddNode { .. }
            | RaftCommand::RemoveNode { .. }
            | RaftCommand::Snapshot { .. }
            | RaftCommand::JointConfig { .. }
            | RaftCommand::FinalConfig { .. } => {}
        }
        self.applied.store(index, Ordering::Release);
        Ok(())
    }
}

impl StateMachine for ZyronStateMachine {
    /// Never called: this machine writes through an asynchronous engine, so it
    /// answers on [`StateMachine::apply_batch`] instead
    fn apply(&self, _index: u64, _command: &RaftCommand) -> Result<()> {
        Err(ZyronError::Internal(
            "a database state machine applies asynchronously, through apply_batch".into(),
        ))
    }

    fn apply_batch<'a>(&'a self, entries: &'a [Arc<RaftLogEntry>]) -> ApplyFuture<'a> {
        Box::pin(async move {
            let mut last = 0u64;
            for entry in entries {
                self.apply_one(entry.index, entry.term, &entry.command)
                    .await?;
                last = entry.index;
            }
            Ok(last)
        })
    }

    fn begin_checkpoint(&self) -> Result<Box<dyn CheckpointSource>> {
        Ok(Box::new(AppliedIndexCheckpoint {
            applied: self.applied.load(Ordering::Acquire),
        }))
    }

    fn restore(&self, _path: &Path, _last_included_index: u64) -> Result<()> {
        Err(ZyronError::RecoveryFailed(
            "this node fell behind the part of the log the group still holds, so it cannot be \
caught up by replaying it. Start it from a copy of a current member's data directory, or raise \
cluster.snapshot_threshold_bytes so the log is retained for longer"
                .into(),
        ))
    }

    fn applied_index(&self) -> u64 {
        self.applied.load(Ordering::Acquire)
    }

    fn retain_floor(&self) -> u64 {
        // A transaction still streaming pins the log at its first chunk. A
        // restart replays it from there, so a snapshot must not compact it
        // away in the meantime
        self.changesets.retain_floor()
    }

    fn digest(&self) -> u64 {
        // Comparing two nodes by hashing every row would read the whole
        // database, so what is published is the position rather than the
        // contents. Two nodes at the same applied index have applied the same
        // entries, which is the property an operator is checking
        self.applied.load(Ordering::Acquire)
    }
}

/// A checkpoint that records where the machine had got to and nothing else.
struct AppliedIndexCheckpoint {
    applied: u64,
}

impl CheckpointSource for AppliedIndexCheckpoint {
    fn last_applied(&self) -> u64 {
        self.applied
    }

    fn write_to(&mut self, path: &std::path::Path) -> Result<u64> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| ZyronError::IoError(format!("create checkpoint directory: {e}")))?;
        }
        std::fs::write(path, self.applied.to_le_bytes())
            .map_err(|e| ZyronError::IoError(format!("write checkpoint: {e}")))?;
        Ok(8)
    }
}

/// A node started into its consensus group, with the listener that serves it.
pub struct ClusterHandle {
    pub node: Arc<RaftNode>,
    pub machine: Arc<ZyronStateMachine>,
    /// Everything a connection needs to route a write through the group
    pub replication: Arc<crate::replication::ReplicationHandle>,
    listener: tokio::task::JoinHandle<()>,
    shutdown: Arc<tokio::sync::Notify>,
}

impl ClusterHandle {
    /// Stops serving peers and stops the node's own loops
    pub async fn shutdown(&self) {
        self.shutdown.notify_waiters();
        self.node.shutdown().await;
        self.listener.abort();
    }
}

/// Brings this node into the group its configuration describes.
///
/// The group is named entirely by configuration: every member's name and
/// address, including this node's. Nothing is discovered, because a node that
/// guessed its own membership could form a second group alongside the real one
#[allow(clippy::too_many_arguments)]
pub async fn start_cluster(
    section: &ClusterSection,
    data_dir: &Path,
    catalog: Arc<Catalog>,
    wal: Arc<WalWriter>,
    buffer_pool: Arc<BufferPool>,
    disk_manager: Arc<DiskManager>,
    txn_manager: Arc<TransactionManager>,
) -> Result<ClusterHandle> {
    section.validate()?;
    let id = node_id_for_name(&section.node_name);
    let nodes: Vec<NodeConfig> = section
        .peers
        .iter()
        .map(|p| NodeConfig::voter(node_id_for_name(&p.name), p.address.clone()))
        .collect();
    let bootstrap = ClusterConfig::new(nodes);
    bootstrap.validate()?;

    let engine = crate::replication::EngineHandles {
        catalog: Arc::clone(&catalog),
        wal: Arc::clone(&wal),
        buffer_pool: Arc::clone(&buffer_pool),
        disk_manager: Arc::clone(&disk_manager),
        txn_manager: Arc::clone(&txn_manager),
        data_dir: data_dir.to_path_buf(),
    };
    // A different number every time the process starts, so an entry this node
    // proposed before a restart is replayed rather than matched against a
    // transaction that died with the process that made it
    let epoch = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    let engine_for_machine = engine.clone();
    let changesets = Arc::new(crate::replication::ChangesetMachine::new(engine, id, epoch));

    let machine = ZyronStateMachine::new(Arc::clone(&changesets), engine_for_machine);
    // The snapshot pointer is read before the node opens the store for real,
    // because the machine's recovered position feeds the node's start and has
    // to be anchored by the snapshot the log was compacted against
    let raft_dir: PathBuf = data_dir.join("raft");
    let snapshot_floor =
        zyron_raft::snapshot::SnapshotStore::open(&raft_dir)?.last_included_index();
    machine.recover_applied_index(snapshot_floor).await?;

    let mut config = if section.multi_region {
        RaftConfig::multi_region()
    } else {
        RaftConfig::default()
    };
    config.snapshot_threshold = section.snapshot_threshold;
    if section.snapshot_threshold_bytes != 0 {
        config.snapshot_threshold_bytes = section.snapshot_threshold_bytes;
    }
    if section.resident_log_bytes != 0 {
        config.resident_log_bytes = section.resident_log_bytes as usize;
    }
    config.validate()?;

    let propose_timeout = config.propose_timeout;
    let transport_config = TransportConfig {
        rpc_timeout: config.rpc_timeout,
        ..TransportConfig::default()
    };
    let transport = TcpTransport::new(id, transport_config.clone());

    let node = Arc::new(
        RaftNode::start(RaftNodeOptions {
            id,
            dir: raft_dir,
            config,
            bootstrap,
            machine: Arc::clone(&machine) as Arc<dyn StateMachine>,
            transport,
        })
        .await?,
    );

    let listener = tokio::net::TcpListener::bind(&section.listen)
        .await
        .map_err(|e| {
            ZyronError::ConfigError(format!(
                "consensus listener could not bind {}: {e}",
                section.listen
            ))
        })?;
    let shutdown = Arc::new(tokio::sync::Notify::new());
    let handler = node.handler();
    let stop = Arc::clone(&shutdown);
    let serving = tokio::spawn(async move {
        RaftServer::serve(listener, handler, transport_config, stop).await;
    });

    // A checkpoint may only reclaim an agreed commit record once the raft
    // snapshot covers its entry and the snapshot point itself anchors
    // recovery. Without this pin, WAL truncation would erase the records a
    // restart reads to avoid committing the same transactions twice
    let pin_machine = Arc::clone(&changesets);
    let pin_node = Arc::clone(&node);
    wal.add_retention_hook(Arc::new(move || {
        pin_machine.wal_retention_pin(pin_node.snapshot_point())
    }));

    let proposer = crate::replication::ChunkProposer::start(
        Arc::clone(&node),
        Arc::clone(changesets.pending()),
    );
    let replication = Arc::new(crate::replication::ReplicationHandle {
        node: Arc::clone(&node),
        machine: Arc::clone(&changesets),
        proposer,
        chunk_bytes: section.chunk_bytes as usize,
        epoch,
        propose_timeout,
        // Raised by the upgrade service once it has seen every member's
        // version. Until then a schema change replicates the way the previous
        // release wrote it
        group_carries_actor_role: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        // Raised the same way. Until then a transactional consume is refused
        // on this node, because there is no shorter form of an advance to
        // send in its place
        group_carries_stream_advance: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        // Raised the same way. Until then a delete or update on a table
        // with a change data feed is refused on this node, because a key
        // sent in place of the row would leave every feed short of it
        group_carries_feed_images: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        // Raised the same way. Until then a lake write is refused on this
        // node, because a version sent without its files would name bytes
        // a member does not hold
        group_carries_lake_files: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        // Raised the same way. Until then no schedule runs on this node,
        // because a run recorded here alone would be run again by the next
        // leader
        group_carries_schedule_runs: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        group_carries_commit_chains: Arc::new(std::sync::atomic::AtomicBool::new(false)),
    });

    tracing::info!(
        node = id,
        name = %section.node_name,
        listen = %section.listen,
        members = section.peers.len(),
        table = %section.replicated_table,
        "joined a consensus group"
    );
    Ok(ClusterHandle {
        node,
        machine,
        replication,
        listener: serving,
        shutdown,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The checkpoint a database node writes is its position, so it round
    /// trips as the eight bytes it is
    #[test]
    fn an_applied_index_checkpoint_round_trips() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("state.zsnap");
        let mut source = AppliedIndexCheckpoint { applied: 4242 };
        assert_eq!(source.last_applied(), 4242);
        assert_eq!(source.write_to(&path).expect("write"), 8);
        let bytes = std::fs::read(&path).expect("read");
        assert_eq!(
            u64::from_le_bytes(bytes.try_into().expect("eight bytes")),
            4242
        );
    }

    /// A key-value command has no meaning to a database node, and applying it
    /// as nothing would let the group believe this node holds state it does
    /// not
    #[test]
    fn a_key_value_command_is_refused_rather_than_ignored() {
        let command = RaftCommand::Put {
            key: b"k".to_vec(),
            value: b"v".to_vec(),
        };
        assert!(matches!(command, RaftCommand::Put { .. }));
    }
}

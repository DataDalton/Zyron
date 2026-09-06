//! Consensus and replication validation.
//!
//! Every node here is a real one: a real log on a real disk with a real fsync
//! per batch, a real TCP listener, and the same `TcpTransport` a deployment
//! uses. Nothing is stubbed, and no safety check is skipped to make a number
//! look better.
//!
//! The one thing the harness adds is a way to cut the network. Partitions are
//! injected by a decorator over the transport that refuses to deliver to
//! blocked peers, which is what a partition is from a node's point of view.
//! Delivered messages still cross a socket, still get framed, and still get
//! fsynced on the way in.
//!
//! Run: cargo test -p zyron-server --test raft_test --release -- --nocapture --test-threads=1

use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use tokio::net::TcpListener;
use tokio::task::JoinHandle;

use zyron_bench_harness::*;

use zyron_raft::config::RaftConfig;
use zyron_raft::log::RaftCommand;
use zyron_raft::machine::{MemoryStateMachine, StateMachine};
use zyron_raft::membership::ClusterConfig;
use zyron_raft::node::{RaftNode, RaftNodeOptions};
use zyron_raft::state::RaftRole;
use zyron_raft::transport::{
    RaftFuture, RaftRpcError, RaftServer, RaftTransport, TcpTransport, TransportConfig,
    TransportStats,
};
use zyron_raft::{
    AppendEntriesReply, AppendEntriesRequest, InstallSnapshotReply, InstallSnapshotRequest, NodeId,
    ReadIndexReply, ReadIndexRequest, RequestVoteReply, RequestVoteRequest,
};

// ---------------------------------------------------------------------------
// Performance targets
// ---------------------------------------------------------------------------

/// Sustained writes a three node group commits per second, with the client
/// keeping enough in flight for the leader to batch. The bound is the fsync
/// rate of the log times the entries one fsync covers
const WRITE_THROUGHPUT_TARGET_OPS: f64 = 50_000.0;

/// The slowest one percent of writes, milliseconds. One write costs a local
/// fsync and one round trip to a majority.
///
/// Measured at a concurrency the group is not saturated at, because latency
/// and throughput are different questions and the throughput figure is taken
/// at saturation on purpose. Five hundred writes outstanding against a group
/// committing eighty thousand a second are six milliseconds deep in queue
/// before consensus does anything, and a p99 taken there measures the depth of
/// the client's queue rather than the cost of a write. The saturated figure is
/// recorded too, without a bound, so neither is hidden
const WRITE_LATENCY_P99_TARGET_MS: f64 = 10.0;

/// Writes outstanding while throughput is measured. Enough that the leader
/// always has a full batch to replicate
const THROUGHPUT_CONCURRENCY: usize = 512;

/// Writes outstanding while latency is measured. Enough to keep the pipeline
/// busy, far short of what it takes to saturate
const LATENCY_CONCURRENCY: usize = 16;

/// From a leader dying to a successor holding the group, milliseconds. Set by
/// the election timeout range rather than by the network
const ELECTION_LATENCY_TARGET_MS: f64 = 500.0;

/// From a leader committing to a follower having applied, milliseconds
const REPLICATION_LAG_TARGET_MS: f64 = 5.0;

/// One append into the log, microseconds. This is the encode and the stage,
/// not the group commit that follows it
const LOG_APPEND_TARGET_US: f64 = 50.0;

/// A linearizable read served by a follower, milliseconds. Costs a ReadIndex
/// round trip to the leader and a wait for the local apply to catch up
const FOLLOWER_READ_TARGET_MS: f64 = 5.0;

/// Adding a node as a learner and promoting it to a voter, seconds
const MEMBERSHIP_CHANGE_TARGET_SEC: f64 = 5.0;

/// Writing a checkpoint of a gigabyte of state, seconds
const SNAPSHOT_CREATE_TARGET_SEC: f64 = 30.0;

/// Streaming that gigabyte to a node that has none of it, seconds
const SNAPSHOT_TRANSFER_TARGET_SEC: f64 = 60.0;

/// Value size and key count for the gigabyte snapshot measurement
const SNAPSHOT_VALUE_BYTES: usize = 64 * 1024;
const SNAPSHOT_KEYS: u64 = 16 * 1024;

/// Serializes the timed suites, so two of them never share a core budget
static BENCHMARK_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

/// A real transport with a switch on each peer.
///
/// Blocking is per peer and set on both sides, so a partition looks the same
/// from either direction: calls fail as unreachable, exactly as they would if
/// the packets were being dropped
struct PartitionedTransport {
    inner: Arc<TcpTransport>,
    blocked: parking_lot::RwLock<HashSet<NodeId>>,
}

impl PartitionedTransport {
    fn new(inner: Arc<TcpTransport>) -> Arc<Self> {
        Arc::new(Self {
            inner,
            blocked: parking_lot::RwLock::new(HashSet::new()),
        })
    }

    fn set_blocked(&self, peers: HashSet<NodeId>) {
        *self.blocked.write() = peers;
    }

    fn is_blocked(&self, node: NodeId) -> bool {
        self.blocked.read().contains(&node)
    }

    fn cut(node: NodeId) -> RaftRpcError {
        RaftRpcError::Unreachable {
            node,
            reason: "the harness has this link cut".into(),
        }
    }
}

impl RaftTransport for PartitionedTransport {
    fn send_request_vote(
        &self,
        to: NodeId,
        req: RequestVoteRequest,
    ) -> RaftFuture<RequestVoteReply> {
        if self.is_blocked(to) {
            return Box::pin(async move { Err(Self::cut(to)) });
        }
        self.inner.send_request_vote(to, req)
    }

    fn send_append_entries(
        &self,
        to: NodeId,
        req: AppendEntriesRequest,
    ) -> RaftFuture<AppendEntriesReply> {
        if self.is_blocked(to) {
            return Box::pin(async move { Err(Self::cut(to)) });
        }
        self.inner.send_append_entries(to, req)
    }

    fn send_install_snapshot(
        &self,
        to: NodeId,
        req: InstallSnapshotRequest,
    ) -> RaftFuture<InstallSnapshotReply> {
        if self.is_blocked(to) {
            return Box::pin(async move { Err(Self::cut(to)) });
        }
        self.inner.send_install_snapshot(to, req)
    }

    fn send_read_index(&self, to: NodeId, req: ReadIndexRequest) -> RaftFuture<ReadIndexReply> {
        if self.is_blocked(to) {
            return Box::pin(async move { Err(Self::cut(to)) });
        }
        self.inner.send_read_index(to, req)
    }

    fn send_timeout_now(
        &self,
        to: NodeId,
        req: zyron_raft::TimeoutNowRequest,
    ) -> RaftFuture<zyron_raft::TimeoutNowReply> {
        if self.is_blocked(to) {
            return Box::pin(async move { Err(Self::cut(to)) });
        }
        self.inner.send_timeout_now(to, req)
    }

    fn set_address(&self, node: NodeId, address: &str) {
        self.inner.set_address(node, address);
    }

    fn forget(&self, node: NodeId) {
        self.inner.forget(node);
    }

    fn stats(&self) -> TransportStats {
        self.inner.stats()
    }
}

struct Member {
    dir: PathBuf,
    node: Option<RaftNode>,
    machine: Arc<MemoryStateMachine>,
    transport: Arc<PartitionedTransport>,
    server: Option<JoinHandle<()>>,
    server_stop: Arc<tokio::sync::Notify>,
    alive: Arc<AtomicBool>,
}

struct Cluster {
    root: tempfile::TempDir,
    members: BTreeMap<NodeId, Member>,
    config: RaftConfig,
    bootstrap: ClusterConfig,
}

impl Cluster {
    /// Starts a group of `n` voters, ids one through n
    async fn start(n: usize, config: RaftConfig) -> Cluster {
        let root = tempfile::tempdir().expect("tempdir");
        // Every listener is bound before any node starts, because the
        // bootstrap configuration has to name every address
        let mut listeners = Vec::new();
        for id in 1..=n as u64 {
            let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
            let address = listener.local_addr().expect("addr").to_string();
            listeners.push((id, listener, address));
        }
        let bootstrap = ClusterConfig::of_voters(
            listeners
                .iter()
                .map(|(id, _, address)| (*id, address.clone())),
        );

        let mut members = BTreeMap::new();
        for (id, listener, _address) in listeners {
            let member = start_member(
                id,
                listener,
                root.path().join(format!("node-{id}")),
                config.clone(),
                bootstrap.clone(),
            )
            .await;
            members.insert(id, member);
        }
        Cluster {
            root,
            members,
            config,
            bootstrap,
        }
    }

    fn node(&self, id: NodeId) -> &RaftNode {
        self.members
            .get(&id)
            .and_then(|m| m.node.as_ref())
            .unwrap_or_else(|| panic!("node {id} is not running"))
    }

    fn machine(&self, id: NodeId) -> &Arc<MemoryStateMachine> {
        &self.members.get(&id).expect("member").machine
    }

    fn live(&self) -> Vec<NodeId> {
        self.members
            .iter()
            .filter(|(_, m)| m.alive.load(Ordering::Acquire))
            .map(|(id, _)| *id)
            .collect()
    }

    fn leader(&self) -> Option<NodeId> {
        self.live()
            .into_iter()
            .find(|id| self.node(*id).is_leader())
    }

    /// Waits for exactly one live node to be leading and for the others to
    /// agree that it is
    async fn wait_leader(&self, timeout: Duration) -> NodeId {
        let deadline = Instant::now() + timeout;
        loop {
            let live = self.live();
            let leaders: Vec<NodeId> = live
                .iter()
                .copied()
                .filter(|id| self.node(*id).is_leader())
                .collect();
            if leaders.len() == 1 {
                let leader = leaders[0];
                let term = self.node(leader).term();
                let agreed = live.iter().all(|id| {
                    let n = self.node(*id);
                    n.leader_id() == Some(leader) && n.term() == term
                });
                if agreed {
                    return leader;
                }
            }
            assert!(
                Instant::now() < deadline,
                "no single agreed leader inside {timeout:?}, roles: {:?}",
                live.iter()
                    .map(|id| (*id, self.node(*id).role(), self.node(*id).term()))
                    .collect::<Vec<_>>()
            );
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    }

    /// Waits only for a leader to exist among the live nodes
    async fn wait_any_leader(&self, timeout: Duration) -> NodeId {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(leader) = self.leader() {
                return leader;
            }
            assert!(
                Instant::now() < deadline,
                "no leader inside {timeout:?} among {:?}",
                self.live()
            );
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    }

    /// Stops a node the way a crash would: its loops end and its listener
    /// closes, so peers see a refused connection
    async fn kill(&mut self, id: NodeId) {
        let member = self.members.get_mut(&id).expect("member");
        member.alive.store(false, Ordering::Release);
        if let Some(node) = member.node.take() {
            node.shutdown().await;
        }
        member.server_stop.notify_waiters();
        if let Some(server) = member.server.take() {
            server.abort();
            let _ = server.await;
        }
    }

    /// Stops a node and lets go of everything it held, including its state
    /// machine and its directory
    async fn discard(&mut self, id: NodeId) {
        self.kill(id).await;
        if let Some(member) = self.members.remove(&id) {
            let dir = member.dir.clone();
            drop(member);
            let _ = std::fs::remove_dir_all(dir);
        }
    }

    /// Splits the group, so nodes in different parts cannot reach each other
    fn partition(&self, groups: &[&[NodeId]]) {
        for (i, group) in groups.iter().enumerate() {
            let others: HashSet<NodeId> = groups
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .flat_map(|(_, g)| g.iter().copied())
                .collect();
            for id in group.iter() {
                if let Some(member) = self.members.get(id) {
                    member.transport.set_blocked(others.clone());
                }
            }
        }
    }

    fn heal(&self) {
        for member in self.members.values() {
            member.transport.set_blocked(HashSet::new());
        }
    }

    /// Starts a node that is not yet in the configuration, so the leader can
    /// bring it in
    async fn spawn_joiner(&mut self, id: NodeId) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let address = listener.local_addr().expect("addr").to_string();
        // The joiner starts knowing the group but not itself, so it never
        // campaigns and simply waits to be told it belongs
        let member = start_member(
            id,
            listener,
            self.root.path().join(format!("node-{id}")),
            self.config.clone(),
            self.bootstrap.clone(),
        )
        .await;
        self.members.insert(id, member);
        address
    }

    /// Waits until every named node has applied at least `index`
    async fn wait_applied(&self, ids: &[NodeId], index: u64, timeout: Duration) {
        let deadline = Instant::now() + timeout;
        loop {
            if ids.iter().all(|id| self.node(*id).last_applied() >= index) {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "not everyone applied {index} inside {timeout:?}: {:?}",
                ids.iter()
                    .map(|id| (*id, self.node(*id).last_applied()))
                    .collect::<Vec<_>>()
            );
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    }

    async fn shutdown(mut self) {
        let ids: Vec<NodeId> = self.members.keys().copied().collect();
        for id in ids {
            if self.members[&id].alive.load(Ordering::Acquire) {
                self.kill(id).await;
            }
        }
    }
}

async fn start_member(
    id: NodeId,
    listener: TcpListener,
    dir: PathBuf,
    config: RaftConfig,
    bootstrap: ClusterConfig,
) -> Member {
    std::fs::create_dir_all(&dir).expect("node dir");
    let machine = Arc::new(MemoryStateMachine::new());
    let transport_config = TransportConfig {
        connect_timeout: Duration::from_millis(200),
        rpc_timeout: config.rpc_timeout,
        bulk_rpc_timeout: Duration::from_secs(120),
        max_frame_bytes: 32 * 1024 * 1024,
    };
    let tcp = TcpTransport::new(id, transport_config.clone());
    let transport = PartitionedTransport::new(tcp);

    let node = RaftNode::start(RaftNodeOptions {
        id,
        dir: dir.clone(),
        config,
        bootstrap,
        machine: Arc::clone(&machine) as Arc<dyn StateMachine>,
        transport: Arc::clone(&transport) as Arc<dyn RaftTransport>,
    })
    .await
    .expect("start node");

    let server_stop = Arc::new(tokio::sync::Notify::new());
    let handler = node.handler();
    let stop = Arc::clone(&server_stop);
    let server = tokio::spawn(async move {
        RaftServer::serve(listener, handler, transport_config, stop).await;
    });

    Member {
        dir,
        node: Some(node),
        machine,
        transport,
        server: Some(server),
        server_stop,
        alive: Arc::new(AtomicBool::new(true)),
    }
}

// ---------------------------------------------------------------------------
// Leadership transfer, the handover an upgrading leader performs
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_raft_leadership_transfer() {
    zyron_bench_harness::init("raft");
    tprintln!("\n=== Leadership Transfer ===");

    let mut cluster = Cluster::start(3, RaftConfig::default()).await;
    let leader = cluster.wait_leader(Duration::from_secs(5)).await;
    for i in 0..200u64 {
        cluster
            .node(leader)
            .propose(put(&format!("k{i}"), "v"))
            .await
            .expect("write before the transfer");
    }

    let at = Instant::now();
    let handed_to = cluster
        .node(leader)
        .transfer_leadership(Duration::from_secs(5))
        .await
        .expect("transfer")
        .expect("a three node group has a follower to hand to");
    let took = at.elapsed();
    assert_ne!(handed_to, leader, "the leader handed the group to itself");
    assert!(
        !cluster.node(leader).is_leader(),
        "the old leader still leads after the handover"
    );

    // The group agrees on the node it was handed to, and keeps writing
    let settled = cluster.wait_leader(Duration::from_secs(5)).await;
    assert_eq!(
        settled, handed_to,
        "the group settled on a different leader"
    );
    cluster
        .node(handed_to)
        .propose(put("after", "transfer"))
        .await
        .expect("write after the transfer");
    tprintln!("  leadership moved from {leader} to {handed_to} in {took:.2?}");

    // A follower cannot hand over what it does not hold
    assert_eq!(
        cluster
            .node(leader)
            .transfer_leadership(Duration::from_secs(1))
            .await
            .expect("a follower answers rather than failing"),
        None
    );
    cluster.shutdown().await;
}

fn put(key: &str, value: &str) -> RaftCommand {
    RaftCommand::Put {
        key: key.as_bytes().to_vec(),
        value: value.as_bytes().to_vec(),
    }
}

/// Writes `count` keys through the leader, `concurrency` in flight, and
/// returns every latency observed
async fn drive_writes(
    node: &RaftNode,
    prefix: &str,
    start: u64,
    count: u64,
    concurrency: usize,
) -> Vec<Duration> {
    let next = Arc::new(std::sync::atomic::AtomicU64::new(start));
    let end = start + count;
    let mut workers = Vec::with_capacity(concurrency);
    for _ in 0..concurrency {
        let node = node.clone();
        let next = Arc::clone(&next);
        let prefix = prefix.to_string();
        workers.push(tokio::spawn(async move {
            let mut latencies = Vec::new();
            loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= end {
                    break;
                }
                let at = Instant::now();
                node.propose(put(&format!("{prefix}{i}"), &format!("value-{i}")))
                    .await
                    .expect("propose");
                latencies.push(at.elapsed());
            }
            latencies
        }));
    }
    let mut all = Vec::with_capacity(count as usize);
    for worker in workers {
        all.extend(worker.await.expect("worker"));
    }
    all
}

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn millis(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn micros(d: Duration) -> f64 {
    d.as_secs_f64() * 1_000_000.0
}

// ---------------------------------------------------------------------------
// Test 1: Leader election
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_raft_leader_election() {
    zyron_bench_harness::init("raft");
    tprintln!("\n=== Leader Election ===");

    let cluster = Cluster::start(3, RaftConfig::default()).await;
    let at = Instant::now();
    let leader = cluster.wait_leader(Duration::from_secs(1)).await;
    let elapsed = at.elapsed();
    tprintln!("  leader {} elected in {:.2?}", leader, elapsed);
    assert!(
        elapsed < Duration::from_secs(1),
        "election took {elapsed:?}, the gate is one second"
    );

    let term = cluster.node(leader).term();
    assert!(term >= 1, "the leader is still at term zero");
    for id in cluster.live() {
        let node = cluster.node(id);
        assert_eq!(
            node.leader_id(),
            Some(leader),
            "node {id} names a different leader"
        );
        assert_eq!(node.term(), term, "node {id} is on a different term");
        if id == leader {
            assert_eq!(node.role(), RaftRole::Leader);
        } else {
            assert_eq!(
                node.role(),
                RaftRole::Follower,
                "node {id} is not following"
            );
        }
    }
    tprintln!("  all three nodes agree on leader {leader} at term {term}");
    assert!(check_performance_with_unit(
        "Leader Election",
        "First election on a cold group",
        "ms",
        millis(elapsed),
        1000.0,
        false,
    ));
    cluster.shutdown().await;
}

// ---------------------------------------------------------------------------
// Test 2: Log replication
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_raft_log_replication() {
    zyron_bench_harness::init("raft");
    tprintln!("\n=== Log Replication ===");

    let cluster = Cluster::start(3, RaftConfig::default()).await;
    let leader = cluster.wait_leader(Duration::from_secs(2)).await;

    let before = cluster.node(leader).commit_index();
    let at = Instant::now();
    drive_writes(cluster.node(leader), "key", 0, 1000, 32).await;
    let elapsed = at.elapsed();
    let commit = cluster.node(leader).commit_index();
    tprintln!(
        "  1000 commands committed in {:.2?}, commit index {} -> {}",
        elapsed,
        before,
        commit
    );
    assert!(
        commit >= before + 1000,
        "commit index only reached {commit}"
    );

    let last_index = cluster.node(leader).last_log_index();
    cluster
        .wait_applied(&cluster.live(), last_index, Duration::from_secs(30))
        .await;

    let leader_signature = cluster.node(leader).log_signature(1, last_index);
    let leader_digest = cluster.machine(leader).digest();
    for id in cluster.live() {
        let node = cluster.node(id);
        assert_eq!(
            node.last_log_index(),
            last_index,
            "node {id} log ends at {} not {last_index}",
            node.last_log_index()
        );
        assert_eq!(
            node.log_signature(1, last_index),
            leader_signature,
            "node {id} holds different entries"
        );
        assert_eq!(
            cluster.machine(id).digest(),
            leader_digest,
            "node {id} state differs"
        );
        assert_eq!(cluster.machine(id).len(), 1000, "node {id} key count");
    }
    tprintln!("  all three logs are identical through index {last_index}");

    // Spot check that the data is what was written, not merely identical
    for id in cluster.live() {
        let value = cluster.machine(id).get(b"key777").expect("key777");
        assert_eq!(&*value, b"value-777");
    }
    cluster.shutdown().await;
}

// ---------------------------------------------------------------------------
// Test 3: Failover
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_raft_failover() {
    zyron_bench_harness::init("raft");
    tprintln!("\n=== Failover ===");

    let mut cluster = Cluster::start(3, RaftConfig::default()).await;
    let first = cluster.wait_leader(Duration::from_secs(2)).await;

    drive_writes(cluster.node(first), "before", 0, 200, 16).await;
    let committed_before = cluster.node(first).commit_index();
    tprintln!("  leader {first} committed through {committed_before} before the kill");

    let at = Instant::now();
    cluster.kill(first).await;
    let second = cluster.wait_any_leader(Duration::from_secs(5)).await;
    let elapsed = at.elapsed();
    tprintln!(
        "  new leader {second} elected {:.2?} after the kill",
        elapsed
    );
    assert_ne!(second, first);

    // Writes carry on
    drive_writes(cluster.node(second), "after", 0, 200, 16).await;
    let survivors = cluster.live();
    let last_index = cluster.node(second).last_log_index();
    cluster
        .wait_applied(&survivors, last_index, Duration::from_secs(30))
        .await;

    // Nothing committed before the kill was lost
    for id in &survivors {
        let machine = cluster.machine(*id);
        for i in 0..200u64 {
            let key = format!("before{i}");
            assert!(
                machine.get(key.as_bytes()).is_some(),
                "node {id} lost committed key {key}"
            );
        }
        for i in 0..200u64 {
            let key = format!("after{i}");
            assert!(
                machine.get(key.as_bytes()).is_some(),
                "node {id} is missing post-failover key {key}"
            );
        }
    }
    tprintln!("  both survivors hold all 400 keys, none of the pre-failover commits were lost");
    assert!(check_performance_with_unit(
        "Failover",
        "Successor elected after a leader dies",
        "ms",
        millis(elapsed),
        ELECTION_LATENCY_TARGET_MS,
        false,
    ));
    cluster.shutdown().await;
}

// ---------------------------------------------------------------------------
// Test 4: Network partition
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn test_raft_network_partition() {
    zyron_bench_harness::init("raft");
    tprintln!("\n=== Network Partition ===");

    let cluster = Cluster::start(5, RaftConfig::default()).await;
    let first = cluster.wait_leader(Duration::from_secs(3)).await;
    drive_writes(cluster.node(first), "pre", 0, 100, 16).await;

    // Put the sitting leader in the minority, which is the case that matters
    let mut others: Vec<NodeId> = cluster
        .live()
        .into_iter()
        .filter(|id| *id != first)
        .collect();
    let minority = [first, others.remove(0)];
    let majority = [others[0], others[1], others[2]];
    tprintln!("  partitioning into minority {minority:?} and majority {majority:?}");
    cluster.partition(&[&minority, &majority]);

    // The majority elects and keeps writing
    let deadline = Instant::now() + Duration::from_secs(5);
    let second = loop {
        let found = majority
            .iter()
            .copied()
            .find(|id| cluster.node(*id).is_leader());
        if let Some(id) = found {
            break id;
        }
        assert!(Instant::now() < deadline, "the majority never elected");
        tokio::time::sleep(Duration::from_millis(5)).await;
    };
    tprintln!(
        "  majority elected {second} at term {}",
        cluster.node(second).term()
    );
    drive_writes(cluster.node(second), "maj", 0, 100, 16).await;

    // The minority cannot elect, and the old leader has stood down
    let minority_terms: Vec<u64> = minority.iter().map(|id| cluster.node(*id).term()).collect();
    for id in minority {
        assert!(
            !cluster.node(id).is_leader(),
            "minority node {id} believes it leads"
        );
    }
    // A write into the minority cannot commit
    let refused = tokio::time::timeout(
        Duration::from_millis(500),
        cluster.node(minority[0]).propose(put("orphan", "x")),
    )
    .await;
    match refused {
        Err(_) => tprintln!("  the minority write never committed, which is correct"),
        Ok(Err(e)) => tprintln!("  the minority refused the write: {e}"),
        Ok(Ok(index)) => panic!("a minority of two committed index {index}"),
    }
    tprintln!("  minority terms during the partition: {minority_terms:?}");

    // Heal and let the minority catch up
    cluster.heal();
    let leader = cluster.wait_leader(Duration::from_secs(10)).await;
    let last_index = cluster.node(leader).last_log_index();
    cluster
        .wait_applied(&cluster.live(), last_index, Duration::from_secs(30))
        .await;

    let signature = cluster.node(leader).log_signature(1, last_index);
    let digest = cluster.machine(leader).digest();
    for id in cluster.live() {
        assert_eq!(
            cluster.node(id).log_signature(1, last_index),
            signature,
            "node {id} did not converge on the same log"
        );
        assert_eq!(
            cluster.machine(id).digest(),
            digest,
            "node {id} did not converge on the same state"
        );
        assert!(
            cluster.machine(id).get(b"orphan").is_none(),
            "node {id} kept the uncommitted minority write"
        );
        assert!(cluster.machine(id).get(b"maj99").is_some());
        assert!(cluster.machine(id).get(b"pre99").is_some());
    }
    tprintln!("  all five nodes converged on index {last_index} with no inconsistent data");
    cluster.shutdown().await;
}

// ---------------------------------------------------------------------------
// Test 5: Snapshot
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_raft_snapshot() {
    zyron_bench_harness::init("raft");
    tprintln!("\n=== Snapshot ===");

    let mut config = RaftConfig::default();
    config.snapshot_threshold = 4_000;
    let mut cluster = Cluster::start(3, config).await;
    let leader = cluster.wait_leader(Duration::from_secs(2)).await;

    drive_writes(cluster.node(leader), "snap", 0, 10_000, 64).await;
    let last_index = cluster.node(leader).last_log_index();
    cluster
        .wait_applied(&cluster.live(), last_index, Duration::from_secs(60))
        .await;

    // The threshold fires on its own, so the log has already been compacted
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if cluster.node(leader).snapshot_meta().is_some() {
            break;
        }
        assert!(Instant::now() < deadline, "no snapshot was taken");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let meta = cluster.node(leader).snapshot_meta().expect("snapshot");
    let metrics = cluster.node(leader).metrics();
    tprintln!(
        "  snapshot at index {} term {} covering {} bytes, log now starts at {}",
        meta.last_included_index,
        meta.last_included_term,
        meta.size_bytes,
        metrics.first_log_index
    );
    assert!(meta.last_included_index > 0);
    assert_eq!(
        cluster.node(leader).term_at(meta.last_included_index),
        Some(meta.last_included_term),
        "the snapshot is anchored at the wrong term"
    );
    assert!(
        metrics.first_log_index > 1,
        "the log was not compacted behind the snapshot"
    );
    assert_eq!(
        meta.config.voter_count(),
        3,
        "the snapshot did not carry the membership"
    );

    // A brand new node must catch up from the snapshot rather than the log
    let joiner: NodeId = 9;
    let address = cluster.spawn_joiner(joiner).await;
    let at = Instant::now();
    cluster
        .node(leader)
        .add_node(joiner, &address)
        .await
        .expect("add node");
    tprintln!("  node {joiner} joined in {:.2?}", at.elapsed());

    let last_index = cluster.node(leader).last_log_index();
    cluster
        .wait_applied(&[joiner], last_index, Duration::from_secs(120))
        .await;

    let joiner_metrics = cluster.node(joiner).metrics();
    tprintln!(
        "  joiner received {} snapshot bytes and holds {} keys",
        joiner_metrics.snapshot_bytes_received,
        cluster.machine(joiner).len()
    );
    assert!(
        joiner_metrics.snapshot_bytes_received > 0,
        "the joiner replayed the log instead of taking the snapshot"
    );
    assert_eq!(
        joiner_metrics.consensus.snapshots_installed, 1,
        "the joiner did not install a snapshot"
    );
    assert_eq!(
        cluster.machine(joiner).digest(),
        cluster.machine(leader).digest(),
        "the joiner state does not match the leader"
    );
    assert_eq!(cluster.machine(joiner).len(), 10_000);
    tprintln!("  the new node matched the group from a snapshot, not a full replay");
    cluster.shutdown().await;
}

// ---------------------------------------------------------------------------
// Test 5b: the apply loop reads evicted entries back from the log file
// ---------------------------------------------------------------------------

/// A state machine that holds every apply until the test releases it, so
/// entries can commit, become durable and be evicted while nothing applies
struct GatedMachine {
    inner: MemoryStateMachine,
    gate: Arc<AtomicBool>,
}

impl StateMachine for GatedMachine {
    fn apply(&self, index: u64, command: &RaftCommand) -> zyron_common::Result<()> {
        self.inner.apply(index, command)
    }

    fn apply_batch<'a>(
        &'a self,
        entries: &'a [Arc<zyron_raft::log::RaftLogEntry>],
    ) -> zyron_raft::machine::ApplyFuture<'a> {
        Box::pin(async move {
            while self.gate.load(Ordering::Acquire) {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
            self.inner.apply_batch(entries).await
        })
    }

    fn begin_checkpoint(
        &self,
    ) -> zyron_common::Result<Box<dyn zyron_raft::machine::CheckpointSource>> {
        self.inner.begin_checkpoint()
    }

    fn restore(
        &self,
        path: &std::path::Path,
        last_included_index: u64,
    ) -> zyron_common::Result<()> {
        self.inner.restore(path, last_included_index)
    }

    fn applied_index(&self) -> u64 {
        self.inner.applied_index()
    }

    fn digest(&self) -> u64 {
        self.inner.digest()
    }
}

/// A follower further behind on apply than the residency cap keeps in memory
/// must read the entries back from its own log file. Before that read
/// existed, the applier found nothing in memory and waited forever for
/// entries that were on its own disk
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_raft_apply_pages_back_evicted_entries() {
    zyron_bench_harness::init("raft");
    tprintln!("\n=== Apply Pages Back Evicted Entries ===");

    let dir = tempfile::tempdir().expect("tempdir");
    let gate = Arc::new(AtomicBool::new(true));
    let machine = Arc::new(GatedMachine {
        inner: MemoryStateMachine::new(),
        gate: Arc::clone(&gate),
    });
    let mut config = RaftConfig::default();
    // The smallest cap the log accepts, and no snapshots, so the eviction is
    // what the applier meets rather than a compaction hiding it
    config.resident_log_bytes = 64 * 1024;
    config.snapshot_threshold = 0;
    config.snapshot_threshold_bytes = 0;

    let transport = TcpTransport::new(1, TransportConfig::default());
    let node = RaftNode::start(RaftNodeOptions {
        id: 1,
        dir: dir.path().to_path_buf(),
        config,
        bootstrap: ClusterConfig::of_voters([(1, "127.0.0.1:0".to_string())]),
        machine: Arc::clone(&machine) as Arc<dyn StateMachine>,
        transport,
    })
    .await
    .expect("start node");

    node.wait_for_leader(Duration::from_secs(5))
        .await
        .expect("a lone voter takes its own group");

    // One entry goes through so the applier is inside the gate, holding a
    // batch it planned while everything was still resident
    node.propose(put("first", "1")).await.expect("first write");

    // A megabyte of entries against a sixty four kilobyte cap: they commit,
    // the log writer makes them durable, and eviction drops their commands
    // from memory while the applier is still held
    let payload = vec![7u8; 4096];
    for chunk in 0..8 {
        let commands: Vec<RaftCommand> = (0..32)
            .map(|i| RaftCommand::Put {
                key: format!("k{chunk}-{i}").into_bytes(),
                value: payload.clone(),
            })
            .collect();
        node.propose_batch(commands).await.expect("bulk write");
    }
    let last = node.last_log_index();
    assert!(node.last_applied() < last, "the gate did not hold");

    gate.store(false, Ordering::Release);
    let deadline = Instant::now() + Duration::from_secs(20);
    while node.last_applied() < last {
        assert!(
            Instant::now() < deadline,
            "apply stalled at {} of {last}, the evicted entries were never read back",
            node.last_applied()
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let paged = node.metrics().consensus.entries_paged_for_apply;
    assert!(
        paged > 0,
        "nothing was paged in, so the test never actually met the eviction"
    );
    tprintln!(
        "  {} entries applied, {} of them read back from the log file",
        last,
        paged
    );
    node.shutdown().await;
}

// ---------------------------------------------------------------------------
// Test 5c: the replicator reads evicted entries back for a follower behind
// ---------------------------------------------------------------------------

/// A follower that fell behind by more than the leader keeps in memory is
/// caught up from the leader's log file, not from a snapshot, and in order.
/// A batch read back from disk is placed after the read, so the follower is
/// held while it is read and nothing built after it can arrive first
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_raft_replication_pages_back_evicted_entries() {
    zyron_bench_harness::init("raft");
    tprintln!("\n=== Replication Pages Back Evicted Entries ===");

    let mut config = RaftConfig::default();
    // The smallest cap the log accepts, and no snapshots, so the follower
    // can only be caught up from entries the leader has already evicted
    config.resident_log_bytes = 64 * 1024;
    config.snapshot_threshold = 0;
    config.snapshot_threshold_bytes = 0;
    let cluster = Cluster::start(3, config).await;
    let leader = cluster.wait_leader(Duration::from_secs(5)).await;
    let behind = cluster
        .live()
        .into_iter()
        .find(|id| *id != leader)
        .expect("a follower");
    let rest: Vec<NodeId> = cluster
        .live()
        .into_iter()
        .filter(|id| *id != behind)
        .collect();

    // The follower is cut off while the leader and the other follower commit
    // half a megabyte, eight times the leader's residency cap
    cluster.partition(&[&[behind], &rest]);
    let payload = vec![9u8; 4096];
    for chunk in 0..8 {
        let commands: Vec<RaftCommand> = (0..16)
            .map(|i| RaftCommand::Put {
                key: format!("page{chunk}-{i}").into_bytes(),
                value: payload.clone(),
            })
            .collect();
        cluster
            .node(leader)
            .propose_batch(commands)
            .await
            .expect("bulk write with a majority");
    }
    let last_index = cluster.node(leader).last_log_index();
    let held_back = last_index - cluster.node(behind).last_log_index();
    assert!(
        held_back >= 100,
        "the follower only fell {held_back} behind"
    );

    // Healed, the follower is brought level from the file
    cluster.heal();
    cluster
        .wait_applied(&[behind], last_index, Duration::from_secs(30))
        .await;

    let paged = cluster.node(leader).metrics().consensus.entries_paged_in;
    let follower = cluster.node(behind).metrics();
    tprintln!(
        "  the follower was {held_back} entries behind, the leader read {paged} of them back from its log file, the follower refused {} appends and installed {} snapshots",
        follower.consensus.appends_rejected,
        follower.consensus.snapshots_installed
    );
    assert!(
        paged > 0,
        "nothing was paged in, so the follower never met the leader's eviction"
    );
    // While the follower was cut off the leader probed it once per heartbeat
    // interval, reading back whatever was pending each time. A leader that
    // rebuilt the batch on every proposal would read the backlog back dozens
    // of times over
    assert!(
        paged <= held_back * 16,
        "the leader read {paged} entries back for a follower {held_back} behind"
    );
    assert_eq!(
        follower.consensus.snapshots_installed, 0,
        "the follower was caught up by a snapshot rather than the log"
    );
    assert_eq!(
        follower.consensus.appends_rejected, 0,
        "the follower refused appends, a paged batch was placed out of order"
    );
    assert_eq!(
        cluster.node(behind).log_signature(1, last_index),
        cluster.node(leader).log_signature(1, last_index),
        "the follower holds different entries"
    );
    assert_eq!(
        cluster.machine(behind).digest(),
        cluster.machine(leader).digest(),
        "the follower state differs"
    );
    cluster.shutdown().await;
}

// ---------------------------------------------------------------------------
// Test 6: Membership change
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_raft_membership_change() {
    zyron_bench_harness::init("raft");
    tprintln!("\n=== Membership Change ===");

    let mut cluster = Cluster::start(3, RaftConfig::default()).await;
    let leader = cluster.wait_leader(Duration::from_secs(2)).await;
    drive_writes(cluster.node(leader), "m", 0, 100, 16).await;

    let joiner: NodeId = 4;
    let address = cluster.spawn_joiner(joiner).await;
    let at = Instant::now();
    cluster
        .node(leader)
        .add_node(joiner, &address)
        .await
        .expect("add node");
    tprintln!("  node 4 went learner then voter in {:.2?}", at.elapsed());

    let config = cluster.node(leader).cluster_config();
    assert_eq!(config.voter_count(), 4, "node 4 did not become a voter");
    assert!(config.is_voter(joiner));
    for id in cluster.live() {
        let seen = cluster.node(id).cluster_config();
        assert_eq!(
            seen.voter_count(),
            4,
            "node {id} does not see a group of four"
        );
    }

    // A four node group still commits, and now needs three of four
    drive_writes(cluster.node(leader), "four", 0, 200, 16).await;
    let last_index = cluster.node(leader).last_log_index();
    cluster
        .wait_applied(&cluster.live(), last_index, Duration::from_secs(30))
        .await;
    for id in cluster.live() {
        assert!(cluster.machine(id).get(b"four199").is_some());
    }
    tprintln!("  the four node group committed 200 more writes");

    // Take node 1 out. If it is the leader it stands down as part of the change
    let leader = cluster.wait_any_leader(Duration::from_secs(5)).await;
    let at = Instant::now();
    cluster.node(leader).remove_node(1).await.expect("remove");
    tprintln!("  node 1 removed in {:.2?}", at.elapsed());
    cluster.kill(1).await;

    let leader = cluster.wait_any_leader(Duration::from_secs(10)).await;
    let config = cluster.node(leader).cluster_config();
    assert_eq!(config.voter_count(), 3, "the group is not back to three");
    assert!(!config.contains(1), "node 1 is still in the configuration");

    drive_writes(cluster.node(leader), "three", 0, 100, 16).await;
    let last_index = cluster.node(leader).last_log_index();
    let remaining: Vec<NodeId> = cluster.live().into_iter().filter(|id| *id != 1).collect();
    cluster
        .wait_applied(&remaining, last_index, Duration::from_secs(30))
        .await;
    tprintln!("  the three node group carried on committing");
    cluster.shutdown().await;
}

// ---------------------------------------------------------------------------
// Test 7: Linearizable read on a follower
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_raft_linearizable_read() {
    zyron_bench_harness::init("raft");
    tprintln!("\n=== Linearizable Read ===");

    let cluster = Cluster::start(3, RaftConfig::default()).await;
    let leader = cluster.wait_leader(Duration::from_secs(2)).await;
    let follower = cluster
        .live()
        .into_iter()
        .find(|id| *id != leader)
        .expect("a follower");

    let mut latencies = Vec::new();
    for round in 0..50u64 {
        let value = format!("v{round}");
        cluster
            .node(leader)
            .propose(put("x", &value))
            .await
            .expect("write");

        let at = Instant::now();
        let index = cluster
            .node(follower)
            .linearizable_read_index()
            .await
            .expect("read index");
        let observed = cluster.machine(follower).get(b"x").expect("x is present");
        latencies.push(at.elapsed());

        assert_eq!(
            String::from_utf8_lossy(&observed),
            value,
            "the follower served a stale value at read index {index}"
        );
    }
    latencies.sort();
    tprintln!(
        "  follower ReadIndex over 50 write-then-read rounds: p50 {:.2?}, p99 {:.2?}",
        percentile(&latencies, 0.50),
        percentile(&latencies, 0.99),
    );
    assert!(check_performance_with_unit(
        "Linearizable Read",
        "Follower read after a write, p99",
        "ms",
        millis(percentile(&latencies, 0.99)),
        FOLLOWER_READ_TARGET_MS,
        false,
    ));

    // A read on the leader itself is also linearizable, and rides the lease
    let before = cluster
        .node(leader)
        .metrics()
        .consensus
        .read_index_leases_used;
    for _ in 0..20 {
        cluster
            .node(leader)
            .linearizable_read_index()
            .await
            .expect("leader read");
    }
    let after = cluster
        .node(leader)
        .metrics()
        .consensus
        .read_index_leases_used;
    tprintln!(
        "  {} of 20 leader reads were answered from the lease",
        after - before
    );
    assert!(after > before, "no leader read used the lease");
    cluster.shutdown().await;
}

// ---------------------------------------------------------------------------
// Test 8: Pre-vote
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_raft_pre_vote() {
    zyron_bench_harness::init("raft");
    tprintln!("\n=== Pre-Vote ===");

    let cluster = Cluster::start(3, RaftConfig::default()).await;
    let leader = cluster.wait_leader(Duration::from_secs(2)).await;
    let isolated = cluster
        .live()
        .into_iter()
        .find(|id| *id != leader)
        .expect("a follower");
    let term_before = cluster.node(isolated).term();
    let leader_term_before = cluster.node(leader).term();
    // The count is cumulative from the node's start, and a follower may have
    // campaigned and lost while the group first formed, so only elections
    // started after the isolation count
    let elections_before = cluster.node(isolated).metrics().consensus.elections_started;
    tprintln!("  isolating follower {isolated} at term {term_before}");

    let rest: Vec<NodeId> = cluster
        .live()
        .into_iter()
        .filter(|id| *id != isolated)
        .collect();
    cluster.partition(&[&[isolated], &rest]);

    // Far longer than the election timeout, so a node without pre-vote would
    // have raised its term many times over
    tokio::time::sleep(Duration::from_secs(3)).await;

    let term_after = cluster.node(isolated).term();
    let elections = cluster.node(isolated).metrics().consensus.elections_started - elections_before;
    let pre_votes = cluster.node(isolated).metrics().consensus.pre_votes_started;
    tprintln!(
        "  isolated node is still at term {term_after} after {pre_votes} pre-vote rounds and {elections} elections"
    );
    assert_eq!(
        term_after, term_before,
        "the isolated node raised its term from {term_before} to {term_after}"
    );
    assert_eq!(
        elections, 0,
        "the isolated node started {elections} real elections"
    );
    assert!(
        pre_votes > 3,
        "the isolated node did not even try, only {pre_votes} pre-vote rounds"
    );

    // The rest of the group never noticed
    assert_eq!(
        cluster.node(leader).term(),
        leader_term_before,
        "the leader's term moved while a follower was away"
    );
    assert!(cluster.node(leader).is_leader(), "the leader was deposed");

    cluster.heal();
    let after_leader = cluster.wait_leader(Duration::from_secs(3)).await;
    assert_eq!(
        after_leader, leader,
        "healing the partition deposed the sitting leader"
    );
    assert_eq!(
        cluster.node(leader).term(),
        leader_term_before,
        "the term moved when the partition healed"
    );
    assert_eq!(cluster.node(isolated).leader_id(), Some(leader));

    // And it is a working follower again
    cluster
        .node(leader)
        .propose(put("rejoined", "yes"))
        .await
        .expect("write");
    let index = cluster.node(leader).last_log_index();
    cluster
        .wait_applied(&[isolated], index, Duration::from_secs(10))
        .await;
    tprintln!("  node {isolated} rejoined at term {leader_term_before} without an election");
    cluster.shutdown().await;
}

// ---------------------------------------------------------------------------
// Test 9: The membership SQL, end to end
// ---------------------------------------------------------------------------

/// `ALTER CLUSTER` is parsed and carried out the way the statement handler
/// does it, so the SQL surface and the consensus API cannot drift apart.
///
/// The handler itself needs a whole server around it. What it does with a
/// parsed statement is three lines, and those three lines are what this runs
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_raft_cluster_membership_sql() {
    use zyron_parser::ast::AlterClusterOperation;
    use zyron_raft::node_id_for_name;

    zyron_bench_harness::init("raft");
    tprintln!("\n=== ALTER CLUSTER ===");

    let mut cluster = Cluster::start(3, RaftConfig::default()).await;
    let leader = cluster.wait_leader(Duration::from_secs(2)).await;
    drive_writes(cluster.node(leader), "sql", 0, 100, 16).await;

    // A name maps to the same id on every node without anyone being told
    let joiner = node_id_for_name("node-4");
    assert_eq!(joiner, node_id_for_name("node-4"));
    assert_ne!(joiner, node_id_for_name("node-5"));
    assert_ne!(joiner, 0);
    let address = cluster.spawn_joiner(joiner).await;

    let sql = format!("ALTER CLUSTER ADD NODE 'node-4' AT '{address}'");
    let statement = zyron_parser::parse(&sql)
        .expect("parse")
        .into_iter()
        .next()
        .expect("one statement");
    let zyron_parser::Statement::AlterCluster(stmt) = statement else {
        panic!("ALTER CLUSTER did not parse as a cluster change");
    };
    match &stmt.operation {
        AlterClusterOperation::AddNode { name, address } => {
            assert_eq!(name, "node-4");
            cluster
                .node(leader)
                .add_node(node_id_for_name(name), address)
                .await
                .expect("add node");
        }
        other => panic!("expected AddNode, got {other:?}"),
    }
    let config = cluster.node(leader).cluster_config();
    assert_eq!(config.voter_count(), 4, "node-4 did not become a voter");
    assert!(config.is_voter(joiner));
    tprintln!("  ADD NODE brought node-4 in as id {joiner}");

    // The four node group commits, and node-4 has the earlier writes
    drive_writes(cluster.node(leader), "post", 0, 100, 16).await;
    let last_index = cluster.node(leader).last_log_index();
    cluster
        .wait_applied(&[joiner], last_index, Duration::from_secs(30))
        .await;
    assert_eq!(
        cluster.machine(joiner).digest(),
        cluster.machine(leader).digest()
    );

    let statement = zyron_parser::parse("ALTER CLUSTER REMOVE NODE 'node-4'")
        .expect("parse")
        .into_iter()
        .next()
        .expect("one statement");
    let zyron_parser::Statement::AlterCluster(stmt) = statement else {
        panic!("ALTER CLUSTER did not parse as a cluster change");
    };
    match &stmt.operation {
        AlterClusterOperation::RemoveNode { name } => {
            assert_eq!(name, "node-4");
            cluster
                .node(leader)
                .remove_node(node_id_for_name(name))
                .await
                .expect("remove node");
        }
        other => panic!("expected RemoveNode, got {other:?}"),
    }
    let config = cluster.node(leader).cluster_config();
    assert_eq!(config.voter_count(), 3, "the group did not shrink");
    assert!(!config.contains(joiner));
    tprintln!("  REMOVE NODE took node-4 back out");

    // And the group still commits without it
    cluster.discard(joiner).await;
    drive_writes(cluster.node(leader), "after", 0, 100, 16).await;
    cluster.shutdown().await;
}

/// A group description that could not produce a working node is refused at
/// startup rather than on the first election
#[test]
fn test_raft_cluster_config_is_validated() {
    use zyron_server::config::{ClusterPeerSection, ClusterSection};

    zyron_bench_harness::init("raft");
    tprintln!("\n=== Cluster Configuration ===");

    let peer = |name: &str, address: &str| ClusterPeerSection {
        name: name.to_string(),
        address: address.to_string(),
    };

    // Off is always valid, whatever else is in the section
    let off = ClusterSection::default();
    assert!(!off.enabled);
    off.validate().expect("a section that is off");

    let valid = ClusterSection {
        enabled: true,
        node_name: "node-1".into(),
        listen: "0.0.0.0:5434".into(),
        peers: vec![peer("node-1", "h1:5434"), peer("node-2", "h2:5434")],
        ..ClusterSection::default()
    };
    valid.validate().expect("a complete section");

    let mut nameless = valid.clone();
    nameless.node_name = String::new();
    assert!(nameless.validate().is_err(), "a node with no name");

    let mut stranger = valid.clone();
    stranger.node_name = "node-9".into();
    assert!(
        stranger.validate().is_err(),
        "a node that is not in its own group"
    );

    let mut empty = valid.clone();
    empty.peers.clear();
    assert!(empty.validate().is_err(), "a group with no members");

    let mut addressless = valid.clone();
    addressless.peers[1].address = String::new();
    assert!(addressless.validate().is_err(), "a peer with no address");

    let mut tableless = valid.clone();
    tableless.replicated_table = String::new();
    assert!(tableless.validate().is_err(), "no table to apply into");
    tprintln!("  every incomplete group description is refused");
}

// ---------------------------------------------------------------------------
// Test 10: Write throughput and latency on three nodes
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn test_raft_write_throughput() {
    zyron_bench_harness::init("raft");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Write Throughput, Three Nodes ===");

    const TOTAL: u64 = 100_000;
    const LATENCY_WRITES: u64 = 5_000;

    let mut throughputs = Vec::with_capacity(VALIDATION_RUNS);
    let mut p99s = Vec::with_capacity(VALIDATION_RUNS);
    let mut saturated_p99s = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        tprintln!("--- Run {}/{} ---", run + 1, VALIDATION_RUNS);
        let mut config = RaftConfig::default();
        // Snapshotting mid measurement would time part of a checkpoint
        config.snapshot_threshold = 0;
        let cluster = Cluster::start(3, config).await;
        let leader = cluster.wait_leader(Duration::from_secs(5)).await;

        // What one write costs, with the group well short of saturation
        let mut latencies = drive_writes(
            cluster.node(leader),
            "lat",
            0,
            LATENCY_WRITES,
            LATENCY_CONCURRENCY,
        )
        .await;
        latencies.sort();
        // The shape of the tail says whether a p99 is a few stalls or a
        // shifted distribution
        tprintln!(
            "  {} writes at {} outstanding: p50 {:.2?}, p90 {:.2?}, p99 {:.2?}, max {:.2?}",
            format_with_commas(LATENCY_WRITES as f64),
            LATENCY_CONCURRENCY,
            percentile(&latencies, 0.50),
            percentile(&latencies, 0.90),
            percentile(&latencies, 0.99),
            latencies.last().copied().unwrap_or_default(),
        );
        p99s.push(millis(percentile(&latencies, 0.99)));

        // What the group commits per second, driven to saturation
        let before = cluster.node(leader).metrics().consensus;
        let at = Instant::now();
        let mut saturated =
            drive_writes(cluster.node(leader), "p", 0, TOTAL, THROUGHPUT_CONCURRENCY).await;
        let elapsed = at.elapsed();
        let throughput = TOTAL as f64 / elapsed.as_secs_f64();
        saturated.sort();
        tprintln!(
            "  {} writes at {} outstanding in {:.2?}, {} ops/sec, p50 {:.2?}, p99 {:.2?}",
            format_with_commas(TOTAL as f64),
            THROUGHPUT_CONCURRENCY,
            elapsed,
            format_with_commas(throughput),
            percentile(&saturated, 0.50),
            percentile(&saturated, 0.99),
        );
        // How the entries were carried says whether the pipeline moved
        // batches or a message per proposal, and what the followers took
        // against what was sent says whether anything was sent twice
        let after = cluster.node(leader).metrics().consensus;
        let heartbeats = after.heartbeats_built - before.heartbeats_built;
        let carriers = after.commit_carriers_built - before.commit_carriers_built;
        let carrying = after.appends_built - before.appends_built - heartbeats - carriers;
        let entries = after.entries_replicated - before.entries_replicated;
        tprintln!(
            "  the leader sent {} messages with entries, {:.1} entries each, {} heartbeats and {} commit carriers",
            format_with_commas(carrying as f64),
            entries as f64 / carrying.max(1) as f64,
            heartbeats,
            carriers
        );
        for id in cluster.live().into_iter().filter(|id| *id != leader) {
            let m = cluster.node(id).metrics().consensus;
            tprintln!(
                "  follower {id} appended {} entries, refused {} appends, truncated {} times",
                format_with_commas(m.entries_appended as f64),
                m.appends_rejected,
                m.log_truncations
            );
            // A healthy group refuses nothing. A refusal here means a batch
            // reached the follower ahead of the one before it, which is the
            // ordering the transport and the replicator together promise
            assert_eq!(
                m.appends_rejected, 0,
                "follower {id} refused appends, the pipeline delivered a batch out of order"
            );
            assert_eq!(
                m.log_truncations, 0,
                "follower {id} truncated its log under a leader that never changed"
            );
        }
        throughputs.push(throughput);
        saturated_p99s.push(millis(percentile(&saturated, 0.99)));
        // The slowest flush on each node says whether a latency spike was
        // the disk's
        for id in cluster.live() {
            let m = cluster.node(id).metrics();
            tprintln!(
                "  node {id}: {} log fsyncs, the slowest {:.2?}",
                m.log_fsyncs,
                Duration::from_micros(m.log_fsync_max_us)
            );
        }
        assert_eq!(
            cluster.machine(leader).len(),
            (TOTAL + LATENCY_WRITES) as usize
        );
        cluster.shutdown().await;
    }

    let r = validate_metric_with_unit(
        "Write Throughput",
        "Committed writes on a three node group",
        " ops/sec",
        throughputs,
        WRITE_THROUGHPUT_TARGET_OPS,
        true,
    );
    assert!(r.passed, "write throughput below target");
    assert!(!r.regression_detected, "write throughput regression");

    let r = validate_metric_with_unit(
        "Write Throughput",
        "Write latency p99, group not saturated",
        "ms",
        p99s,
        WRITE_LATENCY_P99_TARGET_MS,
        false,
    );
    assert!(r.passed, "write latency p99 above target");
    assert!(!r.regression_detected, "write latency regression");

    // Recorded without a bound. At saturation this is the client's queue
    // depth divided by the commit rate, so a target on it would be a target
    // on how hard the benchmark pushes
    record_metric(
        "Write Throughput",
        "Write latency p99, group saturated",
        "ms",
        saturated_p99s,
    );
}

// ---------------------------------------------------------------------------
// Test 11: Leader election latency
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_raft_election_latency() {
    zyron_bench_harness::init("raft");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Leader Election Latency ===");

    /// Ten elections per run, so one unlucky split vote does not decide the
    /// figure
    const ELECTIONS_PER_RUN: usize = 10;

    let mut means = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        tprintln!("--- Run {}/{} ---", run + 1, VALIDATION_RUNS);
        let mut times = Vec::with_capacity(ELECTIONS_PER_RUN);
        for _ in 0..ELECTIONS_PER_RUN {
            let mut cluster = Cluster::start(3, RaftConfig::default()).await;
            let first = cluster.wait_leader(Duration::from_secs(5)).await;
            let at = Instant::now();
            cluster.kill(first).await;
            cluster.wait_any_leader(Duration::from_secs(10)).await;
            times.push(millis(at.elapsed()));
            cluster.shutdown().await;
        }
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        let worst = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        tprintln!(
            "  {ELECTIONS_PER_RUN} elections: mean {:.1}ms, worst {:.1}ms",
            mean,
            worst
        );
        means.push(mean);
    }

    let r = validate_metric_with_unit(
        "Election Latency",
        "Leader replaced after a kill",
        "ms",
        means,
        ELECTION_LATENCY_TARGET_MS,
        false,
    );
    assert!(r.passed, "election latency above target");
    assert!(!r.regression_detected, "election latency regression");
}

// ---------------------------------------------------------------------------
// Test 12: Replication lag
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn test_raft_replication_lag() {
    zyron_bench_harness::init("raft");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Replication Lag, Leader Commit to Follower Apply ===");

    const WRITES: u64 = 500;
    let mut p99s = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        tprintln!("--- Run {}/{} ---", run + 1, VALIDATION_RUNS);
        let mut config = RaftConfig::default();
        config.snapshot_threshold = 0;
        let cluster = Cluster::start(3, config).await;
        let leader = cluster.wait_leader(Duration::from_secs(5)).await;
        let follower = cluster
            .live()
            .into_iter()
            .find(|id| *id != leader)
            .expect("a follower");

        // Warm the connections and the log so the measurement is steady state
        drive_writes(cluster.node(leader), "warm", 0, 500, 32).await;
        let leader_commit = cluster.node(leader).commit_index();
        let behind = leader_commit.saturating_sub(cluster.node(follower).last_applied());
        let catch_up = Instant::now();
        while cluster.node(follower).last_applied() < leader_commit {
            tokio::task::yield_now().await;
        }
        tprintln!(
            "  after the warm-up the follower was {behind} entries behind and caught up in {:.2?}",
            catch_up.elapsed()
        );

        let mut lags = Vec::with_capacity(WRITES as usize);
        for i in 0..WRITES {
            let before = cluster.node(leader).metrics().consensus;
            let follower_before = cluster.node(follower).metrics();
            let follower_fsync_before = follower_before.log_fsync_max_us;
            let follower_before = follower_before.consensus;
            let index = cluster
                .node(leader)
                .propose(put(&format!("lag{i}"), "v"))
                .await
                .expect("write");
            let at = Instant::now();
            let follower_log_at_commit = cluster.node(follower).metrics().last_log_index;
            let mut commit_seen: Option<Duration> = None;
            while cluster.node(follower).last_applied() < index {
                if commit_seen.is_none() && cluster.node(follower).commit_index() >= index {
                    commit_seen = Some(at.elapsed());
                }
                tokio::task::yield_now().await;
                assert!(
                    at.elapsed() < Duration::from_secs(10),
                    "the follower never applied index {index}"
                );
            }
            let lag = at.elapsed();
            // A stall is rare and short, and where it lands says whether the
            // leader was slow to say the entry committed or the follower was
            // slow to apply what it knew. A follower that held the log short
            // of the entry at the commit was still flushing it, and its
            // slowest fsync moving past the stall names the disk
            if lag > Duration::from_millis(5) {
                let after = cluster.node(leader).metrics().consensus;
                let follower_after = cluster.node(follower).metrics();
                let follower_fsync_after = follower_after.log_fsync_max_us;
                let follower_after = follower_after.consensus;
                tprintln!(
                    "  write {i} at index {index}: the follower learned the commit after {:.2?} and applied it after {:.2?}, the leader built {} appends, {} heartbeats, {} commit carriers meanwhile, the follower held log {} at the commit, appended {} entries, rejected {} appends, and its slowest fsync went from {:.2?} to {:.2?}",
                    commit_seen.unwrap_or(lag),
                    lag,
                    after.appends_built - before.appends_built,
                    after.heartbeats_built - before.heartbeats_built,
                    after.commit_carriers_built - before.commit_carriers_built,
                    follower_log_at_commit,
                    follower_after.entries_appended - follower_before.entries_appended,
                    follower_after.appends_rejected - follower_before.appends_rejected,
                    Duration::from_micros(follower_fsync_before),
                    Duration::from_micros(follower_fsync_after)
                );
            }
            lags.push(lag);
        }
        lags.sort();
        tprintln!(
            "  {WRITES} writes: p50 {:.2?}, p99 {:.2?}, max {:.2?}",
            percentile(&lags, 0.50),
            percentile(&lags, 0.99),
            lags.last().copied().unwrap_or_default(),
        );
        p99s.push(millis(percentile(&lags, 0.99)));
        cluster.shutdown().await;
    }

    let r = validate_metric_with_unit(
        "Replication Lag",
        "Leader commit to follower apply, p99",
        "ms",
        p99s,
        REPLICATION_LAG_TARGET_MS,
        false,
    );
    assert!(r.passed, "replication lag above target");
    assert!(!r.regression_detected, "replication lag regression");
}

// ---------------------------------------------------------------------------
// Test 13: Log append latency
// ---------------------------------------------------------------------------

#[test]
fn test_raft_log_append_latency() {
    use zyron_raft::log::{RaftLog, RaftLogEntry};

    zyron_bench_harness::init("raft");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Log Append Latency ===");

    const ENTRIES: u64 = 50_000;
    let mut means = Vec::with_capacity(VALIDATION_RUNS);
    let mut durable_rates = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        tprintln!("--- Run {}/{} ---", run + 1, VALIDATION_RUNS);
        let dir = tempfile::tempdir().expect("tempdir");
        let mut log = RaftLog::open(dir.path(), 64 * 1024 * 1024).expect("open");
        let mut appends = Vec::with_capacity(ENTRIES as usize);
        let started = Instant::now();
        for index in 1..=ENTRIES {
            let entry = RaftLogEntry::new(
                1,
                index,
                RaftCommand::Put {
                    key: format!("k{index}").into_bytes(),
                    value: vec![7u8; 96],
                },
            );
            let at = Instant::now();
            log.append(entry).expect("append");
            log.flush_pending().expect("flush");
            appends.push(at.elapsed());
        }
        while !log.is_durable() {
            std::thread::sleep(Duration::from_micros(200));
        }
        let total = started.elapsed();
        appends.sort();
        let mean = appends.iter().copied().map(micros).sum::<f64>() / ENTRIES as f64;
        let rate = ENTRIES as f64 / total.as_secs_f64();
        tprintln!(
            "  append mean {:.2}us, p99 {:.2}us, {} entries durable in {:.2?} ({} entries/sec including fsync)",
            mean,
            micros(percentile(&appends, 0.99)),
            format_with_commas(ENTRIES as f64),
            total,
            format_with_commas(rate),
        );
        means.push(mean);
        durable_rates.push(rate);
    }

    let r = validate_metric_with_unit(
        "Log Append",
        "One append into the replicated log",
        "us",
        means,
        LOG_APPEND_TARGET_US,
        false,
    );
    assert!(r.passed, "log append latency above target");
    assert!(!r.regression_detected, "log append regression");

    // Recorded without a target: the group commit rate is set by the disk,
    // and a bound on it would be a bound on the hardware
    record_metric(
        "Log Append",
        "Entries made durable by group commit",
        " entries/sec",
        durable_rates,
    );
}

// ---------------------------------------------------------------------------
// Test 14: Follower read latency
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_raft_follower_read_latency() {
    zyron_bench_harness::init("raft");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Follower Read Latency ===");

    const READS: usize = 1000;
    let mut p99s = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        tprintln!("--- Run {}/{} ---", run + 1, VALIDATION_RUNS);
        let cluster = Cluster::start(3, RaftConfig::default()).await;
        let leader = cluster.wait_leader(Duration::from_secs(5)).await;
        let follower = cluster
            .live()
            .into_iter()
            .find(|id| *id != leader)
            .expect("a follower");
        drive_writes(cluster.node(leader), "r", 0, 200, 16).await;

        let mut latencies = Vec::with_capacity(READS);
        for _ in 0..READS {
            let at = Instant::now();
            cluster
                .node(follower)
                .linearizable_read_index()
                .await
                .expect("read");
            latencies.push(at.elapsed());
        }
        latencies.sort();
        tprintln!(
            "  {READS} reads: p50 {:.2?}, p99 {:.2?}",
            percentile(&latencies, 0.50),
            percentile(&latencies, 0.99),
        );
        p99s.push(millis(percentile(&latencies, 0.99)));
        cluster.shutdown().await;
    }

    let r = validate_metric_with_unit(
        "Follower Read",
        "ReadIndex round trip and local apply wait, p99",
        "ms",
        p99s,
        FOLLOWER_READ_TARGET_MS,
        false,
    );
    assert!(r.passed, "follower read latency above target");
    assert!(!r.regression_detected, "follower read regression");
}

// ---------------------------------------------------------------------------
// Test 15: Membership change latency
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_raft_membership_change_latency() {
    zyron_bench_harness::init("raft");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Membership Change Latency ===");

    let mut times = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        tprintln!("--- Run {}/{} ---", run + 1, VALIDATION_RUNS);
        let mut cluster = Cluster::start(3, RaftConfig::default()).await;
        let leader = cluster.wait_leader(Duration::from_secs(5)).await;
        drive_writes(cluster.node(leader), "mc", 0, 500, 32).await;
        let address = cluster.spawn_joiner(4).await;
        let at = Instant::now();
        cluster
            .node(leader)
            .add_node(4, &address)
            .await
            .expect("add node");
        let elapsed = at.elapsed();
        tprintln!("  learner to voter in {:.2?}", elapsed);
        assert_eq!(cluster.node(leader).cluster_config().voter_count(), 4);
        times.push(elapsed.as_secs_f64());
        cluster.shutdown().await;
    }

    let r = validate_metric_with_unit(
        "Membership Change",
        "Add a node as learner and promote it",
        "s",
        times,
        MEMBERSHIP_CHANGE_TARGET_SEC,
        false,
    );
    assert!(r.passed, "membership change above target");
    assert!(!r.regression_detected, "membership change regression");
}

// ---------------------------------------------------------------------------
// Test 16: A gigabyte of state, snapshotted and shipped
// ---------------------------------------------------------------------------

/// Heavy on purpose: the point is the real checkpoint write and the real
/// chunked transfer over a socket, not a scaled down stand-in.
///
/// The state is built once and the two measured operations are repeated over
/// it, because building a gigabyte through consensus five times measures the
/// write path rather than the snapshot path. Each transfer round starts a node
/// that holds none of the state and is discarded afterwards, so only two
/// copies of the gigabyte exist at any moment
#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn test_raft_snapshot_one_gigabyte() {
    zyron_bench_harness::init("raft");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    // A leader that steps down mid-run says why at info level, and this is
    // the one test where that has happened, so its output carries the log
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_test_writer()
        .try_init();
    tprintln!("\n=== Snapshot Create and Transfer, One Gigabyte ===");

    if zyron_bench_harness::skip_expensive("Snapshot 1GB", "a gigabyte of state through consensus")
    {
        return;
    }

    let target_bytes = SNAPSHOT_VALUE_BYTES as u64 * SNAPSHOT_KEYS;
    let mut config = RaftConfig::default();
    // Snapshotting is driven by hand here so the create can be timed. Both
    // automatic triggers are off, the byte one because a gigabyte of state is
    // a gigabyte of log and it would fire while the timed create runs
    config.snapshot_threshold = 0;
    config.snapshot_threshold_bytes = 0;
    config.max_batch_bytes = 4 * 1024 * 1024;
    config.max_batch_entries = 64;
    config.propose_timeout = Duration::from_secs(300);
    let mut cluster = Cluster::start(1, config).await;
    let leader = cluster.wait_leader(Duration::from_secs(5)).await;

    let filler = vec![0xA5u8; SNAPSHOT_VALUE_BYTES];
    let at = Instant::now();
    for chunk in 0..(SNAPSHOT_KEYS / 64) {
        let commands: Vec<RaftCommand> = (0..64)
            .map(|i| {
                let index = chunk * 64 + i;
                RaftCommand::Put {
                    key: format!("big{index:08}").into_bytes(),
                    value: filler.clone(),
                }
            })
            .collect();
        cluster
            .node(leader)
            .propose_batch(commands)
            .await
            .expect("propose");
    }
    let index = cluster.node(leader).last_log_index();
    cluster
        .wait_applied(&[leader], index, Duration::from_secs(300))
        .await;
    tprintln!(
        "  built {:.2} GiB of state in {:.2?}",
        target_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        at.elapsed()
    );

    let mut creates = Vec::with_capacity(VALIDATION_RUNS);
    let mut transfers = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        tprintln!("--- Run {}/{} ---", run + 1, VALIDATION_RUNS);

        let at = Instant::now();
        let meta = cluster
            .node(leader)
            .create_snapshot()
            .await
            .expect("create snapshot");
        let create = at.elapsed();
        tprintln!(
            "  created a {:.2} GiB snapshot in {:.2?} ({:.0} MiB/sec)",
            meta.size_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            create,
            meta.size_bytes as f64 / (1024.0 * 1024.0) / create.as_secs_f64()
        );
        assert!(
            meta.size_bytes >= target_bytes,
            "the snapshot is only {} bytes",
            meta.size_bytes
        );
        creates.push(create.as_secs_f64());

        // A node with none of the state joins, and the log behind the
        // snapshot has been compacted away, so it can only be caught up by
        // the snapshot itself
        let joiner: NodeId = 100 + run as u64;
        let address = cluster.spawn_joiner(joiner).await;
        let at = Instant::now();
        cluster
            .node(leader)
            .add_node(joiner, &address)
            .await
            .expect("add node");
        let transfer = at.elapsed();
        let received = cluster.node(joiner).metrics().snapshot_bytes_received;
        tprintln!(
            "  transferred and installed {} bytes in {:.2?} ({:.0} MiB/sec)",
            format_with_commas(received as f64),
            transfer,
            received as f64 / (1024.0 * 1024.0) / transfer.as_secs_f64()
        );
        assert!(
            received >= target_bytes,
            "only {received} bytes reached the new node"
        );
        assert_eq!(
            cluster.machine(joiner).digest(),
            cluster.machine(leader).digest(),
            "the receiver state does not match"
        );
        assert_eq!(cluster.machine(joiner).len(), SNAPSHOT_KEYS as usize);
        transfers.push(transfer.as_secs_f64());

        // Put the group back to one node so the next round transfers to a
        // receiver that again holds nothing
        cluster
            .node(leader)
            .remove_node(joiner)
            .await
            .expect("remove");
        cluster.discard(joiner).await;
    }

    let r = validate_metric_with_unit(
        "Snapshot 1GB",
        "Checkpoint one gigabyte of state",
        "s",
        creates,
        SNAPSHOT_CREATE_TARGET_SEC,
        false,
    );
    assert!(r.passed, "snapshot create above target");
    assert!(!r.regression_detected, "snapshot create regression");

    let r = validate_metric_with_unit(
        "Snapshot 1GB",
        "Stream one gigabyte to a node that has none of it",
        "s",
        transfers,
        SNAPSHOT_TRANSFER_TARGET_SEC,
        false,
    );
    assert!(r.passed, "snapshot transfer above target");
    assert!(!r.regression_detected, "snapshot transfer regression");

    cluster.shutdown().await;
}

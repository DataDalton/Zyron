//! Carrying consensus messages between nodes.
//!
//! ## One connection per peer, many calls on it
//!
//! Consensus talks to the same handful of peers continuously and forever. A
//! connection per call would put a handshake in front of every heartbeat, so
//! each peer gets a persistent connection instead, and calls are multiplexed
//! over it by request id: the writer stamps an id, the reader matches replies
//! back to the caller that is waiting. Replies may come back in any order,
//! which is what lets sixteen AppendEntries be outstanding at once.
//!
//! ## Two lanes, so a snapshot cannot silence a heartbeat
//!
//! A snapshot chunk is a megabyte and a heartbeat is fifty two bytes. Sharing
//! one connection means the heartbeat waits behind whatever is already in the
//! socket, and a node transferring a gigabyte would look dead to its own group
//! for as long as the transfer took. Snapshots therefore get a second
//! connection of their own, and control traffic never queues behind bulk.
//!
//! ## The receiving end group commits
//!
//! A leader keeps several AppendEntries in flight. If the receiver handled
//! them one at a time, each would wait for its own fsync before the next was
//! even read, and the pipeline would buy nothing: sixteen messages would cost
//! sixteen fsyncs and the group would run at the disk's sync rate rather than
//! its bandwidth.
//!
//! So a connection is read by one task and answered by another. The reader
//! takes frames off the socket as fast as they arrive; the answering task
//! drains everything queued, hands each to the handler in arrival order, and
//! only then waits for durability. The handler's synchronous half has already
//! put every one of those entries into the log by that point, so the log
//! writer covers all of them with a single fsync, and the replies go back in
//! the order the requests came.
//!
//! ## Failure is a fact about a call, not about a node
//!
//! A call that cannot be delivered returns [`RaftRpcError`] rather than
//! retrying inside the transport. Consensus already knows what to do about a
//! peer that did not answer, and it will try again on its own schedule with a
//! message built from current state. A transport that retried on its own would
//! deliver a stale AppendEntries after the leader had already moved on.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Mutex as AsyncMutex, mpsc, oneshot};
use zyron_common::error::ZyronError;

use crate::NodeId;
use crate::codec::{Cursor, put_str};
use crate::election::{RequestVoteReply, RequestVoteRequest};
use crate::replication::{
    AppendEntriesReply, AppendEntriesRequest, ReadIndexReply, ReadIndexRequest,
};
use crate::snapshot::{InstallSnapshotReply, InstallSnapshotRequest};

/// Identifies a consensus frame, so a stray connection to the wrong port
/// fails on the first four bytes rather than on a decoded field
const FRAME_MAGIC: u32 = 0x5A52_4654;
/// magic 4, payload length 4, request id 8, kind 1, flags 1, reserved 2
const FRAME_HEADER_LEN: usize = 20;

const KIND_REQUEST_VOTE: u8 = 1;
const KIND_APPEND_ENTRIES: u8 = 2;
const KIND_INSTALL_SNAPSHOT: u8 = 3;
const KIND_READ_INDEX: u8 = 4;

const FLAG_REPLY: u8 = 0x01;
const FLAG_ERROR: u8 = 0x02;

/// How many frames one connection may hold between arrival and reply.
///
/// This is the width of the receiver's group commit. It bounds what one peer
/// can make this node hold, and the reader task blocks on a full queue, which
/// is the backpressure a flooded receiver needs
const PIPELINE_DEPTH: usize = 64;

/// Control traffic: votes, appends, heartbeats, reads
const LANE_CONTROL: u8 = 0;
/// Bulk traffic: snapshot chunks
const LANE_BULK: u8 = 1;

/// Why a call produced no answer.
///
/// Concrete rather than boxed, because the caller decides what to do from the
/// variant alone: a peer that was unreachable is worth trying again on the
/// next heartbeat, and a payload this build cannot parse is a version
/// mismatch that retrying will not fix
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RaftRpcError {
    /// The peer could not be reached
    Unreachable { node: NodeId, reason: String },
    /// The peer did not answer inside the deadline
    TimedOut { node: NodeId, after_ms: u64 },
    /// The peer answered with something this build cannot read
    Malformed { reason: String },
    /// The peer answered, and refused
    Refused { reason: String },
    /// This node is shutting down and will not send anything else
    Shutdown,
    /// The peer is not in this node's directory
    UnknownPeer { node: NodeId },
}

impl RaftRpcError {
    /// Whether the same call could succeed later
    pub fn transient(&self) -> bool {
        matches!(
            self,
            RaftRpcError::Unreachable { .. } | RaftRpcError::TimedOut { .. }
        )
    }
}

impl std::fmt::Display for RaftRpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RaftRpcError::Unreachable { node, reason } => {
                write!(f, "node {node} could not be reached: {reason}")
            }
            RaftRpcError::TimedOut { node, after_ms } => {
                write!(f, "node {node} did not answer after {after_ms}ms")
            }
            RaftRpcError::Malformed { reason } => write!(f, "malformed consensus frame: {reason}"),
            RaftRpcError::Refused { reason } => write!(f, "the peer refused: {reason}"),
            RaftRpcError::Shutdown => f.write_str("this node is shutting down"),
            RaftRpcError::UnknownPeer { node } => {
                write!(f, "node {node} has no address in this node's directory")
            }
        }
    }
}

impl std::error::Error for RaftRpcError {}

impl From<RaftRpcError> for ZyronError {
    fn from(e: RaftRpcError) -> Self {
        ZyronError::RaftTransport(e.to_string())
    }
}

/// What a call returns once it has crossed a node boundary
pub type RaftFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T, RaftRpcError>> + Send + 'a>>;

/// What one node asks another.
///
/// The consensus driver is written against this rather than against sockets,
/// so a deployment that carries these calls another way replaces one file
pub trait RaftTransport: Send + Sync {
    fn send_request_vote(
        &self,
        to: NodeId,
        req: RequestVoteRequest,
    ) -> RaftFuture<'_, RequestVoteReply>;

    fn send_append_entries(
        &self,
        to: NodeId,
        req: AppendEntriesRequest,
    ) -> RaftFuture<'_, AppendEntriesReply>;

    fn send_install_snapshot(
        &self,
        to: NodeId,
        req: InstallSnapshotRequest,
    ) -> RaftFuture<'_, InstallSnapshotReply>;

    fn send_read_index(&self, to: NodeId, req: ReadIndexRequest) -> RaftFuture<'_, ReadIndexReply>;

    /// Records where a node answers, called when membership changes
    fn set_address(&self, node: NodeId, address: &str);

    /// Drops whatever is held for a node that has left the group
    fn forget(&self, node: NodeId);

    fn stats(&self) -> TransportStats;
}

/// What a node answers when asked.
///
/// The futures are boxed and the synchronous part of each call runs before the
/// future is returned. That matters for AppendEntries: the entries are taken
/// into the log in the order the frames arrived, and only the wait for the
/// fsync is deferred into the future
pub type RaftHandlerFuture<'a, T> =
    Pin<Box<dyn Future<Output = Result<T, ZyronError>> + Send + 'a>>;

pub trait RaftRequestHandler: Send + Sync {
    fn on_request_vote(&self, req: RequestVoteRequest) -> RaftHandlerFuture<'_, RequestVoteReply>;

    fn on_append_entries(
        &self,
        req: AppendEntriesRequest,
    ) -> RaftHandlerFuture<'_, AppendEntriesReply>;

    fn on_install_snapshot(
        &self,
        req: InstallSnapshotRequest,
    ) -> RaftHandlerFuture<'_, InstallSnapshotReply>;

    fn on_read_index(&self, req: ReadIndexRequest) -> RaftHandlerFuture<'_, ReadIndexReply>;
}

/// Timers and bounds the transport enforces.
#[derive(Debug, Clone)]
pub struct TransportConfig {
    pub connect_timeout: Duration,
    /// How long a control call waits. Sized for a message the peer answers
    /// from memory
    pub rpc_timeout: Duration,
    /// How long a snapshot chunk waits.
    ///
    /// A separate figure because the work behind the two is not comparable. A
    /// heartbeat is answered out of memory; a snapshot chunk is a megabyte
    /// written and fsynced, and the last chunk of all is a whole checkpoint
    /// loaded into the state machine before the reply can honestly say it
    /// installed. Holding both to the heartbeat's deadline made the sender
    /// abandon transfers that were succeeding and start them again
    pub bulk_rpc_timeout: Duration,
    /// Largest frame either side will read, which bounds what a peer can make
    /// this node allocate
    pub max_frame_bytes: usize,
}

impl Default for TransportConfig {
    fn default() -> Self {
        Self {
            connect_timeout: Duration::from_millis(500),
            rpc_timeout: Duration::from_millis(1000),
            bulk_rpc_timeout: Duration::from_secs(120),
            max_frame_bytes: 32 * 1024 * 1024,
        }
    }
}

/// Counters for the consensus transport.
#[derive(Debug, Default, Clone)]
pub struct TransportStats {
    pub calls_sent: u64,
    pub calls_failed: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub connections_opened: u64,
    pub connections_lost: u64,
}

#[derive(Debug, Default)]
struct StatsInner {
    calls_sent: AtomicU64,
    calls_failed: AtomicU64,
    bytes_sent: AtomicU64,
    bytes_received: AtomicU64,
    connections_opened: AtomicU64,
    connections_lost: AtomicU64,
}

impl StatsInner {
    fn snapshot(&self) -> TransportStats {
        TransportStats {
            calls_sent: self.calls_sent.load(Ordering::Relaxed),
            calls_failed: self.calls_failed.load(Ordering::Relaxed),
            bytes_sent: self.bytes_sent.load(Ordering::Relaxed),
            bytes_received: self.bytes_received.load(Ordering::Relaxed),
            connections_opened: self.connections_opened.load(Ordering::Relaxed),
            connections_lost: self.connections_lost.load(Ordering::Relaxed),
        }
    }
}

// ---------------------------------------------------------------------------
// Framing
// ---------------------------------------------------------------------------

struct Frame {
    request_id: u64,
    kind: u8,
    flags: u8,
    payload: Vec<u8>,
}

/// A buffer with room for the frame header already at the front.
///
/// Bodies are encoded straight into this, and the header is stamped over the
/// reserved bytes once the length is known. Building the header separately and
/// concatenating meant every message was copied a second time on its way out,
/// which for a megabyte snapshot chunk or a full replication batch is the same
/// order of work as sending it
fn frame_buffer(body_hint: usize) -> Vec<u8> {
    let mut buf = Vec::with_capacity(FRAME_HEADER_LEN + body_hint);
    buf.resize(FRAME_HEADER_LEN, 0);
    buf
}

/// Stamps the header over the reserved prefix of a [`frame_buffer`]
fn stamp_frame(buf: &mut [u8], request_id: u64, kind: u8, flags: u8) {
    let body_len = (buf.len() - FRAME_HEADER_LEN) as u32;
    buf[0..4].copy_from_slice(&FRAME_MAGIC.to_le_bytes());
    buf[4..8].copy_from_slice(&body_len.to_le_bytes());
    buf[8..16].copy_from_slice(&request_id.to_le_bytes());
    buf[16] = kind;
    buf[17] = flags;
    buf[18..20].copy_from_slice(&0u16.to_le_bytes());
}

async fn read_frame<R>(reader: &mut R, max_bytes: usize) -> std::io::Result<Option<Frame>>
where
    R: tokio::io::AsyncRead + Unpin,
{
    let mut header = [0u8; FRAME_HEADER_LEN];
    match reader.read_exact(&mut header).await {
        Ok(_) => {}
        Err(e)
            if e.kind() == std::io::ErrorKind::UnexpectedEof
                || e.kind() == std::io::ErrorKind::ConnectionReset
                || e.kind() == std::io::ErrorKind::ConnectionAborted =>
        {
            return Ok(None);
        }
        Err(e) => return Err(e),
    }
    let magic = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
    if magic != FRAME_MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("consensus frame magic is {magic:#010x}"),
        ));
    }
    let len = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;
    if len > max_bytes {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("consensus frame is {len} bytes, the limit is {max_bytes}"),
        ));
    }
    let request_id = u64::from_le_bytes([
        header[8], header[9], header[10], header[11], header[12], header[13], header[14],
        header[15],
    ]);
    let mut payload = vec![0u8; len];
    reader.read_exact(&mut payload).await?;
    Ok(Some(Frame {
        request_id,
        kind: header[16],
        flags: header[17],
        payload,
    }))
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

type PendingMap = Arc<parking_lot::Mutex<HashMap<u64, oneshot::Sender<Frame>>>>;

struct Conn {
    tx: mpsc::UnboundedSender<Vec<u8>>,
    pending: PendingMap,
    next_id: AtomicU64,
    alive: Arc<AtomicBool>,
}

impl Conn {
    fn fail_all(&self) {
        self.alive.store(false, Ordering::Release);
        self.pending.lock().clear();
    }
}

struct PeerLane {
    node: NodeId,
    address: parking_lot::RwLock<String>,
    conn: AsyncMutex<Option<Arc<Conn>>>,
    config: TransportConfig,
    /// The deadline for calls on this lane, which is the bulk one on the lane
    /// snapshots use
    call_timeout: Duration,
    stats: Arc<StatsInner>,
}

impl PeerLane {
    async fn connection(&self) -> Result<Arc<Conn>, RaftRpcError> {
        {
            let guard = self.conn.lock().await;
            if let Some(conn) = guard.as_ref() {
                if conn.alive.load(Ordering::Acquire) {
                    return Ok(Arc::clone(conn));
                }
            }
        }
        let mut guard = self.conn.lock().await;
        // Another caller may have reconnected while this one waited
        if let Some(conn) = guard.as_ref() {
            if conn.alive.load(Ordering::Acquire) {
                return Ok(Arc::clone(conn));
            }
        }
        let address = self.address.read().clone();
        if address.is_empty() {
            return Err(RaftRpcError::UnknownPeer { node: self.node });
        }
        let stream =
            tokio::time::timeout(self.config.connect_timeout, TcpStream::connect(&address))
                .await
                .map_err(|_| RaftRpcError::Unreachable {
                    node: self.node,
                    reason: format!("connect to {address} timed out"),
                })?
                .map_err(|e| RaftRpcError::Unreachable {
                    node: self.node,
                    reason: format!("connect to {address}: {e}"),
                })?;
        // Consensus messages are small and latency bound, so waiting to
        // coalesce them is exactly the wrong trade
        let _ = stream.set_nodelay(true);

        let (mut read_half, mut write_half) = stream.into_split();
        let (tx, mut rx) = mpsc::unbounded_channel::<Vec<u8>>();
        let pending: PendingMap = Arc::new(parking_lot::Mutex::new(HashMap::new()));
        let alive = Arc::new(AtomicBool::new(true));
        let conn = Arc::new(Conn {
            tx,
            pending: Arc::clone(&pending),
            next_id: AtomicU64::new(1),
            alive: Arc::clone(&alive),
        });

        let writer_alive = Arc::clone(&alive);
        let writer_pending = Arc::clone(&pending);
        let writer_stats = Arc::clone(&self.stats);
        tokio::spawn(async move {
            while let Some(bytes) = rx.recv().await {
                if write_half.write_all(&bytes).await.is_err() {
                    break;
                }
                writer_stats
                    .bytes_sent
                    .fetch_add(bytes.len() as u64, Ordering::Relaxed);
            }
            writer_alive.store(false, Ordering::Release);
            writer_pending.lock().clear();
            let _ = write_half.shutdown().await;
        });

        let reader_alive = Arc::clone(&alive);
        let reader_pending = Arc::clone(&pending);
        let reader_stats = Arc::clone(&self.stats);
        let max_frame = self.config.max_frame_bytes;
        tokio::spawn(async move {
            loop {
                match read_frame(&mut read_half, max_frame).await {
                    Ok(Some(frame)) => {
                        reader_stats.bytes_received.fetch_add(
                            (FRAME_HEADER_LEN + frame.payload.len()) as u64,
                            Ordering::Relaxed,
                        );
                        let waiter = reader_pending.lock().remove(&frame.request_id);
                        if let Some(waiter) = waiter {
                            let _ = waiter.send(frame);
                        }
                    }
                    Ok(None) | Err(_) => break,
                }
            }
            reader_alive.store(false, Ordering::Release);
            reader_pending.lock().clear();
        });

        self.stats
            .connections_opened
            .fetch_add(1, Ordering::Relaxed);
        *guard = Some(Arc::clone(&conn));
        Ok(conn)
    }

    /// Sends one call. `framed` is a [`frame_buffer`] whose body is already
    /// written, so the header is stamped in place rather than concatenated
    async fn call(&self, kind: u8, framed: Vec<u8>) -> Result<Vec<u8>, RaftRpcError> {
        self.stats.calls_sent.fetch_add(1, Ordering::Relaxed);
        let result = self.call_inner(kind, framed).await;
        if result.is_err() {
            self.stats.calls_failed.fetch_add(1, Ordering::Relaxed);
        }
        result
    }

    async fn call_inner(&self, kind: u8, mut framed: Vec<u8>) -> Result<Vec<u8>, RaftRpcError> {
        let conn = self.connection().await?;
        let id = conn.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        conn.pending.lock().insert(id, tx);
        stamp_frame(&mut framed, id, kind, 0);
        if conn.tx.send(framed).is_err() {
            conn.fail_all();
            self.stats.connections_lost.fetch_add(1, Ordering::Relaxed);
            return Err(RaftRpcError::Unreachable {
                node: self.node,
                reason: "connection closed before the call was written".into(),
            });
        }
        match tokio::time::timeout(self.call_timeout, rx).await {
            Ok(Ok(frame)) => {
                if frame.flags & FLAG_ERROR != 0 {
                    let mut c = Cursor::new(&frame.payload);
                    let reason = c.string().unwrap_or_else(|_| "unreadable".into());
                    return Err(RaftRpcError::Refused { reason });
                }
                Ok(frame.payload)
            }
            Ok(Err(_)) => {
                self.stats.connections_lost.fetch_add(1, Ordering::Relaxed);
                Err(RaftRpcError::Unreachable {
                    node: self.node,
                    reason: "connection dropped while the call was outstanding".into(),
                })
            }
            Err(_) => {
                conn.pending.lock().remove(&id);
                Err(RaftRpcError::TimedOut {
                    node: self.node,
                    after_ms: self.call_timeout.as_millis() as u64,
                })
            }
        }
    }
}

/// The consensus transport over TCP.
pub struct TcpTransport {
    local: NodeId,
    directory: parking_lot::RwLock<HashMap<NodeId, String>>,
    lanes: parking_lot::Mutex<HashMap<(NodeId, u8), Arc<PeerLane>>>,
    config: TransportConfig,
    stats: Arc<StatsInner>,
    shutdown: AtomicBool,
}

impl TcpTransport {
    pub fn new(local: NodeId, config: TransportConfig) -> Arc<Self> {
        Arc::new(Self {
            local,
            directory: parking_lot::RwLock::new(HashMap::new()),
            lanes: parking_lot::Mutex::new(HashMap::new()),
            config,
            stats: Arc::new(StatsInner::default()),
            shutdown: AtomicBool::new(false),
        })
    }

    /// Builds a transport that already knows where everyone answers
    pub fn with_directory(
        local: NodeId,
        addresses: impl IntoIterator<Item = (NodeId, String)>,
        config: TransportConfig,
    ) -> Arc<Self> {
        let transport = Self::new(local, config);
        for (node, address) in addresses {
            transport.set_address(node, &address);
        }
        transport
    }

    pub fn local_id(&self) -> NodeId {
        self.local
    }

    /// Stops new calls, so a shutting down node does not open a connection to
    /// a peer it is about to stop talking to
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Release);
        self.lanes.lock().clear();
    }

    fn lane(&self, node: NodeId, lane: u8) -> Result<Arc<PeerLane>, RaftRpcError> {
        if self.shutdown.load(Ordering::Acquire) {
            return Err(RaftRpcError::Shutdown);
        }
        let address = match self.directory.read().get(&node) {
            Some(a) => a.clone(),
            None => return Err(RaftRpcError::UnknownPeer { node }),
        };
        let mut lanes = self.lanes.lock();
        let entry = lanes.entry((node, lane)).or_insert_with(|| {
            Arc::new(PeerLane {
                node,
                address: parking_lot::RwLock::new(address.clone()),
                conn: AsyncMutex::new(None),
                call_timeout: if lane == LANE_BULK {
                    self.config.bulk_rpc_timeout
                } else {
                    self.config.rpc_timeout
                },
                config: self.config.clone(),
                stats: Arc::clone(&self.stats),
            })
        });
        Ok(Arc::clone(entry))
    }

    async fn call<T, F>(
        lane: Result<Arc<PeerLane>, RaftRpcError>,
        kind: u8,
        framed: Vec<u8>,
        decode: F,
    ) -> Result<T, RaftRpcError>
    where
        F: FnOnce(&[u8]) -> Result<T, zyron_common::error::ZyronError>,
    {
        let lane = lane?;
        let bytes = lane.call(kind, framed).await?;
        decode(&bytes).map_err(|e| RaftRpcError::Malformed {
            reason: e.to_string(),
        })
    }
}

impl RaftTransport for TcpTransport {
    fn send_request_vote(
        &self,
        to: NodeId,
        req: RequestVoteRequest,
    ) -> RaftFuture<'_, RequestVoteReply> {
        let lane = self.lane(to, LANE_CONTROL);
        let mut payload = frame_buffer(48);
        req.encode(&mut payload);
        Box::pin(Self::call(lane, KIND_REQUEST_VOTE, payload, |b| {
            RequestVoteReply::decode(&mut Cursor::new(b))
        }))
    }

    fn send_append_entries(
        &self,
        to: NodeId,
        req: AppendEntriesRequest,
    ) -> RaftFuture<'_, AppendEntriesReply> {
        let lane = self.lane(to, LANE_CONTROL);
        let mut payload = frame_buffer(req.encoded_len());
        req.encode(&mut payload);
        Box::pin(Self::call(lane, KIND_APPEND_ENTRIES, payload, |b| {
            AppendEntriesReply::decode(&mut Cursor::new(b))
        }))
    }

    fn send_install_snapshot(
        &self,
        to: NodeId,
        req: InstallSnapshotRequest,
    ) -> RaftFuture<'_, InstallSnapshotReply> {
        let lane = self.lane(to, LANE_BULK);
        let mut payload = frame_buffer(req.data.len() + 128);
        req.encode(&mut payload);
        Box::pin(Self::call(lane, KIND_INSTALL_SNAPSHOT, payload, |b| {
            InstallSnapshotReply::decode(&mut Cursor::new(b))
        }))
    }

    fn send_read_index(&self, to: NodeId, req: ReadIndexRequest) -> RaftFuture<'_, ReadIndexReply> {
        let lane = self.lane(to, LANE_CONTROL);
        let mut payload = frame_buffer(16);
        req.encode(&mut payload);
        Box::pin(Self::call(lane, KIND_READ_INDEX, payload, |b| {
            ReadIndexReply::decode(&mut Cursor::new(b))
        }))
    }

    fn set_address(&self, node: NodeId, address: &str) {
        let changed = {
            let mut dir = self.directory.write();
            match dir.get(&node) {
                Some(existing) if existing == address => false,
                _ => {
                    dir.insert(node, address.to_string());
                    true
                }
            }
        };
        if changed {
            // An address change means the old connection points at the wrong
            // place, so it is dropped rather than reused
            let mut lanes = self.lanes.lock();
            lanes.retain(|(id, _), _| *id != node);
        }
    }

    fn forget(&self, node: NodeId) {
        self.directory.write().remove(&node);
        self.lanes.lock().retain(|(id, _), _| *id != node);
    }

    fn stats(&self) -> TransportStats {
        self.stats.snapshot()
    }
}

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

/// Accepts consensus connections and answers them.
pub struct RaftServer;

impl RaftServer {
    /// Serves until `shutdown` is notified.
    ///
    /// Frames on one connection are answered in the order they arrived,
    /// because AppendEntries from one leader is an ordered stream and
    /// answering out of order would have the leader believing entries landed
    /// that a later frame had not yet delivered
    pub async fn serve(
        listener: TcpListener,
        handler: Arc<dyn RaftRequestHandler>,
        config: TransportConfig,
        shutdown: Arc<tokio::sync::Notify>,
    ) {
        loop {
            let accepted = tokio::select! {
                biased;
                _ = shutdown.notified() => break,
                accepted = listener.accept() => accepted,
            };
            let (stream, peer) = match accepted {
                Ok(pair) => pair,
                Err(e) => {
                    tracing::warn!(error = %e, "consensus listener could not accept");
                    continue;
                }
            };
            let _ = stream.set_nodelay(true);
            let handler = Arc::clone(&handler);
            let config = config.clone();
            let shutdown = Arc::clone(&shutdown);
            tokio::spawn(async move {
                if let Err(e) = serve_connection(stream, handler, config, shutdown).await {
                    tracing::debug!(peer = %peer, error = %e, "consensus connection ended");
                }
            });
        }
    }
}

/// One reply, once its durability wait is over
type ReplyFuture<'a> = Pin<Box<dyn Future<Output = (u8, Vec<u8>)> + Send + 'a>>;

async fn serve_connection(
    stream: TcpStream,
    handler: Arc<dyn RaftRequestHandler>,
    config: TransportConfig,
    shutdown: Arc<tokio::sync::Notify>,
) -> std::io::Result<()> {
    let (mut read_half, mut write_half) = stream.into_split();
    let (frames_tx, mut frames) = mpsc::channel::<Frame>(PIPELINE_DEPTH);
    let max_frame = config.max_frame_bytes;

    // Reading is its own task so that waiting for an fsync never stops the
    // socket being drained. It is also what makes the read cancellation safe:
    // nothing ever abandons a half read frame
    let reader = tokio::spawn(async move {
        loop {
            match read_frame(&mut read_half, max_frame).await {
                Ok(Some(frame)) => {
                    if frames_tx.send(frame).await.is_err() {
                        return;
                    }
                }
                Ok(None) | Err(_) => return,
            }
        }
    });

    let mut batch: Vec<(u64, u8, ReplyFuture<'_>)> = Vec::with_capacity(PIPELINE_DEPTH);
    let result = loop {
        let first = tokio::select! {
            biased;
            _ = shutdown.notified() => break Ok(()),
            frame = frames.recv() => frame,
        };
        let Some(first) = first else {
            break Ok(());
        };
        batch.push(begin_reply(&handler, first));
        // Everything already on the socket joins this group commit. The
        // handler's synchronous half runs here, in arrival order, so the
        // entries are in the log before any of the waits below start
        while let Ok(frame) = frames.try_recv() {
            batch.push(begin_reply(&handler, frame));
            if batch.len() >= PIPELINE_DEPTH {
                break;
            }
        }
        let mut failed = None;
        for (request_id, kind, reply) in batch.drain(..) {
            let (flags, mut framed) = reply.await;
            if failed.is_some() {
                continue;
            }
            stamp_frame(&mut framed, request_id, kind, flags);
            if let Err(e) = write_half.write_all(&framed).await {
                failed = Some(e);
            }
        }
        if let Some(e) = failed {
            break Err(e);
        }
    };
    reader.abort();
    result
}

/// Runs the synchronous half of one request and returns what is left to wait
/// for.
///
/// Deliberately not an async function: the decode and the handler call have to
/// happen at call time so that a run of AppendEntries reaches the log in the
/// order it arrived, whatever order the waits are polled in afterwards
fn begin_reply<'a>(
    handler: &'a Arc<dyn RaftRequestHandler>,
    frame: Frame,
) -> (u64, u8, ReplyFuture<'a>) {
    let Frame {
        request_id,
        kind,
        payload,
        ..
    } = frame;
    let reply: ReplyFuture<'a> = match kind {
        KIND_REQUEST_VOTE => match RequestVoteRequest::decode(&mut Cursor::new(&payload)) {
            Ok(req) => {
                let fut = handler.on_request_vote(req);
                Box::pin(async move { finish(fut.await) })
            }
            Err(e) => refused(e.to_string()),
        },
        KIND_APPEND_ENTRIES => match AppendEntriesRequest::decode(&mut Cursor::new(&payload)) {
            Ok(req) => {
                let fut = handler.on_append_entries(req);
                Box::pin(async move { finish(fut.await) })
            }
            Err(e) => refused(e.to_string()),
        },
        KIND_INSTALL_SNAPSHOT => match InstallSnapshotRequest::decode(&mut Cursor::new(&payload)) {
            Ok(req) => {
                let fut = handler.on_install_snapshot(req);
                Box::pin(async move { finish(fut.await) })
            }
            Err(e) => refused(e.to_string()),
        },
        KIND_READ_INDEX => match ReadIndexRequest::decode(&mut Cursor::new(&payload)) {
            Ok(req) => {
                let fut = handler.on_read_index(req);
                Box::pin(async move { finish(fut.await) })
            }
            Err(e) => refused(e.to_string()),
        },
        other => refused(format!("consensus frame kind {other} is not known")),
    };
    (request_id, kind, reply)
}

/// Turns a handler answer into a frame body
fn finish<T: Encodable>(result: Result<T, ZyronError>) -> (u8, Vec<u8>) {
    match result {
        Ok(reply) => {
            let mut out = frame_buffer(64);
            reply.encode(&mut out);
            (FLAG_REPLY, out)
        }
        Err(e) => error_body(e.to_string()),
    }
}

fn refused<'a>(reason: String) -> ReplyFuture<'a> {
    Box::pin(async move { error_body(reason) })
}

fn error_body(reason: String) -> (u8, Vec<u8>) {
    let mut err = frame_buffer(reason.len() + 8);
    put_str(&mut err, &reason);
    (FLAG_REPLY | FLAG_ERROR, err)
}

/// What a reply can be written as. Implemented by the four reply types so one
/// generic finish covers them all
pub trait Encodable {
    fn encode(&self, buf: &mut Vec<u8>);
}

impl Encodable for RequestVoteReply {
    fn encode(&self, buf: &mut Vec<u8>) {
        RequestVoteReply::encode(self, buf)
    }
}

impl Encodable for AppendEntriesReply {
    fn encode(&self, buf: &mut Vec<u8>) {
        AppendEntriesReply::encode(self, buf)
    }
}

impl Encodable for InstallSnapshotReply {
    fn encode(&self, buf: &mut Vec<u8>) {
        InstallSnapshotReply::encode(self, buf)
    }
}

impl Encodable for ReadIndexReply {
    fn encode(&self, buf: &mut Vec<u8>) {
        ReadIndexReply::encode(self, buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::log::{RaftCommand, RaftLogEntry};

    struct Echo {
        seen: Arc<AtomicU64>,
    }

    impl RaftRequestHandler for Echo {
        fn on_request_vote(
            &self,
            req: RequestVoteRequest,
        ) -> RaftHandlerFuture<'_, RequestVoteReply> {
            self.seen.fetch_add(1, Ordering::Relaxed);
            Box::pin(async move {
                Ok(RequestVoteReply {
                    term: req.term,
                    vote_granted: true,
                    pre_vote: req.pre_vote,
                    voter_id: 99,
                })
            })
        }

        fn on_append_entries(
            &self,
            req: AppendEntriesRequest,
        ) -> RaftHandlerFuture<'_, AppendEntriesReply> {
            self.seen.fetch_add(1, Ordering::Relaxed);
            Box::pin(async move {
                Ok(AppendEntriesReply {
                    term: req.term,
                    success: true,
                    match_index: req.prev_log_index + req.entries.len() as u64,
                    hint_index: 0,
                    read_round: req.read_round,
                    follower_id: 99,
                })
            })
        }

        fn on_install_snapshot(
            &self,
            req: InstallSnapshotRequest,
        ) -> RaftHandlerFuture<'_, InstallSnapshotReply> {
            Box::pin(async move {
                Ok(InstallSnapshotReply {
                    term: req.term,
                    success: true,
                    bytes_received: req.offset + req.data.len() as u64,
                    follower_id: 99,
                })
            })
        }

        fn on_read_index(&self, req: ReadIndexRequest) -> RaftHandlerFuture<'_, ReadIndexReply> {
            Box::pin(async move {
                Ok(ReadIndexReply {
                    term: req.term,
                    success: true,
                    read_index: 4242,
                    leader_id: Some(99),
                })
            })
        }
    }

    struct Refuser;

    impl RaftRequestHandler for Refuser {
        fn on_request_vote(
            &self,
            _req: RequestVoteRequest,
        ) -> RaftHandlerFuture<'_, RequestVoteReply> {
            Box::pin(async move { Err(ZyronError::Internal("no votes today".into())) })
        }
        fn on_append_entries(
            &self,
            _req: AppendEntriesRequest,
        ) -> RaftHandlerFuture<'_, AppendEntriesReply> {
            Box::pin(async move { Err(ZyronError::Internal("no".into())) })
        }
        fn on_install_snapshot(
            &self,
            _req: InstallSnapshotRequest,
        ) -> RaftHandlerFuture<'_, InstallSnapshotReply> {
            Box::pin(async move { Err(ZyronError::Internal("no".into())) })
        }
        fn on_read_index(&self, _req: ReadIndexRequest) -> RaftHandlerFuture<'_, ReadIndexReply> {
            Box::pin(async move { Err(ZyronError::Internal("no".into())) })
        }
    }

    async fn start(handler: Arc<dyn RaftRequestHandler>) -> (String, Arc<tokio::sync::Notify>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let address = listener.local_addr().expect("addr").to_string();
        let shutdown = Arc::new(tokio::sync::Notify::new());
        let stop = Arc::clone(&shutdown);
        tokio::spawn(async move {
            RaftServer::serve(listener, handler, TransportConfig::default(), stop).await;
        });
        (address, shutdown)
    }

    #[tokio::test]
    async fn every_call_round_trips_over_tcp() {
        let seen = Arc::new(AtomicU64::new(0));
        let (address, _stop) = start(Arc::new(Echo {
            seen: Arc::clone(&seen),
        }))
        .await;
        let transport = TcpTransport::with_directory(1, [(2, address)], TransportConfig::default());

        let vote = transport
            .send_request_vote(
                2,
                RequestVoteRequest {
                    term: 5,
                    candidate_id: 1,
                    last_log_index: 3,
                    last_log_term: 4,
                    pre_vote: true,
                },
            )
            .await
            .expect("vote");
        assert!(vote.vote_granted);
        assert!(vote.pre_vote);

        let append = transport
            .send_append_entries(
                2,
                AppendEntriesRequest {
                    term: 5,
                    leader_id: 1,
                    prev_log_index: 10,
                    prev_log_term: 4,
                    entries: (11..=15)
                        .map(|i| Arc::new(RaftLogEntry::new(5, i, RaftCommand::Noop)))
                        .collect(),
                    leader_commit: 9,
                    read_round: 7,
                },
            )
            .await
            .expect("append");
        assert_eq!(append.match_index, 15);
        assert_eq!(append.read_round, 7);

        let read = transport
            .send_read_index(2, ReadIndexRequest { term: 5, from: 1 })
            .await
            .expect("read");
        assert_eq!(read.read_index, 4242);

        let install = transport
            .send_install_snapshot(
                2,
                InstallSnapshotRequest {
                    term: 5,
                    leader_id: 1,
                    last_included_index: 900,
                    last_included_term: 4,
                    config: crate::membership::ClusterConfig::of_voters([(1, "a:1".to_string())]),
                    offset: 100,
                    data: vec![1u8; 4096],
                    done: true,
                },
            )
            .await
            .expect("install");
        assert_eq!(install.bytes_received, 4196);

        // Four calls, and the control lane opened one connection for three of
        // them while the snapshot took the bulk lane
        let stats = transport.stats();
        assert_eq!(stats.calls_sent, 4);
        assert_eq!(stats.calls_failed, 0);
        assert_eq!(stats.connections_opened, 2);
    }

    #[tokio::test]
    async fn calls_multiplex_over_one_connection() {
        let seen = Arc::new(AtomicU64::new(0));
        let (address, _stop) = start(Arc::new(Echo {
            seen: Arc::clone(&seen),
        }))
        .await;
        let transport = TcpTransport::with_directory(1, [(2, address)], TransportConfig::default());
        let mut calls = Vec::new();
        for i in 0..64u64 {
            calls.push(transport.send_append_entries(
                2,
                AppendEntriesRequest {
                    term: 1,
                    leader_id: 1,
                    prev_log_index: i,
                    prev_log_term: 1,
                    entries: Vec::new(),
                    leader_commit: 0,
                    read_round: i,
                },
            ));
        }
        for (i, call) in calls.into_iter().enumerate() {
            let reply = call.await.expect("reply");
            assert_eq!(reply.read_round, i as u64);
        }
        assert_eq!(transport.stats().connections_opened, 1);
        assert_eq!(seen.load(Ordering::Relaxed), 64);
    }

    #[tokio::test]
    async fn an_unreachable_peer_reports_rather_than_hangs() {
        let transport = TcpTransport::with_directory(
            1,
            [(2, "127.0.0.1:1".to_string())],
            TransportConfig {
                connect_timeout: Duration::from_millis(100),
                rpc_timeout: Duration::from_millis(200),
                ..TransportConfig::default()
            },
        );
        let err = transport
            .send_read_index(2, ReadIndexRequest { term: 1, from: 1 })
            .await
            .expect_err("should fail");
        assert!(err.transient(), "{err}");
    }

    #[tokio::test]
    async fn a_peer_with_no_address_is_named() {
        let transport = TcpTransport::new(1, TransportConfig::default());
        let err = transport
            .send_read_index(9, ReadIndexRequest { term: 1, from: 1 })
            .await
            .expect_err("should fail");
        assert_eq!(err, RaftRpcError::UnknownPeer { node: 9 });
    }

    #[tokio::test]
    async fn a_refusal_comes_back_as_a_refusal() {
        let (address, _stop) = start(Arc::new(Refuser)).await;
        let transport = TcpTransport::with_directory(1, [(2, address)], TransportConfig::default());
        let err = transport
            .send_request_vote(
                2,
                RequestVoteRequest {
                    term: 1,
                    candidate_id: 1,
                    last_log_index: 0,
                    last_log_term: 0,
                    pre_vote: false,
                },
            )
            .await
            .expect_err("should refuse");
        assert!(!err.transient(), "{err}");
        assert!(err.to_string().contains("no votes today"), "{err}");
    }

    #[tokio::test]
    async fn a_dead_connection_is_replaced_on_the_next_call() {
        let seen = Arc::new(AtomicU64::new(0));
        let (address, stop) = start(Arc::new(Echo {
            seen: Arc::clone(&seen),
        }))
        .await;
        let transport = TcpTransport::with_directory(
            1,
            [(2, address.clone())],
            TransportConfig {
                connect_timeout: Duration::from_millis(200),
                rpc_timeout: Duration::from_millis(300),
                ..TransportConfig::default()
            },
        );
        transport
            .send_read_index(2, ReadIndexRequest { term: 1, from: 1 })
            .await
            .expect("first");

        // The server goes away, so the call fails and the connection is dropped
        stop.notify_waiters();
        tokio::time::sleep(Duration::from_millis(50)).await;
        let _ = transport
            .send_read_index(2, ReadIndexRequest { term: 1, from: 1 })
            .await;

        // A new server on the same address is reached without any other help
        let listener = TcpListener::bind(address.parse::<std::net::SocketAddr>().expect("addr"))
            .await
            .expect("rebind");
        let shutdown = Arc::new(tokio::sync::Notify::new());
        tokio::spawn(RaftServer::serve(
            listener,
            Arc::new(Echo {
                seen: Arc::clone(&seen),
            }),
            TransportConfig::default(),
            shutdown,
        ));
        let reply = transport
            .send_read_index(2, ReadIndexRequest { term: 1, from: 1 })
            .await
            .expect("after restart");
        assert_eq!(reply.read_index, 4242);
        assert!(transport.stats().connections_opened >= 2);
    }
}

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
//! ## Calls go out in the order they are made
//!
//! A call's frame is placed on its connection's queue when the call is made,
//! before the future for its reply is returned, and the queue is the order
//! the socket writes in. The leader's pipeline rests on that: the batches it
//! builds in sequence reach the follower in sequence, and the consistency
//! check on each finds the one before it already in the log. A frame placed
//! from whichever task happened to run first would arrive ahead of its
//! predecessor often enough to have a tenth of all appends refused, and
//! everything behind each refusal sent again. A lane with no connection
//! places nothing until one is open, so calls made across a reconnect can
//! cross, and a refused batch then costs one resend.
//!
//! ## Failure is a fact about a call, not about a node
//!
//! A call that cannot be delivered returns [`RaftRpcError`] rather than
//! retrying inside the transport. Consensus already knows what to do about a
//! peer that did not answer, and it will try again on its own schedule with a
//! message built from current state. A transport that retried on its own would
//! deliver a stale AppendEntries after the leader had already moved on.
//!
//! ## Every frame names the protocol it speaks
//!
//! The members of one group run two adjacent releases for the length of a
//! rolling upgrade, so a frame carries [`CONSENSUS_PROTOCOL_VERSION`] in its
//! header and a reply carries the version of the request it answers. A
//! message changes inside a version by appending a field after the ones
//! that shipped, and its decoder reads an absent trailing field as the
//! default, which is how a build without the field reads the request it
//! knows and a build with it reads the older build's request. A change
//! that cannot be expressed that way is a new version. A frame from a
//! version newer than this build speaks is refused by name rather than
//! decoded, so the sender's log says which of the two is behind, and a
//! header that names no version at all is version one

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
/// magic 4, payload length 4, request id 8, kind 1, flags 1, protocol
/// version 2
const FRAME_HEADER_LEN: usize = 20;

/// The consensus protocol version this build speaks, stamped into every
/// frame it sends. A header carrying zero is read as version one, which is
/// what a build that stamped nothing sent
pub const CONSENSUS_PROTOCOL_VERSION: u16 = 1;

const KIND_REQUEST_VOTE: u8 = 1;
const KIND_APPEND_ENTRIES: u8 = 2;
const KIND_INSTALL_SNAPSHOT: u8 = 3;
const KIND_READ_INDEX: u8 = 4;
const KIND_TIMEOUT_NOW: u8 = 5;

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

/// What a call returns once it has crossed a node boundary.
///
/// Owned rather than borrowed from the transport, so the caller that makes
/// the call can hand the wait for its answer to another task
pub type RaftFuture<T> = Pin<Box<dyn Future<Output = Result<T, RaftRpcError>> + Send + 'static>>;

/// What one node asks another.
///
/// The consensus driver is written against this rather than against sockets,
/// so a deployment that carries these calls another way replaces one file
pub trait RaftTransport: Send + Sync {
    fn send_request_vote(
        &self,
        to: NodeId,
        req: RequestVoteRequest,
    ) -> RaftFuture<RequestVoteReply>;

    /// Sends one batch, or a heartbeat.
    ///
    /// The call's frame is on the connection's queue when this returns, and
    /// two calls made in sequence to one peer are delivered in that
    /// sequence. The leader's pipeline rests on this: a batch reaching the
    /// follower ahead of the one before it is refused
    fn send_append_entries(
        &self,
        to: NodeId,
        req: AppendEntriesRequest,
    ) -> RaftFuture<AppendEntriesReply>;

    fn send_install_snapshot(
        &self,
        to: NodeId,
        req: InstallSnapshotRequest,
    ) -> RaftFuture<InstallSnapshotReply>;

    fn send_read_index(&self, to: NodeId, req: ReadIndexRequest) -> RaftFuture<ReadIndexReply>;

    /// Tells a follower to campaign at once, which hands it the group
    fn send_timeout_now(
        &self,
        to: NodeId,
        req: crate::election::TimeoutNowRequest,
    ) -> RaftFuture<crate::election::TimeoutNowReply>;

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

    fn on_timeout_now(
        &self,
        req: crate::election::TimeoutNowRequest,
    ) -> RaftHandlerFuture<'_, crate::election::TimeoutNowReply>;
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
    /// The protocol version the sender stamped, one when it stamped nothing
    version: u16,
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
fn stamp_frame(buf: &mut [u8], request_id: u64, kind: u8, flags: u8, version: u16) {
    let body_len = (buf.len() - FRAME_HEADER_LEN) as u32;
    buf[0..4].copy_from_slice(&FRAME_MAGIC.to_le_bytes());
    buf[4..8].copy_from_slice(&body_len.to_le_bytes());
    buf[8..16].copy_from_slice(&request_id.to_le_bytes());
    buf[16] = kind;
    buf[17] = flags;
    buf[18..20].copy_from_slice(&version.to_le_bytes());
}

/// The refusal a frame from a newer protocol gets, so the sender's log names
/// which build is behind
fn version_refusal(version: u16) -> String {
    format!(
        "consensus protocol version {version} is newer than this build speaks, which is \
         {CONSENSUS_PROTOCOL_VERSION}"
    )
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
    let version = match u16::from_le_bytes([header[18], header[19]]) {
        0 => 1,
        stamped => stamped,
    };
    let mut payload = vec![0u8; len];
    reader.read_exact(&mut payload).await?;
    Ok(Some(Frame {
        request_id,
        kind: header[16],
        flags: header[17],
        version,
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

    /// Stamps one frame and puts it on the socket queue, returning where its
    /// reply arrives. The frame comes back when the connection has closed,
    /// so the caller can send it on the next one
    fn send(
        &self,
        kind: u8,
        mut framed: Vec<u8>,
    ) -> Result<(u64, oneshot::Receiver<Frame>), Vec<u8>> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        stamp_frame(&mut framed, id, kind, 0, CONSENSUS_PROTOCOL_VERSION);
        let (tx, rx) = oneshot::channel();
        self.pending.lock().insert(id, tx);
        match self.tx.send(framed) {
            Ok(()) => Ok((id, rx)),
            Err(mpsc::error::SendError(framed)) => {
                self.pending.lock().remove(&id);
                self.fail_all();
                Err(framed)
            }
        }
    }
}

/// One call after its frame has been placed
enum Placed {
    /// On the socket queue of a live connection, in the order it was made
    Sent {
        conn: Arc<Conn>,
        id: u64,
        rx: oneshot::Receiver<Frame>,
    },
    /// The lane had no connection, so the frame goes out once one is open
    Unsent { kind: u8, framed: Vec<u8> },
}

struct PeerLane {
    node: NodeId,
    address: parking_lot::RwLock<String>,
    /// The connection calls go out on, behind a lock that is never held
    /// across an await, so a call places its frame at call time
    conn: parking_lot::Mutex<Option<Arc<Conn>>>,
    /// Held by the one task opening a connection, so a burst of calls on a
    /// lane with none opens one socket rather than one each
    connecting: AsyncMutex<()>,
    config: TransportConfig,
    /// The deadline for calls on this lane, which is the bulk one on the lane
    /// snapshots use
    call_timeout: Duration,
    stats: Arc<StatsInner>,
}

impl PeerLane {
    /// The connection this lane holds, while it is still open
    fn live(&self) -> Option<Arc<Conn>> {
        let mut slot = self.conn.lock();
        match slot.as_ref() {
            Some(conn) if conn.alive.load(Ordering::Acquire) => Some(Arc::clone(conn)),
            Some(_) => {
                *slot = None;
                None
            }
            None => None,
        }
    }

    async fn connection(&self) -> Result<Arc<Conn>, RaftRpcError> {
        if let Some(conn) = self.live() {
            return Ok(conn);
        }
        let _opening = self.connecting.lock().await;
        // Another caller may have connected while this one waited
        if let Some(conn) = self.live() {
            return Ok(conn);
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
            let mut coalesced: Vec<u8> = Vec::new();
            while let Some(first) = rx.recv().await {
                // Whatever is queued behind a frame goes out in the same
                // write. A pipelined run of appends is many small frames,
                // and the syscall per frame costs more than joining them
                let bytes: &[u8] = match rx.try_recv() {
                    Ok(second) => {
                        coalesced.clear();
                        coalesced.extend_from_slice(&first);
                        coalesced.extend_from_slice(&second);
                        while coalesced.len() < WRITE_COALESCE_BYTES {
                            match rx.try_recv() {
                                Ok(more) => coalesced.extend_from_slice(&more),
                                Err(_) => break,
                            }
                        }
                        &coalesced
                    }
                    Err(_) => &first,
                };
                if write_half.write_all(bytes).await.is_err() {
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
        *self.conn.lock() = Some(Arc::clone(&conn));
        Ok(conn)
    }

    /// Places one call. `framed` is a [`frame_buffer`] whose body is already
    /// written, so the header is stamped in place rather than concatenated.
    ///
    /// On a live connection the frame is on the socket queue when this
    /// returns. The queue is the order the socket writes in and the lock is
    /// held for the placing, so two calls made in sequence on one lane reach
    /// the peer in that sequence, whatever order their replies are awaited in
    fn place(&self, kind: u8, framed: Vec<u8>) -> Placed {
        self.stats.calls_sent.fetch_add(1, Ordering::Relaxed);
        let mut slot = self.conn.lock();
        let live = slot
            .as_ref()
            .filter(|conn| conn.alive.load(Ordering::Acquire))
            .cloned();
        match live {
            Some(conn) => match conn.send(kind, framed) {
                Ok((id, rx)) => Placed::Sent { conn, id, rx },
                Err(framed) => {
                    *slot = None;
                    self.stats.connections_lost.fetch_add(1, Ordering::Relaxed);
                    Placed::Unsent { kind, framed }
                }
            },
            None => {
                *slot = None;
                Placed::Unsent { kind, framed }
            }
        }
    }

    /// Waits for the answer to a placed call, opening the connection first
    /// when the call found none
    async fn finish(&self, placed: Placed) -> Result<Vec<u8>, RaftRpcError> {
        let result = self.finish_inner(placed).await;
        if result.is_err() {
            self.stats.calls_failed.fetch_add(1, Ordering::Relaxed);
        }
        result
    }

    async fn finish_inner(&self, placed: Placed) -> Result<Vec<u8>, RaftRpcError> {
        let (conn, id, rx) = match placed {
            Placed::Sent { conn, id, rx } => (conn, id, rx),
            Placed::Unsent { kind, framed } => {
                let conn = self.connection().await?;
                match conn.send(kind, framed) {
                    Ok((id, rx)) => (conn, id, rx),
                    Err(_) => {
                        self.stats.connections_lost.fetch_add(1, Ordering::Relaxed);
                        return Err(RaftRpcError::Unreachable {
                            node: self.node,
                            reason: "connection closed before the call was written".into(),
                        });
                    }
                }
            }
        };
        match tokio::time::timeout(self.call_timeout, rx).await {
            Ok(Ok(frame)) => {
                if frame.flags & FLAG_ERROR != 0 {
                    let mut c = Cursor::new(&frame.payload);
                    let reason = c.string().unwrap_or_else(|_| "unreadable".into());
                    return Err(RaftRpcError::Refused { reason });
                }
                // A peer answers in the version of the request it was sent,
                // so a reply from a newer version is one this build never
                // asked for and cannot read
                if frame.version > CONSENSUS_PROTOCOL_VERSION {
                    return Err(RaftRpcError::Malformed {
                        reason: version_refusal(frame.version),
                    });
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

    /// The lane a call to `node` goes out on, opened on the first call.
    ///
    /// The directory is consulted only when the lane does not exist yet. A
    /// lane that exists was opened at the directory's address, and an
    /// address change drops it, so the lookup and the string copy it costs
    /// are not paid on every message
    fn lane(&self, node: NodeId, lane: u8) -> Result<Arc<PeerLane>, RaftRpcError> {
        if self.shutdown.load(Ordering::Acquire) {
            return Err(RaftRpcError::Shutdown);
        }
        if let Some(existing) = self.lanes.lock().get(&(node, lane)) {
            return Ok(Arc::clone(existing));
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
                conn: parking_lot::Mutex::new(None),
                connecting: AsyncMutex::new(()),
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

    /// Places a call on its lane and returns the wait for its answer. The
    /// placing happens here, at call time, which is the ordering
    /// [`RaftTransport::send_append_entries`] promises
    fn call<T, F>(
        lane: Result<Arc<PeerLane>, RaftRpcError>,
        kind: u8,
        framed: Vec<u8>,
        decode: F,
    ) -> RaftFuture<T>
    where
        T: Send + 'static,
        F: FnOnce(&[u8]) -> Result<T, zyron_common::error::ZyronError> + Send + 'static,
    {
        let lane = match lane {
            Ok(lane) => lane,
            Err(e) => return Box::pin(async move { Err(e) }),
        };
        let placed = lane.place(kind, framed);
        Box::pin(async move {
            let bytes = lane.finish(placed).await?;
            decode(&bytes).map_err(|e| RaftRpcError::Malformed {
                reason: e.to_string(),
            })
        })
    }
}

impl RaftTransport for TcpTransport {
    fn send_request_vote(
        &self,
        to: NodeId,
        req: RequestVoteRequest,
    ) -> RaftFuture<RequestVoteReply> {
        let lane = self.lane(to, LANE_CONTROL);
        let mut payload = frame_buffer(48);
        req.encode(&mut payload);
        Self::call(lane, KIND_REQUEST_VOTE, payload, |b| {
            RequestVoteReply::decode(&mut Cursor::new(b))
        })
    }

    fn send_timeout_now(
        &self,
        to: NodeId,
        req: crate::election::TimeoutNowRequest,
    ) -> RaftFuture<crate::election::TimeoutNowReply> {
        let lane = self.lane(to, LANE_CONTROL);
        let mut payload = frame_buffer(24);
        req.encode(&mut payload);
        Self::call(lane, KIND_TIMEOUT_NOW, payload, |b| {
            crate::election::TimeoutNowReply::decode(&mut Cursor::new(b))
        })
    }

    fn send_append_entries(
        &self,
        to: NodeId,
        req: AppendEntriesRequest,
    ) -> RaftFuture<AppendEntriesReply> {
        let lane = self.lane(to, LANE_CONTROL);
        let mut payload = frame_buffer(req.encoded_len());
        req.encode(&mut payload);
        Self::call(lane, KIND_APPEND_ENTRIES, payload, |b| {
            AppendEntriesReply::decode(&mut Cursor::new(b))
        })
    }

    fn send_install_snapshot(
        &self,
        to: NodeId,
        req: InstallSnapshotRequest,
    ) -> RaftFuture<InstallSnapshotReply> {
        let lane = self.lane(to, LANE_BULK);
        let mut payload = frame_buffer(req.data.len() + 128);
        req.encode(&mut payload);
        Self::call(lane, KIND_INSTALL_SNAPSHOT, payload, |b| {
            InstallSnapshotReply::decode(&mut Cursor::new(b))
        })
    }

    fn send_read_index(&self, to: NodeId, req: ReadIndexRequest) -> RaftFuture<ReadIndexReply> {
        let lane = self.lane(to, LANE_CONTROL);
        let mut payload = frame_buffer(16);
        req.encode(&mut payload);
        Self::call(lane, KIND_READ_INDEX, payload, |b| {
            ReadIndexReply::decode(&mut Cursor::new(b))
        })
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

    let mut batch: Vec<(u64, u8, u16, ReplyFuture<'_>)> = Vec::with_capacity(PIPELINE_DEPTH);
    let mut answers: Vec<u8> = Vec::new();
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
        // The group's answers share one write. They became ready together,
        // behind the one fsync the group waited for, so nothing is held back
        // by joining them. Each answer is stamped with the version of the
        // request it answers, which the sender is known to read
        answers.clear();
        for (request_id, kind, version, reply) in batch.drain(..) {
            let (flags, mut framed) = reply.await;
            stamp_frame(&mut framed, request_id, kind, flags, version);
            answers.extend_from_slice(&framed);
        }
        if let Err(e) = write_half.write_all(&answers).await {
            break Err(e);
        }
    };
    reader.abort();
    result
}

/// Bytes a coalesced write grows to before it goes out on its own. A
/// snapshot chunk is this size and never needs company
const WRITE_COALESCE_BYTES: usize = 1024 * 1024;

/// Runs the synchronous half of one request and returns what is left to wait
/// for.
///
/// Deliberately not an async function: the decode and the handler call have to
/// happen at call time so that a run of AppendEntries reaches the log in the
/// order it arrived, whatever order the waits are polled in afterwards.
///
/// A frame from a protocol version newer than this build speaks is refused
/// before it is decoded, and the refusal goes back at this build's own
/// version, which is the newest the sender is known to read
fn begin_reply<'a>(
    handler: &'a Arc<dyn RaftRequestHandler>,
    frame: Frame,
) -> (u64, u8, u16, ReplyFuture<'a>) {
    let Frame {
        request_id,
        kind,
        version,
        payload,
        ..
    } = frame;
    if version > CONSENSUS_PROTOCOL_VERSION {
        return (
            request_id,
            kind,
            CONSENSUS_PROTOCOL_VERSION,
            refused(version_refusal(version)),
        );
    }
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
        KIND_TIMEOUT_NOW => {
            match crate::election::TimeoutNowRequest::decode(&mut Cursor::new(&payload)) {
                Ok(req) => {
                    let fut = handler.on_timeout_now(req);
                    Box::pin(async move { finish(fut.await) })
                }
                Err(e) => refused(e.to_string()),
            }
        }
        other => refused(format!("consensus frame kind {other} is not known")),
    };
    (request_id, kind, version, reply)
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

impl Encodable for crate::election::TimeoutNowReply {
    fn encode(&self, buf: &mut Vec<u8>) {
        crate::election::TimeoutNowReply::encode(self, buf)
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

        fn on_timeout_now(
            &self,
            req: crate::election::TimeoutNowRequest,
        ) -> RaftHandlerFuture<'_, crate::election::TimeoutNowReply> {
            Box::pin(async move {
                Ok(crate::election::TimeoutNowReply {
                    term: req.term + 1,
                    started: true,
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

        fn on_timeout_now(
            &self,
            _req: crate::election::TimeoutNowRequest,
        ) -> RaftHandlerFuture<'_, crate::election::TimeoutNowReply> {
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
                    transfer: false,
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
                    transfer: false,
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

    /// A frame from a newer protocol is refused with both numbers in the
    /// reason, at this build's own version, and the connection stays open
    /// for the frames this build reads. A header that names no version is
    /// version one, which is what a build that stamped nothing sent
    #[tokio::test]
    async fn a_newer_protocol_frame_is_refused_by_name_and_a_zero_version_reads_as_one() {
        let (address, _stop) = start(Arc::new(Echo {
            seen: Arc::new(AtomicU64::new(0)),
        }))
        .await;
        let mut stream = TcpStream::connect(&address).await.expect("connect");

        let mut newer = frame_buffer(16);
        ReadIndexRequest { term: 1, from: 1 }.encode(&mut newer);
        stamp_frame(
            &mut newer,
            7,
            KIND_READ_INDEX,
            0,
            CONSENSUS_PROTOCOL_VERSION + 1,
        );
        stream.write_all(&newer).await.expect("write");
        let reply = read_frame(&mut stream, 1024)
            .await
            .expect("read")
            .expect("a frame");
        assert_eq!(reply.request_id, 7);
        assert_ne!(
            reply.flags & FLAG_ERROR,
            0,
            "the frame was decoded rather than refused"
        );
        assert_eq!(reply.version, CONSENSUS_PROTOCOL_VERSION);
        let mut cursor = Cursor::new(&reply.payload);
        let reason = cursor.string().expect("reason");
        assert!(reason.contains("version 2"), "{reason}");
        assert!(reason.contains("which is 1"), "{reason}");

        let mut unstamped = frame_buffer(16);
        ReadIndexRequest { term: 1, from: 1 }.encode(&mut unstamped);
        stamp_frame(&mut unstamped, 8, KIND_READ_INDEX, 0, 0);
        stream.write_all(&unstamped).await.expect("write");
        let reply = read_frame(&mut stream, 1024)
            .await
            .expect("read")
            .expect("a frame");
        assert_eq!(reply.request_id, 8);
        assert_eq!(reply.flags & FLAG_ERROR, 0, "{:?}", reply.payload);
        assert_eq!(reply.version, 1);
        let decoded = ReadIndexReply::decode(&mut Cursor::new(&reply.payload)).expect("decodes");
        assert_eq!(decoded.read_index, 4242);
    }
}

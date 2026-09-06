//! The calls one node makes to another, what they carry, and where they land.
//!
//! ## Versioned paths, one transport behind a trait
//!
//! Every call has a path under `/internal/mesh/v1/`. The version is in the
//! path rather than a header because a node talking to a peer running a
//! different build has to fail on the route rather than on the payload: a 404
//! naming a version is a diagnosable answer, and a field quietly missing from
//! a body is not.
//!
//! [`MeshRpc`] is the trait the scheduler is written against.
//! [`crate::transport`] implements it over the HTTP the node already serves.
//! When the binary protocol lands, the implementation is swapped and nothing
//! in the scheduler changes, which is the reason the trait exists at all.
//!
//! ## How a body changes inside one path version
//!
//! The members of one group run two adjacent releases for the length of a
//! rolling upgrade, and either side of a call may be the older one. So every
//! body decodes with any field absent, each absent field reading as its
//! default, and every body ignores a field its reader does not know. That is
//! what lets a release add a field: the older side leaves it out or skips
//! it, and the newer side reads the default. A field added after a type
//! ships carries a default that means "the sender did not say", and the
//! caller treats it that way. A call added after the version ships is a new
//! path, which an older node answers with a 404 the caller reads as
//! [`MeshRpcError::Unknown`]. A change that cannot be expressed as an added
//! field or an added path, a removed field, a renamed one, a new variant of
//! an enum a body carries, is a new path version, registered beside this one
//! and spoken only once every node in the group reads it.
//!
//! ## Sized for framing
//!
//! These payloads will eventually cross a length-prefixed binary protocol.
//! Every field is bounded at the type level or by a documented constant, and
//! nothing is a map with arbitrary keys at the top level:
//!
//! - Names are owned strings with a stated maximum, checked by the
//!   validators, so a frame's size can be computed from its field count
//!   before it is built.
//! - Page id lists arrive in [`HotSetChunk`]s of at most [`MAX_CHUNK_PAGES`],
//!   so a manifest of any size is a stream of same-shaped frames rather than
//!   one frame whose size is the sender's cache.
//! - Every response carries the sequence of the request it answers, so a
//!   reply arriving after a retry is discarded rather than applied twice.
//!
//! Doing this now costs nothing. Doing it later means discovering, in the wire
//! code, that a payload cannot be framed, and picking a bound in the place
//! least able to reason about it.

use std::future::Future;
use std::pin::Pin;

use serde::{Deserialize, Serialize};
use zyron_common::format::wire_version::{WireProtocol, WireProtocolVersion, WireVersionStatus};

/// The mesh protocol version this binary speaks, which is the number in the
/// path prefix
pub const MESH_PROTOCOL_VERSION: u32 = 1;

inventory::submit! {
    WireProtocolVersion {
        protocol: WireProtocol::Mesh,
        version: MESH_PROTOCOL_VERSION,
        status: WireVersionStatus::Current,
        introduced_in_binary_version: "0.6.0",
        retired_in_binary_version: None,
        notes: "JSON bodies over HTTP under /internal/mesh/v1/, every body decodes with any \
                field absent",
    }
}

/// Longest a node name, tenant name, session id, or ticket may be.
///
/// Two hundred and fifty five so a length fits in one byte on the wire.
pub const MAX_NAME_BYTES: usize = 255;

/// Most page ids one hot-set chunk carries.
///
/// A thousand ids is eight kilobytes of payload, which keeps a chunk inside a
/// single page-sized frame with room for its header.
pub const MAX_CHUNK_PAGES: usize = 1000;

/// The prefix every mesh call lives under.
pub const MESH_PATH_PREFIX: &str = "/internal/mesh/v1/";

/// One call's path.
pub const PATH_BEGIN_DRAIN: &str = "/internal/mesh/v1/begin_drain";
pub const PATH_DRAIN_STATUS: &str = "/internal/mesh/v1/drain_status";
pub const PATH_HOT_SET_MANIFEST: &str = "/internal/mesh/v1/hot_set_manifest";
pub const PATH_PREFETCH: &str = "/internal/mesh/v1/prefetch";
pub const PATH_RELOCATE_SESSION: &str = "/internal/mesh/v1/relocate_session";
pub const PATH_CANCEL_PROVISIONING: &str = "/internal/mesh/v1/cancel_provisioning";
pub const PATH_NODE_STATUS: &str = "/internal/mesh/v1/node_status";
pub const PATH_STAGE_RELEASE: &str = "/internal/mesh/v1/stage_release";
pub const PATH_RESTART_INTO_STAGED: &str = "/internal/mesh/v1/restart_into_staged";
pub const PATH_ROLLBACK_TO_PREVIOUS: &str = "/internal/mesh/v1/rollback_to_previous";
pub const PATH_SET_CLUSTER_SETTING: &str = "/internal/mesh/v1/set_cluster_setting";

/// Longest a detail line in an acknowledgement may be
pub const MAX_DETAIL_BYTES: usize = 1024;

/// Whether a path is one of ours, so a server can route the whole family
/// without listing them.
pub fn is_mesh_path(path: &str) -> bool {
    path.starts_with(MESH_PATH_PREFIX)
}

/// What a call returns, once it has crossed a node boundary.
pub type MeshFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T, MeshRpcError>> + Send + 'a>>;

/// Why a call did not produce an answer.
///
/// Concrete rather than a boxed error, because these cross a process boundary
/// and the caller has to decide what to do from the variant alone: retry a
/// peer that was unreachable, give up on one that refused, and treat a
/// malformed payload as a version mismatch rather than a transient fault.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeshRpcError {
    /// The peer could not be reached, so a retry may work
    Unreachable { node: String, reason: String },
    /// The peer answered that it does not know this node, tenant, or session
    Unknown { what: String },
    /// The peer refused, and this is why
    Refused { reason: String },
    /// A payload broke one of the framing bounds above
    Malformed { field: String },
    /// The peer did not answer inside the caller's deadline
    TimedOut { after_ms: u32 },
}

impl std::fmt::Display for MeshRpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MeshRpcError::Unreachable { node, reason } => {
                write!(f, "node {node} could not be reached: {reason}")
            }
            MeshRpcError::Unknown { what } => write!(f, "the peer does not know {what}"),
            MeshRpcError::Refused { reason } => write!(f, "the peer refused: {reason}"),
            MeshRpcError::Malformed { field } => write!(f, "malformed payload field {field}"),
            MeshRpcError::TimedOut { after_ms } => write!(f, "no answer after {after_ms}ms"),
        }
    }
}

impl std::error::Error for MeshRpcError {}

impl MeshRpcError {
    /// Whether trying the same call again could succeed.
    ///
    /// Read by the scheduler, which retries a drain poll against a peer that
    /// blinked and does not retry one that refused.
    pub fn transient(&self) -> bool {
        matches!(
            self,
            MeshRpcError::Unreachable { .. } | MeshRpcError::TimedOut { .. }
        )
    }
}

/// One node, by the identity it keeps across restarts.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(default)]
pub struct NodeRef {
    /// The persisted node id, which survives a restart and a move of the data
    /// directory. Zero when the peer has not been contacted yet and its id is
    /// still unknown
    pub node_id: u64,
    /// The operator-facing name, bounded by `MAX_NAME_BYTES`
    pub name: String,
}

impl NodeRef {
    pub fn new(node_id: u64, name: impl Into<String>) -> Self {
        Self {
            node_id,
            name: name.into(),
        }
    }

    /// Whether this fits the framing bounds.
    pub fn valid(&self) -> bool {
        !self.name.is_empty() && self.name.len() <= MAX_NAME_BYTES
    }
}

/// Asks a node to stop taking new work and finish what it has.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct BeginDrainRequest {
    pub target: NodeRef,
    /// Sequence, so a retried request is recognized as the same one
    pub sequence: u64,
    /// How long the node may take before the caller stops waiting. A u32 of
    /// milliseconds, which is forty nine days and is a bound rather than an
    /// aspiration
    pub deadline_ms: u32,
    /// Whether sessions may be moved rather than ended
    pub relocate_sessions: bool,
}

impl BeginDrainRequest {
    pub fn valid(&self) -> bool {
        self.target.valid()
    }
}

/// Asks how far a drain has got.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct DrainStatusRequest {
    pub target: NodeRef,
    pub sequence: u64,
}

impl DrainStatusRequest {
    pub fn valid(&self) -> bool {
        self.target.valid()
    }
}

/// Where a drain has got to.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct DrainStatus {
    pub target: NodeRef,
    /// The sequence of the request this answers
    pub sequence: u64,
    /// Queries still running on the node
    pub queries_in_flight: u32,
    /// Sessions still attached
    pub sessions_attached: u32,
    /// Transactions the node is holding open
    pub transactions_open: u32,
    /// True once the node has nothing left to finish
    pub drained: bool,
}

impl DrainStatus {
    /// What a node reports when it has nothing running.
    pub fn idle(target: NodeRef, sequence: u64) -> Self {
        Self {
            target,
            sequence,
            queries_in_flight: 0,
            sessions_attached: 0,
            transactions_open: 0,
            drained: true,
        }
    }
}

/// Asks a node what it had in memory, one chunk at a time.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct HotSetManifestRequest {
    pub target: NodeRef,
    pub sequence: u64,
    /// Which chunk, from zero. The peer reports the total so the caller knows
    /// when it has them all
    pub chunk: u32,
}

impl HotSetManifestRequest {
    pub fn valid(&self) -> bool {
        self.target.valid()
    }
}

/// One chunk of a hot set.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct HotSetChunk {
    pub source: NodeRef,
    pub sequence: u64,
    pub chunk: u32,
    /// Chunks in the whole manifest, so a caller can tell a short read from
    /// the end
    pub chunks_total: u32,
    /// At most `MAX_CHUNK_PAGES` ids
    pub page_ids: Vec<u64>,
}

impl HotSetChunk {
    pub fn valid(&self) -> bool {
        self.page_ids.len() <= MAX_CHUNK_PAGES && self.source.valid()
    }
}

/// Asks a node to read pages before it is given the work that needs them.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct PrefetchRequest {
    pub target: NodeRef,
    pub sequence: u64,
    /// At most `MAX_CHUNK_PAGES` ids, for the same reason a manifest chunk is
    pub page_ids: Vec<u64>,
    /// Bytes the peer may spend on this. A prefetch that evicts the pages the
    /// peer is already serving from is worse than no prefetch
    pub byte_budget: u64,
}

impl PrefetchRequest {
    pub fn valid(&self) -> bool {
        self.page_ids.len() <= MAX_CHUNK_PAGES && self.target.valid()
    }
}

/// How much of a prefetch landed.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct PrefetchStatus {
    pub target: NodeRef,
    pub sequence: u64,
    pub pages_read: u32,
    pub pages_skipped: u32,
    /// True when the byte budget stopped it before the list ran out
    pub budget_exhausted: bool,
}

/// Asks that a session be moved to another node.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct RelocateSessionRequest {
    pub from: NodeRef,
    pub to: NodeRef,
    pub sequence: u64,
    /// Bounded by `MAX_NAME_BYTES`
    pub session_id: String,
}

impl RelocateSessionRequest {
    pub fn valid(&self) -> bool {
        self.from.valid()
            && self.to.valid()
            && !self.session_id.is_empty()
            && self.session_id.len() <= MAX_NAME_BYTES
    }
}

/// What became of a session that was asked to move.
///
/// An enum crosses the wire as its variant name, so a variant added later is
/// one an older reader cannot decode. A new outcome is a new path version
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelocationOutcome {
    /// The session is on the target node
    Moved { to: NodeRef },
    /// The session ended before it could be moved
    Ended,
    /// The session is holding something that cannot move, and this is what
    Pinned { reason: String },
}

/// Asks that a provisioning request in flight be abandoned.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct CancelProvisioningRequest {
    pub sequence: u64,
    /// The ticket the provisioner returned, bounded by `MAX_NAME_BYTES`
    pub ticket_id: String,
}

impl CancelProvisioningRequest {
    pub fn valid(&self) -> bool {
        !self.ticket_id.is_empty() && self.ticket_id.len() <= MAX_NAME_BYTES
    }
}

/// Asks a node what it runs and how it is doing.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct NodeStatusRequest {
    pub target: NodeRef,
    pub sequence: u64,
}

impl NodeStatusRequest {
    pub fn valid(&self) -> bool {
        self.target.valid()
    }
}

/// What a node runs and how it is doing, which the upgrade driver reads
/// before and after it restarts a node.
///
/// Rates are integers so the frame carries no floating point: throughput in
/// thousandths of a query per second, the failed fraction in parts per
/// million
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct NodeStatus {
    pub target: NodeRef,
    pub sequence: u64,
    /// The binary version this node runs, bounded by `MAX_NAME_BYTES`
    pub version: String,
    /// The version staged beside the running binary, empty when none is
    pub staged_version: String,
    pub draining: bool,
    pub accepting: bool,
    pub queries_in_flight: u32,
    pub sessions_attached: u32,
    pub transactions_open: u32,
    pub p50_latency_us: u64,
    pub p99_latency_us: u64,
    pub throughput_milli_per_sec: u64,
    pub error_rate_ppm: u64,
    /// Queries the rate window held, which says whether the rates above
    /// mean anything
    pub queries_in_window: u64,
    pub uptime_secs: u64,
}

impl NodeStatus {
    pub fn valid(&self) -> bool {
        self.target.valid()
            && self.version.len() <= MAX_NAME_BYTES
            && self.staged_version.len() <= MAX_NAME_BYTES
    }
}

/// Asks a node to fetch, verify, and stage one release.
///
/// Staging runs after the answer, so the caller polls `node_status` until
/// the staged version is the one it asked for
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct StageReleaseRequest {
    pub target: NodeRef,
    pub sequence: u64,
    /// Bounded by `MAX_NAME_BYTES`
    pub version: String,
}

impl StageReleaseRequest {
    pub fn valid(&self) -> bool {
        self.target.valid() && !self.version.is_empty() && self.version.len() <= MAX_NAME_BYTES
    }
}

/// Asks a node to restart on the binary it staged for one version.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct RestartRequest {
    pub target: NodeRef,
    pub sequence: u64,
    /// Bounded by `MAX_NAME_BYTES`
    pub version: String,
}

impl RestartRequest {
    pub fn valid(&self) -> bool {
        self.target.valid() && !self.version.is_empty() && self.version.len() <= MAX_NAME_BYTES
    }
}

/// Asks a node to restart on the binary it ran before its last activation.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct RollbackRequest {
    pub target: NodeRef,
    pub sequence: u64,
}

impl RollbackRequest {
    pub fn valid(&self) -> bool {
        self.target.valid()
    }
}

/// Hands the leader one cluster setting to write to the replicated log.
///
/// Only the leader appends, so a node that took `ALTER SYSTEM SET` of an
/// upgrade setting, or paused itself after a rollback, sends the setting
/// here. The leader appends it on its next pass, after answering
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct SetClusterSettingRequest {
    pub target: NodeRef,
    pub sequence: u64,
    /// The config key, bounded by `MAX_NAME_BYTES`
    pub key: String,
    /// Bounded by `MAX_DETAIL_BYTES`
    pub value: String,
}

impl SetClusterSettingRequest {
    pub fn valid(&self) -> bool {
        self.target.valid()
            && !self.key.is_empty()
            && self.key.len() <= MAX_NAME_BYTES
            && self.value.len() <= MAX_DETAIL_BYTES
    }
}

/// Whether a node took a request it carries out after answering, and why
/// not when it did not.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct NodeAck {
    pub target: NodeRef,
    pub sequence: u64,
    pub accepted: bool,
    /// Bounded by `MAX_DETAIL_BYTES`
    pub detail: String,
}

impl NodeAck {
    pub fn valid(&self) -> bool {
        self.target.valid() && self.detail.len() <= MAX_DETAIL_BYTES
    }
}

/// What one node asks of another.
///
/// Async, because every one of these crosses a network and the caller is a
/// background task that has other nodes to talk to while it waits. Boxed
/// futures rather than `async fn`, because the scheduler holds this behind a
/// trait object so the transport can be replaced without touching it.
pub trait MeshRpc: Send + Sync {
    /// Asks a node to stop taking new work.
    fn begin_drain(&self, request: BeginDrainRequest) -> MeshFuture<'_, DrainStatus>;

    /// Asks how far a drain has got.
    fn drain_status(&self, request: DrainStatusRequest) -> MeshFuture<'_, DrainStatus>;

    /// Reads one chunk of a node's hot set.
    fn hot_set_manifest(&self, request: HotSetManifestRequest) -> MeshFuture<'_, HotSetChunk>;

    /// Asks a node to read pages before the work that needs them arrives.
    fn prefetch(&self, request: PrefetchRequest) -> MeshFuture<'_, PrefetchStatus>;

    /// Asks that a session move to another node.
    fn relocate_session(
        &self,
        request: RelocateSessionRequest,
    ) -> MeshFuture<'_, RelocationOutcome>;

    /// Abandons a provisioning request in flight.
    fn cancel_provisioning(&self, request: CancelProvisioningRequest) -> MeshFuture<'_, ()>;

    /// Asks a node what it runs and how it is doing.
    fn node_status(&self, request: NodeStatusRequest) -> MeshFuture<'_, NodeStatus>;

    /// Asks a node to fetch, verify, and stage one release.
    fn stage_release(&self, request: StageReleaseRequest) -> MeshFuture<'_, NodeAck>;

    /// Asks a node to restart on the binary it staged.
    fn restart_into_staged(&self, request: RestartRequest) -> MeshFuture<'_, NodeAck>;

    /// Asks a node to restart on the binary it ran before.
    fn rollback_to_previous(&self, request: RollbackRequest) -> MeshFuture<'_, NodeAck>;

    /// Hands the leader a cluster setting for the replicated log.
    fn set_cluster_setting(&self, request: SetClusterSettingRequest) -> MeshFuture<'_, NodeAck>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node() -> NodeRef {
        NodeRef::new(7, "node-7")
    }

    /// Every path is under the one prefix a server routes on, and each is
    /// distinct, so a typo cannot silently land two calls on one handler.
    #[test]
    fn the_paths_are_distinct_and_all_under_the_prefix() {
        let paths = [
            PATH_BEGIN_DRAIN,
            PATH_DRAIN_STATUS,
            PATH_HOT_SET_MANIFEST,
            PATH_PREFETCH,
            PATH_RELOCATE_SESSION,
            PATH_CANCEL_PROVISIONING,
            PATH_NODE_STATUS,
            PATH_STAGE_RELEASE,
            PATH_RESTART_INTO_STAGED,
            PATH_ROLLBACK_TO_PREVIOUS,
            PATH_SET_CLUSTER_SETTING,
        ];
        for path in paths {
            assert!(is_mesh_path(path), "{path} is not under the mesh prefix");
        }
        let mut sorted = paths.to_vec();
        sorted.sort_unstable();
        let before = sorted.len();
        sorted.dedup();
        assert_eq!(before, sorted.len(), "two calls share a path");
        assert!(
            !is_mesh_path("/pressure"),
            "the prefix matched a foreign path"
        );
    }

    /// The number in the path is the registered protocol version, so the
    /// registry and the route cannot say different things
    #[test]
    fn the_path_prefix_names_the_registered_version() {
        assert_eq!(
            MESH_PATH_PREFIX,
            format!("/internal/mesh/v{MESH_PROTOCOL_VERSION}/")
        );
        let registered = inventory::iter::<WireProtocolVersion>
            .into_iter()
            .filter(|v| v.protocol == WireProtocol::Mesh)
            .collect::<Vec<_>>();
        assert_eq!(registered.len(), 1, "the mesh protocol registers once");
        assert_eq!(registered[0].version, MESH_PROTOCOL_VERSION);
        assert_eq!(registered[0].status, WireVersionStatus::Current);
    }

    /// Every body decodes from an object with no fields at all, which is
    /// what lets a later release add a field the older side leaves out.
    /// The validators still refuse a body whose required fields are absent,
    /// so the default is never mistaken for a real request
    #[test]
    fn every_payload_decodes_with_every_field_absent() {
        fn absent<T: serde::de::DeserializeOwned + Default + PartialEq + std::fmt::Debug>() {
            let decoded: T = serde_json::from_str("{}").expect(std::any::type_name::<T>());
            assert_eq!(decoded, T::default(), "{}", std::any::type_name::<T>());
        }
        absent::<NodeRef>();
        absent::<BeginDrainRequest>();
        absent::<DrainStatusRequest>();
        absent::<DrainStatus>();
        absent::<HotSetManifestRequest>();
        absent::<HotSetChunk>();
        absent::<PrefetchRequest>();
        absent::<PrefetchStatus>();
        absent::<RelocateSessionRequest>();
        absent::<CancelProvisioningRequest>();
        absent::<NodeStatusRequest>();
        absent::<NodeStatus>();
        absent::<StageReleaseRequest>();
        absent::<RestartRequest>();
        absent::<RollbackRequest>();
        absent::<SetClusterSettingRequest>();
        absent::<NodeAck>();

        assert!(!NodeStatusRequest::default().valid());
        assert!(!StageReleaseRequest::default().valid());
        assert!(!SetClusterSettingRequest::default().valid());

        // A field the reader does not know is skipped, and one it knows is
        // read beside it
        let status: NodeStatus = serde_json::from_str(
            r#"{"target":{"node_id":7,"name":"node-7"},"version":"0.13.0","added_later":true}"#,
        )
        .expect("decodes with an unknown field");
        assert_eq!(status.version, "0.13.0");
        assert_eq!(status.target, node());
        assert_eq!(status.sequence, 0);
    }

    /// The bounds are checkable before a payload is framed, which is the
    /// property that lets a frame size be computed rather than discovered.
    #[test]
    fn a_payload_past_its_bound_is_rejected_by_its_own_validator() {
        let long = "x".repeat(MAX_NAME_BYTES + 1);
        assert!(!NodeRef::new(1, long.clone()).valid());
        assert!(
            !HotSetChunk {
                source: node(),
                sequence: 1,
                chunk: 0,
                chunks_total: 1,
                page_ids: vec![0; MAX_CHUNK_PAGES + 1],
            }
            .valid()
        );
        assert!(
            !PrefetchRequest {
                target: node(),
                sequence: 1,
                page_ids: vec![0; MAX_CHUNK_PAGES + 1],
                byte_budget: 1,
            }
            .valid()
        );
        assert!(
            !RelocateSessionRequest {
                from: node(),
                to: node(),
                sequence: 1,
                session_id: long,
            }
            .valid()
        );
        assert!(
            !CancelProvisioningRequest {
                sequence: 1,
                ticket_id: String::new(),
            }
            .valid()
        );
    }

    /// A payload inside its bounds passes, so the validators are not simply
    /// refusing everything.
    #[test]
    fn a_payload_inside_its_bounds_is_accepted() {
        assert!(node().valid());
        assert!(
            HotSetChunk {
                source: node(),
                sequence: 1,
                chunk: 0,
                chunks_total: 2,
                page_ids: vec![0; MAX_CHUNK_PAGES],
            }
            .valid()
        );
        assert!(
            RelocateSessionRequest {
                from: node(),
                to: node(),
                sequence: 1,
                session_id: "s-1".into(),
            }
            .valid()
        );
    }

    /// Every payload survives a round trip.
    #[test]
    fn every_payload_round_trips() {
        let status = DrainStatus {
            target: node(),
            sequence: 3,
            queries_in_flight: 2,
            sessions_attached: 1,
            transactions_open: 0,
            drained: false,
        };
        let text = serde_json::to_string(&status).expect("encode");
        assert_eq!(
            serde_json::from_str::<DrainStatus>(&text).expect("decode"),
            status
        );

        let outcome = RelocationOutcome::Pinned {
            reason: "holds an open cursor".into(),
        };
        let text = serde_json::to_string(&outcome).expect("encode");
        assert_eq!(
            serde_json::from_str::<RelocationOutcome>(&text).expect("decode"),
            outcome
        );

        let status = NodeStatus {
            target: node(),
            sequence: 9,
            version: "0.12.0".into(),
            staged_version: "0.13.0".into(),
            draining: false,
            accepting: true,
            queries_in_flight: 4,
            sessions_attached: 2,
            transactions_open: 1,
            p50_latency_us: 700,
            p99_latency_us: 4_000,
            throughput_milli_per_sec: 12_500,
            error_rate_ppm: 250,
            queries_in_window: 750,
            uptime_secs: 86_400,
        };
        assert!(status.valid());
        let text = serde_json::to_string(&status).expect("encode");
        assert_eq!(
            serde_json::from_str::<NodeStatus>(&text).expect("decode"),
            status
        );
        let ack = NodeAck {
            target: node(),
            sequence: 9,
            accepted: false,
            detail: "no release 0.13.0 is staged here".into(),
        };
        assert!(ack.valid());
        let text = serde_json::to_string(&ack).expect("encode");
        assert_eq!(serde_json::from_str::<NodeAck>(&text).expect("decode"), ack);
        assert!(
            !StageReleaseRequest {
                target: node(),
                sequence: 1,
                version: String::new(),
            }
            .valid(),
            "an empty version names no release"
        );

        let error = MeshRpcError::TimedOut { after_ms: 250 };
        let text = serde_json::to_string(&error).expect("encode");
        assert_eq!(
            serde_json::from_str::<MeshRpcError>(&text).expect("decode"),
            error
        );
    }

    /// Only the faults worth retrying report themselves as retryable.
    #[test]
    fn a_refusal_is_not_retried_and_a_blink_is() {
        assert!(
            MeshRpcError::Unreachable {
                node: "n".into(),
                reason: "connection refused".into()
            }
            .transient()
        );
        assert!(MeshRpcError::TimedOut { after_ms: 10 }.transient());
        assert!(
            !MeshRpcError::Refused {
                reason: "not draining".into()
            }
            .transient()
        );
        assert!(
            !MeshRpcError::Malformed {
                field: "page_ids".into()
            }
            .transient()
        );
    }
}

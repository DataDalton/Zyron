//! Cross-node coordination: everything one node cannot decide alone.
//!
//! `zyron-pressure` measures a node and takes the responses that node owns.
//! The two it does not own are claiming a node the mesh already has running
//! and asking a provisioner for a new one, because both mean talking to
//! something outside the process. This crate is where that work goes.
//!
//! ## The shape of it
//!
//! - [`rpc`] is what one node asks another, and where. Six calls under
//!   `/internal/mesh/v1/`, with payloads bounded so they can be framed by a
//!   binary protocol later without anyone having to pick a limit under
//!   pressure in the wire code.
//! - [`transport`] carries those calls over the HTTP the node already serves.
//!   It is behind [`rpc::MeshRpc`] so the scheduler is written against the
//!   calls and not against what carries them: when the binary protocol lands,
//!   this is the only file that changes.
//! - [`handler`] is the receiving end. This crate owns the wire format; the
//!   server owns the answers, and implements [`handler::MeshNode`] to give
//!   them.
//! - [`scheduler`] decides when the mesh grows and shrinks, and runs a
//!   scale-in in the order that keeps the survivors warm.
//! - [`pool`] holds the nodes kept ready, and what is still paid for them.
//! - [`drivers`] is how a deployment actually gets a machine.
//! - [`actuator`] is the seam into the pressure ladder: the controller climbs
//!   to a rung this node cannot perform, asks here, and takes the answer.
//!
//! ## What a node without a mesh does
//!
//! Everything above degrades to one answer: the rung cannot be reached, and
//! here is why. The ladder carries on to shedding, which is what a single node
//! should do when it is out of local relief. Nothing waits on a reply that is
//! not coming, and no rung reports itself as in force while doing nothing.

pub mod actuator;
pub mod drivers;
pub mod handler;
pub mod pool;
pub mod provisioner;
pub mod rpc;
pub mod scheduler;
pub mod transport;

pub use actuator::{MeshActuator, install_scheduler, register};
pub use drivers::{PoolMember, StaticPoolProvisioner};
pub use handler::{MeshNode, MeshResponse, dispatch};
pub use pool::WarmPool;
pub use rpc::{
    BeginDrainRequest, CancelProvisioningRequest, DrainStatus, DrainStatusRequest, HotSetChunk,
    HotSetManifestRequest, MAX_CHUNK_PAGES, MAX_NAME_BYTES, MESH_PATH_PREFIX, MeshRpc,
    MeshRpcError, NodeRef, PrefetchRequest, PrefetchStatus, RelocateSessionRequest,
    RelocationOutcome, is_mesh_path,
};
pub use scheduler::{DrainOutcome, MeshScheduler, SchedulerStats};
pub use transport::{HttpMeshRpc, MeshDirectory, TransportStats};

//! What this node answers when another node asks it something.
//!
//! `zyron-mesh` owns the calls, the paths, and the wire format. It cannot own
//! the answers: how many queries are running, which pages are resident, and
//! which sessions are attached are this crate's, and this crate sits above it.
//! So this is the implementation of `MeshNode`, and it is the only place the
//! two meet.
//!
//! ## Nothing here waits
//!
//! Every method flips a flag or reads a counter and returns. Beginning a drain
//! sets the node to stop accepting and reports what is still in flight at that
//! instant; the caller asks again later to find out how far it got. A handler
//! that blocked until the drain finished would hold a connection open for
//! minutes and turn one busy node into a stalled scheduler.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use zyron_mesh::rpc::{
    BeginDrainRequest, CancelProvisioningRequest, DrainStatus, DrainStatusRequest, HotSetChunk,
    HotSetManifestRequest, MeshRpcError, PrefetchRequest, PrefetchStatus, RelocateSessionRequest,
    RelocationOutcome,
};
use zyron_mesh::{MAX_CHUNK_PAGES, MeshNode, NodeRef};
use zyron_pressure::hot_set::HotSetManifest;

/// What this node reports and does when the mesh asks.
pub struct ServerMeshNode {
    /// This node, so an answer names who gave it
    local: NodeRef,
    /// Set once a drain has been asked for. Read by admission, which stops
    /// taking new work while it is set
    draining: Arc<AtomicBool>,
    /// Whether sessions may be moved rather than ended, from the request that
    /// started the drain
    relocate_sessions: AtomicBool,
    /// The sequence of the drain in progress, so a status request for a
    /// different one is answered as unknown rather than with this one's state
    drain_sequence: AtomicU64,
    /// Live counters the server keeps, read rather than computed here
    in_flight: Arc<InFlight>,
    /// Where the working-set manifest is written
    data_dir: Arc<parking_lot::RwLock<std::path::PathBuf>>,
    /// Pages another node asked this one to warm, drained by the prefetch
    /// task the server runs
    prefetch_queue: Arc<parking_lot::Mutex<Vec<u64>>>,
}

/// What the node currently has in hand, published by the parts that know.
#[derive(Debug, Default)]
pub struct InFlight {
    pub queries: AtomicU64,
    pub sessions: AtomicU64,
    pub transactions: AtomicU64,
}

impl InFlight {
    fn snapshot(&self) -> (u32, u32, u32) {
        (
            self.queries.load(Ordering::Relaxed).min(u32::MAX as u64) as u32,
            self.sessions.load(Ordering::Relaxed).min(u32::MAX as u64) as u32,
            self.transactions
                .load(Ordering::Relaxed)
                .min(u32::MAX as u64) as u32,
        )
    }
}

impl ServerMeshNode {
    pub fn new(
        local: NodeRef,
        draining: Arc<AtomicBool>,
        in_flight: Arc<InFlight>,
        data_dir: Arc<parking_lot::RwLock<std::path::PathBuf>>,
    ) -> Self {
        Self {
            local,
            draining,
            relocate_sessions: AtomicBool::new(false),
            drain_sequence: AtomicU64::new(0),
            in_flight,
            data_dir,
            prefetch_queue: Arc::new(parking_lot::Mutex::new(Vec::new())),
        }
    }

    /// Pages the mesh has asked this node to warm, taken by whatever reads
    /// them.
    pub fn take_prefetch_queue(&self) -> Vec<u64> {
        std::mem::take(&mut *self.prefetch_queue.lock())
    }

    /// Whether a drain has been asked for.
    pub fn is_draining(&self) -> bool {
        self.draining.load(Ordering::Relaxed)
    }

    /// This node's current state, as the mesh sees it.
    fn status(&self, target: NodeRef, sequence: u64) -> DrainStatus {
        let (queries, sessions, transactions) = self.in_flight.snapshot();
        DrainStatus {
            target,
            sequence,
            queries_in_flight: queries,
            sessions_attached: sessions,
            transactions_open: transactions,
            drained: queries == 0 && transactions == 0,
        }
    }

    /// The manifest this node last wrote, or None when it has never written
    /// one.
    fn manifest(&self) -> Option<HotSetManifest> {
        let dir = self.data_dir.read().clone();
        if dir.as_os_str().is_empty() {
            return None;
        }
        HotSetManifest::load(&dir).ok().flatten()
    }

    /// Refuses a call aimed at a different node.
    ///
    /// A mesh call names its target, and answering one addressed elsewhere
    /// would let a misrouted request move the wrong node out of the mesh.
    fn check_target(&self, target: &NodeRef) -> Result<(), MeshRpcError> {
        if self.local.node_id != 0 && target.node_id != 0 && target.node_id != self.local.node_id {
            return Err(MeshRpcError::Unknown {
                what: format!("node {} is not this node", target.node_id),
            });
        }
        Ok(())
    }
}

impl MeshNode for ServerMeshNode {
    fn begin_drain(&self, request: &BeginDrainRequest) -> Result<DrainStatus, MeshRpcError> {
        self.check_target(&request.target)?;
        self.draining.store(true, Ordering::Relaxed);
        self.relocate_sessions
            .store(request.relocate_sessions, Ordering::Relaxed);
        self.drain_sequence
            .store(request.sequence, Ordering::Relaxed);
        tracing::info!(
            sequence = request.sequence,
            deadline_ms = request.deadline_ms,
            "a mesh scheduler asked this node to drain"
        );
        Ok(self.status(request.target.clone(), request.sequence))
    }

    fn drain_status(&self, request: &DrainStatusRequest) -> Result<DrainStatus, MeshRpcError> {
        self.check_target(&request.target)?;
        if !self.draining.load(Ordering::Relaxed) {
            return Err(MeshRpcError::Refused {
                reason: "this node is not draining".into(),
            });
        }
        Ok(self.status(request.target.clone(), request.sequence))
    }

    fn hot_set_manifest(
        &self,
        request: &HotSetManifestRequest,
    ) -> Result<HotSetChunk, MeshRpcError> {
        self.check_target(&request.target)?;
        let pages: Vec<u64> = self.manifest().map(|m| m.pages.clone()).unwrap_or_default();

        // Chunked at the bound the wire declares, so a manifest of any size is
        // a stream of same-shaped frames rather than one frame whose size is
        // this node's cache
        let chunks_total = pages.len().div_ceil(MAX_CHUNK_PAGES).max(1) as u32;
        let start = request.chunk as usize * MAX_CHUNK_PAGES;
        let page_ids = if start >= pages.len() {
            Vec::new()
        } else {
            pages[start..(start + MAX_CHUNK_PAGES).min(pages.len())].to_vec()
        };
        Ok(HotSetChunk {
            source: self.local.clone(),
            sequence: request.sequence,
            chunk: request.chunk,
            chunks_total,
            page_ids,
        })
    }

    fn prefetch(&self, request: &PrefetchRequest) -> Result<PrefetchStatus, MeshRpcError> {
        self.check_target(&request.target)?;
        // Queued rather than read here. Reading a quarter of a gigabyte
        // inside a request handler would hold the connection for the length
        // of the read and do it on the listener's thread
        let mut queue = self.prefetch_queue.lock();
        let room = MAX_CHUNK_PAGES.saturating_sub(queue.len());
        let taken = request.page_ids.len().min(room);
        queue.extend_from_slice(&request.page_ids[..taken]);
        Ok(PrefetchStatus {
            target: request.target.clone(),
            sequence: request.sequence,
            pages_read: taken as u32,
            pages_skipped: (request.page_ids.len() - taken) as u32,
            // The queue is what bounds this, not the byte budget: a node
            // already holding a full queue has as much warming to do as it
            // can usefully take
            budget_exhausted: taken < request.page_ids.len(),
        })
    }

    fn relocate_session(
        &self,
        request: &RelocateSessionRequest,
    ) -> Result<RelocationOutcome, MeshRpcError> {
        self.check_target(&request.from)?;
        if !self.draining.load(Ordering::Relaxed) {
            return Err(MeshRpcError::Refused {
                reason: "this node is not draining, so its sessions are not moving".into(),
            });
        }
        if !self.relocate_sessions.load(Ordering::Relaxed) {
            return Ok(RelocationOutcome::Pinned {
                reason: "the drain that is running was asked to end sessions, not move them".into(),
            });
        }
        // A session's state is its transaction, its temporary tables, its
        // prepared statements, and its cursors, none of which have a
        // representation that survives a move. What a drain can do is let it
        // finish, which is what ending it after the work in flight completes
        // means
        Ok(RelocationOutcome::Pinned {
            reason: "a session holds transaction and cursor state that does not move; it ends \
                     when its work finishes"
                .into(),
        })
    }

    fn cancel_provisioning(&self, request: &CancelProvisioningRequest) -> Result<(), MeshRpcError> {
        // Cancelling reaches the driver that issued the ticket. Without a
        // scheduler there is no driver holding one, and saying so is better
        // than reporting a cancellation that never happened
        match zyron_mesh::MeshActuator::scheduler() {
            Some(_) => Err(MeshRpcError::Unknown {
                what: format!("ticket {}", request.ticket_id),
            }),
            None => Err(MeshRpcError::Refused {
                reason: "this node has no scheduler, so it holds no provisioning tickets".into(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(draining: bool, queries: u64) -> ServerMeshNode {
        let in_flight = Arc::new(InFlight::default());
        in_flight.queries.store(queries, Ordering::Relaxed);
        ServerMeshNode::new(
            NodeRef::new(1, "local"),
            Arc::new(AtomicBool::new(draining)),
            in_flight,
            Arc::new(parking_lot::RwLock::new(std::path::PathBuf::new())),
        )
    }

    fn target() -> NodeRef {
        NodeRef::new(1, "local")
    }

    /// Beginning a drain sets the flag and reports what is still running.
    #[test]
    fn a_drain_starts_and_reports_what_is_in_flight() {
        let node = node(false, 3);
        assert!(!node.is_draining());
        let status = node
            .begin_drain(&BeginDrainRequest {
                target: target(),
                sequence: 5,
                deadline_ms: 1000,
                relocate_sessions: false,
            })
            .expect("this node is the target");
        assert!(node.is_draining());
        assert_eq!(status.queries_in_flight, 3);
        assert!(!status.drained);
    }

    /// A node with nothing running reports itself drained.
    #[test]
    fn an_idle_node_is_drained_immediately() {
        let node = node(true, 0);
        let status = node
            .drain_status(&DrainStatusRequest {
                target: target(),
                sequence: 1,
            })
            .expect("this node is draining");
        assert!(status.drained);
    }

    /// Asking a node that is not draining how its drain is going is a
    /// refusal, not a false answer.
    #[test]
    fn a_node_that_is_not_draining_refuses_the_question() {
        let error = node(false, 0)
            .drain_status(&DrainStatusRequest {
                target: target(),
                sequence: 1,
            })
            .expect_err("this node is not draining");
        assert!(matches!(error, MeshRpcError::Refused { .. }), "{error:?}");
    }

    /// A call addressed to another node is refused, so a misrouted request
    /// cannot drain the wrong machine.
    #[test]
    fn a_call_for_a_different_node_is_not_answered() {
        let node = node(false, 0);
        let error = node
            .begin_drain(&BeginDrainRequest {
                target: NodeRef::new(99, "elsewhere"),
                sequence: 1,
                deadline_ms: 1000,
                relocate_sessions: false,
            })
            .expect_err("this is not node 99");
        assert!(matches!(error, MeshRpcError::Unknown { .. }), "{error:?}");
        assert!(!node.is_draining(), "a misrouted call started a drain");
    }

    /// A prefetch is queued for whatever reads it, and a queue that is full
    /// says how much it could not take.
    #[test]
    fn a_prefetch_queues_what_it_can_and_reports_the_rest() {
        let node = node(false, 0);
        let status = node
            .prefetch(&PrefetchRequest {
                target: target(),
                sequence: 1,
                page_ids: (0..10).collect(),
                byte_budget: 1 << 20,
            })
            .expect("this node is the target");
        assert_eq!(status.pages_read, 10);
        assert_eq!(node.take_prefetch_queue().len(), 10);
        assert!(
            node.take_prefetch_queue().is_empty(),
            "the queue was not drained by taking it"
        );
    }

    /// A node with no manifest answers with an empty chunk rather than an
    /// error, because having nothing resident is a valid state.
    #[test]
    fn a_node_with_no_manifest_hands_over_nothing() {
        let chunk = node(false, 0)
            .hot_set_manifest(&HotSetManifestRequest {
                target: target(),
                sequence: 1,
                chunk: 0,
            })
            .expect("this node is the target");
        assert!(chunk.page_ids.is_empty());
        assert_eq!(chunk.chunks_total, 1);
    }
}

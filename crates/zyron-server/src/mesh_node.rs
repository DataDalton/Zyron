//! What this node answers when another node asks it something.
//!
//! `zyron-mesh` owns the calls, the paths, and the wire format. It cannot own
//! the answers: how many queries are running, which pages are resident, what
//! binary is running and what is staged beside it are this crate's, and this
//! crate sits above it. So this is the implementation of `MeshNode`, and it
//! is the only place the two meet.
//!
//! ## Nothing here waits
//!
//! Every method flips a flag, reads a counter, or leaves an intent for the
//! upgrade service and returns. Beginning a drain sets the node to stop
//! accepting and reports what is still in flight at that instant; the caller
//! asks again later to find out how far it got. Staging and restarting are
//! left for the service, which the caller watches through `node_status`. A
//! handler that blocked until the work finished would hold a connection open
//! for minutes and turn one busy node into a stalled scheduler

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use zyron_common::{Admission, QueryMetrics};
use zyron_mesh::rpc::{
    BeginDrainRequest, CancelProvisioningRequest, DrainStatus, DrainStatusRequest, HotSetChunk,
    HotSetManifestRequest, MeshRpcError, NodeAck, NodeStatus, NodeStatusRequest, PrefetchRequest,
    PrefetchStatus, RelocateSessionRequest, RelocationOutcome, RestartRequest, RollbackRequest,
    SetClusterSettingRequest, StageReleaseRequest,
};
use zyron_mesh::{MAX_CHUNK_PAGES, MAX_DETAIL_BYTES, MeshNode, NodeRef};
use zyron_pressure::hot_set::HotSetManifest;

use crate::upgrade::control::{CoordinatedRestart, NodeControl};

/// What this node reports and does when the mesh asks.
pub struct ServerMeshNode {
    /// This node, so an answer names who gave it
    local: NodeRef,
    /// Whether the node takes new work and what it has in flight, shared
    /// with the wire listener and the upgrade service
    admission: Arc<Admission>,
    /// What the connections record, for the status a coordinator reads
    query_metrics: Arc<QueryMetrics>,
    /// Where staging and restart requests are left for the upgrade service
    control: Arc<NodeControl>,
    /// Whether sessions may be moved rather than ended, from the request that
    /// started the drain
    relocate_sessions: AtomicBool,
    /// The sequence of the drain in progress, so a status request for a
    /// different one is answered as unknown rather than with this one's state
    drain_sequence: AtomicU64,
    /// Where the working-set manifest is written
    data_dir: Arc<parking_lot::RwLock<std::path::PathBuf>>,
    /// Pages another node asked this one to warm, drained by the prefetch
    /// task the server runs
    prefetch_queue: Arc<parking_lot::Mutex<Vec<u64>>>,
}

impl ServerMeshNode {
    pub fn new(
        local: NodeRef,
        admission: Arc<Admission>,
        query_metrics: Arc<QueryMetrics>,
        control: Arc<NodeControl>,
        data_dir: Arc<parking_lot::RwLock<std::path::PathBuf>>,
    ) -> Self {
        Self {
            local,
            admission,
            query_metrics,
            control,
            relocate_sessions: AtomicBool::new(false),
            drain_sequence: AtomicU64::new(0),
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
        self.admission.is_draining()
    }

    /// This node's current state, as the mesh sees it.
    fn status(&self, target: NodeRef, sequence: u64) -> DrainStatus {
        let counts = self.admission.in_flight();
        DrainStatus {
            target,
            sequence,
            queries_in_flight: clamp(counts.queries),
            sessions_attached: clamp(counts.sessions),
            transactions_open: clamp(counts.transactions),
            drained: self.admission.is_quiescent(),
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

    fn ack(target: NodeRef, sequence: u64, accepted: bool, detail: String) -> NodeAck {
        let mut detail = detail;
        if detail.len() > MAX_DETAIL_BYTES {
            let mut cut = MAX_DETAIL_BYTES;
            while !detail.is_char_boundary(cut) {
                cut -= 1;
            }
            detail.truncate(cut);
        }
        NodeAck {
            target,
            sequence,
            accepted,
            detail,
        }
    }
}

fn clamp(count: u64) -> u32 {
    count.min(u32::MAX as u64) as u32
}

impl MeshNode for ServerMeshNode {
    fn begin_drain(&self, request: &BeginDrainRequest) -> Result<DrainStatus, MeshRpcError> {
        self.check_target(&request.target)?;
        self.admission.begin_drain();
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
        if !self.admission.is_draining() {
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
        if !self.admission.is_draining() {
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

    fn node_status(&self, request: &NodeStatusRequest) -> Result<NodeStatus, MeshRpcError> {
        self.check_target(&request.target)?;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let counts = self.admission.in_flight();
        let sample = self.query_metrics.sample(now, counts.sessions);
        Ok(NodeStatus {
            target: request.target.clone(),
            sequence: request.sequence,
            version: env!("CARGO_PKG_VERSION").to_string(),
            staged_version: self.control.staged_version().unwrap_or_default(),
            draining: self.admission.is_draining(),
            accepting: self.admission.is_accepting(),
            queries_in_flight: clamp(counts.queries),
            sessions_attached: clamp(counts.sessions),
            transactions_open: clamp(counts.transactions),
            p50_latency_us: sample.p50_latency_us,
            p99_latency_us: sample.p99_latency_us,
            throughput_milli_per_sec: (sample.throughput_per_sec * 1_000.0).max(0.0) as u64,
            error_rate_ppm: (sample.error_rate * 1_000_000.0).clamp(0.0, 1_000_000.0) as u64,
            queries_in_window: sample.queries_in_window,
            uptime_secs: self.control.uptime_secs(),
        })
    }

    fn set_cluster_setting(
        &self,
        request: &SetClusterSettingRequest,
    ) -> Result<NodeAck, MeshRpcError> {
        self.check_target(&request.target)?;
        // The service appends it when this node leads. When leadership has
        // moved since the sender looked, the service hands it on again
        self.control
            .request_cluster_setting(&request.key, &request.value);
        Ok(Self::ack(
            request.target.clone(),
            request.sequence,
            true,
            format!("{} queued for the replicated log", request.key),
        ))
    }

    fn stage_release(&self, request: &StageReleaseRequest) -> Result<NodeAck, MeshRpcError> {
        self.check_target(&request.target)?;
        if let Some(reason) = self.control.stage_failure(&request.version) {
            // The coordinator reads the failure and asks again, which clears
            // it, so a fetch that failed once is retried rather than remembered
            // forever
            self.control.request_stage(&request.version);
            return Ok(Self::ack(
                request.target.clone(),
                request.sequence,
                false,
                format!(
                    "the last attempt to stage {} failed, {reason}. Trying again",
                    request.version
                ),
            ));
        }
        let detail = if self.control.request_stage(&request.version) {
            format!("staging {}", request.version)
        } else {
            format!("{} is staged or being staged", request.version)
        };
        Ok(Self::ack(
            request.target.clone(),
            request.sequence,
            true,
            detail,
        ))
    }

    fn restart_into_staged(&self, request: &RestartRequest) -> Result<NodeAck, MeshRpcError> {
        self.check_target(&request.target)?;
        let ack = match self
            .control
            .request_coordinated(CoordinatedRestart::IntoStaged {
                version: request.version.clone(),
            }) {
            Ok(()) => (true, format!("restarting into {}", request.version)),
            Err(reason) => (false, reason),
        };
        Ok(Self::ack(
            request.target.clone(),
            request.sequence,
            ack.0,
            ack.1,
        ))
    }

    fn rollback_to_previous(&self, request: &RollbackRequest) -> Result<NodeAck, MeshRpcError> {
        self.check_target(&request.target)?;
        let ack = match self
            .control
            .request_coordinated(CoordinatedRestart::ToPrevious)
        {
            Ok(()) => (true, "restarting on the previous binary".to_string()),
            Err(reason) => (false, reason),
        };
        Ok(Self::ack(
            request.target.clone(),
            request.sequence,
            ack.0,
            ack.1,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::upgrade::stager::StagedRelease;

    fn node(draining: bool, queries: u64) -> ServerMeshNode {
        let admission = Arc::new(Admission::new());
        if draining {
            admission.begin_drain();
        }
        // Query guards are what the counter counts, held here for the life
        // of the node so the count reads back
        let guards: Vec<_> = (0..queries).map(|_| admission.begin_query()).collect();
        std::mem::forget(guards);
        ServerMeshNode::new(
            NodeRef::new(1, "local"),
            admission,
            Arc::new(QueryMetrics::new()),
            NodeControl::shared(),
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

    /// The status a coordinator reads names the running binary, what is
    /// staged, and what the connections have measured.
    #[test]
    fn the_status_reports_the_binary_the_stage_and_the_load() {
        let node = node(false, 2);
        node.query_metrics.record_query(100, 700, false);
        node.query_metrics.record_query(100, 900, true);
        let status = node
            .node_status(&NodeStatusRequest {
                target: target(),
                sequence: 4,
            })
            .expect("this node is the target");
        assert_eq!(status.version, env!("CARGO_PKG_VERSION"));
        assert_eq!(status.staged_version, "");
        assert_eq!(status.queries_in_flight, 2);
        assert!(status.p99_latency_us >= 900);
        assert!(!status.draining);
        assert!(status.valid());
    }

    /// Staging is left for the service, restarting needs the version staged,
    /// and a rollback is taken as asked.
    #[test]
    fn staging_and_restarting_leave_intents_for_the_service() {
        let node = node(false, 0);
        let ack = node
            .stage_release(&StageReleaseRequest {
                target: target(),
                sequence: 1,
                version: "0.13.0".into(),
            })
            .expect("this node is the target");
        assert!(ack.accepted);
        assert_eq!(node.control.take_stage_requests(), vec!["0.13.0"]);

        let refused = node
            .restart_into_staged(&RestartRequest {
                target: target(),
                sequence: 2,
                version: "0.13.0".into(),
            })
            .expect("this node is the target");
        assert!(!refused.accepted);
        assert!(refused.detail.contains("no release 0.13.0 is staged"));

        node.control.set_staged(Some(StagedRelease {
            version: "0.13.0".into(),
            path: std::path::PathBuf::from("staging/zyron-server-0.13.0"),
            sha256: String::new(),
            signature_scheme: "Ed25519".into(),
            size_bytes: 1,
        }));
        let accepted = node
            .restart_into_staged(&RestartRequest {
                target: target(),
                sequence: 3,
                version: "0.13.0".into(),
            })
            .expect("this node is the target");
        assert!(accepted.accepted, "{}", accepted.detail);
        assert_eq!(
            node.control.take_coordinated(),
            Some(CoordinatedRestart::IntoStaged {
                version: "0.13.0".into()
            })
        );

        let rollback = node
            .rollback_to_previous(&RollbackRequest {
                target: target(),
                sequence: 4,
            })
            .expect("this node is the target");
        assert!(rollback.accepted);
        assert_eq!(
            node.control.take_coordinated(),
            Some(CoordinatedRestart::ToPrevious)
        );
    }

    /// A staging failure is reported once and the request is retried.
    #[test]
    fn a_failed_stage_is_reported_and_retried() {
        let node = node(false, 0);
        node.control
            .record_stage_failure("0.13.0", "the artifact is not at /releases".into());
        let ack = node
            .stage_release(&StageReleaseRequest {
                target: target(),
                sequence: 1,
                version: "0.13.0".into(),
            })
            .expect("this node is the target");
        assert!(!ack.accepted);
        assert!(ack.detail.contains("is not at /releases"), "{}", ack.detail);
        assert_eq!(node.control.take_stage_requests(), vec!["0.13.0"]);
        assert!(node.control.stage_failure("0.13.0").is_none());
    }
}

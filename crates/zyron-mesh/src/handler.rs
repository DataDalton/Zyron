//! The receiving end of a mesh call.
//!
//! This crate owns the paths, the payloads, and the decoding. It does not own
//! the answers: what a node has in flight, what it holds in memory, and which
//! sessions are attached to it are the server's, and the server sits above
//! this crate. So the split is a trait the server implements and a dispatcher
//! this crate provides, which means the wire format has exactly one definition
//! and the two ends of it cannot drift.
//!
//! ## Every handler returns immediately
//!
//! [`MeshNode`] is synchronous, and each of its methods either flips a flag or
//! reads a counter. Nothing here waits for a drain to finish or for pages to
//! be read: the caller asked the node to start, and asks again later to find
//! out how far it got. A handler that blocked on the work would hold a
//! connection open for the length of a drain and turn one slow node into a
//! stalled scheduler.

use crate::rpc::{
    BeginDrainRequest, CancelProvisioningRequest, DrainStatus, DrainStatusRequest, HotSetChunk,
    HotSetManifestRequest, MeshRpcError, NodeAck, NodeStatus, NodeStatusRequest, PATH_BEGIN_DRAIN,
    PATH_CANCEL_PROVISIONING, PATH_DRAIN_STATUS, PATH_HOT_SET_MANIFEST, PATH_NODE_STATUS,
    PATH_PREFETCH, PATH_RELOCATE_SESSION, PATH_RESTART_INTO_STAGED, PATH_ROLLBACK_TO_PREVIOUS,
    PATH_SET_CLUSTER_SETTING, PATH_STAGE_RELEASE, PrefetchRequest, PrefetchStatus,
    RelocateSessionRequest, RelocationOutcome, RestartRequest, RollbackRequest,
    SetClusterSettingRequest, StageReleaseRequest,
};

/// What a node has to be able to answer to be part of a mesh.
///
/// Implemented by the server, which is the only place that knows how many
/// queries are running and which pages are resident.
pub trait MeshNode: Send + Sync {
    /// Stops taking new work and starts finishing what is in flight. Returns
    /// where the drain stands at this instant, which for a busy node is the
    /// beginning of it.
    fn begin_drain(&self, request: &BeginDrainRequest) -> Result<DrainStatus, MeshRpcError>;

    /// Where a drain has got to.
    fn drain_status(&self, request: &DrainStatusRequest) -> Result<DrainStatus, MeshRpcError>;

    /// One chunk of what this node has in memory.
    fn hot_set_manifest(
        &self,
        request: &HotSetManifestRequest,
    ) -> Result<HotSetChunk, MeshRpcError>;

    /// Queues pages to be read before the work that needs them arrives.
    /// Returns what was queued, not what has been read.
    fn prefetch(&self, request: &PrefetchRequest) -> Result<PrefetchStatus, MeshRpcError>;

    /// Moves a session to another node, or says why it cannot.
    fn relocate_session(
        &self,
        request: &RelocateSessionRequest,
    ) -> Result<RelocationOutcome, MeshRpcError>;

    /// Abandons a provisioning request this node has in flight.
    fn cancel_provisioning(&self, request: &CancelProvisioningRequest) -> Result<(), MeshRpcError>;

    /// What this node runs and how it is doing.
    fn node_status(&self, request: &NodeStatusRequest) -> Result<NodeStatus, MeshRpcError>;

    /// Takes a request to stage a release. The staging itself runs after
    /// the answer
    fn stage_release(&self, request: &StageReleaseRequest) -> Result<NodeAck, MeshRpcError>;

    /// Takes a cluster setting for the replicated log. The append runs after
    /// the answer, on the leader's next pass
    fn set_cluster_setting(
        &self,
        request: &SetClusterSettingRequest,
    ) -> Result<NodeAck, MeshRpcError>;

    /// Takes a request to restart on the staged binary. The restart runs
    /// after the answer
    fn restart_into_staged(&self, request: &RestartRequest) -> Result<NodeAck, MeshRpcError>;

    /// Takes a request to restart on the previous binary. The restart runs
    /// after the answer
    fn rollback_to_previous(&self, request: &RollbackRequest) -> Result<NodeAck, MeshRpcError>;
}

/// An HTTP status and a JSON body, ready for the listener to write.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshResponse {
    pub status: u16,
    pub body: String,
}

impl MeshResponse {
    fn ok(body: String) -> Self {
        Self { status: 200, body }
    }

    /// The status a fault maps to.
    ///
    /// Chosen so the caller's own error mapping recovers the same variant
    /// without reading the body: 404 becomes Unknown, everything else becomes
    /// Refused with the reason attached.
    fn from_error(error: MeshRpcError) -> Self {
        let status = match &error {
            MeshRpcError::Unknown { .. } => 404,
            MeshRpcError::Malformed { .. } => 400,
            MeshRpcError::Refused { .. } => 409,
            MeshRpcError::Unreachable { .. } | MeshRpcError::TimedOut { .. } => 503,
        };
        let body = serde_json::to_string(&error)
            .unwrap_or_else(|_| "{\"Refused\":{\"reason\":\"unencodable error\"}}".to_string());
        Self { status, body }
    }
}

/// Decodes a request, runs it, and encodes the answer.
///
/// A path that is not one of ours returns None, so the listener can keep
/// routing rather than being told about a 404 it did not ask for.
pub fn dispatch(node: &dyn MeshNode, path: &str, body: &str) -> Option<MeshResponse> {
    let answer = match path {
        PATH_BEGIN_DRAIN => run(body, |r: BeginDrainRequest| {
            require(r.valid(), "begin_drain.target")?;
            node.begin_drain(&r)
        }),
        PATH_DRAIN_STATUS => run(body, |r: DrainStatusRequest| {
            require(r.valid(), "drain_status.target")?;
            node.drain_status(&r)
        }),
        PATH_HOT_SET_MANIFEST => run(body, |r: HotSetManifestRequest| {
            require(r.valid(), "hot_set_manifest.target")?;
            let chunk = node.hot_set_manifest(&r)?;
            // Bounded on the way out as well as on the way in, so a node
            // cannot make a peer allocate past the bound this crate declares
            require(chunk.valid(), "hot_set_manifest.page_ids")?;
            Ok(chunk)
        }),
        PATH_PREFETCH => run(body, |r: PrefetchRequest| {
            require(r.valid(), "prefetch.page_ids")?;
            node.prefetch(&r)
        }),
        PATH_RELOCATE_SESSION => run(body, |r: RelocateSessionRequest| {
            require(r.valid(), "relocate_session.session_id")?;
            node.relocate_session(&r)
        }),
        PATH_CANCEL_PROVISIONING => run(body, |r: CancelProvisioningRequest| {
            require(r.valid(), "cancel_provisioning.ticket_id")?;
            node.cancel_provisioning(&r)
        }),
        PATH_NODE_STATUS => run(body, |r: NodeStatusRequest| {
            require(r.valid(), "node_status.target")?;
            let status = node.node_status(&r)?;
            require(status.valid(), "node_status.version")?;
            Ok(status)
        }),
        PATH_STAGE_RELEASE => run(body, |r: StageReleaseRequest| {
            require(r.valid(), "stage_release.version")?;
            let ack = node.stage_release(&r)?;
            require(ack.valid(), "stage_release.detail")?;
            Ok(ack)
        }),
        PATH_SET_CLUSTER_SETTING => run(body, |r: SetClusterSettingRequest| {
            require(r.valid(), "set_cluster_setting.key")?;
            let ack = node.set_cluster_setting(&r)?;
            require(ack.valid(), "set_cluster_setting.detail")?;
            Ok(ack)
        }),
        PATH_RESTART_INTO_STAGED => run(body, |r: RestartRequest| {
            require(r.valid(), "restart_into_staged.version")?;
            let ack = node.restart_into_staged(&r)?;
            require(ack.valid(), "restart_into_staged.detail")?;
            Ok(ack)
        }),
        PATH_ROLLBACK_TO_PREVIOUS => run(body, |r: RollbackRequest| {
            require(r.valid(), "rollback_to_previous.target")?;
            let ack = node.rollback_to_previous(&r)?;
            require(ack.valid(), "rollback_to_previous.detail")?;
            Ok(ack)
        }),
        _ => return None,
    };
    Some(answer)
}

/// Fails a request whose payload broke its own bound.
fn require(ok: bool, field: &str) -> Result<(), MeshRpcError> {
    if ok {
        Ok(())
    } else {
        Err(MeshRpcError::Malformed {
            field: field.to_string(),
        })
    }
}

/// Decodes one payload, runs the handler, encodes the answer.
fn run<Q, A, F>(body: &str, handler: F) -> MeshResponse
where
    Q: serde::de::DeserializeOwned,
    A: serde::Serialize,
    F: FnOnce(Q) -> Result<A, MeshRpcError>,
{
    let request = match serde_json::from_str::<Q>(body) {
        Ok(request) => request,
        Err(e) => {
            return MeshResponse::from_error(MeshRpcError::Malformed {
                field: e.to_string(),
            });
        }
    };
    match handler(request) {
        Ok(answer) => match serde_json::to_string(&answer) {
            Ok(text) => MeshResponse::ok(text),
            Err(e) => MeshResponse::from_error(MeshRpcError::Malformed {
                field: e.to_string(),
            }),
        },
        Err(e) => MeshResponse::from_error(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rpc::NodeRef;

    struct Node {
        drained: bool,
    }

    impl MeshNode for Node {
        fn begin_drain(&self, request: &BeginDrainRequest) -> Result<DrainStatus, MeshRpcError> {
            Ok(DrainStatus {
                target: request.target.clone(),
                sequence: request.sequence,
                queries_in_flight: if self.drained { 0 } else { 3 },
                sessions_attached: 0,
                transactions_open: 0,
                drained: self.drained,
            })
        }
        fn drain_status(&self, request: &DrainStatusRequest) -> Result<DrainStatus, MeshRpcError> {
            Ok(DrainStatus::idle(request.target.clone(), request.sequence))
        }
        fn hot_set_manifest(
            &self,
            request: &HotSetManifestRequest,
        ) -> Result<HotSetChunk, MeshRpcError> {
            Ok(HotSetChunk {
                source: request.target.clone(),
                sequence: request.sequence,
                chunk: request.chunk,
                chunks_total: 1,
                page_ids: vec![1, 2, 3],
            })
        }
        fn prefetch(&self, request: &PrefetchRequest) -> Result<PrefetchStatus, MeshRpcError> {
            Ok(PrefetchStatus {
                target: request.target.clone(),
                sequence: request.sequence,
                pages_read: request.page_ids.len() as u32,
                pages_skipped: 0,
                budget_exhausted: false,
            })
        }
        fn relocate_session(
            &self,
            _request: &RelocateSessionRequest,
        ) -> Result<RelocationOutcome, MeshRpcError> {
            Ok(RelocationOutcome::Ended)
        }
        fn cancel_provisioning(
            &self,
            _request: &CancelProvisioningRequest,
        ) -> Result<(), MeshRpcError> {
            Err(MeshRpcError::Unknown {
                what: "ticket".into(),
            })
        }
        fn node_status(&self, request: &NodeStatusRequest) -> Result<NodeStatus, MeshRpcError> {
            Ok(NodeStatus {
                target: request.target.clone(),
                sequence: request.sequence,
                version: "0.12.0".into(),
                staged_version: String::new(),
                draining: false,
                accepting: true,
                queries_in_flight: if self.drained { 0 } else { 3 },
                sessions_attached: 0,
                transactions_open: 0,
                p50_latency_us: 100,
                p99_latency_us: 1_000,
                throughput_milli_per_sec: 10_000,
                error_rate_ppm: 0,
                queries_in_window: 600,
                uptime_secs: 60,
                upgrade: None,
            })
        }
        fn stage_release(&self, request: &StageReleaseRequest) -> Result<NodeAck, MeshRpcError> {
            Ok(NodeAck {
                target: request.target.clone(),
                sequence: request.sequence,
                accepted: true,
                detail: format!("staging {}", request.version),
            })
        }
        fn set_cluster_setting(
            &self,
            request: &SetClusterSettingRequest,
        ) -> Result<NodeAck, MeshRpcError> {
            Ok(NodeAck {
                target: request.target.clone(),
                sequence: request.sequence,
                accepted: true,
                detail: format!("{} = {} queued", request.key, request.value),
            })
        }
        fn restart_into_staged(&self, request: &RestartRequest) -> Result<NodeAck, MeshRpcError> {
            Ok(NodeAck {
                target: request.target.clone(),
                sequence: request.sequence,
                accepted: false,
                detail: format!("no release {} is staged here", request.version),
            })
        }
        fn rollback_to_previous(&self, request: &RollbackRequest) -> Result<NodeAck, MeshRpcError> {
            Ok(NodeAck {
                target: request.target.clone(),
                sequence: request.sequence,
                accepted: true,
                detail: "restarting on the previous binary".into(),
            })
        }
    }

    fn node() -> NodeRef {
        NodeRef::new(2, "peer")
    }

    /// A path this crate does not own is left for the listener to route.
    #[test]
    fn a_foreign_path_is_not_answered_here() {
        assert!(dispatch(&Node { drained: true }, "/pressure", "").is_none());
    }

    /// A call round trips through the dispatcher and comes back decodable by
    /// the same types the client uses.
    #[test]
    fn a_call_answers_with_what_the_client_expects() {
        let request = DrainStatusRequest {
            target: node(),
            sequence: 42,
        };
        let body = serde_json::to_string(&request).expect("encode");
        let answer = dispatch(&Node { drained: true }, PATH_DRAIN_STATUS, &body)
            .expect("the dispatcher owns this path");
        assert_eq!(answer.status, 200);
        let status: DrainStatus = serde_json::from_str(&answer.body).expect("decode");
        assert_eq!(status.sequence, 42);
        assert!(status.drained);
    }

    /// A body that is not the payload is a four hundred, and the reason names
    /// what could not be read.
    #[test]
    fn an_undecodable_body_is_a_bad_request() {
        let answer = dispatch(&Node { drained: true }, PATH_BEGIN_DRAIN, "not json")
            .expect("the dispatcher owns this path");
        assert_eq!(answer.status, 400);
        let error: MeshRpcError = serde_json::from_str(&answer.body).expect("decode");
        assert!(matches!(error, MeshRpcError::Malformed { .. }), "{error:?}");
    }

    /// A payload past its bound is refused by the dispatcher rather than by
    /// the node, so a handler never sees one.
    #[test]
    fn an_oversized_payload_never_reaches_the_node() {
        let request = PrefetchRequest {
            target: node(),
            sequence: 1,
            page_ids: vec![0; crate::rpc::MAX_CHUNK_PAGES + 1],
            byte_budget: 1,
        };
        let body = serde_json::to_string(&request).expect("encode");
        let answer = dispatch(&Node { drained: true }, PATH_PREFETCH, &body).expect("path is ours");
        assert_eq!(answer.status, 400);
    }

    /// A fault maps to a status the caller can turn back into the same
    /// variant without parsing the body.
    #[test]
    fn an_unknown_thing_answers_four_oh_four() {
        let request = CancelProvisioningRequest {
            sequence: 1,
            ticket_id: "t-1".into(),
        };
        let body = serde_json::to_string(&request).expect("encode");
        let answer = dispatch(&Node { drained: true }, PATH_CANCEL_PROVISIONING, &body)
            .expect("path is ours");
        assert_eq!(answer.status, 404);
    }
}

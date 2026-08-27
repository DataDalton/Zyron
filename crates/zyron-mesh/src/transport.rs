//! The mesh calls, carried over the HTTP the node already serves.
//!
//! ## Why this transport
//!
//! Every node already listens on its health port and already answers
//! `/pressure` and the hot-set manifest there. A mesh call is the same shape
//! of thing: small, infrequent, and about the node rather than about data. So
//! the calls go there rather than waiting for a protocol that does not exist,
//! and [`crate::rpc::MeshRpc`] is what keeps that a transport decision instead
//! of a scheduler decision.
//!
//! ## One connection per call
//!
//! Deliberate. These are control-plane calls at controller cadence, a handful
//! per minute per peer, and a pool would hold an idle socket open to every
//! node in the mesh to save a connect on a call that already costs a round
//! trip and a JSON parse. The server closes the connection when it answers,
//! which is what makes reading to end-of-stream the whole body.
//!
//! ## Timeouts are the caller's
//!
//! Every call takes a deadline from the request or from the client's default,
//! and enforces it around the whole exchange rather than around each read. A
//! peer that accepts a connection and then stops writing is the failure this
//! catches, and a per-read timeout would let it hold the caller forever by
//! sending one byte at a time.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use serde::Serialize;
use serde::de::DeserializeOwned;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use crate::rpc::{
    BeginDrainRequest, CancelProvisioningRequest, DrainStatus, DrainStatusRequest, HotSetChunk,
    HotSetManifestRequest, MeshFuture, MeshRpc, MeshRpcError, NodeRef, PATH_BEGIN_DRAIN,
    PATH_CANCEL_PROVISIONING, PATH_DRAIN_STATUS, PATH_HOT_SET_MANIFEST, PATH_PREFETCH,
    PATH_RELOCATE_SESSION, PrefetchRequest, PrefetchStatus, RelocateSessionRequest,
    RelocationOutcome,
};

/// How long a call waits before giving up, when the request does not say.
///
/// Five seconds: long enough for a loaded peer to answer a question it
/// answers from memory, short enough that a scheduler polling a drain does
/// not stall behind one unreachable node.
pub const DEFAULT_CALL_TIMEOUT: Duration = Duration::from_secs(5);

/// Largest response body a peer may send.
///
/// A hot-set chunk is the biggest of these and is bounded at a thousand page
/// ids. A megabyte is far past that and stops a peer, or something answering
/// on its port, from making the caller allocate without limit.
const MAX_RESPONSE_BYTES: usize = 1024 * 1024;

/// Where each node answers, and how to reach it.
///
/// Addresses come from the peer registry the operator maintains, resolved once
/// when the client is built rather than looked up per call: a call is made
/// from a background task that must not block on a registry lock.
#[derive(Debug, Clone, Default)]
pub struct MeshDirectory {
    by_name: HashMap<String, String>,
    by_id: HashMap<u64, String>,
}

impl MeshDirectory {
    pub fn new() -> Self {
        Self::default()
    }

    /// Records where a node answers. The address is `host:port` of its health
    /// listener, which is the port the mesh paths are served on.
    pub fn insert(&mut self, node: &NodeRef, address: impl Into<String>) {
        let address = address.into();
        if node.node_id != 0 {
            self.by_id.insert(node.node_id, address.clone());
        }
        self.by_name.insert(node.name.clone(), address);
    }

    /// Where a node answers, by id if it has been contacted and by name
    /// otherwise.
    ///
    /// The id is preferred because a name is operator-chosen and can be
    /// reused, while the id is minted once and persists with the data
    /// directory.
    pub fn address(&self, node: &NodeRef) -> Option<&str> {
        if node.node_id != 0 {
            if let Some(address) = self.by_id.get(&node.node_id) {
                return Some(address.as_str());
            }
        }
        self.by_name.get(&node.name).map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.by_name.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_name.is_empty()
    }
}

/// What the transport has done, for the mesh views.
#[derive(Debug, Default)]
pub struct TransportStats {
    pub calls: AtomicU64,
    pub failures: AtomicU64,
    pub timeouts: AtomicU64,
    pub bytes_sent: AtomicU64,
    pub bytes_received: AtomicU64,
}

/// Speaks the mesh calls to other nodes over HTTP.
pub struct HttpMeshRpc {
    directory: Arc<MeshDirectory>,
    timeout: Duration,
    stats: Arc<TransportStats>,
}

impl HttpMeshRpc {
    pub fn new(directory: MeshDirectory) -> Self {
        Self {
            directory: Arc::new(directory),
            timeout: DEFAULT_CALL_TIMEOUT,
            stats: Arc::new(TransportStats::default()),
        }
    }

    /// Overrides how long a call waits.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn stats(&self) -> Arc<TransportStats> {
        Arc::clone(&self.stats)
    }

    pub fn directory(&self) -> &MeshDirectory {
        &self.directory
    }

    /// Sends one call and decodes its answer.
    async fn call<Q: Serialize, A: DeserializeOwned>(
        &self,
        target: &NodeRef,
        path: &'static str,
        request: &Q,
        timeout: Duration,
    ) -> Result<A, MeshRpcError> {
        let Some(address) = self.directory.address(target) else {
            return Err(MeshRpcError::Unknown {
                what: format!("node {}", target.name),
            });
        };
        let body = serde_json::to_string(request).map_err(|e| MeshRpcError::Malformed {
            field: e.to_string(),
        })?;

        self.stats.calls.fetch_add(1, Ordering::Relaxed);
        let outcome = tokio::time::timeout(
            timeout,
            exchange(address, path, &body, Arc::clone(&self.stats)),
        )
        .await;

        let text = match outcome {
            Ok(Ok(text)) => text,
            Ok(Err(e)) => {
                self.stats.failures.fetch_add(1, Ordering::Relaxed);
                return Err(e);
            }
            Err(_) => {
                self.stats.failures.fetch_add(1, Ordering::Relaxed);
                self.stats.timeouts.fetch_add(1, Ordering::Relaxed);
                return Err(MeshRpcError::TimedOut {
                    after_ms: timeout.as_millis().min(u32::MAX as u128) as u32,
                });
            }
        };

        serde_json::from_str(&text).map_err(|e| {
            self.stats.failures.fetch_add(1, Ordering::Relaxed);
            MeshRpcError::Malformed {
                field: format!("{path}: {e}"),
            }
        })
    }

    /// The deadline for a call that carries one, capped by the client's own.
    ///
    /// A peer that is given a long deadline still has to answer inside what
    /// this node is willing to wait, because the deadline in the request is
    /// how long the *peer* may take to finish draining, not how long it may
    /// take to acknowledge the request.
    fn acknowledge_timeout(&self) -> Duration {
        self.timeout
    }
}

/// Connects, sends one request, and reads the whole answer.
async fn exchange(
    address: &str,
    path: &str,
    body: &str,
    stats: Arc<TransportStats>,
) -> Result<String, MeshRpcError> {
    let mut stream = TcpStream::connect(address)
        .await
        .map_err(|e| MeshRpcError::Unreachable {
            node: address.to_string(),
            reason: e.to_string(),
        })?;
    // Control-plane calls are one small write and one small read, so waiting
    // for a full segment before sending would add a round trip to every one
    let _ = stream.set_nodelay(true);

    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream
        .write_all(request.as_bytes())
        .await
        .map_err(|e| MeshRpcError::Unreachable {
            node: address.to_string(),
            reason: e.to_string(),
        })?;
    stats
        .bytes_sent
        .fetch_add(request.len() as u64, Ordering::Relaxed);

    // The server answers with Connection: close, so end of stream is end of
    // body and no chunked decoding is needed
    let mut raw = Vec::with_capacity(4096);
    let mut chunk = [0u8; 8192];
    loop {
        let read = stream
            .read(&mut chunk)
            .await
            .map_err(|e| MeshRpcError::Unreachable {
                node: address.to_string(),
                reason: e.to_string(),
            })?;
        if read == 0 {
            break;
        }
        if raw.len() + read > MAX_RESPONSE_BYTES {
            return Err(MeshRpcError::Malformed {
                field: format!("response past {MAX_RESPONSE_BYTES} bytes"),
            });
        }
        raw.extend_from_slice(&chunk[..read]);
    }
    stats
        .bytes_received
        .fetch_add(raw.len() as u64, Ordering::Relaxed);

    let text = String::from_utf8(raw).map_err(|_| MeshRpcError::Malformed {
        field: "response is not utf8".into(),
    })?;
    let (head, answer) = text
        .split_once("\r\n\r\n")
        .ok_or_else(|| MeshRpcError::Malformed {
            field: "response has no header terminator".into(),
        })?;
    let status = head
        .split_whitespace()
        .nth(1)
        .and_then(|code| code.parse::<u16>().ok())
        .ok_or_else(|| MeshRpcError::Malformed {
            field: "response has no status code".into(),
        })?;

    match status {
        200..=299 => Ok(answer.to_string()),
        404 => Err(MeshRpcError::Unknown {
            what: format!("{path} on {address}"),
        }),
        _ => Err(MeshRpcError::Refused {
            reason: format!("{status}: {}", answer.trim()),
        }),
    }
}

impl MeshRpc for HttpMeshRpc {
    fn begin_drain(&self, request: BeginDrainRequest) -> MeshFuture<'_, DrainStatus> {
        Box::pin(async move {
            if !request.valid() {
                return Err(MeshRpcError::Malformed {
                    field: "begin_drain.target".into(),
                });
            }
            let target = request.target.clone();
            self.call(
                &target,
                PATH_BEGIN_DRAIN,
                &request,
                self.acknowledge_timeout(),
            )
            .await
        })
    }

    fn drain_status(&self, request: DrainStatusRequest) -> MeshFuture<'_, DrainStatus> {
        Box::pin(async move {
            if !request.valid() {
                return Err(MeshRpcError::Malformed {
                    field: "drain_status.target".into(),
                });
            }
            let target = request.target.clone();
            self.call(
                &target,
                PATH_DRAIN_STATUS,
                &request,
                self.acknowledge_timeout(),
            )
            .await
        })
    }

    fn hot_set_manifest(&self, request: HotSetManifestRequest) -> MeshFuture<'_, HotSetChunk> {
        Box::pin(async move {
            if !request.valid() {
                return Err(MeshRpcError::Malformed {
                    field: "hot_set_manifest.target".into(),
                });
            }
            let target = request.target.clone();
            let chunk: HotSetChunk = self
                .call(
                    &target,
                    PATH_HOT_SET_MANIFEST,
                    &request,
                    self.acknowledge_timeout(),
                )
                .await?;
            // Checked on arrival as well as on the way out, because a peer
            // running a build with a larger bound would otherwise make this
            // node allocate past its own
            if !chunk.valid() {
                return Err(MeshRpcError::Malformed {
                    field: "hot_set_manifest.page_ids".into(),
                });
            }
            Ok(chunk)
        })
    }

    fn prefetch(&self, request: PrefetchRequest) -> MeshFuture<'_, PrefetchStatus> {
        Box::pin(async move {
            if !request.valid() {
                return Err(MeshRpcError::Malformed {
                    field: "prefetch.page_ids".into(),
                });
            }
            let target = request.target.clone();
            self.call(&target, PATH_PREFETCH, &request, self.acknowledge_timeout())
                .await
        })
    }

    fn relocate_session(
        &self,
        request: RelocateSessionRequest,
    ) -> MeshFuture<'_, RelocationOutcome> {
        Box::pin(async move {
            if !request.valid() {
                return Err(MeshRpcError::Malformed {
                    field: "relocate_session.session_id".into(),
                });
            }
            let target = request.from.clone();
            self.call(
                &target,
                PATH_RELOCATE_SESSION,
                &request,
                self.acknowledge_timeout(),
            )
            .await
        })
    }

    fn cancel_provisioning(&self, request: CancelProvisioningRequest) -> MeshFuture<'_, ()> {
        Box::pin(async move {
            if !request.valid() {
                return Err(MeshRpcError::Malformed {
                    field: "cancel_provisioning.ticket_id".into(),
                });
            }
            // Cancelling is asked of whichever node is holding the ticket,
            // which is this one: the provisioner is local and the call exists
            // so a scheduler elsewhere in the mesh can reach it
            let target = NodeRef::new(0, "self");
            match self.directory.address(&target) {
                Some(_) => {
                    self.call::<_, ()>(
                        &target,
                        PATH_CANCEL_PROVISIONING,
                        &request,
                        self.acknowledge_timeout(),
                    )
                    .await
                }
                None => Err(MeshRpcError::Unknown {
                    what: "the node holding the ticket".into(),
                }),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_node_is_found_by_id_first_and_by_name_otherwise() {
        let mut directory = MeshDirectory::new();
        directory.insert(&NodeRef::new(9, "alpha"), "10.0.0.9:8080");
        directory.insert(&NodeRef::new(0, "beta"), "10.0.0.8:8080");

        // Known id wins even when the name was reused elsewhere
        assert_eq!(
            directory.address(&NodeRef::new(9, "renamed")),
            Some("10.0.0.9:8080")
        );
        // A peer never contacted has no id, so the name is all there is
        assert_eq!(
            directory.address(&NodeRef::new(0, "beta")),
            Some("10.0.0.8:8080")
        );
        assert_eq!(directory.address(&NodeRef::new(0, "gamma")), None);
    }

    /// A node the directory does not know is reported as unknown rather than
    /// unreachable, because retrying will not help.
    #[tokio::test]
    async fn a_call_to_an_unknown_node_is_not_retryable() {
        let client = HttpMeshRpc::new(MeshDirectory::new());
        let error = client
            .drain_status(DrainStatusRequest {
                target: NodeRef::new(1, "missing"),
                sequence: 1,
            })
            .await
            .expect_err("a node that is not in the directory cannot be called");
        assert!(matches!(error, MeshRpcError::Unknown { .. }), "{error:?}");
        assert!(!error.transient());
    }

    /// A payload that breaks its own bound never reaches the wire.
    #[tokio::test]
    async fn an_oversized_payload_is_refused_before_it_is_sent() {
        let mut directory = MeshDirectory::new();
        directory.insert(&NodeRef::new(1, "peer"), "127.0.0.1:1");
        let client = HttpMeshRpc::new(directory);
        let error = client
            .prefetch(PrefetchRequest {
                target: NodeRef::new(1, "peer"),
                sequence: 1,
                page_ids: vec![0; crate::rpc::MAX_CHUNK_PAGES + 1],
                byte_budget: 1,
            })
            .await
            .expect_err("a chunk past the bound is not sendable");
        assert!(matches!(error, MeshRpcError::Malformed { .. }), "{error:?}");
        assert_eq!(
            client.stats().calls.load(Ordering::Relaxed),
            0,
            "an unsendable payload still opened a connection"
        );
    }

    /// A peer that is not listening is unreachable, which is retryable.
    #[tokio::test]
    async fn a_peer_that_is_not_listening_is_retryable() {
        let mut directory = MeshDirectory::new();
        // Port zero never accepts, which is the point
        directory.insert(&NodeRef::new(1, "down"), "127.0.0.1:1");
        let client =
            HttpMeshRpc::new(directory).with_timeout(std::time::Duration::from_millis(500));
        let error = client
            .drain_status(DrainStatusRequest {
                target: NodeRef::new(1, "down"),
                sequence: 1,
            })
            .await
            .expect_err("nothing is listening there");
        assert!(error.transient(), "{error:?}");
        assert_eq!(client.stats().failures.load(Ordering::Relaxed), 1);
    }
}

//! A mesh call actually crosses a socket.
//!
//! Every other test of this proves one half: the client encodes what the
//! dispatcher decodes, or the node answers what the handler asks it. This one
//! runs both halves against each other over a real listener, because the two
//! halves agreeing about a struct is not the same as them agreeing about the
//! bytes, the path, the status codes, and where the body starts.
//!
//! The listener here is the same routing the server uses, reached through
//! `zyron_mesh::dispatch`, so a change to either end shows up here rather than
//! the first time two nodes talk.
//!
//! Run: cargo test -p zyron-server --test mesh_rpc_test

use std::sync::Arc;
use std::sync::atomic::Ordering;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

use zyron_common::{Admission, InFlightGuard, QueryMetrics};
use zyron_mesh::rpc::{
    BeginDrainRequest, DrainStatusRequest, HotSetManifestRequest, MeshRpc, MeshRpcError,
    PrefetchRequest,
};
use zyron_mesh::{HttpMeshRpc, MeshDirectory, MeshNode, NodeRef};
use zyron_server::mesh_node::ServerMeshNode;
use zyron_server::upgrade::control::NodeControl;

/// Serves mesh calls the way the health listener does, and stops when the
/// test drops the handle.
async fn serve(node: Arc<ServerMeshNode>) -> (String, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let address = listener.local_addr().expect("addr").to_string();
    let handle = tokio::spawn(async move {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                return;
            };
            let node = Arc::clone(&node);
            tokio::spawn(async move {
                let mut buffer = vec![0u8; 64 * 1024];
                let Ok(read) = stream.read(&mut buffer).await else {
                    return;
                };
                let request = String::from_utf8_lossy(&buffer[..read]).into_owned();
                let path = request
                    .lines()
                    .next()
                    .and_then(|line| line.split_whitespace().nth(1))
                    .unwrap_or("/")
                    .to_string();
                let body = request.split_once("\r\n\r\n").map(|(_, b)| b).unwrap_or("");

                let (status, answer) = match zyron_mesh::dispatch(node.as_ref(), &path, body) {
                    Some(response) => (response.status, response.body),
                    None => (404, "{}".to_string()),
                };
                let response = format!(
                    "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\n\
                     Content-Length: {}\r\nConnection: close\r\n\r\n{answer}",
                    answer.len()
                );
                let _ = stream.write_all(response.as_bytes()).await;
            });
        }
    });
    (address, handle)
}

fn local() -> NodeRef {
    NodeRef::new(1, "served")
}

/// A node with `queries` statements running on it, and the guards that keep
/// them running.
///
/// The count is not a field to be set: a query is in flight because something
/// holds a guard, so the guards come back with the node and the caller keeps
/// them for as long as the count is supposed to hold. Its own `NodeControl`
/// rather than the shared one, so a drain one test starts is invisible to the
/// next
fn node(queries: u64) -> (Arc<ServerMeshNode>, Vec<InFlightGuard>) {
    let admission = Arc::new(Admission::new());
    let held: Vec<InFlightGuard> = (0..queries).map(|_| admission.begin_query()).collect();
    let node = Arc::new(ServerMeshNode::new(
        local(),
        admission,
        Arc::new(QueryMetrics::new()),
        Arc::new(NodeControl::new()),
        Arc::new(parking_lot::RwLock::new(std::path::PathBuf::new())),
    ));
    (node, held)
}

fn client(address: &str) -> HttpMeshRpc {
    let mut directory = MeshDirectory::new();
    directory.insert(&local(), address);
    HttpMeshRpc::new(directory).with_timeout(std::time::Duration::from_secs(5))
}

/// A drain begins over the wire, and the answer is the node's real state.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_drain_crosses_the_wire_and_comes_back_with_the_nodes_state() {
    let (served, _held) = node(4);
    let (address, handle) = serve(Arc::clone(&served)).await;
    let client = client(&address);

    assert!(!served.is_draining());
    let status = client
        .begin_drain(BeginDrainRequest {
            target: local(),
            sequence: 77,
            deadline_ms: 30_000,
            relocate_sessions: true,
        })
        .await
        .expect("the node answers");

    assert_eq!(status.sequence, 77, "the answer lost the sequence");
    assert_eq!(status.queries_in_flight, 4);
    assert!(!status.drained);
    assert!(
        served.is_draining(),
        "the call was answered without starting the drain"
    );
    handle.abort();
}

/// A node that is not draining refuses the status question, and the refusal
/// survives the round trip as a refusal rather than as a transport fault.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_refusal_arrives_as_a_refusal() {
    let (served, _held) = node(0);
    let (address, handle) = serve(served).await;
    let client = client(&address);

    let error = client
        .drain_status(DrainStatusRequest {
            target: local(),
            sequence: 1,
        })
        .await
        .expect_err("this node is not draining");
    assert!(matches!(error, MeshRpcError::Refused { .. }), "{error:?}");
    assert!(
        !error.transient(),
        "a refusal was reported as worth retrying"
    );
    handle.abort();
}

/// A prefetch reaches the node's queue, and the count comes back.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_prefetch_reaches_the_node_and_is_queued() {
    let (served, _held) = node(0);
    let (address, handle) = serve(Arc::clone(&served)).await;
    let client = client(&address);

    let status = client
        .prefetch(PrefetchRequest {
            target: local(),
            sequence: 5,
            page_ids: (100..140).collect(),
            byte_budget: 1 << 20,
        })
        .await
        .expect("the node answers");
    assert_eq!(status.pages_read, 40);
    assert_eq!(
        served.take_prefetch_queue().len(),
        40,
        "the pages never reached the node"
    );
    handle.abort();
}

/// A node with nothing resident answers the manifest call with an empty
/// chunk, which is what tells a scheduler there is nothing to hand over.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_empty_manifest_is_an_answer_not_an_error() {
    let (served, _held) = node(0);
    let (address, handle) = serve(served).await;
    let client = client(&address);

    let chunk = client
        .hot_set_manifest(HotSetManifestRequest {
            target: local(),
            sequence: 9,
            chunk: 0,
        })
        .await
        .expect("the node answers");
    assert!(chunk.page_ids.is_empty());
    assert_eq!(chunk.chunks_total, 1);
    assert_eq!(chunk.sequence, 9);
    handle.abort();
}

/// A call for a different node is refused by the node that received it, so a
/// misrouted request cannot drain the wrong machine.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_misrouted_call_is_refused_by_the_node_that_got_it() {
    let (served, _held) = node(0);
    let (address, handle) = serve(Arc::clone(&served)).await;

    // The directory points a different node's name at this listener, which is
    // exactly what a stale address book does
    let mut directory = MeshDirectory::new();
    let elsewhere = NodeRef::new(42, "elsewhere");
    directory.insert(&elsewhere, &address);
    let client = HttpMeshRpc::new(directory);

    let error = client
        .begin_drain(BeginDrainRequest {
            target: elsewhere,
            sequence: 1,
            deadline_ms: 1000,
            relocate_sessions: false,
        })
        .await
        .expect_err("this listener is not node 42");
    assert!(matches!(error, MeshRpcError::Unknown { .. }), "{error:?}");
    assert!(
        !served.is_draining(),
        "a misrouted call drained the wrong node"
    );
    handle.abort();
}

/// The transport counts what it did, so the mesh views have something to
/// report.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_transport_counts_its_calls_and_its_bytes() {
    let (served, _held) = node(0);
    let (address, handle) = serve(served).await;
    let client = client(&address);
    let stats = client.stats();

    client
        .hot_set_manifest(HotSetManifestRequest {
            target: local(),
            sequence: 1,
            chunk: 0,
        })
        .await
        .expect("the node answers");

    assert_eq!(stats.calls.load(Ordering::Relaxed), 1);
    assert_eq!(stats.failures.load(Ordering::Relaxed), 0);
    assert!(stats.bytes_sent.load(Ordering::Relaxed) > 0);
    assert!(stats.bytes_received.load(Ordering::Relaxed) > 0);
    handle.abort();
}

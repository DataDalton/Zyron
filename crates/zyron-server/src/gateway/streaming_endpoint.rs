// -----------------------------------------------------------------------------
// Streaming endpoint integration.
//
// Bridges a dynamic WebSocket or SSE endpoint to a local publication channel.
// The concrete subscription is owned by the CDC/publication manager. This
// module formats outbound batches according to the endpoint's message_format
// and applies the configured backpressure policy.
// -----------------------------------------------------------------------------

use std::collections::VecDeque;

use super::sse;
use super::websocket::{WsOpcode, encode_frame};
use zyron_catalog::schema::{BackpressurePolicy, EndpointMessageFormat};

/// Transport kind selected by the client during the upgrade.
pub enum WsOrSse {
    WebSocket,
    Sse,
}

/// Outbound message with associated byte buffer.
pub struct OutgoingMessage {
    pub bytes: Vec<u8>,
    pub sequence: u64,
}

/// Queue with a bounded capacity and a configurable overflow policy. The
/// runtime wires this queue to both a publication subscriber task and the
/// transport writer task.
pub struct StreamQueue {
    queue: VecDeque<OutgoingMessage>,
    capacity: usize,
    policy: BackpressurePolicy,
    dropped: u64,
    sequence_counter: u64,
    closed: bool,
}

impl StreamQueue {
    pub fn new(capacity: usize, policy: BackpressurePolicy) -> Self {
        Self {
            queue: VecDeque::with_capacity(capacity.max(1)),
            capacity,
            policy,
            dropped: 0,
            sequence_counter: 0,
            closed: false,
        }
    }

    /// Result of a push attempt.
    pub fn push(&mut self, bytes: Vec<u8>) -> PushResult {
        if self.closed {
            return PushResult::Closed;
        }
        if self.queue.len() >= self.capacity {
            match self.policy {
                BackpressurePolicy::DropOldest => {
                    if self.queue.pop_front().is_some() {
                        self.dropped += 1;
                    }
                }
                BackpressurePolicy::CloseSlow => {
                    self.closed = true;
                    return PushResult::Closed;
                }
                BackpressurePolicy::Block => return PushResult::Blocked,
            }
        }
        self.sequence_counter += 1;
        self.queue.push_back(OutgoingMessage {
            bytes,
            sequence: self.sequence_counter,
        });
        PushResult::Accepted
    }

    pub fn pop(&mut self) -> Option<OutgoingMessage> {
        self.queue.pop_front()
    }

    pub fn len(&self) -> usize {
        self.queue.len()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    pub fn is_closed(&self) -> bool {
        self.closed
    }

    pub fn close(&mut self) {
        self.closed = true;
    }
}

/// Outcome of StreamQueue::push.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PushResult {
    Accepted,
    Blocked,
    Closed,
}

/// Encodes an in-memory payload for the chosen transport and message format.
pub fn encode_outbound(
    transport: &WsOrSse,
    format: EndpointMessageFormat,
    payload: &[u8],
) -> Vec<u8> {
    match transport {
        WsOrSse::WebSocket => match format {
            EndpointMessageFormat::Json | EndpointMessageFormat::JsonLines => {
                encode_frame(WsOpcode::Text, payload, true)
            }
            EndpointMessageFormat::Protobuf => encode_frame(WsOpcode::Binary, payload, true),
        },
        WsOrSse::Sse => {
            let data = std::str::from_utf8(payload).unwrap_or("");
            sse::encode_event(Some("batch"), data, None)
        }
    }
}

/// Heartbeat frame for keepalive. For SSE, a comment line; for WebSocket, a
/// ping frame with an empty payload.
pub fn encode_heartbeat(transport: &WsOrSse) -> Vec<u8> {
    match transport {
        WsOrSse::WebSocket => encode_frame(WsOpcode::Ping, &[], true),
        WsOrSse::Sse => sse::encode_comment("ping"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drop_oldest_replaces_head() {
        let mut q = StreamQueue::new(2, BackpressurePolicy::DropOldest);
        q.push(b"a".to_vec());
        q.push(b"b".to_vec());
        let r = q.push(b"c".to_vec());
        assert_eq!(r, PushResult::Accepted);
        assert_eq!(q.len(), 2);
        assert_eq!(q.dropped(), 1);
        assert_eq!(q.pop().unwrap().bytes, b"b");
        assert_eq!(q.pop().unwrap().bytes, b"c");
    }

    #[test]
    fn close_slow_policy_closes_queue() {
        let mut q = StreamQueue::new(1, BackpressurePolicy::CloseSlow);
        assert_eq!(q.push(b"a".to_vec()), PushResult::Accepted);
        assert_eq!(q.push(b"b".to_vec()), PushResult::Closed);
        assert!(q.is_closed());
    }

    #[test]
    fn block_policy_signals_caller() {
        let mut q = StreamQueue::new(1, BackpressurePolicy::Block);
        assert_eq!(q.push(b"a".to_vec()), PushResult::Accepted);
        assert_eq!(q.push(b"b".to_vec()), PushResult::Blocked);
    }

    #[test]
    fn encode_ws_text() {
        let out = encode_outbound(
            &WsOrSse::WebSocket,
            EndpointMessageFormat::Json,
            b"{\"a\":1}",
        );
        assert_eq!(out[0], 0x81); // fin+text
    }

    #[test]
    fn encode_sse_frame() {
        let out = encode_outbound(&WsOrSse::Sse, EndpointMessageFormat::Json, b"data");
        let s = String::from_utf8(out).unwrap();
        assert!(s.contains("event: batch"));
        assert!(s.contains("data: data"));
    }

    #[test]
    fn heartbeat_differs_by_transport() {
        let ws = encode_heartbeat(&WsOrSse::WebSocket);
        assert_eq!(ws[0] & 0x0F, 0x9);
        let sse = encode_heartbeat(&WsOrSse::Sse);
        assert!(sse.starts_with(b": ping"));
    }
}

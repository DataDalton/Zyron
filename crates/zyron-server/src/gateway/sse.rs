// -----------------------------------------------------------------------------
// Server-Sent Events emitter.
//
// Formats outgoing messages per the WHATWG EventStream spec: each event ends
// with a blank line, multiple `data:` lines split on \n, and optional `id` /
// `event` fields. Keepalive comments ensure proxies do not idle out the
// connection.
// -----------------------------------------------------------------------------

/// Encodes an SSE event as its on-wire byte sequence.
pub fn encode_event(event: Option<&str>, data: &str, id: Option<&str>) -> Vec<u8> {
    let mut out = String::with_capacity(data.len() + 32);
    if let Some(i) = id {
        for line in i.split('\n') {
            out.push_str("id: ");
            out.push_str(line);
            out.push('\n');
        }
    }
    if let Some(e) = event {
        for line in e.split('\n') {
            out.push_str("event: ");
            out.push_str(line);
            out.push('\n');
        }
    }
    for line in data.split('\n') {
        out.push_str("data: ");
        out.push_str(line);
        out.push('\n');
    }
    out.push('\n');
    out.into_bytes()
}

/// Encodes a keepalive comment. Clients discard lines starting with `:`.
pub fn encode_comment(msg: &str) -> Vec<u8> {
    let mut out = String::with_capacity(msg.len() + 8);
    for line in msg.split('\n') {
        out.push_str(": ");
        out.push_str(line);
        out.push('\n');
    }
    out.push('\n');
    out.into_bytes()
}

/// HTTP response headers for an SSE stream, serialized as a single block.
pub fn sse_response_headers() -> Vec<u8> {
    b"HTTP/1.1 200 OK\r\n\
      Content-Type: text/event-stream\r\n\
      Cache-Control: no-cache\r\n\
      Connection: keep-alive\r\n\r\n"
        .to_vec()
}

/// Stream state held between writes. A thin wrapper so caller code stays
/// symmetric with WebSocketConnection.
pub struct SseStream {
    last_event_id: Option<String>,
}

impl SseStream {
    pub fn new() -> Self {
        Self {
            last_event_id: None,
        }
    }

    pub fn next_event_frame(
        &mut self,
        event: Option<&str>,
        data: &str,
        id: Option<&str>,
    ) -> Vec<u8> {
        if let Some(i) = id {
            self.last_event_id = Some(i.to_string());
        }
        encode_event(event, data, id)
    }

    pub fn keepalive_frame(&self) -> Vec<u8> {
        encode_comment("ping")
    }

    pub fn last_event_id(&self) -> Option<&str> {
        self.last_event_id.as_deref()
    }
}

impl Default for SseStream {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_event() {
        let raw = encode_event(Some("update"), "hello", Some("1"));
        let s = String::from_utf8(raw).unwrap();
        assert_eq!(s, "id: 1\nevent: update\ndata: hello\n\n");
    }

    #[test]
    fn multiline_data_splits_on_newline() {
        let raw = encode_event(None, "line1\nline2", None);
        let s = String::from_utf8(raw).unwrap();
        assert_eq!(s, "data: line1\ndata: line2\n\n");
    }

    #[test]
    fn comment_is_prefixed_with_colon() {
        let raw = encode_comment("ping");
        let s = String::from_utf8(raw).unwrap();
        assert_eq!(s, ": ping\n\n");
    }

    #[test]
    fn stream_remembers_last_id() {
        let mut s = SseStream::new();
        let _ = s.next_event_frame(None, "payload", Some("abc"));
        assert_eq!(s.last_event_id(), Some("abc"));
    }

    #[test]
    fn headers_include_event_stream_ct() {
        let raw = sse_response_headers();
        let s = String::from_utf8(raw).unwrap();
        assert!(s.contains("Content-Type: text/event-stream"));
        assert!(s.contains("Cache-Control: no-cache"));
    }
}

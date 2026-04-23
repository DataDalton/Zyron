// -----------------------------------------------------------------------------
// RFC 6455 WebSocket framing.
//
// Provides upgrade-handshake helpers and frame encode/decode. The protocol
// layer is synchronous over byte slices and owns neither the socket nor the
// async reader. Caller glues framing to a tokio AsyncRead/AsyncWrite pair.
// -----------------------------------------------------------------------------

use super::request::HttpRequest;

/// Magic value appended to Sec-WebSocket-Key before SHA-1 hashing, per
/// RFC 6455 §1.3.
pub const WS_GUID: &str = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

/// Opcode values defined by RFC 6455 §5.2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WsOpcode {
    Continuation = 0x0,
    Text = 0x1,
    Binary = 0x2,
    Close = 0x8,
    Ping = 0x9,
    Pong = 0xA,
}

impl WsOpcode {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0x0 => Some(WsOpcode::Continuation),
            0x1 => Some(WsOpcode::Text),
            0x2 => Some(WsOpcode::Binary),
            0x8 => Some(WsOpcode::Close),
            0x9 => Some(WsOpcode::Ping),
            0xA => Some(WsOpcode::Pong),
            _ => None,
        }
    }
}

/// Decoded WebSocket message from the client side.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WsMessage {
    Text(String),
    Binary(Vec<u8>),
    Ping(Vec<u8>),
    Pong(Vec<u8>),
    Close { code: u16, reason: String },
}

/// Error returned from frame decoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WsError {
    Incomplete,
    InvalidOpcode(u8),
    InvalidMask,
    OversizedPayload,
    InvalidUtf8,
}

/// Checks the incoming HTTP request for a valid WebSocket upgrade and returns
/// the Sec-WebSocket-Accept value required in the 101 response.
pub fn upgrade_accept(req: &HttpRequest) -> Option<String> {
    let upgrade = req.header("upgrade")?;
    if !upgrade.eq_ignore_ascii_case("websocket") {
        return None;
    }
    let conn = req.header("connection")?;
    if !conn.to_ascii_lowercase().contains("upgrade") {
        return None;
    }
    let version = req.header("sec-websocket-version")?;
    if version.trim() != "13" {
        return None;
    }
    let key = req.header("sec-websocket-key")?;
    Some(compute_accept(key))
}

/// Builds the full HTTP/1.1 101 Switching Protocols response for a handshake.
pub fn upgrade_response(accept: &str) -> Vec<u8> {
    let body = format!(
        "HTTP/1.1 101 Switching Protocols\r\n\
         Upgrade: websocket\r\n\
         Connection: Upgrade\r\n\
         Sec-WebSocket-Accept: {}\r\n\r\n",
        accept
    );
    body.into_bytes()
}

/// Computes the Sec-WebSocket-Accept value from the client key.
pub fn compute_accept(key: &str) -> String {
    let mut h = sha1_zyron::Sha1::new();
    h.update(key.as_bytes());
    h.update(WS_GUID.as_bytes());
    let digest = h.finalize();
    base64_encode(&digest)
}

/// Encodes a server-to-client frame. Server frames MUST NOT be masked
/// (RFC 6455 §5.3).
pub fn encode_frame(opcode: WsOpcode, payload: &[u8], fin: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity(payload.len() + 10);
    let first = (if fin { 0x80 } else { 0x00 }) | (opcode as u8 & 0x0F);
    out.push(first);
    let len = payload.len();
    if len < 126 {
        out.push(len as u8);
    } else if len <= u16::MAX as usize {
        out.push(126);
        out.extend_from_slice(&(len as u16).to_be_bytes());
    } else {
        out.push(127);
        out.extend_from_slice(&(len as u64).to_be_bytes());
    }
    out.extend_from_slice(payload);
    out
}

/// Decodes a single client-to-server frame from `buf`. Returns the number of
/// bytes consumed along with the message when a frame fits entirely in the
/// buffer. When the buffer is short, returns WsError::Incomplete.
pub fn decode_frame(buf: &[u8]) -> Result<(usize, DecodedFrame), WsError> {
    if buf.len() < 2 {
        return Err(WsError::Incomplete);
    }
    let first = buf[0];
    let second = buf[1];
    let fin = first & 0x80 != 0;
    let opcode_raw = first & 0x0F;
    let opcode = WsOpcode::from_u8(opcode_raw).ok_or(WsError::InvalidOpcode(opcode_raw))?;
    let masked = second & 0x80 != 0;
    let mut len = (second & 0x7F) as usize;
    let mut cursor = 2usize;
    if len == 126 {
        if buf.len() < cursor + 2 {
            return Err(WsError::Incomplete);
        }
        len = u16::from_be_bytes([buf[cursor], buf[cursor + 1]]) as usize;
        cursor += 2;
    } else if len == 127 {
        if buf.len() < cursor + 8 {
            return Err(WsError::Incomplete);
        }
        let raw = u64::from_be_bytes([
            buf[cursor],
            buf[cursor + 1],
            buf[cursor + 2],
            buf[cursor + 3],
            buf[cursor + 4],
            buf[cursor + 5],
            buf[cursor + 6],
            buf[cursor + 7],
        ]);
        if raw > (u32::MAX as u64) {
            return Err(WsError::OversizedPayload);
        }
        len = raw as usize;
        cursor += 8;
    }
    let mask_key = if masked {
        if buf.len() < cursor + 4 {
            return Err(WsError::Incomplete);
        }
        let mk = [
            buf[cursor],
            buf[cursor + 1],
            buf[cursor + 2],
            buf[cursor + 3],
        ];
        cursor += 4;
        Some(mk)
    } else {
        None
    };
    if buf.len() < cursor + len {
        return Err(WsError::Incomplete);
    }
    let mut payload = buf[cursor..cursor + len].to_vec();
    if let Some(mk) = mask_key {
        for (i, b) in payload.iter_mut().enumerate() {
            *b ^= mk[i % 4];
        }
    } else {
        // RFC 6455 requires client frames to be masked. Server-side readers
        // must reject unmasked frames from the client, but the decoder is
        // symmetric so the caller decides based on direction.
    }
    cursor += len;
    Ok((
        cursor,
        DecodedFrame {
            fin,
            opcode,
            payload,
        },
    ))
}

/// Raw frame before opcode dispatch.
#[derive(Debug, Clone)]
pub struct DecodedFrame {
    pub fin: bool,
    pub opcode: WsOpcode,
    pub payload: Vec<u8>,
}

/// WebSocket connection primitive. Owns the fragment reassembly buffer.
pub struct WebSocketConnection {
    pending_opcode: Option<WsOpcode>,
    pending_payload: Vec<u8>,
    closed: bool,
}

impl WebSocketConnection {
    pub fn new() -> Self {
        Self {
            pending_opcode: None,
            pending_payload: Vec::new(),
            closed: false,
        }
    }

    pub fn is_closed(&self) -> bool {
        self.closed
    }

    /// Consumes any number of completed frames from `buf`. Returns each
    /// decoded application message and the number of bytes consumed.
    pub fn absorb(&mut self, buf: &[u8]) -> (usize, Vec<WsMessage>) {
        let mut consumed = 0usize;
        let mut out = Vec::new();
        loop {
            let slice = &buf[consumed..];
            match decode_frame(slice) {
                Err(WsError::Incomplete) => return (consumed, out),
                Err(_) => return (consumed, out),
                Ok((n, frame)) => {
                    consumed += n;
                    match frame.opcode {
                        WsOpcode::Continuation => {
                            self.pending_payload.extend_from_slice(&frame.payload);
                            if frame.fin {
                                let op = self.pending_opcode.take().unwrap_or(WsOpcode::Binary);
                                let payload = std::mem::take(&mut self.pending_payload);
                                if let Some(msg) = assemble(op, payload) {
                                    out.push(msg);
                                }
                            }
                        }
                        WsOpcode::Text | WsOpcode::Binary => {
                            if frame.fin {
                                if let Some(msg) = assemble(frame.opcode, frame.payload) {
                                    out.push(msg);
                                }
                            } else {
                                self.pending_opcode = Some(frame.opcode);
                                self.pending_payload = frame.payload;
                            }
                        }
                        WsOpcode::Ping => out.push(WsMessage::Ping(frame.payload)),
                        WsOpcode::Pong => out.push(WsMessage::Pong(frame.payload)),
                        WsOpcode::Close => {
                            let (code, reason) = parse_close(&frame.payload);
                            self.closed = true;
                            out.push(WsMessage::Close { code, reason });
                            return (consumed, out);
                        }
                    }
                }
            }
        }
    }
}

impl Default for WebSocketConnection {
    fn default() -> Self {
        Self::new()
    }
}

fn assemble(op: WsOpcode, payload: Vec<u8>) -> Option<WsMessage> {
    match op {
        WsOpcode::Text => String::from_utf8(payload).ok().map(WsMessage::Text),
        WsOpcode::Binary => Some(WsMessage::Binary(payload)),
        _ => None,
    }
}

fn parse_close(payload: &[u8]) -> (u16, String) {
    if payload.len() < 2 {
        return (1000, String::new());
    }
    let code = u16::from_be_bytes([payload[0], payload[1]]);
    let reason = String::from_utf8_lossy(&payload[2..]).to_string();
    (code, reason)
}

fn base64_encode(input: &[u8]) -> String {
    const TABLE: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((input.len() + 2) / 3 * 4);
    let mut i = 0;
    while i + 3 <= input.len() {
        let n = ((input[i] as u32) << 16) | ((input[i + 1] as u32) << 8) | (input[i + 2] as u32);
        out.push(TABLE[((n >> 18) & 0x3F) as usize] as char);
        out.push(TABLE[((n >> 12) & 0x3F) as usize] as char);
        out.push(TABLE[((n >> 6) & 0x3F) as usize] as char);
        out.push(TABLE[(n & 0x3F) as usize] as char);
        i += 3;
    }
    let rem = input.len() - i;
    if rem == 1 {
        let n = (input[i] as u32) << 16;
        out.push(TABLE[((n >> 18) & 0x3F) as usize] as char);
        out.push(TABLE[((n >> 12) & 0x3F) as usize] as char);
        out.push_str("==");
    } else if rem == 2 {
        let n = ((input[i] as u32) << 16) | ((input[i + 1] as u32) << 8);
        out.push(TABLE[((n >> 18) & 0x3F) as usize] as char);
        out.push(TABLE[((n >> 12) & 0x3F) as usize] as char);
        out.push(TABLE[((n >> 6) & 0x3F) as usize] as char);
        out.push('=');
    }
    out
}

// ---------------------------------------------------------------------------
// SHA-1 implementation.
//
// RFC 6455 requires SHA-1 for the accept hash. The auth crate already uses
// SHA-1 transitively (see workspace deps). This small implementation keeps
// the gateway from taking a direct sha1-crate dependency.
// ---------------------------------------------------------------------------

mod sha1_zyron {
    pub struct Sha1 {
        h: [u32; 5],
        buf: Vec<u8>,
        len: u64,
    }

    impl Sha1 {
        pub fn new() -> Self {
            Self {
                h: [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0],
                buf: Vec::with_capacity(64),
                len: 0,
            }
        }

        pub fn update(&mut self, data: &[u8]) {
            self.len += data.len() as u64;
            self.buf.extend_from_slice(data);
            while self.buf.len() >= 64 {
                let block: [u8; 64] = {
                    let mut b = [0u8; 64];
                    b.copy_from_slice(&self.buf[..64]);
                    b
                };
                self.process(&block);
                self.buf.drain(..64);
            }
        }

        pub fn finalize(mut self) -> [u8; 20] {
            let bit_len = self.len.wrapping_mul(8);
            self.buf.push(0x80);
            while self.buf.len() % 64 != 56 {
                self.buf.push(0);
            }
            self.buf.extend_from_slice(&bit_len.to_be_bytes());
            while !self.buf.is_empty() {
                let mut block = [0u8; 64];
                block.copy_from_slice(&self.buf[..64]);
                self.process(&block);
                self.buf.drain(..64);
            }
            let mut out = [0u8; 20];
            for i in 0..5 {
                out[i * 4..i * 4 + 4].copy_from_slice(&self.h[i].to_be_bytes());
            }
            out
        }

        fn process(&mut self, block: &[u8; 64]) {
            let mut w = [0u32; 80];
            for i in 0..16 {
                w[i] = u32::from_be_bytes([
                    block[i * 4],
                    block[i * 4 + 1],
                    block[i * 4 + 2],
                    block[i * 4 + 3],
                ]);
            }
            for i in 16..80 {
                w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
            }
            let mut a = self.h[0];
            let mut b = self.h[1];
            let mut c = self.h[2];
            let mut d = self.h[3];
            let mut e = self.h[4];
            for i in 0..80 {
                let (f, k) = if i < 20 {
                    ((b & c) | ((!b) & d), 0x5A827999)
                } else if i < 40 {
                    (b ^ c ^ d, 0x6ED9EBA1)
                } else if i < 60 {
                    ((b & c) | (b & d) | (c & d), 0x8F1BBCDC)
                } else {
                    (b ^ c ^ d, 0xCA62C1D6)
                };
                let temp = a
                    .rotate_left(5)
                    .wrapping_add(f)
                    .wrapping_add(e)
                    .wrapping_add(k)
                    .wrapping_add(w[i]);
                e = d;
                d = c;
                c = b.rotate_left(30);
                b = a;
                a = temp;
            }
            self.h[0] = self.h[0].wrapping_add(a);
            self.h[1] = self.h[1].wrapping_add(b);
            self.h[2] = self.h[2].wrapping_add(c);
            self.h[3] = self.h[3].wrapping_add(d);
            self.h[4] = self.h[4].wrapping_add(e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_accept_vector() {
        // RFC 6455 §1.3 example.
        let got = compute_accept("dGhlIHNhbXBsZSBub25jZQ==");
        assert_eq!(got, "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=");
    }

    #[test]
    fn handshake_requires_all_headers() {
        use crate::gateway::router::HttpMethod;
        use std::collections::HashMap;
        let mut headers = HashMap::new();
        headers.insert("upgrade".into(), "websocket".into());
        headers.insert("connection".into(), "Upgrade".into());
        headers.insert("sec-websocket-version".into(), "13".into());
        headers.insert(
            "sec-websocket-key".into(),
            "dGhlIHNhbXBsZSBub25jZQ==".into(),
        );
        let req = HttpRequest {
            method: HttpMethod::Get,
            path: "/ws".into(),
            query_string: String::new(),
            headers,
            body: Vec::new(),
            peer_addr: None,
            tls_info: None,
        };
        assert!(upgrade_accept(&req).is_some());
    }

    #[test]
    fn encode_small_text_frame() {
        let frame = encode_frame(WsOpcode::Text, b"hi", true);
        assert_eq!(frame[0], 0x81);
        assert_eq!(frame[1], 0x02);
        assert_eq!(&frame[2..], b"hi");
    }

    #[test]
    fn encode_large_payload_uses_extended_length() {
        let payload = vec![0u8; 200];
        let frame = encode_frame(WsOpcode::Binary, &payload, true);
        assert_eq!(frame[0], 0x82);
        assert_eq!(frame[1], 126);
        assert_eq!(u16::from_be_bytes([frame[2], frame[3]]), 200);
    }

    #[test]
    fn decode_masked_text() {
        // Build a client frame manually: FIN=1, opcode=text, masked, len=5, mask=0x01020304
        let mask = [0x01, 0x02, 0x03, 0x04];
        let msg = b"hello";
        let masked: Vec<u8> = msg
            .iter()
            .enumerate()
            .map(|(i, b)| b ^ mask[i % 4])
            .collect();
        let mut frame = vec![0x81, 0x85];
        frame.extend_from_slice(&mask);
        frame.extend_from_slice(&masked);
        let (n, d) = decode_frame(&frame).unwrap();
        assert_eq!(n, frame.len());
        assert_eq!(d.opcode, WsOpcode::Text);
        assert_eq!(d.payload, b"hello");
    }

    #[test]
    fn incomplete_frame_returns_incomplete() {
        assert!(matches!(decode_frame(&[0x81]), Err(WsError::Incomplete)));
    }

    #[test]
    fn absorb_fragmented_text() {
        let mut conn = WebSocketConnection::new();
        let mask = [0, 0, 0, 0];
        // Fragment 1: fin=0, opcode=Text, masked, "he"
        let mut f1 = vec![0x01, 0x82];
        f1.extend_from_slice(&mask);
        f1.extend_from_slice(b"he");
        // Fragment 2: fin=1, opcode=Continuation, masked, "llo"
        let mut f2 = vec![0x80, 0x83];
        f2.extend_from_slice(&mask);
        f2.extend_from_slice(b"llo");
        let (_, msgs1) = conn.absorb(&f1);
        assert!(msgs1.is_empty());
        let (_, msgs2) = conn.absorb(&f2);
        assert_eq!(msgs2, vec![WsMessage::Text("hello".to_string())]);
    }

    #[test]
    fn close_frame_closes_connection() {
        let mut conn = WebSocketConnection::new();
        let payload = [0x03, 0xE8]; // code 1000
        let mask = [0, 0, 0, 0];
        let mut frame = vec![0x88, 0x82];
        frame.extend_from_slice(&mask);
        frame.extend_from_slice(&payload);
        let (_, msgs) = conn.absorb(&frame);
        assert_eq!(msgs.len(), 1);
        assert!(conn.is_closed());
    }
}

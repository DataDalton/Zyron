//! PostgreSQL wire protocol codec for message framing.
//!
//! Handles the startup phase asymmetry where the first client message
//! has no type byte, then switches to normal framing for all subsequent
//! messages.

use bytes::{Buf, BytesMut};

use crate::messages::{FrontendMessage, ProtocolError};

/// Maximum single message size: 1 GB (matches PostgreSQL).
const MAX_MESSAGE_SIZE: usize = 1_073_741_824;

/// Codec for PostgreSQL wire protocol v3 message framing.
///
/// During the startup phase, the client sends messages without a type byte
/// (just length + payload). After startup completes, all messages follow
/// the normal format: type byte (1) + length (4) + payload.
pub struct PostgresCodec {
    /// True during the initial startup before the first typed message.
    startup_phase: bool,
}

impl PostgresCodec {
    pub fn new() -> Self {
        Self {
            startup_phase: true,
        }
    }

    /// Switches from startup mode to normal message framing.
    /// Called by the connection handler after processing the startup message.
    pub fn set_normal_mode(&mut self) {
        self.startup_phase = false;
    }

    /// Returns true if the codec is still in the startup phase.
    pub fn is_startup_phase(&self) -> bool {
        self.startup_phase
    }

    /// Encodes a backend message into the destination buffer.
    pub fn encode(
        &mut self,
        item: crate::messages::BackendMessage,
        dst: &mut BytesMut,
    ) -> Result<(), ProtocolError> {
        item.encode(dst);
        Ok(())
    }

    /// Tries to decode one complete message from the buffer.
    /// Returns Ok(None) if not enough data is available yet.
    pub fn decode(&mut self, src: &mut BytesMut) -> Result<Option<FrontendMessage>, ProtocolError> {
        if self.startup_phase {
            // Startup messages have no type byte.
            // Format: 4-byte length (includes itself) + payload.
            if src.len() < 4 {
                return Ok(None);
            }

            let len = i32::from_be_bytes([src[0], src[1], src[2], src[3]]) as usize;
            if len < 4 {
                return Err(ProtocolError::Malformed(
                    "Startup message length less than 4".into(),
                ));
            }
            if len > MAX_MESSAGE_SIZE {
                return Err(ProtocolError::MessageTooLarge {
                    size: len,
                    max: MAX_MESSAGE_SIZE,
                });
            }
            if src.len() < len {
                return Ok(None);
            }

            let mut frame = src.split_to(len);
            frame.advance(4); // skip the length field
            FrontendMessage::decode_startup(&mut frame).map(Some)
        } else {
            // Normal messages: type byte (1) + length (4, includes itself) + payload.
            if src.len() < 5 {
                return Ok(None);
            }

            let msg_type = src[0];
            let len = i32::from_be_bytes([src[1], src[2], src[3], src[4]]) as usize;
            if len < 4 {
                return Err(ProtocolError::Malformed(
                    "Message length less than 4".into(),
                ));
            }
            if len > MAX_MESSAGE_SIZE {
                return Err(ProtocolError::MessageTooLarge {
                    size: len,
                    max: MAX_MESSAGE_SIZE,
                });
            }

            let total = 1 + len; // type byte + length-inclusive payload
            if src.len() < total {
                return Ok(None);
            }

            src.advance(5); // skip type byte + length
            let payload_len = len - 4;
            let mut payload = src.split_to(payload_len);
            FrontendMessage::decode(msg_type, &mut payload).map(Some)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::BufMut;

    #[test]
    fn test_decode_startup_message() {
        let mut codec = PostgresCodec::new();
        let mut buf = BytesMut::new();

        // Build a startup message: length + version 3.0 + params + terminator
        let mut payload = BytesMut::new();
        payload.put_i32(196608); // version 3.0
        payload.put_slice(b"user\0test\0");
        payload.put_u8(0); // terminator

        let total_len = 4 + payload.len(); // length field + payload
        buf.put_i32(total_len as i32);
        buf.put_slice(&payload);

        let msg = codec.decode(&mut buf).unwrap().unwrap();
        match msg {
            FrontendMessage::Startup(startup) => {
                assert_eq!(startup.params.get("user").unwrap(), "test");
            }
            _ => panic!("Expected Startup"),
        }
    }

    #[test]
    fn test_decode_partial_startup() {
        let mut codec = PostgresCodec::new();
        let mut buf = BytesMut::new();

        // Only 2 bytes available, need at least 4
        buf.put_u8(0);
        buf.put_u8(0);

        let result = codec.decode(&mut buf).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_decode_normal_query() {
        let mut codec = PostgresCodec::new();
        codec.set_normal_mode();

        let mut buf = BytesMut::new();
        let sql = b"SELECT 1\0";
        let len = 4 + sql.len(); // length includes itself

        buf.put_u8(b'Q');
        buf.put_i32(len as i32);
        buf.put_slice(sql);

        let msg = codec.decode(&mut buf).unwrap().unwrap();
        match msg {
            FrontendMessage::Query { sql } => assert_eq!(sql, "SELECT 1"),
            _ => panic!("Expected Query"),
        }
    }

    #[test]
    fn test_decode_partial_normal() {
        let mut codec = PostgresCodec::new();
        codec.set_normal_mode();

        let mut buf = BytesMut::new();
        // Only type byte + partial length
        buf.put_u8(b'Q');
        buf.put_u8(0);

        let result = codec.decode(&mut buf).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_decode_incomplete_payload() {
        let mut codec = PostgresCodec::new();
        codec.set_normal_mode();

        let mut buf = BytesMut::new();
        buf.put_u8(b'Q');
        buf.put_i32(20); // claims 20 bytes but we provide less
        buf.put_slice(b"SEL"); // only 3 payload bytes

        let result = codec.decode(&mut buf).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_decode_message_too_large() {
        let mut codec = PostgresCodec::new();
        codec.set_normal_mode();

        let mut buf = BytesMut::new();
        buf.put_u8(b'Q');
        buf.put_i32(i32::MAX); // way too large

        let result = codec.decode(&mut buf);
        assert!(matches!(result, Err(ProtocolError::MessageTooLarge { .. })));
    }

    #[test]
    fn test_encode_backend_message() {
        use crate::messages::backend::{BackendMessage, TransactionState};

        let mut buf = BytesMut::new();
        BackendMessage::ReadyForQuery(TransactionState::Idle).encode(&mut buf);

        assert_eq!(buf[0], b'Z');
        assert_eq!(buf.len(), 6); // type(1) + length(4) + state(1)
    }

    #[test]
    fn test_decode_terminate() {
        let mut codec = PostgresCodec::new();
        codec.set_normal_mode();

        let mut buf = BytesMut::new();
        buf.put_u8(b'X');
        buf.put_i32(4); // length includes itself

        let msg = codec.decode(&mut buf).unwrap().unwrap();
        assert!(matches!(msg, FrontendMessage::Terminate));
    }

    #[test]
    fn test_decode_sync() {
        let mut codec = PostgresCodec::new();
        codec.set_normal_mode();

        let mut buf = BytesMut::new();
        buf.put_u8(b'S');
        buf.put_i32(4);

        let msg = codec.decode(&mut buf).unwrap().unwrap();
        assert!(matches!(msg, FrontendMessage::Sync));
    }

    #[test]
    fn test_decode_ssl_request() {
        let mut codec = PostgresCodec::new();
        let mut buf = BytesMut::new();

        buf.put_i32(8); // length
        buf.put_i32(80877103); // SSL request code

        let msg = codec.decode(&mut buf).unwrap().unwrap();
        assert!(matches!(msg, FrontendMessage::SslRequest));
    }

    #[test]
    fn test_startup_length_too_small() {
        let mut codec = PostgresCodec::new();
        let mut buf = BytesMut::new();

        buf.put_i32(2); // length less than 4

        let result = codec.decode(&mut buf);
        assert!(result.is_err());
    }

    #[test]
    fn test_multiple_messages() {
        let mut codec = PostgresCodec::new();
        codec.set_normal_mode();

        let mut buf = BytesMut::new();

        // First message: Query
        let sql = b"SELECT 1\0";
        buf.put_u8(b'Q');
        buf.put_i32(4 + sql.len() as i32);
        buf.put_slice(sql);

        // Second message: Sync
        buf.put_u8(b'S');
        buf.put_i32(4);

        let msg1 = codec.decode(&mut buf).unwrap().unwrap();
        assert!(matches!(msg1, FrontendMessage::Query { .. }));

        let msg2 = codec.decode(&mut buf).unwrap().unwrap();
        assert!(matches!(msg2, FrontendMessage::Sync));

        // Buffer should be empty
        assert!(buf.is_empty());
    }
}

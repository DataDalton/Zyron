//! Frontend (client-to-server) message definitions and parsing.
//!
//! Handles all messages a PostgreSQL client can send, including the
//! startup handshake (no type byte), SSL/cancel requests, authentication
//! responses, simple query, extended query protocol, and COPY data.

use bytes::{Buf, BytesMut};
use std::collections::HashMap;

use super::ProtocolError;

// SSL and cancel request magic numbers in the protocol version field.
const SSL_REQUEST_CODE: i32 = 80877103;
const CANCEL_REQUEST_CODE: i32 = 80877102;

/// Messages sent from client to server.
#[derive(Debug, Clone)]
pub enum FrontendMessage {
    /// Initial connection with protocol version and parameters (user, database).
    Startup(StartupMessage),
    /// SSL negotiation request.
    SslRequest,
    /// Cancel a running query on another connection.
    CancelRequest { process_id: i32, secret_key: i32 },
    /// Password response during authentication.
    Password(PasswordMessage),
    /// Simple query protocol: one or more SQL statements as a string.
    Query { sql: String },
    /// Extended query: parse a prepared statement.
    Parse {
        name: String,
        query: String,
        param_types: Vec<i32>,
    },
    /// Extended query: bind parameters to a prepared statement, creating a portal.
    Bind {
        portal: String,
        statement: String,
        param_formats: Vec<i16>,
        param_values: Vec<Option<Vec<u8>>>,
        result_formats: Vec<i16>,
    },
    /// Extended query: execute a bound portal.
    Execute { portal: String, max_rows: i32 },
    /// Extended query: request metadata for a prepared statement or portal.
    Describe {
        target: DescribeTarget,
        name: String,
    },
    /// Extended query: close a prepared statement or portal.
    Close {
        target: DescribeTarget,
        name: String,
    },
    /// Extended query: synchronization point, triggers ReadyForQuery.
    Sync,
    /// Extended query: flush output buffer without sync.
    Flush,
    /// Client termination.
    Terminate,
    /// COPY data chunk from client.
    CopyData(Vec<u8>),
    /// COPY completion from client.
    CopyDone,
    /// COPY failure from client.
    CopyFail { message: String },
}

/// Initial startup message with protocol version and connection parameters.
#[derive(Debug, Clone)]
pub struct StartupMessage {
    pub protocol_version: i32,
    pub params: HashMap<String, String>,
}

/// Password response variants during authentication.
#[derive(Debug, Clone)]
pub enum PasswordMessage {
    /// Cleartext password.
    Cleartext(String),
    /// MD5-hashed password (35 bytes: "md5" + 32 hex chars).
    Md5(Vec<u8>),
    /// SASL initial response with mechanism name and optional data.
    SaslInitial { mechanism: String, data: Vec<u8> },
    /// SASL continuation response.
    SaslResponse(Vec<u8>),
}

/// Target type for Describe and Close messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DescribeTarget {
    Statement,
    Portal,
}

impl FrontendMessage {
    /// Parses the initial startup packet. Called before normal message framing
    /// because the startup message has no type byte, only a 4-byte protocol
    /// version followed by null-terminated key-value pairs.
    pub fn decode_startup(payload: &mut BytesMut) -> Result<Self, ProtocolError> {
        if payload.remaining() < 4 {
            return Err(ProtocolError::Malformed("Startup message too short".into()));
        }

        let version = payload.get_i32();

        if version == SSL_REQUEST_CODE {
            return Ok(FrontendMessage::SslRequest);
        }

        if version == CANCEL_REQUEST_CODE {
            if payload.remaining() < 8 {
                return Err(ProtocolError::Malformed("Cancel request too short".into()));
            }
            let process_id = payload.get_i32();
            let secret_key = payload.get_i32();
            return Ok(FrontendMessage::CancelRequest {
                process_id,
                secret_key,
            });
        }

        // Protocol version 3.0 = 0x00030000 = 196608
        let major = version >> 16;
        let minor = version & 0xFFFF;
        if major != 3 {
            return Err(ProtocolError::UnsupportedProtocol(version));
        }

        let mut params = HashMap::new();
        loop {
            let key = read_cstring(payload)?;
            if key.is_empty() {
                break;
            }
            let value = read_cstring(payload)?;
            params.insert(key, value);
        }

        Ok(FrontendMessage::Startup(StartupMessage {
            protocol_version: (major << 16) | minor,
            params,
        }))
    }

    /// Parses a normal frontend message from the type byte and payload.
    /// The type byte and length prefix have already been consumed by the codec.
    pub fn decode(msg_type: u8, payload: &mut BytesMut) -> Result<Self, ProtocolError> {
        match msg_type {
            b'Q' => {
                let sql = read_cstring(payload)?;
                Ok(FrontendMessage::Query { sql })
            }

            b'P' => {
                let name = read_cstring(payload)?;
                let query = read_cstring(payload)?;
                let num_params = payload.get_i16() as usize;
                let mut param_types = Vec::with_capacity(num_params);
                for _ in 0..num_params {
                    param_types.push(payload.get_i32());
                }
                Ok(FrontendMessage::Parse {
                    name,
                    query,
                    param_types,
                })
            }

            b'B' => {
                let portal = read_cstring(payload)?;
                let statement = read_cstring(payload)?;

                // Parameter format codes
                let num_formats = payload.get_i16() as usize;
                let mut param_formats = Vec::with_capacity(num_formats);
                for _ in 0..num_formats {
                    param_formats.push(payload.get_i16());
                }

                // Parameter values
                let num_values = payload.get_i16() as usize;
                let mut param_values = Vec::with_capacity(num_values);
                for _ in 0..num_values {
                    let len = payload.get_i32();
                    if len == -1 {
                        param_values.push(None);
                    } else {
                        let len = len as usize;
                        let data = payload.split_to(len).to_vec();
                        param_values.push(Some(data));
                    }
                }

                // Result format codes
                let num_result_formats = payload.get_i16() as usize;
                let mut result_formats = Vec::with_capacity(num_result_formats);
                for _ in 0..num_result_formats {
                    result_formats.push(payload.get_i16());
                }

                Ok(FrontendMessage::Bind {
                    portal,
                    statement,
                    param_formats,
                    param_values,
                    result_formats,
                })
            }

            b'E' => {
                let portal = read_cstring(payload)?;
                let max_rows = payload.get_i32();
                Ok(FrontendMessage::Execute { portal, max_rows })
            }

            b'D' => {
                let target_byte = payload.get_u8();
                let target = match target_byte {
                    b'S' => DescribeTarget::Statement,
                    b'P' => DescribeTarget::Portal,
                    _ => {
                        return Err(ProtocolError::Malformed(format!(
                            "Invalid Describe target: 0x{:02x}",
                            target_byte
                        )));
                    }
                };
                let name = read_cstring(payload)?;
                Ok(FrontendMessage::Describe { target, name })
            }

            b'C' => {
                let target_byte = payload.get_u8();
                let target = match target_byte {
                    b'S' => DescribeTarget::Statement,
                    b'P' => DescribeTarget::Portal,
                    _ => {
                        return Err(ProtocolError::Malformed(format!(
                            "Invalid Close target: 0x{:02x}",
                            target_byte
                        )));
                    }
                };
                let name = read_cstring(payload)?;
                Ok(FrontendMessage::Close { target, name })
            }

            b'S' => Ok(FrontendMessage::Sync),

            b'H' => Ok(FrontendMessage::Flush),

            b'X' => Ok(FrontendMessage::Terminate),

            b'p' => {
                // Password message. Search directly in payload to avoid copying
                // the entire buffer before inspecting it.
                let bytes = payload.as_ref();
                if let Some(null_pos) = bytes.iter().position(|&b| b == 0) {
                    // Check for MD5 hash (starts with "md5", 35 bytes total).
                    if null_pos == 35 && bytes.starts_with(b"md5") {
                        let md5_data = bytes[..null_pos].to_vec();
                        payload.advance(null_pos + 1);
                        Ok(FrontendMessage::Password(PasswordMessage::Md5(md5_data)))
                    } else {
                        let password =
                            String::from_utf8(bytes[..null_pos].to_vec()).map_err(|e| {
                                ProtocolError::Malformed(format!(
                                    "Invalid UTF-8 in password: {}",
                                    e
                                ))
                            })?;
                        payload.advance(null_pos + 1);
                        Ok(FrontendMessage::Password(PasswordMessage::Cleartext(
                            password,
                        )))
                    }
                } else {
                    // No null terminator, treat entire payload as cleartext.
                    let password = String::from_utf8(bytes.to_vec()).map_err(|e| {
                        ProtocolError::Malformed(format!("Invalid UTF-8 in password: {}", e))
                    })?;
                    payload.advance(bytes.len());
                    Ok(FrontendMessage::Password(PasswordMessage::Cleartext(
                        password,
                    )))
                }
            }

            b'd' => {
                let data = payload.to_vec();
                Ok(FrontendMessage::CopyData(data))
            }

            b'c' => Ok(FrontendMessage::CopyDone),

            b'f' => {
                let message = read_cstring(payload)?;
                Ok(FrontendMessage::CopyFail { message })
            }

            _ => Err(ProtocolError::InvalidMessageType(msg_type)),
        }
    }
}

/// Reads a null-terminated string from the buffer.
/// Uses from_utf8 instead of from_utf8_lossy to avoid double allocation
/// and to correctly reject invalid UTF-8 from clients.
fn read_cstring(buf: &mut BytesMut) -> Result<String, ProtocolError> {
    let bytes = buf.as_ref();
    let null_pos = bytes
        .iter()
        .position(|&b| b == 0)
        .ok_or_else(|| ProtocolError::Malformed("Missing null terminator".into()))?;
    let s = String::from_utf8(bytes[..null_pos].to_vec())
        .map_err(|e| ProtocolError::Malformed(format!("Invalid UTF-8: {}", e)))?;
    buf.advance(null_pos + 1);
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_startup() {
        let mut buf = BytesMut::new();
        // Protocol version 3.0
        buf.extend_from_slice(&196608_i32.to_be_bytes());
        // user=test
        buf.extend_from_slice(b"user\0test\0");
        // database=zyron
        buf.extend_from_slice(b"database\0zyron\0");
        // terminator
        buf.extend_from_slice(b"\0");

        let msg = FrontendMessage::decode_startup(&mut buf).unwrap();
        match msg {
            FrontendMessage::Startup(startup) => {
                assert_eq!(startup.protocol_version, 196608);
                assert_eq!(startup.params.get("user").unwrap(), "test");
                assert_eq!(startup.params.get("database").unwrap(), "zyron");
            }
            _ => panic!("Expected Startup message"),
        }
    }

    #[test]
    fn test_decode_ssl_request() {
        let mut buf = BytesMut::new();
        buf.extend_from_slice(&SSL_REQUEST_CODE.to_be_bytes());

        let msg = FrontendMessage::decode_startup(&mut buf).unwrap();
        assert!(matches!(msg, FrontendMessage::SslRequest));
    }

    #[test]
    fn test_decode_cancel_request() {
        let mut buf = BytesMut::new();
        buf.extend_from_slice(&CANCEL_REQUEST_CODE.to_be_bytes());
        buf.extend_from_slice(&42_i32.to_be_bytes());
        buf.extend_from_slice(&99_i32.to_be_bytes());

        let msg = FrontendMessage::decode_startup(&mut buf).unwrap();
        match msg {
            FrontendMessage::CancelRequest {
                process_id,
                secret_key,
            } => {
                assert_eq!(process_id, 42);
                assert_eq!(secret_key, 99);
            }
            _ => panic!("Expected CancelRequest"),
        }
    }

    #[test]
    fn test_decode_query() {
        let mut buf = BytesMut::new();
        buf.extend_from_slice(b"SELECT 1\0");

        let msg = FrontendMessage::decode(b'Q', &mut buf).unwrap();
        match msg {
            FrontendMessage::Query { sql } => assert_eq!(sql, "SELECT 1"),
            _ => panic!("Expected Query message"),
        }
    }

    #[test]
    fn test_decode_parse() {
        let mut buf = BytesMut::new();
        buf.extend_from_slice(b"stmt1\0SELECT $1\0");
        buf.extend_from_slice(&1_i16.to_be_bytes()); // 1 param type
        buf.extend_from_slice(&23_i32.to_be_bytes()); // int4

        let msg = FrontendMessage::decode(b'P', &mut buf).unwrap();
        match msg {
            FrontendMessage::Parse {
                name,
                query,
                param_types,
            } => {
                assert_eq!(name, "stmt1");
                assert_eq!(query, "SELECT $1");
                assert_eq!(param_types, vec![23]);
            }
            _ => panic!("Expected Parse message"),
        }
    }

    #[test]
    fn test_decode_bind() {
        let mut buf = BytesMut::new();
        buf.extend_from_slice(b"\0"); // unnamed portal
        buf.extend_from_slice(b"stmt1\0"); // statement name
        buf.extend_from_slice(&1_i16.to_be_bytes()); // 1 param format
        buf.extend_from_slice(&0_i16.to_be_bytes()); // text format
        buf.extend_from_slice(&1_i16.to_be_bytes()); // 1 param value
        buf.extend_from_slice(&3_i32.to_be_bytes()); // 3 bytes
        buf.extend_from_slice(b"123"); // value
        buf.extend_from_slice(&1_i16.to_be_bytes()); // 1 result format
        buf.extend_from_slice(&0_i16.to_be_bytes()); // text format

        let msg = FrontendMessage::decode(b'B', &mut buf).unwrap();
        match msg {
            FrontendMessage::Bind {
                portal,
                statement,
                param_formats,
                param_values,
                result_formats,
            } => {
                assert_eq!(portal, "");
                assert_eq!(statement, "stmt1");
                assert_eq!(param_formats, vec![0]);
                assert_eq!(param_values, vec![Some(b"123".to_vec())]);
                assert_eq!(result_formats, vec![0]);
            }
            _ => panic!("Expected Bind message"),
        }
    }

    #[test]
    fn test_decode_bind_null_param() {
        let mut buf = BytesMut::new();
        buf.extend_from_slice(b"\0\0"); // unnamed portal, unnamed statement
        buf.extend_from_slice(&0_i16.to_be_bytes()); // 0 param formats
        buf.extend_from_slice(&1_i16.to_be_bytes()); // 1 param value
        buf.extend_from_slice(&(-1_i32).to_be_bytes()); // NULL
        buf.extend_from_slice(&0_i16.to_be_bytes()); // 0 result formats

        let msg = FrontendMessage::decode(b'B', &mut buf).unwrap();
        match msg {
            FrontendMessage::Bind { param_values, .. } => {
                assert_eq!(param_values, vec![None]);
            }
            _ => panic!("Expected Bind message"),
        }
    }

    #[test]
    fn test_decode_execute() {
        let mut buf = BytesMut::new();
        buf.extend_from_slice(b"\0"); // unnamed portal
        buf.extend_from_slice(&0_i32.to_be_bytes()); // no row limit

        let msg = FrontendMessage::decode(b'E', &mut buf).unwrap();
        match msg {
            FrontendMessage::Execute { portal, max_rows } => {
                assert_eq!(portal, "");
                assert_eq!(max_rows, 0);
            }
            _ => panic!("Expected Execute message"),
        }
    }

    #[test]
    fn test_decode_describe() {
        let mut buf = BytesMut::new();
        buf.extend_from_slice(&[b'S']);
        buf.extend_from_slice(b"stmt1\0");

        let msg = FrontendMessage::decode(b'D', &mut buf).unwrap();
        match msg {
            FrontendMessage::Describe { target, name } => {
                assert_eq!(target, DescribeTarget::Statement);
                assert_eq!(name, "stmt1");
            }
            _ => panic!("Expected Describe message"),
        }
    }

    #[test]
    fn test_decode_close() {
        let mut buf = BytesMut::new();
        buf.extend_from_slice(&[b'P']);
        buf.extend_from_slice(b"portal1\0");

        let msg = FrontendMessage::decode(b'C', &mut buf).unwrap();
        match msg {
            FrontendMessage::Close { target, name } => {
                assert_eq!(target, DescribeTarget::Portal);
                assert_eq!(name, "portal1");
            }
            _ => panic!("Expected Close message"),
        }
    }

    #[test]
    fn test_decode_sync() {
        let mut buf = BytesMut::new();
        let msg = FrontendMessage::decode(b'S', &mut buf).unwrap();
        assert!(matches!(msg, FrontendMessage::Sync));
    }

    #[test]
    fn test_decode_flush() {
        let mut buf = BytesMut::new();
        let msg = FrontendMessage::decode(b'H', &mut buf).unwrap();
        assert!(matches!(msg, FrontendMessage::Flush));
    }

    #[test]
    fn test_decode_terminate() {
        let mut buf = BytesMut::new();
        let msg = FrontendMessage::decode(b'X', &mut buf).unwrap();
        assert!(matches!(msg, FrontendMessage::Terminate));
    }

    #[test]
    fn test_decode_copy_data() {
        let mut buf = BytesMut::new();
        buf.extend_from_slice(b"row data here");

        let msg = FrontendMessage::decode(b'd', &mut buf).unwrap();
        match msg {
            FrontendMessage::CopyData(data) => {
                assert_eq!(data, b"row data here");
            }
            _ => panic!("Expected CopyData message"),
        }
    }

    #[test]
    fn test_decode_copy_done() {
        let mut buf = BytesMut::new();
        let msg = FrontendMessage::decode(b'c', &mut buf).unwrap();
        assert!(matches!(msg, FrontendMessage::CopyDone));
    }

    #[test]
    fn test_decode_copy_fail() {
        let mut buf = BytesMut::new();
        buf.extend_from_slice(b"error reason\0");

        let msg = FrontendMessage::decode(b'f', &mut buf).unwrap();
        match msg {
            FrontendMessage::CopyFail { message } => {
                assert_eq!(message, "error reason");
            }
            _ => panic!("Expected CopyFail message"),
        }
    }

    #[test]
    fn test_decode_cleartext_password() {
        let mut buf = BytesMut::new();
        buf.extend_from_slice(b"mypassword\0");

        let msg = FrontendMessage::decode(b'p', &mut buf).unwrap();
        match msg {
            FrontendMessage::Password(PasswordMessage::Cleartext(pw)) => {
                assert_eq!(pw, "mypassword");
            }
            _ => panic!("Expected Cleartext password"),
        }
    }

    #[test]
    fn test_decode_invalid_type() {
        let mut buf = BytesMut::new();
        let result = FrontendMessage::decode(0xFF, &mut buf);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_unsupported_protocol() {
        let mut buf = BytesMut::new();
        buf.extend_from_slice(&131072_i32.to_be_bytes()); // version 2.0
        let result = FrontendMessage::decode_startup(&mut buf);
        assert!(result.is_err());
    }
}

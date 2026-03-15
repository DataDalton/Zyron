//! PostgreSQL wire protocol v3 message definitions.
//!
//! Defines all frontend (client-to-server) and backend (server-to-client)
//! messages, plus protocol-level error types for message parsing failures.

pub mod backend;
pub mod frontend;

pub use backend::{
    AuthenticationMessage, BackendMessage, ErrorFields, FieldDescription, TransactionState,
};
pub use frontend::{DescribeTarget, FrontendMessage, PasswordMessage, StartupMessage};

/// Protocol-level errors for message parsing and framing.
#[derive(Debug, thiserror::Error)]
pub enum ProtocolError {
    #[error("Invalid message type: 0x{0:02x}")]
    InvalidMessageType(u8),

    #[error("Message too large: {size} bytes, max {max}")]
    MessageTooLarge { size: usize, max: usize },

    #[error("Malformed message: {0}")]
    Malformed(String),

    #[error("Unsupported protocol version: {0}")]
    UnsupportedProtocol(i32),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Connection closed")]
    ConnectionClosed,

    #[error("Authentication failed: {0}")]
    AuthFailed(String),

    #[error("Database error: {0}")]
    Database(#[from] zyron_common::ZyronError),
}

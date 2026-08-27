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

    /// An ErrorResponse the server sent, with its SQLSTATE kept.
    ///
    /// The code is the difference between an error a caller should give up on
    /// and one it should retry. A node refusing work because it is at capacity
    /// sends 53400, and a client that has flattened that into a message string
    /// has thrown away the only machine-readable part of the answer.
    #[error("server error {code}: {message}")]
    Server {
        code: String,
        message: String,
        hint: Option<String>,
    },
}

impl ProtocolError {
    /// The SQLSTATE the server sent, when the error came from the server.
    pub fn sqlstate(&self) -> Option<&str> {
        match self {
            ProtocolError::Server { code, .. } => Some(code.as_str()),
            _ => None,
        }
    }

    /// Whether the server refused the work rather than failing it, which is
    /// the case a caller may retry or route elsewhere.
    pub fn is_admission_shed(&self) -> bool {
        self.sqlstate() == Some("53400")
    }
}

//! Error types for ZyronDB.

use thiserror::Error;

/// Result type alias using ZyronError.
pub type Result<T> = std::result::Result<T, ZyronError>;

/// Errors that can occur in ZyronDB operations.
#[derive(Debug, Error)]
pub enum ZyronError {
    // I/O errors
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    // Storage errors
    #[error("I/O error: {0}")]
    IoError(String),

    #[error("Page not found: {page_id}")]
    PageNotFound { page_id: u64 },

    #[error("Page corrupted: {page_id}, reason: {reason}")]
    PageCorrupted { page_id: u64, reason: String },

    #[error("Buffer pool full, unable to allocate frame")]
    BufferPoolFull,

    #[error("Page full, unable to insert tuple")]
    PageFull,

    #[error("Tuple not found: {0}")]
    TupleNotFound(String),

    #[error("Page size mismatch: expected {expected}, got {actual}")]
    PageSizeMismatch { expected: usize, actual: usize },

    // B+ tree errors
    #[error("Key not found")]
    KeyNotFound,

    #[error("Duplicate key")]
    DuplicateKey,

    #[error("B+ tree node full")]
    NodeFull,

    #[error("B+ tree node underflow")]
    NodeUnderflow,

    #[error("Invalid node type")]
    InvalidNodeType,

    #[error("Key too large: {size} bytes (max {max})")]
    KeyTooLarge { size: usize, max: usize },

    #[error("B+ tree corrupted: {0}")]
    BTreeCorrupted(String),

    // WAL errors
    #[error("WAL write failed: {0}")]
    WalWriteFailed(String),

    #[error("WAL corrupted at LSN {lsn}: {reason}")]
    WalCorrupted { lsn: u64, reason: String },

    #[error("Recovery failed: {0}")]
    RecoveryFailed(String),

    // Type errors
    #[error("Type mismatch: expected {expected}, got {actual}")]
    TypeMismatch { expected: String, actual: String },

    #[error("Unsupported type: {0}")]
    UnsupportedType(String),

    #[error("Null value not allowed")]
    NullNotAllowed,

    // Transaction errors
    #[error("Transaction aborted: {0}")]
    TransactionAborted(String),

    #[error("Write conflict on page {page_id}")]
    WriteConflict { page_id: u64 },

    #[error("Deadlock detected")]
    DeadlockDetected,

    // Catalog errors
    #[error("Table not found: {0}")]
    TableNotFound(String),

    #[error("Table already exists: {0}")]
    TableAlreadyExists(String),

    #[error("Column not found: {0}")]
    ColumnNotFound(String),

    #[error("Index not found: {0}")]
    IndexNotFound(String),

    // Query errors
    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Plan error: {0}")]
    PlanError(String),

    #[error("Execution error: {0}")]
    ExecutionError(String),

    // Configuration errors
    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Invalid parameter: {name} = {value}")]
    InvalidParameter { name: String, value: String },

    // Internal errors
    #[error("Internal error: {0}")]
    Internal(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Error as IoError, ErrorKind};

    #[test]
    fn test_io_error_conversion() {
        let io_err = IoError::new(ErrorKind::NotFound, "file not found");
        let zyron_err: ZyronError = io_err.into();
        assert!(matches!(zyron_err, ZyronError::Io(_)));
        assert!(zyron_err.to_string().contains("I/O error"));
    }

    #[test]
    fn test_page_not_found_display() {
        let err = ZyronError::PageNotFound { page_id: 42 };
        assert_eq!(err.to_string(), "Page not found: 42");
    }

    #[test]
    fn test_page_corrupted_display() {
        let err = ZyronError::PageCorrupted {
            page_id: 100,
            reason: "invalid checksum".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "Page corrupted: 100, reason: invalid checksum"
        );
    }

    #[test]
    fn test_buffer_pool_full_display() {
        let err = ZyronError::BufferPoolFull;
        assert_eq!(err.to_string(), "Buffer pool full, unable to allocate frame");
    }

    #[test]
    fn test_page_size_mismatch_display() {
        let err = ZyronError::PageSizeMismatch {
            expected: 16384,
            actual: 8192,
        };
        assert_eq!(
            err.to_string(),
            "Page size mismatch: expected 16384, got 8192"
        );
    }

    #[test]
    fn test_wal_errors_display() {
        let err = ZyronError::WalWriteFailed("disk full".to_string());
        assert_eq!(err.to_string(), "WAL write failed: disk full");

        let err = ZyronError::WalCorrupted {
            lsn: 12345,
            reason: "truncated record".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "WAL corrupted at LSN 12345: truncated record"
        );

        let err = ZyronError::RecoveryFailed("missing segment".to_string());
        assert_eq!(err.to_string(), "Recovery failed: missing segment");
    }

    #[test]
    fn test_type_errors_display() {
        let err = ZyronError::TypeMismatch {
            expected: "INT64".to_string(),
            actual: "VARCHAR".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "Type mismatch: expected INT64, got VARCHAR"
        );

        let err = ZyronError::UnsupportedType("GEOMETRY".to_string());
        assert_eq!(err.to_string(), "Unsupported type: GEOMETRY");

        let err = ZyronError::NullNotAllowed;
        assert_eq!(err.to_string(), "Null value not allowed");
    }

    #[test]
    fn test_transaction_errors_display() {
        let err = ZyronError::TransactionAborted("serialization failure".to_string());
        assert_eq!(
            err.to_string(),
            "Transaction aborted: serialization failure"
        );

        let err = ZyronError::WriteConflict { page_id: 999 };
        assert_eq!(err.to_string(), "Write conflict on page 999");

        let err = ZyronError::DeadlockDetected;
        assert_eq!(err.to_string(), "Deadlock detected");
    }

    #[test]
    fn test_catalog_errors_display() {
        let err = ZyronError::TableNotFound("users".to_string());
        assert_eq!(err.to_string(), "Table not found: users");

        let err = ZyronError::TableAlreadyExists("orders".to_string());
        assert_eq!(err.to_string(), "Table already exists: orders");

        let err = ZyronError::ColumnNotFound("email".to_string());
        assert_eq!(err.to_string(), "Column not found: email");

        let err = ZyronError::IndexNotFound("idx_users_email".to_string());
        assert_eq!(err.to_string(), "Index not found: idx_users_email");
    }

    #[test]
    fn test_query_errors_display() {
        let err = ZyronError::ParseError("unexpected token".to_string());
        assert_eq!(err.to_string(), "Parse error: unexpected token");

        let err = ZyronError::PlanError("no valid join order".to_string());
        assert_eq!(err.to_string(), "Plan error: no valid join order");

        let err = ZyronError::ExecutionError("division by zero".to_string());
        assert_eq!(err.to_string(), "Execution error: division by zero");
    }

    #[test]
    fn test_config_errors_display() {
        let err = ZyronError::ConfigError("missing data_dir".to_string());
        assert_eq!(err.to_string(), "Configuration error: missing data_dir");

        let err = ZyronError::InvalidParameter {
            name: "max_connections".to_string(),
            value: "-1".to_string(),
        };
        assert_eq!(err.to_string(), "Invalid parameter: max_connections = -1");
    }

    #[test]
    fn test_internal_error_display() {
        let err = ZyronError::Internal("assertion failed".to_string());
        assert_eq!(err.to_string(), "Internal error: assertion failed");
    }

    #[test]
    fn test_result_type_alias() {
        fn returns_ok() -> Result<i32> {
            Ok(42)
        }

        fn returns_err() -> Result<i32> {
            Err(ZyronError::Internal("test".to_string()))
        }

        assert!(returns_ok().is_ok());
        assert!(returns_err().is_err());
    }

    #[test]
    fn test_error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ZyronError>();
    }
}

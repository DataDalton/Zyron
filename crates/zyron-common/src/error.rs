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

    #[error("Optimistic version conflict during page access")]
    VersionConflict,

    #[error("Page not in buffer pool cache")]
    CacheMiss,

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

    #[error("Transaction conflict for txn {txn_id}: {reason}")]
    TransactionConflict { txn_id: u64, reason: String },

    // Catalog errors
    #[error("Database not found: {0}")]
    DatabaseNotFound(String),

    #[error("Database already exists: {0}")]
    DatabaseAlreadyExists(String),

    #[error("Schema not found: {0}")]
    SchemaNotFound(String),

    #[error("Schema already exists: {0}")]
    SchemaAlreadyExists(String),

    #[error("Table not found: {0}")]
    TableNotFound(String),

    #[error("Table already exists: {0}")]
    TableAlreadyExists(String),

    #[error("Column not found: {0}")]
    ColumnNotFound(String),

    #[error("Index not found: {0}")]
    IndexNotFound(String),

    #[error("Index already exists: {0}")]
    IndexAlreadyExists(String),

    #[error("Catalog corrupted: {0}")]
    CatalogCorrupted(String),

    // Authentication and authorization errors
    #[error("Authentication failed for user \"{0}\"")]
    AuthenticationFailed(String),

    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    #[error("Insufficient clearance for column \"{0}\"")]
    InsufficientClearance(String),

    #[error("Role not found: {0}")]
    RoleNotFound(String),

    #[error("Role already exists: {0}")]
    RoleAlreadyExists(String),

    #[error("Circular role dependency detected")]
    CircularRoleDependency,

    #[error("Invalid credential: {0}")]
    InvalidCredential(String),

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

    // Columnar storage errors
    #[error("Encoding failed: {0}")]
    EncodingFailed(String),

    #[error("Decoding failed: {0}")]
    DecodingFailed(String),

    #[error("Invalid .zyr file: {0}")]
    InvalidZyrFile(String),

    #[error("Corrupted segment in column {column_id}: {reason}")]
    CorruptedSegment { column_id: u32, reason: String },

    #[error("Compaction failed: {0}")]
    CompactionFailed(String),

    // Brute force and rate limiting errors
    #[error("Account locked: {0}")]
    AccountLocked(String),

    #[error("IP blocked: {0}")]
    IpBlocked(String),

    #[error("Rate limited, retry after {0}ms")]
    RateLimited(u64),

    // Encryption errors
    #[error("Encryption failed: {0}")]
    EncryptionFailed(String),

    #[error("Decryption failed: {0}")]
    DecryptionFailed(String),

    #[error("Encryption key not found: {0}")]
    EncryptionKeyNotFound(String),

    // Policy errors
    #[error("Policy not found: {0}")]
    PolicyNotFound(String),

    #[error("Policy already exists: {0}")]
    PolicyAlreadyExists(String),

    #[error("Webhook verification failed: {0}")]
    WebhookVerificationFailed(String),

    // Versioning errors
    #[error("Version not found: {0}")]
    VersionNotFound(u64),

    #[error("Branch conflict: {0}")]
    BranchConflict(String),

    #[error("Branch not found: {0}")]
    BranchNotFound(String),

    #[error("Branch already exists: {0}")]
    BranchAlreadyExists(String),

    #[error("Version log corrupted for table {table_id}: {reason}")]
    VersionLogCorrupted { table_id: u32, reason: String },

    #[error("Temporal constraint violated: {0}")]
    TemporalViolation(String),

    #[error("SCD configuration error: {0}")]
    ScdConfigError(String),

    // CDC errors
    #[error("Slot not found: {0}")]
    SlotNotFound(String),

    #[error("Slot already exists: {0}")]
    SlotAlreadyExists(String),

    #[error("CDC stream error: {0}")]
    CdcStreamError(String),

    #[error("CDC feed not enabled for table {table_id}")]
    CdcFeedNotEnabled { table_id: u32 },

    #[error("CDC ingest error: {0}")]
    CdcIngestError(String),

    #[error("CDC decoder error: {0}")]
    CdcDecoderError(String),

    #[error("Replication slot lag exceeded limit for slot \"{0}\"")]
    SlotLagExceeded(String),

    #[error("CDC snapshot failed: {0}")]
    CdcSnapshotFailed(String),

    // Pipeline errors
    #[error("Pipeline not found: {0}")]
    PipelineNotFound(String),

    #[error("Pipeline already exists: {0}")]
    PipelineAlreadyExists(String),

    #[error("Pipeline execution failed: {0}")]
    PipelineExecutionFailed(String),

    #[error("Circular pipeline dependency: {0}")]
    CircularPipelineDependency(String),

    #[error("Pipeline SLA breach for \"{pipeline}\": {reason}")]
    PipelineSlaBreach { pipeline: String, reason: String },

    // Quality check errors
    #[error("Quality check failed on table \"{table}\": {reason}")]
    QualityCheckFailed { table: String, reason: String },

    #[error("Quality drift detected on column \"{column}\": {reason}")]
    QualityDriftDetected { column: String, reason: String },

    // Trigger errors
    #[error("Trigger not found: {0}")]
    TriggerNotFound(String),

    #[error("Trigger already exists: {0}")]
    TriggerAlreadyExists(String),

    #[error("Trigger execution failed: {0}")]
    TriggerExecutionFailed(String),

    #[error("Trigger cancelled operation: {0}")]
    TriggerCancelled(String),

    #[error("Trigger recursion limit reached for \"{trigger}\" at depth {depth}")]
    TriggerRecursionLimit { trigger: String, depth: u32 },

    // UDF errors
    #[error("Function not found: {0}")]
    FunctionNotFound(String),

    #[error("Function already exists: {0}")]
    FunctionAlreadyExists(String),

    #[error("UDF execution error: {0}")]
    UdfExecutionError(String),

    #[error("Failed to load UDF library \"{library}\": {reason}")]
    UdfLibraryLoadFailed { library: String, reason: String },

    // Materialized view errors
    #[error("Materialized view not found: {0}")]
    MaterializedViewNotFound(String),

    #[error("Materialized view already exists: {0}")]
    MaterializedViewAlreadyExists(String),

    #[error("Materialized view refresh failed: {0}")]
    MaterializedViewRefreshFailed(String),

    // Schedule errors
    #[error("Schedule not found: {0}")]
    ScheduleNotFound(String),

    #[error("Schedule already exists: {0}")]
    ScheduleAlreadyExists(String),

    // Stored procedure errors
    #[error("Procedure not found: {0}")]
    ProcedureNotFound(String),

    #[error("Procedure execution error: {0}")]
    ProcedureExecutionError(String),

    // Event handler errors
    #[error("Event handler not found: {0}")]
    EventHandlerNotFound(String),

    // Dependency errors
    #[error("Cannot drop {object}, depended on by: {dependents}")]
    DependencyViolation { object: String, dependents: String },

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
        assert_eq!(
            err.to_string(),
            "Buffer pool full, unable to allocate frame"
        );
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
    fn test_versioning_errors_display() {
        let err = ZyronError::VersionNotFound(42);
        assert_eq!(err.to_string(), "Version not found: 42");

        let err = ZyronError::BranchConflict("page 100 modified in both branches".to_string());
        assert_eq!(
            err.to_string(),
            "Branch conflict: page 100 modified in both branches"
        );

        let err = ZyronError::BranchNotFound("dev".to_string());
        assert_eq!(err.to_string(), "Branch not found: dev");

        let err = ZyronError::BranchAlreadyExists("dev".to_string());
        assert_eq!(err.to_string(), "Branch already exists: dev");

        let err = ZyronError::VersionLogCorrupted {
            table_id: 5,
            reason: "truncated entry".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "Version log corrupted for table 5: truncated entry"
        );

        let err = ZyronError::TemporalViolation("period start must precede end".to_string());
        assert_eq!(
            err.to_string(),
            "Temporal constraint violated: period start must precede end"
        );

        let err = ZyronError::ScdConfigError("unknown scd_type 5".to_string());
        assert_eq!(
            err.to_string(),
            "SCD configuration error: unknown scd_type 5"
        );
    }

    #[test]
    fn test_cdc_errors_display() {
        let err = ZyronError::SlotNotFound("my_slot".to_string());
        assert_eq!(err.to_string(), "Slot not found: my_slot");

        let err = ZyronError::SlotAlreadyExists("my_slot".to_string());
        assert_eq!(err.to_string(), "Slot already exists: my_slot");

        let err = ZyronError::CdcStreamError("sink unreachable".to_string());
        assert_eq!(err.to_string(), "CDC stream error: sink unreachable");

        let err = ZyronError::CdcFeedNotEnabled { table_id: 42 };
        assert_eq!(err.to_string(), "CDC feed not enabled for table 42");

        let err = ZyronError::CdcIngestError("parse failed".to_string());
        assert_eq!(err.to_string(), "CDC ingest error: parse failed");

        let err = ZyronError::CdcDecoderError("unknown record type".to_string());
        assert_eq!(err.to_string(), "CDC decoder error: unknown record type");

        let err = ZyronError::SlotLagExceeded("my_slot".to_string());
        assert_eq!(
            err.to_string(),
            "Replication slot lag exceeded limit for slot \"my_slot\""
        );

        let err = ZyronError::CdcSnapshotFailed("table not found".to_string());
        assert_eq!(err.to_string(), "CDC snapshot failed: table not found");
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

//! Write-Ahead Log (WAL) for ZyronDB.
//!
//! Provides durability by logging all modifications before they are applied.
//! Supports crash recovery through log replay.

pub mod constants;
pub mod reader;
pub mod record;
pub mod segment;
pub mod writer;

pub use reader::{RecoveryManager, RecoveryResult, WalReader};
pub use record::{LogRecord, LogRecordType, Lsn};
pub use segment::{LogSegment, SegmentId};
pub use writer::{TxnWalHandle, WalWriter, WalWriterConfig};

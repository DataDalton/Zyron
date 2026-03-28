//! Write-Ahead Log (WAL) for ZyronDB.
//!
//! Provides durability by logging all modifications before they are applied.
//! Supports crash recovery through log replay.
//!
//! The `WalWriter` uses lock-free atomic operations for LSN assignment and a
//! ring buffer for high-throughput concurrent writes. A dedicated flush thread
//! batches writes to disk with group commit for amortized fsync cost.

pub mod checksum;
pub mod constants;
pub mod reader;
pub mod record;
pub mod ring_buffer;
pub mod segment;
pub mod sequencer;
pub mod writer;

pub use checksum::data_checksum;
pub use reader::{RecoveryManager, RecoveryResult, WalReader};
pub use record::{LogRecord, LogRecordType, Lsn};
pub use segment::{LogSegment, SegmentId};
pub use writer::{TxnWalHandle, WalWriter, WalWriterConfig};

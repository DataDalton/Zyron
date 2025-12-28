//! Write-Ahead Log (WAL) for ZyronDB.
//!
//! Provides durability by logging all modifications before they are applied.
//! Supports crash recovery through log replay.

pub mod record;
pub mod segment;
pub mod writer;
pub mod reader;

pub use record::{LogRecord, LogRecordType, Lsn};
pub use segment::{LogSegment, SegmentId};
pub use writer::WalWriter;
pub use reader::WalReader;

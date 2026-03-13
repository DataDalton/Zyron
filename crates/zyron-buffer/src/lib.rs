//! Buffer pool management for ZyronDB.
//!
//! This crate provides in-memory page caching with:
//! - Fixed-size buffer pool with configurable page count
//! - Clock eviction policy for cache management
//! - Pin counting for concurrent access
//! - Dirty page tracking for write-back

pub mod background_writer;
mod frame;
mod page_table;
mod pool;
mod replacer;

pub use background_writer::{BackgroundWriter, BackgroundWriterConfig, WriteFn};
pub use frame::{BufferFrame, FrameId};
pub use pool::{BufferPool, BufferPoolConfig, EvictedPage};
pub use replacer::{ClockReplacer, Replacer};

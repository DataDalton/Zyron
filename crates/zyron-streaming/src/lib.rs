//! Streaming and real-time processing engine for ZyronDB.
//!
//! Implements an operator DAG model with push-based micro-batch processing,
//! barrier-based checkpointing, and zero-lock hot paths for windowed
//! aggregations, stream-stream joins, watermark tracking, late data
//! handling, and backpressure management.

pub mod accumulator;
pub mod backpressure;
pub mod checkpoint;
pub mod column;
pub mod hash;
pub mod job;
pub mod late_data;
pub mod metrics;
pub mod record;
pub mod sink_connector;
pub mod source_connector;
pub mod spsc;
pub mod state;
pub mod stream_join;
pub mod stream_operator;
pub mod watermark;
pub mod window;

// Re-exports for convenience.
pub use column::{NullBitmap, StreamBatch, StreamColumn, StreamColumnData};
pub use hash::{FlatHashTable, FlatU64Map};
pub use record::{ChangeFlag, ChangelogMode, StreamRecord, StreamRecordPool};
pub use spsc::{SpscReceiver, SpscSender};
pub use watermark::Watermark;
pub use window::{WindowAssigner, WindowRange, WindowType};

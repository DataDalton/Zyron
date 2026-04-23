//! Streaming and real-time processing engine for ZyronDB.
//!
//! Implements an operator DAG model with push-based micro-batch processing,
//! barrier-based checkpointing, and zero-lock hot paths for windowed
//! aggregations, stream-stream joins, watermark tracking, late data
//! handling, and backpressure management.

pub mod accumulator;
pub mod agg_runner;
pub mod backpressure;
pub mod checkpoint;
pub mod column;
pub mod copy_external;
pub mod dlq;
pub mod external_sink;
pub mod external_source;
pub mod format;
pub mod hash;
pub mod interval_join_runner;
pub mod job;
pub mod job_runner;
pub mod late_data;
pub mod metrics;
pub mod record;
pub mod retry;
pub mod row_codec;
pub mod sink_connector;
pub mod source_connector;
pub mod spsc;
pub mod state;
pub mod stream_join;
pub mod stream_operator;
pub mod upsert_sink;
pub mod watermark;
pub mod window;
pub mod window_state;

// Re-exports for convenience.
pub use column::{NullBitmap, StreamBatch, StreamColumn, StreamColumnData};
pub use hash::{FlatHashTable, FlatU64Map};
pub use job_runner::{BinaryOpKind, ExprSpec, StreamJobHandle, StreamingJobSpec};
pub use late_data::{LateDataHandler, LateDataPolicy, LateDataStats};
pub use record::{ChangeFlag, ChangelogMode, StreamRecord, StreamRecordPool};
pub use sink_connector::{ZyronRowSink, ZyronSinkAdapter};
pub use source_connector::{CdfChange, ZyronSourceAdapter, ZyronTableSource};
pub use spsc::{SpscReceiver, SpscSender};
pub use upsert_sink::ZyronUpsertSink;
pub use watermark::{Watermark, WatermarkGenerator, WatermarkStrategy};
pub use window::{SessionAssigner, WindowAssigner, WindowRange, WindowType};
pub use window_state::WindowStateStore;

//! Change Data Capture (CDC) for Zyron.
//!
//! Provides change data feeds, RETURNING with OLD/NEW, replication slots,
//! logical decoders, outbound CDC streams, inbound CDC ingestion,
//! multi-table publications, initial snapshots, and retention management.

pub mod apply;
pub mod cdc_ingest;
pub mod cdc_stream;
pub mod change_feed;
pub mod change_stream;
pub mod decoder;
pub mod derived_index;
pub mod format;
pub mod metrics;
pub mod publication;
pub mod replication_slot;
pub mod retention;
pub mod returning;
pub mod schema_evolution;
pub mod segment_columns;
pub mod sink_io;
pub mod snapshot;
pub mod source_io;

pub use apply::{
    ApplyAction, ApplyChange, ApplyPlan, ApplyRunCounts, ApplySpec, ScdType, SequenceValue,
    TrackHistory,
};
pub use cdc_ingest::{CdcIngestConfig, CdcIngestManager, CdcIngestSource, OnConflict};
pub use cdc_stream::{
    CdcOutputStream, CdcSink, CdcSinkConfig, CdcStreamManager, OutputFormat, TxnDecision,
    build_sink, drive_stream_changes, drive_stream_once,
};
pub use change_feed::{
    CdfCodec, CdfRegistry, ChangeDataFeed, ChangeRange, ChangeRecord, ChangeRecordRef, ChangeType,
    CompactionPlan, DerivedChangeSource, FeedBoundary, FeedConfig, FrameRestore, ReadPlan,
    SegmentSummary, SourceRead, TxnSpan, VersionCount, restore_logged_frames, version_index,
};
pub use change_stream::{
    ALERT_TEMPLATES, AlertFiring, AlertTemplate, AttentionReason, ChangeStreamRuntime, ResetTarget,
    SourceWindow, StaleReason, StreamReadPlan, StreamStatus,
};
pub use decoder::{
    AvroDecoder, DebeziumDecoder, DecodedChange, DecoderPlugin, LogicalDecoder, Wal2JsonDecoder,
    ZyronCdcDecoder,
};
pub use derived_index::{CountedVersion, DerivedBoundary, DerivedRecordIndex};
pub use metrics::CdcMetrics;
pub use publication::{Publication, PublicationManager};
pub use replication_slot::{ReplicationSlot, SlotLagConfig, SlotManager};
pub use retention::{CdcRetentionManager, CdcRetentionPolicy};
pub use returning::{OldNewResolver, ReturnClause, ReturnColumn, ReturnSource};
pub use schema_evolution::ProjectionSchema;
pub use snapshot::{SnapshotExport, SnapshotReader, TableSnapshotInfo};

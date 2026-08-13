//! Change Data Capture (CDC) for Zyron.
//!
//! Provides change data feeds, RETURNING with OLD/NEW, replication slots,
//! logical decoders, outbound CDC streams, inbound CDC ingestion,
//! multi-table publications, initial snapshots, and retention management.

pub mod cdc_ingest;
pub mod cdc_stream;
pub mod change_feed;
pub mod decoder;
pub mod metrics;
pub mod publication;
pub mod replication_slot;
pub mod retention;
pub mod returning;
pub mod sink_io;
pub mod snapshot;
pub mod source_io;

pub use cdc_ingest::{CdcIngestConfig, CdcIngestManager, CdcIngestSource, OnConflict};
pub use cdc_stream::{
    CdcOutputStream, CdcSink, CdcSinkConfig, CdcStreamManager, OutputFormat, build_sink,
    drive_stream_changes, drive_stream_once,
};
pub use change_feed::{CdfRegistry, ChangeDataFeed, ChangeRecord, ChangeType};
pub use decoder::{
    AvroDecoder, DebeziumDecoder, DecodedChange, DecoderPlugin, LogicalDecoder, Wal2JsonDecoder,
    ZyronCdcDecoder,
};
pub use metrics::CdcMetrics;
pub use publication::{Publication, PublicationManager};
pub use replication_slot::{ReplicationSlot, SlotLagConfig, SlotManager};
pub use retention::{CdcRetentionManager, CdcRetentionPolicy};
pub use returning::{OldNewResolver, ReturnClause, ReturnColumn, ReturnSource};
pub use snapshot::{SnapshotExport, SnapshotReader, TableSnapshotInfo};

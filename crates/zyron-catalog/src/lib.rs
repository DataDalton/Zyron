//! Catalog module for ZyronDB.
//!
//! Manages database metadata: databases, schemas, tables, columns, indexes,
//! constraints, and statistics. Persists catalog entries in heap file system
//! tables, caches them in memory with LRU eviction, and logs DDL operations
//! to the WAL for crash safety.

pub mod cache;
pub mod catalog;
pub mod checkpoint;
pub mod encoding;
pub mod ids;
pub mod resolver;
pub mod schema;
pub mod sequence;
pub mod stats;
pub mod storage;

pub use cache::{CatalogCache, TableIndexSnapshot};
pub use catalog::{Catalog, DropOutcome};
pub use ids::{
    ColumnId, DatabaseId, EndpointId, ExternalSinkId, ExternalSourceId, IndexId, Oid, OidAllocator,
    PublicationId, SYSTEM_DATABASE_ID, SYSTEM_SCHEMA_ID, SYSTEM_SCHEMA_NAME, SchemaId,
    SecurityMapId, SequenceId, StreamingJobId, SubscriptionId, TableId, USER_OID_START,
};
pub use resolver::NameResolver;
pub use schema::SequenceEntry;
pub use schema::{
    AggregateEntry, BackpressurePolicy, CatalogClassification, CatalogStreamingWriteMode,
    ColumnEntry, CommentEntry, ConstraintEntry, ConstraintType, DatabaseEntry, EndpointAuthMode,
    EndpointEntry, EndpointKind, EndpointMessageFormat, EndpointOutputFormat, EventHandlerEntry,
    ExpectationAction, ExpectationEntry, ExternalBackend, ExternalFormat, ExternalMode,
    ExternalSinkEntry, ExternalSourceEntry, FunctionEntry, HttpMethod, IndexColumnEntry,
    IndexEntry, IndexType, MaterializedViewEntry, PipelineEntry, ProcedureEntry, PublicationEntry,
    PublicationTableEntry, RateLimitPeriod, RateLimitScope, RateLimitSpec, ReferentialAction,
    RowFormat, ScheduleEntry, SchemaEntry, SecurityMapEntry, SecurityMapKind, StreamingJobEntry,
    StreamingJobStatus, SubscriptionEntry, SubscriptionMode, SubscriptionState, TableEntry,
    TriggerEntry, VersionTagEntry, ViewEntry,
};
pub use sequence::LiveSequence;
pub use stats::{ColumnStats, Histogram, TableStats, analyze_table};
pub use storage::{CatalogStorage, HeapCatalogStorage};

//! Catalog persistence layer.
//!
//! CatalogStorage trait abstracts how catalog entries are stored and retrieved.
//! HeapCatalogStorage implements it using heap files from zyron-storage,
//! storing each entry type in a dedicated system table.

use crate::ids::*;
use crate::schema::*;
use async_trait::async_trait;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use zyron_buffer::BufferPool;
use zyron_common::{PAGE_SIZE, Result, ZyronError};
use zyron_storage::{DiskManager, HeapFile, HeapFileConfig, Tuple, TupleId};

// System table file ID assignments (reserved range 100-119).
const DATABASES_HEAP_FILE_ID: u32 = 100;
const DATABASES_FSM_FILE_ID: u32 = 101;
const SCHEMAS_HEAP_FILE_ID: u32 = 102;
const SCHEMAS_FSM_FILE_ID: u32 = 103;
const TABLES_HEAP_FILE_ID: u32 = 104;
const TABLES_FSM_FILE_ID: u32 = 105;
const COLUMNS_HEAP_FILE_ID: u32 = 106;
const COLUMNS_FSM_FILE_ID: u32 = 107;
const INDEXES_HEAP_FILE_ID: u32 = 108;
const INDEXES_FSM_FILE_ID: u32 = 109;
const STREAMING_JOBS_HEAP_FILE_ID: u32 = 132;
const STREAMING_JOBS_FSM_FILE_ID: u32 = 133;
const EXTERNAL_SOURCES_HEAP_FILE_ID: u32 = 134;
const EXTERNAL_SOURCES_FSM_FILE_ID: u32 = 135;
const EXTERNAL_SINKS_HEAP_FILE_ID: u32 = 136;
const EXTERNAL_SINKS_FSM_FILE_ID: u32 = 137;
const PUBLICATIONS_HEAP_FILE_ID: u32 = 138;
const PUBLICATIONS_FSM_FILE_ID: u32 = 139;
const PUBLICATION_TABLES_HEAP_FILE_ID: u32 = 140;
const PUBLICATION_TABLES_FSM_FILE_ID: u32 = 141;
const SUBSCRIPTIONS_HEAP_FILE_ID: u32 = 142;
const SUBSCRIPTIONS_FSM_FILE_ID: u32 = 143;
const ENDPOINTS_HEAP_FILE_ID: u32 = 144;
const ENDPOINTS_FSM_FILE_ID: u32 = 145;
const SECURITY_MAPS_HEAP_FILE_ID: u32 = 146;
const SECURITY_MAPS_FSM_FILE_ID: u32 = 147;
const LEGAL_HOLDS_HEAP_FILE_ID: u32 = 148;
const LEGAL_HOLDS_FSM_FILE_ID: u32 = 149;
const RETENTION_POLICIES_HEAP_FILE_ID: u32 = 150;
const RETENTION_POLICIES_FSM_FILE_ID: u32 = 151;
const RETENTION_JOBS_HEAP_FILE_ID: u32 = 152;
const RETENTION_JOBS_FSM_FILE_ID: u32 = 153;
const COMPLIANCE_LOG_HEAP_FILE_ID: u32 = 154;
const COMPLIANCE_LOG_FSM_FILE_ID: u32 = 155;
const SEQUENCES_HEAP_FILE_ID: u32 = 156;
const SEQUENCES_FSM_FILE_ID: u32 = 157;
const VIEWS_HEAP_FILE_ID: u32 = 158;
const VIEWS_FSM_FILE_ID: u32 = 159;
const MVIEWS_HEAP_FILE_ID: u32 = 160;
const MVIEWS_FSM_FILE_ID: u32 = 161;
const FUNCTIONS_HEAP_FILE_ID: u32 = 162;
const FUNCTIONS_FSM_FILE_ID: u32 = 163;
const COMMENTS_HEAP_FILE_ID: u32 = 164;
const COMMENTS_FSM_FILE_ID: u32 = 165;
const AGGREGATES_HEAP_FILE_ID: u32 = 166;
const AGGREGATES_FSM_FILE_ID: u32 = 167;
const PROCEDURES_HEAP_FILE_ID: u32 = 168;
const PROCEDURES_FSM_FILE_ID: u32 = 169;
const SCHEDULES_HEAP_FILE_ID: u32 = 170;
const SCHEDULES_FSM_FILE_ID: u32 = 171;
const TRIGGERS_HEAP_FILE_ID: u32 = 172;
const TRIGGERS_FSM_FILE_ID: u32 = 173;
const PIPELINES_HEAP_FILE_ID: u32 = 174;
const PIPELINES_FSM_FILE_ID: u32 = 175;
const EVENT_HANDLERS_HEAP_FILE_ID: u32 = 176;
const EVENT_HANDLERS_FSM_FILE_ID: u32 = 177;
const VERSION_TAGS_HEAP_FILE_ID: u32 = 178;
const VERSION_TAGS_FSM_FILE_ID: u32 = 179;

/// Starting file ID for user-created heap files (heap=200, fsm=201, ...).
const USER_HEAP_FILE_START: u32 = 200;

/// Starting file ID for user-created index files.
const USER_INDEX_FILE_START: u32 = 10000;

/// Scans every tuple in a catalog heap and decodes it, returning CatalogCorrupted
/// on the first tuple that fails to decode. A decode failure means a catalog row
/// is unreadable, dropping it would silently lose a table, index, or other object,
/// so recovery stops loudly instead.
fn scan_decode<T>(
    heap: &HeapFile,
    entity: &str,
    decode: impl Fn(&[u8]) -> Result<T>,
) -> Result<Vec<T>> {
    let mut entries = Vec::new();
    let mut decode_err: Option<ZyronError> = None;
    let guard = heap.scan()?;
    guard.try_for_each(|_tid, view| match decode(view.data) {
        Ok(entry) => {
            entries.push(entry);
            true
        }
        Err(e) => {
            decode_err = Some(ZyronError::CatalogCorrupted(format!(
                "{} catalog tuple failed to decode: {}",
                entity, e
            )));
            false
        }
    });
    if let Some(e) = decode_err {
        return Err(e);
    }
    Ok(entries)
}

/// Abstraction over catalog persistence.
#[async_trait]
pub trait CatalogStorage: Send + Sync {
    // Database operations
    async fn load_databases(&self) -> Result<Vec<DatabaseEntry>>;
    async fn store_database(&self, entry: &DatabaseEntry) -> Result<TupleId>;
    async fn delete_database(&self, id: DatabaseId) -> Result<bool>;

    // Schema operations
    async fn load_schemas(&self) -> Result<Vec<SchemaEntry>>;
    async fn store_schema(&self, entry: &SchemaEntry) -> Result<TupleId>;
    async fn delete_schema(&self, id: SchemaId) -> Result<bool>;

    // Table operations
    async fn load_tables(&self) -> Result<Vec<TableEntry>>;
    /// Resolves a single table by (schema_id, name). The default filters
    /// `load_tables` for trait impls that have no targeted path; the heap
    /// implementation overrides this with an early-return scan so a cache
    /// miss does not deserialize and allocate every table in the catalog.
    async fn load_table_by_name(
        &self,
        schema_id: SchemaId,
        name: &str,
    ) -> Result<Option<TableEntry>> {
        Ok(self
            .load_tables()
            .await?
            .into_iter()
            .find(|t| t.schema_id == schema_id && t.name == name))
    }
    async fn store_table(&self, entry: &TableEntry) -> Result<TupleId>;
    async fn delete_table(&self, id: TableId) -> Result<bool>;

    // Column operations
    async fn load_columns(&self, table_id: TableId) -> Result<Vec<ColumnEntry>>;
    async fn store_columns(&self, columns: &[ColumnEntry]) -> Result<Vec<TupleId>>;
    async fn delete_columns(&self, table_id: TableId) -> Result<usize>;

    // Index operations
    async fn load_indexes(&self) -> Result<Vec<IndexEntry>>;
    async fn store_index(&self, entry: &IndexEntry) -> Result<TupleId>;
    async fn delete_index(&self, id: IndexId) -> Result<bool>;

    // Sequence operations
    async fn load_sequences(&self) -> Result<Vec<SequenceEntry>>;
    async fn store_sequence(&self, entry: &SequenceEntry) -> Result<TupleId>;
    async fn update_sequence(&self, entry: &SequenceEntry) -> Result<bool>;
    async fn delete_sequence(&self, id: u32) -> Result<bool>;

    // View operations
    async fn load_views(&self) -> Result<Vec<ViewEntry>>;
    async fn store_view(&self, entry: &ViewEntry) -> Result<TupleId>;
    async fn update_view(&self, entry: &ViewEntry) -> Result<bool>;
    async fn delete_view(&self, id: u32) -> Result<bool>;

    // Materialized view operations
    async fn load_mviews(&self) -> Result<Vec<MaterializedViewEntry>>;
    async fn store_mview(&self, entry: &MaterializedViewEntry) -> Result<TupleId>;
    async fn update_mview(&self, entry: &MaterializedViewEntry) -> Result<bool>;
    async fn delete_mview(&self, id: u32) -> Result<bool>;

    // Function operations
    async fn load_functions(&self) -> Result<Vec<FunctionEntry>>;
    async fn store_function(&self, entry: &FunctionEntry) -> Result<TupleId>;
    async fn delete_function(&self, id: u32) -> Result<bool>;

    // Comment operations
    async fn load_comments(&self) -> Result<Vec<CommentEntry>>;
    async fn store_comment(&self, entry: &CommentEntry) -> Result<TupleId>;
    async fn delete_comment(&self, id: u32) -> Result<bool>;

    // Aggregate operations
    async fn load_aggregates(&self) -> Result<Vec<AggregateEntry>>;
    async fn store_aggregate(&self, entry: &AggregateEntry) -> Result<TupleId>;
    async fn delete_aggregate(&self, id: u32) -> Result<bool>;

    // Procedure operations
    async fn load_procedures(&self) -> Result<Vec<ProcedureEntry>>;
    async fn store_procedure(&self, entry: &ProcedureEntry) -> Result<TupleId>;
    async fn delete_procedure(&self, id: u32) -> Result<bool>;

    // Schedule operations
    async fn load_schedules(&self) -> Result<Vec<ScheduleEntry>>;
    async fn store_schedule(&self, entry: &ScheduleEntry) -> Result<TupleId>;
    async fn delete_schedule(&self, id: u32) -> Result<bool>;

    // Trigger operations
    async fn load_triggers(&self) -> Result<Vec<TriggerEntry>>;
    async fn store_trigger(&self, entry: &TriggerEntry) -> Result<TupleId>;
    async fn delete_trigger(&self, id: u32) -> Result<bool>;

    // Pipeline operations
    async fn load_pipelines(&self) -> Result<Vec<PipelineEntry>>;
    async fn store_pipeline(&self, entry: &PipelineEntry) -> Result<TupleId>;
    async fn delete_pipeline(&self, id: u32) -> Result<bool>;

    // Event handler operations
    async fn load_event_handlers(&self) -> Result<Vec<EventHandlerEntry>>;
    async fn store_event_handler(&self, entry: &EventHandlerEntry) -> Result<TupleId>;
    async fn delete_event_handler(&self, id: u32) -> Result<bool>;

    // Version tag operations
    async fn load_version_tags(&self) -> Result<Vec<VersionTagEntry>>;
    async fn store_version_tag(&self, entry: &VersionTagEntry) -> Result<TupleId>;
    async fn delete_version_tag(&self, id: u32) -> Result<bool>;

    // Streaming job operations
    async fn load_streaming_jobs(&self) -> Result<Vec<StreamingJobEntry>>;
    async fn store_streaming_job(&self, entry: &StreamingJobEntry) -> Result<TupleId>;
    async fn update_streaming_job(&self, entry: &StreamingJobEntry) -> Result<bool>;
    async fn delete_streaming_job(&self, id: StreamingJobId) -> Result<bool>;

    // External source operations
    async fn load_external_sources(&self) -> Result<Vec<ExternalSourceEntry>>;
    async fn store_external_source(&self, entry: &ExternalSourceEntry) -> Result<TupleId>;
    async fn update_external_source(&self, entry: &ExternalSourceEntry) -> Result<bool>;
    async fn delete_external_source(&self, id: ExternalSourceId) -> Result<bool>;

    // External sink operations
    async fn load_external_sinks(&self) -> Result<Vec<ExternalSinkEntry>>;
    async fn store_external_sink(&self, entry: &ExternalSinkEntry) -> Result<TupleId>;
    async fn update_external_sink(&self, entry: &ExternalSinkEntry) -> Result<bool>;
    async fn delete_external_sink(&self, id: ExternalSinkId) -> Result<bool>;

    // Publication operations
    async fn load_publications(&self) -> Result<Vec<PublicationEntry>>;
    async fn store_publication(&self, entry: &PublicationEntry) -> Result<TupleId>;
    async fn update_publication(&self, entry: &PublicationEntry) -> Result<bool>;
    async fn delete_publication(&self, id: PublicationId) -> Result<bool>;

    // Publication-table junction operations
    async fn load_publication_tables(&self) -> Result<Vec<PublicationTableEntry>>;
    async fn store_publication_table(&self, entry: &PublicationTableEntry) -> Result<TupleId>;
    async fn update_publication_table(&self, entry: &PublicationTableEntry) -> Result<bool>;
    async fn delete_publication_table(
        &self,
        publication_id: PublicationId,
        table_id: TableId,
    ) -> Result<bool>;

    // Subscription operations
    async fn load_subscriptions(&self) -> Result<Vec<SubscriptionEntry>>;
    async fn store_subscription(&self, entry: &SubscriptionEntry) -> Result<TupleId>;
    async fn update_subscription(&self, entry: &SubscriptionEntry) -> Result<bool>;
    async fn delete_subscription(&self, id: SubscriptionId) -> Result<bool>;

    // Endpoint operations
    async fn load_endpoints(&self) -> Result<Vec<EndpointEntry>>;
    async fn store_endpoint(&self, entry: &EndpointEntry) -> Result<TupleId>;
    async fn update_endpoint(&self, entry: &EndpointEntry) -> Result<bool>;
    async fn delete_endpoint(&self, id: EndpointId) -> Result<bool>;

    // Security map operations
    async fn load_security_maps(&self) -> Result<Vec<SecurityMapEntry>>;
    async fn store_security_map(&self, entry: &SecurityMapEntry) -> Result<TupleId>;
    async fn update_security_map(&self, entry: &SecurityMapEntry) -> Result<bool>;
    async fn delete_security_map(&self, id: SecurityMapId) -> Result<bool>;

    // Data lifecycle operations
    async fn load_legal_holds(&self) -> Result<Vec<LegalHoldEntry>>;
    async fn store_legal_hold(&self, entry: &LegalHoldEntry) -> Result<TupleId>;
    async fn update_legal_hold(&self, entry: &LegalHoldEntry) -> Result<bool>;
    async fn delete_legal_hold(&self, id: u32) -> Result<bool>;

    async fn load_retention_policies(&self) -> Result<Vec<RetentionPolicyEntry>>;
    async fn store_retention_policy(&self, entry: &RetentionPolicyEntry) -> Result<TupleId>;
    /// Replaces all policy rows for a table with the given set (delete + insert).
    async fn replace_retention_policies(
        &self,
        table_id: u32,
        entries: &[RetentionPolicyEntry],
    ) -> Result<()>;

    async fn load_retention_jobs(&self) -> Result<Vec<RetentionJobEntry>>;
    async fn store_retention_job(&self, entry: &RetentionJobEntry) -> Result<TupleId>;

    async fn load_compliance_log(&self) -> Result<Vec<ComplianceLogEntry>>;
    async fn store_compliance_log(&self, entry: &ComplianceLogEntry) -> Result<TupleId>;

    // Bootstrap and recovery
    async fn is_bootstrapped(&self) -> Result<bool>;
    async fn bootstrap(&self) -> Result<()>;

    /// Initializes any cached counters or in-memory structures the storage
    /// implementation needs after open. Called unconditionally by
    /// Catalog::new, including on already-bootstrapped storage, so reopens
    /// after a crash reseed heap page counts from disk.
    async fn init(&self) -> Result<()> {
        Ok(())
    }

    /// Flushes every dirty page held by the storage backend to durable disk
    /// storage. This is the data-side half of the catalog checkpoint
    /// barrier: after this returns, every committed catalog write whose
    /// WAL LSN is at or below the corresponding wal.flushed_lsn() is
    /// guaranteed reflected on disk, so reopen recovery can be skipped for
    /// those records.
    async fn flush_all_dirty(&self) -> Result<()> {
        Ok(())
    }

    // File ID allocation for user tables and indexes
    fn next_heap_file_id(&self) -> (u32, u32);
    fn next_index_file_id(&self) -> u32;
}

/// Catalog storage backed by heap files (self-hosting).
/// Each catalog entity type has its own heap file (system table).
pub struct HeapCatalogStorage {
    databases_heap: HeapFile,
    schemas_heap: HeapFile,
    tables_heap: HeapFile,
    columns_heap: HeapFile,
    indexes_heap: HeapFile,
    streaming_jobs_heap: HeapFile,
    external_sources_heap: HeapFile,
    external_sinks_heap: HeapFile,
    publications_heap: HeapFile,
    publication_tables_heap: HeapFile,
    subscriptions_heap: HeapFile,
    endpoints_heap: HeapFile,
    security_maps_heap: HeapFile,
    legal_holds_heap: HeapFile,
    retention_policies_heap: HeapFile,
    retention_jobs_heap: HeapFile,
    compliance_log_heap: HeapFile,
    sequences_heap: HeapFile,
    views_heap: HeapFile,
    mviews_heap: HeapFile,
    functions_heap: HeapFile,
    comments_heap: HeapFile,
    aggregates_heap: HeapFile,
    procedures_heap: HeapFile,
    schedules_heap: HeapFile,
    triggers_heap: HeapFile,
    pipelines_heap: HeapFile,
    event_handlers_heap: HeapFile,
    version_tags_heap: HeapFile,
    next_heap_file: AtomicU32,
    next_index_file: AtomicU32,
    init_done: std::sync::atomic::AtomicBool,
    /// Shared disk manager. Kept here so flush_all_dirty can take a single
    /// pass over the buffer pool and write directly to the on-disk files
    /// without going through each HeapFile's own filtered flush.
    disk: Arc<DiskManager>,
    pool: Arc<BufferPool>,
}

impl HeapCatalogStorage {
    /// Creates a new HeapCatalogStorage with system table heap files.
    pub fn new(disk: Arc<DiskManager>, pool: Arc<BufferPool>) -> Result<Self> {
        let databases_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: DATABASES_HEAP_FILE_ID,
                fsm_file_id: DATABASES_FSM_FILE_ID,
            },
        )?;
        let schemas_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: SCHEMAS_HEAP_FILE_ID,
                fsm_file_id: SCHEMAS_FSM_FILE_ID,
            },
        )?;
        let tables_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: TABLES_HEAP_FILE_ID,
                fsm_file_id: TABLES_FSM_FILE_ID,
            },
        )?;
        let columns_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: COLUMNS_HEAP_FILE_ID,
                fsm_file_id: COLUMNS_FSM_FILE_ID,
            },
        )?;
        let indexes_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: INDEXES_HEAP_FILE_ID,
                fsm_file_id: INDEXES_FSM_FILE_ID,
            },
        )?;
        let streaming_jobs_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: STREAMING_JOBS_HEAP_FILE_ID,
                fsm_file_id: STREAMING_JOBS_FSM_FILE_ID,
            },
        )?;
        let external_sources_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: EXTERNAL_SOURCES_HEAP_FILE_ID,
                fsm_file_id: EXTERNAL_SOURCES_FSM_FILE_ID,
            },
        )?;
        let external_sinks_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: EXTERNAL_SINKS_HEAP_FILE_ID,
                fsm_file_id: EXTERNAL_SINKS_FSM_FILE_ID,
            },
        )?;
        let publications_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: PUBLICATIONS_HEAP_FILE_ID,
                fsm_file_id: PUBLICATIONS_FSM_FILE_ID,
            },
        )?;
        let publication_tables_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: PUBLICATION_TABLES_HEAP_FILE_ID,
                fsm_file_id: PUBLICATION_TABLES_FSM_FILE_ID,
            },
        )?;
        let subscriptions_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: SUBSCRIPTIONS_HEAP_FILE_ID,
                fsm_file_id: SUBSCRIPTIONS_FSM_FILE_ID,
            },
        )?;
        let endpoints_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: ENDPOINTS_HEAP_FILE_ID,
                fsm_file_id: ENDPOINTS_FSM_FILE_ID,
            },
        )?;
        let security_maps_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: SECURITY_MAPS_HEAP_FILE_ID,
                fsm_file_id: SECURITY_MAPS_FSM_FILE_ID,
            },
        )?;
        let legal_holds_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: LEGAL_HOLDS_HEAP_FILE_ID,
                fsm_file_id: LEGAL_HOLDS_FSM_FILE_ID,
            },
        )?;
        let retention_policies_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: RETENTION_POLICIES_HEAP_FILE_ID,
                fsm_file_id: RETENTION_POLICIES_FSM_FILE_ID,
            },
        )?;
        let retention_jobs_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: RETENTION_JOBS_HEAP_FILE_ID,
                fsm_file_id: RETENTION_JOBS_FSM_FILE_ID,
            },
        )?;
        let compliance_log_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: COMPLIANCE_LOG_HEAP_FILE_ID,
                fsm_file_id: COMPLIANCE_LOG_FSM_FILE_ID,
            },
        )?;
        let sequences_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: SEQUENCES_HEAP_FILE_ID,
                fsm_file_id: SEQUENCES_FSM_FILE_ID,
            },
        )?;
        let views_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: VIEWS_HEAP_FILE_ID,
                fsm_file_id: VIEWS_FSM_FILE_ID,
            },
        )?;
        let mviews_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: MVIEWS_HEAP_FILE_ID,
                fsm_file_id: MVIEWS_FSM_FILE_ID,
            },
        )?;
        let functions_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: FUNCTIONS_HEAP_FILE_ID,
                fsm_file_id: FUNCTIONS_FSM_FILE_ID,
            },
        )?;
        let comments_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: COMMENTS_HEAP_FILE_ID,
                fsm_file_id: COMMENTS_FSM_FILE_ID,
            },
        )?;
        let aggregates_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: AGGREGATES_HEAP_FILE_ID,
                fsm_file_id: AGGREGATES_FSM_FILE_ID,
            },
        )?;
        let procedures_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: PROCEDURES_HEAP_FILE_ID,
                fsm_file_id: PROCEDURES_FSM_FILE_ID,
            },
        )?;
        let schedules_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: SCHEDULES_HEAP_FILE_ID,
                fsm_file_id: SCHEDULES_FSM_FILE_ID,
            },
        )?;
        let triggers_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: TRIGGERS_HEAP_FILE_ID,
                fsm_file_id: TRIGGERS_FSM_FILE_ID,
            },
        )?;
        let pipelines_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: PIPELINES_HEAP_FILE_ID,
                fsm_file_id: PIPELINES_FSM_FILE_ID,
            },
        )?;
        let event_handlers_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: EVENT_HANDLERS_HEAP_FILE_ID,
                fsm_file_id: EVENT_HANDLERS_FSM_FILE_ID,
            },
        )?;
        let version_tags_heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: VERSION_TAGS_HEAP_FILE_ID,
                fsm_file_id: VERSION_TAGS_FSM_FILE_ID,
            },
        )?;

        Ok(Self {
            databases_heap,
            schemas_heap,
            tables_heap,
            columns_heap,
            indexes_heap,
            streaming_jobs_heap,
            external_sources_heap,
            external_sinks_heap,
            publications_heap,
            publication_tables_heap,
            subscriptions_heap,
            endpoints_heap,
            security_maps_heap,
            legal_holds_heap,
            retention_policies_heap,
            retention_jobs_heap,
            compliance_log_heap,
            sequences_heap,
            views_heap,
            mviews_heap,
            functions_heap,
            comments_heap,
            aggregates_heap,
            procedures_heap,
            schedules_heap,
            triggers_heap,
            pipelines_heap,
            event_handlers_heap,
            version_tags_heap,
            next_heap_file: AtomicU32::new(USER_HEAP_FILE_START),
            next_index_file: AtomicU32::new(USER_INDEX_FILE_START),
            init_done: std::sync::atomic::AtomicBool::new(false),
            disk,
            pool,
        })
    }

    /// Initializes page count caches for all system table heap files.
    /// Runs all init calls concurrently to minimize cold-start latency.
    /// Idempotent: the init_done flag is CAS-set the first time through,
    /// so callers that drive both this method and the catalog's own
    /// init() do not pay the cost twice.
    pub async fn init_cache(&self) -> Result<()> {
        use std::sync::atomic::Ordering;
        if self
            .init_done
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Ok(());
        }
        tokio::try_join!(
            self.databases_heap.init_cache(),
            self.schemas_heap.init_cache(),
            self.tables_heap.init_cache(),
            self.columns_heap.init_cache(),
            self.indexes_heap.init_cache(),
            self.streaming_jobs_heap.init_cache(),
            self.external_sources_heap.init_cache(),
            self.external_sinks_heap.init_cache(),
            self.publications_heap.init_cache(),
            self.publication_tables_heap.init_cache(),
            self.subscriptions_heap.init_cache(),
            self.endpoints_heap.init_cache(),
            self.security_maps_heap.init_cache(),
            self.legal_holds_heap.init_cache(),
            self.retention_policies_heap.init_cache(),
            self.retention_jobs_heap.init_cache(),
            self.compliance_log_heap.init_cache(),
            self.sequences_heap.init_cache(),
            self.views_heap.init_cache(),
            self.mviews_heap.init_cache(),
            self.functions_heap.init_cache(),
            self.comments_heap.init_cache(),
            self.aggregates_heap.init_cache(),
            self.procedures_heap.init_cache(),
            self.schedules_heap.init_cache(),
            self.triggers_heap.init_cache(),
            self.pipelines_heap.init_cache(),
            self.event_handlers_heap.init_cache(),
            self.version_tags_heap.init_cache(),
        )?;
        Ok(())
    }

    /// Recovers file ID counters by scanning existing tables and indexes
    /// to find the maximum file IDs in use.
    pub async fn recover_file_id_counters(&self) -> Result<()> {
        let tables = self.load_tables().await?;
        let mut max_heap = USER_HEAP_FILE_START;
        for t in &tables {
            let candidate = t.fsm_file_id + 1;
            if candidate > max_heap {
                max_heap = candidate;
            }
        }
        // Round up to next even number for heap alignment (heap, fsm pairs)
        if max_heap % 2 != 0 {
            max_heap += 1;
        }
        self.next_heap_file.store(max_heap, Ordering::Relaxed);

        let indexes = self.load_indexes().await?;
        let mut max_idx = USER_INDEX_FILE_START;
        for i in &indexes {
            let candidate = i.index_file_id + 1;
            if candidate > max_idx {
                max_idx = candidate;
            }
        }
        self.next_index_file.store(max_idx, Ordering::Relaxed);

        Ok(())
    }
}

#[async_trait]
impl CatalogStorage for HeapCatalogStorage {
    async fn init(&self) -> Result<()> {
        // Seed cached heap page counts from on-disk file sizes. Without
        // this, a reopened storage observes cached_heap_pages = 0 even when
        // the file holds many pages, and scans return zero tuples until the
        // next insert path bumps the counter.
        self.init_cache().await
    }

    async fn flush_all_dirty(&self) -> Result<()> {
        // Single pass over the buffer pool. The naive variant (one
        // HeapFile::flush() per heap) walked BufferPool::page_table 17
        // times and stamped every dirty page that did not match the
        // current heap's file ids. That is O(heaps * pool_size). One
        // dispatch table over every catalog file id lets us walk the pool
        // exactly once and write directly to the matching file.
        let data_dir = self.disk.data_dir().to_path_buf();
        let allowed: std::collections::HashSet<u32> = [
            DATABASES_HEAP_FILE_ID,
            DATABASES_FSM_FILE_ID,
            SCHEMAS_HEAP_FILE_ID,
            SCHEMAS_FSM_FILE_ID,
            TABLES_HEAP_FILE_ID,
            TABLES_FSM_FILE_ID,
            COLUMNS_HEAP_FILE_ID,
            COLUMNS_FSM_FILE_ID,
            INDEXES_HEAP_FILE_ID,
            INDEXES_FSM_FILE_ID,
            STREAMING_JOBS_HEAP_FILE_ID,
            STREAMING_JOBS_FSM_FILE_ID,
            EXTERNAL_SOURCES_HEAP_FILE_ID,
            EXTERNAL_SOURCES_FSM_FILE_ID,
            EXTERNAL_SINKS_HEAP_FILE_ID,
            EXTERNAL_SINKS_FSM_FILE_ID,
            PUBLICATIONS_HEAP_FILE_ID,
            PUBLICATIONS_FSM_FILE_ID,
            PUBLICATION_TABLES_HEAP_FILE_ID,
            PUBLICATION_TABLES_FSM_FILE_ID,
            SUBSCRIPTIONS_HEAP_FILE_ID,
            SUBSCRIPTIONS_FSM_FILE_ID,
            ENDPOINTS_HEAP_FILE_ID,
            ENDPOINTS_FSM_FILE_ID,
            SECURITY_MAPS_HEAP_FILE_ID,
            SECURITY_MAPS_FSM_FILE_ID,
            LEGAL_HOLDS_HEAP_FILE_ID,
            LEGAL_HOLDS_FSM_FILE_ID,
            RETENTION_POLICIES_HEAP_FILE_ID,
            RETENTION_POLICIES_FSM_FILE_ID,
            RETENTION_JOBS_HEAP_FILE_ID,
            RETENTION_JOBS_FSM_FILE_ID,
            COMPLIANCE_LOG_HEAP_FILE_ID,
            COMPLIANCE_LOG_FSM_FILE_ID,
            SEQUENCES_HEAP_FILE_ID,
            SEQUENCES_FSM_FILE_ID,
            VIEWS_HEAP_FILE_ID,
            VIEWS_FSM_FILE_ID,
            MVIEWS_HEAP_FILE_ID,
            MVIEWS_FSM_FILE_ID,
            FUNCTIONS_HEAP_FILE_ID,
            FUNCTIONS_FSM_FILE_ID,
            COMMENTS_HEAP_FILE_ID,
            COMMENTS_FSM_FILE_ID,
            AGGREGATES_HEAP_FILE_ID,
            AGGREGATES_FSM_FILE_ID,
            PROCEDURES_HEAP_FILE_ID,
            PROCEDURES_FSM_FILE_ID,
            SCHEDULES_HEAP_FILE_ID,
            SCHEDULES_FSM_FILE_ID,
            TRIGGERS_HEAP_FILE_ID,
            TRIGGERS_FSM_FILE_ID,
            PIPELINES_HEAP_FILE_ID,
            PIPELINES_FSM_FILE_ID,
            EVENT_HANDLERS_HEAP_FILE_ID,
            EVENT_HANDLERS_FSM_FILE_ID,
            VERSION_TAGS_HEAP_FILE_ID,
            VERSION_TAGS_FSM_FILE_ID,
        ]
        .into_iter()
        .collect();

        self.pool.flush_all(|page_id, data| {
            if !allowed.contains(&page_id.file_id) {
                return Ok(());
            }
            let path = data_dir.join(format!("{:08}.dat", page_id.file_id));
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .open(&path)
                .map_err(|e| {
                    ZyronError::IoError(format!("flush open {}: {}", path.display(), e))
                })?;
            let offset = page_id.page_num * (PAGE_SIZE as u64);
            std::io::Seek::seek(&mut file, std::io::SeekFrom::Start(offset))
                .map_err(|e| ZyronError::IoError(format!("flush seek: {}", e)))?;
            std::io::Write::write_all(&mut file, data)
                .map_err(|e| ZyronError::IoError(format!("flush write: {}", e)))?;
            Ok(())
        })?;
        Ok(())
    }

    async fn load_databases(&self) -> Result<Vec<DatabaseEntry>> {
        scan_decode(&self.databases_heap, "database", DatabaseEntry::from_bytes)
    }

    async fn store_database(&self, entry: &DatabaseEntry) -> Result<TupleId> {
        let bytes = entry.to_bytes();
        let tuple = Tuple::new(bytes, 0);
        let ids = self.databases_heap.insert_batch(&[tuple]).await?;
        Ok(ids[0])
    }

    async fn delete_database(&self, id: DatabaseId) -> Result<bool> {
        let mut target = None;
        let guard = self.databases_heap.scan()?;
        guard.for_each(|tid, view| {
            if let Ok(entry) = DatabaseEntry::from_bytes(view.data) {
                if entry.id == id {
                    target = Some(tid);
                }
            }
        });
        match target {
            Some(tid) => self.databases_heap.delete(tid).await,
            None => Ok(false),
        }
    }

    async fn load_schemas(&self) -> Result<Vec<SchemaEntry>> {
        scan_decode(&self.schemas_heap, "schema", SchemaEntry::from_bytes)
    }

    async fn store_schema(&self, entry: &SchemaEntry) -> Result<TupleId> {
        let bytes = entry.to_bytes();
        let tuple = Tuple::new(bytes, 0);
        let ids = self.schemas_heap.insert_batch(&[tuple]).await?;
        Ok(ids[0])
    }

    async fn delete_schema(&self, id: SchemaId) -> Result<bool> {
        let mut target = None;
        let guard = self.schemas_heap.scan()?;
        guard.for_each(|tid, view| {
            if let Ok(entry) = SchemaEntry::from_bytes(view.data) {
                if entry.id == id {
                    target = Some(tid);
                }
            }
        });
        match target {
            Some(tid) => self.schemas_heap.delete(tid).await,
            None => Ok(false),
        }
    }

    async fn load_tables(&self) -> Result<Vec<TableEntry>> {
        scan_decode(&self.tables_heap, "table", TableEntry::from_bytes)
    }

    async fn load_table_by_name(
        &self,
        schema_id: SchemaId,
        name: &str,
    ) -> Result<Option<TableEntry>> {
        let guard = self.tables_heap.scan()?;
        let mut found = None;
        let mut decode_err: Option<ZyronError> = None;
        guard.try_for_each(|_tid, view| match TableEntry::from_bytes(view.data) {
            Ok(entry) => {
                if entry.schema_id == schema_id && entry.name == name {
                    found = Some(entry);
                    return false; // stop scanning at the first match
                }
                true
            }
            Err(e) => {
                decode_err = Some(ZyronError::CatalogCorrupted(format!(
                    "table catalog tuple failed to decode: {}",
                    e
                )));
                false
            }
        });
        if let Some(e) = decode_err {
            return Err(e);
        }
        Ok(found)
    }

    async fn store_table(&self, entry: &TableEntry) -> Result<TupleId> {
        let bytes = entry.to_bytes();
        let tuple = Tuple::new(bytes, 0);
        let ids = self.tables_heap.insert_batch(&[tuple]).await?;
        Ok(ids[0])
    }

    async fn delete_table(&self, id: TableId) -> Result<bool> {
        let mut target = None;
        let guard = self.tables_heap.scan()?;
        guard.for_each(|tid, view| {
            if let Ok(entry) = TableEntry::from_bytes(view.data) {
                if entry.id == id {
                    target = Some(tid);
                }
            }
        });
        match target {
            Some(tid) => self.tables_heap.delete(tid).await,
            None => Ok(false),
        }
    }

    async fn load_columns(&self, table_id: TableId) -> Result<Vec<ColumnEntry>> {
        let mut entries = scan_decode(&self.columns_heap, "column", ColumnEntry::from_bytes)?;
        entries.retain(|c| c.table_id == table_id);
        entries.sort_by_key(|c| c.ordinal);
        Ok(entries)
    }

    async fn store_columns(&self, columns: &[ColumnEntry]) -> Result<Vec<TupleId>> {
        if columns.is_empty() {
            return Ok(Vec::new());
        }
        let tuples: Vec<Tuple> = columns
            .iter()
            .map(|c| Tuple::new(c.to_bytes(), 0))
            .collect();
        self.columns_heap.insert_batch(&tuples).await
    }

    async fn delete_columns(&self, table_id: TableId) -> Result<usize> {
        let mut targets = Vec::new();
        let guard = self.columns_heap.scan()?;
        guard.for_each(|tid, view| {
            if let Ok(entry) = ColumnEntry::from_bytes(view.data) {
                if entry.table_id == table_id {
                    targets.push(tid);
                }
            }
        });
        if targets.is_empty() {
            return Ok(0);
        }
        self.columns_heap.delete_batch(&targets).await
    }

    async fn load_indexes(&self) -> Result<Vec<IndexEntry>> {
        scan_decode(&self.indexes_heap, "index", IndexEntry::from_bytes)
    }

    async fn store_index(&self, entry: &IndexEntry) -> Result<TupleId> {
        let bytes = entry.to_bytes();
        let tuple = Tuple::new(bytes, 0);
        let ids = self.indexes_heap.insert_batch(&[tuple]).await?;
        Ok(ids[0])
    }

    async fn delete_index(&self, id: IndexId) -> Result<bool> {
        let mut target = None;
        let guard = self.indexes_heap.scan()?;
        guard.for_each(|tid, view| {
            if let Ok(entry) = IndexEntry::from_bytes(view.data) {
                if entry.id == id {
                    target = Some(tid);
                }
            }
        });
        match target {
            Some(tid) => self.indexes_heap.delete(tid).await,
            None => Ok(false),
        }
    }

    async fn load_streaming_jobs(&self) -> Result<Vec<StreamingJobEntry>> {
        scan_decode(
            &self.streaming_jobs_heap,
            "streaming job",
            StreamingJobEntry::from_bytes,
        )
    }

    async fn store_streaming_job(&self, entry: &StreamingJobEntry) -> Result<TupleId> {
        let bytes = entry.to_bytes();
        let tuple = Tuple::new(bytes, 0);
        let ids = self.streaming_jobs_heap.insert_batch(&[tuple]).await?;
        Ok(ids[0])
    }

    async fn update_streaming_job(&self, entry: &StreamingJobEntry) -> Result<bool> {
        let mut target = None;
        let guard = self.streaming_jobs_heap.scan()?;
        guard.for_each(|tid, view| {
            if let Ok(existing) = StreamingJobEntry::from_bytes(view.data) {
                if existing.id == entry.id {
                    target = Some(tid);
                }
            }
        });
        drop(guard);
        match target {
            Some(tid) => {
                self.streaming_jobs_heap.delete(tid).await?;
                let bytes = entry.to_bytes();
                let tuple = Tuple::new(bytes, 0);
                self.streaming_jobs_heap.insert_batch(&[tuple]).await?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    async fn delete_streaming_job(&self, id: StreamingJobId) -> Result<bool> {
        let mut target = None;
        let guard = self.streaming_jobs_heap.scan()?;
        guard.for_each(|tid, view| {
            if let Ok(entry) = StreamingJobEntry::from_bytes(view.data) {
                if entry.id == id {
                    target = Some(tid);
                }
            }
        });
        match target {
            Some(tid) => self.streaming_jobs_heap.delete(tid).await,
            None => Ok(false),
        }
    }

    async fn load_sequences(&self) -> Result<Vec<SequenceEntry>> {
        scan_decode(&self.sequences_heap, "sequence", SequenceEntry::from_bytes)
    }

    async fn store_sequence(&self, entry: &SequenceEntry) -> Result<TupleId> {
        let tuple = Tuple::new(entry.to_bytes(), 0);
        let ids = self.sequences_heap.insert_batch(&[tuple]).await?;
        Ok(ids[0])
    }

    async fn update_sequence(&self, entry: &SequenceEntry) -> Result<bool> {
        let mut target = None;
        let guard = self.sequences_heap.scan()?;
        guard.for_each(|tid, view| {
            if let Ok(existing) = SequenceEntry::from_bytes(view.data) {
                if existing.id == entry.id {
                    target = Some(tid);
                }
            }
        });
        drop(guard);
        match target {
            Some(tid) => {
                self.sequences_heap.delete(tid).await?;
                let tuple = Tuple::new(entry.to_bytes(), 0);
                self.sequences_heap.insert_batch(&[tuple]).await?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    async fn delete_sequence(&self, id: u32) -> Result<bool> {
        let mut target = None;
        let guard = self.sequences_heap.scan()?;
        guard.for_each(|tid, view| {
            if let Ok(entry) = SequenceEntry::from_bytes(view.data) {
                if entry.id == id {
                    target = Some(tid);
                }
            }
        });
        match target {
            Some(tid) => self.sequences_heap.delete(tid).await,
            None => Ok(false),
        }
    }

    async fn load_views(&self) -> Result<Vec<ViewEntry>> {
        scan_decode(&self.views_heap, "view", ViewEntry::from_bytes)
    }

    async fn store_view(&self, entry: &ViewEntry) -> Result<TupleId> {
        let tuple = Tuple::new(entry.to_bytes(), 0);
        let ids = self.views_heap.insert_batch(&[tuple]).await?;
        Ok(ids[0])
    }

    async fn update_view(&self, entry: &ViewEntry) -> Result<bool> {
        let mut target = None;
        let guard = self.views_heap.scan()?;
        guard.for_each(|tid, row| {
            if let Ok(existing) = ViewEntry::from_bytes(row.data) {
                if existing.id == entry.id {
                    target = Some(tid);
                }
            }
        });
        drop(guard);
        match target {
            Some(tid) => {
                self.views_heap.delete(tid).await?;
                let tuple = Tuple::new(entry.to_bytes(), 0);
                self.views_heap.insert_batch(&[tuple]).await?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    async fn delete_view(&self, id: u32) -> Result<bool> {
        let mut target = None;
        let guard = self.views_heap.scan()?;
        guard.for_each(|tid, row| {
            if let Ok(entry) = ViewEntry::from_bytes(row.data) {
                if entry.id == id {
                    target = Some(tid);
                }
            }
        });
        match target {
            Some(tid) => self.views_heap.delete(tid).await,
            None => Ok(false),
        }
    }

    async fn load_mviews(&self) -> Result<Vec<MaterializedViewEntry>> {
        scan_decode(
            &self.mviews_heap,
            "materialized view",
            MaterializedViewEntry::from_bytes,
        )
    }

    async fn store_mview(&self, entry: &MaterializedViewEntry) -> Result<TupleId> {
        let tuple = Tuple::new(entry.to_bytes(), 0);
        let ids = self.mviews_heap.insert_batch(&[tuple]).await?;
        Ok(ids[0])
    }

    async fn update_mview(&self, entry: &MaterializedViewEntry) -> Result<bool> {
        let mut target = None;
        let guard = self.mviews_heap.scan()?;
        guard.for_each(|tid, row| {
            if let Ok(existing) = MaterializedViewEntry::from_bytes(row.data) {
                if existing.id == entry.id {
                    target = Some(tid);
                }
            }
        });
        drop(guard);
        match target {
            Some(tid) => {
                self.mviews_heap.delete(tid).await?;
                let tuple = Tuple::new(entry.to_bytes(), 0);
                self.mviews_heap.insert_batch(&[tuple]).await?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    async fn delete_mview(&self, id: u32) -> Result<bool> {
        let mut target = None;
        let guard = self.mviews_heap.scan()?;
        guard.for_each(|tid, row| {
            if let Ok(entry) = MaterializedViewEntry::from_bytes(row.data) {
                if entry.id == id {
                    target = Some(tid);
                }
            }
        });
        match target {
            Some(tid) => self.mviews_heap.delete(tid).await,
            None => Ok(false),
        }
    }

    async fn load_functions(&self) -> Result<Vec<FunctionEntry>> {
        scan_decode(&self.functions_heap, "function", FunctionEntry::from_bytes)
    }

    async fn store_function(&self, entry: &FunctionEntry) -> Result<TupleId> {
        let tuple = Tuple::new(entry.to_bytes(), 0);
        let ids = self.functions_heap.insert_batch(&[tuple]).await?;
        Ok(ids[0])
    }

    async fn delete_function(&self, id: u32) -> Result<bool> {
        let mut target = None;
        let guard = self.functions_heap.scan()?;
        guard.for_each(|tid, row| {
            if let Ok(entry) = FunctionEntry::from_bytes(row.data) {
                if entry.id == id {
                    target = Some(tid);
                }
            }
        });
        match target {
            Some(tid) => self.functions_heap.delete(tid).await,
            None => Ok(false),
        }
    }

    async fn load_comments(&self) -> Result<Vec<CommentEntry>> {
        scan_decode(&self.comments_heap, "comment", CommentEntry::from_bytes)
    }

    async fn store_comment(&self, entry: &CommentEntry) -> Result<TupleId> {
        let tuple = Tuple::new(entry.to_bytes(), 0);
        let ids = self.comments_heap.insert_batch(&[tuple]).await?;
        Ok(ids[0])
    }

    async fn delete_comment(&self, id: u32) -> Result<bool> {
        let mut target = None;
        let guard = self.comments_heap.scan()?;
        guard.for_each(|tid, row| {
            if let Ok(entry) = CommentEntry::from_bytes(row.data) {
                if entry.id == id {
                    target = Some(tid);
                }
            }
        });
        match target {
            Some(tid) => self.comments_heap.delete(tid).await,
            None => Ok(false),
        }
    }

    async fn load_aggregates(&self) -> Result<Vec<AggregateEntry>> {
        scan_decode(
            &self.aggregates_heap,
            "aggregate",
            AggregateEntry::from_bytes,
        )
    }

    async fn store_aggregate(&self, entry: &AggregateEntry) -> Result<TupleId> {
        let tuple = Tuple::new(entry.to_bytes(), 0);
        let ids = self.aggregates_heap.insert_batch(&[tuple]).await?;
        Ok(ids[0])
    }

    async fn delete_aggregate(&self, id: u32) -> Result<bool> {
        let mut target = None;
        let guard = self.aggregates_heap.scan()?;
        guard.for_each(|tid, row| {
            if let Ok(entry) = AggregateEntry::from_bytes(row.data) {
                if entry.id == id {
                    target = Some(tid);
                }
            }
        });
        match target {
            Some(tid) => self.aggregates_heap.delete(tid).await,
            None => Ok(false),
        }
    }

    async fn load_procedures(&self) -> Result<Vec<ProcedureEntry>> {
        scan_decode(
            &self.procedures_heap,
            "procedure",
            ProcedureEntry::from_bytes,
        )
    }

    async fn store_procedure(&self, entry: &ProcedureEntry) -> Result<TupleId> {
        let tuple = Tuple::new(entry.to_bytes(), 0);
        let ids = self.procedures_heap.insert_batch(&[tuple]).await?;
        Ok(ids[0])
    }

    async fn delete_procedure(&self, id: u32) -> Result<bool> {
        let mut target = None;
        let guard = self.procedures_heap.scan()?;
        guard.for_each(|tid, row| {
            if let Ok(entry) = ProcedureEntry::from_bytes(row.data) {
                if entry.id == id {
                    target = Some(tid);
                }
            }
        });
        match target {
            Some(tid) => self.procedures_heap.delete(tid).await,
            None => Ok(false),
        }
    }

    async fn load_schedules(&self) -> Result<Vec<ScheduleEntry>> {
        scan_decode(&self.schedules_heap, "schedule", ScheduleEntry::from_bytes)
    }

    async fn store_schedule(&self, entry: &ScheduleEntry) -> Result<TupleId> {
        let tuple = Tuple::new(entry.to_bytes(), 0);
        let ids = self.schedules_heap.insert_batch(&[tuple]).await?;
        Ok(ids[0])
    }

    async fn delete_schedule(&self, id: u32) -> Result<bool> {
        let mut target = None;
        let guard = self.schedules_heap.scan()?;
        guard.for_each(|tid, row| {
            if let Ok(entry) = ScheduleEntry::from_bytes(row.data) {
                if entry.id == id {
                    target = Some(tid);
                }
            }
        });
        match target {
            Some(tid) => self.schedules_heap.delete(tid).await,
            None => Ok(false),
        }
    }

    async fn load_triggers(&self) -> Result<Vec<TriggerEntry>> {
        scan_decode(&self.triggers_heap, "trigger", TriggerEntry::from_bytes)
    }

    async fn store_trigger(&self, entry: &TriggerEntry) -> Result<TupleId> {
        let tuple = Tuple::new(entry.to_bytes(), 0);
        let ids = self.triggers_heap.insert_batch(&[tuple]).await?;
        Ok(ids[0])
    }

    async fn delete_trigger(&self, id: u32) -> Result<bool> {
        let mut target = None;
        let guard = self.triggers_heap.scan()?;
        guard.for_each(|tid, row| {
            if let Ok(entry) = TriggerEntry::from_bytes(row.data) {
                if entry.id == id {
                    target = Some(tid);
                }
            }
        });
        match target {
            Some(tid) => self.triggers_heap.delete(tid).await,
            None => Ok(false),
        }
    }

    async fn load_pipelines(&self) -> Result<Vec<PipelineEntry>> {
        scan_decode(&self.pipelines_heap, "pipeline", PipelineEntry::from_bytes)
    }

    async fn store_pipeline(&self, entry: &PipelineEntry) -> Result<TupleId> {
        let tuple = Tuple::new(entry.to_bytes(), 0);
        let ids = self.pipelines_heap.insert_batch(&[tuple]).await?;
        Ok(ids[0])
    }

    async fn delete_pipeline(&self, id: u32) -> Result<bool> {
        let mut target = None;
        let guard = self.pipelines_heap.scan()?;
        guard.for_each(|tid, row| {
            if let Ok(entry) = PipelineEntry::from_bytes(row.data) {
                if entry.id == id {
                    target = Some(tid);
                }
            }
        });
        match target {
            Some(tid) => self.pipelines_heap.delete(tid).await,
            None => Ok(false),
        }
    }

    async fn load_event_handlers(&self) -> Result<Vec<EventHandlerEntry>> {
        scan_decode(
            &self.event_handlers_heap,
            "event handler",
            EventHandlerEntry::from_bytes,
        )
    }

    async fn store_event_handler(&self, entry: &EventHandlerEntry) -> Result<TupleId> {
        let tuple = Tuple::new(entry.to_bytes(), 0);
        let ids = self.event_handlers_heap.insert_batch(&[tuple]).await?;
        Ok(ids[0])
    }

    async fn delete_event_handler(&self, id: u32) -> Result<bool> {
        let mut target = None;
        let guard = self.event_handlers_heap.scan()?;
        guard.for_each(|tid, row| {
            if let Ok(entry) = EventHandlerEntry::from_bytes(row.data) {
                if entry.id == id {
                    target = Some(tid);
                }
            }
        });
        match target {
            Some(tid) => self.event_handlers_heap.delete(tid).await,
            None => Ok(false),
        }
    }

    async fn load_version_tags(&self) -> Result<Vec<VersionTagEntry>> {
        scan_decode(
            &self.version_tags_heap,
            "version tag",
            VersionTagEntry::from_bytes,
        )
    }

    async fn store_version_tag(&self, entry: &VersionTagEntry) -> Result<TupleId> {
        let tuple = Tuple::new(entry.to_bytes(), 0);
        let ids = self.version_tags_heap.insert_batch(&[tuple]).await?;
        Ok(ids[0])
    }

    async fn delete_version_tag(&self, id: u32) -> Result<bool> {
        let mut target = None;
        let guard = self.version_tags_heap.scan()?;
        guard.for_each(|tid, row| {
            if let Ok(entry) = VersionTagEntry::from_bytes(row.data) {
                if entry.id == id {
                    target = Some(tid);
                }
            }
        });
        match target {
            Some(tid) => self.version_tags_heap.delete(tid).await,
            None => Ok(false),
        }
    }

    async fn load_external_sources(&self) -> Result<Vec<ExternalSourceEntry>> {
        scan_decode(
            &self.external_sources_heap,
            "external source",
            ExternalSourceEntry::from_bytes,
        )
    }

    async fn store_external_source(&self, entry: &ExternalSourceEntry) -> Result<TupleId> {
        let bytes = entry.to_bytes();
        let tuple = Tuple::new(bytes, 0);
        let ids = self.external_sources_heap.insert_batch(&[tuple]).await?;
        Ok(ids[0])
    }

    async fn update_external_source(&self, entry: &ExternalSourceEntry) -> Result<bool> {
        let mut target = None;
        let guard = self.external_sources_heap.scan()?;
        guard.for_each(|tid, view| {
            if let Ok(existing) = ExternalSourceEntry::from_bytes(view.data) {
                if existing.id == entry.id {
                    target = Some(tid);
                }
            }
        });
        drop(guard);
        match target {
            Some(tid) => {
                self.external_sources_heap.delete(tid).await?;
                let bytes = entry.to_bytes();
                let tuple = Tuple::new(bytes, 0);
                self.external_sources_heap.insert_batch(&[tuple]).await?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    async fn delete_external_source(&self, id: ExternalSourceId) -> Result<bool> {
        let mut target = None;
        let guard = self.external_sources_heap.scan()?;
        guard.for_each(|tid, view| {
            if let Ok(entry) = ExternalSourceEntry::from_bytes(view.data) {
                if entry.id == id {
                    target = Some(tid);
                }
            }
        });
        match target {
            Some(tid) => self.external_sources_heap.delete(tid).await,
            None => Ok(false),
        }
    }

    async fn load_external_sinks(&self) -> Result<Vec<ExternalSinkEntry>> {
        scan_decode(
            &self.external_sinks_heap,
            "external sink",
            ExternalSinkEntry::from_bytes,
        )
    }

    async fn store_external_sink(&self, entry: &ExternalSinkEntry) -> Result<TupleId> {
        let bytes = entry.to_bytes();
        let tuple = Tuple::new(bytes, 0);
        let ids = self.external_sinks_heap.insert_batch(&[tuple]).await?;
        Ok(ids[0])
    }

    async fn update_external_sink(&self, entry: &ExternalSinkEntry) -> Result<bool> {
        let mut target = None;
        let guard = self.external_sinks_heap.scan()?;
        guard.for_each(|tid, view| {
            if let Ok(existing) = ExternalSinkEntry::from_bytes(view.data) {
                if existing.id == entry.id {
                    target = Some(tid);
                }
            }
        });
        drop(guard);
        match target {
            Some(tid) => {
                self.external_sinks_heap.delete(tid).await?;
                let bytes = entry.to_bytes();
                let tuple = Tuple::new(bytes, 0);
                self.external_sinks_heap.insert_batch(&[tuple]).await?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    async fn delete_external_sink(&self, id: ExternalSinkId) -> Result<bool> {
        let mut target = None;
        let guard = self.external_sinks_heap.scan()?;
        guard.for_each(|tid, view| {
            if let Ok(entry) = ExternalSinkEntry::from_bytes(view.data) {
                if entry.id == id {
                    target = Some(tid);
                }
            }
        });
        match target {
            Some(tid) => self.external_sinks_heap.delete(tid).await,
            None => Ok(false),
        }
    }

    async fn load_publications(&self) -> Result<Vec<PublicationEntry>> {
        scan_decode(
            &self.publications_heap,
            "publication",
            PublicationEntry::from_bytes,
        )
    }

    async fn store_publication(&self, entry: &PublicationEntry) -> Result<TupleId> {
        let bytes = entry.to_bytes();
        let tuple = Tuple::new(bytes, 0);
        let ids = self.publications_heap.insert_batch(&[tuple]).await?;
        Ok(ids[0])
    }

    async fn update_publication(&self, entry: &PublicationEntry) -> Result<bool> {
        let mut target = None;
        let guard = self.publications_heap.scan()?;
        guard.for_each(|tid, view| {
            if let Ok(existing) = PublicationEntry::from_bytes(view.data) {
                if existing.id == entry.id {
                    target = Some(tid);
                }
            }
        });
        drop(guard);
        match target {
            Some(tid) => {
                self.publications_heap.delete(tid).await?;
                let bytes = entry.to_bytes();
                let tuple = Tuple::new(bytes, 0);
                self.publications_heap.insert_batch(&[tuple]).await?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    async fn delete_publication(&self, id: PublicationId) -> Result<bool> {
        let mut target = None;
        let guard = self.publications_heap.scan()?;
        guard.for_each(|tid, view| {
            if let Ok(entry) = PublicationEntry::from_bytes(view.data) {
                if entry.id == id {
                    target = Some(tid);
                }
            }
        });
        match target {
            Some(tid) => self.publications_heap.delete(tid).await,
            None => Ok(false),
        }
    }

    async fn load_publication_tables(&self) -> Result<Vec<PublicationTableEntry>> {
        scan_decode(
            &self.publication_tables_heap,
            "publication table",
            PublicationTableEntry::from_bytes,
        )
    }

    async fn store_publication_table(&self, entry: &PublicationTableEntry) -> Result<TupleId> {
        let bytes = entry.to_bytes();
        let tuple = Tuple::new(bytes, 0);
        let ids = self.publication_tables_heap.insert_batch(&[tuple]).await?;
        Ok(ids[0])
    }

    async fn update_publication_table(&self, entry: &PublicationTableEntry) -> Result<bool> {
        let mut target = None;
        let guard = self.publication_tables_heap.scan()?;
        guard.for_each(|tid, view| {
            if let Ok(existing) = PublicationTableEntry::from_bytes(view.data) {
                if existing.id == entry.id {
                    target = Some(tid);
                }
            }
        });
        drop(guard);
        match target {
            Some(tid) => {
                self.publication_tables_heap.delete(tid).await?;
                let bytes = entry.to_bytes();
                let tuple = Tuple::new(bytes, 0);
                self.publication_tables_heap.insert_batch(&[tuple]).await?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    async fn delete_publication_table(
        &self,
        publication_id: PublicationId,
        table_id: TableId,
    ) -> Result<bool> {
        let mut target = None;
        let guard = self.publication_tables_heap.scan()?;
        guard.for_each(|tid, view| {
            if let Ok(entry) = PublicationTableEntry::from_bytes(view.data) {
                if entry.publication_id == publication_id && entry.table_id == table_id {
                    target = Some(tid);
                }
            }
        });
        match target {
            Some(tid) => self.publication_tables_heap.delete(tid).await,
            None => Ok(false),
        }
    }

    async fn load_subscriptions(&self) -> Result<Vec<SubscriptionEntry>> {
        scan_decode(
            &self.subscriptions_heap,
            "subscription",
            SubscriptionEntry::from_bytes,
        )
    }

    async fn store_subscription(&self, entry: &SubscriptionEntry) -> Result<TupleId> {
        let bytes = entry.to_bytes();
        let tuple = Tuple::new(bytes, 0);
        let ids = self.subscriptions_heap.insert_batch(&[tuple]).await?;
        Ok(ids[0])
    }

    async fn update_subscription(&self, entry: &SubscriptionEntry) -> Result<bool> {
        let mut target = None;
        let guard = self.subscriptions_heap.scan()?;
        guard.for_each(|tid, view| {
            if let Ok(existing) = SubscriptionEntry::from_bytes(view.data) {
                if existing.id == entry.id {
                    target = Some(tid);
                }
            }
        });
        drop(guard);
        match target {
            Some(tid) => {
                self.subscriptions_heap.delete(tid).await?;
                let bytes = entry.to_bytes();
                let tuple = Tuple::new(bytes, 0);
                self.subscriptions_heap.insert_batch(&[tuple]).await?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    async fn delete_subscription(&self, id: SubscriptionId) -> Result<bool> {
        let mut target = None;
        let guard = self.subscriptions_heap.scan()?;
        guard.for_each(|tid, view| {
            if let Ok(entry) = SubscriptionEntry::from_bytes(view.data) {
                if entry.id == id {
                    target = Some(tid);
                }
            }
        });
        match target {
            Some(tid) => self.subscriptions_heap.delete(tid).await,
            None => Ok(false),
        }
    }

    async fn load_endpoints(&self) -> Result<Vec<EndpointEntry>> {
        scan_decode(&self.endpoints_heap, "endpoint", EndpointEntry::from_bytes)
    }

    async fn store_endpoint(&self, entry: &EndpointEntry) -> Result<TupleId> {
        let bytes = entry.to_bytes();
        let tuple = Tuple::new(bytes, 0);
        let ids = self.endpoints_heap.insert_batch(&[tuple]).await?;
        Ok(ids[0])
    }

    async fn update_endpoint(&self, entry: &EndpointEntry) -> Result<bool> {
        let mut target = None;
        let guard = self.endpoints_heap.scan()?;
        guard.for_each(|tid, view| {
            if let Ok(existing) = EndpointEntry::from_bytes(view.data) {
                if existing.id == entry.id {
                    target = Some(tid);
                }
            }
        });
        drop(guard);
        match target {
            Some(tid) => {
                self.endpoints_heap.delete(tid).await?;
                let bytes = entry.to_bytes();
                let tuple = Tuple::new(bytes, 0);
                self.endpoints_heap.insert_batch(&[tuple]).await?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    async fn delete_endpoint(&self, id: EndpointId) -> Result<bool> {
        let mut target = None;
        let guard = self.endpoints_heap.scan()?;
        guard.for_each(|tid, view| {
            if let Ok(entry) = EndpointEntry::from_bytes(view.data) {
                if entry.id == id {
                    target = Some(tid);
                }
            }
        });
        match target {
            Some(tid) => self.endpoints_heap.delete(tid).await,
            None => Ok(false),
        }
    }

    async fn load_security_maps(&self) -> Result<Vec<SecurityMapEntry>> {
        scan_decode(
            &self.security_maps_heap,
            "security map",
            SecurityMapEntry::from_bytes,
        )
    }

    async fn store_security_map(&self, entry: &SecurityMapEntry) -> Result<TupleId> {
        let bytes = entry.to_bytes();
        let tuple = Tuple::new(bytes, 0);
        let ids = self.security_maps_heap.insert_batch(&[tuple]).await?;
        Ok(ids[0])
    }

    async fn update_security_map(&self, entry: &SecurityMapEntry) -> Result<bool> {
        let mut target = None;
        let guard = self.security_maps_heap.scan()?;
        guard.for_each(|tid, view| {
            if let Ok(existing) = SecurityMapEntry::from_bytes(view.data) {
                if existing.id == entry.id {
                    target = Some(tid);
                }
            }
        });
        drop(guard);
        match target {
            Some(tid) => {
                self.security_maps_heap.delete(tid).await?;
                let bytes = entry.to_bytes();
                let tuple = Tuple::new(bytes, 0);
                self.security_maps_heap.insert_batch(&[tuple]).await?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    async fn delete_security_map(&self, id: SecurityMapId) -> Result<bool> {
        let mut target = None;
        let guard = self.security_maps_heap.scan()?;
        guard.for_each(|tid, view| {
            if let Ok(entry) = SecurityMapEntry::from_bytes(view.data) {
                if entry.id == id {
                    target = Some(tid);
                }
            }
        });
        match target {
            Some(tid) => self.security_maps_heap.delete(tid).await,
            None => Ok(false),
        }
    }

    // ----- Data lifecycle -----

    async fn load_legal_holds(&self) -> Result<Vec<LegalHoldEntry>> {
        scan_decode(
            &self.legal_holds_heap,
            "legal hold",
            LegalHoldEntry::from_bytes,
        )
    }

    async fn store_legal_hold(&self, entry: &LegalHoldEntry) -> Result<TupleId> {
        let tuple = Tuple::new(entry.to_bytes(), 0);
        let ids = self.legal_holds_heap.insert_batch(&[tuple]).await?;
        Ok(ids[0])
    }

    async fn update_legal_hold(&self, entry: &LegalHoldEntry) -> Result<bool> {
        let mut target = None;
        let guard = self.legal_holds_heap.scan()?;
        guard.for_each(|tid, view| {
            if let Ok(existing) = LegalHoldEntry::from_bytes(view.data) {
                if existing.id == entry.id {
                    target = Some(tid);
                }
            }
        });
        drop(guard);
        match target {
            Some(tid) => {
                self.legal_holds_heap.delete(tid).await?;
                let tuple = Tuple::new(entry.to_bytes(), 0);
                self.legal_holds_heap.insert_batch(&[tuple]).await?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    async fn delete_legal_hold(&self, id: u32) -> Result<bool> {
        let mut target = None;
        let guard = self.legal_holds_heap.scan()?;
        guard.for_each(|tid, view| {
            if let Ok(e) = LegalHoldEntry::from_bytes(view.data) {
                if e.id == id {
                    target = Some(tid);
                }
            }
        });
        match target {
            Some(tid) => self.legal_holds_heap.delete(tid).await,
            None => Ok(false),
        }
    }

    async fn load_retention_policies(&self) -> Result<Vec<RetentionPolicyEntry>> {
        scan_decode(
            &self.retention_policies_heap,
            "retention policy",
            RetentionPolicyEntry::from_bytes,
        )
    }

    async fn store_retention_policy(&self, entry: &RetentionPolicyEntry) -> Result<TupleId> {
        let tuple = Tuple::new(entry.to_bytes(), 0);
        let ids = self.retention_policies_heap.insert_batch(&[tuple]).await?;
        Ok(ids[0])
    }

    async fn replace_retention_policies(
        &self,
        table_id: u32,
        entries: &[RetentionPolicyEntry],
    ) -> Result<()> {
        let mut stale = Vec::new();
        let mut decode_err: Option<ZyronError> = None;
        let guard = self.retention_policies_heap.scan()?;
        guard.try_for_each(
            |tid, view| match RetentionPolicyEntry::from_bytes(view.data) {
                Ok(e) => {
                    if e.table_id == table_id {
                        stale.push(tid);
                    }
                    true
                }
                Err(e) => {
                    decode_err = Some(ZyronError::CatalogCorrupted(format!(
                        "retention policy catalog tuple failed to decode: {}",
                        e
                    )));
                    false
                }
            },
        );
        drop(guard);
        if let Some(e) = decode_err {
            return Err(e);
        }
        for tid in stale {
            self.retention_policies_heap.delete(tid).await?;
        }
        for e in entries {
            let tuple = Tuple::new(e.to_bytes(), 0);
            self.retention_policies_heap.insert_batch(&[tuple]).await?;
        }
        Ok(())
    }

    async fn load_retention_jobs(&self) -> Result<Vec<RetentionJobEntry>> {
        scan_decode(
            &self.retention_jobs_heap,
            "retention job",
            RetentionJobEntry::from_bytes,
        )
    }

    async fn store_retention_job(&self, entry: &RetentionJobEntry) -> Result<TupleId> {
        let tuple = Tuple::new(entry.to_bytes(), 0);
        let ids = self.retention_jobs_heap.insert_batch(&[tuple]).await?;
        Ok(ids[0])
    }

    async fn load_compliance_log(&self) -> Result<Vec<ComplianceLogEntry>> {
        let mut entries = scan_decode(
            &self.compliance_log_heap,
            "compliance log",
            ComplianceLogEntry::from_bytes,
        )?;
        entries.sort_by_key(|e| e.event_id);
        Ok(entries)
    }

    async fn store_compliance_log(&self, entry: &ComplianceLogEntry) -> Result<TupleId> {
        let tuple = Tuple::new(entry.to_bytes(), 0);
        let ids = self.compliance_log_heap.insert_batch(&[tuple]).await?;
        Ok(ids[0])
    }

    async fn is_bootstrapped(&self) -> Result<bool> {
        let dbs = self.load_databases().await?;
        Ok(!dbs.is_empty())
    }

    async fn bootstrap(&self) -> Result<()> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Create default database
        let db = DatabaseEntry {
            id: SYSTEM_DATABASE_ID,
            name: "zyron".to_string(),
            owner: "system".to_string(),
            created_at: now,
        };
        self.store_database(&db).await?;

        // Create the internal system schema. Only Zyron's own bookkeeping
        // tables live here (database list, schema list, table list, columns,
        // indexes, roles, privileges, replication slots, etc). User writes
        // to this schema are rejected by the catalog.
        let system_schema = SchemaEntry {
            id: SYSTEM_SCHEMA_ID,
            database_id: SYSTEM_DATABASE_ID,
            name: SYSTEM_SCHEMA_NAME.to_string(),
            owner: "system".to_string(),
        };
        self.store_schema(&system_schema).await?;

        Ok(())
    }

    fn next_heap_file_id(&self) -> (u32, u32) {
        let heap = self.next_heap_file.fetch_add(2, Ordering::Relaxed);
        (heap, heap + 1)
    }

    fn next_index_file_id(&self) -> u32 {
        self.next_index_file.fetch_add(1, Ordering::Relaxed)
    }
}

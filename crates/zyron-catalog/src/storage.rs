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
use zyron_common::Result;
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

/// Starting file ID for user-created heap files (heap=200, fsm=201, ...).
const USER_HEAP_FILE_START: u32 = 200;

/// Starting file ID for user-created index files.
const USER_INDEX_FILE_START: u32 = 10000;

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

    // Bootstrap and recovery
    async fn is_bootstrapped(&self) -> Result<bool>;
    async fn bootstrap(&self) -> Result<()>;

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
    next_heap_file: AtomicU32,
    next_index_file: AtomicU32,
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
            next_heap_file: AtomicU32::new(USER_HEAP_FILE_START),
            next_index_file: AtomicU32::new(USER_INDEX_FILE_START),
        })
    }

    /// Initializes page count caches for all system table heap files.
    /// Runs all 5 init calls concurrently to minimize cold-start latency.
    pub async fn init_cache(&self) -> Result<()> {
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
    async fn load_databases(&self) -> Result<Vec<DatabaseEntry>> {
        let mut entries = Vec::new();
        let guard = self.databases_heap.scan()?;
        guard.for_each(|_tid, view| {
            if let Ok(entry) = DatabaseEntry::from_bytes(view.data) {
                entries.push(entry);
            }
        });
        Ok(entries)
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
        let mut entries = Vec::new();
        let guard = self.schemas_heap.scan()?;
        guard.for_each(|_tid, view| {
            if let Ok(entry) = SchemaEntry::from_bytes(view.data) {
                entries.push(entry);
            }
        });
        Ok(entries)
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
        let mut entries = Vec::new();
        let guard = self.tables_heap.scan()?;
        guard.for_each(|_tid, view| {
            if let Ok(entry) = TableEntry::from_bytes(view.data) {
                entries.push(entry);
            }
        });
        Ok(entries)
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
        let mut entries = Vec::new();
        let guard = self.columns_heap.scan()?;
        guard.for_each(|_tid, view| {
            if let Ok(entry) = ColumnEntry::from_bytes(view.data) {
                if entry.table_id == table_id {
                    entries.push(entry);
                }
            }
        });
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
        let mut entries = Vec::new();
        let guard = self.indexes_heap.scan()?;
        guard.for_each(|_tid, view| {
            if let Ok(entry) = IndexEntry::from_bytes(view.data) {
                entries.push(entry);
            }
        });
        Ok(entries)
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
        let mut entries = Vec::new();
        let guard = self.streaming_jobs_heap.scan()?;
        guard.for_each(|_tid, view| {
            if let Ok(entry) = StreamingJobEntry::from_bytes(view.data) {
                entries.push(entry);
            }
        });
        Ok(entries)
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

    async fn load_external_sources(&self) -> Result<Vec<ExternalSourceEntry>> {
        let mut entries = Vec::new();
        let guard = self.external_sources_heap.scan()?;
        guard.for_each(|_tid, view| {
            if let Ok(entry) = ExternalSourceEntry::from_bytes(view.data) {
                entries.push(entry);
            }
        });
        Ok(entries)
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
        let mut entries = Vec::new();
        let guard = self.external_sinks_heap.scan()?;
        guard.for_each(|_tid, view| {
            if let Ok(entry) = ExternalSinkEntry::from_bytes(view.data) {
                entries.push(entry);
            }
        });
        Ok(entries)
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
        let mut entries = Vec::new();
        let guard = self.publications_heap.scan()?;
        guard.for_each(|_tid, view| {
            if let Ok(entry) = PublicationEntry::from_bytes(view.data) {
                entries.push(entry);
            }
        });
        Ok(entries)
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
        let mut entries = Vec::new();
        let guard = self.publication_tables_heap.scan()?;
        guard.for_each(|_tid, view| {
            if let Ok(entry) = PublicationTableEntry::from_bytes(view.data) {
                entries.push(entry);
            }
        });
        Ok(entries)
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
        let mut entries = Vec::new();
        let guard = self.subscriptions_heap.scan()?;
        guard.for_each(|_tid, view| {
            if let Ok(entry) = SubscriptionEntry::from_bytes(view.data) {
                entries.push(entry);
            }
        });
        Ok(entries)
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
        let mut entries = Vec::new();
        let guard = self.endpoints_heap.scan()?;
        guard.for_each(|_tid, view| {
            if let Ok(entry) = EndpointEntry::from_bytes(view.data) {
                entries.push(entry);
            }
        });
        Ok(entries)
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
        let mut entries = Vec::new();
        let guard = self.security_maps_heap.scan()?;
        guard.for_each(|_tid, view| {
            if let Ok(entry) = SecurityMapEntry::from_bytes(view.data) {
                entries.push(entry);
            }
        });
        Ok(entries)
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

        // Create "public" schema
        let public_schema = SchemaEntry {
            id: DEFAULT_SCHEMA_ID,
            database_id: SYSTEM_DATABASE_ID,
            name: "public".to_string(),
            owner: "system".to_string(),
        };
        self.store_schema(&public_schema).await?;

        // Create "zyron_catalog" internal schema
        let catalog_schema = SchemaEntry {
            id: CATALOG_SCHEMA_ID,
            database_id: SYSTEM_DATABASE_ID,
            name: "zyron_catalog".to_string(),
            owner: "system".to_string(),
        };
        self.store_schema(&catalog_schema).await?;

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

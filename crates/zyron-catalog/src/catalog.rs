//! Central catalog manager for Zyron.
//!
//! Coordinates DDL operations with WAL logging, cache updates,
//! and storage persistence. All DDL operations are crash-safe
//! through WAL integration.

use crate::cache::CatalogCache;
use crate::ids::*;
use crate::resolver::NameResolver;
use crate::schema::*;
use crate::stats::{ColumnStats, TableStats};
use crate::storage::CatalogStorage;
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use zyron_common::{Result, ZyronError};
use zyron_parser::ast::{
    ColumnConstraint, ColumnDef, DataType, TableConstraint, TableConstraintKind,
};
use zyron_wal::RecoveryManager;
use zyron_wal::record::{LogRecordType, Lsn};
use zyron_wal::writer::WalWriter;

/// DDL operation type prefixes for WAL payloads.
const DDL_CREATE_DATABASE: u8 = 0x01;
const DDL_DROP_DATABASE: u8 = 0x02;
const DDL_CREATE_SCHEMA: u8 = 0x03;
const DDL_DROP_SCHEMA: u8 = 0x04;
const DDL_CREATE_TABLE: u8 = 0x05;
const DDL_DROP_TABLE: u8 = 0x06;
const DDL_CREATE_INDEX: u8 = 0x07;
const DDL_DROP_INDEX: u8 = 0x08;
const DDL_CREATE_STREAMING_JOB: u8 = 0x09;
const DDL_DROP_STREAMING_JOB: u8 = 0x0A;
const DDL_ALTER_STREAMING_JOB: u8 = 0x0B;
const DDL_CREATE_EXTERNAL_SOURCE: u8 = 0x0C;
const DDL_DROP_EXTERNAL_SOURCE: u8 = 0x0D;
const DDL_ALTER_EXTERNAL_SOURCE: u8 = 0x0E;
const DDL_CREATE_EXTERNAL_SINK: u8 = 0x0F;
const DDL_DROP_EXTERNAL_SINK: u8 = 0x10;
const DDL_ALTER_EXTERNAL_SINK: u8 = 0x11;
const DDL_CREATE_PUBLICATION: u8 = 0x12;
const DDL_DROP_PUBLICATION: u8 = 0x13;
const DDL_ALTER_PUBLICATION: u8 = 0x14;
const DDL_CREATE_SUBSCRIPTION: u8 = 0x15;
const DDL_DROP_SUBSCRIPTION: u8 = 0x16;
const DDL_UPDATE_SUBSCRIPTION: u8 = 0x17;
const DDL_CREATE_ENDPOINT: u8 = 0x18;
const DDL_DROP_ENDPOINT: u8 = 0x19;
const DDL_ALTER_ENDPOINT: u8 = 0x1A;
const DDL_CREATE_SECURITY_MAP: u8 = 0x1B;
const DDL_DROP_SECURITY_MAP: u8 = 0x1C;
const DDL_ADD_PUBLICATION_TABLE: u8 = 0x1D;
const DDL_REMOVE_PUBLICATION_TABLE: u8 = 0x1E;
const DDL_CREATE_SEQUENCE: u8 = 0x1F;
const DDL_DROP_SEQUENCE: u8 = 0x20;
const DDL_UPDATE_SEQUENCE: u8 = 0x21;
const DDL_CREATE_VIEW: u8 = 0x22;
const DDL_DROP_VIEW: u8 = 0x23;
const DDL_UPDATE_VIEW: u8 = 0x24;
const DDL_CREATE_MVIEW: u8 = 0x25;
const DDL_DROP_MVIEW: u8 = 0x26;
const DDL_UPDATE_MVIEW: u8 = 0x27;
const DDL_CREATE_FUNCTION: u8 = 0x28;
const DDL_DROP_FUNCTION: u8 = 0x29;
const DDL_SET_COMMENT: u8 = 0x2A;
const DDL_DROP_COMMENT: u8 = 0x2B;
const DDL_CREATE_AGGREGATE: u8 = 0x2C;
const DDL_DROP_AGGREGATE: u8 = 0x2D;
const DDL_CREATE_PROCEDURE: u8 = 0x2E;
const DDL_DROP_PROCEDURE: u8 = 0x2F;
const DDL_CREATE_SCHEDULE: u8 = 0x30;
const DDL_DROP_SCHEDULE: u8 = 0x31;
const DDL_CREATE_TRIGGER: u8 = 0x32;
const DDL_DROP_TRIGGER: u8 = 0x33;
const DDL_CREATE_PIPELINE: u8 = 0x34;
const DDL_DROP_PIPELINE: u8 = 0x35;
const DDL_CREATE_EVENT_HANDLER: u8 = 0x36;
const DDL_DROP_EVENT_HANDLER: u8 = 0x37;
const DDL_CREATE_VERSION_TAG: u8 = 0x38;
const DDL_DROP_VERSION_TAG: u8 = 0x39;

/// Result of a `drop_table` call. `soft_dropped` is true when the table went
/// to the recycle bin (entry and backing files retained for UNDROP); false
/// when it was physically removed and the caller should reclaim the files.
#[derive(Debug, Clone)]
pub struct DropOutcome {
    pub soft_dropped: bool,
    pub table_id: TableId,
    pub heap_file_id: u32,
    pub fsm_file_id: u32,
}

/// Central catalog manager.
pub struct Catalog {
    storage: Arc<dyn CatalogStorage>,
    cache: Arc<CatalogCache>,
    wal: Arc<WalWriter>,
    oid_allocator: OidAllocator,
    // Read-mostly: written only by the background stats refresh, read on
    // every cardinality estimate. The value is an Arc so a read is a single
    // refcount bump, not a deep clone of TableStats + Vec<ColumnStats>; the
    // RwLock read section is then just a map get and is uncontended in
    // practice (writes are seconds apart, readers do not block readers).
    stats: RwLock<HashMap<TableId, Arc<(TableStats, Vec<ColumnStats>)>>>,
    /// Monotonically advancing version stamp bumped on every DDL. Wire-layer
    /// plan caches read this with a single atomic load to validate that a
    /// cached PhysicalPlan is still consistent with the live catalog.
    schema_version: std::sync::atomic::AtomicU64,
    /// Live sequence managers keyed by (schema_id, name) for resolution and by
    /// id for default-value lookup. Each value is shared so concurrent
    /// sessions hand out values from one block cursor.
    sequences_by_name: RwLock<HashMap<(u32, String), Arc<crate::sequence::LiveSequence>>>,
    sequences_by_id: RwLock<HashMap<u32, Arc<crate::sequence::LiveSequence>>>,
    /// View definitions keyed by (schema_id, name) for binder expansion and by
    /// id for drop/alter. Read-mostly: queries look up by name on every bind.
    views_by_name: RwLock<HashMap<(u32, String), Arc<crate::schema::ViewEntry>>>,
    views_by_id: RwLock<HashMap<u32, Arc<crate::schema::ViewEntry>>>,
    /// Materialized view definitions keyed by (schema_id, name) and by id. The
    /// query result lives in a backing table; these entries carry the query
    /// text for REFRESH and the backing table id.
    mviews_by_name: RwLock<HashMap<(u32, String), Arc<crate::schema::MaterializedViewEntry>>>,
    mviews_by_id: RwLock<HashMap<u32, Arc<crate::schema::MaterializedViewEntry>>>,
    /// SQL scalar functions keyed by bare name, each name mapping to its
    /// overloads (distinguished by parameter count/types). The binder resolves
    /// a call against these and inlines the body. Also indexed by id for drop.
    functions_by_name: RwLock<HashMap<String, Vec<Arc<crate::schema::FunctionEntry>>>>,
    functions_by_id: RwLock<HashMap<u32, Arc<crate::schema::FunctionEntry>>>,
    /// Object comments keyed by (object kind, object name, column name). The
    /// column name is empty for object-level comments. Introspection reads
    /// these to describe schema objects.
    comments: RwLock<HashMap<(u8, String, String), Arc<crate::schema::CommentEntry>>>,
    /// SQL user-defined aggregates keyed by bare name, each name mapping to its
    /// overloads (distinguished by input count/types). The binder resolves an
    /// aggregate call against these and inlines the state and final functions.
    /// Also indexed by id for drop.
    aggregates_by_name: RwLock<HashMap<String, Vec<Arc<crate::schema::AggregateEntry>>>>,
    aggregates_by_id: RwLock<HashMap<u32, Arc<crate::schema::AggregateEntry>>>,
    /// SQL stored procedures keyed by bare name, each name mapping to its
    /// overloads (distinguished by input count/types). CALL resolves against
    /// these and runs the body. Also indexed by id for drop.
    procedures_by_name: RwLock<HashMap<String, Vec<Arc<crate::schema::ProcedureEntry>>>>,
    procedures_by_id: RwLock<HashMap<u32, Arc<crate::schema::ProcedureEntry>>>,
    /// Scheduled tasks keyed by name (unique). The background worker reads the
    /// active ones and fires those whose next_run has elapsed. Also indexed by
    /// id for drop and run-state persistence.
    schedules_by_name: RwLock<HashMap<String, Arc<crate::schema::ScheduleEntry>>>,
    schedules_by_id: RwLock<HashMap<u32, Arc<crate::schema::ScheduleEntry>>>,
    /// Triggers indexed by id (for drop) and by table id (for the executor's
    /// per-event lookup on the DML hot path).
    triggers_by_id: RwLock<HashMap<u32, Arc<crate::schema::TriggerEntry>>>,
    triggers_by_table: RwLock<HashMap<u32, Vec<Arc<crate::schema::TriggerEntry>>>>,
    /// Declarative pipelines keyed by name (unique) and by id (for drop and
    /// run-state persistence). The definition SQL is re-parsed on RUN.
    pipelines_by_name: RwLock<HashMap<String, Arc<crate::schema::PipelineEntry>>>,
    pipelines_by_id: RwLock<HashMap<u32, Arc<crate::schema::PipelineEntry>>>,
    /// Event handlers keyed by name (unique) and by id (for drop). The event
    /// dispatcher is rebuilt from these on startup.
    event_handlers_by_name: RwLock<HashMap<String, Arc<crate::schema::EventHandlerEntry>>>,
    event_handlers_by_id: RwLock<HashMap<u32, Arc<crate::schema::EventHandlerEntry>>>,
    /// Named version tags keyed by name (unique) and by id (for drop).
    version_tags_by_name: RwLock<HashMap<String, Arc<crate::schema::VersionTagEntry>>>,
    version_tags_by_id: RwLock<HashMap<u32, Arc<crate::schema::VersionTagEntry>>>,
    /// Serializes compliance-log appends so the tamper-evident hash chain is
    /// linear. The load-compute-store sequence runs under this lock so two
    /// concurrent appends cannot read the same tail and fork the chain.
    compliance_append_lock: tokio::sync::Mutex<()>,
}

impl Catalog {
    /// Creates a new catalog. Bootstraps system tables on first init.
    pub async fn new(
        storage: Arc<dyn CatalogStorage>,
        cache: Arc<CatalogCache>,
        wal: Arc<WalWriter>,
    ) -> Result<Self> {
        let catalog = Self {
            storage,
            cache,
            wal,
            oid_allocator: OidAllocator::new(USER_OID_START),
            stats: RwLock::new(HashMap::new()),
            schema_version: std::sync::atomic::AtomicU64::new(1),
            sequences_by_name: RwLock::new(HashMap::new()),
            sequences_by_id: RwLock::new(HashMap::new()),
            views_by_name: RwLock::new(HashMap::new()),
            views_by_id: RwLock::new(HashMap::new()),
            mviews_by_name: RwLock::new(HashMap::new()),
            mviews_by_id: RwLock::new(HashMap::new()),
            functions_by_name: RwLock::new(HashMap::new()),
            functions_by_id: RwLock::new(HashMap::new()),
            comments: RwLock::new(HashMap::new()),
            aggregates_by_name: RwLock::new(HashMap::new()),
            aggregates_by_id: RwLock::new(HashMap::new()),
            procedures_by_name: RwLock::new(HashMap::new()),
            procedures_by_id: RwLock::new(HashMap::new()),
            schedules_by_name: RwLock::new(HashMap::new()),
            schedules_by_id: RwLock::new(HashMap::new()),
            triggers_by_id: RwLock::new(HashMap::new()),
            triggers_by_table: RwLock::new(HashMap::new()),
            pipelines_by_name: RwLock::new(HashMap::new()),
            pipelines_by_id: RwLock::new(HashMap::new()),
            event_handlers_by_name: RwLock::new(HashMap::new()),
            event_handlers_by_id: RwLock::new(HashMap::new()),
            version_tags_by_name: RwLock::new(HashMap::new()),
            version_tags_by_id: RwLock::new(HashMap::new()),
            compliance_append_lock: tokio::sync::Mutex::new(()),
        };

        if !catalog.storage.is_bootstrapped().await? {
            catalog.storage.bootstrap().await?;
        }

        // Seed storage-internal counters (heap page caches, FSM pages) from
        // on-disk file sizes. This is required even on already-bootstrapped
        // storage so reopens after a crash see the real page counts and
        // scans iterate every persisted tuple.
        catalog.storage.init().await?;

        // Replay committed DDL records from the WAL that the storage pages
        // have not yet absorbed. Cheap-skip path: when the catalog's
        // checkpoint marker file says every WAL byte already written is
        // reflected in storage, skip the WAL scan entirely. This is the
        // common clean-shutdown reopen and runs in O(1).
        let wal_dir = catalog.wal.wal_dir();
        let marker = crate::checkpoint::read(wal_dir).unwrap_or(None);
        let wal_frontier = catalog.wal.flushed_lsn().0;
        let skip_recover = match marker {
            Some(m) => m.last_applied_lsn >= wal_frontier,
            None => false,
        };
        if !skip_recover {
            catalog.recover_unflushed_ddl().await?;
        }

        catalog.load().await?;
        Ok(catalog)
    }

    /// Drives the catalog checkpoint barrier.
    ///
    /// Captures the current WAL flushed LSN, flushes every catalog heap's
    /// dirty pages to disk so the on-disk storage view catches up to that
    /// LSN, then writes a WAL CheckpointEnd record marking the boundary.
    /// The CheckpointEnd write is itself waited for durability so a
    /// subsequent crash cannot lose the checkpoint marker.
    ///
    /// After this returns, every committed DDL whose LSN is at or below
    /// the checkpoint LSN is guaranteed reflected in storage pages on
    /// disk. The next Catalog::new sees the CheckpointEnd record during
    /// WAL recovery and clears its redo buffer, so reopen does O(1) work
    /// when no DDL was issued after the checkpoint.
    pub async fn checkpoint(&self) -> Result<()> {
        // 1. Capture the WAL frontier we are committing to disk. Any
        //    catalog DDL whose commit LSN is at or below this value will
        //    be durable after step 2 because log_ddl is synchronous
        //    against wait_for_flush.
        let chkpt_lsn = self.wal.flushed_lsn();

        // 2. Push every dirty catalog page to disk. After this point the
        //    on-disk pages reflect every DDL whose commit LSN <= chkpt_lsn.
        //    Ordering matters: storage must land before the CheckpointEnd
        //    record so a crash between the two steps over-replays
        //    (harmless) rather than under-replays (would lose DDL).
        self.storage.flush_all_dirty().await?;

        // 3. Record the checkpoint in the WAL. payload encodes the LSN
        //    value the storage view is known to have reached, matching
        //    the format zyron_wal::RecoveryManager expects (first 8 bytes
        //    little-endian u64). Wait for the record's durability so the
        //    next crash cannot lose it.
        let lsn_payload = chkpt_lsn.0.to_le_bytes();
        let end_lsn = self.wal.log_checkpoint_end(&lsn_payload)?;
        self.wal.wait_for_flush(end_lsn)?;

        // 4. Persist the checkpoint marker file. Once this lands the next
        //    Catalog::new can compare the marker against the WAL frontier
        //    and skip the recovery scan when the marker covers it. The
        //    marker is written atomically (write-temp + fsync + rename)
        //    so a crash mid-write either leaves the previous marker intact
        //    or none at all; in either case recovery is still correct.
        let wal_dir = self.wal.wal_dir().to_path_buf();
        let marker = crate::checkpoint::CatalogCheckpoint {
            last_applied_lsn: end_lsn.0,
        };
        crate::checkpoint::write_atomic(&wal_dir, &marker)?;
        Ok(())
    }

    /// Replays DDL records from the WAL into storage to recover writes that
    /// committed before a crash but had not been flushed by the buffer pool.
    /// Each record is applied in LSN order. Stores are idempotent against
    /// existing rows; deletes are no-ops when the row is already absent.
    async fn recover_unflushed_ddl(&self) -> Result<()> {
        let wal_dir = self.wal.wal_dir().to_path_buf();
        let rm = RecoveryManager::new(&wal_dir)?;
        let result = rm.recover()?;
        if result.redo_records.is_empty() {
            return Ok(());
        }

        // Snapshot what storage currently holds so we can skip rows that are
        // already durably present. This snapshot does not need to be
        // consistent with concurrent writers because Catalog::new runs
        // before the server accepts any external traffic.
        let mut have_databases: HashSet<u32> = self
            .storage
            .load_databases()
            .await?
            .into_iter()
            .map(|e| e.id.0)
            .collect();
        let mut have_schemas: HashSet<u32> = self
            .storage
            .load_schemas()
            .await?
            .into_iter()
            .map(|e| e.id.0)
            .collect();
        let mut have_tables: HashSet<u32> = self
            .storage
            .load_tables()
            .await?
            .into_iter()
            .map(|e| e.id.0)
            .collect();
        let mut have_indexes: HashSet<u32> = self
            .storage
            .load_indexes()
            .await?
            .into_iter()
            .map(|e| e.id.0)
            .collect();
        let mut have_publications: HashSet<u32> = self
            .storage
            .load_publications()
            .await?
            .into_iter()
            .map(|e| e.id.0)
            .collect();
        let mut have_pub_tables: HashSet<(u32, u32)> = self
            .storage
            .load_publication_tables()
            .await?
            .into_iter()
            .map(|e| (e.publication_id.0, e.table_id.0))
            .collect();
        let mut have_subscriptions: HashSet<u32> = self
            .storage
            .load_subscriptions()
            .await?
            .into_iter()
            .map(|e| e.id.0)
            .collect();
        let mut have_endpoints: HashSet<u32> = self
            .storage
            .load_endpoints()
            .await?
            .into_iter()
            .map(|e| e.id.0)
            .collect();
        let mut have_security_maps: HashSet<u32> = self
            .storage
            .load_security_maps()
            .await?
            .into_iter()
            .map(|e| e.id.0)
            .collect();
        let mut have_external_sources: HashSet<u32> = self
            .storage
            .load_external_sources()
            .await?
            .into_iter()
            .map(|e| e.id.0)
            .collect();
        let mut have_external_sinks: HashSet<u32> = self
            .storage
            .load_external_sinks()
            .await?
            .into_iter()
            .map(|e| e.id.0)
            .collect();
        let mut have_streaming_jobs: HashSet<u32> = self
            .storage
            .load_streaming_jobs()
            .await?
            .into_iter()
            .map(|e| e.id.0)
            .collect();
        let mut have_sequences: HashSet<u32> = self
            .storage
            .load_sequences()
            .await?
            .into_iter()
            .map(|e| e.id)
            .collect();
        let mut have_views: HashSet<u32> = self
            .storage
            .load_views()
            .await?
            .into_iter()
            .map(|e| e.id)
            .collect();
        let mut have_mviews: HashSet<u32> = self
            .storage
            .load_mviews()
            .await?
            .into_iter()
            .map(|e| e.id)
            .collect();
        let mut have_functions: HashSet<u32> = self
            .storage
            .load_functions()
            .await?
            .into_iter()
            .map(|e| e.id)
            .collect();
        let mut have_comments: HashSet<u32> = self
            .storage
            .load_comments()
            .await?
            .into_iter()
            .map(|e| e.id)
            .collect();
        let mut have_aggregates: HashSet<u32> = self
            .storage
            .load_aggregates()
            .await?
            .into_iter()
            .map(|e| e.id)
            .collect();
        let mut have_procedures: HashSet<u32> = self
            .storage
            .load_procedures()
            .await?
            .into_iter()
            .map(|e| e.id)
            .collect();
        let mut have_schedules: HashSet<u32> = self
            .storage
            .load_schedules()
            .await?
            .into_iter()
            .map(|e| e.id)
            .collect();
        let mut have_triggers: HashSet<u32> = self
            .storage
            .load_triggers()
            .await?
            .into_iter()
            .map(|e| e.id)
            .collect();
        let mut have_pipelines: HashSet<u32> = self
            .storage
            .load_pipelines()
            .await?
            .into_iter()
            .map(|e| e.id)
            .collect();
        let mut have_event_handlers: HashSet<u32> = self
            .storage
            .load_event_handlers()
            .await?
            .into_iter()
            .map(|e| e.id)
            .collect();
        let mut have_version_tags: HashSet<u32> = self
            .storage
            .load_version_tags()
            .await?
            .into_iter()
            .map(|e| e.id)
            .collect();

        // Pre-dedupe redo records in LSN order, keeping only the latest
        // record per (entity-kind, id) tuple. Subsequent records for the
        // same object always supersede prior ones, so applying every
        // intermediate write is wasted work that dominates recovery time
        // at scale. After this pass the dispatch below does at most one
        // storage operation per logical object.
        //
        // The dedup key uses the entity kind (CREATE and UPDATE for the
        // same kind share a key, DROP shares the same key so a later DROP
        // wins over earlier CREATE/UPDATE) plus the affected id. For the
        // pub-table junction the key is the (publication_id, table_id)
        // pair.
        let mut redo = result.redo_records;
        redo.sort_by_key(|r| r.lsn.0);

        fn entity_key(ddl_type: u8, entry_bytes: &[u8]) -> Option<(u8, u64)> {
            fn read_u32(b: &[u8], off: usize) -> Option<u32> {
                if b.len() < off + 4 {
                    return None;
                }
                Some(u32::from_le_bytes([
                    b[off],
                    b[off + 1],
                    b[off + 2],
                    b[off + 3],
                ]))
            }
            let id_u32 = |b: &[u8]| read_u32(b, 0);
            match ddl_type {
                DDL_CREATE_DATABASE | DDL_DROP_DATABASE => {
                    let id: u32 = if ddl_type == DDL_CREATE_DATABASE {
                        DatabaseEntry::from_bytes(entry_bytes)
                            .ok()
                            .map(|e| e.id.0)?
                    } else {
                        id_u32(entry_bytes)?
                    };
                    Some((1, id as u64))
                }
                DDL_CREATE_SCHEMA | DDL_DROP_SCHEMA => {
                    let id: u32 = if ddl_type == DDL_CREATE_SCHEMA {
                        SchemaEntry::from_bytes(entry_bytes).ok().map(|e| e.id.0)?
                    } else {
                        id_u32(entry_bytes)?
                    };
                    Some((2, id as u64))
                }
                DDL_CREATE_TABLE | DDL_DROP_TABLE => {
                    let id: u32 = if ddl_type == DDL_CREATE_TABLE {
                        TableEntry::from_bytes(entry_bytes).ok().map(|e| e.id.0)?
                    } else {
                        id_u32(entry_bytes)?
                    };
                    Some((3, id as u64))
                }
                DDL_CREATE_INDEX | DDL_DROP_INDEX => {
                    let id: u32 = if ddl_type == DDL_CREATE_INDEX {
                        IndexEntry::from_bytes(entry_bytes).ok().map(|e| e.id.0)?
                    } else {
                        id_u32(entry_bytes)?
                    };
                    Some((4, id as u64))
                }
                DDL_CREATE_STREAMING_JOB | DDL_ALTER_STREAMING_JOB | DDL_DROP_STREAMING_JOB => {
                    let id: u32 = if ddl_type == DDL_DROP_STREAMING_JOB {
                        id_u32(entry_bytes)?
                    } else {
                        StreamingJobEntry::from_bytes(entry_bytes)
                            .ok()
                            .map(|e| e.id.0)?
                    };
                    Some((5, id as u64))
                }
                DDL_CREATE_EXTERNAL_SOURCE
                | DDL_ALTER_EXTERNAL_SOURCE
                | DDL_DROP_EXTERNAL_SOURCE => {
                    let id: u32 = if ddl_type == DDL_DROP_EXTERNAL_SOURCE {
                        id_u32(entry_bytes)?
                    } else {
                        ExternalSourceEntry::from_bytes(entry_bytes)
                            .ok()
                            .map(|e| e.id.0)?
                    };
                    Some((6, id as u64))
                }
                DDL_CREATE_EXTERNAL_SINK | DDL_ALTER_EXTERNAL_SINK | DDL_DROP_EXTERNAL_SINK => {
                    let id: u32 = if ddl_type == DDL_DROP_EXTERNAL_SINK {
                        id_u32(entry_bytes)?
                    } else {
                        ExternalSinkEntry::from_bytes(entry_bytes)
                            .ok()
                            .map(|e| e.id.0)?
                    };
                    Some((7, id as u64))
                }
                DDL_CREATE_PUBLICATION | DDL_ALTER_PUBLICATION | DDL_DROP_PUBLICATION => {
                    let id: u32 = if ddl_type == DDL_DROP_PUBLICATION {
                        id_u32(entry_bytes)?
                    } else {
                        PublicationEntry::from_bytes(entry_bytes)
                            .ok()
                            .map(|e| e.id.0)?
                    };
                    Some((8, id as u64))
                }
                DDL_ADD_PUBLICATION_TABLE | DDL_REMOVE_PUBLICATION_TABLE => {
                    let (pid, tid): (u32, u32) = if ddl_type == DDL_ADD_PUBLICATION_TABLE {
                        let e = PublicationTableEntry::from_bytes(entry_bytes).ok()?;
                        (e.publication_id.0, e.table_id.0)
                    } else {
                        (read_u32(entry_bytes, 0)?, read_u32(entry_bytes, 4)?)
                    };
                    Some((9, ((pid as u64) << 32) | (tid as u64)))
                }
                DDL_CREATE_SUBSCRIPTION | DDL_UPDATE_SUBSCRIPTION | DDL_DROP_SUBSCRIPTION => {
                    let id: u32 = if ddl_type == DDL_DROP_SUBSCRIPTION {
                        id_u32(entry_bytes)?
                    } else {
                        SubscriptionEntry::from_bytes(entry_bytes)
                            .ok()
                            .map(|e| e.id.0)?
                    };
                    Some((10, id as u64))
                }
                DDL_CREATE_ENDPOINT | DDL_ALTER_ENDPOINT | DDL_DROP_ENDPOINT => {
                    let id: u32 = if ddl_type == DDL_DROP_ENDPOINT {
                        id_u32(entry_bytes)?
                    } else {
                        EndpointEntry::from_bytes(entry_bytes)
                            .ok()
                            .map(|e| e.id.0)?
                    };
                    Some((11, id as u64))
                }
                DDL_CREATE_SECURITY_MAP | DDL_DROP_SECURITY_MAP => {
                    let id: u32 = if ddl_type == DDL_CREATE_SECURITY_MAP {
                        SecurityMapEntry::from_bytes(entry_bytes)
                            .ok()
                            .map(|e| e.id.0)?
                    } else {
                        id_u32(entry_bytes)?
                    };
                    Some((12, id as u64))
                }
                DDL_CREATE_SEQUENCE | DDL_UPDATE_SEQUENCE | DDL_DROP_SEQUENCE => {
                    let id: u32 = if ddl_type == DDL_DROP_SEQUENCE {
                        id_u32(entry_bytes)?
                    } else {
                        SequenceEntry::from_bytes(entry_bytes).ok().map(|e| e.id)?
                    };
                    Some((13, id as u64))
                }
                DDL_CREATE_VIEW | DDL_UPDATE_VIEW | DDL_DROP_VIEW => {
                    let id: u32 = if ddl_type == DDL_DROP_VIEW {
                        id_u32(entry_bytes)?
                    } else {
                        crate::schema::ViewEntry::from_bytes(entry_bytes)
                            .ok()
                            .map(|e| e.id)?
                    };
                    Some((14, id as u64))
                }
                DDL_CREATE_MVIEW | DDL_UPDATE_MVIEW | DDL_DROP_MVIEW => {
                    let id: u32 = if ddl_type == DDL_DROP_MVIEW {
                        id_u32(entry_bytes)?
                    } else {
                        crate::schema::MaterializedViewEntry::from_bytes(entry_bytes)
                            .ok()
                            .map(|e| e.id)?
                    };
                    Some((15, id as u64))
                }
                DDL_CREATE_FUNCTION | DDL_DROP_FUNCTION => {
                    let id: u32 = if ddl_type == DDL_DROP_FUNCTION {
                        id_u32(entry_bytes)?
                    } else {
                        crate::schema::FunctionEntry::from_bytes(entry_bytes)
                            .ok()
                            .map(|e| e.id)?
                    };
                    Some((16, id as u64))
                }
                DDL_SET_COMMENT | DDL_DROP_COMMENT => {
                    let id: u32 = if ddl_type == DDL_DROP_COMMENT {
                        id_u32(entry_bytes)?
                    } else {
                        crate::schema::CommentEntry::from_bytes(entry_bytes)
                            .ok()
                            .map(|e| e.id)?
                    };
                    Some((17, id as u64))
                }
                DDL_CREATE_AGGREGATE | DDL_DROP_AGGREGATE => {
                    let id: u32 = if ddl_type == DDL_DROP_AGGREGATE {
                        id_u32(entry_bytes)?
                    } else {
                        crate::schema::AggregateEntry::from_bytes(entry_bytes)
                            .ok()
                            .map(|e| e.id)?
                    };
                    Some((18, id as u64))
                }
                DDL_CREATE_PROCEDURE | DDL_DROP_PROCEDURE => {
                    let id: u32 = if ddl_type == DDL_DROP_PROCEDURE {
                        id_u32(entry_bytes)?
                    } else {
                        crate::schema::ProcedureEntry::from_bytes(entry_bytes)
                            .ok()
                            .map(|e| e.id)?
                    };
                    Some((19, id as u64))
                }
                DDL_CREATE_SCHEDULE | DDL_DROP_SCHEDULE => {
                    let id: u32 = if ddl_type == DDL_DROP_SCHEDULE {
                        id_u32(entry_bytes)?
                    } else {
                        crate::schema::ScheduleEntry::from_bytes(entry_bytes)
                            .ok()
                            .map(|e| e.id)?
                    };
                    Some((20, id as u64))
                }
                DDL_CREATE_TRIGGER | DDL_DROP_TRIGGER => {
                    let id: u32 = if ddl_type == DDL_DROP_TRIGGER {
                        id_u32(entry_bytes)?
                    } else {
                        crate::schema::TriggerEntry::from_bytes(entry_bytes)
                            .ok()
                            .map(|e| e.id)?
                    };
                    Some((21, id as u64))
                }
                DDL_CREATE_PIPELINE | DDL_DROP_PIPELINE => {
                    let id: u32 = if ddl_type == DDL_DROP_PIPELINE {
                        id_u32(entry_bytes)?
                    } else {
                        crate::schema::PipelineEntry::from_bytes(entry_bytes)
                            .ok()
                            .map(|e| e.id)?
                    };
                    Some((22, id as u64))
                }
                DDL_CREATE_EVENT_HANDLER | DDL_DROP_EVENT_HANDLER => {
                    let id: u32 = if ddl_type == DDL_DROP_EVENT_HANDLER {
                        id_u32(entry_bytes)?
                    } else {
                        crate::schema::EventHandlerEntry::from_bytes(entry_bytes)
                            .ok()
                            .map(|e| e.id)?
                    };
                    Some((23, id as u64))
                }
                DDL_CREATE_VERSION_TAG | DDL_DROP_VERSION_TAG => {
                    let id: u32 = if ddl_type == DDL_DROP_VERSION_TAG {
                        id_u32(entry_bytes)?
                    } else {
                        crate::schema::VersionTagEntry::from_bytes(entry_bytes)
                            .ok()
                            .map(|e| e.id)?
                    };
                    Some((24, id as u64))
                }
                _ => None,
            }
        }

        // Records carrying a recognized ddl_type but a payload that fails to
        // decode are real catalog writes that recovery cannot apply. Skipping
        // them would silently drop a committed object, so a known ddl_type
        // whose entry does not decode fails recovery.
        fn known_ddl_type(ddl_type: u8) -> bool {
            matches!(
                ddl_type,
                DDL_CREATE_DATABASE
                    | DDL_DROP_DATABASE
                    | DDL_CREATE_SCHEMA
                    | DDL_DROP_SCHEMA
                    | DDL_CREATE_TABLE
                    | DDL_DROP_TABLE
                    | DDL_CREATE_INDEX
                    | DDL_DROP_INDEX
                    | DDL_CREATE_STREAMING_JOB
                    | DDL_ALTER_STREAMING_JOB
                    | DDL_DROP_STREAMING_JOB
                    | DDL_CREATE_EXTERNAL_SOURCE
                    | DDL_ALTER_EXTERNAL_SOURCE
                    | DDL_DROP_EXTERNAL_SOURCE
                    | DDL_CREATE_EXTERNAL_SINK
                    | DDL_ALTER_EXTERNAL_SINK
                    | DDL_DROP_EXTERNAL_SINK
                    | DDL_CREATE_PUBLICATION
                    | DDL_ALTER_PUBLICATION
                    | DDL_DROP_PUBLICATION
                    | DDL_ADD_PUBLICATION_TABLE
                    | DDL_REMOVE_PUBLICATION_TABLE
                    | DDL_CREATE_SUBSCRIPTION
                    | DDL_UPDATE_SUBSCRIPTION
                    | DDL_DROP_SUBSCRIPTION
                    | DDL_CREATE_ENDPOINT
                    | DDL_ALTER_ENDPOINT
                    | DDL_DROP_ENDPOINT
                    | DDL_CREATE_SECURITY_MAP
                    | DDL_DROP_SECURITY_MAP
                    | DDL_CREATE_SEQUENCE
                    | DDL_UPDATE_SEQUENCE
                    | DDL_DROP_SEQUENCE
                    | DDL_CREATE_VIEW
                    | DDL_UPDATE_VIEW
                    | DDL_DROP_VIEW
                    | DDL_CREATE_MVIEW
                    | DDL_UPDATE_MVIEW
                    | DDL_DROP_MVIEW
                    | DDL_CREATE_FUNCTION
                    | DDL_DROP_FUNCTION
                    | DDL_SET_COMMENT
                    | DDL_DROP_COMMENT
                    | DDL_CREATE_AGGREGATE
                    | DDL_DROP_AGGREGATE
                    | DDL_CREATE_PROCEDURE
                    | DDL_DROP_PROCEDURE
                    | DDL_CREATE_SCHEDULE
                    | DDL_DROP_SCHEDULE
                    | DDL_CREATE_TRIGGER
                    | DDL_DROP_TRIGGER
                    | DDL_CREATE_PIPELINE
                    | DDL_DROP_PIPELINE
                    | DDL_CREATE_EVENT_HANDLER
                    | DDL_DROP_EVENT_HANDLER
                    | DDL_CREATE_VERSION_TAG
                    | DDL_DROP_VERSION_TAG
            )
        }

        let mut latest: HashMap<(u8, u64), zyron_wal::record::LogRecord> = HashMap::new();
        for record in redo
            .into_iter()
            .filter(|r| r.record_type == LogRecordType::Insert && !r.payload.is_empty())
        {
            let ddl_type = record.payload[0];
            let entry_bytes = &record.payload[1..];
            match entity_key(ddl_type, entry_bytes) {
                Some(key) => {
                    latest.insert(key, record);
                }
                None if known_ddl_type(ddl_type) => {
                    return Err(ZyronError::CatalogCorrupted(format!(
                        "redo record ddl_type {} at lsn {} failed to decode its entry",
                        ddl_type, record.lsn.0
                    )));
                }
                None => {}
            }
        }
        let mut deduped: Vec<zyron_wal::record::LogRecord> = latest.into_values().collect();
        deduped.sort_by_key(|r| r.lsn.0);

        for record in deduped {
            let ddl_type = record.payload[0];
            let entry_bytes = &record.payload[1..];
            match ddl_type {
                DDL_CREATE_DATABASE => {
                    if let Ok(entry) = DatabaseEntry::from_bytes(entry_bytes) {
                        if !have_databases.contains(&entry.id.0) {
                            let _ = self.storage.store_database(&entry).await;
                            have_databases.insert(entry.id.0);
                        }
                    }
                }
                DDL_DROP_DATABASE => {
                    if entry_bytes.len() >= 4 {
                        let id = u32::from_le_bytes([
                            entry_bytes[0],
                            entry_bytes[1],
                            entry_bytes[2],
                            entry_bytes[3],
                        ]);
                        if have_databases.remove(&id) {
                            let _ = self.storage.delete_database(DatabaseId(id)).await;
                        }
                    }
                }
                DDL_CREATE_SCHEMA => {
                    if let Ok(entry) = SchemaEntry::from_bytes(entry_bytes) {
                        if !have_schemas.contains(&entry.id.0) {
                            let _ = self.storage.store_schema(&entry).await;
                            have_schemas.insert(entry.id.0);
                        }
                    }
                }
                DDL_DROP_SCHEMA => {
                    if entry_bytes.len() >= 4 {
                        let id = u32::from_le_bytes([
                            entry_bytes[0],
                            entry_bytes[1],
                            entry_bytes[2],
                            entry_bytes[3],
                        ]);
                        if have_schemas.remove(&id) {
                            let _ = self.storage.delete_schema(SchemaId(id)).await;
                        }
                    }
                }
                DDL_CREATE_TABLE => {
                    if let Ok(entry) = TableEntry::from_bytes(entry_bytes) {
                        // CREATE TABLE log records are re-emitted by every
                        // ALTER. When the id is already present we treat
                        // the record as an update so column metadata stays
                        // in sync with the latest committed shape.
                        if have_tables.contains(&entry.id.0) {
                            let _ = self.storage.delete_table(entry.id).await;
                        }
                        let _ = self.storage.store_table(&entry).await;
                        have_tables.insert(entry.id.0);
                    }
                }
                DDL_DROP_TABLE => {
                    if entry_bytes.len() >= 4 {
                        let id = u32::from_le_bytes([
                            entry_bytes[0],
                            entry_bytes[1],
                            entry_bytes[2],
                            entry_bytes[3],
                        ]);
                        if have_tables.remove(&id) {
                            let _ = self.storage.delete_table(TableId(id)).await;
                        }
                    }
                }
                DDL_CREATE_INDEX => {
                    if let Ok(entry) = IndexEntry::from_bytes(entry_bytes) {
                        // CREATE INDEX records are re-emitted by ALTER INDEX
                        // RENAME. When the id is already present treat the record
                        // as an update so the index name stays in sync with the
                        // latest committed state.
                        if have_indexes.contains(&entry.id.0) {
                            let _ = self.storage.delete_index(entry.id).await;
                        }
                        let _ = self.storage.store_index(&entry).await;
                        have_indexes.insert(entry.id.0);
                    }
                }
                DDL_DROP_INDEX => {
                    if entry_bytes.len() >= 4 {
                        let id = u32::from_le_bytes([
                            entry_bytes[0],
                            entry_bytes[1],
                            entry_bytes[2],
                            entry_bytes[3],
                        ]);
                        if have_indexes.remove(&id) {
                            let _ = self.storage.delete_index(IndexId(id)).await;
                        }
                    }
                }
                DDL_CREATE_STREAMING_JOB | DDL_ALTER_STREAMING_JOB => {
                    if let Ok(entry) = StreamingJobEntry::from_bytes(entry_bytes) {
                        if have_streaming_jobs.contains(&entry.id.0) {
                            let _ = self.storage.delete_streaming_job(entry.id).await;
                        }
                        let _ = self.storage.store_streaming_job(&entry).await;
                        have_streaming_jobs.insert(entry.id.0);
                    }
                }
                DDL_DROP_STREAMING_JOB => {
                    if entry_bytes.len() >= 4 {
                        let id = u32::from_le_bytes([
                            entry_bytes[0],
                            entry_bytes[1],
                            entry_bytes[2],
                            entry_bytes[3],
                        ]);
                        if have_streaming_jobs.remove(&id) {
                            let _ = self.storage.delete_streaming_job(StreamingJobId(id)).await;
                        }
                    }
                }
                DDL_CREATE_SEQUENCE | DDL_UPDATE_SEQUENCE => {
                    if let Ok(entry) = SequenceEntry::from_bytes(entry_bytes) {
                        if have_sequences.contains(&entry.id) {
                            let _ = self.storage.update_sequence(&entry).await;
                        } else {
                            let _ = self.storage.store_sequence(&entry).await;
                            have_sequences.insert(entry.id);
                        }
                    }
                }
                DDL_DROP_SEQUENCE => {
                    if entry_bytes.len() >= 4 {
                        let id = u32::from_le_bytes([
                            entry_bytes[0],
                            entry_bytes[1],
                            entry_bytes[2],
                            entry_bytes[3],
                        ]);
                        if have_sequences.remove(&id) {
                            let _ = self.storage.delete_sequence(id).await;
                        }
                    }
                }
                DDL_CREATE_VIEW | DDL_UPDATE_VIEW => {
                    if let Ok(entry) = crate::schema::ViewEntry::from_bytes(entry_bytes) {
                        if have_views.contains(&entry.id) {
                            let _ = self.storage.update_view(&entry).await;
                        } else {
                            let _ = self.storage.store_view(&entry).await;
                            have_views.insert(entry.id);
                        }
                    }
                }
                DDL_DROP_VIEW => {
                    if entry_bytes.len() >= 4 {
                        let id = u32::from_le_bytes([
                            entry_bytes[0],
                            entry_bytes[1],
                            entry_bytes[2],
                            entry_bytes[3],
                        ]);
                        if have_views.remove(&id) {
                            let _ = self.storage.delete_view(id).await;
                        }
                    }
                }
                DDL_CREATE_MVIEW | DDL_UPDATE_MVIEW => {
                    if let Ok(entry) = crate::schema::MaterializedViewEntry::from_bytes(entry_bytes)
                    {
                        if have_mviews.contains(&entry.id) {
                            let _ = self.storage.update_mview(&entry).await;
                        } else {
                            let _ = self.storage.store_mview(&entry).await;
                            have_mviews.insert(entry.id);
                        }
                    }
                }
                DDL_DROP_MVIEW => {
                    if entry_bytes.len() >= 4 {
                        let id = u32::from_le_bytes([
                            entry_bytes[0],
                            entry_bytes[1],
                            entry_bytes[2],
                            entry_bytes[3],
                        ]);
                        if have_mviews.remove(&id) {
                            let _ = self.storage.delete_mview(id).await;
                        }
                    }
                }
                DDL_CREATE_FUNCTION => {
                    if let Ok(entry) = crate::schema::FunctionEntry::from_bytes(entry_bytes) {
                        if !have_functions.contains(&entry.id) {
                            let _ = self.storage.store_function(&entry).await;
                            have_functions.insert(entry.id);
                        }
                    }
                }
                DDL_DROP_FUNCTION => {
                    if entry_bytes.len() >= 4 {
                        let id = u32::from_le_bytes([
                            entry_bytes[0],
                            entry_bytes[1],
                            entry_bytes[2],
                            entry_bytes[3],
                        ]);
                        if have_functions.remove(&id) {
                            let _ = self.storage.delete_function(id).await;
                        }
                    }
                }
                DDL_SET_COMMENT => {
                    if let Ok(entry) = crate::schema::CommentEntry::from_bytes(entry_bytes) {
                        if have_comments.contains(&entry.id) {
                            let _ = self.storage.delete_comment(entry.id).await;
                        }
                        let _ = self.storage.store_comment(&entry).await;
                        have_comments.insert(entry.id);
                    }
                }
                DDL_DROP_COMMENT => {
                    if entry_bytes.len() >= 4 {
                        let id = u32::from_le_bytes([
                            entry_bytes[0],
                            entry_bytes[1],
                            entry_bytes[2],
                            entry_bytes[3],
                        ]);
                        if have_comments.remove(&id) {
                            let _ = self.storage.delete_comment(id).await;
                        }
                    }
                }
                DDL_CREATE_AGGREGATE => {
                    if let Ok(entry) = crate::schema::AggregateEntry::from_bytes(entry_bytes) {
                        if !have_aggregates.contains(&entry.id) {
                            let _ = self.storage.store_aggregate(&entry).await;
                            have_aggregates.insert(entry.id);
                        }
                    }
                }
                DDL_DROP_AGGREGATE => {
                    if entry_bytes.len() >= 4 {
                        let id = u32::from_le_bytes([
                            entry_bytes[0],
                            entry_bytes[1],
                            entry_bytes[2],
                            entry_bytes[3],
                        ]);
                        if have_aggregates.remove(&id) {
                            let _ = self.storage.delete_aggregate(id).await;
                        }
                    }
                }
                DDL_CREATE_PROCEDURE => {
                    if let Ok(entry) = crate::schema::ProcedureEntry::from_bytes(entry_bytes) {
                        if !have_procedures.contains(&entry.id) {
                            let _ = self.storage.store_procedure(&entry).await;
                            have_procedures.insert(entry.id);
                        }
                    }
                }
                DDL_DROP_PROCEDURE => {
                    if entry_bytes.len() >= 4 {
                        let id = u32::from_le_bytes([
                            entry_bytes[0],
                            entry_bytes[1],
                            entry_bytes[2],
                            entry_bytes[3],
                        ]);
                        if have_procedures.remove(&id) {
                            let _ = self.storage.delete_procedure(id).await;
                        }
                    }
                }
                DDL_CREATE_SCHEDULE => {
                    if let Ok(entry) = crate::schema::ScheduleEntry::from_bytes(entry_bytes) {
                        // Re-emitted by pause/resume and run-state updates, so
                        // treat an existing id as an update.
                        if have_schedules.contains(&entry.id) {
                            let _ = self.storage.delete_schedule(entry.id).await;
                        }
                        let _ = self.storage.store_schedule(&entry).await;
                        have_schedules.insert(entry.id);
                    }
                }
                DDL_DROP_SCHEDULE => {
                    if entry_bytes.len() >= 4 {
                        let id = u32::from_le_bytes([
                            entry_bytes[0],
                            entry_bytes[1],
                            entry_bytes[2],
                            entry_bytes[3],
                        ]);
                        if have_schedules.remove(&id) {
                            let _ = self.storage.delete_schedule(id).await;
                        }
                    }
                }
                DDL_CREATE_TRIGGER => {
                    if let Ok(entry) = crate::schema::TriggerEntry::from_bytes(entry_bytes) {
                        if !have_triggers.contains(&entry.id) {
                            let _ = self.storage.store_trigger(&entry).await;
                            have_triggers.insert(entry.id);
                        }
                    }
                }
                DDL_DROP_TRIGGER => {
                    if entry_bytes.len() >= 4 {
                        let id = u32::from_le_bytes([
                            entry_bytes[0],
                            entry_bytes[1],
                            entry_bytes[2],
                            entry_bytes[3],
                        ]);
                        if have_triggers.remove(&id) {
                            let _ = self.storage.delete_trigger(id).await;
                        }
                    }
                }
                DDL_CREATE_PIPELINE => {
                    if let Ok(entry) = crate::schema::PipelineEntry::from_bytes(entry_bytes) {
                        // Re-emitted after each RUN for run-state updates, so
                        // treat an existing id as an update.
                        if have_pipelines.contains(&entry.id) {
                            let _ = self.storage.delete_pipeline(entry.id).await;
                        }
                        let _ = self.storage.store_pipeline(&entry).await;
                        have_pipelines.insert(entry.id);
                    }
                }
                DDL_DROP_PIPELINE => {
                    if entry_bytes.len() >= 4 {
                        let id = u32::from_le_bytes([
                            entry_bytes[0],
                            entry_bytes[1],
                            entry_bytes[2],
                            entry_bytes[3],
                        ]);
                        if have_pipelines.remove(&id) {
                            let _ = self.storage.delete_pipeline(id).await;
                        }
                    }
                }
                DDL_CREATE_EVENT_HANDLER => {
                    if let Ok(entry) = crate::schema::EventHandlerEntry::from_bytes(entry_bytes) {
                        if have_event_handlers.contains(&entry.id) {
                            let _ = self.storage.delete_event_handler(entry.id).await;
                        }
                        let _ = self.storage.store_event_handler(&entry).await;
                        have_event_handlers.insert(entry.id);
                    }
                }
                DDL_DROP_EVENT_HANDLER => {
                    if entry_bytes.len() >= 4 {
                        let id = u32::from_le_bytes([
                            entry_bytes[0],
                            entry_bytes[1],
                            entry_bytes[2],
                            entry_bytes[3],
                        ]);
                        if have_event_handlers.remove(&id) {
                            let _ = self.storage.delete_event_handler(id).await;
                        }
                    }
                }
                DDL_CREATE_VERSION_TAG => {
                    if let Ok(entry) = crate::schema::VersionTagEntry::from_bytes(entry_bytes) {
                        if !have_version_tags.contains(&entry.id) {
                            let _ = self.storage.store_version_tag(&entry).await;
                            have_version_tags.insert(entry.id);
                        }
                    }
                }
                DDL_DROP_VERSION_TAG => {
                    if entry_bytes.len() >= 4 {
                        let id = u32::from_le_bytes([
                            entry_bytes[0],
                            entry_bytes[1],
                            entry_bytes[2],
                            entry_bytes[3],
                        ]);
                        if have_version_tags.remove(&id) {
                            let _ = self.storage.delete_version_tag(id).await;
                        }
                    }
                }
                DDL_CREATE_EXTERNAL_SOURCE | DDL_ALTER_EXTERNAL_SOURCE => {
                    if let Ok(entry) = ExternalSourceEntry::from_bytes(entry_bytes) {
                        if have_external_sources.contains(&entry.id.0) {
                            let _ = self.storage.delete_external_source(entry.id).await;
                        }
                        let _ = self.storage.store_external_source(&entry).await;
                        have_external_sources.insert(entry.id.0);
                    }
                }
                DDL_DROP_EXTERNAL_SOURCE => {
                    if entry_bytes.len() >= 4 {
                        let id = u32::from_le_bytes([
                            entry_bytes[0],
                            entry_bytes[1],
                            entry_bytes[2],
                            entry_bytes[3],
                        ]);
                        if have_external_sources.remove(&id) {
                            let _ = self
                                .storage
                                .delete_external_source(ExternalSourceId(id))
                                .await;
                        }
                    }
                }
                DDL_CREATE_EXTERNAL_SINK | DDL_ALTER_EXTERNAL_SINK => {
                    if let Ok(entry) = ExternalSinkEntry::from_bytes(entry_bytes) {
                        if have_external_sinks.contains(&entry.id.0) {
                            let _ = self.storage.delete_external_sink(entry.id).await;
                        }
                        let _ = self.storage.store_external_sink(&entry).await;
                        have_external_sinks.insert(entry.id.0);
                    }
                }
                DDL_DROP_EXTERNAL_SINK => {
                    if entry_bytes.len() >= 4 {
                        let id = u32::from_le_bytes([
                            entry_bytes[0],
                            entry_bytes[1],
                            entry_bytes[2],
                            entry_bytes[3],
                        ]);
                        if have_external_sinks.remove(&id) {
                            let _ = self.storage.delete_external_sink(ExternalSinkId(id)).await;
                        }
                    }
                }
                DDL_CREATE_PUBLICATION | DDL_ALTER_PUBLICATION => {
                    if let Ok(entry) = PublicationEntry::from_bytes(entry_bytes) {
                        if have_publications.contains(&entry.id.0) {
                            let _ = self.storage.update_publication(&entry).await;
                        } else {
                            let _ = self.storage.store_publication(&entry).await;
                            have_publications.insert(entry.id.0);
                        }
                    }
                }
                DDL_DROP_PUBLICATION => {
                    if entry_bytes.len() >= 4 {
                        let id = u32::from_le_bytes([
                            entry_bytes[0],
                            entry_bytes[1],
                            entry_bytes[2],
                            entry_bytes[3],
                        ]);
                        if have_publications.remove(&id) {
                            let _ = self.storage.delete_publication(PublicationId(id)).await;
                        }
                    }
                }
                DDL_ADD_PUBLICATION_TABLE => {
                    if let Ok(entry) = PublicationTableEntry::from_bytes(entry_bytes) {
                        let k = (entry.publication_id.0, entry.table_id.0);
                        if !have_pub_tables.contains(&k) {
                            let _ = self.storage.store_publication_table(&entry).await;
                            have_pub_tables.insert(k);
                        }
                    }
                }
                DDL_REMOVE_PUBLICATION_TABLE => {
                    if entry_bytes.len() >= 8 {
                        let pid = u32::from_le_bytes([
                            entry_bytes[0],
                            entry_bytes[1],
                            entry_bytes[2],
                            entry_bytes[3],
                        ]);
                        let tid = u32::from_le_bytes([
                            entry_bytes[4],
                            entry_bytes[5],
                            entry_bytes[6],
                            entry_bytes[7],
                        ]);
                        if have_pub_tables.remove(&(pid, tid)) {
                            let _ = self
                                .storage
                                .delete_publication_table(PublicationId(pid), TableId(tid))
                                .await;
                        }
                    }
                }
                DDL_CREATE_SUBSCRIPTION | DDL_UPDATE_SUBSCRIPTION => {
                    if let Ok(entry) = SubscriptionEntry::from_bytes(entry_bytes) {
                        if have_subscriptions.contains(&entry.id.0) {
                            let _ = self.storage.update_subscription(&entry).await;
                        } else {
                            let _ = self.storage.store_subscription(&entry).await;
                            have_subscriptions.insert(entry.id.0);
                        }
                    }
                }
                DDL_DROP_SUBSCRIPTION => {
                    if entry_bytes.len() >= 4 {
                        let id = u32::from_le_bytes([
                            entry_bytes[0],
                            entry_bytes[1],
                            entry_bytes[2],
                            entry_bytes[3],
                        ]);
                        if have_subscriptions.remove(&id) {
                            let _ = self.storage.delete_subscription(SubscriptionId(id)).await;
                        }
                    }
                }
                DDL_CREATE_ENDPOINT | DDL_ALTER_ENDPOINT => {
                    if let Ok(entry) = EndpointEntry::from_bytes(entry_bytes) {
                        if have_endpoints.contains(&entry.id.0) {
                            let _ = self.storage.update_endpoint(&entry).await;
                        } else {
                            let _ = self.storage.store_endpoint(&entry).await;
                            have_endpoints.insert(entry.id.0);
                        }
                    }
                }
                DDL_DROP_ENDPOINT => {
                    if entry_bytes.len() >= 4 {
                        let id = u32::from_le_bytes([
                            entry_bytes[0],
                            entry_bytes[1],
                            entry_bytes[2],
                            entry_bytes[3],
                        ]);
                        if have_endpoints.remove(&id) {
                            let _ = self.storage.delete_endpoint(EndpointId(id)).await;
                        }
                    }
                }
                DDL_CREATE_SECURITY_MAP => {
                    if let Ok(entry) = SecurityMapEntry::from_bytes(entry_bytes) {
                        if !have_security_maps.contains(&entry.id.0) {
                            let _ = self.storage.store_security_map(&entry).await;
                            have_security_maps.insert(entry.id.0);
                        }
                    }
                }
                DDL_DROP_SECURITY_MAP => {
                    if entry_bytes.len() >= 4 {
                        let id = u32::from_le_bytes([
                            entry_bytes[0],
                            entry_bytes[1],
                            entry_bytes[2],
                            entry_bytes[3],
                        ]);
                        if have_security_maps.remove(&id) {
                            let _ = self.storage.delete_security_map(SecurityMapId(id)).await;
                        }
                    }
                }
                _ => {
                    // Unknown DDL type byte. Skip rather than panic so the
                    // catalog tolerates forward-compatible WAL records.
                }
            }
        }
        Ok(())
    }

    /// Loads all catalog data from storage into cache and recovers OID counter.
    /// Runs all 4 storage scans concurrently to minimize cold-start latency.
    pub async fn load(&self) -> Result<()> {
        self.cache.invalidate_all();

        let (
            databases,
            schemas,
            tables,
            indexes,
            streaming_jobs,
            external_sources,
            external_sinks,
            publications,
            publication_tables,
            subscriptions,
            endpoints,
            security_maps,
            sequences,
            views,
            mviews,
            functions,
            comments,
            aggregates,
            procedures,
            schedules,
            triggers,
            pipelines,
            event_handlers,
            version_tags,
        ) = tokio::try_join!(
            self.storage.load_databases(),
            self.storage.load_schemas(),
            self.storage.load_tables(),
            self.storage.load_indexes(),
            self.storage.load_streaming_jobs(),
            self.storage.load_external_sources(),
            self.storage.load_external_sinks(),
            self.storage.load_publications(),
            self.storage.load_publication_tables(),
            self.storage.load_subscriptions(),
            self.storage.load_endpoints(),
            self.storage.load_security_maps(),
            self.storage.load_sequences(),
            self.storage.load_views(),
            self.storage.load_mviews(),
            self.storage.load_functions(),
            self.storage.load_comments(),
            self.storage.load_aggregates(),
            self.storage.load_procedures(),
            self.storage.load_schedules(),
            self.storage.load_triggers(),
            self.storage.load_pipelines(),
            self.storage.load_event_handlers(),
            self.storage.load_version_tags(),
        )?;

        let mut max_oid: u32 = USER_OID_START;

        for db in databases {
            if db.id.0 >= max_oid {
                max_oid = db.id.0 + 1;
            }
            self.cache.put_database(db);
        }

        for schema in schemas {
            if schema.id.0 >= max_oid {
                max_oid = schema.id.0 + 1;
            }
            self.cache.put_schema(schema);
        }

        for table in tables {
            if table.id.0 >= max_oid {
                max_oid = table.id.0 + 1;
            }
            self.cache.put_table(table);
        }

        for index in indexes {
            if index.id.0 >= max_oid {
                max_oid = index.id.0 + 1;
            }
            self.cache.put_index(index);
        }

        for job in streaming_jobs {
            if job.id.0 >= max_oid {
                max_oid = job.id.0 + 1;
            }
            self.cache.put_streaming_job(job);
        }

        for src in external_sources {
            if src.id.0 >= max_oid {
                max_oid = src.id.0 + 1;
            }
            self.cache.put_external_source(src);
        }

        for sink in external_sinks {
            if sink.id.0 >= max_oid {
                max_oid = sink.id.0 + 1;
            }
            self.cache.put_external_sink(sink);
        }

        for pubn in publications {
            if pubn.id.0 >= max_oid {
                max_oid = pubn.id.0 + 1;
            }
            self.cache.put_publication(pubn);
        }

        for pt in publication_tables {
            if pt.id >= max_oid {
                max_oid = pt.id + 1;
            }
            self.cache.put_publication_table(pt);
        }

        for sub in subscriptions {
            if sub.id.0 >= max_oid {
                max_oid = sub.id.0 + 1;
            }
            self.cache.put_subscription(sub);
        }

        for ep in endpoints {
            if ep.id.0 >= max_oid {
                max_oid = ep.id.0 + 1;
            }
            self.cache.put_endpoint(ep);
        }

        for sm in security_maps {
            if sm.id.0 >= max_oid {
                max_oid = sm.id.0 + 1;
            }
            self.cache.put_security_map(sm);
        }

        {
            let mut by_name = self.sequences_by_name.write();
            let mut by_id = self.sequences_by_id.write();
            by_name.clear();
            by_id.clear();
            for seq in sequences {
                if seq.id >= max_oid {
                    max_oid = seq.id + 1;
                }
                let live = Arc::new(crate::sequence::LiveSequence::from_entry(&seq));
                by_name.insert((seq.schema_id.0, seq.name.clone()), Arc::clone(&live));
                by_id.insert(seq.id, live);
            }
        }

        {
            let mut by_name = self.views_by_name.write();
            let mut by_id = self.views_by_id.write();
            by_name.clear();
            by_id.clear();
            for view in views {
                if view.id >= max_oid {
                    max_oid = view.id + 1;
                }
                let entry = Arc::new(view);
                by_name.insert((entry.schema_id.0, entry.name.clone()), Arc::clone(&entry));
                by_id.insert(entry.id, entry);
            }
        }

        {
            let mut by_name = self.mviews_by_name.write();
            let mut by_id = self.mviews_by_id.write();
            by_name.clear();
            by_id.clear();
            for mview in mviews {
                if mview.id >= max_oid {
                    max_oid = mview.id + 1;
                }
                let entry = Arc::new(mview);
                by_name.insert((entry.schema_id.0, entry.name.clone()), Arc::clone(&entry));
                by_id.insert(entry.id, entry);
            }
        }

        {
            let mut by_name = self.functions_by_name.write();
            let mut by_id = self.functions_by_id.write();
            by_name.clear();
            by_id.clear();
            for func in functions {
                if func.id >= max_oid {
                    max_oid = func.id + 1;
                }
                let entry = Arc::new(func);
                by_name
                    .entry(entry.name.clone())
                    .or_default()
                    .push(Arc::clone(&entry));
                by_id.insert(entry.id, entry);
            }
        }

        {
            let mut map = self.comments.write();
            map.clear();
            for c in comments {
                if c.id >= max_oid {
                    max_oid = c.id + 1;
                }
                let key = (c.object_type, c.object_name.clone(), c.column_name.clone());
                map.insert(key, Arc::new(c));
            }
        }

        {
            let mut by_name = self.aggregates_by_name.write();
            let mut by_id = self.aggregates_by_id.write();
            by_name.clear();
            by_id.clear();
            for agg in aggregates {
                if agg.id >= max_oid {
                    max_oid = agg.id + 1;
                }
                let entry = Arc::new(agg);
                by_name
                    .entry(entry.name.clone())
                    .or_default()
                    .push(Arc::clone(&entry));
                by_id.insert(entry.id, entry);
            }
        }

        {
            let mut by_name = self.procedures_by_name.write();
            let mut by_id = self.procedures_by_id.write();
            by_name.clear();
            by_id.clear();
            for proc in procedures {
                if proc.id >= max_oid {
                    max_oid = proc.id + 1;
                }
                let entry = Arc::new(proc);
                by_name
                    .entry(entry.name.clone())
                    .or_default()
                    .push(Arc::clone(&entry));
                by_id.insert(entry.id, entry);
            }
        }

        {
            let mut by_name = self.schedules_by_name.write();
            let mut by_id = self.schedules_by_id.write();
            by_name.clear();
            by_id.clear();
            for sched in schedules {
                if sched.id >= max_oid {
                    max_oid = sched.id + 1;
                }
                let entry = Arc::new(sched);
                by_name.insert(entry.name.clone(), Arc::clone(&entry));
                by_id.insert(entry.id, entry);
            }
        }

        {
            let mut by_id = self.triggers_by_id.write();
            let mut by_table = self.triggers_by_table.write();
            by_id.clear();
            by_table.clear();
            for trig in triggers {
                if trig.id >= max_oid {
                    max_oid = trig.id + 1;
                }
                let entry = Arc::new(trig);
                by_table
                    .entry(entry.table_id)
                    .or_default()
                    .push(Arc::clone(&entry));
                by_id.insert(entry.id, entry);
            }
        }

        {
            let mut by_name = self.pipelines_by_name.write();
            let mut by_id = self.pipelines_by_id.write();
            by_name.clear();
            by_id.clear();
            for pipe in pipelines {
                if pipe.id >= max_oid {
                    max_oid = pipe.id + 1;
                }
                let entry = Arc::new(pipe);
                by_name.insert(entry.name.clone(), Arc::clone(&entry));
                by_id.insert(entry.id, entry);
            }
        }

        {
            let mut by_name = self.event_handlers_by_name.write();
            let mut by_id = self.event_handlers_by_id.write();
            by_name.clear();
            by_id.clear();
            for handler in event_handlers {
                if handler.id >= max_oid {
                    max_oid = handler.id + 1;
                }
                let entry = Arc::new(handler);
                by_name.insert(entry.name.clone(), Arc::clone(&entry));
                by_id.insert(entry.id, entry);
            }
        }

        {
            let mut by_name = self.version_tags_by_name.write();
            let mut by_id = self.version_tags_by_id.write();
            by_name.clear();
            by_id.clear();
            for tag in version_tags {
                if tag.id >= max_oid {
                    max_oid = tag.id + 1;
                }
                let entry = Arc::new(tag);
                by_name.insert(entry.name.clone(), Arc::clone(&entry));
                by_id.insert(entry.id, entry);
            }
        }

        self.oid_allocator.reset(max_oid);
        Ok(())
    }

    /// Allocates the next OID.
    pub fn next_oid(&self) -> Oid {
        self.oid_allocator.next()
    }

    /// Allocates a fresh (heap_file_id, fsm_file_id) pair. The table-rewrite
    /// engine uses this to build the new heap in side files before swapping
    /// the catalog, so the old heap stays intact until the rewrite commits.
    pub fn alloc_heap_files(&self) -> (u32, u32) {
        self.storage.next_heap_file_id()
    }

    /// Creates a NameResolver bound to the given database and search path.
    pub fn resolver(&self, database_id: DatabaseId, search_path: Vec<String>) -> NameResolver {
        NameResolver::new(
            database_id,
            search_path,
            Arc::clone(&self.cache),
            Arc::clone(&self.storage),
        )
    }

    // -----------------------------------------------------------------------
    // Database operations
    // -----------------------------------------------------------------------

    pub async fn create_database(&self, name: &str, owner: &str) -> Result<DatabaseId> {
        if self.cache.get_database_by_name(name).is_some() {
            return Err(ZyronError::DatabaseAlreadyExists(name.to_string()));
        }

        let id = DatabaseId(self.oid_allocator.next());
        let now = current_timestamp();
        let entry = DatabaseEntry {
            id,
            name: name.to_string(),
            owner: owner.to_string(),
            created_at: now,
        };

        self.log_ddl(DDL_CREATE_DATABASE, &entry.to_bytes())?;
        self.storage.store_database(&entry).await?;
        self.cache.put_database(entry);
        Ok(id)
    }

    pub async fn drop_database(&self, name: &str) -> Result<()> {
        let db = self
            .cache
            .get_database_by_name(name)
            .ok_or_else(|| ZyronError::DatabaseNotFound(name.to_string()))?;

        let id = db.id;
        let mut payload = vec![0u8; 4];
        payload[..4].copy_from_slice(&id.0.to_le_bytes());
        self.log_ddl(DDL_DROP_DATABASE, &payload)?;
        self.storage.delete_database(id).await?;
        self.cache.invalidate_database(id);
        Ok(())
    }

    pub fn get_database(&self, name: &str) -> Result<Arc<DatabaseEntry>> {
        self.cache
            .get_database_by_name(name)
            .ok_or_else(|| ZyronError::DatabaseNotFound(name.to_string()))
    }

    // -----------------------------------------------------------------------
    // Schema operations
    // -----------------------------------------------------------------------

    pub async fn create_schema(
        &self,
        db_id: DatabaseId,
        name: &str,
        owner: &str,
    ) -> Result<SchemaId> {
        if name == SYSTEM_SCHEMA_NAME {
            return Err(ZyronError::PermissionDenied(format!(
                "schema name `{}` is reserved for Zyron internals",
                SYSTEM_SCHEMA_NAME
            )));
        }
        if self.cache.get_schema_by_name(db_id, name).is_some() {
            return Err(ZyronError::SchemaAlreadyExists(name.to_string()));
        }

        let id = SchemaId(self.oid_allocator.next());
        let entry = SchemaEntry {
            id,
            database_id: db_id,
            name: name.to_string(),
            owner: owner.to_string(),
        };

        self.log_ddl(DDL_CREATE_SCHEMA, &entry.to_bytes())?;
        self.storage.store_schema(&entry).await?;
        self.cache.put_schema(entry);
        Ok(id)
    }

    pub async fn drop_schema(&self, db_id: DatabaseId, name: &str) -> Result<()> {
        if name == SYSTEM_SCHEMA_NAME {
            return Err(ZyronError::PermissionDenied(format!(
                "schema `{}` is reserved for Zyron internals and cannot be dropped",
                SYSTEM_SCHEMA_NAME
            )));
        }
        let schema = self
            .cache
            .get_schema_by_name(db_id, name)
            .ok_or_else(|| ZyronError::SchemaNotFound(name.to_string()))?;

        let id = schema.id;
        let mut payload = vec![0u8; 4];
        payload[..4].copy_from_slice(&id.0.to_le_bytes());
        self.log_ddl(DDL_DROP_SCHEMA, &payload)?;
        self.storage.delete_schema(id).await?;
        self.cache.invalidate_schema(id);
        Ok(())
    }

    pub fn get_schema(&self, db_id: DatabaseId, name: &str) -> Result<Arc<SchemaEntry>> {
        self.cache
            .get_schema_by_name(db_id, name)
            .ok_or_else(|| ZyronError::SchemaNotFound(name.to_string()))
    }

    // -----------------------------------------------------------------------
    // Table operations
    // -----------------------------------------------------------------------

    pub async fn create_table(
        &self,
        schema_id: SchemaId,
        name: &str,
        column_defs: &[ColumnDef],
        table_constraints: &[TableConstraint],
    ) -> Result<TableId> {
        if schema_id == SYSTEM_SCHEMA_ID {
            return Err(ZyronError::PermissionDenied(format!(
                "schema `{}` is reserved for Zyron internals and cannot hold user tables",
                SYSTEM_SCHEMA_NAME
            )));
        }
        if self.cache.get_table_by_name(schema_id, name).is_some() {
            return Err(ZyronError::TableAlreadyExists(name.to_string()));
        }

        if column_defs.len() > u16::MAX as usize {
            return Err(ZyronError::Internal(format!(
                "table has {} columns, max is {}",
                column_defs.len(),
                u16::MAX
            )));
        }

        // Validate no duplicate column names
        let mut seen_names = HashSet::with_capacity(column_defs.len());
        for def in column_defs {
            if !seen_names.insert(&def.name) {
                return Err(ZyronError::Internal(format!(
                    "duplicate column name: {}",
                    def.name
                )));
            }
        }

        let table_id = TableId(self.oid_allocator.next());
        let (heap_file_id, fsm_file_id) = self.storage.next_heap_file_id();
        let now = current_timestamp();

        // Convert parser ColumnDefs to catalog ColumnEntries
        let columns = convert_column_defs(table_id, column_defs)?;

        // Convert parser constraints to catalog ConstraintEntries
        let mut constraints = convert_table_constraints(table_constraints, &columns)?;

        // Resolve table-level FOREIGN KEY targets to catalog ids. The conversion
        // above preserves order, so each ForeignKey table constraint lines up
        // with its produced entry. The referenced table and columns must exist.
        for (tc, entry) in table_constraints.iter().zip(constraints.iter_mut()) {
            if let TableConstraintKind::ForeignKey {
                ref_table,
                ref_columns,
                ..
            } = &tc.kind
            {
                let ref_entry = self
                    .cache
                    .get_table_by_name(schema_id, ref_table)
                    .ok_or_else(|| ZyronError::TableNotFound(ref_table.clone()))?;
                let mut ref_col_ids = Vec::with_capacity(ref_columns.len());
                for rc in ref_columns {
                    let c = ref_entry
                        .columns
                        .iter()
                        .find(|c| c.name == *rc)
                        .ok_or_else(|| ZyronError::ColumnNotFound(format!("{ref_table}.{rc}")))?;
                    ref_col_ids.push(c.id);
                }
                entry.ref_table_id = Some(ref_entry.id);
                entry.ref_columns = ref_col_ids;
            }
        }

        // Extract inline column constraints (PrimaryKey, Unique, NotNull, Check, References)
        for (i, col_def) in column_defs.iter().enumerate() {
            for cc in &col_def.constraints {
                let col_id = ColumnId(i as u16);
                match cc {
                    ColumnConstraint::PrimaryKey => {
                        constraints.push(ConstraintEntry {
                            name: format!("pk_{}_{}", name, col_def.name),
                            constraint_type: ConstraintType::PrimaryKey,
                            columns: vec![col_id],
                            ref_table_id: None,
                            ref_columns: vec![],
                            check_expr: None,
                            on_delete: ReferentialAction::NoAction,
                            on_update: ReferentialAction::NoAction,
                            enforced: true,
                            on_violation: ConstraintViolationAction::Fail,
                            quarantine_table_id: None,
                        });
                    }
                    ColumnConstraint::Unique => {
                        constraints.push(ConstraintEntry {
                            name: format!("uq_{}_{}", name, col_def.name),
                            constraint_type: ConstraintType::Unique,
                            columns: vec![col_id],
                            ref_table_id: None,
                            ref_columns: vec![],
                            check_expr: None,
                            on_delete: ReferentialAction::NoAction,
                            on_update: ReferentialAction::NoAction,
                            enforced: true,
                            on_violation: ConstraintViolationAction::Fail,
                            quarantine_table_id: None,
                        });
                    }
                    ColumnConstraint::NotNull => {
                        constraints.push(ConstraintEntry {
                            name: format!("nn_{}_{}", name, col_def.name),
                            constraint_type: ConstraintType::NotNull,
                            columns: vec![col_id],
                            ref_table_id: None,
                            ref_columns: vec![],
                            check_expr: None,
                            on_delete: ReferentialAction::NoAction,
                            on_update: ReferentialAction::NoAction,
                            enforced: true,
                            on_violation: ConstraintViolationAction::Fail,
                            quarantine_table_id: None,
                        });
                    }
                    ColumnConstraint::Check(expr) => {
                        constraints.push(ConstraintEntry {
                            name: format!("ck_{}_{}", name, col_def.name),
                            constraint_type: ConstraintType::Check,
                            columns: vec![col_id],
                            ref_table_id: None,
                            ref_columns: vec![],
                            check_expr: Some(zyron_parser::expr_to_sql(expr)),
                            on_delete: ReferentialAction::NoAction,
                            on_update: ReferentialAction::NoAction,
                            enforced: true,
                            on_violation: ConstraintViolationAction::Fail,
                            quarantine_table_id: None,
                        });
                    }
                    ColumnConstraint::References {
                        table: ref_table,
                        column: ref_column,
                        on_delete,
                        on_update,
                    } => {
                        // Resolve the referenced table and column to catalog ids
                        // so foreign keys can be enforced. The target must exist.
                        let ref_entry = self
                            .cache
                            .get_table_by_name(schema_id, ref_table)
                            .ok_or_else(|| ZyronError::TableNotFound(ref_table.clone()))?;
                        let ref_col = ref_entry
                            .columns
                            .iter()
                            .find(|c| c.name == *ref_column)
                            .ok_or_else(|| {
                                ZyronError::ColumnNotFound(format!("{ref_table}.{ref_column}"))
                            })?;
                        constraints.push(ConstraintEntry {
                            name: format!("fk_{}_{}", name, col_def.name),
                            constraint_type: ConstraintType::ForeignKey,
                            columns: vec![col_id],
                            ref_table_id: Some(ref_entry.id),
                            ref_columns: vec![ref_col.id],
                            check_expr: None,
                            on_delete: map_ref_action(*on_delete),
                            on_update: map_ref_action(*on_update),
                            enforced: true,
                            on_violation: ConstraintViolationAction::Fail,
                            quarantine_table_id: None,
                        });
                    }
                    ColumnConstraint::Default(_) => {
                        // Default values are already captured in ColumnEntry.default_expr
                    }
                }
            }
        }

        let entry = TableEntry {
            id: table_id,
            schema_id,
            name: name.to_string(),
            heap_file_id,
            fsm_file_id,
            columns,
            constraints,
            created_at: now,
            versioning_enabled: false,
            scd_type: None,
            system_versioned: false,
            history_table_id: None,
            cdf_enabled: false,
            cdf_retention_days: 0,
            lifecycle: Default::default(),
            columnar: Default::default(),
            dropped_at: None,
            expectations: Vec::new(),
            time_travel_retention_secs: 0,
            lake: Default::default(),
            cluster: Default::default(),
            foreign: Default::default(),
        };

        self.log_ddl(DDL_CREATE_TABLE, &entry.to_bytes())?;
        self.storage.store_table(&entry).await?;
        self.cache.put_table(entry);
        Ok(table_id)
    }

    /// Registers a table whose rows live on a peer.
    ///
    /// No heap file and no free-space map are allocated, because there is
    /// nothing local to store: file id zero is the reserved "no file" value,
    /// so any path that tried to open storage for this table fails visibly
    /// rather than creating an empty heap nobody writes to.
    ///
    /// Constraints are not accepted. A declared constraint that nothing
    /// enforces is a lie, and enforcement belongs to the node that owns the
    /// rows.
    pub async fn create_foreign_table(
        &self,
        schema_id: SchemaId,
        name: &str,
        column_defs: &[ColumnDef],
        peer: &str,
        remote_table: &str,
    ) -> Result<TableId> {
        if schema_id == SYSTEM_SCHEMA_ID {
            return Err(ZyronError::PermissionDenied(format!(
                "schema `{}` is reserved for Zyron internals and cannot hold user tables",
                SYSTEM_SCHEMA_NAME
            )));
        }
        if self.cache.get_table_by_name(schema_id, name).is_some() {
            return Err(ZyronError::TableAlreadyExists(name.to_string()));
        }
        if column_defs.is_empty() {
            return Err(ZyronError::Internal(format!(
                "foreign table `{}` declares no column",
                name
            )));
        }
        if column_defs.len() > u16::MAX as usize {
            return Err(ZyronError::Internal(format!(
                "table has {} columns, max is {}",
                column_defs.len(),
                u16::MAX
            )));
        }
        let mut seen_names = HashSet::with_capacity(column_defs.len());
        for def in column_defs {
            if !seen_names.insert(&def.name) {
                return Err(ZyronError::Internal(format!(
                    "duplicate column name: {}",
                    def.name
                )));
            }
        }

        let table_id = TableId(self.oid_allocator.next());
        let columns = convert_column_defs(table_id, column_defs)?;
        let entry = TableEntry {
            id: table_id,
            schema_id,
            name: name.to_string(),
            heap_file_id: 0,
            fsm_file_id: 0,
            columns,
            constraints: Vec::new(),
            created_at: current_timestamp(),
            versioning_enabled: false,
            scd_type: None,
            system_versioned: false,
            history_table_id: None,
            cdf_enabled: false,
            cdf_retention_days: 0,
            lifecycle: Default::default(),
            columnar: Default::default(),
            dropped_at: None,
            expectations: Vec::new(),
            time_travel_retention_secs: 0,
            lake: Default::default(),
            cluster: Default::default(),
            foreign: crate::schema::ForeignConfig {
                peer: peer.to_string(),
                table: remote_table.to_string(),
            },
        };
        self.log_ddl(DDL_CREATE_TABLE, &entry.to_bytes())?;
        self.storage.store_table(&entry).await?;
        self.cache.put_table(entry);
        Ok(table_id)
    }

    pub async fn drop_table(&self, schema_id: SchemaId, name: &str) -> Result<DropOutcome> {
        if schema_id == SYSTEM_SCHEMA_ID {
            return Err(ZyronError::PermissionDenied(format!(
                "tables in `{}` are reserved for Zyron internals and cannot be dropped",
                SYSTEM_SCHEMA_NAME
            )));
        }
        let table = self
            .cache
            .get_table_by_name(schema_id, name)
            .ok_or_else(|| ZyronError::TableNotFound(name.to_string()))?;

        let id = table.id;
        let heap_file_id = table.heap_file_id;
        let fsm_file_id = table.fsm_file_id;

        // Recycle bin: tables configured with a recycle window are hidden from
        // lookups but kept intact so UNDROP can restore them. The reaper purges
        // them physically once the window elapses. Files and indexes stay live.
        if table.lifecycle.recycle_window_seconds > 0 {
            let mut entry = (*table).clone();
            entry.dropped_at = Some(current_timestamp());
            self.update_table(entry).await?;
            return Ok(DropOutcome {
                soft_dropped: true,
                table_id: id,
                heap_file_id,
                fsm_file_id,
            });
        }

        // Cascade dependent catalog rows before removing the table entry.
        // Indexes and comments are stored as their own rows, so dropping only
        // the table entry would orphan them. Expectations live on the table
        // entry itself and are removed with it. Every removal is logged in one
        // transaction with a single durable flush, so drop latency is one fsync
        // regardless of how many indexes and comments the table has.
        let indexes = self.cache.get_indexes_for_table(id);
        let comment_ids = self.stale_comment_ids(name);

        let mut ddl_records: Vec<(u8, Vec<u8>)> =
            Vec::with_capacity(indexes.len() + comment_ids.len() + 1);
        for idx in &indexes {
            ddl_records.push((DDL_DROP_INDEX, idx.id.0.to_le_bytes().to_vec()));
        }
        for cid in &comment_ids {
            ddl_records.push((DDL_DROP_COMMENT, cid.to_le_bytes().to_vec()));
        }
        ddl_records.push((DDL_DROP_TABLE, id.0.to_le_bytes().to_vec()));
        self.log_ddl_batch(&ddl_records)?;

        // Apply the storage and cache mutations now that the batch is durable.
        // Storage deletes are buffered and the WAL is the source of truth, so a
        // crash here is redone as a unit on recovery.
        for idx in &indexes {
            self.storage.delete_index(idx.id).await?;
            self.cache.invalidate_index(idx.id);
        }
        self.purge_table_comments(&comment_ids, name).await?;
        self.storage.delete_table(id).await?;
        self.cache.invalidate_table(id);
        Ok(DropOutcome {
            soft_dropped: false,
            table_id: id,
            heap_file_id,
            fsm_file_id,
        })
    }

    /// Collects the ids of table-level and column-level comments attached to a
    /// table so a cascading drop can log their removal in one batch. Comments
    /// are keyed by object kind and name (table kind 0, column kind 1), both
    /// addressed by table name.
    fn stale_comment_ids(&self, table_name: &str) -> Vec<u32> {
        self.comments
            .read()
            .values()
            .filter(|c| (c.object_type == 0 || c.object_type == 1) && c.object_name == table_name)
            .map(|c| c.id)
            .collect()
    }

    /// Removes a dropped table's comments from storage and the in-memory map
    /// after their removal has been logged. Storage deletes are buffered, so no
    /// per-comment flush happens here.
    async fn purge_table_comments(&self, ids: &[u32], table_name: &str) -> Result<()> {
        for id in ids {
            self.storage.delete_comment(*id).await?;
        }
        if !ids.is_empty() {
            self.comments.write().retain(|_, c| {
                !((c.object_type == 0 || c.object_type == 1) && c.object_name == table_name)
            });
        }
        Ok(())
    }

    /// Restores a soft-dropped table from the recycle bin by clearing its drop
    /// marker. Fails if a live table already holds the name, or if no recycled
    /// table with the name exists. Storage and indexes were retained on the
    /// soft drop, so the restored table is immediately queryable.
    pub async fn undrop_table(&self, schema_id: SchemaId, name: &str) -> Result<()> {
        if self.cache.get_table_by_name(schema_id, name).is_some() {
            return Err(ZyronError::TableAlreadyExists(format!(
                "{} exists as a live table, rename or drop it before restoring the recycled copy",
                name
            )));
        }
        // Cache first, then a storage scan in case the recycle-bin entry was
        // evicted from the bounded cache.
        let dropped = match self.cache.get_dropped_table_by_name(schema_id, name) {
            Some(e) => (*e).clone(),
            None => self
                .storage
                .load_tables()
                .await?
                .into_iter()
                .find(|t| t.schema_id == schema_id && t.name == name && t.dropped_at.is_some())
                .ok_or_else(|| {
                    ZyronError::TableNotFound(format!(
                        "no recycled table named {} to restore",
                        name
                    ))
                })?,
        };
        let mut entry = dropped;
        entry.dropped_at = None;
        self.update_table(entry).await?;
        Ok(())
    }

    /// Physically purges a soft-dropped table once its recycle window elapses.
    /// Drops the table's index catalog entries and the table entry, then returns
    /// the removed entry so the caller can reclaim the backing heap/index files
    /// and in-memory index handles. Returns None if the table is not soft-dropped
    /// (already restored or already purged), making the reaper idempotent.
    pub async fn finalize_dropped_table(&self, id: TableId) -> Result<Option<Arc<TableEntry>>> {
        let entry = match self.cache.get_table_including_dropped(id) {
            Some(e) if e.dropped_at.is_some() => e,
            _ => return Ok(None),
        };

        // Log the table and its dependent index/comment removals in one
        // transaction with a single durable flush, then apply the buffered
        // storage and cache mutations.
        let indexes = self.cache.get_indexes_for_table(id);
        let comment_ids = self.stale_comment_ids(&entry.name);

        let mut ddl_records: Vec<(u8, Vec<u8>)> =
            Vec::with_capacity(indexes.len() + comment_ids.len() + 1);
        for idx in &indexes {
            ddl_records.push((DDL_DROP_INDEX, idx.id.0.to_le_bytes().to_vec()));
        }
        for cid in &comment_ids {
            ddl_records.push((DDL_DROP_COMMENT, cid.to_le_bytes().to_vec()));
        }
        ddl_records.push((DDL_DROP_TABLE, id.0.to_le_bytes().to_vec()));
        self.log_ddl_batch(&ddl_records)?;

        for idx in &indexes {
            self.storage.delete_index(idx.id).await?;
            self.cache.invalidate_index(idx.id);
        }
        self.purge_table_comments(&comment_ids, &entry.name).await?;
        self.storage.delete_table(id).await?;
        self.cache.invalidate_table(id);
        Ok(Some(entry))
    }

    /// Returns all tables currently in the recycle bin (soft-dropped). Used by
    /// the background reaper to find tables whose window has elapsed.
    pub fn list_dropped_tables(&self) -> Vec<Arc<TableEntry>> {
        self.cache.list_dropped_tables()
    }

    /// Returns every foreign-key constraint in any table that references the
    /// given parent table, paired with the child table that owns it. Used by
    /// the executor to enforce ON DELETE / ON UPDATE actions when a parent row
    /// changes. Scans the cached table set, which holds the full catalog.
    pub fn referencing_constraints(
        &self,
        parent_id: TableId,
    ) -> Vec<(Arc<TableEntry>, ConstraintEntry)> {
        let mut out = Vec::new();
        for table in self.cache.list_all_tables() {
            for con in &table.constraints {
                if con.constraint_type == ConstraintType::ForeignKey
                    && con.ref_table_id == Some(parent_id)
                {
                    out.push((Arc::clone(&table), con.clone()));
                }
            }
        }
        out
    }

    pub fn get_table(&self, schema_id: SchemaId, name: &str) -> Result<Arc<TableEntry>> {
        self.cache
            .get_table_by_name(schema_id, name)
            .ok_or_else(|| ZyronError::TableNotFound(name.to_string()))
    }

    pub fn get_table_by_id(&self, id: TableId) -> Result<Arc<TableEntry>> {
        self.cache
            .get_table(id)
            .ok_or_else(|| ZyronError::TableNotFound(format!("id={}", id.0)))
    }

    pub fn get_schema_by_id(&self, id: SchemaId) -> Result<Arc<SchemaEntry>> {
        self.cache
            .get_schema(id)
            .ok_or_else(|| ZyronError::SchemaNotFound(format!("id={}", id.0)))
    }

    pub fn list_tables(&self, schema_id: SchemaId) -> Vec<Arc<TableEntry>> {
        self.cache.list_tables(schema_id)
    }

    /// Returns all cached tables across all schemas.
    pub fn list_all_tables(&self) -> Vec<Arc<TableEntry>> {
        self.cache.list_all_tables()
    }

    // -----------------------------------------------------------------------
    // Index operations
    // -----------------------------------------------------------------------

    pub async fn create_index(
        &self,
        table_id: TableId,
        schema_id: SchemaId,
        name: &str,
        column_names: &[String],
        unique: bool,
        index_type: IndexType,
    ) -> Result<IndexId> {
        // Check for duplicate index name in cache
        let existing = self.cache.get_indexes_for_table(table_id);
        for idx in &existing {
            if idx.name == name {
                return Err(ZyronError::IndexAlreadyExists(name.to_string()));
            }
        }

        let table = self.get_table_by_id(table_id)?;
        let index_id = IndexId(self.oid_allocator.next());
        let index_file_id = self.storage.next_index_file_id();

        // Resolve column names to ColumnIds
        let mut columns = Vec::with_capacity(column_names.len());
        for (ordinal, col_name) in column_names.iter().enumerate() {
            let col = table
                .columns
                .iter()
                .find(|c| c.name == *col_name)
                .ok_or_else(|| ZyronError::ColumnNotFound(col_name.clone()))?;
            columns.push(IndexColumnEntry {
                column_id: col.id,
                ordinal: ordinal as u16,
                descending: false,
            });
        }

        let entry = IndexEntry {
            id: index_id,
            table_id,
            schema_id,
            name: name.to_string(),
            columns,
            unique,
            index_file_id,
            index_type,
            parameters: None,
        };

        self.log_ddl(DDL_CREATE_INDEX, &entry.to_bytes())?;
        self.storage.store_index(&entry).await?;
        self.cache.put_index(entry);
        Ok(index_id)
    }

    /// Creates a B+tree index recording a sort direction per key column.
    ///
    /// Only a B+tree has an order to record. Every other index kind is
    /// unordered, so `create_index` covers those and this covers the one
    /// that can answer an ORDER BY without a sort.
    pub async fn create_btree_index(
        &self,
        table_id: TableId,
        schema_id: SchemaId,
        name: &str,
        columns: &[(String, bool)],
        unique: bool,
    ) -> Result<IndexId> {
        let existing = self.cache.get_indexes_for_table(table_id);
        for idx in &existing {
            if idx.name == name {
                return Err(ZyronError::IndexAlreadyExists(name.to_string()));
            }
        }

        let table = self.get_table_by_id(table_id)?;
        let index_id = IndexId(self.oid_allocator.next());
        let index_file_id = self.storage.next_index_file_id();

        let mut resolved = Vec::with_capacity(columns.len());
        for (ordinal, (col_name, descending)) in columns.iter().enumerate() {
            let col = table
                .columns
                .iter()
                .find(|c| c.name == *col_name)
                .ok_or_else(|| ZyronError::ColumnNotFound(col_name.clone()))?;
            resolved.push(IndexColumnEntry {
                column_id: col.id,
                ordinal: ordinal as u16,
                descending: *descending,
            });
        }

        let entry = IndexEntry {
            id: index_id,
            table_id,
            schema_id,
            name: name.to_string(),
            columns: resolved,
            unique,
            index_file_id,
            index_type: IndexType::BTree,
            parameters: None,
        };

        self.log_ddl(DDL_CREATE_INDEX, &entry.to_bytes())?;
        self.storage.store_index(&entry).await?;
        self.cache.put_index(entry);
        Ok(index_id)
    }

    /// Like create_index, but also stores the opaque parameters blob on the
    /// index entry. Used by spatial and vector indexes that persist tuning
    /// options (dims, srid, HNSW config, etc.) so startup recovery can
    /// reconstruct live state without re-reading the CREATE statement.
    pub async fn create_index_with_params(
        &self,
        table_id: TableId,
        schema_id: SchemaId,
        name: &str,
        column_names: &[String],
        unique: bool,
        index_type: IndexType,
        parameters: Option<Vec<u8>>,
    ) -> Result<IndexId> {
        let existing = self.cache.get_indexes_for_table(table_id);
        for idx in &existing {
            if idx.name == name {
                return Err(ZyronError::IndexAlreadyExists(name.to_string()));
            }
        }

        let table = self.get_table_by_id(table_id)?;
        let index_id = IndexId(self.oid_allocator.next());
        let index_file_id = self.storage.next_index_file_id();

        let mut columns = Vec::with_capacity(column_names.len());
        for (ordinal, col_name) in column_names.iter().enumerate() {
            let col = table
                .columns
                .iter()
                .find(|c| c.name == *col_name)
                .ok_or_else(|| ZyronError::ColumnNotFound(col_name.clone()))?;
            columns.push(IndexColumnEntry {
                column_id: col.id,
                ordinal: ordinal as u16,
                descending: false,
            });
        }

        let entry = IndexEntry {
            id: index_id,
            table_id,
            schema_id,
            name: name.to_string(),
            columns,
            unique,
            index_file_id,
            index_type,
            parameters,
        };

        self.log_ddl(DDL_CREATE_INDEX, &entry.to_bytes())?;
        self.storage.store_index(&entry).await?;
        self.cache.put_index(entry);
        Ok(index_id)
    }

    pub async fn drop_index(&self, table_id: TableId, name: &str) -> Result<()> {
        let indexes = self.cache.get_indexes_for_table(table_id);
        let idx = indexes
            .iter()
            .find(|i| i.name == name)
            .ok_or_else(|| ZyronError::IndexNotFound(name.to_string()))?;

        let id = idx.id;
        let mut payload = vec![0u8; 4];
        payload[..4].copy_from_slice(&id.0.to_le_bytes());
        self.log_ddl(DDL_DROP_INDEX, &payload)?;
        self.storage.delete_index(id).await?;
        self.cache.invalidate_index(id);
        Ok(())
    }

    pub fn get_indexes_for_table(&self, table_id: TableId) -> Vec<Arc<IndexEntry>> {
        self.cache.get_indexes_for_table(table_id)
    }

    /// Renames an index. The index id, columns, and backing file are unchanged;
    /// only the catalog name is updated. The renamed entry is re-logged so a
    /// restart recovers the new name (CREATE INDEX recovery upserts by id).
    pub async fn rename_index(
        &self,
        table_id: TableId,
        old_name: &str,
        new_name: &str,
    ) -> Result<()> {
        let indexes = self.cache.get_indexes_for_table(table_id);
        if indexes.iter().any(|i| i.name == new_name) {
            return Err(ZyronError::IndexAlreadyExists(new_name.to_string()));
        }
        let idx = indexes
            .iter()
            .find(|i| i.name == old_name)
            .ok_or_else(|| ZyronError::IndexNotFound(old_name.to_string()))?;
        let mut entry = (**idx).clone();
        entry.name = new_name.to_string();
        self.log_ddl(DDL_CREATE_INDEX, &entry.to_bytes())?;
        self.storage.delete_index(entry.id).await?;
        self.storage.store_index(&entry).await?;
        self.cache.invalidate_index(entry.id);
        self.cache.put_index(entry);
        Ok(())
    }

    /// Returns the lock-free, pre-partitioned index snapshot for a table.
    /// DML operators consult this once per statement instead of hitting four
    /// separate catalog `RwLock` reads + allocations per batch.
    pub fn index_snapshot(&self, table_id: TableId) -> Arc<crate::cache::TableIndexSnapshot> {
        self.cache.index_snapshot(table_id)
    }

    // -----------------------------------------------------------------------
    // Streaming job operations
    // -----------------------------------------------------------------------

    pub async fn create_streaming_job(
        &self,
        mut entry: StreamingJobEntry,
    ) -> Result<StreamingJobId> {
        if self
            .cache
            .get_streaming_job_by_name(entry.source_schema_id, &entry.name)
            .is_some()
        {
            return Err(ZyronError::Internal(format!(
                "streaming job '{}' already exists",
                entry.name
            )));
        }

        if entry.id.0 == 0 {
            entry.id = StreamingJobId(self.oid_allocator.next());
        }

        let id = entry.id;
        self.log_ddl(DDL_CREATE_STREAMING_JOB, &entry.to_bytes())?;
        self.storage.store_streaming_job(&entry).await?;
        self.cache.put_streaming_job(entry);
        Ok(id)
    }

    pub fn get_streaming_job(
        &self,
        schema_id: SchemaId,
        name: &str,
    ) -> Option<Arc<StreamingJobEntry>> {
        self.cache.get_streaming_job_by_name(schema_id, name)
    }

    pub fn get_streaming_job_by_id(&self, id: StreamingJobId) -> Option<Arc<StreamingJobEntry>> {
        self.cache.get_streaming_job(id)
    }

    pub fn list_streaming_jobs(&self) -> Vec<Arc<StreamingJobEntry>> {
        self.cache.list_streaming_jobs()
    }

    pub async fn drop_streaming_job(&self, schema_id: SchemaId, name: &str) -> Result<()> {
        let job = self
            .cache
            .get_streaming_job_by_name(schema_id, name)
            .ok_or_else(|| ZyronError::Internal(format!("streaming job '{name}' not found")))?;

        let id = job.id;
        let mut payload = vec![0u8; 4];
        payload[..4].copy_from_slice(&id.0.to_le_bytes());
        self.log_ddl(DDL_DROP_STREAMING_JOB, &payload)?;
        self.storage.delete_streaming_job(id).await?;
        self.cache.invalidate_streaming_job(id);
        Ok(())
    }

    pub async fn update_streaming_job_status(
        &self,
        id: StreamingJobId,
        status: StreamingJobStatus,
        last_error: Option<String>,
    ) -> Result<()> {
        let current = self
            .cache
            .get_streaming_job(id)
            .ok_or_else(|| ZyronError::Internal("streaming job not found".to_string()))?;

        let mut updated = (*current).clone();
        updated.status = status;
        updated.last_error = last_error;

        self.log_ddl(DDL_ALTER_STREAMING_JOB, &updated.to_bytes())?;
        self.storage.update_streaming_job(&updated).await?;
        self.cache.invalidate_streaming_job(id);
        self.cache.put_streaming_job(updated);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Sequence operations
    // -----------------------------------------------------------------------

    /// Creates a sequence. The entry's `reserved` is set to `start -
    /// increment` so the first nextval returns `start`. Assigns an id when the
    /// entry carries 0. Errors when a sequence of the same name already exists
    /// in the schema.
    pub async fn create_sequence(&self, mut entry: SequenceEntry) -> Result<u32> {
        let key = (entry.schema_id.0, entry.name.clone());
        if self.sequences_by_name.read().contains_key(&key) {
            return Err(ZyronError::Internal(format!(
                "sequence '{}' already exists",
                entry.name
            )));
        }
        if entry.cache < 1 {
            entry.cache = 1;
        }
        if entry.id == 0 {
            entry.id = self.oid_allocator.next();
        }
        entry.reserved = entry.start.saturating_sub(entry.increment);

        let id = entry.id;
        self.log_ddl(DDL_CREATE_SEQUENCE, &entry.to_bytes())?;
        self.storage.store_sequence(&entry).await?;

        let live = Arc::new(crate::sequence::LiveSequence::from_entry(&entry));
        self.sequences_by_name
            .write()
            .insert(key, Arc::clone(&live));
        self.sequences_by_id.write().insert(id, live);
        Ok(id)
    }

    /// Resolves a live sequence by schema and name.
    pub fn get_sequence(
        &self,
        schema_id: SchemaId,
        name: &str,
    ) -> Option<Arc<crate::sequence::LiveSequence>> {
        self.sequences_by_name
            .read()
            .get(&(schema_id.0, name.to_string()))
            .map(Arc::clone)
    }

    /// Resolves a live sequence by id.
    pub fn get_sequence_by_id(&self, id: u32) -> Option<Arc<crate::sequence::LiveSequence>> {
        self.sequences_by_id.read().get(&id).map(Arc::clone)
    }

    /// Lists every live sequence.
    pub fn list_sequences(&self) -> Vec<Arc<crate::sequence::LiveSequence>> {
        self.sequences_by_name
            .read()
            .values()
            .map(Arc::clone)
            .collect()
    }

    /// Resolves a sequence by its bare name (the last dotted component) across
    /// all schemas. Used by nextval/currval/setval at execution time where the
    /// session schema is not threaded into the executor. Errors when the name
    /// is ambiguous across schemas or not found.
    pub fn find_sequence_by_name(&self, name: &str) -> Result<Arc<crate::sequence::LiveSequence>> {
        let bare = name.rsplit('.').next().unwrap_or(name);
        let map = self.sequences_by_name.read();
        let mut found: Option<Arc<crate::sequence::LiveSequence>> = None;
        for ((_, n), live) in map.iter() {
            if n == bare {
                if found.is_some() {
                    return Err(ZyronError::Internal(format!(
                        "sequence name '{bare}' is ambiguous across schemas; qualify it"
                    )));
                }
                found = Some(Arc::clone(live));
            }
        }
        found.ok_or_else(|| ZyronError::Internal(format!("sequence '{name}' not found")))
    }

    /// Drops a sequence and removes its durable entry.
    pub async fn drop_sequence(&self, schema_id: SchemaId, name: &str) -> Result<()> {
        let key = (schema_id.0, name.to_string());
        let live = self
            .sequences_by_name
            .read()
            .get(&key)
            .map(Arc::clone)
            .ok_or_else(|| ZyronError::Internal(format!("sequence '{name}' not found")))?;

        let id = live.id;
        self.log_ddl(DDL_DROP_SEQUENCE, &id.to_le_bytes())?;
        self.storage.delete_sequence(id).await?;
        self.sequences_by_name.write().remove(&key);
        self.sequences_by_id.write().remove(&id);
        Ok(())
    }

    /// Replaces a sequence definition. Builds a fresh entry from the supplied
    /// fields, persists it, and swaps the live manager so the next nextval
    /// reflects the new parameters. The cursor resets to the new `reserved`.
    pub async fn alter_sequence(&self, entry: SequenceEntry) -> Result<()> {
        let key = (entry.schema_id.0, entry.name.clone());
        let existing_id = self
            .sequences_by_name
            .read()
            .get(&key)
            .map(|s| s.id)
            .ok_or_else(|| ZyronError::Internal(format!("sequence '{}' not found", entry.name)))?;

        let mut entry = entry;
        entry.id = existing_id;
        if entry.cache < 1 {
            entry.cache = 1;
        }

        self.log_ddl(DDL_UPDATE_SEQUENCE, &entry.to_bytes())?;
        if !self.storage.update_sequence(&entry).await? {
            return Err(ZyronError::CatalogCorrupted(format!(
                "sequence '{}' row missing or undecodable on update",
                entry.name
            )));
        }

        let live = Arc::new(crate::sequence::LiveSequence::from_entry(&entry));
        self.sequences_by_name
            .write()
            .insert(key, Arc::clone(&live));
        self.sequences_by_id.write().insert(existing_id, live);
        Ok(())
    }

    /// Returns the next value of the sequence. Hands out from the cached block
    /// when possible; otherwise reserves and durably persists a new block
    /// before returning, so a crash skips at most `cache` values.
    pub async fn sequence_nextval(&self, schema_id: SchemaId, name: &str) -> Result<i64> {
        let live = self
            .get_sequence(schema_id, name)
            .ok_or_else(|| ZyronError::Internal(format!("sequence '{name}' not found")))?;
        self.nextval_on(&live).await
    }

    /// nextval against an already-resolved live sequence. The DEFAULT path
    /// resolves by id once and reuses the handle for every inserted row.
    pub async fn nextval_on(&self, live: &Arc<crate::sequence::LiveSequence>) -> Result<i64> {
        if let Some(v) = live.try_next() {
            return Ok(v);
        }
        let _gate = live.lock_refill().await;
        // Another session may have reserved a block while this one waited.
        if let Some(v) = live.try_next() {
            return Ok(v);
        }
        let (entry, slot) = live.plan_refill()?;
        self.log_sequence_reserve(&entry)?;
        if !self.storage.update_sequence(&entry).await? {
            return Err(ZyronError::CatalogCorrupted(format!(
                "sequence id {} row missing or undecodable on refill",
                entry.id
            )));
        }
        live.install_refill(slot)
    }

    /// Sets the sequence value. With `is_called` true the value becomes the
    /// current value and the next nextval returns `value + increment`; with
    /// false the next nextval returns `value`.
    pub async fn sequence_setval(
        &self,
        schema_id: SchemaId,
        name: &str,
        value: i64,
        is_called: bool,
    ) -> Result<i64> {
        let live = self
            .get_sequence(schema_id, name)
            .ok_or_else(|| ZyronError::Internal(format!("sequence '{name}' not found")))?;
        self.setval_on(&live, value, is_called).await
    }

    /// setval against an already-resolved live sequence.
    pub async fn setval_on(
        &self,
        live: &Arc<crate::sequence::LiveSequence>,
        value: i64,
        is_called: bool,
    ) -> Result<i64> {
        let _gate = live.lock_refill().await;
        let (entry, slot) = live.plan_setval(value, is_called)?;
        self.log_sequence_reserve(&entry)?;
        if !self.storage.update_sequence(&entry).await? {
            return Err(ZyronError::CatalogCorrupted(format!(
                "sequence id {} row missing or undecodable on setval",
                entry.id
            )));
        }
        live.install_window(slot);
        Ok(value)
    }

    /// Persists an advanced sequence high-water to the WAL and waits for its
    /// durability. Unlike `log_ddl` this does not bump `schema_version`: a
    /// reserved-block advance is not a schema change and must not invalidate
    /// cached plans.
    fn log_sequence_reserve(&self, entry: &SequenceEntry) -> Result<Lsn> {
        let bytes = entry.to_bytes();
        let txn_id = self.wal.allocate_txn_id()?;
        let begin_lsn = self.wal.log_begin(txn_id)?;
        let mut payload = Vec::with_capacity(1 + bytes.len());
        payload.push(DDL_UPDATE_SEQUENCE);
        payload.extend_from_slice(&bytes);
        let insert_lsn = self.wal.log_insert(txn_id, begin_lsn, &payload)?;
        let commit_lsn = self.wal.log_commit(txn_id, insert_lsn)?;
        self.wal.wait_for_flush(commit_lsn)?;
        Ok(commit_lsn)
    }

    // -----------------------------------------------------------------------
    // View operations
    // -----------------------------------------------------------------------

    /// Creates a view. Assigns an id when the entry carries 0. With
    /// `or_replace` an existing view of the same name is overwritten; otherwise
    /// a duplicate name is an error.
    pub async fn create_view(
        &self,
        mut entry: crate::schema::ViewEntry,
        or_replace: bool,
    ) -> Result<u32> {
        let key = (entry.schema_id.0, entry.name.clone());
        let existing_id = self.views_by_name.read().get(&key).map(|v| v.id);
        match existing_id {
            Some(id) if or_replace => {
                entry.id = id;
                self.log_ddl(DDL_UPDATE_VIEW, &entry.to_bytes())?;
                if !self.storage.update_view(&entry).await? {
                    return Err(ZyronError::CatalogCorrupted(format!(
                        "view '{}' row missing or undecodable on replace",
                        entry.name
                    )));
                }
                let e = Arc::new(entry);
                self.views_by_name.write().insert(key, Arc::clone(&e));
                self.views_by_id.write().insert(id, e);
                Ok(id)
            }
            Some(_) => Err(ZyronError::Internal(format!(
                "view '{}' already exists",
                entry.name
            ))),
            None => {
                if entry.id == 0 {
                    entry.id = self.oid_allocator.next();
                }
                let id = entry.id;
                self.log_ddl(DDL_CREATE_VIEW, &entry.to_bytes())?;
                self.storage.store_view(&entry).await?;
                let e = Arc::new(entry);
                self.views_by_name.write().insert(key, Arc::clone(&e));
                self.views_by_id.write().insert(id, e);
                Ok(id)
            }
        }
    }

    /// Resolves a view by schema and name.
    pub fn get_view(
        &self,
        schema_id: SchemaId,
        name: &str,
    ) -> Option<Arc<crate::schema::ViewEntry>> {
        self.views_by_name
            .read()
            .get(&(schema_id.0, name.to_string()))
            .map(Arc::clone)
    }

    /// Resolves a view by its bare name across all schemas. Used by the binder,
    /// which resolves view references without a threaded schema. Errors when
    /// the bare name is ambiguous across schemas.
    pub fn find_view_by_name(&self, name: &str) -> Result<Option<Arc<crate::schema::ViewEntry>>> {
        let bare = name.rsplit('.').next().unwrap_or(name);
        let map = self.views_by_name.read();
        let mut found: Option<Arc<crate::schema::ViewEntry>> = None;
        for ((_, n), entry) in map.iter() {
            if n == bare {
                if found.is_some() {
                    return Err(ZyronError::Internal(format!(
                        "view name '{bare}' is ambiguous across schemas; qualify it"
                    )));
                }
                found = Some(Arc::clone(entry));
            }
        }
        Ok(found)
    }

    /// Lists every view.
    pub fn list_views(&self) -> Vec<Arc<crate::schema::ViewEntry>> {
        self.views_by_name.read().values().map(Arc::clone).collect()
    }

    /// Drops a view and removes its durable entry.
    pub async fn drop_view(&self, schema_id: SchemaId, name: &str) -> Result<()> {
        let key = (schema_id.0, name.to_string());
        let id = self
            .views_by_name
            .read()
            .get(&key)
            .map(|v| v.id)
            .ok_or_else(|| ZyronError::Internal(format!("view '{name}' not found")))?;

        self.log_ddl(DDL_DROP_VIEW, &id.to_le_bytes())?;
        self.storage.delete_view(id).await?;
        self.views_by_name.write().remove(&key);
        self.views_by_id.write().remove(&id);
        Ok(())
    }

    /// Renames a view. Rewrites the entry under the new name and rebuilds the
    /// lookup maps.
    pub async fn rename_view(&self, schema_id: SchemaId, name: &str, new_name: &str) -> Result<()> {
        let old_key = (schema_id.0, name.to_string());
        let existing = self
            .views_by_name
            .read()
            .get(&old_key)
            .map(Arc::clone)
            .ok_or_else(|| ZyronError::Internal(format!("view '{name}' not found")))?;
        if self
            .views_by_name
            .read()
            .contains_key(&(schema_id.0, new_name.to_string()))
        {
            return Err(ZyronError::Internal(format!(
                "view '{new_name}' already exists"
            )));
        }

        let mut entry = (*existing).clone();
        entry.name = new_name.to_string();
        self.log_ddl(DDL_UPDATE_VIEW, &entry.to_bytes())?;
        if !self.storage.update_view(&entry).await? {
            return Err(ZyronError::CatalogCorrupted(format!(
                "view '{}' row missing or undecodable on rename",
                entry.name
            )));
        }

        let id = entry.id;
        let e = Arc::new(entry);
        {
            let mut by_name = self.views_by_name.write();
            by_name.remove(&old_key);
            by_name.insert((schema_id.0, new_name.to_string()), Arc::clone(&e));
        }
        self.views_by_id.write().insert(id, e);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Materialized view operations
    // -----------------------------------------------------------------------

    /// Creates a table from pre-resolved column entries, bypassing parser
    /// ColumnDefs. Used to build a materialized view's backing table from the
    /// query's output schema, whose types are already resolved to TypeIds.
    pub async fn create_table_from_columns(
        &self,
        schema_id: SchemaId,
        name: &str,
        columns: &[(String, zyron_common::TypeId, bool, Option<u8>)],
    ) -> Result<TableId> {
        if self.cache.get_table_by_name(schema_id, name).is_some() {
            return Err(ZyronError::TableAlreadyExists(name.to_string()));
        }
        let table_id = TableId(self.oid_allocator.next());
        let (heap_file_id, fsm_file_id) = self.storage.next_heap_file_id();
        let column_entries: Vec<ColumnEntry> = columns
            .iter()
            .enumerate()
            .map(
                |(i, (col_name, type_id, nullable, fractional_digits))| ColumnEntry {
                    id: ColumnId(i as u16),
                    table_id,
                    name: col_name.clone(),
                    type_id: *type_id,
                    ordinal: i as u16,
                    nullable: *nullable,
                    default_expr: None,
                    max_length: None,
                    fractional_digits: *fractional_digits,
                    tz_offset_secs: None,
                    element_type: None,
                },
            )
            .collect();

        let entry = TableEntry {
            id: table_id,
            schema_id,
            name: name.to_string(),
            heap_file_id,
            fsm_file_id,
            columns: column_entries,
            constraints: Vec::new(),
            created_at: current_timestamp(),
            versioning_enabled: false,
            scd_type: None,
            system_versioned: false,
            history_table_id: None,
            cdf_enabled: false,
            cdf_retention_days: 0,
            lifecycle: Default::default(),
            columnar: Default::default(),
            dropped_at: None,
            expectations: Vec::new(),
            time_travel_retention_secs: 0,
            lake: Default::default(),
            cluster: Default::default(),
            foreign: Default::default(),
        };
        self.log_ddl(DDL_CREATE_TABLE, &entry.to_bytes())?;
        self.storage.store_table(&entry).await?;
        self.cache.put_table(entry);
        Ok(table_id)
    }

    /// Registers a materialized view. The backing table must already exist
    /// (created via `create_table_from_columns` and populated by the caller).
    pub async fn create_mview(
        &self,
        mut entry: crate::schema::MaterializedViewEntry,
    ) -> Result<u32> {
        let key = (entry.schema_id.0, entry.name.clone());
        if self.mviews_by_name.read().contains_key(&key) {
            return Err(ZyronError::Internal(format!(
                "materialized view '{}' already exists",
                entry.name
            )));
        }
        if entry.id == 0 {
            entry.id = self.oid_allocator.next();
        }
        let id = entry.id;
        self.log_ddl(DDL_CREATE_MVIEW, &entry.to_bytes())?;
        self.storage.store_mview(&entry).await?;
        let e = Arc::new(entry);
        self.mviews_by_name.write().insert(key, Arc::clone(&e));
        self.mviews_by_id.write().insert(id, e);
        Ok(id)
    }

    /// Resolves a materialized view by schema and name.
    pub fn get_mview(
        &self,
        schema_id: SchemaId,
        name: &str,
    ) -> Option<Arc<crate::schema::MaterializedViewEntry>> {
        self.mviews_by_name
            .read()
            .get(&(schema_id.0, name.to_string()))
            .map(Arc::clone)
    }

    /// Resolves a materialized view by its bare name across all schemas.
    pub fn find_mview_by_name(
        &self,
        name: &str,
    ) -> Result<Option<Arc<crate::schema::MaterializedViewEntry>>> {
        let bare = name.rsplit('.').next().unwrap_or(name);
        let map = self.mviews_by_name.read();
        let mut found: Option<Arc<crate::schema::MaterializedViewEntry>> = None;
        for ((_, n), entry) in map.iter() {
            if n == bare {
                if found.is_some() {
                    return Err(ZyronError::Internal(format!(
                        "materialized view name '{bare}' is ambiguous across schemas; qualify it"
                    )));
                }
                found = Some(Arc::clone(entry));
            }
        }
        Ok(found)
    }

    /// Lists every materialized view.
    pub fn list_mviews(&self) -> Vec<Arc<crate::schema::MaterializedViewEntry>> {
        self.mviews_by_name
            .read()
            .values()
            .map(Arc::clone)
            .collect()
    }

    /// Drops a materialized view's metadata entry. The caller drops the backing
    /// table separately.
    pub async fn drop_mview(&self, schema_id: SchemaId, name: &str) -> Result<()> {
        let key = (schema_id.0, name.to_string());
        let id = self
            .mviews_by_name
            .read()
            .get(&key)
            .map(|m| m.id)
            .ok_or_else(|| ZyronError::Internal(format!("materialized view '{name}' not found")))?;
        self.log_ddl(DDL_DROP_MVIEW, &id.to_le_bytes())?;
        self.storage.delete_mview(id).await?;
        self.mviews_by_name.write().remove(&key);
        self.mviews_by_id.write().remove(&id);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Function operations
    // -----------------------------------------------------------------------

    /// Registers a SQL scalar function. With `or_replace` an existing overload
    /// with the same name and parameter types is replaced; otherwise a
    /// duplicate signature is an error.
    pub async fn create_function(
        &self,
        mut entry: crate::schema::FunctionEntry,
        or_replace: bool,
    ) -> Result<u32> {
        // Detect an existing overload with the identical signature.
        let existing_id = {
            let map = self.functions_by_name.read();
            map.get(&entry.name).and_then(|overloads| {
                overloads
                    .iter()
                    .find(|f| f.param_types == entry.param_types)
                    .map(|f| f.id)
            })
        };
        if let Some(id) = existing_id {
            if !or_replace {
                return Err(ZyronError::Internal(format!(
                    "function '{}' with these parameter types already exists",
                    entry.name
                )));
            }
            // Replace: drop the old overload first.
            self.remove_function_by_id(id).await?;
        }

        if entry.id == 0 {
            entry.id = self.oid_allocator.next();
        }
        let id = entry.id;
        self.log_ddl(DDL_CREATE_FUNCTION, &entry.to_bytes())?;
        self.storage.store_function(&entry).await?;
        let e = Arc::new(entry);
        self.functions_by_name
            .write()
            .entry(e.name.clone())
            .or_default()
            .push(Arc::clone(&e));
        self.functions_by_id.write().insert(id, e);
        Ok(id)
    }

    /// Resolves a function overload by bare name and argument count. When
    /// several overloads share the arity, the one whose parameter types match
    /// `arg_types` wins; otherwise the first same-arity overload is returned.
    pub fn find_function(
        &self,
        name: &str,
        arg_types: &[zyron_common::TypeId],
    ) -> Option<Arc<crate::schema::FunctionEntry>> {
        let bare = name.rsplit('.').next().unwrap_or(name);
        let map = self.functions_by_name.read();
        let overloads = map.get(bare)?;
        let same_arity: Vec<&Arc<crate::schema::FunctionEntry>> = overloads
            .iter()
            .filter(|f| f.param_types.len() == arg_types.len())
            .collect();
        if same_arity.is_empty() {
            return None;
        }
        same_arity
            .iter()
            .find(|f| f.param_types.as_slice() == arg_types)
            .or_else(|| same_arity.first())
            .map(|f| Arc::clone(f))
    }

    /// Lists every registered function.
    pub fn list_functions(&self) -> Vec<Arc<crate::schema::FunctionEntry>> {
        self.functions_by_id
            .read()
            .values()
            .map(Arc::clone)
            .collect()
    }

    /// Drops every overload of a function by name. Errors when none exist.
    pub async fn drop_function(&self, name: &str) -> Result<()> {
        let bare = name.rsplit('.').next().unwrap_or(name);
        let ids: Vec<u32> = {
            let map = self.functions_by_name.read();
            match map.get(bare) {
                Some(overloads) => overloads.iter().map(|f| f.id).collect(),
                None => Vec::new(),
            }
        };
        if ids.is_empty() {
            return Err(ZyronError::Internal(format!("function '{name}' not found")));
        }
        for id in ids {
            self.remove_function_by_id(id).await?;
        }
        Ok(())
    }

    /// Removes a single function overload by id from storage and both maps.
    async fn remove_function_by_id(&self, id: u32) -> Result<()> {
        self.log_ddl(DDL_DROP_FUNCTION, &id.to_le_bytes())?;
        self.storage.delete_function(id).await?;
        let name = self
            .functions_by_id
            .write()
            .remove(&id)
            .map(|e| e.name.clone());
        if let Some(name) = name {
            let mut by_name = self.functions_by_name.write();
            if let Some(overloads) = by_name.get_mut(&name) {
                overloads.retain(|f| f.id != id);
                if overloads.is_empty() {
                    by_name.remove(&name);
                }
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Aggregate operations
    // -----------------------------------------------------------------------

    /// Registers a user-defined aggregate. Overloads with distinct input types
    /// coexist under the same name. An identical signature is rejected unless
    /// `or_replace` is set, in which case the prior overload is dropped first.
    pub async fn create_aggregate(
        &self,
        mut entry: crate::schema::AggregateEntry,
        or_replace: bool,
    ) -> Result<u32> {
        let existing_id = {
            let map = self.aggregates_by_name.read();
            map.get(&entry.name).and_then(|overloads| {
                overloads
                    .iter()
                    .find(|a| a.input_types == entry.input_types)
                    .map(|a| a.id)
            })
        };
        if let Some(id) = existing_id {
            if !or_replace {
                return Err(ZyronError::Internal(format!(
                    "aggregate '{}' with these input types already exists",
                    entry.name
                )));
            }
            self.remove_aggregate_by_id(id).await?;
        }

        if entry.id == 0 {
            entry.id = self.oid_allocator.next();
        }
        let id = entry.id;
        self.log_ddl(DDL_CREATE_AGGREGATE, &entry.to_bytes())?;
        self.storage.store_aggregate(&entry).await?;
        let e = Arc::new(entry);
        self.aggregates_by_name
            .write()
            .entry(e.name.clone())
            .or_default()
            .push(Arc::clone(&e));
        self.aggregates_by_id.write().insert(id, e);
        Ok(id)
    }

    /// Resolves an aggregate overload by bare name and argument count. When
    /// several overloads share the arity, the one whose input types match
    /// `arg_types` wins; otherwise the first same-arity overload is returned.
    pub fn find_aggregate(
        &self,
        name: &str,
        arg_types: &[zyron_common::TypeId],
    ) -> Option<Arc<crate::schema::AggregateEntry>> {
        let bare = name.rsplit('.').next().unwrap_or(name);
        let map = self.aggregates_by_name.read();
        let overloads = map.get(bare)?;
        let same_arity: Vec<&Arc<crate::schema::AggregateEntry>> = overloads
            .iter()
            .filter(|a| a.input_types.len() == arg_types.len())
            .collect();
        if same_arity.is_empty() {
            return None;
        }
        same_arity
            .iter()
            .find(|a| a.input_types.as_slice() == arg_types)
            .or_else(|| same_arity.first())
            .map(|a| Arc::clone(a))
    }

    /// Returns true when any aggregate overload is registered under the name.
    pub fn is_aggregate(&self, name: &str) -> bool {
        let bare = name.rsplit('.').next().unwrap_or(name);
        self.aggregates_by_name.read().contains_key(bare)
    }

    /// Lists every registered aggregate.
    pub fn list_aggregates(&self) -> Vec<Arc<crate::schema::AggregateEntry>> {
        self.aggregates_by_id
            .read()
            .values()
            .map(Arc::clone)
            .collect()
    }

    /// Drops every overload of an aggregate by name. Errors when none exist.
    pub async fn drop_aggregate(&self, name: &str) -> Result<()> {
        let bare = name.rsplit('.').next().unwrap_or(name);
        let ids: Vec<u32> = {
            let map = self.aggregates_by_name.read();
            match map.get(bare) {
                Some(overloads) => overloads.iter().map(|a| a.id).collect(),
                None => Vec::new(),
            }
        };
        if ids.is_empty() {
            return Err(ZyronError::Internal(format!(
                "aggregate '{name}' not found"
            )));
        }
        for id in ids {
            self.remove_aggregate_by_id(id).await?;
        }
        Ok(())
    }

    /// Removes a single aggregate overload by id from storage and both maps.
    async fn remove_aggregate_by_id(&self, id: u32) -> Result<()> {
        self.log_ddl(DDL_DROP_AGGREGATE, &id.to_le_bytes())?;
        self.storage.delete_aggregate(id).await?;
        let name = self
            .aggregates_by_id
            .write()
            .remove(&id)
            .map(|e| e.name.clone());
        if let Some(name) = name {
            let mut by_name = self.aggregates_by_name.write();
            if let Some(overloads) = by_name.get_mut(&name) {
                overloads.retain(|a| a.id != id);
                if overloads.is_empty() {
                    by_name.remove(&name);
                }
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Procedure operations
    // -----------------------------------------------------------------------

    /// Registers a stored procedure. Overloads with distinct input types coexist
    /// under the same name. An identical signature is rejected unless
    /// `or_replace` is set, in which case the prior overload is dropped first.
    pub async fn create_procedure(
        &self,
        mut entry: crate::schema::ProcedureEntry,
        or_replace: bool,
    ) -> Result<u32> {
        let existing_id = {
            let map = self.procedures_by_name.read();
            map.get(&entry.name).and_then(|overloads| {
                overloads
                    .iter()
                    .find(|p| p.param_types == entry.param_types)
                    .map(|p| p.id)
            })
        };
        if let Some(id) = existing_id {
            if !or_replace {
                return Err(ZyronError::Internal(format!(
                    "procedure '{}' with these parameter types already exists",
                    entry.name
                )));
            }
            self.remove_procedure_by_id(id).await?;
        }

        if entry.id == 0 {
            entry.id = self.oid_allocator.next();
        }
        let id = entry.id;
        self.log_ddl(DDL_CREATE_PROCEDURE, &entry.to_bytes())?;
        self.storage.store_procedure(&entry).await?;
        let e = Arc::new(entry);
        self.procedures_by_name
            .write()
            .entry(e.name.clone())
            .or_default()
            .push(Arc::clone(&e));
        self.procedures_by_id.write().insert(id, e);
        Ok(id)
    }

    /// Resolves a procedure overload by bare name and argument count. When
    /// several overloads share the arity, the one whose parameter types match
    /// `arg_types` wins; otherwise the first same-arity overload is returned.
    pub fn find_procedure(
        &self,
        name: &str,
        arg_types: &[zyron_common::TypeId],
    ) -> Option<Arc<crate::schema::ProcedureEntry>> {
        let bare = name.rsplit('.').next().unwrap_or(name);
        let map = self.procedures_by_name.read();
        let overloads = map.get(bare)?;
        let same_arity: Vec<&Arc<crate::schema::ProcedureEntry>> = overloads
            .iter()
            .filter(|p| p.param_types.len() == arg_types.len())
            .collect();
        if same_arity.is_empty() {
            return None;
        }
        same_arity
            .iter()
            .find(|p| p.param_types.as_slice() == arg_types)
            .or_else(|| same_arity.first())
            .map(|p| Arc::clone(p))
    }

    /// Resolves a procedure by bare name, ignoring argument types. Used by CALL
    /// to find the single overload when only the name is known.
    pub fn find_procedure_by_name(&self, name: &str) -> Option<Arc<crate::schema::ProcedureEntry>> {
        let bare = name.rsplit('.').next().unwrap_or(name);
        self.procedures_by_name
            .read()
            .get(bare)
            .and_then(|overloads| overloads.first().map(Arc::clone))
    }

    /// Lists every registered procedure.
    pub fn list_procedures(&self) -> Vec<Arc<crate::schema::ProcedureEntry>> {
        self.procedures_by_id
            .read()
            .values()
            .map(Arc::clone)
            .collect()
    }

    /// Drops every overload of a procedure by name. Errors when none exist.
    pub async fn drop_procedure(&self, name: &str) -> Result<()> {
        let bare = name.rsplit('.').next().unwrap_or(name);
        let ids: Vec<u32> = {
            let map = self.procedures_by_name.read();
            match map.get(bare) {
                Some(overloads) => overloads.iter().map(|p| p.id).collect(),
                None => Vec::new(),
            }
        };
        if ids.is_empty() {
            return Err(ZyronError::Internal(format!(
                "procedure '{name}' not found"
            )));
        }
        for id in ids {
            self.remove_procedure_by_id(id).await?;
        }
        Ok(())
    }

    /// Removes a single procedure overload by id from storage and both maps.
    async fn remove_procedure_by_id(&self, id: u32) -> Result<()> {
        self.log_ddl(DDL_DROP_PROCEDURE, &id.to_le_bytes())?;
        self.storage.delete_procedure(id).await?;
        let name = self
            .procedures_by_id
            .write()
            .remove(&id)
            .map(|e| e.name.clone());
        if let Some(name) = name {
            let mut by_name = self.procedures_by_name.write();
            if let Some(overloads) = by_name.get_mut(&name) {
                overloads.retain(|p| p.id != id);
                if overloads.is_empty() {
                    by_name.remove(&name);
                }
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Schedule operations
    // -----------------------------------------------------------------------

    /// Registers a scheduled task. Names are unique; an existing name is rejected
    /// unless `or_replace` is set, in which case the prior schedule is dropped.
    pub async fn create_schedule(
        &self,
        mut entry: crate::schema::ScheduleEntry,
        or_replace: bool,
    ) -> Result<u32> {
        let existing_id = self.schedules_by_name.read().get(&entry.name).map(|e| e.id);
        if let Some(id) = existing_id {
            if !or_replace {
                return Err(ZyronError::Internal(format!(
                    "schedule '{}' already exists",
                    entry.name
                )));
            }
            self.remove_schedule_by_id(id).await?;
        }

        if entry.id == 0 {
            entry.id = self.oid_allocator.next();
        }
        let id = entry.id;
        self.log_ddl(DDL_CREATE_SCHEDULE, &entry.to_bytes())?;
        self.storage.store_schedule(&entry).await?;
        let e = Arc::new(entry);
        self.schedules_by_name
            .write()
            .insert(e.name.clone(), Arc::clone(&e));
        self.schedules_by_id.write().insert(id, e);
        Ok(id)
    }

    /// Persists a mutated schedule (pause/resume or last_run/next_run advance).
    /// Re-logs the entry so a restart recovers the latest state.
    pub async fn update_schedule(&self, entry: crate::schema::ScheduleEntry) -> Result<()> {
        self.log_ddl(DDL_CREATE_SCHEDULE, &entry.to_bytes())?;
        self.storage.delete_schedule(entry.id).await?;
        self.storage.store_schedule(&entry).await?;
        let e = Arc::new(entry);
        self.schedules_by_name
            .write()
            .insert(e.name.clone(), Arc::clone(&e));
        self.schedules_by_id.write().insert(e.id, e);
        Ok(())
    }

    pub fn get_schedule_by_name(&self, name: &str) -> Option<Arc<crate::schema::ScheduleEntry>> {
        self.schedules_by_name.read().get(name).map(Arc::clone)
    }

    pub fn list_schedules(&self) -> Vec<Arc<crate::schema::ScheduleEntry>> {
        self.schedules_by_id
            .read()
            .values()
            .map(Arc::clone)
            .collect()
    }

    pub async fn drop_schedule(&self, name: &str) -> Result<()> {
        let id = self.schedules_by_name.read().get(name).map(|e| e.id);
        match id {
            Some(id) => self.remove_schedule_by_id(id).await,
            None => Err(ZyronError::Internal(format!("schedule '{name}' not found"))),
        }
    }

    async fn remove_schedule_by_id(&self, id: u32) -> Result<()> {
        self.log_ddl(DDL_DROP_SCHEDULE, &id.to_le_bytes())?;
        self.storage.delete_schedule(id).await?;
        let name = self
            .schedules_by_id
            .write()
            .remove(&id)
            .map(|e| e.name.clone());
        if let Some(name) = name {
            self.schedules_by_name.write().remove(&name);
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Pipeline operations

    /// Registers a new pipeline definition. Rejects a duplicate name unless
    /// `or_replace`, which drops the prior definition (and its run state) first.
    pub async fn create_pipeline(
        &self,
        mut entry: crate::schema::PipelineEntry,
        or_replace: bool,
    ) -> Result<u32> {
        let existing_id = self.pipelines_by_name.read().get(&entry.name).map(|e| e.id);
        if let Some(id) = existing_id {
            if !or_replace {
                return Err(ZyronError::Internal(format!(
                    "pipeline '{}' already exists",
                    entry.name
                )));
            }
            self.remove_pipeline_by_id(id).await?;
        }

        if entry.id == 0 {
            entry.id = self.oid_allocator.next();
        }
        let id = entry.id;
        self.log_ddl(DDL_CREATE_PIPELINE, &entry.to_bytes())?;
        self.storage.store_pipeline(&entry).await?;
        let e = Arc::new(entry);
        self.pipelines_by_name
            .write()
            .insert(e.name.clone(), Arc::clone(&e));
        self.pipelines_by_id.write().insert(id, e);
        Ok(id)
    }

    /// Persists a mutated pipeline (run-state advance after RUN). Re-logs the
    /// entry so a restart recovers the latest outcome.
    pub async fn update_pipeline(&self, entry: crate::schema::PipelineEntry) -> Result<()> {
        self.log_ddl(DDL_CREATE_PIPELINE, &entry.to_bytes())?;
        self.storage.delete_pipeline(entry.id).await?;
        self.storage.store_pipeline(&entry).await?;
        let e = Arc::new(entry);
        self.pipelines_by_name
            .write()
            .insert(e.name.clone(), Arc::clone(&e));
        self.pipelines_by_id.write().insert(e.id, e);
        Ok(())
    }

    pub fn get_pipeline_by_name(&self, name: &str) -> Option<Arc<crate::schema::PipelineEntry>> {
        self.pipelines_by_name.read().get(name).map(Arc::clone)
    }

    pub fn list_pipelines(&self) -> Vec<Arc<crate::schema::PipelineEntry>> {
        self.pipelines_by_id
            .read()
            .values()
            .map(Arc::clone)
            .collect()
    }

    pub async fn drop_pipeline(&self, name: &str) -> Result<()> {
        let id = self.pipelines_by_name.read().get(name).map(|e| e.id);
        match id {
            Some(id) => self.remove_pipeline_by_id(id).await,
            None => Err(ZyronError::Internal(format!("pipeline '{name}' not found"))),
        }
    }

    async fn remove_pipeline_by_id(&self, id: u32) -> Result<()> {
        self.log_ddl(DDL_DROP_PIPELINE, &id.to_le_bytes())?;
        self.storage.delete_pipeline(id).await?;
        let name = self
            .pipelines_by_id
            .write()
            .remove(&id)
            .map(|e| e.name.clone());
        if let Some(name) = name {
            self.pipelines_by_name.write().remove(&name);
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Event handler operations

    /// Registers a new event handler. Rejects a duplicate name unless
    /// `or_replace`, which drops the prior handler first.
    pub async fn create_event_handler(
        &self,
        mut entry: crate::schema::EventHandlerEntry,
        or_replace: bool,
    ) -> Result<u32> {
        let existing_id = self
            .event_handlers_by_name
            .read()
            .get(&entry.name)
            .map(|e| e.id);
        if let Some(id) = existing_id {
            if !or_replace {
                return Err(ZyronError::Internal(format!(
                    "event handler '{}' already exists",
                    entry.name
                )));
            }
            self.remove_event_handler_by_id(id).await?;
        }

        if entry.id == 0 {
            entry.id = self.oid_allocator.next();
        }
        let id = entry.id;
        self.log_ddl(DDL_CREATE_EVENT_HANDLER, &entry.to_bytes())?;
        self.storage.store_event_handler(&entry).await?;
        let e = Arc::new(entry);
        self.event_handlers_by_name
            .write()
            .insert(e.name.clone(), Arc::clone(&e));
        self.event_handlers_by_id.write().insert(id, e);
        Ok(id)
    }

    pub fn get_event_handler_by_name(
        &self,
        name: &str,
    ) -> Option<Arc<crate::schema::EventHandlerEntry>> {
        self.event_handlers_by_name.read().get(name).map(Arc::clone)
    }

    pub fn list_event_handlers(&self) -> Vec<Arc<crate::schema::EventHandlerEntry>> {
        self.event_handlers_by_id
            .read()
            .values()
            .map(Arc::clone)
            .collect()
    }

    pub async fn drop_event_handler(&self, name: &str) -> Result<()> {
        let id = self.event_handlers_by_name.read().get(name).map(|e| e.id);
        match id {
            Some(id) => self.remove_event_handler_by_id(id).await,
            None => Err(ZyronError::Internal(format!(
                "event handler '{name}' not found"
            ))),
        }
    }

    async fn remove_event_handler_by_id(&self, id: u32) -> Result<()> {
        self.log_ddl(DDL_DROP_EVENT_HANDLER, &id.to_le_bytes())?;
        self.storage.delete_event_handler(id).await?;
        let name = self
            .event_handlers_by_id
            .write()
            .remove(&id)
            .map(|e| e.name.clone());
        if let Some(name) = name {
            self.event_handlers_by_name.write().remove(&name);
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Version tag operations

    /// Registers a named version tag. Rejects a duplicate name.
    pub async fn create_version_tag(
        &self,
        mut entry: crate::schema::VersionTagEntry,
    ) -> Result<u32> {
        if self.version_tags_by_name.read().contains_key(&entry.name) {
            return Err(ZyronError::Internal(format!(
                "version '{}' already exists",
                entry.name
            )));
        }
        if entry.id == 0 {
            entry.id = self.oid_allocator.next();
        }
        let id = entry.id;
        self.log_ddl(DDL_CREATE_VERSION_TAG, &entry.to_bytes())?;
        self.storage.store_version_tag(&entry).await?;
        let e = Arc::new(entry);
        self.version_tags_by_name
            .write()
            .insert(e.name.clone(), Arc::clone(&e));
        self.version_tags_by_id.write().insert(id, e);
        Ok(id)
    }

    pub fn get_version_tag_by_name(
        &self,
        name: &str,
    ) -> Option<Arc<crate::schema::VersionTagEntry>> {
        self.version_tags_by_name.read().get(name).map(Arc::clone)
    }

    pub fn list_version_tags(&self) -> Vec<Arc<crate::schema::VersionTagEntry>> {
        self.version_tags_by_id
            .read()
            .values()
            .map(Arc::clone)
            .collect()
    }

    pub async fn drop_version_tag(&self, name: &str) -> Result<()> {
        let id = self.version_tags_by_name.read().get(name).map(|e| e.id);
        match id {
            Some(id) => {
                self.log_ddl(DDL_DROP_VERSION_TAG, &id.to_le_bytes())?;
                self.storage.delete_version_tag(id).await?;
                self.version_tags_by_id.write().remove(&id);
                self.version_tags_by_name.write().remove(name);
                Ok(())
            }
            None => Err(ZyronError::Internal(format!("version '{name}' not found"))),
        }
    }

    // -----------------------------------------------------------------------
    // Trigger operations
    // -----------------------------------------------------------------------

    /// Registers a trigger on a table. Trigger names are unique per table.
    pub async fn create_trigger(&self, mut entry: crate::schema::TriggerEntry) -> Result<u32> {
        {
            let by_table = self.triggers_by_table.read();
            if let Some(v) = by_table.get(&entry.table_id) {
                if v.iter().any(|t| t.name == entry.name) {
                    return Err(ZyronError::Internal(format!(
                        "trigger '{}' already exists on this table",
                        entry.name
                    )));
                }
            }
        }
        if entry.id == 0 {
            entry.id = self.oid_allocator.next();
        }
        let id = entry.id;
        self.log_ddl(DDL_CREATE_TRIGGER, &entry.to_bytes())?;
        self.storage.store_trigger(&entry).await?;
        let e = Arc::new(entry);
        self.triggers_by_table
            .write()
            .entry(e.table_id)
            .or_default()
            .push(Arc::clone(&e));
        self.triggers_by_id.write().insert(id, e);
        Ok(id)
    }

    /// Returns every trigger defined on a table. The executor calls this per DML
    /// event; it is a single read-lock + clone of the small per-table vector.
    pub fn triggers_for_table(&self, table_id: TableId) -> Vec<Arc<crate::schema::TriggerEntry>> {
        self.triggers_by_table
            .read()
            .get(&table_id.0)
            .cloned()
            .unwrap_or_default()
    }

    pub fn find_trigger(
        &self,
        table_id: TableId,
        name: &str,
    ) -> Option<Arc<crate::schema::TriggerEntry>> {
        self.triggers_by_table
            .read()
            .get(&table_id.0)
            .and_then(|v| v.iter().find(|t| t.name == name).map(Arc::clone))
    }

    pub async fn drop_trigger(&self, table_id: TableId, name: &str) -> Result<()> {
        let id = self
            .find_trigger(table_id, name)
            .map(|t| t.id)
            .ok_or_else(|| {
                ZyronError::Internal(format!("trigger '{name}' not found on this table"))
            })?;
        self.log_ddl(DDL_DROP_TRIGGER, &id.to_le_bytes())?;
        self.storage.delete_trigger(id).await?;
        self.triggers_by_id.write().remove(&id);
        let mut by_table = self.triggers_by_table.write();
        if let Some(v) = by_table.get_mut(&table_id.0) {
            v.retain(|t| t.id != id);
            if v.is_empty() {
                by_table.remove(&table_id.0);
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Comment operations
    // -----------------------------------------------------------------------

    /// Sets or removes a comment on an object. `comment` of None removes any
    /// existing comment for the key; Some replaces it. `column_name` is empty
    /// for an object-level comment.
    pub async fn set_comment(
        &self,
        object_type: u8,
        object_name: &str,
        column_name: &str,
        comment: Option<String>,
    ) -> Result<()> {
        let key = (
            object_type,
            object_name.to_string(),
            column_name.to_string(),
        );
        let existing_id = self.comments.read().get(&key).map(|e| e.id);
        match comment {
            Some(text) => {
                let id = existing_id.unwrap_or_else(|| self.oid_allocator.next());
                let entry = crate::schema::CommentEntry {
                    id,
                    object_type,
                    object_name: object_name.to_string(),
                    column_name: column_name.to_string(),
                    comment: text,
                };
                self.log_ddl(DDL_SET_COMMENT, &entry.to_bytes())?;
                if existing_id.is_some() {
                    self.storage.delete_comment(id).await?;
                }
                self.storage.store_comment(&entry).await?;
                self.comments.write().insert(key, Arc::new(entry));
            }
            None => {
                if let Some(id) = existing_id {
                    self.log_ddl(DDL_DROP_COMMENT, &id.to_le_bytes())?;
                    self.storage.delete_comment(id).await?;
                    self.comments.write().remove(&key);
                }
            }
        }
        Ok(())
    }

    /// Returns the comment text on an object, if any.
    pub fn get_comment(
        &self,
        object_type: u8,
        object_name: &str,
        column_name: &str,
    ) -> Option<String> {
        self.comments
            .read()
            .get(&(
                object_type,
                object_name.to_string(),
                column_name.to_string(),
            ))
            .map(|e| e.comment.clone())
    }

    /// Lists every stored comment.
    pub fn list_comments(&self) -> Vec<Arc<crate::schema::CommentEntry>> {
        self.comments.read().values().map(Arc::clone).collect()
    }

    // -----------------------------------------------------------------------
    // External source operations
    // -----------------------------------------------------------------------

    pub async fn create_external_source(
        &self,
        mut entry: ExternalSourceEntry,
    ) -> Result<ExternalSourceId> {
        if self
            .cache
            .get_external_source_by_name(entry.schema_id, &entry.name)
            .is_some()
        {
            return Err(ZyronError::Internal(format!(
                "external source '{}' already exists",
                entry.name
            )));
        }

        if entry.id.0 == 0 {
            entry.id = ExternalSourceId(self.oid_allocator.next());
        }

        let id = entry.id;
        self.log_ddl(DDL_CREATE_EXTERNAL_SOURCE, &entry.to_bytes())?;
        self.storage.store_external_source(&entry).await?;
        self.cache.put_external_source(entry);
        Ok(id)
    }

    pub fn get_external_source(
        &self,
        schema_id: SchemaId,
        name: &str,
    ) -> Option<Arc<ExternalSourceEntry>> {
        self.cache.get_external_source_by_name(schema_id, name)
    }

    pub fn get_external_source_by_id(
        &self,
        id: ExternalSourceId,
    ) -> Option<Arc<ExternalSourceEntry>> {
        self.cache.get_external_source(id)
    }

    pub fn list_external_sources(&self) -> Vec<Arc<ExternalSourceEntry>> {
        self.cache.list_external_sources()
    }

    pub async fn drop_external_source(&self, schema_id: SchemaId, name: &str) -> Result<()> {
        let src = self
            .cache
            .get_external_source_by_name(schema_id, name)
            .ok_or_else(|| ZyronError::Internal(format!("external source '{name}' not found")))?;

        let id = src.id;
        let mut payload = vec![0u8; 4];
        payload[..4].copy_from_slice(&id.0.to_le_bytes());
        self.log_ddl(DDL_DROP_EXTERNAL_SOURCE, &payload)?;
        self.storage.delete_external_source(id).await?;
        self.cache.invalidate_external_source(id);
        Ok(())
    }

    pub async fn update_external_source(&self, entry: ExternalSourceEntry) -> Result<()> {
        let id = entry.id;
        self.log_ddl(DDL_ALTER_EXTERNAL_SOURCE, &entry.to_bytes())?;
        self.storage.update_external_source(&entry).await?;
        self.cache.invalidate_external_source(id);
        self.cache.put_external_source(entry);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // External sink operations
    // -----------------------------------------------------------------------

    pub async fn create_external_sink(
        &self,
        mut entry: ExternalSinkEntry,
    ) -> Result<ExternalSinkId> {
        if self
            .cache
            .get_external_sink_by_name(entry.schema_id, &entry.name)
            .is_some()
        {
            return Err(ZyronError::Internal(format!(
                "external sink '{}' already exists",
                entry.name
            )));
        }

        if entry.id.0 == 0 {
            entry.id = ExternalSinkId(self.oid_allocator.next());
        }

        let id = entry.id;
        self.log_ddl(DDL_CREATE_EXTERNAL_SINK, &entry.to_bytes())?;
        self.storage.store_external_sink(&entry).await?;
        self.cache.put_external_sink(entry);
        Ok(id)
    }

    pub fn get_external_sink(
        &self,
        schema_id: SchemaId,
        name: &str,
    ) -> Option<Arc<ExternalSinkEntry>> {
        self.cache.get_external_sink_by_name(schema_id, name)
    }

    pub fn get_external_sink_by_id(&self, id: ExternalSinkId) -> Option<Arc<ExternalSinkEntry>> {
        self.cache.get_external_sink(id)
    }

    pub fn list_external_sinks(&self) -> Vec<Arc<ExternalSinkEntry>> {
        self.cache.list_external_sinks()
    }

    pub async fn drop_external_sink(&self, schema_id: SchemaId, name: &str) -> Result<()> {
        let sink = self
            .cache
            .get_external_sink_by_name(schema_id, name)
            .ok_or_else(|| ZyronError::Internal(format!("external sink '{name}' not found")))?;

        let id = sink.id;
        let mut payload = vec![0u8; 4];
        payload[..4].copy_from_slice(&id.0.to_le_bytes());
        self.log_ddl(DDL_DROP_EXTERNAL_SINK, &payload)?;
        self.storage.delete_external_sink(id).await?;
        self.cache.invalidate_external_sink(id);
        Ok(())
    }

    pub async fn update_external_sink(&self, entry: ExternalSinkEntry) -> Result<()> {
        let id = entry.id;
        self.log_ddl(DDL_ALTER_EXTERNAL_SINK, &entry.to_bytes())?;
        self.storage.update_external_sink(&entry).await?;
        self.cache.invalidate_external_sink(id);
        self.cache.put_external_sink(entry);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Publication operations
    // -----------------------------------------------------------------------

    pub async fn create_publication(&self, mut entry: PublicationEntry) -> Result<PublicationId> {
        if self
            .cache
            .get_publication_by_name(entry.schema_id, &entry.name)
            .is_some()
        {
            return Err(ZyronError::Internal(format!(
                "publication '{}' already exists",
                entry.name
            )));
        }

        if entry.id.0 == 0 {
            entry.id = PublicationId(self.oid_allocator.next());
        }

        let id = entry.id;
        self.log_ddl(DDL_CREATE_PUBLICATION, &entry.to_bytes())?;
        self.storage.store_publication(&entry).await?;
        self.cache.put_publication(entry);
        Ok(id)
    }

    pub fn get_publication(
        &self,
        schema_id: SchemaId,
        name: &str,
    ) -> Option<Arc<PublicationEntry>> {
        self.cache.get_publication_by_name(schema_id, name)
    }

    pub fn get_publication_by_id(&self, id: PublicationId) -> Option<Arc<PublicationEntry>> {
        self.cache.get_publication(id)
    }

    pub fn list_publications(&self) -> Vec<Arc<PublicationEntry>> {
        self.cache.list_publications()
    }

    pub async fn drop_publication(&self, schema_id: SchemaId, name: &str) -> Result<()> {
        let pubn = self
            .cache
            .get_publication_by_name(schema_id, name)
            .ok_or_else(|| ZyronError::Internal(format!("publication '{name}' not found")))?;

        let id = pubn.id;
        let mut payload = vec![0u8; 4];
        payload[..4].copy_from_slice(&id.0.to_le_bytes());
        self.log_ddl(DDL_DROP_PUBLICATION, &payload)?;
        self.storage.delete_publication(id).await?;
        self.cache.invalidate_publication(id);
        self.cache.invalidate_publication_tables_for(id);
        Ok(())
    }

    pub async fn update_publication(&self, entry: PublicationEntry) -> Result<()> {
        let id = entry.id;
        self.log_ddl(DDL_ALTER_PUBLICATION, &entry.to_bytes())?;
        self.storage.update_publication(&entry).await?;
        self.cache.invalidate_publication(id);
        self.cache.put_publication(entry);
        Ok(())
    }

    pub async fn add_publication_table(&self, mut entry: PublicationTableEntry) -> Result<u32> {
        if entry.id == 0 {
            entry.id = self.oid_allocator.next();
        }
        let id = entry.id;
        self.log_ddl(DDL_ADD_PUBLICATION_TABLE, &entry.to_bytes())?;
        self.storage.store_publication_table(&entry).await?;
        self.cache.put_publication_table(entry);
        Ok(id)
    }

    pub fn get_publication_tables(
        &self,
        publication_id: PublicationId,
    ) -> Vec<Arc<PublicationTableEntry>> {
        self.cache.get_publication_tables(publication_id)
    }

    pub async fn remove_publication_table(
        &self,
        publication_id: PublicationId,
        table_id: TableId,
    ) -> Result<()> {
        let mut payload = Vec::with_capacity(8);
        payload.extend_from_slice(&publication_id.0.to_le_bytes());
        payload.extend_from_slice(&table_id.0.to_le_bytes());
        self.log_ddl(DDL_REMOVE_PUBLICATION_TABLE, &payload)?;
        self.storage
            .delete_publication_table(publication_id, table_id)
            .await?;
        self.cache
            .invalidate_publication_table(publication_id, table_id);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Subscription operations
    // -----------------------------------------------------------------------

    pub async fn create_subscription(
        &self,
        mut entry: SubscriptionEntry,
    ) -> Result<SubscriptionId> {
        if entry.id.0 == 0 {
            entry.id = SubscriptionId(self.oid_allocator.next());
        }
        let id = entry.id;
        self.log_ddl(DDL_CREATE_SUBSCRIPTION, &entry.to_bytes())?;
        self.storage.store_subscription(&entry).await?;
        self.cache.put_subscription(entry);
        Ok(id)
    }

    pub fn get_subscription(&self, id: SubscriptionId) -> Option<Arc<SubscriptionEntry>> {
        self.cache.get_subscription(id)
    }

    pub fn list_subscriptions(&self) -> Vec<Arc<SubscriptionEntry>> {
        self.cache.list_subscriptions()
    }

    pub fn list_publication_subscribers(
        &self,
        pub_id: PublicationId,
    ) -> Vec<Arc<SubscriptionEntry>> {
        self.cache.list_publication_subscribers(pub_id)
    }

    pub async fn update_subscription(&self, entry: SubscriptionEntry) -> Result<()> {
        let id = entry.id;
        self.log_ddl(DDL_UPDATE_SUBSCRIPTION, &entry.to_bytes())?;
        self.storage.update_subscription(&entry).await?;
        self.cache.invalidate_subscription(id);
        self.cache.put_subscription(entry);
        Ok(())
    }

    pub async fn update_subscription_lsn(&self, id: SubscriptionId, new_lsn: u64) -> Result<()> {
        let current = self
            .cache
            .get_subscription(id)
            .ok_or_else(|| ZyronError::Internal(format!("subscription {} not found", id.0)))?;
        let mut updated = (*current).clone();
        updated.last_seen_lsn = new_lsn;
        updated.last_poll_at = current_timestamp();
        self.log_ddl(DDL_UPDATE_SUBSCRIPTION, &updated.to_bytes())?;
        self.storage.update_subscription(&updated).await?;
        self.cache.invalidate_subscription(id);
        self.cache.put_subscription(updated);
        Ok(())
    }

    pub async fn update_subscription_state(
        &self,
        id: SubscriptionId,
        state: SubscriptionState,
        last_error: Option<String>,
    ) -> Result<()> {
        let current = self
            .cache
            .get_subscription(id)
            .ok_or_else(|| ZyronError::Internal(format!("subscription {} not found", id.0)))?;
        let mut updated = (*current).clone();
        updated.state = state;
        updated.last_error = last_error;
        self.log_ddl(DDL_UPDATE_SUBSCRIPTION, &updated.to_bytes())?;
        self.storage.update_subscription(&updated).await?;
        self.cache.invalidate_subscription(id);
        self.cache.put_subscription(updated);
        Ok(())
    }

    pub async fn drop_subscription(&self, id: SubscriptionId) -> Result<()> {
        let mut payload = vec![0u8; 4];
        payload[..4].copy_from_slice(&id.0.to_le_bytes());
        self.log_ddl(DDL_DROP_SUBSCRIPTION, &payload)?;
        self.storage.delete_subscription(id).await?;
        self.cache.invalidate_subscription(id);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Endpoint operations
    // -----------------------------------------------------------------------

    pub async fn create_endpoint(&self, mut entry: EndpointEntry) -> Result<EndpointId> {
        if self
            .cache
            .get_endpoint_by_name(entry.schema_id, &entry.name)
            .is_some()
        {
            return Err(ZyronError::Internal(format!(
                "endpoint '{}' already exists",
                entry.name
            )));
        }
        if self.cache.get_endpoint_by_path(&entry.path).is_some() {
            return Err(ZyronError::Internal(format!(
                "endpoint path '{}' already in use",
                entry.path
            )));
        }

        if entry.id.0 == 0 {
            entry.id = EndpointId(self.oid_allocator.next());
        }

        let id = entry.id;
        self.log_ddl(DDL_CREATE_ENDPOINT, &entry.to_bytes())?;
        self.storage.store_endpoint(&entry).await?;
        self.cache.put_endpoint(entry);
        Ok(id)
    }

    pub fn get_endpoint(&self, schema_id: SchemaId, name: &str) -> Option<Arc<EndpointEntry>> {
        self.cache.get_endpoint_by_name(schema_id, name)
    }

    pub fn get_endpoint_by_id(&self, id: EndpointId) -> Option<Arc<EndpointEntry>> {
        self.cache.get_endpoint(id)
    }

    pub fn get_endpoint_by_path(&self, path: &str) -> Option<Arc<EndpointEntry>> {
        self.cache.get_endpoint_by_path(path)
    }

    pub fn list_endpoints(&self) -> Vec<Arc<EndpointEntry>> {
        self.cache.list_endpoints()
    }

    pub async fn drop_endpoint(&self, schema_id: SchemaId, name: &str) -> Result<()> {
        let ep = self
            .cache
            .get_endpoint_by_name(schema_id, name)
            .ok_or_else(|| ZyronError::Internal(format!("endpoint '{name}' not found")))?;
        let id = ep.id;
        let mut payload = vec![0u8; 4];
        payload[..4].copy_from_slice(&id.0.to_le_bytes());
        self.log_ddl(DDL_DROP_ENDPOINT, &payload)?;
        self.storage.delete_endpoint(id).await?;
        self.cache.invalidate_endpoint(id);
        Ok(())
    }

    pub async fn update_endpoint(&self, entry: EndpointEntry) -> Result<()> {
        let id = entry.id;
        self.log_ddl(DDL_ALTER_ENDPOINT, &entry.to_bytes())?;
        self.storage.update_endpoint(&entry).await?;
        self.cache.invalidate_endpoint(id);
        self.cache.put_endpoint(entry);
        Ok(())
    }

    pub async fn set_endpoint_enabled(&self, id: EndpointId, enabled: bool) -> Result<()> {
        let current = self
            .cache
            .get_endpoint(id)
            .ok_or_else(|| ZyronError::Internal(format!("endpoint {} not found", id.0)))?;
        let mut updated = (*current).clone();
        updated.enabled = enabled;
        self.log_ddl(DDL_ALTER_ENDPOINT, &updated.to_bytes())?;
        self.storage.update_endpoint(&updated).await?;
        self.cache.invalidate_endpoint(id);
        self.cache.put_endpoint(updated);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Security map operations
    // -----------------------------------------------------------------------

    pub async fn create_security_map(&self, mut entry: SecurityMapEntry) -> Result<SecurityMapId> {
        if entry.id.0 == 0 {
            entry.id = SecurityMapId(self.oid_allocator.next());
        }
        let id = entry.id;
        self.log_ddl(DDL_CREATE_SECURITY_MAP, &entry.to_bytes())?;
        self.storage.store_security_map(&entry).await?;
        self.cache.put_security_map(entry);
        Ok(id)
    }

    pub fn list_security_maps(&self) -> Vec<Arc<SecurityMapEntry>> {
        self.cache.list_security_maps()
    }

    pub fn resolve_security_map(&self, kind: SecurityMapKind, key: &str) -> Option<u32> {
        self.cache.resolve_security_map(kind, key)
    }

    pub async fn drop_security_map(&self, id: SecurityMapId) -> Result<()> {
        let mut payload = vec![0u8; 4];
        payload[..4].copy_from_slice(&id.0.to_le_bytes());
        self.log_ddl(DDL_DROP_SECURITY_MAP, &payload)?;
        self.storage.delete_security_map(id).await?;
        self.cache.invalidate_security_map(id);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /// Stores pre-computed statistics for a table.
    pub fn put_stats(
        &self,
        table_id: TableId,
        table_stats: TableStats,
        column_stats: Vec<ColumnStats>,
    ) {
        self.stats
            .write()
            .insert(table_id, Arc::new((table_stats, column_stats)));
    }

    /// Retrieves statistics for a table. Returns an `Arc`: cloning it is a
    /// refcount bump, never a copy of the stats payload.
    pub fn get_stats(&self, table_id: TableId) -> Option<Arc<(TableStats, Vec<ColumnStats>)>> {
        self.stats.read().get(&table_id).cloned()
    }

    // -----------------------------------------------------------------------
    // WAL integration
    // -----------------------------------------------------------------------

    /// Logs a DDL operation to the WAL as a transactional insert and waits
    /// for the commit record to reach durable storage. DDL is a low-volume,
    /// high-importance write path: a crash between log_commit and the next
    /// flush would otherwise lose the schema change even though storage
    /// pages are lazy. Blocking on wait_for_flush here makes catalog DDL
    /// crash-safe end-to-end and lets recover_unflushed_ddl on the next
    /// boot put storage back in sync.
    fn log_ddl(&self, ddl_type: u8, entry_bytes: &[u8]) -> Result<Lsn> {
        let txn_id = self.wal.allocate_txn_id()?;
        let begin_lsn = self.wal.log_begin(txn_id)?;

        // Build DDL payload: 1-byte type prefix + entry bytes
        let mut payload = Vec::with_capacity(1 + entry_bytes.len());
        payload.push(ddl_type);
        payload.extend_from_slice(entry_bytes);

        let insert_lsn = self.wal.log_insert(txn_id, begin_lsn, &payload)?;
        let commit_lsn = self.wal.log_commit(txn_id, insert_lsn)?;
        self.wal.wait_for_flush(commit_lsn)?;
        // Bump the schema version so any cached PhysicalPlans become stale
        // and the wire layer re-plans on the next reference. AcqRel pairs
        // with the Acquire load in plan-cache lookup.
        self.schema_version
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        Ok(commit_lsn)
    }

    /// Logs several DDL records under one transaction with a single durable
    /// flush, instead of one begin/commit/fsync per record. Cascading DDL
    /// (DROP TABLE removes the table plus each of its indexes and comments)
    /// uses this so drop latency is one flush regardless of how many dependent
    /// objects exist, rather than scaling with their count. The records are
    /// applied in order under one begin/commit and recovery redoes them as a
    /// unit. Returns the commit LSN.
    fn log_ddl_batch(&self, records: &[(u8, Vec<u8>)]) -> Result<Lsn> {
        let txn_id = self.wal.allocate_txn_id()?;
        let mut chain_lsn = self.wal.log_begin(txn_id)?;
        for (ddl_type, entry_bytes) in records {
            let mut payload = Vec::with_capacity(1 + entry_bytes.len());
            payload.push(*ddl_type);
            payload.extend_from_slice(entry_bytes);
            chain_lsn = self.wal.log_insert(txn_id, chain_lsn, &payload)?;
        }
        let commit_lsn = self.wal.log_commit(txn_id, chain_lsn)?;
        self.wal.wait_for_flush(commit_lsn)?;
        self.schema_version
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        Ok(commit_lsn)
    }

    /// Returns the current schema version. Bumped on every DDL.
    #[inline]
    pub fn schema_version(&self) -> u64 {
        self.schema_version
            .load(std::sync::atomic::Ordering::Acquire)
    }

    /// Invalidates every cached physical plan.
    ///
    /// `log_ddl` bumps the version because the catalog changed. State that
    /// lives outside the catalog can change what a plan should be without
    /// writing a catalog record: a clustering commit changes the layout a
    /// scan costs, and a background pass does it with no DDL at all. Those
    /// callers bump it here, otherwise a plan cached against the old
    /// layout keeps being reused after the layout is gone.
    #[inline]
    pub fn bump_schema_version(&self) {
        self.schema_version
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
    }

    /// Persists a mutated table entry (used by ALTER TABLE lifecycle ops).
    /// Re-logs the entry, replaces the stored tuple, and refreshes the cache.
    /// Columns and indexes are unaffected (separate system tables).
    pub async fn update_table(&self, entry: TableEntry) -> Result<()> {
        self.log_ddl(DDL_CREATE_TABLE, &entry.to_bytes())?;
        self.storage.delete_table(entry.id).await?;
        self.storage.store_table(&entry).await?;
        self.cache.put_table(entry);
        Ok(())
    }

    /// Replaces the cached table entry without WAL logging or a storage
    /// rewrite. The compaction worker uses this for the common per-fold
    /// columnar-registry update so a fold is O(1) instead of re-serializing
    /// and re-persisting every prior segment (which is O(segments) per fold,
    /// O(n^2) over a table's life). Durable persistence is amortized via a
    /// periodic `update_table`; a crash before the next durable persist is
    /// reconciled at startup from the WAL `CompactionEnd` records, which is
    /// already the columnar registry's recovery path.
    pub fn cache_put_table(&self, entry: TableEntry) {
        self.cache.put_table(entry);
    }

    // ----- Data lifecycle accessors -----

    pub async fn load_legal_holds(&self) -> Result<Vec<crate::schema::LegalHoldEntry>> {
        self.storage.load_legal_holds().await
    }

    pub async fn store_legal_hold(&self, e: &crate::schema::LegalHoldEntry) -> Result<()> {
        self.storage.store_legal_hold(e).await?;
        Ok(())
    }

    pub async fn update_legal_hold(&self, e: &crate::schema::LegalHoldEntry) -> Result<bool> {
        self.storage.update_legal_hold(e).await
    }

    pub async fn delete_legal_hold(&self, id: u32) -> Result<bool> {
        self.storage.delete_legal_hold(id).await
    }

    pub async fn load_retention_policies(
        &self,
    ) -> Result<Vec<crate::schema::RetentionPolicyEntry>> {
        self.storage.load_retention_policies().await
    }

    pub async fn replace_retention_policies(
        &self,
        table_id: u32,
        entries: &[crate::schema::RetentionPolicyEntry],
    ) -> Result<()> {
        self.storage
            .replace_retention_policies(table_id, entries)
            .await
    }

    pub async fn load_retention_jobs(&self) -> Result<Vec<crate::schema::RetentionJobEntry>> {
        self.storage.load_retention_jobs().await
    }

    pub async fn store_retention_job(&self, e: &crate::schema::RetentionJobEntry) -> Result<()> {
        self.storage.store_retention_job(e).await?;
        Ok(())
    }

    pub async fn load_compliance_log(&self) -> Result<Vec<crate::schema::ComplianceLogEntry>> {
        self.storage.load_compliance_log().await
    }

    /// Appends a compliance log entry, chaining its hash over the latest
    /// entry's hash so the audit log is tamper-evident.
    pub async fn append_compliance_log(
        &self,
        mut entry: crate::schema::ComplianceLogEntry,
    ) -> Result<()> {
        // Serialize the read-modify-write so concurrent appends chain off the
        // same tail in sequence rather than racing and forking the chain.
        let _guard = self.compliance_append_lock.lock().await;
        let existing = self.storage.load_compliance_log().await?;
        let prev_hash = existing.last().map(|e| e.entry_hash).unwrap_or(0);
        let next_id = existing.last().map(|e| e.event_id + 1).unwrap_or(1);
        entry.event_id = next_id;
        entry.prev_hash = prev_hash;
        entry.entry_hash = entry.compute_hash();
        self.storage.store_compliance_log(&entry).await?;
        Ok(())
    }

    /// Verifies the compliance log hash chain. Returns the count of verified
    /// entries and whether the whole chain is intact.
    pub async fn verify_compliance_chain(&self) -> Result<(usize, bool)> {
        let log = self.storage.load_compliance_log().await?;
        let mut prev = 0u32;
        let mut verified = 0usize;
        let mut intact = true;
        for e in &log {
            if e.prev_hash != prev || e.entry_hash != e.compute_hash() {
                intact = false;
                break;
            }
            prev = e.entry_hash;
            verified += 1;
        }
        Ok((verified, intact))
    }
}

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

/// Converts parser ColumnDefs to catalog ColumnEntries.
/// Column count must already be validated to fit in u16.
fn convert_column_defs(table_id: TableId, defs: &[ColumnDef]) -> Result<Vec<ColumnEntry>> {
    let mut entries = Vec::with_capacity(defs.len());
    for (i, def) in defs.iter().enumerate() {
        let type_id = def.data_type.to_type_id();
        let max_length = extract_max_length(&def.data_type);
        let nullable = def.nullable.unwrap_or(true);
        // Store the default as re-parseable SQL so INSERT can fill an omitted
        // column with it (a debug rendering would not round-trip).
        let default_expr = def.default.as_ref().map(zyron_parser::expr_to_sql);

        entries.push(ColumnEntry {
            id: ColumnId(i as u16),
            table_id,
            name: def.name.clone(),
            type_id,
            ordinal: i as u16,
            nullable,
            default_expr,
            max_length,
            fractional_digits: def.data_type.fractional_digits(),
            tz_offset_secs: None,
            element_type: extract_element_type(&def.data_type),
        });
    }
    Ok(entries)
}

/// The element type of an `T[]` declaration, so array values written to the
/// column are re-encoded to the width it declares.
#[inline]
fn extract_element_type(dt: &DataType) -> Option<zyron_common::TypeId> {
    dt.declared_element_type()
}

/// Extracts the declared size of a sized type.
///
/// Character and binary types measure a length, a vector measures its
/// dimension count, and a decimal measures its total digits. All three are
/// the one number the declaration bounds the value by, so they share the
/// slot and each write path reads it under its own type.
#[inline]
fn extract_max_length(dt: &DataType) -> Option<usize> {
    dt.declared_max_length()
}

/// Converts parser TableConstraints to catalog ConstraintEntries. Foreign-key
/// reference targets are resolved by the caller (create_table) which has
/// catalog access; here a ForeignKey records its local columns with the
/// reference left unresolved.
fn convert_table_constraints(
    constraints: &[TableConstraint],
    columns: &[ColumnEntry],
) -> Result<Vec<ConstraintEntry>> {
    let mut result = Vec::with_capacity(constraints.len());
    for tc in constraints {
        let entry = match &tc.kind {
            TableConstraintKind::PrimaryKey(col_names) => ConstraintEntry {
                name: tc
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("pk_{}", col_names.join("_"))),
                constraint_type: ConstraintType::PrimaryKey,
                columns: resolve_column_ids(col_names, columns)?,
                ref_table_id: None,
                ref_columns: vec![],
                check_expr: None,
                on_delete: ReferentialAction::NoAction,
                on_update: ReferentialAction::NoAction,
                enforced: tc.enforced,
                on_violation: map_violation_action(tc.on_violation),
                quarantine_table_id: None,
            },
            TableConstraintKind::Unique(col_names) => ConstraintEntry {
                name: tc
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("uq_{}", col_names.join("_"))),
                constraint_type: ConstraintType::Unique,
                columns: resolve_column_ids(col_names, columns)?,
                ref_table_id: None,
                ref_columns: vec![],
                check_expr: None,
                on_delete: ReferentialAction::NoAction,
                on_update: ReferentialAction::NoAction,
                enforced: tc.enforced,
                on_violation: map_violation_action(tc.on_violation),
                quarantine_table_id: None,
            },
            TableConstraintKind::Check(expr) => ConstraintEntry {
                name: tc.name.clone().unwrap_or_else(|| "ck_table".to_string()),
                constraint_type: ConstraintType::Check,
                columns: vec![],
                ref_table_id: None,
                ref_columns: vec![],
                check_expr: Some(zyron_parser::expr_to_sql(expr)),
                on_delete: ReferentialAction::NoAction,
                on_update: ReferentialAction::NoAction,
                enforced: tc.enforced,
                on_violation: map_violation_action(tc.on_violation),
                quarantine_table_id: None,
            },
            TableConstraintKind::ForeignKey {
                columns: col_names,
                ref_table: _,
                ref_columns: _,
                on_delete,
                on_update,
            } => ConstraintEntry {
                name: tc
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("fk_{}", col_names.join("_"))),
                constraint_type: ConstraintType::ForeignKey,
                columns: resolve_column_ids(col_names, columns)?,
                ref_table_id: None,
                ref_columns: vec![],
                check_expr: None,
                on_delete: map_ref_action(*on_delete),
                on_update: map_ref_action(*on_update),
                enforced: tc.enforced,
                on_violation: map_violation_action(tc.on_violation),
                quarantine_table_id: None,
            },
        };
        result.push(entry);
    }
    Ok(result)
}

/// Maps a parser referential action to its catalog representation.
fn map_ref_action(a: zyron_parser::ast::ReferentialAction) -> ReferentialAction {
    use zyron_parser::ast::ReferentialAction as P;
    match a {
        P::NoAction => ReferentialAction::NoAction,
        P::Restrict => ReferentialAction::Restrict,
        P::Cascade => ReferentialAction::Cascade,
        P::SetNull => ReferentialAction::SetNull,
        P::SetDefault => ReferentialAction::SetDefault,
    }
}

/// Resolves column names to ColumnIds. Returns an error if any column name is not found.
/// Maps the parsed violation mode onto its catalog form. Only Fail and
/// Quarantine are constraint modes, the expectation-only actions have no
/// meaning for a constraint and read as Fail.
fn map_violation_action(action: zyron_parser::ast::ViolationAction) -> ConstraintViolationAction {
    match action {
        zyron_parser::ast::ViolationAction::Quarantine => ConstraintViolationAction::Quarantine,
        _ => ConstraintViolationAction::Fail,
    }
}

fn resolve_column_ids(names: &[String], columns: &[ColumnEntry]) -> Result<Vec<ColumnId>> {
    let mut ids = Vec::with_capacity(names.len());
    for name in names {
        let col = columns
            .iter()
            .find(|c| c.name == *name)
            .ok_or_else(|| ZyronError::ColumnNotFound(name.clone()))?;
        ids.push(col.id);
    }
    Ok(ids)
}

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyron_common::TypeId;

    #[test]
    fn test_convert_column_defs() {
        let defs = vec![
            ColumnDef {
                name: "id".to_string(),
                data_type: DataType::BigInt,
                nullable: Some(false),
                default: None,
                constraints: vec![ColumnConstraint::PrimaryKey],
            },
            ColumnDef {
                name: "email".to_string(),
                data_type: DataType::Varchar(Some(255)),
                nullable: None,
                default: None,
                constraints: vec![],
            },
        ];

        let cols = convert_column_defs(TableId(1), &defs).unwrap();
        assert_eq!(cols.len(), 2);
        assert_eq!(cols[0].name, "id");
        assert_eq!(cols[0].type_id, TypeId::Int64);
        assert_eq!(cols[0].nullable, false);
        assert_eq!(cols[0].ordinal, 0);
        assert_eq!(cols[1].name, "email");
        assert_eq!(cols[1].type_id, TypeId::Varchar);
        assert_eq!(cols[1].nullable, true);
        assert_eq!(cols[1].max_length, Some(255));
    }

    #[test]
    fn test_extract_max_length() {
        assert_eq!(extract_max_length(&DataType::Varchar(Some(100))), Some(100));
        assert_eq!(extract_max_length(&DataType::Char(None)), None);
        assert_eq!(extract_max_length(&DataType::Vector(Some(128))), Some(128));
        assert_eq!(extract_max_length(&DataType::Int), None);
        assert_eq!(extract_max_length(&DataType::Text), None);
    }

    #[test]
    fn test_convert_table_constraints() {
        let cols = vec![
            ColumnEntry {
                id: ColumnId(0),
                table_id: TableId(1),
                name: "a".to_string(),
                type_id: TypeId::Int32,
                ordinal: 0,
                nullable: false,
                default_expr: None,
                max_length: None,
                fractional_digits: None,
                tz_offset_secs: None,
                element_type: None,
            },
            ColumnEntry {
                id: ColumnId(1),
                table_id: TableId(1),
                name: "b".to_string(),
                type_id: TypeId::Int32,
                ordinal: 1,
                nullable: false,
                default_expr: None,
                max_length: None,
                fractional_digits: None,
                tz_offset_secs: None,
                element_type: None,
            },
        ];
        let tcs = vec![
            TableConstraint {
                name: None,
                kind: TableConstraintKind::PrimaryKey(vec!["a".to_string()]),
                enforced: true,
                on_violation: zyron_parser::ast::ViolationAction::Fail,
            },
            TableConstraint {
                name: None,
                kind: TableConstraintKind::Unique(vec!["a".to_string(), "b".to_string()]),
                enforced: true,
                on_violation: zyron_parser::ast::ViolationAction::Fail,
            },
        ];
        let result = convert_table_constraints(&tcs, &cols).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].constraint_type, ConstraintType::PrimaryKey);
        assert_eq!(result[0].columns, vec![ColumnId(0)]);
        assert_eq!(result[1].constraint_type, ConstraintType::Unique);
        assert_eq!(result[1].columns, vec![ColumnId(0), ColumnId(1)]);
    }
}

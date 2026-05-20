//! Central catalog manager for ZyronDB.
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
use zyron_parser::ast::{ColumnConstraint, ColumnDef, DataType, TableConstraint};
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
        let rm = match RecoveryManager::new(&wal_dir) {
            Ok(rm) => rm,
            Err(_) => return Ok(()),
        };
        let result = match rm.recover() {
            Ok(r) => r,
            Err(_) => return Ok(()),
        };
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
                _ => None,
            }
        }

        let mut latest: HashMap<(u8, u64), zyron_wal::record::LogRecord> = HashMap::new();
        for record in redo
            .into_iter()
            .filter(|r| r.record_type == LogRecordType::Insert && !r.payload.is_empty())
        {
            let ddl_type = record.payload[0];
            let entry_bytes = &record.payload[1..];
            if let Some(key) = entity_key(ddl_type, entry_bytes) {
                latest.insert(key, record);
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
                        if !have_indexes.contains(&entry.id.0) {
                            let _ = self.storage.store_index(&entry).await;
                            have_indexes.insert(entry.id.0);
                        }
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

        self.oid_allocator.reset(max_oid);
        Ok(())
    }

    /// Allocates the next OID.
    pub fn next_oid(&self) -> Oid {
        self.oid_allocator.next()
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
                        });
                    }
                    ColumnConstraint::Check(expr) => {
                        constraints.push(ConstraintEntry {
                            name: format!("ck_{}_{}", name, col_def.name),
                            constraint_type: ConstraintType::Check,
                            columns: vec![col_id],
                            ref_table_id: None,
                            ref_columns: vec![],
                            check_expr: Some(format!("{:?}", expr)),
                        });
                    }
                    ColumnConstraint::References {
                        table: _,
                        column: _,
                    } => {
                        // Foreign key references need the referenced table to be resolved.
                        // Store the constraint with the reference info as a string for now.
                        // Full resolution happens at query time.
                        constraints.push(ConstraintEntry {
                            name: format!("fk_{}_{}", name, col_def.name),
                            constraint_type: ConstraintType::ForeignKey,
                            columns: vec![col_id],
                            ref_table_id: None,
                            ref_columns: vec![],
                            check_expr: None,
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
        };

        self.log_ddl(DDL_CREATE_TABLE, &entry.to_bytes())?;
        self.storage.store_table(&entry).await?;
        self.cache.put_table(entry);
        Ok(table_id)
    }

    pub async fn drop_table(&self, schema_id: SchemaId, name: &str) -> Result<()> {
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
        let mut payload = vec![0u8; 4];
        payload[..4].copy_from_slice(&id.0.to_le_bytes());
        self.log_ddl(DDL_DROP_TABLE, &payload)?;
        self.storage.delete_table(id).await?;
        self.cache.invalidate_table(id);
        Ok(())
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
        let txn_id = self.wal.allocate_txn_id();
        let begin_lsn = self.wal.log_begin(txn_id)?;

        // Build DDL payload: 1-byte type prefix + entry bytes
        let mut payload = Vec::with_capacity(1 + entry_bytes.len());
        payload.push(ddl_type);
        payload.extend_from_slice(entry_bytes);

        let insert_lsn = self.wal.log_insert(txn_id, begin_lsn, &payload)?;
        let commit_lsn = self.wal.log_commit(txn_id, insert_lsn)?;
        self.wal.wait_for_flush(commit_lsn)?;
        Ok(commit_lsn)
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

    // ----- Phase 17 data lifecycle accessors -----

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
        let default_expr = def.default.as_ref().map(|e| format!("{:?}", e));

        entries.push(ColumnEntry {
            id: ColumnId(i as u16),
            table_id,
            name: def.name.clone(),
            type_id,
            ordinal: i as u16,
            nullable,
            default_expr,
            max_length,
            ts_precision: def.data_type.timestamp_precision(),
            tz_offset_secs: None,
        });
    }
    Ok(entries)
}

/// Extracts the max_length parameter from sized data types.
fn extract_max_length(dt: &DataType) -> Option<usize> {
    match dt {
        DataType::Char(n)
        | DataType::Varchar(n)
        | DataType::Binary(n)
        | DataType::Varbinary(n)
        | DataType::Vector(n) => *n,
        _ => None,
    }
}

/// Converts parser TableConstraints to catalog ConstraintEntries.
fn convert_table_constraints(
    constraints: &[TableConstraint],
    columns: &[ColumnEntry],
) -> Result<Vec<ConstraintEntry>> {
    let mut result = Vec::with_capacity(constraints.len());
    for tc in constraints {
        let entry = match tc {
            TableConstraint::PrimaryKey(col_names) => ConstraintEntry {
                name: format!("pk_{}", col_names.join("_")),
                constraint_type: ConstraintType::PrimaryKey,
                columns: resolve_column_ids(col_names, columns)?,
                ref_table_id: None,
                ref_columns: vec![],
                check_expr: None,
            },
            TableConstraint::Unique(col_names) => ConstraintEntry {
                name: format!("uq_{}", col_names.join("_")),
                constraint_type: ConstraintType::Unique,
                columns: resolve_column_ids(col_names, columns)?,
                ref_table_id: None,
                ref_columns: vec![],
                check_expr: None,
            },
            TableConstraint::Check(expr) => ConstraintEntry {
                name: "ck_table".to_string(),
                constraint_type: ConstraintType::Check,
                columns: vec![],
                ref_table_id: None,
                ref_columns: vec![],
                check_expr: Some(format!("{:?}", expr)),
            },
            TableConstraint::ForeignKey {
                columns: col_names,
                ref_table: _,
                ref_columns: _,
            } => ConstraintEntry {
                name: format!("fk_{}", col_names.join("_")),
                constraint_type: ConstraintType::ForeignKey,
                columns: resolve_column_ids(col_names, columns)?,
                ref_table_id: None,
                ref_columns: vec![],
                check_expr: None,
            },
        };
        result.push(entry);
    }
    Ok(result)
}

/// Resolves column names to ColumnIds. Returns an error if any column name is not found.
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
                ts_precision: None,
                tz_offset_secs: None,
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
                ts_precision: None,
                tz_offset_secs: None,
            },
        ];
        let tcs = vec![
            TableConstraint::PrimaryKey(vec!["a".to_string()]),
            TableConstraint::Unique(vec!["a".to_string(), "b".to_string()]),
        ];
        let result = convert_table_constraints(&tcs, &cols).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].constraint_type, ConstraintType::PrimaryKey);
        assert_eq!(result[0].columns, vec![ColumnId(0)]);
        assert_eq!(result[1].constraint_type, ConstraintType::Unique);
        assert_eq!(result[1].columns, vec![ColumnId(0), ColumnId(1)]);
    }
}

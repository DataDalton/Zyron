//! In-memory cache for catalog entries with LRU eviction.
//!
//! ID-based maps use RwLock<LruMap> for access-time tracking.
//! Name-based maps use scc::HashMap with pre-hashed u64 keys and an
//! identity hasher for lock-free concurrent reads. Every read verifies
//! the entry name to guarantee correctness on hash collision (treated
//! as a cache miss, never silent corruption). Invalidation also verifies
//! before removing to prevent accidental eviction of a different entry.

use crate::ids::*;
use crate::schema::*;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::hash::{BuildHasher, Hasher};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Fast hash + identity hasher for zero-allocation, zero-double-hash lookups
// ---------------------------------------------------------------------------

/// FxHash-style rotate-xor-multiply hash for (u32 id, name) pairs.
/// Produces well-distributed u64 keys for catalog name lookups without
/// allocation. Not cryptographic, but collision-safe because all reads
/// verify the actual entry name before returning.
#[inline]
fn name_key(id: u32, name: &str) -> u64 {
    const K: u64 = 0x517cc1b727220a95;
    let mut h = (id as u64).wrapping_mul(K);
    for &b in name.as_bytes() {
        h = (h.rotate_left(5) ^ b as u64).wrapping_mul(K);
    }
    // Avalanche mix for good bit distribution across scc shards
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h
}

/// Passes pre-hashed u64 keys through without re-hashing.
/// Safe because name_key() already produces well-distributed output.
#[derive(Default)]
struct IdentityHasher(u64);

impl Hasher for IdentityHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, _bytes: &[u8]) {}

    #[inline]
    fn write_u64(&mut self, n: u64) {
        self.0 = n;
    }
}

#[derive(Clone, Default)]
struct IdentityBuildHasher;

impl BuildHasher for IdentityBuildHasher {
    type Hasher = IdentityHasher;

    #[inline]
    fn build_hasher(&self) -> IdentityHasher {
        IdentityHasher(0)
    }
}

type NameMap<V> = scc::HashMap<u64, V, IdentityBuildHasher>;

// ---------------------------------------------------------------------------
// CatalogCache
// ---------------------------------------------------------------------------

/// In-memory catalog cache with LRU eviction for tables and schemas.
pub struct CatalogCache {
    // Database entries (small cardinality, no LRU needed)
    databases: RwLock<HashMap<DatabaseId, Arc<DatabaseEntry>>>,
    database_names: NameMap<Arc<DatabaseEntry>>,

    // Schema entries with LRU (ID-based) + lock-free name lookup
    schemas: RwLock<LruMap<SchemaId, Arc<SchemaEntry>>>,
    schema_by_name: NameMap<Arc<SchemaEntry>>,

    // Table entries with LRU (ID-based) + lock-free name lookup
    tables: RwLock<LruMap<TableId, Arc<TableEntry>>>,
    table_by_name: NameMap<Arc<TableEntry>>,

    // Index entries (keyed by IndexId, with table_id reverse index)
    indexes: RwLock<HashMap<IndexId, Arc<IndexEntry>>>,
    table_indexes: RwLock<HashMap<TableId, Vec<IndexId>>>,

    // Streaming job entries. The name-index keys on the job's source_schema_id
    // since streaming jobs do not have a dedicated owning schema yet.
    streaming_jobs: RwLock<HashMap<StreamingJobId, Arc<StreamingJobEntry>>>,
    streaming_jobs_by_name: RwLock<HashMap<(SchemaId, String), StreamingJobId>>,

    // External source and sink entries, keyed by ID with name-based reverse maps.
    external_sources: RwLock<HashMap<ExternalSourceId, Arc<ExternalSourceEntry>>>,
    external_sources_by_name: RwLock<HashMap<(SchemaId, String), ExternalSourceId>>,
    external_sinks: RwLock<HashMap<ExternalSinkId, Arc<ExternalSinkEntry>>>,
    external_sinks_by_name: RwLock<HashMap<(SchemaId, String), ExternalSinkId>>,

    // Publication, subscription, endpoint, security-map entries (Zyron-to-Zyron).
    publications: RwLock<HashMap<PublicationId, Arc<PublicationEntry>>>,
    publications_by_name: RwLock<HashMap<(SchemaId, String), PublicationId>>,
    publication_tables: RwLock<HashMap<u32, Arc<PublicationTableEntry>>>,
    publication_tables_by_pub: RwLock<HashMap<PublicationId, Vec<u32>>>,
    subscriptions: RwLock<HashMap<SubscriptionId, Arc<SubscriptionEntry>>>,
    subscriptions_by_pub: RwLock<HashMap<PublicationId, Vec<SubscriptionId>>>,
    endpoints: RwLock<HashMap<EndpointId, Arc<EndpointEntry>>>,
    endpoints_by_name: RwLock<HashMap<(SchemaId, String), EndpointId>>,
    endpoints_by_path: RwLock<HashMap<String, EndpointId>>,
    security_maps: RwLock<HashMap<SecurityMapId, Arc<SecurityMapEntry>>>,
    security_map_lookup: RwLock<HashMap<(SecurityMapKind, String), SecurityMapId>>,
}

/// Simple LRU map using a HashMap with access timestamps.
/// Evicts the least recently accessed entry when capacity is exceeded.
pub struct LruMap<K, V> {
    entries: HashMap<K, (V, u64)>,
    capacity: usize,
    clock: AtomicU64,
}

impl<K: std::hash::Hash + Eq + Copy, V> LruMap<K, V> {
    fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::with_capacity(capacity),
            capacity,
            clock: AtomicU64::new(1),
        }
    }

    fn get(&self, key: &K) -> Option<&V> {
        self.entries.get(key).map(|(v, _)| v)
    }

    fn get_mut(&mut self, key: &K) -> Option<&V> {
        let ts = self.clock.fetch_add(1, Ordering::Relaxed);
        self.entries.get_mut(key).map(|(v, t)| {
            *t = ts;
            &*v
        })
    }

    fn insert(&mut self, key: K, value: V) {
        let ts = self.clock.fetch_add(1, Ordering::Relaxed);
        if self.entries.len() >= self.capacity && !self.entries.contains_key(&key) {
            self.evict_one();
        }
        self.entries.insert(key, (value, ts));
    }

    fn remove(&mut self, key: &K) -> Option<V> {
        self.entries.remove(key).map(|(v, _)| v)
    }

    fn clear(&mut self) {
        self.entries.clear();
    }

    fn values(&self) -> impl Iterator<Item = &V> {
        self.entries.values().map(|(v, _)| v)
    }

    fn evict_one(&mut self) {
        if let Some((&evict_key, _)) = self.entries.iter().min_by_key(|(_, (_, ts))| *ts) {
            self.entries.remove(&evict_key);
        }
    }
}

fn new_name_map<V>() -> NameMap<V> {
    scc::HashMap::with_hasher(IdentityBuildHasher)
}

impl CatalogCache {
    /// Creates a new catalog cache with the given capacity limits.
    pub fn new(max_tables: usize, max_schemas: usize) -> Self {
        Self {
            databases: RwLock::new(HashMap::new()),
            database_names: new_name_map(),
            schemas: RwLock::new(LruMap::new(max_schemas)),
            schema_by_name: new_name_map(),
            tables: RwLock::new(LruMap::new(max_tables)),
            table_by_name: new_name_map(),
            indexes: RwLock::new(HashMap::new()),
            table_indexes: RwLock::new(HashMap::new()),
            streaming_jobs: RwLock::new(HashMap::new()),
            streaming_jobs_by_name: RwLock::new(HashMap::new()),
            external_sources: RwLock::new(HashMap::new()),
            external_sources_by_name: RwLock::new(HashMap::new()),
            external_sinks: RwLock::new(HashMap::new()),
            external_sinks_by_name: RwLock::new(HashMap::new()),
            publications: RwLock::new(HashMap::new()),
            publications_by_name: RwLock::new(HashMap::new()),
            publication_tables: RwLock::new(HashMap::new()),
            publication_tables_by_pub: RwLock::new(HashMap::new()),
            subscriptions: RwLock::new(HashMap::new()),
            subscriptions_by_pub: RwLock::new(HashMap::new()),
            endpoints: RwLock::new(HashMap::new()),
            endpoints_by_name: RwLock::new(HashMap::new()),
            endpoints_by_path: RwLock::new(HashMap::new()),
            security_maps: RwLock::new(HashMap::new()),
            security_map_lookup: RwLock::new(HashMap::new()),
        }
    }

    // -----------------------------------------------------------------------
    // Database operations
    // -----------------------------------------------------------------------

    pub fn get_database(&self, id: DatabaseId) -> Option<Arc<DatabaseEntry>> {
        self.databases.read().get(&id).cloned()
    }

    pub fn get_database_by_name(&self, name: &str) -> Option<Arc<DatabaseEntry>> {
        let key = name_key(0, name);
        self.database_names
            .read_sync(&key, |_, entry| {
                if entry.name == name {
                    Some(Arc::clone(entry))
                } else {
                    None
                }
            })
            .flatten()
    }

    pub fn put_database(&self, entry: DatabaseEntry) {
        let id = entry.id;
        let key = name_key(0, &entry.name);
        let arc = Arc::new(entry);
        let _ = self.database_names.insert_sync(key, Arc::clone(&arc));
        self.databases.write().insert(id, arc);
    }

    pub fn invalidate_database(&self, id: DatabaseId) {
        if let Some(entry) = self.databases.write().remove(&id) {
            let key = name_key(0, &entry.name);
            self.database_names
                .remove_if_sync(&key, |v| v.name == entry.name);
        }
    }

    // -----------------------------------------------------------------------
    // Schema operations
    // -----------------------------------------------------------------------

    pub fn get_schema(&self, id: SchemaId) -> Option<Arc<SchemaEntry>> {
        self.schemas.read().get(&id).cloned()
    }

    /// Lock-free name lookup with verify-on-read for collision safety.
    pub fn get_schema_by_name(&self, db_id: DatabaseId, name: &str) -> Option<Arc<SchemaEntry>> {
        let key = name_key(db_id.0, name);
        self.schema_by_name
            .read_sync(&key, |_, entry| {
                if entry.database_id == db_id && entry.name == name {
                    Some(Arc::clone(entry))
                } else {
                    None
                }
            })
            .flatten()
    }

    pub fn put_schema(&self, entry: SchemaEntry) {
        let id = entry.id;
        let key = name_key(entry.database_id.0, &entry.name);
        let arc = Arc::new(entry);
        self.schemas.write().insert(id, Arc::clone(&arc));
        let _ = self.schema_by_name.insert_sync(key, arc);
    }

    pub fn invalidate_schema(&self, id: SchemaId) {
        if let Some(entry) = self.schemas.write().remove(&id) {
            let key = name_key(entry.database_id.0, &entry.name);
            self.schema_by_name
                .remove_if_sync(&key, |v| v.id == entry.id);
        }
    }

    // -----------------------------------------------------------------------
    // Table operations
    // -----------------------------------------------------------------------

    pub fn get_table(&self, id: TableId) -> Option<Arc<TableEntry>> {
        self.tables.read().get(&id).cloned()
    }

    pub fn get_table_mut(&self, id: TableId) -> Option<Arc<TableEntry>> {
        self.tables.write().get_mut(&id).cloned()
    }

    /// Lock-free name lookup with verify-on-read for collision safety.
    pub fn get_table_by_name(&self, schema_id: SchemaId, name: &str) -> Option<Arc<TableEntry>> {
        let key = name_key(schema_id.0, name);
        self.table_by_name
            .read_sync(&key, |_, entry| {
                if entry.schema_id == schema_id && entry.name == name {
                    Some(Arc::clone(entry))
                } else {
                    None
                }
            })
            .flatten()
    }

    pub fn put_table(&self, entry: TableEntry) {
        let id = entry.id;
        let key = name_key(entry.schema_id.0, &entry.name);
        let arc = Arc::new(entry);
        self.tables.write().insert(id, Arc::clone(&arc));
        let _ = self.table_by_name.insert_sync(key, arc);
    }

    pub fn invalidate_table(&self, id: TableId) {
        if let Some(entry) = self.tables.write().remove(&id) {
            let key = name_key(entry.schema_id.0, &entry.name);
            self.table_by_name
                .remove_if_sync(&key, |v| v.id == entry.id);
        }
    }

    /// Returns all cached tables for a given schema.
    pub fn list_tables(&self, schema_id: SchemaId) -> Vec<Arc<TableEntry>> {
        self.tables
            .read()
            .values()
            .filter(|t| t.schema_id == schema_id)
            .cloned()
            .collect()
    }

    /// Returns all cached tables across all schemas.
    pub fn list_all_tables(&self) -> Vec<Arc<TableEntry>> {
        self.tables.read().values().cloned().collect()
    }

    // -----------------------------------------------------------------------
    // Index operations
    // -----------------------------------------------------------------------

    pub fn get_index(&self, id: IndexId) -> Option<Arc<IndexEntry>> {
        self.indexes.read().get(&id).cloned()
    }

    pub fn get_indexes_for_table(&self, table_id: TableId) -> Vec<Arc<IndexEntry>> {
        let idx_map = self.table_indexes.read();
        let idx_store = self.indexes.read();
        match idx_map.get(&table_id) {
            Some(ids) => ids
                .iter()
                .filter_map(|id| idx_store.get(id).cloned())
                .collect(),
            None => Vec::new(),
        }
    }

    pub fn put_index(&self, entry: IndexEntry) {
        let id = entry.id;
        let table_id = entry.table_id;
        let arc = Arc::new(entry);
        self.indexes.write().insert(id, arc);
        self.table_indexes
            .write()
            .entry(table_id)
            .or_default()
            .push(id);
    }

    pub fn invalidate_index(&self, id: IndexId) {
        if let Some(entry) = self.indexes.write().remove(&id) {
            let mut ti = self.table_indexes.write();
            if let Some(ids) = ti.get_mut(&entry.table_id) {
                ids.retain(|i| *i != id);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Streaming job operations
    // -----------------------------------------------------------------------

    pub fn put_streaming_job(&self, entry: StreamingJobEntry) {
        let id = entry.id;
        // Owning schema for name lookup uses source_schema_id since streaming
        // jobs do not have a dedicated owning schema field.
        let key = (entry.source_schema_id, entry.name.clone());
        let arc = Arc::new(entry);
        self.streaming_jobs.write().insert(id, arc);
        self.streaming_jobs_by_name.write().insert(key, id);
    }

    pub fn get_streaming_job(&self, id: StreamingJobId) -> Option<Arc<StreamingJobEntry>> {
        self.streaming_jobs.read().get(&id).cloned()
    }

    pub fn get_streaming_job_by_name(
        &self,
        schema_id: SchemaId,
        name: &str,
    ) -> Option<Arc<StreamingJobEntry>> {
        let id = {
            let map = self.streaming_jobs_by_name.read();
            map.get(&(schema_id, name.to_string())).copied()
        }?;
        self.streaming_jobs.read().get(&id).cloned()
    }

    pub fn list_streaming_jobs(&self) -> Vec<Arc<StreamingJobEntry>> {
        self.streaming_jobs.read().values().cloned().collect()
    }

    pub fn invalidate_streaming_job(&self, id: StreamingJobId) {
        if let Some(entry) = self.streaming_jobs.write().remove(&id) {
            let key = (entry.source_schema_id, entry.name.clone());
            let mut name_map = self.streaming_jobs_by_name.write();
            if let Some(existing) = name_map.get(&key) {
                if *existing == id {
                    name_map.remove(&key);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // External source operations
    // -----------------------------------------------------------------------

    pub fn put_external_source(&self, entry: ExternalSourceEntry) {
        let id = entry.id;
        let key = (entry.schema_id, entry.name.clone());
        let arc = Arc::new(entry);
        self.external_sources.write().insert(id, arc);
        self.external_sources_by_name.write().insert(key, id);
    }

    pub fn get_external_source(&self, id: ExternalSourceId) -> Option<Arc<ExternalSourceEntry>> {
        self.external_sources.read().get(&id).cloned()
    }

    pub fn get_external_source_by_name(
        &self,
        schema_id: SchemaId,
        name: &str,
    ) -> Option<Arc<ExternalSourceEntry>> {
        let id = {
            let map = self.external_sources_by_name.read();
            map.get(&(schema_id, name.to_string())).copied()
        }?;
        self.external_sources.read().get(&id).cloned()
    }

    pub fn list_external_sources(&self) -> Vec<Arc<ExternalSourceEntry>> {
        self.external_sources.read().values().cloned().collect()
    }

    pub fn invalidate_external_source(&self, id: ExternalSourceId) {
        if let Some(entry) = self.external_sources.write().remove(&id) {
            let key = (entry.schema_id, entry.name.clone());
            let mut name_map = self.external_sources_by_name.write();
            if let Some(existing) = name_map.get(&key) {
                if *existing == id {
                    name_map.remove(&key);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // External sink operations
    // -----------------------------------------------------------------------

    pub fn put_external_sink(&self, entry: ExternalSinkEntry) {
        let id = entry.id;
        let key = (entry.schema_id, entry.name.clone());
        let arc = Arc::new(entry);
        self.external_sinks.write().insert(id, arc);
        self.external_sinks_by_name.write().insert(key, id);
    }

    pub fn get_external_sink(&self, id: ExternalSinkId) -> Option<Arc<ExternalSinkEntry>> {
        self.external_sinks.read().get(&id).cloned()
    }

    pub fn get_external_sink_by_name(
        &self,
        schema_id: SchemaId,
        name: &str,
    ) -> Option<Arc<ExternalSinkEntry>> {
        let id = {
            let map = self.external_sinks_by_name.read();
            map.get(&(schema_id, name.to_string())).copied()
        }?;
        self.external_sinks.read().get(&id).cloned()
    }

    pub fn list_external_sinks(&self) -> Vec<Arc<ExternalSinkEntry>> {
        self.external_sinks.read().values().cloned().collect()
    }

    pub fn invalidate_external_sink(&self, id: ExternalSinkId) {
        if let Some(entry) = self.external_sinks.write().remove(&id) {
            let key = (entry.schema_id, entry.name.clone());
            let mut name_map = self.external_sinks_by_name.write();
            if let Some(existing) = name_map.get(&key) {
                if *existing == id {
                    name_map.remove(&key);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Publication operations
    // -----------------------------------------------------------------------

    pub fn put_publication(&self, entry: PublicationEntry) {
        let id = entry.id;
        let key = (entry.schema_id, entry.name.clone());
        let arc = Arc::new(entry);
        self.publications.write().insert(id, arc);
        self.publications_by_name.write().insert(key, id);
    }

    pub fn get_publication(&self, id: PublicationId) -> Option<Arc<PublicationEntry>> {
        self.publications.read().get(&id).cloned()
    }

    pub fn get_publication_by_name(
        &self,
        schema_id: SchemaId,
        name: &str,
    ) -> Option<Arc<PublicationEntry>> {
        let id = {
            let map = self.publications_by_name.read();
            map.get(&(schema_id, name.to_string())).copied()
        }?;
        self.publications.read().get(&id).cloned()
    }

    pub fn list_publications(&self) -> Vec<Arc<PublicationEntry>> {
        self.publications.read().values().cloned().collect()
    }

    pub fn invalidate_publication(&self, id: PublicationId) {
        if let Some(entry) = self.publications.write().remove(&id) {
            let key = (entry.schema_id, entry.name.clone());
            let mut name_map = self.publications_by_name.write();
            if let Some(existing) = name_map.get(&key) {
                if *existing == id {
                    name_map.remove(&key);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Publication table junction operations
    // -----------------------------------------------------------------------

    pub fn put_publication_table(&self, entry: PublicationTableEntry) {
        let id = entry.id;
        let pub_id = entry.publication_id;
        let arc = Arc::new(entry);
        self.publication_tables.write().insert(id, arc);
        let mut by_pub = self.publication_tables_by_pub.write();
        let list = by_pub.entry(pub_id).or_default();
        if !list.contains(&id) {
            list.push(id);
        }
    }

    pub fn get_publication_tables(
        &self,
        publication_id: PublicationId,
    ) -> Vec<Arc<PublicationTableEntry>> {
        let ids = {
            let by_pub = self.publication_tables_by_pub.read();
            by_pub.get(&publication_id).cloned().unwrap_or_default()
        };
        let store = self.publication_tables.read();
        ids.iter()
            .filter_map(|id| store.get(id).cloned())
            .collect()
    }

    pub fn invalidate_publication_table(
        &self,
        publication_id: PublicationId,
        table_id: TableId,
    ) {
        let mut drop_id: Option<u32> = None;
        {
            let store = self.publication_tables.read();
            let by_pub = self.publication_tables_by_pub.read();
            if let Some(ids) = by_pub.get(&publication_id) {
                for id in ids {
                    if let Some(entry) = store.get(id) {
                        if entry.table_id == table_id {
                            drop_id = Some(*id);
                            break;
                        }
                    }
                }
            }
        }
        if let Some(id) = drop_id {
            self.publication_tables.write().remove(&id);
            let mut by_pub = self.publication_tables_by_pub.write();
            if let Some(list) = by_pub.get_mut(&publication_id) {
                list.retain(|x| *x != id);
            }
        }
    }

    pub fn invalidate_publication_tables_for(&self, publication_id: PublicationId) {
        let ids = {
            let mut by_pub = self.publication_tables_by_pub.write();
            by_pub.remove(&publication_id).unwrap_or_default()
        };
        let mut store = self.publication_tables.write();
        for id in ids {
            store.remove(&id);
        }
    }

    // -----------------------------------------------------------------------
    // Subscription operations
    // -----------------------------------------------------------------------

    pub fn put_subscription(&self, entry: SubscriptionEntry) {
        let id = entry.id;
        let pub_id = entry.publication_id;
        let arc = Arc::new(entry);
        self.subscriptions.write().insert(id, arc);
        let mut by_pub = self.subscriptions_by_pub.write();
        let list = by_pub.entry(pub_id).or_default();
        if !list.contains(&id) {
            list.push(id);
        }
    }

    pub fn get_subscription(&self, id: SubscriptionId) -> Option<Arc<SubscriptionEntry>> {
        self.subscriptions.read().get(&id).cloned()
    }

    pub fn list_subscriptions(&self) -> Vec<Arc<SubscriptionEntry>> {
        self.subscriptions.read().values().cloned().collect()
    }

    pub fn list_publication_subscribers(
        &self,
        pub_id: PublicationId,
    ) -> Vec<Arc<SubscriptionEntry>> {
        let ids = {
            let by_pub = self.subscriptions_by_pub.read();
            by_pub.get(&pub_id).cloned().unwrap_or_default()
        };
        let store = self.subscriptions.read();
        ids.iter()
            .filter_map(|id| store.get(id).cloned())
            .collect()
    }

    pub fn invalidate_subscription(&self, id: SubscriptionId) {
        if let Some(entry) = self.subscriptions.write().remove(&id) {
            let mut by_pub = self.subscriptions_by_pub.write();
            if let Some(list) = by_pub.get_mut(&entry.publication_id) {
                list.retain(|x| *x != id);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Endpoint operations
    // -----------------------------------------------------------------------

    pub fn put_endpoint(&self, entry: EndpointEntry) {
        let id = entry.id;
        let name_key = (entry.schema_id, entry.name.clone());
        let path_key = entry.path.clone();
        let arc = Arc::new(entry);
        self.endpoints.write().insert(id, arc);
        self.endpoints_by_name.write().insert(name_key, id);
        self.endpoints_by_path.write().insert(path_key, id);
    }

    pub fn get_endpoint(&self, id: EndpointId) -> Option<Arc<EndpointEntry>> {
        self.endpoints.read().get(&id).cloned()
    }

    pub fn get_endpoint_by_name(
        &self,
        schema_id: SchemaId,
        name: &str,
    ) -> Option<Arc<EndpointEntry>> {
        let id = {
            let map = self.endpoints_by_name.read();
            map.get(&(schema_id, name.to_string())).copied()
        }?;
        self.endpoints.read().get(&id).cloned()
    }

    pub fn get_endpoint_by_path(&self, path: &str) -> Option<Arc<EndpointEntry>> {
        let id = {
            let map = self.endpoints_by_path.read();
            map.get(path).copied()
        }?;
        self.endpoints.read().get(&id).cloned()
    }

    pub fn list_endpoints(&self) -> Vec<Arc<EndpointEntry>> {
        self.endpoints.read().values().cloned().collect()
    }

    pub fn invalidate_endpoint(&self, id: EndpointId) {
        if let Some(entry) = self.endpoints.write().remove(&id) {
            let name_key = (entry.schema_id, entry.name.clone());
            let mut name_map = self.endpoints_by_name.write();
            if let Some(existing) = name_map.get(&name_key) {
                if *existing == id {
                    name_map.remove(&name_key);
                }
            }
            let mut path_map = self.endpoints_by_path.write();
            if let Some(existing) = path_map.get(&entry.path) {
                if *existing == id {
                    path_map.remove(&entry.path);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Security map operations
    // -----------------------------------------------------------------------

    pub fn put_security_map(&self, entry: SecurityMapEntry) {
        let id = entry.id;
        let lookup_key = (entry.kind, entry.key.clone());
        let arc = Arc::new(entry);
        self.security_maps.write().insert(id, arc);
        self.security_map_lookup.write().insert(lookup_key, id);
    }

    pub fn get_security_map(&self, id: SecurityMapId) -> Option<Arc<SecurityMapEntry>> {
        self.security_maps.read().get(&id).cloned()
    }

    pub fn list_security_maps(&self) -> Vec<Arc<SecurityMapEntry>> {
        self.security_maps.read().values().cloned().collect()
    }

    pub fn resolve_security_map(&self, kind: SecurityMapKind, key: &str) -> Option<u32> {
        let id = {
            let map = self.security_map_lookup.read();
            map.get(&(kind, key.to_string())).copied()
        }?;
        self.security_maps.read().get(&id).map(|e| e.role_id)
    }

    pub fn invalidate_security_map(&self, id: SecurityMapId) {
        if let Some(entry) = self.security_maps.write().remove(&id) {
            let key = (entry.kind, entry.key.clone());
            let mut map = self.security_map_lookup.write();
            if let Some(existing) = map.get(&key) {
                if *existing == id {
                    map.remove(&key);
                }
            }
        }
    }

    /// Clears all cached entries.
    pub fn invalidate_all(&self) {
        self.databases.write().clear();
        self.database_names.retain_sync(|_, _| false);
        self.schemas.write().clear();
        self.schema_by_name.retain_sync(|_, _| false);
        self.tables.write().clear();
        self.table_by_name.retain_sync(|_, _| false);
        self.indexes.write().clear();
        self.table_indexes.write().clear();
        self.streaming_jobs.write().clear();
        self.streaming_jobs_by_name.write().clear();
        self.external_sources.write().clear();
        self.external_sources_by_name.write().clear();
        self.external_sinks.write().clear();
        self.external_sinks_by_name.write().clear();
        self.publications.write().clear();
        self.publications_by_name.write().clear();
        self.publication_tables.write().clear();
        self.publication_tables_by_pub.write().clear();
        self.subscriptions.write().clear();
        self.subscriptions_by_pub.write().clear();
        self.endpoints.write().clear();
        self.endpoints_by_name.write().clear();
        self.endpoints_by_path.write().clear();
        self.security_maps.write().clear();
        self.security_map_lookup.write().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyron_common::TypeId;

    fn make_db(id: u32, name: &str) -> DatabaseEntry {
        DatabaseEntry {
            id: DatabaseId(id),
            name: name.to_string(),
            owner: "test".to_string(),
            created_at: 0,
        }
    }

    fn make_schema(id: u32, db_id: u32, name: &str) -> SchemaEntry {
        SchemaEntry {
            id: SchemaId(id),
            database_id: DatabaseId(db_id),
            name: name.to_string(),
            owner: "test".to_string(),
        }
    }

    fn make_table(id: u32, schema_id: u32, name: &str) -> TableEntry {
        TableEntry {
            id: TableId(id),
            schema_id: SchemaId(schema_id),
            name: name.to_string(),
            heap_file_id: 200,
            fsm_file_id: 201,
            columns: vec![ColumnEntry {
                id: ColumnId(0),
                table_id: TableId(id),
                name: "id".to_string(),
                type_id: TypeId::Int64,
                ordinal: 0,
                nullable: false,
                default_expr: None,
                max_length: None,
            }],
            constraints: vec![],
            created_at: 0,
            versioning_enabled: false,
            scd_type: None,
            system_versioned: false,
            history_table_id: None,
            cdf_enabled: false,
            cdf_retention_days: 0,
        }
    }

    fn make_index(id: u32, table_id: u32, name: &str) -> IndexEntry {
        IndexEntry {
            id: IndexId(id),
            table_id: TableId(table_id),
            schema_id: SchemaId(1),
            name: name.to_string(),
            columns: vec![IndexColumnEntry {
                column_id: ColumnId(0),
                ordinal: 0,
                descending: false,
            }],
            unique: false,
            index_file_id: 10000,
            index_type: IndexType::BTree,
            parameters: None,
        }
    }

    #[test]
    fn test_database_cache() {
        let cache = CatalogCache::new(100, 100);
        let db = make_db(1, "testdb");
        cache.put_database(db);

        assert!(cache.get_database(DatabaseId(1)).is_some());
        assert_eq!(cache.get_database(DatabaseId(1)).unwrap().name, "testdb");
        assert!(cache.get_database_by_name("testdb").is_some());
        assert!(cache.get_database_by_name("nonexistent").is_none());

        cache.invalidate_database(DatabaseId(1));
        assert!(cache.get_database(DatabaseId(1)).is_none());
        assert!(cache.get_database_by_name("testdb").is_none());
    }

    #[test]
    fn test_schema_cache() {
        let cache = CatalogCache::new(100, 100);
        let schema = make_schema(1, 1, "public");
        cache.put_schema(schema);

        assert!(cache.get_schema(SchemaId(1)).is_some());
        assert!(cache.get_schema_by_name(DatabaseId(1), "public").is_some());
        assert!(cache.get_schema_by_name(DatabaseId(2), "public").is_none());

        cache.invalidate_schema(SchemaId(1));
        assert!(cache.get_schema(SchemaId(1)).is_none());
    }

    #[test]
    fn test_table_cache() {
        let cache = CatalogCache::new(100, 100);
        let table = make_table(10, 1, "users");
        cache.put_table(table);

        assert!(cache.get_table(TableId(10)).is_some());
        assert!(cache.get_table_by_name(SchemaId(1), "users").is_some());
        assert!(cache.get_table_by_name(SchemaId(1), "orders").is_none());

        let tables = cache.list_tables(SchemaId(1));
        assert_eq!(tables.len(), 1);

        cache.invalidate_table(TableId(10));
        assert!(cache.get_table(TableId(10)).is_none());
    }

    #[test]
    fn test_index_cache() {
        let cache = CatalogCache::new(100, 100);
        let idx = make_index(1, 10, "idx_users_id");
        cache.put_index(idx);

        assert!(cache.get_index(IndexId(1)).is_some());
        let idxs = cache.get_indexes_for_table(TableId(10));
        assert_eq!(idxs.len(), 1);

        cache.invalidate_index(IndexId(1));
        assert!(cache.get_index(IndexId(1)).is_none());
        let idxs = cache.get_indexes_for_table(TableId(10));
        assert_eq!(idxs.len(), 0);
    }

    #[test]
    fn test_lru_eviction() {
        let cache = CatalogCache::new(2, 100);

        cache.put_table(make_table(1, 1, "t1"));
        cache.put_table(make_table(2, 1, "t2"));
        // Both should be present
        assert!(cache.get_table(TableId(1)).is_some());
        assert!(cache.get_table(TableId(2)).is_some());

        // Access t1 to make it more recent
        let _ = cache.get_table_mut(TableId(1));

        // Insert t3, should evict t2 (least recently used)
        cache.put_table(make_table(3, 1, "t3"));

        assert!(cache.get_table(TableId(1)).is_some());
        assert!(cache.get_table(TableId(2)).is_none());
        assert!(cache.get_table(TableId(3)).is_some());
    }

    #[test]
    fn test_invalidate_all() {
        let cache = CatalogCache::new(100, 100);
        cache.put_database(make_db(1, "db1"));
        cache.put_schema(make_schema(1, 1, "public"));
        cache.put_table(make_table(1, 1, "t1"));
        cache.put_index(make_index(1, 1, "idx1"));

        cache.invalidate_all();

        assert!(cache.get_database(DatabaseId(1)).is_none());
        assert!(cache.get_schema(SchemaId(1)).is_none());
        assert!(cache.get_table(TableId(1)).is_none());
        assert!(cache.get_index(IndexId(1)).is_none());
    }

    #[test]
    fn test_multiple_indexes_per_table() {
        let cache = CatalogCache::new(100, 100);
        cache.put_index(make_index(1, 10, "idx1"));
        cache.put_index(make_index(2, 10, "idx2"));
        cache.put_index(make_index(3, 20, "idx3"));

        let idxs = cache.get_indexes_for_table(TableId(10));
        assert_eq!(idxs.len(), 2);

        let idxs = cache.get_indexes_for_table(TableId(20));
        assert_eq!(idxs.len(), 1);
    }
}

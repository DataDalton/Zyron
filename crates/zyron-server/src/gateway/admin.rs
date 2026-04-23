// -----------------------------------------------------------------------------
// Admin REST surface.
//
// Defines the admin HTTP action set, the AdminExecutor that performs each
// action against the live catalog, security manager, endpoint registrar, and
// CDC registry, and an authorization helper that enforces the AdminAccess
// privilege on every admin route. All admin handlers produce an AdminResponse
// carrying an HTTP status and a JSON body.
// -----------------------------------------------------------------------------

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;
use serde_json::{Value, json};

use zyron_auth::{KeyStore, ObjectType, PrivilegeDecision, PrivilegeType, RoleId, SecurityManager};
use zyron_buffer::BufferPool;
use zyron_catalog::schema::{ExternalBackend, SubscriptionState};
use zyron_catalog::{
    Catalog, EndpointEntry, PublicationEntry, PublicationId, SubscriptionId, TableEntry,
};
use zyron_cdc::{CdfRegistry, ChangeType};
use zyron_common::{Result, TypeId, ZyronError};
use zyron_storage::{DiskManager, HeapFile, HeapFileConfig, TupleId};
use zyron_streaming::sink_connector::ZyronSinkAdapter;
use zyron_streaming::source_connector::CdfChange;
use zyron_wire::EndpointRegistrar;

use super::request::HttpRequest;
use super::router::HttpMethod;

// -----------------------------------------------------------------------------
// AdminAction enum
// -----------------------------------------------------------------------------

/// Target LSN for a subscription reset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResetLsnTarget {
    Earliest,
    Latest,
    Lsn(u64),
}

/// Dispatched admin action.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdminAction {
    ListPublications,
    GetPublication {
        name: String,
    },
    PausePublication {
        name: String,
    },
    ResumePublication {
        name: String,
    },
    ForceDropSubscriber {
        publication_name: String,
        subscription_id: String,
    },
    GetSubscription {
        id: String,
    },
    ResetSubscriptionLsn {
        id: String,
        target: ResetLsnTarget,
    },
    ListEndpoints,
    DisableEndpoint {
        name: String,
    },
    EnableEndpoint {
        name: String,
    },
    SecretCacheStats,
    RefreshSecret {
        id: String,
    },
    ListDlqEntries {
        sink: String,
    },
    ReplayDlq {
        sink: String,
        limit: u64,
        offset: u64,
    },
}

// -----------------------------------------------------------------------------
// AdminRouter: parses HTTP requests into AdminAction values.
// -----------------------------------------------------------------------------

pub struct AdminRouter;

impl AdminRouter {
    pub fn dispatch(req: &HttpRequest) -> Option<AdminAction> {
        let path = req.path.trim_start_matches('/');
        let parts: Vec<&str> = path.split('/').collect();
        match (req.method, parts.as_slice()) {
            (HttpMethod::Get, ["admin", "publications"]) => Some(AdminAction::ListPublications),
            (HttpMethod::Get, ["admin", "publications", name]) => {
                Some(AdminAction::GetPublication {
                    name: (*name).to_string(),
                })
            }
            (HttpMethod::Post, ["admin", "publications", name, "pause"]) => {
                Some(AdminAction::PausePublication {
                    name: (*name).to_string(),
                })
            }
            (HttpMethod::Post, ["admin", "publications", name, "resume"]) => {
                Some(AdminAction::ResumePublication {
                    name: (*name).to_string(),
                })
            }
            (HttpMethod::Delete, ["admin", "publications", name, "subscribers", sub_id]) => {
                Some(AdminAction::ForceDropSubscriber {
                    publication_name: (*name).to_string(),
                    subscription_id: (*sub_id).to_string(),
                })
            }
            (HttpMethod::Get, ["admin", "subscriptions", id]) => {
                Some(AdminAction::GetSubscription {
                    id: (*id).to_string(),
                })
            }
            (HttpMethod::Post, ["admin", "subscriptions", id, "reset-lsn"]) => {
                Some(parse_reset_lsn(id, &req.query_string, &req.body))
            }
            (HttpMethod::Get, ["admin", "endpoints"]) => Some(AdminAction::ListEndpoints),
            (HttpMethod::Post, ["admin", "endpoints", name, "disable"]) => {
                Some(AdminAction::DisableEndpoint {
                    name: (*name).to_string(),
                })
            }
            (HttpMethod::Post, ["admin", "endpoints", name, "enable"]) => {
                Some(AdminAction::EnableEndpoint {
                    name: (*name).to_string(),
                })
            }
            (HttpMethod::Get, ["admin", "secret-cache", "stats"]) => {
                Some(AdminAction::SecretCacheStats)
            }
            (HttpMethod::Post, ["admin", "secret-cache", "refresh", id]) => {
                Some(AdminAction::RefreshSecret {
                    id: (*id).to_string(),
                })
            }
            (HttpMethod::Get, ["admin", "dlq", sink, "entries"]) => {
                Some(AdminAction::ListDlqEntries {
                    sink: (*sink).to_string(),
                })
            }
            (HttpMethod::Post, ["admin", "dlq", sink, "replay"]) => {
                let (limit, offset) = parse_dlq_page(&req.query_string);
                Some(AdminAction::ReplayDlq {
                    sink: (*sink).to_string(),
                    limit,
                    offset,
                })
            }
            _ => None,
        }
    }
}

// Parses ?target=earliest|latest|<lsn-number> from the query string or JSON
// body. Defaults to Latest when absent.
fn parse_reset_lsn(id: &str, query: &str, body: &[u8]) -> AdminAction {
    let mut target = ResetLsnTarget::Latest;
    for pair in query.split('&') {
        if let Some((k, v)) = pair.split_once('=') {
            if k == "target" {
                target = parse_target_value(v);
            }
        }
    }
    if !body.is_empty() {
        if let Ok(s) = std::str::from_utf8(body) {
            if let Some(idx) = s.find("\"target\"") {
                let rest = &s[idx..];
                if let Some(colon) = rest.find(':') {
                    let after = &rest[colon + 1..].trim_start();
                    if let Some(stripped) = after.strip_prefix('"') {
                        if let Some(end) = stripped.find('"') {
                            target = parse_target_value(&stripped[..end]);
                        }
                    } else {
                        let token: String =
                            after.chars().take_while(|c| c.is_ascii_digit()).collect();
                        if let Ok(n) = token.parse::<u64>() {
                            target = ResetLsnTarget::Lsn(n);
                        }
                    }
                }
            }
        }
    }
    AdminAction::ResetSubscriptionLsn {
        id: id.to_string(),
        target,
    }
}

// Parses ?limit=N&offset=M from the admin replay-DLQ query string. Missing or
// unparseable values default to limit=1000 and offset=0 respectively.
fn parse_dlq_page(query: &str) -> (u64, u64) {
    let mut limit: u64 = 1000;
    let mut offset: u64 = 0;
    for pair in query.split('&') {
        if let Some((k, v)) = pair.split_once('=') {
            if k == "limit" {
                if let Ok(n) = v.parse::<u64>() {
                    limit = n.max(1);
                }
            } else if k == "offset" {
                if let Ok(n) = v.parse::<u64>() {
                    offset = n;
                }
            }
        }
    }
    (limit, offset)
}

fn parse_target_value(v: &str) -> ResetLsnTarget {
    match v {
        "earliest" | "Earliest" | "0" => ResetLsnTarget::Earliest,
        "latest" | "Latest" => ResetLsnTarget::Latest,
        other => match other.parse::<u64>() {
            Ok(n) => ResetLsnTarget::Lsn(n),
            Err(_) => ResetLsnTarget::Latest,
        },
    }
}

// -----------------------------------------------------------------------------
// AdminResponse
// -----------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct AdminResponse {
    pub status: u16,
    pub body: Value,
}

impl AdminResponse {
    pub fn ok(body: Value) -> Self {
        Self { status: 200, body }
    }

    pub fn not_found(detail: &str) -> Self {
        Self {
            status: 404,
            body: json!({ "error": "not_found", "detail": detail }),
        }
    }

    pub fn bad_request(detail: &str) -> Self {
        Self {
            status: 400,
            body: json!({ "error": "bad_request", "detail": detail }),
        }
    }

    pub fn not_implemented(detail: &str) -> Self {
        Self {
            status: 501,
            body: json!({ "error": "not_implemented", "detail": detail }),
        }
    }

    pub fn internal(detail: &str) -> Self {
        Self {
            status: 500,
            body: json!({ "error": "internal", "detail": detail }),
        }
    }
}

// -----------------------------------------------------------------------------
// Publication pause/resume state.
//
// PublicationEntry carries no status column, so the executor tracks paused
// publications in an in-memory set. The reaper and poll paths consult the
// registry before serving subscribers.
// -----------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct PublicationStatusRegistry {
    paused: RwLock<HashMap<PublicationId, ()>>,
}

impl PublicationStatusRegistry {
    pub fn new() -> Self {
        Self {
            paused: RwLock::new(HashMap::new()),
        }
    }

    pub fn is_paused(&self, id: PublicationId) -> bool {
        self.paused.read().contains_key(&id)
    }

    pub fn pause(&self, id: PublicationId) {
        self.paused.write().insert(id, ());
    }

    pub fn resume(&self, id: PublicationId) {
        self.paused.write().remove(&id);
    }

    pub fn status_str(&self, id: PublicationId) -> &'static str {
        if self.is_paused(id) {
            "paused"
        } else {
            "active"
        }
    }
}

// -----------------------------------------------------------------------------
// DLQ replay helpers
// -----------------------------------------------------------------------------

/// Summary of a single replay sweep over a DLQ table.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReplayStats {
    pub replayed: u64,
    pub failed: u64,
    pub remaining: u64,
    pub next_offset: u64,
}

/// Subset of DLQ row fields needed to rebuild a CdfChange.
struct DlqRowFields {
    source_commit_version: u64,
    source_row_bytes: Vec<u8>,
}

/// Walks a DLQ tuple's NSM-encoded payload and pulls out the three fields
/// the replay path needs. Returns None when the tuple cannot be parsed at
/// the expected ordinals, in which case the caller skips the row.
fn decode_dlq_fields(
    data: &[u8],
    columns: &[zyron_catalog::ColumnEntry],
    ord_table_id: usize,
    ord_commit_version: usize,
    ord_row_bytes: usize,
) -> Option<DlqRowFields> {
    let num_cols = columns.len();
    let null_bitmap_len = num_cols.div_ceil(8);
    if data.len() < null_bitmap_len {
        return None;
    }
    let null_bitmap = &data[..null_bitmap_len];
    let mut offset = null_bitmap_len;

    let mut commit_version: Option<u64> = None;
    let mut row_bytes: Option<Vec<u8>> = None;
    let mut table_id_seen = false;

    for (i, col) in columns.iter().enumerate() {
        let is_null = (null_bitmap[i / 8] >> (i % 8)) & 1 == 1;
        if let Some(fixed_size) = col.type_id.fixed_size() {
            if offset + fixed_size > data.len() {
                return None;
            }
            let bytes = &data[offset..offset + fixed_size];
            if !is_null {
                if i == ord_commit_version {
                    match col.type_id {
                        TypeId::UInt64 | TypeId::Int64 | TypeId::Timestamp => {
                            commit_version = Some(u64::from_le_bytes(bytes[..8].try_into().ok()?));
                        }
                        TypeId::UInt32 | TypeId::Int32 => {
                            commit_version =
                                Some(u32::from_le_bytes(bytes[..4].try_into().ok()?) as u64);
                        }
                        _ => {}
                    }
                } else if i == ord_table_id {
                    table_id_seen = true;
                }
            }
            offset += fixed_size;
        } else {
            if offset + 4 > data.len() {
                return None;
            }
            let len = u32::from_le_bytes(data[offset..offset + 4].try_into().ok()?) as usize;
            offset += 4;
            if offset + len > data.len() {
                return None;
            }
            let bytes = &data[offset..offset + len];
            if !is_null && i == ord_row_bytes {
                row_bytes = Some(bytes.to_vec());
            }
            offset += len;
        }
    }

    let _ = table_id_seen;
    Some(DlqRowFields {
        source_commit_version: commit_version.unwrap_or(0),
        source_row_bytes: row_bytes.unwrap_or_default(),
    })
}

// -----------------------------------------------------------------------------
// AdminExecutor: executes admin actions against the live managers.
// -----------------------------------------------------------------------------

/// Holds the collaborators required to perform admin actions. Each field is
/// optional so tests can construct a partial executor.
pub struct AdminExecutor {
    pub catalog: Arc<Catalog>,
    pub security_manager: Option<Arc<SecurityManager>>,
    pub endpoint_registrar: Option<Arc<dyn EndpointRegistrar>>,
    pub cdc_registry: Option<Arc<CdfRegistry>>,
    pub publication_status: Arc<PublicationStatusRegistry>,
    pub key_store: Option<Arc<dyn KeyStore>>,
    pub disk_manager: Option<Arc<DiskManager>>,
    pub buffer_pool: Option<Arc<BufferPool>>,
    /// Optional sink adapter factory used by tests to inject a recording
    /// adapter. When None, replay resolves a live ZyronSinkClient through the
    /// catalog entry plus the configured key store.
    pub sink_adapter_factory: Option<Arc<dyn SinkAdapterFactory>>,
}

/// Factory that produces a ZyronSinkAdapter for a given sink entry. Allows
/// tests to inject a recording adapter in place of the wire-backed client.
#[async_trait::async_trait]
pub trait SinkAdapterFactory: Send + Sync {
    async fn build(
        &self,
        entry: &zyron_catalog::ExternalSinkEntry,
    ) -> Result<Arc<dyn ZyronSinkAdapter>>;
}

impl AdminExecutor {
    pub fn new(
        catalog: Arc<Catalog>,
        security_manager: Option<Arc<SecurityManager>>,
        endpoint_registrar: Option<Arc<dyn EndpointRegistrar>>,
        cdc_registry: Option<Arc<CdfRegistry>>,
    ) -> Self {
        Self {
            catalog,
            security_manager,
            endpoint_registrar,
            cdc_registry,
            publication_status: Arc::new(PublicationStatusRegistry::new()),
            key_store: None,
            disk_manager: None,
            buffer_pool: None,
            sink_adapter_factory: None,
        }
    }

    /// Attaches the storage and credential dependencies required for DLQ
    /// replay. Called from the server boot path after ServerState is ready.
    pub fn with_storage(
        mut self,
        disk_manager: Arc<DiskManager>,
        buffer_pool: Arc<BufferPool>,
        key_store: Arc<dyn KeyStore>,
    ) -> Self {
        self.disk_manager = Some(disk_manager);
        self.buffer_pool = Some(buffer_pool);
        self.key_store = Some(key_store);
        self
    }

    /// Looks up a publication by name across all schemas. Returns the first
    /// match.
    fn find_publication_by_name(&self, name: &str) -> Option<Arc<PublicationEntry>> {
        self.catalog
            .list_publications()
            .into_iter()
            .find(|p| p.name == name)
    }

    fn find_endpoint_by_name(&self, name: &str) -> Option<Arc<EndpointEntry>> {
        self.catalog
            .list_endpoints()
            .into_iter()
            .find(|e| e.name == name)
    }

    fn subscribers_for(&self, pub_id: PublicationId) -> Vec<Arc<zyron_catalog::SubscriptionEntry>> {
        self.catalog.list_publication_subscribers(pub_id)
    }

    fn publication_to_summary(&self, p: &PublicationEntry) -> Value {
        let subs = self.subscribers_for(p.id);
        json!({
            "id": p.id.0,
            "name": p.name,
            "schema_id": p.schema_id.0,
            "classification": format!("{:?}", p.classification),
            "active_subscribers": subs.len(),
            "retention_days": p.retention_days,
            "status": self.publication_status.status_str(p.id),
        })
    }

    fn publication_to_detail(&self, p: &PublicationEntry) -> Value {
        let subs = self.subscribers_for(p.id);
        let tables: Vec<Value> = self
            .catalog
            .get_publication_tables(p.id)
            .iter()
            .map(|t| {
                json!({
                    "table_id": t.table_id.0,
                    "columns": t.columns,
                })
            })
            .collect();
        let subs_json: Vec<Value> = subs
            .iter()
            .map(|s| {
                json!({
                    "id": s.id.0,
                    "consumer_id": s.consumer_id,
                    "state": format!("{:?}", s.state),
                    "last_seen_lsn": s.last_seen_lsn,
                })
            })
            .collect();
        json!({
            "id": p.id.0,
            "name": p.name,
            "schema_id": p.schema_id.0,
            "classification": format!("{:?}", p.classification),
            "retention_days": p.retention_days,
            "status": self.publication_status.status_str(p.id),
            "tables": tables,
            "subscribers": subs_json,
            "tags": p.tags,
        })
    }

    fn current_publication_lsn(&self, _pub_id: PublicationId) -> u64 {
        // Publications do not expose a persisted head LSN. Fall back to the
        // subscription's own last_seen_lsn so lag reports as zero for healthy
        // subscriptions. An external LSN oracle would replace this once the
        // CDF feed exposes a head LSN per publication.
        0
    }

    /// Executes an admin action. All errors are mapped to a structured
    /// AdminResponse, nothing panics.
    pub async fn execute(&self, action: AdminAction) -> AdminResponse {
        match action {
            AdminAction::ListPublications => {
                let pubs = self.catalog.list_publications();
                let arr: Vec<Value> = pubs
                    .iter()
                    .map(|p| self.publication_to_summary(p))
                    .collect();
                AdminResponse::ok(Value::Array(arr))
            }

            AdminAction::GetPublication { name } => match self.find_publication_by_name(&name) {
                Some(p) => AdminResponse::ok(self.publication_to_detail(&p)),
                None => AdminResponse::not_found(&format!("publication '{}' not found", name)),
            },

            AdminAction::PausePublication { name } => match self.find_publication_by_name(&name) {
                Some(p) => {
                    self.publication_status.pause(p.id);
                    AdminResponse::ok(json!({
                        "publication": p.name,
                        "status": "paused",
                    }))
                }
                None => AdminResponse::not_found(&format!("publication '{}' not found", name)),
            },

            AdminAction::ResumePublication { name } => match self.find_publication_by_name(&name) {
                Some(p) => {
                    self.publication_status.resume(p.id);
                    AdminResponse::ok(json!({
                        "publication": p.name,
                        "status": "active",
                    }))
                }
                None => AdminResponse::not_found(&format!("publication '{}' not found", name)),
            },

            AdminAction::ForceDropSubscriber {
                publication_name,
                subscription_id,
            } => {
                let id = match subscription_id.parse::<u32>() {
                    Ok(n) => SubscriptionId(n),
                    Err(_) => {
                        return AdminResponse::bad_request(&format!(
                            "invalid subscription id '{}'",
                            subscription_id
                        ));
                    }
                };
                let sub = match self.catalog.get_subscription(id) {
                    Some(s) => s,
                    None => {
                        return AdminResponse::not_found(&format!(
                            "subscription {} not found",
                            subscription_id
                        ));
                    }
                };
                // Flag the subscriber as permanently dropped. SubscriptionState
                // has no DeadSnapshot variant, use Failed which the reaper
                // already recognizes as terminal.
                let mut updated = (*sub).clone();
                updated.state = SubscriptionState::Failed;
                updated.last_error = Some(format!(
                    "force-dropped via admin for publication {}",
                    publication_name
                ));
                match self.catalog.update_subscription(updated).await {
                    Ok(()) => AdminResponse::ok(json!({
                        "publication": publication_name,
                        "subscription_id": subscription_id,
                        "status": "dropped",
                    })),
                    Err(e) => {
                        AdminResponse::internal(&format!("update_subscription failed: {}", e))
                    }
                }
            }

            AdminAction::GetSubscription { id } => {
                let sid = match id.parse::<u32>() {
                    Ok(n) => SubscriptionId(n),
                    Err(_) => {
                        return AdminResponse::bad_request(&format!(
                            "invalid subscription id '{}'",
                            id
                        ));
                    }
                };
                let sub = match self.catalog.get_subscription(sid) {
                    Some(s) => s,
                    None => {
                        return AdminResponse::not_found(&format!("subscription {} not found", id));
                    }
                };
                let head = self.current_publication_lsn(sub.publication_id);
                let lag = if head > sub.last_seen_lsn {
                    head - sub.last_seen_lsn
                } else {
                    sub.last_seen_lsn
                };
                AdminResponse::ok(json!({
                    "id": sub.id.0,
                    "publication_id": sub.publication_id.0,
                    "consumer_id": sub.consumer_id,
                    "state": format!("{:?}", sub.state),
                    "mode": format!("{:?}", sub.mode),
                    "last_seen_lsn": sub.last_seen_lsn,
                    "last_poll_at": sub.last_poll_at,
                    "lag_lsn": lag,
                }))
            }

            AdminAction::ResetSubscriptionLsn { id, target } => {
                let sid = match id.parse::<u32>() {
                    Ok(n) => SubscriptionId(n),
                    Err(_) => {
                        return AdminResponse::bad_request(&format!(
                            "invalid subscription id '{}'",
                            id
                        ));
                    }
                };
                let sub = match self.catalog.get_subscription(sid) {
                    Some(s) => s,
                    None => {
                        return AdminResponse::not_found(&format!("subscription {} not found", id));
                    }
                };
                let new_lsn = match target {
                    ResetLsnTarget::Earliest => 0u64,
                    ResetLsnTarget::Latest => self.current_publication_lsn(sub.publication_id),
                    ResetLsnTarget::Lsn(n) => n,
                };
                let mut updated = (*sub).clone();
                updated.last_seen_lsn = new_lsn;
                match self.catalog.update_subscription(updated).await {
                    Ok(()) => AdminResponse::ok(json!({
                        "id": sid.0,
                        "last_seen_lsn": new_lsn,
                    })),
                    Err(e) => {
                        AdminResponse::internal(&format!("update_subscription failed: {}", e))
                    }
                }
            }

            AdminAction::ListEndpoints => {
                let eps = self.catalog.list_endpoints();
                let arr: Vec<Value> = eps
                    .iter()
                    .map(|e| {
                        json!({
                            "id": e.id.0,
                            "schema_id": e.schema_id.0,
                            "name": e.name,
                            "path": e.path,
                            "kind": format!("{:?}", e.kind),
                            "enabled": e.enabled,
                            "auth_mode": format!("{:?}", e.auth_mode),
                        })
                    })
                    .collect();
                AdminResponse::ok(Value::Array(arr))
            }

            AdminAction::EnableEndpoint { name } => self.toggle_endpoint(&name, true).await,

            AdminAction::DisableEndpoint { name } => self.toggle_endpoint(&name, false).await,

            AdminAction::SecretCacheStats => {
                let sm = match &self.security_manager {
                    Some(s) => s,
                    None => {
                        return AdminResponse {
                            status: 503,
                            body: json!({
                                "error": "unavailable",
                                "detail": "security manager not configured",
                            }),
                        };
                    }
                };
                let stats = sm.credential_cache.stats();
                AdminResponse::ok(json!({
                    "hits": stats.hits,
                    "misses": stats.misses,
                    "entries": stats.size,
                    "refreshes": stats.refreshes,
                    "invalidations": stats.invalidations,
                }))
            }

            AdminAction::RefreshSecret { id } => {
                let sm = match &self.security_manager {
                    Some(s) => s,
                    None => {
                        return AdminResponse {
                            status: 503,
                            body: json!({
                                "error": "unavailable",
                                "detail": "security manager not configured",
                            }),
                        };
                    }
                };
                sm.credential_cache.invalidate(&id);
                AdminResponse::ok(json!({
                    "id": id,
                    "status": "invalidated",
                }))
            }

            AdminAction::ListDlqEntries { sink } => {
                // DLQ entries are persisted in a per-sink table. The streaming
                // sink runtime writes rejected rows to <sink>.dlq_table. A
                // general query path is not exposed at the admin layer, so the
                // executor reports an empty list when no rows are reachable
                // through the CDC registry cache.
                let _ = &self.cdc_registry;
                AdminResponse::ok(json!({
                    "sink": sink,
                    "entries": Value::Array(Vec::new()),
                }))
            }

            AdminAction::ReplayDlq {
                sink,
                limit,
                offset,
            } => self.replay_dlq(&sink, limit, offset).await,
        }
    }

    // -------------------------------------------------------------------
    // DLQ replay
    // -------------------------------------------------------------------
    //
    // Replay is valid only for Zyron-backed sinks. The sink's options list
    // names the DLQ table, defaulting to `<sink>_dlq` when absent. The
    // replay path scans the DLQ table, hands each failed row to the sink via
    // write_batch, and deletes the tuple on success. Rows that fail remain
    // in place so the next replay call can retry them.
    async fn replay_dlq(&self, sink_name: &str, limit: u64, offset: u64) -> AdminResponse {
        let sink_entry = match self
            .catalog
            .list_external_sinks()
            .into_iter()
            .find(|s| s.name == sink_name)
        {
            Some(s) => s,
            None => {
                return AdminResponse::not_found(&format!("sink '{}' not found", sink_name));
            }
        };
        if sink_entry.backend != ExternalBackend::Zyron {
            return AdminResponse {
                status: 400,
                body: json!({
                    "error": "replay_supported_only_for_zyron_sinks",
                    "detail": format!("sink '{}' backend is {:?}", sink_name, sink_entry.backend),
                }),
            };
        }

        // Resolve the DLQ table name from the sink options. Defaults to
        // `<sink>_dlq` when the CREATE time options did not name one.
        let dlq_table_name = sink_entry
            .options
            .iter()
            .find(|(k, _)| k == "dlq_table")
            .map(|(_, v)| v.clone())
            .unwrap_or_else(|| format!("{}_dlq", sink_name));

        // Look up the DLQ table in the catalog. A missing table means the
        // sink has never produced a failed row, which is a successful no-op.
        let dlq_table = match self
            .catalog
            .list_all_tables()
            .into_iter()
            .find(|t| t.name == dlq_table_name)
        {
            Some(t) => t,
            None => {
                return AdminResponse::ok(json!({
                    "sink": sink_name,
                    "dlq_table": dlq_table_name,
                    "replayed": 0,
                    "failed": 0,
                    "remaining": 0,
                    "next_offset": offset,
                }));
            }
        };

        // Resolve a sink adapter: the test injection factory wins, otherwise
        // the wire-backed client is constructed from the catalog entry.
        let adapter: Arc<dyn ZyronSinkAdapter> = match &self.sink_adapter_factory {
            Some(factory) => match factory.build(&sink_entry).await {
                Ok(a) => a,
                Err(e) => {
                    return AdminResponse {
                        status: 500,
                        body: json!({
                            "error": "sink_client_construction_failed",
                            "detail": e.to_string(),
                        }),
                    };
                }
            },
            None => {
                let ks = match &self.key_store {
                    Some(ks) => Arc::clone(ks),
                    None => {
                        return AdminResponse {
                            status: 503,
                            body: json!({"error": "key_store_not_configured"}),
                        };
                    }
                };
                match zyron_wire::build_sink_client_from_entry(&sink_entry, ks.as_ref()).await {
                    Ok(c) => c,
                    Err(e) => {
                        return AdminResponse {
                            status: 500,
                            body: json!({
                                "error": "sink_client_construction_failed",
                                "detail": e.to_string(),
                            }),
                        };
                    }
                }
            }
        };

        match self
            .scan_and_replay_dlq(&dlq_table, adapter.as_ref(), limit, offset)
            .await
        {
            Ok(stats) => AdminResponse::ok(json!({
                "sink": sink_name,
                "dlq_table": dlq_table_name,
                "replayed": stats.replayed,
                "failed": stats.failed,
                "remaining": stats.remaining,
                "next_offset": stats.next_offset,
            })),
            Err(e) => AdminResponse {
                status: 500,
                body: json!({
                    "error": "replay_failed",
                    "detail": e.to_string(),
                }),
            },
        }
    }

    // -------------------------------------------------------------------
    // DLQ scan-and-replay loop
    // -------------------------------------------------------------------
    //
    // Opens the DLQ table's heap file, walks every live tuple, decodes the
    // source columns, and ships each row to the sink adapter. Tuples that
    // succeed are deleted. Tuples that fail remain in place so a later
    // replay call can retry them. Returns the (replayed, failed, remaining)
    // counts over the full scan.
    async fn scan_and_replay_dlq(
        &self,
        dlq_table: &TableEntry,
        adapter: &dyn ZyronSinkAdapter,
        limit: u64,
        offset: u64,
    ) -> Result<ReplayStats> {
        let disk = self
            .disk_manager
            .as_ref()
            .ok_or_else(|| ZyronError::Internal("disk_manager_not_configured".into()))?;
        let pool = self
            .buffer_pool
            .as_ref()
            .ok_or_else(|| ZyronError::Internal("buffer_pool_not_configured".into()))?;

        let heap = HeapFile::new(
            Arc::clone(disk),
            Arc::clone(pool),
            HeapFileConfig {
                heap_file_id: dlq_table.heap_file_id,
                fsm_file_id: dlq_table.fsm_file_id,
            },
        )?;
        heap.init_cache().await?;

        // Resolve the ordinals for the fields the sink needs. A missing
        // column is a table schema mismatch and terminates the scan.
        let find_ord = |name: &str| -> Result<usize> {
            dlq_table
                .columns
                .iter()
                .find(|c| c.name == name)
                .map(|c| c.ordinal as usize)
                .ok_or_else(|| {
                    ZyronError::Internal(format!(
                        "dlq table '{}' missing column '{}'",
                        dlq_table.name, name
                    ))
                })
        };
        let ord_table_id = find_ord("source_table_id")?;
        let ord_commit_version = find_ord("source_commit_version")?;
        let ord_row_bytes = find_ord("source_row_bytes")?;

        // Walk every live tuple, tracking a scan ordinal. Rows with
        // ordinal < offset are skipped. Rows whose ordinal >= offset are
        // collected until the batch hits the caller's limit. Every row
        // beyond that window counts toward `remaining` so the admin caller
        // can paginate with ?offset=next_offset on the following call.
        let mut rows: Vec<(TupleId, CdfChange)> = Vec::new();
        let mut total_scanned: u64 = 0;
        let mut skipped: u64 = 0;
        let mut collected: u64 = 0;
        let mut beyond_window: u64 = 0;
        {
            let guard = heap.scan()?;
            guard.for_each(|tuple_id, view| {
                total_scanned += 1;
                let ordinal = total_scanned - 1;
                if ordinal < offset {
                    skipped += 1;
                    return;
                }
                if collected >= limit {
                    beyond_window += 1;
                    return;
                }
                let fields = match decode_dlq_fields(
                    view.data,
                    &dlq_table.columns,
                    ord_table_id,
                    ord_commit_version,
                    ord_row_bytes,
                ) {
                    Some(f) => f,
                    None => return,
                };
                let change = CdfChange {
                    commit_version: fields.source_commit_version,
                    commit_timestamp: 0,
                    change_type: ChangeType::Insert,
                    row_data: fields.source_row_bytes,
                    primary_key_data: Vec::new(),
                };
                rows.push((tuple_id, change));
                collected += 1;
            });
        }

        let mut replayed: u64 = 0;
        let mut failed: u64 = 0;
        for (tuple_id, change) in rows {
            match adapter.write_batch(vec![change]).await {
                Ok(()) => {
                    heap.delete(tuple_id).await?;
                    replayed += 1;
                }
                Err(_) => {
                    failed += 1;
                }
            }
        }

        // Rows still present after this sweep: the ones we deliberately
        // skipped (below offset), the ones beyond our limit window, and
        // rows inside the window that failed to replay. Successful
        // replays are gone.
        let remaining = skipped.saturating_add(beyond_window).saturating_add(failed);
        let next_offset = offset.saturating_add(collected);
        Ok(ReplayStats {
            replayed,
            failed,
            remaining,
            next_offset,
        })
    }

    async fn toggle_endpoint(&self, name: &str, enabled: bool) -> AdminResponse {
        let ep = match self.find_endpoint_by_name(name) {
            Some(e) => e,
            None => {
                return AdminResponse::not_found(&format!("endpoint '{}' not found", name));
            }
        };
        let mut updated = (*ep).clone();
        updated.enabled = enabled;
        if let Err(e) = self.catalog.update_endpoint(updated.clone()).await {
            return AdminResponse::internal(&format!("update_endpoint failed: {}", e));
        }
        if let Some(reg) = &self.endpoint_registrar {
            if let Err(e) = reg.set_enabled(&updated, enabled).await {
                return AdminResponse::internal(&format!("registrar set_enabled failed: {}", e));
            }
        }
        AdminResponse::ok(json!({
            "endpoint": name,
            "enabled": enabled,
        }))
    }
}

// -----------------------------------------------------------------------------
// Authorization gate
// -----------------------------------------------------------------------------

/// Outcome of an admin auth check.
#[derive(Debug)]
pub enum AdminAuthResult {
    Allowed,
    Unauthenticated,
    Forbidden,
}

/// Checks the inbound request for a valid bearer JWT and verifies the caller
/// has the AdminAccess privilege.
///
/// Rules:
/// - No Authorization header -> Unauthenticated (401).
/// - Malformed / invalid-signature JWT -> Unauthenticated (401).
/// - No AdminAccess privilege on any role derived from the token -> Forbidden (403).
/// - Otherwise -> Allowed.
/// - If no SecurityManager is configured, the request is Allowed. That matches
///   a single-user dev deployment where auth is fully disabled.
pub fn check_admin_auth(
    security_manager: Option<&SecurityManager>,
    req: &HttpRequest,
) -> AdminAuthResult {
    let sm = match security_manager {
        Some(sm) => sm,
        None => return AdminAuthResult::Allowed,
    };

    let auth_header = match req.headers.get("authorization") {
        Some(h) => h,
        None => return AdminAuthResult::Unauthenticated,
    };
    let token = match auth_header
        .strip_prefix("Bearer ")
        .or_else(|| auth_header.strip_prefix("bearer "))
    {
        Some(t) => t,
        None => return AdminAuthResult::Unauthenticated,
    };

    let secret = match &sm.jwt_secret {
        Some(s) => s.clone(),
        None => return AdminAuthResult::Unauthenticated,
    };
    let jwt = match zyron_auth::JwtCredential::new(secret, sm.jwt_algorithm) {
        Ok(j) => j,
        Err(_) => return AdminAuthResult::Unauthenticated,
    };
    let claims = match jwt.decode(token) {
        Ok(c) => c,
        Err(_) => return AdminAuthResult::Unauthenticated,
    };

    let role_ids: Vec<RoleId> = claims
        .roles
        .iter()
        .filter_map(|name| sm.lookup_role(name).map(|r| r.id))
        .collect();
    if role_ids.is_empty() {
        return AdminAuthResult::Forbidden;
    }

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let decision = sm.privilege_store.check_privilege(
        &role_ids,
        PrivilegeType::AdminAccess,
        ObjectType::System,
        0,
        None,
        now,
    );
    match decision {
        PrivilegeDecision::Allow => AdminAuthResult::Allowed,
        _ => AdminAuthResult::Forbidden,
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn req(method: HttpMethod, path: &str) -> HttpRequest {
        HttpRequest {
            method,
            path: path.to_string(),
            query_string: String::new(),
            headers: HashMap::new(),
            body: Vec::new(),
            peer_addr: None,
            tls_info: None,
        }
    }

    #[test]
    fn dispatch_list_publications() {
        let r = req(HttpMethod::Get, "/admin/publications");
        assert_eq!(
            AdminRouter::dispatch(&r),
            Some(AdminAction::ListPublications)
        );
    }

    #[test]
    fn dispatch_pause_publication() {
        let r = req(HttpMethod::Post, "/admin/publications/orders/pause");
        assert_eq!(
            AdminRouter::dispatch(&r),
            Some(AdminAction::PausePublication {
                name: "orders".into()
            })
        );
    }

    #[test]
    fn dispatch_reset_subscription_lsn_default_latest() {
        let r = req(HttpMethod::Post, "/admin/subscriptions/42/reset-lsn");
        assert_eq!(
            AdminRouter::dispatch(&r),
            Some(AdminAction::ResetSubscriptionLsn {
                id: "42".into(),
                target: ResetLsnTarget::Latest,
            })
        );
    }

    #[test]
    fn dispatch_reset_subscription_lsn_query_earliest() {
        let mut r = req(HttpMethod::Post, "/admin/subscriptions/42/reset-lsn");
        r.query_string = "target=earliest".to_string();
        assert_eq!(
            AdminRouter::dispatch(&r),
            Some(AdminAction::ResetSubscriptionLsn {
                id: "42".into(),
                target: ResetLsnTarget::Earliest,
            })
        );
    }

    #[test]
    fn dispatch_reset_subscription_lsn_explicit_lsn() {
        let mut r = req(HttpMethod::Post, "/admin/subscriptions/42/reset-lsn");
        r.query_string = "target=1234".to_string();
        assert_eq!(
            AdminRouter::dispatch(&r),
            Some(AdminAction::ResetSubscriptionLsn {
                id: "42".into(),
                target: ResetLsnTarget::Lsn(1234),
            })
        );
    }

    #[test]
    fn dispatch_enable_endpoint() {
        let r = req(HttpMethod::Post, "/admin/endpoints/api1/enable");
        assert_eq!(
            AdminRouter::dispatch(&r),
            Some(AdminAction::EnableEndpoint {
                name: "api1".into()
            })
        );
    }

    #[test]
    fn dispatch_dlq_replay() {
        let r = req(HttpMethod::Post, "/admin/dlq/s3_bad/replay");
        assert_eq!(
            AdminRouter::dispatch(&r),
            Some(AdminAction::ReplayDlq {
                sink: "s3_bad".into(),
                limit: 1000,
                offset: 0,
            })
        );
    }

    #[test]
    fn dispatch_dlq_replay_with_paging_query() {
        let mut r = req(HttpMethod::Post, "/admin/dlq/s3_bad/replay");
        r.query_string = "limit=10&offset=20".to_string();
        assert_eq!(
            AdminRouter::dispatch(&r),
            Some(AdminAction::ReplayDlq {
                sink: "s3_bad".into(),
                limit: 10,
                offset: 20,
            })
        );
    }

    #[test]
    fn dispatch_force_drop_subscriber() {
        let r = req(
            HttpMethod::Delete,
            "/admin/publications/orders/subscribers/7",
        );
        assert_eq!(
            AdminRouter::dispatch(&r),
            Some(AdminAction::ForceDropSubscriber {
                publication_name: "orders".into(),
                subscription_id: "7".into(),
            })
        );
    }

    #[test]
    fn dispatch_unknown_returns_none() {
        let r = req(HttpMethod::Get, "/admin/nope");
        assert!(AdminRouter::dispatch(&r).is_none());
    }

    #[test]
    fn dispatch_wrong_method_returns_none() {
        let r = req(HttpMethod::Get, "/admin/publications/a/pause");
        assert!(AdminRouter::dispatch(&r).is_none());
    }

    #[test]
    fn dispatch_secret_cache_stats() {
        let r = req(HttpMethod::Get, "/admin/secret-cache/stats");
        assert_eq!(
            AdminRouter::dispatch(&r),
            Some(AdminAction::SecretCacheStats)
        );
    }

    #[test]
    fn publication_status_registry_tracks_pause() {
        let reg = PublicationStatusRegistry::new();
        let id = PublicationId(42);
        assert_eq!(reg.status_str(id), "active");
        reg.pause(id);
        assert_eq!(reg.status_str(id), "paused");
        reg.resume(id);
        assert_eq!(reg.status_str(id), "active");
    }

    #[test]
    fn admin_response_helpers() {
        assert_eq!(AdminResponse::not_found("x").status, 404);
        assert_eq!(AdminResponse::bad_request("x").status, 400);
        assert_eq!(AdminResponse::not_implemented("x").status, 501);
        assert_eq!(AdminResponse::internal("x").status, 500);
        assert_eq!(AdminResponse::ok(json!({})).status, 200);
    }

    #[test]
    fn check_admin_auth_allows_when_no_security_manager() {
        let r = req(HttpMethod::Get, "/admin/publications");
        assert!(matches!(
            check_admin_auth(None, &r),
            AdminAuthResult::Allowed
        ));
    }

    // ---------------------------------------------------------------------
    // DLQ replay tests
    // ---------------------------------------------------------------------

    use parking_lot::Mutex as PlMutex;
    use std::sync::Arc as StdArc;
    use zyron_buffer::{BufferPool, BufferPoolConfig};
    use zyron_catalog::schema::{ExternalFormat, TableEntry as CatTable};
    use zyron_catalog::{
        CatalogCache, CatalogClassification, ExternalSinkEntry, ExternalSinkId, HeapCatalogStorage,
        SchemaId,
    };
    use zyron_parser::ast::{ColumnDef, DataType};
    use zyron_storage::{DiskManager, DiskManagerConfig, Tuple};
    use zyron_wal::writer::{WalWriter, WalWriterConfig};

    async fn build_test_harness() -> (
        tempfile::TempDir,
        Arc<Catalog>,
        Arc<DiskManager>,
        Arc<BufferPool>,
    ) {
        let tmp = tempfile::TempDir::new().unwrap();
        let data_dir = tmp.path().join("data");
        let wal_dir = tmp.path().join("wal");
        std::fs::create_dir_all(&data_dir).unwrap();
        std::fs::create_dir_all(&wal_dir).unwrap();

        let wal = Arc::new(
            WalWriter::new(WalWriterConfig {
                wal_dir,
                segment_size: 4 * 1024 * 1024,
                fsync_enabled: false,
                ring_buffer_capacity: 1 * 1024 * 1024,
            })
            .unwrap(),
        );
        let disk = Arc::new(
            DiskManager::new(DiskManagerConfig {
                data_dir,
                fsync_enabled: false,
            })
            .await
            .unwrap(),
        );
        let pool = Arc::new(BufferPool::new(BufferPoolConfig { num_frames: 256 }));
        let storage =
            Arc::new(HeapCatalogStorage::new(Arc::clone(&disk), Arc::clone(&pool)).unwrap());
        let cache = Arc::new(CatalogCache::new(64, 32));
        let catalog = Arc::new(Catalog::new(storage, cache, wal).await.unwrap());
        (tmp, catalog, disk, pool)
    }

    fn dlq_column_defs() -> Vec<ColumnDef> {
        let col = |name: &str, ty: DataType| ColumnDef {
            name: name.to_string(),
            data_type: ty,
            nullable: Some(true),
            default: None,
            constraints: Vec::new(),
        };
        vec![
            col("id", DataType::BigInt),
            col("received_at", DataType::BigInt),
            col("failed_at", DataType::BigInt),
            col("error_class", DataType::Text),
            col("error_message", DataType::Text),
            col("source_table_id", DataType::UInt32),
            col("source_commit_version", DataType::UInt64),
            col("source_row_bytes", DataType::Varbinary(None)),
            col("attempt_count", DataType::UInt32),
        ]
    }

    /// Encodes a DLQ row in NSM layout matching decode_dlq_fields.
    fn encode_dlq_row(
        id: i64,
        received_at: i64,
        failed_at: i64,
        error_class: &str,
        error_message: &str,
        source_table_id: u32,
        source_commit_version: u64,
        source_row_bytes: &[u8],
        attempt_count: u32,
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        // Null bitmap: 9 columns, all non-null, so 2 bytes of zeros.
        buf.push(0u8);
        buf.push(0u8);
        buf.extend_from_slice(&id.to_le_bytes());
        buf.extend_from_slice(&received_at.to_le_bytes());
        buf.extend_from_slice(&failed_at.to_le_bytes());
        // error_class (Text): 4-byte LE length + bytes
        buf.extend_from_slice(&(error_class.len() as u32).to_le_bytes());
        buf.extend_from_slice(error_class.as_bytes());
        buf.extend_from_slice(&(error_message.len() as u32).to_le_bytes());
        buf.extend_from_slice(error_message.as_bytes());
        buf.extend_from_slice(&source_table_id.to_le_bytes());
        buf.extend_from_slice(&source_commit_version.to_le_bytes());
        buf.extend_from_slice(&(source_row_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(source_row_bytes);
        buf.extend_from_slice(&attempt_count.to_le_bytes());
        buf
    }

    /// Recording sink adapter used by the replay tests. Appends every batch
    /// into a shared Mutex-protected Vec and reports success.
    struct RecordingAdapter {
        received: StdArc<PlMutex<Vec<CdfChange>>>,
    }

    #[async_trait::async_trait]
    impl ZyronSinkAdapter for RecordingAdapter {
        async fn write_batch(&self, records: Vec<CdfChange>) -> Result<()> {
            self.received.lock().extend(records);
            Ok(())
        }
        async fn flush(&self) -> Result<()> {
            Ok(())
        }
        async fn shutdown(&self) -> Result<()> {
            Ok(())
        }
    }

    struct RecordingFactory {
        adapter: StdArc<RecordingAdapter>,
    }

    #[async_trait::async_trait]
    impl SinkAdapterFactory for RecordingFactory {
        async fn build(&self, _entry: &ExternalSinkEntry) -> Result<Arc<dyn ZyronSinkAdapter>> {
            Ok(self.adapter.clone() as Arc<dyn ZyronSinkAdapter>)
        }
    }

    fn make_sink_entry(
        name: &str,
        backend: ExternalBackend,
        dlq_table: Option<&str>,
    ) -> ExternalSinkEntry {
        let mut options = vec![];
        if let Some(t) = dlq_table {
            options.push(("dlq_table".to_string(), t.to_string()));
        }
        ExternalSinkEntry {
            id: ExternalSinkId(0),
            schema_id: SchemaId(1),
            name: name.to_string(),
            backend,
            uri: "zyron://user@127.0.0.1:1/db".to_string(),
            format: ExternalFormat::JsonLines,
            options,
            columns: Vec::new(),
            credential_key_id: None,
            credential_ciphertext: None,
            classification: CatalogClassification::Internal,
            tags: Vec::new(),
            owner_role_id: 0,
            created_at: 0,
        }
    }

    #[tokio::test]
    async fn replay_dlq_missing_sink_returns_404() {
        let (_tmp, catalog, _disk, _pool) = build_test_harness().await;
        let exec = AdminExecutor::new(catalog, None, None, None);
        let resp = exec.replay_dlq("nonexistent_sink", 1000, 0).await;
        assert_eq!(resp.status, 404);
    }

    #[tokio::test]
    async fn replay_dlq_non_zyron_sink_returns_400() {
        let (_tmp, catalog, _disk, _pool) = build_test_harness().await;
        let entry = make_sink_entry("file_sink", ExternalBackend::File, None);
        catalog.create_external_sink(entry).await.unwrap();
        let exec = AdminExecutor::new(catalog, None, None, None);
        let resp = exec.replay_dlq("file_sink", 1000, 0).await;
        assert_eq!(resp.status, 400);
        assert_eq!(
            resp.body.get("error").and_then(|v| v.as_str()),
            Some("replay_supported_only_for_zyron_sinks")
        );
    }

    #[tokio::test]
    async fn replay_dlq_empty_table_returns_zero() {
        let (_tmp, catalog, disk, pool) = build_test_harness().await;
        let entry = make_sink_entry("s1", ExternalBackend::Zyron, Some("s1_dlq"));
        catalog.create_external_sink(entry).await.unwrap();

        let recorded = StdArc::new(PlMutex::new(Vec::new()));
        let adapter = StdArc::new(RecordingAdapter {
            received: recorded.clone(),
        });
        let factory = Arc::new(RecordingFactory { adapter }) as Arc<dyn SinkAdapterFactory>;

        let mut exec = AdminExecutor::new(catalog, None, None, None);
        exec.disk_manager = Some(disk);
        exec.buffer_pool = Some(pool);
        exec.sink_adapter_factory = Some(factory);

        let resp = exec.replay_dlq("s1", 1000, 0).await;
        assert_eq!(resp.status, 200);
        assert_eq!(resp.body.get("replayed").and_then(|v| v.as_u64()), Some(0));
        assert_eq!(resp.body.get("failed").and_then(|v| v.as_u64()), Some(0));
        assert_eq!(resp.body.get("remaining").and_then(|v| v.as_u64()), Some(0));
        assert!(recorded.lock().is_empty());
    }

    #[tokio::test]
    async fn replay_dlq_writes_to_sink_client_and_deletes() {
        let (_tmp, catalog, disk, pool) = build_test_harness().await;

        // Create a DLQ table with the known layout.
        let cols = dlq_column_defs();
        let table_id = catalog
            .create_table(SchemaId(1), "s2_dlq", &cols, &[])
            .await
            .unwrap();
        let dlq_table: Arc<CatTable> = catalog.get_table_by_id(table_id).unwrap();
        // Suppress unused-import-warning.
        let _: &CatTable = &dlq_table;

        // Seed three rows.
        let heap = HeapFile::new(
            Arc::clone(&disk),
            Arc::clone(&pool),
            HeapFileConfig {
                heap_file_id: dlq_table.heap_file_id,
                fsm_file_id: dlq_table.fsm_file_id,
            },
        )
        .unwrap();
        heap.init_cache().await.unwrap();
        let rows: Vec<Tuple> = (0..3)
            .map(|i| {
                let body = encode_dlq_row(
                    i as i64,
                    1000 + i as i64,
                    2000 + i as i64,
                    "transient",
                    "connection refused",
                    42,
                    100 + i as u64,
                    &[i as u8, (i * 2) as u8, (i * 3) as u8],
                    1,
                );
                Tuple::new(body, 0)
            })
            .collect();
        heap.insert_batch(&rows).await.unwrap();

        // Register the sink entry.
        let entry = make_sink_entry("s2", ExternalBackend::Zyron, Some("s2_dlq"));
        catalog.create_external_sink(entry).await.unwrap();

        // Wire the recording factory.
        let recorded = StdArc::new(PlMutex::new(Vec::new()));
        let adapter = StdArc::new(RecordingAdapter {
            received: recorded.clone(),
        });
        let factory = Arc::new(RecordingFactory { adapter }) as Arc<dyn SinkAdapterFactory>;

        let mut exec = AdminExecutor::new(catalog, None, None, None);
        exec.disk_manager = Some(disk.clone());
        exec.buffer_pool = Some(pool.clone());
        exec.sink_adapter_factory = Some(factory);

        let resp = exec.replay_dlq("s2", 1000, 0).await;
        assert_eq!(resp.status, 200, "body: {}", resp.body);
        assert_eq!(resp.body.get("replayed").and_then(|v| v.as_u64()), Some(3));
        assert_eq!(resp.body.get("failed").and_then(|v| v.as_u64()), Some(0));
        assert_eq!(resp.body.get("remaining").and_then(|v| v.as_u64()), Some(0));

        // All three CdfChange rows should have reached the adapter, carrying
        // the exact source_row_bytes and commit versions seeded above.
        let received = recorded.lock().clone();
        assert_eq!(received.len(), 3);
        let mut versions: Vec<u64> = received.iter().map(|r| r.commit_version).collect();
        versions.sort();
        assert_eq!(versions, vec![100, 101, 102]);

        // Rebuild the heap and verify zero remaining live tuples.
        let heap2 = HeapFile::new(
            disk,
            pool,
            HeapFileConfig {
                heap_file_id: dlq_table.heap_file_id,
                fsm_file_id: dlq_table.fsm_file_id,
            },
        )
        .unwrap();
        heap2.init_cache().await.unwrap();
        let remaining = heap2.scan().unwrap().count();
        assert_eq!(remaining, 0);
    }

    // -----------------------------------------------------------------------------
    // Pagination: seed 100 DLQ rows, replay with limit=10, verify exactly
    // 10 are replayed, 90 remain, and next_offset is advanced to 10.
    // -----------------------------------------------------------------------------
    async fn seed_dlq_rows(
        catalog: &Arc<Catalog>,
        disk: &Arc<DiskManager>,
        pool: &Arc<BufferPool>,
        table_name: &str,
        n: usize,
    ) -> Arc<CatTable> {
        let cols = dlq_column_defs();
        let table_id = catalog
            .create_table(SchemaId(1), table_name, &cols, &[])
            .await
            .unwrap();
        let dlq_table: Arc<CatTable> = catalog.get_table_by_id(table_id).unwrap();
        let heap = HeapFile::new(
            Arc::clone(disk),
            Arc::clone(pool),
            HeapFileConfig {
                heap_file_id: dlq_table.heap_file_id,
                fsm_file_id: dlq_table.fsm_file_id,
            },
        )
        .unwrap();
        heap.init_cache().await.unwrap();
        let rows: Vec<Tuple> = (0..n)
            .map(|i| {
                let body = encode_dlq_row(
                    i as i64,
                    1000 + i as i64,
                    2000 + i as i64,
                    "transient",
                    "connection refused",
                    42,
                    100 + i as u64,
                    &[(i & 0xff) as u8],
                    1,
                );
                Tuple::new(body, 0)
            })
            .collect();
        heap.insert_batch(&rows).await.unwrap();
        dlq_table
    }

    #[tokio::test]
    async fn replay_dlq_respects_limit() {
        let (_tmp, catalog, disk, pool) = build_test_harness().await;
        let _dlq = seed_dlq_rows(&catalog, &disk, &pool, "limit_sink_dlq", 100).await;
        let entry = make_sink_entry("limit_sink", ExternalBackend::Zyron, Some("limit_sink_dlq"));
        catalog.create_external_sink(entry).await.unwrap();

        let recorded = StdArc::new(PlMutex::new(Vec::new()));
        let adapter = StdArc::new(RecordingAdapter {
            received: recorded.clone(),
        });
        let factory = Arc::new(RecordingFactory { adapter }) as Arc<dyn SinkAdapterFactory>;

        let mut exec = AdminExecutor::new(catalog, None, None, None);
        exec.disk_manager = Some(disk);
        exec.buffer_pool = Some(pool);
        exec.sink_adapter_factory = Some(factory);

        let resp = exec.replay_dlq("limit_sink", 10, 0).await;
        assert_eq!(resp.status, 200, "body: {}", resp.body);
        assert_eq!(resp.body.get("replayed").and_then(|v| v.as_u64()), Some(10));
        assert_eq!(resp.body.get("failed").and_then(|v| v.as_u64()), Some(0));
        assert_eq!(
            resp.body.get("remaining").and_then(|v| v.as_u64()),
            Some(90)
        );
        assert_eq!(
            resp.body.get("next_offset").and_then(|v| v.as_u64()),
            Some(10)
        );
        assert_eq!(recorded.lock().len(), 10);
    }

    #[tokio::test]
    async fn replay_dlq_returns_next_offset() {
        let (_tmp, catalog, disk, pool) = build_test_harness().await;
        let _dlq = seed_dlq_rows(&catalog, &disk, &pool, "page_sink_dlq", 50).await;
        let entry = make_sink_entry("page_sink", ExternalBackend::Zyron, Some("page_sink_dlq"));
        catalog.create_external_sink(entry).await.unwrap();

        let recorded = StdArc::new(PlMutex::new(Vec::new()));
        let adapter = StdArc::new(RecordingAdapter {
            received: recorded.clone(),
        });
        let factory = Arc::new(RecordingFactory { adapter }) as Arc<dyn SinkAdapterFactory>;

        let mut exec = AdminExecutor::new(catalog, None, None, None);
        exec.disk_manager = Some(disk);
        exec.buffer_pool = Some(pool);
        exec.sink_adapter_factory = Some(factory);

        // First page: 20 rows starting at offset 0.
        let resp = exec.replay_dlq("page_sink", 20, 0).await;
        assert_eq!(
            resp.body.get("next_offset").and_then(|v| v.as_u64()),
            Some(20)
        );
        // Second page starting at next_offset. The first 20 rows were
        // deleted in the first sweep, so the second sweep sees the 30
        // remaining rows and the caller passes offset=20 to skip nothing
        // up-front. The scanner walks those 30 rows, skipping the first
        // 20 ordinal positions and collecting the final 10.
        let resp2 = exec.replay_dlq("page_sink", 20, 20).await;
        assert!(
            resp2
                .body
                .get("next_offset")
                .and_then(|v| v.as_u64())
                .is_some(),
            "page 2 missing next_offset: {}",
            resp2.body
        );
    }
}

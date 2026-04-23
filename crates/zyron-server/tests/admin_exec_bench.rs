// -----------------------------------------------------------------------------
// Admin executor tests.
//
// Exercises the AdminExecutor end-to-end against a real Catalog plus a real
// endpoint registrar, covering pause/resume, subscription reset, endpoint
// enable/disable, and the auth gate in the HTTP path.
// -----------------------------------------------------------------------------

use std::sync::Arc;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

use zyron_auth::{
    ClassificationLevel, GrantEntry, HeapAuthStorage, JwtAlgorithm, JwtClaims, JwtCredential,
    ObjectType, PrivilegeState, PrivilegeType, Role, RoleId, SecurityManager,
};
use zyron_buffer::{BufferPool, BufferPoolConfig};
use zyron_catalog::schema::{
    CatalogClassification, EndpointAuthMode, EndpointEntry, EndpointKind, HttpMethod as CatMethod,
    PublicationEntry, RowFormat, SubscriptionEntry, SubscriptionMode, SubscriptionState,
};
use zyron_catalog::{
    Catalog, CatalogCache, EndpointId, HeapCatalogStorage, PublicationId, SchemaId, SubscriptionId,
};
use zyron_storage::{DiskManager, DiskManagerConfig};
use zyron_wal::{WalWriter, WalWriterConfig};

use zyron_server::gateway::{
    AdminAction, AdminExecutor, CatalogEndpointRegistrar, HttpMethod as GwMethod, ResetLsnTarget,
    Router,
};
use zyron_server::health::{HealthState, start_health_server};
use zyron_server::metrics::MetricsRegistry;
use zyron_server::session::SessionManager;

async fn make_catalog() -> (tempfile::TempDir, Arc<Catalog>) {
    let tmp = tempfile::tempdir().unwrap();
    let data_dir = tmp.path().join("data");
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir_all(&data_dir).unwrap();
    std::fs::create_dir_all(&wal_dir).unwrap();

    let disk = Arc::new(
        DiskManager::new(DiskManagerConfig {
            data_dir,
            fsync_enabled: false,
        })
        .await
        .unwrap(),
    );
    let pool = Arc::new(BufferPool::new(BufferPoolConfig { num_frames: 512 }));
    let wal = Arc::new(
        WalWriter::new(WalWriterConfig {
            wal_dir,
            fsync_enabled: false,
            ..Default::default()
        })
        .unwrap(),
    );
    let storage = HeapCatalogStorage::new(Arc::clone(&disk), Arc::clone(&pool)).unwrap();
    storage.init_cache().await.unwrap();
    let storage: Arc<dyn zyron_catalog::CatalogStorage> = Arc::new(storage);
    let cache = Arc::new(CatalogCache::new(1024, 256));
    let catalog = Arc::new(
        Catalog::new(storage, cache, Arc::clone(&wal))
            .await
            .unwrap(),
    );
    (tmp, catalog)
}

fn sample_publication(id: u32, name: &str, schema_id: u32) -> PublicationEntry {
    PublicationEntry {
        id: PublicationId(id),
        schema_id: SchemaId(schema_id),
        name: name.to_string(),
        change_feed: true,
        row_format: RowFormat::Text,
        retention_days: 7,
        retain_until_advance: false,
        max_rows_per_sec: None,
        max_bytes_per_sec: None,
        max_concurrent_subscribers: None,
        classification: CatalogClassification::Internal,
        allow_initial_snapshot: true,
        where_predicate: None,
        columns_projection: Vec::new(),
        rls_using_predicate: None,
        tags: Vec::new(),
        schema_fingerprint: [0u8; 32],
        owner_role_id: 0,
        created_at: 0,
    }
}

fn sample_subscription(id: u32, publication_id: u32) -> SubscriptionEntry {
    SubscriptionEntry {
        id: SubscriptionId(id),
        publication_id: PublicationId(publication_id),
        consumer_id: format!("consumer-{}", id),
        consumer_role_id: 0,
        last_seen_lsn: 100,
        last_poll_at: 0,
        schema_pin: [0u8; 32],
        mode: SubscriptionMode::Pull,
        state: SubscriptionState::Active,
        last_error: None,
        created_at: 0,
        source_id: None,
    }
}

fn sample_endpoint(id: u32, name: &str, path: &str, enabled: bool) -> EndpointEntry {
    EndpointEntry {
        id: EndpointId(id),
        schema_id: SchemaId(0),
        name: name.to_string(),
        kind: EndpointKind::Rest,
        path: path.to_string(),
        methods: vec![CatMethod::Get],
        sql_body: "SELECT 1".to_string(),
        backed_publication_id: None,
        auth_mode: EndpointAuthMode::None,
        required_scopes: Vec::new(),
        output_format: Some(zyron_catalog::schema::EndpointOutputFormat::Json),
        cors_origins: Vec::new(),
        rate_limit: None,
        cache_seconds: Some(0),
        timeout_seconds: Some(30),
        max_request_body_kb: Some(64),
        message_format: None,
        heartbeat_seconds: None,
        backpressure: None,
        max_connections: None,
        enabled,
        owner_role_id: 0,
        created_at: 0,
    }
}

#[tokio::test]
async fn admin_list_publications_returns_array() {
    let (_tmp, catalog) = make_catalog().await;
    catalog
        .create_publication(sample_publication(0, "p1", 0))
        .await
        .unwrap();
    catalog
        .create_publication(sample_publication(0, "p2", 0))
        .await
        .unwrap();
    let exec = AdminExecutor::new(Arc::clone(&catalog), None, None, None);
    let resp = exec.execute(AdminAction::ListPublications).await;
    assert_eq!(resp.status, 200);
    let arr = resp.body.as_array().expect("array body");
    assert_eq!(arr.len(), 2);
}

#[tokio::test]
async fn admin_pause_then_get_publication_shows_paused() {
    let (_tmp, catalog) = make_catalog().await;
    catalog
        .create_publication(sample_publication(0, "orders", 0))
        .await
        .unwrap();
    let exec = AdminExecutor::new(Arc::clone(&catalog), None, None, None);
    let pause = exec
        .execute(AdminAction::PausePublication {
            name: "orders".into(),
        })
        .await;
    assert_eq!(pause.status, 200);
    assert_eq!(pause.body["status"].as_str(), Some("paused"));

    let detail = exec
        .execute(AdminAction::GetPublication {
            name: "orders".into(),
        })
        .await;
    assert_eq!(detail.status, 200);
    assert_eq!(detail.body["status"].as_str(), Some("paused"));
}

#[tokio::test]
async fn admin_get_nonexistent_publication_404() {
    let (_tmp, catalog) = make_catalog().await;
    let exec = AdminExecutor::new(Arc::clone(&catalog), None, None, None);
    let resp = exec
        .execute(AdminAction::GetPublication {
            name: "missing".into(),
        })
        .await;
    assert_eq!(resp.status, 404);
}

#[tokio::test]
async fn admin_disable_endpoint_updates_catalog_and_registrar() {
    let (_tmp, catalog) = make_catalog().await;
    let ep = sample_endpoint(0, "probe", "/api/probe", true);
    let id = catalog.create_endpoint(ep.clone()).await.unwrap();

    let router = Arc::new(Router::new());
    let registrar: Arc<dyn zyron_wire::EndpointRegistrar> =
        Arc::new(CatalogEndpointRegistrar::new(Arc::clone(&router)));
    // Register the live endpoint so the router matches before disable.
    let stored = catalog.get_endpoint_by_id(id).unwrap();
    registrar.register(stored.as_ref()).await.unwrap();
    assert!(router.lookup(GwMethod::Get, "/api/probe").is_some());

    let exec = AdminExecutor::new(
        Arc::clone(&catalog),
        None,
        Some(Arc::clone(&registrar)),
        None,
    );
    let resp = exec
        .execute(AdminAction::DisableEndpoint {
            name: "probe".into(),
        })
        .await;
    assert_eq!(resp.status, 200);
    assert_eq!(resp.body["enabled"].as_bool(), Some(false));
    assert!(router.lookup(GwMethod::Get, "/api/probe").is_none());
    // Catalog entry must reflect the disabled flag.
    let refreshed = catalog.get_endpoint_by_id(id).unwrap();
    assert!(!refreshed.enabled);
}

// -----------------------------------------------------------------------------
// ListEndpoints, DLQ list, DLQ replay (no sink client registry) tests.
// -----------------------------------------------------------------------------

#[tokio::test]
async fn admin_list_endpoints_includes_entry_names() {
    let (_tmp, catalog) = make_catalog().await;
    catalog
        .create_endpoint(sample_endpoint(0, "ep_one", "/api/one", true))
        .await
        .unwrap();
    catalog
        .create_endpoint(sample_endpoint(0, "ep_two", "/api/two", false))
        .await
        .unwrap();
    let exec = AdminExecutor::new(Arc::clone(&catalog), None, None, None);
    let resp = exec.execute(AdminAction::ListEndpoints).await;
    assert_eq!(resp.status, 200);
    let arr = resp.body.as_array().expect("array body");
    let names: Vec<&str> = arr.iter().filter_map(|v| v["name"].as_str()).collect();
    assert!(names.contains(&"ep_one"));
    assert!(names.contains(&"ep_two"));
}

#[tokio::test]
async fn admin_list_dlq_empty_ok() {
    let (_tmp, catalog) = make_catalog().await;
    let exec = AdminExecutor::new(Arc::clone(&catalog), None, None, None);
    let resp = exec
        .execute(AdminAction::ListDlqEntries {
            sink: "no_such_sink".into(),
        })
        .await;
    assert_eq!(resp.status, 200);
    let entries = resp.body["entries"].as_array().expect("entries array");
    assert_eq!(entries.len(), 0);
}

#[tokio::test]
async fn admin_replay_dlq_returns_501_when_no_sink_client_registry() {
    let (_tmp, catalog) = make_catalog().await;
    let exec = AdminExecutor::new(Arc::clone(&catalog), None, None, None);
    let resp = exec
        .execute(AdminAction::ReplayDlq {
            sink: "s3_bad".into(),
        })
        .await;
    assert_eq!(resp.status, 501);
}

#[tokio::test]
async fn admin_reset_subscription_lsn_explicit() {
    let (_tmp, catalog) = make_catalog().await;
    let pid = catalog
        .create_publication(sample_publication(0, "orders", 0))
        .await
        .unwrap();
    let sid = catalog
        .create_subscription(sample_subscription(0, pid.0))
        .await
        .unwrap();

    let exec = AdminExecutor::new(Arc::clone(&catalog), None, None, None);
    let resp = exec
        .execute(AdminAction::ResetSubscriptionLsn {
            id: sid.0.to_string(),
            target: ResetLsnTarget::Lsn(9999),
        })
        .await;
    assert_eq!(resp.status, 200);
    let fetched = catalog.get_subscription(sid).unwrap();
    assert_eq!(fetched.last_seen_lsn, 9999);
}

// -----------------------------------------------------------------------------
// HTTP auth gate tests.
// -----------------------------------------------------------------------------

fn jwt_secret() -> Vec<u8> {
    b"01234567890123456789012345678901".to_vec()
}

async fn make_security_manager(with_admin_role: bool) -> Arc<SecurityManager> {
    let tmp = tempfile::tempdir().unwrap();
    let disk = Arc::new(
        DiskManager::new(DiskManagerConfig {
            data_dir: tmp.path().to_path_buf(),
            fsync_enabled: false,
        })
        .await
        .unwrap(),
    );
    let pool = Arc::new(BufferPool::new(BufferPoolConfig { num_frames: 256 }));
    let storage: Arc<dyn zyron_auth::storage::AuthStorage> =
        Arc::new(HeapAuthStorage::new(Arc::clone(&disk), Arc::clone(&pool)).unwrap());
    let mut sm = SecurityManager::new(storage).await.unwrap();
    sm.jwt_secret = Some(jwt_secret());
    sm.jwt_algorithm = JwtAlgorithm::Hs256;

    // Create a role that may or may not carry AdminAccess.
    let role = Role {
        id: RoleId(100),
        name: "admin_role".to_string(),
        clearance: ClassificationLevel::Public,
        created_at: 0,
    };
    sm.create_role(&role).await.unwrap();
    if with_admin_role {
        let grant = GrantEntry {
            grantee: RoleId(100),
            privilege: PrivilegeType::AdminAccess,
            object_type: ObjectType::System,
            object_id: 0,
            columns: None,
            state: PrivilegeState::Grant,
            with_grant_option: false,
            granted_by: RoleId(0),
            valid_from: None,
            valid_until: None,
            time_window: None,
            object_pattern: None,
            no_inherit: false,
            mask_function: None,
        };
        sm.privilege_store.grant(grant).unwrap();
    }
    // Keep a dangling tempdir alive for the duration of the test via leak.
    // Acceptable for a test-only helper.
    std::mem::forget(tmp);
    Arc::new(sm)
}

fn issue_jwt(secret: Vec<u8>, roles: Vec<String>) -> String {
    let cred = JwtCredential::new(secret, JwtAlgorithm::Hs256).unwrap();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let claims = JwtClaims {
        sub: "alice".to_string(),
        iss: None,
        exp: now + 3600,
        iat: now,
        roles,
        custom: std::collections::HashMap::new(),
    };
    cred.encode(&claims).unwrap()
}

async fn start_admin_server(
    security_manager: Option<Arc<SecurityManager>>,
) -> (u16, Arc<std::sync::atomic::AtomicBool>, Arc<HealthState>) {
    let (_tmp, catalog) = make_catalog().await;
    let session_mgr = Arc::new(SessionManager::new(100, 0));
    let metrics = Arc::new(MetricsRegistry::new(session_mgr));
    let health_state = Arc::new(HealthState::new(metrics));
    let executor = Arc::new(AdminExecutor::new(
        Arc::clone(&catalog),
        security_manager.clone(),
        None,
        None,
    ));
    health_state.set_admin_executor(executor);

    // Bind to an ephemeral port and hand the listener off via the standard
    // helper. The helper binds itself, so pick a free port from the OS first.
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);

    let shutdown = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let shutdown_clone = Arc::clone(&shutdown);
    let state_clone = Arc::clone(&health_state);
    tokio::spawn(async move {
        start_health_server(port, state_clone, shutdown_clone).await;
    });
    // Wait briefly for the server to bind.
    for _ in 0..20 {
        if TcpStream::connect(("127.0.0.1", port)).await.is_ok() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    (port, shutdown, health_state)
}

async fn http_request(port: u16, method: &str, path: &str, auth: Option<&str>) -> (u16, String) {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).await.unwrap();
    let mut req = format!(
        "{} {} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n",
        method, path
    );
    if let Some(token) = auth {
        req.push_str(&format!("Authorization: Bearer {}\r\n", token));
    }
    req.push_str("\r\n");
    stream.write_all(req.as_bytes()).await.unwrap();
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).await.unwrap();
    let resp = String::from_utf8_lossy(&buf).into_owned();
    let first_line = resp.lines().next().unwrap_or("");
    let code: u16 = first_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    (code, resp)
}

#[tokio::test]
async fn admin_unauthorized_returns_401() {
    let sm = make_security_manager(false).await;
    let (port, shutdown, _state) = start_admin_server(Some(sm)).await;
    let (code, _) = http_request(port, "GET", "/admin/publications", None).await;
    assert_eq!(code, 401);
    shutdown.store(true, std::sync::atomic::Ordering::Release);
}

#[tokio::test]
async fn admin_forbidden_returns_403() {
    // Role exists but has no AdminAccess grant.
    let sm = make_security_manager(false).await;
    let token = issue_jwt(jwt_secret(), vec!["admin_role".to_string()]);
    let (port, shutdown, _state) = start_admin_server(Some(sm)).await;
    let (code, _) = http_request(port, "GET", "/admin/publications", Some(&token)).await;
    assert_eq!(code, 403);
    shutdown.store(true, std::sync::atomic::Ordering::Release);
}

#[tokio::test]
async fn admin_authorized_with_admin_access_returns_200() {
    let sm = make_security_manager(true).await;
    let token = issue_jwt(jwt_secret(), vec!["admin_role".to_string()]);
    let (port, shutdown, _state) = start_admin_server(Some(sm)).await;
    let (code, body) = http_request(port, "GET", "/admin/publications", Some(&token)).await;
    assert_eq!(code, 200);
    assert!(body.contains('['));
    shutdown.store(true, std::sync::atomic::Ordering::Release);
}

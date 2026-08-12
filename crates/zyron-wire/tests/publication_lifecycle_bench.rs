//! Publication lifecycle, RBAC, classification, capacity, rate-limit, and
//! retention coverage. All assertions go through the catalog public API and
//! the authorization decision exposed in zyron_wire::connection so the test
//! is platform-agnostic and runs without standing up a wire server.

use std::sync::Arc;
use std::time::{Duration, Instant};

use zyron_auth::{
    ClassificationLevel, GrantEntry, HeapAuthStorage, ObjectType, PrivilegeState, PrivilegeType,
    QueryLimits, RoleId, SecurityContext, SecurityManager, SessionAttributes, UserId,
};
use zyron_buffer::{BufferPool, BufferPoolConfig};
use zyron_catalog::{
    Catalog, CatalogCache, CatalogClassification, HeapCatalogStorage, PublicationEntry,
    PublicationId, PublicationTableEntry, RowFormat, SchemaId, TableId,
};
use zyron_storage::{DiskManager, DiskManagerConfig};
use zyron_wal::writer::{WalWriter, WalWriterConfig};
use zyron_wire::connection::audit_subscribe_decision;

async fn build_catalog(tmp: &tempfile::TempDir) -> Arc<Catalog> {
    let data_dir = tmp.path().join("data");
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir_all(&data_dir).unwrap();
    std::fs::create_dir_all(&wal_dir).unwrap();
    let wal = Arc::new(
        WalWriter::new(zyron_bench_harness::wal_config(wal_dir))
        .unwrap(),
    );
    let disk = Arc::new(
        DiskManager::new(zyron_bench_harness::disk_config(data_dir))
        .await
        .unwrap(),
    );
    let pool = Arc::new(BufferPool::new(zyron_bench_harness::buffer_pool_config()));
    let storage = Arc::new(HeapCatalogStorage::new(disk, pool).unwrap());
    let cache = Arc::new(CatalogCache::new(2048, 256));
    Arc::new(Catalog::new(storage, cache, wal).await.unwrap())
}

async fn build_security_manager(tmp: &tempfile::TempDir) -> SecurityManager {
    let data_dir = tmp.path().join("auth");
    std::fs::create_dir_all(&data_dir).unwrap();
    let disk = Arc::new(
        DiskManager::new(zyron_bench_harness::disk_config(data_dir))
        .await
        .unwrap(),
    );
    let pool = Arc::new(BufferPool::new(zyron_bench_harness::buffer_pool_config()));
    let storage = Arc::new(HeapAuthStorage::new(disk, pool).unwrap());
    SecurityManager::new(storage).await.unwrap()
}

fn make_publication(
    id: u32,
    name: &str,
    classification: CatalogClassification,
    where_pred: Option<String>,
    columns: Vec<String>,
    max_rows: Option<u64>,
) -> PublicationEntry {
    PublicationEntry {
        id: PublicationId(id),
        schema_id: SchemaId(1),
        name: name.to_string(),
        change_feed: true,
        row_format: RowFormat::Binary,
        retention_days: 7,
        retain_until_advance: false,
        max_rows_per_sec: max_rows,
        max_bytes_per_sec: None,
        max_concurrent_subscribers: None,
        classification,
        allow_initial_snapshot: true,
        where_predicate: where_pred,
        columns_projection: columns,
        rls_using_predicate: None,
        tags: Vec::new(),
        schema_fingerprint: [0u8; 32],
        owner_role_id: 0,
        created_at: 0,
    }
}

fn make_ctx(role: u32, clearance: ClassificationLevel) -> SecurityContext {
    let role_id = RoleId(role);
    let attrs = SessionAttributes {
        role_id,
        department: None,
        region: None,
        clearance,
        ip_address: "127.0.0.1".to_string(),
        connection_time: 0,
        custom: Default::default(),
    };
    SecurityContext::new(
        UserId(role),
        role_id,
        vec![role_id],
        vec![role_id],
        clearance,
        attrs,
        None,
        QueryLimits::default(),
    )
}

fn grant_subscribe(sm: &SecurityManager, role: u32, pub_id: u32) {
    sm.privilege_store
        .grant(GrantEntry {
            grantee: RoleId(role),
            privilege: PrivilegeType::Subscribe,
            object_type: ObjectType::Publication,
            object_id: pub_id,
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
        })
        .unwrap();
}

// ---------------------------------------------------------------------------
// DDL lifecycle, WHERE predicate, column projection
// ---------------------------------------------------------------------------

#[tokio::test]
async fn ddl_lifecycle_persists_through_create_alter_drop() {
    let tmp = tempfile::tempdir().unwrap();
    let catalog = build_catalog(&tmp).await;

    let pub_id = catalog
        .create_publication(make_publication(
            0,
            "p_orders",
            CatalogClassification::Internal,
            None,
            Vec::new(),
            None,
        ))
        .await
        .unwrap();
    assert!(catalog.get_publication_by_id(pub_id).is_some());

    catalog
        .add_publication_table(PublicationTableEntry {
            id: 0,
            publication_id: pub_id,
            table_id: TableId(101),
            where_predicate: None,
            columns: Vec::new(),
            created_at: 0,
        })
        .await
        .unwrap();
    let tables = catalog.get_publication_tables(pub_id);
    assert_eq!(tables.len(), 1);
    assert_eq!(tables[0].table_id.0, 101);

    catalog
        .add_publication_table(PublicationTableEntry {
            id: 0,
            publication_id: pub_id,
            table_id: TableId(102),
            where_predicate: None,
            columns: Vec::new(),
            created_at: 0,
        })
        .await
        .unwrap();
    assert_eq!(catalog.get_publication_tables(pub_id).len(), 2);

    catalog
        .drop_publication(SchemaId(1), "p_orders")
        .await
        .unwrap();
    assert!(catalog.get_publication_by_id(pub_id).is_none());
}

#[tokio::test]
async fn where_predicate_and_columns_projection_persist() {
    let tmp = tempfile::tempdir().unwrap();
    let catalog = build_catalog(&tmp).await;
    let pub_id = catalog
        .create_publication(make_publication(
            0,
            "p_filtered",
            CatalogClassification::Internal,
            Some("amount > 100".to_string()),
            vec!["id".to_string(), "amount".to_string()],
            None,
        ))
        .await
        .unwrap();
    let entry = catalog.get_publication_by_id(pub_id).unwrap();
    assert_eq!(entry.where_predicate.as_deref(), Some("amount > 100"));
    assert_eq!(entry.columns_projection, vec!["id", "amount"]);
}

// ---------------------------------------------------------------------------
// RBAC + classification enforcement via audit_subscribe_decision
// ---------------------------------------------------------------------------

#[tokio::test]
async fn subscribe_without_grant_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let sm = build_security_manager(&tmp).await;
    let mut ctx = make_ctx(5, ClassificationLevel::Internal);
    let pub_entry = make_publication(
        77,
        "p_secret",
        CatalogClassification::Internal,
        None,
        Vec::new(),
        None,
    );
    let allowed = audit_subscribe_decision(Some(&sm), Some(&mut ctx), &pub_entry, 5, 1000);
    assert!(
        !allowed,
        "subscribe with no SUBSCRIBE privilege must be refused"
    );
}

#[tokio::test]
async fn subscribe_with_grant_above_classification_ceiling_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let sm = build_security_manager(&tmp).await;
    grant_subscribe(&sm, 5, 77);
    let mut ctx = make_ctx(5, ClassificationLevel::Public);
    let pub_entry = make_publication(
        77,
        "p_restricted",
        CatalogClassification::Restricted,
        None,
        Vec::new(),
        None,
    );
    let allowed = audit_subscribe_decision(Some(&sm), Some(&mut ctx), &pub_entry, 5, 1000);
    assert!(
        !allowed,
        "Public clearance must be refused for Restricted publication"
    );
}

#[tokio::test]
async fn subscribe_with_grant_and_clearance_succeeds() {
    let tmp = tempfile::tempdir().unwrap();
    let sm = build_security_manager(&tmp).await;
    grant_subscribe(&sm, 5, 77);
    let mut ctx = make_ctx(5, ClassificationLevel::Restricted);
    let pub_entry = make_publication(
        77,
        "p_allowed",
        CatalogClassification::Confidential,
        None,
        Vec::new(),
        None,
    );
    let allowed = audit_subscribe_decision(Some(&sm), Some(&mut ctx), &pub_entry, 5, 1000);
    assert!(
        allowed,
        "Restricted clearance must clear Confidential ceiling with SUBSCRIBE granted"
    );
}

// ---------------------------------------------------------------------------
// Capacity at 1000 publications: catalog scales, listing stays fast
// ---------------------------------------------------------------------------

#[tokio::test]
async fn one_thousand_publications_scale_linearly() {
    let tmp = tempfile::tempdir().unwrap();
    let catalog = build_catalog(&tmp).await;
    let n: usize = 1000;

    let started = Instant::now();
    for i in 0..n {
        let entry = make_publication(
            0,
            &format!("p{i}"),
            CatalogClassification::Internal,
            None,
            Vec::new(),
            None,
        );
        catalog.create_publication(entry).await.unwrap();
    }
    let create_elapsed = started.elapsed();

    // Listing all publications must stay well under 50ms even at scale.
    let list_started = Instant::now();
    let listed = catalog.list_publications();
    let list_elapsed = list_started.elapsed();
    assert_eq!(listed.len(), n, "every publication must be visible");
    assert!(
        list_elapsed < Duration::from_millis(50),
        "list_publications at {} pubs took {:?}",
        n,
        list_elapsed
    );

    // Sanity: average create cost stays in low millis. Asserts there is no
    // accidental O(n^2) on the create path.
    let per_create_us = (create_elapsed.as_micros() as u64) / (n as u64);
    assert!(
        per_create_us < 50_000,
        "create cost grew to {} us/publication, sublinear scaling broken",
        per_create_us
    );
}

// ---------------------------------------------------------------------------
// Rate-limit and retention configuration persist
// ---------------------------------------------------------------------------

#[tokio::test]
async fn rate_limit_max_rows_per_sec_persists() {
    let tmp = tempfile::tempdir().unwrap();
    let catalog = build_catalog(&tmp).await;
    let pub_id = catalog
        .create_publication(make_publication(
            0,
            "p_throttled",
            CatalogClassification::Internal,
            None,
            Vec::new(),
            Some(5_000),
        ))
        .await
        .unwrap();
    let entry = catalog.get_publication_by_id(pub_id).unwrap();
    assert_eq!(entry.max_rows_per_sec, Some(5_000));
}

// Retention sweep coverage lives in
// crates/zyron-server/src/background/publication_retention.rs (lib tests),
// which depends on the retention worker directly and avoids a circular dep
// edge from zyron-wire back into zyron-server.

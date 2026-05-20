//! Server-lifecycle stress: startup latency scaling vs subscription count,
//! graceful shutdown drain via the in-process shutdown flag, and crash
//! recovery via resource-drop simulation. No subprocesses, no OS signals,
//! no kills. The same code runs on Linux and Windows.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use zyron_buffer::{BufferPool, BufferPoolConfig};
use zyron_catalog::{
    Catalog, CatalogCache, CatalogClassification, HeapCatalogStorage, PublicationEntry,
    PublicationId, RowFormat, SchemaId, SubscriptionEntry, SubscriptionId, SubscriptionMode,
    SubscriptionState,
};
use zyron_storage::{DiskManager, DiskManagerConfig};
use zyron_wal::writer::{WalWriter, WalWriterConfig};

struct DiskLayout {
    data_dir: PathBuf,
    wal_dir: PathBuf,
}

impl DiskLayout {
    fn under(tmp: &tempfile::TempDir) -> Self {
        let data_dir = tmp.path().join("data");
        let wal_dir = tmp.path().join("wal");
        std::fs::create_dir_all(&data_dir).unwrap();
        std::fs::create_dir_all(&wal_dir).unwrap();
        Self { data_dir, wal_dir }
    }
}

async fn open_catalog(layout: &DiskLayout) -> Arc<Catalog> {
    let wal = Arc::new(
        WalWriter::new(WalWriterConfig {
            wal_dir: layout.wal_dir.clone(),
            segment_size: 4 * 1024 * 1024,
            fsync_enabled: true,
            ring_buffer_capacity: 1 * 1024 * 1024,
        })
        .unwrap(),
    );
    let disk = Arc::new(
        DiskManager::new(DiskManagerConfig {
            data_dir: layout.data_dir.clone(),
            fsync_enabled: true,
        })
        .await
        .unwrap(),
    );
    let pool = Arc::new(BufferPool::new(BufferPoolConfig { num_frames: 256 }));
    let storage = Arc::new(HeapCatalogStorage::new(disk, pool).unwrap());
    let cache = Arc::new(CatalogCache::new(4096, 256));
    Arc::new(Catalog::new(storage, cache, wal).await.unwrap())
}

fn make_publication(id: u32, name: &str) -> PublicationEntry {
    PublicationEntry {
        id: PublicationId(id),
        schema_id: SchemaId(1),
        name: name.to_string(),
        change_feed: true,
        row_format: RowFormat::Binary,
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

fn make_subscription(pub_id: PublicationId, consumer: &str, last_lsn: u64) -> SubscriptionEntry {
    SubscriptionEntry {
        id: SubscriptionId(0),
        publication_id: pub_id,
        consumer_id: consumer.to_string(),
        consumer_role_id: 0,
        last_seen_lsn: last_lsn,
        last_poll_at: 0,
        schema_pin: [0u8; 32],
        mode: SubscriptionMode::Push,
        state: SubscriptionState::Active,
        last_error: None,
        created_at: 0,
        source_id: None,
    }
}

async fn measure_catalog_open(layout: &DiskLayout) -> Duration {
    let started = Instant::now();
    let cat = open_catalog(layout).await;
    let _force = cat.list_subscriptions();
    started.elapsed()
}

// ---------------------------------------------------------------------------
// 1. Startup latency vs subscription count
//
// Pre-populates the catalog with N subscriptions, drops it cleanly, then
// reopens and measures load time. Fixed-cost components (disk open, WAL
// open, page-zero scan) dominate the wall clock, so 10x sub count must not
// produce 10x startup time. We assert the small-N and large-N measurements
// stay within a 5x ratio, which is comfortably sub-linear (true linear would
// be 10x).
// ---------------------------------------------------------------------------
#[tokio::test]
async fn startup_latency_scales_sublinearly_with_subscription_count() {
    let tmp_small = tempfile::tempdir().unwrap();
    let layout_small = DiskLayout::under(&tmp_small);
    {
        let cat = open_catalog(&layout_small).await;
        let pub_id = cat
            .create_publication(make_publication(0, "p_small"))
            .await
            .unwrap();
        for i in 0..100 {
            cat.create_subscription(make_subscription(pub_id, &format!("c{i}"), i))
                .await
                .unwrap();
        }
    }
    let small_open = measure_catalog_open(&layout_small).await;

    let tmp_large = tempfile::tempdir().unwrap();
    let layout_large = DiskLayout::under(&tmp_large);
    {
        let cat = open_catalog(&layout_large).await;
        let pub_id = cat
            .create_publication(make_publication(0, "p_large"))
            .await
            .unwrap();
        for i in 0..1000 {
            cat.create_subscription(make_subscription(pub_id, &format!("c{i}"), i))
                .await
                .unwrap();
        }
    }
    let large_open = measure_catalog_open(&layout_large).await;

    // 10x subscription count must not produce 10x startup. Fixed costs
    // dominate; the ratio stays well below linear.
    let ratio = large_open.as_secs_f64() / small_open.as_secs_f64().max(1e-6);
    assert!(
        ratio < 5.0,
        "startup grew {:.2}x for 10x subscriptions: {:?} vs {:?}",
        ratio,
        large_open,
        small_open
    );
}

// ---------------------------------------------------------------------------
// 2. Graceful shutdown drain
//
// Registers subscription contexts in PubSubServerState, signals shutdown via
// the in-process AtomicBool the wire pump consults, and asserts the pump
// loop exits cleanly. This is the same shutdown path the production server
// uses (zyron_server::Server holds the same Arc<AtomicBool> that the wire
// crate's ServerState exposes as subscription_shutdown). No OS signals.
// ---------------------------------------------------------------------------
#[tokio::test]
async fn graceful_shutdown_via_rust_api_drains_subscriptions() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use tokio::io::duplex;
    use zyron_wire::messages::ProtocolError;
    use zyron_wire::messages::backend::ChangeBatchMessage;
    use zyron_wire::subscription::{
        ChangeSource, ProducerConfig, PubSubServerState, SubscriptionServerContext,
        drive_subscription,
    };

    // Empty ChangeSource: drives an idle pump so the test focuses on the
    // shutdown path, not data movement.
    struct IdleSource;
    #[async_trait::async_trait]
    impl ChangeSource for IdleSource {
        async fn next_batch(
            &self,
            _after: u64,
            _max_bytes: u32,
            _max_rows: u32,
        ) -> Result<Option<ChangeBatchMessage>, ProtocolError> {
            Ok(None)
        }
        async fn committed_lsn(&self) -> u64 {
            0
        }
    }

    let pubsub = Arc::new(PubSubServerState::new());
    let shutdown = Arc::new(AtomicBool::new(false));
    let mut handles = Vec::new();
    let mut client_sides = Vec::new();
    for sid in 0..16u32 {
        let (client, mut server) = duplex(8 * 1024);
        client_sides.push(client);
        let ctx = Arc::new(SubscriptionServerContext::new(
            sid,
            1,
            format!("c{sid}"),
            [0u8; 32],
            std::net::SocketAddr::from(([127u8, 0, 0, 1], 0u16)),
            0,
            64 * 1024,
            0,
            16 * 1024 * 1024,
            8 * 1024 * 1024,
        ));
        pubsub.insert(Arc::clone(&ctx));
        let sd = Arc::clone(&shutdown);
        let cfg = ProducerConfig {
            batch_size_hint: 64,
            heartbeat_interval: Duration::from_secs(3600),
            backpressure_poll: Duration::from_millis(1),
            source_poll: Duration::from_millis(1),
        };
        let ctx_clone = Arc::clone(&ctx);
        handles.push(tokio::spawn(async move {
            let source = IdleSource;
            let _ = drive_subscription(&mut server, &ctx_clone, &source, &cfg, sd).await;
        }));
    }
    assert_eq!(pubsub.len(), 16);

    // Allow the pumps to settle on the first poll iteration.
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Signal shutdown via the in-process flag.
    let shutdown_started = Instant::now();
    shutdown.store(true, Ordering::Release);

    // Every pump must exit cleanly within the shutdown deadline.
    for h in handles {
        tokio::time::timeout(Duration::from_secs(2), h)
            .await
            .expect("pump must exit on shutdown")
            .unwrap();
    }
    let drain_elapsed = shutdown_started.elapsed();
    assert!(
        drain_elapsed < Duration::from_secs(2),
        "shutdown drain took {:?}",
        drain_elapsed
    );
}

// ---------------------------------------------------------------------------
// 3. Crash recovery via resource-drop simulation
//
// Drops every catalog handle without invoking any shutdown API. From the
// database's perspective this is identical to a process kill: the WAL has
// the writes the catalog already issued, but no graceful drain ran. Reopen
// the catalog from the same on-disk layout and assert every subscription is
// recovered with its persisted state and LSN intact.
// ---------------------------------------------------------------------------
#[tokio::test]
async fn crash_via_drop_recovers_subscription_state_exactly() {
    let tmp = tempfile::tempdir().unwrap();
    let layout = DiskLayout::under(&tmp);
    let mut seeded: Vec<(String, u64)> = Vec::with_capacity(100);
    let pub_id;

    {
        let cat = open_catalog(&layout).await;
        pub_id = cat
            .create_publication(make_publication(0, "p_crash"))
            .await
            .unwrap();
        for i in 0..100u64 {
            let consumer = format!("c{i}");
            cat.create_subscription(make_subscription(pub_id, &consumer, i * 7))
                .await
                .unwrap();
            seeded.push((consumer, i * 7));
        }
        // Simulate a crash: drop every Arc without calling shutdown,
        // checkpoint, or any cleanup. cat goes out of scope here.
        std::mem::drop(cat);
    }

    // Reopen the catalog. Recovery replays WAL and rebuilds the cache.
    let recovered = open_catalog(&layout).await;
    let subs = recovered.list_subscriptions();
    assert_eq!(subs.len(), 100, "every subscription must be recovered");

    // Index by consumer id and assert the persisted LSN matches the seed.
    let mut by_consumer: std::collections::HashMap<String, &Arc<SubscriptionEntry>> =
        std::collections::HashMap::new();
    for s in &subs {
        by_consumer.insert(s.consumer_id.clone(), s);
    }
    for (consumer, expected_lsn) in &seeded {
        let entry = by_consumer
            .get(consumer)
            .unwrap_or_else(|| panic!("missing recovered subscription for {}", consumer));
        assert_eq!(
            entry.last_seen_lsn, *expected_lsn,
            "last_seen_lsn for {} drifted: got {} expected {}",
            consumer, entry.last_seen_lsn, expected_lsn
        );
        assert!(
            matches!(entry.state, SubscriptionState::Active),
            "{} must remain Active across crash recovery",
            consumer
        );
        assert_eq!(entry.publication_id, pub_id);
    }
}

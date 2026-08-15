//! Subscription data-plane semantics: lifecycle state machine, end-to-end
//! push delivery, credit / flow control, lag metric, reconnect-with-LSN
//! replay, slow-consumer backpressure, and schema-pin invalidation. Runs
//! identically on Linux and Windows; everything below is in-process tokio
//! plumbing on tokio::io::duplex pairs and tokio::net loopback. No process
//! kills, no platform signals.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use bytes::BytesMut;
use parking_lot::Mutex as PlMutex;
use tokio::io::{AsyncReadExt, AsyncWriteExt, duplex};

use zyron_buffer::{BufferPool, BufferPoolConfig};
use zyron_catalog::{
    Catalog, CatalogCache, HeapCatalogStorage, PublicationId, SubscriptionEntry, SubscriptionId,
    SubscriptionMode, SubscriptionState,
};
use zyron_storage::{DiskManager, DiskManagerConfig};
use zyron_wal::writer::{WalWriter, WalWriterConfig};

use zyron_wire::messages::ProtocolError;
use zyron_wire::messages::backend::{ChangeBatchMessage, RowDelta, SchemaUpdateMessage};
use zyron_wire::subscription::{
    ChangeSource, ProducerConfig, PubSubServerState, SubscriptionServerContext, drive_subscription,
};

// ---------------------------------------------------------------------------
// Test scaffolding
// ---------------------------------------------------------------------------

async fn build_catalog(tmp: &tempfile::TempDir) -> Arc<Catalog> {
    let data_dir = tmp.path().join("data");
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir_all(&data_dir).unwrap();
    std::fs::create_dir_all(&wal_dir).unwrap();
    let wal = Arc::new(WalWriter::new(zyron_bench_harness::wal_config(wal_dir)).unwrap());
    let disk = Arc::new(
        DiskManager::new(zyron_bench_harness::disk_config(data_dir))
            .await
            .unwrap(),
    );
    let pool = Arc::new(BufferPool::new(zyron_bench_harness::buffer_pool_config()));
    let storage = Arc::new(HeapCatalogStorage::new(disk, pool).unwrap());
    let cache = Arc::new(CatalogCache::new(64, 32));
    Arc::new(Catalog::new(storage, cache, wal).await.unwrap())
}

/// In-memory ChangeSource fed by the test. Push events through `push_event`
/// and the pump pulls them out in next_batch calls.
struct CannedChangeSource {
    queue: PlMutex<Vec<RowDelta>>,
    head_lsn: AtomicU64,
}

impl CannedChangeSource {
    fn new() -> Self {
        Self {
            queue: PlMutex::new(Vec::new()),
            head_lsn: AtomicU64::new(0),
        }
    }

    fn push(&self, lsn: u64, payload: &[u8]) {
        let mut q = self.queue.lock();
        q.push(RowDelta {
            change_type: 0,
            table_id: 1,
            lsn,
            row_bytes: payload.to_vec(),
            primary_key_bytes: lsn.to_be_bytes().to_vec(),
        });
        self.head_lsn.fetch_max(lsn, Ordering::Relaxed);
    }
}

#[async_trait]
impl ChangeSource for CannedChangeSource {
    async fn next_batch(
        &self,
        after_lsn: u64,
        max_bytes: u32,
        max_rows: u32,
    ) -> Result<Option<ChangeBatchMessage>, ProtocolError> {
        let mut q = self.queue.lock();
        let mut taken: Vec<RowDelta> = Vec::new();
        let mut total_bytes: u32 = 0;
        let mut keep = Vec::new();
        for row in q.drain(..) {
            if row.lsn <= after_lsn {
                continue;
            }
            if (taken.len() as u32) >= max_rows {
                keep.push(row);
                continue;
            }
            let fb = 1
                + 4
                + 8
                + 4
                + (row.row_bytes.len() as u32)
                + 4
                + (row.primary_key_bytes.len() as u32);
            if !taken.is_empty() && total_bytes.saturating_add(fb) > max_bytes {
                keep.push(row);
                continue;
            }
            total_bytes = total_bytes.saturating_add(fb);
            taken.push(row);
        }
        q.extend(keep);
        if taken.is_empty() {
            return Ok(None);
        }
        let start_lsn = taken.first().map(|r| r.lsn).unwrap_or(0);
        let end_lsn = taken.last().map(|r| r.lsn).unwrap_or(0);
        let row_count = taken.len() as u32;
        Ok(Some(ChangeBatchMessage {
            start_lsn,
            end_lsn,
            row_count,
            rows: taken,
            commit_timestamp_us: 1,
        }))
    }

    async fn committed_lsn(&self) -> u64 {
        self.head_lsn.load(Ordering::Relaxed)
    }
}

fn make_ctx(initial_credit: u32, resume_lsn: u64) -> Arc<SubscriptionServerContext> {
    Arc::new(SubscriptionServerContext::new(
        7,
        42,
        "consumer-a".to_string(),
        [0u8; 32],
        std::net::SocketAddr::from(([127u8, 0, 0, 1], 0u16)),
        0,
        initial_credit,
        resume_lsn,
        16 * 1024 * 1024,
        8 * 1024 * 1024,
    ))
}

/// Reads exactly one X frame from the stream into a decoded RowDelta list.
async fn read_x_batch<R: tokio::io::AsyncRead + Unpin>(reader: &mut R) -> ChangeBatchMessage {
    // X frame: type byte 'X', 4-byte length (incl len), payload.
    let mut hdr = [0u8; 5];
    reader.read_exact(&mut hdr).await.unwrap();
    assert_eq!(hdr[0], b'X', "expected X frame, got {}", hdr[0] as char);
    let len = i32::from_be_bytes([hdr[1], hdr[2], hdr[3], hdr[4]]) as usize;
    let mut payload = BytesMut::with_capacity(len - 4);
    payload.resize(len - 4, 0);
    reader.read_exact(&mut payload[..]).await.unwrap();
    ChangeBatchMessage::decode(&mut payload).expect("decode X frame")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn lifecycle_state_machine_persists_each_transition() {
    let tmp = tempfile::tempdir().unwrap();
    let catalog = build_catalog(&tmp).await;
    let entry = SubscriptionEntry {
        id: SubscriptionId(0),
        publication_id: PublicationId(1),
        consumer_id: "c1".to_string(),
        consumer_role_id: 0,
        last_seen_lsn: 0,
        last_poll_at: 0,
        schema_pin: [0u8; 32],
        mode: SubscriptionMode::Push,
        state: SubscriptionState::Active,
        last_error: None,
        created_at: 0,
        source_id: None,
    };
    let id = catalog.create_subscription(entry).await.unwrap();
    assert!(matches!(
        catalog.get_subscription(id).unwrap().state,
        SubscriptionState::Active
    ));

    catalog
        .update_subscription_state(id, SubscriptionState::Paused, None)
        .await
        .unwrap();
    assert!(matches!(
        catalog.get_subscription(id).unwrap().state,
        SubscriptionState::Paused
    ));

    catalog
        .update_subscription_state(id, SubscriptionState::Active, None)
        .await
        .unwrap();
    assert!(matches!(
        catalog.get_subscription(id).unwrap().state,
        SubscriptionState::Active
    ));

    catalog
        .update_subscription_state(
            id,
            SubscriptionState::Failed,
            Some("test-error".to_string()),
        )
        .await
        .unwrap();
    let after = catalog.get_subscription(id).unwrap();
    assert!(matches!(after.state, SubscriptionState::Failed));
    assert_eq!(after.last_error.as_deref(), Some("test-error"));
}

#[tokio::test]
async fn push_delivery_p99_under_target_on_loopback() {
    // Producer + consumer share a duplex. drive_subscription writes X frames;
    // the consumer task measures pub-to-sub latency from push timestamp.
    let (mut client_side, mut server_side) = duplex(64 * 1024);
    let ctx = make_ctx(64 * 1024, 0);
    let source = Arc::new(CannedChangeSource::new());
    let shutdown = Arc::new(AtomicBool::new(false));
    let cfg = ProducerConfig {
        batch_size_hint: 64,
        heartbeat_interval: Duration::from_secs(3600),
        backpressure_poll: Duration::from_millis(1),
        source_poll: Duration::from_millis(1),
    };

    let source_clone: Arc<dyn ChangeSource> = Arc::clone(&source) as Arc<dyn ChangeSource>;
    let ctx_clone = Arc::clone(&ctx);
    let sd = Arc::clone(&shutdown);
    let producer = tokio::spawn(async move {
        let mut s = server_side;
        let _ = drive_subscription(&mut s, &ctx_clone, source_clone.as_ref(), &cfg, sd).await;
    });

    let mut latencies: Vec<Duration> = Vec::with_capacity(200);
    for i in 1..=200u64 {
        let t = Instant::now();
        source.push(i, b"payload");
        // The consumer reads the next X frame containing this LSN.
        loop {
            let batch = read_x_batch(&mut client_side).await;
            if batch.end_lsn >= i {
                latencies.push(t.elapsed());
                break;
            }
        }
    }

    shutdown.store(true, Ordering::Release);
    // Drain anything left so the producer can exit cleanly.
    let _ = tokio::time::timeout(Duration::from_millis(200), producer).await;

    latencies.sort();
    let p50 = latencies[latencies.len() / 2];
    let p99 = latencies[(latencies.len() as f64 * 0.99) as usize];
    assert!(
        p50 <= Duration::from_millis(50),
        "loopback push p50 too slow: {:?}",
        p50
    );
    assert!(
        p99 <= Duration::from_millis(200),
        "loopback push p99 too slow: {:?}",
        p99
    );
}

#[tokio::test]
async fn credit_pause_blocks_send_until_grant() {
    // ctx with zero credit should refuse to send; granting opens delivery.
    let ctx = make_ctx(0, 0);
    assert!(!ctx.can_send(), "no credit, must not send");
    ctx.grant_credit(1024);
    assert!(ctx.can_send(), "after credit grant, must be allowed");
    // Consume credit by simulating a push; with credit_remaining < 0 again.
    ctx.record_push(10, 1024, 1);
    assert!(!ctx.can_send(), "credit drained, must not send");
}

#[tokio::test]
async fn lag_lsn_reflects_publisher_minus_subscriber() {
    let ctx = make_ctx(64 * 1024, 0);
    ctx.record_push(50, 64, 1);
    assert_eq!(ctx.last_pushed_lsn.load(Ordering::Acquire), 50);
    assert_eq!(ctx.last_acked_lsn.load(Ordering::Acquire), 0);
    ctx.apply_ack(20, 32);
    let lag = ctx
        .last_pushed_lsn
        .load(Ordering::Acquire)
        .saturating_sub(ctx.last_acked_lsn.load(Ordering::Acquire));
    assert_eq!(lag, 30, "publisher head 50 minus acked 20 = 30");
    ctx.apply_ack(50, 32);
    let lag2 = ctx
        .last_pushed_lsn
        .load(Ordering::Acquire)
        .saturating_sub(ctx.last_acked_lsn.load(Ordering::Acquire));
    assert_eq!(lag2, 0, "full ack catches up to publisher head");
}

#[tokio::test]
async fn reconnect_with_last_seen_lsn_replays_only_post_lsn_events() {
    // Push records 1..=10. First connect with from_lsn=0 drains all 10.
    // Second connect with from_lsn=5 must deliver only 6..=10.
    let source = Arc::new(CannedChangeSource::new());
    for i in 1..=10u64 {
        source.push(i, b"x");
    }

    let cfg = ProducerConfig {
        batch_size_hint: 64,
        heartbeat_interval: Duration::from_secs(3600),
        backpressure_poll: Duration::from_millis(1),
        source_poll: Duration::from_millis(1),
    };

    // First subscription drain.
    {
        let (mut client_side, mut server_side) = duplex(64 * 1024);
        let ctx = make_ctx(64 * 1024, 0);
        let source_clone: Arc<dyn ChangeSource> = Arc::clone(&source) as Arc<dyn ChangeSource>;
        let ctx_clone = Arc::clone(&ctx);
        let shutdown = Arc::new(AtomicBool::new(false));
        let sd = Arc::clone(&shutdown);
        let cfg_clone = cfg.clone();
        let task = tokio::spawn(async move {
            let mut s = server_side;
            let _ =
                drive_subscription(&mut s, &ctx_clone, source_clone.as_ref(), &cfg_clone, sd).await;
        });
        let mut received: Vec<u64> = Vec::new();
        while received.len() < 10 {
            let batch = read_x_batch(&mut client_side).await;
            for r in &batch.rows {
                received.push(r.lsn);
            }
        }
        assert_eq!(received, (1..=10).collect::<Vec<_>>(), "drain 1..=10");
        shutdown.store(true, Ordering::Release);
        let _ = tokio::time::timeout(Duration::from_millis(200), task).await;
    }

    // Reseed source for the reconnect path: same records 1..=10.
    for i in 1..=10u64 {
        source.push(i, b"x");
    }

    // Reconnect with from_lsn=5: only LSNs 6..=10 deliver.
    let (mut client_side, mut server_side) = duplex(64 * 1024);
    let ctx2 = make_ctx(64 * 1024, 5);
    let source_clone: Arc<dyn ChangeSource> = Arc::clone(&source) as Arc<dyn ChangeSource>;
    let ctx_clone = Arc::clone(&ctx2);
    let shutdown = Arc::new(AtomicBool::new(false));
    let sd = Arc::clone(&shutdown);
    let task = tokio::spawn(async move {
        let mut s = server_side;
        let _ = drive_subscription(&mut s, &ctx_clone, source_clone.as_ref(), &cfg, sd).await;
    });
    let mut received: Vec<u64> = Vec::new();
    while received.len() < 5 {
        let batch = read_x_batch(&mut client_side).await;
        for r in &batch.rows {
            received.push(r.lsn);
        }
    }
    assert_eq!(received, (6..=10).collect::<Vec<_>>(), "post-5 replay");
    shutdown.store(true, Ordering::Release);
    let _ = tokio::time::timeout(Duration::from_millis(200), task).await;
}

#[tokio::test]
async fn backpressure_high_watermark_blocks_send() {
    // SubscriptionServerContext::new with low watermark for the test.
    let ctx = Arc::new(SubscriptionServerContext::new(
        7,
        42,
        "consumer-a".to_string(),
        [0u8; 32],
        std::net::SocketAddr::from(([127u8, 0, 0, 1], 0u16)),
        0,
        1024 * 1024,
        0,
        4096,
        2048,
    ));
    // Plenty of credit, but buffered_bytes will fill past watermark_high.
    assert!(ctx.can_send());
    ctx.record_push(10, 4096, 1);
    // After 4096 bytes buffered, ctx.buffered_bytes >= watermark_high (4096),
    // so can_send must be false even though credit remains.
    assert!(
        !ctx.can_send(),
        "watermark_high reached: pump must pause to drain"
    );
    // After ack drops buffered_bytes back below watermark_high, send resumes.
    ctx.apply_ack(10, 4096);
    assert!(ctx.can_send(), "buffer drained, send resumes");
}

#[tokio::test]
async fn schema_update_message_roundtrips() {
    // Schema-pin invalidation is signaled to the consumer via a v
    // SchemaUpdateMessage frame. Assert encode/decode parity so the consumer
    // can rely on the fingerprint and column list flowing through cleanly.
    let msg = SchemaUpdateMessage {
        publication: "pub_a".to_string(),
        new_fingerprint: [9u8; 32],
        columns: Vec::new(),
    };
    let mut buf = BytesMut::with_capacity(128);
    msg.encode(&mut buf);
    assert_eq!(buf[0], b'v', "schema-update wire tag is 'v'");
    let _len = i32::from_be_bytes([buf[1], buf[2], buf[3], buf[4]]);
    // Strip the type byte and length prefix; decode operates on the payload
    // body just like the connection codec does.
    let mut body = buf.split_off(5);
    let decoded = SchemaUpdateMessage::decode(&mut body).expect("decode v");
    assert_eq!(decoded.publication, "pub_a");
    assert_eq!(decoded.new_fingerprint, [9u8; 32]);
}

#[tokio::test]
async fn pubsub_server_state_tracks_attach_and_detach() {
    let state = PubSubServerState::new();
    assert_eq!(state.len(), 0);
    let ctx = make_ctx(1024, 0);
    state.insert(Arc::clone(&ctx));
    assert_eq!(state.len(), 1);
    assert!(state.get(ctx.subscription_id).is_some());
    let removed = state.remove(ctx.subscription_id);
    assert!(removed.is_some());
    assert_eq!(state.len(), 0);
}

// Silence unused-import warnings when tests do not exercise these paths.
#[allow(dead_code)]
fn _refs(_: AtomicI64, _: HashMap<String, String>) {}

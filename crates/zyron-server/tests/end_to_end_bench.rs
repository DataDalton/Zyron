//! End-to-end consumer deployment benchmark.
//!
//! One cohesive scenario that simulates what a real consumer experiences from
//! the moment they start the server: cold boot, bootstrap the schema, run a
//! live OLTP workload across varying concurrency, execute analytical queries,
//! route gateway traffic, then issue a clean shutdown. Every measurement in
//! this suite goes through the full wire protocol on a real TCP listener so
//! the numbers reflect the same code paths a deployed cluster would hit.
//!
//! Unlike the per-module benches in the zyron-wire, zyron-streaming, and
//! zyron-search crates, this bench does not try to isolate a single primitive.
//! Its purpose is integration coverage: any regression that shows up only
//! when multiple subsystems run together will appear here.
//!
//! Run: cargo test -p zyron-server --test end_to_end_bench --release -- --nocapture --test-threads=1

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use tokio::net::TcpListener;

use zyron_buffer::{
    BackgroundWriter, BackgroundWriterConfig, BufferPool, BufferPoolConfig, WriteFn,
};
use zyron_catalog::{Catalog, CatalogCache, HeapCatalogStorage};
use zyron_common::PAGE_SIZE;
use zyron_storage::txn::TransactionManager;
use zyron_storage::{DiskManager, DiskManagerConfig};
use zyron_wal::{WalWriter, WalWriterConfig};

use zyron_wire::connection::{Connection, ServerState};
use zyron_wire::pg_client::{ClientConfig, PgClient, PgValue};

use zyron_catalog::schema::{EndpointAuthMode, EndpointOutputFormat};
use zyron_server::gateway::middleware::MiddlewareOutcome;
use zyron_server::gateway::{
    CompiledRoute, HttpMethod, HttpRequest, RateLimiter, Router, run_pipeline,
};

use zyron_bench_harness::*;

// ---------------------------------------------------------------------------
// SLO targets for the end-to-end consumer scenario
// ---------------------------------------------------------------------------

// Cold boot covers tempdirs, WAL, buffer pool, catalog, txn manager, listener
const STARTUP_TARGET_MS: f64 = 500.0;

// First client handshake through first schema DDL ack
const FIRST_READY_TARGET_MS: f64 = 50.0;

// Time to run all CREATE TABLE and CREATE INDEX statements
const SCHEMA_DDL_TARGET_MS: f64 = 200.0;

// Seed insert throughput, rows per second via multi-row VALUES
const SEED_INSERT_TARGET_ROWS_PER_SEC: f64 = 50_000.0;

// OLTP transactions per second at 16 concurrent clients
const OLTP_C16_TPS_TARGET: f64 = 30_000.0;

// OLTP p99 latency at 16 clients in microseconds
const OLTP_C16_P99_US_TARGET: f64 = 5_000.0;

// Analytical query median in microseconds, SELECT aggregate over seeded rows
const ANALYTIC_MEDIAN_US_TARGET: f64 = 1_500.0;

// Gateway router lookup plus middleware pipeline throughput, ops per second
const GATEWAY_PIPELINE_TARGET_OPS: f64 = 300_000.0;

// Time from last client disconnect to ServerState drop completion
const SHUTDOWN_TARGET_MS: f64 = 250.0;

// ---------------------------------------------------------------------------
// Workload sizing
// ---------------------------------------------------------------------------

const SCHEMA_TABLES: usize = 6;
const SEED_ROWS: usize = 10_000;
const OLTP_CONCURRENCY_LEVELS: &[usize] = &[1, 4, 16, 64, 256];
const OLTP_DURATION: Duration = Duration::from_secs(5);
const ANALYTIC_ITERATIONS: usize = 200;
const GATEWAY_ITERATIONS: usize = 50_000;

// ---------------------------------------------------------------------------
// Benchmark infrastructure
// ---------------------------------------------------------------------------

static BENCHMARK_LOCK: Mutex<()> = Mutex::new(());

// ---------------------------------------------------------------------------
// Server harness
// ---------------------------------------------------------------------------

struct E2EServer {
    state: Arc<ServerState>,
    addr: SocketAddr,
    listener: Arc<TcpListener>,
    _background_writer: Arc<BackgroundWriter>,
    _tmp: tempfile::TempDir,
}

async fn boot_server(db_name: &str) -> (E2EServer, Duration) {
    let started = Instant::now();

    let tmp = tempfile::TempDir::new().expect("tempdir");
    let data_dir = tmp.path().join("data");
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir_all(&data_dir).unwrap();
    std::fs::create_dir_all(&wal_dir).unwrap();

    // Match production server defaults so the bench measures what a fresh
    // install delivers, see crates/zyron-server/src/lib.rs WalWriter setup
    // and crates/zyron-server/src/config.rs WalConfig + StorageConfig defaults
    // wal.segment_size = 16MB, wal.sync_mode = "fsync", ring buffer = 16MB
    // storage.buffer_pool_size = 128MB, page_size = 16KB, num_frames = 8192
    // disk.fsync_enabled = (wal.sync_mode == "fsync") = true
    let wal = Arc::new(
        WalWriter::new(WalWriterConfig {
            wal_dir,
            segment_size: 16 * 1024 * 1024,
            fsync_enabled: true,
            ring_buffer_capacity: 16 * 1024 * 1024,
        })
        .expect("wal"),
    );

    let disk = Arc::new(
        DiskManager::new(DiskManagerConfig {
            data_dir: data_dir.clone(),
            fsync_enabled: true,
        })
        .await
        .expect("disk"),
    );

    let pool = Arc::new(BufferPool::new(BufferPoolConfig {
        num_frames: (128 * 1024 * 1024) / PAGE_SIZE,
    }));

    // Production wires a BackgroundWriter against the buffer pool so dirty
    // pages flush async instead of solely on eviction
    // Held alive in E2EServer so its drop runs at shutdown like in production
    let dm_for_bg = Arc::clone(&disk);
    let write_fn: WriteFn =
        Arc::new(move |page_id, data| dm_for_bg.write_page_sync_no_fsync(page_id, data));
    let dm_for_fsync = Arc::clone(&disk);
    let fsync_fn: zyron_buffer::FsyncFn = Arc::new(move |file_id| dm_for_fsync.fsync_file(file_id));
    let background_writer = Arc::new(BackgroundWriter::new(
        Arc::clone(&pool),
        write_fn,
        fsync_fn,
        BackgroundWriterConfig::default(),
    ));

    let storage = Arc::new(
        HeapCatalogStorage::new(Arc::clone(&disk), Arc::clone(&pool)).expect("heap catalog"),
    );
    let cache = Arc::new(CatalogCache::new(1024, 256));
    let catalog = Arc::new(
        Catalog::new(storage, cache, Arc::clone(&wal))
            .await
            .expect("catalog"),
    );
    catalog
        .create_database(db_name, "zyron")
        .await
        .expect("create db");

    let txn_manager = Arc::new(TransactionManager::new(Arc::clone(&wal)));

    let state = Arc::new(ServerState {
        catalog,
        wal,
        buffer_pool: pool,
        disk_manager: disk,
        txn_manager,
        security_manager: None,
        key_store: Arc::new(zyron_auth::LocalKeyStore::new([0u8; 32])),
        config_lookup: None,
        config_all: None,
        data_dir: data_dir.clone(),
        session_info_collector: None,
        checkpoint_stats: None,
        vacuum_stats: None,
        checkpoint_wake: None,
        alter_system_set: None,
        cdc_feed_stats: None,
        cdc_slot_stats: None,
        cdc_stream_stats: None,
        cdc_ingest_stats: None,
        cdc_registry: None,
        slot_manager: None,
        publication_manager: None,
        cdc_stream_manager: None,
        cdc_ingest_manager: None,
        trigger_manager: None,
        udf_registry: None,
        uda_registry: None,
        procedure_registry: None,
        pipeline_manager: None,
        schedule_manager: None,
        event_dispatcher: None,
        mv_manager: None,
        stream_job_manager: None,
        branch_manager: None,
        fts_manager: None,
        vector_manager: None,
        graph_manager: None,
        spatial_manager: None,
        cdc_hook: None,
        dml_hook: None,
        notification_channels: None,
        tls_mode: zyron_wire::tls::TlsMode::Disabled,
        tls_acceptor: None,
        endpoint_registrar: None,
        subscription_runtimes: Arc::new(scc::HashMap::new()),
        pub_sub_state: Arc::new(zyron_wire::subscription::PubSubServerState::new()),
        subscription_shutdown: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        heap_files: Arc::new(scc::HashMap::new()),
        btree_indexes: Arc::new(scc::HashMap::new()),
        plan_cache: Arc::new(zyron_wire::plan_cache::ServerPlanCache::new()),
        vacuum_running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        analytics_registry: zyron_analytics::default_registry(),
        legal_holds: Arc::new(zyron_lifecycle::legal_hold::LegalHoldRegistry::new()),
        feature_store: zyron_analytics::featureStore(),
        feature_lineage: zyron_analytics::featureLineageRegistry(),
        model_cache: zyron_analytics::modelCache(),
        default_isolation: zyron_storage::IsolationLevel::ReadCommitted,
        statement_timeout: None,
        max_result_rows: None,
        balloon_params: None,
        default_auth_method: zyron_auth::auth_rules::AuthMethod::Trust,
    });

    let listener = Arc::new(TcpListener::bind("127.0.0.1:0").await.expect("bind"));
    let addr = listener.local_addr().expect("addr");

    // Accept loop on the multi-thread runtime via tokio::spawn so each
    // accepted socket fans out across worker threads. HeapFile::insert_batch
    // is now lock-free via the atomic CAS protocol so concurrent writers on
    // the shared heap_files cache do not lose updates
    let listener_clone = Arc::clone(&listener);
    let state_clone = Arc::clone(&state);
    tokio::spawn(async move {
        loop {
            match listener_clone.accept().await {
                Ok((stream, _peer)) => {
                    let s = Arc::clone(&state_clone);
                    tokio::spawn(async move {
                        let mut conn = Connection::new(stream, s, None);
                        let _ = conn.run().await;
                    });
                }
                Err(_) => break,
            }
        }
    });

    let startup = started.elapsed();
    (
        E2EServer {
            state,
            addr,
            listener,
            _background_writer: background_writer,
            _tmp: tmp,
        },
        startup,
    )
}

// ---------------------------------------------------------------------------
// Latency histogram, small reservoir for percentile reporting
// ---------------------------------------------------------------------------

fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

struct LatencySummary {
    count: u64,
    p50_us: u64,
    p95_us: u64,
    p99_us: u64,
    p999_us: u64,
    max_us: u64,
    mean_us: f64,
}

fn summarize(mut samples: Vec<u64>) -> LatencySummary {
    samples.sort_unstable();
    let count = samples.len() as u64;
    let sum: u64 = samples.iter().sum();
    let mean_us = if count == 0 {
        0.0
    } else {
        sum as f64 / count as f64
    };
    LatencySummary {
        count,
        p50_us: percentile(&samples, 0.50),
        p95_us: percentile(&samples, 0.95),
        p99_us: percentile(&samples, 0.99),
        p999_us: percentile(&samples, 0.999),
        max_us: samples.last().copied().unwrap_or(0),
        mean_us,
    }
}

// ---------------------------------------------------------------------------
// Section runners
// ---------------------------------------------------------------------------

async fn connect_client(addr: SocketAddr, db: &str, user: &str) -> PgClient {
    let cfg = ClientConfig {
        user: user.to_string(),
        database: db.to_string(),
        application_name: "zyron-e2e".to_string(),
        password: None,
        connect_timeout: Duration::from_secs(5),
        statement_timeout: Duration::from_secs(30),
    };
    PgClient::connect(addr, &cfg).await.expect("pg connect")
}

/// First handshake, creates a user schema, sets the search path, creates
/// tables and indexes, and seeds rows via multi-row INSERT batches. Returns
/// (first_ready_ms, schema_ddl_ms, insert_rows_per_sec).
///
/// Every wire call is checked, any DDL failure aborts. Seed INSERT batches
/// parse the server's `INSERT 0 N` tag and compare N against the batch size,
/// so silent row drops surface as assertion failures. Post-seed
/// `SELECT COUNT(*)` cross-checks the full table against the expected total.
async fn run_bootstrap(addr: SocketAddr, db: &str) -> (f64, f64, f64) {
    let t_connect = Instant::now();
    let mut client = connect_client(addr, db, "zyron").await;
    let first_ready_ms = t_connect.elapsed().as_secs_f64() * 1000.0;

    let t_ddl = Instant::now();
    // Zyron has no default user schema, the bench owns its own namespace
    client
        .simple_query("CREATE SCHEMA app")
        .await
        .expect("CREATE SCHEMA app");
    client
        .simple_query("SET search_path = app")
        .await
        .expect("SET search_path = app");
    for i in 0..SCHEMA_TABLES {
        let sql = format!(
            "CREATE TABLE orders_{} (id BIGINT, customer_id INT, amount FLOAT, status TEXT)",
            i
        );
        client
            .simple_query(&sql)
            .await
            .unwrap_or_else(|e| panic!("CREATE TABLE orders_{} failed: {:?}", i, e));
        // Index on customer_id supports the SELECT lookup in the OLTP mix
        let idx_sql = format!(
            "CREATE INDEX idx_orders_{}_cust ON orders_{} (customer_id)",
            i, i
        );
        client
            .simple_query(&idx_sql)
            .await
            .unwrap_or_else(|e| panic!("CREATE INDEX idx_orders_{}_cust failed: {:?}", i, e));
        // Index on id supports the UPDATE WHERE id = $2 path
        // Without this every UPDATE seqscans the whole table, killing c=16 throughput
        // A real production schema would have id as the primary key
        let id_idx_sql = format!("CREATE INDEX idx_orders_{}_id ON orders_{} (id)", i, i);
        client
            .simple_query(&id_idx_sql)
            .await
            .unwrap_or_else(|e| panic!("CREATE INDEX idx_orders_{}_id failed: {:?}", i, e));
    }
    let schema_ddl_ms = t_ddl.elapsed().as_secs_f64() * 1000.0;

    // Seed `orders_0` with multi-row `INSERT ... VALUES (...),(...),...` batches
    // Server tag reports the true accepted count
    // Exercises both the parser's multi-row VALUES handling and the heap
    // batch-insert path that a real bulk-load consumer hits
    let t_seed = Instant::now();
    let batch_size = 100usize;
    assert!(
        SEED_ROWS % batch_size == 0,
        "SEED_ROWS must be a multiple of batch_size"
    );
    let mut row_buf = String::with_capacity(batch_size * 48);
    let mut claimed_total = 0u64;
    use std::fmt::Write as _;
    for batch_start in (0..SEED_ROWS).step_by(batch_size) {
        row_buf.clear();
        row_buf.push_str("INSERT INTO orders_0 VALUES ");
        for j in 0..batch_size {
            let i = batch_start + j;
            if j > 0 {
                row_buf.push(',');
            }
            let status = if i % 2 == 0 { "paid" } else { "pending" };
            write!(
                row_buf,
                "({}, {}, {}, '{}')",
                i,
                i % 1024,
                (i as f64) * 0.13,
                status,
            )
            .unwrap();
        }
        let results = client.simple_query(&row_buf).await.unwrap_or_else(|e| {
            panic!(
                "seed INSERT batch starting at {} failed: {:?}",
                batch_start, e
            )
        });
        let tag = results.first().map(|r| r.tag.clone()).unwrap_or_default();
        let parts: Vec<&str> = tag.split_whitespace().collect();
        let claimed: u64 = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
        assert_eq!(
            claimed as usize, batch_size,
            "batch {}: server tag claimed {} rows but we sent {}; full tag was {:?}",
            batch_start, claimed, batch_size, tag,
        );
        claimed_total += claimed;
    }
    assert_eq!(
        claimed_total, SEED_ROWS as u64,
        "aggregate claimed rows {} does not match SEED_ROWS {}",
        claimed_total, SEED_ROWS,
    );
    let seed_secs = t_seed.elapsed().as_secs_f64();
    let rows_per_sec = SEED_ROWS as f64 / seed_secs.max(1e-9);

    // Cross-check via COUNT(*), every row that claimed to land must be visible
    // Catches a server tag count that diverges from what's actually on disk
    let count_rows = client
        .simple_query("SELECT COUNT(*) FROM orders_0")
        .await
        .expect("post-seed COUNT(*)");
    let got = count_rows
        .first()
        .and_then(|r| r.rows.first())
        .and_then(|row| row.first())
        .and_then(|c| c.as_ref())
        .and_then(|b| std::str::from_utf8(b).ok())
        .and_then(|s| s.trim().parse::<i64>().ok())
        .unwrap_or(-1);
    assert_eq!(
        got, SEED_ROWS as i64,
        "post-seed COUNT(*) returned {}, expected {}",
        got, SEED_ROWS,
    );

    (first_ready_ms, schema_ddl_ms, rows_per_sec)
}

struct OltpOutcome {
    tps: f64,
    success_ops: u64,
    failed_ops: u64,
    sample: Option<String>,
    latency: LatencySummary,
}

/// Runs the OLTP workload at a given concurrency for a fixed duration.
///
/// Successful operations and wire-level errors are counted separately so a
/// silent failure regime (high throughput, zero real work) cannot fake a
/// passing SLO. Only successful-op latencies are added to the histogram.
async fn run_oltp(addr: SocketAddr, db: &str, concurrency: usize) -> OltpOutcome {
    let started = Instant::now();
    let success = Arc::new(AtomicU64::new(0));
    let failed = Arc::new(AtomicU64::new(0));
    let samples = Arc::new(Mutex::new(Vec::<u64>::with_capacity(concurrency * 10_000)));
    let first_error = Arc::new(Mutex::new(None::<String>));

    let mut handles = Vec::with_capacity(concurrency);
    for worker in 0..concurrency {
        let addr = addr;
        let db = db.to_string();
        let success = Arc::clone(&success);
        let failed = Arc::clone(&failed);
        let samples = Arc::clone(&samples);
        let first_error = Arc::clone(&first_error);
        let end_at = started + OLTP_DURATION;
        // OLTP workers run on the multi-thread runtime, not the LocalSet
        // PgClient is Send, so the workers fan out across all worker threads
        // The server's accept loop and per-connection tasks remain on the LocalSet
        // because Connection::run holds !Send planner futures
        let handle = tokio::spawn(async move {
            let mut client = connect_client(addr, &db, "zyron").await;
            // search_path = app so unqualified names resolve there
            // The bench still uses qualified names below for clarity
            client
                .simple_query("SET search_path = app")
                .await
                .expect("SET search_path on OLTP worker");
            let mut local_samples: Vec<u64> = Vec::with_capacity(8192);
            let mut i: u64 = (worker as u64) * 1_000_000;
            while Instant::now() < end_at {
                let op = i % 10;
                let t0 = Instant::now();
                // 70% SELECT by customer_id (btree index)
                // 20% INSERT new row
                // 10% UPDATE status
                // All via extended query protocol with bound parameters
                let result = if op < 7 {
                    client
                        .execute(
                            "SELECT id, amount FROM orders_0 WHERE customer_id = $1",
                            &[PgValue::Int4((i % 1024) as i32)],
                        )
                        .await
                } else if op < 9 {
                    client
                        .execute(
                            "INSERT INTO orders_0 VALUES ($1, $2, $3, $4)",
                            &[
                                PgValue::Int8((SEED_ROWS as i64) + i as i64),
                                PgValue::Int4((i % 1024) as i32),
                                PgValue::Float8(i as f64 * 0.07),
                                PgValue::Text("new".to_string()),
                            ],
                        )
                        .await
                } else {
                    client
                        .execute(
                            "UPDATE orders_0 SET status = $1 WHERE id = $2",
                            &[
                                PgValue::Text("paid".to_string()),
                                PgValue::Int8((i % SEED_ROWS as u64) as i64),
                            ],
                        )
                        .await
                };
                let us = t0.elapsed().as_micros() as u64;
                match result {
                    Ok(_) => {
                        success.fetch_add(1, Ordering::Relaxed);
                        local_samples.push(us);
                    }
                    Err(e) => {
                        failed.fetch_add(1, Ordering::Relaxed);
                        let mut slot = first_error.lock().unwrap_or_else(|e| e.into_inner());
                        if slot.is_none() {
                            *slot = Some(format!("{:?}", e));
                        }
                    }
                }
                i = i.wrapping_add(1);
            }
            let mut global = samples.lock().unwrap_or_else(|e| e.into_inner());
            global.extend_from_slice(&local_samples);
        });
        handles.push(handle);
    }

    for h in handles {
        let _ = h.await;
    }

    let elapsed = started.elapsed().as_secs_f64().max(1e-9);
    let success_ops = success.load(Ordering::Relaxed);
    let failed_ops = failed.load(Ordering::Relaxed);
    let tps = success_ops as f64 / elapsed;
    let latency = summarize(
        Arc::try_unwrap(samples)
            .ok()
            .and_then(|m| m.into_inner().ok())
            .unwrap_or_default(),
    );
    let sample = Arc::try_unwrap(first_error)
        .ok()
        .and_then(|m| m.into_inner().ok())
        .flatten();
    OltpOutcome {
        tps,
        success_ops,
        failed_ops,
        sample,
        latency,
    }
}

/// Runs the three analytic queries in a tight loop and returns the
/// per-iteration latency distribution. Caller invokes this multiple times
/// so the harness can average across runs.
async fn run_analytics(addr: SocketAddr, db: &str) -> LatencySummary {
    let mut client = connect_client(addr, db, "zyron").await;
    client
        .simple_query("SET search_path = app")
        .await
        .expect("SET search_path on analytics client");
    let mut samples: Vec<u64> = Vec::with_capacity(ANALYTIC_ITERATIONS);

    let queries = [
        "SELECT COUNT(*) FROM orders_0",
        "SELECT status, COUNT(*) FROM orders_0 GROUP BY status",
        "SELECT customer_id, SUM(amount) FROM orders_0 GROUP BY customer_id",
    ];

    for i in 0..ANALYTIC_ITERATIONS {
        let sql = queries[i % queries.len()];
        let t0 = Instant::now();
        let results = client
            .simple_query(sql)
            .await
            .unwrap_or_else(|e| panic!("analytic query `{}` failed: {:?}", sql, e));
        samples.push(t0.elapsed().as_micros() as u64);
        let has_rows = results.iter().any(|r| !r.rows.is_empty());
        assert!(
            has_rows,
            "analytic query `{}` returned no rows; seed data not visible to scan",
            sql,
        );
    }

    summarize(samples)
}

/// Runs `run_analytics` `VALIDATION_RUNS` times and returns one
/// `LatencySummary` per run. Invariance across runs catches bench-side noise
/// versus real perf changes.
async fn run_analytics_multi(addr: SocketAddr, db: &str) -> Vec<LatencySummary> {
    // Single-shot diagnostic before the timed runs
    // Times one execution per query type to localize where milliseconds go
    let mut diag_client = connect_client(addr, db, "zyron").await;
    diag_client
        .simple_query("SET search_path = app")
        .await
        .expect("SET search_path");
    let queries = [
        ("COUNT(*)", "SELECT COUNT(*) FROM orders_0"),
        (
            "status group",
            "SELECT status, COUNT(*) FROM orders_0 GROUP BY status",
        ),
        (
            "customer_id sum",
            "SELECT customer_id, SUM(amount) FROM orders_0 GROUP BY customer_id",
        ),
    ];
    tprintln!("  [diag] single-shot per-query timing (20-sample distribution)");
    for (label, sql) in &queries {
        let mut samples_us: Vec<u128> = Vec::with_capacity(20);
        for _ in 0..20 {
            let t = Instant::now();
            let _ = diag_client.simple_query(sql).await.expect("diag");
            samples_us.push(t.elapsed().as_micros());
        }
        samples_us.sort_unstable();
        let min = samples_us[0];
        let med = samples_us[samples_us.len() / 2];
        let max = samples_us[samples_us.len() - 1];
        tprintln!(
            "    {:<18}  min={:>6} us  med={:>6} us  max={:>6} us",
            label,
            min,
            med,
            max
        );
    }

    // Linear-growth probe, run COUNT(*) 100 times in one connection
    // and report per-bucket median to see if latency rises with iteration
    tprintln!("  [diag] 100 sequential COUNT(*) queries, bucketed median latency");
    let mut bucket_samples: [Vec<u128>; 10] = std::array::from_fn(|_| Vec::with_capacity(10));
    let mut row_counts: [Vec<i64>; 10] = std::array::from_fn(|_| Vec::with_capacity(10));
    for i in 0..100u32 {
        let t = Instant::now();
        let res = diag_client
            .simple_query("SELECT COUNT(*) FROM orders_0")
            .await
            .expect("diag growth probe");
        bucket_samples[(i / 10) as usize].push(t.elapsed().as_micros());
        let cnt: i64 = res
            .first()
            .and_then(|r| r.rows.first())
            .and_then(|row| row.first())
            .and_then(|c| c.as_ref())
            .and_then(|b| std::str::from_utf8(b).ok())
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(-1);
        row_counts[(i / 10) as usize].push(cnt);
    }
    for (b, samples) in bucket_samples.iter_mut().enumerate() {
        samples.sort_unstable();
        let med = samples[samples.len() / 2];
        let row_med = {
            let mut rc = row_counts[b].clone();
            rc.sort_unstable();
            rc[rc.len() / 2]
        };
        tprintln!(
            "    iter {:>3}-{:<3}  med={:>6} us  COUNT(*)={}",
            b * 10,
            b * 10 + 9,
            med,
            row_med
        );
    }
    drop(diag_client);

    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for _ in 0..VALIDATION_RUNS {
        runs.push(run_analytics(addr, db).await);
    }
    runs
}

/// Gateway routes and middleware pipeline exercised in-process.
///
/// Does not spin up an HTTP listener because the endpoint executor requires
/// the endpoint_registrar and publication_manager wiring that lives in the
/// full Server::init path. What we measure here is the same router and
/// middleware code that every HTTP request flows through once those
/// managers are bound.
fn run_gateway() -> f64 {
    let router = Arc::new(Router::new());
    for i in 0..64 {
        router.insert(make_gateway_route(&format!("/api/v1/orders/{}", i)));
    }
    for i in 0..16 {
        router.insert(make_gateway_route(&format!(
            "/api/v1/customers/:id/slot_{}",
            i
        )));
    }

    let rate = Arc::new(RateLimiter::new());
    let req = HttpRequest {
        method: HttpMethod::Get,
        path: "/api/v1/orders/17".to_string(),
        query_string: String::new(),
        headers: std::collections::HashMap::new(),
        body: Vec::new(),
        peer_addr: None,
        tls_info: None,
    };

    // Warm the router path so the first iteration does not skew the measurement.
    for _ in 0..128 {
        let _ = router.lookup(HttpMethod::Get, &req.path);
    }

    let started = Instant::now();
    let mut hits = 0u64;
    for _ in 0..GATEWAY_ITERATIONS {
        if let Some((route, params)) = router.lookup(HttpMethod::Get, &req.path) {
            let outcome = run_pipeline(route, params, &req, &rate, None);
            if matches!(outcome, MiddlewareOutcome::Execute { .. }) {
                hits += 1;
            }
        }
    }
    let elapsed = started.elapsed().as_secs_f64().max(1e-9);
    assert_eq!(
        hits, GATEWAY_ITERATIONS as u64,
        "pipeline should reach Execute on all"
    );
    GATEWAY_ITERATIONS as f64 / elapsed
}

fn make_gateway_route(pattern: &str) -> CompiledRoute {
    CompiledRoute::compile(
        zyron_catalog::EndpointId(1),
        "ep".to_string(),
        pattern.to_string(),
        vec![HttpMethod::Get, HttpMethod::Post],
        EndpointAuthMode::None,
        Vec::new(),
        EndpointOutputFormat::Json,
        vec!["*".to_string()],
        0,
        30,
        65_536,
        "SELECT 1".to_string(),
    )
}

// ---------------------------------------------------------------------------
// Top-level scenario
// ---------------------------------------------------------------------------

#[test]
fn test_e2e_consumer_deployment() {
    zyron_bench_harness::init("end_to_end");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Zyron End-to-End Consumer Deployment ===\n");
    tprintln!(
        "  Scenario simulates: cold boot -> schema -> seed -> OLTP -> analytics -> gateway -> shutdown\n"
    );

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(8)
        .enable_all()
        .build()
        .expect("rt");

    let local = tokio::task::LocalSet::new();
    local.block_on(&rt, async {
        // ------------------------------------------------------------------
        // Cold startup
        // ------------------------------------------------------------------
        tprintln!("=== Cold startup ===");
        let (server, startup_elapsed) = boot_server("app").await;
        let startup_ms = startup_elapsed.as_secs_f64() * 1000.0;
        tprintln!(
            "  Listener ready at {} in {:.2} ms",
            server.addr, startup_ms
        );

        // ------------------------------------------------------------------
        // Bootstrap (handshake + schema DDL + seed)
        // ------------------------------------------------------------------
        tprintln!("\n=== Bootstrap (handshake, schema, seed) ===");
        let (first_ready_ms, schema_ddl_ms, seed_rps) =
            run_bootstrap(server.addr, "app").await;
        tprintln!(
            "  First ReadyForQuery {:.2} ms, schema DDL {:.2} ms, seed {:.0} rows/sec",
            first_ready_ms, schema_ddl_ms, seed_rps,
        );

        // ------------------------------------------------------------------
        // Pre-OLTP analytics probe
        // ------------------------------------------------------------------
        tprintln!("\n=== Pre-OLTP analytics probe ===");
        {
            let app_id = server
                .state
                .catalog
                .get_schema(server.state.catalog.get_database("app").map(|d| d.id).unwrap_or(zyron_catalog::SYSTEM_DATABASE_ID), "app")
                .map(|s| s.id);
            tprintln!("  app schema id = {:?}", app_id);
            if let Ok(sid) = app_id {
                match server.state.catalog.get_table(sid, "orders_0") {
                    Ok(t) => {
                        let n = server
                            .state
                            .disk_manager
                            .num_pages(t.heap_file_id)
                            .await
                            .unwrap_or(0);
                        tprintln!(
                            "  orders_0 heap_file_id={} num_pages BEFORE OLTP = {}",
                            t.heap_file_id, n
                        );
                    }
                    Err(e) => tprintln!("  get_table orders_0 failed: {:?}", e),
                }
            }
        }
        let _ = run_analytics_multi(server.addr, "app").await;

        // ------------------------------------------------------------------
        // OLTP across varying concurrency
        // ------------------------------------------------------------------
        tprintln!("\n=== OLTP workload ({}s at each concurrency) ===", OLTP_DURATION.as_secs());
        let mut oltp_results: Vec<(usize, OltpOutcome)> = Vec::new();
        for &conc in OLTP_CONCURRENCY_LEVELS {
            // Reset phase counters so the breakdown below covers this
            // concurrency point alone. Compiled out entirely without
            // --features profile; gated again at runtime by ZYRON_PROFILE.
            #[cfg(feature = "profile")]
            zyron_common::profile::reset();
            let outcome = run_oltp(server.addr, "app", conc).await;
            let total_ops = outcome.success_ops + outcome.failed_ops;
            let fail_pct = if total_ops == 0 {
                0.0
            } else {
                100.0 * outcome.failed_ops as f64 / total_ops as f64
            };
            tprintln!(
                "  c={:>2}  tps={:>10.0}  ok={:>8}  fail={:>6} ({:>4.1}%)  p50={:>4} us  p95={:>4} us  p99={:>5} us  p999={:>6} us  max={:>6} us",
                conc, outcome.tps,
                outcome.success_ops, outcome.failed_ops, fail_pct,
                outcome.latency.p50_us, outcome.latency.p95_us, outcome.latency.p99_us,
                outcome.latency.p999_us, outcome.latency.max_us,
            );
            if let Some(ref err) = outcome.sample {
                tprintln!("    first error: {}", err);
            }
            // Full-stack per-phase wall-clock breakdown for this concurrency,
            // written through tprintln so it lands in the benchmark .txt file.
            // wire.* isolates protocol/parse/plan/execute, txn.* the commit
            // path, flush.* the WAL. Compiled out entirely without --features
            // profile; gated again at runtime by ZYRON_PROFILE.
            #[cfg(feature = "profile")]
            if zyron_common::profile::is_enabled() {
                tprintln!("[OLTP c={}]\n{}", conc, zyron_common::profile::report());
            }
            // Non-trivial failure rate means the workload isn't exercising
            // what the SLO pretends to measure
            assert!(
                fail_pct < 5.0,
                "OLTP c={} has {:.1}% failure rate, sample error {:?}",
                conc, fail_pct, outcome.sample,
            );
            oltp_results.push((conc, outcome));
        }

        // ------------------------------------------------------------------
        // Analytical queries (5-run average)
        // ------------------------------------------------------------------
        tprintln!("\n=== Analytical queries (5-run average) ===");
        {
            if let Ok(sid) = server
                .state
                .catalog
                .get_schema(server.state.catalog.get_database("app").map(|d| d.id).unwrap_or(zyron_catalog::SYSTEM_DATABASE_ID), "app")
                .map(|s| s.id)
            {
                if let Ok(t) = server.state.catalog.get_table(sid, "orders_0") {
                    let n = server
                        .state
                        .disk_manager
                        .num_pages(t.heap_file_id)
                        .await
                        .unwrap_or(0);
                    let on_disk = std::fs::metadata(
                        server
                            .state
                            .data_dir
                            .join(format!("{}.data", t.heap_file_id)),
                    )
                    .map(|m| m.len() / PAGE_SIZE as u64)
                    .unwrap_or(0);
                    tprintln!(
                        "  orders_0 heap_file_id={} reported_num_pages={} actual_file_pages={}",
                        t.heap_file_id, n, on_disk
                    );
                }
            }
        }
        let analytic_runs = run_analytics_multi(server.addr, "app").await;
        for (i, r) in analytic_runs.iter().enumerate() {
            tprintln!(
                "  run {}/{}  mean={:.1} us  p50={} us  p95={} us  p99={} us  (n={})",
                i + 1,
                analytic_runs.len(),
                r.mean_us,
                r.p50_us,
                r.p95_us,
                r.p99_us,
                r.count,
            );
        }
        let p50_runs: Vec<f64> = analytic_runs.iter().map(|r| r.p50_us as f64).collect();
        let p50_avg = p50_runs.iter().sum::<f64>() / p50_runs.len() as f64;
        let p50_min = p50_runs.iter().cloned().fold(f64::INFINITY, f64::min);
        let p50_max = p50_runs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        tprintln!(
            "  median (5-run) avg {:.0} min {:.0} max {:.0} us",
            p50_avg, p50_min, p50_max
        );
        let analytic = analytic_runs
            .into_iter()
            .min_by_key(|r| r.p50_us)
            .expect("at least one run");

        // ------------------------------------------------------------------
        // HTTP gateway router and pipeline
        // ------------------------------------------------------------------
        tprintln!("\n=== Gateway routing and middleware ===");
        let gateway_ops = run_gateway();
        tprintln!("  pipeline throughput {:.0} ops/sec", gateway_ops);

        // ------------------------------------------------------------------
        // Shutdown
        // ------------------------------------------------------------------
        tprintln!("\n=== Shutdown ===");
        let t_shutdown = Instant::now();
        // Drop the listener so the accept loop exits cleanly
        // ServerState drops after remaining connection tasks finish
        // WAL flush on drop is what real shutdown measures
        let E2EServer {
            state,
            listener,
            _background_writer,
            _tmp,
            addr: _,
        } = server;
        drop(listener);
        // Allow in-flight client tasks to observe EOF and unwind
        tokio::time::sleep(Duration::from_millis(50)).await;
        drop(state);
        // BackgroundWriter drop runs final dirty page flush like production
        drop(_background_writer);
        let shutdown_ms = t_shutdown.elapsed().as_secs_f64() * 1000.0;
        tprintln!("  teardown complete in {:.2} ms", shutdown_ms);

        // ------------------------------------------------------------------
        // Validate against SLO targets
        // ------------------------------------------------------------------
        tprintln!("\n=== SLO Validation ===");
        check_performance(
            "Startup",
            "cold boot latency (ms)",
            startup_ms,
            STARTUP_TARGET_MS,
            false,
        );
        check_performance(
            "Bootstrap",
            "first ReadyForQuery (ms)",
            first_ready_ms,
            FIRST_READY_TARGET_MS,
            false,
        );
        check_performance(
            "Bootstrap",
            "schema DDL total (ms)",
            schema_ddl_ms,
            SCHEMA_DDL_TARGET_MS,
            false,
        );
        check_performance(
            "Bootstrap",
            "seed insert (rows/sec)",
            seed_rps,
            SEED_INSERT_TARGET_ROWS_PER_SEC,
            true,
        );

        // OLTP: record all concurrency points, gate on c=16 as the SLO anchor.
        for (conc, outcome) in &oltp_results {
            check_performance(
                "OLTP",
                &format!("c{} tps", conc),
                outcome.tps,
                // only c16 target enforced; lower concurrencies use sentinel
                if *conc == 16 { OLTP_C16_TPS_TARGET } else { 1.0 },
                true,
            );
            check_performance(
                "OLTP",
                &format!("c{} p99 us", conc),
                outcome.latency.p99_us as f64,
                if *conc == 16 { OLTP_C16_P99_US_TARGET } else { f64::INFINITY },
                false,
            );
        }

        check_performance(
            "Analytics",
            "median query us",
            analytic.p50_us as f64,
            ANALYTIC_MEDIAN_US_TARGET,
            false,
        );
        check_performance(
            "Gateway",
            "pipeline ops/sec",
            gateway_ops,
            GATEWAY_PIPELINE_TARGET_OPS,
            true,
        );
        check_performance(
            "Shutdown",
            "teardown (ms)",
            shutdown_ms,
            SHUTDOWN_TARGET_MS,
            false,
        );

        tprintln!("\n=== End-to-End complete ===\n");
    });
}

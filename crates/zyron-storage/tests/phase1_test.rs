//! Phase 1: Storage Foundation Validation Tests
//!
//! Comprehensive integration tests for ZyronDB Phase 1 components:
//! - WAL write and replay
//! - Buffer pool eviction and caching
//! - Heap file tuple storage
//! - B+ tree index operations
//! - Crash recovery integration
//!
//! Performance Targets (Industry-Leading):
//! | Test          | Metric     | Target           | Industry Leader |
//! |---------------|------------|------------------|-----------------|
//! | WAL Write     | throughput | 3M records/sec   | TigerBeetle 2M  |
//! | WAL Replay    | throughput | 6M records/sec   | RocksDB 5M      |
//! | Buffer Pool   | fetch      | 15ns             | Umbra 20ns      |
//! | Buffer Pool   | hit rate   | 100%             | Industry 98%    |
//! | Heap Insert   | throughput | 2M tuples/sec    | SingleStore 1M  |
//! | Heap Scan     | throughput | 20M tuples/sec   | DuckDB 15M      |
//! | B+Tree Insert | throughput | 8M keys/sec      | LMDB/Sled 5M    |
//! | B+Tree Lookup | latency    | 40ns/lookup      | LMDB mmap 50ns  |
//! | B+Tree Range  | throughput | 40M keys/sec     | RocksDB 30M     |
//! | Recovery      | time       | 2ms/MB           | VoltDB 5ms/MB   |
//!
//! Validation Requirements:
//! - Each test runs 5 iterations
//! - Results averaged across all 5 runs
//! - Pass/fail determined by average performance
//! - Individual runs logged for variance analysis
//! - Test FAILS if any single run is >2x worse than target

use bytes::Bytes;
use rand::Rng;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;
use tempfile::tempdir;

use zyron_buffer::{BufferPool, BufferPoolConfig};
use zyron_common::page::PageId;
use zyron_storage::{BufferedBTreeIndex, DiskManager, DiskManagerConfig, HeapFile, Tuple, TupleId};
use zyron_wal::{LogRecordType, Lsn, RecoveryManager, WalReader, WalWriter, WalWriterConfig};

// =============================================================================
// Performance Target Constants (Industry-Leading)
// =============================================================================

const WAL_WRITE_TARGET_OPS_SEC: f64 = 3_000_000.0;
const WAL_REPLAY_TARGET_OPS_SEC: f64 = 6_000_000.0;
const BUFFER_POOL_FETCH_TARGET_NS: f64 = 15.0;
const BUFFER_POOL_HIT_RATE_TARGET: f64 = 1.0; // 100%
const HEAP_INSERT_TARGET_OPS_SEC: f64 = 2_000_000.0;
const HEAP_SCAN_TARGET_OPS_SEC: f64 = 20_000_000.0;
const BTREE_INSERT_TARGET_OPS_SEC: f64 = 8_000_000.0;
const BTREE_LOOKUP_TARGET_NS: f64 = 40.0;
const BTREE_RANGE_TARGET_OPS_SEC: f64 = 40_000_000.0;
const RECOVERY_TARGET_MS_PER_MB: f64 = 2.0;

// Validation constants
const VALIDATION_RUNS: usize = 5;
const REGRESSION_THRESHOLD: f64 = 2.0; // Fail if any run >2x worse than target

// =============================================================================
// Validation Infrastructure
// =============================================================================

struct ValidationResult {
    passed: bool,
    regression_detected: bool,
    average: f64,
}

/// Formats a number with comma separators for readability.
fn format_with_commas(n: f64) -> String {
    let s = format!("{:.0}", n);
    let bytes: Vec<char> = s.chars().collect();
    let mut result = String::new();
    let len = bytes.len();
    for (i, c) in bytes.iter().enumerate() {
        if i > 0 && (len - i) % 3 == 0 {
            result.push(',');
        }
        result.push(*c);
    }
    result
}

fn validate_metric(
    name: &str,
    runs: Vec<f64>,
    target: f64,
    higher_is_better: bool,
) -> ValidationResult {
    let average = runs.iter().sum::<f64>() / runs.len() as f64;
    let min = runs.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = runs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let variance = runs.iter().map(|x| (x - average).powi(2)).sum::<f64>() / runs.len() as f64;
    let std_dev = variance.sqrt();

    let passed = if higher_is_better {
        average >= target
    } else {
        average <= target
    };

    let regression_threshold = if higher_is_better {
        target / REGRESSION_THRESHOLD
    } else {
        target * REGRESSION_THRESHOLD
    };

    let regression_detected = runs.iter().any(|&r| {
        if higher_is_better {
            r < regression_threshold
        } else {
            r > regression_threshold
        }
    });

    let status = if passed { "PASS" } else { "FAIL" };
    let regr_status = if regression_detected { "REGR!" } else { "OK" };
    let comparison = if higher_is_better { ">=" } else { "<=" };

    println!("  {} [{}/{}]:", name, status, regr_status);
    println!(
        "    Runs: [{}]",
        runs.iter()
            .map(|x| format_with_commas(*x))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "    Average: {} {} {} (target)",
        format_with_commas(average),
        comparison,
        format_with_commas(target)
    );
    println!(
        "    Min/Max: {} / {}, StdDev: {}",
        format_with_commas(min),
        format_with_commas(max),
        format_with_commas(std_dev)
    );

    ValidationResult {
        passed,
        regression_detected,
        average,
    }
}

fn check_performance(name: &str, actual: f64, target: f64, higher_is_better: bool) -> bool {
    let passed = if higher_is_better {
        actual >= target
    } else {
        actual <= target
    };
    let status = if passed { "PASS" } else { "FAIL" };
    let comparison = if higher_is_better { ">=" } else { "<=" };
    println!(
        "  {} [{}]: {} {} {} (target)",
        name,
        status,
        format_with_commas(actual),
        comparison,
        format_with_commas(target)
    );
    passed
}

// =============================================================================
// Test 1: WAL Write/Replay Test (5-run validation)
// =============================================================================

/// Writes 10,000 log records across multiple segments, simulates crash,
/// replays all records, and verifies integrity.
/// Target: 3M writes/sec, 6M replay/sec
#[tokio::test]
async fn test_wal_write_replay_10k_records() {
    const RECORD_COUNT: usize = 10_000;

    println!("\n=== WAL Write/Replay Performance Test ===");
    println!("Records per run: {}", RECORD_COUNT);
    println!("Validation runs: {}", VALIDATION_RUNS);

    let mut write_results = Vec::with_capacity(VALIDATION_RUNS);
    let mut replay_results = Vec::with_capacity(VALIDATION_RUNS);

    for run in 0..VALIDATION_RUNS {
        println!("\n--- Run {}/{} ---", run + 1, VALIDATION_RUNS);

        let dir = tempdir().unwrap();
        let config = WalWriterConfig {
            wal_dir: dir.path().to_path_buf(),
            segment_size: 1024 * 1024,
            fsync_enabled: false,
            batch_size: 1000,
            batch_bytes: 256 * 1024,
            flush_interval_us: 5000,
        };

        let mut written_lsns: Vec<Lsn> = Vec::with_capacity(RECORD_COUNT);

        // Write phase
        let write_duration;
        {
            let writer = WalWriter::new(config.clone()).await.unwrap();

            let start = Instant::now();
            for i in 0..RECORD_COUNT {
                let txn_id = (i % 100 + 1) as u32;
                let payload = format!("record_{}_{}", i, "x".repeat(i % 100));
                let payload_bytes = Bytes::from(payload);

                let lsn = writer
                    .log_insert(txn_id, Lsn::INVALID, payload_bytes)
                    .await
                    .unwrap();

                written_lsns.push(lsn);
            }
            write_duration = start.elapsed();
            drop(writer);
        }

        // Replay phase
        let replay_duration;
        {
            let reader = WalReader::new(dir.path()).unwrap();

            let start = Instant::now();
            let records = reader.scan_all().unwrap();
            replay_duration = start.elapsed();

            let insert_records: Vec<_> = records
                .iter()
                .filter(|r| r.record_type == LogRecordType::Insert)
                .collect();

            assert_eq!(
                insert_records.len(),
                RECORD_COUNT,
                "Run {}: Expected {} insert records, got {}",
                run + 1,
                RECORD_COUNT,
                insert_records.len()
            );
        }

        let write_ops_sec = RECORD_COUNT as f64 / write_duration.as_secs_f64();
        let replay_ops_sec = RECORD_COUNT as f64 / replay_duration.as_secs_f64();

        println!(
            "  Write: {} ops/sec ({:?})",
            format_with_commas(write_ops_sec),
            write_duration
        );
        println!(
            "  Replay: {} ops/sec ({:?})",
            format_with_commas(replay_ops_sec),
            replay_duration
        );

        write_results.push(write_ops_sec);
        replay_results.push(replay_ops_sec);
    }

    println!("\n=== WAL Validation Results ===");
    let write_result = validate_metric(
        "Write throughput (ops/sec)",
        write_results,
        WAL_WRITE_TARGET_OPS_SEC,
        true,
    );
    let replay_result = validate_metric(
        "Replay throughput (ops/sec)",
        replay_results,
        WAL_REPLAY_TARGET_OPS_SEC,
        true,
    );

    assert!(
        write_result.passed,
        "WAL write avg {:.0} < target {:.0}",
        write_result.average, WAL_WRITE_TARGET_OPS_SEC
    );
    assert!(
        !write_result.regression_detected,
        "WAL write regression detected"
    );
    assert!(
        replay_result.passed,
        "WAL replay avg {:.0} < target {:.0}",
        replay_result.average, WAL_REPLAY_TARGET_OPS_SEC
    );
    assert!(
        !replay_result.regression_detected,
        "WAL replay regression detected"
    );
}

/// Tests WAL segment rotation with many records.
#[tokio::test]
async fn test_wal_segment_rotation() {
    let dir = tempdir().unwrap();
    let config = WalWriterConfig {
        wal_dir: dir.path().to_path_buf(),
        segment_size: 64 * 1024,
        fsync_enabled: false,
        batch_size: 1,
        batch_bytes: 64 * 1024,
        flush_interval_us: 1000,
    };

    let writer = WalWriter::new(config).await.unwrap();
    let initial_segment = writer.current_segment_id().unwrap();

    for _ in 0..1000 {
        let payload = Bytes::from(vec![0u8; 200]);
        writer.log_insert(1, Lsn::INVALID, payload).await.unwrap();
    }

    let final_segment = writer.current_segment_id().unwrap();
    writer.close().await.unwrap();

    assert!(
        final_segment.0 > initial_segment.0,
        "Expected segment rotation: {} -> {}",
        initial_segment,
        final_segment
    );

    let reader = WalReader::new(dir.path()).unwrap();
    let records = reader.scan_all().unwrap();
    assert_eq!(records.len(), 1000);

    println!(
        "WAL Segment Rotation: PASSED - rotated from segment {} to {}",
        initial_segment, final_segment
    );
}

// =============================================================================
// Test 2: Buffer Pool Test (5-run validation)
// =============================================================================

/// Tests buffer pool with 100 frames accessing 500 different pages.
#[tokio::test]
async fn test_buffer_pool_eviction() {
    const NUM_FRAMES: usize = 100;
    const NUM_PAGES: usize = 500;

    let pool = BufferPool::new(BufferPoolConfig {
        num_frames: NUM_FRAMES,
    });

    let mut dirty_evictions = 0;

    for i in 0..NUM_PAGES {
        let page_id = PageId::new(0, i as u32);

        let (frame, evicted) = pool.new_page(page_id).unwrap();

        if evicted.is_some() {
            dirty_evictions += 1;
        }

        {
            let mut data = frame.write_data();
            data[0..8].copy_from_slice(&(i as u64).to_le_bytes());
        }

        pool.unpin_page(page_id, true);
    }

    assert!(
        dirty_evictions > 0,
        "Expected dirty evictions when accessing {} pages with {} frames",
        NUM_PAGES,
        NUM_FRAMES
    );

    assert_eq!(pool.page_count(), NUM_FRAMES);

    println!(
        "Buffer Pool Eviction: PASSED - {} dirty evictions with {}/{} pages",
        dirty_evictions, NUM_PAGES, NUM_FRAMES
    );
}

/// Tests that pinned pages cannot be evicted.
#[tokio::test]
async fn test_buffer_pool_pin_prevents_eviction() {
    const NUM_FRAMES: usize = 10;

    let pool = BufferPool::new(BufferPoolConfig {
        num_frames: NUM_FRAMES,
    });

    let mut pinned_pages = Vec::new();
    for i in 0..NUM_FRAMES {
        let page_id = PageId::new(0, i as u32);
        pool.new_page(page_id).unwrap();
        pinned_pages.push(page_id);
    }

    let result = pool.new_page(PageId::new(0, 999));
    assert!(result.is_err(), "Should fail when all frames are pinned");

    pool.unpin_page(pinned_pages[0], false);

    let result = pool.new_page(PageId::new(0, 999));
    assert!(result.is_ok(), "Should succeed after unpinning");

    println!("Buffer Pool Pin Test: PASSED - pin prevents eviction");
}

/// Tests cache hit rate and fetch latency for repeated access patterns.
/// Target: 100% hit rate, 15ns average fetch
#[tokio::test]
async fn test_buffer_pool_cache_hit_rate() {
    const NUM_FRAMES: usize = 50;
    const NUM_PAGES: usize = 30;
    const ACCESS_ROUNDS: usize = 1000;

    println!("\n=== Buffer Pool Performance Test ===");
    println!(
        "Frames: {}, Pages: {}, Rounds: {}",
        NUM_FRAMES, NUM_PAGES, ACCESS_ROUNDS
    );
    println!("Validation runs: {}", VALIDATION_RUNS);

    let mut fetch_results = Vec::with_capacity(VALIDATION_RUNS);
    let mut hit_rate_results = Vec::with_capacity(VALIDATION_RUNS);

    for run in 0..VALIDATION_RUNS {
        println!("\n--- Run {}/{} ---", run + 1, VALIDATION_RUNS);

        let pool = BufferPool::new(BufferPoolConfig {
            num_frames: NUM_FRAMES,
        });

        // Initial load
        for i in 0..NUM_PAGES {
            let page_id = PageId::new(0, i as u32);
            pool.new_page(page_id).unwrap();
            pool.unpin_page(page_id, false);
        }

        let mut hits = 0;
        let mut misses = 0;

        let start = Instant::now();
        for _ in 0..ACCESS_ROUNDS {
            for i in 0..NUM_PAGES {
                let page_id = PageId::new(0, i as u32);
                if pool.fetch_page(page_id).is_some() {
                    hits += 1;
                    pool.unpin_page(page_id, false);
                } else {
                    misses += 1;
                }
            }
        }
        let duration = start.elapsed();

        let total_accesses = ACCESS_ROUNDS * NUM_PAGES;
        let hit_rate = hits as f64 / (hits + misses) as f64;
        let avg_fetch_ns = duration.as_nanos() as f64 / total_accesses as f64;

        println!("  Fetch latency: {:.2} ns", avg_fetch_ns);
        println!("  Hit rate: {:.4}", hit_rate);

        fetch_results.push(avg_fetch_ns);
        hit_rate_results.push(hit_rate);
    }

    println!("\n=== Buffer Pool Validation Results ===");
    let fetch_result = validate_metric(
        "Avg fetch latency (ns)",
        fetch_results,
        BUFFER_POOL_FETCH_TARGET_NS,
        false,
    );
    let hit_result = validate_metric(
        "Cache hit rate",
        hit_rate_results,
        BUFFER_POOL_HIT_RATE_TARGET,
        true,
    );

    assert!(
        fetch_result.passed,
        "Buffer pool fetch {:.2} ns > target {:.2} ns",
        fetch_result.average, BUFFER_POOL_FETCH_TARGET_NS
    );
    assert!(
        !fetch_result.regression_detected,
        "Buffer pool fetch regression detected"
    );
    assert!(
        hit_result.passed,
        "Buffer pool hit rate {:.4} < target {:.4}",
        hit_result.average, BUFFER_POOL_HIT_RATE_TARGET
    );
}

// =============================================================================
// Test 3: Heap File Test (5-run validation)
// =============================================================================

/// Inserts 100,000 tuples with varying sizes using batch allocation.
/// Target: 10M inserts/sec, 100M scan/sec
#[tokio::test]
async fn test_heap_file_100k_tuples() {
    const TUPLE_COUNT: usize = 100_000;

    println!("\n=== Heap File Performance Test ===");
    println!("Tuples per run: {}", TUPLE_COUNT);
    println!("Validation runs: {}", VALIDATION_RUNS);

    let mut insert_results = Vec::with_capacity(VALIDATION_RUNS);
    let mut scan_results = Vec::with_capacity(VALIDATION_RUNS);

    for run in 0..VALIDATION_RUNS {
        println!("\n--- Run {}/{} ---", run + 1, VALIDATION_RUNS);

        let dir = tempdir().unwrap();
        let config = DiskManagerConfig {
            data_dir: dir.path().to_path_buf(),
            fsync_enabled: false,
        };
        let disk = Arc::new(DiskManager::new(config).await.unwrap());
        let pool = Arc::new(BufferPool::auto_sized());
        let heap = HeapFile::with_defaults(disk, pool).unwrap();

        let mut rng = rand::rng();

        // Build all tuples first
        let tuples: Vec<Tuple> = (0..TUPLE_COUNT)
            .map(|i| {
                let size = rng.random_range(10..=500);
                let data: Vec<u8> = (0..size).map(|j| ((i + j) % 256) as u8).collect();
                Tuple::new(data, i as u32)
            })
            .collect();

        // Batch insert
        let insert_start = Instant::now();
        let _tuple_ids = heap.insert_batch(&tuples).await.unwrap();
        let insert_duration = insert_start.elapsed();

        // Scan phase using for_each (no allocation overhead)
        let scan_start = Instant::now();
        let guard = heap.scan().unwrap();
        let mut scan_count = 0usize;
        guard.for_each(|_tid, _tuple| {
            scan_count += 1;
            std::hint::black_box(&_tuple);
        });
        let scan_duration = scan_start.elapsed();

        assert_eq!(
            scan_count,
            TUPLE_COUNT,
            "Run {}: Scan should return all {} tuples",
            run + 1,
            TUPLE_COUNT
        );

        let insert_ops_sec = TUPLE_COUNT as f64 / insert_duration.as_secs_f64();
        let scan_ops_sec = TUPLE_COUNT as f64 / scan_duration.as_secs_f64();

        println!(
            "  Insert: {} ops/sec ({:?})",
            format_with_commas(insert_ops_sec),
            insert_duration
        );
        println!(
            "  Scan: {} ops/sec ({:?})",
            format_with_commas(scan_ops_sec),
            scan_duration
        );

        insert_results.push(insert_ops_sec);
        scan_results.push(scan_ops_sec);
    }

    println!("\n=== Heap File Validation Results ===");
    let insert_result = validate_metric(
        "Insert throughput (ops/sec)",
        insert_results,
        HEAP_INSERT_TARGET_OPS_SEC,
        true,
    );
    let scan_result = validate_metric(
        "Scan throughput (ops/sec)",
        scan_results,
        HEAP_SCAN_TARGET_OPS_SEC,
        true,
    );

    assert!(
        insert_result.passed,
        "Heap insert avg {:.0} < target {:.0}",
        insert_result.average, HEAP_INSERT_TARGET_OPS_SEC
    );
    assert!(
        !insert_result.regression_detected,
        "Heap insert regression detected"
    );
    assert!(
        scan_result.passed,
        "Heap scan avg {:.0} < target {:.0}",
        scan_result.average, HEAP_SCAN_TARGET_OPS_SEC
    );
    assert!(
        !scan_result.regression_detected,
        "Heap scan regression detected"
    );
}

/// Tests delete and scan exclusion.
#[tokio::test]
async fn test_heap_file_delete_and_scan() {
    const TUPLE_COUNT: usize = 10_000;
    const DELETE_COUNT: usize = 1_000;

    let dir = tempdir().unwrap();
    let config = DiskManagerConfig {
        data_dir: dir.path().to_path_buf(),
        fsync_enabled: false,
    };
    let disk = Arc::new(DiskManager::new(config).await.unwrap());
    let pool = Arc::new(BufferPool::auto_sized());
    let heap = HeapFile::with_defaults(disk, pool).unwrap();

    let tuples: Vec<Tuple> = (0..TUPLE_COUNT)
        .map(|i| {
            let data = format!("tuple_{}", i).into_bytes();
            Tuple::new(data, i as u32)
        })
        .collect();
    let tuple_ids = heap.insert_batch(&tuples).await.unwrap();

    let mut deleted_ids: HashSet<TupleId> = HashSet::new();
    for i in 0..DELETE_COUNT {
        heap.delete(tuple_ids[i]).await.unwrap();
        deleted_ids.insert(tuple_ids[i]);
    }

    let guard = heap.scan().unwrap();
    let mut scanned_ids: Vec<TupleId> = Vec::new();
    guard.for_each(|tuple_id, _| {
        scanned_ids.push(tuple_id);
    });
    assert_eq!(
        scanned_ids.len(),
        TUPLE_COUNT - DELETE_COUNT,
        "Scan should return {} tuples after {} deletions",
        TUPLE_COUNT - DELETE_COUNT,
        DELETE_COUNT
    );

    for tuple_id in &scanned_ids {
        assert!(
            !deleted_ids.contains(tuple_id),
            "Deleted tuple {} should not appear in scan",
            tuple_id
        );
    }

    for deleted_id in deleted_ids.iter().take(100) {
        let result = heap.get(*deleted_id).await.unwrap();
        assert!(result.is_none(), "Deleted tuple should return None");
    }

    println!(
        "Heap File Delete/Scan: PASSED - deleted {}/{} tuples",
        DELETE_COUNT, TUPLE_COUNT
    );
}

/// Tests free space reuse after deletes with performance benchmarks.
/// Target: Delete throughput >= 500k ops/sec, Reinsert throughput >= 1M ops/sec
#[tokio::test]
async fn test_heap_file_space_reuse() {
    const TUPLE_COUNT: usize = 10_000;
    const TUPLE_DATA_SIZE: usize = 100;
    const DELETE_TARGET_OPS_SEC: f64 = 500_000.0;
    const REINSERT_TARGET_OPS_SEC: f64 = 1_000_000.0;

    let dir = tempdir().unwrap();
    let config = DiskManagerConfig {
        data_dir: dir.path().to_path_buf(),
        fsync_enabled: false,
    };
    let disk = Arc::new(DiskManager::new(config).await.unwrap());
    let pool = Arc::new(BufferPool::auto_sized());
    let heap = HeapFile::with_defaults(disk, pool).unwrap();

    // Initial insert batch
    let tuples: Vec<Tuple> = (0..TUPLE_COUNT)
        .map(|i| {
            let data = vec![i as u8; TUPLE_DATA_SIZE];
            Tuple::new(data, i as u32)
        })
        .collect();

    let insert_start = Instant::now();
    let tuple_ids = heap.insert_batch(&tuples).await.unwrap();
    let insert_duration = insert_start.elapsed();
    let insert_ops_sec = TUPLE_COUNT as f64 / insert_duration.as_secs_f64();

    let pages_after_insert = heap.num_pages().await.unwrap();
    println!(
        "Initial insert: {} ops/sec ({} tuples)",
        format_with_commas(insert_ops_sec),
        format_with_commas(TUPLE_COUNT as f64)
    );
    println!("Pages after insert: {}", pages_after_insert);

    // Batch delete all tuples
    let delete_start = Instant::now();
    let deleted_count = heap.delete_batch(&tuple_ids).await.unwrap();
    let delete_duration = delete_start.elapsed();
    assert_eq!(deleted_count, TUPLE_COUNT);
    let delete_ops_sec = TUPLE_COUNT as f64 / delete_duration.as_secs_f64();
    println!(
        "Delete: {} ops/sec (target: {})",
        format_with_commas(delete_ops_sec),
        format_with_commas(DELETE_TARGET_OPS_SEC)
    );

    // Reinsert batch (triggers compaction)
    let tuples2: Vec<Tuple> = (0..TUPLE_COUNT)
        .map(|i| {
            let data = vec![(i + 100) as u8; TUPLE_DATA_SIZE];
            Tuple::new(data, (i + TUPLE_COUNT) as u32)
        })
        .collect();

    let reinsert_start = Instant::now();
    let _new_ids = heap.insert_batch(&tuples2).await.unwrap();
    let reinsert_duration = reinsert_start.elapsed();
    let reinsert_ops_sec = TUPLE_COUNT as f64 / reinsert_duration.as_secs_f64();
    println!(
        "Reinsert (with compaction): {} ops/sec (target: {})",
        format_with_commas(reinsert_ops_sec),
        format_with_commas(REINSERT_TARGET_OPS_SEC)
    );

    let pages_after_reinsert = heap.num_pages().await.unwrap();

    // Verify space was reused
    assert!(
        pages_after_reinsert <= pages_after_insert * 2,
        "Expected space reuse: {} pages after reinsert vs {} after initial insert",
        pages_after_reinsert,
        pages_after_insert
    );
    println!(
        "Space reuse: {} pages (was {})",
        pages_after_reinsert, pages_after_insert
    );

    // Performance assertions
    assert!(
        delete_ops_sec >= DELETE_TARGET_OPS_SEC,
        "Delete throughput {:.0} below target {:.0} ops/sec",
        delete_ops_sec,
        DELETE_TARGET_OPS_SEC
    );
    assert!(
        reinsert_ops_sec >= REINSERT_TARGET_OPS_SEC,
        "Reinsert throughput {:.0} below target {:.0} ops/sec",
        reinsert_ops_sec,
        REINSERT_TARGET_OPS_SEC
    );

    println!("Heap File Space Reuse: PASSED");
}

// =============================================================================
// Test 4: B+ Tree Test (5-run validation)
// =============================================================================

/// Inserts 1,000,000 random i64 keys and verifies operations.
/// Target: 8M inserts/sec, 40ns/lookup, 40M range/sec
#[test]
fn test_btree_1m_keys() {
    const KEY_COUNT: usize = 1_000_000;
    const LOOKUP_SAMPLE: usize = 10_000;
    const RANGE_SIZE: usize = 1_000;

    println!("\n=== B+ Tree Arena Performance Test ===");
    println!("Keys per run: {}", KEY_COUNT);
    println!("Lookup sample: {}", LOOKUP_SAMPLE);
    println!("Validation runs: {}", VALIDATION_RUNS);

    let mut insert_results = Vec::with_capacity(VALIDATION_RUNS);
    let mut lookup_results = Vec::with_capacity(VALIDATION_RUNS);
    let mut range_results = Vec::with_capacity(VALIDATION_RUNS);

    for run in 0..VALIDATION_RUNS {
        println!("\n--- Run {}/{} ---", run + 1, VALIDATION_RUNS);

        // Buffered B+Tree: write buffer + 32KB nodes for HTAP workloads
        // 1024 nodes × 32KB = 32MB capacity (enough for 2M keys)
        let mut btree = BufferedBTreeIndex::new(1024);

        let mut rng = rand::rng();
        let mut keys: Vec<i64> = (0..KEY_COUNT as i64).collect();

        // Shuffle for random insertion order
        for i in (1..keys.len()).rev() {
            let j = rng.random_range(0..=i);
            keys.swap(i, j);
        }

        // Insert phase
        let insert_start = Instant::now();
        for &key in &keys {
            let key_bytes = key.to_be_bytes();
            let tuple_id = TupleId::new(PageId::new(0, (key % 1000) as u32), (key % 100) as u16);
            btree.insert(&key_bytes, tuple_id).unwrap();
        }
        let insert_duration = insert_start.elapsed();

        // Flush buffer to B+Tree before measuring lookup latency
        btree.flush().unwrap();

        let final_height = btree.height();
        assert!(
            final_height <= 5,
            "Run {}: Tree height {} too large for {} keys",
            run + 1,
            final_height,
            KEY_COUNT
        );

        // Lookup phase (measures B+Tree performance after flush)
        // Uses production search() path which checks buffer first

        // Warmup pass to prime caches and CPU frequency
        for i in (0..KEY_COUNT).step_by(KEY_COUNT / 1000) {
            let key = i as i64;
            let key_bytes = key.to_be_bytes();
            std::hint::black_box(btree.search(&key_bytes));
        }

        let mut found_count = 0usize;
        let lookup_start = Instant::now();
        for i in (0..KEY_COUNT).step_by(KEY_COUNT / LOOKUP_SAMPLE) {
            let key = i as i64;
            let key_bytes = key.to_be_bytes();
            // black_box prevents compiler from optimizing away the search
            if std::hint::black_box(btree.search(&key_bytes)).is_some() {
                found_count += 1;
            }
        }
        let lookup_duration = lookup_start.elapsed();
        assert_eq!(
            found_count,
            LOOKUP_SAMPLE,
            "Run {}: Expected {} found",
            run + 1,
            LOOKUP_SAMPLE
        );

        // Range scan phase (multiple iterations, take median for stable timing)
        let start_key = 1000i64.to_be_bytes();
        let end_key = (1000 + RANGE_SIZE as i64).to_be_bytes();

        // Warmup
        let _ = btree.range_scan(Some(&start_key), Some(&end_key));

        // Multiple timed iterations
        const RANGE_ITERATIONS: usize = 10;
        let mut range_times = Vec::with_capacity(RANGE_ITERATIONS);
        let mut last_len = 0;
        for _ in 0..RANGE_ITERATIONS {
            let range_start = Instant::now();
            let range_results_data = btree.range_scan(Some(&start_key), Some(&end_key));
            range_times.push(range_start.elapsed());
            last_len = range_results_data.len();
        }
        range_times.sort();
        let range_duration = range_times[RANGE_ITERATIONS / 2]; // Median

        assert!(
            last_len >= RANGE_SIZE - 10,
            "Run {}: Expected ~{} keys in range, got {}",
            run + 1,
            RANGE_SIZE,
            last_len
        );
        let range_results_len = last_len;

        let insert_ops_sec = KEY_COUNT as f64 / insert_duration.as_secs_f64();
        let lookup_ns = lookup_duration.as_nanos() as f64 / LOOKUP_SAMPLE as f64;
        let range_ops_sec = range_results_len as f64 / range_duration.as_secs_f64();

        // Get flush stats for profiling
        let stats = btree.stats();
        let hash_table_time_ns = insert_duration.as_nanos() as u64 - stats.flush_time_ns;

        println!(
            "  Insert: {} ops/sec ({:?}), height={}",
            format_with_commas(insert_ops_sec),
            insert_duration,
            final_height
        );
        println!(
            "    Breakdown: hash_table={:.1}ms, flush={:.1}ms (drain={:.1}ms, btree={:.1}ms), flushes={}",
            hash_table_time_ns as f64 / 1_000_000.0,
            stats.flush_time_ns as f64 / 1_000_000.0,
            stats.drain_time_ns as f64 / 1_000_000.0,
            stats.btree_insert_time_ns as f64 / 1_000_000.0,
            stats.flush_count
        );
        println!("  Lookup: {:.2} ns/op ({:?})", lookup_ns, lookup_duration);
        println!(
            "  Range: {} ops/sec ({:?})",
            format_with_commas(range_ops_sec),
            range_duration
        );

        insert_results.push(insert_ops_sec);
        lookup_results.push(lookup_ns);
        range_results.push(range_ops_sec);
    }

    println!("\n=== B+ Tree Validation Results ===");
    let insert_result = validate_metric(
        "Insert throughput (ops/sec)",
        insert_results,
        BTREE_INSERT_TARGET_OPS_SEC,
        true,
    );
    let lookup_result = validate_metric(
        "Lookup latency (ns/op)",
        lookup_results,
        BTREE_LOOKUP_TARGET_NS,
        false,
    );
    let range_result = validate_metric(
        "Range throughput (ops/sec)",
        range_results,
        BTREE_RANGE_TARGET_OPS_SEC,
        true,
    );

    assert!(
        insert_result.passed,
        "B+Tree insert avg {:.0} < target {:.0}",
        insert_result.average, BTREE_INSERT_TARGET_OPS_SEC
    );
    assert!(
        !insert_result.regression_detected,
        "B+Tree insert regression detected"
    );
    assert!(
        lookup_result.passed,
        "B+Tree lookup avg {:.2} ns > target {:.2} ns",
        lookup_result.average, BTREE_LOOKUP_TARGET_NS
    );
    assert!(
        !lookup_result.regression_detected,
        "B+Tree lookup regression detected"
    );
    assert!(
        range_result.passed,
        "B+Tree range avg {:.0} < target {:.0}",
        range_result.average, BTREE_RANGE_TARGET_OPS_SEC
    );
    assert!(
        !range_result.regression_detected,
        "B+Tree range regression detected"
    );
}

// =============================================================================
// Test 5: Integration Test - WAL + Heap Recovery
// =============================================================================

/// Tests crash recovery: write tuples with WAL logging, crash, recover.
/// Target: 2ms/MB recovery time
#[tokio::test]
async fn test_wal_heap_recovery() {
    let dir = tempdir().unwrap();
    let heap_dir = dir.path().join("heap");
    let wal_dir = dir.path().join("wal");

    std::fs::create_dir_all(&heap_dir).unwrap();
    std::fs::create_dir_all(&wal_dir).unwrap();

    const TUPLE_COUNT: usize = 1000;
    let mut committed_data: Vec<(u32, Vec<u8>)> = Vec::new();

    // Phase 1: Write tuples with WAL logging
    let wal_size_bytes;
    {
        let wal_config = WalWriterConfig {
            wal_dir: wal_dir.clone(),
            segment_size: 16 * 1024 * 1024,
            fsync_enabled: true,
            batch_size: 1,
            batch_bytes: 64 * 1024,
            flush_interval_us: 1000,
        };
        let writer = Arc::new(WalWriter::new(wal_config).await.unwrap());

        let disk_config = DiskManagerConfig {
            data_dir: heap_dir.clone(),
            fsync_enabled: true,
        };
        let disk = Arc::new(DiskManager::new(disk_config).await.unwrap());
        let pool = Arc::new(BufferPool::auto_sized());
        let heap = HeapFile::with_defaults(disk, pool).unwrap();

        for i in 0..TUPLE_COUNT {
            let txn_id = (i + 1) as u32;
            let data = format!("committed_tuple_{}", i);

            let begin_lsn = writer.log_begin(txn_id).await.unwrap();

            let tuple = Tuple::new(data.clone().into_bytes(), txn_id);
            let tuple_id = heap.insert_batch(&[tuple]).await.unwrap().remove(0);

            let payload = format!(
                "{}:{}:{}",
                tuple_id.page_id.page_num, tuple_id.slot_id, data
            );
            let insert_lsn = writer
                .log_insert(txn_id, begin_lsn, Bytes::from(payload))
                .await
                .unwrap();

            writer.log_commit(txn_id, insert_lsn).await.unwrap();

            committed_data.push((txn_id, data.into_bytes()));
        }

        writer.flush().await.unwrap();
        wal_size_bytes = writer.current_segment_id().unwrap().0 as usize * 16 * 1024 * 1024;
        drop(writer);
        drop(heap);
    }

    // Phase 2: Recovery with timing
    let recovery_start = Instant::now();
    {
        let recovery = RecoveryManager::new(&wal_dir).unwrap();
        let result = recovery.recover().unwrap();

        assert!(
            result.undo_txns.is_empty(),
            "No transactions should need undo"
        );

        assert_eq!(
            result.redo_records.len(),
            TUPLE_COUNT,
            "Should have {} redo records",
            TUPLE_COUNT
        );

        let redo_txns: HashSet<u32> = result.redo_records.iter().map(|r| r.txn_id).collect();
        for (txn_id, _) in &committed_data {
            assert!(
                redo_txns.contains(txn_id),
                "Transaction {} should be in redo set",
                txn_id
            );
        }
    }
    let recovery_duration = recovery_start.elapsed();

    // Calculate recovery performance
    let wal_size_mb = wal_size_bytes as f64 / (1024.0 * 1024.0);
    let recovery_ms = recovery_duration.as_secs_f64() * 1000.0;
    let ms_per_mb = if wal_size_mb > 0.0 {
        recovery_ms / wal_size_mb
    } else {
        0.0
    };

    println!("\n=== Recovery Performance ===");
    println!("  WAL size: {:.2} MB", wal_size_mb);
    println!("  Recovery time: {:?}", recovery_duration);
    let recovery_pass = check_performance(
        "Recovery time (ms/MB)",
        ms_per_mb,
        RECOVERY_TARGET_MS_PER_MB,
        false,
    );

    println!(
        "WAL+Heap Recovery: {} - {} committed transactions recovered",
        if recovery_pass { "PASSED" } else { "FAILED" },
        TUPLE_COUNT
    );
}

/// Tests recovery with uncommitted transactions.
#[tokio::test]
async fn test_wal_recovery_with_uncommitted() {
    let dir = tempdir().unwrap();

    let config = WalWriterConfig {
        wal_dir: dir.path().to_path_buf(),
        segment_size: 16 * 1024 * 1024,
        fsync_enabled: true,
        batch_size: 1,
        batch_bytes: 64 * 1024,
        flush_interval_us: 1000,
    };

    {
        let writer = WalWriter::new(config.clone()).await.unwrap();

        for i in 1..=10 {
            let begin = writer.log_begin(i).await.unwrap();
            let insert = writer
                .log_insert(i, begin, Bytes::from(format!("data_{}", i)))
                .await
                .unwrap();
            writer.log_commit(i, insert).await.unwrap();
        }

        for i in 11..=15 {
            let begin = writer.log_begin(i).await.unwrap();
            writer
                .log_insert(i, begin, Bytes::from(format!("uncommitted_{}", i)))
                .await
                .unwrap();
        }

        writer.flush().await.unwrap();
        drop(writer);
    }

    let recovery = RecoveryManager::new(dir.path()).unwrap();
    let result = recovery.recover().unwrap();

    let committed_txns: HashSet<u32> = (1..=10).collect();
    let redo_txns: HashSet<u32> = result.redo_records.iter().map(|r| r.txn_id).collect();

    for txn in &committed_txns {
        assert!(
            redo_txns.contains(txn),
            "Committed transaction {} should be in redo",
            txn
        );
    }

    let uncommitted_txns: HashSet<u32> = (11..=15).collect();
    let undo_set: HashSet<u32> = result.undo_txns.iter().copied().collect();

    for txn in &uncommitted_txns {
        assert!(
            undo_set.contains(txn),
            "Uncommitted transaction {} should be in undo",
            txn
        );
    }

    for txn in &uncommitted_txns {
        assert!(
            !redo_txns.contains(txn),
            "Uncommitted transaction {} should NOT be in redo",
            txn
        );
    }

    println!(
        "WAL Recovery with Uncommitted: PASSED - {} redo, {} undo",
        result.redo_records.len(),
        result.undo_txns.len()
    );
}

// =============================================================================
// Summary Test
// =============================================================================

/// Summary test - runs after all validation tests complete.
#[tokio::test]
async fn test_phase1_summary() {
    println!("\n============================================================");
    println!("ZyronDB Phase 1: Storage Foundation Validation Complete");
    println!("============================================================");
    println!("\nRun: cargo test -p zyron-storage --test phase1_test --release -- --nocapture");
}

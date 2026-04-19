#![allow(non_snake_case, unused_assignments)]

//! Spatial Index (R-tree) Benchmark Suite
//!
//! Measures the zyron-types R-tree for:
//!
//! - STR bulk load throughput at 100K and 1M points (parallel path)
//! - Dynamic insert throughput
//! - KNN query latency on a 100K-point tree
//! - Range query latency on a 100K-point tree
//! - ST_DWithin (point-radius) candidate retrieval
//! - delete_by_data throughput (exercises the O(1) inverse map)
//! - Snapshot save + load roundtrip wall time
//! - Concurrent reader throughput during a single writer
//! - Small-fanout stress (forces many splits)
//!
//! Each perf test runs `VALIDATION_RUNS` iterations and `validate_metric`
//! writes JSON/TXT output under `benchmarks/spatial_index/`.
//!
//! Run: cargo test -p zyron-types --test spatial_index_bench --release -- --nocapture

use std::sync::Arc;
use std::sync::Mutex;
use std::time::Instant;

use zyron_bench_harness::*;
use zyron_types::spatial_index::{LeafEntry, Mbr, RTree, RTreeConfig};

// Performance targets
const BULK_LOAD_100K_TARGET_POINTS_SEC: f64 = 1_500_000.0;
const BULK_LOAD_1M_TARGET_POINTS_SEC: f64 = 2_000_000.0;
const INSERT_TARGET_POINTS_SEC: f64 = 300_000.0;
const KNN_100K_TARGET_US: f64 = 500.0;
const RANGE_100K_TARGET_US: f64 = 1_500.0;
const DWITHIN_100K_TARGET_US: f64 = 1_500.0;
const DELETE_BY_DATA_TARGET_PER_SEC: f64 = 500_000.0;
const SNAPSHOT_ROUNDTRIP_100K_TARGET_MS: f64 = 1_000.0;

static BENCHMARK_LOCK: Mutex<()> = Mutex::new(());

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

/// Deterministic pseudo-random point grid in [0, 10000) x [0, 10000).
fn make_entries(n: u64, seed: u64) -> Vec<LeafEntry<u64>> {
    let mut out = Vec::with_capacity(n as usize);
    let mut state = seed.wrapping_add(0x9e3779b97f4a7c15);
    for i in 0..n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let x = ((state >> 16) as u32 % 10_000) as f64;
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let y = ((state >> 16) as u32 % 10_000) as f64;
        out.push(LeafEntry {
            mbr: Mbr::point(&[x, y]),
            data: i,
            deleted: false,
        });
    }
    out
}

fn tempdir_like(name: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("{}_{}", name, std::process::id()));
    p
}

// -----------------------------------------------------------------------
// Test 1: Bulk-load throughput at 100K points
// -----------------------------------------------------------------------

#[test]
fn test_bulk_load_100k() {
    zyron_bench_harness::init("spatial_index");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Bulk load 100K points ===");
    const N: u64 = 100_000;

    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        let entries = make_entries(N, 1);
        let tree = RTree::<u64>::new(2);
        let start = Instant::now();
        tree.bulk_load(entries);
        let elapsed_ns = start.elapsed().as_nanos();
        assert_eq!(tree.len(), N);
        let pps = (N as f64) / ((elapsed_ns as f64) / 1e9);
        runs.push(pps);
        tprintln!(
            "  Run {}: {} points/sec ({:.1} ms build)",
            run + 1,
            format_with_commas(pps),
            (elapsed_ns as f64) / 1e6
        );
    }

    let _ = validate_metric(
        "Performance",
        "bulk_load 100K points/sec",
        runs,
        BULK_LOAD_100K_TARGET_POINTS_SEC,
        true,
    );
}

// -----------------------------------------------------------------------
// Test 2: Bulk-load throughput at 1M points (exercises parallel STR)
// -----------------------------------------------------------------------

#[test]
fn test_bulk_load_1m_parallel_path() {
    zyron_bench_harness::init("spatial_index");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Bulk load 1M points (parallel STR) ===");
    const N: u64 = 1_000_000;

    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        let entries = make_entries(N, 42);
        let tree = RTree::<u64>::new(2);
        let start = Instant::now();
        tree.bulk_load(entries);
        let elapsed_ns = start.elapsed().as_nanos();
        assert_eq!(tree.len(), N);
        let pps = (N as f64) / ((elapsed_ns as f64) / 1e9);
        runs.push(pps);
        tprintln!(
            "  Run {}: {} points/sec ({:.2} s build)",
            run + 1,
            format_with_commas(pps),
            (elapsed_ns as f64) / 1e9
        );
    }

    let _ = validate_metric(
        "Performance",
        "bulk_load 1M points/sec (parallel STR)",
        runs,
        BULK_LOAD_1M_TARGET_POINTS_SEC,
        true,
    );
}

// -----------------------------------------------------------------------
// Test 3: Dynamic insert throughput
// -----------------------------------------------------------------------

#[test]
fn test_dynamic_insert_throughput() {
    zyron_bench_harness::init("spatial_index");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Dynamic insert throughput (50K points) ===");
    const N: u64 = 50_000;

    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        let entries = make_entries(N, (run as u64) + 1);
        let tree = RTree::<u64>::new(2);
        let start = Instant::now();
        for e in entries {
            tree.insert(e);
        }
        let elapsed_ns = start.elapsed().as_nanos();
        assert_eq!(tree.len(), N);
        let pps = (N as f64) / ((elapsed_ns as f64) / 1e9);
        runs.push(pps);
        tprintln!(
            "  Run {}: {} points/sec ({:.2} s)",
            run + 1,
            format_with_commas(pps),
            (elapsed_ns as f64) / 1e9
        );
    }

    let _ = validate_metric(
        "Performance",
        "insert points/sec",
        runs,
        INSERT_TARGET_POINTS_SEC,
        true,
    );
}

// -----------------------------------------------------------------------
// Test 4: KNN latency on a 100K-point tree
// -----------------------------------------------------------------------

#[test]
fn test_knn_latency_100k() {
    zyron_bench_harness::init("spatial_index");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== KNN latency on 100K-point tree (k=10) ===");
    const N: u64 = 100_000;

    let tree = RTree::<u64>::new(2);
    tree.bulk_load(make_entries(N, 7));
    assert_eq!(tree.len(), N);

    let query: Vec<f64> = vec![5000.0, 5000.0];
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        const ITERS: u64 = 10_000;
        let start = Instant::now();
        let mut acc: u64 = 0;
        for _ in 0..ITERS {
            let hits = tree.knn(&query, 10);
            acc = acc.wrapping_add(hits.len() as u64);
        }
        let elapsed_ns = start.elapsed().as_nanos();
        std::hint::black_box(acc);
        let us_per = (elapsed_ns as f64) / 1000.0 / (ITERS as f64);
        runs.push(us_per);
        tprintln!("  Run {}: {:.2} us/query", run + 1, us_per);
    }

    let _ = validate_metric(
        "Performance",
        "KNN 100K k=10 (us)",
        runs,
        KNN_100K_TARGET_US,
        false,
    );
}

// -----------------------------------------------------------------------
// Test 5: Range query latency on 100K tree
// -----------------------------------------------------------------------

#[test]
fn test_range_query_latency_100k() {
    zyron_bench_harness::init("spatial_index");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Range query latency on 100K-point tree (1% area) ===");
    const N: u64 = 100_000;

    let tree = RTree::<u64>::new(2);
    tree.bulk_load(make_entries(N, 13));

    // 1% of the total 10000x10000 area -> 1000x1000 window returns ~1000 points.
    let query = Mbr::from_extents(&[4500.0, 4500.0], &[5500.0, 5500.0]);
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        const ITERS: u64 = 5_000;
        let start = Instant::now();
        let mut acc: u64 = 0;
        for _ in 0..ITERS {
            let hits = tree.range(&query);
            acc = acc.wrapping_add(hits.len() as u64);
        }
        let elapsed_ns = start.elapsed().as_nanos();
        std::hint::black_box(acc);
        let us_per = (elapsed_ns as f64) / 1000.0 / (ITERS as f64);
        runs.push(us_per);
        tprintln!("  Run {}: {:.2} us/query", run + 1, us_per);
    }

    let _ = validate_metric(
        "Performance",
        "range 100K 1% area (us)",
        runs,
        RANGE_100K_TARGET_US,
        false,
    );
}

// -----------------------------------------------------------------------
// Test 6: ST_DWithin latency on 100K tree
// -----------------------------------------------------------------------

#[test]
fn test_dwithin_latency_100k() {
    zyron_bench_harness::init("spatial_index");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== DWithin latency on 100K-point tree (radius=500) ===");
    const N: u64 = 100_000;

    let tree = RTree::<u64>::new(2);
    tree.bulk_load(make_entries(N, 21));

    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        const ITERS: u64 = 5_000;
        let start = Instant::now();
        let mut acc: u64 = 0;
        for _ in 0..ITERS {
            let hits = tree.dwithin(&[5000.0, 5000.0], 500.0);
            acc = acc.wrapping_add(hits.len() as u64);
        }
        let elapsed_ns = start.elapsed().as_nanos();
        std::hint::black_box(acc);
        let us_per = (elapsed_ns as f64) / 1000.0 / (ITERS as f64);
        runs.push(us_per);
        tprintln!("  Run {}: {:.2} us/query", run + 1, us_per);
    }

    let _ = validate_metric(
        "Performance",
        "dwithin 100K r=500 (us)",
        runs,
        DWITHIN_100K_TARGET_US,
        false,
    );
}

// -----------------------------------------------------------------------
// Test 7: delete_by_data throughput (inverse map path)
// -----------------------------------------------------------------------

#[test]
fn test_delete_by_data_throughput() {
    zyron_bench_harness::init("spatial_index");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== delete_by_data throughput (10K deletes on 100K tree) ===");
    const N: u64 = 100_000;
    const DEL: u64 = 10_000;

    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        let tree = RTree::<u64>::new(2);
        tree.bulk_load(make_entries(N, 99));
        let start = Instant::now();
        for id in 0..DEL {
            tree.delete_by_data(&id);
        }
        let elapsed_ns = start.elapsed().as_nanos();
        let dps = (DEL as f64) / ((elapsed_ns as f64) / 1e9);
        runs.push(dps);
        tprintln!(
            "  Run {}: {} deletes/sec ({:.2} ms)",
            run + 1,
            format_with_commas(dps),
            (elapsed_ns as f64) / 1e6
        );
    }

    let _ = validate_metric(
        "Performance",
        "delete_by_data/sec",
        runs,
        DELETE_BY_DATA_TARGET_PER_SEC,
        true,
    );
}

// -----------------------------------------------------------------------
// Test 8: Snapshot save + load roundtrip
// -----------------------------------------------------------------------

#[test]
fn test_snapshot_roundtrip_100k() {
    zyron_bench_harness::init("spatial_index");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Snapshot save+load roundtrip (100K points) ===");
    const N: u64 = 100_000;

    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        let tree = RTree::<u64>::new(2);
        tree.bulk_load(make_entries(N, (run as u64) + 3));

        let tmp = tempdir_like(&format!("spatial_snapshot_bench_{}", run));
        let start = Instant::now();
        tree.save_to(&tmp).unwrap();
        let save_ns = start.elapsed().as_nanos();

        let start = Instant::now();
        let loaded = RTree::<u64>::load_from(&tmp).unwrap();
        let load_ns = start.elapsed().as_nanos();
        assert_eq!(loaded.len(), N);

        let total_ms = ((save_ns + load_ns) as f64) / 1e6;
        runs.push(total_ms);
        tprintln!(
            "  Run {}: save {:.1} ms, load {:.1} ms, total {:.1} ms",
            run + 1,
            (save_ns as f64) / 1e6,
            (load_ns as f64) / 1e6,
            total_ms
        );

        let _ = std::fs::remove_file(&tmp);
    }

    let _ = validate_metric(
        "Performance",
        "snapshot roundtrip 100K (ms)",
        runs,
        SNAPSHOT_ROUNDTRIP_100K_TARGET_MS,
        false,
    );
}

// -----------------------------------------------------------------------
// Test 9: Concurrent reader throughput during single writer
// -----------------------------------------------------------------------

#[test]
fn test_concurrent_reader_throughput() {
    zyron_bench_harness::init("spatial_index");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Concurrent 4-reader throughput during 1 writer (50K inserts) ===");

    let tree = Arc::new(RTree::<u64>::new(2));
    for e in make_entries(50_000, 0) {
        tree.insert(e);
    }

    let start = Instant::now();
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let reader_count = Arc::new(std::sync::atomic::AtomicU64::new(0));

    let mut reader_handles = Vec::new();
    for _ in 0..4 {
        let t = tree.clone();
        let s = stop.clone();
        let c = reader_count.clone();
        reader_handles.push(std::thread::spawn(move || {
            while !s.load(std::sync::atomic::Ordering::Relaxed) {
                let _ = t.range(&Mbr::from_extents(&[0.0, 0.0], &[5000.0, 5000.0]));
                c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }));
    }

    let writer = {
        let t = tree.clone();
        std::thread::spawn(move || {
            for e in make_entries(50_000, 99) {
                let mut e = e;
                e.data += 1_000_000;
                t.insert(e);
            }
        })
    };
    writer.join().unwrap();

    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    for h in reader_handles {
        h.join().unwrap();
    }

    let elapsed_sec = start.elapsed().as_secs_f64();
    let queries = reader_count.load(std::sync::atomic::Ordering::Relaxed);
    let qps = (queries as f64) / elapsed_sec;
    tprintln!(
        "  Reader queries completed during writer: {} in {:.2} s -> {} q/s",
        format_with_commas(queries as f64),
        elapsed_sec,
        format_with_commas(qps)
    );
    assert_eq!(tree.len(), 100_000);
}

// -----------------------------------------------------------------------
// Test 10: Small-fanout stress (max_fill=8, forces many splits)
// -----------------------------------------------------------------------

#[test]
fn test_small_fanout_stress() {
    zyron_bench_harness::init("spatial_index");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Small-fanout stress (max_fill=8, 50K inserts) ===");
    let tree = RTree::<u64>::with_config(
        2,
        RTreeConfig {
            min_fill: 3,
            max_fill: 8,
            srid: 4326,
        },
    );
    let entries = make_entries(50_000, 17);
    let start = Instant::now();
    for e in entries {
        tree.insert(e);
    }
    let elapsed_ms = (start.elapsed().as_nanos() as f64) / 1e6;
    let snap = tree.metrics().snapshot();
    let stats = tree.stats();
    tprintln!(
        "  50K inserts in {:.1} ms (height={}, splits={})",
        elapsed_ms,
        stats.height,
        snap.splits
    );
    assert_eq!(tree.len(), 50_000);
    assert!(snap.splits > 0, "small-fanout run produced no splits");
}

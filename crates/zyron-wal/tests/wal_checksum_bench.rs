//! Latency micro-benchmarks for the WAL record checksum primitives.
//!
//! Measures wal_checksum (one-shot read path), WalHasher (incremental write
//! path) and data_checksum at record sizes the WAL actually produces. These
//! numbers are the acceptance gate for consolidating the checksum into
//! zyron-common: post-consolidation p50/p90/p99 must hold within 1% of the
//! recorded baseline

use std::sync::Mutex;
use zyron_bench_harness::*;
use zyron_wal::WalHasher;

// Serialize benchmarks to avoid CPU contention between tests
static BENCHMARK_LOCK: Mutex<()> = Mutex::new(());

/// Batches per run. Percentiles are taken over per-batch means, since a
/// single sub-100ns call cannot be timed individually
const BATCHES: usize = 256;

/// Calls per batch. Long enough that one batch is comfortably above timer
/// resolution, short enough that 256 batches finish quickly
const CALLS_PER_BATCH: usize = 512;

// Targets set from measured pre-consolidation latencies, they catch gross
// regressions on the WAL checksum hot path
const WAL_CHECKSUM_64B_P50_NS: f64 = 25.0;
const WAL_CHECKSUM_280B_P50_NS: f64 = 60.0;
const WAL_CHECKSUM_8K_P50_NS: f64 = 900.0;
const WAL_HASHER_280B_P50_NS: f64 = 60.0;
const DATA_CHECKSUM_64B_P50_NS: f64 = 25.0;

/// Fills a buffer with a deterministic byte pattern
fn make_record(total_len: usize) -> Vec<u8> {
    (0..total_len).map(|i| (i * 37 + 13) as u8).collect()
}

/// Times `f` in batches and returns (p50, p90, p99) of per-call nanoseconds
fn measure_percentiles(mut f: impl FnMut()) -> (f64, f64, f64) {
    // Warmup pass so the first batch is not paying cold-cache costs
    for _ in 0..CALLS_PER_BATCH {
        f();
    }
    let mut samples = Vec::with_capacity(BATCHES);
    for _ in 0..BATCHES {
        let start = Instant::now();
        for _ in 0..CALLS_PER_BATCH {
            f();
        }
        let elapsed = start.elapsed().as_nanos() as f64;
        samples.push(elapsed / CALLS_PER_BATCH as f64);
    }
    samples.sort_by(|a, b| a.total_cmp(b));
    let pick = |p: f64| samples[((samples.len() as f64 * p) as usize).min(samples.len() - 1)];
    (pick(0.50), pick(0.90), pick(0.99))
}

/// Runs `measure_percentiles` VALIDATION_RUNS times and validates each
/// percentile as its own metric under `test`
fn bench_percentiles(test: &str, label: &str, p50_target: f64, mut f: impl FnMut()) {
    let mut p50s = Vec::with_capacity(VALIDATION_RUNS);
    let mut p90s = Vec::with_capacity(VALIDATION_RUNS);
    let mut p99s = Vec::with_capacity(VALIDATION_RUNS);
    for _ in 0..VALIDATION_RUNS {
        let (p50, p90, p99) = measure_percentiles(&mut f);
        p50s.push(p50);
        p90s.push(p90);
        p99s.push(p99);
    }
    validate_metric(test, &format!("{label} p50 (ns)"), p50s, p50_target, false);
    // p90/p99 carry looser targets, they exist to expose tail movement
    validate_metric(
        test,
        &format!("{label} p90 (ns)"),
        p90s,
        p50_target * 2.0,
        false,
    );
    validate_metric(
        test,
        &format!("{label} p99 (ns)"),
        p99s,
        p50_target * 4.0,
        false,
    );
}

#[test]
fn test_bench_wal_checksum_one_shot() {
    zyron_bench_harness::init("wal_checksum");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Benchmark: wal_checksum one-shot ===");
    calibrate();
    let before = take_util_snapshot();

    // 24-byte header + 40-byte payload, a small OLTP insert record
    let small = make_record(64);
    bench_percentiles(
        "WAL Checksum One-Shot",
        "wal_checksum 64B",
        WAL_CHECKSUM_64B_P50_NS,
        || {
            std::hint::black_box(zyron_wal::wal_checksum(std::hint::black_box(&small), 24));
        },
    );

    // 24-byte header + 256-byte payload, a mid-size row
    let medium = make_record(280);
    bench_percentiles(
        "WAL Checksum One-Shot",
        "wal_checksum 280B",
        WAL_CHECKSUM_280B_P50_NS,
        || {
            std::hint::black_box(zyron_wal::wal_checksum(std::hint::black_box(&medium), 24));
        },
    );

    // 24-byte header + 8KB payload, a full-page-sized record
    let large = make_record(24 + 8192);
    bench_percentiles(
        "WAL Checksum One-Shot",
        "wal_checksum 8KB",
        WAL_CHECKSUM_8K_P50_NS,
        || {
            std::hint::black_box(zyron_wal::wal_checksum(std::hint::black_box(&large), 24));
        },
    );

    record_test_util("WAL Checksum One-Shot", before, take_util_snapshot());
}

#[test]
fn test_bench_wal_hasher_incremental() {
    zyron_bench_harness::init("wal_checksum");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Benchmark: WalHasher incremental ===");
    calibrate();
    let before = take_util_snapshot();

    let payload = make_record(256);
    bench_percentiles(
        "WAL Hasher Incremental",
        "WalHasher 280B",
        WAL_HASHER_280B_P50_NS,
        || {
            let mut hasher = WalHasher::new(24 + payload.len());
            hasher.write_header_fields(
                std::hint::black_box(0x0000000100000040u64),
                0x0000000100000000u64,
                42,
                10,
                0,
                payload.len() as u16,
            );
            hasher.write_payload(std::hint::black_box(&payload));
            std::hint::black_box(hasher.finish());
        },
    );

    record_test_util("WAL Hasher Incremental", before, take_util_snapshot());
}

#[test]
fn test_bench_data_checksum() {
    zyron_bench_harness::init("wal_checksum");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Benchmark: data_checksum ===");
    calibrate();
    let before = take_util_snapshot();

    let data = make_record(64);
    bench_percentiles(
        "Data Checksum",
        "data_checksum 64B",
        DATA_CHECKSUM_64B_P50_NS,
        || {
            std::hint::black_box(zyron_wal::data_checksum(std::hint::black_box(&data)));
        },
    );

    record_test_util("Data Checksum", before, take_util_snapshot());
}

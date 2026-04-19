#![allow(non_snake_case, unused_assignments)]

//! Central Checksum/Hash Benchmark Suite
//!
//! Measures the zyron-common hash primitive (scalar + x86 AES-NI + ARM AES
//! dispatched at runtime). Covers:
//!
//! - One-shot throughput at multiple input sizes (16 B up to 64 MB)
//! - Small-input fast-path latency (<=32 B)
//! - Seeded-variant overhead vs unseeded
//! - Streaming `Hasher` update throughput (incremental)
//! - Cross-tier consistency (scalar vs HW produces identical output)
//! - Dispatch overhead (tier-select is cached, should be negligible after
//!   the first call)
//!
//! Each perf test runs `VALIDATION_RUNS` iterations and `validate_metric`
//! averages + writes JSON/TXT output under `benchmarks/checksum/`.
//!
//! Run: cargo test -p zyron-common --test checksum_bench --release -- --nocapture

use std::sync::Mutex;
use std::time::Instant;

use zyron_bench_harness::*;
use zyron_common::{
    Hasher, checksum, hash32, hash32_seeded, hash64, hash64_seeded, hash128, hash128_seeded,
};

// -----------------------------------------------------------------------
// Performance targets. Values are per-byte throughput for the chosen
// input size bucket, or per-call latency for small inputs / small ops.
// Minimum targets; faster is better. These reflect realistic numbers for
// an x86_64 AES-NI machine; Tier dispatch will silently fall back to
// scalar on unsupported hardware.
// -----------------------------------------------------------------------

const SMALL_HASH32_TARGET_NS: f64 = 120.0; // <=32B small-input fast path
const ONE_KB_HASH32_TARGET_MB_SEC: f64 = 3_000.0;
const ONE_MB_HASH32_TARGET_MB_SEC: f64 = 6_000.0;
const SIXTEEN_MB_HASH32_TARGET_MB_SEC: f64 = 6_000.0;
const STREAMING_1MB_TARGET_MB_SEC: f64 = 5_000.0;
const SEEDED_OVERHEAD_MAX_PCT: f64 = 20.0; // seeded hash within 20% of unseeded

static BENCHMARK_LOCK: Mutex<()> = Mutex::new(());

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

/// Generates `n` pseudo-random bytes. Deterministic across runs.
fn make_data(n: usize, seed: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(n);
    let mut state = seed.wrapping_add(0x9e3779b97f4a7c15);
    while out.len() < n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bytes = state.to_le_bytes();
        let take = (n - out.len()).min(8);
        out.extend_from_slice(&bytes[..take]);
    }
    out
}

fn mb_per_sec(bytes: u64, elapsed_ns: u128) -> f64 {
    let mb = (bytes as f64) / (1024.0 * 1024.0);
    let sec = (elapsed_ns as f64) / 1e9;
    mb / sec
}

// -----------------------------------------------------------------------
// Test 1: Small-input latency (<=32 bytes)
// -----------------------------------------------------------------------

#[test]
fn test_hash32_small_input_latency() {
    zyron_bench_harness::init("checksum");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== hash32 small-input (<=32 B) latency ===");
    tprintln!("  Active tier: {}", checksum::active_tier().name());

    let inputs: Vec<Vec<u8>> = [0usize, 1, 4, 8, 16, 24, 32]
        .iter()
        .map(|&n| make_data(n, n as u64))
        .collect();

    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        const ITERS: u64 = 10_000_000;
        let start = Instant::now();
        let mut acc: u64 = 0;
        for i in 0..ITERS {
            let idx = (i as usize) % inputs.len();
            acc = acc.wrapping_add(hash32(&inputs[idx]) as u64);
        }
        let elapsed_ns = start.elapsed().as_nanos();
        std::hint::black_box(acc);
        let ns_per = (elapsed_ns as f64) / (ITERS as f64);
        runs.push(ns_per);
        tprintln!("  Run {}: {:.1} ns/call", run + 1, ns_per);
    }

    let _ = validate_metric(
        "Performance",
        "hash32 <=32B latency (ns)",
        runs,
        SMALL_HASH32_TARGET_NS,
        false,
    );
}

// -----------------------------------------------------------------------
// Test 2: Throughput at 1 KB
// -----------------------------------------------------------------------

#[test]
fn test_hash32_throughput_1kb() {
    zyron_bench_harness::init("checksum");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== hash32 throughput at 1 KB ===");
    tprintln!("  Active tier: {}", checksum::active_tier().name());

    let data = make_data(1024, 0xabcd);
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        const ITERS: u64 = 2_000_000;
        let start = Instant::now();
        let mut acc: u64 = 0;
        for _ in 0..ITERS {
            acc = acc.wrapping_add(hash32(&data) as u64);
        }
        let elapsed_ns = start.elapsed().as_nanos();
        std::hint::black_box(acc);
        let total_bytes = ITERS * data.len() as u64;
        let mbps = mb_per_sec(total_bytes, elapsed_ns);
        runs.push(mbps);
        tprintln!("  Run {}: {:.1} MB/s", run + 1, mbps);
    }

    let _ = validate_metric(
        "Performance",
        "hash32 1KB throughput (MB/s)",
        runs,
        ONE_KB_HASH32_TARGET_MB_SEC,
        true,
    );
}

// -----------------------------------------------------------------------
// Test 3: Throughput at 1 MB
// -----------------------------------------------------------------------

#[test]
fn test_hash32_throughput_1mb() {
    zyron_bench_harness::init("checksum");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== hash32 throughput at 1 MB ===");
    tprintln!("  Active tier: {}", checksum::active_tier().name());

    let data = make_data(1024 * 1024, 0xdeadbeef);
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        const ITERS: u64 = 2_000;
        let start = Instant::now();
        let mut acc: u64 = 0;
        for _ in 0..ITERS {
            acc = acc.wrapping_add(hash32(&data) as u64);
        }
        let elapsed_ns = start.elapsed().as_nanos();
        std::hint::black_box(acc);
        let total_bytes = ITERS * data.len() as u64;
        let mbps = mb_per_sec(total_bytes, elapsed_ns);
        runs.push(mbps);
        tprintln!("  Run {}: {:.1} MB/s", run + 1, mbps);
    }

    let _ = validate_metric(
        "Performance",
        "hash32 1MB throughput (MB/s)",
        runs,
        ONE_MB_HASH32_TARGET_MB_SEC,
        true,
    );
}

// -----------------------------------------------------------------------
// Test 4: Throughput at 16 MB (file-sized checksum)
// -----------------------------------------------------------------------

#[test]
fn test_hash32_throughput_16mb() {
    zyron_bench_harness::init("checksum");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== hash32 throughput at 16 MB ===");
    tprintln!("  Active tier: {}", checksum::active_tier().name());

    let data = make_data(16 * 1024 * 1024, 0xcafebabe);
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        const ITERS: u64 = 50;
        let start = Instant::now();
        let mut acc: u64 = 0;
        for _ in 0..ITERS {
            acc = acc.wrapping_add(hash32(&data) as u64);
        }
        let elapsed_ns = start.elapsed().as_nanos();
        std::hint::black_box(acc);
        let total_bytes = ITERS * data.len() as u64;
        let mbps = mb_per_sec(total_bytes, elapsed_ns);
        runs.push(mbps);
        tprintln!("  Run {}: {:.1} MB/s", run + 1, mbps);
    }

    let _ = validate_metric(
        "Performance",
        "hash32 16MB throughput (MB/s)",
        runs,
        SIXTEEN_MB_HASH32_TARGET_MB_SEC,
        true,
    );
}

// -----------------------------------------------------------------------
// Test 5: Streaming Hasher throughput
// -----------------------------------------------------------------------

#[test]
fn test_streaming_hasher_throughput_1mb() {
    zyron_bench_harness::init("checksum");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== streaming Hasher update+finish throughput at 1 MB ===");
    tprintln!("  Active tier: {}", checksum::active_tier().name());

    let data = make_data(1024 * 1024, 0x12345678);
    const CHUNK: usize = 16 * 1024;

    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        const ITERS: u64 = 2_000;
        let start = Instant::now();
        let mut acc: u64 = 0;
        for _ in 0..ITERS {
            let mut h = Hasher::new();
            for chunk in data.chunks(CHUNK) {
                h.update(chunk);
            }
            acc = acc.wrapping_add(h.finish32() as u64);
        }
        let elapsed_ns = start.elapsed().as_nanos();
        std::hint::black_box(acc);
        let total_bytes = ITERS * data.len() as u64;
        let mbps = mb_per_sec(total_bytes, elapsed_ns);
        runs.push(mbps);
        tprintln!("  Run {}: {:.1} MB/s (chunk={}B)", run + 1, mbps, CHUNK);
    }

    let _ = validate_metric(
        "Performance",
        "streaming 1MB throughput (MB/s)",
        runs,
        STREAMING_1MB_TARGET_MB_SEC,
        true,
    );
}

// -----------------------------------------------------------------------
// Test 6: Seeded-vs-unseeded overhead
// -----------------------------------------------------------------------

#[test]
fn test_seeded_overhead() {
    zyron_bench_harness::init("checksum");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Seeded variant overhead vs unseeded ===");

    let data = make_data(64 * 1024, 0xfeedface);
    const ITERS: u64 = 50_000;

    let mut unseeded_runs = Vec::with_capacity(VALIDATION_RUNS);
    let mut seeded_runs = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        let start = Instant::now();
        let mut acc: u64 = 0;
        for _ in 0..ITERS {
            acc = acc.wrapping_add(hash64(&data));
        }
        std::hint::black_box(acc);
        let un_ns = start.elapsed().as_nanos();

        let start = Instant::now();
        let mut acc: u64 = 0;
        for i in 0..ITERS {
            acc = acc.wrapping_add(hash64_seeded(&data, i));
        }
        std::hint::black_box(acc);
        let se_ns = start.elapsed().as_nanos();

        let un_mbps = mb_per_sec(ITERS * data.len() as u64, un_ns);
        let se_mbps = mb_per_sec(ITERS * data.len() as u64, se_ns);
        let overhead_pct = (un_mbps - se_mbps) / un_mbps * 100.0;
        unseeded_runs.push(un_mbps);
        seeded_runs.push(se_mbps);
        tprintln!(
            "  Run {}: unseeded {:.0} MB/s, seeded {:.0} MB/s (overhead {:.1}%)",
            run + 1,
            un_mbps,
            se_mbps,
            overhead_pct
        );
    }

    let mean_un = unseeded_runs.iter().sum::<f64>() / unseeded_runs.len() as f64;
    let mean_se = seeded_runs.iter().sum::<f64>() / seeded_runs.len() as f64;
    let overhead_pct = (mean_un - mean_se) / mean_un * 100.0;
    tprintln!(
        "  Average overhead: {:.1}% (target: <= {:.1}%)",
        overhead_pct,
        SEEDED_OVERHEAD_MAX_PCT
    );
    assert!(
        overhead_pct <= SEEDED_OVERHEAD_MAX_PCT,
        "seeded variant overhead {:.1}% exceeds target {:.1}%",
        overhead_pct,
        SEEDED_OVERHEAD_MAX_PCT
    );
}

// -----------------------------------------------------------------------
// Test 7: Cross-tier consistency (scalar vs active tier produce identical
// output across a range of inputs)
// -----------------------------------------------------------------------

#[test]
fn test_cross_tier_consistency() {
    zyron_bench_harness::init("checksum");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Cross-tier consistency (scalar reference vs active tier) ===");
    tprintln!("  Active tier: {}", checksum::active_tier().name());

    let sizes = [
        0usize, 1, 7, 16, 31, 32, 33, 128, 1024, 4096, 65_536, 1_048_576,
    ];
    let mut mismatches = 0u32;
    for &n in &sizes {
        let data = make_data(n, n as u64 + 7);
        let active = hash128(&data);
        let scalar = checksum::scalar::hash_scalar(&data, 0);
        if active != scalar {
            mismatches += 1;
            tprintln!("  MISMATCH at size {}", n);
        }
    }
    assert_eq!(
        mismatches,
        0,
        "active tier disagrees with scalar reference on {} of {} sizes",
        mismatches,
        sizes.len()
    );
    tprintln!("  All {} sizes match scalar reference: PASS", sizes.len());
}

// -----------------------------------------------------------------------
// Test 8: 64-bit and 128-bit output widths on hot hashmap-style inputs
// -----------------------------------------------------------------------

#[test]
fn test_hash64_hashmap_key_throughput() {
    zyron_bench_harness::init("checksum");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== hash64 (8-16 byte keys, hashmap pattern) ===");
    tprintln!("  Active tier: {}", checksum::active_tier().name());

    // Mix of typical DB key sizes: 8 bytes (rowid), 16 bytes (uuid), 24 bytes (composite).
    let keys: Vec<Vec<u8>> = (0..1024u64)
        .flat_map(|i| {
            [
                i.to_le_bytes().to_vec(),
                [i.to_le_bytes(), (i ^ 0x5555).to_le_bytes()]
                    .concat()
                    .to_vec(),
                [
                    i.to_le_bytes(),
                    (i ^ 0x5555).to_le_bytes(),
                    (i ^ 0xaaaa).to_le_bytes(),
                ]
                .concat()
                .to_vec(),
            ]
        })
        .collect();

    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        const ITERS: u64 = 5_000_000;
        let start = Instant::now();
        let mut acc: u64 = 0;
        for i in 0..ITERS {
            let k = &keys[(i as usize) % keys.len()];
            acc = acc.wrapping_add(hash64(k));
        }
        let elapsed_ns = start.elapsed().as_nanos();
        std::hint::black_box(acc);
        let ns_per = (elapsed_ns as f64) / (ITERS as f64);
        runs.push(ns_per);
        tprintln!("  Run {}: {:.1} ns/call", run + 1, ns_per);
    }

    // Not asserting a target here; we log so operators can spot regressions
    // against a later target once SMHasher-validated.
    tprintln!(
        "  Average: {:.1} ns/call",
        runs.iter().sum::<f64>() / runs.len() as f64
    );
}

#[test]
fn test_hash128_bloom_pattern() {
    zyron_bench_harness::init("checksum");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== hash128 (Bloom filter insert/probe pattern) ===");
    tprintln!("  Active tier: {}", checksum::active_tier().name());

    // Typical Bloom insert: hash once, extract two 64-bit halves, do k probes.
    let values: Vec<Vec<u8>> = (0..4096u64).map(|i| i.to_le_bytes().to_vec()).collect();

    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        const ITERS: u64 = 2_000_000;
        let start = Instant::now();
        let mut acc: u128 = 0;
        for i in 0..ITERS {
            let v = &values[(i as usize) % values.len()];
            acc = acc.wrapping_add(hash128(v));
        }
        let elapsed_ns = start.elapsed().as_nanos();
        std::hint::black_box(acc);
        let ns_per = (elapsed_ns as f64) / (ITERS as f64);
        runs.push(ns_per);
        tprintln!("  Run {}: {:.1} ns/call", run + 1, ns_per);
    }

    tprintln!(
        "  Average: {:.1} ns/call",
        runs.iter().sum::<f64>() / runs.len() as f64
    );
}

// -----------------------------------------------------------------------
// Test 9: Dispatch stability (tier doesn't change across calls)
// -----------------------------------------------------------------------

#[test]
fn test_dispatch_is_stable() {
    zyron_bench_harness::init("checksum");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Dispatch stability ===");
    let first = checksum::active_tier();
    for _ in 0..1_000 {
        assert_eq!(checksum::active_tier(), first);
    }
    tprintln!("  Tier stable across 1000 calls: {} PASS", first.name());
}

// -----------------------------------------------------------------------
// Seeded hash variants parity (hash128_seeded reachable on any tier)
// -----------------------------------------------------------------------

#[test]
fn test_seeded_variants_reach_all_widths() {
    zyron_bench_harness::init("checksum");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Seeded hash variants produce distinct outputs ===");
    let data = make_data(64, 0);
    let h32_a = hash32_seeded(&data, 1);
    let h32_b = hash32_seeded(&data, 2);
    assert_ne!(h32_a, h32_b);
    let h64_a = hash64_seeded(&data, 1);
    let h64_b = hash64_seeded(&data, 2);
    assert_ne!(h64_a, h64_b);
    let h128_a = hash128_seeded(&data, 1);
    let h128_b = hash128_seeded(&data, 2);
    assert_ne!(h128_a, h128_b);
    tprintln!("  Distinct seeds produce distinct outputs across all widths: PASS");
}

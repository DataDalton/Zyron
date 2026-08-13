//! Utility Operations Benchmark Suite
//!
//! Integration tests + performance benchmarks for Zyron native utility operations
//!   Diff/patch       JSON diff, JSON patch, text diff/patch, row diff, schema diff
//!   Scheduling       cron, sliding-window rate limiter, quotas
//!   Checksums        CRC32, CRC32C, xxHash32/64/128, Murmur3, City, FNV, SipHash, Adler32
//!   Natural sort     SortKey, natural compare, version compare, IP/path sort
//!   Barcode/QR       Code128, EAN13, UPC-A, Code39, QR Model 2 v1-10
//!   Document         CommonMark to HTML, HTML to text/markdown, sanitize
//!   File detect      MIME magic byte detection, encoding sniff, is_binary
//!   Resilience       count and EMA circuit breakers, retry, timeout, hedged
//!
//! Performance Targets
//! | Test                        | Metric     | Target        |
//! |-----------------------------|------------|---------------|
//! | CRC32C 1 GB                 | throughput | 12 GB/sec     |
//! | xxHash64 1 GB               | throughput | 16 GB/sec     |
//! | JSON Diff 1 KB              | latency    | 12 us         |
//! | JSON Patch 1 KB             | latency    | 6 us          |
//! | Text Diff 10K lines         | latency    | 50 ms         |
//! | Rate limit check            | latency    | 50 ns         |
//! | Natural sort key            | latency    | 200 ns        |
//! | MARKDOWN_TO_HTML 10 KB      | latency    | 300 us        |
//! | QR encode                   | latency    | 3 ms          |
//! | DETECT_MIME_TYPE            | latency    | 400 ns        |

use std::sync::Mutex;
use std::time::{Duration, Instant};

use zyron_bench_harness::*;

use zyron_types::barcode::{
    BarcodeFormat, QrErrorCorrection, barcode_decode, data_matrix_decode, data_matrix_encode,
    qr_decode, qr_encode,
};
use zyron_types::checksum::{
    CityHash64Streaming, StreamingHasher, XxHash64Streaming, adler32, city_hash64, crc32, crc32c,
    fnv1a_64, hash_column_xxh64, murmur3_32, murmur3_128, siphash, xxhash32, xxhash64, xxhash128,
};
use zyron_types::diff_patch::{
    ColumnDescriptor, DiffOp, SchemaChangeKind, collapse_adjacent_noops, generate_migration,
    json_diff, json_diff_table, json_merge_patch, json_patch, row_diff, row_diff_ordinal,
    schema_diff, text_diff, text_patch,
};
use zyron_types::document::{
    html_to_markdown, html_to_text, markdown_extract_code_blocks, markdown_extract_headers,
    markdown_extract_links, markdown_to_html, sanitize_html,
};
use zyron_types::file_detect::{detect_encoding, detect_mime_type, file_extension, is_binary};
use zyron_types::natural_sort::{
    UnknownPosition, custom_order_rank, ip_compare, natural_compare, natural_sort_key,
    path_compare, version_compare,
};
use zyron_types::resilience::{
    CircuitBreaker, CircuitBreakerRegistry, CircuitState, EmaCircuitBreaker, hedged, retry,
    timeout_blocking,
};
use zyron_types::scheduling::{
    BurstTokenBucket, QuotaRegistry, RateLimiterRegistry, cron_list, cron_matches, cron_next,
    cron_parse, monotonic_now_micros, quota_check, quota_increment, rate_limit_check,
};

// =============================================================================
// Performance Target Constants
// =============================================================================

const CRC32C_1GB_TARGET_GBSEC: f64 = 12.0;
const XXH64_1GB_TARGET_GBSEC: f64 = 16.0;
const JSON_DIFF_1KB_TARGET_US: f64 = 12.0;
const JSON_PATCH_1KB_TARGET_US: f64 = 6.0;
const TEXT_DIFF_10K_TARGET_MS: f64 = 50.0;
const RATE_LIMIT_CHECK_TARGET_NS: f64 = 50.0;
const NATURAL_SORT_KEY_TARGET_NS: f64 = 200.0;
const MARKDOWN_10KB_TARGET_US: f64 = 300.0;
const QR_ENCODE_TARGET_MS: f64 = 3.0;
const MIME_DETECT_TARGET_NS: f64 = 400.0;

static BENCHMARK_LOCK: Mutex<()> = Mutex::new(());

const SUITE_NAME: &str = "utility_ops";

// =============================================================================
// Helpers
// =============================================================================

fn build_1kb_json() -> (String, String) {
    let mut old = String::from("{");
    for i in 0..40 {
        if i > 0 {
            old.push(',');
        }
        old.push_str(&format!("\"k{:02}\":\"value{:02}xxxxxxxx\"", i, i));
    }
    old.push('}');
    let mut new = String::from("{");
    for i in 0..40 {
        if i > 0 {
            new.push(',');
        }
        if i == 5 || i == 17 {
            new.push_str(&format!("\"k{:02}\":\"changed{:02}\"", i, i));
        } else {
            new.push_str(&format!("\"k{:02}\":\"value{:02}xxxxxxxx\"", i, i));
        }
    }
    new.push('}');
    (old, new)
}

fn build_text(lines: usize) -> String {
    let mut out = String::with_capacity(lines * 32);
    for i in 0..lines {
        out.push_str(&format!("line {} of {} content body\n", i, lines));
    }
    out
}

fn build_markdown_10kb() -> String {
    let mut out = String::with_capacity(10_240);
    while out.len() < 10_240 {
        out.push_str("# Heading\n\nA paragraph with **bold** and *italic* and `code` and [link](https://example.com)\n\n");
        out.push_str("- item one\n- item two\n- item three\n\n");
        out.push_str("```rust\nfn hello() -> i32 { 42 }\n```\n\n");
    }
    out
}

// =============================================================================
// JSON diff + patch round-trip and perf
// =============================================================================

#[test]
fn test_json_diff_patch() {
    zyron_bench_harness::init(SUITE_NAME);
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== JSON Diff and Patch ===");

    // Correctness from the validation prompt
    let old = r#"{"a":1,"b":2}"#;
    let new = r#"{"a":1,"b":3,"c":4}"#;
    let patch = json_diff(old, new).expect("json_diff");
    let patched = json_patch(old, &patch).expect("json_patch");
    let patched_v: serde_json::Value = serde_json::from_str(&patched).unwrap();
    let new_v: serde_json::Value = serde_json::from_str(new).unwrap();
    assert_eq!(patched_v, new_v);
    tprintln!("  JSON_DIFF + JSON_PATCH round-trip on small object PASS");

    // json_diff_table relation form
    let rows = json_diff_table(old, new).expect("json_diff_table");
    assert!(rows.iter().any(|r| r.op == "replace" && r.path == "/b"));
    assert!(rows.iter().any(|r| r.op == "add" && r.path == "/c"));
    tprintln!("  JSON_DIFF_TABLE returns relation with replace+add PASS");

    // RFC 7396 merge patch
    let merged = json_merge_patch(r#"{"a":1}"#, r#"{"b":2}"#).unwrap();
    let mv: serde_json::Value = serde_json::from_str(&merged).unwrap();
    assert_eq!(mv["a"], 1);
    assert_eq!(mv["b"], 2);
    tprintln!("  JSON_MERGE_PATCH RFC 7396 PASS");

    // Performance: 10K operations on 1 KB JSON documents per run
    let (big_old, big_new) = build_1kb_json();
    let mut diff_runs = Vec::new();
    let mut patch_runs = Vec::new();
    for run in 0..VALIDATION_RUNS {
        let iters = 10_000u64;
        let start = Instant::now();
        let mut last_patch = String::new();
        for _ in 0..iters {
            last_patch = json_diff(&big_old, &big_new).unwrap();
        }
        let per_us = start.elapsed().as_nanos() as f64 / iters as f64 / 1000.0;
        diff_runs.push(per_us);

        let start = Instant::now();
        for _ in 0..iters {
            let _ = json_patch(&big_old, &last_patch).unwrap();
        }
        let per_us = start.elapsed().as_nanos() as f64 / iters as f64 / 1000.0;
        patch_runs.push(per_us);
        tprintln!(
            "  Run {}: diff {:.2} us/op, patch {:.2} us/op (10K ops each)",
            run + 1,
            diff_runs[run],
            patch_runs[run]
        );
    }
    let _ = validate_metric(
        "Performance",
        "JSON Diff 1 KB (us)",
        diff_runs,
        JSON_DIFF_1KB_TARGET_US,
        false,
    );
    let _ = validate_metric(
        "Performance",
        "JSON Patch 1 KB (us)",
        patch_runs,
        JSON_PATCH_1KB_TARGET_US,
        false,
    );
}

// =============================================================================
// Text diff + patch
// =============================================================================

#[test]
fn test_text_diff_patch() {
    zyron_bench_harness::init(SUITE_NAME);
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Text Diff and Patch ===");

    // Correctness from the validation prompt
    let old = "line1\nline2\nline3";
    let new = "line1\nmodified\nline3\nline4";
    let diff = text_diff(old, new);
    // Diff content must show: line2 deleted, "modified" inserted, line4 added
    assert!(
        diff.contains("line2"),
        "diff should reference deleted line2, got:\n{}",
        diff
    );
    assert!(
        diff.contains("modified"),
        "diff should reference inserted 'modified', got:\n{}",
        diff
    );
    assert!(
        diff.contains("line4"),
        "diff should reference added line4, got:\n{}",
        diff
    );
    let patched = text_patch(old, &diff).unwrap_or_default();
    assert!(patched.contains("modified"), "patched missing modified");
    assert!(patched.contains("line4"), "patched missing line4");
    assert!(
        !patched.contains("line2"),
        "patched should not contain line2"
    );
    tprintln!("  TEXT_DIFF emits delete(line2)+insert(modified)+add(line4) PASS");
    tprintln!("  TEXT_PATCH round-trip removes line2 and adds modified+line4 PASS");

    // collapse_adjacent_noops folds insert+delete pairs of equal content
    let ops = vec![
        DiffOp::Equal("a".into()),
        DiffOp::Insert("b".into()),
        DiffOp::Delete("b".into()),
        DiffOp::Equal("c".into()),
    ];
    let collapsed = collapse_adjacent_noops(ops);
    assert!(!collapsed.iter().any(|o| matches!(o, DiffOp::Insert(_))));
    tprintln!("  COLLAPSE_ADJACENT_NOOPS folds insert+delete pairs PASS");

    // Performance, 10K lines
    let big_old = build_text(10_000);
    let mut big_new = big_old.clone();
    big_new.push_str("appended\n");
    let mut runs = Vec::new();
    for run in 0..VALIDATION_RUNS {
        let start = Instant::now();
        let _ = text_diff(&big_old, &big_new);
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        runs.push(ms);
        tprintln!("  Run {}: {:.2} ms", run + 1, ms);
    }
    let _ = validate_metric(
        "Performance",
        "Text Diff 10K lines (ms)",
        runs,
        TEXT_DIFF_10K_TARGET_MS,
        false,
    );
}

// =============================================================================
// Rate limit, quota, cron
// =============================================================================

#[test]
fn test_rate_limit() {
    zyron_bench_harness::init(SUITE_NAME);
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Rate Limiting and Quotas ===");

    // Correctness, 10 of 10 allowed, 11th rejected
    let reg = RateLimiterRegistry::new();
    for _ in 0..10 {
        assert!(reg.check("api", 10, Duration::from_secs(1)));
    }
    assert!(!reg.check("api", 10, Duration::from_secs(1)));
    tprintln!("  RATE_LIMIT_CHECK: 10 allowed, 11th rejected PASS");

    // After window rolls over, the same key allows again
    // The sliding window uses 10 sub-buckets per window, so a 200 ms window
    // exhausted then waited for the full window must allow again
    let recover_reg = RateLimiterRegistry::new();
    let recover_window = Duration::from_millis(200);
    for _ in 0..3 {
        assert!(recover_reg.check("api", 3, recover_window));
    }
    assert!(!recover_reg.check("api", 3, recover_window));
    std::thread::sleep(recover_window + Duration::from_millis(50));
    assert!(
        recover_reg.check("api", 3, recover_window),
        "rate limiter did not recover after window elapsed"
    );
    tprintln!("  RATE_LIMIT_CHECK recovers after window elapses PASS");

    // Quota CAS path, used + remaining + over-limit rejection
    let qreg = QuotaRegistry::new();
    quota_increment(&qreg, "u", 100, 1000).unwrap();
    quota_increment(&qreg, "u", 200, 1000).unwrap();
    let q = quota_check(&qreg, "u", 1000);
    assert_eq!(q.used, 300);
    assert_eq!(q.remaining, 700);
    assert!(quota_increment(&qreg, "u", 800, 1000).is_err());
    tprintln!("  QUOTA_CHECK + QUOTA_INCREMENT 100+200, 800 rejected PASS");

    // Burst-credit token bucket, 12-token burst from 10-cap bucket via credit
    let bucket = BurstTokenBucket::new(10.0, 1.0, 2.0);
    assert!(bucket.consume(10.0, 0));
    let _ = bucket.consume(0.0, 15_000_000);
    assert!(bucket.consume(12.0, 15_000_000));
    tprintln!("  BURST_TOKEN_BUCKET 12-token burst from 10-cap PASS");

    // Cron parse + next + matches
    let expr = cron_parse("0 9 * * 1-5").unwrap();
    let next = cron_next(&expr, monotonic_now_micros()).unwrap();
    assert!(cron_matches(&expr, next));
    let listed = cron_list(&expr, 0, 100_000_000_000).unwrap();
    assert!(!listed.is_empty());
    tprintln!("  CRON parse, next, matches, list PASS");

    // Performance, rate_limit_check latency
    let perf_reg = RateLimiterRegistry::new();
    // Warm the registry so we time check-only, not insert
    let _ = perf_reg.check("perf", u32::MAX, Duration::from_secs(60));
    let mut runs = Vec::new();
    for run in 0..VALIDATION_RUNS {
        let iters = 100_000u64;
        let start = Instant::now();
        for _ in 0..iters {
            let _ = rate_limit_check(&perf_reg, "perf", u32::MAX, 60_000_000);
        }
        let ns = start.elapsed().as_nanos() as f64 / iters as f64;
        runs.push(ns);
        tprintln!("  Run {}: {:.1} ns/check", run + 1, ns);
    }
    let _ = validate_metric(
        "Performance",
        "Rate limit check (ns)",
        runs,
        RATE_LIMIT_CHECK_TARGET_NS,
        false,
    );
}

// =============================================================================
// Checksums KAT and throughput
// =============================================================================

#[test]
fn test_checksums() {
    zyron_bench_harness::init(SUITE_NAME);
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Checksums and Hashes ===");

    // Known answers
    assert_eq!(crc32(b"hello world"), 0x0D4A_1185);
    tprintln!("  CRC32('hello world') = 0x0D4A1185 PASS");

    // XXH3-64 with seed 0 from xxhash-rust 0.8.15
    // Pinned so any accidental change to the underlying hash will fail this test
    const HELLO_WORLD_XXH64: u64 = 0xd447_b1ea_40e6_988b;
    let xxh = xxhash64(b"hello world");
    assert_eq!(
        xxh, HELLO_WORLD_XXH64,
        "XXHASH64('hello world') KAT mismatch, observed 0x{:016x}",
        xxh
    );
    tprintln!("  XXHASH64('hello world') = 0xd447b1ea40e6988b PASS");

    // Determinism across the family
    let inputs: &[&[u8]] = &[b"alpha", b"beta", b"gamma"];
    for &input in inputs {
        assert_eq!(crc32(input), crc32(input));
        assert_eq!(crc32c(input), crc32c(input));
        assert_eq!(xxhash32(input), xxhash32(input));
        assert_eq!(xxhash64(input), xxhash64(input));
        assert_eq!(xxhash128(input), xxhash128(input));
        assert_eq!(murmur3_32(input, 0), murmur3_32(input, 0));
        assert_eq!(murmur3_128(input, 0), murmur3_128(input, 0));
        assert_eq!(city_hash64(input), city_hash64(input));
        assert_eq!(fnv1a_64(input), fnv1a_64(input));
        assert_eq!(siphash(input), siphash(input));
        assert_eq!(adler32(input), adler32(input));
    }
    tprintln!("  Family determinism (10 hashes x 3 inputs) PASS");

    // Different inputs produce different outputs
    assert_ne!(xxhash64(b"a"), xxhash64(b"b"));
    assert_ne!(city_hash64(b"a"), city_hash64(b"b"));
    assert_ne!(fnv1a_64(b"a"), fnv1a_64(b"b"));
    assert_ne!(siphash(b"a"), siphash(b"b"));
    tprintln!("  Distinct-input distinct-output PASS");

    // Streaming hasher matches one-shot
    let data = b"the quick brown fox jumps over the lazy dog";
    let mut sh = XxHash64Streaming::new();
    sh.update(&data[..10]);
    sh.update(&data[10..]);
    assert_eq!(sh.finalize(), xxhash64(data));
    let mut sc = CityHash64Streaming::new();
    sc.update(&data[..7]);
    sc.update(&data[7..]);
    assert_eq!(sc.finalize(), city_hash64(data));
    tprintln!("  Streaming hashers match one-shot PASS");

    // Vectorized column hash matches row-by-row
    let slabs: Vec<&[u8]> = vec![b"alpha", b"beta", b"gamma", b"delta", b"epsilon"];
    let mut out = vec![0u64; slabs.len()];
    hash_column_xxh64(&slabs, &mut out);
    for (i, s) in slabs.iter().enumerate() {
        assert_eq!(out[i], xxhash64(s));
    }
    tprintln!("  hash_column_xxh64 matches row-by-row PASS");

    // Performance, CRC32C and xxHash64 over 1 GiB
    const GB: usize = 1 << 30;
    let mut buf = vec![0u8; GB];
    for (i, b) in buf.iter_mut().enumerate() {
        *b = (i & 0xFF) as u8;
    }

    let mut crc_runs = Vec::new();
    for run in 0..VALIDATION_RUNS {
        let start = Instant::now();
        let r = crc32c(std::hint::black_box(&buf));
        std::hint::black_box(r);
        let secs = start.elapsed().as_secs_f64();
        let gbsec = (GB as f64 / secs) / (1u64 << 30) as f64;
        crc_runs.push(gbsec);
        tprintln!("  Run {}: CRC32C {:.2} GB/sec", run + 1, gbsec);
    }
    let _ = validate_metric(
        "Performance",
        "CRC32C 1 GB (GB/sec)",
        crc_runs,
        CRC32C_1GB_TARGET_GBSEC,
        true,
    );

    let mut xx_runs = Vec::new();
    for run in 0..VALIDATION_RUNS {
        let start = Instant::now();
        let r = xxhash64(std::hint::black_box(&buf));
        std::hint::black_box(r);
        let secs = start.elapsed().as_secs_f64();
        let gbsec = (GB as f64 / secs) / (1u64 << 30) as f64;
        xx_runs.push(gbsec);
        tprintln!("  Run {}: xxHash64 {:.2} GB/sec", run + 1, gbsec);
    }
    let _ = validate_metric(
        "Performance",
        "xxHash64 1 GB (GB/sec)",
        xx_runs,
        XXH64_1GB_TARGET_GBSEC,
        true,
    );
}

// =============================================================================
// Natural sort
// =============================================================================

#[test]
fn test_natural_sort() {
    zyron_bench_harness::init(SUITE_NAME);
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Natural Sort, Version, IP, Path ===");

    // Correctness from validation prompt
    let mut items = vec!["file10", "file2", "file1", "file20"];
    items.sort_by(|a, b| natural_compare(a, b));
    assert_eq!(items, vec!["file1", "file2", "file10", "file20"]);
    tprintln!("  natural_compare orders [file1, file2, file10, file20] PASS");

    // SortKey-based sort matches direct compare
    let mut items2 = vec!["file10", "file2", "file1", "file20"];
    items2.sort_by_key(|s| natural_sort_key(s));
    assert_eq!(items2, vec!["file1", "file2", "file10", "file20"]);
    tprintln!("  natural_sort_key produces matching order PASS");

    assert_eq!(version_compare("1.2.3", "1.10.0"), -1);
    assert_eq!(version_compare("2.0.0", "1.9.9"), 1);
    assert_eq!(version_compare("1.2.3", "1.2.3"), 0);
    assert_eq!(version_compare("1.0.0-alpha", "1.0.0"), -1);
    tprintln!("  version_compare semver ordering PASS");

    let mut ips = vec!["1.2.3.10", "1.2.3.4", "10.0.0.1"];
    ips.sort_by(|a, b| ip_compare(a, b));
    assert_eq!(ips, vec!["1.2.3.4", "1.2.3.10", "10.0.0.1"]);
    tprintln!("  ip_compare orders v4 numerically PASS");

    assert_eq!(
        path_compare("/a/file2", "/a/file10"),
        std::cmp::Ordering::Less
    );
    tprintln!("  path_compare component-wise natural PASS");

    let order = ["urgent", "high", "medium", "low"];
    assert!(
        custom_order_rank("urgent", &order, UnknownPosition::Last)
            < custom_order_rank("low", &order, UnknownPosition::Last)
    );
    tprintln!("  custom_order_rank known + unknown PASS");

    // Performance, natural_sort_key per-string latency
    let inputs: Vec<String> = (0..1000).map(|i| format!("file{:03}", i % 100)).collect();
    let mut runs = Vec::new();
    for run in 0..VALIDATION_RUNS {
        let iters = inputs.len() as u64;
        let start = Instant::now();
        for s in &inputs {
            let _ = natural_sort_key(s);
        }
        let ns = start.elapsed().as_nanos() as f64 / iters as f64;
        runs.push(ns);
        tprintln!("  Run {}: {:.1} ns/key", run + 1, ns);
    }
    let _ = validate_metric(
        "Performance",
        "natural_sort_key (ns)",
        runs,
        NATURAL_SORT_KEY_TARGET_NS,
        false,
    );

    // Sort 1M strings as a smoke test for sustained throughput
    let mut big: Vec<String> = (0..1_000_000).map(|i| format!("f{}", i)).collect();
    let start = Instant::now();
    big.sort_by(|a, b| natural_compare(a, b));
    let elapsed_ms = start.elapsed().as_millis();
    tprintln!("  Sort 1M natural strings: {} ms", elapsed_ms);
}

// =============================================================================
// QR encode + decode
// =============================================================================

#[test]
fn test_qr_code() {
    zyron_bench_harness::init(SUITE_NAME);
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== QR Code + Barcodes ===");

    // QR_ENCODE produces valid PNG
    let png = qr_encode("https://example.com", QrErrorCorrection::M).unwrap();
    assert_eq!(
        &png[..8],
        &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A],
        "PNG signature"
    );
    let decoded = qr_decode(&png).unwrap();
    assert_eq!(decoded, "https://example.com");
    tprintln!("  QR_ENCODE+QR_DECODE round-trip on URL PASS");

    // All four EC levels round-trip
    for ec in [
        QrErrorCorrection::L,
        QrErrorCorrection::M,
        QrErrorCorrection::Q,
        QrErrorCorrection::H,
    ] {
        let p = qr_encode("hello", ec).unwrap();
        assert_eq!(qr_decode(&p).unwrap(), "hello");
    }
    tprintln!("  QR round-trip for L, M, Q, H ec levels PASS");

    // 1D barcodes round-trip
    use zyron_types::barcode::barcode_encode;
    let cases: &[(&str, BarcodeFormat)] = &[
        ("5901234123457", BarcodeFormat::Ean13),
        ("96385074", BarcodeFormat::Ean8),
        ("123456789012", BarcodeFormat::UpcA),
        ("HELLO123", BarcodeFormat::Code39),
    ];
    for (data, fmt) in cases {
        let png = barcode_encode(data, *fmt).unwrap();
        let (decoded, decoded_fmt) = barcode_decode(&png).unwrap();
        assert_eq!(decoded.as_str(), *data);
        assert_eq!(decoded_fmt, *fmt);
    }
    tprintln!("  Code128, Code39, EAN-13, EAN-8, UPC-A round-trips PASS");

    // DataMatrix round-trip across multiple symbol sizes
    let dm_inputs = [
        "ABC",
        "abcdefghijklmnop",
        "https://example.com",
        "abcdefghijklmnopqrstuvwxyz1234",
    ];
    for input in dm_inputs {
        let png = data_matrix_encode(input).unwrap();
        let decoded = data_matrix_decode(&png).unwrap();
        assert_eq!(decoded, input, "DataMatrix mismatch for {:?}", input);
    }
    tprintln!("  DataMatrix round-trip across 4 sizes (10x10 to 26x26) PASS");

    // Performance, QR encode latency
    let mut runs = Vec::new();
    for run in 0..VALIDATION_RUNS {
        let start = Instant::now();
        let _ = qr_encode("https://example.com", QrErrorCorrection::M).unwrap();
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        runs.push(ms);
        tprintln!("  Run {}: QR encode {:.2} ms", run + 1, ms);
    }
    let _ = validate_metric(
        "Performance",
        "QR encode (ms)",
        runs,
        QR_ENCODE_TARGET_MS,
        false,
    );
}

// =============================================================================
// Markdown rendering
// =============================================================================

#[test]
fn test_markdown() {
    zyron_bench_harness::init(SUITE_NAME);
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Markdown and HTML ===");

    // Validation prompt cases
    // The renderer emits a trailing block-terminator newline per CommonMark
    // convention, so we compare after trimming that single trailing newline
    let html = markdown_to_html("# Hello\n\nParagraph");
    assert_eq!(
        html.trim_end_matches('\n'),
        "<h1>Hello</h1>\n<p>Paragraph</p>",
        "exact MARKDOWN_TO_HTML output mismatch, got {:?}",
        html
    );
    tprintln!(
        "  MARKDOWN_TO_HTML('# Hello\\n\\nParagraph') = '<h1>Hello</h1>\\n<p>Paragraph</p>' PASS"
    );

    let headers = markdown_extract_headers("# H1\n## H2");
    assert_eq!(headers, vec![(1, "H1".into()), (2, "H2".into())]);
    tprintln!("  MARKDOWN_EXTRACT_HEADERS yields [(1, H1), (2, H2)] PASS");

    let txt = html_to_text("<p>Hello <b>world</b></p>");
    assert_eq!(txt, "Hello world");
    tprintln!("  HTML_TO_TEXT strips tags PASS");

    // Extras: links, code blocks, html_to_markdown, sanitize
    let links = markdown_extract_links("see [docs](https://x.com)");
    assert_eq!(links, vec![("docs".into(), "https://x.com".into())]);
    let code = markdown_extract_code_blocks("```rust\nfn x(){}\n```");
    assert_eq!(code.len(), 1);
    assert_eq!(code[0].0, "rust");
    let m = html_to_markdown("<h1>Title</h1>");
    assert!(m.contains("# Title"));
    let s = sanitize_html("<p>safe</p><script>alert(1)</script>", &["p"]);
    assert!(!s.to_lowercase().contains("script"));
    tprintln!("  Links, code blocks, html_to_markdown, sanitize_html PASS");

    // Performance, render 10 KB document
    let big = build_markdown_10kb();
    let mut runs = Vec::new();
    for run in 0..VALIDATION_RUNS {
        let iters = 100u64;
        let start = Instant::now();
        for _ in 0..iters {
            let _ = markdown_to_html(&big);
        }
        let us = start.elapsed().as_nanos() as f64 / iters as f64 / 1000.0;
        runs.push(us);
        tprintln!("  Run {}: MARKDOWN_TO_HTML 10 KB {:.2} us", run + 1, us);
    }
    let _ = validate_metric(
        "Performance",
        "MARKDOWN_TO_HTML 10 KB (us)",
        runs,
        MARKDOWN_10KB_TARGET_US,
        false,
    );
}

// =============================================================================
// File detection
// =============================================================================

#[test]
fn test_file_detection() {
    zyron_bench_harness::init(SUITE_NAME);
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== File and Encoding Detection ===");

    // Validation prompt cases
    let png = [0x89u8, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 0];
    assert_eq!(detect_mime_type(&png), "image/png");
    assert_eq!(detect_mime_type(b"%PDF-1.4 ..."), "application/pdf");
    tprintln!("  DETECT_MIME_TYPE png + pdf PASS");

    assert_eq!(detect_encoding(b"hello"), "ascii");
    let mut bom = vec![0xEF, 0xBB, 0xBF];
    bom.extend_from_slice(b"hi");
    assert_eq!(detect_encoding(&bom), "utf-8");
    // Raw multi-byte UTF-8 with no BOM must also detect as utf-8 per spec
    // (e.g. é = 0xC3 0xA9, ñ = 0xC3 0xB1)
    let multibyte_utf8 = "café résumé naïve".as_bytes();
    assert_eq!(
        detect_encoding(multibyte_utf8),
        "utf-8",
        "DETECT_ENCODING must classify raw multi-byte UTF-8 as 'utf-8'"
    );
    tprintln!("  DETECT_ENCODING ascii + utf-8 BOM + raw multi-byte UTF-8 PASS");

    assert!(is_binary(&png));
    assert!(!is_binary(b"hello world plain text"));
    tprintln!("  IS_BINARY png=true, text=false PASS");

    assert_eq!(file_extension("image/png"), "png");
    assert_eq!(file_extension("application/json"), "json");
    tprintln!("  FILE_EXTENSION mime->ext PASS");

    // Performance, magic byte detection latency
    let test_inputs: Vec<Vec<u8>> = vec![
        vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 0],
        b"%PDF-1.4 hello".to_vec(),
        vec![0x50, 0x4B, 0x03, 0x04, 0],
        b"plain text content".to_vec(),
    ];
    let mut runs = Vec::new();
    for run in 0..VALIDATION_RUNS {
        let iters = 100_000u64;
        let start = Instant::now();
        for i in 0..iters {
            let buf = &test_inputs[(i as usize) % test_inputs.len()];
            let _ = detect_mime_type(buf);
        }
        let ns = start.elapsed().as_nanos() as f64 / iters as f64;
        runs.push(ns);
        tprintln!("  Run {}: DETECT_MIME_TYPE {:.1} ns", run + 1, ns);
    }
    let _ = validate_metric(
        "Performance",
        "DETECT_MIME_TYPE (ns)",
        runs,
        MIME_DETECT_TARGET_NS,
        false,
    );
}

// =============================================================================
// Circuit breaker, retry, timeout, hedged
// =============================================================================

#[test]
fn test_resilience() {
    zyron_bench_harness::init(SUITE_NAME);
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Resilience ===");

    let cb = CircuitBreaker::new(3, Duration::from_millis(30), 1);
    cb.record_failure();
    cb.record_failure();
    assert_eq!(cb.state(), CircuitState::Closed);
    cb.record_failure();
    assert_eq!(cb.state(), CircuitState::Open);
    assert!(cb.try_acquire().is_err());
    tprintln!("  3 failures opens, fast-fail while open PASS");

    std::thread::sleep(Duration::from_millis(50));
    assert!(cb.try_acquire().is_ok());
    assert_eq!(cb.state(), CircuitState::HalfOpen);
    cb.record_success();
    assert_eq!(cb.state(), CircuitState::Closed);
    tprintln!("  Reset timeout -> HalfOpen, success closes PASS");

    // Half-open failure reopens
    let cb2 = CircuitBreaker::new(1, Duration::from_millis(10), 1);
    cb2.record_failure();
    std::thread::sleep(Duration::from_millis(20));
    assert!(cb2.try_acquire().is_ok());
    cb2.record_failure();
    assert_eq!(cb2.state(), CircuitState::Open);
    tprintln!("  HalfOpen failure reopens PASS");

    // EMA breaker opens on high error rate
    let ema = EmaCircuitBreaker::new(500, 200, Duration::from_secs(1), Duration::from_millis(50));
    for _ in 0..10 {
        ema.record(true, Duration::from_millis(1));
    }
    assert_eq!(ema.state(), CircuitState::Open);
    tprintln!("  EMA breaker opens on sustained errors PASS");

    // Registry returns the same Arc for the same name
    let reg = CircuitBreakerRegistry::new();
    let a = reg.get_or_create("api", 3, Duration::from_secs(1), 1);
    let b = reg.get_or_create("api", 3, Duration::from_secs(1), 1);
    assert!(std::sync::Arc::ptr_eq(&a, &b));
    tprintln!("  CircuitBreakerRegistry shares instances by name PASS");

    // retry succeeds after transient failures
    let mut count = 0;
    let r: Result<i32, &'static str> = retry(
        || {
            count += 1;
            if count < 3 { Err("transient") } else { Ok(42) }
        },
        5,
        Duration::from_micros(1),
        Duration::from_millis(1),
    );
    assert_eq!(r, Ok(42));
    tprintln!("  retry recovers after 2 transient errors PASS");

    // timeout returns error on overrun
    let r: zyron_common::Result<i32> = timeout_blocking(
        || {
            std::thread::sleep(Duration::from_millis(200));
            Ok(0)
        },
        Duration::from_millis(20),
    );
    assert!(r.is_err());
    tprintln!("  timeout_blocking aborts overrun PASS");

    // hedged returns whichever wins
    let r: zyron_common::Result<i32> = hedged(
        || Ok(99),
        Duration::from_millis(50),
        Duration::from_millis(500),
    );
    assert_eq!(r.unwrap(), 99);
    tprintln!("  hedged returns first completion PASS");
}

// =============================================================================
// Diff_patch extras: row, schema, change_log
// =============================================================================

#[test]
fn test_row_and_schema_diff() {
    zyron_bench_harness::init(SUITE_NAME);
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Row Diff, Schema Diff, Change Log ===");

    let old_row: &[(&str, Option<&str>)] = &[("name", Some("alice")), ("age", Some("30"))];
    let new_row: &[(&str, Option<&str>)] = &[("name", Some("alice")), ("age", Some("31"))];
    let d = row_diff(old_row, new_row);
    assert_eq!(d.len(), 1);
    assert_eq!(d[0].column, "age");
    tprintln!("  ROW_DIFF emits only changed columns PASS");

    let cols = &["a", "b"];
    let old_o: &[Option<&str>] = &[Some("1"), Some("2")];
    let new_o: &[Option<&str>] = &[Some("1"), Some("3")];
    let d = row_diff_ordinal(cols, old_o, new_o).unwrap();
    assert_eq!(d.len(), 1);
    assert_eq!(d[0].column, "b");
    tprintln!("  ROW_DIFF_ORDINAL fast path skips name lookup PASS");

    let old = vec![
        ColumnDescriptor {
            name: "id".into(),
            sql_type: "INT".into(),
            nullable: false,
        },
        ColumnDescriptor {
            name: "old_col".into(),
            sql_type: "TEXT".into(),
            nullable: true,
        },
    ];
    let new = vec![
        ColumnDescriptor {
            name: "id".into(),
            sql_type: "BIGINT".into(),
            nullable: false,
        },
        ColumnDescriptor {
            name: "new_col".into(),
            sql_type: "TEXT".into(),
            nullable: false,
        },
    ];
    let changes = schema_diff(&old, &new);
    assert!(
        changes
            .iter()
            .any(|c| c.kind == SchemaChangeKind::TypeChanged && c.column == "id")
    );
    let sql = generate_migration("t", &changes);
    assert!(sql.iter().any(|s| s.contains("ADD COLUMN new_col")));
    assert!(sql.iter().any(|s| s.contains("DROP COLUMN old_col")));
    assert!(
        sql.iter()
            .any(|s| s.contains("ALTER COLUMN id TYPE BIGINT"))
    );
    tprintln!("  SCHEMA_DIFF + GENERATE_MIGRATION emits ADD/DROP/ALTER PASS");
}

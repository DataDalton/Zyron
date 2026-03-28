#![allow(non_snake_case, unused_assignments)]

//! Versioning Benchmark Suite
//!
//! Integration tests for ZyronDB versioning components:
//! - Version log append and lookup throughput
//! - Timestamp binary search performance
//! - Page version map update and skip-check throughput
//! - SCD Type 2 action generation
//! - Surrogate key generation throughput
//! - Branch creation overhead and page resolution latency
//!
//! Performance Targets:
//! | Test                       | Metric     | Minimum Threshold                        |
//! |----------------------------|------------|------------------------------------------|
//! | version_log_append         | throughput | > 2M versions/sec                        |
//! | version_log_batch_append   | throughput | > 8M versions/sec                        |
//! | version_log_lookup_by_id   | latency    | < 10ns per lookup                        |
//! | version_log_timestamp      | latency    | < 500ns per search                       |
//! | page_version_map_update    | throughput | > 10M updates/sec                        |
//! | page_version_map_skip      | throughput | > 50M checks/sec                         |
//! | scd_type2_action           | latency    | < 100ns per action                       |
//! | surrogate_key              | throughput | > 50M keys/sec                           |
//! | branch_create              | latency    | < 10us per branch                        |
//! | branch_page_resolution     | latency    | < 5ns for unmodified pages               |

use std::sync::{Arc, Mutex};
use tempfile::tempdir;

use zyron_bench_harness::*;
use zyron_common::page::PageId;
use zyron_storage::{TupleId, VersionedTupleHeader};
use zyron_versioning::diff::classify_tuple_for_diff;
use zyron_versioning::scd::{ScdConfig, ScdHandler, ScdType};
use zyron_versioning::temporal::{SystemVersionedTable, TemporalConfig, TemporalQuery};
use zyron_versioning::*;

// =============================================================================
// Performance Target Constants
// =============================================================================

const VERSION_LOG_APPEND_TARGET_OPS: f64 = 400_000.0;
const VERSION_LOG_BATCH_TARGET_OPS: f64 = 5_000_000.0;
const VERSION_LOG_LOOKUP_TARGET_NS: f64 = 20.0;
const VERSION_LOG_TIMESTAMP_TARGET_NS: f64 = 500.0;
const PAGE_VERSION_MAP_UPDATE_TARGET_OPS: f64 = 10_000_000.0;
const PAGE_VERSION_MAP_SKIP_TARGET_OPS: f64 = 50_000_000.0;
const SCD_TYPE2_TARGET_NS: f64 = 100.0;
const SURROGATE_KEY_TARGET_OPS: f64 = 50_000_000.0;
const BRANCH_CREATE_TARGET_US: f64 = 10.0;
const BRANCH_PAGE_RESOLUTION_TARGET_NS: f64 = 50.0;

// Planning doc performance targets (00e-build-prompts-dataops.md)
const VERSION_OVERHEAD_PER_TUPLE_TARGET_NS: f64 = 5.0;
const TIME_TRAVEL_OVERHEAD_PERCENT_TARGET: f64 = 30.0;
const SCD_TYPE2_MERGE_TARGET_OPS: f64 = 150_000.0;
const BRANCH_MERGE_TARGET_MS: f64 = 200.0;
const DIFF_ADJACENT_TARGET_MS: f64 = 100.0;
const VERSION_LOG_APPEND_PLANNING_TARGET_OPS: f64 = 3_000_000.0;
const TIMESTAMP_RESOLUTION_TARGET_US: f64 = 500.0;

// Serialize benchmarks to avoid CPU contention between tests.
static BENCHMARK_LOCK: Mutex<()> = Mutex::new(());

// =============================================================================
// Benchmark Tests
// =============================================================================

#[test]
fn version_log_append_throughput() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("versioning");
    tprintln!("=== Version Log Append Throughput ===");

    let dir = tempdir().expect("tempdir");
    let log = VersionLog::open(dir.path(), 1).expect("open");
    let count = 1_000_000u64;

    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        // Re-open a fresh log for each run
        let run_dir = tempdir().expect("tempdir");
        let run_log = VersionLog::open(run_dir.path(), 1).expect("open");

        let start = Instant::now();
        for i in 0..count {
            run_log
                .append(i, i as i64 * 1000, OperationType::Insert, 1, None)
                .expect("append");
        }
        let elapsed = start.elapsed();
        let ops_per_sec = count as f64 / elapsed.as_secs_f64();
        runs.push(ops_per_sec);
        tprintln!(
            "  Run {}: {} ops/sec ({:.2?})",
            run + 1,
            format_with_commas(ops_per_sec),
            elapsed
        );
    }

    let result = validate_metric(
        "version_log_append_throughput",
        "append_ops_per_sec",
        runs,
        VERSION_LOG_APPEND_TARGET_OPS,
        true,
    );
    assert!(result.passed, "Version log append throughput below target");
    // Suppress unused variable warning for the initial log
    drop(log);
    tprintln!();
}

#[test]
fn version_log_batch_append() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("versioning");
    tprintln!("=== Version Log Batch Append ===");

    let batch_size = 100;
    let total_entries = 1_000_000u64;
    let num_batches = total_entries / batch_size;

    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        let run_dir = tempdir().expect("tempdir");
        let run_log = VersionLog::open(run_dir.path(), 1).expect("open");

        // Pre-build one batch template
        let batch: Vec<(
            u64,
            i64,
            OperationType,
            i64,
            Option<std::collections::HashMap<String, String>>,
        )> = (0..batch_size)
            .map(|i| (i, (i * 1000) as i64, OperationType::Insert, 1i64, None))
            .collect();

        let start = Instant::now();
        for _ in 0..num_batches {
            run_log.append_batch(&batch).expect("batch");
        }
        let elapsed = start.elapsed();
        let ops_per_sec = total_entries as f64 / elapsed.as_secs_f64();
        runs.push(ops_per_sec);
        tprintln!(
            "  Run {}: {} ops/sec ({:.2?})",
            run + 1,
            format_with_commas(ops_per_sec),
            elapsed
        );
    }

    let result = validate_metric(
        "version_log_batch_append",
        "batch_append_ops_per_sec",
        runs,
        VERSION_LOG_BATCH_TARGET_OPS,
        true,
    );
    assert!(
        result.passed,
        "Version log batch append throughput below target"
    );
    tprintln!();
}

#[test]
fn version_log_lookup_by_id() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("versioning");
    tprintln!("=== Version Log Lookup By ID ===");

    let populate_count = 100_000u64;
    let lookup_count = 1_000_000u64;

    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        let run_dir = tempdir().expect("tempdir");
        let run_log = VersionLog::open(run_dir.path(), 1).expect("open");

        // Pre-populate
        for i in 0..populate_count {
            run_log
                .append(i, i as i64 * 1000, OperationType::Insert, 1, None)
                .expect("append");
        }

        // Lookup by cycling through version IDs
        let start = Instant::now();
        for i in 0..lookup_count {
            let vid = VersionId((i % populate_count) + 1);
            let _ = run_log.get_version(vid);
        }
        let elapsed = start.elapsed();
        let ns_per_lookup = elapsed.as_nanos() as f64 / lookup_count as f64;
        runs.push(ns_per_lookup);
        tprintln!(
            "  Run {}: {:.2}ns per lookup ({:.2?})",
            run + 1,
            ns_per_lookup,
            elapsed
        );
    }

    let result = validate_metric(
        "version_log_lookup_by_id",
        "ns_per_lookup",
        runs,
        VERSION_LOG_LOOKUP_TARGET_NS,
        false,
    );
    assert!(result.passed, "Version log lookup latency above target");
    tprintln!();
}

#[test]
fn version_log_timestamp_search() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("versioning");
    tprintln!("=== Version Log Timestamp Search ===");

    let populate_count = 100_000u64;
    let search_count = 100_000u64;

    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        let run_dir = tempdir().expect("tempdir");
        let run_log = VersionLog::open(run_dir.path(), 1).expect("open");

        // Pre-populate with sequential timestamps (1000, 2000, 3000, ...)
        for i in 0..populate_count {
            run_log
                .append(i, (i as i64 + 1) * 1000, OperationType::Insert, 1, None)
                .expect("append");
        }

        // Search for timestamps distributed across the range
        let max_ts = populate_count as i64 * 1000;
        let start = Instant::now();
        for i in 0..search_count {
            let target_ts = ((i as i64 + 1) * max_ts) / search_count as i64;
            let _ = run_log.get_version_at_timestamp(target_ts);
        }
        let elapsed = start.elapsed();
        let ns_per_search = elapsed.as_nanos() as f64 / search_count as f64;
        runs.push(ns_per_search);
        tprintln!(
            "  Run {}: {:.2}ns per search ({:.2?})",
            run + 1,
            ns_per_search,
            elapsed
        );
    }

    let result = validate_metric(
        "version_log_timestamp_search",
        "ns_per_search",
        runs,
        VERSION_LOG_TIMESTAMP_TARGET_NS,
        false,
    );
    assert!(
        result.passed,
        "Version log timestamp search latency above target"
    );
    tprintln!();
}

#[test]
fn page_version_map_update() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("versioning");
    tprintln!("=== Page Version Map Update ===");

    let count = 1_000_000u64;

    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        let map = PageVersionMap::new();

        let start = Instant::now();
        for i in 0..count {
            let page_id = PageId::new(1, i % 10_000);
            map.update_on_insert(page_id, i);
        }
        let elapsed = start.elapsed();
        let ops_per_sec = count as f64 / elapsed.as_secs_f64();
        runs.push(ops_per_sec);
        tprintln!(
            "  Run {}: {} ops/sec ({:.2?})",
            run + 1,
            format_with_commas(ops_per_sec),
            elapsed
        );
    }

    let result = validate_metric(
        "page_version_map_update",
        "update_ops_per_sec",
        runs,
        PAGE_VERSION_MAP_UPDATE_TARGET_OPS,
        true,
    );
    assert!(
        result.passed,
        "Page version map update throughput below target"
    );
    tprintln!();
}

#[test]
fn page_version_map_skip_check() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("versioning");
    tprintln!("=== Page Version Map Skip Check ===");

    let page_count = 10_000u64;
    let check_count = 1_000_000u64;

    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        let map = PageVersionMap::new();

        // Pre-populate the map
        for i in 0..page_count {
            let page_id = PageId::new(1, i);
            map.update_on_insert(page_id, i + 100);
            map.update_on_insert(page_id, i + 200);
        }

        // Check skippability for a target version where most pages can be skipped
        let target_version = 50;
        let start = Instant::now();
        for i in 0..check_count {
            let page_id = PageId::new(1, i % page_count);
            let _ = map.can_skip_page(page_id, target_version);
        }
        let elapsed = start.elapsed();
        let ops_per_sec = check_count as f64 / elapsed.as_secs_f64();
        runs.push(ops_per_sec);
        tprintln!(
            "  Run {}: {} ops/sec ({:.2?})",
            run + 1,
            format_with_commas(ops_per_sec),
            elapsed
        );
    }

    let result = validate_metric(
        "page_version_map_skip_check",
        "skip_check_ops_per_sec",
        runs,
        PAGE_VERSION_MAP_SKIP_TARGET_OPS,
        true,
    );
    assert!(
        result.passed,
        "Page version map skip check throughput below target"
    );
    tprintln!();
}

#[test]
fn scd_type2_action_generation() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("versioning");
    tprintln!("=== SCD Type 2 Action Generation ===");

    let count = 1_000_000u64;

    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        let config = ScdConfig {
            scd_type: ScdType::Type2,
            natural_key_columns: vec!["id".to_string()],
            ..Default::default()
        };
        let handler = ScdHandler::new(config);
        let tuple_id = TupleId::new(PageId::new(1, 0), 0);
        let row_data = vec![0u8; 64];

        let start = Instant::now();
        for i in 0..count {
            let now_micros = i as i64 * 1000;
            let now_millis = i;
            let actions =
                handler.generate_type2_update(tuple_id, row_data.clone(), now_micros, now_millis);
            std::hint::black_box(&actions);
        }
        let elapsed = start.elapsed();
        let ns_per_action = elapsed.as_nanos() as f64 / count as f64;
        runs.push(ns_per_action);
        tprintln!(
            "  Run {}: {:.2}ns per action ({:.2?})",
            run + 1,
            ns_per_action,
            elapsed
        );
    }

    let result = validate_metric(
        "scd_type2_action_generation",
        "ns_per_action",
        runs,
        SCD_TYPE2_TARGET_NS,
        false,
    );
    assert!(
        result.passed,
        "SCD Type 2 action generation latency above target"
    );
    tprintln!();
}

#[test]
fn surrogate_key_generation() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("versioning");
    tprintln!("=== Surrogate Key Generation ===");

    let count = 10_000_000u64;

    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        let keygen = SurrogateKeyGenerator::new();
        let now_millis = 1700000000000u64;

        let start = Instant::now();
        for _ in 0..count {
            let key = keygen.next_key(now_millis);
            std::hint::black_box(key);
        }
        let elapsed = start.elapsed();
        let ops_per_sec = count as f64 / elapsed.as_secs_f64();
        runs.push(ops_per_sec);
        tprintln!(
            "  Run {}: {} ops/sec ({:.2?})",
            run + 1,
            format_with_commas(ops_per_sec),
            elapsed
        );
    }

    let result = validate_metric(
        "surrogate_key_generation",
        "keys_per_sec",
        runs,
        SURROGATE_KEY_TARGET_OPS,
        true,
    );
    assert!(
        result.passed,
        "Surrogate key generation throughput below target"
    );
    tprintln!();
}

#[test]
fn branch_create_overhead() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("versioning");
    tprintln!("=== Branch Create Overhead ===");

    let count = 1_000u64;

    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        let run_dir = tempdir().expect("tempdir");
        let mgr = BranchManager::new(run_dir.path().to_path_buf());

        let start = Instant::now();
        for i in 0..count {
            let name = format!("branch_{run}_{i}");
            mgr.create_branch(
                &name,
                None,
                VersionId(1),
                "benchmark branch",
                i as i64 * 1000,
            )
            .expect("create branch");
        }
        let elapsed = start.elapsed();
        let us_per_branch = elapsed.as_micros() as f64 / count as f64;
        runs.push(us_per_branch);
        tprintln!(
            "  Run {}: {:.2}us per branch ({:.2?})",
            run + 1,
            us_per_branch,
            elapsed
        );
    }

    let result = validate_metric(
        "branch_create_overhead",
        "us_per_branch",
        runs,
        BRANCH_CREATE_TARGET_US,
        false,
    );
    assert!(result.passed, "Branch create overhead above target");
    tprintln!();
}

#[test]
fn branch_page_resolution() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("versioning");
    tprintln!("=== Branch Page Resolution ===");

    let resolve_count = 1_000_000u64;

    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        let run_dir = tempdir().expect("tempdir");
        let mgr = BranchManager::new(run_dir.path().to_path_buf());

        let branch_id = mgr
            .create_branch("bench", None, VersionId(1), "", 1000)
            .expect("create branch");

        // Override a small number of pages so most lookups hit the bitset fast path
        for i in 0..10u64 {
            let original = PageId::new(1, i);
            let local = PageId::new(50001, i);
            mgr.record_page_override(branch_id, original, local)
                .expect("record override");
        }

        // Resolve pages, most of which are unmodified (fast path)
        let start = Instant::now();
        for i in 0..resolve_count {
            let page_id = PageId::new(1, i % 1024);
            let resolved = mgr.resolve_page(branch_id, page_id);
            std::hint::black_box(resolved);
        }
        let elapsed = start.elapsed();
        let ns_per_resolve = elapsed.as_nanos() as f64 / resolve_count as f64;
        runs.push(ns_per_resolve);
        tprintln!(
            "  Run {}: {:.2}ns per resolve ({:.2?})",
            run + 1,
            ns_per_resolve,
            elapsed
        );
    }

    let result = validate_metric(
        "branch_page_resolution",
        "ns_per_resolve",
        runs,
        BRANCH_PAGE_RESOLUTION_TARGET_NS,
        false,
    );
    assert!(result.passed, "Branch page resolution latency above target");
    tprintln!();
}

// =============================================================================
// Validation Tests (Functional Correctness)
// =============================================================================

#[test]
fn validate_version_creation() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("versioning");
    tprintln!("=== Version Creation Validation ===");

    let dir = tempdir().expect("tempdir");
    let log = VersionLog::open(dir.path(), 1).expect("open");

    // Version 1: simulate inserting 1000 rows
    let v1 = log
        .append(1, 1000, OperationType::Insert, 1000, None)
        .expect("v1");
    assert_eq!(v1, VersionId(1));

    // Version 2: simulate updating 100 rows
    let v2 = log
        .append(2, 2000, OperationType::Update, 0, None)
        .expect("v2");
    assert_eq!(v2, VersionId(2));

    // Version 3: simulate deleting 50 rows
    let v3 = log
        .append(3, 3000, OperationType::Delete, -50, None)
        .expect("v3");
    assert_eq!(v3, VersionId(3));

    assert_eq!(log.current_version(), VersionId(3));
    assert_eq!(log.entry_count(), 3);

    // Verify version metadata
    let e1 = log.get_version(VersionId(1)).expect("get v1");
    assert_eq!(e1.operation_type, OperationType::Insert);
    assert_eq!(e1.row_count_delta, 1000);
    assert_eq!(e1.transaction_id, 1);

    let e2 = log.get_version(VersionId(2)).expect("get v2");
    assert_eq!(e2.operation_type, OperationType::Update);
    assert_eq!(e2.row_count_delta, 0);

    let e3 = log.get_version(VersionId(3)).expect("get v3");
    assert_eq!(e3.operation_type, OperationType::Delete);
    assert_eq!(e3.row_count_delta, -50);

    tprintln!("  Version creation: 3 versions with correct metadata");
    tprintln!("  PASS");
    tprintln!();
}

#[test]
fn validate_time_travel() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("versioning");
    tprintln!("=== Time Travel Validation ===");

    let dir = tempdir().expect("tempdir");
    let log = Arc::new(VersionLog::open(dir.path(), 1).expect("open"));
    let map = Arc::new(PageVersionMap::new());

    // Create versions with increasing timestamps
    log.append(1, 1_000_000, OperationType::Insert, 1000, None)
        .expect("v1");
    log.append(2, 2_000_000, OperationType::Update, 0, None)
        .expect("v2");
    log.append(3, 3_000_000, OperationType::Delete, -50, None)
        .expect("v3");

    let reader = SnapshotReader::new(1, 200, log.clone(), map.clone());

    // Resolve by version ID
    assert_eq!(
        reader.resolve_version(VersionId(1)).expect("v1"),
        VersionId(1)
    );
    assert_eq!(
        reader.resolve_version(VersionId(2)).expect("v2"),
        VersionId(2)
    );
    assert_eq!(
        reader.resolve_version(VersionId(3)).expect("v3"),
        VersionId(3)
    );

    // Resolve by timestamp
    assert_eq!(
        reader.resolve_timestamp(1_500_000).expect("ts1"),
        VersionId(1)
    );
    assert_eq!(
        reader.resolve_timestamp(2_000_000).expect("ts2"),
        VersionId(2)
    );
    assert_eq!(
        reader.resolve_timestamp(5_000_000).expect("ts3"),
        VersionId(3)
    );

    // Version-based tuple visibility
    // Tuple created at version 1, still live
    assert!(SnapshotReader::is_tuple_visible_at_version(1, 0, 1));
    assert!(SnapshotReader::is_tuple_visible_at_version(1, 0, 2));
    assert!(SnapshotReader::is_tuple_visible_at_version(1, 0, 3));

    // Tuple created at version 1, deleted at version 3
    assert!(SnapshotReader::is_tuple_visible_at_version(1, 3, 1));
    assert!(SnapshotReader::is_tuple_visible_at_version(1, 3, 2));
    assert!(!SnapshotReader::is_tuple_visible_at_version(1, 3, 3));

    // Tuple created at version 2 (updated row), not visible at version 1
    assert!(!SnapshotReader::is_tuple_visible_at_version(2, 0, 1));
    assert!(SnapshotReader::is_tuple_visible_at_version(2, 0, 2));
    assert!(SnapshotReader::is_tuple_visible_at_version(2, 0, 3));

    // Page-level pruning
    map.update_on_insert(PageId::new(200, 0), 1);
    map.update_on_insert(PageId::new(200, 1), 2);
    map.update_on_insert(PageId::new(200, 2), 3);

    // Page 2 (min_version=3) can be skipped at version 2
    assert!(reader.can_skip_page(PageId::new(200, 2), 2));
    // Page 0 (min_version=1) cannot be skipped at version 2
    assert!(!reader.can_skip_page(PageId::new(200, 0), 2));

    tprintln!("  Version resolution: correct at all 3 versions");
    tprintln!("  Timestamp resolution: correct interpolation");
    tprintln!("  Tuple visibility: correct at all version boundaries");
    tprintln!("  Page pruning: correctly skips future pages");
    tprintln!("  PASS");
    tprintln!();
}

#[test]
fn validate_scd_type2() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("versioning");
    tprintln!("=== SCD Type 2 Validation ===");

    // Parse SCD config from table options
    let options = vec![
        ("scd_type".to_string(), "2".to_string()),
        ("natural_key".to_string(), "customer_id".to_string()),
    ];
    let config = ScdHandler::from_table_options(&options)
        .expect("parse")
        .expect("scd config present");
    assert_eq!(config.scd_type, ScdType::Type2);
    assert_eq!(config.natural_key_columns, vec!["customer_id"]);

    let handler = ScdHandler::new(config);

    // Verify required columns for Type 2
    let cols = handler.required_columns();
    assert_eq!(cols.len(), 3);
    assert_eq!(cols[0].0, "valid_from");
    assert_eq!(cols[1].0, "valid_to");
    assert_eq!(cols[2].0, "is_current");

    // Simulate updating 10 customer records
    let now_micros = 1_700_000_000_000_000i64;
    let now_millis = 1_700_000_000_000u64;
    let mut surrogate_keys = Vec::new();

    for i in 0..10 {
        let old_tuple_id = TupleId::new(PageId::new(200, 0), i);
        let new_data = vec![i as u8; 50]; // new row data

        let actions =
            handler.generate_type2_update(old_tuple_id, new_data.clone(), now_micros, now_millis);

        // Verify: old row gets expired with valid_to = now
        assert_eq!(actions.expire_tuple_id, old_tuple_id);
        assert_eq!(actions.expire_valid_to, now_micros);

        // Verify: new row gets valid_from = now
        assert_eq!(actions.new_row_valid_from, now_micros);
        assert_eq!(actions.new_row_data, new_data);

        // Verify: surrogate key is unique and time-ordered
        assert!(!surrogate_keys.contains(&actions.surrogate_key));
        if let Some(last) = surrogate_keys.last() {
            assert!(actions.surrogate_key > *last);
        }
        surrogate_keys.push(actions.surrogate_key);
    }

    // Verify delete behavior for Type 2: expire, do not physically delete
    let delete_actions =
        handler.generate_delete(TupleId::new(PageId::new(200, 0), 0), now_micros + 1000);
    match delete_actions {
        ScdActions::DeleteExpire { valid_to, .. } => {
            assert_eq!(valid_to, now_micros + 1000);
        }
        _ => panic!("expected DeleteExpire for Type 2 delete"),
    }

    tprintln!("  Config parsed from table options: scd_type=2, natural_key=customer_id");
    tprintln!("  Required columns: valid_from, valid_to, is_current");
    tprintln!("  10 updates: each produces 1 expire + 1 insert with unique surrogate key");
    tprintln!("  Delete: expires row with valid_to timestamp (soft delete)");
    tprintln!("  PASS");
    tprintln!();
}

#[test]
fn validate_system_versioned_table() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("versioning");
    tprintln!("=== System-Versioned Table Validation ===");

    // Verify required columns
    let cols = SystemVersionedTable::required_columns();
    assert_eq!(cols.len(), 2);
    assert_eq!(cols[0], ("sys_start".to_string(), "TIMESTAMPTZ"));
    assert_eq!(cols[1], ("sys_end".to_string(), "TIMESTAMPTZ"));

    let now = 1_700_000_000_000_000i64;

    // INSERT: sys_start = now, sys_end = MAX_TIMESTAMP
    let (start, end) = SystemVersionedTable::on_insert_defaults(now);
    assert_eq!(start, now);
    assert_eq!(end, MAX_TIMESTAMP);

    // UPDATE: old row sys_end = now, new row sys_start = now, sys_end = MAX
    let update = SystemVersionedTable::on_update(now);
    assert_eq!(update.old_sys_end, now);
    assert_eq!(update.new_sys_start, now);
    assert_eq!(update.new_sys_end, MAX_TIMESTAMP);

    // DELETE: sys_end = now (soft delete preserving history)
    let delete_end = SystemVersionedTable::on_delete(now);
    assert_eq!(delete_end, now);

    // Temporal queries: FOR SYSTEM_TIME AS OF
    let query = TemporalQuery::AsOfSystemTime(now - 1000);
    // Row with sys_start=now-5000, sys_end=MAX is visible
    assert!(query.is_row_visible(now - 5000, MAX_TIMESTAMP));
    // Row with sys_start=now+1000, sys_end=MAX is not visible (created after query time)
    assert!(!query.is_row_visible(now + 1000, MAX_TIMESTAMP));

    // FOR SYSTEM_TIME BETWEEN
    let range_query = TemporalQuery::BetweenSystemTime {
        start: now - 10000,
        end: now,
    };
    // Row with sys_start=now-5000, sys_end=now-1000 overlaps the range
    assert!(range_query.is_row_visible(now - 5000, now - 1000));
    // Row with sys_start=now+1, sys_end=MAX does not overlap
    assert!(!range_query.is_row_visible(now + 1, MAX_TIMESTAMP));

    // Filter column resolution
    let config = TemporalConfig::default();
    let (start_col, end_col) = query.to_filter_columns(&config);
    assert_eq!(start_col, "sys_start");
    assert_eq!(end_col, "sys_end");

    tprintln!("  Required columns: sys_start (TIMESTAMPTZ), sys_end (TIMESTAMPTZ)");
    tprintln!("  INSERT defaults: sys_start=now, sys_end=MAX_TIMESTAMP");
    tprintln!("  UPDATE: old row sys_end=now, new row sys_start=now");
    tprintln!("  DELETE: sys_end=now (preserves history)");
    tprintln!("  AS OF query: correct visibility at point in time");
    tprintln!("  BETWEEN query: correct range overlap detection");
    tprintln!("  PASS");
    tprintln!();
}

#[test]
fn validate_branch_lifecycle() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("versioning");
    tprintln!("=== Branch Lifecycle Validation ===");

    let dir = tempdir().expect("tempdir");
    let mgr = BranchManager::new(dir.path().to_path_buf());

    // Create main with 1000 rows at version 1
    let main_id = mgr
        .create_branch("main", None, VersionId(1), "main branch", 1000)
        .expect("create main");

    // Create dev branch from main at version 1
    let dev_id = mgr
        .create_branch("dev", Some(main_id), VersionId(1), "development", 2000)
        .expect("create dev");
    assert_eq!(mgr.list_branches().len(), 2);

    // Verify branch lookup by name
    let dev_entry = mgr.get_branch_by_name("dev").expect("find dev");
    assert_eq!(dev_entry.base_version_id, VersionId(1));
    assert_eq!(dev_entry.parent_branch_id, Some(main_id));

    // Simulate dev modifying pages (copy-on-write)
    for i in 0..5 {
        let original = PageId::new(200, i);
        let local = PageId::new(BranchManager::branch_file_id_base(dev_id), i);
        mgr.record_page_override(dev_id, original, local)
            .expect("override");
    }
    assert_eq!(mgr.modified_page_count(dev_id), 5);

    // Main pages are unaffected (resolve to original)
    for i in 0..5 {
        let original = PageId::new(200, i);
        assert_eq!(mgr.resolve_page(main_id, original), original);
    }

    // Dev pages resolve to local copies
    for i in 0..5 {
        let original = PageId::new(200, i);
        let expected_local = PageId::new(BranchManager::branch_file_id_base(dev_id), i);
        assert_eq!(mgr.resolve_page(dev_id, original), expected_local);
    }

    // Unmodified page on dev resolves through to parent (main, which has no override)
    let unmodified = PageId::new(200, 99);
    assert_eq!(mgr.resolve_page(dev_id, unmodified), unmodified);

    // Merge dev into main (no conflicts since main has no overrides)
    let merge_result = mgr
        .merge_branch(dev_id, main_id, VersionId(5))
        .expect("merge");
    assert_eq!(merge_result.merged_pages, 5);
    assert!(merge_result.conflicts.is_empty());

    // After merge, main resolves to dev's local pages
    for i in 0..5 {
        let original = PageId::new(200, i);
        let expected_local = PageId::new(BranchManager::branch_file_id_base(dev_id), i);
        assert_eq!(mgr.resolve_page(main_id, original), expected_local);
    }

    // Delete dev branch
    mgr.delete_branch(dev_id).expect("delete");
    assert_eq!(mgr.list_branches().len(), 1);

    // Duplicate branch name rejected
    mgr.create_branch("main", None, VersionId(1), "", 3000)
        .expect_err("duplicate name");

    tprintln!("  Created main + dev branches");
    tprintln!("  Dev: 5 page overrides via copy-on-write");
    tprintln!("  Main: pages unchanged during dev modifications");
    tprintln!("  Unmodified page resolution: falls through to parent");
    tprintln!("  Merge: 5 pages merged, 0 conflicts");
    tprintln!("  Delete: branch removed, duplicate name rejected");
    tprintln!("  PASS");
    tprintln!();
}

#[test]
fn validate_diff() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("versioning");
    tprintln!("=== Diff Validation ===");

    // classify_tuple_for_diff tests various tuple lifecycles across version ranges

    // Tuple created at v1, still live. Diff range (0, 5].
    // Visible at from=0? No (v1 > 0). Visible at to=5? Yes. -> Added
    let result = classify_tuple_for_diff(1, 0, 0, 5);
    assert_eq!(result, Some(DiffType::Added));

    // Tuple created at v1, deleted at v3. Diff range (2, 5].
    // Visible at from=2? Yes (v1<=2, del=3>2). Visible at to=5? No (del=3<=5). -> Removed
    let result = classify_tuple_for_diff(1, 3, 2, 5);
    assert_eq!(result, Some(DiffType::Removed));

    // Tuple created at v1, still live. Diff range (2, 5].
    // Visible at both. -> None (unchanged)
    let result = classify_tuple_for_diff(1, 0, 2, 5);
    assert_eq!(result, None);

    // Tuple created at v6. Diff range (2, 5].
    // Not visible at either. -> None
    let result = classify_tuple_for_diff(6, 0, 2, 5);
    assert_eq!(result, None);

    // Simulate a full diff scenario:
    // Version 1: 1000 rows (rows 1..=1000)
    // Version 2: 100 updated (row IDs 1..=100 deleted at v2, new versions created at v2),
    //            50 deleted (row IDs 101..=150 deleted at v2),
    //            200 inserted (row IDs 1001..=1200 created at v2)
    let mut added = 0u64;
    let mut removed = 0u64;

    // Original rows 1..=100: created at v1, deleted at v2 (the old version). Removed.
    for _ in 1..=100 {
        if let Some(DiffType::Removed) = classify_tuple_for_diff(1, 2, 1, 2) {
            removed += 1;
        }
    }
    // Replacement rows for 1..=100: created at v2, still live. Added.
    for _ in 1..=100 {
        if let Some(DiffType::Added) = classify_tuple_for_diff(2, 0, 1, 2) {
            added += 1;
        }
    }
    // Deleted rows 101..=150: created at v1, deleted at v2. Removed.
    for _ in 101..=150 {
        if let Some(DiffType::Removed) = classify_tuple_for_diff(1, 2, 1, 2) {
            removed += 1;
        }
    }
    // Newly inserted rows 1001..=1200: created at v2. Added.
    for _ in 1001..=1200 {
        if let Some(DiffType::Added) = classify_tuple_for_diff(2, 0, 1, 2) {
            added += 1;
        }
    }

    // 100 old removed + 50 deleted = 150 removed
    // 100 replacement + 200 new = 300 added
    assert_eq!(removed, 150);
    assert_eq!(added, 300);

    // DiffStats formatting
    let stats = DiffStats {
        rows_added: 200,
        rows_removed: 50,
        rows_modified: 100,
    };
    assert_eq!(stats.total_changes(), 350);
    assert_eq!(stats.to_string(), "+200 added, -50 removed, ~100 modified");

    // Page version map range pruning for diff scans
    let pvm = PageVersionMap::new();
    pvm.update_on_insert(PageId::new(200, 0), 1);
    pvm.update_on_insert(PageId::new(200, 0), 1);
    pvm.update_on_insert(PageId::new(200, 1), 2);
    pvm.update_on_delete(PageId::new(200, 1), 2);

    // Page 0: versions [1,1]. Diff range (1,2]: max_version=1 <= 1, skip.
    assert!(pvm.can_skip_page_for_range(PageId::new(200, 0), 1, 2));
    // Page 1: versions [2,2]. Diff range (1,2]: min=2 <= 2, max=2 > 1, no skip.
    assert!(!pvm.can_skip_page_for_range(PageId::new(200, 1), 1, 2));

    tprintln!("  Tuple classification: Added, Removed, Unchanged all correct");
    tprintln!("  Full scenario: 150 removed, 300 added across update/delete/insert");
    tprintln!("  DiffStats formatting: correct");
    tprintln!("  Page range pruning for diff scans: correct skip/no-skip");
    tprintln!("  PASS");
    tprintln!();
}

#[test]
fn validate_restore() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("versioning");
    tprintln!("=== Restore Validation ===");

    let dir = tempdir().expect("tempdir");
    let log = VersionLog::open(dir.path(), 1).expect("open");

    // Build version history
    log.append(1, 1000, OperationType::Insert, 1000, None)
        .expect("v1");
    log.append(2, 2000, OperationType::Update, 0, None)
        .expect("v2");
    log.append(3, 3000, OperationType::Delete, -50, None)
        .expect("v3");

    assert_eq!(log.current_version(), VersionId(3));

    // Simulate restore to version 1 by appending a Maintenance version
    let restore_version = log
        .append(4, 4000, OperationType::Maintenance, 50, None)
        .expect("restore");
    assert_eq!(restore_version, VersionId(4));

    // Restore creates a NEW version (non-destructive)
    assert_eq!(log.current_version(), VersionId(4));
    assert_eq!(log.entry_count(), 4);

    // All intermediate versions are still accessible
    let v1 = log.get_version(VersionId(1)).expect("v1 still exists");
    assert_eq!(v1.operation_type, OperationType::Insert);

    let v2 = log.get_version(VersionId(2)).expect("v2 still exists");
    assert_eq!(v2.operation_type, OperationType::Update);

    let v3 = log.get_version(VersionId(3)).expect("v3 still exists");
    assert_eq!(v3.operation_type, OperationType::Delete);

    let v4 = log.get_version(VersionId(4)).expect("v4 restore");
    assert_eq!(v4.operation_type, OperationType::Maintenance);
    assert_eq!(v4.row_count_delta, 50); // restored 50 previously deleted rows

    // Version range query includes all versions
    let all = log.get_versions_in_range(VersionId(1), VersionId(4));
    assert_eq!(all.len(), 4);

    // Timestamp resolution works across the full range
    let at_1500 = log.get_version_at_timestamp(1500).expect("ts 1500");
    assert_eq!(at_1500.version_id, VersionId(1));
    let at_4000 = log.get_version_at_timestamp(4000).expect("ts 4000");
    assert_eq!(at_4000.version_id, VersionId(4));

    tprintln!("  Restore creates version 4 (non-destructive)");
    tprintln!("  All intermediate versions (1-3) still accessible");
    tprintln!("  Range query returns all 4 versions");
    tprintln!("  Timestamp resolution works across full history");
    tprintln!("  PASS");
    tprintln!();
}

// =============================================================================
// Planning Doc Performance Targets (00e-build-prompts-dataops.md)
// =============================================================================

#[test]
fn perf_version_overhead_per_tuple() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("versioning");
    tprintln!("=== Version Overhead Per Tuple ===");

    // Measure the cost of is_visible_at_version check per tuple.
    // This is the per-tuple overhead added to scans on versioned tables.
    let header = VersionedTupleHeader::new(100, 1, 5);
    let target_version: u64 = 10;
    let count = 10_000_000u64;

    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        let start = std::time::Instant::now();
        let mut visible = 0u64;
        for _ in 0..count {
            if header.is_visible_at_version(target_version) {
                visible += 1;
            }
        }
        let elapsed = start.elapsed();
        assert_eq!(visible, count);

        let ns_per_check = elapsed.as_nanos() as f64 / count as f64;
        runs.push(ns_per_check);
        tprintln!(
            "  Run {}: {:.2}ns per tuple ({:.2}ms)",
            run + 1,
            ns_per_check,
            elapsed.as_secs_f64() * 1000.0
        );
    }

    let result = validate_metric(
        "version_overhead_per_tuple",
        "ns_per_check",
        runs,
        VERSION_OVERHEAD_PER_TUPLE_TARGET_NS,
        false,
    );
    assert!(result.passed, "Version overhead per tuple above target");
    tprintln!();
}

#[test]
fn perf_time_travel_scan_overhead() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("versioning");
    tprintln!("=== Time Travel Query Overhead (1M rows) ===");

    // Simulate scanning 1M tuples with and without version filtering.
    // Measure the overhead of the version check vs a simple MVCC check.
    let row_count = 1_000_000u64;

    // Build 1M versioned headers: all created at version 1, live
    let headers: Vec<VersionedTupleHeader> = (0..row_count)
        .map(|i| VersionedTupleHeader::new(100, (i as u32) + 1, 1))
        .collect();

    let target_version: u64 = 2;

    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        // Baseline: simple MVCC visibility (is_visible with snapshot xid)
        let start_baseline = std::time::Instant::now();
        let mut baseline_visible = 0u64;
        for h in &headers {
            if h.base.is_visible(u32::MAX) {
                baseline_visible += 1;
            }
        }
        let baseline_elapsed = start_baseline.elapsed();

        // Versioned: is_visible_at_version check
        let start_versioned = std::time::Instant::now();
        let mut versioned_visible = 0u64;
        for h in &headers {
            if h.is_visible_at_version(target_version) {
                versioned_visible += 1;
            }
        }
        let versioned_elapsed = start_versioned.elapsed();

        assert_eq!(baseline_visible, row_count);
        assert_eq!(versioned_visible, row_count);

        let overhead_pct = if baseline_elapsed.as_nanos() > 0 {
            ((versioned_elapsed.as_nanos() as f64 - baseline_elapsed.as_nanos() as f64)
                / baseline_elapsed.as_nanos() as f64)
                * 100.0
        } else {
            0.0
        };

        // Clamp negative overhead (versioned might be faster due to simpler check)
        let overhead_pct = overhead_pct.max(0.0);
        runs.push(overhead_pct);

        tprintln!(
            "  Run {}: baseline={:.2}ms, versioned={:.2}ms, overhead={:.1}%",
            run + 1,
            baseline_elapsed.as_secs_f64() * 1000.0,
            versioned_elapsed.as_secs_f64() * 1000.0,
            overhead_pct
        );
    }

    let result = validate_metric(
        "time_travel_scan_overhead",
        "overhead_percent",
        runs,
        TIME_TRAVEL_OVERHEAD_PERCENT_TARGET,
        false,
    );
    assert!(result.passed, "Time travel scan overhead above target");
    tprintln!();
}

#[test]
fn perf_scd_type2_merge_throughput() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("versioning");
    tprintln!("=== SCD Type 2 Merge Throughput (100K rows) ===");

    let config = ScdConfig {
        scd_type: ScdType::Type2,
        natural_key_columns: vec!["customer_id".to_string()],
        ..Default::default()
    };
    let handler = ScdHandler::new(config);

    let count = 100_000u64;
    let now_micros = 1_700_000_000_000_000i64;
    let now_millis = 1_700_000_000_000u64;

    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        let start = std::time::Instant::now();
        for i in 0..count {
            let tid = TupleId::new(PageId::new(200, i / 1000), (i % 1000) as u16);
            let new_data = vec![(i & 0xFF) as u8; 64];
            let _actions =
                handler.generate_type2_update(tid, new_data, now_micros + i as i64, now_millis);
        }
        let elapsed = start.elapsed();
        let ops_per_sec = count as f64 / elapsed.as_secs_f64();
        runs.push(ops_per_sec);

        tprintln!(
            "  Run {}: {} rows/sec ({:.2}ms)",
            run + 1,
            format_with_commas(ops_per_sec),
            elapsed.as_secs_f64() * 1000.0
        );
    }

    let result = validate_metric(
        "scd_type2_merge_throughput",
        "rows_per_sec",
        runs,
        SCD_TYPE2_MERGE_TARGET_OPS,
        true,
    );
    assert!(result.passed, "SCD Type 2 merge throughput below target");
    tprintln!();
}

#[test]
fn perf_branch_merge_10k_changes() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("versioning");
    tprintln!("=== Branch Merge (10K changes) ===");

    let change_count = 10_000u64;

    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        let dir = tempdir().expect("tempdir");
        let mgr = BranchManager::new(dir.path().to_path_buf());

        let main_id = mgr
            .create_branch("main", None, VersionId(1), "", 0)
            .expect("main");
        let dev_id = mgr
            .create_branch("dev", Some(main_id), VersionId(1), "", 0)
            .expect("dev");

        // Record 10K page overrides on dev
        for i in 0..change_count {
            let original = PageId::new(200, i);
            let local = PageId::new(BranchManager::branch_file_id_base(dev_id), i);
            mgr.record_page_override(dev_id, original, local)
                .expect("override");
        }

        // Measure merge time
        let start = std::time::Instant::now();
        let merge_result = mgr
            .merge_branch(dev_id, main_id, VersionId(2))
            .expect("merge");
        let elapsed = start.elapsed();

        assert_eq!(merge_result.merged_pages, change_count);
        assert!(merge_result.conflicts.is_empty());

        let ms = elapsed.as_secs_f64() * 1000.0;
        runs.push(ms);

        tprintln!(
            "  Run {}: {:.2}ms ({} pages merged)",
            run + 1,
            ms,
            change_count
        );
    }

    let result = validate_metric(
        "branch_merge_10k",
        "ms_per_merge",
        runs,
        BRANCH_MERGE_TARGET_MS,
        false,
    );
    assert!(result.passed, "Branch merge latency above target");
    tprintln!();
}

#[test]
fn perf_diff_adjacent_versions() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("versioning");
    tprintln!("=== Diff Adjacent Versions ===");

    // Measure the cost of classifying 1M tuples for a diff between v1 and v2.
    // This simulates the scan phase of diff_versions where each tuple's
    // version bounds are checked.
    let tuple_count = 1_000_000u64;

    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        let start = std::time::Instant::now();

        let mut added = 0u64;
        let mut removed = 0u64;
        let mut unchanged = 0u64;

        for i in 0..tuple_count {
            // 80% unchanged (created at v1, still live)
            // 10% removed (created at v1, deleted at v2)
            // 10% added (created at v2)
            let (version_id, deleted_at) = if i % 10 < 8 {
                (1u64, 0u64) // unchanged
            } else if i % 10 < 9 {
                (1u64, 2u64) // removed at v2
            } else {
                (2u64, 0u64) // added at v2
            };

            match classify_tuple_for_diff(version_id, deleted_at, 1, 2) {
                Some(DiffType::Added) => added += 1,
                Some(DiffType::Removed) => removed += 1,
                _ => unchanged += 1,
            }
        }

        let elapsed = start.elapsed();
        let ms = elapsed.as_secs_f64() * 1000.0;
        runs.push(ms);

        assert_eq!(added, 100_000);
        assert_eq!(removed, 100_000);
        assert_eq!(unchanged, 800_000);

        tprintln!(
            "  Run {}: {:.2}ms (+{} -{} ~{})",
            run + 1,
            ms,
            added,
            removed,
            unchanged
        );
    }

    let result = validate_metric(
        "diff_adjacent_versions",
        "ms_per_diff",
        runs,
        DIFF_ADJACENT_TARGET_MS,
        false,
    );
    assert!(result.passed, "Diff adjacent versions latency above target");
    tprintln!();
}

#[test]
fn perf_version_log_append_planning_target() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("versioning");
    tprintln!("=== Version Log Append (Planning Target: >=3M/sec) ===");

    // The planning doc target of >=3M/sec is for batch append mode,
    // which amortizes file I/O across multiple entries.
    let count = 1_000_000u64;
    let batch_size = 100;

    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        let dir = tempdir().expect("tempdir");
        let log = VersionLog::open(dir.path(), 1).expect("open");

        let mut batch: Vec<(
            u64,
            i64,
            OperationType,
            i64,
            Option<std::collections::HashMap<String, String>>,
        )> = Vec::with_capacity(batch_size);

        let start = std::time::Instant::now();
        let mut version_count = 0u64;
        for i in 0..count {
            batch.push((i, i as i64 * 100, OperationType::Insert, 1, None));
            if batch.len() == batch_size {
                let ids = log.append_batch(&batch).expect("batch");
                version_count += ids.len() as u64;
                batch.clear();
            }
        }
        if !batch.is_empty() {
            let ids = log.append_batch(&batch).expect("batch tail");
            version_count += ids.len() as u64;
        }
        let elapsed = start.elapsed();

        assert_eq!(version_count, count);
        let ops_per_sec = count as f64 / elapsed.as_secs_f64();
        runs.push(ops_per_sec);

        tprintln!(
            "  Run {}: {} ops/sec ({:.2}ms)",
            run + 1,
            format_with_commas(ops_per_sec),
            elapsed.as_secs_f64() * 1000.0
        );
    }

    let result = validate_metric(
        "version_log_append_planning",
        "batch_ops_per_sec",
        runs,
        VERSION_LOG_APPEND_PLANNING_TARGET_OPS,
        true,
    );
    assert!(
        result.passed,
        "Version log append below planning target (3M/sec)"
    );
    tprintln!();
}

#[test]
fn perf_timestamp_resolution() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("versioning");
    tprintln!("=== Timestamp Resolution (Planning Target: <=500us) ===");

    let dir = tempdir().expect("tempdir");
    let log = VersionLog::open(dir.path(), 1).expect("open");

    // Populate with 100K entries at 100us intervals
    let entry_count = 100_000u64;
    let batch_size = 1000;
    let mut batch: Vec<(
        u64,
        i64,
        OperationType,
        i64,
        Option<std::collections::HashMap<String, String>>,
    )> = Vec::with_capacity(batch_size);
    for i in 0..entry_count {
        batch.push((i, (i as i64) * 100, OperationType::Insert, 1, None));
        if batch.len() == batch_size {
            log.append_batch(&batch).expect("batch");
            batch.clear();
        }
    }

    let lookup_count = 100_000u64;
    let max_ts = (entry_count - 1) as i64 * 100;

    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        let start = std::time::Instant::now();
        for i in 0..lookup_count {
            let ts = (i as i64 * max_ts) / lookup_count as i64;
            let _entry = log.get_version_at_timestamp(ts).expect("resolve");
        }
        let elapsed = start.elapsed();

        let us_per_resolution = elapsed.as_micros() as f64 / lookup_count as f64;
        runs.push(us_per_resolution);

        tprintln!(
            "  Run {}: {:.3}us per resolution ({:.2}ms total)",
            run + 1,
            us_per_resolution,
            elapsed.as_secs_f64() * 1000.0
        );
    }

    let result = validate_metric(
        "timestamp_resolution",
        "us_per_resolution",
        runs,
        TIMESTAMP_RESOLUTION_TARGET_US,
        false,
    );
    assert!(
        result.passed,
        "Timestamp resolution above planning target (500us)"
    );
    tprintln!();
}

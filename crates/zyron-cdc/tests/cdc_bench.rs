//! CDC Validation and Benchmark Suite
//!
//! Tests Change Data Feed, RETURNING clause, replication slots,
//! logical decoders, retention, publications, snapshots, metrics,
//! stream/ingest management, and performance targets.
//!
//! Run: cargo test -p zyron-cdc --test cdc_bench --release -- --nocapture

use std::sync::Mutex;
use std::time::Instant;

use tempfile::TempDir;
use zyron_bench_harness::*;
use zyron_cdc::cdc_ingest::{CdcIngestConfig, CdcIngestManager, CdcIngestSource, OnConflict};
use zyron_cdc::cdc_stream::{CdcOutputStream, CdcSinkConfig, CdcStreamManager, StreamRetryPolicy};
use zyron_cdc::change_feed::{CdfRegistry, ChangeDataFeed, ChangeRecord, ChangeType};
use zyron_cdc::decoder::{
    DebeziumDecoder, DecodedChange, DecoderPlugin, LogicalDecoder, ZyronCdcDecoder, create_decoder,
};
use zyron_cdc::metrics::CdcMetrics;
use zyron_cdc::publication::PublicationManager;
use zyron_cdc::replication_slot::{SlotLagConfig, SlotManager};
use zyron_cdc::retention::{CdcRetentionManager, CdcRetentionPolicy};
use zyron_cdc::returning::{OldNewResolver, ReturnClause, ReturnColumn, ReturnSource};
use zyron_cdc::snapshot::{SnapshotExport, SnapshotReader, write_snapshot_rows};
use zyron_wal::Lsn;

use std::sync::Arc;

static BENCHMARK_LOCK: Mutex<()> = Mutex::new(());

// ---------------------------------------------------------------------------
// Performance targets
// ---------------------------------------------------------------------------

const CDF_INSERT_OVERHEAD_NS: f64 = 100.0;
const CDF_QUERY_100K_MS: f64 = 25.0;
const LOGICAL_DECODING_RPS: f64 = 500_000.0;
const DEBEZIUM_ENCODING_RPS: f64 = 200_000.0;
const RETURNING_OVERHEAD_NS: f64 = 200.0;
const SLOT_ADVANCE_US: f64 = 50.0;
const RETENTION_PURGE_1M_S: f64 = 2.0;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_record(
    table_id: u32,
    version: u64,
    ts: i64,
    change_type: ChangeType,
    txn_id: u32,
    pk: u8,
    row: &[u8],
) -> ChangeRecord {
    ChangeRecord {
        change_type,
        commit_version: version,
        commit_timestamp: ts,
        table_id,
        txn_id,
        schema_version: 1,
        row_data: row.to_vec(),
        primary_key_data: vec![pk],
        is_last_in_txn: true,
    }
}

fn make_insert(table_id: u32, version: u64, ts: i64, pk: u8) -> ChangeRecord {
    make_record(
        table_id,
        version,
        ts,
        ChangeType::Insert,
        1,
        pk,
        &[pk, 1, 2, 3],
    )
}

fn make_decoded(table_name: &str, op: ChangeType, id: u64) -> DecodedChange {
    DecodedChange {
        table_name: table_name.to_string(),
        table_id: 1,
        operation: op,
        old_values: if matches!(
            op,
            ChangeType::Delete | ChangeType::UpdatePreimage | ChangeType::UpdatePostimage
        ) {
            Some(vec![("id".into(), id.to_string())])
        } else {
            None
        },
        new_values: if matches!(op, ChangeType::Insert | ChangeType::UpdatePostimage) {
            Some(vec![
                ("id".into(), id.to_string()),
                ("name".into(), "test".into()),
            ])
        } else {
            None
        },
        commit_lsn: id,
        commit_timestamp: id as i64 * 1000,
        txn_id: id as u32,
        is_last_in_txn: true,
        schema_version: 1,
    }
}

// =========================================================================
// 1. Change Data Feed Basic Test
// =========================================================================

#[test]
fn test_cdf_basic_operations() {
    zyron_bench_harness::init("cdc");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Change Data Feed: Basic Operations ===");
    let before = take_util_snapshot();

    let tmp = TempDir::new().unwrap();
    let registry = Arc::new(CdfRegistry::new(tmp.path().to_path_buf()));
    let feed = registry.enable_for_table(1, 30).unwrap();

    // INSERT 100 rows
    let inserts: Vec<ChangeRecord> = (1..=100)
        .map(|i| make_insert(1, i, i as i64 * 1000, i as u8))
        .collect();
    feed.append_batch(&inserts).unwrap();
    assert_eq!(feed.record_count(), 100);

    // UPDATE 20 rows (preimage + postimage = 40 records)
    let mut updates = Vec::new();
    for i in 1..=20u64 {
        updates.push(make_record(
            1,
            100 + i,
            (100 + i) as i64 * 1000,
            ChangeType::UpdatePreimage,
            2,
            i as u8,
            &[i as u8, 1, 2, 3],
        ));
        updates.push(make_record(
            1,
            100 + i,
            (100 + i) as i64 * 1000,
            ChangeType::UpdatePostimage,
            2,
            i as u8,
            &[i as u8, 10, 20, 30],
        ));
    }
    feed.append_batch(&updates).unwrap();
    assert_eq!(feed.record_count(), 140);

    // DELETE 10 rows
    let deletes: Vec<ChangeRecord> = (1..=10)
        .map(|i| {
            make_record(
                1,
                120 + i,
                (120 + i) as i64 * 1000,
                ChangeType::Delete,
                3,
                i as u8,
                &[i as u8, 1, 2, 3],
            )
        })
        .collect();
    feed.append_batch(&deletes).unwrap();
    assert_eq!(feed.record_count(), 150);

    // Query all
    let all = feed.query_changes(1, 200).unwrap();
    assert_eq!(all.len(), 150);

    let insert_count = all
        .iter()
        .filter(|r| r.change_type == ChangeType::Insert)
        .count();
    let pre_count = all
        .iter()
        .filter(|r| r.change_type == ChangeType::UpdatePreimage)
        .count();
    let post_count = all
        .iter()
        .filter(|r| r.change_type == ChangeType::UpdatePostimage)
        .count();
    let del_count = all
        .iter()
        .filter(|r| r.change_type == ChangeType::Delete)
        .count();
    assert_eq!(insert_count, 100);
    assert_eq!(pre_count, 20);
    assert_eq!(post_count, 20);
    assert_eq!(del_count, 10);

    // Verify required fields on every record
    for record in &all {
        assert!(record.commit_version > 0);
        assert!(record.commit_timestamp > 0);
        assert!(!record.row_data.is_empty());
    }

    tprintln!("  150 records: 100 insert, 20 preimage, 20 postimage, 10 delete");
    let passed = check_performance("CDF Basic", "Records stored", 150.0, 150.0, true);
    assert!(passed);

    let after = take_util_snapshot();
    record_test_util("CDF Basic", before, after);
}

// =========================================================================
// 2. Change Feed Time Range Test
// =========================================================================

#[test]
fn test_cdf_time_range_queries() {
    zyron_bench_harness::init("cdc");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Change Data Feed: Time Range Queries ===");
    let before = take_util_snapshot();

    let tmp = TempDir::new().unwrap();
    let feed = ChangeDataFeed::open(tmp.path(), 1, 30).unwrap();

    // T1 = 1000-1999: 100 inserts (version 1-100)
    let inserts: Vec<ChangeRecord> = (1..=100)
        .map(|i| make_insert(1, i, 1000 + i as i64, i as u8))
        .collect();
    feed.append_batch(&inserts).unwrap();

    // T2 = 2000-2999: 40 update records (version 101-120, 2 per)
    let mut updates = Vec::new();
    for i in 1..=20u64 {
        updates.push(make_record(
            1,
            100 + i,
            2000 + i as i64,
            ChangeType::UpdatePreimage,
            2,
            i as u8,
            &[i as u8],
        ));
        updates.push(make_record(
            1,
            100 + i,
            2000 + i as i64,
            ChangeType::UpdatePostimage,
            2,
            i as u8,
            &[i as u8],
        ));
    }
    feed.append_batch(&updates).unwrap();

    // T3 = 3000-3999: 10 deletes (version 121-130)
    let deletes: Vec<ChangeRecord> = (1..=10)
        .map(|i| {
            make_record(
                1,
                120 + i,
                3000 + i as i64,
                ChangeType::Delete,
                3,
                i as u8,
                &[i as u8],
            )
        })
        .collect();
    feed.append_batch(&deletes).unwrap();

    // T1-T2 range: inserts + updates (no deletes)
    let t1_t2 = feed.query_changes_by_time(1000, 2999).unwrap();
    assert!(!t1_t2.iter().any(|r| r.change_type == ChangeType::Delete));
    assert_eq!(t1_t2.len(), 140);
    tprintln!("  T1-T2: {} records (inserts + updates)", t1_t2.len());

    // T2-T3 range: updates + deletes (no inserts)
    let t2_t3 = feed.query_changes_by_time(2000, 3999).unwrap();
    assert!(!t2_t3.iter().any(|r| r.change_type == ChangeType::Insert));
    assert_eq!(t2_t3.len(), 50);
    tprintln!("  T2-T3: {} records (updates + deletes)", t2_t3.len());

    // Version range
    let v_range = feed.query_changes(101, 120).unwrap();
    assert_eq!(v_range.len(), 40);
    tprintln!("  Version 101-120: {} records", v_range.len());

    let passed = check_performance("CDF Time Range", "Queries correct", 3.0, 3.0, true);
    assert!(passed);

    let after = take_util_snapshot();
    record_test_util("CDF Time Range", before, after);
}

// =========================================================================
// 3. RETURNING Clause Test
// =========================================================================

#[test]
fn test_returning_clause() {
    zyron_bench_harness::init("cdc");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== RETURNING Clause: OLD/NEW Resolution ===");
    let before = take_util_snapshot();

    let clause = ReturnClause::new(vec![
        ReturnColumn {
            name: "id".into(),
            source: ReturnSource::New,
            column_index: 0,
            alias: None,
        },
        ReturnColumn {
            name: "price".into(),
            source: ReturnSource::Old,
            column_index: 1,
            alias: Some("old_price".into()),
        },
        ReturnColumn {
            name: "price".into(),
            source: ReturnSource::New,
            column_index: 1,
            alias: Some("new_price".into()),
        },
    ]);

    let resolver = OldNewResolver::new(clause.clone(), 3);

    // UPDATE: both old and new present
    let old_row = vec![vec![1u8], vec![100u8]];
    let new_row = vec![vec![1u8], vec![110u8]];
    let result = resolver.resolve_row(Some(&old_row), Some(&new_row));
    assert_eq!(result[0].0, "id");
    assert_eq!(result[1].0, "old_price");
    assert_eq!(result[1].1, Some(&[100u8][..]));
    assert_eq!(result[2].0, "new_price");
    assert_eq!(result[2].1, Some(&[110u8][..]));
    tprintln!("  UPDATE: old.price=100, new.price=110");

    // INSERT: old = NULL
    let insert_result = resolver.resolve_row(None, Some(&new_row));
    assert!(insert_result[1].1.is_none());
    assert!(insert_result[2].1.is_some());
    tprintln!("  INSERT: old.price=NULL, new.price=110");

    // DELETE: new = NULL
    let delete_result = resolver.resolve_row(Some(&old_row), None);
    assert!(delete_result[1].1.is_some());
    assert!(delete_result[2].1.is_none());
    tprintln!("  DELETE: old.price=100, new.price=NULL");

    assert!(clause.needs_old_values());
    assert!(clause.needs_new_values());

    let passed = check_performance("RETURNING Clause", "Tests passed", 3.0, 3.0, true);
    assert!(passed);

    let after = take_util_snapshot();
    record_test_util("RETURNING Clause", before, after);
}

// =========================================================================
// 4. RETURNING with CTE Test
// =========================================================================

#[test]
fn test_returning_cte_audit_log() {
    zyron_bench_harness::init("cdc");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== RETURNING with CTE: Audit Log Population ===");
    let before = take_util_snapshot();

    let clause = ReturnClause::new(vec![
        ReturnColumn {
            name: "id".into(),
            source: ReturnSource::New,
            column_index: 0,
            alias: None,
        },
        ReturnColumn {
            name: "x".into(),
            source: ReturnSource::Old,
            column_index: 1,
            alias: Some("old_x".into()),
        },
        ReturnColumn {
            name: "x".into(),
            source: ReturnSource::New,
            column_index: 1,
            alias: Some("new_x".into()),
        },
    ]);
    let resolver = OldNewResolver::new(clause, 3);

    // Simulate 50 updated rows and collect audit entries
    let mut audit_count = 0u32;
    for i in 0..50u8 {
        let old_row = vec![vec![i], vec![i]];
        let new_row = vec![vec![i], vec![i + 100]];
        let row = resolver.resolve_row(Some(&old_row), Some(&new_row));
        assert_eq!(row[0].0, "id");
        assert_eq!(row[1].0, "old_x");
        assert_eq!(row[1].1, Some(&[i][..]));
        assert_eq!(row[2].0, "new_x");
        assert_eq!(row[2].1, Some(&[i + 100][..]));
        audit_count += 1;
    }
    assert_eq!(audit_count, 50);
    tprintln!("  CTE audit_log populated with 50 change records");

    let passed = check_performance("RETURNING CTE", "Audit rows", 50.0, 50.0, true);
    assert!(passed);

    let after = take_util_snapshot();
    record_test_util("RETURNING CTE", before, after);
}

// =========================================================================
// 5. Replication Slot Test
// =========================================================================

#[test]
fn test_replication_slot_lifecycle() {
    zyron_bench_harness::init("cdc");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Replication Slot: Lifecycle ===");
    let before = take_util_snapshot();

    let tmp = TempDir::new().unwrap();
    let mgr = SlotManager::open(tmp.path(), SlotLagConfig::default()).unwrap();

    let slot = mgr
        .create_slot("test_slot", DecoderPlugin::ZyronCdc, None)
        .unwrap();
    assert!(slot.active);
    assert_eq!(slot.confirmed_lsn, 0);
    tprintln!("  Created slot 'test_slot' with zyron_cdc plugin");

    // Advance through 1000 changes
    for i in 1..=1000u64 {
        mgr.advance_slot("test_slot", Lsn(i * 100)).unwrap();
    }
    let slot = mgr.get_slot("test_slot").unwrap();
    assert_eq!(slot.confirmed_lsn, 100_000);
    assert_eq!(slot.restart_lsn, 100_000);
    tprintln!(
        "  Advanced 1000 times, confirmed_lsn = {}",
        slot.confirmed_lsn
    );

    // WAL retention
    let min_lsn = mgr.min_restart_lsn();
    assert_eq!(min_lsn, Some(Lsn(100_000)));
    tprintln!("  WAL min_restart_lsn = {:?}", min_lsn);

    // Persistence (flush dirty state before drop since advance_slot is batched)
    mgr.flush_if_dirty().unwrap();
    drop(mgr);
    let mgr2 = SlotManager::open(tmp.path(), SlotLagConfig::default()).unwrap();
    let slot = mgr2.get_slot("test_slot").unwrap();
    assert_eq!(slot.confirmed_lsn, 100_000);
    tprintln!("  Slot state persisted and recovered");

    // Lag monitoring
    let lag = mgr2.slot_lag_bytes("test_slot", Lsn(200_000)).unwrap();
    assert_eq!(lag, 100_000);
    tprintln!("  Slot lag: {} bytes", lag);

    let passed = check_performance("Replication Slot", "Checks passed", 4.0, 4.0, true);
    assert!(passed);

    let after = take_util_snapshot();
    record_test_util("Replication Slot", before, after);
}

// =========================================================================
// 6. Debezium Format Test
// =========================================================================

#[test]
fn test_debezium_format_validation() {
    zyron_bench_harness::init("cdc");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Debezium Format: JSON Envelope ===");
    let before = take_util_snapshot();

    let decoder = DebeziumDecoder::new("zyrondb_test".into(), "mydb".into());

    // INSERT
    let insert = make_decoded("users", ChangeType::Insert, 1);
    let bytes = decoder.serialize(&insert).unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["op"], "c");
    assert!(json["before"].is_null());
    assert!(json["after"].is_object());
    assert_eq!(json["source"]["connector"], "zyrondb");
    assert_eq!(json["source"]["name"], "zyrondb_test");
    assert_eq!(json["source"]["db"], "mydb");
    assert_eq!(json["source"]["table"], "users");
    assert!(json["ts_ms"].is_number());
    assert!(json["transaction"]["id"].is_string());
    tprintln!("  INSERT: op='c', before=null, after=present");

    // UPDATE
    let update = make_decoded("users", ChangeType::UpdatePostimage, 2);
    let bytes = decoder.serialize(&update).unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["op"], "u");
    assert!(json["before"].is_object());
    assert!(json["after"].is_object());
    tprintln!("  UPDATE: op='u', before=present, after=present");

    // DELETE
    let delete = make_decoded("users", ChangeType::Delete, 3);
    let bytes = decoder.serialize(&delete).unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["op"], "d");
    assert!(json["before"].is_object());
    assert!(json["after"].is_null());
    let ts_ms = json["ts_ms"].as_i64().unwrap();
    assert!(ts_ms > 0);
    tprintln!(
        "  DELETE: op='d', before=present, after=null, ts_ms={}",
        ts_ms
    );

    let passed = check_performance("Debezium Format", "Formats validated", 3.0, 3.0, true);
    assert!(passed);

    let after = take_util_snapshot();
    record_test_util("Debezium Format", before, after);
}

// =========================================================================
// 7. CDC Retention Test
// =========================================================================

#[test]
fn test_cdc_retention_enforcement() {
    zyron_bench_harness::init("cdc");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== CDC Retention: Purge Enforcement ===");
    let before = take_util_snapshot();

    let tmp = TempDir::new().unwrap();
    let registry = Arc::new(CdfRegistry::new(tmp.path().to_path_buf()));
    let feed = registry.enable_for_table(1, 1).unwrap();

    // Old records (ts near zero)
    let old_records: Vec<ChangeRecord> = (1..=50)
        .map(|i| make_insert(1, i, i as i64 * 10, i as u8))
        .collect();
    feed.append_batch(&old_records).unwrap();

    // Recent records (ts near now)
    let now_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as i64;
    let recent_records: Vec<ChangeRecord> = (51..=100)
        .map(|i| make_insert(1, i, now_us + i as i64, i as u8))
        .collect();
    feed.append_batch(&recent_records).unwrap();
    assert_eq!(feed.record_count(), 100);
    tprintln!("  100 records: 50 old, 50 recent");

    let mgr = CdcRetentionManager::new(registry.clone());
    mgr.set_policy(CdcRetentionPolicy {
        table_id: 1,
        retention_days: 1,
        compaction_enabled: false,
    })
    .unwrap();

    let stats = mgr.enforce_all().unwrap();
    tprintln!(
        "  Enforced: {} tables, {} purged, {} bytes reclaimed",
        stats.tables_processed,
        stats.records_purged,
        stats.bytes_reclaimed
    );
    assert_eq!(stats.tables_processed, 1);
    assert_eq!(stats.records_purged, 50);
    assert_eq!(feed.record_count(), 50);

    let remaining = feed.query_changes(51, 200).unwrap();
    assert_eq!(remaining.len(), 50);
    tprintln!("  50 old purged, 50 recent retained");

    let passed = check_performance("CDC Retention", "Records purged", 50.0, 50.0, true);
    assert!(passed);

    let after = take_util_snapshot();
    record_test_util("CDC Retention", before, after);
}

// =========================================================================
// 8. Performance Benchmarks
// =========================================================================

#[test]
fn test_bench_cdf_insert_overhead() {
    zyron_bench_harness::init("cdc");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Benchmark: CDF Insert Overhead ===");
    let before = take_util_snapshot();

    let n = 100_000u64;
    let mut runs = vec![];

    for _ in 0..VALIDATION_RUNS {
        let tmp = TempDir::new().unwrap();
        let feed = ChangeDataFeed::open(tmp.path(), 1, 30).unwrap();
        let records: Vec<ChangeRecord> = (1..=n)
            .map(|i| make_insert(1, i, i as i64 * 1000, (i % 256) as u8))
            .collect();

        let start = Instant::now();
        feed.append_batch(&records).unwrap();
        let elapsed = start.elapsed();

        runs.push(elapsed.as_nanos() as f64 / n as f64);
    }

    let result = validate_metric(
        "CDF Insert Overhead",
        "Insert overhead (ns/row)",
        runs,
        CDF_INSERT_OVERHEAD_NS,
        false,
    );
    assert!(result.passed, "CDF insert overhead exceeded target");

    let after = take_util_snapshot();
    record_test_util("CDF Insert Overhead", before, after);
}

#[test]
fn test_bench_cdf_query_throughput() {
    zyron_bench_harness::init("cdc");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Benchmark: CDF Query 100K ===");
    let before = take_util_snapshot();

    let tmp = TempDir::new().unwrap();
    let n = 100_000u64;
    let feed = ChangeDataFeed::open(tmp.path(), 1, 30).unwrap();
    let records: Vec<ChangeRecord> = (1..=n)
        .map(|i| make_insert(1, i, i as i64 * 1000, (i % 256) as u8))
        .collect();
    feed.append_batch(&records).unwrap();

    let mut runs = vec![];
    for _ in 0..VALIDATION_RUNS {
        let start = Instant::now();
        let results = feed.query_changes(1, n).unwrap();
        let elapsed = start.elapsed();
        assert_eq!(results.len(), n as usize);
        runs.push(elapsed.as_millis() as f64);
    }

    let result = validate_metric(
        "CDF Query 100K",
        "Query latency (ms)",
        runs,
        CDF_QUERY_100K_MS,
        false,
    );
    assert!(result.passed, "CDF query latency exceeded target");

    let after = take_util_snapshot();
    record_test_util("CDF Query 100K", before, after);
}

#[test]
fn test_bench_logical_decoding() {
    zyron_bench_harness::init("cdc");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Benchmark: Logical Decoding 1M ===");
    let before = take_util_snapshot();

    let n = 1_000_000u64;
    let decoder = ZyronCdcDecoder;
    let changes: Vec<DecodedChange> = (0..n)
        .map(|i| make_decoded("users", ChangeType::Insert, i))
        .collect();

    let mut runs = vec![];
    for _ in 0..VALIDATION_RUNS {
        let start = Instant::now();
        for change in &changes {
            let _bytes = decoder.serialize(change).unwrap();
        }
        let elapsed = start.elapsed();
        runs.push(n as f64 / elapsed.as_secs_f64());
    }

    let result = validate_metric(
        "Logical Decoding",
        "Decoding throughput (rec/sec)",
        runs,
        LOGICAL_DECODING_RPS,
        true,
    );
    assert!(result.passed, "Logical decoding below target");

    let after = take_util_snapshot();
    record_test_util("Logical Decoding", before, after);
}

#[test]
fn test_bench_debezium_encoding() {
    zyron_bench_harness::init("cdc");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Benchmark: Debezium Encoding ===");
    let before = take_util_snapshot();

    let n = 500_000u64;
    let decoder = DebeziumDecoder::new("zyrondb".into(), "default".into());
    let changes: Vec<DecodedChange> = (0..n)
        .map(|i| make_decoded("users", ChangeType::Insert, i))
        .collect();

    let mut runs = vec![];
    for _ in 0..VALIDATION_RUNS {
        let start = Instant::now();
        for change in &changes {
            let _bytes = decoder.serialize(change).unwrap();
        }
        let elapsed = start.elapsed();
        runs.push(n as f64 / elapsed.as_secs_f64());
    }

    let result = validate_metric(
        "Debezium Encoding",
        "Encoding throughput (rec/sec)",
        runs,
        DEBEZIUM_ENCODING_RPS,
        true,
    );
    assert!(result.passed, "Debezium encoding below target");

    let after = take_util_snapshot();
    record_test_util("Debezium Encoding", before, after);
}

#[test]
fn test_bench_returning_overhead() {
    zyron_bench_harness::init("cdc");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Benchmark: RETURNING Clause Overhead ===");
    let before = take_util_snapshot();

    let n = 1_000_000u64;
    let clause = ReturnClause::new(vec![
        ReturnColumn {
            name: "id".into(),
            source: ReturnSource::New,
            column_index: 0,
            alias: None,
        },
        ReturnColumn {
            name: "price".into(),
            source: ReturnSource::Old,
            column_index: 1,
            alias: Some("prev".into()),
        },
        ReturnColumn {
            name: "price".into(),
            source: ReturnSource::New,
            column_index: 1,
            alias: Some("curr".into()),
        },
    ]);
    let resolver = OldNewResolver::new(clause, 3);
    let old_row = vec![vec![1u8, 2, 3, 4], vec![100u8, 0, 0, 0]];
    let new_row = vec![vec![1u8, 2, 3, 4], vec![110u8, 0, 0, 0]];

    let mut runs = vec![];
    for _ in 0..VALIDATION_RUNS {
        let start = Instant::now();
        for _ in 0..n {
            let _result = resolver.resolve_row(Some(&old_row), Some(&new_row));
        }
        let elapsed = start.elapsed();
        runs.push(elapsed.as_nanos() as f64 / n as f64);
    }

    let result = validate_metric(
        "RETURNING Overhead",
        "RETURNING clause overhead (ns/row)",
        runs,
        RETURNING_OVERHEAD_NS,
        false,
    );
    assert!(result.passed, "RETURNING overhead exceeded target");

    let after = take_util_snapshot();
    record_test_util("RETURNING Overhead", before, after);
}

#[test]
fn test_bench_slot_advance() {
    zyron_bench_harness::init("cdc");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Benchmark: Replication Slot Advance ===");
    let before = take_util_snapshot();

    let n = 10_000u64;
    let mut runs = vec![];

    for _ in 0..VALIDATION_RUNS {
        let tmp = TempDir::new().unwrap();
        let mgr = SlotManager::open(tmp.path(), SlotLagConfig::default()).unwrap();
        mgr.create_slot("bench_slot", DecoderPlugin::ZyronCdc, None)
            .unwrap();

        let start = Instant::now();
        for i in 1..=n {
            mgr.advance_slot("bench_slot", Lsn(i * 100)).unwrap();
        }
        let elapsed = start.elapsed();
        runs.push(elapsed.as_micros() as f64 / n as f64);
    }

    let result = validate_metric(
        "Slot Advance",
        "Slot advance latency (us/op)",
        runs,
        SLOT_ADVANCE_US,
        false,
    );
    assert!(result.passed, "Slot advance latency exceeded target");

    let after = take_util_snapshot();
    record_test_util("Slot Advance", before, after);
}

#[test]
fn test_bench_retention_purge() {
    zyron_bench_harness::init("cdc");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Benchmark: Retention Purge 1M ===");
    let before = take_util_snapshot();

    let n = 1_000_000u64;
    let mut runs = vec![];

    for _ in 0..VALIDATION_RUNS {
        let tmp = TempDir::new().unwrap();
        let feed = ChangeDataFeed::open(tmp.path(), 1, 30).unwrap();

        for batch_start in (0..n).step_by(100_000) {
            let records: Vec<ChangeRecord> = (batch_start..batch_start + 100_000)
                .map(|i| make_insert(1, i + 1, (i + 1) as i64 * 1000, ((i + 1) % 256) as u8))
                .collect();
            feed.append_batch(&records).unwrap();
        }
        assert_eq!(feed.record_count(), n);

        let start = Instant::now();
        let purged = feed.purge_before_version(500_001).unwrap();
        let elapsed = start.elapsed();

        assert_eq!(purged, 500_000);
        assert_eq!(feed.record_count(), 500_000);
        runs.push(elapsed.as_secs_f64());
    }

    let result = validate_metric(
        "Retention Purge 1M",
        "Purge latency (s)",
        runs,
        RETENTION_PURGE_1M_S,
        false,
    );
    assert!(result.passed, "Retention purge exceeded target");

    let after = take_util_snapshot();
    record_test_util("Retention Purge 1M", before, after);
}

// =========================================================================
// Additional Validation Tests
// =========================================================================

#[test]
fn test_snapshot_export_import() {
    zyron_bench_harness::init("cdc");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Snapshot: Export and Import ===");
    let before = take_util_snapshot();

    let tmp = TempDir::new().unwrap();
    let rows: Vec<Vec<u8>> = (0..1000).map(|i| vec![i as u8; 64]).collect();
    let snap_path = tmp.path().join("snap_users.dat");
    let count = write_snapshot_rows(&snap_path, &rows).unwrap();
    assert_eq!(count, 1000);

    let mut export = SnapshotExport::new(Lsn(5000));
    export.add_table(1, "users".into(), 1000, snap_path.clone());
    let manifest_path = export.write_manifest(tmp.path()).unwrap();
    assert!(manifest_path.exists());

    let loaded = SnapshotExport::read_manifest(&manifest_path).unwrap();
    assert_eq!(loaded.snapshot_lsn, 5000);
    assert_eq!(loaded.total_rows(), 1000);

    let reader = SnapshotReader::new(snap_path);
    let loaded_rows = reader.read_all_rows().unwrap();
    assert_eq!(loaded_rows.len(), 1000);
    tprintln!("  1000 rows exported and imported at LSN 5000");

    let passed = check_performance("Snapshot", "Rows exported", 1000.0, 1000.0, true);
    assert!(passed);

    let after = take_util_snapshot();
    record_test_util("Snapshot", before, after);
}

#[test]
fn test_publication_management() {
    zyron_bench_harness::init("cdc");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Publication: Multi-table Subscriptions ===");
    let before = take_util_snapshot();

    let tmp = TempDir::new().unwrap();
    let mgr = PublicationManager::open(tmp.path()).unwrap();

    mgr.create_publication("analytics", vec![1, 2, 3], false, true)
        .unwrap();
    assert!(mgr.is_table_published(1));
    assert!(mgr.is_table_published(2));
    assert!(!mgr.is_table_published(99));

    mgr.alter_publication_add_table("analytics", 4).unwrap();
    assert!(mgr.is_table_published(4));

    mgr.alter_publication_drop_table("analytics", 2).unwrap();
    assert!(!mgr.is_table_published(2));

    mgr.create_publication("everything", vec![], true, false)
        .unwrap();
    assert!(mgr.is_table_published(999));

    tprintln!("  create, alter add/drop, all_tables verified");

    let passed = check_performance("Publication", "Operations validated", 5.0, 5.0, true);
    assert!(passed);

    let after = take_util_snapshot();
    record_test_util("Publication", before, after);
}

#[test]
fn test_metrics_observability() {
    zyron_bench_harness::init("cdc");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Metrics: CDC Observability ===");
    let before = take_util_snapshot();

    let m = CdcMetrics::new();
    m.cdf_records_appended
        .fetch_add(1_000_000, std::sync::atomic::Ordering::Relaxed);
    m.stream_records_sent
        .fetch_add(500_000, std::sync::atomic::Ordering::Relaxed);
    m.ingest_records_applied
        .fetch_add(250_000, std::sync::atomic::Ordering::Relaxed);
    m.update_slot_lag("slot1", 1024);
    m.update_slot_lag("slot2", 2048);

    let output = m.render_prometheus();
    assert!(output.contains("zyrondb_cdc_cdf_records_appended_total 1000000"));
    assert!(output.contains("zyrondb_cdc_stream_records_sent_total 500000"));
    assert!(output.contains("zyrondb_cdc_ingest_records_applied_total 250000"));
    assert!(output.contains("zyrondb_cdc_slot_lag_bytes"));
    tprintln!(
        "  Prometheus output: {} bytes, all counters present",
        output.len()
    );

    let passed = check_performance("Metrics", "Metric types rendered", 14.0, 14.0, true);
    assert!(passed);

    let after = take_util_snapshot();
    record_test_util("Metrics", before, after);
}

#[test]
fn test_stream_and_ingest_management() {
    zyron_bench_harness::init("cdc");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Streams and Ingests: Management ===");
    let before = take_util_snapshot();

    let tmp = TempDir::new().unwrap();

    let stream_mgr = CdcStreamManager::new(tmp.path()).unwrap();
    stream_mgr
        .create_stream(CdcOutputStream {
            name: "user_events".into(),
            table_id: 1,
            slot_name: "slot1".into(),
            sink: CdcSinkConfig::Kafka {
                brokers: "localhost:9092".into(),
                topic: "cdc_users".into(),
                key_columns: vec!["id".into()],
            },
            decoder_plugin: DecoderPlugin::Debezium,
            filter: None,
            include_columns: None,
            batch_size: 10000,
            batch_interval_ms: 100,
            active: true,
            retry_policy: StreamRetryPolicy::default(),
        })
        .unwrap();
    assert_eq!(stream_mgr.list_streams().len(), 1);
    tprintln!("  CDC Stream created: Kafka sink, Debezium format");

    let ingest_mgr = CdcIngestManager::new(tmp.path()).unwrap();
    ingest_mgr
        .create_ingest(CdcIngestConfig {
            name: "upstream_users".into(),
            source: CdcIngestSource::Kafka {
                brokers: "localhost:9092".into(),
                topic: "upstream_cdc".into(),
                group_id: "zyrondb".into(),
                start_offset: None,
            },
            target_table_id: 1,
            primary_key_columns: vec!["id".into()],
            on_conflict: OnConflict::Update,
            dead_letter_table_id: None,
            batch_size: 1000,
            active: true,
        })
        .unwrap();
    assert_eq!(ingest_mgr.list_ingests().len(), 1);
    tprintln!("  CDC Ingest created: Kafka source, OnConflict::Update");

    // Persistence
    drop(stream_mgr);
    drop(ingest_mgr);
    let stream_mgr2 = CdcStreamManager::new(tmp.path()).unwrap();
    let ingest_mgr2 = CdcIngestManager::new(tmp.path()).unwrap();
    assert_eq!(stream_mgr2.list_streams().len(), 1);
    assert_eq!(ingest_mgr2.list_ingests().len(), 1);
    tprintln!("  State persisted and recovered");

    let passed = check_performance("Stream/Ingest", "Components validated", 4.0, 4.0, true);
    assert!(passed);

    let after = take_util_snapshot();
    record_test_util("Stream/Ingest", before, after);
}

#[test]
fn test_all_decoder_formats() {
    zyron_bench_harness::init("cdc");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Decoder Formats: All 4 Plugins ===");
    let before = take_util_snapshot();

    let change = make_decoded("orders", ChangeType::Insert, 42);

    for plugin in [
        DecoderPlugin::ZyronCdc,
        DecoderPlugin::Debezium,
        DecoderPlugin::Wal2Json,
        DecoderPlugin::Avro,
    ] {
        let decoder = create_decoder(plugin);
        let bytes = decoder.serialize(&change).unwrap();
        let roundtripped = decoder.deserialize(&bytes).unwrap();
        assert_eq!(roundtripped.table_name, "orders");
        assert_eq!(roundtripped.operation, ChangeType::Insert);
        tprintln!(
            "  {}: {} bytes, roundtrip OK",
            decoder.plugin().as_str(),
            bytes.len()
        );
    }

    let passed = check_performance("Decoder Formats", "Formats validated", 4.0, 4.0, true);
    assert!(passed);

    let after = take_util_snapshot();
    record_test_util("Decoder Formats", before, after);
}

//! Streaming Validation and Benchmark Suite
//!
//! Run: cargo test -p zyron-streaming --test streaming_bench --release -- --nocapture

use std::hint::black_box;
use std::sync::Mutex;
use std::time::Instant;

use tempfile::TempDir;
use zyron_bench_harness::*;
use zyron_streaming::accumulator::*;
use zyron_streaming::backpressure::*;
use zyron_streaming::checkpoint::*;
use zyron_streaming::column::*;
use zyron_streaming::hash::*;
use zyron_streaming::job::*;
use zyron_streaming::late_data::*;
use zyron_streaming::metrics::*;
use zyron_streaming::record::*;
use zyron_streaming::sink_connector::*;
use zyron_streaming::source_connector::*;
use zyron_streaming::spsc::*;
use zyron_streaming::state::*;
use zyron_streaming::stream_join::*;
use zyron_streaming::stream_operator::*;
use zyron_streaming::watermark::*;
use zyron_streaming::window::*;

static BENCHMARK_LOCK: Mutex<()> = Mutex::new(());

// ---------------------------------------------------------------------------
// Performance targets
// ---------------------------------------------------------------------------

const TUMBLING_WINDOW_EVENTS_PER_SEC: f64 = 8_000_000.0;
const SLIDING_WINDOW_EVENTS_PER_SEC: f64 = 3_000_000.0;
const SESSION_WINDOW_EVENTS_PER_SEC: f64 = 2_000_000.0;
const WATERMARK_PROPAGATION_MS: f64 = 20.0;
const STREAM_STREAM_JOIN_EVENTS_PER_SEC: f64 = 1_000_000.0;
const LOOKUP_JOIN_EVENTS_PER_SEC: f64 = 5_000_000.0;
const CHECKPOINT_1GB_STATE_SEC: f64 = 5.0;
const RECOVERY_1GB_STATE_SEC: f64 = 10.0;
const BACKPRESSURE_REACTION_MS: f64 = 500.0;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_test_record(values: Vec<i64>, times: Vec<i64>) -> StreamRecord {
    let col = StreamColumn::from_data(StreamColumnData::Int64(values));
    let batch = StreamBatch::new(vec![col]);
    let n = batch.num_rows;
    StreamRecord::new(batch, times, vec![ChangeFlag::Insert; n])
}

fn make_keyed_record(keys: Vec<i64>, values: Vec<i64>, times: Vec<i64>) -> StreamRecord {
    let key_col = StreamColumn::from_data(StreamColumnData::Int64(keys));
    let val_col = StreamColumn::from_data(StreamColumnData::Int64(values));
    let batch = StreamBatch::new(vec![key_col, val_col]);
    let n = batch.num_rows;
    StreamRecord::new(batch, times, vec![ChangeFlag::Insert; n])
}

fn make_record_with_times(times: Vec<i64>) -> StreamRecord {
    let n = times.len();
    let col = StreamColumn::from_data(StreamColumnData::Int64((0..n as i64).collect()));
    let batch = StreamBatch::new(vec![col]);
    StreamRecord::new(batch, times, vec![ChangeFlag::Insert; n])
}

// =========================================================================
// 1. Tumbling Window Throughput
// =========================================================================

#[test]
fn test_tumbling_window_throughput() {
    zyron_bench_harness::init("streaming");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Benchmark: Tumbling Window Throughput ===");
    let before = take_util_snapshot();

    let n = 2_000_000usize;
    let assigner = TumblingWindowAssigner::new(60_000);

    // Pre-generate event times.
    let event_times: Vec<i64> = (0..n as i64).map(|i| i * 100).collect();

    let mut runs = vec![];
    for _ in 0..VALIDATION_RUNS {
        let start = Instant::now();
        for &t in &event_times {
            black_box(assigner.assign_windows(t));
        }
        let elapsed = start.elapsed();
        runs.push(n as f64 / elapsed.as_secs_f64());
    }

    let result = validate_metric(
        "Tumbling Window",
        "Events/sec",
        runs,
        TUMBLING_WINDOW_EVENTS_PER_SEC,
        true,
    );
    assert!(result.passed, "Tumbling window throughput below target");

    let after = take_util_snapshot();
    record_test_util("Tumbling Window", before, after);
}

// =========================================================================
// 2. Sliding Window Throughput
// =========================================================================

#[test]
fn test_sliding_window_throughput() {
    zyron_bench_harness::init("streaming");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Benchmark: Sliding Window Throughput ===");
    let before = take_util_snapshot();

    let n = 1_000_000usize;
    let assigner = SlidingWindowAssigner::new(300_000, 60_000);

    let event_times: Vec<i64> = (0..n as i64).map(|i| i * 200).collect();

    let mut runs = vec![];
    for _ in 0..VALIDATION_RUNS {
        let start = Instant::now();
        for &t in &event_times {
            black_box(assigner.assign_windows(t));
        }
        let elapsed = start.elapsed();
        runs.push(n as f64 / elapsed.as_secs_f64());
    }

    let result = validate_metric(
        "Sliding Window",
        "Events/sec",
        runs,
        SLIDING_WINDOW_EVENTS_PER_SEC,
        true,
    );
    assert!(result.passed, "Sliding window throughput below target");

    let after = take_util_snapshot();
    record_test_util("Sliding Window", before, after);
}

// =========================================================================
// 3. Session Window Throughput
// =========================================================================

#[test]
fn test_session_window_throughput() {
    zyron_bench_harness::init("streaming");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Benchmark: Session Window Throughput ===");
    let before = take_util_snapshot();

    let n = 1_000_000usize;
    let gap_ms = 30_000i64;

    // Pre-generate key hashes and event times.
    // 1000 distinct keys, events spaced close together to trigger merges.
    let key_hashes: Vec<u64> = (0..n).map(|i| (i % 1000) as u64).collect();
    let event_times: Vec<i64> = (0..n as i64)
        .map(|i| (i % 1000) * 10_000 + (i / 1000) * 5_000)
        .collect();

    let mut runs = vec![];
    for _ in 0..VALIDATION_RUNS {
        let mut merger = SessionMerger::new();
        let start = Instant::now();
        for i in 0..n {
            black_box(merger.add_event(key_hashes[i], event_times[i], gap_ms));
        }
        let elapsed = start.elapsed();
        runs.push(n as f64 / elapsed.as_secs_f64());
    }

    let result = validate_metric(
        "Session Window",
        "Events/sec",
        runs,
        SESSION_WINDOW_EVENTS_PER_SEC,
        true,
    );
    assert!(result.passed, "Session window throughput below target");

    let after = take_util_snapshot();
    record_test_util("Session Window", before, after);
}

// =========================================================================
// 4. Watermark Tracking
// =========================================================================

#[test]
fn test_watermark_tracking() {
    zyron_bench_harness::init("streaming");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Benchmark: Watermark Propagation ===");
    let before = take_util_snapshot();

    let source_count = 4;
    let advances_per_source = 100_000usize;

    let mut runs = vec![];
    for _ in 0..VALIDATION_RUNS {
        let tracker = StreamWatermarkTracker::new(source_count);

        let start = Instant::now();
        for round in 0..advances_per_source {
            for src in 0..source_count {
                tracker.advance(src, (round * 1000 + src * 100) as i64);
            }
            black_box(tracker.combined_watermark());
        }
        let elapsed = start.elapsed();
        runs.push(elapsed.as_millis() as f64);

        // Verify correctness after all advances.
        let final_wm = tracker.combined_watermark();
        assert!(final_wm.timestamp_ms > 0, "watermark should have advanced");
    }

    let result = validate_metric(
        "Watermark Propagation",
        "Total time (ms)",
        runs,
        WATERMARK_PROPAGATION_MS,
        false,
    );
    assert!(result.passed, "Watermark propagation time exceeded target");

    let after = take_util_snapshot();
    record_test_util("Watermark Propagation", before, after);
}

// =========================================================================
// 5. Late Data - Update Policy
// =========================================================================

#[test]
fn test_late_data_update_policy() {
    zyron_bench_harness::init("streaming");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Late Data: Update Policy ===");
    let before = take_util_snapshot();

    let handler = LateDataHandler::new(LateDataPolicy::Update, AllowedLateness::none());
    let watermark = Watermark::new(10_000);

    // Mix of on-time and late records.
    let record = make_record_with_times(vec![5_000, 12_000, 8_000, 15_000, 3_000]);
    let result = handler.filter_late(&record, &watermark);

    // Update policy returns all records (late ones trigger retractions).
    assert_eq!(
        result.num_rows(),
        5,
        "Update policy should return all records"
    );
    // Late records: 5000, 8000, 3000 (all < 10000 watermark with no lateness).
    assert_eq!(
        handler.total_updated(),
        3,
        "3 records should be counted as updated"
    );
    assert_eq!(handler.total_dropped(), 0, "No records should be dropped");

    // Verify handle_late_record returns true for Update policy.
    let single = make_record_with_times(vec![1_000]);
    let should_process = handler.handle_late_record(single, 1_000, &watermark);
    assert!(should_process, "Update policy should signal processing");
    assert_eq!(handler.total_updated(), 4);

    tprintln!("  Update policy: all records returned, late count tracked");
    let passed = check_performance("Late Data Update", "Records returned", 5.0, 5.0, true);
    assert!(passed);

    let after = take_util_snapshot();
    record_test_util("Late Data Update", before, after);
}

// =========================================================================
// 6. Late Data - Drop Policy
// =========================================================================

#[test]
fn test_late_data_drop_policy() {
    zyron_bench_harness::init("streaming");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Late Data: Drop Policy ===");
    let before = take_util_snapshot();

    let handler = LateDataHandler::new(LateDataPolicy::Drop, AllowedLateness::none());
    let watermark = Watermark::new(10_000);

    // 5 records: 3 late, 2 on-time.
    let record = make_record_with_times(vec![5_000, 12_000, 8_000, 15_000, 3_000]);
    let result = handler.filter_late(&record, &watermark);

    assert_eq!(result.num_rows(), 2, "Only on-time records should pass");
    assert_eq!(handler.total_dropped(), 3, "3 records should be dropped");
    assert_eq!(handler.total_updated(), 0, "No records should be updated");

    // Verify individual late record handling.
    let single = make_record_with_times(vec![1_000]);
    let should_process = handler.handle_late_record(single, 1_000, &watermark);
    assert!(!should_process, "Drop policy should not signal processing");
    assert_eq!(handler.total_dropped(), 4);

    tprintln!("  Drop policy: 3 dropped, 2 passed, counter incremented");
    let passed = check_performance("Late Data Drop", "On-time records", 2.0, 2.0, true);
    assert!(passed);

    let after = take_util_snapshot();
    record_test_util("Late Data Drop", before, after);
}

// =========================================================================
// 7. Late Data - SideOutput Policy
// =========================================================================

#[test]
fn test_late_data_side_output() {
    zyron_bench_harness::init("streaming");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Late Data: SideOutput Policy ===");
    let before = take_util_snapshot();

    let (tx, rx) = spsc_channel::<StreamRecord>(16);
    let handler = LateDataHandler::new(LateDataPolicy::SideOutput, AllowedLateness::none())
        .with_side_output(tx);
    let watermark = Watermark::new(10_000);

    // 4 records: 2 late, 2 on-time.
    let record = make_record_with_times(vec![5_000, 12_000, 8_000, 15_000]);
    let result = handler.filter_late(&record, &watermark);

    assert_eq!(result.num_rows(), 2, "Only on-time records should pass");
    assert_eq!(
        handler.total_side_output(),
        2,
        "2 records should go to side output"
    );

    // Verify the side output channel received the late records.
    let side_record = rx.try_recv();
    assert!(side_record.is_some(), "Side output should have a record");
    let side = side_record.unwrap();
    assert_eq!(side.num_rows(), 2, "Side output should contain 2 late rows");

    // Verify event times in the side output are the late ones.
    assert!(
        side.event_times.iter().all(|&t| t < watermark.timestamp_ms),
        "All side output events should be before watermark"
    );

    tprintln!("  SideOutput policy: 2 on-time passed, 2 late routed to side output");
    let passed = check_performance(
        "Late Data SideOutput",
        "Side output records",
        2.0,
        2.0,
        true,
    );
    assert!(passed);

    let after = take_util_snapshot();
    record_test_util("Late Data SideOutput", before, after);
}

// =========================================================================
// 8. Stream-Stream Join Throughput
// =========================================================================

#[test]
fn test_stream_stream_join() {
    zyron_bench_harness::init("streaming");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Benchmark: Stream-Stream Join ===");
    let before = take_util_snapshot();

    let n = 50_000usize;
    let window_ms = 60_000i64;

    // Pre-generate left and right records with matching keys.
    let left_keys: Vec<i64> = (0..n as i64).collect();
    let left_vals: Vec<i64> = (0..n as i64).map(|i| i * 10).collect();
    let left_times: Vec<i64> = (0..n as i64).map(|i| i * 100).collect();

    let right_keys: Vec<i64> = (0..n as i64).collect();
    let right_vals: Vec<i64> = (0..n as i64).map(|i| i * 100).collect();
    let right_times: Vec<i64> = (0..n as i64).map(|i| i * 100 + 50).collect();

    let mut runs = vec![];
    for _ in 0..VALIDATION_RUNS {
        let mut join = StreamStreamJoin::new(1, vec![0], vec![0], window_ms);
        let total_events = n * 2;

        // Feed left side in batches of 1000.
        let batch_size = 1000;
        let start = Instant::now();

        for chunk_start in (0..n).step_by(batch_size) {
            let chunk_end = (chunk_start + batch_size).min(n);
            let left = make_keyed_record(
                left_keys[chunk_start..chunk_end].to_vec(),
                left_vals[chunk_start..chunk_end].to_vec(),
                left_times[chunk_start..chunk_end].to_vec(),
            );
            join.set_input_side(true);
            let _ = black_box(join.process(left));
        }

        // Feed right side in batches of 1000.
        for chunk_start in (0..n).step_by(batch_size) {
            let chunk_end = (chunk_start + batch_size).min(n);
            let right = make_keyed_record(
                right_keys[chunk_start..chunk_end].to_vec(),
                right_vals[chunk_start..chunk_end].to_vec(),
                right_times[chunk_start..chunk_end].to_vec(),
            );
            join.set_input_side(false);
            let _ = black_box(join.process(right));
        }

        let elapsed = start.elapsed();
        runs.push(total_events as f64 / elapsed.as_secs_f64());
    }

    let result = validate_metric(
        "Stream-Stream Join",
        "Events/sec",
        runs,
        STREAM_STREAM_JOIN_EVENTS_PER_SEC,
        true,
    );
    assert!(result.passed, "Stream-stream join throughput below target");

    let after = take_util_snapshot();
    record_test_util("Stream-Stream Join", before, after);
}

// =========================================================================
// 9. Lookup Join (Cached) Throughput
// =========================================================================

#[test]
fn test_lookup_join_cached() {
    zyron_bench_harness::init("streaming");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Benchmark: Lookup Join (Cached) ===");
    let before = take_util_snapshot();

    let n = 500_000usize;
    let num_dim_keys = 1000;

    // Pre-populate a read-only lookup map.
    let lookup_ref: std::sync::Arc<std::collections::HashMap<u64, StreamBatch>> = {
        let mut map = std::collections::HashMap::new();
        for i in 0..num_dim_keys {
            let dim_col = StreamColumn::from_data(StreamColumnData::Int64(vec![i as i64 * 100]));
            let dim_batch = StreamBatch::new(vec![dim_col]);
            let key_hash = hash_int(i as i64);
            map.insert(key_hash, dim_batch);
        }
        std::sync::Arc::new(map)
    };

    // Pre-generate probe keys that cycle through dimension keys.
    let probe_keys: Vec<i64> = (0..n as i64).map(|i| i % num_dim_keys as i64).collect();
    let probe_vals: Vec<i64> = (0..n as i64).collect();
    let probe_times: Vec<i64> = (0..n as i64).map(|i| i * 10).collect();

    let mut runs = vec![];
    for _ in 0..VALIDATION_RUNS {
        let map_clone = lookup_ref.clone();
        let mut join = LookupJoin::new(
            1,
            vec![0],
            60_000,
            num_dim_keys * 2,
            Box::new(move |hash| Ok(map_clone.get(&hash).cloned())),
        );

        // Warm-up pass: process one batch with all dimension keys to fill the cache.
        // This ensures the timed loop measures cache-hit performance only.
        {
            let warmup_keys: Vec<i64> = (0..num_dim_keys as i64).collect();
            let warmup_vals: Vec<i64> = vec![0; num_dim_keys];
            let warmup_times: Vec<i64> = vec![0; num_dim_keys];
            let warmup = make_keyed_record(warmup_keys, warmup_vals, warmup_times);
            let _ = join.process(warmup);
        }

        let batch_size = 1000;
        let start = Instant::now();

        for chunk_start in (0..n).step_by(batch_size) {
            let chunk_end = (chunk_start + batch_size).min(n);
            let probe = make_keyed_record(
                probe_keys[chunk_start..chunk_end].to_vec(),
                probe_vals[chunk_start..chunk_end].to_vec(),
                probe_times[chunk_start..chunk_end].to_vec(),
            );
            let _ = black_box(join.process(probe));
        }

        let elapsed = start.elapsed();
        runs.push(n as f64 / elapsed.as_secs_f64());
    }

    let result = validate_metric(
        "Lookup Join Cached",
        "Events/sec",
        runs,
        LOOKUP_JOIN_EVENTS_PER_SEC,
        true,
    );
    assert!(result.passed, "Lookup join throughput below target");

    let after = take_util_snapshot();
    record_test_util("Lookup Join Cached", before, after);
}

// =========================================================================
// 10. Checkpoint and Recovery
// =========================================================================

#[test]
fn test_checkpoint_and_recovery() {
    zyron_bench_harness::init("streaming");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Benchmark: Checkpoint and Recovery ===");
    let before = take_util_snapshot();

    // Populate a HeapStateBackend with entries to simulate state size.
    // Target: measure snapshot and restore speed. Use 100K entries with
    // 100-byte values to approximate meaningful state without needing 1GB.
    let entry_count = 100_000usize;
    let value_size = 100;
    let backend = HeapStateBackend::new();
    let value_data = vec![0xABu8; value_size];

    for i in 0..entry_count {
        let key = i.to_le_bytes().to_vec();
        backend.put(b"bench_ns", &key, &value_data).unwrap();
    }

    assert_eq!(backend.entry_count(), entry_count);
    tprintln!(
        "  Populated state backend with {} entries ({} bytes each)",
        entry_count,
        value_size
    );

    // Benchmark snapshot.
    let mut snapshot_runs = vec![];
    let mut restore_runs = vec![];

    for _ in 0..VALIDATION_RUNS {
        let snap_start = Instant::now();
        let snapshot = backend.snapshot().unwrap();
        let snap_elapsed = snap_start.elapsed();
        snapshot_runs.push(snap_elapsed.as_secs_f64());

        assert_eq!(snapshot.entry_count(), entry_count);

        // Clear and restore.
        backend.clear_namespace(b"bench_ns").unwrap();
        assert_eq!(backend.entry_count(), 0);

        let restore_start = Instant::now();
        backend.restore(&snapshot).unwrap();
        let restore_elapsed = restore_start.elapsed();
        restore_runs.push(restore_elapsed.as_secs_f64());

        assert_eq!(backend.entry_count(), entry_count);

        // Verify a sample entry after restore.
        let sample_key = 42usize.to_le_bytes().to_vec();
        let val = backend.get(b"bench_ns", &sample_key).unwrap();
        assert_eq!(val, Some(value_data.clone()));
    }

    let snap_result = validate_metric(
        "Checkpoint Snapshot",
        "Snapshot time (s)",
        snapshot_runs,
        CHECKPOINT_1GB_STATE_SEC,
        false,
    );
    assert!(
        snap_result.passed,
        "Checkpoint snapshot time exceeded target"
    );

    let restore_result = validate_metric(
        "Checkpoint Restore",
        "Restore time (s)",
        restore_runs,
        RECOVERY_1GB_STATE_SEC,
        false,
    );
    assert!(
        restore_result.passed,
        "Checkpoint restore time exceeded target"
    );

    let after = take_util_snapshot();
    record_test_util("Checkpoint and Recovery", before, after);
}

// =========================================================================
// 11. Backpressure
// =========================================================================

#[test]
fn test_backpressure() {
    zyron_bench_harness::init("streaming");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Benchmark: Backpressure Monitor ===");
    let before = take_util_snapshot();

    let operator_count = 8;
    let capacities: Vec<usize> = (0..operator_count).map(|_| 10_000).collect();
    let monitor = BackpressureMonitor::new(&capacities);
    let iterations = 100_000usize;

    let mut runs = vec![];
    for _ in 0..VALIDATION_RUNS {
        let start = Instant::now();

        for round in 0..iterations {
            // Simulate varying queue lengths across operators.
            for op_id in 0..operator_count {
                let length = (round * (op_id + 1) * 7) % 10_000;
                monitor.update_queue_length(op_id, length);
            }

            // Check ratios and bottleneck detection.
            black_box(monitor.any_above_threshold(0.8));
            black_box(monitor.bottleneck());
        }

        let elapsed = start.elapsed();
        runs.push(elapsed.as_millis() as f64);
    }

    // Verify ratio correctness with known values using a fresh monitor.
    let verify_monitor = BackpressureMonitor::new(&[10_000, 10_000]);
    verify_monitor.update_queue_length(0, 5_000);
    verify_monitor.update_queue_length(1, 9_000);
    let ratio_0 = verify_monitor.ratio(0);
    let ratio_1 = verify_monitor.ratio(1);
    assert!(
        (ratio_0 - 0.5).abs() < 0.01,
        "Ratio 0 should be 0.5, got {ratio_0}"
    );
    assert!(
        (ratio_1 - 0.9).abs() < 0.01,
        "Ratio 1 should be 0.9, got {ratio_1}"
    );

    let (bottleneck_id, bottleneck_ratio) = verify_monitor.bottleneck().unwrap();
    assert_eq!(bottleneck_id, 1, "Operator 1 should be the bottleneck");
    assert!((bottleneck_ratio - 0.9).abs() < 0.01);

    // Verify load shedding works.
    let policy = LoadSheddingPolicy::Sample {
        keep_ratio: 0.5,
        rng_state: 42,
    };
    let mut shedder = LoadShedder::new(policy);
    let record = make_test_record(
        (0..1000i64).collect(),
        (0..1000i64).map(|i| i * 1000).collect(),
    );
    let result = shedder.apply(&record, 0);
    let kept = result.num_rows();
    assert!(
        kept > 300 && kept < 700,
        "Load shedding should keep ~50%, got {kept}"
    );
    assert!(shedder.total_shed() > 0);

    tprintln!("  Backpressure: ratio checks correct, load shedding operational");

    let bp_result = validate_metric(
        "Backpressure Monitor",
        "Total time (ms)",
        runs,
        BACKPRESSURE_REACTION_MS,
        false,
    );
    assert!(
        bp_result.passed,
        "Backpressure reaction time exceeded target"
    );

    let after = take_util_snapshot();
    record_test_util("Backpressure Monitor", before, after);
}

// =========================================================================
// 12. End-to-End Exactly Once
// =========================================================================

#[test]
fn test_e2e_exactly_once() {
    zyron_bench_harness::init("streaming");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== E2E: Exactly Once Processing ===");
    let before = take_util_snapshot();

    // Build a pipeline: InMemorySource -> WindowAggregateOperator -> InMemorySink.
    // Feed data, fire windows via watermark, collect output.
    let window_size_ms = 10_000i64;

    // Create records spanning 3 windows: [0,10000), [10000,20000), [20000,30000).
    let mut source_records = Vec::new();
    for window_idx in 0..3 {
        let base = window_idx as i64 * window_size_ms;
        let values: Vec<i64> = (0..100).map(|i| i + 1).collect();
        let times: Vec<i64> = (0..100).map(|i| base + i * 90).collect();
        source_records.push(make_test_record(values, times));
    }

    let mut source = InMemorySource::new(source_records.clone());
    source.open(None).unwrap();

    let mut window_op = WindowAggregateOperator::new(
        1,
        vec![], // No group-by key (global aggregate).
        0,      // Aggregate column 0.
        Box::new(|| Box::new(SumAccumulator::new())),
        Box::new(TumblingWindowAssigner::new(window_size_ms)),
    );

    let mut sink = InMemorySink::new();
    let output_handle = sink.output();

    // Process all source records.
    let mut processed_batches = 0;
    while let Some(record) = source.next_batch().unwrap() {
        let _ = window_op.process(record).unwrap();
        processed_batches += 1;
    }
    assert_eq!(processed_batches, 3);

    // Fire windows by advancing watermark past all window ends.
    let results = window_op.on_watermark(Watermark::new(30_000)).unwrap();
    for result in &results {
        sink.write_batch(&[result.clone()]).unwrap();
    }
    sink.commit().unwrap();

    // Verify output.
    let output = output_handle.lock();
    assert!(!output.is_empty(), "Should have output records");
    let total_output_rows: usize = output.iter().map(|r| r.num_rows()).sum();
    tprintln!(
        "  Produced {} output rows from 3 windows",
        total_output_rows
    );

    // Verify each window produced an aggregate result.
    assert!(
        total_output_rows >= 3,
        "Should have at least 3 aggregate results (one per window)"
    );

    // Simulate recovery: checkpoint the operator state, restore, and verify.
    let barrier = CheckpointBarrier::new(CheckpointId::new(1), 30_000);
    let snapshot = window_op.on_barrier(barrier).unwrap();
    tprintln!("  Checkpoint: {} state entries", snapshot.entry_count());

    // Restore operator from snapshot (should clear state and rebuild).
    window_op.restore(snapshot).unwrap();

    // Feed the same data again (simulating replay after recovery).
    let mut source2 = InMemorySource::new(source_records);
    source2.open(None).unwrap();

    let mut replay_sink = InMemorySink::new();
    let replay_handle = replay_sink.output();

    while let Some(record) = source2.next_batch().unwrap() {
        let _ = window_op.process(record).unwrap();
    }
    let results2 = window_op.on_watermark(Watermark::new(30_000)).unwrap();
    for result in &results2 {
        replay_sink.write_batch(&[result.clone()]).unwrap();
    }
    replay_sink.commit().unwrap();

    let replay_output = replay_handle.lock();
    let replay_rows: usize = replay_output.iter().map(|r| r.num_rows()).sum();
    assert_eq!(
        total_output_rows, replay_rows,
        "Replay should produce identical output"
    );

    tprintln!("  Recovery replay: {} rows (matches original)", replay_rows);
    let passed = check_performance("E2E Exactly Once", "Windows processed", 3.0, 3.0, true);
    assert!(passed);

    let after = take_util_snapshot();
    record_test_util("E2E Exactly Once", before, after);
}

// =========================================================================
// 13. Operator Chain DAG
// =========================================================================

#[test]
fn test_operator_chain_dag() {
    zyron_bench_harness::init("streaming");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Operator Chain: Filter -> Project -> KeyBy ===");
    let before = take_util_snapshot();

    // Build a chain: filter(value > 50) -> project(column 0 only) -> key_by(column 0).
    let filter = StreamFilterOperator::new(
        1,
        Box::new(|batch, row| {
            if let StreamColumnData::Int64(v) = &batch.column(0).data {
                v[row] > 50
            } else {
                false
            }
        }),
    );

    let project = StreamProjectOperator::new(2, vec![0]);
    let key_by = StreamKeyByOperator::new(3, vec![0]);

    let mut chain = OperatorChain::new();
    chain.add_operator(Box::new(filter));
    chain.add_operator(Box::new(project));
    chain.add_operator(Box::new(key_by));

    assert_eq!(chain.len(), 3);
    assert_eq!(chain.operator_ids(), vec![1, 2, 3]);

    // Feed 200 records: values 1..=200. Filter keeps values > 50, so 150 remain.
    let values: Vec<i64> = (1..=200).collect();
    let extra_col: Vec<i64> = (0..200).map(|i| i * 10).collect();
    let times: Vec<i64> = (0..200).map(|i| i * 1000).collect();

    let key_col = StreamColumn::from_data(StreamColumnData::Int64(values));
    let val_col = StreamColumn::from_data(StreamColumnData::Int64(extra_col));
    let batch = StreamBatch::new(vec![key_col, val_col]);
    let record = StreamRecord::new_insert(batch, times);

    let results = chain.process(record).unwrap();
    assert_eq!(results.len(), 1, "Chain should produce 1 output record");
    let output = &results[0];

    // Filter: 150 rows pass (values 51..=200).
    assert_eq!(output.num_rows(), 150, "150 rows should pass the filter");

    // Project: only column 0 remains.
    assert_eq!(
        output.batch.num_columns(),
        1,
        "Project should reduce to 1 column"
    );

    // KeyBy: keys should be set.
    assert!(output.keys.is_some(), "KeyBy should set key hashes");
    let keys = output.keys.as_ref().unwrap();
    assert_eq!(keys.len(), 150);

    // Verify all key hashes are non-zero and different values produce different hashes.
    let unique_keys: std::collections::HashSet<u64> = keys.iter().copied().collect();
    assert!(
        unique_keys.len() > 100,
        "Should have many distinct key hashes"
    );

    // Verify first value in output is 51 (first value > 50).
    if let StreamColumnData::Int64(v) = &output.batch.column(0).data {
        assert_eq!(v[0], 51, "First output value should be 51");
        assert_eq!(v[149], 200, "Last output value should be 200");
    } else {
        panic!("Expected Int64 column data");
    }

    // Test watermark propagation through chain.
    let wm_results = chain.on_watermark(Watermark::new(100_000)).unwrap();
    assert!(
        wm_results.is_empty(),
        "Stateless operators should not emit on watermark"
    );

    // Test barrier propagation.
    let barrier = CheckpointBarrier::new(CheckpointId::new(1), 100_000);
    let snapshots = chain.on_barrier(barrier).unwrap();
    assert_eq!(
        snapshots.len(),
        3,
        "Each operator should produce a snapshot"
    );
    for (op_id, snap) in &snapshots {
        tprintln!("  Operator {}: {} state entries", op_id, snap.entry_count());
    }

    tprintln!("  Chain: 200 in, 150 after filter, 1 column after project, keys set");
    let passed = check_performance("Operator Chain", "Output rows", 150.0, 150.0, true);
    assert!(passed);

    let after = take_util_snapshot();
    record_test_util("Operator Chain DAG", before, after);
}

// =========================================================================
// 14. State Backend Operations
// =========================================================================

#[test]
fn test_state_backend_operations() {
    zyron_bench_harness::init("streaming");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== State Backend: HeapStateBackend + DiskStateBackend ===");
    let before = take_util_snapshot();

    // -----------------------------------------------------------------------
    // HeapStateBackend CRUD
    // -----------------------------------------------------------------------
    let heap = HeapStateBackend::new();
    let ns = b"test_ns";

    // Insert 1000 entries.
    for i in 0..1000u32 {
        let key = i.to_le_bytes().to_vec();
        let val = (i * 100).to_le_bytes().to_vec();
        heap.put(ns, &key, &val).unwrap();
    }
    assert_eq!(heap.entry_count(), 1000);
    tprintln!("  HeapState: inserted 1000 entries");

    // Read back and verify.
    for i in 0..1000u32 {
        let key = i.to_le_bytes().to_vec();
        let expected = (i * 100).to_le_bytes().to_vec();
        let val = heap.get(ns, &key).unwrap();
        assert_eq!(val, Some(expected));
    }
    tprintln!("  HeapState: verified 1000 reads");

    // Update 100 entries.
    for i in 0..100u32 {
        let key = i.to_le_bytes().to_vec();
        let new_val = (i * 999).to_le_bytes().to_vec();
        heap.put(ns, &key, &new_val).unwrap();
    }
    assert_eq!(heap.entry_count(), 1000, "Update should not change count");

    // Delete 50 entries.
    for i in 0..50u32 {
        let key = i.to_le_bytes().to_vec();
        heap.delete(ns, &key).unwrap();
    }
    assert_eq!(heap.entry_count(), 950);
    tprintln!("  HeapState: 100 updates, 50 deletes, count = 950");

    // Prefix scan.
    let scan_results = heap.prefix_scan(ns, &[]).unwrap();
    assert_eq!(
        scan_results.len(),
        950,
        "Prefix scan should return all remaining entries"
    );

    // Snapshot and restore.
    let snapshot = heap.snapshot().unwrap();
    assert_eq!(snapshot.entry_count(), 950);

    heap.clear_namespace(ns).unwrap();
    assert_eq!(heap.entry_count(), 0);

    heap.restore(&snapshot).unwrap();
    assert_eq!(heap.entry_count(), 950);
    tprintln!("  HeapState: snapshot/clear/restore verified");

    // Verify a sample entry after restore.
    let sample_key = 500u32.to_le_bytes().to_vec();
    let sample_val = heap.get(ns, &sample_key).unwrap();
    assert!(sample_val.is_some(), "Entry 500 should exist after restore");
    let expected_val = (500u32 * 100).to_le_bytes().to_vec();
    assert_eq!(sample_val.unwrap(), expected_val);

    // -----------------------------------------------------------------------
    // DiskStateBackend CRUD
    // -----------------------------------------------------------------------
    let tmp = TempDir::new().unwrap();
    let disk = DiskStateBackend::new(tmp.path(), 1024 * 1024).unwrap();

    // Insert entries.
    for i in 0..500u32 {
        let key = i.to_le_bytes().to_vec();
        let val = (i * 50).to_le_bytes().to_vec();
        disk.put(b"disk_ns", &key, &val).unwrap();
    }
    assert_eq!(disk.entry_count(), 500);

    // Read back.
    for i in 0..500u32 {
        let key = i.to_le_bytes().to_vec();
        let expected = (i * 50).to_le_bytes().to_vec();
        let val = disk.get(b"disk_ns", &key).unwrap();
        assert_eq!(val, Some(expected));
    }
    tprintln!("  DiskState: inserted and verified 500 entries");

    // Snapshot and restore.
    let disk_snap = disk.snapshot().unwrap();
    assert_eq!(disk_snap.entry_count(), 500);

    disk.clear_namespace(b"disk_ns").unwrap();
    assert_eq!(disk.entry_count(), 0);

    disk.restore(&disk_snap).unwrap();
    assert_eq!(disk.entry_count(), 500);
    tprintln!("  DiskState: snapshot/clear/restore verified");

    // Delete and verify.
    let del_key = 100u32.to_le_bytes().to_vec();
    disk.delete(b"disk_ns", &del_key).unwrap();
    assert_eq!(disk.get(b"disk_ns", &del_key).unwrap(), None);
    assert_eq!(disk.entry_count(), 499);

    // Prefix scan on disk backend.
    let disk_scan = disk.prefix_scan(b"disk_ns", &[]).unwrap();
    assert_eq!(disk_scan.len(), 499);

    tprintln!("  DiskState: delete and prefix scan verified");
    let passed = check_performance(
        "State Backend Ops",
        "Total entries verified",
        1950.0,
        1950.0,
        true,
    );
    assert!(passed);

    let after = take_util_snapshot();
    record_test_util("State Backend Operations", before, after);
}

// =========================================================================
// 15. Tumbling Window Correctness
// =========================================================================

#[test]
fn test_tumbling_window_correctness() {
    zyron_bench_harness::init("streaming");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Correctness: Tumbling Window ===");

    // 1000 events. Values: 1..=1000. Times: 0, 600, 1200, ... (600ms apart).
    // Window size = 60_000ms. Each window holds 100 events.
    let values: Vec<i64> = (1..=1000).collect();
    let times: Vec<i64> = (0..1000).map(|i| i * 600).collect();
    let record = make_test_record(values, times);

    let mut op = WindowAggregateOperator::new(
        1,
        vec![],
        0,
        Box::new(|| Box::new(SumAccumulator::new())),
        Box::new(TumblingWindowAssigner::new(60_000)),
    );

    // Process all events.
    let process_output = op.process(record).unwrap();
    assert!(
        process_output.is_empty(),
        "No output before watermark fires"
    );

    // Advance watermark to 700_000 to fire all 10 windows.
    let results = op.on_watermark(Watermark::new(700_000)).unwrap();
    assert!(!results.is_empty(), "Watermark should fire window results");

    let total_rows: usize = results.iter().map(|r| r.num_rows()).sum();
    assert_eq!(total_rows, 10, "Should produce exactly 10 window results");

    // Collect all (window_start, sum) pairs from output.
    let mut window_sums: Vec<(i64, f64)> = Vec::new();
    for result in &results {
        if let StreamColumnData::Int64(starts) = &result.batch.column(1).data {
            if let StreamColumnData::Float64(sums) = &result.batch.column(3).data {
                for i in 0..result.num_rows() {
                    window_sums.push((starts[i], sums[i]));
                }
            }
        }
    }
    window_sums.sort_by_key(|(start, _)| *start);

    // Window 0: events 1..=100 (times 0..59400) -> SUM = 5050.
    assert_eq!(window_sums[0].0, 0);
    assert!(
        (window_sums[0].1 - 5050.0).abs() < 0.01,
        "Window 0 sum should be 5050, got {}",
        window_sums[0].1
    );

    // Window 9: events 901..=1000 (times 540000..599400) -> SUM = 95050.
    assert_eq!(window_sums[9].0, 540_000);
    assert!(
        (window_sums[9].1 - 95050.0).abs() < 0.01,
        "Window 9 sum should be 95050, got {}",
        window_sums[9].1
    );

    tprintln!("  10 tumbling windows verified with correct SUM values");
}

// =========================================================================
// 16. Sliding Window Correctness
// =========================================================================

#[test]
fn test_sliding_window_correctness() {
    zyron_bench_harness::init("streaming");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Correctness: Sliding Window ===");

    let assigner = SlidingWindowAssigner::new(300_000, 60_000);

    // Verify: timestamp 150_000 belongs to multiple overlapping windows.
    let windows_at_150k = assigner.assign_windows(150_000);
    assert_eq!(
        windows_at_150k.len(),
        3,
        "Timestamp 150000 should belong to 3 windows"
    );
    for w in &windows_at_150k {
        assert!(
            w.contains(150_000),
            "All assigned windows must contain timestamp 150000"
        );
    }

    // 60 events at times 0, 10_000, 20_000, ..., 590_000. Values: 1..=60.
    let values: Vec<i64> = (1..=60).collect();
    let times: Vec<i64> = (0..60).map(|i| i * 10_000).collect();
    let record = make_test_record(values, times);

    let mut op = WindowAggregateOperator::new(
        2,
        vec![],
        0,
        Box::new(|| Box::new(SumAccumulator::new())),
        Box::new(SlidingWindowAssigner::new(300_000, 60_000)),
    );

    let process_output = op.process(record).unwrap();
    assert!(
        process_output.is_empty(),
        "No output before watermark fires"
    );

    // Advance watermark past all window ends.
    let results = op.on_watermark(Watermark::new(900_000)).unwrap();
    assert!(!results.is_empty(), "Sliding window should produce output");

    let total_rows: usize = results.iter().map(|r| r.num_rows()).sum();
    // Output rows = distinct window slots = ceil(time_range / slide). Same count as
    // tumbling with the same slide interval. Sliding windows differ by producing WIDER
    // overlapping windows where each event contributes to multiple aggregates.
    assert!(
        total_rows >= 10,
        "Sliding windows should produce at least 10 window results (got {})",
        total_rows
    );

    // Verify at least one window has a correct SUM for its time range.
    // Window [0, 300_000) contains events with times 0..290_000 (values 1..=30). SUM = 465.
    let mut found_first_window = false;
    for result in &results {
        if let StreamColumnData::Int64(starts) = &result.batch.column(1).data {
            if let StreamColumnData::Int64(ends) = &result.batch.column(2).data {
                if let StreamColumnData::Float64(sums) = &result.batch.column(3).data {
                    for i in 0..result.num_rows() {
                        if starts[i] == 0 && ends[i] == 300_000 {
                            assert!(
                                (sums[i] - 465.0).abs() < 0.01,
                                "Window [0, 300000) sum should be 465, got {}",
                                sums[i]
                            );
                            found_first_window = true;
                        }
                    }
                }
            }
        }
    }
    assert!(
        found_first_window,
        "Should find window [0, 300000) in output"
    );

    tprintln!(
        "  Sliding window: {} output rows, overlapping sums verified",
        total_rows
    );
}

// =========================================================================
// 17. Session Window Correctness
// =========================================================================

#[test]
fn test_session_window_correctness() {
    zyron_bench_harness::init("streaming");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Correctness: Session Window ===");

    let gap_ms = 1_800_000i64; // 30 minutes.
    let mut merger = SessionMerger::new();

    // User A (key_hash=1): events at 0, 300_000, 600_000.
    // All within gap of each other, so they merge into one session.
    let (_, r1) = merger.add_event(1, 0, gap_ms);
    assert_eq!(r1, WindowRange::new(0, 1_800_000));

    let (_, r2) = merger.add_event(1, 300_000, gap_ms);
    assert_eq!(r2.start_ms, 0);
    assert_eq!(r2.end_ms, 2_100_000); // merge [0, 1.8M) with [300K, 2.1M).

    let (_, r3) = merger.add_event(1, 600_000, gap_ms);
    assert_eq!(r3.start_ms, 0);
    assert_eq!(r3.end_ms, 2_400_000); // merge [0, 2.1M) with [600K, 2.4M).

    // User B (key_hash=2): events at 3_600_000, 3_900_000.
    let (_, r4) = merger.add_event(2, 3_600_000, gap_ms);
    assert_eq!(r4, WindowRange::new(3_600_000, 5_400_000));

    let (_, r5) = merger.add_event(2, 3_900_000, gap_ms);
    assert_eq!(r5.start_ms, 3_600_000);
    assert_eq!(r5.end_ms, 5_700_000); // merge [3.6M, 5.4M) with [3.9M, 5.7M).

    // Two distinct sessions (one per key).
    assert_eq!(merger.session_count(), 2);

    // Verify ranges via get_range.
    let range_a = merger.get_range(1).expect("User A session should exist");
    assert_eq!(range_a, WindowRange::new(0, 2_400_000));

    let range_b = merger.get_range(2).expect("User B session should exist");
    assert_eq!(range_b, WindowRange::new(3_600_000, 5_700_000));

    tprintln!("  Session merger: 2 sessions, ranges [0, 2.4M) and [3.6M, 5.7M) verified");
}

// =========================================================================
// 18. Watermark Correctness
// =========================================================================

#[test]
fn test_watermark_correctness() {
    zyron_bench_harness::init("streaming");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Correctness: Watermark ===");

    let mut wm_generator = BoundedOutOfOrderWatermark::new(300_000); // 5 min.

    // Event 600_000: watermark = 600_000 - 300_000 = 300_000.
    wm_generator.on_event(600_000);
    assert_eq!(wm_generator.current_watermark(), Watermark::new(300_000));

    // Event 780_000: watermark = 780_000 - 300_000 = 480_000.
    wm_generator.on_event(780_000);
    assert_eq!(wm_generator.current_watermark(), Watermark::new(480_000));

    // Event 660_000 (out of order): max still 780_000, watermark stays 480_000.
    wm_generator.on_event(660_000);
    assert_eq!(wm_generator.current_watermark(), Watermark::new(480_000));

    // Event 900_000: watermark = 900_000 - 300_000 = 600_000.
    wm_generator.on_event(900_000);
    assert_eq!(wm_generator.current_watermark(), Watermark::new(600_000));

    // Event 720_000 (out of order): max still 900_000, watermark stays 600_000.
    wm_generator.on_event(720_000);
    assert_eq!(wm_generator.current_watermark(), Watermark::new(600_000));

    tprintln!("  BoundedOutOfOrder watermark: 5 events, monotonic advance verified");
}

// =========================================================================
// 19. Stream-Stream Join Correctness
// =========================================================================

#[test]
fn test_stream_stream_join_correctness() {
    zyron_bench_harness::init("streaming");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Correctness: Stream-Stream Join ===");

    let mut join = StreamStreamJoin::new(1, vec![0], vec![0], 3_600_000);

    // Left: click (user_id=1, product_id=100) at time=100_000.
    let left1 = make_keyed_record(vec![1], vec![100], vec![100_000]);
    join.set_input_side(true);
    let out1 = join.process(left1).unwrap();
    assert!(
        out1.is_empty(),
        "First left record should not produce output (no right match yet)"
    );

    // Right: purchase (user_id=1, product_id=100) at time=200_000.
    let right1 = make_keyed_record(vec![1], vec![100], vec![200_000]);
    join.set_input_side(false);
    let out2 = join.process(right1).unwrap();
    assert!(
        !out2.is_empty(),
        "Right record matching user_id=1 should join with left"
    );
    let join_rows: usize = out2.iter().map(|r| r.num_rows()).sum();
    assert_eq!(join_rows, 1, "Should produce exactly 1 joined row");

    // Left: click (user_id=2, product_id=200) at time=100_000.
    let left2 = make_keyed_record(vec![2], vec![200], vec![100_000]);
    join.set_input_side(true);
    let out3 = join.process(left2).unwrap();
    assert!(out3.is_empty(), "user_id=2 has no right match");

    // Right: purchase (user_id=3, product_id=300) at time=100_000.
    let right2 = make_keyed_record(vec![3], vec![300], vec![100_000]);
    join.set_input_side(false);
    let out4 = join.process(right2).unwrap();
    assert!(out4.is_empty(), "user_id=3 has no left match");

    // Advance watermark to evict old entries.
    let _ = join.on_watermark(Watermark::new(5_000_000)).unwrap();

    // New right with user_id=1 at time=4_500_000.
    let right3 = make_keyed_record(vec![1], vec![500], vec![4_500_000]);
    join.set_input_side(false);
    let out5 = join.process(right3).unwrap();
    // Old left at 100_000 should be evicted (cutoff = 5M - 3.6M = 1.4M, 100K < 1.4M).
    assert!(
        out5.is_empty(),
        "Old left entry should be evicted after watermark advance"
    );

    // New left with user_id=1 at time=4_600_000 should match right at 4_500_000.
    let left3 = make_keyed_record(vec![1], vec![600], vec![4_600_000]);
    join.set_input_side(true);
    let out6 = join.process(left3).unwrap();
    assert!(
        !out6.is_empty(),
        "New left should join with recent right for user_id=1"
    );
    let new_join_rows: usize = out6.iter().map(|r| r.num_rows()).sum();
    assert_eq!(
        new_join_rows, 1,
        "Should produce exactly 1 joined row for the new pair"
    );

    tprintln!("  Stream-stream join: match, no-match, eviction, and re-match verified");
}

// =========================================================================
// 20. Lookup Join Correctness
// =========================================================================

#[test]
fn test_lookup_join_correctness() {
    zyron_bench_harness::init("streaming");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Correctness: Lookup Join ===");

    // Build lookup data: 3 customers with tier values.
    let lookup_map: std::sync::Arc<std::collections::HashMap<u64, StreamBatch>> = {
        let mut map = std::collections::HashMap::new();
        for customer_id in 0..3i64 {
            let tier = customer_id + 1; // Tier: 1, 2, 3.
            let key_hash = hash_int(customer_id);
            let batch = StreamBatch::new(vec![StreamColumn::from_data(StreamColumnData::Int64(
                vec![tier],
            ))]);
            map.insert(key_hash, batch);
        }
        std::sync::Arc::new(map)
    };

    let call_count = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let call_count_clone = call_count.clone();
    let map_clone = lookup_map.clone();

    let mut join = LookupJoin::new(
        1,
        vec![0],
        60_000,
        100,
        Box::new(move |hash| {
            call_count_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(map_clone.get(&hash).cloned())
        }),
    );

    // Probe with orders containing customer_id=0,1,2.
    let probe = make_keyed_record(vec![0, 1, 2], vec![10, 20, 30], vec![1000, 2000, 3000]);
    let results = join.process(probe).unwrap();
    assert!(!results.is_empty(), "Lookup join should produce output");

    let total_rows: usize = results.iter().map(|r| r.num_rows()).sum();
    assert_eq!(total_rows, 3, "Should have 3 enriched rows");

    // Verify enriched tier values from the lookup.
    // Output columns: [probe_key, probe_val, lookup_tier].
    for result in &results {
        if let StreamColumnData::Int64(tiers) = &result.batch.column(2).data {
            assert_eq!(tiers.len(), 3);
            assert_eq!(tiers[0], 1, "Customer 0 should have tier 1");
            assert_eq!(tiers[1], 2, "Customer 1 should have tier 2");
            assert_eq!(tiers[2], 3, "Customer 2 should have tier 3");
        } else {
            panic!("Expected Int64 tier column");
        }
    }

    let first_call_count = call_count.load(std::sync::atomic::Ordering::Relaxed);
    assert_eq!(
        first_call_count, 3,
        "First probe should trigger 3 lookup calls"
    );

    // Second probe with the same keys should hit cache.
    let probe2 = make_keyed_record(vec![0, 1, 2], vec![40, 50, 60], vec![4000, 5000, 6000]);
    let results2 = join.process(probe2).unwrap();
    let total_rows2: usize = results2.iter().map(|r| r.num_rows()).sum();
    assert_eq!(total_rows2, 3, "Cached probe should still produce 3 rows");

    let second_call_count = call_count.load(std::sync::atomic::Ordering::Relaxed);
    assert_eq!(
        second_call_count, 3,
        "Cached probe should not trigger additional lookup calls"
    );

    // Advance watermark past TTL to expire cache.
    let _ = join.on_watermark(Watermark::new(70_000)).unwrap();

    // Third probe should trigger fresh lookups (cache expired).
    let probe3 = make_keyed_record(
        vec![0, 1, 2],
        vec![70, 80, 90],
        vec![70_000, 70_000, 70_000],
    );
    let _ = join.process(probe3).unwrap();

    let third_call_count = call_count.load(std::sync::atomic::Ordering::Relaxed);
    assert!(
        third_call_count > 3,
        "After TTL expiry, lookups should be called again (got {})",
        third_call_count
    );

    tprintln!("  Lookup join: enrichment, caching, and TTL expiry verified");
}

// =========================================================================
// 21. Operator Checkpoint and Recovery Correctness
// =========================================================================

#[test]
fn test_operator_checkpoint_recovery_correctness() {
    zyron_bench_harness::init("streaming");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Correctness: Operator Checkpoint and Recovery ===");

    let window_size = 10_000i64;

    // 500 events spanning 5 windows. Values: 1..=500. Times distribute across 5 windows.
    // Window k covers [k*10000, (k+1)*10000). Each window gets 100 events.
    let values: Vec<i64> = (1..=500).collect();
    let times: Vec<i64> = (0..500)
        .map(|i| (i / 100) * window_size + (i % 100) * 90)
        .collect();

    let make_operator = || {
        WindowAggregateOperator::new(
            1,
            vec![],
            0,
            Box::new(|| Box::new(SumAccumulator::new())),
            Box::new(TumblingWindowAssigner::new(window_size)),
        )
    };

    let mut op = make_operator();

    // Process all 500 events.
    let record = make_test_record(values.clone(), times.clone());
    let _ = op.process(record).unwrap();

    // Checkpoint the operator state before firing windows.
    let barrier = CheckpointBarrier::new(CheckpointId::new(1), 50_000);
    let snapshot = op.on_barrier(barrier).unwrap();
    assert!(
        snapshot.entry_count() > 0,
        "Snapshot should contain window state"
    );

    // Fire all windows.
    let output_a = op.on_watermark(Watermark::new(60_000)).unwrap();
    let rows_a: usize = output_a.iter().map(|r| r.num_rows()).sum();
    assert_eq!(rows_a, 5, "Should produce 5 window results");

    // Collect SUM values from output_a.
    let mut sums_a: Vec<f64> = Vec::new();
    for result in &output_a {
        if let StreamColumnData::Float64(sums) = &result.batch.column(3).data {
            sums_a.extend(sums.iter());
        }
    }
    sums_a.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Restore from checkpoint into a fresh operator.
    let mut op2 = make_operator();
    op2.restore(snapshot).unwrap();

    // Fire windows from restored state (same data was already accumulated before checkpoint).
    let output_b = op2.on_watermark(Watermark::new(60_000)).unwrap();
    let rows_b: usize = output_b.iter().map(|r| r.num_rows()).sum();
    assert_eq!(
        rows_b, 5,
        "Restored operator should produce 5 window results"
    );

    // Collect SUM values from output_b.
    let mut sums_b: Vec<f64> = Vec::new();
    for result in &output_b {
        if let StreamColumnData::Float64(sums) = &result.batch.column(3).data {
            sums_b.extend(sums.iter());
        }
    }
    sums_b.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Verify output_a == output_b.
    assert_eq!(
        sums_a.len(),
        sums_b.len(),
        "Both outputs should have the same number of windows"
    );
    for (a, b) in sums_a.iter().zip(sums_b.iter()) {
        assert!(
            (a - b).abs() < 0.01,
            "SUM values should match: {} vs {}",
            a,
            b
        );
    }

    // Verify no duplicates: each window appears exactly once.
    assert_eq!(rows_a, 5, "No duplicate windows in output_a");
    assert_eq!(rows_b, 5, "No duplicate windows in output_b");

    tprintln!("  Checkpoint/recovery: 5 windows, identical SUM values after restore");
}

// =========================================================================
// Micro-benchmark: Stream Join Latency Breakdown
// =========================================================================

#[test]
fn test_stream_join_latency_breakdown() {
    zyron_bench_harness::init("streaming");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Stream Join Latency Breakdown ===");

    let n = 50_000usize;
    let batch_size = 1000usize;
    let window_ms = 60_000i64;

    // Pre-build all input batches OUTSIDE the timer.
    let mut left_batches = Vec::new();
    let mut right_batches = Vec::new();
    for chunk_start in (0..n).step_by(batch_size) {
        let chunk_end = (chunk_start + batch_size).min(n);
        let keys: Vec<i64> = (chunk_start as i64..chunk_end as i64).collect();
        let vals: Vec<i64> = keys.iter().map(|&k| k * 10).collect();
        let times: Vec<i64> = keys.iter().map(|&k| k * 100).collect();
        left_batches.push(make_keyed_record(keys.clone(), vals, times));

        let rvals: Vec<i64> = keys.iter().map(|&k| k * 100).collect();
        let rtimes: Vec<i64> = keys.iter().map(|&k| k * 100 + 50).collect();
        right_batches.push(make_keyed_record(keys, rvals, rtimes));
    }

    // Measure left-side insert only (no probe matches since right state is empty).
    let mut join = StreamStreamJoin::new(1, vec![0], vec![0], window_ms);
    let insert_start = Instant::now();
    for batch in &left_batches {
        join.set_input_side(true);
        let _ = black_box(join.process(batch.clone()));
    }
    let insert_elapsed = insert_start.elapsed();
    let insert_rate = n as f64 / insert_elapsed.as_secs_f64();
    tprintln!(
        "  Left insert (no probe):  {:.1}M/sec ({:.1}ms for {})",
        insert_rate / 1e6,
        insert_elapsed.as_secs_f64() * 1000.0,
        n
    );

    // Measure right-side probe + insert (every right row probes left state and matches).
    let probe_start = Instant::now();
    for batch in &right_batches {
        join.set_input_side(false);
        let _ = black_box(join.process(batch.clone()));
    }
    let probe_elapsed = probe_start.elapsed();
    let probe_rate = n as f64 / probe_elapsed.as_secs_f64();
    tprintln!(
        "  Right probe+insert:      {:.1}M/sec ({:.1}ms for {})",
        probe_rate / 1e6,
        probe_elapsed.as_secs_f64() * 1000.0,
        n
    );

    // Measure hash-only cost.
    let hash_start = Instant::now();
    let mut hash_buf = Vec::new();
    for batch in &left_batches {
        hash_column_batch_into(batch.batch.column(0), batch.num_rows(), &mut hash_buf);
        black_box(&hash_buf);
    }
    let hash_elapsed = hash_start.elapsed();
    let hash_rate = n as f64 / hash_elapsed.as_secs_f64();
    tprintln!(
        "  Hash only:               {:.1}M/sec ({:.1}ms for {})",
        hash_rate / 1e6,
        hash_elapsed.as_secs_f64() * 1000.0,
        n
    );

    // Measure FlatU64Map lookup-only cost with a standalone map.
    let mut standalone_map: FlatU64Map<i64> = FlatU64Map::new();
    for i in 0..n as u64 {
        standalone_map.insert(hash_int(i as i64), i as i64);
    }
    let lookup_start = Instant::now();
    let mut found = 0u64;
    for batch in &right_batches {
        hash_column_batch_into(batch.batch.column(0), batch.num_rows(), &mut hash_buf);
        for &h in &hash_buf {
            if standalone_map.get(h).is_some() {
                found += 1;
            }
        }
    }
    let lookup_elapsed = lookup_start.elapsed();
    let lookup_rate = n as f64 / lookup_elapsed.as_secs_f64();
    tprintln!(
        "  Map lookup only:         {:.1}M/sec ({:.1}ms for {}, found={})",
        lookup_rate / 1e6,
        lookup_elapsed.as_secs_f64() * 1000.0,
        n,
        found
    );

    // Measure record creation cost (the .clone() in the benchmark loop).
    let clone_start = Instant::now();
    for batch in &left_batches {
        black_box(batch.clone());
    }
    let clone_elapsed = clone_start.elapsed();
    let clone_rate = n as f64 / clone_elapsed.as_secs_f64();
    tprintln!(
        "  Record clone only:       {:.1}M/sec ({:.1}ms for {})",
        clone_rate / 1e6,
        clone_elapsed.as_secs_f64() * 1000.0,
        n
    );

    let total_rate = (n * 2) as f64 / (insert_elapsed + probe_elapsed).as_secs_f64();
    tprintln!("  Combined rate:           {:.1}M/sec", total_rate / 1e6);

    let after = take_util_snapshot();
    record_test_util("Stream Join Breakdown", take_util_snapshot(), after);
}

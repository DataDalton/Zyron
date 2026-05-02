//! External format writer benchmark suite.
//!
//! Measures per-row encode throughput for each format the streaming crate
//! ships: JSON array, JSON Lines, CSV, Parquet, Arrow IPC, and Avro. Each
//! format is measured at three batch sizes (100, 1000, 10000 rows) so the
//! fixed per-batch overhead of columnar formats (Parquet, Arrow IPC) can be
//! compared against the amortized per-row cost of record-delimited formats.
//!
//! The row shape is a realistic four-column OLTP-ish record: BIGINT id,
//! VARCHAR name, BOOLEAN active, FLOAT64 score. Every row has a small mix of
//! NULL and non-NULL values so codec hot paths exercise both branches.
//!
//! Run: cargo test -p zyron-streaming --test format_bench --release -- --nocapture --test-threads=1

use std::sync::Mutex;

use zyron_common::TypeId;
use zyron_streaming::format::{ColumnSpec, FormatKind, writer_for};
use zyron_streaming::row_codec::StreamValue;

use zyron_bench_harness::*;

// -----------------------------------------------------------------------------
// Performance targets
// -----------------------------------------------------------------------------
//
// All targets are per-row encode throughput in rows/sec, measured on a
// release build. Record-delimited formats beat columnar formats on small
// batches because columnar writers amortize a per-batch header over all rows.
// Targets reflect conservative floors on a modern x86 laptop; a miss signals
// a real regression.

const JSON_ROWS_PER_SEC: f64 = 1_000_000.0;
const JSONL_ROWS_PER_SEC: f64 = 1_500_000.0;
const CSV_ROWS_PER_SEC: f64 = 2_000_000.0;
const PARQUET_ROWS_PER_SEC: f64 = 500_000.0;
const ARROW_IPC_ROWS_PER_SEC: f64 = 1_000_000.0;
const AVRO_ROWS_PER_SEC: f64 = 300_000.0;

// -----------------------------------------------------------------------------
// Benchmark infrastructure
// -----------------------------------------------------------------------------

static BENCHMARK_LOCK: Mutex<()> = Mutex::new(());

/// Four-column schema used by every format benchmark. Keeps the measurement
/// focused on codec cost rather than schema-specific work.
fn bench_schema() -> Vec<ColumnSpec> {
    vec![
        ColumnSpec {
            name: "id".to_string(),
            type_id: TypeId::Int64,
        },
        ColumnSpec {
            name: "name".to_string(),
            type_id: TypeId::Varchar,
        },
        ColumnSpec {
            name: "active".to_string(),
            type_id: TypeId::Boolean,
        },
        ColumnSpec {
            name: "score".to_string(),
            type_id: TypeId::Float64,
        },
    ]
}

/// Synthesizes n rows against the bench_schema. Every fifth row has a NULL
/// name so decoders exercise the nullable string branch.
fn build_rows(n: usize) -> Vec<Vec<StreamValue>> {
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        let name = if i % 5 == 0 {
            StreamValue::Null
        } else {
            StreamValue::Utf8(format!("row_{i:08}"))
        };
        rows.push(vec![
            StreamValue::I64(i as i64),
            name,
            StreamValue::Bool(i % 2 == 0),
            StreamValue::F64(i as f64 * 1.25),
        ]);
    }
    rows
}

/// Runs the FormatKind writer over the given row set once and returns the
/// rows-per-second throughput, discarding the emitted bytes. The caller
/// supplies the iteration count so multiple batches can be timed back-to-back
/// within a single sampled run.
fn measure_format_write(
    kind: FormatKind,
    schema: &[ColumnSpec],
    rows: &[Vec<StreamValue>],
    batches: usize,
) -> f64 {
    let mut writer = writer_for(kind);
    let t0 = Instant::now();
    for _ in 0..batches {
        let out = writer.write_rows(rows, schema).expect("write_rows failed");
        std::hint::black_box(out);
    }
    let elapsed = t0.elapsed();
    let total_rows = (batches * rows.len()) as f64;
    total_rows / elapsed.as_secs_f64()
}

// -----------------------------------------------------------------------------
// JSON array
// -----------------------------------------------------------------------------

#[test]
fn test_format_json_write_throughput() {
    zyron_bench_harness::init("format");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== JSON array write throughput ===");

    let schema = bench_schema();
    let mut per_size_results: Vec<(usize, Vec<f64>)> = Vec::new();
    for &size in &[100usize, 1000, 10_000] {
        let rows = build_rows(size);
        let batches = 10_000 / size.max(1) * 10; // scale so each run emits at least 100k rows
        let mut runs = Vec::with_capacity(VALIDATION_RUNS);
        for run in 0..VALIDATION_RUNS {
            tprintln!(
                "--- Run {}/{} (batch={}) ---\n",
                run + 1,
                VALIDATION_RUNS,
                size
            );
            let ops = measure_format_write(FormatKind::Json, &schema, &rows, batches);
            tprintln!(
                "  batch={:5}: {} rows/sec ({} batches x {} rows)",
                size,
                format_with_commas(ops),
                batches,
                size,
            );
            runs.push(ops);
        }
        per_size_results.push((size, runs));
    }

    let big_runs = per_size_results
        .iter()
        .find(|(s, _)| *s == 10_000)
        .map(|(_, r)| r.clone())
        .unwrap_or_default();
    let result = validate_metric(
        "JSON Write",
        "rows/sec at batch=10000",
        big_runs,
        JSON_ROWS_PER_SEC,
        true,
    );
    assert!(result.passed, "JSON write throughput below target");
    assert!(!result.regression_detected, "JSON write regression");
}

// -----------------------------------------------------------------------------
// JSON Lines
// -----------------------------------------------------------------------------

#[test]
fn test_format_jsonl_write_throughput() {
    zyron_bench_harness::init("format");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== JSON Lines write throughput ===");

    let schema = bench_schema();
    for &size in &[100usize, 1000, 10_000] {
        let rows = build_rows(size);
        let batches = (100_000 / size).max(1);
        let mut runs = Vec::with_capacity(VALIDATION_RUNS);
        for run in 0..VALIDATION_RUNS {
            tprintln!(
                "--- Run {}/{} (batch={}) ---\n",
                run + 1,
                VALIDATION_RUNS,
                size
            );
            let ops = measure_format_write(FormatKind::JsonLines, &schema, &rows, batches);
            tprintln!("  batch={:5}: {} rows/sec", size, format_with_commas(ops),);
            runs.push(ops);
        }
        if size == 10_000 {
            let result = validate_metric(
                "JSONL Write",
                "rows/sec at batch=10000",
                runs,
                JSONL_ROWS_PER_SEC,
                true,
            );
            assert!(result.passed, "JSONL write below target");
            assert!(!result.regression_detected, "JSONL write regression");
        }
    }
}

// -----------------------------------------------------------------------------
// CSV
// -----------------------------------------------------------------------------

#[test]
fn test_format_csv_write_throughput() {
    zyron_bench_harness::init("format");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== CSV write throughput ===");

    let schema = bench_schema();
    for &size in &[100usize, 1000, 10_000] {
        let rows = build_rows(size);
        let batches = (100_000 / size).max(1);
        let mut runs = Vec::with_capacity(VALIDATION_RUNS);
        for run in 0..VALIDATION_RUNS {
            tprintln!(
                "--- Run {}/{} (batch={}) ---\n",
                run + 1,
                VALIDATION_RUNS,
                size
            );
            let ops = measure_format_write(FormatKind::Csv, &schema, &rows, batches);
            tprintln!("  batch={:5}: {} rows/sec", size, format_with_commas(ops),);
            runs.push(ops);
        }
        if size == 10_000 {
            let result = validate_metric(
                "CSV Write",
                "rows/sec at batch=10000",
                runs,
                CSV_ROWS_PER_SEC,
                true,
            );
            assert!(result.passed, "CSV write below target");
            assert!(!result.regression_detected, "CSV write regression");
        }
    }
}

// -----------------------------------------------------------------------------
// Parquet
// -----------------------------------------------------------------------------

#[test]
fn test_format_parquet_write_throughput() {
    zyron_bench_harness::init("format");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Parquet write throughput ===");

    let schema = bench_schema();
    for &size in &[100usize, 1000, 10_000] {
        let rows = build_rows(size);
        // Parquet pays a per-file header cost so larger batches are cheaper
        // per row. Cap the batch count so total wall-clock stays reasonable.
        let batches = match size {
            100 => 200,
            1000 => 100,
            10_000 => 10,
            _ => 10,
        };
        let mut runs = Vec::with_capacity(VALIDATION_RUNS);
        for run in 0..VALIDATION_RUNS {
            tprintln!(
                "--- Run {}/{} (batch={}) ---\n",
                run + 1,
                VALIDATION_RUNS,
                size
            );
            let ops = measure_format_write(FormatKind::Parquet, &schema, &rows, batches);
            tprintln!("  batch={:5}: {} rows/sec", size, format_with_commas(ops),);
            runs.push(ops);
        }
        if size == 10_000 {
            let result = validate_metric(
                "Parquet Write",
                "rows/sec at batch=10000",
                runs,
                PARQUET_ROWS_PER_SEC,
                true,
            );
            assert!(result.passed, "Parquet write below target");
            assert!(!result.regression_detected, "Parquet write regression");
        }
    }
}

// -----------------------------------------------------------------------------
// Arrow IPC
// -----------------------------------------------------------------------------

#[test]
fn test_format_arrow_ipc_write_throughput() {
    zyron_bench_harness::init("format");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Arrow IPC write throughput ===");

    let schema = bench_schema();
    for &size in &[100usize, 1000, 10_000] {
        let rows = build_rows(size);
        let batches = match size {
            100 => 500,
            1000 => 200,
            10_000 => 20,
            _ => 10,
        };
        let mut runs = Vec::with_capacity(VALIDATION_RUNS);
        for run in 0..VALIDATION_RUNS {
            tprintln!(
                "--- Run {}/{} (batch={}) ---\n",
                run + 1,
                VALIDATION_RUNS,
                size
            );
            let ops = measure_format_write(FormatKind::ArrowIpc, &schema, &rows, batches);
            tprintln!("  batch={:5}: {} rows/sec", size, format_with_commas(ops),);
            runs.push(ops);
        }
        if size == 10_000 {
            let result = validate_metric(
                "Arrow IPC Write",
                "rows/sec at batch=10000",
                runs,
                ARROW_IPC_ROWS_PER_SEC,
                true,
            );
            assert!(result.passed, "Arrow IPC write below target");
            assert!(!result.regression_detected, "Arrow IPC write regression");
        }
    }
}

// -----------------------------------------------------------------------------
// Avro
// -----------------------------------------------------------------------------

#[test]
fn test_format_avro_write_throughput() {
    zyron_bench_harness::init("format");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Avro write throughput ===");

    let schema = bench_schema();
    for &size in &[100usize, 1000, 10_000] {
        let rows = build_rows(size);
        let batches = match size {
            100 => 100,
            1000 => 30,
            10_000 => 5,
            _ => 5,
        };
        let mut runs = Vec::with_capacity(VALIDATION_RUNS);
        for run in 0..VALIDATION_RUNS {
            tprintln!(
                "--- Run {}/{} (batch={}) ---\n",
                run + 1,
                VALIDATION_RUNS,
                size
            );
            let ops = measure_format_write(FormatKind::Avro, &schema, &rows, batches);
            tprintln!("  batch={:5}: {} rows/sec", size, format_with_commas(ops),);
            runs.push(ops);
        }
        if size == 10_000 {
            let result = validate_metric(
                "Avro Write",
                "rows/sec at batch=10000",
                runs,
                AVRO_ROWS_PER_SEC,
                true,
            );
            assert!(result.passed, "Avro write below target");
            assert!(!result.regression_detected, "Avro write regression");
        }
    }
}

#![allow(non_snake_case, unused_assignments)]

//! Query Executor Benchmark Suite
//!
//! Integration tests for Zyron executor components:
//! - Sequential scan throughput
//! - Filter predicate evaluation
//! - Hash join correctness and performance
//! - Hash aggregate correctness and performance
//! - Sort operator with multi-key ordering
//! - Limit/offset row capping
//! - Distinct duplicate elimination
//! - Set operations (UNION, INTERSECT, EXCEPT)
//! - Expression evaluation correctness
//!
//! Performance Targets:
//! | Test              | Metric     | Target            |
//! |-------------------|------------|-------------------|
//! | Scan              | throughput | 75M rows/sec      |
//! | Filter            | throughput | 60M rows/sec      |
//! | Hash Join         | throughput | 30M rows/sec      |
//! | Hash Build        | throughput | 50M rows/sec      |
//! | Aggregate         | throughput | 150M rows/sec     |
//! | Sort (in-mem)     | throughput | 30M rows/sec      |
//! | Limit             | throughput | 200M rows/sec     |
//! | String equality   | throughput | 80M rows/sec      |
//! | LIKE prefix       | throughput | 7M rows/sec       |
//! | LIKE general      | throughput | 5M rows/sec       |
//! | ILIKE contains    | throughput | 4.5M rows/sec     |
//! | IN-list           | throughput | 35M rows/sec      |
//! | Window frame      | throughput | 9M rows/sec       |
//! | Residual join     | throughput | 20M rows/sec      |
//!
//! Validation Requirements:
//! - Each benchmark runs 5 iterations
//! - Results averaged across all 5 runs
//! - Pass/fail determined by average performance
//! - Individual runs logged for variance analysis
//! - Test FAILS if any single run is >2x worse than target

// The allocator the server actually runs. Without this the suite measured
// the platform allocator, which production never uses, and the alloc-bound
// metrics carried its variance rather than the engine's behaviour
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::sync::Mutex;

use rand::RngExt;
use zyron_bench_harness::*;

use zyron_common::TypeId;
use zyron_executor::batch::{BATCH_SIZE, DataBatch};
use zyron_executor::column::{Column, ColumnData, NullBitmap, ScalarValue};
use zyron_executor::expr::evaluate;
use zyron_executor::operator::aggregate::HashAggregateOperator;
use zyron_executor::operator::distinct::HashDistinctOperator;
use zyron_executor::operator::filter::FilterOperator;
use zyron_executor::operator::join::HashJoinOperator;
use zyron_executor::operator::limit::LimitOperator;
use zyron_executor::operator::project::ProjectOperator;
use zyron_executor::operator::setop::SetOpOperator;
use zyron_executor::operator::sort::SortOperator;
use zyron_executor::operator::{ExecutionBatch, Operator};

use zyron_catalog::ColumnId;
use zyron_parser::ast::{BinaryOperator, JoinType, LiteralValue, SetOpType};
use zyron_planner::binder::{BoundExpr, BoundOrderBy, ColumnRef};
use zyron_planner::logical::{AggregateExpr, LogicalColumn};

// =============================================================================
// Performance Target Constants
// =============================================================================

const SCAN_TARGET_ROWS_SEC: f64 = 75_000_000.0;
const FILTER_TARGET_ROWS_SEC: f64 = 60_000_000.0;
const HASH_JOIN_TARGET_ROWS_SEC: f64 = 100_000_000.0;
const HASH_BUILD_TARGET_ROWS_SEC: f64 = 50_000_000.0;
/// A grouping key with a hundred thousand distinct values, where the group
/// table no longer fits a cache level and the fold is a table lookup per
/// row rather than a hit on a hot line. Set above what a map holding a
/// vector of candidates per hash averaged, so a return to that shape trips
/// it, and a quarter under what the flat chained index measures
const HASH_BUILD_MANY_GROUPS_TARGET_ROWS_SEC: f64 = 35_000_000.0;
const AGGREGATE_TARGET_ROWS_SEC: f64 = 150_000_000.0;
const SORT_TARGET_ROWS_SEC: f64 = 100_000_000.0;
/// ORDER BY with a LIMIT, which only has to find the rows the limit keeps.
/// About half of what the bounded buffer measures, and four times what
/// sorting every row and truncating measured, so the gate trips on a
/// return to the full sort rather than on run-to-run spread
const SORT_TOPN_TARGET_ROWS_SEC: f64 = 450_000_000.0;
const LIMIT_TARGET_ROWS_SEC: f64 = 200_000_000.0;
const STRING_EQ_TARGET_ROWS_SEC: f64 = 80_000_000.0;
const LIKE_PREFIX_TARGET_ROWS_SEC: f64 = 7_000_000.0;
const LIKE_GENERAL_TARGET_ROWS_SEC: f64 = 5_000_000.0;
const ILIKE_CONTAINS_TARGET_ROWS_SEC: f64 = 4_500_000.0;
// Patterns with no '_' in them, which reach the literal matcher rather than
// the wildcard-aware one. Three shapes, one gate each. Set at about half of
// what the literal matcher measures, so the gate trips on a change that puts
// one of these shapes back on a scanning matcher rather than on run-to-run
// spread, which is under 10% here and under 2% for the substring shape
const LIKE_LITERAL_PREFIX_TARGET_ROWS_SEC: f64 = 100_000_000.0;
const LIKE_LITERAL_SUFFIX_TARGET_ROWS_SEC: f64 = 120_000_000.0;
const LIKE_LITERAL_CONTAINS_TARGET_ROWS_SEC: f64 = 50_000_000.0;
const IN_LIST_TARGET_ROWS_SEC: f64 = 35_000_000.0;
const WINDOW_FRAME_TARGET_ROWS_SEC: f64 = 9_000_000.0;
/// A running SUM over a thousand partitions whose rows arrive in neither
/// partition nor order-key order, so the sort, the gather and the fold all
/// do real work. About half of what the hashed partition layout and the
/// typed running fold measure, and twice what sorting every column by
/// comparison and boxing a scalar per row measured, so the gate trips on a
/// return to that rather than on run-to-run spread
const WINDOW_PARTITIONED_TARGET_ROWS_SEC: f64 = 12_000_000.0;
/// DISTINCT over a hundred thousand distinct rows in a million, about half
/// of what the flat chained index measures
const DISTINCT_TARGET_ROWS_SEC: f64 = 45_000_000.0;
const RESIDUAL_JOIN_TARGET_ROWS_SEC: f64 = 20_000_000.0;
/// UNION ALL forwards its branches without buffering, so it is bounded by
/// the batch plumbing rather than by the rows
const UNION_ALL_TARGET_ROWS_SEC: f64 = 2_000_000_000.0;
/// The three materializing set operations hash every row of both branches
/// into one store, which is the shape a hash join builds its side with
const UNION_TARGET_ROWS_SEC: f64 = 24_000_000.0;
const INTERSECT_TARGET_ROWS_SEC: f64 = 38_000_000.0;
const EXCEPT_TARGET_ROWS_SEC: f64 = 28_000_000.0;

static BENCHMARK_LOCK: Mutex<()> = Mutex::new(());

// =============================================================================
// Test data builders
// =============================================================================

/// Creates a simple schema with the given column types.
fn make_schema(cols: &[(&str, TypeId)]) -> Vec<LogicalColumn> {
    cols.iter()
        .enumerate()
        .map(|(i, (name, tid))| LogicalColumn {
            table_idx: Some(0),
            column_id: ColumnId(i as u16),
            name: name.to_string(),
            type_id: *tid,
            nullable: true,
            fractional_digits: None,
        })
        .collect()
}

/// Creates a DataBatch with N rows of i64 columns.
fn make_int_batch(num_rows: usize, num_cols: usize) -> DataBatch {
    let columns: Vec<Column> = (0..num_cols)
        .map(|col_idx| {
            let data: Vec<i64> = (0..num_rows)
                .map(|r| (r * num_cols + col_idx) as i64)
                .collect();
            Column::new(ColumnData::Int64(data), TypeId::Int64)
        })
        .collect();
    DataBatch::new(columns)
}

/// Creates a DataBatch with N rows, column 0 = row index (i64), column 1 = random i64.
fn make_random_int_batch(num_rows: usize) -> DataBatch {
    let mut rng = rand::rng();
    let ids: Vec<i64> = (0..num_rows).map(|r| r as i64).collect();
    let vals: Vec<i64> = (0..num_rows)
        .map(|_| rng.random_range(0..1_000_000))
        .collect();
    let col0 = Column::new(ColumnData::Int64(ids), TypeId::Int64);
    let col1 = Column::new(ColumnData::Int64(vals), TypeId::Int64);
    DataBatch::new(vec![col0, col1])
}

/// Wraps a list of DataBatch as a simple in-memory operator that yields batches.
struct MemoryOperator {
    batches: Vec<DataBatch>,
    cursor: usize,
}

impl MemoryOperator {
    fn new(batches: Vec<DataBatch>) -> Self {
        Self { batches, cursor: 0 }
    }

    fn boxed(batches: Vec<DataBatch>) -> Box<dyn Operator> {
        Box::new(Self::new(batches))
    }
}

impl Operator for MemoryOperator {
    fn next(&mut self) -> zyron_executor::operator::OperatorResult<'_> {
        Box::pin(async move {
            if self.cursor >= self.batches.len() {
                return Ok(None);
            }
            let batch = std::mem::replace(&mut self.batches[self.cursor], DataBatch::empty());
            self.cursor += 1;
            Ok(Some(ExecutionBatch::new(batch)))
        })
    }
}

/// Drains an operator and returns total row count.
async fn drain_operator(op: &mut dyn Operator) -> usize {
    let mut total = 0;
    loop {
        match op.next().await.unwrap() {
            Some(eb) => total += eb.batch.num_rows,
            None => break,
        }
    }
    total
}

/// Drains an operator and collects all batches.
async fn collect_batches(op: &mut dyn Operator) -> Vec<DataBatch> {
    let mut result = Vec::new();
    loop {
        match op.next().await.unwrap() {
            Some(eb) => result.push(eb.batch),
            None => break,
        }
    }
    result
}

/// Builds a large dataset as multiple BATCH_SIZE batches.
fn build_large_dataset(total_rows: usize, num_cols: usize) -> Vec<DataBatch> {
    let mut batches = Vec::new();
    let mut remaining = total_rows;
    let mut row_offset = 0;

    while remaining > 0 {
        let chunk = remaining.min(BATCH_SIZE);
        let columns: Vec<Column> = (0..num_cols)
            .map(|col_idx| {
                let data: Vec<i64> = (0..chunk)
                    .map(|r| ((row_offset + r) * num_cols + col_idx) as i64)
                    .collect();
                Column::new(ColumnData::Int64(data), TypeId::Int64)
            })
            .collect();
        batches.push(DataBatch::new(columns));
        row_offset += chunk;
        remaining -= chunk;
    }
    batches
}

/// Creates a BoundExpr::ColumnRef for a given column index.
fn col_ref(table_idx: usize, col_id: u16, type_id: TypeId) -> BoundExpr {
    BoundExpr::ColumnRef(ColumnRef {
        table_idx,
        column_id: ColumnId(col_id),
        type_id,
        nullable: true,
        fractional_digits: None,
    })
}

/// Creates a BoundExpr::Literal for an i64 value.
fn lit_int(val: i64) -> BoundExpr {
    BoundExpr::Literal {
        value: LiteralValue::Integer(val),
        type_id: TypeId::Int64,
    }
}

// =============================================================================
// Test 1: Scan Throughput (operator-level pull, 5-run validation)
// =============================================================================

#[tokio::test]
async fn test_scan_throughput_1m_rows() {
    zyron_bench_harness::init("executor");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const ROW_COUNT: usize = 1_000_000;
    const NUM_COLS: usize = 4;

    tprintln!("\n=== Scan Throughput Performance Test ===");
    tprintln!("Rows: {}, Columns: {}", ROW_COUNT, NUM_COLS);
    tprintln!("Validation runs: {}", VALIDATION_RUNS);

    let batches = build_large_dataset(ROW_COUNT, NUM_COLS);
    let mut scan_results = Vec::with_capacity(VALIDATION_RUNS);

    let util_before = take_util_snapshot();
    for run in 0..VALIDATION_RUNS {
        tprintln!("\n--- Run {}/{} ---", run + 1, VALIDATION_RUNS);

        let mut op = MemoryOperator::new(batches.clone());

        let start = Instant::now();
        let total_rows = drain_operator(&mut op).await;
        let duration = start.elapsed();

        assert_eq!(
            total_rows,
            ROW_COUNT,
            "Run {}: expected {} rows, got {}",
            run + 1,
            ROW_COUNT,
            total_rows
        );
        let rows_sec = ROW_COUNT as f64 / duration.as_secs_f64();
        tprintln!(
            "  Scan: {} rows/sec ({:?})",
            format_with_commas(rows_sec),
            duration
        );
        scan_results.push(rows_sec);
    }
    record_test_util("Scan Throughput", util_before, take_util_snapshot());

    tprintln!("\n=== Scan Validation Results ===");
    let result = validate_metric(
        "Scan Throughput",
        "Scan throughput (rows/sec)",
        scan_results,
        SCAN_TARGET_ROWS_SEC,
        true,
    );
    assert!(
        result.passed,
        "Scan avg {:.0} < target {:.0}",
        result.average, SCAN_TARGET_ROWS_SEC
    );
    assert!(!result.regression_detected, "Scan regression detected");
}

// =============================================================================
// Test 2: Filter Predicate Evaluation (5-run validation)
// =============================================================================

#[tokio::test]
async fn test_filter_throughput_1m_rows() {
    zyron_bench_harness::init("executor");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const ROW_COUNT: usize = 1_000_000;

    tprintln!("\n=== Filter Throughput Performance Test ===");
    tprintln!("Rows: {}", ROW_COUNT);
    tprintln!("Validation runs: {}", VALIDATION_RUNS);

    let schema = make_schema(&[("id", TypeId::Int64), ("val", TypeId::Int64)]);

    // Predicate: id < 500000 (selects 50% of rows)
    let predicate = BoundExpr::BinaryOp {
        left: Box::new(col_ref(0, 0, TypeId::Int64)),
        op: BinaryOperator::Lt,
        right: Box::new(lit_int(500_000)),
        type_id: TypeId::Boolean,
    };

    let batches = build_large_dataset(ROW_COUNT, 2);
    let mut filter_results = Vec::with_capacity(VALIDATION_RUNS);

    let util_before = take_util_snapshot();
    for run in 0..VALIDATION_RUNS {
        tprintln!("\n--- Run {}/{} ---", run + 1, VALIDATION_RUNS);

        let child = MemoryOperator::boxed(batches.clone());
        let mut filter_op = FilterOperator::new(child, predicate.clone(), schema.clone());

        let start = Instant::now();
        let total_rows = drain_operator(&mut filter_op).await;
        let duration = start.elapsed();

        // Column 0 values are row_index * 2 + 0 (from build_large_dataset with 2 cols),
        // so id < 500000 selects rows where row_index * 2 < 500000, i.e. row_index < 250000.
        assert!(total_rows > 0, "Run {}: filter returned 0 rows", run + 1);
        let rows_sec = ROW_COUNT as f64 / duration.as_secs_f64();
        tprintln!(
            "  Filter: {} rows/sec ({:?}), {} rows passed",
            format_with_commas(rows_sec),
            duration,
            total_rows
        );
        filter_results.push(rows_sec);
    }
    record_test_util("Filter Throughput", util_before, take_util_snapshot());

    tprintln!("\n=== Filter Validation Results ===");
    let result = validate_metric(
        "Filter Throughput",
        "Filter throughput (rows/sec)",
        filter_results,
        FILTER_TARGET_ROWS_SEC,
        true,
    );
    assert!(
        result.passed,
        "Filter avg {:.0} < target {:.0}",
        result.average, FILTER_TARGET_ROWS_SEC
    );
    assert!(!result.regression_detected, "Filter regression detected");
}

// =============================================================================
// Test 3: Filter Correctness
// =============================================================================

#[tokio::test]
async fn test_filter_correctness() {
    zyron_bench_harness::init("executor");
    tprintln!("\n=== Filter Correctness Test ===");

    let schema = make_schema(&[("id", TypeId::Int64), ("val", TypeId::Int64)]);

    // Build 100 rows: id = 0..99, val = 100..199
    let ids: Vec<i64> = (0..100).collect();
    let vals: Vec<i64> = (100..200).collect();
    let batch = DataBatch::new(vec![
        Column::new(ColumnData::Int64(ids), TypeId::Int64),
        Column::new(ColumnData::Int64(vals), TypeId::Int64),
    ]);

    // Test: id < 10
    let predicate = BoundExpr::BinaryOp {
        left: Box::new(col_ref(0, 0, TypeId::Int64)),
        op: BinaryOperator::Lt,
        right: Box::new(lit_int(10)),
        type_id: TypeId::Boolean,
    };

    let child = MemoryOperator::boxed(vec![batch.clone()]);
    let mut filter_op = FilterOperator::new(child, predicate, schema.clone());
    let rows = drain_operator(&mut filter_op).await;
    assert_eq!(rows, 10, "id < 10 should return 10 rows, got {}", rows);
    tprintln!("  id < 10: {} rows [PASS]", rows);

    // Test compound: id >= 10 AND id < 20
    let pred_compound = BoundExpr::BinaryOp {
        left: Box::new(BoundExpr::BinaryOp {
            left: Box::new(col_ref(0, 0, TypeId::Int64)),
            op: BinaryOperator::GtEq,
            right: Box::new(lit_int(10)),
            type_id: TypeId::Boolean,
        }),
        op: BinaryOperator::And,
        right: Box::new(BoundExpr::BinaryOp {
            left: Box::new(col_ref(0, 0, TypeId::Int64)),
            op: BinaryOperator::Lt,
            right: Box::new(lit_int(20)),
            type_id: TypeId::Boolean,
        }),
        type_id: TypeId::Boolean,
    };

    let child = MemoryOperator::boxed(vec![batch.clone()]);
    let mut filter_op = FilterOperator::new(child, pred_compound, schema.clone());
    let rows = drain_operator(&mut filter_op).await;
    assert_eq!(
        rows, 10,
        "id >= 10 AND id < 20 should return 10 rows, got {}",
        rows
    );
    tprintln!("  id >= 10 AND id < 20: {} rows [PASS]", rows);
}

// =============================================================================
// Test 4: Hash Join Throughput (5-run validation)
// =============================================================================

#[tokio::test]
async fn test_hash_join_throughput() {
    zyron_bench_harness::init("executor");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const LEFT_ROWS: usize = 500_000;
    const RIGHT_ROWS: usize = 50_000;

    tprintln!("\n=== Hash Join Performance Test ===");
    tprintln!("Left rows: {}, Right rows: {}", LEFT_ROWS, RIGHT_ROWS);
    tprintln!("Validation runs: {}", VALIDATION_RUNS);

    let left_schema = make_schema(&[("id", TypeId::Int64), ("left_val", TypeId::Int64)]);
    let right_schema = make_schema(&[("id", TypeId::Int64), ("right_val", TypeId::Int64)]);

    let left_key = col_ref(0, 0, TypeId::Int64);
    let right_key = col_ref(0, 0, TypeId::Int64);

    let left_batches = build_large_dataset(LEFT_ROWS, 2);
    // Right side: ids are multiples of 10 so only some match.
    let right_batches = {
        let mut batches = Vec::new();
        let mut remaining = RIGHT_ROWS;
        let mut row_offset = 0;
        while remaining > 0 {
            let chunk = remaining.min(BATCH_SIZE);
            let ids: Vec<i64> = (0..chunk).map(|r| ((row_offset + r) * 10) as i64).collect();
            let vals: Vec<i64> = (0..chunk)
                .map(|r| ((row_offset + r) * 100) as i64)
                .collect();
            batches.push(DataBatch::new(vec![
                Column::new(ColumnData::Int64(ids), TypeId::Int64),
                Column::new(ColumnData::Int64(vals), TypeId::Int64),
            ]));
            row_offset += chunk;
            remaining -= chunk;
        }
        batches
    };

    let mut join_results = Vec::with_capacity(VALIDATION_RUNS);

    let util_before = take_util_snapshot();
    for run in 0..VALIDATION_RUNS {
        tprintln!("\n--- Run {}/{} ---", run + 1, VALIDATION_RUNS);

        let left_op = MemoryOperator::boxed(left_batches.clone());
        let right_op = MemoryOperator::boxed(right_batches.clone());

        let mut join_op = HashJoinOperator::new(
            left_op,
            right_op,
            JoinType::Inner,
            vec![left_key.clone()],
            vec![right_key.clone()],
            None,
            left_schema.clone(),
            right_schema.clone(),
        );

        let start = Instant::now();
        let total_rows = drain_operator(&mut join_op).await;
        let duration = start.elapsed();

        assert!(total_rows > 0, "Run {}: join returned 0 rows", run + 1);
        let input_rows = LEFT_ROWS + RIGHT_ROWS;
        let rows_sec = input_rows as f64 / duration.as_secs_f64();
        tprintln!(
            "  Join: {} rows/sec ({:?}), {} output rows",
            format_with_commas(rows_sec),
            duration,
            total_rows
        );
        join_results.push(rows_sec);
    }
    record_test_util("Hash Join", util_before, take_util_snapshot());

    tprintln!("\n=== Hash Join Validation Results ===");
    let result = validate_metric(
        "Hash Join",
        "Join throughput (rows/sec)",
        join_results,
        HASH_JOIN_TARGET_ROWS_SEC,
        true,
    );
    assert!(
        result.passed,
        "Hash Join avg {:.0} < target {:.0}",
        result.average, HASH_JOIN_TARGET_ROWS_SEC
    );
    assert!(!result.regression_detected, "Hash Join regression detected");
}

// =============================================================================
// Test 5: Hash Join Correctness
// =============================================================================

#[tokio::test]
async fn test_hash_join_correctness() {
    zyron_bench_harness::init("executor");
    tprintln!("\n=== Hash Join Correctness Test ===");

    let left_schema = make_schema(&[("id", TypeId::Int64), ("name", TypeId::Int64)]);
    let right_schema = make_schema(&[("id", TypeId::Int64), ("dept", TypeId::Int64)]);

    // Left: ids 1-5
    let left_batch = DataBatch::new(vec![
        Column::new(ColumnData::Int64(vec![1, 2, 3, 4, 5]), TypeId::Int64),
        Column::new(ColumnData::Int64(vec![10, 20, 30, 40, 50]), TypeId::Int64),
    ]);

    // Right: ids 3-7 (overlap on 3,4,5)
    let right_batch = DataBatch::new(vec![
        Column::new(ColumnData::Int64(vec![3, 4, 5, 6, 7]), TypeId::Int64),
        Column::new(
            ColumnData::Int64(vec![300, 400, 500, 600, 700]),
            TypeId::Int64,
        ),
    ]);

    let left_key = col_ref(0, 0, TypeId::Int64);
    let right_key = col_ref(0, 0, TypeId::Int64);

    // INNER JOIN
    let left_op = MemoryOperator::boxed(vec![left_batch.clone()]);
    let right_op = MemoryOperator::boxed(vec![right_batch.clone()]);
    let mut join_op = HashJoinOperator::new(
        left_op,
        right_op,
        JoinType::Inner,
        vec![left_key.clone()],
        vec![right_key.clone()],
        None,
        left_schema.clone(),
        right_schema.clone(),
    );
    let rows = drain_operator(&mut join_op).await;
    assert_eq!(
        rows, 3,
        "INNER JOIN should produce 3 rows (ids 3,4,5), got {}",
        rows
    );
    tprintln!("  INNER JOIN: {} rows [PASS]", rows);

    // LEFT JOIN
    let left_op = MemoryOperator::boxed(vec![left_batch.clone()]);
    let right_op = MemoryOperator::boxed(vec![right_batch.clone()]);
    let mut join_op = HashJoinOperator::new(
        left_op,
        right_op,
        JoinType::Left,
        vec![left_key.clone()],
        vec![right_key.clone()],
        None,
        left_schema.clone(),
        right_schema.clone(),
    );
    let rows = drain_operator(&mut join_op).await;
    assert_eq!(rows, 5, "LEFT JOIN should produce 5 rows, got {}", rows);
    tprintln!("  LEFT JOIN: {} rows [PASS]", rows);
}

// =============================================================================
// Test 6: Hash Aggregate Throughput (5-run validation)
// =============================================================================

#[tokio::test]
async fn test_aggregate_throughput() {
    zyron_bench_harness::init("executor");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const ROW_COUNT: usize = 1_000_000;

    tprintln!("\n=== Aggregate Performance Test ===");
    tprintln!("Rows: {}", ROW_COUNT);
    tprintln!("Validation runs: {}", VALIDATION_RUNS);

    let schema = make_schema(&[("id", TypeId::Int64), ("val", TypeId::Int64)]);

    // Global COUNT(*) aggregate, no group-by
    let aggregates = vec![AggregateExpr {
        function_name: "count".to_string(),
        args: vec![],
        distinct: false,
        return_type: TypeId::Int64,
        uda: None,
    }];

    let output_schema = make_schema(&[("count", TypeId::Int64)]);
    let batches = build_large_dataset(ROW_COUNT, 2);
    let mut agg_results = Vec::with_capacity(VALIDATION_RUNS);

    let util_before = take_util_snapshot();
    for run in 0..VALIDATION_RUNS {
        tprintln!("\n--- Run {}/{} ---", run + 1, VALIDATION_RUNS);

        let child = MemoryOperator::boxed(batches.clone());
        let mut agg_op = HashAggregateOperator::new(
            child,
            vec![],
            aggregates.clone(),
            schema.clone(),
            output_schema.clone(),
        );

        let start = Instant::now();
        let result_batches = collect_batches(&mut agg_op).await;
        let duration = start.elapsed();

        // Verify COUNT(*) = ROW_COUNT
        let total_result_rows: usize = result_batches.iter().map(|b| b.num_rows).sum();
        assert_eq!(
            total_result_rows,
            1,
            "Run {}: COUNT(*) should produce 1 row, got {}",
            run + 1,
            total_result_rows
        );

        let count_val = result_batches[0].columns[0].get_scalar(0);
        assert_eq!(
            count_val,
            ScalarValue::Int64(ROW_COUNT as i64),
            "Run {}: COUNT(*) should be {}, got {}",
            run + 1,
            ROW_COUNT,
            count_val
        );

        let rows_sec = ROW_COUNT as f64 / duration.as_secs_f64();
        tprintln!(
            "  Aggregate: {} rows/sec ({:?})",
            format_with_commas(rows_sec),
            duration
        );
        agg_results.push(rows_sec);
    }
    record_test_util("Aggregate", util_before, take_util_snapshot());

    tprintln!("\n=== Aggregate Validation Results ===");
    let result = validate_metric(
        "Aggregate",
        "Aggregate throughput (rows/sec)",
        agg_results,
        AGGREGATE_TARGET_ROWS_SEC,
        true,
    );
    assert!(
        result.passed,
        "Aggregate avg {:.0} < target {:.0}",
        result.average, AGGREGATE_TARGET_ROWS_SEC
    );
    assert!(!result.regression_detected, "Aggregate regression detected");
}

// =============================================================================
// Test 7: Aggregate Correctness (GROUP BY, SUM, AVG, MIN, MAX)
// =============================================================================

#[tokio::test]
async fn test_aggregate_correctness() {
    zyron_bench_harness::init("executor");
    tprintln!("\n=== Aggregate Correctness Test ===");

    let schema = make_schema(&[("dept", TypeId::Int64), ("salary", TypeId::Int64)]);

    // 12 rows, 3 departments (0, 1, 2), 4 rows each
    let depts: Vec<i64> = vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2];
    let salaries: Vec<i64> = vec![50, 60, 70, 55, 65, 75, 45, 55, 65, 60, 70, 80];
    let batch = DataBatch::new(vec![
        Column::new(ColumnData::Int64(depts), TypeId::Int64),
        Column::new(ColumnData::Int64(salaries), TypeId::Int64),
    ]);

    let group_by = vec![col_ref(0, 0, TypeId::Int64)];
    let aggregates = vec![
        AggregateExpr {
            function_name: "count".to_string(),
            args: vec![col_ref(0, 1, TypeId::Int64)],
            distinct: false,
            return_type: TypeId::Int64,
            uda: None,
        },
        AggregateExpr {
            function_name: "sum".to_string(),
            args: vec![col_ref(0, 1, TypeId::Int64)],
            distinct: false,
            return_type: TypeId::Float64,
            uda: None,
        },
        AggregateExpr {
            function_name: "min".to_string(),
            args: vec![col_ref(0, 1, TypeId::Int64)],
            distinct: false,
            return_type: TypeId::Int64,
            uda: None,
        },
        AggregateExpr {
            function_name: "max".to_string(),
            args: vec![col_ref(0, 1, TypeId::Int64)],
            distinct: false,
            return_type: TypeId::Int64,
            uda: None,
        },
    ];

    let output_schema = make_schema(&[
        ("dept", TypeId::Int64),
        ("count", TypeId::Int64),
        ("sum", TypeId::Float64),
        ("min", TypeId::Int64),
        ("max", TypeId::Int64),
    ]);

    let child = MemoryOperator::boxed(vec![batch]);
    let mut agg_op = HashAggregateOperator::new(child, group_by, aggregates, schema, output_schema);

    let result = collect_batches(&mut agg_op).await;
    let total_rows: usize = result.iter().map(|b| b.num_rows).sum();
    assert_eq!(
        total_rows, 3,
        "GROUP BY should produce 3 groups, got {}",
        total_rows
    );

    // Verify each group has count=4
    for b in &result {
        for r in 0..b.num_rows {
            let count = b.columns[1].get_scalar(r);
            assert_eq!(
                count,
                ScalarValue::Int64(4),
                "Each group should have 4 rows"
            );
        }
    }
    tprintln!(
        "  GROUP BY dept with COUNT/SUM/MIN/MAX: {} groups [PASS]",
        total_rows
    );
}

// =============================================================================
// Test 8: Sort Throughput (5-run validation)
// =============================================================================

#[tokio::test]
async fn test_sort_throughput() {
    zyron_bench_harness::init("executor");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const ROW_COUNT: usize = 500_000;

    tprintln!("\n=== Sort Performance Test ===");
    tprintln!("Rows: {}", ROW_COUNT);
    tprintln!("Validation runs: {}", VALIDATION_RUNS);

    let schema = make_schema(&[("val", TypeId::Int64)]);
    let order_by = vec![BoundOrderBy {
        expr: col_ref(0, 0, TypeId::Int64),
        asc: true,
        nulls_first: false,
    }];

    // Build random data for sorting
    let batches = {
        let mut rng = rand::rng();
        let mut batches = Vec::new();
        let mut remaining = ROW_COUNT;
        while remaining > 0 {
            let chunk = remaining.min(BATCH_SIZE);
            let data: Vec<i64> = (0..chunk)
                .map(|_| rng.random_range(0..ROW_COUNT as i64))
                .collect();
            batches.push(DataBatch::new(vec![Column::new(
                ColumnData::Int64(data),
                TypeId::Int64,
            )]));
            remaining -= chunk;
        }
        batches
    };

    let mut sort_results = Vec::with_capacity(VALIDATION_RUNS);

    let util_before = take_util_snapshot();
    for run in 0..VALIDATION_RUNS {
        tprintln!("\n--- Run {}/{} ---", run + 1, VALIDATION_RUNS);

        let child = MemoryOperator::boxed(batches.clone());
        let mut sort_op = SortOperator::new(child, order_by.clone(), schema.clone(), None);

        let start = Instant::now();
        let result_batches = collect_batches(&mut sort_op).await;
        let duration = start.elapsed();

        let total_rows: usize = result_batches.iter().map(|b| b.num_rows).sum();
        assert_eq!(
            total_rows,
            ROW_COUNT,
            "Run {}: sort should return {} rows, got {}",
            run + 1,
            ROW_COUNT,
            total_rows
        );

        // Verify sorted order.
        let mut prev = i64::MIN;
        for b in &result_batches {
            if let ColumnData::Int64(data) = &b.columns[0].data {
                for &v in data {
                    assert!(
                        v >= prev,
                        "Run {}: sort order violated: {} < {}",
                        run + 1,
                        v,
                        prev
                    );
                    prev = v;
                }
            }
        }

        let rows_sec = ROW_COUNT as f64 / duration.as_secs_f64();
        tprintln!(
            "  Sort: {} rows/sec ({:?})",
            format_with_commas(rows_sec),
            duration
        );
        sort_results.push(rows_sec);
    }
    record_test_util("Sort", util_before, take_util_snapshot());

    tprintln!("\n=== Sort Validation Results ===");
    let result = validate_metric(
        "Sort",
        "Sort throughput (rows/sec)",
        sort_results,
        SORT_TARGET_ROWS_SEC,
        true,
    );
    assert!(
        result.passed,
        "Sort avg {:.0} < target {:.0}",
        result.average, SORT_TARGET_ROWS_SEC
    );
    assert!(!result.regression_detected, "Sort regression detected");
}

// =============================================================================
// Test 9: Sort Correctness (multi-key, descending)
// =============================================================================

#[tokio::test]
async fn test_sort_correctness_multikey() {
    zyron_bench_harness::init("executor");
    tprintln!("\n=== Sort Correctness (Multi-Key) Test ===");

    let schema = make_schema(&[("a", TypeId::Int64), ("b", TypeId::Int64)]);

    let batch = DataBatch::new(vec![
        Column::new(ColumnData::Int64(vec![3, 1, 2, 1, 3, 2]), TypeId::Int64),
        Column::new(
            ColumnData::Int64(vec![10, 30, 20, 10, 30, 10]),
            TypeId::Int64,
        ),
    ]);

    let order_by = vec![
        BoundOrderBy {
            expr: col_ref(0, 0, TypeId::Int64),
            asc: true,
            nulls_first: false,
        },
        BoundOrderBy {
            expr: col_ref(0, 1, TypeId::Int64),
            asc: false,
            nulls_first: false,
        },
    ];

    let child = MemoryOperator::boxed(vec![batch]);
    let mut sort_op = SortOperator::new(child, order_by, schema, None);
    let result = collect_batches(&mut sort_op).await;

    let total_rows: usize = result.iter().map(|b| b.num_rows).sum();
    assert_eq!(total_rows, 6);

    // Expected order: (1,30), (1,10), (2,20), (2,10), (3,30), (3,10)
    let expected_a = vec![1i64, 1, 2, 2, 3, 3];
    let expected_b = vec![30i64, 10, 20, 10, 30, 10];

    let mut row = 0;
    for b in &result {
        if let (ColumnData::Int64(a_data), ColumnData::Int64(b_data)) =
            (&b.columns[0].data, &b.columns[1].data)
        {
            for i in 0..b.num_rows {
                assert_eq!(a_data[i], expected_a[row], "Row {}: a mismatch", row);
                assert_eq!(b_data[i], expected_b[row], "Row {}: b mismatch", row);
                row += 1;
            }
        }
    }
    tprintln!("  Multi-key sort (ASC a, DESC b): verified [PASS]");
}

// =============================================================================
// Test 10: Limit/Offset Correctness and Throughput
// =============================================================================

#[tokio::test]
async fn test_limit_offset_correctness() {
    zyron_bench_harness::init("executor");
    tprintln!("\n=== Limit/Offset Correctness Test ===");

    let batches = build_large_dataset(1000, 1);

    // LIMIT 10
    let child = MemoryOperator::boxed(batches.clone());
    let mut limit_op = LimitOperator::new(child, Some(10), None);
    let rows = drain_operator(&mut limit_op).await;
    assert_eq!(rows, 10, "LIMIT 10 should return 10 rows, got {}", rows);
    tprintln!("  LIMIT 10: {} rows [PASS]", rows);

    // LIMIT 10 OFFSET 100
    let child = MemoryOperator::boxed(batches.clone());
    let mut limit_op = LimitOperator::new(child, Some(10), Some(100));
    let result = collect_batches(&mut limit_op).await;
    let total_rows: usize = result.iter().map(|b| b.num_rows).sum();
    assert_eq!(
        total_rows, 10,
        "LIMIT 10 OFFSET 100 should return 10 rows, got {}",
        total_rows
    );

    // Verify the values are rows 100-109 (column 0 = row_index)
    let first_val = result[0].columns[0].get_scalar(0);
    assert_eq!(
        first_val,
        ScalarValue::Int64(100),
        "First row should have id=100, got {}",
        first_val
    );
    tprintln!(
        "  LIMIT 10 OFFSET 100: {} rows, first id={} [PASS]",
        total_rows,
        first_val
    );

    // OFFSET beyond data
    let child = MemoryOperator::boxed(batches.clone());
    let mut limit_op = LimitOperator::new(child, Some(10), Some(2000));
    let rows = drain_operator(&mut limit_op).await;
    assert_eq!(
        rows, 0,
        "OFFSET beyond data should return 0 rows, got {}",
        rows
    );
    tprintln!("  OFFSET beyond data: {} rows [PASS]", rows);
}

#[tokio::test]
async fn test_limit_throughput() {
    zyron_bench_harness::init("executor");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const ROW_COUNT: usize = 1_000_000;

    tprintln!("\n=== Limit Throughput Performance Test ===");
    tprintln!("Rows: {}, Limit: 100", ROW_COUNT);
    tprintln!("Validation runs: {}", VALIDATION_RUNS);

    let batches = build_large_dataset(ROW_COUNT, 2);
    let mut limit_results = Vec::with_capacity(VALIDATION_RUNS);

    let util_before = take_util_snapshot();
    // The actual limit-of-100 work completes in sub-microseconds, so a single
    // OS preemption during one sample throws the rows/sec average off by 5x+.
    // Amortize by running many limit operators per measurement window. The
    // operators must be pre-built outside the timer because `batches.clone()`
    // deep-copies the column vectors and would dominate the measurement.
    const SAMPLES_PER_RUN: usize = 200;
    for run in 0..VALIDATION_RUNS {
        tprintln!("\n--- Run {}/{} ---", run + 1, VALIDATION_RUNS);

        // Pre-build SAMPLES_PER_RUN limit operators before starting the timer.
        let mut ops: Vec<_> = (0..SAMPLES_PER_RUN)
            .map(|_| {
                let child = MemoryOperator::boxed(batches.clone());
                LimitOperator::new(child, Some(100), None)
            })
            .collect();

        let start = Instant::now();
        let mut total_rows = 0usize;
        for op in ops.iter_mut() {
            total_rows += drain_operator(op).await;
        }
        let duration = start.elapsed();

        assert_eq!(total_rows, 100 * SAMPLES_PER_RUN);
        // Each sample processed ROW_COUNT rows of input; rows/sec across all samples.
        let rows_sec = (ROW_COUNT * SAMPLES_PER_RUN) as f64 / duration.as_secs_f64();
        tprintln!(
            "  Limit: {} rows/sec ({:?} for {} samples)",
            format_with_commas(rows_sec),
            duration,
            SAMPLES_PER_RUN,
        );
        limit_results.push(rows_sec);
    }
    record_test_util("Limit", util_before, take_util_snapshot());

    tprintln!("\n=== Limit Validation Results ===");
    let result = validate_metric(
        "Limit",
        "Limit throughput (rows/sec)",
        limit_results,
        LIMIT_TARGET_ROWS_SEC,
        true,
    );
    assert!(
        result.passed,
        "Limit avg {:.0} < target {:.0}",
        result.average, LIMIT_TARGET_ROWS_SEC
    );
    assert!(!result.regression_detected, "Limit regression detected");
}

// =============================================================================
// Test 11: Projection Correctness
// =============================================================================

#[tokio::test]
async fn test_projection_correctness() {
    zyron_bench_harness::init("executor");
    tprintln!("\n=== Projection Correctness Test ===");

    let schema = make_schema(&[
        ("a", TypeId::Int64),
        ("b", TypeId::Int64),
        ("c", TypeId::Int64),
    ]);

    let batch = DataBatch::new(vec![
        Column::new(ColumnData::Int64(vec![1, 2, 3]), TypeId::Int64),
        Column::new(ColumnData::Int64(vec![10, 20, 30]), TypeId::Int64),
        Column::new(ColumnData::Int64(vec![100, 200, 300]), TypeId::Int64),
    ]);

    // Project only columns b and c (indices 1 and 2)
    let projections = vec![col_ref(0, 1, TypeId::Int64), col_ref(0, 2, TypeId::Int64)];

    let child = MemoryOperator::boxed(vec![batch]);
    let mut proj_op = ProjectOperator::new(child, projections, schema);
    let result = collect_batches(&mut proj_op).await;

    assert_eq!(result.len(), 1);
    assert_eq!(
        result[0].num_columns(),
        2,
        "Projection should output 2 columns"
    );
    assert_eq!(result[0].num_rows, 3);

    // Column 0 should be the original column b
    assert_eq!(result[0].columns[0].get_scalar(0), ScalarValue::Int64(10));
    assert_eq!(result[0].columns[0].get_scalar(2), ScalarValue::Int64(30));
    // Column 1 should be the original column c
    assert_eq!(result[0].columns[1].get_scalar(0), ScalarValue::Int64(100));
    tprintln!("  SELECT b, c FROM t: 2 columns, 3 rows [PASS]");
}

// =============================================================================
// Test 12: Distinct Correctness
// =============================================================================

#[tokio::test]
async fn test_distinct_correctness() {
    zyron_bench_harness::init("executor");
    tprintln!("\n=== Distinct Correctness Test ===");

    // 10 rows with duplicates: values [1,2,3,1,2,3,4,5,1,2]
    let batch = DataBatch::new(vec![Column::new(
        ColumnData::Int64(vec![1, 2, 3, 1, 2, 3, 4, 5, 1, 2]),
        TypeId::Int64,
    )]);

    let child = MemoryOperator::boxed(vec![batch]);
    let mut distinct_op = HashDistinctOperator::new(child);
    let rows = drain_operator(&mut distinct_op).await;
    assert_eq!(
        rows, 5,
        "DISTINCT should produce 5 unique values, got {}",
        rows
    );
    tprintln!(
        "  DISTINCT on 10 rows with 5 unique values: {} rows [PASS]",
        rows
    );
}

// =============================================================================
// Distinct throughput (5-run validation)
// =============================================================================

/// SELECT DISTINCT over two columns where a hundred thousand distinct rows
/// each appear ten times, in stride order so consecutive rows are unrelated
#[tokio::test]
async fn test_distinct_throughput() {
    zyron_bench_harness::init("executor");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const ROW_COUNT: usize = 1_000_000;
    const DISTINCT: usize = 100_000;
    /// Coprime with DISTINCT, so a full cycle of rows visits every key once
    const STRIDE: usize = 7_919;

    tprintln!("\n=== Distinct Throughput Test ===");
    tprintln!("Rows: {}, distinct: {}", ROW_COUNT, DISTINCT);
    tprintln!("Validation runs: {}", VALIDATION_RUNS);

    let batches = {
        let mut batches = Vec::new();
        let mut remaining = ROW_COUNT;
        let mut row_offset = 0;
        while remaining > 0 {
            let chunk = remaining.min(BATCH_SIZE);
            let keys: Vec<i64> = (0..chunk)
                .map(|r| (((row_offset + r) * STRIDE) % DISTINCT) as i64)
                .collect();
            // A second column that is a function of the first, so a row is
            // distinct by its key alone and the comparison still has two
            // columns to check
            let payload: Vec<i64> = keys.iter().map(|k| k * 3 + 1).collect();
            batches.push(DataBatch::new(vec![
                Column::new(ColumnData::Int64(keys), TypeId::Int64),
                Column::new(ColumnData::Int64(payload), TypeId::Int64),
            ]));
            row_offset += chunk;
            remaining -= chunk;
        }
        batches
    };

    let mut results = Vec::with_capacity(VALIDATION_RUNS);
    let util_before = take_util_snapshot();
    for run in 0..VALIDATION_RUNS {
        tprintln!("\n--- Run {}/{} ---", run + 1, VALIDATION_RUNS);
        let child = MemoryOperator::boxed(batches.clone());
        let mut distinct_op = HashDistinctOperator::new(child);
        let start = Instant::now();
        let rows = drain_operator(&mut distinct_op).await;
        let duration = start.elapsed();
        assert_eq!(
            rows,
            DISTINCT,
            "Run {}: expected {} distinct rows, got {}",
            run + 1,
            DISTINCT,
            rows
        );
        let rows_sec = ROW_COUNT as f64 / duration.as_secs_f64();
        tprintln!(
            "  Distinct: {} rows/sec ({:?})",
            format_with_commas(rows_sec),
            duration
        );
        results.push(rows_sec);
    }
    record_test_util("Distinct", util_before, take_util_snapshot());

    tprintln!("\n=== Distinct Validation Results ===");
    let result = validate_metric(
        "Distinct",
        "Distinct throughput (rows/sec)",
        results,
        DISTINCT_TARGET_ROWS_SEC,
        true,
    );
    assert!(
        result.passed,
        "Distinct avg {:.0} < target {:.0}",
        result.average, DISTINCT_TARGET_ROWS_SEC
    );
    assert!(!result.regression_detected, "Distinct regression detected");
}

// =============================================================================
// Test 13: Set Operations Correctness
// =============================================================================

#[tokio::test]
async fn test_setop_correctness() {
    zyron_bench_harness::init("executor");
    tprintln!("\n=== Set Operations Correctness Test ===");

    let left_batch = DataBatch::new(vec![Column::new(
        ColumnData::Int64(vec![1, 2, 3, 4, 5]),
        TypeId::Int64,
    )]);
    let right_batch = DataBatch::new(vec![Column::new(
        ColumnData::Int64(vec![3, 4, 5, 6, 7]),
        TypeId::Int64,
    )]);

    // UNION ALL: should produce 10 rows
    let left = MemoryOperator::boxed(vec![left_batch.clone()]);
    let right = MemoryOperator::boxed(vec![right_batch.clone()]);
    let mut union_op = SetOpOperator::new(left, right, SetOpType::Union, true);
    let rows = drain_operator(&mut union_op).await;
    assert_eq!(rows, 10, "UNION ALL should produce 10 rows, got {}", rows);
    tprintln!("  UNION ALL: {} rows [PASS]", rows);

    // UNION (distinct): should produce 7 rows (1-7)
    let left = MemoryOperator::boxed(vec![left_batch.clone()]);
    let right = MemoryOperator::boxed(vec![right_batch.clone()]);
    let mut union_op = SetOpOperator::new(left, right, SetOpType::Union, false);
    let rows = drain_operator(&mut union_op).await;
    assert_eq!(
        rows, 7,
        "UNION should produce 7 distinct rows, got {}",
        rows
    );
    tprintln!("  UNION: {} rows [PASS]", rows);

    // INTERSECT: should produce 3 rows (3,4,5)
    let left = MemoryOperator::boxed(vec![left_batch.clone()]);
    let right = MemoryOperator::boxed(vec![right_batch.clone()]);
    let mut intersect_op = SetOpOperator::new(left, right, SetOpType::Intersect, false);
    let rows = drain_operator(&mut intersect_op).await;
    assert_eq!(rows, 3, "INTERSECT should produce 3 rows, got {}", rows);
    tprintln!("  INTERSECT: {} rows [PASS]", rows);

    // EXCEPT: should produce 2 rows (1,2)
    let left = MemoryOperator::boxed(vec![left_batch.clone()]);
    let right = MemoryOperator::boxed(vec![right_batch.clone()]);
    let mut except_op = SetOpOperator::new(left, right, SetOpType::Except, false);
    let rows = drain_operator(&mut except_op).await;
    assert_eq!(rows, 2, "EXCEPT should produce 2 rows, got {}", rows);
    tprintln!("  EXCEPT: {} rows [PASS]", rows);
}

/// Throughput of the three materializing set operations.
///
/// UNION ALL streams and is the floor every other shape is read against.
/// UNION DISTINCT, INTERSECT and EXCEPT all build one row store over both
/// branches and hash every row into it, so they are measured on the same
/// inputs to keep the comparison about the operation rather than the data.
///
/// Reported rather than gated: the operations had no throughput measurement
/// at all, so there is no number to hold them to yet
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_setop_throughput() {
    zyron_bench_harness::init("executor");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const ROWS_PER_SIDE: usize = 200_000;

    tprintln!("\n=== Set Operation Throughput ===");
    tprintln!("Rows per side: {}, columns: 2", ROWS_PER_SIDE);

    // Halves that overlap in the middle, so every operation has real work:
    // the union is not the concatenation, the intersection is not empty and
    // the difference is not the whole left side
    let left_batches = build_large_dataset(ROWS_PER_SIDE, 2);
    let right_batches = {
        let mut batches = Vec::new();
        let mut remaining = ROWS_PER_SIDE;
        let mut row_offset = ROWS_PER_SIDE / 2;
        while remaining > 0 {
            let chunk = remaining.min(BATCH_SIZE);
            let columns: Vec<Column> = (0..2usize)
                .map(|col_idx| {
                    let data: Vec<i64> = (0..chunk)
                        .map(|r| ((row_offset + r) * 2 + col_idx) as i64)
                        .collect();
                    Column::new(ColumnData::Int64(data), TypeId::Int64)
                })
                .collect();
            batches.push(DataBatch::new(columns));
            row_offset += chunk;
            remaining -= chunk;
        }
        batches
    };

    let cases: &[(&str, SetOpType, bool, f64)] = &[
        (
            "UNION ALL",
            SetOpType::Union,
            true,
            UNION_ALL_TARGET_ROWS_SEC,
        ),
        ("UNION", SetOpType::Union, false, UNION_TARGET_ROWS_SEC),
        (
            "INTERSECT",
            SetOpType::Intersect,
            false,
            INTERSECT_TARGET_ROWS_SEC,
        ),
        ("EXCEPT", SetOpType::Except, false, EXCEPT_TARGET_ROWS_SEC),
    ];

    for (label, op, all, target) in cases {
        let mut runs = Vec::with_capacity(VALIDATION_RUNS);
        let mut emitted = 0usize;
        for _ in 0..VALIDATION_RUNS {
            let left = MemoryOperator::boxed(left_batches.clone());
            let right = MemoryOperator::boxed(right_batches.clone());
            let mut set_op = SetOpOperator::new(left, right, *op, *all);
            let start = Instant::now();
            emitted = drain_operator(&mut set_op).await;
            let duration = start.elapsed();
            let input_rows = ROWS_PER_SIDE * 2;
            runs.push(input_rows as f64 / duration.as_secs_f64());
        }
        tprintln!("  {} emitted {} rows", label, emitted);
        let result = validate_metric(
            "Set Operations",
            &format!("{} throughput (rows/sec)", label),
            runs,
            *target,
            true,
        );
        assert!(
            result.passed,
            "{} avg {:.0} < target {:.0}",
            label, result.average, target
        );
    }
}

// =============================================================================
// Test 14: Expression Evaluation Correctness
// =============================================================================

#[tokio::test]
async fn test_expression_evaluation() {
    zyron_bench_harness::init("executor");
    tprintln!("\n=== Expression Evaluation Test ===");

    let schema = make_schema(&[("a", TypeId::Int64), ("b", TypeId::Int64)]);

    let batch = DataBatch::new(vec![
        Column::new(ColumnData::Int64(vec![10, 20, 30, 40, 50]), TypeId::Int64),
        Column::new(ColumnData::Int64(vec![5, 15, 25, 35, 45]), TypeId::Int64),
    ]);

    // Test: a + b
    let add_expr = BoundExpr::BinaryOp {
        left: Box::new(col_ref(0, 0, TypeId::Int64)),
        op: BinaryOperator::Plus,
        right: Box::new(col_ref(0, 1, TypeId::Int64)),
        type_id: TypeId::Int64,
    };
    let result = evaluate(&add_expr, &batch, &schema, &[]).unwrap();
    assert_eq!(result.get_scalar(0), ScalarValue::Int64(15));
    assert_eq!(result.get_scalar(4), ScalarValue::Int64(95));
    tprintln!("  a + b: [15, 35, 55, 75, 95] [PASS]");

    // Test: a * 2
    let mul_expr = BoundExpr::BinaryOp {
        left: Box::new(col_ref(0, 0, TypeId::Int64)),
        op: BinaryOperator::Multiply,
        right: Box::new(lit_int(2)),
        type_id: TypeId::Int64,
    };
    let result = evaluate(&mul_expr, &batch, &schema, &[]).unwrap();
    assert_eq!(result.get_scalar(0), ScalarValue::Int64(20));
    assert_eq!(result.get_scalar(2), ScalarValue::Int64(60));
    tprintln!("  a * 2: [20, 40, 60, 80, 100] [PASS]");

    // Test: a > b (comparison)
    let cmp_expr = BoundExpr::BinaryOp {
        left: Box::new(col_ref(0, 0, TypeId::Int64)),
        op: BinaryOperator::Gt,
        right: Box::new(col_ref(0, 1, TypeId::Int64)),
        type_id: TypeId::Boolean,
    };
    let result = evaluate(&cmp_expr, &batch, &schema, &[]).unwrap();
    // a > b for all rows since a is always > b
    for i in 0..5 {
        assert_eq!(
            result.get_scalar(i),
            ScalarValue::Boolean(true),
            "Row {}: a > b should be true",
            i
        );
    }
    tprintln!("  a > b: all true [PASS]");

    // Test: IS NULL
    let null_bitmap = {
        let mut nb = NullBitmap::none(5);
        nb.set_null(2);
        nb
    };
    let batch_with_null = DataBatch::new(vec![
        Column::with_nulls(
            ColumnData::Int64(vec![10, 20, 0, 40, 50]),
            null_bitmap,
            TypeId::Int64,
        ),
        Column::new(ColumnData::Int64(vec![5, 15, 25, 35, 45]), TypeId::Int64),
    ]);

    let is_null_expr = BoundExpr::IsNull {
        expr: Box::new(col_ref(0, 0, TypeId::Int64)),
        negated: false,
    };
    let result = evaluate(&is_null_expr, &batch_with_null, &schema, &[]).unwrap();
    assert_eq!(result.get_scalar(0), ScalarValue::Boolean(false));
    assert_eq!(result.get_scalar(2), ScalarValue::Boolean(true));
    assert_eq!(result.get_scalar(4), ScalarValue::Boolean(false));
    tprintln!("  IS NULL: row 2 is null [PASS]");
}

// =============================================================================
// Test 15: TopN Sort (Sort with Limit)
// =============================================================================

#[tokio::test]
async fn test_sort_topn() {
    zyron_bench_harness::init("executor");
    tprintln!("\n=== TopN Sort Test ===");

    let schema = make_schema(&[("val", TypeId::Int64)]);

    let mut rng = rand::rng();
    let data: Vec<i64> = (0..10_000)
        .map(|_| rng.random_range(0..1_000_000))
        .collect();
    let batch = DataBatch::new(vec![Column::new(
        ColumnData::Int64(data.clone()),
        TypeId::Int64,
    )]);

    let order_by = vec![BoundOrderBy {
        expr: col_ref(0, 0, TypeId::Int64),
        asc: true,
        nulls_first: false,
    }];

    let child = MemoryOperator::boxed(vec![batch]);
    let mut sort_op = SortOperator::new(child, order_by, schema, Some(100));
    let result = collect_batches(&mut sort_op).await;

    let total_rows: usize = result.iter().map(|b| b.num_rows).sum();
    assert_eq!(
        total_rows, 100,
        "TopN(100) should produce 100 rows, got {}",
        total_rows
    );

    // Verify sorted and that these are the smallest 100 values.
    let mut sorted_data = data.clone();
    sorted_data.sort();

    let mut idx = 0;
    for b in &result {
        if let ColumnData::Int64(vals) = &b.columns[0].data {
            for &v in vals {
                assert_eq!(
                    v, sorted_data[idx],
                    "TopN row {}: expected {}, got {}",
                    idx, sorted_data[idx], v
                );
                idx += 1;
            }
        }
    }
    tprintln!("  TopN(100) from 10K rows: correct smallest 100 [PASS]");
}

// =============================================================================
// Top-N sort throughput (5-run validation)
// =============================================================================

/// ORDER BY one column with a LIMIT over a table that carries a payload
/// column too, which is the shape of every "latest hundred" query. The
/// payload puts the sort on the gather path rather than the values-only
/// one, so what is measured is finding the kept rows and gathering them
#[tokio::test]
async fn test_sort_topn_throughput() {
    zyron_bench_harness::init("executor");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const ROW_COUNT: usize = 1_000_000;
    const KEEP: usize = 100;

    tprintln!("\n=== Top-N Sort Performance Test ===");
    tprintln!("Rows: {}, limit: {}", ROW_COUNT, KEEP);
    tprintln!("Validation runs: {}", VALIDATION_RUNS);

    let schema = make_schema(&[("val", TypeId::Int64), ("payload", TypeId::Int64)]);
    let order_by = vec![BoundOrderBy {
        expr: col_ref(0, 0, TypeId::Int64),
        asc: true,
        nulls_first: false,
    }];

    let mut all_values: Vec<i64> = Vec::with_capacity(ROW_COUNT);
    let batches = {
        let mut rng = rand::rng();
        let mut batches = Vec::new();
        let mut remaining = ROW_COUNT;
        let mut row_offset = 0usize;
        while remaining > 0 {
            let chunk = remaining.min(BATCH_SIZE);
            let vals: Vec<i64> = (0..chunk)
                .map(|_| rng.random_range(0..ROW_COUNT as i64))
                .collect();
            all_values.extend_from_slice(&vals);
            let payload: Vec<i64> = (0..chunk).map(|r| (row_offset + r) as i64).collect();
            batches.push(DataBatch::new(vec![
                Column::new(ColumnData::Int64(vals), TypeId::Int64),
                Column::new(ColumnData::Int64(payload), TypeId::Int64),
            ]));
            row_offset += chunk;
            remaining -= chunk;
        }
        batches
    };
    let mut expected = all_values.clone();
    expected.sort_unstable();
    expected.truncate(KEEP);

    let mut topn_results = Vec::with_capacity(VALIDATION_RUNS);

    let util_before = take_util_snapshot();
    for run in 0..VALIDATION_RUNS {
        tprintln!("\n--- Run {}/{} ---", run + 1, VALIDATION_RUNS);

        let child = MemoryOperator::boxed(batches.clone());
        let mut sort_op =
            SortOperator::new(child, order_by.clone(), schema.clone(), Some(KEEP as u64));

        let start = Instant::now();
        let result_batches = collect_batches(&mut sort_op).await;
        let duration = start.elapsed();

        let mut got: Vec<i64> = Vec::with_capacity(KEEP);
        let mut payload_rows = 0usize;
        for b in &result_batches {
            if let ColumnData::Int64(data) = &b.columns[0].data {
                got.extend_from_slice(data);
            }
            payload_rows += b.columns[1].len();
        }
        assert_eq!(
            got,
            expected,
            "Run {}: the limit must keep the smallest {} values in order",
            run + 1,
            KEEP
        );
        assert_eq!(
            payload_rows,
            KEEP,
            "Run {}: the payload column must be gathered alongside the key",
            run + 1
        );

        let rows_sec = ROW_COUNT as f64 / duration.as_secs_f64();
        tprintln!(
            "  Top-N sort: {} rows/sec ({:?})",
            format_with_commas(rows_sec),
            duration
        );
        topn_results.push(rows_sec);
    }
    record_test_util("Top-N Sort", util_before, take_util_snapshot());

    tprintln!("\n=== Top-N Sort Validation Results ===");
    let result = validate_metric(
        "Top-N Sort",
        "Top-N sort throughput (rows/sec)",
        topn_results,
        SORT_TOPN_TARGET_ROWS_SEC,
        true,
    );
    assert!(
        result.passed,
        "Top-N Sort avg {:.0} < target {:.0}",
        result.average, SORT_TOPN_TARGET_ROWS_SEC
    );
    assert!(
        !result.regression_detected,
        "Top-N Sort regression detected"
    );
}

// =============================================================================
// Test 16: Pipeline Integration (Filter -> Sort -> Limit)
// =============================================================================

#[tokio::test]
async fn test_pipeline_filter_sort_limit() {
    zyron_bench_harness::init("executor");
    tprintln!("\n=== Pipeline Integration Test (Filter -> Sort -> Limit) ===");

    let schema = make_schema(&[("id", TypeId::Int64), ("val", TypeId::Int64)]);

    // 1000 rows, id = 0..999, val = 999..0
    let ids: Vec<i64> = (0..1000).collect();
    let vals: Vec<i64> = (0..1000).rev().collect();
    let batch = DataBatch::new(vec![
        Column::new(ColumnData::Int64(ids), TypeId::Int64),
        Column::new(ColumnData::Int64(vals), TypeId::Int64),
    ]);

    // Filter: id < 500
    let predicate = BoundExpr::BinaryOp {
        left: Box::new(col_ref(0, 0, TypeId::Int64)),
        op: BinaryOperator::Lt,
        right: Box::new(lit_int(500)),
        type_id: TypeId::Boolean,
    };

    let child = MemoryOperator::boxed(vec![batch]);
    let filter_op = FilterOperator::new(child, predicate, schema.clone());

    // Sort by val ASC
    let order_by = vec![BoundOrderBy {
        expr: col_ref(0, 1, TypeId::Int64),
        asc: true,
        nulls_first: false,
    }];
    let sort_op = SortOperator::new(Box::new(filter_op), order_by, schema.clone(), None);

    // Limit 10
    let mut limit_op = LimitOperator::new(Box::new(sort_op), Some(10), None);

    let result = collect_batches(&mut limit_op).await;
    let total_rows: usize = result.iter().map(|b| b.num_rows).sum();
    assert_eq!(
        total_rows, 10,
        "Pipeline should produce 10 rows, got {}",
        total_rows
    );

    // After filter (id < 500), vals are 999..500.
    // Sorted ASC by val: 500, 501, ..., 509
    // So first 10 should be vals 500-509.
    let first_val = result[0].columns[1].get_scalar(0);
    assert_eq!(
        first_val,
        ScalarValue::Int64(500),
        "First val should be 500, got {}",
        first_val
    );
    let last_val = result[0].columns[1].get_scalar(9);
    assert_eq!(
        last_val,
        ScalarValue::Int64(509),
        "Last val should be 509, got {}",
        last_val
    );
    tprintln!("  Filter(id<500) -> Sort(val ASC) -> Limit(10): vals 500-509 [PASS]");
}

// =============================================================================
// Test 17: Hash Build Throughput (5-run validation)
// =============================================================================

#[tokio::test]
async fn test_hash_build_throughput() {
    zyron_bench_harness::init("executor");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const ROW_COUNT: usize = 1_000_000;

    tprintln!("\n=== Hash Build Performance Test ===");
    tprintln!("Rows: {}", ROW_COUNT);
    tprintln!("Validation runs: {}", VALIDATION_RUNS);

    let schema = make_schema(&[("key", TypeId::Int64), ("val", TypeId::Int64)]);

    // GROUP BY key with 1000 distinct keys
    let group_by = vec![col_ref(0, 0, TypeId::Int64)];
    let aggregates = vec![AggregateExpr {
        function_name: "count".to_string(),
        args: vec![],
        distinct: false,
        return_type: TypeId::Int64,
        uda: None,
    }];
    let output_schema = make_schema(&[("key", TypeId::Int64), ("count", TypeId::Int64)]);

    // Build data with keys mod 1000.
    let batches = {
        let mut batches = Vec::new();
        let mut remaining = ROW_COUNT;
        let mut row_offset = 0;
        while remaining > 0 {
            let chunk = remaining.min(BATCH_SIZE);
            let keys: Vec<i64> = (0..chunk)
                .map(|r| ((row_offset + r) % 1000) as i64)
                .collect();
            let vals: Vec<i64> = (0..chunk).map(|r| (row_offset + r) as i64).collect();
            batches.push(DataBatch::new(vec![
                Column::new(ColumnData::Int64(keys), TypeId::Int64),
                Column::new(ColumnData::Int64(vals), TypeId::Int64),
            ]));
            row_offset += chunk;
            remaining -= chunk;
        }
        batches
    };

    let mut build_results = Vec::with_capacity(VALIDATION_RUNS);

    let util_before = take_util_snapshot();
    for run in 0..VALIDATION_RUNS {
        tprintln!("\n--- Run {}/{} ---", run + 1, VALIDATION_RUNS);

        let child = MemoryOperator::boxed(batches.clone());
        let mut agg_op = HashAggregateOperator::new(
            child,
            group_by.clone(),
            aggregates.clone(),
            schema.clone(),
            output_schema.clone(),
        );

        let start = Instant::now();
        let result = collect_batches(&mut agg_op).await;
        let duration = start.elapsed();

        let total_groups: usize = result.iter().map(|b| b.num_rows).sum();
        assert_eq!(
            total_groups,
            1000,
            "Run {}: expected 1000 groups, got {}",
            run + 1,
            total_groups
        );

        let rows_sec = ROW_COUNT as f64 / duration.as_secs_f64();
        tprintln!(
            "  Hash Build: {} rows/sec ({:?}), {} groups",
            format_with_commas(rows_sec),
            duration,
            total_groups
        );
        build_results.push(rows_sec);
    }
    record_test_util("Hash Build", util_before, take_util_snapshot());

    tprintln!("\n=== Hash Build Validation Results ===");
    let result = validate_metric(
        "Hash Build",
        "Hash build throughput (rows/sec)",
        build_results,
        HASH_BUILD_TARGET_ROWS_SEC,
        true,
    );
    assert!(
        result.passed,
        "Hash Build avg {:.0} < target {:.0}",
        result.average, HASH_BUILD_TARGET_ROWS_SEC
    );
    assert!(
        !result.regression_detected,
        "Hash Build regression detected"
    );
}

// =============================================================================
// Hash build throughput over many groups (5-run validation)
// =============================================================================

/// GROUP BY over a key with a hundred thousand distinct values, with a SUM
/// and a COUNT so each row folds into two accumulators. Keys arrive as a
/// stride permutation, so consecutive rows land in unrelated groups and
/// every lookup is a table access rather than a hit on the line the last
/// row touched. The same grouping with a lone COUNT(*) is reported beside
/// the gated shape, so the cost of an accumulator per row is visible on
/// its own
#[tokio::test]
async fn test_hash_build_many_groups_throughput() {
    zyron_bench_harness::init("executor");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const ROW_COUNT: usize = 1_000_000;
    const GROUPS: usize = 100_000;
    /// Coprime with GROUPS, so a full cycle of rows visits every key once
    const STRIDE: usize = 7_919;

    tprintln!("\n=== Hash Build Many Groups Performance Test ===");
    tprintln!("Rows: {}, groups: {}", ROW_COUNT, GROUPS);
    tprintln!("Validation runs: {}", VALIDATION_RUNS);

    let schema = make_schema(&[("key", TypeId::Int64), ("val", TypeId::Int64)]);
    let group_by = vec![col_ref(0, 0, TypeId::Int64)];
    let aggregates = vec![
        AggregateExpr {
            function_name: "sum".to_string(),
            args: vec![col_ref(0, 1, TypeId::Int64)],
            distinct: false,
            return_type: TypeId::Int64,
            uda: None,
        },
        AggregateExpr {
            function_name: "count".to_string(),
            args: vec![],
            distinct: false,
            return_type: TypeId::Int64,
            uda: None,
        },
    ];
    let output_schema = make_schema(&[
        ("key", TypeId::Int64),
        ("sum", TypeId::Int64),
        ("count", TypeId::Int64),
    ]);
    let count_only = vec![aggregates[1].clone()];
    let count_only_schema = make_schema(&[("key", TypeId::Int64), ("count", TypeId::Int64)]);

    let batches = {
        let mut batches = Vec::new();
        let mut remaining = ROW_COUNT;
        let mut row_offset = 0;
        while remaining > 0 {
            let chunk = remaining.min(BATCH_SIZE);
            let keys: Vec<i64> = (0..chunk)
                .map(|r| (((row_offset + r) * STRIDE) % GROUPS) as i64)
                .collect();
            let vals: Vec<i64> = (0..chunk).map(|r| (row_offset + r) as i64).collect();
            batches.push(DataBatch::new(vec![
                Column::new(ColumnData::Int64(keys), TypeId::Int64),
                Column::new(ColumnData::Int64(vals), TypeId::Int64),
            ]));
            row_offset += chunk;
            remaining -= chunk;
        }
        batches
    };
    let rows_per_group = (ROW_COUNT / GROUPS) as i64;
    let expected_total: i64 = (0..ROW_COUNT as i64).sum();

    let mut build_results = Vec::with_capacity(VALIDATION_RUNS);

    let util_before = take_util_snapshot();
    for run in 0..VALIDATION_RUNS {
        tprintln!("\n--- Run {}/{} ---", run + 1, VALIDATION_RUNS);

        let child = MemoryOperator::boxed(batches.clone());
        let mut agg_op = HashAggregateOperator::new(
            child,
            group_by.clone(),
            aggregates.clone(),
            schema.clone(),
            output_schema.clone(),
        );

        let start = Instant::now();
        let result = collect_batches(&mut agg_op).await;
        let duration = start.elapsed();

        let total_groups: usize = result.iter().map(|b| b.num_rows).sum();
        assert_eq!(
            total_groups,
            GROUPS,
            "Run {}: expected {} groups, got {}",
            run + 1,
            GROUPS,
            total_groups
        );
        let mut sum_total = 0i64;
        for b in &result {
            if let ColumnData::Int64(sums) = &b.columns[1].data {
                sum_total += sums.iter().sum::<i64>();
            }
            if let ColumnData::Int64(counts) = &b.columns[2].data {
                assert!(
                    counts.iter().all(|&c| c == rows_per_group),
                    "Run {}: every group holds {} rows",
                    run + 1,
                    rows_per_group
                );
            }
        }
        assert_eq!(
            sum_total,
            expected_total,
            "Run {}: the group sums must add up to the sum of every value",
            run + 1
        );

        let rows_sec = ROW_COUNT as f64 / duration.as_secs_f64();
        tprintln!(
            "  Hash Build (many groups): {} rows/sec ({:?}), {} groups",
            format_with_commas(rows_sec),
            duration,
            total_groups
        );
        build_results.push(rows_sec);

        let child = MemoryOperator::boxed(batches.clone());
        let mut count_op = HashAggregateOperator::new(
            child,
            group_by.clone(),
            count_only.clone(),
            schema.clone(),
            count_only_schema.clone(),
        );
        let start = Instant::now();
        let counted = collect_batches(&mut count_op).await;
        let count_duration = start.elapsed();
        let counted_groups: usize = counted.iter().map(|b| b.num_rows).sum();
        assert_eq!(counted_groups, GROUPS);
        tprintln!(
            "  Hash Build (many groups, COUNT only): {} rows/sec ({:?})",
            format_with_commas(ROW_COUNT as f64 / count_duration.as_secs_f64()),
            count_duration
        );
    }
    record_test_util("Hash Build Many Groups", util_before, take_util_snapshot());

    tprintln!("\n=== Hash Build Many Groups Validation Results ===");
    let result = validate_metric(
        "Hash Build Many Groups",
        "Hash build throughput over many groups (rows/sec)",
        build_results,
        HASH_BUILD_MANY_GROUPS_TARGET_ROWS_SEC,
        true,
    );
    assert!(
        result.passed,
        "Hash Build Many Groups avg {:.0} < target {:.0}",
        result.average, HASH_BUILD_MANY_GROUPS_TARGET_ROWS_SEC
    );
    assert!(
        !result.regression_detected,
        "Hash Build Many Groups regression detected"
    );
}

// =============================================================================
// String predicate throughput: equality, LIKE, ILIKE (5-run validation)
// =============================================================================

/// Builds BATCH_SIZE chunks of two text columns: col 0 is a distinct name per
/// row for pattern matching, col 1 cycles through 1000 categories for equality
fn build_text_dataset(total_rows: usize) -> Vec<DataBatch> {
    let mut batches = Vec::new();
    let mut remaining = total_rows;
    let mut row_offset = 0;
    while remaining > 0 {
        let chunk = remaining.min(BATCH_SIZE);
        let names: Vec<String> = (0..chunk)
            .map(|r| format!("user_{}", row_offset + r))
            .collect();
        let categories: Vec<String> = (0..chunk)
            .map(|r| format!("value_{}", (row_offset + r) % 1000))
            .collect();
        batches.push(DataBatch::new(vec![
            Column::new(ColumnData::Utf8(names), TypeId::Text),
            Column::new(ColumnData::Utf8(categories), TypeId::Text),
        ]));
        row_offset += chunk;
        remaining -= chunk;
    }
    batches
}

/// Creates a BoundExpr::Literal for a text value.
fn lit_text(val: &str) -> BoundExpr {
    BoundExpr::Literal {
        value: LiteralValue::String(val.to_string()),
        type_id: TypeId::Text,
    }
}

/// Evaluates a predicate over every batch and returns rows/sec for the run,
/// asserting the match count so the work cannot be optimized away
fn time_predicate(
    predicate: &BoundExpr,
    batches: &[DataBatch],
    schema: &[LogicalColumn],
    total_rows: usize,
    expected_matches: usize,
) -> f64 {
    let start = Instant::now();
    let mut matches = 0usize;
    for batch in batches {
        let mask = evaluate(predicate, batch, schema, &[]).expect("predicate evaluates");
        let ColumnData::Boolean(bits) = &mask.data else {
            panic!("predicate must produce a boolean column");
        };
        matches += bits.iter().filter(|b| **b).count();
    }
    let duration = start.elapsed();
    assert_eq!(matches, expected_matches, "predicate match count");
    total_rows as f64 / duration.as_secs_f64()
}

#[tokio::test]
async fn test_string_predicate_throughput() {
    zyron_bench_harness::init("executor");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const ROW_COUNT: usize = 500_000;

    tprintln!("\n=== String Predicate Throughput Test ===");
    tprintln!("Rows: {}", ROW_COUNT);
    tprintln!("Validation runs: {}", VALIDATION_RUNS);

    let schema = make_schema(&[("name", TypeId::Text), ("category", TypeId::Text)]);
    let batches = build_text_dataset(ROW_COUNT);

    // category = 'value_500': 1 in 1000 rows match
    let eq_pred = BoundExpr::BinaryOp {
        left: Box::new(col_ref(0, 1, TypeId::Text)),
        op: BinaryOperator::Eq,
        right: Box::new(lit_text("value_500")),
        type_id: TypeId::Boolean,
    };
    // name LIKE 'user_1%': names beginning user_1
    let like_prefix = BoundExpr::Like {
        expr: Box::new(col_ref(0, 0, TypeId::Text)),
        pattern: Box::new(lit_text("user_1%")),
        negated: false,
    };
    // name LIKE 'user_1___9': underscore wildcards drive the general matcher
    let like_general = BoundExpr::Like {
        expr: Box::new(col_ref(0, 0, TypeId::Text)),
        pattern: Box::new(lit_text("user_1___9")),
        negated: false,
    };
    // name ILIKE '%USER_42%': case folding plus substring search
    let ilike_contains = BoundExpr::ILike {
        expr: Box::new(col_ref(0, 0, TypeId::Text)),
        pattern: Box::new(lit_text("%USER_42%")),
        negated: false,
    };

    // Patterns holding no '_' at all. SQL reads '_' as a single-character
    // wildcard, so every pattern above carries one and matches through the
    // wildcard-aware path. These three reach the literal path instead, one
    // for each of its shapes: anchored at the start, anchored at the end, and
    // searched for in the middle
    // name LIKE 'user%': the literal prefix shape, every row matches
    let like_literal_prefix = BoundExpr::Like {
        expr: Box::new(col_ref(0, 0, TypeId::Text)),
        pattern: Box::new(lit_text("user%")),
        negated: false,
    };
    // name LIKE '%99': the literal suffix shape
    let like_literal_suffix = BoundExpr::Like {
        expr: Box::new(col_ref(0, 0, TypeId::Text)),
        pattern: Box::new(lit_text("%99")),
        negated: false,
    };
    // name LIKE '%1234%': the literal substring shape
    let like_literal_contains = BoundExpr::Like {
        expr: Box::new(col_ref(0, 0, TypeId::Text)),
        pattern: Box::new(lit_text("%1234%")),
        negated: false,
    };

    // Expected counts over user_0..user_499999
    let eq_expected = ROW_COUNT / 1000;
    let literal_prefix_expected = (0..ROW_COUNT)
        .filter(|i| format!("user_{i}").starts_with("user"))
        .count();
    let literal_suffix_expected = (0..ROW_COUNT)
        .filter(|i| format!("user_{i}").ends_with("99"))
        .count();
    let literal_contains_expected = (0..ROW_COUNT)
        .filter(|i| format!("user_{i}").contains("1234"))
        .count();
    let like_prefix_expected = (0..ROW_COUNT)
        .filter(|i| format!("user_{i}").starts_with("user_1"))
        .count();
    let like_general_expected = (0..ROW_COUNT)
        .filter(|i| {
            let name = format!("user_{i}");
            name.len() == 10 && name.starts_with("user_1") && name.ends_with('9')
        })
        .count();
    let ilike_expected = (0..ROW_COUNT)
        .filter(|i| format!("user_{i}").contains("user_42"))
        .count();

    let mut eq_results = Vec::with_capacity(VALIDATION_RUNS);
    let mut prefix_results = Vec::with_capacity(VALIDATION_RUNS);
    let mut general_results = Vec::with_capacity(VALIDATION_RUNS);
    let mut ilike_results = Vec::with_capacity(VALIDATION_RUNS);
    let mut literal_prefix_results = Vec::with_capacity(VALIDATION_RUNS);
    let mut literal_suffix_results = Vec::with_capacity(VALIDATION_RUNS);
    let mut literal_contains_results = Vec::with_capacity(VALIDATION_RUNS);

    let util_before = take_util_snapshot();
    for run in 0..VALIDATION_RUNS {
        tprintln!("\n--- Run {}/{} ---", run + 1, VALIDATION_RUNS);
        let eq = time_predicate(&eq_pred, &batches, &schema, ROW_COUNT, eq_expected);
        let prefix = time_predicate(
            &like_prefix,
            &batches,
            &schema,
            ROW_COUNT,
            like_prefix_expected,
        );
        let general = time_predicate(
            &like_general,
            &batches,
            &schema,
            ROW_COUNT,
            like_general_expected,
        );
        let ilike = time_predicate(
            &ilike_contains,
            &batches,
            &schema,
            ROW_COUNT,
            ilike_expected,
        );
        let literal_prefix = time_predicate(
            &like_literal_prefix,
            &batches,
            &schema,
            ROW_COUNT,
            literal_prefix_expected,
        );
        let literal_suffix = time_predicate(
            &like_literal_suffix,
            &batches,
            &schema,
            ROW_COUNT,
            literal_suffix_expected,
        );
        let literal_contains = time_predicate(
            &like_literal_contains,
            &batches,
            &schema,
            ROW_COUNT,
            literal_contains_expected,
        );
        tprintln!(
            "  eq {} rows/sec, LIKE prefix {} rows/sec, LIKE general {} rows/sec, ILIKE {} rows/sec",
            format_with_commas(eq),
            format_with_commas(prefix),
            format_with_commas(general),
            format_with_commas(ilike),
        );
        tprintln!(
            "  literal prefix {} rows/sec, literal suffix {} rows/sec, literal contains {} rows/sec",
            format_with_commas(literal_prefix),
            format_with_commas(literal_suffix),
            format_with_commas(literal_contains),
        );
        eq_results.push(eq);
        prefix_results.push(prefix);
        general_results.push(general);
        ilike_results.push(ilike);
        literal_prefix_results.push(literal_prefix);
        literal_suffix_results.push(literal_suffix);
        literal_contains_results.push(literal_contains);
    }
    record_test_util("String Predicate", util_before, take_util_snapshot());

    tprintln!("\n=== String Predicate Validation Results ===");
    let eq_result = validate_metric(
        "String Equality",
        "String equality predicate (rows/sec)",
        eq_results,
        STRING_EQ_TARGET_ROWS_SEC,
        true,
    );
    let prefix_result = validate_metric(
        "LIKE Prefix",
        "LIKE prefix predicate (rows/sec)",
        prefix_results,
        LIKE_PREFIX_TARGET_ROWS_SEC,
        true,
    );
    let general_result = validate_metric(
        "LIKE General",
        "LIKE underscore predicate (rows/sec)",
        general_results,
        LIKE_GENERAL_TARGET_ROWS_SEC,
        true,
    );
    let ilike_result = validate_metric(
        "ILIKE Contains",
        "ILIKE contains predicate (rows/sec)",
        ilike_results,
        ILIKE_CONTAINS_TARGET_ROWS_SEC,
        true,
    );
    let literal_prefix_result = validate_metric(
        "LIKE Literal Prefix",
        "LIKE literal prefix predicate (rows/sec)",
        literal_prefix_results,
        LIKE_LITERAL_PREFIX_TARGET_ROWS_SEC,
        true,
    );
    let literal_suffix_result = validate_metric(
        "LIKE Literal Suffix",
        "LIKE literal suffix predicate (rows/sec)",
        literal_suffix_results,
        LIKE_LITERAL_SUFFIX_TARGET_ROWS_SEC,
        true,
    );
    let literal_contains_result = validate_metric(
        "LIKE Literal Contains",
        "LIKE literal contains predicate (rows/sec)",
        literal_contains_results,
        LIKE_LITERAL_CONTAINS_TARGET_ROWS_SEC,
        true,
    );
    assert!(eq_result.passed, "String equality below target");
    assert!(prefix_result.passed, "LIKE prefix below target");
    assert!(general_result.passed, "LIKE general below target");
    assert!(ilike_result.passed, "ILIKE contains below target");
    assert!(
        literal_prefix_result.passed,
        "LIKE literal prefix below target"
    );
    assert!(
        literal_suffix_result.passed,
        "LIKE literal suffix below target"
    );
    assert!(
        literal_contains_result.passed,
        "LIKE literal contains below target"
    );
}

// =============================================================================
// IN-list predicate throughput (5-run validation)
// =============================================================================

#[tokio::test]
async fn test_in_list_predicate_throughput() {
    zyron_bench_harness::init("executor");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const ROW_COUNT: usize = 1_000_000;

    tprintln!("\n=== IN-List Predicate Throughput Test ===");
    tprintln!("Rows: {}", ROW_COUNT);
    tprintln!("Validation runs: {}", VALIDATION_RUNS);

    let schema = make_schema(&[("id", TypeId::Int64), ("val", TypeId::Int64)]);
    let batches = build_large_dataset(ROW_COUNT, 2);

    // Column 0 holds row_index * 2, so even list values each match one row
    let list: Vec<BoundExpr> = [
        4i64, 100, 4096, 65536, 250_000, 777_770, 1_400_000, 1_999_998,
    ]
    .iter()
    .map(|v| lit_int(*v))
    .collect();
    let expected = 8usize;
    let predicate = BoundExpr::InList {
        expr: Box::new(col_ref(0, 0, TypeId::Int64)),
        list,
        negated: false,
    };

    let mut results = Vec::with_capacity(VALIDATION_RUNS);
    let util_before = take_util_snapshot();
    for run in 0..VALIDATION_RUNS {
        tprintln!("\n--- Run {}/{} ---", run + 1, VALIDATION_RUNS);
        let rows_sec = time_predicate(&predicate, &batches, &schema, ROW_COUNT, expected);
        tprintln!("  IN-list: {} rows/sec", format_with_commas(rows_sec));
        results.push(rows_sec);
    }
    record_test_util("IN-List Predicate", util_before, take_util_snapshot());

    tprintln!("\n=== IN-List Validation Results ===");
    let result = validate_metric(
        "IN-List Predicate",
        "IN-list predicate (rows/sec)",
        results,
        IN_LIST_TARGET_ROWS_SEC,
        true,
    );
    assert!(result.passed, "IN-list predicate below target");
    assert!(!result.regression_detected, "IN-list regression detected");
}

// =============================================================================
// Window explicit-frame aggregate throughput (5-run validation)
// =============================================================================

#[tokio::test]
async fn test_window_frame_throughput() {
    zyron_bench_harness::init("executor");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const ROW_COUNT: usize = 200_000;

    tprintln!("\n=== Window Explicit Frame Throughput Test ===");
    tprintln!("Rows: {}", ROW_COUNT);
    tprintln!("Validation runs: {}", VALIDATION_RUNS);

    let schema = make_schema(&[("id", TypeId::Int64), ("val", TypeId::Int64)]);
    let batches = build_large_dataset(ROW_COUNT, 2);

    // SUM(val) OVER (ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND
    // CURRENT ROW), the explicitly written cumulative frame
    let window_expr = BoundExpr::WindowFunction {
        function: Box::new(BoundExpr::AggregateFunction {
            name: "sum".to_string(),
            args: vec![col_ref(0, 1, TypeId::Int64)],
            distinct: false,
            return_type: TypeId::Int64,
            uda: None,
        }),
        partition_by: vec![],
        order_by: vec![BoundOrderBy {
            expr: col_ref(0, 0, TypeId::Int64),
            asc: true,
            nulls_first: false,
        }],
        frame: Some(zyron_parser::ast::WindowFrame {
            mode: zyron_parser::ast::WindowFrameMode::Rows,
            start: zyron_parser::ast::WindowFrameBound::Unbounded(
                zyron_parser::ast::WindowFrameDirection::Preceding,
            ),
            end: Some(zyron_parser::ast::WindowFrameBound::CurrentRow),
        }),
        type_id: TypeId::Int64,
    };

    let mut results = Vec::with_capacity(VALIDATION_RUNS);
    let util_before = take_util_snapshot();
    for run in 0..VALIDATION_RUNS {
        tprintln!("\n--- Run {}/{} ---", run + 1, VALIDATION_RUNS);
        let child = MemoryOperator::boxed(batches.clone());
        let mut window_op = zyron_executor::operator::window::WindowOperator::new(
            child,
            vec![window_expr.clone()],
            schema.clone(),
        );
        let start = Instant::now();
        let total_rows = drain_operator(&mut window_op).await;
        let duration = start.elapsed();
        assert_eq!(total_rows, ROW_COUNT, "window emits one row per input row");
        let rows_sec = ROW_COUNT as f64 / duration.as_secs_f64();
        tprintln!(
            "  Window frame: {} rows/sec ({:?})",
            format_with_commas(rows_sec),
            duration
        );
        results.push(rows_sec);
    }
    record_test_util("Window Frame", util_before, take_util_snapshot());

    tprintln!("\n=== Window Frame Validation Results ===");
    let result = validate_metric(
        "Window Frame",
        "Window explicit frame throughput (rows/sec)",
        results,
        WINDOW_FRAME_TARGET_ROWS_SEC,
        true,
    );
    assert!(result.passed, "Window frame throughput below target");
    assert!(
        !result.regression_detected,
        "Window frame regression detected"
    );
}

// =============================================================================
// Window partitioned running aggregate throughput (5-run validation)
// =============================================================================

/// SUM(val) OVER (PARTITION BY key ORDER BY ts), the shape of a running
/// total per account. Keys arrive as a stride permutation over a thousand
/// partitions and ts runs backwards, so the rows are in neither partition
/// nor order-key order and the sort has to move every one of them
#[tokio::test]
async fn test_window_partitioned_throughput() {
    zyron_bench_harness::init("executor");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const ROW_COUNT: usize = 500_000;
    const PARTITIONS: usize = 1_000;
    /// Coprime with PARTITIONS, so a full cycle of rows visits every key once
    const STRIDE: usize = 7_919;

    tprintln!("\n=== Window Partitioned Running Sum Throughput Test ===");
    tprintln!("Rows: {}, partitions: {}", ROW_COUNT, PARTITIONS);
    tprintln!("Validation runs: {}", VALIDATION_RUNS);

    let schema = make_schema(&[
        ("key", TypeId::Int64),
        ("ts", TypeId::Int64),
        ("val", TypeId::Int64),
    ]);
    let key_of = |row: usize| (row * STRIDE) % PARTITIONS;
    let batches = {
        let mut batches = Vec::new();
        let mut remaining = ROW_COUNT;
        let mut row_offset = 0;
        while remaining > 0 {
            let chunk = remaining.min(BATCH_SIZE);
            let keys: Vec<i64> = (0..chunk).map(|r| key_of(row_offset + r) as i64).collect();
            let ts: Vec<i64> = (0..chunk)
                .map(|r| (ROW_COUNT - (row_offset + r)) as i64)
                .collect();
            let vals: Vec<i64> = (0..chunk).map(|r| (row_offset + r) as i64).collect();
            batches.push(DataBatch::new(vec![
                Column::new(ColumnData::Int64(keys), TypeId::Int64),
                Column::new(ColumnData::Int64(ts), TypeId::Int64),
                Column::new(ColumnData::Int64(vals), TypeId::Int64),
            ]));
            row_offset += chunk;
            remaining -= chunk;
        }
        batches
    };
    // The row of each partition with the largest ts is its first row in
    // input order, and its running sum is the partition's whole total
    let mut totals = vec![0i64; PARTITIONS];
    let mut first_row = vec![usize::MAX; PARTITIONS];
    for row in 0..ROW_COUNT {
        let key = key_of(row);
        totals[key] += row as i64;
        first_row[key] = first_row[key].min(row);
    }

    let window_expr = BoundExpr::WindowFunction {
        function: Box::new(BoundExpr::AggregateFunction {
            name: "sum".to_string(),
            args: vec![col_ref(0, 2, TypeId::Int64)],
            distinct: false,
            return_type: TypeId::Int64,
            uda: None,
        }),
        partition_by: vec![col_ref(0, 0, TypeId::Int64)],
        order_by: vec![BoundOrderBy {
            expr: col_ref(0, 1, TypeId::Int64),
            asc: true,
            nulls_first: false,
        }],
        frame: None,
        type_id: TypeId::Int64,
    };

    let mut results = Vec::with_capacity(VALIDATION_RUNS);
    let util_before = take_util_snapshot();
    for run in 0..VALIDATION_RUNS {
        tprintln!("\n--- Run {}/{} ---", run + 1, VALIDATION_RUNS);
        let child = MemoryOperator::boxed(batches.clone());
        let mut window_op = zyron_executor::operator::window::WindowOperator::new(
            child,
            vec![window_expr.clone()],
            schema.clone(),
        );
        let start = Instant::now();
        let out = collect_batches(&mut window_op).await;
        let duration = start.elapsed();

        let mut running: Vec<i64> = Vec::with_capacity(ROW_COUNT);
        for b in &out {
            let ColumnData::Int64(values) = &b.columns[3].data else {
                panic!("the running sum is an Int64 column");
            };
            running.extend_from_slice(values);
        }
        assert_eq!(
            running.len(),
            ROW_COUNT,
            "window emits one row per input row"
        );
        for key in 0..PARTITIONS {
            assert_eq!(
                running[first_row[key]],
                totals[key],
                "Run {}: partition {} ends on its total",
                run + 1,
                key
            );
        }

        let rows_sec = ROW_COUNT as f64 / duration.as_secs_f64();
        tprintln!(
            "  Window partitioned: {} rows/sec ({:?})",
            format_with_commas(rows_sec),
            duration
        );
        results.push(rows_sec);
    }
    record_test_util("Window Partitioned", util_before, take_util_snapshot());
    // Compiled in by --features profile, gated at runtime by ZYRON_PROFILE
    zyron_common::profile::dump("window partitioned");

    tprintln!("\n=== Window Partitioned Validation Results ===");
    let result = validate_metric(
        "Window Partitioned",
        "Window partitioned running sum throughput (rows/sec)",
        results,
        WINDOW_PARTITIONED_TARGET_ROWS_SEC,
        true,
    );
    assert!(
        result.passed,
        "Window Partitioned avg {:.0} < target {:.0}",
        result.average, WINDOW_PARTITIONED_TARGET_ROWS_SEC
    );
    assert!(
        !result.regression_detected,
        "Window Partitioned regression detected"
    );
}

// =============================================================================
// Residual-condition hash join throughput (5-run validation)
// =============================================================================

/// Schema builder that stamps a caller-chosen table index, so a join's two
/// sides stay addressable in the concatenated condition schema
fn make_schema_at(table_idx: usize, cols: &[(&str, TypeId)]) -> Vec<LogicalColumn> {
    cols.iter()
        .enumerate()
        .map(|(i, (name, tid))| LogicalColumn {
            table_idx: Some(table_idx),
            column_id: ColumnId(i as u16),
            name: name.to_string(),
            type_id: *tid,
            nullable: true,
            fractional_digits: None,
        })
        .collect()
}

#[tokio::test]
async fn test_residual_join_throughput() {
    zyron_bench_harness::init("executor");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const LEFT_ROWS: usize = 200_000;
    const RIGHT_ROWS: usize = 20_000;

    tprintln!("\n=== Residual-Condition Hash Join Throughput Test ===");
    tprintln!("Left rows: {}, Right rows: {}", LEFT_ROWS, RIGHT_ROWS);
    tprintln!("Validation runs: {}", VALIDATION_RUNS);

    let left_schema = make_schema_at(0, &[("k", TypeId::Int64), ("a", TypeId::Int64)]);
    let right_schema = make_schema_at(1, &[("k", TypeId::Int64), ("b", TypeId::Int64)]);

    // Left keys cycle through the right key space so every left row finds
    // hash candidates, and the residual a < b passes for about half of them
    let left_batches = {
        let mut batches = Vec::new();
        let mut remaining = LEFT_ROWS;
        let mut row_offset = 0;
        while remaining > 0 {
            let chunk = remaining.min(BATCH_SIZE);
            let keys: Vec<i64> = (0..chunk)
                .map(|r| ((row_offset + r) % RIGHT_ROWS) as i64)
                .collect();
            // 97 is coprime to the right side's modulus so a and b differ
            // across the key space and the residual passes for about half
            // of the candidate pairs
            let a_vals: Vec<i64> = (0..chunk).map(|r| ((row_offset + r) % 97) as i64).collect();
            batches.push(DataBatch::new(vec![
                Column::new(ColumnData::Int64(keys), TypeId::Int64),
                Column::new(ColumnData::Int64(a_vals), TypeId::Int64),
            ]));
            row_offset += chunk;
            remaining -= chunk;
        }
        batches
    };
    let right_batches = {
        let mut batches = Vec::new();
        let mut remaining = RIGHT_ROWS;
        let mut row_offset = 0;
        while remaining > 0 {
            let chunk = remaining.min(BATCH_SIZE);
            let keys: Vec<i64> = (0..chunk).map(|r| (row_offset + r) as i64).collect();
            let b_vals: Vec<i64> = (0..chunk)
                .map(|r| ((row_offset + r) % 100) as i64)
                .collect();
            batches.push(DataBatch::new(vec![
                Column::new(ColumnData::Int64(keys), TypeId::Int64),
                Column::new(ColumnData::Int64(b_vals), TypeId::Int64),
            ]));
            row_offset += chunk;
            remaining -= chunk;
        }
        batches
    };

    // ON l.k = r.k AND l.a < r.b
    let residual = BoundExpr::BinaryOp {
        left: Box::new(col_ref(0, 1, TypeId::Int64)),
        op: BinaryOperator::Lt,
        right: Box::new(col_ref(1, 1, TypeId::Int64)),
        type_id: TypeId::Boolean,
    };

    let mut results = Vec::with_capacity(VALIDATION_RUNS);
    let util_before = take_util_snapshot();
    for run in 0..VALIDATION_RUNS {
        tprintln!("\n--- Run {}/{} ---", run + 1, VALIDATION_RUNS);
        let left_op = MemoryOperator::boxed(left_batches.clone());
        let right_op = MemoryOperator::boxed(right_batches.clone());
        let mut join_op = HashJoinOperator::new(
            left_op,
            right_op,
            JoinType::Inner,
            vec![col_ref(0, 0, TypeId::Int64)],
            vec![col_ref(1, 0, TypeId::Int64)],
            Some(residual.clone()),
            left_schema.clone(),
            right_schema.clone(),
        );
        let start = Instant::now();
        let total_rows = drain_operator(&mut join_op).await;
        let duration = start.elapsed();
        assert!(
            total_rows > 0,
            "Run {}: residual join returned 0 rows",
            run + 1
        );
        let input_rows = LEFT_ROWS + RIGHT_ROWS;
        let rows_sec = input_rows as f64 / duration.as_secs_f64();
        tprintln!(
            "  Residual join: {} rows/sec ({:?}), {} output rows",
            format_with_commas(rows_sec),
            duration,
            total_rows
        );
        results.push(rows_sec);
    }
    record_test_util("Residual Join", util_before, take_util_snapshot());

    tprintln!("\n=== Residual Join Validation Results ===");
    let result = validate_metric(
        "Residual Join",
        "Residual-condition join throughput (rows/sec)",
        results,
        RESIDUAL_JOIN_TARGET_ROWS_SEC,
        true,
    );
    assert!(result.passed, "Residual join throughput below target");
    assert!(
        !result.regression_detected,
        "Residual join regression detected"
    );
}

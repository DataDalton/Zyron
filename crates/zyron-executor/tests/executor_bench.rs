#![allow(non_snake_case, unused_assignments)]

//! Query Executor Benchmark Suite
//!
//! Integration tests for ZyronDB executor components:
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
//!
//! Validation Requirements:
//! - Each benchmark runs 5 iterations
//! - Results averaged across all 5 runs
//! - Pass/fail determined by average performance
//! - Individual runs logged for variance analysis
//! - Test FAILS if any single run is >2x worse than target

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
const AGGREGATE_TARGET_ROWS_SEC: f64 = 150_000_000.0;
const SORT_TARGET_ROWS_SEC: f64 = 100_000_000.0;
const LIMIT_TARGET_ROWS_SEC: f64 = 200_000_000.0;

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
        },
        AggregateExpr {
            function_name: "sum".to_string(),
            args: vec![col_ref(0, 1, TypeId::Int64)],
            distinct: false,
            return_type: TypeId::Float64,
        },
        AggregateExpr {
            function_name: "min".to_string(),
            args: vec![col_ref(0, 1, TypeId::Int64)],
            distinct: false,
            return_type: TypeId::Int64,
        },
        AggregateExpr {
            function_name: "max".to_string(),
            args: vec![col_ref(0, 1, TypeId::Int64)],
            distinct: false,
            return_type: TypeId::Int64,
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

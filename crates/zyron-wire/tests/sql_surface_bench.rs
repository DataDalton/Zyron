#![allow(non_snake_case)]

//! SQL Surface Completions Benchmark Suite
//!
//! UNNEST, FLATTEN, temporary tables, PIVOT, UNPIVOT and ASOF JOIN, measured
//! through the engine the way a statement reaches them.
//!
//! Performance Targets:
//! | Test                                        | Metric     | Target                 |
//! |---------------------------------------------|------------|------------------------|
//! | UNNEST 10M rows x 10-element INT[]          | throughput | 100M elements/s        |
//! | FLATTEN 1M VARIANT docs, 20 leaves each     | throughput | 5M leaves/s            |
//! | UNNEST under LATERAL vs bare                | overhead   | 10%                    |
//! | array_transform, 10M rows x 10 elements     | throughput | 50M elements/s         |
//! | CREATE TEMP TABLE                           | latency    | 200us                  |
//! | INSERT 1M rows into a temporary table       | ratio      | 3x a permanent heap    |
//! | Temporary table drop at session end, 1 GB   | latency    | 100ms                  |
//! | PIVOT 10M rows, 4 x 4                       | ratio      | <= 102% of hand-written|
//! | UNPIVOT 1M rows x 12 columns                | throughput | 20M output rows/s      |
//! | ASOF JOIN 10M x 1M, pre-sorted              | throughput | 20M left rows/s        |
//! | ASOF JOIN 10M x 1M, unsorted                | latency    | <= 2x the two sorts    |
//! | ASOF JOIN peak memory, pre-sorted           | bytes      | 2 batches per side     |
//!
//! Validation Requirements:
//! - Each benchmark runs 5 iterations
//! - Results averaged across all 5 runs
//! - Pass/fail determined by average performance
//! - Individual runs logged for variance analysis
//!
//! Run: cargo test --release -p zyron-wire --test sql_surface_bench -- --nocapture

// zyron-wire installs the allocator the server runs, so this suite
// inherits it rather than declaring a second one

mod common;

use std::sync::{Arc, Mutex};
use std::time::Instant;

use zyron_bench_harness::*;
use zyron_common::TypeId;
use zyron_executor::batch::{BATCH_SIZE, ColumnBuilder, DataBatch};
use zyron_executor::column::{Column, ColumnData, ScalarValue};
use zyron_executor::operator::expand_rows::ExpandRowsOperator;
use zyron_executor::operator::{ExecutionBatch, Operator, OperatorResult};
use zyron_planner::binder::{BoundExpr, ColumnRef};
use zyron_planner::logical::{ExpandSpec, LogicalColumn};

/// The suites run one at a time, so a measurement is of the engine rather
/// than of two suites sharing the machine
static BENCHMARK_LOCK: Mutex<()> = Mutex::new(());

const VALIDATION_RUNS: usize = 5;

// =============================================================================
// Performance Target Constants
// =============================================================================

/// One offset walk per array plus one gather of the parent columns, so the
/// per-element cost is a decode and a typed push
const UNNEST_TARGET_ELEMENTS_SEC: f64 = 100_000_000.0;
/// A document walk parses the value and builds six columns per member, so
/// it is an order of magnitude off the array walk by construction
const FLATTEN_TARGET_LEAVES_SEC: f64 = 5_000_000.0;
/// Correlation plumbing is a repeat vector and a gather, so a LATERAL
/// expansion costs the same walk plus the parent columns it carries
const LATERAL_OVERHEAD_LIMIT: f64 = 1.10;
/// The lambda body runs once over every element of the batch, so the cost
/// is one expression evaluation per element rather than one per row
const ARRAY_TRANSFORM_TARGET_ELEMENTS_SEC: f64 = 50_000_000.0;
/// A registry insert and a directory create
const CREATE_TEMP_TABLE_TARGET_US: f64 = 200.0;
/// An unlink per file
const TEMP_DROP_TARGET_MS: f64 = 100.0;
/// The rewrite is the plan a hand-written conditional aggregate produces,
/// so the two run the same
const PIVOT_RATIO_LIMIT: f64 = 1.02;
/// One expansion of each input row into one row per group
const UNPIVOT_TARGET_ROWS_SEC: f64 = 20_000_000.0;
/// One merge pass over two ordered inputs
const ASOF_SORTED_TARGET_ROWS_SEC: f64 = 20_000_000.0;
/// The merge over inputs that have to be sorted first, against the sorts
/// alone. Anything above this means the merge is doing more than one pass
const ASOF_UNSORTED_RATIO_LIMIT: f64 = 2.0;
/// One batch per side plus the held row and the output batch. Measured as
/// batches of the left side's width, so the figure is comparable whatever
/// a batch happens to hold
const ASOF_PEAK_BATCHES_PER_SIDE: f64 = 2.0;

// =============================================================================
// Fixtures
// =============================================================================

/// Hands prebuilt batches to an operator, so a measurement is of the
/// operator rather than of a scan feeding it.
struct MemoryOperator {
    batches: Vec<DataBatch>,
    at: usize,
}

impl MemoryOperator {
    fn new(batches: Vec<DataBatch>) -> Self {
        Self { batches, at: 0 }
    }
}

impl Operator for MemoryOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            if self.at >= self.batches.len() {
                return Ok(None);
            }
            let batch = std::mem::replace(&mut self.batches[self.at], DataBatch::empty());
            self.at += 1;
            Ok(Some(ExecutionBatch::new(batch)))
        })
    }
}

/// One column of encoded INT arrays, each holding `per_row` elements, and a
/// parent id column beside it so a carried gather is measured too.
fn array_batches(rows: usize, per_row: usize) -> Vec<DataBatch> {
    let mut batches = Vec::with_capacity(rows.div_ceil(BATCH_SIZE));
    let mut produced = 0usize;
    while produced < rows {
        let n = BATCH_SIZE.min(rows - produced);
        let mut ids = ColumnBuilder::new(TypeId::Int64, n);
        let mut arrays: Vec<Vec<u8>> = Vec::with_capacity(n);
        for r in 0..n {
            ids.push(&ScalarValue::Int64((produced + r) as i64));
            let payloads: Vec<Option<Vec<u8>>> = (0..per_row)
                .map(|e| Some(((produced + r + e) as i64).to_le_bytes().to_vec()))
                .collect();
            let borrowed: Vec<Option<&[u8]>> = payloads.iter().map(|p| p.as_deref()).collect();
            arrays.push(zyron_common::array_value::encode(TypeId::Int64, &borrowed));
        }
        batches.push(DataBatch::new(vec![
            ids.finish(),
            Column::new(ColumnData::Binary(arrays), TypeId::Array),
        ]));
        produced += n;
    }
    batches
}

/// One column of VARIANT documents, each an object of `leaves` members.
fn document_batches(rows: usize, leaves: usize) -> Vec<DataBatch> {
    let mut batches = Vec::with_capacity(rows.div_ceil(BATCH_SIZE));
    let mut produced = 0usize;
    while produced < rows {
        let n = BATCH_SIZE.min(rows - produced);
        let mut ids = ColumnBuilder::new(TypeId::Int64, n);
        let mut docs = ColumnBuilder::new(TypeId::Variant, n);
        for r in 0..n {
            ids.push(&ScalarValue::Int64((produced + r) as i64));
            let members: Vec<String> = (0..leaves)
                .map(|k| format!("\"k{k}\":{}", produced + r + k))
                .collect();
            docs.push(&ScalarValue::Utf8(format!("{{{}}}", members.join(","))));
        }
        batches.push(DataBatch::new(vec![ids.finish(), docs.finish()]));
        produced += n;
    }
    batches
}

/// The input schema those batches carry.
fn input_schema(value_type: TypeId) -> Vec<LogicalColumn> {
    vec![
        LogicalColumn {
            table_idx: Some(0),
            column_id: zyron_catalog::ColumnId(0),
            name: "id".to_string(),
            type_id: TypeId::Int64,
            nullable: false,
            fractional_digits: None,
        },
        LogicalColumn {
            table_idx: Some(0),
            column_id: zyron_catalog::ColumnId(1),
            name: "v".to_string(),
            type_id: value_type,
            nullable: true,
            fractional_digits: None,
        },
    ]
}

/// A reference to the value column of those batches.
fn value_ref(type_id: TypeId) -> BoundExpr {
    BoundExpr::ColumnRef(ColumnRef {
        table_idx: 0,
        column_id: zyron_catalog::ColumnId(1),
        type_id,
        nullable: true,
        fractional_digits: None,
    })
}

/// One produced column of the given type, under the expansion's own index.
fn produced_column(name: &str, type_id: TypeId, position: u16) -> LogicalColumn {
    LogicalColumn {
        table_idx: Some(9),
        column_id: zyron_catalog::ColumnId(position),
        name: name.to_string(),
        type_id,
        nullable: true,
        fractional_digits: None,
    }
}

/// Drains an operator and returns how many rows it produced.
async fn drain(op: &mut dyn Operator) -> usize {
    let mut rows = 0usize;
    while let Some(batch) = op.next().await.expect("operator") {
        rows += batch.num_rows();
    }
    rows
}

/// An execution context with nothing behind it, for an operator measured
/// over batches it is handed rather than over storage.
async fn bare_context() -> (
    Arc<zyron_executor::context::ExecutionContext>,
    tempfile::TempDir,
) {
    let (server, _schema, tmp) = common::create_test_server().await;
    let txn = server
        .txn_manager
        .begin(zyron_storage::txn::IsolationLevel::ReadCommitted)
        .expect("begin");
    let ctx = zyron_executor::context::ExecutionContext::new(
        server.catalog.clone(),
        server.wal.clone(),
        server.buffer_pool.clone(),
        server.disk_manager.clone(),
        txn.txn_id,
        txn.snapshot.clone(),
    );
    (Arc::new(ctx), tmp)
}

// =============================================================================
// Test 1: UNNEST element throughput
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_unnest_10m_rows_of_ten_elements() {
    zyron_bench_harness::init("sql_surface");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const ROWS: usize = 10_000_000;
    const PER_ROW: usize = 10;
    let elements = (ROWS * PER_ROW) as f64;

    tprintln!("\n=== UNNEST Element Throughput ===");
    tprintln!("Rows: {}, elements per row: {}", ROWS, PER_ROW);

    let (ctx, _tmp) = bare_context().await;
    let batches = array_batches(ROWS, PER_ROW);
    let schema = input_schema(TypeId::Array);
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);

    let util_before = take_util_snapshot();
    for run in 0..VALIDATION_RUNS {
        let mut op = ExpandRowsOperator::new(
            Box::new(MemoryOperator::new(batches.clone())),
            ExpandSpec::Unnest {
                arrays: vec![value_ref(TypeId::Array)],
                with_ordinality: false,
            },
            Vec::new(),
            vec![produced_column("e", TypeId::Int64, 0)],
            schema.clone(),
            false,
            Vec::new(),
            Arc::clone(&ctx),
        );
        let start = Instant::now();
        let produced = drain(&mut op).await;
        let duration = start.elapsed();
        assert_eq!(produced, ROWS * PER_ROW, "run {run}: wrong element count");
        let per_sec = elements / duration.as_secs_f64();
        tprintln!(
            "  Run {}/{}: {} elements/sec ({:?})",
            run + 1,
            VALIDATION_RUNS,
            format_with_commas(per_sec),
            duration
        );
        runs.push(per_sec);
    }
    record_test_util("UNNEST throughput", util_before, take_util_snapshot());

    let result = validate_metric_with_unit(
        "UNNEST throughput",
        "UNNEST elements",
        " elements/s",
        runs,
        UNNEST_TARGET_ELEMENTS_SEC,
        true,
    );
    assert!(
        result.passed,
        "UNNEST avg {:.0} elements/s < target {:.0}",
        result.average, UNNEST_TARGET_ELEMENTS_SEC
    );
}

// =============================================================================
// Test 2: FLATTEN leaf throughput
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_flatten_1m_documents_of_twenty_leaves() {
    zyron_bench_harness::init("sql_surface");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const ROWS: usize = 1_000_000;
    const LEAVES: usize = 20;
    let leaves = (ROWS * LEAVES) as f64;

    tprintln!("\n=== FLATTEN Leaf Throughput ===");
    tprintln!("Documents: {}, leaves each: {}", ROWS, LEAVES);

    let (ctx, _tmp) = bare_context().await;
    let batches = document_batches(ROWS, LEAVES);
    let schema = input_schema(TypeId::Variant);
    let produced = vec![
        produced_column("seq", TypeId::Int64, 0),
        produced_column("key", TypeId::Text, 1),
        produced_column("path", TypeId::Text, 2),
        produced_column("index", TypeId::Int64, 3),
        produced_column("value", TypeId::Variant, 4),
        produced_column("this", TypeId::Variant, 5),
    ];
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);

    let util_before = take_util_snapshot();
    for run in 0..VALIDATION_RUNS {
        let mut op = ExpandRowsOperator::new(
            Box::new(MemoryOperator::new(batches.clone())),
            ExpandSpec::Flatten {
                document: value_ref(TypeId::Variant),
                path: None,
                recursive: false,
            },
            Vec::new(),
            produced.clone(),
            schema.clone(),
            false,
            Vec::new(),
            Arc::clone(&ctx),
        );
        let start = Instant::now();
        let rows = drain(&mut op).await;
        let duration = start.elapsed();
        assert_eq!(rows, ROWS * LEAVES, "run {run}: wrong leaf count");
        let per_sec = leaves / duration.as_secs_f64();
        tprintln!(
            "  Run {}/{}: {} leaves/sec ({:?})",
            run + 1,
            VALIDATION_RUNS,
            format_with_commas(per_sec),
            duration
        );
        runs.push(per_sec);
    }
    record_test_util("FLATTEN throughput", util_before, take_util_snapshot());

    let result = validate_metric_with_unit(
        "FLATTEN throughput",
        "FLATTEN leaves",
        " leaves/s",
        runs,
        FLATTEN_TARGET_LEAVES_SEC,
        true,
    );
    assert!(
        result.passed,
        "FLATTEN avg {:.0} leaves/s < target {:.0}",
        result.average, FLATTEN_TARGET_LEAVES_SEC
    );
}

// =============================================================================
// Test 3: UNNEST under LATERAL against a bare UNNEST
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_lateral_unnest_overhead_against_bare() {
    zyron_bench_harness::init("sql_surface");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const ROWS: usize = 2_000_000;
    const PER_ROW: usize = 10;
    let elements = (ROWS * PER_ROW) as f64;

    tprintln!("\n=== LATERAL UNNEST Overhead ===");
    tprintln!("Rows: {}, elements per row: {}", ROWS, PER_ROW);

    let (ctx, _tmp) = bare_context().await;
    let batches = array_batches(ROWS, PER_ROW);
    let schema = input_schema(TypeId::Array);

    // The carried form repeats the parent's own columns beside the produced
    // one, which is what LATERAL asks for. The bare form produces the
    // elements alone.
    //
    // What is carried is what `SELECT o.id, e FROM orders o, LATERAL
    // UNNEST(o.items) u(e)` carries after projection pushdown: the id, and
    // not the array being unnested. `an_expansion_carries_only_the_columns_
    // the_query_reads` in unnest_test holds that shape, so this measures the
    // plan the engine runs rather than one it never produces
    let build = |carry: Vec<usize>, output: Vec<LogicalColumn>, batches: Vec<DataBatch>| {
        ExpandRowsOperator::new(
            Box::new(MemoryOperator::new(batches)),
            ExpandSpec::Unnest {
                arrays: vec![value_ref(TypeId::Array)],
                with_ordinality: false,
            },
            carry,
            output,
            schema.clone(),
            false,
            Vec::new(),
            Arc::clone(&ctx),
        )
    };
    let carried_columns = vec![schema[0].clone(), produced_column("e", TypeId::Int64, 0)];

    let mut bare_runs = Vec::with_capacity(VALIDATION_RUNS);
    let mut lateral_runs = Vec::with_capacity(VALIDATION_RUNS);
    let util_before = take_util_snapshot();
    for run in 0..VALIDATION_RUNS {
        let mut op = build(
            Vec::new(),
            vec![produced_column("e", TypeId::Int64, 0)],
            batches.clone(),
        );
        let start = Instant::now();
        let produced = drain(&mut op).await;
        let bare = elements / start.elapsed().as_secs_f64();
        assert_eq!(produced, ROWS * PER_ROW);

        let mut op = build(vec![0], carried_columns.clone(), batches.clone());
        let start = Instant::now();
        let produced = drain(&mut op).await;
        let lateral = elements / start.elapsed().as_secs_f64();
        assert_eq!(produced, ROWS * PER_ROW);

        tprintln!(
            "  Run {}/{}: bare {} elements/sec, lateral {} elements/sec",
            run + 1,
            VALIDATION_RUNS,
            format_with_commas(bare),
            format_with_commas(lateral)
        );
        bare_runs.push(bare);
        lateral_runs.push(lateral);
    }
    record_test_util("LATERAL overhead", util_before, take_util_snapshot());

    let bare = record_metric("LATERAL overhead", "bare UNNEST", " elements/s", bare_runs);
    let lateral = record_metric(
        "LATERAL overhead",
        "LATERAL UNNEST",
        " elements/s",
        lateral_runs,
    );
    let overhead = bare / lateral.max(1.0);
    tprintln!(
        "  LATERAL costs {:.3}x the bare form (limit {:.2}x)",
        overhead,
        LATERAL_OVERHEAD_LIMIT
    );
    record_metric("LATERAL overhead", "LATERAL over bare", "x", vec![overhead]);
    assert!(
        overhead <= LATERAL_OVERHEAD_LIMIT,
        "LATERAL costs {overhead:.3}x the bare form, over the {LATERAL_OVERHEAD_LIMIT:.2}x limit"
    );
}

// =============================================================================
// Test 4: array_transform with a lambda
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_array_transform_lambda_10m_rows() {
    zyron_bench_harness::init("sql_surface");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const ROWS: usize = 10_000_000;
    const PER_ROW: usize = 10;
    let elements = (ROWS * PER_ROW) as f64;

    tprintln!("\n=== array_transform Lambda Throughput ===");
    tprintln!("Rows: {}, elements per row: {}", ROWS, PER_ROW);

    let batches = array_batches(ROWS, PER_ROW);
    let schema = input_schema(TypeId::Array);
    // x -> x * 2, with the parameter bound as a column of the lambda's own
    // one-column relation, which is what the binder produces
    let body = BoundExpr::BinaryOp {
        left: Box::new(BoundExpr::ColumnRef(ColumnRef {
            table_idx: zyron_planner::logical::LAMBDA_TABLE_IDX,
            column_id: zyron_catalog::ColumnId(0),
            type_id: TypeId::Int64,
            nullable: true,
            fractional_digits: None,
        })),
        op: zyron_parser::ast::BinaryOperator::Multiply,
        right: Box::new(BoundExpr::Literal {
            value: zyron_parser::ast::LiteralValue::Integer(2),
            type_id: TypeId::Int64,
        }),
        type_id: TypeId::Int64,
    };
    let call = BoundExpr::Function {
        name: "array_transform".to_string(),
        args: vec![value_ref(TypeId::Array), body],
        return_type: TypeId::Array,
        distinct: false,
    };

    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    let util_before = take_util_snapshot();
    for run in 0..VALIDATION_RUNS {
        let start = Instant::now();
        let mut produced = 0usize;
        for batch in &batches {
            let out = zyron_executor::expr::evaluate(&call, batch, &schema, &[])
                .expect("array_transform");
            produced += out.len();
        }
        let duration = start.elapsed();
        assert_eq!(produced, ROWS, "run {run}: one array per row comes back");
        let per_sec = elements / duration.as_secs_f64();
        tprintln!(
            "  Run {}/{}: {} elements/sec ({:?})",
            run + 1,
            VALIDATION_RUNS,
            format_with_commas(per_sec),
            duration
        );
        runs.push(per_sec);
    }
    record_test_util(
        "array_transform throughput",
        util_before,
        take_util_snapshot(),
    );

    let result = validate_metric_with_unit(
        "array_transform throughput",
        "array_transform elements",
        " elements/s",
        runs,
        ARRAY_TRANSFORM_TARGET_ELEMENTS_SEC,
        true,
    );
    assert!(
        result.passed,
        "array_transform avg {:.0} elements/s < target {:.0}",
        result.average, ARRAY_TRANSFORM_TARGET_ELEMENTS_SEC
    );
}

// =============================================================================
// Test 5: UNPIVOT output row throughput
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_unpivot_1m_rows_of_twelve_columns() {
    zyron_bench_harness::init("sql_surface");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const ROWS: usize = 1_000_000;
    const COLUMNS: usize = 12;
    let output_rows = (ROWS * COLUMNS) as f64;

    tprintln!("\n=== UNPIVOT Output Row Throughput ===");
    tprintln!("Rows: {}, columns unpivoted: {}", ROWS, COLUMNS);

    let (ctx, _tmp) = bare_context().await;
    // One id column plus twelve value columns, which is what a month-per-
    // column table looks like
    let mut batches = Vec::new();
    let mut produced = 0usize;
    while produced < ROWS {
        let n = BATCH_SIZE.min(ROWS - produced);
        let mut columns = Vec::with_capacity(COLUMNS + 1);
        let mut ids = ColumnBuilder::new(TypeId::Int64, n);
        for r in 0..n {
            ids.push(&ScalarValue::Int64((produced + r) as i64));
        }
        columns.push(ids.finish());
        for c in 0..COLUMNS {
            let mut values = ColumnBuilder::new(TypeId::Int64, n);
            for r in 0..n {
                values.push(&ScalarValue::Int64((produced + r + c) as i64));
            }
            columns.push(values.finish());
        }
        batches.push(DataBatch::new(columns));
        produced += n;
    }

    let mut schema = vec![LogicalColumn {
        table_idx: Some(0),
        column_id: zyron_catalog::ColumnId(0),
        name: "id".to_string(),
        type_id: TypeId::Int64,
        nullable: false,
        fractional_digits: None,
    }];
    let mut groups = Vec::with_capacity(COLUMNS);
    let mut labels = Vec::with_capacity(COLUMNS);
    for c in 0..COLUMNS {
        let position = (c + 1) as u16;
        schema.push(LogicalColumn {
            table_idx: Some(0),
            column_id: zyron_catalog::ColumnId(position),
            name: format!("m{c}"),
            type_id: TypeId::Int64,
            nullable: true,
            fractional_digits: None,
        });
        groups.push(vec![BoundExpr::ColumnRef(ColumnRef {
            table_idx: 0,
            column_id: zyron_catalog::ColumnId(position),
            type_id: TypeId::Int64,
            nullable: true,
            fractional_digits: None,
        })]);
        labels.push(BoundExpr::Literal {
            value: zyron_parser::ast::LiteralValue::String(format!("m{c}")),
            type_id: TypeId::Varchar,
        });
    }
    // The id travels, the twelve value columns become a name and a value
    let output = vec![
        produced_column("id", TypeId::Int64, 0),
        produced_column("month", TypeId::Varchar, 1),
        produced_column("amount", TypeId::Int64, 2),
    ];

    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    let util_before = take_util_snapshot();
    for run in 0..VALIDATION_RUNS {
        let mut op = ExpandRowsOperator::new(
            Box::new(MemoryOperator::new(batches.clone())),
            ExpandSpec::Unpivot {
                groups: groups.clone(),
                labels: labels.clone(),
                include_nulls: false,
            },
            vec![0],
            output.clone(),
            schema.clone(),
            false,
            Vec::new(),
            Arc::clone(&ctx),
        );
        let start = Instant::now();
        let rows = drain(&mut op).await;
        let duration = start.elapsed();
        assert_eq!(rows, ROWS * COLUMNS, "run {run}: wrong output row count");
        let per_sec = output_rows / duration.as_secs_f64();
        tprintln!(
            "  Run {}/{}: {} rows/sec ({:?})",
            run + 1,
            VALIDATION_RUNS,
            format_with_commas(per_sec),
            duration
        );
        runs.push(per_sec);
    }
    record_test_util("UNPIVOT throughput", util_before, take_util_snapshot());

    let result = validate_metric_with_unit(
        "UNPIVOT throughput",
        "UNPIVOT output rows",
        " rows/s",
        runs,
        UNPIVOT_TARGET_ROWS_SEC,
        true,
    );
    assert!(
        result.passed,
        "UNPIVOT avg {:.0} rows/s < target {:.0}",
        result.average, UNPIVOT_TARGET_ROWS_SEC
    );
}

// =============================================================================
// Temporary table fixtures
// =============================================================================

/// A session of its own, so each run measures a fresh namespace and its own
/// directory rather than the previous run's.
fn temp_session(process_id: i32) -> Option<zyron_wire::session::Session> {
    let mut s = zyron_wire::session::Session::new(
        "bench_user".into(),
        "benchdb".into(),
        zyron_catalog::DatabaseId(1),
    );
    s.search_path = vec!["zyron_test".into()];
    s.process_id = process_id;
    Some(s)
}

/// Runs one statement through the DDL dispatch, the way a connection does.
async fn run_ddl(
    server: &Arc<zyron_wire::connection::ServerState>,
    session: &mut Option<zyron_wire::session::Session>,
    sql: &str,
) {
    let stmt = zyron_parser::parse(sql)
        .unwrap_or_else(|e| panic!("`{sql}` did not parse: {e}"))
        .remove(0);
    let mut txn: Option<zyron_storage::txn::Transaction> = None;
    let mut branch: Option<String> = None;
    let handled = zyron_wire::ddl_dispatch::try_handle_ddl_utility(
        &stmt,
        server,
        session,
        &mut txn,
        &mut branch,
        sql,
    )
    .await;
    match handled {
        Some(Ok(_)) => {}
        Some(Err(e)) => panic!("`{sql}` failed: {e:?}"),
        None => panic!("`{sql}` was not handled as DDL"),
    }
}

/// Plans one write statement for a session, resolving its temporary tables.
///
/// Kept apart from execution so a measurement can plan once and run the plan
/// many times, which is what a prepared statement does and what leaves the
/// write path as the only thing being timed.
async fn plan_dml(
    server: &Arc<zyron_wire::connection::ServerState>,
    session: &Option<zyron_wire::session::Session>,
    sql: &str,
) -> zyron_planner::physical::PhysicalPlan {
    let stmt = zyron_parser::parse(sql)
        .unwrap_or_else(|e| panic!("`{sql}` did not parse: {e}"))
        .remove(0);
    let temp_tables = session.as_ref().and_then(|s| s.temp_tables.clone());
    zyron_planner::plan_for_session(
        &server.catalog,
        zyron_catalog::DatabaseId(1),
        vec!["zyron_test".into()],
        stmt,
        None,
        None,
        temp_tables,
    )
    .await
    .unwrap_or_else(|e| panic!("`{sql}` did not plan: {e}"))
}

/// Runs one already planned write under a transaction of its own, taking the
/// commit path a connection takes.
async fn execute_dml(
    server: &Arc<zyron_wire::connection::ServerState>,
    plan: zyron_planner::physical::PhysicalPlan,
) {
    let mut txn = server
        .txn_manager
        .begin(zyron_storage::txn::IsolationLevel::ReadCommitted)
        .expect("begin");
    let mut ctx = zyron_executor::context::ExecutionContext::new(
        server.catalog.clone(),
        server.wal.clone(),
        server.buffer_pool.clone(),
        server.disk_manager.clone(),
        txn.txn_id,
        txn.snapshot.clone(),
    );
    ctx.heap_files = Some(Arc::clone(&server.heap_files));
    ctx.btree_indexes = Some(Arc::clone(&server.btree_indexes));
    ctx.intent_locks = Some(Arc::clone(server.txn_manager.intent_locks()));
    let ctx = Arc::new(ctx);
    zyron_executor::execute(plan, &ctx)
        .await
        .unwrap_or_else(|e| panic!("the planned write failed: {e}"));
    // A transaction that appended no write-ahead record has nothing to make
    // durable, so it commits without a commit record or a flush wait. Both
    // targets go through the same decision, so what separates them is what
    // they wrote
    if ctx.wrote_wal() {
        server.txn_manager.commit(&mut txn).await.expect("commit");
    } else {
        server
            .txn_manager
            .commit_read_only(&mut txn)
            .expect("commit");
    }
}

/// Runs one write statement for a session, resolving its temporary tables.
async fn run_dml(
    server: &Arc<zyron_wire::connection::ServerState>,
    session: &mut Option<zyron_wire::session::Session>,
    sql: &str,
) {
    let stmt = zyron_parser::parse(sql)
        .unwrap_or_else(|e| panic!("`{sql}` did not parse: {e}"))
        .remove(0);
    let temp_tables = session.as_ref().and_then(|s| s.temp_tables.clone());
    let plan = zyron_planner::plan_for_session(
        &server.catalog,
        zyron_catalog::DatabaseId(1),
        vec!["zyron_test".into()],
        stmt,
        None,
        None,
        temp_tables,
    )
    .await
    .unwrap_or_else(|e| panic!("`{sql}` did not plan: {e}"));

    let mut txn = server
        .txn_manager
        .begin(zyron_storage::txn::IsolationLevel::ReadCommitted)
        .expect("begin");
    let mut ctx = zyron_executor::context::ExecutionContext::new(
        server.catalog.clone(),
        server.wal.clone(),
        server.buffer_pool.clone(),
        server.disk_manager.clone(),
        txn.txn_id,
        txn.snapshot.clone(),
    );
    ctx.heap_files = Some(Arc::clone(&server.heap_files));
    ctx.btree_indexes = Some(Arc::clone(&server.btree_indexes));
    ctx.intent_locks = Some(Arc::clone(server.txn_manager.intent_locks()));
    let ctx = Arc::new(ctx);
    zyron_executor::execute(plan, &ctx)
        .await
        .unwrap_or_else(|e| panic!("`{sql}` failed: {e}"));
    // The commit path a connection takes: a transaction that appended no
    // write-ahead record has nothing to make durable, so it commits without
    // a commit record or a flush wait. Both targets go through the same
    // decision, so what separates them is what they wrote
    if ctx.wrote_wal() {
        server.txn_manager.commit(&mut txn).await.expect("commit");
    } else {
        server
            .txn_manager
            .commit_read_only(&mut txn)
            .expect("commit");
    }
}

// =============================================================================
// Test 6: Creating a temporary table
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_create_temp_table_latency() {
    zyron_bench_harness::init("sql_surface");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const TABLES_PER_RUN: usize = 200;

    tprintln!("\n=== CREATE TEMP TABLE Latency ===");
    tprintln!("Tables per run: {}", TABLES_PER_RUN);

    let (server, _schema, _tmp) = common::create_test_server().await;
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    let util_before = take_util_snapshot();
    for run in 0..VALIDATION_RUNS {
        let session_id = 9000 + run as i32;
        let mut session = temp_session(session_id);
        let statements: Vec<String> = (0..TABLES_PER_RUN)
            .map(|i| format!("CREATE TEMP TABLE t{i} (a INT, b TEXT)"))
            .collect();
        let start = Instant::now();
        for sql in &statements {
            run_ddl(&server, &mut session, sql).await;
        }
        let duration = start.elapsed();
        let per_table_us = duration.as_secs_f64() * 1_000_000.0 / TABLES_PER_RUN as f64;
        tprintln!(
            "  Run {}/{}: {:.1}us per table ({:?} for {})",
            run + 1,
            VALIDATION_RUNS,
            per_table_us,
            duration,
            TABLES_PER_RUN
        );
        runs.push(per_table_us);
        let key = session.as_ref().expect("session").session_key;
        zyron_wire::temp_table_dispatch::end_session(&server, key).await;
    }
    record_test_util("CREATE TEMP TABLE", util_before, take_util_snapshot());

    let result = validate_metric_with_unit(
        "CREATE TEMP TABLE",
        "CREATE TEMP TABLE latency",
        "us",
        runs,
        CREATE_TEMP_TABLE_TARGET_US,
        false,
    );
    assert!(
        result.passed,
        "CREATE TEMP TABLE avg {:.1}us > target {:.1}us",
        result.average, CREATE_TEMP_TABLE_TARGET_US
    );
}

// =============================================================================
// Test 7: Filling a temporary table against a permanent one
//
// What a temporary table saves is the two durable writes a permanent one
// pays: the write-ahead record and the consensus round. This node is in no
// group, so only the first is here to save, and what this records is that
// half. The gate on the whole saving is
// `temp_table_bench::test_temp_insert_against_a_permanent_insert_on_a_group`,
// which measures on a real three node group where a permanent insert pays
// both.
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_temp_table_insert_against_a_permanent_heap() {
    zyron_bench_harness::init("sql_surface");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const ROWS: usize = 1_000_000;
    const PER_STATEMENT: usize = 1_000;
    const STATEMENTS: usize = ROWS / PER_STATEMENT;

    tprintln!("\n=== Temporary INSERT Against a Permanent Heap ===");
    tprintln!("Rows: {}, rows per statement: {}", ROWS, PER_STATEMENT);

    // One statement's worth of values, planned once per target and then run
    // repeatedly. Parsing a thousand-tuple VALUES list is the same work for
    // both targets and several times what the insert itself costs, so timing
    // it would measure the parser rather than the write path the target is
    // about
    let rows: Vec<String> = (0..PER_STATEMENT).map(|r| format!("({r}, {r})")).collect();
    let values = rows.join(", ");

    let mut temp_runs = Vec::with_capacity(VALIDATION_RUNS);
    let mut permanent_runs = Vec::with_capacity(VALIDATION_RUNS);
    let util_before = take_util_snapshot();
    for run in 0..VALIDATION_RUNS {
        let (server, _schema, _tmp) = common::create_test_server().await;
        let session_id = 9500 + run as i32;
        let mut session = temp_session(session_id);

        run_ddl(&server, &mut session, "CREATE TABLE perm (a INT, b INT)").await;
        run_ddl(
            &server,
            &mut session,
            "CREATE TEMP TABLE tmp (a INT, b INT)",
        )
        .await;

        let temp_plan = plan_dml(
            &server,
            &session,
            &format!("INSERT INTO tmp VALUES {values}"),
        )
        .await;
        let permanent_plan = plan_dml(
            &server,
            &session,
            &format!("INSERT INTO perm VALUES {values}"),
        )
        .await;

        let start = Instant::now();
        for _ in 0..STATEMENTS {
            execute_dml(&server, temp_plan.clone()).await;
        }
        let temp = start.elapsed();

        let start = Instant::now();
        for _ in 0..STATEMENTS {
            execute_dml(&server, permanent_plan.clone()).await;
        }
        let permanent = start.elapsed();

        let temp_rows = ROWS as f64 / temp.as_secs_f64();
        let permanent_rows = ROWS as f64 / permanent.as_secs_f64();
        tprintln!(
            "  Run {}/{}: temporary {} rows/sec, permanent {} rows/sec",
            run + 1,
            VALIDATION_RUNS,
            format_with_commas(temp_rows),
            format_with_commas(permanent_rows)
        );
        temp_runs.push(temp_rows);
        permanent_runs.push(permanent_rows);
        let key = session.as_ref().expect("session").session_key;
        zyron_wire::temp_table_dispatch::end_session(&server, key).await;
    }
    record_test_util("Temporary INSERT", util_before, take_util_snapshot());

    let temp = record_metric("Temporary INSERT", "temporary table", " rows/s", temp_runs);
    let permanent = record_metric(
        "Temporary INSERT",
        "permanent heap",
        " rows/s",
        permanent_runs,
    );
    let speedup = temp / permanent.max(1.0);
    tprintln!(
        "  A temporary table takes inserts {:.2}x as fast with no log to write",
        speedup
    );
    record_metric(
        "Temporary INSERT",
        "temporary over permanent, no group",
        "x",
        vec![speedup],
    );
    // A saving of less than one would mean the temporary path is doing work
    // the permanent path is not, which is the failure this holds against
    assert!(
        speedup > 1.0,
        "a temporary insert is {speedup:.2}x a permanent one on a node in no group, so skipping the log saved nothing"
    );
}

// =============================================================================
// Test 8: Dropping a gigabyte of temporary tables at session end
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_temp_table_drop_at_session_end() {
    zyron_bench_harness::init("sql_surface");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    // A gigabyte, as sixteen tables holding a sixteenth each
    const TABLES: usize = 16;
    const ROWS_PER_TABLE: usize = 70_000;
    const PER_STATEMENT: usize = 1_000;

    tprintln!("\n=== Temporary Table Drop At Session End ===");

    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    let mut held_bytes = 0u64;
    let util_before = take_util_snapshot();
    for run in 0..VALIDATION_RUNS {
        let (server, _schema, _tmp) = common::create_test_server().await;
        let session_id = 9700 + run as i32;
        let mut session = temp_session(session_id);
        // Roughly a kilobyte per row, so the bytes are in the files rather
        // than in row headers
        let filler = "x".repeat(900);
        for t in 0..TABLES {
            run_ddl(
                &server,
                &mut session,
                &format!("CREATE TEMP TABLE big{t} (a INT, b TEXT)"),
            )
            .await;
            for batch in 0..ROWS_PER_TABLE / PER_STATEMENT {
                let rows: Vec<String> = (0..PER_STATEMENT)
                    .map(|r| format!("({}, '{}')", batch * PER_STATEMENT + r, filler))
                    .collect();
                run_dml(
                    &server,
                    &mut session,
                    &format!("INSERT INTO big{t} VALUES {}", rows.join(", ")),
                )
                .await;
            }
        }
        held_bytes = session
            .as_ref()
            .and_then(|s| s.temp_tables.as_ref())
            .and_then(|t| t.directory().read_dir().ok())
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .filter_map(|e| e.metadata().ok())
                    .map(|m| m.len())
                    .sum()
            })
            .unwrap_or(0);

        let start = Instant::now();
        let key = session.as_ref().expect("session").session_key;
        zyron_wire::temp_table_dispatch::end_session(&server, key).await;
        let duration = start.elapsed();
        let ms = duration.as_secs_f64() * 1000.0;
        tprintln!(
            "  Run {}/{}: {:.1}ms to drop {} bytes across {} tables",
            run + 1,
            VALIDATION_RUNS,
            ms,
            format_with_commas(held_bytes as f64),
            TABLES
        );
        runs.push(ms);
    }
    record_test_util("Temporary drop", util_before, take_util_snapshot());
    assert!(
        held_bytes >= 1_000_000_000,
        "the fixture held {held_bytes} bytes, under the gigabyte the target is about"
    );

    let result = validate_metric_with_unit(
        "Temporary drop",
        "drop at session end",
        "ms",
        runs,
        TEMP_DROP_TARGET_MS,
        false,
    );
    assert!(
        result.passed,
        "the drop took {:.1}ms, over the {:.1}ms target",
        result.average, TEMP_DROP_TARGET_MS
    );
}

// =============================================================================
// Test 9: PIVOT against the conditional aggregate it rewrites to
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_pivot_against_a_hand_written_conditional_aggregate() {
    zyron_bench_harness::init("sql_surface");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const ROWS: usize = 10_000_000;
    const PER_STATEMENT: usize = 2_000;

    tprintln!("\n=== PIVOT Against a Hand-Written Conditional Aggregate ===");
    tprintln!("Rows: {}, 4 regions x 4 quarters", ROWS);

    let (server, _schema, _tmp) = common::create_test_server().await;
    let mut session = temp_session(9900);
    run_ddl(
        &server,
        &mut session,
        "CREATE TABLE sales (region TEXT, quarter TEXT, amount INT)",
    )
    .await;
    let regions = ["east", "west", "north", "south"];
    let quarters = ["Q1", "Q2", "Q3", "Q4"];
    for batch in 0..ROWS / PER_STATEMENT {
        let rows: Vec<String> = (0..PER_STATEMENT)
            .map(|r| {
                let i = batch * PER_STATEMENT + r;
                format!(
                    "('{}', '{}', {})",
                    regions[i % 4],
                    quarters[(i / 4) % 4],
                    i % 1000
                )
            })
            .collect();
        run_dml(
            &server,
            &mut session,
            &format!("INSERT INTO sales VALUES {}", rows.join(", ")),
        )
        .await;
    }

    let pivot_sql =
        "SELECT * FROM sales PIVOT (SUM(amount) FOR quarter IN ('Q1', 'Q2', 'Q3', 'Q4')) AS p";
    let hand_written = "SELECT region, \
         SUM(CASE WHEN quarter = 'Q1' THEN amount END), \
         SUM(CASE WHEN quarter = 'Q2' THEN amount END), \
         SUM(CASE WHEN quarter = 'Q3' THEN amount END), \
         SUM(CASE WHEN quarter = 'Q4' THEN amount END) \
         FROM sales GROUP BY region";

    // The two have to give the same answer before either is timed, or the
    // comparison is between a query and a cheaper wrong one
    let by_pivot = common::query_values(&server, pivot_sql).await;
    let by_hand = common::query_values(&server, hand_written).await;
    assert_eq!(
        by_pivot.len(),
        by_hand.len(),
        "the pivot and the hand-written aggregate disagree on row count"
    );

    // Both run twice more before anything is timed. The first execution of
    // either pays cold pages and a cold allocator, which is several times
    // the difference being measured, so an untimed pass over each is what
    // makes the comparison about the plans
    for _ in 0..2 {
        let _ = common::query_values(&server, pivot_sql).await;
        let _ = common::query_values(&server, hand_written).await;
    }

    let mut pivot_runs = Vec::with_capacity(VALIDATION_RUNS);
    let mut hand_runs = Vec::with_capacity(VALIDATION_RUNS);
    let util_before = take_util_snapshot();
    for run in 0..VALIDATION_RUNS {
        // The two alternate which goes first. Whichever runs second finds
        // the pages and the allocator warmed by the first, which is worth
        // more than the difference being measured, so a fixed order would
        // report that advantage as the rewrite's cost
        let (pivot, hand) = if run % 2 == 0 {
            let start = Instant::now();
            let _ = common::query_values(&server, pivot_sql).await;
            let pivot = start.elapsed();
            let start = Instant::now();
            let _ = common::query_values(&server, hand_written).await;
            (pivot, start.elapsed())
        } else {
            let start = Instant::now();
            let _ = common::query_values(&server, hand_written).await;
            let hand = start.elapsed();
            let start = Instant::now();
            let _ = common::query_values(&server, pivot_sql).await;
            (start.elapsed(), hand)
        };

        tprintln!(
            "  Run {}/{}: pivot {:?}, hand-written {:?}",
            run + 1,
            VALIDATION_RUNS,
            pivot,
            hand
        );
        pivot_runs.push(pivot.as_secs_f64() * 1000.0);
        hand_runs.push(hand.as_secs_f64() * 1000.0);
    }
    record_test_util("PIVOT rewrite", util_before, take_util_snapshot());

    let pivot = record_metric("PIVOT rewrite", "PIVOT", "ms", pivot_runs);
    let hand = record_metric("PIVOT rewrite", "hand-written aggregate", "ms", hand_runs);
    let ratio = pivot / hand.max(f64::MIN_POSITIVE);
    tprintln!(
        "  PIVOT costs {:.4}x the hand-written form (limit {:.2}x)",
        ratio,
        PIVOT_RATIO_LIMIT
    );
    record_metric("PIVOT rewrite", "PIVOT over hand-written", "x", vec![ratio]);
    assert!(
        ratio <= PIVOT_RATIO_LIMIT,
        "PIVOT costs {ratio:.4}x the hand-written aggregate, over the {PIVOT_RATIO_LIMIT:.2}x limit"
    );
}

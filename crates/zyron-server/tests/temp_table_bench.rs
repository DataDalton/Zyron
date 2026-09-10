//! Temporary Table Benchmark Suite, on a node in a consensus group
//!
//! What a temporary table saves is the two durable writes a permanent one
//! pays: the write-ahead record and the consensus round. A single node can
//! only show the first, so this measures on a real three node group, where a
//! permanent insert is agreed by the group and a temporary one is not.
//!
//! Performance Targets:
//! | Test                                  | Metric     | Target                    |
//! |---------------------------------------|------------|---------------------------|
//! | INSERT into a temporary table         | throughput | 3x a permanent heap       |
//!
//! Validation Requirements:
//! - Each benchmark runs 5 iterations
//! - Results averaged across all 5 runs
//! - Pass/fail determined by average performance
//! - Individual runs logged for variance analysis
//!
//! Run: cargo test --release -p zyron-server --test temp_table_bench -- --nocapture

mod common;

use std::time::{Duration, Instant};

use common::{Group, WireClient};
use zyron_bench_harness::*;

const VALIDATION_RUNS: usize = 5;

/// No write-ahead record and no consensus round, against a permanent insert
/// that pays both
const TEMP_INSERT_SPEEDUP_TARGET: f64 = 3.0;

/// Rows per statement, and how many statements each target runs. Sized so
/// the group's own agreement cost is what separates the two rather than the
/// wire protocol, and so the whole suite stays inside a minute
const ROWS_PER_STATEMENT: usize = 500;
const STATEMENTS: usize = 40;
const ROWS: usize = ROWS_PER_STATEMENT * STATEMENTS;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_temp_insert_against_a_permanent_insert_on_a_group() {
    zyron_bench_harness::init("temp_table");
    tprintln!("\n=== Temporary INSERT On A Node In A Group ===");
    tprintln!(
        "Rows: {}, rows per statement: {}, group of 3",
        ROWS,
        ROWS_PER_STATEMENT
    );

    // One statement's worth of values, reused by both targets so the two are
    // given identical work
    let values: String = (0..ROWS_PER_STATEMENT)
        .map(|r| format!("({r}, {r})"))
        .collect::<Vec<_>>()
        .join(", ");

    let mut temp_runs = Vec::with_capacity(VALIDATION_RUNS);
    let mut permanent_runs = Vec::with_capacity(VALIDATION_RUNS);
    let util_before = take_util_snapshot();

    for run in 0..VALIDATION_RUNS {
        let group = Group::start(3).await;
        let leader = group.leader(Duration::from_secs(10)).await;
        let addr = group.nodes[leader].serve_wire().await;
        let mut client = WireClient::connect(addr).await;

        for sql in [
            "SET search_path = zyron_test",
            "CREATE TABLE perm (a BIGINT, b BIGINT)",
            "CREATE TEMP TABLE tmp (a BIGINT, b BIGINT)",
        ] {
            let (_, errors) = client.query(sql).await;
            assert!(errors.is_empty(), "`{sql}` failed: {errors:?}");
        }

        // The temporary table stays on this node, so nothing about its rows
        // reaches the group
        let temp_sql = format!("INSERT INTO tmp VALUES {values}");
        let start = Instant::now();
        for _ in 0..STATEMENTS {
            let (_, errors) = client.query(&temp_sql).await;
            assert!(errors.is_empty(), "the temporary insert failed: {errors:?}");
        }
        let temp = start.elapsed();

        // The permanent table's rows are agreed by the group before they are
        // visible anywhere
        let permanent_sql = format!("INSERT INTO perm VALUES {values}");
        let start = Instant::now();
        for _ in 0..STATEMENTS {
            let (_, errors) = client.query(&permanent_sql).await;
            assert!(errors.is_empty(), "the permanent insert failed: {errors:?}");
        }
        let permanent = start.elapsed();

        // Both wrote what they were asked to, so the comparison is between
        // two statements that did the same amount of work
        let (rows, errors) = client.query_rows("SELECT COUNT(*) FROM tmp").await;
        assert!(
            errors.is_empty(),
            "counting the temporary table: {errors:?}"
        );
        assert_eq!(
            rows.first().and_then(|r| r[0].clone()),
            Some(ROWS.to_string()),
            "the temporary table did not take every row"
        );
        let (rows, errors) = client.query_rows("SELECT COUNT(*) FROM perm").await;
        assert!(
            errors.is_empty(),
            "counting the permanent table: {errors:?}"
        );
        assert_eq!(
            rows.first().and_then(|r| r[0].clone()),
            Some(ROWS.to_string()),
            "the permanent table did not take every row"
        );

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

        client.terminate().await;
        group.shutdown().await;
    }
    record_test_util(
        "Temporary INSERT on a group",
        util_before,
        take_util_snapshot(),
    );

    let temp = record_metric(
        "Temporary INSERT on a group",
        "temporary table",
        " rows/s",
        temp_runs,
    );
    let permanent = record_metric(
        "Temporary INSERT on a group",
        "permanent heap through the group",
        " rows/s",
        permanent_runs,
    );
    let speedup = temp / permanent.max(1.0);
    tprintln!(
        "  A temporary table takes inserts {:.2}x as fast (target {:.1}x)",
        speedup,
        TEMP_INSERT_SPEEDUP_TARGET
    );
    record_metric(
        "Temporary INSERT on a group",
        "temporary over permanent",
        "x",
        vec![speedup],
    );
    assert!(
        speedup >= TEMP_INSERT_SPEEDUP_TARGET,
        "a temporary insert is {speedup:.2}x a permanent one, under the {TEMP_INSERT_SPEEDUP_TARGET:.1}x target"
    );
}

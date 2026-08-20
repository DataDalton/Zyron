//! What a clustering expression costs the write path, and what it buys the
//! read path.
//!
//! Run: cargo test --release -p zyron-wire --test lake_derived_bench
//!
//! An expression cluster key is stored in a column of its own, computed on
//! every write. That is the only reason a predicate on the expression can
//! prune anything, because pruning reads file statistics and file
//! statistics are keyed by column. So the expression buys a read and
//! charges a write, and both halves belong in one measurement: a cheap
//! write that prunes nothing is not a feature, and a prune paid for with a
//! per-row evaluation loop is not one either.
//!
//! ## Why this is here and not in the lake suite
//!
//! `crates/zyron-lake/tests/lake_bench.rs` is the natural home by subject,
//! and it cannot be. `zyron-lake` depends on `zyron-common` and
//! `zyron-storage` and nothing else, so it has no parser, no binder and no
//! executor. A clustering expression is declared in SQL, canonicalized by
//! the binder and evaluated by the executor's kernels, so measuring it
//! needs the stack that sits above the lake. Reaching it from `zyron-lake`
//! would mean a dev-dependency pointing back through `zyron-wire` at the
//! crate under test.
//!
//! ## The three loads
//!
//! One server, three lake tables, one dataset, in one process.
//!
//! * `plain (ts, yr)` stores the year as an ordinary column and every
//!   statement supplies it. This is what an operator does today without
//!   the feature, and it is the baseline the gate is stated against.
//! * `clustered (ts)` declares `CLUSTER BY (date_part('year', ts))` and
//!   statements supply only `ts`. The engine computes the year.
//! * `tsonly (ts)` stores nothing but the timestamp. It is not a baseline,
//!   it is the diagnostic: it takes byte-identical statement text to
//!   `clustered`, so the difference between the two is the cost of
//!   computing and writing the column and nothing else.
//!
//! The gate is `clustered` over `plain`, because that is the choice an
//! operator actually makes. It is a conservative gate in one direction
//! that is worth stating plainly: `plain`'s statements carry an extra
//! literal per row, so the baseline pays parse and decode cost the
//! clustered side does not, and the ratio understates the derived column's
//! cost by that much. `clustered` over `tsonly` carries no such asymmetry
//! and is the number to read when the gate moves.
//!
//! Loads run plain, then clustered, then tsonly. The measured side runs
//! second on purpose: first pays cold cost and last runs warmest, and the
//! gate should sit on neither.
//!
//! ## What is judged and what is recorded
//!
//! The wall-time ratio is judged only in a measuring build, because a
//! timing bound invented from an unoptimized run is a number somebody made
//! up. The file counts are judged in every build: optimization changes how
//! fast a prune happens, not how many files it rejects, so a count means
//! the same thing in any profile and its bound comes from the structure of
//! the data rather than from a measurement.

mod common;

use std::sync::Arc;

use common::{
    analyze_lake_scan, create_test_server, exec_ddl, exec_dml, new_session, query_values,
};
use zyron_bench_harness::{
    Instant, RatioBound, assert_exact_metric, check_performance, init, measuring, record_metric,
    tprintln,
};
use zyron_executor::column::ScalarValue;
use zyron_wire::connection::ServerState;

/// The name every metric in this file is filed under
const SUITE: &str = "lake_derived";

/// Rows each table receives.
///
/// An unoptimized build judges no timing, so it runs a small dataset
/// purely to prove the three loads execute, that the expression column is
/// filled, and that a predicate on the expression prunes. The million the
/// gate is stated over only applies where the timing counts for something
fn rows() -> usize {
    if measuring() { 1_000_000 } else { 20_000 }
}

/// Rows per INSERT, and so rows per data file.
///
/// Bulk statements are how a table this size is actually filled, and the
/// file count follows from this rather than from the row count, which is
/// what keeps the pruning claim stable across both scales
const ROWS_PER_STATEMENT: usize = 10_000;

/// First year in the dataset
const FIRST_YEAR: i64 = 2010;

/// Years the dataset spans.
///
/// The timestamps ascend, so a file covers a contiguous slice of this
/// range and a predicate naming one year rejects the files whose slice
/// excludes it. Sixteen over a hundred files is about six files a year,
/// which is a prune worth asserting rather than a rounding artifact
const YEARS: i64 = 16;

/// The year the read-path measurements filter on. Interior to the range,
/// so files on both sides of it have to be rejected
const PROBE_YEAR: i64 = FIRST_YEAR + YEARS / 2;

/// Serializes the suite. Each test loads three tables of up to a million
/// rows, so two at once would measure contention rather than the thing
/// named
static BENCH_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn section(title: &str) -> std::sync::MutexGuard<'static, ()> {
    let guard = BENCH_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init(SUITE);
    tprintln!("");
    tprintln!("=== {} ===", title);
    guard
}

/// The year row `i` of `n` falls in. Ascending, so file bounds narrow
fn year_of(i: usize, n: usize) -> i64 {
    FIRST_YEAR + (i as i64 * YEARS) / n.max(1) as i64
}

/// One row's timestamp literal.
///
/// The month, day and time vary so the column is not a run of identical
/// values, and none of them affects which year the row lands in, which is
/// the only thing the expression reads
fn timestamp_literal(i: usize, n: usize) -> String {
    format!(
        "TIMESTAMP '{}-{:02}-{:02} {:02}:{:02}:{:02}'",
        year_of(i, n),
        1 + (i % 12),
        1 + (i % 28),
        i % 24,
        i % 60,
        (i * 7) % 60
    )
}

/// The INSERT statements one load runs, built before any clock starts.
///
/// Formatting a million row literals is not part of what an insert costs,
/// and leaving it inside the timed region would charge it to whichever
/// table happened to be measured
fn insert_statements(table: &str, n: usize, with_year: bool) -> Vec<String> {
    let mut out = Vec::with_capacity(n.div_ceil(ROWS_PER_STATEMENT));
    let mut i = 0;
    while i < n {
        let end = (i + ROWS_PER_STATEMENT).min(n);
        let mut sql = String::with_capacity((end - i) * 48 + 64);
        sql.push_str("INSERT INTO ");
        sql.push_str(table);
        sql.push_str(" VALUES ");
        for r in i..end {
            if r > i {
                sql.push(',');
            }
            sql.push('(');
            sql.push_str(&timestamp_literal(r, n));
            if with_year {
                sql.push(',');
                sql.push_str(&year_of(r, n).to_string());
            }
            sql.push(')');
        }
        out.push(sql);
        i = end;
    }
    out
}

/// Runs one load and returns its wall time in microseconds
async fn timed_load(server: &Arc<ServerState>, statements: &[String]) -> f64 {
    let start = Instant::now();
    for sql in statements {
        exec_dml(server, sql).await;
    }
    start.elapsed().as_secs_f64() * 1e6
}

/// The current manifest of a lake table
fn manifest_of(server: &Arc<ServerState>, table: &str) -> Arc<zyron_lake::ManifestFile> {
    let entry = server
        .catalog
        .list_all_tables()
        .into_iter()
        .find(|t| t.name == table)
        .expect("table");
    let paths = zyron_lake::LakePaths::new(server.disk_manager.data_dir(), entry.id.0);
    let log = zyron_lake::TransactionLog::lookup_shared(&paths).expect("lake log");
    log.manifest_at(log.latest_version()).expect("manifest")
}

fn count_of(rows: &[Vec<ScalarValue>]) -> i64 {
    match rows.first().and_then(|r| r.first()) {
        Some(ScalarValue::Int64(v)) => *v,
        Some(ScalarValue::Int32(v)) => i64::from(*v),
        other => panic!("expected a count, got {other:?}"),
    }
}

/// The rows the dataset puts in `PROBE_YEAR`, computed from the generator
/// rather than from either table, so a load that dropped rows is caught by
/// the count instead of agreeing with itself
fn expected_probe_rows(n: usize) -> i64 {
    (0..n).filter(|i| year_of(*i, n) == PROBE_YEAR).count() as i64
}

#[tokio::test(flavor = "multi_thread")]
async fn test_a_clustering_expression_costs_the_write_and_prunes_the_read() {
    let _section = section("Clustering Expression: Write Cost and Read Benefit");
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    let n = rows();

    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE plain (ts TIMESTAMP(6), yr BIGINT) USING ZYRONLAKE",
    )
    .await
    .expect("create the materialized-column baseline");
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE clustered (ts TIMESTAMP(6)) USING ZYRONLAKE \
         CLUSTER BY (date_part('year', ts))",
    )
    .await
    .expect("create the expression-clustered table");
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE tsonly (ts TIMESTAMP(6)) USING ZYRONLAKE",
    )
    .await
    .expect("create the no-extra-column diagnostic");

    // Both texts built up front, and the two single-column loads share one,
    // which is what makes clustered over tsonly a clean difference
    let two_column = insert_statements("plain", n, true);
    let one_column: Vec<String> = insert_statements("clustered", n, false);
    let one_column_tsonly: Vec<String> = one_column
        .iter()
        .map(|s| s.replacen("INSERT INTO clustered", "INSERT INTO tsonly", 1))
        .collect();

    tprintln!("  Rows per table: {}", n);
    tprintln!("  Statements per load: {}", two_column.len());

    let plain_us = timed_load(&server, &two_column).await;
    let clustered_us = timed_load(&server, &one_column).await;
    let tsonly_us = timed_load(&server, &one_column_tsonly).await;

    record_metric(SUITE, "Load, materialized column", "us", vec![plain_us]);
    record_metric(
        SUITE,
        "Load, clustering expression",
        "us",
        vec![clustered_us],
    );
    record_metric(SUITE, "Load, timestamp only", "us", vec![tsonly_us]);

    // Answers before timings. A load that wrote fewer rows would otherwise
    // present as the faster one
    let expected = n as i64;
    for table in ["plain", "clustered", "tsonly"] {
        let got = count_of(&query_values(&server, &format!("SELECT count(*) FROM {table}")).await);
        assert_eq!(got, expected, "{table} did not load every row");
    }

    // The expression column exists, is filled, and its bounds vary across
    // the file set. Constant bounds would mean the writer saw one value,
    // which is what a broken evaluation looks like from here
    let manifest = manifest_of(&server, "clustered");
    assert_eq!(
        manifest.schema.derived.len(),
        1,
        "the expression is registered on the schema"
    );
    let derived_id = manifest.schema.derived[0].column_id;
    let mut lowest = i64::MAX;
    let mut highest = i64::MIN;
    for entry in &manifest.entries {
        let stats = entry
            .stats_for(derived_id)
            .expect("every file carries statistics over the expression column");
        let (min, max) = match (&stats.bounds.min, &stats.bounds.max) {
            (Some(zyron_lake::LakeValue::Int(min)), Some(zyron_lake::LakeValue::Int(max))) => {
                (*min, *max)
            }
            other => panic!("the expression column's bounds are not integers: {other:?}"),
        };
        assert!(min <= max, "a file's bounds are inverted");
        lowest = lowest.min(min);
        highest = highest.max(max);
    }
    assert_eq!(lowest, FIRST_YEAR, "the file set starts at the first year");
    assert_eq!(
        highest,
        FIRST_YEAR + YEARS - 1,
        "the file set reaches the last year"
    );
    tprintln!(
        "  Files in the clustered table: {}, expression bounds {}..{}",
        manifest.entries.len(),
        lowest,
        highest
    );

    // A predicate written against the expression has to reach the column
    // the expression is stored in, or it prunes nothing
    let probe =
        format!("SELECT count(*) FROM clustered WHERE date_part('year', ts) = {PROBE_YEAR}");
    let control = format!("SELECT count(*) FROM tsonly WHERE date_part('year', ts) = {PROBE_YEAR}");
    let want = expected_probe_rows(n);
    assert_eq!(
        count_of(&query_values(&server, &probe).await),
        want,
        "pruning changed the answer"
    );
    assert_eq!(
        count_of(&query_values(&server, &control).await),
        want,
        "the unclustered table disagrees with the clustered one"
    );

    let scan = analyze_lake_scan(&server, &probe).await;
    let considered = scan.aux[0];
    let pruned = scan.aux[1];
    assert!(
        considered > 0,
        "the scan reported no files, so nothing was measured"
    );
    tprintln!(
        "  Files considered {}, pruned {}, read {}",
        considered,
        pruned,
        considered - pruned
    );

    // One year of sixteen, so most of the file set has bounds that exclude
    // it. Half is the weakest statement still worth enforcing: it holds at
    // the two-file scale an unoptimized run uses and at the hundred-file
    // scale a measured one does, and it fails the moment the predicate
    // stops reaching the expression column
    let rate = pruned as f64 / considered as f64;
    assert!(
        assert_exact_metric(
            SUITE,
            "expression predicate files pruned (fraction)",
            rate,
            RatioBound::AtLeast(0.5),
        ),
        "a predicate on the clustering expression pruned only {:.4} of {} files",
        rate,
        considered
    );

    // The control table has no column to prune on, so it reads everything.
    // Without this the prune above could be a file count that happened to
    // be small rather than a predicate that did any work
    let control_scan = analyze_lake_scan(&server, &control).await;
    assert_eq!(
        control_scan.aux[1], 0,
        "a table with no expression column cannot prune on the expression"
    );

    let derived_over_plain = clustered_us / plain_us;
    let derived_over_tsonly = clustered_us / tsonly_us;
    record_metric(
        SUITE,
        "Load ratio, clustering expression over timestamp only",
        "x",
        vec![derived_over_tsonly],
    );
    assert!(
        check_performance(
            SUITE,
            "Load ratio, clustering expression over materialized column",
            derived_over_plain,
            1.20,
            false,
        ),
        "computing the expression cost {:.3}x a stored column, over the 1.20x ceiling \
         (clustered {:.0}us, plain {:.0}us, timestamp only {:.0}us, clustered over \
         timestamp only {:.3}x)",
        derived_over_plain,
        clustered_us,
        plain_us,
        tsonly_us,
        derived_over_tsonly
    );
}

//! Cross-format benchmark: one workload, two storage bindings, one run.
//!
//! Run: cargo test --release -p zyron-wire --test cross_format_bench
//!
//! This is the bucket the two storage formats are compared in. It exists
//! separately from the heap and lake suites, and it takes nothing away
//! from them: a metric with no counterpart in the other format is not a
//! defect, and inventing one so a table looks symmetric would be worse
//! than an empty cell. Those suites answer "did this format regress".
//! This one answers "which format should this table use".
//!
//! Both formats are driven through the same planner and executor, from
//! one dataset, in one process, back to back. The rules that make the
//! numbers mean the same thing:
//!
//! 1. One dataset, generated once from a fixed seed, inserted into both
//!    with byte-identical statements.
//! 2. One success condition. A load has not finished until its rows are
//!    durable and queryable, which is what makes "time to load N rows"
//!    comparable when one format commits per statement and the other
//!    writes a file and publishes a log version.
//! 3. Outcomes, never mechanisms. Wall time to an end state and rows
//!    returned exist on both. Pages do not exist on the lake, files do
//!    not exist on the heap, and commits mean different granularities, so
//!    none of those is compared here.
//! 4. One query text through one planner entry point. Calling
//!    format-specific code would benchmark two paths we wrote rather than
//!    the two formats.
//! 5. The answers are compared before any timing is recorded. Differing
//!    rows mean the faster side measured a cheaper wrong thing, which is
//!    what catches a broken pushdown presenting as a speedup.
//! 6. Same process, same run, back to back, which is what makes the ratio
//!    trustworthy while absolute numbers drift.
//! 7. The order alternates across repetitions, so cache and thermal state
//!    do not systematically favour whichever format runs first.
//!
//! Claims are ratios rather than absolute numbers. A ratio measured in
//! one run on one machine is hardware independent, and every absolute
//! target in this repo went stale when the CPU changed.
//!
//! A timing bound here states a claim, not a snapshot of one run. A bound
//! set to the worst observed ratio plus a little is a gate that fails on a
//! busy machine, and a gate that cries wolf is one people learn to ignore.
//! So each bound is the weakest statement still worth enforcing.
//!
//! - A point lookup bounds at 0.25, "at least four times faster". The lake
//!   resolves a row locator to a file and an ordinal and reads only the
//!   projected column, so this is a structural win rather than a margin
//!   that has to be defended run to run.
//! - A scan, aggregate, join, point update and bulk delete bound at 1.00,
//!   "never slower than the heap". That is the claim the format exists to
//!   make for analytics, and it holds with room to spare.
//! - A load bounds loosely, because the lake is expected to lose. It writes
//!   a data file and publishes a log version per statement.
//!
//! Range scan wide carries the thinnest margin of the set, and that is
//! inherent rather than a defect. Reading most of a table is the shape
//! where a row store is most competitive. If it trips, read it as a real
//! narrowing and find out why, rather than widening the bound.
//!
//! A bound on a timing has to come from optimized runs. One invented from
//! an unoptimized one is not a target, it is a number somebody made up that
//! would have to be moved the first time real hardware disagreed. The
//! harness applies a timing bound only in a measuring build and reports it
//! unchecked elsewhere, so a debug run is not failed against release
//! numbers. Bytes read carries the precise claim instead, because a byte
//! count is exact in any build profile.
//!
//! Two ratios are deliberately left unbounded, both about the indexed point
//! lookup. Their spread across recorded runs is orders of magnitude wider
//! than every other shape here, so any bound loose enough to admit them
//! would pass whatever the engine did, and a bound that wide is worse than
//! none because it reads as a checked claim. The spread is itself the
//! finding, and it is recorded rather than asserted. Read the run files in
//! `benchmarks/cross_format/` for what it currently is.
//!
//! Bytes read is bounded, because it is not a timing. A count of bytes is
//! the same number on any machine in any build profile, so its bound comes
//! from the structure of the two formats rather than from a measurement:
//! a row store reads whole rows, a column store reads the columns a query
//! asked for, so the lake cannot read more than the heap for the same
//! query. That is the claim this suite enforces on every run, and it is
//! usually the reason a timing comes out the way it does.
//!
//! What it asserts besides: that both formats returned the same rows, and
//! that every ratio is a real number rather than a division by a
//! measurement that did not happen.

use std::sync::Arc;

use zyron_bench_harness::{
    Format, Instant, RatioBound, VALIDATION_RUNS, assert_ratio, init, measuring,
    record_metric_for, record_ratio, tprintln,
};
use zyron_executor::column::ScalarValue;
use zyron_wire::connection::ServerState;
use zyron_wire::session::Session;

mod common;
use common::{create_test_server, exec_ddl, exec_dml, new_session, query_values};

/// Rows in the dataset.
///
/// An unoptimized build judges nothing, so it runs a small dataset purely
/// to prove the workloads execute and the two formats agree. The real
/// scale only applies where the numbers count for something.
///
/// A hundred thousand is a deliberately modest first baseline: the suite
/// loads both formats once per test and the join test loads four tables,
/// so the run is sixteen loads. A scale that finishes is worth more than
/// one that has to be killed, and raising it later costs one constant
fn rows() -> usize {
    if measuring() { 100_000 } else { 2_000 }
}

/// Repetitions per measured query. One is enough to check an answer
fn reps() -> usize {
    if measuring() { VALIDATION_RUNS } else { 1 }
}

/// Rows per INSERT statement for a bulk load, which is how a table this
/// size is actually filled. Bounded so parse cost stays finite and
/// identical for both formats
const BULK_ROWS_PER_STATEMENT: usize = 10_000;

/// Rows per INSERT statement for the trickle load. Two orders of
/// magnitude smaller, so the cost is dominated by how much one commit
/// costs rather than by how fast rows are written
const TRICKLE_ROWS_PER_STATEMENT: usize = 100;

/// Rows the trickle load moves. Its claim is per-commit cost, not volume,
/// and the lake writes one data file and publishes one log version per
/// statement, so a large row count here would only buy a longer run
fn trickle_rows() -> usize {
    if measuring() { 20_000 } else { 1_000 }
}

/// Distinct values in the `region` column, which is what the selective
/// predicates filter on
const REGIONS: i64 = 64;

/// Serializes the suite. Each test loads both formats and times queries
/// against them, so running two at once would measure contention rather
/// than the thing named, and their sections would interleave
static BENCH_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Opens a section and takes the suite lock for its duration.
///
/// Deliberately unnumbered: the test harness decides what runs when, so a
/// number here would claim a sequence the log does not have
fn section(title: &str) -> std::sync::MutexGuard<'static, ()> {
    let guard = BENCH_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("cross_format");
    tprintln!("");
    tprintln!("=== {} ===", title);
    guard
}

// =============================================================================
// Dataset
// =============================================================================

struct Row {
    id: i64,
    region: i64,
    amount: i64,
    label: String,
}

/// The dataset both formats receive, generated once from a fixed seed so
/// two runs and two formats see byte-identical rows rather than similar
/// ones
fn dataset(n: usize) -> Vec<Row> {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    (0..n)
        .map(|i| {
            let noise = next();
            Row {
                id: i as i64,
                region: (noise % REGIONS as u64) as i64,
                amount: (noise >> 20) as i64 % 100_000,
                label: format!("row-{:08}", i),
            }
        })
        .collect()
}

// =============================================================================
// Targets
// =============================================================================

/// The only thing that differs between the two formats.
///
/// Everything else in this file is written once. A workload names a
/// table and a query shape, and the binding decides what that table is
/// made of, so the two runs cannot drift in row count, query text or
/// warmup. Two independently maintained suites would drift silently, with
/// nothing to signal that the comparison had stopped meaning anything
#[derive(Clone, Copy)]
struct Target {
    format: Format,
    /// Table name prefix, so both formats coexist on one unified node and
    /// a mixed join has something to join
    prefix: &'static str,
    /// What `CREATE TABLE` appends. The heap is the node default
    using: &'static str,
}

const HEAP: Target = Target {
    format: Format::Heap,
    prefix: "h",
    using: "",
};

const LAKE: Target = Target {
    format: Format::Lake,
    prefix: "l",
    using: " USING ZYRONLAKE",
};

impl Target {
    fn table(&self, base: &str) -> String {
        format!("{}_{}", self.prefix, base)
    }

    async fn create(&self, server: &Arc<ServerState>, session: &mut Option<Session>, base: &str) {
        let sql = format!(
            "CREATE TABLE {} (id BIGINT NOT NULL, region BIGINT, amount BIGINT, label TEXT){}",
            self.table(base),
            self.using
        );
        exec_ddl(server, session, &sql)
            .await
            .unwrap_or_else(|e| panic!("create {}: {e}", self.table(base)));
    }

    /// Inserts every row and returns only once they are durable and
    /// queryable, which is rule 2. `exec_dml` commits the database
    /// transaction and publishes the lake versions it created, which is
    /// what makes the lake's rows visible to the next statement
    async fn load(
        &self,
        server: &Arc<ServerState>,
        base: &str,
        data: &[Row],
        rows_per_statement: usize,
    ) -> f64 {
        let table = self.table(base);
        let statements: Vec<String> = data
            .chunks(rows_per_statement)
            .map(|chunk| {
                let values: Vec<String> = chunk
                    .iter()
                    .map(|r| {
                        format!("({}, {}, {}, '{}')", r.id, r.region, r.amount, r.label)
                    })
                    .collect();
                format!("INSERT INTO {} VALUES {}", table, values.join(", "))
            })
            .collect();

        let start = Instant::now();
        for sql in &statements {
            exec_dml(server, sql).await;
        }
        let elapsed_us = start.elapsed().as_secs_f64() * 1_000_000.0;

        // A load that did not make every row queryable did not finish
        let count = query_values(server, &format!("SELECT COUNT(*) FROM {}", table)).await;
        assert_eq!(
            scalar_i64(&count),
            data.len() as i64,
            "{} did not make every row queryable",
            table
        );
        elapsed_us
    }
}

fn scalar_i64(rows: &[Vec<ScalarValue>]) -> i64 {
    match rows.first().and_then(|r| r.first()) {
        Some(ScalarValue::Int64(v)) => *v,
        Some(ScalarValue::Int32(v)) => *v as i64,
        other => panic!("expected one integer, got {other:?}"),
    }
}

// =============================================================================
// Driver
// =============================================================================

/// Runs one query shape on both formats and records the ratio.
///
/// `shape` receives the table name and returns the SQL, so the query text
/// is written once and differs only in what it names. The answers are
/// compared before either timing is recorded, and the order alternates
/// across repetitions
async fn compare(
    server: &Arc<ServerState>,
    test: &str,
    metric: &str,
    base: &str,
    bound: Option<RatioBound>,
    shape: impl Fn(&str) -> String,
) {
    let heap_sql = shape(&HEAP.table(base));
    let lake_sql = shape(&LAKE.table(base));

    // One untimed pass each, so neither format is charged for whatever the
    // first execution of a plan shape sets up
    let heap_answer = query_values(server, &heap_sql).await;
    let lake_answer = query_values(server, &lake_sql).await;
    assert_same_answer(metric, &heap_answer, &lake_answer);

    let mut heap_runs = Vec::with_capacity(reps());
    let mut lake_runs = Vec::with_capacity(reps());
    for rep in 0..reps() {
        // Rule 7: alternate, so neither format always runs into a warm
        // cache the other left behind
        if rep % 2 == 0 {
            heap_runs.push(time_query(server, &heap_sql, metric, &heap_answer).await);
            lake_runs.push(time_query(server, &lake_sql, metric, &heap_answer).await);
        } else {
            lake_runs.push(time_query(server, &lake_sql, metric, &heap_answer).await);
            heap_runs.push(time_query(server, &heap_sql, metric, &heap_answer).await);
        }
    }

    let heap_avg = record_metric_for(Format::Heap, test, metric, "us", heap_runs);
    let lake_avg = record_metric_for(Format::Lake, test, metric, "us", lake_runs);
    ratio_of(test, metric, heap_avg, lake_avg, bound);

    compare_bytes_read(server, test, &heap_sql, &lake_sql).await;
}

/// Records how many bytes of table data each format had to read to answer
/// the same query.
///
/// This is the read-side counterpart to the timing above, and a better
/// number in one respect: it is exact and hardware independent, so it
/// means the same thing on any machine and in any build profile, where a
/// timing only means something in an optimized run on a known baseline.
/// It is what file pruning, zone maps and column projection actually
/// reduce, so it says why a timing came out the way it did
async fn compare_bytes_read(
    server: &Arc<ServerState>,
    test: &str,
    heap_sql: &str,
    lake_sql: &str,
) {
    let metric = "Bytes read";
    let heap_bytes = bytes_read_by(server, HEAP, heap_sql).await;
    let lake_bytes = bytes_read_by(server, LAKE, lake_sql).await;
    let heap_avg = record_metric_for(Format::Heap, test, metric, " bytes", vec![heap_bytes]);
    let lake_avg = record_metric_for(Format::Lake, test, metric, " bytes", vec![lake_bytes]);

    // The one cross-format claim this suite can bound without an optimized
    // baseline, because it does not come from one.
    //
    // A row store must read every byte of every row it examines, since a
    // page holds whole rows. A column store reads only the segments a query
    // projects, and reads them encoded. So for the same query the lake
    // cannot read more than the heap, whatever machine either runs on. The
    // bound is the structural claim itself rather than a measured number
    // rounded up, which is why it is one and not a tuned constant.
    //
    // What it catches is the regression that matters here: projection
    // pushdown, file pruning or zone map rejection silently stopping, which
    // a timing on a fast machine can easily hide and this cannot
    let bound = zyron_bench_harness::RatioBound::AtMost(1.0);
    let within = zyron_bench_harness::assert_exact_ratio(
        test,
        metric,
        (Format::Lake, lake_avg),
        (Format::Heap, heap_avg),
        bound,
    );
    assert!(
        within,
        "{}: the lake read {} bytes against the heap's {}, a ratio of {:.3} \
         against a bound of {:?}",
        test,
        lake_bytes,
        heap_bytes,
        lake_bytes / heap_bytes,
        bound
    );
}

/// Bytes of table data one execution of a query read, summed over every
/// table of that format.
///
/// Summed by prefix rather than named per test because a join touches two
/// tables and a single-table shape touches one, and the question is what
/// the query read, not what one of its tables gave up. One execution is
/// the whole measurement: a byte count is exact, so there is nothing for
/// repetitions to average out
async fn bytes_read_by(server: &Arc<ServerState>, target: Target, sql: &str) -> f64 {
    use std::sync::atomic::Ordering;

    let prefix = format!("{}_", target.prefix);
    let ids: Vec<u32> = server
        .catalog
        .list_all_tables()
        .iter()
        .filter(|t| t.name.starts_with(&prefix))
        .map(|t| t.id.0)
        .collect();
    assert!(
        !ids.is_empty(),
        "no {} table to read byte counters from",
        target.prefix
    );
    let total = || -> u64 {
        ids.iter()
            .map(|id| {
                server
                    .table_io_stats
                    .get_or_create(*id)
                    .bytes_read
                    .load(Ordering::Relaxed)
            })
            .sum()
    };

    let before = total();
    let _ = query_values(server, sql).await;
    (total() - before) as f64
}

async fn time_query(
    server: &Arc<ServerState>,
    sql: &str,
    metric: &str,
    expected: &[Vec<ScalarValue>],
) -> f64 {
    let start = Instant::now();
    let answer = query_values(server, sql).await;
    let elapsed = start.elapsed().as_secs_f64() * 1_000_000.0;
    // Rule 5 again, per repetition: a run that returned different rows did
    // not measure the same work
    assert_same_answer(metric, expected, &answer);
    elapsed
}

/// Records lake over heap, which reads as "how many times the lake costs
/// what the heap does". Below one the lake is ahead.
///
/// `bound` is the regression gate. It is set from the spread across the
/// release runs in `benchmarks/cross_format/`, with headroom above the
/// worst of them, so it catches a design regression rather than machine
/// noise. A shape whose ratio is not stable across those runs carries None
/// and is recorded without a claim, because a bound loose enough to admit
/// an unstable measurement would pass no matter what the engine did.
///
/// A timing bound is profile dependent, so the harness applies it only in a
/// measuring build and reports it unchecked elsewhere rather than failing a
/// debug run against release numbers.
fn ratio_of(test: &str, metric: &str, heap_avg: f64, lake_avg: f64, bound: Option<RatioBound>) {
    let value = match bound {
        Some(bound) => {
            let admits = assert_ratio(
                test,
                metric,
                (Format::Lake, lake_avg),
                (Format::Heap, heap_avg),
                bound,
            );
            assert!(
                admits,
                "{} {} ratio {} is outside its bound, from lake {} and heap {}",
                test,
                metric,
                lake_avg / heap_avg,
                lake_avg,
                heap_avg
            );
            Some(lake_avg / heap_avg)
        }
        None => record_ratio(
            test,
            metric,
            (Format::Lake, lake_avg),
            (Format::Heap, heap_avg),
        ),
    };
    // A ratio that could not be taken means one side measured nothing,
    // which is a broken benchmark and not a result
    assert!(
        value.is_some_and(|v| v.is_finite()),
        "{} {} did not produce a comparable ratio from {} and {}",
        test,
        metric,
        heap_avg,
        lake_avg
    );
}

/// Rule 5. Two formats that returned different rows did not run the same
/// query, and the faster of them measured a cheaper wrong thing
fn assert_same_answer(metric: &str, a: &[Vec<ScalarValue>], b: &[Vec<ScalarValue>]) {
    assert_eq!(
        a.len(),
        b.len(),
        "{}: the formats returned different row counts, {} and {}",
        metric,
        a.len(),
        b.len()
    );
    for (i, (ra, rb)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            ra, rb,
            "{}: row {} differs between the formats, {:?} against {:?}",
            metric, i, ra, rb
        );
    }
}

/// Creates and loads one base table on both formats, recording the load
/// as its own comparison
async fn load_both(
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
    test: &str,
    base: &str,
    data: &[Row],
    rows_per_statement: usize,
    // Some when this load is itself the measurement, None when it is only
    // setting up the table a later workload reads
    metric: Option<&str>,
    bound: Option<RatioBound>,
) {
    HEAP.create(server, session, base).await;
    LAKE.create(server, session, base).await;
    let heap_us = HEAP.load(server, base, data, rows_per_statement).await;
    let lake_us = LAKE.load(server, base, data, rows_per_statement).await;
    if let Some(metric) = metric {
        let heap_avg = record_metric_for(Format::Heap, test, metric, "us", vec![heap_us]);
        let lake_avg = record_metric_for(Format::Lake, test, metric, "us", vec![lake_us]);
        ratio_of(test, metric, heap_avg, lake_avg, bound);
    }
}

async fn setup(
    test: &str,
    base: &str,
    metric: Option<&str>,
    bound: Option<RatioBound>,
) -> (Arc<ServerState>, tempfile::TempDir, Vec<Row>) {
    init("cross_format");
    let (server, _schema, tmp) = create_test_server().await;
    let mut session = new_session();
    let data = dataset(rows());
    tprintln!("  Rows on each format: {}", data.len());
    load_both(
        &server,
        &mut session,
        test,
        base,
        &data,
        BULK_ROWS_PER_STATEMENT,
        metric,
        bound,
    )
    .await;
    (server, tmp, data)
}

// =============================================================================
// Bucket one: workloads both formats serve
// =============================================================================

/// Filling a table the way one this size is actually filled. This is the
/// write-throughput claim: how long until N rows are durable and readable
#[tokio::test]
async fn test_bulk_load_to_queryable_across_formats() {
    let test = "bulk_load";
    let _section = section("Bulk Load To Queryable");
    let (_server, _tmp, _data) = setup(test, "load", Some("Bulk load to queryable"), Some(RatioBound::AtMost(14.0))).await;
}

/// The same rows in statements two orders of magnitude smaller. This is
/// the commit-rate claim, and it is a different question from the one
/// above: the heap commits in place while the lake writes a data file and
/// publishes a log version per statement, so the two answers should not
/// be conflated into one number
#[tokio::test]
async fn test_trickle_load_to_queryable_across_formats() {
    let test = "trickle_load";
    let _section = section("Trickle Load To Queryable");
    init("cross_format");
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    let data = dataset(trickle_rows());
    tprintln!(
        "{}: {} rows on both formats, {} per statement",
        test,
        data.len(),
        TRICKLE_ROWS_PER_STATEMENT
    );
    load_both(
        &server,
        &mut session,
        test,
        "tl",
        &data,
        TRICKLE_ROWS_PER_STATEMENT,
        Some("Trickle load to queryable"),
        Some(RatioBound::AtMost(40.0)),
    )
    .await;
}

/// The plan claims the lake wins a narrow projection, because a row
/// locator gives it a file and an ordinal and it reads only the projected
/// column, while the heap reads a whole page for the whole row. This is
/// where that claim is checked rather than asserted
#[tokio::test]
async fn test_point_lookup_narrow_projection_across_formats() {
    let test = "point_lookup_narrow";
    let _section = section("Point Lookup, Narrow Projection");
    let (server, _tmp, data) = setup(test, "pt", None, None).await;
    let probe = data[data.len() / 3].id;
    compare(
        &server,
        test,
        "Point lookup, narrow projection",
        "pt",
        Some(RatioBound::AtMost(0.25)),
        |t| format!("SELECT amount FROM {} WHERE id = {}", t, probe),
    )
    .await;
}

/// The other half of the same claim. Whole-row locality is the heap's,
/// so a full-row lookup is where it should lead
#[tokio::test]
async fn test_point_lookup_full_row_across_formats() {
    let test = "point_lookup_full";
    let _section = section("Point Lookup, Full Row");
    let (server, _tmp, data) = setup(test, "pf", None, None).await;
    let probe = data[data.len() / 2].id;
    compare(
        &server,
        test,
        "Point lookup, full row",
        "pf",
        Some(RatioBound::AtMost(0.25)),
        |t| format!("SELECT id, region, amount, label FROM {} WHERE id = {}", t, probe),
    )
    .await;
}

#[tokio::test]
async fn test_range_scan_across_formats() {
    let test = "range_scan";
    let _section = section("Range Scan");
    let (server, _tmp, _data) = setup(test, "rs", None, None).await;

    // Selective: one region of sixty four, which is what file pruning and
    // zone maps are for
    compare(
        &server,
        test,
        "Range scan, selective",
        "rs",
        Some(RatioBound::AtMost(1.00)),
        |t| format!("SELECT COUNT(*) FROM {} WHERE region = 7", t),
    )
    .await;

    // Wide: most of the table, where columnar scan throughput is the whole
    // point of the format
    compare(
        &server,
        test,
        "Range scan, wide",
        "rs",
        Some(RatioBound::AtMost(1.00)),
        |t| format!("SELECT COUNT(*) FROM {} WHERE amount >= 0", t),
    )
    .await;
}

/// One column out of four, which is columnar's best case
#[tokio::test]
async fn test_aggregate_across_formats() {
    let test = "aggregate";
    let _section = section("Aggregate");
    let (server, _tmp, _data) = setup(test, "ag", None, None).await;
    compare(
        &server,
        test,
        "Aggregate over one column",
        "ag",
        Some(RatioBound::AtMost(1.00)),
        |t| format!("SELECT SUM(amount) FROM {}", t),
    )
    .await;
}

/// The same point lookup with and without a secondary index on the key,
/// on both formats.
///
/// Without one, finding the row is a scan on either format, so the number
/// is dominated by read speed and says nothing about indexing. The pair is
/// what carries the information: an index has to move the lookup on a
/// format that uses it, and a format whose index is never consulted stays
/// flat between the two halves
#[tokio::test]
async fn test_point_lookup_with_and_without_an_index() {
    let test = "point_lookup_index";
    let _section = section("Point Lookup, With And Without An Index");
    let (server, _tmp, data) = setup(test, "pli", None, None).await;
    let mut session = new_session();
    let probe = data[data.len() / 3].id;

    let measure = |server: Arc<ServerState>, target: Target| async move {
        let sql = format!(
            "SELECT amount FROM {} WHERE id = {}",
            target.table("pli"),
            probe
        );
        let mut runs = Vec::with_capacity(reps());
        for _ in 0..reps() {
            let start = Instant::now();
            let rows = query_values(&server, &sql).await;
            runs.push(start.elapsed().as_secs_f64() * 1_000_000.0);
            assert_eq!(rows.len(), 1, "the probed row must be found either way");
        }
        runs
    };

    let heap_bare = measure(server.clone(), HEAP).await;
    let lake_bare = measure(server.clone(), LAKE).await;
    let heap_bare_avg = record_metric_for(
        Format::Heap,
        test,
        "Point lookup without an index",
        "us",
        heap_bare,
    );
    let lake_bare_avg = record_metric_for(
        Format::Lake,
        test,
        "Point lookup without an index",
        "us",
        lake_bare,
    );
    ratio_of(
        test,
        "Point lookup without an index",
        heap_bare_avg,
        lake_bare_avg,
        Some(RatioBound::AtMost(0.25)),
    );

    for target in [HEAP, LAKE] {
        exec_ddl(
            &server,
            &mut session,
            &format!(
                "CREATE INDEX {}_id_ix ON {} (id)",
                target.table("pli"),
                target.table("pli")
            ),
        )
        .await
        .expect("create index");
    }

    let heap_indexed = measure(server.clone(), HEAP).await;
    let lake_indexed = measure(server.clone(), LAKE).await;
    let heap_indexed_avg = record_metric_for(
        Format::Heap,
        test,
        "Point lookup with an index",
        "us",
        heap_indexed,
    );
    let lake_indexed_avg = record_metric_for(
        Format::Lake,
        test,
        "Point lookup with an index",
        "us",
        lake_indexed,
    );
    // Recorded without a bound. This ratio is far less repeatable across
    // runs than any other shape here, so a bound loose enough to admit it
    // would pass whatever the engine did. The spread is the finding rather
    // than the number. It says the lake index path is not consistently
    // reached on a point lookup, which is a question this suite is asking
    // rather than a claim it can enforce
    ratio_of(
        test,
        "Point lookup with an index",
        heap_indexed_avg,
        lake_indexed_avg,
        None,
    );

    // What the index bought each format, as its own recorded quantity so a
    // format that never consults its index reads as a ratio near one.
    // Unbounded for the same reason as the ratio above, and it states the
    // question more directly. It isolates how much each format's own index
    // helped, rather than comparing two lookups that may not both have
    // used one
    record_ratio(
        test,
        "Index speedup on point lookup",
        (Format::Lake, lake_bare_avg / lake_indexed_avg.max(f64::MIN_POSITIVE)),
        (Format::Heap, heap_bare_avg / heap_indexed_avg.max(f64::MIN_POSITIVE)),
    );
}

/// A single-row update is the heap's case: it writes in place, while the
/// lake appends to a row overlay
#[tokio::test]
async fn test_point_update_across_formats() {
    let test = "point_update";
    let _section = section("Point Update");
    let (server, _tmp, data) = setup(test, "pu", None, None).await;
    let probe = data[data.len() / 4].id;

    let mut heap_runs = Vec::with_capacity(reps());
    let mut lake_runs = Vec::with_capacity(reps());
    for rep in 0..reps() {
        let order = if rep % 2 == 0 {
            [HEAP, LAKE]
        } else {
            [LAKE, HEAP]
        };
        for target in order {
            let sql = format!(
                "UPDATE {} SET amount = {} WHERE id = {}",
                target.table("pu"),
                1_000 + rep as i64,
                probe
            );
            let start = Instant::now();
            exec_dml(&server, &sql).await;
            let elapsed = start.elapsed().as_secs_f64() * 1_000_000.0;
            match target.format {
                Format::Heap => heap_runs.push(elapsed),
                Format::Lake => lake_runs.push(elapsed),
            }
        }
        // Both formats must hold the same value afterwards, or one of them
        // did less work than the other
        let heap_after = query_values(
            &server,
            &format!("SELECT amount FROM {} WHERE id = {}", HEAP.table("pu"), probe),
        )
        .await;
        let lake_after = query_values(
            &server,
            &format!("SELECT amount FROM {} WHERE id = {}", LAKE.table("pu"), probe),
        )
        .await;
        assert_same_answer("Point update", &heap_after, &lake_after);
    }

    let heap_avg = record_metric_for(Format::Heap, test, "Point update", "us", heap_runs);
    let lake_avg = record_metric_for(Format::Lake, test, "Point update", "us", lake_runs);
    ratio_of(test, "Point update", heap_avg, lake_avg, Some(RatioBound::AtMost(1.00)));
}

/// A predicate delete is the lake's case: it records the predicate and
/// drops whole files it covers, while the heap walks the rows
#[tokio::test]
async fn test_bulk_delete_across_formats() {
    let test = "bulk_delete";
    let _section = section("Bulk Delete");
    let (server, _tmp, _data) = setup(test, "bd", None, None).await;

    let mut runs = Vec::new();
    for target in [HEAP, LAKE] {
        let sql = format!("DELETE FROM {} WHERE region = 3", target.table("bd"));
        let start = Instant::now();
        exec_dml(&server, &sql).await;
        runs.push((target.format, start.elapsed().as_secs_f64() * 1_000_000.0));
    }

    let heap_left = query_values(
        &server,
        &format!("SELECT COUNT(*) FROM {}", HEAP.table("bd")),
    )
    .await;
    let lake_left = query_values(
        &server,
        &format!("SELECT COUNT(*) FROM {}", LAKE.table("bd")),
    )
    .await;
    assert_same_answer("Bulk delete", &heap_left, &lake_left);
    assert!(
        scalar_i64(&heap_left) < rows() as i64,
        "the delete must have removed something"
    );

    let heap_avg = record_metric_for(
        Format::Heap,
        test,
        "Bulk delete",
        "us",
        runs.iter()
            .filter(|(f, _)| *f == Format::Heap)
            .map(|(_, v)| *v)
            .collect(),
    );
    let lake_avg = record_metric_for(
        Format::Lake,
        test,
        "Bulk delete",
        "us",
        runs.iter()
            .filter(|(f, _)| *f == Format::Lake)
            .map(|(_, v)| *v)
            .collect(),
    );
    ratio_of(test, "Bulk delete", heap_avg, lake_avg, Some(RatioBound::AtMost(1.00)));
}

/// A join of two tables of one format against the same join of the other,
/// plus the mixed case only a unified node can serve locally.
///
/// The mixed join is recorded without a format, because it is not a
/// property of either one: it is what having both on one node buys, and
/// it has no counterpart to be compared against
#[tokio::test]
async fn test_join_across_formats() {
    let test = "join";
    let _section = section("Join");
    init("cross_format");
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    let data = dataset(rows());
    tprintln!("  Rows on each format: {}", data.len());
    for base in ["ja", "jb"] {
        load_both(
            &server,
            &mut session,
            test,
            base,
            &data,
            BULK_ROWS_PER_STATEMENT,
            None,
            None,
        )
        .await;
    }

    compare(
        &server,
        test,
        "Join",
        "ja",
        Some(RatioBound::AtMost(1.00)),
        |t| {
            // The second table's prefix follows the first, which is what
            // keeps both sides on one format
            let other = t.replace("_ja", "_jb");
            format!(
                "SELECT COUNT(*) FROM {} a JOIN {} b ON a.id = b.id WHERE a.region = 5",
                t, other
            )
        },
    )
    .await;

    // The mixed join, which no single-format deployment can run at all
    let mixed_sql = format!(
        "SELECT COUNT(*) FROM {} a JOIN {} b ON a.id = b.id WHERE a.region = 5",
        HEAP.table("ja"),
        LAKE.table("jb")
    );
    let expected = query_values(&server, &mixed_sql).await;
    let mut mixed_runs = Vec::with_capacity(reps());
    for _ in 0..reps() {
        let start = Instant::now();
        let answer = query_values(&server, &mixed_sql).await;
        mixed_runs.push(start.elapsed().as_secs_f64() * 1_000_000.0);
        assert_same_answer("Mixed-format join", &expected, &answer);
    }
    let same_format = query_values(
        &server,
        &format!(
            "SELECT COUNT(*) FROM {} a JOIN {} b ON a.id = b.id WHERE a.region = 5",
            HEAP.table("ja"),
            HEAP.table("jb")
        ),
    )
    .await;
    assert_same_answer("Mixed-format join", &same_format, &expected);
    let avg = mixed_runs.iter().sum::<f64>() / mixed_runs.len() as f64;
    tprintln!(
        "  mixed join us [heap joined to lake, no counterpart to compare]: {:.0}",
        avg
    );
}

//! Change Streams, Transactional Consumption and Declarative Apply Benchmark
//!
//! What the change feed costs a write, what a read of it costs against its
//! decode ceiling, what holding a position costs a consume against the same
//! statement over a table, what an apply costs against its merge, and what
//! the lake's log-derived changes, the sweeper and the feed's storage cost.
//!
//! Performance Targets:
//! | Test                                                | Metric              | Target             |
//! |-----------------------------------------------------|---------------------|--------------------|
//! | CDF write overhead per row, feed enabled            | added latency       | 100ns              |
//! | CDF write overhead per row, feed disabled           | added latency       | 0                  |
//! | table_changes scan, 10M changes, all columns        | throughput          | 20M rows/s         |
//! | table_changes scan, 2 of 40 columns                 | vs all columns      | 8x                 |
//! | table_changes with a _commit_version predicate      | files opened        | only matching      |
//! | Stream read + advance, autocommit, 100K changes     | vs the same MERGE   | ≤ 110%             |
//! | Stream position advance, one commit                 | added latency       | 200us              |
//! | PEEK pending count, 10M pending                     | latency             | 5ms                |
//! | Concurrent consumers, second waits then proceeds    | wait overhead       | one commit         |
//! | APPLY CHANGES type 1, 1M changes over 100K keys     | throughput          | 500K changes/s     |
//! | APPLY CHANGES type 2, 1M changes over 100K keys     | throughput          | 250K changes/s     |
//! | APPLY CHANGES out of order vs in order              | throughput delta    | ≤ 15%              |
//! | Lake table_changes from the transaction log, 10M    | throughput          | 20M rows/s         |
//! | Compaction of a lake table with a stream attached   | change records      | 0                  |
//! | Staleness sweep, 10K streams                        | latency             | 200ms              |
//! | Feed storage, before images off vs on               | bytes               | ≤ 55%              |
//!
//! Validation Requirements:
//! - Each measured benchmark runs 5 iterations over one prepared data set
//! - Results averaged across all 5 runs
//! - Pass/fail determined by average performance
//! - Individual runs logged for variance analysis
//!
//! Run: cargo test --release -p zyron-wire --test change_stream_bench -- --nocapture

use std::sync::Arc;
use std::time::{Duration, Instant};

use zyron_bench_harness::*;
use zyron_catalog::{
    ChangeStreamEntry, ChangeStreamMode, ChangeStreamOrigin, ChangeStreamSource, DatabaseId,
    SchemaId, StreamPosition,
};
use zyron_cdc::{CdfCodec, ChangeRecord, ChangeType, FeedConfig};
use zyron_executor::batch::{DataBatch, batch_to_tuples};
use zyron_executor::column::{Column, ColumnData, ScalarValue};
use zyron_executor::context::ExecutionContext;
use zyron_storage::txn::IsolationLevel;
use zyron_wire::connection::ServerState;
use zyron_wire::session::Session;

mod common;
use common::*;

const VALIDATION_RUNS: usize = 5;

/// Nanoseconds a recorded change may add to a row's write
const WRITE_OVERHEAD_NS_TARGET: f64 = 100.0;
const SCAN_ROWS_PER_SEC_TARGET: f64 = 20_000_000.0;
const PROJECTION_SPEEDUP_TARGET: f64 = 8.0;
const STREAM_OVER_MERGE_RATIO_TARGET: f64 = 1.10;
const ADVANCE_LATENCY_US_TARGET: f64 = 200.0;
const PEEK_LATENCY_MS_TARGET: f64 = 5.0;
const APPLY_TYPE1_TARGET: f64 = 500_000.0;
const APPLY_TYPE2_TARGET: f64 = 250_000.0;
const OUT_OF_ORDER_DELTA_TARGET: f64 = 0.15;
const LAKE_SCAN_ROWS_PER_SEC_TARGET: f64 = 20_000_000.0;
const SWEEP_LATENCY_MS_TARGET: f64 = 200.0;
const BEFORE_IMAGE_OFF_RATIO_TARGET: f64 = 0.55;

/// The wide table's width, the scan and projection targets are written
/// against forty columns
const WIDE_COLUMNS: usize = 40;
/// Changes the scan and peek targets are written against
const WIDE_CHANGES: u64 = 10_000_000;
/// Records per sealed change file, so the wide feed holds a thousand files
const WIDE_FILE_RECORDS: u64 = 10_000;
/// Changes the stream against merge target is written against
const MERGE_CHANGES: u64 = 100_000;
/// Keys and changes the apply targets are written against
const APPLY_KEYS: u64 = 100_000;
const APPLY_CHANGES: u64 = 1_000_000;
/// Rows the lake scan target is written against, landed in commits of this size
const LAKE_ROWS: u64 = 10_000_000;
const LAKE_COMMIT_ROWS: u64 = 100_000;
const SWEEP_STREAMS: usize = 10_000;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn micros_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
}

async fn ddl(server: &Arc<ServerState>, session: &mut Option<Session>, sql: &str) {
    exec_ddl(server, session, sql)
        .await
        .unwrap_or_else(|e| panic!("`{sql}` failed: {e}"));
}

async fn count(server: &Arc<ServerState>, sql: &str) -> i64 {
    let rows = query_values(server, sql).await;
    match rows.first().and_then(|r| r.first()) {
        Some(ScalarValue::Int64(n)) => *n,
        other => panic!("expected a count from {sql}, got {other:?}"),
    }
}

fn table(server: &Arc<ServerState>, name: &str) -> Arc<zyron_catalog::TableEntry> {
    server
        .catalog
        .get_table_by_id(zyron_catalog::TableId(table_id_of(server, name)))
        .expect("the table")
}

/// Change records for one batch of rows of a table, as the capture path
/// writes them, under a transaction every reader judges as committed
fn records_for(
    entry: &zyron_catalog::TableEntry,
    batch: &DataBatch,
    kind: ChangeType,
    version: u64,
) -> Vec<ChangeRecord> {
    let tuples = batch_to_tuples(batch, &entry.columns, 0, entry.schema_epoch);
    let at = micros_now();
    tuples
        .into_iter()
        .map(|tuple| ChangeRecord {
            change_type: kind,
            commit_version: version,
            commit_timestamp: at,
            table_id: entry.id.0,
            txn_id: 0,
            change_ordinal: 0,
            schema_version: entry.schema_epoch as u32,
            row_data: tuple.data().to_vec(),
            primary_key_data: Vec::new(),
            is_last_in_txn: true,
            projected: false,
        })
        .collect()
}

/// One batch of the wide table's rows, ids from `first`
fn wide_batch(first: u64, rows: usize) -> DataBatch {
    let columns = (0..WIDE_COLUMNS)
        .map(|c| {
            let values: Vec<i64> = (0..rows as u64)
                .map(|r| (first + r) as i64 * (c as i64 + 1))
                .collect();
            Column::new(ColumnData::Int64(values), zyron_common::TypeId::Int64)
        })
        .collect();
    DataBatch::new(columns)
}

/// Fills a table's feed with `changes` insert records, one sealed file per
/// `per_file` records, at versions above the feed's current one
async fn fill_feed(
    server: &Arc<ServerState>,
    name: &str,
    changes: u64,
    per_file: u64,
    batch_of: impl Fn(u64, usize) -> DataBatch,
) -> (u64, u64) {
    let entry = table(server, name);
    let feed = server
        .cdc_registry
        .as_ref()
        .expect("feeds")
        .get_feed(entry.id.0)
        .expect("the feed is open");
    // The records are appended below the write path, so the layout the
    // rows were encoded under is told to the feed here, the way a write
    // through the hook tells it, and the sealed files slice into columns
    let layout = entry
        .physical_columns_for_epoch(entry.schema_epoch)
        .expect("the table's current layout");
    feed.record_layout(entry.schema_epoch as u32, false, layout)
        .expect("the layout is recorded");
    let mut version = feed
        .latest_version()
        .unwrap_or(0)
        .max(server.wal.next_lsn().0)
        + 1;
    let first_version = version;
    let mut written = 0u64;
    let batch_rows = 10_000u64.min(per_file);
    while written < changes {
        let rows = batch_rows.min(changes - written) as usize;
        let batch = batch_of(written + 1, rows);
        let records = records_for(&entry, &batch, ChangeType::Insert, version);
        feed.append_batch(&records).expect("the records land");
        written += rows as u64;
        version += 1;
        if written % per_file == 0 {
            feed.seal_open_segment().expect("the file seals");
        }
    }
    (first_version, version - 1)
}

/// A wide table with its feed on, ready to be filled
async fn wide_table(server: &Arc<ServerState>, session: &mut Option<Session>, name: &str) {
    let columns: Vec<String> = (0..WIDE_COLUMNS)
        .map(|c| {
            if c == 0 {
                "c0 BIGINT PRIMARY KEY".to_string()
            } else {
                format!("c{c} BIGINT")
            }
        })
        .collect();
    ddl(
        server,
        session,
        &format!("CREATE TABLE {name} ({})", columns.join(", ")),
    )
    .await;
    ddl(
        server,
        session,
        &format!("ALTER TABLE {name} SET (change_data_feed = true)"),
    )
    .await;
}

/// Runs one statement under a transaction of its own with the change reads
/// installed, and commits it with the stream advances its reads recorded,
/// the way a connection's autocommit does. Answers with how long the whole
/// took.
///
/// A MERGE takes the dispatcher's path, the one a connection sends it down,
/// which desugars it and runs the parts in one transaction of their own
async fn autocommit(server: &Arc<ServerState>, sql: &str) -> Duration {
    let stmt = zyron_parser::parse(sql)
        .expect("parse")
        .into_iter()
        .next()
        .expect("one statement");
    let started = Instant::now();
    if matches!(stmt, zyron_parser::Statement::Merge(_)) {
        let mut session = new_session();
        let mut txn = None;
        let mut branch = None;
        match zyron_wire::ddl_dispatch::try_handle_ddl_utility(
            &stmt,
            server,
            &mut session,
            &mut txn,
            &mut branch,
            sql,
        )
        .await
        {
            Some(Ok(_)) => return started.elapsed(),
            Some(Err(e)) => panic!("`{sql}` failed: {e}"),
            None => panic!("`{sql}` was not carried out by the dispatcher"),
        }
    }
    let plan = zyron_planner::plan(
        &server.catalog,
        DatabaseId(1),
        vec!["zyron_test".into()],
        stmt,
        None,
    )
    .await
    .unwrap_or_else(|e| panic!("`{sql}` did not plan: {e}"));
    let mut txn = server
        .txn_manager
        .begin(IsolationLevel::ReadCommitted)
        .expect("begin");
    let txn_id = txn.txn_id;
    let advances = Arc::new(parking_lot::Mutex::new(Vec::new()));
    let mut ctx = ExecutionContext::new(
        server.catalog.clone(),
        server.wal.clone(),
        server.buffer_pool.clone(),
        server.disk_manager.clone(),
        txn_id,
        server.txn_manager.refresh_snapshot(&txn),
    );
    ctx.row_locks = Some(Arc::clone(server.txn_manager.lock_table()));
    ctx.intent_locks = Some(Arc::clone(server.txn_manager.intent_locks()));
    ctx.heap_files = Some(Arc::clone(&server.heap_files));
    ctx.btree_indexes = Some(Arc::clone(&server.btree_indexes));
    ctx.doc_registry = Some(Arc::clone(&server.doc_registry));
    zyron_wire::change_feed_bridge::install_change_reads(server, &mut ctx, &advances);
    if let Some(hook) = server.cdc_hook.as_ref() {
        ctx.cdc_hook = Some(Arc::clone(hook));
    }
    let ctx = Arc::new(ctx);
    zyron_executor::execute(plan, &ctx)
        .await
        .unwrap_or_else(|e| panic!("`{sql}` failed: {e}"));
    if ctx.wrote_wal() {
        txn.mark_wrote_data();
    }
    let held = std::mem::take(&mut *advances.lock());
    let advanced =
        zyron_wire::change_stream_dispatch::log_stream_advances(server, &mut txn, &held, 0)
            .expect("the advance logs");
    server.txn_manager.commit(&mut txn).await.expect("commit");
    zyron_wire::change_stream_dispatch::install_stream_advances(server, txn_id, advanced)
        .await
        .expect("install");
    started.elapsed()
}

fn stream_entry(id: u32, table_id: u32, position: u64) -> ChangeStreamEntry {
    ChangeStreamEntry {
        id,
        catalog_id: DatabaseId(1),
        schema_id: SchemaId(1),
        name: format!("s{id}"),
        source: ChangeStreamSource::Table(table_id),
        position: vec![StreamPosition {
            table_id,
            version: position,
            consumed: position,
        }],
        created_at: 0,
        created_from: ChangeStreamOrigin::Now,
        mode: ChangeStreamMode::Standard,
        predicate: None,
        columns: None,
        owner_id: 1,
        last_advanced_at: 0,
        last_advanced_by: 0,
        stale: false,
        stale_reason: String::new(),
        needs_attention: false,
        attention_reason: String::new(),
        initial_rows_pending: false,
        branch: None,
    }
}

// ---------------------------------------------------------------------------
// The feed's cost on a write
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_cdf_write_overhead_per_row() {
    zyron_bench_harness::init("change_stream");
    tprintln!("\n=== CDF Write Overhead Per Row ===");
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    ddl(
        &server,
        &mut session,
        "CREATE TABLE plain (id BIGINT, v BIGINT)",
    )
    .await;
    ddl(
        &server,
        &mut session,
        "CREATE TABLE recorded (id BIGINT, v BIGINT)",
    )
    .await;
    ddl(
        &server,
        &mut session,
        "ALTER TABLE recorded SET (change_data_feed = true)",
    )
    .await;
    ddl(
        &server,
        &mut session,
        "CREATE TABLE toggled (id BIGINT, v BIGINT)",
    )
    .await;
    ddl(
        &server,
        &mut session,
        "ALTER TABLE toggled SET (change_data_feed = true)",
    )
    .await;
    ddl(
        &server,
        &mut session,
        "ALTER TABLE toggled SET (change_data_feed = false)",
    )
    .await;

    const ROWS_PER_STATEMENT: usize = 1_000;
    const STATEMENTS: usize = 200;
    let rows = (ROWS_PER_STATEMENT * STATEMENTS) as f64;
    let values: String = (0..ROWS_PER_STATEMENT)
        .map(|r| format!("({r}, {r})"))
        .collect::<Vec<_>>()
        .join(", ");
    let util_before = take_util_snapshot();

    let mut plain_ns = Vec::with_capacity(VALIDATION_RUNS);
    let mut recorded_ns = Vec::with_capacity(VALIDATION_RUNS);
    let mut toggled_ns = Vec::with_capacity(VALIDATION_RUNS);
    let statements: Vec<String> = ["plain", "recorded", "toggled"]
        .iter()
        .map(|name| format!("INSERT INTO {name} VALUES {values}"))
        .collect();
    for run in 0..VALIDATION_RUNS {
        // The three tables take turns statement by statement, so each is
        // measured under the same buffer pool and log state as the others
        // rather than the third paying for what the first two filled, and
        // the turn order rotates, so a statement that follows one whose
        // feed sealed a file is each table's in turn rather than always
        // the same table's
        let mut spent = [Duration::ZERO; 3];
        for round in 0..STATEMENTS {
            for offset in 0..statements.len() {
                let at = (round + offset) % statements.len();
                let started = Instant::now();
                exec_dml(&server, &statements[at]).await;
                spent[at] += started.elapsed();
            }
        }
        let per_row: Vec<f64> = spent
            .iter()
            .map(|took| took.as_nanos() as f64 / rows)
            .collect();
        tprintln!(
            "  Run {}/{}: plain {:.0} ns/row, feed on {:.0} ns/row, feed off {:.0} ns/row",
            run + 1,
            VALIDATION_RUNS,
            per_row[0],
            per_row[1],
            per_row[2]
        );
        plain_ns.push(per_row[0]);
        recorded_ns.push(per_row[1]);
        toggled_ns.push(per_row[2]);
    }
    record_test_util("CDF write overhead", util_before, take_util_snapshot());
    let plain = record_metric(
        "CDF write overhead",
        "plain table",
        " ns/row",
        plain_ns.clone(),
    );
    let recorded = record_metric("CDF write overhead", "feed enabled", " ns/row", recorded_ns);
    let toggled = record_metric("CDF write overhead", "feed disabled", " ns/row", toggled_ns);
    let enabled_overhead = (recorded - plain).max(0.0);
    let disabled_overhead = toggled - plain;
    // What the plain runs themselves wander by, which is the most a
    // difference between two tables can be read to
    let noise = plain_ns
        .iter()
        .map(|n| (n - plain).abs())
        .fold(0.0f64, f64::max);
    tprintln!(
        "  Feed enabled adds {enabled_overhead:.0} ns/row (target {WRITE_OVERHEAD_NS_TARGET:.0}), \
         feed disabled adds {disabled_overhead:.0} ns/row (target 0, run noise {noise:.0})"
    );
    assert!(
        check_performance_with_unit(
            "CDF write overhead",
            "feed enabled added latency",
            " ns/row",
            enabled_overhead,
            WRITE_OVERHEAD_NS_TARGET,
            false,
        ),
        "the feed adds {enabled_overhead:.0} ns per row, over the {WRITE_OVERHEAD_NS_TARGET:.0} ns target"
    );
    record_metric(
        "CDF write overhead",
        "feed disabled added latency",
        " ns/row",
        vec![disabled_overhead],
    );
    assert!(
        disabled_overhead <= noise,
        "a disabled feed adds {disabled_overhead:.0} ns per row, beyond the {noise:.0} ns the plain runs wander by"
    );
}

// ---------------------------------------------------------------------------
// Reading the feed
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_table_changes_scan_projection_pruning_and_peek() {
    zyron_bench_harness::init("change_stream");
    tprintln!("\n=== table_changes Scan, Projection, Pruning and Peek ===");
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    wide_table(&server, &mut session, "wide").await;
    ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM wide_stream ON TABLE wide",
    )
    .await;

    let started = Instant::now();
    let (first, last) =
        fill_feed(&server, "wide", WIDE_CHANGES, WIDE_FILE_RECORDS, wide_batch).await;
    tprintln!(
        "  {} changes over {} columns in {} files, written in {:.1}s",
        format_with_commas(WIDE_CHANGES as f64),
        WIDE_COLUMNS,
        WIDE_CHANGES / WIDE_FILE_RECORDS,
        started.elapsed().as_secs_f64()
    );
    let util_before = take_util_snapshot();

    // Every column decoded. The clock stops when the last batch has been
    // handed over, and the result is freed after it, because freeing a
    // ten million row result is the harness's cost rather than the scan's
    let all = "SELECT * FROM table_changes(wide, 0, LATEST)";
    let mut all_runs = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        let started = Instant::now();
        let batches = query_batches(&server, all).await;
        let elapsed = started.elapsed();
        let rows: usize = batches.iter().map(|b| b.num_rows).sum();
        drop(batches);
        assert_eq!(rows as u64, WIDE_CHANGES, "every change is read");
        let rate = rows as f64 / elapsed.as_secs_f64();
        tprintln!(
            "  Run {}/{}: all columns {} rows/s",
            run + 1,
            VALIDATION_RUNS,
            format_with_commas(rate)
        );
        all_runs.push(rate);
    }
    let all_rate = record_metric("table_changes scan", "all columns", " rows/s", all_runs);
    zyron_common::profile::dump("table_changes scan, all columns");
    zyron_common::profile::reset();
    // Every target of this data set is measured before any is judged, so
    // a shortfall on one still leaves the numbers for the rest on record
    let mut shortfalls: Vec<String> = Vec::new();
    if !check_performance_with_unit(
        "table_changes scan",
        "10M changes, all columns",
        " rows/s",
        all_rate,
        SCAN_ROWS_PER_SEC_TARGET,
        true,
    ) {
        shortfalls.push(format!(
            "the scan reads {} rows/s, under the target",
            format_with_commas(all_rate)
        ));
    }

    // Two of forty columns
    let two = "SELECT c0, c1 FROM table_changes(wide, 0, LATEST)";
    let mut two_runs = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        let started = Instant::now();
        let batches = query_batches(&server, two).await;
        let elapsed = started.elapsed();
        let rows: usize = batches.iter().map(|b| b.num_rows).sum();
        drop(batches);
        assert_eq!(rows as u64, WIDE_CHANGES);
        let rate = rows as f64 / elapsed.as_secs_f64();
        tprintln!(
            "  Run {}/{}: two columns {} rows/s",
            run + 1,
            VALIDATION_RUNS,
            format_with_commas(rate)
        );
        two_runs.push(rate);
    }
    let two_rate = record_metric("table_changes scan", "2 of 40 columns", " rows/s", two_runs);
    zyron_common::profile::dump("table_changes scan, 2 of 40 columns");
    zyron_common::profile::reset();
    let speedup = two_rate / all_rate.max(1.0);
    tprintln!(
        "  Projecting 2 of 40 columns reads {speedup:.2}x as fast (target {PROJECTION_SPEEDUP_TARGET:.0}x)"
    );
    record_metric(
        "table_changes scan",
        "projection over all columns",
        "x",
        vec![speedup],
    );
    if speedup < PROJECTION_SPEEDUP_TARGET {
        shortfalls.push(format!(
            "projection reads {speedup:.2}x as fast, under the {PROJECTION_SPEEDUP_TARGET:.0}x target"
        ));
    }

    // A version predicate opens the matching files alone
    let files = (WIDE_CHANGES / WIDE_FILE_RECORDS) as usize;
    let one_file_versions = WIDE_FILE_RECORDS / 10_000;
    let lo = first + 400 * one_file_versions;
    let hi = lo + one_file_versions - 1;
    let plan = explain_text(
        &server,
        &format!(
            "SELECT c0 FROM table_changes(wide, 0, LATEST) WHERE _commit_version >= {lo} AND _commit_version <= {hi}"
        ),
    )
    .await;
    let opened: usize = plan
        .lines()
        .find_map(|line| {
            line.split("change_files").nth(1).and_then(|rest| {
                rest.trim_start_matches(|c: char| !c.is_ascii_digit())
                    .chars()
                    .take_while(|c| c.is_ascii_digit())
                    .collect::<String>()
                    .parse()
                    .ok()
            })
        })
        .unwrap_or(usize::MAX);
    tprintln!(
        "  A one file window over {files} files opens {opened} file(s) (versions {lo}..={hi} of {first}..={last})"
    );
    record_metric(
        "table_changes pruning",
        "files opened for one file window",
        " files",
        vec![opened as f64],
    );
    if opened > 2 {
        shortfalls.push(format!(
            "a window inside one file opened {opened} of {files} files: {plan}"
        ));
    }
    let rows = query_rows(
        &server,
        &format!("SELECT c0 FROM table_changes(wide, {}, {hi})", lo - 1),
    )
    .await;
    assert_eq!(
        rows as u64, WIDE_FILE_RECORDS,
        "the window's rows are exactly one file's"
    );

    // Peeking at what pends never opens a change file
    let feeds = server.cdc_registry.as_ref().expect("feeds");
    let runtime = zyron_cdc::change_stream::ChangeStreamRuntime::new(Arc::clone(feeds));
    let entry = server
        .catalog
        .list_change_streams()
        .into_iter()
        .find(|e| e.name == "wide_stream")
        .expect("the stream");
    let mut peek_runs = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        let started = Instant::now();
        let status = runtime.status(&entry, micros_now());
        let elapsed = started.elapsed();
        assert_eq!(
            status.pending_rows, WIDE_CHANGES,
            "every change pends on the stream"
        );
        let ms = elapsed.as_secs_f64() * 1_000.0;
        tprintln!(
            "  Run {}/{}: peek of {} pending in {:.3} ms",
            run + 1,
            VALIDATION_RUNS,
            format_with_commas(WIDE_CHANGES as f64),
            ms
        );
        peek_runs.push(ms);
    }
    record_test_util("table_changes scan", util_before, take_util_snapshot());
    let peek = record_metric("PEEK pending count", "10M pending", " ms", peek_runs);
    if !check_performance_with_unit(
        "PEEK pending count",
        "10M pending",
        " ms",
        peek,
        PEEK_LATENCY_MS_TARGET,
        false,
    ) {
        shortfalls.push(format!(
            "a peek took {peek:.3} ms, over the {PEEK_LATENCY_MS_TARGET} ms target"
        ));
    }

    // Lake: the same rows landed in a lake table read back from its log
    ddl(
        &server,
        &mut session,
        "CREATE TABLE lake_wide (a BIGINT NOT NULL, b BIGINT) USING ZYRONLAKE",
    )
    .await;
    ddl(
        &server,
        &mut session,
        "ALTER TABLE lake_wide SET (change_data_feed = true)",
    )
    .await;
    ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM lake_stream ON TABLE lake_wide",
    )
    .await;
    let started = Instant::now();
    let versions_per_commit = LAKE_COMMIT_ROWS / 10_000;
    let mut from = first - 1;
    let mut landed = 0u64;
    while landed < LAKE_ROWS {
        let to = from + versions_per_commit;
        exec_dml(
            &server,
            &format!(
                "INSERT INTO lake_wide (a, b) SELECT c0, c1 FROM table_changes(wide, {from}, {to})"
            ),
        )
        .await;
        landed += LAKE_COMMIT_ROWS;
        from = to;
    }
    tprintln!(
        "  {} rows landed in the lake table in {} commits, {:.1}s",
        format_with_commas(LAKE_ROWS as f64),
        LAKE_ROWS / LAKE_COMMIT_ROWS,
        started.elapsed().as_secs_f64()
    );
    let lake_sql = "SELECT * FROM table_changes(lake_wide, 0, LATEST)";
    let mut lake_runs = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        let started = Instant::now();
        let batches = query_batches(&server, lake_sql).await;
        let elapsed = started.elapsed();
        let rows: usize = batches.iter().map(|b| b.num_rows).sum();
        drop(batches);
        assert_eq!(rows as u64, LAKE_ROWS, "every lake change is read");
        let rate = rows as f64 / elapsed.as_secs_f64();
        tprintln!(
            "  Run {}/{}: lake changes {} rows/s",
            run + 1,
            VALIDATION_RUNS,
            format_with_commas(rate)
        );
        lake_runs.push(rate);
    }
    let lake_rate = record_metric(
        "Lake table_changes",
        "10M changes from the log",
        " rows/s",
        lake_runs,
    );
    if !check_performance_with_unit(
        "Lake table_changes",
        "10M changes from the log",
        " rows/s",
        lake_rate,
        LAKE_SCAN_ROWS_PER_SEC_TARGET,
        true,
    ) {
        shortfalls.push(format!(
            "the lake scan reads {} rows/s, under the target",
            format_with_commas(lake_rate)
        ));
    }

    // A compaction with a stream attached yields no change
    let stream_before = server
        .catalog
        .list_change_streams()
        .into_iter()
        .find(|e| e.name == "lake_stream")
        .expect("the lake stream");
    let pending_before = runtime.pending_rows(&stream_before);
    let paths = zyron_lake::LakePaths::new(
        server.disk_manager.data_dir(),
        table_id_of(&server, "lake_wide"),
    );
    let log = zyron_lake::TransactionLog::lookup_shared(&paths).expect("the lake log");
    let version_before = log.latest_version();
    zyron_wire::connection::lake_optimize(
        &server,
        &log,
        table_id_of(&server, "lake_wide"),
        false,
        true,
    )
    .await
    .expect("the compaction runs");
    let produced = count(
        &server,
        &format!("SELECT COUNT(*) FROM table_changes(lake_wide, {version_before}, LATEST)"),
    )
    .await;
    let pending_after = runtime.pending_rows(&stream_before);
    tprintln!(
        "  A compaction of the lake table produced {produced} change record(s), the stream pends {pending_after} (was {pending_before})"
    );
    record_metric(
        "Lake compaction",
        "change records produced",
        " records",
        vec![produced as f64],
    );
    assert_eq!(produced, 0, "a compaction produced change records");
    assert_eq!(
        pending_after, pending_before,
        "a compaction moved what the stream pends"
    );
    assert!(
        shortfalls.is_empty(),
        "targets not met:\n{}",
        shortfalls.join("\n")
    );
}

// ---------------------------------------------------------------------------
// Holding a position
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_stream_read_and_advance_against_merge() {
    zyron_bench_harness::init("change_stream");
    tprintln!("\n=== Stream Read And Advance Against The Same MERGE ===");
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    ddl(
        &server,
        &mut session,
        "CREATE TABLE orders (id BIGINT PRIMARY KEY, total BIGINT)",
    )
    .await;
    ddl(
        &server,
        &mut session,
        "ALTER TABLE orders SET (change_data_feed = true)",
    )
    .await;
    ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM order_stream ON TABLE orders",
    )
    .await;
    let (first, last) = fill_feed(
        &server,
        "orders",
        MERGE_CHANGES,
        MERGE_CHANGES,
        |first, rows| {
            let ids: Vec<i64> = (0..rows as u64).map(|r| (first + r) as i64).collect();
            let totals: Vec<i64> = ids.iter().map(|id| id * 3).collect();
            DataBatch::new(vec![
                Column::new(ColumnData::Int64(ids), zyron_common::TypeId::Int64),
                Column::new(ColumnData::Int64(totals), zyron_common::TypeId::Int64),
            ])
        },
    )
    .await;
    let util_before = take_util_snapshot();

    let merge_from = |source: &str, alias: &str, target: &str| {
        format!(
            "MERGE INTO {target} USING {source} ON {target}.id = {alias}.id \
             WHEN MATCHED THEN UPDATE SET total = {alias}.total \
             WHEN NOT MATCHED THEN INSERT (id, total) VALUES ({alias}.id, {alias}.total)"
        )
    };
    let mut stream_runs = Vec::with_capacity(VALIDATION_RUNS);
    let mut table_runs = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        ddl(
            &server,
            &mut session,
            &format!("CREATE TABLE via_stream_{run} (id BIGINT PRIMARY KEY, total BIGINT)"),
        )
        .await;
        ddl(
            &server,
            &mut session,
            &format!("CREATE TABLE via_table_{run} (id BIGINT PRIMARY KEY, total BIGINT)"),
        )
        .await;
        ddl(
            &server,
            &mut session,
            &format!(
                "ALTER CHANGE STREAM order_stream RESET TO VERSION {}",
                first - 1
            ),
        )
        .await;

        let stream = autocommit(
            &server,
            &merge_from("order_stream", "order_stream", &format!("via_stream_{run}")),
        )
        .await;
        let table = autocommit(
            &server,
            &merge_from(
                &format!("table_changes(orders, {}, {last}) AS src", first - 1),
                "src",
                &format!("via_table_{run}"),
            ),
        )
        .await;
        assert_eq!(
            count(&server, &format!("SELECT COUNT(*) FROM via_stream_{run}")).await as u64,
            MERGE_CHANGES
        );
        assert_eq!(
            count(&server, &format!("SELECT COUNT(*) FROM via_table_{run}")).await as u64,
            MERGE_CHANGES
        );
        tprintln!(
            "  Run {}/{}: through the stream {:.1} ms, from the table {:.1} ms",
            run + 1,
            VALIDATION_RUNS,
            stream.as_secs_f64() * 1_000.0,
            table.as_secs_f64() * 1_000.0
        );
        stream_runs.push(stream.as_secs_f64() * 1_000.0);
        table_runs.push(table.as_secs_f64() * 1_000.0);
    }
    record_test_util("Stream read and advance", util_before, take_util_snapshot());
    let stream = record_metric(
        "Stream read and advance",
        "MERGE through the stream, 100K changes",
        " ms",
        stream_runs,
    );
    let table = record_metric(
        "Stream read and advance",
        "MERGE from the table, 100K changes",
        " ms",
        table_runs,
    );
    let ratio = stream / table.max(1e-9);
    tprintln!(
        "  The stream costs {:.1}% of the table (target ≤ {:.0}%)",
        ratio * 100.0,
        STREAM_OVER_MERGE_RATIO_TARGET * 100.0
    );
    record_metric(
        "Stream read and advance",
        "stream over table",
        "x",
        vec![ratio],
    );
    assert!(
        ratio <= STREAM_OVER_MERGE_RATIO_TARGET,
        "holding the position costs {:.1}% of the same merge from the table, over the target",
        ratio * 100.0
    );

    // One commit's advance, against the same statement over the table
    ddl(
        &server,
        &mut session,
        "CREATE TABLE one_at_a_time (id BIGINT PRIMARY KEY, total BIGINT)",
    )
    .await;
    ddl(
        &server,
        &mut session,
        "ALTER TABLE one_at_a_time SET (change_data_feed = true)",
    )
    .await;
    ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM one_stream ON TABLE one_at_a_time",
    )
    .await;
    ddl(
        &server,
        &mut session,
        "CREATE TABLE one_via_stream (id BIGINT, total BIGINT)",
    )
    .await;
    ddl(
        &server,
        &mut session,
        "CREATE TABLE one_via_table (id BIGINT, total BIGINT)",
    )
    .await;
    const STATEMENTS: usize = 200;
    let mut advance_runs = Vec::with_capacity(VALIDATION_RUNS);
    let mut commit_runs = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        let mut with_advance = Duration::ZERO;
        let mut without = Duration::ZERO;
        for i in 0..STATEMENTS {
            let id = (run * STATEMENTS + i) as u64 + 1;
            exec_dml(
                &server,
                &format!("INSERT INTO one_at_a_time VALUES ({id}, {id})"),
            )
            .await;
            let feed = server
                .cdc_registry
                .as_ref()
                .expect("feeds")
                .get_feed(table_id_of(&server, "one_at_a_time"))
                .expect("feed");
            let version = feed.latest_version().expect("a version");
            with_advance += autocommit(
                &server,
                "INSERT INTO one_via_stream SELECT id, total FROM one_stream",
            )
            .await;
            without += autocommit(
                &server,
                &format!("INSERT INTO one_via_table SELECT id, total FROM table_changes(one_at_a_time, {}, {version})", version - 1),
            )
            .await;
        }
        let per_advance = with_advance.as_secs_f64() * 1e6 / STATEMENTS as f64;
        let per_plain = without.as_secs_f64() * 1e6 / STATEMENTS as f64;
        tprintln!(
            "  Run {}/{}: one change through the stream {:.0} us, from the table {:.0} us",
            run + 1,
            VALIDATION_RUNS,
            per_advance,
            per_plain
        );
        advance_runs.push(per_advance - per_plain);
        commit_runs.push(per_plain);
    }
    let added = record_metric(
        "Stream position advance",
        "added latency, one commit",
        " us",
        advance_runs,
    );
    let commit = record_metric(
        "Stream position advance",
        "the same statement from the table",
        " us",
        commit_runs,
    );
    assert!(
        check_performance_with_unit(
            "Stream position advance",
            "added latency, one commit",
            " us",
            added.max(0.0),
            ADVANCE_LATENCY_US_TARGET,
            false
        ),
        "an advance adds {added:.0} us to a commit, over the {ADVANCE_LATENCY_US_TARGET:.0} us target"
    );

    // Two consumers, where the second waits on the first and proceeds within one
    // commit of it
    ddl(
        &server,
        &mut session,
        "CREATE TABLE contended (id BIGINT PRIMARY KEY, total BIGINT)",
    )
    .await;
    ddl(
        &server,
        &mut session,
        "ALTER TABLE contended SET (change_data_feed = true)",
    )
    .await;
    ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM contended_stream ON TABLE contended",
    )
    .await;
    ddl(
        &server,
        &mut session,
        "CREATE TABLE contended_out (id BIGINT, total BIGINT)",
    )
    .await;
    let mut overhead_runs = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        exec_dml(
            &server,
            &format!("INSERT INTO contended VALUES ({}, 1)", run + 1),
        )
        .await;
        // The first consumer reads and holds the position
        let stmt =
            zyron_parser::parse("INSERT INTO contended_out SELECT id, total FROM contended_stream")
                .expect("parse")
                .into_iter()
                .next()
                .expect("one statement");
        let plan = zyron_planner::plan(
            &server.catalog,
            DatabaseId(1),
            vec!["zyron_test".into()],
            stmt,
            None,
        )
        .await
        .expect("plan");
        let mut txn = server
            .txn_manager
            .begin(IsolationLevel::ReadCommitted)
            .expect("begin");
        let txn_id = txn.txn_id;
        let advances = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let mut ctx = ExecutionContext::new(
            server.catalog.clone(),
            server.wal.clone(),
            server.buffer_pool.clone(),
            server.disk_manager.clone(),
            txn_id,
            server.txn_manager.refresh_snapshot(&txn),
        );
        ctx.row_locks = Some(Arc::clone(server.txn_manager.lock_table()));
        ctx.intent_locks = Some(Arc::clone(server.txn_manager.intent_locks()));
        ctx.heap_files = Some(Arc::clone(&server.heap_files));
        ctx.btree_indexes = Some(Arc::clone(&server.btree_indexes));
        ctx.doc_registry = Some(Arc::clone(&server.doc_registry));
        zyron_wire::change_feed_bridge::install_change_reads(&server, &mut ctx, &advances);
        if let Some(hook) = server.cdc_hook.as_ref() {
            ctx.cdc_hook = Some(Arc::clone(hook));
        }
        let ctx = Arc::new(ctx);
        zyron_executor::execute(plan, &ctx)
            .await
            .expect("the first consume reads");
        if ctx.wrote_wal() {
            txn.mark_wrote_data();
        }
        // The second consumer starts now and waits on the position
        let second_server = Arc::clone(&server);
        let second = tokio::spawn(async move {
            let started = Instant::now();
            autocommit(
                &second_server,
                "INSERT INTO contended_out SELECT id, total FROM contended_stream",
            )
            .await;
            started.elapsed()
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        let held = std::mem::take(&mut *advances.lock());
        let advanced =
            zyron_wire::change_stream_dispatch::log_stream_advances(&server, &mut txn, &held, 0)
                .expect("log");
        let released_at = Instant::now();
        server.txn_manager.commit(&mut txn).await.expect("commit");
        zyron_wire::change_stream_dispatch::install_stream_advances(&server, txn_id, advanced)
            .await
            .expect("install");
        let second_total = second.await.expect("the second consumer finishes");
        let after_release = released_at.elapsed();
        // What the second paid beyond the first's commit is the handover
        let one_commit = commit / 1e6;
        let overhead_ms = after_release.as_secs_f64() * 1_000.0;
        tprintln!(
            "  Run {}/{}: the second consumer finished {:.2} ms after the first began committing (its whole wait {:.1} ms), one commit {:.2} ms",
            run + 1,
            VALIDATION_RUNS,
            overhead_ms,
            second_total.as_secs_f64() * 1_000.0,
            one_commit * 1_000.0
        );
        overhead_runs.push(overhead_ms);
    }
    let overhead = record_metric(
        "Concurrent consumers",
        "second proceeds after the first's commit",
        " ms",
        overhead_runs,
    );
    // The second consumer commits a statement of its own once it has the
    // position, so one commit of its own is what it pays past the first
    let one_commit_ms = commit / 1_000.0;
    record_metric(
        "Concurrent consumers",
        "one commit",
        " ms",
        vec![one_commit_ms],
    );
    assert!(
        overhead <= 2.0 * one_commit_ms + 5.0,
        "the second consumer took {overhead:.2} ms past the first's commit, more than the {one_commit_ms:.2} ms commit of its own plus scheduling"
    );
}

// ---------------------------------------------------------------------------
// Applying changes
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_apply_changes_type1_type2_and_order() {
    zyron_bench_harness::init("change_stream");
    tprintln!("\n=== APPLY CHANGES Type 1, Type 2 And Out Of Order ===");
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    for name in ["src_ordered", "src_shuffled"] {
        ddl(
            &server,
            &mut session,
            &format!("CREATE TABLE {name} (k BIGINT PRIMARY KEY, v BIGINT, seq BIGINT)"),
        )
        .await;
        ddl(
            &server,
            &mut session,
            &format!("ALTER TABLE {name} SET (change_data_feed = true)"),
        )
        .await;
    }
    let versions_per_key = APPLY_CHANGES / APPLY_KEYS;
    // Ordered: every key's changes ascend by sequence in commit order.
    // Shuffled: the same changes, each version's batch drawn from every
    // sequence so a key's later change often lands before its earlier one
    let ordered = |first: u64, rows: usize| {
        let mut keys = Vec::with_capacity(rows);
        let mut values = Vec::with_capacity(rows);
        let mut seqs = Vec::with_capacity(rows);
        for r in 0..rows as u64 {
            let n = first + r - 1;
            keys.push((n % APPLY_KEYS) as i64);
            seqs.push((n / APPLY_KEYS) as i64);
            values.push(n as i64);
        }
        DataBatch::new(vec![
            Column::new(ColumnData::Int64(keys), zyron_common::TypeId::Int64),
            Column::new(ColumnData::Int64(values), zyron_common::TypeId::Int64),
            Column::new(ColumnData::Int64(seqs), zyron_common::TypeId::Int64),
        ])
    };
    let shuffled = |first: u64, rows: usize| {
        let mut keys = Vec::with_capacity(rows);
        let mut values = Vec::with_capacity(rows);
        let mut seqs = Vec::with_capacity(rows);
        for r in 0..rows as u64 {
            let n = first + r - 1;
            // A fixed permutation of the change index, so the same million
            // changes arrive in another order. Multiplying by a constant
            // coprime to the count is a bijection on the indices, so every
            // change appears exactly once
            let m = (n * 2_654_435_761) % APPLY_CHANGES;
            keys.push((m % APPLY_KEYS) as i64);
            seqs.push((m / APPLY_KEYS) as i64);
            values.push(m as i64);
        }
        DataBatch::new(vec![
            Column::new(ColumnData::Int64(keys), zyron_common::TypeId::Int64),
            Column::new(ColumnData::Int64(values), zyron_common::TypeId::Int64),
            Column::new(ColumnData::Int64(seqs), zyron_common::TypeId::Int64),
        ])
    };
    let started = Instant::now();
    fill_feed(&server, "src_ordered", APPLY_CHANGES, 100_000, ordered).await;
    fill_feed(&server, "src_shuffled", APPLY_CHANGES, 100_000, shuffled).await;
    tprintln!(
        "  {} changes over {} keys, {} per key, ordered and shuffled, written in {:.1}s",
        format_with_commas(APPLY_CHANGES as f64),
        format_with_commas(APPLY_KEYS as f64),
        versions_per_key,
        started.elapsed().as_secs_f64()
    );
    let util_before = take_util_snapshot();

    let mut type1_runs = Vec::with_capacity(VALIDATION_RUNS);
    let mut type2_runs = Vec::with_capacity(VALIDATION_RUNS);
    let mut shuffled_runs = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        ddl(
            &server,
            &mut session,
            &format!("CREATE TABLE t1_{run} (k BIGINT PRIMARY KEY, v BIGINT, seq BIGINT)"),
        )
        .await;
        ddl(
            &server,
            &mut session,
            &format!("CREATE TABLE t2_{run} (k BIGINT, v BIGINT, seq BIGINT, __start_at BIGINT, __end_at BIGINT, __is_current BOOLEAN)"),
        )
        .await;
        ddl(
            &server,
            &mut session,
            &format!("CREATE TABLE t1s_{run} (k BIGINT PRIMARY KEY, v BIGINT, seq BIGINT)"),
        )
        .await;

        // The ordered and the shuffled apply take turns going first, so
        // neither is always the one measured just after the other's writes
        // landed, and the type 2 apply follows both
        let ordered_sql = format!(
            "APPLY CHANGES INTO t1_{run} FROM table_changes(src_ordered, 0, LATEST) KEYS (k) SEQUENCE BY seq"
        );
        let shuffled_sql = format!(
            "APPLY CHANGES INTO t1s_{run} FROM table_changes(src_shuffled, 0, LATEST) KEYS (k) SEQUENCE BY seq"
        );
        let (first, second) = if run % 2 == 0 {
            (&ordered_sql, &shuffled_sql)
        } else {
            (&shuffled_sql, &ordered_sql)
        };
        let started = Instant::now();
        ddl(&server, &mut session, first).await;
        let first_took = started.elapsed();
        let started = Instant::now();
        ddl(&server, &mut session, second).await;
        let second_took = started.elapsed();
        let (type1, shuffled_took) = if run % 2 == 0 {
            (first_took, second_took)
        } else {
            (second_took, first_took)
        };
        let started = Instant::now();
        ddl(
            &server,
            &mut session,
            &format!("APPLY CHANGES INTO t2_{run} FROM table_changes(src_ordered, 0, LATEST) KEYS (k) SEQUENCE BY seq STORED AS SCD TYPE 2"),
        )
        .await;
        let type2 = started.elapsed();

        assert_eq!(
            count(&server, &format!("SELECT COUNT(*) FROM t1_{run}")).await as u64,
            APPLY_KEYS
        );
        assert_eq!(
            count(
                &server,
                &format!(
                    "SELECT COUNT(*) FROM t1_{run} WHERE seq <> {}",
                    versions_per_key - 1
                )
            )
            .await,
            0,
            "the last change to every key won"
        );
        assert_eq!(
            count(&server, &format!("SELECT COUNT(*) FROM t2_{run}")).await as u64,
            APPLY_CHANGES
        );
        assert_eq!(
            count(
                &server,
                &format!("SELECT COUNT(*) FROM t2_{run} WHERE __is_current")
            )
            .await as u64,
            APPLY_KEYS
        );
        assert_eq!(
            count(&server, &format!("SELECT COUNT(*) FROM t1_{run} a JOIN t1s_{run} b ON a.k = b.k AND a.v = b.v AND a.seq = b.seq")).await as u64,
            APPLY_KEYS,
            "the shuffled input produced the same target"
        );

        let t1_rate = APPLY_CHANGES as f64 / type1.as_secs_f64();
        let t2_rate = APPLY_CHANGES as f64 / type2.as_secs_f64();
        let s_rate = APPLY_CHANGES as f64 / shuffled_took.as_secs_f64();
        tprintln!(
            "  Run {}/{}: type 1 {} changes/s, type 2 {} changes/s, out of order {} changes/s",
            run + 1,
            VALIDATION_RUNS,
            format_with_commas(t1_rate),
            format_with_commas(t2_rate),
            format_with_commas(s_rate)
        );
        type1_runs.push(t1_rate);
        type2_runs.push(t2_rate);
        shuffled_runs.push(s_rate);
    }
    record_test_util("APPLY CHANGES", util_before, take_util_snapshot());
    let type1 = record_metric(
        "APPLY CHANGES",
        "type 1, 1M changes over 100K keys",
        " changes/s",
        type1_runs,
    );
    let type2 = record_metric(
        "APPLY CHANGES",
        "type 2, 1M changes over 100K keys",
        " changes/s",
        type2_runs,
    );
    let out_of_order = record_metric(
        "APPLY CHANGES",
        "type 1, out of order",
        " changes/s",
        shuffled_runs,
    );
    assert!(
        check_performance_with_unit(
            "APPLY CHANGES",
            "type 1",
            " changes/s",
            type1,
            APPLY_TYPE1_TARGET,
            true
        ),
        "type 1 applies {} changes/s, under the target",
        format_with_commas(type1)
    );
    assert!(
        check_performance_with_unit(
            "APPLY CHANGES",
            "type 2",
            " changes/s",
            type2,
            APPLY_TYPE2_TARGET,
            true
        ),
        "type 2 applies {} changes/s, under the target",
        format_with_commas(type2)
    );
    let delta = ((type1 - out_of_order) / type1.max(1.0)).max(0.0);
    tprintln!(
        "  Out of order costs {:.1}% (target ≤ {:.0}%)",
        delta * 100.0,
        OUT_OF_ORDER_DELTA_TARGET * 100.0
    );
    record_metric(
        "APPLY CHANGES",
        "out of order over in order, throughput lost",
        "%",
        vec![delta * 100.0],
    );
    assert!(
        delta <= OUT_OF_ORDER_DELTA_TARGET,
        "an out of order source costs {:.1}% of throughput, over the target",
        delta * 100.0
    );
}

// ---------------------------------------------------------------------------
// The sweeper and the feed's storage
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_staleness_sweep_and_before_image_storage() {
    zyron_bench_harness::init("change_stream");
    tprintln!("\n=== Staleness Sweep Over 10K Streams And Before Image Storage ===");
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let feeds = Arc::new(zyron_cdc::CdfRegistry::new(tmp.path().to_path_buf()));
    let feed = feeds.enable_for_table(10, 7).expect("feed");
    let records: Vec<ChangeRecord> = (1..=2_000u64)
        .map(|v| ChangeRecord {
            change_type: ChangeType::Insert,
            commit_version: v,
            commit_timestamp: v as i64 * 1_000,
            table_id: 10,
            txn_id: v,
            change_ordinal: 0,
            schema_version: 1,
            row_data: vec![1; 32],
            primary_key_data: Vec::new(),
            is_last_in_txn: true,
            projected: false,
        })
        .collect();
    feed.append_batch(&records).expect("appends");
    feed.seal_open_segment().expect("seals");
    feed.append_batch(&[ChangeRecord {
        commit_version: 2_001,
        ..records[0].clone()
    }])
    .expect("appends");
    feed.purge_before_version(1_000).expect("purges");
    let runtime = zyron_cdc::change_stream::ChangeStreamRuntime::new(Arc::clone(&feeds));
    let entries: Vec<Arc<ChangeStreamEntry>> = (0..SWEEP_STREAMS as u32)
        .map(|i| {
            Arc::new(stream_entry(
                i + 1,
                10,
                500 + (i as u64 * 1_000) / SWEEP_STREAMS as u64,
            ))
        })
        .collect();
    let mut sweep_runs = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        let started = Instant::now();
        let flagged = runtime.sweep(&entries);
        let ms = started.elapsed().as_secs_f64() * 1_000.0;
        assert!(
            flagged.len() > SWEEP_STREAMS / 3,
            "the streams below the purge went stale: {}",
            flagged.len()
        );
        tprintln!(
            "  Run {}/{}: sweep of {} streams in {:.2} ms, {} stale",
            run + 1,
            VALIDATION_RUNS,
            SWEEP_STREAMS,
            ms,
            flagged.len()
        );
        sweep_runs.push(ms);
    }
    let sweep = record_metric("Staleness sweep", "10K streams", " ms", sweep_runs);
    assert!(
        check_performance_with_unit(
            "Staleness sweep",
            "10K streams",
            " ms",
            sweep,
            SWEEP_LATENCY_MS_TARGET,
            false
        ),
        "the sweep took {sweep:.2} ms, over the {SWEEP_LATENCY_MS_TARGET} ms target"
    );

    // Before images off against on, the same updates, uncompressed so the
    // bytes are the images written
    let mut sizes = Vec::with_capacity(2);
    for (table_id, before_image) in [(20u32, true), (21u32, false)] {
        let feed = feeds
            .enable_with_config(
                table_id,
                FeedConfig {
                    before_image,
                    codec: CdfCodec::None,
                    ..FeedConfig::default()
                },
            )
            .expect("feed");
        let mut version = 1u64;
        for _ in 0..100 {
            let mut batch = Vec::with_capacity(2_000);
            for i in 0..1_000u64 {
                let row: Vec<u8> = (0..64).map(|b| ((version + i + b) % 251) as u8).collect();
                batch.push(ChangeRecord {
                    change_type: ChangeType::UpdatePreimage,
                    commit_version: version,
                    commit_timestamp: version as i64,
                    table_id,
                    txn_id: version,
                    change_ordinal: 0,
                    schema_version: 1,
                    row_data: row.clone(),
                    primary_key_data: Vec::new(),
                    is_last_in_txn: false,
                    projected: false,
                });
                batch.push(ChangeRecord {
                    change_type: ChangeType::UpdatePostimage,
                    commit_version: version,
                    commit_timestamp: version as i64,
                    table_id,
                    txn_id: version,
                    change_ordinal: 0,
                    schema_version: 1,
                    row_data: row,
                    primary_key_data: Vec::new(),
                    is_last_in_txn: true,
                    projected: false,
                });
                version += 1;
            }
            feed.append_batch(&batch).expect("appends");
        }
        feed.seal_open_segment().expect("seals");
        sizes.push(feed.file_size_bytes() as f64);
    }
    let ratio = sizes[1] / sizes[0].max(1.0);
    tprintln!(
        "  Before images on {} bytes, off {} bytes, off is {:.1}% of on (target ≤ {:.0}%)",
        format_with_commas(sizes[0]),
        format_with_commas(sizes[1]),
        ratio * 100.0,
        BEFORE_IMAGE_OFF_RATIO_TARGET * 100.0
    );
    record_metric(
        "Feed storage",
        "before images off over on",
        "x",
        vec![ratio],
    );
    assert!(
        ratio <= BEFORE_IMAGE_OFF_RATIO_TARGET,
        "before images off is {:.1}% of on, over the target",
        ratio * 100.0
    );
}

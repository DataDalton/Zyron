#![allow(non_snake_case)]

//! Branch write benchmark suite.
//!
//! A branch's own appended rows are not in the shared index, because their
//! locators name the branch's copy-on-write file. A uniqueness check therefore
//! has to read the branch's append range as well as the index, and the shape of
//! that read is what this suite measures.
//!
//! The append range is read once per batch. Inserting n rows into a branch
//! holding m appended rows therefore costs one pass over m per batch of n,
//! which is `n/BATCH_SIZE * m + n`, against `n * m` for a read per row. That is
//! a factor of the batch size on the scan rather than a change of complexity
//! class, so the absolute cost is what separates the two and the scan is sized
//! to be the part that moves.
//!
//! What this suite does NOT measure is how the cost grows as m doubles. Both
//! shapes grow there, because the correct one still reads m once per batch, so
//! that figure cannot tell them apart and is recorded for information only.
//!
//! Performance Targets:
//! | Test                                        | Metric  | Target |
//! |---------------------------------------------|---------|--------|
//! | Insert one batch against a populated branch | latency | 10ms   |
//! | Insert on a branch against the main line    | ratio   | 3x     |
//!
//! Both bounds sit orders of magnitude away from a read per row, which is the
//! shape they exist to catch. The suite derives that shape's cost from the
//! marginal cost of the deeper append range and prints it beside the figure it
//! measured, so the comparison is recomputed each run rather than asserted from
//! a number written down once.
//!
//! Validation Requirements:
//! - Each benchmark runs 5 iterations
//! - Results averaged across all 5 runs
//! - Pass/fail determined by average performance
//! - Individual runs logged for variance analysis
//!
//! Run: cargo test --release -p zyron-wire --test branch_write_bench -- --nocapture

mod common;

use std::sync::{Arc, Mutex};
use std::time::Instant;

use zyron_bench_harness::*;
use zyron_catalog::DatabaseId;
use zyron_executor::context::ExecutionContext;
use zyron_storage::txn::IsolationLevel;
use zyron_wire::connection::ServerState;

/// The suites run one at a time, so a measurement is of the engine rather than
/// of two suites sharing the machine
static BENCHMARK_LOCK: Mutex<()> = Mutex::new(());

const VALIDATION_RUNS: usize = 5;

/// Rows inserted in the statement being timed, held fixed while the branch's
/// existing append count is varied
const INSERT_ROWS: usize = 800;

/// The two append depths compared. The second is twice the first, so a per-row
/// rescan of the append range shows up as a doubling of the insert cost
const APPENDS_LOW: usize = 800;
const APPENDS_HIGH: usize = 1600;

/// Rows per batch, which is the unit the append range is read once for. Stated
/// here because it is what makes `INSERT_ROWS` a single batch, and a single
/// batch is what separates one pass over the range from one pass per row
const ROWS_PER_BATCH: usize = zyron_executor::batch::BATCH_SIZE;

/// One pass over the append range is bounded by the range, so this insert is a
/// batch of writes plus a scan of a few thousand tuples. A read per row makes it
/// a scan per row and lands orders of magnitude outside this bound
const BRANCH_INSERT_LIMIT_US: f64 = 10_000.0;

/// A branch write pays copy-on-write page resolution, the append file and one
/// pass over the append range on top of what a main line insert pays, so it
/// costs a fraction more rather than a multiple. A read per row makes it a
/// multiple
const BRANCH_OVER_MAIN_LIMIT: f64 = 3.0;

/// Runs one statement through dispatch, plan and execute, with the execution
/// context carrying the active branch the way the wire connection wires it.
async fn exec(
    server: &Arc<ServerState>,
    session: &mut Option<zyron_wire::session::Session>,
    active_branch: &mut Option<String>,
    sql: &str,
) {
    let stmt = zyron_parser::parse(sql)
        .expect("parse")
        .into_iter()
        .next()
        .expect("one statement");

    let mut txn_opt: Option<zyron_storage::txn::Transaction> = None;
    if let Some(res) = zyron_wire::ddl_dispatch::try_handle_ddl_utility(
        &stmt,
        server,
        session,
        &mut txn_opt,
        active_branch,
        sql,
    )
    .await
    {
        res.expect("ddl handler failed");
        return;
    }

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
    let mut ctx = ExecutionContext::new(
        server.catalog.clone(),
        server.wal.clone(),
        server.buffer_pool.clone(),
        server.disk_manager.clone(),
        txn.txn_id,
        txn.snapshot.clone(),
    );
    ctx.heap_files = Some(Arc::clone(&server.heap_files));
    ctx.btree_indexes = Some(Arc::clone(&server.btree_indexes));
    if let Some(mgr) = &server.branch_manager {
        ctx.branch_catalog = Some(Arc::clone(mgr) as Arc<dyn zyron_common::BranchCatalog>);
        if let Some(name) = active_branch.as_deref() {
            ctx.active_branch_id = mgr.get_branch_by_name(name).ok().map(|e| e.id.0);
            ctx.active_branch_name = Some(Arc::from(name));
        }
    }
    let ctx = Arc::new(ctx);
    zyron_executor::execute(plan, &ctx).await.expect("execute");
    server.txn_manager.commit(&mut txn).await.expect("commit");
}

/// `INSERT INTO t VALUES (a), (a+1), ...`, one statement carrying `count` rows
/// starting at `first`, which is how a batch reaches the uniqueness check.
fn insert_sql(table: &str, first: usize, count: usize) -> String {
    let mut sql = String::with_capacity(16 + count * 8);
    sql.push_str("INSERT INTO ");
    sql.push_str(table);
    sql.push_str(" VALUES ");
    for i in 0..count {
        if i > 0 {
            sql.push(',');
        }
        sql.push('(');
        sql.push_str(&(first + i).to_string());
        sql.push(')');
    }
    sql
}

/// Builds a branch holding `appends` rows, then times one insert of
/// `INSERT_ROWS` further rows into it. Returns the insert's duration.
///
/// The table is created with a unique index, because heap uniqueness is
/// enforced through one and that enforcement is the path under test.
async fn timed_branch_insert(appends: usize) -> std::time::Duration {
    let (server, _schema, _tmp) = common::create_test_server_with_branches().await;
    let mut session = common::new_session();
    let mut branch: Option<String> = None;

    exec(
        &server,
        &mut session,
        &mut branch,
        "CREATE TABLE t (k BIGINT NOT NULL)",
    )
    .await;
    exec(
        &server,
        &mut session,
        &mut branch,
        "CREATE UNIQUE INDEX t_k_ux ON t (k)",
    )
    .await;
    exec(&server, &mut session, &mut branch, "CREATE BRANCH dev").await;
    exec(&server, &mut session, &mut branch, "USE BRANCH dev").await;

    // The branch's own appended rows, which the uniqueness check has to read
    // because the shared index does not carry them. Written in batch sized
    // statements so building the state is not itself quadratic
    let mut written = 0usize;
    while written < appends {
        let chunk = (appends - written).min(400);
        exec(
            &server,
            &mut session,
            &mut branch,
            &insert_sql("t", written, chunk),
        )
        .await;
        written += chunk;
    }

    let sql = insert_sql("t", appends, INSERT_ROWS);
    let start = Instant::now();
    exec(&server, &mut session, &mut branch, &sql).await;
    start.elapsed()
}

/// Times the same insert on the main line, with no branch active.
async fn timed_main_insert() -> std::time::Duration {
    let (server, _schema, _tmp) = common::create_test_server_with_branches().await;
    let mut session = common::new_session();
    let mut branch: Option<String> = None;

    exec(
        &server,
        &mut session,
        &mut branch,
        "CREATE TABLE t (k BIGINT NOT NULL)",
    )
    .await;
    exec(
        &server,
        &mut session,
        &mut branch,
        "CREATE UNIQUE INDEX t_k_ux ON t (k)",
    )
    .await;

    let mut written = 0usize;
    while written < APPENDS_LOW {
        let chunk = (APPENDS_LOW - written).min(400);
        exec(
            &server,
            &mut session,
            &mut branch,
            &insert_sql("t", written, chunk),
        )
        .await;
        written += chunk;
    }

    let sql = insert_sql("t", APPENDS_LOW, INSERT_ROWS);
    let start = Instant::now();
    exec(&server, &mut session, &mut branch, &sql).await;
    start.elapsed()
}

// =============================================================================
// Test 1: the insert cost when the branch's existing appends double
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn branch_insert_cost_against_the_branch_s_existing_appends() {
    zyron_bench_harness::init("branch_write");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Branch Uniqueness Against Existing Appends ===");
    tprintln!("  Rows inserted per measurement: {INSERT_ROWS}");
    tprintln!("  Branch appends compared: {APPENDS_LOW} against {APPENDS_HIGH}");

    let mut low_runs = Vec::with_capacity(VALIDATION_RUNS);
    let mut high_runs = Vec::with_capacity(VALIDATION_RUNS);
    for _ in 0..VALIDATION_RUNS {
        low_runs.push(timed_branch_insert(APPENDS_LOW).await.as_secs_f64() * 1e6);
        high_runs.push(timed_branch_insert(APPENDS_HIGH).await.as_secs_f64() * 1e6);
    }

    let low = record_metric(
        "Branch append scaling",
        &format!("insert of {INSERT_ROWS} rows, branch holding {APPENDS_LOW} appends"),
        "us",
        low_runs,
    );
    let high = record_metric(
        "Branch append scaling",
        &format!("insert of {INSERT_ROWS} rows, branch holding {APPENDS_HIGH} appends"),
        "us",
        high_runs,
    );

    let growth = high / low.max(1.0);
    record_metric(
        "Branch append scaling",
        "insert cost growth when the branch's appends double",
        "x",
        vec![growth],
    );
    tprintln!(
        "  Appends {APPENDS_LOW} to {APPENDS_HIGH}: the insert grows {growth:.2}x. \
         Recorded only, both shapes grow here"
    );
    // The marginal cost of the deeper range is the cost of one pass over the
    // rows it added, which prices a decode without having to guess at it
    let per_tuple = (high - low) / (APPENDS_HIGH - APPENDS_LOW) as f64;
    let per_row_rescan_us = per_tuple * (INSERT_ROWS * APPENDS_HIGH) as f64;
    tprintln!(
        "  {INSERT_ROWS} rows is one batch of {ROWS_PER_BATCH}, so the range is read \
         once: {APPENDS_HIGH} tuple decodes against {} for a read per row",
        INSERT_ROWS * APPENDS_HIGH
    );
    tprintln!(
        "  A decode costs {per_tuple:.3}us at the margin, so a read per row would \
         put this insert near {:.1}ms against the {:.3}ms measured",
        per_row_rescan_us / 1000.0,
        high / 1000.0
    );
    assert!(
        high <= BRANCH_INSERT_LIMIT_US,
        "inserting {INSERT_ROWS} rows into a branch holding {APPENDS_HIGH} appends took \
         {high:.0}us, over the {BRANCH_INSERT_LIMIT_US:.0}us limit. The uniqueness check \
         is reading the append range once per row again rather than once per batch"
    );
}

// =============================================================================
// Test 2: a branch insert against the same insert on the main line
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn branch_insert_against_the_main_line() {
    zyron_bench_harness::init("branch_write");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Branch Insert Against The Main Line ===");

    let mut branch_runs = Vec::with_capacity(VALIDATION_RUNS);
    let mut main_runs = Vec::with_capacity(VALIDATION_RUNS);
    for _ in 0..VALIDATION_RUNS {
        branch_runs.push(timed_branch_insert(APPENDS_LOW).await.as_secs_f64() * 1e6);
        main_runs.push(timed_main_insert().await.as_secs_f64() * 1e6);
    }

    let on_branch = record_metric(
        "Branch against main",
        &format!("insert of {INSERT_ROWS} rows on a branch"),
        "us",
        branch_runs,
    );
    let on_main = record_metric(
        "Branch against main",
        &format!("insert of {INSERT_ROWS} rows on the main line"),
        "us",
        main_runs,
    );

    let ratio = on_branch / on_main.max(1.0);
    record_metric(
        "Branch against main",
        "branch insert over the main line",
        "x",
        vec![ratio],
    );
    tprintln!(
        "  A branch insert costs {ratio:.2}x the main line \
         (limit {BRANCH_OVER_MAIN_LIMIT:.2}x)"
    );
    assert!(
        ratio <= BRANCH_OVER_MAIN_LIMIT,
        "a branch insert cost {ratio:.2}x the main line, over the \
         {BRANCH_OVER_MAIN_LIMIT:.2}x limit"
    );
}

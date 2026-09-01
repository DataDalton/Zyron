//! A heap larger than the buffer pool must still hold every row.
//!
//! A load big enough to overflow the pool sends pages out to disk while the
//! writer is still appending, so every row's survival depends on the eviction
//! write, the page count the scan reads and the frame identity the pool hands
//! back all agreeing. A row lost in that window shows up only as a count that
//! is short, which is silent: the statements all succeeded.
//!
//! The pool is sized down rather than the table sized up, because the ratio
//! between them is what these tests are about and a small pool reaches the
//! same ratio in seconds instead of minutes.
//!
//! Counts are cross-checked against a sum and a scan rather than compared
//! against themselves, so a load that dropped one row and duplicated another
//! cannot pass.
//!
//! Run: cargo test -p zyron-wire --test large_heap_integrity_test

mod common;

use std::sync::Arc;

use common::{create_test_server_with_pool_frames, exec_ddl, exec_dml, new_session, query_values};
use zyron_executor::column::ScalarValue;
use zyron_wire::connection::ServerState;

/// Frames the pool gets. Sixty-four frames is half a megabyte, which every
/// table below overruns many times over
const POOL_FRAMES: usize = 64;

/// Rows per table. Enough that the heap is tens of times the pool
const ROWS: usize = 60_000;

/// Rows per INSERT, matching the multi-row batches a bulk load sends
const PER_STATEMENT: usize = 2_000;

fn scalar_i64(rows: &[Vec<ScalarValue>], what: &str) -> i64 {
    match rows.first().and_then(|r| r.first()) {
        Some(ScalarValue::Int64(v)) => *v,
        Some(ScalarValue::Int32(v)) => i64::from(*v),
        Some(ScalarValue::Float64(v)) => *v as i64,
        other => panic!("{what}: expected a number, got {other:?}"),
    }
}

/// Row `i` of the generated corpus. Wide enough that a page holds tens of
/// rows rather than hundreds, so the load crosses many page boundaries
fn row_values(i: usize) -> String {
    format!(
        "{i}, 'row {i} padded so a page holds a few dozen of these and the load crosses many page boundaries'"
    )
}

fn insert_statements(table: &str, first_id: usize, n: usize, per_stmt: usize) -> Vec<String> {
    let mut out = Vec::with_capacity(n.div_ceil(per_stmt));
    let mut i = 0;
    while i < n {
        let end = (i + per_stmt).min(n);
        let mut sql = String::with_capacity((end - i) * 96 + 32);
        sql.push_str("INSERT INTO ");
        sql.push_str(table);
        sql.push_str(" VALUES ");
        for r in i..end {
            if r > i {
                sql.push(',');
            }
            sql.push('(');
            sql.push_str(&row_values(first_id + r));
            sql.push(')');
        }
        out.push(sql);
        i = end;
    }
    out
}

async fn load(server: &Arc<ServerState>, table: &str, n: usize) {
    for sql in insert_statements(table, 0, n, PER_STATEMENT) {
        exec_dml(server, &sql).await;
    }
}

/// Every way of counting the table has to give the same answer, and that
/// answer has to be the number of rows written.
///
/// The sum is what makes this more than a self-consistency check: ids run
/// `0..n`, so the sum pins down which rows are present, and a load that lost
/// one row and wrote another twice fails here even though the count would
/// agree.
async fn assert_intact(server: &Arc<ServerState>, table: &str, n: usize) {
    let n64 = n as i64;
    let expected_sum = n64 * (n64 - 1) / 2;

    let star = scalar_i64(
        &query_values(server, &format!("SELECT count(*) FROM {table}")).await,
        "count(*)",
    );
    assert_eq!(star, n64, "{table}: count(*) does not see every row");

    let col = scalar_i64(
        &query_values(server, &format!("SELECT count(id) FROM {table}")).await,
        "count(id)",
    );
    assert_eq!(col, n64, "{table}: count(id) disagrees with count(*)");

    let sum = scalar_i64(
        &query_values(server, &format!("SELECT sum(id) FROM {table}")).await,
        "sum(id)",
    );
    assert_eq!(
        sum, expected_sum,
        "{table}: the rows present are not the rows written"
    );

    let max = scalar_i64(
        &query_values(server, &format!("SELECT max(id) FROM {table}")).await,
        "max(id)",
    );
    assert_eq!(max, n64 - 1, "{table}: the last row written is missing");

    // A scan that materializes every row, so the count is checked against the
    // rows themselves rather than against another aggregate over the same scan
    let scanned = query_values(server, &format!("SELECT id FROM {table}")).await;
    assert_eq!(
        scanned.len(),
        n,
        "{table}: a full scan produced a different number of rows than count(*)"
    );
}

/// The plain case: one table, loaded past the pool, counted every way
#[tokio::test]
async fn test_a_a_heap_many_times_the_pool_keeps_every_row() {
    let (server, _schema, _tmp) = create_test_server_with_pool_frames(POOL_FRAMES).await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE wide (id INT, body TEXT)",
    )
    .await
    .expect("create the table");
    load(&server, "wide", ROWS).await;
    assert_intact(&server, "wide", ROWS).await;
}

/// The same load with an index maintained on every insert, which is the shape
/// the hybrid search corpus is built in. Index maintenance pins and reads
/// pages of its own while the load is evicting, so it competes for the same
/// frames the appender needs
#[tokio::test]
async fn test_b_an_index_maintained_during_the_load_loses_nothing() {
    let (server, _schema, _tmp) = create_test_server_with_pool_frames(POOL_FRAMES).await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE indexed (id INT, body TEXT)",
    )
    .await
    .expect("create the table");
    exec_ddl(
        &server,
        &mut session,
        "CREATE INDEX indexed_id ON indexed (id)",
    )
    .await
    .expect("create the index");
    load(&server, "indexed", ROWS).await;
    assert_intact(&server, "indexed", ROWS).await;
}

/// Several tables loaded at once through one server, so the appenders are
/// contending for frames in the same pool and allocating pages in the same
/// disk manager at the same time.
///
/// This is the case a serial load cannot reach: two writers claiming page
/// ranges concurrently, and an eviction driven by one writer landing on a
/// page another writer is still filling
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_c_concurrent_loads_through_one_pool_lose_nothing() {
    let (server, _schema, _tmp) = create_test_server_with_pool_frames(POOL_FRAMES).await;
    let mut session = new_session();
    let tables = ["t0", "t1", "t2", "t3"];
    for table in tables {
        exec_ddl(
            &server,
            &mut session,
            &format!("CREATE TABLE {table} (id INT, body TEXT)"),
        )
        .await
        .expect("create the table");
    }

    // A quarter of the rows each, so the run costs what one table costs while
    // four writers share the pool
    let each = ROWS / 4;
    let mut set = tokio::task::JoinSet::new();
    for table in tables {
        let server = Arc::clone(&server);
        set.spawn(async move {
            load(&server, table, each).await;
        });
    }
    while let Some(joined) = set.join_next().await {
        joined.expect("a concurrent load panicked");
    }

    for table in tables {
        assert_intact(&server, table, each).await;
    }
}

/// Deleting most of a heap and reloading it drives the page-reuse path, where
/// an appender takes a page pruning reclaimed rather than growing the file.
/// A row written into a reused page is the one most likely to be missed by a
/// scan that bounds itself on the page count
#[tokio::test]
async fn test_d_rows_written_into_reclaimed_pages_are_still_found() {
    let (server, _schema, _tmp) = create_test_server_with_pool_frames(POOL_FRAMES).await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE churn (id INT, body TEXT)",
    )
    .await
    .expect("create the table");

    let first = ROWS / 2;
    load(&server, "churn", first).await;
    exec_dml(&server, "DELETE FROM churn WHERE id % 4 <> 0").await;
    let kept = (0..first).filter(|i| i % 4 == 0).count();
    let kept_sum: i64 = (0..first).filter(|i| i % 4 == 0).map(|i| i as i64).sum();

    let after_delete = scalar_i64(
        &query_values(&server, "SELECT count(*) FROM churn").await,
        "count(*)",
    );
    assert_eq!(
        after_delete, kept as i64,
        "the delete removed the wrong rows"
    );

    // A second load on top, with ids continuing past the first, which the
    // appender places into whatever pruning freed before it grows the file
    for sql in insert_statements("churn", first, first, PER_STATEMENT) {
        exec_dml(&server, &sql).await;
    }

    let total = scalar_i64(
        &query_values(&server, "SELECT count(*) FROM churn").await,
        "count(*)",
    );
    assert_eq!(
        total,
        (kept + first) as i64,
        "the reload into reclaimed pages did not keep every row"
    );
    let sum = scalar_i64(
        &query_values(&server, "SELECT sum(id) FROM churn").await,
        "sum(id)",
    );
    let reload_sum: i64 = (first..first * 2).map(|i| i as i64).sum();
    assert_eq!(
        sum,
        kept_sum + reload_sum,
        "the rows present after the reload are not the rows written"
    );
}

//! ORDER BY answered by reading a B+tree index in key order.
//!
//! A B+tree already holds its keys in order, so a query ordering by a
//! leading run of an index's key columns can read the index instead of
//! sorting. Descending is the same walk in reverse, which is why
//! `CREATE INDEX ... (a DESC)` and `(a ASC)` declare the same physical index
//! and both serve both directions.
//!
//! The order a query returns is part of its answer, so every test here
//! asserts the rows, not only the plan shape. The plan assertions exist so a
//! silent regression to sorting is caught as well.
//!
//! Run: cargo test -p zyron-wire --test index_order_test

mod common;

use common::{create_test_server, exec_ddl, exec_dml, new_session, query_values};
use zyron_executor::column::ScalarValue;
use zyron_planner::physical::{PhysicalPlan, ScanDirection};

async fn plan_of(server: &std::sync::Arc<zyron_wire::connection::ServerState>, sql: &str) -> PhysicalPlan {
    let stmt = zyron_parser::parse(sql)
        .expect("parse")
        .into_iter()
        .next()
        .expect("one statement");
    zyron_planner::plan(
        &server.catalog,
        zyron_catalog::DatabaseId(1),
        vec!["public".to_string()],
        stmt,
        None,
    )
    .await
    .expect("plan")
}

/// The single-child chain these queries plan into, top to bottom. Anything
/// with more than one input is not produced here, so a node that is not in
/// this list ends the walk.
fn chain(plan: &PhysicalPlan) -> Vec<&PhysicalPlan> {
    let mut out = vec![plan];
    let mut node = plan;
    loop {
        node = match node {
            PhysicalPlan::Project { child, .. }
            | PhysicalPlan::Filter { child, .. }
            | PhysicalPlan::Sort { child, .. }
            | PhysicalPlan::Limit { child, .. } => child.as_ref(),
            _ => break,
        };
        out.push(node);
    }
    out
}

fn find_index_scan(plan: &PhysicalPlan) -> Option<ScanDirection> {
    chain(plan).into_iter().find_map(|n| match n {
        PhysicalPlan::IndexScan { scan_direction, .. } => Some(*scan_direction),
        _ => None,
    })
}

fn has_sort(plan: &PhysicalPlan) -> bool {
    chain(plan)
        .into_iter()
        .any(|n| matches!(n, PhysicalPlan::Sort { .. }))
}

fn as_i64(rows: &[Vec<ScalarValue>]) -> Vec<i64> {
    rows.iter()
        .map(|r| match r.first() {
            Some(ScalarValue::Int64(v)) => *v,
            Some(ScalarValue::Int32(v)) => *v as i64,
            other => panic!("expected an integer, got {other:?}"),
        })
        .collect()
}

async fn indexed_table(
    server: &std::sync::Arc<zyron_wire::connection::ServerState>,
    session: &mut Option<zyron_wire::session::Session>,
    index_sql: &str,
) {
    exec_ddl(
        server,
        session,
        "CREATE TABLE t (k BIGINT NOT NULL, v BIGINT)",
    )
    .await
    .expect("create");
    exec_dml(server, "INSERT INTO t VALUES (3, 30), (1, 10), (4, 40), (2, 20)").await;
    exec_ddl(server, session, index_sql).await.expect("index");
}

/// Ascending order is a forward walk of the index and the sort is gone.
#[tokio::test]
async fn test_ascending_order_reads_the_index_forward() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    indexed_table(&server, &mut session, "CREATE INDEX ix ON t (k)").await;

    let plan = plan_of(&server, "SELECT k FROM t ORDER BY k").await;
    assert_eq!(find_index_scan(&plan), Some(ScanDirection::Forward));
    assert!(!has_sort(&plan), "the index already yields this order");

    assert_eq!(
        as_i64(&query_values(&server, "SELECT k FROM t ORDER BY k").await),
        vec![1, 2, 3, 4]
    );
}

/// Descending order is the same index read backward.
#[tokio::test]
async fn test_descending_order_reads_the_index_backward() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    indexed_table(&server, &mut session, "CREATE INDEX ix ON t (k)").await;

    let plan = plan_of(&server, "SELECT k FROM t ORDER BY k DESC").await;
    assert_eq!(find_index_scan(&plan), Some(ScanDirection::Backward));
    assert!(!has_sort(&plan));

    assert_eq!(
        as_i64(&query_values(&server, "SELECT k FROM t ORDER BY k DESC").await),
        vec![4, 3, 2, 1]
    );
}

/// A DESC-declared index is the same physical index as an ASC one, so it
/// serves both directions. This is what `CREATE INDEX ... (k DESC)` buys:
/// not different bytes, but an ORDER BY that runs without a sort.
#[tokio::test]
async fn test_a_desc_declared_index_serves_both_directions() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    indexed_table(&server, &mut session, "CREATE INDEX ix ON t (k DESC)").await;

    let desc = plan_of(&server, "SELECT k FROM t ORDER BY k DESC").await;
    assert_eq!(find_index_scan(&desc), Some(ScanDirection::Backward));
    assert!(!has_sort(&desc));
    assert_eq!(
        as_i64(&query_values(&server, "SELECT k FROM t ORDER BY k DESC").await),
        vec![4, 3, 2, 1]
    );

    let asc = plan_of(&server, "SELECT k FROM t ORDER BY k").await;
    assert_eq!(find_index_scan(&asc), Some(ScanDirection::Forward));
    assert!(!has_sort(&asc));
    assert_eq!(
        as_i64(&query_values(&server, "SELECT k FROM t ORDER BY k").await),
        vec![1, 2, 3, 4]
    );
}

/// A WHERE clause between the sort and the scan becomes the scan's residual
/// filter. Filtering keeps the rows in the order they arrived, so the sort
/// is still unnecessary.
#[tokio::test]
async fn test_a_filter_under_the_sort_keeps_the_index_order() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    indexed_table(&server, &mut session, "CREATE INDEX ix ON t (k)").await;

    let sql = "SELECT k FROM t WHERE k > 1 ORDER BY k DESC";
    let plan = plan_of(&server, sql).await;
    assert_eq!(find_index_scan(&plan), Some(ScanDirection::Backward));
    assert!(!has_sort(&plan));
    assert_eq!(as_i64(&query_values(&server, sql).await), vec![4, 3, 2]);
}

/// A row with a null in a key column is left out of the index entirely, so
/// ordering by a nullable column has to sort: an index scan would drop the
/// null rows rather than place them.
#[tokio::test]
async fn test_a_nullable_key_column_still_sorts() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(&server, &mut session, "CREATE TABLE n (k BIGINT, v BIGINT)")
        .await
        .expect("create");
    exec_dml(&server, "INSERT INTO n VALUES (2, 20), (1, 10)").await;
    exec_ddl(&server, &mut session, "CREATE INDEX nx ON n (k)")
        .await
        .expect("index");
    exec_dml(&server, "INSERT INTO n VALUES (NULL, 99)").await;

    let plan = plan_of(&server, "SELECT k FROM n ORDER BY k").await;
    assert!(
        has_sort(&plan),
        "a nullable key column cannot be answered from the index"
    );
    assert_eq!(
        query_values(&server, "SELECT k FROM n ORDER BY k").await.len(),
        3,
        "the null row is still returned"
    );
}

/// Ordering by a column the index does not lead with is not the index's
/// order, so the sort stays.
#[tokio::test]
async fn test_a_non_leading_column_still_sorts() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    indexed_table(&server, &mut session, "CREATE INDEX ix ON t (k)").await;

    let plan = plan_of(&server, "SELECT v FROM t ORDER BY v").await;
    assert!(has_sort(&plan));
    assert_eq!(
        as_i64(&query_values(&server, "SELECT v FROM t ORDER BY v").await),
        vec![10, 20, 30, 40]
    );
}

/// Terms running opposite ways cannot both come from one walk, so the sort
/// stays and the answer is still right.
#[tokio::test]
async fn test_mixed_order_by_directions_still_sort() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE m (a BIGINT NOT NULL, b BIGINT NOT NULL)",
    )
    .await
    .expect("create");
    exec_dml(&server, "INSERT INTO m VALUES (1, 1), (1, 2), (2, 1)").await;
    exec_ddl(&server, &mut session, "CREATE INDEX mx ON m (a, b)")
        .await
        .expect("index");

    let plan = plan_of(&server, "SELECT a FROM m ORDER BY a ASC, b DESC").await;
    assert!(has_sort(&plan), "one walk cannot produce two directions");
}

/// An index cannot store ascending and descending key columns at once, so
/// the declaration is refused rather than quietly created in one direction.
#[tokio::test]
async fn test_a_mixed_direction_index_is_refused() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE m (a BIGINT NOT NULL, b BIGINT NOT NULL)",
    )
    .await
    .expect("create");

    let err = exec_ddl(&server, &mut session, "CREATE INDEX mx ON m (a ASC, b DESC)")
        .await
        .expect_err("a mixed-direction index must be refused");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("ASC") && msg.contains("DESC"),
        "the refusal says what cannot be stored: {msg}"
    );
}

/// A lake table's index is a lake artifact committed into its own transaction
/// log, not a B+tree over heap addresses, so its catalog entry must not be
/// mistaken for one that can be walked in key order.
#[tokio::test]
async fn test_a_lake_table_does_not_get_a_heap_index_scan() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE l (k BIGINT NOT NULL, v BIGINT) USING ZYRONLAKE",
    )
    .await
    .expect("create lake table");
    exec_dml(&server, "INSERT INTO l VALUES (2, 20), (1, 10), (3, 30)").await;
    exec_ddl(&server, &mut session, "CREATE INDEX lx ON l (k)")
        .await
        .expect("index");

    let plan = plan_of(&server, "SELECT k FROM l ORDER BY k").await;
    assert!(
        find_index_scan(&plan).is_none(),
        "a lake table is read through its own scan"
    );
    assert_eq!(
        as_i64(&query_values(&server, "SELECT k FROM l ORDER BY k").await),
        vec![1, 2, 3]
    );
    assert_eq!(
        as_i64(&query_values(&server, "SELECT k FROM l ORDER BY k DESC").await),
        vec![3, 2, 1]
    );
}

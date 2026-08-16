//! B+tree index keys must be order-preserving for every indexable type.
//!
//! The tree compares keys as unsigned bytes, so every encoder has to map
//! values to bytes whose unsigned order equals the value order. A signed
//! integer cast straight to unsigned puts every negative above every
//! positive, which misorders index-order ORDER BY and makes range
//! predicates answered from index bounds drop or invent rows. Types with
//! no encoder arm at all leave their index permanently empty, so a scan
//! that trusts the index returns nothing from a populated table.
//!
//! Run: cargo test -p zyron-wire --test index_key_encoding_test

mod common;

use common::{create_test_server, exec_ddl, exec_dml, query_values};
use zyron_executor::column::ScalarValue;
use zyron_planner::physical::PhysicalPlan;

async fn plan_of(
    server: &std::sync::Arc<zyron_wire::connection::ServerState>,
    sql: &str,
) -> PhysicalPlan {
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

/// The single-child chain these queries plan into, top to bottom.
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

fn uses_index_scan(plan: &PhysicalPlan) -> bool {
    chain(plan)
        .into_iter()
        .any(|n| matches!(n, PhysicalPlan::IndexScan { .. }))
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
            Some(ScalarValue::Int16(v)) => *v as i64,
            other => panic!("expected an integer, got {other:?}"),
        })
        .collect()
}

/// Index-order ORDER BY over a signed BIGINT key returns negatives first.
#[tokio::test]
async fn test_signed_bigint_order_by_reads_index_in_value_order() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = common::new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE t (k BIGINT NOT NULL, v BIGINT)",
    )
    .await
    .expect("create");
    exec_dml(&server, "INSERT INTO t VALUES (-3, 30), (1, 10), (5, 50)").await;
    exec_ddl(&server, &mut session, "CREATE INDEX ix ON t (k)")
        .await
        .expect("index");

    let sql = "SELECT k FROM t ORDER BY k";
    let plan = plan_of(&server, sql).await;
    assert!(uses_index_scan(&plan), "ORDER BY should read the index");
    assert!(!has_sort(&plan), "the index already yields this order");
    assert_eq!(as_i64(&query_values(&server, sql).await), vec![-3, 1, 5]);

    let desc = "SELECT k FROM t ORDER BY k DESC";
    let plan = plan_of(&server, desc).await;
    assert!(
        uses_index_scan(&plan),
        "DESC should read the index backward"
    );
    assert_eq!(as_i64(&query_values(&server, desc).await), vec![5, 1, -3]);
}

/// Range predicates answered from index bounds keep negative rows.
#[tokio::test]
async fn test_signed_bigint_range_predicates_via_index() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = common::new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE t (k BIGINT NOT NULL, v BIGINT)",
    )
    .await
    .expect("create");
    exec_dml(
        &server,
        "INSERT INTO t VALUES (-4, 1), (-1, 2), (0, 3), (2, 4), (5, 5)",
    )
    .await;
    exec_ddl(&server, &mut session, "CREATE INDEX ix ON t (k)")
        .await
        .expect("index");

    let cases: Vec<(&str, Vec<i64>)> = vec![
        ("SELECT k FROM t WHERE k < 5 ORDER BY k", vec![-4, -1, 0, 2]),
        (
            "SELECT k FROM t WHERE k > -5 ORDER BY k",
            vec![-4, -1, 0, 2, 5],
        ),
        (
            "SELECT k FROM t WHERE k >= -1 ORDER BY k",
            vec![-1, 0, 2, 5],
        ),
        ("SELECT k FROM t WHERE k <= 0 ORDER BY k", vec![-4, -1, 0]),
        (
            "SELECT k FROM t WHERE k > -2 AND k < 3 ORDER BY k",
            vec![-1, 0, 2],
        ),
        ("SELECT k FROM t WHERE k = -1", vec![-1]),
    ];
    for (sql, expected) in cases {
        let plan = plan_of(&server, sql).await;
        assert!(uses_index_scan(&plan), "expected an index scan for {sql}");
        assert_eq!(
            as_i64(&query_values(&server, sql).await),
            expected,
            "wrong rows for {sql}"
        );
    }
}

/// The narrower signed widths share the encoder, so INT keys misorder the
/// same way BIGINT keys do.
#[tokio::test]
async fn test_signed_int32_keys_via_index() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = common::new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE t (k INT NOT NULL, v BIGINT)",
    )
    .await
    .expect("create");
    exec_dml(&server, "INSERT INTO t VALUES (-3, 1), (1, 2), (5, 3)").await;
    exec_ddl(&server, &mut session, "CREATE INDEX ix ON t (k)")
        .await
        .expect("index");

    let sql = "SELECT k FROM t ORDER BY k";
    let plan = plan_of(&server, sql).await;
    assert!(uses_index_scan(&plan), "ORDER BY should read the index");
    assert_eq!(as_i64(&query_values(&server, sql).await), vec![-3, 1, 5]);

    let range = "SELECT k FROM t WHERE k < 5 ORDER BY k";
    let plan = plan_of(&server, range).await;
    assert!(uses_index_scan(&plan), "expected an index scan for {range}");
    assert_eq!(as_i64(&query_values(&server, range).await), vec![-3, 1]);
}

/// A UUID index must serve rows: raw big-endian bytes are already order
/// preserving for the 16-byte value.
#[tokio::test]
async fn test_uuid_index_scan_returns_rows() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = common::new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE u (id UUID NOT NULL, v BIGINT)",
    )
    .await
    .expect("create");
    exec_dml(
        &server,
        "INSERT INTO u VALUES \
         (CAST('00000000000000000000000000000003' AS UUID), 3), \
         (CAST('00000000000000000000000000000001' AS UUID), 1), \
         (CAST('00000000000000000000000000000002' AS UUID), 2)",
    )
    .await;
    exec_ddl(&server, &mut session, "CREATE INDEX ix ON u (id)")
        .await
        .expect("index");

    let sql = "SELECT v FROM u ORDER BY id";
    let plan = plan_of(&server, sql).await;
    assert!(uses_index_scan(&plan), "ORDER BY should read the index");
    assert_eq!(as_i64(&query_values(&server, sql).await), vec![1, 2, 3]);
}

/// A boolean index must serve rows, false ordering below true.
#[tokio::test]
async fn test_boolean_index_scan_returns_rows() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = common::new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE b (flag BOOLEAN NOT NULL, v BIGINT)",
    )
    .await
    .expect("create");
    exec_dml(
        &server,
        "INSERT INTO b VALUES (true, 1), (false, 2), (true, 3)",
    )
    .await;
    exec_ddl(&server, &mut session, "CREATE INDEX ix ON b (flag)")
        .await
        .expect("index");

    let sql = "SELECT v FROM b ORDER BY flag";
    let plan = plan_of(&server, sql).await;
    assert!(uses_index_scan(&plan), "ORDER BY should read the index");
    let rows = query_values(&server, sql).await;
    assert_eq!(rows.len(), 3, "index scan dropped rows");
    let flags: Vec<i64> = as_i64(&rows);
    assert_eq!(flags[0], 2, "false rows order first");
    assert_eq!(
        {
            let mut rest = flags[1..].to_vec();
            rest.sort();
            rest
        },
        vec![1, 3]
    );
}

/// INTERVAL has no order-preserving fixed encoding, so indexing one is
/// refused instead of building a tree that can never hold the rows.
#[tokio::test]
async fn test_interval_index_refused() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = common::new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE iv (d INTERVAL NOT NULL, v BIGINT)",
    )
    .await
    .expect("create");
    let result = exec_ddl(&server, &mut session, "CREATE INDEX ix ON iv (d)").await;
    assert!(
        result.is_err(),
        "CREATE INDEX on an INTERVAL column must be refused"
    );
}

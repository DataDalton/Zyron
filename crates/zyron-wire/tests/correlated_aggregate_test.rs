//! A correlated scalar aggregate answered from one grouped run.
//!
//! Run: cargo test -p zyron-wire --test correlated_aggregate_test
//!
//! `(SELECT MAX(x) FROM inner WHERE inner.k = outer.k)` asks for one group
//! of `SELECT k, MAX(x) FROM inner GROUP BY k`, and a different group per
//! outer row. Running the grouped form once answers every row at the cost
//! of one, and the whole risk is in what a missing group means.
//!
//! An outer row whose group the run never produced is a subquery over no
//! rows at all. Every aggregate but one is NULL there. `COUNT` is zero,
//! and a rewrite that returned NULL for it would be wrong in a way no
//! amount of matching rows would reveal.

use std::sync::Arc;

use zyron_executor::column::ScalarValue;
use zyron_wire::connection::ServerState;

mod common;
use common::{create_test_server, exec_ddl, exec_dml, new_session, query_values};

/// Customer 1 has three orders, 2 and 3 and 6 have one each, 5 has none,
/// and 4's key is NULL so nothing can equal it. Order 90's customer is
/// NULL and belongs to no group
async fn seeded() -> (Arc<ServerState>, tempfile::TempDir) {
    let (server, _schema_id, tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE cust (id BIGINT NOT NULL, key BIGINT, region BIGINT)",
    )
    .await
    .expect("create cust");
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE ord (id BIGINT NOT NULL, cust_key BIGINT, region BIGINT, amount BIGINT)",
    )
    .await
    .expect("create ord");
    exec_dml(
        &server,
        "INSERT INTO cust VALUES \
         (1, 10, 100), (2, 20, 100), (3, 30, 200), \
         (4, NULL, 100), (5, 50, 200), (6, 60, 100)",
    )
    .await;
    exec_dml(
        &server,
        "INSERT INTO ord VALUES \
         (11, 10, 100, 5), (12, 10, 100, 7), (13, 10, 200, 9), \
         (21, 20, 999, 3), (31, 30, 200, 4), (61, 60, 100, 6), \
         (90, NULL, 100, 1)",
    )
    .await;
    (server, tmp)
}

/// The second column of each row rendered, so a NULL is distinguishable
/// from a zero rather than both reading as absent
async fn per_customer(server: &Arc<ServerState>, sql: &str) -> Vec<String> {
    query_values(server, sql)
        .await
        .into_iter()
        .map(|row| format!("{:?}", row.get(1)))
        .collect()
}

async fn ids(server: &Arc<ServerState>, sql: &str) -> Vec<i64> {
    query_values(server, sql)
        .await
        .into_iter()
        .filter_map(|row| match row.first() {
            Some(ScalarValue::Int64(v)) => Some(*v),
            _ => None,
        })
        .collect()
}

/// A count over no rows is zero. Customers 4 and 5 have no matching order
/// and must read zero, not NULL
#[tokio::test]
async fn a_count_over_an_empty_group_is_zero_not_null() {
    let (server, _tmp) = seeded().await;
    let got = per_customer(
        &server,
        "SELECT c.id, (SELECT COUNT(*) FROM ord o WHERE o.cust_key = c.key) \
         FROM cust c ORDER BY c.id",
    )
    .await;
    assert_eq!(
        got,
        vec![
            "Some(Int64(3))",
            "Some(Int64(1))",
            "Some(Int64(1))",
            "Some(Int64(0))",
            "Some(Int64(0))",
            "Some(Int64(1))"
        ],
        "customer 4's key is NULL and customer 5 has no orders, and both count zero"
    );
}

/// Every other aggregate over no rows is NULL
#[tokio::test]
async fn an_extremum_over_an_empty_group_is_null() {
    let (server, _tmp) = seeded().await;
    let got = per_customer(
        &server,
        "SELECT c.id, (SELECT MAX(o.amount) FROM ord o WHERE o.cust_key = c.key) \
         FROM cust c ORDER BY c.id",
    )
    .await;
    assert_eq!(
        got,
        vec![
            "Some(Int64(9))",
            "Some(Int64(3))",
            "Some(Int64(4))",
            "Some(Null)",
            "Some(Null)",
            "Some(Int64(6))"
        ]
    );
}

#[tokio::test]
async fn a_sum_over_an_empty_group_is_null() {
    let (server, _tmp) = seeded().await;
    let got = per_customer(
        &server,
        "SELECT c.id, (SELECT SUM(o.amount) FROM ord o WHERE o.cust_key = c.key) \
         FROM cust c ORDER BY c.id",
    )
    .await;
    assert_eq!(
        got,
        vec![
            "Some(Int64(21))",
            "Some(Int64(3))",
            "Some(Int64(4))",
            "Some(Null)",
            "Some(Null)",
            "Some(Int64(6))"
        ]
    );
}

/// The same subquery in a predicate rather than a projection
#[tokio::test]
async fn a_correlated_aggregate_in_a_predicate_filters_the_same_rows() {
    let (server, _tmp) = seeded().await;
    let more_than_one = ids(
        &server,
        "SELECT c.id FROM cust c \
         WHERE (SELECT COUNT(*) FROM ord o WHERE o.cust_key = c.key) > 1 ORDER BY c.id",
    )
    .await;
    assert_eq!(more_than_one, vec![1]);

    // Zero compares, so the customers with no orders are found by it
    let none = ids(
        &server,
        "SELECT c.id FROM cust c \
         WHERE (SELECT COUNT(*) FROM ord o WHERE o.cust_key = c.key) = 0 ORDER BY c.id",
    )
    .await;
    assert_eq!(none, vec![4, 5]);
}

/// Two equalities group by both, and a row reads only the group where both
/// agree. Customer 2's only order is in another region
#[tokio::test]
async fn a_composite_correlation_groups_by_every_key() {
    let (server, _tmp) = seeded().await;
    let got = per_customer(
        &server,
        "SELECT c.id, (SELECT COUNT(*) FROM ord o \
         WHERE o.cust_key = c.key AND o.region = c.region) FROM cust c ORDER BY c.id",
    )
    .await;
    assert_eq!(
        got,
        vec![
            "Some(Int64(2))",
            "Some(Int64(0))",
            "Some(Int64(1))",
            "Some(Int64(0))",
            "Some(Int64(0))",
            "Some(Int64(1))"
        ],
        "customer 1 has two orders in region 100 and one in region 200"
    );
}

/// A term inside the subquery narrows the rows each group is taken over
#[tokio::test]
async fn a_term_inside_the_subquery_narrows_each_group() {
    let (server, _tmp) = seeded().await;
    let got = per_customer(
        &server,
        "SELECT c.id, (SELECT COUNT(*) FROM ord o \
         WHERE o.cust_key = c.key AND o.amount >= 6) FROM cust c ORDER BY c.id",
    )
    .await;
    assert_eq!(
        got,
        vec![
            "Some(Int64(2))",
            "Some(Int64(0))",
            "Some(Int64(0))",
            "Some(Int64(0))",
            "Some(Int64(0))",
            "Some(Int64(1))"
        ]
    );
}

/// An aggregate wrapped in arithmetic. The subquery is still the whole
/// projection item's inner node, and the expression around it is evaluated
/// on the value the group produced
#[tokio::test]
async fn an_expression_around_the_aggregate_still_reads_its_group() {
    let (server, _tmp) = seeded().await;
    let got = per_customer(
        &server,
        "SELECT c.id, (SELECT MAX(o.amount) FROM ord o WHERE o.cust_key = c.key) + 100 \
         FROM cust c ORDER BY c.id",
    )
    .await;
    assert_eq!(
        got,
        vec![
            "Some(Int64(109))",
            "Some(Int64(103))",
            "Some(Int64(104))",
            "Some(Null)",
            "Some(Null)",
            "Some(Int64(106))"
        ],
        "NULL plus a hundred is still NULL"
    );
}

/// Shapes the rewrite refuses, each answered by the per row path and each
/// expectation worked out by hand
#[tokio::test]
async fn shapes_the_rewrite_refuses_still_answer_correctly() {
    let (server, _tmp) = seeded().await;

    // A LIMIT caps what one grouped run would see
    let limited = per_customer(
        &server,
        "SELECT c.id, (SELECT o.amount FROM ord o WHERE o.cust_key = c.key LIMIT 1) \
         FROM cust c ORDER BY c.id",
    )
    .await;
    assert_eq!(limited.len(), 6);
    assert_eq!(
        limited[3], "Some(Null)",
        "customer 4's key matches no order"
    );

    // A non-equality correlation cannot become a grouping key
    let inequality = per_customer(
        &server,
        "SELECT c.id, (SELECT COUNT(*) FROM ord o WHERE o.cust_key > c.key) \
         FROM cust c ORDER BY c.id",
    )
    .await;
    assert_eq!(
        inequality,
        vec![
            "Some(Int64(3))",
            "Some(Int64(2))",
            "Some(Int64(1))",
            "Some(Int64(0))",
            "Some(Int64(1))",
            "Some(Int64(0))"
        ],
        "the order keys above ten are twenty, thirty and sixty, and a NULL outer key \
         compares true against nothing so its count is zero"
    );

    // No correlation at all is an uncorrelated subquery, folded once to a
    // constant before any row is evaluated
    let uncorrelated = per_customer(
        &server,
        "SELECT c.id, (SELECT COUNT(*) FROM ord o) FROM cust c ORDER BY c.id",
    )
    .await;
    assert_eq!(uncorrelated, vec!["Some(Int64(7))"; 6]);
}

/// An UPDATE whose SET value is a correlated aggregate reads the same
/// groups a SELECT would
#[tokio::test]
async fn an_update_set_from_a_correlated_aggregate_writes_the_same_values() {
    let (server, _tmp) = seeded().await;
    exec_dml(
        &server,
        "UPDATE cust SET region = (SELECT COUNT(*) FROM ord o WHERE o.cust_key = cust.key)",
    )
    .await;
    let regions = query_values(&server, "SELECT id, region FROM cust ORDER BY id")
        .await
        .into_iter()
        .map(|row| format!("{:?}", row.get(1)))
        .collect::<Vec<_>>();
    assert_eq!(
        regions,
        vec![
            "Some(Int64(3))",
            "Some(Int64(1))",
            "Some(Int64(1))",
            "Some(Int64(0))",
            "Some(Int64(0))",
            "Some(Int64(1))"
        ]
    );
}

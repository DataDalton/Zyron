//! Correlated EXISTS answered as a hash semi join.
//!
//! Run: cargo test -p zyron-wire --test semi_join_test
//!
//! The rewrite runs the subquery once and probes its keys, instead of
//! running it per outer row. That is only sound where an equality decides
//! the match, so these tests cover the two things a semi join gets wrong:
//! NULL keys, which equal nothing and so match nothing, and duplicate
//! inner keys, which must not multiply an outer row the way a plain join
//! would.
//!
//! Shapes the rewrite refuses are here too. They keep the per row path and
//! have to keep answering correctly, so each one is checked against the
//! answer worked out by hand rather than against the other path.

use std::sync::Arc;

use zyron_executor::column::ScalarValue;
use zyron_wire::connection::ServerState;

mod common;
use common::{create_test_server, exec_ddl, exec_dml, new_session, query_values};

/// Customers 1..6 and orders against some of them.
///
/// Customer 4 has a NULL key, so nothing can equal it. Customer 5 has no
/// orders. Customer 1 has three orders, which is what says a semi join
/// returns it once rather than three times. Order 90 carries a NULL
/// customer, which must match no one
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

/// Customer 1 has three matching orders and must come back once
#[tokio::test]
async fn a_semi_join_returns_an_outer_row_once_however_many_rows_match() {
    let (server, _tmp) = seeded().await;
    let got = ids(
        &server,
        "SELECT c.id FROM cust c WHERE EXISTS \
         (SELECT 1 FROM ord o WHERE o.cust_key = c.key) ORDER BY c.id",
    )
    .await;
    assert_eq!(got, vec![1, 2, 3, 6]);
}

/// Customer 4's key is NULL, so no equality against it is ever true. It is
/// absent from EXISTS and present in NOT EXISTS, and order 90's NULL key
/// matches nobody either
#[tokio::test]
async fn a_null_key_matches_nothing_on_either_side() {
    let (server, _tmp) = seeded().await;
    let present = ids(
        &server,
        "SELECT c.id FROM cust c WHERE EXISTS \
         (SELECT 1 FROM ord o WHERE o.cust_key = c.key) ORDER BY c.id",
    )
    .await;
    assert!(
        !present.contains(&4),
        "a NULL outer key cannot equal anything, so EXISTS is false for it"
    );

    let absent = ids(
        &server,
        "SELECT c.id FROM cust c WHERE NOT EXISTS \
         (SELECT 1 FROM ord o WHERE o.cust_key = c.key) ORDER BY c.id",
    )
    .await;
    assert_eq!(
        absent,
        vec![4, 5],
        "customer 4 has no match because its key is NULL, customer 5 because it has no orders"
    );
}

/// Two equalities in the correlation make a composite key, and a row
/// matches only when both sides agree. Customer 2's only order sits in a
/// different region, so it drops out of a pairing it was in before
#[tokio::test]
async fn a_composite_correlation_matches_on_every_equality() {
    let (server, _tmp) = seeded().await;
    let got = ids(
        &server,
        "SELECT c.id FROM cust c WHERE EXISTS \
         (SELECT 1 FROM ord o WHERE o.cust_key = c.key AND o.region = c.region) \
         ORDER BY c.id",
    )
    .await;
    assert_eq!(got, vec![1, 3, 6]);
}

/// A term beside the EXISTS is still applied, and applies to the same rows
#[tokio::test]
async fn an_ordinary_term_beside_the_exists_still_filters() {
    let (server, _tmp) = seeded().await;
    let got = ids(
        &server,
        "SELECT c.id FROM cust c WHERE c.region = 100 AND EXISTS \
         (SELECT 1 FROM ord o WHERE o.cust_key = c.key) ORDER BY c.id",
    )
    .await;
    assert_eq!(got, vec![1, 2, 6]);
}

/// Two lifted subqueries in one predicate, one of each polarity
#[tokio::test]
async fn two_exists_terms_in_one_predicate_are_both_applied() {
    let (server, _tmp) = seeded().await;
    let got = ids(
        &server,
        "SELECT c.id FROM cust c WHERE EXISTS \
         (SELECT 1 FROM ord o WHERE o.cust_key = c.key) \
         AND NOT EXISTS (SELECT 1 FROM ord o2 WHERE o2.cust_key = c.key AND o2.amount > 6) \
         ORDER BY c.id",
    )
    .await;
    assert_eq!(
        got,
        vec![2, 3, 6],
        "customer 1 has an order over six and is excluded by the second term"
    );
}

/// An inner side that matches nothing leaves every outer row to NOT EXISTS
#[tokio::test]
async fn an_inner_side_matching_nothing_keeps_every_row_for_not_exists() {
    let (server, _tmp) = seeded().await;
    let got = ids(
        &server,
        "SELECT c.id FROM cust c WHERE NOT EXISTS \
         (SELECT 1 FROM ord o WHERE o.cust_key = c.key AND o.amount > 1000) ORDER BY c.id",
    )
    .await;
    assert_eq!(got, vec![1, 2, 3, 4, 5, 6]);
}

/// A residual term inside the subquery narrows the keys it produces
#[tokio::test]
async fn a_term_inside_the_subquery_narrows_the_keys() {
    let (server, _tmp) = seeded().await;
    let got = ids(
        &server,
        "SELECT c.id FROM cust c WHERE EXISTS \
         (SELECT 1 FROM ord o WHERE o.cust_key = c.key AND o.amount >= 6) ORDER BY c.id",
    )
    .await;
    assert_eq!(
        got,
        vec![1, 6],
        "customer 3's only order is under the cutoff, so its key is not among the ones produced"
    );
}

/// Shapes the lift refuses. Each keeps the per row path and each answer is
/// worked out by hand, so a rewrite that widened its own conditions would
/// be caught here rather than silently changing results
#[tokio::test]
async fn shapes_the_lift_refuses_still_answer_correctly() {
    let (server, _tmp) = seeded().await;

    // A non-equality correlation. A hash set cannot answer `>`, so this
    // has to stay on the per row path
    let inequality = ids(
        &server,
        "SELECT c.id FROM cust c WHERE EXISTS \
         (SELECT 1 FROM ord o WHERE o.cust_key > c.key) ORDER BY c.id",
    )
    .await;
    assert_eq!(
        inequality,
        vec![1, 2, 3, 5],
        "every customer with a key below the largest order key of sixty"
    );

    // The correlation sits under an OR, so no single term can be lifted
    let disjunction = ids(
        &server,
        "SELECT c.id FROM cust c WHERE EXISTS \
         (SELECT 1 FROM ord o WHERE o.cust_key = c.key OR o.amount = 1) ORDER BY c.id",
    )
    .await;
    assert_eq!(
        disjunction,
        vec![1, 2, 3, 4, 5, 6],
        "order 90 has amount one, so the disjunction is true for every customer"
    );

    // An aggregate makes the subquery produce a row whether or not
    // anything matched, so EXISTS is true for every outer row
    let aggregated = ids(
        &server,
        "SELECT c.id FROM cust c WHERE EXISTS \
         (SELECT COUNT(*) FROM ord o WHERE o.cust_key = c.key) ORDER BY c.id",
    )
    .await;
    assert_eq!(aggregated, vec![1, 2, 3, 4, 5, 6]);

    // A LIMIT caps what one lifted run would see
    let limited = ids(
        &server,
        "SELECT c.id FROM cust c WHERE EXISTS \
         (SELECT 1 FROM ord o WHERE o.cust_key = c.key LIMIT 1) ORDER BY c.id",
    )
    .await;
    assert_eq!(limited, vec![1, 2, 3, 6]);

    // No correlation at all is an uncorrelated subquery, folded to a
    // constant before the evaluator runs
    let uncorrelated = ids(
        &server,
        "SELECT c.id FROM cust c WHERE EXISTS (SELECT 1 FROM ord o WHERE o.amount > 100) \
         ORDER BY c.id",
    )
    .await;
    assert_eq!(uncorrelated, Vec::<i64>::new());
}

/// An uncorrelated subquery sitting beside a lifted one.
///
/// What is left of the predicate is handed to the synchronous expression
/// evaluator, which cannot run a plan, so a subquery left in it has to be
/// folded to a constant first. TPC-H Q22 is exactly this shape, an
/// uncorrelated average beside a correlated NOT EXISTS
#[tokio::test]
async fn an_uncorrelated_subquery_beside_a_lifted_one_is_folded_first() {
    let (server, _tmp) = seeded().await;
    let got = ids(
        &server,
        "SELECT c.id FROM cust c \
         WHERE c.key > (SELECT MIN(o.cust_key) FROM ord o) \
         AND NOT EXISTS (SELECT 1 FROM ord o2 WHERE o2.cust_key = c.key) \
         ORDER BY c.id",
    )
    .await;
    assert_eq!(
        got,
        vec![5],
        "customer 5's key is above the smallest order key and has no order of its own"
    );
}

/// The rewrite must not change what a DELETE removes, which reads the same
/// predicate to decide its rows
#[tokio::test]
async fn a_delete_through_a_correlated_exists_removes_the_same_rows() {
    let (server, _tmp) = seeded().await;
    exec_dml(
        &server,
        "DELETE FROM cust WHERE NOT EXISTS (SELECT 1 FROM ord o WHERE o.cust_key = cust.key)",
    )
    .await;
    let left = ids(&server, "SELECT id FROM cust ORDER BY id").await;
    assert_eq!(
        left,
        vec![1, 2, 3, 6],
        "the two customers with no matching order are the two that go"
    );
}

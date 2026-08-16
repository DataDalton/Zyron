//! The in-memory sort must order signed integers correctly.
//!
//! The radix sort maps values to unsigned keys. Sign-extending a narrow
//! signed integer to u64 and flipping only the narrow sign bit leaves the
//! high bits set on negatives, so every negative sorts above every
//! positive and ORDER BY returns positives first. The correct transform
//! truncates to the unsigned twin before widening. These queries create
//! no index, so the Sort operator itself must produce the order.
//!
//! Run: cargo test -p zyron-wire --test sort_order_test

mod common;

use common::{create_test_server, exec_ddl, exec_dml, query_values};
use zyron_executor::column::ScalarValue;

fn first_col_as_i64(rows: &[Vec<ScalarValue>]) -> Vec<i64> {
    rows.iter()
        .map(|r| match r.first() {
            Some(ScalarValue::Int64(v)) => *v,
            Some(ScalarValue::Int32(v)) => *v as i64,
            Some(ScalarValue::Int16(v)) => *v as i64,
            other => panic!("expected an integer, got {other:?}"),
        })
        .collect()
}

/// Multi-column select forces the pair radix path for the INT key. This
/// path has no size threshold, so a handful of rows exercises it.
#[tokio::test]
async fn test_multi_column_order_by_int_puts_negatives_first() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = common::new_session();
    exec_ddl(&server, &mut session, "CREATE TABLE t (a INT, b BIGINT)")
        .await
        .expect("create");
    exec_dml(
        &server,
        "INSERT INTO t VALUES (5, 1), (-3, 2), (0, 3), (-250, 4), (17, 5), (-1, 6), (2, 7)",
    )
    .await;

    let rows = query_values(&server, "SELECT a, b FROM t ORDER BY a").await;
    assert_eq!(first_col_as_i64(&rows), vec![-250, -3, -1, 0, 2, 5, 17]);

    let rows = query_values(&server, "SELECT a, b FROM t ORDER BY a DESC").await;
    assert_eq!(first_col_as_i64(&rows), vec![17, 5, 2, 0, -1, -3, -250]);
}

/// Single-column select over enough rows to cross the radix threshold,
/// covering the values-only fused path for a narrow signed type.
#[tokio::test]
async fn test_single_column_order_by_int_sorts_mixed_signs_above_the_radix_threshold() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = common::new_session();
    exec_ddl(&server, &mut session, "CREATE TABLE t (a INT)")
        .await
        .expect("create");

    let mut expect: Vec<i64> = Vec::new();
    let mut tuples: Vec<String> = Vec::new();
    for i in 0..600i64 {
        let v = ((i * 37) % 501) - 250;
        expect.push(v);
        tuples.push(format!("({v})"));
    }
    exec_dml(
        &server,
        &format!("INSERT INTO t VALUES {}", tuples.join(", ")),
    )
    .await;
    expect.sort_unstable();

    let rows = query_values(&server, "SELECT a FROM t ORDER BY a").await;
    assert_eq!(first_col_as_i64(&rows), expect);

    let rows = query_values(&server, "SELECT a FROM t ORDER BY a DESC").await;
    let mut expect_desc = expect.clone();
    expect_desc.reverse();
    assert_eq!(first_col_as_i64(&rows), expect_desc);

    let rows = query_values(&server, "SELECT a FROM t ORDER BY a LIMIT 5").await;
    assert_eq!(first_col_as_i64(&rows), expect[..5].to_vec());
}

/// SMALLINT goes through the 16 bit signed transform, and BIGINT guards
/// the top-bit case that was already correct.
#[tokio::test]
async fn test_smallint_and_bigint_order_by_agree_with_a_comparison_oracle() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = common::new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE t (s SMALLINT, g BIGINT)",
    )
    .await
    .expect("create");

    let mut small: Vec<i64> = Vec::new();
    let mut big: Vec<i64> = Vec::new();
    let mut tuples: Vec<String> = Vec::new();
    for i in 0..300i64 {
        let s = ((i * 91) % 401) - 200;
        let g = ((i * 137) % 1001 - 500) * 4_000_000_000;
        small.push(s);
        big.push(g);
        tuples.push(format!("({s}, {g})"));
    }
    exec_dml(
        &server,
        &format!("INSERT INTO t VALUES {}", tuples.join(", ")),
    )
    .await;

    small.sort_unstable();
    let rows = query_values(&server, "SELECT s, g FROM t ORDER BY s").await;
    assert_eq!(first_col_as_i64(&rows), small);

    big.sort_unstable();
    let rows = query_values(&server, "SELECT g FROM t ORDER BY g").await;
    assert_eq!(first_col_as_i64(&rows), big);
}

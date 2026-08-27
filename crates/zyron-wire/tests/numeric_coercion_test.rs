//! Comparisons whose two sides arrive at different numeric widths.
//!
//! An INT column stores Int32 values while an integer literal evaluates as
//! Int64. Every comparison site has to align the pair to a common type
//! before comparing: the ScalarValue fallback compares distinct enum
//! variants as never-equal, so a site that skips coercion silently returns
//! zero rows (or its inverse returns every row). BETWEEN and the simple
//! CASE form each had that gap, and a compound query's trailing ORDER BY
//! and LIMIT applied to only the first UNION branch.
//!
//! Run: cargo test -p zyron-wire --test numeric_coercion_test

mod common;

use common::{create_test_server, exec_ddl, exec_dml, new_session, query_values};
use zyron_executor::column::ScalarValue;

fn ints(rows: &[Vec<ScalarValue>]) -> Vec<i64> {
    rows.iter()
        .map(|r| match r.first() {
            Some(ScalarValue::Int8(v)) => *v as i64,
            Some(ScalarValue::Int16(v)) => *v as i64,
            Some(ScalarValue::Int32(v)) => *v as i64,
            Some(ScalarValue::Int64(v)) => *v,
            other => panic!("expected an integer, got {other:?}"),
        })
        .collect()
}

fn texts(rows: &[Vec<ScalarValue>]) -> Vec<String> {
    rows.iter()
        .map(|r| match r.first() {
            Some(ScalarValue::Utf8(s)) => s.clone(),
            other => panic!("expected text, got {other:?}"),
        })
        .collect()
}

/// BETWEEN on an INT column against integer literals selects the range.
#[tokio::test]
async fn test_between_on_int_column() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(&server, &mut session, "CREATE TABLE t (v INT)")
        .await
        .expect("create");
    exec_dml(
        &server,
        "INSERT INTO t VALUES (1), (2), (3), (4), (5), (6), (7), (8)",
    )
    .await;

    let rows = query_values(
        &server,
        "SELECT v FROM t WHERE v BETWEEN 3 AND 5 ORDER BY v",
    )
    .await;
    assert_eq!(ints(&rows), vec![3, 4, 5], "BETWEEN selects the range");

    let rows = query_values(
        &server,
        "SELECT v FROM t WHERE v NOT BETWEEN 3 AND 5 ORDER BY v",
    )
    .await;
    assert_eq!(
        ints(&rows),
        vec![1, 2, 6, 7, 8],
        "NOT BETWEEN selects the complement"
    );
}

/// The simple CASE form matches an INT column against integer WHEN values.
#[tokio::test]
async fn test_simple_case_on_int_column() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(&server, &mut session, "CREATE TABLE t (k INT)")
        .await
        .expect("create");
    exec_dml(&server, "INSERT INTO t VALUES (1), (2), (3)").await;

    let rows = query_values(
        &server,
        "SELECT CASE k WHEN 2 THEN 'two' ELSE 'other' END FROM t ORDER BY k",
    )
    .await;
    assert_eq!(
        texts(&rows),
        vec!["other".to_string(), "two".to_string(), "other".to_string()],
        "the WHEN value matches its row instead of always taking ELSE"
    );
}

/// A multi-branch CASE over decimals keeps its scale through the merge, so
/// each branch's value compares equal to the same literal afterward.
#[tokio::test]
async fn test_case_decimal_branches_keep_scale() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(&server, &mut session, "CREATE TABLE t (k INT)")
        .await
        .expect("create");
    exec_dml(&server, "INSERT INTO t VALUES (1), (2), (3)").await;

    let rows = query_values(
        &server,
        "SELECT k FROM t WHERE (CASE WHEN k = 1 THEN CAST(1.50 AS DECIMAL(10,2)) \
         WHEN k = 2 THEN CAST(2.25 AS DECIMAL(10,2)) \
         ELSE CAST(0.00 AS DECIMAL(10,2)) END) = CAST(2.25 AS DECIMAL(10,2))",
    )
    .await;
    assert_eq!(
        ints(&rows),
        vec![2],
        "the second WHEN branch keeps 2.25 instead of a rescaled value"
    );
}

/// A scalar subquery over a DECIMAL column folds back onto its own scale,
/// so equality against the source column still matches the row it came
/// from instead of comparing a rescaled constant.
#[tokio::test]
async fn test_scalar_subquery_decimal_keeps_scale() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE p (id INT, price DECIMAL(10,2))",
    )
    .await
    .expect("create");
    exec_dml(&server, "INSERT INTO p VALUES (1, 10.50), (2, 7.25)").await;

    let rows = query_values(
        &server,
        "SELECT id FROM p WHERE price = (SELECT MAX(price) FROM p)",
    )
    .await;
    assert_eq!(
        ints(&rows),
        vec![1],
        "the folded maximum compares on the column's scale"
    );
}

/// A correlated scalar subquery over a DECIMAL column carries its scale
/// through the per-row parameter channel, so equality against the outer
/// column matches instead of comparing a raw scaled integer.
#[tokio::test]
async fn test_correlated_scalar_subquery_decimal_keeps_scale() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE o (id INT, grp INT, price DECIMAL(10,2))",
    )
    .await
    .expect("create o");
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE d (grp INT, price DECIMAL(10,2))",
    )
    .await
    .expect("create d");
    exec_dml(
        &server,
        "INSERT INTO o VALUES (1, 1, 10.50), (2, 1, 7.25), (3, 2, 3.75)",
    )
    .await;
    exec_dml(&server, "INSERT INTO d VALUES (1, 10.50), (2, 2.00)").await;

    // The subquery correlates on grp, and its decimal result compares
    // against the outer decimal column
    let rows = query_values(
        &server,
        "SELECT id FROM o WHERE price = (SELECT MAX(price) FROM d WHERE d.grp = o.grp) \
         ORDER BY id",
    )
    .await;
    assert_eq!(
        ints(&rows),
        vec![1],
        "the correlated maximum compares on the column's scale"
    );
}

/// A correlated subquery that references an outer DECIMAL column binds the
/// value on its own scale, so the comparison inside the subquery matches.
#[tokio::test]
async fn test_correlated_outer_decimal_param_keeps_scale() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE o (id INT, price DECIMAL(10,2))",
    )
    .await
    .expect("create o");
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE d (id INT, price DECIMAL(10,2))",
    )
    .await
    .expect("create d");
    exec_dml(&server, "INSERT INTO o VALUES (1, 10.50), (2, 7.25)").await;
    exec_dml(&server, "INSERT INTO d VALUES (7, 10.50), (8, 2.00)").await;

    let rows = query_values(
        &server,
        "SELECT id FROM o WHERE EXISTS (SELECT 1 FROM d WHERE d.price = o.price) ORDER BY id",
    )
    .await;
    assert_eq!(
        ints(&rows),
        vec![1],
        "the outer decimal binds at its own scale inside the subquery"
    );
}

/// A correlated IN subquery over DECIMAL columns compares probe and list on
/// one scale.
#[tokio::test]
async fn test_correlated_in_subquery_decimal_membership() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE o (id INT, grp INT, price DECIMAL(10,2))",
    )
    .await
    .expect("create o");
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE d (grp INT, price DECIMAL(10,3))",
    )
    .await
    .expect("create d");
    exec_dml(
        &server,
        "INSERT INTO o VALUES (1, 1, 10.50), (2, 1, 7.25), (3, 2, 10.50)",
    )
    .await;
    exec_dml(&server, "INSERT INTO d VALUES (1, 10.500), (2, 2.000)").await;

    // The probe is DECIMAL(10,2) and the list DECIMAL(10,3), so membership
    // has to align the scales before comparing raw integers
    let rows = query_values(
        &server,
        "SELECT id FROM o WHERE price IN (SELECT price FROM d WHERE d.grp = o.grp) ORDER BY id",
    )
    .await;
    assert_eq!(
        ints(&rows),
        vec![1],
        "membership compares values, not raw scaled integers"
    );
}

/// A compound query's trailing ORDER BY and LIMIT govern the whole set
/// operation result, not just the first branch.
#[tokio::test]
async fn test_union_order_by_limit_covers_whole_result() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(&server, &mut session, "CREATE TABLE a (v INT)")
        .await
        .expect("create a");
    exec_ddl(&server, &mut session, "CREATE TABLE b (v INT)")
        .await
        .expect("create b");
    exec_dml(&server, "INSERT INTO a VALUES (5), (3), (9)").await;
    exec_dml(&server, "INSERT INTO b VALUES (1), (7), (2)").await;

    let rows = query_values(
        &server,
        "SELECT v FROM a UNION ALL SELECT v FROM b ORDER BY v LIMIT 4",
    )
    .await;
    assert_eq!(
        ints(&rows),
        vec![1, 2, 3, 5],
        "the sort and cap see both branches"
    );

    let rows = query_values(
        &server,
        "SELECT v FROM a UNION ALL SELECT v FROM b ORDER BY v DESC LIMIT 2",
    )
    .await;
    assert_eq!(
        ints(&rows),
        vec![9, 7],
        "descending order spans both branches"
    );
}

/// A temporal column compared against a plain string literal parses the
/// string as an instant instead of falling to the cross-variant fallback
/// that never matches.
#[tokio::test]
async fn test_timestamp_vs_string_literal_comparisons() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE ev (id INT, at TIMESTAMP, d DATE)",
    )
    .await
    .expect("create");
    exec_dml(
        &server,
        "INSERT INTO ev VALUES \
         (1, CAST('2025-12-31 23:00:00' AS TIMESTAMP), CAST('2025-12-31' AS DATE)), \
         (2, CAST('2026-01-01 00:00:00' AS TIMESTAMP), CAST('2026-01-01' AS DATE)), \
         (3, CAST('2026-02-15 12:30:00' AS TIMESTAMP), CAST('2026-02-15' AS DATE))",
    )
    .await;

    let rows = query_values(
        &server,
        "SELECT id FROM ev WHERE at >= '2026-01-01' ORDER BY id",
    )
    .await;
    assert_eq!(ints(&rows), vec![2, 3], ">= parses the string bound");

    let rows = query_values(&server, "SELECT id FROM ev WHERE at < '2026-01-01'").await;
    assert_eq!(ints(&rows), vec![1], "< parses the string bound");

    let rows = query_values(
        &server,
        "SELECT id FROM ev WHERE at = '2026-01-01 00:00:00'",
    )
    .await;
    assert_eq!(ints(&rows), vec![2], "equality parses the string bound");

    let rows = query_values(
        &server,
        "SELECT id FROM ev WHERE at BETWEEN '2026-01-01' AND '2026-03-01' ORDER BY id",
    )
    .await;
    assert_eq!(ints(&rows), vec![2, 3], "BETWEEN parses both string bounds");

    let rows = query_values(&server, "SELECT id FROM ev WHERE d = '2026-01-01'").await;
    assert_eq!(ints(&rows), vec![2], "DATE equality parses the string");

    let rows = query_values(
        &server,
        "SELECT id FROM ev WHERE d IN ('2026-01-01', '2026-02-15') ORDER BY id",
    )
    .await;
    assert_eq!(ints(&rows), vec![2, 3], "IN list parses each string");
}

/// An unparseable string against a temporal column is a loud error, never
/// an empty result.
#[tokio::test]
async fn test_timestamp_vs_garbage_string_errors() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE ev (id INT, at TIMESTAMP)",
    )
    .await
    .expect("create");
    exec_dml(
        &server,
        "INSERT INTO ev VALUES (1, CAST('2026-01-01 00:00:00' AS TIMESTAMP))",
    )
    .await;
    let result =
        common::try_query_values(&server, "SELECT id FROM ev WHERE at > 'not a time'").await;
    assert!(
        result.is_err(),
        "an unparseable temporal string must error, not filter everything out"
    );
}

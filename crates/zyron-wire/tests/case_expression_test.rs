//! CASE expressions, their nulls, and branches whose types differ.
//!
//! Two defects sat here. The merge loop seeded its null bitmap at the row
//! count and then appended a row per row, leaving the bitmap twice the
//! length of its data, and every read of it landed in the seeded half, so a
//! CASE that produced a null reported a value. And a branch whose column
//! type differed from the result type was merged as it came, which is a
//! buffer of one variant taking a value of another.
//!
//! `THEN <money> ELSE 0` is the ordinary way to write a conditional sum, so
//! both were reachable from plain SQL.
//!
//! Run: cargo test -p zyron-wire --test case_expression_test

mod common;

use common::{create_test_server, exec_ddl, exec_dml, new_session, query_values};
use zyron_executor::column::ScalarValue;

fn scalars(rows: &[Vec<ScalarValue>]) -> Vec<ScalarValue> {
    rows.iter()
        .map(|r| r.first().cloned().unwrap_or(ScalarValue::Null))
        .collect()
}

fn as_f64(v: &ScalarValue) -> f64 {
    match v {
        ScalarValue::Int8(x) => *x as f64,
        ScalarValue::Int16(x) => *x as f64,
        ScalarValue::Int32(x) => *x as f64,
        ScalarValue::Int64(x) => *x as f64,
        ScalarValue::Float32(x) => *x as f64,
        ScalarValue::Float64(x) => *x,
        other => panic!("expected a number, got {other:?}"),
    }
}

/// A CASE branch that yields NULL has to read back as NULL.
#[tokio::test]
async fn test_a_case_branch_yielding_null_reports_null() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(&server, &mut session, "CREATE TABLE t (k INT)")
        .await
        .expect("create");
    exec_dml(&server, "INSERT INTO t VALUES (1), (2), (3)").await;

    let rows = query_values(
        &server,
        "SELECT CASE WHEN k = 2 THEN NULL ELSE k END FROM t ORDER BY k",
    )
    .await;
    let values = scalars(&rows);
    assert_eq!(values.len(), 3);
    assert!(
        matches!(values[1], ScalarValue::Null),
        "the k = 2 row took the NULL branch but reported {:?}",
        values[1]
    );
    assert!(!matches!(values[0], ScalarValue::Null));
    assert!(!matches!(values[2], ScalarValue::Null));
}

/// A NULL a CASE produced is skipped by an aggregate, the same as any other
/// NULL. This is the shape that turns a wrong bitmap into a wrong number.
#[tokio::test]
async fn test_an_aggregate_skips_the_nulls_a_case_produced() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(&server, &mut session, "CREATE TABLE t (k INT)")
        .await
        .expect("create");
    exec_dml(&server, "INSERT INTO t VALUES (1), (2), (3), (4)").await;

    let counted = query_values(
        &server,
        "SELECT COUNT(CASE WHEN k > 2 THEN k ELSE NULL END) FROM t",
    )
    .await;
    assert_eq!(
        as_f64(&scalars(&counted)[0]),
        2.0,
        "only k = 3 and k = 4 are counted"
    );

    let summed = query_values(
        &server,
        "SELECT SUM(CASE WHEN k > 2 THEN k ELSE NULL END) FROM t",
    )
    .await;
    assert_eq!(as_f64(&scalars(&summed)[0]), 7.0, "3 + 4");
}

/// Branches whose types differ unify to the type the whole CASE produces,
/// rather than the merge taking a value of one variant into a buffer of
/// another.
#[tokio::test]
async fn test_branches_of_different_types_unify() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE t (k INT, price DOUBLE PRECISION)",
    )
    .await
    .expect("create");
    exec_dml(
        &server,
        "INSERT INTO t VALUES (1, 10.50), (2, 20.25), (3, 5.00)",
    )
    .await;

    // The conditional-sum shape: a floating branch against an integer zero
    let rows = query_values(
        &server,
        "SELECT SUM(CASE WHEN k < 3 THEN price ELSE 0 END) FROM t",
    )
    .await;
    let total = as_f64(&scalars(&rows)[0]);
    assert!(
        (total - 30.75).abs() < 1e-6,
        "10.50 + 20.25 with the third row contributing zero, got {total}"
    );

    // And the reverse order, an integer branch against a money one
    let rows = query_values(
        &server,
        "SELECT SUM(CASE WHEN k < 3 THEN 0 ELSE price END) FROM t",
    )
    .await;
    let total = as_f64(&scalars(&rows)[0]);
    assert!(
        (total - 5.0).abs() < 1e-6,
        "only the third row, got {total}"
    );
}

/// A decimal branch against an integer one stays a decimal.
///
/// The branches unify through the same numeric promotion binary operators
/// use, and that promotion did not know about decimals: it picked Int64,
/// which then could not hold the scaled i128 and refused the whole query.
/// Widening to a float instead would have given away the exactness the type
/// exists for. `THEN <money> ELSE 0` is the ordinary conditional sum, so
/// this is the shape that hit it.
#[tokio::test]
async fn test_a_decimal_branch_against_an_integer_stays_a_decimal() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE t (k INT, price DECIMAL(12,2))",
    )
    .await
    .expect("create");
    exec_dml(
        &server,
        "INSERT INTO t VALUES (1, 10.50), (2, 20.25), (3, 5.00)",
    )
    .await;

    let rows = query_values(
        &server,
        "SELECT SUM(CASE WHEN k < 3 THEN price ELSE 0 END) FROM t",
    )
    .await;
    let total = match scalars(&rows).first() {
        Some(ScalarValue::Int128(v)) => zyron_common::format_decimal(*v, 2),
        other => panic!("expected a decimal, got {other:?}"),
    };
    assert_eq!(
        total, "30.75",
        "10.50 + 20.25, the third row contributing 0"
    );

    // And the reverse order, an integer branch against a decimal one
    let rows = query_values(
        &server,
        "SELECT SUM(CASE WHEN k < 3 THEN 0 ELSE price END) FROM t",
    )
    .await;
    let total = match scalars(&rows).first() {
        Some(ScalarValue::Int128(v)) => zyron_common::format_decimal(*v, 2),
        other => panic!("expected a decimal, got {other:?}"),
    };
    assert_eq!(total, "5.00", "only the third row");
}

/// A CASE with no ELSE yields NULL where nothing matched, in the result
/// type rather than a placeholder one.
#[tokio::test]
async fn test_a_case_without_an_else_yields_null_where_nothing_matched() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(&server, &mut session, "CREATE TABLE t (k INT)")
        .await
        .expect("create");
    exec_dml(&server, "INSERT INTO t VALUES (1), (2)").await;

    let rows = query_values(
        &server,
        "SELECT CASE WHEN k = 1 THEN 100 END FROM t ORDER BY k",
    )
    .await;
    let values = scalars(&rows);
    assert_eq!(as_f64(&values[0]), 100.0);
    assert!(
        matches!(values[1], ScalarValue::Null),
        "k = 2 matched no branch and there is no ELSE, got {:?}",
        values[1]
    );
}

/// Several WHEN arms, where the first match wins.
#[tokio::test]
async fn test_the_first_matching_arm_wins() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(&server, &mut session, "CREATE TABLE t (k INT)")
        .await
        .expect("create");
    exec_dml(&server, "INSERT INTO t VALUES (1), (5), (10)").await;

    let rows = query_values(
        &server,
        "SELECT CASE WHEN k < 3 THEN 'low' WHEN k < 8 THEN 'mid' ELSE 'high' END FROM t ORDER BY k",
    )
    .await;
    let values: Vec<String> = scalars(&rows)
        .iter()
        .map(|v| match v {
            ScalarValue::Utf8(s) => s.clone(),
            other => panic!("expected text, got {other:?}"),
        })
        .collect();
    assert_eq!(values, vec!["low", "mid", "high"]);
}

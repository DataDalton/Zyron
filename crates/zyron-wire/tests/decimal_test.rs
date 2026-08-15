//! DECIMAL / NUMERIC values, end to end.
//!
//! A DECIMAL is stored as an i128 holding the value multiplied by ten to the
//! column's scale. Nothing captured that scale and nothing converted a value
//! onto it, so every DECIMAL column silently stored zero: the declaration
//! parsed, the column existed, and the data was discarded.
//!
//! The point of the type is exactness, so these check the values rather than
//! only that a statement succeeded.
//!
//! Run: cargo test -p zyron-wire --test decimal_test

mod common;

use common::{create_test_server, exec_ddl, exec_dml, exec_dml_result, new_session, query_values};
use zyron_executor::column::ScalarValue;

/// Reads a decimal column back as the text a client would see, which is the
/// only rendering that shows where the point sits.
async fn decimals(
    server: &std::sync::Arc<zyron_wire::connection::ServerState>,
    sql: &str,
    scale: u8,
) -> Vec<String> {
    query_values(server, sql)
        .await
        .iter()
        .map(|row| match row.first() {
            Some(ScalarValue::Int128(v)) => zyron_common::format_decimal(*v, scale),
            Some(ScalarValue::Null) => "NULL".to_string(),
            other => panic!("expected a decimal, got {other:?}"),
        })
        .collect()
}

#[tokio::test]
async fn test_a_decimal_column_stores_the_value_it_was_given() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE d (id INT, amount DECIMAL(15,2))",
    )
    .await
    .expect("create");

    exec_dml(
        &server,
        "INSERT INTO d VALUES (1, 10.50), (2, 0.01), (3, -3.25), (4, 7), (5, 1000)",
    )
    .await;

    assert_eq!(
        decimals(&server, "SELECT amount FROM d ORDER BY id", 2).await,
        vec!["10.50", "0.01", "-3.25", "7.00", "1000.00"],
    );
}

/// A scale of zero keeps whole numbers whole, rather than treating the value
/// as if it had digits below the point.
#[tokio::test]
async fn test_a_zero_scale_decimal_holds_whole_numbers() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE d (id INT, n DECIMAL(10,0), m NUMERIC(8,3))",
    )
    .await
    .expect("create");
    exec_dml(&server, "INSERT INTO d VALUES (1, 7, 1.234), (2, -5, -2.5)").await;

    assert_eq!(
        decimals(&server, "SELECT n FROM d ORDER BY id", 0).await,
        vec!["7", "-5"]
    );
    assert_eq!(
        decimals(&server, "SELECT m FROM d ORDER BY id", 3).await,
        vec!["1.234", "-2.500"]
    );
}

/// The reason the type exists: a tenth is a tenth, not the nearest binary
/// fraction, so ten of them make exactly one.
#[tokio::test]
async fn test_decimal_arithmetic_is_exact_where_a_float_would_drift() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(&server, &mut session, "CREATE TABLE d (v DECIMAL(12,2))")
        .await
        .expect("create");
    for _ in 0..10 {
        exec_dml(&server, "INSERT INTO d VALUES (0.10)").await;
    }
    assert_eq!(
        decimals(&server, "SELECT SUM(v) FROM d", 2).await,
        vec!["1.00"],
        "ten tenths sum to exactly one"
    );
}

/// SUM, MIN and MAX run over the scaled integers, so they keep the scale.
#[tokio::test]
async fn test_aggregates_keep_the_scale() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE d (id INT, v DECIMAL(12,2))",
    )
    .await
    .expect("create");
    exec_dml(
        &server,
        "INSERT INTO d VALUES (1, 10.50), (2, 20.25), (3, 5.00)",
    )
    .await;

    assert_eq!(
        decimals(&server, "SELECT SUM(v) FROM d", 2).await,
        vec!["35.75"]
    );
    assert_eq!(
        decimals(&server, "SELECT MIN(v) FROM d", 2).await,
        vec!["5.00"]
    );
    assert_eq!(
        decimals(&server, "SELECT MAX(v) FROM d", 2).await,
        vec!["20.25"]
    );
}

/// A predicate compares against the column's scale rather than against the
/// raw stored integer.
#[tokio::test]
async fn test_a_predicate_compares_on_the_declared_scale() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE d (id INT, v DECIMAL(12,2))",
    )
    .await
    .expect("create");
    exec_dml(
        &server,
        "INSERT INTO d VALUES (1, 10.50), (2, 20.25), (3, 5.00)",
    )
    .await;

    assert_eq!(
        decimals(&server, "SELECT v FROM d WHERE v > 10.00 ORDER BY id", 2).await,
        vec!["10.50", "20.25"]
    );
    assert_eq!(
        decimals(&server, "SELECT v FROM d WHERE v = 10.50", 2).await,
        vec!["10.50"]
    );
    assert_eq!(
        decimals(&server, "SELECT v FROM d WHERE v < 6 ORDER BY id", 2).await,
        vec!["5.00"],
        "a whole number in the predicate takes the column's scale"
    );
}

/// The declared precision is a bound, so a value wider than it is refused
/// rather than stored as something else.
#[tokio::test]
async fn test_a_value_wider_than_the_declared_precision_is_refused() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(&server, &mut session, "CREATE TABLE d (v DECIMAL(5,2))")
        .await
        .expect("create");

    // DECIMAL(5,2) reaches 999.99
    exec_dml(&server, "INSERT INTO d VALUES (999.99)").await;
    let err = exec_dml_result(&server, "INSERT INTO d VALUES (1234.56)")
        .await
        .expect_err("a value past the declared precision must be refused");
    assert!(
        err.to_string().contains("DECIMAL"),
        "the refusal names the declaration: {err}"
    );
    assert_eq!(
        decimals(&server, "SELECT v FROM d", 2).await,
        vec!["999.99"]
    );
}

/// Digits below the declared scale round rather than truncate, and rounding
/// is half away from zero.
#[tokio::test]
async fn test_digits_below_the_scale_round_half_away_from_zero() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE d (id INT, v DECIMAL(12,2))",
    )
    .await
    .expect("create");
    exec_dml(
        &server,
        "INSERT INTO d VALUES (1, 1.005), (2, 1.004), (3, -1.005)",
    )
    .await;

    assert_eq!(
        decimals(&server, "SELECT v FROM d ORDER BY id", 2).await,
        vec!["1.01", "1.00", "-1.01"]
    );
}

/// An UPDATE puts its value on the column's scale, the same as an INSERT.
#[tokio::test]
async fn test_an_update_scales_its_value_too() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE d (id INT, v DECIMAL(12,2))",
    )
    .await
    .expect("create");
    exec_dml(&server, "INSERT INTO d VALUES (1, 10.50)").await;
    exec_dml(&server, "UPDATE d SET v = 42.75 WHERE id = 1").await;

    assert_eq!(decimals(&server, "SELECT v FROM d", 2).await, vec!["42.75"]);
}

/// A NULL decimal stays null rather than becoming zero.
#[tokio::test]
async fn test_a_null_decimal_stays_null() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE d (id INT, v DECIMAL(12,2))",
    )
    .await
    .expect("create");
    exec_dml(&server, "INSERT INTO d VALUES (1, NULL), (2, 1.25)").await;

    assert_eq!(
        decimals(&server, "SELECT v FROM d ORDER BY id", 2).await,
        vec!["NULL", "1.25"]
    );
    assert_eq!(
        decimals(&server, "SELECT SUM(v) FROM d", 2).await,
        vec!["1.25"],
        "a null contributes nothing to the sum"
    );
}

/// An explicit cast produces the same value the column would store.
#[tokio::test]
async fn test_an_explicit_cast_to_decimal_keeps_the_value() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(&server, &mut session, "CREATE TABLE d (v DECIMAL(12,2))")
        .await
        .expect("create");
    exec_dml(
        &server,
        "INSERT INTO d VALUES (CAST('10.50' AS DECIMAL(12,2)))",
    )
    .await;

    assert_eq!(decimals(&server, "SELECT v FROM d", 2).await, vec!["10.50"]);
}

/// Arithmetic keeps the operands' units. A sum stays on the scale, a product
/// comes back down from the doubled scale it lands on, and a quotient is
/// raised back onto it.
#[tokio::test]
async fn test_arithmetic_keeps_the_scale() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE d (id INT, v DECIMAL(12,2))",
    )
    .await
    .expect("create");
    exec_dml(&server, "INSERT INTO d VALUES (1, 10.50)").await;

    assert_eq!(
        decimals(&server, "SELECT v + 1.25 FROM d", 2).await,
        vec!["11.75"]
    );
    assert_eq!(
        decimals(&server, "SELECT v - 0.50 FROM d", 2).await,
        vec!["10.00"]
    );
    assert_eq!(
        decimals(&server, "SELECT v * 2 FROM d", 2).await,
        vec!["21.00"],
        "a product comes back to the operands' scale"
    );
    assert_eq!(
        decimals(&server, "SELECT v / 2 FROM d", 2).await,
        vec!["5.25"],
        "a quotient is raised back onto the scale"
    );
}

/// Two columns of different scale meet on the wider one, so neither loses
/// digits.
#[tokio::test]
async fn test_two_scales_meet_on_the_wider_one() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE d (a DECIMAL(12,2), b DECIMAL(12,4))",
    )
    .await
    .expect("create");
    exec_dml(&server, "INSERT INTO d VALUES (1.25, 0.0625)").await;

    assert_eq!(
        decimals(&server, "SELECT a + b FROM d", 4).await,
        vec!["1.3125"],
        "the four digit scale is kept"
    );
    assert_eq!(
        decimals(&server, "SELECT a FROM d WHERE a > b", 2).await,
        vec!["1.25"],
        "comparing across scales reads both as numbers"
    );
}

/// Ordering follows the number, which for one scale is the stored integer.
#[tokio::test]
async fn test_ordering_follows_the_numeric_value() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(&server, &mut session, "CREATE TABLE d (v DECIMAL(12,2))")
        .await
        .expect("create");
    exec_dml(
        &server,
        "INSERT INTO d VALUES (10.50), (2.00), (-3.25), (100.00), (0.01)",
    )
    .await;

    assert_eq!(
        decimals(&server, "SELECT v FROM d ORDER BY v", 2).await,
        vec!["-3.25", "0.01", "2.00", "10.50", "100.00"]
    );
}

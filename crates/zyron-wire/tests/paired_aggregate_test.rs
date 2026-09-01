//! Aggregates that read two columns at once.
//!
//! Correlation, covariance and a time weighted average each need both of
//! their arguments paired on the same row, which a single argument
//! accumulator cannot express. All three were typed in the registry with no
//! implementation until now, so these tests check the values rather than
//! only that the call is recognized.
//!
//! Run: cargo test -p zyron-wire --test paired_aggregate_test

mod common;

use common::{create_test_server, exec_ddl, exec_dml, new_session, query_values};
use zyron_executor::column::ScalarValue;

fn as_f64(rows: &[Vec<ScalarValue>]) -> Option<f64> {
    match rows.first().and_then(|r| r.first()) {
        Some(ScalarValue::Float64(v)) => Some(*v),
        Some(ScalarValue::Float32(v)) => Some(f64::from(*v)),
        Some(ScalarValue::Null) | None => None,
        other => panic!("expected a float, got {other:?}"),
    }
}

fn close(actual: Option<f64>, expected: f64, what: &str) {
    let got = actual.unwrap_or_else(|| panic!("{what} came back null"));
    assert!(
        (got - expected).abs() < 1e-9,
        "{what}: expected {expected}, got {got}"
    );
}

/// A perfect straight line correlates at exactly one, and reversing the
/// slope gives exactly minus one
#[tokio::test]
async fn test_correlation_of_a_straight_line() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE pairs (x DOUBLE PRECISION, y DOUBLE PRECISION, z DOUBLE PRECISION)",
    )
    .await
    .expect("create");
    exec_dml(
        &server,
        "INSERT INTO pairs VALUES (1,3,10), (2,5,8), (3,7,6), (4,9,4), (5,11,2)",
    )
    .await;

    close(
        as_f64(&query_values(&server, "SELECT correlation_agg(x, y) FROM pairs").await),
        1.0,
        "correlation of an ascending line",
    );
    close(
        as_f64(&query_values(&server, "SELECT correlation_agg(x, z) FROM pairs").await),
        -1.0,
        "correlation of a descending line",
    );
}

/// Covariance is the sample form, dividing by one less than the count, and
/// a column that never varies has none
#[tokio::test]
async fn test_covariance_is_the_sample_form() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE cov (x DOUBLE PRECISION, y DOUBLE PRECISION, flat DOUBLE PRECISION)",
    )
    .await
    .expect("create");
    // x deviations -2,-1,0,1,2 and y deviations -4,-2,0,2,4 give a sum of
    // products of 20, over four degrees of freedom
    exec_dml(
        &server,
        "INSERT INTO cov VALUES (1,2,7), (2,4,7), (3,6,7), (4,8,7), (5,10,7)",
    )
    .await;
    close(
        as_f64(&query_values(&server, "SELECT covariance_agg(x, y) FROM cov").await),
        5.0,
        "sample covariance",
    );
    // A constant column has no variation for the ratio to divide by
    assert_eq!(
        as_f64(&query_values(&server, "SELECT correlation_agg(x, flat) FROM cov").await),
        None,
        "a constant column has no correlation"
    );
    close(
        as_f64(&query_values(&server, "SELECT covariance_agg(x, flat) FROM cov").await),
        0.0,
        "covariance against a constant",
    );
}

/// A row missing either half of the pair is not a pair, so it contributes
/// nothing rather than counting as a zero
#[tokio::test]
async fn test_pairs_need_both_halves() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE holes (x DOUBLE PRECISION, y DOUBLE PRECISION)",
    )
    .await
    .expect("create");
    exec_dml(
        &server,
        "INSERT INTO holes VALUES (1,2), (2,4), (NULL,6), (3,NULL), (4,8), (5,10)",
    )
    .await;
    // The four complete pairs still lie on the same line
    close(
        as_f64(&query_values(&server, "SELECT correlation_agg(x, y) FROM holes").await),
        1.0,
        "correlation over the complete pairs",
    );

    // Fewer than two pairs leaves nothing to compare
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE lonely (x DOUBLE PRECISION, y DOUBLE PRECISION)",
    )
    .await
    .expect("create");
    exec_dml(&server, "INSERT INTO lonely VALUES (1,2)").await;
    assert_eq!(
        as_f64(&query_values(&server, "SELECT covariance_agg(x, y) FROM lonely").await),
        None,
        "one pair has no sample covariance"
    );
}

/// A value held for a long interval counts for more than one held briefly,
/// which is the whole point of weighting by time.
///
/// Readings are folded in the order they arrive, so the rows are written in
/// time order here. Putting them in that order is the caller's to do, the
/// same way it is for any aggregate whose result depends on sequence
#[tokio::test]
async fn test_time_weight_weights_by_the_interval() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE readings (v DOUBLE PRECISION, t DOUBLE PRECISION)",
    )
    .await
    .expect("create");
    // 10 holds for 9 seconds, 20 holds for 1, so the weighted mean sits far
    // below the plain mean of 15
    exec_dml(
        &server,
        "INSERT INTO readings VALUES (10,0), (20,9), (30,10)",
    )
    .await;
    close(
        as_f64(&query_values(&server, "SELECT time_weight(v, t) FROM readings").await),
        11.0,
        "time weighted average",
    );

    // Evenly spaced readings weight equally, so the result is the mean of
    // every reading but the last, which closes no interval
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE even (v DOUBLE PRECISION, t DOUBLE PRECISION)",
    )
    .await
    .expect("create");
    exec_dml(
        &server,
        "INSERT INTO even VALUES (2,0), (4,1), (6,2), (8,3)",
    )
    .await;
    close(
        as_f64(&query_values(&server, "SELECT time_weight(v, t) FROM even").await),
        4.0,
        "evenly spaced readings",
    );
}

/// Grouped, so each group keeps its own running sums
#[tokio::test]
async fn test_paired_aggregates_group_independently() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE grouped (g INT, x DOUBLE PRECISION, y DOUBLE PRECISION)",
    )
    .await
    .expect("create");
    exec_dml(
        &server,
        "INSERT INTO grouped VALUES \
         (1,1,2), (1,2,4), (1,3,6), \
         (2,1,9), (2,2,6), (2,3,3)",
    )
    .await;
    let rows = query_values(
        &server,
        "SELECT g, correlation_agg(x, y) FROM grouped GROUP BY g ORDER BY g",
    )
    .await;
    assert_eq!(rows.len(), 2, "expected one row per group");
    let value_of = |row: &Vec<ScalarValue>| match row[1] {
        ScalarValue::Float64(v) => v,
        ref other => panic!("expected a float, got {other:?}"),
    };
    assert!(
        (value_of(&rows[0]) - 1.0).abs() < 1e-9,
        "the ascending group did not correlate at one"
    );
    assert!(
        (value_of(&rows[1]) + 1.0).abs() < 1e-9,
        "the descending group did not correlate at minus one"
    );
}

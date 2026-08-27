//! Wrong-answer fixes across executor semantics: reversed join keys,
//! set-operation validation and alignment, unsigned and float integer
//! casts, integer overflow detection, high-scale decimal multiply,
//! decimal CEIL/FLOOR, GREATEST/LEAST type unification, zero-sign
//! grouping, and DISTINCT decimal aggregates.
//!
//! Run: cargo test -p zyron-wire --test exec_semantics_tail_test

mod common;

use common::{create_test_server, exec_ddl, exec_dml, new_session, query_values, try_query_values};
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

/// A LEFT JOIN whose ON clause names the right table first joins the same
/// rows as the conventional spelling.
#[tokio::test]
async fn test_left_join_reversed_on_order() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(&server, &mut session, "CREATE TABLE a (id INT, v INT)")
        .await
        .expect("create a");
    exec_ddl(&server, &mut session, "CREATE TABLE b (id INT, w INT)")
        .await
        .expect("create b");
    exec_dml(&server, "INSERT INTO a VALUES (1, 10), (2, 20), (3, 30)").await;
    exec_dml(&server, "INSERT INTO b VALUES (1, 100), (3, 300)").await;

    let forward = query_values(
        &server,
        "SELECT a.id FROM a LEFT JOIN b ON a.id = b.id WHERE b.w IS NOT NULL ORDER BY a.id",
    )
    .await;
    let reversed = query_values(
        &server,
        "SELECT a.id FROM a LEFT JOIN b ON b.id = a.id WHERE b.w IS NOT NULL ORDER BY a.id",
    )
    .await;
    assert_eq!(ints(&forward), vec![1, 3]);
    assert_eq!(
        ints(&reversed),
        vec![1, 3],
        "ON b.id = a.id joins the same rows as ON a.id = b.id"
    );
}

/// A set operation whose branches disagree on column count is a bind-time
/// error, not a panic or a truncated merge.
#[tokio::test]
async fn test_union_arity_mismatch_errors() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(&server, &mut session, "CREATE TABLE t (a INT, b INT)")
        .await
        .expect("create");
    exec_dml(&server, "INSERT INTO t VALUES (1, 2)").await;

    let result = try_query_values(&server, "SELECT a, b FROM t UNION ALL SELECT a FROM t").await;
    assert!(result.is_err(), "arity mismatch must error at bind time");

    let result = try_query_values(&server, "SELECT a FROM t UNION SELECT 'text'").await;
    assert!(
        result.is_err(),
        "INT vs TEXT branches must error at bind time"
    );
}

/// Numerically compatible branches of different widths merge by value.
#[tokio::test]
async fn test_union_numeric_widths_merge_by_value() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(&server, &mut session, "CREATE TABLE s (v SMALLINT)")
        .await
        .expect("create s");
    exec_ddl(&server, &mut session, "CREATE TABLE g (v BIGINT)")
        .await
        .expect("create g");
    exec_dml(&server, "INSERT INTO s VALUES (1), (2)").await;
    exec_dml(&server, "INSERT INTO g VALUES (2), (3)").await;

    let rows = query_values(&server, "SELECT v FROM g UNION SELECT v FROM s ORDER BY v").await;
    assert_eq!(
        ints(&rows),
        vec![1, 2, 3],
        "the value 2 merges across widths instead of surviving twice"
    );
}

/// A u64 column above 2^63 compares by value against signed literals.
#[tokio::test]
async fn test_u64_above_i64_max_compares_by_value() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(&server, &mut session, "CREATE TABLE u (id INT, v UINT64)")
        .await
        .expect("create");
    exec_dml(
        &server,
        "INSERT INTO u VALUES (1, 5), (2, 9300000000000000000)",
    )
    .await;

    let rows = query_values(&server, "SELECT id FROM u WHERE v > 0 ORDER BY id").await;
    assert_eq!(
        ints(&rows),
        vec![1, 2],
        "a u64 above i64::MAX stays positive in the comparison"
    );

    let rows = query_values(&server, "SELECT id FROM u WHERE v > 100").await;
    assert_eq!(ints(&rows), vec![2]);
}

/// Integer arithmetic overflow reports loudly instead of wrapping into a
/// different number.
#[tokio::test]
async fn test_integer_overflow_errors() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(&server, &mut session, "CREATE TABLE t (v BIGINT)")
        .await
        .expect("create");
    exec_dml(&server, "INSERT INTO t VALUES (9223372036854775807)").await;

    let result = try_query_values(&server, "SELECT v + 1 FROM t").await;
    assert!(result.is_err(), "BIGINT overflow must error, not wrap");

    let result = try_query_values(&server, "SELECT v * 2 FROM t").await;
    assert!(result.is_err(), "BIGINT multiply overflow must error");
}

/// A float that cannot be an integer refuses the cast instead of becoming
/// zero or a saturated extreme, and an in-range float rounds.
#[tokio::test]
async fn test_float_to_bigint_cast_semantics() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(&server, &mut session, "CREATE TABLE f (x DOUBLE PRECISION)")
        .await
        .expect("create");
    exec_dml(&server, "INSERT INTO f VALUES (2.6)").await;

    let rows = query_values(&server, "SELECT CAST(x AS BIGINT) FROM f").await;
    assert_eq!(ints(&rows), vec![3], "2.6 rounds to 3");

    let result = try_query_values(
        &server,
        "SELECT CAST(CAST('NaN' AS DOUBLE PRECISION) AS BIGINT)",
    )
    .await;
    assert!(result.is_err(), "NaN has no integer value");

    let result = try_query_values(
        &server,
        "SELECT CAST(CAST('1e300' AS DOUBLE PRECISION) AS BIGINT)",
    )
    .await;
    assert!(result.is_err(), "an out-of-range float refuses the cast");
}

/// Decimal multiplication at scale 20 keeps the point where it belongs.
#[tokio::test]
async fn test_decimal_multiply_high_scale() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE d (a DECIMAL(38,20), b DECIMAL(38,20))",
    )
    .await
    .expect("create");
    exec_dml(&server, "INSERT INTO d VALUES (2.0, 3.0)").await;

    let rows = query_values(&server, "SELECT id FROM (SELECT 1 AS id, a * b AS p FROM d) q WHERE q.p = CAST(6 AS DECIMAL(38,20))").await;
    assert_eq!(ints(&rows), vec![1], "2.0 * 3.0 at scale 20 is 6, not 600");
}

/// CEIL and FLOOR move a decimal to the enclosing integers on its own
/// scale instead of passing the scaled integer through untouched.
#[tokio::test]
async fn test_decimal_ceil_floor() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(&server, &mut session, "CREATE TABLE d (v DECIMAL(10,2))")
        .await
        .expect("create");
    exec_dml(&server, "INSERT INTO d VALUES (2.10), (-2.10)").await;

    let rows = query_values(
        &server,
        "SELECT CAST(ceil(v) AS BIGINT) FROM d ORDER BY v DESC",
    )
    .await;
    assert_eq!(
        ints(&rows),
        vec![3, -2],
        "ceil moves toward positive infinity"
    );

    let rows = query_values(
        &server,
        "SELECT CAST(floor(v) AS BIGINT) FROM d ORDER BY v DESC",
    )
    .await;
    assert_eq!(
        ints(&rows),
        vec![2, -3],
        "floor moves toward negative infinity"
    );
}

/// GREATEST unifies its argument types instead of truncating everything
/// onto the first argument's type.
#[tokio::test]
async fn test_greatest_least_unify_types() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(&server, &mut session, "CREATE TABLE t (k INT)")
        .await
        .expect("create");
    exec_dml(&server, "INSERT INTO t VALUES (1)").await;

    let rows = query_values(&server, "SELECT greatest(k, 2.5) FROM t").await;
    match rows[0].first() {
        Some(ScalarValue::Float64(v)) => {
            assert_eq!(*v, 2.5, "the float argument survives at full value")
        }
        other => panic!("expected a float result, got {other:?}"),
    }

    let rows = query_values(
        &server,
        "SELECT CAST(least(k, 0.5) AS DOUBLE PRECISION) FROM t",
    )
    .await;
    match rows[0].first() {
        Some(ScalarValue::Float64(v)) => assert_eq!(*v, 0.5),
        other => panic!("expected a float result, got {other:?}"),
    }
}

/// Negative zero groups with positive zero, matching what = reports.
#[tokio::test]
async fn test_negative_zero_groups_with_zero() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(&server, &mut session, "CREATE TABLE z (x DOUBLE PRECISION)")
        .await
        .expect("create");
    exec_dml(&server, "INSERT INTO z VALUES (0.0), (-0.0), (1.5)").await;

    let rows = query_values(&server, "SELECT CAST(COUNT(*) AS BIGINT) FROM z GROUP BY x").await;
    let mut counts = ints(&rows);
    counts.sort_unstable();
    assert_eq!(counts, vec![1, 2], "both zeros land in one group of two");

    let rows = query_values(&server, "SELECT CAST(COUNT(DISTINCT x) AS BIGINT) FROM z").await;
    assert_eq!(ints(&rows), vec![2], "DISTINCT keeps one zero");
}

/// AVG(DISTINCT) over a decimal averages values, not raw scaled integers.
#[tokio::test]
async fn test_avg_distinct_decimal_on_value_scale() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(&server, &mut session, "CREATE TABLE p (v DECIMAL(10,2))")
        .await
        .expect("create");
    exec_dml(&server, "INSERT INTO p VALUES (1.00), (1.00), (3.00)").await;

    let rows = query_values(&server, "SELECT AVG(DISTINCT v) FROM p").await;
    match rows[0].first() {
        Some(ScalarValue::Float64(v)) => {
            assert!(
                (*v - 2.0).abs() < 1e-9,
                "AVG(DISTINCT 1.00, 3.00) is 2, got {v}"
            )
        }
        other => panic!("expected a float result, got {other:?}"),
    }
}

/// A NULL timestamp plus an interval is NULL, not an epoch-based instant.
#[tokio::test]
async fn test_null_timestamp_plus_interval_is_null() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE e (id INT, at TIMESTAMP)",
    )
    .await
    .expect("create");
    exec_dml(
        &server,
        "INSERT INTO e VALUES (1, NULL), (2, CAST('2026-01-01 00:00:00' AS TIMESTAMP))",
    )
    .await;

    let rows = query_values(
        &server,
        "SELECT id FROM e WHERE at + INTERVAL '1 day' IS NULL",
    )
    .await;
    assert_eq!(
        ints(&rows),
        vec![1],
        "NULL + interval stays NULL instead of becoming an epoch instant"
    );
}

/// An interval scaled by a fractional factor keeps the fraction: half a
/// day is twelve hours, not zero.
#[tokio::test]
async fn test_interval_float_multiplier() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(&server, &mut session, "CREATE TABLE e (at TIMESTAMP)")
        .await
        .expect("create");
    exec_dml(
        &server,
        "INSERT INTO e VALUES (CAST('2026-01-01 00:00:00' AS TIMESTAMP))",
    )
    .await;

    let rows = query_values(
        &server,
        "SELECT CAST(COUNT(*) AS BIGINT) FROM e \
         WHERE at + INTERVAL '1 day' * 0.5 = CAST('2026-01-01 12:00:00' AS TIMESTAMP)",
    )
    .await;
    assert_eq!(ints(&rows), vec![1], "half a day is twelve hours, not zero");
}

/// A RANGE frame under DESC measures value distance along the descending
/// axis instead of searching a descending array as if ascending.
#[tokio::test]
async fn test_range_frame_desc() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(&server, &mut session, "CREATE TABLE r (v BIGINT)")
        .await
        .expect("create");
    exec_dml(&server, "INSERT INTO r VALUES (10), (20), (30), (40)").await;

    // Descending order: each row's frame covers values within 10 above
    // it. In input row order: 10 -> {20, 10}, 20 -> {30, 20},
    // 30 -> {40, 30}, 40 -> {40}
    let rows = query_values(
        &server,
        "SELECT CAST(SUM(v) OVER (ORDER BY v DESC RANGE BETWEEN 10 PRECEDING AND CURRENT ROW) \
         AS BIGINT) FROM r",
    )
    .await;
    assert_eq!(
        ints(&rows),
        vec![30, 50, 70, 40],
        "each descending frame spans ten units toward larger values"
    );
}

/// A RANGE frame over an ascending axis with the same data, as the
/// control for the DESC case.
#[tokio::test]
async fn test_range_frame_asc_control() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(&server, &mut session, "CREATE TABLE r (v BIGINT)")
        .await
        .expect("create");
    exec_dml(&server, "INSERT INTO r VALUES (10), (20), (30), (40)").await;

    let rows = query_values(
        &server,
        "SELECT CAST(SUM(v) OVER (ORDER BY v RANGE BETWEEN 10 PRECEDING AND CURRENT ROW) \
         AS BIGINT) FROM r",
    )
    .await;
    assert_eq!(ints(&rows), vec![10, 30, 50, 70]);
}

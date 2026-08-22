//! Narrowing casts between integer widths.
//!
//! `cast_scalar`'s Int64 arm listed every narrower integer, both floats, a
//! boolean and text, but not Int128, so casting the 128 bit width down was
//! refused outright even for a value that fits. It surfaced through a
//! decimal, whose stored form is an i128, but the gap is the cast's own.
//!
//! A narrowing that does not fit has to be refused rather than wrapped: a
//! value silently becoming a different number is worse than a statement that
//! fails.
//!
//! Run: cargo test -p zyron-wire --test numeric_cast_test

mod common;

use common::{create_test_server, exec_ddl, exec_dml, exec_dml_result, new_session, query_values};
use zyron_executor::column::ScalarValue;

fn as_i64(rows: &[Vec<ScalarValue>]) -> Vec<i64> {
    rows.iter()
        .map(|r| match r.first() {
            Some(ScalarValue::Int64(v)) => *v,
            Some(ScalarValue::Int32(v)) => *v as i64,
            other => panic!("expected a 64 bit integer, got {other:?}"),
        })
        .collect()
}

fn as_i128(rows: &[Vec<ScalarValue>]) -> Vec<i128> {
    rows.iter()
        .map(|r| match r.first() {
            Some(ScalarValue::Int128(v)) => *v,
            other => panic!("expected a 128 bit integer, got {other:?}"),
        })
        .collect()
}

/// A literal wider than i64 reaches an INT128 column with every digit
/// intact.
///
/// The lexer read every integer literal into an i64 and refused anything
/// larger, so a column whose whole purpose is holding those values could not
/// be given one. Parsing is only half of it: the value has to survive the
/// write and read back as itself.
#[tokio::test]
async fn test_a_literal_wider_than_64_bits_round_trips_through_int128() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(&server, &mut session, "CREATE TABLE t (id INT, v INT128)")
        .await
        .expect("create");

    // The widest values an i128 holds, either side of zero, and one just
    // past the 64 bit range where the old lexer gave up
    exec_dml(
        &server,
        "INSERT INTO t VALUES \
         (1, 170141183460469231731687303715884105727), \
         (2, -170141183460469231731687303715884105728), \
         (3, 9223372036854775808), \
         (4, -9223372036854775809)",
    )
    .await;

    assert_eq!(
        as_i128(&query_values(&server, "SELECT v FROM t ORDER BY id").await),
        vec![
            i128::MAX,
            i128::MIN,
            9_223_372_036_854_775_808i128,
            -9_223_372_036_854_775_809i128,
        ],
        "every digit survives the write"
    );
}

/// A literal past the i128 range has no integer type to land in, so it is
/// refused with a message saying so rather than wrapping.
/// The refusal is a parse error, so it is read off the parser rather than
/// through the execution path, which treats an unparseable statement as a
/// caller mistake and panics.
#[test]
fn test_a_literal_past_every_integer_type_is_refused() {
    let err =
        zyron_parser::parse("INSERT INTO t VALUES (1701411834604692317316873037158841057270000)")
            .expect_err("a literal past i128 must be refused");
    assert!(
        err.to_string().contains("too large"),
        "the refusal says what happened: {err}"
    );

    // And negated, where the sign is folded onto the magnitude
    let err =
        zyron_parser::parse("INSERT INTO t VALUES (-1701411834604692317316873037158841057270000)")
            .expect_err("a negative literal past i128 must be refused");
    assert!(err.to_string().contains("too large"), "{err}");

    // The boundary itself parses, in both directions
    for sql in [
        "SELECT 170141183460469231731687303715884105727",
        "SELECT -170141183460469231731687303715884105728",
    ] {
        zyron_parser::parse(sql).unwrap_or_else(|e| panic!("{sql} should parse: {e}"));
    }
}

/// A wide literal also reaches a DECIMAL column, which is i128 backed and
/// scales the value on the way in.
#[tokio::test]
async fn test_a_wide_literal_reaches_a_decimal_column() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(&server, &mut session, "CREATE TABLE t (v DECIMAL(30,2))")
        .await
        .expect("create");
    exec_dml(&server, "INSERT INTO t VALUES (9223372036854775808)").await;

    let stored = as_i128(&query_values(&server, "SELECT v FROM t").await);
    assert_eq!(
        zyron_common::format_decimal(stored[0], 2),
        "9223372036854775808.00",
        "the wide whole number lands on the column's scale"
    );
}

#[tokio::test]
async fn test_a_128_bit_value_that_fits_narrows_to_bigint() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(&server, &mut session, "CREATE TABLE t (id INT, v INT128)")
        .await
        .expect("create");
    exec_dml(&server, "INSERT INTO t VALUES (1, 42), (2, -7), (3, 0)").await;

    assert_eq!(
        as_i64(&query_values(&server, "SELECT CAST(v AS BIGINT) FROM t ORDER BY id").await),
        vec![42, -7, 0]
    );
}

/// The widest value a BIGINT holds still narrows, so the bound itself is
/// inclusive rather than off by one.
#[tokio::test]
async fn test_the_widest_value_a_bigint_holds_still_narrows() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(&server, &mut session, "CREATE TABLE t (id INT, v INT128)")
        .await
        .expect("create");
    exec_dml(
        &server,
        "INSERT INTO t VALUES (1, 9223372036854775807), (2, -9223372036854775808)",
    )
    .await;

    assert_eq!(
        as_i64(&query_values(&server, "SELECT CAST(v AS BIGINT) FROM t ORDER BY id").await),
        vec![i64::MAX, i64::MIN]
    );
}

/// A value past the 64 bit range is refused rather than wrapped into a
/// different number.
#[tokio::test]
async fn test_a_value_past_the_bigint_range_is_refused() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(&server, &mut session, "CREATE TABLE t (id INT, v INT128)")
        .await
        .expect("create");
    // One past i64::MAX
    exec_dml(&server, "INSERT INTO t VALUES (1, 9223372036854775808)").await;

    let err = exec_dml_result(
        &server,
        "INSERT INTO t SELECT 2, CAST(v AS BIGINT) FROM t WHERE id = 1",
    )
    .await
    .expect_err("a value past the range must be refused");
    assert!(
        err.to_string().contains("out of range"),
        "the refusal says what happened: {err}"
    );
}

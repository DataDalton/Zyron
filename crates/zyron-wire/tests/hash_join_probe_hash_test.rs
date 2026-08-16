//! Hash join build and probe sides must hash keys with the same function.
//!
//! The build side picks between a fused integer hash and the generic
//! column hash depending on its own key shape. The probe side has to
//! follow that choice, not re-derive one from the probe key's shape. A
//! build side with a NULL-bearing or expression key hashes generically,
//! and a clean integer probe side that hashes with the integer function
//! then looks up buckets that were never filled, so the join returns
//! zero matches and RIGHT/FULL joins emit spurious NULL-padded rows.
//!
//! Run: cargo test -p zyron-wire --test hash_join_probe_hash_test

mod common;

use common::{create_test_server, exec_ddl, exec_dml, query_values};
use zyron_executor::column::ScalarValue;

fn as_i64_pairs(rows: &[Vec<ScalarValue>]) -> Vec<(i64, i64)> {
    let mut out: Vec<(i64, i64)> = rows
        .iter()
        .map(|r| {
            let a = match &r[0] {
                ScalarValue::Int64(v) => *v,
                other => panic!("expected Int64, got {other:?}"),
            };
            let b = match &r[1] {
                ScalarValue::Int64(v) => *v,
                other => panic!("expected Int64, got {other:?}"),
            };
            (a, b)
        })
        .collect();
    out.sort();
    out
}

/// A NULL key on the build side must not change how matching rows hash.
#[tokio::test]
async fn test_null_bearing_build_side_still_matches_clean_probe() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = common::new_session();
    exec_ddl(&server, &mut session, "CREATE TABLE l (a BIGINT, x BIGINT)")
        .await
        .expect("create l");
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE r (b BIGINT NOT NULL, y BIGINT)",
    )
    .await
    .expect("create r");
    exec_dml(&server, "INSERT INTO l VALUES (1, 10), (2, 20), (NULL, 30)").await;
    exec_dml(&server, "INSERT INTO r VALUES (1, 100), (2, 200)").await;

    let rows = query_values(&server, "SELECT l.x, r.y FROM l JOIN r ON l.a = r.b").await;
    assert_eq!(
        as_i64_pairs(&rows),
        vec![(10, 100), (20, 200)],
        "inner join lost matches when the build side carries a NULL key"
    );
}

/// An expression key on the build side hashes generically, and the clean
/// integer probe side has to follow.
#[tokio::test]
async fn test_expression_build_key_still_matches_clean_probe() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = common::new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE l (a BIGINT NOT NULL, x BIGINT)",
    )
    .await
    .expect("create l");
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE r (b BIGINT NOT NULL, y BIGINT)",
    )
    .await
    .expect("create r");
    exec_dml(&server, "INSERT INTO l VALUES (0, 10), (1, 20)").await;
    exec_dml(&server, "INSERT INTO r VALUES (1, 100), (2, 200)").await;

    let rows = query_values(&server, "SELECT l.x, r.y FROM l JOIN r ON l.a + 1 = r.b").await;
    assert_eq!(
        as_i64_pairs(&rows),
        vec![(10, 100), (20, 200)],
        "inner join lost matches when the build key is an expression"
    );
}

/// RIGHT JOIN with a NULL-bearing build side: matched probe rows must not
/// come back NULL-padded.
#[tokio::test]
async fn test_right_join_null_bearing_build_side_pads_only_true_misses() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = common::new_session();
    exec_ddl(&server, &mut session, "CREATE TABLE l (a BIGINT, x BIGINT)")
        .await
        .expect("create l");
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE r (b BIGINT NOT NULL, y BIGINT)",
    )
    .await
    .expect("create r");
    exec_dml(&server, "INSERT INTO l VALUES (1, 10), (NULL, 30)").await;
    exec_dml(&server, "INSERT INTO r VALUES (1, 100), (7, 700)").await;

    let rows = query_values(&server, "SELECT l.x, r.y FROM l RIGHT JOIN r ON l.a = r.b").await;
    assert_eq!(rows.len(), 2, "one match plus one true miss");
    let mut matched = 0;
    let mut padded = 0;
    for r in &rows {
        match (&r[0], &r[1]) {
            (ScalarValue::Int64(10), ScalarValue::Int64(100)) => matched += 1,
            (ScalarValue::Null, ScalarValue::Int64(700)) => padded += 1,
            other => panic!("unexpected row {other:?}"),
        }
    }
    assert_eq!((matched, padded), (1, 1));
}

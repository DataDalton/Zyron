//! Hash join correctness when the build side is chosen at run time.
//!
//! The operator hashes whichever input materialized fewer rows and
//! mirrors LEFT and RIGHT internally, emitting output columns in the
//! original left-then-right order. Every query here has a right side
//! smaller than the left, so the swapped code paths run. The empty-side
//! cases pin the rule that an outer join owes its outer rows even when
//! the other input produced nothing.
//!
//! Run: cargo test -p zyron-wire --test join_build_swap_test

mod common;

use common::{create_test_server, exec_ddl, exec_dml, query_values};
use zyron_executor::column::ScalarValue;

fn as_i64(v: &ScalarValue) -> Option<i64> {
    match v {
        ScalarValue::Int64(x) => Some(*x),
        ScalarValue::Int32(x) => Some(*x as i64),
        ScalarValue::Int16(x) => Some(*x as i64),
        ScalarValue::Null => None,
        other => panic!("expected an integer or null, got {other:?}"),
    }
}

async fn setup(server: &std::sync::Arc<zyron_wire::connection::ServerState>) {
    let mut session = common::new_session();
    exec_ddl(server, &mut session, "CREATE TABLE big (k INT, lv INT)")
        .await
        .expect("create big");
    exec_ddl(server, &mut session, "CREATE TABLE small (k INT, rv INT)")
        .await
        .expect("create small");

    // 300 rows, keys 0..99 each three times.
    let mut tuples: Vec<String> = Vec::new();
    for i in 0..300i64 {
        tuples.push(format!("({}, {})", i % 100, i));
    }
    exec_dml(
        server,
        &format!("INSERT INTO big VALUES {}", tuples.join(", ")),
    )
    .await;

    // 10 rows, nine keys matching three big rows each plus one orphan.
    let mut tuples: Vec<String> = Vec::new();
    for k in (5..=45).step_by(5) {
        tuples.push(format!("({k}, {})", k * 2));
    }
    tuples.push("(1000, 2000)".to_string());
    exec_dml(
        server,
        &format!("INSERT INTO small VALUES {}", tuples.join(", ")),
    )
    .await;
}

#[tokio::test]
async fn test_all_join_types_agree_with_expected_counts_when_the_right_side_is_smaller() {
    let (server, _schema, _tmp) = create_test_server().await;
    setup(&server).await;

    // Nine matching keys, three big rows each.
    let rows = query_values(
        &server,
        "SELECT COUNT(*) FROM big b JOIN small s ON b.k = s.k",
    )
    .await;
    assert_eq!(as_i64(&rows[0][0]), Some(27));

    // Every big row survives, unmatched ones with a null small side.
    let rows = query_values(
        &server,
        "SELECT COUNT(*), COUNT(s.rv) FROM big b LEFT JOIN small s ON b.k = s.k",
    )
    .await;
    assert_eq!(as_i64(&rows[0][0]), Some(300));
    assert_eq!(as_i64(&rows[0][1]), Some(27));

    // Every small row survives, the orphan with a null big side.
    let rows = query_values(
        &server,
        "SELECT COUNT(*), COUNT(b.lv) FROM big b RIGHT JOIN small s ON b.k = s.k",
    )
    .await;
    assert_eq!(as_i64(&rows[0][0]), Some(28));
    assert_eq!(as_i64(&rows[0][1]), Some(27));

    // Both sides survive.
    let rows = query_values(
        &server,
        "SELECT COUNT(*) FROM big b FULL JOIN small s ON b.k = s.k",
    )
    .await;
    assert_eq!(as_i64(&rows[0][0]), Some(301));
}

#[tokio::test]
async fn test_output_columns_keep_left_then_right_order_and_null_sides_land_correctly() {
    let (server, _schema, _tmp) = create_test_server().await;
    setup(&server).await;

    // A matched key: both sides populated, left columns first.
    let rows = query_values(
        &server,
        "SELECT b.k, b.lv, s.k, s.rv FROM big b JOIN small s ON b.k = s.k \
         WHERE b.lv = 5 ORDER BY b.lv",
    )
    .await;
    assert_eq!(rows.len(), 1);
    assert_eq!(as_i64(&rows[0][0]), Some(5));
    assert_eq!(as_i64(&rows[0][1]), Some(5));
    assert_eq!(as_i64(&rows[0][2]), Some(5));
    assert_eq!(as_i64(&rows[0][3]), Some(10));

    // The orphan small row under RIGHT JOIN: big columns null, small kept.
    let rows = query_values(
        &server,
        "SELECT b.k, b.lv, s.k, s.rv FROM big b RIGHT JOIN small s ON b.k = s.k \
         WHERE s.k = 1000",
    )
    .await;
    assert_eq!(rows.len(), 1);
    assert_eq!(as_i64(&rows[0][0]), None);
    assert_eq!(as_i64(&rows[0][1]), None);
    assert_eq!(as_i64(&rows[0][2]), Some(1000));
    assert_eq!(as_i64(&rows[0][3]), Some(2000));

    // An unmatched big row under LEFT JOIN: small columns null.
    let rows = query_values(
        &server,
        "SELECT b.k, s.k, s.rv FROM big b LEFT JOIN small s ON b.k = s.k \
         WHERE b.lv = 1 ORDER BY b.k",
    )
    .await;
    assert_eq!(rows.len(), 1);
    assert_eq!(as_i64(&rows[0][0]), Some(1));
    assert_eq!(as_i64(&rows[0][1]), None);
    assert_eq!(as_i64(&rows[0][2]), None);
}

#[tokio::test]
async fn test_residual_condition_filters_matches_on_the_swapped_path() {
    let (server, _schema, _tmp) = create_test_server().await;
    setup(&server).await;

    // Equi key plus a residual comparison: of the three big rows per
    // matching key (lv = k, k+100, k+200), only lv >= 100 survive.
    let rows = query_values(
        &server,
        "SELECT COUNT(*) FROM big b JOIN small s ON b.k = s.k AND b.lv >= 100",
    )
    .await;
    assert_eq!(as_i64(&rows[0][0]), Some(18));
}

#[tokio::test]
async fn test_outer_joins_still_owe_rows_when_one_input_is_empty() {
    let (server, _schema, _tmp) = create_test_server().await;
    setup(&server).await;
    let mut session = common::new_session();
    exec_ddl(&server, &mut session, "CREATE TABLE nothing (k INT, nv INT)")
        .await
        .expect("create nothing");

    // Empty left input: RIGHT JOIN owes every right row null-padded.
    let rows = query_values(
        &server,
        "SELECT COUNT(*), COUNT(n.nv) FROM nothing n RIGHT JOIN small s ON n.k = s.k",
    )
    .await;
    assert_eq!(as_i64(&rows[0][0]), Some(10));
    assert_eq!(as_i64(&rows[0][1]), Some(0));

    // Empty left input: FULL JOIN likewise.
    let rows = query_values(
        &server,
        "SELECT COUNT(*) FROM nothing n FULL JOIN small s ON n.k = s.k",
    )
    .await;
    assert_eq!(as_i64(&rows[0][0]), Some(10));

    // Empty right input: LEFT JOIN owes every left row null-padded.
    let rows = query_values(
        &server,
        "SELECT COUNT(*), COUNT(n.nv) FROM small s LEFT JOIN nothing n ON s.k = n.k",
    )
    .await;
    assert_eq!(as_i64(&rows[0][0]), Some(10));
    assert_eq!(as_i64(&rows[0][1]), Some(0));

    // Empty both ways under INNER stays empty.
    let rows = query_values(
        &server,
        "SELECT COUNT(*) FROM nothing n JOIN small s ON n.k = s.k",
    )
    .await;
    assert_eq!(as_i64(&rows[0][0]), Some(0));
}

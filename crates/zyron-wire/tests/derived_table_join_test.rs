//! Joining a derived table, where the predicate lives in WHERE rather than
//! in an ON clause.
//!
//! Predicate pushdown decides which side of a join a conjunct belongs to by
//! collecting the table indices each side produces, and that walk only saw
//! `Scan` nodes. A derived table is addressed by the index its enclosing
//! query gave it, which lives on its projection, so the walk reported the
//! indices of the scans *inside* it instead. A predicate joining a base
//! table to a derived one then looked like it touched only the base table
//! and was pushed into that side, where the derived table's column does not
//! exist.
//!
//! `FROM a, b WHERE a.x = b.y` is the shape that breaks, because that is the
//! shape whose predicate reaches pushdown as a Filter. The same query
//! written `FROM a JOIN b ON a.x = b.y` carries the predicate in the join
//! condition and was never affected, which is why this went unnoticed.
//!
//! Run: cargo test -p zyron-wire --test derived_table_join_test

mod common;

use common::{create_test_server, exec_ddl, exec_dml, new_session, query_result};

async fn seeded() -> (
    std::sync::Arc<zyron_wire::connection::ServerState>,
    tempfile::TempDir,
) {
    let (server, _schema, tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(&server, &mut session, "CREATE TABLE t (k INT, v INT)")
        .await
        .expect("create t");
    exec_ddl(&server, &mut session, "CREATE TABLE u (k INT, w INT)")
        .await
        .expect("create u");
    exec_dml(&server, "INSERT INTO t VALUES (1, 10), (1, 20), (2, 5)").await;
    exec_dml(&server, "INSERT INTO u VALUES (1, 100), (2, 200)").await;
    (server, tmp)
}

/// The minimal shape, and the answer it has to give. A comma join and an
/// explicit join of the same two inputs are the same query, so they must
/// return the same rows.
#[tokio::test]
async fn test_a_comma_join_against_a_derived_table_matches_the_explicit_join() {
    let (server, _tmp) = seeded().await;

    let comma = query_result(
        &server,
        "SELECT t.k, t.v FROM t, (SELECT k AS dk FROM u) AS d WHERE t.k = d.dk ORDER BY t.k, t.v",
    )
    .await
    .expect("the comma form executes");

    let explicit = query_result(
        &server,
        "SELECT t.k, t.v FROM t JOIN (SELECT k AS dk FROM u) AS d ON t.k = d.dk ORDER BY t.k, t.v",
    )
    .await
    .expect("the explicit form executes");

    assert_eq!(comma.len(), 3, "every row of t has a matching key in u");
    assert_eq!(
        comma, explicit,
        "the two spellings of one join disagree on the rows"
    );
}

/// The derived table on either side of the comma, since the side decides
/// which way the predicate was pushed.
#[tokio::test]
async fn test_the_derived_table_may_sit_on_either_side_of_the_join() {
    let (server, _tmp) = seeded().await;

    let right = query_result(
        &server,
        "SELECT t.k FROM t, (SELECT k AS dk FROM u) AS d WHERE t.k = d.dk ORDER BY t.k",
    )
    .await
    .expect("derived on the right");
    let left = query_result(
        &server,
        "SELECT t.k FROM (SELECT k AS dk FROM u) AS d, t WHERE t.k = d.dk ORDER BY t.k",
    )
    .await
    .expect("derived on the left");

    assert_eq!(right.len(), 3);
    assert_eq!(
        right, left,
        "which side the derived table sits on changed the rows"
    );
}

/// A derived table that aggregates, which is the form the query set uses and
/// the one whose projection is not a plain passthrough.
#[tokio::test]
async fn test_a_comma_join_against_an_aggregating_derived_table() {
    let (server, _tmp) = seeded().await;

    let rows = query_result(
        &server,
        "SELECT t.k, d.total FROM t, (SELECT k AS dk, SUM(v) AS total FROM t GROUP BY k) AS d \
         WHERE t.k = d.dk ORDER BY t.k, t.v",
    )
    .await
    .expect("executes");
    assert_eq!(
        rows.len(),
        3,
        "three rows of t, each matching its own group"
    );
}

/// A predicate that really does belong to one side is still pushed there.
/// The fix narrows which conjuncts move, and a conjunct over a single table
/// has to keep moving or the pushdown stopped doing its job.
#[tokio::test]
async fn test_a_single_sided_predicate_still_reaches_its_side() {
    let (server, _tmp) = seeded().await;

    let rows = query_result(
        &server,
        "SELECT t.k, d.dk FROM t, (SELECT k AS dk FROM u) AS d \
         WHERE t.k = d.dk AND t.v > 8 AND d.dk < 2 ORDER BY t.k",
    )
    .await
    .expect("executes");
    assert_eq!(
        rows.len(),
        2,
        "k=1 has two rows above v=8, and k=2 is excluded by dk < 2"
    );
}

/// Join reordering has to carry every predicate, not just one.
///
/// The reorder rule picked the first available predicate as each join's
/// condition, so on three or more inputs it repeated that one at every level
/// and dropped the rest, turning the unmatched pairs into a cross product.
#[tokio::test]
async fn test_a_reordered_join_keeps_every_predicate() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    for ddl in [
        "CREATE TABLE a (k INT, v INT)",
        "CREATE TABLE b (k INT, w INT)",
        "CREATE TABLE c (k INT, x INT)",
        "CREATE TABLE e (k INT, y INT)",
    ] {
        exec_ddl(&server, &mut session, ddl).await.expect("create");
    }
    // Each table holds the same keys, so a dropped predicate shows up as a
    // row count that multiplied rather than as an empty result
    for t in ["a", "b", "c", "e"] {
        exec_dml(
            &server,
            &format!("INSERT INTO {t} VALUES (1, 10), (2, 20), (3, 30)"),
        )
        .await;
    }

    let three = query_result(
        &server,
        "SELECT a.k FROM a, b, c WHERE a.k = b.k AND b.k = c.k ORDER BY a.k",
    )
    .await
    .expect("three way join executes");
    assert_eq!(
        three.len(),
        3,
        "one row per matching key, not a cross product"
    );

    let four = query_result(
        &server,
        "SELECT a.k FROM a, b, c, e WHERE a.k = b.k AND b.k = c.k AND c.k = e.k ORDER BY a.k",
    )
    .await
    .expect("four way join executes");
    assert_eq!(four.len(), 3, "one row per matching key across four inputs");

    // A predicate that also filters one input has to survive the reorder
    let filtered = query_result(
        &server,
        "SELECT a.k FROM a, b, c WHERE a.k = b.k AND b.k = c.k AND c.x > 15 ORDER BY a.k",
    )
    .await
    .expect("executes");
    assert_eq!(filtered.len(), 2, "keys 2 and 3 clear the filter");
}

/// Three inputs mixing base and derived tables, so more than one conjunct
/// has to be routed.
#[tokio::test]
async fn test_a_three_way_comma_join_mixing_base_and_derived_tables() {
    let (server, _tmp) = seeded().await;

    let rows = query_result(
        &server,
        "SELECT t.k, u.w, d.total FROM t, u, (SELECT k AS dk, SUM(v) AS total FROM t GROUP BY k) AS d \
         WHERE t.k = u.k AND u.k = d.dk ORDER BY t.k, t.v",
    )
    .await
    .expect("executes");
    assert_eq!(rows.len(), 3);
}

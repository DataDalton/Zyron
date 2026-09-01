//! Temporal keys and temporal referential integrity.
//!
//! Two constraints share the period machinery and are checked here. A
//! WITHOUT OVERLAPS key refuses two rows of one entity whose periods
//! intersect, and it has to keep refusing them under UPDATE, where a row
//! must not collide with the period it is itself replacing. A PERIOD
//! foreign key holds a child period inside the parent's, and the standard
//! reads the parent as the union of its matching rows, so two adjacent
//! parent periods cover a child that spans both.
//!
//! The overlap check reads a backing index rather than the table. Two of
//! these tests exist because an index can answer wrongly in ways a scan
//! cannot: a deleted row leaves its entry behind until vacuum, so the entry
//! alone must not refuse a write, and a deleted period sitting inside a
//! live one must not stop the walk before it reaches the live row.
//!
//! Run: cargo test -p zyron-wire --test temporal_integrity_test

mod common;

use common::{create_test_server, exec_ddl, exec_dml, exec_dml_result, new_session, query_values};
use zyron_executor::column::ScalarValue;

fn count_of(rows: &[Vec<ScalarValue>]) -> i64 {
    match rows.first().and_then(|r| r.first()) {
        Some(ScalarValue::Int64(v)) => *v,
        Some(ScalarValue::Int32(v)) => i64::from(*v),
        other => panic!("expected a count, got {other:?}"),
    }
}

/// Periods that do not touch are all insertable, periods that intersect are
/// not, and periods meeting at a half open boundary do not intersect
#[tokio::test]
async fn test_without_overlaps_admits_disjoint_and_refuses_intersecting() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE booking (room INT, stay DATERANGE, PRIMARY KEY (room, stay WITHOUT OVERLAPS))",
    )
    .await
    .expect("create");

    exec_dml(
        &server,
        "INSERT INTO booking VALUES \
         (1, '[2026-01-01,2026-01-05)'), \
         (1, '[2026-01-05,2026-01-10)'), \
         (1, '[2026-02-01,2026-02-04)')",
    )
    .await;
    assert_eq!(
        count_of(&query_values(&server, "SELECT count(*) FROM booking").await),
        3,
        "three disjoint periods did not all land"
    );

    // A different room shares the dates and is unaffected by the first
    exec_dml(
        &server,
        "INSERT INTO booking VALUES (2, '[2026-01-02,2026-01-04)')",
    )
    .await;

    let err = exec_dml_result(
        &server,
        "INSERT INTO booking VALUES (1, '[2026-01-04,2026-01-06)')",
    )
    .await
    .expect_err("an intersecting period was accepted");
    assert!(
        format!("{err:?}").contains("overlaps"),
        "unexpected error: {err:?}"
    );

    // A period wholly containing two stored ones is still an overlap
    let err = exec_dml_result(
        &server,
        "INSERT INTO booking VALUES (1, '[2025-12-01,2026-03-01)')",
    )
    .await
    .expect_err("a covering period was accepted");
    assert!(format!("{err:?}").contains("overlaps"));

    assert_eq!(
        count_of(&query_values(&server, "SELECT count(*) FROM booking").await),
        4,
        "a refused insert left a row behind"
    );
}

/// A batch whose own rows intersect is refused whole, before anything is
/// written
#[tokio::test]
async fn test_without_overlaps_refuses_a_self_intersecting_batch() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE shift (staff INT, worked DATERANGE, PRIMARY KEY (staff, worked WITHOUT OVERLAPS))",
    )
    .await
    .expect("create");

    let err = exec_dml_result(
        &server,
        "INSERT INTO shift VALUES \
         (7, '[2026-03-01,2026-03-10)'), \
         (7, '[2026-03-08,2026-03-12)')",
    )
    .await
    .expect_err("a self intersecting batch was accepted");
    assert!(format!("{err:?}").contains("overlap"), "{err:?}");
    assert_eq!(
        count_of(&query_values(&server, "SELECT count(*) FROM shift").await),
        0,
        "a refused batch wrote rows"
    );
}

/// An UPDATE moving a period must not read the row's own stored period as a
/// conflict with itself, and must still refuse a move onto another row
#[tokio::test]
async fn test_without_overlaps_on_update_excludes_the_row_itself() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE lease (id INT, unit INT, term DATERANGE,          PRIMARY KEY (unit, term WITHOUT OVERLAPS))",
    )
    .await
    .expect("create");
    exec_dml(
        &server,
        "INSERT INTO lease VALUES (1, 7, '[2026-01-01,2026-02-01)'),          (2, 7, '[2026-03-01,2026-04-01)')",
    )
    .await;

    // Widening a period into free space is allowed: the only stored period
    // it would collide with is the one this row already holds
    exec_dml(
        &server,
        "UPDATE lease SET term = '[2026-01-01,2026-02-15)' WHERE id = 1",
    )
    .await;
    assert_eq!(
        count_of(&query_values(&server, "SELECT count(*) FROM lease").await),
        2
    );

    // Moving it onto the other row is refused
    let err = exec_dml_result(
        &server,
        "UPDATE lease SET term = '[2026-03-15,2026-05-01)' WHERE id = 1",
    )
    .await
    .expect_err("an update onto another period was accepted");
    assert!(format!("{err:?}").contains("overlap"), "{err:?}");

    // And the refused update left the row as it was
    assert_eq!(
        count_of(
            &query_values(
                &server,
                "SELECT count(*) FROM lease WHERE id = 1 AND term = '[2026-01-01,2026-02-15)'"
            )
            .await
        ),
        1,
        "the refused update changed the row"
    );
}

/// A deleted row leaves its index entry behind until vacuum, and that entry
/// must not go on refusing writes the live table no longer conflicts with
#[tokio::test]
async fn test_a_deleted_period_stops_conflicting() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE slot (bay INT, held DATERANGE, PRIMARY KEY (bay, held WITHOUT OVERLAPS))",
    )
    .await
    .expect("create");
    exec_dml(
        &server,
        "INSERT INTO slot VALUES (1, '[2026-05-01,2026-06-01)')",
    )
    .await;
    exec_dml(&server, "DELETE FROM slot WHERE bay = 1").await;

    // The same period again, which only the stale index entry could refuse
    exec_dml(
        &server,
        "INSERT INTO slot VALUES (1, '[2026-05-01,2026-06-01)')",
    )
    .await;
    assert_eq!(
        count_of(&query_values(&server, "SELECT count(*) FROM slot").await),
        1,
        "the period could not be reused after its row was deleted"
    );
}

/// A deleted period sitting inside a live one must not end the index walk
/// early. The walk stops at the first row starting after the new period
/// ends, and a dead row there proves nothing, so it has to keep going and
/// find the live row it sits inside
#[tokio::test]
async fn test_a_dead_period_inside_a_live_one_does_not_hide_it() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE window_seat (seat INT, held DATERANGE, PRIMARY KEY (seat WITHOUT OVERLAPS))",
    )
    .await
    .ok();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE hold (seat INT, held DATERANGE, PRIMARY KEY (seat, held WITHOUT OVERLAPS))",
    )
    .await
    .expect("create");

    // A short period is written and deleted, then a long one that spans it
    // is written in its place
    exec_dml(
        &server,
        "INSERT INTO hold VALUES (1, '[2026-01-10,2026-01-20)')",
    )
    .await;
    exec_dml(&server, "DELETE FROM hold WHERE seat = 1").await;
    exec_dml(
        &server,
        "INSERT INTO hold VALUES (1, '[2026-01-01,2026-03-01)')",
    )
    .await;

    // The dead short period sorts inside the live long one. A write starting
    // before both has to reach the live row rather than stopping at the dead
    // marker
    let err = exec_dml_result(
        &server,
        "INSERT INTO hold VALUES (1, '[2026-01-05,2026-01-08)')",
    )
    .await
    .expect_err("an overlap hidden behind a deleted row was accepted");
    assert!(format!("{err:?}").contains("overlap"), "{err:?}");
    assert_eq!(
        count_of(&query_values(&server, "SELECT count(*) FROM hold").await),
        1
    );
}

/// A period foreign key holds the child inside the parent, and reads the
/// parent as the union of the rows sharing the key
#[tokio::test]
async fn test_period_foreign_key_reads_the_parent_as_a_union() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE employment (emp INT, span DATERANGE, PRIMARY KEY (emp, span WITHOUT OVERLAPS))",
    )
    .await
    .expect("create parent");
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE assignment (emp INT, span DATERANGE, \
         FOREIGN KEY (emp, span) REFERENCES employment (emp, span) PERIOD)",
    )
    .await
    .expect("create child");

    // Two adjacent parent periods, which the standard reads as one covered
    // stretch from January to March
    exec_dml(
        &server,
        "INSERT INTO employment VALUES (1, '[2026-01-01,2026-02-01)'), (1, '[2026-02-01,2026-03-01)')",
    )
    .await;

    // Inside one parent period
    exec_dml(
        &server,
        "INSERT INTO assignment VALUES (1, '[2026-01-05,2026-01-20)')",
    )
    .await;
    // Spanning the boundary between two adjacent parent periods
    exec_dml(
        &server,
        "INSERT INTO assignment VALUES (1, '[2026-01-20,2026-02-10)')",
    )
    .await;
    assert_eq!(
        count_of(&query_values(&server, "SELECT count(*) FROM assignment").await),
        2,
        "a covered child period was refused"
    );

    // Reaching past where the parent stops
    let err = exec_dml_result(
        &server,
        "INSERT INTO assignment VALUES (1, '[2026-02-20,2026-03-10)')",
    )
    .await
    .expect_err("an uncovered child period was accepted");
    assert!(
        format!("{err:?}").to_lowercase().contains("foreign key"),
        "unexpected error: {err:?}"
    );

    // An employee the parent does not know at all
    let err = exec_dml_result(
        &server,
        "INSERT INTO assignment VALUES (2, '[2026-01-05,2026-01-06)')",
    )
    .await
    .expect_err("a child of a missing parent was accepted");
    assert!(
        format!("{err:?}").to_lowercase().contains("foreign key"),
        "{err:?}"
    );

    // A gap between parent periods is not covered even though periods sit on
    // both sides of it
    exec_dml(
        &server,
        "INSERT INTO employment VALUES (3, '[2026-01-01,2026-02-01)')",
    )
    .await;
    exec_dml(
        &server,
        "INSERT INTO employment VALUES (3, '[2026-03-01,2026-04-01)')",
    )
    .await;
    let err = exec_dml_result(
        &server,
        "INSERT INTO assignment VALUES (3, '[2026-01-15,2026-03-15)')",
    )
    .await
    .expect_err("a child spanning a parent gap was accepted");
    assert!(
        format!("{err:?}").to_lowercase().contains("foreign key"),
        "{err:?}"
    );
}

// ---------------------------------------------------------------------------
// The parent side of a period foreign key
// ---------------------------------------------------------------------------

/// Builds a parent whose periods run January to March in two adjacent rows,
/// with a child table declaring whatever referential actions the test needs.
async fn period_fk_tables(
    actions: &str,
) -> (
    std::sync::Arc<zyron_wire::connection::ServerState>,
    tempfile::TempDir,
) {
    let (server, _schema, tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE employment (emp INT, span DATERANGE, PRIMARY KEY (emp, span WITHOUT OVERLAPS))",
    )
    .await
    .expect("create parent");
    exec_ddl(
        &server,
        &mut session,
        &format!(
            "CREATE TABLE assignment (emp INT, span DATERANGE, \
             FOREIGN KEY (emp, span) REFERENCES employment (emp, span) PERIOD {actions})"
        ),
    )
    .await
    .expect("create child");
    exec_dml(
        &server,
        "INSERT INTO employment VALUES (1, '[2026-01-01,2026-02-01)'), (1, '[2026-02-01,2026-03-01)')",
    )
    .await;
    (server, tmp)
}

/// Deleting a parent period a child is still standing on is refused.
///
/// The child period equals neither parent period, it spans the boundary
/// between them, so a check that matched the declared columns for equality
/// found no dependent row and let the delete through. That left the child
/// referencing a stretch of time its parent no longer covered, behind a
/// statement that reported success.
#[tokio::test]
async fn test_deleting_a_parent_period_a_child_stands_on_is_refused() {
    let (server, _tmp) = period_fk_tables("").await;
    exec_dml(
        &server,
        "INSERT INTO assignment VALUES (1, '[2026-01-20,2026-02-10)')",
    )
    .await;

    let err = exec_dml_result(
        &server,
        "DELETE FROM employment WHERE span = '[2026-02-01,2026-03-01)'",
    )
    .await
    .expect_err("the parent period holding a child up was deleted");
    assert!(
        format!("{err:?}").to_lowercase().contains("foreign key"),
        "unexpected error: {err:?}"
    );
    assert_eq!(
        count_of(&query_values(&server, "SELECT count(*) FROM assignment").await),
        1,
        "the child did not survive the refused delete"
    );
    assert_eq!(
        count_of(&query_values(&server, "SELECT count(*) FROM employment").await),
        2,
        "the refused delete removed a parent row anyway"
    );
}

/// A parent period no child is standing on can still go.
///
/// The constraint is about cover, so a period whose removal leaves every
/// child inside what remains is not a dependency at all. Refusing it would
/// make the check a proxy for "any parent row exists" rather than for the
/// thing the standard states.
#[tokio::test]
async fn test_a_parent_period_no_child_needs_can_go() {
    let (server, _tmp) = period_fk_tables("").await;
    // Wholly inside the first parent period
    exec_dml(
        &server,
        "INSERT INTO assignment VALUES (1, '[2026-01-05,2026-01-20)')",
    )
    .await;

    exec_dml(
        &server,
        "DELETE FROM employment WHERE span = '[2026-02-01,2026-03-01)'",
    )
    .await;
    assert_eq!(
        count_of(&query_values(&server, "SELECT count(*) FROM employment").await),
        1,
        "a parent period nothing depended on was refused"
    );
    assert_eq!(
        count_of(&query_values(&server, "SELECT count(*) FROM assignment").await),
        1,
        "the child was removed although its cover was untouched"
    );
}

/// A cascade takes the children that lost cover and leaves the rest.
#[tokio::test]
async fn test_cascade_removes_only_the_children_that_lost_cover() {
    let (server, _tmp) = period_fk_tables("ON DELETE CASCADE").await;
    // The first stays covered by January, the second spans the boundary and
    // needs February as well
    exec_dml(
        &server,
        "INSERT INTO assignment VALUES (1, '[2026-01-05,2026-01-20)'), (1, '[2026-01-20,2026-02-10)')",
    )
    .await;

    exec_dml(
        &server,
        "DELETE FROM employment WHERE span = '[2026-02-01,2026-03-01)'",
    )
    .await;
    let rows = query_values(&server, "SELECT count(*) FROM assignment").await;
    assert_eq!(
        count_of(&rows),
        1,
        "the cascade took the wrong number of children"
    );
    // The survivor is the one January still covers
    let kept = query_values(&server, "SELECT emp FROM assignment").await;
    assert_eq!(kept.len(), 1, "expected one surviving child");
}

/// Shrinking a parent period out from under a child is refused, even though
/// the parent's key never moved.
#[tokio::test]
async fn test_shrinking_a_parent_period_strands_a_child() {
    let (server, _tmp) = period_fk_tables("").await;
    exec_dml(
        &server,
        "INSERT INTO assignment VALUES (1, '[2026-01-05,2026-01-20)')",
    )
    .await;

    let err = exec_dml_result(
        &server,
        "UPDATE employment SET span = '[2026-01-01,2026-01-10)' \
         WHERE span = '[2026-01-01,2026-02-01)'",
    )
    .await
    .expect_err("a parent period shrank away from under its child");
    assert!(
        format!("{err:?}").to_lowercase().contains("foreign key"),
        "unexpected error: {err:?}"
    );
}

/// Shrinking a period that still covers every child is allowed.
#[tokio::test]
async fn test_shrinking_a_parent_period_that_still_covers_is_allowed() {
    let (server, _tmp) = period_fk_tables("").await;
    exec_dml(
        &server,
        "INSERT INTO assignment VALUES (1, '[2026-01-05,2026-01-08)')",
    )
    .await;

    exec_dml(
        &server,
        "UPDATE employment SET span = '[2026-01-01,2026-01-10)' \
         WHERE span = '[2026-01-01,2026-02-01)'",
    )
    .await;
    assert_eq!(
        count_of(&query_values(&server, "SELECT count(*) FROM assignment").await),
        1,
        "a child still inside its parent was disturbed"
    );
}

/// Moving the parent's key takes its children with it. The period is not
/// part of what moves: the child keeps the stretch of time it always had,
/// and lands on the same stretch of the parent's new key.
#[tokio::test]
async fn test_moving_the_parent_key_cascades_the_child() {
    let (server, _tmp) = period_fk_tables("ON UPDATE CASCADE").await;
    exec_dml(
        &server,
        "INSERT INTO assignment VALUES (1, '[2026-01-20,2026-02-10)')",
    )
    .await;

    exec_dml(&server, "UPDATE employment SET emp = 9 WHERE emp = 1").await;
    assert_eq!(
        count_of(&query_values(&server, "SELECT count(*) FROM assignment WHERE emp = 9").await),
        1,
        "the child did not follow its parent's key"
    );
    assert_eq!(
        count_of(&query_values(&server, "SELECT count(*) FROM assignment WHERE emp = 1").await),
        0,
        "the child was left behind on the old key"
    );
}

/// A cascade cannot repair a child whose parent key never moved.
///
/// The parent's period shrank, so there is nowhere to move the child to: the
/// key it references is still the key it has. Writing nothing and reporting
/// success would leave exactly the orphan the constraint exists to prevent,
/// so the statement is refused instead.
#[tokio::test]
async fn test_a_cascade_cannot_invent_cover_for_a_shrunken_period() {
    let (server, _tmp) = period_fk_tables("ON UPDATE CASCADE").await;
    exec_dml(
        &server,
        "INSERT INTO assignment VALUES (1, '[2026-01-05,2026-01-20)')",
    )
    .await;

    let err = exec_dml_result(
        &server,
        "UPDATE employment SET span = '[2026-01-01,2026-01-10)' \
         WHERE span = '[2026-01-01,2026-02-01)'",
    )
    .await
    .expect_err("a cascade claimed to repair a child it could not move");
    assert!(
        format!("{err:?}").to_lowercase().contains("foreign key"),
        "unexpected error: {err:?}"
    );
    assert_eq!(
        count_of(&query_values(&server, "SELECT count(*) FROM assignment").await),
        1,
        "the child was removed by a refused update"
    );
}

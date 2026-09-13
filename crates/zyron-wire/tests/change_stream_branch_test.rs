//! Change streams and table_changes on a branch.
//!
//! A write on a branch is recorded in the branch's own feed, so a stream
//! created on the branch reads the branch's changes and keeps a position of
//! its own, a stream on the table sees nothing of them, table_changes inside
//! the branch reads the table's history up to the branch point and the
//! branch's after it, merging the branch lands its rows as the table's
//! changes and consumes the branch's streams with the branch, and dropping
//! the branch drops the streams created on it.
//!
//! Run: cargo test -p zyron-wire --test change_stream_branch_test -- --nocapture

use std::sync::Arc;

use zyron_executor::column::ScalarValue;
use zyron_wire::connection::ServerState;
use zyron_wire::session::Session;

mod common;
use common::*;

async fn seeded(server: &Arc<ServerState>, session: &mut Option<Session>) {
    exec_ddl(
        server,
        session,
        "CREATE TABLE orders (id BIGINT PRIMARY KEY, total BIGINT)",
    )
    .await
    .expect("create orders");
    exec_ddl(
        server,
        session,
        "ALTER TABLE orders SET (change_data_feed = true)",
    )
    .await
    .expect("turn the feed on");
    exec_dml(server, "INSERT INTO orders VALUES (1, 10)").await;
    exec_dml(server, "INSERT INTO orders VALUES (2, 20)").await;
}

/// The same table and rows in the lake, whose changes on a branch are
/// derived from the head the branch keeps on it
async fn seeded_lake(server: &Arc<ServerState>, session: &mut Option<Session>) {
    exec_ddl(
        server,
        session,
        "CREATE TABLE orders (id BIGINT NOT NULL, total BIGINT) USING ZYRONLAKE",
    )
    .await
    .expect("create orders");
    exec_ddl(
        server,
        session,
        "ALTER TABLE orders SET (change_data_feed = true)",
    )
    .await
    .expect("turn the feed on");
    exec_dml(server, "INSERT INTO orders VALUES (1, 10)").await;
    exec_dml(server, "INSERT INTO orders VALUES (2, 20)").await;
}

fn stream(server: &Arc<ServerState>, name: &str) -> Option<Arc<zyron_catalog::ChangeStreamEntry>> {
    server
        .catalog
        .list_change_streams()
        .into_iter()
        .find(|e| e.name == name)
}

fn consumed(server: &Arc<ServerState>, name: &str) -> u64 {
    let table_id = table_id_of(server, "orders");
    stream(server, name)
        .expect("the stream exists")
        .consumed_of(table_id)
}

fn pending(server: &Arc<ServerState>, name: &str) -> u64 {
    let feeds = server.cdc_registry.as_ref().expect("feeds");
    let runtime = zyron_cdc::change_stream::ChangeStreamRuntime::new(Arc::clone(feeds));
    runtime.pending_rows(&stream(server, name).expect("the stream exists"))
}

fn ids(rows: &[Vec<ScalarValue>]) -> Vec<i64> {
    let mut out: Vec<i64> = rows
        .iter()
        .filter_map(|r| match r.first() {
            Some(ScalarValue::Int64(n)) => Some(*n),
            _ => None,
        })
        .collect();
    out.sort_unstable();
    out
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_stream_on_a_branch_reads_the_branch_and_keeps_its_own_position() {
    let (server, _schema, _tmp) = create_test_server_with_cdc_and_branches().await;
    let mut session = new_session();
    seeded(&server, &mut session).await;
    exec_ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM main_changes ON TABLE orders",
    )
    .await
    .expect("a stream on the table");
    exec_ddl(&server, &mut session, "CREATE BRANCH dev")
        .await
        .expect("create the branch");
    exec_ddl_on_branch(
        &server,
        &mut session,
        "CREATE CHANGE STREAM dev_changes ON TABLE orders",
        "dev",
    )
    .await
    .expect("a stream on the branch");
    let dev = stream(&server, "dev_changes").expect("the branch stream");
    assert!(dev.branch.is_some(), "the stream is bound to its branch");
    assert!(
        stream(&server, "main_changes")
            .expect("the main stream")
            .branch
            .is_none()
    );

    // Writes on the branch pend for the branch's stream and not the table's
    run_on_branch(&server, "INSERT INTO orders VALUES (3, 30)", "dev")
        .await
        .expect("a branch write");
    run_on_branch(&server, "INSERT INTO orders VALUES (4, 40)", "dev")
        .await
        .expect("a branch write");
    assert_eq!(pending(&server, "dev_changes"), 2);
    assert_eq!(pending(&server, "main_changes"), 0);

    // Reading the branch's stream yields the branch's changes and moves the
    // branch's position, leaving the table's stream where it was
    let rows = run_on_branch(&server, "SELECT id FROM dev_changes", "dev")
        .await
        .expect("read the branch stream");
    assert_eq!(ids(&rows), vec![3, 4]);
    assert_eq!(consumed(&server, "dev_changes"), 2);
    assert_eq!(
        consumed(&server, "main_changes"),
        2,
        "the table's stream stands where it was"
    );
    assert_eq!(pending(&server, "dev_changes"), 0);

    // A write on the table pends for the table's stream and not the branch's
    exec_dml(&server, "INSERT INTO orders VALUES (5, 50)").await;
    assert_eq!(pending(&server, "main_changes"), 1);
    assert_eq!(pending(&server, "dev_changes"), 0);
    let rows = query_values(&server, "SELECT id FROM main_changes").await;
    assert_eq!(ids(&rows), vec![5]);

    // The branch shows in the operator view
    let (fields, rows) = zyron_wire::system_views::query_system_view(
        "zyron_sys.cdc.change_streams",
        &server,
        &zyron_wire::system_views::SystemViewFilters::default(),
    )
    .await
    .expect("the view answers")
    .expect("a system view");
    let at = fields
        .iter()
        .position(|f| f.name == "branch")
        .expect("a branch column");
    let name_at = fields
        .iter()
        .position(|f| f.name == "stream")
        .expect("a stream column");
    let branch_of = |name: &str| -> String {
        rows.iter()
            .find(|r| {
                r[name_at]
                    .as_deref()
                    .map(|b| String::from_utf8_lossy(b).into_owned())
                    == Some(name.to_string())
            })
            .and_then(|r| {
                r[at]
                    .as_deref()
                    .map(|b| String::from_utf8_lossy(b).into_owned())
            })
            .unwrap_or_default()
    };
    assert_eq!(branch_of("dev_changes"), "dev");
    assert_eq!(branch_of("main_changes"), "");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn table_changes_inside_a_branch_reads_the_table_up_to_the_branch_point_and_the_branch_after()
{
    let (server, _schema, _tmp) = create_test_server_with_cdc_and_branches().await;
    let mut session = new_session();
    seeded(&server, &mut session).await;
    exec_ddl(&server, &mut session, "CREATE BRANCH dev")
        .await
        .expect("create the branch");
    run_on_branch(&server, "INSERT INTO orders VALUES (3, 30)", "dev")
        .await
        .expect("a branch write");
    exec_dml(&server, "INSERT INTO orders VALUES (4, 40)").await;

    // Inside the branch, the table's history to the branch point, then the
    // branch's own
    let rows = run_on_branch(
        &server,
        "SELECT id FROM table_changes(orders, 0, LATEST)",
        "dev",
    )
    .await
    .expect("read inside the branch");
    assert_eq!(ids(&rows), vec![1, 2, 3]);

    // On the table, its own history, the branch's write not among it
    let rows = query_values(&server, "SELECT id FROM table_changes(orders, 0, LATEST)").await;
    assert_eq!(ids(&rows), vec![1, 2, 4]);

    // A range wholly before the branch point reads the same either way
    let table_id = table_id_of(&server, "orders");
    let point = server
        .cdc_registry
        .as_ref()
        .expect("feeds")
        .branch_feed(table_id, {
            server
                .branch_manager
                .as_ref()
                .expect("branches")
                .get_branch_by_name("dev")
                .expect("the branch")
                .id
                .0
        })
        .expect("the branch feed")
        .branch_point();
    let rows = run_on_branch(
        &server,
        &format!("SELECT id FROM table_changes(orders, 0, {point})"),
        "dev",
    )
    .await
    .expect("read the shared history inside the branch");
    assert_eq!(ids(&rows), vec![1, 2]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn merging_lands_changes_on_the_table_and_dropping_the_branch_drops_its_streams() {
    let (server, _schema, _tmp) = create_test_server_with_cdc_and_branches().await;
    let mut session = new_session();
    seeded(&server, &mut session).await;
    exec_ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM main_changes ON TABLE orders",
    )
    .await
    .expect("a stream on the table");
    exec_ddl(&server, &mut session, "CREATE BRANCH dev")
        .await
        .expect("create the branch");
    exec_ddl_on_branch(
        &server,
        &mut session,
        "CREATE CHANGE STREAM dev_changes ON TABLE orders",
        "dev",
    )
    .await
    .expect("a stream on the branch");
    run_on_branch(&server, "INSERT INTO orders VALUES (3, 30)", "dev")
        .await
        .expect("a branch write");
    run_on_branch(&server, "DELETE FROM orders WHERE id = 1", "dev")
        .await
        .expect("a branch delete");
    assert_eq!(pending(&server, "dev_changes"), 2);
    assert_eq!(pending(&server, "main_changes"), 0);

    // Merging consumes the branch. The rows it lands on the table are the
    // table's changes, so the table's stream sees them, and the streams
    // that read the branch go with the branch rather than being merged
    // into anything
    exec_ddl(&server, &mut session, "MERGE BRANCH dev INTO main")
        .await
        .expect("merge the branch");
    assert!(
        stream(&server, "dev_changes").is_none(),
        "the branch's stream went with it"
    );
    assert!(stream(&server, "main_changes").is_some());
    let rows = query_values(
        &server,
        "SELECT id, _change_type FROM main_changes ORDER BY _commit_version, _change_ordinal",
    )
    .await;
    let seen: Vec<(i64, String)> = rows
        .iter()
        .filter_map(|r| match (r.first(), r.get(1)) {
            (Some(ScalarValue::Int64(id)), Some(ScalarValue::Utf8(kind))) => {
                Some((*id, kind.clone()))
            }
            _ => None,
        })
        .collect();
    assert!(
        seen.contains(&(1, "delete".to_string())),
        "the merge's delete reached the table's stream: {seen:?}"
    );
    assert!(
        seen.contains(&(3, "insert".to_string())),
        "the merge's insert reached the table's stream: {seen:?}"
    );

    // A branch dropped without a merge takes its streams and its feeds and
    // leaves the table's stream where it stands
    exec_ddl(&server, &mut session, "CREATE BRANCH scratch")
        .await
        .expect("create another branch");
    exec_ddl_on_branch(
        &server,
        &mut session,
        "CREATE CHANGE STREAM scratch_changes ON TABLE orders",
        "scratch",
    )
    .await
    .expect("a stream on the branch");
    run_on_branch(&server, "INSERT INTO orders VALUES (9, 90)", "scratch")
        .await
        .expect("a branch write");
    let scratch_id = server
        .branch_manager
        .as_ref()
        .expect("branches")
        .get_branch_by_name("scratch")
        .expect("the branch")
        .id
        .0;
    let main_before = consumed(&server, "main_changes");
    exec_ddl(&server, &mut session, "DROP BRANCH scratch")
        .await
        .expect("drop the branch");
    assert!(
        stream(&server, "scratch_changes").is_none(),
        "the branch's stream went with it"
    );
    assert!(
        stream(&server, "main_changes").is_some(),
        "the table's stream stays"
    );
    assert_eq!(consumed(&server, "main_changes"), main_before);
    assert_eq!(
        pending(&server, "main_changes"),
        0,
        "the dropped branch's write never lands"
    );
    let table_id = table_id_of(&server, "orders");
    let feeds = server.cdc_registry.as_ref().expect("feeds");
    assert!(
        feeds.branch_feed(table_id, scratch_id).is_none(),
        "the branch's feed is gone"
    );
}

/// A stream on a branch over a lake table reads the head the branch keeps
/// on the table, derived from the branch's own versions, and keeps a
/// position of its own, while the table's stream sees nothing of the
/// branch and the branch's stream nothing of the table. Merging lands the
/// branch's rows as the table's changes and takes the branch's stream
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_stream_on_a_branch_reads_a_lake_tables_branch_and_keeps_its_own_position() {
    let (server, _schema, _tmp) = create_test_server_with_cdc_and_branches().await;
    let mut session = new_session();
    seeded_lake(&server, &mut session).await;
    exec_ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM main_changes ON TABLE orders",
    )
    .await
    .expect("a stream on the table");
    exec_ddl(&server, &mut session, "CREATE BRANCH dev")
        .await
        .expect("create the branch");
    exec_ddl_on_branch(
        &server,
        &mut session,
        "CREATE CHANGE STREAM dev_changes ON TABLE orders",
        "dev",
    )
    .await
    .expect("a stream on the branch");
    let dev = stream(&server, "dev_changes").expect("the branch stream");
    assert!(dev.branch.is_some(), "the stream is bound to its branch");

    // Writes on the branch are commits on the branch's head, pending for
    // the branch's stream and not the table's
    run_on_branch(&server, "INSERT INTO orders VALUES (3, 30)", "dev")
        .await
        .expect("a branch write");
    run_on_branch(&server, "INSERT INTO orders VALUES (4, 40)", "dev")
        .await
        .expect("a branch write");
    assert_eq!(pending(&server, "dev_changes"), 2);
    assert_eq!(pending(&server, "main_changes"), 0);

    // Reading the branch's stream yields the branch's changes and moves the
    // branch's position, leaving the table's stream where it was
    let rows = run_on_branch(&server, "SELECT id FROM dev_changes", "dev")
        .await
        .expect("read the branch stream");
    assert_eq!(ids(&rows), vec![3, 4]);
    assert_eq!(consumed(&server, "dev_changes"), 2);
    assert_eq!(
        consumed(&server, "main_changes"),
        2,
        "the table's stream stands where it was"
    );
    assert_eq!(pending(&server, "dev_changes"), 0);

    // A write on the table pends for the table's stream and not the branch's
    exec_dml(&server, "INSERT INTO orders VALUES (5, 50)").await;
    assert_eq!(pending(&server, "main_changes"), 1);
    assert_eq!(pending(&server, "dev_changes"), 0);
    let rows = query_values(&server, "SELECT id FROM main_changes").await;
    assert_eq!(ids(&rows), vec![5]);

    // table_changes inside the branch reads the table's history up to the
    // fork and the branch's after it
    let rows = run_on_branch(
        &server,
        "SELECT id FROM table_changes(orders, 0, LATEST)",
        "dev",
    )
    .await
    .expect("table_changes inside the branch");
    assert_eq!(ids(&rows), vec![1, 2, 3, 4]);

    // Merging lands the branch's rows as the table's changes and takes the
    // branch's stream with the branch
    exec_ddl(&server, &mut session, "MERGE BRANCH dev INTO main")
        .await
        .expect("merge the branch");
    assert!(
        stream(&server, "dev_changes").is_none(),
        "the branch's stream went with it"
    );
    let rows = query_values(&server, "SELECT id FROM main_changes").await;
    assert_eq!(
        ids(&rows),
        vec![3, 4],
        "the merge's rows reach the table's stream"
    );
    let table_id = table_id_of(&server, "orders");
    let feeds = server.cdc_registry.as_ref().expect("feeds");
    assert!(
        feeds.derived_on(table_id, dev.branch).is_none(),
        "the branch's source is gone"
    );
}

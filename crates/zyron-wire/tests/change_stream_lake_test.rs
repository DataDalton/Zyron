//! Change streams over a lake table.
//!
//! A lake table records no change file. Its transaction log is the change
//! record, so the same reads, streams, positions and applies work over a
//! lake table as over a heap table, the position a commit version paired
//! with the count of records at or below it. A compaction rewrites files
//! and changes no row, so it yields no change. A stream on a lake table is
//! stale exactly when time travel to its position fails.
//!
//! Run: cargo test -p zyron-wire --test change_stream_lake_test -- --nocapture

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
        "CREATE TABLE orders (id BIGINT NOT NULL, region TEXT, total BIGINT) USING ZYRONLAKE",
    )
    .await
    .expect("create the lake table");
    exec_ddl(
        server,
        session,
        "ALTER TABLE orders SET (change_data_feed = true)",
    )
    .await
    .expect("turn the feed on");
}

async fn count(server: &Arc<ServerState>, sql: &str) -> i64 {
    let rows = query_values(server, sql).await;
    match rows.first().and_then(|r| r.first()) {
        Some(ScalarValue::Int64(n)) => *n,
        other => panic!("expected a count from {sql}, got {other:?}"),
    }
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

fn stream(server: &Arc<ServerState>, name: &str) -> Arc<zyron_catalog::ChangeStreamEntry> {
    server
        .catalog
        .list_change_streams()
        .into_iter()
        .find(|e| e.name == name)
        .expect("the stream exists")
}

fn lake_log(server: &Arc<ServerState>) -> Arc<zyron_lake::TransactionLog> {
    let paths = zyron_lake::LakePaths::new(
        server.disk_manager.data_dir(),
        table_id_of(server, "orders"),
    );
    zyron_lake::TransactionLog::lookup_shared(&paths).expect("the lake log is open")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_lake_table_reads_its_changes_from_the_log() {
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    seeded(&server, &mut session).await;
    for id in 1..=3 {
        exec_dml(
            &server,
            &format!("INSERT INTO orders VALUES ({id}, 'eu', {})", id * 10),
        )
        .await;
    }
    exec_dml(&server, "DELETE FROM orders WHERE id = 2").await;

    // Every commit is one version, and its rows are the changes
    let rows = query_values(
        &server,
        "SELECT id, _change_type FROM table_changes(orders, 0, LATEST) ORDER BY _commit_version",
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
    assert_eq!(
        seen,
        vec![
            (1, "insert".to_string()),
            (2, "insert".to_string()),
            (3, "insert".to_string()),
            (2, "delete".to_string()),
        ]
    );

    // A range ends at a commit, so a read from where the last one ended
    // continues it
    let latest = lake_log(&server).latest_version();
    let rows = query_values(
        &server,
        &format!(
            "SELECT id FROM table_changes(orders, {}, LATEST)",
            latest - 1
        ),
    )
    .await;
    assert_eq!(
        ids(&rows),
        vec![2],
        "the delete alone lies in the last commit"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_stream_on_a_lake_table_moves_by_commit_and_applies_into_a_heap_table() {
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    seeded(&server, &mut session).await;
    exec_ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM order_changes ON TABLE orders",
    )
    .await
    .expect("create the stream");
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE dim_orders (id BIGINT PRIMARY KEY, region TEXT, total BIGINT)",
    )
    .await
    .expect("create the target");
    for id in 1..=4 {
        exec_dml(
            &server,
            &format!("INSERT INTO orders VALUES ({id}, 'eu', {})", id * 10),
        )
        .await;
    }
    exec_dml(&server, "UPDATE orders SET total = 44 WHERE id = 4").await;

    // The position is the commit version the stream was created at, and
    // the count that names it is the records at or below it, none, since
    // the feed began there
    let table_id = table_id_of(&server, "orders");
    let before = stream(&server, "order_changes");
    assert_eq!(before.position_of(table_id), 1);
    assert_eq!(before.consumed_of(table_id), 0);

    exec_ddl(
        &server,
        &mut session,
        "APPLY CHANGES INTO dim_orders FROM order_changes KEYS (id) SEQUENCE BY _commit_version",
    )
    .await
    .expect("apply the stream");
    assert_eq!(count(&server, "SELECT COUNT(*) FROM dim_orders").await, 4);
    assert_eq!(
        count(&server, "SELECT total FROM dim_orders WHERE id = 4").await,
        44
    );
    let after = stream(&server, "order_changes");
    assert_eq!(
        after.position_of(table_id),
        lake_log(&server).latest_version()
    );
    // Four inserts and an update's two images, the same count a heap feed
    // would carry
    assert_eq!(
        after.consumed_of(table_id) as i64,
        count(
            &server,
            "SELECT COUNT(*) FROM table_changes(orders, 0, LATEST)"
        )
        .await
    );
    assert_eq!(after.consumed_of(table_id), 6);

    // A second apply finds nothing
    exec_ddl(
        &server,
        &mut session,
        "APPLY CHANGES INTO dim_orders FROM order_changes KEYS (id) SEQUENCE BY _commit_version",
    )
    .await
    .expect("apply again");
    assert_eq!(count(&server, "SELECT COUNT(*) FROM dim_orders").await, 4);

    // A later delete moves the stream one commit and removes the row
    exec_dml(&server, "DELETE FROM orders WHERE id = 1").await;
    exec_ddl(
        &server,
        &mut session,
        "APPLY CHANGES INTO dim_orders FROM order_changes KEYS (id) SEQUENCE BY _commit_version",
    )
    .await
    .expect("apply the delete");
    assert_eq!(count(&server, "SELECT COUNT(*) FROM dim_orders").await, 3);
    assert_eq!(
        stream(&server, "order_changes").position_of(table_id),
        lake_log(&server).latest_version()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_compaction_yields_no_change_and_a_stream_over_it_stays_put() {
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    seeded(&server, &mut session).await;
    for id in 1..=6 {
        exec_dml(
            &server,
            &format!("INSERT INTO orders VALUES ({id}, 'eu', {id})"),
        )
        .await;
    }
    exec_ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM order_changes ON TABLE orders",
    )
    .await
    .expect("create the stream");
    let feeds = server.cdc_registry.as_ref().expect("feeds");
    let runtime = zyron_cdc::change_stream::ChangeStreamRuntime::new(Arc::clone(feeds));
    let before = count(
        &server,
        "SELECT COUNT(*) FROM table_changes(orders, 0, LATEST)",
    )
    .await;
    assert_eq!(before, 6);

    // Six one-row files merge into one, which changes no row. The lake half
    // of OPTIMIZE TABLE is the function the statement reaches
    let log = lake_log(&server);
    zyron_wire::connection::lake_optimize(
        &server,
        &log,
        table_id_of(&server, "orders"),
        false,
        true,
    )
    .await
    .expect("compact the table");
    assert!(
        log.latest_version() > 7,
        "the compaction committed a version of its own"
    );
    assert_eq!(
        count(
            &server,
            "SELECT COUNT(*) FROM table_changes(orders, 0, LATEST)"
        )
        .await,
        before,
        "a compaction produced no change record"
    );
    assert_eq!(
        runtime.pending_rows(&stream(&server, "order_changes")),
        0,
        "nothing pends on a stream after a compaction"
    );
    let rows = query_values(&server, "SELECT id FROM order_changes").await;
    assert!(
        rows.is_empty(),
        "a read after the compaction yields nothing: {rows:?}"
    );

    // The rows are still exactly the ones the inserts made
    assert_eq!(count(&server, "SELECT COUNT(*) FROM orders").await, 6);
}

/// A second lake table with its feed on, and a stream over both
async fn seeded_with_dim(server: &Arc<ServerState>, session: &mut Option<Session>) {
    seeded(server, session).await;
    exec_ddl(
        server,
        session,
        "CREATE TABLE dim (region TEXT PRIMARY KEY, label TEXT) USING ZYRONLAKE",
    )
    .await
    .expect("create the lake dimension");
    exec_ddl(
        server,
        session,
        "ALTER TABLE dim SET (change_data_feed = true)",
    )
    .await
    .expect("feed on dim");
    exec_ddl(
        server,
        session,
        "CREATE CHANGE STREAM both ON TABLES (orders, dim)",
    )
    .await
    .expect("create the stream");
}

/// The source table and first column of every row a read of `both` hands
/// over, sorted, so a read is compared as a set of halves
async fn halves(server: &Arc<ServerState>, sql: &str) -> Vec<(i64, String)> {
    let mut out: Vec<(i64, String)> = query_values(server, sql)
        .await
        .iter()
        .map(|row| {
            let table = match row[0] {
                ScalarValue::Int64(t) => t,
                _ => -1,
            };
            (table, format!("{:?}", row[1]))
        })
        .collect();
    out.sort();
    out
}

/// A transaction that commits to two lake tables is handed over whole by
/// a bounded read. The bound lands on the first table's commit of the
/// transaction and the second table's window rises to its commit there,
/// past a commit the bound alone would have stopped at, and the reads
/// that follow drain the rest one commit at a time
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_bounded_read_over_two_lake_tables_hands_over_a_transaction_whole() {
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    seeded_with_dim(&server, &mut session).await;
    let orders = table_id_of(&server, "orders") as i64;
    let dim = table_id_of(&server, "dim") as i64;

    // A commit to dim alone, then one transaction over both tables, then
    // two more commits to orders alone
    exec_dml(&server, "INSERT INTO dim VALUES ('us', 'America')").await;
    exec_dml_script(
        &server,
        &[
            "INSERT INTO orders VALUES (1, 'eu', 1)",
            "INSERT INTO dim VALUES ('eu', 'Europe')",
        ],
    )
    .await
    .expect("one transaction over both");
    exec_dml(&server, "INSERT INTO orders VALUES (2, 'eu', 2)").await;
    exec_dml(&server, "INSERT INTO orders VALUES (3, 'eu', 3)").await;

    // One record's worth from each source is orders' first commit and
    // dim's first commit. The orders commit belongs to the transaction
    // over both, so dim's window rises to the transaction's commit there
    let peek = "SELECT _source_table, COALESCE(id, 0), region FROM both \
                WITH (peek => true, max_rows => 1)";
    assert_eq!(
        halves(&server, peek).await,
        vec![
            (orders, "Int64(1)".to_string()),
            (dim, "Int64(0)".to_string()),
            (dim, "Int64(0)".to_string()),
        ],
        "the transaction's dim half comes with its orders half"
    );
    let regions: Vec<String> = query_values(&server, peek)
        .await
        .iter()
        .map(|row| format!("{:?}", row[2]))
        .collect();
    assert!(
        regions.iter().filter(|r| r.contains("eu")).count() == 2
            && regions.iter().any(|r| r.contains("us")),
        "{regions:?}"
    );

    // Consumed the same way, then the remaining commits one per read
    let consume = "SELECT _source_table, COALESCE(id, 0), region FROM both WITH (max_rows => 1)";
    assert_eq!(halves(&server, consume).await.len(), 3);
    assert_eq!(
        halves(&server, consume).await,
        vec![(orders, "Int64(2)".to_string())]
    );
    assert_eq!(
        halves(&server, consume).await,
        vec![(orders, "Int64(3)".to_string())]
    );
    assert_eq!(halves(&server, consume).await.len(), 0, "drained");
}

/// A transaction over two lake tables publishes one table's commit and
/// then the other's, and a read landing between the two hands over
/// neither half, the way a read over a heap and a lake table does
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_transaction_over_two_lake_tables_waits_for_both_commits_to_publish() {
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    seeded_with_dim(&server, &mut session).await;
    let peek = "SELECT _source_table, COALESCE(id, 0) FROM both WITH (peek => true)";

    let mut txn = Txn::begin(&server);
    txn.run("INSERT INTO orders VALUES (1, 'eu', 1)")
        .await
        .expect("the orders write");
    txn.run("INSERT INTO dim VALUES ('eu', 'Europe')")
        .await
        .expect("the dim write");
    assert_eq!(query_values(&server, peek).await.len(), 0);
    // The commit record is durable, both lake commits are still pending
    txn.commit_holding_lake().await;
    assert_eq!(query_values(&server, peek).await.len(), 0);

    // The orders commit publishes first
    let orders_log = lake_log(&server);
    orders_log
        .publish(orders_log.head_version())
        .expect("publish the orders commit");
    zyron_wire::connection::refresh_lake_stats(&server, &[Arc::clone(&orders_log)]);
    assert_eq!(
        query_values(&server, peek).await.len(),
        0,
        "the orders half waits for the dim half"
    );

    // Then dim's, and the read hands over both
    let dim_paths =
        zyron_lake::LakePaths::new(server.disk_manager.data_dir(), table_id_of(&server, "dim"));
    let dim_log = zyron_lake::TransactionLog::lookup_shared(&dim_paths).expect("dim's log");
    dim_log
        .publish(dim_log.head_version())
        .expect("publish the dim commit");
    zyron_wire::connection::refresh_lake_stats(&server, &[Arc::clone(&dim_log)]);
    assert_eq!(
        halves(&server, peek).await.len(),
        2,
        "both halves once both are visible"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_lake_stream_is_stale_exactly_when_time_travel_to_its_position_fails() {
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    seeded(&server, &mut session).await;
    exec_ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM early ON TABLE orders AT VERSION 1",
    )
    .await
    .expect("create a stream at the first version");
    for id in 1..=5 {
        exec_dml(
            &server,
            &format!("INSERT INTO orders VALUES ({id}, 'eu', {id})"),
        )
        .await;
    }
    exec_ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM late ON TABLE orders",
    )
    .await
    .expect("create a stream at the newest version");

    // Reclaim the versions below a checkpoint, the way retention does
    let log = lake_log(&server);
    let floor = 4;
    log.checkpoint(floor).expect("checkpoint");
    let removed = log.gc_versions(floor).expect("reclaim");
    assert!(removed > 0, "versions below the floor were reclaimed");

    let feeds = server.cdc_registry.as_ref().expect("feeds");
    let runtime = zyron_cdc::change_stream::ChangeStreamRuntime::new(Arc::clone(feeds));
    // The sweeper flags the stale stream before a reader arrives
    let flagged = runtime.sweep(&server.catalog.list_change_streams());
    assert_eq!(
        flagged.len(),
        1,
        "the sweep found the one stale stream: {flagged:?}"
    );
    zyron_wire::change_stream_dispatch::persist_stream_changes(&server, flagged)
        .await
        .expect("persist");
    let travels = |version: u64| {
        zyron_lake::manifest_as_of(&log, zyron_lake::TimeTravelSpec::Version(version)).is_ok()
    };
    let table_id = table_id_of(&server, "orders");
    for name in ["early", "late"] {
        let entry = stream(&server, name);
        let position = entry.position_of(table_id);
        let stale = runtime.staleness(&entry).is_some();
        assert_eq!(
            stale,
            !travels(position),
            "stream {name} at {position}: stale {stale}, time travel {}",
            travels(position)
        );
    }
    let refused = query_error(&server, "SELECT id FROM early").await;
    assert!(refused.contains("ChangeStreamStale"), "{refused}");
    let rows = query_values(&server, "SELECT id FROM late").await;
    assert!(
        rows.is_empty(),
        "the stream at the newest version reads: {rows:?}"
    );

    // A reset below what the log still stands at is refused, one at it
    // recovers the stream and the read yields the changes above it
    let refused = exec_ddl(
        &server,
        &mut session,
        "ALTER CHANGE STREAM early RESET TO VERSION 1",
    )
    .await
    .expect_err("a reset below the floor is refused");
    assert!(refused.contains("Retention reclaimed"), "{refused}");
    exec_ddl(
        &server,
        &mut session,
        &format!("ALTER CHANGE STREAM early RESET TO VERSION {floor}"),
    )
    .await
    .expect("a reset at the floor");
    let rows = query_values(&server, "SELECT id FROM early").await;
    assert_eq!(
        ids(&rows).len(),
        (lake_log(&server).latest_version() - floor) as usize
    );
}

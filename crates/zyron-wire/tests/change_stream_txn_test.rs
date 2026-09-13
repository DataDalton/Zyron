//! A change stream is a durable position over a change feed, moved in the
//! consumer's own transaction.
//!
//! What these prove is the position model. A read inside a transaction takes
//! the position lock and records where it read to, COMMIT moves the position
//! in the same commit as the consumer's writes, and ROLLBACK, a statement
//! error or a killed session all leave it where it was. Two consumers of one
//! stream never take the same change and never split a batch.
//!
//! Run: cargo test -p zyron-wire --test change_stream_txn_test -- --nocapture

use std::sync::Arc;

use zyron_executor::column::ScalarValue;
use zyron_wire::connection::ServerState;
use zyron_wire::session::Session;

mod common;
use common::*;

async fn seeded(server: &Arc<ServerState>, session: &mut Option<Session>, storage: Storage) {
    exec_ddl(
        server,
        session,
        &storage.create("CREATE TABLE bronze (id BIGINT PRIMARY KEY, region TEXT, total BIGINT)"),
    )
    .await
    .expect("create bronze");
    exec_ddl(
        server,
        session,
        "ALTER TABLE bronze SET (change_data_feed = true)",
    )
    .await
    .expect("turn the feed on");
    exec_ddl(
        server,
        session,
        "CREATE TABLE silver (id BIGINT PRIMARY KEY, region TEXT, total BIGINT)",
    )
    .await
    .expect("create silver");
}

fn stream_position(server: &Arc<ServerState>, name: &str) -> u64 {
    server
        .catalog
        .resolve_change_stream(zyron_catalog::DatabaseId(1), name)
        .expect("the stream exists")
        .lowest_position()
}

fn stream_consumed(server: &Arc<ServerState>, name: &str) -> u64 {
    let entry = server
        .catalog
        .resolve_change_stream(zyron_catalog::DatabaseId(1), name)
        .expect("the stream exists");
    entry.position.iter().map(|p| p.consumed).sum()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_stream_created_now_yields_only_what_follows() {
    for storage in Storage::BOTH {
        a_stream_created_now_yields_only_what_follows_on(storage).await;
    }
}

async fn a_stream_created_now_yields_only_what_follows_on(storage: Storage) {
    println!("{storage}");
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    seeded(&server, &mut session, storage).await;
    exec_dml(&server, "INSERT INTO bronze VALUES (1, 'eu', 10)").await;

    exec_ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM s ON TABLE bronze",
    )
    .await
    .expect("create the stream");
    assert_eq!(
        query_rows(&server, "SELECT * FROM s WITH (peek => true)").await,
        0,
        "nothing pending at creation"
    );

    exec_dml(&server, "INSERT INTO bronze VALUES (2, 'us', 20)").await;
    let rows = query_values(
        &server,
        "SELECT id, _change_type FROM s WITH (peek => true)",
    )
    .await;
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0][0], ScalarValue::Int64(2));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn autocommit_read_and_advance_are_one_transaction() {
    for storage in Storage::BOTH {
        autocommit_read_and_advance_are_one_transaction_on(storage).await;
    }
}

async fn autocommit_read_and_advance_are_one_transaction_on(storage: Storage) {
    println!("{storage}");
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    seeded(&server, &mut session, storage).await;
    exec_ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM s ON TABLE bronze",
    )
    .await
    .expect("create the stream");
    for id in 1..=5 {
        exec_dml(
            &server,
            &format!("INSERT INTO bronze VALUES ({id}, 'eu', {id})"),
        )
        .await;
    }

    let before = stream_position(&server, "s");
    // An INSERT ... SELECT from the stream consumes it in one implicit
    // transaction, which is the shape a one-statement apply takes
    exec_dml_result(
        &server,
        "INSERT INTO silver SELECT id, region, total FROM s",
    )
    .await
    .expect("consume into silver");
    let after = stream_position(&server, "s");
    assert!(after > before, "the position moved once");
    assert_eq!(query_rows(&server, "SELECT * FROM silver").await, 5);

    // An immediate re-run yields nothing and moves nothing
    exec_dml_result(
        &server,
        "INSERT INTO silver SELECT id, region, total FROM s",
    )
    .await
    .expect("the second run is empty");
    assert_eq!(stream_position(&server, "s"), after);
    assert_eq!(query_rows(&server, "SELECT * FROM silver").await, 5);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rollback_leaves_the_position_and_commit_moves_it() {
    for storage in Storage::BOTH {
        rollback_leaves_the_position_and_commit_moves_it_on(storage).await;
    }
}

async fn rollback_leaves_the_position_and_commit_moves_it_on(storage: Storage) {
    println!("{storage}");
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    seeded(&server, &mut session, storage).await;
    exec_ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM s ON TABLE bronze",
    )
    .await
    .expect("create the stream");
    for id in 1..=3 {
        exec_dml(
            &server,
            &format!("INSERT INTO bronze VALUES ({id}, 'eu', {id})"),
        )
        .await;
    }
    let start = stream_position(&server, "s");

    // BEGIN, read, write, ROLLBACK
    let mut txn = Txn::begin(&server);
    let read = txn
        .run("SELECT id, region, total FROM s")
        .await
        .expect("read");
    assert_eq!(read.len(), 3);
    txn.run("INSERT INTO silver SELECT id, region, total FROM s")
        .await
        .expect("write");
    txn.rollback();
    assert_eq!(
        stream_position(&server, "s"),
        start,
        "rollback left the position"
    );
    assert_eq!(query_rows(&server, "SELECT * FROM silver").await, 0);

    // The same with COMMIT
    let mut txn = Txn::begin(&server);
    txn.run("INSERT INTO silver SELECT id, region, total FROM s")
        .await
        .expect("write");
    txn.commit().await;
    assert!(
        stream_position(&server, "s") > start,
        "commit moved the position"
    );
    assert_eq!(query_rows(&server, "SELECT * FROM silver").await, 3);

    // A dropped transaction is a killed session. It aborts and leaves the
    // position where the commit put it
    exec_dml(&server, "INSERT INTO bronze VALUES (4, 'eu', 4)").await;
    let settled = stream_position(&server, "s");
    {
        let mut txn = Txn::begin(&server);
        let read = txn.run("SELECT id FROM s").await.expect("read");
        assert_eq!(read.len(), 1);
    }
    assert_eq!(
        stream_position(&server, "s"),
        settled,
        "a killed session moved nothing"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_statement_error_leaves_the_position() {
    for storage in Storage::BOTH {
        a_statement_error_leaves_the_position_on(storage).await;
    }
}

async fn a_statement_error_leaves_the_position_on(storage: Storage) {
    println!("{storage}");
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    seeded(&server, &mut session, storage).await;
    exec_ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM s ON TABLE bronze",
    )
    .await
    .expect("create the stream");
    exec_dml(&server, "INSERT INTO bronze VALUES (1, 'eu', 1)").await;
    let start = stream_position(&server, "s");

    // The write fails on a duplicate key after the read consumed the stream
    exec_dml(&server, "INSERT INTO silver VALUES (1, 'x', 0)").await;
    let failed = exec_dml_result(
        &server,
        "INSERT INTO silver SELECT id, region, total FROM s",
    )
    .await;
    assert!(failed.is_err(), "the duplicate key fails the statement");
    assert_eq!(
        stream_position(&server, "s"),
        start,
        "a failed statement moved nothing"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reading_a_stream_twice_in_one_transaction_returns_the_same_rows() {
    for storage in Storage::BOTH {
        reading_a_stream_twice_in_one_transaction_returns_the_same_rows_on(storage).await;
    }
}

async fn reading_a_stream_twice_in_one_transaction_returns_the_same_rows_on(storage: Storage) {
    println!("{storage}");
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    seeded(&server, &mut session, storage).await;
    exec_ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM s ON TABLE bronze",
    )
    .await
    .expect("create the stream");
    for id in 1..=4 {
        exec_dml(
            &server,
            &format!("INSERT INTO bronze VALUES ({id}, 'eu', {id})"),
        )
        .await;
    }

    let mut txn = Txn::begin(&server);
    let first = txn
        .run("SELECT id FROM s ORDER BY id")
        .await
        .expect("first read");
    // A change that lands between the two reads is past the window the
    // first read took, so the second read yields the same rows and the
    // change stays pending for the next transaction
    exec_dml(&server, "INSERT INTO bronze VALUES (5, 'eu', 5)").await;
    let second = txn
        .run("SELECT id FROM s ORDER BY id")
        .await
        .expect("second read");
    assert_eq!(
        first, second,
        "the position has not moved inside the transaction"
    );
    assert_eq!(first.len(), 4);
    txn.commit().await;

    let after = query_rows(&server, "SELECT * FROM s WITH (peek => true)").await;
    assert_eq!(
        after, 1,
        "one advance for both reads, past what both of them read"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_consumers_never_process_the_same_change() {
    for storage in Storage::BOTH {
        two_consumers_never_process_the_same_change_on(storage).await;
    }
}

async fn two_consumers_never_process_the_same_change_on(storage: Storage) {
    println!("{storage}");
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    seeded(&server, &mut session, storage).await;
    exec_ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM s ON TABLE bronze",
    )
    .await
    .expect("create the stream");
    for id in 1..=6 {
        exec_dml(
            &server,
            &format!("INSERT INTO bronze VALUES ({id}, 'eu', {id})"),
        )
        .await;
    }

    // A reads and holds
    let mut a = Txn::begin(&server);
    let a_rows = a.run("SELECT id FROM s").await.expect("A reads");
    assert_eq!(a_rows.len(), 6);

    // B waits on the position while A holds it
    let server_b = Arc::clone(&server);
    let b = tokio::spawn(async move {
        let mut b = Txn::begin(&server_b);
        let started = std::time::Instant::now();
        let rows = b.run("SELECT id FROM s").await.expect("B reads");
        b.commit().await;
        (rows.len(), started.elapsed())
    });

    // Meanwhile a peek during A's open transaction returns the pending set
    // without blocking
    let peek = query_rows(&server, "SELECT * FROM s WITH (peek => true)").await;
    assert_eq!(
        peek, 6,
        "a peek sees what is pending while A holds the lock"
    );

    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    // More changes land while A holds the position
    exec_dml(&server, "INSERT INTO bronze VALUES (7, 'eu', 7)").await;
    a.commit().await;

    let (b_rows, waited) = b.await.expect("B ran to completion");
    assert!(
        waited >= std::time::Duration::from_millis(150),
        "B waited for A rather than reading alongside it"
    );
    assert_eq!(
        b_rows, 1,
        "B sees only what A left: the change that landed after A read"
    );
    assert_eq!(
        query_rows(&server, "SELECT * FROM s WITH (peek => true)").await,
        0
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_stream_behind_retention_is_stale_and_reset_recovers_it() {
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    seeded(&server, &mut session, Storage::Heap).await;
    exec_ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM s ON TABLE bronze",
    )
    .await
    .expect("create the stream");
    for id in 1..=20 {
        exec_dml(
            &server,
            &format!("INSERT INTO bronze VALUES ({id}, 'eu', {id})"),
        )
        .await;
    }

    // Retention reclaims everything but the newest change
    let registry = server.cdc_registry.as_ref().expect("cdc is enabled");
    let feed = registry
        .get_feed(table_id_of(&server, "bronze"))
        .expect("the feed is open");
    let newest = feed.latest_version().expect("a newest change");
    feed.purge_before_version(newest).expect("purge");

    // The sweeper flags it before a reader arrives
    let runtime = zyron_wire::change_stream_dispatch::runtime_of(&server).expect("runtime");
    let flagged = runtime.sweep(&server.catalog.list_change_streams());
    assert_eq!(flagged.len(), 1, "the sweep found the stale stream");
    assert!(flagged[0].stale);
    zyron_wire::change_stream_dispatch::persist_stream_changes(&server, flagged)
        .await
        .expect("persist");

    let text = query_error(&server, "SELECT * FROM s").await;
    assert!(text.contains("ChangeStreamStale"), "{text}");
    assert!(text.contains("RESET"), "{text}");

    // A reset outside retention is refused
    let refused = exec_ddl(
        &server,
        &mut session,
        "ALTER CHANGE STREAM s RESET TO VERSION 1",
    )
    .await;
    assert!(
        refused.is_err(),
        "a reset below the oldest change is refused"
    );

    // A reset to what is still held recovers the stream
    exec_ddl(
        &server,
        &mut session,
        &format!("ALTER CHANGE STREAM s RESET TO VERSION {newest}"),
    )
    .await
    .expect("reset inside retention");
    exec_dml(&server, "INSERT INTO bronze VALUES (21, 'eu', 21)").await;
    assert_eq!(
        query_rows(&server, "SELECT * FROM s WITH (peek => true)").await,
        1
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn append_only_skips_updates_and_deletes_without_going_stale() {
    for storage in Storage::BOTH {
        append_only_skips_updates_and_deletes_without_going_stale_on(storage).await;
    }
}

async fn append_only_skips_updates_and_deletes_without_going_stale_on(storage: Storage) {
    println!("{storage}");
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    seeded(&server, &mut session, storage).await;
    exec_ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM s ON TABLE bronze APPEND_ONLY",
    )
    .await
    .expect("create the stream");
    exec_dml(&server, "INSERT INTO bronze VALUES (1, 'eu', 1)").await;
    exec_dml(&server, "INSERT INTO bronze VALUES (2, 'eu', 2)").await;
    exec_dml(&server, "UPDATE bronze SET total = 9 WHERE id = 1").await;
    exec_dml(&server, "DELETE FROM bronze WHERE id = 2").await;

    let rows = query_values(&server, "SELECT _change_type FROM s WITH (peek => true)").await;
    assert_eq!(rows.len(), 2, "inserts only");
    for row in &rows {
        assert_eq!(row[0], ScalarValue::Utf8("insert".to_string()));
    }

    // Reclaiming the updates and deletes takes nothing the stream would
    // yield. A heap feed is purged by version, a lake table's log keeps
    // its versions until vacuumed, which the lake suite covers
    if storage == Storage::Heap {
        let registry = server.cdc_registry.as_ref().expect("cdc is enabled");
        let feed = registry
            .get_feed(table_id_of(&server, "bronze"))
            .expect("the feed is open");
        let newest = feed.latest_version().expect("a newest change");
        feed.purge_before_version(newest).expect("purge");
        let runtime = zyron_wire::change_stream_dispatch::runtime_of(&server).expect("runtime");
        assert!(
            runtime
                .sweep(&server.catalog.list_change_streams())
                .is_empty(),
            "an append-only stream does not go stale over reclaimed updates"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn where_and_columns_narrow_what_the_stream_yields() {
    for storage in Storage::BOTH {
        where_and_columns_narrow_what_the_stream_yields_on(storage).await;
    }
}

async fn where_and_columns_narrow_what_the_stream_yields_on(storage: Storage) {
    println!("{storage}");
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    seeded(&server, &mut session, storage).await;
    exec_ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM eu ON TABLE bronze WHERE region = 'eu' COLUMNS (id, total)",
    )
    .await
    .expect("create the stream");
    exec_dml(&server, "INSERT INTO bronze VALUES (1, 'eu', 1)").await;
    exec_dml(&server, "INSERT INTO bronze VALUES (2, 'us', 2)").await;
    // The predicate reads the postimage for an update, so a row moving into
    // the region yields
    exec_dml(&server, "UPDATE bronze SET region = 'eu' WHERE id = 2").await;
    // And the preimage for a delete, so a row leaving from the region yields
    exec_dml(&server, "DELETE FROM bronze WHERE id = 1").await;

    let rows = query_values(
        &server,
        "SELECT id, total, _change_type FROM eu WITH (peek => true) \
         ORDER BY _commit_version, _change_ordinal",
    )
    .await;
    let kinds: Vec<String> = rows
        .iter()
        .map(|r| match &r[2] {
            ScalarValue::Utf8(k) => k.clone(),
            other => format!("{other:?}"),
        })
        .collect();
    assert_eq!(
        kinds,
        vec!["insert", "update_postimage", "delete"],
        "{rows:?}"
    );

    // The projection is the two named columns plus the metadata
    let wide = query_error(&server, "SELECT region FROM eu WITH (peek => true)").await;
    assert!(
        wide.contains("region"),
        "a column the stream leaves out is unknown: {wide}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_multi_table_stream_reads_to_one_boundary() {
    for storage in Storage::BOTH {
        a_multi_table_stream_reads_to_one_boundary_on(storage).await;
    }
}

async fn a_multi_table_stream_reads_to_one_boundary_on(storage: Storage) {
    println!("{storage}");
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    seeded(&server, &mut session, storage).await;
    exec_ddl(
        &server,
        &mut session,
        &storage.create("CREATE TABLE dim (region TEXT PRIMARY KEY, label TEXT)"),
    )
    .await
    .expect("create dim");
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE dim SET (change_data_feed = true)",
    )
    .await
    .expect("feed on dim");
    exec_ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM both ON TABLES (bronze, dim)",
    )
    .await
    .expect("create the stream");

    // One transaction touches the fact and the dimension
    exec_dml_script(
        &server,
        &[
            "INSERT INTO bronze VALUES (1, 'eu', 1)",
            "INSERT INTO dim VALUES ('eu', 'Europe')",
        ],
    )
    .await
    .expect("one transaction over both");

    let rows = query_values(
        &server,
        "SELECT _source_table, _change_type FROM both WITH (peek => true)",
    )
    .await;
    assert_eq!(rows.len(), 2, "both halves of the transaction");
    let sources: std::collections::HashSet<i64> = rows
        .iter()
        .map(|r| match r[0] {
            ScalarValue::Int64(t) => t,
            _ => -1,
        })
        .collect();
    assert_eq!(sources.len(), 2, "each half names its table");

    // A source that stops changing does not stall the other
    exec_dml(&server, "INSERT INTO bronze VALUES (2, 'eu', 2)").await;
    exec_dml(&server, "INSERT INTO bronze VALUES (3, 'eu', 3)").await;
    let rows = query_rows(&server, "SELECT * FROM both WITH (peek => true)").await;
    assert_eq!(rows, 4);

    // The advance is one catalog write covering every position
    exec_dml_result(
        &server,
        "INSERT INTO silver SELECT id, region, total FROM both WHERE _source_table = 0 OR true",
    )
    .await
    .ok();
    let entry = server
        .catalog
        .resolve_change_stream(zyron_catalog::DatabaseId(1), "both")
        .expect("the stream exists");
    assert_eq!(entry.position.len(), 2, "one position per source");
}

/// A transaction that writes a heap table and a lake table is one
/// transaction to a stream over both.
///
/// The heap rows become visible with the commit record and the lake
/// commit publishes a step later, so a read landing between the two must
/// hand over neither half. Once the lake commit is published the read
/// hands over both, and a rolled back transaction hands over nothing
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_transaction_over_a_heap_and_a_lake_table_is_handed_over_whole() {
    for lake_first in [false, true] {
        a_transaction_over_a_heap_and_a_lake_table_is_handed_over_whole_writing(lake_first).await;
    }
}

async fn a_transaction_over_a_heap_and_a_lake_table_is_handed_over_whole_writing(lake_first: bool) {
    println!("lake written first: {lake_first}");
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    seeded(&server, &mut session, Storage::Heap).await;
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE dim (region TEXT PRIMARY KEY, label TEXT) USING ZYRONLAKE",
    )
    .await
    .expect("create the lake dimension");
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE dim SET (change_data_feed = true)",
    )
    .await
    .expect("feed on dim");
    exec_ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM both ON TABLES (bronze, dim)",
    )
    .await
    .expect("create the stream");
    let peek = "SELECT _source_table, _change_type FROM both WITH (peek => true)";

    let mut txn = Txn::begin(&server);
    let heap = "INSERT INTO bronze VALUES (1, 'eu', 1)";
    let lake = "INSERT INTO dim VALUES ('eu', 'Europe')";
    let (first, second) = if lake_first {
        (lake, heap)
    } else {
        (heap, lake)
    };
    txn.run(first).await.expect("the first write");
    txn.run(second).await.expect("the second write");
    assert_eq!(
        query_values(&server, peek).await.len(),
        0,
        "an open transaction hands over nothing from either table"
    );

    // The commit record is durable and the heap rows are visible, the
    // lake commit is still pending under the transaction
    let txn_id = txn.commit_holding_lake().await;
    assert_eq!(
        query_values(&server, peek).await.len(),
        0,
        "the heap half waits for the lake half"
    );

    let logs = zyron_lake::publish_txn(server.disk_manager.data_dir(), txn_id).expect("publish");
    zyron_wire::connection::refresh_lake_stats(&server, &logs);
    let rows = query_values(&server, peek).await;
    assert_eq!(rows.len(), 2, "both halves once both are visible");
    let sources: std::collections::HashSet<i64> = rows
        .iter()
        .map(|r| match r[0] {
            ScalarValue::Int64(t) => t,
            _ => -1,
        })
        .collect();
    assert_eq!(sources.len(), 2, "each half names its table");

    // A consumer takes both halves in one read and the position covers
    // both sources. The dimension's row has no id, so it takes one of its
    // own in silver
    exec_dml(
        &server,
        "INSERT INTO silver SELECT COALESCE(id, 100), region, COALESCE(total, 0) FROM both",
    )
    .await;
    assert_eq!(query_rows(&server, "SELECT * FROM silver").await, 2);
    assert_eq!(query_values(&server, peek).await.len(), 0, "consumed");

    // A rolled back transaction over both hands over nothing, before or
    // after the rollback
    let mut txn = Txn::begin(&server);
    txn.run("INSERT INTO bronze VALUES (2, 'us', 2)")
        .await
        .expect("the heap write");
    txn.run("INSERT INTO dim VALUES ('us', 'America')")
        .await
        .expect("the lake write");
    assert_eq!(query_values(&server, peek).await.len(), 0);
    txn.rollback();
    assert_eq!(
        query_values(&server, peek).await.len(),
        0,
        "a rolled back transaction never reaches the stream"
    );
    exec_dml(&server, "INSERT INTO dim VALUES ('us', 'America')").await;
    exec_dml(&server, "INSERT INTO bronze VALUES (2, 'us', 2)").await;
    assert_eq!(
        query_values(&server, peek).await.len(),
        2,
        "the writes after it are read"
    );
}

/// A transaction over a heap and a lake table whose heap half stands
/// behind an open transaction's changes is handed over by neither half.
///
/// The heap read stops short of the open transaction's first change, the
/// committed transaction's heap change lies above it, and its lake commit
/// is visible on its own. The read holds the lake half back with the heap
/// half, and hands over both, with the open transaction's change, once
/// that transaction commits
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_lake_half_waits_while_its_heap_half_stands_behind_an_open_transaction() {
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    seeded(&server, &mut session, Storage::Heap).await;
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE dim (region TEXT PRIMARY KEY, label TEXT) USING ZYRONLAKE",
    )
    .await
    .expect("create the lake dimension");
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE dim SET (change_data_feed = true)",
    )
    .await
    .expect("feed on dim");
    exec_ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM both ON TABLES (bronze, dim)",
    )
    .await
    .expect("create the stream");
    let peek = "SELECT _source_table, _change_type FROM both WITH (peek => true)";

    // One transaction writes the heap table and stays open
    let mut open = Txn::begin(&server);
    open.run("INSERT INTO bronze VALUES (9, 'jp', 9)")
        .await
        .expect("the open transaction's heap write");
    // Another writes both tables after it and commits
    exec_dml_script(
        &server,
        &[
            "INSERT INTO bronze VALUES (1, 'eu', 1)",
            "INSERT INTO dim VALUES ('eu', 'Europe')",
        ],
    )
    .await
    .expect("one transaction over both");
    assert_eq!(
        query_values(&server, peek).await.len(),
        0,
        "the lake half waits with the heap half behind the open transaction"
    );

    open.commit().await;
    let rows = query_values(&server, peek).await;
    assert_eq!(rows.len(), 3, "both transactions once the open one commits");
    let dim_rows = rows
        .iter()
        .filter(|r| r[0] == ScalarValue::Int64(table_id_of(&server, "dim") as i64))
        .count();
    assert_eq!(dim_rows, 1, "the lake half comes with the heap half");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_stream_on_a_view_reads_the_base_table_through_the_view() {
    for storage in Storage::BOTH {
        a_stream_on_a_view_reads_the_base_table_through_the_view_on(storage).await;
    }
}

async fn a_stream_on_a_view_reads_the_base_table_through_the_view_on(storage: Storage) {
    println!("{storage}");
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    seeded(&server, &mut session, storage).await;
    exec_ddl(
        &server,
        &mut session,
        "CREATE VIEW eu_orders AS SELECT id, total FROM bronze WHERE region = 'eu'",
    )
    .await
    .expect("create the view");
    exec_ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM v ON VIEW eu_orders",
    )
    .await
    .expect("a single-table view carries a stream");
    exec_dml(&server, "INSERT INTO bronze VALUES (1, 'eu', 1)").await;
    exec_dml(&server, "INSERT INTO bronze VALUES (2, 'us', 2)").await;

    let rows = query_values(&server, "SELECT id, total FROM v WITH (peek => true)").await;
    assert_eq!(rows.len(), 1, "the view's predicate applies");
    assert_eq!(rows[0][0], ScalarValue::Int64(1));

    // A join, an aggregate and a set operation are refused naming themselves
    exec_ddl(
        &server,
        &mut session,
        "CREATE VIEW joined AS SELECT b.id FROM bronze AS b JOIN silver AS s ON b.id = s.id",
    )
    .await
    .expect("create the join view");
    let text = exec_ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM j ON VIEW joined",
    )
    .await
    .expect_err("refused");
    assert!(text.contains("a join"), "{text}");

    exec_ddl(
        &server,
        &mut session,
        "CREATE VIEW totals AS SELECT region, SUM(total) AS t FROM bronze GROUP BY region",
    )
    .await
    .expect("create the aggregate view");
    let text = exec_ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM a ON VIEW totals",
    )
    .await
    .expect_err("refused");
    assert!(text.contains("an aggregate"), "{text}");

    exec_ddl(
        &server,
        &mut session,
        "CREATE VIEW unioned AS SELECT id FROM bronze UNION SELECT id FROM silver",
    )
    .await
    .expect("create the union view");
    let text = exec_ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM u ON VIEW unioned",
    )
    .await
    .expect_err("refused");
    assert!(text.contains("a set operation"), "{text}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn show_initial_rows_seeds_a_target_and_then_continues() {
    for storage in Storage::BOTH {
        show_initial_rows_seeds_a_target_and_then_continues_on(storage).await;
    }
}

async fn show_initial_rows_seeds_a_target_and_then_continues_on(storage: Storage) {
    println!("{storage}");
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        &storage.create("CREATE TABLE existing (id BIGINT PRIMARY KEY, v BIGINT)"),
    )
    .await
    .expect("create the table");
    for id in 1..=5 {
        exec_dml(
            &server,
            &format!("INSERT INTO existing VALUES ({id}, {id})"),
        )
        .await;
    }
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE existing SET (change_data_feed = true)",
    )
    .await
    .expect("turn the feed on");
    exec_ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM seed ON TABLE existing SHOW INITIAL ROWS",
    )
    .await
    .expect("create the stream");
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE copy (id BIGINT PRIMARY KEY, v BIGINT)",
    )
    .await
    .expect("create the copy");

    // The first read yields every existing row as an insert
    let rows = query_values(
        &server,
        "SELECT id, _change_type FROM seed WITH (peek => true)",
    )
    .await;
    assert_eq!(rows.len(), 5, "exactly the existing rows");
    for row in &rows {
        assert_eq!(row[1], ScalarValue::Utf8("insert".to_string()));
    }

    exec_dml_result(&server, "INSERT INTO copy SELECT id, v FROM seed")
        .await
        .expect("seed the copy");
    assert_eq!(query_rows(&server, "SELECT * FROM copy").await, 5);

    // Then it continues incrementally with no definition change
    exec_dml(&server, "INSERT INTO existing VALUES (6, 6)").await;
    let rows = query_values(&server, "SELECT id FROM seed WITH (peek => true)").await;
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0][0], ScalarValue::Int64(6));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_consumed_count_is_the_replicated_form_of_the_position() {
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    seeded(&server, &mut session, Storage::Heap).await;
    exec_ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM s ON TABLE bronze",
    )
    .await
    .expect("create the stream");
    for id in 1..=7 {
        exec_dml(
            &server,
            &format!("INSERT INTO bronze VALUES ({id}, 'eu', {id})"),
        )
        .await;
    }
    assert_eq!(stream_consumed(&server, "s"), 0);
    exec_dml_result(
        &server,
        "INSERT INTO silver SELECT id, region, total FROM s",
    )
    .await
    .expect("consume");
    assert_eq!(
        stream_consumed(&server, "s"),
        7,
        "the count of records consumed is what travels to another member"
    );

    // RESET TO POSITION takes the count form directly
    exec_ddl(
        &server,
        &mut session,
        "ALTER CHANGE STREAM s RESET TO POSITION 3",
    )
    .await
    .expect("reset by count");
    assert_eq!(stream_consumed(&server, "s"), 3);
    assert_eq!(
        query_rows(&server, "SELECT * FROM s WITH (peek => true)").await,
        4
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn disabling_the_feed_marks_streams_stale_without_dropping_them() {
    for storage in Storage::BOTH {
        disabling_the_feed_marks_streams_stale_without_dropping_them_on(storage).await;
    }
}

async fn disabling_the_feed_marks_streams_stale_without_dropping_them_on(storage: Storage) {
    println!("{storage}");
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    seeded(&server, &mut session, storage).await;
    exec_ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM s ON TABLE bronze",
    )
    .await
    .expect("create the stream");
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE bronze SET (change_data_feed = false)",
    )
    .await
    .expect("turn the feed off");

    let entry = server
        .catalog
        .resolve_change_stream(zyron_catalog::DatabaseId(1), "s")
        .expect("the stream still exists");
    assert!(entry.stale);
    assert_eq!(entry.stale_reason, "feed_disabled");

    let text = query_error(&server, "SELECT * FROM s").await;
    assert!(text.contains("feed_disabled"), "{text}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn show_and_drop_change_stream() {
    for storage in Storage::BOTH {
        show_and_drop_change_stream_on(storage).await;
    }
}

async fn show_and_drop_change_stream_on(storage: Storage) {
    println!("{storage}");
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    seeded(&server, &mut session, storage).await;
    exec_ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM one ON TABLE bronze",
    )
    .await
    .expect("create one");
    exec_ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM two ON TABLE bronze",
    )
    .await
    .expect("create two");
    assert_eq!(server.catalog.list_change_streams().len(), 2);

    exec_ddl(&server, &mut session, "SHOW CHANGE STREAMS ON TABLE bronze")
        .await
        .expect("show lists");
    exec_ddl(&server, &mut session, "SHOW CHANGE STREAM one")
        .await
        .expect("show one");
    exec_ddl(&server, &mut session, "DROP CHANGE STREAM one")
        .await
        .expect("drop one");
    assert_eq!(server.catalog.list_change_streams().len(), 1);
    exec_ddl(&server, &mut session, "DROP CHANGE STREAM IF EXISTS one")
        .await
        .expect("drop if exists is quiet");
    assert!(
        exec_ddl(&server, &mut session, "DROP CHANGE STREAM one")
            .await
            .is_err(),
        "a plain drop of a missing stream is refused"
    );

    // Dropping the table drops the streams on it
    exec_ddl(&server, &mut session, "DROP TABLE bronze")
        .await
        .expect("drop the table");
    assert!(server.catalog.list_change_streams().is_empty());
}

/// One system view, rows keyed by a column, cells read as text
async fn view(server: &Arc<ServerState>, name: &str) -> (Vec<String>, Vec<Vec<String>>) {
    let (fields, rows) = zyron_wire::system_views::query_system_view(
        name,
        server,
        &zyron_wire::system_views::SystemViewFilters::default(),
    )
    .await
    .expect("the view answers")
    .unwrap_or_else(|| panic!("{name} is not a system view"));
    let columns: Vec<String> = fields.iter().map(|f| f.name.clone()).collect();
    let rows = rows
        .into_iter()
        .map(|cells| {
            cells
                .into_iter()
                .map(|cell| {
                    cell.map(|b| String::from_utf8_lossy(&b).into_owned())
                        .unwrap_or_default()
                })
                .collect()
        })
        .collect();
    (columns, rows)
}

fn cell<'a>(columns: &[String], row: &'a [String], name: &str) -> &'a str {
    let at = columns
        .iter()
        .position(|c| c == name)
        .unwrap_or_else(|| panic!("no column {name} in {columns:?}"));
    &row[at]
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_observability_views_answer_from_counters() {
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    seeded(&server, &mut session, Storage::Heap).await;
    exec_ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM s ON TABLE bronze",
    )
    .await
    .expect("create the stream");
    for id in 1..=7 {
        exec_dml(
            &server,
            &format!("INSERT INTO bronze VALUES ({id}, 'eu', {id})"),
        )
        .await;
    }

    let (columns, rows) = view(&server, "zyron_sys.cdc.change_streams").await;
    let row = rows
        .iter()
        .find(|r| cell(&columns, r, "stream") == "s")
        .expect("the stream is listed");
    assert_eq!(cell(&columns, row, "pending_rows"), "7");
    assert_eq!(cell(&columns, row, "pending_versions"), "7");
    assert_eq!(cell(&columns, row, "stale"), "f");
    assert_eq!(cell(&columns, row, "mode"), "standard");
    assert!(cell(&columns, row, "sources").contains("bronze"));
    let lag: i64 = cell(&columns, row, "lag_seconds")
        .parse()
        .expect("a number");
    assert!(
        lag >= 0 && lag < 60,
        "the oldest unconsumed change is recent: {lag}"
    );

    let (columns, rows) = view(&server, "zyron_sys.cdc.feeds").await;
    let row = rows
        .iter()
        .find(|r| cell(&columns, r, "table") == "bronze")
        .expect("the feed is listed");
    assert_eq!(cell(&columns, row, "enabled"), "t");
    assert_eq!(cell(&columns, row, "rows"), "7");
    assert_eq!(cell(&columns, row, "before_image"), "t");
    assert_eq!(cell(&columns, row, "compression"), "lz4");
    assert!(cell(&columns, row, "bytes").parse::<u64>().expect("bytes") > 0);

    // An apply run lands in its view, and a consumed stream reads as empty
    exec_ddl(
        &server,
        &mut session,
        "APPLY CHANGES INTO silver FROM s KEYS (id)",
    )
    .await
    .expect("apply");
    let (columns, rows) = view(&server, "zyron_sys.cdc.apply_runs").await;
    let run = rows
        .iter()
        .rev()
        .find(|r| cell(&columns, r, "target") == "silver")
        .expect("the run is listed");
    assert_eq!(cell(&columns, run, "rows_upserted"), "7");
    assert_eq!(cell(&columns, run, "error"), "");
    let (columns, rows) = view(&server, "zyron_sys.cdc.change_streams").await;
    let row = rows
        .iter()
        .find(|r| cell(&columns, r, "stream") == "s")
        .expect("the stream is listed");
    assert_eq!(cell(&columns, row, "pending_rows"), "0");

    let (columns, rows) = view(&server, "zyron_sys.alert.templates").await;
    let names: Vec<&str> = rows.iter().map(|r| cell(&columns, r, "name")).collect();
    for expected in [
        "cdc_stream_lag",
        "cdc_stream_stale",
        "cdc_stream_needs_attention",
        "cdf_retention_pressure",
        "cdc_apply_failed",
    ] {
        assert!(
            names.contains(&expected),
            "{expected} is declared: {names:?}"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_sixty_fifth_stream_on_a_table_is_refused_naming_the_cap() {
    for storage in Storage::BOTH {
        the_sixty_fifth_stream_on_a_table_is_refused_naming_the_cap_on(storage).await;
    }
}

async fn the_sixty_fifth_stream_on_a_table_is_refused_naming_the_cap_on(storage: Storage) {
    println!("{storage}");
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    seeded(&server, &mut session, storage).await;
    for i in 0..64 {
        exec_ddl(
            &server,
            &mut session,
            &format!("CREATE CHANGE STREAM reader_{i} ON TABLE bronze"),
        )
        .await
        .expect("a stream under the cap");
    }
    let refused = exec_ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM reader_64 ON TABLE bronze",
    )
    .await
    .expect_err("the cap refuses the next one");
    assert!(refused.contains("64"), "{refused}");
    assert!(refused.contains("change_streams_per_table"), "{refused}");
    // Dropping one makes room again
    exec_ddl(&server, &mut session, "DROP CHANGE STREAM reader_0")
        .await
        .expect("drop");
    exec_ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM reader_64 ON TABLE bronze",
    )
    .await
    .expect("room again");
}

/// MERGE runs as one generated body of its own, so a MERGE that reads a
/// stream moves the position in the body's commit, and the rows it lands in
/// a target with its feed on are the target's changes
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_merge_from_a_stream_moves_the_position_and_records_its_writes() {
    for storage in Storage::BOTH {
        a_merge_from_a_stream_moves_the_position_and_records_its_writes_on(storage).await;
    }
}

async fn a_merge_from_a_stream_moves_the_position_and_records_its_writes_on(storage: Storage) {
    println!("{storage}");
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    seeded(&server, &mut session, storage).await;
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE silver SET (change_data_feed = true)",
    )
    .await
    .expect("turn the target feed on");
    exec_ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM s ON TABLE bronze",
    )
    .await
    .expect("create the stream");
    for id in 1..=3 {
        exec_dml(
            &server,
            &format!("INSERT INTO bronze VALUES ({id}, 'eu', {id})"),
        )
        .await;
    }

    exec_ddl(
        &server,
        &mut session,
        "MERGE INTO silver USING s ON silver.id = s.id \
         WHEN MATCHED THEN UPDATE SET total = s.total \
         WHEN NOT MATCHED THEN INSERT (id, region, total) VALUES (s.id, s.region, s.total)",
    )
    .await
    .expect("merge the pending changes");
    assert_eq!(query_rows(&server, "SELECT * FROM silver").await, 3);
    assert_eq!(
        query_rows(&server, "SELECT * FROM s WITH (peek => true)").await,
        0,
        "the merge's commit moved the position"
    );
    assert_eq!(
        query_rows(&server, "SELECT * FROM table_changes(silver, 0, LATEST)").await,
        3,
        "the rows the merge landed are the target's changes"
    );

    // An update is two changes, the row before and the row after, and
    // MERGE refuses a source that matches one target row twice. The source
    // that lands updates is the stream read without its preimages, which
    // is a derived table over the stream and moves the position the same
    exec_dml(&server, "UPDATE bronze SET total = 30 WHERE id = 3").await;
    let refused = exec_ddl(
        &server,
        &mut session,
        "MERGE INTO silver USING s ON silver.id = s.id \
         WHEN MATCHED THEN UPDATE SET total = s.total \
         WHEN NOT MATCHED THEN INSERT (id, region, total) VALUES (s.id, s.region, s.total)",
    )
    .await
    .expect_err("both images of the update match the row");
    assert!(refused.contains("2 rows"), "{refused}");
    assert_eq!(
        query_rows(&server, "SELECT * FROM s WITH (peek => true)").await,
        2,
        "the refused merge moved nothing"
    );
    exec_ddl(
        &server,
        &mut session,
        "MERGE INTO silver USING (SELECT id, region, total FROM s \
         WHERE _change_type <> 'update_preimage') AS src ON silver.id = src.id \
         WHEN MATCHED THEN UPDATE SET total = src.total \
         WHEN NOT MATCHED THEN INSERT (id, region, total) VALUES (src.id, src.region, src.total)",
    )
    .await
    .expect("merge the update");
    let totals = query_values(&server, "SELECT total FROM silver WHERE id = 3").await;
    assert_eq!(totals[0][0], ScalarValue::Int64(30));
    assert_eq!(
        query_rows(&server, "SELECT * FROM s WITH (peek => true)").await,
        0
    );
}

/// A scheduled statement runs as a generated body, so one that reads a
/// stream moves the position when it commits and lands its rows as the
/// target's changes
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_scheduled_statement_consumes_a_stream() {
    for storage in Storage::BOTH {
        a_scheduled_statement_consumes_a_stream_on(storage).await;
    }
}

async fn a_scheduled_statement_consumes_a_stream_on(storage: Storage) {
    println!("{storage}");
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    seeded(&server, &mut session, storage).await;
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE silver SET (change_data_feed = true)",
    )
    .await
    .expect("turn the target feed on");
    exec_ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM s ON TABLE bronze",
    )
    .await
    .expect("create the stream");
    for id in 1..=4 {
        exec_dml(
            &server,
            &format!("INSERT INTO bronze VALUES ({id}, 'eu', {id})"),
        )
        .await;
    }
    exec_ddl(
        &server,
        &mut session,
        "CREATE SCHEDULE land EVERY 1 HOURS DO INSERT INTO silver SELECT id, region, total FROM s",
    )
    .await
    .expect("create the schedule");

    // The first sweep past the schedule's start runs it
    let far_future = 10_000_000_000_000_000i64;
    let report = zyron_wire::ddl_dispatch::run_due_schedules(&server, far_future).await;
    assert_eq!(report.executed, 1, "{report:?}");
    assert_eq!(report.failed, 0);
    assert_eq!(query_rows(&server, "SELECT * FROM silver").await, 4);
    assert_eq!(
        query_rows(&server, "SELECT * FROM s WITH (peek => true)").await,
        0,
        "the scheduled statement's commit moved the position"
    );
    assert_eq!(
        query_rows(&server, "SELECT * FROM table_changes(silver, 0, LATEST)").await,
        4,
        "the rows the schedule landed are the target's changes"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_read_over_many_files_keeps_record_order_and_stops_at_an_open_transaction() {
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE wide (id BIGINT PRIMARY KEY, v BIGINT)",
    )
    .await
    .expect("create the table");
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE wide SET (change_data_feed = true)",
    )
    .await
    .expect("turn the feed on");
    let registry = server.cdc_registry.as_ref().expect("cdc is enabled");
    let feed = registry
        .get_feed(table_id_of(&server, "wide"))
        .expect("the feed is open");

    // Six files of committed changes, decoded across the pool's threads and
    // handed over in the order they were recorded
    let mut next = 1i64;
    for _ in 0..6 {
        for _ in 0..40 {
            exec_dml(
                &server,
                &format!("INSERT INTO wide VALUES ({next}, {})", next * 10),
            )
            .await;
            next += 1;
        }
        feed.seal_open_segment().expect("seal the file");
    }
    let spawned_before = zyron_executor::parallel_pool::ParallelPool::global().tasks_spawned();
    let rows = query_values(
        &server,
        "SELECT id, _commit_version, _change_ordinal FROM table_changes(wide, 0, LATEST)",
    )
    .await;
    assert_eq!(rows.len(), 240, "every change of every file");
    assert!(
        zyron_executor::parallel_pool::ParallelPool::global().tasks_spawned() - spawned_before >= 6,
        "one pool task per file"
    );
    let ids: Vec<i64> = rows
        .iter()
        .map(|r| match r[0] {
            ScalarValue::Int64(id) => id,
            _ => panic!("ids are whole numbers"),
        })
        .collect();
    assert_eq!(
        ids,
        (1..=240).collect::<Vec<i64>>(),
        "record order across files"
    );
    let versions: Vec<(i64, i64)> = rows
        .iter()
        .map(|r| match (&r[1], &r[2]) {
            (ScalarValue::Int64(v), ScalarValue::Int64(o)) => (*v, *o),
            _ => panic!("positions are whole numbers"),
        })
        .collect();
    assert!(
        versions.windows(2).all(|w| w[0] < w[1]),
        "commit version then position, ascending"
    );

    // A transaction left open in the middle file stops the read just before
    // its change, and the files behind it are not handed over
    let mut open = Txn::begin(&server);
    open.run("INSERT INTO wide VALUES (1000, 0)")
        .await
        .expect("an uncommitted insert");
    feed.seal_open_segment().expect("seal the file");
    for _ in 0..40 {
        exec_dml(
            &server,
            &format!("INSERT INTO wide VALUES ({next}, {})", next * 10),
        )
        .await;
        next += 1;
    }
    feed.seal_open_segment().expect("seal the file");
    let rows = query_values(&server, "SELECT id FROM table_changes(wide, 0, LATEST)").await;
    assert_eq!(
        rows.len(),
        240,
        "the read stops before the open transaction's change"
    );

    // Once it commits, everything reads, still in order
    open.commit().await;
    let rows = query_values(&server, "SELECT id FROM table_changes(wide, 0, LATEST)").await;
    let ids: Vec<i64> = rows
        .iter()
        .map(|r| match r[0] {
            ScalarValue::Int64(id) => id,
            _ => panic!("ids are whole numbers"),
        })
        .collect();
    let mut expected: Vec<i64> = (1..=240).collect();
    expected.push(1000);
    expected.extend(241..=280);
    assert_eq!(ids, expected);
}

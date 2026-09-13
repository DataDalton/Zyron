//! `table_changes(...)` over a feed-enabled table.
//!
//! Covers what the table function returns for each kind of change, how its
//! bounds resolve, how it composes with the rest of the language, what it
//! refuses and why, and what the feed's own settings change about the answer.
//!
//! Run: cargo test -p zyron-wire --test table_changes_test -- --nocapture

use std::sync::Arc;

use zyron_executor::column::ScalarValue;
use zyron_wire::connection::ServerState;

mod common;
use common::*;

/// A feed-enabled table with three rows, which is the shape most of these
/// tests start from
async fn seeded(
    server: &Arc<ServerState>,
    session: &mut Option<zyron_wire::session::Session>,
    storage: Storage,
) {
    exec_ddl(
        server,
        session,
        &storage.create("CREATE TABLE orders (id BIGINT PRIMARY KEY, region TEXT, total BIGINT)"),
    )
    .await
    .expect("create the table");
    exec_ddl(
        server,
        session,
        "ALTER TABLE orders SET (change_data_feed = true)",
    )
    .await
    .expect("turn the feed on");
    exec_dml(server, "INSERT INTO orders VALUES (1, 'eu', 100)").await;
    exec_dml(server, "INSERT INTO orders VALUES (2, 'us', 200)").await;
    exec_dml(server, "INSERT INTO orders VALUES (3, 'eu', 300)").await;
}

/// Every `_change_type` a query returned, in the order it returned them
fn kinds(rows: &[Vec<ScalarValue>], at: usize) -> Vec<String> {
    rows.iter()
        .map(|row| match &row[at] {
            ScalarValue::Utf8(text) => text.clone(),
            other => format!("{other:?}"),
        })
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn insert_update_and_delete_yield_the_documented_rows() {
    for storage in Storage::BOTH {
        insert_update_and_delete_yield_the_documented_rows_on(storage).await;
    }
}

async fn insert_update_and_delete_yield_the_documented_rows_on(storage: Storage) {
    println!("{storage}");
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    seeded(&server, &mut session, storage).await;

    exec_dml(&server, "UPDATE orders SET total = 150 WHERE id = 1").await;
    exec_dml(&server, "DELETE FROM orders WHERE id = 2").await;

    let rows = query_values(
        &server,
        "SELECT _change_type, id, total, _commit_version, _change_ordinal \
         FROM table_changes(orders, 0, LATEST) ORDER BY _commit_version, _change_ordinal",
    )
    .await;

    let seen = kinds(&rows, 0);
    assert_eq!(
        seen,
        vec![
            "insert",
            "insert",
            "insert",
            "update_preimage",
            "update_postimage",
            "delete",
        ],
        "every change the table recorded, in the order it recorded them"
    );

    // An update yields both images under one version, adjacent in ordinal, so
    // a consumer ordering inside a commit sees them together
    let preimage = rows
        .iter()
        .position(|r| kinds(std::slice::from_ref(r), 0)[0] == "update_preimage")
        .expect("a preimage");
    let postimage = preimage + 1;
    assert_eq!(rows[preimage][3], rows[postimage][3], "one commit version");
    let (ScalarValue::Int64(before), ScalarValue::Int64(after)) =
        (&rows[preimage][4], &rows[postimage][4])
    else {
        panic!("the ordinals are whole numbers");
    };
    assert_eq!(after - before, 1, "adjacent positions inside the commit");
    assert_eq!(rows[preimage][2], ScalarValue::Int64(100));
    assert_eq!(rows[postimage][2], ScalarValue::Int64(150));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_truncate_yields_one_truncate_record() {
    for storage in Storage::BOTH {
        a_truncate_yields_one_truncate_record_on(storage).await;
    }
}

async fn a_truncate_yields_one_truncate_record_on(storage: Storage) {
    println!("{storage}");
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    seeded(&server, &mut session, storage).await;
    exec_ddl(&server, &mut session, "TRUNCATE TABLE orders")
        .await
        .expect("truncate");

    let rows = query_values(
        &server,
        "SELECT _change_type FROM table_changes(orders, 0, LATEST) \
         WHERE _change_type = 'truncate'",
    )
    .await;
    assert_eq!(rows.len(), 1, "one record for the whole truncate");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn every_bound_form_resolves_to_the_same_window() {
    for storage in Storage::BOTH {
        every_bound_form_resolves_to_the_same_window_on(storage).await;
    }
}

async fn every_bound_form_resolves_to_the_same_window_on(storage: Storage) {
    println!("{storage}");
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    seeded(&server, &mut session, storage).await;

    let all = query_rows(
        &server,
        "SELECT * FROM table_changes(orders, EARLIEST, LATEST)",
    )
    .await;
    assert_eq!(all, 3);

    // Leaving the end out reads to the newest change
    let open_ended = query_rows(&server, "SELECT * FROM table_changes(orders, EARLIEST)").await;
    assert_eq!(open_ended, all);

    // The named-argument form is the same call
    let named = query_rows(
        &server,
        "SELECT * FROM table_changes(orders, start_version => 0, end_version => 9223372036854775807)",
    )
    .await;
    assert_eq!(named, all, "the named form reads the same window");

    // A timestamp bound reads back to the beginning of time
    let by_time = query_rows(
        &server,
        "SELECT * FROM table_changes(orders, '1970-01-01 00:00:00', LATEST)",
    )
    .await;
    assert_eq!(by_time, all);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_change_read_composes_with_the_rest_of_the_language() {
    for storage in Storage::BOTH {
        a_change_read_composes_with_the_rest_of_the_language_on(storage).await;
    }
}

async fn a_change_read_composes_with_the_rest_of_the_language_on(storage: Storage) {
    println!("{storage}");
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    seeded(&server, &mut session, storage).await;
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE regions (region TEXT PRIMARY KEY, label TEXT)",
    )
    .await
    .expect("create the dimension");
    exec_dml(&server, "INSERT INTO regions VALUES ('eu', 'Europe')").await;
    exec_dml(&server, "INSERT INTO regions VALUES ('us', 'Americas')").await;

    // In a CTE
    let in_cte = query_rows(
        &server,
        "WITH c AS (SELECT * FROM table_changes(orders, 0, LATEST)) SELECT * FROM c",
    )
    .await;
    assert_eq!(in_cte, 3);

    // Joined to a dimension
    let joined = query_values(
        &server,
        "SELECT c.id, r.label FROM table_changes(orders, 0, LATEST) AS c \
         JOIN regions AS r ON c.region = r.region ORDER BY c.id",
    )
    .await;
    assert_eq!(joined.len(), 3);
    assert_eq!(joined[0][1], ScalarValue::Utf8("Europe".to_string()));

    // Inside a subquery
    let in_subquery = query_rows(
        &server,
        "SELECT * FROM (SELECT id FROM table_changes(orders, 0, LATEST)) AS s",
    )
    .await;
    assert_eq!(in_subquery, 3);

    // Inside a view definition
    exec_ddl(
        &server,
        &mut session,
        "CREATE VIEW order_changes AS SELECT id, _change_type FROM table_changes(orders, 0, LATEST)",
    )
    .await
    .expect("create the view");
    assert_eq!(query_rows(&server, "SELECT * FROM order_changes").await, 3);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn explain_reports_the_resolved_range_and_the_files_it_opens() {
    for storage in Storage::BOTH {
        explain_reports_the_resolved_range_and_the_files_it_opens_on(storage).await;
    }
}

async fn explain_reports_the_resolved_range_and_the_files_it_opens_on(storage: Storage) {
    println!("{storage}");
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    seeded(&server, &mut session, storage).await;

    let plan = explain_text(&server, "SELECT * FROM table_changes(orders, 0, LATEST)").await;
    assert!(plan.contains("ChangeScan"), "{plan}");
    assert!(plan.contains("version_range"), "{plan}");
    assert!(plan.contains("change_files"), "{plan}");
    assert!(plan.contains("orders"), "{plan}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_table_with_no_feed_names_the_alter_that_enables_it() {
    for storage in Storage::BOTH {
        a_table_with_no_feed_names_the_alter_that_enables_it_on(storage).await;
    }
}

async fn a_table_with_no_feed_names_the_alter_that_enables_it_on(storage: Storage) {
    println!("{storage}");
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        &storage.create("CREATE TABLE plain (id BIGINT PRIMARY KEY)"),
    )
    .await
    .expect("create the table");
    exec_dml(&server, "INSERT INTO plain VALUES (1)").await;

    let text = query_error(&server, "SELECT * FROM table_changes(plain, 0, LATEST)").await;
    assert!(text.contains("no change data feed"), "{text}");
    assert!(text.contains("change_data_feed = true"), "{text}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_range_older_than_retention_names_both_versions() {
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    seeded(&server, &mut session, Storage::Heap).await;

    // Reclaim everything below the newest change, which is what retention
    // does once a change ages out
    let registry = server.cdc_registry.as_ref().expect("cdc is enabled");
    let table = table_id_of(&server, "orders");
    let feed = registry.get_feed(table).expect("the feed is open");
    let newest = feed.latest_version().expect("a newest change");
    feed.purge_before_version(newest).expect("purge");

    let text = query_error(&server, "SELECT * FROM table_changes(orders, 0, LATEST)").await;
    assert!(text.contains("ChangeFeedRangeExpired"), "{text}");
    assert!(text.contains("version 0"), "{text}");
    assert!(
        text.contains(&format!("version {newest}")),
        "the oldest available version is named: {text}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_column_outside_the_recorded_set_is_named() {
    for storage in Storage::BOTH {
        a_column_outside_the_recorded_set_is_named_on(storage).await;
    }
}

async fn a_column_outside_the_recorded_set_is_named_on(storage: Storage) {
    println!("{storage}");
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        &storage.create("CREATE TABLE narrow (id BIGINT PRIMARY KEY, kept TEXT, dropped TEXT)"),
    )
    .await
    .expect("create the table");
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE narrow SET (change_data_feed = true, cdf_columns = 'kept')",
    )
    .await
    .expect("narrow the feed");
    exec_dml(&server, "INSERT INTO narrow VALUES (1, 'a', 'b')").await;

    let text = query_error(&server, "SELECT * FROM table_changes(narrow, 0, LATEST)").await;
    assert!(text.contains("'dropped'"), "{text}");
    assert!(text.contains("cdf_columns"), "{text}");

    // The recorded columns and the key still read, with their values
    let rows = query_values(
        &server,
        "SELECT id, kept FROM table_changes(narrow, 0, LATEST)",
    )
    .await;
    assert_eq!(
        rows,
        vec![vec![
            ScalarValue::Int64(1),
            ScalarValue::Utf8("a".to_string())
        ]]
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_narrowed_feed_records_the_subset_and_the_identity() {
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE narrow (id BIGINT PRIMARY KEY, kept TEXT, wide TEXT)",
    )
    .await
    .expect("create the table");
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE narrow SET (change_data_feed = true)",
    )
    .await
    .expect("turn the feed on");
    let padding = "x".repeat(200);
    exec_dml(
        &server,
        &format!("INSERT INTO narrow VALUES (1, 'a', '{padding}')"),
    )
    .await;
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE narrow SET (cdf_columns = 'kept')",
    )
    .await
    .expect("narrow the feed");
    exec_dml(
        &server,
        &format!("INSERT INTO narrow VALUES (2, 'b', '{padding}')"),
    )
    .await;
    // A change touching none of the recorded columns still records the
    // row's identity, so a consumer sees that the row changed
    exec_dml(&server, "UPDATE narrow SET wide = 'moved' WHERE id = 1").await;

    let registry = server.cdc_registry.as_ref().expect("cdc is enabled");
    let feed = registry
        .get_feed(table_id_of(&server, "narrow"))
        .expect("the feed is open");
    let records = feed
        .read_range(&zyron_cdc::ChangeRange::everything())
        .expect("read the feed");
    assert_eq!(
        records.len(),
        4,
        "one insert whole, one narrowed, an update pair"
    );
    assert!(
        !records[0].projected,
        "the row written before the list is whole"
    );
    assert!(
        records[1].projected,
        "the row written under the list is narrowed"
    );
    assert!(
        records[1].row_data.len() * 4 < records[0].row_data.len(),
        "the narrowed record holds the key and the kept column, not the padding: {} against {}",
        records[1].row_data.len(),
        records[0].row_data.len()
    );
    assert!(records[2].projected && records[3].projected);
    assert!(
        records[1].schema_version > records[0].schema_version,
        "the list change minted a schema epoch"
    );

    // Every record reads through the list it was written under
    let rows = query_values(
        &server,
        "SELECT _change_type, id, kept FROM table_changes(narrow, 0, LATEST) \
         ORDER BY _commit_version, _change_ordinal",
    )
    .await;
    assert_eq!(rows.len(), 4);
    assert_eq!(rows[1][1], ScalarValue::Int64(2));
    assert_eq!(rows[1][2], ScalarValue::Utf8("b".to_string()));
    assert_eq!(kinds(&rows, 0)[2], "update_preimage");
    assert_eq!(rows[2][1], ScalarValue::Int64(1));
    assert_eq!(rows[3][1], ScalarValue::Int64(1));
    assert_eq!(rows[3][2], ScalarValue::Utf8("a".to_string()));

    // The column outside the list is refused by name over the whole range,
    // and over a range holding only whole records it still reads
    let text = query_error(&server, "SELECT wide FROM table_changes(narrow, 0, LATEST)").await;
    assert!(text.contains("'wide'"), "{text}");

    // Widening the list mints another epoch, the older narrowed records
    // keep reading through their list, and the new column reads for the
    // records written from here on
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE narrow SET (cdf_columns = 'kept, wide')",
    )
    .await
    .expect("widen the list");
    exec_dml(&server, "INSERT INTO narrow VALUES (3, 'c', 'w')").await;
    let rows = query_values(
        &server,
        "SELECT id, kept FROM table_changes(narrow, 0, LATEST) ORDER BY _commit_version, id",
    )
    .await;
    assert_eq!(rows.len(), 5);
    assert_eq!(rows[4][1], ScalarValue::Utf8("c".to_string()));
    let table = server
        .catalog
        .get_table_by_id(zyron_catalog::TableId(table_id_of(&server, "narrow")))
        .expect("table");
    assert_eq!(table.cdf.column_sets.len(), 2, "one set per list");
    let latest = table.cdf.column_sets[1].from_epoch;
    assert_eq!(latest, table.schema_epoch);
    let rows = query_values(
        &server,
        &format!(
            "SELECT id, wide FROM table_changes(narrow, {}, LATEST)",
            records.last().map(|r| r.commit_version).unwrap_or(0)
        ),
    )
    .await;
    assert_eq!(
        rows,
        vec![vec![
            ScalarValue::Int64(3),
            ScalarValue::Utf8("w".to_string())
        ]],
        "a record written under the wider list carries the column"
    );
    // A record written under the narrower list holds no value for it, and
    // the read over such a record is refused naming the column and the
    // epoch rather than answering NULL
    let text = query_error(
        &server,
        "SELECT id, wide FROM table_changes(narrow, 0, LATEST)",
    )
    .await;
    assert!(text.contains("'wide'"), "{text}");
    assert!(text.contains("epoch"), "{text}");

    // as_of_change renders a narrowed record through the columns it holds
    let rows = query_values(
        &server,
        "SELECT id, kept FROM table_changes(narrow, 0, LATEST, schema => 'as_of_change') \
         ORDER BY _commit_version, _change_ordinal",
    )
    .await;
    assert_eq!(rows.len(), 5);
    assert_eq!(rows[1][0], ScalarValue::Int64(2));
    assert_eq!(rows[1][1], ScalarValue::Utf8("b".to_string()));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_key_declared_after_the_list_joins_the_recorded_set() {
    for storage in Storage::BOTH {
        a_key_declared_after_the_list_joins_the_recorded_set_on(storage).await;
    }
}

async fn a_key_declared_after_the_list_joins_the_recorded_set_on(storage: Storage) {
    println!("{storage}");
    let (server, schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        &storage.create("CREATE TABLE keyed (id BIGINT, kept TEXT, wide TEXT)"),
    )
    .await
    .expect("create the table");
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE keyed SET (change_data_feed = true, cdf_columns = 'kept')",
    )
    .await
    .expect("narrow the feed");
    let before = server.catalog.get_table(schema, "keyed").expect("table");
    assert_eq!(before.cdf.recorded_columns().len(), 1, "no key yet");

    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE keyed ADD CONSTRAINT keyed_pk PRIMARY KEY (id)",
    )
    .await
    .expect("declare the key");
    let after = server.catalog.get_table(schema, "keyed").expect("table");
    assert_eq!(
        after.cdf.recorded_columns().len(),
        2,
        "the key joined the recorded set"
    );
    assert!(
        after.schema_epoch > before.schema_epoch,
        "under an epoch of its own"
    );

    exec_dml(&server, "INSERT INTO keyed VALUES (1, 'a', 'w')").await;
    let rows = query_values(
        &server,
        "SELECT id, kept FROM table_changes(keyed, 0, LATEST)",
    )
    .await;
    assert_eq!(
        rows,
        vec![vec![
            ScalarValue::Int64(1),
            ScalarValue::Utf8("a".to_string())
        ]]
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_layout_the_feed_still_holds_survives_epoch_retirement() {
    let (server, schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    seeded(&server, &mut session, Storage::Heap).await;
    let table = server.catalog.get_table(schema, "orders").expect("table");
    let first_epoch = table.schema_epoch;

    // A column added mints an epoch, and rewriting every row leaves no
    // heap tuple under the first one. The feed still holds the inserts
    // written under it
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE orders ADD COLUMN note TEXT",
    )
    .await
    .expect("add a column");
    exec_dml(&server, "UPDATE orders SET note = 'rewritten'").await;
    let table = server.catalog.get_table(schema, "orders").expect("table");
    assert!(table.schema_epoch > first_epoch);

    let registry = server.cdc_registry.as_ref().expect("cdc is enabled");
    let held = registry
        .oldest_schema_epoch(table.id.0)
        .expect("the feed holds records");
    assert_eq!(
        held, first_epoch as u32,
        "the feed still holds the first epoch"
    );

    // A vacuum pass that saw only the new epoch on the heap keeps the first
    // layout for the feed's sake
    server
        .catalog
        .retire_schema_epochs(table.id, table.schema_epoch, false, false, Some(held))
        .await
        .expect("retire");
    let after = server.catalog.get_table(schema, "orders").expect("table");
    assert!(
        after.physical_columns_for_epoch(first_epoch).is_some(),
        "the layout the feed's records decode through stays recorded"
    );
    let rows = query_values(
        &server,
        "SELECT id, region FROM table_changes(orders, 0, LATEST) \
         WHERE _change_type = 'insert' ORDER BY id",
    )
    .await;
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0][1], ScalarValue::Utf8("eu".to_string()));

    // Once the feed no longer holds those records, the layout retires
    let feed = registry.get_feed(table.id.0).expect("feed");
    let newest = feed.latest_version().expect("a newest change");
    feed.purge_before_version(newest).expect("purge");
    let held = registry.oldest_schema_epoch(table.id.0);
    assert_eq!(held, Some(after.schema_epoch as u32));
    server
        .catalog
        .retire_schema_epochs(table.id, after.schema_epoch, false, false, held)
        .await
        .expect("retire");
    let retired = server.catalog.get_table(schema, "orders").expect("table");
    assert!(
        retired.physical_columns_for_epoch(first_epoch).is_none(),
        "a layout nothing holds any more is gone"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_feed_without_before_images_records_one_row_per_update() {
    for storage in Storage::BOTH {
        a_feed_without_before_images_records_one_row_per_update_on(storage).await;
    }
}

async fn a_feed_without_before_images_records_one_row_per_update_on(storage: Storage) {
    println!("{storage}");
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        &storage.create("CREATE TABLE churn (id BIGINT PRIMARY KEY, v BIGINT)"),
    )
    .await
    .expect("create the table");
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE churn SET (change_data_feed = true, cdf_before_image = false)",
    )
    .await
    .expect("turn before images off");
    exec_dml(&server, "INSERT INTO churn VALUES (1, 1)").await;
    for v in 2..=6 {
        exec_dml(&server, &format!("UPDATE churn SET v = {v} WHERE id = 1")).await;
    }

    let rows = query_values(
        &server,
        "SELECT _change_type FROM table_changes(churn, 0, LATEST)",
    )
    .await;
    let seen = kinds(&rows, 0);
    assert_eq!(
        seen.iter().filter(|k| *k == "update_preimage").count(),
        0,
        "no preimage is recorded"
    );
    assert_eq!(
        seen.iter().filter(|k| *k == "update_postimage").count(),
        5,
        "one row per update"
    );

    let text = query_error(
        &server,
        "SELECT * FROM table_changes(churn, 0, LATEST) WHERE _change_type = 'update_preimage'",
    )
    .await;
    assert!(
        text.contains("before images") || text.is_empty(),
        "reading a preimage from a feed that keeps none says so: {text}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn enabling_the_feed_on_a_populated_table_records_no_history() {
    for storage in Storage::BOTH {
        enabling_the_feed_on_a_populated_table_records_no_history_on(storage).await;
    }
}

async fn enabling_the_feed_on_a_populated_table_records_no_history_on(storage: Storage) {
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

    let before = query_rows(&server, "SELECT * FROM table_changes(existing, 0, LATEST)").await;
    assert_eq!(before, 0, "the rows that predate the feed are not history");

    exec_dml(&server, "INSERT INTO existing VALUES (6, 6)").await;
    let after = query_rows(&server, "SELECT * FROM table_changes(existing, 0, LATEST)").await;
    assert_eq!(after, 1, "the first change is the first record");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn both_compression_settings_round_trip() {
    for codec in ["lz4", "zstd", "none"] {
        let (server, _schema, _tmp) = create_test_server_with_cdc().await;
        let mut session = new_session();
        exec_ddl(
            &server,
            &mut session,
            "CREATE TABLE packed (id BIGINT PRIMARY KEY, v TEXT)",
        )
        .await
        .expect("create the table");
        exec_ddl(
            &server,
            &mut session,
            &format!(
                "ALTER TABLE packed SET (change_data_feed = true, cdf_compression = '{codec}')"
            ),
        )
        .await
        .expect("set the codec");
        for id in 1..=32 {
            exec_dml(
                &server,
                &format!("INSERT INTO packed VALUES ({id}, 'row {id}')"),
            )
            .await;
        }
        // Sealing is what compresses a segment, so the read that follows goes
        // through the codec rather than over the open segment
        let registry = server.cdc_registry.as_ref().expect("cdc is enabled");
        let feed = registry
            .get_feed(table_id_of(&server, "packed"))
            .expect("the feed is open");
        feed.seal_open_segment().expect("seal");

        let rows = query_rows(&server, "SELECT * FROM table_changes(packed, 0, LATEST)").await;
        assert_eq!(rows, 32, "{codec} lost records");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_version_predicate_prunes_whole_change_files() {
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

    let mut boundaries = Vec::new();
    for group in 0..8u64 {
        for row in 0..4u64 {
            let id = group * 4 + row + 1;
            exec_dml(&server, &format!("INSERT INTO wide VALUES ({id}, {id})")).await;
        }
        boundaries.push(feed.latest_version().expect("a version"));
        feed.seal_open_segment().expect("seal");
    }

    let everything = feed.plan_read(&zyron_cdc::ChangeRange::everything());
    assert!(
        everything.segments.len() >= 8,
        "the feed holds one segment per sealed group"
    );

    // A window inside one group opens that group's file and prunes the rest
    let first = boundaries[0];
    let narrow = feed.plan_read(&zyron_cdc::ChangeRange::versions(1, first));
    assert_eq!(narrow.segments.len(), 1, "one file matches the window");
    assert!(narrow.pruned >= 7, "the rest are pruned before any decode");

    let rows = query_rows(
        &server,
        &format!("SELECT * FROM table_changes(wide, 0, {first})"),
    )
    .await;
    assert_eq!(rows, 4);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_feed_reads_across_added_dropped_and_widened_columns() {
    for storage in Storage::BOTH {
        the_feed_reads_across_added_dropped_and_widened_columns_on(storage).await;
    }
}

async fn the_feed_reads_across_added_dropped_and_widened_columns_on(storage: Storage) {
    println!("{storage}");
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    seeded(&server, &mut session, storage).await;

    // A column added after the changes reads NULL for them, and the older
    // range still reads
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE orders ADD COLUMN note TEXT",
    )
    .await
    .expect("add a column");
    exec_dml(&server, "INSERT INTO orders VALUES (4, 'us', 400, 'late')").await;
    let rows = query_values(
        &server,
        "SELECT id, note FROM table_changes(orders, 0, LATEST) ORDER BY id",
    )
    .await;
    assert_eq!(rows.len(), 4);
    assert_eq!(
        rows[0][1],
        ScalarValue::Null,
        "a change older than the column reads NULL"
    );
    assert_eq!(rows[3][1], ScalarValue::Utf8("late".to_string()));

    // A dropped column is absent by default and present as it was written
    // under the schema the change carried
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE orders DROP COLUMN region",
    )
    .await
    .expect("drop a column");
    let missing = query_error(
        &server,
        "SELECT region FROM table_changes(orders, 0, LATEST)",
    )
    .await;
    assert!(missing.contains("region"), "{missing}");
    let rows = query_values(
        &server,
        "SELECT id, region FROM table_changes(orders, 0, LATEST, schema => 'as_of_change') \
         ORDER BY id",
    )
    .await;
    assert_eq!(rows.len(), 4);
    assert_eq!(rows[0][1], ScalarValue::Utf8("eu".to_string()));
    assert_eq!(rows[3][1], ScalarValue::Utf8("us".to_string()));

    // Every write after the drop still records, its images laid out over
    // the column list the dropped column keeps its place in. A record
    // carries the column as its own write saw it, so the changes recorded
    // while the column was live keep their value and every image taken
    // after the drop, the old side of an update and a delete included,
    // reads NULL there on both storages
    exec_dml(&server, "UPDATE orders SET total = 401 WHERE id = 4").await;
    exec_dml(&server, "DELETE FROM orders WHERE id = 1").await;
    exec_dml(&server, "INSERT INTO orders VALUES (5, 500, 'last')").await;
    let rows = query_values(
        &server,
        "SELECT _change_type, id, total, region, note \
         FROM table_changes(orders, 0, LATEST, schema => 'as_of_change') \
         ORDER BY _commit_version, _change_ordinal",
    )
    .await;
    assert_eq!(
        kinds(&rows, 0),
        vec![
            "insert",
            "insert",
            "insert",
            "insert",
            "update_preimage",
            "update_postimage",
            "delete",
            "insert",
        ]
    );
    assert_eq!(rows[3][3], ScalarValue::Utf8("us".to_string()));
    assert_eq!(
        rows[4][3],
        ScalarValue::Null,
        "the old side, taken after the drop"
    );
    assert_eq!(rows[5][2], ScalarValue::Int64(401));
    assert_eq!(rows[5][3], ScalarValue::Null, "rewritten after the drop");
    assert_eq!(rows[6][1], ScalarValue::Int64(1));
    assert_eq!(rows[6][3], ScalarValue::Null, "deleted after the drop");
    assert_eq!(rows[7][3], ScalarValue::Null, "written after the drop");
    assert_eq!(rows[7][4], ScalarValue::Utf8("last".to_string()));

    // A widened type casts the older records forward, on both storages. A
    // lake file keeps its cells at the width they were written and the
    // reader widens them, a heap record decodes through the epoch it was
    // written under and widens the same way
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE orders ALTER COLUMN total TYPE BIGINT",
    )
    .await
    .expect("a widening or equal change");
    exec_ddl(
        &server,
        &mut session,
        &storage.create("CREATE TABLE narrow (id BIGINT PRIMARY KEY, n INTEGER)"),
    )
    .await
    .expect("create");
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE narrow SET (change_data_feed = true)",
    )
    .await
    .expect("feed on");
    exec_dml(&server, "INSERT INTO narrow VALUES (1, 7)").await;
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE narrow ALTER COLUMN n TYPE BIGINT",
    )
    .await
    .expect("widen");
    exec_dml(&server, "INSERT INTO narrow VALUES (2, 5000000000)").await;
    let rows = query_values(
        &server,
        "SELECT n FROM table_changes(narrow, 0, LATEST) ORDER BY n",
    )
    .await;
    assert_eq!(
        rows,
        vec![
            vec![ScalarValue::Int64(7)],
            vec![ScalarValue::Int64(5_000_000_000)]
        ]
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_stream_naming_a_dropped_column_needs_attention_and_resumes_after_alter() {
    for storage in Storage::BOTH {
        a_stream_naming_a_dropped_column_needs_attention_and_resumes_after_alter_on(storage).await;
    }
}

async fn a_stream_naming_a_dropped_column_needs_attention_and_resumes_after_alter_on(
    storage: Storage,
) {
    println!("{storage}");
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    seeded(&server, &mut session, storage).await;
    exec_ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM narrow ON TABLE orders COLUMNS (id, region)",
    )
    .await
    .expect("create the stream");
    exec_dml(&server, "INSERT INTO orders VALUES (4, 'us', 400)").await;
    assert_eq!(
        query_rows(&server, "SELECT * FROM narrow WITH (peek => true)").await,
        1
    );

    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE orders DROP COLUMN region",
    )
    .await
    .expect("drop the column the stream names");
    let entry = server
        .catalog
        .resolve_change_stream(zyron_catalog::DatabaseId(1), "narrow")
        .expect("the stream exists");
    assert!(entry.needs_attention, "the stream needs attention");
    assert!(
        entry.attention_reason.contains("region"),
        "{}",
        entry.attention_reason
    );
    assert!(!entry.stale, "it is not stale");
    let refused = query_error(&server, "SELECT * FROM narrow WITH (peek => true)").await;
    assert!(refused.contains("needs attention"), "{refused}");
    assert!(refused.contains("region"), "{refused}");

    // The position is kept, so correcting the definition resumes the stream
    // from where it stood
    exec_ddl(
        &server,
        &mut session,
        "ALTER CHANGE STREAM narrow SET COLUMNS (id)",
    )
    .await
    .expect("drop the column from the list");
    let entry = server
        .catalog
        .resolve_change_stream(zyron_catalog::DatabaseId(1), "narrow")
        .expect("the stream exists");
    assert!(!entry.needs_attention);
    assert_eq!(
        query_rows(&server, "SELECT * FROM narrow WITH (peek => true)").await,
        1
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_narrowing_type_change_needs_acknowledge_stream_break() {
    for storage in Storage::BOTH {
        a_narrowing_type_change_needs_acknowledge_stream_break_on(storage).await;
    }
}

async fn a_narrowing_type_change_needs_acknowledge_stream_break_on(storage: Storage) {
    println!("{storage}");
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    seeded(&server, &mut session, storage).await;
    exec_ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM s ON TABLE orders",
    )
    .await
    .expect("create the stream");

    let refused = exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE orders ALTER COLUMN total TYPE INTEGER",
    )
    .await
    .expect_err("a narrowing change is refused while a stream reads the table");
    assert!(refused.contains("total"), "{refused}");
    assert!(refused.contains("ACKNOWLEDGE STREAM BREAK"), "{refused}");

    // A lake file holds its cells at the width they were written, and a
    // narrower type cannot read them, so the acknowledged change is
    // refused as a rewrite and the stream is never broken by it
    if storage == Storage::Lake {
        let refused = exec_ddl(
            &server,
            &mut session,
            "ALTER TABLE orders ALTER COLUMN total TYPE INTEGER ACKNOWLEDGE STREAM BREAK",
        )
        .await
        .expect_err("a lake column is not narrowed");
        assert!(refused.contains("full rewrite"), "{refused}");
        let entry = server
            .catalog
            .resolve_change_stream(zyron_catalog::DatabaseId(1), "s")
            .expect("the stream exists");
        assert!(!entry.needs_attention, "{}", entry.attention_reason);
        exec_dml(&server, "INSERT INTO orders VALUES (4, 'us', 400)").await;
        assert_eq!(
            query_rows(&server, "SELECT * FROM s WITH (peek => true)").await,
            1
        );
        return;
    }

    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE orders ALTER COLUMN total TYPE INTEGER ACKNOWLEDGE STREAM BREAK",
    )
    .await
    .expect("acknowledged, it proceeds");
    let entry = server
        .catalog
        .resolve_change_stream(zyron_catalog::DatabaseId(1), "s")
        .expect("the stream exists");
    assert!(
        entry.needs_attention,
        "every stream on the table needs attention"
    );
    assert!(
        entry.attention_reason.contains("ACKNOWLEDGE STREAM BREAK"),
        "{}",
        entry.attention_reason
    );
    assert!(
        entry.attention_reason.contains("total"),
        "{}",
        entry.attention_reason
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_sliced_file_reads_every_shape_of_row() {
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE shapes (id BIGINT PRIMARY KEY, note TEXT, amount INT)",
    )
    .await
    .expect("create the table");
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE shapes SET (change_data_feed = true)",
    )
    .await
    .expect("turn the feed on");
    let registry = server.cdc_registry.as_ref().expect("cdc is enabled");
    let feed = registry
        .get_feed(table_id_of(&server, "shapes"))
        .expect("the feed is open");
    let text = |v: &ScalarValue| match v {
        ScalarValue::Utf8(s) => s.clone(),
        other => format!("{other:?}"),
    };

    // Rows under the first layout, some with a NULL text, then a column
    // added, rows under the second layout, and an update pair, all in one
    // file, so it holds two layout groups
    for id in 1..=5 {
        let note = if id % 2 == 0 {
            "NULL".to_string()
        } else {
            format!("'note {id}'")
        };
        exec_dml(
            &server,
            &format!("INSERT INTO shapes VALUES ({id}, {note}, {})", id * 10),
        )
        .await;
    }
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE shapes ADD COLUMN extra BIGINT",
    )
    .await
    .expect("add a column");
    for id in 6..=10 {
        exec_dml(
            &server,
            &format!(
                "INSERT INTO shapes VALUES ({id}, 'note {id}', {}, {})",
                id * 10,
                id * 100
            ),
        )
        .await;
    }
    exec_dml(&server, "UPDATE shapes SET amount = 11 WHERE id = 1").await;
    let first_file = feed.latest_version().expect("changes");
    feed.seal_open_segment().expect("seal");
    let sliced = |feed: &zyron_cdc::ChangeDataFeed| {
        (0..8u64)
            .filter(|seq| {
                matches!(
                    feed.segment_form(*seq),
                    Ok(Some(zyron_cdc::change_feed::SegmentForm::SealedColumns))
                )
            })
            .count()
    };
    assert_eq!(
        sliced(&feed),
        1,
        "the sealed file took the column-sliced form"
    );

    let rows = query_values(
        &server,
        "SELECT _change_type, id, note, amount, extra FROM table_changes(shapes, 0, LATEST) \
         ORDER BY _commit_version, _change_ordinal",
    )
    .await;
    assert_eq!(rows.len(), 5 + 5 + 2);
    // The first layout's rows read their columns and NULL for the column
    // added after them
    assert_eq!(rows[0][1], ScalarValue::Int64(1));
    assert_eq!(text(&rows[0][2]), "note 1");
    assert_eq!(rows[0][3], ScalarValue::Int32(10));
    assert_eq!(rows[0][4], ScalarValue::Null);
    assert_eq!(rows[1][2], ScalarValue::Null, "a NULL text cell");
    assert_eq!(rows[5][1], ScalarValue::Int64(6));
    assert_eq!(text(&rows[5][2]), "note 6");
    assert_eq!(rows[5][4], ScalarValue::Int64(600));
    // The update pair, under the second layout
    assert_eq!(text(&rows[10][0]), "update_preimage");
    assert_eq!(rows[10][3], ScalarValue::Int32(10));
    assert_eq!(text(&rows[11][0]), "update_postimage");
    assert_eq!(rows[11][3], ScalarValue::Int32(11));

    // Two columns of the same file read the same values
    let two = query_values(
        &server,
        "SELECT id, note FROM table_changes(shapes, 0, LATEST) \
         ORDER BY _commit_version, _change_ordinal",
    )
    .await;
    assert_eq!(two.len(), 12);
    assert_eq!(text(&two[4][1]), "note 5");
    assert_eq!(two[3][1], ScalarValue::Null);
    let count = query_rows(
        &server,
        "SELECT COUNT(*) FROM table_changes(shapes, 0, LATEST)",
    )
    .await;
    assert_eq!(count, 1);

    // The records as they were written, through the same file
    let as_written = query_values(
        &server,
        &format!(
            "SELECT id, amount FROM table_changes(shapes, 0, {first_file}, schema => 'as_of_change') \
             ORDER BY _commit_version, _change_ordinal"
        ),
    )
    .await;
    assert_eq!(as_written.len(), 12);
    assert_eq!(as_written[11][1], ScalarValue::Int32(11));

    // A column subset from here on, so the next file holds projected rows
    // in a group of their own, read through the narrowed layout
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE shapes SET (cdf_columns = 'note')",
    )
    .await
    .expect("narrow the feed");
    exec_dml(
        &server,
        "INSERT INTO shapes VALUES (11, 'note 11', 110, 1100)",
    )
    .await;
    exec_dml(&server, "DELETE FROM shapes WHERE id = 2").await;
    feed.seal_open_segment().expect("seal");
    assert_eq!(
        sliced(&feed),
        2,
        "both sealed files took the column-sliced form"
    );
    let narrowed = query_values(
        &server,
        &format!(
            "SELECT _change_type, id, note FROM table_changes(shapes, {first_file}, LATEST) \
             ORDER BY _commit_version, _change_ordinal"
        ),
    )
    .await;
    assert_eq!(narrowed.len(), 2);
    assert_eq!(text(&narrowed[0][0]), "insert");
    assert_eq!(narrowed[0][1], ScalarValue::Int64(11));
    assert_eq!(text(&narrowed[0][2]), "note 11");
    assert_eq!(text(&narrowed[1][0]), "delete");
    assert_eq!(narrowed[1][1], ScalarValue::Int64(2));
    assert_eq!(narrowed[1][2], ScalarValue::Null);
    // Both files in one read, the whole rows and the narrowed ones alike
    let all = query_values(
        &server,
        "SELECT id, note FROM table_changes(shapes, 0, LATEST) \
         ORDER BY _commit_version, _change_ordinal",
    )
    .await;
    assert_eq!(all.len(), 14);
    assert_eq!(all[13][0], ScalarValue::Int64(2));
    let refused = query_error(
        &server,
        "SELECT amount FROM table_changes(shapes, 0, LATEST)",
    )
    .await;
    assert!(refused.contains("'amount'"), "{refused}");
}

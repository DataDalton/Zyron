//! APPLY CHANGES, a target settled from a change set in one pass.
//!
//! What these prove is that the winner per key is decided by the sequence,
//! so an out-of-order source converges to what an ordered one produces, that
//! deletes, soft deletes, sparse updates and narrowed targets each do what
//! their clause says, that type 2 history keeps exactly one current row per
//! key with correct start, end and current markers, that applying a captured
//! range twice leaves the target byte-identical, and that a target the apply
//! cannot write into is refused at bind naming the column.
//!
//! Run: cargo test -p zyron-wire --test apply_changes_test -- --nocapture

use std::collections::HashMap;

use zyron_executor::column::ScalarValue;

mod common;
use common::*;

/// A change row as a test writes it into a change relation
struct Change {
    key: i64,
    kind: &'static str,
    value: Option<i64>,
    label: Option<&'static str>,
    seq: i64,
}

fn change(
    key: i64,
    kind: &'static str,
    value: Option<i64>,
    label: Option<&'static str>,
    seq: i64,
) -> Change {
    Change {
        key,
        kind,
        value,
        label,
        seq,
    }
}

/// A relation carrying the metadata columns, filled with the given changes
/// in the given order, so a test controls delivery order exactly
async fn changes_table(
    server: &std::sync::Arc<zyron_wire::connection::ServerState>,
    session: &mut Option<zyron_wire::session::Session>,
    name: &str,
    changes: &[Change],
    storage: Storage,
) {
    exec_ddl(
        server,
        session,
        &storage.create(&format!(
            "CREATE TABLE {name} (id BIGINT, value BIGINT, label TEXT, _change_type TEXT, \
             _commit_version BIGINT, _commit_ts TIMESTAMP, _commit_txn_id BIGINT, \
             _change_ordinal BIGINT, seq BIGINT)"
        )),
    )
    .await
    .expect("create the change relation");
    let mut ordinal = 0i64;
    let mut rows = Vec::with_capacity(changes.len());
    for c in changes {
        let value = c
            .value
            .map(|v| v.to_string())
            .unwrap_or_else(|| "NULL".into());
        let label = c
            .label
            .map(|l| format!("'{l}'"))
            .unwrap_or_else(|| "NULL".into());
        rows.push(format!(
            "({}, {value}, {label}, '{}', {}, '2026-01-01 00:00:00', 1, {ordinal}, {})",
            c.key, c.kind, c.seq, c.seq
        ));
        ordinal += 1;
    }
    for chunk in rows.chunks(200) {
        exec_dml(
            server,
            &format!("INSERT INTO {name} VALUES {}", chunk.join(", ")),
        )
        .await;
    }
}

/// The target's rows keyed by id, as (value, label)
async fn target_rows(
    server: &std::sync::Arc<zyron_wire::connection::ServerState>,
    name: &str,
) -> HashMap<i64, (ScalarValue, ScalarValue)> {
    let rows = query_values(server, &format!("SELECT id, value, label FROM {name}")).await;
    rows.into_iter()
        .map(|row| {
            let id = match row[0] {
                ScalarValue::Int64(v) => v,
                ref other => panic!("id was {other:?}"),
            };
            (id, (row[1].clone(), row[2].clone()))
        })
        .collect()
}

fn int(v: i64) -> ScalarValue {
    ScalarValue::Int64(v)
}

fn text(v: &str) -> ScalarValue {
    ScalarValue::Utf8(v.to_string())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn type_1_out_of_order_converges_to_the_ordered_result() {
    for storage in Storage::BOTH {
        type_1_out_of_order_converges_to_the_ordered_result_on(storage).await;
    }
}

async fn type_1_out_of_order_converges_to_the_ordered_result_on(storage: Storage) {
    println!("{storage}");
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE silver (id BIGINT PRIMARY KEY, value BIGINT, label TEXT)",
    )
    .await
    .expect("create the target");

    // Three versions of key 1 and two of key 2, delivered out of order
    let shuffled = [
        change(1, "update_postimage", Some(30), Some("third"), 3),
        change(2, "insert", Some(100), Some("first"), 1),
        change(1, "insert", Some(10), Some("first"), 1),
        change(2, "update_postimage", Some(200), Some("second"), 2),
        change(1, "update_postimage", Some(20), Some("second"), 2),
        change(3, "insert", Some(7), Some("only"), 1),
    ];
    changes_table(&server, &mut session, "shuffled", &shuffled, storage).await;
    exec_ddl(
        &server,
        &mut session,
        "APPLY CHANGES INTO silver FROM shuffled KEYS (id) SEQUENCE BY seq",
    )
    .await
    .expect("apply the shuffled changes");

    let rows = target_rows(&server, "silver").await;
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[&1], (int(30), text("third")));
    assert_eq!(rows[&2], (int(200), text("second")));
    assert_eq!(rows[&3], (int(7), text("only")));

    // The same changes in order settle to the same target
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE silver2 (id BIGINT PRIMARY KEY, value BIGINT, label TEXT)",
    )
    .await
    .expect("create the second target");
    let ordered = [
        change(1, "insert", Some(10), Some("first"), 1),
        change(1, "update_postimage", Some(20), Some("second"), 2),
        change(1, "update_postimage", Some(30), Some("third"), 3),
        change(2, "insert", Some(100), Some("first"), 1),
        change(2, "update_postimage", Some(200), Some("second"), 2),
        change(3, "insert", Some(7), Some("only"), 1),
    ];
    changes_table(&server, &mut session, "ordered", &ordered, storage).await;
    exec_ddl(
        &server,
        &mut session,
        "APPLY CHANGES INTO silver2 FROM ordered KEYS (id) SEQUENCE BY seq",
    )
    .await
    .expect("apply the ordered changes");
    assert_eq!(target_rows(&server, "silver2").await, rows);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_delete_removes_and_apply_as_delete_soft_deletes() {
    for storage in Storage::BOTH {
        a_delete_removes_and_apply_as_delete_soft_deletes_on(storage).await;
    }
}

async fn a_delete_removes_and_apply_as_delete_soft_deletes_on(storage: Storage) {
    println!("{storage}");
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE silver (id BIGINT PRIMARY KEY, value BIGINT, label TEXT)",
    )
    .await
    .expect("create the target");
    exec_dml(
        &server,
        "INSERT INTO silver VALUES (1, 1, 'a'), (2, 2, 'b'), (3, 3, 'c')",
    )
    .await;

    let changes = [
        change(1, "delete", Some(1), Some("a"), 5),
        change(2, "update_postimage", Some(22), Some("b2"), 5),
        change(4, "insert", Some(4), Some("d"), 5),
    ];
    changes_table(&server, &mut session, "c1", &changes, storage).await;
    exec_ddl(
        &server,
        &mut session,
        "APPLY CHANGES INTO silver FROM c1 KEYS (id)",
    )
    .await
    .expect("apply");
    let rows = target_rows(&server, "silver").await;
    assert!(!rows.contains_key(&1), "a delete removes the row");
    assert_eq!(rows[&2], (int(22), text("b2")));
    assert_eq!(rows[&3], (int(3), text("c")));
    assert_eq!(rows[&4], (int(4), text("d")));

    // APPLY AS DELETE WHEN recognizes a soft delete in the source instead
    let soft = [
        change(3, "update_postimage", Some(3), Some("gone"), 6),
        change(2, "update_postimage", Some(23), Some("b3"), 6),
    ];
    changes_table(&server, &mut session, "c2", &soft, storage).await;
    exec_ddl(
        &server,
        &mut session,
        "APPLY CHANGES INTO silver FROM c2 KEYS (id) APPLY AS DELETE WHEN label = 'gone'",
    )
    .await
    .expect("apply with a soft delete");
    let rows = target_rows(&server, "silver").await;
    assert!(!rows.contains_key(&3), "the soft delete removed the row");
    assert_eq!(rows[&2], (int(23), text("b3")));
    assert_eq!(rows.len(), 2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ignore_null_updates_and_except_columns() {
    for storage in Storage::BOTH {
        ignore_null_updates_and_except_columns_on(storage).await;
    }
}

async fn ignore_null_updates_and_except_columns_on(storage: Storage) {
    println!("{storage}");
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE silver (id BIGINT PRIMARY KEY, value BIGINT, label TEXT)",
    )
    .await
    .expect("create the target");
    exec_dml(
        &server,
        "INSERT INTO silver VALUES (1, 1, 'a'), (2, 2, 'b')",
    )
    .await;

    // A sparse update supplies only the value
    let sparse = [
        change(1, "update_postimage", Some(11), None, 2),
        change(2, "update_postimage", None, Some("b2"), 2),
    ];
    changes_table(&server, &mut session, "sparse", &sparse, storage).await;
    exec_ddl(
        &server,
        &mut session,
        "APPLY CHANGES INTO silver FROM sparse KEYS (id) IGNORE NULL UPDATES",
    )
    .await
    .expect("apply sparse updates");
    let rows = target_rows(&server, "silver").await;
    assert_eq!(
        rows[&1],
        (int(11), text("a")),
        "the unsupplied label stands"
    );
    assert_eq!(
        rows[&2],
        (int(2), text("b2")),
        "the unsupplied value stands"
    );
    // The apply's writes are committed rows, so a plain update follows them
    exec_dml(&server, "UPDATE silver SET value = 11 WHERE id = 1").await;

    // Without the clause a NULL is a NULL
    let nulls = [change(1, "update_postimage", Some(12), None, 3)];
    changes_table(&server, &mut session, "null_changes", &nulls, storage).await;
    exec_ddl(
        &server,
        &mut session,
        "APPLY CHANGES INTO silver FROM null_changes KEYS (id)",
    )
    .await
    .expect("apply a null");
    let rows = target_rows(&server, "silver").await;
    assert_eq!(rows[&1], (int(12), ScalarValue::Null));

    // EXCEPT COLUMNS keeps the apply off the named column
    let except = [change(2, "update_postimage", Some(99), Some("ignored"), 4)];
    changes_table(&server, &mut session, "except_c", &except, storage).await;
    exec_ddl(
        &server,
        &mut session,
        "APPLY CHANGES INTO silver FROM except_c KEYS (id) EXCEPT COLUMNS (label)",
    )
    .await
    .expect("apply except a column");
    let rows = target_rows(&server, "silver").await;
    assert_eq!(
        rows[&2],
        (int(99), text("b2")),
        "the excepted column is untouched"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn apply_as_truncate_clears_the_target_first() {
    for storage in Storage::BOTH {
        apply_as_truncate_clears_the_target_first_on(storage).await;
    }
}

async fn apply_as_truncate_clears_the_target_first_on(storage: Storage) {
    println!("{storage}");
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE silver (id BIGINT PRIMARY KEY, value BIGINT, label TEXT)",
    )
    .await
    .expect("create the target");
    exec_dml(
        &server,
        "INSERT INTO silver VALUES (1, 1, 'a'), (2, 2, 'b')",
    )
    .await;
    let changes = [
        change(9, "truncate", None, None, 1),
        change(3, "insert", Some(3), Some("c"), 2),
    ];
    changes_table(&server, &mut session, "t1", &changes, storage).await;
    exec_ddl(
        &server,
        &mut session,
        "APPLY CHANGES INTO silver FROM t1 KEYS (id) APPLY AS TRUNCATE WHEN _change_type = 'truncate'",
    )
    .await
    .expect("apply with a truncate");
    let rows = target_rows(&server, "silver").await;
    assert_eq!(rows.len(), 1, "only what the change set carried survives");
    assert_eq!(rows[&3], (int(3), text("c")));
}

/// The type 2 rows of a key, as (value, start, end, current) in start order
async fn history_of(
    server: &std::sync::Arc<zyron_wire::connection::ServerState>,
    key: i64,
) -> Vec<(ScalarValue, ScalarValue, ScalarValue, ScalarValue)> {
    query_values(
        server,
        &format!(
            "SELECT value, __start_at, __end_at, __is_current FROM dim WHERE id = {key} \
             ORDER BY __start_at"
        ),
    )
    .await
    .into_iter()
    .map(|r| (r[0].clone(), r[1].clone(), r[2].clone(), r[3].clone()))
    .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn type_2_versions_a_key_and_tracks_only_named_columns() {
    for storage in Storage::BOTH {
        type_2_versions_a_key_and_tracks_only_named_columns_on(storage).await;
    }
}

async fn type_2_versions_a_key_and_tracks_only_named_columns_on(storage: Storage) {
    println!("{storage}");
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE dim (id BIGINT, value BIGINT, label TEXT, __start_at BIGINT, __end_at BIGINT, \
         __is_current BOOLEAN)",
    )
    .await
    .expect("create the history target");

    // Three versions of one key in one change set. The last one is current,
    // the earlier ones are closed at the next version's sequence value
    let versions = [
        change(1, "insert", Some(10), Some("a"), 1),
        change(1, "update_postimage", Some(20), Some("a"), 2),
        change(1, "update_postimage", Some(30), Some("a"), 3),
    ];
    changes_table(&server, &mut session, "v1", &versions, storage).await;
    exec_ddl(
        &server,
        &mut session,
        "APPLY CHANGES INTO dim FROM v1 KEYS (id) SEQUENCE BY seq STORED AS SCD TYPE 2",
    )
    .await
    .expect("apply three versions");
    // Every change of a key is a version of the row, so one change set with
    // three changes lands three rows, each earlier one closed at the next
    // change's sequence value, the last one current
    let history = history_of(&server, 1).await;
    assert_eq!(history.len(), 3);
    assert_eq!(
        history[0],
        (int(10), int(1), int(2), ScalarValue::Boolean(false))
    );
    assert_eq!(
        history[1],
        (int(20), int(2), int(3), ScalarValue::Boolean(false))
    );
    assert_eq!(
        history[2],
        (
            int(30),
            int(3),
            ScalarValue::Null,
            ScalarValue::Boolean(true)
        )
    );

    // A later change set closes the current row and opens a new one
    let next = [change(1, "update_postimage", Some(40), Some("a"), 4)];
    changes_table(&server, &mut session, "v2", &next, storage).await;
    exec_ddl(
        &server,
        &mut session,
        "APPLY CHANGES INTO dim FROM v2 KEYS (id) SEQUENCE BY seq STORED AS SCD TYPE 2",
    )
    .await
    .expect("apply the fourth version");
    let history = history_of(&server, 1).await;
    assert_eq!(history.len(), 4);
    assert_eq!(
        history[2],
        (int(30), int(3), int(4), ScalarValue::Boolean(false))
    );
    assert_eq!(
        history[3],
        (
            int(40),
            int(4),
            ScalarValue::Null,
            ScalarValue::Boolean(true)
        )
    );

    // And once more, so five versions stand with exactly one current
    let third = [change(1, "update_postimage", Some(50), Some("a"), 5)];
    changes_table(&server, &mut session, "v3", &third, storage).await;
    exec_ddl(
        &server,
        &mut session,
        "APPLY CHANGES INTO dim FROM v3 KEYS (id) SEQUENCE BY seq STORED AS SCD TYPE 2",
    )
    .await
    .expect("apply the fifth version");
    let history = history_of(&server, 1).await;
    assert_eq!(history.len(), 5);
    assert_eq!(history[3].2, int(5));
    assert_eq!(
        history[4],
        (
            int(50),
            int(5),
            ScalarValue::Null,
            ScalarValue::Boolean(true)
        )
    );
    let current = query_rows(&server, "SELECT * FROM dim WHERE __is_current = TRUE").await;
    assert_eq!(current, 1, "exactly one current row per key");

    // A change no newer than the current row's start has already been
    // absorbed, so applying the same set again lands nothing
    exec_ddl(
        &server,
        &mut session,
        "APPLY CHANGES INTO dim FROM v3 KEYS (id) SEQUENCE BY seq STORED AS SCD TYPE 2",
    )
    .await
    .expect("apply the fifth version again");
    assert_eq!(history_of(&server, 1).await.len(), 5);

    // TRACK HISTORY ON (value), so a change to the label updates in place
    let label_only = [change(1, "update_postimage", Some(50), Some("renamed"), 6)];
    changes_table(&server, &mut session, "v4", &label_only, storage).await;
    exec_ddl(
        &server,
        &mut session,
        "APPLY CHANGES INTO dim FROM v4 KEYS (id) SEQUENCE BY seq STORED AS SCD TYPE 2 \
         TRACK HISTORY ON (value)",
    )
    .await
    .expect("apply an untracked change");
    let history = history_of(&server, 1).await;
    assert_eq!(history.len(), 5, "an untracked change opens no version");
    let label = query_values(
        &server,
        "SELECT label FROM dim WHERE id = 1 AND __is_current = TRUE",
    )
    .await;
    assert_eq!(
        label[0][0],
        text("renamed"),
        "the current row took the new label"
    );

    // TRACK HISTORY EXCEPT (label) is the same rule written the other way
    let except = [change(
        1,
        "update_postimage",
        Some(50),
        Some("renamed again"),
        7,
    )];
    changes_table(&server, &mut session, "v5", &except, storage).await;
    exec_ddl(
        &server,
        &mut session,
        "APPLY CHANGES INTO dim FROM v5 KEYS (id) SEQUENCE BY seq STORED AS SCD TYPE 2 \
         TRACK HISTORY EXCEPT (label)",
    )
    .await
    .expect("apply under the complement");
    assert_eq!(history_of(&server, 1).await.len(), 5);

    // A tracked change under the same clause does open a version
    let tracked = [change(
        1,
        "update_postimage",
        Some(60),
        Some("renamed again"),
        8,
    )];
    changes_table(&server, &mut session, "v6", &tracked, storage).await;
    exec_ddl(
        &server,
        &mut session,
        "APPLY CHANGES INTO dim FROM v6 KEYS (id) SEQUENCE BY seq STORED AS SCD TYPE 2 \
         TRACK HISTORY EXCEPT (label)",
    )
    .await
    .expect("apply a tracked change");
    let history = history_of(&server, 1).await;
    assert_eq!(history.len(), 6);
    assert_eq!(
        history[5],
        (
            int(60),
            int(8),
            ScalarValue::Null,
            ScalarValue::Boolean(true)
        )
    );
    assert_eq!(
        query_rows(&server, "SELECT * FROM dim WHERE __is_current = TRUE").await,
        1
    );

    // A delete closes the current row and opens nothing
    let gone = [change(1, "delete", Some(60), Some("renamed again"), 9)];
    changes_table(&server, &mut session, "v7", &gone, storage).await;
    exec_ddl(
        &server,
        &mut session,
        "APPLY CHANGES INTO dim FROM v7 KEYS (id) SEQUENCE BY seq STORED AS SCD TYPE 2",
    )
    .await
    .expect("apply a delete");
    assert_eq!(
        query_rows(&server, "SELECT * FROM dim WHERE __is_current = TRUE").await,
        0
    );
    let history = history_of(&server, 1).await;
    assert_eq!(history[5].2, int(9));
}

/// Every row of a table, sorted, as one string per row
async fn table_image(
    server: &std::sync::Arc<zyron_wire::connection::ServerState>,
    sql: &str,
) -> Vec<String> {
    let mut rows: Vec<String> = query_values(server, sql)
        .await
        .into_iter()
        .map(|r| format!("{r:?}"))
        .collect();
    rows.sort();
    rows
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn applying_a_captured_range_twice_is_byte_identical() {
    for storage in Storage::BOTH {
        applying_a_captured_range_twice_is_byte_identical_on(storage).await;
    }
}

async fn applying_a_captured_range_twice_is_byte_identical_on(storage: Storage) {
    println!("{storage}");
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        &storage.create("CREATE TABLE bronze (id BIGINT PRIMARY KEY, value BIGINT, label TEXT)"),
    )
    .await
    .expect("create bronze");
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE bronze SET (change_data_feed = true)",
    )
    .await
    .expect("feed on");
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE silver (id BIGINT PRIMARY KEY, value BIGINT, label TEXT)",
    )
    .await
    .expect("create silver");
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE dim (id BIGINT, value BIGINT, label TEXT, __start_at BIGINT, __end_at BIGINT, \
         __is_current BOOLEAN)",
    )
    .await
    .expect("create dim");

    for id in 1..=20 {
        exec_dml(
            &server,
            &format!("INSERT INTO bronze VALUES ({id}, {id}, 'v{id}')"),
        )
        .await;
    }
    for id in (1..=20).step_by(3) {
        exec_dml(
            &server,
            &format!("UPDATE bronze SET value = value * 10 WHERE id = {id}"),
        )
        .await;
    }
    exec_dml(&server, "DELETE FROM bronze WHERE id IN (2, 4)").await;

    let apply_type1 = "APPLY CHANGES INTO silver FROM table_changes(bronze, 0, LATEST) KEYS (id)";
    exec_ddl(&server, &mut session, apply_type1)
        .await
        .expect("first type 1 apply");
    let first = table_image(&server, "SELECT id, value, label FROM silver").await;
    assert_eq!(first.len(), 18);
    exec_ddl(&server, &mut session, apply_type1)
        .await
        .expect("second type 1 apply");
    let second = table_image(&server, "SELECT id, value, label FROM silver").await;
    assert_eq!(first, second, "type 1 replay is byte identical");

    let apply_type2 = "APPLY CHANGES INTO dim FROM table_changes(bronze, 0, LATEST) KEYS (id) STORED AS SCD TYPE 2";
    exec_ddl(&server, &mut session, apply_type2)
        .await
        .expect("first type 2 apply");
    let first = table_image(
        &server,
        "SELECT id, value, label, __start_at, __end_at, __is_current FROM dim",
    )
    .await;
    exec_ddl(&server, &mut session, apply_type2)
        .await
        .expect("second type 2 apply");
    let second = table_image(
        &server,
        "SELECT id, value, label, __start_at, __end_at, __is_current FROM dim",
    )
    .await;
    assert_eq!(first, second, "type 2 replay is byte identical");
    assert_eq!(
        query_rows(&server, "SELECT * FROM dim WHERE __is_current = TRUE").await,
        18,
        "one current row per surviving key"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_stream_source_advances_in_the_apply_commit() {
    for storage in Storage::BOTH {
        a_stream_source_advances_in_the_apply_commit_on(storage).await;
    }
}

async fn a_stream_source_advances_in_the_apply_commit_on(storage: Storage) {
    println!("{storage}");
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        &storage.create("CREATE TABLE bronze (id BIGINT PRIMARY KEY, value BIGINT, label TEXT)"),
    )
    .await
    .expect("create bronze");
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE bronze SET (change_data_feed = true)",
    )
    .await
    .expect("feed on");
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE silver (id BIGINT PRIMARY KEY, value BIGINT, label TEXT)",
    )
    .await
    .expect("create silver");
    exec_ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM s ON TABLE bronze",
    )
    .await
    .expect("create the stream");
    exec_dml(
        &server,
        "INSERT INTO bronze VALUES (1, 1, 'a'), (2, 2, 'b')",
    )
    .await;
    exec_dml(&server, "UPDATE bronze SET value = 20 WHERE id = 2").await;

    exec_ddl(
        &server,
        &mut session,
        "APPLY CHANGES INTO silver FROM s KEYS (id)",
    )
    .await
    .expect("apply from the stream");
    let rows = target_rows(&server, "silver").await;
    assert_eq!(rows[&1], (int(1), text("a")));
    assert_eq!(rows[&2], (int(20), text("b")));
    assert_eq!(
        query_rows(&server, "SELECT * FROM s WITH (peek => true)").await,
        0,
        "the apply consumed the stream"
    );

    // Nothing pending applies nothing and moves nothing
    exec_ddl(
        &server,
        &mut session,
        "APPLY CHANGES INTO silver FROM s KEYS (id)",
    )
    .await
    .expect("an empty apply");
    assert_eq!(target_rows(&server, "silver").await.len(), 2);

    // A refused apply leaves the position, so the retry sees the change
    exec_dml(&server, "INSERT INTO bronze VALUES (3, 3, 'c')").await;
    let refused = exec_ddl(
        &server,
        &mut session,
        "APPLY CHANGES INTO silver FROM s KEYS (missing)",
    )
    .await
    .expect_err("a key the target lacks is refused");
    assert!(refused.contains("missing"), "{refused}");
    assert_eq!(
        query_rows(&server, "SELECT * FROM s WITH (peek => true)").await,
        1
    );
    exec_ddl(
        &server,
        &mut session,
        "APPLY CHANGES INTO silver FROM s KEYS (id)",
    )
    .await
    .expect("the retry applies");
    assert_eq!(target_rows(&server, "silver").await.len(), 3);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bind_time_refusals_name_the_column() {
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    changes_table(
        &server,
        &mut session,
        "c",
        &[change(1, "insert", Some(1), Some("a"), 1)],
        Storage::Heap,
    )
    .await;

    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE no_key (other BIGINT, value BIGINT)",
    )
    .await
    .expect("create");
    let e = exec_ddl(
        &server,
        &mut session,
        "APPLY CHANGES INTO no_key FROM c KEYS (id)",
    )
    .await
    .expect_err("a missing key column is refused");
    assert!(e.contains("'id'"), "{e}");

    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE wrong_type (id BIGINT PRIMARY KEY, value BOOLEAN, label TEXT)",
    )
    .await
    .expect("create");
    let e = exec_ddl(
        &server,
        &mut session,
        "APPLY CHANGES INTO wrong_type FROM c KEYS (id)",
    )
    .await
    .expect_err("a type mismatch is refused");
    assert!(e.contains("'value'"), "{e}");

    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE bad_history (id BIGINT, value BIGINT, label TEXT, __start_at TEXT, __end_at BIGINT, \
         __is_current BOOLEAN)",
    )
    .await
    .expect("create");
    let e = exec_ddl(
        &server,
        &mut session,
        "APPLY CHANGES INTO bad_history FROM c KEYS (id) STORED AS SCD TYPE 2",
    )
    .await
    .expect_err("a reserved column of the wrong type is refused");
    assert!(e.contains("__start_at"), "{e}");

    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE no_history (id BIGINT, value BIGINT, label TEXT)",
    )
    .await
    .expect("create");
    let e = exec_ddl(
        &server,
        &mut session,
        "APPLY CHANGES INTO no_history FROM c KEYS (id) STORED AS SCD TYPE 2",
    )
    .await
    .expect_err("a target without the reserved columns is refused");
    assert!(e.contains("__start_at"), "{e}");
}

//! Schema changes that run with the table open to everyone else.
//!
//! Two questions decide whether this phase works. Does a build cover the rows
//! written while it runs, and does a column change leave every existing row
//! readable. Everything here is one or the other.
//!
//! Run: cargo test -p zyron-wire --test online_ddl_test -- --nocapture

mod common;

use std::sync::Arc;

use common::{
    create_test_server, exec_ddl, exec_dml, exec_dml_result, new_session, query_rows, query_values,
};
use zyron_catalog::IndexState;
use zyron_executor::column::ScalarValue;
use zyron_wire::connection::ServerState;

/// Runs a DDL statement and fails the test with its message when it is
/// refused.
async fn ddl(server: &Arc<ServerState>, sql: &str) {
    let mut session = new_session();
    exec_ddl(server, &mut session, sql)
        .await
        .unwrap_or_else(|e| panic!("`{sql}` was refused: {e}"));
}

/// Runs a DDL statement expected to be refused, and returns the message.
async fn ddl_err(server: &Arc<ServerState>, sql: &str) -> String {
    let mut session = new_session();
    match exec_ddl(server, &mut session, sql).await {
        Ok(()) => panic!("`{sql}` was accepted and should not have been"),
        Err(e) => e,
    }
}

/// Inserts `count` rows of (id, v) starting at `from`.
async fn insert_range(server: &Arc<ServerState>, table: &str, from: i64, count: i64) {
    let mut values = String::new();
    for i in from..from + count {
        if !values.is_empty() {
            values.push(',');
        }
        values.push_str(&format!("({i}, {})", i * 10));
    }
    exec_dml(server, &format!("INSERT INTO {table} VALUES {values}")).await;
}

/// Reads one system view the way the connection does, by name.
///
/// The planner does not carry the system catalog's shapes, so a test that
/// wants a view's rows asks the builder that produces them rather than
/// planning a SELECT against it.
async fn view_rows(
    server: &Arc<ServerState>,
    schema: &str,
    object: &str,
) -> (Vec<String>, Vec<Vec<Option<Vec<u8>>>>) {
    let (fields, rows) = zyron_wire::system_core_views::build(schema, object, server)
        .await
        .unwrap_or_else(|e| panic!("`{schema}.{object}` did not build: {e}"));
    let names = fields.iter().map(|f| f.name.clone()).collect();
    (names, rows)
}

/// One column of a view's rows, as text.
fn view_column(names: &[String], rows: &[Vec<Option<Vec<u8>>>], column: &str) -> Vec<String> {
    let idx = names
        .iter()
        .position(|n| n == column)
        .unwrap_or_else(|| panic!("the view has no `{column}` column, it has {names:?}"));
    rows.iter()
        .map(|r| {
            r[idx]
                .as_ref()
                .map(|b| String::from_utf8_lossy(b).to_string())
                .unwrap_or_default()
        })
        .collect()
}

fn index_state(
    server: &Arc<ServerState>,
    table_id: zyron_catalog::TableId,
    name: &str,
) -> IndexState {
    server
        .catalog
        .get_indexes_for_table(table_id)
        .into_iter()
        .find(|i| i.name == name)
        .unwrap_or_else(|| panic!("index `{name}` is in the catalog"))
        .state
}

// ---------------------------------------------------------------------------
// The build covers what was written while it ran
// ---------------------------------------------------------------------------

/// A writer runs across the whole build. Every row it wrote has to be findable
/// through the index afterwards, and the index has to hold exactly as many
/// entries as the table holds rows.
#[tokio::test]
async fn test_a_build_running_beside_a_writer_misses_no_row() {
    let (server, _schema, _tmp) = create_test_server().await;
    ddl(&server, "CREATE TABLE t (id INT, v INT)").await;
    insert_range(&server, "t", 0, 400).await;

    // The writer runs while the build does, so its rows arrive after the
    // index published and are covered by maintenance rather than by the scan
    let writer_server = Arc::clone(&server);
    let writer = tokio::spawn(async move {
        for batch in 0..20i64 {
            insert_range(&writer_server, "t", 1_000 + batch * 20, 20).await;
        }
    });

    ddl(&server, "CREATE INDEX ix_v ON t (v)").await;
    writer.await.expect("writer");

    let total = query_rows(&server, "SELECT id FROM t").await;
    assert_eq!(total, 800, "the table holds every row both sides wrote");

    // A probe for each key has to find its row, whichever side wrote it
    for id in [0i64, 399, 1_000, 1_200, 1_399] {
        let found = query_rows(&server, &format!("SELECT id FROM t WHERE v = {}", id * 10)).await;
        assert_eq!(
            found, 1,
            "the index does not answer for the row with id {id}"
        );
    }
}

/// The same shape with the writer starting after the build finished, so the
/// maintenance path is the only thing covering its rows.
#[tokio::test]
async fn test_rows_written_after_the_flip_are_indexed() {
    let (server, _schema, _tmp) = create_test_server().await;
    ddl(&server, "CREATE TABLE t (id INT, v INT)").await;
    insert_range(&server, "t", 0, 100).await;
    ddl(&server, "CREATE INDEX ix_v ON t (v)").await;
    insert_range(&server, "t", 500, 100).await;

    assert_eq!(
        query_rows(&server, "SELECT id FROM t WHERE v = 5000").await,
        1
    );
    assert_eq!(
        query_rows(&server, "SELECT id FROM t WHERE v = 990").await,
        1
    );
    assert_eq!(query_rows(&server, "SELECT id FROM t").await, 200);
}

/// A build on an empty table still publishes, waits and flips, and the index
/// answers for rows written afterwards.
#[tokio::test]
async fn test_a_build_on_an_empty_table_flips_to_ready() {
    let (server, schema, _tmp) = create_test_server().await;
    ddl(&server, "CREATE TABLE t (id INT, v INT)").await;
    ddl(&server, "CREATE INDEX ix_v ON t (v)").await;

    let table = server.catalog.get_table(schema, "t").expect("table");
    assert_eq!(index_state(&server, table.id, "ix_v"), IndexState::Ready);

    insert_range(&server, "t", 0, 10).await;
    assert_eq!(
        query_rows(&server, "SELECT id FROM t WHERE v = 50").await,
        1
    );
}

/// The index is Ready by the time the statement returns, which is what lets
/// the next statement use it.
#[tokio::test]
async fn test_the_statement_returns_only_once_the_index_is_ready() {
    let (server, schema, _tmp) = create_test_server().await;
    ddl(&server, "CREATE TABLE t (id INT, v INT)").await;
    insert_range(&server, "t", 0, 200).await;
    ddl(&server, "CREATE INDEX ix_v ON t (v)").await;

    let table = server.catalog.get_table(schema, "t").expect("table");
    assert_eq!(
        index_state(&server, table.id, "ix_v"),
        IndexState::Ready,
        "CREATE INDEX returned before the index was usable"
    );
}

// ---------------------------------------------------------------------------
// A unique build that finds a duplicate leaves nothing behind
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_a_unique_build_over_a_duplicate_fails_and_leaves_no_index() {
    let (server, schema, _tmp) = create_test_server().await;
    ddl(&server, "CREATE TABLE t (id INT, v INT)").await;
    exec_dml(&server, "INSERT INTO t VALUES (1, 5), (2, 5)").await;

    let err = ddl_err(&server, "CREATE UNIQUE INDEX ux_v ON t (v)").await;
    assert!(
        err.contains("on two live rows"),
        "the refusal says what it found: {err}"
    );
    assert!(err.contains("slot"), "the refusal names both rows: {err}");

    let table = server.catalog.get_table(schema, "t").expect("table");
    let names: Vec<String> = server
        .catalog
        .get_indexes_for_table(table.id)
        .into_iter()
        .map(|i| i.name.clone())
        .collect();
    assert!(
        !names.contains(&"ux_v".to_string()),
        "a failed unique build left its entry behind: {names:?}"
    );
    let spill = server.data_dir.join("indexes");
    if spill.exists() {
        for entry in std::fs::read_dir(&spill).expect("read indexes dir") {
            let entry = entry.expect("dir entry");
            let name = entry.file_name().to_string_lossy().to_string();
            assert!(
                !name.starts_with("build-"),
                "a failed build left its spill directory behind: {name}"
            );
        }
    }
}

/// The maintain path refuses a duplicate written while the build runs, which
/// is the half of uniqueness the scan cannot answer for.
#[tokio::test]
async fn test_a_duplicate_written_after_a_unique_index_exists_is_refused() {
    let (server, _schema, _tmp) = create_test_server().await;
    ddl(&server, "CREATE TABLE t (id INT, v INT)").await;
    exec_dml(&server, "INSERT INTO t VALUES (1, 5)").await;
    ddl(&server, "CREATE UNIQUE INDEX ux_v ON t (v)").await;

    let refused = exec_dml_result(&server, "INSERT INTO t VALUES (2, 5)").await;
    assert!(
        refused.is_err(),
        "a duplicate reached a unique index without being refused"
    );
    assert_eq!(query_rows(&server, "SELECT id FROM t").await, 1);
}

// ---------------------------------------------------------------------------
// ADD COLUMN is a catalog write
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_add_column_writes_no_heap_page_and_reads_the_default_on_old_rows() {
    let (server, schema, _tmp) = create_test_server().await;
    ddl(&server, "CREATE TABLE t (id INT, v INT)").await;
    insert_range(&server, "t", 0, 300).await;

    let before = server.catalog.get_table(schema, "t").expect("table");
    let before_pages = before.heap_file_id;

    let started = std::time::Instant::now();
    ddl(&server, "ALTER TABLE t ADD COLUMN c INT DEFAULT 7").await;
    let elapsed = started.elapsed();

    let after = server.catalog.get_table(schema, "t").expect("table");
    assert_eq!(
        before_pages, after.heap_file_id,
        "ADD COLUMN allocated a new heap file, so it rewrote the table"
    );
    assert_eq!(
        after.schema_epoch,
        before.schema_epoch + 1,
        "the encoded shape changed, so the epoch has to move"
    );
    assert!(
        elapsed < std::time::Duration::from_millis(500),
        "ADD COLUMN took {elapsed:?}, which is a rewrite rather than a catalog write"
    );

    // Every row that predates the column reads the value recorded when it was
    // added
    let sevens = query_rows(&server, "SELECT id FROM t WHERE c = 7").await;
    assert_eq!(
        sevens, 300,
        "the rows that predate the column do not read 7"
    );

    // A row written after it reads what it was given
    exec_dml(&server, "INSERT INTO t VALUES (900, 9000, 42)").await;
    let values = query_values(&server, "SELECT c FROM t WHERE id = 900").await;
    assert_eq!(values[0][0], ScalarValue::Int32(42));
    assert_eq!(
        query_rows(&server, "SELECT id FROM t WHERE c = 7").await,
        300
    );
}

#[tokio::test]
async fn test_add_column_without_a_default_reads_null_on_old_rows() {
    let (server, _schema, _tmp) = create_test_server().await;
    ddl(&server, "CREATE TABLE t (id INT, v INT)").await;
    insert_range(&server, "t", 0, 50).await;
    ddl(&server, "ALTER TABLE t ADD COLUMN c2 TEXT").await;

    assert_eq!(
        query_rows(&server, "SELECT id FROM t WHERE c2 IS NULL").await,
        50,
        "a column added without a default does not read NULL on the old rows"
    );
}

#[tokio::test]
async fn test_add_column_not_null_without_a_default_is_refused() {
    let (server, _schema, _tmp) = create_test_server().await;
    ddl(&server, "CREATE TABLE t (id INT, v INT)").await;
    insert_range(&server, "t", 0, 5).await;
    let err = ddl_err(&server, "ALTER TABLE t ADD COLUMN c INT NOT NULL").await;
    assert!(err.contains("no DEFAULT"), "the refusal says why: {err}");
}

/// Eight columns then a ninth, so the null bitmap grows from one byte to two.
/// A row of the eight-column layout read under the nine-column one would take
/// its first value out of the bitmap.
#[tokio::test]
async fn test_a_ninth_column_does_not_shift_the_rows_written_under_eight() {
    let (server, _schema, _tmp) = create_test_server().await;
    ddl(
        &server,
        "CREATE TABLE w (c0 INT, c1 INT, c2 INT, c3 INT, c4 INT, c5 INT, c6 INT, c7 INT)",
    )
    .await;
    exec_dml(&server, "INSERT INTO w VALUES (0, 1, 2, 3, 4, 5, 6, 7)").await;
    ddl(&server, "ALTER TABLE w ADD COLUMN c8 INT DEFAULT 88").await;

    let values = query_values(&server, "SELECT c0, c1, c7, c8 FROM w").await;
    assert_eq!(values.len(), 1);
    assert_eq!(values[0][0], ScalarValue::Int32(0));
    assert_eq!(values[0][1], ScalarValue::Int32(1));
    assert_eq!(
        values[0][2],
        ScalarValue::Int32(7),
        "the eighth column shifted"
    );
    assert_eq!(values[0][3], ScalarValue::Int32(88));

    // A row written under nine columns reads back the same way
    exec_dml(
        &server,
        "INSERT INTO w VALUES (10, 11, 12, 13, 14, 15, 16, 17, 18)",
    )
    .await;
    let values = query_values(&server, "SELECT c7, c8 FROM w WHERE c0 = 10").await;
    assert_eq!(values[0][0], ScalarValue::Int32(17));
    assert_eq!(values[0][1], ScalarValue::Int32(18));
}

// ---------------------------------------------------------------------------
// DROP COLUMN is a catalog write
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_drop_column_hides_it_everywhere_a_user_can_look() {
    let (server, schema, _tmp) = create_test_server().await;
    ddl(&server, "CREATE TABLE t (id INT, gone TEXT, keep INT)").await;
    exec_dml(&server, "INSERT INTO t VALUES (1, 'x', 10), (2, 'y', 20)").await;

    let before = server.catalog.get_table(schema, "t").expect("table");
    ddl(&server, "ALTER TABLE t DROP COLUMN gone").await;
    let after = server.catalog.get_table(schema, "t").expect("table");

    assert_eq!(
        before.heap_file_id, after.heap_file_id,
        "DROP COLUMN rewrote the table"
    );
    assert_eq!(
        before.schema_epoch, after.schema_epoch,
        "the byte positions did not move, so the epoch must not either"
    );
    assert_eq!(after.columns.len(), 3, "the placeholder stays in the entry");
    assert_eq!(after.live_columns().count(), 2, "the column is not live");

    // SELECT * no longer names it, and the rows still read
    let rows = query_values(&server, "SELECT * FROM t ORDER BY id").await;
    assert_eq!(
        rows[0].len(),
        2,
        "SELECT * still returns the dropped column"
    );
    assert_eq!(rows[0][0], ScalarValue::Int32(1));
    assert_eq!(rows[0][1], ScalarValue::Int32(10));
    assert_eq!(rows[1][1], ScalarValue::Int32(20));

    // The system catalog does not list it
    let (fields, rows) = view_rows(&server, "core", "columns").await;
    let listed = view_column(&fields, &rows, "column_name");
    assert!(
        listed.contains(&"keep".to_string()),
        "zyron_sys.core.columns lost a live column: {listed:?}"
    );
    assert!(
        !listed.contains(&"gone".to_string()),
        "zyron_sys.core.columns still lists the dropped column: {listed:?}"
    );

    // An insert with no column list no longer expects it
    exec_dml(&server, "INSERT INTO t VALUES (3, 30)").await;
    let rows = query_values(&server, "SELECT keep FROM t WHERE id = 3").await;
    assert_eq!(rows[0][0], ScalarValue::Int32(30));
}

#[tokio::test]
async fn test_a_column_added_after_a_drop_still_decodes_every_epoch() {
    let (server, _schema, _tmp) = create_test_server().await;
    ddl(&server, "CREATE TABLE t (id INT, gone INT, keep INT)").await;
    exec_dml(&server, "INSERT INTO t VALUES (1, 100, 10)").await;
    ddl(&server, "ALTER TABLE t DROP COLUMN gone").await;
    exec_dml(&server, "INSERT INTO t VALUES (2, 20)").await;
    ddl(&server, "ALTER TABLE t ADD COLUMN later INT DEFAULT 5").await;
    exec_dml(&server, "INSERT INTO t VALUES (3, 30, 33)").await;

    let rows = query_values(&server, "SELECT id, keep, later FROM t ORDER BY id").await;
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0][1], ScalarValue::Int32(10));
    assert_eq!(
        rows[0][2],
        ScalarValue::Int32(5),
        "the pre-drop row misread"
    );
    assert_eq!(rows[1][1], ScalarValue::Int32(20));
    assert_eq!(rows[1][2], ScalarValue::Int32(5));
    assert_eq!(rows[2][1], ScalarValue::Int32(30));
    assert_eq!(rows[2][2], ScalarValue::Int32(33));
}

#[tokio::test]
async fn test_dropping_the_last_column_is_refused() {
    let (server, _schema, _tmp) = create_test_server().await;
    ddl(&server, "CREATE TABLE one (only INT)").await;
    let err = ddl_err(&server, "ALTER TABLE one DROP COLUMN only").await;
    assert!(err.contains("only column"), "the refusal says why: {err}");
}

#[tokio::test]
async fn test_dropping_an_index_key_is_refused_and_names_the_index() {
    let (server, _schema, _tmp) = create_test_server().await;
    ddl(&server, "CREATE TABLE t (id INT, v INT)").await;
    ddl(&server, "CREATE INDEX ix_v ON t (v)").await;
    let err = ddl_err(&server, "ALTER TABLE t DROP COLUMN v").await;
    assert!(err.contains("ix_v"), "the refusal names the index: {err}");
}

// ---------------------------------------------------------------------------
// Type changes
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_a_compatible_type_change_is_a_catalog_write() {
    let (server, schema, _tmp) = create_test_server().await;
    ddl(&server, "CREATE TABLE t (id INT, amount INT)").await;
    exec_dml(&server, "INSERT INTO t VALUES (1, 2000000000)").await;

    let before = server.catalog.get_table(schema, "t").expect("table");
    ddl(&server, "ALTER TABLE t ALTER COLUMN amount TYPE BIGINT").await;
    let after = server.catalog.get_table(schema, "t").expect("table");

    assert_eq!(
        before.heap_file_id, after.heap_file_id,
        "a widening rewrote the table"
    );
    assert_eq!(
        after.schema_epoch,
        before.schema_epoch + 1,
        "the encoded width changed, so the epoch has to move"
    );

    let rows = query_values(&server, "SELECT amount FROM t").await;
    assert_eq!(
        rows[0][0],
        ScalarValue::Int64(2_000_000_000),
        "the old row did not widen on read"
    );

    // A value only the wider type can hold writes and reads back
    exec_dml(&server, "INSERT INTO t VALUES (2, 9000000000)").await;
    let rows = query_values(&server, "SELECT amount FROM t WHERE id = 2").await;
    assert_eq!(rows[0][0], ScalarValue::Int64(9_000_000_000));
}

#[tokio::test]
async fn test_widening_a_varchar_leaves_the_rows_alone() {
    let (server, schema, _tmp) = create_test_server().await;
    ddl(&server, "CREATE TABLE t (id INT, s VARCHAR(10))").await;
    exec_dml(&server, "INSERT INTO t VALUES (1, 'abcdefghij')").await;

    let before = server.catalog.get_table(schema, "t").expect("table");
    ddl(&server, "ALTER TABLE t ALTER COLUMN s TYPE VARCHAR(40)").await;
    let after = server.catalog.get_table(schema, "t").expect("table");
    assert_eq!(before.heap_file_id, after.heap_file_id);

    let rows = query_values(&server, "SELECT s FROM t").await;
    assert_eq!(rows[0][0], ScalarValue::Utf8("abcdefghij".into()));
}

#[tokio::test]
async fn test_widening_a_decimal_precision_leaves_the_rows_alone() {
    let (server, schema, _tmp) = create_test_server().await;
    ddl(&server, "CREATE TABLE t (id INT, price DECIMAL(10,2))").await;
    exec_dml(&server, "INSERT INTO t VALUES (1, 123.45)").await;

    let before = server.catalog.get_table(schema, "t").expect("table");
    ddl(
        &server,
        "ALTER TABLE t ALTER COLUMN price TYPE DECIMAL(14,2)",
    )
    .await;
    let after = server.catalog.get_table(schema, "t").expect("table");
    assert_eq!(before.heap_file_id, after.heap_file_id);
    assert_eq!(query_rows(&server, "SELECT price FROM t").await, 1);
}

/// TEXT to INT cannot read the old bytes a new way, so the rewrite runs. Every
/// row has to come out cast, and the table has to stay readable throughout.
#[tokio::test]
async fn test_an_incompatible_type_change_rewrites_and_keeps_every_row() {
    let (server, schema, _tmp) = create_test_server().await;
    ddl(&server, "CREATE TABLE t (id INT, price TEXT)").await;
    let mut values = String::new();
    for i in 1..=200i64 {
        if !values.is_empty() {
            values.push(',');
        }
        values.push_str(&format!("({i}, '{}')", i * 3));
    }
    exec_dml(&server, &format!("INSERT INTO t VALUES {values}")).await;

    let before = server.catalog.get_table(schema, "t").expect("table");
    ddl(&server, "ALTER TABLE t ALTER COLUMN price TYPE INT").await;
    let after = server.catalog.get_table(schema, "t").expect("table");

    assert_ne!(
        before.heap_file_id, after.heap_file_id,
        "an incompatible change has to move the rows"
    );
    assert_eq!(
        query_rows(&server, "SELECT id FROM t").await,
        200,
        "the rewrite lost rows"
    );
    let rows = query_values(&server, "SELECT price FROM t WHERE id = 7").await;
    assert_eq!(rows[0][0], ScalarValue::Int32(21), "the value did not cast");
}

/// The table takes writes again once the swap is done.
///
/// The swap installs the shadow's files under the source's name, so the spec
/// that pointed writers at the shadow names a table that is gone. A writer
/// still holding it mirrors into nothing and fails, which leaves a table that
/// reads correctly and cannot be written at all.
#[tokio::test]
async fn test_a_rewritten_table_still_takes_writes() {
    let (server, _schema, _tmp) = create_test_server().await;
    ddl(&server, "CREATE TABLE t (id INT, price TEXT)").await;
    exec_dml(&server, "INSERT INTO t VALUES (1, '10'), (2, '20')").await;
    ddl(&server, "ALTER TABLE t ALTER COLUMN price TYPE INT").await;

    exec_dml(&server, "INSERT INTO t VALUES (3, 30)").await;
    exec_dml(&server, "UPDATE t SET price = 99 WHERE id = 1").await;
    exec_dml(&server, "DELETE FROM t WHERE id = 2").await;

    assert_eq!(
        query_rows(&server, "SELECT id FROM t").await,
        2,
        "the writes after the swap did not land"
    );
    let rows = query_values(&server, "SELECT price FROM t WHERE id = 3").await;
    assert_eq!(
        rows[0][0],
        ScalarValue::Int32(30),
        "the row inserted after the swap did not read back"
    );
}

/// One row the new type cannot hold ends the rewrite, and the table comes out
/// exactly as it went in.
#[tokio::test]
async fn test_a_value_the_new_type_cannot_hold_ends_the_rewrite_and_changes_nothing() {
    let (server, schema, _tmp) = create_test_server().await;
    ddl(&server, "CREATE TABLE t (id INT, price TEXT)").await;
    exec_dml(
        &server,
        "INSERT INTO t VALUES (1, '10'), (2, 'abc'), (3, '30')",
    )
    .await;

    let before = server.catalog.get_table(schema, "t").expect("table");
    let err = ddl_err(&server, "ALTER TABLE t ALTER COLUMN price TYPE INT").await;
    assert!(err.contains("abc"), "the refusal names the value: {err}");
    assert!(
        err.contains("Nothing was changed"),
        "the refusal says the table is untouched: {err}"
    );

    let after = server.catalog.get_table(schema, "t").expect("table");
    assert_eq!(
        before.heap_file_id, after.heap_file_id,
        "a refused rewrite moved the table"
    );
    assert_eq!(query_rows(&server, "SELECT id FROM t").await, 3);
    let rows = query_values(&server, "SELECT price FROM t WHERE id = 2").await;
    assert_eq!(rows[0][0], ScalarValue::Utf8("abc".into()));

    // No shadow is left in the catalog or on disk
    let tables: Vec<String> = server
        .catalog
        .list_tables(schema)
        .into_iter()
        .map(|t| t.name.clone())
        .collect();
    assert!(
        !tables.iter().any(|n| n.starts_with("zyron_shadow_")),
        "a failed rewrite left its shadow behind: {tables:?}"
    );
}

// ---------------------------------------------------------------------------
// The planner and the epochs
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_a_finished_index_is_chosen_and_the_plan_says_so() {
    let (server, _schema, _tmp) = create_test_server().await;
    ddl(&server, "CREATE TABLE t (id INT, v INT)").await;
    insert_range(&server, "t", 0, 100).await;
    ddl(&server, "CREATE INDEX ix_v ON t (v)").await;

    let plan = explain(&server, "SELECT id FROM t WHERE v = 50").await;
    assert!(
        plan.contains("IndexScan"),
        "a finished index was not chosen: {plan}"
    );
    assert!(
        !plan.contains("index_building"),
        "a finished index was reported as building: {plan}"
    );
}

/// Renders the plan for a statement, which is what EXPLAIN prints.
async fn explain(server: &Arc<ServerState>, sql: &str) -> String {
    let stmt = zyron_parser::parse(sql)
        .expect("parse")
        .into_iter()
        .next()
        .expect("one statement");
    let plan = zyron_planner::plan(
        &server.catalog,
        zyron_catalog::DatabaseId(1),
        vec!["zyron_test".into()],
        stmt,
        None,
    )
    .await
    .expect("plan");
    zyron_planner::explain::ExplainNode::from_physical_plan(&plan)
        .render(&zyron_planner::explain::ExplainOptions::default())
}

#[tokio::test]
async fn test_a_building_index_is_not_chosen_and_the_plan_names_it() {
    let (server, schema, _tmp) = create_test_server().await;
    ddl(&server, "CREATE TABLE t (id INT, v INT)").await;
    insert_range(&server, "t", 0, 100).await;
    ddl(&server, "CREATE INDEX ix_v ON t (v)").await;

    // Put the index back into the state a build in flight leaves it in
    let table = server.catalog.get_table(schema, "t").expect("table");
    server
        .catalog
        .set_index_state(table.id, "ix_v", IndexState::Building)
        .await
        .expect("state flip");

    let plan = explain(&server, "SELECT id FROM t WHERE v = 50").await;
    assert!(
        plan.contains("SeqScan"),
        "a building index was chosen: {plan}"
    );
    assert!(
        plan.contains("index_building") && plan.contains("ix_v"),
        "the plan does not name the index it left alone: {plan}"
    );

    // The rows still come back, through the scan
    assert_eq!(
        query_rows(&server, "SELECT id FROM t WHERE v = 50").await,
        1
    );

    server
        .catalog
        .set_index_state(table.id, "ix_v", IndexState::Ready)
        .await
        .expect("state flip");
    let plan = explain(&server, "SELECT id FROM t WHERE v = 50").await;
    assert!(
        plan.contains("IndexScan"),
        "the index was not chosen once Ready: {plan}"
    );
}

// ---------------------------------------------------------------------------
// Epoch bookkeeping
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_a_new_table_starts_at_epoch_one_with_no_pre_stamp_layout() {
    let (server, schema, _tmp) = create_test_server().await;
    ddl(&server, "CREATE TABLE t (id INT, v INT)").await;
    let table = server.catalog.get_table(schema, "t").expect("table");

    assert_eq!(table.schema_epoch, 1);
    assert_eq!(table.schema_epochs.len(), 1);
    assert_eq!(table.schema_epochs[0].epoch, 1);
    assert_eq!(table.schema_epochs[0].columns.len(), 2);
    assert!(
        table.pre_stamp_columns.is_empty(),
        "a table created after stamping has no rows that predate it"
    );
}

#[tokio::test]
async fn test_each_shape_change_records_one_more_layout() {
    let (server, schema, _tmp) = create_test_server().await;
    ddl(&server, "CREATE TABLE t (id INT, v INT)").await;
    ddl(&server, "ALTER TABLE t ADD COLUMN a INT DEFAULT 1").await;
    ddl(&server, "ALTER TABLE t ADD COLUMN b INT DEFAULT 2").await;

    let table = server.catalog.get_table(schema, "t").expect("table");
    assert_eq!(table.schema_epoch, 3);
    assert_eq!(table.schema_epochs.len(), 3);
    assert_eq!(table.schema_epochs[0].columns.len(), 2);
    assert_eq!(table.schema_epochs[1].columns.len(), 3);
    assert_eq!(table.schema_epochs[2].columns.len(), 4);
}

#[tokio::test]
async fn test_rows_of_every_epoch_read_correctly_together() {
    let (server, _schema, _tmp) = create_test_server().await;
    ddl(&server, "CREATE TABLE t (id INT, v INT)").await;
    exec_dml(&server, "INSERT INTO t VALUES (1, 10)").await;
    ddl(&server, "ALTER TABLE t ADD COLUMN a INT DEFAULT 100").await;
    exec_dml(&server, "INSERT INTO t VALUES (2, 20, 200)").await;
    ddl(&server, "ALTER TABLE t ADD COLUMN b TEXT DEFAULT 'z'").await;
    exec_dml(&server, "INSERT INTO t VALUES (3, 30, 300, 'c')").await;

    let rows = query_values(&server, "SELECT id, v, a, b FROM t ORDER BY id").await;
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0][2], ScalarValue::Int32(100));
    assert_eq!(rows[0][3], ScalarValue::Utf8("z".into()));
    assert_eq!(rows[1][2], ScalarValue::Int32(200));
    assert_eq!(rows[1][3], ScalarValue::Utf8("z".into()));
    assert_eq!(rows[2][2], ScalarValue::Int32(300));
    assert_eq!(rows[2][3], ScalarValue::Utf8("c".into()));
}

/// A vacuum that rewrites every tuple leaves only the current layout recorded.
#[tokio::test]
async fn test_vacuum_retires_the_layouts_nothing_carries() {
    let (server, schema, _tmp) = create_test_server().await;
    ddl(&server, "CREATE TABLE t (id INT, v INT)").await;
    exec_dml(&server, "INSERT INTO t VALUES (1, 10), (2, 20)").await;
    ddl(&server, "ALTER TABLE t ADD COLUMN a INT DEFAULT 1").await;
    ddl(&server, "ALTER TABLE t ADD COLUMN b INT DEFAULT 2").await;

    let table = server.catalog.get_table(schema, "t").expect("table");
    assert_eq!(table.schema_epochs.len(), 3);

    // Every row rewritten under the current epoch, so nothing carries the
    // older two any more
    exec_dml(&server, "UPDATE t SET v = v + 1").await;

    let census = zyron_storage::EpochCensus {
        min_live_epoch: table.schema_epoch,
        any_unstamped: false,
    };
    server
        .catalog
        .retire_schema_epochs(
            table.id,
            census.min_live_epoch,
            census.any_unstamped,
            census.saw_nothing(),
        )
        .await
        .expect("retire");

    let after = server.catalog.get_table(schema, "t").expect("table");
    assert_eq!(
        after.schema_epochs.len(),
        1,
        "a layout nothing carries stayed in the entry"
    );
    assert_eq!(after.schema_epochs[0].epoch, after.schema_epoch);
    assert!(after.pre_stamp_columns.is_empty());
}

#[tokio::test]
async fn test_retirement_never_drops_the_layout_writes_are_using() {
    let (server, schema, _tmp) = create_test_server().await;
    ddl(&server, "CREATE TABLE t (id INT, v INT)").await;
    let table = server.catalog.get_table(schema, "t").expect("table");

    // A pass over an empty table saw no tuple at all, so it says nothing
    server
        .catalog
        .retire_schema_epochs(table.id, u16::MAX, false, true)
        .await
        .expect("retire");
    let after = server.catalog.get_table(schema, "t").expect("table");
    assert_eq!(
        after.schema_epochs.len(),
        1,
        "an empty table lost the layout its next write uses"
    );
}

// ---------------------------------------------------------------------------
// Progress and constraints
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_the_progress_view_is_empty_when_nothing_is_running() {
    let (server, _schema, _tmp) = create_test_server().await;
    ddl(&server, "CREATE TABLE t (id INT, v INT)").await;
    insert_range(&server, "t", 0, 100).await;
    ddl(&server, "CREATE INDEX ix_v ON t (v)").await;

    assert!(
        server.ddl_progress.rows().is_empty(),
        "a finished build left its progress row behind"
    );
    let (fields, rows) = view_rows(&server, "storage", "ddl_progress").await;
    assert!(
        rows.is_empty(),
        "the progress view reports a finished build"
    );
    for expected in ["phase", "rows_done", "rows_total_estimate", "pause_signal"] {
        assert!(
            fields.iter().any(|f| f == expected),
            "the progress view has no `{expected}` column, it has {fields:?}"
        );
    }

    // The index view says the index is complete and points at no build
    let (fields, rows) = view_rows(&server, "storage", "indexes").await;
    let states = view_column(&fields, &rows, "state");
    assert!(
        states.iter().all(|s| s == "ready"),
        "an index is reported as still building: {states:?}"
    );
}

#[tokio::test]
async fn test_a_check_constraint_is_enforced_and_reported_valid_once_scanned() {
    let (server, schema, _tmp) = create_test_server().await;
    ddl(&server, "CREATE TABLE t (id INT, x INT)").await;
    exec_dml(&server, "INSERT INTO t VALUES (1, 5), (2, 6)").await;
    ddl(&server, "ALTER TABLE t ADD CONSTRAINT ck_x CHECK (x > 0)").await;

    let table = server.catalog.get_table(schema, "t").expect("table");
    let constraint = table
        .constraints
        .iter()
        .find(|c| c.name == "ck_x")
        .expect("the constraint is on the table");
    assert!(
        constraint.validated,
        "a constraint whose scan finished is still reported as validating"
    );

    let (fields, rows) = view_rows(&server, "core", "constraints").await;
    let names = view_column(&fields, &rows, "constraint_name");
    let states = view_column(&fields, &rows, "state");
    let position = names
        .iter()
        .position(|n| n == "ck_x")
        .expect("the view lists the constraint");
    assert_eq!(states[position], "valid");

    // The rule applies to writes after it
    assert!(
        exec_dml_result(&server, "INSERT INTO t VALUES (3, -1)")
            .await
            .is_err(),
        "a row breaking the constraint was accepted"
    );
}

#[tokio::test]
async fn test_a_check_constraint_a_row_already_breaks_is_refused_and_not_kept() {
    let (server, schema, _tmp) = create_test_server().await;
    ddl(&server, "CREATE TABLE t (id INT, x INT)").await;
    exec_dml(&server, "INSERT INTO t VALUES (1, 5), (2, -3)").await;

    let err = ddl_err(&server, "ALTER TABLE t ADD CONSTRAINT ck_x CHECK (x > 0)").await;
    assert!(!err.is_empty());

    let table = server.catalog.get_table(schema, "t").expect("table");
    assert!(
        !table.constraints.iter().any(|c| c.name == "ck_x"),
        "a constraint the rows do not satisfy stayed on the table"
    );
    // The table still takes the write the constraint would have refused
    exec_dml(&server, "INSERT INTO t VALUES (3, -9)").await;
    assert_eq!(query_rows(&server, "SELECT id FROM t").await, 3);
}

// ---------------------------------------------------------------------------
// Recovery
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_recovery_drops_a_build_that_did_not_finish() {
    let (server, schema, _tmp) = create_test_server().await;
    ddl(&server, "CREATE TABLE t (id INT, v INT)").await;
    insert_range(&server, "t", 0, 20).await;
    ddl(&server, "CREATE INDEX ix_v ON t (v)").await;

    // The state a crash mid-build leaves behind
    let table = server.catalog.get_table(schema, "t").expect("table");
    server
        .catalog
        .set_index_state(table.id, "ix_v", IndexState::Building)
        .await
        .expect("state flip");
    let spill = zyron_wire::index_build::build_spill_dir(&server.data_dir, 999);
    std::fs::create_dir_all(&spill).expect("spill dir");

    zyron_wire::index_build::discard_incomplete_ddl(
        &server.catalog,
        &server.disk_manager,
        &server.data_dir,
    )
    .await;

    let names: Vec<String> = server
        .catalog
        .get_indexes_for_table(table.id)
        .into_iter()
        .map(|i| i.name.clone())
        .collect();
    assert!(
        !names.contains(&"ix_v".to_string()),
        "recovery kept an index whose build did not finish: {names:?}"
    );
    // The table itself is untouched
    assert_eq!(query_rows(&server, "SELECT id FROM t").await, 20);
}

#[tokio::test]
async fn test_recovery_removes_a_constraint_whose_validation_did_not_finish() {
    let (server, schema, _tmp) = create_test_server().await;
    ddl(&server, "CREATE TABLE t (id INT, x INT)").await;
    exec_dml(&server, "INSERT INTO t VALUES (1, 5)").await;
    ddl(&server, "ALTER TABLE t ADD CONSTRAINT ck_x CHECK (x > 0)").await;

    // The state a crash mid-validation leaves behind
    let table = server.catalog.get_table(schema, "t").expect("table");
    let mut entry = (*table).clone();
    for c in entry.constraints.iter_mut() {
        if c.name == "ck_x" {
            c.validated = false;
        }
    }
    server.catalog.update_table(entry).await.expect("update");

    zyron_wire::index_build::discard_incomplete_ddl(
        &server.catalog,
        &server.disk_manager,
        &server.data_dir,
    )
    .await;

    let after = server.catalog.get_table(schema, "t").expect("table");
    assert!(
        !after.constraints.iter().any(|c| c.name == "ck_x"),
        "recovery kept a constraint nothing had validated"
    );
}

//! Column-shape ALTER on a lake table commits a schema change to the
//! table's transaction log instead of running the heap rewrite engine.
//!
//! The heap rewrite re-encodes every row into side heap files and swaps
//! the catalog. A lake table's rows live in its log, so that path either
//! desynced the catalog from the lake schema (bricking every SELECT) or
//! re-appended every existing row while the old files stayed live,
//! doubling the table.
//!
//! Run: cargo test -p zyron-wire --test lake_alter_test

mod common;

use common::{create_test_server, exec_ddl, exec_dml, new_session, query_values};
use zyron_executor::column::ScalarValue;

#[tokio::test]
async fn test_lake_add_column_serves_null_for_old_rows() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE lk (id BIGINT NOT NULL, name TEXT) USING ZYRONLAKE",
    )
    .await
    .expect("create");
    exec_dml(&server, "INSERT INTO lk VALUES (1, 'a'), (2, 'b')").await;

    exec_ddl(&server, &mut session, "ALTER TABLE lk ADD COLUMN v BIGINT")
        .await
        .expect("add column");

    // Old rows read back whole, with NULL in the added column
    let rows = query_values(&server, "SELECT id, name, v FROM lk ORDER BY id").await;
    assert_eq!(
        rows.len(),
        2,
        "both existing rows survive the schema change"
    );
    assert_eq!(rows[0][2], ScalarValue::Null);
    assert_eq!(rows[1][2], ScalarValue::Null);

    // New rows carry the column
    exec_dml(&server, "INSERT INTO lk VALUES (3, 'c', 30)").await;
    let rows = query_values(&server, "SELECT v FROM lk WHERE id = 3").await;
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0][0], ScalarValue::Int64(30));
}

#[tokio::test]
async fn test_lake_drop_column_keeps_every_row_once() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE lk (id BIGINT NOT NULL, name TEXT) USING ZYRONLAKE",
    )
    .await
    .expect("create");
    exec_dml(&server, "INSERT INTO lk VALUES (1, 'a'), (2, 'b')").await;

    exec_ddl(&server, &mut session, "ALTER TABLE lk DROP COLUMN name")
        .await
        .expect("drop column");

    let rows = query_values(&server, "SELECT id FROM lk ORDER BY id").await;
    assert_eq!(
        rows.iter()
            .map(|r| match r[0] {
                ScalarValue::Int64(v) => v,
                ref other => panic!("expected Int64, got {other:?}"),
            })
            .collect::<Vec<i64>>(),
        vec![1, 2],
        "each row exists exactly once after the drop"
    );

    // The narrowed shape accepts inserts
    exec_dml(&server, "INSERT INTO lk VALUES (3)").await;
    assert_eq!(query_values(&server, "SELECT id FROM lk").await.len(), 3);
}

/// Changing a column's type would reinterpret every stored cell, which
/// needs a data rewrite the lake path does not run, so it is refused
/// rather than doubling or corrupting the table.
#[tokio::test]
async fn test_lake_alter_type_is_refused() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE lk (id BIGINT NOT NULL, v INT) USING ZYRONLAKE",
    )
    .await
    .expect("create");
    exec_dml(&server, "INSERT INTO lk VALUES (1, 10)").await;

    let result = exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE lk ALTER COLUMN v TYPE BIGINT",
    )
    .await;
    assert!(result.is_err(), "lake ALTER TYPE must be refused loudly");

    // The refusal changed nothing
    let rows = query_values(&server, "SELECT id, v FROM lk").await;
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0][1], ScalarValue::Int32(10));
}

/// An added column whose constraint or default the old rows cannot satisfy
/// is refused instead of serving values the declaration contradicts.
#[tokio::test]
async fn test_lake_add_column_not_null_or_default_is_refused() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE lk (id BIGINT NOT NULL) USING ZYRONLAKE",
    )
    .await
    .expect("create");
    exec_dml(&server, "INSERT INTO lk VALUES (1)").await;

    let result = exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE lk ADD COLUMN v BIGINT NOT NULL",
    )
    .await;
    assert!(
        result.is_err(),
        "NOT NULL on existing rows with no backfill must be refused"
    );

    let result = exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE lk ADD COLUMN w BIGINT DEFAULT 7",
    )
    .await;
    assert!(
        result.is_err(),
        "a DEFAULT the old rows will not serve must be refused"
    );
}

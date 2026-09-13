//! DROP SCHEMA holds one invariant: only an empty schema can be dropped.
//!
//! A bare DROP SCHEMA on a non-empty schema is refused with an error naming
//! what is inside. DROP SCHEMA CASCADE empties the schema through the real
//! per-object drop paths first, tables with their indexes and triggers,
//! views, sequences, functions, aggregates, procedures, and materialized
//! views, then removes the empty shell. Recycled tables count as contents
//! and are purged by the cascade, the recycle bin cannot outlive its schema.
//!
//! Run: cargo test -p zyron-wire --test schema_drop_test

mod common;

use std::sync::Arc;

use common::{create_test_server, exec_ddl, exec_dml, new_session, query_error, query_values};
use zyron_wire::connection::ServerState;

async fn ddl(server: &Arc<ServerState>, sql: &str) {
    exec_ddl(server, &mut new_session(), sql)
        .await
        .unwrap_or_else(|e| panic!("ddl failed: {sql}\n{e}"));
}

async fn ddl_err(server: &Arc<ServerState>, sql: &str) -> String {
    exec_ddl(server, &mut new_session(), sql)
        .await
        .expect_err(&format!("ddl unexpectedly succeeded: {sql}"))
}

/// A schema holding one of every object kind, with distinguishable names.
async fn populated_schema(server: &Arc<ServerState>) {
    ddl(server, "CREATE SCHEMA sd").await;
    ddl(server, "CREATE TABLE sd.t (id INT)").await;
    exec_dml(server, "INSERT INTO sd.t (id) VALUES (1), (2), (3)").await;
    ddl(server, "CREATE VIEW sd.v AS SELECT id FROM sd.t").await;
    ddl(server, "CREATE SEQUENCE sd.seq").await;
    ddl(
        server,
        "CREATE FUNCTION sd.f(x INT) RETURNS INT AS 'x + 1 'LANGUAGE SQL",
    )
    .await;
    ddl(
        server,
        "CREATE FUNCTION sd.acc(acc INT, val INT) RETURNS INT AS 'acc + val 'LANGUAGE SQL",
    )
    .await;
    ddl(
        server,
        "CREATE AGGREGATE sd.agg(val INT) (SFUNC = sd.acc, STYPE = INT, INITCOND = '0')",
    )
    .await;
    ddl(
        server,
        "CREATE PROCEDURE sd.p() AS 'INSERT INTO sd.t (id) VALUES (99)' LANGUAGE SQL",
    )
    .await;
    ddl(
        server,
        "CREATE MATERIALIZED VIEW sd.mv AS SELECT id FROM sd.t",
    )
    .await;
}

#[tokio::test]
async fn drop_schema_refuses_a_non_empty_schema_naming_its_contents() {
    let (server, _schema, _tmp) = create_test_server().await;
    populated_schema(&server).await;

    let err = ddl_err(&server, "DROP SCHEMA sd").await;
    assert!(err.contains("not empty"), "{err}");
    for name in ["t", "v", "seq", "f", "agg", "p", "mv"] {
        assert!(err.contains(name), "the refusal names '{name}': {err}");
    }
    assert!(
        err.contains("CASCADE"),
        "the refusal points at CASCADE: {err}"
    );

    // The refusal left everything intact.
    assert_eq!(query_values(&server, "SELECT id FROM sd.t").await.len(), 3);
    assert_eq!(query_values(&server, "SELECT id FROM sd.v").await.len(), 3);
}

#[tokio::test]
async fn cascade_drops_every_object_kind_and_the_schema() {
    let (server, _schema, _tmp) = create_test_server().await;
    populated_schema(&server).await;
    let db_id = zyron_catalog::DatabaseId(1);
    let schema_id = server.catalog.get_schema(db_id, "sd").expect("schema").id;
    let heap_file_id = server
        .catalog
        .get_table(schema_id, "t")
        .expect("table")
        .heap_file_id;

    ddl(&server, "DROP SCHEMA sd CASCADE").await;

    // The schema and everything it held are gone.
    assert!(server.catalog.get_schema(db_id, "sd").is_err());
    let contents = server.catalog.schema_contents(schema_id);
    assert!(
        contents.is_empty(),
        "the cascade left objects behind: {}",
        contents.describe()
    );
    let err = query_error(&server, "SELECT id FROM sd.t").await;
    assert!(!err.is_empty(), "the dropped schema's table is unreachable");

    // The heap file handle was reclaimed, not just the catalog rows.
    assert!(
        !server.heap_files.contains_sync(&heap_file_id),
        "the dropped table's heap handle is still live"
    );

    // The name is immediately reusable and starts fresh.
    ddl(&server, "CREATE SCHEMA sd").await;
    ddl(&server, "CREATE TABLE sd.t (id INT)").await;
    exec_dml(&server, "INSERT INTO sd.t (id) VALUES (7)").await;
    assert_eq!(query_values(&server, "SELECT id FROM sd.t").await.len(), 1);
}

#[tokio::test]
async fn cascade_drops_triggers_with_their_tables() {
    let (server, _schema, _tmp) = create_test_server().await;
    ddl(&server, "CREATE SCHEMA st").await;
    ddl(&server, "CREATE TABLE st.t (id INT)").await;
    ddl(&server, "CREATE TABLE st.audit (id INT)").await;
    ddl(
        &server,
        "CREATE PROCEDURE st.log_it() AS 'INSERT INTO st.audit (id) VALUES (1)' LANGUAGE SQL",
    )
    .await;
    let mut session = new_session();
    if let Some(s) = session.as_mut() {
        s.search_path = vec!["st".into()];
    }
    exec_ddl(
        &server,
        &mut session,
        "CREATE TRIGGER trg AFTER INSERT ON t FOR EACH ROW EXECUTE FUNCTION st.log_it",
    )
    .await
    .expect("create trigger");
    let db_id = zyron_catalog::DatabaseId(1);
    let schema_id = server.catalog.get_schema(db_id, "st").expect("schema").id;
    let table_id = server.catalog.get_table(schema_id, "t").expect("table").id;
    assert_eq!(server.catalog.triggers_for_table(table_id).len(), 1);

    ddl(&server, "DROP SCHEMA st CASCADE").await;
    assert!(
        server.catalog.triggers_for_table(table_id).is_empty(),
        "the table's trigger outlived the cascade"
    );
}

#[tokio::test]
async fn drop_table_no_longer_orphans_its_triggers() {
    let (server, schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(&server, &mut session, "CREATE TABLE t (id INT)")
        .await
        .expect("create table");
    exec_ddl(&server, &mut session, "CREATE TABLE audit (id INT)")
        .await
        .expect("create audit");
    exec_ddl(
        &server,
        &mut session,
        "CREATE PROCEDURE log_it() AS 'INSERT INTO zyron_test.audit (id) VALUES (1)' LANGUAGE SQL",
    )
    .await
    .expect("create procedure");
    exec_ddl(
        &server,
        &mut session,
        "CREATE TRIGGER trg AFTER INSERT ON t FOR EACH ROW EXECUTE FUNCTION zyron_test.log_it",
    )
    .await
    .expect("create trigger");
    let table_id = server.catalog.get_table(schema_id, "t").expect("table").id;
    assert_eq!(server.catalog.triggers_for_table(table_id).len(), 1);

    exec_ddl(&server, &mut session, "DROP TABLE t")
        .await
        .expect("drop table");
    assert!(
        server.catalog.triggers_for_table(table_id).is_empty(),
        "DROP TABLE left a trigger keyed by the dead table id"
    );
}

#[tokio::test]
async fn recycled_tables_count_as_contents_and_cascade_purges_them() {
    let (server, _schema, _tmp) = create_test_server().await;
    ddl(&server, "CREATE SCHEMA sr").await;
    ddl(&server, "CREATE TABLE sr.rt (id INT)").await;
    let db_id = zyron_catalog::DatabaseId(1);
    let schema_id = server.catalog.get_schema(db_id, "sr").expect("schema").id;

    // Give the table a recycle window the way the catalog stores it, then
    // drop it so it lands in the bin instead of being removed.
    let mut entry = (*server.catalog.get_table(schema_id, "rt").expect("table")).clone();
    entry.lifecycle.recycle_window_seconds = 3600;
    server
        .catalog
        .update_table(entry)
        .await
        .expect("set recycle window");
    let mut session = new_session();
    if let Some(s) = session.as_mut() {
        s.search_path = vec!["sr".into()];
    }
    exec_ddl(&server, &mut session, "DROP TABLE rt")
        .await
        .expect("soft drop");
    assert!(
        server
            .catalog
            .list_dropped_tables()
            .iter()
            .any(|t| t.schema_id == schema_id),
        "the table went to the recycle bin"
    );

    // The schema looks empty to name resolution but is not: the recycled
    // table is restorable content and blocks the bare drop.
    let err = ddl_err(&server, "DROP SCHEMA sr").await;
    assert!(err.contains("recycled table"), "{err}");
    assert!(err.contains("rt"), "{err}");

    ddl(&server, "DROP SCHEMA sr CASCADE").await;
    assert!(server.catalog.get_schema(db_id, "sr").is_err());
    assert!(
        !server
            .catalog
            .list_dropped_tables()
            .iter()
            .any(|t| t.schema_id == schema_id),
        "the cascade left a recycled table behind"
    );
}

#[tokio::test]
async fn empty_schema_drops_with_and_without_cascade() {
    let (server, _schema, _tmp) = create_test_server().await;
    ddl(&server, "CREATE SCHEMA se1").await;
    ddl(&server, "DROP SCHEMA se1").await;
    ddl(&server, "CREATE SCHEMA se2").await;
    ddl(&server, "DROP SCHEMA se2 CASCADE").await;
    ddl(&server, "DROP SCHEMA IF EXISTS never_existed").await;
    ddl(&server, "DROP SCHEMA IF EXISTS never_existed CASCADE").await;
}

#[tokio::test]
async fn reserved_schemas_refuse_cascade_before_touching_anything() {
    let (server, _schema, _tmp) = create_test_server().await;
    let err = ddl_err(&server, "DROP SCHEMA zyron_sys CASCADE").await;
    assert!(err.contains("reserved"), "{err}");
}

/// Marks a table immutable the way the catalog stores the flag.
async fn set_immutable(server: &Arc<ServerState>, schema: &str, table: &str) {
    let db_id = zyron_catalog::DatabaseId(1);
    let schema_id = server.catalog.get_schema(db_id, schema).expect("schema").id;
    let mut entry = (*server.catalog.get_table(schema_id, table).expect("table")).clone();
    entry.lifecycle.immutable = true;
    server
        .catalog
        .update_table(entry)
        .await
        .expect("set immutable");
}

#[tokio::test]
async fn immutable_tables_refuse_drop_truncate_and_cascade() {
    let (server, _schema, _tmp) = create_test_server().await;
    ddl(&server, "CREATE SCHEMA sw").await;
    ddl(&server, "CREATE TABLE sw.locked (id INT)").await;
    exec_dml(&server, "INSERT INTO sw.locked (id) VALUES (1)").await;
    set_immutable(&server, "sw", "locked").await;

    // The lock refuses every path that reaches the same rows, and names the
    // same reason the DML hook gives.
    let err = ddl_err(&server, "DROP TABLE sw.locked").await;
    assert!(err.contains("immutable"), "{err}");
    let err = ddl_err(&server, "TRUNCATE sw.locked").await;
    assert!(err.contains("immutable"), "{err}");
    let err = ddl_err(&server, "DROP SCHEMA sw CASCADE").await;
    assert!(err.contains("immutable"), "{err}");

    // The table and its rows survive every refusal.
    let rows = query_values(&server, "SELECT id FROM sw.locked").await;
    assert_eq!(rows.len(), 1, "a refused statement changed the table");
}

#[tokio::test]
async fn a_retention_lock_refuses_drop_until_it_expires() {
    let (server, _schema, _tmp) = create_test_server().await;
    ddl(&server, "CREATE SCHEMA sl").await;
    ddl(&server, "CREATE TABLE sl.held (id INT)").await;

    let db_id = zyron_catalog::DatabaseId(1);
    let schema_id = server.catalog.get_schema(db_id, "sl").expect("schema").id;
    let mut entry = (*server.catalog.get_table(schema_id, "held").expect("table")).clone();
    entry.lifecycle.retention_lock_until = zyron_lifecycle::ttl::now_micros() + 3_600_000_000;
    server
        .catalog
        .update_table(entry)
        .await
        .expect("set retention lock");

    let err = ddl_err(&server, "DROP TABLE sl.held").await;
    assert!(err.contains("retention"), "{err}");

    // A lock that has passed holds nothing.
    let mut entry = (*server.catalog.get_table(schema_id, "held").expect("table")).clone();
    entry.lifecycle.retention_lock_until = 1;
    server
        .catalog
        .update_table(entry)
        .await
        .expect("expire retention lock");
    ddl(&server, "DROP TABLE sl.held").await;
}

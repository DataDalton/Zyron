//! A qualified name reaches the object in the schema it names.
//!
//! Parsing `schema.object` is half the work. The other half is the handler
//! resolving the schema from the name rather than from the session's search
//! path, because a name that parses but is not resolved looks up an object
//! literally called `schema.object` and fails, or worse finds the wrong one.
//!
//! Every test here holds a table of the same name in two schemas, so a handler
//! that ignored the schema part would act on the session's one and be caught.

mod common;

use common::{create_test_server, exec_ddl, new_session};

/// Creates a second schema beside the session's own, each holding a table of
/// the same name, and returns the other schema's id.
async fn two_schemas_one_table_name(
    server: &std::sync::Arc<zyron_wire::connection::ServerState>,
    session: &mut Option<zyron_wire::session::Session>,
    table: &str,
) -> zyron_catalog::SchemaId {
    let other = server
        .catalog
        .create_schema(
            zyron_catalog::SYSTEM_DATABASE_ID,
            "other_schema",
            "test_user",
        )
        .await
        .expect("the second schema is created");
    exec_ddl(server, session, &format!("CREATE TABLE {table} (a INT)"))
        .await
        .expect("the session's own table is created");
    exec_ddl(
        server,
        session,
        &format!("CREATE TABLE other_schema.{table} (a INT)"),
    )
    .await
    .expect("the other schema's table is created");
    other
}

#[tokio::test]
async fn drop_table_drops_the_one_the_schema_names() {
    let (server, session_schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    let other = two_schemas_one_table_name(&server, &mut session, "t").await;

    exec_ddl(&server, &mut session, "DROP TABLE other_schema.t")
        .await
        .expect("a qualified drop is accepted");

    assert!(
        server.catalog.get_table(other, "t").is_err(),
        "the named schema's table is still there, so the schema part was ignored"
    );
    assert!(
        server.catalog.get_table(session_schema, "t").is_ok(),
        "the session's table was dropped instead of the named one"
    );
}

#[tokio::test]
async fn truncate_keeps_the_other_schema_s_table() {
    let (server, session_schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    let other = two_schemas_one_table_name(&server, &mut session, "t").await;

    exec_ddl(&server, &mut session, "TRUNCATE TABLE other_schema.t")
        .await
        .expect("a qualified truncate is accepted");

    // Both tables still exist. The point is that the statement resolved at all
    assert!(server.catalog.get_table(other, "t").is_ok());
    assert!(server.catalog.get_table(session_schema, "t").is_ok());
}

#[tokio::test]
async fn alter_table_adds_the_column_to_the_named_schema() {
    let (server, session_schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    let other = two_schemas_one_table_name(&server, &mut session, "t").await;

    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE other_schema.t ADD COLUMN b INT",
    )
    .await
    .expect("a qualified alter is accepted");

    let altered = server.catalog.get_table(other, "t").expect("named table");
    let untouched = server
        .catalog
        .get_table(session_schema, "t")
        .expect("session table");
    assert_eq!(
        altered.columns.len(),
        2,
        "the named schema's table did not gain the column"
    );
    assert_eq!(
        untouched.columns.len(),
        1,
        "the session's table gained the column, so the schema part was ignored"
    );
}

#[tokio::test]
async fn create_index_lands_in_the_table_s_own_schema() {
    let (server, session_schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    let other = two_schemas_one_table_name(&server, &mut session, "t").await;

    exec_ddl(
        &server,
        &mut session,
        "CREATE INDEX i ON other_schema.t (a)",
    )
    .await
    .expect("a qualified index target is accepted");

    let named = server.catalog.get_table(other, "t").expect("named table");
    let session_table = server
        .catalog
        .get_table(session_schema, "t")
        .expect("session table");
    assert_eq!(
        server.catalog.get_indexes_for_table(named.id).len(),
        1,
        "the index was not created on the named schema's table"
    );
    assert!(
        server
            .catalog
            .get_indexes_for_table(session_table.id)
            .is_empty(),
        "the index landed on the session's table instead"
    );
}

#[tokio::test]
async fn drop_index_finds_the_index_in_the_named_schema() {
    let (server, session_schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    let other = two_schemas_one_table_name(&server, &mut session, "t").await;
    // One index of the same name in each schema, so only the schema part can
    // tell them apart
    exec_ddl(&server, &mut session, "CREATE INDEX i ON t (a)")
        .await
        .expect("session index");
    exec_ddl(
        &server,
        &mut session,
        "CREATE INDEX i ON other_schema.t (a)",
    )
    .await
    .expect("other index");

    exec_ddl(&server, &mut session, "DROP INDEX other_schema.i")
        .await
        .expect("a qualified index name is accepted");

    let named = server.catalog.get_table(other, "t").expect("named table");
    let session_table = server
        .catalog
        .get_table(session_schema, "t")
        .expect("session table");
    assert!(
        server.catalog.get_indexes_for_table(named.id).is_empty(),
        "the named schema's index survived"
    );
    assert_eq!(
        server.catalog.get_indexes_for_table(session_table.id).len(),
        1,
        "the session's index was dropped instead"
    );
}

#[tokio::test]
async fn a_bare_name_still_reaches_the_session_s_schema() {
    // The fallback has to keep working, because every statement written without
    // a schema depends on it
    let (server, session_schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    let other = two_schemas_one_table_name(&server, &mut session, "t").await;

    exec_ddl(&server, &mut session, "DROP TABLE t")
        .await
        .expect("a bare drop is accepted");

    assert!(
        server.catalog.get_table(session_schema, "t").is_err(),
        "the session's table survived a bare drop"
    );
    assert!(
        server.catalog.get_table(other, "t").is_ok(),
        "a bare name reached another schema's table"
    );
}

#[tokio::test]
async fn a_schema_that_does_not_exist_is_refused() {
    let (server, _session_schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(&server, &mut session, "CREATE TABLE t (a INT)")
        .await
        .expect("table is created");

    let err = exec_ddl(&server, &mut session, "DROP TABLE no_such_schema.t")
        .await
        .expect_err("a name in a schema that does not exist is refused");
    assert!(
        err.to_lowercase().contains("schema"),
        "the error does not name the schema as the problem: {err}"
    );
}

#[tokio::test]
async fn a_maintenance_statement_matches_on_the_schema_it_names() {
    // ANALYZE, VACUUM, REINDEX and OPTIMIZE TABLE match against the whole
    // table list rather than resolving through a schema, because they also run
    // over everything when given no name. The schema part still has to count
    let (server, _session_schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    two_schemas_one_table_name(&server, &mut session, "t").await;

    exec_ddl(&server, &mut session, "ANALYZE other_schema.t")
        .await
        .expect("a qualified name reaches the other schema's table");
    exec_ddl(&server, &mut session, "ANALYZE t")
        .await
        .expect("a bare name still reaches the session's table");

    let missing_table = exec_ddl(&server, &mut session, "ANALYZE other_schema.nope")
        .await
        .expect_err("a table the named schema does not hold is refused");
    assert!(
        missing_table.contains("nope"),
        "the error does not name the table that is absent: {missing_table}"
    );

    let missing_schema = exec_ddl(&server, &mut session, "ANALYZE no_such_schema.t")
        .await
        .expect_err("a schema that does not exist is refused even though a table named t exists");
    assert!(
        missing_schema.contains("no_such_schema"),
        "a name in an absent schema matched the session's table: {missing_schema}"
    );
}

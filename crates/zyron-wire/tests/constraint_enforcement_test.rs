//! Declared PRIMARY KEY and UNIQUE constraints on a heap table.
//!
//! Heap uniqueness is enforced by probing unique B+tree indexes, and a
//! declared constraint used to create no index, so `UNIQUE (k)` in a CREATE
//! TABLE was stored in the catalog, reported over the wire, enforced on a
//! lake table, and never checked on the heap. Every duplicate a constraint
//! named was accepted.
//!
//! A second defect sat behind it: a B+tree key carried the leading column
//! only, so a multi-column unique index rejected rows whose full key was
//! distinct. Fixing the first without the second would have turned silent
//! non-enforcement into false rejection.
//!
//! Run: cargo test -p zyron-wire --test constraint_enforcement_test

mod common;

use common::{create_test_server, exec_ddl, exec_dml, exec_dml_result, new_session, query_rows};

/// Runs a statement that must be refused and returns the message, so a test
/// names the rule it expects rather than only that something failed.
async fn refuse(server: &std::sync::Arc<zyron_wire::connection::ServerState>, sql: &str) -> String {
    match exec_dml_result(server, sql).await {
        Ok(()) => panic!("expected `{sql}` to be refused"),
        Err(e) => e.to_string(),
    }
}

#[tokio::test]
async fn test_inline_unique_table_constraint_refuses_a_duplicate() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE t (id INT, k INT, UNIQUE (k))",
    )
    .await
    .expect("create");

    exec_dml(&server, "INSERT INTO t VALUES (1, 10)").await;
    let err = refuse(&server, "INSERT INTO t VALUES (2, 10)").await;
    assert!(err.contains("nique"), "{err}");

    assert_eq!(query_rows(&server, "SELECT * FROM t").await, 1);
}

#[tokio::test]
async fn test_inline_column_unique_and_primary_key_refuse_a_duplicate() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE t (id INT PRIMARY KEY, email VARCHAR UNIQUE)",
    )
    .await
    .expect("create");

    exec_dml(&server, "INSERT INTO t VALUES (1, 'a@example.com')").await;

    let err = refuse(&server, "INSERT INTO t VALUES (1, 'b@example.com')").await;
    assert!(err.contains("nique"), "primary key duplicate: {err}");

    let err = refuse(&server, "INSERT INTO t VALUES (2, 'a@example.com')").await;
    assert!(err.contains("nique"), "column unique duplicate: {err}");

    exec_dml(&server, "INSERT INTO t VALUES (2, 'b@example.com')").await;
    assert_eq!(query_rows(&server, "SELECT * FROM t").await, 2);
}

/// The constraint has to hold against rows already stored, not only against
/// others in the same batch.
#[tokio::test]
async fn test_a_duplicate_inside_one_insert_batch_is_refused() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE t (id INT, k INT, UNIQUE (k))",
    )
    .await
    .expect("create");

    let err = refuse(&server, "INSERT INTO t VALUES (1, 5), (2, 5)").await;
    assert!(err.contains("nique"), "{err}");
    assert_eq!(
        query_rows(&server, "SELECT * FROM t").await,
        0,
        "a refused statement leaves no partial write"
    );
}

/// A multi-column key conflicts only when every column matches. Enforcing on
/// the leading column alone refuses rows the constraint permits.
#[tokio::test]
async fn test_composite_unique_admits_a_repeated_leading_column() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE t (tenant INT, code INT, UNIQUE (tenant, code))",
    )
    .await
    .expect("create");

    exec_dml(&server, "INSERT INTO t VALUES (1, 100)").await;
    // Same tenant, different code. The whole key differs, so this is legal
    exec_dml(&server, "INSERT INTO t VALUES (1, 200)").await;
    // Same code, different tenant
    exec_dml(&server, "INSERT INTO t VALUES (2, 100)").await;
    assert_eq!(query_rows(&server, "SELECT * FROM t").await, 3);

    let err = refuse(&server, "INSERT INTO t VALUES (1, 100)").await;
    assert!(err.contains("nique"), "{err}");
    assert!(
        err.contains("tenant") && err.contains("code"),
        "the refusal names the whole key: {err}"
    );
}

#[tokio::test]
async fn test_composite_unique_index_admits_a_repeated_leading_column() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(&server, &mut session, "CREATE TABLE t (a INT, b INT)")
        .await
        .expect("create");
    exec_ddl(
        &server,
        &mut session,
        "CREATE UNIQUE INDEX t_ab ON t (a, b)",
    )
    .await
    .expect("index");

    exec_dml(&server, "INSERT INTO t VALUES (1, 1)").await;
    exec_dml(&server, "INSERT INTO t VALUES (1, 2)").await;
    assert_eq!(query_rows(&server, "SELECT * FROM t").await, 2);

    let err = refuse(&server, "INSERT INTO t VALUES (1, 2)").await;
    assert!(err.contains("nique"), "{err}");
}

/// A variable-length leading component must be delimited, or `('ab','c')` and
/// `('a','bc')` produce the same key bytes and one is refused as a duplicate
/// of the other.
#[tokio::test]
async fn test_composite_unique_over_text_columns_separates_the_components() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE t (a VARCHAR, b VARCHAR, UNIQUE (a, b))",
    )
    .await
    .expect("create");

    exec_dml(&server, "INSERT INTO t VALUES ('ab', 'c')").await;
    exec_dml(&server, "INSERT INTO t VALUES ('a', 'bc')").await;
    // A value that is a prefix of another still orders below it, so neither
    // shadows the other
    exec_dml(&server, "INSERT INTO t VALUES ('a', 'b')").await;
    assert_eq!(query_rows(&server, "SELECT * FROM t").await, 3);

    let err = refuse(&server, "INSERT INTO t VALUES ('a', 'bc')").await;
    assert!(err.contains("nique"), "{err}");
}

/// SQL treats nulls as distinct for uniqueness, so a null in the key never
/// conflicts.
#[tokio::test]
async fn test_a_null_in_the_key_never_conflicts() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE t (a INT, b INT, UNIQUE (a, b))",
    )
    .await
    .expect("create");

    exec_dml(&server, "INSERT INTO t VALUES (1, NULL)").await;
    exec_dml(&server, "INSERT INTO t VALUES (1, NULL)").await;
    exec_dml(&server, "INSERT INTO t VALUES (NULL, NULL)").await;
    assert_eq!(query_rows(&server, "SELECT * FROM t").await, 3);
}

/// An UPDATE moving a row onto a key another row holds is a violation, and a
/// row keeping its own key is not.
#[tokio::test]
async fn test_update_onto_an_existing_key_is_refused_and_a_no_op_update_is_not() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE t (id INT, k INT, UNIQUE (k))",
    )
    .await
    .expect("create");

    exec_dml(&server, "INSERT INTO t VALUES (1, 10), (2, 20)").await;

    let err = refuse(&server, "UPDATE t SET k = 10 WHERE id = 2").await;
    assert!(err.contains("nique"), "{err}");

    // Rewriting a row with the key it already holds conflicts with nothing
    exec_dml(&server, "UPDATE t SET k = 20 WHERE id = 2").await;
    exec_dml(&server, "UPDATE t SET k = 30 WHERE id = 2").await;
    assert_eq!(query_rows(&server, "SELECT * FROM t WHERE k = 30").await, 1);
}

/// A key a deleted row held is free again.
#[tokio::test]
async fn test_a_deleted_rows_key_can_be_reused() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE t (id INT, k INT, UNIQUE (k))",
    )
    .await
    .expect("create");

    exec_dml(&server, "INSERT INTO t VALUES (1, 10)").await;
    exec_dml(&server, "DELETE FROM t WHERE id = 1").await;
    exec_dml(&server, "INSERT INTO t VALUES (2, 10)").await;
    assert_eq!(query_rows(&server, "SELECT * FROM t").await, 1);
}

/// ALTER TABLE ADD CONSTRAINT validated the rows already stored and then left
/// nothing enforcing the rule, so the next duplicate landed.
#[tokio::test]
async fn test_added_constraint_enforces_later_inserts() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(&server, &mut session, "CREATE TABLE t (id INT, k INT)")
        .await
        .expect("create");
    exec_dml(&server, "INSERT INTO t VALUES (1, 10), (2, 20)").await;

    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE t ADD CONSTRAINT t_k_uq UNIQUE (k)",
    )
    .await
    .expect("add constraint");

    // Against a row that predates the constraint
    let err = refuse(&server, "INSERT INTO t VALUES (3, 10)").await;
    assert!(err.contains("nique"), "{err}");

    exec_dml(&server, "INSERT INTO t VALUES (3, 30)").await;
    let err = refuse(&server, "INSERT INTO t VALUES (4, 30)").await;
    assert!(err.contains("nique"), "{err}");
}

/// A constraint that already violates the stored rows is refused, and refusing
/// it leaves nothing enforcing behind.
#[tokio::test]
async fn test_added_constraint_is_refused_when_existing_rows_violate_it() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(&server, &mut session, "CREATE TABLE t (id INT, k INT)")
        .await
        .expect("create");
    exec_dml(&server, "INSERT INTO t VALUES (1, 10), (2, 10)").await;

    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE t ADD CONSTRAINT t_k_uq UNIQUE (k)",
    )
    .await
    .expect_err("existing duplicate must refuse the constraint");

    // The duplicate the constraint would have banned is still insertable,
    // because the constraint was not created
    exec_dml(&server, "INSERT INTO t VALUES (3, 10)").await;
    assert_eq!(query_rows(&server, "SELECT * FROM t").await, 3);
}

/// Dropping the constraint takes its enforcement with it.
#[tokio::test]
async fn test_dropped_constraint_stops_enforcing() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE t (id INT, k INT, CONSTRAINT t_k_uq UNIQUE (k))",
    )
    .await
    .expect("create");

    exec_dml(&server, "INSERT INTO t VALUES (1, 10)").await;
    let err = refuse(&server, "INSERT INTO t VALUES (2, 10)").await;
    assert!(err.contains("nique"), "{err}");

    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE t DROP CONSTRAINT t_k_uq",
    )
    .await
    .expect("drop constraint");

    exec_dml(&server, "INSERT INTO t VALUES (2, 10)").await;
    assert_eq!(query_rows(&server, "SELECT * FROM t").await, 2);
}

/// NOT ENFORCED keeps the declaration for the planner and asks the write path
/// to skip it, so it must not gain an index that enforces it anyway.
#[tokio::test]
async fn test_not_enforced_unique_admits_a_duplicate() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE t (id INT, k INT, UNIQUE (k) NOT ENFORCED)",
    )
    .await
    .expect("create");

    exec_dml(&server, "INSERT INTO t VALUES (1, 10)").await;
    exec_dml(&server, "INSERT INTO t VALUES (2, 10)").await;
    assert_eq!(query_rows(&server, "SELECT * FROM t").await, 2);
}

/// A transaction's own writes are certain to it before they commit, so the
/// key a statement freed is reusable by the next statement in the same
/// transaction, and a key it just took is already taken. Judging candidates
/// by commit status alone gets both backwards.
#[tokio::test]
async fn test_a_transaction_sees_its_own_delete_and_its_own_insert() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE t (id INT, k INT, UNIQUE (k))",
    )
    .await
    .expect("create");
    exec_dml(&server, "INSERT INTO t VALUES (1, 10)").await;

    // Delete then reinsert the same key, both inside one transaction
    common::exec_dml_script(
        &server,
        &["DELETE FROM t WHERE k = 10", "INSERT INTO t VALUES (2, 10)"],
    )
    .await
    .expect("delete then reinsert in one transaction");
    assert_eq!(query_rows(&server, "SELECT * FROM t WHERE k = 10").await, 1);

    // Two statements of one transaction taking the same key still collide
    let err = common::exec_dml_script(
        &server,
        &[
            "INSERT INTO t VALUES (3, 30)",
            "INSERT INTO t VALUES (4, 30)",
        ],
    )
    .await
    .expect_err("the second statement takes a key the first took");
    assert!(err.to_string().contains("nique"), "{err}");
}

/// The index a constraint creates serves reads too, so the rows stay
/// findable through it rather than only through a scan.
#[tokio::test]
async fn test_the_constraint_index_returns_rows_on_a_point_lookup() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE t (id INT PRIMARY KEY, v INT)",
    )
    .await
    .expect("create");
    for i in 0..40 {
        exec_dml(&server, &format!("INSERT INTO t VALUES ({i}, {})", i * 3)).await;
    }

    assert_eq!(
        query_rows(&server, "SELECT v FROM t WHERE id = 17").await,
        1
    );
    assert_eq!(
        query_rows(&server, "SELECT v FROM t WHERE id = 99").await,
        0
    );
    assert_eq!(
        query_rows(&server, "SELECT v FROM t WHERE id >= 10 AND id < 20").await,
        10
    );
}

/// A declared length is part of the type. Storing a longer value would put a
/// row in the table that the schema says cannot exist, and hand the next
/// reader a value wider than the column it read from.
#[tokio::test]
async fn test_a_value_longer_than_the_column_declares_is_refused() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE t (id INT, code VARCHAR(5), tag CHAR(3))",
    )
    .await
    .expect("create");

    // At the limit and under it, both stored as written
    exec_dml(&server, "INSERT INTO t VALUES (1, 'abcde', 'xyz')").await;
    exec_dml(&server, "INSERT INTO t VALUES (2, 'ab', 'x')").await;
    assert_eq!(query_rows(&server, "SELECT * FROM t").await, 2);

    let err = refuse(&server, "INSERT INTO t VALUES (3, 'abcdef', 'xyz')").await;
    assert!(err.contains("too long") && err.contains("code"), "{err}");

    let err = refuse(&server, "INSERT INTO t VALUES (4, 'abc', 'wxyz')").await;
    assert!(err.contains("too long") && err.contains("tag"), "{err}");

    // An UPDATE that lengthens a value is held to the same limit
    let err = refuse(&server, "UPDATE t SET code = 'abcdefgh' WHERE id = 1").await;
    assert!(err.contains("too long"), "{err}");

    // A NULL carries no length
    exec_dml(&server, "INSERT INTO t VALUES (5, NULL, NULL)").await;
    assert_eq!(query_rows(&server, "SELECT * FROM t").await, 3);
}

/// A length counts characters rather than bytes, so a multi-byte string is
/// measured the way SQL declares it.
#[tokio::test]
async fn test_a_declared_text_length_counts_characters_not_bytes() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(&server, &mut session, "CREATE TABLE t (s VARCHAR(4))")
        .await
        .expect("create");

    // Four characters, more than four bytes
    exec_dml(&server, "INSERT INTO t VALUES ('åéîø')").await;
    assert_eq!(query_rows(&server, "SELECT * FROM t").await, 1);

    let err = refuse(&server, "INSERT INTO t VALUES ('åéîøu')").await;
    assert!(err.contains("too long"), "{err}");
}

/// A constraint over a column that already carries a matching unique index
/// reuses it instead of building a second tree over the same key.
#[tokio::test]
async fn test_a_constraint_over_an_indexed_key_reuses_the_index() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(&server, &mut session, "CREATE TABLE t (id INT, k INT)")
        .await
        .expect("create");
    exec_ddl(&server, &mut session, "CREATE UNIQUE INDEX t_k_ux ON t (k)")
        .await
        .expect("index");
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE t ADD CONSTRAINT t_k_uq UNIQUE (k)",
    )
    .await
    .expect("add constraint");

    let table = server
        .catalog
        .list_all_tables()
        .into_iter()
        .find(|t| t.name == "t")
        .expect("table");
    let btrees: Vec<String> = server
        .catalog
        .get_indexes_for_table(table.id)
        .into_iter()
        .filter(|i| i.index_type == zyron_catalog::IndexType::BTree)
        .map(|i| i.name.clone())
        .collect();
    assert_eq!(
        btrees,
        vec!["t_k_ux".to_string()],
        "the existing index answers for the constraint"
    );

    exec_dml(&server, "INSERT INTO t VALUES (1, 10)").await;
    let err = refuse(&server, "INSERT INTO t VALUES (2, 10)").await;
    assert!(err.contains("nique"), "{err}");
}

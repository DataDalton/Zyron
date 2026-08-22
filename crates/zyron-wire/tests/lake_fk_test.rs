//! Foreign keys and triggers hold on lake DML the way they hold on heap DML.
//!
//! LakeDeleteOperator and LakeUpdateOperator committed straight to the
//! table's log without a referential check, so a referenced parent row
//! deleted cleanly, an update could orphan its children, and DELETE
//! triggers never fired.
//!
//! Run: cargo test -p zyron-wire --test lake_fk_test

mod common;

use common::{create_test_server, exec_ddl, exec_dml, exec_dml_result, new_session, query_rows};

/// Deleting a lake parent row that a heap child references is refused.
#[tokio::test]
async fn test_lake_delete_enforces_restrict() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE p (k BIGINT NOT NULL) USING ZYRONLAKE",
    )
    .await
    .expect("create parent");
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE c (id BIGINT NOT NULL, k BIGINT, FOREIGN KEY (k) REFERENCES p(k))",
    )
    .await
    .expect("create child");
    exec_dml(&server, "INSERT INTO p VALUES (1), (2)").await;
    exec_dml(&server, "INSERT INTO c VALUES (10, 1)").await;

    let err = exec_dml_result(&server, "DELETE FROM p WHERE k = 1")
        .await
        .expect_err("a referenced lake parent row must not delete");
    assert!(
        err.to_string().contains("foreign key"),
        "the refusal names the constraint: {err}"
    );
    assert_eq!(
        query_rows(&server, "SELECT k FROM p").await,
        2,
        "the refused delete removed nothing"
    );

    // The unreferenced row still deletes
    exec_dml(&server, "DELETE FROM p WHERE k = 2").await;
    assert_eq!(query_rows(&server, "SELECT k FROM p").await, 1);
}

/// Updating a lake child to reference a missing parent is refused.
#[tokio::test]
async fn test_lake_update_enforces_child_fk() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(&server, &mut session, "CREATE TABLE p (k BIGINT NOT NULL)")
        .await
        .expect("create parent");
    exec_ddl(&server, &mut session, "CREATE UNIQUE INDEX p_k ON p (k)")
        .await
        .expect("index");
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE c (id BIGINT NOT NULL, k BIGINT, FOREIGN KEY (k) REFERENCES p(k)) USING ZYRONLAKE",
    )
    .await
    .expect("create child");
    exec_dml(&server, "INSERT INTO p VALUES (1), (2)").await;
    exec_dml(&server, "INSERT INTO c VALUES (10, 1)").await;

    let err = exec_dml_result(&server, "UPDATE c SET k = 99 WHERE id = 10")
        .await
        .expect_err("an update orphaning the child must be refused");
    assert!(
        err.to_string().contains("foreign key"),
        "the refusal names the constraint: {err}"
    );

    // A legal retarget still lands
    exec_dml(&server, "UPDATE c SET k = 2 WHERE id = 10").await;
    assert_eq!(query_rows(&server, "SELECT k FROM c WHERE k = 2").await, 1);
}

/// Updating a referenced lake parent key out from under its children is
/// refused.
#[tokio::test]
async fn test_lake_update_enforces_parent_restrict() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE p (k BIGINT NOT NULL) USING ZYRONLAKE",
    )
    .await
    .expect("create parent");
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE c (id BIGINT NOT NULL, k BIGINT, FOREIGN KEY (k) REFERENCES p(k))",
    )
    .await
    .expect("create child");
    exec_dml(&server, "INSERT INTO p VALUES (1)").await;
    exec_dml(&server, "INSERT INTO c VALUES (10, 1)").await;

    let err = exec_dml_result(&server, "UPDATE p SET k = 5 WHERE k = 1")
        .await
        .expect_err("moving a referenced key must be refused");
    assert!(
        err.to_string().contains("foreign key"),
        "the refusal names the constraint: {err}"
    );
}

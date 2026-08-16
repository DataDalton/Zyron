//! INSERT ... ON CONFLICT is refused until it is implemented.
//!
//! The clause parses but nothing reads it, so DO NOTHING raised a unique
//! violation and DO UPDATE inserted a plain row, both silently dropping
//! the conflict action the statement asked for. A statement whose clause
//! cannot be honored is refused loudly instead.
//!
//! Run: cargo test -p zyron-wire --test on_conflict_test

mod common;

use common::{create_test_server, exec_ddl, exec_dml, new_session, query_error, query_rows};

#[tokio::test]
async fn test_on_conflict_is_refused_not_ignored() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE t (k BIGINT NOT NULL, v BIGINT)",
    )
    .await
    .expect("create");
    exec_ddl(&server, &mut session, "CREATE UNIQUE INDEX ix ON t (k)")
        .await
        .expect("index");
    exec_dml(&server, "INSERT INTO t VALUES (1, 10)").await;

    let err = query_error(
        &server,
        "INSERT INTO t VALUES (2, 20) ON CONFLICT (k) DO NOTHING",
    )
    .await;
    assert!(
        err.contains("ON CONFLICT"),
        "the refusal names the clause: {err}"
    );

    let err = query_error(
        &server,
        "INSERT INTO t VALUES (1, 30) ON CONFLICT (k) DO UPDATE SET v = 30",
    )
    .await;
    assert!(
        err.contains("ON CONFLICT"),
        "the refusal names the clause: {err}"
    );

    // The refused statements changed nothing
    assert_eq!(query_rows(&server, "SELECT k FROM t").await, 1);
}

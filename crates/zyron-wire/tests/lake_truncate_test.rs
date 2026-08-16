//! TRUNCATE on a lake table removes its rows.
//!
//! A lake table's rows live in its transaction log manifest, not in the
//! heap file the generic truncate path clears, so the statement used to
//! report success while every row survived.
//!
//! Run: cargo test -p zyron-wire --test lake_truncate_test

mod common;

use common::{create_test_server, exec_ddl, exec_dml, new_session, query_rows};

#[tokio::test]
async fn test_truncate_empties_a_lake_table() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE lk (id BIGINT NOT NULL, name TEXT) USING ZYRONLAKE",
    )
    .await
    .expect("create");
    exec_dml(&server, "INSERT INTO lk VALUES (1, 'a'), (2, 'b'), (3, 'c')").await;
    assert_eq!(query_rows(&server, "SELECT id FROM lk").await, 3);

    exec_ddl(&server, &mut session, "TRUNCATE TABLE lk")
        .await
        .expect("truncate");

    assert_eq!(
        query_rows(&server, "SELECT id FROM lk").await,
        0,
        "TRUNCATE must remove the manifest's rows, not only the empty heap"
    );

    // The table stays usable after the truncate
    exec_dml(&server, "INSERT INTO lk VALUES (4, 'd')").await;
    assert_eq!(query_rows(&server, "SELECT id FROM lk").await, 1);
}

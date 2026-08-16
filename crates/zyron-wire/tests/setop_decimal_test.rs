//! Set operations over decimal columns of different declared scales.
//!
//! The stored representation is a scaled integer, so 10.50 at scale two and
//! 10.500 at scale three are different raw values for one number. A set
//! operation that hashes and compares the raw integers keeps duplicates in
//! UNION and misses every cross-branch match in INTERSECT and EXCEPT.

mod common;

use common::{create_test_server, exec_ddl, exec_dml, new_session, query_rows};

#[tokio::test]
async fn set_operations_compare_decimal_values_not_raw_integers() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE sd_a (v DECIMAL(10,2) NOT NULL)",
    )
    .await
    .expect("create a");
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE sd_b (v DECIMAL(10,3) NOT NULL)",
    )
    .await
    .expect("create b");
    exec_dml(&server, "INSERT INTO sd_a VALUES (10.50), (7.25)").await;
    exec_dml(&server, "INSERT INTO sd_b VALUES (10.500), (3.125)").await;

    assert_eq!(
        query_rows(
            &server,
            "SELECT v FROM sd_a UNION SELECT v FROM sd_b ORDER BY v"
        )
        .await,
        3,
        "10.50 and 10.500 are one value, the union holds 3.125, 7.25, 10.50"
    );
    assert_eq!(
        query_rows(
            &server,
            "SELECT v FROM sd_a INTERSECT SELECT v FROM sd_b"
        )
        .await,
        1,
        "the branches share exactly the value 10.5"
    );
    assert_eq!(
        query_rows(&server, "SELECT v FROM sd_a EXCEPT SELECT v FROM sd_b").await,
        1,
        "only 7.25 is exclusive to the left branch"
    );
}

#[tokio::test]
async fn same_scale_set_operations_still_work() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE sd_c (v DECIMAL(10,2) NOT NULL)",
    )
    .await
    .expect("create c");
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE sd_d (v DECIMAL(10,2) NOT NULL)",
    )
    .await
    .expect("create d");
    exec_dml(&server, "INSERT INTO sd_c VALUES (1.00), (2.00)").await;
    exec_dml(&server, "INSERT INTO sd_d VALUES (2.00), (3.00)").await;

    assert_eq!(
        query_rows(&server, "SELECT v FROM sd_c UNION SELECT v FROM sd_d").await,
        3
    );
    assert_eq!(
        query_rows(
            &server,
            "SELECT v FROM sd_c INTERSECT SELECT v FROM sd_d"
        )
        .await,
        1
    );
    assert_eq!(
        query_rows(&server, "SELECT v FROM sd_c EXCEPT SELECT v FROM sd_d").await,
        1
    );
}

//! Lake unique enforcement across pending versions.
//!
//! The unique probe must read the pending-inclusive head, not only the
//! published manifest. Probing the published head admits a duplicate key
//! twice inside one transaction, and admits it across two overlapping
//! transactions whose commits rule R6 declares conflict-free.

mod common;

use std::sync::Arc;

use common::{create_test_server, exec_ddl, exec_dml, exec_dml_script, new_session, query_rows};
use zyron_catalog::DatabaseId;
use zyron_storage::txn::IsolationLevel;
use zyron_wire::connection::ServerState;

/// Runs one statement inside an already-open transaction without resolving
/// its lake versions, so a second transaction can observe the pending state
async fn run_stmt_pending(
    server: &Arc<ServerState>,
    txn_id: u32,
    snapshot: zyron_storage::Snapshot,
    sql: &str,
) -> Result<(), zyron_common::ZyronError> {
    let stmt = zyron_parser::parse(sql)
        .expect("parse")
        .into_iter()
        .next()
        .expect("one statement");
    let plan = zyron_planner::plan(
        &server.catalog,
        DatabaseId(1),
        vec!["public".into()],
        stmt,
        None,
    )
    .await
    .expect("plan");
    let mut ctx = zyron_executor::context::ExecutionContext::new(
        server.catalog.clone(),
        server.wal.clone(),
        server.buffer_pool.clone(),
        server.disk_manager.clone(),
        txn_id,
        snapshot,
    );
    ctx.heap_files = Some(Arc::clone(&server.heap_files));
    ctx.btree_indexes = Some(Arc::clone(&server.btree_indexes));
    ctx.doc_registry = Some(Arc::clone(&server.doc_registry));
    ctx.row_locks = Some(Arc::clone(server.txn_manager.lock_table()));
    let ctx = Arc::new(ctx);
    zyron_executor::execute(plan, &ctx).await.map(|_| ())
}

#[tokio::test]
async fn same_transaction_duplicate_key_is_refused() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE uq_txn (id BIGINT NOT NULL PRIMARY KEY, tag TEXT) USING ZYRONLAKE",
    )
    .await
    .expect("create");

    // The second statement's probe must see the first statement's pending
    // version, a probe of the published head sees an empty table twice
    let result = exec_dml_script(
        &server,
        &[
            "INSERT INTO uq_txn VALUES (7, 'first')",
            "INSERT INTO uq_txn VALUES (7, 'second')",
        ],
    )
    .await;
    let err = result.expect_err("the duplicate key must be refused");
    assert!(
        err.to_string().contains("duplicate key"),
        "expected a duplicate key violation, got {err}"
    );
    assert_eq!(
        query_rows(&server, "SELECT id FROM uq_txn").await,
        0,
        "the failed transaction leaves nothing behind"
    );
}

#[tokio::test]
async fn overlapping_transactions_cannot_both_insert_one_key() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE uq_two (id BIGINT NOT NULL PRIMARY KEY, tag TEXT) USING ZYRONLAKE",
    )
    .await
    .expect("create");

    let mut t1 = server
        .txn_manager
        .begin(IsolationLevel::ReadCommitted)
        .expect("begin t1");
    let s1 = t1.snapshot.clone();
    run_stmt_pending(
        &server,
        t1.txn_id as u32,
        s1,
        "INSERT INTO uq_two VALUES (42, 'first')",
    )
    .await
    .expect("first writer inserts");

    // The second transaction probes while the first is still pending. The
    // pending-inclusive head carries the key, so this is a duplicate
    let mut t2 = server
        .txn_manager
        .begin(IsolationLevel::ReadCommitted)
        .expect("begin t2");
    let s2 = t2.snapshot.clone();
    let second = run_stmt_pending(
        &server,
        t2.txn_id as u32,
        s2,
        "INSERT INTO uq_two VALUES (42, 'second')",
    )
    .await;
    let err = second.expect_err("the overlapping duplicate must be refused");
    assert!(
        err.to_string().contains("duplicate key"),
        "expected a duplicate key violation, got {err}"
    );
    let _ = zyron_lake::abandon_txn(server.disk_manager.data_dir(), t2.txn_id);
    server.txn_manager.abort(&mut t2).expect("abort t2");

    server.txn_manager.commit(&mut t1).await.expect("commit t1");
    let logs =
        zyron_lake::publish_txn(server.disk_manager.data_dir(), t1.txn_id).expect("publish t1");
    zyron_wire::connection::refresh_lake_stats(&server, &logs);

    assert_eq!(
        query_rows(&server, "SELECT id FROM uq_two WHERE id = 42").await,
        1,
        "exactly one row carries the key"
    );
}

#[tokio::test]
async fn sequential_inserts_of_distinct_keys_still_pass() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE uq_ok (id BIGINT NOT NULL PRIMARY KEY, tag TEXT) USING ZYRONLAKE",
    )
    .await
    .expect("create");

    exec_dml(&server, "INSERT INTO uq_ok VALUES (1, 'a'), (2, 'b')").await;
    exec_dml(&server, "INSERT INTO uq_ok VALUES (3, 'c')").await;
    exec_dml_script(
        &server,
        &[
            "INSERT INTO uq_ok VALUES (4, 'd')",
            "INSERT INTO uq_ok VALUES (5, 'e')",
        ],
    )
    .await
    .expect("distinct keys inside one transaction pass");
    assert_eq!(query_rows(&server, "SELECT id FROM uq_ok").await, 5);

    // A rewrite that keeps its own key does not collide with itself
    exec_dml(&server, "UPDATE uq_ok SET tag = 'a2' WHERE id = 1").await;
    assert_eq!(query_rows(&server, "SELECT id FROM uq_ok").await, 5);
}

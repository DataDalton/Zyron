//! First-committer-wins for heap UPDATE and DELETE.
//!
//! Two transactions whose snapshots overlap write the same row. The first
//! to commit wins, the second must fail with a transaction conflict rather
//! than re-stamp the superseded image. Re-stamping duplicates the row on
//! update-update, silently loses a delete on update-delete, and resurrects
//! a deleted row on delete-update.

mod common;

use std::sync::Arc;

use zyron_catalog::DatabaseId;
use zyron_executor::column::ScalarValue;
use zyron_storage::txn::{IsolationLevel, Transaction};
use zyron_wire::connection::ServerState;

/// Runs one statement inside an already-open transaction using the snapshot
/// the caller captured, which is how an overlapping writer sees the world
/// as it was before the other transaction committed
async fn run_stmt(
    server: &Arc<ServerState>,
    txn: &Transaction,
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
        txn.txn_id as u32,
        snapshot,
    );
    ctx.heap_files = Some(Arc::clone(&server.heap_files));
    ctx.btree_indexes = Some(Arc::clone(&server.btree_indexes));
    ctx.doc_registry = Some(Arc::clone(&server.doc_registry));
    ctx.row_locks = Some(Arc::clone(server.txn_manager.lock_table()));
    let ctx = Arc::new(ctx);
    zyron_executor::execute(plan, &ctx).await.map(|_| ())
}

fn is_conflict(r: &Result<(), zyron_common::ZyronError>) -> bool {
    matches!(
        r,
        Err(zyron_common::ZyronError::TransactionConflict { .. })
    )
}

/// Collects the single integer column of a result as sorted values
async fn int_column(server: &Arc<ServerState>, sql: &str) -> Vec<i32> {
    let mut out: Vec<i32> = common::query_values(server, sql)
        .await
        .into_iter()
        .map(|row| match row[0] {
            ScalarValue::Int32(v) => v,
            ref other => panic!("unexpected scalar {other:?}"),
        })
        .collect();
    out.sort();
    out
}

#[tokio::test]
async fn concurrent_updates_of_one_row_conflict_instead_of_duplicating() {
    let (server, _schema, _tmp) = common::create_test_server().await;
    let mut session = common::new_session();
    common::exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE wc_upd (id INT NOT NULL, v INT NOT NULL)",
    )
    .await
    .expect("create table");
    common::exec_dml(&server, "INSERT INTO wc_upd VALUES (1, 0)").await;

    // Both snapshots predate either commit, so each scan returns the
    // original image of the row
    let mut t1 = server
        .txn_manager
        .begin(IsolationLevel::ReadCommitted)
        .expect("begin t1");
    let mut t2 = server
        .txn_manager
        .begin(IsolationLevel::ReadCommitted)
        .expect("begin t2");
    let s1 = t1.snapshot.clone();
    let s2 = t2.snapshot.clone();

    run_stmt(&server, &t1, s1, "UPDATE wc_upd SET v = 10 WHERE id = 1")
        .await
        .expect("first writer updates");
    server.txn_manager.commit(&mut t1).await.expect("commit t1");

    let second = run_stmt(&server, &t2, s2, "UPDATE wc_upd SET v = 20 WHERE id = 1").await;
    if second.is_ok() {
        server.txn_manager.commit(&mut t2).await.expect("commit t2");
    } else {
        server.txn_manager.abort(&mut t2).expect("abort t2");
    }

    let values = int_column(&server, "SELECT v FROM wc_upd WHERE id = 1").await;
    assert_eq!(
        values.len(),
        1,
        "concurrent updates duplicated the row: {values:?}"
    );
    assert_eq!(values, vec![10], "the first committer's value survives");
    assert!(
        is_conflict(&second),
        "the second writer must fail with a transaction conflict, got {second:?}"
    );
}

#[tokio::test]
async fn concurrent_update_then_delete_conflicts_instead_of_losing_the_delete() {
    let (server, _schema, _tmp) = common::create_test_server().await;
    let mut session = common::new_session();
    common::exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE wc_ud (id INT NOT NULL, v INT NOT NULL)",
    )
    .await
    .expect("create table");
    common::exec_dml(&server, "INSERT INTO wc_ud VALUES (1, 0)").await;

    let mut t1 = server
        .txn_manager
        .begin(IsolationLevel::ReadCommitted)
        .expect("begin t1");
    let mut t2 = server
        .txn_manager
        .begin(IsolationLevel::ReadCommitted)
        .expect("begin t2");
    let s1 = t1.snapshot.clone();
    let s2 = t2.snapshot.clone();

    run_stmt(&server, &t1, s1, "UPDATE wc_ud SET v = 10 WHERE id = 1")
        .await
        .expect("first writer updates");
    server.txn_manager.commit(&mut t1).await.expect("commit t1");

    // The delete targets the superseded image. Stamping it reports success
    // while the updated row lives on, a silently lost delete
    let second = run_stmt(&server, &t2, s2, "DELETE FROM wc_ud WHERE id = 1").await;
    if second.is_ok() {
        server.txn_manager.commit(&mut t2).await.expect("commit t2");
    } else {
        server.txn_manager.abort(&mut t2).expect("abort t2");
    }

    assert!(
        is_conflict(&second),
        "a delete of a concurrently updated row must conflict, got {second:?}"
    );
    let values = int_column(&server, "SELECT v FROM wc_ud WHERE id = 1").await;
    assert_eq!(values, vec![10], "the committed update survives the conflict");
}

#[tokio::test]
async fn concurrent_delete_then_update_conflicts_instead_of_resurrecting() {
    let (server, _schema, _tmp) = common::create_test_server().await;
    let mut session = common::new_session();
    common::exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE wc_du (id INT NOT NULL, v INT NOT NULL)",
    )
    .await
    .expect("create table");
    common::exec_dml(&server, "INSERT INTO wc_du VALUES (1, 0)").await;

    let mut t1 = server
        .txn_manager
        .begin(IsolationLevel::ReadCommitted)
        .expect("begin t1");
    let mut t2 = server
        .txn_manager
        .begin(IsolationLevel::ReadCommitted)
        .expect("begin t2");
    let s1 = t1.snapshot.clone();
    let s2 = t2.snapshot.clone();

    run_stmt(&server, &t1, s1, "DELETE FROM wc_du WHERE id = 1")
        .await
        .expect("first writer deletes");
    server.txn_manager.commit(&mut t1).await.expect("commit t1");

    // The update would re-stamp the deleted image with its own xmax and
    // append a new version, bringing the deleted row back to life
    let second = run_stmt(&server, &t2, s2, "UPDATE wc_du SET v = 20 WHERE id = 1").await;
    if second.is_ok() {
        server.txn_manager.commit(&mut t2).await.expect("commit t2");
    } else {
        server.txn_manager.abort(&mut t2).expect("abort t2");
    }

    let values = int_column(&server, "SELECT v FROM wc_du WHERE id = 1").await;
    assert_eq!(
        values.len(),
        0,
        "a committed delete must not be resurrected by a concurrent update: {values:?}"
    );
    assert!(
        is_conflict(&second),
        "the update must fail with a transaction conflict, got {second:?}"
    );
}

#[tokio::test]
async fn sequential_writers_do_not_conflict() {
    let (server, _schema, _tmp) = common::create_test_server().await;
    let mut session = common::new_session();
    common::exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE wc_seq (id INT NOT NULL, v INT NOT NULL)",
    )
    .await
    .expect("create table");
    common::exec_dml(&server, "INSERT INTO wc_seq VALUES (1, 0)").await;

    // The second transaction begins after the first commits, its snapshot
    // sees the new image and updates it without any conflict
    common::exec_dml(&server, "UPDATE wc_seq SET v = 10 WHERE id = 1").await;
    common::exec_dml(&server, "UPDATE wc_seq SET v = 20 WHERE id = 1").await;
    common::exec_dml(&server, "DELETE FROM wc_seq WHERE id = 1").await;

    let values = int_column(&server, "SELECT v FROM wc_seq").await;
    assert_eq!(values.len(), 0, "sequential writers apply cleanly");
}

#[tokio::test]
async fn same_transaction_restamps_its_own_rows_without_conflict() {
    let (server, _schema, _tmp) = common::create_test_server().await;
    let mut session = common::new_session();
    common::exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE wc_own (id INT NOT NULL, v INT NOT NULL)",
    )
    .await
    .expect("create table");
    common::exec_dml(&server, "INSERT INTO wc_own VALUES (1, 0)").await;

    // Two updates and a delete of the same row inside one transaction, the
    // recheck must treat this transaction's own stamps as its own
    common::exec_dml_script(
        &server,
        &[
            "UPDATE wc_own SET v = 1 WHERE id = 1",
            "UPDATE wc_own SET v = 2 WHERE id = 1",
            "DELETE FROM wc_own WHERE id = 1",
        ],
    )
    .await
    .expect("one transaction rewrites its own row");

    let values = int_column(&server, "SELECT v FROM wc_own").await;
    assert_eq!(values.len(), 0, "the transaction's final delete applies");
}

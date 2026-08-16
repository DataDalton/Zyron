//! Lake DML must notify the CDC hook the way heap DML does.
//!
//! A CDC feed on a lake table would otherwise stay silent: the lake insert,
//! update and delete paths commit through the table's transaction log and
//! never passed their row images to the hook, so a feed, slot or stream
//! watching the table missed every change while the statement reported
//! success.

mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use common::{create_test_server, exec_ddl, new_session};
use zyron_catalog::DatabaseId;
use zyron_executor::context::CdcHook;
use zyron_wire::connection::ServerState;

#[derive(Default)]
struct CountingHook {
    inserts: AtomicUsize,
    updates: AtomicUsize,
    deletes: AtomicUsize,
    insert_bytes: Mutex<Vec<usize>>,
}

impl CdcHook for CountingHook {
    fn on_insert(
        &self,
        _table_id: u32,
        tuples: &[&[u8]],
        _version: u64,
        _timestamp: i64,
        _txn_id: u32,
        _is_last_in_txn: bool,
    ) -> zyron_common::Result<()> {
        self.inserts.fetch_add(tuples.len(), Ordering::SeqCst);
        self.insert_bytes
            .lock()
            .expect("lock")
            .extend(tuples.iter().map(|t| t.len()));
        Ok(())
    }

    fn on_delete(
        &self,
        _table_id: u32,
        old_data: &[&[u8]],
        _version: u64,
        _timestamp: i64,
        _txn_id: u32,
        _is_last_in_txn: bool,
    ) -> zyron_common::Result<()> {
        self.deletes.fetch_add(old_data.len(), Ordering::SeqCst);
        Ok(())
    }

    fn on_update(
        &self,
        _table_id: u32,
        old_data: &[&[u8]],
        new_data: &[&[u8]],
        _version: u64,
        _timestamp: i64,
        _txn_id: u32,
        _is_last_in_txn: bool,
    ) -> zyron_common::Result<()> {
        assert_eq!(
            old_data.len(),
            new_data.len(),
            "update pairs old and new one to one"
        );
        self.updates.fetch_add(new_data.len(), Ordering::SeqCst);
        Ok(())
    }
}

/// Runs one DML statement with the hook installed, committing and publishing
/// the way the wire layer does.
async fn exec_dml_hooked(server: &Arc<ServerState>, hook: &Arc<CountingHook>, sql: &str) {
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
    let mut txn = server
        .txn_manager
        .begin(zyron_storage::txn::IsolationLevel::ReadCommitted)
        .expect("begin");
    let txn_id = txn.txn_id;
    let snapshot = txn.snapshot.clone();
    let mut ctx = zyron_executor::context::ExecutionContext::new(
        server.catalog.clone(),
        server.wal.clone(),
        server.buffer_pool.clone(),
        server.disk_manager.clone(),
        txn_id as u32,
        snapshot,
    );
    ctx.heap_files = Some(Arc::clone(&server.heap_files));
    ctx.btree_indexes = Some(Arc::clone(&server.btree_indexes));
    ctx.doc_registry = Some(Arc::clone(&server.doc_registry));
    ctx.cdc_hook = Some(Arc::clone(hook) as Arc<dyn CdcHook>);
    let ctx = Arc::new(ctx);
    zyron_executor::execute(plan, &ctx).await.expect("execute");
    server.txn_manager.commit(&mut txn).await.expect("commit");
    let logs = zyron_lake::publish_txn(server.disk_manager.data_dir(), txn_id).expect("publish");
    zyron_wire::connection::refresh_lake_stats(server, &logs);
}

#[tokio::test]
async fn lake_dml_reaches_the_cdc_hook() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE cdc_lake (k BIGINT, v BIGINT) USING ZYRONLAKE",
    )
    .await
    .expect("create");
    let hook = Arc::new(CountingHook::default());

    exec_dml_hooked(
        &server,
        &hook,
        "INSERT INTO cdc_lake VALUES (1, 10), (2, 20), (3, 30)",
    )
    .await;
    assert_eq!(
        hook.inserts.load(Ordering::SeqCst),
        3,
        "every inserted row reaches the hook"
    );
    assert!(
        hook.insert_bytes
            .lock()
            .expect("lock")
            .iter()
            .all(|len| *len > 0),
        "the hook receives encoded row images"
    );

    exec_dml_hooked(&server, &hook, "UPDATE cdc_lake SET v = 99 WHERE k = 2").await;
    assert_eq!(
        hook.updates.load(Ordering::SeqCst),
        1,
        "the replaced row reaches the hook with its new image"
    );

    exec_dml_hooked(&server, &hook, "DELETE FROM cdc_lake WHERE k = 1").await;
    assert_eq!(
        hook.deletes.load(Ordering::SeqCst),
        1,
        "the removed row reaches the hook"
    );
}

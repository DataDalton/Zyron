//! A consume that dies at any point leaves the position and the target in
//! agreement, and the retry produces the target one clean run would.
//!
//! The consume is a transaction that reads a stream and writes what it read.
//! Its position advance is written into the transaction's own log chain
//! before the commit record and installed after it, so the points a process
//! can die at are before anything, after the write, after the advance was
//! logged, after the commit record, and after the install. The server is
//! dropped at that point and opened again over the same directories, which
//! recovers the log the way a restart does, and what stands afterwards is
//! checked against what the commit did or did not promise. A hundred
//! iterations draw the kill point at random.
//!
//! Run: cargo test -p zyron-wire --test change_stream_kill_test -- --nocapture

use std::sync::Arc;

use zyron_executor::column::ScalarValue;
use zyron_executor::context::ExecutionContext;
use zyron_storage::txn::IsolationLevel;
use zyron_wire::connection::ServerState;

mod common;
use common::*;

/// Where the process dies during one consume
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KillPoint {
    /// Before the read, so the transaction never touched the stream
    BeforeRead,
    /// After the read and the write, before anything reached the log
    AfterWrite,
    /// After the advance record was logged, before the commit record
    AfterAdvanceLogged,
    /// After the commit record, before the entry was installed in memory
    AfterCommit,
    /// After everything
    AfterInstall,
}

const KILL_POINTS: [KillPoint; 5] = [
    KillPoint::BeforeRead,
    KillPoint::AfterWrite,
    KillPoint::AfterAdvanceLogged,
    KillPoint::AfterCommit,
    KillPoint::AfterInstall,
];

/// Whether a consume killed at the point counts as committed afterwards
fn commits(point: KillPoint) -> bool {
    matches!(point, KillPoint::AfterCommit | KillPoint::AfterInstall)
}

/// A small deterministic generator, so a failing run names its sequence
struct Draw(u64);

impl Draw {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

async fn count(server: &Arc<ServerState>, sql: &str) -> i64 {
    let rows = query_values(server, sql).await;
    match rows.first().and_then(|r| r.first()) {
        Some(ScalarValue::Int64(n)) => *n,
        other => panic!("expected a count from {sql}, got {other:?}"),
    }
}

fn consumed(server: &Arc<ServerState>, stream: &str) -> u64 {
    let table_id = table_id_of(server, "orders");
    server
        .catalog
        .list_change_streams()
        .into_iter()
        .find(|e| e.name == stream)
        .expect("the stream exists")
        .consumed_of(table_id)
}

/// Runs one consume up to the kill point, then abandons whatever is in
/// hand without finishing it, the way a dying process does
async fn consume_until(server: &Arc<ServerState>, point: KillPoint) {
    if point == KillPoint::BeforeRead {
        return;
    }
    let stmt = zyron_parser::parse("INSERT INTO silver SELECT id, total FROM s")
        .expect("parse")
        .into_iter()
        .next()
        .expect("one statement");
    let plan = zyron_planner::plan(
        &server.catalog,
        zyron_catalog::DatabaseId(1),
        vec!["zyron_test".into()],
        stmt,
        None,
    )
    .await
    .expect("plan");
    let mut txn = server
        .txn_manager
        .begin(IsolationLevel::ReadCommitted)
        .expect("begin");
    let txn_id = txn.txn_id;
    let advances = Arc::new(parking_lot::Mutex::new(Vec::new()));
    let mut ctx = ExecutionContext::new(
        server.catalog.clone(),
        server.wal.clone(),
        server.buffer_pool.clone(),
        server.disk_manager.clone(),
        txn_id,
        server.txn_manager.refresh_snapshot(&txn),
    );
    ctx.row_locks = Some(Arc::clone(server.txn_manager.lock_table()));
    ctx.intent_locks = Some(Arc::clone(server.txn_manager.intent_locks()));
    ctx.heap_files = Some(Arc::clone(&server.heap_files));
    ctx.btree_indexes = Some(Arc::clone(&server.btree_indexes));
    ctx.doc_registry = Some(Arc::clone(&server.doc_registry));
    zyron_wire::change_feed_bridge::install_change_reads(server, &mut ctx, &advances);
    if let Some(hook) = server.cdc_hook.as_ref() {
        ctx.cdc_hook = Some(Arc::clone(hook));
    }
    let ctx = Arc::new(ctx);
    zyron_executor::execute(plan, &ctx)
        .await
        .expect("the consume runs");
    if ctx.wrote_wal() {
        txn.mark_wrote_data();
    }
    if point == KillPoint::AfterWrite {
        // Dropped undecided, so the transaction manager sees it end without
        // a commit record, the way a dead process leaves one
        std::mem::forget(txn);
        return;
    }
    let held = std::mem::take(&mut *advances.lock());
    let advanced =
        zyron_wire::change_stream_dispatch::log_stream_advances(server, &mut txn, &held, 0)
            .expect("the advance logs");
    if point == KillPoint::AfterAdvanceLogged {
        std::mem::forget(txn);
        return;
    }
    server.txn_manager.commit(&mut txn).await.expect("commit");
    if point == KillPoint::AfterCommit {
        return;
    }
    zyron_wire::change_stream_dispatch::install_stream_advances(server, txn_id, advanced)
        .await
        .expect("install");
}

/// Writes every dirty page the way the background writer would have by
/// the time a process dies some time after its last commit, so what the
/// reopened server reads of the heap is what the log promised. The log
/// reaches disk first, the way the writer's own hook orders it, so a page
/// never lands ahead of the records that describe it
fn settle_pages(server: &Arc<ServerState>) {
    let disk = Arc::clone(&server.disk_manager);
    server.wal.flush().expect("the log flushes");
    server
        .buffer_pool
        .flush_all(|page_id, data| {
            let page: &mut [u8; zyron_common::page::PAGE_SIZE] = data
                .try_into()
                .map_err(|_| zyron_common::ZyronError::Internal("a frame is one page".into()))?;
            disk.write_page_sync(page_id, page)?;
            Ok(zyron_buffer::FlushOutcome::Written)
        })
        .expect("the pages flush");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_consume_killed_anywhere_leaves_the_position_and_the_target_in_agreement() {
    for storage in Storage::BOTH {
        a_consume_killed_anywhere_leaves_the_position_and_the_target_in_agreement_on(storage).await;
    }
}

async fn a_consume_killed_anywhere_leaves_the_position_and_the_target_in_agreement_on(
    storage: Storage,
) {
    println!("{storage}");
    let (server, _schema, tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        &storage.create("CREATE TABLE orders (id BIGINT PRIMARY KEY, total BIGINT)"),
    )
    .await
    .expect("create orders");
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE orders SET (change_data_feed = true)",
    )
    .await
    .expect("turn the feed on");
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE silver (id BIGINT PRIMARY KEY, total BIGINT)",
    )
    .await
    .expect("create silver");
    exec_ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM s ON TABLE orders",
    )
    .await
    .expect("create the stream");

    let mut server = server;
    let mut tmp = tmp;
    let mut draw = Draw(0x9E37_79B9_7F4A_7C15);
    let iterations = 100;
    let mut kills_by_point = [0usize; 5];
    for i in 1..=iterations {
        // One new change per iteration, so what a consume takes is known
        exec_dml(
            &server,
            &format!("INSERT INTO orders VALUES ({i}, {})", i * 10),
        )
        .await;
        let before_position = consumed(&server, "s");
        let before_rows = count(&server, "SELECT COUNT(*) FROM silver").await;
        let at = (draw.next() % KILL_POINTS.len() as u64) as usize;
        let point = KILL_POINTS[at];
        kills_by_point[at] += 1;

        consume_until(&server, point).await;
        settle_pages(&server);

        // The process dies here. Every handle it held goes with it, and the
        // next one opens over what reached disk
        drop(session);
        drop(server);
        let reopened = reopen_test_server_with_cdc(tmp).await;
        server = reopened.0;
        tmp = reopened.2;
        session = new_session();

        let position = consumed(&server, "s");
        let rows = count(&server, "SELECT COUNT(*) FROM silver").await;
        if commits(point) {
            assert_eq!(
                position, i as u64,
                "iteration {i} at {point:?}: the commit moved the position"
            );
            assert_eq!(
                rows, i,
                "iteration {i} at {point:?}: the commit landed the rows"
            );
        } else {
            assert_eq!(
                position, before_position,
                "iteration {i} at {point:?}: a consume that never committed moved nothing"
            );
            assert_eq!(
                rows, before_rows,
                "iteration {i} at {point:?}: a consume that never committed wrote nothing"
            );
            // The retry takes exactly what the dead consume was holding
            consume_until(&server, KillPoint::AfterInstall).await;
            assert_eq!(
                consumed(&server, "s"),
                i as u64,
                "iteration {i}: the retry moved the position"
            );
            assert_eq!(
                count(&server, "SELECT COUNT(*) FROM silver").await,
                i,
                "iteration {i}: the retry landed the rows once"
            );
        }
        // Whatever the kill point, the target holds each change once
        assert_eq!(
            count(&server, "SELECT COUNT(DISTINCT id) FROM silver").await,
            i,
            "iteration {i}: no change was lost or repeated"
        );
    }
    println!("kills by point {kills_by_point:?}");
    assert!(
        kills_by_point.iter().all(|n| *n > 0),
        "every kill point was drawn: {kills_by_point:?}"
    );
    let _ = tmp;
}

/// One consume of a stream over two tables moves both positions in one
/// record, so a process dying anywhere leaves both moved or neither
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_multi_table_advance_is_one_record_whatever_the_kill_point() {
    for storage in Storage::BOTH {
        a_multi_table_advance_is_one_record_whatever_the_kill_point_on(storage).await;
    }
}

async fn a_multi_table_advance_is_one_record_whatever_the_kill_point_on(storage: Storage) {
    println!("{storage}");
    let (server, _schema, tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    for table in ["orders", "dims"] {
        exec_ddl(
            &server,
            &mut session,
            &storage.create(&format!(
                "CREATE TABLE {table} (id BIGINT PRIMARY KEY, total BIGINT)"
            )),
        )
        .await
        .expect("create the table");
        exec_ddl(
            &server,
            &mut session,
            &format!("ALTER TABLE {table} SET (change_data_feed = true)"),
        )
        .await
        .expect("turn the feed on");
    }
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE landing (id BIGINT, src BIGINT, total BIGINT)",
    )
    .await
    .expect("create the landing table");
    exec_ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM m ON TABLES (orders, dims)",
    )
    .await
    .expect("create the stream");

    let positions = |server: &Arc<ServerState>| -> (u64, u64) {
        let entry = server
            .catalog
            .list_change_streams()
            .into_iter()
            .find(|e| e.name == "m")
            .expect("the stream exists");
        (
            entry.consumed_of(table_id_of(server, "orders")),
            entry.consumed_of(table_id_of(server, "dims")),
        )
    };

    let mut server = server;
    let mut tmp = tmp;
    let mut draw = Draw(0x2545_F491_4F6C_DD1D);
    for i in 1..=40u64 {
        exec_dml(&server, &format!("INSERT INTO orders VALUES ({i}, {i})")).await;
        exec_dml(&server, &format!("INSERT INTO dims VALUES ({i}, {i})")).await;
        let before = positions(&server);
        let point = KILL_POINTS[(draw.next() % KILL_POINTS.len() as u64) as usize];

        // The same consume as above, over the two table stream
        let plan_sql = "INSERT INTO landing SELECT id, _source_table, total FROM m";
        if point != KillPoint::BeforeRead {
            let stmt = zyron_parser::parse(plan_sql)
                .expect("parse")
                .into_iter()
                .next()
                .expect("one statement");
            let plan = zyron_planner::plan(
                &server.catalog,
                zyron_catalog::DatabaseId(1),
                vec!["zyron_test".into()],
                stmt,
                None,
            )
            .await
            .expect("plan");
            let mut txn = server
                .txn_manager
                .begin(IsolationLevel::ReadCommitted)
                .expect("begin");
            let txn_id = txn.txn_id;
            let advances = Arc::new(parking_lot::Mutex::new(Vec::new()));
            let mut ctx = ExecutionContext::new(
                server.catalog.clone(),
                server.wal.clone(),
                server.buffer_pool.clone(),
                server.disk_manager.clone(),
                txn_id,
                server.txn_manager.refresh_snapshot(&txn),
            );
            ctx.row_locks = Some(Arc::clone(server.txn_manager.lock_table()));
            ctx.intent_locks = Some(Arc::clone(server.txn_manager.intent_locks()));
            ctx.heap_files = Some(Arc::clone(&server.heap_files));
            ctx.btree_indexes = Some(Arc::clone(&server.btree_indexes));
            ctx.doc_registry = Some(Arc::clone(&server.doc_registry));
            zyron_wire::change_feed_bridge::install_change_reads(&server, &mut ctx, &advances);
            if let Some(hook) = server.cdc_hook.as_ref() {
                ctx.cdc_hook = Some(Arc::clone(hook));
            }
            let ctx = Arc::new(ctx);
            zyron_executor::execute(plan, &ctx)
                .await
                .expect("the consume runs");
            if ctx.wrote_wal() {
                txn.mark_wrote_data();
            }
            if point == KillPoint::AfterWrite {
                std::mem::forget(txn);
            } else {
                let held = std::mem::take(&mut *advances.lock());
                let advanced = zyron_wire::change_stream_dispatch::log_stream_advances(
                    &server, &mut txn, &held, 0,
                )
                .expect("the advance logs");
                if point == KillPoint::AfterAdvanceLogged {
                    std::mem::forget(txn);
                } else {
                    server.txn_manager.commit(&mut txn).await.expect("commit");
                    if point == KillPoint::AfterInstall {
                        zyron_wire::change_stream_dispatch::install_stream_advances(
                            &server, txn_id, advanced,
                        )
                        .await
                        .expect("install");
                    }
                }
            }
        }
        settle_pages(&server);
        drop(session);
        drop(server);
        let reopened = reopen_test_server_with_cdc(tmp).await;
        server = reopened.0;
        tmp = reopened.2;
        session = new_session();

        let after = positions(&server);
        if commits(point) {
            assert_eq!(
                after,
                (i, i),
                "iteration {i} at {point:?}: both positions moved together"
            );
        } else {
            assert_eq!(
                after, before,
                "iteration {i} at {point:?}: neither position moved"
            );
            // The retry moves both
            let rows = query_values(&server, plan_sql).await;
            let _ = rows;
        }
        assert_eq!(
            positions(&server),
            (i, i),
            "iteration {i}: after the retry both positions stand together"
        );
        assert_eq!(
            count(&server, "SELECT COUNT(*) FROM landing").await,
            2 * i as i64,
            "iteration {i}: each change landed once"
        );
    }
    let _ = tmp;
}

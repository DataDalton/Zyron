//! A commit of a verified table lands with its chain entry or not at all.
//!
//! The entry is written into the transaction's own log chain ahead of its
//! commit record, so the points a process can die at are before anything,
//! after the rows are written, after the chain entry is logged, and after
//! the commit record. The server is dropped at that point and opened again
//! over the same directories, which recovers the log the way a restart does,
//! and what stands afterwards is checked against what the commit did or did
//! not promise. A hundred iterations draw the kill point at random.
//!
//! Run: cargo test -p zyron-wire --test verify_kill_test -- --nocapture

use std::sync::Arc;

use zyron_executor::context::ExecutionContext;
use zyron_lifecycle::verify::PendingChainWrites;
use zyron_storage::txn::IsolationLevel;
use zyron_wire::connection::ServerState;

mod common;
use common::*;

/// Where the process dies during one insert
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KillPoint {
    /// Before the statement ran, so nothing was written
    BeforeWrite,
    /// After the rows reached the heap, before anything else
    AfterWrite,
    /// After the chain entry was logged, before the commit record
    AfterChainLogged,
    /// After the commit record, which is the point the commit stands from
    AfterCommit,
}

const KILL_POINTS: [KillPoint; 4] = [
    KillPoint::BeforeWrite,
    KillPoint::AfterWrite,
    KillPoint::AfterChainLogged,
    KillPoint::AfterCommit,
];

/// Whether an insert killed at the point counts as committed afterwards
fn committed(point: KillPoint) -> bool {
    point == KillPoint::AfterCommit
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

fn chain_commits(server: &Arc<ServerState>) -> u64 {
    let table_id = table_id_of(server, "ledger");
    server
        .chain_registry
        .as_ref()
        .expect("the node holds chains")
        .chain(table_id)
        .expect("the chain opens")
        .head()
        .commits
}

async fn row_count(server: &Arc<ServerState>) -> i64 {
    let rows = query_values(server, "SELECT COUNT(*) FROM ledger").await;
    match rows.first().and_then(|r| r.first()) {
        Some(zyron_executor::column::ScalarValue::Int64(n)) => *n,
        other => panic!("expected a count, got {other:?}"),
    }
}

/// Runs one insert up to the kill point, then abandons whatever is in hand
/// without finishing it, the way a dying process does
async fn insert_until(server: &Arc<ServerState>, id: i64, point: KillPoint) {
    if point == KillPoint::BeforeWrite {
        return;
    }
    let sql = format!("INSERT INTO ledger VALUES ({id}, {})", id * 10);
    let stmt = zyron_parser::parse(&sql)
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
    let chain_writes = Arc::new(PendingChainWrites::new());
    zyron_wire::verify_dispatch::install_chain_writes(server, &mut ctx, &chain_writes);
    if let Some(hook) = server.cdc_hook.as_ref() {
        ctx.cdc_hook = Some(Arc::clone(hook));
    }
    let ctx = Arc::new(ctx);
    zyron_executor::execute(plan, &ctx)
        .await
        .expect("the insert runs");
    if ctx.wrote_wal() {
        txn.mark_wrote_data();
    }
    if point == KillPoint::AfterWrite {
        // Dropped undecided, so the transaction manager sees it end with no
        // commit record, the way a dead process leaves one
        std::mem::forget(txn);
        return;
    }
    let chained = zyron_wire::verify_dispatch::log_commit_chains(server, &mut txn, &chain_writes)
        .expect("the chain entry logs");
    if point == KillPoint::AfterChainLogged {
        std::mem::forget(txn);
        return;
    }
    server.txn_manager.commit(&mut txn).await.expect("commit");
    zyron_wire::verify_dispatch::publish_commit_chains(server, &chained);
}

/// Writes every dirty page the way the background writer would have by the
/// time a process dies some time after its last commit. The log reaches
/// disk first, so a page never lands ahead of the records describing it
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
async fn a_commit_killed_anywhere_lands_with_its_chain_entry_or_not_at_all() {
    let (server, _schema, tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE ledger (id BIGINT PRIMARY KEY, amount BIGINT)",
    )
    .await
    .expect("create ledger");
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE ledger SET (immutable = true, verified = true)",
    )
    .await
    .expect("verified");

    let mut server = server;
    let mut tmp = tmp;
    let mut draw = Draw(0x2545_F491_4F6C_DD1D);
    let iterations = 100;
    let mut kills_by_point = [0usize; 4];
    let mut landed = 0i64;

    for i in 1..=iterations {
        let before_rows = row_count(&server).await;
        let before_commits = chain_commits(&server);
        assert_eq!(
            before_rows as u64, before_commits,
            "iteration {i}: one chained commit per row so far"
        );

        let at = (draw.next() % KILL_POINTS.len() as u64) as usize;
        let point = KILL_POINTS[at];
        kills_by_point[at] += 1;

        insert_until(&server, i, point).await;
        settle_pages(&server);

        // The process dies here. Every handle it held goes with it, and the
        // next one opens over what reached disk
        drop(session);
        drop(server);
        let reopened = reopen_test_server_with_cdc(tmp).await;
        server = reopened.0;
        tmp = reopened.2;
        session = new_session();

        let rows = row_count(&server).await;
        let commits = chain_commits(&server);
        if committed(point) {
            landed += 1;
            assert_eq!(rows, landed, "iteration {i} at {point:?}: the rows landed");
            assert_eq!(
                commits, landed as u64,
                "iteration {i} at {point:?}: the chain entry landed with them"
            );
        } else {
            assert_eq!(
                rows, landed,
                "iteration {i} at {point:?}: a commit that never happened wrote no rows"
            );
            assert_eq!(
                commits, landed as u64,
                "iteration {i} at {point:?}: and chained nothing"
            );
        }
        // Whatever the kill point, the chain covers exactly the rows that
        // are there, which is the invariant the whole phase rests on
        assert_eq!(
            rows as u64, commits,
            "iteration {i} at {point:?}: the chain and the table agree"
        );
    }

    println!("kills by point {kills_by_point:?}");
    assert!(
        kills_by_point.iter().all(|n| *n > 0),
        "every kill point was drawn: {kills_by_point:?}"
    );

    // What stands at the end walks clean, so no partial commit left an
    // entry the rows do not support
    let table_id = table_id_of(&server, "ledger");
    let chain = server
        .chain_registry
        .as_ref()
        .expect("chains")
        .chain(table_id)
        .expect("chain");
    let head = chain.head();
    assert_eq!(head.commits, landed as u64);
    if head.commits > 0 {
        let entries = chain.read_range(0, head.commits - 1).expect("reads");
        let mut prev = zyron_lifecycle::verify::NO_PREVIOUS;
        for entry in &entries {
            assert_eq!(entry.prev_hash, prev, "entry {} links", entry.sequence);
            assert_eq!(entry.compute_entry_hash(), entry.entry_hash);
            assert_eq!(entry.row_count, 1);
            prev = entry.entry_hash;
        }
        assert_eq!(prev, head.head_hash);
    }
    let _ = tmp;
}

//! Verifiable tables through SQL.
//!
//! What a person types and what comes back: turning a chain on, what is
//! refused, what a genesis entry covers, what VERIFY TABLE reports over an
//! untouched table, and what the views say about a chain.
//!
//! Run: cargo test -p zyron-wire --test verify_dispatch_test -- --nocapture

use std::sync::Arc;

use zyron_wire::connection::ServerState;

mod common;
use common::*;

/// The row `VERIFY TABLE` answers with, as the columns it declares
struct VerifyRow {
    commits_checked: i64,
    rows_checked: i64,
    anchors_checked: i64,
    mode: String,
    intact: bool,
    detail: String,
}

/// Runs one statement through the dispatcher and hands back the rows it
/// answered with, the way a connection reads a result set
async fn ddl_rows(
    server: &Arc<ServerState>,
    session: &mut Option<zyron_wire::session::Session>,
    sql: &str,
) -> Result<Vec<Vec<String>>, String> {
    let stmt = zyron_parser::parse(sql)
        .expect("parse")
        .into_iter()
        .next()
        .expect("one statement");
    let mut txn_opt: Option<zyron_storage::txn::Transaction> = None;
    let mut branch: Option<String> = None;
    match zyron_wire::ddl_dispatch::try_handle_ddl_utility(
        &stmt,
        server,
        session,
        &mut txn_opt,
        &mut branch,
        sql,
    )
    .await
    {
        Some(Ok(zyron_wire::ddl_dispatch::DdlResult::Rows { rows, .. })) => Ok(rows),
        Some(Ok(other)) => Err(format!("expected rows, got {other:?}")),
        Some(Err(e)) => Err(format!("{e:?}")),
        None => Err(format!("statement was not handled: {sql}")),
    }
}

async fn verify(server: &Arc<ServerState>, sql: &str) -> VerifyRow {
    let mut session = new_session();
    let rows = ddl_rows(server, &mut session, sql)
        .await
        .expect("VERIFY TABLE answers");
    let row = rows.first().expect("VERIFY TABLE answers with one row");
    VerifyRow {
        commits_checked: row[1].parse().expect("a count"),
        rows_checked: row[2].parse().expect("a count"),
        anchors_checked: row[3].parse().expect("a count"),
        mode: row[4].clone(),
        intact: row[5] == "true",
        detail: row[6].clone(),
    }
}

/// One system view, rows keyed by column name, cells read as text
async fn view(server: &Arc<ServerState>, name: &str) -> (Vec<String>, Vec<Vec<String>>) {
    let (fields, rows) = zyron_wire::system_views::query_system_view(
        name,
        server,
        &zyron_wire::system_views::SystemViewFilters::default(),
    )
    .await
    .expect("the view answers")
    .unwrap_or_else(|| panic!("{name} is not a system view"));
    let columns: Vec<String> = fields.iter().map(|f| f.name.clone()).collect();
    let rows = rows
        .into_iter()
        .map(|cells| {
            cells
                .into_iter()
                .map(|cell| {
                    cell.map(|b| String::from_utf8_lossy(&b).into_owned())
                        .unwrap_or_default()
                })
                .collect()
        })
        .collect();
    (columns, rows)
}

fn cell<'a>(columns: &[String], row: &'a [String], name: &str) -> &'a str {
    let at = columns
        .iter()
        .position(|c| c == name)
        .unwrap_or_else(|| panic!("no column {name} in {columns:?}"));
    &row[at]
}

fn chain_commits(server: &Arc<ServerState>, table: &str) -> u64 {
    let table_id = table_id_of(server, table);
    server
        .chain_registry
        .as_ref()
        .expect("the node holds chains")
        .chain(table_id)
        .expect("the chain opens")
        .head()
        .commits
}

/// A chain over rows that DELETE, UPDATE, TRUNCATE or a row-rewriting schema
/// change can still alter states nothing about them, so the combination is
/// refused naming all four
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn verified_without_immutable_is_refused_naming_the_reason() {
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE ledger (id BIGINT PRIMARY KEY, amount BIGINT)",
    )
    .await
    .expect("create ledger");

    let refused = exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE ledger SET (verified = true)",
    )
    .await
    .err()
    .expect("refused");
    let text = refused.to_string();
    assert!(
        text.contains("DELETE, UPDATE, TRUNCATE and a row-rewriting"),
        "{text}"
    );
    assert!(text.contains("immutable = true"), "{text}");

    // With both in one statement it is accepted
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE ledger SET (immutable = true, verified = true)",
    )
    .await
    .expect("both together");
    let entry = server
        .catalog
        .get_table_by_id(zyron_catalog::TableId(table_id_of(&server, "ledger")))
        .expect("the table");
    assert!(entry.lifecycle.verified);
    assert!(entry.lifecycle.immutable);
    assert_eq!(
        entry.lifecycle.chain_algorithm,
        zyron_lifecycle::verify::DEFAULT_CHAIN_ALGORITHM
    );
}

/// A chain that can be switched off is not evidence, so turning
/// verification off is refused once the chain holds a commit
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn verification_cannot_be_turned_off_once_a_commit_is_chained() {
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
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

    // An empty chain can still be turned off, because nothing was chained
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE ledger SET (verified = false)",
    )
    .await
    .expect("nothing chained yet");
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE ledger SET (immutable = true, verified = true)",
    )
    .await
    .expect("verified again");

    exec_dml(&server, "INSERT INTO ledger VALUES (1, 100)").await;
    assert_eq!(chain_commits(&server, "ledger"), 1);

    let refused = exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE ledger SET (verified = false)",
    )
    .await
    .err()
    .expect("refused");
    let text = refused.to_string();
    assert!(text.contains("not evidence"), "{text}");
    assert!(text.contains("stands as long as the table does"), "{text}");
}

/// Enabling on a populated table covers the rows already there with a
/// genesis entry, and the view states that they are covered as a set
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_populated_table_gets_a_genesis_entry_covering_what_it_held() {
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE ledger (id BIGINT PRIMARY KEY, amount BIGINT)",
    )
    .await
    .expect("create ledger");
    for id in 1..=7 {
        exec_dml(
            &server,
            &format!("INSERT INTO ledger VALUES ({id}, {})", id * 10),
        )
        .await;
    }
    assert_eq!(chain_commits(&server, "ledger"), 0);

    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE ledger SET (immutable = true, verified = true)",
    )
    .await
    .expect("verified");

    assert_eq!(
        chain_commits(&server, "ledger"),
        1,
        "the genesis entry covers what was already there"
    );
    let table_id = table_id_of(&server, "ledger");
    let chain = server
        .chain_registry
        .as_ref()
        .expect("chains")
        .chain(table_id)
        .expect("chain");
    let genesis = chain.read_range(0, 0).expect("reads").remove(0);
    assert_eq!(genesis.row_count, 7);
    assert_eq!(genesis.prev_hash, zyron_lifecycle::verify::NO_PREVIOUS);
    assert!(genesis.genesis, "the entry is flagged as the genesis");
    let registry = server.chain_registry.as_ref().expect("chains");
    let chained = registry.chained(table_id).expect("the table is chained");
    assert_eq!(chained.genesis_sequence(), Some(0));
    assert!(
        genesis.txn_id >= chained.fence && chained.fence > 0,
        "the genesis names the fence the table was chained at"
    );
    let placed = server
        .catalog
        .get_table_by_id(zyron_catalog::TableId(table_id))
        .expect("the table");
    assert_eq!(
        placed.lifecycle.genesis_at, 1,
        "the catalog records where it landed"
    );

    let (columns, rows) = view(&server, "zyron_sys.verify.tables").await;
    let row = rows
        .iter()
        .find(|row| cell(&columns, row, "table") == "ledger")
        .expect("the ledger is listed");
    assert_eq!(cell(&columns, row, "commits"), "1");
    let covers = cell(&columns, row, "covers");
    assert!(covers.contains("7 row(s)"), "{covers}");
    assert!(covers.contains("covered as a set"), "{covers}");

    // The set is read back under a full pass and matches, and a commit
    // after it is covered on its own
    exec_dml(&server, "INSERT INTO ledger VALUES (8, 80)").await;
    let full = verify(&server, "VERIFY TABLE ledger WITH (rows => 'all')").await;
    assert!(full.intact, "{}", full.detail);
    assert_eq!(full.commits_checked, 2);
    assert_eq!(full.rows_checked, 8, "seven in the set and one after it");
    // A sampled pass checks the genesis entry's link and says it did not
    // read the set back
    let sampled = verify(&server, "VERIFY TABLE ledger").await;
    assert!(sampled.intact, "{}", sampled.detail);
    assert_eq!(sampled.rows_checked, 1);
    assert!(
        sampled.detail.contains("the genesis set not read"),
        "{}",
        sampled.detail
    );
}

/// An untouched table verifies, and the result states the mode it ran in
/// and how many commits, rows and anchors it checked
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_untouched_table_verifies_and_states_its_mode() {
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
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
    for id in 1..=5 {
        exec_dml(
            &server,
            &format!("INSERT INTO ledger VALUES ({id}, {})", id * 10),
        )
        .await;
    }
    assert_eq!(chain_commits(&server, "ledger"), 5);

    let sampled = verify(&server, "VERIFY TABLE ledger").await;
    assert!(sampled.intact, "{}", sampled.detail);
    assert_eq!(sampled.commits_checked, 5);
    assert_eq!(sampled.mode, "sampled");
    assert!(sampled.detail.contains("sampled"), "{}", sampled.detail);

    let full = verify(&server, "VERIFY TABLE ledger WITH (rows => 'all')").await;
    assert!(full.intact, "{}", full.detail);
    assert_eq!(full.commits_checked, 5);
    assert_eq!(full.rows_checked, 5, "one row per commit");
    assert_eq!(full.mode, "all");
    assert!(
        full.detail.contains("every one of their 5 row(s) rehashed"),
        "{}",
        full.detail
    );
    assert_eq!(sampled.anchors_checked, 0, "nothing anchored yet");
}

/// A table with no chain has nothing to walk, and the refusal says what to
/// do about it rather than answering with an empty pass
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn verifying_a_table_with_no_chain_is_refused() {
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE plain (id BIGINT PRIMARY KEY)",
    )
    .await
    .expect("create plain");
    let refused = ddl_rows(&server, &mut session, "VERIFY TABLE plain")
        .await
        .err()
        .expect("refused");
    assert!(refused.contains("is not verified"), "{refused}");
    assert!(refused.contains("commit chain"), "{refused}");
}

/// Every run is on the record, with who asked for it, the mode it ran in
/// and what it found
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn every_run_is_recorded_with_its_actor_and_mode() {
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
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
    exec_dml(&server, "INSERT INTO ledger VALUES (1, 10)").await;

    let _ = verify(&server, "VERIFY TABLE ledger").await;
    let _ = verify(&server, "VERIFY TABLE ledger WITH (rows => 'all')").await;

    let (columns, runs) = view(&server, "zyron_sys.verify.runs").await;
    assert_eq!(runs.len(), 2, "both runs are on the record");
    // Newest first
    assert_eq!(cell(&columns, &runs[0], "mode"), "all");
    assert_eq!(cell(&columns, &runs[1], "mode"), "sampled");
    for run in &runs {
        assert_eq!(cell(&columns, run, "table"), "ledger");
        assert_eq!(cell(&columns, run, "intact"), "t");
        assert_eq!(cell(&columns, run, "commits_checked"), "1");
        assert!(!cell(&columns, run, "actor").is_empty());
    }
}

/// A range narrows what the walk covers, and a range the chain does not
/// reach covers nothing rather than everything
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_version_range_narrows_the_walk() {
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
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
    for id in 1..=6 {
        exec_dml(&server, &format!("INSERT INTO ledger VALUES ({id}, {id})")).await;
    }

    let table_id = table_id_of(&server, "ledger");
    let chain = server
        .chain_registry
        .as_ref()
        .expect("chains")
        .chain(table_id)
        .expect("chain");
    let entries = chain.read_range(0, 5).expect("reads");
    let third = entries[2].commit_version;
    let fifth = entries[4].commit_version;

    let whole = verify(&server, "VERIFY TABLE ledger WITH (rows => 'all')").await;
    assert_eq!(whole.commits_checked, 6);

    let narrowed = verify(
        &server,
        &format!(
            "VERIFY TABLE ledger FROM VERSION {third} TO VERSION {fifth} WITH (rows => 'all')"
        ),
    )
    .await;
    assert!(narrowed.intact, "{}", narrowed.detail);
    assert_eq!(narrowed.commits_checked, 3);
}

/// The chain a table carries costs one fixed record per commit, which is
/// what the view reports and what the documentation states
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_chain_costs_one_fixed_record_per_commit() {
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
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
    for id in 1..=20 {
        exec_dml(&server, &format!("INSERT INTO ledger VALUES ({id}, {id})")).await;
    }

    let (columns, rows) = view(&server, "zyron_sys.verify.tables").await;
    let row = rows.first().expect("one verified table");
    let commits: i64 = cell(&columns, row, "commits").parse().expect("a count");
    let bytes: i64 = cell(&columns, row, "chain_bytes").parse().expect("a count");
    assert_eq!(commits, 20);
    assert_eq!(
        bytes,
        commits * zyron_lifecycle::verify::CHAIN_RECORD_LEN as i64
    );
    assert_eq!(zyron_lifecycle::verify::CHAIN_RECORD_LEN, 96);
}

/// An immutable table refuses UPDATE and DELETE, so the only writes a chain
/// ever covers are the rows a commit added
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_verified_table_takes_inserts_and_refuses_the_rest() {
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
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
    exec_dml(&server, "INSERT INTO ledger VALUES (1, 10)").await;

    let update = query_error(&server, "UPDATE ledger SET amount = 20 WHERE id = 1").await;
    assert!(update.to_lowercase().contains("immutable"), "{update}");
    let delete = query_error(&server, "DELETE FROM ledger WHERE id = 1").await;
    assert!(delete.to_lowercase().contains("immutable"), "{delete}");

    let outcome = verify(&server, "VERIFY TABLE ledger WITH (rows => 'all')").await;
    assert!(outcome.intact, "{}", outcome.detail);
    assert_eq!(outcome.commits_checked, 1);
}

/// Runs one statement inside an open transaction the way a connection does,
/// hashing what it writes into `chain_writes`, and leaves the transaction
/// open for the caller to commit
async fn run_in_txn(
    server: &Arc<ServerState>,
    txn: &zyron_storage::txn::Transaction,
    chain_writes: &Arc<zyron_lifecycle::verify::PendingChainWrites>,
    sql: &str,
) -> Result<(), zyron_common::ZyronError> {
    let stmt = zyron_parser::parse(sql)
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
    let mut ctx = zyron_executor::context::ExecutionContext::new(
        server.catalog.clone(),
        server.wal.clone(),
        server.buffer_pool.clone(),
        server.disk_manager.clone(),
        txn.txn_id,
        server.txn_manager.refresh_snapshot(txn),
    );
    ctx.row_locks = Some(Arc::clone(server.txn_manager.lock_table()));
    ctx.intent_locks = Some(Arc::clone(server.txn_manager.intent_locks()));
    ctx.heap_files = Some(Arc::clone(&server.heap_files));
    ctx.btree_indexes = Some(Arc::clone(&server.btree_indexes));
    ctx.doc_registry = Some(Arc::clone(&server.doc_registry));
    zyron_wire::verify_dispatch::install_chain_writes(server, &mut ctx, chain_writes);
    if let Some(hook) = server.cdc_hook.as_ref() {
        ctx.cdc_hook = Some(Arc::clone(hook));
    }
    if let Some(hook) = server.dml_hook.as_ref() {
        ctx.dml_hook = Some(Arc::clone(hook));
    }
    let ctx = Arc::new(ctx);
    zyron_executor::execute(plan, &ctx).await.map(|_| ())
}

/// A transaction of several statements is one commit and one entry, and the
/// rows it wrote read back in the order they were hashed whichever thread
/// each statement ran on, because the rows go through the transaction's
/// own cursor
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_transaction_of_several_statements_is_one_entry_that_verifies() {
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
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

    let statements: Vec<String> = (1..=40)
        .map(|id| format!("INSERT INTO ledger VALUES ({id}, {})", id * 10))
        .collect();
    let refs: Vec<&str> = statements.iter().map(String::as_str).collect();
    exec_dml_script(&server, &refs)
        .await
        .expect("the script commits");
    assert_eq!(chain_commits(&server, "ledger"), 1, "one commit, one entry");

    let outcome = verify(&server, "VERIFY TABLE ledger WITH (rows => 'all')").await;
    assert!(outcome.intact, "{}", outcome.detail);
    assert_eq!(outcome.commits_checked, 1);
    assert_eq!(outcome.rows_checked, 40);
}

/// A transaction that wrote to a table before it became verifiable is
/// refused at commit, naming why, and leaves nothing behind: its rows are
/// covered by neither the genesis set nor an entry of their own
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_transaction_from_before_the_table_was_verified_is_refused_at_commit() {
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE ledger (id BIGINT PRIMARY KEY, amount BIGINT)",
    )
    .await
    .expect("create ledger");
    exec_dml(&server, "INSERT INTO ledger VALUES (1, 10)").await;

    // A transaction writes to the table and stays open across the change
    let mut txn = server
        .txn_manager
        .begin(zyron_storage::IsolationLevel::ReadCommitted)
        .expect("begin");
    let chain_writes = Arc::new(zyron_lifecycle::verify::PendingChainWrites::new());
    run_in_txn(
        &server,
        &txn,
        &chain_writes,
        "INSERT INTO ledger VALUES (2, 20)",
    )
    .await
    .expect("the insert runs");
    txn.mark_wrote_data();

    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE ledger SET (immutable = true, verified = true)",
    )
    .await
    .expect("verified while a writer is open");
    assert_eq!(
        chain_commits(&server, "ledger"),
        1,
        "the genesis covers the row that was committed"
    );

    let refused = zyron_wire::verify_dispatch::log_commit_chains(&server, &mut txn, &chain_writes)
        .expect_err("the straddling transaction is refused");
    assert!(
        refused.to_string().contains("became verifiable"),
        "{refused}"
    );
    server.txn_manager.abort(&mut txn).expect("rolled back");

    // The table holds the one row the genesis covers, and verifies whole
    assert_eq!(query_rows(&server, "SELECT * FROM ledger").await, 1);
    let outcome = verify(&server, "VERIFY TABLE ledger WITH (rows => 'all')").await;
    assert!(outcome.intact, "{}", outcome.detail);
    assert_eq!(outcome.rows_checked, 1);

    // A transaction started after the change is chained on its own
    exec_dml(&server, "INSERT INTO ledger VALUES (3, 30)").await;
    assert_eq!(chain_commits(&server, "ledger"), 2);
    let outcome = verify(&server, "VERIFY TABLE ledger WITH (rows => 'all')").await;
    assert!(outcome.intact, "{}", outcome.detail);
    assert_eq!(outcome.rows_checked, 2);
}

/// A transaction that removed rows from a table before it became verifiable
/// is refused at commit too, because the rows it removed were counted into
/// the genesis set
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_transaction_that_removed_rows_before_the_change_is_refused_at_commit() {
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE ledger (id BIGINT PRIMARY KEY, amount BIGINT)",
    )
    .await
    .expect("create ledger");
    exec_dml(&server, "INSERT INTO ledger VALUES (1, 10)").await;
    exec_dml(&server, "INSERT INTO ledger VALUES (2, 20)").await;

    let mut txn = server
        .txn_manager
        .begin(zyron_storage::IsolationLevel::ReadCommitted)
        .expect("begin");
    let chain_writes = Arc::new(zyron_lifecycle::verify::PendingChainWrites::new());
    run_in_txn(
        &server,
        &txn,
        &chain_writes,
        "DELETE FROM ledger WHERE id = 2",
    )
    .await
    .expect("the delete runs");
    txn.mark_wrote_data();

    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE ledger SET (immutable = true, verified = true)",
    )
    .await
    .expect("verified while a deleter is open");

    let refused = zyron_wire::verify_dispatch::log_commit_chains(&server, &mut txn, &chain_writes)
        .expect_err("the deleting transaction is refused");
    assert!(refused.to_string().contains("removed rows"), "{refused}");
    server.txn_manager.abort(&mut txn).expect("rolled back");

    assert_eq!(query_rows(&server, "SELECT * FROM ledger").await, 2);
    let outcome = verify(&server, "VERIFY TABLE ledger WITH (rows => 'all')").await;
    assert!(outcome.intact, "{}", outcome.detail);
    assert_eq!(outcome.rows_checked, 2, "both rows are in the genesis set");
}

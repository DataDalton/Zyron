//! Temporary tables, through the engine.
//!
//! A temporary table lives in the session that created it and on the node
//! that session is connected to. These tests hold that line from both sides:
//! what one session can do with one, and what nothing else can see of it.
//!
//! Run: cargo test -p zyron-wire --test temp_table_test -- --nocapture

mod common;

use std::sync::Arc;

use common::create_test_server;
use zyron_catalog::DatabaseId;
use zyron_executor::batch::DataBatch;
use zyron_executor::column::ScalarValue;
use zyron_executor::context::ExecutionContext;
use zyron_storage::txn::IsolationLevel;
use zyron_wire::connection::ServerState;
use zyron_wire::session::Session;

/// A session of its own, so two of them can be told apart by process id the
/// way two connections are.
fn session_with_id(process_id: i32) -> Option<Session> {
    let mut s = Session::new("test_user".into(), "testdb".into(), DatabaseId(1));
    s.search_path = vec!["zyron_test".into()];
    s.process_id = process_id;
    Some(s)
}

/// Runs a statement for one session, taking the DDL path first the way a
/// connection does, then the planner and the executor.
async fn run(
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
    sql: &str,
) -> std::result::Result<Vec<DataBatch>, String> {
    let stmt = zyron_parser::parse(sql)
        .map_err(|e| e.to_string())?
        .into_iter()
        .next()
        .ok_or_else(|| "no statement".to_string())?;

    let mut txn_opt: Option<zyron_storage::txn::Transaction> = None;
    let mut active_branch: Option<String> = None;
    if let Some(res) = zyron_wire::ddl_dispatch::try_handle_ddl_utility(
        &stmt,
        server,
        session,
        &mut txn_opt,
        &mut active_branch,
        sql,
    )
    .await
    {
        return res.map(|_| Vec::new()).map_err(|e| format!("{e:?}"));
    }

    zyron_wire::temp_table_dispatch::refresh_statistics(server, session).await;
    let temp_written = zyron_wire::temp_table_dispatch::write_target(&stmt)
        .map(|(name, kind)| (name.to_string(), kind));
    let temp_tables = session.as_ref().and_then(|s| s.temp_tables.clone());
    let plan = zyron_planner::plan_for_session(
        &server.catalog,
        DatabaseId(1),
        vec!["zyron_test".into()],
        stmt,
        None,
        None,
        temp_tables,
    )
    .await
    .map_err(|e| e.to_string())?;

    let mut txn = server
        .txn_manager
        .begin(IsolationLevel::ReadCommitted)
        .map_err(|e| e.to_string())?;
    let snapshot = txn.snapshot.clone();
    let txn_id = txn.txn_id;
    let mut ctx = ExecutionContext::new(
        server.catalog.clone(),
        server.wal.clone(),
        server.buffer_pool.clone(),
        server.disk_manager.clone(),
        txn_id,
        snapshot,
    );
    ctx.heap_files = Some(Arc::clone(&server.heap_files));
    ctx.btree_indexes = Some(Arc::clone(&server.btree_indexes));
    ctx.intent_locks = Some(Arc::clone(server.txn_manager.intent_locks()));
    ctx.row_locks = Some(Arc::clone(server.txn_manager.lock_table()));
    let ctx = Arc::new(ctx);
    match zyron_executor::execute(plan, &ctx).await {
        Ok(batches) => {
            server
                .txn_manager
                .commit(&mut txn)
                .await
                .map_err(|e| e.to_string())?;
            if let Some((name, kind)) = &temp_written {
                zyron_wire::temp_table_dispatch::mark_written(session, name, *kind, &batches);
            }
            zyron_wire::temp_table_dispatch::on_commit(server, session).await;
            Ok(batches)
        }
        Err(e) => {
            let _ = server.txn_manager.abort(&mut txn);
            Err(e.to_string())
        }
    }
}

async fn ok(server: &Arc<ServerState>, session: &mut Option<Session>, sql: &str) {
    run(server, session, sql)
        .await
        .unwrap_or_else(|e| panic!("statement failed: {sql}\n{e}"));
}

async fn err(server: &Arc<ServerState>, session: &mut Option<Session>, sql: &str) -> String {
    match run(server, session, sql).await {
        Ok(_) => panic!("`{sql}` should have been refused"),
        Err(e) => e,
    }
}

fn ints(batches: &[DataBatch], column: usize) -> Vec<i64> {
    let mut out = Vec::new();
    for b in batches {
        if let Some(col) = b.columns.get(column) {
            for r in 0..b.num_rows {
                out.push(match col.get_scalar(r) {
                    ScalarValue::Int64(v) => v,
                    ScalarValue::Int32(v) => v as i64,
                    ScalarValue::Null => i64::MIN,
                    other => panic!("expected an integer, got {other:?}"),
                });
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// One session's own table
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn create_insert_and_select_work_in_one_session() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = session_with_id(101);
    ok(&server, &mut session, "CREATE TEMP TABLE scratch (a INT)").await;
    ok(
        &server,
        &mut session,
        "INSERT INTO scratch VALUES (1), (2), (3)",
    )
    .await;
    let rows = run(&server, &mut session, "SELECT a FROM scratch ORDER BY a")
        .await
        .expect("select");
    assert_eq!(ints(&rows, 0), vec![1, 2, 3]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_second_session_sees_nothing_and_may_create_its_own_of_the_same_name() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut one = session_with_id(201);
    let mut two = session_with_id(202);
    ok(&server, &mut one, "CREATE TEMP TABLE scratch (a INT)").await;
    ok(&server, &mut one, "INSERT INTO scratch VALUES (7)").await;

    let missing = err(&server, &mut two, "SELECT a FROM scratch").await;
    assert!(
        missing.contains("scratch"),
        "the other session should not resolve it, got {missing}"
    );

    // The same bare name is free in the other session
    ok(&server, &mut two, "CREATE TEMP TABLE scratch (a INT)").await;
    ok(&server, &mut two, "INSERT INTO scratch VALUES (9)").await;
    let theirs = run(&server, &mut two, "SELECT a FROM scratch")
        .await
        .expect("select");
    assert_eq!(ints(&theirs, 0), vec![9], "each session reads its own");
    let ours = run(&server, &mut one, "SELECT a FROM scratch")
        .await
        .expect("select");
    assert_eq!(ints(&ours, 0), vec![7]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_temporary_table_shadows_a_permanent_one_and_the_qualified_name_still_reaches_it() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = session_with_id(301);
    ok(&server, &mut session, "CREATE TABLE shared (a INT)").await;
    ok(&server, &mut session, "INSERT INTO shared VALUES (100)").await;
    ok(&server, &mut session, "CREATE TEMP TABLE shared (a INT)").await;
    ok(&server, &mut session, "INSERT INTO shared VALUES (1)").await;

    let bare = run(&server, &mut session, "SELECT a FROM shared")
        .await
        .expect("select");
    assert_eq!(
        ints(&bare, 0),
        vec![1],
        "a bare name reads the temporary one"
    );

    let qualified = run(&server, &mut session, "SELECT a FROM zyron_test.shared")
        .await
        .expect("select");
    assert_eq!(
        ints(&qualified, 0),
        vec![100],
        "a qualified name always reaches the permanent one"
    );

    // The other session sees only the permanent table under the bare name
    let mut other = session_with_id(302);
    let theirs = run(&server, &mut other, "SELECT a FROM shared")
        .await
        .expect("select");
    assert_eq!(ints(&theirs, 0), vec![100]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn temp_and_temporary_and_or_replace_all_work() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = session_with_id(401);
    ok(&server, &mut session, "CREATE TEMPORARY TABLE a (x INT)").await;
    ok(&server, &mut session, "CREATE TEMP TABLE b (x INT)").await;
    // A second CREATE of the same name is refused without OR REPLACE
    let clash = err(&server, &mut session, "CREATE TEMP TABLE a (x INT)").await;
    assert!(
        clash.contains('a'),
        "the clash names the table, got {clash}"
    );
    ok(
        &server,
        &mut session,
        "CREATE OR REPLACE TEMP TABLE a (y INT)",
    )
    .await;
    ok(&server, &mut session, "INSERT INTO a VALUES (5)").await;
    let rows = run(&server, &mut session, "SELECT y FROM a")
        .await
        .expect("the replacement's own column is what reads");
    assert_eq!(ints(&rows, 0), vec![5]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn create_temp_as_select_and_select_into_temp_both_fill_the_table() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = session_with_id(501);
    ok(&server, &mut session, "CREATE TABLE src (a INT, b INT)").await;
    ok(
        &server,
        &mut session,
        "INSERT INTO src VALUES (1, 10), (2, 20)",
    )
    .await;

    ok(
        &server,
        &mut session,
        "CREATE TEMP TABLE copied AS SELECT a, b FROM src",
    )
    .await;
    let rows = run(&server, &mut session, "SELECT a FROM copied ORDER BY a")
        .await
        .expect("select");
    assert_eq!(ints(&rows, 0), vec![1, 2]);

    ok(
        &server,
        &mut session,
        "SELECT a INTO TEMP just_a FROM src WHERE a = 2",
    )
    .await;
    let rows = run(&server, &mut session, "SELECT a FROM just_a")
        .await
        .expect("select");
    assert_eq!(ints(&rows, 0), vec![2]);
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn on_commit_delete_rows_empties_the_table_and_on_commit_drop_removes_it() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = session_with_id(601);
    ok(
        &server,
        &mut session,
        "CREATE TEMP TABLE emptied (a INT) ON COMMIT DELETE ROWS",
    )
    .await;
    ok(&server, &mut session, "INSERT INTO emptied VALUES (1), (2)").await;
    // The insert's own commit empties it, so the next read finds nothing and
    // the definition is still there
    let rows = run(&server, &mut session, "SELECT a FROM emptied")
        .await
        .expect("the table still exists");
    assert!(rows.iter().all(|b| b.num_rows == 0), "the rows are gone");

    ok(
        &server,
        &mut session,
        "CREATE TEMP TABLE fleeting (a INT) ON COMMIT DROP",
    )
    .await;
    ok(&server, &mut session, "INSERT INTO fleeting VALUES (1)").await;
    let gone = err(&server, &mut session, "SELECT a FROM fleeting").await;
    assert!(
        gone.contains("fleeting"),
        "the first commit drops it, got {gone}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_rolled_back_insert_is_invisible() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = session_with_id(701);
    ok(&server, &mut session, "CREATE TEMP TABLE scratch (a INT)").await;
    ok(&server, &mut session, "INSERT INTO scratch VALUES (1)").await;

    // A write that aborts leaves nothing behind, because a temporary table's
    // rows carry the session's own transaction ids like any heap table's
    let stmt = zyron_parser::parse("INSERT INTO scratch VALUES (99)")
        .expect("parse")
        .into_iter()
        .next()
        .expect("one statement");
    let temp_tables = session.as_ref().and_then(|s| s.temp_tables.clone());
    let plan = zyron_planner::plan_for_session(
        &server.catalog,
        DatabaseId(1),
        vec!["zyron_test".into()],
        stmt,
        None,
        None,
        temp_tables,
    )
    .await
    .expect("plan");
    let mut txn = server
        .txn_manager
        .begin(IsolationLevel::ReadCommitted)
        .expect("begin");
    let snapshot = txn.snapshot.clone();
    let txn_id = txn.txn_id;
    let mut ctx = ExecutionContext::new(
        server.catalog.clone(),
        server.wal.clone(),
        server.buffer_pool.clone(),
        server.disk_manager.clone(),
        txn_id,
        snapshot,
    );
    ctx.heap_files = Some(Arc::clone(&server.heap_files));
    ctx.intent_locks = Some(Arc::clone(server.txn_manager.intent_locks()));
    let ctx = Arc::new(ctx);
    zyron_executor::execute(plan, &ctx).await.expect("insert");
    server.txn_manager.abort(&mut txn).expect("rollback");

    let rows = run(&server, &mut session, "SELECT a FROM scratch")
        .await
        .expect("select");
    assert_eq!(
        ints(&rows, 0),
        vec![1],
        "the rolled back row is not visible after ROLLBACK"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ending_the_session_removes_the_table_and_its_directory() {
    let (server, _schema, tmp) = create_test_server().await;
    let mut session = session_with_id(801);
    ok(&server, &mut session, "CREATE TEMP TABLE scratch (a INT)").await;
    ok(&server, &mut session, "INSERT INTO scratch VALUES (1)").await;

    let directory = session
        .as_ref()
        .and_then(|s| s.temp_tables.as_ref())
        .map(|t| t.directory().to_path_buf())
        .expect("the session has a directory");
    assert!(directory.exists(), "the files are on disk while it lives");

    let key = session.as_ref().expect("session").session_key;
    zyron_wire::temp_table_dispatch::end_session(&server, key).await;
    assert!(
        !directory.exists(),
        "the session's directory goes with the session"
    );
    let root = zyron_catalog::temp_root(tmp.path());
    if root.exists() {
        let left: Vec<_> = std::fs::read_dir(&root)
            .expect("read")
            .filter_map(|e| e.ok())
            .collect();
        assert!(left.is_empty(), "nothing of the session is left under tmp");
    }

    // A fresh session of the same id starts with nothing
    let mut again = session_with_id(801);
    let gone = err(&server, &mut again, "SELECT a FROM scratch").await;
    assert!(gone.contains("scratch"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_node_start_clears_a_temporary_directory_a_crash_left_behind() {
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let root = zyron_catalog::temp_root(tmp.path());
    let leftover = root.join("999");
    std::fs::create_dir_all(&leftover).expect("create");
    std::fs::write(leftover.join("00000042.dat"), b"stale").expect("write");
    assert!(leftover.exists());

    zyron_wire::temp_table_dispatch::clear_on_start(tmp.path()).expect("clears");
    assert!(
        !root.exists(),
        "the whole tree goes before the node accepts a connection"
    );

    // Clearing a directory that is not there is not an error
    zyron_wire::temp_table_dispatch::clear_on_start(tmp.path()).expect("clears again");
}

// ---------------------------------------------------------------------------
// Nothing durable may name one
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_durable_definition_naming_one_is_refused_with_the_reason() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = session_with_id(901);
    ok(&server, &mut session, "CREATE TEMP TABLE scratch (a INT)").await;
    ok(&server, &mut session, "CREATE TABLE permanent (a INT)").await;

    for (sql, expected) in [
        ("CREATE VIEW v AS SELECT a FROM scratch", "view"),
        (
            "CREATE MATERIALIZED VIEW mv AS SELECT a FROM scratch",
            "materialized view",
        ),
        (
            "CREATE TRIGGER t AFTER INSERT ON scratch FOR EACH ROW EXECUTE FUNCTION f()",
            "trigger",
        ),
        (
            "CREATE ABAC POLICY p ON TABLE scratch WHERE a > 0",
            "row security policy",
        ),
        ("GRANT SELECT ON scratch TO someone", "grant"),
        ("COMMENT ON TABLE scratch IS 'note'", "COMMENT ON"),
        (
            "CREATE TABLE child (a INT, FOREIGN KEY (a) REFERENCES scratch(a))",
            "foreign key",
        ),
    ] {
        let error = err(&server, &mut session, sql).await;
        assert!(
            error.contains(expected),
            "`{sql}` should be refused naming {expected}, got {error}"
        );
        assert!(
            error.contains("scratch"),
            "`{sql}` should name the temporary table, got {error}"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn archive_and_restore_of_one_are_refused() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = session_with_id(1001);
    ok(&server, &mut session, "CREATE TEMP TABLE scratch (a INT)").await;
    for (sql, verb) in [
        ("ARCHIVE TABLE scratch TO 'file:///tmp/a'", "ARCHIVE"),
        ("RESTORE TABLE scratch FROM 'file:///tmp/a'", "RESTORE"),
    ] {
        let error = err(&server, &mut session, sql).await;
        assert!(
            error.contains(verb) && error.contains("scratch"),
            "`{sql}` should be refused naming {verb}, got {error}"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn creating_one_inside_a_branch_is_refused() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = session_with_id(1101);
    let stmt = zyron_parser::parse("CREATE TEMP TABLE scratch (a INT)")
        .expect("parse")
        .into_iter()
        .next()
        .expect("one statement");
    let mut txn_opt: Option<zyron_storage::txn::Transaction> = None;
    let mut active_branch = Some("feature".to_string());
    let outcome = zyron_wire::ddl_dispatch::try_handle_ddl_utility(
        &stmt,
        &server,
        &mut session,
        &mut txn_opt,
        &mut active_branch,
        "CREATE TEMP TABLE scratch (a INT)",
    )
    .await
    .expect("handled");
    let error = format!("{:?}", outcome.expect_err("refused"));
    assert!(
        error.contains("branch"),
        "the refusal names the branch, got {error}"
    );
}

// ---------------------------------------------------------------------------
// Limits, statistics, and the operator's view
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_count_limit_refuses_a_further_table_naming_itself() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = session_with_id(1201);
    ok(&server, &mut session, "CREATE TEMP TABLE one (a INT)").await;
    session
        .as_ref()
        .and_then(|s| s.temp_tables.as_ref())
        .expect("namespace")
        .set_limits(u64::MAX, 1);
    let error = err(&server, &mut session, "CREATE TEMP TABLE two (a INT)").await;
    assert!(
        error.contains("temp_table_max_count"),
        "the refusal names the limit, got {error}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_byte_limit_refuses_a_further_table_naming_itself() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = session_with_id(1301);
    ok(&server, &mut session, "CREATE TEMP TABLE one (a INT)").await;
    ok(&server, &mut session, "INSERT INTO one VALUES (1)").await;
    let namespace = session
        .as_ref()
        .and_then(|s| s.temp_tables.as_ref())
        .expect("namespace")
        .clone();
    // Collect statistics so the held bytes are measured, then set a limit
    // below what is already held
    zyron_wire::temp_table_dispatch::refresh_statistics(&server, &session).await;
    namespace.set_limits(1, 256);
    let error = err(&server, &mut session, "CREATE TEMP TABLE two (a INT)").await;
    assert!(
        error.contains("temp_table_max_bytes"),
        "the refusal names the limit, got {error}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_operator_view_lists_the_table_with_its_bytes_and_rows() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = session_with_id(1401);
    ok(&server, &mut session, "CREATE TEMP TABLE scratch (a INT)").await;
    ok(&server, &mut session, "INSERT INTO scratch VALUES (1), (2)").await;
    zyron_wire::temp_table_dispatch::refresh_statistics(&server, &session).await;

    let listed = server.catalog.temp_tables().report();
    let row = listed
        .iter()
        .find(|t| t.name == "scratch" && t.session_id == 1401)
        .expect("the view lists it while it exists");
    assert_eq!(
        row.rows, 2,
        "the row count is what the writes carried forward"
    );
    assert!(
        row.bytes > 0,
        "the bytes are what the table's own files hold"
    );

    // And nothing about it is in a catalog listing
    assert!(
        !server
            .catalog
            .list_all_tables()
            .iter()
            .any(|t| t.name == "scratch"),
        "zyron_sys.catalog.tables never lists a temporary table"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn dropping_one_removes_it_from_the_session_and_the_operator_view() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = session_with_id(1501);
    ok(&server, &mut session, "CREATE TEMP TABLE scratch (a INT)").await;
    ok(&server, &mut session, "DROP TABLE scratch").await;
    assert!(
        !server
            .catalog
            .temp_tables()
            .report()
            .iter()
            .any(|t| t.name == "scratch"),
        "the drop takes it out of the view"
    );
    let gone = err(&server, &mut session, "SELECT a FROM scratch").await;
    assert!(gone.contains("scratch"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn nothing_about_one_reaches_the_catalog_or_the_id_space_a_permanent_table_uses() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = session_with_id(1601);
    // The catalog's own allocator is a function of the applied consensus
    // log, so a temporary table must not move it: two members would then
    // number every later object differently
    ok(&server, &mut session, "CREATE TABLE before_it (a INT)").await;
    let before = server
        .catalog
        .list_all_tables()
        .into_iter()
        .find(|t| t.name == "before_it")
        .map(|t| t.id.0)
        .expect("created");

    ok(&server, &mut session, "CREATE TEMP TABLE scratch (a INT)").await;
    ok(&server, &mut session, "CREATE TABLE after_it (a INT)").await;
    let after = server
        .catalog
        .list_all_tables()
        .into_iter()
        .find(|t| t.name == "after_it")
        .map(|t| t.id.0)
        .expect("created");
    assert_eq!(
        after,
        before + 1,
        "the temporary table took no id from the catalog's allocator"
    );

    // Its own id is in the node-local range and resolves only by id
    let temp_id = session
        .as_ref()
        .and_then(|s| s.temp_tables.as_ref())
        .and_then(|t| t.resolve("scratch"))
        .map(|e| e.id)
        .expect("resolves in the session");
    assert!(
        temp_id.0 > u32::MAX - 1_000_000_000,
        "a temporary id comes from the node-local range, got {}",
        temp_id.0
    );
    assert!(
        server.catalog.get_table_by_id(temp_id).is_ok(),
        "every layer below the planner addresses it by id"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_session_holding_one_is_pinned_and_names_temp_tables_as_the_reason() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = session_with_id(1701);
    assert!(
        !server.catalog.temp_tables().any_held(),
        "nothing pins a session before one is created"
    );
    ok(&server, &mut session, "CREATE TEMP TABLE scratch (a INT)").await;
    assert!(
        server.catalog.temp_tables().any_held(),
        "the registry is what the mesh reads to refuse a relocation"
    );
    assert!(
        session.as_ref().expect("session").holds_temp_tables(),
        "the session says so too"
    );
    let key = session.as_ref().expect("session").session_key;
    zyron_wire::temp_table_dispatch::end_session(&server, key).await;
    assert!(!server.catalog.temp_tables().any_held());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_temporary_table_statement_classifies_local() {
    use zyron_wire::connection::{ReplicationClass, replication_class};
    let temp = zyron_parser::parse("CREATE TEMP TABLE scratch (a INT)")
        .expect("parse")
        .remove(0);
    assert_eq!(
        replication_class(&temp),
        ReplicationClass::Local,
        "a temporary table is node-local by design, so its statement runs here and nowhere else"
    );
    let permanent = zyron_parser::parse("CREATE TABLE shared (a INT)")
        .expect("parse")
        .remove(0);
    assert_eq!(
        replication_class(&permanent),
        ReplicationClass::Statement,
        "a permanent table still reaches the group as itself"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_schema_qualifier_on_a_temporary_table_is_refused_at_parse() {
    assert!(
        zyron_parser::parse("CREATE TEMP TABLE app.scratch (a INT)").is_err(),
        "a temporary table takes a bare name"
    );
    let _ = create_test_server().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_index_may_be_created_on_one_and_lives_in_the_same_directory() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = session_with_id(1801);
    ok(&server, &mut session, "CREATE TEMP TABLE scratch (a INT)").await;
    ok(
        &server,
        &mut session,
        "INSERT INTO scratch VALUES (1), (2), (3)",
    )
    .await;
    // The index build reads the table through the same heap path a scan
    // does, so it resolves the session's own namespace
    match run(&server, &mut session, "CREATE INDEX ix ON scratch (a)").await {
        Ok(_) => {
            let rows = run(&server, &mut session, "SELECT a FROM scratch WHERE a = 2")
                .await
                .expect("select through the index");
            assert_eq!(ints(&rows, 0), vec![2]);
        }
        Err(e) => panic!("an index on a temporary table should build: {e}"),
    }
}

/// The release is bound to the session, not to the connection that opened it.
///
/// Nothing here plays the part of a wire connection: the session is dropped on
/// its own, the way an expiring HTTP session's state is dropped out of the
/// map that holds it, and the tables and their directory go with it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn dropping_the_session_alone_releases_its_temporary_tables() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = session_with_id(1802);
    ok(&server, &mut session, "CREATE TEMP TABLE scratch (a INT)").await;
    ok(&server, &mut session, "INSERT INTO scratch VALUES (1)").await;

    let directory = session
        .as_ref()
        .and_then(|s| s.temp_tables.as_ref())
        .map(|t| t.directory().to_path_buf())
        .expect("the session has a directory");
    assert!(directory.exists(), "the files are on disk while it lives");
    assert!(
        server.catalog.temp_tables().any_held(),
        "the registry holds the table while the session lives"
    );

    drop(session);

    // The registry entry goes synchronously with the drop, so the session's
    // share of the limits is free immediately
    assert!(
        !server.catalog.temp_tables().any_held(),
        "the registry entry goes with the session"
    );

    // The files are removed by the task the drop spawned
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while directory.exists() && std::time::Instant::now() < deadline {
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert!(
        !directory.exists(),
        "the session's directory goes with the session"
    );
}

/// Two sessions sharing a backend process id still get separate namespaces.
///
/// The process id is 32 bits and the wire protocol reuses it when its counter
/// wraps, so it cannot be what separates one session's tables from another's.
/// The session key does that, and every session takes its own when it is
/// built. This is also the case a second id counter elsewhere would produce.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn sessions_sharing_a_process_id_do_not_share_a_namespace() {
    let (server, _schema, _tmp) = create_test_server().await;

    // The same process id for both, which is what reuse or a second counter
    // hands out
    let mut first = session_with_id(4242);
    let mut second = session_with_id(4242);
    assert_ne!(
        first.as_ref().expect("session").session_key,
        second.as_ref().expect("session").session_key,
        "each session takes a key of its own"
    );
    ok(&server, &mut first, "CREATE TEMP TABLE scratch (a INT)").await;
    ok(&server, &mut first, "INSERT INTO scratch VALUES (1)").await;
    ok(&server, &mut second, "CREATE TEMP TABLE scratch (a INT)").await;
    ok(&server, &mut second, "INSERT INTO scratch VALUES (2)").await;

    let rows = run(&server, &mut first, "SELECT a FROM scratch")
        .await
        .expect("select");
    assert_eq!(ints(&rows, 0), vec![1], "each session reads its own table");

    let directories: Vec<_> = [&first, &second]
        .iter()
        .map(|s| {
            s.as_ref()
                .and_then(|s| s.temp_tables.as_ref())
                .map(|t| t.directory().to_path_buf())
                .expect("the session has a directory")
        })
        .collect();
    assert_ne!(
        directories[0], directories[1],
        "two sessions never share a directory"
    );
}

/// Each table reports the bytes of its own files, not of the session's
/// directory.
///
/// The directory holds every table the session owns, so measuring it per
/// table would report each table as costing what all of them cost together,
/// and `check_limits` sums the per-table figures, so the byte limit would
/// refuse at a fraction of the budget that grows with the table count.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_table_reports_its_own_bytes_and_not_the_whole_session() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = session_with_id(1901);
    for table in ["one", "two", "three"] {
        ok(
            &server,
            &mut session,
            &format!("CREATE TEMP TABLE {table} (a INT)"),
        )
        .await;
        ok(
            &server,
            &mut session,
            &format!("INSERT INTO {table} VALUES (1), (2)"),
        )
        .await;
    }
    zyron_wire::temp_table_dispatch::refresh_statistics(&server, &session).await;

    let listed = server.catalog.temp_tables().report();
    let held: Vec<_> = listed
        .iter()
        .filter(|t| t.session_id == 1901)
        .map(|t| (t.name.clone(), t.bytes, t.rows))
        .collect();
    assert_eq!(held.len(), 3, "all three are listed");
    for (name, _, rows) in &held {
        assert_eq!(*rows, 2, "{name} carried its own two rows forward");
    }

    let namespace = session
        .as_ref()
        .and_then(|s| s.temp_tables.as_ref())
        .expect("the session has a namespace");
    let directory_bytes: u64 = std::fs::read_dir(namespace.directory())
        .expect("read")
        .filter_map(|e| e.ok())
        .filter_map(|e| e.metadata().ok())
        .map(|m| m.len())
        .sum();
    let summed: u64 = held.iter().map(|(_, bytes, _)| *bytes).sum();
    assert!(
        summed <= directory_bytes.max(1),
        "the three tables together report {summed} bytes against {directory_bytes} on disk, \
         so a table is reporting more than its own files"
    );
}

//! CREATE TABLE USING ZYRONLAKE must materialize the table's transaction
//! log and flip the catalog format flag, atomically from the user's view.
//!
//! Run: cargo test -p zyron-wire --test lake_ddl_test

use std::sync::Arc;

use zyron_catalog::DatabaseId;
use zyron_lake::{AllCommitted, ClusterStrategy, LakePaths, TransactionLog};
use zyron_wire::connection::ServerState;
use zyron_wire::session::Session;

mod common;
use common::{
    create_test_server, create_test_server_in_mode, exec_ddl, exec_dml, new_session, query_error,
    query_rows, query_values, render_plan, run_on_branch,
};

#[tokio::test]
async fn test_create_lake_table_writes_log_and_flips_catalog() {
    let (server, schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE events (id BIGINT NOT NULL, payload TEXT) USING ZYRONLAKE \
         CLUSTER BY (id USING RangePartition) WITH (target_file_size = '268435456')",
    )
    .await
    .expect("create lake table");

    let entry = server
        .catalog
        .get_table(schema_id, "events")
        .expect("entry");
    assert!(entry.lake.is_lake(), "catalog format flag must flip");

    let paths = LakePaths::new(server.disk_manager.data_dir(), entry.id.0);
    assert!(paths.version_file(1).exists(), "version one must exist");
    let log = TransactionLog::open(paths, &AllCommitted).expect("log opens");
    assert_eq!(log.latest_version(), 1);
    let manifest = log.latest_manifest().expect("manifest");
    assert_eq!(manifest.schema.columns.len(), 2);
    assert_eq!(
        manifest.schema.column_by_name("id").map(|c| c.nullable),
        Some(false)
    );
    assert_eq!(manifest.cluster_spec.keys.len(), 1);
    assert_eq!(
        manifest.cluster_spec.keys[0].strategy,
        ClusterStrategy::RangePartition
    );
    assert_eq!(
        manifest
            .properties
            .get("target_file_size")
            .map(|s| s.as_str()),
        Some("268435456")
    );
    // A bare key list seeds the layout and leaves measurement free to move
    // it, and the schedule is written at creation so the manifest states the
    // policy rather than leaning on a fallback
    assert_eq!(
        manifest
            .properties
            .get("clustering_mode")
            .map(|s| s.as_str()),
        Some("auto")
    );
    assert_eq!(
        manifest
            .properties
            .get("clustering_schedule")
            .map(|s| s.as_str()),
        Some("continuous")
    );
}

#[tokio::test]
async fn test_heap_create_table_is_unchanged() {
    let (server, schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(&server, &mut session, "CREATE TABLE plain (id BIGINT)")
        .await
        .expect("create heap table");
    let entry = server.catalog.get_table(schema_id, "plain").expect("entry");
    assert!(!entry.lake.is_lake());
    assert!(
        !server
            .disk_manager
            .data_dir()
            .join("lake")
            .join(format!("t{}", entry.id.0))
            .exists()
    );
}

/// Runs a DML statement and returns the affected row count it reports
async fn rows_affected(server: &Arc<ServerState>, sql: &str) -> i64 {
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
    let snapshot = txn.snapshot.clone();
    let txn_id = txn.txn_id;
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
    let ctx = Arc::new(ctx);
    let batches = zyron_executor::execute(plan, &ctx).await.expect("execute");
    server.txn_manager.commit(&mut txn).await.expect("commit");
    zyron_lake::publish_txn(server.disk_manager.data_dir(), txn_id).expect("publish");
    batches
        .first()
        .map(|b| match b.columns[0].get_scalar(0) {
            zyron_executor::column::ScalarValue::Int64(v) => v,
            other => panic!("unexpected count scalar {other:?}"),
        })
        .unwrap_or(0)
}

#[tokio::test]
async fn test_lake_scan_reads_appended_rows_and_sql_insert_publishes_on_commit() {
    let (server, schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE lk (id BIGINT NOT NULL, name TEXT) USING ZYRONLAKE",
    )
    .await
    .expect("create lake table");

    // An empty lake table scans to zero rows through the full SQL path
    assert_eq!(query_rows(&server, "SELECT * FROM lk").await, 0);

    // Rows appended through the lake operations land as a data file plus
    // one log commit, then SQL reads them back
    let entry = server.catalog.get_table(schema_id, "lk").expect("entry");
    let paths = LakePaths::new(server.disk_manager.data_dir(), entry.id.0);
    let log = zyron_lake::TransactionLog::open_shared(paths, &AllCommitted).expect("shared log");
    let columns = vec![
        zyron_lake::ColumnData {
            column_id: entry.columns[0].id.0 as u32,
            cells: vec![
                Some(1i64.to_le_bytes().to_vec()),
                Some(2i64.to_le_bytes().to_vec()),
                Some(3i64.to_le_bytes().to_vec()),
            ],
        },
        zyron_lake::ColumnData {
            column_id: entry.columns[1].id.0 as u32,
            cells: vec![Some(b"alice".to_vec()), None, Some(b"carol".to_vec())],
        },
    ];
    let attempt = zyron_lake::CommitAttempt {
        operation: zyron_lake::OperationKind::Append,
        db_txn_id: 0,
        commit_lsn: 0,
        timestamp_us: 1_754_700_000_000_000,
        read_predicate: None,
        read_version: 0,
        audit: None,
        deadline: None,
    };
    let out = zyron_lake::append_rows(&log, attempt, entry.id.0 as u64, &columns).expect("append");
    assert_eq!(out.rows, 3);

    assert_eq!(query_rows(&server, "SELECT * FROM lk").await, 3);
    assert_eq!(
        query_rows(&server, "SELECT id FROM lk WHERE id >= 2").await,
        2
    );
    assert_eq!(
        query_rows(&server, "SELECT name FROM lk WHERE name = 'alice'").await,
        1
    );

    // SQL INSERT routes through the lake append path, pending under its
    // transaction until the commit publishes it
    exec_dml(&server, "INSERT INTO lk VALUES (9, 'dave'), (10, NULL)").await;
    assert_eq!(query_rows(&server, "SELECT * FROM lk").await, 5);
    assert_eq!(
        query_rows(&server, "SELECT id FROM lk WHERE name = 'dave'").await,
        1
    );

    // An aborted transaction's lake version never becomes visible
    {
        let stmt = zyron_parser::parse("INSERT INTO lk VALUES (99, 'ghost')")
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
        let snapshot = txn.snapshot.clone();
        let txn_id = txn.txn_id;
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
        let ctx = Arc::new(ctx);
        let _ = zyron_executor::execute(plan, &ctx).await.expect("execute");
        zyron_lake::abandon_txn(server.disk_manager.data_dir(), txn_id);
        server.txn_manager.abort(&mut txn).expect("abort");
    }
    assert_eq!(query_rows(&server, "SELECT * FROM lk").await, 5);

    // Predicate DELETE removes matching rows and leaves the rest
    exec_dml(&server, "DELETE FROM lk WHERE id >= 9").await;
    assert_eq!(query_rows(&server, "SELECT * FROM lk").await, 3);
    assert_eq!(
        query_rows(&server, "SELECT id FROM lk WHERE id >= 9").await,
        0
    );

    // DELETE with no predicate empties the table
    exec_dml(&server, "DELETE FROM lk").await;
    assert_eq!(query_rows(&server, "SELECT * FROM lk").await, 0);
}

#[tokio::test]
async fn test_lake_update_replaces_rows_without_deleting_its_own_writes() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE ct (id BIGINT NOT NULL, name TEXT) USING ZYRONLAKE",
    )
    .await
    .expect("create lake table");
    exec_dml(
        &server,
        "INSERT INTO ct VALUES (1, 'a'), (2, 'b'), (60, 'c')",
    )
    .await;

    // The new images still satisfy the predicate that matched them, the
    // classic case where a delete recorded after the write would eat them
    assert_eq!(
        rows_affected(&server, "UPDATE ct SET id = id + 1 WHERE id < 50").await,
        2
    );
    assert_eq!(query_rows(&server, "SELECT * FROM ct").await, 3);
    assert_eq!(
        query_rows(&server, "SELECT id FROM ct WHERE id < 50").await,
        2
    );
    assert_eq!(
        query_rows(&server, "SELECT id FROM ct WHERE id = 2").await,
        1
    );
    assert_eq!(
        query_rows(&server, "SELECT id FROM ct WHERE id = 3").await,
        1
    );
    assert_eq!(
        query_rows(&server, "SELECT id FROM ct WHERE id = 1").await,
        0
    );

    // Unassigned columns keep their values through the rewrite
    assert_eq!(
        query_rows(&server, "SELECT id FROM ct WHERE name = 'a'").await,
        1
    );

    // An update over every row replaces the whole table
    assert_eq!(rows_affected(&server, "UPDATE ct SET name = 'z'").await, 3);
    assert_eq!(
        query_rows(&server, "SELECT id FROM ct WHERE name = 'z'").await,
        3
    );

    // A predicate with no exact lake equivalent is refused
    let stmt = zyron_parser::parse("UPDATE ct SET name = 'q' WHERE name LIKE 'z%'")
        .expect("parse")
        .into_iter()
        .next()
        .expect("one statement");
    let err = zyron_planner::plan(
        &server.catalog,
        DatabaseId(1),
        vec!["public".into()],
        stmt,
        None,
    )
    .await
    .expect_err("LIKE has no exact lake equivalent");
    assert!(err.to_string().contains("no exact equivalent"), "{err}");
}

#[tokio::test]
async fn test_lake_delete_drops_covered_files_and_reports_exact_counts() {
    let (server, schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE ev (id BIGINT NOT NULL, name TEXT) USING ZYRONLAKE",
    )
    .await
    .expect("create lake table");

    // Two files, one entirely below the cut and one straddling it
    exec_dml(&server, "INSERT INTO ev VALUES (1, 'a'), (2, 'b')").await;
    exec_dml(&server, "INSERT INTO ev VALUES (40, 'c'), (60, 'd')").await;
    let entry = server.catalog.get_table(schema_id, "ev").expect("entry");
    let paths = LakePaths::new(server.disk_manager.data_dir(), entry.id.0);
    let log = zyron_lake::TransactionLog::open_shared(paths, &AllCommitted).expect("log");
    assert_eq!(log.latest_manifest().expect("manifest").entries.len(), 2);

    // The low file is fully covered and drops whole, the straddling file
    // keeps the predicate and filters at read time
    assert_eq!(
        rows_affected(&server, "DELETE FROM ev WHERE id < 50").await,
        3
    );
    let manifest = log.latest_manifest().expect("manifest");
    assert_eq!(
        manifest.entries.len(),
        1,
        "the covered file was removed whole"
    );
    assert_eq!(manifest.delete_predicates.len(), 1);
    assert_eq!(query_rows(&server, "SELECT * FROM ev").await, 1);

    // Repeating the delete matches nothing and commits nothing
    let before = log.latest_version();
    assert_eq!(
        rows_affected(&server, "DELETE FROM ev WHERE id < 50").await,
        0
    );
    assert_eq!(log.latest_version(), before);

    // A predicate the lake format cannot record exactly is refused, not
    // approximated
    let stmt = zyron_parser::parse("DELETE FROM ev WHERE name LIKE 'd%'")
        .expect("parse")
        .into_iter()
        .next()
        .expect("one statement");
    let err = zyron_planner::plan(
        &server.catalog,
        DatabaseId(1),
        vec!["public".into()],
        stmt,
        None,
    )
    .await
    .expect_err("LIKE has no exact lake equivalent");
    assert!(err.to_string().contains("no exact equivalent"), "{err}");
}

#[tokio::test]
async fn test_drop_lake_table_reclaims_the_whole_root() {
    let (server, schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE gone (id BIGINT NOT NULL) USING ZYRONLAKE",
    )
    .await
    .expect("create lake table");
    let entry = server.catalog.get_table(schema_id, "gone").expect("entry");
    let paths = LakePaths::new(server.disk_manager.data_dir(), entry.id.0);
    let log =
        zyron_lake::TransactionLog::open_shared(paths.clone(), &AllCommitted).expect("shared log");
    let attempt = zyron_lake::CommitAttempt {
        operation: zyron_lake::OperationKind::Append,
        db_txn_id: 0,
        commit_lsn: 0,
        timestamp_us: 1_754_700_000_000_000,
        read_predicate: None,
        read_version: 0,
        audit: None,
        deadline: None,
    };
    zyron_lake::append_rows(
        &log,
        attempt,
        entry.id.0 as u64,
        &[zyron_lake::ColumnData {
            column_id: entry.columns[0].id.0 as u32,
            cells: vec![Some(1i64.to_le_bytes().to_vec())],
        }],
    )
    .expect("append");
    assert!(paths.root().exists());
    drop(log);

    exec_ddl(&server, &mut session, "DROP TABLE gone")
        .await
        .expect("drop");
    assert!(
        !paths.root().exists(),
        "the lake root, log and data files must be reclaimed"
    );
    assert!(server.catalog.get_table(schema_id, "gone").is_err());
}

#[tokio::test]
async fn test_bad_cluster_column_rolls_back_the_table() {
    let (server, schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    let err = exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE broken (id BIGINT) USING ZYRONLAKE CLUSTER BY (nope)",
    )
    .await
    .expect_err("must fail");
    assert!(err.contains("nope"), "error names the column: {err}");
    assert!(
        server.catalog.get_table(schema_id, "broken").is_err(),
        "half-created table must be rolled back"
    );
}

// ---------------------------------------------------------------------------
// Deployment modes. The mode picks the format a CREATE TABLE with no USING
// clause lands in and refuses DDL naming the format the node does not run.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_db_node_rejects_lake_ddl_with_actionable_error() {
    let (server, schema_id, _tmp) =
        create_test_server_in_mode(zyron_common::DeploymentMode::Db).await;
    let mut session = new_session();

    let err = exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE events (id BIGINT) USING ZYRONLAKE",
    )
    .await
    .expect_err("a db node must refuse lake DDL");

    assert!(err.contains("USING ZYRONLAKE is refused"), "{err}");
    assert!(err.contains("deployment mode"), "{err}");
    assert!(err.contains("heap tables only"), "{err}");
    // Actionable: the message names the setting and both modes that accept it
    assert!(err.contains("storage.deployment_mode"), "{err}");
    assert!(err.contains("unified"), "{err}");
    assert!(err.contains("lake"), "{err}");
    assert!(
        server.catalog.get_table(schema_id, "events").is_err(),
        "the refusal must land before the catalog entry exists"
    );

    // The format the node does run is untouched
    exec_ddl(&server, &mut session, "CREATE TABLE plain (id BIGINT)")
        .await
        .expect("heap table on a db node");
    let entry = server.catalog.get_table(schema_id, "plain").expect("entry");
    assert!(!entry.lake.is_lake());
}

#[tokio::test]
async fn test_lake_node_rejects_heap_ddl_with_actionable_error() {
    let (server, schema_id, _tmp) =
        create_test_server_in_mode(zyron_common::DeploymentMode::Lake).await;
    let mut session = new_session();

    let err = exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE rows_only (id BIGINT) USING HEAP",
    )
    .await
    .expect_err("a lake node must refuse heap DDL");

    assert!(err.contains("USING HEAP is refused"), "{err}");
    assert!(err.contains("ZyronLake tables only"), "{err}");
    assert!(err.contains("storage.deployment_mode"), "{err}");
    assert!(
        server.catalog.get_table(schema_id, "rows_only").is_err(),
        "the refusal must land before the catalog entry exists"
    );
}

#[tokio::test]
async fn test_lake_node_defaults_create_table_to_the_lake_format() {
    let (server, schema_id, _tmp) =
        create_test_server_in_mode(zyron_common::DeploymentMode::Lake).await;
    let mut session = new_session();

    // No USING clause: on a single-format node nobody types one
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE events (id BIGINT NOT NULL, payload TEXT) CLUSTER BY (id)",
    )
    .await
    .expect("create table on a lake node");

    let entry = server
        .catalog
        .get_table(schema_id, "events")
        .expect("entry");
    assert!(
        entry.lake.is_lake(),
        "the deployment default must pick the lake format"
    );
    let paths = LakePaths::new(server.disk_manager.data_dir(), entry.id.0);
    assert!(paths.version_file(1).exists(), "version one must exist");
    let log = TransactionLog::open(paths, &AllCommitted).expect("log opens");
    assert_eq!(
        log.latest_manifest()
            .expect("manifest")
            .cluster_spec
            .keys
            .len(),
        1
    );
}

#[tokio::test]
async fn test_unified_node_accepts_both_formats_and_defaults_to_heap() {
    let (server, schema_id, _tmp) =
        create_test_server_in_mode(zyron_common::DeploymentMode::Unified).await;
    let mut session = new_session();

    exec_ddl(&server, &mut session, "CREATE TABLE d (id BIGINT)")
        .await
        .expect("default format");
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE h (id BIGINT) USING HEAP",
    )
    .await
    .expect("explicit heap");
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE l (id BIGINT) USING ZYRONLAKE",
    )
    .await
    .expect("explicit lake");

    assert!(
        !server
            .catalog
            .get_table(schema_id, "d")
            .expect("d")
            .lake
            .is_lake()
    );
    assert!(
        !server
            .catalog
            .get_table(schema_id, "h")
            .expect("h")
            .lake
            .is_lake()
    );
    assert!(
        server
            .catalog
            .get_table(schema_id, "l")
            .expect("l")
            .lake
            .is_lake()
    );
}

#[tokio::test]
async fn test_cluster_by_on_a_heap_table_is_refused_not_silently_accepted() {
    let (server, schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();

    let err = exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE h (id BIGINT) CLUSTER BY (id)",
    )
    .await
    .expect_err("CLUSTER BY must not report success on a heap table");
    assert!(
        err.contains("CLUSTER BY applies to ZyronLake tables"),
        "{err}"
    );
    assert!(
        server.catalog.get_table(schema_id, "h").is_err(),
        "nothing may be created by a refused statement"
    );
}

/// The LakeScan node of an explain tree, wherever the plan puts it.
fn lake_scan_node(
    plan: &zyron_planner::physical::PhysicalPlan,
) -> zyron_planner::explain::ExplainNode {
    fn find(
        node: &zyron_planner::explain::ExplainNode,
    ) -> Option<zyron_planner::explain::ExplainNode> {
        if node.operator_name == "LakeScan" {
            return Some(node.clone());
        }
        node.children.iter().find_map(find)
    }
    let tree = zyron_planner::explain::ExplainNode::from_physical_plan(plan);
    find(&tree).expect("the plan must contain a LakeScan")
}

/// Plans one statement and returns the physical plan for inspection.
async fn plan_select(
    server: &Arc<ServerState>,
    sql: &str,
) -> zyron_planner::physical::PhysicalPlan {
    let stmt = zyron_parser::parse(sql)
        .expect("parse")
        .into_iter()
        .next()
        .expect("one statement");
    zyron_planner::plan(
        &server.catalog,
        DatabaseId(1),
        vec!["public".into()],
        stmt,
        None,
    )
    .await
    .expect("plan")
}

/// A lake table never runs ANALYZE. Its statistics come from the manifest
/// every commit produces, and the planner must use them instead of its
/// no-statistics defaults.
#[tokio::test]
async fn test_lake_plan_uses_manifest_stats_without_calling_analyze() {
    let (server, schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE m (id BIGINT NOT NULL, name TEXT) USING ZYRONLAKE",
    )
    .await
    .expect("create lake table");

    let entry = server.catalog.get_table(schema_id, "m").expect("entry");
    // An empty lake table reports zero rows, not the 1000-row default
    let empty = server.catalog.get_stats(entry.id).expect("stats at create");
    assert_eq!(empty.0.row_count, 0);

    exec_dml(
        &server,
        "INSERT INTO m VALUES (1, 'alice'), (2, 'bob'), (3, NULL), (4, 'dave')",
    )
    .await;

    let stats = server
        .catalog
        .get_stats(entry.id)
        .expect("statistics published by the commit");
    assert_eq!(stats.0.row_count, 4, "row count comes from the manifest");
    assert!(stats.0.page_count > 0, "the data file occupies pages");
    assert!(stats.0.avg_row_size > 0);

    // Exact null fraction, straight off the writer's per-column counts
    let name_col = entry.columns[1].id;
    let name_stats = stats
        .1
        .iter()
        .find(|c| c.column_id == name_col)
        .expect("column stats");
    assert!(
        (name_stats.null_fraction - 0.25).abs() < 1e-9,
        "one of four names is NULL, got {}",
        name_stats.null_fraction
    );
    let id_stats = stats
        .1
        .iter()
        .find(|c| c.column_id == entry.columns[0].id)
        .expect("column stats");
    assert_eq!(id_stats.null_fraction, 0.0);

    // The plan sees them: a scan of four rows must not cost the no-statistics
    // default of 1000
    let plan = plan_select(&server, "SELECT id FROM m").await;
    let node = lake_scan_node(&plan);
    let cost = node.estimated_cost.expect("the scan is costed");
    assert_eq!(cost.row_count, 4.0, "cardinality comes from the manifest");
}

/// EXPLAIN must say whether a filter skips whole data files, which is the
/// difference between opening one file and opening all of them.
#[tokio::test]
async fn test_explain_lake_scan_reports_file_pruning() {
    let (server, schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE p (id BIGINT NOT NULL, name TEXT) USING ZYRONLAKE",
    )
    .await
    .expect("create lake table");
    let _ = schema_id;

    exec_dml(&server, "INSERT INTO p VALUES (1, 'alice')").await;

    // A lowerable predicate prunes files
    let plan = plan_select(&server, "SELECT id FROM p WHERE id = 1").await;
    let node = lake_scan_node(&plan);
    let pruning = node
        .details
        .iter()
        .find(|(k, _)| k == "file_pruning")
        .map(|(_, v)| v.clone());
    assert_eq!(pruning.as_deref(), Some("exact"));

    // LIKE has no exact lake equivalent, so no file is skipped and EXPLAIN
    // says so rather than implying pruning happened
    let plan = plan_select(&server, "SELECT id FROM p WHERE name LIKE 'a%'").await;
    let node = lake_scan_node(&plan);
    let pruning = node
        .details
        .iter()
        .find(|(k, _)| k == "file_pruning")
        .map(|(_, v)| v.clone());
    assert_eq!(pruning.as_deref(), Some("none"));
}

/// EXPLAIN ANALYZE has to report what the scan actually did, not only what
/// the planner expected. The pruning counts are the ones that matter on a
/// lake table: they are the difference between opening one file and all of
/// them, and nothing else in the output reveals them.
#[tokio::test]
async fn test_explain_analyze_reports_measured_lake_pruning() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE p (id BIGINT NOT NULL, name TEXT) USING ZYRONLAKE CLUSTER BY (id)",
    )
    .await
    .expect("create lake table");

    // Four files with disjoint id ranges, so a predicate on id can skip
    for batch in 0..4i64 {
        let base = batch * 100;
        exec_dml(
            &server,
            &format!(
                "INSERT INTO p VALUES ({}, 'a'), ({}, 'b')",
                base + 1,
                base + 2
            ),
        )
        .await;
    }

    let plan = plan_select(&server, "SELECT id FROM p WHERE id = 1").await;
    let mut tree = zyron_planner::ExplainNode::from_physical_plan(&plan);

    let mut txn = server
        .txn_manager
        .begin(zyron_storage::txn::IsolationLevel::ReadCommitted)
        .expect("begin");
    let snapshot = txn.snapshot.clone();
    let txn_id = txn.txn_id;
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
    // What EXPLAIN ANALYZE sets, and the only thing that makes the executor
    // wrap its operators in metrics collectors
    ctx.analyze = true;
    let ctx = Arc::new(ctx);
    let (_batches, metrics) = zyron_executor::execute_analyze(plan, &ctx)
        .await
        .expect("analyze");
    server.txn_manager.commit(&mut txn).await.expect("commit");

    let metrics = metrics.expect("analyze mode produces metrics");
    let node_metrics = node_metrics_of(&metrics);
    assert!(
        tree.merge_metrics(&node_metrics) > 0,
        "the plan and the executor must agree on the operator names"
    );

    let scan = find_named(&tree, "LakeScan").expect("a LakeScan node");
    let actual = scan
        .actual_metrics
        .as_ref()
        .expect("the scan reports what it measured");
    let considered = actual.aux[0];
    let pruned = actual.aux[1];
    assert_eq!(considered, 4, "four files were listed");
    assert_eq!(pruned, 3, "three of them could not hold id = 1");
    assert!(
        actual.aux[3] > 0 && actual.aux[3] < actual.aux[2],
        "the pruned bytes are a real fraction of the listed bytes"
    );

    let text = tree.render(&zyron_planner::ExplainOptions {
        analyze: true,
        ..Default::default()
    });
    assert!(text.contains("files_considered=4"), "{text}");
    assert!(text.contains("files_pruned=3"), "{text}");
}

/// Reads the executor's counters into the planner's tree, the way the wire
/// layer does before merging them into an explain plan
fn node_metrics_of(
    metrics: &zyron_executor::operator::OperatorMetrics,
) -> zyron_planner::NodeMetrics {
    use std::sync::atomic::Ordering;
    let mut aux = [0u64; zyron_planner::ACTUAL_AUX_SLOTS];
    for (slot, value) in aux.iter_mut().enumerate() {
        *value = metrics.aux(slot);
    }
    zyron_planner::NodeMetrics {
        name: metrics.name.clone(),
        rows: metrics.rows_produced.load(Ordering::Relaxed),
        elapsed_ns: metrics.elapsed_ns.load(Ordering::Relaxed),
        batches: metrics.batches.load(Ordering::Relaxed),
        aux,
        children: metrics
            .children
            .iter()
            .map(|c| node_metrics_of(c))
            .collect(),
    }
}

fn find_named<'a>(
    node: &'a zyron_planner::ExplainNode,
    name: &str,
) -> Option<&'a zyron_planner::ExplainNode> {
    if node.operator_name == name {
        return Some(node);
    }
    node.children.iter().find_map(|c| find_named(c, name))
}

/// Peering is declared, never discovered, and it survives a restart: the
/// registry lives beside the data rather than in the catalog, because it
/// is this node's view of the mesh and not something to replicate.
#[tokio::test]
async fn test_peers_are_declared_persisted_and_dropped() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let data_dir = server.disk_manager.data_dir().to_path_buf();
    let mut session = new_session();

    exec_ddl(
        &server,
        &mut session,
        "CREATE PEER west ADDRESS '10.0.0.2:5433' MODE lake",
    )
    .await
    .expect("declare peer");
    exec_ddl(
        &server,
        &mut session,
        "CREATE PEER east ADDRESS '10.0.0.3:5433'",
    )
    .await
    .expect("declare peer with unknown mode");

    {
        let peers = server.peers.read();
        assert_eq!(peers.peers().len(), 2);
        assert_eq!(
            peers.get("west").and_then(|p| p.mode),
            Some(zyron_common::DeploymentMode::Lake)
        );
        // An undeclared mode is unknown, not a guess
        assert_eq!(peers.get("east").expect("east").mode, None);
    }

    // Declaring the same name again does not silently move the address
    let again = exec_ddl(
        &server,
        &mut session,
        "CREATE PEER west ADDRESS 'somewhere-else:5433'",
    )
    .await;
    assert!(
        again.is_err(),
        "a redeclaration must not replace an address"
    );
    exec_ddl(
        &server,
        &mut session,
        "CREATE PEER IF NOT EXISTS west ADDRESS 'somewhere-else:5433'",
    )
    .await
    .expect("IF NOT EXISTS is a no-op");
    assert_eq!(
        server.peers.read().get("west").expect("west").address,
        "10.0.0.2:5433"
    );

    // An unknown mode name is refused rather than stored as a mystery
    assert!(
        exec_ddl(
            &server,
            &mut session,
            "CREATE PEER bad ADDRESS 'h:1' MODE sideways"
        )
        .await
        .is_err()
    );

    // The membership is on disk, where a restart will find it
    let reloaded = zyron_common::PeerRegistry::load(&data_dir).expect("reload");
    assert_eq!(reloaded.peers().len(), 2);

    // DROP PEER, and IF EXISTS on one that is gone
    exec_ddl(&server, &mut session, "DROP PEER east")
        .await
        .expect("drop");
    assert!(
        exec_ddl(&server, &mut session, "DROP PEER east")
            .await
            .is_err()
    );
    exec_ddl(&server, &mut session, "DROP PEER IF EXISTS east")
        .await
        .expect("IF EXISTS is a no-op");
    assert_eq!(
        zyron_common::PeerRegistry::load(&data_dir)
            .expect("reload")
            .peers()
            .len(),
        1
    );
}

/// zyron_nodes reports this node for certain and a peer only as far as it
/// has been told, so an unreached peer reads as unknown rather than as a
/// guess that happens to be wrong.
#[tokio::test]
async fn test_zyron_nodes_reports_mode_format_and_membership() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE lake_t (id BIGINT NOT NULL) USING ZYRONLAKE",
    )
    .await
    .expect("lake table");
    exec_ddl(&server, &mut session, "CREATE TABLE heap_t (id BIGINT)")
        .await
        .expect("heap table");
    exec_ddl(
        &server,
        &mut session,
        "CREATE PEER west ADDRESS '10.0.0.2:5433' MODE db",
    )
    .await
    .expect("peer");

    let (columns, rows) = stat_view(&server, "SELECT * FROM zyron_nodes").expect("nodes view");
    let col = |name: &str| columns.iter().position(|c| c == name).expect(name);
    assert_eq!(rows.len(), 2, "this node plus one peer");

    let local = rows
        .iter()
        .find(|r| r[col("is_local")] == "t")
        .expect("a local row");
    assert_eq!(local[col("mode")], "unified");
    assert_ne!(
        local[col("node_id")],
        "",
        "the local node knows its own id for certain"
    );
    assert_eq!(local[col("lake_tables")], "1");
    assert_eq!(local[col("heap_tables")], "1");

    let peer = rows
        .iter()
        .find(|r| r[col("is_local")] == "f")
        .expect("a peer row");
    assert_eq!(peer[col("node_name")], "west");
    assert_eq!(peer[col("address")], "10.0.0.2:5433");
    assert_eq!(peer[col("mode")], "db");
    assert_eq!(
        peer[col("node_id")],
        "",
        "a peer's id is learned when it is reached, not declared"
    );
    assert_eq!(
        peer[col("lake_tables")],
        "",
        "what a peer holds is the peer's to report"
    );

    // The view narrows the way every other stat view does
    let (_, narrowed) = stat_view(
        &server,
        "SELECT * FROM zyron_nodes WHERE node_name = 'west'",
    )
    .expect("narrowed nodes view");
    assert_eq!(narrowed.len(), 1);
}

/// A reader is entitled to know how current an answer is before trusting
/// it. A table this node writes is its own authority; a table it follows
/// is only as current as the last version it replayed, and the view says
/// which one it is looking at.
#[tokio::test]
async fn test_table_freshness_distinguishes_a_leader_from_a_lagging_follower() {
    let (server, schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE orders (id BIGINT NOT NULL) USING ZYRONLAKE",
    )
    .await
    .expect("create");
    // Several commits, so a follower that replays one is provably behind
    for id in 1..=4i64 {
        exec_dml(&server, &format!("INSERT INTO orders VALUES ({})", id)).await;
    }

    // A table this node writes reports itself current, with no lag
    let (columns, rows) =
        stat_view(&server, "SELECT * FROM zyron_table_freshness").expect("freshness view");
    let col = |name: &str| columns.iter().position(|c| c == name).expect(name);
    let row = rows
        .iter()
        .find(|r| r[col("table_name")] == "orders")
        .expect("a row for orders");
    assert_eq!(row[col("role")], "leader");
    assert_eq!(row[col("is_current")], "t");
    assert_eq!(row[col("lag_versions")], "0");
    assert_ne!(row[col("version")], "0");

    // Now make it a follower that has replayed part of a leader's log. The
    // leader here is the table's own history, replayed into a second table
    // sharing its data files, which is what shared storage means
    let entry = server
        .catalog
        .get_table(schema_id, "orders")
        .expect("entry");
    let leader_paths = zyron_lake::LakePaths::new(server.disk_manager.data_dir(), entry.id.0);
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE replica (id BIGINT NOT NULL) USING ZYRONLAKE",
    )
    .await
    .expect("create replica");
    let replica_entry = server
        .catalog
        .get_table(schema_id, "replica")
        .expect("entry");
    let replica_paths =
        zyron_lake::LakePaths::new(server.disk_manager.data_dir(), replica_entry.id.0)
            .with_shared_data(&leader_paths);
    let replica_log =
        zyron_lake::TransactionLog::open(replica_paths.clone(), &zyron_lake::AllCommitted)
            .expect("open replica");

    // Replay one version only, so the follower is deliberately behind
    zyron_lake::sync(&leader_paths, &replica_log, 1).expect("partial sync");
    let cursor = zyron_lake::load_cursor(&replica_paths).expect("cursor");
    assert!(
        !cursor.is_current(),
        "the follower must actually be behind for this to prove anything"
    );

    // The registry holds the log the view reads, so the replica's shared
    // paths have to be the ones registered
    zyron_lake::TransactionLog::register_shared(std::sync::Arc::new(replica_log));

    let (columns, rows) =
        stat_view(&server, "SELECT * FROM zyron_table_freshness").expect("freshness view");
    let col = |name: &str| columns.iter().position(|c| c == name).expect(name);
    let replica_row = rows
        .iter()
        .find(|r| r[col("table_name")] == "replica")
        .expect("a row for the replica");
    assert_eq!(replica_row[col("role")], "follower");
    assert_eq!(replica_row[col("is_current")], "f");
    assert_ne!(
        replica_row[col("lag_versions")],
        "0",
        "a follower that is behind says how far"
    );
    // And reads against it still work, which is the point: stale but
    // available beats current but stalled
    assert!(replica_row[col("version")].parse::<u64>().expect("version") > 0);
}

/// A peer that cannot be reached stays declared, records why, and keeps
/// whatever was learned about it before. A mesh that refused to remember a
/// peer until it was up could not be configured before it was running.
#[tokio::test]
async fn test_an_unreachable_peer_stays_declared_and_records_why() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let data_dir = server.disk_manager.data_dir().to_path_buf();
    let mut session = new_session();

    // Nothing listens on port 1, so contact fails fast and definitively
    exec_ddl(
        &server,
        &mut session,
        "CREATE PEER down ADDRESS '127.0.0.1:1' MODE lake",
    )
    .await
    .expect("declaring a peer must not depend on it being up");

    // Contact runs off the statement's path, so wait for it to land
    zyron_wire::ddl_dispatch::contact_peer(&server, "down", "127.0.0.1:1").await;

    let peers = server.peers.read();
    let peer = peers.get("down").expect("the peer is still declared");
    assert!(!peer.is_known(), "an unreachable peer has no learned id");
    let reason = peer.last_error.as_deref().expect("a recorded reason");
    assert!(reason.contains("127.0.0.1:1"), "{reason}");
    // What the operator declared survives, since it is all that is known
    assert_eq!(
        peer.effective_mode(),
        Some(zyron_common::DeploymentMode::Lake)
    );
    drop(peers);

    // It survives a restart with the reason intact
    let reloaded = zyron_common::PeerRegistry::load(&data_dir).expect("reload");
    let peer = reloaded.get("down").expect("still there");
    assert!(peer.last_error.is_some());
    assert!(!peer.is_known());

    // And the mesh view says the id is unknown rather than inventing one
    let (columns, rows) = stat_view(
        &server,
        "SELECT * FROM zyron_nodes WHERE node_name = 'down'",
    )
    .expect("nodes view");
    let col = |name: &str| columns.iter().position(|c| c == name).expect(name);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0][col("node_id")], "");
    assert_eq!(rows[0][col("mode")], "lake");
}

/// A node reached by a second address is itself, not a peer. Recording it
/// as one would give the mesh a cycle of length one, and every freshness
/// answer through it would be about this node's own data.
#[tokio::test]
async fn test_contacting_this_node_by_another_address_is_not_a_peer() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE PEER mirror ADDRESS '127.0.0.1:1'",
    )
    .await
    .expect("declare");

    // Give the peer this node's own id, which is what a probe of a second
    // address for this node would return
    {
        let mut guard = server.peers.write();
        let peers = std::sync::Arc::make_mut(&mut guard);
        let peer = peers.get_mut("mirror").expect("mirror");
        peer.observed(
            server.node_identity.node_id,
            zyron_common::DeploymentMode::Unified,
            1,
        );
    }
    // A later contact recognizes it and refuses to keep treating it as a
    // peer, rather than leaving a self-reference in the mesh
    zyron_wire::ddl_dispatch::contact_peer(&server, "mirror", "127.0.0.1:1").await;
    let peers = server.peers.read();
    let peer = peers.get("mirror").expect("mirror");
    assert!(peer.last_error.is_some(), "the self-reference is reported");
}

/// A follower that can only reach a leader over the wire must end up with
/// exactly what one reading the leader's filesystem would. If the two
/// paths ever disagree, a replica's state depends on how it was reached.
#[tokio::test]
async fn test_the_published_log_decodes_to_the_same_versions_as_the_filesystem() {
    let (server, schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE shipped (id BIGINT NOT NULL) USING ZYRONLAKE",
    )
    .await
    .expect("create");
    for id in 1..=3i64 {
        exec_dml(&server, &format!("INSERT INTO shipped VALUES ({})", id)).await;
    }

    let entry = server
        .catalog
        .get_table(schema_id, "shipped")
        .expect("entry");
    let paths = zyron_lake::LakePaths::new(server.disk_manager.data_dir(), entry.id.0);

    // What a follower on the same filesystem would read
    let local = zyron_lake::read_versions_after(&paths, 1, 64).expect("local read");
    assert!(local.len() >= 3, "the leader committed real versions");

    // What the leader publishes over the wire
    let (columns, rows) = stat_view(
        &server,
        "SELECT * FROM zyron_lake_log WHERE table_name = 'shipped' AND from_version = 1",
    )
    .expect("log view");
    let col = |name: &str| columns.iter().position(|c| c == name).expect(name);
    let published: Vec<(u64, String)> = rows
        .iter()
        .map(|r| {
            (
                r[col("version")].parse::<u64>().expect("version"),
                r[col("payload")].clone(),
            )
        })
        .collect();
    assert_eq!(published.len(), local.len());

    let decoded = zyron_lake::decode_log_rows(1, &published).expect("decode");
    assert_eq!(decoded.len(), local.len());
    for (wire, disk) in decoded.iter().zip(local.iter()) {
        assert_eq!(wire.version, disk.version);
        assert_eq!(wire.timestamp_us, disk.timestamp_us);
        assert_eq!(wire.operation, disk.operation);
        assert_eq!(
            wire.entries, disk.entries,
            "a version read over the wire must be the version on disk"
        );
    }

    // A follower asks from where it is, and gets only what it lacks
    let (_, rows) = stat_view(
        &server,
        "SELECT * FROM zyron_lake_log WHERE table_name = 'shipped' AND from_version = 3",
    )
    .expect("log view");
    let versions: Vec<u64> = rows
        .iter()
        .map(|r| r[col("version")].parse::<u64>().expect("version"))
        .collect();
    assert!(
        versions.iter().all(|v| *v > 3),
        "a follower is not sent versions it already applied, got {versions:?}"
    );

    // A truncated payload is refused rather than decoded as a shorter,
    // structurally valid commit
    let mut mangled = published.clone();
    let payload = mangled[0].1.clone();
    mangled[0].1 = payload[..payload.len() - 2].to_string();
    assert!(
        zyron_lake::decode_log_rows(1, &mangled).is_err(),
        "a truncated version file must not decode"
    );
}

/// Following is declared, and every way of declaring it wrong is refused
/// with a reason rather than accepted into a state that cannot work.
#[tokio::test]
async fn test_follow_is_declared_and_its_preconditions_are_enforced() {
    let (server, schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE replica (id BIGINT NOT NULL) USING ZYRONLAKE",
    )
    .await
    .expect("create");
    exec_ddl(&server, &mut session, "CREATE TABLE plain (id BIGINT)")
        .await
        .expect("create heap");

    // A peer this node was never told about is not followable. Accepting it
    // would be discovery by another name
    let unknown = exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE replica FOLLOW west.orders",
    )
    .await;
    let message = format!("{:?}", unknown.expect_err("unknown peer"));
    assert!(message.contains("CREATE PEER"), "{message}");

    exec_ddl(
        &server,
        &mut session,
        "CREATE PEER west ADDRESS '127.0.0.1:1'",
    )
    .await
    .expect("declare peer");

    // A heap table has no log to replay
    let heap = exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE plain FOLLOW west.orders",
    )
    .await;
    assert!(heap.is_err(), "a heap table has no log to follow");

    // Now it is legal, and the catalog records the leader
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE replica FOLLOW west.orders",
    )
    .await
    .expect("follow");
    let entry = server
        .catalog
        .get_table(schema_id, "replica")
        .expect("entry");
    assert_eq!(entry.lake.follows(), Some(("west", "orders")));

    // UNFOLLOW leaves it its own authority, holding what it applied
    exec_ddl(&server, &mut session, "ALTER TABLE replica UNFOLLOW")
        .await
        .expect("unfollow");
    let entry = server
        .catalog
        .get_table(schema_id, "replica")
        .expect("entry");
    assert_eq!(entry.lake.follows(), None);

    // A table already holding its own data has a second source of truth,
    // and replay assumes one
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE occupied (id BIGINT NOT NULL) USING ZYRONLAKE",
    )
    .await
    .expect("create");
    exec_dml(&server, "INSERT INTO occupied VALUES (1)").await;
    let occupied = exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE occupied FOLLOW west.orders",
    )
    .await;
    let message = format!("{:?}", occupied.expect_err("already has data"));
    assert!(message.contains("replays a leader"), "{message}");
}

/// A table name reaches a peer inside a query, so it is checked before it
/// is sent rather than trusted because it came from a catalog
#[tokio::test]
async fn test_a_table_name_a_peer_is_asked_for_is_checked() {
    let bad = zyron_wire::peer_probe::fetch_remote_versions(
        "127.0.0.1:1",
        "zyron",
        "zyron",
        "orders'; DROP TABLE users; --",
        0,
        16,
    )
    .await
    .expect_err("must be refused before any connection");
    let message = bad.to_string();
    assert!(message.contains("not a table name"), "{message}");

    // A legitimate name gets as far as the connection, which is where an
    // unreachable peer is reported
    let unreachable = zyron_wire::peer_probe::fetch_remote_versions(
        "127.0.0.1:1",
        "zyron",
        "zyron",
        "orders",
        0,
        16,
    )
    .await
    .expect_err("nothing listens there");
    assert!(unreachable.to_string().contains("127.0.0.1:1"));
}

// ---------------------------------------------------------------------------
// Lake version history views
// ---------------------------------------------------------------------------

/// Runs a SELECT against a virtual view exactly as the wire path does:
/// parse, read the supported clauses, build and narrow. Returns the column
/// names paired with the rows rendered as text.
fn stat_view(
    server: &Arc<ServerState>,
    sql: &str,
) -> Result<(Vec<String>, Vec<Vec<String>>), String> {
    let stmt = zyron_parser::parse(sql)
        .expect("parse")
        .into_iter()
        .next()
        .expect("one statement");
    let sel = match stmt {
        zyron_parser::Statement::Select(sel) => sel,
        other => return Err(format!("not a select: {other:?}")),
    };
    let name = match &sel.from[0] {
        zyron_parser::TableRef::Table { name, .. } => name.clone(),
        other => return Err(format!("not a plain table ref: {other:?}")),
    };
    assert!(
        zyron_wire::stat_views::is_stat_view(&name),
        "{name} must be a registered view"
    );
    let filters =
        zyron_wire::stat_views::parse_stat_view_query(&name, &sel).map_err(|e| e.to_string())?;
    let (fields, rows) = zyron_wire::stat_views::query_stat_view(&name, server, &filters)
        .map_err(|e| e.to_string())?
        .expect("a registered view builds");
    let names = fields.iter().map(|f| f.name.clone()).collect();
    let rows = rows
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|cell| {
                    cell.map(|b| String::from_utf8_lossy(&b).into_owned())
                        .unwrap_or_default()
                })
                .collect()
        })
        .collect();
    Ok((names, rows))
}

fn column_of(names: &[String], column: &str) -> usize {
    names
        .iter()
        .position(|n| n == column)
        .unwrap_or_else(|| panic!("view has no column {column}"))
}

/// Builds a lake table with four versions: create, two appends, one delete.
async fn history_fixture(server: &Arc<ServerState>, session: &mut Option<Session>, name: &str) {
    exec_ddl(
        server,
        session,
        &format!("CREATE TABLE {name} (id BIGINT NOT NULL, tag TEXT) USING ZYRONLAKE"),
    )
    .await
    .expect("create lake table");
    exec_dml(
        server,
        &format!("INSERT INTO {name} VALUES (1, 'a'), (2, 'b')"),
    )
    .await;
    exec_dml(server, &format!("INSERT INTO {name} VALUES (10, 'c')")).await;
    exec_dml(server, &format!("DELETE FROM {name} WHERE id < 5")).await;
}

#[tokio::test]
async fn test_table_history_view_reports_every_commit() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    history_fixture(&server, &mut session, "h").await;

    let (names, rows) = stat_view(
        &server,
        "SELECT * FROM zyron_table_history WHERE table_name = 'h'",
    )
    .expect("history");
    let version = column_of(&names, "version");
    let operation = column_of(&names, "operation");
    assert_eq!(rows.len(), 4, "create, two appends, one delete");
    assert_eq!(rows[0][version], "4");
    assert_eq!(rows[0][operation], "DELETE");
    assert_eq!(rows[3][version], "1");
    assert_eq!(rows[3][operation], "SCHEMA CHANGE");
    let rows_added = column_of(&names, "rows_added");
    assert_eq!(rows[2][rows_added], "2");

    // LIMIT bounds the walk rather than being dropped
    let (_, limited) = stat_view(
        &server,
        "SELECT * FROM zyron_table_history WHERE table_name = 'h' LIMIT 2",
    )
    .expect("history");
    assert_eq!(limited.len(), 2);
    assert_eq!(limited[0][version], "4");

    // OFFSET skips from the newest end
    let (_, offset) = stat_view(
        &server,
        "SELECT * FROM zyron_table_history WHERE table_name = 'h' LIMIT 1 OFFSET 1",
    )
    .expect("history");
    assert_eq!(offset.len(), 1);
    assert_eq!(offset[0][version], "3");
}

/// The WHERE clause used to be dropped, so every view answered a narrowed
/// question with every row it had.
#[tokio::test]
async fn test_stat_view_where_clause_is_honored_not_dropped() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    history_fixture(&server, &mut session, "one").await;
    history_fixture(&server, &mut session, "two").await;

    let (_, both) = stat_view(&server, "SELECT * FROM zyron_table_history").expect("history");
    assert_eq!(both.len(), 8, "two tables of four versions each");

    let (names, only_one) = stat_view(
        &server,
        "SELECT * FROM zyron_table_history WHERE table_name = 'one'",
    )
    .expect("history");
    assert_eq!(only_one.len(), 4);
    let table = column_of(&names, "table_name");
    assert!(only_one.iter().all(|r| r[table] == "one"));

    // A filter on a second column narrows further
    let (_, appends) = stat_view(
        &server,
        "SELECT * FROM zyron_table_history WHERE table_name = 'one' AND operation = 'APPEND'",
    )
    .expect("history");
    assert_eq!(appends.len(), 2);

    // A table this node does not have matches nothing
    let (_, none) = stat_view(
        &server,
        "SELECT * FROM zyron_table_history WHERE table_name = 'absent'",
    )
    .expect("history");
    assert!(none.is_empty());
}

/// A clause the view cannot apply must be refused, never silently ignored.
#[tokio::test]
async fn test_stat_view_refuses_clauses_it_cannot_apply() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    history_fixture(&server, &mut session, "h").await;

    let err = stat_view(
        &server,
        "SELECT * FROM zyron_table_history WHERE table_name LIKE 'h%'",
    )
    .expect_err("LIKE is not a supported filter");
    assert!(err.contains("column = literal"), "{err}");

    let err = stat_view(
        &server,
        "SELECT * FROM zyron_table_history ORDER BY version",
    )
    .expect_err("ORDER BY is not applied");
    assert!(err.contains("ORDER BY"), "{err}");

    let err = stat_view(
        &server,
        "SELECT * FROM zyron_table_history WHERE version > 2",
    )
    .expect_err("range filters are not supported");
    assert!(err.contains("column = literal"), "{err}");
}

#[tokio::test]
async fn test_version_scoped_views_read_the_past() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    history_fixture(&server, &mut session, "h").await;

    // Files at version 3, before the delete dropped one
    let (names, files) = stat_view(
        &server,
        "SELECT * FROM zyron_version_files WHERE table_name = 'h' AND version = 3",
    )
    .expect("files");
    assert_eq!(files.len(), 2);
    let row_count = column_of(&names, "row_count");
    let total: u64 = files
        .iter()
        .map(|r| r[row_count].parse::<u64>().unwrap())
        .sum();
    assert_eq!(total, 3);

    // The latest version is the default, and the delete removed a file
    let (_, latest) = stat_view(
        &server,
        "SELECT * FROM zyron_version_files WHERE table_name = 'h'",
    )
    .expect("files");
    assert_eq!(latest.len(), 1);

    // Schema at a past version
    let (names, schema) = stat_view(
        &server,
        "SELECT * FROM zyron_schema_at_version WHERE table_name = 'h' AND version = 1",
    )
    .expect("schema");
    assert_eq!(schema.len(), 2);
    let column_name = column_of(&names, "column_name");
    assert_eq!(schema[0][column_name], "id");
    assert_eq!(schema[1][column_name], "tag");

    // Details carry the operation and the recorded delete predicate
    let (names, details) = stat_view(
        &server,
        "SELECT * FROM zyron_version_details WHERE table_name = 'h' AND version = 4",
    )
    .expect("details");
    assert_eq!(details.len(), 1);
    let operation = column_of(&names, "operation");
    assert_eq!(details[0][operation], "DELETE");
    let removed = column_of(&names, "files_removed");
    assert_eq!(details[0][removed], "1");

    // Lineage follows the read_version links back to version one
    let (names, lineage) = stat_view(
        &server,
        "SELECT * FROM zyron_version_lineage WHERE table_name = 'h' AND version = 4",
    )
    .expect("lineage");
    let ancestor = column_of(&names, "ancestor_version");
    let chain: Vec<&str> = lineage.iter().map(|r| r[ancestor].as_str()).collect();
    assert_eq!(chain, vec!["4", "3", "2", "1"]);
}

#[tokio::test]
async fn test_diff_versions_view_needs_both_endpoints() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    history_fixture(&server, &mut session, "h").await;

    let err = stat_view(
        &server,
        "SELECT * FROM zyron_diff_versions WHERE table_name = 'h'",
    )
    .expect_err("a diff without endpoints has no answer");
    assert!(err.contains("from_version"), "{err}");

    let (names, rows) = stat_view(
        &server,
        "SELECT * FROM zyron_diff_versions WHERE table_name = 'h' AND from_version = 2 AND to_version = 3",
    )
    .expect("diff");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0][column_of(&names, "files_added")], "1");
    assert_eq!(rows[0][column_of(&names, "rows_added")], "1");
    assert_eq!(rows[0][column_of(&names, "files_removed")], "0");

    let (names, rows) = stat_view(
        &server,
        "SELECT * FROM zyron_diff_versions WHERE table_name = 'h' AND from_version = 3 AND to_version = 4",
    )
    .expect("diff");
    assert_eq!(rows[0][column_of(&names, "files_removed")], "1");
    assert_eq!(rows[0][column_of(&names, "rows_removed")], "2");
}

// ---------------------------------------------------------------------------
// Change records derived from the lake log
// ---------------------------------------------------------------------------

/// Decodes one NSM row of (BIGINT, TEXT) the way a CDC consumer does.
fn decode_id_tag(row: &[u8]) -> (Option<i64>, Option<String>) {
    let null_bitmap = row[0];
    let mut off = 1usize;
    let id = if null_bitmap & 1 != 0 {
        off += 8;
        None
    } else {
        let mut b = [0u8; 8];
        b.copy_from_slice(&row[off..off + 8]);
        off += 8;
        Some(i64::from_le_bytes(b))
    };
    let len = u32::from_le_bytes([row[off], row[off + 1], row[off + 2], row[off + 3]]) as usize;
    off += 4;
    let tag = if null_bitmap & 2 != 0 {
        None
    } else {
        Some(String::from_utf8_lossy(&row[off..off + len]).into_owned())
    };
    (id, tag)
}

fn open_log(
    server: &Arc<ServerState>,
    entry: &zyron_catalog::TableEntry,
) -> Arc<zyron_lake::TransactionLog> {
    let paths = LakePaths::new(server.disk_manager.data_dir(), entry.id.0);
    zyron_lake::TransactionLog::lookup_shared(&paths).expect("the table's log is registered")
}

#[tokio::test]
async fn test_lake_change_records_come_from_the_log_with_no_capture_file() {
    let (server, schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE c (id BIGINT NOT NULL, tag TEXT) USING ZYRONLAKE",
    )
    .await
    .expect("create lake table");
    exec_dml(&server, "INSERT INTO c VALUES (1, 'a'), (2, 'b')").await;
    exec_dml(&server, "INSERT INTO c VALUES (3, 'c')").await;

    let entry = server.catalog.get_table(schema_id, "c").expect("entry");
    let log = open_log(&server, &entry);
    let records = zyron_wire::lake_changes::lake_change_records(&log, &entry, 1, u64::MAX)
        .expect("change records");

    assert_eq!(records.len(), 3, "three inserted rows");
    assert!(
        records
            .iter()
            .all(|r| r.change_type == zyron_cdc::ChangeType::Insert)
    );
    assert_eq!(records[0].table_id, entry.id.0);
    assert_eq!(records[0].commit_version, 2);
    assert_eq!(records[2].commit_version, 3);

    // Row images decode back to what was inserted
    let decoded: Vec<(Option<i64>, Option<String>)> =
        records.iter().map(|r| decode_id_tag(&r.row_data)).collect();
    assert!(decoded.contains(&(Some(1), Some("a".into()))));
    assert!(decoded.contains(&(Some(2), Some("b".into()))));
    assert!(decoded.contains(&(Some(3), Some("c".into()))));

    // The last record of each version closes it
    assert!(!records[0].is_last_in_txn);
    assert!(records[1].is_last_in_txn, "version two ends here");
    assert!(records[2].is_last_in_txn, "version three ends here");

    // A version range narrows the feed without re-reading the earlier ones
    let later = zyron_wire::lake_changes::lake_change_records(&log, &entry, 3, u64::MAX)
        .expect("change records");
    assert_eq!(later.len(), 1);
    assert_eq!(later[0].commit_version, 3);

    // Nothing was captured into a change file, the log is the record
    assert!(
        server.cdc_registry.is_none()
            || server
                .cdc_registry
                .as_ref()
                .and_then(|r| r.get_feed(entry.id.0))
                .is_none(),
        "a lake table needs no capture file"
    );
}

#[tokio::test]
async fn test_lake_update_and_delete_produce_paired_change_records() {
    let (server, schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE u (id BIGINT NOT NULL, tag TEXT) USING ZYRONLAKE",
    )
    .await
    .expect("create lake table");
    exec_dml(&server, "INSERT INTO u VALUES (1, 'a'), (2, 'b'), (9, 'z')").await;
    exec_dml(&server, "UPDATE u SET tag = 'x' WHERE id < 5").await;
    exec_dml(&server, "DELETE FROM u WHERE id = 9").await;

    let entry = server.catalog.get_table(schema_id, "u").expect("entry");
    let log = open_log(&server, &entry);

    // The update commit pairs pre and post images under one version
    let update =
        zyron_wire::lake_changes::lake_change_records(&log, &entry, 3, 3).expect("change records");
    let pre: Vec<_> = update
        .iter()
        .filter(|r| r.change_type == zyron_cdc::ChangeType::UpdatePreimage)
        .collect();
    let post: Vec<_> = update
        .iter()
        .filter(|r| r.change_type == zyron_cdc::ChangeType::UpdatePostimage)
        .collect();
    assert_eq!(pre.len(), 2, "two old images");
    assert_eq!(post.len(), 2, "two new images");
    assert!(
        post.iter()
            .all(|r| decode_id_tag(&r.row_data).1 == Some("x".into()))
    );
    assert!(
        pre.iter()
            .all(|r| decode_id_tag(&r.row_data).1 != Some("x".into()))
    );
    assert!(update.iter().all(|r| r.commit_version == 3));
    assert_eq!(
        update.iter().filter(|r| r.is_last_in_txn).count(),
        1,
        "exactly one record closes the version"
    );

    // The delete commit reports the removed row, not the surviving ones
    let delete =
        zyron_wire::lake_changes::lake_change_records(&log, &entry, 4, 4).expect("change records");
    assert_eq!(delete.len(), 1);
    assert_eq!(delete[0].change_type, zyron_cdc::ChangeType::Delete);
    assert_eq!(decode_id_tag(&delete[0].row_data).0, Some(9));
}

#[tokio::test]
async fn test_lake_branches_view_lists_branches_and_their_lead() {
    let (server, schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE b (id BIGINT NOT NULL) USING ZYRONLAKE",
    )
    .await
    .expect("create lake table");
    exec_dml(&server, "INSERT INTO b VALUES (1), (2)").await;

    let entry = server.catalog.get_table(schema_id, "b").expect("entry");
    let log = open_log(&server, &entry);
    zyron_lake::create_branch(&log, "staging", None, 4242).expect("create branch");

    let (names, rows) = stat_view(&server, "SELECT * FROM zyron_lake_branches").expect("branches");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0][column_of(&names, "table_name")], "b");
    assert_eq!(rows[0][column_of(&names, "branch_name")], "staging");
    assert_eq!(rows[0][column_of(&names, "base_version")], "2");
    assert_eq!(rows[0][column_of(&names, "commits_ahead")], "0");
    assert_eq!(rows[0][column_of(&names, "created_us")], "4242");

    // A branch commit moves it ahead of its fork point
    let branch = zyron_lake::open_branch(log.paths(), "staging").expect("open");
    zyron_lake::append_rows(
        &branch,
        zyron_lake::CommitAttempt {
            operation: zyron_lake::OperationKind::Append,
            db_txn_id: 0,
            commit_lsn: 0,
            timestamp_us: 1,
            read_predicate: None,
            read_version: 0,
            audit: None,
            deadline: None,
        },
        entry.id.0 as u64,
        &[zyron_lake::ColumnData {
            column_id: entry.columns[0].id.0 as u32,
            cells: vec![Some(7i64.to_le_bytes().to_vec())],
        }],
    )
    .expect("branch append");

    let (names, rows) = stat_view(
        &server,
        "SELECT * FROM zyron_lake_branches WHERE branch_name = 'staging'",
    )
    .expect("branches");
    assert_eq!(rows[0][column_of(&names, "commits_ahead")], "1");
    assert_eq!(rows[0][column_of(&names, "head_version")], "3");
}

#[tokio::test]
async fn test_lake_maintenance_procedures_run_over_the_log() {
    let (server, schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE mt (id BIGINT NOT NULL) USING ZYRONLAKE",
    )
    .await
    .expect("create lake table");
    exec_dml(&server, "INSERT INTO mt VALUES (1), (2)").await;
    exec_dml(&server, "INSERT INTO mt VALUES (3)").await;

    let rows = call_rows(&server, &mut session, "CALL zyronlake_validate('mt')").await;
    assert_eq!(metric(&rows, "healthy"), Some("true".to_string()));
    assert_eq!(metric(&rows, "files_checked"), Some("2".to_string()));

    // A repair pass on a healthy table changes nothing
    let rows = call_rows(&server, &mut session, "CALL zyronlake_repair('mt')").await;
    assert_eq!(metric(&rows, "files_removed"), Some("0".to_string()));
    assert_eq!(metric(&rows, "committed_version"), Some(String::new()));

    // Orphan cleanup with everything reachable removes nothing
    let rows = call_rows(
        &server,
        &mut session,
        "CALL zyronlake_cleanup_orphans('mt')",
    )
    .await;
    assert_eq!(metric(&rows, "files_removed"), Some("0".to_string()));

    // A heap table is refused rather than silently reporting nothing
    exec_ddl(&server, &mut session, "CREATE TABLE hp (id BIGINT)")
        .await
        .expect("create heap table");
    let err = exec_ddl(&server, &mut session, "CALL zyronlake_validate('hp')")
        .await
        .expect_err("a heap table has no lake log");
    assert!(err.contains("heap table"), "{err}");
    let _ = schema_id;
}

/// Runs a CALL and returns its metric rows.
async fn call_rows(
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
    sql: &str,
) -> Vec<Vec<String>> {
    let stmt = zyron_parser::parse(sql)
        .expect("parse")
        .into_iter()
        .next()
        .expect("one statement");
    let mut txn_opt: Option<zyron_storage::txn::Transaction> = None;
    let mut active_branch: Option<String> = None;
    match zyron_wire::ddl_dispatch::try_handle_ddl_utility(
        &stmt,
        server,
        session,
        &mut txn_opt,
        &mut active_branch,
        sql,
    )
    .await
    {
        Some(Ok(zyron_wire::ddl_dispatch::DdlResult::Rows { rows, .. })) => rows,
        other => panic!("expected rows from {sql}, got {other:?}"),
    }
}

fn metric(rows: &[Vec<String>], name: &str) -> Option<String> {
    rows.iter().find(|r| r[0] == name).map(|r| r[1].clone())
}

#[tokio::test]
async fn test_branch_ddl_routes_to_the_lake_log() {
    let (server, schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE bt (id BIGINT NOT NULL) USING ZYRONLAKE",
    )
    .await
    .expect("create lake table");
    exec_dml(&server, "INSERT INTO bt VALUES (1), (2)").await;

    exec_ddl(&server, &mut session, "CREATE BRANCH work ON bt")
        .await
        .expect("create branch");
    let entry = server.catalog.get_table(schema_id, "bt").expect("entry");
    let log = open_log(&server, &entry);
    let branches = zyron_lake::list_branches(log.paths()).expect("list");
    assert_eq!(branches.len(), 1);
    assert_eq!(branches[0].name, "work");
    assert_eq!(branches[0].base_version, 2);

    // A branch commit stays off main, then the merge brings it over
    let branch = zyron_lake::open_branch(log.paths(), "work").expect("open");
    zyron_lake::append_rows(
        &branch,
        zyron_lake::CommitAttempt {
            operation: zyron_lake::OperationKind::Append,
            db_txn_id: 0,
            commit_lsn: 0,
            timestamp_us: 1,
            read_predicate: None,
            read_version: 0,
            audit: None,
            deadline: None,
        },
        entry.id.0 as u64,
        &[zyron_lake::ColumnData {
            column_id: entry.columns[0].id.0 as u32,
            cells: vec![Some(9i64.to_le_bytes().to_vec())],
        }],
    )
    .expect("branch append");
    assert_eq!(query_rows(&server, "SELECT * FROM bt").await, 2);

    let rows = call_rows(
        &server,
        &mut session,
        "MERGE BRANCH work INTO main FOR TABLE bt",
    )
    .await;
    assert_eq!(metric(&rows, "files_added"), Some("1".to_string()));
    assert_eq!(query_rows(&server, "SELECT * FROM bt").await, 3);

    exec_ddl(&server, &mut session, "DROP BRANCH work ON bt")
        .await
        .expect("drop branch");
    assert!(
        zyron_lake::list_branches(log.paths())
            .expect("list")
            .is_empty()
    );
    exec_ddl(&server, &mut session, "DROP BRANCH IF EXISTS work ON bt")
        .await
        .expect("drop is idempotent with IF EXISTS");

    // A heap table is refused, database-wide branches take no ON clause
    exec_ddl(&server, &mut session, "CREATE TABLE hb (id BIGINT)")
        .await
        .expect("create heap table");
    let err = exec_ddl(&server, &mut session, "CREATE BRANCH nope ON hb")
        .await
        .expect_err("a heap table has no lake log");
    assert!(err.contains("heap table"), "{err}");
}

/// A branch is an alternate log head, so naming one in FROM reads that head
/// while main is unchanged. This is the read side of CREATE BRANCH: without
/// the FROM qualifier a branch could be created and written but never queried.
#[tokio::test]
async fn test_select_in_branch_reads_the_branch_head() {
    let (server, schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE ib (id BIGINT NOT NULL) USING ZYRONLAKE",
    )
    .await
    .expect("create lake table");
    exec_dml(&server, "INSERT INTO ib VALUES (1), (2)").await;

    exec_ddl(&server, &mut session, "CREATE BRANCH work ON ib")
        .await
        .expect("create branch");
    let entry = server.catalog.get_table(schema_id, "ib").expect("entry");
    let log = open_log(&server, &entry);
    let branch = zyron_lake::open_branch(log.paths(), "work").expect("open");
    zyron_lake::append_rows(
        &branch,
        zyron_lake::CommitAttempt {
            operation: zyron_lake::OperationKind::Append,
            db_txn_id: 0,
            commit_lsn: 0,
            timestamp_us: 1,
            read_predicate: None,
            read_version: 0,
            audit: None,
            deadline: None,
        },
        entry.id.0 as u64,
        &[zyron_lake::ColumnData {
            column_id: entry.columns[0].id.0 as u32,
            cells: vec![Some(9i64.to_le_bytes().to_vec())],
        }],
    )
    .expect("branch append");

    // Main is untouched by the branch commit
    assert_eq!(query_rows(&server, "SELECT id FROM ib").await, 2);

    let branch_values = query_values(&server, "SELECT id FROM ib IN BRANCH 'work'").await;
    let mut ids: Vec<i64> = branch_values
        .iter()
        .map(|row| match &row[0] {
            zyron_executor::column::ScalarValue::Int64(v) => *v,
            other => panic!("expected an integer id, got {other:?}"),
        })
        .collect();
    ids.sort_unstable();
    assert_eq!(
        ids,
        vec![1, 2, 9],
        "the branch head carries its own commit on top of the fork point"
    );

    // The qualifier reads on either side of an alias
    assert_eq!(
        query_rows(&server, "SELECT b.id FROM ib AS b IN BRANCH 'work'").await,
        3
    );

    let err = query_error(&server, "SELECT id FROM ib IN BRANCH 'missing'").await;
    assert!(
        err.contains("missing"),
        "an unknown branch names itself: {err}"
    );
}

/// The write side of a lake branch: INSERT, UPDATE and DELETE all land on
/// the branch head and leave main where it was, and the merge brings the
/// whole divergence over at once.
#[tokio::test]
async fn test_branch_writes_land_on_the_branch_head_and_leave_main_alone() {
    let (server, schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE bw (id BIGINT NOT NULL, tag TEXT) USING ZYRONLAKE",
    )
    .await
    .expect("create lake table");
    exec_dml(&server, "INSERT INTO bw VALUES (1, 'a'), (2, 'b')").await;
    exec_ddl(&server, &mut session, "CREATE BRANCH work ON bw")
        .await
        .expect("create branch");

    // Three write shapes on the branch
    run_on_branch(&server, "INSERT INTO bw VALUES (3, 'c')", "work")
        .await
        .expect("branch insert");
    run_on_branch(&server, "UPDATE bw SET tag = 'z' WHERE id = 1", "work")
        .await
        .expect("branch update");
    run_on_branch(&server, "DELETE FROM bw WHERE id = 2", "work")
        .await
        .expect("branch delete");

    // Main never saw any of it
    let main: Vec<(i64, String)> = tagged(&query_values(&server, "SELECT id, tag FROM bw").await);
    assert_eq!(
        main,
        vec![(1, "a".to_string()), (2, "b".to_string())],
        "main is exactly what it was before the branch wrote"
    );

    // The branch sees all three, whether it is named by the query or bound
    // to the session
    let expected = vec![(1, "z".to_string()), (3, "c".to_string())];
    let qualified = tagged(&query_values(&server, "SELECT id, tag FROM bw IN BRANCH 'work'").await);
    assert_eq!(qualified, expected);
    let session_bound = tagged(
        &run_on_branch(&server, "SELECT id, tag FROM bw", "work")
            .await
            .expect("branch read"),
    );
    assert_eq!(session_bound, expected);

    // And the merge carries the divergence to main in one commit
    let rows = call_rows(
        &server,
        &mut session,
        "MERGE BRANCH work INTO main FOR TABLE bw",
    )
    .await;
    assert!(metric(&rows, "merged_version").is_some());
    let merged = tagged(&query_values(&server, "SELECT id, tag FROM bw").await);
    assert_eq!(merged, expected, "main now reads what the branch built");

    let entry = server.catalog.get_table(schema_id, "bw").expect("entry");
    let log = open_log(&server, &entry);
    assert_eq!(
        zyron_lake::branch_info(log.paths(), "work")
            .expect("info")
            .head_version
            - zyron_lake::branch_info(log.paths(), "work")
                .expect("info")
                .base_version,
        3,
        "one branch version per write statement"
    );
}

/// A session on a branch writes a table the branch never forked by forking
/// it there first, so a database-wide branch reaches a lake table created
/// after it without the write silently landing on main.
#[tokio::test]
async fn test_a_session_branch_forks_a_lake_table_it_has_not_touched() {
    let (server, schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE lf (id BIGINT NOT NULL) USING ZYRONLAKE",
    )
    .await
    .expect("create lake table");
    exec_dml(&server, "INSERT INTO lf VALUES (1)").await;
    exec_ddl(&server, &mut session, "CREATE BRANCH late ON lf")
        .await
        .expect("create branch");

    // A second table the branch has never seen
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE lg (id BIGINT NOT NULL) USING ZYRONLAKE",
    )
    .await
    .expect("create second lake table");
    exec_dml(&server, "INSERT INTO lg VALUES (10)").await;

    let entry = server.catalog.get_table(schema_id, "lg").expect("entry");
    let log = open_log(&server, &entry);
    assert!(
        zyron_lake::branch_info(log.paths(), "late").is_err(),
        "the branch has no head on this table yet"
    );

    // Reading through it sees main, because the branch has nothing of its own
    assert_eq!(
        run_on_branch(&server, "SELECT id FROM lg", "late")
            .await
            .expect("branch read")
            .len(),
        1
    );

    // Writing forks it at the current version rather than landing on main
    run_on_branch(&server, "INSERT INTO lg VALUES (11)", "late")
        .await
        .expect("branch insert forks the table");
    let info = zyron_lake::branch_info(log.paths(), "late").expect("the head now exists");
    assert_eq!(info.base_version, 2, "forked at the version it found");
    assert_eq!(info.head_version, 3);
    assert_eq!(query_rows(&server, "SELECT id FROM lg").await, 1, "main");
    assert_eq!(
        run_on_branch(&server, "SELECT id FROM lg", "late")
            .await
            .expect("branch read")
            .len(),
        2
    );
}

/// A branch write is pending until its transaction commits, so an abort
/// leaves the branch exactly where it was.
#[tokio::test]
async fn test_an_aborted_branch_write_leaves_the_branch_head_unmoved() {
    let (server, schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE ba (id BIGINT NOT NULL, UNIQUE (id)) USING ZYRONLAKE",
    )
    .await
    .expect("create lake table");
    exec_dml(&server, "INSERT INTO ba VALUES (1)").await;
    exec_ddl(&server, &mut session, "CREATE BRANCH work ON ba")
        .await
        .expect("create branch");

    run_on_branch(&server, "INSERT INTO ba VALUES (2)", "work")
        .await
        .expect("branch insert");
    let entry = server.catalog.get_table(schema_id, "ba").expect("entry");
    let log = open_log(&server, &entry);
    let before = zyron_lake::branch_info(log.paths(), "work").expect("info");

    // Uniqueness reads the branch head, so the key the branch just added
    // collides there while main has never seen it
    let err = run_on_branch(&server, "INSERT INTO ba VALUES (2)", "work")
        .await
        .expect_err("the branch's own key must collide");
    assert!(err.to_string().contains("unique constraint"), "{err}");
    let after = zyron_lake::branch_info(log.paths(), "work").expect("info");
    assert_eq!(
        after.head_version, before.head_version,
        "the refused write left no version behind"
    );

    // Main never had the key, so the same insert lands there
    exec_dml(&server, "INSERT INTO ba VALUES (2)").await;
    assert_eq!(query_rows(&server, "SELECT id FROM ba").await, 2);
}

/// Pairs each result row as (id, tag), so a branch's contents compare as
/// values rather than as a row count.
fn tagged(rows: &[Vec<zyron_executor::column::ScalarValue>]) -> Vec<(i64, String)> {
    use zyron_executor::column::ScalarValue;
    let mut out: Vec<(i64, String)> = rows
        .iter()
        .map(|row| {
            let id = match &row[0] {
                ScalarValue::Int64(v) => *v,
                other => panic!("expected an integer id, got {other:?}"),
            };
            let tag = match &row[1] {
                ScalarValue::Utf8(s) => s.to_string(),
                ScalarValue::Null => String::new(),
                other => panic!("expected a text tag, got {other:?}"),
            };
            (id, tag)
        })
        .collect();
    out.sort();
    out
}

#[tokio::test]
async fn test_branch_ddl_forks_a_past_version() {
    let (server, schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE pv (id BIGINT NOT NULL) USING ZYRONLAKE",
    )
    .await
    .expect("create lake table");
    exec_dml(&server, "INSERT INTO pv VALUES (1)").await;
    exec_dml(&server, "INSERT INTO pv VALUES (2)").await;

    exec_ddl(
        &server,
        &mut session,
        "CREATE BRANCH old ON pv FROM VERSION 2",
    )
    .await
    .expect("create branch at a past version");
    let entry = server.catalog.get_table(schema_id, "pv").expect("entry");
    let log = open_log(&server, &entry);
    let info = zyron_lake::branch_info(log.paths(), "old").expect("info");
    assert_eq!(info.base_version, 2);

    let branch = zyron_lake::open_branch(log.paths(), "old").expect("open");
    let rows: u64 = branch
        .latest_manifest()
        .expect("manifest")
        .entries
        .iter()
        .map(|e| e.row_count)
        .sum();
    assert_eq!(
        rows, 1,
        "the branch sees the table as it was at version two"
    );

    let err = exec_ddl(
        &server,
        &mut session,
        "CREATE BRANCH bad ON pv FROM VERSION 99",
    )
    .await
    .expect_err("a fork point must exist");
    assert!(
        err.contains("VersionNotFound") || err.contains("99"),
        "{err}"
    );
}

#[tokio::test]
async fn test_unique_enforced_by_default_on_lake_table() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE uq (id BIGINT NOT NULL PRIMARY KEY, tag TEXT) USING ZYRONLAKE",
    )
    .await
    .expect("create lake table");
    exec_dml(&server, "INSERT INTO uq VALUES (1, 'a'), (2, 'b')").await;
    assert_eq!(query_rows(&server, "SELECT * FROM uq").await, 2);

    // A key already stored is refused, and nothing lands
    let err = exec_dml_result(&server, "INSERT INTO uq VALUES (3, 'c'), (2, 'dup')")
        .await
        .expect_err("a stored key must be refused");
    assert!(err.contains("unique constraint"), "{err}");
    assert_eq!(query_rows(&server, "SELECT * FROM uq").await, 2);

    // Two rows of one statement colliding with each other is also refused
    let err = exec_dml_result(&server, "INSERT INTO uq VALUES (7, 'x'), (7, 'y')")
        .await
        .expect_err("a batch-local duplicate must be refused");
    assert!(err.contains("same key"), "{err}");
    assert_eq!(query_rows(&server, "SELECT * FROM uq").await, 2);

    // A fresh key still inserts
    exec_dml(&server, "INSERT INTO uq VALUES (3, 'c')").await;
    assert_eq!(query_rows(&server, "SELECT * FROM uq").await, 3);
}

/// Runs a DML statement and returns its error rather than panicking.
async fn exec_dml_result(server: &Arc<ServerState>, sql: &str) -> Result<(), String> {
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
    .map_err(|e| e.to_string())?;
    let mut txn = server
        .txn_manager
        .begin(zyron_storage::txn::IsolationLevel::ReadCommitted)
        .expect("begin");
    let snapshot = txn.snapshot.clone();
    let txn_id = txn.txn_id;
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
    let ctx = Arc::new(ctx);
    let result = zyron_executor::execute(plan, &ctx).await;
    match result {
        Ok(_) => {
            server.txn_manager.commit(&mut txn).await.expect("commit");
            let logs =
                zyron_lake::publish_txn(server.disk_manager.data_dir(), txn_id).expect("publish");
            zyron_wire::connection::refresh_lake_stats(server, &logs);
            Ok(())
        }
        Err(e) => {
            let logs = zyron_lake::abandon_txn(server.disk_manager.data_dir(), txn_id);
            zyron_wire::connection::refresh_lake_stats(server, &logs);
            let _ = server.txn_manager.abort(&mut txn);
            Err(e.to_string())
        }
    }
}

#[tokio::test]
async fn test_not_enforced_unique_is_recorded_but_never_checked() {
    let (server, schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE ne (id BIGINT NOT NULL, tag TEXT, UNIQUE (id) NOT ENFORCED) USING ZYRONLAKE",
    )
    .await
    .expect("create lake table");

    // The declaration reaches the catalog, where the planner can read it
    let entry = server.catalog.get_table(schema_id, "ne").expect("entry");
    let constraint = entry
        .constraints
        .iter()
        .find(|c| c.constraint_type == zyron_catalog::schema::ConstraintType::Unique)
        .expect("the constraint is recorded");
    assert!(!constraint.enforced);
    assert_eq!(constraint.columns.len(), 1);

    // And the write path does not check it
    exec_dml(&server, "INSERT INTO ne VALUES (1, 'a')").await;
    exec_dml(&server, "INSERT INTO ne VALUES (1, 'b')").await;
    assert_eq!(query_rows(&server, "SELECT * FROM ne").await, 2);

    // The same table with an enforced constraint refuses the duplicate
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE en (id BIGINT NOT NULL, tag TEXT, UNIQUE (id)) USING ZYRONLAKE",
    )
    .await
    .expect("create lake table");
    exec_dml(&server, "INSERT INTO en VALUES (1, 'a')").await;
    let err = exec_dml_result(&server, "INSERT INTO en VALUES (1, 'b')")
        .await
        .expect_err("enforced is the default");
    assert!(err.contains("unique constraint"), "{err}");
}

#[tokio::test]
async fn test_foreign_keys_are_enforced_on_lake_tables_in_both_directions() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    // A lake parent and a lake child
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE lp (k BIGINT NOT NULL PRIMARY KEY) USING ZYRONLAKE",
    )
    .await
    .expect("create lake parent");
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE lc (id BIGINT NOT NULL, k BIGINT, FOREIGN KEY (k) REFERENCES lp(k)) USING ZYRONLAKE",
    )
    .await
    .expect("create lake child");
    exec_dml(&server, "INSERT INTO lp VALUES (1), (2)").await;

    // A referenced key that exists inserts
    exec_dml(&server, "INSERT INTO lc VALUES (10, 1)").await;
    assert_eq!(query_rows(&server, "SELECT * FROM lc").await, 1);

    // One that does not is refused, and nothing lands
    let err = exec_dml_result(&server, "INSERT INTO lc VALUES (11, 99)")
        .await
        .expect_err("an absent parent key must be refused");
    assert!(err.contains("foreign key"), "{err}");
    assert_eq!(query_rows(&server, "SELECT * FROM lc").await, 1);

    // A NULL reference is not checked, MATCH SIMPLE
    exec_dml(&server, "INSERT INTO lc VALUES (12, NULL)").await;
    assert_eq!(query_rows(&server, "SELECT * FROM lc").await, 2);

    // A heap child referencing the lake parent reads the same probe
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE hc (id BIGINT NOT NULL, k BIGINT, FOREIGN KEY (k) REFERENCES lp(k))",
    )
    .await
    .expect("create heap child");
    exec_dml(&server, "INSERT INTO hc VALUES (20, 2)").await;
    let err = exec_dml_result(&server, "INSERT INTO hc VALUES (21, 77)")
        .await
        .expect_err("a heap child must see the lake parent's keys");
    assert!(err.contains("foreign key"), "{err}");
}

#[tokio::test]
async fn test_fk_violation_quarantine_does_not_abort_bulk_load() {
    let (server, schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE qp (k BIGINT NOT NULL PRIMARY KEY) USING ZYRONLAKE",
    )
    .await
    .expect("create lake parent");
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE qc (id BIGINT NOT NULL, k BIGINT, \
         FOREIGN KEY (k) REFERENCES qp(k) ON VIOLATION QUARANTINE) USING ZYRONLAKE",
    )
    .await
    .expect("create lake child");
    exec_dml(&server, "INSERT INTO qp VALUES (1), (2)").await;

    // The companion table is provisioned by the declaration, not by the
    // first violation
    let child = server.catalog.get_table(schema_id, "qc").expect("entry");
    let fk = child
        .constraints
        .iter()
        .find(|c| c.constraint_type == zyron_catalog::schema::ConstraintType::ForeignKey)
        .expect("the constraint is recorded");
    assert_eq!(
        fk.on_violation,
        zyron_catalog::schema::ConstraintViolationAction::Quarantine
    );
    let quarantine_id = fk.quarantine_table_id.expect("a quarantine table exists");

    // A bulk load with good and bad rows lands the good ones and finishes
    exec_dml(
        &server,
        "INSERT INTO qc VALUES (10, 1), (11, 99), (12, 2), (13, 98), (14, NULL)",
    )
    .await;
    assert_eq!(
        query_rows(&server, "SELECT * FROM qc").await,
        3,
        "two references to real parents plus the NULL row"
    );

    // The rejected rows are preserved, not dropped
    let q_entry = server
        .catalog
        .get_table_by_id(zyron_catalog::TableId(quarantine_id))
        .expect("quarantine table");
    assert_eq!(
        query_rows(&server, &format!("SELECT * FROM {}", q_entry.name)).await,
        2,
        "the two absent-parent rows are kept"
    );
}

#[tokio::test]
async fn test_an_update_cannot_be_quarantined() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE up (k BIGINT NOT NULL PRIMARY KEY)",
    )
    .await
    .expect("create parent");
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE uc (id BIGINT NOT NULL, k BIGINT, \
         FOREIGN KEY (k) REFERENCES up(k) ON VIOLATION QUARANTINE)",
    )
    .await
    .expect("create child");
    exec_dml(&server, "INSERT INTO up VALUES (1)").await;
    exec_dml(&server, "INSERT INTO uc VALUES (10, 1)").await;

    // The row is already in the table, so diverting it would delete it
    let err = exec_dml_result(&server, "UPDATE uc SET k = 99 WHERE id = 10")
        .await
        .expect_err("an update that breaks the key must be refused");
    assert!(err.contains("cannot"), "{err}");
    assert_eq!(query_rows(&server, "SELECT * FROM uc").await, 1);
}

#[tokio::test]
async fn test_begin_zyronlake_transaction_commits_several_tables_together() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE xa (id BIGINT NOT NULL) USING ZYRONLAKE",
    )
    .await
    .expect("create a");
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE xb (id BIGINT NOT NULL) USING ZYRONLAKE",
    )
    .await
    .expect("create b");

    // The statement parses and carries the marker
    let stmt = zyron_parser::parse("BEGIN ZYRONLAKE TRANSACTION")
        .expect("parse")
        .into_iter()
        .next()
        .expect("one statement");
    match stmt {
        zyron_parser::Statement::Begin(b) => assert!(b.lake, "the lake marker is carried"),
        other => panic!("expected BEGIN, got {other:?}"),
    }
    let plain = zyron_parser::parse("BEGIN")
        .expect("parse")
        .into_iter()
        .next()
        .expect("one statement");
    match plain {
        zyron_parser::Statement::Begin(b) => assert!(!b.lake, "plain BEGIN is not a lake txn"),
        other => panic!("expected BEGIN, got {other:?}"),
    }

    // Two tables written under one intent become visible together
    let now = 7_000_000i64;
    let mut lake_txn =
        zyron_lake::CrossTableTxn::begin(server.disk_manager.data_dir(), now).expect("begin");
    lake_txn.prepare().expect("prepare");
    exec_dml_in_lake_txn(&server, "INSERT INTO xa VALUES (1), (2)", lake_txn.txn_id()).await;
    exec_dml_in_lake_txn(&server, "INSERT INTO xb VALUES (3)", lake_txn.txn_id()).await;

    // Written and invisible until the intent commits
    assert_eq!(query_rows(&server, "SELECT * FROM xa").await, 0);
    assert_eq!(query_rows(&server, "SELECT * FROM xb").await, 0);

    let logs = lake_txn.commit().expect("commit");
    zyron_wire::connection::refresh_lake_stats(&server, &logs);
    assert_eq!(query_rows(&server, "SELECT * FROM xa").await, 2);
    assert_eq!(query_rows(&server, "SELECT * FROM xb").await, 1);
}

/// Runs a DML statement whose lake writes commit under a cross-table intent.
async fn exec_dml_in_lake_txn(server: &Arc<ServerState>, sql: &str, lake_txn_id: u64) {
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
    let snapshot = txn.snapshot.clone();
    let txn_id = txn.txn_id;
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
    ctx.lake_txn_id = Some(lake_txn_id);
    let ctx = Arc::new(ctx);
    let _ = zyron_executor::execute(plan, &ctx).await.expect("execute");
    server.txn_manager.commit(&mut txn).await.expect("commit");
}

#[tokio::test]
async fn test_alter_table_set_using_converts_in_place_both_ways() {
    let (server, schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE cv (id BIGINT NOT NULL, tag TEXT)",
    )
    .await
    .expect("create heap table");
    exec_dml(
        &server,
        "INSERT INTO cv VALUES (1, 'a'), (2, NULL), (3, 'c')",
    )
    .await;
    assert_eq!(query_rows(&server, "SELECT * FROM cv").await, 3);

    // Heap to lake: the rows move and the format flips
    exec_ddl(&server, &mut session, "ALTER TABLE cv SET USING ZYRONLAKE")
        .await
        .expect("convert to lake");
    let entry = server.catalog.get_table(schema_id, "cv").expect("entry");
    assert!(entry.lake.is_lake());
    assert_eq!(query_rows(&server, "SELECT * FROM cv").await, 3);
    let paths = LakePaths::new(server.disk_manager.data_dir(), entry.id.0);
    assert!(paths.version_file(1).exists(), "the log was created");

    // The converted table takes writes as a lake table
    exec_dml(&server, "INSERT INTO cv VALUES (4, 'd')").await;
    assert_eq!(query_rows(&server, "SELECT * FROM cv").await, 4);

    // Converting to the format it already has is refused
    let err = exec_ddl(&server, &mut session, "ALTER TABLE cv SET USING ZYRONLAKE")
        .await
        .expect_err("already lake");
    assert!(err.contains("already stored"), "{err}");

    // Lake back to heap, keeping the history by default
    exec_ddl(&server, &mut session, "ALTER TABLE cv SET USING HEAP")
        .await
        .expect("convert to heap");
    let entry = server.catalog.get_table(schema_id, "cv").expect("entry");
    assert!(!entry.lake.is_lake());
    assert!(
        entry.lake.retained_history,
        "the history outlives the format unless dropped"
    );
    assert!(paths.root().exists(), "the root is kept");
    assert_eq!(query_rows(&server, "SELECT * FROM cv").await, 4);
}

#[tokio::test]
async fn test_set_using_heap_with_drop_history_reclaims_the_root() {
    let (server, schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE dh (id BIGINT NOT NULL) USING ZYRONLAKE",
    )
    .await
    .expect("create lake table");
    exec_dml(&server, "INSERT INTO dh VALUES (1), (2)").await;
    let entry = server.catalog.get_table(schema_id, "dh").expect("entry");
    let paths = LakePaths::new(server.disk_manager.data_dir(), entry.id.0);
    assert!(paths.root().exists());

    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE dh SET USING HEAP WITH (drop_history = true)",
    )
    .await
    .expect("convert with drop_history");

    let entry = server.catalog.get_table(schema_id, "dh").expect("entry");
    assert!(!entry.lake.is_lake());
    assert!(!entry.lake.retained_history);
    assert!(!paths.root().exists(), "the history was dropped as asked");
    assert_eq!(query_rows(&server, "SELECT * FROM dh").await, 2);
}

/// `ALTER TABLE t CLUSTER BY (...)` commits a new spec, and later appends
/// lay their rows out under it
#[tokio::test]
async fn test_alter_cluster_by_commits_a_new_spec_that_later_writes_use() {
    let (server, schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE events (id BIGINT NOT NULL, region TEXT) USING ZYRONLAKE CLUSTER BY (id)",
    )
    .await
    .expect("create");

    let entry = server
        .catalog
        .get_table(schema_id, "events")
        .expect("entry");
    let paths = LakePaths::new(server.disk_manager.data_dir(), entry.id.0);
    let log = TransactionLog::lookup_shared(&paths).expect("shared log");
    let before = log.latest_manifest().expect("manifest");
    assert_eq!(before.cluster_spec.keys.len(), 1);
    assert_eq!(before.cluster_spec.keys[0].column_id, 0);

    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE events CLUSTER BY (region USING BitInterleave, id) FORCE",
    )
    .await
    .expect("alter cluster by");

    let after = log.latest_manifest().expect("manifest");
    assert!(after.cluster_spec.spec_id > before.cluster_spec.spec_id);
    assert_eq!(after.cluster_spec.keys.len(), 2);
    assert_eq!(after.cluster_spec.keys[0].column_id, 1);
    assert_eq!(
        after.cluster_spec.keys[0].strategy,
        ClusterStrategy::BitInterleave
    );
    assert_eq!(after.cluster_spec.keys[1].column_id, 0);
    assert_eq!(after.clustering_mode(), zyron_lake::ClusterMode::Force);
    // Force pins the whole spec, so the anchors are its keys
    assert_eq!(after.clustering_anchors(), vec![1, 0]);

    // A row written after the change carries the new spec id
    exec_dml(&server, "INSERT INTO events VALUES (1, 'eu'), (2, 'us')").await;
    let written = log.latest_manifest().expect("manifest");
    let file = written.entries.last().expect("a data file");
    assert_eq!(file.cluster_spec_id, after.cluster_spec.spec_id);
}

/// `CLUSTER BY AUTO` hands the choice to measurement without discarding
/// the layout the table already has
#[tokio::test]
async fn test_cluster_by_auto_changes_the_policy_and_keeps_the_layout() {
    let (server, schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE events (id BIGINT NOT NULL, region TEXT) USING ZYRONLAKE CLUSTER BY (id)",
    )
    .await
    .expect("create");
    let entry = server
        .catalog
        .get_table(schema_id, "events")
        .expect("entry");
    let paths = LakePaths::new(server.disk_manager.data_dir(), entry.id.0);
    let log = TransactionLog::lookup_shared(&paths).expect("shared log");
    let before = log.latest_manifest().expect("manifest");

    exec_ddl(&server, &mut session, "ALTER TABLE events CLUSTER BY AUTO")
        .await
        .expect("alter to auto");
    let after = log.latest_manifest().expect("manifest");
    assert_eq!(after.clustering_mode(), zyron_lake::ClusterMode::Auto);
    assert!(after.clustering_anchors().is_empty());
    assert_eq!(
        after.cluster_spec, before.cluster_spec,
        "AUTO must not throw away a working layout to wait for evidence"
    );

    // Hybrid anchors the listed keys and leaves the rest to measurement
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE events CLUSTER BY (id) AUTO",
    )
    .await
    .expect("alter to hybrid");
    let hybrid = log.latest_manifest().expect("manifest");
    assert_eq!(hybrid.clustering_mode(), zyron_lake::ClusterMode::Hybrid);
    assert_eq!(hybrid.clustering_anchors(), vec![0]);
}

/// The schedule is a policy commit and nothing else. It applies to both
/// formats: a lake table records it in its log, a heap table in its
/// catalog entry, because clustering governs the fold tier too
#[tokio::test]
async fn test_clustering_schedule_is_persisted_on_both_formats() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE events (id BIGINT NOT NULL) USING ZYRONLAKE CLUSTER BY (id)",
    )
    .await
    .expect("create");
    exec_ddl(&server, &mut session, "CREATE TABLE plain (id BIGINT)")
        .await
        .expect("create heap");

    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE events SET CLUSTERING SCHEDULE = Continuous",
    )
    .await
    .expect("set schedule");

    let entry = server
        .catalog
        .get_table(_schema_id, "events")
        .expect("entry");
    let paths = LakePaths::new(server.disk_manager.data_dir(), entry.id.0);
    let log = TransactionLog::lookup_shared(&paths).expect("shared log");
    let manifest = log.latest_manifest().expect("manifest");
    assert_eq!(
        manifest.clustering_schedule(),
        zyron_lake::ClusteringSchedule::Continuous
    );

    // A heap table's policy lands in the catalog, where the fold tier
    // reads it when it lays out the next segment
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE plain SET CLUSTERING SCHEDULE = Incremental",
    )
    .await
    .expect("heap schedule");
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE plain CLUSTER BY (id USING BitInterleave)",
    )
    .await
    .expect("heap cluster by");

    let plain = server
        .catalog
        .get_table(_schema_id, "plain")
        .expect("entry");
    assert_eq!(
        plain.cluster.schedule(),
        zyron_common::ClusteringSchedule::Incremental
    );
    // A bare key list declares the keys without pinning them, on either
    // format
    assert_eq!(plain.cluster.mode(), zyron_common::ClusterMode::Auto);
    let keys = plain.cluster.fold_keys();
    assert_eq!(keys.len(), 1);
    assert_eq!(keys[0].column_id, 0);
    assert_eq!(
        keys[0].strategy,
        zyron_common::ClusterStrategy::BitInterleave
    );
    assert!(
        plain.cluster.spec_id > 0,
        "declaring keys advances the spec so folded segments are distinguishable"
    );
}

/// A cluster key that names a column twice, or an anchor list with no
/// keys, is a statement that cannot mean anything
#[tokio::test]
async fn test_malformed_cluster_by_is_refused() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE events (id BIGINT NOT NULL, region TEXT) USING ZYRONLAKE",
    )
    .await
    .expect("create");

    let duplicate = exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE events CLUSTER BY (id, id)",
    )
    .await;
    assert!(duplicate.is_err(), "a repeated key orders nothing new");

    let unknown = exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE events CLUSTER BY (nope)",
    )
    .await;
    assert!(unknown.is_err(), "an unknown column cannot be a key");
}

/// SHOW CLUSTERING reports the layout, what was measured, and what
/// measurement would choose, and changes nothing while doing it
#[tokio::test]
async fn test_show_clustering_reports_without_changing_the_pinned_choice() {
    let (server, schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE events (id BIGINT NOT NULL, region TEXT) USING ZYRONLAKE \
         CLUSTER BY (id USING RangePartition) FORCE",
    )
    .await
    .expect("create");
    exec_dml(&server, "INSERT INTO events VALUES (1, 'eu'), (2, 'us')").await;

    let entry = server
        .catalog
        .get_table(schema_id, "events")
        .expect("entry");
    let paths = LakePaths::new(server.disk_manager.data_dir(), entry.id.0);
    let log = TransactionLog::lookup_shared(&paths).expect("shared log");
    let before = log.latest_manifest().expect("manifest");

    let rows = show_rows(&server, &mut session, "SHOW CLUSTERING FOR events").await;
    let value = |property: &str| -> String {
        rows.iter()
            .find(|r| r[0] == property)
            .map(|r| r[1].clone())
            .unwrap_or_else(|| panic!("SHOW CLUSTERING has no {property} row"))
    };
    // FORCE pins the declared keys, and the schedule is the creation
    // default because the statement named no other one
    assert_eq!(value("mode"), "FORCE");
    assert_eq!(value("schedule"), "CONTINUOUS");
    assert_eq!(value("keys"), "id USING RangePartition");
    assert_eq!(value("anchors"), "id");
    assert_eq!(value("spec_id"), "1");
    assert_eq!(value("files"), "1");
    // The observer is process wide and keyed by table id, so another test
    // in this binary may already have observed scans against this id. What
    // has to hold is that the fit is either absent or a real rate, never a
    // fabricated zero
    let fit = value("measured_fit");
    if fit != "(no scans observed)" {
        let rate: f64 = fit.parse().expect("measured_fit must be a number");
        assert!((0.0..=1.0).contains(&rate), "fit {rate} is not a rate");
    }

    // The keys are byte identical afterwards: reporting is not deciding
    let after = log.latest_manifest().expect("manifest");
    assert_eq!(after.cluster_spec, before.cluster_spec);
    assert_eq!(after.snapshot_id, before.snapshot_id);

    // A heap table reports its own policy rather than being refused: its
    // layout is decided at fold time, so there is no manifest to measure
    // against and the honest answer is the policy and the segments that
    // have reached it
    exec_ddl(&server, &mut session, "CREATE TABLE plain (id BIGINT)")
        .await
        .expect("create heap");
    exec_ddl(&server, &mut session, "ALTER TABLE plain CLUSTER BY (id)")
        .await
        .expect("heap cluster by");
    let heap = show_rows(&server, &mut session, "SHOW CLUSTERING FOR plain").await;
    let heap_value = |property: &str| -> String {
        heap.iter()
            .find(|r| r[0] == property)
            .map(|r| r[1].clone())
            .unwrap_or_else(|| panic!("SHOW CLUSTERING has no {property} row"))
    };
    assert_eq!(heap_value("format"), "HEAP");
    assert_eq!(heap_value("keys"), "id USING RangePartition");
    assert_eq!(heap_value("segments"), "0");
    assert_eq!(heap_value("spec_id"), "1");
}

/// Runs a SHOW that returns rows and hands back the rendered cells
async fn show_rows(
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
    sql: &str,
) -> Vec<Vec<String>> {
    let stmt = zyron_parser::Parser::new(sql)
        .expect("lex")
        .parse_statement()
        .expect("parse");
    let result = zyron_wire::ddl_dispatch::try_handle_ddl_utility(
        &stmt, server, session, &mut None, &mut None, sql,
    )
    .await
    .expect("statement was handled")
    .expect("statement succeeded");
    match result {
        zyron_wire::ddl_dispatch::DdlResult::Rows { rows, .. } => rows,
        other => panic!("expected rows, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Foreign tables
// ---------------------------------------------------------------------------

/// A foreign table is a declaration, not storage: it allocates no heap file,
/// no free-space map and no lake root, and every read of it is a read of
/// the peer
#[tokio::test]
async fn test_create_foreign_table_allocates_no_local_storage() {
    let (server, schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE PEER west ADDRESS '127.0.0.1:1' MODE lake",
    )
    .await
    .expect("declare peer");
    exec_ddl(
        &server,
        &mut session,
        "CREATE FOREIGN TABLE orders (id BIGINT, total DOUBLE PRECISION) SERVER west",
    )
    .await
    .expect("create foreign table");

    let entry = server
        .catalog
        .get_table(schema_id, "orders")
        .expect("entry");
    assert!(entry.foreign.is_foreign());
    assert_eq!(entry.foreign.remote(), Some(("west", "orders")));
    assert_eq!(
        entry.heap_file_id, 0,
        "a foreign table has no heap file, and zero is the reserved no-file id"
    );
    assert_eq!(entry.fsm_file_id, 0);
    assert!(!entry.lake.is_lake(), "no lake root either");
    assert!(
        !LakePaths::new(server.disk_manager.data_dir(), entry.id.0)
            .root()
            .exists()
    );
    assert_eq!(entry.columns.len(), 2);
    assert!(
        entry.constraints.is_empty(),
        "the peer enforces its own constraints"
    );
}

/// TABLE names the remote when it differs from the local name
#[tokio::test]
async fn test_a_foreign_table_may_be_named_differently_than_its_remote() {
    let (server, schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE PEER west ADDRESS '127.0.0.1:1' MODE db",
    )
    .await
    .expect("declare peer");
    exec_ddl(
        &server,
        &mut session,
        "CREATE FOREIGN TABLE west_orders (id BIGINT) SERVER west TABLE orders",
    )
    .await
    .expect("create foreign table");

    let entry = server
        .catalog
        .get_table(schema_id, "west_orders")
        .expect("entry");
    assert_eq!(entry.foreign.remote(), Some(("west", "orders")));
}

/// Naming a node this one was never told about would be discovery by
/// another name, and peering is stated on purpose
#[tokio::test]
async fn test_a_foreign_table_naming_an_undeclared_peer_is_refused() {
    let (server, schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    let err = exec_ddl(
        &server,
        &mut session,
        "CREATE FOREIGN TABLE orders (id BIGINT) SERVER nowhere",
    )
    .await
    .expect_err("an undeclared peer must be refused");
    assert!(
        err.contains("nowhere") && err.contains("CREATE PEER"),
        "{err}"
    );
    assert!(
        server.catalog.get_table(schema_id, "orders").is_err(),
        "a refused statement leaves no half-created table"
    );
}

/// The two DROP statements mean different things, and one silently doing
/// the other would delete data on the strength of a typo
#[tokio::test]
async fn test_drop_foreign_table_refuses_a_local_table() {
    let (server, schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE PEER west ADDRESS '127.0.0.1:1' MODE lake",
    )
    .await
    .expect("declare peer");
    exec_ddl(&server, &mut session, "CREATE TABLE mine (id BIGINT)")
        .await
        .expect("create local table");
    exec_ddl(
        &server,
        &mut session,
        "CREATE FOREIGN TABLE theirs (id BIGINT) SERVER west",
    )
    .await
    .expect("create foreign table");

    let err = exec_ddl(&server, &mut session, "DROP FOREIGN TABLE mine")
        .await
        .expect_err("a local table is not a declaration to remove");
    assert!(err.contains("DROP TABLE"), "{err}");
    assert!(
        server.catalog.get_table(schema_id, "mine").is_ok(),
        "the refused drop left the table alone"
    );

    exec_ddl(&server, &mut session, "DROP FOREIGN TABLE theirs")
        .await
        .expect("drop the declaration");
    assert!(server.catalog.get_table(schema_id, "theirs").is_err());

    // IF EXISTS makes a second drop a no-op, without it the drop reports
    exec_ddl(&server, &mut session, "DROP FOREIGN TABLE IF EXISTS theirs")
        .await
        .expect("second drop is a no-op");
    assert!(
        exec_ddl(&server, &mut session, "DROP FOREIGN TABLE theirs")
            .await
            .is_err()
    );
}

/// A query over a foreign table plans as a ForeignScan with the projection,
/// the filter and the row cap travelling to the peer
#[tokio::test]
async fn test_a_query_on_a_foreign_table_pushes_projection_filter_and_limit() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE PEER west ADDRESS '127.0.0.1:1' MODE lake",
    )
    .await
    .expect("declare peer");
    exec_ddl(
        &server,
        &mut session,
        "CREATE FOREIGN TABLE orders (id BIGINT, region TEXT, total DOUBLE PRECISION, \
         notes TEXT) SERVER west TABLE all_orders",
    )
    .await
    .expect("create foreign table");

    let plan = plan_sql(
        &server,
        "SELECT id, total FROM orders WHERE region = 'west' LIMIT 25",
    )
    .await;
    let scan = find_foreign_scan(&plan).expect("query routes to a foreign scan");
    assert_eq!(scan.peer, "west");
    assert_eq!(scan.table, "all_orders", "the remote's own name is used");
    // The scan fetches what it outputs: the projection plus the columns the
    // filter names. A column the query never mentions is not asked for, and
    // that is the saving a wide table gets from a narrow SELECT
    assert!(
        !scan.columns.contains(&"notes".to_string()),
        "an unmentioned column never crosses the wire: {:?}",
        scan.columns
    );
    assert!(scan.columns.contains(&"id".to_string()));
    assert!(scan.columns.contains(&"total".to_string()));
    assert_eq!(
        scan.columns.len(),
        scan.column_types.len(),
        "every fetched column says what it should decode as"
    );
    let pushed = scan
        .predicate
        .as_deref()
        .expect("the filter goes to the peer");
    assert!(
        pushed.contains("region") && pushed.contains("west"),
        "{pushed}"
    );
    assert_eq!(scan.limit, Some(25), "a row cap does not fetch a table");
}

/// A conjunct with no faithful SQL rendering stays here and the rest still
/// travels. A LIMIT does not, because a remote cap applied before a local
/// filter would cut rows the filter was going to keep
#[tokio::test]
async fn test_an_unpushable_conjunct_stays_local_and_holds_back_the_limit() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE PEER west ADDRESS '127.0.0.1:1' MODE db",
    )
    .await
    .expect("declare peer");
    exec_ddl(
        &server,
        &mut session,
        "CREATE FOREIGN TABLE orders (id BIGINT, region TEXT) SERVER west",
    )
    .await
    .expect("create foreign table");

    let plan = plan_sql(
        &server,
        "SELECT id FROM orders WHERE region = 'west' AND id IN (SELECT 1) LIMIT 10",
    )
    .await;
    let scan = find_foreign_scan(&plan).expect("routes to a foreign scan");
    let pushed = scan
        .predicate
        .as_deref()
        .expect("the renderable half still travels");
    assert!(pushed.contains("region"), "{pushed}");
    assert!(
        !pushed.to_ascii_uppercase().contains("SELECT"),
        "a subquery has no row filter form: {pushed}"
    );
    assert_eq!(
        scan.limit, None,
        "a cap ahead of a local filter would cut rows the filter keeps"
    );
}

/// This node holds no history for a table it does not store, and saying so
/// beats returning the peer's current rows under an AS OF it never saw
#[tokio::test]
async fn test_time_travel_on_a_foreign_table_is_refused() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE PEER west ADDRESS '127.0.0.1:1' MODE lake",
    )
    .await
    .expect("declare peer");
    exec_ddl(
        &server,
        &mut session,
        "CREATE FOREIGN TABLE orders (id BIGINT) SERVER west",
    )
    .await
    .expect("create foreign table");

    let stmt = zyron_parser::parse("SELECT id FROM orders VERSION AS OF 3")
        .expect("parse")
        .into_iter()
        .next()
        .expect("one statement");
    let err = zyron_planner::plan(
        &server.catalog,
        DatabaseId(1),
        vec!["public".into()],
        stmt,
        Some(&server.peer_facts()),
    )
    .await
    .expect_err("AS OF must be refused on a foreign table");
    let message = err.to_string();
    assert!(message.contains("west"), "{message}");
}

/// What a peer stores decides how much a pushed filter saves it, so the two
/// modes cost differently and a lake peer is the cheaper one
#[tokio::test]
async fn test_a_lake_peer_costs_a_selective_filter_below_a_db_peer() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    for sql in [
        "CREATE PEER lakeside ADDRESS '127.0.0.1:1' MODE lake",
        "CREATE PEER dbside ADDRESS '127.0.0.2:1' MODE db",
        "CREATE FOREIGN TABLE from_lake (id BIGINT, region TEXT) SERVER lakeside",
        "CREATE FOREIGN TABLE from_db (id BIGINT, region TEXT) SERVER dbside",
    ] {
        exec_ddl(&server, &mut session, sql).await.expect(sql);
    }

    let lakePlan = plan_sql(&server, "SELECT id FROM from_lake WHERE region = 'w'").await;
    let dbPlan = plan_sql(&server, "SELECT id FROM from_db WHERE region = 'w'").await;
    let lakeCost = plan_cost(&lakePlan);
    let dbCost = plan_cost(&dbPlan);
    assert!(
        lakeCost <= dbCost,
        "file pruning saves a peer more than an index walk: lake {lakeCost} vs db {dbCost}"
    );
}

/// EXPLAIN names the peer and reports which half of the filter runs where,
/// because that split is what a foreign scan actually costs
#[tokio::test]
async fn test_explain_reports_the_peer_and_the_filter_split() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE PEER west ADDRESS '127.0.0.1:1' MODE lake",
    )
    .await
    .expect("declare peer");
    exec_ddl(
        &server,
        &mut session,
        "CREATE FOREIGN TABLE orders (id BIGINT, region TEXT) SERVER west TABLE all_orders",
    )
    .await
    .expect("create foreign table");

    let plan = plan_sql(&server, "SELECT id FROM orders WHERE region = 'west'").await;
    let text = zyron_planner::ExplainNode::from_physical_plan(&plan).render(&Default::default());
    assert!(text.contains("ForeignScan"), "{text}");
    assert!(text.contains("peer"), "{text}");
    assert!(text.contains("all_orders"), "{text}");
    assert!(text.contains("pushed_filter"), "{text}");
}

/// Plans a query against the server's catalog and mesh view
async fn plan_sql(server: &Arc<ServerState>, sql: &str) -> zyron_planner::physical::PhysicalPlan {
    let stmt = zyron_parser::parse(sql)
        .expect("parse")
        .into_iter()
        .next()
        .expect("one statement");
    zyron_planner::plan(
        &server.catalog,
        DatabaseId(1),
        vec!["public".into()],
        stmt,
        Some(&server.peer_facts()),
    )
    .await
    .expect("plan")
}

/// The one foreign scan in a plan, wherever a Project or Limit put it
fn find_foreign_scan(
    plan: &zyron_planner::physical::PhysicalPlan,
) -> Option<&zyron_common::ForeignRequest> {
    use zyron_planner::physical::PhysicalPlan as P;
    match plan {
        P::ForeignScan { request, .. } => Some(request),
        P::Filter { child, .. }
        | P::Project { child, .. }
        | P::Limit { child, .. }
        | P::Sort { child, .. } => find_foreign_scan(child),
        _ => None,
    }
}

/// The foreign scan's own estimated IO cost
fn plan_cost(plan: &zyron_planner::physical::PhysicalPlan) -> f64 {
    use zyron_planner::physical::PhysicalPlan as P;
    match plan {
        P::ForeignScan { cost, .. } => cost.io_cost,
        P::Filter { child, .. }
        | P::Project { child, .. }
        | P::Limit { child, .. }
        | P::Sort { child, .. } => plan_cost(child),
        other => panic!("no foreign scan under {other:?}"),
    }
}

/// Every query shape that projects no column, on both formats.
///
/// A batch built from zero column builders reports zero rows however many
/// it holds, so a scan that took that path answered zero for a table full
/// of rows. That is what `SELECT COUNT(*)` on a lake table did: a wrong
/// answer rather than a slow one, and only an unfiltered count reached it
/// because a predicate or a named column forces a projection that hides
/// the defect. The whole shape class is pinned here rather than the one
/// query that happened to expose it.
#[tokio::test]
async fn test_queries_that_project_no_column_agree_across_formats() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    for (name, using) in [("np_heap", ""), ("np_lake", " USING ZYRONLAKE")] {
        exec_ddl(
            &server,
            &mut session,
            &format!("CREATE TABLE {name} (id BIGINT NOT NULL, region BIGINT){using}"),
        )
        .await
        .expect("create");
        exec_dml(
            &server,
            &format!("INSERT INTO {name} VALUES (1, 7), (2, 7), (3, 8), (4, 8), (5, 9)"),
        )
        .await;
    }

    for shape in [
        "SELECT COUNT(*) FROM {}",
        "SELECT COUNT(*) FROM {} WHERE region = 7",
        "SELECT COUNT(id) FROM {}",
        "SELECT 1 FROM {}",
        "SELECT EXISTS (SELECT 1 FROM {})",
        "SELECT (SELECT COUNT(*) FROM {})",
    ] {
        let heap = query_values(&server, &shape.replace("{}", "np_heap")).await;
        let lake = query_values(&server, &shape.replace("{}", "np_lake")).await;
        assert_eq!(
            heap,
            lake,
            "the formats disagree on {}",
            shape.replace("{}", "<table>")
        );
        assert!(
            !heap.is_empty(),
            "{} returned nothing on either format",
            shape.replace("{}", "<table>")
        );
    }

    // An unqualified DELETE projects no column either, and still has to
    // address every row it removes
    for name in ["np_heap", "np_lake"] {
        exec_dml(&server, &format!("DELETE FROM {name}")).await;
    }
    let heap = query_values(&server, "SELECT COUNT(*) FROM np_heap").await;
    let lake = query_values(&server, "SELECT COUNT(*) FROM np_lake").await;
    assert_eq!(
        heap, lake,
        "the formats disagree after an unqualified delete"
    );
    match heap.first().and_then(|r| r.first()) {
        Some(zyron_executor::column::ScalarValue::Int64(0)) => {}
        other => panic!("an unqualified delete must empty the table, got {other:?}"),
    }
}

/// A table clustered by an expression stores that expression in a column of
/// its own, and every write fills it. Without the stored values there are no
/// file statistics over the expression, so a query filtering on it could
/// never prune, which is the whole point of allowing the key
#[tokio::test]
async fn test_cluster_by_expression_stores_and_fills_a_derived_column() {
    let (server, schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE events (id BIGINT NOT NULL, ts TIMESTAMP, country TEXT) \
         USING ZYRONLAKE CLUSTER BY (date_part('year', ts))",
    )
    .await
    .expect("create with an expression cluster key");

    let entry = server
        .catalog
        .get_table(schema_id, "events")
        .expect("entry");
    // The expression is mirrored into the catalog so the planner can match a
    // query against it without opening the log
    assert_eq!(entry.cluster.derived.len(), 1, "one expression, one column");
    let mirrored = &entry.cluster.derived[0];
    assert!(
        mirrored.sql.contains("date_part"),
        "the mirror carries the expression text, got {}",
        mirrored.sql
    );
    assert_ne!(mirrored.canonical_hash, 0);

    let paths = LakePaths::new(server.disk_manager.data_dir(), entry.id.0);
    let log = TransactionLog::lookup_shared(&paths).expect("shared log");
    let manifest = log.latest_manifest().expect("manifest");
    assert_eq!(
        manifest.schema.derived.len(),
        1,
        "version one already describes the expression column"
    );
    let derived_id = manifest.schema.derived[0].column_id;
    assert_eq!(
        manifest.schema.derived[0].source_columns,
        vec![1],
        "the expression reads ts, which is what makes dropping ts refusable"
    );
    assert_eq!(
        manifest.cluster_spec.keys.len(),
        1,
        "the key list orders by the column holding the expression"
    );
    assert_eq!(manifest.cluster_spec.keys[0].column_id, derived_id);
    // The table declared three columns, clustering added the fourth
    assert_eq!(manifest.schema.columns.len(), 4);
    assert_eq!(manifest.schema.user_columns().count(), 3);

    // The insert has to fill the expression column or the writer rejects the
    // batch as short of the schema
    exec_dml(
        &server,
        "INSERT INTO events VALUES \
         (1, TIMESTAMP '2024-03-01 00:00:00', 'eu'), \
         (2, TIMESTAMP '2026-07-04 00:00:00', 'us')",
    )
    .await;

    let after = log.latest_manifest().expect("manifest");
    let file = after.entries.last().expect("a data file");
    let stats = file
        .stats_for(derived_id)
        .expect("the expression column carries statistics, which is what prunes");
    assert!(
        stats.bounds.min.is_some() && stats.bounds.max.is_some(),
        "computed values reached the writer's statistics pass"
    );
    assert_ne!(
        stats.bounds.min, stats.bounds.max,
        "two different years must not collapse to one bound"
    );

    // The rows themselves are unchanged by the extra column
    let rows = query_rows(&server, "SELECT id FROM events ORDER BY id").await;
    assert_eq!(rows, 2, "the extra column did not change the row set");
}

/// An expression the engine cannot evaluate has to be refused when it is
/// declared. Accepting it would create a table whose every insert fails
#[tokio::test]
async fn test_cluster_by_an_uncomputable_expression_is_refused() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    let err = exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE events (id BIGINT NOT NULL, ts TIMESTAMP) USING ZYRONLAKE \
         CLUSTER BY (no_such_function(ts))",
    )
    .await
    .expect_err("an unknown function cannot be a cluster key");
    let text = format!("{err:?}").to_lowercase();
    assert!(
        text.contains("no_such_function") || text.contains("unknown function"),
        "the refusal has to name what it could not compute, got {text}"
    );

    // A volatile expression is refused too: its value would differ between
    // the write that stored it and the query that filters on it
    assert!(
        exec_ddl(
            &server,
            &mut session,
            "CREATE TABLE v (id BIGINT NOT NULL, ts TIMESTAMP) USING ZYRONLAKE \
             CLUSTER BY (date_part('year', now()))",
        )
        .await
        .is_err(),
        "a volatile expression must not become a cluster key"
    );
}

/// Everything an expression is built from reaches a kernel that runs over
/// a whole column, except a short list of predicates the evaluator answers
/// a row at a time. Declaring one of those as a cluster key is legal and
/// expensive, so the statement that declares it has to say so while the
/// operator can still choose something else. Silence would leave the cost
/// to be discovered from a slow insert months later
#[tokio::test]
async fn test_a_row_at_a_time_clustering_expression_warns_at_create() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();

    let logs = CapturedWarnings::default();
    let captured = {
        let subscriber = tracing_subscriber::fmt()
            .with_writer(logs.clone())
            .with_max_level(tracing::Level::WARN)
            .with_ansi(false)
            .finish();
        let _guard = tracing::subscriber::set_default(subscriber);
        exec_ddl(
            &server,
            &mut session,
            "CREATE TABLE neighbors (id BIGINT NOT NULL, embedding VECTOR(3)) USING ZYRONLAKE \
             CLUSTER BY (embedding <-> ARRAY[0.0, 1.0, 0.0])",
        )
        .await
        .expect("a row-at-a-time expression is expensive, not illegal");
        logs.text()
    };

    assert!(
        captured.contains("row at a time"),
        "declaring a per-row expression said nothing about what it costs, log was: {captured}"
    );
    assert!(
        captured.contains("vector_distance_l2"),
        "the warning has to name the function that costs the per-row walk, log was: {captured}"
    );
    assert!(
        captured.contains("neighbors"),
        "the warning has to name the table, log was: {captured}"
    );

    // Expensive and still correct: the column is allocated and a write
    // fills it, which is what makes the warning about cost rather than
    // about a refusal
    exec_dml(
        &server,
        "INSERT INTO neighbors VALUES (1, ARRAY[1.0, 0.0, 0.0]), (2, ARRAY[0.0, 1.0, 0.0])",
    )
    .await;
    assert_eq!(query_rows(&server, "SELECT id FROM neighbors").await, 2);
}

/// A vectorized expression must not warn, or the warning stops meaning
/// anything
#[tokio::test]
async fn test_a_vectorized_clustering_expression_does_not_warn() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();

    let logs = CapturedWarnings::default();
    let captured = {
        let subscriber = tracing_subscriber::fmt()
            .with_writer(logs.clone())
            .with_max_level(tracing::Level::WARN)
            .with_ansi(false)
            .finish();
        let _guard = tracing::subscriber::set_default(subscriber);
        exec_ddl(
            &server,
            &mut session,
            "CREATE TABLE hits (id BIGINT NOT NULL, ts TIMESTAMP) USING ZYRONLAKE \
             CLUSTER BY (date_part('year', ts))",
        )
        .await
        .expect("create");
        logs.text()
    };

    assert!(
        !captured.contains("row at a time"),
        "a kernel-evaluated expression must not be reported as per-row, log was: {captured}"
    );
}

/// Collects what a statement wrote to the WARN level.
///
/// The warning is the entire observable effect of the per-row check, so
/// there is nothing else to assert on. Capturing it is what makes the
/// check testable rather than a line nobody ever reads
#[derive(Clone, Default)]
struct CapturedWarnings(Arc<std::sync::Mutex<Vec<u8>>>);

impl CapturedWarnings {
    fn text(&self) -> String {
        let buffer = self.0.lock().unwrap_or_else(|e| e.into_inner());
        String::from_utf8_lossy(&buffer).into_owned()
    }
}

impl std::io::Write for CapturedWarnings {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CapturedWarnings {
    type Writer = CapturedWarnings;

    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

/// The stored text is read back on every write to recompute the column, so
/// an expression the renderer cannot write back has to be refused where it
/// is declared. Accepting it would create a table whose every insert fails
#[tokio::test]
async fn test_a_clustering_expression_that_cannot_be_read_back_is_refused() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    let err = exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE docs (id BIGINT NOT NULL, body TEXT) USING ZYRONLAKE \
         CLUSTER BY (MATCH(body) AGAINST ('needle'))",
    )
    .await
    .expect_err("a search predicate has no form the write path can read back");
    let text = format!("{err:?}");
    assert!(
        text.contains("read back") || text.contains("cannot be stored"),
        "the refusal has to say the expression could not be stored, got {text}"
    );
}

/// A derived column added to a table that already holds rows would be null
/// for every one of them, and a query rewritten onto it would then drop
/// rows that match. Filling them in means rewriting every data file, which
/// is the rewrite a lake ALTER refuses, so the statement is refused instead
/// of quietly losing the rows
#[tokio::test]
async fn test_add_derived_column_on_a_table_with_rows_is_refused() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE hits (id BIGINT NOT NULL, ts TIMESTAMP) USING ZYRONLAKE",
    )
    .await
    .expect("create");
    exec_dml(
        &server,
        "INSERT INTO hits VALUES (1, TIMESTAMP '2013-01-01 00:00:00'), \
         (2, TIMESTAMP '2019-01-01 00:00:00')",
    )
    .await;

    let err = exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE hits ADD DERIVED COLUMN yr AS date_part('year', ts)",
    )
    .await
    .expect_err("rows written before the expression have no value for it");
    let text = format!("{err:?}");
    assert!(
        text.contains("2 rows") && text.contains("drop them"),
        "the refusal has to say how many rows are at stake and what would happen to them, \
         got {text}"
    );

    // Refused means refused: the table is untouched and its rows still
    // answer the query they always did
    let entry = server
        .catalog
        .list_all_tables()
        .into_iter()
        .find(|t| t.name == "hits")
        .expect("table");
    assert!(
        entry.cluster.derived.is_empty(),
        "a refused ALTER must not leave a half-registered expression behind"
    );
    assert_eq!(query_rows(&server, "SELECT id FROM hits").await, 2);
}

/// A derived column declared on an empty table is filled by every write
/// after it, which is the case that needs no backfill
#[tokio::test]
async fn test_add_derived_column_on_an_empty_table_is_filled_by_later_writes() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE hits (id BIGINT NOT NULL, ts TIMESTAMP) USING ZYRONLAKE",
    )
    .await
    .expect("create");
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE hits ADD DERIVED COLUMN yr AS date_part('year', ts)",
    )
    .await
    .expect("a derived column on an empty table needs nothing backfilled");

    exec_dml(
        &server,
        "INSERT INTO hits VALUES (1, TIMESTAMP '2013-01-01 00:00:00'), \
         (2, TIMESTAMP '2019-01-01 00:00:00')",
    )
    .await;

    let entry = server
        .catalog
        .list_all_tables()
        .into_iter()
        .find(|t| t.name == "hits")
        .expect("table");
    assert_eq!(
        entry.cluster.derived.len(),
        1,
        "the catalog mirror is what the planner matches a query against"
    );
    let paths = LakePaths::new(server.disk_manager.data_dir(), entry.id.0);
    let log = TransactionLog::lookup_shared(&paths).expect("shared log");
    let manifest = log.latest_manifest().expect("manifest");
    let derived_id = manifest.schema.derived[0].column_id;
    let file = manifest.entries.last().expect("a data file");
    let stats = file
        .stats_for(derived_id)
        .expect("the write filled the derived column");
    assert_eq!(stats.bounds.min, Some(zyron_lake::LakeValue::Int(2013)));
    assert_eq!(stats.bounds.max, Some(zyron_lake::LakeValue::Int(2019)));

    let rows = query_values(
        &server,
        "SELECT id FROM hits WHERE date_part('year', ts) = 2013 ORDER BY id",
    )
    .await;
    assert_eq!(rows.len(), 1, "the expression predicate lost a row");
}

/// An expression the evaluator cannot compute has to be refused by ALTER
/// for the same reason CREATE refuses it
#[tokio::test]
async fn test_add_derived_column_refuses_an_uncomputable_expression() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE hits (id BIGINT NOT NULL, ts TIMESTAMP) USING ZYRONLAKE",
    )
    .await
    .expect("create");
    assert!(
        exec_ddl(
            &server,
            &mut session,
            "ALTER TABLE hits ADD DERIVED COLUMN bad AS no_such_function(ts)",
        )
        .await
        .is_err(),
        "a function the engine does not have must not become a derived column"
    );
    assert!(
        exec_ddl(
            &server,
            &mut session,
            "ALTER TABLE hits ADD DERIVED COLUMN whenever AS date_part('year', now())",
        )
        .await
        .is_err(),
        "a volatile expression must not become a derived column"
    );
}

/// A plan says how much the table's layout does for it, before it runs.
///
/// The scan counters say how many files were skipped, which is the outcome.
/// This is the reason: a predicate that reached no cluster key skips files
/// only by luck, and the counters alone cannot tell that apart from a
/// predicate that reached one and matched everything
#[tokio::test]
async fn test_explain_reports_what_the_layout_does_for_a_predicate() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE sales (region BIGINT, sku BIGINT, amount BIGINT) USING ZYRONLAKE \
         CLUSTER BY (region, sku) FORCE",
    )
    .await
    .expect("create");
    exec_dml(
        &server,
        "INSERT INTO sales VALUES (1, 10, 100), (2, 20, 200)",
    )
    .await;

    let leading = render_plan(&server, "SELECT amount FROM sales WHERE region = 1").await;
    assert!(
        leading.contains("cluster_fit=good (region)"),
        "the leading key is what decides which files open, {leading}"
    );

    let secondary = render_plan(&server, "SELECT amount FROM sales WHERE sku = 10").await;
    assert!(
        secondary.contains("cluster_fit=fair (sku is cluster key 2)"),
        "a later key narrows within a run rather than across the file set, {secondary}"
    );

    let missed = render_plan(&server, "SELECT region FROM sales WHERE amount = 100").await;
    assert!(
        missed.contains("cluster_fit=poor (fell back to amount)"),
        "a predicate reaching no key has to say which column it reached instead, {missed}"
    );

    // Nothing to judge when the table has no layout, and saying "poor"
    // there would read as a defect rather than as an absence
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE plainlake (a BIGINT, b BIGINT) USING ZYRONLAKE",
    )
    .await
    .expect("create");
    let unclustered = render_plan(&server, "SELECT b FROM plainlake WHERE a = 1").await;
    assert!(
        !unclustered.contains("cluster_fit"),
        "a table with no cluster keys has no fit to report, {unclustered}"
    );
}

/// An expression cluster key is stored in a column with no name a user
/// wrote, so a verdict about it has to name the expression instead of a
/// column number nobody can act on
#[tokio::test]
async fn test_explain_names_the_expression_behind_an_expression_cluster_key() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE hits (id BIGINT, ts TIMESTAMP) USING ZYRONLAKE \
         CLUSTER BY (date_part('year', ts)) FORCE",
    )
    .await
    .expect("create");

    let text = render_plan(
        &server,
        "SELECT id FROM hits WHERE date_part('year', ts) = 2013",
    )
    .await;
    assert!(
        text.contains("cluster_fit=good (date_part("),
        "the verdict has to name the expression the key stores, {text}"
    );
}

/// A bloom filter on the leading cluster key is not built, because the
/// key's own file bounds already resolve it. The plan has to say so: the
/// operator asked for something the writer did not do, and a filter that
/// is silently absent looks the same as one that is not helping
#[tokio::test]
async fn test_a_bloom_on_the_leading_cluster_key_is_reported_as_not_built() {
    let (server, schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE sales (region BIGINT, sku BIGINT, amount BIGINT) USING ZYRONLAKE \
         CLUSTER BY (region, sku) FORCE WITH (bloom_filter_columns = 'region,sku')",
    )
    .await
    .expect("create");
    exec_dml(
        &server,
        "INSERT INTO sales VALUES (1, 10, 100), (2, 20, 200)",
    )
    .await;

    let entry = server.catalog.get_table(schema_id, "sales").expect("entry");
    assert_eq!(
        entry.cluster.bloom_columns.len(),
        2,
        "the catalog mirrors what was asked for, not what was built"
    );
    assert_eq!(
        entry.cluster.redundant_bloom_columns().len(),
        1,
        "only the leading key is covered by the layout"
    );

    let text = render_plan(&server, "SELECT amount FROM sales WHERE sku = 10").await;
    assert!(
        text.contains("bloom_redundant=region"),
        "the plan has to name the filter it did not build, {text}"
    );
    assert!(
        !text.contains("bloom_redundant=sku"),
        "a filter on a key after the leading one is built and must not be reported as \
         dropped, {text}"
    );

    // The writer agrees with the plan: no filter exists on the leading key
    let paths = LakePaths::new(server.disk_manager.data_dir(), entry.id.0);
    let log = TransactionLog::lookup_shared(&paths).expect("shared log");
    let manifest = log.latest_manifest().expect("manifest");
    assert_eq!(manifest.bloom_columns(), vec![1], "only sku gets a filter");
    let file = manifest.entries.last().expect("a data file");
    assert!(
        file.stats_for(0).map(|s| s.bloom.is_none()).unwrap_or(true),
        "a filter was built for the column the layout already resolves"
    );
    assert!(
        file.stats_for(1)
            .map(|s| s.bloom.is_some())
            .unwrap_or(false),
        "the filter that was not redundant has to exist"
    );
}

/// A table with no layout builds every filter it was asked for, and reports
/// nothing as dropped
#[tokio::test]
async fn test_an_unclustered_table_builds_every_declared_bloom() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE flat (a BIGINT, b BIGINT) USING ZYRONLAKE \
         WITH (bloom_filter_columns = 'a,b')",
    )
    .await
    .expect("create");
    exec_dml(&server, "INSERT INTO flat VALUES (1, 2)").await;

    let text = render_plan(&server, "SELECT b FROM flat WHERE a = 1").await;
    assert!(!text.contains("bloom_redundant"), "{text}");
}

/// The catalog's clustering policy has to say what the manifest says.
///
/// The manifest is the authority and the planner never opens one, so
/// everything the planner decides about a layout is decided from the
/// mirror. This went unnoticed once already: CREATE TABLE wrote the policy
/// to the manifest and nothing to the catalog, so every lake table planned
/// as though it had no layout at all, and nothing caught it because there
/// was no consumer of the mirror until a cost model tried to read it. This
/// pins the two together at the statement that fills them, so a future
/// divergence trips here rather than in whatever joins next
#[tokio::test]
async fn test_the_catalog_mirror_matches_the_manifest_it_was_taken_from() {
    let (server, schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE sales (region BIGINT, sku BIGINT, ts TIMESTAMP) USING ZYRONLAKE \
         CLUSTER BY (region, sku) FORCE WITH (bloom_filter_columns = 'region,sku')",
    )
    .await
    .expect("create");

    let entry = server.catalog.get_table(schema_id, "sales").expect("entry");
    let paths = LakePaths::new(server.disk_manager.data_dir(), entry.id.0);
    let log = TransactionLog::lookup_shared(&paths).expect("shared log");
    let manifest = log.latest_manifest().expect("manifest");

    assert_eq!(
        entry.cluster.mode,
        manifest.clustering_mode().to_u8(),
        "the mode a plan reads has to be the mode the table runs"
    );
    assert_eq!(
        entry.cluster.schedule,
        manifest.clustering_schedule().to_u8(),
        "the schedule decides whether the layout a plan priced will survive"
    );
    assert_eq!(entry.cluster.spec_id, manifest.cluster_spec.spec_id);
    assert_eq!(
        entry.cluster.fold_keys(),
        manifest.cluster_spec.keys,
        "the declared keys are what a plan is judged against until a pass replaces them"
    );
    assert_eq!(
        entry.cluster.bloom_columns,
        manifest.declared_bloom_columns(),
        "the planner reports which declared filters were not built, from this"
    );
    assert_eq!(
        entry.cluster.derived.len(),
        manifest.schema.derived.len(),
        "an expression the planner cannot see is one it cannot rewrite a predicate onto"
    );

    // An ALTER that moves the layout has to move the mirror with it, which
    // is the half that was missing when the mirror had no reader
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE sales CLUSTER BY (sku) FORCE",
    )
    .await
    .expect("redeclare the layout");
    let entry = server.catalog.get_table(schema_id, "sales").expect("entry");
    let manifest = log.latest_manifest().expect("manifest");
    assert_eq!(entry.cluster.fold_keys(), manifest.cluster_spec.keys);
    assert_eq!(entry.cluster.spec_id, manifest.cluster_spec.spec_id);
    assert!(
        entry.cluster.active_keys.is_empty(),
        "a new declaration supersedes whatever a pass had chosen"
    );
}

/// A numeric literal has to read back as the same kind of number.
///
/// The stored text is re-parsed on every write, so a literal that renders
/// without its point comes back as an integer and binds at a different
/// type than the one that was written. That is a wrong answer generator
/// rather than a cosmetic defect: the stored column would hold values of
/// one type and the query's constant would be compared at another.
/// `prove_storable` refuses anything that does not round trip, so a CREATE
/// succeeding here is the proof
#[tokio::test]
async fn test_every_numeric_literal_kind_survives_the_round_trip() {
    let cases: &[(&str, &str)] = &[
        ("whole", "amount + 1"),
        ("negative_whole", "amount + -1"),
        ("zero", "amount + 0"),
        // The exact shape that was broken: an integral float rendered as
        // `0` and re-parsed as an integer
        ("integral_float", "amount * 1.0"),
        ("zero_float", "amount * 0.0"),
        ("fractional", "amount * 1.5"),
        ("negative_fractional", "amount * -2.25"),
        ("many_places", "amount * 1.0625"),
        ("large_whole", "amount + 9007199254740993"),
    ];
    for (label, expr) in cases {
        let (server, _schema_id, _tmp) = create_test_server().await;
        let mut session = new_session();
        exec_ddl(
            &server,
            &mut session,
            &format!(
                "CREATE TABLE t_{label} (id BIGINT NOT NULL, amount BIGINT) USING ZYRONLAKE \
                 CLUSTER BY ({expr})"
            ),
        )
        .await
        .unwrap_or_else(|e| panic!("{label}: \"{expr}\" did not round trip: {e}"));

        // And it computes, which is the other half of what the stored text
        // has to be good for
        exec_dml(&server, &format!("INSERT INTO t_{label} VALUES (1, 4)")).await;
        assert_eq!(
            query_rows(&server, &format!("SELECT id FROM t_{label}")).await,
            1,
            "{label}: the write path could not compute the expression it stored"
        );
    }
}

/// A join is evidence about both tables in it.
///
/// Until this was wired, measurement could not see a join at all: a table
/// joined on a column a thousand times a minute looked exactly like a table
/// nobody read, so the layout it was given served the filters and left the
/// join reading everything against everything. Both sides are credited,
/// because a join is only co-located when both are ordered by the key
#[tokio::test]
async fn test_a_join_credits_the_key_on_both_tables() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE orders (id BIGINT NOT NULL, customer BIGINT) USING ZYRONLAKE",
    )
    .await
    .expect("create");
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE customers (cid BIGINT NOT NULL, region BIGINT) USING ZYRONLAKE",
    )
    .await
    .expect("create");
    exec_dml(&server, "INSERT INTO orders VALUES (1, 10), (2, 20)").await;
    exec_dml(&server, "INSERT INTO customers VALUES (10, 1), (20, 2)").await;

    let orders = table_id_of(&server, "orders");
    let customers = table_id_of(&server, "customers");
    // Column one on each side is the join key: orders.customer and
    // customers.region are both ordinal one, and the key is orders.customer
    // against customers.cid, which is ordinal zero
    let before_left = join_score(orders, 1);
    let before_right = join_score(customers, 0);

    let rows = query_rows(
        &server,
        "SELECT orders.id FROM orders JOIN customers ON orders.customer = customers.cid",
    )
    .await;
    assert_eq!(
        rows, 2,
        "the join has to actually answer before it is evidence"
    );

    assert!(
        join_score(orders, 1) > before_left,
        "the left side's key was not credited"
    );
    assert!(
        join_score(customers, 0) > before_right,
        "the right side's key was not credited, so the join can never become co-located"
    );

    // A column the join did not name is untouched, or every column on a
    // joined table would drift toward being a cluster key
    assert_eq!(
        join_score(orders, 0),
        0.0,
        "a column no join key names must not be credited"
    );
}

/// A join key is only evidence when it names a plain column on a lake table
/// on both sides. Anything else would have to be attributed by guessing,
/// and evidence nobody can trust moves a layout for a query that never ran
#[tokio::test]
async fn test_a_join_against_a_heap_table_credits_nothing() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE lakeside (id BIGINT NOT NULL, k BIGINT) USING ZYRONLAKE",
    )
    .await
    .expect("create");
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE heapside (hid BIGINT NOT NULL, k BIGINT)",
    )
    .await
    .expect("create");
    exec_dml(&server, "INSERT INTO lakeside VALUES (1, 10)").await;
    exec_dml(&server, "INSERT INTO heapside VALUES (1, 10)").await;

    let lake = table_id_of(&server, "lakeside");
    let before = join_score(lake, 1);
    let rows = query_rows(
        &server,
        "SELECT lakeside.id FROM lakeside JOIN heapside ON lakeside.k = heapside.k",
    )
    .await;
    assert_eq!(rows, 1);
    assert_eq!(
        join_score(lake, 1),
        before,
        "one side is not a lake table, so there is no layout on it to ask for"
    );
}

fn table_id_of(server: &Arc<ServerState>, table: &str) -> u32 {
    server
        .catalog
        .list_all_tables()
        .into_iter()
        .find(|t| t.name == table)
        .expect("table")
        .id
        .0
}

/// How much joining this table has asked for an ordering on this column
fn join_score(table_id: u32, column_id: u32) -> f64 {
    zyron_lake::observer().score(
        table_id,
        zyron_lake::column_term(column_id, zyron_lake::TERM_JOIN_KEY),
        zyron_lake::current_epoch(),
    )
}

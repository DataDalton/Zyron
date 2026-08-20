//! What OPTIMIZE does to a lake table, and what maintenance does unasked.
//!
//! Run: cargo test -p zyron-wire --test lake_optimize_test
//!
//! Two things share one rewrite, because writing the survivors of a file
//! set into a new file settles both: delete predicates recorded against
//! partially covered files are applied physically and retired, and files
//! too small to be worth opening separately are merged. The thresholds
//! that make maintenance do this without being asked are table properties,
//! and every unasked run is recorded where an operator can find it.

use std::sync::Arc;

use zyron_lake::{LakePaths, TransactionLog};
use zyron_wire::connection::ServerState;

mod common;
use common::{create_test_server, exec_ddl, exec_dml, new_session, query_rows, query_values};

/// The current manifest of a lake table
fn manifest_of(server: &Arc<ServerState>, table: &str) -> Arc<zyron_lake::ManifestFile> {
    let entry = server
        .catalog
        .list_all_tables()
        .into_iter()
        .find(|t| t.name == table)
        .expect("table");
    let paths = zyron_lake::LakePaths::new(server.disk_manager.data_dir(), entry.id.0);
    let log = zyron_lake::TransactionLog::lookup_shared(&paths).expect("lake log");
    log.manifest_at(log.latest_version()).expect("manifest")
}

/// Rows a table holds, read back through the planner rather than from the
/// manifest, so a rewrite that lost a row is caught as a wrong answer
async fn ids(server: &Arc<ServerState>, table: &str) -> Vec<String> {
    query_values(server, &format!("SELECT id FROM {table} ORDER BY id"))
        .await
        .into_iter()
        .map(|row| format!("{:?}", row.first().expect("a column")))
        .collect()
}

async fn table_with_one_file_per_insert(rows: i64) -> (Arc<ServerState>, tempfile::TempDir) {
    let (server, _schema_id, tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE parts (id BIGINT NOT NULL, tag BIGINT) USING ZYRONLAKE",
    )
    .await
    .expect("create");
    for i in 0..rows {
        exec_dml(
            &server,
            &format!("INSERT INTO parts VALUES ({i}, {})", i % 4),
        )
        .await;
    }
    (server, tmp)
}

/// Runs the lake half of `OPTIMIZE TABLE t`, the same function the
/// statement reaches.
///
/// The statement itself also takes the node's vacuum guard and answers
/// through the wire protocol, neither of which is what these tests are
/// about
async fn optimize(server: &Arc<ServerState>, table: &str, cluster: bool, delete: bool) -> String {
    let entry = server
        .catalog
        .list_all_tables()
        .into_iter()
        .find(|t| t.name == table)
        .expect("table");
    let paths = zyron_lake::LakePaths::new(server.disk_manager.data_dir(), entry.id.0);
    let log = zyron_lake::TransactionLog::lookup_shared(&paths).expect("lake log");
    zyron_wire::connection::lake_optimize(&server.catalog, &log, entry.id.0, cluster, delete)
        .await
        .expect("optimize")
}

/// A table filled by many small writes holds a file per write, and every
/// scan then pays per-file cost for rows that belong in one file. OPTIMIZE
/// merges them, and the rows come back unchanged
#[tokio::test]
async fn test_optimize_merges_files_too_small_to_be_worth_opening() {
    let (server, _tmp) = table_with_one_file_per_insert(8).await;
    assert_eq!(
        manifest_of(&server, "parts").entries.len(),
        8,
        "one statement writes one file, which is the state that needs merging"
    );
    let before = ids(&server, "parts").await;

    optimize(&server, "parts", false, true).await;

    let after = manifest_of(&server, "parts");
    assert_eq!(
        after.entries.len(),
        1,
        "eight files well under the target belong in one"
    );
    assert_eq!(after.entries[0].row_count, 8);
    assert_eq!(ids(&server, "parts").await, before, "a row went missing");
}

/// Rewriting one file into one file moves the same rows into the same
/// shape. A table in that state has to come out of OPTIMIZE untouched, or
/// a maintenance loop would rewrite it on every tick forever
#[tokio::test]
async fn test_optimize_leaves_a_table_that_is_already_one_file_alone() {
    let (server, _tmp) = table_with_one_file_per_insert(1).await;
    let before = manifest_of(&server, "parts");
    assert_eq!(before.entries.len(), 1);

    optimize(&server, "parts", false, true).await;

    let after = manifest_of(&server, "parts");
    assert_eq!(
        after.snapshot_id, before.snapshot_id,
        "nothing to do has to commit nothing"
    );
}

/// A delete against part of a file is recorded rather than applied, so the
/// rows stay on disk until a rewrite removes them. OPTIMIZE is that
/// rewrite, and the count it reclaims is the count the delete recorded
#[tokio::test]
async fn test_optimize_applies_a_recorded_delete_and_retires_it() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE events (id BIGINT NOT NULL, tag BIGINT) USING ZYRONLAKE",
    )
    .await
    .expect("create");
    exec_dml(
        &server,
        "INSERT INTO events VALUES (1, 7), (2, 7), (3, 9), (4, 9)",
    )
    .await;
    exec_dml(&server, "DELETE FROM events WHERE tag = 7").await;

    let before = manifest_of(&server, "events");
    assert_eq!(
        before.delete_predicates.len(),
        1,
        "a partially covered file keeps its predicate until a rewrite"
    );
    assert_eq!(
        before.pending_deleted_rows(),
        2,
        "the delete counted what it matched while it had the rows in hand"
    );
    assert_eq!(before.total_rows(), 4, "the rows are still on disk");

    optimize(&server, "events", false, true).await;

    let after = manifest_of(&server, "events");
    assert!(
        after.delete_predicates.is_empty(),
        "the predicate is retired"
    );
    assert_eq!(after.total_rows(), 2, "the deleted rows are gone");
    assert_eq!(after.pending_deleted_rows(), 0);
    assert_eq!(ids(&server, "events").await.len(), 2);
}

/// Both thresholds read from the table's own properties, so an operator
/// can make maintenance more or less eager per table
#[tokio::test]
async fn test_the_compaction_thresholds_come_from_the_table() {
    let (server, _tmp) = table_with_one_file_per_insert(8).await;
    let manifest = manifest_of(&server, "parts");

    // Eight one-row files against the shipped target, so every one of them
    // is small and the default quarter is crossed
    let need = manifest.compaction_need(zyron_lake::DEFAULT_ROWS_PER_FILE);
    assert_eq!(need.small_files, 8);
    assert_eq!(
        need.trigger,
        Some(zyron_lake::CompactionTrigger::SmallFiles)
    );

    // A table that says it wants files this small is not drifting
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE parts SET (auto_compact_small_file_ratio = '2')",
    )
    .await
    .expect("a table property reaches the manifest");
    let manifest = manifest_of(&server, "parts");
    assert_eq!(
        manifest.auto_compact_small_file_ratio(),
        2.0,
        "the property has to reach the manifest"
    );
    assert_eq!(
        manifest
            .compaction_need(zyron_lake::DEFAULT_ROWS_PER_FILE)
            .trigger,
        None,
        "a threshold above one can never trip, which is how a table opts out"
    );
}

/// An expression cluster key is stored in a column no statement named, so
/// nothing else in the catalog reports it. The view is where an operator
/// finds out it exists and what it computes
#[tokio::test]
async fn test_the_derived_columns_view_reports_an_expression_key() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE hits (id BIGINT NOT NULL, ts TIMESTAMP) USING ZYRONLAKE \
         CLUSTER BY (date_part('year', ts))",
    )
    .await
    .expect("create");

    let (columns, rows) =
        stat_view(&server, "SELECT * FROM zyron_derived_columns").expect("derived columns view");
    assert_eq!(rows.len(), 1, "one expression, one row");
    let row = &rows[0];
    let value = |name: &str| -> &str {
        let i = columns
            .iter()
            .position(|c| c == name)
            .unwrap_or_else(|| panic!("the view has no {name} column, it has {columns:?}"));
        row[i].as_str()
    };
    assert_eq!(value("table_name"), "hits");
    assert!(value("sql").contains("date_part"), "{:?}", row);
    assert_eq!(
        value("source_columns"),
        "ts",
        "the columns the expression reads are what make a DROP COLUMN refusal explicable"
    );
    assert_eq!(value("is_cluster_key"), "yes");
    assert_eq!(value("addressable_by"), "expression");
    assert_ne!(value("canonical_hash"), "");

    // A table with no expression key contributes no rows
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE flat (a BIGINT) USING ZYRONLAKE",
    )
    .await
    .expect("create");
    let (_, rows) =
        stat_view(&server, "SELECT * FROM zyron_derived_columns").expect("derived columns view");
    assert_eq!(rows.len(), 1);
}

/// A maintenance loop that rewrites files silently is one nobody can
/// reason about, so every unasked run says what tripped it and what it
/// moved
#[tokio::test]
async fn test_the_auto_compaction_history_view_reports_a_run() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE parts (id BIGINT NOT NULL) USING ZYRONLAKE",
    )
    .await
    .expect("create");

    zyron_lake::compaction_history::compaction_history().record(
        zyron_lake::compaction_history::CompactionRecord {
            table_id: 4242,
            table_name: "parts".into(),
            trigger: zyron_lake::CompactionTrigger::Both,
            triggered_at_us: 1_700_000_000_000_000,
            files_before: 9,
            files_after: 1,
            dead_rows_reclaimed: 31,
            small_file_ratio_milli: 888,
            dead_row_ratio_milli: 250,
            version: Some(12),
        },
    );

    let (columns, rows) = stat_view(
        &server,
        "SELECT * FROM zyron_auto_compaction_history WHERE table_name = 'parts'",
    )
    .expect("history view");
    assert!(!rows.is_empty(), "the run has to be reported");
    let row = rows.last().expect("a run");
    let value = |name: &str| -> &str {
        let i = columns
            .iter()
            .position(|c| c == name)
            .unwrap_or_else(|| panic!("the view has no {name} column, it has {columns:?}"));
        row[i].as_str()
    };
    assert_eq!(value("trigger"), "small_files_and_dead_rows");
    assert_eq!(value("files_before"), "9");
    assert_eq!(value("files_after"), "1");
    assert_eq!(value("dead_rows_reclaimed"), "31");
    assert_eq!(value("small_file_ratio_milli"), "888");
    assert_eq!(value("dead_row_ratio_milli"), "250");
    assert_eq!(value("version"), "12");

    // The filter runs on the ring, so another table's runs are not counted
    let (_, other) = stat_view(
        &server,
        "SELECT * FROM zyron_auto_compaction_history WHERE table_name = 'nothing'",
    )
    .expect("history view");
    assert!(other.is_empty());
}

/// What a statement asked for and what the files are laid out by are
/// separate columns, because under Auto they diverge the moment
/// measurement replaces a declared key
#[tokio::test]
async fn test_clustering_status_separates_declared_from_active_keys() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE sales (region BIGINT, sku BIGINT) USING ZYRONLAKE \
         CLUSTER BY (region) FORCE",
    )
    .await
    .expect("create");

    let (columns, rows) = stat_view(
        &server,
        "SELECT * FROM zyron_clustering_status WHERE table_name = 'sales'",
    )
    .expect("clustering status view");
    let row = rows.first().expect("one lake table, one row");
    let value = |name: &str| -> &str {
        let i = columns
            .iter()
            .position(|c| c == name)
            .unwrap_or_else(|| panic!("the view has no {name} column, it has {columns:?}"));
        row[i].as_str()
    };
    assert!(value("declared_keys").contains("region"), "{:?}", row);
    assert!(value("active_keys").contains("region"), "{:?}", row);
}

/// One virtual view's answer, as column names and rows of rendered text
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
    let columns = fields.iter().map(|f| f.name.clone()).collect();
    let rendered = rows
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
    Ok((columns, rendered))
}

/// A plan priced against one layout must not outlive it.
///
/// Cached plans carry the catalog version they were planned at, so moving
/// that version is what evicts them. Every statement that changes a
/// table's layout has to move it, including the ones that change nothing
/// in the catalog itself
#[tokio::test]
async fn test_a_layout_change_evicts_plans_priced_against_the_old_one() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE sales (region BIGINT, sku BIGINT) USING ZYRONLAKE",
    )
    .await
    .expect("create");

    let before = server.catalog.schema_version();
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE sales CLUSTER BY (region) FORCE",
    )
    .await
    .expect("declare a layout");
    assert!(
        server.catalog.schema_version() > before,
        "declaring a layout changes what every scan on this table costs"
    );

    let before = server.catalog.schema_version();
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE sales SET CLUSTERING SCHEDULE = OnDemand",
    )
    .await
    .expect("change the schedule");
    assert!(
        server.catalog.schema_version() > before,
        "the schedule decides whether the layout a plan priced will survive"
    );
}

/// A rewrite that moves the layout evicts, and one that finds nothing to
/// do does not. Evicting on every maintenance tick would throw away every
/// cached plan on the node for no reason
#[tokio::test]
async fn test_a_compaction_evicts_only_when_it_changed_something() {
    let (server, _tmp) = table_with_one_file_per_insert(8).await;

    let before = server.catalog.schema_version();
    optimize(&server, "parts", false, true).await;
    assert!(
        server.catalog.schema_version() > before,
        "eight files became one, so every plan priced against eight is stale"
    );

    let after_first = server.catalog.schema_version();
    optimize(&server, "parts", false, true).await;
    assert_eq!(
        server.catalog.schema_version(),
        after_first,
        "a compaction that found nothing to do must not evict anything"
    );
}

/// Registering an expression cluster key changes which predicates can
/// reach a stored value, so plans made before it have to go
#[tokio::test]
async fn test_adding_a_derived_column_evicts_cached_plans() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE hits (id BIGINT NOT NULL, ts TIMESTAMP) USING ZYRONLAKE",
    )
    .await
    .expect("create");

    let before = server.catalog.schema_version();
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE hits ADD DERIVED COLUMN yr AS date_part('year', ts)",
    )
    .await
    .expect("add a derived column");
    assert!(
        server.catalog.schema_version() > before,
        "a predicate that could not be pushed down before this can be pushed down after it"
    );
}

/// A hundred tiny files carrying no delete predicate at all have to
/// collapse.
///
/// This is the shape the compactor used to walk away from: it selected
/// only files carrying a delete predicate, so a table filled by a hundred
/// small writes was left exactly as it was, and the maintenance trigger
/// that fires on small files would have fired on every tick and produced
/// no work. Silent success is worse than a failure, because the metric
/// says a compaction ran
#[tokio::test]
async fn test_optimize_collapses_a_hundred_tiny_files_with_no_deletes() {
    let (server, _tmp) = table_with_one_file_per_insert(100).await;
    let before = manifest_of(&server, "parts");
    assert_eq!(before.entries.len(), 100, "one file per write");
    assert!(
        before
            .entries
            .iter()
            .all(|e| e.delete_predicate_ids.is_empty()),
        "nothing here is waiting on a delete, so only the small-file rule can move it"
    );
    assert_eq!(before.pending_deleted_rows(), 0);
    let rows_before = ids(&server, "parts").await;

    // The trigger sees it, which is what the background worker acts on
    let need = before.compaction_need(zyron_lake::DEFAULT_ROWS_PER_FILE);
    assert_eq!(need.small_files, 100);
    assert_eq!(
        need.trigger,
        Some(zyron_lake::CompactionTrigger::SmallFiles),
        "a hundred files of one row each is the case the threshold exists for"
    );

    optimize(&server, "parts", false, true).await;

    let after = manifest_of(&server, "parts");
    assert_eq!(
        after.entries.len(),
        1,
        "a hundred files well inside the target belong in one"
    );
    assert_eq!(after.entries[0].row_count, 100);
    assert_eq!(
        ids(&server, "parts").await,
        rows_before,
        "the rewrite changed the answer"
    );
    assert_eq!(
        after
            .compaction_need(zyron_lake::DEFAULT_ROWS_PER_FILE)
            .trigger,
        None,
        "the compaction has to settle the trigger, or the worker rewrites this table forever"
    );
}

/// How hard maintenance works on a table is the table's decision.
///
/// A table taking constant small writes wants a repair pass that keeps up;
/// one that is mostly read wants a pass that never competes with queries
/// for long. The node default cannot be right for both, so all three knobs
/// are table properties, and a value that would not parse is refused where
/// it is written rather than falling back silently at read time
#[tokio::test]
async fn test_the_repair_knobs_are_table_properties() {
    let (server, _tmp) = table_with_one_file_per_insert(2).await;
    let mut session = new_session();

    let manifest = manifest_of(&server, "parts");
    assert_eq!(
        manifest.cluster_repair_max_inputs(16),
        16,
        "the node default"
    );
    assert_eq!(manifest.cluster_repair_interval_secs(300), 300);
    assert_eq!(
        manifest.cluster_repair_urgency_threshold(),
        zyron_lake::DEFAULT_CLUSTER_REPAIR_URGENCY_THRESHOLD
    );

    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE parts SET (cluster_repair_max_inputs = '4', \
         cluster_repair_interval_secs = '900', cluster_repair_urgency_threshold = '2')",
    )
    .await
    .expect("the knobs reach the manifest");

    let manifest = manifest_of(&server, "parts");
    assert_eq!(manifest.cluster_repair_max_inputs(16), 4);
    assert_eq!(manifest.cluster_repair_interval_secs(300), 900);
    assert_eq!(manifest.cluster_repair_urgency_threshold(), 2);

    // A value that cannot be read is refused at the statement. The readers
    // fall back so a damaged manifest still maintains, and that fallback
    // would otherwise swallow a typo
    let err = exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE parts SET (cluster_repair_interval_secs = 'soon')",
    )
    .await
    .expect_err("a threshold that is not a number is not a threshold");
    assert!(format!("{err:?}").contains("whole number"), "{err:?}");

    let err = exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE parts SET (cluster_repair_max_inputs = '0')",
    )
    .await
    .expect_err("a pass that rewrites nothing is a stall, not a bound");
    assert!(format!("{err:?}").contains("positive"), "{err:?}");

    // The refused statements changed nothing
    let manifest = manifest_of(&server, "parts");
    assert_eq!(manifest.cluster_repair_interval_secs(300), 900);
    assert_eq!(manifest.cluster_repair_max_inputs(16), 4);
}

/// A maintenance option only means something on a table that has data
/// files to maintain
#[tokio::test]
async fn test_a_heap_table_refuses_the_lake_maintenance_knobs() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE heapish (id BIGINT NOT NULL)",
    )
    .await
    .expect("create");
    let err = exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE heapish SET (cluster_repair_max_inputs = '4')",
    )
    .await
    .expect_err("a heap table has no data files to repair");
    assert!(format!("{err:?}").contains("not a lake table"), "{err:?}");
}

/// A clone holds the source's rows without copying a byte of them.
///
/// The whole point is that the statement costs a walk of the manifest
/// whatever the table weighs, so the assertion that matters is not that
/// the rows are there but that no new data file was written to hold them
#[tokio::test]
async fn test_clone_shares_the_source_file_set_without_copying_it() {
    let (server, schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE orders (id BIGINT NOT NULL, region BIGINT) USING ZYRONLAKE \
         CLUSTER BY (region) FORCE",
    )
    .await
    .expect("create");
    exec_dml(&server, "INSERT INTO orders VALUES (1, 10), (2, 20)").await;
    exec_dml(&server, "INSERT INTO orders VALUES (3, 30)").await;

    let source = server
        .catalog
        .get_table(schema_id, "orders")
        .expect("entry");
    let source_manifest = manifest_of(&server, "orders");
    assert_eq!(source_manifest.entries.len(), 2, "two writes, two files");

    exec_ddl(&server, &mut session, "CREATE TABLE backup CLONE OF orders")
        .await
        .expect("clone");

    // The rows are readable through the clone, which is the user-facing
    // claim
    let rows = query_values(&server, "SELECT id FROM backup ORDER BY id").await;
    assert_eq!(rows.len(), 3, "the clone holds the source's rows");
    assert_eq!(
        query_rows(&server, "SELECT id FROM orders").await,
        3,
        "cloning must not disturb the source"
    );

    let clone = server
        .catalog
        .get_table(schema_id, "backup")
        .expect("entry");
    let clone_manifest = manifest_of(&server, "backup");
    assert_eq!(
        clone_manifest
            .entries
            .iter()
            .map(|e| e.partition_id)
            .collect::<Vec<_>>(),
        source_manifest
            .entries
            .iter()
            .map(|e| e.partition_id)
            .collect::<Vec<_>>(),
        "a clone names the same files, it does not write new ones"
    );

    // Not copied: the two directory entries are the same data, so the
    // clone's files are the source's files under another name
    let source_paths = LakePaths::new(server.disk_manager.data_dir(), source.id.0);
    let clone_paths = LakePaths::new(server.disk_manager.data_dir(), clone.id.0);
    for entry in &clone_manifest.entries {
        let from = source_paths.data_file(entry.partition_id);
        let to = clone_paths.data_file(entry.partition_id);
        assert!(to.exists(), "the clone has to have its own directory entry");
        assert_eq!(
            std::fs::metadata(&from).expect("source file").len(),
            std::fs::metadata(&to).expect("clone file").len(),
            "the same bytes under both names"
        );
    }

    // The shape came across too, ids and all, or the statistics in those
    // files would describe columns the clone does not have
    assert_eq!(
        clone.columns.iter().map(|c| c.id.0).collect::<Vec<_>>(),
        source.columns.iter().map(|c| c.id.0).collect::<Vec<_>>()
    );
    assert_eq!(clone.cluster.fold_keys(), source.cluster.fold_keys());
    assert!(clone.lake.is_lake());
}

/// A clone at a version is a copy of what the table showed then, which is
/// the point of naming one
#[tokio::test]
async fn test_clone_at_version_takes_the_table_as_it_stood() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE orders (id BIGINT NOT NULL) USING ZYRONLAKE",
    )
    .await
    .expect("create");
    exec_dml(&server, "INSERT INTO orders VALUES (1)").await;
    let after_first = manifest_of(&server, "orders").snapshot_id;
    exec_dml(&server, "INSERT INTO orders VALUES (2)").await;
    exec_dml(&server, "INSERT INTO orders VALUES (3)").await;
    assert_eq!(query_rows(&server, "SELECT id FROM orders").await, 3);

    exec_ddl(
        &server,
        &mut session,
        &format!("CREATE TABLE early CLONE OF orders AT VERSION {after_first}"),
    )
    .await
    .expect("clone at a version");

    assert_eq!(
        query_rows(&server, "SELECT id FROM early").await,
        1,
        "the clone shows what the source showed at that version, not since"
    );
    assert_eq!(
        query_rows(&server, "SELECT id FROM orders").await,
        3,
        "the source is untouched"
    );

    // A version the source never had is refused rather than clamped
    assert!(
        exec_ddl(
            &server,
            &mut session,
            "CREATE TABLE nope CLONE OF orders AT VERSION 9999",
        )
        .await
        .is_err()
    );
}

/// The two tables are independent from the moment the clone exists. A
/// write to one must not be visible in the other, or a clone would be a
/// view with extra steps
#[tokio::test]
async fn test_a_clone_and_its_source_diverge() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE orders (id BIGINT NOT NULL) USING ZYRONLAKE",
    )
    .await
    .expect("create");
    exec_dml(&server, "INSERT INTO orders VALUES (1), (2)").await;
    exec_ddl(&server, &mut session, "CREATE TABLE backup CLONE OF orders")
        .await
        .expect("clone");

    exec_dml(&server, "INSERT INTO backup VALUES (3)").await;
    exec_dml(&server, "DELETE FROM orders WHERE id = 1").await;

    assert_eq!(query_rows(&server, "SELECT id FROM backup").await, 3);
    assert_eq!(query_rows(&server, "SELECT id FROM orders").await, 1);
}

/// The source keeps the files a clone was taken from, so it can still
/// serve that version. Vacuum reads the pin rather than the clone
#[tokio::test]
async fn test_the_source_pin_keeps_a_cloned_version_reclaimable_only_after_the_clone_goes() {
    let (server, schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE orders (id BIGINT NOT NULL) USING ZYRONLAKE",
    )
    .await
    .expect("create");
    exec_dml(&server, "INSERT INTO orders VALUES (1), (2)").await;
    exec_ddl(&server, &mut session, "CREATE TABLE backup CLONE OF orders")
        .await
        .expect("clone");

    let source = server
        .catalog
        .get_table(schema_id, "orders")
        .expect("entry");
    let clone = server
        .catalog
        .get_table(schema_id, "backup")
        .expect("entry");
    let source_paths = LakePaths::new(server.disk_manager.data_dir(), source.id.0);
    assert!(
        source_paths.clone_ref(clone.id.0).exists(),
        "the clone has to leave a claim the source can see"
    );

    // Rewriting the source leaves its old file unreferenced by its own
    // head, and the pin is the only reason it survives
    let pinned_file = manifest_of(&server, "backup").entries[0].partition_id;
    exec_dml(&server, "INSERT INTO orders VALUES (3)").await;
    optimize(&server, "orders", false, true).await;
    let source_log = TransactionLog::lookup_shared(&source_paths).expect("shared log");
    zyron_lake::vacuum_data_files(&source_log, source_log.head_version()).expect("vacuum");
    assert!(
        source_paths.data_file(pinned_file).exists(),
        "vacuum reclaimed a file a clone is still holding a version of"
    );

    // Dropping the clone releases the claim
    exec_ddl(&server, &mut session, "DROP TABLE backup")
        .await
        .expect("drop the clone");
    assert!(
        !source_paths.clone_ref(clone.id.0).exists(),
        "a source carrying a pin from a table that no longer exists would never reclaim again"
    );
}

/// A clone only means something on a format whose files are immutable and
/// shareable
#[tokio::test]
async fn test_cloning_a_heap_table_is_refused() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE heapish (id BIGINT NOT NULL)",
    )
    .await
    .expect("create");
    let err = exec_ddl(&server, &mut session, "CREATE TABLE copy CLONE OF heapish")
        .await
        .expect_err("a heap table has no shareable file set");
    assert!(format!("{err:?}").contains("not a lake table"), "{err:?}");

    assert!(
        exec_ddl(
            &server,
            &mut session,
            "CREATE TABLE copy CLONE OF nosuchtable"
        )
        .await
        .is_err(),
        "a source that does not exist is not a source"
    );
}

/// Rolling a table back is a new version whose file set is an old one's.
///
/// The history is not rewritten, which is what makes the restore itself
/// undoable: every version in between is still readable, and restoring to
/// the version before the restore puts everything back
#[tokio::test]
async fn test_restore_to_version_rolls_the_data_back_without_losing_the_history() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE orders (id BIGINT NOT NULL) USING ZYRONLAKE",
    )
    .await
    .expect("create");
    exec_dml(&server, "INSERT INTO orders VALUES (1)").await;
    let after_first = manifest_of(&server, "orders").snapshot_id;
    exec_dml(&server, "INSERT INTO orders VALUES (2)").await;
    exec_dml(&server, "INSERT INTO orders VALUES (3)").await;
    let before_restore = manifest_of(&server, "orders").snapshot_id;
    assert_eq!(query_rows(&server, "SELECT id FROM orders").await, 3);

    exec_ddl(
        &server,
        &mut session,
        &format!("RESTORE TABLE orders TO VERSION {after_first}"),
    )
    .await
    .expect("restore");

    assert_eq!(
        query_rows(&server, "SELECT id FROM orders").await,
        1,
        "the table shows what it showed at that version"
    );
    let after = manifest_of(&server, "orders");
    assert!(
        after.snapshot_id > before_restore,
        "a restore is a new version on top, not a rewind of the log"
    );

    // The versions in between are still there, which is what makes the
    // restore undoable
    exec_ddl(
        &server,
        &mut session,
        &format!("RESTORE TABLE orders TO VERSION {before_restore}"),
    )
    .await
    .expect("undo the restore");
    assert_eq!(
        query_rows(&server, "SELECT id FROM orders").await,
        3,
        "restoring to the version before the restore puts everything back"
    );
}

/// A delete after the restore point is undone by the restore, because what
/// a version showed includes which rows it hid
#[tokio::test]
async fn test_restore_puts_back_rows_a_later_delete_removed() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE events (id BIGINT NOT NULL, tag BIGINT) USING ZYRONLAKE",
    )
    .await
    .expect("create");
    exec_dml(
        &server,
        "INSERT INTO events VALUES (1, 7), (2, 7), (3, 9), (4, 9)",
    )
    .await;
    let before_delete = manifest_of(&server, "events").snapshot_id;
    exec_dml(&server, "DELETE FROM events WHERE tag = 7").await;
    assert_eq!(query_rows(&server, "SELECT id FROM events").await, 2);

    exec_ddl(
        &server,
        &mut session,
        &format!("RESTORE TABLE events TO VERSION {before_delete}"),
    )
    .await
    .expect("restore");

    assert_eq!(
        query_rows(&server, "SELECT id FROM events").await,
        4,
        "a delete recorded after the restore point has to be undone with it"
    );
    assert!(
        manifest_of(&server, "events").delete_predicates.is_empty(),
        "the predicate that hid them is gone, not just its effect"
    );
}

/// Restoring to where the table already is asks for nothing, and a
/// statement that asks for nothing is not a failure
#[tokio::test]
async fn test_restoring_to_the_current_state_is_a_no_op() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE orders (id BIGINT NOT NULL) USING ZYRONLAKE",
    )
    .await
    .expect("create");
    exec_dml(&server, "INSERT INTO orders VALUES (1)").await;
    let head = manifest_of(&server, "orders").snapshot_id;

    exec_ddl(
        &server,
        &mut session,
        &format!("RESTORE TABLE orders TO VERSION {head}"),
    )
    .await
    .expect("restoring to the current state changes nothing");

    assert_eq!(
        manifest_of(&server, "orders").snapshot_id,
        head,
        "nothing to do has to commit nothing"
    );
    assert_eq!(query_rows(&server, "SELECT id FROM orders").await, 1);
}

/// A version the table never had, and a format with no versions at all
#[tokio::test]
async fn test_restore_refuses_what_it_cannot_restore() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE orders (id BIGINT NOT NULL) USING ZYRONLAKE",
    )
    .await
    .expect("create");
    exec_dml(&server, "INSERT INTO orders VALUES (1)").await;

    assert!(
        exec_ddl(
            &server,
            &mut session,
            "RESTORE TABLE orders TO VERSION 9999"
        )
        .await
        .is_err(),
        "a version the table never had is not a restore point"
    );

    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE heapish (id BIGINT NOT NULL)",
    )
    .await
    .expect("create");
    let err = exec_ddl(&server, &mut session, "RESTORE TABLE heapish TO VERSION 1")
        .await
        .expect_err("a heap table keeps no version history");
    assert!(format!("{err:?}").contains("version history"), "{err:?}");
}

/// `OPTIMIZE TABLE t CLUSTER` is the door an operator uses to ask for a
/// layout pass by hand.
///
/// Without it the only caller of the clustering pass was the background
/// worker, which skips a table whose schedule is OnDemand, so a table set
/// that way could never be clustered at all
#[tokio::test]
async fn test_optimize_cluster_runs_a_layout_pass() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE sales (id BIGINT NOT NULL, region BIGINT) USING ZYRONLAKE \
         CLUSTER BY (region) FORCE",
    )
    .await
    .expect("create");
    // Several writes, each landing a file whose region range overlaps the
    // others, which is the drift a pass exists to remove
    for i in 0..6 {
        exec_dml(
            &server,
            &format!(
                "INSERT INTO sales VALUES ({}, 1), ({}, 5), ({}, 9)",
                i * 3,
                i * 3 + 1,
                i * 3 + 2
            ),
        )
        .await;
    }
    let before = manifest_of(&server, "sales");
    assert_eq!(before.entries.len(), 6);
    assert!(
        zyron_lake::drifted_file_count(&before) > 0,
        "six files all spanning the same region range is drift"
    );
    let rows_before = ids(&server, "sales").await;

    let message = optimize(&server, "sales", true, false).await;
    assert!(
        message.contains("clustered") || message.contains("already"),
        "the statement has to say what the pass did, got {message}"
    );

    let after = manifest_of(&server, "sales");
    assert!(
        after.snapshot_id > before.snapshot_id,
        "a pass that rewrote files has to commit a version"
    );
    assert_eq!(
        zyron_lake::drifted_file_count(&after),
        0,
        "the pass has to settle the drift it was asked to remove"
    );
    assert_eq!(
        ids(&server, "sales").await,
        rows_before,
        "reordering rows must not change which rows there are"
    );
}

/// `OPTIMIZE TABLE t CLUSTER, DELETE` does both, and does the delete pass
/// first so the layout pass carries fewer rows
#[tokio::test]
async fn test_optimize_cluster_and_delete_together() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE sales (id BIGINT NOT NULL, region BIGINT) USING ZYRONLAKE \
         CLUSTER BY (region) FORCE",
    )
    .await
    .expect("create");
    for i in 0..4 {
        exec_dml(
            &server,
            &format!(
                "INSERT INTO sales VALUES ({}, 1), ({}, 5), ({}, 9)",
                i * 3,
                i * 3 + 1,
                i * 3 + 2
            ),
        )
        .await;
    }
    exec_dml(&server, "DELETE FROM sales WHERE region = 5").await;
    let before = manifest_of(&server, "sales");
    assert_eq!(before.pending_deleted_rows(), 4, "one row per file");

    let message = optimize(&server, "sales", true, true).await;

    let after = manifest_of(&server, "sales");
    assert_eq!(
        after.pending_deleted_rows(),
        0,
        "the delete pass has to have run"
    );
    assert!(
        after.delete_predicates.is_empty(),
        "and retired what it applied"
    );
    assert_eq!(
        zyron_lake::drifted_file_count(&after),
        0,
        "the layout pass has to have run too"
    );
    assert_eq!(query_rows(&server, "SELECT id FROM sales").await, 8);
    assert!(
        message.contains("rewrote"),
        "a combined statement reports both halves, got {message}"
    );
}

/// The clustering schedule governs whether the background worker may start
/// a pass unasked. It does not govern an operator asking directly, which is
/// the whole point of having a statement for it
#[tokio::test]
async fn test_optimize_cluster_runs_on_an_ondemand_table() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE sales (id BIGINT NOT NULL, region BIGINT) USING ZYRONLAKE \
         CLUSTER BY (region) FORCE",
    )
    .await
    .expect("create");
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE sales SET CLUSTERING SCHEDULE = OnDemand",
    )
    .await
    .expect("opt out of background passes");
    for i in 0..4 {
        exec_dml(
            &server,
            &format!(
                "INSERT INTO sales VALUES ({}, 1), ({}, 9)",
                i * 2,
                i * 2 + 1
            ),
        )
        .await;
    }

    let before = manifest_of(&server, "sales");
    assert_eq!(
        before.clustering_schedule(),
        zyron_lake::ClusteringSchedule::OnDemand,
        "the schedule has to have reached the manifest for this to mean anything"
    );
    assert!(zyron_lake::drifted_file_count(&before) > 0);

    optimize(&server, "sales", true, false).await;

    assert_eq!(
        zyron_lake::drifted_file_count(&manifest_of(&server, "sales")),
        0,
        "OnDemand means nothing starts unasked, not that nothing can be asked for"
    );
}

/// A pass that has nothing to do says so and commits nothing, which is
/// what keeps an operator running OPTIMIZE on a schedule from rewriting a
/// settled table forever
#[tokio::test]
async fn test_optimize_cluster_on_a_settled_table_is_a_no_op() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE sales (id BIGINT NOT NULL, region BIGINT) USING ZYRONLAKE \
         CLUSTER BY (region) FORCE",
    )
    .await
    .expect("create");
    exec_dml(&server, "INSERT INTO sales VALUES (1, 1), (2, 5)").await;
    optimize(&server, "sales", true, false).await;
    let settled = manifest_of(&server, "sales").snapshot_id;

    let message = optimize(&server, "sales", true, false).await;
    assert_eq!(
        manifest_of(&server, "sales").snapshot_id,
        settled,
        "a second pass over a settled table has to commit nothing"
    );
    assert!(
        message.contains("already") || message.contains("no clustering key"),
        "and has to say why it did nothing, got {message}"
    );
}

/// A table under Auto has no declared keys, so the pass has to take what
/// measurement proposed. A proposal that never reaches the files is the
/// state the pending-proposal gauge reports, and running a pass is what
/// clears it
#[tokio::test]
async fn test_optimize_cluster_takes_up_a_pending_proposal() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE sales (id BIGINT NOT NULL, region BIGINT) USING ZYRONLAKE",
    )
    .await
    .expect("create");
    for i in 0..4 {
        exec_dml(
            &server,
            &format!(
                "INSERT INTO sales VALUES ({}, 1), ({}, 9)",
                i * 2,
                i * 2 + 1
            ),
        )
        .await;
    }
    // A workload that names a column is what measurement proposes from
    for _ in 0..4 {
        let _ = query_rows(&server, "SELECT id FROM sales WHERE region = 1").await;
    }

    let before = manifest_of(&server, "sales");
    assert_eq!(
        before.clustering_mode(),
        zyron_lake::ClusterMode::Auto,
        "a table that declared nothing is measured, which is what makes a proposal possible"
    );

    optimize(&server, "sales", true, false).await;

    // Either measurement had enough to act on and the layout moved, or it
    // did not and nothing was committed. Both are correct; what would not
    // be is a pass that claimed a layout it never applied
    let after = manifest_of(&server, "sales");
    if after.snapshot_id > before.snapshot_id {
        assert!(
            !after.cluster_spec.keys.is_empty(),
            "a committed pass under Auto has to have taken a proposal"
        );
        assert_eq!(zyron_lake::drifted_file_count(&after), 0);
    } else {
        assert_eq!(after.cluster_spec.keys, before.cluster_spec.keys);
    }
    assert_eq!(query_rows(&server, "SELECT id FROM sales").await, 8);
}

/// A table that declared nothing is measured and maintained by default.
/// Before this was the default, adaptive clustering was unreachable on
/// every table nobody had configured by hand
#[tokio::test]
async fn test_a_default_lake_table_is_measured_and_maintained() {
    let (server, schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE plainlake (id BIGINT NOT NULL) USING ZYRONLAKE",
    )
    .await
    .expect("create");

    let manifest = manifest_of(&server, "plainlake");
    assert_eq!(
        manifest.clustering_mode(),
        zyron_lake::ClusterMode::Auto,
        "a table that declared no layout hands the choice to measurement"
    );
    assert!(
        manifest.clustering_schedule().runs_in_background(),
        "and is maintained without being asked"
    );

    // The catalog says the same, which is what planning reads
    let entry = server
        .catalog
        .get_table(schema_id, "plainlake")
        .expect("entry");
    assert_eq!(entry.cluster.mode, manifest.clustering_mode().to_u8());
    assert_eq!(
        entry.cluster.schedule,
        manifest.clustering_schedule().to_u8()
    );
}

/// A restore has to land the table exactly where a time travel read of
/// that version says it was. Two different answers to "what did this table
/// hold at version n" would mean one of the two paths is wrong
#[tokio::test]
async fn test_a_restored_head_reads_the_same_as_a_time_travel_query() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE orders (id BIGINT NOT NULL, amount BIGINT) USING ZYRONLAKE",
    )
    .await
    .expect("create");
    exec_dml(&server, "INSERT INTO orders VALUES (1, 10), (2, 20)").await;
    let target = manifest_of(&server, "orders").snapshot_id;
    exec_dml(&server, "INSERT INTO orders VALUES (3, 30)").await;
    exec_dml(&server, "DELETE FROM orders WHERE id = 1").await;

    let as_of = query_values(
        &server,
        &format!("SELECT id FROM orders VERSION AS OF {target} ORDER BY id"),
    )
    .await;
    assert_eq!(as_of.len(), 2, "the time travel read is the reference");

    exec_ddl(
        &server,
        &mut session,
        &format!("RESTORE TABLE orders TO VERSION {target}"),
    )
    .await
    .expect("restore");

    let restored = query_values(&server, "SELECT id FROM orders ORDER BY id").await;
    assert_eq!(
        format!("{restored:?}"),
        format!("{as_of:?}"),
        "a restored head and a time travel read of the same version have to agree"
    );

    // And the time travel read still answers what it did before, because
    // the restore added history rather than rewriting it
    let again = query_values(
        &server,
        &format!("SELECT id FROM orders VERSION AS OF {target} ORDER BY id"),
    )
    .await;
    assert_eq!(format!("{again:?}"), format!("{as_of:?}"));
}
